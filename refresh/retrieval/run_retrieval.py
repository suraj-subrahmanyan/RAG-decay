"""
Run retrieval experiments using BM25, BGE, and Fusion methods.

BM25: sparse lexical retrieval via Pyserini
BGE: dense semantic retrieval with multilingual embeddings
Fusion: reciprocal rank fusion combining BM25 and BGE
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_queries(queries_file: Path) -> pd.DataFrame:
    """Load queries from parquet file."""
    df = pd.read_parquet(queries_file)
    logger.info(f"Loaded {len(df)} queries")
    return df


def retrieve_bm25(
    queries: pd.DataFrame,
    index_dir: Path,
    k: int = 50
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Retrieve using BM25.
    
    Returns:
        Dict mapping query_id -> [(doc_id, score), ...]
    """
    
    logger.info("Initializing BM25 searcher...")
    searcher = LuceneSearcher(str(index_dir))
    
    results = {}
    
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="BM25 retrieval"):
        query_id = str(row['query_id'])
        query_text = row['query_text']
        
        hits = searcher.search(query_text, k=k)
        
        results[query_id] = [
            (hit.docid, hit.score)
            for hit in hits
        ]
    
    logger.info(f"Retrieved for {len(results)} queries")
    return results


def retrieve_bge(
    queries: pd.DataFrame,
    index_dir: Path,
    model_name: str = "BAAI/bge-multilingual-gemma2",
    k: int = 50,
    device: str = "cuda"
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Retrieve using BGE dense embeddings.
    
    Returns:
        Dict mapping query_id -> [(doc_id, score), ...]
    """
    
    logger.info("Loading BGE model...")
    model = SentenceTransformer(
        model_name,
        model_kwargs={"dtype": "float16"},
        device=device
    )
    
    logger.info("Loading FAISS index...")
    index = faiss.read_index(str(index_dir / "index.faiss"))
    
    with open(index_dir / "docids.pkl", 'rb') as f:
        docids = pickle.load(f)
    
    logger.info(f"Index contains {index.ntotal:,} documents")
    
    results = {}
    
    # Encode queries in batches
    query_texts = queries['query_text'].tolist()
    query_ids = queries['query_id'].astype(str).tolist()
    
    logger.info(f"Encoding {len(query_texts)} queries...")
    query_embeddings = model.encode(
        query_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=4  # Small batch for 9B model
    )
    
    logger.info("Searching FAISS index...")
    scores, indices = index.search(query_embeddings.astype(np.float32), k)
    
    for query_id, score_list, idx_list in tqdm(
        zip(query_ids, scores, indices),
        total=len(query_ids),
        desc="BGE retrieval"
    ):
        results[query_id] = [
            (docids[idx], float(score))
            for idx, score in zip(idx_list, score_list)
            if idx != -1  # FAISS returns -1 for invalid indices
        ]
    
    logger.info(f"Retrieved for {len(results)} queries")
    return results


def reciprocal_rank_fusion(
    bm25_results: Dict[str, List[Tuple[str, float]]],
    bge_results: Dict[str, List[Tuple[str, float]]],
    k: int = 50,
    rrf_k: int = 60
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Combine BM25 and BGE using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank_i)) for each retriever
    """
    
    logger.info("Fusing BM25 and BGE results...")
    
    fused_results = {}
    
    for query_id in tqdm(bm25_results.keys(), desc="Fusion"):
        # Get ranked lists
        bm25_docs = [doc_id for doc_id, _ in bm25_results.get(query_id, [])]
        bge_docs = [doc_id for doc_id, _ in bge_results.get(query_id, [])]
        
        # Compute RRF scores
        scores = {}
        
        for rank, doc_id in enumerate(bm25_docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        
        for rank, doc_id in enumerate(bge_docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rrf_k + rank)
        
        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        fused_results[query_id] = ranked
    
    logger.info(f"Fused results for {len(fused_results)} queries")
    return fused_results


def save_results(
    results: Dict[str, List[Tuple[str, float]]],
    output_file: Path,
    method: str
) -> None:
    """Save retrieval results to JSONL."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id, ranked_docs in results.items():
            f.write(json.dumps({
                "query_id": query_id,
                "method": method,
                "results": [
                    {"doc_id": doc_id, "score": score, "rank": rank}
                    for rank, (doc_id, score) in enumerate(ranked_docs, start=1)
                ]
            }) + '\n')
    
    logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run retrieval on queries")
    parser.add_argument("--queries", required=True, help="Queries parquet file")
    parser.add_argument("--bm25_index", required=True, help="BM25 index directory")
    parser.add_argument("--bge_index", required=True, help="BGE index directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--version", required=True, help="Version (e.g., oct_2024)")
    parser.add_argument("--k", type=int, default=50, help="Number of docs to retrieve")
    parser.add_argument("--device", default="cuda", help="Device for BGE")
    args = parser.parse_args()
    
    queries_file = Path(args.queries)
    bm25_index_dir = Path(args.bm25_index)
    bge_index_dir = Path(args.bge_index)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info(f"RETRIEVAL: {args.version}")
    logger.info("="*80)
    logger.info(f"Queries: {queries_file}")
    logger.info(f"BM25 index: {bm25_index_dir}")
    logger.info(f"BGE index: {bge_index_dir}")
    logger.info(f"Top-k: {args.k}")
    
    # Load queries
    queries = load_queries(queries_file)
    
    # BM25 retrieval
    logger.info("\n" + "="*80)
    logger.info("BM25 RETRIEVAL")
    logger.info("="*80)
    bm25_results = retrieve_bm25(queries, bm25_index_dir, k=args.k)
    save_results(
        bm25_results,
        output_dir / f"bm25_{args.version}.jsonl",
        "bm25"
    )
    
    # BGE retrieval
    logger.info("\n" + "="*80)
    logger.info("BGE RETRIEVAL")
    logger.info("="*80)
    bge_results = retrieve_bge(queries, bge_index_dir, k=args.k, device=args.device)
    save_results(
        bge_results,
        output_dir / f"bge_{args.version}.jsonl",
        "bge"
    )
    
    # Fusion
    logger.info("\n" + "="*80)
    logger.info("FUSION (RRF)")
    logger.info("="*80)
    fusion_results = reciprocal_rank_fusion(bm25_results, bge_results, k=args.k)
    save_results(
        fusion_results,
        output_dir / f"fusion_{args.version}.jsonl",
        "fusion"
    )
    
    logger.info("\n" + "="*80)
    logger.info("Retrieval complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()

