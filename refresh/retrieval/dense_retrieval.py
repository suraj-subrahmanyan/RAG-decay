"""
Dense retrieval using BGE, E5, or Qwen embedding models with FAISS indices.

Supports multiple query formulations: question, answer, nuggets, closed_book_answer,
and subquestions. Applies E5-specific instruction prefix when necessary.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class QueryEncoder:
    """Encoder for query texts."""
    
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, model_kwargs={"dtype":"float16"}, device=device)
        logger.info(f"Model loaded on {device}")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode a single query text, applying E5 prefix if necessary."""
        if "e5" in self.model_name.lower():
            text = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {text}"
        document_embeddings = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return document_embeddings


def load_index_and_docids(index_dir: Path):
    """Load FAISS index and document IDs."""
    logger.info(f"Loading index from {index_dir}")
    
    index_file = index_dir / "index.faiss"
    docid_file = index_dir / "docids.pkl"
    
    index = faiss.read_index(str(index_file))
    
    with open(docid_file, 'rb') as f:
        doc_ids = pickle.load(f)
    
    logger.info(f"Index loaded ({index.ntotal} documents)")
    return index, doc_ids


def load_queries(queries_file: Path):
    """Load queries from JSONL file."""
    logger.info(f"Loading queries from {queries_file}")
    queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def get_query_text(query_data: dict, query_field: str) -> tuple:
    """
    Extract query text from specified field.
    
    Args:
        query_data: Query dictionary
        query_field: Field to use (question, answer, nuggets, closed_book_answer, subquestions)
        
    Returns:
        Tuple of (query_text, has_field)
    """
    if query_field == "question":
        text = query_data.get('question', '')
        return text, bool(text)
    
    elif query_field == "answer":
        text = query_data.get('answer', '')
        return text, bool(text)
    
    elif query_field == "nuggets":
        nuggets = query_data.get('nuggets', [])
        if not nuggets:
            return "", False
        text = "\n".join(nuggets)
        return text, True
    
    elif query_field == "closed_book_answer":
        text = query_data.get('closed_book_answer', '')
        return text, bool(text)
    
    elif query_field == "subquestions":
        subquestions = query_data.get('subquestions', [])
        if not subquestions:
            return "", False
        text = "\n".join(subquestions)
        return text, True
    
    else:
        raise ValueError(f"Unknown query field: {query_field}")


def retrieve_for_query_field(
    encoder: QueryEncoder,
    index: faiss.Index,
    doc_ids: list,
    query_data: dict,
    query_field: str,
    top_k: int,
    output_file
):
    """
    Retrieve documents for a single query using specified field.
    
    Args:
        encoder: Query encoder
        index: FAISS index
        doc_ids: List of document IDs
        query_data: Query dictionary
        query_field: Field to use for retrieval
        top_k: Number of results to retrieve
        output_file: Open file handle for writing results
        
    Returns:
        True if successful, False if skipped
    """
    query_id = query_data['query_id']
    answer_id = query_data.get('answer_id', '')
    
    # Get query text from specified field
    query_text, has_field = get_query_text(query_data, query_field)
    
    if not has_field:
        logger.warning(f"Query {query_id}: missing or empty field '{query_field}', skipping")
        return False
    
    try:
        query_embedding = encoder.encode(query_text)
        
        # Search
        # Reshape to (1, d) for FAISS
        query_embedding = query_embedding.reshape(1, -1)
        
        # Returns (Distances, Indices)
        D, I = index.search(query_embedding.astype(np.float32), top_k)
        
        # Iterate over results for the single query
        for rank, (doc_idx, score) in enumerate(zip(I[0], D[0]), 1):
            if doc_idx != -1 and doc_idx < len(doc_ids):  # Valid index
                result = {
                    'query_id': query_id,
                    'answer_id': answer_id,
                    'doc_id': doc_ids[doc_idx],
                    'score': float(score),
                    'rank': rank,
                    'query_field': query_field
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True
        
    except Exception as e:
        logger.error(f"Error retrieving for query {query_id} ({query_field}): {e}")
        return False


def process_corpus_version(
    config: dict,
    corpus_version: str,
    model: str,
    query_fields: list,
    encoder: QueryEncoder,
    queries: list
):
    """
    Process retrieval for a specific corpus version.
    
    Args:
        config: Configuration dictionary
        corpus_version: Corpus version string
        model: Model name (bge or e5)
        query_fields: List of query fields to process
        encoder: Loaded QueryEncoder
        queries: List of loaded queries
    """
    logger.info(f"{model.upper()} Retrieval - {corpus_version}")
    logger.info(f"Query fields: {', '.join(query_fields)}")
    
    # Setup paths
    index_dir = Path(config['index_base_dir']) / corpus_version / model
    output_dir = Path(config['output_dir']) / corpus_version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file_path = output_dir / f"{model}.jsonl"
    
    # Load index and doc IDs
    try:
        index, doc_ids = load_index_and_docids(index_dir)
    except Exception as e:
        logger.error(f"Failed to load index for {corpus_version}: {e}")
        return
    
    # Statistics
    stats = {field: {'success': 0, 'skipped': 0} for field in query_fields}
    
    # Retrieve for each query and each field
    logger.info(f"Retrieving top-{config['top_k']} documents...")
    
    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for query_data in tqdm(queries, desc=f"Processing queries ({corpus_version})"):
            for query_field in query_fields:
                success = retrieve_for_query_field(
                    encoder,
                    index,
                    doc_ids,
                    query_data,
                    query_field,
                    config['top_k'],
                    fout
                )
                
                if success:
                    stats[query_field]['success'] += 1
                else:
                    stats[query_field]['skipped'] += 1
    
    # Summary
    logger.info(f"RETRIEVAL SUMMARY ({corpus_version})")
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Query fields processed: {len(query_fields)}")
    
    for field in query_fields:
        success = stats[field]['success']
        skipped = stats[field]['skipped']
        logger.info(f"{field}:")
        logger.info(f"Success: {success}")
        if skipped > 0:
            logger.info(f"Skipped: {skipped}")
            
    logger.info(f"Results saved to: {output_file_path}")


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval with multiple query fields")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to retrieval configuration YAML"
    )
    parser.add_argument(
        "--indexing-config",
        type=str,
        default="config/indexing_config.yaml",
        help="Path to indexing configuration (for model details)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bge", "e5", "qwen"],
        help="Model to use (bge or e5)"
    )
    parser.add_argument(
        "--corpus-version",
        type=str,
        default="all",
        choices=["oct_2024", "oct_2025", "all"],
        help="Corpus version to search"
    )
    parser.add_argument(
        "--query-fields",
        type=str,
        nargs="+",
        default=["question", "answer", "nuggets", "closed_book_answer", "subquestions"],
        choices=["question", "answer", "nuggets", "closed_book_answer", "subquestions"],
        help="Query fields to use (default: all 5)"
    )
    
    args = parser.parse_args()
    
    # Load configs
    config = load_config(args.config)
    indexing_config = load_config(args.indexing_config)
    
    # Get model config
    model_config = indexing_config['dense_models'][args.model]
    
    # Load encoder (once for all versions)
    encoder = QueryEncoder(
        model_name=model_config['model_name'],
        device=config.get('device', 'cuda'),
        max_length=model_config['max_length']
    )
    
    # Load queries (once for all versions)
    queries = load_queries(Path(config['queries_file']))
    
    # Determine versions to process
    if args.corpus_version == "all":
        versions = ["oct_2024", "oct_2025"]
    else:
        versions = [args.corpus_version]
    
    for version in versions:
        process_corpus_version(
            config=config,
            corpus_version=version,
            model=args.model,
            query_fields=args.query_fields,
            encoder=encoder,
            queries=queries
        )


if __name__ == "__main__":
    main()