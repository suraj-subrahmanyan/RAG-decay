"""
Build dense BGE index using SentenceTransformer + FAISS.
Uses float16
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_corpus(corpus_file: Path) -> Tuple[List[str], List[str]]:
    """
    Load corpus from JSONL file.
    
    Args:
        corpus_file: Path to FreshStack corpus
        
    Returns:
        Tuple of (document_ids, texts)
    """
    
    doc_ids, texts = [], []
    errors = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            if not line.strip():
                continue
                
            try:
                doc = json.loads(line)
                
                # Validate required fields
                if '_id' not in doc or 'text' not in doc:
                    errors += 1
                    continue
                    
                doc_ids.append(doc['_id'])
                texts.append(doc['text'])
                
            except (json.JSONDecodeError, KeyError):
                errors += 1
                continue
    
    if errors > 0:
        logger.warning(f"Skipped {errors} invalid documents")
    logger.info(f"Loaded {len(doc_ids):,} documents")
    
    return doc_ids, texts


def encode_corpus(
    texts: List[str], 
    model: SentenceTransformer, 
    batch_size: int = 8
) -> np.ndarray:
    """
    Encode corpus in batches with progress tracking.
    
    Args:
        texts: List of document texts
        model: SentenceTransformer model
        batch_size: Encoding batch size
        
    Returns:
        NumPy array of embeddings (n_docs, embedding_dim)
    """
    
    logger.info(f"Encoding {len(texts):,} documents (batch_size={batch_size})...")
    
    all_embeddings = []
    
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            embeddings = model.encode(
                batch, 
                show_progress_bar=False, 
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for cosine similarity
            )
            all_embeddings.append(embeddings)
        
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"Embeddings shape: {embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        raise


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS flat index for exact similarity search.
    
    Args:
        embeddings: Normalized embedding vectors
        
    Returns:
        FAISS IndexFlatIP (inner product, equivalent to cosine for normalized vectors)
    """
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
    index.add(embeddings)
    
    logger.info(f"FAISS index built: {index.ntotal:,} vectors, {dim} dimensions")
    return index


def save_index(
    index: faiss.Index, 
    doc_ids: List[str], 
    output_dir: Path
) -> None:
    """
    Save FAISS index and document ID mapping.
    
    Args:
        index: FAISS index
        doc_ids: List of document IDs (order must match index)
        output_dir: Directory to save artifacts
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanity check
    assert len(doc_ids) == index.ntotal, "Doc ID count mismatch with index size"
    
    # Save FAISS index
    index_file = output_dir / "index.faiss"
    faiss.write_index(index, str(index_file))
    logger.info(f"Index saved: {index_file} ({index_file.stat().st_size / 1e6:.1f} MB)")
    
    # Save document ID mapping
    docid_file = output_dir / "docids.pkl"
    with open(docid_file, 'wb') as f:
        pickle.dump(doc_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Doc IDs saved: {docid_file}")


def main() -> bool:
    parser = argparse.ArgumentParser(description="Build BGE dense index")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL file")
    parser.add_argument("--output", required=True, help="Index output directory")
    parser.add_argument("--model", default="BAAI/bge-multilingual-gemma2", help="Model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Encoding batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()
    
    corpus_file = Path(args.corpus)
    output_dir = Path(args.output)
    
    if not corpus_file.exists():
        logger.error(f"Corpus not found: {corpus_file}")
        return False
    
    logger.info("="*60)
    logger.info("BGE DENSE INDEX CONSTRUCTION")
    logger.info("="*60)
    logger.info(f"Corpus: {corpus_file}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Load corpus
    doc_ids, texts = load_corpus(corpus_file)
    if not texts:
        logger.error("No documents found")
        return False
    
    # Load model
    logger.info("Loading model...")
    
    try:
        model = SentenceTransformer(
            args.model,
            model_kwargs={"dtype": "float16"},
            device=args.device
        )
        logger.info("Model loaded with float16 precision")
            
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Encode corpus
    try:
        embeddings = encode_corpus(texts, model, args.batch_size)
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        return False
    
    # Build FAISS index
    try:
        index = build_faiss_index(embeddings)
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return False
    
    # Save artifacts
    try:
        save_index(index, doc_ids, output_dir)
    except Exception as e:
        logger.error(f"Saving failed: {e}")
        return False
    
    logger.info("BGE indexing complete")
    return True


if __name__ == "__main__":
    exit(0 if main() else 1)
