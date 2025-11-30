import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

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


class DenseEncoder:
    """Encoder for dense retrieval models."""
    
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512):
        """
        Initialize encoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (cuda or cpu)
            max_length: Maximum sequence length
        """
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, model_kwargs={"dtype":"float16"}, device=device)
        logger.info(f"Model loaded on {device}")
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings (batch_size, embedding_dim)
        """
        document_embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return document_embeddings


def load_corpus(corpus_file: Path) -> Tuple[List[str], List[str]]:
    """
    Load corpus from JSONL file.
    
    Args:
        corpus_file: Path to corpus JSONL
        
    Returns:
        Tuple of (doc_ids, texts)
    """
    logger.info(f"Loading corpus from {corpus_file}")
    
    doc_ids = []
    texts = []
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            if not line.strip():
                continue
            
            try:
                doc = json.loads(line)
                doc_ids.append(doc['_id'])
                texts.append(doc['text'])
            except json.JSONDecodeError:
                continue
    
    logger.info(f"Loaded {len(doc_ids)} documents")
    return doc_ids, texts


def encode_corpus(
    texts: List[str],
    encoder: DenseEncoder,
    batch_size: int = 32
) -> np.ndarray:
    """
    Encode entire corpus.
    
    Args:
        texts: List of text strings
        encoder: Dense encoder instance
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings (num_docs, embedding_dim)
    """
    logger.info(f"Encoding {len(texts)} documents...")
    
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = encoder.encode_batch(batch_texts)
        all_embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Encoded corpus shape: {embeddings.shape}")
    
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index (flat/exact search).
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS index
    """
    logger.info("Building FAISS index (flat)...")
    
    embedding_dim = embeddings.shape[1]
    
    # Create flat index for exact search
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
    
    # Add embeddings
    index.add(embeddings.astype(np.float32))
    
    logger.info(f"Index built with {index.ntotal} vectors")
    return index


def save_index_and_metadata(
    index: faiss.Index,
    doc_ids: List[str],
    output_dir: Path
):
    """
    Save FAISS index and metadata.
    
    Args:
        index: FAISS index
        doc_ids: List of document IDs
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_file = output_dir / "index.faiss"
    faiss.write_index(index, str(index_file))
    logger.info(f"Index saved to {index_file}")
    
    # Save document IDs mapping
    docid_file = output_dir / "docids.pkl"
    with open(docid_file, 'wb') as f:
        pickle.dump(doc_ids, f)
    logger.info(f"Document IDs saved to {docid_file}")


def process_corpus_version(
    corpus_file: Path,
    version: str,
    model_name: str,
    model_config: dict,
    device: str
) -> bool:
    """
    Process a single corpus version.
    
    Args:
        corpus_file: Path to corpus JSONL
        version: Version identifier
        model_name: Model name (bge or e5)
        model_config: Model configuration
        device: Device to use
        
    Returns:
        True if successful
    """
    logger.info(f"Processing: {version} with {model_name.upper()}")
    
    try:
        # Load corpus
        doc_ids, texts = load_corpus(corpus_file)
        
        if not texts:
            logger.error("No documents to encode")
            return False
        
        # Initialize encoder
        encoder = DenseEncoder(
            model_name=model_config['model_name'],
            device=device,
            max_length=model_config['max_length']
        )
        
        # Encode corpus
        embeddings = encode_corpus(
            texts,
            encoder,
            batch_size=model_config['batch_size']
        )
        
        # Build FAISS index
        index = build_faiss_index(embeddings)
        
        # Save index and metadata
        output_dir = Path(model_config['output_dir']) / version / model_name
        save_index_and_metadata(index, doc_ids, output_dir)
        
        logger.info(f"Successfully indexed {version} with {model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {version}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build dense indices using BGE or E5 models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to indexing configuration YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["bge", "e5", "qwen"],
        help="Model to use (bge, e5, or qwen)"
    )
    parser.add_argument(
        "--corpus-version",
        type=str,
        choices=["oct_2024", "oct_2025", "all"],
        default="all",
        help="Which corpus version to index (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    logger.info(f"Dense Index Builder - {args.model.upper()}")
    
    # Get model config
    model_config = config['dense_models'][args.model]
    model_config['output_dir'] = config['index_output_dir']
    device = config.get('device', 'cpu')
    
    logger.info(f"Model: {model_config['model_name']}")
    logger.info(f"Device: {device}")
    logger.info(f"Batch size: {model_config['batch_size']}")
    
    # Determine which versions to process
    if args.corpus_version == "all":
        versions = ["oct_2024", "oct_2025"]
    else:
        versions = [args.corpus_version]
    
    # Process each version
    results = {}
    for version in versions:
        corpus_file = Path(config['corpus_files'][version])
        
        if not corpus_file.exists():
            logger.error(f"Corpus file not found: {corpus_file}")
            results[version] = False
            continue
        
        success = process_corpus_version(
            corpus_file,
            version,
            args.model,
            model_config,
            device
        )
        results[version] = success
    
    # Summary
    logger.info("INDEXING SUMMARY")
    for version, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"{version}: {status}")
    
    # Exit with error if any failed
    if not all(results.values()):
        exit(1)


if __name__ == "__main__":
    main()