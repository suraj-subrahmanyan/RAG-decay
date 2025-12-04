"""
Build BM25 index using Pyserini/Anserini.

Standard sparse retrieval baseline for temporal corpus evaluation.
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def prepare_corpus(corpus_file: Path, output_dir: Path) -> int:
    """
    Convert corpus to Pyserini format.
    
    Args:
        corpus_file: Input corpus in FreshStack format (_id, text, metadata)
        output_dir: Directory for converted corpus
        
    Returns:
        Number of documents successfully processed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "corpus.jsonl"
    
    doc_count = 0
    errors = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Preparing corpus"):
            if not line.strip():
                continue
            
            try:
                doc = json.loads(line)
                
                # Validate required fields
                if '_id' not in doc or 'text' not in doc:
                    errors += 1
                    continue
                
                pyserini_doc = {
                    'id': doc['_id'],
                    'contents': doc['text'],
                    'metadata': doc.get('metadata', {})
                }
                fout.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
                doc_count += 1
                
            except (json.JSONDecodeError, KeyError) as e:
                errors += 1
                continue
    
    if errors > 0:
        logger.warning(f"Skipped {errors} invalid documents")
    logger.info(f"Prepared {doc_count:,} documents")
    
    return doc_count


def build_index(corpus_dir: Path, index_dir: Path) -> bool:
    """
    Build BM25 index using Pyserini/Anserini.
    
    Args:
        corpus_dir: Directory containing prepared corpus
        index_dir: Output directory for Lucene index
        
    Returns:
        True if indexing succeeded, False otherwise
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(corpus_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions",  # Required for phrase queries
        "--storeDocvectors",  # Required for relevance feedback
        "--storeRaw"  # Store original documents
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=1800  # 30 min timeout
        )
        logger.info(f"Index saved to: {index_dir}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Indexing timed out (>30 minutes)")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Indexing failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main() -> bool:
    parser = argparse.ArgumentParser(description="Build BM25 index")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL file")
    parser.add_argument("--output", required=True, help="Index output directory")
    args = parser.parse_args()
    
    corpus_file = Path(args.corpus)
    index_dir = Path(args.output)
    
    if not corpus_file.exists():
        logger.error(f"Corpus not found: {corpus_file}")
        return False
    
    logger.info("="*60)
    logger.info("BM25 INDEX CONSTRUCTION")
    logger.info("="*60)
    logger.info(f"Corpus: {corpus_file}")
    logger.info(f"Output: {index_dir}")
    
    # Use temp directory for prepared corpus
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_corpus = Path(temp_dir)
        
        doc_count = prepare_corpus(corpus_file, temp_corpus)
        if doc_count == 0:
            logger.error("No documents found")
            return False
        
        success = build_index(temp_corpus, index_dir)
    
    if success:
        logger.info("BM25 indexing complete")
    else:
        logger.error("BM25 indexing failed")
    
    return success


if __name__ == "__main__":
    exit(0 if main() else 1)
