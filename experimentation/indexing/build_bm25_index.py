import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import yaml
from tqdm import tqdm

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


def prepare_pyserini_corpus(corpus_file: Path, output_dir: Path) -> int:
    """
    Prepare corpus in Pyserini's expected format.
    
    Pyserini expects JSONL with 'id', 'contents', and optionally 'metadata'.
    
    Args:
        corpus_file: Input corpus JSONL file
        output_dir: Directory to save prepared corpus
        
    Returns:
        Number of documents processed
    """
    logger.info(f"Preparing corpus for Pyserini from {corpus_file}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "corpus.jsonl"
    
    doc_count = 0
    
    with open(corpus_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Preparing corpus"):
            if not line.strip():
                continue
            
            try:
                doc = json.loads(line)
                
                # Pyserini format
                pyserini_doc = {
                    'id': doc['_id'],
                    'contents': doc['text'],
                    'metadata': doc.get('metadata', {})
                }
                
                fout.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
                doc_count += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON line: {e}")
                continue
    
    logger.info(f"Prepared {doc_count} documents")
    return doc_count


def build_bm25_index(corpus_dir: Path, index_dir: Path) -> bool:
    """
    Build BM25 index using Pyserini.
    
    Args:
        corpus_dir: Directory containing prepared corpus
        index_dir: Output directory for index
        
    Returns:
        True if successful
    """
    logger.info(f"Building BM25 index...")
    logger.info(f"Corpus: {corpus_dir}")
    logger.info(f"Index: {index_dir}")
    
    index_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run Pyserini indexer
        cmd = [
            "python", "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", str(corpus_dir),
            "--index", str(index_dir),
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        _ = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("BM25 index built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build BM25 index: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def process_corpus_version(
    corpus_file: Path,
    version: str,
    config: dict
) -> bool:
    """
    Process a single corpus version.
    
    Args:
        corpus_file: Path to corpus JSONL
        version: Version identifier (oct_2024 or oct_2025)
        config: Configuration dictionary
        
    Returns:
        True if successful
    """
    logger.info(f"Processing corpus version: {version}")
    
    # Create paths
    index_base = Path(config['index_output_dir']) / version / "bm25"
    
    # Use temp directory for prepared corpus
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_corpus_dir = Path(temp_dir)
        
        # Prepare corpus
        doc_count = prepare_pyserini_corpus(corpus_file, temp_corpus_dir)
        
        if doc_count == 0:
            logger.error(f"No documents found in {corpus_file}")
            return False
        
        # Build index
        success = build_bm25_index(temp_corpus_dir, index_base)
        
        if success:
            logger.info(f"Index saved to: {index_base}")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Build BM25 indices using Pyserini"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to indexing configuration YAML"
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
        
        success = process_corpus_version(corpus_file, version, config)
        results[version] = success
    
    # Summary
    logger.info("INDEXING SUMMARY")
    for version, success in results.items():
        status = "SUCCESS" if success else  "FAILED"
        logger.info(f"{version}: {status}")
    
    # Exit with error if any failed
    if not all(results.values()):
        exit(1)


if __name__ == "__main__":
    main()