import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def extract_repo_name_from_filename(filename: str) -> str:
    """
    Extract repository name from corpus filename.
    
    Args:
        filename: e.g., "corpus.openai-python.jsonl"
        
    Returns:
        Repository name, e.g., "openai-python"
    """
    if filename.startswith("corpus."):
        filename = filename[7:]  # Remove "corpus."
    if filename.endswith(".jsonl"):
        filename = filename[:-6]  # Remove ".jsonl"
    return filename


def update_chunk_id(chunk: Dict, repo_name: str) -> Dict:
    """
    Update the _id field to include repo name prefix.
    
    Args:
        chunk: Original chunk dictionary
        repo_name: Repository name to prepend
        
    Returns:
        Updated chunk dictionary
    """
    original_id = chunk.get("_id", "")
    # Add repo name prefix if not already present
    if not original_id.startswith(f"{repo_name}/"):
        chunk["_id"] = f"{repo_name}/{original_id}"
    return chunk


def concatenate_version(
    version_dir: Path,
    version_name: str,
    output_dir: Path
) -> Path:
    """
    Concatenate all corpus files in a version directory.
    
    Args:
        version_dir: Directory containing corpus.*.jsonl files
        version_name: Version identifier (e.g., "oct_2024")
        output_dir: Output directory for concatenated file
        
    Returns:
        Path to output file
    """
    logger.info(f"Processing version: {version_name}")
    corpus_files = sorted(version_dir.glob("corpus.*.jsonl"))
    
    if not corpus_files:
        logger.warning(f"No corpus files found in {version_dir}")
        return None
    
    logger.info(f"Found {len(corpus_files)} corpus files:")
    for f in corpus_files:
        logger.info(f"{f.name}")
    
    output_file = output_dir / f"corpus_{version_name}.jsonl"
    
    
    # Process each corpus file
    with open(output_file, 'w', encoding='utf-8') as fout:
        for corpus_file in corpus_files:
            repo_name = extract_repo_name_from_filename(corpus_file.name)
            
            with open(corpus_file, 'r', encoding='utf-8') as fin:
                pbar = tqdm(fin, leave=False, desc=f"Processing {corpus_file.name} (repo: {repo_name})...")
                for line in pbar:
                    if line.strip:
                        try:
                            chunk = json.loads(line)
                            
                            # Update _id with repo prefix
                            chunk = update_chunk_id(chunk, repo_name)
                            
                            # Write updated chunk
                            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line in {corpus_file.name}: {e}")
                            continue
            
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate corpus JSONL files into single files per version"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing version subdirectories (e.g., experimentation/scripts/dataset/langchain)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input-dir)"
    )
    parser.add_argument(
        "--versions",
        type=str,
        nargs="+",
        default=["oct_2024", "oct_2025"],
        help="Version directories to process (default: oct_2024 oct_2025)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Corpus Concatenation Script")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Versions to process: {', '.join(args.versions)}")
    
    # Process each version
    results = {}
    for version in args.versions:
        version_dir = input_dir / version
        
        if not version_dir.exists():
            logger.warning(f"Version directory not found: {version_dir}")
            continue
        
        if not version_dir.is_dir():
            logger.warning(f"Not a directory: {version_dir}")
            continue
        
        output_file = concatenate_version(version_dir, version, output_dir)
        results[version] = output_file
    
    # Final summary
    logger.info("FINAL SUMMARY")
    for version, output_file in results.items():
        if output_file:
            logger.info(f"{version}: {output_file}")
        else:
            logger.info(f"{version}: FAILED")


if __name__ == "__main__":
    main()