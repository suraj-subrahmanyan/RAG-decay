import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_and_save_queries(output_path: str):
    """
    Load queries from HuggingFace and save to JSONL.
    
    Args:
        output_path: Path to output JSONL file
    """
    logger.info("Loading FreshStack Queries Dataset")
    try:
        dataset = load_dataset("freshstack/queries-oct-2024", "langchain")
        logger.info(f"Dataset loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    dataset = dataset['test']
    logger.info(f"Total queries: {len(dataset)}")
    logger.info(f"Available columns: {dataset.column_names}")
    
    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Process and save
    logger.info(f"Processing queries...")
    saved_count = 0
    skipped_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for item in tqdm(dataset, desc="Processing queries"):
            try:
                query_data = {
                    'query_id': item.get('query_id'),
                    'question': item.get('query_text'),
                    'answer': item.get('answer_text'),
                    'answer_id': item.get('answer_id'),
                }

                fout.write(json.dumps(query_data, ensure_ascii=False) + "\n")
                saved_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing item: {e}")
                skipped_count += 1
                continue
    
    # Summary
    logger.info("SUMMARY")
    logger.info(f"Queries saved: {saved_count}")
    if skipped_count > 0:
        logger.info(f"Queries skipped: {skipped_count}")
    logger.info(f"Output file: {output_file}")
    
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Load queries and answers from FreshStack dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_results/queries_answers.jsonl",
        help="Output JSONL file path (default: retrieval_results/queries_answers.jsonl)"
    )
    
    args = parser.parse_args()
    
    try:
        load_and_save_queries(args.output)
    except Exception as e:
        logger.error(f"Failed to load queries: {e}")
        exit(1)


if __name__ == "__main__":
    main()