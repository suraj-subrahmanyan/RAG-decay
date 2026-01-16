import argparse
import glob
import json
import logging
import os
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def convert_to_trec(base_dir: str, output_dir: str, output_filename: str, dates: list[str]):
    """
    Convert JSONL retrieval results to TREC format.

    Args:
        base_dir: Base directory containing retrieval results.
        output_dir: Directory to save the output TREC file.
        output_filename: Name of the output TREC file.
        dates: List of date folders to process.
    """
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Writing to {output_path}")

    with open(output_path, 'w') as out_f:
        # Progress bar for dates
        for date in tqdm(dates, desc="Dates", leave=False):
            search_path = os.path.join(base_dir, date, "*.jsonl")
            files = glob.glob(search_path)
            
            if not files:
                logger.warning(f"No files found in {search_path}")
                continue
            
            # Progress bar for files
            for file_path in tqdm(files, desc=f"Files in {date}", leave=False):
                filename = os.path.basename(file_path)
                model_name = os.path.splitext(filename)[0]
                
                # count lines for progress bar
                try:
                    with open(file_path, 'r') as f:
                        total_lines = sum(1 for _ in f)
                except Exception as e:
                    logger.error(f"Error counting lines for {filename}: {e}")
                    continue

                with open(file_path, 'r') as in_f:
                    # Progress bar for lines
                    for line in tqdm(in_f, total=total_lines, desc=f"Processing {filename}", leave=False):
                        try:
                            data = json.loads(line)
                            
                            query_id = data.get("query_id")
                            doc_id = data.get("doc_id")
                            rank = data.get("rank")
                            score = data.get("score")
                            query_field = data.get("query_field")
                            
                            if None in [query_id, doc_id, rank, score, query_field]:
                                logger.debug(f"Skipping malformed line in {filename}")
                                continue
                            
                            # run_name construction
                            run_name = f"{model_name}_{date}_{query_field}"
                            
                            # TREC format: query_id Q0 doc_id rank score run_name
                            trec_line = f"{query_id} Q0 {doc_id} {rank} {score} {run_name}\n"
                            out_f.write(trec_line)
                            
                        except json.JSONDecodeError:
                            logger.error(f"Error decoding JSON in {filename}")
                        except Exception as e:
                            logger.error(f"Error processing line in {filename}: {e}")

    logger.info("Conversion complete.")

def main():
    parser = argparse.ArgumentParser(description="Convert JSONL retrieval results to TREC format.")
    
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing retrieval results folders."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/nathank/RAG-decay/experimentation/retrieval_results/trec_runs",
        help="Directory to save the TREC output file."
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="all_runs.trec",
        help="Name of the output TREC file."
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["oct_2024", "oct_2025"],
        help="List of date folders to process (e.g., oct_2024 oct_2025)."
    )

    args = parser.parse_args()
    
    convert_to_trec(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        dates=args.dates
    )

if __name__ == "__main__":
    main()
