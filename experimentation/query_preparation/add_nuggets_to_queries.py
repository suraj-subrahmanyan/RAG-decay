import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_nuggets(nuggets_file: Path) -> dict:
    """
    Load nuggets from JSON file and group by query_id.
    
    Args:
        nuggets_file: Path to nuggets JSON file
        
    Returns:
        Dictionary mapping query_id to list of nuggets
    """
    logger.info(f"Loading nuggets from {nuggets_file}")
    
    with open(nuggets_file, 'r', encoding='utf-8') as f:
        nuggets_data = json.load(f)
    
    logger.info(f"Loaded {len(nuggets_data)} nugget entries")
    
    # Group nuggets by query_id
    nuggets_by_query = defaultdict(list)
    
    for key, nugget_text in nuggets_data.items():
        # Extract base query_id by removing the _N suffix
        if '_' in key:
            query_id = key.rsplit('_', 1)[0]
        else:
            # No underscore, use the whole key as query_id
            query_id = key
            logger.warning(f"Nugget key without underscore: {key}")
        
        nuggets_by_query[query_id].append(nugget_text)
    
    logger.info(f"Grouped nuggets into {len(nuggets_by_query)} unique queries")
    
    # Log statistics
    nugget_counts = [len(nuggets) for nuggets in nuggets_by_query.values()]
    if nugget_counts:
        logger.info(f"Min nuggets per query: {min(nugget_counts)}")
        logger.info(f"Max nuggets per query: {max(nugget_counts)}")
        logger.info(f"Avg nuggets per query: {sum(nugget_counts) / len(nugget_counts):.1f}")
    
    return dict(nuggets_by_query)


def load_queries(queries_file: Path) -> list:
    """
    Load queries from JSONL file.
    
    Args:
        queries_file: Path to queries JSONL file
        
    Returns:
        List of query dictionaries
    """
    logger.info(f"Loading queries from {queries_file}")
    
    queries = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    query = json.loads(line)
                    queries.append(query)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def enrich_queries_with_nuggets(queries: list, nuggets_by_query: dict) -> list:
    """
    Add nuggets field to each query.
    
    Args:
        queries: List of query dictionaries
        nuggets_by_query: Dictionary mapping query_id to nugget list
        
    Returns:
        Enriched list of queries
    """
    logger.info("Enriching queries with nuggets...")
    
    enriched_queries = []
    queries_with_nuggets = 0
    queries_without_nuggets = 0
    
    for query in queries:
        query_id = str(query.get('query_id', ''))
        
        if query_id in nuggets_by_query:
            # Add nuggets
            query['nuggets'] = nuggets_by_query[query_id]
            queries_with_nuggets += 1
        else:
            # No nuggets found - add empty list and warn
            query['nuggets'] = []
            queries_without_nuggets += 1
            logger.warning(f"No nuggets found for query_id: {query_id}")
        
        enriched_queries.append(query)
    
    logger.info(f"Enrichment complete:")
    logger.info(f"Queries with nuggets: {queries_with_nuggets}")
    logger.info(f"Queries without nuggets: {queries_without_nuggets}")
    
    # Check for unused nuggets
    used_query_ids = {str(q.get('query_id', '')) for q in queries}
    unused_nugget_ids = set(nuggets_by_query.keys()) - used_query_ids
    
    if unused_nugget_ids:
        logger.warning(f"Found {len(unused_nugget_ids)} nugget groups without matching queries:")
        for unused_id in sorted(list(unused_nugget_ids)[:10]):  # Show first 10
            logger.warning(f"Unused nuggets for query_id: {unused_id}")
        if len(unused_nugget_ids) > 10:
            logger.warning(f"  ... and {len(unused_nugget_ids) - 10} more")
    
    return enriched_queries


def save_queries(queries: list, output_file: Path, backup: bool = True):
    """
    Save enriched queries to JSONL file.
    
    Args:
        queries: List of enriched query dictionaries
        output_file: Path to output file
        backup: Whether to create backup of existing file
    """
    # Create backup if file exists
    if backup and output_file.exists():
        backup_file = output_file.with_suffix('.backup.jsonl')
        logger.info(f"Creating backup: {backup_file}")
        shutil.copy2(output_file, backup_file)
    
    # Write enriched queries
    logger.info(f"Writing enriched queries to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(queries)} enriched queries")


def main():
    parser = argparse.ArgumentParser(
        description="Add nuggets to queries_answers.jsonl"
    )
    parser.add_argument(
        "--queries",
        type=str,
        default="retrieval_results/queries_answers.jsonl",
        help="Path to queries JSONL file (default: retrieval_results/queries_answers.jsonl)"
    )
    parser.add_argument(
        "--nuggets",
        type=str,
        default="data/nuggets.json",
        help="Path to nuggets JSON file (default: data/nuggets.json)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original queries file"
    )
    
    args = parser.parse_args()
    
    logger.info("Add Nuggets to Queries")
    
    # Convert to Path objects
    queries_file = Path(args.queries)
    nuggets_file = Path(args.nuggets)
    
    # Validate files exist
    if not queries_file.exists():
        logger.error(f"Queries file not found: {queries_file}")
        exit(1)
    
    if not nuggets_file.exists():
        logger.error(f"Nuggets file not found: {nuggets_file}")
        exit(1)
    
    try:
        # Load data
        nuggets_by_query = load_nuggets(nuggets_file)
        queries = load_queries(queries_file)
        
        if not queries:
            logger.error("No queries found in file")
            exit(1)
        
        # Enrich queries
        enriched_queries = enrich_queries_with_nuggets(queries, nuggets_by_query)
        
        # Save results
        save_queries(enriched_queries, queries_file, backup=not args.no_backup)
        
        # Summary
        logger.info("SUMMARY")
        logger.info(f"Successfully enriched {len(enriched_queries)} queries with nuggets")
        logger.info(f"Output: {queries_file}")
        if not args.no_backup:
            logger.info(f"Backup: {queries_file.with_suffix('.jsonl.backup')}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()