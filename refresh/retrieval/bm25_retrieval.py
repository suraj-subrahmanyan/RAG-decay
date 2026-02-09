"""
BM25 sparse retrieval using Pyserini with multiple query formulations.

Supports answer, nuggets, closed_book_answer, and subquestions fields.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List

import yaml
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Maximum query length for BM25
MAX_QUERY_LENGTH = 10000


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def truncate_query(query_text: str, query_id: str, query_field: str) -> str:
    """
    Truncate query if too long for BM25.
    
    Args:
        query_text: Query text
        query_id: Query identifier
        query_field: Which field is being used
        
    Returns:
        Truncated query text
    """
    if len(query_text) > MAX_QUERY_LENGTH:
        logger.warning(
            f"Query {query_id} ({query_field}) truncated: "
            f"{len(query_text)}, {MAX_QUERY_LENGTH} chars"
        )
        return query_text[:MAX_QUERY_LENGTH]
    return query_text


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
        # Join with newlines
        text = "\n".join(nuggets)
        return text, True
    
    elif query_field == "closed_book_answer":
        text = query_data.get('closed_book_answer', '')
        return text, bool(text)
    
    elif query_field == "subquestions":
        subquestions = query_data.get('subquestions', [])
        if not subquestions:
            return "", False
        # Join with newlines
        text = "\n".join(subquestions)
        return text, True
    
    else:
        raise ValueError(f"Unknown query field: {query_field}")


def retrieve_for_query_field(
    searcher: LuceneSearcher,
    query_data: dict,
    query_field: str,
    top_k: int,
    output_file
):
    """
    Retrieve documents for a single query using specified field.
    
    Args:
        searcher: Pyserini searcher
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
    
    # Truncate if needed
    query_text = truncate_query(query_text, query_id, query_field)
    
    try:
        # Search using query text
        hits = searcher.search(query_text, k=top_k)
        
        # Write results
        for rank, hit in enumerate(hits, 1):
            result = {
                'query_id': query_id,
                'answer_id': answer_id,
                'doc_id': hit.docid,
                'score': float(hit.score),
                'rank': rank,
                'query_field': query_field
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Error retrieving for query {query_id} ({query_field}): {e}")
        return False


def process_corpus_version(config: dict, corpus_version: str, query_fields: List[str], queries: List[dict]):
    """
    Process retrieval for a specific corpus version.
    
    Args:
        config: Configuration dictionary
        corpus_version: Corpus version string
        query_fields: List of query fields to process
        queries: List of loaded queries
    """
    logger.info(f"BM25 Retrieval - {corpus_version}")
    logger.info(f"Query fields: {', '.join(query_fields)}")
    
    # Setup paths
    index_dir = Path(config['index_base_dir']) / corpus_version / "bm25"
    output_dir = Path(config['output_dir']) / corpus_version
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file_path = output_dir / "bm25.jsonl"
    
    # Load searcher
    logger.info(f"Loading BM25 index from {index_dir}")
    try:
        searcher = LuceneSearcher(str(index_dir))
        logger.info(f"Index loaded ({searcher.num_docs} documents)")
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
                    searcher,
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
    parser = argparse.ArgumentParser(description="BM25 retrieval with multiple query fields")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to retrieval configuration YAML"
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
        help="Query fields to use (default: all 4)"
    )
    
    args = parser.parse_args()
    
    # Load config and queries (shared)
    config = load_config(args.config)
    queries = load_queries(Path(config['queries_file']))
    
    # Determine versions to process
    if args.corpus_version == "all":
        versions = ["oct_2024", "oct_2025"]
    else:
        versions = [args.corpus_version]
        
    for version in versions:
        process_corpus_version(config, version, args.query_fields, queries)


if __name__ == "__main__":
    main()