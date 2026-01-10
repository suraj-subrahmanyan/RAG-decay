import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

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


def load_method_results(results_base_dir: Path, method: str, query_id: str):
    """
    Load retrieval results for a query from a specific method's JSONL file.
    
    Args:
        results_base_dir: Base directory containing method result files
        method: Method name (e.g., 'bge', 'e5', 'qwen', 'bm25')
        query_id: Query identifier to filter results
    
    Returns:
        Dict mapping query_field to {doc_id: score}
    """
    method_file = results_base_dir / f"{method}.jsonl"
    
    if not method_file.exists():
        logger.warning(f"Results file not found: {method_file}")
        return {}
    
    results = defaultdict(dict)
    
    try:
        with open(method_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    # Filter by query_id
                    if result['query_id'] == query_id:
                        field = result.get('query_field', 'unknown')
                        results[field][result['doc_id']] = result['score']
    except Exception as e:
        logger.error(f"Error loading results from {method_file}: {e}")
        return {}
    
    return dict(results)


def normalize_scores(scores: dict) -> dict:
    """
    Min-max normalize scores to [0, 1].
    
    Args:
        scores: Dict mapping doc_id to score
        
    Returns:
        Normalized scores dict
    """
    if not scores:
        return {}
    
    score_values = list(scores.values())
    min_score = min(score_values)
    max_score = max(score_values)
    
    # Avoid division by zero
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in scores}
    
    normalized = {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in scores.items()
    }
    
    return normalized


def fuse_results(
    results_by_method: dict,
    top_k: int
) -> list:
    """
    Fuse results from multiple methods using score normalization.
    
    Each method's scores are normalized independently to [0, 1] range
    before aggregation to ensure fair contribution from each model.
    
    Args:
        results_by_method: Dict mapping method_name to {doc_id: score}
        top_k: Number of final results to return
        
    Returns:
        List of (doc_id, fused_score, rank) tuples
    """
    # Normalize scores for each method independently
    # This ensures each model contributes equally regardless of score scale
    normalized_results = {
        method: normalize_scores(results)
        for method, results in results_by_method.items()
    }
    
    # Sum normalized scores across methods
    fused_scores = defaultdict(float)
    
    for method, results in normalized_results.items():
        for doc_id, score in results.items():
            fused_scores[doc_id] += score
    
    # Sort by fused score
    sorted_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return top-k with ranks
    return [
        (doc_id, score, rank)
        for rank, (doc_id, score) in enumerate(sorted_docs[:top_k], 1)
    ]


def get_query_ids(results_base_dir: Path, methods: list) -> set:
    """
    Get all query IDs from method JSONL files.
    
    Args:
        results_base_dir: Base directory containing method result files
        methods: List of method names
        
    Returns:
        Set of query IDs
    """
    query_ids = set()
    
    for method in methods:
        method_file = results_base_dir / f"{method}.jsonl"
        if method_file.exists():
            try:
                with open(method_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            query_ids.add(result['query_id'])
            except Exception as e:
                logger.warning(f"Error reading query IDs from {method_file}: {e}")
    
    return query_ids


def load_query_metadata(queries_file: Path, query_id: str) -> dict:
    """Load metadata for a specific query."""
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                query_data = json.loads(line)
                if query_data['query_id'] == query_id:
                    return query_data
    return {'query_id': query_id, 'answer_id': ''}


def fuse_query_results(
    query_id: str,
    results_base_dir: Path,
    methods: list,
    queries_file: Path,
    top_k: int
):
    """
    Fuse results for a single query, preserving query fields.
    
    Args:
        query_id: Query identifier
        results_base_dir: Base directory with method results
        methods: List of method names to fuse
        queries_file: Path to queries file (for metadata)
        top_k: Number of results to output per field
        
    Returns:
        List of result dictionaries, or None if no results found
    """
    # Load results from each method, grouped by field
    # Structure: {method: {field: {doc_id: score}}}
    results_by_method_field = {}
    all_fields = set()
    
    for method in methods:
        results = load_method_results(results_base_dir, method, query_id)
        if results:
            results_by_method_field[method] = results
            all_fields.update(results.keys())
    
    if not results_by_method_field:
        logger.warning(f"No results found for query {query_id}")
        return None
    
    # Get query metadata
    query_metadata = load_query_metadata(queries_file, query_id)
    answer_id = query_metadata.get('answer_id', '')
    
    all_fused_results = []
    
    # Fuse results per field
    for field in all_fields:
        # constant per field fusion
        field_results_by_method = {}
        
        for method, method_results in results_by_method_field.items():
            if field in method_results:
                field_results_by_method[method] = method_results[field]
        
        if not field_results_by_method:
            continue
            
        # Fuse results for this field
        fused_results = fuse_results(field_results_by_method, top_k)
        
        # Build result dictionaries
        for doc_id, score, rank in fused_results:
            result = {
                'query_id': query_id,
                'answer_id': answer_id,
                'doc_id': doc_id,
                'score': float(score),
                'rank': rank,
                'query_field': field
            }
            all_fused_results.append(result)
    
    return all_fused_results


def main():
    parser = argparse.ArgumentParser(
        description="Fusion retrieval combining multiple methods"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to retrieval configuration YAML"
    )
    parser.add_argument(
        "--corpus-version",
        type=str,
        required=True,
        choices=["oct_2024", "oct_2025"],
        help="Corpus version"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["bm25", "bge", "e5", "qwen"],
        help="Methods to fuse (default: bm25 bge e5 qwen)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    logger.info(f"Fusion Retrieval - {args.corpus_version}")
    logger.info(f"Fusing methods: {', '.join(args.methods)}")
    
    # Setup paths
    results_base_dir = Path(config['output_dir']) / args.corpus_version
    output_file = results_base_dir / "fusion.jsonl"
    
    queries_file = Path(config['queries_file'])
    
    # Get all query IDs
    logger.info("Finding queries to process...")
    query_ids = get_query_ids(results_base_dir, args.methods)
    logger.info(f"Found {len(query_ids)} queries")
    
    # Check that all methods have results
    missing_methods = []
    for method in args.methods:
        method_file = results_base_dir / f"{method}.jsonl"
        if not method_file.exists():
            missing_methods.append(method)
    
    if missing_methods:
        logger.warning(f"Missing results for methods: {', '.join(missing_methods)}")
        logger.warning("Fusion will proceed with available methods only")
    
    # Fuse results for each query and write to single file
    logger.info(f"Fusing results (top-{config['top_k']})...")
    
    total_results = 0
    with open(output_file, 'w', encoding='utf-8') as fout:
        for query_id in tqdm(sorted(query_ids), desc="Processing queries"):
            results = fuse_query_results(
                query_id,
                results_base_dir,
                args.methods,
                queries_file,
                config['top_k']
            )
            
            if results:
                for result in results:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                total_results += len(results)
    
    # Summary
    logger.info("FUSION SUMMARY")
    logger.info(f"Processed {len(query_ids)} queries")
    logger.info(f"Total results: {total_results}")
    logger.info(f"Methods used: {', '.join(args.methods)}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()