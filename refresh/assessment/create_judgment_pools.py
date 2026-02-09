"""
Create judgment pools from retrieval results using TREC-style depth pooling.

Combines top-k documents from all retrieval methods to form a unified pool
for assessment, enabling fair comparison across different retrieval systems.
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_trec_results(trec_file: Path) -> Dict[str, Set[str]]:
    """
    Load TREC format results.
    
    Returns:
        Dict mapping query_id -> set of retrieved doc_ids
    """
    results = defaultdict(set)
    
    if not trec_file.exists():
        logger.warning(f"File not found: {trec_file}")
        return results
    
    with open(trec_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                query_id = parts[0]
                doc_id = parts[2]
                results[query_id].add(doc_id)
    
    return results


def create_pools(results_dir: Path, corpus_version: str, k: int = 100) -> Dict[str, Set[str]]:
    """
    Create judgment pools by combining top-k from all methods.
    
    Args:
        results_dir: Directory containing TREC result files
        corpus_version: "2024" or "2025"
        k: Pool depth (number of docs to include per method)
        
    Returns:
        Dict mapping query_id -> set of pooled doc_ids
    """
    logger.info(f"\nCreating judgment pools for {corpus_version}")
    
    # Find all TREC files for this corpus version
    pattern = f"*_{corpus_version}_*.trec"
    trec_files = list(results_dir.glob(pattern))
    
    logger.info(f"Found {len(trec_files)} result files:")
    for f in trec_files:
        logger.info(f"  - {f.name}")
    
    # Load all results and combine
    pools = defaultdict(set)
    method_stats = {}
    
    for trec_file in trec_files:
        # Robust method name extraction: handle different query field patterns
        # Expected formats: method_2024_queryfield_results.trec
        # Extract just the method name (bge, e5, bm25, fusion, etc.)
        stem = trec_file.stem
        # Remove corpus version and anything after it
        if f"_{corpus_version}_" in stem:
            method_name = stem.split(f"_{corpus_version}_")[0]
        else:
            # Fallback: just take first part before underscore
            method_name = stem.split('_')[0]
        
        method_results = load_trec_results(trec_file)
        
        # Add to pools (limiting to top-k per method)
        for query_id, doc_ids in method_results.items():
            # Take only top-k from this method (they're already ranked in file)
            pooled_docs = list(doc_ids)[:k]
            pools[query_id].update(pooled_docs)
        
        method_stats[method_name] = len(method_results)
    
    # Report statistics
    logger.info(f"\nPool Statistics:")
    logger.info(f"  Methods: {len(trec_files)}")
    for method, num_queries in method_stats.items():
        logger.info(f"    - {method}: {num_queries} queries")
    
    total_queries = len(pools)
    total_docs = sum(len(docs) for docs in pools.values())
    avg_pool_size = total_docs / total_queries if total_queries > 0 else 0
    
    logger.info(f"\n  Total queries: {total_queries}")
    logger.info(f"  Total pooled documents: {total_docs}")
    logger.info(f"  Average pool size: {avg_pool_size:.1f} docs/query")
    
    return pools


def save_pools(pools: Dict[str, Set[str]], output_file: Path):
    """Save judgment pools to JSON file."""
    # Convert sets to lists for JSON serialization
    pools_serializable = {
        query_id: list(doc_ids)
        for query_id, doc_ids in pools.items()
    }
    
    with open(output_file, "w") as f:
        json.dump(pools_serializable, f, indent=2)
    
    logger.info(f"\nSaved pools: {output_file}")


def save_pool_stats(pools: Dict[str, Set[str]], output_file: Path):
    """Save pool statistics."""
    stats = {
        "total_queries": len(pools),
        "total_pooled_documents": sum(len(docs) for docs in pools.values()),
        "avg_pool_size": sum(len(docs) for docs in pools.values()) / len(pools) if pools else 0,
        "min_pool_size": min(len(docs) for docs in pools.values()) if pools else 0,
        "max_pool_size": max(len(docs) for docs in pools.values()) if pools else 0,
        "pool_size_distribution": {
            query_id: len(doc_ids)
            for query_id, doc_ids in pools.items()
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved stats: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Create judgment pools")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="retrieval_results",
        help="Directory containing TREC result files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="judgment_pools",
        help="Output directory for pools"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Pool depth (docs per method)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pools for both corpus versions
    for corpus_version in ["2024", "2025"]:
        pools = create_pools(results_dir, corpus_version, k=args.k)
        
        # Save pools
        output_file = output_dir / f"judgment_pool_{corpus_version}.json"
        save_pools(pools, output_file)
        
        # Save stats
        stats_file = output_dir / f"pool_stats_{corpus_version}.json"
        save_pool_stats(pools, stats_file)
    
    logger.info("\nJudgment pool creation complete")


if __name__ == "__main__":
    main()
