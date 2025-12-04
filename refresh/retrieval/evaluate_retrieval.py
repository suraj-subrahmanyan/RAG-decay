"""
Evaluate retrieval results using nugget-level metrics.

Computes α-nDCG@10, Coverage@20, and Recall@50 for retrieved documents.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_queries(queries_file: Path) -> pd.DataFrame:
    """Load queries with nuggets from parquet file."""
    df = pd.read_parquet(queries_file)
    logger.info(f"Loaded {len(df)} queries with nuggets")
    return df


def load_retrieval_results(results_file: Path) -> Dict[str, List[str]]:
    """
    Load retrieval results.
    
    Returns:
        Dict mapping query_id -> [ranked doc_ids]
    """
    
    results = {}
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            doc_ids = [r['doc_id'] for r in data['results']]
            results[query_id] = doc_ids
    
    logger.info(f"Loaded results for {len(results)} queries")
    return results


def compute_alpha_ndcg(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int = 10,
    alpha: float = 0.5
) -> float:
    """
    Compute α-nDCG@k with graded relevance.
    
    α-nDCG accounts for both relevant and non-relevant documents:
    - Relevant doc: gain = 1
    - Non-relevant doc: gain = -α
    
    Args:
        retrieved_docs: Ranked list of retrieved document IDs
        relevant_docs: Set of relevant document IDs for this nugget
        k: Cutoff rank
        alpha: Penalty weight for non-relevant docs
    """
    
    if len(relevant_docs) == 0:
        return 0.0
    
    # Truncate to top-k
    retrieved_docs = retrieved_docs[:k]
    
    # Compute DCG
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_docs, start=1):
        gain = 1.0 if doc_id in relevant_docs else -alpha
        dcg += gain / np.log2(rank + 1)
    
    # Compute IDCG (ideal DCG)
    # Best case: all relevant docs first, then non-relevant
    num_relevant = len(relevant_docs)
    ideal_ranking = min(k, num_relevant) * [1.0]  # Relevant docs
    if len(ideal_ranking) < k:
        ideal_ranking += (k - len(ideal_ranking)) * [-alpha]  # Non-relevant
    
    idcg = sum(
        gain / np.log2(rank + 1)
        for rank, gain in enumerate(ideal_ranking, start=1)
    )
    
    # Normalize
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_coverage(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int = 20
) -> float:
    """
    Compute Coverage@k: fraction of relevant docs found in top-k.
    
    Coverage = |retrieved ∩ relevant| / |relevant|
    """
    
    if len(relevant_docs) == 0:
        return 0.0
    
    retrieved_set = set(retrieved_docs[:k])
    found = len(retrieved_set & relevant_docs)
    
    return found / len(relevant_docs)


def compute_recall(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int = 50
) -> float:
    """
    Compute Recall@k: same as Coverage@k.
    
    Recall = |retrieved ∩ relevant| / |relevant|
    """
    
    return compute_coverage(retrieved_docs, relevant_docs, k=k)


def evaluate_nugget_level(
    queries: pd.DataFrame,
    retrieval_results: Dict[str, List[str]]
) -> Dict:
    """
    Evaluate retrieval at nugget level.
    
    For each query, compute metrics for each nugget, then average.
    """
    
    all_ndcg_10 = []
    all_coverage_20 = []
    all_recall_50 = []
    
    query_level_metrics = []
    
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="Evaluating"):
        query_id = str(row['query_id'])
        nuggets = row['nuggets']
        
        # Skip if no retrieval results
        if query_id not in retrieval_results:
            logger.warning(f"No results for query {query_id}")
            continue
        
        retrieved_docs = retrieval_results[query_id]
        
        # Compute metrics for each nugget
        nugget_ndcg_10 = []
        nugget_coverage_20 = []
        nugget_recall_50 = []
        
        for nugget in nuggets:
            relevant = set(nugget['relevant_corpus_ids'])
            
            # Skip nuggets with no relevant docs
            if len(relevant) == 0:
                continue
            
            ndcg = compute_alpha_ndcg(retrieved_docs, relevant, k=10)
            cov = compute_coverage(retrieved_docs, relevant, k=20)
            rec = compute_recall(retrieved_docs, relevant, k=50)
            
            nugget_ndcg_10.append(ndcg)
            nugget_coverage_20.append(cov)
            nugget_recall_50.append(rec)
        
        # Average across nuggets for this query
        if len(nugget_ndcg_10) > 0:
            query_level_metrics.append({
                'query_id': query_id,
                'num_nuggets': len(nugget_ndcg_10),
                'alpha_ndcg_10': np.mean(nugget_ndcg_10),
                'coverage_20': np.mean(nugget_coverage_20),
                'recall_50': np.mean(nugget_recall_50)
            })
            
            all_ndcg_10.extend(nugget_ndcg_10)
            all_coverage_20.extend(nugget_coverage_20)
            all_recall_50.extend(nugget_recall_50)
    
    # Overall metrics (micro-average across all nuggets)
    metrics = {
        'num_queries': len(query_level_metrics),
        'num_nuggets': len(all_ndcg_10),
        'alpha_ndcg_10': {
            'mean': float(np.mean(all_ndcg_10)),
            'std': float(np.std(all_ndcg_10)),
            'median': float(np.median(all_ndcg_10))
        },
        'coverage_20': {
            'mean': float(np.mean(all_coverage_20)),
            'std': float(np.std(all_coverage_20)),
            'median': float(np.median(all_coverage_20))
        },
        'recall_50': {
            'mean': float(np.mean(all_recall_50)),
            'std': float(np.std(all_recall_50)),
            'median': float(np.median(all_recall_50))
        },
        'query_level': query_level_metrics
    }
    
    return metrics


def save_metrics(metrics: Dict, output_file: Path) -> None:
    """Save evaluation metrics to JSON."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved metrics to {output_file}")


def print_summary(metrics: Dict, method: str, version: str) -> None:
    """Print evaluation summary."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATION SUMMARY: {method.upper()} on {version}")
    logger.info(f"{'='*80}")
    logger.info(f"Queries evaluated: {metrics['num_queries']}")
    logger.info(f"Total nuggets: {metrics['num_nuggets']}")
    logger.info(f"\nα-nDCG@10:   {metrics['alpha_ndcg_10']['mean']:.4f} "
                f"(±{metrics['alpha_ndcg_10']['std']:.4f})")
    logger.info(f"Coverage@20: {metrics['coverage_20']['mean']:.4f} "
                f"(±{metrics['coverage_20']['std']:.4f})")
    logger.info(f"Recall@50:   {metrics['recall_50']['mean']:.4f} "
                f"(±{metrics['recall_50']['std']:.4f})")
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results")
    parser.add_argument("--queries", required=True, help="Queries parquet file")
    parser.add_argument("--results", required=True, help="Retrieval results JSONL")
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    parser.add_argument("--method", required=True, help="Method name (bm25/bge/fusion)")
    parser.add_argument("--version", required=True, help="Version (oct_2024/oct_2025)")
    args = parser.parse_args()
    
    queries_file = Path(args.queries)
    results_file = Path(args.results)
    output_file = Path(args.output)
    
    logger.info("="*80)
    logger.info(f"EVALUATING: {args.method} on {args.version}")
    logger.info("="*80)
    logger.info(f"Queries: {queries_file}")
    logger.info(f"Results: {results_file}")
    
    # Load data
    queries = load_queries(queries_file)
    retrieval_results = load_retrieval_results(results_file)
    
    # Evaluate
    metrics = evaluate_nugget_level(queries, retrieval_results)
    
    # Save and print
    save_metrics(metrics, output_file)
    print_summary(metrics, args.method, args.version)
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
