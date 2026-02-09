"""
Evaluate retrieval results using qrels (nugget-level relevance judgments).

Computes α-nDCG@10, Coverage@20, and Recall@50.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_qrels(qrels_file: Path) -> Dict[str, Set[str]]:
    """
    Load qrels (ground truth relevance judgments).
    
    Returns:
        Dict mapping "query_id_nugget_idx" -> set(relevant_doc_ids)
    """
    with open(qrels_file, 'r') as f:
        data = json.load(f)
    
    qrels = {}
    for query_nugget_key, doc_judgments in data['qrels_nuggets'].items():
        # Only keep docs with relevance = 1
        relevant_docs = {doc_id for doc_id, rel in doc_judgments.items() if rel == 1}
        qrels[query_nugget_key] = relevant_docs
    
    logger.info(f"Loaded qrels for {len(qrels)} query-nugget pairs")
    return qrels


def load_retrieval_results(results_file: Path) -> Dict[str, List[str]]:
    """
    Load retrieval results from JSONL.
    
    Format: {"query_id": "...", "doc_id": "...", "rank": ..., "score": ...}
    
    Returns:
        Dict mapping query_id -> [doc_ids in rank order]
    """
    results = defaultdict(list)
    
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = str(data['query_id'])
            doc_id = data['doc_id']
            rank = data['rank']
            results[query_id].append((rank, doc_id))
    
    # Sort by rank and extract doc_ids
    sorted_results = {}
    for query_id, docs in results.items():
        docs.sort(key=lambda x: x[0])  # Sort by rank
        sorted_results[query_id] = [doc_id for _, doc_id in docs]
    
    logger.info(f"Loaded results for {len(sorted_results)} queries")
    return sorted_results


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
    """
    return compute_coverage(retrieved_docs, relevant_docs, k=k)


def evaluate_nugget_level(
    qrels: Dict[str, Set[str]],
    retrieval_results: Dict[str, List[str]]
) -> Dict:
    """
    Evaluate retrieval at nugget level.
    
    For each query-nugget pair, compute metrics, then aggregate.
    """
    
    all_ndcg_10 = []
    all_coverage_20 = []
    all_recall_50 = []
    
    query_level_metrics = defaultdict(lambda: {
        'nuggets': [],
        'ndcg_10': [],
        'coverage_20': [],
        'recall_50': []
    })
    
    for query_nugget_key, relevant_docs in tqdm(qrels.items(), desc="Evaluating"):
        # Parse query_id from key (format: "query_id_nugget_idx")
        query_id = '_'.join(query_nugget_key.split('_')[:-1])
        nugget_idx = query_nugget_key.split('_')[-1]
        
        # Skip if no retrieval results
        if query_id not in retrieval_results:
            logger.warning(f"No results for query {query_id}")
            continue
        
        retrieved_docs = retrieval_results[query_id]
        
        # Skip nuggets with no relevant docs
        if len(relevant_docs) == 0:
            continue
        
        # Compute metrics
        ndcg = compute_alpha_ndcg(retrieved_docs, relevant_docs, k=10)
        cov = compute_coverage(retrieved_docs, relevant_docs, k=20)
        rec = compute_recall(retrieved_docs, relevant_docs, k=50)
        
        # Store
        all_ndcg_10.append(ndcg)
        all_coverage_20.append(cov)
        all_recall_50.append(rec)
        
        # Track per-query
        query_level_metrics[query_id]['nuggets'].append(nugget_idx)
        query_level_metrics[query_id]['ndcg_10'].append(ndcg)
        query_level_metrics[query_id]['coverage_20'].append(cov)
        query_level_metrics[query_id]['recall_50'].append(rec)
    
    # Aggregate query-level metrics (average across nuggets per query)
    query_aggregated = []
    for query_id, data in query_level_metrics.items():
        query_aggregated.append({
            'query_id': query_id,
            'num_nuggets': len(data['nuggets']),
            'alpha_ndcg_10': float(np.mean(data['ndcg_10'])),
            'coverage_20': float(np.mean(data['coverage_20'])),
            'recall_50': float(np.mean(data['recall_50']))
        })
    
    # Overall metrics (micro-average across all nuggets)
    metrics = {
        'num_queries': len(query_level_metrics),
        'num_nuggets': len(all_ndcg_10),
        'alpha_ndcg_10': {
            'mean': float(np.mean(all_ndcg_10)) if all_ndcg_10 else 0.0,
            'std': float(np.std(all_ndcg_10)) if all_ndcg_10 else 0.0,
            'median': float(np.median(all_ndcg_10)) if all_ndcg_10 else 0.0
        },
        'coverage_20': {
            'mean': float(np.mean(all_coverage_20)) if all_coverage_20 else 0.0,
            'std': float(np.std(all_coverage_20)) if all_coverage_20 else 0.0,
            'median': float(np.median(all_coverage_20)) if all_coverage_20 else 0.0
        },
        'recall_50': {
            'mean': float(np.mean(all_recall_50)) if all_recall_50 else 0.0,
            'std': float(np.std(all_recall_50)) if all_recall_50 else 0.0,
            'median': float(np.median(all_recall_50)) if all_recall_50 else 0.0
        },
        'query_level': query_aggregated
    }
    
    return metrics


def save_metrics(metrics: Dict, output_file: Path) -> None:
    """Save evaluation metrics to JSON."""
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
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
    parser.add_argument("--qrels", required=True, help="qrels JSON file")
    parser.add_argument("--results", required=True, help="Retrieval results JSONL")
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    parser.add_argument("--method", required=True, help="Method name (bm25/bge/fusion)")
    parser.add_argument("--version", required=True, help="Version (oct_2024/oct_2025)")
    args = parser.parse_args()
    
    qrels_file = Path(args.qrels)
    results_file = Path(args.results)
    output_file = Path(args.output)
    
    logger.info("="*80)
    logger.info(f"EVALUATING: {args.method} on {args.version}")
    logger.info("="*80)
    logger.info(f"qrels: {qrels_file}")
    logger.info(f"Results: {results_file}")
    
    # Load data
    qrels = load_qrels(qrels_file)
    retrieval_results = load_retrieval_results(results_file)
    
    # Evaluate
    metrics = evaluate_nugget_level(qrels, retrieval_results)
    
    # Save and print
    save_metrics(metrics, output_file)
    print_summary(metrics, args.method, args.version)
    
    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()
