#!/usr/bin/env python3
"""
Nugget-level retrieval evaluation with graded relevance metrics.

This script:
1. Loads assessment results (JSONL with per-document judgments)
2. Loads retrieval results (JSONL with ranked lists)
3. Loads queries with nuggets
4. Computes nugget-level metrics: α-nDCG, Coverage, and Recall
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_queries(queries_file: Path) -> Dict:
    """Load queries with nuggets."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = str(data['query_id'])
            queries[query_id] = {
                'nuggets': data['nuggets'],
                'question': data['question'],
                'answer': data['answer']
            }
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def load_assessment_results(assessment_file: Path) -> Dict:
    """
    Load assessment results and convert to qrels format.
    
    Returns:
        qrels_nuggets: {query_id: {nugget_idx: {doc_id: 1/0}}}
        qrels_docs: {query_id: set of relevant doc_ids}
    """
    qrels_nuggets = defaultdict(lambda: defaultdict(dict))
    qrels_docs = defaultdict(set)
    
    with open(assessment_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = str(data['query_id'])
            doc_id = data['doc_id']
            nugget_support = data.get('nugget_support_details', [])
            
            # For each nugget this doc supports, mark as relevant
            # nugget_support can be either list of ints or list of dicts
            for item in nugget_support:
                if isinstance(item, dict):
                    nugget_idx = item.get('nugget_index', item.get('nugget_idx'))
                else:
                    nugget_idx = item
                
                if nugget_idx is not None:
                    qrels_nuggets[query_id][str(nugget_idx)][doc_id] = 1
                    qrels_docs[query_id].add(doc_id)
    
    logger.info(f"Loaded assessments for {len(qrels_nuggets)} queries")
    return dict(qrels_nuggets), {k: list(v) for k, v in qrels_docs.items()}


def load_retrieval_results(results_file: Path, query_field: str) -> Dict:
    """
    Load retrieval results and convert to ranked lists with scores.
    
    Returns:
        {query_id: {doc_id: score}}
    """
    results = {}
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Filter by query_field
            if data.get('query_field') != query_field:
                continue
                
            query_id = str(data['query_id'])
            doc_id = data['doc_id']
            score = data['score']
            
            if query_id not in results:
                results[query_id] = {}
            
            # Store this doc with its score
            results[query_id][doc_id] = score
    
    logger.info(f"Loaded results for {len(results)} queries (field={query_field})")
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
    ideal_ranking = min(k, num_relevant) * [1.0]
    if len(ideal_ranking) < k:
        ideal_ranking += (k - len(ideal_ranking)) * [-alpha]
    
    idcg = sum(
        gain / np.log2(rank + 1)
        for rank, gain in enumerate(ideal_ranking, start=1)
    )
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_coverage(
    retrieved_docs: List[str],
    relevant_docs: Set[str],
    k: int = 20
) -> float:
    """Compute Coverage@k: fraction of relevant docs found in top-k."""
    
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
    """Compute Recall@k: same as Coverage@k."""
    return compute_coverage(retrieved_docs, relevant_docs, k=k)


def evaluate_nugget_level(
    queries: Dict,
    qrels_nuggets: Dict,
    retrieval_results: Dict,
    k_values: List[int] = [10, 20, 50]
) -> Tuple[Dict, List]:
    """
    Evaluate retrieval at nugget level.
    
    For each query:
      - For each nugget: compute metrics based on docs that support that nugget
      - Average across nuggets for query-level metrics
    """
    
    all_metrics = defaultdict(lambda: defaultdict(list))
    query_level_results = []
    
    for query_id in tqdm(sorted(queries.keys()), desc="Evaluating"):
        if query_id not in retrieval_results:
            logger.warning(f"No retrieval results for query {query_id}")
            continue
        
        if query_id not in qrels_nuggets:
            logger.warning(f"No assessments for query {query_id}")
            continue
        
        # Get ranked list of retrieved docs
        retrieved_scores = retrieval_results[query_id]
        retrieved_docs = sorted(retrieved_scores.keys(), key=lambda x: retrieved_scores[x], reverse=True)
        
        nuggets = queries[query_id]['nuggets']
        nugget_qrels = qrels_nuggets[query_id]
        
        # Track metrics for this query across nuggets
        query_metrics = defaultdict(list)
        
        # For each nugget, compute metrics
        for nugget_idx in range(len(nuggets)):
            nugget_id = str(nugget_idx)
            
            # Get docs that support this nugget
            relevant_docs = set(nugget_qrels.get(nugget_id, {}).keys())
            
            if len(relevant_docs) == 0:
                continue
            
            # Compute metrics for this nugget
            for k in k_values:
                if k == 10:
                    ndcg = compute_alpha_ndcg(retrieved_docs, relevant_docs, k=10)
                    query_metrics[f'alpha_ndcg@{k}'].append(ndcg)
                    all_metrics[f'alpha_ndcg@{k}'][query_id].append(ndcg)
                elif k == 20:
                    cov = compute_coverage(retrieved_docs, relevant_docs, k=20)
                    query_metrics[f'coverage@{k}'].append(cov)
                    all_metrics[f'coverage@{k}'][query_id].append(cov)
                elif k == 50:
                    rec = compute_recall(retrieved_docs, relevant_docs, k=50)
                    query_metrics[f'recall@{k}'].append(rec)
                    all_metrics[f'recall@{k}'][query_id].append(rec)
        
        # Average across nuggets for this query
        query_result = {'query_id': query_id}
        for metric_name, values in query_metrics.items():
            if values:
                query_result[metric_name] = np.mean(values)
        
        if len(query_result) > 1:  # Has at least one metric
            query_level_results.append(query_result)
    
    # Compute overall metrics (micro-average across all nuggets)
    overall_metrics = {}
    for metric_name in all_metrics:
        all_nugget_scores = []
        for query_id, scores in all_metrics[metric_name].items():
            all_nugget_scores.extend(scores)
        
        if all_nugget_scores:
            overall_metrics[metric_name] = {
                'mean': float(np.mean(all_nugget_scores)),
                'median': float(np.median(all_nugget_scores)),
                'std': float(np.std(all_nugget_scores)),
                'count': len(all_nugget_scores)
            }
    
    return overall_metrics, query_level_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval with nugget-level judgments")
    parser.add_argument("--queries", type=str, required=True, help="Path to queries JSONL file")
    parser.add_argument("--assessment", type=str, required=True, help="Path to assessment results JSONL")
    parser.add_argument("--retrieval", type=str, required=True, help="Path to retrieval results JSONL")
    parser.add_argument("--query-field", type=str, required=True, help="Query field to evaluate")
    parser.add_argument("--output", type=str, required=True, help="Output path for evaluation results")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 20, 50], help="K values for metrics")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    queries = load_queries(Path(args.queries))
    qrels_nuggets, qrels_docs = load_assessment_results(Path(args.assessment))
    retrieval_results = load_retrieval_results(Path(args.retrieval), args.query_field)
    
    # Evaluate
    logger.info("Computing metrics...")
    overall_metrics, query_level_results = evaluate_nugget_level(
        queries, qrels_nuggets, retrieval_results, args.k_values
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'overall_metrics': overall_metrics,
        'query_level_results': query_level_results,
        'config': {
            'queries_file': args.queries,
            'assessment_file': args.assessment,
            'retrieval_file': args.retrieval,
            'query_field': args.query_field,
            'k_values': args.k_values
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric_name, stats in sorted(overall_metrics.items()):
        print(f"\n{metric_name}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Count:  {stats['count']}")
    print("="*60)


if __name__ == "__main__":
    main()
