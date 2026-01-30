"""
Evaluation script using FreshStack's EvaluateRetrieval class.

Computes α-nDCG@k, Coverage@k, and Recall@k from assessed retrieval results.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from freshstack import util
from freshstack.datasets import DataLoader
from freshstack.retrieval.evaluation import EvaluateRetrieval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_assessed_results(jsonl_file: Path) -> dict[str, dict[str, float]]:
    """
    Load assessed retrieval results from JSONL.
    
    Input format (one line per retrieved doc):
        {"query_id": "q1", "doc_id": "d1", "score": 0.9, "nugget_level_judgment": 1, "supported_nuggets": [0, 2]}
    
    Output format (for FreshStack evaluator):
        {"q1": {"d1": 0.9, "d2": 0.8}, "q2": {...}}
    
    Uses the original retrieval score (not nugget_level_judgment) for ranking.
    """
    logger.info(f"Loading assessed results from {jsonl_file}")
    results = defaultdict(dict)
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                query_id = result['query_id']
                doc_id = result['doc_id']
                # Use original retrieval score for ranking (not binary judgment)
                score = float(result.get('score', 0))
                results[query_id][doc_id] = score
    
    logger.info(f"Loaded results for {len(results)} queries")
    return dict(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval results using FreshStack metrics")
    parser.add_argument(
        "--assessed_file",
        type=str,
        required=True,
        help="Path to assessed retrieval results (JSONL format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for metrics JSON"
    )
    parser.add_argument(
        "--queries_repo",
        type=str,
        default="freshstack/queries-oct-2024",
        help="HuggingFace queries repository"
    )
    parser.add_argument(
        "--corpus_repo",
        type=str,
        default="freshstack/corpus-oct-2024",
        help="HuggingFace corpus repository"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="langchain",
        help="Topic name (e.g., langchain)"
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="List of k values for evaluation metrics (α-nDCG limited to k≤20)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load assessed retrieval results
    retrieval_results = load_assessed_results(Path(args.assessed_file))
    
    # Load qrels using FreshStack's DataLoader
    logger.info(f"Loading qrels for topic: {args.topic}")
    dataloader = DataLoader(
        queries_repo=args.queries_repo,
        corpus_repo=args.corpus_repo,
        topic=args.topic
    )
    
    # Load corpus, queries, and nuggets
    corpus, queries, nuggets = dataloader.load(split=args.split)
    logger.info(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(nuggets)} nuggets")
    
    # Load qrels (three required structures)
    qrels_nuggets, qrels_query, query_to_nuggets = dataloader.load_qrels(split=args.split)
    logger.info(f"Loaded qrels: {len(qrels_nuggets)} nugget qrels, {len(qrels_query)} query qrels")
    
    # Initialize evaluator
    evaluator = EvaluateRetrieval(k_values=args.k_values)
    
    # Run evaluation
    logger.info("Running FreshStack evaluation...")
    alpha_ndcg, coverage, recall = evaluator.evaluate(
        qrels_nuggets=qrels_nuggets,
        query_to_nuggets=query_to_nuggets,
        qrels_query=qrels_query,
        results=retrieval_results,
    )
    
    # Log results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    logger.info("\nα-nDCG (Diversity-Aware):")
    for metric, value in alpha_ndcg.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nCoverage (Nugget Coverage):")
    for metric, value in coverage.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("\nRecall (Document Recall):")
    for metric, value in recall.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save results using FreshStack's official format
    util.save_results(
        output_file=str(output_path),
        alpha_ndcg=alpha_ndcg,
        coverage=coverage,
        recall=recall
    )
    
    logger.info(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
