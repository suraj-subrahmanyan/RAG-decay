"""
Retrieval overlap heatmap using Jaccard similarity.

Computes and visualizes overlap between different retrieval methods.
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sigir_style import set_sigir_style

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_retrieval_results(results_file: Path, top_k: int = 20) -> Dict[str, Set[str]]:
    """Load retrieval results and return doc sets per query."""
    query_docs = defaultdict(set)
    
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            query_field = data.get('query_field', 'unknown')
            doc_id = data['doc_id']
            rank = data['rank']
            
            if rank <= top_k:
                key = f"{query_id}_{query_field}"
                query_docs[key].add(doc_id)
    
    return query_docs


def compute_jaccard_matrix(
    method_results: Dict[str, Dict[str, Set[str]]],
    methods: List[str]
) -> np.ndarray:
    """
    Compute Jaccard similarity matrix between methods.
    
    Args:
        method_results: Dict mapping method name to query->docs dict
        methods: Ordered list of method names
        
    Returns:
        NxN Jaccard similarity matrix
    """
    n = len(methods)
    matrix = np.zeros((n, n))
    
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i == j:
                matrix[i, j] = 1.0
            else:
                results_i = method_results[method_i]
                results_j = method_results[method_j]
                
                common_queries = set(results_i.keys()) & set(results_j.keys())
                
                if not common_queries:
                    continue
                
                jaccards = []
                for query_key in common_queries:
                    docs_i = results_i[query_key]
                    docs_j = results_j[query_key]
                    
                    if len(docs_i) == 0 and len(docs_j) == 0:
                        continue
                    
                    intersection = len(docs_i & docs_j)
                    union = len(docs_i | docs_j)
                    
                    if union > 0:
                        jaccards.append(intersection / union)
                
                if jaccards:
                    matrix[i, j] = np.mean(jaccards)
    
    return matrix


def plot_overlap_heatmap(
    jaccard_matrix: np.ndarray,
    methods: List[str],
    output_file: Path,
    corpus_version: str,
    query_field: str
):
    """Generate overlap heatmap."""
    set_sigir_style()
    
    # Slice to show only lower triangle without diagonal
    # Rows: BGE, E5, Qwen (skip BM25)
    # Cols: BM25, BGE, E5 (skip Qwen)
    plot_data = jaccard_matrix[1:, :-1]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(
        plot_data,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=0.3,  # Match actual data range for better contrast
        square=True,
        linewidths=1.0,
        linecolor='white',
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8, "label": "Jaccard Similarity"},
        xticklabels=[m.upper() for m in methods[:-1]],
        yticklabels=[m.upper() for m in methods[1:]],
        ax=ax,
        annot_kws={"fontsize": 14, "fontweight": "bold"}
    )
    
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    title = f"Retrieval Overlap (Top-20)"
    plt.title(title, pad=20, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate SIGIR overlap heatmap")
    parser.add_argument("--results_dir", required=True, help="Retrieval results directory")
    parser.add_argument("--corpus_version", required=True, help="oct_2024 or oct_2025")
    parser.add_argument("--query_field", default="answer", help="Query field to analyze")
    parser.add_argument("--output_file", required=True, help="Output PDF path")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k documents")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    methods = ['bm25', 'bge', 'e5', 'qwen']
    
    logger.info("Loading retrieval results...")
    method_results = {}
    
    for method in methods:
        results_file = results_dir / args.corpus_version / f"{method}.jsonl"
        if results_file.exists():
            method_results[method] = load_retrieval_results(results_file, args.top_k)
            logger.info(f"  {method}: {len(method_results[method])} queries")
        else:
            logger.warning(f"  {method}: File not found")
    
    if len(method_results) < 2:
        logger.error("Need at least 2 methods for comparison")
        return
    
    logger.info("Computing Jaccard matrix...")
    jaccard_matrix = compute_jaccard_matrix(method_results, methods)
    
    logger.info("Generating heatmap...")
    plot_overlap_heatmap(
        jaccard_matrix,
        methods,
        Path(args.output_file),
        args.corpus_version,
        args.query_field
    )
    
    logger.info("Complete")


if __name__ == "__main__":
    main()
