"""
Retrieval overlap heatmap showing Jaccard similarity between different methods.

Justifies fusion by demonstrating that different retrieval methods find different documents.
Uses only raw retrieval results - no assessment data required.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Set, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Style configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("white")


def load_retrieval_results(results_file: Path, query_field: str, top_k: int = 20) -> Dict[str, Set[str]]:
    """
    Load retrieval results for a specific query field.
    
    Args:
        results_file: JSONL file with retrieval results
        query_field: Which query field to filter on (e.g., "question", "nuggets")
        top_k: Number of top documents to consider
    
    Returns:
        Dict mapping query_id -> set of top-k doc_ids
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
            rank = data['rank']
            
            # Only keep top-k
            if rank > top_k:
                continue
            
            if query_id not in results:
                results[query_id] = set()
            
            results[query_id].add(doc_id)
    
    return results


def compute_jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """Compute Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_overlap_matrix(
    all_results: Dict[str, Dict[str, Set[str]]],
    methods: List[str]
) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity matrix across methods.
    
    Args:
        all_results: {method: {query_id: set(doc_ids)}}
        methods: List of method names in order
    
    Returns:
        NxN numpy array of Jaccard similarities
    """
    n = len(methods)
    matrix = np.zeros((n, n))
    
    # Get union of all query_ids
    all_query_ids = set()
    for method_results in all_results.values():
        all_query_ids.update(method_results.keys())
    
    all_query_ids = sorted(all_query_ids)
    
    # Compute pairwise similarities
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i == j:
                matrix[i, j] = 1.0
            else:
                # Average Jaccard across all queries
                similarities = []
                for query_id in all_query_ids:
                    set_i = all_results[method_i].get(query_id, set())
                    set_j = all_results[method_j].get(query_id, set())
                    
                    # Skip if both are empty
                    if len(set_i) == 0 and len(set_j) == 0:
                        continue
                    
                    sim = compute_jaccard_similarity(set_i, set_j)
                    similarities.append(sim)
                
                if similarities:
                    matrix[i, j] = np.mean(similarities)
                else:
                    matrix[i, j] = 0.0
    
    return matrix


def plot_overlap_heatmap(
    matrix: np.ndarray,
    methods: List[str],
    corpus_version: str,
    query_field: str,
    top_k: int,
    output_dir: Path
) -> None:
    """Plot heatmap of retrieval overlap."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=20)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_yticklabels(methods)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(methods)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title(f'Retrieval Overlap (Jaccard Similarity)\n{corpus_version} | {query_field} | Top-{top_k}',
                 fontsize=12, fontweight='bold', pad=20)
    
    # Add interpretation text
    interpretation = """
    Low overlap (red) = Methods retrieve different documents → Fusion beneficial
    High overlap (green) = Methods retrieve similar documents → Fusion redundant
    """
    fig.text(0.5, 0.02, interpretation, ha='center', fontsize=9, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / f'retrieval_overlap_{corpus_version}_{query_field}_top{top_k}.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_overlap_statistics(matrix: np.ndarray, methods: List[str]) -> None:
    """Print overlap statistics."""
    
    print("\n" + "=" * 80)
    print("RETRIEVAL OVERLAP ANALYSIS")
    print("=" * 80)
    
    print("\nJaccard Similarity Matrix:")
    print(f"{'':15}", end='')
    for method in methods:
        print(f"{method:>12}", end='')
    print()
    
    for i, method_i in enumerate(methods):
        print(f"{method_i:15}", end='')
        for j in range(len(methods)):
            print(f"{matrix[i, j]:12.3f}", end='')
        print()
    
    # Compute average pairwise similarity (excluding diagonal)
    n = len(methods)
    pairwise_sims = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_sims.append(matrix[i, j])
    
    if pairwise_sims:
        avg_sim = np.mean(pairwise_sims)
        print(f"\nAverage pairwise similarity: {avg_sim:.3f}")
        
        if avg_sim < 0.3:
            print("→ LOW overlap: Methods find very different documents (Fusion highly beneficial)")
        elif avg_sim < 0.6:
            print("→ MEDIUM overlap: Methods have some uniqueness (Fusion beneficial)")
        else:
            print("→ HIGH overlap: Methods find similar documents (Fusion less beneficial)")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze retrieval overlap between methods')
    parser.add_argument('--results_dir', required=True, help='Directory containing retrieval results')
    parser.add_argument('--corpus_version', required=True, help='Corpus version (oct_2024 or oct_2025)')
    parser.add_argument('--query_field', default='question', help='Query field to analyze')
    parser.add_argument('--methods', nargs='+', required=True, help='Methods to compare (e.g., bge e5 qwen bm25)')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top documents to consider')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) / args.corpus_version
    
    print(f"Loading retrieval results for {args.corpus_version}...")
    all_results = {}
    
    for method in args.methods:
        results_file = results_dir / f"{method}.jsonl"
        
        if not results_file.exists():
            print(f"  WARNING: {results_file} not found, skipping {method}")
            continue
        
        print(f"  Loading {method}...")
        method_results = load_retrieval_results(results_file, args.query_field, args.top_k)
        all_results[method] = method_results
        print(f"    → {len(method_results)} queries")
    
    if len(all_results) < 2:
        print("ERROR: Need at least 2 methods to compute overlap")
        return
    
    methods = sorted(all_results.keys())
    
    print("\nComputing overlap matrix...")
    matrix = compute_overlap_matrix(all_results, methods)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating heatmap...")
    plot_overlap_heatmap(matrix, methods, args.corpus_version, args.query_field, 
                        args.top_k, output_dir)
    
    print_overlap_statistics(matrix, methods)
    
    print(f"\n✓ Heatmap saved to {output_dir}")


if __name__ == '__main__':
    main()
