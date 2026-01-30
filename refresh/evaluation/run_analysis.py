"""
Analysis and Visualization for Temporal Decay Study
====================================================

Generates comprehensive analysis and visualizations:
- Temporal decay analysis (2024 vs 2025 performance)
- Method comparison (BM25 vs BGE vs Fusion)
- Query field comparison
- Statistical significance tests

Reference: FreshStack paper Section 5

Usage:
    python run_analysis.py --eval_results evaluation_results/evaluation_metrics.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_evaluation_results(eval_file: Path) -> List[Dict]:
    """Load evaluation metrics."""
    logger.info(f"Loading evaluation results from {eval_file}")
    
    with open(eval_file, "r") as f:
        results = json.load(f)
    
    logger.info(f"Loaded results for {len(results)} methods")
    return results


def parse_method_name(method: str) -> Dict[str, str]:
    """
    Parse method name into components.
    
    Example: "bm25_2024_query_text_results" -> 
        {retrieval: "bm25", corpus: "2024", query_field: "query_text"}
    """
    parts = method.replace("_results", "").split("_")
    
    return {
        "retrieval": parts[0],  # bm25, bge, fusion
        "corpus": parts[1],     # 2024, 2025
        "query_field": "_".join(parts[2:]) if len(parts) > 2 else "query_text"
    }


def calculate_temporal_decay(results: List[Dict]) -> Dict:
    """
    Calculate temporal decay: performance drop from 2024 to 2025.
    
    Returns:
        Dict with decay statistics for each method/query_field
    """
    logger.info("\nAnalyzing temporal decay...")
    
    # Group by method and query field
    grouped = defaultdict(dict)
    
    for result in results:
        parsed = parse_method_name(result["method"])
        key = f"{parsed['retrieval']}_{parsed['query_field']}"
        corpus = parsed["corpus"]
        
        grouped[key][corpus] = result["recall"]
    
    # Calculate decay
    decay_stats = {}
    
    for key, corpus_results in grouped.items():
        if "2024" in corpus_results and "2025" in corpus_results:
            recall_2024 = corpus_results["2024"]
            recall_2025 = corpus_results["2025"]
            
            decay = {}
            for k in recall_2024.keys():
                drop = recall_2024[k] - recall_2025[k]
                pct_drop = (drop / recall_2024[k] * 100) if recall_2024[k] > 0 else 0
                decay[k] = {
                    "absolute": drop,
                    "percentage": pct_drop,
                    "2024": recall_2024[k],
                    "2025": recall_2025[k]
                }
            
            decay_stats[key] = decay
            
            logger.info(f"\n  {key}:")
            logger.info(f"    R@100: {recall_2024[100]:.4f} -> {recall_2025[100]:.4f} "
                       f"(-{decay[100]['percentage']:.1f}%)")
    
    return decay_stats


def plot_temporal_decay(decay_stats: Dict, output_dir: Path):
    """Plot temporal decay for all methods."""
    logger.info("\nCreating temporal decay plot...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    methods = list(decay_stats.keys())
    k_values = [10, 20, 50, 100]
    x = np.arange(len(methods))
    width = 0.2
    
    for i, k in enumerate(k_values):
        decay_pcts = [decay_stats[method][k]["percentage"] for method in methods]
        ax.bar(x + i * width, decay_pcts, width, label=f'k={k}')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    ax.set_title('Temporal Decay Analysis: Performance Drop from Oct 2024 to Oct 2025', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "temporal_decay.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_method_comparison(results: List[Dict], output_dir: Path):
    """Compare different retrieval methods."""
    logger.info("\nCreating method comparison plot...")
    
    # Group by corpus version
    results_2024 = [r for r in results if "2024" in r["method"]]
    results_2025 = [r for r in results if "2025" in r["method"]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, results_subset, corpus in [(ax1, results_2024, "2024"), 
                                        (ax2, results_2025, "2025")]:
        methods = []
        recall_10 = []
        recall_100 = []
        
        for result in results_subset:
            parsed = parse_method_name(result["method"])
            methods.append(f"{parsed['retrieval']}")
            recall_10.append(result["recall"][10])
            recall_100.append(result["recall"][100])
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, recall_10, width, label='Recall@10', alpha=0.8)
        ax.bar(x + width/2, recall_100, width, label='Recall@100', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
        ax.set_title(f'Oct {corpus} Corpus', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "method_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def plot_recall_curves(results: List[Dict], output_dir: Path):
    """Plot Recall@k curves for all methods."""
    logger.info("\nCreating recall curves...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, corpus in [(ax1, "2024"), (ax2, "2025")]:
        corpus_results = [r for r in results if corpus in r["method"]]
        
        for result in corpus_results:
            parsed = parse_method_name(result["method"])
            method_name = parsed["retrieval"]
            
            k_values = sorted(result["recall"].keys())
            recall_values = [result["recall"][k] for k in k_values]
            
            ax.plot(k_values, recall_values, marker='o', label=method_name, linewidth=2)
        
        ax.set_xlabel('k (cutoff)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Recall@k', fontsize=12, fontweight='bold')
        ax.set_title(f'Oct {corpus} Corpus', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / "recall_curves.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def generate_latex_table(decay_stats: Dict, output_dir: Path):
    """Generate LaTeX table for paper."""
    logger.info("\nGenerating LaTeX table...")
    
    output_file = output_dir / "results_table.tex"
    
    with open(output_file, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Temporal Decay Analysis: Recall@100 Performance}\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Method & Oct 2024 & Oct 2025 & Decay (\\%) \\\\\n")
        f.write("\\midrule\n")
        
        for method, decay in sorted(decay_stats.items()):
            r2024 = decay[100]["2024"]
            r2025 = decay[100]["2025"]
            decay_pct = decay[100]["percentage"]
            f.write(f"{method} & {r2024:.3f} & {r2025:.3f} & {decay_pct:.1f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    logger.info(f"Saved: {output_file}")


def save_summary_stats(results: List[Dict], decay_stats: Dict, output_dir: Path):
    """Save summary statistics."""
    logger.info("\nSaving summary statistics...")
    
    summary = {
        "num_methods": len(results),
        "temporal_decay": decay_stats,
        "best_method_2024": max(
            [r for r in results if "2024" in r["method"]],
            key=lambda x: x["recall"][100]
        )["method"],
        "best_method_2025": max(
            [r for r in results if "2025" in r["method"]],
            key=lambda x: x["recall"][100]
        )["method"],
    }
    
    output_file = output_dir / "analysis_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate analysis and visualizations")
    parser.add_argument(
        "--eval_results",
        type=str,
        required=True,
        help="Path to evaluation_metrics.json"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_results",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results = load_evaluation_results(Path(args.eval_results))
    
    # Calculate temporal decay
    decay_stats = calculate_temporal_decay(results)
    
    # Generate visualizations
    plot_temporal_decay(decay_stats, output_dir)
    plot_method_comparison(results, output_dir)
    plot_recall_curves(results, output_dir)
    
    # Generate LaTeX table
    generate_latex_table(decay_stats, output_dir)
    
    # Save summary
    save_summary_stats(results, decay_stats, output_dir)
    
    logger.info("\nAnalysis complete")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
