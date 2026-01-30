"""
Compare retrieval performance between Oct 2024 and Oct 2025 corpora.

Analyzes how temporal corpus drift affects retrieval metrics.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def load_metrics(metrics_file: Path) -> Dict:
    """Load evaluation metrics from JSON."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def compare_metrics(
    metrics_2024: Dict,
    metrics_2025: Dict,
    method: str
) -> Dict:
    """
    Compare metrics between 2024 and 2025.
    
    Returns:
        Dictionary with comparison statistics
    """
    
    comparison = {
        'method': method,
        'metrics': {}
    }
    
    for metric_name in ['alpha_ndcg_10', 'coverage_20', 'recall_50']:
        val_2024 = metrics_2024[metric_name]['mean']
        val_2025 = metrics_2025[metric_name]['mean']
        
        absolute_change = val_2025 - val_2024
        relative_change = (absolute_change / val_2024) * 100 if val_2024 > 0 else 0
        
        comparison['metrics'][metric_name] = {
            '2024': val_2024,
            '2025': val_2025,
            'absolute_change': absolute_change,
            'relative_change_pct': relative_change
        }
    
    return comparison


def create_comparison_plots(
    comparisons: List[Dict],
    output_dir: Path
) -> None:
    """Generate comparison visualizations."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [c['method'] for c in comparisons]
    metrics = ['alpha_ndcg_10', 'coverage_20', 'recall_50']
    metric_labels = ['α-nDCG@10', 'Coverage@20', 'Recall@50']
    
    # Plot 1: Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        vals_2024 = [c['metrics'][metric]['2024'] for c in comparisons]
        vals_2025 = [c['metrics'][metric]['2025'] for c in comparisons]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, vals_2024, width, label='Oct 2024', alpha=0.8)
        ax.bar(x + width/2, vals_2025, width, label='Oct 2025', alpha=0.8)
        
        ax.set_xlabel('Retrieval Method')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in methods])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison.png', bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'metric_comparison.png'}")
    plt.close()
    
    # Plot 2: Relative change heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    change_matrix = np.array([
        [c['metrics'][m]['relative_change_pct'] for m in metrics]
        for c in comparisons
    ])
    
    im = ax.imshow(change_matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metric_labels)
    ax.set_yticklabels([m.upper() for m in methods])
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{change_matrix[i, j]:.1f}%',
                          ha="center", va="center", color="black", fontsize=12)
    
    ax.set_title('Relative Performance Change (2024 to 2025)', fontsize=14, pad=20)
    plt.colorbar(im, ax=ax, label='% Change')
    plt.tight_layout()
    plt.savefig(output_dir / 'relative_change_heatmap.png', bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'relative_change_heatmap.png'}")
    plt.close()


def generate_report(
    comparisons: List[Dict],
    output_file: Path
) -> None:
    """Generate text report of comparison."""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEMPORAL CORPUS ANALYSIS: RETRIEVAL PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("Comparing retrieval performance on Oct 2024 vs Oct 2025 LangChain corpus\n\n")
        
        for comp in comparisons:
            method = comp['method'].upper()
            f.write(f"\n{'='*80}\n")
            f.write(f"{method} Retrieval\n")
            f.write(f"{'='*80}\n\n")
            
            for metric_name, data in comp['metrics'].items():
                metric_label = metric_name.replace('_', ' ').title()
                f.write(f"{metric_label}:\n")
                f.write(f"  Oct 2024: {data['2024']:.4f}\n")
                f.write(f"  Oct 2025: {data['2025']:.4f}\n")
                f.write(f"  Change:   {data['absolute_change']:+.4f} ({data['relative_change_pct']:+.2f}%)\n")
                
                if data['absolute_change'] > 0:
                    f.write(f"  Performance IMPROVED on 2025 corpus\n")
                elif data['absolute_change'] < 0:
                    f.write(f"  Performance DEGRADED on 2025 corpus\n")
                else:
                    f.write(f"  No change\n")
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Overall trend
        avg_changes = {}
        for metric in ['alpha_ndcg_10', 'coverage_20', 'recall_50']:
            avg_change = np.mean([
                c['metrics'][metric]['relative_change_pct']
                for c in comparisons
            ])
            avg_changes[metric] = avg_change
        
        f.write("Average performance change across all methods:\n")
        for metric, change in avg_changes.items():
            metric_label = metric.replace('_', ' ').title()
            f.write(f"  {metric_label}: {change:+.2f}%\n")
        
        f.write("\nInterpretation:\n")
        if all(c < -5 for c in avg_changes.values()):
            f.write("  - Significant performance DEGRADATION on 2025 corpus\n")
            f.write("  - Corpus evolution negatively impacted retrieval\n")
        elif all(c > 5 for c in avg_changes.values()):
            f.write("  - Significant performance IMPROVEMENT on 2025 corpus\n")
            f.write("  - Corpus evolution benefited retrieval quality\n")
        else:
            f.write("  - Mixed results across metrics\n")
            f.write("  - Corpus changes had varied impact\n")
    
    logger.info(f"Saved report: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare temporal retrieval performance")
    parser.add_argument("--metrics_2024_dir", required=True, help="2024 metrics directory")
    parser.add_argument("--metrics_2025_dir", required=True, help="2025 metrics directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    metrics_2024_dir = Path(args.metrics_2024_dir)
    metrics_2025_dir = Path(args.metrics_2025_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info("TEMPORAL COMPARISON: 2024 vs 2025")
    logger.info("="*80)
    
    # Load metrics for each method
    methods = ['bm25', 'bge', 'fusion']
    comparisons = []
    
    for method in methods:
        logger.info(f"\nComparing {method.upper()}...")
        
        metrics_2024 = load_metrics(metrics_2024_dir / f"{method}_oct_2024.json")
        metrics_2025 = load_metrics(metrics_2025_dir / f"{method}_oct_2025.json")
        
        comparison = compare_metrics(metrics_2024, metrics_2025, method)
        comparisons.append(comparison)
        
        # Print summary
        for metric_name, data in comparison['metrics'].items():
            logger.info(f"  {metric_name}: {data['2024']:.4f} -> {data['2025']:.4f} "
                       f"({data['relative_change_pct']:+.2f}%)")
    
    # Save full comparison
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'full_comparison.json', 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    create_comparison_plots(comparisons, output_dir)
    
    # Generate report
    logger.info("\nGenerating report...")
    generate_report(comparisons, output_dir / 'comparison_report.txt')
    
    logger.info("\n" + "="*80)
    logger.info("Comparison complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()

