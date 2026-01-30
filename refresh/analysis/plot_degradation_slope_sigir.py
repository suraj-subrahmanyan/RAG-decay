"""
Temporal degradation slope chart.

Shows performance changes from Oct 2024 to Oct 2025.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

from sigir_style import set_sigir_style

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_metrics(metrics_file: Path) -> Dict:
    """Load evaluation metrics from JSON."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_slope(
    methods: List[str],
    scores_2024: List[float],
    scores_2025: List[float],
    output_file: Path,
    metric_name: str = "nDCG@10"
):
    """Generate slope chart showing performance changes."""
    set_sigir_style()
    
    fig, ax = plt.subplots(figsize=(5, 4))
    
    for i, method in enumerate(methods):
        color = '#d62728' if method.lower() == 'fusion' else '#7f7f7f'
        alpha = 1.0 if method.lower() == 'fusion' else 0.6
        linewidth = 2.5 if method.lower() == 'fusion' else 1.5
        
        ax.plot(
            [0, 1],
            [scores_2024[i], scores_2025[i]],
            marker='o',
            color=color,
            alpha=alpha,
            linewidth=linewidth
        )
        
        if method.lower() in ['fusion', 'qwen', 'bm25']:
            ax.text(
                -0.05,
                scores_2024[i],
                f"{method.upper()}",
                ha='right',
                va='center',
                fontsize=9,
                color=color,
                fontweight='bold'
            )
        
        drop = ((scores_2025[i] - scores_2024[i]) / scores_2024[i]) * 100
        ax.text(
            1.05,
            scores_2025[i],
            f"{drop:.1f}%",
            ha='left',
            va='center',
            fontsize=9,
            color=color
        )
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Oct 2024\n(In-Domain)', 'Oct 2025\n(Temporal Shift)'])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(metric_name)
    ax.set_title("Performance Degradation over 1 Year", fontsize=11, fontweight='bold')
    
    sns.despine(trim=True)
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate temporal degradation slope chart")
    parser.add_argument("--metrics_2024", required=True, help="2024 metrics JSON")
    parser.add_argument("--metrics_2025", required=True, help="2025 metrics JSON")
    parser.add_argument("--output_file", required=True, help="Output PDF path")
    parser.add_argument("--metric", default="alpha_ndcg_10", help="Metric to plot")
    
    args = parser.parse_args()
    
    logger.info("Loading metrics...")
    metrics_2024 = load_metrics(Path(args.metrics_2024))
    metrics_2025 = load_metrics(Path(args.metrics_2025))
    
    methods = ['bm25', 'bge', 'e5', 'qwen', 'fusion']
    scores_2024 = []
    scores_2025 = []
    
    for method in methods:
        if method in metrics_2024 and method in metrics_2025:
            score_24 = metrics_2024[method][args.metric]['mean']
            score_25 = metrics_2025[method][args.metric]['mean']
            scores_2024.append(score_24)
            scores_2025.append(score_25)
        else:
            logger.warning(f"Method {method} not found in metrics")
            scores_2024.append(0)
            scores_2025.append(0)
    
    logger.info("Generating slope chart...")
    plot_slope(
        methods,
        scores_2024,
        scores_2025,
        Path(args.output_file),
        metric_name="α-nDCG@10"
    )
    
    logger.info("Complete")


if __name__ == "__main__":
    main()
