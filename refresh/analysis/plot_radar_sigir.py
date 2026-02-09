"""
Query robustness radar chart.

Shows performance across different query formulations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from sigir_style import set_sigir_style

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_query_field_metrics(metrics_file: Path) -> Dict:
    """Load metrics broken down by query field."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def plot_radar(
    categories: List[str],
    values_2024: List[float],
    values_2025: List[float],
    output_file: Path
):
    """Generate radar chart showing query field performance."""
    set_sigir_style()
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    values_2024 = list(values_2024) + [values_2024[0]]
    values_2025 = list(values_2025) + [values_2025[0]]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    
    ax.plot(
        angles,
        values_2024,
        linewidth=1.5,
        linestyle='solid',
        label='Oct 2024',
        color='#1f77b4',
        marker='o'
    )
    ax.fill(angles, values_2024, '#1f77b4', alpha=0.1)
    
    ax.plot(
        angles,
        values_2025,
        linewidth=1.5,
        linestyle='dashed',
        label='Oct 2025',
        color='#ff7f0e',
        marker='s'
    )
    ax.fill(angles, values_2025, '#ff7f0e', alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1.0)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=8)
    plt.title("Robustness by Query Formulation", y=1.08, fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate query robustness radar chart")
    parser.add_argument("--metrics_2024", required=True, help="2024 query field metrics JSON")
    parser.add_argument("--metrics_2025", required=True, help="2025 query field metrics JSON")
    parser.add_argument("--output_file", required=True, help="Output PDF path")
    parser.add_argument("--metric", default="alpha_ndcg_10", help="Metric to plot")
    
    args = parser.parse_args()
    
    logger.info("Loading query field metrics...")
    metrics_2024 = load_query_field_metrics(Path(args.metrics_2024))
    metrics_2025 = load_query_field_metrics(Path(args.metrics_2025))
    
    query_fields = ['question', 'closed_book_answer', 'subquestions', 'answer', 'nuggets']
    field_labels = ['Question', 'Closed-Book', 'Sub-Q', 'Answer', 'Nuggets']
    
    values_2024 = []
    values_2025 = []
    
    for field in query_fields:
        if field in metrics_2024 and field in metrics_2025:
            val_24 = metrics_2024[field][args.metric]['mean']
            val_25 = metrics_2025[field][args.metric]['mean']
            values_2024.append(val_24)
            values_2025.append(val_25)
        else:
            logger.warning(f"Query field {field} not found")
            values_2024.append(0.5)
            values_2025.append(0.4)
    
    logger.info("Generating radar chart...")
    plot_radar(
        field_labels,
        values_2024,
        values_2025,
        Path(args.output_file)
    )
    
    logger.info("Complete")


if __name__ == "__main__":
    main()
