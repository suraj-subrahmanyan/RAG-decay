"""
Corpus drift Sankey diagram.

Shows document retention, modification, deletion, and addition over time.
"""

import argparse
import json
import logging
from pathlib import Path

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly kaleido")

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_drift_stats(drift_file: Path) -> dict:
    """Load drift statistics from analysis output."""
    with open(drift_file, 'r') as f:
        data = json.load(f)
    return data


def plot_sankey(stats: dict, output_file: Path):
    """Generate Sankey diagram showing corpus changes."""
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly is required for Sankey diagrams")
        return
    
    total_2024 = stats.get('total_2024', 49514)
    retained = stats.get('retained', 38000)
    modified = stats.get('modified', 8000)
    deleted = total_2024 - retained - modified
    added = stats.get('added', 5000)
    
    labels = [
        "<b>Oct 2024</b><br>(Source)",
        "Retained",
        "Modified",
        "Deleted",
        "<b>Oct 2025</b><br>(Target)",
        "Added"
    ]
    
    colors = [
        "#252525",  # 2024 (dark gray)
        "#636363",  # Retained (gray)
        "#969696",  # Modified (light gray)
        "#d62728",  # Deleted (red)
        "#252525",  # 2025 (dark gray)
        "#1f77b4"   # Added (blue)
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=15,
            line=dict(color="black", width=0.5),
            label=labels,
            color=colors
        ),
        link=dict(
            source=[0, 0, 0, 1, 2, 5],
            target=[1, 2, 3, 4, 4, 4],
            value=[retained, modified, deleted, retained, modified, added],
            color=[
                "rgba(99, 99, 99, 0.4)",    # Retained flow
                "rgba(150, 150, 150, 0.4)",  # Modified flow
                "rgba(214, 39, 40, 0.4)",    # Deleted flow
                "rgba(99, 99, 99, 0.4)",     # Retained -> 2025
                "rgba(150, 150, 150, 0.4)",  # Modified -> 2025
                "rgba(31, 119, 180, 0.4)"    # Added -> 2025
            ]
        )
    )])
    
    fig.update_layout(
        font_family="Times New Roman",
        font_size=14,
        width=800,
        height=450,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.write_image(str(output_file))
    logger.info(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate corpus drift Sankey diagram")
    parser.add_argument("--drift_stats", required=True, help="Drift statistics JSON")
    parser.add_argument("--output_file", required=True, help="Output PDF path")
    
    args = parser.parse_args()
    
    logger.info("Loading drift statistics...")
    stats = load_drift_stats(Path(args.drift_stats))
    
    logger.info("Generating Sankey diagram...")
    plot_sankey(stats, Path(args.output_file))
    
    logger.info("Complete")


if __name__ == "__main__":
    main()
