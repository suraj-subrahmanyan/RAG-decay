"""
Document length distribution comparison between Oct 2024 and Oct 2025 corpora.

Shows how document sizes changed over time (possible cause of retrieval degradation).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Style configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def load_document_lengths(corpus_file: Path) -> list:
    """Load document text lengths from corpus."""
    lengths = []
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            text_len = len(doc['text'])
            lengths.append(text_len)
    
    return lengths


def plot_length_distributions(lengths_2024: list, lengths_2025: list, output_dir: Path) -> None:
    """Plot side-by-side violin plots of document length distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Violin plot
    ax1 = axes[0]
    data_violin = [lengths_2024, lengths_2025]
    positions = [1, 2]
    colors = ['#2196F3', '#FF5722']
    
    parts = ax1.violinplot(data_violin, positions=positions, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax1.set_xticks(positions)
    ax1.set_xticklabels(['Oct 2024', 'Oct 2025'])
    ax1.set_ylabel('Document Length (characters)')
    ax1.set_title('Document Length Distribution')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    stats_text = f"""
    Oct 2024:
    Mean: {np.mean(lengths_2024):.0f}
    Median: {np.median(lengths_2024):.0f}
    Std: {np.std(lengths_2024):.0f}
    
    Oct 2025:
    Mean: {np.mean(lengths_2025):.0f}
    Median: {np.median(lengths_2025):.0f}
    Std: {np.std(lengths_2025):.0f}
    """
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Box plot for comparison
    ax2 = axes[1]
    bp = ax2.boxplot([lengths_2024, lengths_2025], 
                       labels=['Oct 2024', 'Oct 2025'],
                       patch_artist=True,
                       widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Document Length (characters)')
    ax2.set_title('Document Length Box Plot')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Document Length Analysis: Temporal Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'document_length_distribution.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_length_histogram(lengths_2024: list, lengths_2025: list, output_dir: Path) -> None:
    """Plot overlapping histograms of document lengths."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Use log scale for better visualization
    bins = np.logspace(np.log10(100), np.log10(max(max(lengths_2024), max(lengths_2025))), 50)
    
    ax.hist(lengths_2024, bins=bins, alpha=0.6, label='Oct 2024', color='#2196F3', edgecolor='black')
    ax.hist(lengths_2025, bins=bins, alpha=0.6, label='Oct 2025', color='#FF5722', edgecolor='black')
    
    ax.set_xscale('log')
    ax.set_xlabel('Document Length (characters, log scale)')
    ax.set_ylabel('Frequency')
    ax.set_title('Document Length Distribution (Histogram)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    output_path = output_dir / 'document_length_histogram.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_statistics(lengths_2024: list, lengths_2025: list) -> None:
    """Print summary statistics."""
    
    print("\n" + "=" * 80)
    print("DOCUMENT LENGTH STATISTICS")
    print("=" * 80)
    
    print(f"\nOct 2024 ({len(lengths_2024):,} documents):")
    print(f"  Mean:   {np.mean(lengths_2024):,.0f} characters")
    print(f"  Median: {np.median(lengths_2024):,.0f} characters")
    print(f"  Std:    {np.std(lengths_2024):,.0f} characters")
    print(f"  Min:    {np.min(lengths_2024):,.0f} characters")
    print(f"  Max:    {np.max(lengths_2024):,.0f} characters")
    print(f"  Q1:     {np.percentile(lengths_2024, 25):,.0f} characters")
    print(f"  Q3:     {np.percentile(lengths_2024, 75):,.0f} characters")
    
    print(f"\nOct 2025 ({len(lengths_2025):,} documents):")
    print(f"  Mean:   {np.mean(lengths_2025):,.0f} characters")
    print(f"  Median: {np.median(lengths_2025):,.0f} characters")
    print(f"  Std:    {np.std(lengths_2025):,.0f} characters")
    print(f"  Min:    {np.min(lengths_2025):,.0f} characters")
    print(f"  Max:    {np.max(lengths_2025):,.0f} characters")
    print(f"  Q1:     {np.percentile(lengths_2025, 25):,.0f} characters")
    print(f"  Q3:     {np.percentile(lengths_2025, 75):,.0f} characters")
    
    # Comparative metrics
    mean_change = ((np.mean(lengths_2025) / np.mean(lengths_2024)) - 1) * 100
    median_change = ((np.median(lengths_2025) / np.median(lengths_2024)) - 1) * 100
    
    print(f"\nChange:")
    print(f"  Mean:   {mean_change:+.1f}%")
    print(f"  Median: {median_change:+.1f}%")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze document length distributions')
    parser.add_argument('--corpus_2024', required=True, help='Oct 2024 corpus JSONL')
    parser.add_argument('--corpus_2025', required=True, help='Oct 2025 corpus JSONL')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots')
    args = parser.parse_args()
    
    print("Loading Oct 2024 corpus...")
    lengths_2024 = load_document_lengths(Path(args.corpus_2024))
    
    print("Loading Oct 2025 corpus...")
    lengths_2025 = load_document_lengths(Path(args.corpus_2025))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    plot_length_distributions(lengths_2024, lengths_2025, output_dir)
    plot_length_histogram(lengths_2024, lengths_2025, output_dir)
    
    print_statistics(lengths_2024, lengths_2025)
    
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
