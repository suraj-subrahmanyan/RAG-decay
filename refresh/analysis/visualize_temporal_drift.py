"""
Generate publication-quality visualizations for temporal corpus drift analysis.

Creates:
1. Repository chunk comparison (bar chart)
2. Document stability pie chart
3. Ground truth impact visualization
4. Temporal drift summary
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Style configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def plot_repository_comparison(analysis: dict, output_dir: Path) -> None:
    """Bar chart comparing repository sizes between 2024 and 2025."""
    
    repos = analysis['repository_changes']
    
    # Filter to top 10 repos by 2024 size
    repos = sorted(repos, key=lambda r: r['chunks_2024'], reverse=True)[:10]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(repos))
    width = 0.35
    
    bars_2024 = ax.bar(x - width/2, [r['chunks_2024'] for r in repos], width, 
                        label='Oct 2024 (Official)', color='#2196F3', alpha=0.8)
    bars_2025 = ax.bar(x + width/2, [r['chunks_2025'] for r in repos], width,
                        label='Oct 2025', color='#FF5722', alpha=0.8)
    
    ax.set_xlabel('Repository')
    ax.set_ylabel('Number of Chunks')
    ax.set_title('Corpus Size by Repository: Oct 2024 vs Oct 2025')
    ax.set_xticks(x)
    ax.set_xticklabels([r['repository'] for r in repos], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars_2024:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height):,}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'repository_comparison.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'repository_comparison.png'}")
    plt.close()


def plot_document_stability(analysis: dict, output_dir: Path) -> None:
    """Pie chart showing document retention vs churn."""
    
    d = analysis['document_stability']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart: What happened to 2024 docs?
    sizes = [d['retained'], d['removed']]
    labels = [f"Retained\n({d['retained']:,})", f"Removed\n({d['removed']:,})"]
    colors = ['#4CAF50', '#f44336']
    explode = (0, 0.05)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax1.set_title('Fate of Oct 2024 Documents')
    
    # Pie chart: Composition of 2025 corpus
    sizes2 = [d['retained'], d['added']]
    labels2 = [f"From 2024\n({d['retained']:,})", f"New in 2025\n({d['added']:,})"]
    colors2 = ['#4CAF50', '#2196F3']
    
    ax2.pie(sizes2, labels=labels2, colors=colors2, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax2.set_title('Composition of Oct 2025 Corpus')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'document_stability.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'document_stability.png'}")
    plt.close()


def plot_ground_truth_impact(analysis: dict, output_dir: Path) -> None:
    """Visualization of ground truth availability across versions."""
    
    g = analysis['ground_truth_impact']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Oct 2024', 'Oct 2025']
    found = [g['found_in_2024'], g['found_in_2025']]
    missing = [g['total_relevant_docs'] - g['found_in_2024'], 
               g['total_relevant_docs'] - g['found_in_2025']]
    
    x = np.arange(len(categories))
    width = 0.5
    
    bars_found = ax.bar(x, found, width, label='Ground Truth Docs Found', color='#4CAF50')
    bars_missing = ax.bar(x, missing, width, bottom=found, label='Ground Truth Docs Missing', color='#f44336')
    
    ax.set_ylabel('Number of Documents')
    ax.set_title(f'Ground Truth Document Availability\n(Total relevant docs: {g["total_relevant_docs"]:,})')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, (f, m) in enumerate(zip(found, missing)):
        total = f + m
        ax.annotate(f'{f:,}\n({f/total*100:.1f}%)', xy=(i, f/2), ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')
        if m > 0:
            ax.annotate(f'{m:,}\n({m/total*100:.1f}%)', xy=(i, f + m/2), ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ground_truth_impact.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'ground_truth_impact.png'}")
    plt.close()


def plot_drift_summary(analysis: dict, output_dir: Path) -> None:
    """Summary infographic of temporal drift metrics."""
    
    o = analysis['overall']
    d = analysis['document_stability']
    g = analysis['ground_truth_impact']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metric 1: Corpus Size Change
    ax1 = axes[0, 0]
    sizes = [o['chunks_2024'], o['chunks_2025']]
    ax1.bar(['Oct 2024', 'Oct 2025'], sizes, color=['#2196F3', '#FF5722'])
    ax1.set_title(f'Corpus Size\n({o["chunk_change_pct"]:+.1f}% change)')
    ax1.set_ylabel('Number of Chunks')
    for i, v in enumerate(sizes):
        ax1.text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')
    
    # Metric 2: Document Retention
    ax2 = axes[0, 1]
    ax2.bar(['Retained', 'Removed', 'Added'], 
            [d['retained'], d['removed'], d['added']],
            color=['#4CAF50', '#f44336', '#2196F3'])
    ax2.set_title(f'Document ID Stability\n(Jaccard: {d["jaccard_similarity"]:.1f}%)')
    ax2.set_ylabel('Number of Documents')
    
    # Metric 3: Ground Truth Loss
    ax3 = axes[1, 0]
    ax3.bar(['Found in 2024', 'Found in 2025'], 
            [g['found_in_2024'], g['found_in_2025']],
            color=['#4CAF50', '#f44336'])
    ax3.set_title(f'Ground Truth Availability\n({g["lost_pct"]:.1f}% lost)')
    ax3.set_ylabel('Number of Relevant Docs')
    ax3.axhline(y=g['total_relevant_docs'], color='gray', linestyle='--', 
                label=f'Total GT: {g["total_relevant_docs"]:,}')
    ax3.legend()
    
    # Metric 4: Key Statistics Text Box
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    TEMPORAL DRIFT SUMMARY
    ======================================
    
    Corpus Evolution:
    - Oct 2024: {o['chunks_2024']:,} chunks
    - Oct 2025: {o['chunks_2025']:,} chunks
    - Change: {o['chunk_change_pct']:+.1f}%
    
    Document Stability:
    - Retained: {d['retained']:,} ({d['retained_pct']:.1f}%)
    - Removed: {d['removed']:,} ({d['removed_pct']:.1f}%)
    - Jaccard Similarity: {d['jaccard_similarity']:.1f}%
    
    Ground Truth Impact:
    - 2024 Coverage: {g['found_in_2024_pct']:.1f}%
    - 2025 Coverage: {g['found_in_2025_pct']:.1f}%
    - GT Documents Lost: {g['lost_pct']:.1f}%
    
    ======================================
    KEY FINDING: Severe temporal drift
    renders ID-based evaluation invalid!
    """
    
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('FreshStack Temporal Corpus Drift Analysis\nOct 2024 to Oct 2025', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'drift_summary.png', bbox_inches='tight')
    print(f"Saved: {output_dir / 'drift_summary.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize temporal corpus drift')
    parser.add_argument('--analysis', required=True, help='Analysis JSON from analyze_temporal_drift.py')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots')
    args = parser.parse_args()
    
    # Load analysis
    with open(args.analysis, 'r') as f:
        analysis = json.load(f)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    plot_repository_comparison(analysis, output_dir)
    plot_document_stability(analysis, output_dir)
    plot_ground_truth_impact(analysis, output_dir)
    plot_drift_summary(analysis, output_dir)
    
    print(f"\nAll visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()

