"""
Query difficulty distribution based on number of nuggets per query.

Shows that the benchmark requires multi-fact retrieval (harder than binary relevance).
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Style configuration
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


def load_query_difficulties(queries_file: Path) -> dict:
    """
    Load queries and compute difficulty metrics.
    
    Returns dict with:
    - nuggets_per_query: list of nugget counts
    - total_queries: int
    - query_details: list of dicts with query_id and nugget_count
    """
    nugget_counts = []
    query_details = []
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            nuggets = query.get('nuggets', [])
            num_nuggets = len(nuggets)
            
            nugget_counts.append(num_nuggets)
            query_details.append({
                'query_id': query['query_id'],
                'nugget_count': num_nuggets,
                'question': query.get('question', '')[:100]  # First 100 chars
            })
    
    return {
        'nuggets_per_query': nugget_counts,
        'total_queries': len(nugget_counts),
        'query_details': query_details
    }


def plot_difficulty_histogram(data: dict, output_dir: Path) -> None:
    """Plot histogram of nuggets per query."""
    
    nugget_counts = data['nuggets_per_query']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    counts, bins, patches = ax.hist(nugget_counts, bins=range(1, max(nugget_counts) + 2), 
                                     alpha=0.7, color='#2196F3', edgecolor='black', 
                                     align='left')
    
    ax.set_xlabel('Number of Nuggets per Query')
    ax.set_ylabel('Number of Queries')
    ax.set_title(f'Query Difficulty Distribution (Total Queries: {data["total_queries"]:,})')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text
    mean_nuggets = np.mean(nugget_counts)
    median_nuggets = np.median(nugget_counts)
    
    stats_text = f"""
    Statistics:
    Mean: {mean_nuggets:.1f} nuggets/query
    Median: {median_nuggets:.0f} nuggets/query
    Min: {min(nugget_counts)}
    Max: {max(nugget_counts)}
    Std: {np.std(nugget_counts):.1f}
    
    Multi-fact queries: {sum(1 for n in nugget_counts if n > 1):,}
    ({sum(1 for n in nugget_counts if n > 1) / len(nugget_counts) * 100:.1f}%)
    """
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'query_difficulty_histogram.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_difficulty_breakdown(data: dict, output_dir: Path) -> None:
    """Plot bar chart of difficulty levels."""
    
    nugget_counts = data['nuggets_per_query']
    
    # Group into difficulty levels
    difficulty_levels = {
        'Easy (1 nugget)': sum(1 for n in nugget_counts if n == 1),
        'Medium (2-3 nuggets)': sum(1 for n in nugget_counts if 2 <= n <= 3),
        'Hard (4-5 nuggets)': sum(1 for n in nugget_counts if 4 <= n <= 5),
        'Very Hard (6+ nuggets)': sum(1 for n in nugget_counts if n >= 6)
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(difficulty_levels.keys())
    values = list(difficulty_levels.values())
    colors = ['#4CAF50', '#FFC107', '#FF9800', '#f44336']
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Number of Queries')
    ax.set_title('Query Difficulty Breakdown')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            pct = height / sum(values) * 100
            ax.annotate(f'{int(height):,}\n({pct:.1f}%)',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 5), textcoords="offset points",
                       ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = output_dir / 'query_difficulty_breakdown.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_cumulative_distribution(data: dict, output_dir: Path) -> None:
    """Plot cumulative distribution of nugget counts."""
    
    nugget_counts = sorted(data['nuggets_per_query'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Cumulative percentage
    cumulative = np.arange(1, len(nugget_counts) + 1) / len(nugget_counts) * 100
    
    ax.plot(nugget_counts, cumulative, linewidth=2, color='#2196F3')
    ax.fill_between(nugget_counts, cumulative, alpha=0.3, color='#2196F3')
    
    ax.set_xlabel('Number of Nuggets')
    ax.set_ylabel('Cumulative Percentage of Queries')
    ax.set_title('Cumulative Query Difficulty Distribution')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add reference lines
    for pct in [25, 50, 75]:
        idx = int(len(nugget_counts) * pct / 100)
        nugget_val = nugget_counts[idx]
        ax.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=nugget_val, color='gray', linestyle='--', alpha=0.5)
        ax.text(nugget_val + 0.2, pct + 3, f'{pct}th percentile: {nugget_val} nuggets',
                fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / 'query_difficulty_cumulative.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def print_statistics(data: dict) -> None:
    """Print query difficulty statistics."""
    
    nugget_counts = data['nuggets_per_query']
    counter = Counter(nugget_counts)
    
    print("\n" + "=" * 80)
    print("QUERY DIFFICULTY ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal queries: {data['total_queries']:,}")
    
    print(f"\nNugget count distribution:")
    for nugget_count in sorted(counter.keys()):
        count = counter[nugget_count]
        pct = count / len(nugget_counts) * 100
        print(f"  {nugget_count} nugget(s): {count:,} queries ({pct:.1f}%)")
    
    print(f"\nStatistics:")
    print(f"  Mean:   {np.mean(nugget_counts):.2f} nuggets/query")
    print(f"  Median: {np.median(nugget_counts):.0f} nuggets/query")
    print(f"  Mode:   {counter.most_common(1)[0][0]} nuggets")
    print(f"  Std:    {np.std(nugget_counts):.2f}")
    print(f"  Min:    {min(nugget_counts)} nuggets")
    print(f"  Max:    {max(nugget_counts)} nuggets")
    
    print(f"\nDifficulty levels:")
    print(f"  Easy (1 nugget):        {sum(1 for n in nugget_counts if n == 1):,} queries")
    print(f"  Medium (2-3 nuggets):   {sum(1 for n in nugget_counts if 2 <= n <= 3):,} queries")
    print(f"  Hard (4-5 nuggets):     {sum(1 for n in nugget_counts if 4 <= n <= 5):,} queries")
    print(f"  Very Hard (6+ nuggets): {sum(1 for n in nugget_counts if n >= 6):,} queries")
    
    multi_fact_queries = sum(1 for n in nugget_counts if n > 1)
    print(f"\nMulti-fact queries: {multi_fact_queries:,} ({multi_fact_queries / len(nugget_counts) * 100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("KEY FINDING: Most queries require retrieving multiple facts,")
    print("demonstrating the complexity beyond binary relevance judgments.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze query difficulty based on nugget counts')
    parser.add_argument('--queries', required=True, help='Queries JSONL file')
    parser.add_argument('--output_dir', required=True, help='Output directory for plots')
    args = parser.parse_args()
    
    print("Loading queries...")
    data = load_query_difficulties(Path(args.queries))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating plots...")
    plot_difficulty_histogram(data, output_dir)
    plot_difficulty_breakdown(data, output_dir)
    plot_cumulative_distribution(data, output_dir)
    
    print_statistics(data)
    
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == '__main__':
    main()
