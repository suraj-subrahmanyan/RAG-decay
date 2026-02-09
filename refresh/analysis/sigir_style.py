"""
Plotting style configuration for paper figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def set_sigir_style():
    """Configure matplotlib with serif fonts and appropriate figure sizing."""
    sns.set_theme(style="whitegrid", context="paper")
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'legend.title_fontsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })


def get_colorblind_palette():
    """
    Return a colorblind-safe palette for method comparison.
    
    Returns:
        List of hex colors suitable for distinguishing methods
    """
    return ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']
