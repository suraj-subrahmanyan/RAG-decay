"""
Master script to generate all SIGIR publication-quality figures.

This script orchestrates the generation of all 4 main figures
for the SIGIR short paper submission.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required libraries are available."""
    missing = []
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    try:
        import seaborn
    except ImportError:
        missing.append("seaborn")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import plotly
    except ImportError:
        logger.warning("plotly not found (needed for Sankey). Install with: pip install plotly kaleido")
    
    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Install with: pip install matplotlib seaborn numpy")
        return False
    
    return True


def generate_overlap_heatmap(results_dir: Path, output_dir: Path, corpus_version: str):
    """Generate retrieval overlap heatmap (Figure 2)."""
    logger.info("="*80)
    logger.info("GENERATING FIGURE 2: Retrieval Overlap Heatmap")
    logger.info("="*80)
    
    try:
        from plot_overlap_heatmap_sigir import main as overlap_main
        import sys
        
        sys.argv = [
            'plot_overlap_heatmap_sigir.py',
            '--results_dir', str(results_dir),
            '--corpus_version', corpus_version,
            '--query_field', 'answer',
            '--output_file', str(output_dir / f'fig2_overlap_{corpus_version}.pdf'),
            '--top_k', '20'
        ]
        
        overlap_main()
        logger.info("✓ Figure 2 complete\n")
        return True
        
    except Exception as e:
        logger.error(f"Failed to generate Figure 2: {e}")
        return False


def generate_all_figures(args):
    """Generate all SIGIR figures."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("SIGIR 2026 FIGURE GENERATION")
    logger.info("="*80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    results = {
        'fig2_overlap': False,
    }
    
    if args.results_dir:
        results['fig2_overlap'] = generate_overlap_heatmap(
            Path(args.results_dir),
            output_dir,
            args.corpus_version
        )
    
    logger.info("="*80)
    logger.info("GENERATION SUMMARY")
    logger.info("="*80)
    for fig, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {fig}")
    
    logger.info("")
    if all(results.values()):
        logger.info("✓ ALL FIGURES GENERATED SUCCESSFULLY")
    else:
        logger.warning("⚠ Some figures failed to generate")
    
    logger.info(f"\nOutput location: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate all SIGIR publication-quality figures"
    )
    
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Output directory for figures'
    )
    
    parser.add_argument(
        '--results_dir',
        help='Retrieval results directory (for overlap heatmap)'
    )
    
    parser.add_argument(
        '--corpus_version',
        default='oct_2024',
        choices=['oct_2024', 'oct_2025'],
        help='Corpus version for analysis'
    )
    
    args = parser.parse_args()
    
    if not check_dependencies():
        return
    
    generate_all_figures(args)


if __name__ == "__main__":
    main()
