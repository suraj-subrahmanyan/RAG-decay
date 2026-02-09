# Figure Generation Scripts

Scripts for generating paper figures in vector PDF format.

## Setup

```bash
pip install matplotlib seaborn numpy
pip install plotly kaleido  # optional, for Sankey diagrams
```

## Scripts

### `plot_overlap_heatmap_sigir.py`
Jaccard similarity heatmap between retrieval methods.

```bash
python plot_overlap_heatmap_sigir.py \
  --results_dir /path/to/retrieval_results \
  --corpus_version oct_2024 \
  --query_field answer \
  --output_file fig_overlap.pdf \
  --top_k 20
```

### `plot_drift_sankey_sigir.py`
Sankey diagram showing document retention/modification/deletion.

```bash
python plot_drift_sankey_sigir.py \
  --drift_stats /path/to/stats.json \
  --output_file fig_drift.pdf
```

### `plot_degradation_slope_sigir.py`
Slope chart showing performance changes from 2024 to 2025.

```bash
python plot_degradation_slope_sigir.py \
  --metrics_2024 /path/to/metrics_2024.json \
  --metrics_2025 /path/to/metrics_2025.json \
  --output_file fig_degradation.pdf \
  --metric alpha_ndcg_10
```

### `plot_radar_sigir.py`
Radar chart comparing query formulations.

```bash
python plot_radar_sigir.py \
  --metrics_2024 /path/to/query_metrics_2024.json \
  --metrics_2025 /path/to/query_metrics_2025.json \
  --output_file fig_radar.pdf \
  --metric alpha_ndcg_10
```

## Output

All figures are generated as vector PDFs (300 DPI) with serif fonts and colorblind-safe palettes.

Output directory: `refresh/analysis_results/sigir_figures/`
