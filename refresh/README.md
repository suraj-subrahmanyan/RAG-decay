# FreshStack Temporal Corpus Drift Analysis

Evaluating retrieval performance on evolving documentation corpora.

**Research Question:** How do retrieval scores change across different temporal snapshots of a corpus?

**Key Finding:** 94.2% of ground truth documents from Oct 2024 no longer exist in Oct 2025.

## Pre-Requisites

- python 3.10+
- java 21 (for Pyserini BM25)
- GPU with ~24GB VRAM (for BGE dense indexing)

## Setup

```bash
git clone https://github.com/suraj-subrahmanyan/RAG-decay.git
cd RAG-decay/refresh
conda create python=3.11 -n freshstack -y
conda activate freshstack
pip install -r requirements.txt
```

## Usage

### Build Corpus
```bash
python data_ingestion/chunk_corpus.py --config config/langchain_repos.yaml --version oct_2025
```

### Analyze Drift
```bash
python analysis/analyze_temporal_drift.py \
    --corpus_2024 dataset/langchain/oct_2024/corpus.jsonl \
    --corpus_2025 dataset/langchain/oct_2025/corpus.jsonl \
    --queries data/queries/oct_2024/langchain/test-00000-of-00001.parquet \
    --output analysis_results/drift_analysis.json

python analysis/visualize_temporal_drift.py \
    --analysis analysis_results/drift_analysis.json \
    --output_dir analysis_results/plots
```

### Build Indices
```bash
python indexing/build_bm25_index.py --corpus dataset/langchain/oct_2024/corpus.jsonl --output indices/bm25_oct_2024
python indexing/build_bge_index.py --corpus dataset/langchain/oct_2024/corpus.jsonl --output indices/bge_oct_2024 --device cuda
```

### Run Retrieval
```bash
python retrieval/run_retrieval.py \
    --queries data/queries/oct_2024/langchain/test-00000-of-00001.parquet \
    --bm25_index indices/bm25_oct_2024 \
    --bge_index indices/bge_oct_2024 \
    --output_dir retrieval_results \
    --k 50
```

## Analysis Results

Results and visualizations are in `analysis_results/`:

```
analysis_results/
├── drift_analysis.json
└── plots/
    ├── drift_summary.png
    ├── repository_comparison.png
    ├── ground_truth_impact.png
    └── document_stability.png
```

## Data Sources

Official FreshStack data from HuggingFace:
```python
from huggingface_hub import snapshot_download

snapshot_download("freshstack/corpus-oct-2024", repo_type="dataset", local_dir="data/corpus")
snapshot_download("freshstack/queries-oct-2024", repo_type="dataset", local_dir="data/queries")
```

## References

- [FreshStack Paper](https://arxiv.org/abs/2504.13128)
- [FreshStack GitHub](https://github.com/fresh-stack/freshstack)

## Author

Suraj Subrahmanyan - University of Waterloo
