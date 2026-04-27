# Paper-style benchmark structure

This repository now contains a lightweight benchmark framework for scGPT fine-tuning and baseline comparison.

## Structure

```text
.
├── benchmark_runner.py         # orchestrates dataset × seed × method runs
├── benchmark_baselines.py      # PCA / logistic regression / SVM / kNN / Leiden baselines
├── benchmark_analysis.py       # UMAP, ARI, NMI, marker-gene summaries
├── configs/
│   └── benchmark.yaml          # experiment configuration template
├── experiments/
│   ├── __init__.py
│   ├── analysis.py
│   ├── datasets.py
│   └── metrics.py
├── real_pbmc68k_finetune.py    # scGPT fine-tuning entry point (supports multiple datasets)
├── scripts/
│   └── run_benchmark.sh        # convenience wrapper
└── runs/
    └── benchmark/
        └── <dataset>/<method>/seed_<seed>/...
```

## Datasets

The benchmark runner currently supports these public Scanpy datasets:

- `pbmc68k_reduced` with `bulk_labels`
- `paul15` with `paul15_clusters`
- `pbmc3k_processed` with `louvain`

## Methods

- `scgpt`
- `pca_logreg`
- `pca_svm`
- `pca_knn`
- `leiden`

## Metrics

Each run can produce:

- accuracy
- macro-F1
- ARI
- NMI
- UMAP visualizations
- marker-gene summaries for true and predicted groups

## Example

Run the full benchmark suite:

```bash
bash scripts/run_benchmark.sh --datasets pbmc68k_reduced paul15 --seeds 0 1 2
```
