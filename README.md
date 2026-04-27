# scGPT PBMC Benchmark Release

This repository packages a paper-style scGPT benchmark on public single-cell datasets.
It keeps one verified real-data run, then generalizes the same workflow into a small
benchmark suite with multiple datasets, multiple seeds, baseline methods, and analysis artifacts.

## What is included

- a reproducible scGPT fine-tuning entry point
- a second public dataset for comparison
- a multi-seed benchmark runner
- classical baseline methods for reference
- UMAP / ARI / NMI / marker-gene analysis outputs
- archived run artifacts, logs, checkpoints, and preprocessing outputs

## Paper-style framing

The project is organized the way we would write an appendix for a short methods paper:

- **data**: which public datasets were used, and which label field defines the task
- **protocol**: how samples are split, seeded, and evaluated
- **baselines**: what non-scGPT reference methods are compared
- **analysis**: what plots and summary metrics are generated per run
- **reproducibility**: exact configs, scripts, and archived outputs

## Repository layout

```text
.
├── benchmark_analysis.py
├── benchmark_baselines.py
├── benchmark_runner.py
├── configs/
│   └── benchmark.yaml
├── docs/
│   └── benchmark.md
├── experiments/
│   ├── __init__.py
│   ├── analysis.py
│   ├── datasets.py
│   └── metrics.py
├── quick_scgpt_train.py
├── real_pbmc68k_finetune.py
├── requirements.txt
├── environment.yml
├── scripts/
│   └── run_benchmark.sh
└── runs/
    └── pbmc68k_scgpt_finetune_20260427_195037/
        ├── args.json
        ├── script.py
        ├── vocab.json
        ├── requirements.txt
        ├── metrics.jsonl
        ├── summary.json
        ├── final_test.json
        ├── logs/
        │   └── train.log
        ├── data/
        │   └── preprocessed_data.h5ad
        └── checkpoints/
            ├── best_checkpoint.pt
            ├── best_model.pt
            ├── last_checkpoint.pt
            └── last_model.pt
```

## Main verified run

The archived real-data run is stored at:

`runs/pbmc68k_scgpt_finetune_20260427_195037/`

That directory contains the exact script snapshot, run arguments, vocabulary,
preprocessed data, training log, metrics, summary, and checkpoints.

### Verified training configuration

- `epochs=5`
- `batch_size=128`
- `lr=5e-4`
- `weight_decay=1e-2`
- `mask_ratio=0.15`
- `cls_weight=1.0`
- `mlm_weight=1.0`
- `hvg=128`
- `n_bins=51`
- `d_model=64`
- `nhead=4`
- `d_hid=128`
- `nlayers=2`
- `nlayers_cls=2`
- `dropout=0.2`
- `test_size=0.15`
- `val_size=0.15`
- `seed=42`

### Verified results

- **Best epoch:** 5
- **Best validation accuracy:** `0.34285714285714286`
- **Best validation loss:** `794.4338989257812`
- **Final test accuracy:** `0.34285714285714286`
- **Final classification loss:** `1.9761472940444946`

## Benchmark protocol

The benchmark suite is designed to run:

- multiple public datasets
- multiple random seeds
- one scGPT fine-tuning method
- several baseline methods
- one analysis pass per produced run

### Supported datasets

- `pbmc68k_reduced` with `bulk_labels`
- `paul15` with `paul15_clusters`
- `pbmc3k_processed` with `louvain`

### Supported methods

- `scgpt`
- `pca_logreg`
- `pca_svm`
- `pca_knn`
- `leiden`

### Metrics and artifacts

Each benchmark run can produce:

- accuracy
- macro-F1
- ARI
- NMI
- UMAP figures
- marker-gene tables for true and predicted groups

## How to reproduce

### 1. Install dependencies

This repository provides both:

- `requirements.txt` for pip-based installation
- `environment.yml` for conda-based installation

Example:

```bash
conda env create -f environment.yml
conda activate scgpt-pbmc68k
```

Then install the upstream `scGPT` source that matches the archived run.
The package is not vendored here.

### 2. Re-run the single verified experiment

```bash
python real_pbmc68k_finetune.py \
  --epochs 5 \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 1e-2 \
  --mask-ratio 0.15 \
  --cls-weight 1.0 \
  --mlm-weight 1.0 \
  --hvg 128 \
  --n-bins 51 \
  --d-model 64 \
  --nhead 4 \
  --d-hid 128 \
  --nlayers 2 \
  --nlayers-cls 2 \
  --dropout 0.2 \
  --test-size 0.15 \
  --val-size 0.15 \
  --seed 42 \
  --num-workers 0
```

### 3. Re-run the benchmark suite

```bash
bash scripts/run_benchmark.sh --datasets pbmc68k_reduced paul15 --seeds 0 1 2
```

This runs scGPT, baselines, and analysis for each dataset/seed combination.

## Appendix A. Experimental setup

### A.1 Dataset notes

The benchmark uses public Scanpy datasets so the workflow stays small,
inspectable, and reproducible.

Supported datasets in this release:

- `pbmc68k_reduced` with `bulk_labels`
- `paul15` with `paul15_clusters`
- `pbmc3k_processed` with `louvain`

### A.2 Evaluation notes

Classification performance is reported on the held-out test split.
If a method emits a latent embedding, the analysis pass adds UMAP and
cluster-comparison summaries.

The benchmark uses repeated seeds and writes run-local artifacts for each
dataset/method/seed combination so that the summary can be regenerated from
the archived outputs.

### A.3 Artifact policy

The repository keeps the important run outputs on purpose:

- logs
- checkpoints
- preprocessing output
- metrics
- summary files
- analysis figures and tables

That makes the experiment auditable rather than just reproducible in theory.

## Appendix B. Results and artifacts

### B.1 Verified single-run result

The archived real-data run is stored at:

`runs/pbmc68k_scgpt_finetune_20260427_195037/`

That directory contains the exact script snapshot, run arguments, vocabulary,
preprocessed data, training log, metrics, summary, and checkpoints.

Verified configuration:

- `epochs=5`
- `batch_size=128`
- `lr=5e-4`
- `weight_decay=1e-2`
- `mask_ratio=0.15`
- `cls_weight=1.0`
- `mlm_weight=1.0`
- `hvg=128`
- `n_bins=51`
- `d_model=64`
- `nhead=4`
- `d_hid=128`
- `nlayers=2`
- `nlayers_cls=2`
- `dropout=0.2`
- `test_size=0.15`
- `val_size=0.15`
- `seed=42`

Verified result:

- **Best epoch:** 5
- **Best validation accuracy:** `0.34285714285714286`
- **Best validation loss:** `794.4338989257812`
- **Final test accuracy:** `0.34285714285714286`
- **Final classification loss:** `1.9761472940444946`

### B.2 Full benchmark summary

The full benchmark completed successfully with 30 rows in the aggregated summary.
It covered two public datasets, three seeds, one scGPT run per seed, and four
classical baselines.

#### Mean ± std over seeds

| Dataset | Method | Accuracy | ARI | Macro-F1 | NMI | Notes |
|---|---:|---:|---:|---:|---:|---|
| pbmc68k_reduced | scGPT | 0.3429 ± 0.0000 | — | — | — | best_epoch 4.7 ± 0.5 |
| pbmc68k_reduced | pca_logreg | 0.7619 ± 0.0000 | 0.6007 ± 0.0161 | 0.6437 ± 0.0072 | 0.7114 ± 0.0118 | test split |
| pbmc68k_reduced | pca_svm | 0.7143 ± 0.0404 | 0.5732 ± 0.0600 | 0.5732 ± 0.0511 | 0.6389 ± 0.0697 | test split |
| pbmc68k_reduced | pca_knn | 0.7873 ± 0.0359 | 0.6477 ± 0.0410 | 0.6190 ± 0.0562 | 0.7363 ± 0.0446 | test split |
| pbmc68k_reduced | leiden | 0.0730 ± 0.0119 | 0.5356 ± 0.0581 | 0.0728 ± 0.0101 | 0.7119 ± 0.0158 | test split |
| paul15 | scGPT | 0.1366 ± 0.0000 | — | — | — | best_epoch 5.0 ± 0.0 |
| paul15 | pca_logreg | 0.5764 ± 0.0169 | 0.3647 ± 0.0135 | 0.5799 ± 0.0335 | 0.5903 ± 0.0098 | test split |
| paul15 | pca_svm | 0.5472 ± 0.0136 | 0.3357 ± 0.0056 | 0.5491 ± 0.0399 | 0.5720 ± 0.0095 | test split |
| paul15 | pca_knn | 0.4724 ± 0.0011 | 0.2994 ± 0.0043 | 0.5061 ± 0.0078 | 0.5541 ± 0.0184 | test split |
| paul15 | leiden | 0.0260 ± 0.0023 | 0.3232 ± 0.0154 | 0.0185 ± 0.0097 | 0.5658 ± 0.0161 | test split |

### B.3 Artifact policy

The benchmark keeps the important run outputs on purpose:

- logs
- checkpoints
- preprocessing output
- metrics
- summary files
- analysis figures and tables

That makes the experiment auditable rather than just reproducible in theory.

## Key files

- `real_pbmc68k_finetune.py` — real-data scGPT fine-tuning entry point
- `benchmark_runner.py` — multi-dataset, multi-seed benchmark orchestrator
- `benchmark_baselines.py` — classical baseline methods
- `benchmark_analysis.py` — UMAP / ARI / NMI / marker analysis
- `docs/benchmark.md` — structure and benchmark notes
- `configs/benchmark.yaml` — benchmark config template
- `scripts/run_benchmark.sh` — convenience wrapper
- `runs/pbmc68k_scgpt_finetune_20260427_195037/summary.json` — final summary
- `runs/pbmc68k_scgpt_finetune_20260427_195037/final_test.json` — final test metrics
- `runs/pbmc68k_scgpt_finetune_20260427_195037/logs/train.log` — training log
- `runs/benchmark_full/summary.json` — full benchmark per-run summary
- `runs/benchmark_full/summary_agg.csv` — aggregated mean/std summary

## License

MIT. See `LICENSE`.
