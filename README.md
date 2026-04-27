# scGPT PBMC68k Fine-tuning

This repository provides a reproducible scGPT fine-tuning run on the public **PBMC68k reduced** dataset from `scanpy`.

The goal of this project is to document a small but complete real-data fine-tuning workflow, including:

- dataset loading and preprocessing
- tokenization and vocabulary construction
- multi-objective scGPT fine-tuning
- checkpointing and metric logging
- reproducible experiment artifacts

## Highlights

- **Dataset:** `scanpy.datasets.pbmc68k_reduced()`
- **Task:** scGPT fine-tuning with masked value prediction + cell-type classification
- **Artifacts preserved:** checkpoints, logs, metrics, preprocessing outputs, and runnable scripts
- **Status:** one successful end-to-end training run has been completed and archived in this repository

## Repository layout

```text
.
├── README.md
├── quick_scgpt_train.py
├── real_pbmc68k_finetune.py
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

## Reproducibility

The main experiment is archived under:

`runs/pbmc68k_scgpt_finetune_20260427_195037/`

It contains the exact run configuration, script snapshot, vocabulary, metrics, final test result, preprocessing output, and checkpoints.

### Training configuration

The successful run used the following configuration:

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

## Results

Final metrics from the archived successful run:

- **Best epoch:** 5
- **Best validation accuracy:** `0.34285714285714286`
- **Best validation loss:** `794.4338989257812`
- **Final test accuracy:** `0.34285714285714286`
- **Final classification loss:** `1.9761472940444946`

## How to run

### 1. Install dependencies

This project assumes an environment with the following core packages available:

- `torch`
- `scanpy`
- `anndata`
- `scgpt`
- `numpy`
- `pandas`
- `scikit-learn`

Exact versions may vary depending on your environment.

### 2. Launch the fine-tuning script

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

The script will create a timestamped run directory under `./runs/` unless `--run-dir` is specified.

## Files of interest

- `real_pbmc68k_finetune.py` — the real-data fine-tuning entry point
- `quick_scgpt_train.py` — the earlier smoke-test script
- `runs/pbmc68k_scgpt_finetune_20260427_195037/summary.json` — final summary
- `runs/pbmc68k_scgpt_finetune_20260427_195037/final_test.json` — final test result
- `runs/pbmc68k_scgpt_finetune_20260427_195037/logs/train.log` — full training log
- `runs/pbmc68k_scgpt_finetune_20260427_195037/checkpoints/` — checkpoint artifacts

## Notes

- The experiment uses the public PBMC68k reduced dataset bundled through `scanpy`.
- The repository includes the successful run artifacts so the experiment can be audited or replayed.
- Large binary checkpoint and data files are intentionally included because this repository is meant to serve as a complete training record.

## License

No explicit license has been added yet. Please add one if you want to distribute or reuse the code and artifacts publicly.
