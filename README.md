# scGPT PBMC68k Fine-tuning Release

This repository contains a reproducible scGPT fine-tuning run on the public PBMC68k reduced dataset.

## What is included
- `real_pbmc68k_finetune.py`: the real-data fine-tuning script
- `quick_scgpt_train.py`: the earlier minimal smoke-test script
- `runs/pbmc68k_scgpt_finetune_20260427_195037/`: successful experiment artifacts
  - `args.json`
  - `script.py`
  - `vocab.json`
  - `requirements.txt`
  - `logs/train.log`
  - `metrics.jsonl`
  - `summary.json`
  - `final_test.json`
  - `data/preprocessed_data.h5ad`
  - `checkpoints/best_checkpoint.pt`
  - `checkpoints/best_model.pt`
  - `checkpoints/last_checkpoint.pt`
  - `checkpoints/last_model.pt`

## Result summary
- Best epoch: 5
- Best validation accuracy: 0.34285714285714286
- Final test accuracy: 0.34285714285714286

## Notes
The run uses the scanpy PBMC68k reduced dataset and a lightweight scGPT fine-tuning configuration.
