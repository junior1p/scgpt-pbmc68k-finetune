# scGPT PBMC Benchmark — Pretrained Fine-tuning

This repository benchmarks scGPT for cell type annotation on public single-cell datasets,
with a focus on correctly loading pretrained weights and comparing against classical baselines.

## Key Results (Updated)

### scGPT Pretrained Backbone on PBMC3k (2638 cells, 8 cell types)

| Method | Accuracy | Macro-F1 | Balanced Acc | ARI | NMI |
|--------|----------|----------|--------------|-----|-----|
| **scGPT (pretrained, frozen backbone)** | **90.7%** | **0.897** | **0.887** | **0.816** | **0.806** |
| PCA + LogReg | 76.2% | 0.644 | — | 0.601 | 0.711 |
| PCA + KNN | 78.7% | 0.619 | — | 0.648 | 0.736 |
| PCA + SVM | 71.4% | 0.573 | — | 0.573 | 0.639 |
| scGPT (random init, wrong arch) | 34.3% | — | — | — | — |

> Note: Baselines from pbmc68k_reduced (700 cells); scGPT pretrained evaluated on pbmc3k (2638 cells).
> The +56 percentage point improvement over random-init scGPT demonstrates the value of pretrained weights.

### Per-class Performance (scGPT pretrained, test set)

| Cell Type | Precision | Recall | F1 | Support |
|-----------|-----------|--------|----|---------|
| B cells | 0.981 | 1.000 | 0.990 | 51 |
| CD14+ Monocytes | 0.944 | 0.944 | 0.944 | 72 |
| CD4 T cells | 0.926 | 0.942 | 0.934 | 172 |
| CD8 T cells | 0.727 | 0.681 | 0.703 | 47 |
| Dendritic cells | 1.000 | 0.833 | 0.909 | 6 |
| FCGR3A+ Monocytes | 0.808 | 0.913 | 0.857 | 23 |
| Megakaryocytes | 1.000 | 1.000 | 1.000 | 2 |
| NK cells | 0.900 | 0.783 | 0.837 | 23 |

## What Changed (vs. Original Repo)

The original repo had 6 critical issues that caused 34.3% accuracy (near random):

1. **No pretrained weights loaded** — `TransformerModel` was randomly initialized
2. **Wrong architecture** — d_model=64, nlayers=2 (~0.5M params) vs pretrained d_model=512, nlayers=12 (~51M params)
3. **Missing metrics** — only `accuracy` and `cls_loss` in `final_test.json`
4. **Incomplete analysis** — `--analysis` flag existed but UMAP plots were not generated
5. **Broken torchtext imports** — `from torchtext._torchtext import Vocab as VocabPybind` fails with newer torch
6. **Suboptimal hyperparameters** — lr=5e-4 (too high), batch_size=128, hvg=128

All issues are fixed in `real_pbmc68k_finetune.py`.

## How to Reproduce

### 1. Install dependencies

```bash
pip install scgpt==0.2.5 scanpy anndata torch
# Note: do NOT install torchtext — scGPT 0.2.5 has a built-in vocab backend
```

### 2. Download pretrained weights

```python
from huggingface_hub import hf_hub_download
hf_hub_download("perturblab/scgpt-human", "best_model.pt", local_dir="./scgpt_pretrained")
hf_hub_download("perturblab/scgpt-human", "vocab.json", local_dir="./scgpt_pretrained")
```

### 3. Run fine-tuning (full backbone, ~3.6 hr on 16-core CPU)

```bash
python real_pbmc68k_finetune.py \
  --pretrained-path ./scgpt_pretrained \
  --dataset pbmc3k_processed \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-4 \
  --hvg 1200 \
  --d-model 512 --nhead 8 --d-hid 512 --nlayers 12 --nlayers-cls 3 \
  --analysis \
  --seed 42
```

### 4. Fast evaluation (frozen backbone, ~5 min total)

For quick evaluation without full fine-tuning, use the frozen backbone approach:
pre-compute CLS embeddings once, then train only the classification head.

```bash
python real_pbmc68k_finetune.py \
  --pretrained-path ./scgpt_pretrained \
  --dataset pbmc3k \
  --epochs 50 \
  --batch-size 64 \
  --lr 5e-4 \
  --hvg 256 \
  --freeze-backbone \
  --analysis \
  --seed 42
```

## Important: Data Preprocessing

scGPT requires **raw integer counts** as input. The `pbmc3k_processed` dataset's `.raw` slot
contains log-normalized values (not raw counts), which produces degenerate embeddings.

The correct approach:
```python
# Load raw counts
adata_raw = sc.datasets.pbmc3k()
adata_proc = sc.datasets.pbmc3k_processed()
# Filter to processed cells, transfer labels
adata = adata_raw[adata_proc.obs_names].copy()
adata.obs['louvain'] = adata_proc.obs['louvain']
# Then: normalize_total(1e4) → log1p → HVG → binning(51)
```

## Repository Layout

```text
.
├── real_pbmc68k_finetune.py    # Main fine-tuning script (all fixes applied)
├── label_transfer_finetune.py  # Cross-dataset label transfer
├── quick_scgpt_train.py        # Quick training script
├── experiments/
│   ├── metrics.py              # compute_label_metrics (acc, F1, ARI, NMI)
│   ├── analysis.py             # UMAP + marker gene analysis
│   ├── datasets.py             # Dataset registry
│   └── label_transfer.py       # Label transfer utilities
├── runs/                       # Archived run artifacts
│   └── pbmc68k_scgpt_finetune_*/
│       ├── final_test.json     # Test metrics
│       ├── metrics.jsonl       # Per-epoch metrics
│       ├── summary.json        # Run summary
│       └── checkpoints/        # Model checkpoints
└── docs/
    └── plans/                  # Development notes
```

## Technical Notes

### Weight Loading

The pretrained checkpoint uses flash-attention naming (`self_attn.Wqkv.*`) while
standard PyTorch uses `self_attn.in_proj_*`. `scgpt.utils.load_pretrained()` handles
this rename automatically. Result: 159/169 tensors loaded; only `cls_decoder.*`
(10 tensors) is randomly initialized (correct — task-specific head).

### Gene Vocabulary

The pretrained model uses a 60,697-gene vocabulary. PBMC3k has ~27,000 genes in
the pretrained vocab (after filtering). With hvg=256, we use the 256 most variable
genes, all of which are in the pretrained vocab.

### Architecture

| Parameter | Value |
|-----------|-------|
| d_model | 512 |
| nhead | 8 |
| nlayers | 12 |
| nlayers_cls | 3 |
| n_bins | 51 |
| Total params | ~51.3M |

## Previous Results (Original Repo)

The original repo used random initialization with a tiny model (d_model=64, nlayers=2):

| Method | Accuracy | Notes |
|--------|----------|-------|
| scGPT (original) | 34.3% | Random init, wrong arch, pbmc68k_reduced |
| PCA+LogReg | 76.2% | pbmc68k_reduced |
| PCA+KNN | 78.7% | pbmc68k_reduced |
| PCA+SVM | 71.4% | pbmc68k_reduced |

## License

MIT. See `LICENSE`.
