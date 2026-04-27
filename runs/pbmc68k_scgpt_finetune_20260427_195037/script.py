#!/usr/bin/env python3
"""scGPT fine-tuning on a real public dataset (PBMC68k reduced).

This script is self-contained and saves:
- checkpoints/best_checkpoint.pt
- checkpoints/last_checkpoint.pt
- checkpoints/best_model.pt
- logs/train.log
- metrics.jsonl
- args.json
- vocab.json
- preprocessed_data.h5ad
- script copy for reproducibility
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
import torch
from anndata import AnnData
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import scgpt as scg
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed


@dataclass
class RunConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 1e-2
    mask_ratio: float = 0.15
    cls_weight: float = 1.0
    mlm_weight: float = 1.0
    hvg: int = 128
    n_bins: int = 51
    d_model: int = 64
    nhead: int = 4
    d_hid: int = 128
    nlayers: int = 2
    nlayers_cls: int = 2
    dropout: float = 0.2
    test_size: float = 0.15
    val_size: float = 0.15
    max_grad_norm: float = 1.0
    num_workers: int = 0
    run_dir: str = ""


class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self) -> int:
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.data.items()}


def build_run_dir(base_dir: Path, run_dir: str) -> Path:
    if run_dir:
        out = Path(run_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = base_dir / f"pbmc68k_scgpt_finetune_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "logs", "artifacts", "data"]:
        (out / sub).mkdir(exist_ok=True)
    return out


def setup_logger(log_file: Path) -> logging.Logger:
    logger = scg.logger
    logger.setLevel(logging.INFO)
    scg.utils.add_file_handler(logger, log_file)
    return logger


def load_and_preprocess(config: RunConfig, logger: logging.Logger) -> Tuple[AnnData, np.ndarray, np.ndarray]:
    logger.info("Loading PBMC68k reduced dataset from scanpy...")
    adata = sc.datasets.pbmc68k_reduced()
    celltype = adata.obs["bulk_labels"].astype("category")

    # Use the raw layer, which holds non-negative expression values suitable for scGPT.
    raw = adata.raw.to_adata()
    raw.obs["celltype"] = celltype
    raw.obs["celltype_id"] = celltype.cat.codes.astype(np.int64)
    raw.var["gene_name"] = raw.var_names.astype(str)

    logger.info("Preprocessing: HVG selection + binning")
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=0,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        subset_hvg=config.hvg,
        hvg_flavor="cell_ranger",
        binning=config.n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(raw)

    # Save exact preprocessed data for reproducibility.
    return raw, raw.layers["X_binned"], raw.obs["celltype_id"].to_numpy()


def build_vocab(genes: list[str], out_dir: Path) -> GeneVocab:
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab(genes, specials=special_tokens, special_first=True, default_token="<pad>")
    vocab.save_json(out_dir / "vocab.json")
    return vocab


def split_indices(labels: np.ndarray, seed: int, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=labels,
    )
    # Re-stratify the hold-out split into val/test.
    holdout_labels = labels[temp_idx]
    rel_test = test_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        random_state=seed,
        stratify=holdout_labels,
    )
    return train_idx, val_idx, test_idx

def prepare_tokenized_data(raw: AnnData, vocab: GeneVocab, config: RunConfig) -> Dict[str, torch.Tensor]:
    counts = raw.layers["X_binned"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    genes = raw.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    max_seq_len = min(config.hvg + 1, counts.shape[1] + 1)

    tokenized = tokenize_and_pad_batch(
        counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=0,
        append_cls=True,
        include_zero_gene=True,
    )
    return tokenized


def make_loader(data: Dict[str, torch.Tensor], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        SeqDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


@torch.no_grad()
def evaluate_cls(model, loader, device, pad_id: int) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["celltype"].to(device)
        mask = gene_ids.eq(pad_id)
        out = model(gene_ids, values, src_key_padding_mask=mask, CLS=True)
        logits = out["cls_output"]
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return {"cls_loss": total_loss / total, "accuracy": total_correct / total}


@torch.no_grad()
def evaluate_mlm(model, loader, device, pad_id: int, config: RunConfig) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["celltype"].to(device)
        mask = gene_ids.eq(pad_id)
        masked_values = random_mask_value(
            values,
            mask_ratio=config.mask_ratio,
            mask_value=-1,
            pad_value=0,
        ).to(device)
        out = model(gene_ids, masked_values, src_key_padding_mask=mask, CLS=True)
        masked_positions = masked_values.eq(-1)
        loss = masked_mse_loss(out["mlm_output"], values, masked_positions)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return {"mlm_loss": total_loss / total}


def train_one_epoch(model, loader, optimizer, device, pad_id: int, config: RunConfig) -> Dict[str, float]:
    model.train()
    criterion_cls = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_cls = 0.0
    total_mlm = 0.0
    total = 0
    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["celltype"].to(device)
        mask = gene_ids.eq(pad_id)
        masked_values = random_mask_value(
            values,
            mask_ratio=config.mask_ratio,
            mask_value=-1,
            pad_value=0,
        ).to(device)
        out = model(gene_ids, masked_values, src_key_padding_mask=mask, CLS=True)
        masked_positions = masked_values.eq(-1)
        mlm_loss = masked_mse_loss(out["mlm_output"], values, masked_positions)
        cls_loss = criterion_cls(out["cls_output"], labels)
        loss = config.mlm_weight * mlm_loss + config.cls_weight * cls_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_cls += cls_loss.item() * labels.size(0)
        total_mlm += mlm_loss.item() * labels.size(0)
        total += labels.size(0)
    return {
        "train_loss": total_loss / total,
        "train_cls_loss": total_cls / total,
        "train_mlm_loss": total_mlm / total,
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    best_val_acc: float,
    config: RunConfig,
    extra: Dict[str, float] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "config": asdict(config),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def dump_environment(run_dir: Path) -> None:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "requirements.txt").write_text(freeze, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data scGPT fine-tuning on PBMC68k reduced")
    parser.add_argument("--run-dir", default="", help="Output directory. Default: timestamped run dir under ./runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--hvg", type=int, default=128)
    parser.add_argument("--n-bins", type=int, default=51)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--d-hid", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--nlayers-cls", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = RunConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        cls_weight=args.cls_weight,
        mlm_weight=args.mlm_weight,
        hvg=args.hvg,
        n_bins=args.n_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        nlayers_cls=args.nlayers_cls,
        dropout=args.dropout,
        test_size=args.test_size,
        val_size=args.val_size,
        num_workers=args.num_workers,
        run_dir=args.run_dir,
    )

    set_seed(config.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    base_dir = REPO_DIR / "runs"
    run_dir = build_run_dir(base_dir, config.run_dir)
    logger = setup_logger(run_dir / "logs" / "train.log")

    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration: %s", json.dumps(asdict(config), indent=2, sort_keys=True))

    # Reproducibility artifacts.
    shutil.copy2(Path(__file__).resolve(), run_dir / "script.py")
    (run_dir / "args.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")

    raw, _, labels = load_and_preprocess(config, logger)
    raw.write_h5ad(run_dir / "data" / "preprocessed_data.h5ad")

    genes = raw.var["gene_name"].tolist()
    vocab = build_vocab(genes, run_dir)
    pad_id = vocab["<pad>"]

    tokenized = prepare_tokenized_data(raw, vocab, config)
    all_genes = tokenized["genes"]
    all_values = tokenized["values"]
    labels_tensor = torch.from_numpy(labels).long()

    train_idx, val_idx, test_idx = split_indices(labels, config.seed, config.test_size, config.val_size)
    train_idx_t = torch.from_numpy(train_idx).long()
    val_idx_t = torch.from_numpy(val_idx).long()
    test_idx_t = torch.from_numpy(test_idx).long()

    train_data = {
        "gene_ids": all_genes[train_idx_t],
        "values": all_values[train_idx_t],
        "celltype": labels_tensor[train_idx_t],
    }
    val_data = {
        "gene_ids": all_genes[val_idx_t],
        "values": all_values[val_idx_t],
        "celltype": labels_tensor[val_idx_t],
    }
    test_data = {
        "gene_ids": all_genes[test_idx_t],
        "values": all_values[test_idx_t],
        "celltype": labels_tensor[test_idx_t],
    }

    train_loader = make_loader(train_data, config.batch_size, True, config.num_workers)
    val_loader = make_loader(val_data, config.batch_size, False, config.num_workers)
    test_loader = make_loader(test_data, config.batch_size, False, config.num_workers)

    device = torch.device("cpu")
    n_classes = int(labels_tensor.max().item()) + 1
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config.d_model,
        nhead=config.nhead,
        d_hid=config.d_hid,
        nlayers=config.nlayers,
        nlayers_cls=config.nlayers_cls,
        n_cls=n_classes,
        vocab=vocab,
        dropout=config.dropout,
        pad_token="<pad>",
        pad_value=0,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=config.n_bins,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.3,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    metrics_path = run_dir / "metrics.jsonl"
    best_val_acc = -1.0
    best_epoch = -1
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, pad_id, config)
        val_cls = evaluate_cls(model, val_loader, device, pad_id)
        val_mlm = evaluate_mlm(model, val_loader, device, pad_id, config)
        test_cls = evaluate_cls(model, test_loader, device, pad_id)
        elapsed = time.time() - epoch_start
        scheduler.step()

        combined_val_loss = val_cls["cls_loss"] + val_mlm["mlm_loss"]
        row = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 3),
            **train_metrics,
            "val_cls_loss": val_cls["cls_loss"],
            "val_acc": val_cls["accuracy"],
            "val_mlm_loss": val_mlm["mlm_loss"],
            "test_acc": test_cls["accuracy"],
            "lr": scheduler.get_last_lr()[0],
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(
            "epoch=%d train_loss=%.4f train_cls=%.4f train_mlm=%.4f val_cls=%.4f val_acc=%.4f val_mlm=%.4f test_acc=%.4f elapsed=%.1fs",
            epoch,
            train_metrics["train_loss"],
            train_metrics["train_cls_loss"],
            train_metrics["train_mlm_loss"],
            val_cls["cls_loss"],
            val_cls["accuracy"],
            val_mlm["mlm_loss"],
            test_cls["accuracy"],
            elapsed,
        )

        save_checkpoint(run_dir / "checkpoints" / "last_checkpoint.pt", model, optimizer, epoch, best_val_acc, config, extra=row)
        torch.save(model.state_dict(), run_dir / "checkpoints" / "last_model.pt")

        if val_cls["accuracy"] > best_val_acc or (
            val_cls["accuracy"] == best_val_acc and combined_val_loss < best_val_loss
        ):
            best_val_acc = val_cls["accuracy"]
            best_val_loss = combined_val_loss
            best_epoch = epoch
            save_checkpoint(run_dir / "checkpoints" / "best_checkpoint.pt", model, optimizer, epoch, best_val_acc, config, extra=row)
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best_model.pt")
            logger.info("New best checkpoint at epoch %d: val_acc=%.4f", epoch, best_val_acc)

    final_test = evaluate_cls(model, test_loader, device, pad_id)
    (run_dir / "final_test.json").write_text(json.dumps(final_test, indent=2, sort_keys=True), encoding="utf-8")
    dump_environment(run_dir)

    summary = {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test": final_test,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Summary: %s", json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
