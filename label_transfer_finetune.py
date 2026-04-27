#!/usr/bin/env python3
"""Cross-dataset scGPT fine-tuning for cell type annotation / label transfer.

This script is the annotation benchmark entry point. It trains on a reference
atlas and evaluates on a held-out query atlas, while keeping the same
reproducibility artifacts as the single-dataset script.
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
import pandas as pd
import torch
from anndata import AnnData
from torch import nn
from torch.utils.data import DataLoader
from torchtext._torchtext import Vocab as VocabPybind
from torchtext.vocab import Vocab

REPO_DIR = Path(__file__).resolve().parent
for candidate in [REPO_DIR, REPO_DIR.parent, Path("/mnt/scgpt_trial"), Path("/mnt/scgpt_trial/scgpt_trial")]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import scanpy as sc
import scgpt as scg
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

from experiments.analysis import build_analysis_adata, compute_marker_summary, run_umap, save_umap_plot, summarize_predictions
from experiments.label_transfer import AnnDataset, encode_labels, make_loader, prepare_transfer_bundle, split_reference_indices
from experiments.metrics import compute_label_metrics


@dataclass
class TransferConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 64
    lr: float = 5e-4
    weight_decay: float = 1e-2
    mask_ratio: float = 0.15
    cls_weight: float = 1.0
    mlm_weight: float = 1.0
    hvg: int = 256
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
    dataset: str = "gutatlas_transfer"
    analysis: bool = False


class SeqDataset(torch.utils.data.Dataset):
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
        out = base_dir / f"annotation_transfer_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "logs", "artifacts", "data", "analysis"]:
        (out / sub).mkdir(exist_ok=True)
    return out


def setup_logger(log_file: Path) -> logging.Logger:
    logger = scg.logger
    logger.setLevel(logging.INFO)
    scg.utils.add_file_handler(logger, log_file)
    return logger


def build_vocab(genes: list[str], out_dir: Path) -> GeneVocab:
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    vocab = GeneVocab(genes, specials=special_tokens, special_first=True, default_token="<pad>")
    vocab.save_json(out_dir / "vocab.json")
    return vocab


def prepare_tokenized_data(raw: AnnData, vocab: GeneVocab, n_bins: int) -> Dict[str, torch.Tensor]:
    counts = raw.layers["X_binned"]
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    genes = raw.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    tokenized = tokenize_and_pad_batch(
        counts,
        gene_ids,
        max_len=len(genes) + 1,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=0,
        append_cls=True,
        include_zero_gene=True,
    )
    return tokenized


def preprocess_all(raw: AnnData, hvg: int, n_bins: int) -> AnnData:
    adata = raw.copy()
    adata.var["gene_name"] = adata.var_names.astype(str)
    preprocessor = Preprocessor(
        use_key="X",
        filter_gene_by_counts=0,
        filter_cell_by_counts=False,
        normalize_total=False,
        log1p=False,
        subset_hvg=hvg,
        hvg_flavor="cell_ranger",
        binning=n_bins,
        result_binned_key="X_binned",
    )
    preprocessor(adata)
    return adata


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
    return {"cls_loss": total_loss / max(total, 1), "accuracy": total_correct / max(total, 1)}


@torch.no_grad()
def evaluate_mlm(model, loader, device, pad_id: int, mask_ratio: float) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        labels = batch["celltype"].to(device)
        mask = gene_ids.eq(pad_id)
        masked_values = random_mask_value(values, mask_ratio=mask_ratio, mask_value=-1, pad_value=0).to(device)
        out = model(gene_ids, masked_values, src_key_padding_mask=mask, CLS=True)
        masked_positions = masked_values.eq(-1)
        loss = masked_mse_loss(out["mlm_output"], values, masked_positions)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return {"mlm_loss": total_loss / max(total, 1)}


def train_one_epoch(model, loader, optimizer, device, pad_id: int, config: TransferConfig) -> Dict[str, float]:
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
        masked_values = random_mask_value(values, mask_ratio=config.mask_ratio, mask_value=-1, pad_value=0).to(device)
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
        "train_loss": total_loss / max(total, 1),
        "train_cls_loss": total_cls / max(total, 1),
        "train_mlm_loss": total_mlm / max(total, 1),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_val_acc: float, config: TransferConfig, extra: Dict[str, float] | None = None) -> None:
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


@torch.no_grad()
def collect_embeddings(model, loader, device, pad_id: int):
    model.eval()
    cell_embs = []
    pred_ids = []
    for batch in loader:
        gene_ids = batch["gene_ids"].to(device)
        values = batch["values"].to(device)
        mask = gene_ids.eq(pad_id)
        out = model(gene_ids, values, src_key_padding_mask=mask, CLS=True)
        cell_embs.append(out["cell_emb"].detach().cpu())
        pred_ids.append(out["cls_output"].argmax(dim=1).detach().cpu())
    return torch.cat(cell_embs, dim=0).numpy(), torch.cat(pred_ids, dim=0).numpy()


def dump_environment(run_dir: Path) -> None:
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (run_dir / "requirements.txt").write_text(freeze, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset scGPT fine-tuning for cell type annotation")
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--dataset", default="gutatlas_transfer")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--cls-weight", type=float, default=1.0)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--hvg", type=int, default=256)
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
    parser.add_argument("--analysis", action="store_true")
    args = parser.parse_args()

    config = TransferConfig(
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
        dataset=args.dataset,
        analysis=args.analysis,
    )

    set_seed(config.seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    run_dir = build_run_dir(REPO_DIR / "runs", config.run_dir)
    logger = setup_logger(run_dir / "logs" / "train.log")
    logger.info("Run directory: %s", run_dir)
    logger.info("Configuration: %s", json.dumps(asdict(config), indent=2, sort_keys=True))
    shutil.copy2(Path(__file__).resolve(), run_dir / "script.py")
    (run_dir / "args.json").write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")

    from experiments.datasets import get_paired_dataset_spec

    paired_spec = get_paired_dataset_spec(config.dataset)
    logger.info("Benchmark pair: %s -> %s", paired_spec.reference_name, paired_spec.query_name)

    bundle = prepare_transfer_bundle(config.dataset, hvg=config.hvg, n_bins=config.n_bins)
    reference = bundle.reference
    query = bundle.query
    shared_labels = bundle.shared_labels
    logger.info("Reference cells: %d, query cells: %d, shared labels: %d, shared genes: %d", reference.n_obs, query.n_obs, len(shared_labels), len(bundle.shared_genes))

    reference.write_h5ad(run_dir / "data" / "reference_preprocessed.h5ad")
    query.write_h5ad(run_dir / "data" / "query_preprocessed.h5ad")

    vocab = build_vocab(bundle.shared_genes, run_dir)
    pad_id = vocab["<pad>"]

    ref_labels = encode_labels(reference.obs["celltype"], shared_labels)
    qry_labels = encode_labels(query.obs["celltype"], shared_labels)

    ref_tokenized = prepare_tokenized_data(reference, vocab, config.n_bins)
    qry_tokenized = prepare_tokenized_data(query, vocab, config.n_bins)

    ref_gene_ids = ref_tokenized["genes"]
    ref_values = ref_tokenized["values"]
    qry_gene_ids = qry_tokenized["genes"]
    qry_values = qry_tokenized["values"]

    train_idx, val_idx, _ = split_reference_indices(ref_labels, config.seed, config.test_size, config.val_size)
    ref_labels_tensor = torch.from_numpy(ref_labels).long()
    qry_labels_tensor = torch.from_numpy(qry_labels).long()

    train_data = {"gene_ids": ref_gene_ids[torch.from_numpy(train_idx).long()], "values": ref_values[torch.from_numpy(train_idx).long()], "celltype": ref_labels_tensor[torch.from_numpy(train_idx).long()]}
    val_data = {"gene_ids": ref_gene_ids[torch.from_numpy(val_idx).long()], "values": ref_values[torch.from_numpy(val_idx).long()], "celltype": ref_labels_tensor[torch.from_numpy(val_idx).long()]}
    query_data = {"gene_ids": qry_gene_ids, "values": qry_values, "celltype": qry_labels_tensor}

    train_loader = make_loader(train_data, config.batch_size, True, config.num_workers)
    val_loader = make_loader(val_data, config.batch_size, False, config.num_workers)
    query_loader = make_loader(query_data, config.batch_size, False, config.num_workers)
    ref_all_loader = make_loader({"gene_ids": ref_gene_ids, "values": ref_values, "celltype": ref_labels_tensor}, config.batch_size, False, config.num_workers)
    qry_all_loader = make_loader(query_data, config.batch_size, False, config.num_workers)

    device = torch.device("cpu")
    n_classes = len(shared_labels)
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
        val_mlm = evaluate_mlm(model, val_loader, device, pad_id, config.mask_ratio)
        query_cls = evaluate_cls(model, query_loader, device, pad_id)
        elapsed = time.time() - epoch_start
        scheduler.step()

        combined_val_loss = val_cls["cls_loss"] + val_mlm["mlm_loss"]
        if val_cls["accuracy"] > best_val_acc:
            best_val_acc = val_cls["accuracy"]
            best_epoch = epoch
            best_val_loss = combined_val_loss
            save_checkpoint(run_dir / "checkpoints" / "best_checkpoint.pt", model, optimizer, epoch, best_val_acc, config)
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best_model.pt")
        save_checkpoint(run_dir / "checkpoints" / "last_checkpoint.pt", model, optimizer, epoch, best_val_acc, config)
        torch.save(model.state_dict(), run_dir / "checkpoints" / "last_model.pt")

        payload = {
            "epoch": epoch,
            **train_metrics,
            "val_cls_loss": val_cls["cls_loss"],
            "val_acc": val_cls["accuracy"],
            "val_mlm_loss": val_mlm["mlm_loss"],
            "query_acc": query_cls["accuracy"],
            "query_cls_loss": query_cls["cls_loss"],
            "elapsed": elapsed,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        logger.info(
            "epoch=%d train_loss=%.4f train_cls=%.4f train_mlm=%.4f val_cls=%.4f val_acc=%.4f val_mlm=%.4f query_acc=%.4f elapsed=%.1fs",
            epoch,
            train_metrics["train_loss"],
            train_metrics["train_cls_loss"],
            train_metrics["train_mlm_loss"],
            val_cls["cls_loss"],
            val_cls["accuracy"],
            val_mlm["mlm_loss"],
            query_cls["accuracy"],
            elapsed,
        )

    ref_embs, ref_preds = collect_embeddings(model, ref_all_loader, device, pad_id)
    qry_embs, qry_preds = collect_embeddings(model, qry_all_loader, device, pad_id)

    ref_obs = reference.obs.copy()
    ref_obs["dataset_split"] = "reference"
    ref_obs["eval_split"] = "train"
    ref_obs.loc[ref_obs.index[val_idx], "eval_split"] = "val"
    ref_obs["true_label"] = ref_obs["celltype"].astype("category")
    ref_obs["pred_label"] = pd.Categorical.from_codes(ref_preds, categories=shared_labels)
    ref_adata = reference.copy()
    ref_adata.obs = ref_obs
    ref_adata.obsm["X_scGPT"] = ref_embs

    qry_obs = query.obs.copy()
    qry_obs["dataset_split"] = "query"
    qry_obs["eval_split"] = "test"
    qry_obs["true_label"] = qry_obs["celltype"].astype("category")
    qry_obs["pred_label"] = pd.Categorical.from_codes(qry_preds, categories=shared_labels)
    qry_adata = query.copy()
    qry_adata.obs = qry_obs
    qry_adata.obsm["X_scGPT"] = qry_embs

    analysis_adata = ref_adata.concatenate(qry_adata, batch_key="split_origin", batch_categories=["reference", "query"], index_unique=None)
    analysis_adata.obsm["X_scGPT"] = np.vstack([ref_embs, qry_embs])
    analysis_adata.write_h5ad(run_dir / "analysis" / "analysis_input.h5ad")

    final_test = compute_label_metrics(qry_labels, qry_preds)
    summary = {
        "dataset": config.dataset,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "final_test": final_test,
        "n_reference": int(reference.n_obs),
        "n_query": int(query.n_obs),
        "n_shared_labels": len(shared_labels),
        "n_shared_genes": len(bundle.shared_genes),
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "final_test.json").write_text(json.dumps(final_test, indent=2, sort_keys=True), encoding="utf-8")
    dump_environment(run_dir)

    if config.analysis:
        # The combined analysis input already has the requested labels and embeddings.
        adata = sc.read_h5ad(run_dir / "analysis" / "analysis_input.h5ad")
        metrics = summarize_predictions(adata, true_key="true_label", pred_key="pred_label")
        (run_dir / "analysis" / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        adata = run_umap(adata, embedding_key="X_scGPT", random_state=config.seed)
        save_umap_plot(adata, run_dir / "analysis" / "umap_true_label.png", color="true_label", title="UMAP colored by true label")
        save_umap_plot(adata, run_dir / "analysis" / "umap_pred_label.png", color="pred_label", title="UMAP colored by predicted label")
        compute_marker_summary(adata, groupby="true_label", n_top=10).to_csv(run_dir / "analysis" / "marker_genes_true.csv", index=False)
        compute_marker_summary(adata, groupby="pred_label", n_top=10).to_csv(run_dir / "analysis" / "marker_genes_pred.csv", index=False)

    logger.info("Summary: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
