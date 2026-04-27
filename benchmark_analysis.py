#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import scanpy as sc
from anndata import read_h5ad

from experiments.analysis import build_analysis_adata, compute_marker_summary, run_umap, save_umap_plot, summarize_predictions


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Analysis utilities for scGPT benchmark outputs")
    parser.add_argument("--input-h5ad", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embedding-key", default="X_scGPT")
    parser.add_argument("--true-label-key", default="celltype")
    parser.add_argument("--pred-label-key", default="pred_label")
    parser.add_argument("--umap-color", nargs="*", default=None)
    parser.add_argument("--top-markers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    adata = read_h5ad(args.input_h5ad)
    adata = build_analysis_adata(adata, adata.obsm[args.embedding_key], args.embedding_key, args.true_label_key, args.pred_label_key)

    metrics = summarize_predictions(adata, true_key="true_label", pred_key="pred_label")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    adata = run_umap(adata, embedding_key=args.embedding_key, random_state=args.seed)
    colors = args.umap_color or ["true_label", "pred_label"]
    for color in colors:
        save_umap_plot(adata, out_dir / f"umap_{color}.png", color=color, title=f"UMAP colored by {color}")

    marker_true = compute_marker_summary(adata, groupby="true_label", n_top=args.top_markers)
    marker_pred = compute_marker_summary(adata, groupby="pred_label", n_top=args.top_markers)
    marker_true.to_csv(out_dir / "marker_genes_true.csv", index=False)
    marker_pred.to_csv(out_dir / "marker_genes_pred.csv", index=False)

    adata.write_h5ad(out_dir / "analysis_input.h5ad")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
