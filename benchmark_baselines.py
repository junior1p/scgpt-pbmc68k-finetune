#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from experiments.datasets import (
    harmonize_shared_labels,
    is_paired_dataset,
    load_dataset,
    load_paired_dataset,
    preprocess_for_baseline,
    select_shared_hvgs,
    subset_shared_genes,
)

METHODS = ("pca_logreg", "pca_svm", "pca_knn", "leiden", "majority_class")


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def split_indices(labels, seed, test_size, val_size):
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=test_size + val_size, random_state=seed, stratify=labels
    )
    holdout_labels = labels[temp_idx]
    rel_test = test_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, random_state=seed, stratify=holdout_labels
    )
    return train_idx, val_idx, test_idx


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_supervised(method: str, X_train, y_train, X_eval):
    if method == "pca_logreg":
        clf = LogisticRegression(max_iter=3000, multi_class="auto")
    elif method == "pca_svm":
        clf = LinearSVC()
    elif method == "pca_knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError(method)
    clf.fit(X_train, y_train)
    return clf.predict(X_eval)


def prepare_single_dataset(dataset: str, seed: int, hvg: int, test_size: float, val_size: float):
    raw, label_key, spec = load_dataset(dataset)
    labels = raw.obs[label_key].astype("category")
    raw.obs["true_label"] = labels
    raw.obs["celltype"] = labels
    adata = preprocess_for_baseline(raw, hvg=hvg)
    X = adata.obsm["X_pca"]
    label_ids = adata.obs["celltype"].cat.codes.to_numpy()
    train_idx, val_idx, test_idx = split_indices(label_ids, seed, test_size, val_size)
    return adata, X, label_ids, train_idx, val_idx, test_idx


def prepare_paired_dataset(dataset: str, seed: int, hvg: int, test_size: float, val_size: float):
    reference, query, spec = load_paired_dataset(dataset)
    reference, query, shared_labels = harmonize_shared_labels(
        reference,
        query,
        spec.reference_label_key,
        spec.query_label_key,
    )
    shared_genes = select_shared_hvgs(reference, query, hvg)
    reference, query = subset_shared_genes(reference, query, shared_genes)
    reference.obs["split_origin"] = "reference"
    query.obs["split_origin"] = "query"
    reference.obs["celltype"] = pd.Categorical(reference.obs["celltype"].astype(str), categories=shared_labels)
    query.obs["celltype"] = pd.Categorical(query.obs["celltype"].astype(str), categories=shared_labels)
    combined = reference.concatenate(query, batch_key="split_origin", batch_categories=["reference", "query"], index_unique=None)
    sc.pp.normalize_total(combined, target_sum=1e4)
    sc.pp.log1p(combined)
    sc.pp.scale(combined, max_value=10)
    sc.pp.pca(combined, n_comps=min(50, combined.n_vars - 1), svd_solver="arpack")
    X = combined.obsm["X_pca"]
    label_ids = combined.obs["celltype"].cat.codes.to_numpy()
    ref_mask = combined.obs["split_origin"] == "reference"
    query_mask = combined.obs["split_origin"] == "query"
    ref_labels = label_ids[ref_mask]
    train_idx_rel, val_idx_rel, _ = split_indices(ref_labels, seed, test_size, val_size)
    ref_idx = np.where(ref_mask)[0]
    train_idx = ref_idx[train_idx_rel]
    val_idx = ref_idx[val_idx_rel]
    test_idx = np.where(query_mask)[0]
    return combined, X, label_ids, train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser(description="Baseline comparisons for the scGPT PBMC / annotation benchmark")
    parser.add_argument("--dataset", default="pbmc68k_reduced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--hvg", type=int, default=128)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "benchmark" / args.dataset / f"baseline_seed_{args.seed}"
    ensure_dir(out_dir)

    if is_paired_dataset(args.dataset):
        adata, X, labels, train_idx, val_idx, test_idx = prepare_paired_dataset(args.dataset, args.seed, args.hvg, args.test_size, args.val_size)
    else:
        adata, X, labels, train_idx, val_idx, test_idx = prepare_single_dataset(args.dataset, args.seed, args.hvg, args.test_size, args.val_size)

    results = {}

    for method in args.methods:
        method_dir = ensure_dir(out_dir / method)
        if method in {"pca_logreg", "pca_svm", "pca_knn"}:
            test_pred = run_supervised(method, X[train_idx], labels[train_idx], X[test_idx])
            full_pred = run_supervised(method, X[train_idx], labels[train_idx], X)
            result = metrics(labels[test_idx], test_pred)
            result.update({"method": method, "dataset": args.dataset, "seed": args.seed, "split": "test"})
            results[method] = result
            ad = adata.copy()
            ad.obsm["X_pca"] = X
            ad.obs["true_label"] = ad.obs["celltype"].astype("category")
            ad.obs["pred_label"] = pd.Categorical.from_codes(full_pred, categories=ad.obs["true_label"].cat.categories)
            ad.obs["eval_split"] = "train"
            ad.obs.iloc[test_idx, ad.obs.columns.get_loc("eval_split")] = "test"
            ad.write_h5ad(method_dir / "analysis_input.h5ad")
        elif method == "majority_class":
            majority = np.bincount(labels[train_idx]).argmax()
            test_pred = np.full(len(test_idx), majority, dtype=int)
            full_pred = np.full(len(labels), majority, dtype=int)
            result = metrics(labels[test_idx], test_pred)
            result.update({"method": method, "dataset": args.dataset, "seed": args.seed, "split": "test"})
            results[method] = result
            ad = adata.copy()
            ad.obsm["X_pca"] = X
            ad.obs["true_label"] = ad.obs["celltype"].astype("category")
            ad.obs["pred_label"] = pd.Categorical.from_codes(full_pred, categories=ad.obs["true_label"].cat.categories)
            ad.obs["eval_split"] = "train"
            ad.obs.iloc[test_idx, ad.obs.columns.get_loc("eval_split")] = "test"
            ad.write_h5ad(method_dir / "analysis_input.h5ad")
        elif method == "leiden":
            ad = adata.copy()
            sc.pp.neighbors(ad, use_rep="X_pca")
            sc.tl.leiden(ad, random_state=args.seed)
            pred = ad.obs["leiden"].astype("category").cat.codes.to_numpy()
            result = metrics(labels[test_idx], pred[test_idx])
            result.update({"method": method, "dataset": args.dataset, "seed": args.seed, "split": "test"})
            results[method] = result
            ad.obs["true_label"] = ad.obs["celltype"].astype("category")
            ad.obs["pred_label"] = ad.obs["leiden"].astype("category")
            ad.obs["eval_split"] = "train"
            ad.obs.iloc[test_idx, ad.obs.columns.get_loc("eval_split")] = "test"
            ad.write_h5ad(method_dir / "analysis_input.h5ad")
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        (method_dir / "metrics.json").write_text(json.dumps(results[method], indent=2, sort_keys=True), encoding="utf-8")

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
