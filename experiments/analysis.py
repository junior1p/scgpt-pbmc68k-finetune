from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from .metrics import compute_label_metrics


def build_analysis_adata(
    adata: AnnData,
    embedding: np.ndarray,
    embedding_key: str,
    true_label_key: str,
    pred_label_key: Optional[str] = None,
) -> AnnData:
    out = adata.copy()
    out.obsm[embedding_key] = embedding
    out.obs["true_label"] = out.obs[true_label_key].astype("category")
    if pred_label_key and pred_label_key in out.obs:
        out.obs["pred_label"] = out.obs[pred_label_key].astype("category")
    return out


def run_umap(adata: AnnData, embedding_key: str = "X_scGPT", random_state: int = 0) -> AnnData:
    sc.pp.neighbors(adata, use_rep=embedding_key)
    sc.tl.umap(adata, random_state=random_state)
    return adata


def save_umap_plot(adata: AnnData, out_path: Path, color: str, title: str) -> None:
    fig = sc.pl.umap(adata, color=color, show=False, return_fig=True)
    if hasattr(fig, "suptitle"):
        fig.suptitle(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_marker_summary(adata: AnnData, groupby: str, n_top: int = 10) -> pd.DataFrame:
    counts = adata.obs[groupby].value_counts()
    valid_groups = counts[counts >= 2].index
    if len(valid_groups) == 0:
        return pd.DataFrame(columns=["group", "rank", "gene", "score", "pval_adj"])

    ad = adata[adata.obs[groupby].isin(valid_groups)].copy()
    ad.obs[groupby] = ad.obs[groupby].astype("category")
    try:
        sc.tl.rank_genes_groups(ad, groupby=groupby, method="wilcoxon")
    except ValueError:
        return pd.DataFrame(columns=["group", "rank", "gene", "score", "pval_adj"])

    frames = []
    for group in ad.obs[groupby].cat.categories:
        df = sc.get.rank_genes_groups_df(ad, group=str(group)).head(n_top).copy()
        df.insert(0, "group", str(group))
        df.insert(1, "rank", range(1, len(df) + 1))
        frames.append(df[["group", "rank", "names", "scores", "pvals_adj"]].rename(columns={"names": "gene", "scores": "score", "pvals_adj": "pval_adj"}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["group", "rank", "gene", "score", "pval_adj"])


def summarize_predictions(adata: AnnData, true_key: str = "true_label", pred_key: str = "pred_label"):
    if "eval_split" in adata.obs:
        subset = adata.obs["eval_split"] == "test"
        obs = adata.obs.loc[subset]
    else:
        obs = adata.obs
    return compute_label_metrics(obs[true_key].cat.codes, obs[pred_key].cat.codes)
