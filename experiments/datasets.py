from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData

try:
    from scgpt.preprocess import Preprocessor
except Exception:  # pragma: no cover - optional during import-time docs/builds
    Preprocessor = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable[[], AnnData]
    label_key: str
    description: str


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "pbmc68k_reduced": DatasetSpec(
        name="pbmc68k_reduced",
        loader=sc.datasets.pbmc68k_reduced,
        label_key="bulk_labels",
        description="Public PBMC68k reduced dataset with curated bulk_labels annotations.",
    ),
    "paul15": DatasetSpec(
        name="paul15",
        loader=sc.datasets.paul15,
        label_key="paul15_clusters",
        description="Public Paul et al. 2015 hematopoiesis dataset with paul15_clusters annotations.",
    ),
    "pbmc3k_processed": DatasetSpec(
        name="pbmc3k_processed",
        loader=sc.datasets.pbmc3k_processed,
        label_key="louvain",
        description="PBMC3k processed dataset with Louvain cluster labels.",
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def load_dataset(name: str) -> Tuple[AnnData, str, DatasetSpec]:
    spec = get_dataset_spec(name)
    adata = spec.loader()
    if adata.raw is not None:
        raw = adata.raw.to_adata()
    else:
        raw = adata.copy()
    return raw, spec.label_key, spec


def prepare_label_column(adata: AnnData, label_key: str, target_key: str = "celltype") -> np.ndarray:
    labels = adata.obs[label_key].astype("category")
    adata.obs[target_key] = labels
    adata.obs[f"{target_key}_id"] = labels.cat.codes.astype(np.int64)
    return labels.cat.codes.to_numpy(dtype=np.int64)


def preprocess_for_scgpt(adata: AnnData, hvg: int, n_bins: int) -> AnnData:
    if Preprocessor is None:
        raise RuntimeError("scGPT Preprocessor is unavailable in the current environment")
    adata = adata.copy()
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


def preprocess_for_baseline(adata: AnnData, hvg: int) -> AnnData:
    adata = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=hvg, flavor="cell_ranger")
    adata = adata[:, adata.var.highly_variable].copy()
    sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1), svd_solver="arpack")
    return adata
