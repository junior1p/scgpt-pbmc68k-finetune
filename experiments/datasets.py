from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
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


@dataclass(frozen=True)
class PairedDatasetSpec:
    name: str
    reference_name: str
    query_name: str
    reference_loader: Callable[[], AnnData]
    query_loader: Callable[[], AnnData]
    reference_label_key: str
    query_label_key: str
    description: str


SCGPT_DATASET_REGISTRY: Dict[str, DatasetSpec] = {
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


def _cache_path(filename: str) -> Path:
    cache_dir = Path.home() / ".cache" / "scgpt_annotation_benchmark" / "celltypist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / filename


def _load_celltypist_gut_reference() -> AnnData:
    url = "https://cellgeni.cog.sanger.ac.uk/gutcellatlas/Full_obj_raw_counts_nosoupx.h5ad"
    return sc.read(_cache_path("gut_cell_atlas_Elmentaite.h5ad"), backup_url=url)


def _load_celltypist_gut_query() -> AnnData:
    url = "https://cellgeni.cog.sanger.ac.uk/gutcellatlas/Colon_cell_atlas.h5ad"
    return sc.read(_cache_path("gut_cell_atlas_James.h5ad"), backup_url=url)


PAIRED_DATASET_REGISTRY: Dict[str, PairedDatasetSpec] = {
    "gutatlas_transfer": PairedDatasetSpec(
        name="gutatlas_transfer",
        reference_name="gut_cell_atlas_Elmentaite",
        query_name="gut_cell_atlas_James",
        reference_loader=_load_celltypist_gut_reference,
        query_loader=_load_celltypist_gut_query,
        reference_label_key="Integrated_05",
        query_label_key="cell_type",
        description=(
            "CellTypist gut-atlas label-transfer benchmark using the official tutorial pair "
            "(Elmentaite reference, James query) with harmonized overlapping labels."
        ),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    if name not in SCGPT_DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(SCGPT_DATASET_REGISTRY)}")
    return SCGPT_DATASET_REGISTRY[name]


def get_paired_dataset_spec(name: str) -> PairedDatasetSpec:
    if name not in PAIRED_DATASET_REGISTRY:
        raise KeyError(f"Unknown paired dataset '{name}'. Available: {sorted(PAIRED_DATASET_REGISTRY)}")
    return PAIRED_DATASET_REGISTRY[name]


def is_paired_dataset(name: str) -> bool:
    return name in PAIRED_DATASET_REGISTRY


def load_dataset(name: str) -> Tuple[AnnData, str, DatasetSpec]:
    spec = get_dataset_spec(name)
    adata = spec.loader()
    if adata.raw is not None:
        raw = adata.raw.to_adata()
    else:
        raw = adata.copy()
    return raw, spec.label_key, spec


def load_paired_dataset(name: str) -> Tuple[AnnData, AnnData, PairedDatasetSpec]:
    spec = get_paired_dataset_spec(name)
    reference = spec.reference_loader()
    query = spec.query_loader()
    if reference.raw is not None:
        reference = reference.raw.to_adata()
    else:
        reference = reference.copy()
    if query.raw is not None:
        query = query.raw.to_adata()
    else:
        query = query.copy()
    return reference, query, spec


def prepare_label_column(adata: AnnData, label_key: str, target_key: str = "celltype") -> np.ndarray:
    labels = adata.obs[label_key].astype("category")
    adata.obs[target_key] = labels
    adata.obs[f"{target_key}_id"] = labels.cat.codes.astype(np.int64)
    return labels.cat.codes.to_numpy(dtype=np.int64)


def harmonize_shared_labels(
    reference: AnnData,
    query: AnnData,
    reference_label_key: str,
    query_label_key: str,
    target_key: str = "celltype",
) -> Tuple[AnnData, AnnData, List[str]]:
    ref = reference.copy()
    qry = query.copy()
    ref_labels = ref.obs[reference_label_key].astype(str)
    qry_labels = qry.obs[query_label_key].astype(str)
    shared_labels = sorted(set(ref_labels.unique()).intersection(qry_labels.unique()))
    if not shared_labels:
        raise ValueError(
            f"No overlapping labels between reference='{reference_label_key}' and query='{query_label_key}'."
        )
    ref = ref[ref.obs[reference_label_key].astype(str).isin(shared_labels)].copy()
    qry = qry[qry.obs[query_label_key].astype(str).isin(shared_labels)].copy()
    ref.obs[target_key] = pd.Categorical(ref.obs[reference_label_key].astype(str), categories=shared_labels)
    qry.obs[target_key] = pd.Categorical(qry.obs[query_label_key].astype(str), categories=shared_labels)
    ref.obs[f"{target_key}_id"] = ref.obs[target_key].cat.codes.astype(np.int64)
    qry.obs[f"{target_key}_id"] = qry.obs[target_key].cat.codes.astype(np.int64)
    return ref, qry, shared_labels


def intersect_genes(reference: AnnData, query: AnnData) -> List[str]:
    query_genes = set(query.var_names.astype(str))
    shared = [gene for gene in reference.var_names.astype(str) if gene in query_genes]
    if not shared:
        raise ValueError("No shared genes between reference and query datasets.")
    return shared


def select_shared_hvgs(reference: AnnData, query: AnnData, n_hvg: int) -> List[str]:
    ref = reference.copy()
    if ref.raw is not None:
        ref = ref.raw.to_adata()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    top_n = min(n_hvg, ref.n_vars)
    sc.pp.highly_variable_genes(ref, n_top_genes=top_n, flavor="cell_ranger")
    candidate_genes = [gene for gene, keep in zip(ref.var_names.astype(str), ref.var["highly_variable"].to_numpy()) if keep]
    query_genes = set(query.var_names.astype(str))
    shared = [gene for gene in candidate_genes if gene in query_genes]
    if len(shared) < min(top_n, len(candidate_genes)):
        fallback = [gene for gene in reference.var_names.astype(str) if gene in query_genes and gene not in shared]
        for gene in fallback:
            shared.append(gene)
            if len(shared) >= top_n:
                break
    if not shared:
        raise ValueError("Failed to select shared HVGs for annotation benchmark.")
    return shared[:top_n]


def subset_shared_genes(reference: AnnData, query: AnnData, genes: List[str]) -> Tuple[AnnData, AnnData]:
    ref = reference[:, genes].copy()
    qry = query[:, genes].copy()
    return ref, qry


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
