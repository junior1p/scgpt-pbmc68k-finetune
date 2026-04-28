"""Shared helpers for scGPT benchmark-style experiments.

Keep this module lightweight so baseline analysis can import dataset helpers
without pulling in the training stack (and its optional torch dependency).
"""

from .datasets import (
    DatasetSpec,
    PairedDatasetSpec,
    PAIRED_DATASET_REGISTRY,
    SCGPT_DATASET_REGISTRY,
    get_dataset_spec,
    get_paired_dataset_spec,
    harmonize_shared_labels,
    intersect_genes,
    is_paired_dataset,
    load_dataset,
    load_paired_dataset,
    prepare_label_column,
    preprocess_for_baseline,
    preprocess_for_scgpt,
    select_shared_hvgs,
    subset_shared_genes,
)
from .metrics import compute_label_metrics

__all__ = [
    "DatasetSpec",
    "PairedDatasetSpec",
    "PAIRED_DATASET_REGISTRY",
    "SCGPT_DATASET_REGISTRY",
    "get_dataset_spec",
    "get_paired_dataset_spec",
    "harmonize_shared_labels",
    "intersect_genes",
    "is_paired_dataset",
    "load_dataset",
    "load_paired_dataset",
    "prepare_label_column",
    "preprocess_for_baseline",
    "preprocess_for_scgpt",
    "select_shared_hvgs",
    "subset_shared_genes",
    "compute_label_metrics",
]
