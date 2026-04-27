"""Shared helpers for scGPT benchmark-style experiments."""

from .datasets import (
    DATASET_REGISTRY,
    DatasetSpec,
    get_dataset_spec,
    load_dataset,
    prepare_label_column,
    preprocess_for_baseline,
    preprocess_for_scgpt,
)
from .metrics import compute_label_metrics
