from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .datasets import (
    harmonize_shared_labels,
    load_paired_dataset,
    preprocess_for_scgpt,
    select_shared_hvgs,
    subset_shared_genes,
)


@dataclass(frozen=True)
class TransferBundle:
    reference: AnnData
    query: AnnData
    shared_labels: List[str]
    shared_genes: List[str]
    label_key: str = "celltype"


class AnnDataset(Dataset):
    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data

    def __len__(self) -> int:
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.data.items()}


def split_reference_indices(labels: np.ndarray, seed: int, test_size: float, val_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=labels,
    )
    holdout_labels = labels[temp_idx]
    rel_test = test_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=rel_test,
        random_state=seed,
        stratify=holdout_labels,
    )
    return train_idx, val_idx, test_idx


def prepare_transfer_bundle(dataset_name: str, hvg: int, n_bins: int) -> TransferBundle:
    reference, query, spec = load_paired_dataset(dataset_name)
    reference, query, shared_labels = harmonize_shared_labels(
        reference,
        query,
        spec.reference_label_key,
        spec.query_label_key,
    )
    shared_genes = select_shared_hvgs(reference, query, hvg)
    reference, query = subset_shared_genes(reference, query, shared_genes)
    reference = preprocess_for_scgpt(reference, hvg=len(shared_genes), n_bins=n_bins)
    query = preprocess_for_scgpt(query, hvg=len(shared_genes), n_bins=n_bins)
    return TransferBundle(reference=reference, query=query, shared_labels=shared_labels, shared_genes=shared_genes)


def make_loader(data: dict[str, torch.Tensor], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        AnnDataset(data),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def encode_labels(labels: pd.Categorical, categories: List[str]) -> np.ndarray:
    ordered = pd.Categorical(labels.astype(str), categories=categories)
    return ordered.codes.astype(np.int64)
