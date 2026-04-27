import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import issparse
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

sys.path.append("/mnt/scgpt_trial")

from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.tokenizer import random_mask_value, tokenize_and_pad_batch


def build_synthetic_counts(n_cells=256, n_genes=512, seed=0):
    rng = np.random.default_rng(seed)
    # low-rank-ish synthetic gene expression counts
    cell_factors = rng.gamma(shape=2.0, scale=1.0, size=(n_cells, 8))
    gene_factors = rng.gamma(shape=2.0, scale=1.0, size=(8, n_genes))
    lam = cell_factors @ gene_factors / 4.0
    lam = np.clip(lam, 0.05, 8.0)
    counts = rng.poisson(lam=lam).astype(np.float32)
    return counts


def normalize_and_bin(counts, n_bins=51):
    counts = counts.astype(np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    normed = counts / totals * 1e4
    logged = np.log1p(normed)
    # bin each cell independently to [0, n_bins-1]
    binned = np.zeros_like(logged, dtype=np.float32)
    for i in range(logged.shape[0]):
        row = logged[i]
        if np.allclose(row.max(), row.min()):
            continue
        edges = np.quantile(row, np.linspace(0, 1, n_bins))
        edges = np.unique(edges)
        if len(edges) < 2:
            continue
        # np.digitize expects interior edges
        bins = np.digitize(row, edges[1:-1], right=True)
        binned[i] = bins.astype(np.float32)
    return binned


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(min(16, os.cpu_count() or 1))

    outdir = Path("/mnt/scgpt_trial/quick_run")
    outdir.mkdir(parents=True, exist_ok=True)

    print("Building synthetic dataset...")
    counts = build_synthetic_counts(n_cells=256, n_genes=512, seed=0)
    binned = normalize_and_bin(counts, n_bins=51)
    genes = [f"GENE{i}" for i in range(binned.shape[1])]

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab[pad_token])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized = tokenize_and_pad_batch(
        binned,
        gene_ids,
        max_len=len(genes) + 1,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True,
    )
    print("Tokenized genes shape:", tokenized["genes"].shape)

    mask_ratio = 0.25
    mask_value = -1
    pad_value = -2
    masked_values = random_mask_value(
        tokenized["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    data_pt = {
        "gene_ids": tokenized["genes"],
        "values": masked_values,
        "target_values": tokenized["values"],
        "batch_labels": torch.zeros(tokenized["genes"].shape[0], dtype=torch.long),
    }

    loader = DataLoader(SeqDataset(data_pt), batch_size=32, shuffle=True, drop_last=False)

    device = torch.device("cpu")
    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        d_model=64,
        nhead=2,
        d_hid=64,
        nlayers=2,
        nlayers_cls=1,
        n_cls=1,
        vocab=vocab,
        dropout=0.1,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=1,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=51,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = masked_mse_loss
    epochs = 2
    print("Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        start = time.time()
        for batch in loader:
            input_gene_ids = batch["gene_ids"].to(device)
            input_values = batch["values"].to(device)
            target_values = batch["target_values"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            out = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
                MVC=False,
                ECS=False,
            )
            masked_positions = input_values.eq(mask_value)
            loss = criterion(out["mlm_output"], target_values, masked_positions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            count += 1
        print(f"epoch={epoch} loss={total/max(count,1):.5f} elapsed={time.time()-start:.1f}s")

    ckpt = outdir / "quick_scgpt_cpu.pt"
    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)

    batch = next(iter(loader))
    with torch.no_grad():
        out = model(
            batch["gene_ids"].to(device),
            batch["values"].to(device),
            src_key_padding_mask=batch["gene_ids"].to(device).eq(vocab[pad_token]),
            batch_labels=None,
            MVC=False,
            ECS=False,
        )
    print("Sanity output keys:", sorted(out.keys()))
    print("mlm_output shape:", tuple(out["mlm_output"].shape))


if __name__ == "__main__":
    main()
