"""
DFlash v7 — Multi-context dataset for expanded context training.

Supports two formats:
  context_hidden.bin  — float16[n_layers, n_positions, hidden_dim] (ctx_len>1)
  anchor_hidden.bin   — float16[n_layers, hidden_dim] (ctx_len=1, backward compat)

Each sample directory also contains:
  block_tokens.bin    — int32[block_size]
  metadata.json       — {anchor_pos, seq_len, ctx_len, n_positions, ...}
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DFlashMultiCtxDataset(Dataset):
    """Dataset for multi-position context features with single-position fallback.

    Loads context_hidden.bin if present (multi-position),
    otherwise falls back to anchor_hidden.bin (single-position).
    """

    def __init__(self, features_dir, block_size=16, num_layers=5, hidden_dim=2048):
        self.features_dir = Path(features_dir)
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Discover samples — accept either context_hidden.bin or anchor_hidden.bin
        all_dirs = sorted([
            d for d in self.features_dir.iterdir()
            if d.is_dir() and (
                (d / "context_hidden.bin").exists() or
                (d / "anchor_hidden.bin").exists()
            )
        ])

        self.sample_dirs = []
        self.sample_is_multi = []  # True if context_hidden.bin, False if anchor_hidden.bin
        n_bad = 0

        for d in all_dirs:
            ctx_file = d / "context_hidden.bin"
            anchor_file = d / "anchor_hidden.bin"
            bt = d / "block_tokens.bin"

            if not bt.exists():
                n_bad += 1
                continue

            is_multi = ctx_file.exists()

            if is_multi:
                # Validate: size must be n_layers * n_positions * hidden_dim * 2
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    n_pos = meta.get("n_positions", 1)
                    expected_size = num_layers * n_pos * hidden_dim * 2
                    if ctx_file.stat().st_size != expected_size:
                        n_bad += 1
                        continue
                # Spot-check every 500th sample
                if len(self.sample_dirs) % 500 == 0:
                    h = np.fromfile(ctx_file, dtype=np.float16)
                    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                        n_bad += 1
                        continue
            else:
                # Legacy single-position validation
                expected_size = num_layers * hidden_dim * 2
                if anchor_file.stat().st_size != expected_size:
                    n_bad += 1
                    continue
                if len(self.sample_dirs) % 500 == 0:
                    h = np.fromfile(anchor_file, dtype=np.float16)
                    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                        n_bad += 1
                        continue

            self.sample_dirs.append(d)
            self.sample_is_multi.append(is_multi)

        n_multi = sum(self.sample_is_multi)
        n_single = len(self.sample_dirs) - n_multi

        if n_bad > 0:
            print(f"DFlashMultiCtxDataset: skipped {n_bad} bad samples")

        print(f"DFlashMultiCtxDataset: {len(self.sample_dirs)} samples "
              f"({n_multi} multi-ctx, {n_single} single-pos)")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        d = self.sample_dirs[idx]
        is_multi = self.sample_is_multi[idx]

        # Load metadata
        anchor_pos = 0
        n_positions = 1
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            anchor_pos = meta.get("anchor_pos", 0)
            n_positions = meta.get("n_positions", 1)

        # Load hidden states
        if is_multi:
            # context_hidden.bin: float16[n_layers, n_positions, hidden_dim]
            raw = np.fromfile(d / "context_hidden.bin", dtype=np.float16)
            raw = raw.reshape(self.num_layers, n_positions, self.hidden_dim).astype(np.float32)
            context_hidden_list = [
                torch.from_numpy(raw[i])  # [n_positions, H]
                for i in range(self.num_layers)
            ]
            ctx_length = n_positions
        else:
            # anchor_hidden.bin: float16[n_layers, hidden_dim] → expand to [n_layers, 1, hidden_dim]
            raw = np.fromfile(d / "anchor_hidden.bin", dtype=np.float16)
            raw = raw.reshape(self.num_layers, self.hidden_dim).astype(np.float32)
            context_hidden_list = [
                torch.from_numpy(raw[i:i+1])  # [1, H]
                for i in range(self.num_layers)
            ]
            ctx_length = 1

        # Load block tokens: int32[block_size]
        block_tokens = np.fromfile(d / "block_tokens.bin", dtype=np.int32)
        if len(block_tokens) < self.block_size:
            block_tokens = np.pad(block_tokens, (0, self.block_size - len(block_tokens)))

        return {
            "block_input_ids": torch.from_numpy(block_tokens.astype(np.int64)),
            "context_hidden_list": context_hidden_list,
            "context_length": ctx_length,
            "anchor_pos": anchor_pos,
        }


def collate_multi_ctx(batch):
    """Collate multi-context samples with variable ctx_len padding.

    Pads context to max ctx_len in the batch (left-padding with zeros).

    Returns:
        block_input_ids: [B, K]
        context_hidden_list: list of num_layers tensors [B, max_ctx_len, H]
        context_lengths: [B] — actual context lengths per sample
        anchor_positions: [B] — absolute anchor positions for RoPE
    """
    B = len(batch)
    n_layers = len(batch[0]["context_hidden_list"])
    H = batch[0]["context_hidden_list"][0].shape[-1]

    block_input_ids = torch.stack([b["block_input_ids"] for b in batch])
    ctx_lengths_list = [b["context_length"] for b in batch]
    max_ctx_len = max(ctx_lengths_list)
    context_lengths = torch.tensor(ctx_lengths_list, dtype=torch.long)
    anchor_positions = torch.tensor([b.get("anchor_pos", 0) for b in batch], dtype=torch.long)

    context_hidden_list = []
    for layer_idx in range(n_layers):
        padded = torch.zeros(B, max_ctx_len, H)
        for bi in range(B):
            ctx = batch[bi]["context_hidden_list"][layer_idx]  # [ctx_len_i, H]
            ctx_len_i = ctx.shape[0]
            # Right-align: pad on the left so last position = anchor
            padded[bi, max_ctx_len - ctx_len_i:] = ctx
        context_hidden_list.append(padded)

    return {
        "block_input_ids": block_input_ids,
        "context_hidden_list": context_hidden_list,
        "context_lengths": context_lengths,
        "anchor_positions": anchor_positions,
    }
