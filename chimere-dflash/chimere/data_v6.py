"""
DFlash v6 — Single-position dataset for anchor-conditioned training.

Each sample directory contains:
  anchor_hidden.bin  — float16[5, 2048] (5 layers × hidden_dim at anchor position)
  block_tokens.bin   — int32[16] (next block_size tokens after anchor)
  metadata.json      — {anchor_pos, seq_len, layers, hidden_dim, dtype}

The model receives ctx_len=1: the anchor position's fused hidden state.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DFlashSinglePosDataset(Dataset):
    """Dataset for single-position anchor features.

    Each sample is one (anchor_hidden, block_tokens) pair.
    No random sampling needed — positions were chosen at extraction time.
    """

    def __init__(self, features_dir, block_size=16, num_layers=5, hidden_dim=2048):
        self.features_dir = Path(features_dir)
        self.block_size = block_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Discover and validate samples
        expected_size = num_layers * hidden_dim * 2  # float16 bytes
        all_dirs = sorted([
            d for d in self.features_dir.iterdir()
            if d.is_dir() and (d / "anchor_hidden.bin").exists()
        ])

        self.sample_dirs = []
        n_bad = 0
        for d in all_dirs:
            ah = d / "anchor_hidden.bin"
            bt = d / "block_tokens.bin"
            if ah.stat().st_size != expected_size or not bt.exists():
                n_bad += 1
                continue
            # Spot-check every 500th sample for NaN/Inf
            if len(self.sample_dirs) % 500 == 0:
                h = np.fromfile(ah, dtype=np.float16)
                if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                    n_bad += 1
                    continue
            self.sample_dirs.append(d)

        if n_bad > 0:
            print(f"DFlashSinglePosDataset: skipped {n_bad} bad samples")

        print(f"DFlashSinglePosDataset: {len(self.sample_dirs)} samples, "
              f"block_size={block_size}, ctx_len=1")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        d = self.sample_dirs[idx]

        # Load anchor hidden: float16[5, 2048] → float32
        anchor = np.fromfile(d / "anchor_hidden.bin", dtype=np.float16)
        anchor = anchor.reshape(self.num_layers, self.hidden_dim).astype(np.float32)

        # Load block tokens: int32[16]
        block_tokens = np.fromfile(d / "block_tokens.bin", dtype=np.int32)
        if len(block_tokens) < self.block_size:
            block_tokens = np.pad(block_tokens, (0, self.block_size - len(block_tokens)))

        # Load anchor_pos from metadata (for absolute RoPE positions)
        anchor_pos = 0
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            anchor_pos = meta.get("anchor_pos", 0)

        # Build context_hidden_list: one [1, H] tensor per layer
        context_hidden_list = [
            torch.from_numpy(anchor[i:i+1])  # [1, H]
            for i in range(self.num_layers)
        ]

        return {
            "block_input_ids": torch.from_numpy(block_tokens.astype(np.int64)),
            "context_hidden_list": context_hidden_list,
            "context_length": 1,
            "anchor_pos": anchor_pos,
        }


def collate_single_pos(batch):
    """Collate single-position samples. ctx_len is always 1, no padding needed.

    Returns:
        block_input_ids: [B, K]
        context_hidden_list: list of num_layers tensors [B, 1, H]
        context_lengths: [B] (all ones)
    """
    B = len(batch)
    n_layers = len(batch[0]["context_hidden_list"])

    block_input_ids = torch.stack([b["block_input_ids"] for b in batch])
    context_lengths = torch.ones(B, dtype=torch.long)
    anchor_positions = torch.tensor([b.get("anchor_pos", 0) for b in batch], dtype=torch.long)

    context_hidden_list = []
    for layer_idx in range(n_layers):
        stacked = torch.stack([
            b["context_hidden_list"][layer_idx] for b in batch
        ])  # [B, 1, H]
        context_hidden_list.append(stacked)

    return {
        "block_input_ids": block_input_ids,
        "context_hidden_list": context_hidden_list,
        "context_lengths": context_lengths,
        "anchor_positions": anchor_positions,
    }
