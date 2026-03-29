"""
DFlash v5 — Sequence-level dataset for full-context KV injection training.

Instead of isolated fixed-size blocks, each sample loads a full sequence's
hidden states and randomly samples a block position. The context (all hidden
states before the block) is provided for cross-attention KV injection.

Reads directly from the existing features directory:
  data/features/<sample_id>/layer_*.bin + input_ids.bin + metadata.json
"""
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DFlashSeqDataset(Dataset):
    """Sequence-level dataset for DFlash v5 training.

    Each __getitem__ call:
    1. Picks a sequence (sample)
    2. Randomly samples an anchor position within the sequence
    3. Returns the block tokens [anchor..anchor+K) and all context hidden
       states [0..anchor] from each target layer.
    """

    def __init__(self, features_dir, block_size=16,
                 target_layers=(2, 11, 20, 29, 37),
                 blocks_per_seq=20, max_ctx_len=512,
                 cache_size=200):
        """
        Args:
            features_dir: Path to extracted features (per-sample directories)
            block_size: Number of tokens per draft block
            target_layers: Which target model layers to use
            blocks_per_seq: Virtual multiplier — how many blocks to sample
                           per sequence per epoch
            max_ctx_len: Maximum context length (clip older positions)
            cache_size: Number of samples to keep in LRU cache
        """
        self.features_dir = Path(features_dir)
        self.block_size = block_size
        self.target_layers = list(target_layers)
        self.blocks_per_seq = blocks_per_seq
        self.max_ctx_len = max_ctx_len

        # Discover valid samples
        self.sample_dirs = []
        self.metas = []

        n_nan_skipped = 0
        all_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])
        for d in all_dirs:
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            n_tokens = meta.get("seq_len", meta.get("n_tokens", 0))
            # Need at least 1 context token + block_size tokens
            if n_tokens <= block_size + 1:
                continue
            # Skip samples with NaN/Inf in hidden states (MXFP4 overflow on long seqs)
            h_dim = meta["hidden_dim"]
            has_bad = False
            for l in self.target_layers:
                layer_path = d / f"layer_{l:02d}.bin"
                if not layer_path.exists():
                    layer_path = d / f"layer_{l}.bin"
                if not layer_path.exists():
                    has_bad = True
                    break
                data = np.fromfile(layer_path, dtype=np.float32)
                if np.isnan(data).any() or np.isinf(data).any():
                    has_bad = True
                    break
            if has_bad:
                n_nan_skipped += 1
                continue
            self.sample_dirs.append(d)
            self.metas.append(meta)

        # LRU cache for loaded samples
        self._cache = OrderedDict()
        self._cache_size = cache_size

        n_seq = len(self.sample_dirs)
        total_tokens = sum(
            m.get("seq_len", m.get("n_tokens", 0)) for m in self.metas
        )
        if n_nan_skipped > 0:
            print(f"DFlashSeqDataset: skipped {n_nan_skipped} samples with NaN/Inf")
        print(f"DFlashSeqDataset: {n_seq} sequences, {total_tokens} tokens, "
              f"block_size={block_size}, max_ctx={max_ctx_len}, "
              f"{n_seq * blocks_per_seq} virtual items/epoch")

    def __len__(self):
        return len(self.sample_dirs) * self.blocks_per_seq

    def _load_sample(self, sample_idx):
        """Load a sample's tokens and hidden states, with LRU caching."""
        if sample_idx in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(sample_idx)
            return self._cache[sample_idx]

        d = self.sample_dirs[sample_idx]
        meta = self.metas[sample_idx]
        n_tokens = meta.get("seq_len", meta.get("n_tokens", 0))
        h_dim = meta["hidden_dim"]

        # Load tokens
        tokens_path = d / "input_ids.bin"
        if not tokens_path.exists():
            tokens_path = d / "tokens.bin"
        tokens = np.fromfile(tokens_path, dtype=np.int32)[:n_tokens]

        # Load hidden states per layer
        layer_hidden = {}
        for l in self.target_layers:
            layer_path = d / f"layer_{l:02d}.bin"
            if not layer_path.exists():
                layer_path = d / f"layer_{l}.bin"
            data = np.fromfile(layer_path, dtype=np.float32).reshape(n_tokens, h_dim)
            layer_hidden[l] = data

        sample = {
            "tokens": tokens,
            "layer_hidden": layer_hidden,
            "n_tokens": n_tokens,
        }

        # LRU eviction
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[sample_idx] = sample
        return sample

    def __getitem__(self, idx):
        sample_idx = idx % len(self.sample_dirs)
        sample = self._load_sample(sample_idx)
        n_tokens = sample["n_tokens"]

        # Random anchor position: need block_size tokens after it
        max_anchor = n_tokens - self.block_size
        min_anchor = 1  # at least 1 context token
        if max_anchor < min_anchor:
            max_anchor = min_anchor
        anchor_pos = random.randint(min_anchor, max_anchor)

        # Block tokens: [anchor_pos, anchor_pos + block_size)
        block_end = min(anchor_pos + self.block_size, n_tokens)
        block_tokens = sample["tokens"][anchor_pos:block_end]
        if len(block_tokens) < self.block_size:
            block_tokens = np.pad(block_tokens, (0, self.block_size - len(block_tokens)))

        # Context: hidden states from positions [ctx_start, anchor_pos+1)
        # anchor_pos+1 includes the anchor position itself (verified token)
        ctx_end = anchor_pos + 1
        ctx_start = max(0, ctx_end - self.max_ctx_len)
        ctx_len = ctx_end - ctx_start

        # Build context hidden list: one tensor per target layer
        context_hidden_list = []
        for l in self.target_layers:
            h = sample["layer_hidden"][l][ctx_start:ctx_end]  # [ctx_len, 2048]
            context_hidden_list.append(
                torch.from_numpy(h.astype(np.float32))
            )

        return {
            "block_input_ids": torch.from_numpy(block_tokens.astype(np.int64)),
            "context_hidden_list": context_hidden_list,
            "context_length": ctx_len,
        }


def collate_v5(batch):
    """Collate variable-length context sequences with padding.

    Returns:
        block_input_ids: [B, K]
        context_hidden_list: list of n_layers tensors [B, max_ctx_len, H] (padded)
        context_lengths: [B] int64
    """
    B = len(batch)
    n_layers = len(batch[0]["context_hidden_list"])
    hidden_dim = batch[0]["context_hidden_list"][0].shape[-1]
    max_ctx_len = max(b["context_length"] for b in batch)

    block_input_ids = torch.stack([b["block_input_ids"] for b in batch])
    context_lengths = torch.tensor(
        [b["context_length"] for b in batch], dtype=torch.long
    )

    # Pad context hidden states per layer
    context_hidden_list = []
    for layer_idx in range(n_layers):
        padded = torch.zeros(B, max_ctx_len, hidden_dim)
        for i, b in enumerate(batch):
            ctx_len = b["context_length"]
            padded[i, :ctx_len] = b["context_hidden_list"][layer_idx]
        context_hidden_list.append(padded)

    return {
        "block_input_ids": block_input_ids,
        "context_hidden_list": context_hidden_list,
        "context_lengths": context_lengths,
    }
