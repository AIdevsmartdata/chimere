"""Dataset class for loading pre-extracted DFlash training blocks."""
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class DFlashDataset(Dataset):
    """
    Loads pre-extracted training blocks.

    Supports two storage formats:
      1. Packed memmap (fast): blocks_dir contains pack_meta.json + .npy memmap files
      2. Individual .pt files (slow): blocks_dir contains block_*.pt files

    Use scripts/pack_blocks.py to convert from format 2 → 1 for ~50x I/O speedup.
    """

    def __init__(self, blocks_dir, block_size=16):
        self.blocks_dir = Path(blocks_dir)
        self.block_size = block_size

        meta_path = self.blocks_dir / "pack_meta.json"
        if meta_path.exists():
            self._init_memmap(meta_path)
        else:
            self._init_files()

    def _init_memmap(self, meta_path):
        """Fast path: memory-mapped numpy arrays."""
        with open(meta_path) as f:
            meta = json.load(f)

        self.mode = "memmap"
        self.n_blocks = meta["n_blocks"]
        bs = meta["block_size"]
        k = meta["num_layers"]
        H = meta["hidden_dim"]

        self.ids_mmap = np.memmap(
            self.blocks_dir / meta["files"]["block_input_ids"],
            dtype=np.int64, mode="r", shape=(self.n_blocks, bs)
        )
        self.hidden_mmap = np.memmap(
            self.blocks_dir / meta["files"]["block_hidden"],
            dtype=np.float32, mode="r", shape=(self.n_blocks, k, bs, H)
        )
        self.prefix_mmap = np.memmap(
            self.blocks_dir / meta["files"]["prefix_hidden"],
            dtype=np.float32, mode="r", shape=(self.n_blocks, k, H)
        )
        print(f"DFlashDataset: {self.n_blocks} blocks from {self.blocks_dir} (memmap, fast)")

    def _init_files(self):
        """Slow path: individual .pt files."""
        self.mode = "files"
        self.files = sorted(self.blocks_dir.glob("block_*.pt"))
        if not self.files:
            raise ValueError(f"No block_*.pt or pack_meta.json found in {self.blocks_dir}")
        self.n_blocks = len(self.files)
        print(f"DFlashDataset: {self.n_blocks} blocks from {self.blocks_dir} (individual files, slow)")

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        if self.mode == "memmap":
            return {
                "block_input_ids": torch.from_numpy(self.ids_mmap[idx].copy()),
                "block_hidden": torch.from_numpy(self.hidden_mmap[idx].copy()),
                "prefix_hidden": torch.from_numpy(self.prefix_mmap[idx].copy()),
            }
        else:
            data = torch.load(self.files[idx], weights_only=True)
            return {
                "block_input_ids": data["block_input_ids"],
                "block_hidden": data["block_hidden"],
                "prefix_hidden": data["prefix_hidden"],
            }


def collate_dflash(batch):
    """
    Collate function that stacks blocks and splits hidden states
    into the list-of-tensors format expected by FeatureFusion.

    Returns:
        input_ids: [B, block_size] int64
        target_hidden_states_list: list of k tensors [B, block_size, hidden_dim]
        prefix_hidden_list: list of k tensors [B, hidden_dim]
    """
    input_ids = torch.stack([b["block_input_ids"] for b in batch])
    block_hidden = torch.stack([b["block_hidden"] for b in batch])    # [B, k, S, H]
    prefix_hidden = torch.stack([b["prefix_hidden"] for b in batch])  # [B, k, H]

    k = block_hidden.shape[1]

    # Unpack to list of k tensors [B, S, H] — what FeatureFusion expects
    target_hidden_states_list = [block_hidden[:, i, :, :] for i in range(k)]

    # Prefix hidden: list of k tensors [B, H] (for potential prefix conditioning)
    prefix_hidden_list = [prefix_hidden[:, i, :] for i in range(k)]

    return {
        "input_ids": input_ids,
        "target_hidden_states_list": target_hidden_states_list,
        "prefix_hidden_list": prefix_hidden_list,
    }
