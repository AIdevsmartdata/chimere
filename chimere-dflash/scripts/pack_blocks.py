#!/usr/bin/env python3
"""
pack_blocks.py — Pack individual block .pt files into consolidated numpy memmap files.

Converts 72K+ small .pt files into 3 large memory-mapped arrays for fast sequential I/O.
Speedup: ~50-100x faster DataLoader (no per-file open/close overhead).

Output files:
  - block_input_ids.npy  [N, block_size]       int64
  - block_hidden.npy     [N, k, block_size, H] float32  (main data, ~47GB)
  - prefix_hidden.npy    [N, k, H]             float32
  - prefix_token_ids.npy [N]                   int64
  - pack_meta.json       metadata (N, shapes, etc.)

Usage:
  python scripts/pack_blocks.py --blocks-dir data/blocks --output-dir data/blocks_packed
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-dir", type=str, default="data/blocks")
    parser.add_argument("--output-dir", type=str, default="data/blocks_packed")
    args = parser.parse_args()

    blocks_dir = Path(args.blocks_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(blocks_dir.glob("block_*.pt"))
    N = len(files)
    if N == 0:
        print(f"No block_*.pt files found in {blocks_dir}")
        return

    print(f"Packing {N} blocks from {blocks_dir} → {output_dir}")

    # Probe first block for shapes
    sample = torch.load(files[0], weights_only=True)
    block_size = sample["block_input_ids"].shape[0]
    k, S, H = sample["block_hidden"].shape
    assert S == block_size

    print(f"  block_size={block_size}, k={k}, hidden_dim={H}")
    total_bytes = N * (block_size * 8 + k * block_size * H * 4 + k * H * 4 + 8)
    print(f"  Estimated total: {total_bytes / 1e9:.1f} GB")

    # Create memory-mapped files
    t0 = time.time()

    ids_mmap = np.memmap(
        output_dir / "block_input_ids.npy", dtype=np.int64,
        mode="w+", shape=(N, block_size)
    )
    hidden_mmap = np.memmap(
        output_dir / "block_hidden.npy", dtype=np.float32,
        mode="w+", shape=(N, k, block_size, H)
    )
    prefix_mmap = np.memmap(
        output_dir / "prefix_hidden.npy", dtype=np.float32,
        mode="w+", shape=(N, k, H)
    )
    prefix_ids_mmap = np.memmap(
        output_dir / "prefix_token_ids.npy", dtype=np.int64,
        mode="w+", shape=(N,)
    )

    # Fill in
    for i, f in enumerate(tqdm(files, desc="Packing")):
        data = torch.load(f, weights_only=True)
        ids_mmap[i] = data["block_input_ids"].numpy()
        hidden_mmap[i] = data["block_hidden"].numpy()
        prefix_mmap[i] = data["prefix_hidden"].numpy()
        prefix_ids_mmap[i] = data.get("prefix_token_id", 0)

    # Flush to disk
    ids_mmap.flush()
    hidden_mmap.flush()
    prefix_mmap.flush()
    prefix_ids_mmap.flush()

    elapsed = time.time() - t0
    print(f"\nPacked {N} blocks in {elapsed:.1f}s")

    # Save metadata
    meta = {
        "n_blocks": N,
        "block_size": block_size,
        "num_layers": k,
        "hidden_dim": H,
        "files": {
            "block_input_ids": "block_input_ids.npy",
            "block_hidden": "block_hidden.npy",
            "prefix_hidden": "prefix_hidden.npy",
            "prefix_token_ids": "prefix_token_ids.npy",
        },
    }
    with open(output_dir / "pack_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata: {output_dir / 'pack_meta.json'}")

    # Verify
    print("\nVerification:")
    ids_check = np.memmap(output_dir / "block_input_ids.npy", dtype=np.int64, mode="r", shape=(N, block_size))
    hidden_check = np.memmap(output_dir / "block_hidden.npy", dtype=np.float32, mode="r", shape=(N, k, block_size, H))
    ref = torch.load(files[0], weights_only=True)
    assert np.array_equal(ids_check[0], ref["block_input_ids"].numpy()), "ID mismatch!"
    assert np.allclose(hidden_check[0], ref["block_hidden"].numpy()), "Hidden mismatch!"
    ref_last = torch.load(files[-1], weights_only=True)
    assert np.array_equal(ids_check[-1], ref_last["block_input_ids"].numpy()), "Last ID mismatch!"
    print("  First and last blocks verified OK")

    # Size report
    total_size = sum((output_dir / v).stat().st_size for v in meta["files"].values())
    print(f"  Total packed size: {total_size / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
