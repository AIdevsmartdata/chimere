#!/usr/bin/env python3
"""
prepare_blocks.py — Convert extracted hidden states into DFlash training blocks.

Reads the output of extract_hidden_states (per-sample binary files) and creates
fixed-size blocks of (prefix_hidden_states, block_token_ids) for drafter training.

Input:  features/<sample_id>/layer_*.bin + tokens.bin + meta.json
Output: blocks/<block_id>.pt — PyTorch tensors ready for DataLoader

Each .pt file contains:
  - 'block_input_ids': LongTensor [block_size]     — target token IDs
  - 'prefix_hidden':   FloatTensor [k, hidden_dim]  — k hidden states from target layers
                                                       at the prefix position (last token
                                                       before the block)
  - 'block_hidden':    FloatTensor [k, block_size, hidden_dim] — hidden states for each
                                                                    token in the block
                                                                    (for KV injection training)
  - 'prefix_token_id': int                          — token ID at prefix position

Usage:
  python prepare_blocks.py \\
    --features-dir data/features \\
    --output-dir data/blocks \\
    --block-size 16 \\
    --stride 8
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def load_sample(sample_dir: Path, hidden_dim: int, layers: list):
    """Load one sample's extracted features."""
    # Support both naming conventions (metadata.json / meta.json)
    meta_path = sample_dir / "metadata.json"
    if not meta_path.exists():
        meta_path = sample_dir / "meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    n_tokens = meta.get("seq_len", meta.get("n_tokens"))
    h_dim = meta["hidden_dim"]

    # Load tokens (support both input_ids.bin and tokens.bin)
    tokens_path = sample_dir / "input_ids.bin"
    if not tokens_path.exists():
        tokens_path = sample_dir / "tokens.bin"
    tokens = np.fromfile(tokens_path, dtype=np.int32)
    # C++ extractor may save all tokenized input_ids but only process
    # the first n_tokens through the eval callback. Truncate to match.
    if len(tokens) > n_tokens:
        tokens = tokens[:n_tokens]
    elif len(tokens) < n_tokens:
        print(f"  [WARN] Too few tokens in {sample_dir.name}: {len(tokens)} < {n_tokens}")
        return None

    # Load hidden states per layer (try zero-padded and plain names)
    layer_hidden = {}
    for layer_id in layers:
        layer_path = sample_dir / f"layer_{layer_id:02d}.bin"
        if not layer_path.exists():
            layer_path = sample_dir / f"layer_{layer_id}.bin"
        if not layer_path.exists():
            print(f"  [WARN] Missing layer {layer_id} for {sample_dir.name}")
            continue
        data = np.fromfile(layer_path, dtype=np.float32)
        data = data.reshape(n_tokens, h_dim)
        layer_hidden[layer_id] = data

    if len(layer_hidden) != len(layers):
        return None

    return {
        "tokens": tokens,
        "layer_hidden": layer_hidden,
        "n_tokens": n_tokens,
        "hidden_dim": h_dim,
        "layers": sorted(layer_hidden.keys()),
    }


def create_blocks(sample: dict, block_size: int, stride: int):
    """Slice a sample into shifted blocks for training.

    Context hidden states come from block_size positions BEFORE the block,
    matching the online inference setup where hidden states are from the
    verified prefix and the drafter predicts the next block of tokens.

    Layout:
      context hidden: positions [start - block_size, start)
      prediction ids:  positions [start, start + block_size)
    """
    blocks = []
    n_tokens = sample["n_tokens"]
    layers = sample["layers"]

    # Need block_size for context + block_size for prediction
    min_len = 2 * block_size + 1
    if n_tokens < min_len:
        return blocks

    for start in range(block_size, n_tokens - block_size + 1, stride):
        ctx_start = start - block_size
        prefix_pos = start - 1
        block_end = start + block_size

        block_ids = sample["tokens"][start:block_end]

        prefix_hidden = np.stack([
            sample["layer_hidden"][l][prefix_pos]
            for l in layers
        ])  # [k, hidden_dim]

        # SHIFTED: context hidden states from BEFORE the prediction block
        block_hidden = np.stack([
            sample["layer_hidden"][l][ctx_start:start]
            for l in layers
        ])  # [k, block_size, hidden_dim]

        blocks.append({
            "block_input_ids": torch.from_numpy(block_ids.astype(np.int64)),
            "prefix_hidden": torch.from_numpy(prefix_hidden.astype(np.float32)),
            "block_hidden": torch.from_numpy(block_hidden.astype(np.float32)),
            "prefix_token_id": int(sample["tokens"][prefix_pos]),
        })

    return blocks


def main():
    parser = argparse.ArgumentParser(description="Convert extracted features to DFlash training blocks")
    parser.add_argument("--features-dir", type=str, required=True,
                        help="Directory containing extracted features")
    parser.add_argument("--output-dir", type=str, default="data/blocks",
                        help="Output directory for .pt block files")
    parser.add_argument("--block-size", type=int, default=16,
                        help="Block size (number of tokens per block)")
    parser.add_argument("--stride", type=int, default=8,
                        help="Stride between consecutive blocks (overlap = block_size - stride)")
    parser.add_argument("--layers", type=str, default="2,11,20,29,37",
                        help="Comma-separated layer indices (must match extraction)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to process (0 = all)")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(x) for x in args.layers.split(",")]

    sample_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir()])
    if args.max_samples > 0:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"Found {len(sample_dirs)} samples")
    print(f"Block size: {args.block_size}, stride: {args.stride}")
    print(f"Target layers: {layers}")
    print()

    block_id = 0
    total_tokens = 0
    failed = 0

    for sample_dir in tqdm(sample_dirs, desc="Processing samples"):
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            failed += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        sample = load_sample(sample_dir, meta["hidden_dim"], layers)
        if sample is None:
            failed += 1
            continue

        blocks = create_blocks(sample, args.block_size, args.stride)
        for block in blocks:
            torch.save(block, output_dir / f"block_{block_id:06d}.pt")
            block_id += 1

        total_tokens += sample["n_tokens"]

    dataset_meta = {
        "n_blocks": block_id,
        "block_size": args.block_size,
        "stride": args.stride,
        "layers": layers,
        "n_samples": len(sample_dirs) - failed,
        "n_failed": failed,
        "total_tokens": total_tokens,
    }

    with open(output_dir / "dataset_meta.json", "w") as f:
        json.dump(dataset_meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Blocks created: {block_id}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Samples processed: {len(sample_dirs) - failed}/{len(sample_dirs)}")
    print(f"  Avg blocks/sample: {block_id / max(1, len(sample_dirs) - failed):.1f}")
    print(f"  Output: {output_dir}")
    print(f"  Disk usage: ~{block_id * args.block_size * meta.get('hidden_dim', 2048) * len(layers) * 4 / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
