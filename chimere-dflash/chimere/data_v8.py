"""
DFlash v8 — Full-sequence multi-anchor dataset with packed shard storage.

Storage format (per shard):
  shard_XXXX.bin  — packed binary: [hidden_0, tokens_0, hidden_1, tokens_1, ...]
                    hidden_i = float16[n_layers, seq_len_i, hidden_dim]
                    tokens_i = int32[seq_len_i]
  shard_XXXX.idx  — JSON index: list of {offset, seq_len, hidden_bytes, token_bytes}

Training behavior:
  Each __getitem__ loads one full sequence, then samples `anchors_per_seq`
  random anchor positions. Each anchor defines:
    - context: hidden states [0..anchor] from each layer (clipped to max_ctx_len)
    - block: tokens [anchor..anchor+block_size]
    - anchor_pos: absolute position for RoPE

  Returns a list of anchors that get flattened in the collate function,
  so effective batch_size = loader_batch_size * anchors_per_seq.

Packing script:
  Use pack_shards_v8() to convert per-sample directories to packed format.
"""
import json
import mmap
import os
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shard packing utilities (extraction → packed format)
# ---------------------------------------------------------------------------

def pack_shards_v8(
    features_dir: str,
    output_dir: str,
    layers: List[int] = (1, 10, 19, 28, 37),
    hidden_dim: int = 2048,
    max_seq_len: int = 1024,
    samples_per_shard: int = 200,
    dtype_str: str = "float16",
):
    """Convert per-sample directories (v6 format) to packed shards.

    Reads from features_dir/sample_XXXXXX/{layer_XX.bin, input_ids.bin, metadata.json}
    and writes to output_dir/shard_XXXX.{bin,idx}.

    Args:
        features_dir: directory with per-sample subdirectories
        output_dir: destination for packed shards
        layers: target layer indices to store
        hidden_dim: hidden dimension per layer
        max_seq_len: clip sequences beyond this length
        samples_per_shard: how many samples per shard file
        dtype_str: "float16" or "float32" for hidden state storage
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(layers)
    dtype = np.float16 if dtype_str == "float16" else np.float32
    bytes_per_element = 2 if dtype_str == "float16" else 4

    sample_dirs = sorted([
        d for d in features_dir.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    ])

    print(f"pack_shards_v8: {len(sample_dirs)} sample directories found")
    print(f"  layers={layers}, hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")
    print(f"  samples_per_shard={samples_per_shard}, dtype={dtype_str}")

    shard_idx = 0
    shard_samples = []
    shard_bin = None
    shard_offset = 0
    total_written = 0
    total_skipped = 0

    def _flush_shard():
        nonlocal shard_bin, shard_samples, shard_idx, shard_offset
        if shard_bin is not None:
            shard_bin.close()
            idx_path = output_dir / f"shard_{shard_idx:04d}.idx"
            with open(idx_path, "w") as f:
                json.dump({
                    "n_layers": n_layers,
                    "layers": layers,
                    "hidden_dim": hidden_dim,
                    "dtype": dtype_str,
                    "samples": shard_samples,
                }, f)
            shard_idx += 1
            shard_samples = []
            shard_offset = 0

    for si, sdir in enumerate(sample_dirs):
        if (si + 1) % 500 == 0:
            print(f"  [{si+1}/{len(sample_dirs)}] written={total_written}, "
                  f"skipped={total_skipped}", flush=True)

        # Load metadata
        with open(sdir / "metadata.json") as f:
            meta = json.load(f)
        seq_len = meta.get("seq_len", meta.get("n_tokens", 0))
        if seq_len < 18:  # need at least 1 context + block_size=16 + 1
            total_skipped += 1
            continue

        # Clip to max_seq_len
        clip_len = min(seq_len, max_seq_len)

        # Load and validate hidden states
        layer_data = []
        valid = True
        for l in layers:
            lpath = sdir / f"layer_{l:02d}.bin"
            if not lpath.exists():
                valid = False
                break
            # Detect stored dtype from file size
            file_size = lpath.stat().st_size
            stored_elements = file_size // 2  # assume float16 first
            if stored_elements == seq_len * hidden_dim:
                h = np.fromfile(lpath, dtype=np.float16).reshape(seq_len, hidden_dim)
            else:
                stored_elements = file_size // 4
                if stored_elements == seq_len * hidden_dim:
                    h = np.fromfile(lpath, dtype=np.float32).reshape(seq_len, hidden_dim)
                else:
                    valid = False
                    break

            h = h[:clip_len]
            if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                valid = False
                break
            layer_data.append(h.astype(dtype))

        if not valid:
            total_skipped += 1
            continue

        # Load tokens
        tokens_path = sdir / "input_ids.bin"
        if not tokens_path.exists():
            tokens_path = sdir / "tokens.bin"
        if not tokens_path.exists():
            total_skipped += 1
            continue
        tokens = np.fromfile(tokens_path, dtype=np.int32)[:clip_len]

        # Open new shard if needed
        if shard_bin is None or len(shard_samples) >= samples_per_shard:
            _flush_shard()
            bin_path = output_dir / f"shard_{shard_idx:04d}.bin"
            shard_bin = open(bin_path, "wb")
            shard_offset = 0

        # Write hidden states: [n_layers, clip_len, hidden_dim] interleaved
        hidden_bytes = n_layers * clip_len * hidden_dim * bytes_per_element
        for h in layer_data:
            shard_bin.write(h.tobytes())

        # Write tokens: int32[clip_len]
        token_bytes = clip_len * 4
        shard_bin.write(tokens.tobytes())

        shard_samples.append({
            "offset": shard_offset,
            "seq_len": clip_len,
            "original_seq_len": seq_len,
            "hidden_bytes": hidden_bytes,
            "token_bytes": token_bytes,
        })
        shard_offset += hidden_bytes + token_bytes
        total_written += 1

    _flush_shard()

    # Write global manifest
    manifest = {
        "n_samples": total_written,
        "n_shards": shard_idx,
        "n_layers": n_layers,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "dtype": dtype_str,
        "max_seq_len": max_seq_len,
        "samples_per_shard": samples_per_shard,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(
        f.stat().st_size for f in output_dir.glob("shard_*.bin")
    )
    print(f"\npack_shards_v8 complete:")
    print(f"  {total_written} samples in {shard_idx} shards")
    print(f"  {total_skipped} skipped")
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")
    print(f"  Per sample: {total_bytes / max(1, total_written) / 1e6:.1f} MB")

    return manifest


def pack_shards_v2(
    features_dir: str,
    output_dir: str,
    max_seq_len: int = 1024,
    samples_per_shard: int = 200,
):
    """Convert V2 per-sample directories (--extract-all format) to packed shards.

    Reads from features_dir/sample_XXXXXX/{context_hidden.bin, tokens.bin, metadata.json}
    and writes to output_dir/shard_XXXX.{bin,idx}.

    Unlike pack_shards_v8 (V1 format with layer_XX.bin), this reads the V2 format
    where context_hidden.bin already contains all layers stacked as float16[n_layers, seq_len, hidden_dim].

    Args:
        features_dir: directory with per-sample subdirectories (V2 format)
        output_dir: destination for packed shards
        max_seq_len: clip sequences beyond this length
        samples_per_shard: how many samples per shard file
    """
    features_dir = Path(features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([
        d for d in features_dir.iterdir()
        if d.is_dir() and (d / "context_hidden.bin").exists() and (d / "tokens.bin").exists()
    ])

    if not sample_dirs:
        print("pack_shards_v2: no V2 samples found")
        return None

    # Read first sample to get params
    with open(sample_dirs[0] / "metadata.json") as f:
        meta0 = json.load(f)
    n_layers = len(meta0.get("layers", [1, 10, 19, 28, 37]))
    hidden_dim = meta0.get("hidden_dim", 2048)
    layers = meta0.get("layers", [1, 10, 19, 28, 37])
    dtype_str = meta0.get("dtype", "float16")
    bytes_per_element = 2 if dtype_str == "float16" else 4

    print(f"pack_shards_v2: {len(sample_dirs)} V2 sample directories found")
    print(f"  n_layers={n_layers}, hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")
    print(f"  samples_per_shard={samples_per_shard}, dtype={dtype_str}")

    shard_idx = 0
    shard_samples = []
    shard_bin = None
    shard_offset = 0
    total_written = 0
    total_skipped = 0

    def _flush_shard():
        nonlocal shard_bin, shard_samples, shard_idx, shard_offset
        if shard_bin is not None:
            shard_bin.close()
            idx_path = output_dir / f"shard_{shard_idx:04d}.idx"
            with open(idx_path, "w") as f:
                json.dump({
                    "n_layers": n_layers,
                    "layers": layers,
                    "hidden_dim": hidden_dim,
                    "dtype": dtype_str,
                    "samples": shard_samples,
                }, f)
            shard_idx += 1
            shard_samples = []
            shard_offset = 0

    for si, sdir in enumerate(sample_dirs):
        if (si + 1) % 1000 == 0:
            print(f"  [{si+1}/{len(sample_dirs)}] written={total_written}, "
                  f"skipped={total_skipped}", flush=True)

        with open(sdir / "metadata.json") as f:
            meta = json.load(f)
        n_pos = meta.get("n_positions", meta.get("seq_len", 0))
        if n_pos < 18:
            total_skipped += 1
            continue

        clip_len = min(n_pos, max_seq_len)

        # Read context_hidden.bin — already [n_layers, n_positions, hidden_dim] as float16
        hidden_path = sdir / "context_hidden.bin"
        expected_size = n_layers * n_pos * hidden_dim * bytes_per_element
        if hidden_path.stat().st_size != expected_size:
            total_skipped += 1
            continue

        hidden_raw = np.fromfile(hidden_path, dtype=np.float16)
        hidden = hidden_raw.reshape(n_layers, n_pos, hidden_dim)[:, :clip_len, :]

        if np.any(np.isnan(hidden)) or np.any(np.isinf(hidden)):
            total_skipped += 1
            continue

        # Read tokens.bin
        tokens = np.fromfile(sdir / "tokens.bin", dtype=np.int32)[:clip_len]

        # Open new shard if needed
        if shard_bin is None or len(shard_samples) >= samples_per_shard:
            _flush_shard()
            bin_path = output_dir / f"shard_{shard_idx:04d}.bin"
            shard_bin = open(bin_path, "wb")
            shard_offset = 0

        # Write hidden states: [n_layers, clip_len, hidden_dim]
        hidden_bytes = n_layers * clip_len * hidden_dim * bytes_per_element
        shard_bin.write(hidden[:, :clip_len, :].tobytes())

        # Write tokens: int32[clip_len]
        token_bytes = clip_len * 4
        shard_bin.write(tokens.tobytes())

        shard_samples.append({
            "offset": shard_offset,
            "seq_len": clip_len,
            "original_seq_len": n_pos,
            "hidden_bytes": hidden_bytes,
            "token_bytes": token_bytes,
        })
        shard_offset += hidden_bytes + token_bytes
        total_written += 1

    _flush_shard()

    manifest = {
        "n_samples": total_written,
        "n_shards": shard_idx,
        "n_layers": n_layers,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "dtype": dtype_str,
        "max_seq_len": max_seq_len,
        "samples_per_shard": samples_per_shard,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total_bytes = sum(
        f.stat().st_size for f in output_dir.glob("shard_*.bin")
    )
    print(f"\npack_shards_v2 complete:")
    print(f"  {total_written} samples in {shard_idx} shards")
    print(f"  {total_skipped} skipped")
    print(f"  Total size: {total_bytes / 1e9:.2f} GB")
    print(f"  Per sample: {total_bytes / max(1, total_written) / 1e6:.1f} MB")

    return manifest


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class DFlashFullSeqDataset(Dataset):
    """Full-sequence multi-anchor dataset for DFlash v8.

    Loads packed shards via memory-mapping. Each __getitem__ returns
    multiple anchor samples from a single sequence, amortizing I/O.

    The collate function flattens these into a standard batch.
    """

    def __init__(
        self,
        shard_dir: str,
        block_size: int = 16,
        max_ctx_len: int = 1024,
        anchors_per_seq: int = 8,
        num_layers: int = 5,
        hidden_dim: int = 2048,
    ):
        """
        Args:
            shard_dir: directory containing shard_XXXX.{bin,idx} and manifest.json
            block_size: tokens per draft block (K)
            max_ctx_len: maximum context positions fed to the drafter
            anchors_per_seq: how many random anchors to sample per sequence
            num_layers: number of target feature layers
            hidden_dim: hidden dimension per layer
        """
        self.shard_dir = Path(shard_dir)
        self.block_size = block_size
        self.max_ctx_len = max_ctx_len
        self.anchors_per_seq = anchors_per_seq
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Load manifest
        manifest_path = self.shard_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.num_layers = manifest.get("n_layers", num_layers)
            self.hidden_dim = manifest.get("hidden_dim", hidden_dim)
            self._dtype_str = manifest.get("dtype", "float16")
        else:
            self._dtype_str = "float16"

        self._np_dtype = np.float16 if self._dtype_str == "float16" else np.float32
        self._bytes_per_elem = 2 if self._dtype_str == "float16" else 4

        # Load shard indices and build global sample list
        self._shard_mmaps: Dict[int, mmap.mmap] = {}
        self._shard_files: Dict[int, object] = {}
        self._samples: List[Tuple[int, dict]] = []  # (shard_id, sample_entry)

        shard_idx_files = sorted(self.shard_dir.glob("shard_*.idx"))
        for idx_path in shard_idx_files:
            shard_id = int(idx_path.stem.split("_")[1])
            with open(idx_path) as f:
                idx_data = json.load(f)
            for sample_entry in idx_data["samples"]:
                self._samples.append((shard_id, sample_entry))

        # Filter out sequences too short for even one block
        min_len = block_size + 2  # need at least 1 context + anchor + block_size-1
        before = len(self._samples)
        self._samples = [
            (sid, s) for sid, s in self._samples
            if s["seq_len"] >= min_len
        ]
        n_filtered = before - len(self._samples)

        print(f"DFlashFullSeqDataset: {len(self._samples)} sequences from "
              f"{len(shard_idx_files)} shards")
        if n_filtered > 0:
            print(f"  Filtered {n_filtered} sequences shorter than {min_len}")
        print(f"  block_size={block_size}, max_ctx_len={max_ctx_len}, "
              f"anchors_per_seq={anchors_per_seq}")
        print(f"  Effective items/epoch: {len(self._samples)} "
              f"(x{anchors_per_seq} anchors = "
              f"{len(self._samples) * anchors_per_seq} training examples)")

    def _get_mmap(self, shard_id: int) -> mmap.mmap:
        """Lazily open and mmap shard files."""
        if shard_id not in self._shard_mmaps:
            bin_path = self.shard_dir / f"shard_{shard_id:04d}.bin"
            f = open(bin_path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self._shard_files[shard_id] = f
            self._shard_mmaps[shard_id] = mm
        return self._shard_mmaps[shard_id]

    def __len__(self):
        return len(self._samples)

    def _load_sample(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load hidden states and tokens for a sample via mmap.

        Returns:
            hidden: float16/32 array [n_layers, seq_len, hidden_dim]
            tokens: int32 array [seq_len]
        """
        shard_id, entry = self._samples[idx]
        mm = self._get_mmap(shard_id)

        offset = entry["offset"]
        seq_len = entry["seq_len"]
        hidden_bytes = entry["hidden_bytes"]
        token_bytes = entry["token_bytes"]

        # Read hidden states
        hidden_raw = mm[offset:offset + hidden_bytes]
        hidden = np.frombuffer(hidden_raw, dtype=self._np_dtype).reshape(
            self.num_layers, seq_len, self.hidden_dim
        )

        # Read tokens
        token_raw = mm[offset + hidden_bytes:offset + hidden_bytes + token_bytes]
        tokens = np.frombuffer(token_raw, dtype=np.int32).copy()

        return hidden, tokens

    def __getitem__(self, idx):
        """Load a full sequence and sample multiple anchor positions.

        Returns a dict with lists of length `anchors_per_seq`:
            block_input_ids:      list of [K] int64 tensors
            context_hidden_list:  list of (list of n_layers [ctx_len, H] tensors)
            context_lengths:      list of int
            anchor_positions:     list of int
        """
        hidden, tokens = self._load_sample(idx)
        seq_len = len(tokens)

        K = self.block_size
        # Valid anchor range: [1, seq_len - K]
        # anchor_pos is the first token of the block; context = [0..anchor_pos)
        min_anchor = 1
        max_anchor = seq_len - K
        if max_anchor < min_anchor:
            max_anchor = min_anchor

        # Sample anchor positions (without replacement if possible)
        n_possible = max_anchor - min_anchor + 1
        n_anchors = min(self.anchors_per_seq, n_possible)

        if n_possible <= self.anchors_per_seq:
            anchors = list(range(min_anchor, max_anchor + 1))
        else:
            anchors = (np.random.choice(n_possible, size=n_anchors, replace=False) + min_anchor).tolist()

        # Build per-anchor outputs
        result_blocks = []
        result_ctx_hidden = []
        result_ctx_lengths = []
        result_anchor_pos = []

        for anchor_pos in anchors:
            block_end = min(anchor_pos + K, seq_len)
            block_tokens = tokens[anchor_pos:block_end]
            if len(block_tokens) < K:
                block_tokens = np.pad(block_tokens, (0, K - len(block_tokens)))

            ctx_end = anchor_pos + 1
            ctx_start = max(0, ctx_end - self.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            ctx_hidden_list = []
            for li in range(self.num_layers):
                h = hidden[li, ctx_start:ctx_end, :]
                ctx_hidden_list.append(
                    torch.from_numpy(h.astype(np.float32))
                )

            result_blocks.append(
                torch.from_numpy(block_tokens.astype(np.int64))
            )
            result_ctx_hidden.append(ctx_hidden_list)
            result_ctx_lengths.append(ctx_len)
            result_anchor_pos.append(anchor_pos)

        return {
            "block_input_ids": result_blocks,
            "context_hidden_list": result_ctx_hidden,
            "context_lengths": result_ctx_lengths,
            "anchor_positions": result_anchor_pos,
            "n_anchors": n_anchors,
        }

    def close(self):
        """Release mmap resources."""
        for mm in self._shard_mmaps.values():
            mm.close()
        for f in self._shard_files.values():
            f.close()
        self._shard_mmaps.clear()
        self._shard_files.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Collate function — flattens multi-anchor samples into a flat batch
# ---------------------------------------------------------------------------

def collate_v8(batch):
    """Collate multi-anchor samples into a flat padded batch.

    Input: list of dicts from DFlashFullSeqDataset.__getitem__,
           each containing lists of length anchors_per_seq.

    Output: flat batch with B = sum(n_anchors across batch items).
        block_input_ids:     [B, K]
        context_hidden_list: list of n_layers tensors [B, max_ctx_len, H]
        context_lengths:     [B] int64
        anchor_positions:    [B] int64
    """
    # Flatten all anchors across all sequences in the batch
    all_blocks = []
    all_ctx_hidden = []  # list of (list of n_layers tensors)
    all_ctx_lengths = []
    all_anchor_pos = []

    for item in batch:
        n = item["n_anchors"]
        for i in range(n):
            all_blocks.append(item["block_input_ids"][i])
            all_ctx_hidden.append(item["context_hidden_list"][i])
            all_ctx_lengths.append(item["context_lengths"][i])
            all_anchor_pos.append(item["anchor_positions"][i])

    B = len(all_blocks)
    if B == 0:
        raise ValueError("Empty batch after flattening")

    n_layers = len(all_ctx_hidden[0])
    H = all_ctx_hidden[0][0].shape[-1]

    block_input_ids = torch.stack(all_blocks)  # [B, K]
    max_ctx_len = max(all_ctx_lengths)
    context_lengths = torch.tensor(all_ctx_lengths, dtype=torch.long)
    anchor_positions = torch.tensor(all_anchor_pos, dtype=torch.long)

    # Pad context hidden states (left-padding for consistency with v7)
    context_hidden_list = []
    for layer_idx in range(n_layers):
        padded = torch.zeros(B, max_ctx_len, H)
        for bi in range(B):
            ctx = all_ctx_hidden[bi][layer_idx]  # [ctx_len_i, H]
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


# ---------------------------------------------------------------------------
# Fallback: per-directory full-sequence dataset (no packing required)
# ---------------------------------------------------------------------------

class DFlashFullSeqDirDataset(Dataset):
    """Full-sequence multi-anchor dataset reading from per-sample directories.

    Supports TWO extraction formats:
      V2 (--extract-all): context_hidden.bin [n_layers, seq_len, H] + tokens.bin [seq_len]
      V1 (legacy):        layer_XX.bin [seq_len, H] per layer + input_ids.bin/tokens.bin

    Hidden states are loaded from disk each time (no mmap), with optional
    LRU caching.
    """

    def __init__(
        self,
        features_dir: str,
        block_size: int = 16,
        max_ctx_len: int = 1024,
        anchors_per_seq: int = 8,
        target_layers: Tuple[int, ...] = (1, 10, 19, 28, 37),
        hidden_dim: int = 2048,
        cache_size: int = 200,
        preload: bool = False,
    ):
        self.features_dir = Path(features_dir)
        self.block_size = block_size
        self.max_ctx_len = max_ctx_len
        self.anchors_per_seq = anchors_per_seq
        self.target_layers = list(target_layers)
        self.num_layers = len(target_layers)
        self.hidden_dim = hidden_dim
        self._preloaded = preload

        from collections import OrderedDict
        self._cache = OrderedDict()
        self._cache_size = cache_size

        # Discover valid samples — detect format per sample
        self.sample_dirs = []
        self.sample_format = []  # "v2" or "v1"
        self.metas = []
        n_skipped = 0

        all_dirs = sorted([d for d in self.features_dir.iterdir() if d.is_dir()])
        for d in all_dirs:
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            seq_len = meta.get("seq_len", meta.get("n_tokens", 0))
            if seq_len < block_size + 2:
                n_skipped += 1
                continue

            # V2 format: context_hidden.bin + tokens.bin (from --extract-all)
            ctx_hidden_path = d / "context_hidden.bin"
            tokens_path = d / "tokens.bin"
            if ctx_hidden_path.exists() and tokens_path.exists():
                h_dim = meta.get("hidden_dim", hidden_dim)
                n_lay = len(meta.get("layers", target_layers))
                file_size = ctx_hidden_path.stat().st_size
                bytes_per_pos = n_lay * h_dim * 2
                if file_size % bytes_per_pos != 0 or file_size == 0:
                    n_skipped += 1
                    continue
                # Derive n_positions from file size (metadata may be stale)
                real_n_pos = file_size // bytes_per_pos
                if real_n_pos < block_size + 2:
                    n_skipped += 1
                    continue
                meta["n_positions"] = real_n_pos
                self.sample_dirs.append(d)
                self.sample_format.append("v2")
                self.metas.append(meta)
                continue

            # V1 format: layer_XX.bin per layer
            has_all = all(
                (d / f"layer_{l:02d}.bin").exists() for l in self.target_layers
            )
            has_tokens = (d / "input_ids.bin").exists() or (d / "tokens.bin").exists()
            if has_all and has_tokens:
                self.sample_dirs.append(d)
                self.sample_format.append("v1")
                self.metas.append(meta)
                continue

            n_skipped += 1

        n_v2 = sum(1 for f in self.sample_format if f == "v2")
        n_v1 = len(self.sample_dirs) - n_v2

        print(f"DFlashFullSeqDirDataset: {len(self.sample_dirs)} sequences "
              f"({n_v2} v2, {n_v1} v1), {n_skipped} skipped")
        print(f"  block_size={block_size}, max_ctx_len={max_ctx_len}, "
              f"anchors_per_seq={anchors_per_seq}")
        print(f"  Effective items/epoch: {len(self.sample_dirs) * anchors_per_seq}")

        # Preload all samples into RAM (eliminates I/O during training)
        if preload and len(self.sample_dirs) > 0:
            print(f"  Preloading {len(self.sample_dirs)} samples into RAM...", flush=True)
            t0 = time.time()
            self._preloaded_data = []
            for i in range(len(self.sample_dirs)):
                self._preloaded_data.append(self._load_sample_from_disk(i))
                if (i + 1) % 2000 == 0:
                    print(f"    [{i+1}/{len(self.sample_dirs)}]", flush=True)
            elapsed = time.time() - t0
            total_mb = sum(
                s["hidden"].nbytes + s["tokens"].nbytes
                for s in self._preloaded_data
            ) / 1e6
            print(f"  Preloaded in {elapsed:.1f}s ({total_mb:.0f} MB in RAM)")
            self._cache_size = 0  # disable LRU cache
        else:
            self._preloaded_data = None

    def __len__(self):
        return len(self.sample_dirs)

    def _load_sample_from_disk(self, idx):
        """Load sample from disk (no caching)."""
        d = self.sample_dirs[idx]
        fmt = self.sample_format[idx]
        meta = self.metas[idx]
        seq_len = meta.get("seq_len", meta.get("n_tokens", 0))
        h_dim = meta.get("hidden_dim", self.hidden_dim)

        if fmt == "v2":
            n_lay = len(meta.get("layers", self.target_layers))
            hidden = np.fromfile(d / "context_hidden.bin", dtype=np.float16)
            n_pos = hidden.size // (n_lay * h_dim)
            hidden = hidden.reshape(n_lay, n_pos, h_dim)
            tokens = np.fromfile(d / "tokens.bin", dtype=np.int32)
            actual_len = min(len(tokens), n_pos)
            tokens = tokens[:actual_len]
            hidden = hidden[:, :actual_len, :]
        else:
            tokens_path = d / "input_ids.bin"
            if not tokens_path.exists():
                tokens_path = d / "tokens.bin"
            tokens = np.fromfile(tokens_path, dtype=np.int32)[:seq_len]
            layer_hidden = []
            for l in self.target_layers:
                lpath = d / f"layer_{l:02d}.bin"
                file_size = lpath.stat().st_size
                if file_size == seq_len * h_dim * 2:
                    h = np.fromfile(lpath, dtype=np.float16).reshape(seq_len, h_dim)
                else:
                    h = np.fromfile(lpath, dtype=np.float32).reshape(seq_len, h_dim)
                layer_hidden.append(h)
            hidden = np.stack(layer_hidden, axis=0)
            actual_len = seq_len

        return {"hidden": hidden, "tokens": tokens, "seq_len": actual_len}

    def _load_sample(self, idx):
        if self._preloaded_data is not None:
            return self._preloaded_data[idx]

        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        sample = self._load_sample_from_disk(idx)

        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[idx] = sample
        return sample

    def __getitem__(self, idx):
        """Same output format as DFlashFullSeqDataset."""
        sample = self._load_sample(idx)
        hidden = sample["hidden"]
        tokens = sample["tokens"]
        seq_len = sample["seq_len"]

        K = self.block_size
        min_anchor = 1
        max_anchor = seq_len - K
        if max_anchor < min_anchor:
            max_anchor = min_anchor

        n_possible = max_anchor - min_anchor + 1
        n_anchors = min(self.anchors_per_seq, n_possible)

        if n_possible <= self.anchors_per_seq:
            anchors = list(range(min_anchor, max_anchor + 1))
        else:
            anchors = (np.random.choice(n_possible, size=n_anchors, replace=False) + min_anchor).tolist()

        result_blocks = []
        result_ctx_hidden = []
        result_ctx_lengths = []
        result_anchor_pos = []

        for anchor_pos in anchors:
            block_end = min(anchor_pos + K, seq_len)
            block_tokens = tokens[anchor_pos:block_end]
            if len(block_tokens) < K:
                block_tokens = np.pad(block_tokens, (0, K - len(block_tokens)))

            ctx_end = anchor_pos + 1
            ctx_start = max(0, ctx_end - self.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            ctx_hidden_list = []
            for li in range(self.num_layers):
                h = hidden[li, ctx_start:ctx_end, :]
                ctx_hidden_list.append(
                    torch.from_numpy(h.astype(np.float32))
                )

            result_blocks.append(
                torch.from_numpy(block_tokens.astype(np.int64))
            )
            result_ctx_hidden.append(ctx_hidden_list)
            result_ctx_lengths.append(ctx_len)
            result_anchor_pos.append(anchor_pos)

        return {
            "block_input_ids": result_blocks,
            "context_hidden_list": result_ctx_hidden,
            "context_lengths": result_ctx_lengths,
            "anchor_positions": result_anchor_pos,
            "n_anchors": n_anchors,
        }


# ---------------------------------------------------------------------------
# CLI: pack shards from command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pack DFlash v8 shards")
    parser.add_argument("--input", type=str, required=True,
                        help="Per-sample features directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for packed shards")
    parser.add_argument("--format", type=str, choices=["v1", "v2", "auto"], default="auto",
                        help="Input format: v1 (layer_XX.bin), v2 (context_hidden.bin), auto-detect")
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--samples-per-shard", type=int, default=200)
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    # Auto-detect format
    fmt = args.format
    if fmt == "auto":
        input_path = Path(args.input)
        sample_dirs = [d for d in input_path.iterdir() if d.is_dir()]
        if sample_dirs and (sample_dirs[0] / "context_hidden.bin").exists():
            fmt = "v2"
        else:
            fmt = "v1"
        print(f"Auto-detected format: {fmt}")

    if fmt == "v2":
        pack_shards_v2(
            features_dir=args.input,
            output_dir=args.output,
            max_seq_len=args.max_seq_len,
            samples_per_shard=args.samples_per_shard,
        )
    else:
        pack_shards_v8(
            features_dir=args.input,
            output_dir=args.output,
            layers=layers,
            hidden_dim=args.hidden_dim,
            max_seq_len=args.max_seq_len,
            samples_per_shard=args.samples_per_shard,
        )
