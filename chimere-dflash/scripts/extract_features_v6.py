#!/usr/bin/env python3
"""
extract_features_v6.py — Extract hidden state features for DFlash v6 training.

Compact extraction: for each sample, stores only the context window
(max_ctx_len positions) before randomly sampled block anchors.

This is much more storage-efficient than storing full sequences:
  - Full seq (400 tok × 5 layers × 2048 × 4B) = ~16 MB/sample
  - Compact (64 tok × 5 layers × 2048 × 2B) = ~1.3 MB/sample
  - Single pos (1 tok × 5 layers × 2048 × 2B) = ~20 KB/sample

Uses the C++ target_daemon binary for fast inference via binary protocol.

Usage:
  python scripts/extract_features_v6.py \
    --input data/prompts_v6/ready_for_extraction.jsonl \
    --output data/features_iq3 \
    --model ~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ3_S.gguf \
    --max-samples 100000 \
    --max-ctx-len 64
"""

import argparse
import json
import os
import sys
import time
import struct
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v6 import DFlashV6Config, build_target_layer_ids

# Default config
DEFAULT_LAYERS = [1, 10, 19, 28, 37]  # v6 target layers for 40-layer Qwen3.5
DEFAULT_MAX_CTX = 64  # context positions to store
DEFAULT_BLOCK_SIZE = 16
HIDDEN_DIM = 2048


def extract_with_daemon(daemon_path, model_path, input_file, output_dir,
                        layers, max_samples, max_ctx_len, max_seq_len,
                        blocks_per_seq, block_size, n_gpu_layers,
                        extra_args):
    """Extract features using the target_daemon binary protocol."""
    from chimere.target_daemon import TargetDaemon

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"Loading prompts from {input_file}...", flush=True)
    prompts = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    print(f"  {len(prompts)} prompts loaded", flush=True)

    if max_samples > 0 and len(prompts) > max_samples:
        prompts = prompts[:max_samples]
        print(f"  Truncated to {max_samples}", flush=True)

    # Start daemon
    print(f"Starting target daemon...", flush=True)
    print(f"  Model: {model_path}", flush=True)
    print(f"  Layers: {layers}", flush=True)
    print(f"  Max ctx: {max_ctx_len}", flush=True)

    daemon = TargetDaemon(
        daemon_path=daemon_path,
        model_path=model_path,
        layers=layers,
        n_gpu_layers=n_gpu_layers,
        context_size=max_seq_len,
        extra_args=extra_args,
    )
    daemon.start()

    try:
        _extract_samples(daemon, prompts, output_dir, layers,
                         max_ctx_len, blocks_per_seq, block_size)
    finally:
        daemon.stop()


def extract_with_cpp_binary(extractor_path, model_path, input_file, output_dir,
                             layers, max_samples, max_seq_len, n_gpu_layers,
                             extra_args):
    """Extract features using the C++ extract_hidden_states binary.

    Runs full extraction then post-processes to compact format.
    """
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    layers_str = ",".join(str(l) for l in layers)

    cmd = [
        extractor_path,
        "-m", model_path,
        "--input", input_file,
        "--output", str(raw_dir),
        "--layers", layers_str,
        "-ngl", str(n_gpu_layers),
        "-c", str(max_seq_len),
    ]
    if max_samples > 0:
        cmd += ["--max-samples", str(max_samples)]
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    return raw_dir


def compact_features(raw_dir, output_dir, layers, max_ctx_len,
                     blocks_per_seq, block_size):
    """Post-process raw features to compact format.

    For each sample, randomly sample block anchors and store only
    the context window (max_ctx_len positions) before each anchor.
    """
    import random
    random.seed(42)

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(raw_dir.glob("sample_*"))
    print(f"\nCompacting {len(sample_dirs)} samples...", flush=True)

    n_written = 0
    n_skipped = 0

    for si, sdir in enumerate(sample_dirs):
        if (si + 1) % 100 == 0:
            print(f"  [{si+1}/{len(sample_dirs)}] written={n_written}", flush=True)

        # Load metadata
        meta_path = sdir / "metadata.json"
        if not meta_path.exists():
            n_skipped += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        seq_len = meta["seq_len"]
        hidden_dim = meta["hidden_dim"]

        if seq_len < block_size + 2:
            n_skipped += 1
            continue

        # Load tokens
        tokens = np.fromfile(sdir / "input_ids.bin", dtype=np.int32)

        # Load hidden states per layer
        layer_hidden = {}
        valid = True
        for l in layers:
            lpath = sdir / f"layer_{l:02d}.bin"
            if not lpath.exists():
                valid = False
                break
            h = np.fromfile(lpath, dtype=np.float32).reshape(seq_len, hidden_dim)
            if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                valid = False
                break
            layer_hidden[l] = h

        if not valid:
            n_skipped += 1
            continue

        # Create output sample directory
        out_sdir = output_dir / f"sample_{n_written:06d}"
        out_sdir.mkdir(exist_ok=True)

        # Save tokens (full sequence — training will sample blocks from it)
        tokens.astype(np.int32).tofile(out_sdir / "input_ids.bin")

        # Save context-clipped hidden states per layer
        # Store up to max_ctx_len positions from each layer
        # The data_v5 loader will sample random blocks at training time
        store_len = min(seq_len, max_ctx_len + block_size)
        # We store the LAST store_len positions (most useful for block prediction)
        # Actually, store the full sequence but clipped to max length
        # The training code samples random blocks within the sequence
        clip_len = min(seq_len, max_ctx_len * 4)  # some room for multiple blocks

        for l in layers:
            h = layer_hidden[l][:clip_len]
            # Save as float16 to halve storage
            h.astype(np.float16).tofile(out_sdir / f"layer_{l:02d}.bin")

        # Save metadata
        out_meta = {
            "seq_len": clip_len,
            "original_seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "dtype": "float16",
            "prompt": meta.get("prompt", "")[:200],
        }
        with open(out_sdir / "metadata.json", "w") as f:
            json.dump(out_meta, f)

        n_written += 1

    print(f"  Done: {n_written} samples written, {n_skipped} skipped", flush=True)
    return n_written


def _extract_samples(daemon, prompts, output_dir, layers,
                     max_ctx_len, blocks_per_seq, block_size):
    """Extract features for each prompt using daemon."""
    n_written = 0
    n_failed = 0
    t0 = time.time()

    for i, prompt_data in enumerate(prompts):
        prompt = prompt_data["prompt"]
        response = prompt_data.get("response", "")

        # Build full text
        full_text = prompt
        if response:
            full_text = prompt + "\n" + response

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            print(f"  [{i+1}/{len(prompts)}] "
                  f"{rate:.1f} samples/s, "
                  f"ETA {eta/60:.0f}m, "
                  f"written={n_written}, failed={n_failed}", flush=True)

        try:
            # Tokenize
            tokens = daemon.tokenize(full_text)
            seq_len = len(tokens)

            if seq_len < block_size + 2:
                n_failed += 1
                continue

            # Forward pass — get hidden states for all layers
            hidden_states, logits = daemon.eval_full(tokens)
            # hidden_states: [n_layers, seq_len, hidden_dim]

            # Clip to reasonable length
            clip_len = min(seq_len, max_ctx_len * 4)

            # Save
            out_sdir = output_dir / f"sample_{n_written:06d}"
            out_sdir.mkdir(exist_ok=True)

            # Tokens
            np.array(tokens[:clip_len], dtype=np.int32).tofile(
                out_sdir / "input_ids.bin"
            )

            # Hidden states per layer (float16)
            for li, l in enumerate(layers):
                h = hidden_states[li, :clip_len, :]
                h.astype(np.float16).tofile(out_sdir / f"layer_{l:02d}.bin")

            # Metadata
            meta = {
                "seq_len": clip_len,
                "original_seq_len": seq_len,
                "hidden_dim": HIDDEN_DIM,
                "layers": layers,
                "dtype": "float16",
                "prompt": prompt[:200],
            }
            with open(out_sdir / "metadata.json", "w") as f:
                json.dump(meta, f)

            n_written += 1

        except Exception as e:
            print(f"  FAILED sample {i}: {e}", flush=True)
            n_failed += 1

    elapsed = time.time() - t0
    print(f"\nExtraction complete: {n_written} samples in {elapsed:.1f}s "
          f"({n_written/elapsed:.1f} samples/s), {n_failed} failed", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract DFlash v6 features")
    parser.add_argument("--input", type=str, required=True,
                        help="JSONL input file (prompt + response)")
    parser.add_argument("--output", type=str, default="data/features_iq3",
                        help="Output directory for features")
    parser.add_argument("--model", type=str,
                        default=os.path.expanduser(
                            "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/"
                            "Qwen3.5-35B-A3B-UD-IQ3_S.gguf"))
    parser.add_argument("--extractor", type=str,
                        default=os.path.expanduser(
                            "~/llama.cpp/build/bin/llama-extract-hidden-states"))
    parser.add_argument("--daemon", type=str,
                        default=os.path.expanduser(
                            "~/chimere-dflash/extract/build/target_daemon"))
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-ctx-len", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--blocks-per-seq", type=int, default=20)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--mode", choices=["cpp", "daemon", "compact"],
                        default="cpp",
                        help="cpp: use extract_hidden_states binary, "
                             "daemon: use target_daemon, "
                             "compact: post-process existing raw features")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    print("=" * 60)
    print(" DFlash v6 Feature Extraction")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Model:      {args.model}")
    print(f"  Input:      {args.input}")
    print(f"  Output:     {args.output}")
    print(f"  Layers:     {layers}")
    print(f"  Max ctx:    {args.max_ctx_len}")
    print(f"  Block size: {args.block_size}")
    print()

    if args.mode == "cpp":
        # Extract with C++ binary then compact
        extra_args = []
        if "IQ3" in args.model or "iq3" in args.model:
            # IQ3 can run fully on GPU
            pass
        elif "Q5" in args.model or "q5" in args.model:
            extra_args = ["-ot", ".ffn_.*_exps.=CPU"]

        raw_dir = extract_with_cpp_binary(
            args.extractor, args.model, args.input, args.output,
            layers, args.max_samples, args.max_seq_len, args.ngl,
            extra_args,
        )
        compact_features(
            raw_dir, args.output, layers,
            args.max_ctx_len, args.blocks_per_seq, args.block_size,
        )

    elif args.mode == "daemon":
        extra_args = ["--flash-attn", "on"]
        if "Q5" in args.model or "q5" in args.model:
            extra_args += ["-ot", ".ffn_.*_exps.=CPU"]

        extract_with_daemon(
            args.daemon, args.model, args.input, args.output,
            layers, args.max_samples, args.max_ctx_len,
            args.max_seq_len, args.blocks_per_seq, args.block_size,
            args.ngl, extra_args,
        )

    elif args.mode == "compact":
        # Just compact existing raw features
        compact_features(
            args.input, args.output, layers,
            args.max_ctx_len, args.blocks_per_seq, args.block_size,
        )

    # Print stats
    output_dir = Path(args.output)
    n_samples = len(list(output_dir.glob("sample_*")))
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())

    print(f"\n{'=' * 60}")
    print(f" Extraction Summary")
    print(f"{'=' * 60}")
    print(f"  Samples:    {n_samples}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"  Per sample: {total_size / max(1, n_samples) / 1e3:.1f} KB")
    print(f"  Output:     {args.output}")

    # Estimated for different dataset sizes
    per_sample_kb = total_size / max(1, n_samples) / 1024
    for n in [10000, 30000, 50000, 100000]:
        est_gb = n * per_sample_kb / 1024 / 1024
        print(f"  {n:>6d} samples → ~{est_gb:.1f} GB")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
