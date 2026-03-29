#!/usr/bin/env python3
"""
re_extract_with_daemon.py — Re-extract hidden states using the C++ target daemon.

Ensures training data uses the EXACT same extraction pipeline as online inference.
Reads prompts from existing features metadata, re-tokenizes and re-extracts.

Usage:
  python scripts/re_extract_with_daemon.py \
    --input-dir data/features \
    --output-dir data/features_daemon
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.target_daemon import TargetDaemon


def main():
    parser = argparse.ArgumentParser(description="Re-extract features via daemon")
    parser.add_argument("--input-dir", type=str, default="data/features",
                        help="Original features directory (for prompts)")
    parser.add_argument("--output-dir", type=str, default="data/features_daemon",
                        help="Output directory for re-extracted features")
    parser.add_argument("--daemon-path", type=str, default="extract/build/target_daemon")
    parser.add_argument("--model-path", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/"
                                    "Qwen3.5-35B-A3B-MXFP4_MOE.gguf"))
    parser.add_argument("--layers", type=str, default="2,11,20,29,37")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    # Discover samples
    sample_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if args.max_samples > 0:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"Found {len(sample_dirs)} samples")
    print(f"Layers: {layers}")
    print(f"Output: {output_dir}")
    print()

    extra_args = ["-ot", ".ffn_.*_exps.=CPU", "--flash-attn", "on"]
    with TargetDaemon(
        daemon_path=args.daemon_path,
        model_path=args.model_path,
        layers=layers,
        n_gpu_layers=99,
        extra_args=extra_args,
    ) as daemon:
        total_tokens = 0
        n_success = 0
        n_failed = 0

        for sample_dir in tqdm(sample_dirs, desc="Re-extracting"):
            # Read original metadata for the prompt text
            meta_path = sample_dir / "metadata.json"
            if not meta_path.exists():
                meta_path = sample_dir / "meta.json"
            if not meta_path.exists():
                n_failed += 1
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            prompt = meta.get("prompt", "")
            if not prompt:
                n_failed += 1
                continue

            # Tokenize via daemon (same tokenizer as online inference)
            daemon.clear_kv()
            tokens = daemon.tokenize(prompt)
            n_tokens = len(tokens)

            if n_tokens < 2:
                n_failed += 1
                continue

            # Extract hidden states via eval_full
            hidden, logits = daemon.eval_full(tokens)
            # hidden: [n_layers, seq_len, hidden_dim]

            hidden_dim = hidden.shape[2]

            # Save in same format as original
            out_dir = output_dir / sample_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Save tokens
            np.array(tokens, dtype=np.int32).tofile(out_dir / "input_ids.bin")

            # Save hidden states per layer
            for i, l in enumerate(layers):
                hidden[i].astype(np.float32).tofile(out_dir / f"layer_{l:02d}.bin")

            # Save metadata
            new_meta = {
                "seq_len": n_tokens,
                "hidden_dim": hidden_dim,
                "layers": layers,
                "prompt": prompt,
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(new_meta, f, indent=2)

            total_tokens += n_tokens
            n_success += 1

    print(f"\nDone!")
    print(f"  Samples: {n_success}/{len(sample_dirs)} ({n_failed} failed)")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
