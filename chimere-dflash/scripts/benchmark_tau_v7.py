#!/usr/bin/env python3
"""
benchmark_tau_v7.py — Measure τ (acceptance rate) for DFlash v7 drafter.

Same as v6 benchmark but uses v7 model (RoPE + absolute positions).

Usage:
  python scripts/benchmark_tau_v7.py --checkpoint checkpoints_v7_98k/best.pt --n-samples 1000
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.modeling_v7 import DFlashDraftModelV7


def main():
    parser = argparse.ArgumentParser(description="Benchmark DFlash v7 τ")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--features-dir", type=str, default="data/features_q5")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0=greedy, >0=sampling")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    features_dir = Path(args.features_dir)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "config" in ckpt:
        from dataclasses import fields
        config = DFlashV7Config(**{
            f.name: ckpt["config"][f.name]
            for f in fields(DFlashV7Config)
            if f.name in ckpt["config"]
        })
    else:
        config = DFlashV7Config()

    # Detect architecture: eagle or dflash
    is_eagle = ckpt.get("architecture") == "eagle"

    if is_eagle:
        from chimere.modeling_eagle import EagleDrafter
        model = EagleDrafter(config).to(device)
    else:
        model = DFlashDraftModelV7(config).to(device)

    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Keep everything float32 for consistent inference
    model = model.float()

    # Check if multi-ctx
    sample_check = sorted([d for d in features_dir.iterdir() if d.is_dir()])[:1]
    has_multi_ctx = any((d / "context_hidden.bin").exists() for d in sample_check)
    fmt_str = "multi-ctx" if has_multi_ctx else "single-pos"
    print(f"  block_size={config.block_size}, layers={config.num_hidden_layers}, "
          f"features={config.num_feature_layers}, rope_theta={config.rope_theta:.0f}, "
          f"format={fmt_str}")

    # Discover samples (support both multi-ctx and single-pos)
    sample_dirs = sorted([
        d for d in features_dir.iterdir()
        if d.is_dir() and (
            (d / "context_hidden.bin").exists() or
            (d / "anchor_hidden.bin").exists()
        )
    ])
    n_total = len(sample_dirs)
    n_eval = min(args.n_samples, n_total)

    indices = np.linspace(0, n_total - 1, n_eval, dtype=int)
    print(f"  Evaluating {n_eval}/{n_total} samples\n")

    # Benchmark
    total_drafted = 0
    total_accepted = 0
    total_blocks = 0
    per_block_accepted = []
    draft_times = []
    mask_token_id = config.mask_token_id

    t0 = time.time()
    with torch.no_grad():
        for eval_idx, sample_idx in enumerate(indices):
            d = sample_dirs[sample_idx]

            # Load hidden states (multi-ctx or single-pos)
            ctx_file = d / "context_hidden.bin"
            anchor_file = d / "anchor_hidden.bin"

            # Load metadata
            anchor_pos = 0
            n_positions = 1
            meta_path = d / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                anchor_pos = meta.get("anchor_pos", 0)
                n_positions = meta.get("n_positions", 1)

            if ctx_file.exists():
                # Multi-context: float16[n_layers, n_positions, hidden_dim]
                raw = np.fromfile(ctx_file, dtype=np.float16)
                raw = raw.reshape(config.num_feature_layers, n_positions, config.target_hidden_size)
                raw = raw.astype(np.float32)
                ctx_len = n_positions
            else:
                # Single-pos fallback: float16[n_layers, hidden_dim] → [n_layers, 1, hidden_dim]
                raw = np.fromfile(anchor_file, dtype=np.float16)
                raw = raw.reshape(config.num_feature_layers, 1, config.target_hidden_size)
                raw = raw.astype(np.float32)
                ctx_len = 1

            # Load ground truth block tokens
            gt_tokens = np.fromfile(d / "block_tokens.bin", dtype=np.int32)

            n_real = np.sum(gt_tokens != mask_token_id)
            if n_real < 4:
                continue

            # Build input: list of [1, ctx_len, H] tensors
            hidden_list = [
                torch.from_numpy(raw[i]).unsqueeze(0).to(device)  # [1, ctx_len, H]
                for i in range(config.num_feature_layers)
            ]
            ctx_lengths = torch.tensor([ctx_len], device=device, dtype=torch.long)
            anchor_positions = torch.tensor([anchor_pos], device=device, dtype=torch.long)

            anchor_token_id = int(gt_tokens[0])

            # Draft with absolute positions
            t_draft = time.time()
            draft_ids, draft_logits = model.generate_block(
                hidden_list,
                context_lengths=ctx_lengths,
                temperature=args.temperature,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            draft_time = (time.time() - t_draft) * 1000
            draft_times.append(draft_time)

            draft = draft_ids[0].cpu().numpy()  # [K-1] = [15]

            # Count sequential accepted tokens
            # draft[j] predicts token at position j+1 (since draft = logits[:,1:])
            n_accepted = 0
            for j in range(config.block_size - 1):  # j = 0..14
                target_pos = j + 1
                if target_pos >= len(gt_tokens) or gt_tokens[target_pos] == mask_token_id:
                    break
                if draft[j] == gt_tokens[target_pos]:
                    n_accepted += 1
                else:
                    break

            total_drafted += min(n_real - 1, config.block_size - 1)
            total_accepted += n_accepted
            total_blocks += 1
            per_block_accepted.append(n_accepted)

            if (eval_idx + 1) % 100 == 0:
                running_rate = total_accepted / max(1, total_drafted)
                avg_draft_ms = np.mean(draft_times[-100:])
                print(f"  [{eval_idx+1}/{n_eval}] τ={running_rate:.2%} | "
                      f"avg_accepted={np.mean(per_block_accepted[-100:]):.1f}/15 | "
                      f"draft={avg_draft_ms:.1f}ms", flush=True)

    elapsed = time.time() - t0

    tau = total_accepted / max(1, total_drafted)
    avg_accepted = np.mean(per_block_accepted)
    avg_draft_ms = np.mean(draft_times)

    tokens_per_call = avg_accepted + 1
    target_ms_estimate = 25
    real_speedup = tokens_per_call * target_ms_estimate / (avg_draft_ms + target_ms_estimate)

    print(f"\n{'='*60}")
    print(f" DFlash v7 — Offline τ Benchmark (RoPE + Absolute Pos)")
    print(f"{'='*60}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Samples eval:     {total_blocks}")
    print(f"  Block size:       {config.block_size}")
    print(f"  Temperature:      {args.temperature}")
    print(f"  RoPE theta:       {config.rope_theta:.0f}")
    print(f"")
    print(f"  Total drafted:    {total_drafted}")
    print(f"  Total accepted:   {total_accepted}")
    print(f"  τ (accept rate):  {tau:.2%}")
    print(f"  Avg accepted:     {avg_accepted:.1f} / {config.block_size - 1}")
    print(f"  Avg draft time:   {avg_draft_ms:.1f}ms")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"")
    print(f"  Tokens/target_call:  {tokens_per_call:.1f}")
    print(f"  Theoretical max:     {config.block_size}x")
    print(f"  Est. real speedup:   {real_speedup:.2f}x")
    print(f"    (assuming target={target_ms_estimate}ms/token, "
          f"draft={avg_draft_ms:.0f}ms/block)")
    print(f"{'='*60}")

    if per_block_accepted:
        print(f"\n  Acceptance distribution (tokens accepted per block):")
        dist = collections.Counter(per_block_accepted)
        for k in sorted(dist.keys()):
            pct = dist[k] / total_blocks * 100
            bar = "#" * int(pct / 2)
            print(f"    {k:2d}: {dist[k]:4d} ({pct:5.1f}%) {bar}")

    print()


if __name__ == "__main__":
    main()
