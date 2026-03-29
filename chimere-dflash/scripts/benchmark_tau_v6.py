#!/usr/bin/env python3
"""
benchmark_tau_v6.py — Measure τ (acceptance rate) for DFlash v6 drafter.

Uses pre-extracted single-position hidden states + ground truth tokens.
Simulates greedy speculative decoding: draft 16 tokens, count sequential
matches against ground truth left-to-right.

Usage:
  python scripts/benchmark_tau_v6.py --checkpoint checkpoints_v6_75k/best.pt --n-samples 1000
"""

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v6 import DFlashV6Config
from chimere.modeling_v6 import DFlashDraftModelV6


def main():
    parser = argparse.ArgumentParser(description="Benchmark DFlash v6 τ")
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
        config = DFlashV6Config(**{
            f.name: ckpt["config"][f.name]
            for f in fields(DFlashV6Config)
            if f.name in ckpt["config"]
        })
    else:
        config = DFlashV6Config()

    # Build model
    model = DFlashDraftModelV6(config).to(device)
    # Strip torch.compile _orig_mod. prefix if present
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Cast frozen params to BF16
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    print(f"  block_size={config.block_size}, layers={config.num_hidden_layers}, "
          f"features={config.num_feature_layers}")

    # Discover samples
    sample_dirs = sorted([
        d for d in features_dir.iterdir()
        if d.is_dir() and (d / "anchor_hidden.bin").exists()
    ])
    n_total = len(sample_dirs)
    n_eval = min(args.n_samples, n_total)

    # Use evenly spaced samples across the dataset for representative eval
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

            # Load anchor hidden: float16[5, 2048] → float32
            anchor = np.fromfile(d / "anchor_hidden.bin", dtype=np.float16)
            anchor = anchor.reshape(config.num_feature_layers, config.target_hidden_size)
            anchor = anchor.astype(np.float32)

            # Load ground truth block tokens: int32[16]
            gt_tokens = np.fromfile(d / "block_tokens.bin", dtype=np.int32)

            # Skip if mostly padding
            n_real = np.sum(gt_tokens != mask_token_id)
            if n_real < 4:
                continue

            # Build input: list of 5 tensors [1, 1, H]
            hidden_list = [
                torch.from_numpy(anchor[i:i+1]).unsqueeze(0).to(device)  # [1, 1, H]
                for i in range(config.num_feature_layers)
            ]
            ctx_lengths = torch.tensor([1], device=device, dtype=torch.long)

            # Anchor token = gt_tokens[0] (the verified token at position 0)
            anchor_token_id = int(gt_tokens[0])

            # Draft
            t_draft = time.time()
            draft_ids, draft_logits = model.generate_block(
                hidden_list,
                context_lengths=ctx_lengths,
                temperature=args.temperature,
                anchor_token_id=anchor_token_id,
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

            total_drafted += min(n_real - 1, config.block_size - 1)  # exclude anchor
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

    # Results
    tau = total_accepted / max(1, total_drafted)
    avg_accepted = np.mean(per_block_accepted)
    avg_draft_ms = np.mean(draft_times)

    # Estimated speedup: tokens per target call
    # In spec decode: 1 target call verifies all drafted tokens
    # tokens_per_call = avg_accepted + 1 (for correction/bonus)
    tokens_per_call = avg_accepted + 1
    # Real speedup accounts for draft overhead
    target_ms_estimate = 25  # ~25ms per token at 42 tok/s gen speed
    real_speedup = tokens_per_call * target_ms_estimate / (avg_draft_ms + target_ms_estimate)

    print(f"\n{'='*60}")
    print(f" DFlash v6 — Offline τ Benchmark")
    print(f"{'='*60}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Samples eval:     {total_blocks}")
    print(f"  Block size:       {config.block_size}")
    print(f"  Temperature:      {args.temperature}")
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

    # Distribution
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
