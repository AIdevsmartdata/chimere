#!/usr/bin/env python3
"""
benchmark_spec_decode.py — Measure DFlash drafter acceptance rate.

Uses pre-extracted hidden states + ground truth tokens to simulate
speculative decoding and measure how many draft tokens the target
model would accept.

Usage:
  python scripts/benchmark_spec_decode.py \
    --checkpoint checkpoints/best.pt \
    --blocks-dir data/blocks \
    --n-blocks 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config import DFlashConfig
from chimere.modeling import DFlashDraftModel
from chimere.spec_decode import SpeculativeDecoder, SpecDecodeStats
from chimere.data import DFlashDataset


def main():
    parser = argparse.ArgumentParser(description="Benchmark DFlash spec decode")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained drafter checkpoint")
    parser.add_argument("--blocks-dir", type=str, default="data/blocks",
                        help="Directory with extracted blocks")
    parser.add_argument("--n-blocks", type=int, default=0,
                        help="Number of blocks to evaluate (0 = all)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Drafting temperature")
    parser.add_argument("--n-steps", type=int, default=8,
                        help="Number of denoising steps (1=single-step, 8=default)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "config" in ckpt:
        from dataclasses import fields
        config_dict = ckpt["config"]
        config = DFlashConfig(**{
            f.name: config_dict[f.name]
            for f in fields(DFlashConfig)
            if f.name in config_dict
        })
    else:
        config = DFlashConfig()

    # Build model
    model = DFlashDraftModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Cast frozen params to BF16
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    print(f"  Config: block_size={config.block_size}, layers={config.drafter_num_layers}")
    print()

    # Load blocks
    dataset = DFlashDataset(args.blocks_dir, block_size=config.block_size)
    n_blocks = args.n_blocks if args.n_blocks > 0 else len(dataset)
    n_blocks = min(n_blocks, len(dataset))
    print(f"  Evaluating {n_blocks}/{len(dataset)} blocks")
    print()

    # Create decoder (no target URL needed for offline benchmark)
    decoder = SpeculativeDecoder(
        drafter=model,
        config=config,
        device=str(device),
    )

    # Run offline benchmark
    total_stats = SpecDecodeStats()
    per_block_accepted = []

    t0 = time.time()
    for i in range(n_blocks):
        block = dataset[i]

        # Wrap as single-block list for benchmark_offline
        block_data = [{
            "block_hidden": block["block_hidden"],
            "block_input_ids": block["block_input_ids"],
        }]

        stats = decoder.benchmark_offline(
            prompt_tokens=[],
            block_hidden_states=block_data,
            ground_truth_tokens=block["block_input_ids"].tolist(),
            temperature=args.temperature,
            n_steps=args.n_steps,
        )

        total_stats.total_drafted += stats.total_drafted
        total_stats.total_accepted += stats.total_accepted
        total_stats.total_tokens += stats.total_tokens
        total_stats.total_steps += stats.total_steps
        total_stats.total_target_calls += stats.total_target_calls
        total_stats.draft_time_ms += stats.draft_time_ms

        per_block_accepted.append(stats.total_accepted)

        if (i + 1) % 50 == 0:
            running_rate = total_stats.acceptance_rate
            print(f"  [{i+1}/{n_blocks}] running accept_rate={running_rate:.2%}")

    elapsed = time.time() - t0
    total_stats.total_time_ms = elapsed * 1000

    # Results
    print()
    print("=" * 60)
    print(" Offline Benchmark Results")
    print("=" * 60)
    print(f"  Blocks evaluated:    {n_blocks}")
    print(f"  Block size:          {config.block_size}")
    print(f"  Temperature:         {args.temperature}")
    print(f"  Denoising steps:     {args.n_steps}")
    print(f"  Total drafted:       {total_stats.total_drafted}")
    print(f"  Total accepted:      {total_stats.total_accepted}")
    print(f"  Acceptance rate:     {total_stats.acceptance_rate:.2%}")
    print(f"  Avg tokens/step:     {total_stats.tokens_per_step:.1f}")
    print(f"  Estimated speedup:   {total_stats.speedup_vs_ar:.2f}x")
    print(f"  Draft time:          {total_stats.draft_time_ms:.0f}ms "
          f"({total_stats.draft_time_ms / max(1, n_blocks):.1f}ms/block)")
    print(f"  Total time:          {elapsed:.1f}s")
    print()

    # Acceptance distribution
    if per_block_accepted:
        import collections
        dist = collections.Counter(per_block_accepted)
        print("  Acceptance distribution (tokens accepted per block):")
        for k in sorted(dist.keys()):
            bar = "#" * dist[k]
            print(f"    {k:2d}: {dist[k]:4d} {bar}")
    print()

    # What this means for real speedup
    avg_accept = total_stats.acceptance_rate
    K = config.block_size
    # Expected tokens per step = sum_{i=0}^{K} (1-alpha)^i = (1 - (1-alpha)^(K+1)) / alpha
    # Simplified: avg_accepted + 1 (for bonus/resample)
    est_tokens_per_step = total_stats.tokens_per_step
    print(f"  With K={K} draft tokens:")
    print(f"    AR baseline:         1 token/target_call")
    print(f"    Spec decode:         {est_tokens_per_step:.1f} tokens/target_call")
    print(f"    Ideal max speedup:   {K + 1}x (all accepted + bonus)")
    print(f"    Your speedup:        {total_stats.speedup_vs_ar:.2f}x")
    print()
    print("  NOTE: Real speedup depends on draft_time/target_time ratio.")
    print("  If draft takes 5ms and target takes 100ms:")
    target_ms = 100  # estimate
    draft_ms = total_stats.draft_time_ms / max(1, n_blocks)
    real_speedup = est_tokens_per_step * target_ms / (draft_ms + target_ms)
    print(f"    Real speedup ≈ {real_speedup:.2f}x "
          f"(draft={draft_ms:.0f}ms, target~{target_ms}ms)")


if __name__ == "__main__":
    main()
