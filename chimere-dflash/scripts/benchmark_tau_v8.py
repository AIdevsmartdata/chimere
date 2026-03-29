#!/usr/bin/env python3
"""
benchmark_tau_v8.py — Measure τ (acceptance rate) for DFlash v8 on full-seq holdout.

Unlike v7 benchmark which reads block_tokens.bin (single anchor), this reads
full-sequence data (context_hidden.bin + tokens.bin) and tests MULTIPLE random
anchors per sequence, giving a more representative τ estimate.

Usage:
  python scripts/benchmark_tau_v8.py \
    --checkpoint checkpoints_v8_10k/best.pt \
    --features-dir data/features_holdout_fullseq \
    --n-samples 500 --anchors-per-sample 10
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
    parser = argparse.ArgumentParser(description="Benchmark DFlash v8 τ (full-seq holdout)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--features-dir", type=str, required=True)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--anchors-per-sample", type=int, default=10,
                        help="Random anchors to test per sequence")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)
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

    # Detect model version: use V8 (deep KV) if checkpoint has ctx_k_projs keys or version contains "deepkv"
    version = ckpt.get("version", "")
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    use_deepkv = "deepkv" in str(version) or any("ctx_k_projs" in k for k in state_dict)

    if use_deepkv:
        from chimere.modeling_v8 import DFlashDraftModelV8
        print(f"  Detected deep KV model (version={version})")
        model = DFlashDraftModelV8(config).to(device)
    else:
        print(f"  Using standard V7 model (version={version})")
        model = DFlashDraftModelV7(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.float()

    K = config.block_size
    n_layers = config.num_feature_layers
    H = config.target_hidden_size

    # Discover samples — support both full-seq (tokens.bin) and legacy (block_tokens.bin)
    sample_dirs = sorted([
        d for d in features_dir.iterdir()
        if d.is_dir() and (d / "context_hidden.bin").exists()
    ])
    n_total = len(sample_dirs)
    n_eval = min(args.n_samples, n_total)

    indices = np.linspace(0, n_total - 1, n_eval, dtype=int)
    print(f"  block_size={K}, layers={config.num_hidden_layers}, "
          f"anchors_per_sample={args.anchors_per_sample}")
    print(f"  Evaluating {n_eval}/{n_total} samples "
          f"(up to {n_eval * args.anchors_per_sample} anchor tests)\n")

    # Benchmark
    total_drafted = 0
    total_accepted = 0
    total_blocks = 0
    per_block_accepted = []
    draft_times = []

    t0 = time.time()
    with torch.no_grad():
        for eval_idx, sample_idx in enumerate(indices):
            d = sample_dirs[sample_idx]

            # Load metadata
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            n_positions = meta.get("n_positions", meta.get("seq_len", 0))
            mode = meta.get("mode", "anchor")

            # Load hidden states
            raw = np.fromfile(d / "context_hidden.bin", dtype=np.float16)
            raw = raw.reshape(n_layers, n_positions, H).astype(np.float32)

            # Load tokens
            tokens_path = d / "tokens.bin"
            if tokens_path.exists():
                tokens = np.fromfile(tokens_path, dtype=np.int32)
            elif (d / "block_tokens.bin").exists():
                # Legacy: only one anchor possible
                block_tokens = np.fromfile(d / "block_tokens.bin", dtype=np.int32)
                anchor_pos = meta.get("anchor_pos", 0)
                # Can only test single anchor
                ctx_start = max(0, anchor_pos - n_positions + 1)
                ctx_end = anchor_pos + 1
                ctx_len = ctx_end - ctx_start

                hidden_list = [
                    torch.from_numpy(raw[i, :ctx_len]).unsqueeze(0).to(device)
                    for i in range(n_layers)
                ]
                ctx_lengths = torch.tensor([ctx_len], device=device, dtype=torch.long)
                anchor_positions_t = torch.tensor([anchor_pos], device=device, dtype=torch.long)

                anchor_token_id = int(block_tokens[0])
                t_draft = time.time()
                draft_ids, _, _ = model.generate_block(
                    hidden_list, context_lengths=ctx_lengths,
                    temperature=args.temperature,
                    anchor_token_id=anchor_token_id,
                    anchor_positions=anchor_positions_t,
                )
                draft_time = (time.time() - t_draft) * 1000
                draft_times.append(draft_time)

                draft = draft_ids[0].cpu().numpy()
                mask_id = config.mask_token_id
                n_real = np.sum(block_tokens != mask_id)
                n_accepted = 0
                for j in range(K - 1):
                    tp = j + 1
                    if tp >= len(block_tokens) or block_tokens[tp] == mask_id:
                        break
                    if draft[j] == block_tokens[tp]:
                        n_accepted += 1
                    else:
                        break

                total_drafted += min(n_real - 1, K - 1)
                total_accepted += n_accepted
                total_blocks += 1
                per_block_accepted.append(n_accepted)
                continue

            seq_len = min(len(tokens), n_positions)

            # Sample random anchors
            min_anchor = 1
            max_anchor = seq_len - K
            if max_anchor < min_anchor:
                continue

            n_possible = max_anchor - min_anchor + 1
            n_anchors = min(args.anchors_per_sample, n_possible)
            if n_possible <= n_anchors:
                anchors = list(range(min_anchor, max_anchor + 1))
            else:
                anchors = np.random.choice(
                    range(min_anchor, max_anchor + 1), size=n_anchors, replace=False
                ).tolist()

            for anchor_pos in anchors:
                # Context: hidden states up to and including anchor
                ctx_end = anchor_pos + 1
                ctx_start = max(0, ctx_end - config.max_ctx_len)  # match training
                ctx_len = ctx_end - ctx_start

                hidden_list = [
                    torch.from_numpy(raw[i, ctx_start:ctx_end]).unsqueeze(0).to(device)
                    for i in range(n_layers)
                ]
                ctx_lengths = torch.tensor([ctx_len], device=device, dtype=torch.long)
                anchor_positions_t = torch.tensor([anchor_pos], device=device, dtype=torch.long)

                # Ground truth block tokens
                gt_block = tokens[anchor_pos:min(anchor_pos + K, seq_len)]
                if len(gt_block) < 2:
                    continue

                anchor_token_id = int(gt_block[0])

                t_draft = time.time()
                draft_ids, _, _ = model.generate_block(
                    hidden_list, context_lengths=ctx_lengths,
                    temperature=args.temperature,
                    anchor_token_id=anchor_token_id,
                    anchor_positions=anchor_positions_t,
                )
                draft_time = (time.time() - t_draft) * 1000
                draft_times.append(draft_time)

                draft = draft_ids[0].cpu().numpy()  # [K-1]

                # Count sequential accepted tokens
                n_accepted = 0
                for j in range(min(K - 1, len(gt_block) - 1)):
                    if draft[j] == gt_block[j + 1]:
                        n_accepted += 1
                    else:
                        break

                n_draftable = min(len(gt_block) - 1, K - 1)
                total_drafted += n_draftable
                total_accepted += n_accepted
                total_blocks += 1
                per_block_accepted.append(n_accepted)

            if (eval_idx + 1) % 50 == 0:
                running_tau = total_accepted / max(1, total_drafted)
                avg_acc = np.mean(per_block_accepted) if per_block_accepted else 0
                avg_ms = np.mean(draft_times[-100:]) if draft_times else 0
                print(f"  [{eval_idx+1}/{n_eval}] τ={running_tau:.2%} | "
                      f"avg_accepted={avg_acc:.1f}/{K-1} | "
                      f"blocks={total_blocks} | draft={avg_ms:.1f}ms", flush=True)

    elapsed = time.time() - t0

    tau = total_accepted / max(1, total_drafted)
    avg_accepted = np.mean(per_block_accepted) if per_block_accepted else 0
    avg_draft_ms = np.mean(draft_times) if draft_times else 0

    tokens_per_call = avg_accepted + 1
    target_ms_estimate = 25
    real_speedup = tokens_per_call * target_ms_estimate / (avg_draft_ms + target_ms_estimate)

    print(f"\n{'='*60}")
    print(f" DFlash v8 — Offline τ Benchmark (Full-Seq Multi-Anchor)")
    print(f"{'='*60}")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Samples:          {n_eval} sequences, {total_blocks} anchor tests")
    print(f"  Block size:       {K}")
    print(f"  Temperature:      {args.temperature}")
    print(f"")
    print(f"  Total drafted:    {total_drafted}")
    print(f"  Total accepted:   {total_accepted}")
    print(f"  τ (accept rate):  {tau:.2%}")
    print(f"  Avg accepted:     {avg_accepted:.1f} / {K - 1}")
    print(f"  Avg draft time:   {avg_draft_ms:.1f}ms")
    print(f"  Total time:       {elapsed:.1f}s")
    print(f"")
    print(f"  Tokens/target_call:  {tokens_per_call:.1f}")
    print(f"  Theoretical max:     {K}x")
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

    # Verdict
    print(f"\n  VERDICT: ", end="")
    if tau > 0.35:
        print("EXCELLENT — proche z-lab, scale up!")
    elif tau > 0.20:
        print("VIABLE — DFlash fonctionne, on scale les données")
    elif tau > 0.05:
        print("PROMETTEUR — amélioration significative vs v7, continuer")
    else:
        print("INSUFFISANT — encore de la mémorisation probable")

    print()


if __name__ == "__main__":
    main()
