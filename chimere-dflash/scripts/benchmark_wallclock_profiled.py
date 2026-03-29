#!/usr/bin/env python3
"""
Measure wall-clock speedup: tokens/s WITH drafter vs WITHOUT (autoregressive baseline).
PROFILED VERSION — adds per-operation timing breakdown inside bench_speculative.

Runs the same prompts in two modes:
  1. Autoregressive: target generates one token at a time
  2. Speculative: drafter proposes K tokens, target verifies

Reports: tokens/s, latency/token, speedup ratio.
Per-cycle timings: draft, save, verify, restore, correction, bonus, hidden (numpy ops).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.modeling_v8 import DFlashDraftModelV8
from chimere.target_daemon import TargetDaemon


def load_drafter(checkpoint_path: str, device: torch.device):
    from dataclasses import fields
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DFlashV7Config(**{
        f.name: ckpt["config"][f.name]
        for f in fields(DFlashV7Config)
        if f.name in ckpt["config"]
    })
    model = DFlashDraftModelV8(config)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval().float()
    return model, config


BENCH_PROMPTS = [
    "Explain how speculative decoding works in large language models.",
    "Write a Python function to compute the Fibonacci sequence using memoization.",
    "What are the advantages of MoE (Mixture of Experts) architectures?",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "The transformer architecture introduced in 'Attention Is All You Need' uses",
    "Describe the difference between TCP and UDP protocols.",
    "Les principes fondamentaux de la kinésithérapie reposent sur",
    "Docker containers differ from virtual machines because",
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self",
    "The CAP theorem states that a distributed system cannot simultaneously guarantee",
]


def bench_autoregressive(target, prompts, max_tokens):
    """Baseline: target generates one token at a time."""
    results = []
    for prompt in prompts:
        tokens = target.tokenize(prompt)
        target.clear_kv()

        # Prefill
        _, argmax = target.eval_full(tokens)
        generated = [int(argmax[-1])]

        t0 = time.perf_counter()
        for _ in range(max_tokens - 1):
            _, argmax = target.eval_incr([generated[-1]])
            next_tok = int(argmax[0])
            generated.append(next_tok)
            if next_tok in (151643, 151645):
                break
        elapsed = time.perf_counter() - t0

        target.clear_kv()
        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": len(generated) / elapsed,
        })
    return results


def bench_speculative(target, drafter, config, prompts, max_tokens, drafter_device, adaptive=True):
    """Speculative decoding with DFlash drafter + state save/restore.
    PROFILED: per-cycle timing for draft, save, verify, restore, correction, bonus, hidden ops.
    """
    K = config.block_size
    n_layers = config.num_feature_layers
    results = []

    for prompt_idx, prompt in enumerate(prompts):
        tokens = target.tokenize(prompt)
        target.clear_kv()

        # Prefill
        hidden_all, argmax = target.eval_full(tokens)
        last_target_pred = int(argmax[-1])
        generated = []
        total_accepted = 0
        total_drafted = 0
        recent_tau = []

        # Per-prompt accumulated timings
        acc_draft    = 0.0
        acc_save     = 0.0
        acc_verify   = 0.0
        acc_restore  = 0.0
        acc_correction = 0.0
        acc_bonus    = 0.0
        acc_hidden   = 0.0
        n_cycles     = 0

        t0 = time.perf_counter()
        while len(generated) < max_tokens:
            # ----------------------------------------------------------------
            # Adaptive draft length
            # ----------------------------------------------------------------
            if adaptive:
                if len(recent_tau) > 0:
                    avg_tau = sum(recent_tau[-3:]) / len(recent_tau[-3:])
                else:
                    avg_tau = 1.0  # optimistic start
                adaptive_k = 15 if avg_tau > 0.5 else (8 if avg_tau > 0.2 else 4)
            else:
                adaptive_k = K - 1

            # ----------------------------------------------------------------
            # Hidden-state prep (numpy / tensor ops)
            # ----------------------------------------------------------------
            _th0 = time.perf_counter()

            seq_len = hidden_all.shape[1]
            ctx_end = seq_len
            ctx_start = max(0, ctx_end - config.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            hidden_list = [
                torch.from_numpy(hidden_all[j, ctx_start:ctx_end].copy())
                .unsqueeze(0).to(drafter_device)
                for j in range(n_layers)
            ]
            ctx_lengths = torch.tensor([ctx_len], device=drafter_device, dtype=torch.long)
            anchor_pos = ctx_end - 1
            anchor_positions = torch.tensor([anchor_pos], device=drafter_device, dtype=torch.long)

            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            acc_hidden += time.perf_counter() - _th0

            # ----------------------------------------------------------------
            # 1. Draft
            # ----------------------------------------------------------------
            _td0 = time.perf_counter()
            draft_ids, _, _ = drafter.generate_block(
                hidden_list, context_lengths=ctx_lengths,
                temperature=0.0,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            draft_tokens = draft_ids[0].cpu().tolist()
            acc_draft += time.perf_counter() - _td0

            # Truncate to adaptive length
            draft_tokens = draft_tokens[:adaptive_k]

            # ----------------------------------------------------------------
            # 2. Save state BEFORE verification eval
            # ----------------------------------------------------------------
            _ts0 = time.perf_counter()
            target.save_state()
            acc_save += time.perf_counter() - _ts0

            # ----------------------------------------------------------------
            # 3. Verify with eval_incr (O(K) not O(n))
            # ----------------------------------------------------------------
            _tv0 = time.perf_counter()
            incr_hidden, incr_argmax = target.eval_incr(draft_tokens)
            acc_verify += time.perf_counter() - _tv0

            # Verification: draft[0] vs last_target_pred, draft[j] vs incr_argmax[j-1]
            target_preds = [last_target_pred] + incr_argmax[:-1].tolist()

            n_accepted = 0
            for j in range(len(draft_tokens)):
                if j < len(target_preds) and draft_tokens[j] == target_preds[j]:
                    n_accepted += 1
                else:
                    break

            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            if n_accepted < len(draft_tokens):
                # ----------------------------------------------------------------
                # Partial accept: restore state, eval accepted + correction
                # ----------------------------------------------------------------
                correction = target_preds[n_accepted]
                generated.append(correction)

                _tr0 = time.perf_counter()
                target.restore_state()
                acc_restore += time.perf_counter() - _tr0

                accepted_plus_correction = draft_tokens[:n_accepted] + [correction]

                _tc0 = time.perf_counter()
                re_hidden, re_argmax = target.eval_incr(accepted_plus_correction)
                acc_correction += time.perf_counter() - _tc0

                last_target_pred = int(re_argmax[-1])

                # Update hidden_all with accepted hidden states
                _th1 = time.perf_counter()
                n_new = len(accepted_plus_correction)
                hidden_all = np.concatenate([
                    hidden_all, re_hidden[:, :n_new, :]
                ], axis=1)
                acc_hidden += time.perf_counter() - _th1

            else:
                # ----------------------------------------------------------------
                # All accepted: bonus token
                # ----------------------------------------------------------------
                bonus = int(incr_argmax[-1])
                generated.append(bonus)

                _tb0 = time.perf_counter()
                bonus_hidden, bonus_argmax = target.eval_incr([bonus])
                acc_bonus += time.perf_counter() - _tb0

                last_target_pred = int(bonus_argmax[0])

                # Update hidden_all with all draft + bonus hidden states
                _th1 = time.perf_counter()
                hidden_all = np.concatenate([
                    hidden_all, incr_hidden, bonus_hidden
                ], axis=1)
                acc_hidden += time.perf_counter() - _th1

            total_accepted += n_accepted
            total_drafted += len(draft_tokens)
            n_cycles += 1

            # Update recent_tau for adaptive K (cycle τ = n_accepted / draft_used)
            cycle_tau = n_accepted / len(draft_tokens) if draft_tokens else 0.0
            recent_tau.append(cycle_tau)
            if len(recent_tau) > 3:
                recent_tau = recent_tau[-3:]

            if generated and generated[-1] in (151643, 151645):
                break

        elapsed = time.perf_counter() - t0
        target.clear_kv()

        tau = total_accepted / max(1, total_drafted)

        # ----------------------------------------------------------------
        # Per-prompt profiling printout
        # ----------------------------------------------------------------
        tok_per_s = len(generated) / elapsed if elapsed > 0 else 0.0
        accounted = acc_draft + acc_save + acc_verify + acc_restore + acc_correction + acc_bonus + acc_hidden
        unaccounted = elapsed - accounted

        print(f"  [{prompt_idx+1}] {len(generated)} tok | {elapsed:.1f}s | {tok_per_s:.1f} tok/s | τ={tau:.1%} | K={adaptive_k}")
        print(f"      draft={acc_draft:.3f}s  save={acc_save:.3f}s  verify={acc_verify:.3f}s  "
              f"restore={acc_restore:.3f}s  correction={acc_correction:.3f}s  "
              f"bonus={acc_bonus:.3f}s  hidden={acc_hidden:.4f}s  cycles={n_cycles}")
        print(f"      (accounted={accounted:.3f}s  unaccounted={unaccounted:.3f}s)", flush=True)

        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": tok_per_s,
            "tau": tau,
            "accepted_per_block": total_accepted / max(1, total_drafted / (K - 1)),
            "adaptive_k": adaptive_k,
            # timing breakdown
            "t_draft":      acc_draft,
            "t_save":       acc_save,
            "t_verify":     acc_verify,
            "t_restore":    acc_restore,
            "t_correction": acc_correction,
            "t_bonus":      acc_bonus,
            "t_hidden":     acc_hidden,
            "n_cycles":     n_cycles,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Wall-clock speedup benchmark (profiled)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v8_online_c3/best.pt")
    parser.add_argument("--model", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"))
    parser.add_argument("--daemon", type=str, default="extract/build/target_daemon")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--drafter-device", type=str, default="cpu")
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                        help="Use adaptive draft length based on recent acceptance rate (default: True)")
    args = parser.parse_args()

    drafter_device = torch.device(args.drafter_device)
    layers = [int(x) for x in args.layers.split(",")]
    prompts = BENCH_PROMPTS

    # Load drafter
    print("Loading drafter...", flush=True)
    drafter, config = load_drafter(args.checkpoint, drafter_device)

    # Launch daemon
    print("Launching target daemon...", flush=True)
    target = TargetDaemon(
        daemon_path=args.daemon,
        model_path=args.model,
        layers=layers,
        n_gpu_layers=99,
        extra_args=[
            "-c", "512", "-t", "6",
            "-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
        ],
    )
    target.tokenize("warmup")
    print("Daemon ready.\n", flush=True)

    # Warmup
    print("Warmup (1 prompt each mode)...", flush=True)
    bench_autoregressive(target, prompts[:1], 32)
    bench_speculative(target, drafter, config, prompts[:1], 32, drafter_device, adaptive=args.adaptive)

    # Benchmark autoregressive
    print(f"\n{'='*60}")
    print(f" AUTOREGRESSIVE BASELINE ({len(prompts)} prompts, {args.max_tokens} tokens)")
    print(f"{'='*60}")
    ar_results = bench_autoregressive(target, prompts, args.max_tokens)
    for i, r in enumerate(ar_results):
        print(f"  [{i+1:2d}] {r['tokens']:3d} tok | {r['time']:5.1f}s | {r['tok_per_s']:5.1f} tok/s")
    ar_avg_tps = np.mean([r['tok_per_s'] for r in ar_results])
    ar_avg_time = np.mean([r['time'] for r in ar_results])
    print(f"\n  AVG: {ar_avg_tps:.1f} tok/s, {ar_avg_time:.1f}s/prompt")

    # Benchmark speculative (profiled)
    print(f"\n{'='*60}")
    print(f" SPECULATIVE DECODING — PROFILED ({len(prompts)} prompts, {args.max_tokens} tokens)")
    print(f"{'='*60}")
    spec_results = bench_speculative(
        target, drafter, config, prompts, args.max_tokens, drafter_device, adaptive=args.adaptive
    )

    # Overall averages
    spec_avg_tps     = np.mean([r['tok_per_s']  for r in spec_results])
    spec_avg_time    = np.mean([r['time']        for r in spec_results])
    spec_avg_tau     = np.mean([r['tau']         for r in spec_results])
    avg_draft        = np.mean([r['t_draft']     for r in spec_results])
    avg_save         = np.mean([r['t_save']      for r in spec_results])
    avg_verify       = np.mean([r['t_verify']    for r in spec_results])
    avg_restore      = np.mean([r['t_restore']   for r in spec_results])
    avg_correction   = np.mean([r['t_correction'] for r in spec_results])
    avg_bonus        = np.mean([r['t_bonus']     for r in spec_results])
    avg_hidden       = np.mean([r['t_hidden']    for r in spec_results])
    avg_cycles       = np.mean([r['n_cycles']    for r in spec_results])

    print(f"\n{'='*60}")
    print(f" SPECULATIVE TIMING AVERAGES (per prompt)")
    print(f"{'='*60}")
    print(f"  draft      : {avg_draft:.3f}s")
    print(f"  save       : {avg_save:.3f}s")
    print(f"  verify     : {avg_verify:.3f}s")
    print(f"  restore    : {avg_restore:.3f}s  (rejection path only)")
    print(f"  correction : {avg_correction:.3f}s  (rejection path only)")
    print(f"  bonus      : {avg_bonus:.3f}s  (full-accept path only)")
    print(f"  hidden/np  : {avg_hidden:.4f}s")
    print(f"  cycles     : {avg_cycles:.1f}")
    print(f"  τ (accept) : {spec_avg_tau:.1%}")
    print(f"  tok/s      : {spec_avg_tps:.1f}")
    print(f"  time/prompt: {spec_avg_time:.1f}s")

    # Summary
    speedup = spec_avg_tps / ar_avg_tps
    print(f"\n{'='*60}")
    print(f" WALL-CLOCK SPEEDUP")
    print(f"{'='*60}")
    print(f"  Autoregressive:  {ar_avg_tps:.1f} tok/s")
    print(f"  Speculative:     {spec_avg_tps:.1f} tok/s")
    print(f"  Speedup:         {speedup:.2f}×")
    print(f"  τ (acceptance):  {spec_avg_tau:.1%}")
    print(f"{'='*60}")

    if speedup > 1.0:
        print(f"\n  VERDICT: SPECULATIVE IS {speedup:.1f}× FASTER")
    else:
        print(f"\n  VERDICT: SPECULATIVE IS SLOWER ({speedup:.2f}×)")
        print(f"  (overhead from save/restore + eval_incr dominates at this τ level)")

    target.close()


if __name__ == "__main__":
    main()
