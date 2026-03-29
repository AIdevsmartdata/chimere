#!/usr/bin/env python3
"""
Measure wall-clock speedup: tokens/s WITH drafter vs WITHOUT (autoregressive baseline).

Runs the same prompts in two modes:
  1. Autoregressive: target generates one token at a time
  2. Speculative: drafter proposes K tokens, target verifies

Reports: tokens/s, latency/token, speedup ratio.
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
    if device.type == "cuda":
        # Split-device BF16: blocks+projs on GPU (~0.98 GB), embed_tokens+lm_head on CPU
        model.eval().float()  # start all on CPU FP32
        bf = torch.bfloat16
        for block in model.layers:
            block.to(device=device, dtype=bf)
        for proj in model.ctx_k_projs:
            proj.to(device=device, dtype=bf)
        for proj in model.ctx_v_projs:
            proj.to(device=device, dtype=bf)
        model.fc.to(device=device, dtype=bf)
        model.hidden_norm.to(device=device, dtype=bf)
        model.norm.to(device=device, dtype=bf)
        model.lm_head.to(device=device, dtype=bf)
        model.rotary_emb.to(device=device)
        if model.input_proj is not None:
            model.input_proj.to(device=device, dtype=bf)
        if model.output_proj is not None:
            model.output_proj.to(device=device, dtype=bf)
        # embed_tokens stays CPU FP32 (lookup only, fast)
    else:
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

        t0 = time.time()
        for _ in range(max_tokens - 1):
            _, argmax = target.eval_incr([generated[-1]])
            next_tok = int(argmax[0])
            generated.append(next_tok)
            if next_tok in (151643, 151645):
                break
        elapsed = time.time() - t0

        target.clear_kv()
        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": len(generated) / elapsed,
        })
    return results


def bench_speculative(target, drafter, config, prompts, max_tokens, drafter_device, adaptive=True,
                      single_save=False, gdn_save=False):
    """Speculative decoding with DFlash drafter + state save/restore.

    Two save/restore strategies are supported:

    Default (single_save=False):
        Classic per-cycle save — target.save_state() before every eval_incr(draft),
        then target.restore_state() on rejection.  Cost: ~63 MB serialized every cycle.

    single_save=True:
        Save the target state ONCE after prefill, keep it for the entire prompt.
        On rejection: restore to that post-prefill snapshot, then eval_incr over ALL
        generated tokens so far (including the correction token).  This rebuilds the
        KV cache in O(generated) time instead of O(prompt + generated).  The resulting
        hidden states become the new hidden_all for the drafter.
        Trade-off:
          - Saves ~63 MB serialisation overhead on every accepted cycle.
          - Rejection is more expensive when generated sequence is long (extra eval_incr).
          - Wins when τ is high and rejection events are rare; breaks even around τ=50%.
    """
    K = config.block_size
    n_layers = config.num_feature_layers
    results = []

    for prompt in prompts:
        tokens = target.tokenize(prompt)
        target.clear_kv()

        # Prefill — cast to float16 to match training data precision
        hidden_all, argmax = target.eval_full(tokens)
        hidden_all = hidden_all.astype(np.float16)
        last_target_pred = int(argmax[-1])
        generated = []
        total_accepted = 0
        total_drafted = 0
        recent_tau = []

        # --single-save: checkpoint state right after prefill, hold it for the whole prompt
        if single_save:
            target.save_state()
            # prefill_hidden_all is the hidden states for the prompt only; we track
            # generated hidden states separately and concatenate after each rejection.
            prefill_hidden_all = hidden_all.copy()

        t0 = time.time()
        while len(generated) < max_tokens:
            # Compute adaptive draft length based on recent acceptance rate
            if adaptive:
                if len(recent_tau) > 0:
                    avg_tau = sum(recent_tau[-3:]) / len(recent_tau[-3:])
                else:
                    avg_tau = 1.0  # optimistic start
                adaptive_k = 15 if avg_tau > 0.5 else (8 if avg_tau > 0.2 else 4)
            else:
                adaptive_k = K - 1

            seq_len = hidden_all.shape[1]
            ctx_end = seq_len
            ctx_start = max(0, ctx_end - config.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            hidden_list = [
                torch.from_numpy(
                    hidden_all[j, ctx_start:ctx_end].astype(np.float16).astype(np.float32)
                ).unsqueeze(0).to(drafter_device)
                for j in range(n_layers)
            ]
            ctx_lengths = torch.tensor([ctx_len], device=drafter_device, dtype=torch.long)
            anchor_pos = ctx_end - 1
            anchor_positions = torch.tensor([anchor_pos], device=drafter_device, dtype=torch.long)

            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            # Draft
            draft_ids, _, _ = drafter.generate_block(
                hidden_list, context_lengths=ctx_lengths,
                temperature=0.0,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            draft_tokens = draft_ids[0].cpu().tolist()

            # Truncate to adaptive length
            draft_tokens = draft_tokens[:adaptive_k]

            # Save state before verification
            if gdn_save:
                # GDN-only save (~2 MB): save recurrent states, trim attention KV later
                save_kv_pos = len(tokens) + len(generated)  # current KV position
                target.save_state_gdn()
            elif not single_save:
                # Default: save full state (~63 MB)
                target.save_state()

            # Verify with eval_incr (O(K) not O(n))
            incr_hidden, incr_argmax = target.eval_incr(draft_tokens)

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
                # ── Rejection path ──────────────────────────────────────────────────────
                correction = target_preds[n_accepted]
                generated.append(correction)

                if gdn_save:
                    # GDN-only restore: restore recurrent states + trim attention KV
                    keep_n = save_kv_pos + n_accepted + 1  # keep prompt + accepted + correction
                    target.restore_state_gdn(save_kv_pos)
                    # Re-eval accepted + correction to update GDN state and get hidden states
                    accepted_plus_correction = draft_tokens[:n_accepted] + [correction]
                    re_hidden, re_argmax = target.eval_incr(accepted_plus_correction)
                    last_target_pred = int(re_argmax[-1])

                    n_new = len(accepted_plus_correction)
                    hidden_all = np.concatenate([
                        hidden_all, re_hidden[:, :n_new, :]
                    ], axis=1)
                elif single_save:
                    # Restore to the post-prefill snapshot, then replay ALL generated
                    # tokens (including correction) via eval_incr.  This is O(generated)
                    # rather than O(prompt + generated), and amortises the single save cost.
                    target.restore_state()
                    # generated already contains the accepted tokens + correction appended above
                    re_hidden, re_argmax = target.eval_incr(generated)
                    last_target_pred = int(re_argmax[-1])

                    # Rebuild hidden_all: prefill portion + all generated positions
                    hidden_all = np.concatenate([
                        prefill_hidden_all, re_hidden
                    ], axis=1)
                else:
                    # Default: restore to per-cycle checkpoint, eval only accepted + correction
                    target.restore_state()
                    accepted_plus_correction = draft_tokens[:n_accepted] + [correction]
                    re_hidden, re_argmax = target.eval_incr(accepted_plus_correction)
                    last_target_pred = int(re_argmax[-1])

                    # Update hidden_all with accepted hidden states
                    n_new = len(accepted_plus_correction)
                    hidden_all = np.concatenate([
                        hidden_all, re_hidden[:, :n_new, :]
                    ], axis=1)
            else:
                # ── Full accept path ────────────────────────────────────────────────────
                # All accepted — bonus token
                bonus = int(incr_argmax[-1])
                generated.append(bonus)
                bonus_hidden, bonus_argmax = target.eval_incr([bonus])
                last_target_pred = int(bonus_argmax[0])

                # Update hidden_all with all draft + bonus hidden states
                hidden_all = np.concatenate([
                    hidden_all, incr_hidden, bonus_hidden
                ], axis=1)
                # Note: in single_save mode the KV cache grows naturally here — no
                # save/restore needed.  hidden_all already reflects the true state.

            total_accepted += n_accepted
            total_drafted += len(draft_tokens)

            # Update recent_tau for adaptive K (cycle τ = n_accepted / draft_used)
            cycle_tau = n_accepted / len(draft_tokens) if draft_tokens else 0.0
            recent_tau.append(cycle_tau)
            if len(recent_tau) > 3:
                recent_tau = recent_tau[-3:]

            if generated and generated[-1] in (151643, 151645):
                break

        elapsed = time.time() - t0
        target.clear_kv()

        tau = total_accepted / max(1, total_drafted)
        results.append({
            "tokens": len(generated),
            "time": elapsed,
            "tok_per_s": len(generated) / elapsed,
            "tau": tau,
            "accepted_per_block": total_accepted / max(1, total_drafted / (K - 1)),
            "adaptive_k": adaptive_k,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Wall-clock speedup benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v8_online_c4/best.pt")
    parser.add_argument("--model", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"))
    parser.add_argument("--daemon", type=str, default="extract/build/target_daemon")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--drafter-device", type=str, default="cpu")
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                        help="Use adaptive draft length based on recent acceptance rate (default: True)")
    parser.add_argument("--ctx-size", type=int, default=256,
                        help="Context size for target daemon (default: 256, enough for benchmark)")
    parser.add_argument("--no-offload", action="store_true", default=False,
                        help="Disable CPU offload (for IQ3 full-GPU mode)")
    parser.add_argument("--single-save", action="store_true", default=False,
                        help="Save state once after prefill (amortized, good for high τ)")
    parser.add_argument("--gdn-save", action="store_true", default=False,
                        help=(
                            "GDN-only save/restore: save only recurrent states (~2 MB) "
                            "instead of full KV+recurrent (~63 MB). Trims attention KV "
                            "cache with seq_rm (O(1)). Requires llama.cpp build with "
                            "LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY support."
                        ))
    args = parser.parse_args()

    drafter_device = torch.device(args.drafter_device)
    layers = [int(x) for x in args.layers.split(",")]
    prompts = BENCH_PROMPTS

    # Launch daemon FIRST (needs most VRAM)
    print("Launching target daemon...", flush=True)
    extra_args = [
        "-c", str(args.ctx_size), "-t", "6",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
    ]
    if not args.no_offload:
        extra_args += ["-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU"]
    target = TargetDaemon(
        daemon_path=args.daemon,
        model_path=args.model,
        layers=layers,
        n_gpu_layers=99,
        extra_args=extra_args,
    )
    target.tokenize("warmup")
    print("Daemon ready.\n", flush=True)

    # Load drafter (after daemon to fit in remaining VRAM)
    print("Loading drafter...", flush=True)
    drafter, config = load_drafter(args.checkpoint, drafter_device)

    # Warmup
    print("Warmup (1 prompt each mode)...", flush=True)
    bench_autoregressive(target, prompts[:1], 32)
    bench_speculative(target, drafter, config, prompts[:1], 32, drafter_device,
                      adaptive=args.adaptive, single_save=args.single_save,
                      gdn_save=args.gdn_save)

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

    # Benchmark speculative
    save_mode_label = "GDN-only save (~2 MB)" if args.gdn_save else ("single-save (post-prefill)" if args.single_save else "per-cycle save (~63 MB)")
    print(f"\n{'='*60}")
    print(f" SPECULATIVE DECODING ({len(prompts)} prompts, {args.max_tokens} tokens, {save_mode_label})")
    print(f"{'='*60}")
    spec_results = bench_speculative(target, drafter, config, prompts, args.max_tokens, drafter_device,
                                     adaptive=args.adaptive, single_save=args.single_save,
                                     gdn_save=args.gdn_save)
    for i, r in enumerate(spec_results):
        print(f"  [{i+1:2d}] {r['tokens']:3d} tok | {r['time']:5.1f}s | {r['tok_per_s']:5.1f} tok/s | τ={r['tau']:.1%} | K={r['adaptive_k']}")
    spec_avg_tps = np.mean([r['tok_per_s'] for r in spec_results])
    spec_avg_time = np.mean([r['time'] for r in spec_results])
    spec_avg_tau = np.mean([r['tau'] for r in spec_results])

    print(f"\n  AVG: {spec_avg_tps:.1f} tok/s, {spec_avg_time:.1f}s/prompt, τ={spec_avg_tau:.1%}")

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
        print(f"  (overhead from clear_kv + eval_full dominates at this τ level)")

    target.close()


if __name__ == "__main__":
    main()
