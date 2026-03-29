#!/usr/bin/env python3
"""
benchmark_online.py — Wall-clock speculative decoding benchmark.

Measures real end-to-end speedup: DFlash drafter proposes tokens,
Qwen3.5 verifies them live via C++ target daemon.

Usage:
  python scripts/benchmark_online.py \
    --checkpoint checkpoints/best.pt \
    --eval-prompts data/eval_prompts.jsonl \
    --max-tokens 256
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config import DFlashConfig
from chimere.modeling import DFlashDraftModel
from chimere.modeling_v5 import DFlashDraftModelV5
from chimere.spec_decode import SpeculativeDecoder, SpecDecodeStats, generate_online_v5
from chimere.target_daemon import TargetDaemon


def _generate_ar(daemon, prompt_text, max_tokens, stop_token_ids=None):
    """Pure AR baseline using target daemon (model-agnostic)."""
    if stop_token_ids is None:
        stop_token_ids = {248046}

    stats = SpecDecodeStats()
    t_start = time.time()

    prompt_tokens = daemon.tokenize(prompt_text)
    _, logits = daemon.eval_full(prompt_tokens)

    generated = []
    while len(generated) < max_tokens:
        next_token = int(logits[-1])
        if next_token in stop_token_ids:
            break
        generated.append(next_token)
        stats.total_target_calls += 1
        _, logits = daemon.eval_incr([next_token])

    stats.total_tokens = len(generated)
    stats.total_steps = len(generated)
    stats.total_time_ms = (time.time() - t_start) * 1000

    text = daemon.detokenize(generated) if generated else ""
    return text, stats


def load_eval_prompts(path, n_prompts=0):
    """Load evaluation prompts from JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                prompts.append(data.get("text", data.get("prompt", "")))
    if n_prompts > 0:
        prompts = prompts[:n_prompts]
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Online wall-clock DFlash speculative decoding benchmark"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained drafter checkpoint",
    )
    parser.add_argument(
        "--eval-prompts", type=str, default="data/eval_prompts.jsonl",
        help="Path to JSONL file with evaluation prompts",
    )
    parser.add_argument(
        "--n-prompts", type=int, default=0,
        help="Number of prompts to evaluate (0 = all)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens to generate per prompt",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Drafting temperature (0 = greedy)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device for drafter model",
    )
    parser.add_argument(
        "--daemon-path", type=str, default="extract/build/target_daemon",
        help="Path to C++ target daemon binary",
    )
    parser.add_argument(
        "--model-path", type=str,
        default=os.path.expanduser(
            "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/"
            "Qwen3.5-35B-A3B-MXFP4_MOE.gguf"
        ),
        help="Path to target model GGUF",
    )
    parser.add_argument(
        "--layers", type=str, default="2,11,20,29,37",
        help="Comma-separated target layer IDs for hidden state extraction",
    )
    parser.add_argument(
        "--no-stop-server", action="store_true",
        help="Don't stop/restart qwen35-llama systemd service",
    )
    parser.add_argument(
        "--output", type=str, default="logs/benchmark_online_results.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--skip-ar", action="store_true",
        help="Skip AR baseline (only run speculative decoding)",
    )
    parser.add_argument(
        "--n-denoise-steps", type=int, default=1,
        help="Number of denoising steps for v5 multi-step drafting (1=single-step)",
    )
    args = parser.parse_args()

    # ── Load checkpoint & build drafter model ──────────────────────

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "config" in ckpt:
        config_dict = ckpt["config"]
        config = DFlashConfig(**{
            f.name: config_dict[f.name]
            for f in fields(DFlashConfig)
            if f.name in config_dict
        })
    else:
        config = DFlashConfig()

    # Detect v5 (KV injection) vs v3/v4 (fixed-size blocks)
    is_v5 = ckpt.get("version", "").startswith("v5")
    if is_v5:
        print("  Architecture: v5 (full context KV injection)")
        model = DFlashDraftModelV5(config).to(device)
    else:
        print("  Architecture: v3/v4 (fixed-size blocks)")
        model = DFlashDraftModel(config).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Cast frozen embedding/LM-head to BF16
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    print(f"  Config: block_size={config.block_size}, "
          f"layers={config.drafter_num_layers}")

    # ── Load prompts ───────────────────────────────────────────────

    prompts = load_eval_prompts(args.eval_prompts, args.n_prompts)
    print(f"Loaded {len(prompts)} evaluation prompts")
    if not prompts:
        print("ERROR: no prompts found, exiting.")
        sys.exit(1)

    # ── Stop llama-server if needed ────────────────────────────────

    if not args.no_stop_server:
        print("Stopping qwen35-llama service...")
        subprocess.run(
            ["systemctl", "--user", "stop", "qwen35-llama"], check=False
        )
        time.sleep(2)

    # ── Launch C++ target daemon ───────────────────────────────────

    extra_args = ["-ot", ".ffn_.*_exps.=CPU", "--flash-attn", "on"]
    layers = [int(x) for x in args.layers.split(",")]

    print(f"Launching target daemon: {args.daemon_path}")
    print(f"  Model: {args.model_path}")
    print(f"  Layers: {layers}")

    try:
        with TargetDaemon(
            daemon_path=args.daemon_path,
            model_path=args.model_path,
            layers=layers,
            n_gpu_layers=99,
            extra_args=extra_args,
        ) as daemon:
            if not is_v5:
                decoder = SpeculativeDecoder(
                    drafter=model, config=config, device=str(device)
                )

            results = []
            ar_total_time = 0.0
            sd_total_time = 0.0
            ar_total_tokens = 0
            sd_total_tokens = 0

            for i, prompt in enumerate(prompts):
                print(f"\n[{i+1}/{len(prompts)}] {prompt[:80]}...")
                result = {"prompt": prompt[:200], "prompt_idx": i}

                # ── AR baseline ────────────────────────────────────
                if not args.skip_ar:
                    daemon.clear_kv()
                    ar_text, ar_stats = _generate_ar(
                        daemon, prompt, args.max_tokens
                    )
                    ar_tok_s = (
                        ar_stats.total_tokens
                        / max(0.001, ar_stats.total_time_ms / 1000)
                    )
                    result["ar"] = {
                        "tokens": ar_stats.total_tokens,
                        "time_ms": ar_stats.total_time_ms,
                        "tok_per_sec": ar_tok_s,
                    }
                    ar_total_time += ar_stats.total_time_ms
                    ar_total_tokens += ar_stats.total_tokens
                    print(
                        f"  AR:  {ar_stats.total_tokens} tok "
                        f"in {ar_stats.total_time_ms:.0f}ms "
                        f"({ar_tok_s:.1f} tok/s)"
                    )

                # ── Speculative decoding ───────────────────────────
                daemon.clear_kv()
                if is_v5:
                    sd_text, sd_stats = generate_online_v5(
                        model, prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        target_daemon=daemon,
                        max_ctx_len=512,
                        n_denoise_steps=args.n_denoise_steps,
                    )
                else:
                    sd_text, sd_stats = decoder.generate_online(
                        prompt,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        target_daemon=daemon,
                    )
                sd_tok_s = (
                    sd_stats.total_tokens
                    / max(0.001, sd_stats.total_time_ms / 1000)
                )
                result["spec_decode"] = {
                    "tokens": sd_stats.total_tokens,
                    "time_ms": sd_stats.total_time_ms,
                    "steps": sd_stats.total_steps,
                    "drafted": sd_stats.total_drafted,
                    "accepted": sd_stats.total_accepted,
                    "acceptance_rate": sd_stats.acceptance_rate,
                    "tokens_per_step": sd_stats.tokens_per_step,
                    "draft_time_ms": sd_stats.draft_time_ms,
                    "verify_time_ms": sd_stats.verify_time_ms,
                    "tok_per_sec": sd_tok_s,
                }
                sd_total_time += sd_stats.total_time_ms
                sd_total_tokens += sd_stats.total_tokens

                # Per-prompt speedup
                if not args.skip_ar and ar_stats.total_time_ms > 0:
                    speedup = ar_stats.total_time_ms / max(
                        1, sd_stats.total_time_ms
                    )
                    result["speedup"] = speedup
                    print(
                        f"  SD:  {sd_stats.total_tokens} tok "
                        f"in {sd_stats.total_time_ms:.0f}ms "
                        f"(accept={sd_stats.acceptance_rate:.1%}, "
                        f"t={sd_stats.tokens_per_step:.1f}, "
                        f"speedup={speedup:.2f}x)"
                    )
                else:
                    print(
                        f"  SD:  {sd_stats.total_tokens} tok "
                        f"in {sd_stats.total_time_ms:.0f}ms "
                        f"(accept={sd_stats.acceptance_rate:.1%}, "
                        f"t={sd_stats.tokens_per_step:.1f})"
                    )

                results.append(result)

    finally:
        # ── Restart llama-server if we stopped it ──────────────────
        if not args.no_stop_server:
            print("\nRestarting qwen35-llama service...")
            subprocess.run(
                ["systemctl", "--user", "start", "qwen35-llama"], check=False
            )

    # ── Aggregate results ──────────────────────────────────────────

    n = len(results)
    if n == 0:
        print("No results collected.")
        return

    print("\n" + "=" * 70)
    print(" Online Benchmark Results")
    print("=" * 70)

    if not args.skip_ar and ar_total_time > 0:
        ar_tok_s = ar_total_tokens / max(0.001, ar_total_time / 1000)
        print(
            f"  AR baseline:     {ar_total_tokens} tokens "
            f"in {ar_total_time / 1000:.1f}s ({ar_tok_s:.1f} tok/s)"
        )

    sd_tok_s = sd_total_tokens / max(0.001, sd_total_time / 1000)
    print(
        f"  Spec decode:     {sd_total_tokens} tokens "
        f"in {sd_total_time / 1000:.1f}s ({sd_tok_s:.1f} tok/s)"
    )

    overall_speedup = None
    if not args.skip_ar and ar_total_time > 0:
        ar_tok_s_agg = ar_total_tokens / max(0.001, ar_total_time / 1000)
        overall_speedup = sd_tok_s / max(0.001, ar_tok_s_agg)
        print(f"  Overall speedup: {overall_speedup:.2f}x")

    # Acceptance stats
    accept_rates = [r["spec_decode"]["acceptance_rate"] for r in results]
    tokens_per_step = [r["spec_decode"]["tokens_per_step"] for r in results]
    print(
        f"\n  Acceptance rate:  mean={np.mean(accept_rates):.1%}, "
        f"median={np.median(accept_rates):.1%}, "
        f"p5={np.percentile(accept_rates, 5):.1%}, "
        f"p95={np.percentile(accept_rates, 95):.1%}"
    )
    print(
        f"  Tokens/step (t): mean={np.mean(tokens_per_step):.1f}, "
        f"median={np.median(tokens_per_step):.1f}"
    )

    # Timing breakdown
    draft_times = [r["spec_decode"]["draft_time_ms"] for r in results]
    verify_times = [r["spec_decode"]["verify_time_ms"] for r in results]
    print(f"\n  Draft time:   mean={np.mean(draft_times):.0f}ms/prompt")
    print(f"  Verify time:  mean={np.mean(verify_times):.0f}ms/prompt")

    if not args.skip_ar:
        speedups = [r["speedup"] for r in results if "speedup" in r]
        if speedups:
            print(
                f"\n  Per-prompt speedup: "
                f"mean={np.mean(speedups):.2f}x, "
                f"median={np.median(speedups):.2f}x, "
                f"p5={np.percentile(speedups, 5):.2f}x, "
                f"p95={np.percentile(speedups, 95):.2f}x"
            )

    # ── Save results ───────────────────────────────────────────────

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_prompts": n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "checkpoint": args.checkpoint,
        "model": args.model_path,
        "layers": layers,
        "sd_total_tokens": sd_total_tokens,
        "sd_total_time_ms": sd_total_time,
        "sd_tok_per_sec": sd_tok_s,
        "mean_acceptance_rate": float(np.mean(accept_rates)),
        "mean_tokens_per_step": float(np.mean(tokens_per_step)),
        "mean_draft_time_ms": float(np.mean(draft_times)),
        "mean_verify_time_ms": float(np.mean(verify_times)),
        "per_prompt": results,
    }
    if not args.skip_ar and ar_total_time > 0:
        summary["ar_total_tokens"] = ar_total_tokens
        summary["ar_total_time_ms"] = ar_total_time
        summary["ar_tok_per_sec"] = ar_tok_s_agg
        summary["overall_speedup"] = overall_speedup

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
