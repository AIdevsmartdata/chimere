#!/usr/bin/env python3
"""
bench_iq3_quality.py — Compare IQ3_S (full GPU) vs Q5_K_XL (CPU offload) quality.

Runs the same 10 BENCH_PROMPTS through both quantizations sequentially.
Measures: token-by-token agreement, perplexity proxy (log-sum logprobs),
generation speed, and first-divergence analysis.

VRAM budget analysis (RTX 5060 Ti, 16 GB):
  IQ3_S full GPU:
    Weights:          12.7 GB
    KV cache (q8k/q4v, ctx=4096): ~70 MB
    Compute buffer:   ~300 MB
    TOTAL:            ~13.06 GB
    Remaining for DFlash drafter: ~2.94 GB  ← enough for the drafter (~1.5 GB)

  Q5_K_XL with CPU offload (current config):
    Weights on GPU (layers 0-19):  ~13.9 GB
    KV cache (q8k/q4v, ctx=4096): ~70 MB
    TOTAL on GPU:                  ~14.0 GB
    Remaining for DFlash:          ~2.1 GB

Speed estimate for IQ3_S full GPU:
  Q5 with CPU offload (measured): 42.9 tok/s
  IQ3_S full GPU (estimated):    ~85-110 tok/s
  Ratio: ~2.0-2.6x (weight size ratio: 12.7 GB / 24.9 GB = 1.96x)
  RTX 5060 Ti actual bandwidth: 448 GB/s (GDDR7 128-bit, 28 Gbps)
  NOTE: Earlier estimate used 672 GB/s (RTX 5080 spec) — corrected.
  The weight-size ratio (1.96x) remains the dominant factor.
  PCIe bottleneck eliminated (no -ot CPU offload)

Usage:
  # Run AFTER cycle 5 capture finishes (qwen35-llama service must be stopped)
  python scripts/bench_iq3_quality.py

  # Optional: save raw results as JSON for later analysis
  python scripts/bench_iq3_quality.py --save-json /tmp/iq3_vs_q5_results.json

  # Adjust context size (default 512, enough for 128-token responses + prompt)
  python scripts/bench_iq3_quality.py --ctx-size 1024 --max-tokens 128
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# ── Model paths ────────────────────────────────────────────────────────────────

GGUF_DIR = Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF"
IQ3_MODEL = GGUF_DIR / "Qwen3.5-35B-A3B-UD-IQ3_S.gguf"
Q5_MODEL  = GGUF_DIR / "Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"
LLAMA_SERVER = Path.home() / "llama.cpp/build/bin/llama-server"

# Ports for the two temporary servers (not conflicting with production 8081)
IQ3_PORT = 8092
Q5_PORT  = 8093

# ── Same prompts as benchmark_wallclock.py ─────────────────────────────────────

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


# ── Server lifecycle ───────────────────────────────────────────────────────────

def start_server(
    model_path: Path,
    port: int,
    n_gpu_layers: int = 99,
    extra_args: list[str] | None = None,
    label: str = "",
) -> subprocess.Popen:
    """Launch llama-server as a subprocess. Returns the Popen handle."""
    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "-ngl", str(n_gpu_layers),
        "--flash-attn", "on",
        "--port", str(port),
        "--host", "127.0.0.1",
        "-np", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_label = label or model_path.stem[:30]
    log_path = Path(f"/tmp/bench_iq3_{port}.log")
    print(f"  Launching {log_label} on port {port}...", flush=True)
    print(f"  Log: {log_path}", flush=True)
    print(f"  Cmd: {' '.join(cmd)}", flush=True)

    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
    return proc


def wait_for_server(port: int, timeout: int = 120, proc: subprocess.Popen | None = None) -> bool:
    """Poll /health until the server is ready or timeout."""
    url = f"http://127.0.0.1:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout:
        # Check if process died
        if proc is not None and proc.poll() is not None:
            print(f"  ERROR: server process exited with code {proc.returncode}", flush=True)
            return False
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                elapsed = time.time() - t0
                print(f"  Server ready in {elapsed:.1f}s", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"  ERROR: server not ready after {timeout}s", flush=True)
    return False


def stop_server(proc: subprocess.Popen, label: str = "") -> None:
    """Gracefully stop the server subprocess."""
    if proc.poll() is not None:
        return
    label_str = label or "server"
    print(f"  Stopping {label_str}...", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print(f"  Force-killing {label_str}...", flush=True)
        proc.kill()
        proc.wait()
    # Wait a moment for GPU memory to be released
    time.sleep(3)
    print(f"  {label_str} stopped.", flush=True)


# ── Inference ─────────────────────────────────────────────────────────────────

def generate_greedy(
    port: int,
    prompt: str,
    max_tokens: int = 128,
) -> dict:
    """
    Run greedy generation with logprobs.
    Returns: content, per-token list, generation speed, prompt eval speed.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "bench",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    wall = time.time() - t0

    data = resp.json()
    timings = data.get("timings", {})
    choice = data["choices"][0]
    content = choice["message"]["content"] or ""
    lp_data = choice.get("logprobs", {}) or {}
    token_lp = lp_data.get("content", []) or []

    tokens = []
    for tl in token_lp:
        tokens.append({
            "token": tl.get("token", ""),
            "logprob": tl.get("logprob", 0.0),
            "top5": {t["token"]: t["logprob"] for t in tl.get("top_logprobs", [])},
        })

    return {
        "content": content,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "gen_speed": timings.get("predicted_per_second", 0.0),
        "prompt_speed": timings.get("prompt_per_second", 0.0),
        "wall": wall,
    }


def run_prompts(port: int, label: str, prompts: list[str], max_tokens: int) -> list[dict | None]:
    """Run all prompts against the server, return results list."""
    results: list[dict | None] = []
    for i, prompt in enumerate(prompts):
        short = prompt.replace("\n", " ")[:55]
        print(f"    [{i+1:2d}/{len(prompts)}] {short}...", end=" ", flush=True)
        try:
            r = generate_greedy(port, prompt, max_tokens)
            results.append(r)
            print(f"OK  {r['n_tokens']} tok | gen={r['gen_speed']:.1f} tok/s | "
                  f"pp={r['prompt_speed']:.1f} tok/s", flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            results.append(None)
    return results


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_ppl_proxy(tokens: list[dict]) -> float:
    """Perplexity proxy: exp(-mean(logprob)) over all tokens."""
    if not tokens:
        return float("inf")
    lps = [t["logprob"] for t in tokens if t["logprob"] is not None and math.isfinite(t["logprob"])]
    if not lps:
        return float("inf")
    return math.exp(-sum(lps) / len(lps))


def compare_outputs(iq3_res: dict | None, q5_res: dict | None) -> dict:
    """
    Compare IQ3 and Q5 outputs for a single prompt.
    Returns comparison metrics.
    """
    if iq3_res is None or q5_res is None:
        return {"error": "one or both results missing"}

    iq3_toks = iq3_res["tokens"]
    q5_toks  = q5_res["tokens"]
    n = min(len(iq3_toks), len(q5_toks))

    if n == 0:
        return {"error": "empty token lists"}

    # Token-by-token exact agreement (greedy top-1)
    exact = sum(1 for i in range(n) if iq3_toks[i]["token"] == q5_toks[i]["token"])
    exact_rate = exact / n

    # Is IQ3's token in Q5's top-5? (how well IQ3 predicts Q5's distribution)
    iq3_in_q5_top5 = sum(
        1 for i in range(n)
        if iq3_toks[i]["token"] in q5_toks[i]["top5"]
    ) / n

    # Is Q5's token in IQ3's top-5? (how well Q5 predicts IQ3's distribution)
    q5_in_iq3_top5 = sum(
        1 for i in range(n)
        if q5_toks[i]["token"] in iq3_toks[i]["top5"]
    ) / n

    # First divergence position
    first_div = None
    for i in range(n):
        if iq3_toks[i]["token"] != q5_toks[i]["token"]:
            first_div = {
                "pos": i,
                "iq3_token": iq3_toks[i]["token"],
                "q5_token":  q5_toks[i]["token"],
                "iq3_logprob": iq3_toks[i]["logprob"],
                "q5_logprob":  q5_toks[i]["logprob"],
            }
            break

    # Perplexity proxy for each model
    iq3_ppl = compute_ppl_proxy(iq3_toks)
    q5_ppl  = compute_ppl_proxy(q5_toks)

    # Log-prob delta: how much more/less confident IQ3 is vs Q5
    iq3_mean_lp = sum(t["logprob"] for t in iq3_toks[:n]) / n
    q5_mean_lp  = sum(t["logprob"] for t in q5_toks[:n]) / n

    return {
        "n_compared": n,
        "exact_match": exact_rate,
        "iq3_in_q5_top5": iq3_in_q5_top5,
        "q5_in_iq3_top5": q5_in_iq3_top5,
        "first_divergence": first_div,
        "iq3_ppl_proxy": iq3_ppl,
        "q5_ppl_proxy": q5_ppl,
        "iq3_mean_logprob": iq3_mean_lp,
        "q5_mean_logprob":  q5_mean_lp,
        "logprob_delta": iq3_mean_lp - q5_mean_lp,  # positive = IQ3 more confident
        "iq3_gen_speed": iq3_res["gen_speed"],
        "q5_gen_speed":  q5_res["gen_speed"],
        "iq3_prompt_speed": iq3_res["prompt_speed"],
        "q5_prompt_speed":  q5_res["prompt_speed"],
    }


def print_report(
    prompts: list[str],
    iq3_results: list[dict | None],
    q5_results: list[dict | None],
    comparisons: list[dict],
    max_tokens: int,
) -> None:
    """Print the full comparison report."""
    sep = "=" * 70

    print(f"\n{sep}")
    print(" IQ3_S (full GPU) vs Q5_K_XL (CPU offload) — Quality Comparison")
    print(sep)

    # Per-prompt details
    for i, (prompt, cmp) in enumerate(zip(prompts, comparisons)):
        iq3 = iq3_results[i]
        q5  = q5_results[i]

        short_prompt = prompt.replace("\n", " ")[:55]
        print(f"\n  [{i+1:2d}] {short_prompt}...")

        if "error" in cmp:
            print(f"       ERROR: {cmp['error']}")
            continue

        print(f"       Tokens compared:        {cmp['n_compared']}")
        print(f"       Greedy exact match:      {cmp['exact_match']:.1%}")
        print(f"       IQ3 tok in Q5 top-5:     {cmp['iq3_in_q5_top5']:.1%}  "
              f"(relevance for IQ3→Q5 speculative)")
        print(f"       Q5 tok in IQ3 top-5:     {cmp['q5_in_iq3_top5']:.1%}")
        print(f"       IQ3 PPL proxy:           {cmp['iq3_ppl_proxy']:.3f}")
        print(f"       Q5  PPL proxy:           {cmp['q5_ppl_proxy']:.3f}")
        print(f"       Logprob delta (IQ3-Q5):  {cmp['logprob_delta']:+.3f}  "
              f"({'IQ3 more confident' if cmp['logprob_delta'] > 0 else 'Q5 more confident'})")

        if cmp["first_divergence"] is None:
            print(f"       First divergence:        NONE (perfect agreement)")
        else:
            fd = cmp["first_divergence"]
            print(f"       First divergence at pos {fd['pos']}: "
                  f"IQ3='{fd['iq3_token']}' (lp={fd['iq3_logprob']:.2f}) "
                  f"vs Q5='{fd['q5_token']}' (lp={fd['q5_logprob']:.2f})")

        if iq3 and q5:
            print(f"       Speed: IQ3={iq3['gen_speed']:.1f} tok/s | "
                  f"Q5={q5['gen_speed']:.1f} tok/s | "
                  f"ratio={iq3['gen_speed'] / max(q5['gen_speed'], 0.1):.2f}x")

        # Show a snippet of both outputs for qualitative comparison
        if iq3 and q5 and iq3["content"] and q5["content"]:
            iq3_snip = iq3["content"].replace("\n", " ")[:100]
            q5_snip  = q5["content"].replace("\n", " ")[:100]
            same_prefix = iq3["content"][:30] == q5["content"][:30]
            match_str = "(same start)" if same_prefix else "(different start)"
            print(f"       IQ3 output: {iq3_snip}...")
            print(f"       Q5  output: {q5_snip}... {match_str}")

    # ── Aggregate statistics ───────────────────────────────────────────────────
    valid = [c for c in comparisons if "error" not in c]
    if not valid:
        print("\n  No valid comparisons to aggregate.")
        return

    total_toks = sum(c["n_compared"] for c in valid)
    total_exact = sum(c["exact_match"] * c["n_compared"] for c in valid)
    total_iq3_in_q5 = sum(c["iq3_in_q5_top5"] * c["n_compared"] for c in valid)
    total_q5_in_iq3 = sum(c["q5_in_iq3_top5"] * c["n_compared"] for c in valid)
    avg_iq3_ppl = sum(c["iq3_ppl_proxy"] for c in valid) / len(valid)
    avg_q5_ppl  = sum(c["q5_ppl_proxy"] for c in valid) / len(valid)
    avg_lp_delta = sum(c["logprob_delta"] for c in valid) / len(valid)
    avg_iq3_speed = sum(c["iq3_gen_speed"] for c in valid) / len(valid)
    avg_q5_speed  = sum(c["q5_gen_speed"] for c in valid) / len(valid)
    avg_iq3_pp = sum(c["iq3_prompt_speed"] for c in valid) / len(valid)
    avg_q5_pp  = sum(c["q5_prompt_speed"] for c in valid) / len(valid)

    overall_exact = total_exact / total_toks if total_toks else 0
    overall_iq3_in_q5 = total_iq3_in_q5 / total_toks if total_toks else 0
    overall_q5_in_iq3 = total_q5_in_iq3 / total_toks if total_toks else 0

    print(f"\n{sep}")
    print(f" OVERALL SUMMARY  ({total_toks} tokens across {len(valid)} prompts)")
    print(sep)

    print(f"\n  TOKEN AGREEMENT")
    print(f"    Greedy exact match:             {overall_exact:.1%}")
    print(f"    IQ3 tok in Q5 top-5:            {overall_iq3_in_q5:.1%}")
    print(f"    Q5 tok in IQ3 top-5:            {overall_q5_in_iq3:.1%}")

    print(f"\n  PERPLEXITY PROXY (lower = more confident)")
    print(f"    IQ3_S PPL proxy:                {avg_iq3_ppl:.3f}")
    print(f"    Q5_K_XL PPL proxy:              {avg_q5_ppl:.3f}")
    print(f"    Log-prob delta (IQ3 - Q5):      {avg_lp_delta:+.3f}  "
          f"({'IQ3 more confident on average' if avg_lp_delta > 0 else 'Q5 more confident on average'})")
    ppl_gap_pct = (avg_iq3_ppl / avg_q5_ppl - 1) * 100
    print(f"    PPL gap:                        {ppl_gap_pct:+.1f}%  "
          f"({'IQ3 higher uncertainty' if ppl_gap_pct > 0 else 'IQ3 lower uncertainty'})")

    print(f"\n  GENERATION SPEED")
    print(f"    IQ3_S (full GPU):               {avg_iq3_speed:.1f} tok/s gen, {avg_iq3_pp:.1f} tok/s pp")
    print(f"    Q5_K_XL (CPU offload):          {avg_q5_speed:.1f} tok/s gen, {avg_q5_pp:.1f} tok/s pp")
    if avg_q5_speed > 0:
        speed_ratio = avg_iq3_speed / avg_q5_speed
        print(f"    Speed ratio (IQ3/Q5):           {speed_ratio:.2f}x")
    if avg_q5_pp > 0:
        pp_ratio = avg_iq3_pp / avg_q5_pp
        print(f"    Prompt eval ratio (IQ3/Q5):     {pp_ratio:.2f}x")

    print(f"\n  SPECULATIVE DECODING IMPLICATIONS")
    print(f"    If IQ3 drafts for Q5 (production target):")
    print(f"      Greedy acceptance rate: ~{overall_exact:.0%}")
    print(f"      Top-5 acceptance rate:  ~{overall_iq3_in_q5:.0%}")
    # Expected tokens per block for block_size=16
    tau = overall_exact
    for K in [4, 8, 15, 16]:
        if tau < 1.0:
            expected = (1.0 - tau**K) / (1.0 - tau)
        else:
            expected = float(K)
        print(f"      K={K:2d} → E[accepted] = {expected:.2f} tokens/block")

    print(f"\n  VRAM BUDGET SUMMARY")
    print(f"    IQ3_S full GPU:   ~13.06 GB (12.7 + 0.07 KV + 0.3 compute)")
    print(f"    Q5 CPU offload:   ~14.0 GB  (13.9 measured + 0.07 KV)")
    print(f"    DFlash drafter margin: IQ3={16.0-13.06:.2f} GB  Q5={16.0-14.0:.2f} GB")

    print(f"\n{sep}")

    # Quality verdict
    if overall_exact >= 0.90:
        verdict = "EXCELLENT: IQ3_S and Q5_K_XL are nearly identical (>= 90% agreement)"
    elif overall_exact >= 0.80:
        verdict = "GOOD: IQ3_S agrees with Q5_K_XL >= 80% of tokens — viable as drafter"
    elif overall_exact >= 0.65:
        verdict = "MODERATE: ~65-80% agreement — IQ3 viable as drafter but quality degraded"
    else:
        verdict = f"POOR: Only {overall_exact:.0%} agreement — IQ3_S diverges significantly from Q5"

    print(f"\n  VERDICT: {verdict}")

    if avg_iq3_ppl > avg_q5_ppl * 1.15:
        print(f"  CAUTION: IQ3_S PPL is {ppl_gap_pct:.0f}% higher — model is less certain,")
        print(f"           which may affect quality on complex reasoning tasks.")
    elif avg_iq3_ppl < avg_q5_ppl * 0.85:
        print(f"  NOTE: IQ3_S PPL is {abs(ppl_gap_pct):.0f}% LOWER — unusually more confident,")
        print(f"        possibly due to quantization distribution shift.")
    else:
        print(f"  PPL gap is within ±15% — IQ3_S confidence is comparable to Q5_K_XL.")

    print(f"{sep}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IQ3_S vs Q5_K_XL quality benchmark")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max tokens to generate per prompt (default: 128)")
    parser.add_argument("--ctx-size", type=int, default=512,
                        help="KV cache context size for both servers (default: 512)")
    parser.add_argument("--save-json", type=str, default="",
                        help="Save raw results as JSON to this path")
    parser.add_argument("--skip-q5", action="store_true",
                        help="Only benchmark IQ3_S (skip Q5 phase)")
    parser.add_argument("--threads", type=int, default=6,
                        help="CPU threads for llama-server (default: 6)")
    parser.add_argument("--iq3-model", type=str, default=str(IQ3_MODEL),
                        help="Path to IQ3_S GGUF")
    parser.add_argument("--q5-model", type=str, default=str(Q5_MODEL),
                        help="Path to Q5_K_XL GGUF")
    args = parser.parse_args()

    iq3_path = Path(args.iq3_model)
    q5_path  = Path(args.q5_model)

    # Pre-flight checks
    print("=" * 70)
    print(" bench_iq3_quality.py — IQ3_S vs Q5_K_XL Quality Comparison")
    print("=" * 70)
    print()

    if not iq3_path.exists():
        print(f"ERROR: IQ3_S model not found: {iq3_path}")
        sys.exit(1)
    if not q5_path.exists() and not args.skip_q5:
        print(f"ERROR: Q5_K_XL model not found: {q5_path}")
        sys.exit(1)
    if not LLAMA_SERVER.exists():
        print(f"ERROR: llama-server not found: {LLAMA_SERVER}")
        sys.exit(1)

    # Check if production service is running — warn before starting
    svc_check = subprocess.run(
        ["systemctl", "--user", "is-active", "qwen35-llama"],
        capture_output=True, text=True
    )
    if svc_check.stdout.strip() == "active":
        print("WARNING: qwen35-llama systemd service is active!")
        print("         This benchmark uses ports 8092/8093 (not 8081).")
        print("         If VRAM is tight, stop the service first:")
        print("           systemctl --user stop qwen35-llama")
        print()
        ans = input("Continue anyway? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    print(f"  IQ3_S model:  {iq3_path} ({iq3_path.stat().st_size / 1024**3:.1f} GB)")
    if not args.skip_q5:
        print(f"  Q5_K_XL model:{q5_path} ({q5_path.stat().st_size / 1024**3:.1f} GB)")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"  Ctx size:     {args.ctx_size}")
    print(f"  Prompts:      {len(BENCH_PROMPTS)}")
    print()

    iq3_results: list[dict | None] = []
    q5_results:  list[dict | None] = []

    # ── Phase 1: IQ3_S (full GPU, no CPU offload) ─────────────────────────────
    print(f"{'─'*70}")
    print(f" PHASE 1: IQ3_S — Full GPU (no -ot offload)")
    print(f"{'─'*70}")
    print()
    print("  VRAM estimate: ~13.06 GB (12.7 GB weights + 70 MB KV + 300 MB compute)")
    print()

    iq3_proc = start_server(
        model_path=iq3_path,
        port=IQ3_PORT,
        n_gpu_layers=99,   # ALL layers on GPU — no CPU offload
        extra_args=[
            "-c", str(args.ctx_size),
            "-t", str(args.threads),
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q4_0",
            # NOTE: no -ot flag → all layers including MoE experts on GPU
        ],
        label="IQ3_S full-GPU",
    )

    try:
        print()
        if not wait_for_server(IQ3_PORT, timeout=120, proc=iq3_proc):
            print("ERROR: IQ3_S server failed to start. Check /tmp/bench_iq3_8092.log")
            iq3_proc.kill()
            sys.exit(1)

        print()
        print("  Running prompts on IQ3_S...")
        iq3_results = run_prompts(IQ3_PORT, "IQ3_S", BENCH_PROMPTS, args.max_tokens)
    finally:
        stop_server(iq3_proc, "IQ3_S")

    if args.skip_q5:
        print("\n  --skip-q5 specified, skipping Q5 phase.")
        # Print partial report
        print("\n  IQ3_S results:")
        for i, (p, r) in enumerate(zip(BENCH_PROMPTS, iq3_results)):
            if r:
                print(f"    [{i+1}] {r['n_tokens']} tok | "
                      f"gen={r['gen_speed']:.1f} tok/s | "
                      f"pp={r['prompt_speed']:.1f} tok/s | "
                      f"PPL={compute_ppl_proxy(r['tokens']):.3f}")
        sys.exit(0)

    # ── Phase 2: Q5_K_XL (CPU offload, current production config) ─────────────
    print(f"\n{'─'*70}")
    print(f" PHASE 2: Q5_K_XL — CPU offload (production config)")
    print(f"{'─'*70}")
    print()
    print("  VRAM estimate: ~14.0 GB (layers 0-19 GPU, layers 20-39 experts CPU)")
    print()

    q5_proc = start_server(
        model_path=q5_path,
        port=Q5_PORT,
        n_gpu_layers=99,
        extra_args=[
            "-c", str(args.ctx_size),
            "-t", str(args.threads),
            "-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q4_0",
        ],
        label="Q5_K_XL CPU-offload",
    )

    try:
        print()
        if not wait_for_server(Q5_PORT, timeout=180, proc=q5_proc):
            print("ERROR: Q5_K_XL server failed to start. Check /tmp/bench_iq3_8093.log")
            q5_proc.kill()
            sys.exit(1)

        print()
        print("  Running prompts on Q5_K_XL...")
        q5_results = run_prompts(Q5_PORT, "Q5_K_XL", BENCH_PROMPTS, args.max_tokens)
    finally:
        stop_server(q5_proc, "Q5_K_XL")

    # ── Phase 3: Compare ───────────────────────────────────────────────────────
    comparisons = [
        compare_outputs(iq3, q5)
        for iq3, q5 in zip(iq3_results, q5_results)
    ]

    print_report(BENCH_PROMPTS, iq3_results, q5_results, comparisons, args.max_tokens)

    # ── Optional JSON save ─────────────────────────────────────────────────────
    if args.save_json:
        out = {
            "metadata": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "max_tokens": args.max_tokens,
                "ctx_size": args.ctx_size,
                "iq3_model": str(iq3_path),
                "q5_model": str(q5_path),
            },
            "prompts": BENCH_PROMPTS,
            "iq3_results": iq3_results,
            "q5_results": q5_results,
            "comparisons": comparisons,
        }
        save_path = Path(args.save_json)
        save_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  Raw results saved to: {save_path}")


if __name__ == "__main__":
    main()
