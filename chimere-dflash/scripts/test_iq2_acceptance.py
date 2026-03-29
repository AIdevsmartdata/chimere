#!/usr/bin/env python3
"""TEST 2: Measure token-by-token acceptance rate between IQ2_XXS and MXFP4.

Phase 1: Generate responses from MXFP4 (port 8081), save tokens + logprobs
Phase 2: Score same sequences with IQ2 (port 8090), compare top-1 agreement
"""

import json
import requests
import time
import sys
import os
import subprocess
import signal

MXFP4_URL = "http://127.0.0.1:8081"
IQ2_URL = "http://127.0.0.1:8090"

PROMPTS = [
    "Explain how a neural network learns through backpropagation.",
    "What are the main differences between Python and Rust?",
    "Describe the pathophysiology of type 2 diabetes mellitus.",
    "Write a function to find the longest common subsequence of two strings.",
    "Explain the concept of entropy in information theory.",
    "What is the role of the hippocampus in memory formation?",
    "Compare REST and GraphQL API architectures.",
    "Explain how transformers use attention mechanisms.",
    "What causes aurora borealis?",
    "Describe the process of mRNA translation in ribosomes.",
]

MAX_TOKENS = 128  # per response


def generate_greedy(base_url, prompt, max_tokens=MAX_TOKENS):
    """Generate a response greedily, return tokens and logprobs."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 5,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    timings = data.get("timings", {})

    choice = data["choices"][0]
    content = choice["message"]["content"]
    logprobs_data = choice.get("logprobs", {})
    token_logprobs = logprobs_data.get("content", [])

    tokens = []
    for tl in token_logprobs:
        tokens.append({
            "token": tl["token"],
            "logprob": tl["logprob"],
            "top5": {t["token"]: t["logprob"] for t in tl.get("top_logprobs", [])},
        })

    return {
        "content": content,
        "tokens": tokens,
        "n_tokens": len(tokens),
        "gen_speed": timings.get("predicted_per_second", 0),
        "prompt_speed": timings.get("prompt_per_second", 0),
    }


def score_prefix(base_url, prompt, target_text, max_tokens=1):
    """Feed prompt + target_text, get logprobs for the target tokens.

    Uses completion endpoint to score existing text.
    """
    # Use chat completion with the target as assistant prefill
    # We generate 1 token but we care about the logprobs of the prefix
    # Actually, we need a different approach: feed the full text and get per-token logprobs

    # Better approach: use /completion endpoint with prompt
    # But we need the tokenized chat template...

    # Simplest: generate the same response from IQ2 greedily and compare tokens
    return generate_greedy(base_url, prompt, max_tokens=MAX_TOKENS)


def compute_acceptance(mxfp4_tokens, iq2_tokens):
    """Compute speculative decoding acceptance rate.

    For greedy decoding: acceptance = fraction where top-1 matches.
    Also compute: if IQ2 drafts, how many would MXFP4 accept (and vice versa).
    """
    n = min(len(mxfp4_tokens), len(iq2_tokens))
    if n == 0:
        return {"n": 0, "exact_match": 0, "iq2_in_mxfp4_top5": 0, "mxfp4_in_iq2_top5": 0}

    exact = 0
    iq2_in_mxfp4_top5 = 0
    mxfp4_in_iq2_top5 = 0

    for i in range(n):
        mt = mxfp4_tokens[i]["token"]
        it = iq2_tokens[i]["token"]

        if mt == it:
            exact += 1

        # Is IQ2's token in MXFP4's top-5?
        if it in mxfp4_tokens[i]["top5"]:
            iq2_in_mxfp4_top5 += 1

        # Is MXFP4's token in IQ2's top-5?
        if mt in iq2_tokens[i]["top5"]:
            mxfp4_in_iq2_top5 += 1

    return {
        "n": n,
        "exact_match": exact / n,
        "iq2_in_mxfp4_top5": iq2_in_mxfp4_top5 / n,
        "mxfp4_in_iq2_top5": mxfp4_in_iq2_top5 / n,
    }


def wait_for_server(url, timeout=60):
    """Wait for server to be ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    print("=" * 60)
    print(" TEST 2: IQ2_XXS vs MXFP4 — Token Acceptance Rate")
    print("=" * 60)

    # ── Phase 1: Generate from MXFP4 ──
    print("\n[Phase 1] Generating responses from MXFP4...")

    if not wait_for_server(MXFP4_URL, timeout=10):
        print("ERROR: MXFP4 server not available at", MXFP4_URL)
        sys.exit(1)

    mxfp4_results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:60]}...", end=" ", flush=True)
        try:
            result = generate_greedy(MXFP4_URL, prompt)
            mxfp4_results.append(result)
            print(f"✓ {result['n_tokens']} tok @ {result['gen_speed']:.1f} tok/s")
        except Exception as e:
            print(f"✗ {e}")
            mxfp4_results.append(None)

    # Save MXFP4 results
    with open("/tmp/mxfp4_results.json", "w") as f:
        json.dump(mxfp4_results, f)

    # ── Switch models ──
    print("\n[Switch] Stopping MXFP4, starting IQ2_XXS...")
    subprocess.run(["systemctl", "--user", "stop", "qwen35-llama"], check=True)
    time.sleep(3)

    # Start IQ2
    iq2_model = os.path.expanduser(
        "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
    )
    iq2_proc = subprocess.Popen(
        [
            os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
            "-m", iq2_model,
            "--flash-attn", "on",
            "-ngl", "99",
            "-c", "4096",
            "--port", "8090",
            "-np", "1",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q4_0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("  Waiting for IQ2 server...", end=" ", flush=True)
    if not wait_for_server(IQ2_URL, timeout=60):
        print("FAILED")
        iq2_proc.kill()
        subprocess.run(["systemctl", "--user", "start", "qwen35-llama"])
        sys.exit(1)
    print("ready!")

    # ── Phase 2: Generate same prompts from IQ2 ──
    print("\n[Phase 2] Generating responses from IQ2_XXS...")

    iq2_results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"  [{i+1}/{len(PROMPTS)}] {prompt[:60]}...", end=" ", flush=True)
        try:
            result = generate_greedy(IQ2_URL, prompt)
            iq2_results.append(result)
            print(f"✓ {result['n_tokens']} tok @ {result['gen_speed']:.1f} tok/s")
        except Exception as e:
            print(f"✗ {e}")
            iq2_results.append(None)

    # ── Cleanup: stop IQ2, restart MXFP4 ──
    print("\n[Cleanup] Stopping IQ2, restarting MXFP4...")
    iq2_proc.terminate()
    iq2_proc.wait(timeout=10)
    time.sleep(2)
    subprocess.run(["systemctl", "--user", "start", "qwen35-llama"], check=True)

    # ── Phase 3: Compare ──
    print("\n" + "=" * 60)
    print(" Results")
    print("=" * 60)

    total_exact = 0
    total_iq2_in_top5 = 0
    total_mxfp4_in_top5 = 0
    total_tokens = 0

    for i, (prompt, mr, ir) in enumerate(zip(PROMPTS, mxfp4_results, iq2_results)):
        if mr is None or ir is None:
            continue

        acc = compute_acceptance(mr["tokens"], ir["tokens"])
        total_exact += acc["exact_match"] * acc["n"]
        total_iq2_in_top5 += acc["iq2_in_mxfp4_top5"] * acc["n"]
        total_mxfp4_in_top5 += acc["mxfp4_in_iq2_top5"] * acc["n"]
        total_tokens += acc["n"]

        print(f"\n  Prompt {i+1}: {prompt[:50]}...")
        print(f"    Tokens compared: {acc['n']}")
        print(f"    Exact match (greedy):     {acc['exact_match']:.1%}")
        print(f"    IQ2 tok in MXFP4 top-5:   {acc['iq2_in_mxfp4_top5']:.1%}")
        print(f"    MXFP4 tok in IQ2 top-5:   {acc['mxfp4_in_iq2_top5']:.1%}")

        # Show first divergence
        n = min(len(mr["tokens"]), len(ir["tokens"]))
        for j in range(n):
            if mr["tokens"][j]["token"] != ir["tokens"][j]["token"]:
                print(f"    First divergence at pos {j}: "
                      f"MXFP4='{mr['tokens'][j]['token']}' vs IQ2='{ir['tokens'][j]['token']}'")
                break

    if total_tokens > 0:
        print(f"\n{'=' * 60}")
        print(f" OVERALL ({total_tokens} tokens across {len(PROMPTS)} prompts)")
        print(f"{'=' * 60}")
        print(f"  Greedy exact match:        {total_exact / total_tokens:.1%}")
        print(f"  IQ2 tok in MXFP4 top-5:    {total_iq2_in_top5 / total_tokens:.1%}")
        print(f"  MXFP4 tok in IQ2 top-5:    {total_mxfp4_in_top5 / total_tokens:.1%}")
        print()
        print(f"  → If IQ2 drafts for MXFP4:  ~{total_exact / total_tokens:.0%} acceptance (greedy)")
        print(f"  → If MXFP4 drafts for IQ2:  ~{total_exact / total_tokens:.0%} acceptance (greedy)")
        print(f"  → With top-5 sampling:       ~{total_iq2_in_top5 / total_tokens:.0%} acceptance")

        # Speed implications
        iq2_speed = sum(r["gen_speed"] for r in iq2_results if r) / sum(1 for r in iq2_results if r)
        mxfp4_speed = sum(r["gen_speed"] for r in mxfp4_results if r) / sum(1 for r in mxfp4_results if r)

        acc_rate = total_exact / total_tokens
        # Spec decode effective speed: block_size * acc_rate * verifier_speed
        # (simplified, actual formula is more complex)
        print(f"\n  Speed comparison:")
        print(f"    MXFP4 standalone:  {mxfp4_speed:.1f} tok/s")
        print(f"    IQ2 standalone:    {iq2_speed:.1f} tok/s")

        # If IQ2 drafts K tokens, MXFP4 verifies:
        # Expected accepted = sum_{i=0}^{K-1} acc^i = (1 - acc^K) / (1 - acc)
        # Time = K * t_iq2 + t_mxfp4_verify(K)
        for K in [4, 8, 16]:
            expected = (1 - acc_rate**K) / (1 - acc_rate) if acc_rate < 1 else K
            draft_time = K / iq2_speed  # seconds
            verify_time = K / 372  # prompt eval speed from earlier test
            total_time = draft_time + verify_time
            effective = expected / total_time
            print(f"    IQ2→MXFP4 spec (K={K}): ~{effective:.1f} tok/s "
                  f"(E[accepted]={expected:.1f}, draft={draft_time*1000:.0f}ms, "
                  f"verify={verify_time*1000:.0f}ms)")


if __name__ == "__main__":
    main()
