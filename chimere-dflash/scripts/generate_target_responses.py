#!/usr/bin/env python3
"""
generate_target_responses.py — Generate responses using Qwen3.5 for target-generated training data.

Sends prompts to llama-server and saves {prompt, response, id} as JSONL.
The output can be fed directly to extract_single_position for hidden state extraction.

Usage:
  # Start llama-server first: systemctl --user start qwen35-llama
  python scripts/generate_target_responses.py \
    --input data/prompts_v6/combined_150k.jsonl \
    --output data/prompts_v6/target_generated.jsonl \
    --max-tokens 256 \
    --workers 2 \
    --limit 50000
"""

import argparse
import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from threading import Event

import requests

# Graceful shutdown
_shutdown = Event()


def _signal_handler(sig, frame):
    print(f"\n  Signal {sig} received, finishing current batch...")
    _shutdown.set()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_prompts(input_path: str, limit: int = 0) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt") or obj.get("text", "")
            if not prompt:
                continue

            prompts.append({
                "prompt": prompt,
                "id": obj.get("id", f"prompt_{len(prompts)}"),
            })

            if limit and len(prompts) >= limit:
                break

    return prompts


def load_done_ids(output_path: str) -> set:
    """Load already-generated IDs for resume support."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("id"):
                    done.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return done


def generate_one(prompt: str, prompt_id: str, base_url: str,
                 max_tokens: int, temperature: float,
                 top_p: float, top_k: int) -> dict | None:
    """Send a single prompt to llama-server and return the response."""
    if _shutdown.is_set():
        return None

    payload = {
        "model": "qwen3.5-35b",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  ERROR [{prompt_id}]: {e}", flush=True)
        return None

    choices = data.get("choices", [])
    if not choices:
        print(f"  ERROR [{prompt_id}]: no choices in response", flush=True)
        return None

    content = choices[0].get("message", {}).get("content", "")
    if not content:
        print(f"  WARN [{prompt_id}]: empty response", flush=True)
        return None

    # Strip any leaked thinking tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    if not content:
        print(f"  WARN [{prompt_id}]: empty after stripping think tags", flush=True)
        return None

    usage = data.get("usage", {})
    timings = data.get("timings", {})

    return {
        "prompt": prompt,
        "response": content,
        "id": prompt_id,
        "meta": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "gen_speed": round(timings.get("predicted_per_second", 0), 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate target responses with Qwen3.5")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with prompts")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with responses")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8081",
                        help="llama-server base URL")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k sampling")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent requests (match llama-server -np)")
    parser.add_argument("--limit", type=int, default=0, help="Max prompts to process (0=all)")
    parser.add_argument("--log-every", type=int, default=50, help="Log progress every N prompts")
    args = parser.parse_args()

    # Load prompts
    print(f"Loading prompts from {args.input}...", flush=True)
    prompts = load_prompts(args.input, args.limit)
    print(f"  Loaded {len(prompts)} prompts", flush=True)

    # Resume support
    done_ids = load_done_ids(args.output)
    remaining = [p for p in prompts if p["id"] not in done_ids]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done, {len(remaining)} remaining", flush=True)

    if not remaining:
        print("  Nothing to do!", flush=True)
        return

    # Check server health
    try:
        r = requests.get(f"{args.base_url}/health", timeout=5)
        if r.status_code != 200:
            print(f"ERROR: llama-server not healthy (status={r.status_code})", flush=True)
            sys.exit(1)
    except (requests.ConnectionError, requests.Timeout):
        print(f"ERROR: Cannot connect to llama-server at {args.base_url}", flush=True)
        print("  Start it with: systemctl --user start qwen35-llama", flush=True)
        sys.exit(1)

    print(f"\nGenerating {len(remaining)} responses...", flush=True)
    print(f"  Server:      {args.base_url}", flush=True)
    print(f"  Workers:     {args.workers}", flush=True)
    print(f"  Max tokens:  {args.max_tokens}", flush=True)
    print(f"  Temperature: {args.temperature}", flush=True)
    print(f"  Output:      {args.output}\n", flush=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    n_done = len(done_ids)
    n_total = len(prompts)
    n_failed = 0
    t_start = time.time()

    with open(args.output, "a", encoding="utf-8") as fout:
        if args.workers <= 1:
            # Sequential mode
            for i, p in enumerate(remaining):
                if _shutdown.is_set():
                    print(f"\n  Shutdown requested. Saved {n_done}/{n_total}", flush=True)
                    break

                result = generate_one(
                    p["prompt"], p["id"], args.base_url,
                    args.max_tokens, args.temperature, args.top_p, args.top_k,
                )

                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_done += 1
                else:
                    n_failed += 1

                if (i + 1) % args.log_every == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    eta_h = (len(remaining) - i - 1) / rate / 3600 if rate > 0 else 0
                    print(f"  [{n_done}/{n_total}] {rate:.2f} prompts/s | "
                          f"failed={n_failed} | ETA={eta_h:.1f}h", flush=True)
        else:
            # Parallel mode
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {}
                batch_size = args.workers * 2  # submit ahead
                idx = 0

                def submit_batch():
                    nonlocal idx
                    while idx < len(remaining) and len(futures) < batch_size:
                        if _shutdown.is_set():
                            break
                        p = remaining[idx]
                        fut = executor.submit(
                            generate_one,
                            p["prompt"], p["id"], args.base_url,
                            args.max_tokens, args.temperature, args.top_p, args.top_k,
                        )
                        futures[fut] = idx
                        idx += 1

                submit_batch()

                completed = 0
                while futures:
                    if _shutdown.is_set():
                        print(f"\n  Shutdown requested. Saved {n_done}/{n_total}", flush=True)
                        break

                    try:
                        for fut in as_completed(list(futures), timeout=300):
                            result = fut.result()
                            del futures[fut]

                            if result:
                                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                                fout.flush()
                                n_done += 1
                            else:
                                n_failed += 1

                            completed += 1
                            if completed % args.log_every == 0:
                                elapsed = time.time() - t_start
                                rate = completed / elapsed if elapsed > 0 else 0
                                eta_h = (len(remaining) - completed) / rate / 3600 if rate > 0 else 0
                                print(f"  [{n_done}/{n_total}] {rate:.2f} prompts/s | "
                                      f"failed={n_failed} | ETA={eta_h:.1f}h", flush=True)

                            submit_batch()
                            break  # re-enter while to check shutdown and refill
                    except TimeoutError:
                        print(f"  WARN: timeout waiting for response, "
                              f"{len(futures)} pending", flush=True)

    elapsed = time.time() - t_start
    rate = (n_done - len(done_ids)) / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}", flush=True)
    print(f"  Generation complete!", flush=True)
    print(f"  Total generated: {n_done}/{n_total}", flush=True)
    print(f"  Failed:          {n_failed}", flush=True)
    print(f"  Time:            {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)
    print(f"  Rate:            {rate:.2f} prompts/s", flush=True)
    print(f"  Output:          {args.output}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
