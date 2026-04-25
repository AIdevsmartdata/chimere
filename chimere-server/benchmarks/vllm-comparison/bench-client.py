"""OpenAI-compatible streaming bench client for chimere-server vs llama-server vs vLLM.

Measures TTFT (time to first content token), total wall time, gen tokens/s.
Aggregates across M concurrent requests using ThreadPoolExecutor.
Appends one CSV row per (engine, workload, M, replica).
"""
import argparse, csv, json, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


def stream_one(base_url, prompt, max_tokens):
    """Returns (ttft_ms, total_ms, n_prompt, n_gen, error_str_or_None)."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": "chimere",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    t0 = time.time()
    ttft = None
    n_gen = 0
    try:
        r = requests.post(url, json=payload, stream=True, timeout=300)
        r.encoding = "utf-8"
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                break
            try:
                d = json.loads(chunk)
            except Exception:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or delta.get("reasoning_content")
            if content:
                if ttft is None:
                    ttft = (time.time() - t0) * 1000
                n_gen += 1
        total_ms = (time.time() - t0) * 1000
        return ttft, total_ms, len(prompt.split()), n_gen, None
    except Exception as e:
        return None, (time.time() - t0) * 1000, 0, 0, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--replicas", type=int, default=1)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--vram-log", default="")
    args = ap.parse_args()

    # Run M parallel requests, repeated R times sequentially (cold + warm reps).
    all_results = []
    for r_idx in range(1, args.replicas + 1):
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(stream_one, args.base_url, args.prompt, args.max_tokens)
                    for _ in range(args.concurrency)]
            results = [f.result() for f in as_completed(futs)]
        # Aggregate this replica
        ttfts = [r[0] for r in results if r[0] is not None]
        totals = [r[1] for r in results if r[1] is not None]
        n_gens = [r[3] for r in results]
        errors = [r[4] for r in results if r[4]]
        ttft_p50 = statistics.median(ttfts) if ttfts else 0
        total_p50 = statistics.median(totals) if totals else 0
        n_gen_total = sum(n_gens)
        gen_tokps = (n_gen_total * 1000.0) / total_p50 if total_p50 else 0
        prefill_tokps = (len(args.prompt.split()) * 1000.0) / ttft_p50 if ttft_p50 else 0
        err_str = ";".join(errors)[:200]

        # VRAM peak from log
        vram_peak = 0
        if args.vram_log:
            try:
                with open(args.vram_log) as f:
                    vals = [int(line.strip()) for line in f if line.strip().isdigit()]
                vram_peak = max(vals) if vals else 0
            except Exception:
                pass

        with open(args.output_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"),
                args.engine, "Qwen3.6-35B-A3B", args.workload,
                args.concurrency, r_idx, len(args.prompt.split()), n_gen_total,
                f"{ttft_p50:.1f}", f"{total_p50:.1f}",
                f"{gen_tokps:.2f}", f"{prefill_tokps:.2f}",
                vram_peak, err_str,
            ])
        print(f"[{args.engine}|{args.workload}|M={args.concurrency}|R={r_idx}] "
              f"TTFT={ttft_p50:.0f}ms total={total_p50:.0f}ms gen={gen_tokps:.1f}tok/s "
              f"vram={vram_peak}MB errs={len(errors)}")

if __name__ == "__main__":
    main()
