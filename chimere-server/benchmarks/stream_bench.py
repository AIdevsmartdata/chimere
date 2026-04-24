#!/usr/bin/env python3
"""
stream_bench.py — Streaming E2E bench driver for chimere-server.

Drives N concurrent /v1/chat/completions streams against BENCH_URL with
per-request TTFT, per-token inter-arrival timing, and aggregate stats.
Writes JSON-lines output + a final summary JSON to --out.

Why not bench-m1? The existing Rust bench uses non-streaming which chimere-server
refuses in native multislot mode (CHIMERE_MULTISLOT_NATIVE=1 → 400 on stream=false).

Safety rail: refuses to hit :8081 unless --explicit-prod is set.
"""
import argparse
import json
import os
import ssl
import sys
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROMPTS_SHORT = [
    "Write one short sentence about the Eiffel Tower.",
    "Write one short sentence about Mount Fuji.",
    "Write one short sentence about the Amazon river.",
    "Write one short sentence about the Sahara desert.",
    "Write one short sentence about the Great Wall.",
    "Write one short sentence about the Great Barrier Reef.",
    "Write one short sentence about the Niagara Falls.",
    "Write one short sentence about Machu Picchu.",
]

PROMPTS_LONG = [
    "Explain in three paragraphs the difference between recurrent state space models "
    "like Mamba-2 and transformer attention. Cover at least: (1) computational complexity "
    "in the sequence length, (2) how each caches state during autoregressive generation, "
    "(3) which one empirically wins on retrieval-heavy downstream tasks as of 2026.",
    "Write a thorough review in three paragraphs of the tradeoffs between 3-bit and 4-bit "
    "quantization for large language models. Be specific: discuss per-tensor vs per-channel "
    "scales, outlier handling, and the role of imatrix importance matrices. Give concrete "
    "perplexity numbers if you remember any from the literature.",
    "Draft a detailed three-paragraph postmortem template suitable for a production LLM "
    "serving outage. The template should cover: timeline reconstruction, root cause analysis, "
    "user impact quantification, mitigation, and a follow-up action checklist. Make it "
    "usable as-is, not generic.",
    "Summarize in three paragraphs what 'hybrid' architectures like Qwen3.5-A3B actually do. "
    "Be precise about: the ratio of GDN/SSM layers to full attention layers, the purpose "
    "of the attention layers (global context vs local recall), and why this tradeoff is "
    "appealing for inference-time cost at long context.",
]


def parse_sse_line(line):
    """Return dict or None. Handles 'data: {...}' and 'data: [DONE]'."""
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return {"_done": payload == "[DONE]"}
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def extract_delta_content(evt):
    """Extract text from a chat.completion.chunk. Handles both `content`
    and `reasoning_content` (Qwen3 think blocks)."""
    if not evt or "_done" in evt:
        return None
    try:
        delta = evt["choices"][0].get("delta", {})
    except (KeyError, IndexError):
        return None
    return delta.get("content") or delta.get("reasoning_content")


def finish_reason(evt):
    if not evt or "_done" in evt:
        return None
    try:
        return evt["choices"][0].get("finish_reason")
    except (KeyError, IndexError):
        return None


def one_request(url, prompt, max_tokens, temperature, request_id, extra_body=None):
    """Drive one streaming request. Returns observation dict.
    Measures:
        - send_at: wall time when POST issued
        - ttft_ms: time to first delta chunk (content or reasoning_content)
        - inter_tok_ms: list of gap between consecutive tokens
        - total_ms: wall time from send to [DONE]
        - n_tokens: number of delta events with non-empty content
        - finish: finish_reason
        - error: str or None
        - body_preview: first 200 chars of assembled text
    """
    body = {
        "model": "chimere-deltanet",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if extra_body:
        body.update(extra_body)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    t0 = time.perf_counter()
    ttft_ms = None
    inter_tok_ms = []
    assembled = []
    last_tok_t = None
    finish = None
    n_tokens = 0
    err = None

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            if resp.status != 200:
                err = f"http_{resp.status}"
                return _make_obs(request_id, prompt, t0, None, [], [], 0, None, err)
            for raw in resp:
                # Each SSE chunk may contain multiple data: lines. urlopen
                # iterates lines for us (bytes). Strip trailing CRLF.
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                evt = parse_sse_line(line)
                if evt is None:
                    continue
                if evt.get("_done"):
                    break
                delta = extract_delta_content(evt)
                if delta:
                    now = time.perf_counter()
                    if ttft_ms is None:
                        ttft_ms = (now - t0) * 1000.0
                    else:
                        inter_tok_ms.append((now - last_tok_t) * 1000.0)
                    last_tok_t = now
                    assembled.append(delta)
                    n_tokens += 1
                fr = finish_reason(evt)
                if fr is not None:
                    finish = fr
    except urllib.error.HTTPError as e:
        err = f"httperror_{e.code}: {e.read()[:200].decode(errors='replace')}"
    except urllib.error.URLError as e:
        err = f"urlerror: {e.reason}"
    except (TimeoutError, ConnectionError) as e:
        err = f"timeout: {e}"
    except Exception as e:  # noqa: BLE001 — bench must not crash on one bad req
        err = f"{type(e).__name__}: {e}"

    total_ms = (time.perf_counter() - t0) * 1000.0
    return _make_obs(
        request_id, prompt, t0, ttft_ms, inter_tok_ms, assembled,
        n_tokens, finish, err, total_ms=total_ms,
    )


def _make_obs(rid, prompt, t0, ttft, inter, assembled, n, finish, err, total_ms=None):
    text = "".join(assembled)
    return {
        "rid": rid,
        "prompt": prompt[:80],
        "send_at_ms": t0 * 1000.0,
        "ttft_ms": ttft,
        "inter_tok_ms_list": inter,
        "n_tokens": n,
        "finish": finish,
        "error": err,
        "total_ms": total_ms if total_ms is not None else (time.perf_counter() - t0) * 1000.0,
        "body_preview": text[:200],
        "body_len": len(text),
    }


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return sorted_vals[idx]


def run_pass(url, n, concurrency, max_tokens, prompt_set, label, out_dir, warmup=True):
    jsonl_path = out_dir / f"raw-{label}.jsonl"
    # Warmup
    if warmup:
        print(f"[bench] warmup {label}…", file=sys.stderr, flush=True)
        w = one_request(url, "Say hi.", 4, 0.7, "warmup")
        if w["error"]:
            print(f"[bench] warmup FAILED: {w['error']}", file=sys.stderr)
            return None
        print(
            f"[bench] warmup ok ttft={w['ttft_ms']:.1f}ms tokens={w['n_tokens']} total={w['total_ms']:.1f}ms",
            file=sys.stderr,
        )

    # Wall-time start
    print(f"[bench] pass={label} n={n} conc={concurrency} max_tokens={max_tokens}",
          file=sys.stderr, flush=True)
    t_start = time.perf_counter()
    obs = []
    lock = threading.Lock()

    def task(i):
        p = prompt_set[i % len(prompt_set)]
        return one_request(url, p, max_tokens, 0.7, f"{label}-{i}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(task, i) for i in range(n)]
        with jsonl_path.open("w") as fp:
            for fut in as_completed(futs):
                o = fut.result()
                with lock:
                    obs.append(o)
                fp.write(json.dumps(o) + "\n")
    wall_s = time.perf_counter() - t_start

    # Aggregate
    ok = [o for o in obs if not o["error"]]
    bad = [o for o in obs if o["error"]]
    total_gen = sum(o["n_tokens"] for o in ok)
    agg_tps = total_gen / wall_s if wall_s > 0 else 0.0

    ttfts = sorted(o["ttft_ms"] for o in ok if o["ttft_ms"] is not None)
    totals = sorted(o["total_ms"] for o in ok)
    per_req_tps = []
    inter_all = []
    for o in ok:
        if o["total_ms"] and o["ttft_ms"] is not None and o["n_tokens"] > 0:
            decode_ms = o["total_ms"] - o["ttft_ms"]
            if decode_ms > 0:
                per_req_tps.append(o["n_tokens"] / (decode_ms / 1000.0))
        inter_all.extend(o["inter_tok_ms_list"])
    per_req_tps.sort()
    inter_all.sort()

    summary = {
        "label": label,
        "n_submitted": n,
        "n_ok": len(ok),
        "n_err": len(bad),
        "wall_s": round(wall_s, 3),
        "total_gen_tokens": total_gen,
        "agg_tok_per_s": round(agg_tps, 2),
        "ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p90": percentile(ttfts, 90),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
            "min": ttfts[0] if ttfts else None,
            "max": ttfts[-1] if ttfts else None,
            "count": len(ttfts),
        },
        "total_ms": {
            "p50": percentile(totals, 50),
            "p90": percentile(totals, 90),
            "p99": percentile(totals, 99),
            "min": totals[0] if totals else None,
            "max": totals[-1] if totals else None,
        },
        "per_req_decode_tok_per_s": {
            "p50": percentile(per_req_tps, 50),
            "p90": percentile(per_req_tps, 90),
            "p99": percentile(per_req_tps, 99),
            "mean": (sum(per_req_tps) / len(per_req_tps)) if per_req_tps else None,
            "count": len(per_req_tps),
        },
        "inter_token_ms": {
            "p50": percentile(inter_all, 50),
            "p90": percentile(inter_all, 90),
            "p99": percentile(inter_all, 99),
            "mean": (sum(inter_all) / len(inter_all)) if inter_all else None,
            "count": len(inter_all),
        },
        "errors": [o["error"] for o in bad[:5]],
    }
    return summary


def fetch_metrics(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"# ERROR: {e}\n"


def fetch_status(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True,
                    help="Base URL, e.g. http://127.0.0.1:8081")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", default="pass")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-set", choices=["short", "long"], default="short")
    ap.add_argument("--explicit-prod", action="store_true",
                    help="Allow hitting :8081 (production).")
    ap.add_argument("--capture-metrics", action="store_true")
    args = ap.parse_args()

    if ":8081" in args.url and not args.explicit_prod:
        print(f"[bench] refusing to hit :8081 without --explicit-prod", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chat_url = args.url.rstrip("/") + "/v1/chat/completions"
    metrics_url = args.url.rstrip("/") + "/metrics"
    status_url = args.url.rstrip("/") + "/v1/status"

    pre_metrics = fetch_metrics(metrics_url)
    pre_status = fetch_status(status_url)

    prompts = PROMPTS_SHORT if args.prompt_set == "short" else PROMPTS_LONG

    summary = run_pass(
        chat_url, args.n, args.conc, args.max_tokens,
        prompts, args.label, out_dir,
    )

    post_metrics = fetch_metrics(metrics_url)
    post_status = fetch_status(status_url)

    if args.capture_metrics:
        (out_dir / f"metrics-pre-{args.label}.txt").write_text(pre_metrics)
        (out_dir / f"metrics-post-{args.label}.txt").write_text(post_metrics)
        (out_dir / f"status-pre-{args.label}.json").write_text(json.dumps(pre_status, indent=2))
        (out_dir / f"status-post-{args.label}.json").write_text(json.dumps(post_status, indent=2))

    summary_path = out_dir / f"summary-{args.label}.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
