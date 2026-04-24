#!/usr/bin/env python3
"""
driver_wrapper.py — per-cell driver for sweep-bench.sh.

Responsibilities:
  1. Load prompts from a YAML file (minimal inline parser; we stay
     stdlib-only so no PyYAML dependency).
  2. Import stream_bench from the chimere-server benchmarks/ dir and
     monkey-patch its PROMPTS_SHORT/PROMPTS_LONG with the loaded prompt
     bank, so the per-cell semantics (streaming, TTFT, JSONL) match
     exactly what the repo's existing bench produces.
  3. Drive one pass and drop cell-summary.json in --out-dir. sweep-bench.sh
     picks that up for CSV aggregation.

Stdlib-only. No argparse-free tricks. No deps beyond python3.

Exit code 0 on success (summary written). Non-zero means the caller
should NOT record this cell as OK.
"""

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def _block_scalar_pass(path: Path) -> list[dict]:
    """Reliable block-scalar parser. Scans for `- label: <name>` items,
    then captures sibling `key: value` and `key: |` block scalars until
    the next `- label:` at the same (or lesser) indent.

    Semantics are intentionally tighter than general YAML — only the
    subset needed for prompts.yaml is supported.
    """
    out: list[dict] = []
    lines = path.read_text().splitlines()
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^\s*-\s+label:\s*(.+?)\s*$", lines[i])
        if m:
            entry: dict = {"label": m.group(1).strip().strip('"').strip("'")}
            i += 1
            while i < n:
                line = lines[i]
                if re.match(r"^\s*-\s+label:", line):
                    break
                if line.strip().startswith("#") or line.strip() == "":
                    i += 1
                    continue
                km = re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$", line)
                if km:
                    key, val = km.group(1), km.group(2)
                    if val.strip() == "|":
                        i += 1
                        buf: list[str] = []
                        block_indent: int | None = None
                        while i < n:
                            nxt = lines[i]
                            # Terminate block scalar at the next `- label:`
                            if re.match(r"^\s*-\s+label:", nxt):
                                break
                            if nxt.strip() == "":
                                buf.append("")
                                i += 1
                                continue
                            lead = len(nxt) - len(nxt.lstrip(" "))
                            if block_indent is None:
                                block_indent = lead
                            if lead < block_indent:
                                break
                            buf.append(nxt[block_indent:])
                            i += 1
                        entry[key] = "\n".join(buf).rstrip() + "\n"
                        continue
                    else:
                        entry[key] = val.strip().strip('"').strip("'")
                i += 1
            out.append(entry)
            continue
        i += 1
    return out


def import_stream_bench(path: Path):
    """Import stream_bench.py as a module without adding its dir to
    sys.path permanently."""
    spec = importlib.util.spec_from_file_location("stream_bench_mod", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--stream-bench", required=True,
                    help="path to chimere-server/benchmarks/stream_bench.py")
    ap.add_argument("--prompt-set", required=True, help="prompts.yaml path")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    prompts_path = Path(args.prompt_set)
    prompts = _block_scalar_pass(prompts_path)
    if not prompts:
        print(f"[driver_wrapper] no prompts loaded from {prompts_path}", file=sys.stderr)
        return 2
    prompt_texts = [p["text"].strip() for p in prompts if p.get("text", "").strip()]
    if not prompt_texts:
        print(f"[driver_wrapper] prompts loaded but all empty", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture metrics-pre/status-pre via the stream_bench helpers (they
    # are convenient re-exports). Also fetch /v1/status once to note
    # slot_pool_size etc.
    sb = import_stream_bench(Path(args.stream_bench))
    metrics_url = args.url.rstrip("/") + "/metrics"
    status_url = args.url.rstrip("/") + "/v1/status"
    chat_url = args.url.rstrip("/") + "/v1/chat/completions"

    pre_metrics = sb.fetch_metrics(metrics_url)
    pre_status = sb.fetch_status(status_url)
    (out_dir / "metrics-pre.txt").write_text(pre_metrics)
    (out_dir / "status-pre.json").write_text(json.dumps(pre_status, indent=2))

    # Monkey-patch stream_bench's global prompt banks with our YAML set,
    # then call its run_pass(). This gives us identical JSONL/summary
    # semantics to the repo harness.
    sb.PROMPTS_SHORT = prompt_texts  # used when prompt_set="short"
    sb.PROMPTS_LONG = prompt_texts   # and when "long"; both point at ours

    # run_pass writes raw-{label}.jsonl into out_dir
    summary = sb.run_pass(
        chat_url, args.n, args.conc, args.max_tokens,
        prompt_texts, args.label, out_dir,
    )
    if summary is None:
        print(f"[driver_wrapper] run_pass returned None (warmup failed?)", file=sys.stderr)
        return 3

    post_metrics = sb.fetch_metrics(metrics_url)
    post_status = sb.fetch_status(status_url)
    (out_dir / "metrics-post.txt").write_text(post_metrics)
    (out_dir / "status-post.json").write_text(json.dumps(post_status, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Cell-summary.json bundles enough for csv_append.py to not re-parse
    # intermediate artifacts. Keep it tight.
    cell = {
        "label": args.label,
        "n_submitted": summary["n_submitted"],
        "n_ok": summary["n_ok"],
        "n_err": summary["n_err"],
        "wall_s": summary["wall_s"],
        "total_gen_tokens": summary["total_gen_tokens"],
        "agg_tok_per_s": summary["agg_tok_per_s"],
        "ttft_ms": summary["ttft_ms"],
        "per_req_decode_tok_per_s": summary["per_req_decode_tok_per_s"],
        "inter_token_ms": summary["inter_token_ms"],
        "errors": summary["errors"],
        "prompts_used": [p.get("label", "?") for p in prompts],
    }
    (out_dir / "cell-summary.json").write_text(json.dumps(cell, indent=2))

    # Stdout for the harness log
    print(json.dumps({
        "label": args.label,
        "n_ok": summary["n_ok"],
        "wall_s": summary["wall_s"],
        "agg_tok_per_s": summary["agg_tok_per_s"],
        "ttft_p50_ms": summary["ttft_ms"]["p50"],
        "decode_p50": summary["per_req_decode_tok_per_s"]["p50"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
