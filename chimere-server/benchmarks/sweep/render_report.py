#!/usr/bin/env python3
"""
render_report.py — fill REPORT-TEMPLATE.md from sweep.csv.

Substitutions use `{{KEY}}` tokens. Tables are rendered programmatically
and injected where the template has the corresponding `{{TABLE_xxx}}`
markers.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def fmt_num(v, decimals: int = 1) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def fmt_int(v) -> str:
    if v is None or v == "":
        return "-"
    try:
        return f"{int(float(v))}"
    except (TypeError, ValueError):
        return str(v)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as fp:
        r = csv.DictReader(fp)
        for row in r:
            rows.append(row)
    return rows


def render_grid_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows — sweep produced no data)_"
    lines = [
        "| cell_tag | M | NCMOE | PCH | agg tok/s | decode p50 | decode p99 | TTFT p50 (ms) | TTFT p99 (ms) | VRAM (MiB) | SM p50 | mem p50 | n_ok/n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append("| {tag} | {m} | {n} | {p} | {agg} | {d50} | {d99} | {t50} | {t99} | {vram} | {sm} | {mem} | {ok}/{sub} |".format(
            tag=r.get("cell_tag", "?"),
            m=r.get("multislot", "?"),
            n=r.get("ncmoe", "?"),
            p=r.get("prefill_chunk", "?"),
            agg=fmt_num(r.get("agg_tok_per_s"), 1),
            d50=fmt_num(r.get("per_req_decode_p50"), 1),
            d99=fmt_num(r.get("per_req_decode_p99"), 1),
            t50=fmt_num(r.get("ttft_ms_p50"), 0),
            t99=fmt_num(r.get("ttft_ms_p99"), 0),
            vram=fmt_int(r.get("vram_used_mib_p50")),
            sm=fmt_num(r.get("gpu_sm_p50"), 0),
            mem=fmt_num(r.get("gpu_mem_p50"), 0),
            ok=r.get("n_ok", "?"),
            sub=r.get("n_reqs", "?"),
        ))
    return "\n".join(lines)


def render_ttft_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows)_"
    lines = [
        "| cell_tag | TTFT p50 (ms) | TTFT p99 (ms) | inter-tok p50 (ms) | inter-tok p99 (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {tag} | {t50} | {t99} | {i50} | {i99} |".format(
            tag=r.get("cell_tag", "?"),
            t50=fmt_num(r.get("ttft_ms_p50"), 0),
            t99=fmt_num(r.get("ttft_ms_p99"), 0),
            i50=fmt_num(r.get("inter_tok_ms_p50"), 2),
            i99=fmt_num(r.get("inter_tok_ms_p99"), 2),
        ))
    return "\n".join(lines)


def render_vram_table(rows: list[dict]) -> str:
    if not rows:
        return "_(no rows)_"
    lines = [
        "| cell_tag | VRAM used (MiB) | GPU SM p50 (%) | mem BW p50 (%) | power p50 (W) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {tag} | {vram} | {sm} | {mem} | {pwr} |".format(
            tag=r.get("cell_tag", "?"),
            vram=fmt_int(r.get("vram_used_mib_p50")),
            sm=fmt_num(r.get("gpu_sm_p50"), 0),
            mem=fmt_num(r.get("gpu_mem_p50"), 0),
            pwr=fmt_num(r.get("gpu_pwr_p50_w"), 0),
        ))
    return "\n".join(lines)


def pick_sweet_spot(rows: list[dict]) -> str:
    """Rank cells by a simple utility:

      rank = agg_tok_per_s  - 0.1 * ttft_p99_ms  - 0.02 * vram_mib

    Highest wins. Ties broken by lower VRAM. If all rows have zero agg
    throughput (e.g. all failures), say so.
    """
    valid = []
    for r in rows:
        try:
            agg = float(r.get("agg_tok_per_s") or 0)
            t99 = float(r.get("ttft_ms_p99") or 0)
            vram = float(r.get("vram_used_mib_p50") or 0)
        except ValueError:
            continue
        if agg <= 0:
            continue
        utility = agg - 0.1 * t99 - 0.02 * vram
        valid.append((utility, agg, t99, vram, r))

    if not valid:
        return "No valid cell: all cells failed or returned zero agg tok/s. Check logs/."
    valid.sort(reverse=True)
    _, agg, t99, vram, r = valid[0]
    return (
        f"**{r.get('cell_tag')}** — agg={agg:.1f} tok/s, TTFT p99={t99:.0f} ms, "
        f"VRAM={int(vram)} MiB. "
        f"(M={r.get('multislot')}, NCMOE={r.get('ncmoe')}, "
        f"prefill_chunk={r.get('prefill_chunk')}.)"
    )


def detect_hardware() -> dict:
    """Detect GPU + basic host info via nvidia-smi. Best-effort."""
    info = {"gpu": "unknown", "driver": "unknown", "cuda": "unknown"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            line = out.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["gpu"] = f"{parts[0]} ({parts[2]})"
                info["driver"] = parts[1]
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            info["cuda"] = out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return info


def detect_model_quant(model_path: str) -> str:
    """Extract quant from filename if possible."""
    p = Path(model_path).name
    # Strip common GGUF suffix
    stem = p.replace(".gguf", "")
    # Find the quant token (IQ3_S, Q4_K_M, UD-IQ3_S, etc.) — anything
    # after the last '-' containing digits+letters.
    parts = stem.split("-")
    for token in reversed(parts):
        if any(ch.isdigit() for ch in token) and "_" in token.upper():
            return token
    return stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--server-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--multislot-sweep", required=True)
    ap.add_argument("--ncmoe-sweep", required=True)
    ap.add_argument("--prefill-chunk-sweep", required=True)
    ap.add_argument("--n-reqs-per-pass", required=True)
    ap.add_argument("--max-tokens", required=True)
    ap.add_argument("--prompt-set", required=True)
    args = ap.parse_args()

    template_text = Path(args.template).read_text()
    rows = read_csv(Path(args.csv))
    hw = detect_hardware()

    grid_table = render_grid_table(rows)
    ttft_table = render_ttft_table(rows)
    vram_table = render_vram_table(rows)
    sweet_spot = pick_sweet_spot(rows)

    # Repro command: approximate (the actual sweep-bench.sh invocation).
    repro = (
        "./sweep-bench.sh \\\n"
        f"    --output-dir {args.output_dir} \\\n"
        f"    --multislot-sweep \"{args.multislot_sweep}\" \\\n"
        f"    --ncmoe-sweep \"{args.ncmoe_sweep}\" \\\n"
        f"    --prefill-chunk-sweep \"{args.prefill_chunk_sweep}\" \\\n"
        f"    --n-requests-per-pass {args.n_reqs_per_pass} \\\n"
        f"    --max-tokens {args.max_tokens} \\\n"
        f"    --prompt-set {args.prompt_set}"
    )

    subs = {
        "START_ISO": args.start,
        "END_ISO": args.end,
        "GIT_SHA": args.git_sha,
        "SERVER_ROOT": args.server_root,
        "OUTPUT_DIR": args.output_dir,
        "MODEL_PATH": args.model_path,
        "MODEL_QUANT": detect_model_quant(args.model_path),
        "GPU": hw["gpu"],
        "DRIVER": hw["driver"],
        "CUDA": hw["cuda"],
        "MULTISLOT_SWEEP": args.multislot_sweep,
        "NCMOE_SWEEP": args.ncmoe_sweep,
        "PREFILL_CHUNK_SWEEP": args.prefill_chunk_sweep,
        "N_REQS_PER_PASS": args.n_reqs_per_pass,
        "MAX_TOKENS": args.max_tokens,
        "PROMPT_SET": args.prompt_set,
        "N_CELLS": str(len(rows)),
        "TABLE_GRID": grid_table,
        "TABLE_TTFT": ttft_table,
        "TABLE_VRAM": vram_table,
        "SWEET_SPOT": sweet_spot,
        "REPRO_COMMAND": repro,
    }

    out = template_text
    for k, v in subs.items():
        out = out.replace("{{" + k + "}}", str(v))
    Path(args.output).write_text(out)
    print(f"[render_report] wrote {args.output} ({len(out)} chars, {len(rows)} cells)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
