#!/usr/bin/env python3
"""
csv_append.py — append one aggregated row to sweep.csv for a single cell.

Reads:
  - <cell_raw>/cell-summary.json  (from driver_wrapper.py)
  - <dmon_csv>                    (from `nvidia-smi dmon -s pumct -o T`)
  - vram_baseline (MiB; measured right after server boot, pre-load is
    irrelevant here — we use it to distinguish "what the model costs"
    from "what the OS had before")

Writes: one CSV line appended to <csv>.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    idx = max(0, min(idx, len(s) - 1))
    return s[idx]


def parse_dmon(csv_path: Path) -> dict:
    """Parse `nvidia-smi dmon -s pumct -o T` output.

    dmon format (depends on -s flags). With -s pumct we get columns:
      Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk

    The header starts with '#Time ...'. The second line is '#' + units.
    Data lines have no leading '#'. Active filter: sm > 5 %.
    """
    if not csv_path.exists():
        return {}
    rows = []
    try:
        for line in csv_path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expected min 13 cols (-s pumct)
            if len(parts) < 7:
                continue
            # Heuristic: pick pwr (col 2), sm (col 5), mem (col 6).
            # dmon widths are fixed but we parse whitespace-split.
            try:
                pwr = float(parts[2])
                sm = float(parts[5])
                mem = float(parts[6])
            except (IndexError, ValueError):
                continue
            rows.append((pwr, sm, mem))
        if not rows:
            return {}
        active = [r for r in rows if r[1] > 5.0]
        if not active:
            active = rows
        pwrs = [r[0] for r in active]
        sms = [r[1] for r in active]
        mems = [r[2] for r in active]
        return {
            "gpu_sm_p50": percentile(sms, 50),
            "gpu_mem_p50": percentile(mems, 50),
            "gpu_pwr_p50_w": percentile(pwrs, 50),
        }
    except Exception as e:
        print(f"[csv_append] warning: failed to parse dmon csv: {e}", file=sys.stderr)
        return {}


def read_vram_peak_mib(dmon_csv: Path) -> tuple[float | None, float | None]:
    """`nvidia-smi dmon -s m` gives mem BW util (%), NOT VRAM MiB.
    For VRAM MiB we rely on pre/post nvidia-smi --query-gpu=memory.used.
    This function returns (None, None) — we take VRAM snapshots in the
    shell wrapper instead.
    """
    return (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-tag", required=True)
    ap.add_argument("--git-sha", required=True)
    ap.add_argument("--multislot", required=True)
    ap.add_argument("--ncmoe", required=True)
    ap.add_argument("--prefill-chunk", required=True)
    ap.add_argument("--conc", required=True)
    ap.add_argument("--max-tokens", required=True)
    ap.add_argument("--prompt-set", required=True)
    ap.add_argument("--cell-raw", required=True)
    ap.add_argument("--dmon-csv", required=True)
    ap.add_argument("--vram-baseline", required=True,
                    help="MiB used right after server boot (before load)")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    cell_raw = Path(args.cell_raw)
    summary_path = cell_raw / "cell-summary.json"
    if not summary_path.exists():
        print(f"[csv_append] missing {summary_path}", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text())

    dmon_stats = parse_dmon(Path(args.dmon_csv))

    # VRAM MiB: sample nvidia-smi right now. The harness only runs this
    # post-pass, so it's "post-load + live kv/activation + whatever the
    # driver is doing" — close enough for sweet-spot ranking.
    vram_used = 0
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            vram_used = int(out.stdout.strip().split()[0])
    except Exception:
        pass

    ttft = summary.get("ttft_ms", {}) or {}
    dec = summary.get("per_req_decode_tok_per_s", {}) or {}
    itm = summary.get("inter_token_ms", {}) or {}

    row = [
        args.cell_tag,
        args.git_sha,
        args.multislot,
        args.ncmoe,
        args.prefill_chunk,
        summary.get("n_submitted", 0),
        args.conc,
        args.max_tokens,
        args.prompt_set,
        summary.get("wall_s", 0),
        summary.get("total_gen_tokens", 0),
        summary.get("agg_tok_per_s", 0),
        dec.get("p50", "") if dec.get("p50") is not None else "",
        dec.get("p99", "") if dec.get("p99") is not None else "",
        ttft.get("p50", "") if ttft.get("p50") is not None else "",
        ttft.get("p99", "") if ttft.get("p99") is not None else "",
        itm.get("p50", "") if itm.get("p50") is not None else "",
        itm.get("p99", "") if itm.get("p99") is not None else "",
        vram_used,   # p50 placeholder (we have one sample)
        vram_used,   # p95 placeholder
        dmon_stats.get("gpu_sm_p50", ""),
        dmon_stats.get("gpu_mem_p50", ""),
        dmon_stats.get("gpu_pwr_p50_w", ""),
        summary.get("n_ok", 0),
        summary.get("n_err", 0),
        ";".join((summary.get("errors") or [])[:3]),
    ]

    with Path(args.csv).open("a", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
