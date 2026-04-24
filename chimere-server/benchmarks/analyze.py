#!/usr/bin/env python3
"""Analyze nvidia-smi dmon CSVs and raw JSONL to populate report stats."""
import json
import sys
import statistics
from pathlib import Path


def summarize_dmon(path):
    """Columns per `nvidia-smi dmon -s puct -o T` (space-separated):
    Time gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk rxpci txpci
    """
    sm_vals = []
    mem_vals = []
    pwr_vals = []
    rx_vals = []
    tx_vals = []
    active_sm_vals = []
    mclk_vals = []
    pclk_vals = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 15:
            continue
        try:
            # Time(0) gpu(1) pwr(2) gtemp(3) mtemp(4) sm(5) mem(6) enc(7)
            # dec(8) jpg(9) ofa(10) mclk(11) pclk(12) rxpci(13) txpci(14)
            pwr = float(parts[2])
            sm = float(parts[5])
            mem = float(parts[6])
            mclk = float(parts[11])
            pclk = float(parts[12])
            rx = float(parts[13])
            tx = float(parts[14])
        except ValueError:
            continue
        sm_vals.append(sm)
        mem_vals.append(mem)
        pwr_vals.append(pwr)
        mclk_vals.append(mclk)
        pclk_vals.append(pclk)
        rx_vals.append(rx)
        tx_vals.append(tx)
        if sm > 5:
            active_sm_vals.append(sm)

    def stats(vs):
        if not vs:
            return dict(mean=None, p50=None, p95=None, max=None)
        s = sorted(vs)
        return dict(
            mean=round(statistics.mean(s), 2),
            p50=round(s[len(s) // 2], 2),
            p95=round(s[int(0.95 * (len(s) - 1))], 2),
            max=round(s[-1], 2),
        )

    return {
        "n_samples": len(sm_vals),
        "sm_util_pct_all": stats(sm_vals),
        "mem_util_pct_all": stats(mem_vals),
        "sm_util_pct_active_only": stats(active_sm_vals),
        "power_w": stats(pwr_vals),
        "mclk_mhz": stats(mclk_vals),
        "pclk_mhz": stats(pclk_vals),
        "rxpci_mbps": stats(rx_vals),
        "txpci_mbps": stats(tx_vals),
        "fraction_active_seconds": round(len(active_sm_vals) / max(1, len(sm_vals)), 3),
    }


def summarize_jsonl(path):
    obs = [json.loads(l) for l in Path(path).read_text().splitlines() if l]
    ok = [o for o in obs if not o.get("error")]
    ttfts = sorted(o["ttft_ms"] for o in ok if o.get("ttft_ms") is not None)
    totals = sorted(o["total_ms"] for o in ok)
    # finish reasons
    finishes = {}
    for o in ok:
        f = o.get("finish") or "none"
        finishes[f] = finishes.get(f, 0) + 1
    tokens = [o["n_tokens"] for o in ok]
    body_lens = [o["body_len"] for o in ok]
    return {
        "n_ok": len(ok),
        "ttft_ms_mean": round(statistics.mean(ttfts), 2) if ttfts else None,
        "ttft_ms_stdev": round(statistics.stdev(ttfts), 2) if len(ttfts) > 1 else None,
        "finish_breakdown": finishes,
        "tokens_per_req_mean": round(statistics.mean(tokens), 2) if tokens else None,
        "body_len_mean": round(statistics.mean(body_lens), 2) if body_lens else None,
    }


def parse_prom_metric(text, name, labels=None):
    """Very small Prometheus text parser — finds `name{labels}` value."""
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if not line.startswith(name):
            continue
        rest = line[len(name):]
        if rest.startswith("{"):
            # has labels
            if labels:
                # simple match: check each k=v pair
                ok = all(f'{k}="{v}"' in rest for k, v in labels.items())
                if not ok:
                    continue
            # strip labels
            idx = rest.index("}")
            val = rest[idx + 1:].strip()
        else:
            val = rest.strip()
        try:
            return float(val)
        except ValueError:
            return None
    return None


def metric_delta(pre_text, post_text, name, labels=None):
    a = parse_prom_metric(pre_text, name, labels) or 0.0
    b = parse_prom_metric(post_text, name, labels) or 0.0
    return b - a


def main():
    base = Path("/home/remondiere/Bureau/chimere-drafts/e2e-profile-v1/raw")
    out = {}
    for m in ("M1", "M2", "M4"):
        pre = (base / f"metrics-pre-{m}.txt").read_text()
        post = (base / f"metrics-post-{m}.txt").read_text()
        dmon = summarize_dmon(base / f"nvidia-smi-dmon-{m}.csv")
        jsonl = summarize_jsonl(base / f"raw-{m}.jsonl")
        deltas = {
            "requests_ok": metric_delta(pre, post, "chimere_requests_total", {"status": "ok"}),
            "requests_error": metric_delta(pre, post, "chimere_requests_total", {"status": "error"}),
            "prompt_tokens_total": metric_delta(pre, post, "chimere_prompt_tokens_total"),
            "gen_tokens_total": metric_delta(pre, post, "chimere_gen_tokens_total"),
        }
        ttft_prom = {
            "p50_s": parse_prom_metric(post, "chimere_ttft_seconds", {"quantile": "0.50"}),
            "p90_s": parse_prom_metric(post, "chimere_ttft_seconds", {"quantile": "0.90"}),
            "p95_s": parse_prom_metric(post, "chimere_ttft_seconds", {"quantile": "0.95"}),
            "p99_s": parse_prom_metric(post, "chimere_ttft_seconds", {"quantile": "0.99"}),
            "sum_s": parse_prom_metric(post, "chimere_ttft_seconds_sum"),
            "count": parse_prom_metric(post, "chimere_ttft_seconds_count"),
        }
        out[m] = {
            "prom_deltas": deltas,
            "ttft_prom": ttft_prom,
            "dmon": dmon,
            "jsonl_extra": jsonl,
        }
    (base.parent / "analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
