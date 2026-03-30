#!/usr/bin/env python3
"""scheduler.py -- Lightweight nightly job scheduler for Chimere self-improvement.

Reads configuration from environment variables:
  NIGHTLY_SCHEDULE  HH:MM (24h) -- when to run jobs each day
  NIGHTLY_LORA      true/false  -- LoRA training from quality pairs
  NIGHTLY_ENGRAM    true/false  -- Engram n-gram table updates
  NIGHTLY_DSPY      true/false  -- DSPy prompt optimization
  NIGHTLY_MEZO      true/false  -- MeZO zeroth-order LoRA training
  ODO_BACKEND       URL         -- inference server for scoring/eval

Logs go to /data/logs/nightly/ with one file per run date.
"""

import datetime
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

LOG_DIR = Path("/data/logs/nightly")
HEARTBEAT = Path("/tmp/scheduler-heartbeat")
SCRIPTS = Path("/app/scripts")

JOBS = [
    ("NIGHTLY_ENGRAM", "engram_write_nightly.py", []),
    ("NIGHTLY_LORA",   "nightly_lora.py",         []),
    ("NIGHTLY_MEZO",   "lora_mezo_nightly.py",    []),
    ("NIGHTLY_DSPY",   "dspy_optimize.py",         ["--all", "--auto", "light"]),
]


def is_enabled(env_key: str) -> bool:
    return os.environ.get(env_key, "false").lower() in ("true", "1", "yes")


def parse_schedule() -> tuple[int, int]:
    raw = os.environ.get("NIGHTLY_SCHEDULE", "00:30")
    h, m = raw.split(":")
    return int(h), int(m)


def seconds_until(hour: int, minute: int) -> float:
    now = datetime.datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    return (target - now).total_seconds()


def run_job(name: str, script: str, extra_args: list[str], log_file) -> bool:
    path = SCRIPTS / script
    if not path.exists():
        log_file.write(f"  SKIP {name}: {path} not found\n")
        return False
    log_file.write(f"  START {name} ({script})\n")
    log_file.flush()
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, str(path)] + extra_args,
            capture_output=True, text=True, timeout=3600,
        )
        elapsed = time.monotonic() - t0
        log_file.write(result.stdout)
        if result.stderr:
            log_file.write(f"  STDERR:\n{result.stderr}\n")
        status = "OK" if result.returncode == 0 else f"FAIL (rc={result.returncode})"
        log_file.write(f"  {status} {name} in {elapsed:.1f}s\n\n")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        log_file.write(f"  TIMEOUT {name} after 3600s\n\n")
        return False
    except Exception:
        log_file.write(f"  ERROR {name}:\n{traceback.format_exc()}\n\n")
        return False


def run_nightly():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    log_path = LOG_DIR / f"nightly-{stamp}.log"

    with open(log_path, "w") as log:
        log.write(f"=== Chimere Nightly Run {stamp} ===\n")
        log.write(f"ODO_BACKEND={os.environ.get('ODO_BACKEND', 'unset')}\n\n")

        results = {}
        for env_key, script, args in JOBS:
            if is_enabled(env_key):
                results[script] = run_job(env_key, script, args, log)
            else:
                log.write(f"  DISABLED {env_key}\n")

        ok = sum(1 for v in results.values() if v)
        fail = sum(1 for v in results.values() if not v)
        log.write(f"=== Done: {ok} passed, {fail} failed ===\n")

    print(f"[nightly] Run complete: {log_path} ({ok} ok, {fail} fail)")


def main():
    hour, minute = parse_schedule()
    enabled = [name for env_key, name, _ in JOBS if is_enabled(env_key)]
    print(f"[scheduler] Chimere nightly scheduler started")
    print(f"[scheduler] Schedule: {hour:02d}:{minute:02d} daily")
    print(f"[scheduler] Enabled jobs: {', '.join(enabled) or 'none'}")
    print(f"[scheduler] ODO_BACKEND: {os.environ.get('ODO_BACKEND', 'unset')}")

    while True:
        wait = seconds_until(hour, minute)
        print(f"[scheduler] Next run in {wait/3600:.1f}h ({datetime.datetime.now() + datetime.timedelta(seconds=wait):%Y-%m-%d %H:%M})")
        HEARTBEAT.write_text(str(time.time()))

        # Sleep in 60s increments so the heartbeat stays fresh
        while wait > 0:
            chunk = min(wait, 60)
            time.sleep(chunk)
            wait -= chunk
            HEARTBEAT.write_text(str(time.time()))

        run_nightly()


if __name__ == "__main__":
    main()
