#!/usr/bin/env python3
"""
Chimere GPU Detection Script

Detects the installed NVIDIA GPU, maps it to SM architecture and VRAM,
selects the appropriate quantisation level, and writes a .env file
that docker-compose.yml reads automatically.

Usage:
    python3 scripts/detect-gpu.py            # auto-detect, write ../.env
    python3 scripts/detect-gpu.py --dry-run  # print without writing
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---- GPU database: name pattern -> (sm_arch, typical_vram_gb) ----
# Patterns are matched case-insensitively against the GPU name from nvidia-smi.
# Order matters: first match wins, so put specific models before families.
GPU_DATABASE: list[tuple[str, int, int]] = [
    # Blackwell (sm_120)
    (r"RTX\s*5090",       120, 32),
    (r"RTX\s*5080",       120, 16),
    (r"RTX\s*5070\s*Ti",  120, 16),
    (r"RTX\s*5070",       120, 12),
    (r"RTX\s*5060\s*Ti",  120, 16),
    (r"RTX\s*5060",       120, 8),
    # Ada Lovelace (sm_89)
    (r"RTX\s*4090",       89, 24),
    (r"RTX\s*4080\s*S",   89, 16),
    (r"RTX\s*4080",       89, 16),
    (r"RTX\s*4070\s*Ti\s*S", 89, 16),
    (r"RTX\s*4070\s*Ti",  89, 12),
    (r"RTX\s*4070\s*S",   89, 12),
    (r"RTX\s*4070",       89, 12),
    (r"RTX\s*4060\s*Ti",  89, 16),
    (r"RTX\s*4060",       89, 8),
    # Hopper (sm_90)
    (r"H100",             90, 80),
    (r"H200",             90, 141),
    (r"H20",              90, 96),
    # Ampere (sm_86 consumer, sm_80 datacenter)
    (r"RTX\s*3090\s*Ti",  86, 24),
    (r"RTX\s*3090",       86, 24),
    (r"RTX\s*3080\s*Ti",  86, 12),
    (r"RTX\s*3080",       86, 12),
    (r"RTX\s*3070\s*Ti",  86, 8),
    (r"RTX\s*3070",       86, 8),
    (r"RTX\s*3060\s*Ti",  86, 8),
    (r"RTX\s*3060",       86, 12),
    (r"A100",             80, 80),
    (r"A6000",            86, 48),
    (r"A40",              86, 48),
    (r"A30",              80, 24),
    (r"A10G?",            86, 24),
    # Turing (sm_75)
    (r"RTX\s*2080\s*Ti",  75, 11),
    (r"RTX\s*2080\s*S",   75, 8),
    (r"RTX\s*2080",       75, 8),
    (r"RTX\s*2070",       75, 8),
    (r"RTX\s*2060",       75, 6),
    (r"T4",               75, 16),
    # Datacenter latest
    (r"L40S?",            89, 48),
    (r"L4",               89, 24),
    (r"B200",             120, 192),
    (r"GB200",            120, 192),
]

# ---- VRAM -> quantisation profile ----
VRAM_PROFILES: list[tuple[int, str, str, int]] = [
    # (min_vram_gb, quant_tag, model_filename, ncmoe)
    (24, "q4km",  "Qwen3.5-35B-A3B-Q4_K_M.gguf",           14),
    (16, "iq3s",  "Qwen3.5-35B-A3B-RAMP-v2-15g.gguf",      4),
    (12, "iq2",   "Qwen3.5-35B-A3B-IQ2_M.gguf",            8),
    (8,  "iq2xs", "Qwen3.5-35B-A3B-IQ2_XS.gguf",           12),
]

# HuggingFace model repo base
HF_REPO_BASE = "Kevletesteur"

# Model URLs keyed by quant tag
MODEL_URLS: dict[str, str] = {
    "q4km":  f"https://huggingface.co/{HF_REPO_BASE}/Qwen3.5-35B-A3B-Q4_K_M-GGUF",
    "iq3s":  f"https://huggingface.co/{HF_REPO_BASE}/Qwen3.5-35B-A3B-RAMP-v2-15G",
    "iq2":   f"https://huggingface.co/{HF_REPO_BASE}/Qwen3.5-35B-A3B-IQ2_M-GGUF",
    "iq2xs": f"https://huggingface.co/{HF_REPO_BASE}/Qwen3.5-35B-A3B-IQ2_XS-GGUF",
}


def run_nvidia_smi() -> tuple[str, int]:
    """Run nvidia-smi and return (gpu_name, vram_mb)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. Is the NVIDIA driver installed?", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: nvidia-smi timed out.", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        print(f"ERROR: nvidia-smi failed:\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    # Take the first GPU if multiple are present
    line = result.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        print(f"ERROR: Unexpected nvidia-smi output: {line}", file=sys.stderr)
        sys.exit(1)

    gpu_name = parts[0]
    try:
        vram_mb = int(float(parts[1]))
    except ValueError:
        print(f"ERROR: Could not parse VRAM from: {parts[1]}", file=sys.stderr)
        sys.exit(1)

    return gpu_name, vram_mb


def match_gpu(gpu_name: str) -> tuple[int, None] | tuple[int, int]:
    """Match a GPU name against the database. Returns (sm_arch, db_vram) or (0, None)."""
    for pattern, sm, vram in GPU_DATABASE:
        if re.search(pattern, gpu_name, re.IGNORECASE):
            return sm, vram
    return 0, None


def select_profile(vram_gb: int) -> tuple[str, str, int, str]:
    """Select quant profile based on available VRAM.
    Returns (quant_tag, model_filename, ncmoe, model_url).
    """
    for min_vram, tag, filename, ncmoe in VRAM_PROFILES:
        if vram_gb >= min_vram:
            return tag, filename, ncmoe, MODEL_URLS.get(tag, "")
    # Fallback to smallest quant
    tag, filename, ncmoe = VRAM_PROFILES[-1][1], VRAM_PROFILES[-1][2], VRAM_PROFILES[-1][3]
    return tag, filename, ncmoe, MODEL_URLS.get(tag, "")


def write_env(env_path: Path, values: dict[str, str], dry_run: bool = False) -> None:
    """Write or update a .env file with the given key-value pairs."""
    lines: list[str] = []

    # Read existing .env if present
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0]
                existing[key] = line
            lines.append(line)

    # Update or append new values
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key in values:
                new_lines.append(f"{key}={values[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Append keys that were not already in the file
    for key, val in values.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={val}")

    content = "\n".join(new_lines) + "\n"

    if dry_run:
        print("\n--- .env content (dry run, not written) ---")
        print(content)
        return

    env_path.write_text(content)
    print(f"Wrote {env_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect GPU and configure Chimere .env")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing .env",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file (default: <repo-root>/.env)",
    )
    args = parser.parse_args()

    # Locate repo root (two levels up from scripts/)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    env_path = Path(args.env_file) if args.env_file else repo_root / ".env"

    print("=== Chimere GPU Detection ===\n")

    # Detect GPU
    gpu_name, vram_mb = run_nvidia_smi()
    vram_gb = round(vram_mb / 1024)
    print(f"  GPU:           {gpu_name}")
    print(f"  VRAM:          {vram_mb} MB ({vram_gb} GB)")

    # Match architecture
    sm_arch, db_vram = match_gpu(gpu_name)
    if sm_arch == 0:
        print(f"\n  WARNING: GPU '{gpu_name}' not in database.")
        print("  Defaulting to sm_89 (Ada Lovelace). Override GPU_ARCH in .env if needed.")
        sm_arch = 89
    else:
        print(f"  SM arch:       sm_{sm_arch}")

    # Select quantisation profile
    quant_tag, model_filename, ncmoe, model_url = select_profile(vram_gb)
    print(f"\n  Quant profile: {quant_tag}")
    print(f"  Model file:    {model_filename}")
    print(f"  CPU MoE:       {ncmoe} layers")
    print(f"  Model URL:     {model_url}")

    # Determine context size based on VRAM
    if vram_gb >= 24:
        ctx = 65536
    elif vram_gb >= 16:
        ctx = 32768
    elif vram_gb >= 12:
        ctx = 16384
    else:
        ctx = 8192

    print(f"  Context:       {ctx} tokens")

    # Write .env
    env_values = {
        "GPU_ARCH": str(sm_arch),
        "VRAM_GB": str(vram_gb),
        "MODEL_FILENAME": model_filename,
        "MODEL_URL": model_url,
        "QUANT_TAG": quant_tag,
        "CHIMERE_CTX": str(ctx),
        "CHIMERE_NCMOE": str(ncmoe),
    }

    print(f"\n  .env path:     {env_path}")
    write_env(env_path, env_values, dry_run=args.dry_run)

    print("\n=== Detection complete ===")
    if not args.dry_run:
        print(f"\nNext step: run scripts/download-model.sh to fetch the model.")


if __name__ == "__main__":
    main()
