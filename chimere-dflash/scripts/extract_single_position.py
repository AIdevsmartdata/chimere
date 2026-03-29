#!/usr/bin/env python3
"""
extract_single_position.py — Extract single-position hidden states for DFlash v6.

For each prompt+response, runs a forward pass through the target model and saves
ONLY the hidden state at the last position (the anchor point). This is the only
vector the DFlash drafter needs to condition its block prediction.

Output per sample:
  sample_NNNNNN/
    anchor_hidden.bin    — float16[5, 2048] (5 layers × hidden_dim)
    block_tokens.bin     — int32[block_size] (the next 16 tokens after anchor)
    metadata.json        — {anchor_pos, seq_len, layers, hidden_dim, dtype, source_id}

Total disk: 100K samples × ~20.5 KB = ~2 GB

Auto-restarts daemon on CUDA crashes (up to MAX_RESTARTS times).

Usage:
  python scripts/extract_single_position.py \
    --input data/prompts_v6/ready_for_extraction.jsonl \
    --output data/features_q5 \
    --model ~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chimere.target_daemon import TargetDaemon

LAYERS = [1, 10, 19, 28, 37]
HIDDEN_DIM = 2048
BLOCK_SIZE = 16
MASK_TOKEN_ID = 248077  # <|MASK|> via add_special_tokens (z-lab convention)
MAX_RESTARTS = 50


def make_daemon(args, extra_args):
    """Create and return a new TargetDaemon instance."""
    return TargetDaemon(
        daemon_path=args.daemon,
        model_path=args.model,
        layers=LAYERS,
        n_gpu_layers=99,
        extra_args=extra_args,
    )


def main():
    parser = argparse.ArgumentParser(description="Extract single-position features for DFlash v6")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/features_q5")
    parser.add_argument("--model", type=str,
                        default=os.path.expanduser(
                            "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/"
                            "Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"))
    parser.add_argument("--daemon", type=str,
                        default=os.path.expanduser(
                            "~/chimere-dflash/extract/build/target_daemon"))
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Max tokens to process per sample")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Resume from prompt index N (skips first N prompts)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"Loading prompts from {args.input}...", flush=True)
    prompts = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    if args.max_samples > 0:
        prompts = prompts[:args.max_samples]

    # Sort by length (short first) — faster extraction + crashes on long prompts come last
    prompts.sort(key=lambda p: len(p["prompt"]) + len(p.get("response", "")))
    print(f"  {len(prompts)} prompts (sorted by length)", flush=True)

    # Count existing samples for resume
    existing = len(list(output_dir.glob("sample_*")))
    if existing > 0:
        print(f"  {existing} samples already exist, appending", flush=True)

    est_gb = len(prompts) * 5 * HIDDEN_DIM * 2 / 1e9
    print(f"\n{'='*60}", flush=True)
    print(f" DFlash v6 Single-Position Extraction", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Model:            {args.model}", flush=True)
    print(f"  Layers:           {LAYERS}", flush=True)
    print(f"  Block size:       {BLOCK_SIZE}", flush=True)
    print(f"  Max seq len:      {args.max_seq_len}", flush=True)
    print(f"  Anchor:           last valid position (1 per sample)", flush=True)
    print(f"  Prompts:          {len(prompts)}", flush=True)
    print(f"  Resume from:      {args.resume_from}", flush=True)
    print(f"  Est. disk:        {est_gb:.1f} GB", flush=True)
    print(f"  Max restarts:     {MAX_RESTARTS}", flush=True)
    print(flush=True)

    # Daemon args
    extra_args = [
        "--flash-attn", "on",
        "-c", str(args.max_seq_len + 64),
        "-b", "4096", "-ub", "4096",
        "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
    ]
    if "Q5" in args.model or "q5" in args.model:
        extra_args += ["-ot", "blk.[2-3][0-9].ffn_.*_exps.weight=CPU"]

    # Extract with auto-restart on daemon crash
    prompt_idx = args.resume_from
    n_written = existing
    n_failed = 0
    n_restarts = 0
    t0 = time.time()
    total_tokens = 0

    # Use a mutable state dict so _extract_batch can update progress
    # even when it raises an exception (daemon crash)
    state = {
        "prompt_idx": prompt_idx,
        "start_idx": prompt_idx,
        "n_written": n_written,
        "n_failed": n_failed,
        "total_tokens": total_tokens,
    }

    daemon = None
    try:
        while state["prompt_idx"] < len(prompts) and n_restarts <= MAX_RESTARTS:
            # Start/restart daemon
            if daemon is None:
                print(f"Starting target daemon (restart #{n_restarts})...", flush=True)
                time.sleep(3 if n_restarts > 0 else 0)  # cooldown after crash
                daemon = make_daemon(args, extra_args)

            try:
                _extract_batch(
                    daemon, prompts, output_dir, state,
                    args.max_seq_len, t0,
                )
                # If _extract_batch returns normally, we're done
                break

            except (ConnectionError, BrokenPipeError, OSError, TimeoutError) as e:
                n_restarts += 1
                state["n_failed"] += 1
                # Skip the problematic prompt (state already updated by _extract_batch)
                state["prompt_idx"] += 1
                print(f"\n  DAEMON CRASHED at prompt {state['prompt_idx'] - 1}: {e}", flush=True)
                print(f"  Skipping prompt, restarting daemon "
                      f"({n_restarts}/{MAX_RESTARTS})...", flush=True)
                # Clean up dead daemon
                try:
                    daemon.close()
                except Exception:
                    pass
                daemon = None

    except KeyboardInterrupt:
        print("\nInterrupted! Samples written so far are valid.", flush=True)
    finally:
        if daemon is not None:
            print("Stopping daemon...", flush=True)
            try:
                daemon.close()
            except Exception:
                pass

    # Stats
    elapsed = time.time() - t0
    n_samples = len(list(output_dir.glob("sample_*")))
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"\n{'='*60}", flush=True)
    print(f" Extraction Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Samples:    {n_samples}", flush=True)
    print(f"  Total size: {total_size / 1e6:.1f} MB", flush=True)
    print(f"  Per sample: {total_size / max(1, n_samples) / 1024:.1f} KB", flush=True)
    print(f"  Failed:     {state['n_failed']}", flush=True)
    print(f"  Restarts:   {n_restarts}", flush=True)
    print(f"  Time:       {elapsed/3600:.1f}h", flush=True)
    print(f"{'='*60}", flush=True)


def _extract_batch(daemon, prompts, output_dir, state,
                   max_seq_len, t0):
    """Extract from state['prompt_idx'] until end or daemon crash.

    Updates state dict in-place (prompt_idx, n_written, n_failed, total_tokens).
    Raises ConnectionError/BrokenPipeError on daemon crash.
    """
    for i in range(state["prompt_idx"], len(prompts)):
        state["prompt_idx"] = i  # Track progress for crash recovery
        prompt_data = prompts[i]
        prompt = prompt_data["prompt"]
        response = prompt_data.get("response", "")

        full_text = prompt
        if response:
            full_text = prompt + "\n" + response

        # Progress reporting
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1 - state.get("start_idx", 0)) / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(prompts)}] "
                  f"{rate:.1f} prompts/s | "
                  f"{state['total_tokens'] / elapsed:.0f} tok/s | "
                  f"written={state['n_written']} | failed={state['n_failed']} | "
                  f"ETA={eta/3600:.1f}h", flush=True)

        try:
            # Tokenize — can raise ConnectionError if daemon died
            tokens = daemon.tokenize(full_text)
            seq_len = min(len(tokens), max_seq_len)
            tokens = tokens[:seq_len]
            state["total_tokens"] += seq_len

            if seq_len < BLOCK_SIZE + 2:
                state["n_failed"] += 1
                continue

            anchor_pos = seq_len - BLOCK_SIZE - 1
            if anchor_pos < 0:
                state["n_failed"] += 1
                continue

            # Forward pass — last position only
            eval_tokens = tokens[:anchor_pos + 1]
            hidden_states, _ = daemon.eval_last_pos(eval_tokens)

            anchor_hidden = hidden_states[:, 0, :]
            if np.any(np.isnan(anchor_hidden)) or np.any(np.isinf(anchor_hidden)):
                state["n_failed"] += 1
                continue

            # Save
            out_dir = output_dir / f"sample_{state['n_written']:06d}"
            out_dir.mkdir(exist_ok=True)

            anchor_hidden.astype(np.float16).tofile(out_dir / "anchor_hidden.bin")

            block_tokens = np.array(
                tokens[anchor_pos + 1 : anchor_pos + 1 + BLOCK_SIZE],
                dtype=np.int32,
            )
            if len(block_tokens) < BLOCK_SIZE:
                pad_len = BLOCK_SIZE - len(block_tokens)
                block_tokens = np.pad(block_tokens, (0, pad_len),
                                      constant_values=MASK_TOKEN_ID)
            block_tokens.tofile(out_dir / "block_tokens.bin")

            meta = {
                "anchor_pos": anchor_pos,
                "seq_len": seq_len,
                "block_size": BLOCK_SIZE,
                "layers": LAYERS,
                "hidden_dim": HIDDEN_DIM,
                "dtype": "float16",
                "source_id": prompt_data.get("id", f"prompt_{i}"),
            }
            with open(out_dir / "metadata.json", "w") as f:
                json.dump(meta, f)

            state["n_written"] += 1

        except (ConnectionError, BrokenPipeError, OSError, TimeoutError):
            # Daemon crashed — propagate up for restart
            raise
        except Exception as e:
            state["n_failed"] += 1
            if state["n_failed"] <= 20:
                print(f"  WARN sample {i}: {e}", flush=True)

    state["prompt_idx"] = len(prompts)  # Mark all done


if __name__ == "__main__":
    main()
