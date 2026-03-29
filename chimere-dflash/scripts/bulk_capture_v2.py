#!/usr/bin/env python3
"""
Bulk capture v2 — Optimized with eval_incr + trim_kv (O(n) instead of O(n²)).

Key optimizations vs v1:
  1. eval_incr + trim_kv instead of clear_kv + eval_full (~3-5× faster)
  2. Pre-allocated hidden state buffer (no np.concatenate each cycle)
  3. Batched buffer state saves (every 100 instead of every 1)

Verification protocol with eval_incr:
  - After prefill of prompt (eval_full), KV has positions [0..prompt_len-1]
  - eval_incr(draft_tokens) adds KV at positions [prompt_len..prompt_len+K-2]
  - argmax[j] = target prediction AFTER seeing draft_tokens[j]
  - For verification: draft_tokens[0] must match prefill_argmax (last prefill prediction)
                      draft_tokens[j] must match argmax[j-1] for j>=1
  - On partial accept (n_accepted < K-1):
      trim_kv(prompt_len + n_accepted) to remove rejected draft positions
      eval_incr([correction_token]) to add the correction
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.modeling_v8 import DFlashDraftModelV8
from chimere.online_buffer import OnlineBuffer
from chimere.target_daemon import TargetDaemon


def load_drafter(checkpoint_path: str, device: torch.device):
    from dataclasses import fields
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DFlashV7Config(**{
        f.name: ckpt["config"][f.name]
        for f in fields(DFlashV7Config)
        if f.name in ckpt["config"]
    })
    model = DFlashDraftModelV8(config)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval().float()
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Bulk capture v2 (optimized)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v8_online/best.pt")
    parser.add_argument("--model", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"))
    parser.add_argument("--daemon", type=str, default="extract/build/target_daemon")
    parser.add_argument("--prompts", type=str, default="data/bootstrap_prompts.jsonl")
    parser.add_argument("--buffer-dir", type=str, default="data/online_buffer_merged")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--drafter-device", type=str, default="cpu")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True,
                        help="Adaptive block size: reduce K on low τ, increase on high τ (default: True)")
    parser.add_argument("--full-gpu", action="store_true",
                        help="Full GPU mode (no CPU offload) — use for IQ3_S or other small quants")
    parser.add_argument("--max-prompts", type=int, default=0,
                        help="Max prompts to process (0 = all remaining after start-idx)")
    args = parser.parse_args()

    drafter_device = torch.device(args.drafter_device)
    layers = [int(x) for x in args.layers.split(",")]

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["text"])
    prompts = prompts[args.start_idx:]
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    print(f"Loaded {len(prompts)} prompts (from idx {args.start_idx})")

    # Load drafter
    print(f"Loading drafter on {drafter_device}...", flush=True)
    drafter, config = load_drafter(args.checkpoint, drafter_device)
    K = config.block_size
    n_layers = config.num_feature_layers
    H = config.target_hidden_size

    # Launch target daemon
    print("Launching target daemon...", flush=True)
    extra_daemon = [
        "-c", "512",
        "-t", "6",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q4_0",
    ]
    if not args.full_gpu:
        extra_daemon += ["-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU"]
    target = TargetDaemon(
        daemon_path=args.daemon,
        model_path=args.model,
        layers=layers,
        n_gpu_layers=99,
        extra_args=extra_daemon,
    )
    test_tokens = target.tokenize("Hello")
    print(f"Daemon ready (test: {test_tokens})")

    # Setup buffer
    buffer = OnlineBuffer(args.buffer_dir, capacity=20000, layers=layers, hidden_dim=H)
    initial_size = buffer.size
    print(f"Buffer: {buffer.size}/{buffer.capacity}")

    total_accepted = 0
    total_drafted = 0
    total_generated = 0
    t_global = time.time()

    # Pre-allocate hidden state buffer (avoid np.concatenate each cycle)
    max_seq = 600  # prompt + max_tokens + margin
    hidden_buf = np.zeros((n_layers, max_seq, H), dtype=np.float32)

    for i, prompt in enumerate(prompts):
        t_start = time.time()
        tokens = target.tokenize(prompt)
        prompt_len = len(tokens)

        # === PREFILL: eval_full for the prompt ===
        prefill_hidden, prefill_argmax = target.eval_full(tokens)
        # KV cache now has positions [0..prompt_len-1]
        # prefill_argmax[j] = target prediction after seeing tokens[0..j]
        # So prefill_argmax[-1] = next token prediction after full prompt

        # Copy into pre-allocated buffer
        seq_pos = prompt_len
        if seq_pos > max_seq:
            hidden_buf = np.zeros((n_layers, seq_pos + args.max_tokens + 32, H), dtype=np.float32)
            max_seq = hidden_buf.shape[1]
        hidden_buf[:, :prompt_len, :] = prefill_hidden

        # The last prefill argmax = target's prediction for token after prompt
        last_target_pred = int(prefill_argmax[-1])

        generated = []
        prompt_accepted = 0
        prompt_drafted = 0
        recent_tau = []  # per-prompt cycle τ history (last 3 cycles) for adaptive K

        while len(generated) < args.max_tokens:
            # Prepare drafter context from hidden_buf
            ctx_end = seq_pos
            ctx_start = max(0, ctx_end - config.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            hidden_list = [
                torch.from_numpy(hidden_buf[j, ctx_start:ctx_end].copy())
                .unsqueeze(0).to(drafter_device)
                for j in range(n_layers)
            ]
            ctx_lengths = torch.tensor([ctx_len], device=drafter_device, dtype=torch.long)
            anchor_pos = ctx_end - 1
            anchor_positions = torch.tensor([anchor_pos], device=drafter_device, dtype=torch.long)

            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            # Draft K-1 tokens
            draft_ids, _, _ = drafter.generate_block(
                hidden_list, context_lengths=ctx_lengths,
                temperature=0.0,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            draft_tokens = draft_ids[0].cpu().tolist()

            # Adaptive block size: truncate draft_tokens based on recent τ history
            if args.adaptive and recent_tau:
                avg_recent_tau = sum(recent_tau) / len(recent_tau)
                if avg_recent_tau < 0.20:
                    adaptive_k = 4
                elif avg_recent_tau > 0.50:
                    adaptive_k = K - 1  # full K-1 = config.block_size - 1 = 15
                else:
                    adaptive_k = 8
                draft_tokens = draft_tokens[:adaptive_k]

            # Save state before verification (for restore on rejection)
            target.save_state()

            # === VERIFY with eval_incr (O(K) instead of O(prompt+gen+K)) ===
            incr_hidden, incr_argmax = target.eval_incr(draft_tokens)
            # incr_hidden: [n_layers, K-1, H] — hidden states for draft positions
            # incr_argmax[j] = target argmax AFTER seeing draft_tokens[j]

            # Verification:
            #   draft_tokens[0] should match last_target_pred (from previous cycle)
            #   draft_tokens[j] should match incr_argmax[j-1] for j >= 1
            target_preds = [last_target_pred] + incr_argmax[:-1].tolist()

            n_accepted = 0
            accepted_mask = []
            for j in range(len(draft_tokens)):
                if j < len(target_preds) and draft_tokens[j] == target_preds[j]:
                    n_accepted += 1
                    accepted_mask.append(True)
                else:
                    accepted_mask.append(False)
                    break
            while len(accepted_mask) < len(draft_tokens):
                accepted_mask.append(False)

            # Add accepted tokens to generated
            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            # Handle correction/bonus — defer hidden state writes until outcome known
            if n_accepted < len(draft_tokens):
                # Partial accept — correction = what target wanted at rejected position
                correction = target_preds[n_accepted]
                generated.append(correction)

                # Restore state, then eval_incr accepted + correction in one go
                target.restore_state()
                accepted_plus_correction = draft_tokens[:n_accepted] + [correction]
                re_hidden, re_argmax = target.eval_incr(accepted_plus_correction)

                # Write re-evaluated hidden states (accepted + correction)
                n_new = len(accepted_plus_correction)
                if seq_pos + n_new >= max_seq:
                    new_max = seq_pos + n_new + args.max_tokens + 32
                    new_buf = np.zeros((n_layers, new_max, H), dtype=np.float32)
                    new_buf[:, :seq_pos, :] = hidden_buf[:, :seq_pos, :]
                    hidden_buf = new_buf
                    max_seq = new_max
                hidden_buf[:, seq_pos:seq_pos + n_new, :] = re_hidden[:, :n_new, :]
                seq_pos += n_new
                last_target_pred = int(re_argmax[-1])
            else:
                # All accepted — write initial hidden states, then bonus
                n_new_hidden = min(n_accepted, incr_hidden.shape[1])
                if seq_pos + n_new_hidden + 1 >= max_seq:
                    new_max = seq_pos + n_new_hidden + args.max_tokens + 32
                    new_buf = np.zeros((n_layers, new_max, H), dtype=np.float32)
                    new_buf[:, :seq_pos, :] = hidden_buf[:, :seq_pos, :]
                    hidden_buf = new_buf
                    max_seq = new_max
                hidden_buf[:, seq_pos:seq_pos + n_new_hidden, :] = incr_hidden[:, :n_new_hidden, :]
                seq_pos += n_new_hidden

                bonus = int(incr_argmax[-1])
                generated.append(bonus)
                bonus_hidden, bonus_argmax = target.eval_incr([bonus])
                hidden_buf[:, seq_pos:seq_pos + 1, :] = bonus_hidden
                seq_pos += 1
                last_target_pred = int(bonus_argmax[0])

            prompt_accepted += n_accepted
            prompt_drafted += len(draft_tokens)

            # Track per-cycle τ for adaptive K (keep last 3 cycles)
            cycle_tau = n_accepted / max(1, len(draft_tokens))
            recent_tau.append(cycle_tau)
            if len(recent_tau) > 3:
                recent_tau.pop(0)

            if generated and generated[-1] in (151643, 151645):
                break

        # Store once per prompt (not per cycle — avoids ~8x redundant data)
        if generated:
            full_tokens = np.array(tokens + generated, dtype=np.int32)
            buffer.store(
                hidden_states=hidden_buf[:, :seq_pos, :],
                tokens=full_tokens,
                draft_tokens=[],
                target_tokens=[],
                accepted_mask=[],
                anchor_pos=seq_pos - 1,
                source="bulk_v2",
            )

        # Clear KV for next prompt
        target.clear_kv()

        elapsed = time.time() - t_start
        prompt_tau = prompt_accepted / max(1, prompt_drafted)
        total_accepted += prompt_accepted
        total_drafted += prompt_drafted
        total_generated += len(generated)
        running_tau = total_accepted / max(1, total_drafted)

        idx = args.start_idx + i
        print(f"  [{idx+1:3d}/{args.start_idx+len(prompts)}] "
              f"τ={prompt_tau:5.1%} ({prompt_accepted:2d}/{prompt_drafted:2d}) "
              f"| gen={len(generated):3d} "
              f"| {elapsed:5.1f}s "
              f"| running_τ={running_tau:5.1%} "
              f"| buf={buffer.size}", flush=True)

    # Flush buffer state
    buffer.flush()

    elapsed_total = time.time() - t_global
    new_samples = buffer.size - initial_size
    overall_tau = total_accepted / max(1, total_drafted)

    print(f"\n{'='*60}")
    print(f" Bulk Capture v2 Complete")
    print(f"{'='*60}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  New samples: {new_samples}")
    print(f"  Buffer:      {buffer.size}")
    print(f"  τ:           {overall_tau:.2%}")
    print(f"  Generated:   {total_generated} tokens")
    print(f"  Time:        {elapsed_total:.0f}s ({elapsed_total/max(1,len(prompts)):.1f}s/prompt)")
    print(f"{'='*60}")

    target.close()


if __name__ == "__main__":
    main()
