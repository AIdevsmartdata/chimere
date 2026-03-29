#!/usr/bin/env python3
"""
Bulk capture: run spec decode on many prompts, store experiences in buffer.
Uses the correct verification logic from test_online_spec_decode.py.

Usage:
  python scripts/bulk_capture.py \
    --checkpoint checkpoints_v8_online/best.pt \
    --prompts data/bootstrap_prompts.jsonl \
    --buffer-dir data/online_buffer_merged \
    --max-tokens 64
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
    parser = argparse.ArgumentParser(description="Bulk capture spec decode experiences")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v8_online/best.pt")
    parser.add_argument("--model", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"))
    parser.add_argument("--daemon", type=str, default="extract/build/target_daemon")
    parser.add_argument("--prompts", type=str, default="data/bootstrap_prompts.jsonl")
    parser.add_argument("--buffer-dir", type=str, default="data/online_buffer_merged")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--drafter-device", type=str, default="cpu")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Resume from prompt index")
    args = parser.parse_args()

    drafter_device = torch.device(args.drafter_device)
    layers = [int(x) for x in args.layers.split(",")]

    # Load prompts
    prompts = []
    with open(args.prompts) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                prompts.append(obj["text"])
    prompts = prompts[args.start_idx:]
    print(f"Loaded {len(prompts)} prompts (starting from idx {args.start_idx})")

    # Load drafter
    print(f"Loading drafter on {drafter_device}...", flush=True)
    drafter, config = load_drafter(args.checkpoint, drafter_device)
    K = config.block_size
    n_layers = config.num_feature_layers
    H = config.target_hidden_size

    # Launch target daemon
    print("Launching target daemon...", flush=True)
    target = TargetDaemon(
        daemon_path=args.daemon,
        model_path=args.model,
        layers=layers,
        n_gpu_layers=99,
        extra_args=[
            "-c", "512",
            "-t", "6",
            "-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU",
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q4_0",
        ],
    )

    test_tokens = target.tokenize("Hello")
    print(f"Daemon ready (test tokenize: {test_tokens})")

    # Setup buffer
    buffer = OnlineBuffer(args.buffer_dir, capacity=20000, layers=layers, hidden_dim=H)
    initial_size = buffer.size
    print(f"Buffer: {args.buffer_dir} (current: {initial_size}, capacity: {buffer.capacity})")

    total_accepted = 0
    total_drafted = 0
    total_generated = 0
    t_global = time.time()

    for i, prompt in enumerate(prompts):
        t_start = time.time()
        tokens = target.tokenize(prompt)
        hidden_all, _ = target.eval_full(tokens)

        generated = []
        prompt_accepted = 0
        prompt_drafted = 0

        while len(generated) < args.max_tokens:
            seq_len = hidden_all.shape[1]
            ctx_end = seq_len
            ctx_start = max(0, ctx_end - config.max_ctx_len)
            ctx_len = ctx_end - ctx_start

            hidden_list = [
                torch.from_numpy(hidden_all[j, ctx_start:ctx_end].copy())
                .unsqueeze(0).to(drafter_device)
                for j in range(n_layers)
            ]
            ctx_lengths = torch.tensor([ctx_len], device=drafter_device, dtype=torch.long)
            anchor_pos = ctx_end - 1
            anchor_positions = torch.tensor([anchor_pos], device=drafter_device, dtype=torch.long)

            all_tokens = tokens + generated
            anchor_token_id = all_tokens[-1]

            # Draft
            draft_ids, _, _ = drafter.generate_block(
                hidden_list, context_lengths=ctx_lengths,
                temperature=0.0,
                anchor_token_id=anchor_token_id,
                anchor_positions=anchor_positions,
            )
            draft_tokens = draft_ids[0].cpu().tolist()

            # Verify: full eval
            verify_tokens = all_tokens + draft_tokens
            target.clear_kv()
            target_hidden, target_argmax = target.eval_full(verify_tokens)

            base = len(all_tokens) - 1
            target_preds = target_argmax[base:base + len(draft_tokens)].tolist()

            # Sequential accept
            n_accepted = 0
            accepted_mask = []
            for j in range(min(len(draft_tokens), len(target_preds))):
                if draft_tokens[j] == target_preds[j]:
                    n_accepted += 1
                    accepted_mask.append(True)
                else:
                    accepted_mask.append(False)
                    break
            while len(accepted_mask) < len(draft_tokens):
                accepted_mask.append(False)

            for j in range(n_accepted):
                generated.append(draft_tokens[j])
            if n_accepted < len(draft_tokens) and n_accepted < len(target_preds):
                generated.append(target_preds[n_accepted])
            elif n_accepted == len(draft_tokens):
                bonus_pos = base + len(draft_tokens)
                if bonus_pos < len(target_argmax):
                    generated.append(int(target_argmax[bonus_pos]))

            n_keep = len(tokens) + len(generated)
            hidden_all = target_hidden[:, :n_keep, :]

            # Store experience
            full_tokens = np.array(tokens + generated, dtype=np.int32)
            buffer.store(
                hidden_states=hidden_all,
                tokens=full_tokens,
                draft_tokens=draft_tokens,
                target_tokens=target_preds[:len(draft_tokens)],
                accepted_mask=accepted_mask,
                anchor_pos=anchor_pos,
                source="bulk_capture",
            )

            prompt_accepted += n_accepted
            prompt_drafted += len(draft_tokens)

            if generated and generated[-1] in (151643, 151645):
                break

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

    elapsed_total = time.time() - t_global
    new_samples = buffer.size - initial_size
    overall_tau = total_accepted / max(1, total_drafted)

    print(f"\n{'='*60}")
    print(f" Bulk Capture Complete")
    print(f"{'='*60}")
    print(f"  Prompts processed: {len(prompts)}")
    print(f"  New samples:       {new_samples}")
    print(f"  Total buffer:      {buffer.size}")
    print(f"  Overall τ:         {overall_tau:.2%}")
    print(f"  Total generated:   {total_generated} tokens")
    print(f"  Total time:        {elapsed_total:.0f}s ({elapsed_total/len(prompts):.1f}s/prompt)")
    print(f"{'='*60}")

    target.close()


if __name__ == "__main__":
    main()
