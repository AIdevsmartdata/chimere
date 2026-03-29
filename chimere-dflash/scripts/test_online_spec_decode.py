#!/usr/bin/env python3
"""
Quick test of online speculative decoding with experience capture.

Launches target_daemon, loads drafter, runs N prompts, measures τ in real-time.
"""
import argparse
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
    """Load drafter from checkpoint.

    If device is CUDA, tries GPU first. Falls back to CPU if VRAM is tight
    (target daemon needs ~13.9 GB, drafter ~3 GB — won't fit in 16 GB together).
    CPU drafter inference is ~15ms vs ~5ms GPU — still fast enough.
    """
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


TEST_PROMPTS = [
    "Explain how speculative decoding works in large language models.",
    "Write a Python function to compute the Fibonacci sequence using memoization.",
    "What are the advantages of MoE (Mixture of Experts) architectures?",
    "Describe the difference between TCP and UDP protocols.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    "The French Revolution began in 1789 when",
    "In Rust, the ownership system ensures memory safety by",
    "SELECT u.name, COUNT(o.id) FROM users u JOIN orders o ON",
    "Les principes fondamentaux de la kinésithérapie reposent sur",
    "Docker containers differ from virtual machines because",
    "The transformer architecture introduced in 'Attention Is All You Need' uses",
    "To configure a reverse proxy with Nginx, you need to",
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self",
    "The Bellman equation in reinforcement learning states that",
    "Pour diagnostiquer une lombalgie chronique, le kinésithérapeute doit",
    "async def fetch_data(url: str) -> dict:\n    async with aiohttp.ClientSession() as session:",
    "The key difference between batch normalization and layer normalization is",
    "En physiothérapie, la méthode McKenzie consiste à",
    "git rebase -i HEAD~5 allows you to",
    "The CAP theorem states that a distributed system cannot simultaneously guarantee",
]


def main():
    parser = argparse.ArgumentParser(description="Test online speculative decoding")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v8_deepkv/best.pt")
    parser.add_argument("--model", type=str,
                        default=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"))
    parser.add_argument("--daemon", type=str, default="extract/build/target_daemon")
    parser.add_argument("--buffer-dir", type=str, default="data/online_buffer")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate per prompt (keep short for testing)")
    parser.add_argument("--layers", type=str, default="1,10,19,28,37")
    parser.add_argument("--drafter-device", type=str, default="cpu",
                        help="Device for drafter (cpu recommended — target needs most of GPU VRAM)")
    args = parser.parse_args()

    drafter_device = torch.device(args.drafter_device)
    layers = [int(x) for x in args.layers.split(",")]

    # Load drafter
    print(f"Loading drafter on {drafter_device}...", flush=True)
    drafter, config = load_drafter(args.checkpoint, drafter_device)
    K = config.block_size
    n_layers = config.num_feature_layers
    H = config.target_hidden_size
    print(f"  Drafter loaded: {config.num_hidden_layers} layers, block_size={K}")

    # Launch target daemon
    print("Launching target daemon (Q5 + selective offload)...", flush=True)
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

    # Wait for daemon to be ready by doing a test tokenize
    print("  Waiting for daemon ready...", flush=True)
    t0 = time.time()
    test_tokens = target.tokenize("Hello")
    print(f"  Daemon ready in {time.time()-t0:.1f}s (test tokenize: {test_tokens})")

    # Setup buffer
    buffer = OnlineBuffer(args.buffer_dir, capacity=10000, layers=layers, hidden_dim=H)
    print(f"  Buffer: {args.buffer_dir} (capacity={buffer.capacity})")

    # Run spec decode on test prompts
    print(f"\n{'='*70}")
    print(f" Online Speculative Decoding Test — {args.n_prompts} prompts")
    print(f"{'='*70}\n")

    total_accepted = 0
    total_drafted = 0
    total_generated = 0
    total_calls = 0
    prompt_times = []

    prompts = TEST_PROMPTS[:args.n_prompts]
    for i, prompt in enumerate(prompts):
        t_start = time.time()

        # Tokenize
        tokens = target.tokenize(prompt)

        # Prefill: get hidden states
        hidden_all, _ = target.eval_full(tokens)
        # hidden_all: [n_layers, seq_len, H] float32

        generated = []
        prompt_accepted = 0
        prompt_drafted = 0
        n_calls = 0

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

            # Verify with target: full eval on prompt+generated+draft
            # (M-RoPE requires strictly increasing positions, so incremental
            #  eval after trim doesn't work — use full eval each cycle)
            verify_tokens = all_tokens + draft_tokens
            target.clear_kv()
            target_hidden, target_argmax = target.eval_full(verify_tokens)
            # target_argmax[i] = argmax prediction AFTER seeing verify_tokens[:i+1]
            # So target_argmax[len(all_tokens)-1] = what target predicts at anchor
            # target_argmax[len(all_tokens)+j-1] = target's prediction after seeing draft[0..j-1]

            # Compare: draft[j] should match target's prediction after seeing up to draft[j-1]
            # target_argmax[len(all_tokens)-1+j] = target's next token after seeing all_tokens + draft[:j]
            base = len(all_tokens) - 1  # position of anchor token
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

            # Add accepted + correction/bonus token
            for j in range(n_accepted):
                generated.append(draft_tokens[j])

            if n_accepted < len(draft_tokens) and n_accepted < len(target_preds):
                generated.append(target_preds[n_accepted])
            elif n_accepted == len(draft_tokens):
                # All accepted — bonus token from target
                bonus_pos = base + len(draft_tokens)
                if bonus_pos < len(target_argmax):
                    generated.append(int(target_argmax[bonus_pos]))

            # Update hidden states: use the verified portion from target
            n_keep = len(tokens) + len(generated)
            hidden_all = target_hidden[:, :n_keep, :]

            # Store in buffer
            full_tokens = np.array(tokens + generated, dtype=np.int32)
            buffer.store(
                hidden_states=hidden_all,
                tokens=full_tokens,
                draft_tokens=draft_tokens,
                target_tokens=target_preds[:len(draft_tokens)],
                accepted_mask=accepted_mask,
                anchor_pos=anchor_pos,
                source="online_test",
            )

            prompt_accepted += n_accepted
            prompt_drafted += len(draft_tokens)
            n_calls += 1

            # Check EOS
            if generated and generated[-1] in (151643, 151645):
                break

        # Clear KV for next prompt
        target.clear_kv()

        elapsed = time.time() - t_start
        prompt_times.append(elapsed)
        prompt_tau = prompt_accepted / max(1, prompt_drafted)

        total_accepted += prompt_accepted
        total_drafted += prompt_drafted
        total_generated += len(generated)
        total_calls += n_calls

        running_tau = total_accepted / max(1, total_drafted)

        # Decode output
        output_text = target.detokenize(generated[:30])  # first 30 tokens
        output_preview = output_text[:80].replace('\n', '\\n')

        print(f"  [{i+1:2d}/{len(prompts)}] τ={prompt_tau:5.1%} "
              f"({prompt_accepted:2d}/{prompt_drafted:2d}) "
              f"| gen={len(generated):3d} tok "
              f"| {n_calls:2d} calls "
              f"| {elapsed:5.1f}s "
              f"| running_τ={running_tau:5.1%}"
              f"\n         → {output_preview}...", flush=True)

    # Final stats
    overall_tau = total_accepted / max(1, total_drafted)
    avg_time = np.mean(prompt_times)
    tok_per_sec = total_generated / sum(prompt_times) if prompt_times else 0

    print(f"\n{'='*70}")
    print(f" RESULTS")
    print(f"{'='*70}")
    print(f"  Prompts:          {len(prompts)}")
    print(f"  Total generated:  {total_generated} tokens")
    print(f"  Total drafted:    {total_drafted}")
    print(f"  Total accepted:   {total_accepted}")
    print(f"  Overall τ:        {overall_tau:.2%}")
    print(f"  Avg accepted/block: {total_accepted/max(1,total_calls):.1f} / {K-1}")
    print(f"  Avg time/prompt:  {avg_time:.1f}s")
    print(f"  Throughput:       {tok_per_sec:.1f} tok/s")
    print(f"  Buffer samples:   {buffer.size}")
    print(f"{'='*70}")

    # Comparison
    print(f"\n  OFFLINE τ (holdout): 6.76%")
    print(f"  ONLINE τ (real):     {overall_tau:.2%}")
    if overall_tau > 0.0676:
        print(f"  → ONLINE IS BETTER (+{(overall_tau-0.0676)*100:.1f}pp)")
    else:
        print(f"  → Same ballpark (need more data for on-policy fine-tune)")

    target.close()
    print("\nDone. Target daemon closed.")


if __name__ == "__main__":
    main()
