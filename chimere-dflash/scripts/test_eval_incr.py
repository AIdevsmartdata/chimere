#!/usr/bin/env python3
"""Quick test: verify eval_incr + trim_kv gives same results as clear_kv + eval_full."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chimere.target_daemon import TargetDaemon

daemon = TargetDaemon(
    daemon_path="extract/build/target_daemon",
    model_path=str(Path.home() / ".chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q5_K_XL.gguf"),
    layers=[1, 10, 19, 28, 37],
    n_gpu_layers=99,
    extra_args=[
        "-c", "512", "-t", "6",
        "-ot", r"blk\.[2-3][0-9]\.ffn_.*_exps\.weight=CPU",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q4_0",
    ],
)

# Test prompt
tokens = daemon.tokenize("The quick brown fox jumps over the lazy dog.")
print(f"Prompt: {len(tokens)} tokens")

# Method 1: eval_full for prompt + 5 extra tokens (greedy)
print("\n=== Method 1: eval_full (all at once) ===")
daemon.clear_kv()
# First get the prompt predictions
h_full, argmax_full = daemon.eval_full(tokens)
print(f"Prefill argmax[-1] = {argmax_full[-1]}")

# Generate 5 tokens greedily using eval_full
gen_tokens_full = []
for step in range(5):
    next_tok = int(argmax_full[-1]) if not gen_tokens_full else gen_tokens_full[-1]
    if step == 0:
        next_tok = int(argmax_full[-1])
    gen_tokens_full.append(next_tok)
    daemon.clear_kv()
    h_full, argmax_full = daemon.eval_full(tokens + gen_tokens_full)
    gen_tokens_full[-1] = next_tok  # keep the one we chose
    # Actually we need the argmax after seeing this token
    print(f"  Step {step}: tok={next_tok}, next_pred={argmax_full[-1]}")

print(f"Generated (full): {gen_tokens_full}")
print(f"Text: {daemon.detokenize(gen_tokens_full)}")

# Method 2: eval_full for prefill, then eval_incr for each token
print("\n=== Method 2: eval_full prefill + eval_incr ===")
daemon.clear_kv()
h_pre, argmax_pre = daemon.eval_full(tokens)
print(f"Prefill argmax[-1] = {argmax_pre[-1]}")

gen_tokens_incr = []
last_pred = int(argmax_pre[-1])
for step in range(5):
    tok = last_pred
    gen_tokens_incr.append(tok)
    h_incr, argmax_incr = daemon.eval_incr([tok])
    last_pred = int(argmax_incr[0])
    print(f"  Step {step}: tok={tok}, next_pred={last_pred}")

print(f"Generated (incr): {gen_tokens_incr}")
print(f"Text: {daemon.detokenize(gen_tokens_incr)}")

# Method 3: eval_full prefill, then eval_incr with multiple tokens, then trim
print("\n=== Method 3: eval_full prefill + eval_incr(batch) + trim_kv ===")
daemon.clear_kv()
h_pre3, argmax_pre3 = daemon.eval_full(tokens)
first_pred = int(argmax_pre3[-1])

# Simulate drafting 5 tokens (use the ones from method 2 as "draft")
draft = gen_tokens_incr[:5]
print(f"Draft tokens: {draft}")
h_batch, argmax_batch = daemon.eval_incr(draft)
print(f"Batch argmax: {argmax_batch.tolist()}")

# Verify: draft[0] should match first_pred
# draft[j] should match argmax_batch[j-1] for j>=1
target_preds = [first_pred] + argmax_batch[:-1].tolist()
print(f"Target preds: {target_preds}")
n_match = 0
for j in range(len(draft)):
    match = draft[j] == target_preds[j]
    print(f"  pos {j}: draft={draft[j]} target={target_preds[j]} {'✓' if match else '✗'}")
    if match:
        n_match += 1
    else:
        break
print(f"Accepted: {n_match}/{len(draft)}")

# Now test trim_kv: trim to prompt + 3 accepted, then incr from there
if n_match >= 3:
    trim_to = len(tokens) + 3
    print(f"\nTrimming KV to {trim_to}...")
    daemon.trim_kv(trim_to)
    # Add token at position 3
    next_tok = gen_tokens_incr[3] if len(gen_tokens_incr) > 3 else first_pred
    h_after_trim, argmax_after_trim = daemon.eval_incr([next_tok])
    print(f"After trim+incr: argmax = {argmax_after_trim[0]}")

    # Compare with method 2's prediction at same position
    print(f"Method 2 pred at same pos: {gen_tokens_incr[4] if len(gen_tokens_incr) > 4 else 'N/A'}")

print("\n=== Consistency check ===")
print(f"Method 1 tokens: {gen_tokens_full}")
print(f"Method 2 tokens: {gen_tokens_incr}")
match = gen_tokens_full == gen_tokens_incr
print(f"Match: {'YES ✓' if match else 'NO ✗'}")

daemon.close()
print("\nDone.")
