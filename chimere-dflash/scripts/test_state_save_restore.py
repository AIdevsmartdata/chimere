#!/usr/bin/env python3
"""Test state save/restore: verify it produces identical results to clear_kv+eval_full."""
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

tokens = daemon.tokenize("The quick brown fox jumps over the lazy dog.")
print(f"Prompt: {len(tokens)} tokens")

# === Test 1: save/restore produces same next-token as continuing from checkpoint ===
print("\n=== Test 1: Save, eval draft, restore, re-eval ===")

# Prefill
daemon.clear_kv()
h_pre, argmax_pre = daemon.eval_full(tokens)
first_pred = int(argmax_pre[-1])
print(f"After prefill: next_pred = {first_pred}")

# Save state at this point
state_size = daemon.save_state()
print(f"State saved: {state_size} bytes ({state_size / 1024 / 1024:.1f} MB)")

# Eval 5 tokens incrementally (simulate draft verification)
draft_tokens = [first_pred, 100, 200, 300, 400]
h_draft, argmax_draft = daemon.eval_incr(draft_tokens)
print(f"After eval_incr({draft_tokens}): argmax = {argmax_draft.tolist()}")

# Restore state
daemon.restore_state()
print("State restored")

# Now eval_incr with just the first token — should match what we'd get from a fresh start
h_restored, argmax_restored = daemon.eval_incr([first_pred])
print(f"After restore + eval_incr([{first_pred}]): argmax = {argmax_restored[0]}")

# === Test 2: Compare against clear_kv + eval_full (ground truth) ===
print("\n=== Test 2: Compare with clear_kv + eval_full ===")
daemon.clear_kv()
h_full, argmax_full = daemon.eval_full(tokens + [first_pred])
full_next = int(argmax_full[-1])
print(f"clear_kv + eval_full({len(tokens)+1} tokens): next_pred = {full_next}")
print(f"restore + eval_incr: next_pred = {int(argmax_restored[0])}")
match1 = full_next == int(argmax_restored[0])
print(f"Match: {'YES' if match1 else 'NO'}")

# === Test 3: Speculative decode cycle simulation ===
print("\n=== Test 3: Full spec decode simulation ===")
daemon.clear_kv()
h_pre, argmax_pre = daemon.eval_full(tokens)
pred = int(argmax_pre[-1])

# Generate 3 tokens via save/restore cycle
generated = []
for step in range(3):
    # Save before "draft"
    daemon.save_state()

    # Simulate accepted draft (just use the correct token for 100% accept)
    h_incr, argmax_incr = daemon.eval_incr([pred])
    generated.append(pred)
    pred = int(argmax_incr[0])

print(f"Generated: {generated}")
print(f"Text: {daemon.detokenize(generated)}")

# Compare with autoregressive
daemon.clear_kv()
h_pre2, argmax_pre2 = daemon.eval_full(tokens)
gen_ar = []
pred_ar = int(argmax_pre2[-1])
for step in range(3):
    gen_ar.append(pred_ar)
    h_ar, argmax_ar = daemon.eval_incr([pred_ar])
    pred_ar = int(argmax_ar[0])

print(f"Autoregressive: {gen_ar}")
match2 = generated == gen_ar
print(f"Match: {'YES' if match2 else 'NO'}")

# === Test 4: Restore after rejected draft ===
print("\n=== Test 4: Save, eval wrong draft, restore, eval correct ===")
daemon.clear_kv()
h_pre3, argmax_pre3 = daemon.eval_full(tokens)
correct_pred = int(argmax_pre3[-1])

daemon.save_state()

# Eval wrong draft tokens
wrong_draft = [99999, 88888, 77777]  # Definitely wrong
h_wrong, argmax_wrong = daemon.eval_incr(wrong_draft)
print(f"After wrong draft: argmax = {argmax_wrong.tolist()}")

# Restore and eval correct token
daemon.restore_state()
h_correct, argmax_correct = daemon.eval_incr([correct_pred])
correct_next = int(argmax_correct[0])

# Compare with fresh eval_full
daemon.clear_kv()
h_ref, argmax_ref = daemon.eval_full(tokens + [correct_pred])
ref_next = int(argmax_ref[-1])

print(f"After restore + correct: next = {correct_next}")
print(f"Reference (eval_full):   next = {ref_next}")
match3 = correct_next == ref_next
print(f"Match: {'YES' if match3 else 'NO'}")

print(f"\n{'='*40}")
all_pass = match1 and match2 and match3
print(f"ALL TESTS: {'PASS' if all_pass else 'FAIL'}")
print(f"{'='*40}")

daemon.close()
