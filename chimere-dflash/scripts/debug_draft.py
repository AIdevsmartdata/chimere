#!/usr/bin/env python3
"""Debug: profile draft_time, compare eval_full vs eval_incr hidden states,
and show draft vs target token mismatch."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from chimere.config import DFlashConfig
from chimere.modeling_v5 import DFlashDraftModelV5
from chimere.target_daemon import TargetDaemon
from dataclasses import fields

DAEMON_BIN = "extract/build/target_daemon"
MODEL_PATH = os.path.expanduser(
    "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE.gguf"
)
TARGET_LAYERS = [2, 11, 20, 29, 37]
CHECKPOINT = "checkpoints_v5_daemon/best.pt"
EXTRA_ARGS = ["-ot", ".ffn_.*_exps.=CPU", "--flash-attn", "on"]

device = torch.device("cuda")

# Load drafter
print("Loading drafter...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
config = DFlashConfig(**{
    f.name: ckpt["config"][f.name]
    for f in fields(DFlashConfig)
    if f.name in ckpt["config"]
})
drafter = DFlashDraftModelV5(config).to(device).eval()
drafter.load_state_dict(ckpt["model_state_dict"])
# Cast frozen parts to BF16
for p in drafter.embed_tokens.parameters():
    p.data = p.data.to(torch.bfloat16)
for p in drafter.lm_head.parameters():
    p.data = p.data.to(torch.bfloat16)
print(f"Drafter loaded on GPU: {sum(p.numel() for p in drafter.parameters())/1e6:.0f}M params")

# Load daemon
print("Loading daemon...")
daemon = TargetDaemon(DAEMON_BIN, MODEL_PATH, TARGET_LAYERS, extra_args=EXTRA_ARGS)

prompt = "Explain the difference between a stack and a queue in computer science."
prompt_tokens = daemon.tokenize(prompt)
print(f"Prompt: {len(prompt_tokens)} tokens\n")

# ─── TEST 1: Profile draft time per step ───────────────────────────
print("=" * 60)
print("TEST 1: Profile generate_block vs generate_block_multistep")
print("=" * 60)

hidden, logits = daemon.eval_full(prompt_tokens)
anchor_id = int(logits[-1])

ctx = hidden  # [n_layers, seq_len, hidden_dim]
hidden_list = [
    torch.from_numpy(ctx[i].copy()).unsqueeze(0).to(device, dtype=torch.float32)
    for i in range(5)
]
ctx_lengths = torch.tensor([ctx.shape[1]], device=device, dtype=torch.long)

with torch.no_grad():
    # Warmup
    drafter.generate_block(hidden_list, context_lengths=ctx_lengths,
                           temperature=0.0, anchor_token_id=anchor_id)
    torch.cuda.synchronize()

    # Time single-step
    torch.cuda.synchronize()
    t0 = time.time()
    draft_ids_1, logits_1, _ = drafter.generate_block(
        hidden_list, context_lengths=ctx_lengths,
        temperature=0.0, anchor_token_id=anchor_id)
    torch.cuda.synchronize()
    t1_single = (time.time() - t0) * 1000

    # Time multi-step 4
    torch.cuda.synchronize()
    t0 = time.time()
    draft_ids_4, _ = drafter.generate_block_multistep(
        hidden_list, context_lengths=ctx_lengths,
        temperature=0.0, anchor_token_id=anchor_id, n_steps=4)
    torch.cuda.synchronize()
    t1_multi = (time.time() - t0) * 1000

    print(f"  Single-step:  {t1_single:.1f} ms")
    print(f"  Multi-step 4: {t1_multi:.1f} ms")
    print(f"  Ratio:        {t1_multi/t1_single:.1f}×")

# ─── TEST 2: eval_full vs eval_incr hidden state comparison ──────
print("\n" + "=" * 60)
print("TEST 2: eval_full(all) vs eval_full(prefix) + eval_incr(suffix)")
print("=" * 60)

# Generate some tokens with target to build a test sequence
daemon.clear_kv()
full_hidden, full_logits = daemon.eval_full(prompt_tokens)
# Generate 16 tokens AR
ar_tokens = []
for _ in range(16):
    tok = int(full_logits[-1])
    ar_tokens.append(tok)
    h, l = daemon.eval_incr([tok])
    full_logits = l

# Now compare: eval_full(prompt + ar_tokens) vs incremental
all_tokens = prompt_tokens + ar_tokens
daemon.clear_kv()
full_hidden_all, _ = daemon.eval_full(all_tokens)

daemon.clear_kv()
incr_hidden_prefix, _ = daemon.eval_full(prompt_tokens)
incr_hidden_suffix, _ = daemon.eval_incr(ar_tokens)

# Compare hidden states at the same positions
print(f"  Full eval shape: {full_hidden_all.shape}")
print(f"  Prefix shape: {incr_hidden_prefix.shape}, Suffix shape: {incr_hidden_suffix.shape}")

# The incremental approach returns only the NEW positions
# So incr_hidden_suffix corresponds to positions [n_prompt, n_prompt+16)
# And full_hidden_all[:, n_prompt:n_prompt+16, :] should match
n_prompt = len(prompt_tokens)
for i, l in enumerate(TARGET_LAYERS):
    h_full = full_hidden_all[i, n_prompt:n_prompt+16, :]
    h_incr = incr_hidden_suffix[i, :16, :]
    # Cosine similarity per position
    cos_sims = []
    for pos in range(16):
        cos = np.dot(h_full[pos], h_incr[pos]) / (
            np.linalg.norm(h_full[pos]) * np.linalg.norm(h_incr[pos]) + 1e-10
        )
        cos_sims.append(cos)
    mean_cos = np.mean(cos_sims)
    min_cos = np.min(cos_sims)
    # L2 distance
    l2 = np.linalg.norm(h_full - h_incr) / np.sqrt(h_full.shape[0])
    print(f"  Layer {l}: cosine mean={mean_cos:.6f}, min={min_cos:.6f}, L2/pos={l2:.6f}")

# ─── TEST 3: Draft vs Target token comparison ────────────────────
print("\n" + "=" * 60)
print("TEST 3: Draft tokens vs Target predictions (5 blocks)")
print("=" * 60)

daemon.clear_kv()
hidden, logits = daemon.eval_full(prompt_tokens)
anchor_id = int(logits[-1])
all_hidden = hidden
generated = []

for block_idx in range(5):
    n_ctx = all_hidden.shape[1]
    ctx_start = max(0, n_ctx - 512)
    ctx = all_hidden[:, ctx_start:, :]

    hidden_list = [
        torch.from_numpy(ctx[i].copy()).unsqueeze(0).to(device, dtype=torch.float32)
        for i in range(5)
    ]
    ctx_lengths = torch.tensor([ctx.shape[1]], device=device, dtype=torch.long)

    with torch.no_grad():
        draft_ids_1, _, _ = drafter.generate_block(
            hidden_list, context_lengths=ctx_lengths,
            temperature=0.0, anchor_token_id=anchor_id)
        draft_ids_4, _ = drafter.generate_block_multistep(
            hidden_list, context_lengths=ctx_lengths,
            temperature=0.0, anchor_token_id=anchor_id, n_steps=4)

    draft_1 = draft_ids_1[0].cpu().tolist()
    draft_4 = draft_ids_4[0].cpu().tolist()

    # Verify
    v_hidden, v_logits = daemon.eval_incr(draft_1)
    target = [int(v_logits[j]) for j in range(len(v_logits))]

    # Sequential acceptance
    n_accept_1 = 0
    for j in range(15):
        if draft_1[j+1] == target[j]:
            n_accept_1 += 1
        else:
            break
    n_accept_4 = 0
    for j in range(15):
        if draft_4[j+1] == target[j]:
            n_accept_4 += 1
        else:
            break
    # Total matches (non-sequential)
    n_match_1 = sum(1 for j in range(15) if draft_1[j+1] == target[j])
    n_match_4 = sum(1 for j in range(15) if draft_4[j+1] == target[j])

    draft_text = daemon.detokenize(draft_1[:8])
    target_text = daemon.detokenize([anchor_id] + target[:7])
    print(f"\n  Block {block_idx} (ctx={n_ctx}): anchor={anchor_id}")
    print(f"    1-step: {n_accept_1} seq_accept, {n_match_1}/15 total | {repr(draft_text[:60])}")
    print(f"    4-step: {n_accept_4} seq_accept, {n_match_4}/15 total")
    print(f"    Target: {repr(target_text[:60])}")
    print(f"    Draft[1:5]:  {draft_1[1:5]}")
    print(f"    Target[0:4]: {target[:4]}")

    # Accept and continue
    if n_accept_1 > 0:
        accepted = draft_1[:n_accept_1 + 1] + [target[n_accept_1]]
    else:
        accepted = [draft_1[0], target[0]]
    generated.extend(accepted)
    prefix = prompt_tokens + generated
    daemon.clear_kv()
    all_hidden, new_logits = daemon.eval_full(prefix)
    anchor_id = int(new_logits[-1])

daemon.quit()
print("\nDone.")
