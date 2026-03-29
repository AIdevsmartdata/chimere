#!/usr/bin/env python3
"""Quick test: does adding layer-37 residual fix the entropy problem?
No retraining needed — just modify the forward pass."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from chimere.config import DFlashConfig
from chimere.modeling_v5 import DFlashDraftModelV5, build_attention_mask
from dataclasses import fields

CHECKPOINT = "checkpoints_v5_daemon/best.pt"
device = torch.device("cuda")

# Load drafter
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
config = DFlashConfig(**{
    f.name: ckpt["config"][f.name]
    for f in fields(DFlashConfig)
    if f.name in ckpt["config"]
})
drafter = DFlashDraftModelV5(config).to(device).eval()
drafter.load_state_dict(ckpt["model_state_dict"])
for p in drafter.embed_tokens.parameters():
    p.data = p.data.to(torch.bfloat16)
for p in drafter.lm_head.parameters():
    p.data = p.data.to(torch.bfloat16)

# Load a training sample
sample_dir = "data/features_daemon/sample_000000"
import json
meta = json.load(open(f"{sample_dir}/metadata.json"))
tokens = np.fromfile(f"{sample_dir}/input_ids.bin", dtype=np.int32)
seq_len = meta["seq_len"]

TARGET_LAYERS = [2, 11, 20, 29, 37]
layer_hidden = {}
for l in TARGET_LAYERS:
    h = np.fromfile(f"{sample_dir}/layer_{l:02d}.bin", dtype=np.float32).reshape(-1, 2048)
    layer_hidden[l] = h

# Pick anchor at position 50
anchor_pos = 50
anchor_id = int(tokens[anchor_pos])
target_block = tokens[anchor_pos:anchor_pos + 16]

# Build context: positions [0, anchor_pos+1)
ctx_end = anchor_pos + 1
hidden_list = []
for l in TARGET_LAYERS:
    h = torch.from_numpy(layer_hidden[l][:ctx_end].copy()).unsqueeze(0).to(device, dtype=torch.float32)
    hidden_list.append(h)

# Get layer 37 last hidden state for residual
h37_last = hidden_list[-1][:, -1:, :]  # [1, 1, 2048] — last verified position

ctx_lengths = torch.tensor([ctx_end], device=device, dtype=torch.long)

print("=" * 60)
print("TEST: Effect of layer-37 residual on logits quality")
print("=" * 60)

with torch.no_grad():
    B = 1
    K = config.block_size

    # --- Forward pass (same as generate_block) ---
    ctx = drafter._fuse_context(hidden_list)
    ctx_len = ctx.shape[1]

    # Build masked block
    x = drafter.mask_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
    x[:, 0] = drafter.embed_tokens(torch.tensor([anchor_id], device=device))

    pos_ids = torch.arange(K, device=device)
    x = x + drafter.block_pos_embed(pos_ids)[None, :, :]

    mask_ratio = 15.0 / 16.0
    ratio_input = torch.tensor([[mask_ratio]], device=device, dtype=x.dtype)
    ratio_embed = drafter.mask_ratio_proj(ratio_input)
    x = x + ratio_embed.unsqueeze(1)

    attn_mask = build_attention_mask(K, ctx_lengths, ctx_len, device)

    for layer in drafter.layers:
        x = layer(x, context=ctx, attn_mask=attn_mask)

    x_out = drafter.norm(x)

    # --- TEST A: Original (no residual) ---
    logits_orig = drafter.lm_head(x_out.to(drafter.lm_head.weight.dtype))
    probs_orig = F.softmax(logits_orig[0], dim=-1)
    entropy_orig = -(probs_orig * (probs_orig + 1e-10).log()).sum(dim=-1)
    preds_orig = logits_orig[0].argmax(dim=-1)

    # --- TEST B: Add layer-37 residual (broadcast to all K positions) ---
    h37_residual = h37_last.to(x_out.dtype).expand(B, K, -1)
    x_with_res = x_out + h37_residual
    logits_res = drafter.lm_head(x_with_res.to(drafter.lm_head.weight.dtype))
    probs_res = F.softmax(logits_res[0], dim=-1)
    entropy_res = -(probs_res * (probs_res + 1e-10).log()).sum(dim=-1)
    preds_res = logits_res[0].argmax(dim=-1)

    # --- TEST C: Pure layer-37 (no drafter at all, just h37) ---
    x_pure37 = drafter.norm(h37_residual)
    logits_pure = drafter.lm_head(x_pure37.to(drafter.lm_head.weight.dtype))
    probs_pure = F.softmax(logits_pure[0], dim=-1)
    entropy_pure = -(probs_pure * (probs_pure + 1e-10).log()).sum(dim=-1)
    preds_pure = logits_pure[0].argmax(dim=-1)

    # --- TEST D: Init block from h37_last instead of mask_token ---
    x_init = h37_last.to(x.dtype).expand(B, K, -1).clone()
    x_init[:, 0] = drafter.embed_tokens(torch.tensor([anchor_id], device=device))
    x_init = x_init + drafter.block_pos_embed(pos_ids)[None, :, :]
    x_init = x_init + ratio_embed.unsqueeze(1)

    for layer in drafter.layers:
        x_init = layer(x_init, context=ctx, attn_mask=attn_mask)

    x_init_out = drafter.norm(x_init)
    # With residual too
    x_init_res = x_init_out + h37_residual
    logits_initres = drafter.lm_head(x_init_res.to(drafter.lm_head.weight.dtype))
    probs_initres = F.softmax(logits_initres[0], dim=-1)
    entropy_initres = -(probs_initres * (probs_initres + 1e-10).log()).sum(dim=-1)
    preds_initres = logits_initres[0].argmax(dim=-1)

print(f"\nTarget block tokens: {target_block.tolist()}")
print(f"\n{'Mode':<30} {'Mean H':>8} {'Min H':>8} {'Max H':>8} {'Match/15':>10}")
print("-" * 70)

def count_matches(preds, target):
    return sum(1 for i in range(1, 16) if preds[i].item() == target[i])

modes = [
    ("A: Original (no residual)", entropy_orig, preds_orig),
    ("B: + layer37 residual", entropy_res, preds_res),
    ("C: Pure h37 (no drafter)", entropy_pure, preds_pure),
    ("D: Init h37 + residual", entropy_initres, preds_initres),
]

for name, entropy, preds in modes:
    mean_h = entropy[1:].mean().item()  # skip anchor
    min_h = entropy[1:].min().item()
    max_h = entropy[1:].max().item()
    matches = count_matches(preds, target_block)
    print(f"{name:<30} {mean_h:>8.2f} {min_h:>8.2f} {max_h:>8.2f} {matches:>10}")

# Show top-5 predictions for position 1
print(f"\n--- Position 1 detail (target={target_block[1]}) ---")
for name, logits_tensor in [("A: Original", logits_orig), ("B: +Residual", logits_res),
                              ("C: Pure h37", logits_pure), ("D: Init+Res", logits_initres)]:
    top5 = torch.topk(F.softmax(logits_tensor[0, 1], dim=-1), 5)
    print(f"  {name}: top5={top5.indices.cpu().tolist()} probs={[f'{p:.3f}' for p in top5.values.cpu().tolist()]}")

print("\nDone.")
