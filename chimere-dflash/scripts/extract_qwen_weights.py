#!/usr/bin/env python3
"""Extract embed_tokens, lm_head, output_norm from Qwen GGUF.
Uses gguf library's built-in dequantization."""
import os
import numpy as np
import torch
from gguf import GGUFReader

MODEL = os.path.expanduser(
    "~/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE.gguf"
)
OUTPUT = "data/qwen_shared_weights.pt"

reader = GGUFReader(MODEL)
weights = {}

TARGET_TENSORS = {
    "token_embd.weight": "embed_tokens",
    "output.weight": "lm_head",
    "output_norm.weight": "output_norm",
}

for tensor in reader.tensors:
    if tensor.name not in TARGET_TENSORS:
        continue

    key = TARGET_TENSORS[tensor.name]
    print(f"Extracting {tensor.name} -> {key}")
    print(f"  GGUF shape: {tensor.shape}, type: {tensor.tensor_type.name}")
    print(f"  Raw data shape: {tensor.data.shape}, dtype: {tensor.data.dtype}")

    if tensor.tensor_type.name == "F32":
        # F32: reshape directly
        # GGUF stores in row-major, shape is [cols, rows] in GGUF notation
        n_elements = 1
        for s in tensor.shape:
            n_elements *= s
        data = tensor.data[:n_elements].astype(np.float32)
        # GGUF shape [2048] for norm
        t = torch.from_numpy(data)
    elif tensor.tensor_type.name == "Q8_0":
        # Q8_0 block: 2 bytes (f16 scale) + 32 bytes (i8 values) = 34 bytes per 32 values
        # tensor.data is already the raw Q8_0 data as a flat array
        raw = tensor.data.view(np.uint8)

        # Total elements
        rows = tensor.shape[1]  # 248320 (vocab)
        cols = tensor.shape[0]  # 2048 (hidden)
        n_blocks_per_row = cols // 32
        block_bytes = 34  # 2 (f16 scale) + 32 (i8 values)

        print(f"  Dequantizing: {rows}×{cols}, {n_blocks_per_row} blocks/row")

        # Reshape: [rows, n_blocks_per_row, 34]
        expected_bytes = rows * n_blocks_per_row * block_bytes
        print(f"  Expected {expected_bytes} bytes, have {len(raw)} bytes")

        raw = raw[:expected_bytes].reshape(rows, n_blocks_per_row, block_bytes)

        # Extract scales (first 2 bytes of each block) and values (next 32 bytes)
        scales_raw = raw[:, :, :2].reshape(-1, 2)  # [rows*n_blocks, 2]
        scales = np.frombuffer(scales_raw.tobytes(), dtype=np.float16).reshape(rows, n_blocks_per_row)

        values = raw[:, :, 2:].reshape(rows, n_blocks_per_row, 32).view(np.int8)

        # Dequantize: result = scale * value
        result = values.astype(np.float32) * scales[:, :, np.newaxis].astype(np.float32)
        result = result.reshape(rows, cols)

        t = torch.from_numpy(result)
        print(f"  Result: {t.shape}, mean={t.mean():.6f}, std={t.std():.6f}")
    else:
        print(f"  SKIP: unsupported {tensor.tensor_type.name}")
        continue

    weights[key] = t

# Quick validation: test output_norm + lm_head on a hidden state
if "output_norm" in weights and "lm_head" in weights:
    norm_w = weights["output_norm"]
    lm_w = weights["lm_head"]
    print(f"\nValidation:")
    print(f"  output_norm: shape={norm_w.shape}, mean={norm_w.mean():.4f}")
    print(f"  lm_head: shape={lm_w.shape}, mean={lm_w.mean():.6f}, std={lm_w.std():.4f}")

    # Load a hidden state from training data
    import json
    sample_dir = "data/features_daemon/sample_000000"
    meta = json.load(open(f"{sample_dir}/metadata.json"))
    h37 = np.fromfile(f"{sample_dir}/layer_37.bin", dtype=np.float32).reshape(-1, 2048)
    tokens = np.fromfile(f"{sample_dir}/input_ids.bin", dtype=np.int32)

    # Test: h37[50] -> output_norm -> lm_head -> should predict token[51]
    h = torch.from_numpy(h37[50:51])
    # RMS norm
    variance = h.pow(2).mean(-1, keepdim=True)
    h_normed = h * torch.rsqrt(variance + 1e-6) * norm_w.unsqueeze(0)
    # LM head
    logits = h_normed @ lm_w.T  # [1, vocab]
    probs = torch.softmax(logits[0], dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum().item()
    pred = logits[0].argmax().item()
    target = int(tokens[51])
    rank = (logits[0] > logits[0, target]).sum().item() + 1
    print(f"\n  h37[50] -> norm -> LM head:")
    print(f"    pred={pred}, target={target}, match={pred==target}")
    print(f"    entropy={entropy:.2f}, target_rank={rank}")

torch.save(weights, OUTPUT)
print(f"\nSaved to {OUTPUT} ({os.path.getsize(OUTPUT) / 1e6:.1f} MB)")
