#!/usr/bin/env python3
"""
verify_extraction.py — Verify C++ hidden state extraction against reference.

Part 1: Tokenizer alignment (CPU, instant)
Part 2: Hidden state sanity checks (statistical analysis)
Part 3: Cross-sample consistency check
"""
import json
import sys
from pathlib import Path

import numpy as np

SAMPLE_DIR = Path("data/features/sample_000000")
LAYERS = [2, 11, 20, 29, 37]


def verify_tokenizer():
    """Compare C++ tokenization vs Python HuggingFace tokenizer."""
    print("=" * 60)
    print("PART 1: TOKENIZER ALIGNMENT")
    print("=" * 60)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    with open(SAMPLE_DIR / "metadata.json") as f:
        meta = json.load(f)

    prompt = meta["prompt"]
    cpp_ids = np.fromfile(SAMPLE_DIR / "input_ids.bin", dtype=np.int32)
    py_ids = tok.encode(prompt, add_special_tokens=False)

    print(f"C++ tokens:    {len(cpp_ids)}")
    print(f"Python tokens: {len(py_ids)}")
    print(f"Length match:  {len(cpp_ids) == len(py_ids)}")

    if len(cpp_ids) != len(py_ids):
        # Try with special tokens
        py_ids_special = tok.encode(prompt, add_special_tokens=True)
        if len(py_ids_special) == len(cpp_ids):
            print(f"  → Match WITH add_special_tokens=True ({len(py_ids_special)} tokens)")
            print("  → C++ extractor includes BOS/EOS. Training must account for this!")
            py_ids = py_ids_special
        else:
            print(f"  → MISMATCH even with special tokens ({len(py_ids_special)})")
            print("  FAIL: Token count mismatch is a critical bug.")
            return False

    mismatches = sum(1 for i in range(len(cpp_ids)) if cpp_ids[i] != py_ids[i])
    print(f"Mismatches:    {mismatches} / {len(cpp_ids)}")

    if mismatches > 0:
        print("\nFirst 10 mismatches:")
        count = 0
        for i in range(len(cpp_ids)):
            if cpp_ids[i] != py_ids[i]:
                cpp_tok = tok.decode([int(cpp_ids[i])])
                py_tok = tok.decode([int(py_ids[i])])
                print(f"  pos {i}: C++={cpp_ids[i]} ({repr(cpp_tok)}) vs Py={py_ids[i]} ({repr(py_tok)})")
                count += 1
                if count >= 10:
                    break
        print("\nFAIL: Tokenizer mismatch — all hidden states are misaligned!")
        return False

    print("\nPASS: Tokenizer alignment is perfect.")
    return True


def verify_hidden_states():
    """Analyze hidden state statistics for sanity."""
    print("\n" + "=" * 60)
    print("PART 2: HIDDEN STATE ANALYSIS")
    print("=" * 60)

    with open(SAMPLE_DIR / "metadata.json") as f:
        meta = json.load(f)

    n_tokens = meta["seq_len"]
    h_dim = meta["hidden_dim"]
    all_pass = True

    for layer_id in LAYERS:
        path = SAMPLE_DIR / f"layer_{layer_id:02d}.bin"
        data = np.fromfile(path, dtype=np.float32)
        expected = n_tokens * h_dim

        # Shape check
        if data.shape[0] != expected:
            print(f"Layer {layer_id:2d}: FAIL — shape {data.shape[0]} != expected {expected}")
            all_pass = False
            continue

        reshaped = data.reshape(n_tokens, h_dim)

        # NaN/Inf check
        n_nan = np.isnan(data).sum()
        n_inf = np.isinf(data).sum()
        if n_nan > 0 or n_inf > 0:
            print(f"Layer {layer_id:2d}: FAIL — {n_nan} NaN, {n_inf} Inf values!")
            all_pass = False
            continue

        # Statistics
        norms = np.linalg.norm(reshaped, axis=1)
        print(f"Layer {layer_id:2d}: PASS")
        print(f"  mean={reshaped.mean():.6f}, std={reshaped.std():.4f}")
        print(f"  min={reshaped.min():.4f}, max={reshaped.max():.4f}")
        print(f"  norms: mean={norms.mean():.2f}, range=[{norms.min():.2f}, {norms.max():.2f}]")

    # Check norm progression (should increase with depth for post-residual states)
    print("\nNorm progression check:")
    prev_mean_norm = 0
    for layer_id in LAYERS:
        data = np.fromfile(SAMPLE_DIR / f"layer_{layer_id:02d}.bin", dtype=np.float32)
        reshaped = data.reshape(n_tokens, h_dim)
        mean_norm = np.linalg.norm(reshaped, axis=1).mean()
        trend = "↑" if mean_norm > prev_mean_norm else "↓ WARNING"
        print(f"  Layer {layer_id:2d}: mean_norm={mean_norm:.2f} {trend}")
        prev_mean_norm = mean_norm

    print("\nExpected: norms should generally increase with depth (residual stream grows).")
    print("If norms are flat ~1.0 everywhere → likely post-layernorm (not post-residual).")

    return all_pass


def verify_cross_sample():
    """Check consistency across multiple samples."""
    print("\n" + "=" * 60)
    print("PART 3: CROSS-SAMPLE CONSISTENCY")
    print("=" * 60)

    features_dir = Path("data/features")
    samples = sorted([d for d in features_dir.iterdir() if d.is_dir()])[:5]
    print(f"Checking {len(samples)} samples...")

    for sample_dir in samples:
        meta_path = sample_dir / "metadata.json"
        if not meta_path.exists():
            print(f"  {sample_dir.name}: SKIP (no metadata)")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        n_tokens = meta["seq_len"]
        h_dim = meta["hidden_dim"]
        issues = []

        # Check all layer files exist
        for lid in LAYERS:
            layer_path = sample_dir / f"layer_{lid:02d}.bin"
            if not layer_path.exists():
                issues.append(f"missing layer_{lid:02d}.bin")
                continue
            data = np.fromfile(layer_path, dtype=np.float32)
            if data.shape[0] != n_tokens * h_dim:
                issues.append(f"layer {lid} size mismatch: {data.shape[0]} vs {n_tokens * h_dim}")
            if np.isnan(data).any():
                issues.append(f"layer {lid} has NaN")

        # Check input_ids
        ids_path = sample_dir / "input_ids.bin"
        if ids_path.exists():
            ids = np.fromfile(ids_path, dtype=np.int32)
            if len(ids) < n_tokens:
                issues.append(f"input_ids too short: {len(ids)} < {n_tokens}")

        status = "PASS" if not issues else "FAIL"
        print(f"  {sample_dir.name}: {status} (n_tokens={n_tokens})")
        for issue in issues:
            print(f"    → {issue}")


def main():
    print("DFlash Extraction Verification")
    print(f"Sample: {SAMPLE_DIR}")
    print()

    if not SAMPLE_DIR.exists():
        print(f"ERROR: Sample directory not found: {SAMPLE_DIR}")
        sys.exit(1)

    tok_ok = verify_tokenizer()
    hs_ok = verify_hidden_states()
    verify_cross_sample()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tokenizer alignment: {'PASS' if tok_ok else 'FAIL'}")
    print(f"Hidden state sanity: {'PASS' if hs_ok else 'FAIL'}")

    if tok_ok and hs_ok:
        print("\nExtraction pipeline looks correct.")
        print("Remaining risk: layer indexing (post-FFN vs post-layernorm).")
        print("To verify definitively, need to compare against PyTorch forward")
        print("pass — but that requires loading 35B model (70GB+ RAM).")
    else:
        print("\nCRITICAL ISSUES FOUND — fix before retraining!")


if __name__ == "__main__":
    main()
