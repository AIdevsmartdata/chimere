#!/usr/bin/env python3
"""
validate_features.py — 3-level validation of extracted hidden states.

Level 1: Sanity checks (files exist, correct count, no empty)
Level 2: Statistical validation (NaN/Inf, mean/std ranges, layer diversity)
Level 3: Functional test (micro-training, does loss decrease?)

Usage:
  python scripts/validate_features.py --features-dir data/features
  python scripts/validate_features.py --features-dir data/features --level 3 --blocks-dir data/blocks
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path


def level1_sanity(features_dir: Path, expected_layers=5):
    """Level 1: file existence and completeness."""
    print("=" * 60)
    print(" Level 1 — Sanity Checks")
    print("=" * 60)

    sample_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir()])
    n_samples = len(sample_dirs)
    print(f"  Total samples: {n_samples}")

    if n_samples == 0:
        print("  FAIL: No samples found!")
        return False

    incomplete = []
    empty = []
    no_meta = []
    total_tokens = 0

    for sd in sample_dirs:
        meta_path = sd / "metadata.json"
        if not meta_path.exists():
            meta_path = sd / "meta.json"
        if not meta_path.exists():
            no_meta.append(sd.name)
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        seq_len = meta.get("seq_len", meta.get("n_tokens", 0))
        total_tokens += seq_len

        layers_found = sorted([
            f.stem for f in sd.glob("layer_*.bin")
        ])
        if len(layers_found) < expected_layers:
            incomplete.append((sd.name, len(layers_found)))

        # Check for zero-size files
        for f in sd.iterdir():
            if f.suffix == ".bin" and f.stat().st_size == 0:
                empty.append(f"{sd.name}/{f.name}")

    ok = True
    if no_meta:
        print(f"  FAIL: {len(no_meta)} samples missing metadata: {no_meta[:5]}...")
        ok = False
    else:
        print(f"  OK: All {n_samples} samples have metadata")

    if incomplete:
        print(f"  FAIL: {len(incomplete)} samples have < {expected_layers} layers:")
        for name, count in incomplete[:5]:
            print(f"    {name}: {count} layers")
        ok = False
    else:
        print(f"  OK: All samples have {expected_layers} layers")

    if empty:
        print(f"  FAIL: {len(empty)} empty .bin files: {empty[:5]}...")
        ok = False
    else:
        print(f"  OK: No empty files")

    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg tokens/sample: {total_tokens / max(1, n_samples):.1f}")

    # Check file size consistency
    first_meta_path = sample_dirs[0] / "metadata.json"
    if not first_meta_path.exists():
        first_meta_path = sample_dirs[0] / "meta.json"
    with open(first_meta_path) as f:
        first_meta = json.load(f)
    hidden_dim = first_meta["hidden_dim"]
    print(f"  Hidden dim: {hidden_dim}")

    print()
    return ok


def level2_statistics(features_dir: Path, n_check=20, layers=None):
    """Level 2: statistical validation of hidden state values."""
    print("=" * 60)
    print(" Level 2 — Statistical Validation")
    print("=" * 60)

    sample_dirs = sorted([d for d in features_dir.iterdir() if d.is_dir()])
    check_dirs = sample_dirs[:n_check]
    print(f"  Checking {len(check_dirs)}/{len(sample_dirs)} samples")
    print()

    if layers is None:
        layers = [2, 11, 20, 29, 37]

    all_ok = True
    layer_stats = {l: {"means": [], "stds": []} for l in layers}

    for sd in check_dirs:
        meta_path = sd / "metadata.json"
        if not meta_path.exists():
            meta_path = sd / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        seq_len = meta.get("seq_len", meta.get("n_tokens", 0))
        hidden_dim = meta["hidden_dim"]

        sample_ok = True
        for layer_id in layers:
            layer_path = sd / f"layer_{layer_id:02d}.bin"
            if not layer_path.exists():
                layer_path = sd / f"layer_{layer_id}.bin"
            if not layer_path.exists():
                print(f"  {sd.name} layer {layer_id}: MISSING")
                sample_ok = False
                continue

            data = np.fromfile(layer_path, dtype=np.float32)
            data = data.reshape(seq_len, hidden_dim)

            mean_abs = np.abs(data).mean()
            std = data.std()
            has_nan = np.isnan(data).any()
            has_inf = np.isinf(data).any()

            layer_stats[layer_id]["means"].append(mean_abs)
            layer_stats[layer_id]["stds"].append(std)

            issues = []
            if has_nan:
                issues.append("NaN")
            if has_inf:
                issues.append("Inf")
            if mean_abs < 0.001:
                issues.append(f"mean_abs={mean_abs:.6f} (too low)")
            if mean_abs > 5.0:
                issues.append(f"mean_abs={mean_abs:.4f} (too high)")
            if std < 0.01:
                issues.append(f"std={std:.6f} (too low)")
            if std > 10.0:
                issues.append(f"std={std:.4f} (too high)")

            status = "OK" if not issues else "FAIL"
            if issues:
                print(f"  {sd.name} layer {layer_id}: {status} — {', '.join(issues)}")
                sample_ok = False

        if not sample_ok:
            all_ok = False

    # Layer diversity check
    print()
    print("  Layer statistics (should differ between layers):")
    print(f"  {'Layer':>8} {'Mean(|x|)':>12} {'Std':>12}")
    print(f"  {'-----':>8} {'---------':>12} {'---':>12}")

    layer_means_avg = {}
    for layer_id in layers:
        if layer_stats[layer_id]["means"]:
            m = np.mean(layer_stats[layer_id]["means"])
            s = np.mean(layer_stats[layer_id]["stds"])
            layer_means_avg[layer_id] = m
            print(f"  {layer_id:>8} {m:>12.4f} {s:>12.4f}")

    # Check that layers are not identical (diversity test)
    if len(layer_means_avg) >= 2:
        vals = list(layer_means_avg.values())
        spread = max(vals) - min(vals)
        if spread < 0.01:
            print(f"\n  WARN: Layer means very similar (spread={spread:.6f})")
            print("  This suggests the callback may be capturing the same tensor!")
            all_ok = False
        else:
            print(f"\n  OK: Layer diversity confirmed (spread={spread:.4f})")

    if all_ok:
        print("  OK: All statistical checks passed")
    print()
    return all_ok


def level3_functional(blocks_dir: Path, max_steps=50):
    """Level 3: micro-training to verify features contain learnable signal."""
    print("=" * 60)
    print(" Level 3 — Functional Test (micro-training)")
    print("=" * 60)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("  SKIP: PyTorch not available")
        return True

    # Add project root to path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    from chimere.config import DFlashConfig
    from chimere.modeling import DFlashDraftModel
    from chimere.data import DFlashDataset, collate_dflash

    blocks_path = Path(blocks_dir)
    dataset = DFlashDataset(blocks_path)

    if len(dataset) < 2:
        print(f"  SKIP: Only {len(dataset)} blocks (need >= 2 for meaningful test)")
        return True

    loader = DataLoader(
        dataset,
        batch_size=min(4, len(dataset)),
        shuffle=True,
        collate_fn=collate_dflash,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = DFlashConfig()
    model = DFlashDraftModel(config).to(device)
    model.freeze_shared_params()

    # Cast frozen params to BF16
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    print(f"  Device: {device}")
    print(f"  Blocks: {len(dataset)}")
    print(f"  Max steps: {max_steps}")
    print()

    losses = []
    model.train()
    step = 0

    for epoch in range(max(1, max_steps // max(1, len(loader)))):
        for batch in loader:
            if step >= max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            target_hidden = [h.to(device) for h in batch["target_hidden_states_list"]]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                loss, _ = model.forward_train(input_ids, target_hidden)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % 10 == 0:
                print(f"  Step {step:3d}: loss={loss.item():.4f}")
            step += 1

    if len(losses) < 5:
        print(f"  SKIP: Too few steps ({len(losses)})")
        return True

    first_5 = np.mean(losses[:5])
    last_5 = np.mean(losses[-5:])
    improvement = first_5 - last_5

    print()
    print(f"  First 5 steps avg loss: {first_5:.4f}")
    print(f"  Last 5 steps avg loss:  {last_5:.4f}")
    print(f"  Improvement: {improvement:.4f}")

    if improvement > 0.1:
        print("  OK: Loss is decreasing — features contain learnable signal!")
        return True
    elif improvement > 0:
        print("  WARN: Loss barely decreasing — features may have weak signal")
        return True
    else:
        print("  FAIL: Loss not decreasing — features may be garbage")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate extracted features")
    parser.add_argument("--features-dir", type=str, default="data/features")
    parser.add_argument("--blocks-dir", type=str, default="data/blocks")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3],
                        help="Validation level (1=sanity, 2=stats, 3=functional)")
    parser.add_argument("--n-check", type=int, default=20,
                        help="Number of samples for level 2 stats")
    parser.add_argument("--layers", type=str, default="2,11,20,29,37")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    layers = [int(x) for x in args.layers.split(",")]

    results = {}

    # Always run level 1
    results[1] = level1_sanity(features_dir)

    if args.level >= 2:
        results[2] = level2_statistics(features_dir, n_check=args.n_check, layers=layers)

    if args.level >= 3:
        results[3] = level3_functional(Path(args.blocks_dir))

    # Summary
    print("=" * 60)
    print(" Summary")
    print("=" * 60)
    all_ok = True
    for level, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  Level {level}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All checks passed! Features are ready for training.")
    else:
        print("\n  Some checks failed. Investigate before training.")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
