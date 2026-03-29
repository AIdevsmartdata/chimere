#!/usr/bin/env python3
"""
online_finetune.py — Incremental fine-tune of DFlash drafter on online buffer.

Designed to run as a nightly cron job or manually after accumulating
enough on-policy experiences in the online buffer.

Usage:
  python scripts/online_finetune.py \
    --checkpoint checkpoints_v8_deepkv/best.pt \
    --buffer-dir data/online_buffer \
    --output-dir checkpoints_v8_online \
    --epochs 2 --lr 1e-4 --bf16

Features:
  - Loads from existing checkpoint (warm start)
  - Trains on online buffer samples (same format as features_fullseq)
  - Lower LR than initial training (fine-tune, not train from scratch)
  - Rollback safety: compares τ before/after on recent buffer samples
  - Saves best checkpoint only if τ improves
"""
import argparse
import dataclasses
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.modeling_v8 import DFlashDraftModelV8
from chimere.data_v8 import DFlashFullSeqDirDataset, collate_v8


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    from dataclasses import fields
    config = DFlashV7Config(**{
        f.name: ckpt["config"][f.name]
        for f in fields(DFlashV7Config)
        if f.name in ckpt["config"]
    })

    model = DFlashDraftModelV8(config)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.freeze_shared_params()
    model.enable_gradient_checkpointing()

    # Frozen weights to BF16
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.norm.parameters():
        p.data = p.data.to(torch.bfloat16)

    return model, config, ckpt.get("epoch", 0)


def quick_tau(model, features_dir, config, device, n_samples=50, anchors=5):
    """Quick τ estimate on recent samples."""
    from chimere.data_v8 import DFlashFullSeqDirDataset
    import json

    sample_dirs = sorted([
        d for d in Path(features_dir).iterdir()
        if d.is_dir() and (d / "context_hidden.bin").exists()
    ])[-n_samples:]  # last N (most recent)

    if not sample_dirs:
        return 0.0

    model.eval()
    total_accepted = 0
    total_drafted = 0
    K = config.block_size
    n_layers = config.num_feature_layers
    H = config.target_hidden_size

    with torch.no_grad():
        for d in sample_dirs:
            meta = json.load(open(d / "metadata.json"))
            n_pos = meta.get("n_positions", meta.get("seq_len", 0))
            if n_pos < K + 2:
                continue

            raw = np.fromfile(d / "context_hidden.bin", dtype=np.float16)
            # Compute n_pos from actual file size (metadata may be stale)
            n_pos = raw.size // (n_layers * H)
            if n_pos < K + 2:
                continue
            raw = raw.reshape(n_layers, n_pos, H).astype(np.float32)
            tokens = np.fromfile(d / "tokens.bin", dtype=np.int32)

            # Test a few random anchors
            max_anchor = min(n_pos, len(tokens)) - K
            if max_anchor < 1:
                continue
            test_anchors = np.random.choice(range(1, max_anchor + 1),
                                            size=min(anchors, max_anchor), replace=False)

            for anchor_pos in test_anchors:
                ctx_end = anchor_pos + 1
                ctx_start = max(0, ctx_end - config.max_ctx_len)
                ctx_len = ctx_end - ctx_start

                hidden_list = [
                    torch.from_numpy(raw[i, ctx_start:ctx_end]).unsqueeze(0).to(device)
                    for i in range(n_layers)
                ]
                ctx_lengths = torch.tensor([ctx_len], device=device, dtype=torch.long)
                anchor_positions_t = torch.tensor([anchor_pos], device=device, dtype=torch.long)

                gt_block = tokens[anchor_pos:anchor_pos + K]
                if len(gt_block) < 2:
                    continue

                draft_ids, _, _ = model.generate_block(
                    hidden_list, context_lengths=ctx_lengths,
                    temperature=0.0,
                    anchor_token_id=int(gt_block[0]),
                    anchor_positions=anchor_positions_t,
                )
                draft = draft_ids[0].cpu().numpy()

                for j in range(min(K - 1, len(gt_block) - 1)):
                    if draft[j] == gt_block[j + 1]:
                        total_accepted += 1
                    else:
                        break
                total_drafted += min(len(gt_block) - 1, K - 1)

    return total_accepted / max(1, total_drafted)


def main():
    parser = argparse.ArgumentParser(description="Online fine-tune DFlash drafter")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Starting checkpoint")
    parser.add_argument("--buffer-dir", type=str, required=True,
                        help="Online buffer directory (features_fullseq format)")
    parser.add_argument("--output-dir", type=str, default="checkpoints_v8_online")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower LR for fine-tuning (default 1e-4 vs 3e-4 for initial)")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--anchors-per-seq", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--rollback-threshold", type=float, default=0.95,
                        help="Rollback if new τ < old τ × threshold")
    parser.add_argument("--no-rollback", action="store_true")
    parser.add_argument("--loss-type", type=str, default="lk",
                        choices=["ce", "lk", "lk_alpha"],
                        help="Loss type: ce (original CE+smoothing), lk (hybrid LK), lk_alpha (-log α)")
    parser.add_argument("--lk-eta", type=float, default=3.0,
                        help="LK adaptive schedule decay rate (default 3.0)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Attention dropout for regularization (default 0.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" DFlash Online Fine-Tune")
    print("=" * 60)

    # Load model
    model, config, base_epoch = load_model(args.checkpoint, device)
    config.lk_eta = args.lk_eta
    print(f"  Loaded checkpoint: {args.checkpoint} (epoch {base_epoch})")
    print(f"  Loss type: {args.loss_type}" + (f" (η={args.lk_eta})" if args.loss_type.startswith("lk") else ""))

    # Measure baseline τ
    print(f"  Measuring baseline τ on buffer...", flush=True)
    tau_before = quick_tau(model, args.buffer_dir, config, device)
    print(f"  Baseline τ = {tau_before:.2%}")

    # Load buffer as dataset
    dataset = DFlashFullSeqDirDataset(
        args.buffer_dir,
        block_size=config.block_size,
        max_ctx_len=config.max_ctx_len,
        anchors_per_seq=args.anchors_per_seq,
        target_layers=tuple(config.target_layer_ids),
        hidden_dim=config.target_hidden_size,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=gen
    )

    print(f"  Buffer samples: {n_total} ({n_train} train, {n_val} val)")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")

    def _worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed() % 2**32)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_v8, num_workers=2, pin_memory=True,
        drop_last=True, persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v8, num_workers=2, pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_worker_init_fn,
    )

    # Optimizer — lower LR for fine-tuning
    # Apply dropout if requested
    if args.dropout > 0:
        for layer in model.layers:
            layer.self_attn.attention_dropout = args.dropout

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
        fused=True,
    )

    n_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = max(1, int(n_steps * 0.02))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, n_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = args.bf16 and device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Training steps: {n_steps}")
    print("=" * 60)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            block_ids = batch["block_input_ids"].to(device, non_blocking=True)
            ctx_hidden = [h.to(device, non_blocking=True) for h in batch["context_hidden_list"]]
            ctx_lengths = batch["context_lengths"].to(device, non_blocking=True)
            anchor_positions = batch["anchor_positions"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, _, _ = model.forward_train_multi(
                    block_ids, ctx_hidden, ctx_lengths,
                    anchor_positions=anchor_positions,
                    chunk_size=args.chunk_size,
                    loss_type=args.loss_type,
                )
                loss = loss / args.grad_accum

            loss.backward()
            total_loss += loss.item() * args.grad_accum

            if (batch_idx + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Flush trailing partial accumulation group
        if (batch_idx + 1) % args.grad_accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(1, len(train_loader))

        # Eval
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                block_ids = batch["block_input_ids"].to(device)
                ctx_hidden = [h.to(device) for h in batch["context_hidden_list"]]
                ctx_lengths = batch["context_lengths"].to(device)
                anchor_positions = batch["anchor_positions"].to(device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss, _, _ = model.forward_train_multi(
                        block_ids, ctx_hidden, ctx_lengths,
                        anchor_positions=anchor_positions,
                        chunk_size=args.chunk_size,
                        loss_type=args.loss_type,
                    )
                val_loss += loss.item()
                n_val_batches += 1
        val_loss /= max(1, n_val_batches)
        model.train()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{args.epochs} | train_loss={avg_loss:.4f} | "
              f"val_loss={val_loss:.4f} | {elapsed:.0f}s", flush=True)

    # Measure τ after fine-tune
    print(f"\n  Measuring post-training τ...", flush=True)
    tau_after = quick_tau(model, args.buffer_dir, config, device)
    print(f"  τ before: {tau_before:.2%}")
    print(f"  τ after:  {tau_after:.2%}")

    # Rollback check
    if not args.no_rollback and tau_after < tau_before * args.rollback_threshold:
        print(f"\n  ROLLBACK: τ dropped below threshold "
              f"({tau_after:.2%} < {tau_before:.2%} × {args.rollback_threshold})")
        print(f"  Keeping original checkpoint: {args.checkpoint}")
        return

    # Save
    def _clean_state_dict(sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    ckpt_data = {
        "model_state_dict": _clean_state_dict(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": dataclasses.asdict(config),
        "version": "v8_online",
        "epoch": base_epoch + args.epochs,
        "tau_before": tau_before,
        "tau_after": tau_after,
        "buffer_size": n_total,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_type": args.loss_type,
    }
    save_path = output_dir / "best.pt"
    backup_path = output_dir / "best_backup.pt"
    if save_path.exists():
        save_path.rename(backup_path)
    torch.save(ckpt_data, save_path)
    if backup_path.exists():
        backup_path.unlink()

    print(f"\n  Saved fine-tuned model: {save_path}")
    print(f"  τ improvement: {tau_before:.2%} → {tau_after:.2%} "
          f"({(tau_after - tau_before) / max(tau_before, 1e-8) * 100:+.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
