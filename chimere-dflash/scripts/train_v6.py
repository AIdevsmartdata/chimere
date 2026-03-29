#!/usr/bin/env python3
"""
train_v6.py — Train DFlash v6 drafter (z-lab aligned architecture).

Changes from v5:
  - 8 layers, GQA (32 heads, 4 KV heads), intermediate 6144
  - Concat fusion (5*H → H) instead of weighted average
  - gamma=7 streak distillation loss
  - LR=6e-4, epochs=6, warmup=0.04
  - Unified KV injection (same k_proj/v_proj for context+noise)

Usage:
  python scripts/train_v6.py --features-dir data/features_iq3
  python scripts/train_v6.py --features-dir data/features_iq3 --epochs 6 --bf16
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v6 import DFlashV6Config
from chimere.modeling_v6 import DFlashDraftModelV6
from chimere.data_v5 import DFlashSeqDataset, collate_v5
from chimere.data_v6 import DFlashSinglePosDataset, collate_single_pos


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class _SubsetSeqDataset(torch.utils.data.Dataset):
    """Subset of DFlashSeqDataset using specific sequence indices."""

    def __init__(self, parent_dataset, seq_indices):
        self.parent = parent_dataset
        self.seq_indices = seq_indices
        self.blocks_per_seq = parent_dataset.blocks_per_seq

    def __len__(self):
        return len(self.seq_indices) * self.blocks_per_seq

    def __getitem__(self, idx):
        import random as _random
        import numpy as np

        local_seq_idx = idx % len(self.seq_indices)
        global_seq_idx = self.seq_indices[local_seq_idx]

        sample = self.parent._load_sample(global_seq_idx)
        n_tokens = sample["n_tokens"]
        block_size = self.parent.block_size

        max_anchor = n_tokens - block_size
        min_anchor = 1
        if max_anchor < min_anchor:
            max_anchor = min_anchor

        anchor_pos = _random.randint(min_anchor, max_anchor)

        block_end = min(anchor_pos + block_size, n_tokens)
        block_tokens = sample["tokens"][anchor_pos:block_end]
        if len(block_tokens) < block_size:
            block_tokens = np.pad(block_tokens, (0, block_size - len(block_tokens)))

        ctx_end = anchor_pos + 1
        ctx_start = max(0, ctx_end - self.parent.max_ctx_len)
        ctx_len = ctx_end - ctx_start

        context_hidden_list = []
        for l in self.parent.target_layers:
            h = sample["layer_hidden"][l][ctx_start:ctx_end]
            context_hidden_list.append(
                torch.from_numpy(h.astype(np.float32))
            )

        return {
            "block_input_ids": torch.from_numpy(
                block_tokens.astype(np.int64) if isinstance(block_tokens, np.ndarray)
                else block_tokens
            ),
            "context_hidden_list": context_hidden_list,
            "context_length": ctx_len,
        }


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_amp,
                grad_accum_steps, max_grad_norm, epoch=0, log_every=50):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step_loss = 0.0
    optimizer.zero_grad()
    t0 = time.time()
    n_steps = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        block_ids = batch["block_input_ids"].to(device)
        ctx_hidden = [h.to(device) for h in batch["context_hidden_list"]]
        ctx_lengths = batch["context_lengths"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logits = model.forward_train(block_ids, ctx_hidden, ctx_lengths)
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        cur_loss = loss.item() * grad_accum_steps
        step_loss += cur_loss

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        total_loss += cur_loss
        total_tokens += block_ids.numel()

        # Per-step logging
        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg_step = step_loss / log_every
            lr = optimizer.param_groups[0]["lr"]
            tok_s = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  E{epoch+1} [{batch_idx+1}/{n_steps}] "
                  f"loss={avg_step:.4f} | lr={lr:.2e} | "
                  f"{tok_s:.0f} tok/s", flush=True)
            step_loss = 0.0

    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_samples = 0

    for batch in dataloader:
        block_ids = batch["block_input_ids"].to(device)
        ctx_hidden = [h.to(device) for h in batch["context_hidden_list"]]
        ctx_lengths = batch["context_lengths"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logits = model.forward_train(block_ids, ctx_hidden, ctx_lengths)

        total_loss += loss.item() * block_ids.shape[0]
        n_samples += block_ids.shape[0]

        # Accuracy on positions 1..K-1 (predicted positions only)
        preds = logits[:, 1:, :].argmax(dim=-1)  # [B, K-1]
        targets = block_ids[:, 1:]  # [B, K-1]
        total_correct += (preds == targets).sum().item()
        total_count += targets.numel()

    avg_loss = total_loss / max(1, n_samples)
    accuracy = total_correct / max(1, total_count)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train DFlash v6 drafter (z-lab aligned)")
    parser.add_argument("--features-dir", type=str, default="data/features_iq3")
    parser.add_argument("--output-dir", type=str, default="checkpoints_v6")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-ctx-len", type=int, default=1024)
    parser.add_argument("--blocks-per-seq", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" DFlash v6 — z-lab Aligned Architecture")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Features:    {args.features_dir}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size} (× {args.grad_accum} accum = "
          f"{args.batch_size * args.grad_accum} effective)")
    print(f"  LR:          {args.lr}")
    print(f"  BF16:        {args.bf16}")
    print(f"  Block size:  {args.block_size}")
    print(f"  Max ctx len: {args.max_ctx_len}")
    print(f"  Blocks/seq:  {args.blocks_per_seq}")
    print()

    # Create config early (needed for dataset setup)
    config = DFlashV6Config(block_size=args.block_size)

    # Detect dataset format: single-position (anchor_hidden.bin) or full-sequence (layer_*.bin)
    features_path = Path(args.features_dir)
    sample_dirs = sorted([d for d in features_path.iterdir() if d.is_dir()])
    is_single_pos = any((d / "anchor_hidden.bin").exists() for d in sample_dirs[:5])

    if is_single_pos:
        print(f"  Format: single-position (anchor_hidden.bin)")
        full_dataset = DFlashSinglePosDataset(
            args.features_dir,
            block_size=args.block_size,
            num_layers=config.num_feature_layers,
            hidden_dim=config.target_hidden_size,
        )
        collate_fn = collate_single_pos

        # Train/val split
        n_total = len(full_dataset)
        n_val = max(1, int(n_total * args.val_split))
        n_train = n_total - n_val

        gen = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val], generator=gen
        )

        print(f"  Train: {n_train} samples")
        print(f"  Val:   {n_val} samples")
    else:
        print(f"  Format: full-sequence (layer_*.bin)")
        dataset = DFlashSeqDataset(
            args.features_dir,
            block_size=args.block_size,
            target_layers=tuple(config.target_layer_ids),
            max_ctx_len=args.max_ctx_len,
            blocks_per_seq=args.blocks_per_seq,
        )
        collate_fn = collate_v5

        n_seq = len(dataset.sample_dirs)
        n_val_seq = max(1, int(n_seq * args.val_split))
        n_train_seq = n_seq - n_val_seq

        gen = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(n_seq, generator=gen).tolist()
        train_indices = perm[:n_train_seq]
        val_indices = perm[n_train_seq:]

        train_dataset = _SubsetSeqDataset(dataset, train_indices)
        val_dataset = _SubsetSeqDataset(dataset, val_indices)

        print(f"  Train: {n_train_seq} sequences ({len(train_dataset)} virtual items)")
        print(f"  Val:   {n_val_seq} sequences ({len(val_dataset)} virtual items)")

    print()

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # Create model
    model = DFlashDraftModelV6(config)

    # Load Qwen shared weights
    qwen_weights_path = os.path.join(os.path.dirname(args.features_dir), "qwen_shared_weights.pt")
    if os.path.exists(qwen_weights_path):
        print(f"  Loading Qwen shared weights from {qwen_weights_path}")
        qwen_w = torch.load(qwen_weights_path, map_location="cpu", weights_only=True)
        model.embed_tokens.weight.data = qwen_w["embed_tokens"].float()
        model.lm_head.weight.data = qwen_w["lm_head"].float()
        model.norm.weight.data = qwen_w["output_norm"].float()
        del qwen_w
        print(f"    embed_tokens std={model.embed_tokens.weight.std():.4f}")
        print(f"    lm_head std={model.lm_head.weight.std():.6f}")
        print(f"    output_norm mean={model.norm.weight.mean():.4f}")
    else:
        print(f"  WARNING: {qwen_weights_path} not found!")

    model = model.to(device)

    # torch.compile for faster training
    if not args.no_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)
        print("  torch.compile enabled")

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # Freeze shared params and cast to BF16
    model.freeze_shared_params()
    model.enable_gradient_checkpointing()
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    trainable, total = count_parameters(model)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_bytes = trainable * 4
    frozen_bytes = frozen * 2
    print(f"  Parameters: {trainable / 1e6:.1f}M trainable (FP32) + "
          f"{frozen / 1e6:.1f}M frozen (BF16)")
    print(f"  Model VRAM: ~{(trainable_bytes + frozen_bytes) / 1e9:.2f} GB")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    num_training_steps = len(train_loader) // args.grad_accum * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    if args.resume and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    use_amp = args.bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 doesn't need scaler

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

    print(f"  Training steps: {num_training_steps} (warmup: {num_warmup_steps})")
    print("=" * 60)
    print()

    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss, tokens = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, args.grad_accum, config.max_grad_norm,
            epoch=epoch,
        )

        val_loss, val_acc = evaluate(model, val_loader, device, use_amp)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{start_epoch + args.epochs} "
              f"| train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} "
              f"| val_acc={val_acc:.4f} "
              f"| lr={lr_now:.2e} "
              f"| {elapsed:.1f}s "
              f"| {tokens / elapsed:.0f} tok/s")

        # Save checkpoint (strip _orig_mod. prefix from torch.compile)
        def _clean_state_dict(sd):
            return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": _clean_state_dict(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": dataclasses.asdict(config),
            "version": "v6_zlab_aligned",
        }

        ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
        torch.save(ckpt_data, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": _clean_state_dict(model.state_dict()),
                "config": dataclasses.asdict(config),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "version": "v6_zlab_aligned",
            }, output_dir / "best.pt")
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

    print()
    print("=" * 60)
    print(f" Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
