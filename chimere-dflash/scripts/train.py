#!/usr/bin/env python3
"""
train.py — Train the DFlash block diffusion drafter.

Trains the 5-layer bidirectional transformer denoiser on pre-extracted
hidden states from Qwen3.5-35B-A3B.

Usage:
  python scripts/train.py --blocks-dir data/blocks
  python scripts/train.py --blocks-dir data/blocks --epochs 5 --lr 5e-5 --bf16
"""

import argparse
import dataclasses as _dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config import DFlashConfig
from chimere.modeling import DFlashDraftModel
from chimere.data import DFlashDataset, collate_dflash


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine annealing with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_biased_timesteps(B, T, high_noise_ratio, device):
    """Sample timesteps with bias towards high noise levels.

    With high_noise_ratio=0.7, 70% of samples come from t in [0.7*T, T)
    and 30% from t in [0, 0.7*T). This forces the model to learn from
    situations where the signal is nearly destroyed.
    """
    if high_noise_ratio <= 0.0:
        return torch.randint(0, T, (B,), device=device)

    high_threshold = int(T * 0.7)  # t >= 700 when T=1000
    n_high = int(B * high_noise_ratio)
    n_low = B - n_high

    high_t = torch.randint(high_threshold, T, (n_high,), device=device)
    low_t = torch.randint(0, high_threshold, (n_low,), device=device) if n_low > 0 else torch.empty(0, dtype=torch.long, device=device)

    timesteps = torch.cat([high_t, low_t])
    # Shuffle so high/low aren't grouped
    return timesteps[torch.randperm(B, device=device)]


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_amp, grad_accum_steps,
                high_noise_ratio=0.0):
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step_count = 0
    optimizer.zero_grad()
    T = model.diffusion.num_timesteps

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_hidden = [h.to(device) for h in batch["target_hidden_states_list"]]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logits = model.forward_train(input_ids, target_hidden)
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            step_count += 1

        total_loss += loss.item() * grad_accum_steps
        total_tokens += input_ids.numel()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, step_count, total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        target_hidden = [h.to(device) for h in batch["target_hidden_states_list"]]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logits = model.forward_train(input_ids, target_hidden)

        total_loss += loss.item() * input_ids.shape[0]
        preds = logits.argmax(dim=-1)
        total_correct += (preds == input_ids).sum().item()
        total_tokens += input_ids.numel()

    avg_loss = total_loss / max(1, len(dataloader.dataset))
    accuracy = total_correct / max(1, total_tokens)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train DFlash drafter")
    parser.add_argument("--blocks-dir", type=str, default="data/blocks",
                        help="Directory with training blocks")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--bf16", action="store_true",
                        help="Use BF16 mixed precision")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--high-noise-ratio", type=float, default=0.0,
                        help="Fraction of samples from high timesteps (0=uniform, 0.7=70%% high noise)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" Chimere-DFlash Drafter Training")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Blocks:     {args.blocks_dir}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size} (× {args.grad_accum} accum = {args.batch_size * args.grad_accum} effective)")
    print(f"  LR:         {args.lr}")
    print(f"  BF16:       {args.bf16}")
    print(f"  High-noise: {args.high_noise_ratio}")
    print()

    # Load dataset
    dataset = DFlashDataset(args.blocks_dir, block_size=args.block_size)

    # Load dataset metadata
    meta_path = Path(args.blocks_dir) / "dataset_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            dataset_meta = json.load(f)
        print(f"  Dataset: {dataset_meta['n_blocks']} blocks, "
              f"{dataset_meta['n_samples']} samples, "
              f"{dataset_meta['total_tokens']} tokens")
        print()

    # Train/val split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"  Train: {n_train} blocks, Val: {n_val} blocks")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_dflash,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dflash,
        num_workers=1,
        pin_memory=True,
    )

    # Create model
    config = DFlashConfig(block_size=args.block_size)
    model = DFlashDraftModel(config).to(device)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # Freeze embeddings and LM head (shared from target, not trained)
    model.freeze_shared_params()

    # Cast frozen parameters to BF16 to halve their VRAM footprint
    # (1B frozen params: 4 GB FP32 → 2 GB BF16)
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    trainable, total = count_parameters(model)
    frozen_bf16 = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_bytes = trainable * 4
    frozen_bytes = frozen_bf16 * 2
    print(f"  Parameters: {trainable / 1e6:.1f}M trainable (FP32) + {frozen_bf16 / 1e6:.1f}M frozen (BF16)")
    print(f"  Model VRAM: ~{(trainable_bytes + frozen_bytes) / 1e9:.2f} GB")
    print()

    # Optimizer — only trainable params
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Resume optimizer state
    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    num_training_steps = len(train_loader) // args.grad_accum * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Resume scheduler state
    if args.resume and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    use_amp = args.bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / "config.json"
    if not config_path.exists():
        import dataclasses
        with open(config_path, "w") as f:
            json.dump(_dataclasses.asdict(config), f, indent=2)

    print(f"  Training steps: {num_training_steps} (warmup: {num_warmup_steps})")
    print("=" * 60)
    print()

    best_val_loss = float("inf")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        train_loss, steps, tokens = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, args.grad_accum,
            high_noise_ratio=args.high_noise_ratio
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

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or val_loss < best_val_loss:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": _dataclasses.asdict(config),
            }, ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = output_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": _dataclasses.asdict(config),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, best_path)
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")

    print()
    print("=" * 60)
    print(f" Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
