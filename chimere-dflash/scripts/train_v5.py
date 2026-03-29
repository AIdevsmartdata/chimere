#!/usr/bin/env python3
"""
train_v5.py — Train DFlash v5 drafter with full context KV injection.

Loads full sequences from data/features/ and trains the v5 architecture
where block tokens cross-attend to ALL target hidden states from the
verified prefix at every drafter layer.

Usage:
  python scripts/train_v5.py --features-dir data/features
  python scripts/train_v5.py --features-dir data/features --epochs 10 --bf16
"""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config import DFlashConfig
from chimere.modeling_v5 import DFlashDraftModelV5
from chimere.data_v5 import DFlashSeqDataset, collate_v5


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


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_amp, grad_accum_steps):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()

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

        total_loss += loss.item() * grad_accum_steps
        total_tokens += block_ids.numel()

    avg_loss = total_loss / len(dataloader)
    return avg_loss, total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    n_samples = 0

    for batch in dataloader:
        block_ids = batch["block_input_ids"].to(device)
        ctx_hidden = [h.to(device) for h in batch["context_hidden_list"]]
        ctx_lengths = batch["context_lengths"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, logits = model.forward_train(block_ids, ctx_hidden, ctx_lengths)

        total_loss += loss.item() * block_ids.shape[0]
        n_samples += block_ids.shape[0]

        # Accuracy on ALL positions (not just masked, for comparability)
        preds = logits.argmax(dim=-1)
        total_correct += (preds == block_ids).sum().item()
        total_masked += block_ids.numel()

    avg_loss = total_loss / max(1, n_samples)
    accuracy = total_correct / max(1, total_masked)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train DFlash v5 drafter")
    parser.add_argument("--features-dir", type=str, default="data/features",
                        help="Directory with extracted features")
    parser.add_argument("--output-dir", type=str, default="checkpoints_v5",
                        help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-ctx-len", type=int, default=512,
                        help="Max context length (older positions clipped)")
    parser.add_argument("--blocks-per-seq", type=int, default=20,
                        help="Virtual blocks sampled per sequence per epoch")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" DFlash v5 — Full Context KV Injection Training")
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

    # Load dataset
    dataset = DFlashSeqDataset(
        args.features_dir,
        block_size=args.block_size,
        max_ctx_len=args.max_ctx_len,
        blocks_per_seq=args.blocks_per_seq,
    )
    print()

    # Train/val split (split by sequence, not by virtual items)
    n_seq = len(dataset.sample_dirs)
    n_val_seq = max(1, int(n_seq * args.val_split))
    n_train_seq = n_seq - n_val_seq

    # Create separate datasets for train/val with different sequence subsets
    # (we can't use random_split because items map to sequences via modulo)
    gen = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n_seq, generator=gen).tolist()
    train_indices = perm[:n_train_seq]
    val_indices = perm[n_train_seq:]

    train_dataset = _SubsetSeqDataset(dataset, train_indices)
    val_dataset = _SubsetSeqDataset(dataset, val_indices)

    print(f"  Train: {n_train_seq} sequences ({len(train_dataset)} virtual items)")
    print(f"  Val:   {n_val_seq} sequences ({len(val_dataset)} virtual items)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_v5,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_v5,
        num_workers=1,
        pin_memory=True,
    )

    # Create model
    config = DFlashConfig(block_size=args.block_size)
    model = DFlashDraftModelV5(config)

    # Load real Qwen weights for shared params (embed_tokens, lm_head, output_norm)
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
        print(f"  WARNING: {qwen_weights_path} not found! Using random init for shared params.")

    model = model.to(device)

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # Freeze and cast shared params
    model.freeze_shared_params()
    model.enable_gradient_checkpointing()
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)

    trainable, total = count_parameters(model)
    frozen_bf16 = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_bytes = trainable * 4
    frozen_bytes = frozen_bf16 * 2
    print(f"  Parameters: {trainable / 1e6:.1f}M trainable (FP32) + "
          f"{frozen_bf16 / 1e6:.1f}M frozen (BF16)")
    print(f"  Model VRAM: ~{(trainable_bytes + frozen_bytes) / 1e9:.2f} GB")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    num_training_steps = len(train_loader) // args.grad_accum * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    if args.resume and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    use_amp = args.bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
            device, use_amp, args.grad_accum
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
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": dataclasses.asdict(config),
            "version": "v5_kv_injection",
        }

        ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pt"
        torch.save(ckpt_data, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": dataclasses.asdict(config),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "version": "v5_kv_injection",
            }, output_dir / "best.pt")
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

    print()
    print("=" * 60)
    print(f" Training complete!")
    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {output_dir}")
    print("=" * 60)


class _SubsetSeqDataset(torch.utils.data.Dataset):
    """Subset of DFlashSeqDataset using specific sequence indices."""

    def __init__(self, parent_dataset, seq_indices):
        self.parent = parent_dataset
        self.seq_indices = seq_indices
        self.blocks_per_seq = parent_dataset.blocks_per_seq

    def __len__(self):
        return len(self.seq_indices) * self.blocks_per_seq

    def __getitem__(self, idx):
        local_seq_idx = idx % len(self.seq_indices)
        global_seq_idx = self.seq_indices[local_seq_idx]

        # Temporarily swap index to point to the right sequence
        sample = self.parent._load_sample(global_seq_idx)
        n_tokens = sample["n_tokens"]
        block_size = self.parent.block_size

        max_anchor = n_tokens - block_size
        min_anchor = 1
        if max_anchor < min_anchor:
            max_anchor = min_anchor

        import random as _random
        anchor_pos = _random.randint(min_anchor, max_anchor)

        block_end = min(anchor_pos + block_size, n_tokens)
        block_tokens = sample["tokens"][anchor_pos:block_end]
        if len(block_tokens) < block_size:
            import numpy as np
            block_tokens = np.pad(block_tokens, (0, block_size - len(block_tokens)))

        ctx_end = anchor_pos + 1
        ctx_start = max(0, ctx_end - self.parent.max_ctx_len)
        ctx_len = ctx_end - ctx_start

        import numpy as np
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


if __name__ == "__main__":
    main()
