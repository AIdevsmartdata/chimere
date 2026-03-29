#!/usr/bin/env python3
"""
train_v8.py — Train DFlash v8 drafter with full-sequence multi-anchor data.

Key changes from v7:
  - Uses DFlashFullSeqDirDataset (or DFlashFullSeqDataset for packed shards)
  - Multi-anchor sampling: each sequence yields N anchors per epoch
  - forward_train_multi() with chunked lm_head to fit 16 GB VRAM
  - Validation accuracy computed inside chunks (no full logits tensor)

Usage:
  python scripts/train_v8.py \
    --features-dir data/features_fullseq \
    --output-dir checkpoints_v8 \
    --anchors-per-seq 16 --chunk-size 8 \
    --batch-size 2 --grad-accum 4 --epochs 6 --bf16
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.modeling_v8 import DFlashDraftModelV8
from chimere.data_v8 import DFlashFullSeqDirDataset, DFlashFullSeqDataset, collate_v8


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


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_amp,
                grad_accum_steps, max_grad_norm, chunk_size, epoch=0, log_every=50,
                loss_type="lk"):
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
        anchor_positions = batch["anchor_positions"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _, _ = model.forward_train_multi(
                block_ids, ctx_hidden, ctx_lengths,
                anchor_positions=anchor_positions,
                chunk_size=chunk_size,
                loss_type=loss_type,
            )
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        cur_loss = loss.item() * grad_accum_steps
        step_loss += cur_loss

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == n_steps:
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

        if (batch_idx + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg_step = step_loss / log_every
            lr = optimizer.param_groups[0]["lr"]
            tok_s = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  E{epoch+1} [{batch_idx+1}/{n_steps}] "
                  f"loss={avg_step:.4f} | lr={lr:.2e} | "
                  f"{tok_s:.0f} tok/s | B={block_ids.shape[0]}", flush=True)
            step_loss = 0.0

    avg_loss = total_loss / max(1, n_steps)
    return avg_loss, total_tokens


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp, chunk_size, loss_type="lk"):
    """Evaluate with single forward pass — loss from forward_train_multi, accuracy from chunked lm_head."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0

    for batch in dataloader:
        block_ids = batch["block_input_ids"].to(device)
        ctx_hidden = [h.to(device) for h in batch["context_hidden_list"]]
        ctx_lengths = batch["context_lengths"].to(device)
        anchor_positions = batch["anchor_positions"].to(device)

        B, K = block_ids.shape

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, _, _ = model.forward_train_multi(
                block_ids, ctx_hidden, ctx_lengths,
                anchor_positions=anchor_positions,
                chunk_size=chunk_size,
                loss_type=loss_type,
            )

        total_loss += loss.item() * B
        n_batches += B
        total_count += (K - 1) * B

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, 0.0


def main():
    parser = argparse.ArgumentParser(description="Train DFlash v8 (full-seq multi-anchor)")
    parser.add_argument("--features-dir", type=str, default="data/features_fullseq")
    parser.add_argument("--output-dir", type=str, default="checkpoints_v8")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Sequences per batch (effective = batch * anchors_per_seq)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.04)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-ctx-len", type=int, default=1024)
    parser.add_argument("--anchors-per-seq", type=int, default=16,
                        help="Random anchors sampled per sequence per epoch")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Chunk size for lm_head to limit VRAM")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # Architecture overrides
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    # Dataset format
    parser.add_argument("--packed-shards", type=str, default=None,
                        help="Path to packed shard dir (overrides --features-dir)")
    parser.add_argument("--preload", action="store_true",
                        help="Preload all samples into RAM (eliminates I/O, needs ~10-20 GB)")
    parser.add_argument("--loss-type", type=str, default="lk",
                        choices=["ce", "lk", "lk_alpha"],
                        help="Loss type: ce (CE+smoothing), lk (hybrid LK), lk_alpha (-log α)")
    parser.add_argument("--lk-eta", type=float, default=3.0,
                        help="LK adaptive schedule decay rate")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    effective_batch = args.batch_size * args.anchors_per_seq
    print("=" * 60)
    print(" DFlash v8 — Full-Sequence Multi-Anchor Training")
    print("=" * 60)
    print(f"  Device:          {device}")
    print(f"  Features:        {args.features_dir}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size} seqs x {args.anchors_per_seq} anchors "
          f"= {effective_batch} blocks (x {args.grad_accum} accum "
          f"= {effective_batch * args.grad_accum} effective)")
    print(f"  LR:              {args.lr}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  BF16:            {args.bf16}")
    print(f"  Loss type:       {args.loss_type}" + (f" (η={args.lk_eta})" if args.loss_type.startswith("lk") else ""))
    print(f"  Block size:      {args.block_size}")
    print(f"  Max ctx len:     {args.max_ctx_len}")
    print()

    config_overrides = {"block_size": args.block_size}
    if args.hidden_size is not None:
        config_overrides["hidden_size"] = args.hidden_size
    if args.num_layers is not None:
        config_overrides["num_hidden_layers"] = args.num_layers
    if args.num_heads is not None:
        config_overrides["num_attention_heads"] = args.num_heads
    if args.num_kv_heads is not None:
        config_overrides["num_key_value_heads"] = args.num_kv_heads
    if args.head_dim is not None:
        config_overrides["head_dim"] = args.head_dim
    if args.intermediate_size is not None:
        config_overrides["intermediate_size"] = args.intermediate_size
    if args.dropout is not None:
        config_overrides["attention_dropout"] = args.dropout
    config = DFlashV7Config(**config_overrides)
    config.lk_eta = args.lk_eta

    # Load dataset
    if args.packed_shards:
        print(f"  Dataset: packed shards from {args.packed_shards}")
        full_dataset = DFlashFullSeqDataset(
            args.packed_shards,
            block_size=args.block_size,
            max_ctx_len=args.max_ctx_len,
            anchors_per_seq=args.anchors_per_seq,
            num_layers=config.num_feature_layers,
            hidden_dim=config.target_hidden_size,
        )
    else:
        print(f"  Dataset: per-sample directories from {args.features_dir}")
        full_dataset = DFlashFullSeqDirDataset(
            args.features_dir,
            block_size=args.block_size,
            max_ctx_len=args.max_ctx_len,
            anchors_per_seq=args.anchors_per_seq,
            target_layers=tuple(config.target_layer_ids),
            hidden_dim=config.target_hidden_size,
            preload=args.preload,
        )

    collate_fn = collate_v8

    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=gen
    )

    print(f"  Train: {n_train} sequences ({n_train * args.anchors_per_seq} blocks/epoch)")
    print(f"  Val:   {n_val} sequences ({n_val * args.anchors_per_seq} blocks/epoch)")
    print()

    def _worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed() % 2**32)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    # Create model
    model = DFlashDraftModelV8(config)

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
    else:
        print(f"  WARNING: {qwen_weights_path} not found!")

    model = model.to(device)

    if not args.no_compile and hasattr(torch, "compile"):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    # Freeze shared params
    model.freeze_shared_params()
    model.enable_gradient_checkpointing()
    for p in model.embed_tokens.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.lm_head.parameters():
        p.data = p.data.to(torch.bfloat16)
    for p in model.norm.parameters():
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
        fused=True,
    )

    if args.resume and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    num_training_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    if args.resume and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    use_amp = args.bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    if not config_path.exists():
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(config), f, indent=2)

    # Save training args
    with open(output_dir / "train_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"  Training steps: {num_training_steps} (warmup: {num_warmup_steps})")
    print("=" * 60)
    print()

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss, tokens = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, use_amp, args.grad_accum, config.max_grad_norm,
            args.chunk_size, epoch=epoch, loss_type=args.loss_type,
        )

        val_loss, val_acc = evaluate(model, val_loader, device, use_amp, args.chunk_size,
                                     loss_type=args.loss_type)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{start_epoch + args.epochs} "
              f"| train_loss={train_loss:.4f} "
              f"| val_loss={val_loss:.4f} "
              f"| val_acc={val_acc:.4f} "
              f"| lr={lr_now:.2e} "
              f"| {elapsed:.1f}s "
              f"| {tokens / elapsed:.0f} tok/s")

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
            "version": "v8_multi_anchor",
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
                "version": "v8_multi_anchor",
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
