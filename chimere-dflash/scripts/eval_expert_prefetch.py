#!/usr/bin/env python3
"""
Zero-shot evaluation of ExpertPrefetchHead.

Evaluates the ability of ExpertPrefetchHead to predict MoE expert routing
for Qwen3.5-35B-A3B CPU-offloaded layers (20-39) from drafter hidden states.

Since we have the frozen gate weights extracted from the GGUF model, ground-truth
routing can be computed directly without running the full target model:

    gate_logits = hidden_state @ gate_weight.T   # [256]
    gt_top8 = topk(gate_logits, 8).indices

The challenge: the gate weights expect the actual hidden state at layer L's
input (post-residual at layer L-1), but we only have hidden states at feature
extraction layers [1, 10, 19, 28, 37].

Evaluation strategy:
  - For layers 20-27: use feature layer 19 as a proxy (closest preceding layer).
  - For layers 28-37: use feature layer 28 as a proxy.
  - For layers 38-39: use feature layer 37 as a proxy.

Note: these are PROXY ground-truth labels, not exact labels. They measure how
well the drafter (given features from [1,10,19,28,37]) can approximate routing
decisions relative to what would happen if we used the nearest actual hidden state.

Three evaluation modes are provided:
  1. proxy_recall      — recall vs. proxy ground-truth (as described above)
  2. self_consistency  — same input always gives same output
  3. overlap_diversity — overlap between predictions on different random inputs

Usage:
    python scripts/eval_expert_prefetch.py \\
        --gate-weights data/qwen_gate_weights_20_39.pt \\
        --buffer data/online_buffer_merged \\
        --n-samples 50 \\
        --device cpu

    # With drafter checkpoint (optional — uses random weights if not provided):
    python scripts/eval_expert_prefetch.py \\
        --gate-weights data/qwen_gate_weights_20_39.pt \\
        --checkpoint checkpoints_v8_online_c4/best.pt \\
        --buffer data/online_buffer_merged \\
        --n-samples 50
"""
import argparse
import json
import sys
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chimere.config_v7 import DFlashV7Config
from chimere.expert_prefetch import ExpertPrefetchConfig, ExpertPrefetchHead
from chimere.modeling_v8 import DFlashDraftModelV8


# ---------------------------------------------------------------------------
# Feature layer → proxy MoE layer mapping
# ---------------------------------------------------------------------------

FEATURE_LAYERS = [1, 10, 19, 28, 37]

def _proxy_feature_layer(moe_layer: int) -> int:
    """Return the nearest feature layer index (into FEATURE_LAYERS) for a given MoE layer."""
    # Find the feature layer with the smallest absolute distance
    best_idx = 0
    best_dist = abs(moe_layer - FEATURE_LAYERS[0])
    for i, fl in enumerate(FEATURE_LAYERS):
        d = abs(moe_layer - fl)
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def build_proxy_gt_routing(
    hidden: np.ndarray,
    gate_weights: Dict[int, torch.Tensor],
    routing_layers: List[int],
    n_active: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Dict[int, torch.Tensor]:
    """Compute proxy ground-truth routing for all routing layers.

    For each MoE layer L, uses the hidden state from the nearest feature
    extraction layer as a proxy for the true hidden state at layer L's input.

    Args:
        hidden: float16 numpy array [n_layers, seq_len, hidden_dim]
                where n_layers corresponds to FEATURE_LAYERS = [1,10,19,28,37].
        gate_weights: {layer_id: tensor[n_experts, hidden_dim]} — frozen gate matrices.
        routing_layers: list of MoE layer indices to compute GT for.
        n_active: number of top experts to select.
        device: compute device.

    Returns:
        {layer_id: tensor[seq_len, n_active]} — ground-truth top-k expert indices.
    """
    gt_routing: Dict[int, torch.Tensor] = {}

    # Pre-convert all feature-layer hidden states to float32 tensors
    n_feature_layers, seq_len, hidden_dim = hidden.shape
    feature_tensors = [
        torch.from_numpy(hidden[i].astype(np.float32)).to(device)
        for i in range(n_feature_layers)
    ]

    for layer_id in routing_layers:
        if layer_id not in gate_weights:
            continue

        # Select proxy feature layer
        fl_idx = _proxy_feature_layer(layer_id)
        h = feature_tensors[fl_idx]            # [seq_len, hidden_dim]
        W = gate_weights[layer_id].to(device)   # [n_experts, hidden_dim]

        # Gate logits via matmul (equivalent to the target model's gating)
        logits = h @ W.t()                     # [seq_len, n_experts]
        _, indices = logits.topk(n_active, dim=-1)  # [seq_len, n_active]
        gt_routing[layer_id] = indices

    return gt_routing


# ---------------------------------------------------------------------------
# Sample loading
# ---------------------------------------------------------------------------

def load_sample_v2(
    sample_dir: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Load a V2-format sample directory.

    Returns (hidden, tokens, metadata) or None if the sample is invalid.
    hidden: float16 [n_layers, n_positions, hidden_dim]
    tokens: int32 [n_positions]
    """
    meta_path = sample_dir / "metadata.json"
    hidden_path = sample_dir / "context_hidden.bin"
    tokens_path = sample_dir / "tokens.bin"

    if not (meta_path.exists() and hidden_path.exists() and tokens_path.exists()):
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    n_layers = len(meta.get("layers", [1, 10, 19, 28, 37]))
    n_pos = meta.get("n_positions", meta.get("seq_len", 0))
    h_dim = meta.get("hidden_dim", 2048)

    if n_pos == 0:
        return None

    expected_bytes = n_layers * n_pos * h_dim * 2  # float16 = 2 bytes
    if hidden_path.stat().st_size != expected_bytes:
        return None

    hidden = np.fromfile(hidden_path, dtype=np.float16).reshape(n_layers, n_pos, h_dim)
    tokens = np.fromfile(tokens_path, dtype=np.int32)

    # Validate no NaN/Inf
    if np.any(np.isnan(hidden.astype(np.float32))) or np.any(np.isinf(hidden.astype(np.float32))):
        return None

    return hidden, tokens, meta


def load_buffer_samples(
    buffer_dir: Path,
    n_samples: int,
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Load up to n_samples V2 samples from a buffer directory."""
    sample_dirs = sorted([
        d for d in buffer_dir.iterdir()
        if d.is_dir() and d.name.startswith("sample_")
    ])

    if not sample_dirs:
        raise ValueError(f"No sample_* directories found in {buffer_dir}")

    samples = []
    for d in sample_dirs:
        if len(samples) >= n_samples:
            break
        result = load_sample_v2(d)
        if result is not None:
            samples.append(result)

    print(f"Loaded {len(samples)} valid samples from {buffer_dir}")
    return samples


# ---------------------------------------------------------------------------
# Drafter loading (optional — zero-shot uses random weights if no checkpoint)
# ---------------------------------------------------------------------------

def load_drafter_or_random(
    checkpoint_path: Optional[str],
    device: torch.device,
) -> Tuple[DFlashDraftModelV8, DFlashV7Config]:
    """Load drafter model from checkpoint or create with random weights."""
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = DFlashV7Config(**{
            f.name: ckpt["config"][f.name]
            for f in dc_fields(DFlashV7Config)
            if f.name in ckpt["config"]
        })
        # Ensure expert routing is enabled
        config.predict_expert_routing = True
        model = DFlashDraftModelV8(config)
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in ckpt["model_state_dict"].items()
        }
        # Load only matching keys (expert_prefetch may not be in checkpoint)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys (will use random init): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        print(f"Drafter loaded from {checkpoint_path}")
    else:
        print("No checkpoint provided — using random drafter weights (zero-shot)")
        config = DFlashV7Config(predict_expert_routing=True)
        model = DFlashDraftModelV8(config)

    model = model.to(device).eval().float()
    return model, config


# ---------------------------------------------------------------------------
# Core evaluation: proxy recall
# ---------------------------------------------------------------------------

def eval_proxy_recall(
    model: DFlashDraftModelV8,
    gate_weights: Dict[int, torch.Tensor],
    samples: List[Tuple[np.ndarray, np.ndarray, dict]],
    config: DFlashV7Config,
    device: torch.device,
    max_ctx_len: int = 256,
) -> dict:
    """Evaluate top-k recall against proxy ground-truth routing.

    For each sample, we:
      1. Build a batch of drafter inputs from a context window.
      2. Run the drafter to get fused context hidden states.
      3. Run ExpertPrefetchHead to predict top-8 experts per layer.
      4. Compute proxy GT routing from the same context hidden states.
      5. Compute recall = |predicted ∩ GT| / n_active per layer.

    Returns a dict with per-layer and mean recall statistics.
    """
    prefetch_head = model.expert_prefetch
    if prefetch_head is None:
        raise ValueError("Model does not have expert_prefetch head. "
                         "Ensure config.predict_expert_routing=True.")

    routing_layers = config.routing_layers
    n_active = config.num_active_experts
    n_feature_layers = config.num_feature_layers

    # Accumulators: per layer, collect recall values over all (sample, position) pairs
    layer_recall_sums: Dict[int, float] = {l: 0.0 for l in routing_layers}
    layer_recall_counts: Dict[int, int] = {l: 0 for l in routing_layers}

    t_drafter_total = 0.0
    t_prefetch_total = 0.0
    t_gt_total = 0.0

    for sample_idx, (hidden, tokens, meta) in enumerate(samples):
        n_pos = hidden.shape[1]
        if n_pos < 2:
            continue

        # Use last max_ctx_len positions as context
        ctx_start = max(0, n_pos - max_ctx_len)
        ctx_end = n_pos
        ctx_len = ctx_end - ctx_start

        # Build context hidden list: [n_layers × tensor[1, ctx_len, H]]
        context_hidden_list = []
        for li in range(n_feature_layers):
            h = hidden[li, ctx_start:ctx_end, :].astype(np.float32)  # [ctx_len, H]
            t = torch.from_numpy(h).unsqueeze(0).to(device)          # [1, ctx_len, H]
            context_hidden_list.append(t)

        ctx_lengths = torch.tensor([ctx_len], device=device, dtype=torch.long)
        anchor_positions = torch.tensor([ctx_end - 1], device=device, dtype=torch.long)

        # Step 1: Drafter forward — fuse context into hidden representation
        t0 = time.perf_counter()
        with torch.no_grad():
            ctx = model._fuse_context(context_hidden_list)  # [1, ctx_len, H]
        t_drafter_total += time.perf_counter() - t0

        # Step 2: ExpertPrefetchHead prediction
        t0 = time.perf_counter()
        with torch.no_grad():
            expert_indices, gate_logits = prefetch_head(ctx)
            # expert_indices: {layer_id: [1, ctx_len, n_active]}
        t_prefetch_total += time.perf_counter() - t0

        # Step 3: Proxy ground-truth routing (computed from feature hidden states)
        t0 = time.perf_counter()
        gt_routing = build_proxy_gt_routing(
            hidden[:, ctx_start:ctx_end, :],
            gate_weights,
            routing_layers,
            n_active=n_active,
            device=device,
        )
        # gt_routing: {layer_id: [ctx_len, n_active]}
        t_gt_total += time.perf_counter() - t0

        # Step 4: Compute recall per layer
        for layer_id in routing_layers:
            if layer_id not in expert_indices or layer_id not in gt_routing:
                continue

            pred = expert_indices[layer_id][0]  # [ctx_len, n_active]
            gt = gt_routing[layer_id]            # [ctx_len, n_active]

            # Broadcast intersection: [ctx_len, n_active, 1] vs [ctx_len, 1, n_active]
            pred_exp = pred.unsqueeze(-1)         # [ctx_len, n_active, 1]
            gt_exp = gt.unsqueeze(-2)             # [ctx_len, 1, n_active]
            matched = (pred_exp == gt_exp).any(dim=-1).float()  # [ctx_len, n_active]
            recall = matched.mean().item()

            layer_recall_sums[layer_id] += recall
            layer_recall_counts[layer_id] += 1

        if (sample_idx + 1) % 10 == 0:
            print(f"  [{sample_idx + 1}/{len(samples)}] sample processed", flush=True)

    # Aggregate results
    per_layer_recall = {}
    for layer_id in routing_layers:
        count = layer_recall_counts[layer_id]
        if count > 0:
            per_layer_recall[layer_id] = layer_recall_sums[layer_id] / count
        else:
            per_layer_recall[layer_id] = float("nan")

    valid_recalls = [v for v in per_layer_recall.values() if not np.isnan(v)]
    mean_recall = float(np.mean(valid_recalls)) if valid_recalls else float("nan")

    n = len(samples)
    return {
        "mean_recall": mean_recall,
        "per_layer_recall": per_layer_recall,
        "n_samples": n,
        "timing": {
            "drafter_fuse_ms_per_sample": t_drafter_total * 1000 / max(1, n),
            "prefetch_ms_per_sample": t_prefetch_total * 1000 / max(1, n),
            "gt_compute_ms_per_sample": t_gt_total * 1000 / max(1, n),
        },
    }


# ---------------------------------------------------------------------------
# Core evaluation: self-consistency
# ---------------------------------------------------------------------------

def eval_self_consistency(
    model: DFlashDraftModelV8,
    samples: List[Tuple[np.ndarray, np.ndarray, dict]],
    config: DFlashV7Config,
    device: torch.device,
    n_runs: int = 5,
    n_test_samples: int = 10,
) -> dict:
    """Test that the same input always produces the same expert predictions.

    Since ExpertPrefetchHead is deterministic at eval time (no dropout),
    all n_runs should produce identical results.

    Returns:
        consistency_rate: fraction of (sample, layer, position) triples
                          where all n_runs agree on the top-8 experts.
    """
    prefetch_head = model.expert_prefetch
    routing_layers = config.routing_layers
    n_feature_layers = config.num_feature_layers

    test_samples = samples[:n_test_samples]
    total_positions = 0
    consistent_positions = 0

    for sample_idx, (hidden, tokens, meta) in enumerate(test_samples):
        n_pos = hidden.shape[1]
        ctx_len = min(n_pos, 64)  # small context for speed
        ctx_start = max(0, n_pos - ctx_len)

        context_hidden_list = []
        for li in range(n_feature_layers):
            h = hidden[li, ctx_start:, :].astype(np.float32)
            t = torch.from_numpy(h).unsqueeze(0).to(device)
            context_hidden_list.append(t)

        # Run n_runs times
        all_predictions: List[Dict[int, torch.Tensor]] = []
        with torch.no_grad():
            ctx = model._fuse_context(context_hidden_list)
            for _ in range(n_runs):
                expert_indices, _ = prefetch_head(ctx)
                all_predictions.append({
                    k: v[0].clone() for k, v in expert_indices.items()
                })

        # Check consistency across runs
        for layer_id in routing_layers:
            if layer_id not in all_predictions[0]:
                continue
            ref = all_predictions[0][layer_id]  # [ctx_len, n_active]
            for run_idx in range(1, n_runs):
                pred = all_predictions[run_idx][layer_id]
                # Sort each position's predictions to check set equality
                ref_sorted = ref.sort(dim=-1).values
                pred_sorted = pred.sort(dim=-1).values
                matches = (ref_sorted == pred_sorted).all(dim=-1)  # [ctx_len]
                consistent_positions += matches.sum().item()
                total_positions += ref.shape[0]

    consistency_rate = consistent_positions / max(1, total_positions)
    return {
        "consistency_rate": consistency_rate,
        "n_test_samples": n_test_samples,
        "n_runs": n_runs,
        "total_positions_checked": total_positions,
    }


# ---------------------------------------------------------------------------
# Core evaluation: overlap diversity
# ---------------------------------------------------------------------------

def eval_overlap_diversity(
    model: DFlashDraftModelV8,
    samples: List[Tuple[np.ndarray, np.ndarray, dict]],
    config: DFlashV7Config,
    device: torch.device,
    n_test_samples: int = 20,
) -> dict:
    """Measure pairwise prediction overlap between different inputs.

    A model that always predicts the same experts regardless of input
    has 100% overlap but no predictive value. A good model adapts
    its predictions to the input, giving lower (but non-trivial) overlap.

    For n_active=8, n_experts=256:
        Random baseline: E[overlap] = 8/256 ≈ 3.1%
        Always-same baseline: E[overlap] = 100%
        Useful model: somewhere context-dependent.

    Returns:
        mean_pairwise_overlap: mean |A ∩ B| / n_active for random input pairs.
    """
    prefetch_head = model.expert_prefetch
    routing_layers = config.routing_layers
    n_active = config.num_active_experts
    n_feature_layers = config.num_feature_layers

    test_samples = samples[:n_test_samples]

    # Collect one prediction per sample (from last position only)
    per_sample_preds: List[Dict[int, torch.Tensor]] = []
    with torch.no_grad():
        for hidden, tokens, meta in test_samples:
            n_pos = hidden.shape[1]
            ctx_len = min(n_pos, 32)
            ctx_start = max(0, n_pos - ctx_len)

            context_hidden_list = []
            for li in range(n_feature_layers):
                h = hidden[li, ctx_start:, :].astype(np.float32)
                t = torch.from_numpy(h).unsqueeze(0).to(device)
                context_hidden_list.append(t)

            ctx = model._fuse_context(context_hidden_list)
            expert_indices, _ = prefetch_head(ctx)
            # Extract last position predictions only
            per_sample_preds.append({
                k: v[0, -1, :].clone() for k, v in expert_indices.items()
            })

    # Compute pairwise overlaps
    n = len(per_sample_preds)
    layer_overlaps: Dict[int, List[float]] = {l: [] for l in routing_layers}

    for i in range(n):
        for j in range(i + 1, n):
            for layer_id in routing_layers:
                if layer_id not in per_sample_preds[i]:
                    continue
                a = per_sample_preds[i][layer_id].unsqueeze(-1)  # [n_active, 1]
                b = per_sample_preds[j][layer_id].unsqueeze(-2)  # [1, n_active]
                overlap = (a == b).any(dim=-1).float().mean().item()
                layer_overlaps[layer_id].append(overlap)

    per_layer_overlap = {
        l: float(np.mean(v)) if v else float("nan")
        for l, v in layer_overlaps.items()
    }
    all_overlaps = [v for v in per_layer_overlap.values() if not np.isnan(v)]
    mean_overlap = float(np.mean(all_overlaps)) if all_overlaps else float("nan")

    random_baseline = n_active / 256.0

    return {
        "mean_pairwise_overlap": mean_overlap,
        "random_baseline_overlap": random_baseline,
        "per_layer_overlap": per_layer_overlap,
        "n_test_samples": n_test_samples,
        "n_pairs": n * (n - 1) // 2,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_separator(char: str = "=", width: int = 64):
    print(char * width)


def print_results(results: dict, title: str):
    print_separator()
    print(f"  {title}")
    print_separator()
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, float):
                    print(f"    layer {sub_k}: {sub_v:.4f}")
                else:
                    print(f"    {sub_k}: {sub_v}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of ExpertPrefetchHead",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gate-weights",
        type=str,
        default="data/qwen_gate_weights_20_39.pt",
        help="Path to gate weights .pt file (default: data/qwen_gate_weights_20_39.pt)",
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default="data/online_buffer_merged",
        help="Buffer directory containing sample_* subdirectories",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional drafter checkpoint (.pt). If omitted, random weights are used.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device (default: cpu)",
    )
    parser.add_argument(
        "--max-ctx-len",
        type=int,
        default=256,
        help="Maximum context length to use per sample (default: 256)",
    )
    parser.add_argument(
        "--n-self-consistency-runs",
        type=int,
        default=5,
        help="Number of forward passes for self-consistency test (default: 5)",
    )
    parser.add_argument(
        "--routing-layers",
        type=str,
        default=None,
        help="Comma-separated MoE layers to evaluate (default: 20-39 from config)",
    )
    parser.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip the (slow) proxy-recall evaluation",
    )
    parser.add_argument(
        "--skip-diversity",
        action="store_true",
        help="Skip pairwise overlap diversity evaluation",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write results as JSON",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    project_root = Path(__file__).resolve().parent.parent

    # Resolve paths relative to project root if relative
    gate_weights_path = Path(args.gate_weights)
    if not gate_weights_path.is_absolute():
        gate_weights_path = project_root / gate_weights_path

    buffer_path = Path(args.buffer)
    if not buffer_path.is_absolute():
        buffer_path = project_root / buffer_path

    checkpoint_path = args.checkpoint
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        if not cp.is_absolute():
            checkpoint_path = str(project_root / cp)

    # ------------------------------------------------------------------
    # Load gate weights
    # ------------------------------------------------------------------
    print(f"\nLoading gate weights from {gate_weights_path} ...")
    gate_weights_raw = torch.load(gate_weights_path, map_location="cpu", weights_only=True)
    # Normalise keys to int
    gate_weights: Dict[int, torch.Tensor] = {}
    for k, v in gate_weights_raw.items():
        gate_weights[int(k)] = v
    print(f"  Loaded {len(gate_weights)} gate matrices, "
          f"shape: {next(iter(gate_weights.values())).shape}")

    # ------------------------------------------------------------------
    # Load drafter model with ExpertPrefetchHead
    # ------------------------------------------------------------------
    print(f"\nLoading drafter model ...")
    model, config = load_drafter_or_random(checkpoint_path, device)

    # Override routing_layers if specified
    if args.routing_layers is not None:
        config.routing_layers = [int(x) for x in args.routing_layers.split(",")]

    # Load gate weights into the prefetch head
    model.expert_prefetch.load_gate_weights(str(gate_weights_path))

    # Parameter summary
    trainable = model.expert_prefetch.trainable_parameters()
    frozen = model.expert_prefetch.frozen_parameters()
    print(f"\nExpertPrefetchHead parameters:")
    print(f"  Trainable: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"  Frozen:    {frozen:,} ({frozen / 1e6:.2f}M)")
    print(f"  Routing layers: {config.routing_layers}")
    print(f"  n_experts={config.num_target_experts}, n_active={config.num_active_experts}")

    # ------------------------------------------------------------------
    # Load buffer samples
    # ------------------------------------------------------------------
    print(f"\nLoading up to {args.n_samples} samples from {buffer_path} ...")
    samples = load_buffer_samples(buffer_path, args.n_samples)

    if not samples:
        print("ERROR: No valid samples found. Exiting.")
        sys.exit(1)

    all_results = {}

    # ------------------------------------------------------------------
    # Evaluation 1: Proxy recall
    # ------------------------------------------------------------------
    if not args.skip_recall:
        print(f"\n[1/3] Proxy Recall Evaluation ({len(samples)} samples) ...")
        recall_results = eval_proxy_recall(
            model=model,
            gate_weights=gate_weights,
            samples=samples,
            config=config,
            device=device,
            max_ctx_len=args.max_ctx_len,
        )
        print_results(recall_results, "PROXY RECALL")

        # Summarise per-layer recall in bands (20-27, 28-37, 38-39)
        layer_ids = sorted(recall_results["per_layer_recall"].keys())
        bands = [
            ("layers 20-27", [l for l in layer_ids if 20 <= l <= 27]),
            ("layers 28-37", [l for l in layer_ids if 28 <= l <= 37]),
            ("layers 38-39", [l for l in layer_ids if 38 <= l <= 39]),
        ]
        print("\n  Per-band recall:")
        for band_name, band_layers in bands:
            vals = [recall_results["per_layer_recall"][l] for l in band_layers
                    if not np.isnan(recall_results["per_layer_recall"].get(l, float("nan")))]
            if vals:
                print(f"    {band_name}: {np.mean(vals):.4f} ({len(vals)} layers)")

        all_results["proxy_recall"] = recall_results
    else:
        print("\n[1/3] Proxy recall: SKIPPED")

    # ------------------------------------------------------------------
    # Evaluation 2: Self-consistency
    # ------------------------------------------------------------------
    print(f"\n[2/3] Self-Consistency Test ...")
    consistency_results = eval_self_consistency(
        model=model,
        samples=samples,
        config=config,
        device=device,
        n_runs=args.n_self_consistency_runs,
        n_test_samples=min(10, len(samples)),
    )
    print_results(consistency_results, "SELF-CONSISTENCY")
    if consistency_results["consistency_rate"] < 0.9999:
        print("  WARNING: Predictions are not deterministic! "
              "Check for training-mode dropout or random ops.")
    else:
        print("  PASS: Predictions are perfectly deterministic.")
    all_results["self_consistency"] = consistency_results

    # ------------------------------------------------------------------
    # Evaluation 3: Overlap diversity
    # ------------------------------------------------------------------
    if not args.skip_diversity:
        print(f"\n[3/3] Pairwise Overlap Diversity Test ...")
        diversity_results = eval_overlap_diversity(
            model=model,
            samples=samples,
            config=config,
            device=device,
            n_test_samples=min(20, len(samples)),
        )
        print_results(diversity_results, "PAIRWISE OVERLAP DIVERSITY")
        rng_baseline = diversity_results["random_baseline_overlap"]
        mean_overlap = diversity_results["mean_pairwise_overlap"]
        if mean_overlap > 0.8:
            verdict = "HIGH — model predicts similar experts for all inputs (may be degenerate)"
        elif mean_overlap > 2 * rng_baseline:
            verdict = "MODERATE — model has learned some input-independent structure"
        else:
            verdict = "LOW — predictions are diverse (close to random baseline)"
        print(f"\n  Random baseline: {rng_baseline:.4f}")
        print(f"  Mean overlap:    {mean_overlap:.4f}")
        print(f"  Verdict: {verdict}")
        all_results["overlap_diversity"] = diversity_results
    else:
        print("\n[3/3] Overlap diversity: SKIPPED")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_separator()
    print("  SUMMARY")
    print_separator()
    if "proxy_recall" in all_results:
        mr = all_results["proxy_recall"]["mean_recall"]
        print(f"  Mean proxy recall (all layers): {mr:.4f} "
              f"({'good' if mr > 0.5 else 'poor'} — SOTA target: ~0.8+)")
    if "self_consistency" in all_results:
        cr = all_results["self_consistency"]["consistency_rate"]
        print(f"  Self-consistency rate:          {cr:.4f} (expected: 1.0)")
    if "overlap_diversity" in all_results:
        mo = all_results["overlap_diversity"]["mean_pairwise_overlap"]
        rb = all_results["overlap_diversity"]["random_baseline_overlap"]
        print(f"  Mean pairwise overlap:          {mo:.4f} (random baseline: {rb:.4f})")
    print_separator()

    # ------------------------------------------------------------------
    # Optional JSON output
    # ------------------------------------------------------------------
    if args.output_json:
        # Convert int keys to strings for JSON serialisation
        def _jsonify(obj):
            if isinstance(obj, dict):
                return {str(k): _jsonify(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_jsonify(v) for v in obj]
            elif isinstance(obj, float) and np.isnan(obj):
                return None
            else:
                return obj

        import json as _json
        with open(args.output_json, "w") as f:
            _json.dump(_jsonify(all_results), f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
