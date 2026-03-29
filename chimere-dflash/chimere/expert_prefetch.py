"""
Expert Prefetch Head for MoE speculative decoding.

Predicts which MoE experts will be activated during target model verification,
enabling async CPU→GPU prefetch of expert weights while the drafter generates.

Architecture (SP-MoE style):
    drafter_output [B, K, input_dim]
        → shared_proj: Linear(input_dim, proj_rank) + GELU + Linear(proj_rank, input_dim)
        → For each CPU layer L in routing_layers:
            adapter_L: Linear(input_dim, adapter_rank) + GELU + Linear(adapter_rank, input_dim)
            h_L = shared(x) + adapter_L(x)
            gate_logits_L = frozen_gate_L(h_L)    [input_dim → n_experts]
            top_{n_active}_L = topk(gate_logits_L, n_active)
        → expert_predictions: {layer_id: [B, K, n_active]}

Parameters (default config):
    ~7.3M trainable  (shared_proj: 2*2048*512 + 20 adapters: 20*2*2048*64)
    ~10.5M frozen    (20 gate matrices: 20 * 256 * 2048)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertPrefetchConfig:
    """Configuration for the expert prefetch head.

    Fields must be kept in sync with DFlashV7Config.routing_* counterparts.
    """
    # Which target model layers to predict routing for (typically CPU-offloaded layers)
    routing_layers: List[int] = field(default_factory=lambda: list(range(20, 40)))
    # Total experts per MoE layer (Qwen3.5-35B-A3B: 256)
    n_experts: int = 256
    # Top-K experts selected per token (Qwen3.5-35B-A3B: 8)
    n_active: int = 8
    # Drafter hidden dimension — must match DFlashV7Config.hidden_size (= target n_embd)
    input_dim: int = 2048
    # Shared projection bottleneck rank
    proj_rank: int = 512
    # Per-layer adapter bottleneck rank
    adapter_rank: int = 64


class ExpertPrefetchHead(nn.Module):
    """Predicts top-K expert routing for CPU-offloaded MoE layers.

    Uses frozen gate weights extracted from the target model (Qwen3.5-35B-A3B)
    plus a small set of learned shared+adapter projections to map drafter hidden
    states to expert routing predictions.

    The gate weights (shape [n_experts, input_dim] per layer) are loaded from a
    pre-extracted .pt file and kept frozen throughout training. Only the shared
    projection and per-layer adapters are trained.
    """

    def __init__(self, config: ExpertPrefetchConfig):
        super().__init__()
        self.config = config
        H = config.input_dim

        # ------------------------------------------------------------------
        # Shared projection: applied once across all layers
        # down: [H → proj_rank], up: [proj_rank → H]
        # ------------------------------------------------------------------
        self.shared_down = nn.Linear(H, config.proj_rank, bias=False)
        self.shared_up = nn.Linear(config.proj_rank, H, bias=False)

        # ------------------------------------------------------------------
        # Per-layer adapters: lightweight residual branches
        # adapters[str(layer_idx)] = nn.ModuleDict({"down": ..., "up": ...})
        # Using a two-level ModuleDict so submodules are properly registered.
        # ------------------------------------------------------------------
        self.adapters: nn.ModuleDict = nn.ModuleDict()
        for layer_idx in config.routing_layers:
            self.adapters[str(layer_idx)] = nn.ModuleDict({
                "down": nn.Linear(H, config.adapter_rank, bias=False),
                "up":   nn.Linear(config.adapter_rank, H, bias=False),
            })

        # ------------------------------------------------------------------
        # Frozen gate matrices: loaded from GGUF extraction
        # gates[str(layer_idx)] = Linear(H, n_experts, bias=False) — FROZEN
        # Shape of weight: [n_experts, H] = [256, 2048]
        # ------------------------------------------------------------------
        self.gates: nn.ModuleDict = nn.ModuleDict()
        for layer_idx in config.routing_layers:
            gate = nn.Linear(H, config.n_experts, bias=False)
            gate.weight.requires_grad_(False)   # frozen from the start
            self.gates[str(layer_idx)] = gate

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Initialize shared projection with small gain; adapters near zero.

        Adapters start near-zero so the initial forward pass is dominated by
        the shared projection, allowing stable early training.
        """
        nn.init.kaiming_normal_(self.shared_down.weight, a=0.01)
        nn.init.kaiming_normal_(self.shared_up.weight, a=0.01)

        for key in self.adapters:
            nn.init.normal_(self.adapters[key]["down"].weight, std=0.01)
            nn.init.normal_(self.adapters[key]["up"].weight, std=0.01)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_gate_weights(self, path: str) -> None:
        """Load frozen gate weights from extracted GGUF data.

        The .pt file should contain a dict mapping layer index to a tensor of
        shape [n_experts, input_dim] (e.g. {20: tensor, 21: tensor, ...}).
        Integer keys (int) and string keys (str) are both accepted.

        Args:
            path: Path to ``data/qwen_gate_weights_20_39.pt`` or equivalent.
        """
        gate_dict = torch.load(path, map_location="cpu", weights_only=True)

        loaded = 0
        for layer_idx in self.config.routing_layers:
            key = str(layer_idx)
            # Accept both int and str keys in the saved dict
            weight = gate_dict.get(layer_idx, gate_dict.get(key))
            if weight is None:
                print(f"WARNING: gate weights for layer {layer_idx} not found in {path}")
                continue
            self.gates[key].weight.data.copy_(weight)  # [n_experts, input_dim]
            self.gates[key].weight.requires_grad_(False)
            loaded += 1

        print(
            f"ExpertPrefetchHead: loaded {loaded}/{len(self.config.routing_layers)} "
            f"frozen gate matrices from {path}"
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        """Predict expert routing for all configured target layers.

        Args:
            x: ``[B, K, input_dim]`` — drafter output hidden states
               (last hidden state before LM head, after norm).

        Returns:
            expert_indices: ``{layer_idx: [B, K, n_active]}`` — predicted
                top-K expert indices for each target layer.
            gate_logits:    ``{layer_idx: [B, K, n_experts]}`` — full gate
                logits, needed to compute the training loss.
        """
        # Shared projection (computed once, reused for every layer)
        shared = self.shared_up(F.gelu(self.shared_down(x)))  # [B, K, H]

        expert_indices: Dict[int, torch.Tensor] = {}
        gate_logits: Dict[int, torch.Tensor] = {}

        for layer_idx in self.config.routing_layers:
            key = str(layer_idx)

            # Per-layer adapter: residual branch over original x
            adapter_out = self.adapters[key]["up"](
                F.gelu(self.adapters[key]["down"](x))
            )  # [B, K, H]

            # Combine shared context + layer-specific adaptation
            h = shared + adapter_out  # [B, K, H]

            # Apply frozen gate from the target model
            logits = self.gates[key](h)           # [B, K, n_experts]
            gate_logits[layer_idx] = logits

            # Select top-K experts
            _, indices = logits.topk(self.config.n_active, dim=-1)  # [B, K, n_active]
            expert_indices[layer_idx] = indices

        return expert_indices, gate_logits

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self,
        gate_logits: Dict[int, torch.Tensor],
        target_routing: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Top-K recall loss: maximise probability mass on ground-truth experts.

        For each token position and each layer we create a binary target mask
        (1 for each of the n_active ground-truth experts, 0 elsewhere) and
        optimise binary cross-entropy over sigmoid logits.

        This is equivalent to multi-label classification where each token
        simultaneously belongs to n_active classes out of n_experts.

        Args:
            gate_logits:    ``{layer_idx: [B, K, n_experts]}`` from ``forward()``.
            target_routing: ``{layer_idx: [B, K, n_active]}`` ground-truth expert
                            indices (integer dtype).

        Returns:
            loss: scalar — mean BCE across all layers that have matching labels.
        """
        total_loss = 0.0
        n_layers = 0

        for layer_idx, logits in gate_logits.items():
            if layer_idx not in target_routing:
                continue

            targets = target_routing[layer_idx].long()  # [B, K, n_active]
            B, K, E = logits.shape

            probs = torch.sigmoid(logits)   # [B, K, E]  (independent per expert)

            # Build binary target mask: 1 at each ground-truth expert position
            target_mask = torch.zeros_like(probs)
            target_mask.scatter_(2, targets, 1.0)   # [B, K, E]

            # Binary cross-entropy averaged over all (position, expert) pairs
            loss = F.binary_cross_entropy(probs, target_mask, reduction="mean")
            total_loss += loss
            n_layers += 1

        return total_loss / max(1, n_layers)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_experts(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Inference-only path: return expert indices without storing logits.

        Args:
            x: ``[B, K, input_dim]`` — drafter hidden states.

        Returns:
            ``{layer_idx: [B, K, n_active]}`` — predicted expert indices.
        """
        expert_indices, _ = self.forward(x)
        return expert_indices

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def compute_topk_recall(
        self,
        expert_indices: Dict[int, torch.Tensor],
        routing_labels: Dict[int, torch.Tensor],
    ) -> Dict[str, object]:
        """Compute top-K recall metrics (evaluation only, not used for training).

        For each token position, recall = |predicted ∩ ground_truth| / n_active.

        Args:
            expert_indices: ``{layer_idx: [B, K, n_active]}`` — predicted indices.
            routing_labels: ``{layer_idx: [B, K, n_active]}`` — ground truth indices.

        Returns:
            Dict with keys ``"mean_recall"`` (float) and
            ``"per_layer_recall"`` ({layer_idx: float}).
        """
        recalls: Dict[int, float] = {}

        for layer_idx, pred in expert_indices.items():
            if layer_idx not in routing_labels:
                continue
            gt = routing_labels[layer_idx]  # [B, K, n_active]

            # Expand for set-intersection via broadcasting:
            #   pred [B, K, n_active, 1] vs gt [B, K, 1, n_active]
            pred_exp = pred.unsqueeze(-1)   # [B, K, n_active, 1]
            gt_exp   = gt.unsqueeze(-2)     # [B, K, 1, n_active]
            # For each predicted expert, check if it appears anywhere in gt
            matched = (pred_exp == gt_exp).any(dim=-1).float()  # [B, K, n_active]
            recalls[layer_idx] = matched.mean().item()

        mean_recall = sum(recalls.values()) / max(len(recalls), 1)
        return {"mean_recall": mean_recall, "per_layer_recall": recalls}

    # ------------------------------------------------------------------
    # Parameter counts
    # ------------------------------------------------------------------

    def trainable_parameters(self) -> int:
        """Count trainable (non-frozen) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameters(self) -> int:
        """Count frozen (requires_grad=False) parameters."""
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
