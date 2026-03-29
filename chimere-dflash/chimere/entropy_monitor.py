"""
Entropy Monitor — Confidence estimation from drafter logits.

Two modes:
  1. Lightweight (no extra params): entropy H = -sum(p * log(p)) on drafter logits
     → free signal, already available in the spec decode loop
  2. Energy Verifier (~50M params): learned scalar E(x,y) from hidden states
     → trained on accept/reject from online buffer

Integration points:
  - bulk_capture: log H per block position, correlate with acceptance
  - spec_decode: adaptive K based on running H
  - HALT probe: if H >> threshold → escalate to Claude API
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnergyVerifierConfig:
    hidden_size: int = 2048
    num_layers: int = 2
    intermediate_size: int = 1024
    dropout: float = 0.1
    # Thresholds for HALT decisions
    entropy_low: float = 1.0        # below this: high confidence
    entropy_high: float = 4.0       # above this: low confidence, consider escalation
    energy_escalate: float = 0.8    # energy score above this → escalate to Claude


def token_entropy(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Compute per-token entropy from logits.

    Args:
        logits: [B, S, V] or [S, V] raw logits
        top_k: if > 0, only consider top-k logits (faster, approximate)

    Returns:
        entropy: [B, S] or [S] in nats
    """
    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals, _ = logits.topk(top_k, dim=-1)
        log_probs = F.log_softmax(topk_vals, dim=-1)
        probs = log_probs.exp()
    else:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def block_entropy_stats(
    logits: torch.Tensor, top_k: int = 64
) -> Dict[str, float]:
    """Compute entropy statistics for a block of draft logits.

    Args:
        logits: [K, V] or [B, K, V] draft logits for one block

    Returns:
        dict with mean, max, min, std entropy across positions
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    H = token_entropy(logits, top_k=top_k)  # [B, K]
    return {
        "entropy_mean": H.mean().item(),
        "entropy_max": H.max().item(),
        "entropy_min": H.min().item(),
        "entropy_std": H.std().item(),
        "high_entropy_ratio": (H > 4.0).float().mean().item(),
    }


def adaptive_k_from_entropy(
    block_entropy: float,
    current_k: int,
    max_k: int = 15,
    min_k: int = 2,
    low_thresh: float = 1.5,
    high_thresh: float = 4.0,
) -> int:
    """Adjust block size K based on drafter entropy.

    Low entropy → drafter is confident → use large K.
    High entropy → drafter is uncertain → use small K (save verify cost).
    """
    if block_entropy < low_thresh:
        return max_k
    elif block_entropy > high_thresh:
        return min_k
    else:
        # Linear interpolation
        ratio = (block_entropy - low_thresh) / (high_thresh - low_thresh)
        return max(min_k, int(max_k - ratio * (max_k - min_k)))


class EntropyTracker:
    """Rolling tracker for entropy statistics during generation.

    Maintains a window of recent block entropies for adaptive decisions.
    """

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: List[Dict[str, float]] = []
        self.acceptance_history: List[float] = []

    def record(self, stats: Dict[str, float], acceptance_rate: float = -1.0):
        self.history.append(stats)
        if acceptance_rate >= 0:
            self.acceptance_history.append(acceptance_rate)
        if len(self.history) > self.window_size * 2:
            self.history = self.history[-self.window_size:]
            self.acceptance_history = self.acceptance_history[-self.window_size:]

    @property
    def recent_mean_entropy(self) -> float:
        if not self.history:
            return 0.0
        recent = self.history[-self.window_size:]
        return sum(s["entropy_mean"] for s in recent) / len(recent)

    @property
    def entropy_tau_correlation(self) -> Optional[float]:
        """Pearson correlation between entropy and acceptance rate.

        Negative correlation expected: high entropy → low acceptance.
        """
        if len(self.acceptance_history) < 5:
            return None

        n = min(len(self.history), len(self.acceptance_history))
        entropies = [self.history[-n + i]["entropy_mean"] for i in range(n)]
        taus = self.acceptance_history[-n:]

        mean_e = sum(entropies) / n
        mean_t = sum(taus) / n

        cov = sum((e - mean_e) * (t - mean_t) for e, t in zip(entropies, taus)) / n
        std_e = math.sqrt(sum((e - mean_e) ** 2 for e in entropies) / n)
        std_t = math.sqrt(sum((t - mean_t) ** 2 for t in taus) / n)

        if std_e < 1e-8 or std_t < 1e-8:
            return 0.0
        return cov / (std_e * std_t)

    def should_escalate(self, threshold: float = 5.0) -> bool:
        """Check if recent entropy suggests we should escalate to Claude."""
        return self.recent_mean_entropy > threshold

    def summary(self) -> Dict[str, float]:
        result = {
            "recent_mean_entropy": self.recent_mean_entropy,
            "n_blocks": len(self.history),
        }
        corr = self.entropy_tau_correlation
        if corr is not None:
            result["entropy_tau_correlation"] = corr
        return result


class EnergyVerifier(nn.Module):
    """Learned energy function E(x, y) → scalar confidence score.

    Trained on accept/reject from the online buffer:
      - Accepted blocks → low energy target
      - Rejected blocks → high energy target

    Architecture: 2-layer MLP on pooled hidden states.
    ~2M params (not 50M — keep it tiny for CPU inference).
    """

    def __init__(self, config: EnergyVerifierConfig):
        super().__init__()
        self.config = config

        self.pool_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.layers = nn.Sequential(
            nn.Linear(config.intermediate_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.energy_head = nn.Linear(config.intermediate_size, 1)

        # Initialize energy_head bias so initial predictions are neutral
        nn.init.zeros_(self.energy_head.bias)
        nn.init.normal_(self.energy_head.weight, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute energy score for a block of hidden states.

        Args:
            hidden_states: [B, K, H] drafter hidden states for a block

        Returns:
            energy: [B] scalar energy per batch element (lower = more confident)
        """
        # Mean pool across block positions
        pooled = hidden_states.mean(dim=1)              # [B, H]
        x = F.gelu(self.pool_proj(pooled))              # [B, intermediate]
        x = self.layers(x)                               # [B, intermediate]
        energy = self.energy_head(x).squeeze(-1)         # [B]
        return energy

    def training_loss(
        self,
        hidden_states: torch.Tensor,
        accepted_mask: torch.Tensor,
        margin: float = 1.0,
    ) -> torch.Tensor:
        """Contrastive energy loss.

        Accepted blocks should have lower energy than rejected blocks.

        Args:
            hidden_states: [B, K, H]
            accepted_mask: [B] boolean — True if block was accepted by target
            margin: energy margin between accepted and rejected

        Returns:
            loss: scalar
        """
        energy = self.forward(hidden_states)  # [B]

        # Binary cross-entropy: accepted → target 0, rejected → target 1
        # Normalize energy to [0, 1] with sigmoid
        targets = (~accepted_mask).float()
        loss = F.binary_cross_entropy_with_logits(energy, targets)

        return loss

    def should_escalate(self, hidden_states: torch.Tensor) -> bool:
        """Check if energy is high enough to escalate to Claude."""
        with torch.no_grad():
            energy = self.forward(hidden_states)
            return (torch.sigmoid(energy) > self.config.energy_escalate).any().item()
