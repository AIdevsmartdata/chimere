"""
Engram — Conditional Memory Module with O(1) Lookup.

N-gram hash tables in DRAM for factual recall, freeing MoE experts
for reasoning. Inspired by DeepSeek's memory-augmented architectures.

Architecture:
  1. NgramHasher: maps token n-grams to table indices via k independent hash functions
  2. EngamTable: torch.nn.Embedding tables in CPU RAM (no GPU needed)
  3. ContextGating: scores relevance of each retrieved engram against current hidden state
  4. EngamModule: full pipeline — hash → lookup → gate → fuse

Cost: ~8 GB RAM for 1M entries x 2048 dims (FP32), 0 GPU.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EngramConfig:
    hidden_size: int = 2048
    table_size: int = 1 << 20       # 1M entries
    num_tables: int = 8             # independent hash functions
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3])
    vocab_size: int = 248320
    gate_bias: float = -1.0         # initial bias → conservative gating (default off)
    dropout: float = 0.0
    device: str = "cpu"             # tables live in RAM


class NgramHasher(nn.Module):
    """Maps token n-gram windows to table indices using learned hash projections.

    For each position i, builds n-grams of sizes [2, 3] from tokens[i-n+1:i+1],
    then hashes each n-gram with k independent hash functions → k indices per n-gram.
    """

    def __init__(self, config: EngramConfig):
        super().__init__()
        self.num_tables = config.num_tables
        self.table_size = config.table_size
        self.ngram_sizes = config.ngram_sizes
        self.vocab_size = config.vocab_size

        # Learnable hash coefficients per table per n-gram size
        # hash_k(ngram) = (sum_j coeff[k,j] * token[j]) mod table_size
        max_n = max(config.ngram_sizes)
        # Random prime coefficients (frozen) for hashing — not learnable
        coeffs = torch.randint(1, config.table_size, (config.num_tables, max_n))
        self.register_buffer("coeffs", coeffs)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Hash token n-grams to table indices.

        Args:
            token_ids: [B, seq_len] token IDs

        Returns:
            indices: [B, seq_len, num_tables * len(ngram_sizes)] table indices
        """
        B, S = token_ids.shape
        device = token_ids.device
        all_indices = []

        for n in self.ngram_sizes:
            # Pad left with zeros for positions without full n-gram context
            padded = F.pad(token_ids, (n - 1, 0), value=0)  # [B, S + n - 1]

            for k in range(self.num_tables):
                # Weighted sum of n-gram tokens
                h = torch.zeros(B, S, device=device, dtype=torch.long)
                for j in range(n):
                    h = h + self.coeffs[k, j].item() * padded[:, j:j + S]
                h = h % self.table_size
                all_indices.append(h)

        # [B, S, num_tables * len(ngram_sizes)]
        return torch.stack(all_indices, dim=-1)


class EngramTable(nn.Module):
    """Embedding table storing engram vectors in CPU RAM."""

    def __init__(self, config: EngramConfig):
        super().__init__()
        self.num_lookups = config.num_tables * len(config.ngram_sizes)
        # All tables share one large Embedding (partitioned by index offset)
        total_entries = config.table_size * self.num_lookups
        self.table = nn.Embedding(total_entries, config.hidden_size)
        self.table_size = config.table_size

        # Initialize small — engrams start near zero, gating decides relevance
        nn.init.normal_(self.table.weight, std=0.01)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Lookup engram vectors from indices.

        Args:
            indices: [B, S, num_lookups] — indices from NgramHasher

        Returns:
            engrams: [B, S, num_lookups, hidden_size]
        """
        B, S, K = indices.shape
        # Offset each lookup into its partition
        offsets = torch.arange(K, device=indices.device) * self.table_size
        offset_indices = indices + offsets.unsqueeze(0).unsqueeze(0)
        return self.table(offset_indices)


class ContextGating(nn.Module):
    """Scores relevance of retrieved engrams against current hidden state.

    alpha_i = sigmoid(RMSNorm(h)^T . RMSNorm(e_i) / sqrt(d) + bias)

    Low alpha → engram not relevant → ignored.
    """

    def __init__(self, config: EngramConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm_h = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.norm_e = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.bias = nn.Parameter(torch.tensor(config.gate_bias))
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(config.hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, engrams: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute gated engram contributions.

        Args:
            hidden_states: [B, S, H] current hidden states
            engrams: [B, S, K, H] retrieved engram vectors

        Returns:
            contribution: [B, S, H] weighted sum of relevant engrams
            alphas: [B, S, K] gating scores for analysis
        """
        h_norm = self.norm_h(hidden_states)        # [B, S, H]
        e_norm = self.norm_e(engrams)               # [B, S, K, H]

        # Dot product attention score
        # h_norm: [B, S, 1, H] . e_norm: [B, S, K, H] → [B, S, K]
        scores = torch.einsum("bsh,bskh->bsk", h_norm, e_norm) * self.scale
        alphas = torch.sigmoid(scores + self.bias)  # [B, S, K]

        # Weighted sum
        contribution = torch.einsum("bsk,bskh->bsh", alphas, engrams)
        contribution = self.dropout(contribution)

        return contribution, alphas


class EngramModule(nn.Module):
    """Full Engram pipeline: hash → lookup → gate → fuse.

    Plugs into any transformer layer as:
        h = h + engram_module(token_ids, h)
    """

    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        self.hasher = NgramHasher(config)
        self.table = EngramTable(config)
        self.gating = ContextGating(config)

        # Move tables to configured device (CPU by default)
        self.table = self.table.to(config.device)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """Retrieve and fuse engrams into hidden states.

        Args:
            token_ids: [B, S] input token IDs
            hidden_states: [B, S, H] current hidden states
            return_diagnostics: if True, also return gating stats

        Returns:
            enriched: [B, S, H] = hidden_states + gated engram contribution
            diagnostics: dict with gating stats (only if return_diagnostics=True)
        """
        compute_device = hidden_states.device

        # 1. Hash n-grams → indices (on compute device)
        indices = self.hasher(token_ids)                        # [B, S, K]

        # 2. Lookup engrams (on table device, typically CPU)
        indices_cpu = indices.to(self.table.table.weight.device)
        engrams = self.table(indices_cpu)                       # [B, S, K, H]
        engrams = engrams.to(compute_device)                    # move to GPU if needed

        # 3. Gate and fuse (move gating to compute device if needed)
        if next(self.gating.parameters()).device != compute_device:
            self.gating = self.gating.to(compute_device)
        contribution, alphas = self.gating(hidden_states, engrams)

        enriched = hidden_states + contribution

        if return_diagnostics:
            with torch.no_grad():
                diagnostics = {
                    "mean_alpha": alphas.mean().item(),
                    "active_ratio": (alphas > 0.5).float().mean().item(),
                    "max_alpha": alphas.max().item(),
                    "contribution_norm": contribution.norm(dim=-1).mean().item(),
                    "hidden_norm": hidden_states.norm(dim=-1).mean().item(),
                }
            return enriched, diagnostics

        return enriched

    def update_engrams(
        self,
        token_ids: torch.Tensor,
        target_values: torch.Tensor,
        lr: float = 0.01,
    ):
        """Online update: move engram vectors toward target hidden states.

        For positions where the model got the next token right,
        strengthen the engram. Simple Hebbian-style update.

        Args:
            token_ids: [B, S] tokens that produced correct predictions
            target_values: [B, S, H] target hidden states to store
        """
        with torch.no_grad():
            indices = self.hasher(token_ids)
            indices_cpu = indices.to(self.table.table.weight.device)
            target_cpu = target_values.to(self.table.table.weight.device)

            B, S, K = indices_cpu.shape
            offsets = torch.arange(K, device=indices_cpu.device) * self.table.table_size
            flat_indices = (indices_cpu + offsets.unsqueeze(0).unsqueeze(0)).reshape(-1)

            # Average target across lookups (each lookup should store the same info)
            target_expanded = target_cpu.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, target_cpu.shape[-1])

            # EMA update: engram = (1 - lr) * engram + lr * target
            current = self.table.table.weight[flat_indices]
            updated = (1 - lr) * current + lr * target_expanded
            self.table.table.weight[flat_indices] = updated
