"""DFlash v6 configuration — aligned with z-lab paper (arXiv 2602.06036)."""
from dataclasses import dataclass, field
from typing import List


def build_target_layer_ids(num_target_layers: int, num_feature_layers: int = 5) -> List[int]:
    """Select target layers uniformly between layer 1 and (num_layers - 3).

    Matches z-lab's utils.py:build_target_layer_ids().
    """
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_feature_layers - 1)))
        for i in range(num_feature_layers)
    ]


@dataclass
class DFlashV6Config:
    """Configuration for DFlash v6 drafter (z-lab aligned).

    For Qwen3.5-35B-A3B (40 layers, hidden=2048, MoE):
    - 8 drafter layers (z-lab uses 8 for MoE targets)
    - 32 attention heads, 4 KV heads (GQA)
    - Same hidden_size as target (2048) for shared embed/lm_head
    - 5 target feature layers, uniformly spaced
    """

    # Target model specs (Qwen3.5-35B-A3B)
    target_hidden_size: int = 2048
    target_num_layers: int = 40
    target_vocab_size: int = 248320
    num_feature_layers: int = 5
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 10, 19, 28, 37])

    # Drafter architecture (matches z-lab Qwen3-Coder-30B-A3B-DFlash config)
    hidden_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128  # hidden_size * num_attention_heads might differ, but head_dim=128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.05

    # Block diffusion
    block_size: int = 16
    mask_token_id: int = 248077  # <|MASK|> added via add_special_tokens (z-lab convention)

    # Streak distillation loss
    loss_decay_gamma: float = 7.0  # w_k = exp(-(k-1)/gamma), paper uses ~block_size/2

    # Training hyperparameters (z-lab recipe)
    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.04
    num_epochs: int = 6
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    max_seq_length: int = 4096

    # Data
    max_ctx_len: int = 1024  # max context hidden states to keep
    blocks_per_seq: int = 20  # virtual items per sequence per epoch
