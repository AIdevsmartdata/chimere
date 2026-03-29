"""DFlash v7 configuration — v6 + RoPE positional encoding."""
from dataclasses import dataclass, field
from typing import List

from .config_v6 import build_target_layer_ids


@dataclass
class DFlashV7Config:
    """Configuration for DFlash v7 drafter (v6 + RoPE).

    Only changes from v6:
    - rope_theta: 10_000_000.0 (matches Qwen3.5-35B-A3B and z-lab)
    - max_position_embeddings: 262144 (RoPE buffer size)
    """

    # Target model specs (Qwen3.5-35B-A3B)
    target_hidden_size: int = 2048
    target_num_layers: int = 40
    target_vocab_size: int = 248320
    num_feature_layers: int = 5
    target_layer_ids: List[int] = field(default_factory=lambda: [1, 10, 19, 28, 37])

    # Drafter architecture
    hidden_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.05
    use_output_proj: bool = False  # project hidden_size -> target_hidden_size before lm_head

    # RoPE (NEW in v7)
    rope_theta: float = 10_000_000.0       # identical to Qwen3.5-35B-A3B and z-lab
    max_position_embeddings: int = 262144   # RoPE buffer size

    # Block diffusion
    block_size: int = 16
    mask_token_id: int = 248077

    # Streak distillation loss
    loss_decay_gamma: float = 7.0

    # LK loss parameters (arXiv 2602.23881)
    lk_eta: float = 3.0          # adaptive schedule decay rate
    lk_label_smoothing: float = 0.0  # label smoothing epsilon (0 = pure LK)

    # Training hyperparameters
    learning_rate: float = 6e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.04
    num_epochs: int = 6
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    max_seq_length: int = 4096

    # Data
    max_ctx_len: int = 1024
    blocks_per_seq: int = 20

    # Expert prefetch head (MoE routing prediction)
    predict_expert_routing: bool = False
    routing_layers: List[int] = field(default_factory=lambda: list(range(20, 40)))
    num_target_experts: int = 256
    num_active_experts: int = 8
    routing_proj_rank: int = 512
    routing_adapter_rank: int = 64
    routing_loss_weight: float = 0.1
    gate_weights_path: str = "data/qwen_gate_weights_20_39.pt"
