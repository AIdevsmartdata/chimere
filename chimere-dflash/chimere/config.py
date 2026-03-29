"""DFlash drafter configuration."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DFlashConfig:
    """Configuration for Chimère-DFlash drafter model."""

    # Target model specs
    target_hidden_size: int = 2048
    target_num_layers: int = 40
    target_vocab_size: int = 248320
    target_layer_ids: List[int] = field(
        default_factory=lambda: [2, 11, 20, 29, 37]
    )

    # Drafter architecture
    drafter_hidden_size: int = 2048
    drafter_num_layers: int = 5
    drafter_num_heads: int = 16
    drafter_head_dim: int = 128
    drafter_intermediate_size: int = 5504
    drafter_rms_norm_eps: float = 1e-6

    # Fusion
    num_feature_layers: int = 5
    fusion_dim: int = 2048

    # Diffusion (mask-predict)
    block_size: int = 16
    num_train_timesteps: int = 1000  # kept for backward compat, not used in mask-predict
    mask_schedule: str = "cosine"  # "uniform" or "cosine" mask ratio sampling
    anchor_first_token: bool = True  # keep position 0 unmasked (anchor) per DFlash paper
    loss_decay_gamma: float = 0.1  # exponential loss decay: weight_i = exp(-gamma * i)
    # Legacy Gaussian params (unused in mask-predict, kept for checkpoint compat)
    diffusion_steps: int = 1
    noise_schedule: str = "linear"
    beta_start: float = 0.0001
    beta_end: float = 0.02

    # Training
    max_seq_length: int = 2048
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
