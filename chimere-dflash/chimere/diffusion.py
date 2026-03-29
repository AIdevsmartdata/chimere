"""
Discrete absorbing diffusion for block token prediction (mask-predict).

Training: randomly mask a fraction of token positions → model predicts masked tokens
Inference: start with all positions masked → iteratively unmask by confidence

This is the approach used by DFlash, BD3-LMs, MDLM, and Fast-dLLM.
No noise schedule, no q_sample, no DDPM reverse — just mask/predict/unmask.
"""
import math

import torch
import torch.nn as nn


class MaskDiffusion(nn.Module):
    """
    Discrete absorbing diffusion: mask tokens → predict → unmask by confidence.

    At training time:
    - Sample a mask ratio r ~ Beta(1, 3) or Uniform(0.1, 1.0)
    - Mask r fraction of positions (replace token embeddings with learned [MASK] embedding)
    - Model predicts the original tokens at masked positions

    At inference time:
    - Start with all positions masked
    - Iteratively: forward pass → select most confident predictions → unmask those → repeat
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.mask_schedule = getattr(config, 'mask_schedule', 'cosine')
        self.anchor_first_token = getattr(config, 'anchor_first_token', True)
        # Learned [MASK] embedding (same dim as drafter hidden)
        self.mask_embedding = nn.Parameter(torch.randn(config.drafter_hidden_size) * 0.02)
        # Legacy compat: keep num_timesteps for old checkpoints
        self.num_timesteps = getattr(config, 'num_train_timesteps', 1000)

    def sample_mask_ratio(self, batch_size, device):
        """Sample mask ratios for a batch.

        Cosine schedule: biases towards higher mask ratios (harder tasks),
        which forces the model to rely on context (hidden states) rather
        than just copying from unmasked positions.
        """
        if self.mask_schedule == 'cosine':
            # Sample u ~ Uniform(0, 1), then r = cos(u * pi/2)
            # This gives more weight to high mask ratios
            u = torch.rand(batch_size, device=device)
            ratios = torch.cos(u * math.pi / 2)
            # Clamp to avoid fully unmasked (too easy) or empty mask (no gradient)
            ratios = torch.clamp(ratios, 0.1, 1.0)
        else:
            # Uniform between 0.1 and 1.0
            ratios = torch.rand(batch_size, device=device) * 0.9 + 0.1
        return ratios

    def apply_mask(self, x_0, mask_ratios):
        """Apply random masks to token embeddings.

        If anchor_first_token is True, position 0 is never masked (anchor token),
        matching the DFlash inference protocol where each block starts with a
        verified token from the previous block/prompt.

        Args:
            x_0: [B, S, H] — clean token embeddings
            mask_ratios: [B] — fraction of positions to mask per sample

        Returns:
            x_masked: [B, S, H] — embeddings with masked positions replaced
            mask: [B, S] bool — True at masked positions
        """
        B, S, H = x_0.shape
        device = x_0.device

        # Positions eligible for masking
        if self.anchor_first_token:
            # Position 0 is always unmasked (anchor)
            maskable = S - 1
            offset = 1
        else:
            maskable = S
            offset = 0

        # Generate mask: for each sample, mask `ratio * maskable` random positions
        mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        for b in range(B):
            n_mask = max(1, int(mask_ratios[b].item() * maskable))
            # Only draw from positions [offset, S)
            indices = torch.randperm(maskable, device=device)[:n_mask] + offset
            mask[b, indices] = True

        # Replace masked positions with the learned [MASK] embedding
        x_masked = x_0.clone()
        x_masked[mask] = self.mask_embedding.to(x_0.dtype)

        return x_masked, mask

    def apply_full_mask(self, batch_size, seq_len, hidden_size, device, dtype,
                        anchor_embeds=None):
        """Create masked input for inference start.

        If anchor_first_token and anchor_embeds are provided, position 0 uses
        the anchor embedding (verified token from previous block/prompt).
        Otherwise all positions are masked.

        Args:
            anchor_embeds: [B, H] — embedding of the anchor token (optional)

        Returns:
            x_masked: [B, S, H] — masked input
            mask: [B, S] bool — True at masked positions
        """
        x_masked = self.mask_embedding.to(dtype).unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, hidden_size
        ).clone()
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

        # If anchor provided, set position 0 to anchor embedding (unmasked)
        if self.anchor_first_token and anchor_embeds is not None:
            x_masked[:, 0, :] = anchor_embeds.to(dtype)
            mask[:, 0] = False

        return x_masked, mask


# Legacy compat: keep GaussianDiffusion importable for old checkpoints
class GaussianDiffusion(nn.Module):
    """Legacy Gaussian diffusion — kept for checkpoint loading compatibility."""

    def __init__(self, config):
        super().__init__()
        self.num_timesteps = config.num_train_timesteps
        self.block_size = config.block_size
        betas = torch.linspace(config.beta_start, config.beta_end, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise, noise
