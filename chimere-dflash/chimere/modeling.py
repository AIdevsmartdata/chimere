"""
DFlash Draft Model: 5-layer bidirectional transformer denoiser
with KV injection from target model features.

- Shared embedding + LM head from target (frozen)
- Bidirectional attention (no causal mask)
- Each layer receives fused target features via KV injection
- Discrete mask-predict: single-shot parallel prediction (DFlash paper)
- Anchor token at position 0 (verified), positions 1..K-1 masked → predict all at once
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DFlashConfig
from .diffusion import MaskDiffusion
from .feature_fusion import FeatureFusion


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class DFlashAttention(nn.Module):
    """Bidirectional multi-head attention with KV injection."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.num_heads = config.drafter_num_heads
        self.head_dim = config.drafter_head_dim
        self.hidden_size = config.drafter_hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, hidden_states, context_kv=None):
        B, S, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if context_kv is not None:
            k_ctx, v_ctx = context_kv
            k_ctx = k_ctx.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v_ctx = v_ctx.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([k_ctx, k], dim=2)
            v = torch.cat([v_ctx, v], dim=2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.drafter_hidden_size, config.drafter_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.drafter_hidden_size, config.drafter_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.drafter_intermediate_size, config.drafter_hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DFlashAttention(config, layer_idx)
        self.mlp = DFlashMLP(config)
        self.input_layernorm = RMSNorm(config.drafter_hidden_size, config.drafter_rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.drafter_hidden_size, config.drafter_rms_norm_eps)

    def forward(self, hidden_states, context_kv=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, context_kv=context_kv)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(nn.Module):
    """Complete DFlash block diffusion draft model with mask-predict."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.target_vocab_size, config.drafter_hidden_size)
        self.lm_head = nn.Linear(config.drafter_hidden_size, config.target_vocab_size, bias=False)
        self.diffusion = MaskDiffusion(config)
        self.feature_fusion = FeatureFusion(config)

        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, layer_idx=i) for i in range(config.drafter_num_layers)]
        )
        self.norm = RMSNorm(config.drafter_hidden_size, config.drafter_rms_norm_eps)
        self.block_pos_embed = nn.Embedding(config.block_size, config.drafter_hidden_size)
        # Mask ratio embedding: scalar ratio → hidden_size, broadcast to all positions
        self.mask_ratio_proj = nn.Sequential(
            nn.Linear(1, config.drafter_hidden_size),
            nn.SiLU(),
        )

    def freeze_shared_params(self):
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def load_target_embeddings(self, target_embed_weight, target_lm_head_weight):
        self.embed_tokens.weight.data.copy_(target_embed_weight)
        self.lm_head.weight.data.copy_(target_lm_head_weight)
        self.freeze_shared_params()

    def _forward_block(self, x_input, kv_pairs, mask_ratio=None):
        """Forward pass through the denoiser: pos embed + mask ratio embed + layers + norm + logits.

        Args:
            x_input: [B, S, H] — input embeddings (mix of real tokens and [MASK])
            kv_pairs: list of (k, v) tuples from feature fusion
            mask_ratio: float or None — fraction of positions currently masked (0.0 to 1.0).
                        If None, defaults to 1.0 (fully masked, for backward compat).
        """
        B, S, _ = x_input.shape
        device = x_input.device

        pos_ids = torch.arange(S, device=device)
        x = x_input + self.block_pos_embed(pos_ids)[None, :, :]

        # Inject mask ratio signal: tells the model how much context is available
        if mask_ratio is None:
            mask_ratio = 1.0
        ratio_input = torch.tensor([[mask_ratio]], device=device, dtype=x.dtype)
        ratio_embed = self.mask_ratio_proj(ratio_input)  # [1, H]
        x = x + ratio_embed.unsqueeze(1)  # broadcast [1, 1, H] → [B, S, H]

        for i, layer in enumerate(self.layers):
            x = layer(x, context_kv=kv_pairs[i])

        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)
        return logits

    def forward_train(self, input_ids, target_hidden_states_list, timesteps=None, noise=None):
        """
        Training forward pass with mask-predict.

        1. Embed the correct tokens
        2. Randomly mask a fraction of positions
        3. Forward through denoiser with KV injection
        4. Compute cross-entropy loss on masked positions only

        The `timesteps` and `noise` params are ignored (legacy compat).
        Returns: (loss, logits)
        """
        B, S = input_ids.shape
        device = input_ids.device

        # Embed correct tokens
        x_0 = self.embed_tokens(input_ids)

        # Sample mask ratios and apply masks
        mask_ratios = self.diffusion.sample_mask_ratio(B, device)
        x_masked, mask = self.diffusion.apply_mask(x_0, mask_ratios)

        # Fuse target hidden states for KV injection
        fused, kv_pairs = self.feature_fusion(target_hidden_states_list)

        # Forward pass with mask ratio signal
        avg_mask_ratio = mask_ratios.mean().item()
        logits = self._forward_block(x_masked, kv_pairs, mask_ratio=avg_mask_ratio)

        # Loss: cross-entropy on masked positions with exponential position decay
        # Per DFlash paper: weight_i = exp(-gamma * i) — prioritizes early positions
        # because in speculative decoding, the first rejection invalidates all later tokens
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fct(
            logits.view(-1, self.config.target_vocab_size), input_ids.view(-1)
        ).view(B, S)

        gamma = getattr(self.config, 'loss_decay_gamma', 0.1)
        position_weights = torch.exp(
            -gamma * torch.arange(S, device=device, dtype=torch.float)
        )  # [S] — decays from 1.0 at pos 0

        # Only compute loss on masked positions, weighted by position
        masked_loss = loss_per_token * mask.float() * position_weights[None, :]
        # Normalize by weighted mask count for proper gradient scaling
        weighted_mask_count = (mask.float() * position_weights[None, :]).sum()
        loss = masked_loss.sum() / weighted_mask_count.clamp(min=1.0)

        return loss, logits

    @torch.no_grad()
    def generate_block(self, target_hidden_states_list, temperature=0.0, n_steps=1,
                       anchor_token_id=None):
        """
        Generate a draft block.

        Default mode (n_steps=1): single-shot parallel prediction per DFlash paper.
        The drafter receives anchor token (position 0, verified) + masked positions
        and predicts all masked tokens in ONE forward pass.

        Multi-step mode (n_steps>1): iterative confidence-based unmasking.
        Kept for experimentation but not the DFlash default.

        Args:
            target_hidden_states_list: list of k tensors [B, S, H]
            temperature: sampling temperature (0 = greedy)
            n_steps: number of unmasking steps (1 = single-shot DFlash default)
            anchor_token_id: int or None — token ID for position 0 (anchor).
                             If None, all positions are masked.
        """
        B = target_hidden_states_list[0].shape[0]
        device = target_hidden_states_list[0].device
        S = self.config.block_size
        H = self.config.drafter_hidden_size

        # Precompute KV injection (same for all steps)
        fused, kv_pairs = self.feature_fusion(target_hidden_states_list)

        # Build anchor embedding if provided
        anchor_embeds = None
        if anchor_token_id is not None:
            anchor_ids = torch.tensor([anchor_token_id], device=device).expand(B)
            anchor_embeds = self.embed_tokens(anchor_ids)  # [B, H]

        # Start with positions masked (position 0 may be anchor)
        x_t, mask = self.diffusion.apply_full_mask(
            B, S, H, device, dtype=torch.float32, anchor_embeds=anchor_embeds
        )

        if n_steps == 1:
            # === SINGLE-SHOT (DFlash default) ===
            # One forward pass predicts all masked tokens simultaneously
            current_mask_ratio = mask.float().mean().item()
            logits = self._forward_block(x_t, kv_pairs, mask_ratio=current_mask_ratio)

            if temperature == 0.0:
                draft_ids = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                draft_ids = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, S)

            # Keep anchor token at position 0 if provided
            if anchor_token_id is not None:
                draft_ids[:, 0] = anchor_token_id

            return draft_ids, logits

        # === MULTI-STEP (iterative, for experimentation) ===
        positions_per_step = max(1, mask.sum(dim=-1).min().item() // n_steps)

        final_logits = None
        predicted_ids = torch.zeros(B, S, dtype=torch.long, device=device)
        if anchor_token_id is not None:
            predicted_ids[:, 0] = anchor_token_id

        for step_idx in range(n_steps):
            current_mask_ratio = mask.float().mean().item()
            logits = self._forward_block(x_t, kv_pairs, mask_ratio=current_mask_ratio)
            final_logits = logits

            remaining_masked = mask.sum(dim=-1).min().item()
            if remaining_masked == 0:
                break

            if step_idx == n_steps - 1:
                n_unmask = int(remaining_masked)
            else:
                n_unmask = min(positions_per_step, int(remaining_masked))

            confidence = logits.max(dim=-1).values
            confidence = confidence.masked_fill(~mask, float('-inf'))
            _, top_indices = confidence.topk(n_unmask, dim=-1)

            if temperature == 0.0:
                step_preds = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                step_preds = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, S)

            clean_embeds = self.embed_tokens(step_preds).to(x_t.dtype)
            for b in range(B):
                idx = top_indices[b]
                x_t[b, idx] = clean_embeds[b, idx]
                mask[b, idx] = False
                predicted_ids[b, idx] = step_preds[b, idx]

        if mask.any():
            final_logits = self._forward_block(x_t, kv_pairs, mask_ratio=0.0)

        if temperature == 0.0:
            draft_ids = final_logits.argmax(dim=-1)
        else:
            probs = F.softmax(final_logits / temperature, dim=-1)
            draft_ids = torch.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(B, S)

        if anchor_token_id is not None:
            draft_ids[:, 0] = anchor_token_id

        return draft_ids, final_logits
