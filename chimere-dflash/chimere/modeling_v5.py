"""
DFlash v5 — Drafter with full KV injection from target context.

Key difference from v3/v4: block tokens cross-attend to the FULL target context
(all hidden states from the verified prefix) at every drafter layer, not just
a fixed-size snapshot. This matches the real DFlash paper architecture.

Architecture:
  Target hidden states [0..P] from 5 layers
       ↓ concat + projection
  context [B, ctx_len, d_model]
       ↓
  ┌─ Layer 1 ──────────────────┐
  │  Q from block tokens        │
  │  K = [ctx_K ; self_K]       │  ← KV injection
  │  V = [ctx_V ; self_V]       │
  └────────────────────────────┘
       ↓  ... × 5 layers
  logits [B, K, vocab_size]
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DFlashConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class DFlashAttentionV5(nn.Module):
    """Bidirectional attention with cross-attention KV injection from target context."""

    def __init__(self, d_model, n_heads, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Self-attention projections (for block tokens)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)

        # KV injection projections (for target context)
        self.ctx_k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.ctx_v_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)

        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, x, context, attn_mask=None):
        """
        x: [B, K, d] - block representations
        context: [B, ctx_len, d] - projected target features (padded)
        attn_mask: [B, 1, K, ctx_len+K] float mask (0=attend, -inf=ignore)
        """
        B, K, _ = x.shape
        ctx_len = context.shape[1]

        # Queries from block only
        Q = self.q_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        # [B, n_heads, K, head_dim]

        # Keys/Values from block (self)
        K_self = self.k_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        V_self = self.v_proj(x).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        # Keys/Values from target context (cross)
        K_ctx = self.ctx_k_proj(context).view(B, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)
        V_ctx = self.ctx_v_proj(context).view(B, ctx_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Concat: [context_KV ; block_self_KV] along sequence dimension
        K_full = torch.cat([K_ctx, K_self], dim=2)  # [B, n_heads, ctx_len+K, head_dim]
        V_full = torch.cat([V_ctx, V_self], dim=2)

        # Apply QK normalization
        Q = self.q_norm(Q)
        K_full = self.k_norm(K_full)

        # Attention: block queries attend to full context + all block positions (bidirectional)
        attn_out = F.scaled_dot_product_attention(
            Q, K_full, V_full, attn_mask=attn_mask, dropout_p=0.0
        )  # [B, n_heads, K, head_dim]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, -1)
        return self.out_proj(attn_out)


class DFlashMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, d_model, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashLayerV5(nn.Module):
    def __init__(self, d_model, n_heads, head_dim, intermediate_size, eps=1e-6):
        super().__init__()
        self.attn = DFlashAttentionV5(d_model, n_heads, head_dim)
        self.mlp = DFlashMLP(d_model, intermediate_size)
        self.norm1 = RMSNorm(d_model, eps)
        self.norm2 = RMSNorm(d_model, eps)

    def forward(self, x, context, attn_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, context, attn_mask=attn_mask)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


def build_attention_mask(block_size, context_lengths, max_ctx_len, device):
    """Build attention mask for block queries attending to [context ; block].

    Block tokens use BIDIRECTIONAL attention within the block (mask-predict)
    and FULL attention to all non-padded context positions.

    Returns: [B, 1, K, ctx_len+K] float mask (0=attend, -inf=ignore)
    """
    B = context_lengths.shape[0]
    total_kv_len = max_ctx_len + block_size

    # Start: allow everything
    mask = torch.zeros(B, 1, block_size, total_kv_len, device=device)

    # Mask out padded context positions
    # positions [0, max_ctx_len) — only attend where pos < context_length
    positions = torch.arange(max_ctx_len, device=device).unsqueeze(0)  # [1, max_ctx_len]
    ctx_valid = positions < context_lengths.unsqueeze(1)  # [B, max_ctx_len]
    # Expand to [B, 1, K, max_ctx_len] and set invalid to -inf
    ctx_invalid = ~ctx_valid.unsqueeze(1).unsqueeze(1).expand(-1, 1, block_size, -1)
    mask[:, :, :, :max_ctx_len].masked_fill_(ctx_invalid, float('-inf'))

    return mask


class DFlashDraftModelV5(nn.Module):
    """DFlash v5: block diffusion drafter with full context KV injection."""

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        d = config.drafter_hidden_size

        # Shared with target (frozen)
        self.embed_tokens = nn.Embedding(config.target_vocab_size, d)
        self.lm_head = nn.Linear(d, config.target_vocab_size, bias=False)

        # Mask token (learned)
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)

        # Context fusion: learned weighted average of target layers (EAGLE-3 style)
        # Much cheaper than concat+project (5 params vs 21M), and α tells us
        # which layers are most informative for next-token prediction.
        self.layer_weights = nn.Parameter(torch.ones(config.num_feature_layers))
        # Optional projection if target_hidden_size != drafter_hidden_size
        if config.target_hidden_size != d:
            self.context_proj = nn.Linear(config.target_hidden_size, d, bias=False)
        else:
            self.context_proj = nn.Identity()

        # Block position embeddings (only for block tokens, not context)
        self.block_pos_embed = nn.Embedding(config.block_size, d)

        # Mask ratio embedding
        self.mask_ratio_proj = nn.Sequential(
            nn.Linear(1, d),
            nn.SiLU(),
        )

        # Drafter layers with KV injection
        self.layers = nn.ModuleList([
            DFlashLayerV5(
                d, config.drafter_num_heads, config.drafter_head_dim,
                config.drafter_intermediate_size, config.drafter_rms_norm_eps
            )
            for _ in range(config.drafter_num_layers)
        ])
        self.norm = RMSNorm(d, config.drafter_rms_norm_eps)

    def freeze_shared_params(self):
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        for param in self.lm_head.parameters():
            param.requires_grad = False

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save activation memory."""
        self._gradient_checkpointing = True

    @property
    def gradient_checkpointing(self):
        return getattr(self, '_gradient_checkpointing', False)

    def _fuse_context(self, context_hidden_list):
        """Fuse target hidden states from multiple layers via learned weighted average.

        Args:
            context_hidden_list: list of n_layers tensors [B, ctx_len, target_hidden_size]

        Returns:
            [B, ctx_len, d_model]
        """
        # Learned weighted average: context = Σ αᵢ × layerᵢ
        alpha = F.softmax(self.layer_weights, dim=0)  # [n_layers]
        # Stack → [n_layers, B, ctx_len, H], weighted sum → [B, ctx_len, H]
        stacked = torch.stack(context_hidden_list, dim=0)
        fused = (alpha[:, None, None, None] * stacked).sum(dim=0)
        return self.context_proj(fused)

    def _apply_mask(self, x_0, B, K, device):
        """Apply full mask for training (all positions masked except anchor).

        At inference, ALL draft positions are masked (single-step denoising).
        Training must match this exactly — partial masking leaks information
        through bidirectional attention that isn't available at inference time.

        Returns: (x_masked, mask, mask_ratio)
        """
        # Full mask: all positions masked
        mask = torch.ones(B, K, dtype=torch.bool, device=device)

        # Anchor: position 0 never masked
        if self.config.anchor_first_token:
            mask[:, 0] = False

        # Replace masked positions with [MASK] embedding
        x_masked = x_0.clone()
        x_masked[mask] = self.mask_token.to(x_0.dtype)

        avg_ratio = mask.float().mean().item()
        return x_masked, mask, avg_ratio

    def forward_train(self, block_input_ids, context_hidden_list, context_lengths):
        """
        Training forward pass with full context KV injection.

        Args:
            block_input_ids: [B, K] — target token IDs for the block
            context_hidden_list: list of n_layers tensors [B, max_ctx_len, H] (padded)
            context_lengths: [B] — actual context lengths (for attention mask)

        Returns: (loss, logits)
        """
        B, K = block_input_ids.shape
        device = block_input_ids.device

        # 1. Fuse context hidden states
        ctx = self._fuse_context(context_hidden_list)  # [B, max_ctx_len, d]

        # 2. Embed block tokens and apply mask
        x_0 = self.embed_tokens(block_input_ids)  # [B, K, d]
        x_masked, mask, avg_mask_ratio = self._apply_mask(x_0, B, K, device)

        # 3. Add block position embeddings
        pos_ids = torch.arange(K, device=device)
        x = x_masked + self.block_pos_embed(pos_ids)[None, :, :]

        # 4. Add mask ratio signal
        ratio_input = torch.tensor([[avg_mask_ratio]], device=device, dtype=x.dtype)
        ratio_embed = self.mask_ratio_proj(ratio_input)  # [1, d]
        x = x + ratio_embed.unsqueeze(1)  # broadcast to [B, K, d]

        # 5. Build attention mask for padded context
        max_ctx_len = ctx.shape[1]
        attn_mask = build_attention_mask(K, context_lengths, max_ctx_len, device)

        # 6. Forward through layers with KV injection
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, ctx, attn_mask, use_reentrant=False
                )
            else:
                x = layer(x, context=ctx, attn_mask=attn_mask)

        # 7. Residual: add last layer-37 hidden state (last verified position)
        # This anchors the output in the LM head's expected manifold.
        # h37_last is the hidden state at the last context position (anchor).
        # context_hidden_list[-1] is layer 37, shape [B, max_ctx_len, H]
        h37 = context_hidden_list[-1]  # [B, max_ctx_len, H]
        # Gather last valid position per batch element
        last_idx = (context_lengths - 1).clamp(min=0)  # [B]
        h37_last = h37[torch.arange(B, device=device), last_idx]  # [B, H]
        x = x + h37_last.unsqueeze(1).to(x.dtype)  # broadcast to [B, K, d]

        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)  # [B, K, vocab_size]

        # 7. Loss on masked positions with exponential position decay
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fct(
            logits.view(-1, self.config.target_vocab_size),
            block_input_ids.view(-1)
        ).view(B, K)

        gamma = self.config.loss_decay_gamma
        position_weights = torch.exp(
            -gamma * torch.arange(K, device=device, dtype=torch.float)
        )

        masked_loss = loss_per_token * mask.float() * position_weights[None, :]
        weighted_mask_count = (mask.float() * position_weights[None, :]).sum()
        loss = masked_loss.sum() / weighted_mask_count.clamp(min=1.0)

        return loss, logits

    @torch.no_grad()
    def generate_block(self, context_hidden_list, context_lengths=None,
                       temperature=0.0, anchor_token_id=None):
        """
        Generate a draft block with full context KV injection.

        Args:
            context_hidden_list: list of n_layers tensors [B, ctx_len, H]
            context_lengths: [B] or None (if None, assumes no padding)
            temperature: sampling temperature (0 = greedy)
            anchor_token_id: int or None — verified token for position 0

        Returns: (draft_ids [B, K], logits [B, K, V])
        """
        B = context_hidden_list[0].shape[0]
        device = context_hidden_list[0].device
        K = self.config.block_size
        d = self.config.drafter_hidden_size

        # 1. Fuse context
        ctx = self._fuse_context(context_hidden_list)  # [B, ctx_len, d]
        ctx_len = ctx.shape[1]

        if context_lengths is None:
            context_lengths = torch.full((B,), ctx_len, device=device, dtype=torch.long)

        # 2. Build masked block input
        x = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
        mask = torch.ones(B, K, dtype=torch.bool, device=device)

        if anchor_token_id is not None:
            anchor_ids = torch.tensor([anchor_token_id], device=device).expand(B)
            x[:, 0] = self.embed_tokens(anchor_ids)
            mask[:, 0] = False

        # 3. Position embeddings + mask ratio
        pos_ids = torch.arange(K, device=device)
        x = x + self.block_pos_embed(pos_ids)[None, :, :]

        mask_ratio = mask.float().mean().item()
        ratio_input = torch.tensor([[mask_ratio]], device=device, dtype=x.dtype)
        ratio_embed = self.mask_ratio_proj(ratio_input)
        x = x + ratio_embed.unsqueeze(1)

        # 4. Attention mask
        attn_mask = build_attention_mask(K, context_lengths, ctx_len, device)

        # 5. Forward through layers
        for layer in self.layers:
            x = layer(x, context=ctx, attn_mask=attn_mask)

        # 6. Residual: add last layer-37 hidden state
        h37 = context_hidden_list[-1]  # [B, ctx_len, H]
        last_idx = (context_lengths - 1).clamp(min=0)
        h37_last = h37[torch.arange(B, device=device), last_idx]  # [B, H]
        x = x + h37_last.unsqueeze(1).to(x.dtype)

        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)

        # 7. Decode
        if temperature == 0.0:
            draft_ids = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            draft_ids = torch.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(B, K)

        if anchor_token_id is not None:
            draft_ids[:, 0] = anchor_token_id

        return draft_ids, logits

    def generate_block_multistep(self, context_hidden_list, context_lengths=None,
                                  temperature=0.0, anchor_token_id=None, n_steps=4):
        """
        Multi-step denoising for draft block generation.

        At each step, reveal the most confident positions (lowest entropy)
        and re-denoise the remaining masked positions. This matches the
        cosine schedule masking used during training.

        Args:
            context_hidden_list: list of n_layers tensors [B, ctx_len, H]
            context_lengths: [B] or None
            temperature: sampling temperature (0 = greedy)
            anchor_token_id: int or None
            n_steps: number of denoising steps (1 = single-step, same as generate_block)

        Returns: (draft_ids [B, K], logits [B, K, V])
        """
        B = context_hidden_list[0].shape[0]
        device = context_hidden_list[0].device
        K = self.config.block_size
        d = self.config.drafter_hidden_size

        # Fuse context once (shared across steps)
        ctx = self._fuse_context(context_hidden_list)
        ctx_len = ctx.shape[1]

        if context_lengths is None:
            context_lengths = torch.full((B,), ctx_len, device=device, dtype=torch.long)

        attn_mask = build_attention_mask(K, context_lengths, ctx_len, device)
        pos_embed = self.block_pos_embed(torch.arange(K, device=device))[None, :, :]

        # Initialize: all positions masked (except anchor)
        mask = torch.ones(B, K, dtype=torch.bool, device=device)
        draft_ids = torch.zeros(B, K, dtype=torch.long, device=device)

        if anchor_token_id is not None:
            mask[:, 0] = False
            draft_ids[:, 0] = anchor_token_id

        # How many positions to reveal per step (linear schedule)
        n_masked = mask.sum(dim=1)[0].item()  # K-1 if anchor
        positions_per_step = max(1, n_masked // n_steps)

        final_logits = None

        for step in range(n_steps):
            # Build input: revealed positions get real embeddings, masked get [MASK]
            x = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
            revealed = ~mask
            if revealed.any():
                x[revealed] = self.embed_tokens(draft_ids[revealed]).to(x.dtype)

            # Position embeddings
            x = x + pos_embed

            # Mask ratio signal (decreases each step)
            current_mask_ratio = mask.float().mean().item()
            ratio_input = torch.tensor([[current_mask_ratio]], device=device, dtype=x.dtype)
            ratio_embed = self.mask_ratio_proj(ratio_input)
            x = x + ratio_embed.unsqueeze(1)

            # Forward
            for layer in self.layers:
                x = layer(x, context=ctx, attn_mask=attn_mask)

            x = self.norm(x)
            x = x.to(self.lm_head.weight.dtype)
            logits = self.lm_head(x)
            final_logits = logits

            # Decode predictions for masked positions
            if temperature == 0.0:
                pred_ids = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                pred_ids = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), 1
                ).view(B, K)

            # Update draft_ids at masked positions
            draft_ids[mask] = pred_ids[mask]

            # Last step: don't need to reveal — all done
            if step == n_steps - 1:
                break

            # Select positions to reveal: lowest entropy among still-masked positions
            # Use negative log-prob of the predicted token as confidence proxy
            log_probs = F.log_softmax(logits, dim=-1)  # [B, K, V]
            # Confidence = log prob of the greedy token (higher = more confident)
            confidence = log_probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)  # [B, K]
            # Set already-revealed positions to -inf so they're not selected again
            confidence[~mask] = float('-inf')

            # Reveal the most confident positions
            n_to_reveal = min(positions_per_step, mask.sum(dim=1).min().item())
            if n_to_reveal <= 0:
                break

            # Per-sample: find top-n_to_reveal confident masked positions
            for b in range(B):
                masked_indices = mask[b].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    continue
                conf_vals = confidence[b, masked_indices]
                n_reveal = min(n_to_reveal, len(masked_indices))
                _, top_idx = conf_vals.topk(n_reveal)
                reveal_positions = masked_indices[top_idx]
                mask[b, reveal_positions] = False

        if anchor_token_id is not None:
            draft_ids[:, 0] = anchor_token_id

        return draft_ids, final_logits
