"""
DFlash v6 — Drafter aligned with z-lab paper (arXiv 2602.06036).

Key differences from v5:
  - Unified KV injection: same k_proj/v_proj for context AND noise (z-lab design)
  - Feature fusion: concat 5 layers → linear → RMSNorm (not weighted average)
  - 8 drafter layers for MoE target (not 5)
  - GQA: 32 attn heads, 4 KV heads
  - No block_pos_embed, no mask_ratio_proj, no h37 residual hack
  - gamma=7 streak distillation loss (not 0.1)

Architecture:
  Target hidden states [0..P] from 5 layers
       ↓ concat → Linear(5*H, H) → RMSNorm
  context [B, ctx_len, H]
       ↓
  ┌─ Layer 1 ──────────────────────┐
  │  Q from noise embeddings        │
  │  K = k_proj([ctx ; noise])      │  ← SHARED projections
  │  V = v_proj([ctx ; noise])      │
  │  Bidirectional attention         │
  └─────────────────────────────────┘
       ↓  ... × 8 layers
  Output norm → lm_head → logits
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_v6 import DFlashV6Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class DFlashAttentionV6(nn.Module):
    """Bidirectional GQA attention with unified KV injection.

    Matches z-lab's Qwen3DFlashAttention: the SAME k_proj and v_proj
    are used for both context (target hidden) and noise (draft) inputs.
    Context K/V are prepended to noise K/V.
    """

    def __init__(self, config: DFlashV6Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        # Q only from noise, K/V shared between context and noise
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK normalization (Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, hidden_states, target_hidden, attn_mask=None):
        """
        hidden_states: [B, K, H] — noise (draft) embeddings
        target_hidden: [B, ctx_len, H] — fused target features
        attn_mask: [B, 1, K, ctx_len+K] — float mask (0=attend, -inf=ignore)
        """
        B, K, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from noise only
        q = self.q_proj(hidden_states)
        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V from BOTH context and noise using SAME projections
        k_ctx = self.k_proj(target_hidden).view(B, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_ctx = self.v_proj(target_hidden).view(B, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k_noise = self.k_proj(hidden_states).view(B, K, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_noise = self.v_proj(hidden_states).view(B, K, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Concatenate: [context ; noise] along sequence dim
        k = torch.cat([k_ctx, k_noise], dim=2)  # [B, kv_heads, ctx_len+K, head_dim]
        v = torch.cat([v_ctx, v_noise], dim=2)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, k.shape[3], self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, v.shape[3], self.head_dim)

        # Bidirectional attention (is_causal=False)
        drop_p = self.attention_dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_p
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, -1)
        return self.o_proj(attn_out)


class DFlashMLP(nn.Module):
    """SwiGLU MLP."""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    """Pre-norm transformer layer with unified KV injection."""
    def __init__(self, config: DFlashV6Config):
        super().__init__()
        self.self_attn = DFlashAttentionV6(config)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, attn_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, target_hidden, attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def build_attention_mask(block_size, context_lengths, max_ctx_len, device):
    """Build attention mask for [context ; block] KV layout.

    Block tokens attend bidirectionally to each other and to all
    non-padded context positions.

    Returns: [B, 1, K, ctx_len+K] float mask (0=attend, -inf=ignore)
    """
    B = context_lengths.shape[0]
    total_kv = max_ctx_len + block_size

    mask = torch.zeros(B, 1, block_size, total_kv, device=device)

    # Mask padded context positions
    positions = torch.arange(max_ctx_len, device=device).unsqueeze(0)
    ctx_invalid = positions >= context_lengths.unsqueeze(1)
    ctx_invalid = ctx_invalid.unsqueeze(1).unsqueeze(1).expand(-1, 1, block_size, -1)
    mask[:, :, :, :max_ctx_len].masked_fill_(ctx_invalid, float('-inf'))

    return mask


class DFlashDraftModelV6(nn.Module):
    """DFlash v6: z-lab aligned block diffusion drafter.

    Architecture matches z-lab/Qwen3-Coder-30B-A3B-DFlash:
    - 8-layer dense bidirectional Transformer
    - Shared embed_tokens + lm_head from target (frozen)
    - Unified KV injection (same projections for context and noise)
    - Feature fusion: concat → Linear → RMSNorm
    - Mask token denoising (single forward pass)
    """

    def __init__(self, config: DFlashV6Config):
        super().__init__()
        self.config = config
        H = config.hidden_size

        # Shared with target (will be loaded from Qwen weights, then frozen)
        self.embed_tokens = nn.Embedding(config.target_vocab_size, H)
        self.lm_head = nn.Linear(H, config.target_vocab_size, bias=False)
        self.norm = RMSNorm(H, config.rms_norm_eps)  # output norm (from target)

        # Feature fusion: concat 5 layers → project to hidden_size → norm
        # z-lab: self.fc = Linear(n_features * H, H, bias=False) + RMSNorm
        self.fc = nn.Linear(config.num_feature_layers * config.target_hidden_size, H, bias=False)
        self.hidden_norm = RMSNorm(H, config.rms_norm_eps)

        # Drafter layers
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def freeze_shared_params(self):
        """Freeze embed_tokens, lm_head, and output norm."""
        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True

    @property
    def gradient_checkpointing(self):
        return getattr(self, '_gradient_checkpointing', False)

    def _fuse_context(self, context_hidden_list):
        """Fuse target hidden states: concat along last dim → Linear → RMSNorm.

        Args:
            context_hidden_list: list of num_feature_layers tensors [B, ctx_len, H]

        Returns:
            [B, ctx_len, hidden_size]
        """
        # Concat: [B, ctx_len, num_features * H]
        cat = torch.cat(context_hidden_list, dim=-1)
        return self.hidden_norm(self.fc(cat))

    def forward_train(self, block_input_ids, context_hidden_list, context_lengths):
        """Training forward pass.

        Args:
            block_input_ids: [B, K] — ground truth tokens for the block
                block_input_ids[:, 0] = anchor token
                block_input_ids[:, 1:] = tokens to predict
            context_hidden_list: list of num_feature_layers tensors [B, max_ctx_len, H]
            context_lengths: [B] — actual context lengths

        Returns: (loss, logits)
        """
        B, K = block_input_ids.shape
        device = block_input_ids.device

        # 1. Fuse context hidden states
        ctx = self._fuse_context(context_hidden_list)  # [B, max_ctx_len, H]

        # 2. Build noise input: anchor at pos 0, mask tokens at pos 1..K-1
        mask_emb = self.embed_tokens.weight.new_zeros(self.config.hidden_size)
        # Use the mask_token_id embedding as the mask representation
        with torch.no_grad():
            mask_emb = self.embed_tokens(
                torch.tensor([self.config.mask_token_id], device=device)
            ).squeeze(0)  # [H]

        x = mask_emb.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
        # Position 0: anchor token (real embedding)
        anchor_emb = self.embed_tokens(block_input_ids[:, 0])  # [B, H]
        x[:, 0] = anchor_emb

        # 3. Attention mask for padded context
        max_ctx_len = ctx.shape[1]
        attn_mask = build_attention_mask(K, context_lengths, max_ctx_len, device)

        # 4. Forward through layers with unified KV injection
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, ctx, attn_mask, use_reentrant=False
                )
            else:
                x = layer(x, target_hidden=ctx, attn_mask=attn_mask)

        # 5. Output: norm → lm_head
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)  # [B, K, vocab]

        # 6. Streak distillation loss (positions 1..K-1 only, anchor excluded)
        # w_k = exp(-(k-1) / gamma) for k=1..K-1
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        # Targets: predict block_input_ids[1:] from positions [1..K-1]
        # Actually: position k in drafter output should predict block_input_ids[k]
        # But position 0 is the anchor — we compute loss on positions 1..K-1
        pred_logits = logits[:, 1:, :]  # [B, K-1, vocab] — predictions at pos 1..K-1
        target_ids = block_input_ids[:, 1:]  # [B, K-1] — ground truth tokens 1..K-1

        loss_per_token = loss_fct(
            pred_logits.reshape(-1, self.config.target_vocab_size),
            target_ids.reshape(-1)
        ).view(B, K - 1)

        # Position weights: w_k = exp(-(k-1)/gamma) for k=1..K-1
        # k=1 → w=1.0, k=K-1 → w=exp(-(K-2)/gamma)
        gamma = self.config.loss_decay_gamma
        pos_indices = torch.arange(K - 1, device=device, dtype=torch.float)
        position_weights = torch.exp(-pos_indices / gamma)  # [K-1]

        weighted_loss = (loss_per_token * position_weights[None, :]).sum() / (
            position_weights.sum() * B
        )

        return weighted_loss, logits

    @torch.no_grad()
    def generate_block(self, context_hidden_list, context_lengths=None,
                       temperature=0.0, anchor_token_id=None):
        """Generate a draft block (single-step denoising).

        Args:
            context_hidden_list: list of tensors [B, ctx_len, H]
            context_lengths: [B] or None
            temperature: 0 = greedy
            anchor_token_id: token for position 0

        Returns: (draft_ids [B, K-1], logits [B, K, V])
            draft_ids are the K-1 NEW tokens (excluding anchor)
        """
        B = context_hidden_list[0].shape[0]
        device = context_hidden_list[0].device
        K = self.config.block_size

        # 1. Fuse context
        ctx = self._fuse_context(context_hidden_list)
        ctx_len = ctx.shape[1]

        if context_lengths is None:
            context_lengths = torch.full((B,), ctx_len, device=device, dtype=torch.long)

        # 2. Build noise: all mask tokens, anchor at pos 0
        mask_emb = self.embed_tokens(
            torch.tensor([self.config.mask_token_id], device=device)
        ).squeeze(0)

        x = mask_emb.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
        if anchor_token_id is not None:
            anchor_ids = torch.tensor([anchor_token_id], device=device).expand(B)
            x[:, 0] = self.embed_tokens(anchor_ids)

        # 3. Attention mask
        attn_mask = build_attention_mask(K, context_lengths, ctx_len, device)

        # 4. Forward
        for layer in self.layers:
            x = layer(x, target_hidden=ctx, attn_mask=attn_mask)

        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)

        # 5. Decode positions 1..K-1 (exclude anchor at 0)
        draft_logits = logits[:, 1:, :]  # [B, K-1, V]
        if temperature == 0.0:
            draft_ids = draft_logits.argmax(dim=-1)
        else:
            probs = F.softmax(draft_logits / temperature, dim=-1)
            draft_ids = torch.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(B, K - 1)

        return draft_ids, logits

    def count_parameters(self):
        """Count trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable, frozen
