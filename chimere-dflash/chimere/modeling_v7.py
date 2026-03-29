"""
DFlash v7 — v6 + RoPE positional encoding.

Only change from v6: RoPE applied to Q and K after QK normalization.
Without RoPE, all 16 mask positions in a block receive identical embeddings,
making the model unable to distinguish position 1 from position 15.

Architecture:
  Q = q_proj(noise)           → q_norm(Q)  → apply_rotary(Q, cos[-K:], sin[-K:])
  K = k_proj(ctx ++ noise)    → k_norm(K)  → apply_rotary(K, cos, sin)
  V = v_proj(ctx ++ noise)
  → GQA bidirectional attention → o_proj
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_v7 import DFlashV7Config


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """RoPE positional encoding (no trainable parameters)."""

    def __init__(self, head_dim, rope_theta=10_000_000.0, max_seq_len=262144):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids):
        """
        Args:
            position_ids: [B, seq_len] — absolute positions

        Returns:
            cos, sin: [B, seq_len, head_dim]
        """
        # inv_freq: [head_dim/2]
        # position_ids: [B, seq_len]
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        # freqs: [B, seq_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [B, seq_len, head_dim]
        return emb.cos(), emb.sin()


def _rotate_half(x):
    """Rotate half of the hidden dims: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply RoPE to Q and K, handling Q/K length mismatch.

    Q has K positions (block size), K has ctx_len+K positions.
    Q receives positions [ctx_len..ctx_len+K-1] (last q_len from cos/sin).
    K receives all positions [0..ctx_len+K-1].

    Args:
        q: [B, heads, q_len, head_dim]
        k: [B, kv_heads, kv_len, head_dim]
        cos: [B, kv_len, head_dim]
        sin: [B, kv_len, head_dim]
    """
    q_len = q.shape[2]

    # K gets full cos/sin: [B, 1, kv_len, head_dim]
    cos_k = cos.unsqueeze(1)
    sin_k = sin.unsqueeze(1)
    k_embed = k * cos_k + _rotate_half(k) * sin_k

    # Q gets last q_len positions (= the block positions after context)
    cos_q = cos[:, -q_len:, :].unsqueeze(1)
    sin_q = sin[:, -q_len:, :].unsqueeze(1)
    q_embed = q * cos_q + _rotate_half(q) * sin_q

    return q_embed, k_embed


class DFlashAttentionV7(nn.Module):
    """Bidirectional GQA attention with unified KV injection + RoPE."""

    def __init__(self, config: DFlashV7Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK normalization (Qwen3 style)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, hidden_states, target_hidden, attn_mask=None, position_embeddings=None):
        """
        Args:
            hidden_states: [B, K, H] — noise (draft) embeddings
            target_hidden: [B, ctx_len, H] — fused target features
            attn_mask: [B, 1, K, ctx_len+K] — float mask
            position_embeddings: (cos, sin) from RotaryEmbedding
        """
        B, K, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from noise only
        q = self.q_proj(hidden_states)
        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V from BOTH context and noise
        k_ctx = self.k_proj(target_hidden).view(B, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_ctx = self.v_proj(target_hidden).view(B, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k_noise = self.k_proj(hidden_states).view(B, K, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_noise = self.v_proj(hidden_states).view(B, K, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = torch.cat([k_ctx, k_noise], dim=2)  # [B, kv_heads, ctx_len+K, head_dim]
        v = torch.cat([v_ctx, v_noise], dim=2)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (NEW in v7)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, k.shape[3], self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, v.shape[3], self.head_dim)

        # Bidirectional attention
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
    """Pre-norm transformer layer with unified KV injection + RoPE."""
    def __init__(self, config: DFlashV7Config):
        super().__init__()
        self.self_attn = DFlashAttentionV7(config)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, target_hidden, attn_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, target_hidden, attn_mask, position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def build_attention_mask(block_size, context_lengths, max_ctx_len, device):
    """Build attention mask for [context ; block] KV layout.

    Context is LEFT-PADDED: valid positions are right-aligned.
    For sample i with ctx_len_i, positions 0..max_ctx_len-ctx_len_i-1 are padding (masked -inf),
    and positions max_ctx_len-ctx_len_i..max_ctx_len-1 are valid.
    """
    B = context_lengths.shape[0]
    total_kv = max_ctx_len + block_size

    mask = torch.zeros(B, 1, block_size, total_kv, device=device)

    # Left-padding: padding is at the start (positions 0..pad_len-1)
    # pad_len = max_ctx_len - context_lengths[i]
    positions = torch.arange(max_ctx_len, device=device).unsqueeze(0)  # [1, max_ctx_len]
    pad_len = (max_ctx_len - context_lengths).unsqueeze(1)              # [B, 1]
    ctx_invalid = positions < pad_len                                    # [B, max_ctx_len]
    ctx_invalid = ctx_invalid.unsqueeze(1).unsqueeze(1).expand(-1, 1, block_size, -1)
    mask[:, :, :, :max_ctx_len].masked_fill_(ctx_invalid, float('-inf'))

    return mask


class DFlashDraftModelV7(nn.Module):
    """DFlash v7: v6 + RoPE positional encoding.

    The only architectural change is RoPE applied to Q/K in every
    attention layer, giving the model position awareness within blocks.
    """

    def __init__(self, config: DFlashV7Config):
        super().__init__()
        self.config = config
        H = config.hidden_size
        T = config.target_hidden_size  # embedding/lm_head dimension

        # Shared with target (frozen) — always use target_hidden_size
        self.embed_tokens = nn.Embedding(config.target_vocab_size, T)
        self.lm_head = nn.Linear(T, config.target_vocab_size, bias=False)
        self.norm = RMSNorm(T, config.rms_norm_eps)

        # Feature fusion: target features -> drafter hidden size
        self.fc = nn.Linear(config.num_feature_layers * T, H, bias=False)
        self.hidden_norm = RMSNorm(H, config.rms_norm_eps)

        # Input projection: target embed -> drafter hidden (when sizes differ)
        if H != T:
            self.input_proj = nn.Linear(T, H, bias=False)
        else:
            self.input_proj = None

        # Output projection: drafter hidden -> target hidden (when sizes differ)
        if config.use_output_proj or H != T:
            self.output_proj = nn.Linear(H, T, bias=False)
        else:
            self.output_proj = None

        # RoPE (NEW in v7)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )

        # Drafter layers
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def freeze_shared_params(self):
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
        cat = torch.cat(context_hidden_list, dim=-1)
        return self.hidden_norm(self.fc(cat))

    def _build_position_embeddings(self, ctx_len, block_size, batch_size, device,
                                    anchor_positions=None):
        """Build position IDs and compute RoPE cos/sin.

        With absolute positions (anchor_positions provided):
            Context = positions BEFORE anchor: [anchor_pos-(ctx_len-1), ..., anchor_pos]
            Block = positions AFTER anchor: [anchor_pos+1, ..., anchor_pos+block_size]
            Total: ctx_len + block_size positions

        Without (fallback):
            Positions = [0, 1, ..., ctx_len + block_size - 1]
        """
        total_len = ctx_len + block_size
        if anchor_positions is not None:
            # Offsets: context positions are -(ctx_len-1)..0 relative to anchor,
            #          block positions are +1..+block_size relative to anchor
            offsets = torch.arange(
                -(ctx_len - 1),       # e.g. -31 for ctx_len=32
                block_size + 1,       # e.g. +17 (exclusive) for block_size=16
                device=device,
            )  # total_len elements: [-31, -30, ..., 0, 1, ..., 16]
            position_ids = anchor_positions.unsqueeze(1) + offsets.unsqueeze(0)  # [B, total_len]
            # Clamp for edge cases where anchor_pos < ctx_len-1
            position_ids = position_ids.clamp(min=0)
        else:
            position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rotary_emb(position_ids)
        return cos, sin

    def forward_train(self, block_input_ids, context_hidden_list, context_lengths,
                      anchor_positions=None):
        B, K = block_input_ids.shape
        device = block_input_ids.device

        # 1. Fuse context
        ctx = self._fuse_context(context_hidden_list)

        # 2. Build noise input (in drafter hidden space)
        with torch.no_grad():
            mask_emb = self.embed_tokens(
                torch.tensor([self.config.mask_token_id], device=device)
            ).squeeze(0)
            if self.input_proj is not None:
                mask_emb = self.input_proj(mask_emb.float())

        x = mask_emb.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
        anchor_emb = self.embed_tokens(block_input_ids[:, 0])
        if self.input_proj is not None:
            anchor_emb = self.input_proj(anchor_emb.float())
        x[:, 0] = anchor_emb

        # 3. Attention mask
        max_ctx_len = ctx.shape[1]
        attn_mask = build_attention_mask(K, context_lengths, max_ctx_len, device)

        # 4. RoPE position embeddings with ABSOLUTE positions (NEW in v7)
        cos, sin = self._build_position_embeddings(
            max_ctx_len, K, B, device, anchor_positions=anchor_positions
        )

        # 5. Forward through layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, ctx, attn_mask, (cos, sin), use_reentrant=False
                )
            else:
                x = layer(x, target_hidden=ctx, attn_mask=attn_mask,
                          position_embeddings=(cos, sin))

        # 6. Output (project back to target hidden size if needed)
        if self.output_proj is not None:
            x = self.output_proj(x)
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)

        # 7. Streak distillation loss
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        pred_logits = logits[:, 1:, :]
        target_ids = block_input_ids[:, 1:]

        loss_per_token = loss_fct(
            pred_logits.reshape(-1, self.config.target_vocab_size),
            target_ids.reshape(-1)
        ).view(B, K - 1)

        gamma = self.config.loss_decay_gamma
        pos_indices = torch.arange(K - 1, device=device, dtype=torch.float)
        position_weights = torch.exp(-pos_indices / gamma)

        weighted_loss = (loss_per_token * position_weights[None, :]).sum() / (
            position_weights.sum() * B
        )

        return weighted_loss, logits

    def forward_train_multi(self, block_input_ids, context_hidden_list, context_lengths,
                            anchor_positions=None, chunk_size=16):
        """Training forward with chunked lm_head to save VRAM.

        Instead of materializing [B, K, vocab_size] logits, computes loss
        in chunks of `chunk_size` blocks at a time.

        Returns:
            (loss, None) — no full logits tensor returned
        """
        B, K = block_input_ids.shape
        device = block_input_ids.device

        # 1. Fuse context
        ctx = self._fuse_context(context_hidden_list)

        # 2. Build noise input
        with torch.no_grad():
            mask_emb = self.embed_tokens(
                torch.tensor([self.config.mask_token_id], device=device)
            ).squeeze(0)
            if self.input_proj is not None:
                mask_emb = self.input_proj(mask_emb.float())

        x = mask_emb.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
        anchor_emb = self.embed_tokens(block_input_ids[:, 0])
        if self.input_proj is not None:
            anchor_emb = self.input_proj(anchor_emb.float())
        x[:, 0] = anchor_emb

        # 3. Attention mask
        max_ctx_len = ctx.shape[1]
        attn_mask = build_attention_mask(K, context_lengths, max_ctx_len, device)

        # 4. RoPE
        cos, sin = self._build_position_embeddings(
            max_ctx_len, K, B, device, anchor_positions=anchor_positions
        )

        # 5. Forward through layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, ctx, attn_mask, (cos, sin), use_reentrant=False
                )
            else:
                x = layer(x, target_hidden=ctx, attn_mask=attn_mask,
                          position_embeddings=(cos, sin))

        # 6. Output projection
        if self.output_proj is not None:
            x = self.output_proj(x)
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)

        # 7. Chunked loss computation — avoid full [B, K, vocab] materialization
        # Compute loss on positions 1..K-1 (predict next token in block)
        gamma = self.config.loss_decay_gamma
        pos_indices = torch.arange(K - 1, device=device, dtype=torch.float)
        position_weights = torch.exp(-pos_indices / gamma)

        target_ids = block_input_ids[:, 1:]  # [B, K-1]
        hidden_for_pred = x[:, 1:, :]  # [B, K-1, H] — predict from position 1..K-1

        total_weighted_loss = torch.tensor(0.0, device=device)
        n_pred_positions = K - 1

        for start in range(0, n_pred_positions, chunk_size):
            end = min(start + chunk_size, n_pred_positions)
            chunk_hidden = hidden_for_pred[:, start:end, :]  # [B, chunk, H]
            chunk_logits = self.lm_head(chunk_hidden)  # [B, chunk, vocab]
            chunk_targets = target_ids[:, start:end]  # [B, chunk]
            chunk_weights = position_weights[start:end]  # [chunk]

            loss_per_token = F.cross_entropy(
                chunk_logits.reshape(-1, self.config.target_vocab_size),
                chunk_targets.reshape(-1),
                reduction="none",
            ).view(B, end - start)

            total_weighted_loss = total_weighted_loss + (
                loss_per_token * chunk_weights[None, :]
            ).sum()

        weighted_loss = total_weighted_loss / (position_weights.sum() * B)

        return weighted_loss, None

    @torch.no_grad()
    def generate_block(self, context_hidden_list, context_lengths=None,
                       temperature=0.0, anchor_token_id=None, anchor_positions=None):
        B = context_hidden_list[0].shape[0]
        device = context_hidden_list[0].device
        K = self.config.block_size

        ctx = self._fuse_context(context_hidden_list)
        ctx_len = ctx.shape[1]

        if context_lengths is None:
            context_lengths = torch.full((B,), ctx_len, device=device, dtype=torch.long)

        # Build noise (in drafter hidden space)
        mask_emb = self.embed_tokens(
            torch.tensor([self.config.mask_token_id], device=device)
        ).squeeze(0)
        if self.input_proj is not None:
            mask_emb = self.input_proj(mask_emb.float())

        x = mask_emb.unsqueeze(0).unsqueeze(0).expand(B, K, -1).clone()
        if anchor_token_id is not None:
            anchor_ids = torch.tensor([anchor_token_id], device=device).expand(B)
            anchor_emb = self.embed_tokens(anchor_ids)
            if self.input_proj is not None:
                anchor_emb = self.input_proj(anchor_emb.float())
            x[:, 0] = anchor_emb

        # Attention mask
        attn_mask = build_attention_mask(K, context_lengths, ctx_len, device)

        # RoPE with absolute positions (NEW in v7)
        cos, sin = self._build_position_embeddings(
            ctx_len, K, B, device, anchor_positions=anchor_positions
        )

        # Forward
        for layer in self.layers:
            x = layer(x, target_hidden=ctx, attn_mask=attn_mask,
                      position_embeddings=(cos, sin))

        if self.output_proj is not None:
            x = self.output_proj(x)
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)

        # Decode positions 1..K-1
        draft_logits = logits[:, 1:, :]
        if temperature == 0.0:
            draft_ids = draft_logits.argmax(dim=-1)
        else:
            probs = F.softmax(draft_logits / temperature, dim=-1)
            draft_ids = torch.multinomial(
                probs.view(-1, probs.shape[-1]), 1
            ).view(B, K - 1)

        return draft_ids, logits

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable, frozen
