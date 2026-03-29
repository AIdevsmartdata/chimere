"""
DFlash v8 — Deep KV Injection.

Key change from v7: instead of fusing target hidden states into a single
context vector that all layers share, we project target features into
layer-specific K/V pairs. Each drafter attention layer receives fresh
target signal through its own dedicated K/V projections.

Architecture:
  ctx = fuse(target_layers)                       # [B, ctx_len, H] (same as v7)
  For each layer i:
    ctx_k_i = ctx_k_projs[i](ctx)                # [B, ctx_len, kv_heads * head_dim]
    ctx_v_i = ctx_v_projs[i](ctx)                # [B, ctx_len, kv_heads * head_dim]
    Q = q_proj(noise) -> q_norm -> RoPE
    K = cat(k_norm(ctx_k_i) + RoPE, k_proj(noise) -> k_norm -> RoPE)
    V = cat(ctx_v_i, v_proj(noise))
    -> GQA bidirectional attention -> o_proj

This adds ~67M params (8 layers * 2 * 2048 * 512) but keeps the target
signal fresh at every layer depth, preventing dilution.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config_v7 import DFlashV7Config
from .modeling_v7 import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    _rotate_half,
    build_attention_mask,
    DFlashMLP,
)


class DFlashAttentionV8(nn.Module):
    """Bidirectional GQA attention with deep KV injection + RoPE.

    Unlike V7, this module does NOT project context through its own k_proj/v_proj.
    Instead, it receives pre-projected ctx_k and ctx_v from the parent model's
    per-layer projection modules.
    """

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

    def forward(self, hidden_states, ctx_k, ctx_v, attn_mask=None, position_embeddings=None):
        """
        Args:
            hidden_states: [B, K, H] — noise (draft) embeddings
            ctx_k: [B, ctx_len, num_kv_heads, head_dim] — pre-projected context keys
            ctx_v: [B, ctx_len, num_kv_heads, head_dim] — pre-projected context values
            attn_mask: [B, 1, K, ctx_len+K] — float mask
            position_embeddings: (cos, sin) from RotaryEmbedding
        """
        B, K, _ = hidden_states.shape
        ctx_len = ctx_k.shape[1]

        # Q from noise only
        q = self.q_proj(hidden_states)
        q = q.view(B, K, self.num_heads, self.head_dim).transpose(1, 2)

        # K/V from noise (projected by this layer's own k_proj/v_proj)
        k_noise = self.k_proj(hidden_states).view(B, K, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_noise = self.v_proj(hidden_states).view(B, K, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Context K/V are pre-projected — just transpose to [B, kv_heads, ctx_len, head_dim]
        k_ctx = ctx_k.transpose(1, 2)  # [B, num_kv_heads, ctx_len, head_dim]
        v_ctx = ctx_v.transpose(1, 2)  # [B, num_kv_heads, ctx_len, head_dim]

        # Concatenate context and noise K/V
        k = torch.cat([k_ctx, k_noise], dim=2)  # [B, kv_heads, ctx_len+K, head_dim]
        v = torch.cat([v_ctx, v_noise], dim=2)

        # QK normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
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


class DFlashDecoderLayerV8(nn.Module):
    """Pre-norm transformer layer with deep KV injection + RoPE."""
    def __init__(self, config: DFlashV7Config):
        super().__init__()
        self.self_attn = DFlashAttentionV8(config)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, ctx_k, ctx_v, attn_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, ctx_k, ctx_v, attn_mask, position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashDraftModelV8(nn.Module):
    """DFlash v8: Deep KV Injection.

    Instead of sharing a single fused context across all layers, each drafter
    layer has its own K/V projections that re-project the fused context.
    This keeps the target signal fresh at every layer depth.
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

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.rope_theta, config.max_position_embeddings
        )

        # Drafter layers (V8 attention with deep KV)
        self.layers = nn.ModuleList([
            DFlashDecoderLayerV8(config) for _ in range(config.num_hidden_layers)
        ])

        # Deep KV projections — one pair per layer
        # Project fused context [B, ctx_len, H] -> K/V for each layer
        kv_dim = config.num_key_value_heads * config.head_dim
        self.ctx_k_projs = nn.ModuleList([
            nn.Linear(H, kv_dim, bias=False)
            for _ in range(config.num_hidden_layers)
        ])
        self.ctx_v_projs = nn.ModuleList([
            nn.Linear(H, kv_dim, bias=False)
            for _ in range(config.num_hidden_layers)
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

    def _project_ctx_kv(self, ctx):
        """Pre-compute per-layer K/V projections of the fused context.

        Args:
            ctx: [B, ctx_len, H] — fused context

        Returns:
            ctx_k_list: list of [B, ctx_len, num_kv_heads, head_dim] per layer
            ctx_v_list: list of [B, ctx_len, num_kv_heads, head_dim] per layer
        """
        B, ctx_len, _ = ctx.shape
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        ctx_k_list = []
        ctx_v_list = []
        for i in range(self.config.num_hidden_layers):
            k = self.ctx_k_projs[i](ctx)  # [B, ctx_len, kv_dim]
            v = self.ctx_v_projs[i](ctx)  # [B, ctx_len, kv_dim]
            ctx_k_list.append(k.view(B, ctx_len, num_kv_heads, head_dim))
            ctx_v_list.append(v.view(B, ctx_len, num_kv_heads, head_dim))

        return ctx_k_list, ctx_v_list

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
            offsets = torch.arange(
                -(ctx_len - 1),
                block_size + 1,
                device=device,
            )
            position_ids = anchor_positions.unsqueeze(1) + offsets.unsqueeze(0)
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

        # 2. Pre-compute per-layer K/V projections (DEEP KV)
        ctx_k_list, ctx_v_list = self._project_ctx_kv(ctx)

        # 3. Build noise input (in drafter hidden space)
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

        # 4. Attention mask
        max_ctx_len = ctx.shape[1]
        attn_mask = build_attention_mask(K, context_lengths, max_ctx_len, device)

        # 5. RoPE position embeddings with ABSOLUTE positions
        cos, sin = self._build_position_embeddings(
            max_ctx_len, K, B, device, anchor_positions=anchor_positions
        )

        # 6. Forward through layers with deep KV injection
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, ctx_k_list[i], ctx_v_list[i], attn_mask, (cos, sin),
                    use_reentrant=False
                )
            else:
                x = layer(x, ctx_k=ctx_k_list[i], ctx_v=ctx_v_list[i],
                          attn_mask=attn_mask, position_embeddings=(cos, sin))

        # 7. Output (project back to target hidden size if needed)
        if self.output_proj is not None:
            x = self.output_proj(x)
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)

        # 8. Streak distillation loss
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

        # 2. Pre-compute per-layer K/V projections (DEEP KV)
        ctx_k_list, ctx_v_list = self._project_ctx_kv(ctx)

        # 3. Build noise input
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

        # 4. Attention mask
        max_ctx_len = ctx.shape[1]
        attn_mask = build_attention_mask(K, context_lengths, max_ctx_len, device)

        # 5. RoPE
        cos, sin = self._build_position_embeddings(
            max_ctx_len, K, B, device, anchor_positions=anchor_positions
        )

        # 6. Forward through layers with deep KV injection
        for i, layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, ctx_k_list[i], ctx_v_list[i], attn_mask, (cos, sin),
                    use_reentrant=False
                )
            else:
                x = layer(x, ctx_k=ctx_k_list[i], ctx_v=ctx_v_list[i],
                          attn_mask=attn_mask, position_embeddings=(cos, sin))

        # 7. Output projection
        if self.output_proj is not None:
            x = self.output_proj(x)
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)

        # 8. Chunked loss computation — avoid full [B, K, vocab] materialization
        gamma = self.config.loss_decay_gamma
        pos_indices = torch.arange(K - 1, device=device, dtype=torch.float)
        position_weights = torch.exp(-pos_indices / gamma)

        target_ids = block_input_ids[:, 1:]  # [B, K-1]
        hidden_for_pred = x[:, 1:, :]  # [B, K-1, H]

        total_weighted_loss = torch.tensor(0.0, device=device)
        n_pred_positions = K - 1

        for start in range(0, n_pred_positions, chunk_size):
            end = min(start + chunk_size, n_pred_positions)
            chunk_hidden = hidden_for_pred[:, start:end, :]
            chunk_logits = self.lm_head(chunk_hidden)
            chunk_targets = target_ids[:, start:end]
            chunk_weights = position_weights[start:end]

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

        # Pre-compute per-layer K/V projections (DEEP KV)
        ctx_k_list, ctx_v_list = self._project_ctx_kv(ctx)

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

        # RoPE with absolute positions
        cos, sin = self._build_position_embeddings(
            ctx_len, K, B, device, anchor_positions=anchor_positions
        )

        # Forward with deep KV injection
        for i, layer in enumerate(self.layers):
            x = layer(x, ctx_k=ctx_k_list[i], ctx_v=ctx_v_list[i],
                      attn_mask=attn_mask, position_embeddings=(cos, sin))

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
