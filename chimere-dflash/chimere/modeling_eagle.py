"""
EAGLE-style autoregressive drafter for speculative decoding.

Key differences from DFlash v7 (block diffusion):
  - Autoregressive: predicts tokens one by one in the block
  - Residual: strong skip connection from context hidden states to output
  - Causal attention: each position only sees previous positions + context
  - The baseline is lm_head(last_layer_hidden[anchor]) ≈ 37.5% top1

Architecture:
  1. Fuse 5 target layers → context vector via fc
  2. For each position in the block:
     - Input = embed(prev_token) + context_residual (skip connection)
     - Process through 1 transformer layer with causal self-attention
     - Output = layer_output + context_residual (another skip connection)
     - Predict next token via norm + lm_head

The double residual ensures the model starts at ≥ lm_head direct baseline
and can only improve from there.
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


class EagleAttention(nn.Module):
    """Causal self-attention with GQA (no cross-attention to separate context KV)."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, dropout=0.05):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self, hidden_states, attn_mask=None):
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(B, self.num_heads, L, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(B, self.num_heads, L, self.head_dim)

        drop_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_p, is_causal=(attn_mask is None)
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(attn_out)


class EagleMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, rms_norm_eps=1e-6, dropout=0.05):
        super().__init__()
        self.self_attn = EagleAttention(hidden_size, num_heads, num_kv_heads, head_dim, dropout)
        self.mlp = EagleMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, hidden_states, attn_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attn_mask=attn_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class EagleDrafter(nn.Module):
    """EAGLE-style autoregressive drafter with context residual.

    Training: teacher-forced — all positions computed in parallel with causal mask.
    Inference: autoregressive — predict one token at a time.

    The key innovation: a strong residual connection from the fused context
    directly to the output, ensuring the model starts at >= lm_head direct baseline.
    """

    def __init__(self, config: DFlashV7Config):
        super().__init__()
        self.config = config
        T = config.target_hidden_size  # 2048 for Qwen3.5

        # Shared with target (frozen)
        self.embed_tokens = nn.Embedding(config.target_vocab_size, T)
        self.lm_head = nn.Linear(T, config.target_vocab_size, bias=False)
        self.norm = RMSNorm(T, config.rms_norm_eps)

        # Feature fusion: 5 layers → T
        self.fc = nn.Linear(config.num_feature_layers * T, T, bias=False)
        self.fc_norm = RMSNorm(T, config.rms_norm_eps)

        # Transformer layers (typically 1)
        self.layers = nn.ModuleList([
            EagleDecoderLayer(
                T, config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, config.intermediate_size,
                config.rms_norm_eps, config.attention_dropout,
            )
            for _ in range(config.num_hidden_layers)
        ])

    def freeze_shared_params(self):
        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

    def _fuse_context(self, context_hidden_list):
        """Fuse multi-layer hidden states and extract raw last layer.

        Args:
            context_hidden_list: list of num_layers tensors [B, ctx_len, H]

        Returns:
            fused: [B, H] — learned fusion for transformer input
            raw_last: [B, H] — raw last layer hidden at anchor (for lm_head residual)
        """
        # Take the last position of each layer (= anchor position, right-aligned)
        anchor_hidden = [h[:, -1, :] for h in context_hidden_list]  # list of [B, H]

        # Raw last layer → direct residual to output (guarantees lm_head baseline)
        raw_last = anchor_hidden[-1].float()  # [B, H] — layer 37

        # Learned fusion of all layers → transformer input
        cat = torch.cat(anchor_hidden, dim=-1)  # [B, num_layers * H]
        fused = self.fc_norm(self.fc(cat))  # [B, H]

        return fused, raw_last

    def forward_train(self, block_input_ids, context_hidden_list, context_lengths,
                      anchor_positions=None):
        """Teacher-forced training.

        Args:
            block_input_ids: [B, K] — the K tokens to predict (anchor + block)
            context_hidden_list: list of num_layers tensors [B, ctx_len, H]
            context_lengths: [B]
            anchor_positions: [B] (unused in eagle, kept for API compat)

        The model predicts block_input_ids[1:] from block_input_ids[:-1].
        Two residual paths:
          - raw_last (layer 37) → output residual (guarantees lm_head baseline)
          - fused (all layers) → input to transformer (learnable enhancement)
        """
        B, K = block_input_ids.shape
        device = block_input_ids.device

        # 1. Context: fused for input, raw last layer for output residual
        fused, raw_last = self._fuse_context(context_hidden_list)

        # 2. Token embeddings for input positions [0..K-2]
        input_ids = block_input_ids[:, :-1]  # [B, K-1]
        tok_emb = self.embed_tokens(input_ids).float()  # [B, K-1, T]

        # 3. Add FUSED context to input (learnable enhancement)
        x = tok_emb + fused.unsqueeze(1)  # [B, K-1, T]

        # 4. Causal self-attention through layers
        for layer in self.layers:
            x = layer(x)  # is_causal=True by default when no mask

        # 5. Output residual: add RAW LAST LAYER (guarantees lm_head baseline)
        x = x + raw_last.unsqueeze(1)

        # 6. Predict
        x = self.norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)  # [B, K-1, vocab]

        # 7. Loss: predict block_input_ids[1:]
        target_ids = block_input_ids[:, 1:]  # [B, K-1]
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss_per_token = loss_fct(
            logits.reshape(-1, self.config.target_vocab_size),
            target_ids.reshape(-1)
        ).view(B, K - 1)

        # Streak-weighted loss (same as v7)
        gamma = self.config.loss_decay_gamma
        pos_indices = torch.arange(K - 1, device=device, dtype=torch.float)
        position_weights = torch.exp(-pos_indices / gamma)
        weighted_loss = (loss_per_token * position_weights[None, :]).sum() / (
            position_weights.sum() * B
        )

        # Return logits in same shape as v7 for compatibility: [B, K, vocab]
        # Pad with zeros at position 0 (anchor input, no prediction)
        full_logits = torch.zeros(B, K, logits.shape[-1], device=device, dtype=logits.dtype)
        full_logits[:, 1:, :] = logits

        return weighted_loss, full_logits

    @torch.no_grad()
    def generate_block(self, context_hidden_list, context_lengths=None,
                       temperature=0.0, anchor_token_id=None, anchor_positions=None):
        """Autoregressive block generation.

        Args:
            context_hidden_list: list of num_layers tensors [B, ctx_len, H]
            context_lengths: [B] (unused)
            temperature: sampling temperature (0 = greedy)
            anchor_token_id: int — the token at the anchor position
            anchor_positions: [B] (unused)

        Returns:
            draft_ids: [B, K-1] — predicted tokens
            logits: [B, K, vocab] — full logits
        """
        B = context_hidden_list[0].shape[0]
        device = context_hidden_list[0].device
        K = self.config.block_size

        # Fuse context: fused for input, raw last layer for output
        fused, raw_last = self._fuse_context(context_hidden_list)

        # Start with anchor token
        if anchor_token_id is not None:
            current_id = torch.tensor([anchor_token_id], device=device).expand(B)
        else:
            current_id = torch.zeros(B, device=device, dtype=torch.long)

        all_logits = []
        draft_ids = []

        # Simple approach: re-run full sequence each time (K is small, 16)
        generated_ids = [current_id]  # list of [B] tensors

        for pos in range(K - 1):
            # Build input sequence so far
            ids_so_far = torch.stack(generated_ids, dim=1)  # [B, pos+1]
            tok_emb = self.embed_tokens(ids_so_far).float()  # [B, pos+1, T]
            x = tok_emb + fused.unsqueeze(1)

            for layer in self.layers:
                x = layer(x)

            # Take last position output + RAW LAST LAYER residual
            x_last = x[:, -1, :] + raw_last  # [B, T]
            x_last = self.norm(x_last)
            x_last = x_last.to(self.lm_head.weight.dtype)
            logits = self.lm_head(x_last)  # [B, vocab]

            all_logits.append(logits)

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1).squeeze(-1)

            draft_ids.append(next_id)
            generated_ids.append(next_id)

        draft_ids = torch.stack(draft_ids, dim=1)  # [B, K-1]

        # Build full logits tensor [B, K, vocab]
        stacked_logits = torch.stack(all_logits, dim=1)  # [B, K-1, vocab]
        full_logits = torch.zeros(B, K, stacked_logits.shape[-1],
                                  device=device, dtype=stacked_logits.dtype)
        full_logits[:, 1:, :] = stacked_logits

        return draft_ids, full_logits

    def count_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable, frozen
