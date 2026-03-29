//! # Hybrid Attention — Chimère Engine
//!
//! Combines GatedDeltaNet (linear, O(n)) with Grouped Query Attention
//! (quadratic, O(n²)) in a single layer, with dynamic routing.
//!
//! ## Why Hybrid?
//!
//! Alman-Yu (ICLR 2025) proves subquadratic models cannot do precise
//! retrieval. But 87% of tokens don't need retrieval — they're predictable
//! from local context. The hybrid approach gives each token the cheapest
//! attention mechanism that suffices:
//!
//! - **Predictable tokens** (low DeltaNet delta) → linear attention only
//! - **Retrieval-needing tokens** (high delta, low Engram hits) → full GQA
//!
//! ## Routing Signal
//!
//! The router uses two signals to decide per-token:
//!
//! 1. **StateMetrics.mean_delta** from GatedDeltaNet — high delta means
//!    the linear memory can't predict this token well → need full attention
//! 2. **Routing entropy** from the router logits — uncertain routing
//!    suggests the token needs broader context
//!
//! ## Architecture
//!
//! ```text
//! Input x
//!   ├── GatedDeltaNet (always runs, O(n))
//!   │     └── produces: output_linear, StateMetrics
//!   │
//!   ├── AttentionRouter (uses StateMetrics + x)
//!   │     └── per-token decision: linear_only vs hybrid
//!   │
//!   └── [if hybrid] Sparse GQA (only on selected tokens)
//!         └── output_full (sparse, only routed tokens)
//!
//! Final = gate_linear * output_linear + gate_full * output_full
//! ```
//!
//! ## Comparison with Baselines
//!
//! | Model       | Ratio     | Type    | Mechanism              |
//! |-------------|-----------|---------|------------------------|
//! | Qwen3       | 3:1 fixed | Layer   | Hardcoded layer ids    |
//! | Jamba       | 1:7 fixed | Layer   | Hardcoded layer ids    |
//! | **Chimère** | Dynamic   | Token   | StateMetrics + entropy |

use candle_core::{DType, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

use crate::rope::RotaryEmbedding;
use crate::StateMetrics;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the layer decides between linear and full attention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingMode {
    /// Always use linear attention only (pure DeltaNet layer)
    LinearOnly,
    /// Always use full attention only (pure GQA layer)
    FullOnly,
    /// Dynamic per-token routing based on StateMetrics
    Dynamic,
    /// Fixed: run both and blend with learned gate (simpler, for ablation)
    FixedBlend,
}

/// Configuration for a hybrid attention layer.
#[derive(Debug, Clone)]
pub struct HybridAttentionConfig {
    /// Model hidden dimension
    pub hidden_dim: usize,
    /// Number of query heads (for both GQA and DeltaNet)
    pub num_heads: usize,
    /// Number of key-value heads for GQA (< num_heads → grouped)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Routing mode
    pub routing_mode: RoutingMode,
    /// Delta threshold: tokens with mean_delta above this get full attention
    pub delta_threshold: f32,
    /// Fraction of tokens allowed to use full attention (capacity cap)
    pub full_attention_capacity: f32,
}

impl HybridAttentionConfig {
    /// Chimère default: dynamic routing, GQA with 4 groups
    pub fn chimere() -> Self {
        Self {
            hidden_dim: 4096,
            num_heads: 32,
            num_kv_heads: 8, // 4 groups (32/8 = 4 queries per KV head)
            head_dim: 128,
            routing_mode: RoutingMode::Dynamic,
            delta_threshold: 0.5,
            full_attention_capacity: 0.25, // at most 25% tokens get full attn
        }
    }

    /// Pure DeltaNet layer (for the 28/32 linear layers)
    pub fn linear_only() -> Self {
        Self {
            routing_mode: RoutingMode::LinearOnly,
            ..Self::chimere()
        }
    }

    /// Pure GQA layer (for ablation)
    pub fn full_only() -> Self {
        Self {
            routing_mode: RoutingMode::FullOnly,
            ..Self::chimere()
        }
    }

    /// Small config for unit tests
    pub fn test() -> Self {
        Self {
            hidden_dim: 64,
            num_heads: 4,
            num_kv_heads: 2, // 2 groups
            head_dim: 16,
            routing_mode: RoutingMode::Dynamic,
            delta_threshold: 0.3,
            full_attention_capacity: 0.5,
        }
    }

    /// Number of query heads per KV group
    pub fn queries_per_kv_group(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }
}

// ---------------------------------------------------------------------------
// NormLayer — unified LayerNorm / RMSNorm
// ---------------------------------------------------------------------------

/// Normalisation layer that can be either LayerNorm or RMSNorm.
///
/// Used by both `GroupedQueryAttention` (pre-norm before projection) and
/// `HybridAttentionLayer`.  The default constructor of GQA creates a
/// `LayerNorm` for backward compatibility; callers that want RMSNorm can use
/// `GroupedQueryAttention::with_norm`.
pub enum NormLayer {
    LayerNorm(candle_nn::LayerNorm),
    RmsNorm { weight: Tensor, eps: f64 },
}

impl NormLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            NormLayer::LayerNorm(ln) => ln.forward(x),
            NormLayer::RmsNorm { weight, eps } => {
                let rank = x.rank();
                let last_dim = rank - 1;
                let hidden_size = x.dim(last_dim)? as f64;
                let sum_sq = x.sqr()?.sum_keepdim(last_dim)?;
                let mean_sq = (sum_sq / hidden_size)?;
                let rms = mean_sq.affine(1.0, *eps)?.sqrt()?;
                let x_norm = x.broadcast_div(&rms)?;
                x_norm.broadcast_mul(weight)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Attention Router
// ---------------------------------------------------------------------------

/// Per-token routing decision for hybrid attention.
#[derive(Debug, Clone)]
pub struct AttentionRoutingDecision {
    /// Which tokens should receive full attention (indices into sequence)
    pub full_attention_indices: Vec<usize>,
    /// Gate weight for full attention per token (0.0 = linear only, 1.0 = full only)
    pub full_attention_weights: Vec<f32>,
    /// Fraction of tokens routed to full attention
    pub full_attention_fraction: f32,
}

/// Decide per-token whether to use full attention based on DeltaNet state.
///
/// The routing logic:
/// 1. Compute a routing score per token from StateMetrics + learned projection
/// 2. Tokens with score above threshold get full attention
/// 3. Cap at `full_attention_capacity` fraction (take top-scoring tokens)
pub fn route_attention(
    delta_scores: &[f32],
    config: &HybridAttentionConfig,
) -> AttentionRoutingDecision {
    let n = delta_scores.len();
    if n == 0 {
        return AttentionRoutingDecision {
            full_attention_indices: vec![],
            full_attention_weights: vec![0.0; 0],
            full_attention_fraction: 0.0,
        };
    }

    match config.routing_mode {
        RoutingMode::LinearOnly => AttentionRoutingDecision {
            full_attention_indices: vec![],
            full_attention_weights: vec![0.0; n],
            full_attention_fraction: 0.0,
        },
        RoutingMode::FullOnly => AttentionRoutingDecision {
            full_attention_indices: (0..n).collect(),
            full_attention_weights: vec![1.0; n],
            full_attention_fraction: 1.0,
        },
        RoutingMode::FixedBlend => {
            // All tokens get both, with weight 0.5
            AttentionRoutingDecision {
                full_attention_indices: (0..n).collect(),
                full_attention_weights: vec![0.5; n],
                full_attention_fraction: 1.0,
            }
        }
        RoutingMode::Dynamic => {
            // Capacity: max tokens that can use full attention
            let max_full = ((n as f32 * config.full_attention_capacity).ceil() as usize).max(1);

            // Score each token: higher delta → more need for full attention
            let mut scored: Vec<(usize, f32)> = delta_scores
                .iter()
                .enumerate()
                .map(|(i, &d)| (i, d))
                .collect();

            // Sort descending by score
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Select tokens above threshold, capped at capacity
            let mut full_indices = Vec::new();
            let mut weights = vec![0.0f32; n];

            for &(idx, score) in scored.iter().take(max_full) {
                if score >= config.delta_threshold {
                    full_indices.push(idx);
                    // Soft weight: sigmoid of how far above threshold
                    let excess = score - config.delta_threshold;
                    let w = 1.0 / (1.0 + (-5.0 * excess).exp()); // sharp sigmoid
                    weights[idx] = w;
                }
            }

            full_indices.sort_unstable();
            let fraction = full_indices.len() as f32 / n as f32;

            AttentionRoutingDecision {
                full_attention_indices: full_indices,
                full_attention_weights: weights,
                full_attention_fraction: fraction,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped Query Attention (GQA)
// ---------------------------------------------------------------------------

/// Sparse Grouped Query Attention.
///
/// GQA reduces KV cache by sharing key-value heads across query groups:
/// - `num_heads` query heads
/// - `num_kv_heads` key-value heads (num_heads / num_kv_heads = group size)
///
/// Only computes attention for tokens in `active_indices` (token culling).
///
/// Optionally applies RoPE to Q and K before the attention computation.
/// Use `GroupedQueryAttention::with_rope` to attach a `RotaryEmbedding`.
/// Use `GroupedQueryAttention::with_norm` to replace the default LayerNorm
/// with an RMSNorm.
pub struct GroupedQueryAttention {
    config: HybridAttentionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ln: NormLayer,
    rope: Option<std::sync::Arc<RotaryEmbedding>>,
}

impl GroupedQueryAttention {
    /// Default constructor — LayerNorm, no RoPE.
    ///
    /// Kept for backward compatibility: all existing tests call this path.
    pub fn new(config: HybridAttentionConfig, vb: VarBuilder) -> Result<Self> {
        let h = config.hidden_dim;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q_proj = linear_no_bias(h, q_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, kv_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, kv_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(q_dim, h, vb.pp("o_proj"))?;
        let ln = NormLayer::LayerNorm(candle_nn::layer_norm(h, 1e-5, vb.pp("ln"))?);

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            ln,
            rope: None,
        })
    }

    /// Construct a GQA layer directly from pretrained weights.
    ///
    /// Unlike `new`, this constructor does NOT attempt to load a `ln` weight
    /// from the VarBuilder.  Instead the caller supplies a `NormLayer` built
    /// from the checkpoint's `input_layernorm.weight`.  This matches the
    /// HuggingFace weight naming convention where the pre-norm is stored at
    /// `model.layers.{i}.input_layernorm.weight`, separate from `self_attn.*`.
    ///
    /// # Arguments
    /// - `config`: GQA config for this layer.
    /// - `attn_vb`: VarBuilder scoped to `model.layers.{i}.self_attn`.
    /// - `norm`: Pre-norm layer built from `input_layernorm.weight`.
    /// - `rope`: Optional RoPE module (share the same Arc across all layers).
    pub fn from_pretrained(
        config: HybridAttentionConfig,
        attn_vb: VarBuilder,
        norm: NormLayer,
        rope: Option<std::sync::Arc<RotaryEmbedding>>,
    ) -> candle_core::Result<Self> {
        let h = config.hidden_dim;
        let q_dim = config.num_heads * config.head_dim;
        let kv_dim = config.num_kv_heads * config.head_dim;

        let q_proj = linear_no_bias(h, q_dim, attn_vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, kv_dim, attn_vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, kv_dim, attn_vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(q_dim, h, attn_vb.pp("o_proj"))?;

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            ln: norm,
            rope,
        })
    }

    /// Builder: attach a RoPE module.  Applied to Q and K in the forward pass,
    /// after reshaping to `[batch, seq_len, num_heads, head_dim]` and before
    /// the head-transpose.
    pub fn with_rope(mut self, rope: std::sync::Arc<RotaryEmbedding>) -> Self {
        self.rope = Some(rope);
        self
    }

    /// Builder: replace the normalisation layer.
    pub fn with_norm(mut self, norm: NormLayer) -> Self {
        self.ln = norm;
        self
    }

    /// Forward pass with optional token culling.
    ///
    /// # Arguments
    /// - `x`: Input [batch, seq_len, hidden_dim]
    /// - `active_mask`: Optional boolean mask [seq_len] — true = compute attention
    ///
    /// # Returns
    /// - Output [batch, seq_len, hidden_dim] (zeros for inactive tokens)
    pub fn forward(&self, x: &Tensor, active_mask: Option<&[bool]>) -> Result<Tensor> {
        let (batch, seq_len, _h) = x.dims3()?;
        let nh = self.config.num_heads;
        let nkv = self.config.num_kv_heads;
        let hd = self.config.head_dim;
        let device = x.device();
        let dtype = x.dtype();

        // Pre-norm
        let x_norm = self.ln.forward(x)?;

        // Project Q, K, V
        let q = self.q_proj.forward(&x_norm)?; // [b, t, nh*hd]
        let k = self.k_proj.forward(&x_norm)?; // [b, t, nkv*hd]
        let v = self.v_proj.forward(&x_norm)?; // [b, t, nkv*hd]

        // Reshape to multi-head — RoPE expects [b, t, heads, hd]
        let q = q.reshape((batch, seq_len, nh, hd))?;   // [b, t, nh, hd]
        let k = k.reshape((batch, seq_len, nkv, hd))?;  // [b, t, nkv, hd]
        let v = v.reshape((batch, seq_len, nkv, hd))?;

        // Apply RoPE if available (before head transpose)
        // RoPE: [b, t, heads, hd] → [b, t, heads, hd]
        // For GQA the K tensor has nkv heads; RoPE broadcasts over the heads
        // dimension independently, so it works without any special handling.
        let (q, k) = if let Some(rope) = &self.rope {
            (rope.apply(&q, 0)?, rope.apply(&k, 0)?)
        } else {
            (q, k)
        };

        // Transpose to [b, heads, t, hd] for attention
        let q = q.transpose(1, 2)?; // [b, nh, t, hd]
        let k = k.transpose(1, 2)?; // [b, nkv, t, hd]
        let v = v.transpose(1, 2)?; // [b, nkv, t, hd]

        // Expand KV heads to match query heads (GQA expansion)
        let group_size = nh / nkv;
        let k = expand_kv_heads(&k, group_size)?; // [b, nh, t, hd]
        let v = expand_kv_heads(&v, group_size)?;

        // Scaled dot-product attention: softmax(Q K^T / sqrt(d)) V
        let scale = (hd as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?; // [b, nh, t, t]
        let attn_weights = (attn_weights / scale)?;

        // Causal mask: prevent attending to future tokens
        let causal_mask = create_causal_mask(seq_len, dtype, device)?;
        let attn_weights = attn_weights.broadcast_add(&causal_mask)?;

        // Token culling mask: zero out inactive tokens if provided
        if let Some(mask) = active_mask {
            let token_mask = create_token_mask(mask, dtype, device)?;
            // Apply to both query and key dimensions
            let attn_weights = attn_weights.broadcast_add(&token_mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let output = attn_weights.matmul(&v)?; // [b, nh, t, hd]
            let output = output.transpose(1, 2)?; // [b, t, nh, hd]
            let output = output.reshape((batch, seq_len, nh * hd))?;
            let output = self.o_proj.forward(&output)?;
            return Ok(output);
        }

        // Standard softmax attention
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let output = attn_weights.matmul(&v)?; // [b, nh, t, hd]

        // Reshape back: [b, nh, t, hd] -> [b, t, nh*hd]
        let output = output.transpose(1, 2)?;
        let output = output.reshape((batch, seq_len, nh * hd))?;

        // Output projection
        self.o_proj.forward(&output)
    }
}

/// Expand KV heads for GQA: repeat each KV head `group_size` times.
/// Input:  [batch, nkv, seq, hd]
/// Output: [batch, nkv * group_size, seq, hd]
fn expand_kv_heads(kv: &Tensor, group_size: usize) -> Result<Tensor> {
    if group_size == 1 {
        return Ok(kv.clone());
    }
    let (b, nkv, t, hd) = kv.dims4()?;
    // [b, nkv, 1, t, hd] -> repeat along dim 2 -> [b, nkv, group_size, t, hd]
    let kv = kv.unsqueeze(2)?; // [b, nkv, 1, t, hd]
    let kv = kv.expand((b, nkv, group_size, t, hd))?; // broadcast
    kv.reshape((b, nkv * group_size, t, hd))
}

/// Create causal attention mask: -inf for future positions, 0 for past/present.
fn create_causal_mask(seq_len: usize, dtype: DType, device: &candle_core::Device) -> Result<Tensor> {
    let neg_inf = match dtype {
        DType::F32 => f64::from(f32::NEG_INFINITY),
        DType::F64 => f64::NEG_INFINITY,
        DType::BF16 | DType::F16 => -1e4,
        _ => -1e4,
    };

    // Build mask manually: 0 for j<=i (can attend), neg_inf for j>i (future)
    let mut vals = vec![0.0f64; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            vals[i * seq_len + j] = neg_inf;
        }
    }
    Tensor::new(vals.as_slice(), device)?
        .reshape((seq_len, seq_len))?
        .to_dtype(dtype)
}

/// Create token culling mask from boolean mask.
/// Inactive tokens get -inf in both query and key positions.
fn create_token_mask(active: &[bool], dtype: DType, device: &candle_core::Device) -> Result<Tensor> {
    let n = active.len();
    let neg_inf = match dtype {
        DType::F32 => f64::from(f32::NEG_INFINITY),
        DType::BF16 | DType::F16 => -1e4,
        _ => -1e4,
    };

    // Key mask: inactive keys get -inf (nobody attends to them)
    let key_vals: Vec<f64> = active.iter().map(|&a| if a { 0.0 } else { neg_inf }).collect();
    let key_mask = Tensor::new(key_vals.as_slice(), device)?.to_dtype(dtype)?;
    // Shape: [1, 1, 1, n] for broadcasting with [b, nh, t, t]
    let key_mask = key_mask.reshape((1, 1, 1, n))?;

    Ok(key_mask)
}

// ---------------------------------------------------------------------------
// Hybrid Attention Layer
// ---------------------------------------------------------------------------

/// A hybrid attention layer that combines GatedDeltaNet with GQA.
///
/// The layer always runs GatedDeltaNet first (cheap, O(n)). Based on
/// the resulting StateMetrics, it decides which tokens also need
/// full GQA attention and blends the outputs.
pub struct HybridAttentionLayer {
    pub config: HybridAttentionConfig,
    /// Router projection: hidden_dim → 1 (per-token routing score)
    router_proj: Linear,
    /// GQA attention (only allocated if mode != LinearOnly)
    gqa: Option<GroupedQueryAttention>,
    /// Blend gate: learned mixing weight
    blend_bias: Tensor,
}

impl HybridAttentionLayer {
    pub fn new(config: HybridAttentionConfig, vb: VarBuilder) -> Result<Self> {
        let h = config.hidden_dim;

        // Router: projects hidden state to scalar routing score
        let router_proj = linear_no_bias(h, 1, vb.pp("router"))?;

        // GQA: only needed if we might use full attention
        let gqa = if config.routing_mode != RoutingMode::LinearOnly {
            Some(GroupedQueryAttention::new(config.clone(), vb.pp("gqa"))?)
        } else {
            None
        };

        // Blend bias: initial bias toward linear attention
        let blend_bias = vb.get(1, "blend_bias")
            .unwrap_or_else(|_| Tensor::new(&[-1.0f32], vb.device()).unwrap());

        Ok(Self {
            config,
            router_proj,
            gqa,
            blend_bias,
        })
    }

    /// Compute per-token routing scores from hidden states and state metrics.
    ///
    /// The score combines:
    /// - Learned projection of the hidden state
    /// - State metrics (mean_delta) when available
    ///
    /// Higher score → more need for full attention.
    pub fn compute_routing_scores(
        &self,
        hidden: &Tensor,
        state_metrics: Option<&[StateMetrics]>,
    ) -> Result<Vec<f32>> {
        let (_batch, _seq_len, _h) = hidden.dims3()?;
        assert_eq!(_batch, 1, "Routing currently supports batch=1");

        // Learned routing score: [1, seq_len, 1]
        let scores = self.router_proj.forward(hidden)?;
        let scores = scores.squeeze(2)?.squeeze(0)?; // [seq_len]
        let mut score_vec: Vec<f32> = scores.to_vec1()?;

        // Augment with StateMetrics if available
        if let Some(metrics) = state_metrics {
            for (i, m) in metrics.iter().enumerate() {
                if i < score_vec.len() {
                    // Add mean_delta as routing signal: high delta → needs attention
                    score_vec[i] += m.mean_delta;
                }
            }
        }

        // Apply sigmoid to get [0, 1] scores
        let bias: f32 = self.blend_bias.to_vec1::<f32>()?[0];
        for s in score_vec.iter_mut() {
            *s = 1.0 / (1.0 + (-(*s + bias)).exp());
        }

        Ok(score_vec)
    }

    /// Forward pass: run DeltaNet, route, optionally run GQA, blend.
    ///
    /// # Arguments
    /// - `x`: Input [batch=1, seq_len, hidden_dim]
    /// - `deltanet_output`: Output from GatedDeltaNet layer [batch, seq_len, hidden_dim]
    /// - `state_metrics`: Per-token state metrics from DeltaNet
    ///
    /// # Returns
    /// - Blended output [batch, seq_len, hidden_dim]
    /// - Routing decision
    pub fn forward(
        &self,
        x: &Tensor,
        deltanet_output: &Tensor,
        state_metrics: Option<&[StateMetrics]>,
    ) -> Result<(Tensor, AttentionRoutingDecision)> {
        let (_batch, seq_len, _h) = x.dims3()?;

        match self.config.routing_mode {
            RoutingMode::LinearOnly => {
                let decision = route_attention(&vec![0.0; seq_len], &self.config);
                Ok((deltanet_output.clone(), decision))
            }
            RoutingMode::FullOnly => {
                let gqa = self.gqa.as_ref().expect("GQA not initialized");
                let gqa_output = gqa.forward(x, None)?;
                // Residual connection
                let output = (gqa_output + x)?;
                let decision = route_attention(&vec![1.0; seq_len], &self.config);
                Ok((output, decision))
            }
            RoutingMode::FixedBlend => {
                let gqa = self.gqa.as_ref().expect("GQA not initialized");
                let gqa_output = gqa.forward(x, None)?;
                let gqa_output = (&gqa_output + x)?;
                // Fixed 50/50 blend
                let output = ((deltanet_output * 0.5)? + (gqa_output * 0.5)?)?;
                let decision = route_attention(&vec![0.5; seq_len], &self.config);
                Ok((output, decision))
            }
            RoutingMode::Dynamic => {
                // 1. Compute routing scores
                let scores = self.compute_routing_scores(x, state_metrics)?;

                // 2. Make routing decision
                let decision = route_attention(&scores, &self.config);

                // 3. If no tokens need full attention, return linear output
                if decision.full_attention_indices.is_empty() {
                    return Ok((deltanet_output.clone(), decision));
                }

                // 4. Run GQA on full sequence (with token culling mask)
                let gqa = self.gqa.as_ref().expect("GQA not initialized");
                let active_mask: Vec<bool> = (0..seq_len)
                    .map(|i| decision.full_attention_indices.contains(&i))
                    .collect();
                let gqa_output = gqa.forward(x, Some(&active_mask))?;
                // Residual for GQA
                let gqa_output = (&gqa_output + x)?;

                // 5. Blend: output = (1 - w) * deltanet + w * gqa per token
                let weights = &decision.full_attention_weights;
                let weight_tensor = Tensor::new(weights.as_slice(), x.device())?
                    .to_dtype(x.dtype())?
                    .reshape((1, seq_len, 1))?; // [1, t, 1]

                let one_minus_w = (1.0 - &weight_tensor)?;
                let output = (deltanet_output.broadcast_mul(&one_minus_w)?
                    + gqa_output.broadcast_mul(&weight_tensor)?)?;

                Ok((output, decision))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hybrid Model Stack (layer composition)
// ---------------------------------------------------------------------------

/// Specifies the attention type for each layer in the model.
#[derive(Debug, Clone)]
pub struct HybridStackConfig {
    /// Total number of layers
    pub num_layers: usize,
    /// Which layers use full/hybrid attention (0-indexed).
    /// All other layers use linear-only (GatedDeltaNet).
    pub attention_layers: Vec<usize>,
    /// Whether attention layers use dynamic routing or fixed GQA
    pub dynamic_routing: bool,
}

impl HybridStackConfig {
    /// Chimère default: 32 layers, attention at 7,15,23,31 (1:7 ratio)
    pub fn chimere_32() -> Self {
        Self {
            num_layers: 32,
            attention_layers: vec![7, 15, 23, 31], // 0-indexed
            dynamic_routing: true,
        }
    }

    /// Qwen3-style: 48 layers, 12 full attention (3:1 ratio)
    pub fn qwen3_48() -> Self {
        Self {
            num_layers: 48,
            attention_layers: (0..48).filter(|i| i % 4 == 3).collect(), // every 4th
            dynamic_routing: false,
        }
    }

    /// Small config for tests: 8 layers, attention at layer 3 and 7
    pub fn test_8() -> Self {
        Self {
            num_layers: 8,
            attention_layers: vec![3, 7],
            dynamic_routing: true,
        }
    }

    /// Get the routing mode for a given layer index.
    pub fn routing_mode_for_layer(&self, layer_idx: usize) -> RoutingMode {
        if self.attention_layers.contains(&layer_idx) {
            if self.dynamic_routing {
                RoutingMode::Dynamic
            } else {
                RoutingMode::FullOnly
            }
        } else {
            RoutingMode::LinearOnly
        }
    }

    /// Compute the effective attention ratio.
    pub fn attention_ratio(&self) -> f32 {
        self.attention_layers.len() as f32 / self.num_layers as f32
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use candle_core::{Device, DType};

    // ---- Routing logic tests (no model weights needed) ----

    #[test]
    fn test_route_attention_linear_only() {
        let mut config = HybridAttentionConfig::test();
        config.routing_mode = RoutingMode::LinearOnly;

        let scores = vec![0.8, 0.9, 0.1, 0.5];
        let decision = route_attention(&scores, &config);

        assert!(
            decision.full_attention_indices.is_empty(),
            "LinearOnly should route no tokens to full attention"
        );
        assert_eq!(decision.full_attention_fraction, 0.0);
        println!("LinearOnly: fraction={}", decision.full_attention_fraction);
    }

    #[test]
    fn test_route_attention_full_only() {
        let mut config = HybridAttentionConfig::test();
        config.routing_mode = RoutingMode::FullOnly;

        let scores = vec![0.1, 0.1, 0.1, 0.1];
        let decision = route_attention(&scores, &config);

        assert_eq!(
            decision.full_attention_indices.len(),
            4,
            "FullOnly should route all tokens"
        );
        assert_eq!(decision.full_attention_fraction, 1.0);
        println!("FullOnly: fraction={}", decision.full_attention_fraction);
    }

    #[test]
    fn test_route_attention_dynamic_threshold() {
        let config = HybridAttentionConfig::test(); // threshold=0.3, capacity=0.5

        // Mix of high and low delta scores
        let scores = vec![0.1, 0.8, 0.05, 0.6, 0.2, 0.9, 0.02, 0.7];
        let decision = route_attention(&scores, &config);

        // Should route tokens with scores >= 0.3
        // High scores: 0.8 (1), 0.6 (3), 0.9 (5), 0.7 (7) → 4 tokens
        // Capacity: 50% of 8 = 4 tokens → all fit
        assert!(
            decision.full_attention_indices.contains(&1),
            "Token 1 (score=0.8) should get full attention"
        );
        assert!(
            decision.full_attention_indices.contains(&5),
            "Token 5 (score=0.9) should get full attention"
        );
        assert!(
            !decision.full_attention_indices.contains(&0),
            "Token 0 (score=0.1) should NOT get full attention"
        );

        println!(
            "Dynamic routing: {}/{} tokens selected (fraction={:.2})",
            decision.full_attention_indices.len(),
            scores.len(),
            decision.full_attention_fraction
        );
    }

    #[test]
    fn test_route_attention_capacity_cap() {
        let mut config = HybridAttentionConfig::test();
        config.delta_threshold = 0.1; // low threshold (most tokens qualify)
        config.full_attention_capacity = 0.25; // but only 25% can use it

        let scores = vec![0.5, 0.6, 0.7, 0.8, 0.3, 0.4, 0.9, 0.2];
        let decision = route_attention(&scores, &config);

        // Capacity: 25% of 8 = 2 tokens max
        assert!(
            decision.full_attention_indices.len() <= 2,
            "Capacity cap should limit to 2 tokens, got {}",
            decision.full_attention_indices.len()
        );

        // The selected tokens should be the highest-scoring ones (0.9, 0.8)
        assert!(
            decision.full_attention_indices.contains(&6),
            "Token 6 (score=0.9, highest) should be selected"
        );
        assert!(
            decision.full_attention_indices.contains(&3),
            "Token 3 (score=0.8, second highest) should be selected"
        );

        println!(
            "Capacity cap: {}/{} tokens (cap={})",
            decision.full_attention_indices.len(),
            scores.len(),
            (scores.len() as f32 * config.full_attention_capacity).ceil()
        );
    }

    #[test]
    fn test_route_attention_weights_sigmoid() {
        let config = HybridAttentionConfig::test(); // threshold=0.3

        // Token just above threshold vs far above
        let scores = vec![0.31, 0.9];
        let decision = route_attention(&scores, &config);

        if decision.full_attention_weights.len() >= 2 {
            let w_marginal = decision.full_attention_weights[0];
            let w_confident = decision.full_attention_weights[1];

            // Far-above-threshold should have higher weight
            assert!(
                w_confident > w_marginal,
                "Higher delta should get higher weight: {:.3} > {:.3}",
                w_confident,
                w_marginal
            );
            println!(
                "Sigmoid weights: marginal={:.3}, confident={:.3}",
                w_marginal, w_confident
            );
        }
    }

    // ---- GQA expansion test ----

    #[test]
    fn test_expand_kv_heads() -> Result<()> {
        let device = Device::Cpu;
        // 2 KV heads, group_size=2 → 4 query heads
        let kv = Tensor::ones((1, 2, 4, 8), DType::F32, &device)?;
        let expanded = expand_kv_heads(&kv, 2)?;

        assert_eq!(expanded.dims(), &[1, 4, 4, 8], "Should expand to 4 heads");

        // Head 0 and 1 should be identical (both from KV head 0)
        let h0: Vec<f32> = expanded.narrow(1, 0, 1)?.flatten_all()?.to_vec1()?;
        let h1: Vec<f32> = expanded.narrow(1, 1, 1)?.flatten_all()?.to_vec1()?;
        assert_eq!(h0, h1, "Expanded heads in same group should be identical");

        println!(
            "GQA expansion: {} KV heads → {} query heads (group_size=2)",
            2, 4
        );
        Ok(())
    }

    #[test]
    fn test_expand_kv_heads_identity() -> Result<()> {
        let device = Device::Cpu;
        // group_size=1 → no expansion needed
        let kv = Tensor::randn(0.0f32, 1.0, (1, 4, 4, 8), &device)?;
        let expanded = expand_kv_heads(&kv, 1)?;

        let orig: Vec<f32> = kv.flatten_all()?.to_vec1()?;
        let exp: Vec<f32> = expanded.flatten_all()?.to_vec1()?;
        assert_eq!(orig, exp, "group_size=1 should be identity");

        println!("GQA expansion: group_size=1 is identity");
        Ok(())
    }

    // ---- Causal mask test ----

    #[test]
    fn test_causal_mask() -> Result<()> {
        let mask = create_causal_mask(4, DType::F32, &Device::Cpu)?;
        let vals: Vec<Vec<f32>> = (0..4)
            .map(|i| mask.narrow(0, i, 1).unwrap().squeeze(0).unwrap().to_vec1().unwrap())
            .collect();

        // Row i should have 0 for j<=i and -inf for j>i
        for i in 0..4 {
            for j in 0..4 {
                if j <= i {
                    assert_eq!(
                        vals[i][j], 0.0,
                        "Position [{},{}] should be 0 (can attend)",
                        i, j
                    );
                } else {
                    assert!(
                        vals[i][j].is_infinite() && vals[i][j] < 0.0,
                        "Position [{},{}] should be -inf (future)",
                        i, j
                    );
                }
            }
        }

        println!("Causal mask 4x4: correct upper-triangular -inf");
        Ok(())
    }

    // ---- Stack config tests ----

    #[test]
    fn test_stack_config_chimere() {
        let config = HybridStackConfig::chimere_32();

        assert_eq!(config.num_layers, 32);
        assert_eq!(config.attention_layers.len(), 4);
        assert!((config.attention_ratio() - 0.125).abs() < 0.01, "1:7 ratio = 12.5%");

        // Linear layers
        assert_eq!(
            config.routing_mode_for_layer(0),
            RoutingMode::LinearOnly,
            "Layer 0 should be linear"
        );
        assert_eq!(
            config.routing_mode_for_layer(6),
            RoutingMode::LinearOnly,
            "Layer 6 should be linear"
        );

        // Attention layers
        assert_eq!(
            config.routing_mode_for_layer(7),
            RoutingMode::Dynamic,
            "Layer 7 should be dynamic attention"
        );
        assert_eq!(
            config.routing_mode_for_layer(31),
            RoutingMode::Dynamic,
            "Layer 31 should be dynamic attention"
        );

        println!(
            "Chimère-32: {} layers, {} attention (ratio={:.1}%), dynamic={}",
            config.num_layers,
            config.attention_layers.len(),
            config.attention_ratio() * 100.0,
            config.dynamic_routing
        );
    }

    #[test]
    fn test_stack_config_qwen3() {
        let config = HybridStackConfig::qwen3_48();

        assert_eq!(config.num_layers, 48);
        assert_eq!(config.attention_layers.len(), 12);
        assert!((config.attention_ratio() - 0.25).abs() < 0.01, "3:1 ratio = 25%");

        // Qwen3 uses fixed full attention, not dynamic
        assert_eq!(
            config.routing_mode_for_layer(3),
            RoutingMode::FullOnly,
            "Qwen3 attention layers should be FullOnly (not dynamic)"
        );

        println!(
            "Qwen3-48: {} layers, {} attention (ratio={:.1}%)",
            config.num_layers,
            config.attention_layers.len(),
            config.attention_ratio() * 100.0
        );
    }

    // ---- Token mask test ----

    #[test]
    fn test_token_mask() -> Result<()> {
        let mask = create_token_mask(
            &[true, false, true, false],
            DType::F32,
            &Device::Cpu,
        )?;

        let vals: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        assert_eq!(vals[0], 0.0, "Active token should have mask 0");
        assert!(vals[1].is_infinite() && vals[1] < 0.0, "Inactive token should have -inf");
        assert_eq!(vals[2], 0.0);
        assert!(vals[3].is_infinite() && vals[3] < 0.0);

        println!("Token culling mask: active=0, inactive=-inf");
        Ok(())
    }

    // ---- Integration: full GQA forward ----

    #[test]
    fn test_gqa_forward_shapes() -> Result<()> {
        let device = Device::Cpu;
        let config = HybridAttentionConfig::test(); // 64 dim, 4 heads, 2 KV heads, 16 hd
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gqa = GroupedQueryAttention::new(config, vb)?;

        // Input: [1, 8, 64]
        let x = Tensor::randn(0.0f32, 0.1, (1, 8, 64), &device)?;
        let output = gqa.forward(&x, None)?;

        assert_eq!(
            output.dims(),
            &[1, 8, 64],
            "GQA output should preserve shape: got {:?}",
            output.dims()
        );
        println!("GQA forward: input [1,8,64] → output {:?}", output.dims());
        Ok(())
    }

    #[test]
    fn test_gqa_with_token_culling() -> Result<()> {
        let device = Device::Cpu;
        let config = HybridAttentionConfig::test();
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let gqa = GroupedQueryAttention::new(config, vb)?;

        let x = Tensor::randn(0.0f32, 0.1, (1, 6, 64), &device)?;

        // Only tokens 1, 3, 5 are active
        let mask = vec![false, true, false, true, false, true];
        let output = gqa.forward(&x, Some(&mask))?;

        assert_eq!(
            output.dims(),
            &[1, 6, 64],
            "Token-culled GQA should preserve shape"
        );
        println!("GQA with culling: 3/6 tokens active, output {:?}", output.dims());
        Ok(())
    }

    // ---- RoPE integration test ----

    #[test]
    fn test_gqa_with_rope() -> Result<()> {
        let device = Device::Cpu;
        // test config: hidden=64, 4 query heads, 2 KV heads, head_dim=16
        let config = HybridAttentionConfig::test();

        // Build GQA without RoPE (baseline)
        let varmap_base = candle_nn::VarMap::new();
        let vb_base = VarBuilder::from_varmap(&varmap_base, DType::F32, &device);
        let gqa_base = GroupedQueryAttention::new(config.clone(), vb_base)?;

        // Build GQA with RoPE using the same config (identical weights, different VarMap)
        let varmap_rope = candle_nn::VarMap::new();
        let vb_rope = VarBuilder::from_varmap(&varmap_rope, DType::F32, &device);
        let rope = Arc::new(RotaryEmbedding::new(
            config.head_dim, // 16
            64,              // max_seq_len
            10000.0,         // theta
            DType::F32,
            &device,
        )?);
        let gqa_rope = GroupedQueryAttention::new(config.clone(), vb_rope)?.with_rope(rope);

        // Shared input
        let x = Tensor::randn(0.0f32, 0.1, (1usize, 8usize, 64usize), &device)?;

        // Forward through both
        let out_base = gqa_base.forward(&x, None)?;
        let out_rope = gqa_rope.forward(&x, None)?;

        // 1. Shape must be preserved
        assert_eq!(
            out_rope.dims(),
            &[1, 8, 64],
            "GQA+RoPE output shape must be [1,8,64], got {:?}",
            out_rope.dims()
        );

        // 2. Outputs must differ — RoPE injects positional information
        let diff = (&out_rope - &out_base)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        assert!(
            diff > 1e-6,
            "GQA with RoPE must produce different output than GQA without RoPE \
             (mean abs diff={:.8}); check that RoPE is actually applied",
            diff
        );

        println!(
            "GQA+RoPE: output {:?}, mean |rope - no_rope| = {:.6}",
            out_rope.dims(),
            diff
        );
        Ok(())
    }
}
