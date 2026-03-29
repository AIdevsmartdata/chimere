//! # Gated DeltaNet — Chimère Engine
//!
//! Linear attention with delta rule and gating, as used in Qwen3-Coder-Next
//! (36/48 layers) and Kimi K2.5. This is the core attention mechanism for
//! Chimère's hybrid attention architecture.
//!
//! ## The Delta Rule as Associative Memory
//!
//! The state matrix S ∈ ℝ^(d_v × d_k) per head acts as an associative memory:
//! - **Recall:** o = S · q  (query the memory)
//! - **Prediction error:** δ = v - S · k  (what we observed vs what memory predicted)
//! - **Update:** S' = α·S + β · δ ⊗ k  (correct memory with gated error)
//!
//! This is mathematically equivalent to an Engram codebook with MDL updates:
//! the state only changes when the prediction error is non-zero, achieving
//! minimum description length compression of the input sequence.
//!
//! ## Architecture Reference (Qwen3-Coder-Next)
//!
//! - `state_size` (d_k = d_v): 128
//! - `num_heads`: 32
//! - `conv_kernel`: 4 (causal short convolution before gating)
//! - Gates: scalar per head (α, β) + vector per head (g)
//! - Hidden dim: num_heads × state_size = 4096

pub mod activations;
pub mod agent_scheduler;
pub mod block_diffusion;
pub mod block_generate;
pub mod candle_counter;
pub mod debug_utils;
pub mod config;
pub mod deltanet_kernel;
pub mod engram;
pub mod kernels;
pub mod engram_lookup;
pub mod entropy_router;
pub mod expert;
pub mod generate;
pub mod ggml_backend;
pub mod llama_backend;
pub mod gguf_loader;
pub mod hybrid_attention;
pub mod moe_forward;
pub mod moe_router;
pub mod mtp_scheduler;
pub mod prefill;
pub mod qwen35_model;
pub mod raw_forward;
pub mod raw_weights;
pub mod rope;
pub mod scratch_pool;
pub mod state;
pub mod trace;
pub mod turboquant;
pub mod weight_loader;

#[cfg(feature = "server")]
pub mod server;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Flexible normalization layer: LayerNorm (existing tests) or RMSNorm (real models).
pub enum NormLayer {
    LayerNorm(candle_nn::LayerNorm),
    RmsNorm { weight: Tensor, eps: f64 },
}

impl NormLayer {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            NormLayer::LayerNorm(ln) => Module::forward(ln, x),
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

/// SiLU activation: x * sigmoid(x)
fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = sigmoid(x)?;
    x.mul(&sig)
}

/// Sigmoid activation: 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let one = Tensor::ones(x.shape(), x.dtype(), x.device())?;
    let denom = (&one + &exp_neg)?;
    one.div(&denom)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Gate granularity: scalar (Qwen3) or channel-wise (Kimi K2.5)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateMode {
    /// One scalar gate per head (Qwen3-Coder-Next style)
    Scalar,
    /// One gate value per dimension per head (Kimi KDA style, finer but costlier)
    ChannelWise,
}

/// Configuration for a single GatedDeltaNet layer.
#[derive(Debug, Clone)]
pub struct GatedDeltaNetConfig {
    /// Model hidden dimension (input/output size)
    pub hidden_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head (d_k = d_v), derived: hidden_dim / num_heads
    pub head_dim: usize,
    /// Causal convolution kernel size (0 = no conv)
    pub conv_kernel: usize,
    /// Gate mode: scalar (Qwen3) or channel-wise (Kimi)
    pub gate_mode: GateMode,
}

impl GatedDeltaNetConfig {
    /// Qwen3-Coder-Next default configuration
    pub fn qwen3() -> Self {
        Self {
            hidden_dim: 4096,
            num_heads: 32,
            head_dim: 128,
            conv_kernel: 4,
            gate_mode: GateMode::Scalar,
        }
    }

    /// Chimere default — channel-wise gates for finer control
    pub fn chimere() -> Self {
        Self {
            hidden_dim: 4096,
            num_heads: 32,
            head_dim: 128,
            conv_kernel: 4,
            gate_mode: GateMode::ChannelWise,
        }
    }

    /// Small config for unit tests
    pub fn test() -> Self {
        Self {
            hidden_dim: 64,
            num_heads: 4,
            head_dim: 16,
            conv_kernel: 4,
            gate_mode: GateMode::Scalar,
        }
    }

    /// Nanbeige4.1-3B configuration (for weight loading)
    pub fn nanbeige() -> Self {
        Self {
            hidden_dim: 2560,
            num_heads: 20,
            head_dim: 128,
            conv_kernel: 0, // no conv when loading from MHA
            gate_mode: GateMode::Scalar,
        }
    }

    fn gate_proj_dim(&self) -> usize {
        match self.gate_mode {
            GateMode::Scalar => self.num_heads,
            GateMode::ChannelWise => self.hidden_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Short Causal Convolution
// ---------------------------------------------------------------------------

/// Causal 1D depthwise convolution applied per-channel.
/// Provides local context (kernel_size token window) before gating.
/// This is the same mechanism used in Mamba and Qwen3-Next.
pub struct ShortConv1d {
    /// Weight tensor: [channels, 1, kernel_size]
    weight: Tensor,
    /// Bias tensor: [channels]
    bias: Tensor,
    kernel_size: usize,
}

impl ShortConv1d {
    pub fn new(channels: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((channels, 1, kernel_size), "weight")?;
        let bias = vb.get(channels, "bias")?;
        Ok(Self {
            weight,
            bias,
            kernel_size,
        })
    }

    /// Apply causal convolution: pad left, no future leakage.
    /// Input: [batch, seq_len, channels]
    /// Output: [batch, seq_len, channels]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        // Transpose to [batch, channels, seq_len] for conv1d
        let x = x.transpose(1, 2)?; // [b, c, t]

        // Causal padding: pad (kernel_size - 1) zeros on the left
        let pad = self.kernel_size - 1;
        if pad > 0 {
            let zeros = Tensor::zeros((b, c, pad), x.dtype(), x.device())?;
            let x = Tensor::cat(&[&zeros, &x], 2)?; // [b, c, t + pad]
            // Manual depthwise conv1d: reshape for grouped convolution
            // candle doesn't have native depthwise conv1d, so we do it manually
            let mut output = Vec::with_capacity(t);
            for i in 0..t {
                // Window: [b, c, kernel_size] starting at position i
                let window = x.narrow(2, i, self.kernel_size)?; // [b, c, kernel_size]
                // Element-wise multiply with weight [c, 1, kernel_size] -> broadcast
                let w = self.weight.squeeze(1)?; // [c, kernel_size]
                let prod = window.broadcast_mul(&w)?; // [b, c, kernel_size]
                let summed = prod.sum(2)?; // [b, c]
                let biased = summed.broadcast_add(&self.bias)?; // [b, c]
                output.push(biased);
            }
            let result = Tensor::stack(&output, 1)?; // [b, t, c]
            Ok(result)
        } else {
            // kernel_size = 1, just a pointwise transform
            let w = self.weight.squeeze(1)?.squeeze(1)?; // [c]
            let result = x.broadcast_mul(&w)?.broadcast_add(&self.bias)?;
            Ok(result.transpose(1, 2)?) // [b, t, c]
        }
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet Layer
// ---------------------------------------------------------------------------

/// A single Gated DeltaNet attention layer.
///
/// This layer maintains a persistent state matrix S per head that acts as
/// an associative memory. The delta rule updates S based on prediction error,
/// achieving minimum description length (MDL) encoding of the sequence.
pub struct GatedDeltaNetLayer {
    config: GatedDeltaNetConfig,

    // Linear projections: x -> Q, K, V
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,

    // Output projection: concat(heads) -> hidden_dim
    o_proj: Linear,

    // Gates: x -> alpha (decay), beta (update), g (output)
    alpha_proj: Linear, // -> scalar per head or channel-wise
    beta_proj: Linear,  // -> scalar per head or channel-wise
    gate_proj: Linear,  // -> hidden_dim (always channel-wise for output gate)

    // Short causal convolution on Q, K, V
    q_conv: Option<ShortConv1d>,
    k_conv: Option<ShortConv1d>,
    v_conv: Option<ShortConv1d>,

    // Normalization layer (pre-norm architecture): LayerNorm or RMSNorm
    ln: NormLayer,
}

impl GatedDeltaNetLayer {
    pub fn new(config: GatedDeltaNetConfig, vb: VarBuilder) -> Result<Self> {
        let h = config.hidden_dim;
        let gate_dim = config.gate_proj_dim();

        let q_proj = linear_no_bias(h, h, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, h, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, h, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(h, h, vb.pp("o_proj"))?;

        let alpha_proj = linear_no_bias(h, gate_dim, vb.pp("alpha_proj"))?;
        let beta_proj = linear_no_bias(h, gate_dim, vb.pp("beta_proj"))?;
        let gate_proj = linear_no_bias(h, h, vb.pp("gate_proj"))?;

        let (q_conv, k_conv, v_conv) = if config.conv_kernel > 0 {
            (
                Some(ShortConv1d::new(h, config.conv_kernel, vb.pp("q_conv"))?),
                Some(ShortConv1d::new(h, config.conv_kernel, vb.pp("k_conv"))?),
                Some(ShortConv1d::new(h, config.conv_kernel, vb.pp("v_conv"))?),
            )
        } else {
            (None, None, None)
        };

        let ln = NormLayer::LayerNorm(candle_nn::layer_norm(h, 1e-5, vb.pp("ln"))?);

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            alpha_proj,
            beta_proj,
            gate_proj,
            q_conv,
            k_conv,
            v_conv,
            ln,
        })
    }

    /// Replace the normalization layer (builder pattern).
    /// Allows switching from the default LayerNorm to RMSNorm for real model weight loading.
    pub fn with_norm(mut self, norm: NormLayer) -> Self {
        self.ln = norm;
        self
    }

    /// Construct a DeltaNet layer from pretrained MHA weights.
    ///
    /// K and V projections are expanded from `num_kv_heads` → `num_heads`
    /// using `expand_kv_weight` + symmetry-breaking noise, so the DeltaNet
    /// state matrix has full per-head resolution even when the source model
    /// uses fewer KV heads.
    ///
    /// Alpha, beta, and gate projections are freshly initialised with small
    /// random values (they have no pretrained equivalent).
    ///
    /// # Arguments
    /// - `config`: DeltaNet config for this layer (must have `conv_kernel = 0`).
    /// - `attn_vb`: VarBuilder scoped to `model.layers.{i}.self_attn`.
    /// - `norm`: Pre-norm layer (RmsNorm built from `input_layernorm.weight`).
    /// - `num_kv_heads`: Number of KV heads in the pretrained checkpoint.
    /// - `device`: Target device.
    /// - `dtype`: Target dtype.
    pub fn from_pretrained(
        config: GatedDeltaNetConfig,
        attn_vb: VarBuilder,
        norm: NormLayer,
        num_kv_heads: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        use crate::weight_loader::{add_symmetry_noise, expand_kv_weight};

        let h = config.hidden_dim;
        let nh = config.num_heads;
        let hd = config.head_dim;
        let gate_dim = config.gate_proj_dim();

        // Q and O: direct load (shape already matches full hidden_dim)
        let q_proj = linear_no_bias(h, h, attn_vb.pp("q_proj"))?;
        let o_proj = linear_no_bias(h, h, attn_vb.pp("o_proj"))?;

        // K and V: load at [num_kv_heads * head_dim, hidden_dim], then expand
        let kv_out_dim = num_kv_heads * hd;
        let k_weight_raw = attn_vb
            .pp("k_proj")
            .get_with_hints(&[kv_out_dim, h], "weight", Default::default())?;
        let k_weight = expand_kv_weight(&k_weight_raw, num_kv_heads, nh, hd)?;
        let k_weight = add_symmetry_noise(&k_weight, 0.01)?;
        let k_proj = Linear::new(k_weight, None);

        let v_weight_raw = attn_vb
            .pp("v_proj")
            .get_with_hints(&[kv_out_dim, h], "weight", Default::default())?;
        let v_weight = expand_kv_weight(&v_weight_raw, num_kv_heads, nh, hd)?;
        let v_weight = add_symmetry_noise(&v_weight, 0.01)?;
        let v_proj = Linear::new(v_weight, None);

        // Alpha / beta / gate: random init (no pretrained equivalent)
        // Beta initialised slightly negative so sigmoid(-1) ≈ 0.27 → conservative
        // state updates at the start of fine-tuning.
        let alpha_weight =
            Tensor::randn(0.0f32, 0.01, (gate_dim, h), device)?.to_dtype(dtype)?;
        let alpha_proj = Linear::new(alpha_weight, None);

        let beta_weight =
            Tensor::randn(-1.0f32, 0.01, (gate_dim, h), device)?.to_dtype(dtype)?;
        let beta_proj = Linear::new(beta_weight, None);

        let gate_weight =
            Tensor::randn(0.0f32, 0.02, (h, h), device)?.to_dtype(dtype)?;
        let gate_proj = Linear::new(gate_weight, None);

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            alpha_proj,
            beta_proj,
            gate_proj,
            q_conv: None,
            k_conv: None,
            v_conv: None,
            ln: norm,
        })
    }

    /// Forward pass over a sequence, updating state recurrently.
    ///
    /// # Arguments
    /// - `x`: Input tensor [batch, seq_len, hidden_dim]
    /// - `state`: Mutable state [batch, num_heads, head_dim, head_dim]
    ///            Pass None to initialize to zeros.
    ///
    /// # Returns
    /// - Output tensor [batch, seq_len, hidden_dim]
    /// - Updated state [batch, num_heads, head_dim, head_dim]
    pub fn forward(
        &self,
        x: &Tensor,
        state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _hidden) = x.dims3()?;
        let nh = self.config.num_heads;
        let hd = self.config.head_dim;
        let device = x.device();
        let dtype = x.dtype();

        // Pre-norm (LayerNorm or RMSNorm depending on configuration)
        let x_norm = self.ln.forward(x)?;

        // Project Q, K, V: [b, t, h] -> [b, t, h]
        let q = self.q_proj.forward(&x_norm)?;
        let k = self.k_proj.forward(&x_norm)?;
        let v = self.v_proj.forward(&x_norm)?;

        // Short convolution (causal, provides local context)
        let q = match &self.q_conv {
            Some(conv) => conv.forward(&q)?,
            None => q,
        };
        let k = match &self.k_conv {
            Some(conv) => conv.forward(&k)?,
            None => k,
        };
        let v = match &self.v_conv {
            Some(conv) => conv.forward(&v)?,
            None => v,
        };

        // Apply SiLU activation to Q, K after conv (standard in DeltaNet variants)
        let q = silu(&q)?;
        let k = silu(&k)?;

        // Compute gates from original (pre-conv) input
        let alpha_raw = self.alpha_proj.forward(&x_norm)?; // [b, t, gate_dim]
        let beta_raw = self.beta_proj.forward(&x_norm)?;   // [b, t, gate_dim]
        let g = sigmoid(&self.gate_proj.forward(&x_norm)?)?; // [b, t, h]

        // Sigmoid gates
        let alpha = sigmoid(&alpha_raw)?; // decay in (0, 1)
        let beta = sigmoid(&beta_raw)?;   // update in (0, 1)

        // Reshape to multi-head: [b, t, h] -> [b, t, nh, hd]
        let q = q.reshape((batch, seq_len, nh, hd))?;
        let k = k.reshape((batch, seq_len, nh, hd))?;
        let v = v.reshape((batch, seq_len, nh, hd))?;
        let g = g.reshape((batch, seq_len, nh, hd))?;

        // Reshape gates based on mode
        let (alpha, beta) = match self.config.gate_mode {
            GateMode::Scalar => {
                // [b, t, nh] -> [b, t, nh, 1] for broadcasting
                let alpha = alpha.reshape((batch, seq_len, nh, 1))?;
                let beta = beta.reshape((batch, seq_len, nh, 1))?;
                (alpha, beta)
            }
            GateMode::ChannelWise => {
                // [b, t, h] -> [b, t, nh, hd]
                let alpha = alpha.reshape((batch, seq_len, nh, hd))?;
                let beta = beta.reshape((batch, seq_len, nh, hd))?;
                (alpha, beta)
            }
        };

        // Initialize or use provided state: [b, nh, hd, hd]
        let mut s = match state {
            Some(s) => s.clone(),
            None => Tensor::zeros((batch, nh, hd, hd), dtype, device)?,
        };

        // L2-normalize keys for stable state updates
        let k = l2_normalize(&k, 3)?; // normalize along head_dim axis

        // === Recurrent delta rule loop ===
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract time step t: [b, nh, hd]
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
            let alpha_t = alpha.narrow(1, t, 1)?.squeeze(1)?; // [b, nh, 1] or [b, nh, hd]
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

            // --- Delta Rule ---
            //
            // 1. Predict: v_pred = S . k   (what does memory expect for this key?)
            let k_col = k_t.unsqueeze(3)?; // [b, nh, hd, 1]
            let v_pred = s.matmul(&k_col)?.squeeze(3)?; // [b, nh, hd]

            // 2. Error: delta = v_observed - v_predicted
            let delta = (&v_t - &v_pred)?;

            // 3. State decay: S = alpha . S
            let s_decayed = s.broadcast_mul(&alpha_t.unsqueeze(3)?)?;

            // 4. State update: S += beta . (delta outer k^T)
            let delta_scaled = delta.broadcast_mul(&beta_t)?;
            let delta_col = delta_scaled.unsqueeze(3)?; // [b, nh, hd, 1]
            let k_row = k_t.unsqueeze(2)?; // [b, nh, 1, hd]
            let update = delta_col.matmul(&k_row)?; // [b, nh, hd, hd]

            s = (&s_decayed + &update)?;

            // 5. Query: o = S . q (read from updated memory)
            let q_col = q_t.unsqueeze(3)?;
            let o_t = s.matmul(&q_col)?.squeeze(3)?;

            // 6. Output gate: o = g * o
            let o_t = (&g_t * &o_t)?;

            outputs.push(o_t);
        }

        // Stack outputs: [b, seq_len, nh, hd]
        let output = Tensor::stack(&outputs, 1)?;

        // Reshape back: [b, t, nh, hd] -> [b, t, hidden_dim]
        let output = output.reshape((batch, seq_len, nh * hd))?;

        // Output projection + residual
        let output = self.o_proj.forward(&output)?;
        let output = (output + x)?; // residual connection

        Ok((output, s))
    }
}

/// L2-normalize a tensor along a given dimension.
/// Works for any rank: 1D vectors, 2D matrices, 4D batched tensors.
fn l2_normalize(x: &Tensor, dim: usize) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(dim)?.sqrt()?;
    let eps = 1e-12f64;
    // Add epsilon to avoid division by zero, then divide
    let norm = norm.affine(1.0, eps)?;
    x.broadcast_div(&norm)
}

// ---------------------------------------------------------------------------
// State inspection (for Engram / entropy-router)
// ---------------------------------------------------------------------------

/// Metrics about the state matrix for monitoring convergence and entropy.
#[derive(Debug, Clone)]
pub struct StateMetrics {
    /// Frobenius norm of S (overall memory "fullness")
    pub frobenius_norm: f32,
    /// Mean absolute prediction error across last step
    pub mean_delta: f32,
    /// Effective rank of S (how much of the memory space is used)
    pub effective_rank: f32,
}

/// Compute state metrics for a single head's state matrix.
/// Useful for entropy-adaptive routing: low delta -> low entropy -> use linear.
pub fn compute_state_metrics(state: &Tensor) -> Result<StateMetrics> {
    // state: [hd, hd] for a single head
    let frob = state.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;

    // S^T S for trace computation
    let sts = state.t()?.matmul(state)?;

    // Manual trace: sum of diagonal elements
    let hd = sts.dim(0)?;
    let mut trace = 0.0f32;
    for i in 0..hd {
        let row = sts.narrow(0, i, 1)?;
        let val = row.narrow(1, i, 1)?;
        trace += val.sum_all()?.to_scalar::<f32>()?;
    }

    Ok(StateMetrics {
        frobenius_norm: frob,
        mean_delta: 0.0, // computed during forward pass, placeholder
        effective_rank: trace / (frob * frob + 1e-12),
    })
}

// ---------------------------------------------------------------------------
// Standalone delta rule (no projections, for testing and prototyping)
// ---------------------------------------------------------------------------

/// Pure delta rule state update, no projections or gates.
/// Use this for unit testing the core algorithm.
///
/// # Arguments
/// - `state`: [d_v, d_k] — current associative memory
/// - `key`: [d_k] — key to update
/// - `value`: [d_v] — observed value
/// - `alpha`: scalar — decay rate
/// - `beta`: scalar — update rate
///
/// # Returns
/// - Updated state [d_v, d_k]
/// - Prediction error [d_v]
pub fn delta_rule_step(
    state: &Tensor,
    key: &Tensor,
    value: &Tensor,
    alpha: f64,
    beta: f64,
) -> Result<(Tensor, Tensor)> {
    let _device = state.device();
    let _dtype = state.dtype();

    // Normalize key: compute L2 norm of 1D vector
    let k_norm = key.sqr()?.sum_all()?.sqrt()?;
    let k_norm_val: f32 = k_norm.to_scalar()?;
    let k = if k_norm_val > 1e-12 {
        (key / k_norm_val as f64)?
    } else {
        key.clone()
    };

    // Predict: v_pred = S . k
    let k_col = k.unsqueeze(1)?; // [d_k, 1]
    let v_pred = state.matmul(&k_col)?.squeeze(1)?; // [d_v]

    // Error
    let delta = (value - &v_pred)?;

    // Decay: S = alpha * S
    let s_decayed = (state * alpha)?;

    // Update: S += beta * (delta outer k^T)
    let delta_scaled = (&delta * beta)?;
    let delta_col = delta_scaled.unsqueeze(1)?; // [d_v, 1]
    let k_row = k.unsqueeze(0)?; // [1, d_k]
    let update = delta_col.matmul(&k_row)?; // [d_v, d_k]

    let s_new = (&s_decayed + &update)?;

    Ok((s_new, delta))
}

/// Query the state (associative recall).
/// Returns S . q (the memory's prediction for this query).
pub fn delta_rule_query(state: &Tensor, query: &Tensor) -> Result<Tensor> {
    // Normalize query
    let q_norm: f32 = query.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
    let q = if q_norm > 1e-12 {
        (query / q_norm as f64)?
    } else {
        query.clone()
    };
    let q_col = q.unsqueeze(1)?; // [d_k, 1]
    state.matmul(&q_col)?.squeeze(1) // [d_v]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    const D: usize = 8; // small dimension for tests

    #[test]
    fn test_delta_rule_memorizes_single_association() -> Result<()> {
        // The delta rule should learn to associate a key with a value.
        // After enough updates with the same (k, v), S.k ~ v.
        let device = Device::Cpu;
        let mut state = Tensor::zeros((D, D), DType::F32, &device)?;

        let key = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;
        let value = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;

        // Run 20 updates with alpha=0.95 (slow decay), beta=0.5 (moderate update)
        for _ in 0..20 {
            let (s_new, _delta) = delta_rule_step(&state, &key, &value, 0.95, 0.5)?;
            state = s_new;
        }

        // Query: S . k should approximate v
        let recalled = delta_rule_query(&state, &key)?;
        let recalled_vec: Vec<f32> = recalled.to_vec1()?;

        assert!(
            recalled_vec[0] > 0.8,
            "Expected recall[0] > 0.8, got {}",
            recalled_vec[0]
        );
        for i in 1..D {
            assert!(
                recalled_vec[i].abs() < 0.3,
                "Expected recall[{}] ~ 0, got {}",
                i,
                recalled_vec[i]
            );
        }

        println!("Single association: recalled = {:?}", recalled_vec);
        Ok(())
    }

    #[test]
    fn test_delta_rule_memorizes_multiple_associations() -> Result<()> {
        // Store 3 distinct (key, value) pairs. The delta rule with
        // orthogonal keys should recall each independently.
        let device = Device::Cpu;
        let mut state = Tensor::zeros((D, D), DType::F32, &device)?;

        // Use near-orthogonal keys (standard basis)
        let keys: Vec<Tensor> = (0..3)
            .map(|i| {
                let mut v = vec![0.0f32; D];
                v[i] = 1.0;
                Tensor::new(v.as_slice(), &device).unwrap()
            })
            .collect();

        let values: Vec<Tensor> = vec![
            Tensor::new(&[1.0f32, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?,
            Tensor::new(&[0.0f32, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0], &device)?,
            Tensor::new(&[0.0f32, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0], &device)?,
        ];

        // Train: cycle through all 3 pairs, 30 iterations
        for _ in 0..30 {
            for (k, v) in keys.iter().zip(values.iter()) {
                let (s_new, _) = delta_rule_step(&state, k, v, 0.98, 0.3)?;
                state = s_new;
            }
        }

        // Verify each association
        for (i, (k, v_expected)) in keys.iter().zip(values.iter()).enumerate() {
            let recalled = delta_rule_query(&state, k)?;
            let recalled_vec: Vec<f32> = recalled.to_vec1()?;
            let expected_vec: Vec<f32> = v_expected.to_vec1()?;

            let mse: f32 = recalled_vec
                .iter()
                .zip(expected_vec.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / D as f32;

            assert!(
                mse < 0.1,
                "Association {} MSE too high: {:.4} (recalled: {:?})",
                i,
                mse,
                &recalled_vec[..4]
            );
            println!(
                "Association {}: MSE={:.4}, recalled[:4]={:?}",
                i,
                mse,
                &recalled_vec[..4]
            );
        }

        Ok(())
    }

    #[test]
    fn test_delta_rule_converges_on_repetitive_sequence() -> Result<()> {
        // KEY TEST: A repetitive sequence (ABCABC...) should cause
        // prediction errors to decrease over time. This verifies the
        // delta rule acts as a compressor (MDL principle).
        let device = Device::Cpu;
        let mut state = Tensor::zeros((D, D), DType::F32, &device)?;

        // Repetitive pattern: 4 tokens cycling
        let pattern_keys: Vec<Tensor> = (0..4)
            .map(|i| {
                let mut v = vec![0.1f32; D];
                v[i % D] = 1.0;
                v[(i + 1) % D] = 0.5;
                let t = Tensor::new(v.as_slice(), &device).unwrap();
                l2_normalize(&t.unsqueeze(0).unwrap(), 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
            })
            .collect();

        let pattern_values: Vec<Tensor> = (0..4)
            .map(|i| {
                let mut v = vec![0.0f32; D];
                v[(i + 2) % D] = 1.0;
                Tensor::new(v.as_slice(), &device).unwrap()
            })
            .collect();

        // Run 10 cycles, track prediction error per cycle
        let mut cycle_errors: Vec<f32> = Vec::new();

        for cycle in 0..10 {
            let mut cycle_error = 0.0f32;
            for (k, v) in pattern_keys.iter().zip(pattern_values.iter()) {
                let (s_new, delta) = delta_rule_step(&state, k, v, 0.95, 0.4)?;
                let delta_norm: f32 = delta.sqr()?.sum_all()?.sqrt()?.to_scalar()?;
                cycle_error += delta_norm;
                state = s_new;
            }
            cycle_errors.push(cycle_error / 4.0);
            println!(
                "  Cycle {}: mean |delta| = {:.4}",
                cycle, cycle_errors[cycle]
            );
        }

        // Prediction error should decrease: last cycle < first cycle
        let first = cycle_errors[0];
        let last = *cycle_errors.last().unwrap();
        assert!(
            last < first * 0.5,
            "Expected convergence: last ({:.4}) should be < 50% of first ({:.4})",
            last,
            first
        );

        println!(
            "Convergence verified: error dropped from {:.4} to {:.4} ({:.0}% reduction)",
            first,
            last,
            (1.0 - last / first) * 100.0
        );
        Ok(())
    }

    #[test]
    fn test_decay_gate_controls_forgetting() -> Result<()> {
        // With alpha=0 (full decay), state resets each step. After convergence
        // (delta=0), S decays to zero. This is correct behavior.
        // With alpha=1 (no decay), state accumulates and persists.
        // With alpha=0.5 (moderate), state holds some memory but decays.
        let device = Device::Cpu;

        let key = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;
        let value = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;

        // Test with TWO different keys to see decay effect
        let key2 = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;
        let value2 = Tensor::new(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;

        // alpha=0: learn (k1,v1), then (k2,v2) — k1 should be forgotten
        let mut s_volatile = Tensor::zeros((D, D), DType::F32, &device)?;
        for _ in 0..10 {
            let (s_new, _) = delta_rule_step(&s_volatile, &key, &value, 0.0, 1.0)?;
            s_volatile = s_new;
        }
        // Now overwrite with (k2, v2)
        for _ in 0..10 {
            let (s_new, _) = delta_rule_step(&s_volatile, &key2, &value2, 0.0, 1.0)?;
            s_volatile = s_new;
        }
        let recall_k1_volatile = delta_rule_query(&s_volatile, &key)?;
        let r1_norm: f32 = recall_k1_volatile.sqr()?.sum_all()?.sqrt()?.to_scalar()?;

        // alpha=1: learn (k1,v1), then (k2,v2) — k1 should be retained
        let mut s_persistent = Tensor::zeros((D, D), DType::F32, &device)?;
        for _ in 0..10 {
            let (s_new, _) = delta_rule_step(&s_persistent, &key, &value, 1.0, 1.0)?;
            s_persistent = s_new;
        }
        for _ in 0..10 {
            let (s_new, _) = delta_rule_step(&s_persistent, &key2, &value2, 1.0, 1.0)?;
            s_persistent = s_new;
        }
        let recall_k1_persistent = delta_rule_query(&s_persistent, &key)?;
        let r1_norm_persistent: f32 = recall_k1_persistent.sqr()?.sum_all()?.sqrt()?.to_scalar()?;

        println!(
            "  alpha=0, recall k1 norm: {:.4} (should be ~0, forgotten)",
            r1_norm
        );
        println!(
            "  alpha=1, recall k1 norm: {:.4} (should be >0, retained)",
            r1_norm_persistent
        );

        // With alpha=0, old memory should be weaker
        assert!(
            r1_norm_persistent > r1_norm,
            "Persistent state should retain k1 better: persistent={:.4} > volatile={:.4}",
            r1_norm_persistent,
            r1_norm
        );

        println!("Decay gate behavior verified");
        Ok(())
    }

    #[test]
    fn test_state_metrics() -> Result<()> {
        let device = Device::Cpu;
        let mut state = Tensor::zeros((D, D), DType::F32, &device)?;

        // Empty state should have ~zero norm
        let m0 = compute_state_metrics(&state)?;
        assert!(m0.frobenius_norm < 1e-6, "Empty state should have 0 norm");

        // After some updates, norm should increase
        let key = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;
        let value = Tensor::new(&[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &device)?;
        for _ in 0..5 {
            let (s_new, _) = delta_rule_step(&state, &key, &value, 0.95, 0.5)?;
            state = s_new;
        }

        let m1 = compute_state_metrics(&state)?;
        assert!(
            m1.frobenius_norm > 0.1,
            "Updated state should have non-zero norm"
        );

        println!(
            "State metrics: frob={:.4}, eff_rank={:.4}",
            m1.frobenius_norm, m1.effective_rank
        );
        Ok(())
    }
}
