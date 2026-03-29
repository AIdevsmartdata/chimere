//! Raw weight pointers for zero-overhead forward pass (v4).
//!
//! Extracts all weight `CudaSlice` pointers at model load time and caches
//! `CudaFunction` handles, eliminating HashMap lookups from the hot path.
//!
//! ## Design
//!
//! The Candle `Tensor` / `QMatMul` wrapper owns weight data via `Arc<Storage>`.
//! Accessing the underlying GPU pointer requires `storage_and_layout()` (Arc read
//! lock) + pattern matching + offset computation — repeated hundreds of times per
//! token. This module clones the raw GPU byte slices once at init time, producing
//! owned `CudaSlice` handles with zero runtime overhead.
//!
//! For quantized weights (Q5_K), the raw bytes are already on GPU as flat U8
//! `Tensor`s (`*_raw: Option<Tensor>` fields). We clone the `CudaSlice<u8>` from
//! these tensors — one device-to-device memcpy per weight, done once at load time.
//!
//! For F32 norm/bias tensors, we clone `CudaSlice<f32>` directly.
//!
//! For MoE expert weights (IQ3_S), the raw U8 tensors are already flat byte arrays
//! on GPU. We clone their `CudaSlice<u8>` the same way.
//!
//! Toggle: `CHIMERE_RAW_V4=1` (not yet wired — this is Phase 1 infrastructure).

use candle_core::cuda_backend::cudarc::driver::CudaSlice;
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Result, Storage, Tensor};

use crate::qwen35_model::{Qwen35LayerQ, Qwen35Model};

// ---------------------------------------------------------------------------
// Helper: extract owned CudaSlice from a Candle Tensor
// ---------------------------------------------------------------------------

/// Clone a CudaSlice<f32> from an F32 Tensor.
///
/// Allocates a new device buffer and performs a device-to-device memcpy.
/// The returned slice is fully owned — no borrow on the source Tensor.
fn clone_f32_slice(tensor: &Tensor, dev: &CudaDevice) -> Result<CudaSlice<f32>> {
    let t = tensor.contiguous()?;
    let n = t.elem_count();
    let (stor, lay) = t.storage_and_layout();
    let cuda = match &*stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("clone_f32_slice: tensor not on CUDA"),
    };
    if t.dtype() != candle_core::DType::F32 {
        candle_core::bail!("clone_f32_slice: expected F32, got {:?}", t.dtype());
    }
    let src = cuda.as_cuda_slice::<f32>()?;
    let off = lay.start_offset();
    let src_view = src.slice(off..off + n);

    let mut owned = dev.alloc_zeros::<f32>(n)
        .map_err(|e| candle_core::Error::Msg(format!("clone_f32_slice alloc({n}): {e}")))?;
    dev.memcpy_dtod(&src_view, &mut owned)
        .map_err(|e| candle_core::Error::Msg(format!("clone_f32_slice dtod: {e}")))?;
    Ok(owned)
}

/// Clone a CudaSlice<u8> from a U8 Tensor (raw quantized bytes).
///
/// Same as `clone_f32_slice` but for byte tensors.
fn clone_u8_slice(tensor: &Tensor, dev: &CudaDevice) -> Result<CudaSlice<u8>> {
    let t = tensor.contiguous()?;
    let n = t.elem_count();
    let (stor, lay) = t.storage_and_layout();
    let cuda = match &*stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("clone_u8_slice: tensor not on CUDA"),
    };
    if t.dtype() != candle_core::DType::U8 {
        candle_core::bail!("clone_u8_slice: expected U8, got {:?}", t.dtype());
    }
    let src = cuda.as_cuda_slice::<u8>()?;
    let off = lay.start_offset();
    let src_view = src.slice(off..off + n);

    let mut owned = dev.alloc_zeros::<u8>(n)
        .map_err(|e| candle_core::Error::Msg(format!("clone_u8_slice alloc({n}): {e}")))?;
    dev.memcpy_dtod(&src_view, &mut owned)
        .map_err(|e| candle_core::Error::Msg(format!("clone_u8_slice dtod: {e}")))?;
    Ok(owned)
}

/// Extract a CudaSlice<u8> from an `Option<Tensor>` (raw Q5_K bytes), returning
/// None if the option is None (weight not in Q5_K format, fallback to QMatMul).
fn clone_optional_u8(opt: &Option<Tensor>, dev: &CudaDevice) -> Result<Option<CudaSlice<u8>>> {
    match opt {
        Some(t) => Ok(Some(clone_u8_slice(t, dev)?)),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// RawGdnWeights — per-GDN-layer weight pointers
// ---------------------------------------------------------------------------

/// Raw weight pointers for one GDN-MoE layer.
///
/// All quantized projection weights are stored as `CudaSlice<u8>` (raw Q5_K bytes).
/// Norm/bias tensors are `CudaSlice<f32>`. MoE expert weights are `CudaSlice<u8>`
/// (raw IQ3_S bytes). All slices are owned — no borrows on the source model.
pub struct RawGdnWeights {
    // -- Norms (F32) --
    /// Attention norm weight [hidden_size]
    pub attn_norm: CudaSlice<f32>,
    /// Post-attention norm weight [hidden_size]
    pub post_norm: CudaSlice<f32>,
    /// SSM state norm weight [d_state]
    pub ssm_norm: CudaSlice<f32>,

    // -- SSM projections (Q5_K raw bytes) --
    /// QKV projection: [conv_channels, hidden_size]
    pub attn_qkv: Option<CudaSlice<u8>>,
    /// Gate projection (z): [value_dim, hidden_size]
    pub attn_gate: Option<CudaSlice<u8>>,
    /// SSM output projection: [hidden_size, value_dim]
    pub ssm_out: Option<CudaSlice<u8>>,
    /// SSM beta projection: [dt_rank, hidden_size]
    pub ssm_beta: Option<CudaSlice<u8>>,
    /// SSM alpha projection: [dt_rank, hidden_size]
    pub ssm_alpha: Option<CudaSlice<u8>>,

    // -- SSM parameters (F32, small) --
    /// Recurrence decay: [dt_rank]
    pub ssm_a: CudaSlice<f32>,
    /// DT bias: [dt_rank]
    pub ssm_dt_bias: CudaSlice<f32>,

    // -- Conv1d weight (F32) --
    /// Conv1d depthwise: [conv_channels, conv_kernel]
    pub ssm_conv1d: CudaSlice<f32>,

    // -- Dimension metadata for GEMV calls --
    /// conv_channels (QKV output rows), e.g. 8192
    pub qkv_rows: usize,
    /// value_dim (gate output rows), e.g. 4096
    pub gate_rows: usize,
    /// dt_rank (beta/alpha output rows), e.g. 32
    pub beta_rows: usize,
    /// dt_rank (same as beta)
    pub alpha_rows: usize,
    /// hidden_size (ssm_out output rows), e.g. 2048
    pub ssm_out_rows: usize,
    /// value_dim (ssm_out input cols), e.g. 4096
    pub ssm_out_cols: usize,

    // -- MoE weights --
    /// Router: [num_experts, hidden_size] row-major (F32) — for raw_f32_gemv.
    pub gate_inp: CudaSlice<f32>,
    /// Router transposed: [hidden_size, num_experts] row-major (F32) — for Candle matmul.
    pub gate_inp_t: CudaSlice<f32>,
    /// Shared expert gate bias: [hidden_size] (F32)
    pub gate_inp_shexp: CudaSlice<f32>,
    /// All expert gate weights stacked (IQ3_S raw bytes)
    pub gate_exps: CudaSlice<u8>,
    /// All expert up weights stacked (IQ3_S raw bytes)
    pub up_exps: CudaSlice<u8>,
    /// All expert down weights stacked (IQ3_S raw bytes)
    pub down_exps: CudaSlice<u8>,
    /// Stacked expert shapes (num_experts, rows, cols)
    pub gate_exps_shape: (usize, usize, usize),
    pub up_exps_shape: (usize, usize, usize),
    pub down_exps_shape: (usize, usize, usize),
    /// Shared expert gate (Q5_K raw bytes) — extracted from QMatMul
    pub shared_gate: Option<CudaSlice<u8>>,
    /// Shared expert up (Q5_K raw bytes)
    pub shared_up: Option<CudaSlice<u8>>,
    /// Shared expert down (Q5_K raw bytes)
    pub shared_down: Option<CudaSlice<u8>>,
}

// ---------------------------------------------------------------------------
// RawAttnWeights — per-attention-layer weight pointers
// ---------------------------------------------------------------------------

/// Raw weight pointers for one attention-MoE layer.
pub struct RawAttnWeights {
    // -- Norms (F32) --
    pub attn_norm: CudaSlice<f32>,
    pub post_norm: CudaSlice<f32>,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,

    // -- Attention projections (Q5_K raw bytes) --
    /// Q+gate interleaved: [num_heads * head_dim * 2, hidden_size]
    pub wq: Option<CudaSlice<u8>>,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    pub wk: Option<CudaSlice<u8>>,
    /// V projection: same shape as K
    pub wv: Option<CudaSlice<u8>>,
    /// Output projection: [hidden_size, num_heads * head_dim * 2]
    pub wo: Option<CudaSlice<u8>>,

    // -- Dimension metadata --
    pub wq_rows: usize,
    pub wk_rows: usize,
    pub wv_rows: usize,
    pub wo_rows: usize,
    pub wo_cols: usize,

    // -- MoE weights (same structure as GDN) --
    /// Router: [num_experts, hidden_size] row-major (F32) — for raw_f32_gemv.
    pub gate_inp: CudaSlice<f32>,
    pub gate_inp_t: CudaSlice<f32>,
    pub gate_inp_shexp: CudaSlice<f32>,
    pub gate_exps: CudaSlice<u8>,
    pub up_exps: CudaSlice<u8>,
    pub down_exps: CudaSlice<u8>,
    pub gate_exps_shape: (usize, usize, usize),
    pub up_exps_shape: (usize, usize, usize),
    pub down_exps_shape: (usize, usize, usize),
    pub shared_gate: Option<CudaSlice<u8>>,
    pub shared_up: Option<CudaSlice<u8>>,
    pub shared_down: Option<CudaSlice<u8>>,
}

// ---------------------------------------------------------------------------
// RawWeights — full model
// ---------------------------------------------------------------------------

/// All raw weight pointers for the full model, extracted once at load time.
///
/// This struct owns cloned `CudaSlice` handles to every weight in the model.
/// The original `Qwen35Model` and its `Tensor`/`QMatMul` wrappers remain valid
/// and can still be used for the Candle-based forward pass (correctness reference).
pub struct RawWeights {
    /// Per-GDN-layer weights (30 layers for 35B-A3B)
    pub gdn: Vec<RawGdnWeights>,
    /// Per-attention-layer weights (10 layers for 35B-A3B)
    pub attn: Vec<RawAttnWeights>,
    /// Output norm weight [hidden_size] (F32)
    pub output_norm: CudaSlice<f32>,
    /// LM head weight (Q5_K raw bytes, if available)
    pub lm_head: Option<CudaSlice<u8>>,
    /// LM head dimensions
    pub lm_head_rows: usize,
    pub lm_head_cols: usize,
}

// ---------------------------------------------------------------------------
// QMatMul raw byte extraction
// ---------------------------------------------------------------------------

/// Extract raw quantized bytes from a QMatMul, if it wraps a QTensor.
///
/// Returns the raw bytes as an owned CudaSlice<u8>. For QMatMul::Tensor and
/// QMatMul::TensorF16, returns None (these are dequantized F32/F16, not raw
/// quantized bytes — they can't be used with the Q5_K GEMV kernel).
fn extract_qmatmul_raw(
    qmm: &candle_core::quantized::QMatMul,
    dev: &CudaDevice,
) -> Result<Option<CudaSlice<u8>>> {
    use candle_core::quantized::QMatMul;
    match qmm {
        QMatMul::QTensor(qt) => {
            // QTensor stores raw quantized bytes. Access via dequantize path would
            // lose the quantization; instead we need the raw storage.
            // Unfortunately, Candle's QTensor doesn't expose raw CudaSlice directly.
            // The raw bytes are only accessible through the internal QStorage.
            //
            // For now, return None — the _raw Option<Tensor> fields already provide
            // the raw bytes for the weights we need. This function is a fallback
            // for weights where _raw wasn't explicitly loaded.
            let _ = (qt, dev);
            Ok(None)
        }
        QMatMul::Tensor(_) | QMatMul::TensorF16(_) => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// MoE weight extraction helper
// ---------------------------------------------------------------------------

/// Extract MoE weights common to both GDN and attention layers.
struct MoeRawWeights {
    gate_inp: CudaSlice<f32>,
    gate_inp_t: CudaSlice<f32>,
    gate_inp_shexp: CudaSlice<f32>,
    gate_exps: CudaSlice<u8>,
    up_exps: CudaSlice<u8>,
    down_exps: CudaSlice<u8>,
    gate_exps_shape: (usize, usize, usize),
    up_exps_shape: (usize, usize, usize),
    down_exps_shape: (usize, usize, usize),
    shared_gate: Option<CudaSlice<u8>>,
    shared_up: Option<CudaSlice<u8>>,
    shared_down: Option<CudaSlice<u8>>,
}

fn extract_moe_weights(
    moe: &crate::qwen35_model::MoeFFN,
    dev: &CudaDevice,
) -> Result<MoeRawWeights> {
    let gate_inp = clone_f32_slice(&moe.gate_inp, dev)?;
    let gate_inp_t = clone_f32_slice(&moe.gate_inp_t, dev)?;
    let gate_inp_shexp = clone_f32_slice(&moe.gate_inp_shexp, dev)?;
    let gate_exps = clone_u8_slice(&moe.gate_exps_raw, dev)?;
    let up_exps = clone_u8_slice(&moe.up_exps_raw, dev)?;
    let down_exps = clone_u8_slice(&moe.down_exps_raw, dev)?;

    // Shared expert: extract from QMatMul if possible
    let shared_gate = extract_qmatmul_raw(&moe.gate_shexp, dev)?;
    let shared_up = extract_qmatmul_raw(&moe.up_shexp, dev)?;
    let shared_down = extract_qmatmul_raw(&moe.down_shexp, dev)?;

    Ok(MoeRawWeights {
        gate_inp,
        gate_inp_t,
        gate_inp_shexp,
        gate_exps,
        up_exps,
        down_exps,
        gate_exps_shape: moe.gate_exps_shape,
        up_exps_shape: moe.up_exps_shape,
        down_exps_shape: moe.down_exps_shape,
        shared_gate,
        shared_up,
        shared_down,
    })
}

// ---------------------------------------------------------------------------
// RawWeights construction
// ---------------------------------------------------------------------------

impl RawWeights {
    /// Extract all raw weight pointers from a loaded Qwen35Model.
    ///
    /// This performs one-time device-to-device copies of all weight data.
    /// The cost is proportional to total weight size (~15 GB for 35B-A3B IQ3_S)
    /// but is done only once at model load time.
    ///
    /// # Requirements
    /// - The model must be loaded on a CUDA device.
    /// - The model must have preloaded quantized layers (`q_layers` is Some).
    ///
    /// # Panics
    /// Panics if the model has no preloaded layers (synthetic mode).
    pub fn from_model(model: &Qwen35Model, dev: &CudaDevice) -> Result<Self> {
        let t0 = std::time::Instant::now();

        let q_layers = model.q_layers.as_ref()
            .ok_or_else(|| candle_core::Error::Msg(
                "RawWeights::from_model: model has no preloaded q_layers".into()))?;

        let config = &model.config;
        let hidden_size = config.hidden_size;
        let conv_channels = config.ssm_n_group * config.ssm_d_state * 2 + config.ssm_dt_rank * config.ssm_d_state;
        let value_dim = config.ssm_dt_rank * config.ssm_d_state;
        let dt_rank = config.ssm_dt_rank;
        // Attention dimensions
        let q_head_dim = config.head_dim * 2; // Q+gate interleaved
        let wq_rows = config.num_attention_heads * q_head_dim;
        let wk_rows = config.num_kv_heads * config.head_dim;
        let wv_rows = wk_rows;
        let wo_rows = hidden_size;
        let wo_cols = config.num_attention_heads * q_head_dim;

        let mut gdn_weights = Vec::new();
        let mut attn_weights = Vec::new();

        for (il, layer) in q_layers.iter().enumerate() {
            match layer {
                Qwen35LayerQ::GdnMoE(w) => {
                    let moe_raw = extract_moe_weights(&w.moe, dev)?;
                    gdn_weights.push(RawGdnWeights {
                        attn_norm: clone_f32_slice(&w.attn_norm, dev)?,
                        post_norm: clone_f32_slice(&w.post_norm, dev)?,
                        ssm_norm: clone_f32_slice(&w.ssm_norm, dev)?,
                        attn_qkv: clone_optional_u8(&w.attn_qkv_raw, dev)?,
                        attn_gate: clone_optional_u8(&w.attn_gate_raw, dev)?,
                        ssm_out: clone_optional_u8(&w.ssm_out_raw, dev)?,
                        ssm_beta: clone_optional_u8(&w.ssm_beta_raw, dev)?,
                        ssm_alpha: clone_optional_u8(&w.ssm_alpha_raw, dev)?,
                        ssm_a: clone_f32_slice(&w.ssm_a, dev)?,
                        ssm_dt_bias: clone_f32_slice(&w.ssm_dt_bias, dev)?,
                        ssm_conv1d: clone_f32_slice(&w.ssm_conv1d, dev)?,
                        qkv_rows: conv_channels,
                        gate_rows: value_dim,
                        beta_rows: dt_rank,
                        alpha_rows: dt_rank,
                        ssm_out_rows: hidden_size,
                        ssm_out_cols: value_dim,
                        gate_inp: moe_raw.gate_inp,
                        gate_inp_t: moe_raw.gate_inp_t,
                        gate_inp_shexp: moe_raw.gate_inp_shexp,
                        gate_exps: moe_raw.gate_exps,
                        up_exps: moe_raw.up_exps,
                        down_exps: moe_raw.down_exps,
                        gate_exps_shape: moe_raw.gate_exps_shape,
                        up_exps_shape: moe_raw.up_exps_shape,
                        down_exps_shape: moe_raw.down_exps_shape,
                        shared_gate: moe_raw.shared_gate,
                        shared_up: moe_raw.shared_up,
                        shared_down: moe_raw.shared_down,
                    });
                    eprintln!("[RAW_WEIGHTS] layer {:2} (gdn-moe): extracted", il);
                }
                Qwen35LayerQ::AttentionMoE(w) => {
                    let moe_raw = extract_moe_weights(&w.moe, dev)?;
                    attn_weights.push(RawAttnWeights {
                        attn_norm: clone_f32_slice(&w.attn_norm, dev)?,
                        post_norm: clone_f32_slice(&w.post_norm, dev)?,
                        q_norm: clone_f32_slice(&w.q_norm, dev)?,
                        k_norm: clone_f32_slice(&w.k_norm, dev)?,
                        wq: clone_optional_u8(&w.wq_raw, dev)?,
                        wk: clone_optional_u8(&w.wk_raw, dev)?,
                        wv: clone_optional_u8(&w.wv_raw, dev)?,
                        wo: clone_optional_u8(&w.wo_raw, dev)?,
                        wq_rows,
                        wk_rows,
                        wv_rows,
                        wo_rows,
                        wo_cols,
                        gate_inp: moe_raw.gate_inp,
                        gate_inp_t: moe_raw.gate_inp_t,
                        gate_inp_shexp: moe_raw.gate_inp_shexp,
                        gate_exps: moe_raw.gate_exps,
                        up_exps: moe_raw.up_exps,
                        down_exps: moe_raw.down_exps,
                        gate_exps_shape: moe_raw.gate_exps_shape,
                        up_exps_shape: moe_raw.up_exps_shape,
                        down_exps_shape: moe_raw.down_exps_shape,
                        shared_gate: moe_raw.shared_gate,
                        shared_up: moe_raw.shared_up,
                        shared_down: moe_raw.shared_down,
                    });
                    eprintln!("[RAW_WEIGHTS] layer {:2} (attn-moe): extracted", il);
                }
                _ => {
                    // Dense GDN or attention layer (27B model) — not supported yet.
                    // The 35B-A3B model only has GdnMoE/AttentionMoE layers.
                    eprintln!("[RAW_WEIGHTS] layer {:2} (dense): SKIPPED (not MoE)", il);
                }
            }
        }

        // Output norm
        let output_norm = clone_f32_slice(&model.output_norm, dev)?;

        // LM head: try to extract raw bytes from QMatMul
        let (lm_head, lm_head_rows, lm_head_cols) = if let Some(ref lm) = model.lm_head {
            let raw = extract_qmatmul_raw(lm, dev)?;
            (raw, config.vocab_size, hidden_size)
        } else {
            (None, config.vocab_size, hidden_size)
        };

        let elapsed = t0.elapsed();
        eprintln!(
            "[RAW_WEIGHTS] Extraction complete: {} GDN + {} attn layers in {:.2}s",
            gdn_weights.len(),
            attn_weights.len(),
            elapsed.as_secs_f64()
        );

        Ok(Self {
            gdn: gdn_weights,
            attn: attn_weights,
            output_norm,
            lm_head,
            lm_head_rows,
            lm_head_cols,
        })
    }

    /// Number of GDN layers with extracted weights.
    pub fn num_gdn_layers(&self) -> usize {
        self.gdn.len()
    }

    /// Number of attention layers with extracted weights.
    pub fn num_attn_layers(&self) -> usize {
        self.attn.len()
    }

    // (all_q5k_available removed — dead code, no callers)
}

// ---------------------------------------------------------------------------
// MoeRawWeightRefs — borrow struct for moe_ffn_forward_raw_v2
// ---------------------------------------------------------------------------

/// Borrowed references to MoE weight CudaSlices for the v2 raw MoE forward path.
///
/// Can be constructed from either `RawGdnWeights` or `RawAttnWeights`, since both
/// have identical MoE fields. This avoids duplicating the v2 function signature.
pub struct MoeRawWeightRefs<'a> {
    /// Router: [num_experts, hidden_size] row-major (F32) — for raw_f32_gemv.
    pub gate_inp: &'a CudaSlice<f32>,
    /// Expert gate weights stacked (IQ3_S raw bytes)
    pub gate_exps: &'a CudaSlice<u8>,
    /// Expert up weights stacked (IQ3_S raw bytes)
    pub up_exps: &'a CudaSlice<u8>,
    /// Expert down weights stacked (IQ3_S raw bytes)
    pub down_exps: &'a CudaSlice<u8>,
    /// Stacked expert shapes (num_experts, rows, cols)
    pub gate_exps_shape: (usize, usize, usize),
    pub up_exps_shape: (usize, usize, usize),
    pub down_exps_shape: (usize, usize, usize),
    /// Shared expert gate (Q5_K raw bytes) — for scratch pool path
    pub shared_gate: Option<&'a CudaSlice<u8>>,
    /// Shared expert up (Q5_K raw bytes) — for scratch pool path
    pub shared_up: Option<&'a CudaSlice<u8>>,
    /// Shared expert down (Q5_K raw bytes) — for scratch pool path
    pub shared_down: Option<&'a CudaSlice<u8>>,
    /// Shared expert gate bias: [hidden_size] (F32) — for scratch pool path
    pub gate_inp_shexp: Option<&'a CudaSlice<f32>>,
}

impl RawGdnWeights {
    /// Borrow MoE weight references for the v2 raw forward path.
    pub fn moe_refs(&self) -> MoeRawWeightRefs<'_> {
        MoeRawWeightRefs {
            gate_inp: &self.gate_inp,
            gate_exps: &self.gate_exps,
            up_exps: &self.up_exps,
            down_exps: &self.down_exps,
            gate_exps_shape: self.gate_exps_shape,
            up_exps_shape: self.up_exps_shape,
            down_exps_shape: self.down_exps_shape,
            shared_gate: self.shared_gate.as_ref(),
            shared_up: self.shared_up.as_ref(),
            shared_down: self.shared_down.as_ref(),
            gate_inp_shexp: Some(&self.gate_inp_shexp),
        }
    }
}

impl RawAttnWeights {
    /// Borrow MoE weight references for the v2 raw forward path.
    pub fn moe_refs(&self) -> MoeRawWeightRefs<'_> {
        MoeRawWeightRefs {
            gate_inp: &self.gate_inp,
            gate_exps: &self.gate_exps,
            up_exps: &self.up_exps,
            down_exps: &self.down_exps,
            gate_exps_shape: self.gate_exps_shape,
            up_exps_shape: self.up_exps_shape,
            down_exps_shape: self.down_exps_shape,
            shared_gate: self.shared_gate.as_ref(),
            shared_up: self.shared_up.as_ref(),
            shared_down: self.shared_down.as_ref(),
            gate_inp_shexp: Some(&self.gate_inp_shexp),
        }
    }
}
