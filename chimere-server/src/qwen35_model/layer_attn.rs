//! Attention layer forward pass variants.
//!
//! Extracted from qwen35_model.rs — pure code movement, zero behavioral changes.
//!
//! ## Pure cudarc path
//!
//! `forward_attn_cudarc()` implements the full attention forward pass using only
//! raw `CudaSlice` buffers and custom CUDA kernels. Zero Candle Tensor ops.
//! Called from `ComputeGraph::forward_token()`.

use candle_core::{Device, Module, Result, Tensor, D};
use super::{Qwen35Model, AttnLayerQ, AttnLayerMoE};
use super::compute_graph::{AttnWeightsRaw, KvCacheRaw, ComputeGraph};
use crate::activations::{rms_norm, sigmoid};
use crate::kernels;
use crate::rope::MRoPE;
use crate::state::GdnRecurrentState;

/// Dispatch GEMV to the correct kernel based on quantization type.
/// Supports IQ3_S, Q5_K (via ggml MMVQ), and Q8_0.
///
/// When `ggml_gpu_bufs` is `Some`, the ggml GPU MMVQ path is used first
/// (4x faster than chimere's cubin kernels). Falls back to existing
/// chimere kernels when buffers are not available.
fn quant_gemv(
    weights: &candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    input: &candle_core::cuda_backend::cudarc::driver::CudaView<'_, f32>,
    output: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    nrows: usize,
    ncols: usize,
    quant_type: crate::gguf_loader::GgmlType,
    q5k_bufs: &mut crate::kernels::q5k_mmvq_ggml::GgmlQ5KBuffers,
    ggml_gpu_bufs: Option<&mut crate::kernels::ggml_gpu::GgmlGpuBuffers>,
    dev: &candle_core::cuda_backend::CudaDevice,
) -> Result<()> {
    use crate::gguf_loader::GgmlType;

    // Try ggml GPU MMVQ first (4x faster than chimere cubin kernels).
    // The buffers are only Some when CHIMERE_GGML_GPU=1 and the compute
    // graph allocated them at init time.
    if let Some(bufs) = ggml_gpu_bufs {
        let w_view = weights.slice(..);
        match quant_type {
            GgmlType::Q5K => {
                // Write directly to output buffer using _q8 variant (like GDN does)
                crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                    input, &mut bufs.q8_input, ncols, dev)?;
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                    &w_view, &bufs.q8_input, output, nrows, ncols)?;
                return Ok(());
            }
            GgmlType::Q8_0 => {
                // Q8_0: use chimere kernel directly (no ggml GPU _q8 variant)
                let w_view_q8 = weights.slice(..);
                crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                    &w_view_q8, input, output, nrows, ncols, dev)?;
                return Ok(());
            }
            GgmlType::Iq3S => {
                crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                    input, &mut bufs.q8_input, ncols, dev)?;
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_q8(
                    &w_view, &bufs.q8_input, output, nrows, ncols)?;
                return Ok(());
            }
            _ => {} // fall through to existing chimere dispatch
        }
    }

    // Existing chimere dispatch (fallback when ggml GPU not available)
    #[cfg(feature = "cubin_fallback")]
    {
        match quant_type {
            GgmlType::Q5K => {
                // Q5_K GEMV: quantize input to Q8_1 first, then MMVQ
                crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
                    input, &mut q5k_bufs.q8_input, ncols, dev)?;
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                    &weights.slice(..), &q5k_bufs.q8_input, output, nrows, ncols, dev)?;
                Ok(())
            }
            GgmlType::Q8_0 => {
                // gemv_q8_0_f32 takes CudaView, not CudaSlice — use the view variant
                let w_view = weights.slice(..);
                crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                    &w_view, input, output, nrows, ncols, dev)?;
                Ok(())
            }
            GgmlType::Iq3S => {
                kernels::gemv_iq3s_fused(&weights.slice(..), input, output, nrows, ncols, dev)?;
                Ok(())
            }
            other => candle_core::bail!("Unsupported quant type {:?} for attention GEMV", other),
        }
    }
    #[cfg(not(feature = "cubin_fallback"))]
    {
        let _ = (&q5k_bufs, &dev); // suppress unused warnings
        panic!("cubin fallback disabled — set CHIMERE_GGML_GPU=1 or enable feature cubin_fallback");
    }
}

impl Qwen35Model {
    /// Bridge: extract CudaSlice from Candle Tensors and call raw moe_ffn.
    /// Forward pass for an attention MoE layer — same attention as dense,
    /// but the FFN is replaced by MoE routing.
    pub(crate) fn forward_attn_layer_moe(
        &self,
        il: usize,
        w: &AttnLayerMoE,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        // --- Tracer: activation stats at entry ---
        self.tracer.activation_stats(il, "hidden_in", hidden);
        let _trace_timer_attn = if self.tracer.timing_enabled() { Some(self.tracer.timer_start()) } else { None };

        // Reuse the dense attention forward — identical except FFN.
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let attn_idx = self.config.attn_index(il).unwrap();

        // Candle ops count for attention layer (non-GQA part):
        // norm(1) + Q/K/V proj(3) + reshape/narrow(4) + QK norm(2) + RoPE(~12)
        // + KV cache(~3) + gate+output(~4) = 29
        crate::candle_counter::tick_n(29);

        let normed = rms_norm(hidden, &w.attn_norm, eps)?;

        let q_full = w.wq.forward(&normed)?;
        let k_proj = w.wk.forward(&normed)?;
        let v_proj = w.wv.forward(&normed)?;

        // Q+gate are interleaved: [Q_h0(dim), gate_h0(dim), Q_h1(dim), gate_h1(dim), ...]
        // Total Q_full dim = num_heads * 2 * q_head_dim
        let q_head_dim_x2 = q_full.dim(1)? / num_heads;  // Q_dim + gate_dim per head
        let q_head_dim = q_head_dim_x2 / 2;               // actual Q head dimension
        let q = q_full.reshape((1, num_heads, q_head_dim_x2))?;
        let q = q.narrow(2, 0, q_head_dim)?;
        let q_gate_raw = q_full.reshape((1, num_heads, q_head_dim_x2))?
            .narrow(2, q_head_dim, q_head_dim)?;
        let q = rms_norm(&q, &w.q_norm, eps)?;
        let k = k_proj.reshape((1, num_kv_heads, head_dim))?;
        let k = rms_norm(&k, &w.k_norm, eps)?;

        let positions = crate::rope::MRoPE::text_positions(1, state.position);
        let q_rotated = self.mrope.apply(&q, &positions)?;
        let k_rotated = self.mrope.apply(&k, &positions)?;

        let q_attn = q_rotated.unsqueeze(2)?;
        let k_new = k_rotated.unsqueeze(2)?;
        let v_new = v_proj.reshape((1, num_kv_heads, 1, head_dim))?;

        let (k_cache, v_cache) = state.kv_append(attn_idx, &k_new, &v_new)?;

        let use_flash_attn = crate::kernels::flash_attn::is_enabled();
        let use_gqa_fused = {
            use once_cell::sync::Lazy;
            // Fused GQA attention: single CUDA kernel replaces GQA expand + score +
            // softmax + output matmul (14 Candle ops). Enable with CHIMERE_GQA_FUSED=1.
            static GQA: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GQA_FUSED").is_ok());
            *GQA
        };

        let attn_out = if use_flash_attn && matches!(hidden.device(), Device::Cuda(_)) {
            // Flash Attention path: reads F16 KV cache directly, online softmax.
            // 0 Candle ops. Enable with CHIMERE_FLASH_ATTN=1.
            let scale_f32 = 1.0f32 / (head_dim as f32).sqrt();
            crate::kernels::flash_attn::flash_attn_decode_tensor(
                &q_rotated, &k_cache, &v_cache,
                num_heads, num_kv_heads, head_dim, scale_f32,
            )?
        } else if use_gqa_fused && matches!(hidden.device(), Device::Cuda(_)) {
            // Fused GQA path: 0 Candle ops (custom CUDA kernel, F32 inputs)
            let scale_f32 = 1.0f32 / (head_dim as f32).sqrt();
            crate::kernels::gqa_attention::fused_gqa_attention_tensor(
                &q_rotated, &k_cache, &v_cache,
                num_heads, num_kv_heads, head_dim, scale_f32,
            )?
        } else {
            // Reference path: GQA expand(~10) + attention(~6) = 16 ops
            crate::candle_counter::tick_n(16);
            let group_size = num_heads / num_kv_heads;
            let cached_seq_len = k_cache.dim(2)?;
            let k_exp = k_cache.unsqueeze(2)?
                .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
                .contiguous()?.reshape((1, num_heads, cached_seq_len, head_dim))?;
            let v_exp = v_cache.unsqueeze(2)?
                .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
                .contiguous()?.reshape((1, num_heads, cached_seq_len, head_dim))?;

            let scores = q_attn.matmul(&k_exp.transpose(2, 3)?)?;
            let scale = 1.0 / (head_dim as f64).sqrt();
            let scores = (scores * scale)?;
            let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
            attn_weights.matmul(&v_exp)?.squeeze(2)?
        };

        let gate = sigmoid(&q_gate_raw)?;
        let gated_out = (&attn_out * &gate)?.reshape((1, num_heads * q_head_dim))?;
        let attn_projected = w.wo.forward(&gated_out)?;
        let h_mid = (hidden + &attn_projected)?;

        // === MoE FFN ===
        let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;
        let use_raw_moe = {
            use once_cell::sync::Lazy;
            // Raw MoE path: fused IQ3S GEMV kernel, 21.1 tok/s vs 11.5 Candle.
            // Disable with CHIMERE_NO_RAW_MOE=1 for the Candle reference path.
            static RAW: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_RAW_MOE").is_err());
            *RAW
        };
        let ffn_out = if w.moe.experts_on_cpu && crate::ggml_backend::is_ncmoe_cpu_enabled() {
            // ncmoe CPU path: IQ3_S GEMV on CPU via ggml AVX2 (invert ncmoe)
            Self::moe_ffn_forward_ncmoe_cpu(
                &normed_ffn, &w.moe, self.config.experts_per_token, &self.device,
            )?
        } else if w.moe.experts_on_cpu {
            // ncmoe GPU batch copy path (default): copy expert weights to GPU
            if state.ncmoe_bufs.is_none() {
                let g_elems = w.moe.gate_exps_shape.1 * w.moe.gate_exps_shape.2;
                let u_elems = w.moe.up_exps_shape.1 * w.moe.up_exps_shape.2;
                let d_elems = w.moe.down_exps_shape.1 * w.moe.down_exps_shape.2;
                state.ncmoe_bufs = Some(crate::state::NcmoeBufs::new(
                    g_elems, u_elems, d_elems,
                    w.moe.gate_exps_shape.1,
                    hidden.dim(1)?,
                    self.config.experts_per_token,
                    &self.device,
                )?);
            }
            Self::moe_ffn_forward_cpu(
                &normed_ffn, &w.moe, self.config.experts_per_token, &self.device,
                state.ncmoe_bufs.as_mut().unwrap(),
            )?
        } else if super::moe_cudarc::cudarc_moe_enabled()
            && state.moe_cudarc_bufs.is_some()
            && self.raw_weights.is_some()
        {
            // Pure cudarc MoE path: zero Candle Tensor ops
            let rw = self.raw_weights.as_ref().unwrap();
            let moe_refs = rw.attn[attn_idx].moe_refs();
            let cudarc_bufs = state.moe_cudarc_bufs.as_mut().unwrap();
            let Device::Cuda(cuda_dev) = &self.device else {
                candle_core::bail!("cudarc MoE requires CUDA");
            };
            let (nf_stor, nf_lay) = normed_ffn.storage_and_layout();
            let nf_cuda = match &*nf_stor {
                candle_core::Storage::Cuda(c) => c,
                _ => candle_core::bail!("normed_ffn not on CUDA"),
            };
            let nf_view = nf_cuda.as_cuda_slice::<f32>()?.slice(nf_lay.start_offset()..);
            let result_slice = super::moe_cudarc::moe_ffn_cudarc(
                &nf_view, &moe_refs, cudarc_bufs, cuda_dev,
            )?;
            drop(nf_stor);
            super::moe_cudarc::cuda_slice_to_tensor(
                result_slice, hidden.dim(1)?, cuda_dev,
            )?
        } else if use_raw_moe && state.raw_moe_buffers.is_some() {
            crate::candle_counter::tick_n(3); // shared expert wrapping
            // Try v2 (zero storage_and_layout for expert weights + raw router GEMV)
            if let Some(ref rw) = self.raw_weights {
                let moe_refs = rw.attn[attn_idx].moe_refs();
                // Split borrow: raw_moe_buffers and scratch_pool are separate fields
                let buffers = state.raw_moe_buffers.as_mut().unwrap();
                let scratch = state.scratch_pool.as_mut();
                Self::moe_ffn_forward_raw_v2(
                    &normed_ffn, &w.moe, &moe_refs,
                    self.config.experts_per_token,
                    hidden.dim(1)?, 512,
                    self.config.num_experts,
                    buffers, scratch)?
            } else {
                // Fallback: v1 (6 storage_and_layout calls per layer)
                Self::moe_ffn_forward_raw(&normed_ffn, &w.moe, self.config.experts_per_token,
                    hidden.dim(1)?, 512,
                    self.config.num_experts,
                    state.raw_moe_buffers.as_mut().unwrap())?
            }
        } else {
            crate::candle_counter::tick_n(45); // full Candle MoE
            Self::moe_ffn_forward(&normed_ffn, &w.moe, self.config.experts_per_token)?
        };
        let h_out = (&h_mid + &ffn_out)?;

        // --- Tracer: layer delta at exit + timing ---
        self.tracer.layer_delta(il, hidden, &h_out);
        if let Some(t) = _trace_timer_attn {
            self.tracer.timer_log(il, "attn_layer_total", t);
        }

        Ok(h_out)
    }

    /// Forward pass for an attention layer using preloaded QMatMul weights.
    ///
    /// Implements full multi-head attention with KV cache:
    ///   1. Project Q, K, V from the current token
    ///   2. Apply QK-norm and MRoPE (with position from state)
    ///   3. Append K, V to the KV cache for this layer
    ///   4. Compute scaled dot-product attention over ALL cached K,V
    ///   5. Apply sigmoid gate and output projection
    pub(crate) fn forward_attn_layer_q(
        &self,
        il: usize,
        w: &AttnLayerQ,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        // --- Tracer: activation stats at entry ---
        self.tracer.activation_stats(il, "hidden_in", hidden);

        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim;
        let attn_idx = self.config.attn_index(il).unwrap();

        // Pre-attention norm
        let normed = rms_norm(hidden, &w.attn_norm, eps)?;

        // Q, K, V projections via QMatMul
        let q_full = w.wq.forward(&normed)?;   // [1, num_heads * 2 * head_dim]
        let k_proj = w.wk.forward(&normed)?;   // [1, num_kv_heads * head_dim]
        let v_proj = w.wv.forward(&normed)?;   // [1, num_kv_heads * head_dim]

        // For Qwen3.5, Q+gate are INTERLEAVED per head:
        // Layout: [Q_h0(head_dim), gate_h0(head_dim), Q_h1(head_dim), gate_h1(head_dim), ...]
        let q_full = q_full.reshape((1, num_heads, 2 * head_dim))?;
        let q = q_full.narrow(2, 0, head_dim)?;            // [1, num_heads, head_dim]
        let q_gate_raw = q_full.narrow(2, head_dim, head_dim)?;  // [1, num_heads, head_dim]

        // QK norm (per-head RMSNorm)
        let q = rms_norm(&q, &w.q_norm, eps)?;             // [1, num_heads, head_dim]
        let k = k_proj.reshape((1, num_kv_heads, head_dim))?;
        let k = rms_norm(&k, &w.k_norm, eps)?;             // [1, num_kv_heads, head_dim]

        // Apply MRoPE to Q and K BEFORE caching
        // MRoPE.apply expects [seq_len, num_heads, head_dim]
        // q is [batch=1, num_heads, head_dim], reinterpret batch as seq_len (both are 1)
        let positions = MRoPE::text_positions(1, state.position);
        let q_rotated = self.mrope.apply(&q, &positions)?;       // [1, num_heads, head_dim]
        let k_rotated = self.mrope.apply(&k, &positions)?;       // [1, num_kv_heads, head_dim]

        // Reshape to [1, heads, 1, head_dim] for cache format
        // q_rotated: [1, num_heads, head_dim] -> [1, num_heads, 1, head_dim]
        let q_attn = q_rotated.unsqueeze(2)?;                    // [1, num_heads, 1, head_dim]

        // k_rotated: [1, num_kv_heads, head_dim] -> [1, num_kv_heads, 1, head_dim]
        let k_new = k_rotated.unsqueeze(2)?;                     // [1, num_kv_heads, 1, head_dim]

        let v_new = v_proj.reshape((1, num_kv_heads, 1, head_dim))?;  // [1, num_kv_heads, 1, head_dim]

        // Append to KV cache along seq_len dimension (dim 2)
        let (k_cache, v_cache) = state.kv_append(attn_idx, &k_new, &v_new)?;

        // GQA: expand K,V from num_kv_heads to num_heads by repeating
        let group_size = num_heads / num_kv_heads;
        let cached_seq_len = k_cache.dim(2)?;
        let k_expanded = k_cache
            .unsqueeze(2)?                                   // [1, num_kv_heads, 1, seq_len, head_dim]
            .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, cached_seq_len, head_dim))?;  // [1, num_heads, seq_len, head_dim]
        let v_expanded = v_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, cached_seq_len, head_dim))?;

        // Scaled dot-product attention
        // Q: [1, num_heads, 1, head_dim], K: [1, num_heads, seq_len, head_dim]
        // scores = Q @ K^T / sqrt(head_dim) -> [1, num_heads, 1, seq_len]
        let scores = q_attn.matmul(&k_expanded.transpose(2, 3)?)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = (scores * scale)?;

        // Softmax over seq_len dimension (last dim)
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Attention output: [1, num_heads, 1, head_dim]
        let attn_out = attn_weights.matmul(&v_expanded)?;
        let attn_out = attn_out.squeeze(2)?;                // [1, num_heads, head_dim]

        // Apply gate (sigmoid)
        let gate = sigmoid(&q_gate_raw)?;                   // [1, num_heads, head_dim]
        let gated_out = (&attn_out * &gate)?;               // [1, num_heads, head_dim]
        let gated_out = gated_out.reshape((1, num_heads * head_dim))?;

        // Output projection via QMatMul
        let attn_projected = w.wo.forward(&gated_out)?;

        // Residual
        let h_mid = (hidden + &attn_projected)?;

        // Post-attention norm -> FFN
        let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;
        let ffn_out = self.swiglu_ffn_q(&normed_ffn, &w.ffn_gate, &w.ffn_up, &w.ffn_down)?;

        let h_out = (&h_mid + &ffn_out)?;

        // --- Tracer: layer delta at exit ---
        self.tracer.layer_delta(il, hidden, &h_out);

        Ok(h_out)
    }
}

// ===========================================================================
// Pure cudarc attention forward — zero Candle Tensor ops
// ===========================================================================

impl ComputeGraph {
    /// Pure cudarc attention forward pass. Zero Candle Tensor ops.
    ///
    /// Reads from `self.normed` (pre-attention RMSNorm output, `[hidden_size]`).
    /// Writes the attention output (after O projection) into `self.hidden`.
    ///
    /// The caller (ComputeGraph::forward_token) handles:
    /// - Residual save (`residual = hidden`) before calling this function
    /// - Pre-attention RMSNorm (`normed = RMSNorm(hidden)`) before calling
    /// - Residual add (`hidden += residual`) after this function returns
    /// - FFN norm + MoE FFN + residual add (separate step)
    ///
    /// ## Operation sequence
    ///
    /// 1. Q/K/V projections via IQ3_S GEMV (3 kernel launches)
    /// 2. Deinterleave Q+gate from Qwen3.5 interleaved layout (1 launch)
    /// 3. Per-head RMSNorm on Q and K (2 launches)
    /// 4. MRoPE rotation on Q and K (2 launches)
    /// 5. KV cache append (2 dtod memcpy per KV head)
    /// 6. Fused GQA attention: score + softmax + output (1 launch)
    /// 7. Sigmoid gate multiply (1 launch)
    /// 8. O projection via IQ3_S GEMV (1 launch)
    ///
    /// Total: 11 kernel launches + 4 memcpy. Zero cudaMalloc. Zero Tensor ops.
    ///
    /// ## Dimensions (Qwen3.5-35B-A3B)
    ///
    /// - hidden_size = 2048
    /// - n_heads = 16, n_kv_heads = 2, head_dim = 256
    /// - Q projection output: `[n_heads * head_dim * 2]` = 8192 (fused Q+gate)
    /// - K/V projection output: `[n_kv_heads * head_dim]` = 512
    /// - GQA ratio: 8:1 (8 Q heads share 1 KV head)
    pub(crate) fn forward_attn_cudarc(
        &mut self,
        attn_idx: usize,
        attn_w: &AttnWeightsRaw,
        kv_cache: &mut KvCacheRaw,
        position: usize,
        rope_tables: &kernels::RawMRoPETables,
    ) -> Result<()> {
        let n_heads = self.n_heads;
        let n_kv_heads = self.n_kv_heads;
        let head_dim = self.head_dim;
        let hidden_size = self.hidden_size;
        let eps = 1e-6f32;
        let dev = &self.dev.clone();

        // q_proj output rows = n_heads * head_dim * 2 (fused Q + gate)
        let q_proj_rows = n_heads * head_dim * 2;
        let kv_proj_rows = n_kv_heads * head_dim;

        // -----------------------------------------------------------------
        // 1. Q/K/V projections: dispatch to correct GEMV kernel per quant type.
        //    Q/K are typically Q5_K, V is Q8_0, O is Q5_K in custom-mix.
        // -----------------------------------------------------------------
        {
            let normed_view = self.normed.slice(..);

            // Q projection (Q5_K or IQ3_S)
            quant_gemv(&attn_w.q_raw, &normed_view, &mut self.q_buf,
                q_proj_rows, hidden_size, attn_w.q_quant, &mut self.q5k_bufs,
                self.ggml_gpu_bufs.as_mut(), dev)?;

            // K projection (Q5_K or IQ3_S)
            quant_gemv(&attn_w.k_raw, &normed_view, &mut self.k_buf,
                kv_proj_rows, hidden_size, attn_w.k_quant, &mut self.q5k_bufs,
                self.ggml_gpu_bufs.as_mut(), dev)?;

            // V projection (Q8_0 or IQ3_S)
            quant_gemv(&attn_w.v_raw, &normed_view, &mut self.v_buf,
                kv_proj_rows, hidden_size, attn_w.v_quant, &mut self.q5k_bufs,
                self.ggml_gpu_bufs.as_mut(), dev)?;
        }

        // -----------------------------------------------------------------
        // 2. Deinterleave Q+gate from Qwen3.5 interleaved layout.
        //    q_buf layout: [Q_h0(dim), gate_h0(dim), Q_h1(dim), gate_h1(dim), ...]
        //    -> q_heads: [num_heads * head_dim], gate_heads: [num_heads * head_dim]
        // -----------------------------------------------------------------
        kernels::raw_deinterleave_q_gate(
            &self.q_buf,
            &mut self.q_heads,
            &mut self.gate_heads,
            n_heads,
            head_dim,
            dev,
        )?;

        // -----------------------------------------------------------------
        // 3. Per-head RMSNorm on Q and K.
        //    Q: [n_heads * head_dim] normed per-head with q_norm weight [head_dim]
        //    K: [n_kv_heads * head_dim] normed per-head with k_norm weight [head_dim]
        //
        //    Writes to separate output buffers to avoid Rust aliasing violation
        //    (same CudaSlice as both &input and &mut output is not allowed).
        //    The CUDA kernel itself supports in-place, but Rust's borrow checker
        //    prevents it. Output goes to q_roped/k_roped as temporaries.
        // -----------------------------------------------------------------
        kernels::raw_rms_norm_per_head(
            &self.q_heads,
            &attn_w.q_norm,
            &mut self.q_roped,   // temporary: normed Q into q_roped
            n_heads,
            head_dim,
            eps,
            dev,
        )?;

        kernels::raw_rms_norm_per_head(
            &self.k_buf,
            &attn_w.k_norm,
            &mut self.k_roped,   // temporary: normed K into k_roped
            n_kv_heads,
            head_dim,
            eps,
            dev,
        )?;

        // -----------------------------------------------------------------
        // 4. MRoPE rotation on Q and K.
        //    For text-only: section 0 gets the position, sections 1/2 = 0.
        //    Reads from the RMSNorm output (q_roped/k_roped) and writes
        //    back to q_heads/k_buf (now free for reuse).
        // -----------------------------------------------------------------
        kernels::raw_mrope_apply(
            &self.q_roped,       // normed Q (from step 3)
            &mut self.q_heads,   // RoPE'd Q output (q_heads now free)
            rope_tables,
            position,
            n_heads,
            dev,
        )?;

        kernels::raw_mrope_apply(
            &self.k_roped,       // normed K (from step 3)
            &mut self.k_buf,     // RoPE'd K output (k_buf now free)
            rope_tables,
            position,
            n_kv_heads,
            dev,
        )?;

        // -----------------------------------------------------------------
        // 5. KV cache append: write RoPE'd K (k_buf) and V (v_buf) into
        //    the ring buffer. Cache layout: [num_kv_heads, kv_cap, head_dim].
        //
        //    We do NOT use `raw_kv_append` here because it increments the
        //    shared position counter. All attention layers share the same
        //    sequence position — the counter is incremented once per token
        //    by the caller (ComputeGraph::forward_token) after all layers.
        // -----------------------------------------------------------------
        let pos = kv_cache.pos;
        let kv_cap = kv_cache.max_seq_len;
        if pos >= kv_cap {
            candle_core::bail!(
                "KV cache full: position {} >= capacity {}",
                pos, kv_cap
            );
        }

        {
            let kv_k = &mut kv_cache.k_cache[attn_idx];
            let kv_v = &mut kv_cache.v_cache[attn_idx];

            // Append K and V for each KV head at position `pos`.
            // Source layout: [num_kv_heads, head_dim] flat.
            // Cache layout:  [num_kv_heads, kv_cap, head_dim] flat.
            for h in 0..n_kv_heads {
                let src_off = h * head_dim;
                let dst_off = h * kv_cap * head_dim + pos * head_dim;

                // K source is k_buf (contains RoPE'd K after step 4)
                let k_src = self.k_buf.slice(src_off..src_off + head_dim);
                let mut k_dst = kv_k.slice_mut(dst_off..dst_off + head_dim);
                dev.memcpy_dtod(&k_src, &mut k_dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("kv_append k head {h}: {e}")
                    ))?;

                let v_src = self.v_buf.slice(src_off..src_off + head_dim);
                let mut v_dst = kv_v.slice_mut(dst_off..dst_off + head_dim);
                dev.memcpy_dtod(&v_src, &mut v_dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("kv_append v head {h}: {e}")
                    ))?;
            }
        }

        // Sequence length for attention = pos + 1 (including the token just appended).
        let seq_len = pos + 1;

        // -----------------------------------------------------------------
        // 6. Fused GQA attention: score + softmax + weighted sum.
        //    Q: [num_heads * head_dim] (post-RoPE)
        //    K cache: [num_kv_heads * kv_cap * head_dim] (F32)
        //    V cache: [num_kv_heads * kv_cap * head_dim] (F32)
        //    Output: [num_heads * head_dim]
        //
        //    The kernel handles GQA internally: each Q head reads from
        //    K[h / group_size] and V[h / group_size].
        // -----------------------------------------------------------------
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Re-borrow kv_k/kv_v as views (raw_kv_append consumed the mutable refs).
        let kv_k_view = kv_cache.k_cache[attn_idx].slice(..);
        let kv_v_view = kv_cache.v_cache[attn_idx].slice(..);

        crate::kernels::gqa_attention::raw_gqa_attention_strided(
            &self.q_heads.slice(..),   // RoPE'd Q (from step 4)
            &kv_k_view,
            &kv_v_view,
            &mut self.attn_result,
            n_heads,
            n_kv_heads,
            seq_len,
            head_dim,
            scale,
            kv_cap,     // stride = full cache capacity per head
            dev,
        )?;

        // -----------------------------------------------------------------
        // 7. Sigmoid gate multiply: output = sigmoid(gate) * attn_result.
        //    Both are [n_heads * head_dim] = 4096 elements.
        //    Result goes into q_roped (reusing buffer, no longer needed).
        // -----------------------------------------------------------------
        let gated_n = n_heads * head_dim;
        kernels::raw_sigmoid_gate_mul(
            &self.attn_result,
            &self.gate_heads,
            &mut self.q_roped,  // reuse q_roped as gated output buffer
            gated_n,
            dev,
        )?;

        // -----------------------------------------------------------------
        // 8. O projection: gated[n_heads * head_dim] x o_raw -> hidden[hidden_size]
        //    Dispatch to correct kernel based on quant type (Q5_K for custom-mix).
        // -----------------------------------------------------------------
        quant_gemv(&attn_w.o_raw, &self.q_roped.slice(..),
            &mut self.hidden, hidden_size, gated_n,
            attn_w.o_quant, &mut self.q5k_bufs,
            self.ggml_gpu_bufs.as_mut(), dev)?;

        Ok(())
    }
}

// ===========================================================================
// Batch attention prefill — V2-2: batch elementwise ops, sequential KV+attn
// ===========================================================================

/// Batch attention forward for prefill: processes N tokens through one attention layer.
///
/// ## V2-2 Architecture (Phase A batch + Phase B sequential)
///
/// **Phase A (batched — 1 kernel launch per step, not N):**
///   A1. Batch Q8_1 quantize of normed_batch
///   A2. Per-token GEMV loop for Q+gate, K, V projections (still per-token —
///       GEMV kernels are single-vector; batch = N launches per weight matrix)
///   A3. Batch deinterleave Q+gate → q_heads_batch + gate_batch
///   A4. Batch per-head RMSNorm on Q and K
///   A5. Batch MRoPE on Q and K
///
/// **Phase B (sequential — per-token, for causal attention):**
///   B1. KV cache append (K from k_roped_batch, V from v_batch)
///   B2. GQA attention (Q from q_roped_batch)
///   B3. Sigmoid gate (from gate_batch) * attention output
///   B4. O projection (GEMV)
///   B5. Copy O-projection output to hidden_batch
///
/// ## Performance
///
/// Phase A replaces 5N kernel launches with 5 launches + N*3 GEMVs.
/// For a 512-token prefill: 2560 → 1541 launches (40% reduction in elementwise).
/// The real win is eliminating 5N dtod memcpy for copying between batch ↔ scratch.
///
/// ## KV cache semantics
///
/// Token 0 writes K,V at position `base_position`, attends over `base_position + 1` entries.
/// Token N-1 writes at `base_position + N - 1`, attends over `base_position + N` entries.
pub(crate) fn forward_attn_prefill_cudarc(
    attn_idx: usize,
    attn_w: &super::compute_graph::AttnWeightsRaw,
    graph: &mut super::compute_graph::ComputeGraph,
    prefill: &mut super::compute_graph::PrefillBuffers,
    kv_cache: &mut super::compute_graph::KvCacheRaw,
    base_position: usize,
    n_tokens: usize,
    rope_tables: &crate::kernels::RawMRoPETables,
    _cached: Option<&super::compute_graph::CachedWeightPtrs>,
) -> candle_core::Result<()> {
    let hs = graph.hidden_size;
    let n_heads = graph.n_heads;
    let n_kv_heads = graph.n_kv_heads;
    let head_dim = graph.head_dim;
    let eps = 1e-6f32;
    let dev = graph.dev.clone(); // cheap Arc clone — avoids borrow conflicts

    let q_proj_rows = n_heads * head_dim * 2;  // fused Q + gate
    let kv_proj_rows = n_kv_heads * head_dim;

    // Validate that the KV cache has room for all N tokens.
    if base_position + n_tokens > kv_cache.max_seq_len {
        candle_core::bail!(
            "KV cache overflow in prefill: base_position({}) + n_tokens({}) > capacity({})",
            base_position, n_tokens, kv_cache.max_seq_len,
        );
    }

    // =====================================================================
    // Phase A: Batch operations (elementwise kernels over all N tokens)
    // =====================================================================

    // A1. Batch Q8_1 quantize: normed_batch[N * hs] → q8_batch[N * q8_row_bytes]
    //     Single kernel launch for all N tokens.
    crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1_batched(
        &prefill.normed_batch,
        &mut prefill.q8_batch,
        hs,
        n_tokens,
    )?;

    // A2. Per-token GEMV loop for Q+gate, K, V projections.
    //     GEMV is inherently single-vector (GEMM would need different kernels),
    //     so we loop per-token but read from q8_batch and write directly to
    //     batch output buffers (attn_q_batch, attn_k_batch, attn_v_batch).
    {
        let ncols_padded = crate::kernels::ggml_gpu::pad(hs, crate::kernels::ggml_gpu::MATRIX_ROW_PADDING);
        let q8_row_bytes = (ncols_padded / crate::kernels::ggml_gpu::Q8_1_BLOCK_ELEMS)
                         * crate::kernels::ggml_gpu::Q8_1_BLOCK_BYTES;

        for t in 0..n_tokens {
            let q8_offset = t * q8_row_bytes;
            let q_offset = t * q_proj_rows;
            let kv_offset = t * kv_proj_rows;

            // Copy this token's Q8_1 data into the single-token ggml scratch
            // (the quant_gemv function reads from ggml_gpu_bufs.q8_input).
            // We need to set up the quantized input for the GEMV dispatch.
            //
            // Instead of re-quantizing (which would waste the batch quantize),
            // copy the pre-quantized Q8_1 row into the ggml scratch buffer.
            if let Some(ref mut ggml_bufs) = graph.ggml_gpu_bufs {
                let src = prefill.q8_batch.slice(q8_offset..q8_offset + q8_row_bytes);
                dev.memcpy_dtod(&src, &mut ggml_bufs.q8_input)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 q8 copy t={t}: {e}")))?;

                // Q projection: use pre-quantized Q8_1 input directly.
                // The _q8 variants take &CudaSlice<u8> for the pre-quantized input.
                match attn_w.q_quant {
                    crate::gguf_loader::GgmlType::Q5K => {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                            &attn_w.q_raw.slice(..), &ggml_bufs.q8_input,
                            &mut graph.q_buf, q_proj_rows, hs)?;
                    }
                    crate::gguf_loader::GgmlType::Iq3S => {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_q8(
                            &attn_w.q_raw.slice(..), &ggml_bufs.q8_input,
                            &mut graph.q_buf, q_proj_rows, hs)?;
                    }
                    crate::gguf_loader::GgmlType::Q8_0 => {
                        let normed_src = prefill.normed_batch.slice(t * hs..(t + 1) * hs);
                        crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                            &attn_w.q_raw.slice(..), &normed_src,
                            &mut graph.q_buf, q_proj_rows, hs, &dev)?;
                    }
                    other => candle_core::bail!("Unsupported Q quant {:?}", other),
                }

                // K projection
                match attn_w.k_quant {
                    crate::gguf_loader::GgmlType::Q5K => {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                            &attn_w.k_raw.slice(..), &ggml_bufs.q8_input,
                            &mut graph.k_buf, kv_proj_rows, hs)?;
                    }
                    crate::gguf_loader::GgmlType::Iq3S => {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_q8(
                            &attn_w.k_raw.slice(..), &ggml_bufs.q8_input,
                            &mut graph.k_buf, kv_proj_rows, hs)?;
                    }
                    crate::gguf_loader::GgmlType::Q8_0 => {
                        let normed_src = prefill.normed_batch.slice(t * hs..(t + 1) * hs);
                        crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                            &attn_w.k_raw.slice(..), &normed_src,
                            &mut graph.k_buf, kv_proj_rows, hs, &dev)?;
                    }
                    other => candle_core::bail!("Unsupported K quant {:?}", other),
                }

                // V projection
                match attn_w.v_quant {
                    crate::gguf_loader::GgmlType::Q5K => {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                            &attn_w.v_raw.slice(..), &ggml_bufs.q8_input,
                            &mut graph.v_buf, kv_proj_rows, hs)?;
                    }
                    crate::gguf_loader::GgmlType::Iq3S => {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_q8(
                            &attn_w.v_raw.slice(..), &ggml_bufs.q8_input,
                            &mut graph.v_buf, kv_proj_rows, hs)?;
                    }
                    crate::gguf_loader::GgmlType::Q8_0 => {
                        let normed_src = prefill.normed_batch.slice(t * hs..(t + 1) * hs);
                        crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                            &attn_w.v_raw.slice(..), &normed_src,
                            &mut graph.v_buf, kv_proj_rows, hs, &dev)?;
                    }
                    other => candle_core::bail!("Unsupported V quant {:?}", other),
                }
            } else {
                // Fallback: no ggml GPU buffers, use the original per-token path
                // Copy normed to single-token scratch, run quant_gemv normally
                let src = prefill.normed_batch.slice(t * hs..(t + 1) * hs);
                dev.memcpy_dtod(&src, &mut graph.normed)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 normed copy t={t}: {e}")))?;
                let normed_view = graph.normed.slice(..);
                quant_gemv(&attn_w.q_raw, &normed_view, &mut graph.q_buf,
                    q_proj_rows, hs, attn_w.q_quant, &mut graph.q5k_bufs,
                    graph.ggml_gpu_bufs.as_mut(), &dev)?;
                quant_gemv(&attn_w.k_raw, &normed_view, &mut graph.k_buf,
                    kv_proj_rows, hs, attn_w.k_quant, &mut graph.q5k_bufs,
                    graph.ggml_gpu_bufs.as_mut(), &dev)?;
                quant_gemv(&attn_w.v_raw, &normed_view, &mut graph.v_buf,
                    kv_proj_rows, hs, attn_w.v_quant, &mut graph.q5k_bufs,
                    graph.ggml_gpu_bufs.as_mut(), &dev)?;
            }

            // Copy Q, K, V results to batch buffers
            {
                let mut q_dst = prefill.attn_q_batch.slice_mut(q_offset..q_offset + q_proj_rows);
                dev.memcpy_dtod(&graph.q_buf.slice(..q_proj_rows), &mut q_dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 q copy t={t}: {e}")))?;
            }
            {
                let mut k_dst = prefill.attn_k_batch.slice_mut(kv_offset..kv_offset + kv_proj_rows);
                dev.memcpy_dtod(&graph.k_buf.slice(..kv_proj_rows), &mut k_dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 k copy t={t}: {e}")))?;
            }
            {
                let mut v_dst = prefill.attn_v_batch.slice_mut(kv_offset..kv_offset + kv_proj_rows);
                dev.memcpy_dtod(&graph.v_buf.slice(..kv_proj_rows), &mut v_dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 v copy t={t}: {e}")))?;
            }
        }
    }

    // A3. Batch deinterleave Q+gate → q_heads_batch + gate_heads_batch
    //     1 kernel launch for all N * num_heads blocks.
    kernels::raw_deinterleave_q_gate_batch(
        &prefill.attn_q_batch,
        &mut prefill.attn_q_heads_batch,
        &mut prefill.attn_gate_heads_batch,
        n_heads,
        head_dim,
        n_tokens,
        &dev,
    )?;

    // A4. Batch per-head RMSNorm on Q and K.
    //     Q: [N * n_heads * head_dim] normed with q_norm weight [head_dim]
    //     K: [N * n_kv_heads * head_dim] normed with k_norm weight [head_dim]
    //     2 kernel launches total (not 2N).
    kernels::raw_rms_norm_per_head_batch(
        &prefill.attn_q_heads_batch,
        &attn_w.q_norm,
        &mut prefill.attn_q_normed_batch,
        n_heads,
        head_dim,
        n_tokens,
        eps,
        &dev,
    )?;

    kernels::raw_rms_norm_per_head_batch(
        &prefill.attn_k_batch,
        &attn_w.k_norm,
        &mut prefill.attn_k_normed_batch,
        n_kv_heads,
        head_dim,
        n_tokens,
        eps,
        &dev,
    )?;

    // A5. Batch MRoPE on Q and K.
    //     Upload positions array [base_position, base_position+1, ...] to GPU.
    //     2 kernel launches total (not 2N).
    let positions: Vec<i32> = (0..n_tokens as i32)
        .map(|t| (base_position as i32) + t)
        .collect();
    let mut positions_gpu = dev.alloc_zeros::<i32>(n_tokens)
        .map_err(|e| candle_core::Error::Msg(format!("alloc positions_gpu: {e}")))?;
    dev.memcpy_htod(&positions, &mut positions_gpu)
        .map_err(|e| candle_core::Error::Msg(format!("htod positions: {e}")))?;

    kernels::raw_mrope_apply_batch(
        &prefill.attn_q_normed_batch,
        &mut prefill.attn_q_roped_batch,
        &positions_gpu,
        rope_tables,
        n_heads,
        n_tokens,
        &dev,
    )?;

    kernels::raw_mrope_apply_batch(
        &prefill.attn_k_normed_batch,
        &mut prefill.attn_k_roped_batch,
        &positions_gpu,
        rope_tables,
        n_kv_heads,
        n_tokens,
        &dev,
    )?;

    // =====================================================================
    // Phase B: Attention + gate + O-projection
    //
    // Two paths:
    //   - Flash Attention prefill (CHIMERE_FLASH_PREFILL=1): 1 kernel launch
    //     for causal attention over all N tokens, then per-token gate + O-proj.
    //   - Sequential fallback: per-token KV append + GQA attention + gate + O-proj.
    // =====================================================================

    let q_heads_size = n_heads * head_dim;
    let kv_size = n_kv_heads * head_dim;
    let kv_cap = kv_cache.max_seq_len;

    let use_flash_prefill = {
        use once_cell::sync::Lazy;
        static FLASH: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FLASH_PREFILL").is_ok());
        *FLASH
    };

    let flash_debug = {
        use once_cell::sync::Lazy;
        static DBG: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FLASH_DEBUG").is_ok());
        *DBG
    };

    if use_flash_prefill && n_tokens > 1 && base_position == 0 {
        // =================================================================
        // Flash Attention prefill path: 1 kernel launch replaces N per-token
        // attention calls. O(N^2) work done in a single tiled kernel with
        // causal mask and online softmax.
        //
        // IMPORTANT: Only valid when base_position == 0 (fresh prefill).
        // When base_position > 0, tokens must also attend to previously cached
        // KV entries, which requires the sequential path with KV cache reads.
        // =================================================================

        // B1. Populate KV cache from batch buffers (for subsequent decode steps).
        //     Copy all N tokens' K and V into the cache at once.
        {
            let kv_k = &mut kv_cache.k_cache[attn_idx];
            let kv_v = &mut kv_cache.v_cache[attn_idx];

            for t in 0..n_tokens {
                let position = base_position + t;
                for h in 0..n_kv_heads {
                    let src_off = t * kv_size + h * head_dim;
                    let dst_off = h * kv_cap * head_dim + position * head_dim;

                    let k_src = prefill.attn_k_roped_batch.slice(src_off..src_off + head_dim);
                    let mut k_dst = kv_k.slice_mut(dst_off..dst_off + head_dim);
                    dev.memcpy_dtod(&k_src, &mut k_dst)
                        .map_err(|e| candle_core::Error::Msg(
                            format!("flash_prefill kv_k t={t} h={h}: {e}")))?;

                    let v_src = prefill.attn_v_batch.slice(src_off..src_off + head_dim);
                    let mut v_dst = kv_v.slice_mut(dst_off..dst_off + head_dim);
                    dev.memcpy_dtod(&v_src, &mut v_dst)
                        .map_err(|e| candle_core::Error::Msg(
                            format!("flash_prefill kv_v t={t} h={h}: {e}")))?;
                }
            }
        }

        // B2. Flash Attention causal prefill: 1 kernel launch for all N tokens.
        //     Reads Q from attn_q_roped_batch, K from attn_k_roped_batch, V from attn_v_batch.
        //     Writes output to attn_q_normed_batch (reused — no longer needed after Phase A).
        //
        //     The kernel handles causal masking internally: token t attends to 0..t.
        //
        //     Note: The batch buffers have token-major layout [t, heads, head_dim]
        //     which matches the kernel's expected format.
        kernels::flash_attn_causal_prefill(
            &prefill.attn_q_roped_batch,
            &prefill.attn_k_roped_batch,
            &prefill.attn_v_batch,
            &mut prefill.attn_q_normed_batch,  // reuse as flash attn output
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
            &dev,
        )?;

        // --- Flash debug: compare flash output against per-token GQA reference ---
        if flash_debug {
            let scale = 1.0f32 / (head_dim as f32).sqrt();
            for t in 0..n_tokens {
                let position = base_position + t;
                let seq_len = position + 1;

                // Run GQA attention for this token using the already-populated KV cache.
                let q_src = prefill.attn_q_roped_batch.slice(
                    t * q_heads_size..(t + 1) * q_heads_size);
                dev.memcpy_dtod(&q_src, &mut graph.q_heads)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("flash_debug q copy t={t}: {e}")))?;

                let kv_k_view = kv_cache.k_cache[attn_idx].slice(..);
                let kv_v_view = kv_cache.v_cache[attn_idx].slice(..);

                crate::kernels::gqa_attention::raw_gqa_attention_strided(
                    &graph.q_heads.slice(..),
                    &kv_k_view,
                    &kv_v_view,
                    &mut graph.attn_result,
                    n_heads,
                    n_kv_heads,
                    seq_len,
                    head_dim,
                    scale,
                    kv_cap,
                    &dev,
                )?;

                // Download flash output and GQA output for this token, compare.
                // Copy token's flash output to a scratch CudaSlice, then download.
                dev.memcpy_dtod(
                    &prefill.attn_q_normed_batch.slice(
                        t * q_heads_size..(t + 1) * q_heads_size),
                    &mut graph.q_roped,  // reuse q_roped as dtoh scratch
                ).map_err(|e| candle_core::Error::Msg(
                    format!("flash_debug flash copy t={t}: {e}")))?;
                let flash_host: Vec<f32> = graph.q_roped.stream().clone()
                    .clone_dtoh(&graph.q_roped)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("flash_debug flash dtoh t={t}: {e}")))?;
                let gqa_host: Vec<f32> = graph.attn_result.stream().clone()
                    .clone_dtoh(&graph.attn_result)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("flash_debug gqa dtoh t={t}: {e}")))?;

                let mut max_err = 0.0f32;
                let mut max_err_idx = 0usize;
                for i in 0..q_heads_size {
                    let err = (flash_host[i] - gqa_host[i]).abs();
                    if err > max_err {
                        max_err = err;
                        max_err_idx = i;
                    }
                }
                let head_idx = max_err_idx / head_dim;
                let dim_idx = max_err_idx % head_dim;
                if max_err > 1e-3 || t < 3 {
                    eprintln!(
                        "[FLASH_DEBUG] attn_idx={attn_idx} t={t} seq_len={seq_len}: \
                         max_err={max_err:.6e} at head={head_idx} dim={dim_idx} \
                         (flash={:.6} gqa={:.6})",
                        flash_host[max_err_idx], gqa_host[max_err_idx],
                    );
                }
                if max_err > 0.1 && t < 4 {
                    // For detailed dumps, just log the max error details.
                    // The flash vs GQA output values are already printed above.
                    eprintln!(
                        "[FLASH_DEBUG]   LARGE DIVERGENCE at attn_idx={attn_idx} t={t}: \
                         flash[{max_err_idx}]={:.8} gqa[{max_err_idx}]={:.8} err={max_err:.6e}",
                        flash_host[max_err_idx], gqa_host[max_err_idx],
                    );
                }
            }
        }

        // B3-B5. Per-token: sigmoid gate + O projection + copy to hidden_batch.
        //        These are GEMV operations (inherently single-vector), so we still
        //        loop per-token. But the expensive attention is already done.
        let gated_n = n_heads * head_dim;
        for t in 0..n_tokens {
            // B3. Sigmoid gate: read attention output from flash_attn_output[t],
            //     gate from gate_heads_batch[t].
            {
                // Copy flash attn output for token t into graph.attn_result
                let attn_src = prefill.attn_q_normed_batch.slice(
                    t * q_heads_size..(t + 1) * q_heads_size);
                dev.memcpy_dtod(&attn_src, &mut graph.attn_result)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("flash_prefill attn_out copy t={t}: {e}")))?;

                let gate_src = prefill.attn_gate_heads_batch.slice(
                    t * q_heads_size..(t + 1) * q_heads_size);
                dev.memcpy_dtod(&gate_src, &mut graph.gate_heads)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("flash_prefill gate copy t={t}: {e}")))?;
            }

            kernels::raw_sigmoid_gate_mul(
                &graph.attn_result,
                &graph.gate_heads,
                &mut graph.q_roped,  // reuse as gated output buffer
                gated_n,
                &dev,
            )?;

            // B4. O projection: gated[n_heads * head_dim] -> hidden[hidden_size]
            quant_gemv(&attn_w.o_raw, &graph.q_roped.slice(..),
                &mut graph.hidden, hs, gated_n,
                attn_w.o_quant, &mut graph.q5k_bufs,
                graph.ggml_gpu_bufs.as_mut(), &dev)?;

            // B5. Copy O-projection output to hidden_batch[t]
            {
                let out_offset = t * hs;
                let mut dst = prefill.hidden_batch.slice_mut(out_offset..out_offset + hs);
                dev.memcpy_dtod(&graph.hidden, &mut dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("flash_prefill hidden_out t={t}: {e}")))?;
            }
        }
    } else {
        // =================================================================
        // Sequential fallback: per-token KV append + attention.
        // Used when CHIMERE_FLASH_PREFILL is not set, or for N=1.
        // =================================================================

        for t in 0..n_tokens {
            let position = base_position + t;

            // B1. KV cache append: copy K from k_roped_batch[t] and V from attn_v_batch[t].
            {
                let kv_k = &mut kv_cache.k_cache[attn_idx];
                let kv_v = &mut kv_cache.v_cache[attn_idx];

                for h in 0..n_kv_heads {
                    let src_off = t * kv_size + h * head_dim;
                    let dst_off = h * kv_cap * head_dim + position * head_dim;

                    let k_src = prefill.attn_k_roped_batch.slice(src_off..src_off + head_dim);
                    let mut k_dst = kv_k.slice_mut(dst_off..dst_off + head_dim);
                    dev.memcpy_dtod(&k_src, &mut k_dst)
                        .map_err(|e| candle_core::Error::Msg(
                            format!("attn_prefill v2 kv_k t={t} h={h}: {e}")))?;

                    let v_src = prefill.attn_v_batch.slice(src_off..src_off + head_dim);
                    let mut v_dst = kv_v.slice_mut(dst_off..dst_off + head_dim);
                    dev.memcpy_dtod(&v_src, &mut v_dst)
                        .map_err(|e| candle_core::Error::Msg(
                            format!("attn_prefill v2 kv_v t={t} h={h}: {e}")))?;
                }
            }

            let seq_len = position + 1;

            // B2. GQA attention: Q from q_roped_batch[t], KV from cache.
            {
                let q_src = prefill.attn_q_roped_batch.slice(
                    t * q_heads_size..(t + 1) * q_heads_size);
                dev.memcpy_dtod(&q_src, &mut graph.q_heads)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 q_roped copy t={t}: {e}")))?;
            }

            let scale = 1.0f32 / (head_dim as f32).sqrt();
            let kv_k_view = kv_cache.k_cache[attn_idx].slice(..);
            let kv_v_view = kv_cache.v_cache[attn_idx].slice(..);

            crate::kernels::gqa_attention::raw_gqa_attention_strided(
                &graph.q_heads.slice(..),
                &kv_k_view,
                &kv_v_view,
                &mut graph.attn_result,
                n_heads,
                n_kv_heads,
                seq_len,
                head_dim,
                scale,
                kv_cap,
                &dev,
            )?;

            // B3. Sigmoid gate
            {
                let gate_src = prefill.attn_gate_heads_batch.slice(
                    t * q_heads_size..(t + 1) * q_heads_size);
                dev.memcpy_dtod(&gate_src, &mut graph.gate_heads)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 gate copy t={t}: {e}")))?;
            }

            let gated_n = n_heads * head_dim;
            kernels::raw_sigmoid_gate_mul(
                &graph.attn_result,
                &graph.gate_heads,
                &mut graph.q_roped,
                gated_n,
                &dev,
            )?;

            // B4. O projection
            quant_gemv(&attn_w.o_raw, &graph.q_roped.slice(..),
                &mut graph.hidden, hs, gated_n,
                attn_w.o_quant, &mut graph.q5k_bufs,
                graph.ggml_gpu_bufs.as_mut(), &dev)?;

            // B5. Copy to hidden_batch
            {
                let out_offset = t * hs;
                let mut dst = prefill.hidden_batch.slice_mut(out_offset..out_offset + hs);
                dev.memcpy_dtod(&graph.hidden, &mut dst)
                    .map_err(|e| candle_core::Error::Msg(
                        format!("attn_prefill v2 hidden_out t={t}: {e}")))?;
            }
        }
    }

    // Set final KV cache position after all N tokens are processed.
    kv_cache.pos = base_position + n_tokens;

    Ok(())
}
