//! GDN (Gated DeltaNet) layer forward pass variants.
//!
//! Extracted from qwen35_model.rs — pure code movement, zero behavioral changes.

use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Result, Tensor};

use super::{Qwen35Model, GdnLayerQ, GdnLayerMoE};
use crate::activations::{l2_norm, rms_norm, sigmoid, silu_activation, softplus};
use crate::debug_utils::{
    debug_dump, debug_enabled, l0_dump,
};
use crate::state::GdnRecurrentState;

impl Qwen35Model {
    /// Forward pass for a GDN layer using preloaded QMatMul weights.
    ///
    /// Exact port of llama.cpp's `build_layer_attn_linear` (qwen35.cpp)
    /// + `build_delta_net_autoregressive` (delta-net-base.cpp).
    ///
    /// Operation sequence (matching llama.cpp line-by-line):
    ///
    ///  1. build_qkvz: QKV = W_qkv @ cur,  Z = W_gate @ cur
    ///  2. beta = sigmoid(W_beta @ cur),  alpha → softplus(alpha + dt_bias) * ssm_a → gate
    ///  3. Conv1d on QKV (causal, state-based sliding window)
    ///  4. SiLU on conv output
    ///  5. Split into Q, K, V;  L2-norm Q, K;  expand groups (16 → 48)
    ///  6. build_delta_net_autoregressive:
    ///       q *= 1/sqrt(S_k)
    ///       gate_exp = exp(gate)    — multiplicative decay per head
    ///       S = S * gate_exp        — decay state
    ///       s_t = S^T
    ///       sk = sum_rows(s_t * k)  — S @ k
    ///       d  = (v - sk) * beta    — delta
    ///       kd = k * d^T            — outer product (rank-1)
    ///       s_t += kd
    ///       o  = sum_rows(s_t * q)  — readout: S @ q
    ///  7. build_norm_gated: RMSNorm(output) * SiLU(z)
    ///  8. ssm_out projection + residual
    ///  9. Post-norm → FFN → residual
    pub(crate) fn forward_gdn_layer_q(
        &self,
        il: usize,
        w: &GdnLayerQ,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        let n_group = self.config.ssm_n_group;      // 16  (= num_k_heads = H_k)
        let d_state = self.config.ssm_d_state;      // 128 (= head_k_dim = head_v_dim = S_k = S_v)
        let dt_rank = self.config.ssm_dt_rank;      // 48  (= num_v_heads = H_v)
        let key_dim = n_group * d_state;             // 2048
        let value_dim = dt_rank * d_state;           // 6144 (= d_inner)
        let conv_channels = key_dim * 2 + value_dim; // 10240
        let conv_kernel = self.config.ssm_conv_kernel; // 4
        let gdn_idx = self.config.gdn_index(il).unwrap();

        // --- Tracer: activation stats at entry ---
        self.tracer.activation_stats(il, "hidden_in", hidden);
        let _trace_timer_gdn_q = if self.tracer.timing_enabled() { Some(self.tracer.timer_start()) } else { None };

        // === 1. Pre-attention norm ===
        let _t = std::time::Instant::now();
        let normed = rms_norm(hidden, &w.attn_norm, eps)?;



        if debug_enabled() && il == 0 {
            debug_dump("gdn00_normed_input", &normed);
        }

        // === 2. Input projections (build_qkvz + beta/alpha) ===
        // QKV mixed: W_qkv @ normed → [1, conv_channels=10240]
        let _t = std::time::Instant::now();
        let qkv = w.attn_qkv.forward(&normed)?;
        // Z (gate for norm_gated): W_gate @ normed → [1, value_dim=6144]
        let z = w.attn_gate.forward(&normed)?;

        // Beta: sigmoid(W_beta @ normed) → [1, dt_rank=48]
        let beta_proj = w.ssm_beta.forward(&normed)?;
        let beta = sigmoid(&beta_proj)?;

        // Alpha → gate: softplus(W_alpha @ normed + dt_bias) * ssm_a → [1, dt_rank=48]
        // gate is negative (ssm_a < 0), so exp(gate) ∈ (0, 1) = decay factor
        let alpha_proj = w.ssm_alpha.forward(&normed)?;
        let alpha_biased = (&alpha_proj + &w.ssm_dt_bias.unsqueeze(0)?)?;
        let alpha_sp = softplus(&alpha_biased)?;
        let gate_value = (&alpha_sp * &w.ssm_a.unsqueeze(0)?)?;


        if debug_enabled() && il == 0 {
            debug_dump("gdn00_qkv", &qkv);
            debug_dump("gdn00_z", &z);
        }

        // === 3. Conv1d: causal depthwise convolution with sliding window ===
        // Matches llama.cpp: concat(conv_states, qkv_transposed) → ssm_conv → silu
        let _t = std::time::Instant::now();
        let conv_state = &state.conv_states[gdn_idx];

        let qkv_col = qkv.reshape((1, conv_channels, 1))?;
        let conv_window = Tensor::cat(&[conv_state, &qkv_col], 2)?; // [1, 10240, 4]

        // Save last (kernel-1) states for next token
        let new_conv = conv_window.narrow(2, 1, conv_kernel - 1)?;
        state.conv_states[gdn_idx] = new_conv.contiguous()?;

        // Depthwise 1D conv: output[c] = sum_k(window[c,k] * kernel[c,k])
        let conv_window_2d = conv_window.squeeze(0)?;  // [10240, 4]
        let conv_out = (&conv_window_2d * &w.ssm_conv1d)? // [10240, 4]
            .sum(1)?                                       // [10240]
            .unsqueeze(0)?;                                // [1, 10240]


        // === 4. SiLU activation on conv output ===
        let _t = std::time::Instant::now();
        let conv_activated = silu_activation(&conv_out)?;


        if debug_enabled() && il < 2 {
            debug_dump(&format!("gdn{:02}_conv_out", il), &conv_out);
            debug_dump(&format!("gdn{:02}_conv_silu", il), &conv_activated);
        }

        // === 5. Split Q, K, V;  L2-norm;  expand groups ===
        let _t = std::time::Instant::now();
        // QKV layout: [Q (key_dim) | K (key_dim) | V (value_dim)]
        let q_raw = conv_activated.narrow(1, 0, key_dim)?;
        let k_raw = conv_activated.narrow(1, key_dim, key_dim)?;
        let v_raw = conv_activated.narrow(1, key_dim * 2, value_dim)?;

        // Reshape into [1, n_group=16, d_state=128]
        let q_3d = q_raw.reshape((1, n_group, d_state))?;
        let k_3d = k_raw.reshape((1, n_group, d_state))?;

        // L2-normalise Q, K along last dim (matches ggml_l2_norm)
        let q_normed = l2_norm(&q_3d, eps)?;
        let k_normed = l2_norm(&k_3d, eps)?;

        // Expand from n_group (16) to dt_rank (48) heads via repeat (tiled layout)
        let repeats = dt_rank / n_group;
        let q_expanded = q_normed.repeat(&[1, repeats, 1])?;
        let k_expanded = k_normed.repeat(&[1, repeats, 1])?;

        // V: reshape to [1, dt_rank=48, d_state=128]
        let v_3d = v_raw.reshape((1, dt_rank, d_state))?;

        // Scale q by 1/sqrt(S_k)
        let scale = 1.0 / (d_state as f64).sqrt();
        let q_scaled = (&q_expanded * scale)?;


        // === 6. DeltaNet state update ===
        let _t = std::time::Instant::now();
        let gate_exp = gate_value.exp()?;  // [1, 48]

        // Use fused CUDA kernel unless CHIMERE_NO_FUSED=1
        let use_fused = {
            use once_cell::sync::Lazy;
            // Fused CUDA kernel: correct (err <3e-8), register-based, 1KB shared (k+q only).
            // GDN -12%, attn -3.6%, total 20.4 tok/s vs 19.3 reference.
            // Disable with CHIMERE_NO_FUSED=1 for the Candle reference path.
            static USE_FUSED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_FUSED").is_err());
            *USE_FUSED
        };

        let (new_state, output) = if use_fused {
            // Fused: single CUDA kernel per layer via cudarc NVRTC
            crate::deltanet_kernel::deltanet_step_fused(
                &state.gdn_states[gdn_idx],
                &q_scaled,
                &k_expanded,
                &v_3d,
                &gate_exp,
                &beta,
            )?
        } else {
            // Reference: Candle ops (slower but numerically identical to original)
            // Must redo L2 norm, expansion, and q scaling since kernel normally handles them
            let q_normed = l2_norm(&q_3d, eps)?;
            let k_normed = l2_norm(&k_3d, eps)?;
            let repeats = dt_rank / n_group;
            let q_expanded = q_normed.repeat(&[1, repeats, 1])?;
            let k_expanded = k_normed.repeat(&[1, repeats, 1])?;
            let scale = 1.0 / (d_state as f64).sqrt();
            let q_scaled = (&q_expanded * scale)?;

            let old_state = &state.gdn_states[gdn_idx];
            let gate_4d = gate_exp.unsqueeze(2)?.unsqueeze(3)?;
            let s_decayed = old_state.broadcast_mul(&gate_4d)?;
            let s_t = s_decayed;
            let k_col = k_expanded.unsqueeze(3)?;
            let sk = s_t.transpose(2, 3)?.matmul(&k_col)?.squeeze(3)?;
            let delta_raw = (&v_3d - &sk)?;
            let beta_3d = beta.unsqueeze(2)?;
            let delta = delta_raw.broadcast_mul(&beta_3d)?;
            let k_col_outer = k_expanded.unsqueeze(3)?;
            let d_row = delta.unsqueeze(2)?;
            let kd = k_col_outer.broadcast_mul(&d_row)?;
            let s_t_new = (&s_t + &kd)?;
            let q_col = q_scaled.unsqueeze(3)?;
            let output = s_t_new.transpose(2, 3)?.matmul(&q_col)?.squeeze(3)?;
            (s_t_new, output)
        };

        state.gdn_states[gdn_idx] = new_state;


        // --- Tracer: GDN state metrics after deltanet step ---
        if self.tracer.enabled() {
            let s_frob: f32 = state.gdn_states[gdn_idx]
                .sqr().and_then(|t| t.sum_all()).and_then(|t| t.sqrt())
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.to_scalar()).unwrap_or(0.0);
            let g_mean: f32 = gate_exp.flatten_all()
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.mean_all())
                .and_then(|t| t.to_scalar()).unwrap_or(0.0);
            let b_mean: f32 = beta.flatten_all()
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.mean_all())
                .and_then(|t| t.to_scalar()).unwrap_or(0.0);
            self.tracer.gdn_state_trace(il, s_frob, g_mean, b_mean);
        }

        // === 7. build_norm_gated: RMSNorm(output) * SiLU(z) ===
        let _t = std::time::Instant::now();
        let output_4d = output.reshape((1, dt_rank, d_state))?; // [1, 48, 128]
        let normed_out = rms_norm(&output_4d, &w.ssm_norm, eps)?;
        let normed_flat = normed_out.reshape((1, value_dim))?;   // [1, 6144]
        let gated = (&normed_flat * &silu_activation(&z)?)?;     // [1, 6144]


        // === 8. ssm_out projection + residual ===
        let _t = std::time::Instant::now();
        let projected = w.ssm_out.forward(&gated)?;              // [1, 5120]
        let h_mid = (hidden + &projected)?;


        if debug_enabled() && il == 0 {
            debug_dump("gdn00_gated", &gated);
            debug_dump("gdn00_ssm_out", &projected);
        }

        // === 9. Post-attention norm → FFN → residual ===
        let _t = std::time::Instant::now();
        let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;
        let ffn_out = self.swiglu_ffn_q(&normed_ffn, &w.ffn_gate, &w.ffn_up, &w.ffn_down)?;
        let h_out = (&h_mid + &ffn_out)?;


        if debug_enabled() && il == 0 {
            debug_dump("gdn00_h_mid", &h_mid);
            debug_dump("gdn00_ffn_out", &ffn_out);
        }

        // --- Tracer: layer delta at exit + timing ---
        self.tracer.layer_delta(il, hidden, &h_out);
        if let Some(t) = _trace_timer_gdn_q {
            self.tracer.timer_log(il, "gdn_layer_total", t);
        }

        Ok(h_out)
    }

    /// Forward pass for a GDN MoE layer — same GDN recurrence as dense,
    /// but the FFN is replaced by MoE (top-K expert routing + shared expert).
    pub(crate) fn forward_gdn_layer_moe(
        &self,
        il: usize,
        w: &GdnLayerMoE,
        hidden: &Tensor,
        eps: f64,
        state: &mut GdnRecurrentState,
    ) -> Result<Tensor> {
        // Reuse the dense GDN forward for the SSM part by constructing a
        // temporary GdnLayerQ-compatible reference. The SSM fields are identical.
        // We only diverge at the FFN step.

        let n_group = self.config.ssm_n_group;
        let d_state = self.config.ssm_d_state;
        let dt_rank = self.config.ssm_dt_rank;
        let key_dim = n_group * d_state;
        let value_dim = dt_rank * d_state;
        let conv_channels = key_dim * 2 + value_dim;
        let conv_kernel = self.config.ssm_conv_kernel;
        let gdn_idx = self.config.gdn_index(il).unwrap();

        let _gdn_start = std::time::Instant::now();

        // --- Tracer: activation stats at entry ---
        self.tracer.activation_stats(il, "hidden_in", hidden);
        let _trace_timer_ssm = if self.tracer.timing_enabled() { Some(self.tracer.timer_start()) } else { None };

        // === 1-6: GDN SSM ===
        // Candle ops count: rms_norm(1) + QMatMul×5(5) = 6
        crate::candle_counter::tick_n(6);
        let _gt = std::time::Instant::now();
        let normed = rms_norm(hidden, &w.attn_norm, eps)?;
        // Helper: if raw Q5_K bytes are available, use the custom GEMV kernel (GEMV only —
        // single token path). `raw` is a flat U8 tensor; `inp` must be [1, in_f] F32.
        // Returns [1, out_f] to match QMatMul::forward output shape.
        // Q5_K GEMV disabled pending optimization — currently slower than Candle QMatMul.
        // Toggle with CHIMERE_Q5K_GEMV=1 to enable (individual projections).
        let use_q5k = {
            use once_cell::sync::Lazy;
            static Q5K: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_Q5K_GEMV").is_ok());
            *Q5K
        };
        // Fused dual SSM projection: CHIMERE_FUSED_SSM_PROJ=1 runs qkv+gate in
        // a single kernel launch sharing one Q8_1 quantization pass on the input.
        // Saves 1 launch + 1 input read per GDN layer (~4.5us + cache benefit).
        let use_fused_ssm_proj = {
            use once_cell::sync::Lazy;
            static FSP: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FUSED_SSM_PROJ").is_ok());
            *FSP
        };
        // Raw QMatMul: CHIMERE_RAW_QMATMUL=1 uses pre-allocated buffers for the
        // Candle quantized GEMV kernel. Eliminates ~2 cudaMalloc per QMatMul call.
        // 3 projections × 30 GDN layers = ~90 cudaMalloc/token saved.
        let use_raw_qmatmul = {
            use once_cell::sync::Lazy;
            static RQ: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_RAW_QMATMUL").is_ok());
            *RQ
        };
        // ggml Q5_K MMVQ: use ggml's kernel from the cubin for numerical parity
        // with llama.cpp. Toggle: CHIMERE_GGML_Q5K=1
        let use_ggml_q5k = {
            use once_cell::sync::Lazy;
            static GGML: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
            *GGML
        };
        // Q5_K CPU validation: pure-Rust Q5_K GEMV for attn_qkv on layer 0 only.
        // Compares ggml-compatible CPU result vs Candle QMatMul to validate correctness.
        // Toggle: CHIMERE_GGML_Q5K_CPU=1. Very slow (~300ms per call), validation only.
        let use_ggml_q5k_cpu = crate::ggml_backend::is_q5k_cpu_enabled();
        let gemv_q5k_or_qmm = |raw: &Option<Tensor>, qmm: &QMatMul,
                                inp: &Tensor, out_f: usize, in_f: usize| -> Result<Tensor> {
            if use_q5k {
                if let Some(raw_t) = raw {
                    // ggml Q5_K path: cubin kernel with numerical parity
                    if use_ggml_q5k {
                        use candle_core::Storage;
                        let inp_1d = inp.flatten_all()?;
                        let inp_c = inp_1d.contiguous()?;
                        let raw_c = raw_t.contiguous()?;

                        let (w_stor, w_lay) = raw_c.storage_and_layout();
                        let w_cuda = match &*w_stor {
                            Storage::Cuda(c) => c,
                            _ => candle_core::bail!("ggml_q5k: weight not CUDA"),
                        };
                        let w_view = w_cuda.as_cuda_slice::<u8>()?.slice(
                            w_lay.start_offset()..w_lay.start_offset() + raw_c.elem_count(),
                        );

                        let (i_stor, i_lay) = inp_c.storage_and_layout();
                        let i_cuda = match &*i_stor {
                            Storage::Cuda(c) => c,
                            _ => candle_core::bail!("ggml_q5k: input not CUDA"),
                        };
                        let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(
                            i_lay.start_offset()..i_lay.start_offset() + inp_c.elem_count(),
                        );

                        let Device::Cuda(cuda_dev) = inp_c.device() else {
                            candle_core::bail!("ggml_q5k requires CUDA");
                        };

                        // Allocate temp Q8_1 + output buffers (per-call; for zero-alloc
                        // path use the raw_qmatmul branch with GgmlQ5KBuffers)
                        let ncols_padded = (in_f + 511) / 512 * 512;
                        let q8_size = (ncols_padded / 32) * 36;
                        let mut q8_buf = cuda_dev.alloc_zeros::<u8>(q8_size)
                            .map_err(|e| candle_core::Error::Msg(
                                format!("ggml q8 alloc: {e}"),
                            ))?;
                        let mut out_buf = cuda_dev.alloc_zeros::<f32>(out_f)
                            .map_err(|e| candle_core::Error::Msg(
                                format!("ggml out alloc: {e}"),
                            ))?;

                        crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv_f32(
                            &w_view, &i_view, &mut q8_buf, &mut out_buf,
                            out_f, in_f, cuda_dev,
                        )?;

                        drop(w_stor);
                        drop(i_stor);

                        let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                            out_buf, cuda_dev.clone(),
                        );
                        return Ok(Tensor::from_storage(
                            Storage::Cuda(storage),
                            candle_core::Shape::from_dims(&[1, out_f]),
                            candle_core::op::BackpropOp::none(),
                            false,
                        ));
                    }

                    let inp_1d = inp.flatten_all()?;
                    return crate::deltanet_kernel::gemv_q5k_q8_from_tensor(
                        raw_t, &inp_1d, out_f, in_f, inp.device()
                    )?.unsqueeze(0);
                }
            }
            qmm.forward(inp)
        };

        let _gt = std::time::Instant::now();
        let conv_channels_q5k = key_dim * 2 + value_dim; // 8192 for 35B
        let hidden_size = hidden.dim(1)?;
        let (qkv, z) = if use_raw_qmatmul
            && w.attn_qkv_raw.is_some() && w.attn_gate_raw.is_some()
            && state.raw_moe_buffers.as_ref().map_or(false, |b| b.qmatmul_bufs.is_some())
        {
            // Raw QMatMul path: quantize normed to Q8_1 once, reuse for qkv + gate.
            // 0 cudaMalloc (uses pre-allocated buffers from RawGpuBuffers).
            use candle_core::Storage;
            let normed_c = normed.contiguous()?;

            // Extract CudaView from normed input
            let (n_stor, n_lay) = normed_c.storage_and_layout();
            let n_cuda = match &*n_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("normed not CUDA") };
            let n_view = n_cuda.as_cuda_slice::<f32>()?.slice(n_lay.start_offset()..n_lay.start_offset() + normed_c.elem_count());

            // Extract CudaSlice from qkv raw bytes
            let qkv_raw_t = w.attn_qkv_raw.as_ref().unwrap();
            let (qr_stor, qr_lay) = qkv_raw_t.storage_and_layout();
            let qr_cuda = match &*qr_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("qkv_raw not CUDA") };
            let qkv_raw_slice = qr_cuda.as_cuda_slice::<u8>()?.slice(qr_lay.start_offset()..qr_lay.start_offset() + qkv_raw_t.elem_count());

            // Extract CudaSlice from gate raw bytes
            let gate_raw_t = w.attn_gate_raw.as_ref().unwrap();
            let (gr_stor, gr_lay) = gate_raw_t.storage_and_layout();
            let gr_cuda = match &*gr_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("gate_raw not CUDA") };
            let gate_raw_slice = gr_cuda.as_cuda_slice::<u8>()?.slice(gr_lay.start_offset()..gr_lay.start_offset() + gate_raw_t.elem_count());

            let Device::Cuda(cuda_dev) = normed_c.device() else {
                candle_core::bail!("raw qmatmul requires CUDA");
            };

            // Borrow qmatmul buffers
            let bufs = state.raw_moe_buffers.as_mut().unwrap();
            let qm = bufs.qmatmul_bufs.as_mut().unwrap();

            if use_ggml_q5k {
                // ggml Q5_K MMVQ path: quantize to ggml's Q8_1 layout, then
                // use the ggml vec_dot kernel from the cubin. Numerical parity
                // with llama.cpp. The Q8_1 layout differs from Candle's, so we
                // use separate temp buffers (reuses qm.q8_input / qm.output
                // which are large enough).

                // ggml quantize_q8_1: input → qm.q8_input (ggml Q8_1 layout)
                crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
                    &n_view, &mut qm.q8_input, hidden_size, cuda_dev,
                )?;

                // QKV projection
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                    &qkv_raw_slice, &qm.q8_input, &mut qm.output,
                    conv_channels_q5k, hidden_size, cuda_dev,
                )?;
                let mut qkv_owned = cuda_dev.alloc_zeros::<f32>(conv_channels_q5k)
                    .map_err(|e| candle_core::Error::Msg(format!("alloc qkv output: {e}")))?;
                {
                    let src_view = qm.output.slice(..conv_channels_q5k);
                    cuda_dev.memcpy_dtod(&src_view, &mut qkv_owned)
                        .map_err(|e| candle_core::Error::Msg(format!("copy qkv output: {e}")))?;
                }
                let qkv_t = Tensor::from_storage(
                    Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                        qkv_owned, cuda_dev.clone())),
                    candle_core::Shape::from_dims(&[1, conv_channels_q5k]),
                    candle_core::op::BackpropOp::none(), false);

                // Gate projection (reuses the ggml Q8_1 input)
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                    &gate_raw_slice, &qm.q8_input, &mut qm.output,
                    value_dim, hidden_size, cuda_dev,
                )?;
                let mut z_owned = cuda_dev.alloc_zeros::<f32>(value_dim)
                    .map_err(|e| candle_core::Error::Msg(format!("alloc gate output: {e}")))?;
                {
                    let src_view = qm.output.slice(..value_dim);
                    cuda_dev.memcpy_dtod(&src_view, &mut z_owned)
                        .map_err(|e| candle_core::Error::Msg(format!("copy gate output: {e}")))?;
                }
                let z_t = Tensor::from_storage(
                    Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                        z_owned, cuda_dev.clone())),
                    candle_core::Shape::from_dims(&[1, value_dim]),
                    candle_core::op::BackpropOp::none(), false);

                // Drop storage guards before further use
                drop(n_stor);
                drop(qr_stor);
                drop(gr_stor);
                (qkv_t, z_t)
            } else {

            // Quantize normed to Q8_1 (once, reused for both projections)
            crate::kernels::raw_qmatmul::raw_quantize_q8_1(
                &n_view, qm, hidden_size, cuda_dev,
            )?;

            // QKV projection: [conv_channels, hidden_size] × normed → [conv_channels]
            crate::kernels::raw_qmatmul::raw_q5k_gemv_candle(
                &qkv_raw_slice, qm, hidden_size, conv_channels_q5k, cuda_dev,
            )?;
            // Copy output to owned CudaSlice for Tensor wrapping (1 alloc, unavoidable)
            let mut qkv_owned = cuda_dev.alloc_zeros::<f32>(conv_channels_q5k)
                .map_err(|e| candle_core::Error::Msg(format!("alloc qkv output: {e}")))?;
            {
                let src_view = qm.output.slice(..conv_channels_q5k);
                cuda_dev.memcpy_dtod(&src_view, &mut qkv_owned)
                    .map_err(|e| candle_core::Error::Msg(format!("copy qkv output: {e}")))?;
            }
            let qkv_t = Tensor::from_storage(
                Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    qkv_owned, cuda_dev.clone())),
                candle_core::Shape::from_dims(&[1, conv_channels_q5k]),
                candle_core::op::BackpropOp::none(), false);

            // Gate projection: [value_dim, hidden_size] × normed → [value_dim]
            // Q8_1 input is still valid (same normed), just run GEMV with different weights
            crate::kernels::raw_qmatmul::raw_q5k_gemv_candle(
                &gate_raw_slice, qm, hidden_size, value_dim, cuda_dev,
            )?;
            let mut z_owned = cuda_dev.alloc_zeros::<f32>(value_dim)
                .map_err(|e| candle_core::Error::Msg(format!("alloc gate output: {e}")))?;
            {
                let src_view = qm.output.slice(..value_dim);
                cuda_dev.memcpy_dtod(&src_view, &mut z_owned)
                    .map_err(|e| candle_core::Error::Msg(format!("copy gate output: {e}")))?;
            }
            let z_t = Tensor::from_storage(
                Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    z_owned, cuda_dev.clone())),
                candle_core::Shape::from_dims(&[1, value_dim]),
                candle_core::op::BackpropOp::none(), false);

            // Drop storage guards before further use
            drop(n_stor);
            drop(qr_stor);
            drop(gr_stor);
            (qkv_t, z_t)
            } // close else (Candle raw_qmatmul)
        } else if use_fused_ssm_proj {
            if let (Some(ref qkv_raw), Some(ref gate_raw)) = (&w.attn_qkv_raw, &w.attn_gate_raw) {
                let inp_1d = normed.flatten_all()?;
                let (qkv_1d, z_1d) = crate::deltanet_kernel::gemv_q5k_q8_dual_from_tensor(
                    qkv_raw, gate_raw, &inp_1d,
                    conv_channels_q5k, value_dim,
                    hidden.dim(1)?, normed.device(),
                )?;
                (qkv_1d.unsqueeze(0)?, z_1d.unsqueeze(0)?)
            } else {
                // Fallback: weights are not Q5_K, use QMatMul
                let qkv = w.attn_qkv.forward(&normed)?;
                let z = w.attn_gate.forward(&normed)?;
                (qkv, z)
            }
        } else {
            let qkv = gemv_q5k_or_qmm(
                &w.attn_qkv_raw, &w.attn_qkv, &normed, conv_channels_q5k,
                hidden.dim(1)?)?;
            let z = gemv_q5k_or_qmm(
                &w.attn_gate_raw, &w.attn_gate, &normed, value_dim,
                hidden.dim(1)?)?;
            (qkv, z)
        };

        // --- Q5_K CPU validation: layer 0 attn_qkv only ---
        // When CHIMERE_GGML_Q5K_CPU=1, run the pure-Rust Q5_K GEMV on CPU and
        // compare against whichever GPU path produced `qkv`. This validates that
        // our Rust Q5_K dot product matches Candle/ggml. Only runs on GDN layer 0
        // to avoid slowing down the full model.
        let qkv = if use_ggml_q5k_cpu && il == 0 {
            if let Some(ref raw_t) = w.attn_qkv_raw {
                let result_ggml = crate::ggml_backend::ggml_q5k_forward_cpu(
                    &normed, raw_t,
                    conv_channels_q5k, hidden_size,
                    hidden.device(),
                )?;
                // Compare ggml CPU result vs GPU path (Candle QMatMul or GPU Q5_K kernel)
                let _ = crate::ggml_backend::compare_tensors(
                    "Q5K_attn_qkv_L0", &result_ggml, &qkv,
                );
                // Use the ggml CPU result for this layer (validation mode)
                result_ggml
            } else {
                eprintln!(
                    "[GGML_Q5K_CPU] WARNING: layer 0 attn_qkv_raw is None \
                     (weight is not Q5_K). Skipping CPU validation."
                );
                qkv
            }
        } else {
            qkv
        };

        // Fused elementwise toggle: CHIMERE_FUSED_ELEM=1 uses custom CUDA kernels
        // for beta+alpha+gate and rms_norm+silu_gate, saving ~5 kernel launches/layer.
        let use_fused_elem = {
            use once_cell::sync::Lazy;
            static FE: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FUSED_ELEM").is_ok());
            *FE
        };

        let _gt = std::time::Instant::now();
        let beta_proj = w.ssm_beta.forward(&normed)?;
        let alpha_proj = w.ssm_alpha.forward(&normed)?;
        let (beta, gate_exp) = if use_fused_elem {
            // Fused path: single kernel for sigmoid(beta) + exp(softplus(alpha+bias)*a)
            // Candle ops: 0 (custom CUDA kernel)
            let (b, ge) = crate::deltanet_kernel::fused_beta_alpha_gate_tensor(
                &beta_proj, &alpha_proj, &w.ssm_dt_bias, &w.ssm_a,
            )?;
            (b, ge)
        } else {
            // Reference path: separate Candle ops
            // Candle ops: sigmoid(1) + unsqueeze(1) + add(1) + exp(1) + log(1) + mul(1) + unsqueeze(1) + mul(1) + exp(1) = 9
            crate::candle_counter::tick_n(9);
            let beta = sigmoid(&beta_proj)?;
            let alpha_biased = alpha_proj.broadcast_add(&w.ssm_dt_bias.unsqueeze(0)?)?;
            let alpha_sp = softplus(&alpha_biased)?;
            let gate_value = alpha_sp.broadcast_mul(&w.ssm_a.unsqueeze(0)?)?;
            let gate_exp = gate_value.exp()?;
            (beta, gate_exp)
        };
        // Raw SSM toggle: CHIMERE_RAW_SSM=1 uses fused CUDA kernels for conv1d+split+norm+expand.
        // Saves 27 Candle tensor ops per GDN layer (7 kernel launches vs 27 Candle dispatch calls).
        let use_raw_ssm = {
            use once_cell::sync::Lazy;
            static RAW_SSM: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_RAW_SSM").is_ok());
            *RAW_SSM
        };

        let _gt = std::time::Instant::now();
        let (q_scaled, k_expanded, v_3d) = if use_raw_ssm && matches!(hidden.device(), Device::Cuda(_)) {
            // Raw path: use pre-allocated scratch buffers if available (0 cudaMalloc)
            let mut bufs = state.raw_moe_buffers.take();
            let result = if let Some(ref mut b) = bufs {
                if let Some(ref mut scratch) = b.gdn_scratch {
                    crate::kernels::elementwise::fused_conv_split_norm_expand_with_scratch(
                        &qkv, &state.conv_states[gdn_idx], &w.ssm_conv1d,
                        key_dim, value_dim, conv_channels, conv_kernel,
                        n_group, d_state, dt_rank, eps, scratch,
                    )
                } else {
                    crate::kernels::elementwise::fused_conv_split_norm_expand_tensor(
                        &qkv, &state.conv_states[gdn_idx], &w.ssm_conv1d,
                        key_dim, value_dim, conv_channels, conv_kernel,
                        n_group, d_state, dt_rank, eps,
                    )
                }
            } else {
                crate::kernels::elementwise::fused_conv_split_norm_expand_tensor(
                    &qkv, &state.conv_states[gdn_idx], &w.ssm_conv1d,
                    key_dim, value_dim, conv_channels, conv_kernel,
                    n_group, d_state, dt_rank, eps,
                )
            };
            state.raw_moe_buffers = bufs;
            let (qs, ke, v3, new_conv) = result?;
            state.conv_states[gdn_idx] = new_conv;
            (qs, ke, v3)
        } else {
            // Candle ops: conv1d(9) + split_norm_expand(18) = 27
            crate::candle_counter::tick_n(27);
            let conv_state = &state.conv_states[gdn_idx];
            let qkv_col = qkv.reshape((1, conv_channels, 1))?;
            let conv_window = Tensor::cat(&[conv_state, &qkv_col], 2)?;
            let new_conv = conv_window.narrow(2, 1, conv_kernel - 1)?;
            state.conv_states[gdn_idx] = new_conv.contiguous()?;
            let conv_window_2d = conv_window.squeeze(0)?;
            let conv_out = (&conv_window_2d * &w.ssm_conv1d)?.sum(1)?.unsqueeze(0)?;
            let conv_activated = conv_out.silu()?;
            let q_raw = conv_activated.narrow(1, 0, key_dim)?;
            let k_raw = conv_activated.narrow(1, key_dim, key_dim)?;
            let v_raw = conv_activated.narrow(1, key_dim * 2, value_dim)?;
            let q_3d = q_raw.reshape((1, n_group, d_state))?;
            let k_3d = k_raw.reshape((1, n_group, d_state))?;
            let q_normed = l2_norm(&q_3d, eps)?;
            let k_normed = l2_norm(&k_3d, eps)?;
            let repeats = dt_rank / n_group;
            let q_expanded = q_normed.repeat(&[1, repeats, 1])?;
            let k_expanded = k_normed.repeat(&[1, repeats, 1])?;
            let v_3d = v_raw.reshape((1, dt_rank, d_state))?;
            let scale = 1.0 / (d_state as f64).sqrt();
            let q_scaled = (&q_expanded * scale)?;
            (q_scaled, k_expanded, v_3d)
        };

        if il == 0 { l0_dump("04_q_scaled", &q_scaled); l0_dump("04_k_expanded", &k_expanded); l0_dump("04_v_3d", &v_3d); l0_dump("03_beta", &beta); l0_dump("03_gate_exp", &gate_exp); }
        // gate_exp is already computed in the beta/alpha/gate branch above.
        // Use fused kernel or reference for DeltaNet step
        let use_fused = {
            use once_cell::sync::Lazy;
            static USE_FUSED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_FUSED").is_err());
            *USE_FUSED
        };
        let (new_state, output) = if use_fused {
            // Candle ops: 0 (custom CUDA kernel)
            crate::deltanet_kernel::deltanet_step_fused(
                &state.gdn_states[gdn_idx], &q_scaled, &k_expanded, &v_3d, &gate_exp, &beta,
            )?
        } else {
            // Candle ops: ~15 (unsqueeze×4, broadcast_mul×3, transpose×2, matmul×2, squeeze×2, sub, add)
            crate::candle_counter::tick_n(15);
            let old_s = &state.gdn_states[gdn_idx];
            let g4 = gate_exp.unsqueeze(2)?.unsqueeze(3)?;
            let sd = old_s.broadcast_mul(&g4)?;
            let kc = k_expanded.unsqueeze(3)?;
            let sk = sd.transpose(2, 3)?.matmul(&kc)?.squeeze(3)?;
            let dr = (&v_3d - &sk)?;
            let b3 = beta.unsqueeze(2)?;
            let d = dr.broadcast_mul(&b3)?;
            let ko = k_expanded.unsqueeze(3)?;
            let drow = d.unsqueeze(2)?;
            let kd = ko.broadcast_mul(&drow)?;
            let sn = (&sd + &kd)?;
            let qc = q_scaled.unsqueeze(3)?;
            let o = sn.transpose(2, 3)?.matmul(&qc)?.squeeze(3)?;
            (sn, o)
        };
        if il == 0 { l0_dump("05_deltanet_output", &output); }
        state.gdn_states[gdn_idx] = new_state;

        // --- Tracer: GDN state metrics after deltanet step ---
        if self.tracer.enabled() {
            let s_frob: f32 = state.gdn_states[gdn_idx]
                .sqr().and_then(|t| t.sum_all()).and_then(|t| t.sqrt())
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.to_scalar()).unwrap_or(0.0);
            let g_mean: f32 = gate_exp.flatten_all()
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.mean_all())
                .and_then(|t| t.to_scalar()).unwrap_or(0.0);
            let b_mean: f32 = beta.flatten_all()
                .and_then(|t| t.to_device(&Device::Cpu))
                .and_then(|t| t.mean_all())
                .and_then(|t| t.to_scalar()).unwrap_or(0.0);
            self.tracer.gdn_state_trace(il, s_frob, g_mean, b_mean);
        }
        if let Some(t) = _trace_timer_ssm {
            self.tracer.timer_log(il, "ssm_total", t);
        }

        // === 7. Norm-gated output ===
        let _gt = std::time::Instant::now();
        let output_4d = output.reshape((1, dt_rank, d_state))?;
        let gated = if use_fused_elem {
            // Candle ops: 1 (reshape only)
            crate::candle_counter::tick_n(1);
            // Fused path: single kernel for rms_norm + silu(z) * normed
            crate::deltanet_kernel::fused_rms_norm_silu_gate_tensor(
                &output_4d, &w.ssm_norm, &z, eps,
            )?
        } else {
            // Candle ops: reshape(1) + rms_norm(1) + reshape(1) + silu(1) + mul(1) = 5
            crate::candle_counter::tick_n(5);
            // Reference path: separate Candle ops
            let normed_out = rms_norm(&output_4d, &w.ssm_norm, eps)?;
            let normed_flat = normed_out.reshape((1, value_dim))?;
            (&normed_flat * &z.silu()?)?
        };
        // ssm_out QMatMul(1) + residual add(1) = 2
        crate::candle_counter::tick_n(2);
        // ssm_out: [hidden_size, value_dim] → [1, hidden_size]
        let projected = if use_raw_qmatmul
            && w.ssm_out_raw.is_some()
            && state.raw_moe_buffers.as_ref().map_or(false, |b| b.qmatmul_bufs.is_some())
        {
            // Raw QMatMul path for ssm_out: quantize gated to Q8_1, then GEMV.
            use candle_core::Storage;
            let gated_c = gated.contiguous()?;
            let (gc_stor, gc_lay) = gated_c.storage_and_layout();
            let gc_cuda = match &*gc_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("gated not CUDA") };
            let gc_view = gc_cuda.as_cuda_slice::<f32>()?.slice(gc_lay.start_offset()..gc_lay.start_offset() + gated_c.elem_count());

            let ssm_raw_t = w.ssm_out_raw.as_ref().unwrap();
            let (sr_stor, sr_lay) = ssm_raw_t.storage_and_layout();
            let sr_cuda = match &*sr_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("ssm_out_raw not CUDA") };
            let ssm_raw_slice = sr_cuda.as_cuda_slice::<u8>()?.slice(sr_lay.start_offset()..sr_lay.start_offset() + ssm_raw_t.elem_count());

            let Device::Cuda(cuda_dev) = gated_c.device() else {
                candle_core::bail!("raw qmatmul requires CUDA");
            };

            let bufs = state.raw_moe_buffers.as_mut().unwrap();
            let qm = bufs.qmatmul_bufs.as_mut().unwrap();

            // Quantize gated to Q8_1 (input dim = value_dim = 4096)
            crate::kernels::raw_qmatmul::raw_quantize_q8_1(
                &gc_view, qm, value_dim, cuda_dev,
            )?;

            // GEMV: [hidden_size, value_dim] × gated_q8 → [hidden_size]
            crate::kernels::raw_qmatmul::raw_q5k_gemv_candle(
                &ssm_raw_slice, qm, value_dim, hidden_size, cuda_dev,
            )?;
            // Copy output to owned CudaSlice for Tensor wrapping
            let mut out_owned = cuda_dev.alloc_zeros::<f32>(hidden_size)
                .map_err(|e| candle_core::Error::Msg(format!("alloc ssm_out output: {e}")))?;
            {
                let src_view = qm.output.slice(..hidden_size);
                cuda_dev.memcpy_dtod(&src_view, &mut out_owned)
                    .map_err(|e| candle_core::Error::Msg(format!("copy ssm_out output: {e}")))?;
            }
            let proj_t = Tensor::from_storage(
                Storage::Cuda(candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    out_owned, cuda_dev.clone())),
                candle_core::Shape::from_dims(&[1, hidden_size]),
                candle_core::op::BackpropOp::none(), false);
            drop(gc_stor);
            drop(sr_stor);
            proj_t
        } else {
            gemv_q5k_or_qmm(
                &w.ssm_out_raw, &w.ssm_out, &gated, hidden.dim(1)?, value_dim)?
        };
        let h_mid = (hidden + &projected)?;
        if il == 0 { l0_dump("06_gated", &gated); l0_dump("07_ssm_projected", &projected); l0_dump("08_h_mid", &h_mid); }
        // === 8. MoE FFN — with optional dynamic skip ===
        // CHIMERE_FFN_SKIP=<threshold> skips MoE FFN when SSM residual is small.
        // The SSM projected norm relative to hidden norm determines skip.
        let ffn_skip_thresh = {
            use once_cell::sync::Lazy;
            static FST: Lazy<Option<f32>> = Lazy::new(|| {
                std::env::var("CHIMERE_FFN_SKIP").ok().and_then(|s| s.parse().ok())
            });
            *FST
        };

        let skip_ffn = if let Some(thresh) = ffn_skip_thresh {
            // Quick ratio check: ||projected||² / ||hidden||² (GPU, 2 ops)
            let proj_sq: f32 = projected.sqr()?.sum_all()?.to_scalar()?;
            let h_sq: f32 = hidden.sqr()?.sum_all()?.to_scalar()?;
            let ratio = (proj_sq / (h_sq + 1e-10)).sqrt();
            if ratio < thresh && il >= 8 {
                // Don't skip early layers (0-7) — they're always active
                true
            } else {
                false
            }
        } else {
            false
        };

        let h_out = if skip_ffn {
            // Skip MoE FFN entirely — h_mid IS the output
            h_mid
        } else {
            // Candle ops: rms_norm(1) + residual(1) = 2 (shared between both paths)
            crate::candle_counter::tick_n(2);
            let _gt = std::time::Instant::now();
            let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;

            let use_raw_moe = {
                use once_cell::sync::Lazy;
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
                        w.moe.gate_exps_shape.1, // expert_ffn
                        hidden.dim(1)?,           // hidden_size
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
                let moe_refs = rw.gdn[gdn_idx].moe_refs();
                let cudarc_bufs = state.moe_cudarc_bufs.as_mut().unwrap();
                let Device::Cuda(cuda_dev) = &self.device else {
                    candle_core::bail!("cudarc MoE requires CUDA");
                };
                // Extract normed_ffn as CudaView (1 storage_and_layout)
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
                crate::candle_counter::tick_n(3);
                // Try v2 (zero storage_and_layout for expert weights + raw router GEMV)
                if let Some(ref rw) = self.raw_weights {
                    let moe_refs = rw.gdn[gdn_idx].moe_refs();
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
                crate::candle_counter::tick_n(45);
                Self::moe_ffn_forward(&normed_ffn, &w.moe, self.config.experts_per_token)?
            };

            let h_out = (&h_mid + &ffn_out)?;
            h_out
        };

        // --- Tracer: layer delta at exit ---
        self.tracer.layer_delta(il, hidden, &h_out);

        Ok(h_out)
    }
}

// =====================================================================
// Pure cudarc GDN forward — free function (zero Candle Tensor operations)
// =====================================================================

/// GDN layer forward in pure cudarc. Zero Candle allocs.
///
/// Equivalent to `forward_gdn_layer_moe` SSM path (steps 1-17) but using
/// only `CudaSlice` operations -- no Candle Tensor creation at any point.
///
/// # Operation sequence
///
///  1. RMSNorm(hidden -> normed)                 -- using graph.normed
///  2. Quantize normed to Q8_1                   -- once, reused for 4 projections
///  3. Q5_K GEMV: w_qkv @ normed -> gdn_proj    -- [conv_channels]
///  4. Q5_K GEMV: w_gate @ normed -> gdn_gate   -- [value_dim]
///  5. Q5_K GEMV: w_beta @ normed -> beta_proj   -- [dt_rank]
///  6. Q5_K GEMV: w_alpha @ normed -> alpha_proj -- [dt_rank]
///  7. Fused beta/alpha/gate                      -- sigmoid + exp(softplus*a)
///  8. Fused conv1d + SiLU + state update
///  9. Split Q/K/V from conv output               -- memcpy_dtod (zero-copy views)
/// 10. L2 norm Q, K per group
/// 11. Expand groups (16 -> dt_rank heads)
/// 12. Scale Q by 1/sqrt(d_state)
/// 13. DeltaNet recurrent step                    -- existing fused kernel
/// 14. Fused RMSNorm + SiLU gate
/// 15. Quantize gated to Q8_1
/// 16. Q5_K GEMV: w_ssm_out @ gated -> gdn_out  -- [hidden_size]
/// 17. Residual: hidden += gdn_out
///
/// # Arguments
///
/// - `il`: layer index (global, 0-based)
/// - `gdn_w`: raw quantized weight bytes for this GDN layer
/// - `gdn_state`: per-layer DeltaNet recurrent state `[dt_rank * d_state * d_state]`
/// - `graph`: pre-allocated scratch buffers (also holds hidden/normed/gdn_proj/gdn_gate/gdn_out)
///
/// # Post-conditions
///
/// - `graph.hidden` has been updated with the SSM residual (step 17).
/// - `gdn_state` has been updated with the new DeltaNet state.
/// - The per-layer conv state (inside `graph.gdn_cudarc`) has been updated.
/// - The caller is responsible for running MoE FFN (steps 18-20):
///   RMSNorm(hidden, post_norm) -> normed_ffn, then MoE FFN, then residual.
pub(crate) fn forward_gdn_cudarc(
    il: usize,
    gdn_w: &super::compute_graph::GdnWeightsRaw,
    gdn_state: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    graph: &mut super::compute_graph::ComputeGraph,
) -> candle_core::Result<()> {
    forward_gdn_cudarc_with_cache(il, gdn_w, gdn_state, graph, None)
}

/// GDN cudarc forward with optional cached weight pointers.
///
/// When `cached` is Some, uses pre-extracted raw CUDA pointers for weight
/// tensors, avoiding ~1-3µs per `device_ptr()` call (5 weight accesses per
/// GDN layer × 30 layers = 150 calls/token saved).
pub(crate) fn forward_gdn_cudarc_with_cache(
    il: usize,
    gdn_w: &super::compute_graph::GdnWeightsRaw,
    gdn_state: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    graph: &mut super::compute_graph::ComputeGraph,
    cached: Option<&super::compute_graph::CachedWeightPtrs>,
) -> candle_core::Result<()> {
    let hidden_size = graph.hidden_size;
    let n_group = graph.ssm_n_group;
    let d_state = graph.ssm_d_state;
    let dt_rank = graph.ssm_dt_rank;
    let key_dim = n_group * d_state;               // 2048
    let value_dim = dt_rank * d_state;              // 4096
    let conv_channels = key_dim * 2 + value_dim;    // 8192
    let conv_kernel = graph.ssm_conv_kernel;        // 4
    let eps = 1e-6f32; // Qwen3.5 rms_norm_eps
    // Clone CudaDevice (Arc clone, cheap) to avoid borrow conflicts
    // between &graph.dev and &mut graph.gdn_cudarc.
    let dev = graph.dev.clone();

    // GDN layer index among GDN-only layers (for conv state indexing)
    let gdn_idx = super::compute_graph::gdn_index(
        il, graph.full_attn_interval,
    );

    // Lazy-init the cudarc scratch on first call
    if graph.gdn_cudarc.is_none() {
        graph.gdn_cudarc = Some(
            super::compute_graph::GdnCudarcScratch::from_graph(graph)?
        );
    }
    // Take the scratch out of graph to avoid borrow conflicts.
    // forward_gdn_cudarc needs &mut sc AND &graph.normed/&mut graph.hidden
    // simultaneously, which isn't possible through a single &mut graph.
    // The take+put pattern is safe: we always put it back before returning.
    let mut sc = graph.gdn_cudarc.take().unwrap();
    // Same take+put pattern for ggml GPU buffers (used when CHIMERE_GGML_GPU=1).
    let mut ggml = graph.ggml_gpu_bufs.take();
    let result = forward_gdn_cudarc_inner(
        il, gdn_w, gdn_state, graph, &mut sc, &mut ggml,
        hidden_size, n_group, d_state, dt_rank,
        key_dim, value_dim, conv_channels, conv_kernel, eps, gdn_idx, &dev,
        cached,
    );
    graph.ggml_gpu_bufs = ggml;
    graph.gdn_cudarc = Some(sc);
    return result;
}

/// Inner implementation of GDN cudarc forward (separated for safe take+put pattern).
fn forward_gdn_cudarc_inner(
    _il: usize,
    gdn_w: &super::compute_graph::GdnWeightsRaw,
    gdn_state: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    graph: &mut super::compute_graph::ComputeGraph,
    sc: &mut super::compute_graph::GdnCudarcScratch,
    ggml: &mut Option<crate::kernels::ggml_gpu::GgmlGpuBuffers>,
    hidden_size: usize,
    n_group: usize,
    d_state: usize,
    dt_rank: usize,
    key_dim: usize,
    value_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    eps: f32,
    gdn_idx: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
    cached: Option<&super::compute_graph::CachedWeightPtrs>,
) -> candle_core::Result<()> {
    use candle_core::cuda_backend::cudarc::driver::CudaView;
    use crate::kernels::elementwise::{
        raw_fused_beta_alpha_gate,
        raw_fused_conv1d_silu_update,
        raw_fused_rms_norm_silu_gate,
    };
    use crate::kernels::fused_ops::fused_split_norm_expand_scale;
    use crate::kernels::deltanet_step::deltanet_step_fused_raw_inplace;
    #[cfg(feature = "cubin_fallback")]
    use crate::kernels::raw_qmatmul::{raw_quantize_q8_1_to, raw_q5k_gemv_to};

    // ===================================================================
    // Step 1: RMSNorm — SKIPPED (caller runs it before entering)
    // ===================================================================

    // ===================================================================
    // Steps 2+3 fused: Quantize normed + first GEMV in one FFI call.
    // Steps 4-6: Q5_K GEMV projections (reuse Q8_1 populated by fused step 2+3)
    //
    // When CHIMERE_GGML_GPU=1: use ggml's optimized MMVQ kernels (~4x faster).
    // Otherwise: use Candle's built-in quantize_q8_1 + mul_mat_vec_q5_K_q8_1.
    // ===================================================================
    if let Some(ref mut ggml_bufs) = ggml {
        // --- ggml MMVQ path ---
        // Phase 3.1: Check for stream override (set during CUDA Graph capture).
        // When active, use _on_stream variants so ggml FFI calls are captured
        // on the same stream as cudarc kernels.
        let stream_ptr = ggml_bufs.active_stream();
        let use_stream = !stream_ptr.is_null();

        // Check if we have cached weight pointers for this GDN layer.
        // When available AND not in stream-override mode, skip the CudaView
        // creation and use raw pointers directly (saves ~1-3µs per call).
        let gdn_cache = if !use_stream {
            cached.map(|c| (
                c.gdn_ssm_in.get(gdn_idx).copied(),
                c.gdn_ssm_gate.get(gdn_idx).copied(),
                c.gdn_ssm_beta.get(gdn_idx).copied(),
                c.gdn_ssm_alpha.get(gdn_idx).copied(),
                c.gdn_ssm_out.get(gdn_idx).copied(),
            ))
        } else {
            None
        };

        // Step 2+3 fused: Quantize normed to Q8_1 + QKV projection in one FFI call.
        // The fused call populates q8_input as a side effect, so steps 4-6 reuse it.
        // Cached-ptr path has no fused variant, so falls back to separate quantize + cached GEMV.
        if let Some((Some(ssm_in_ptr), _, _, _, _)) = gdn_cache {
            // Cached path: separate quantize (to populate q8_input) + cached GEMV
            {
                let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                    &normed_view, &mut ggml_bufs.q8_input, hidden_size, dev,
                )?;
            }
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                ssm_in_ptr, &ggml_bufs.q8_input, &mut graph.gdn_proj,
                conv_channels, hidden_size,
            )?;
        } else {
            let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
            let w_view: CudaView<'_, u8> = gdn_w.ssm_in_raw.slice(..);
            if use_stream {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32_on_stream(
                    &w_view, &normed_view, &mut graph.gdn_proj,
                    &mut ggml_bufs.q8_input,
                    conv_channels, hidden_size, stream_ptr,
                )?;
            } else {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32(
                    &w_view, &normed_view, &mut graph.gdn_proj,
                    &mut ggml_bufs.q8_input,
                    conv_channels, hidden_size,
                )?;
            }
        }

        // Step 4: Gate (z) projection -> gdn_gate [value_dim=4096]
        if let Some((_, Some(ssm_gate_ptr), _, _, _)) = gdn_cache {
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                ssm_gate_ptr, &ggml_bufs.q8_input, &mut graph.gdn_gate,
                value_dim, hidden_size,
            )?;
        } else {
            let w_view: CudaView<'_, u8> = gdn_w.ssm_gate_raw.slice(..);
            if use_stream {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_on_stream(
                    &w_view, &ggml_bufs.q8_input, &mut graph.gdn_gate,
                    value_dim, hidden_size, stream_ptr,
                )?;
            } else {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                    &w_view, &ggml_bufs.q8_input, &mut graph.gdn_gate,
                    value_dim, hidden_size,
                )?;
            }
        }

        // Step 5: Beta projection -> beta_proj [dt_rank=32]
        //   Beta can be Q5_K or Q8_0 depending on custom-mix quantization.
        {
            match gdn_w.beta_quant {
                crate::gguf_loader::GgmlType::Q8_0 => {
                    // Q8_0 uses a different kernel, no cached-ptr variant yet
                    let w_view: CudaView<'_, u8> = gdn_w.ssm_beta_raw.slice(..);
                    let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
                    crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                        &w_view, &normed_view, &mut sc.beta_proj, dt_rank, hidden_size, dev)?;
                }
                _ => {
                    if let Some((_, _, Some(ssm_beta_ptr), _, _)) = gdn_cache {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                            ssm_beta_ptr, &ggml_bufs.q8_input, &mut sc.beta_proj,
                            dt_rank, hidden_size,
                        )?;
                    } else {
                        let w_view: CudaView<'_, u8> = gdn_w.ssm_beta_raw.slice(..);
                        if use_stream {
                            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_on_stream(
                                &w_view, &ggml_bufs.q8_input, &mut sc.beta_proj,
                                dt_rank, hidden_size, stream_ptr,
                            )?;
                        } else {
                            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                                &w_view, &ggml_bufs.q8_input, &mut sc.beta_proj,
                                dt_rank, hidden_size,
                            )?;
                        }
                    }
                }
            }
        }

        // Step 6: Alpha projection -> alpha_proj [dt_rank=32]
        {
            match gdn_w.alpha_quant {
                crate::gguf_loader::GgmlType::Q8_0 => {
                    // Q8_0 uses a different kernel, no cached-ptr variant yet
                    let w_view: CudaView<'_, u8> = gdn_w.ssm_alpha_raw.slice(..);
                    let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
                    crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                        &w_view, &normed_view, &mut sc.alpha_proj, dt_rank, hidden_size, dev)?;
                }
                _ => {
                    if let Some((_, _, _, Some(ssm_alpha_ptr), _)) = gdn_cache {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                            ssm_alpha_ptr, &ggml_bufs.q8_input, &mut sc.alpha_proj,
                            dt_rank, hidden_size,
                        )?;
                    } else {
                        let w_view: CudaView<'_, u8> = gdn_w.ssm_alpha_raw.slice(..);
                        if use_stream {
                            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_on_stream(
                                &w_view, &ggml_bufs.q8_input, &mut sc.alpha_proj,
                                dt_rank, hidden_size, stream_ptr,
                            )?;
                        } else {
                            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                                &w_view, &ggml_bufs.q8_input, &mut sc.alpha_proj,
                                dt_rank, hidden_size,
                            )?;
                        }
                    }
                }
            }
        }
    } else {
        // --- Candle fallback path (cubin Q5K kernels) ---
        #[cfg(feature = "cubin_fallback")]
        {
            // Step 2: Quantize normed to Q8_1
            {
                let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
                raw_quantize_q8_1_to(
                    &normed_view, &mut sc.q8_hidden, hidden_size, dev,
                )?;
            }

            // Step 3: QKV projection -> gdn_proj [conv_channels=8192]
            {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_in_raw.slice(..);
                raw_q5k_gemv_to(
                    &w_view, &sc.q8_hidden, &mut graph.gdn_proj,
                    hidden_size, conv_channels, dev,
                )?;
            }

            // Step 4: Gate (z) projection -> gdn_gate [value_dim=4096]
            {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_gate_raw.slice(..);
                raw_q5k_gemv_to(
                    &w_view, &sc.q8_hidden, &mut graph.gdn_gate,
                    hidden_size, value_dim, dev,
                )?;
            }

            // Step 5: Beta projection -> beta_proj [dt_rank=32]
            //   Beta can be Q5_K or Q8_0 depending on custom-mix quantization.
            {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_beta_raw.slice(..);
                match gdn_w.beta_quant {
                    crate::gguf_loader::GgmlType::Q8_0 => {
                        let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
                        crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                            &w_view, &normed_view, &mut sc.beta_proj, dt_rank, hidden_size, dev)?;
                    }
                    _ => {
                        raw_q5k_gemv_to(
                            &w_view, &sc.q8_hidden, &mut sc.beta_proj,
                            hidden_size, dt_rank, dev,
                        )?;
                    }
                }
            }

            // Step 6: Alpha projection -> alpha_proj [dt_rank=32]
            {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_alpha_raw.slice(..);
                match gdn_w.alpha_quant {
                    crate::gguf_loader::GgmlType::Q8_0 => {
                        let normed_view: CudaView<'_, f32> = graph.normed.slice(..);
                        crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                            &w_view, &normed_view, &mut sc.alpha_proj, dt_rank, hidden_size, dev)?;
                    }
                    _ => {
                        raw_q5k_gemv_to(
                            &w_view, &sc.q8_hidden, &mut sc.alpha_proj,
                            hidden_size, dt_rank, dev,
                        )?;
                    }
                }
            }
        }
        #[cfg(not(feature = "cubin_fallback"))]
        panic!("cubin fallback disabled — set CHIMERE_GGML_GPU=1 or enable feature cubin_fallback");
    }

    // ===================================================================
    // Step 7: Fused beta/alpha/gate computation
    //   beta_out[i] = sigmoid(beta_proj[i])
    //   gate_exp_out[i] = exp(softplus(alpha_proj[i] + dt_bias[i]) * ssm_a[i])
    // ===================================================================
    raw_fused_beta_alpha_gate(
        &sc.beta_proj, &sc.alpha_proj,
        &gdn_w.dt_bias, &gdn_w.ssm_a,
        &mut sc.beta_out, &mut sc.gate_exp_out,
        dt_rank, dev,
    )?;

    // ===================================================================
    // Step 8: Fused conv1d + SiLU + state update
    //   Reads conv_state + gdn_proj (QKV input).
    //   Writes conv_output + new_conv_state.
    //   Then copies new_conv_state -> conv_state.
    // ===================================================================
    {
        let conv_state = &sc.conv_states[gdn_idx];
        let cs_view: CudaView<'_, f32> = conv_state.slice(..);
        let input_view: CudaView<'_, f32> = graph.gdn_proj.slice(..);
        let weight_view: CudaView<'_, f32> = gdn_w.conv_weight.slice(..);
        raw_fused_conv1d_silu_update(
            &cs_view,
            &input_view,
            &weight_view,
            &mut sc.conv_output,
            &mut sc.new_conv_state,
            conv_channels,
            conv_kernel,
            dev,
        )?;
    }
    // Copy new conv state back to persistent storage
    {
        let src: CudaView<'_, f32> = sc.new_conv_state.slice(..);
        dev.memcpy_dtod(&src, &mut sc.conv_states[gdn_idx])
            .map_err(|e| candle_core::Error::Msg(
                format!("forward_gdn_cudarc: memcpy new_conv_state: {e}")))?;
    }

    // ===================================================================
    // Steps 9-12 (fused): Split Q/K from conv_output, L2-normalize,
    //   expand groups to dt_rank heads (TILED), scale Q by 1/sqrt(d_state).
    //   Layout: [Q: key_dim | K: key_dim | V: value_dim]
    //   Single kernel launch replaces 5 separate launches (Phase 2.2).
    //   V is still accessed as a zero-copy view (Phase 2.3).
    // ===================================================================
    let v_view: CudaView<'_, f32> = sc.conv_output.slice(key_dim * 2..key_dim * 2 + value_dim);
    let repeats = dt_rank / n_group;
    fused_split_norm_expand_scale(
        &sc.conv_output, &mut sc.q_scaled, &mut sc.k_expanded,
        key_dim, d_state, n_group, repeats, eps, dev,
    )?;

    // ===================================================================
    // Step 13: DeltaNet recurrent step
    //   Reads gdn_state (current), writes state_scratch (next) + ssm_output.
    //   Then copies state_scratch -> gdn_state.
    //   V is passed as a zero-copy view into conv_output (Phase 2.3).
    // ===================================================================
    // In-place state update: the kernel reads each column to registers
    // before writing, so s_in == s_out is safe (no inter-thread dependency).
    // Eliminates one memcpy_dtod per GDN layer (30 × 2MB = 60MB/token).
    deltanet_step_fused_raw_inplace(
        gdn_state,              // state: both input and output (in-place)
        &sc.q_scaled.slice(..),
        &sc.k_expanded.slice(..),
        &v_view,                // zero-copy from conv_output
        &sc.gate_exp_out.slice(..),
        &sc.beta_out.slice(..),
        &mut sc.ssm_output,
        dt_rank,
        d_state,
        dev,
    )?;

    // ===================================================================
    // Step 14: Fused RMSNorm + SiLU gate
    //   gated[g*D+j] = rms_norm(ssm_output[g,j], ssm_norm[j]) * silu(gdn_gate[g*D+j])
    //   ssm_output viewed as [dt_rank groups, d_state each]
    // ===================================================================
    raw_fused_rms_norm_silu_gate(
        &sc.ssm_output,
        &gdn_w.ssm_norm,
        &graph.gdn_gate,    // gate (z) projection output
        &mut sc.gated,
        dt_rank,            // groups
        d_state,            // D per group
        eps,
        dev,
    )?;

    // ===================================================================
    // Steps 15+16 fused: Quantize gated + SSM output projection -> gdn_out [hidden_size]
    //   w_ssm_out: [hidden_size, value_dim] Q5_K
    //   ncols=value_dim (input), nrows=hidden_size (output)
    //
    // When CHIMERE_GGML_GPU=1: use ggml's optimized MMVQ kernels.
    // Otherwise: use Candle's built-in quantize_q8_1 + mul_mat_vec_q5_K_q8_1.
    // ===================================================================
    if let Some(ref mut ggml_bufs) = ggml {
        // --- ggml MMVQ path ---
        // Phase 3.1: Check for stream override (set during CUDA Graph capture).
        let stream_ptr = ggml_bufs.active_stream();
        let use_stream = !stream_ptr.is_null();

        // Step 15+16 fused: Quantize gated to Q8_1 + ssm_out projection in one FFI call.
        // Cached-ptr path falls back to separate quantize + cached GEMV.
        {
            let ssm_out_cached = if !use_stream {
                cached.and_then(|c| c.gdn_ssm_out.get(gdn_idx).copied())
            } else {
                None
            };
            if let Some(ssm_out_ptr) = ssm_out_cached {
                // Cached path: separate quantize + cached GEMV
                {
                    let gated_view: CudaView<'_, f32> = sc.gated.slice(..);
                    crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                        &gated_view, &mut ggml_bufs.q8_input, value_dim, dev,
                    )?;
                }
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                    ssm_out_ptr, &ggml_bufs.q8_input, &mut graph.gdn_out,
                    hidden_size, value_dim,
                )?;
            } else {
                let gated_view: CudaView<'_, f32> = sc.gated.slice(..);
                let w_view: CudaView<'_, u8> = gdn_w.ssm_out_raw.slice(..);
                if use_stream {
                    crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32_on_stream(
                        &w_view, &gated_view, &mut graph.gdn_out,
                        &mut ggml_bufs.q8_input,
                        hidden_size, value_dim, stream_ptr,
                    )?;
                } else {
                    crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32(
                        &w_view, &gated_view, &mut graph.gdn_out,
                        &mut ggml_bufs.q8_input,
                        hidden_size, value_dim,
                    )?;
                }
            }
        }
    } else {
        // --- Candle fallback path (cubin Q5K kernels) ---
        #[cfg(feature = "cubin_fallback")]
        {
            // Step 15: Quantize gated to Q8_1
            {
                let gated_view: CudaView<'_, f32> = sc.gated.slice(..);
                raw_quantize_q8_1_to(
                    &gated_view, &mut sc.q8_gated, value_dim, dev,
                )?;
            }
            // Step 16: ssm_out projection
            {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_out_raw.slice(..);
                raw_q5k_gemv_to(
                    &w_view, &sc.q8_gated, &mut graph.gdn_out,
                    value_dim, hidden_size, dev,
                )?;
            }
        }
        #[cfg(not(feature = "cubin_fallback"))]
        panic!("cubin fallback disabled — set CHIMERE_GGML_GPU=1 or enable feature cubin_fallback");
    }

    // ===================================================================
    // Step 17: Write GDN output to hidden buffer.
    //
    // The caller (forward_token) handles the residual connection:
    //   hidden += residual (saved before entering this function).
    // We just need to write the SSM output projection (gdn_out) into hidden.
    // ===================================================================
    dev.memcpy_dtod(&graph.gdn_out, &mut graph.hidden)
        .map_err(|e| candle_core::Error::Msg(
            format!("forward_gdn_cudarc: memcpy gdn_out->hidden: {e}")))?;

    Ok(())
}

// =====================================================================
// Batch GDN prefill — processes N tokens through one GDN layer
// =====================================================================

/// Batch GDN forward for prefill: processes N tokens through one GDN layer.
///
/// Batch operations where possible (Q8_1 quant, elementwise), sequential
/// where state-dependent (conv1d, DeltaNet recurrence).
///
/// # Operation overview
///
///  **Phase A — Batch projections (steps 2-6):**
///    1. Batch Q8_1 quantize all N normed hidden states (1 kernel launch)
///    2. Loop GEMV per token for QKV, gate, beta, alpha projections
///       (reuse the same Q8_1 quantization — no re-quantization)
///
///  **Phase B — Batch elementwise (step 7):**
///    Fused beta/alpha/gate for all N tokens
///
///  **Phase C — Sequential recurrence (steps 8-13):**
///    For each token: conv1d + SiLU + split/norm/expand + DeltaNet step
///    (state-dependent — cannot be parallelized)
///
///  **Phase D — Batch output (steps 14-16):**
///    Batch norm+gate, then loop GEMV for ssm_out projection, write results
///
/// # Post-conditions
///
/// - `prefill.hidden_batch` has been updated with the SSM output projection
///   (residual add is the caller's responsibility).
/// - `gdn_state` has been updated with the final DeltaNet state after N tokens.
/// - Conv state (inside `sc`) has been updated.
pub(crate) fn forward_gdn_prefill_cudarc(
    il: usize,
    gdn_w: &super::compute_graph::GdnWeightsRaw,
    gdn_state: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    graph: &mut super::compute_graph::ComputeGraph,
    prefill: &mut super::compute_graph::PrefillBuffers,
    n_tokens: usize,
    cached: Option<&super::compute_graph::CachedWeightPtrs>,
) -> candle_core::Result<()> {
    if n_tokens == 0 {
        return Ok(());
    }

    let hidden_size = graph.hidden_size;
    let n_group = graph.ssm_n_group;
    let d_state = graph.ssm_d_state;
    let dt_rank = graph.ssm_dt_rank;
    let key_dim = n_group * d_state;               // 2048
    let value_dim = dt_rank * d_state;              // 4096
    let conv_channels = key_dim * 2 + value_dim;    // 8192
    let conv_kernel = graph.ssm_conv_kernel;        // 4
    let eps = 1e-6f32;
    let dev = graph.dev.clone();

    let gdn_idx = super::compute_graph::gdn_index(
        il, graph.full_attn_interval,
    );

    // Lazy-init the cudarc scratch on first call
    if graph.gdn_cudarc.is_none() {
        graph.gdn_cudarc = Some(
            super::compute_graph::GdnCudarcScratch::from_graph(graph)?
        );
    }

    // Take scratch out to avoid borrow conflicts (same pattern as single-token)
    let mut sc = graph.gdn_cudarc.take().unwrap();
    let mut ggml = graph.ggml_gpu_bufs.take();

    let result = forward_gdn_prefill_inner(
        il, gdn_w, gdn_state, graph, prefill, &mut sc, &mut ggml,
        hidden_size, n_group, d_state, dt_rank,
        key_dim, value_dim, conv_channels, conv_kernel, eps,
        gdn_idx, &dev, n_tokens, cached,
    );

    // Always put scratch back
    graph.ggml_gpu_bufs = ggml;
    graph.gdn_cudarc = Some(sc);
    result
}

/// Inner implementation of batch GDN prefill (separated for safe take+put pattern).
fn forward_gdn_prefill_inner(
    _il: usize,
    gdn_w: &super::compute_graph::GdnWeightsRaw,
    gdn_state: &mut candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
    graph: &mut super::compute_graph::ComputeGraph,
    prefill: &mut super::compute_graph::PrefillBuffers,
    sc: &mut super::compute_graph::GdnCudarcScratch,
    ggml: &mut Option<crate::kernels::ggml_gpu::GgmlGpuBuffers>,
    hidden_size: usize,
    n_group: usize,
    d_state: usize,
    dt_rank: usize,
    key_dim: usize,
    value_dim: usize,
    conv_channels: usize,
    conv_kernel: usize,
    eps: f32,
    gdn_idx: usize,
    dev: &candle_core::cuda_backend::CudaDevice,
    n_tokens: usize,
    cached: Option<&super::compute_graph::CachedWeightPtrs>,
) -> candle_core::Result<()> {
    use candle_core::cuda_backend::cudarc::driver::CudaView;
    use crate::kernels::elementwise::{
        raw_fused_beta_alpha_gate_batch_inplace,
        raw_fused_conv1d_silu_update,
        raw_fused_rms_norm_silu_gate,
    };
    use crate::kernels::fused_ops::fused_split_norm_expand_scale;
    use crate::kernels::deltanet_step::deltanet_step_fused_raw_inplace;

    // Q8_1 row size for hidden_size (used in batch quantize)
    let ncols_padded = crate::kernels::ggml_gpu::pad(hidden_size, crate::kernels::ggml_gpu::MATRIX_ROW_PADDING);
    let q8_row_bytes = (ncols_padded / crate::kernels::ggml_gpu::Q8_1_BLOCK_ELEMS)
                     * crate::kernels::ggml_gpu::Q8_1_BLOCK_BYTES;

    // Get cached weight pointers for this GDN layer if available
    let gdn_cache = cached.map(|c| (
        c.gdn_ssm_in.get(gdn_idx).copied(),
        c.gdn_ssm_gate.get(gdn_idx).copied(),
        c.gdn_ssm_beta.get(gdn_idx).copied(),
        c.gdn_ssm_alpha.get(gdn_idx).copied(),
        c.gdn_ssm_out.get(gdn_idx).copied(),
    ));

    // ===================================================================
    // Phase A: Batch projections (steps 2-6)
    // ===================================================================

    // Step 2: Batch Q8_1 quantize all N normed hidden states at once
    // normed_batch[N*hidden_size] -> q8_batch[N*q8_row_bytes]
    if ggml.is_some() {
        crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1_batched(
            &prefill.normed_batch, &mut prefill.q8_batch,
            hidden_size, n_tokens,
        )?;
    } else {
        // No ggml path for batch — fall back would require cubin_fallback per-token
        candle_core::bail!("batch GDN prefill requires CHIMERE_GGML_GPU=1 (ggml MMVQ)");
    }

    // Steps 3-6: Loop GEMV per token for all 4 projections.
    // Each token's Q8_1 data is at offset t*q8_row_bytes in q8_batch.
    // We reuse the single-token ggml_gpu buffers (q8_input) by copying
    // each token's Q8_1 slice into it, then running the GEMVs.
    let ggml_bufs = ggml.as_mut().unwrap();

    for t in 0..n_tokens {
        let q8_offset = t * q8_row_bytes;

        // Copy this token's Q8_1 data into the ggml q8_input buffer
        {
            let src: CudaView<'_, u8> = prefill.q8_batch.slice(q8_offset..q8_offset + q8_row_bytes);
            let mut dst = ggml_bufs.q8_input.slice_mut(..q8_row_bytes);
            dev.memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy q8 token {t}: {e}")))?;
        }

        // Step 3: QKV projection -> gdn_proj_batch[t*conv_channels..]
        {
            // GEMV into single-token gdn_proj scratch
            if let Some((Some(ssm_in_ptr), _, _, _, _)) = gdn_cache {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                    ssm_in_ptr, &ggml_bufs.q8_input, &mut graph.gdn_proj,
                    conv_channels, hidden_size,
                )?;
            } else {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_in_raw.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                    &w_view, &ggml_bufs.q8_input, &mut graph.gdn_proj,
                    conv_channels, hidden_size,
                )?;
            }
            // Copy result to batch buffer
            let dst_offset = t * conv_channels;
            let src: CudaView<'_, f32> = graph.gdn_proj.slice(..conv_channels);
            let mut dst = prefill.gdn_proj_batch.slice_mut(dst_offset..dst_offset + conv_channels);
            dev.memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy proj token {t}: {e}")))?;
        }

        // Step 4: Gate (z) projection -> gdn_gate_batch[t*value_dim..]
        {
            if let Some((_, Some(ssm_gate_ptr), _, _, _)) = gdn_cache {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                    ssm_gate_ptr, &ggml_bufs.q8_input, &mut graph.gdn_gate,
                    value_dim, hidden_size,
                )?;
            } else {
                let w_view: CudaView<'_, u8> = gdn_w.ssm_gate_raw.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                    &w_view, &ggml_bufs.q8_input, &mut graph.gdn_gate,
                    value_dim, hidden_size,
                )?;
            }
            let dst_offset = t * value_dim;
            let src: CudaView<'_, f32> = graph.gdn_gate.slice(..value_dim);
            let mut dst = prefill.gdn_gate_batch.slice_mut(dst_offset..dst_offset + value_dim);
            dev.memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy gate token {t}: {e}")))?;
        }

        // Step 5: Beta projection -> beta_batch[t*dt_rank..]
        {
            match gdn_w.beta_quant {
                crate::gguf_loader::GgmlType::Q8_0 => {
                    let w_view: CudaView<'_, u8> = gdn_w.ssm_beta_raw.slice(..);
                    // Q8_0 GEMV takes f32 input, not Q8_1
                    let normed_offset = t * hidden_size;
                    let normed_view: CudaView<'_, f32> = prefill.normed_batch.slice(
                        normed_offset..normed_offset + hidden_size);
                    crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                        &w_view, &normed_view, &mut sc.beta_proj,
                        dt_rank, hidden_size, dev)?;
                }
                _ => {
                    if let Some((_, _, Some(ssm_beta_ptr), _, _)) = gdn_cache {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                            ssm_beta_ptr, &ggml_bufs.q8_input, &mut sc.beta_proj,
                            dt_rank, hidden_size,
                        )?;
                    } else {
                        let w_view: CudaView<'_, u8> = gdn_w.ssm_beta_raw.slice(..);
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                            &w_view, &ggml_bufs.q8_input, &mut sc.beta_proj,
                            dt_rank, hidden_size,
                        )?;
                    }
                }
            }
            let dst_offset = t * dt_rank;
            let src: CudaView<'_, f32> = sc.beta_proj.slice(..dt_rank);
            let mut dst = prefill.beta_batch.slice_mut(dst_offset..dst_offset + dt_rank);
            dev.memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy beta token {t}: {e}")))?;
        }

        // Step 6: Alpha projection -> alpha_batch[t*dt_rank..]
        {
            match gdn_w.alpha_quant {
                crate::gguf_loader::GgmlType::Q8_0 => {
                    let w_view: CudaView<'_, u8> = gdn_w.ssm_alpha_raw.slice(..);
                    let normed_offset = t * hidden_size;
                    let normed_view: CudaView<'_, f32> = prefill.normed_batch.slice(
                        normed_offset..normed_offset + hidden_size);
                    crate::kernels::gemv_q8_0::gemv_q8_0_f32(
                        &w_view, &normed_view, &mut sc.alpha_proj,
                        dt_rank, hidden_size, dev)?;
                }
                _ => {
                    if let Some((_, _, _, Some(ssm_alpha_ptr), _)) = gdn_cache {
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                            ssm_alpha_ptr, &ggml_bufs.q8_input, &mut sc.alpha_proj,
                            dt_rank, hidden_size,
                        )?;
                    } else {
                        let w_view: CudaView<'_, u8> = gdn_w.ssm_alpha_raw.slice(..);
                        crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                            &w_view, &ggml_bufs.q8_input, &mut sc.alpha_proj,
                            dt_rank, hidden_size,
                        )?;
                    }
                }
            }
            let dst_offset = t * dt_rank;
            let src: CudaView<'_, f32> = sc.alpha_proj.slice(..dt_rank);
            let mut dst = prefill.alpha_batch.slice_mut(dst_offset..dst_offset + dt_rank);
            dev.memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy alpha token {t}: {e}")))?;
        }
    }

    // ===================================================================
    // Phase B: Batch elementwise (step 7)
    // Fused beta/alpha/gate for all N tokens in ONE kernel launch.
    //
    // beta_batch and alpha_batch are [N * dt_rank] contiguous.
    // The batch kernel uses i % dt_rank to index into dt_bias/ssm_a
    // (shared across all tokens). Results are written IN-PLACE:
    //   beta_batch  <- sigmoid(beta_proj)
    //   alpha_batch <- exp(softplus(alpha_proj + dt_bias) * ssm_a)
    //
    // Replaces N * (4 memcpy + 1 kernel) = 5N GPU ops with 1 kernel.
    // ===================================================================
    raw_fused_beta_alpha_gate_batch_inplace(
        &mut prefill.beta_batch,
        &mut prefill.alpha_batch,
        &gdn_w.dt_bias,
        &gdn_w.ssm_a,
        dt_rank,
        n_tokens,
        &dev,
    )?;
    // After Phase B: beta_batch[t*dt_rank..] = sigmoid(beta_proj[t])
    //                alpha_batch[t*dt_rank..] = exp(softplus(alpha+bias)*a)  (gate_exp)

    // ===================================================================
    // Phase C: Sequential recurrence (steps 8-13)
    //
    // For each token: extract projections from batch buffers into single-
    // token scratch, run conv1d + split/norm/expand + DeltaNet step.
    // Each token updates conv_state and gdn_state sequentially.
    // ===================================================================
    let repeats = dt_rank / n_group;

    for t in 0..n_tokens {
        // --- Step 8: Conv1d + SiLU + state update ---
        // Copy this token's QKV projection from batch to gdn_proj (single-token)
        {
            let proj_offset = t * conv_channels;
            let src: CudaView<'_, f32> = prefill.gdn_proj_batch.slice(
                proj_offset..proj_offset + conv_channels);
            dev.memcpy_dtod(&src, &mut graph.gdn_proj)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy proj t={t}: {e}")))?;
        }

        {
            let conv_state = &sc.conv_states[gdn_idx];
            let cs_view: CudaView<'_, f32> = conv_state.slice(..);
            let input_view: CudaView<'_, f32> = graph.gdn_proj.slice(..);
            let weight_view: CudaView<'_, f32> = gdn_w.conv_weight.slice(..);
            raw_fused_conv1d_silu_update(
                &cs_view,
                &input_view,
                &weight_view,
                &mut sc.conv_output,
                &mut sc.new_conv_state,
                conv_channels,
                conv_kernel,
                dev,
            )?;
        }
        // Copy new conv state back to persistent storage
        {
            let src: CudaView<'_, f32> = sc.new_conv_state.slice(..);
            dev.memcpy_dtod(&src, &mut sc.conv_states[gdn_idx])
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: conv state update t={t}: {e}")))?;
        }

        // --- Steps 9-12: Split Q/K, L2-norm, expand, scale ---
        let v_view: CudaView<'_, f32> = sc.conv_output.slice(
            key_dim * 2..key_dim * 2 + value_dim);
        fused_split_norm_expand_scale(
            &sc.conv_output, &mut sc.q_scaled, &mut sc.k_expanded,
            key_dim, d_state, n_group, repeats, eps, dev,
        )?;

        // --- Step 13: DeltaNet recurrent step ---
        // Copy this token's beta_out and gate_exp from batch buffers
        {
            let beta_offset = t * dt_rank;
            let src: CudaView<'_, f32> = prefill.beta_batch.slice(
                beta_offset..beta_offset + dt_rank);
            dev.memcpy_dtod(&src, &mut sc.beta_out)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy beta_out t={t}: {e}")))?;
        }
        {
            let alpha_offset = t * dt_rank;
            let src: CudaView<'_, f32> = prefill.alpha_batch.slice(
                alpha_offset..alpha_offset + dt_rank);
            dev.memcpy_dtod(&src, &mut sc.gate_exp_out)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy gate_exp t={t}: {e}")))?;
        }

        deltanet_step_fused_raw_inplace(
            gdn_state,
            &sc.q_scaled.slice(..),
            &sc.k_expanded.slice(..),
            &v_view,
            &sc.gate_exp_out.slice(..),
            &sc.beta_out.slice(..),
            &mut sc.ssm_output,
            dt_rank,
            d_state,
            dev,
        )?;

        // --- Step 14: Fused RMSNorm + SiLU gate ---
        // Copy this token's gate (z) from batch to gdn_gate (single-token)
        {
            let gate_offset = t * value_dim;
            let src: CudaView<'_, f32> = prefill.gdn_gate_batch.slice(
                gate_offset..gate_offset + value_dim);
            dev.memcpy_dtod(&src, &mut graph.gdn_gate)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: copy gate t={t}: {e}")))?;
        }

        raw_fused_rms_norm_silu_gate(
            &sc.ssm_output,
            &gdn_w.ssm_norm,
            &graph.gdn_gate,
            &mut sc.gated,
            dt_rank,
            d_state,
            eps,
            dev,
        )?;

        // --- Steps 15+16 fused: Quantize gated + SSM output projection ---
        // Cached-ptr path falls back to separate quantize + cached GEMV.
        if let Some((_, _, _, _, Some(ssm_out_ptr))) = gdn_cache {
            {
                let gated_view: CudaView<'_, f32> = sc.gated.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                    &gated_view, &mut ggml_bufs.q8_input, value_dim, dev,
                )?;
            }
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8_cached(
                ssm_out_ptr, &ggml_bufs.q8_input, &mut graph.gdn_out,
                hidden_size, value_dim,
            )?;
        } else {
            let gated_view: CudaView<'_, f32> = sc.gated.slice(..);
            let w_view: CudaView<'_, u8> = gdn_w.ssm_out_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32(
                &w_view, &gated_view, &mut graph.gdn_out,
                &mut ggml_bufs.q8_input,
                hidden_size, value_dim,
            )?;
        }

        // --- Step 17: Write SSM output to hidden_batch ---
        // hidden_batch[t] = gdn_out (caller handles residual add)
        {
            let dst_offset = t * hidden_size;
            let src: CudaView<'_, f32> = graph.gdn_out.slice(..hidden_size);
            let mut dst = prefill.hidden_batch.slice_mut(dst_offset..dst_offset + hidden_size);
            dev.memcpy_dtod(&src, &mut dst)
                .map_err(|e| candle_core::Error::Msg(
                    format!("prefill gdn: writeback hidden t={t}: {e}")))?;
        }
    }

    Ok(())
}
