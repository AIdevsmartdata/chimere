//! MoE FFN forward pass variants.
//!
//! Extracted from qwen35_model.rs — pure code movement, zero behavioral changes.

use candle_core::{Device, Module, Result, Tensor};

use super::{Qwen35Model, MoeFFN};
use crate::debug_utils::dispatch_prof_enabled;

impl Qwen35Model {
    /// Run MoE FFN: GPU-dequantize IQ3_S expert bytes on demand, then call moe_forward.
    ///
    /// Expert weights are stored as raw IQ3_S bytes in flat U8 Tensors on the GPU.
    /// This function extracts the underlying `CudaSlice<u8>` and passes it to
    /// `dequant_iq3s_gpu`, which runs a CUDA kernel to produce F32 Tensors for the
    /// three stacked expert weight matrices.  Only the active experts are then sliced
    /// by `moe_forward`.
    ///
    /// On non-CUDA devices (CPU tests), falls back to an error stub — the IQ3_S GPU
    /// kernel requires CUDA.  CPU testing should use the dense path (27B) or mock weights.
    /// Raw MoE FFN: routed experts via cudarc direct, shared expert via Candle.
    /// Eliminates ~80 Tensor allocations per layer vs the pure Candle path.
    pub(crate) fn moe_ffn_forward_raw(
        hidden: &Tensor,
        moe: &MoeFFN,
        top_k: usize,
        hidden_size: usize,
        expert_ffn: usize,
        num_experts: usize,
        buffers: &mut crate::raw_forward::RawGpuBuffers,
    ) -> Result<Tensor> {
        use candle_core::Storage;
        let dev = hidden.device();
        let Device::Cuda(cuda_dev) = dev else {
            candle_core::bail!("raw MoE requires CUDA");
        };

        let dprof = dispatch_prof_enabled();
        // Dispatch profiling: track sub-operation costs within MoE FFN
        // Only report for a single layer on token 1 to avoid noise
        let dprof_report = if dprof {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static MOE_DPROF_COUNT: AtomicUsize = AtomicUsize::new(0);
            let c = MOE_DPROF_COUNT.fetch_add(1, Ordering::Relaxed);
            // Report on the FIRST MoE layer of token 1 (skip token 0 warmup)
            // Token 0 has ~39 MoE layers, so token 1 starts at call 39
            c == 39
        } else {
            false
        };

        // --- storage_and_layout timing ---
        let _t_sal = std::time::Instant::now();

        // Extract CudaView from hidden (always contiguous for batch=1 forward)
        let (h_stor, h_lay) = hidden.storage_and_layout();
        let h_cuda = match &*h_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("") };
        let h_view = h_cuda.as_cuda_slice::<f32>()?.slice(h_lay.start_offset()..);

        let (r_stor, r_lay) = moe.gate_inp_t.storage_and_layout();
        let r_cuda = match &*r_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("") };
        let _r_view = r_cuda.as_cuda_slice::<f32>()?.slice(r_lay.start_offset()..);

        let (g_stor, g_lay) = moe.gate_exps_raw.storage_and_layout();
        let g_cuda = match &*g_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("") };
        let g_view = g_cuda.as_cuda_slice::<u8>()?.slice(g_lay.start_offset()..);

        let (u_stor, u_lay) = moe.up_exps_raw.storage_and_layout();
        let u_cuda = match &*u_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("") };
        let u_view = u_cuda.as_cuda_slice::<u8>()?.slice(u_lay.start_offset()..);

        let (d_stor, d_lay) = moe.down_exps_raw.storage_and_layout();
        let d_cuda = match &*d_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("") };
        let d_view = d_cuda.as_cuda_slice::<u8>()?.slice(d_lay.start_offset()..);

        let t_sal_ms = if dprof_report { _t_sal.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // Expert byte sizes
        const IQ3S_BLOCK_BYTES: usize = 110;
        const IQ3S_BLOCK_ELEMS: usize = 256;
        let expert_elements_gate = moe.gate_exps_shape.1 * moe.gate_exps_shape.2;
        let expert_elements_up = moe.up_exps_shape.1 * moe.up_exps_shape.2;
        let expert_elements_down = moe.down_exps_shape.1 * moe.down_exps_shape.2;
        let expert_bytes_gate = (expert_elements_gate / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_up = (expert_elements_up / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_down = (expert_elements_down / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;

        // Trace toggle (defined early — needed for conditional dtoh)
        let trace = {
            use once_cell::sync::Lazy;
            static T: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_TRACE").is_ok());
            *T
        };

        // --- Router timing ---
        let _t_router = std::time::Instant::now();
        // Router matmul stays on GPU via Candle.
        let router_logits = hidden.matmul(&moe.gate_inp_t)?;
        // GPU top-K softmax: no CPU sync for 256 probabilities.
        // Only 64 bytes (8 i32 indices + 8 f32 weights) are transferred back,
        // instead of the old 1 KB (256 f32) transfer via to_vec1().
        {
            let (rl_stor, rl_lay) = router_logits.storage_and_layout();
            let rl_cuda = match &*rl_stor {
                Storage::Cuda(c) => c,
                _ => candle_core::bail!("router_logits not on CUDA"),
            };
            let rl_view = rl_cuda.as_cuda_slice::<f32>()?.slice(rl_lay.start_offset()..);
            crate::deltanet_kernel::gpu_topk_softmax(
                &rl_view,
                &mut buffers.topk_indices_buf,
                &mut buffers.topk_weights_buf,
                num_experts, top_k, cuda_dev,
            )?;
            // Drop storage guard before dtoh to avoid holding the lock across sync.
            drop(rl_stor);
        }
        // GPU-resident top-K: indices and weights stay on GPU.
        // Only read back to CPU when trace logging is needed.
        let selected: Vec<(usize, f32)> = if trace {
            let indices_cpu: Vec<i32> = buffers.topk_indices_buf.stream().clone()
                .clone_dtoh(&buffers.topk_indices_buf)
                .map_err(|e| candle_core::Error::Msg(format!("topk indices dtoh: {e}")))?;
            let weights_cpu: Vec<f32> = buffers.topk_weights_buf.stream().clone()
                .clone_dtoh(&buffers.topk_weights_buf)
                .map_err(|e| candle_core::Error::Msg(format!("topk weights dtoh: {e}")))?;
            indices_cpu.iter().zip(weights_cpu.iter())
                .map(|(&i, &w)| (i as usize, w)).collect()
        } else {
            Vec::new()  // not used when trace is off
        };
        let t_router_ms = if dprof_report { _t_router.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // === TRACER: compare dequant+matmul reference vs GEMV kernel on layer 0 ===
        if trace {
            // Only trace once (first call = layer 0, token 0)
            use std::sync::atomic::{AtomicBool, Ordering};
            static TRACED: AtomicBool = AtomicBool::new(false);
            if !TRACED.swap(true, Ordering::Relaxed) {
                eprintln!("[TRACE] === MoE FFN layer 0, first token ===");

                // 1. Log router selection
                eprintln!("[TRACE] Router: experts={:?} weights={:.4?}",
                    selected.iter().map(|(i,_)| *i).collect::<Vec<_>>(),
                    selected.iter().map(|(_,w)| *w).collect::<Vec<f32>>());

                // 2. Log hidden (input to expert GEMVs)
                let h_cpu: Vec<f32> = hidden.flatten_all()?.to_vec1()?;
                eprintln!("[TRACE] hidden: sum={:.4} min={:.4} max={:.4} first5={:.4?}",
                    h_cpu.iter().sum::<f32>(),
                    h_cpu.iter().cloned().fold(f32::MAX, f32::min),
                    h_cpu.iter().cloned().fold(f32::MIN, f32::max),
                    &h_cpu[..5]);

                // 3. For the FIRST expert: compare dequant+matmul vs GEMV kernel
                let expert_id = selected[0].0;
                let gate_offset = expert_id * expert_bytes_gate;

                // Path A: dequant IQ3_S to F32 tensor, then Candle matmul
                let gate_shape = [expert_ffn, hidden_size];
                let g_slice_raw = g_cuda.as_cuda_slice::<u8>()?;
                let gate_f32 = crate::deltanet_kernel::dequant_iq3s_at_offset(
                    g_slice_raw,
                    g_lay.start_offset() + gate_offset,
                    expert_ffn * hidden_size, &gate_shape, dev)?;
                // gate_f32 is [expert_ffn, hidden_size], hidden is [1, hidden_size]
                let ref_result = hidden.matmul(&gate_f32.t()?)?;
                let ref_vec: Vec<f32> = ref_result.flatten_all()?.to_vec1()?;

                // Path B: GEMV kernel (Q8_1+dp4a or F32 depending on current code)
                use candle_core::backend::BackendDevice;
                let mut kernel_out = cuda_dev.alloc_zeros::<f32>(expert_ffn)
                    .map_err(|e| candle_core::Error::Msg(format!("trace alloc: {e}")))?;
                crate::deltanet_kernel::gemv_iq3s_fused_at_offset(
                    &g_view, gate_offset, &h_view, &mut kernel_out,
                    expert_ffn, hidden_size, cuda_dev)?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("trace sync: {e}")))?;

                // Read kernel output back to CPU
                let kernel_slice = kernel_out.try_clone()
                    .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
                let kernel_storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                    kernel_slice, cuda_dev.clone());
                let kernel_tensor = Tensor::from_storage(
                    candle_core::Storage::Cuda(kernel_storage),
                    candle_core::Shape::from_dims(&[expert_ffn]),
                    candle_core::op::BackpropOp::none(), false);
                let kernel_vec: Vec<f32> = kernel_tensor.to_device(&Device::Cpu)?.to_vec1()?;

                // Compare first 20 elements
                let n_show = 20.min(expert_ffn);
                let mut max_abs = 0.0f32;
                let mut max_rel = 0.0f32;
                for i in 0..expert_ffn {
                    let e = (ref_vec[i] - kernel_vec[i]).abs();
                    max_abs = max_abs.max(e);
                    if ref_vec[i].abs() > 1e-6 {
                        max_rel = max_rel.max(e / ref_vec[i].abs());
                    }
                }
                for i in 0..n_show {
                    let e = (ref_vec[i] - kernel_vec[i]).abs();
                    eprintln!("[TRACE] gate[{:3}]: ref={:+10.4} kernel={:+10.4} err={:.6}{}",
                        i, ref_vec[i], kernel_vec[i], e,
                        if e > 0.01 * ref_vec[i].abs().max(1.0) { " <<<" } else { "" });
                }
                eprintln!("[TRACE] Expert {}: max_abs_err={:.6} max_rel_err={:.4}% over {} elements",
                    expert_id, max_abs, max_rel * 100.0, expert_ffn);
                if max_rel > 0.05 {
                    eprintln!("[TRACE] *** WARNING: >5% relative error — kernel has a bug ***");
                }
            }
        }

        let use_fused = {
            use once_cell::sync::Lazy;
            static FUSED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_FUSED_MOE").is_err());
            *FUSED
        };

        // --- Kernel launch timing ---
        let _t_kernel = std::time::Instant::now();
        if use_fused {
            // Fused kernel: single launch for all 8 experts
            crate::raw_forward::moe_ffn_fused(
                &h_view, &selected,
                &g_view, &u_view, &d_view,
                expert_bytes_gate, expert_bytes_up, expert_bytes_down,
                buffers,
                hidden_size, expert_ffn, top_k,
                cuda_dev,
            )?;
        } else {
            // Per-expert path: 24 individual GEMV launches
            crate::raw_forward::moe_ffn_raw(
                &h_view, &selected,
                &g_view, &u_view, &d_view,
                expert_bytes_gate, expert_bytes_up, expert_bytes_down,
                expert_elements_gate, expert_elements_up, expert_elements_down,
                buffers,
                hidden_size, expert_ffn, num_experts, top_k,
                cuda_dev, dev,
            )?;
        }
        let t_kernel_ms = if dprof_report { _t_kernel.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // --- Drop storage guards timing ---
        let _t_drop = std::time::Instant::now();
        // Drop storage guards explicitly before using moe fields via Candle
        drop(h_stor); drop(r_stor); drop(g_stor); drop(u_stor); drop(d_stor);
        let t_drop_ms = if dprof_report { _t_drop.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // --- Shared expert timing ---
        let _t_shared = std::time::Instant::now();
        // Shared expert via Candle (0.02ms/layer, trivial)
        let use_ggml_shexp = {
            use once_cell::sync::Lazy;
            static G: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
            *G
        };
        let sh_gate = if use_ggml_shexp && moe.gate_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.gate_shexp_raw.as_ref().unwrap(), hidden,
                512, hidden.dim(1)?)?
        } else { moe.gate_shexp.forward(hidden)? };
        let sh_up = if use_ggml_shexp && moe.up_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.up_shexp_raw.as_ref().unwrap(), hidden,
                512, hidden.dim(1)?)?
        } else { moe.up_shexp.forward(hidden)? };
        let sh_activated = sh_gate.silu()?;
        let sh_intermediate = (&sh_activated * &sh_up)?;
        let shared_out = if use_ggml_shexp && moe.down_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.down_shexp_raw.as_ref().unwrap(), &sh_intermediate,
                hidden.dim(1)?, 512)?
        } else { moe.down_shexp.forward(&sh_intermediate)? };
        let gate_logit = hidden.matmul(&moe.gate_inp_shexp.unsqueeze(1)?)?;
        let shared_scale = candle_nn::ops::sigmoid(&gate_logit)?;
        let gated_shared = shared_out.broadcast_mul(&shared_scale)?;
        let t_shared_ms = if dprof_report { _t_shared.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // --- In-place add: gated_shared += combined_buf (zero cudaMalloc) ---
        let _t_wrap = std::time::Instant::now();
        // Instead of try_clone() + Tensor::from_storage + Tensor add (2 allocs),
        // we add combined_buf directly into gated_shared via InplaceOp1 (0 allocs).
        let add_op = crate::raw_forward::AddCudaSliceInplace::new(
            &buffers.combined_buf, hidden_size);
        gated_shared.inplace_op1(&add_op)?;
        let t_wrap_ms = if dprof_report { _t_wrap.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // gated_shared now contains routed + shared expert output
        let output = gated_shared;

        if dprof_report {
            let total = t_sal_ms + t_router_ms + t_kernel_ms + t_wrap_ms + t_drop_ms + t_shared_ms;
            eprintln!("\n[DISPATCH-MOE] ========== MoE FFN Sub-operation Breakdown ==========");
            eprintln!("[DISPATCH-MOE] storage_and_layout (5 tensors): {:.3}ms", t_sal_ms);
            eprintln!("[DISPATCH-MOE] router (matmul+softmax+topk):   {:.3}ms", t_router_ms);
            eprintln!("[DISPATCH-MOE] kernel (moe_ffn_{}):{:>13}{:.3}ms",
                if use_fused { "fused" } else { "raw  " }, " ", t_kernel_ms);
            eprintln!("[DISPATCH-MOE] inplace_add (zero alloc):       {:.3}ms", t_wrap_ms);
            eprintln!("[DISPATCH-MOE] drop (5 storage guards):        {:.3}ms", t_drop_ms);
            eprintln!("[DISPATCH-MOE] shared_expert (Candle ops):     {:.3}ms", t_shared_ms);
            eprintln!("[DISPATCH-MOE] accounted total:                {:.3}ms", total);
            eprintln!("[DISPATCH-MOE] ==========================================================\n");
        }

        Ok(output)
    }

    /// Raw MoE FFN v2: zero `storage_and_layout` calls.
    ///
    /// Takes pre-extracted `CudaSlice` refs from `RawWeights` instead of Candle Tensors.
    /// Eliminates all 6 `storage_and_layout` calls that the v1 path had per layer:
    ///   - hidden: taken as `CudaView<f32>` (caller extracts once from Tensor)
    ///   - gate_inp_t: dead code removed (was unused in v1)
    ///   - gate_exps, up_exps, down_exps: from RawWeights (pre-extracted at load time)
    ///   - router_logits: replaced Candle matmul with `raw_f32_gemv`
    ///
    /// The shared expert still uses Candle (QMatMul) since those are cheap.
    /// The final output is a Tensor for compatibility with the residual add.
    pub(crate) fn moe_ffn_forward_raw_v2(
        hidden: &Tensor,
        moe: &MoeFFN,
        moe_raw: &crate::raw_weights::MoeRawWeightRefs<'_>,
        top_k: usize,
        hidden_size: usize,
        expert_ffn: usize,
        num_experts: usize,
        buffers: &mut crate::raw_forward::RawGpuBuffers,
        scratch: Option<&mut crate::scratch_pool::ScratchPool>,
    ) -> Result<Tensor> {
        use candle_core::Storage;
        let dev = hidden.device();
        let Device::Cuda(cuda_dev) = dev else {
            candle_core::bail!("raw MoE v2 requires CUDA");
        };

        let dprof = dispatch_prof_enabled();
        let dprof_report = if dprof {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static MOE_V2_DPROF_COUNT: AtomicUsize = AtomicUsize::new(0);
            let c = MOE_V2_DPROF_COUNT.fetch_add(1, Ordering::Relaxed);
            c == 39
        } else {
            false
        };

        // --- Extract hidden CudaView (1 storage_and_layout — unavoidable for Tensor input) ---
        // This is the ONLY storage_and_layout call in v2, vs 6 in v1.
        // It will be eliminated when the full raw forward path passes CudaSlice directly.
        let _t_sal = std::time::Instant::now();
        let (h_stor, h_lay) = hidden.storage_and_layout();
        let h_cuda = match &*h_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("") };
        let h_view = h_cuda.as_cuda_slice::<f32>()?.slice(h_lay.start_offset()..);
        let t_sal_ms = if dprof_report { _t_sal.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // Expert byte sizes (from pre-extracted shapes)
        const IQ3S_BLOCK_BYTES: usize = 110;
        const IQ3S_BLOCK_ELEMS: usize = 256;
        let expert_elements_gate = moe_raw.gate_exps_shape.1 * moe_raw.gate_exps_shape.2;
        let expert_elements_up = moe_raw.up_exps_shape.1 * moe_raw.up_exps_shape.2;
        let expert_elements_down = moe_raw.down_exps_shape.1 * moe_raw.down_exps_shape.2;
        let expert_bytes_gate = (expert_elements_gate / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_up = (expert_elements_up / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_down = (expert_elements_down / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;

        // --- Router via raw_f32_gemv: zero Tensor allocation ---
        let _t_router = std::time::Instant::now();
        {
            // gate_inp is [num_experts, hidden_size] row-major.
            // raw_f32_gemv computes y[e] = sum_h(gate_inp[e * hidden_size + h] * hidden[h])
            let gi_view = moe_raw.gate_inp.slice(..);
            crate::kernels::elementwise::raw_f32_gemv(
                &gi_view, &h_view, &mut buffers.gate_buf,
                num_experts, hidden_size, cuda_dev,
            )?;
        }

        // GPU top-K softmax on the router logits (already in buffers.gate_buf)
        {
            let gate_view = buffers.gate_buf.slice(..);
            crate::deltanet_kernel::gpu_topk_softmax(
                &gate_view,
                &mut buffers.topk_indices_buf,
                &mut buffers.topk_weights_buf,
                num_experts, top_k, cuda_dev,
            )?;
        }

        // Trace: read back indices/weights only if needed
        let trace = {
            use once_cell::sync::Lazy;
            static T: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_TRACE").is_ok());
            *T
        };
        let selected: Vec<(usize, f32)> = if trace {
            let indices_cpu: Vec<i32> = buffers.topk_indices_buf.stream().clone()
                .clone_dtoh(&buffers.topk_indices_buf)
                .map_err(|e| candle_core::Error::Msg(format!("topk indices dtoh: {e}")))?;
            let weights_cpu: Vec<f32> = buffers.topk_weights_buf.stream().clone()
                .clone_dtoh(&buffers.topk_weights_buf)
                .map_err(|e| candle_core::Error::Msg(format!("topk weights dtoh: {e}")))?;
            indices_cpu.iter().zip(weights_cpu.iter())
                .map(|(&i, &w)| (i as usize, w)).collect()
        } else {
            Vec::new()
        };
        let t_router_ms = if dprof_report { _t_router.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        let use_fused = {
            use once_cell::sync::Lazy;
            static FUSED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_FUSED_MOE").is_err());
            *FUSED
        };

        // --- Expert kernel launches (using pre-extracted CudaSlice refs) ---
        let _t_kernel = std::time::Instant::now();
        // Borrow views from the owned CudaSlice in RawWeights — zero-cost slice()
        let g_view = moe_raw.gate_exps.slice(..);
        let u_view = moe_raw.up_exps.slice(..);
        let d_view = moe_raw.down_exps.slice(..);

        if use_fused {
            crate::raw_forward::moe_ffn_fused(
                &h_view, &selected,
                &g_view, &u_view, &d_view,
                expert_bytes_gate, expert_bytes_up, expert_bytes_down,
                buffers,
                hidden_size, expert_ffn, top_k,
                cuda_dev,
            )?;
        } else {
            crate::raw_forward::moe_ffn_raw(
                &h_view, &selected,
                &g_view, &u_view, &d_view,
                expert_bytes_gate, expert_bytes_up, expert_bytes_down,
                expert_elements_gate, expert_elements_up, expert_elements_down,
                buffers,
                hidden_size, expert_ffn, num_experts, top_k,
                cuda_dev, dev,
            )?;
        }
        let t_kernel_ms = if dprof_report { _t_kernel.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

        // --- Shared expert + combine ---
        let _t_shared = std::time::Instant::now();
        let (t_shared_ms, t_wrap_ms);

        // Check if scratch pool path is available (all shared expert raw weights present)
        let use_scratch = scratch.is_some()
            && moe_raw.shared_gate.is_some()
            && moe_raw.shared_up.is_some()
            && moe_raw.shared_down.is_some()
            && moe_raw.gate_inp_shexp.is_some();

        let output = if use_scratch {
            // === SCRATCH POOL PATH: zero cudaMalloc for shared expert ===
            // Keep h_stor alive until we finish using h_view for the scratch path
            let scratch = scratch.unwrap();
            let gate_raw_view = moe_raw.shared_gate.unwrap().slice(..);
            let up_raw_view = moe_raw.shared_up.unwrap().slice(..);
            let down_raw_view = moe_raw.shared_down.unwrap().slice(..);
            let shexp_gate_view = moe_raw.gate_inp_shexp.unwrap().slice(..);

            scratch.shared_expert_forward(
                &h_view,
                &gate_raw_view,
                &up_raw_view,
                &down_raw_view,
                &shexp_gate_view,
                &buffers.combined_buf,
                cuda_dev,
            )?;
            t_shared_ms = if dprof_report { _t_shared.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

            // Drop h_stor before wrapping scratch output as Tensor
            drop(h_stor);

            // Wrap shexp_out_buf as Tensor [1, hidden_size] — zero-copy from CudaSlice
            let _t_wrap_start = std::time::Instant::now();
            let out_slice = scratch.shexp_out_buf.try_clone()
                .map_err(|e| candle_core::Error::Msg(format!("scratch out clone: {e}")))?;
            let out_storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
                out_slice, cuda_dev.clone());
            let output = Tensor::from_storage(
                candle_core::Storage::Cuda(out_storage),
                candle_core::Shape::from_dims(&[1, hidden_size]),
                candle_core::op::BackpropOp::none(), false);
            t_wrap_ms = if dprof_report { _t_wrap_start.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };
            output
        } else {
            // === CANDLE PATH: original shared expert (fallback) ===
            // Drop h_stor before using Candle ops on moe tensors
            drop(h_stor);

            let use_ggml_shexp = {
                use once_cell::sync::Lazy;
                static G: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
                *G
            };
            let sh_gate = if use_ggml_shexp && moe.gate_shexp_raw.is_some() {
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                    moe.gate_shexp_raw.as_ref().unwrap(), hidden,
                    512, hidden.dim(1)?)?
            } else { moe.gate_shexp.forward(hidden)? };
            let sh_up = if use_ggml_shexp && moe.up_shexp_raw.is_some() {
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                    moe.up_shexp_raw.as_ref().unwrap(), hidden,
                    512, hidden.dim(1)?)?
            } else { moe.up_shexp.forward(hidden)? };
            let sh_activated = sh_gate.silu()?;
            let sh_intermediate = (&sh_activated * &sh_up)?;
            let shared_out = if use_ggml_shexp && moe.down_shexp_raw.is_some() {
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                    moe.down_shexp_raw.as_ref().unwrap(), &sh_intermediate,
                    hidden.dim(1)?, 512)?
            } else { moe.down_shexp.forward(&sh_intermediate)? };
            let gate_logit = hidden.matmul(&moe.gate_inp_shexp.unsqueeze(1)?)?;
            let shared_scale = candle_nn::ops::sigmoid(&gate_logit)?;
            let gated_shared = shared_out.broadcast_mul(&shared_scale)?;
            t_shared_ms = if dprof_report { _t_shared.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };

            // --- In-place add: gated_shared += combined_buf (zero cudaMalloc) ---
            let _t_wrap_start = std::time::Instant::now();
            let add_op = crate::raw_forward::AddCudaSliceInplace::new(
                &buffers.combined_buf, hidden_size);
            gated_shared.inplace_op1(&add_op)?;
            t_wrap_ms = if dprof_report { _t_wrap_start.elapsed().as_secs_f64() * 1000.0 } else { 0.0 };
            gated_shared
        };

        if dprof_report {
            let total = t_sal_ms + t_router_ms + t_kernel_ms + t_wrap_ms + t_shared_ms;
            eprintln!("\n[DISPATCH-MOE-V2] ========= MoE FFN v2 Sub-operation Breakdown =========");
            eprintln!("[DISPATCH-MOE-V2] storage_and_layout (1 tensor): {:.3}ms", t_sal_ms);
            eprintln!("[DISPATCH-MOE-V2] router (raw_gemv+topk):        {:.3}ms", t_router_ms);
            eprintln!("[DISPATCH-MOE-V2] kernel (moe_ffn_{}):{:>13}{:.3}ms",
                if use_fused { "fused" } else { "raw  " }, " ", t_kernel_ms);
            eprintln!("[DISPATCH-MOE-V2] inplace_add (zero alloc):      {:.3}ms", t_wrap_ms);
            eprintln!("[DISPATCH-MOE-V2] shared_expert{}:{:>9}{:.3}ms",
                if use_scratch { " (scratch)" } else { " (Candle) " }, " ", t_shared_ms);
            eprintln!("[DISPATCH-MOE-V2] accounted total:               {:.3}ms", total);
            eprintln!("[DISPATCH-MOE-V2] ============================================================\n");
        }

        Ok(output)
    }

    pub(crate) fn moe_ffn_forward(hidden: &Tensor, moe: &MoeFFN, top_k: usize) -> Result<Tensor> {
        use candle_core::Storage;
        let dev = hidden.device();
        let hidden_size = hidden.dim(1)?;

        let moe_prof = {
            use once_cell::sync::Lazy;
            static P: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_MOE_PROFILE").is_ok());
            *P
        };
        let mut t_router = std::time::Duration::ZERO;
        let mut t_dequant = std::time::Duration::ZERO;
        let mut t_matmul = std::time::Duration::ZERO;
        let _t0 = std::time::Instant::now();

        // === 1. Router: select top-K experts ===
        // Router — compute on GPU, transfer only the softmax probs (256 floats = 1KB)
        let router_logits = hidden.matmul(&moe.gate_inp_t)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;
        // to_vec1 transfers 256 floats (1 KB) GPU→CPU — unavoidable for top-K selection
        let probs_vec: Vec<f32> = router_probs.squeeze(0)?.to_vec1()?;
        let (top_indices, top_weights) = crate::moe_forward::topk_probs(&probs_vec, top_k);
        if moe_prof { t_router = _t0.elapsed(); }

        // --- Tracer: router analysis (Candle reference path) ---
        {
            use once_cell::sync::Lazy;
            static TRACE_LVL: Lazy<usize> = Lazy::new(|| {
                std::env::var("CHIMERE_TRACE_LEVEL")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(0)
            });
            if *TRACE_LVL >= 1 {
                // We don't have `il` here (static method), use a counter to approximate layer index.
                use std::sync::atomic::{AtomicUsize, Ordering};
                static ROUTER_CALL: AtomicUsize = AtomicUsize::new(0);
                let call = ROUTER_CALL.fetch_add(1, Ordering::Relaxed);
                let pseudo_layer = call % 64; // approximate
                let tracer = crate::trace::Tracer::new();
                tracer.router_trace(pseudo_layer, &probs_vec, &top_indices, &top_weights);
            }
        }

        // === 2. Extract base CudaSlice ONCE (avoid per-expert lock acquisition) ===
        const IQ3S_BLOCK_BYTES: usize = 110;
        const IQ3S_BLOCK_ELEMS: usize = 256;

        let expert_elements_gate = moe.gate_exps_shape.1 * moe.gate_exps_shape.2;
        let expert_elements_up = moe.up_exps_shape.1 * moe.up_exps_shape.2;
        let expert_elements_down = moe.down_exps_shape.1 * moe.down_exps_shape.2;
        let expert_bytes_gate = (expert_elements_gate / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_up = (expert_elements_up / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_down = (expert_elements_down / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;

        // Per-expert shape arrays: constant for all experts in this layer.
        // gate/up: [expert_ffn, hidden]  e.g. [512, 2048]
        // down:    [hidden, expert_ffn]  e.g. [2048, 512]   (stored transposed vs gate/up)
        let gate_shape = [moe.gate_exps_shape.1, moe.gate_exps_shape.2];
        let up_shape   = [moe.up_exps_shape.1,   moe.up_exps_shape.2];
        let down_shape = [moe.down_exps_shape.1, moe.down_exps_shape.2];
        // For the fused gate+up matmul: concatenate along dim-0 → [2*expert_ffn, hidden]
        let expert_ffn = moe.gate_exps_shape.1; // e.g. 512

        // Hold storage guards for the duration — extracted ONCE, not per expert
        let (g_stor, g_lay) = moe.gate_exps_raw.storage_and_layout();
        let (u_stor, u_lay) = moe.up_exps_raw.storage_and_layout();
        let (d_stor, d_lay) = moe.down_exps_raw.storage_and_layout();
        let g_cuda = match &*g_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
        let u_cuda = match &*u_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
        let d_cuda = match &*d_stor { Storage::Cuda(c) => c, _ => candle_core::bail!("not CUDA") };
        let g_slice = g_cuda.as_cuda_slice::<u8>()?;
        let u_slice = u_cuda.as_cuda_slice::<u8>()?;
        let d_slice = d_cuda.as_cuda_slice::<u8>()?;
        let g_offset = g_lay.start_offset();
        let u_offset = u_lay.start_offset();
        let d_offset = d_lay.start_offset();

        // === 3. Process each selected expert using CudaView (no copy, no clone) ===
        //
        // Optimization: fuse gate+up dequant into a single matmul per expert.
        //   gate_w [expert_ffn, hidden] + up_w [expert_ffn, hidden]
        //     → cat(dim=0) → combined_wu [2*expert_ffn, hidden]
        //     → hidden [1, hidden] @ combined_wu.t() [hidden, 2*expert_ffn]
        //     → [1, 2*expert_ffn]   (one cuBLAS call instead of two)
        //   narrow(1, 0, expert_ffn) → gate_out
        //   narrow(1, expert_ffn, expert_ffn) → up_out   (both are views, zero copy)
        //
        // This reduces matmul kernel launches from 2→1 per expert:
        //   8 experts × saved 1 launch = 8 fewer CUDA kernel dispatches per layer.
        let mut combined = Tensor::zeros((1, hidden_size), candle_core::DType::F32, dev)?;

        for (&expert_idx, &weight) in top_indices.iter().zip(top_weights.iter()) {
            let _td = std::time::Instant::now();
            let gate_w = crate::deltanet_kernel::dequant_iq3s_at_offset(
                g_slice, g_offset + expert_idx * expert_bytes_gate,
                expert_elements_gate, &gate_shape, dev)?;
            let up_w = crate::deltanet_kernel::dequant_iq3s_at_offset(
                u_slice, u_offset + expert_idx * expert_bytes_up,
                expert_elements_up, &up_shape, dev)?;
            let down_w = crate::deltanet_kernel::dequant_iq3s_at_offset(
                d_slice, d_offset + expert_idx * expert_bytes_down,
                expert_elements_down, &down_shape, dev)?;
            if moe_prof { t_dequant += _td.elapsed(); }

            // Fused gate+up matmul: one cuBLAS GEMV instead of two.
            //   combined_wu: [2*expert_ffn, hidden] (e.g. [1024, 2048]) — one GPU memcpy
            //   fused_out: [1, 2*expert_ffn] — one cuBLAS kernel
            //   gate_out / up_out: views via narrow — zero copy
            let _tm = std::time::Instant::now();
            let combined_wu = Tensor::cat(&[&gate_w, &up_w], 0)?;
            let fused_out = hidden.matmul(&combined_wu.t()?)?; // [1, 2*expert_ffn]
            let gate_out = fused_out.narrow(1, 0, expert_ffn)?;
            let up_out   = fused_out.narrow(1, expert_ffn, expert_ffn)?;
            let activated    = gate_out.silu()?;
            let intermediate = (&activated * &up_out)?;
            let expert_out   = intermediate.matmul(&down_w.t()?)?;
            combined = (&combined + &(expert_out * weight as f64)?)?;
            if moe_prof { t_matmul += _tm.elapsed(); }
        }

        // Drop storage guards
        drop(g_stor); drop(u_stor); drop(d_stor);

        // === 4. Shared expert (Q5_K QMatMul — Candle native, no dequant) ===
        let _ts = std::time::Instant::now();
        let use_ggml_shexp = {
            use once_cell::sync::Lazy;
            static G: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
            *G
        };
        let sh_gate = if use_ggml_shexp && moe.gate_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.gate_shexp_raw.as_ref().unwrap(), hidden,
                512, hidden.dim(1)?)?
        } else { moe.gate_shexp.forward(hidden)? };
        let sh_up = if use_ggml_shexp && moe.up_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.up_shexp_raw.as_ref().unwrap(), hidden,
                512, hidden.dim(1)?)?
        } else { moe.up_shexp.forward(hidden)? };
        let sh_activated = sh_gate.silu()?;
        let sh_intermediate = (&sh_activated * &sh_up)?;
        let shared_out = if use_ggml_shexp && moe.down_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.down_shexp_raw.as_ref().unwrap(), &sh_intermediate,
                hidden.dim(1)?, 512)?
        } else { moe.down_shexp.forward(&sh_intermediate)? };

        // Shared expert gate: sigmoid(hidden @ gate_inp_shexp) → scalar gate.
        // gate_inp_shexp [hidden_size] is a weight vector — the dot product with hidden
        // produces a single scalar logit, then sigmoid gives a 0-1 gate value.
        // This matches llama.cpp: llm_build_lora_mm(shexp_gate, input) → [1,1] → sigmoid.
        let gate_logit = hidden.matmul(&moe.gate_inp_shexp.unsqueeze(1)?)?; // [1, hidden] @ [hidden, 1] → [1, 1]
        let shared_scale = candle_nn::ops::sigmoid(&gate_logit)?; // [1, 1]
        let gated_shared = shared_out.broadcast_mul(&shared_scale)?; // [1, hidden] * [1, 1]

        // Debug toggles for isolating MoE FFN bugs
        let output = (&combined + &gated_shared)?;
        if moe_prof {
            let t_shared = _ts.elapsed();
            use std::sync::atomic::{AtomicUsize, Ordering};
            static MOE_CALL: AtomicUsize = AtomicUsize::new(0);
            let call = MOE_CALL.fetch_add(1, Ordering::Relaxed);
            if call % 39 == 0 { // print once per token (39 layers)
                eprintln!("[MOE-PROF] router={:.2}ms dequant={:.2}ms matmul={:.2}ms shared={:.2}ms total={:.2}ms",
                    t_router.as_secs_f64()*1000.0, t_dequant.as_secs_f64()*1000.0,
                    t_matmul.as_secs_f64()*1000.0, t_shared.as_secs_f64()*1000.0,
                    (t_router+t_dequant+t_matmul+t_shared).as_secs_f64()*1000.0);
            }
        }
        Ok(output)
    }

    /// MoE FFN forward with CPU-offloaded routed experts (ncmoe path).
    ///
    /// **Batch copy approach** (like llama.cpp): expert weights stay on CPU, but for
    /// each token we assemble the 8 active experts' IQ3_S bytes into a pre-allocated
    /// CPU staging buffer, then do a SINGLE `memcpy_htod` to a pre-allocated GPU buffer.
    /// GEMV kernels then read from offsets within that GPU buffer.
    ///
    /// This replaces 24 individual `to_device()` calls (8 experts x 3 matrices) with
    /// 1 bulk `memcpy_htod` (~10.3 MB), eliminating CUDA driver overhead and Candle
    /// Tensor allocation/deallocation per transfer.
    pub(crate) fn moe_ffn_forward_cpu(
        hidden_gpu: &Tensor,
        moe: &MoeFFN,
        top_k: usize,
        gpu_device: &Device,
        ncmoe: &mut crate::state::NcmoeBufs,
    ) -> Result<Tensor> {
        let hidden_size = hidden_gpu.dim(1)?;
        let expert_ffn = moe.gate_exps_shape.1;
        let ncols_gate = moe.gate_exps_shape.2;
        let ncols_down = moe.down_exps_shape.2;
        let expert_bytes_gate = ncmoe.expert_bytes_gate;
        let expert_bytes_up = ncmoe.expert_bytes_up;
        let expert_bytes_down = ncmoe.expert_bytes_down;

        // === 1. Router: select top-K experts (on GPU) ===
        let router_logits = hidden_gpu.matmul(&moe.gate_inp_t)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;
        let probs_vec: Vec<f32> = router_probs.squeeze(0)?.to_vec1()?;
        let (top_indices, top_weights) = crate::moe_forward::topk_probs(&probs_vec, top_k);

        // === 2. Assemble active experts into CPU staging buffer ===
        // Layout: [gate_0..gate_{k-1} | up_0..up_{k-1} | down_0..down_{k-1}]
        let gate_section_size = top_k * expert_bytes_gate;
        let up_section_offset = gate_section_size;
        let up_section_size = top_k * expert_bytes_up;
        let down_section_offset = gate_section_size + up_section_size;

        {
            // Extract raw byte slices from CPU tensors (zero-copy via storage_and_layout)
            let (g_stor, _) = moe.gate_exps_raw.storage_and_layout();
            let (u_stor, _) = moe.up_exps_raw.storage_and_layout();
            let (d_stor, _) = moe.down_exps_raw.storage_and_layout();

            let g_all = match &*g_stor {
                candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>()?,
                _ => candle_core::bail!("ncmoe: gate_exps_raw expected on CPU"),
            };
            let u_all = match &*u_stor {
                candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>()?,
                _ => candle_core::bail!("ncmoe: up_exps_raw expected on CPU"),
            };
            let d_all = match &*d_stor {
                candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>()?,
                _ => candle_core::bail!("ncmoe: down_exps_raw expected on CPU"),
            };

            for (k, &eid) in top_indices.iter().enumerate() {
                let gs = eid * expert_bytes_gate;
                let gd = k * expert_bytes_gate;
                ncmoe.cpu_staging[gd..gd + expert_bytes_gate]
                    .copy_from_slice(&g_all[gs..gs + expert_bytes_gate]);

                let us = eid * expert_bytes_up;
                let ud = up_section_offset + k * expert_bytes_up;
                ncmoe.cpu_staging[ud..ud + expert_bytes_up]
                    .copy_from_slice(&u_all[us..us + expert_bytes_up]);

                let ds = eid * expert_bytes_down;
                let dd = down_section_offset + k * expert_bytes_down;
                ncmoe.cpu_staging[dd..dd + expert_bytes_down]
                    .copy_from_slice(&d_all[ds..ds + expert_bytes_down]);
            }
        }

        // === 3. Single bulk memcpy_htod: CPU staging -> GPU expert buffer ===
        let total_bytes = gate_section_size + up_section_size + top_k * expert_bytes_down;
        let cuda_dev = match gpu_device {
            Device::Cuda(d) => d,
            _ => candle_core::bail!("ncmoe requires CUDA"),
        };
        cuda_dev.memcpy_htod(
            &ncmoe.cpu_staging[..total_bytes],
            &mut ncmoe.gpu_expert_buf,
        ).map_err(|e| candle_core::Error::Msg(format!("ncmoe batch htod: {e}")))?;

        // === 4. GEMV kernels at offsets within the GPU buffer ===
        use candle_core::Storage;

        let hidden_flat = hidden_gpu.flatten_all()?.contiguous()?;
        let (i_stor, i_lay) = hidden_flat.storage_and_layout();
        let i_cuda = match &*i_stor {
            Storage::Cuda(c) => c,
            _ => candle_core::bail!("hidden not CUDA"),
        };
        let i_view = i_cuda.as_cuda_slice::<f32>()?.slice(
            i_lay.start_offset()..i_lay.start_offset() + hidden_flat.elem_count());
        let gpu_buf_view = ncmoe.gpu_expert_buf.slice(..);

        let mut combined_gpu = Tensor::zeros((1, hidden_size), candle_core::DType::F32, gpu_device)?;

        for (k, &weight) in top_weights.iter().enumerate() {
            // Gate GEMV
            let gate_offset = k * expert_bytes_gate;
            crate::kernels::gemv_iq3s_fused_at_offset(
                &gpu_buf_view, gate_offset,
                &i_view, &mut ncmoe.gate_out,
                expert_ffn, ncols_gate, cuda_dev,
            )?;

            // Up GEMV
            let up_offset = up_section_offset + k * expert_bytes_up;
            crate::kernels::gemv_iq3s_fused_at_offset(
                &gpu_buf_view, up_offset,
                &i_view, &mut ncmoe.up_out,
                expert_ffn, ncols_gate, cuda_dev,
            )?;

            // SwiGLU: silu(gate_out) * up_out
            let gate_owned = {
                let mut buf = cuda_dev.alloc_zeros::<f32>(expert_ffn)
                    .map_err(|e| candle_core::Error::Msg(format!("ncmoe gate_owned: {e}")))?;
                cuda_dev.memcpy_dtod(&ncmoe.gate_out, &mut buf)
                    .map_err(|e| candle_core::Error::Msg(format!("ncmoe gate dtod: {e}")))?;
                buf
            };
            let up_owned = {
                let mut buf = cuda_dev.alloc_zeros::<f32>(expert_ffn)
                    .map_err(|e| candle_core::Error::Msg(format!("ncmoe up_owned: {e}")))?;
                cuda_dev.memcpy_dtod(&ncmoe.up_out, &mut buf)
                    .map_err(|e| candle_core::Error::Msg(format!("ncmoe up dtod: {e}")))?;
                buf
            };

            let gate_tensor = {
                let s = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(gate_owned, cuda_dev.clone());
                Tensor::from_storage(Storage::Cuda(s), candle_core::Shape::from_dims(&[1, expert_ffn]),
                    candle_core::op::BackpropOp::none(), false)
            };
            let up_tensor = {
                let s = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(up_owned, cuda_dev.clone());
                Tensor::from_storage(Storage::Cuda(s), candle_core::Shape::from_dims(&[1, expert_ffn]),
                    candle_core::op::BackpropOp::none(), false)
            };
            let activated = gate_tensor.silu()?;
            let intermediate = (&activated * &up_tensor)?;

            // Down GEMV
            let inter_flat = intermediate.flatten_all()?.contiguous()?;
            let (inter_stor, inter_lay) = inter_flat.storage_and_layout();
            let inter_cuda = match &*inter_stor {
                Storage::Cuda(c) => c,
                _ => candle_core::bail!("inter not CUDA"),
            };
            let inter_view = inter_cuda.as_cuda_slice::<f32>()?.slice(
                inter_lay.start_offset()..inter_lay.start_offset() + inter_flat.elem_count());

            let down_offset = down_section_offset + k * expert_bytes_down;
            crate::kernels::gemv_iq3s_fused_at_offset(
                &gpu_buf_view, down_offset,
                &inter_view, &mut ncmoe.down_out,
                hidden_size, ncols_down, cuda_dev,
            )?;

            // Wrap down_out as Tensor for weighted accumulation
            let down_owned = {
                let mut buf = cuda_dev.alloc_zeros::<f32>(hidden_size)
                    .map_err(|e| candle_core::Error::Msg(format!("ncmoe down_owned: {e}")))?;
                cuda_dev.memcpy_dtod(&ncmoe.down_out, &mut buf)
                    .map_err(|e| candle_core::Error::Msg(format!("ncmoe down dtod: {e}")))?;
                buf
            };
            let expert_out = {
                let s = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(down_owned, cuda_dev.clone());
                Tensor::from_storage(Storage::Cuda(s), candle_core::Shape::from_dims(&[1, hidden_size]),
                    candle_core::op::BackpropOp::none(), false)
            };

            drop(inter_stor);

            combined_gpu = (&combined_gpu + &(expert_out * weight as f64)?)?;
        }

        drop(i_stor);

        // === 5. Shared expert on GPU (Q5_K QMatMul, Candle native, unchanged) ===
        let use_ggml_shexp = {
            use once_cell::sync::Lazy;
            static G: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
            *G
        };
        let sh_gate = if use_ggml_shexp && moe.gate_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.gate_shexp_raw.as_ref().unwrap(), hidden_gpu,
                512, hidden_gpu.dim(1)?)?
        } else { moe.gate_shexp.forward(hidden_gpu)? };
        let sh_up = if use_ggml_shexp && moe.up_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.up_shexp_raw.as_ref().unwrap(), hidden_gpu,
                512, hidden_gpu.dim(1)?)?
        } else { moe.up_shexp.forward(hidden_gpu)? };
        let sh_activated = sh_gate.silu()?;
        let sh_intermediate = (&sh_activated * &sh_up)?;
        let shared_out = if use_ggml_shexp && moe.down_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.down_shexp_raw.as_ref().unwrap(), &sh_intermediate,
                hidden_gpu.dim(1)?, 512)?
        } else { moe.down_shexp.forward(&sh_intermediate)? };

        // Shared expert gate
        let gate_logit = hidden_gpu.matmul(&moe.gate_inp_shexp.unsqueeze(1)?)?;
        let shared_scale = candle_nn::ops::sigmoid(&gate_logit)?;
        let gated_shared = shared_out.broadcast_mul(&shared_scale)?;

        let output = (&combined_gpu + &gated_shared)?;
        Ok(output)
    }

    /// MoE FFN forward with CPU-offloaded routed experts using ggml IQ3_S AVX2 GEMV.
    ///
    /// **"Invert ncmoe" approach**: instead of copying 10 MB of expert weights
    /// to GPU per token (batch copy), copy 8 KB of hidden state to CPU and do
    /// the GEMV using ggml's AVX2-optimized IQ3_S kernels.
    ///
    /// Toggle: `CHIMERE_NCMOE_CPU=1`
    ///
    /// Saves 10 MB PCIe bandwidth per ncmoe layer per token. For 4 ncmoe layers
    /// with 8 experts each, that's 40 MB saved per token.
    ///
    /// The shared expert still runs on GPU (Q5_K QMatMul, unchanged).
    pub(crate) fn moe_ffn_forward_ncmoe_cpu(
        hidden_gpu: &Tensor,
        moe: &MoeFFN,
        top_k: usize,
        gpu_device: &Device,
    ) -> Result<Tensor> {
        let hidden_size = hidden_gpu.dim(1)?;
        let expert_ffn = moe.gate_exps_shape.1;
        let ncols_gate = moe.gate_exps_shape.2;
        let ncols_down = moe.down_exps_shape.2;

        // === 1. Router: select top-K experts (on GPU) ===
        let router_logits = hidden_gpu.matmul(&moe.gate_inp_t)?;
        let router_probs = candle_nn::ops::softmax(&router_logits, candle_core::D::Minus1)?;
        let probs_vec: Vec<f32> = router_probs.squeeze(0)?.to_vec1()?;
        let (top_indices, top_weights) = crate::moe_forward::topk_probs(&probs_vec, top_k);

        // === 2. Extract raw IQ3_S byte slices from CPU tensors ===
        let (g_stor, _) = moe.gate_exps_raw.storage_and_layout();
        let (u_stor, _) = moe.up_exps_raw.storage_and_layout();
        let (d_stor, _) = moe.down_exps_raw.storage_and_layout();

        let g_all = match &*g_stor {
            candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>()?,
            _ => candle_core::bail!("ncmoe_cpu: gate_exps_raw expected on CPU"),
        };
        let u_all = match &*u_stor {
            candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>()?,
            _ => candle_core::bail!("ncmoe_cpu: up_exps_raw expected on CPU"),
        };
        let d_all = match &*d_stor {
            candle_core::Storage::Cpu(cpu) => cpu.as_slice::<u8>()?,
            _ => candle_core::bail!("ncmoe_cpu: down_exps_raw expected on CPU"),
        };

        // === 3. CPU IQ3_S GEMV for routed experts ===
        let routed_out = crate::ggml_backend::ncmoe_cpu_experts_forward(
            hidden_gpu,
            g_all, u_all, d_all,
            &top_indices, &top_weights,
            expert_ffn, hidden_size,
            ncols_gate, ncols_down,
            gpu_device,
        )?;

        drop(g_stor);
        drop(u_stor);
        drop(d_stor);

        // === 4. Shared expert on GPU (Q5_K QMatMul, unchanged) ===
        let use_ggml_shexp = {
            use once_cell::sync::Lazy;
            static G: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K").is_ok());
            *G
        };
        let sh_gate = if use_ggml_shexp && moe.gate_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.gate_shexp_raw.as_ref().unwrap(), hidden_gpu,
                512, hidden_gpu.dim(1)?)?
        } else { moe.gate_shexp.forward(hidden_gpu)? };
        let sh_up = if use_ggml_shexp && moe.up_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.up_shexp_raw.as_ref().unwrap(), hidden_gpu,
                512, hidden_gpu.dim(1)?)?
        } else { moe.up_shexp.forward(hidden_gpu)? };
        let sh_activated = sh_gate.silu()?;
        let sh_intermediate = (&sh_activated * &sh_up)?;
        let shared_out = if use_ggml_shexp && moe.down_shexp_raw.is_some() {
            crate::kernels::q5k_mmvq_ggml::ggml_q5k_tensor_forward(
                moe.down_shexp_raw.as_ref().unwrap(), &sh_intermediate,
                hidden_gpu.dim(1)?, 512)?
        } else { moe.down_shexp.forward(&sh_intermediate)? };

        // Shared expert gate
        let gate_logit = hidden_gpu.matmul(&moe.gate_inp_shexp.unsqueeze(1)?)?;
        let shared_scale = candle_nn::ops::sigmoid(&gate_logit)?;
        let gated_shared = shared_out.broadcast_mul(&shared_scale)?;

        let output = (&routed_out + &gated_shared)?;
        Ok(output)
    }
}

// ===========================================================================
// Pure cudarc MoE FFN forward — zero Candle Tensor allocations
// ===========================================================================
//
// This function operates entirely on pre-allocated CudaSlice buffers from
// ComputeGraph. No Tensor creation, no storage_and_layout, no QMatMul.
//
// Input:  graph.normed  [hidden_size]  (post-RMSNorm, written by caller)
// Output: graph.hidden  [hidden_size]  (routed + gated_shared, caller adds residual)
//
// Two paths:
//   GPU:  fused_moe_iq3s_gpu_resident (single kernel, all 8 experts)
//   CPU:  ggml_ffi::iq3s_gemv_cpu_parallel (ncmoe offloaded layers)

use crate::qwen35_model::compute_graph::{ComputeGraph, MoeWeightsRaw};

/// Pure cudarc MoE FFN forward pass. Zero Candle Tensor allocations.
///
/// Reads from `graph.normed` (post-RMSNorm hidden state).
/// Writes the full MoE output (routed experts + gated shared expert)
/// into `graph.hidden`. The caller is responsible for the residual add
/// (`hidden += residual`) after this function returns.
///
/// For GPU-resident experts, uses the fused MoE IQ3_S kernel (single
/// launch for all 8 active experts). For ncmoe-offloaded layers, copies
/// the 8 KB hidden state to CPU and runs ggml IQ3_S GEMV with AVX2.
///
/// The shared expert always runs on GPU via ggml Q5_K MMVQ kernels
/// with pre-quantized Q8_1 input.
pub(crate) fn forward_moe_cudarc(
    moe_w: &MoeWeightsRaw,
    graph: &mut ComputeGraph,
) -> candle_core::Result<()> {
    forward_moe_cudarc_with_cache(moe_w, graph, None, 0)
}

/// MoE FFN cudarc forward with optional cached weight pointers.
///
/// When `cached` is Some, uses pre-extracted raw CUDA pointers for weight
/// tensors in the batched MoE path, avoiding `device_ptr()` overhead on
/// the large expert weight slices (gate_exps, up_exps, down_exps).
///
/// `layer_idx` is the global layer index (0-based), used to index into
/// the cached pointer arrays.
pub(crate) fn forward_moe_cudarc_with_cache(
    moe_w: &MoeWeightsRaw,
    graph: &mut ComputeGraph,
    cached: Option<&super::compute_graph::CachedWeightPtrs>,
    layer_idx: usize,
) -> candle_core::Result<()> {
    let dev = &graph.dev;
    let hs = graph.hidden_size;
    let eff = graph.expert_ffn;      // 512 (routed expert intermediate)
    let seff = graph.shexp_ffn;      // 512 (shared expert intermediate, may differ)
    let n_experts = graph.n_experts; // 256
    let top_k = graph.top_k;        // 8

    // =====================================================================
    // 1. Router: normed × router_weight -> router_out [n_experts]
    //    router_weight is F32 [n_experts, hidden_size] row-major on GPU.
    //    raw_f32_gemv computes y[e] = sum_h(W[e*hs + h] * x[h]).
    // =====================================================================
    {
        let rw_view = moe_w.router_weight.slice(..);
        let n_view = graph.normed.slice(..);
        crate::kernels::elementwise::raw_f32_gemv(
            &rw_view,
            &n_view,
            &mut graph.router_out,
            n_experts,
            hs,
            dev,
        )?;
    }

    // =====================================================================
    // 2. GPU top-K softmax -> topk_indices [top_k], topk_weights [top_k]
    //    Stays fully on GPU. Zero CPU sync, zero PCIe transfer.
    // =====================================================================
    {
        let rl_view = graph.router_out.slice(..);
        crate::kernels::topk_softmax::gpu_topk_softmax(
            &rl_view,
            &mut graph.topk_indices,
            &mut graph.topk_weights,
            n_experts,
            top_k,
            dev,
        )?;
    }

    // =====================================================================
    // 3. Routed experts
    // =====================================================================
    // Zero the output accumulator
    crate::kernels::fused_ops::zero_f32(
        &mut graph.expert_accum,
        hs,
        dev,
    )?;

    // Toggle: CHIMERE_FUSED_MOE=1 forces the fused GPU-resident MoE path
    // even when CHIMERE_GGML_GPU is set. This eliminates the clone_dtoh sync
    // at the cost of using chimere's naive IQ3_S dequant instead of ggml's MMVQ.
    let force_fused = {
        use once_cell::sync::Lazy;
        static FUSED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FUSED_MOE").is_ok());
        *FUSED
    };

    // Phase 1.2: batched MoE is DEFAULT (zero-sync, +26.6%).
    // Set CHIMERE_NO_BATCHED_MOE=1 to fall back to V1 sequential path.
    let batched_moe = {
        use once_cell::sync::Lazy;
        static BATCHED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_BATCHED_MOE").is_err());
        *BATCHED
    };

    if !moe_w.experts_on_cpu && graph.ggml_gpu_bufs.is_some() && !force_fused && batched_moe {
        // =================================================================
        // V3: GPU-resident batched MoE dispatch — ZERO CPU SYNC
        //
        // Uses ggml MMVQ with expert indirection (ids_data in mmvq_args).
        // Expert IDs stay on GPU from topk_softmax. No memcpy_dtoh.
        // 7 kernel launches total per MoE layer.
        //
        // When cached pointers are available, skips CudaView creation for
        // the 3 expert weight tensors (saves 3 device_ptr() calls/layer).
        // =================================================================
        let ebg = moe_w.expert_bytes_gate;
        let ebu = moe_w.expert_bytes_up;
        let ebd = moe_w.expert_bytes_down;

        // Check for cached expert weight pointers
        let moe_cache = cached.map(|c| (
            c.moe_gate_exps.get(layer_idx).copied(),
            c.moe_up_exps.get(layer_idx).copied(),
            c.moe_down_exps.get(layer_idx).copied(),
        ));

        let mut ggml_bufs = graph.ggml_gpu_bufs.take().unwrap();

        // Phase 3.1: Check for stream override (set during CUDA Graph capture).
        // When active, use _on_stream variants so ggml FFI calls are captured
        // on the same stream as cudarc kernels.
        let stream_ptr = ggml_bufs.active_stream();
        let use_stream = !stream_ptr.is_null();

        // 1. Q8_1 quantize normed hidden state (shared input for all experts)
        {
            let n_view = graph.normed.slice(..);
            if use_stream {
                crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1_on_stream(
                    &n_view,
                    &mut ggml_bufs.q8_input,
                    hs,
                    dev,
                    stream_ptr,
                )?;
            } else {
                crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                    &n_view,
                    &mut ggml_bufs.q8_input,
                    hs,
                    dev,
                )?;
            }
        }

        // 2. Batched gate GEMV: all top_k experts in 1 launch
        //    gate_exps_raw[expert_id * ebg ..] @ Q8_1(normed) → batched_gate[k * eff]
        if use_stream {
            if let Some((Some(g_ptr), _, _)) = moe_cache {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_cached_on_stream(
                    g_ptr,
                    &ggml_bufs.q8_input,
                    &mut ggml_bufs.batched_gate,
                    &graph.topk_indices,
                    eff, hs, ebg, 0,
                    top_k,
                    stream_ptr,
                )?;
            } else {
                let g_view = moe_w.gate_exps_raw.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_on_stream(
                    &g_view,
                    &ggml_bufs.q8_input,
                    &mut ggml_bufs.batched_gate,
                    &graph.topk_indices,
                    eff, hs, ebg, 0,
                    top_k,
                    stream_ptr,
                )?;
            }
        } else if let Some((Some(g_ptr), _, _)) = moe_cache {
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_cached(
                g_ptr,
                &ggml_bufs.q8_input,
                &mut ggml_bufs.batched_gate,
                &graph.topk_indices,
                eff, hs, ebg, 0,
                top_k,
            )?;
        } else {
            let g_view = moe_w.gate_exps_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched(
                &g_view,
                &ggml_bufs.q8_input,
                &mut ggml_bufs.batched_gate,
                &graph.topk_indices,
                eff, hs, ebg, 0,
                top_k,
            )?;
        }

        // 3. Batched up GEMV: all top_k experts in 1 launch
        if use_stream {
            if let Some((_, Some(u_ptr), _)) = moe_cache {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_cached_on_stream(
                    u_ptr,
                    &ggml_bufs.q8_input,
                    &mut ggml_bufs.batched_up,
                    &graph.topk_indices,
                    eff, hs, ebu, 0,
                    top_k,
                    stream_ptr,
                )?;
            } else {
                let u_view = moe_w.up_exps_raw.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_on_stream(
                    &u_view,
                    &ggml_bufs.q8_input,
                    &mut ggml_bufs.batched_up,
                    &graph.topk_indices,
                    eff, hs, ebu, 0,
                    top_k,
                    stream_ptr,
                )?;
            }
        } else if let Some((_, Some(u_ptr), _)) = moe_cache {
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_cached(
                u_ptr,
                &ggml_bufs.q8_input,
                &mut ggml_bufs.batched_up,
                &graph.topk_indices,
                eff, hs, ebu, 0,
                top_k,
            )?;
        } else {
            let u_view = moe_w.up_exps_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched(
                &u_view,
                &ggml_bufs.q8_input,
                &mut ggml_bufs.batched_up,
                &graph.topk_indices,
                eff, hs, ebu, 0,
                top_k,
            )?;
        }

        // 4. Batched SiLU*mul: gate[i] = silu(gate[i]) * up[i] for all experts
        //    Element-wise on contiguous memory [top_k * eff]
        //    (cudarc kernel launch — automatically on device stream, no changes needed)
        crate::kernels::fused_ops::silu_mul_inplace_slices(
            &mut ggml_bufs.batched_gate,
            &ggml_bufs.batched_up,
            top_k * eff,
            dev,
        )?;

        // 5. Batched Q8_1 quantize intermediates (top_k vectors of eff elements)
        if use_stream {
            crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1_batched_on_stream(
                &ggml_bufs.batched_gate,
                &mut ggml_bufs.batched_q8_inter,
                eff,
                top_k,
                stream_ptr,
            )?;
        } else {
            crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1_batched(
                &ggml_bufs.batched_gate,
                &mut ggml_bufs.batched_q8_inter,
                eff,
                top_k,
            )?;
        }

        // 6. Batched down GEMV: all top_k experts in 1 launch
        //    down_exps_raw[expert_id * ebd ..] @ Q8_1(intermediate[k]) → batched_down[k * hs]
        //    input_stride = Q8_1 row bytes for expert_ffn (each expert has its own intermediate)
        let eff_padded = crate::kernels::ggml_gpu::pad(eff, crate::kernels::ggml_gpu::MATRIX_ROW_PADDING);
        let q8_row_bytes = (eff_padded / crate::kernels::ggml_gpu::Q8_1_BLOCK_ELEMS)
            * crate::kernels::ggml_gpu::Q8_1_BLOCK_BYTES;
        if use_stream {
            if let Some((_, _, Some(d_ptr))) = moe_cache {
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_cached_on_stream(
                    d_ptr,
                    &ggml_bufs.batched_q8_inter,
                    &mut ggml_bufs.batched_down,
                    &graph.topk_indices,
                    hs, eff, ebd, q8_row_bytes,
                    top_k,
                    stream_ptr,
                )?;
            } else {
                let d_view = moe_w.down_exps_raw.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_on_stream(
                    &d_view,
                    &ggml_bufs.batched_q8_inter,
                    &mut ggml_bufs.batched_down,
                    &graph.topk_indices,
                    hs, eff, ebd, q8_row_bytes,
                    top_k,
                    stream_ptr,
                )?;
            }
        } else if let Some((_, _, Some(d_ptr))) = moe_cache {
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched_cached(
                d_ptr,
                &ggml_bufs.batched_q8_inter,
                &mut ggml_bufs.batched_down,
                &graph.topk_indices,
                hs, eff, ebd, q8_row_bytes,
                top_k,
            )?;
        } else {
            let d_view = moe_w.down_exps_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_batched(
                &d_view,
                &ggml_bufs.batched_q8_inter,
                &mut ggml_bufs.batched_down,
                &graph.topk_indices,
                hs, eff, ebd, q8_row_bytes,
                top_k,
            )?;
        }

        // 7. Weighted combine: accum[h] = sum_k(weights[k] * down[k*hs + h])
        //    Reads topk_weights from GPU — zero CPU sync
        crate::kernels::fused_ops::weighted_combine_gpu(
            &ggml_bufs.batched_down,
            &graph.topk_weights,
            &mut graph.expert_accum,
            hs,
            top_k,
            dev,
        )?;

        graph.ggml_gpu_bufs = Some(ggml_bufs);
    } else if !moe_w.experts_on_cpu && graph.ggml_gpu_bufs.is_some() && !force_fused {
        // -----------------------------------------------------------------
        // V1: GPU path: ggml MMVQ with CPU sync (CHIMERE_GGML_GPU=1)
        //
        // Per-expert loop with memcpy_dtoh sync. Kept as fallback.
        // -----------------------------------------------------------------
        let ebg = moe_w.expert_bytes_gate;
        let ebu = moe_w.expert_bytes_up;
        let ebd = moe_w.expert_bytes_down;

        let g_view = moe_w.gate_exps_raw.slice(..);
        let u_view = moe_w.up_exps_raw.slice(..);
        let d_view = moe_w.down_exps_raw.slice(..);

        let mut ggml_bufs = graph.ggml_gpu_bufs.take().unwrap();

        {
            let n_view = graph.normed.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_quantize_q8_1(
                &n_view,
                &mut ggml_bufs.q8_input,
                hs,
                dev,
            )?;
        }

        let mut indices_buf = [0i32; 8];
        let mut weights_buf = [0.0f32; 8];
        dev.memcpy_dtoh(&graph.topk_indices, &mut indices_buf)
            .map_err(|e| candle_core::Error::Msg(format!("ggml_gpu topk indices dtoh: {e}")))?;
        dev.memcpy_dtoh(&graph.topk_weights, &mut weights_buf)
            .map_err(|e| candle_core::Error::Msg(format!("ggml_gpu topk weights dtoh: {e}")))?;

        for k in 0..top_k {
            let eid_i32 = indices_buf[k];
            let weight = weights_buf[k];
            let eid = eid_i32 as usize;

            let gate_offset = eid * ebg;
            let gate_w_view = g_view.slice(gate_offset..gate_offset + ebg);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_q8(
                &gate_w_view,
                &ggml_bufs.q8_input,
                &mut graph.gate_buf,
                eff,
                hs,
            )?;

            let up_offset = eid * ebu;
            let up_w_view = u_view.slice(up_offset..up_offset + ebu);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_q8(
                &up_w_view,
                &ggml_bufs.q8_input,
                &mut graph.up_buf,
                eff,
                hs,
            )?;

            crate::kernels::fused_ops::silu_mul_inplace_slices(
                &mut graph.gate_buf,
                &graph.up_buf,
                eff,
                dev,
            )?;

            // Fused: quantize intermediate + down GEMV in one FFI call
            let down_offset = eid * ebd;
            let down_w_view = d_view.slice(down_offset..down_offset + ebd);
            {
                let inter_view = graph.gate_buf.slice(..);
                crate::kernels::ggml_gpu::ggml_gpu_gemv_iq3s_f32(
                    &down_w_view,
                    &inter_view,
                    &mut graph.down_buf,
                    &mut ggml_bufs.q8_input_b,
                    hs,
                    eff,
                )?;
            }

            crate::kernels::fused_ops::scale_add_inplace_slices(
                &mut graph.expert_accum,
                &graph.down_buf,
                weight,
                hs,
                dev,
            )?;
        }

        graph.ggml_gpu_bufs = Some(ggml_bufs);
    } else if !moe_w.experts_on_cpu {
        // -----------------------------------------------------------------
        // GPU path: fused MoE IQ3_S kernel (single launch, all 8 experts)
        // Fallback when CHIMERE_GGML_GPU is not set.
        // -----------------------------------------------------------------
        #[cfg(feature = "cubin_fallback")]
        {
            let g_view = moe_w.gate_exps_raw.slice(..);
            let u_view = moe_w.up_exps_raw.slice(..);
            let d_view = moe_w.down_exps_raw.slice(..);

            crate::kernels::fused_moe::fused_moe_iq3s_gpu_resident(
                &graph.normed.slice(..),
                &g_view,
                &u_view,
                &d_view,
                &graph.topk_indices,
                &graph.topk_weights,
                &mut graph.expert_accum,
                hs,
                eff,
                moe_w.expert_bytes_gate,
                moe_w.expert_bytes_up,
                moe_w.expert_bytes_down,
                top_k,
                dev,
            )?;
        }
        #[cfg(not(feature = "cubin_fallback"))]
        panic!("cubin fallback disabled — set CHIMERE_GGML_GPU=1 or enable feature cubin_fallback");
    } else {
        // -----------------------------------------------------------------
        // CPU path (ncmoe): download hidden, run ggml IQ3_S GEMV on CPU
        // -----------------------------------------------------------------
        // Download topk indices + weights to CPU (64 bytes, negligible)
        let indices_cpu: Vec<i32> = graph.topk_indices.stream().clone()
            .clone_dtoh(&graph.topk_indices)
            .map_err(|e| candle_core::Error::Msg(format!("ncmoe topk indices dtoh: {e}")))?;
        let weights_cpu: Vec<f32> = graph.topk_weights.stream().clone()
            .clone_dtoh(&graph.topk_weights)
            .map_err(|e| candle_core::Error::Msg(format!("ncmoe topk weights dtoh: {e}")))?;

        // Download hidden state to CPU (8 KB for hidden_size=2048)
        let hidden_cpu: Vec<f32> = graph.normed.stream().clone()
            .clone_dtoh(&graph.normed)
            .map_err(|e| candle_core::Error::Msg(format!("ncmoe hidden dtoh: {e}")))?;

        let gate_cpu = moe_w.gate_exps_cpu.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("ncmoe: gate_exps_cpu is None but experts_on_cpu=true".into())
        })?;
        let up_cpu = moe_w.up_exps_cpu.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("ncmoe: up_exps_cpu is None but experts_on_cpu=true".into())
        })?;
        let down_cpu = moe_w.down_exps_cpu.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("ncmoe: down_exps_cpu is None but experts_on_cpu=true".into())
        })?;

        let ebg = moe_w.expert_bytes_gate;
        let ebu = moe_w.expert_bytes_up;
        let ebd = moe_w.expert_bytes_down;

        // CPU accumulator for weighted expert outputs
        let mut combined_cpu = vec![0.0f32; hs];

        for (&eid_i32, &weight) in indices_cpu.iter().zip(weights_cpu.iter()) {
            let eid = eid_i32 as usize;

            // Gate GEMV: gate_exps[eid] @ normed -> [expert_ffn]
            let g_start = eid * ebg;
            let gate_out = ggml_ffi::iq3s_gemv_cpu_parallel(
                &gate_cpu[g_start..g_start + ebg],
                &hidden_cpu,
                eff,
                hs,
                4, // 4 threads
            );

            // Up GEMV: up_exps[eid] @ normed -> [expert_ffn]
            let u_start = eid * ebu;
            let up_out = ggml_ffi::iq3s_gemv_cpu_parallel(
                &up_cpu[u_start..u_start + ebu],
                &hidden_cpu,
                eff,
                hs,
                4,
            );

            // SwiGLU on CPU: silu(gate) * up
            let mut intermediate = vec![0.0f32; eff];
            for i in 0..eff {
                let g = gate_out[i];
                let s = g / (1.0 + (-g).exp()); // silu
                intermediate[i] = s * up_out[i];
            }

            // Down GEMV: down_exps[eid] @ intermediate -> [hidden_size]
            let d_start = eid * ebd;
            let down_out = ggml_ffi::iq3s_gemv_cpu_parallel(
                &down_cpu[d_start..d_start + ebd],
                &intermediate,
                hs,
                eff,
                4,
            );

            // Weighted accumulate
            for i in 0..hs {
                combined_cpu[i] += weight * down_out[i];
            }
        }

        // Upload accumulated output to GPU expert_accum
        dev.memcpy_htod(&combined_cpu, &mut graph.expert_accum)
            .map_err(|e| candle_core::Error::Msg(format!("ncmoe upload combined: {e}")))?;
    }

    // =====================================================================
    // 4. Shared expert (Q5_K weights, always on GPU)
    // =====================================================================

    if graph.ggml_gpu_bufs.is_some() {
        // -----------------------------------------------------------------
        // ggml GPU MMVQ path for shared expert Q5_K
        // -----------------------------------------------------------------
        let mut ggml_bufs = graph.ggml_gpu_bufs.take().unwrap();

        // 4a+4b fused: Quantize normed to Q8_1 + Gate GEMV in one FFI call.
        // The Q8_1 scratch is populated as a side effect, reused by 4c (up GEMV).
        {
            let n_view = graph.normed.slice(..);
            let gate_view = moe_w.shexp_gate_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32(
                &gate_view,
                &n_view,
                &mut graph.shexp_gate,
                &mut ggml_bufs.q8_input,
                seff,
                hs,
            )?;
        }

        // 4c. Up GEMV: Q5_K [shexp_ffn, hidden_size] @ Q8_1(normed) -> shexp_up
        {
            let up_view = moe_w.shexp_up_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_q8(
                &up_view,
                &ggml_bufs.q8_input,
                &mut graph.shexp_up,
                seff,
                hs,
            )?;
        }

        // 4d. SwiGLU: shexp_gate = silu(shexp_gate) * shexp_up (in-place)
        crate::kernels::fused_ops::silu_mul_inplace_slices(
            &mut graph.shexp_gate,
            &graph.shexp_up,
            seff,
            dev,
        )?;

        // 4e fused: Re-quantize SwiGLU intermediate + Down GEMV in one FFI call.
        {
            let inter_view = graph.shexp_gate.slice(..);
            let down_view = moe_w.shexp_down_raw.slice(..);
            crate::kernels::ggml_gpu::ggml_gpu_gemv_q5k_f32(
                &down_view,
                &inter_view,
                &mut graph.shexp_down,
                &mut ggml_bufs.q8_input,
                hs,
                seff,
            )?;
        }

        // Put ggml_gpu_bufs back
        graph.ggml_gpu_bufs = Some(ggml_bufs);
    } else {
        // -----------------------------------------------------------------
        // Fallback: existing Q5_K MMVQ path (PTX/cubin kernels)
        // -----------------------------------------------------------------
        #[cfg(feature = "cubin_fallback")]
        {
            // 4a. Quantize normed to Q8_1 (reused for gate AND up projections)
            {
                let n_view = graph.normed.slice(..);
                crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
                    &n_view,
                    &mut graph.q5k_bufs.q8_input,
                    hs,
                    dev,
                )?;
            }

            // 4b. Gate GEMV: Q5_K [shexp_ffn, hidden_size] @ Q8_1(normed) -> shexp_gate [shexp_ffn]
            {
                let gate_view = moe_w.shexp_gate_raw.slice(..);
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                    &gate_view,
                    &graph.q5k_bufs.q8_input,
                    &mut graph.shexp_gate,
                    seff,
                    hs,
                    dev,
                )?;
            }

            // 4c. Up GEMV: Q5_K [shexp_ffn, hidden_size] @ Q8_1(normed) -> shexp_up [shexp_ffn]
            {
                let up_view = moe_w.shexp_up_raw.slice(..);
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                    &up_view,
                    &graph.q5k_bufs.q8_input,
                    &mut graph.shexp_up,
                    seff,
                    hs,
                    dev,
                )?;
            }

            // 4d. SwiGLU: shexp_gate = silu(shexp_gate) * shexp_up (in-place)
            crate::kernels::fused_ops::silu_mul_inplace_slices(
                &mut graph.shexp_gate,
                &graph.shexp_up,
                seff,
                dev,
            )?;

            // 4e. Down GEMV: Q5_K [hidden_size, shexp_ffn] @ Q8_1(intermediate) -> shexp_down [hidden_size]
            //     First re-quantize the SwiGLU intermediate to Q8_1 (size = shexp_ffn, not hidden_size)
            {
                let inter_view = graph.shexp_gate.slice(..);
                crate::kernels::q5k_mmvq_ggml::ggml_quantize_q8_1(
                    &inter_view,
                    &mut graph.q5k_bufs.q8_input,
                    seff,
                    dev,
                )?;
            }
            {
                let down_view = moe_w.shexp_down_raw.slice(..);
                crate::kernels::q5k_mmvq_ggml::ggml_q5k_gemv(
                    &down_view,
                    &graph.q5k_bufs.q8_input,
                    &mut graph.shexp_down,
                    hs,
                    seff,
                    dev,
                )?;
            }
        }
        #[cfg(not(feature = "cubin_fallback"))]
        panic!("cubin fallback disabled — set CHIMERE_GGML_GPU=1 or enable feature cubin_fallback");
    }

    // =====================================================================
    // 5. Sigmoid gate: dot(normed, shexp_gate_proj) -> scalar logit
    //    shexp_gate_proj is [hidden_size] F32.
    // =====================================================================
    {
        let n_view = graph.normed.slice(..);
        let gp_view = moe_w.shexp_gate_proj.slice(..);
        crate::scratch_pool::raw_dot_product(
            &n_view,
            &gp_view,
            &mut graph.shexp_gate_logit,
            hs,
            dev,
        )?;
    }

    // =====================================================================
    // 6. Combine: hidden = expert_accum + sigmoid(gate_logit) * shexp_down
    //
    //    sigmoid_gate_add_inplace does:
    //      shexp_down[i] = shexp_down[i] * sigmoid(gate_logit[0]) + expert_accum[i]
    //    Then copy shexp_down -> hidden.
    // =====================================================================
    crate::scratch_pool::raw_sigmoid_gate_add_inplace(
        &mut graph.shexp_down,
        &graph.shexp_gate_logit,
        &graph.expert_accum,
        hs,
        dev,
    )?;

    // Copy combined result into hidden (shexp_down now holds the full MoE output)
    dev.memcpy_dtod(&graph.shexp_down, &mut graph.hidden)
        .map_err(|e| candle_core::Error::Msg(format!("moe cudarc dtod result: {e}")))?;

    Ok(())
}
