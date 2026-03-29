//! ggml FFI backend for lm_head, Q5_K validation, and ncmoe CPU GEMV.
//!
//! Provides CPU-based quantized GEMV using two paths:
//!
//! ## 1. Validation (slow, correct)
//!
//!   GPU tensor (f32/u8) -> CPU Vec -> pure-Rust GEMV -> CPU Vec -> GPU Tensor
//!
//! ## 2. ncmoe CPU GEMV (fast, ggml AVX2)
//!
//!   GPU hidden state (8 KB) -> CPU -> ggml IQ3_S GEMV (AVX2) -> CPU -> GPU
//!
//! The "invert ncmoe" approach: instead of copying 10 MB of expert weights
//! to GPU per token, copy 8 KB of hidden state to CPU and do the matmul
//! using ggml's AVX2-optimized IQ3_S kernels. For a 512x2048 GEMV, ggml
//! takes ~0.3ms per expert on CPU (i5-14600KF, AVX2).
//!
//! ## Toggles
//!
//! - `CHIMERE_GGML_LM_HEAD=1`: Q8_0 CPU GEMV for lm_head (output.weight)
//! - `CHIMERE_GGML_Q5K_CPU=1`: Q5_K CPU GEMV for attn_qkv layer 0 only
//! - `CHIMERE_NCMOE_CPU=1`: Use CPU ggml IQ3_S for ncmoe expert GEMV
//!   (replaces GPU batch copy path with CPU-side AVX2 GEMV)
//!
//! ## Integration
//!
//! Called from `qwen35_model.rs::forward_token_preloaded` (lm_head),
//! `forward_gdn_layer_moe` (Q5_K validation on layer 0), and
//! `moe_ffn_forward_cpu` (ncmoe IQ3_S CPU GEMV).

use candle_core::{Device, Result, Tensor};

// Re-export types from the FFI crate for convenience.
pub use ggml_ffi::GgmlType;
pub use ggml_ffi::GgmlCpuContext;

/// Check if the ggml lm_head validation path is enabled.
pub fn is_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_LM_HEAD").is_ok());
    *ENABLED
}

/// Check if the Q5_K CPU validation path is enabled (layer 0 attn_qkv only).
pub fn is_q5k_cpu_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_GGML_Q5K_CPU").is_ok());
    *ENABLED
}

/// Compute logits via ggml-compatible Q8_0 GEMV on CPU.
///
/// # Arguments
/// * `h_last` - Hidden state after output_norm, shape `[1, hidden_size]`, on GPU
/// * `w_raw`  - Raw Q8_0 bytes for output.weight, on CPU, flat `&[u8]`
/// * `vocab_size` - Number of output rows (248320 for Qwen3.5)
/// * `hidden_size` - Number of input columns (2048 for Qwen3.5)
/// * `device` - Target device for the output tensor
///
/// # Returns
/// Logits tensor `[1, vocab_size]` on the target device.
pub fn ggml_lm_head_forward(
    h_last: &Tensor,
    w_raw: &[u8],
    vocab_size: usize,
    hidden_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let t0 = std::time::Instant::now();

    // 1. Transfer h_last from GPU to CPU
    let h_cpu: Vec<f32> = h_last.flatten_all()?.to_vec1()?;
    let t_download = t0.elapsed();

    if h_cpu.len() != hidden_size {
        candle_core::bail!(
            "ggml_lm_head_forward: h_last has {} elements, expected {hidden_size}",
            h_cpu.len()
        );
    }

    // 2. CPU Q8_0 GEMV
    let t1 = std::time::Instant::now();
    let ctx = GgmlCpuContext::new();
    let mut logits_cpu = vec![0.0f32; vocab_size];
    ctx.mul_mat_vec_q8_0(w_raw, vocab_size, hidden_size, &h_cpu, &mut logits_cpu);
    let t_gemv = t1.elapsed();

    // 3. Upload result back to target device
    let t2 = std::time::Instant::now();
    let logits = Tensor::from_vec(logits_cpu, (1, vocab_size), device)?;
    let t_upload = t2.elapsed();

    let total = t0.elapsed();
    eprintln!(
        "[GGML_LM_HEAD] Q8_0 CPU GEMV: download={:.2}ms gemv={:.2}ms upload={:.2}ms total={:.2}ms",
        t_download.as_secs_f64() * 1000.0,
        t_gemv.as_secs_f64() * 1000.0,
        t_upload.as_secs_f64() * 1000.0,
        total.as_secs_f64() * 1000.0,
    );

    Ok(logits)
}

/// Print comparison of top-5 logits between two tensors.
///
/// Useful for comparing ggml path vs Candle QMatMul path.
pub fn compare_top5(label_a: &str, logits_a: &Tensor, label_b: &str, logits_b: &Tensor) -> Result<()> {
    let a: Vec<f32> = logits_a.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = logits_b.flatten_all()?.to_vec1()?;

    let mut idx_a: Vec<(usize, f32)> = a.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut idx_b: Vec<(usize, f32)> = b.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_a.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    idx_b.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());

    eprintln!("[COMPARE] {label_a} vs {label_b} top-5:");
    for i in 0..5.min(idx_a.len()).min(idx_b.len()) {
        let (ta, va) = idx_a[i];
        let (tb, vb) = idx_b[i];
        let match_marker = if ta == tb { "OK" } else { "MISMATCH" };
        eprintln!(
            "  #{}: {} tok={} logit={:.4}  |  {} tok={} logit={:.4}  [{}]",
            i + 1, label_a, ta, va, label_b, tb, vb, match_marker
        );
    }

    // Compute max absolute difference across all logits
    let max_diff = a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let mean_diff = a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>() / a.len() as f32;

    eprintln!("[COMPARE] max_abs_diff={:.6} mean_abs_diff={:.6}", max_diff, mean_diff);

    Ok(())
}

/// Q5_K CPU GEMV for validation. Same pattern as ggml_lm_head_forward.
/// Downloads hidden state to CPU, downloads Q5_K weight bytes from GPU,
/// runs Q5_K dot product in pure Rust, uploads result to GPU.
/// SLOW but CORRECT -- validates numerical parity with ggml.
///
/// # Arguments
/// * `input`        - Hidden state, shape `[1, ncols]`, F32 on GPU
/// * `weight_raw`   - Q5_K raw bytes as flat U8 tensor on GPU
/// * `nrows`        - Number of output features
/// * `ncols`        - Number of input features (must be multiple of 256)
/// * `device`       - Target device for the output tensor
///
/// # Returns
/// Output tensor `[1, nrows]` on the target device.
pub fn ggml_q5k_forward_cpu(
    input: &Tensor,
    weight_raw: &Tensor,
    nrows: usize,
    ncols: usize,
    device: &Device,
) -> Result<Tensor> {
    let t0 = std::time::Instant::now();

    // 1. Download input F32 from GPU to CPU
    let input_cpu: Vec<f32> = input.flatten_all()?.to_vec1()?;
    let t_download_input = t0.elapsed();

    if input_cpu.len() < ncols {
        candle_core::bail!(
            "ggml_q5k_forward_cpu: input has {} elements, expected >= {ncols}",
            input_cpu.len()
        );
    }

    // 2. Download Q5_K weight bytes from GPU to CPU
    let t1 = std::time::Instant::now();
    let weight_bytes: Vec<u8> = weight_raw.flatten_all()?.to_vec1()?;
    let t_download_weight = t1.elapsed();

    // 3. Run Q5_K matmul on CPU (pure Rust, ggml-compatible)
    let t2 = std::time::Instant::now();
    let output_cpu = ggml_ffi::q5k_matmul_cpu(&weight_bytes, &input_cpu, nrows, ncols);
    let t_gemv = t2.elapsed();

    // 4. Upload result back to GPU
    let t3 = std::time::Instant::now();
    let result = Tensor::from_vec(output_cpu, (1, nrows), device)?;
    let t_upload = t3.elapsed();

    let total = t0.elapsed();
    eprintln!(
        "[GGML_Q5K_CPU] Q5_K GEMV [{nrows}x{ncols}]: \
         dl_input={:.2}ms dl_weight={:.2}ms gemv={:.2}ms upload={:.2}ms total={:.2}ms",
        t_download_input.as_secs_f64() * 1000.0,
        t_download_weight.as_secs_f64() * 1000.0,
        t_gemv.as_secs_f64() * 1000.0,
        t_upload.as_secs_f64() * 1000.0,
        total.as_secs_f64() * 1000.0,
    );

    Ok(result)
}

/// Compare two tensors element-wise, printing top-5 by magnitude and diff stats.
///
/// Downloads both tensors to CPU and prints:
///   - Top-5 indices by absolute value for each tensor
///   - Whether the top-5 indices match
///   - Max/mean absolute difference
///   - Relative error (max_diff / max_abs_value)
///
/// This is non-fatal: comparison failures are logged but do not panic.
pub fn compare_tensors(label: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let va: Vec<f32> = a.flatten_all()?.to_vec1()?;
    let vb: Vec<f32> = b.flatten_all()?.to_vec1()?;

    if va.len() != vb.len() {
        eprintln!(
            "[COMPARE_Q5K] {label}: SIZE MISMATCH a={} b={}",
            va.len(), vb.len()
        );
        return Ok(());
    }

    // Top-5 by absolute value
    let mut idx_a: Vec<(usize, f32)> = va.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    let mut idx_b: Vec<(usize, f32)> = vb.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    idx_a.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());
    idx_b.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());

    let n = 5.min(va.len());
    eprintln!("[COMPARE_Q5K] {label}: ggml_cpu vs candle (top-{n} by |value|):");
    let mut top5_match = 0;
    for i in 0..n {
        let (ia, va_i) = idx_a[i];
        let (ib, vb_i) = idx_b[i];
        let marker = if ia == ib { top5_match += 1; "OK" } else { "MISMATCH" };
        eprintln!(
            "  #{}: ggml idx={:5} val={:+.6}  |  candle idx={:5} val={:+.6}  [{}]",
            i + 1, ia, va_i, ib, vb_i, marker
        );
    }

    // Diff statistics
    let max_diff = va.iter().zip(vb.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let mean_diff = va.iter().zip(vb.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>() / va.len() as f32;
    let max_abs = va.iter().map(|v| v.abs()).fold(0.0f32, f32::max)
        .max(vb.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
    let rel_err = if max_abs > 0.0 { max_diff / max_abs } else { 0.0 };

    eprintln!(
        "[COMPARE_Q5K] {label}: top5_match={top5_match}/{n} \
         max_abs_diff={max_diff:.6} mean_abs_diff={mean_diff:.6} \
         rel_err={rel_err:.6} max_abs_val={max_abs:.4}"
    );

    Ok(())
}

// ============================================================================
// ncmoe CPU path — IQ3_S GEMV via ggml AVX2 FFI
// ============================================================================

/// Check if the ncmoe CPU GEMV path is enabled (`CHIMERE_NCMOE_CPU=1`).
pub fn is_ncmoe_cpu_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| {
        let enabled = std::env::var("CHIMERE_NCMOE_CPU").is_ok();
        if enabled {
            let avail = ggml_ffi::iq3s_ffi_available();
            if avail {
                eprintln!("[NCMOE_CPU] IQ3_S CPU GEMV enabled (ggml AVX2 FFI)");
            } else {
                eprintln!("[NCMOE_CPU] WARNING: CHIMERE_NCMOE_CPU=1 but ggml_iq3s feature \
                           not compiled. Falling back to GPU batch copy.");
            }
            avail
        } else {
            false
        }
    });
    *ENABLED
}

/// Number of OpenMP threads to use for the CPU GEMV path.
/// Reads from `CHIMERE_NCMOE_THREADS` (default: 4).
pub fn ncmoe_cpu_threads() -> usize {
    use once_cell::sync::Lazy;
    static THREADS: Lazy<usize> = Lazy::new(|| {
        std::env::var("CHIMERE_NCMOE_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
    });
    *THREADS
}

/// Run a single SwiGLU expert entirely on CPU using ggml IQ3_S AVX2 GEMV.
///
/// Computes: `down_proj( silu(gate_proj(hidden)) * up_proj(hidden) )`
///
/// The expert weight bytes (IQ3_S) are already on CPU (ncmoe offloaded).
/// The hidden state is downloaded from GPU once (8 KB for hidden_size=2048).
///
/// # Arguments
/// * `hidden_cpu`   - f32 hidden state on CPU, length hidden_size
/// * `gate_w`       - IQ3_S gate weight bytes for this expert
/// * `up_w`         - IQ3_S up weight bytes for this expert
/// * `down_w`       - IQ3_S down weight bytes for this expert
/// * `expert_ffn`   - Expert FFN hidden dim (512 for Qwen3.5-35B-A3B)
/// * `hidden_size`  - Model hidden dim (2048 for Qwen3.5-35B-A3B)
///
/// # Returns
/// f32 output vector on CPU, length hidden_size.
pub fn ncmoe_cpu_expert_forward(
    hidden_cpu: &[f32],
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    expert_ffn: usize,
    hidden_size: usize,
) -> Vec<f32> {
    ggml_ffi::iq3s_swiglu_expert_cpu(
        gate_w, up_w, down_w,
        hidden_cpu,
        expert_ffn, hidden_size,
    )
}

/// Full ncmoe CPU forward: download hidden state from GPU, run all selected
/// experts on CPU using ggml IQ3_S GEMV, upload combined result back to GPU.
///
/// This is the "invert ncmoe" approach. Instead of:
///   CPU expert weights -> GPU staging (10 MB htod) -> GPU GEMV
/// We do:
///   GPU hidden state -> CPU (8 KB dtoh) -> CPU IQ3_S GEMV (AVX2) -> GPU (8 KB htod)
///
/// # Arguments
/// * `hidden_gpu`   - Hidden state tensor [1, hidden_size] on GPU
/// * `gate_all`     - All experts' gate weights, IQ3_S, on CPU [num_experts * expert_bytes_gate]
/// * `up_all`       - All experts' up weights, IQ3_S, on CPU
/// * `down_all`     - All experts' down weights, IQ3_S, on CPU
/// * `expert_ids`   - Indices of top-K selected experts
/// * `expert_weights` - Routing weights for selected experts (sum to 1.0)
/// * `expert_ffn`   - Expert FFN hidden dim (512)
/// * `hidden_size`  - Model hidden dim (2048)
/// * `ncols_gate`   - Number of input columns for gate/up (= hidden_size)
/// * `ncols_down`   - Number of input columns for down (= expert_ffn)
/// * `device`       - GPU device for uploading the result
///
/// # Returns
/// Combined expert output tensor [1, hidden_size] on GPU.
pub fn ncmoe_cpu_experts_forward(
    hidden_gpu: &Tensor,
    gate_all: &[u8],
    up_all: &[u8],
    down_all: &[u8],
    expert_ids: &[usize],
    expert_weights: &[f32],
    expert_ffn: usize,
    hidden_size: usize,
    ncols_gate: usize,
    ncols_down: usize,
    device: &Device,
) -> Result<Tensor> {
    let t0 = std::time::Instant::now();

    // 1. Download hidden state from GPU to CPU (8 KB for hidden_size=2048)
    let hidden_cpu: Vec<f32> = hidden_gpu.flatten_all()?.to_vec1()?;
    let t_download = t0.elapsed();

    // 2. Compute byte sizes per expert
    let expert_bytes_gate = ggml_ffi::iq3s_row_bytes(ncols_gate) * expert_ffn;
    let expert_bytes_up = ggml_ffi::iq3s_row_bytes(ncols_gate) * expert_ffn;
    let expert_bytes_down = ggml_ffi::iq3s_row_bytes(ncols_down) * hidden_size;

    // 3. Run all selected experts on CPU and accumulate weighted output
    let t1 = std::time::Instant::now();
    let mut combined = vec![0.0f32; hidden_size];

    for (&eid, &weight) in expert_ids.iter().zip(expert_weights.iter()) {
        // Slice expert weights from the full expert arrays
        let g_start = eid * expert_bytes_gate;
        let g_end = g_start + expert_bytes_gate;
        let u_start = eid * expert_bytes_up;
        let u_end = u_start + expert_bytes_up;
        let d_start = eid * expert_bytes_down;
        let d_end = d_start + expert_bytes_down;

        let expert_out = ncmoe_cpu_expert_forward(
            &hidden_cpu,
            &gate_all[g_start..g_end],
            &up_all[u_start..u_end],
            &down_all[d_start..d_end],
            expert_ffn,
            hidden_size,
        );

        // Weighted accumulation
        for (c, &e) in combined.iter_mut().zip(expert_out.iter()) {
            *c += weight * e;
        }
    }
    let t_gemv = t1.elapsed();

    // 4. Upload combined result back to GPU (8 KB)
    let t2 = std::time::Instant::now();
    let result = Tensor::from_vec(combined, (1, hidden_size), device)?;
    let t_upload = t2.elapsed();

    let total = t0.elapsed();

    // Log timing (once per 100 calls to avoid spam)
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALL_COUNT: AtomicU64 = AtomicU64::new(0);
    static TOTAL_US: AtomicU64 = AtomicU64::new(0);
    let count = CALL_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    TOTAL_US.fetch_add(total.as_micros() as u64, Ordering::Relaxed);
    if count <= 5 || count % 100 == 0 {
        let avg_us = TOTAL_US.load(Ordering::Relaxed) / count;
        eprintln!(
            "[NCMOE_CPU] IQ3_S GEMV #{}: dl={:.2}ms gemv={:.2}ms ul={:.2}ms total={:.2}ms \
             (avg={:.2}ms, {} experts)",
            count,
            t_download.as_secs_f64() * 1000.0,
            t_gemv.as_secs_f64() * 1000.0,
            t_upload.as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
            avg_us as f64 / 1000.0,
            expert_ids.len(),
        );
    }

    Ok(result)
}
