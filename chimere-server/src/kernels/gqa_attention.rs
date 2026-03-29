//! Fused GQA attention kernel: scores + softmax + output WITHOUT expanding KV cache.
//!
//! Eliminates the GQA expand overhead (unsqueeze + expand + contiguous + reshape
//! for both K and V = 10 Candle ops per layer) by computing attention directly
//! from the unexpanded KV cache using strided reads.
//!
//! For each Q head `h`, the kernel reads from `K[h / group_size]` and
//! `V[h / group_size]` — multiple Q heads share the same KV head without
//! materializing a 4× larger buffer.
//!
//! ## Kernel design
//!
//! - **Grid**: `(num_heads, 1, 1)` — one block per Q head.
//! - **Block**: `(256, 1, 1)` — 8 warps cooperate on one head.
//! - **Shared memory**: `seq_len` floats for attention scores, plus 256 floats
//!   for reductions.
//!
//! ### Algorithm (per block / per Q head):
//!
//! 1. **Score computation**: Each thread computes partial dot products
//!    `Q[h, :] · K[kv_h, pos, :]` for a subset of positions, cooperating via
//!    shared memory reduction over the `head_dim` dimension.
//! 2. **Online softmax**: Two-pass (find max, then exp + sum) over scores in
//!    shared memory.
//! 3. **Weighted sum**: Each thread accumulates `weights[pos] * V[kv_h, pos, d]`
//!    for its assigned dimensions, then writes to output.
//!
//! ## Limitations
//!
//! - Single-token decode only (Q has 1 position per head).
//! - No causal mask needed (decode attends to all cached positions).
//! - head_dim must be a multiple of 256 (true for Qwen3.5: head_dim=256).

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CUDA source
// ---------------------------------------------------------------------------

const GQA_ATTENTION_KERNEL_SRC: &str = r#"
// Fused GQA attention: score + softmax + output in one kernel.
//
// Q:      [num_heads, head_dim]                     (single token, already RoPE'd)
// K:      [num_kv_heads, seq_len, head_dim]         (full KV cache)
// V:      [num_kv_heads, seq_len, head_dim]         (full KV cache)
// output: [num_heads, head_dim]                     (attention output)
//
// Grid:  (num_heads, 1, 1) — one block per Q head
// Block: (256, 1, 1) — threads cooperate on dot products and reductions
//
// Shared memory layout (dynamic):
//   float scores[seq_len]    — attention scores for this head
//   float reduce[256]        — reduction workspace

extern "C" __global__ void gqa_attention_kernel(
    const float* __restrict__ Q,       // [num_heads * head_dim]
    const float* __restrict__ K,       // [num_kv_heads * kv_stride * head_dim]
    const float* __restrict__ V,       // [num_kv_heads * kv_stride * head_dim]
    float*       __restrict__ output,  // [num_heads * head_dim]
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    int kv_stride                      // capacity per head (= seq_len if compacted)
) {
    const int h = blockIdx.x;                        // Q head index
    const int tid = threadIdx.x;                     // 0..255
    const int kv_h = h * num_kv_heads / num_heads;   // KV head for this Q head (integer division)

    extern __shared__ char smem_raw[];
    // scores occupies the first seq_len floats
    float* scores = (float*)smem_raw;
    // reduce occupies the next 256 floats (after scores, aligned)
    float* reduce = (float*)(smem_raw + sizeof(float) * seq_len);

    const float* q_ptr = Q + h * head_dim;
    const float* k_base = K + (long long)kv_h * kv_stride * head_dim;
    const float* v_base = V + (long long)kv_h * kv_stride * head_dim;

    // -----------------------------------------------------------------------
    // Step 1: Compute attention scores = Q[h] . K[kv_h, pos, :] * scale
    //
    // Strategy: each thread handles a subset of positions. For each position,
    // we compute the full dot product by having ALL 256 threads cooperate:
    // - Each thread handles head_dim/256 elements of the dot product
    // - Warp-level reduction + shared memory reduction to get the final score
    //
    // But with seq_len potentially up to 64K, we need the faster approach:
    // each thread independently computes scores for its assigned positions.
    // Since head_dim=256 and blockDim=256, each thread computes one element
    // of the dot product per position, then we reduce across threads.
    //
    // Actually, the most efficient approach for head_dim=256, blockDim=256:
    // Loop over positions, each thread contributes q[tid] * k[pos, tid],
    // then reduce the 256 partial products to get the score for that position.
    //
    // But that's O(seq_len) reductions with __syncthreads each. Too slow for
    // large seq_len.
    //
    // Better approach: each thread loops over ALL positions, computing its
    // partial dot product contribution (q[tid] * k[pos, tid]) and storing
    // partial sums. Then reduce per-position across threads. But this requires
    // O(seq_len) shared memory per thread — not feasible.
    //
    // Optimal for head_dim=256, block=256: process positions in tiles.
    // Each tile = 1 position. 256 threads compute 256 multiply-adds,
    // then warp-reduce + cross-warp reduce. This gives 1 score per ~40 cycles.
    // For seq_len=100 → ~4000 cycles = ~2 µs at 2 GHz. Acceptable.
    // -----------------------------------------------------------------------

    for (int pos = 0; pos < seq_len; pos++) {
        const float* k_ptr = k_base + pos * head_dim;

        // Each thread computes q[tid] * k[pos, tid] (one element of the dot product)
        float partial = 0.0f;
        if (tid < head_dim) {
            partial = q_ptr[tid] * k_ptr[tid];
        }

        // Warp-level reduction (within each 32-thread warp)
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
        }

        // Cross-warp reduction via shared memory
        // Lane 0 of each warp writes its warp sum
        if ((tid & 31) == 0) {
            reduce[tid >> 5] = partial;
        }
        __syncthreads();

        // Thread 0 sums all warp contributions
        if (tid == 0) {
            float sum = 0.0f;
            int n_warps = (blockDim.x + 31) >> 5;  // 256/32 = 8
            for (int w = 0; w < n_warps; w++) {
                sum += reduce[w];
            }
            scores[pos] = sum * scale;
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Step 2: Softmax over scores[0..seq_len]
    //
    // Two-pass: find max, then compute exp(score - max) and sum.
    // Threads cooperate by processing chunks of positions.
    // -----------------------------------------------------------------------

    // 2a: Find max score
    float local_max = -1.0e30f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        float s = scores[pos];
        if (s > local_max) local_max = s;
    }

    // Warp reduction for max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }
    if ((tid & 31) == 0) {
        reduce[tid >> 5] = local_max;
    }
    __syncthreads();

    if (tid == 0) {
        float m = reduce[0];
        int n_warps = (blockDim.x + 31) >> 5;
        for (int w = 1; w < n_warps; w++) {
            if (reduce[w] > m) m = reduce[w];
        }
        reduce[0] = m;
    }
    __syncthreads();
    float max_score = reduce[0];
    __syncthreads();

    // 2b: Compute exp(score - max) in-place, and partial sum
    float local_sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        float e = expf(scores[pos] - max_score);
        scores[pos] = e;
        local_sum += e;
    }

    // Warp reduction for sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, offset);
    }
    if ((tid & 31) == 0) {
        reduce[tid >> 5] = local_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float s = 0.0f;
        int n_warps = (blockDim.x + 31) >> 5;
        for (int w = 0; w < n_warps; w++) {
            s += reduce[w];
        }
        reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / fmaxf(reduce[0], 1e-8f);
    __syncthreads();

    // 2c: Normalize in-place: scores[pos] *= inv_sum
    for (int pos = tid; pos < seq_len; pos += blockDim.x) {
        scores[pos] *= inv_sum;
    }
    __syncthreads();

    // -----------------------------------------------------------------------
    // Step 3: Weighted sum over V
    //
    // output[h, d] = sum_pos( scores[pos] * V[kv_h, pos, d] )
    //
    // Each thread handles one or more dimensions (d).
    // For head_dim=256 and blockDim=256: thread tid handles dimension tid.
    // -----------------------------------------------------------------------

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            acc += scores[pos] * v_base[pos * head_dim + tid];
        }
        output[h * head_dim + tid] = acc;
    }
}
"#;

// ---------------------------------------------------------------------------
// PTX cache + compilation
// ---------------------------------------------------------------------------

const MODULE_NAME: &str = "chimere_gqa_attention_v2";
const KERNEL_FUNC: &str = "gqa_attention_kernel";

static PTX_CACHE: OnceLock<String> = OnceLock::new();

fn load_func(
    dev: &CudaDevice,
) -> Result<(
    candle_core::cuda_backend::cudarc::driver::CudaFunction,
    std::sync::Arc<candle_core::cuda_backend::cudarc::driver::CudaStream>,
)> {
    super::nvrtc_compile::get_or_load_func(
        dev,
        KERNEL_FUNC,
        MODULE_NAME,
        GQA_ATTENTION_KERNEL_SRC,
        &PTX_CACHE,
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused GQA attention: computes attention scores + softmax + weighted output
/// without materializing the expanded KV cache.
///
/// Replaces the GQA expand chain (unsqueeze + expand + contiguous + reshape for
/// both K and V = 10 Candle ops per layer) plus the score matmul + softmax +
/// output matmul (4 more ops) with a single kernel launch.
///
/// # Layout
///
/// - `q`: `[num_heads * head_dim]` — current token query, already RoPE'd.
/// - `k_cache`: `[num_kv_heads * seq_len * head_dim]` — full KV cache keys.
/// - `v_cache`: `[num_kv_heads * seq_len * head_dim]` — full KV cache values.
/// - `output`: `[num_heads * head_dim]` — pre-allocated output buffer.
///
/// # Shared memory
///
/// Requires `(seq_len + 256) * 4` bytes of dynamic shared memory.
/// For seq_len=65536 this is ~262 KB — within the 228 KB default limit of
/// sm_120 when `cudaFuncSetAttribute` is used. For very large seq_len the
/// caller should verify the device's `sharedMemPerBlockOptin` limit.
///
/// # Panics
///
/// Panics if `num_heads % num_kv_heads != 0` or `num_heads == 0`.
/// Fused GQA attention with strided KV cache support.
///
/// `kv_stride` is the number of positions allocated per KV head in the cache.
/// For compacted caches (e.g., from Candle Tensor), `kv_stride == seq_len`.
/// For pre-allocated ring buffers, `kv_stride == kv_cap` (total capacity).
///
/// The kernel only processes `seq_len` positions but uses `kv_stride` for
/// the base pointer offset between KV heads.
pub fn raw_gqa_attention(
    q: &CudaView<'_, f32>,
    k_cache: &CudaView<'_, f32>,
    v_cache: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<()> {
    raw_gqa_attention_strided(
        q, k_cache, v_cache, output,
        num_heads, num_kv_heads, seq_len, head_dim,
        scale, seq_len, dev,
    )
}

/// Fused GQA attention with explicit KV stride.
///
/// Same as `raw_gqa_attention` but allows specifying `kv_stride` independently
/// of `seq_len`. Use when the KV cache has pre-allocated capacity > seq_len
/// (e.g., `KvCacheRaw` with `kv_cap > pos`).
pub fn raw_gqa_attention_strided(
    q: &CudaView<'_, f32>,
    k_cache: &CudaView<'_, f32>,
    v_cache: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    kv_stride: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "num_heads ({num_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
    );
    assert!(
        kv_stride >= seq_len,
        "kv_stride ({kv_stride}) must be >= seq_len ({seq_len})"
    );
    if seq_len == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev)?;

    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let seq_len_i32 = seq_len as i32;
    let head_dim_i32 = head_dim as i32;
    let kv_stride_i32 = kv_stride as i32;

    // Shared memory: seq_len floats (scores) + 256 floats (reduction workspace)
    let smem_bytes = ((seq_len + 256) * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(q);
    builder.arg(k_cache);
    builder.arg(v_cache);
    builder.arg(output);
    builder.arg(&num_heads_i32);
    builder.arg(&num_kv_heads_i32);
    builder.arg(&seq_len_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&scale);
    builder.arg(&kv_stride_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("gqa_attention_kernel launch: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor-level wrapper (converts Candle Tensors to CudaViews, calls raw_*)
// ---------------------------------------------------------------------------

/// Fused GQA attention from Candle Tensors.
///
/// Takes Q, K_cache, V_cache as Candle Tensors and returns the attention output
/// as a new Tensor. This is the entry point called from `forward_attn_layer_moe`.
///
/// # Shapes
///
/// - `q`: `[1, num_heads, head_dim]` — current token query (post-RoPE, post-norm).
/// - `k_cache`: `[1, num_kv_heads, seq_len, head_dim]` — KV cache keys.
/// - `v_cache`: `[1, num_kv_heads, seq_len, head_dim]` — KV cache values.
///
/// Returns: `[1, num_heads, head_dim]` — attention output (before gate).
pub fn fused_gqa_attention_tensor(
    q: &candle_core::Tensor,
    k_cache: &candle_core::Tensor,
    v_cache: &candle_core::Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Result<candle_core::Tensor> {
    use candle_core::Storage;

    let device = q.device();
    let dev = match device {
        candle_core::Device::Cuda(d) => d,
        _ => candle_core::bail!("gqa_attention: expected CUDA device"),
    };

    let seq_len = k_cache.dim(2)?;

    // Make inputs contiguous and extract CudaViews
    let q_c = q.contiguous()?;
    let k_c = k_cache.contiguous()?;
    let v_c = v_cache.contiguous()?;

    let (q_stor, q_lay) = q_c.storage_and_layout();
    let q_cuda = match &*q_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("not CUDA"),
    };
    let q_view = q_cuda
        .as_cuda_slice::<f32>()?
        .slice(q_lay.start_offset()..);

    let (k_stor, k_lay) = k_c.storage_and_layout();
    let k_cuda = match &*k_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("not CUDA"),
    };
    let k_view = k_cuda
        .as_cuda_slice::<f32>()?
        .slice(k_lay.start_offset()..);

    let (v_stor, v_lay) = v_c.storage_and_layout();
    let v_cuda = match &*v_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("not CUDA"),
    };
    let v_view = v_cuda
        .as_cuda_slice::<f32>()?
        .slice(v_lay.start_offset()..);

    // Allocate output: [num_heads * head_dim]
    let out_elems = num_heads * head_dim;
    let cuda_stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = cuda_stream
        .alloc_zeros(out_elems)
        .map_err(|e| candle_core::Error::Msg(format!("alloc gqa_output: {e}")))?;

    raw_gqa_attention(
        &q_view, &k_view, &v_view, &mut output,
        num_heads, num_kv_heads, seq_len, head_dim,
        scale, dev,
    )?;

    // Drop storage guards before wrapping output
    drop(q_stor);
    drop(k_stor);
    drop(v_stor);

    // Wrap result as Tensor [1, num_heads, head_dim]
    let out_shape = candle_core::Shape::from_dims(&[1, num_heads, head_dim]);
    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(candle_core::Tensor::from_storage(
        Storage::Cuda(storage),
        out_shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Test that fused GQA attention matches the naive expand + matmul path.
    ///
    /// Reference computation (CPU):
    ///   For each Q head h:
    ///     kv_h = h / group_size
    ///     scores[pos] = sum_d(Q[h,d] * K[kv_h, pos, d]) * scale
    ///     weights = softmax(scores)
    ///     output[h, d] = sum_pos(weights[pos] * V[kv_h, pos, d])
    #[test]
    fn test_gqa_attention_matches_reference() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[GQA_ATTN] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let num_heads: usize = 16;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 256;
        let seq_len: usize = 37; // odd number to catch off-by-one
        let group_size = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Generate deterministic test data
        let q_h: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.0137 - 1.0).sin() * 0.3)
            .collect();
        let k_h: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| ((i as f32) * 0.0071 + 0.5).cos() * 0.2)
            .collect();
        let v_h: Vec<f32> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| ((i as f32) * 0.0043 - 0.3).sin() * 0.4)
            .collect();

        // --- Reference: compute on CPU ---
        let mut ref_output = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;

            // Compute scores
            let mut scores = vec![0.0f32; seq_len];
            for pos in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_h[h * head_dim + d]
                        * k_h[kv_h * seq_len * head_dim + pos * head_dim + d];
                }
                scores[pos] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_e += *s;
            }
            for s in &mut scores {
                *s /= sum_e;
            }

            // Weighted sum over V
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for pos in 0..seq_len {
                    acc += scores[pos]
                        * v_h[kv_h * seq_len * head_dim + pos * head_dim + d];
                }
                ref_output[h * head_dim + d] = acc;
            }
        }

        // --- GPU: run fused kernel ---
        let stream = cuda_dev.cuda_stream();

        let mut q_gpu: CudaSlice<f32> = stream
            .alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc q: {e}")))?;
        cuda_dev
            .memcpy_htod(&q_h, &mut q_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload q: {e}")))?;

        let mut k_gpu: CudaSlice<f32> = stream
            .alloc_zeros(num_kv_heads * seq_len * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc k: {e}")))?;
        cuda_dev
            .memcpy_htod(&k_h, &mut k_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload k: {e}")))?;

        let mut v_gpu: CudaSlice<f32> = stream
            .alloc_zeros(num_kv_heads * seq_len * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc v: {e}")))?;
        cuda_dev
            .memcpy_htod(&v_h, &mut v_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload v: {e}")))?;

        let mut output_gpu: CudaSlice<f32> = stream
            .alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;

        let q_view = q_gpu.slice(..);
        let k_view = k_gpu.slice(..);
        let v_view = v_gpu.slice(..);

        raw_gqa_attention(
            &q_view,
            &k_view,
            &v_view,
            &mut output_gpu,
            num_heads,
            num_kv_heads,
            seq_len,
            head_dim,
            scale,
            cuda_dev,
        )?;

        // Read back results
        let gpu_output: Vec<f32> = output_gpu
            .stream()
            .clone()
            .clone_dtoh(&output_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback output: {e}")))?;

        // --- Compare ---
        let mut max_err = 0.0f32;
        let mut max_err_idx = 0usize;
        for i in 0..num_heads * head_dim {
            let err = (gpu_output[i] - ref_output[i]).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
        }
        eprintln!(
            "[GQA_ATTN] num_heads={num_heads}, num_kv_heads={num_kv_heads}, \
             seq_len={seq_len}, head_dim={head_dim}, max_err={max_err:.2e} (idx={max_err_idx})"
        );
        assert!(
            max_err < 1e-3,
            "GQA attention mismatch: max_err={max_err:.6e} at idx={max_err_idx} (tolerance 1e-3)"
        );

        eprintln!("[GQA_ATTN] PASS: fused kernel matches reference");
        Ok(())
    }

    /// Test with seq_len=1 (first token, edge case).
    #[test]
    fn test_gqa_attention_seq_len_1() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[GQA_ATTN] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let num_heads: usize = 16;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 256;
        let seq_len: usize = 1;
        let group_size = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_h: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k_h: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.007).cos())
            .collect();
        let v_h: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();

        // For seq_len=1, softmax is trivially 1.0, so output = V[kv_h, 0, :]
        let mut ref_output = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;
            for d in 0..head_dim {
                ref_output[h * head_dim + d] = v_h[kv_h * head_dim + d];
            }
        }

        let stream = cuda_dev.cuda_stream();
        let mut q_gpu: CudaSlice<f32> = stream.alloc_zeros(q_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        cuda_dev.memcpy_htod(&q_h, &mut q_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload: {e}")))?;
        let mut k_gpu: CudaSlice<f32> = stream.alloc_zeros(k_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        cuda_dev.memcpy_htod(&k_h, &mut k_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload: {e}")))?;
        let mut v_gpu: CudaSlice<f32> = stream.alloc_zeros(v_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        cuda_dev.memcpy_htod(&v_h, &mut v_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload: {e}")))?;
        let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;

        raw_gqa_attention(
            &q_gpu.slice(..), &k_gpu.slice(..), &v_gpu.slice(..),
            &mut output_gpu,
            num_heads, num_kv_heads, seq_len, head_dim, scale, cuda_dev,
        )?;

        let gpu_output: Vec<f32> = output_gpu.stream().clone().clone_dtoh(&output_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback: {e}")))?;

        let max_err = gpu_output.iter().zip(ref_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GQA_ATTN] seq_len=1 max_err={max_err:.2e}");
        assert!(max_err < 1e-5, "seq_len=1 mismatch: max_err={max_err:.6e}");

        eprintln!("[GQA_ATTN] PASS: seq_len=1 edge case");
        Ok(())
    }

    /// Test with the Tensor-level wrapper.
    #[test]
    fn test_gqa_attention_tensor_wrapper() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        if !matches!(&device, Device::Cuda(_)) {
            eprintln!("[GQA_ATTN] No CUDA device, skipping test.");
            return Ok(());
        }

        let num_heads: usize = 16;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 256;
        let seq_len: usize = 20;
        let group_size = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Create Candle Tensors
        let q = candle_core::Tensor::randn(
            0.0f32, 1.0,
            &[1, num_heads, head_dim],
            &device,
        )?;
        let k_cache = candle_core::Tensor::randn(
            0.0f32, 1.0,
            &[1, num_kv_heads, seq_len, head_dim],
            &device,
        )?;
        let v_cache = candle_core::Tensor::randn(
            0.0f32, 1.0,
            &[1, num_kv_heads, seq_len, head_dim],
            &device,
        )?;

        // Run fused kernel
        let fused_out = fused_gqa_attention_tensor(
            &q, &k_cache, &v_cache,
            num_heads, num_kv_heads, head_dim, scale,
        )?;

        // Run reference: expand + matmul + softmax + matmul
        let q_attn = q.unsqueeze(2)?;  // [1, num_heads, 1, head_dim]
        let k_exp = k_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, seq_len, head_dim))?;
        let v_exp = v_cache
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, seq_len, head_dim))?;
        let scores = q_attn.matmul(&k_exp.transpose(2, 3)?)?;
        let scores = (scores * (scale as f64))?;
        let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let ref_out = weights.matmul(&v_exp)?.squeeze(2)?;  // [1, num_heads, head_dim]

        // Compare
        let fused_flat = fused_out.flatten_all()?.to_vec1::<f32>()?;
        let ref_flat = ref_out.flatten_all()?.to_vec1::<f32>()?;

        let max_err = fused_flat.iter().zip(ref_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[GQA_ATTN] Tensor wrapper max_err={max_err:.2e}");
        assert!(
            max_err < 1e-3,
            "Tensor wrapper mismatch: max_err={max_err:.6e} (tolerance 1e-3)"
        );

        eprintln!("[GQA_ATTN] PASS: Tensor wrapper matches Candle reference");
        Ok(())
    }
}
