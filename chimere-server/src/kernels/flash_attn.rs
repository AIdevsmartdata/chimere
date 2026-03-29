//! Flash Attention decode kernel for chimere-deltanet.
//!
//! Specialized for single-token decode (batch=1, q_len=1) with:
//! - F32 Q (from RoPE'd query, single token)
//! - F16 KV cache (production format, stored as `half::f16`)
//! - F32 output
//! - GQA support (multiple Q heads share one KV head)
//! - Online softmax over KV tiles (O(1) extra memory vs O(seq_len) for naive)
//!
//! ## Architecture (Qwen3.5-35B-A3B)
//!
//! - 10 attention layers (every 4th: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39)
//! - `num_heads=16`, `num_kv_heads=2`, `head_dim=256`
//! - GQA ratio 8:1
//!
//! ## Kernel Design
//!
//! For single-token decode, Q has length 1, so the attention computation
//! simplifies to a fused dot-product + online-softmax + weighted-sum over
//! the KV cache sequence dimension. No Q tiling needed.
//!
//! The kernel tiles over the KV sequence in blocks of 32 positions:
//! 1. Load K tile into shared memory (F16, coalesced)
//! 2. Compute scores: `s[j] = dot(Q[h,:], K[kv_h,j,:]) * scale` via
//!    warp-level shuffle reduction
//! 3. Online softmax update: maintain running max `m`, sum `l`, and
//!    output accumulator `o` across tiles
//! 4. Read V values from global memory and accumulate weighted sum
//! 5. After all tiles: normalize `output = o / l`
//!
//! ## Advantages over existing `gqa_attention` kernel
//!
//! 1. **Reads F16 KV directly** — no F16-to-F32 cast of the entire cache.
//!    The existing `gqa_attention_kernel` takes F32 pointers, requiring the
//!    caller to upcast the F16 KV cache first (materialing 2x the data).
//! 2. **Online softmax** — O(1) shared memory for scores instead of
//!    O(seq_len). The existing kernel allocates `seq_len * sizeof(float)`
//!    in shared memory for the full score vector, which hits the 228 KB
//!    limit at seq_len ~57K.
//! 3. **Tiled K loads** — shared memory reuse reduces global memory traffic.
//!    Each K value is loaded once per tile instead of once per dot product.
//!
//! ## Launch config
//!
//! - Grid: `(num_heads, 1, 1)` -- one block per Q head
//! - Block: `(head_dim, 1, 1)` -- one thread per dimension (256 for Qwen3.5)
//! - Shared memory: `TILE_KV * head_dim * 2 + TILE_KV * 4 + n_warps * 4` bytes
//!   For TILE_KV=32, head_dim=256: 32*256*2 + 32*4 + 8*4 = 16,544 bytes
//!
//! ## Toggle
//!
//! Set `CHIMERE_FLASH_ATTN=1` to use this kernel instead of the existing
//! GQA attention kernel or Candle reference path.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use half::f16;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CUDA source (embedded for NVRTC compilation)
// ---------------------------------------------------------------------------

/// TILE_KV: number of KV positions processed per tile.
const TILE_KV: usize = 32;

const FLASH_ATTN_KERNEL_SRC: &str = r#"
// Flash Attention decode kernel for chimere-deltanet.
//
// Fused score + online-softmax + weighted-sum over F16 KV cache.
//
// Q:      [num_heads, head_dim]                F32 (single token, post-RoPE)
// K:      [num_kv_heads, seq_len, head_dim]    F16 as unsigned short
// V:      [num_kv_heads, seq_len, head_dim]    F16 as unsigned short
// output: [num_heads, head_dim]                F32
//
// Grid:  (num_heads, 1, 1)
// Block: (head_dim, 1, 1)   -- 256 threads = 8 warps for Qwen3.5
//
// Shared memory layout:
//   unsigned short k_tile[TILE_KV * head_dim]  -- K tile in F16 (coalesced load)
//   float          s_tile[TILE_KV]             -- scores for current tile
//   float          warp_reduce[n_warps]        -- cross-warp reduction workspace

#define TILE_KV 32

// F16 -> F32 conversion (software, avoids cuda_fp16.h dependency in NVRTC).
// Same implementation as chimere_kernels.cu.
__device__ __forceinline__ float f16_to_f32_fa(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant =  h        & 0x3FFu;
    if (exp == 0u) {
        // Subnormal
        float r = (float)mant * (1.0f / (1024.0f * 16384.0f));
        return sign ? -r : r;
    } else if (exp == 31u) {
        // Inf / NaN
        unsigned int f32 = (sign << 31) | 0x7F800000u | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    } else {
        // Normal
        unsigned int f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    }
}

extern "C" __global__ void flash_attn_decode_f16kv(
    const float*          __restrict__ Q,       // [num_heads * head_dim]
    const unsigned short* __restrict__ K,       // [num_kv_heads * seq_len * head_dim]
    const unsigned short* __restrict__ V,       // [num_kv_heads * seq_len * head_dim]
    float*                __restrict__ output,  // [num_heads * head_dim]
    int seq_len,
    int head_dim,
    int num_heads,
    int num_kv_heads,
    float scale
) {
    const int h   = blockIdx.x;               // Q head index [0, num_heads)
    const int tid = threadIdx.x;              // [0, head_dim), one thread per dimension
    const int kv_h = h * num_kv_heads / num_heads;  // GQA: KV head for this Q head

    // Each thread loads its Q dimension into a register (stays constant)
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = Q[h * head_dim + tid];
    }

    // Dynamic shared memory
    extern __shared__ char smem[];
    unsigned short* k_tile = (unsigned short*)smem;
    float* s_tile = (float*)(smem + TILE_KV * head_dim * sizeof(unsigned short));
    float* warp_reduce = s_tile + TILE_KV;

    // Online softmax accumulators (per-thread, each owns one output dimension)
    float o_acc = 0.0f;       // running weighted sum for output[h, tid]
    float m_run = -1e30f;     // running max score (for numerical stability)
    float l_run = 0.0f;       // running sum of exp(score - m)

    // Base pointers for this KV head in global memory
    const unsigned short* k_base = K + (long long)kv_h * seq_len * head_dim;
    const unsigned short* v_base = V + (long long)kv_h * seq_len * head_dim;
    const int n_warps = (blockDim.x + 31) >> 5;

    // -----------------------------------------------------------------------
    // Main loop: iterate over KV sequence in tiles of TILE_KV positions
    // -----------------------------------------------------------------------
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_KV) {
        int tile_end = tile_start + TILE_KV;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // --- Step 1: Cooperatively load K tile into shared memory ---
        // K_tile[j][d] for j in [0, tile_len), d in [0, head_dim)
        {
            int total_elems = tile_len * head_dim;
            for (int i = tid; i < total_elems; i += blockDim.x) {
                k_tile[i] = k_base[tile_start * head_dim + i];
            }
        }
        __syncthreads();

        // --- Step 2: Compute dot-product scores for each tile position ---
        // s[j] = dot(Q[h,:], K_tile[j,:]) * scale
        // Strategy: each thread contributes Q[tid] * K[j,tid], then reduce
        for (int j = 0; j < tile_len; j++) {
            // Each thread computes its partial product
            float partial = 0.0f;
            if (tid < head_dim) {
                partial = q_val * f16_to_f32_fa(k_tile[j * head_dim + tid]);
            }

            // Warp-level reduction (xor shuffle, all lanes get the sum)
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
            }

            // Cross-warp reduction: lane 0 of each warp writes to shared mem
            if ((tid & 31) == 0) {
                warp_reduce[tid >> 5] = partial;
            }
            __syncthreads();

            // Thread 0 sums all warp contributions
            if (tid == 0) {
                float dot = 0.0f;
                for (int w = 0; w < n_warps; w++) {
                    dot += warp_reduce[w];
                }
                s_tile[j] = dot * scale;
            }
            __syncthreads();
        }

        // --- Step 3: Find max score in this tile (for online softmax) ---
        float tile_max = -1e30f;
        for (int j = tid; j < tile_len; j += blockDim.x) {
            float s = s_tile[j];
            if (s > tile_max) tile_max = s;
        }
        // Warp reduce max
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, tile_max, offset);
            if (other > tile_max) tile_max = other;
        }
        if ((tid & 31) == 0) {
            warp_reduce[tid >> 5] = tile_max;
        }
        __syncthreads();
        if (tid == 0) {
            float m = warp_reduce[0];
            for (int w = 1; w < n_warps; w++) {
                if (warp_reduce[w] > m) m = warp_reduce[w];
            }
            warp_reduce[0] = m;
        }
        __syncthreads();
        tile_max = warp_reduce[0];
        __syncthreads();

        // --- Step 4: Online softmax update ---
        // m_new = max(m_run, tile_max)
        // correction = exp(m_run - m_new)  -- rescale previous accumulators
        // o_acc *= correction; l_run *= correction
        // For each j: p = exp(s[j] - m_new)
        //   l_run += p  (only thread 0 accumulates the denominator)
        //   o_acc += p * V[kv_h, tile_start+j, tid]  (all threads)
        float m_new = (tile_max > m_run) ? tile_max : m_run;
        float correction = expf(m_run - m_new);
        o_acc *= correction;
        l_run *= correction;

        for (int j = 0; j < tile_len; j++) {
            float p = expf(s_tile[j] - m_new);

            // Only one thread accumulates l_run (avoid 256x over-counting)
            if (tid == 0) {
                l_run += p;
            }

            // All threads with tid < head_dim accumulate V contribution
            if (tid < head_dim) {
                float v_f32 = f16_to_f32_fa(v_base[(tile_start + j) * head_dim + tid]);
                o_acc += p * v_f32;
            }
        }

        m_run = m_new;
        __syncthreads();
    }

    // -----------------------------------------------------------------------
    // Finalize: broadcast l_run and write normalized output
    // -----------------------------------------------------------------------
    if (tid == 0) {
        warp_reduce[0] = l_run;
    }
    __syncthreads();
    float l_final = warp_reduce[0];

    if (tid < head_dim) {
        output[h * head_dim + tid] = (l_final > 0.0f) ? (o_acc / l_final) : 0.0f;
    }
}
"#;

// ---------------------------------------------------------------------------
// PTX cache + compilation
// ---------------------------------------------------------------------------

const MODULE_NAME: &str = "chimere_flash_attn_v1";
const KERNEL_FUNC: &str = "flash_attn_decode_f16kv";

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
        FLASH_ATTN_KERNEL_SRC,
        &PTX_CACHE,
    )
}

/// Check (once) whether Flash Attention is enabled via `CHIMERE_FLASH_ATTN=1`.
pub fn is_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_FLASH_ATTN").is_ok());
    *ENABLED
}

// ---------------------------------------------------------------------------
// Raw API (CudaSlice / CudaView level)
// ---------------------------------------------------------------------------

/// Flash Attention decode: fused score + online-softmax + weighted-sum.
///
/// Takes F32 Q, F16 KV cache (as `half::f16`), outputs F32.
/// Zero Candle tensor ops — pure CUDA kernel launch.
///
/// # Layout
///
/// - `q`: `[num_heads * head_dim]` — current token query (F32, post-RoPE).
/// - `k_cache`: `[num_kv_heads * seq_len * head_dim]` — KV cache keys (F16).
/// - `v_cache`: `[num_kv_heads * seq_len * head_dim]` — KV cache values (F16).
/// - `output`: `[num_heads * head_dim]` — pre-allocated output buffer (F32).
///
/// The F16 `CudaView<half::f16>` pointers are passed directly to the CUDA kernel
/// which interprets them as `unsigned short*` (same 2-byte memory layout).
///
/// # Shared memory
///
/// Uses `TILE_KV * head_dim * 2 + TILE_KV * 4 + n_warps * 4` bytes per block.
/// For TILE_KV=32, head_dim=256: 16,544 bytes — well within sm_120 limits.
pub fn raw_flash_attn_decode(
    q: &CudaView<'_, f32>,
    k_cache: &CudaView<'_, f16>,
    v_cache: &CudaView<'_, f16>,
    output: &mut CudaSlice<f32>,
    num_heads: usize,
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    scale: f32,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "num_heads ({num_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
    );
    if seq_len == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev)?;

    let seq_len_i32 = seq_len as i32;
    let head_dim_i32 = head_dim as i32;
    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;

    // Shared memory: K_tile (F16) + scores (F32) + warp_reduce (F32)
    let n_warps = (head_dim + 31) / 32;
    let smem_bytes = (TILE_KV * head_dim * std::mem::size_of::<u16>()  // K tile: F16
        + TILE_KV * std::mem::size_of::<f32>()                         // scores
        + n_warps * std::mem::size_of::<f32>()                         // warp reduction
    ) as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (head_dim as u32, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    // Build kernel arguments.
    // CudaView<half::f16> passes the device pointer to the kernel, which
    // receives it as `const unsigned short*` -- same 2-byte layout.
    let mut builder = stream.launch_builder(&func);
    builder.arg(q);
    builder.arg(k_cache);
    builder.arg(v_cache);
    builder.arg(output);
    builder.arg(&seq_len_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&num_heads_i32);
    builder.arg(&num_kv_heads_i32);
    builder.arg(&scale);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("flash_attn_decode launch: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor-level wrapper
// ---------------------------------------------------------------------------

/// Flash Attention decode from Candle Tensors.
///
/// Takes Q (F32), K_cache (F16), V_cache (F16) as Candle Tensors and returns
/// the attention output as a new F32 Tensor. This is the entry point called
/// from `forward_attn_layer_moe`.
///
/// # Shapes
///
/// - `q`: `[1, num_heads, head_dim]` — current token query (F32, post-RoPE).
/// - `k_cache`: `[1, num_kv_heads, seq_len, head_dim]` — KV cache keys (F16).
/// - `v_cache`: `[1, num_kv_heads, seq_len, head_dim]` — KV cache values (F16).
///
/// Returns: `[1, num_heads, head_dim]` — attention output (F32, before gate).
///
/// # Type handling
///
/// - Q is cast to F32 if not already (usually already F32).
/// - KV cache is cast to F16 if not already (usually already F16).
/// - Output is always F32.
pub fn flash_attn_decode_tensor(
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
        _ => candle_core::bail!("flash_attn_decode: expected CUDA device"),
    };

    let seq_len = k_cache.dim(2)?;

    // Ensure Q is F32 and contiguous
    let q_c = q.to_dtype(candle_core::DType::F32)?.contiguous()?;
    // Ensure KV cache is F16 and contiguous
    let k_c = k_cache.to_dtype(candle_core::DType::F16)?.contiguous()?;
    let v_c = v_cache.to_dtype(candle_core::DType::F16)?.contiguous()?;

    // Extract CudaViews -- Q as f32
    let (q_stor, q_lay) = q_c.storage_and_layout();
    let q_cuda = match &*q_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("flash_attn: Q not on CUDA"),
    };
    let q_view = q_cuda
        .as_cuda_slice::<f32>()?
        .slice(q_lay.start_offset()..);

    // Extract CudaViews -- K as half::f16
    let (k_stor, k_lay) = k_c.storage_and_layout();
    let k_cuda = match &*k_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("flash_attn: K not on CUDA"),
    };
    let k_view = k_cuda
        .as_cuda_slice::<f16>()?
        .slice(k_lay.start_offset()..);

    // Extract CudaViews -- V as half::f16
    let (v_stor, v_lay) = v_c.storage_and_layout();
    let v_cuda = match &*v_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("flash_attn: V not on CUDA"),
    };
    let v_view = v_cuda
        .as_cuda_slice::<f16>()?
        .slice(v_lay.start_offset()..);

    // Allocate F32 output: [num_heads * head_dim]
    let out_elems = num_heads * head_dim;
    let cuda_stream = dev.cuda_stream();
    let mut output: CudaSlice<f32> = cuda_stream
        .alloc_zeros(out_elems)
        .map_err(|e| candle_core::Error::Msg(format!("alloc flash_attn output: {e}")))?;

    raw_flash_attn_decode(
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

    /// Reference CPU implementation of flash attention for testing.
    ///
    /// Computes: for each Q head h, kv_h = h / group_size,
    ///   scores[pos] = dot(Q[h,:], K[kv_h,pos,:]) * scale
    ///   weights = softmax(scores)
    ///   output[h,d] = sum(weights[pos] * V[kv_h,pos,d])
    fn reference_flash_attn(
        q: &[f32],
        k: &[f16],
        v: &[f16],
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let group_size = num_heads / num_kv_heads;
        let mut output = vec![0.0f32; num_heads * head_dim];

        for h in 0..num_heads {
            let kv_h = h / group_size;

            let mut scores = vec![0.0f32; seq_len];
            for pos in 0..seq_len {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q[h * head_dim + d];
                    let k_val = k[kv_h * seq_len * head_dim + pos * head_dim + d].to_f32();
                    dot += q_val * k_val;
                }
                scores[pos] = dot * scale;
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_e = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                sum_e += *s;
            }
            if sum_e > 0.0 {
                for s in &mut scores {
                    *s /= sum_e;
                }
            }

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for pos in 0..seq_len {
                    let v_val = v[kv_h * seq_len * head_dim + pos * head_dim + d].to_f32();
                    acc += scores[pos] * v_val;
                }
                output[h * head_dim + d] = acc;
            }
        }
        output
    }

    /// Helper: upload F16 data as `CudaSlice<f16>`.
    fn upload_f16(
        data: &[f16],
        dev: &CudaDevice,
    ) -> std::result::Result<CudaSlice<f16>, candle_core::Error> {
        let stream = dev.cuda_stream();
        let mut gpu: CudaSlice<f16> = stream.alloc_zeros(data.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc f16: {e}")))?;
        dev.memcpy_htod(data, &mut gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload f16: {e}")))?;
        Ok(gpu)
    }

    #[test]
    fn test_flash_attn_matches_reference() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[FLASH_ATTN] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let num_heads: usize = 16;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 256;
        let seq_len: usize = 37; // odd number to catch off-by-one
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Generate deterministic test data
        let q_h: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.0137 - 1.0).sin() * 0.3)
            .collect();
        let k_h: Vec<f16> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| f16::from_f32(((i as f32) * 0.0071 + 0.5).cos() * 0.2))
            .collect();
        let v_h: Vec<f16> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| f16::from_f32(((i as f32) * 0.0043 - 0.3).sin() * 0.4))
            .collect();

        // CPU reference
        let ref_output = reference_flash_attn(
            &q_h, &k_h, &v_h,
            num_heads, num_kv_heads, head_dim, seq_len, scale,
        );

        // GPU
        let stream = cuda_dev.cuda_stream();

        let mut q_gpu: CudaSlice<f32> = stream.alloc_zeros(q_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc q: {e}")))?;
        cuda_dev.memcpy_htod(&q_h, &mut q_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload q: {e}")))?;

        let k_gpu = upload_f16(&k_h, cuda_dev)?;
        let v_gpu = upload_f16(&v_h, cuda_dev)?;

        let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;

        raw_flash_attn_decode(
            &q_gpu.slice(..),
            &k_gpu.slice(..),
            &v_gpu.slice(..),
            &mut output_gpu,
            num_heads, num_kv_heads, seq_len, head_dim, scale,
            cuda_dev,
        )?;

        let gpu_output: Vec<f32> = stream.clone()
            .clone_dtoh(&output_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback: {e}")))?;

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
            "[FLASH_ATTN] num_heads={num_heads}, num_kv_heads={num_kv_heads}, \
             seq_len={seq_len}, head_dim={head_dim}, max_err={max_err:.2e} (idx={max_err_idx})"
        );
        assert!(
            max_err < 1e-2,
            "Flash attention mismatch: max_err={max_err:.6e} at idx={max_err_idx} \
             (tolerance 1e-2 due to F16 precision)"
        );

        eprintln!("[FLASH_ATTN] PASS: kernel matches reference");
        Ok(())
    }

    #[test]
    fn test_flash_attn_seq_len_0() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[FLASH_ATTN] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let num_heads: usize = 4;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 64;
        let seq_len: usize = 0;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let stream = cuda_dev.cuda_stream();

        let q_h = vec![1.0f32; num_heads * head_dim];
        let mut q_gpu: CudaSlice<f32> = stream.alloc_zeros(q_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        cuda_dev.memcpy_htod(&q_h, &mut q_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload: {e}")))?;

        let k_gpu: CudaSlice<f16> = stream.alloc_zeros(1)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        let v_gpu: CudaSlice<f16> = stream.alloc_zeros(1)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;

        let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;

        // seq_len=0 should return early without launching the kernel
        raw_flash_attn_decode(
            &q_gpu.slice(..),
            &k_gpu.slice(..),
            &v_gpu.slice(..),
            &mut output_gpu,
            num_heads, num_kv_heads, seq_len, head_dim, scale,
            cuda_dev,
        )?;

        eprintln!("[FLASH_ATTN] PASS: seq_len=0 handled gracefully");
        Ok(())
    }

    #[test]
    fn test_flash_attn_seq_len_1() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[FLASH_ATTN] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let num_heads: usize = 8;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 128;
        let seq_len: usize = 1;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_h: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k_h: Vec<f16> = (0..num_kv_heads * head_dim)
            .map(|i| f16::from_f32(((i as f32) * 0.007).cos()))
            .collect();
        let v_h: Vec<f16> = (0..num_kv_heads * head_dim)
            .map(|i| f16::from_f32(((i as f32) * 0.011).sin()))
            .collect();

        // For seq_len=1, softmax is trivially 1.0, output = V[kv_h, 0, :]
        let group_size = num_heads / num_kv_heads;
        let mut ref_output = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;
            for d in 0..head_dim {
                ref_output[h * head_dim + d] = v_h[kv_h * head_dim + d].to_f32();
            }
        }

        let stream = cuda_dev.cuda_stream();

        let mut q_gpu: CudaSlice<f32> = stream.alloc_zeros(q_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        cuda_dev.memcpy_htod(&q_h, &mut q_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload: {e}")))?;

        let k_gpu = upload_f16(&k_h, cuda_dev)?;
        let v_gpu = upload_f16(&v_h, cuda_dev)?;

        let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;

        raw_flash_attn_decode(
            &q_gpu.slice(..), &k_gpu.slice(..), &v_gpu.slice(..),
            &mut output_gpu,
            num_heads, num_kv_heads, seq_len, head_dim, scale,
            cuda_dev,
        )?;

        let gpu_output: Vec<f32> = stream.clone()
            .clone_dtoh(&output_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback: {e}")))?;

        let max_err = gpu_output.iter().zip(ref_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[FLASH_ATTN] seq_len=1 max_err={max_err:.2e}");
        assert!(max_err < 1e-3, "seq_len=1 mismatch: max_err={max_err:.6e}");

        eprintln!("[FLASH_ATTN] PASS: seq_len=1 edge case");
        Ok(())
    }

    #[test]
    fn test_flash_attn_large_seq_len() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[FLASH_ATTN] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        // seq_len=100 crosses multiple tiles (TILE_KV=32): 3 full + 1 partial
        let num_heads: usize = 4;
        let num_kv_heads: usize = 2;
        let head_dim: usize = 128;
        let seq_len: usize = 100;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let q_h: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.0137 - 1.0).sin() * 0.3)
            .collect();
        let k_h: Vec<f16> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| f16::from_f32(((i as f32) * 0.0071 + 0.5).cos() * 0.2))
            .collect();
        let v_h: Vec<f16> = (0..num_kv_heads * seq_len * head_dim)
            .map(|i| f16::from_f32(((i as f32) * 0.0043 - 0.3).sin() * 0.4))
            .collect();

        let ref_output = reference_flash_attn(
            &q_h, &k_h, &v_h,
            num_heads, num_kv_heads, head_dim, seq_len, scale,
        );

        let stream = cuda_dev.cuda_stream();

        let mut q_gpu: CudaSlice<f32> = stream.alloc_zeros(q_h.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;
        cuda_dev.memcpy_htod(&q_h, &mut q_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload: {e}")))?;

        let k_gpu = upload_f16(&k_h, cuda_dev)?;
        let v_gpu = upload_f16(&v_h, cuda_dev)?;

        let mut output_gpu: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("alloc: {e}")))?;

        raw_flash_attn_decode(
            &q_gpu.slice(..), &k_gpu.slice(..), &v_gpu.slice(..),
            &mut output_gpu,
            num_heads, num_kv_heads, seq_len, head_dim, scale,
            cuda_dev,
        )?;

        let gpu_output: Vec<f32> = stream.clone()
            .clone_dtoh(&output_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("readback: {e}")))?;

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
            "[FLASH_ATTN] seq_len={seq_len} (multi-tile): max_err={max_err:.2e} (idx={max_err_idx})"
        );
        assert!(
            max_err < 1e-2,
            "Multi-tile mismatch: max_err={max_err:.6e} at idx={max_err_idx}"
        );

        eprintln!("[FLASH_ATTN] PASS: multi-tile seq_len={seq_len}");
        Ok(())
    }

    /// Test the Tensor-level wrapper against the Candle reference path.
    #[test]
    fn test_flash_attn_tensor_wrapper() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        if !matches!(&device, Device::Cuda(_)) {
            eprintln!("[FLASH_ATTN] No CUDA device, skipping test.");
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
        )?.to_dtype(candle_core::DType::F16)?;
        let v_cache = candle_core::Tensor::randn(
            0.0f32, 1.0,
            &[1, num_kv_heads, seq_len, head_dim],
            &device,
        )?.to_dtype(candle_core::DType::F16)?;

        // Run flash attention kernel
        let flash_out = flash_attn_decode_tensor(
            &q, &k_cache, &v_cache,
            num_heads, num_kv_heads, head_dim, scale,
        )?;

        // Run Candle reference: cast to F32, expand GQA, matmul softmax matmul
        let k_f32 = k_cache.to_dtype(candle_core::DType::F32)?;
        let v_f32 = v_cache.to_dtype(candle_core::DType::F32)?;
        let q_attn = q.unsqueeze(2)?; // [1, num_heads, 1, head_dim]
        let k_exp = k_f32
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, seq_len, head_dim))?;
        let v_exp = v_f32
            .unsqueeze(2)?
            .expand((1, num_kv_heads, group_size, seq_len, head_dim))?
            .contiguous()?
            .reshape((1, num_heads, seq_len, head_dim))?;
        let scores = q_attn.matmul(&k_exp.transpose(2, 3)?)?;
        let scores = (scores * (scale as f64))?;
        let weights = candle_nn::ops::softmax(&scores, candle_core::D::Minus1)?;
        let ref_out = weights.matmul(&v_exp)?.squeeze(2)?; // [1, num_heads, head_dim]

        // Compare
        let flash_flat = flash_out.flatten_all()?.to_vec1::<f32>()?;
        let ref_flat = ref_out.flatten_all()?.to_vec1::<f32>()?;

        let max_err = flash_flat.iter().zip(ref_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[FLASH_ATTN] Tensor wrapper max_err={max_err:.2e}");
        assert!(
            max_err < 5e-2,
            "Tensor wrapper mismatch: max_err={max_err:.6e} (tolerance 5e-2 for F16 KV)"
        );

        eprintln!("[FLASH_ATTN] PASS: Tensor wrapper matches Candle reference");
        Ok(())
    }
}
