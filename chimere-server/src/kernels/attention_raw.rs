//! Raw CUDA kernels for the attention layer (Phase 4 of raw-forward-v4).
//!
//! All functions take pre-allocated `CudaSlice<f32>` inputs/outputs and
//! perform zero Tensor allocations. Kernels are compiled via NVRTC at first
//! use and cached in `OnceLock` statics (same pattern as `elementwise.rs`).
//!
//! Kernels provided:
//!   - `raw_mrope_apply`          — fused MRoPE rotation (all sections in one launch)
//!   - `raw_deinterleave_q_gate`  — split Q+gate from interleaved layout
//!   - `raw_rms_norm_per_head`    — per-head RMSNorm for QK normalization
//!   - `raw_sigmoid_gate_mul`     — sigmoid(gate) * attn_out
//!   - `raw_kv_append`            — append single K,V row to ring-buffer cache
//!
//! Batch kernels (V2-2 prefill):
//!   - `raw_deinterleave_q_gate_batch` — batch deinterleave for N tokens
//!   - `raw_rms_norm_per_head_batch`   — batch per-head RMSNorm for N tokens
//!   - `raw_mrope_apply_batch`         — batch MRoPE for N tokens (positions from GPU array)

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// =========================================================================
// CUDA kernel sources
// =========================================================================

/// CUDA source for the attention-layer elementwise kernels.
///
/// Contains:
///   - `deinterleave_q_gate_kernel`
///   - `rms_norm_per_head_kernel`
///   - `sigmoid_gate_mul_kernel`
const ATTN_ELEMWISE_KERNEL_SRC: &str = r#"
// -----------------------------------------------------------------------
// Deinterleave Q+gate from the Qwen3.5 interleaved layout.
//
// Input layout (from the WQ projection):
//   [Q_h0(head_dim), gate_h0(head_dim), Q_h1(head_dim), gate_h1(head_dim), ...]
//   Total size = num_heads * 2 * head_dim
//
// Outputs:
//   q_out:    [num_heads * head_dim]   — Q heads concatenated
//   gate_out: [num_heads * head_dim]   — gate heads concatenated
//
// Grid: (num_heads, 1, 1)   Block: (256, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void deinterleave_q_gate_kernel(
    const float* __restrict__ input,     // [num_heads * 2 * head_dim]
    float*       __restrict__ q_out,     // [num_heads * head_dim]
    float*       __restrict__ gate_out,  // [num_heads * head_dim]
    int num_heads,
    int head_dim
) {
    int h = blockIdx.x;        // head index
    int d = threadIdx.x;       // dimension within head (loops if head_dim > blockDim)

    int in_base = h * 2 * head_dim;      // start of [Q_h, gate_h] in input
    int out_base = h * head_dim;          // start of head h in q_out / gate_out

    for (int i = d; i < head_dim; i += blockDim.x) {
        q_out[out_base + i]    = input[in_base + i];
        gate_out[out_base + i] = input[in_base + head_dim + i];
    }
}

// -----------------------------------------------------------------------
// Per-head RMSNorm.
//
// For each head h in [0, num_heads):
//   rms = sqrt(mean(x[h,:]^2) + eps)
//   output[h, d] = x[h, d] / rms * weight[d]
//
// The weight vector is shared across all heads (same learned norm per dim).
//
// Grid: (num_heads, 1, 1)   Block: (block_size, 1, 1)
// Shared mem: block_size * sizeof(float)
// -----------------------------------------------------------------------
extern "C" __global__ void rms_norm_per_head_kernel(
    const float* __restrict__ input,    // [num_heads * head_dim]
    const float* __restrict__ weight,   // [head_dim]
    float*       __restrict__ output,   // [num_heads * head_dim]
    int num_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float* sdata = (float*)smem_raw;

    int base = h * head_dim;

    // Step 1: compute sum of squares
    float local_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = input[base + i];
        local_sq += v * v;
    }
    sdata[tid] = local_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    // Step 2: normalize and scale
    for (int i = tid; i < head_dim; i += blockDim.x) {
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }
}

// -----------------------------------------------------------------------
// sigmoid(gate) * attn_output
//
// output[i] = sigmoid(gate[i]) * input[i]
//           = input[i] / (1 + exp(-gate[i]))
//
// Grid: (ceil(n/256), 1, 1)   Block: (256, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void sigmoid_gate_mul_kernel(
    const float* __restrict__ input,   // attn_output [n]
    const float* __restrict__ gate,    // gate values [n]
    float*       __restrict__ output,  // [n]
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        output[i] = g * input[i];
    }
}
"#;

/// CUDA source for the fused MRoPE kernel.
///
/// Applies Multi-Resolution Rotary Position Embedding to a tensor of shape
/// `[num_heads, head_dim]`.  The first `n_rot` dimensions are split into
/// sections (each with its own position), and each section applies the
/// standard half-rotation:
///
///   out[first_half]  = x[first_half] * cos - x[second_half] * sin
///   out[second_half] = x[first_half] * sin + x[second_half] * cos
///
/// Dimensions beyond `n_rot` are passed through unchanged.
///
/// Grid: (num_heads, 1, 1)   Block: (256, 1, 1)
const MROPE_KERNEL_SRC: &str = r#"
extern "C" __global__ void mrope_apply_kernel(
    const float* __restrict__ input,   // [num_heads * head_dim]
    float*       __restrict__ output,  // [num_heads * head_dim]
    const float* __restrict__ cos_s0,  // [MAX_POS * pairs_s0]
    const float* __restrict__ sin_s0,
    const float* __restrict__ cos_s1,  // [MAX_POS * pairs_s1]
    const float* __restrict__ sin_s1,
    const float* __restrict__ cos_s2,  // [MAX_POS * pairs_s2]
    const float* __restrict__ sin_s2,
    int pos_s0, int pos_s1, int pos_s2,
    int num_heads, int head_dim,
    int pairs_s0, int pairs_s1, int pairs_s2,
    int n_rot
) {
    int h = blockIdx.x;      // head index
    int base = h * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float x = input[base + d];

        if (d >= n_rot) {
            // Pass-through: no rotation for dims beyond n_rot
            output[base + d] = x;
            continue;
        }

        // Determine which section this dimension belongs to.
        // Layout within the rotary region:
        //   section 0: dims [0 .. 2*pairs_s0)
        //   section 1: dims [2*pairs_s0 .. 2*pairs_s0 + 2*pairs_s1)
        //   section 2: dims [2*(pairs_s0+pairs_s1) .. n_rot)
        //
        // Within each section, the first half are the x1 elements and the
        // second half are the x2 elements.  The MRoPE rotation is:
        //   x1' = x1 * cos - x2 * sin
        //   x2' = x1 * sin + x2 * cos

        int dim_s0 = pairs_s0 * 2;
        int dim_s1 = pairs_s1 * 2;

        float cos_val, sin_val;
        int partner_d;
        int is_first;  // 1 if this is x1 (first half), 0 if x2

        if (d < dim_s0) {
            // Section 0
            int local_d = d;
            int n_pairs = pairs_s0;
            is_first = local_d < n_pairs ? 1 : 0;
            int pair_idx = is_first ? local_d : local_d - n_pairs;
            partner_d = is_first ? (d + n_pairs) : (d - n_pairs);
            cos_val = cos_s0[pos_s0 * n_pairs + pair_idx];
            sin_val = sin_s0[pos_s0 * n_pairs + pair_idx];
        } else if (d < dim_s0 + dim_s1) {
            // Section 1
            int local_d = d - dim_s0;
            int n_pairs = pairs_s1;
            is_first = local_d < n_pairs ? 1 : 0;
            int pair_idx = is_first ? local_d : local_d - n_pairs;
            partner_d = is_first ? (d + n_pairs) : (d - n_pairs);
            cos_val = cos_s1[pos_s1 * n_pairs + pair_idx];
            sin_val = sin_s1[pos_s1 * n_pairs + pair_idx];
        } else {
            // Section 2
            int local_d = d - dim_s0 - dim_s1;
            int n_pairs = pairs_s2;
            is_first = local_d < n_pairs ? 1 : 0;
            int pair_idx = is_first ? local_d : local_d - n_pairs;
            partner_d = is_first ? (d + n_pairs) : (d - n_pairs);
            cos_val = cos_s2[pos_s2 * n_pairs + pair_idx];
            sin_val = sin_s2[pos_s2 * n_pairs + pair_idx];
        }

        float partner = input[base + partner_d];

        if (is_first) {
            // x1' = x1 * cos - x2 * sin
            output[base + d] = x * cos_val - partner * sin_val;
        } else {
            // x2' = x1 * sin + x2 * cos
            output[base + d] = partner * sin_val + x * cos_val;
        }
    }
}
"#;

/// CUDA source for batch attention elementwise kernels (V2-2 prefill).
///
/// These kernels process N tokens at once. Each block handles one head of one
/// token: Grid=(N * num_heads, 1, 1), with `bid / num_heads` = token index
/// and `bid % num_heads` = head index.
///
/// Contains:
///   - `deinterleave_q_gate_batch_kernel`
///   - `rms_norm_per_head_batch_kernel`
///   - `mrope_apply_batch_kernel`
const ATTN_BATCH_KERNEL_SRC: &str = r#"
// -----------------------------------------------------------------------
// Batch deinterleave Q+gate from Qwen3.5 interleaved layout (N tokens).
//
// Input layout:  [N * num_heads * 2 * head_dim]
//   For token t, head h: input[t * num_heads * 2 * head_dim + h * 2 * head_dim ..]
//   = [Q_h(head_dim), gate_h(head_dim)]
//
// Outputs:
//   q_out:    [N * num_heads * head_dim]
//   gate_out: [N * num_heads * head_dim]
//
// Grid: (N * num_heads, 1, 1)   Block: (256, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void deinterleave_q_gate_batch_kernel(
    const float* __restrict__ input,
    float*       __restrict__ q_out,
    float*       __restrict__ gate_out,
    int num_heads,
    int head_dim,
    int n_tokens
) {
    int bid = blockIdx.x;
    int t = bid / num_heads;
    int h = bid % num_heads;
    if (t >= n_tokens) return;

    int in_base  = t * num_heads * 2 * head_dim + h * 2 * head_dim;
    int out_base = t * num_heads * head_dim + h * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_out[out_base + i]    = input[in_base + i];
        gate_out[out_base + i] = input[in_base + head_dim + i];
    }
}

// -----------------------------------------------------------------------
// Batch per-head RMSNorm (N tokens).
//
// For each token t and head h:
//   rms = sqrt(mean(x[t,h,:]^2) + eps)
//   output[t,h,d] = x[t,h,d] / rms * weight[d]
//
// Weight is shared across all heads and tokens.
//
// Grid: (N * num_heads, 1, 1)   Block: (block_size, 1, 1)
// Shared mem: block_size * sizeof(float)
// -----------------------------------------------------------------------
extern "C" __global__ void rms_norm_per_head_batch_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    int num_heads,
    int head_dim,
    float eps,
    int n_tokens
) {
    int bid = blockIdx.x;
    int t = bid / num_heads;
    int h = bid % num_heads;
    if (t >= n_tokens) return;

    int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float* sdata = (float*)smem_raw;

    int base = t * num_heads * head_dim + h * head_dim;

    // Step 1: compute sum of squares
    float local_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = input[base + i];
        local_sq += v * v;
    }
    sdata[tid] = local_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    // Step 2: normalize and scale
    for (int i = tid; i < head_dim; i += blockDim.x) {
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }
}

// -----------------------------------------------------------------------
// Batch MRoPE (N tokens).
//
// Same rotation as mrope_apply_kernel but reads per-token positions from
// a GPU array `positions[t]` (section 0 only; sections 1,2 = 0 for text).
//
// Grid: (N * num_heads, 1, 1)   Block: (256, 1, 1)
// -----------------------------------------------------------------------
extern "C" __global__ void mrope_apply_batch_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const int*   __restrict__ positions,  // [n_tokens] — per-token position for section 0
    const float* __restrict__ cos_s0,
    const float* __restrict__ sin_s0,
    const float* __restrict__ cos_s1,
    const float* __restrict__ sin_s1,
    const float* __restrict__ cos_s2,
    const float* __restrict__ sin_s2,
    int num_heads, int head_dim,
    int pairs_s0, int pairs_s1, int pairs_s2,
    int n_rot,
    int n_tokens
) {
    int bid = blockIdx.x;
    int t = bid / num_heads;
    int h = bid % num_heads;
    if (t >= n_tokens) return;

    int base = t * num_heads * head_dim + h * head_dim;

    // For text: section 0 = positions[t], sections 1,2 = 0
    int pos_s0 = positions[t];
    int pos_s1 = 0;
    int pos_s2 = 0;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float x = input[base + d];

        if (d >= n_rot) {
            output[base + d] = x;
            continue;
        }

        int dim_s0 = pairs_s0 * 2;
        int dim_s1 = pairs_s1 * 2;

        float cos_val, sin_val;
        int partner_d;
        int is_first;

        if (d < dim_s0) {
            int local_d = d;
            int n_pairs = pairs_s0;
            is_first = local_d < n_pairs ? 1 : 0;
            int pair_idx = is_first ? local_d : local_d - n_pairs;
            partner_d = is_first ? (d + n_pairs) : (d - n_pairs);
            cos_val = cos_s0[pos_s0 * n_pairs + pair_idx];
            sin_val = sin_s0[pos_s0 * n_pairs + pair_idx];
        } else if (d < dim_s0 + dim_s1) {
            int local_d = d - dim_s0;
            int n_pairs = pairs_s1;
            is_first = local_d < n_pairs ? 1 : 0;
            int pair_idx = is_first ? local_d : local_d - n_pairs;
            partner_d = is_first ? (d + n_pairs) : (d - n_pairs);
            cos_val = cos_s1[pos_s1 * n_pairs + pair_idx];
            sin_val = sin_s1[pos_s1 * n_pairs + pair_idx];
        } else {
            int local_d = d - dim_s0 - dim_s1;
            int n_pairs = pairs_s2;
            is_first = local_d < n_pairs ? 1 : 0;
            int pair_idx = is_first ? local_d : local_d - n_pairs;
            partner_d = is_first ? (d + n_pairs) : (d - n_pairs);
            cos_val = cos_s2[pos_s2 * n_pairs + pair_idx];
            sin_val = sin_s2[pos_s2 * n_pairs + pair_idx];
        }

        float partner = input[base + partner_d];

        if (is_first) {
            output[base + d] = x * cos_val - partner * sin_val;
        } else {
            output[base + d] = partner * sin_val + x * cos_val;
        }
    }
}
"#;

// =========================================================================
// PTX caches
// =========================================================================

static ATTN_ELEMWISE_PTX: OnceLock<String> = OnceLock::new();
static MROPE_PTX: OnceLock<String> = OnceLock::new();
static ATTN_BATCH_PTX: OnceLock<String> = OnceLock::new();

// =========================================================================
// Raw MRoPE precomputed tables (extracted from MRoPE struct)
// =========================================================================

/// Precomputed MRoPE cos/sin tables as raw `CudaSlice<f32>` on GPU.
///
/// Extracted once from `MRoPE::device_cache` at model load time and passed
/// to `raw_mrope_apply` for zero-Tensor rotation.
///
/// Table layout: `cos_s[pos * pairs + pair_idx]` where `pos` is the
/// sequence position and `pair_idx` is the pair index within the section.
pub struct RawMRoPETables {
    /// Section 0 cosine table: `[MAX_POS * pairs[0]]`
    pub cos_s0: CudaSlice<f32>,
    /// Section 0 sine table
    pub sin_s0: CudaSlice<f32>,
    /// Section 1 cosine table: `[MAX_POS * pairs[1]]`
    pub cos_s1: CudaSlice<f32>,
    /// Section 1 sine table
    pub sin_s1: CudaSlice<f32>,
    /// Section 2 cosine table: `[MAX_POS * pairs[2]]`
    pub cos_s2: CudaSlice<f32>,
    /// Section 2 sine table
    pub sin_s2: CudaSlice<f32>,
    /// Number of pairs per section: e.g. `[11, 11, 10]` for Qwen3.5.
    pub pairs: [usize; 3],
    /// Total number of rotary dimensions (e.g. 64).
    pub n_rot: usize,
    /// Full head dimension (e.g. 256).
    pub head_dim: usize,
}

impl RawMRoPETables {
    /// Build precomputed MRoPE cos/sin tables on GPU from model config.
    ///
    /// Computes cos/sin for all positions 0..max_pos for each section,
    /// then uploads the tables to GPU as flat `CudaSlice<f32>`.
    ///
    /// # Arguments
    ///
    /// - `head_dim`  — full head dim (e.g. 256)
    /// - `n_rot`     — how many leading dims get RoPE (e.g. 64)
    /// - `sections`  — pairs per section, e.g. `[11, 11, 10, 0]`
    /// - `rope_theta` — base frequency (e.g. 10_000_000.0)
    /// - `max_pos`   — maximum sequence position to precompute
    /// - `dev`       — CUDA device to upload tables to
    pub fn from_config(
        head_dim: usize,
        n_rot: usize,
        sections: &[usize; 4],
        rope_theta: f64,
        max_pos: usize,
        dev: &CudaDevice,
    ) -> Result<Self> {
        // Compute per-pair inverse frequencies for each section.
        // freq[s][i] = rope_theta ^ (-2 * global_pair / n_rot)
        let mut global_offset = 0usize;
        let mut section_freqs: Vec<Vec<f64>> = Vec::new();
        for &n_pairs in sections.iter() {
            let freqs: Vec<f64> = (0..n_pairs)
                .map(|i| {
                    let g = global_offset + i;
                    let exponent = (2 * g) as f64 / n_rot as f64;
                    rope_theta.powf(-exponent)
                })
                .collect();
            section_freqs.push(freqs);
            global_offset += n_pairs;
        }

        // For each of the 3 active sections, build [max_pos * n_pairs] tables.
        let mut cos_vecs: Vec<Vec<f32>> = Vec::new();
        let mut sin_vecs: Vec<Vec<f32>> = Vec::new();
        for s in 0..3 {
            let n_pairs = sections[s];
            let mut cos_data = Vec::with_capacity(max_pos * n_pairs.max(1));
            let mut sin_data = Vec::with_capacity(max_pos * n_pairs.max(1));
            if n_pairs == 0 {
                // Placeholder: 1 element per position (never read, but CudaSlice can't be empty)
                cos_data.resize(max_pos, 0.0);
                sin_data.resize(max_pos, 0.0);
            } else {
                for pos in 0..max_pos {
                    for pair_i in 0..n_pairs {
                        let angle = pos as f64 * section_freqs[s][pair_i];
                        cos_data.push(angle.cos() as f32);
                        sin_data.push(angle.sin() as f32);
                    }
                }
            }
            cos_vecs.push(cos_data);
            sin_vecs.push(sin_data);
        }

        // Upload to GPU
        let upload = |data: &[f32], name: &str| -> Result<CudaSlice<f32>> {
            let mut dst = dev.alloc_zeros::<f32>(data.len())
                .map_err(|e| candle_core::Error::Msg(format!("alloc {name}: {e}")))?;
            dev.memcpy_htod(data, &mut dst)
                .map_err(|e| candle_core::Error::Msg(format!("htod {name}: {e}")))?;
            Ok(dst)
        };

        let pairs = [sections[0], sections[1], sections[2]];
        let total_bytes = cos_vecs.iter().map(|v| v.len()).sum::<usize>() * 4 * 2;
        eprintln!(
            "[ROPE_TABLES] Precomputed MRoPE tables: {} positions, pairs={:?}, {:.1} KB GPU",
            max_pos, pairs, total_bytes as f64 / 1024.0,
        );

        Ok(Self {
            cos_s0: upload(&cos_vecs[0], "cos_s0")?,
            sin_s0: upload(&sin_vecs[0], "sin_s0")?,
            cos_s1: upload(&cos_vecs[1], "cos_s1")?,
            sin_s1: upload(&sin_vecs[1], "sin_s1")?,
            cos_s2: upload(&cos_vecs[2], "cos_s2")?,
            sin_s2: upload(&sin_vecs[2], "sin_s2")?,
            pairs,
            n_rot,
            head_dim,
        })
    }
}

// =========================================================================
// Public API
// =========================================================================

/// Apply Multi-Resolution Rotary Position Embedding to Q or K heads.
///
/// # Arguments
///
/// - `input`  — `[num_heads * head_dim]` flat f32 buffer (Q or K after norm)
/// - `output` — pre-allocated `[num_heads * head_dim]` output buffer
/// - `tables` — precomputed cos/sin tables on GPU
/// - `position` — current sequence position (for text: only section 0 uses this)
/// - `num_heads` — number of heads (16 for Q, 2 for K in Qwen3.5-35B)
/// - `dev` — CUDA device
///
/// For text-only inference, `pos_s0 = position`, `pos_s1 = 0`, `pos_s2 = 0`.
/// This matches `MRoPE::text_positions(1, position)`.
///
/// # Cost
///
/// 1 kernel launch, 0 Tensor allocations, 0 cudaMalloc.
pub fn raw_mrope_apply(
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    tables: &RawMRoPETables,
    position: usize,
    num_heads: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "mrope_apply_kernel", "chimere_mrope_raw_v1",
        MROPE_KERNEL_SRC, &MROPE_PTX,
    )?;

    // For text-only: section 0 gets the position, sections 1/2 stay at 0.
    let pos_s0 = position as i32;
    let pos_s1 = 0i32;
    let pos_s2 = 0i32;
    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = tables.head_dim as i32;
    let pairs_s0 = tables.pairs[0] as i32;
    let pairs_s1 = tables.pairs[1] as i32;
    let pairs_s2 = tables.pairs[2] as i32;
    let n_rot_i32 = tables.n_rot as i32;

    // Block size: 256 threads, loop over head_dim if > 256
    let block_size = 256u32.min(tables.head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&tables.cos_s0);
    builder.arg(&tables.sin_s0);
    builder.arg(&tables.cos_s1);
    builder.arg(&tables.sin_s1);
    builder.arg(&tables.cos_s2);
    builder.arg(&tables.sin_s2);
    builder.arg(&pos_s0);
    builder.arg(&pos_s1);
    builder.arg(&pos_s2);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&pairs_s0);
    builder.arg(&pairs_s1);
    builder.arg(&pairs_s2);
    builder.arg(&n_rot_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("mrope_apply launch: {e}")))?;
    Ok(())
}

/// Extended MRoPE apply with explicit per-section positions (for multimodal).
///
/// Same as `raw_mrope_apply` but allows specifying positions for all 3 sections
/// independently, which is needed for image/video inputs where sections 1/2
/// encode spatial coordinates.
pub fn raw_mrope_apply_multimodal(
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    tables: &RawMRoPETables,
    pos_s0: usize,
    pos_s1: usize,
    pos_s2: usize,
    num_heads: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "mrope_apply_kernel", "chimere_mrope_raw_v1",
        MROPE_KERNEL_SRC, &MROPE_PTX,
    )?;

    let pos_s0_i32 = pos_s0 as i32;
    let pos_s1_i32 = pos_s1 as i32;
    let pos_s2_i32 = pos_s2 as i32;
    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = tables.head_dim as i32;
    let pairs_s0 = tables.pairs[0] as i32;
    let pairs_s1 = tables.pairs[1] as i32;
    let pairs_s2 = tables.pairs[2] as i32;
    let n_rot_i32 = tables.n_rot as i32;

    let block_size = 256u32.min(tables.head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&tables.cos_s0);
    builder.arg(&tables.sin_s0);
    builder.arg(&tables.cos_s1);
    builder.arg(&tables.sin_s1);
    builder.arg(&tables.cos_s2);
    builder.arg(&tables.sin_s2);
    builder.arg(&pos_s0_i32);
    builder.arg(&pos_s1_i32);
    builder.arg(&pos_s2_i32);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&pairs_s0);
    builder.arg(&pairs_s1);
    builder.arg(&pairs_s2);
    builder.arg(&n_rot_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("mrope_apply_multimodal launch: {e}")))?;
    Ok(())
}

/// Split Q+gate from Qwen3.5's interleaved layout into separate buffers.
///
/// The WQ projection outputs `[Q_h0(dim), gate_h0(dim), Q_h1(dim), gate_h1(dim), ...]`
/// with total size `num_heads * 2 * head_dim`. This kernel deinterleaves into:
/// - `q_out`:    `[num_heads * head_dim]` — all Q heads concatenated
/// - `gate_out`: `[num_heads * head_dim]` — all gate heads concatenated
///
/// # Cost
///
/// 1 kernel launch, 0 Tensor allocations.
pub fn raw_deinterleave_q_gate(
    input: &CudaSlice<f32>,
    q_out: &mut CudaSlice<f32>,
    gate_out: &mut CudaSlice<f32>,
    num_heads: usize,
    head_dim: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "deinterleave_q_gate_kernel", "chimere_attn_elemwise_v1",
        ATTN_ELEMWISE_KERNEL_SRC, &ATTN_ELEMWISE_PTX,
    )?;

    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = head_dim as i32;

    // One block per head; threads within the block cover head_dim
    let block_size = 256u32.min(head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(q_out);
    builder.arg(gate_out);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("deinterleave_q_gate launch: {e}")))?;
    Ok(())
}

/// Per-head RMSNorm for QK normalization.
///
/// Applies RMSNorm independently to each head:
///
///   For head h: output[h, d] = input[h, d] / rms(input[h, :]) * weight[d]
///
/// The `weight` vector has shape `[head_dim]` and is shared across all heads
/// (this is how Qwen3.5 implements q_norm / k_norm).
///
/// Note: `input` and `output` may alias (in-place normalization is safe because
/// the kernel reads the full head into shared memory before writing).
///
/// # Arguments
///
/// - `input`    — `[num_heads * head_dim]` flat buffer
/// - `weight`   — `[head_dim]` learned RMSNorm weight
/// - `output`   — `[num_heads * head_dim]` pre-allocated output (may alias input)
/// - `num_heads` — number of heads
/// - `head_dim`  — dimension per head
/// - `eps`       — RMSNorm epsilon (e.g. 1e-6)
/// - `dev`       — CUDA device
///
/// # Cost
///
/// 1 kernel launch, 0 Tensor allocations.
pub fn raw_rms_norm_per_head(
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "rms_norm_per_head_kernel", "chimere_attn_elemwise_v1",
        ATTN_ELEMWISE_KERNEL_SRC, &ATTN_ELEMWISE_PTX,
    )?;

    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = head_dim as i32;

    // Block size: next power of 2 of head_dim, clamped to [32, 1024].
    // Must be power of 2 for the shared-memory reduction to work correctly.
    let block_size = (head_dim as u32).next_power_of_two().max(32).min(1024);
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4, // float per thread for reduction
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(weight);
    builder.arg(output);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&eps);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("rms_norm_per_head launch: {e}")))?;
    Ok(())
}

/// Compute `sigmoid(gate) * attn_output` element-wise.
///
/// This is the gating step after attention: the raw gate values from the
/// WQ interleaved projection are passed through sigmoid and multiplied
/// with the attention output before the output projection (WO).
///
/// # Arguments
///
/// - `attn_output` — `[n]` attention output values
/// - `gate`        — `[n]` raw gate values (before sigmoid)
/// - `output`      — `[n]` pre-allocated output buffer
/// - `n`           — number of elements (= num_heads * head_dim)
/// - `dev`         — CUDA device
///
/// # Cost
///
/// 1 kernel launch, 0 Tensor allocations.
pub fn raw_sigmoid_gate_mul(
    attn_output: &CudaSlice<f32>,
    gate: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    n: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "sigmoid_gate_mul_kernel", "chimere_attn_elemwise_v1",
        ATTN_ELEMWISE_KERNEL_SRC, &ATTN_ELEMWISE_PTX,
    )?;

    let blocks = ((n as u32) + 255) / 256;
    let n_i32 = n as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(attn_output);
    builder.arg(gate);
    builder.arg(output);
    builder.arg(&n_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("sigmoid_gate_mul launch: {e}")))?;
    Ok(())
}

/// Append a single K,V pair to the KV cache ring buffer.
///
/// Copies `k_new` and `v_new` (single-token K and V, each of shape
/// `[num_kv_heads * head_dim]`) into the appropriate position of the
/// KV cache buffers, then increments `kv_len`.
///
/// The cache layout is `[num_kv_heads, max_cap, head_dim]` stored flat.
/// The new K/V row is written at position `kv_len` for each KV head.
///
/// If `kv_len >= kv_cap`, returns an error (the caller must grow the
/// buffers before calling this function).
///
/// # Arguments
///
/// - `k_new`    — `[num_kv_heads * head_dim]` new K row
/// - `v_new`    — `[num_kv_heads * head_dim]` new V row
/// - `kv_k`     — KV cache for keys: `[num_kv_heads * kv_cap * head_dim]`
/// - `kv_v`     — KV cache for values: same shape
/// - `kv_len`   — current number of filled positions (updated on return)
/// - `kv_cap`   — current capacity of the cache
/// - `num_kv_heads` — number of KV heads (2 for Qwen3.5-35B)
/// - `head_dim`     — dimension per head (256 for Qwen3.5)
/// - `dev`          — CUDA device
///
/// # Cost
///
/// `num_kv_heads` device-to-device memcpy calls (one per head, each copying
/// `head_dim` floats = 1 KB for Qwen3.5). No kernel launch needed.
pub fn raw_kv_append(
    k_new: &CudaSlice<f32>,
    v_new: &CudaSlice<f32>,
    kv_k: &mut CudaSlice<f32>,
    kv_v: &mut CudaSlice<f32>,
    kv_len: &mut usize,
    kv_cap: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let pos = *kv_len;
    if pos >= kv_cap {
        candle_core::bail!(
            "KV cache full: position {} >= capacity {}. Caller must grow buffers.",
            pos, kv_cap
        );
    }

    // Cache layout: [num_kv_heads, kv_cap, head_dim] stored row-major.
    // For KV head `h`, position `pos`, the offset is:
    //   h * kv_cap * head_dim + pos * head_dim
    //
    // k_new/v_new layout: [num_kv_heads, head_dim] flat.
    // For KV head `h`, the source offset is: h * head_dim

    for h in 0..num_kv_heads {
        let src_offset = h * head_dim;
        let dst_offset = h * kv_cap * head_dim + pos * head_dim;

        // Create views into the source and destination slices, then copy
        let k_src = k_new.slice(src_offset..src_offset + head_dim);
        let mut k_dst = kv_k.slice_mut(dst_offset..dst_offset + head_dim);
        dev.memcpy_dtod(&k_src, &mut k_dst)
            .map_err(|e| candle_core::Error::Msg(format!("kv_append k memcpy head {h}: {e}")))?;

        let v_src = v_new.slice(src_offset..src_offset + head_dim);
        let mut v_dst = kv_v.slice_mut(dst_offset..dst_offset + head_dim);
        dev.memcpy_dtod(&v_src, &mut v_dst)
            .map_err(|e| candle_core::Error::Msg(format!("kv_append v memcpy head {h}: {e}")))?;
    }

    *kv_len += 1;
    Ok(())
}

// =========================================================================
// Batch API (V2-2 prefill)
// =========================================================================

/// Batch deinterleave Q+gate for N tokens.
///
/// Input: `[n_tokens * num_heads * 2 * head_dim]` interleaved Q+gate from batch projection.
/// Outputs:
/// - `q_out`: `[n_tokens * num_heads * head_dim]` — all Q heads for all tokens
/// - `gate_out`: `[n_tokens * num_heads * head_dim]` — all gate heads for all tokens
///
/// # Cost
///
/// 1 kernel launch, 0 allocations.
pub fn raw_deinterleave_q_gate_batch(
    input: &CudaSlice<f32>,
    q_out: &mut CudaSlice<f32>,
    gate_out: &mut CudaSlice<f32>,
    num_heads: usize,
    head_dim: usize,
    n_tokens: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "deinterleave_q_gate_batch_kernel", "chimere_attn_batch_v1",
        ATTN_BATCH_KERNEL_SRC, &ATTN_BATCH_PTX,
    )?;

    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let n_tokens_i32 = n_tokens as i32;

    let grid = (n_tokens as u32) * (num_heads as u32);
    let block_size = 256u32.min(head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(q_out);
    builder.arg(gate_out);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&n_tokens_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("deinterleave_q_gate_batch launch: {e}")))?;
    Ok(())
}

/// Batch per-head RMSNorm for N tokens.
///
/// Applies RMSNorm independently to each head of each token:
///   For token t, head h: output[t,h,d] = input[t,h,d] / rms(input[t,h,:]) * weight[d]
///
/// Input/output layout: `[n_tokens * num_heads * head_dim]` flat.
/// Weight: `[head_dim]` shared across all heads and tokens.
///
/// Note: `input` and `output` may alias (in-place normalization is safe because
/// the kernel reads the full head into shared memory before writing).
///
/// # Cost
///
/// 1 kernel launch, 0 allocations.
pub fn raw_rms_norm_per_head_batch(
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    num_heads: usize,
    head_dim: usize,
    n_tokens: usize,
    eps: f32,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "rms_norm_per_head_batch_kernel", "chimere_attn_batch_v1",
        ATTN_BATCH_KERNEL_SRC, &ATTN_BATCH_PTX,
    )?;

    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let n_tokens_i32 = n_tokens as i32;

    // Block size: next power of 2 of head_dim, clamped to [32, 1024].
    // Must be power of 2 for the shared-memory reduction to work correctly.
    let block_size = (head_dim as u32).next_power_of_two().max(32).min(1024);
    let grid = (n_tokens as u32) * (num_heads as u32);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: block_size * 4,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(weight);
    builder.arg(output);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&eps);
    builder.arg(&n_tokens_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("rms_norm_per_head_batch launch: {e}")))?;
    Ok(())
}

/// Batch MRoPE for N tokens.
///
/// Applies Multi-Resolution Rotary Position Embedding to all heads of all tokens
/// in a single kernel launch. Per-token positions are read from a GPU array.
///
/// For text-only inference, sections 1 and 2 positions are hardcoded to 0 inside
/// the kernel. Only section 0 uses the per-token positions array.
///
/// Input/output layout: `[n_tokens * num_heads * head_dim]` flat.
/// Positions: `[n_tokens]` i32 array on GPU.
///
/// # Cost
///
/// 1 kernel launch, 0 allocations.
pub fn raw_mrope_apply_batch(
    input: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    positions: &CudaSlice<i32>,
    tables: &RawMRoPETables,
    num_heads: usize,
    n_tokens: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "mrope_apply_batch_kernel", "chimere_attn_batch_v1",
        ATTN_BATCH_KERNEL_SRC, &ATTN_BATCH_PTX,
    )?;

    let num_heads_i32 = num_heads as i32;
    let head_dim_i32 = tables.head_dim as i32;
    let pairs_s0 = tables.pairs[0] as i32;
    let pairs_s1 = tables.pairs[1] as i32;
    let pairs_s2 = tables.pairs[2] as i32;
    let n_rot_i32 = tables.n_rot as i32;
    let n_tokens_i32 = n_tokens as i32;

    let grid = (n_tokens as u32) * (num_heads as u32);
    let block_size = 256u32.min(tables.head_dim as u32);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = _stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(positions);
    builder.arg(&tables.cos_s0);
    builder.arg(&tables.sin_s0);
    builder.arg(&tables.cos_s1);
    builder.arg(&tables.sin_s1);
    builder.arg(&tables.cos_s2);
    builder.arg(&tables.sin_s2);
    builder.arg(&num_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&pairs_s0);
    builder.arg(&pairs_s1);
    builder.arg(&pairs_s2);
    builder.arg(&n_rot_i32);
    builder.arg(&n_tokens_i32);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("mrope_apply_batch launch: {e}")))?;
    Ok(())
}

// =========================================================================
// Flash Attention Causal Prefill kernel
// =========================================================================

/// CUDA source for the causal Flash Attention prefill kernel.
///
/// Implements Flash Attention 2 style tiled attention with:
/// - Causal mask: token t can only attend to positions 0..t (inclusive)
/// - GQA: multiple Q heads share one KV head (group_size = num_heads / num_kv_heads)
/// - Online softmax: no materialization of the N x N attention matrix
/// - F32 accumulation throughout for numerical stability
///
/// ## Layout
///
/// - Q: `[N, num_heads, head_dim]` — all query tokens, post-RoPE
/// - K: `[N, num_kv_heads, head_dim]` — all key tokens, post-RoPE
/// - V: `[N, num_kv_heads, head_dim]` — all value tokens
/// - O: `[N, num_heads, head_dim]` — output buffer
///
/// ## Grid/Block
///
/// - Grid: `(num_heads, ceil(N / BR), 1)`
///   - blockIdx.x = Q head index
///   - blockIdx.y = Q tile index (which group of BR query rows)
/// - Block: `(NUM_THREADS, 1, 1)` — threads cooperate on dot products + accumulation
///
/// ## Algorithm (Flash Attention 2)
///
/// For each Q tile (BR rows of Q):
///   Initialize O_acc[BR][head_dim] = 0, m[BR] = -inf, l[BR] = 0
///   For each KV tile (BC columns of K,V):
///     1. Load K tile into shared memory
///     2. Compute S[BR][BC] = Q_tile @ K_tile^T * scale
///     3. Apply causal mask: S[i][j] = -inf where col_j > row_i
///     4. Online softmax update: m_new, l_new, rescale O_acc
///     5. Load V tile from global memory, accumulate O_acc += P @ V_tile
///   Finalize: O = O_acc / l (normalize by softmax denominator)
///
/// ## Shared memory
///
/// - K tile: `[BC, head_dim]` floats
/// - S tile: `[BR, BC]` floats (attention scores for current tiles)
/// - Total: `(BC * head_dim + BR * BC) * 4` bytes
///   For BC=32, BR=16, head_dim=256: (32*256 + 16*32)*4 = 34,816 bytes
const FLASH_ATTN_PREFILL_KERNEL_SRC: &str = r#"
// Flash Attention 2 causal prefill kernel for chimere-deltanet.
//
// Processes ALL N query tokens against the causal-masked KV in one kernel launch,
// replacing N separate per-token attention calls.
//
// BR = number of Q rows per tile (per block along gridDim.y)
// BC = number of KV columns per tile (inner loop)
// Both are compile-time constants for shared memory sizing.

#define BR 16
#define BC 32
#define NUM_THREADS 128

// Each thread handles ELEMS_PER_THREAD elements of head_dim for dot products.
// head_dim / NUM_THREADS = 256 / 128 = 2
#define ELEMS_PER_THREAD 2

extern "C" __global__ void flash_attn_causal_prefill_kernel(
    const float* __restrict__ Q,       // [N, num_heads, head_dim]
    const float* __restrict__ K,       // [N, num_kv_heads, head_dim]
    const float* __restrict__ V,       // [N, num_kv_heads, head_dim]
    float*       __restrict__ O,       // [N, num_heads, head_dim]
    int N,                             // total number of tokens
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    // Block identity
    const int q_head = blockIdx.x;                              // Q head index
    const int q_tile_idx = blockIdx.y;                          // which BR-row group
    const int kv_head = q_head * num_kv_heads / num_heads;      // GQA mapping
    const int tid = threadIdx.x;                                // [0, NUM_THREADS)

    // Row range for this Q tile
    const int q_row_start = q_tile_idx * BR;
    const int q_row_end = q_row_start + BR;
    if (q_row_start >= N) return;  // tile fully out of range
    const int actual_br = (q_row_end <= N) ? BR : (N - q_row_start);

    // Shared memory layout:
    //   float k_tile[BC * head_dim]   -- K tile for current KV block
    //   float s_tile[BR * BC]         -- attention scores S = Q @ K^T * scale
    extern __shared__ char smem[];
    float* k_tile = (float*)smem;
    float* s_tile = (float*)(smem + BC * head_dim * sizeof(float));

    // Per-thread accumulators for output: each thread owns ELEMS_PER_THREAD
    // elements of head_dim for each of the BR query rows.
    // O_acc[br][ept] where br in [0, BR), ept in [0, ELEMS_PER_THREAD)
    float o_acc[BR][ELEMS_PER_THREAD];
    float m_i[BR];   // running max for online softmax
    float l_i[BR];   // running sum of exp for online softmax

    // Initialize accumulators
    for (int br = 0; br < BR; br++) {
        m_i[br] = -1e30f;
        l_i[br] = 0.0f;
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            o_acc[br][e] = 0.0f;
        }
    }

    // Pre-load Q values for this tile into registers.
    // Each thread loads ELEMS_PER_THREAD elements for each of BR rows.
    float q_reg[BR][ELEMS_PER_THREAD];
    for (int br = 0; br < BR; br++) {
        int global_row = q_row_start + br;
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int d = tid * ELEMS_PER_THREAD + e;
            if (global_row < N && d < head_dim) {
                q_reg[br][e] = Q[(long long)global_row * num_heads * head_dim + q_head * head_dim + d];
            } else {
                q_reg[br][e] = 0.0f;
            }
        }
    }

    // Maximum column we need to attend to (causal): last row in this Q tile
    int max_kv_col = q_row_start + actual_br - 1;  // inclusive

    // Inner loop: iterate over KV tiles
    for (int kv_start = 0; kv_start <= max_kv_col; kv_start += BC) {
        int kv_end = kv_start + BC;
        if (kv_end > max_kv_col + 1) kv_end = max_kv_col + 1;
        int actual_bc = kv_end - kv_start;

        // --- Load K tile into shared memory ---
        // K layout: [N, num_kv_heads, head_dim]
        // K tile: [actual_bc, head_dim]
        {
            int total = actual_bc * head_dim;
            for (int i = tid; i < total; i += NUM_THREADS) {
                int kv_row = kv_start + i / head_dim;
                int d = i % head_dim;
                k_tile[i] = K[(long long)kv_row * num_kv_heads * head_dim + kv_head * head_dim + d];
            }
        }
        __syncthreads();

        // --- Compute S[br][bc] = dot(Q[q_row_start+br, q_head, :], K_tile[bc, :]) * scale ---
        // Each thread computes partial dot products for its ELEMS_PER_THREAD dimensions,
        // then we reduce across threads using shared memory.
        //
        // Strategy: for each (br, bc) pair, all threads compute partial dot products,
        // then warp-reduce + cross-warp reduce to get the full score.
        // We process multiple bc values per br to amortize syncs.

        for (int br = 0; br < actual_br; br++) {
            int global_row = q_row_start + br;

            for (int bc = 0; bc < actual_bc; bc++) {
                int global_col = kv_start + bc;

                // Causal mask: skip if column > row.
                // Since columns are consecutive, once we hit a masked column,
                // all remaining columns in this tile are also masked.
                if (global_col > global_row) {
                    // Fill remaining columns with -inf.
                    // All threads see the same condition (uniform control flow),
                    // so thread 0 writes and all threads break together.
                    if (tid == 0) {
                        for (int bc2 = bc; bc2 < actual_bc; bc2++) {
                            s_tile[br * BC + bc2] = -1e30f;
                        }
                    }
                    // Ensure thread 0's -inf writes are visible before any
                    // thread proceeds to the next br iteration or the softmax.
                    __syncthreads();
                    break;  // all subsequent bc values are also masked
                }

                // Partial dot product: each thread handles its elements
                float partial = 0.0f;
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    int d = tid * ELEMS_PER_THREAD + e;
                    if (d < head_dim) {
                        partial += q_reg[br][e] * k_tile[bc * head_dim + d];
                    }
                }

                // Warp-level reduction
                for (int offset = 16; offset > 0; offset >>= 1) {
                    partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
                }

                // Cross-warp reduction via shared memory
                // Use a small section of s_tile as scratch (we'll write final score after)
                int warp_id = tid >> 5;
                int lane_id = tid & 31;
                int n_warps = (NUM_THREADS + 31) >> 5;  // 128/32 = 4

                // Temporarily use s_tile[BR*BC .. BR*BC + n_warps] as warp scratch
                // This is safe because BR*BC = 16*32 = 512 floats, and we have
                // BC*head_dim + BR*BC = 32*256 + 512 = 8704 floats total shared mem
                float* warp_scratch = s_tile + BR * BC;

                if (lane_id == 0) {
                    warp_scratch[warp_id] = partial;
                }
                __syncthreads();

                if (tid == 0) {
                    float dot = 0.0f;
                    for (int w = 0; w < n_warps; w++) {
                        dot += warp_scratch[w];
                    }
                    s_tile[br * BC + bc] = dot * scale;
                }
                __syncthreads();
            }
        }

        // Ensure all s_tile writes (including causal mask -inf) are visible
        // to all threads before the softmax update reads them.
        __syncthreads();

        // --- Online softmax update ---
        // For each Q row br, compute:
        //   m_new[br] = max(m_i[br], max_j(S[br][j]))
        //   l_new[br] = exp(m_i[br] - m_new[br]) * l_i[br] + sum_j(exp(S[br][j] - m_new[br]))
        //   O_acc[br] = exp(m_i[br] - m_new[br]) * O_acc[br] + sum_j(exp(S[br][j] - m_new[br]) * V[j])

        for (int br = 0; br < actual_br; br++) {
            // Find max score for this row in the current tile
            // Thread 0 reads all scores (they're already computed)
            float row_max = -1e30f;
            // All threads read the scores (they're in shared memory, broadcast is fine)
            for (int bc = 0; bc < actual_bc; bc++) {
                float s = s_tile[br * BC + bc];
                if (s > row_max) row_max = s;
            }

            float m_new = (row_max > m_i[br]) ? row_max : m_i[br];
            float correction = expf(m_i[br] - m_new);

            // Rescale existing accumulators
            for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                o_acc[br][e] *= correction;
            }
            float l_new = l_i[br] * correction;

            // Accumulate P * V for this tile
            // P[br][bc] = exp(S[br][bc] - m_new)
            for (int bc = 0; bc < actual_bc; bc++) {
                float p = expf(s_tile[br * BC + bc] - m_new);
                l_new += p;

                // Each thread accumulates p * V[kv_start+bc, kv_head, d]
                // for its ELEMS_PER_THREAD dimensions
                int global_kv_row = kv_start + bc;
                for (int e = 0; e < ELEMS_PER_THREAD; e++) {
                    int d = tid * ELEMS_PER_THREAD + e;
                    if (d < head_dim) {
                        float v_val = V[(long long)global_kv_row * num_kv_heads * head_dim + kv_head * head_dim + d];
                        o_acc[br][e] += p * v_val;
                    }
                }
            }

            m_i[br] = m_new;
            l_i[br] = l_new;
        }

        __syncthreads();
    }

    // --- Finalize: normalize by l_i and write to output ---
    for (int br = 0; br < actual_br; br++) {
        int global_row = q_row_start + br;
        if (global_row >= N) break;

        float inv_l = (l_i[br] > 0.0f) ? (1.0f / l_i[br]) : 0.0f;
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int d = tid * ELEMS_PER_THREAD + e;
            if (d < head_dim) {
                O[(long long)global_row * num_heads * head_dim + q_head * head_dim + d] = o_acc[br][e] * inv_l;
            }
        }
    }
}
"#;

static FLASH_ATTN_PREFILL_PTX: OnceLock<String> = OnceLock::new();

/// Flash Attention causal prefill: processes all N query tokens against the
/// causal-masked KV in a single kernel launch.
///
/// Replaces the Phase B per-token attention loop in `forward_attn_prefill_cudarc`
/// with a single Flash Attention 2 style kernel.
///
/// # Layout
///
/// - `q_batch`: `[N * num_heads * head_dim]` — all Q tokens, post-RoPE.
///   Layout: `q_batch[t * num_heads * head_dim + h * head_dim + d]`
/// - `k_batch`: `[N * num_kv_heads * head_dim]` — all K tokens, post-RoPE.
/// - `v_batch`: `[N * num_kv_heads * head_dim]` — all V tokens.
/// - `o_batch`: `[N * num_heads * head_dim]` — pre-allocated output buffer.
///
/// # Causal mask
///
/// Token `t` can attend to positions `0..t` (inclusive). This is the standard
/// causal (autoregressive) mask where no future tokens are visible.
///
/// # GQA
///
/// `num_heads` Q heads share `num_kv_heads` KV heads. The GQA ratio is
/// `num_heads / num_kv_heads` (8:1 for Qwen3.5-35B).
///
/// # Cost
///
/// 1 kernel launch (grid = num_heads * ceil(N/BR)), 0 allocations.
pub fn flash_attn_causal_prefill(
    q_batch: &CudaSlice<f32>,
    k_batch: &CudaSlice<f32>,
    v_batch: &CudaSlice<f32>,
    o_batch: &mut CudaSlice<f32>,
    n_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(
        num_heads > 0 && num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "num_heads ({num_heads}) must be a positive multiple of num_kv_heads ({num_kv_heads})"
    );
    if n_tokens == 0 {
        return Ok(());
    }

    let (func, stream) = super::nvrtc_compile::get_or_load_func(
        dev,
        "flash_attn_causal_prefill_kernel",
        "chimere_flash_attn_prefill_v1",
        FLASH_ATTN_PREFILL_KERNEL_SRC,
        &FLASH_ATTN_PREFILL_PTX,
    )?;

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let n_i32 = n_tokens as i32;
    let num_heads_i32 = num_heads as i32;
    let num_kv_heads_i32 = num_kv_heads as i32;
    let head_dim_i32 = head_dim as i32;

    // Tile sizes (must match CUDA #defines)
    const BR: usize = 16;
    const BC: usize = 32;
    const NUM_THREADS: u32 = 128;

    // Grid: (num_heads, ceil(N / BR))
    let num_q_tiles = (n_tokens + BR - 1) / BR;

    // Shared memory: K tile [BC * head_dim] + S tile [BR * BC] + warp scratch [4]
    let smem_bytes = ((BC * head_dim + BR * BC + 4) * std::mem::size_of::<f32>()) as u32;

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, num_q_tiles as u32, 1),
        block_dim: (NUM_THREADS, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(q_batch);
    builder.arg(k_batch);
    builder.arg(v_batch);
    builder.arg(o_batch);
    builder.arg(&n_i32);
    builder.arg(&num_heads_i32);
    builder.arg(&num_kv_heads_i32);
    builder.arg(&head_dim_i32);
    builder.arg(&scale);

    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("flash_attn_causal_prefill launch: {e}")))?;

    Ok(())
}

/// Flash Attention causal prefill with KV cache stride support.
///
/// Like `flash_attn_causal_prefill`, but reads K,V from a pre-allocated KV cache
/// with a given stride per KV head, and writes the new K,V tokens into the cache
/// before running attention. The Q tokens are from the batch buffers.
///
/// This variant is designed for integration with `forward_attn_prefill_cudarc`:
/// - Phase A (batch ops) produces q_roped_batch, k_roped_batch, v_batch
/// - This function first populates the KV cache from k_roped_batch/v_batch,
///   then runs causal attention reading Q from q_roped_batch and K,V from the
///   fresh batch buffers directly (no cache needed for prefill, since ALL tokens
///   are available in the batch).
///
/// # Layout
///
/// - `q_batch`: `[N * num_heads * head_dim]` — post-RoPE Q, token-major
/// - `k_batch`: `[N * num_kv_heads * head_dim]` — post-RoPE K, token-major
/// - `v_batch`: `[N * num_kv_heads * head_dim]` — V, token-major
/// - `o_batch`: `[N * num_heads * head_dim]` — output buffer
pub fn flash_attn_causal_prefill_from_batch(
    q_roped_batch: &CudaSlice<f32>,
    k_roped_batch: &CudaSlice<f32>,
    v_batch: &CudaSlice<f32>,
    o_batch: &mut CudaSlice<f32>,
    n_tokens: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // The batch buffers already have the correct token-major layout:
    //   [t * num_heads * head_dim + h * head_dim + d]
    // which is exactly what the kernel expects.
    flash_attn_causal_prefill(
        q_roped_batch,
        k_roped_batch,
        v_batch,
        o_batch,
        n_tokens,
        num_heads,
        num_kv_heads,
        head_dim,
        dev,
    )
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Helper: get a CudaDevice for testing, or None if no GPU available.
    fn cuda_dev() -> Option<CudaDevice> {
        match Device::cuda_if_available(0) {
            Ok(Device::Cuda(d)) => Some(d),
            _ => None,
        }
    }

    /// Upload host f32 data to GPU.
    fn htod(dev: &CudaDevice, data: &[f32]) -> CudaSlice<f32> {
        let stream = dev.cuda_stream();
        let mut buf: CudaSlice<f32> = stream
            .alloc_zeros(data.len())
            .expect("alloc");
        dev.memcpy_htod(data, &mut buf).expect("htod");
        buf
    }

    /// Download GPU f32 data to host.
    fn dtoh(slice: &CudaSlice<f32>) -> Vec<f32> {
        slice.stream().clone().clone_dtoh(slice).expect("dtoh")
    }

    #[test]
    fn test_deinterleave_q_gate() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_deinterleave_q_gate");
                return;
            }
        };

        let num_heads = 4usize;
        let head_dim = 8usize;
        // Input: [Q_h0(8), gate_h0(8), Q_h1(8), gate_h1(8), ...]
        let mut input_data = vec![0.0f32; num_heads * 2 * head_dim];
        for h in 0..num_heads {
            for d in 0..head_dim {
                input_data[h * 2 * head_dim + d] = (h * 100 + d) as f32;                // Q
                input_data[h * 2 * head_dim + head_dim + d] = (h * 100 + d + 50) as f32; // gate
            }
        }

        let input = htod(&dev, &input_data);
        let stream = dev.cuda_stream();
        let mut q_out: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .expect("alloc q_out");
        let mut gate_out: CudaSlice<f32> = stream.alloc_zeros(num_heads * head_dim)
            .expect("alloc gate_out");

        raw_deinterleave_q_gate(&input, &mut q_out, &mut gate_out, num_heads, head_dim, &dev)
            .expect("deinterleave_q_gate");

        let q_host = dtoh(&q_out);
        let gate_host = dtoh(&gate_out);

        for h in 0..num_heads {
            for d in 0..head_dim {
                let expected_q = (h * 100 + d) as f32;
                let expected_g = (h * 100 + d + 50) as f32;
                assert_eq!(q_host[h * head_dim + d], expected_q,
                    "Q mismatch at head={h}, dim={d}");
                assert_eq!(gate_host[h * head_dim + d], expected_g,
                    "gate mismatch at head={h}, dim={d}");
            }
        }
    }

    #[test]
    fn test_sigmoid_gate_mul() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_sigmoid_gate_mul");
                return;
            }
        };

        let n = 16;
        let attn_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let gate_data: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.5).collect();

        let attn = htod(&dev, &attn_data);
        let gate = htod(&dev, &gate_data);
        let stream = dev.cuda_stream();
        let mut output: CudaSlice<f32> = stream.alloc_zeros(n).expect("alloc output");

        raw_sigmoid_gate_mul(&attn, &gate, &mut output, n, &dev)
            .expect("sigmoid_gate_mul");

        let result = dtoh(&output);

        for i in 0..n {
            let expected = attn_data[i] * (1.0 / (1.0 + (-gate_data[i]).exp()));
            assert!(
                (result[i] - expected).abs() < 1e-5,
                "sigmoid_gate_mul mismatch at {i}: got {}, expected {}",
                result[i], expected
            );
        }
    }

    /// Reference CPU implementation of causal attention for testing.
    ///
    /// For each token t, head h:
    ///   kv_h = h / group_size
    ///   scores[j] = dot(Q[t,h,:], K[j,kv_h,:]) * scale   for j in 0..t
    ///   weights = softmax(scores)
    ///   output[t,h,d] = sum(weights[j] * V[j,kv_h,d])
    fn reference_causal_attn(
        q: &[f32],   // [N * num_heads * head_dim]
        k: &[f32],   // [N * num_kv_heads * head_dim]
        v: &[f32],   // [N * num_kv_heads * head_dim]
        n_tokens: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let group_size = num_heads / num_kv_heads;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; n_tokens * num_heads * head_dim];

        for t in 0..n_tokens {
            for h in 0..num_heads {
                let kv_h = h / group_size;
                let seq_len = t + 1; // causal: can attend to 0..t inclusive

                // Compute scores
                let mut scores = vec![0.0f32; seq_len];
                for j in 0..seq_len {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        let q_val = q[t * num_heads * head_dim + h * head_dim + d];
                        let k_val = k[j * num_kv_heads * head_dim + kv_h * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[j] = dot * scale;
                }

                // Softmax
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

                // Weighted sum over V
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for j in 0..seq_len {
                        let v_val = v[j * num_kv_heads * head_dim + kv_h * head_dim + d];
                        acc += scores[j] * v_val;
                    }
                    output[t * num_heads * head_dim + h * head_dim + d] = acc;
                }
            }
        }
        output
    }

    #[test]
    fn test_flash_attn_causal_prefill_small() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_flash_attn_causal_prefill_small");
                return;
            }
        };

        // Small test: 5 tokens, 4 heads, 2 KV heads, head_dim=8
        // (small enough to verify by hand if needed)
        let n_tokens = 5usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;

        let q_data: Vec<f32> = (0..n_tokens * num_heads * head_dim)
            .map(|i| ((i as f32) * 0.0137 - 1.0).sin() * 0.3)
            .collect();
        let k_data: Vec<f32> = (0..n_tokens * num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.0071 + 0.5).cos() * 0.2)
            .collect();
        let v_data: Vec<f32> = (0..n_tokens * num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.0043 - 0.3).sin() * 0.4)
            .collect();

        let ref_output = reference_causal_attn(
            &q_data, &k_data, &v_data,
            n_tokens, num_heads, num_kv_heads, head_dim,
        );

        let q_gpu = htod(&dev, &q_data);
        let k_gpu = htod(&dev, &k_data);
        let v_gpu = htod(&dev, &v_data);
        let stream = dev.cuda_stream();
        let mut o_gpu: CudaSlice<f32> = stream
            .alloc_zeros(n_tokens * num_heads * head_dim)
            .expect("alloc output");

        flash_attn_causal_prefill(
            &q_gpu, &k_gpu, &v_gpu, &mut o_gpu,
            n_tokens, num_heads, num_kv_heads, head_dim, &dev,
        ).expect("flash_attn_causal_prefill");

        let gpu_output = dtoh(&o_gpu);

        let mut max_err = 0.0f32;
        let mut max_err_idx = 0usize;
        for i in 0..ref_output.len() {
            let err = (gpu_output[i] - ref_output[i]).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
        }
        eprintln!(
            "[FLASH_PREFILL] small test: n={n_tokens}, heads={num_heads}, kv_heads={num_kv_heads}, \
             head_dim={head_dim}, max_err={max_err:.2e} (idx={max_err_idx})"
        );
        assert!(
            max_err < 1e-4,
            "Flash prefill small test mismatch: max_err={max_err:.6e} at idx={max_err_idx}"
        );
        eprintln!("[FLASH_PREFILL] PASS: small test");
    }

    #[test]
    fn test_flash_attn_causal_prefill_production() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_flash_attn_causal_prefill_production");
                return;
            }
        };

        // Production-like test: 37 tokens, 16 heads, 2 KV heads, head_dim=256
        let n_tokens = 37usize;
        let num_heads = 16usize;
        let num_kv_heads = 2usize;
        let head_dim = 256usize;

        let q_data: Vec<f32> = (0..n_tokens * num_heads * head_dim)
            .map(|i| ((i as f32) * 0.00137 - 1.0).sin() * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..n_tokens * num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.00071 + 0.5).cos() * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..n_tokens * num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.00043 - 0.3).sin() * 0.1)
            .collect();

        let ref_output = reference_causal_attn(
            &q_data, &k_data, &v_data,
            n_tokens, num_heads, num_kv_heads, head_dim,
        );

        let q_gpu = htod(&dev, &q_data);
        let k_gpu = htod(&dev, &k_data);
        let v_gpu = htod(&dev, &v_data);
        let stream = dev.cuda_stream();
        let mut o_gpu: CudaSlice<f32> = stream
            .alloc_zeros(n_tokens * num_heads * head_dim)
            .expect("alloc output");

        flash_attn_causal_prefill(
            &q_gpu, &k_gpu, &v_gpu, &mut o_gpu,
            n_tokens, num_heads, num_kv_heads, head_dim, &dev,
        ).expect("flash_attn_causal_prefill");

        let gpu_output = dtoh(&o_gpu);

        let mut max_err = 0.0f32;
        let mut max_err_idx = 0usize;
        for i in 0..ref_output.len() {
            let err = (gpu_output[i] - ref_output[i]).abs();
            if err > max_err {
                max_err = err;
                max_err_idx = i;
            }
        }
        eprintln!(
            "[FLASH_PREFILL] production test: n={n_tokens}, heads={num_heads}, \
             kv_heads={num_kv_heads}, head_dim={head_dim}, max_err={max_err:.2e} (idx={max_err_idx})"
        );
        assert!(
            max_err < 1e-3,
            "Flash prefill production test mismatch: max_err={max_err:.6e} at idx={max_err_idx}"
        );
        eprintln!("[FLASH_PREFILL] PASS: production test");
    }

    #[test]
    fn test_flash_attn_causal_prefill_single_token() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_flash_attn_causal_prefill_single_token");
                return;
            }
        };

        // Edge case: N=1 (first token, softmax is trivially 1.0, output = V)
        let n_tokens = 1usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 64usize;

        let q_data: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i as f32) * 0.013).sin())
            .collect();
        let k_data: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.007).cos())
            .collect();
        let v_data: Vec<f32> = (0..num_kv_heads * head_dim)
            .map(|i| ((i as f32) * 0.011).sin())
            .collect();

        // For N=1, output should be V[0, kv_h, :] for each Q head h
        let group_size = num_heads / num_kv_heads;
        let mut ref_output = vec![0.0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;
            for d in 0..head_dim {
                ref_output[h * head_dim + d] = v_data[kv_h * head_dim + d];
            }
        }

        let q_gpu = htod(&dev, &q_data);
        let k_gpu = htod(&dev, &k_data);
        let v_gpu = htod(&dev, &v_data);
        let stream = dev.cuda_stream();
        let mut o_gpu: CudaSlice<f32> = stream
            .alloc_zeros(num_heads * head_dim)
            .expect("alloc output");

        flash_attn_causal_prefill(
            &q_gpu, &k_gpu, &v_gpu, &mut o_gpu,
            n_tokens, num_heads, num_kv_heads, head_dim, &dev,
        ).expect("flash_attn_causal_prefill");

        let gpu_output = dtoh(&o_gpu);
        let max_err = gpu_output.iter().zip(ref_output.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[FLASH_PREFILL] single token max_err={max_err:.2e}");
        assert!(max_err < 1e-5, "single token mismatch: max_err={max_err:.6e}");
        eprintln!("[FLASH_PREFILL] PASS: single token edge case");
    }

    /// Integration test: compare flash attention prefill output against
    /// per-token GQA attention with KV cache (production dimensions).
    ///
    /// This tests the EXACT same code path as the integrated prefill:
    /// - Flash: runs flash_attn_causal_prefill on all N tokens at once
    /// - GQA: for each token, appends K/V to a KV cache (head-major layout),
    ///   then runs raw_gqa_attention_strided.
    ///
    /// Both should produce identical attention outputs for each token.
    #[test]
    fn test_flash_vs_gqa_integration() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_flash_vs_gqa_integration");
                return;
            }
        };

        let n_tokens = 20usize;
        let num_heads = 16usize;
        let num_kv_heads = 2usize;
        let head_dim = 256usize;
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let kv_cap = 128usize; // larger than n_tokens to test strided access

        let q_heads_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;

        // Generate deterministic test data
        let q_data: Vec<f32> = (0..n_tokens * q_heads_size)
            .map(|i| ((i as f32) * 0.00137 - 1.0).sin() * 0.1)
            .collect();
        let k_data: Vec<f32> = (0..n_tokens * kv_size)
            .map(|i| ((i as f32) * 0.00071 + 0.5).cos() * 0.1)
            .collect();
        let v_data: Vec<f32> = (0..n_tokens * kv_size)
            .map(|i| ((i as f32) * 0.00043 - 0.3).sin() * 0.1)
            .collect();

        // Upload batch buffers
        let q_gpu = htod(&dev, &q_data);
        let k_gpu = htod(&dev, &k_data);
        let v_gpu = htod(&dev, &v_data);
        let stream = dev.cuda_stream();

        // --- Flash path: single kernel launch ---
        let mut flash_output: CudaSlice<f32> = stream
            .alloc_zeros(n_tokens * q_heads_size)
            .expect("alloc flash_output");

        flash_attn_causal_prefill(
            &q_gpu, &k_gpu, &v_gpu, &mut flash_output,
            n_tokens, num_heads, num_kv_heads, head_dim, &dev,
        ).expect("flash_attn_causal_prefill");

        let flash_host = dtoh(&flash_output);

        // --- GQA path: per-token KV cache append + attention ---
        // Allocate KV cache: [num_kv_heads * kv_cap * head_dim]
        let mut kv_k: CudaSlice<f32> = stream
            .alloc_zeros(num_kv_heads * kv_cap * head_dim)
            .expect("alloc kv_k");
        let mut kv_v: CudaSlice<f32> = stream
            .alloc_zeros(num_kv_heads * kv_cap * head_dim)
            .expect("alloc kv_v");

        // Single-token scratch buffers
        let mut q_scratch: CudaSlice<f32> = stream
            .alloc_zeros(q_heads_size)
            .expect("alloc q_scratch");
        let mut attn_out: CudaSlice<f32> = stream
            .alloc_zeros(q_heads_size)
            .expect("alloc attn_out");

        let mut gqa_host = vec![0.0f32; n_tokens * q_heads_size];

        for t in 0..n_tokens {
            let position = t;

            // Append K and V to cache at position `t`
            for h in 0..num_kv_heads {
                let src_off = t * kv_size + h * head_dim;
                let dst_off = h * kv_cap * head_dim + position * head_dim;

                let k_src = k_gpu.slice(src_off..src_off + head_dim);
                let mut k_dst = kv_k.slice_mut(dst_off..dst_off + head_dim);
                dev.memcpy_dtod(&k_src, &mut k_dst).expect("kv_k append");

                let v_src = v_gpu.slice(src_off..src_off + head_dim);
                let mut v_dst = kv_v.slice_mut(dst_off..dst_off + head_dim);
                dev.memcpy_dtod(&v_src, &mut v_dst).expect("kv_v append");
            }

            let seq_len = position + 1;

            // Copy Q for this token
            let q_src = q_gpu.slice(t * q_heads_size..(t + 1) * q_heads_size);
            dev.memcpy_dtod(&q_src, &mut q_scratch).expect("q copy");

            // GQA attention
            crate::kernels::gqa_attention::raw_gqa_attention_strided(
                &q_scratch.slice(..),
                &kv_k.slice(..),
                &kv_v.slice(..),
                &mut attn_out,
                num_heads,
                num_kv_heads,
                seq_len,
                head_dim,
                scale,
                kv_cap,
                &dev,
            ).expect("gqa_attention");

            // Download GQA output for this token
            let gqa_token = dtoh(&attn_out);
            gqa_host[t * q_heads_size..(t + 1) * q_heads_size]
                .copy_from_slice(&gqa_token);
        }

        // --- Compare flash vs GQA per-token ---
        let mut overall_max_err = 0.0f32;
        let mut worst_token = 0usize;
        for t in 0..n_tokens {
            let offset = t * q_heads_size;
            let mut token_max_err = 0.0f32;
            let mut token_max_idx = 0usize;
            for i in 0..q_heads_size {
                let err = (flash_host[offset + i] - gqa_host[offset + i]).abs();
                if err > token_max_err {
                    token_max_err = err;
                    token_max_idx = i;
                }
            }
            if t < 5 || token_max_err > 1e-3 {
                let head = token_max_idx / head_dim;
                let dim = token_max_idx % head_dim;
                eprintln!(
                    "[FLASH_VS_GQA] t={t}: max_err={token_max_err:.6e} at head={head} dim={dim} \
                     (flash={:.6} gqa={:.6})",
                    flash_host[offset + token_max_idx],
                    gqa_host[offset + token_max_idx],
                );
            }
            if token_max_err > overall_max_err {
                overall_max_err = token_max_err;
                worst_token = t;
            }
        }

        eprintln!(
            "[FLASH_VS_GQA] Overall: max_err={overall_max_err:.6e} at token={worst_token} \
             (n={n_tokens}, heads={num_heads}, kv_heads={num_kv_heads}, \
             head_dim={head_dim}, kv_cap={kv_cap})"
        );
        assert!(
            overall_max_err < 1e-3,
            "Flash vs GQA divergence: max_err={overall_max_err:.6e} at token={worst_token}"
        );
        eprintln!("[FLASH_VS_GQA] PASS: flash matches per-token GQA with KV cache");
    }

    #[test]
    fn test_rms_norm_per_head() {
        let dev = match cuda_dev() {
            Some(d) => d,
            None => {
                eprintln!("No CUDA device, skipping test_rms_norm_per_head");
                return;
            }
        };

        let num_heads = 2;
        let head_dim = 4;
        let eps = 1e-6f32;

        // Simple test: all ones input, all ones weight
        let input_data = vec![1.0f32; num_heads * head_dim];
        let weight_data = vec![1.0f32; head_dim];

        let input = htod(&dev, &input_data);
        let weight = htod(&dev, &weight_data);
        let stream = dev.cuda_stream();
        let mut output: CudaSlice<f32> = stream
            .alloc_zeros(num_heads * head_dim)
            .expect("alloc");

        raw_rms_norm_per_head(&input, &weight, &mut output, num_heads, head_dim, eps, &dev)
            .expect("rms_norm_per_head");

        let result = dtoh(&output);

        // RMS of [1,1,1,1] = sqrt(mean([1,1,1,1]) + eps) = sqrt(1 + eps) ~ 1.0
        // So output should be ~ 1.0 / 1.0 * 1.0 = 1.0 for each element
        for i in 0..result.len() {
            assert!(
                (result[i] - 1.0).abs() < 1e-3,
                "rms_norm_per_head mismatch at {i}: got {}", result[i]
            );
        }
    }
}
