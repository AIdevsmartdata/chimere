//! Fused gate+up+SiLU MMVQ kernel for IQ3_S experts.
//!
//! Inspired by ik_llama's `iqk_fused_mul_mat_vec_q_kernel` which uses dual
//! accumulators (`tmp_u` / `tmp_g`) to compute both gate and up projections
//! in a single kernel launch, then applies `silu(gate) * up` in-register.
//!
//! This kernel replaces 3 separate launches in `moe_ffn_raw`:
//!   1. `gemv_iq3s_q8_batched` for gate projections
//!   2. `gemv_iq3s_q8_batched` for up projections
//!   3. `raw_silu_mul_batched` for silu(gate) * up
//!
//! With a single launch that does everything:
//!   - Grid: `(expert_ffn, top_k, 1)` = `(512, 8, 1)` = 4096 blocks
//!   - Block: `(128, 1, 1)` = 4 warps
//!   - Each block computes ONE output element for ONE expert:
//!     `output[k * expert_ffn + row] = silu(gate_dot) * up_dot`
//!
//! The Q8_1 input vector is read ONCE per sub-group and used for both gate
//! and up dot products — halving global memory bandwidth for the input.

use candle_core::cuda_backend::cudarc::driver::{
    CudaSlice, CudaView, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CUDA kernel source
// ---------------------------------------------------------------------------
//
// The kernel includes the same IQ3_S helpers (f16_to_f32, iq3s_grid,
// block_q8_1_gemv layout, dot_iq3s_q8_ggml) as the batched GEMV kernel,
// then the fused gate+up+silu kernel itself.

const FUSED_GATE_UP_KERNEL_SRC: &str = r#"
// =====================================================================
// Common helpers (same as iq3s_gemv.rs kernel source)
// =====================================================================

// f16 -> f32 via hardware cvt.f32.f16 (single instruction, no branches)
__device__ __forceinline__ float f16_to_f32_gemv(unsigned short h) {
    float f;
    asm("{ .reg .b16 tmp; mov.b16 tmp, %1; cvt.f32.f16 %0, tmp; }"
        : "=f"(f) : "h"(h));
    return f;
}

// Q8_1 block layout: 36 bytes per 32 elements
// [d_bits: u16, s_bits: u16, qs: i8 x 32]
struct block_q8_1_gemv {
    unsigned short d_bits;
    unsigned short s_bits;
    signed char    qs[32];
};

// IQ3_S grid constant — 512 entries (identical to iq3s_gemv.rs)
__constant__ unsigned int iq3s_grid_gemv[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// =====================================================================
// IQ3_S dot product against Q8_1 (ggml-compatible, dp4a)
// =====================================================================
//
// This is the same dot_iq3s_q8_ggml from iq3s_gemv.rs, renamed to avoid
// symbol conflicts when both modules are linked into the same cubin.

__device__ __forceinline__ int get_int_b2_fgu(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}

__device__ __forceinline__ float dot_iq3s_q8_fgu(
    const unsigned char* __restrict__ bq3,   // IQ3_S block pointer (110 bytes)
    const unsigned char* __restrict__ bq8,   // Q8_1 block pointer (36 bytes)
    int iqs                                   // sub-group index (0,2,...,14)
) {
    float d_iq3 = f16_to_f32_gemv(*(const unsigned short*)bq3);
    float d_q8  = f16_to_f32_gemv(*(const unsigned short*)bq8);

    const unsigned char* qs = bq3 + 2;
    const int qs0 = get_int_b2_fgu(qs, iqs + 0);
    const int qs1 = get_int_b2_fgu(qs, iqs + 1);
    const unsigned char* qs_bytes    = (const unsigned char*)&qs0;
    const unsigned char* qs_bytes_hi = (const unsigned char*)&qs1;

    const int qh = (int)bq3[66 + iqs/2];

    const int signs32 = get_int_b2_fgu(bq3 + 74, iqs/2);
    const unsigned char* signs = (const unsigned char*)&signs32;

    const int* q8_qs = (const int*)(bq8 + 4);

    int sumi = 0;

    const unsigned char qs_flat[8] = {
        qs_bytes[0], qs_bytes[1], qs_bytes[2], qs_bytes[3],
        qs_bytes_hi[0], qs_bytes_hi[1], qs_bytes_hi[2], qs_bytes_hi[3]
    };

    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid0 = iq3s_grid_gemv[qs_flat[l0]     | ((qh << (8 - l0)) & 0x100)];
        const int grid1 = iq3s_grid_gemv[qs_flat[l0 + 1] | ((qh << (7 - l0)) & 0x100)];

        const int sb = signs[l0/2];
        const int signs0 = __vcmpne4(((sb & 0x03) << 7) | ((sb & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(((sb & 0x30) << 3) | ((sb & 0xC0) << 17), 0);

        sumi = __dp4a((int)__vsub4(grid0 ^ signs0, signs0), q8_qs[l0 + 0], sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ signs1, signs1), q8_qs[l0 + 1], sumi);
    }

    const int scale_byte = bq3[106 + iqs/4];
    sumi *= 1 + 2 * ((scale_byte >> ((iqs << 1) & 0x04)) & 0x0F);

    return d_iq3 * d_q8 * (float)sumi;
}

// =====================================================================
// Fused gate+up+SiLU kernel
// =====================================================================
//
// Grid:  (expert_ffn, top_k, 1) = (512, 8, 1) = 4096 blocks
// Block: (128, 1, 1) = 4 warps
// Smem:  32 bytes (8 floats: 4 for gate warp sums, 4 for up warp sums)
//
// Each block computes ONE output element for ONE expert:
//   output[k * expert_ffn + row] = silu(gate_dot) * up_dot
//
// Where:
//   gate_dot = sum_j(gate_weights[expert_id][row][j] * q8_input[j])
//   up_dot   = sum_j(up_weights[expert_id][row][j] * q8_input[j])
//
// The Q8_1 input is read ONCE per sub-group and used for both dot products.

extern "C" __global__ __launch_bounds__(128, 2)
void fused_gate_up_silu_iq3s(
    const unsigned char* __restrict__ gate_weights,  // all experts gate [stacked]
    const unsigned char* __restrict__ up_weights,    // all experts up   [stacked]
    float*               __restrict__ output,        // [top_k * expert_ffn]
    const unsigned char* __restrict__ q8_input,      // Q8_1 quantized input [cols/32 * 36]
    const int*           __restrict__ expert_ids,    // [top_k] on GPU
    int                               cols,          // input features (2048)
    int                               expert_stride  // bytes per expert in gate/up weights
) {
    const int row = blockIdx.x;        // output row 0..expert_ffn-1
    const int k   = blockIdx.y;        // which expert in top-K
    const int expert_id = expert_ids[k];
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;  // cols / 256

    // ggml-style thread mapping: qi=16, vdr=2 -> qi/vdr=8
    const int iqs = 2 * (tid & 7);
    const int NWARPS = 4;
    const int WARP_SIZE = 32;
    const int blocks_per_iter = 2 * NWARPS * WARP_SIZE / 16;  // = 16

    // Point to this expert's gate and up weights for this row
    const unsigned char* gate_row = gate_weights
        + (long long)expert_id * expert_stride
        + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* up_row = up_weights
        + (long long)expert_id * expert_stride
        + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;

    // Dual accumulators — the key optimization from ik_llama
    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    // Iterate over IQ3_S blocks using ggml-style thread mapping.
    // Each group of 8 threads processes one block; each thread handles
    // one sub-group (32 elements). Q8_1 input is read ONCE, used twice.
    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += blocks_per_iter) {
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;

        // Gate dot product
        gate_sum += dot_iq3s_q8_fgu(gate_row + kbx * IQ3S_BYTES, bq8, iqs);

        // Up dot product — SAME Q8_1 input, different weights
        up_sum += dot_iq3s_q8_fgu(up_row + kbx * IQ3S_BYTES, bq8, iqs);
    }

    // Warp-level reduction for BOTH accumulators
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_xor_sync(0xffffffff, gate_sum, offset);
        up_sum   += __shfl_xor_sync(0xffffffff, up_sum,   offset);
    }

    // Cross-warp reduction via shared memory
    extern __shared__ float smem[];
    float* gate_sums = smem;       // [4]
    float* up_sums   = smem + 4;   // [4]

    if (lane_id == 0) {
        gate_sums[warp_id] = gate_sum;
        up_sums[warp_id]   = up_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float g = gate_sums[0] + gate_sums[1] + gate_sums[2] + gate_sums[3];
        float u = up_sums[0]   + up_sums[1]   + up_sums[2]   + up_sums[3];

        // SiLU(gate) * up — inline, no separate kernel launch!
        float silu_g = g / (1.0f + expf(-g));
        output[k * gridDim.x + row] = silu_g * u;
    }
}
"#;

// ---------------------------------------------------------------------------
// PTX cache + compilation
// ---------------------------------------------------------------------------

const MODULE_NAME: &str = "chimere_fused_gate_up_v1";
const KERNEL_FUNC: &str = "fused_gate_up_silu_iq3s";

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
        FUSED_GATE_UP_KERNEL_SRC,
        &PTX_CACHE,
    )
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused gate+up+SiLU IQ3_S GEMV: single launch replaces 3 kernels.
///
/// Computes `output[k * expert_ffn + row] = silu(gate_dot) * up_dot` for all
/// `top_k` experts and all `expert_ffn` output rows in a single kernel launch.
///
/// # Arguments
///
/// - `gate_weights`: CudaView into the stacked IQ3_S gate expert weights.
/// - `up_weights`: CudaView into the stacked IQ3_S up expert weights.
/// - `q8_input`: Pre-quantized Q8_1 input (from `quantize_f32_to_q8_1_gpu`).
/// - `output`: Pre-allocated `[top_k * expert_ffn]` f32 buffer.
/// - `expert_ids`: GPU buffer of `top_k` i32 expert indices.
/// - `cols`: Input dimension (must be multiple of 256). Typically 2048.
/// - `expert_ffn`: Expert FFN hidden dim. Typically 512.
/// - `top_k`: Number of selected experts. Typically 8.
/// - `expert_stride`: Byte stride between consecutive experts in gate/up weights.
/// - `dev`: CUDA device.
pub fn fused_gate_up_silu_iq3s(
    gate_weights: &CudaView<'_, u8>,
    up_weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    cols: usize,
    expert_ffn: usize,
    top_k: usize,
    expert_stride: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % 256 == 0, "cols must be multiple of 256");
    if top_k == 0 || expert_ffn == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev)?;

    let cols_i32 = cols as i32;
    let expert_stride_i32 = expert_stride as i32;

    // Grid: (expert_ffn, top_k, 1) — one block per (row, expert) pair
    // Block: (128, 1, 1) — 4 warps
    // Shared memory: 8 floats = 32 bytes (4 gate_sums + 4 up_sums)
    let cfg = LaunchConfig {
        grid_dim: (expert_ffn as u32, top_k as u32, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 32,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(gate_weights);
    builder.arg(up_weights);
    builder.arg(output);
    builder.arg(q8_input);
    builder.arg(expert_ids);
    builder.arg(&cols_i32);
    builder.arg(&expert_stride_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_gate_up_silu_iq3s launch: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::backend::BackendDevice;
    use candle_core::Device;

    /// IQ3_S block size constants (copied from iq3s_gemv.rs).
    const IQ3S_BLOCK_BYTES: usize = 110;
    const IQ3S_BLOCK_ELEMS: usize = 256;
    const Q8_1_BLOCK_BYTES: usize = 36;
    const Q8_1_BLOCK_ELEMS: usize = 32;

    /// Create synthetic IQ3_S weight data (deterministic, not random).
    fn make_synthetic_iq3s_weights(n_rows: usize, n_cols: usize) -> Vec<u8> {
        let n_blocks_per_row = n_cols / IQ3S_BLOCK_ELEMS;
        let total_bytes = n_rows * n_blocks_per_row * IQ3S_BLOCK_BYTES;
        (0..total_bytes).map(|i| (i % 251) as u8).collect()
    }

    /// Create synthetic Q8_1 quantized input.
    fn make_synthetic_q8_input(n_elements: usize) -> Vec<u8> {
        let n_blocks = n_elements / Q8_1_BLOCK_ELEMS;
        let total_bytes = n_blocks * Q8_1_BLOCK_BYTES;
        let mut data = vec![0u8; total_bytes];
        for blk in 0..n_blocks {
            let base = blk * Q8_1_BLOCK_BYTES;
            // Set a small nonzero scale so the kernel does real work
            data[base] = 0x00;     // d_bits low byte
            data[base + 1] = 0x10; // d_bits high byte (f16 ~ small positive)
            data[base + 2] = 0x00; // s_bits
            data[base + 3] = 0x00;
            for j in 0..32 {
                data[base + 4 + j] = ((j % 5) as i8).to_le_bytes()[0];
            }
        }
        data
    }

    /// Test that the fused kernel produces the same results as the separate
    /// gate GEMV + up GEMV + silu_mul pipeline.
    #[test]
    fn test_fused_gate_up_matches_separate() -> Result<()> {
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[FUSED_GATE_UP] No CUDA device, skipping test.");
                return Ok(());
            }
        };

        let cols = 2048usize;       // hidden_size
        let expert_ffn = 512usize;  // expert FFN dim
        let num_experts = 4usize;   // small number for testing
        let top_k = 2usize;

        let n_blocks_per_row = cols / IQ3S_BLOCK_ELEMS; // 8
        let row_bytes = n_blocks_per_row * IQ3S_BLOCK_BYTES; // 880
        let expert_stride = expert_ffn * row_bytes;          // 450560

        // Create synthetic data
        let gate_data = make_synthetic_iq3s_weights(num_experts * expert_ffn, cols);
        let up_data = make_synthetic_iq3s_weights(num_experts * expert_ffn, cols);
        // Use different data for up so results differ
        let up_data: Vec<u8> = up_data.iter().map(|&b| b.wrapping_add(37)).collect();
        let q8_data = make_synthetic_q8_input(cols);
        let expert_ids: Vec<i32> = vec![1, 3]; // select experts 1 and 3

        let stream = cuda_dev.cuda_stream();

        // Upload to GPU
        let mut gate_gpu: CudaSlice<u8> = stream.alloc_zeros(gate_data.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc gate: {e}")))?;
        cuda_dev.memcpy_htod(&gate_data, &mut gate_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload gate: {e}")))?;

        let mut up_gpu: CudaSlice<u8> = stream.alloc_zeros(up_data.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc up: {e}")))?;
        cuda_dev.memcpy_htod(&up_data, &mut up_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload up: {e}")))?;

        let mut q8_gpu: CudaSlice<u8> = stream.alloc_zeros(q8_data.len())
            .map_err(|e| candle_core::Error::Msg(format!("alloc q8: {e}")))?;
        cuda_dev.memcpy_htod(&q8_data, &mut q8_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload q8: {e}")))?;

        let mut ids_gpu: CudaSlice<i32> = stream.alloc_zeros(top_k)
            .map_err(|e| candle_core::Error::Msg(format!("alloc ids: {e}")))?;
        cuda_dev.memcpy_htod(&expert_ids, &mut ids_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload ids: {e}")))?;

        // --- Run fused kernel ---
        let output_size = top_k * expert_ffn;
        let mut fused_output: CudaSlice<f32> = stream.alloc_zeros(output_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc fused_output: {e}")))?;

        let gate_view = gate_gpu.slice(0..);
        let up_view = up_gpu.slice(0..);

        fused_gate_up_silu_iq3s(
            &gate_view,
            &up_view,
            &q8_gpu,
            &mut fused_output,
            &ids_gpu,
            cols,
            expert_ffn,
            top_k,
            expert_stride,
            cuda_dev,
        )?;

        // --- Run separate kernels (gate GEMV + up GEMV + silu_mul) ---
        let mut gate_output: CudaSlice<f32> = stream.alloc_zeros(output_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc gate_output: {e}")))?;
        let mut up_output: CudaSlice<f32> = stream.alloc_zeros(output_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc up_output: {e}")))?;
        let mut silu_mul_output: CudaSlice<f32> = stream.alloc_zeros(output_size)
            .map_err(|e| candle_core::Error::Msg(format!("alloc silu_mul_output: {e}")))?;

        // Gate batched GEMV
        crate::kernels::iq3s_gemv::gemv_iq3s_q8_batched(
            &gate_view,
            &q8_gpu,
            &mut gate_output,
            &ids_gpu,
            cols,
            expert_ffn,
            top_k,
            expert_stride,
            cuda_dev,
        )?;

        // Up batched GEMV
        crate::kernels::iq3s_gemv::gemv_iq3s_q8_batched(
            &up_view,
            &q8_gpu,
            &mut up_output,
            &ids_gpu,
            cols,
            expert_ffn,
            top_k,
            expert_stride,
            cuda_dev,
        )?;

        // silu_mul_batched
        crate::kernels::raw_silu_mul_batched(
            &gate_output,
            &up_output,
            &mut silu_mul_output,
            expert_ffn,
            top_k,
            cuda_dev,
        )?;

        // --- Compare outputs ---
        let fused_host: Vec<f32> = stream.clone().clone_dtoh(&fused_output)
            .map_err(|e| candle_core::Error::Msg(format!("dtoh fused: {e}")))?;
        let separate_host: Vec<f32> = stream.clone().clone_dtoh(&silu_mul_output)
            .map_err(|e| candle_core::Error::Msg(format!("dtoh separate: {e}")))?;

        assert_eq!(fused_host.len(), separate_host.len());
        let n = fused_host.len();

        let mut max_abs_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut sum_sq_diff = 0.0f64;
        let mut sum_sq_ref = 0.0f64;
        let mut mismatches = 0usize;

        for i in 0..n {
            let f = fused_host[i];
            let s = separate_host[i];
            let abs_diff = (f - s).abs();
            let rel_diff = if s.abs() > 1e-8 {
                abs_diff / s.abs()
            } else {
                abs_diff
            };

            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
            sum_sq_diff += (abs_diff as f64).powi(2);
            sum_sq_ref += (s as f64).powi(2);

            // Q8_1 quantization tolerance: results should match exactly since
            // both paths use the same Q8_1 input and the same dp4a dot product.
            // The only potential difference is floating-point accumulation order,
            // which should be identical given the same thread mapping.
            if abs_diff > 1e-4 && rel_diff > 0.005 {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "  Mismatch at [{}]: fused={:.6}, separate={:.6}, abs_diff={:.6}, rel_diff={:.4}",
                        i, f, s, abs_diff, rel_diff
                    );
                }
            }
        }

        let rmse = (sum_sq_diff / n as f64).sqrt();
        let nrmse = if sum_sq_ref > 0.0 {
            rmse / (sum_sq_ref / n as f64).sqrt()
        } else {
            rmse
        };

        eprintln!("[FUSED_GATE_UP] Comparison: n={}", n);
        eprintln!("  max_abs_diff = {:.6}", max_abs_diff);
        eprintln!("  max_rel_diff = {:.6}", max_rel_diff);
        eprintln!("  RMSE         = {:.6}", rmse);
        eprintln!("  NRMSE        = {:.6}", nrmse);
        eprintln!("  mismatches   = {} / {}", mismatches, n);

        // The fused kernel should produce identical results since the
        // dot product and thread mapping are identical.
        assert!(
            mismatches == 0,
            "Fused kernel has {} mismatches out of {} elements (max_abs={:.6}, max_rel={:.6})",
            mismatches, n, max_abs_diff, max_rel_diff
        );

        eprintln!("[FUSED_GATE_UP] PASS: fused kernel matches separate pipeline exactly.");
        Ok(())
    }
}
