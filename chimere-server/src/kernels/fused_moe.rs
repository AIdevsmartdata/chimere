//! Fused MoE IQ3_S kernel — single launch for all selected experts.
//!
//! Replaces 24 individual GEMV launches + 8 silu_mul + 8 weighted_add
//! with a single `fused_moe_iq3s` kernel. Each thread block handles one
//! expert: gate GEMV -> up GEMV -> silu_mul -> down GEMV + weighted atomicAdd.

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::OnceLock;

// -----------------------------------------------------------------------
// CUDA source: IQ3_S grid + helper functions + fused kernel
// -----------------------------------------------------------------------

const FUSED_MOE_KERNEL_SRC: &str = r#"
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant =  h        & 0x3FFu;
    if (exp == 0u) {
        float r = (float)mant * (1.0f / (1024.0f * 16384.0f));
        return sign ? -r : r;
    } else if (exp == 31u) {
        unsigned int f32 = (sign << 31) | 0x7F800000u | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    } else {
        unsigned int f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        float r; *(unsigned int*)&r = f32; return r;
    }
}

__constant__ unsigned int iq3s_grid[512] = {
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
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// Device helper: IQ3_S GEMV from global memory input -> shared-memory output
__device__ void iq3s_gemv_shared(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols, int tid, int num_threads
) {
    const int IQ3S_BYTES = 110;
    const int n_blocks_per_row = cols >> 8;

    for (int row = tid; row < rows; row += num_threads) {
        const unsigned char* row_w = weights + (long long)row * n_blocks_per_row * IQ3S_BYTES;
        float sum = 0.0f;

        for (int blk = 0; blk < n_blocks_per_row; blk++) {
            const unsigned char* b = row_w + blk * IQ3S_BYTES;
            int base_col = blk * 256;

            unsigned short d_bits = b[0] | (b[1] << 8);
            float d = f16_to_f32(d_bits);

            #pragma unroll
            for (int sg = 0; sg < 8; sg += 2) {
                unsigned char scale_byte = b[106 + sg / 2];
                float db1 = d * (float)(1 + 2 * (int)(scale_byte & 0x0F));
                float db2 = d * (float)(1 + 2 * (int)(scale_byte >> 4));

                int qs_off1    = 2 + sg * 8;
                int qh1        = (int)b[66 + sg];
                int signs_off1 = 74 + sg * 4;

                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int gi1 = b[qs_off1 + 2*l]   | (((qh1) << (8-2*l)) & 256);
                    int gi2 = b[qs_off1 + 2*l+1] | (((qh1) << (7-2*l)) & 256);
                    unsigned int g1 = iq3s_grid[gi1];
                    unsigned int g2 = iq3s_grid[gi2];
                    unsigned char sb = b[signs_off1 + l];
                    int col = base_col + sg * 32 + l * 8;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << j)) ? -1.0f : 1.0f;
                        sum += db1 * (float)((g1 >> (j*8)) & 0xFF) * sign * input[col + j];
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << (j+4))) ? -1.0f : 1.0f;
                        sum += db1 * (float)((g2 >> (j*8)) & 0xFF) * sign * input[col + 4 + j];
                    }
                }

                int qs_off2    = 2 + (sg+1) * 8;
                int qh2        = (int)b[66 + (sg+1)];
                int signs_off2 = 74 + (sg+1) * 4;

                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int gi1 = b[qs_off2 + 2*l]   | (((qh2) << (8-2*l)) & 256);
                    int gi2 = b[qs_off2 + 2*l+1] | (((qh2) << (7-2*l)) & 256);
                    unsigned int g1 = iq3s_grid[gi1];
                    unsigned int g2 = iq3s_grid[gi2];
                    unsigned char sb = b[signs_off2 + l];
                    int col = base_col + (sg+1) * 32 + l * 8;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << j)) ? -1.0f : 1.0f;
                        sum += db2 * (float)((g1 >> (j*8)) & 0xFF) * sign * input[col + j];
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << (j+4))) ? -1.0f : 1.0f;
                        sum += db2 * (float)((g2 >> (j*8)) & 0xFF) * sign * input[col + 4 + j];
                    }
                }
            }
        }
        output[row] = sum;
    }
}

// Device helper: Down projection with atomicAdd to output
__device__ void iq3s_gemv_down_atomic(
    const unsigned char* __restrict__ weights,
    const float* __restrict__ input,
    float* __restrict__ output,
    float weight,
    int rows, int cols, int tid, int num_threads
) {
    const int IQ3S_BYTES = 110;
    const int n_blocks_per_row = cols >> 8;

    for (int row = tid; row < rows; row += num_threads) {
        const unsigned char* row_w = weights + (long long)row * n_blocks_per_row * IQ3S_BYTES;
        float sum = 0.0f;

        for (int blk = 0; blk < n_blocks_per_row; blk++) {
            const unsigned char* b = row_w + blk * IQ3S_BYTES;
            int base_col = blk * 256;

            unsigned short d_bits = b[0] | (b[1] << 8);
            float d = f16_to_f32(d_bits);

            #pragma unroll
            for (int sg = 0; sg < 8; sg += 2) {
                unsigned char scale_byte = b[106 + sg / 2];
                float db1 = d * (float)(1 + 2 * (int)(scale_byte & 0x0F));
                float db2 = d * (float)(1 + 2 * (int)(scale_byte >> 4));

                int qs_off1    = 2 + sg * 8;
                int qh1        = (int)b[66 + sg];
                int signs_off1 = 74 + sg * 4;

                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int gi1 = b[qs_off1 + 2*l]   | (((qh1) << (8-2*l)) & 256);
                    int gi2 = b[qs_off1 + 2*l+1] | (((qh1) << (7-2*l)) & 256);
                    unsigned int g1 = iq3s_grid[gi1];
                    unsigned int g2 = iq3s_grid[gi2];
                    unsigned char sb = b[signs_off1 + l];
                    int col = base_col + sg * 32 + l * 8;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << j)) ? -1.0f : 1.0f;
                        sum += db1 * (float)((g1 >> (j*8)) & 0xFF) * sign * input[col + j];
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << (j+4))) ? -1.0f : 1.0f;
                        sum += db1 * (float)((g2 >> (j*8)) & 0xFF) * sign * input[col + 4 + j];
                    }
                }

                int qs_off2    = 2 + (sg+1) * 8;
                int qh2        = (int)b[66 + (sg+1)];
                int signs_off2 = 74 + (sg+1) * 4;

                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    int gi1 = b[qs_off2 + 2*l]   | (((qh2) << (8-2*l)) & 256);
                    int gi2 = b[qs_off2 + 2*l+1] | (((qh2) << (7-2*l)) & 256);
                    unsigned int g1 = iq3s_grid[gi1];
                    unsigned int g2 = iq3s_grid[gi2];
                    unsigned char sb = b[signs_off2 + l];
                    int col = base_col + (sg+1) * 32 + l * 8;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << j)) ? -1.0f : 1.0f;
                        sum += db2 * (float)((g1 >> (j*8)) & 0xFF) * sign * input[col + j];
                    }
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        float sign = (sb & (1u << (j+4))) ? -1.0f : 1.0f;
                        sum += db2 * (float)((g2 >> (j*8)) & 0xFF) * sign * input[col + 4 + j];
                    }
                }
            }
        }
        atomicAdd(&output[row], sum * weight);
    }
}

extern "C" __global__ void fused_moe_iq3s(
    const float*         __restrict__ hidden,
    const unsigned char* __restrict__ gate_exps,
    const unsigned char* __restrict__ up_exps,
    const unsigned char* __restrict__ down_exps,
    const int*           __restrict__ expert_ids,
    const float*         __restrict__ expert_weights,
    float*               __restrict__ output,
    int hidden_size,
    int expert_ffn,
    int expert_bytes_gate,
    int expert_bytes_up,
    int expert_bytes_down
) {
    int expert_id = expert_ids[blockIdx.x];
    float weight  = expert_weights[blockIdx.x];
    int tid       = threadIdx.x;

    extern __shared__ char smem[];
    float* sh_hidden = (float*)smem;
    float* sh_gate   = sh_hidden + hidden_size;
    float* sh_up     = sh_gate   + expert_ffn;
    float* sh_inter  = sh_up     + expert_ffn;

    for (int i = tid; i < hidden_size; i += blockDim.x)
        sh_hidden[i] = hidden[i];
    __syncthreads();

    iq3s_gemv_shared(
        gate_exps + (long long)expert_id * expert_bytes_gate,
        sh_hidden, sh_gate, expert_ffn, hidden_size, tid, blockDim.x
    );
    __syncthreads();

    iq3s_gemv_shared(
        up_exps + (long long)expert_id * expert_bytes_up,
        sh_hidden, sh_up, expert_ffn, hidden_size, tid, blockDim.x
    );
    __syncthreads();

    for (int i = tid; i < expert_ffn; i += blockDim.x) {
        float g = sh_gate[i];
        sh_inter[i] = (g / (1.0f + expf(-g))) * sh_up[i];
    }
    __syncthreads();

    iq3s_gemv_down_atomic(
        down_exps + (long long)expert_id * expert_bytes_down,
        sh_inter, output, weight, hidden_size, expert_ffn, tid, blockDim.x
    );
}
"#;

// -----------------------------------------------------------------------
// PTX cache
// -----------------------------------------------------------------------

static FUSED_MOE_PTX: OnceLock<String> = OnceLock::new();

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

/// Fused MoE IQ3_S kernel: single launch for all selected experts.
///
/// Each thread block handles one expert:
///   1. Load hidden to shared memory
///   2. Gate GEMV (IQ3_S -> shared)
///   3. Up GEMV (IQ3_S -> shared)
///   4. SiLU(gate) * up -> intermediate
///   5. Down GEMV with weighted atomicAdd to output
///
/// # Arguments
/// - `hidden`: CudaView of the [hidden_size] input vector.
/// - `gate_exps`: raw IQ3_S bytes for all experts' gate projections (stacked).
/// - `up_exps`: raw IQ3_S bytes for all experts' up projections (stacked).
/// - `down_exps`: raw IQ3_S bytes for all experts' down projections (stacked).
/// - `expert_ids`: CPU slice of selected expert indices.
/// - `expert_weights`: CPU slice of corresponding expert weights.
/// - `output`: pre-allocated [hidden_size] accumulator (must be zeroed before call).
/// - `hidden_size`, `expert_ffn`: model dimensions.
/// - `expert_bytes_gate`, `expert_bytes_up`, `expert_bytes_down`: byte stride per expert.
/// - `top_k`: number of selected experts.
/// - `dev`: CUDA device.
pub fn fused_moe_iq3s(
    hidden: &CudaView<'_, f32>,
    gate_exps: &CudaView<'_, u8>,
    up_exps: &CudaView<'_, u8>,
    down_exps: &CudaView<'_, u8>,
    expert_ids: &[i32],
    expert_weights: &[f32],
    output: &mut CudaSlice<f32>,
    hidden_size: usize,
    expert_ffn: usize,
    expert_bytes_gate: usize,
    expert_bytes_up: usize,
    expert_bytes_down: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_moe_iq3s", "chimere_fused_moe_iq3s_v3",
        FUSED_MOE_KERNEL_SRC, &FUSED_MOE_PTX,
    )?;

    // Upload expert_ids and expert_weights to GPU
    let mut ids_gpu: CudaSlice<i32> = dev
        .alloc_zeros::<i32>(expert_ids.len())
        .map_err(|e| candle_core::Error::Msg(format!("alloc expert_ids: {e}")))?;
    dev.memcpy_htod(expert_ids, &mut ids_gpu)
        .map_err(|e| candle_core::Error::Msg(format!("upload expert_ids: {e}")))?;
    let mut wts_gpu: CudaSlice<f32> = dev
        .alloc_zeros::<f32>(expert_weights.len())
        .map_err(|e| candle_core::Error::Msg(format!("alloc expert_weights: {e}")))?;
    dev.memcpy_htod(expert_weights, &mut wts_gpu)
        .map_err(|e| candle_core::Error::Msg(format!("upload expert_weights: {e}")))?;

    // Shared memory: hidden + gate + up + inter = hidden_size + 3*expert_ffn floats.
    let smem_bytes = ((hidden_size + 3 * expert_ffn) * 4) as u32;

    // One block per expert, 256 threads per block.
    let cfg = LaunchConfig {
        grid_dim: (top_k as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    let hidden_size_i32 = hidden_size as i32;
    let expert_ffn_i32 = expert_ffn as i32;
    let expert_bytes_gate_i32 = expert_bytes_gate as i32;
    let expert_bytes_up_i32 = expert_bytes_up as i32;
    let expert_bytes_down_i32 = expert_bytes_down as i32;

    let mut builder = _stream.launch_builder(&func);
    builder.arg(hidden);
    builder.arg(gate_exps);
    builder.arg(up_exps);
    builder.arg(down_exps);
    builder.arg(&ids_gpu);
    builder.arg(&wts_gpu);
    builder.arg(output);
    builder.arg(&hidden_size_i32);
    builder.arg(&expert_ffn_i32);
    builder.arg(&expert_bytes_gate_i32);
    builder.arg(&expert_bytes_up_i32);
    builder.arg(&expert_bytes_down_i32);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("fused_moe_iq3s launch: {e}")))?;

    Ok(())
}

/// Same as `fused_moe_iq3s` but expert_ids and weights are already on GPU.
/// Eliminates 2 alloc + 2 memcpy_htod per MoE layer (saves ~40 PCIe round-trips/token).
pub fn fused_moe_iq3s_gpu_resident(
    hidden: &CudaView<'_, f32>,
    gate_exps: &CudaView<'_, u8>,
    up_exps: &CudaView<'_, u8>,
    down_exps: &CudaView<'_, u8>,
    expert_ids_gpu: &CudaSlice<i32>,
    expert_weights_gpu: &CudaSlice<f32>,
    output: &mut CudaSlice<f32>,
    hidden_size: usize,
    expert_ffn: usize,
    expert_bytes_gate: usize,
    expert_bytes_up: usize,
    expert_bytes_down: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, "fused_moe_iq3s", "chimere_fused_moe_iq3s_v3",
        FUSED_MOE_KERNEL_SRC, &FUSED_MOE_PTX,
    )?;

    let smem_bytes = ((hidden_size + 3 * expert_ffn) * 4) as u32;
    let cfg = LaunchConfig {
        grid_dim: (top_k as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem_bytes,
    };

    let hidden_size_i32 = hidden_size as i32;
    let expert_ffn_i32 = expert_ffn as i32;
    let expert_bytes_gate_i32 = expert_bytes_gate as i32;
    let expert_bytes_up_i32 = expert_bytes_up as i32;
    let expert_bytes_down_i32 = expert_bytes_down as i32;

    let mut builder = _stream.launch_builder(&func);
    builder.arg(hidden);
    builder.arg(gate_exps);
    builder.arg(up_exps);
    builder.arg(down_exps);
    builder.arg(expert_ids_gpu);
    builder.arg(expert_weights_gpu);
    builder.arg(output);
    builder.arg(&hidden_size_i32);
    builder.arg(&expert_ffn_i32);
    builder.arg(&expert_bytes_gate_i32);
    builder.arg(&expert_bytes_up_i32);
    builder.arg(&expert_bytes_down_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("fused_moe_iq3s_gpu launch: {e}")))?;

    Ok(())
}
