//! IQ3_S GEMV v3 — ik_llama-inspired optimizations.
//!
//! Key optimizations over v2:
//! 1. **Q8_1 input in shared memory**: All 128 threads cooperatively load Q8_1
//!    blocks into shared memory, then all 8 threads per IQ3_S block read from
//!    shared memory instead of global memory. This eliminates 7/8 redundant
//!    global reads of the Q8_1 input vector.
//! 2. **Two rows per CUDA block**: Following ik_llama's `rows_per_cuda_block=2`
//!    pattern, each block computes 2 output rows, amortizing the shared memory
//!    Q8_1 load across both rows.
//! 3. **Fused gate+up+silu v3**: Same shared-memory + 2-row pattern for the
//!    fused gate+up kernel. Reads Q8_1 once, computes gate AND up for 2 rows.
//!
//! Toggle: `CHIMERE_IQ3S_V3=1` environment variable.
//! Old kernels remain as fallback.

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// CUDA kernel source — v3 optimized IQ3_S GEMV
// ---------------------------------------------------------------------------

const IQ3S_V3_KERNEL_SRC: &str = r#"
// =====================================================================
// IQ3_S GEMV v3 — ik_llama optimizations
//
// Key changes from v2:
//  1. Q8_1 input loaded into shared memory cooperatively
//  2. Two output rows per CUDA block (halves grid size, amortizes smem load)
//  3. ggml-faithful cross-warp reduction with rows_per_cuda_block=2
// =====================================================================

// f16 -> f32 via hardware cvt.f32.f16
__device__ __forceinline__ float f16_to_f32(unsigned short h) {
    float f;
    asm("{ .reg .b16 tmp; mov.b16 tmp, %1; cvt.f32.f16 %0, tmp; }"
        : "=f"(f) : "h"(h));
    return f;
}

// f32 -> f16 using hardware round-to-nearest-even (matches ggml __float2half_rn)
__device__ __forceinline__ unsigned short f32_to_f16_bits(float x) {
    unsigned short h;
    asm("{ .reg .b16 tmp; cvt.rn.f16.f32 tmp, %1; mov.b16 %0, tmp; }" : "=h"(h) : "f"(x));
    return h;
}

// IQ3_S grid constant — 512 entries (identical across all versions)
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
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101
};

// =====================================================================
// Common helpers
// =====================================================================

struct block_q8_1_gemv {
    unsigned short d_bits;
    unsigned short s_bits;
    signed char    qs[32];
};

__device__ void quantize_q8_1_block(
    const float* __restrict__ x,
    struct block_q8_1_gemv* b
) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float ax = x[i] < 0.0f ? -x[i] : x[i];
        if (ax > amax) amax = ax;
    }
    float scale = amax / 127.0f;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    b->d_bits = f32_to_f16_bits(scale);

    int sum = 0;
    for (int i = 0; i < 32; i++) {
        float fq = x[i] * inv_scale;
        int q = (int)(fq + (fq >= 0.0f ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        b->qs[i] = (signed char)q;
        sum += q;
    }
    b->s_bits = f32_to_f16_bits(scale * (float)sum);
}

extern "C" __global__ void quantize_f32_to_q8_1(
    const float* __restrict__ input,
    unsigned char* __restrict__ q8_out,
    int n_blocks
) {
    for (int blk = threadIdx.x + blockIdx.x * blockDim.x; blk < n_blocks; blk += blockDim.x * gridDim.x) {
        quantize_q8_1_block(&input[blk * 32], (struct block_q8_1_gemv*)(q8_out + blk * 36));
    }
}

extern "C" __global__ void quantize_q8_1_batched(
    const float* __restrict__ input_all,
    unsigned char* __restrict__ q8_out_all,
    int blocks_per_expert,
    int n_per_expert,
    int q8_stride
) {
    int k = blockIdx.y;
    int blk = threadIdx.x + blockIdx.x * blockDim.x;
    if (blk >= blocks_per_expert) return;
    quantize_q8_1_block(
        &input_all[k * n_per_expert + blk * 32],
        (struct block_q8_1_gemv*)(q8_out_all + k * q8_stride + blk * 36)
    );
}

// =====================================================================
// IQ3_S dequantisation kernel (unchanged from v1/v2)
// =====================================================================

extern "C" __global__ void dequant_iq3s(
    const unsigned char* __restrict__ data,
    float*               __restrict__ output,
    int                               n_blocks
) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= n_blocks) return;

    const unsigned char* block = data + block_idx * 110;
    float*               out   = output + block_idx * 256;

    unsigned short d_bits = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
    float d = f16_to_f32(d_bits);

    int qs_off    = 2;
    int qh_off    = 66;
    int signs_off = 74;
    int out_off   = 0;

    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        unsigned char scale_byte = block[106 + ib32 / 2];
        float db1 = d * (float)(1 + 2 * (scale_byte & 0x0F));
        float db2 = d * (float)(1 + 2 * (scale_byte >> 4));

        for (int l = 0; l < 4; l++) {
            int grid_idx1 = (int)block[qs_off + 2*l]
                          | ((((int)block[qh_off]) << (8 - 2*l)) & 256);
            int grid_idx2 = (int)block[qs_off + 2*l + 1]
                          | ((((int)block[qh_off]) << (7 - 2*l)) & 256);
            unsigned int g1 = iq3s_grid[grid_idx1];
            unsigned int g2 = iq3s_grid[grid_idx2];
            unsigned char sign_byte = block[signs_off + l];
            for (int j = 0; j < 4; j++) {
                float sign = (sign_byte & (1 << j)) ? -1.0f : 1.0f;
                out[out_off++] = db1 * (float)((g1 >> (j*8)) & 0xFF) * sign;
            }
            for (int j = 0; j < 4; j++) {
                float sign = (sign_byte & (1 << (j+4))) ? -1.0f : 1.0f;
                out[out_off++] = db1 * (float)((g2 >> (j*8)) & 0xFF) * sign;
            }
        }
        qs_off    += 8;
        signs_off += 4;
        for (int l = 0; l < 4; l++) {
            int grid_idx1 = (int)block[qs_off + 2*l]
                          | ((((int)block[qh_off + 1]) << (8 - 2*l)) & 256);
            int grid_idx2 = (int)block[qs_off + 2*l + 1]
                          | ((((int)block[qh_off + 1]) << (7 - 2*l)) & 256);
            unsigned int g1 = iq3s_grid[grid_idx1];
            unsigned int g2 = iq3s_grid[grid_idx2];
            unsigned char sign_byte = block[signs_off + l];
            for (int j = 0; j < 4; j++) {
                float sign = (sign_byte & (1 << j)) ? -1.0f : 1.0f;
                out[out_off++] = db2 * (float)((g1 >> (j*8)) & 0xFF) * sign;
            }
            for (int j = 0; j < 4; j++) {
                float sign = (sign_byte & (1 << (j+4))) ? -1.0f : 1.0f;
                out[out_off++] = db2 * (float)((g2 >> (j*8)) & 0xFF) * sign;
            }
        }
        qh_off    += 2;
        qs_off    += 8;
        signs_off += 4;
    }
}

// =====================================================================
// Vectorized load helpers
// =====================================================================

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}
__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// =====================================================================
// Warp reduction
// =====================================================================
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// =====================================================================
// IQ3_S dot product — v3 with shared-memory Q8_1
//
// Instead of reading Q8_1 from global memory pointer, reads from shared
// memory. The caller loads Q8_1 blocks into smem before calling this.
//
// smem_q8: pointer to Q8_1 block in shared memory (36 bytes per block)
// bq3: IQ3_S block pointer in global memory (110 bytes)
// iqs: sub-group index (0,2,...,14)
// q8_block_base: index of first Q8_1 block for this IQ3_S block in smem
// =====================================================================

__device__ __forceinline__ float dot_iq3s_q8_smem(
    const unsigned char* __restrict__ bq3,
    const unsigned char* __restrict__ smem_q8_base,  // shared mem Q8_1 data
    int iqs
) {
    // Vectorized qs loads
    const int qs_lo = get_int_b2(bq3 + 2, iqs);
    const int qs_hi = get_int_b2(bq3 + 2, iqs + 1);

    const int qh = bq3[66 + iqs/2];

    const int signs32 = get_int_b2(bq3 + 74, iqs/2);
    const unsigned char* sp = (const unsigned char*)&signs32;

    // Q8_1 block pointer in shared memory
    const unsigned char* bq8 = smem_q8_base + (iqs/2) * 36;

    int sumi = 0;

    // Unrolled: 4 iterations x 2 grid lookups = 32 elements
    // l=0
    {
        const int grid0 = iq3s_grid[(qs_lo & 0xFF)        | ((qh << 8) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 8) & 0xFF) | ((qh << 7) & 0x100)];
        const int s0 = __vcmpne4(((sp[0] & 0x03) << 7) | ((sp[0] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[0] & 0x30) << 3) | ((sp[0] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 0), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 1), sumi);
    }
    // l=1
    {
        const int grid0 = iq3s_grid[((qs_lo >> 16) & 0xFF) | ((qh << 6) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 24) & 0xFF) | ((qh << 5) & 0x100)];
        const int s0 = __vcmpne4(((sp[1] & 0x03) << 7) | ((sp[1] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[1] & 0x30) << 3) | ((sp[1] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 2), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 3), sumi);
    }
    // l=2
    {
        const int grid0 = iq3s_grid[(qs_hi & 0xFF)        | ((qh << 4) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 8) & 0xFF) | ((qh << 3) & 0x100)];
        const int s0 = __vcmpne4(((sp[2] & 0x03) << 7) | ((sp[2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[2] & 0x30) << 3) | ((sp[2] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 4), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 5), sumi);
    }
    // l=3
    {
        const int grid0 = iq3s_grid[((qs_hi >> 16) & 0xFF) | ((qh << 2) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 24) & 0xFF) | ((qh << 1) & 0x100)];
        const int s0 = __vcmpne4(((sp[3] & 0x03) << 7) | ((sp[3] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[3] & 0x30) << 3) | ((sp[3] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 6), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 7), sumi);
    }

    sumi *= 1 + 2 * ((bq3[106 + iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = f16_to_f32(*(const unsigned short*)bq3)
                  * f16_to_f32(*(const unsigned short*)bq8);
    return d * (float)sumi;
}

// Also keep a global-memory version for fallback paths
__device__ __forceinline__ float dot_iq3s_q8_gmem(
    const unsigned char* __restrict__ bq3,
    const unsigned char* __restrict__ bq8,
    int iqs
) {
    const int qs_lo = get_int_b2(bq3 + 2, iqs);
    const int qs_hi = get_int_b2(bq3 + 2, iqs + 1);
    const int qh = bq3[66 + iqs/2];
    const int signs32 = get_int_b2(bq3 + 74, iqs/2);
    const unsigned char* sp = (const unsigned char*)&signs32;

    int sumi = 0;
    {
        const int grid0 = iq3s_grid[(qs_lo & 0xFF)        | ((qh << 8) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 8) & 0xFF) | ((qh << 7) & 0x100)];
        const int s0 = __vcmpne4(((sp[0] & 0x03) << 7) | ((sp[0] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[0] & 0x30) << 3) | ((sp[0] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 0), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 1), sumi);
    }
    {
        const int grid0 = iq3s_grid[((qs_lo >> 16) & 0xFF) | ((qh << 6) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 24) & 0xFF) | ((qh << 5) & 0x100)];
        const int s0 = __vcmpne4(((sp[1] & 0x03) << 7) | ((sp[1] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[1] & 0x30) << 3) | ((sp[1] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 2), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 3), sumi);
    }
    {
        const int grid0 = iq3s_grid[(qs_hi & 0xFF)        | ((qh << 4) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 8) & 0xFF) | ((qh << 3) & 0x100)];
        const int s0 = __vcmpne4(((sp[2] & 0x03) << 7) | ((sp[2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[2] & 0x30) << 3) | ((sp[2] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 4), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 5), sumi);
    }
    {
        const int grid0 = iq3s_grid[((qs_hi >> 16) & 0xFF) | ((qh << 2) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 24) & 0xFF) | ((qh << 1) & 0x100)];
        const int s0 = __vcmpne4(((sp[3] & 0x03) << 7) | ((sp[3] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[3] & 0x30) << 3) | ((sp[3] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 6), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 7), sumi);
    }

    sumi *= 1 + 2 * ((bq3[106 + iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = f16_to_f32(*(const unsigned short*)bq3)
                  * f16_to_f32(*(const unsigned short*)bq8);
    return d * (float)sumi;
}

// =====================================================================
// V3 GEMV kernel: shared-memory Q8_1 + 2 rows per block
//
// Block: (32, 4, 1) = 128 threads = 4 warps
// Grid:  (ceil(nrows/2), 1, 1)
//
// Shared memory layout:
//   [Q8_1 staging area: BLOCKS_PER_ITER * 36 bytes]
//   [Cross-warp reduction: 3 * 2 * 32 floats = 768 bytes]
//
// Each iteration:
//   1. All 128 threads cooperatively load BLOCKS_PER_ITER Q8_1 blocks
//      (16 blocks x 36 bytes = 576 bytes) into shared memory
//   2. Each group of 8 threads computes one dot product from smem Q8_1
//   3. Repeat for both rows (same Q8_1 data, different weight rows)
//
// With BLOCKS_PER_ITER=16 and 128 threads: each thread loads
// 576/128 ~ 4.5 bytes per iteration. We round up to 5 per thread.
// =====================================================================

// Constants matching ggml
#define NWARPS        4
#define WARP_SIZE_V3  32
#define QI_IQ3S       16
#define VDR_IQ3S      2
#define QK_IQ3S       256
#define IQ3S_BYTES    110
#define Q8_1_BYTES    36

// 128 threads, qi=16, vdr=2 => blocks_per_iter = 2*4*32/16 = 16
#define BLOCKS_PER_ITER  16

// Shared memory for Q8_1 staging: 16 blocks x 36 bytes = 576 bytes
// Each IQ3_S block maps to 8 Q8_1 blocks, but we only need the Q8_1
// blocks that correspond to the BLOCKS_PER_ITER IQ3_S blocks.
// Actually: each IQ3_S block = 256 elems, Q8_1 block = 32 elems,
// so 1 IQ3_S block uses 8 Q8_1 blocks. 16 IQ3_S blocks = 128 Q8_1 blocks.
// 128 * 36 = 4608 bytes — that's a lot of shared mem.
// Better approach: load Q8_1 blocks on-demand per iteration batch.
// Each thread group processes 1 IQ3_S block with iqs selecting which
// of the 8 Q8_1 sub-blocks. So across the 16 IQ3_S blocks per iter,
// we need 16*8=128 Q8_1 blocks total. But each thread only reads 1.
// The win from shared memory comes from the 2-row optimization:
// we load Q8_1 ONCE and use it for BOTH rows.

// Shared memory size: 128 Q8_1 blocks * 36 bytes = 4608 bytes
// + cross-warp reduction: 3 * 2 * 32 * 4 = 768 bytes
// Total: 5376 bytes per block — well within 48KB limit

#define SMEM_Q8_BLOCKS  (BLOCKS_PER_ITER * 8)   // 128
#define SMEM_Q8_BYTES   (SMEM_Q8_BLOCKS * Q8_1_BYTES)  // 4608
// Reduction area: [nwarps-1][rows_per_block][WARP_SIZE] = [3][2][32]
#define SMEM_REDUCE_FLOATS  (3 * 2 * WARP_SIZE_V3)  // 192
#define SMEM_REDUCE_BYTES   (SMEM_REDUCE_FLOATS * 4)  // 768
#define SMEM_TOTAL_BYTES    (SMEM_Q8_BYTES + SMEM_REDUCE_BYTES)  // 5376

extern "C" __global__
__launch_bounds__(128, 1)
void gemv_iq3s_q8(
    const unsigned char* __restrict__ weights,
    const float*         __restrict__ input_unused,
    float*               __restrict__ output,
    int                               cols,
    const unsigned char* __restrict__ q8_input
) {
    const int row0     = 2 * blockIdx.x;       // first of 2 rows
    const int warp_id  = threadIdx.y;           // 0..3
    const int lane_id  = threadIdx.x;           // 0..31
    const int tid      = warp_id * 32 + lane_id;

    const int n_iq3s_blocks = cols >> 8;   // cols / 256

    // Thread mapping: 8 threads per IQ3_S block
    const int iqs = 2 * (tid & 7);         // 0,2,...,14

    // Shared memory layout:
    //   [0 .. SMEM_Q8_BYTES) = Q8_1 staging area
    //   [SMEM_Q8_BYTES .. SMEM_TOTAL_BYTES) = cross-warp reduction
    extern __shared__ char smem_raw[];
    unsigned char* smem_q8 = (unsigned char*)smem_raw;
    float* tmp_shared = (float*)(smem_raw + SMEM_Q8_BYTES);
    // tmp_shared layout: [warp_idx][row_idx][lane] where warp_idx = 0..2, row_idx = 0..1

    // Row weight pointers for the 2 rows
    const unsigned char* row0_weights =
        weights + (long long)row0 * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* row1_weights =
        weights + (long long)(row0 + 1) * (long long)n_iq3s_blocks * IQ3S_BYTES;

    float partial[2] = {0.0f, 0.0f};

    // Determine total number of rows for bounds checking
    const int nrows = cols;  // Note: nrows is passed implicitly via grid size
    // Actually nrows isn't directly available. We use output bounds from grid.

    for (int kbx_base = 0; kbx_base < n_iq3s_blocks; kbx_base += BLOCKS_PER_ITER) {
        // --- Step 1: Cooperatively load Q8_1 blocks into shared memory ---
        // We need Q8_1 blocks for kbx_base..kbx_base+BLOCKS_PER_ITER IQ3_S blocks.
        // Each IQ3_S block maps to 8 Q8_1 blocks starting at kbx*8.
        // Total Q8_1 blocks = min(BLOCKS_PER_ITER, remaining) * 8
        int n_iq3_this_iter = BLOCKS_PER_ITER;
        if (kbx_base + BLOCKS_PER_ITER > n_iq3s_blocks) {
            n_iq3_this_iter = n_iq3s_blocks - kbx_base;
        }
        int n_q8_blocks = n_iq3_this_iter * 8;

        // 128 threads load n_q8_blocks * 36 bytes
        // Each thread loads 36 bytes worth of Q8_1 blocks (1 block at a time)
        // Round-robin: thread tid loads blocks tid, tid+128, tid+256, ...
        for (int q8idx = tid; q8idx < n_q8_blocks; q8idx += 128) {
            // Source: global Q8_1 at kbx_base*8 + q8idx
            const int src_q8_block = (kbx_base * 8) + q8idx;
            const unsigned char* src = q8_input + (long long)src_q8_block * Q8_1_BYTES;
            unsigned char* dst = smem_q8 + q8idx * Q8_1_BYTES;
            // Copy 36 bytes using int loads for efficiency
            // 36 = 9*4, load as 9 ints
            const int* src4 = (const int*)src;
            int* dst4 = (int*)dst;
            dst4[0] = src4[0]; dst4[1] = src4[1]; dst4[2] = src4[2];
            dst4[3] = src4[3]; dst4[4] = src4[4]; dst4[5] = src4[5];
            dst4[6] = src4[6]; dst4[7] = src4[7]; dst4[8] = src4[8];
        }
        __syncthreads();

        // --- Step 2: Compute dot products using shared-memory Q8_1 ---
        // Each group of 8 threads handles one IQ3_S block
        int local_kbx = tid / 8;  // 0..15
        if (kbx_base + local_kbx < n_iq3s_blocks) {
            int kbx = kbx_base + local_kbx;

            // Q8_1 data for this IQ3_S block starts at local_kbx*8 Q8_1 blocks in smem
            const unsigned char* smem_q8_for_block = smem_q8 + local_kbx * 8 * Q8_1_BYTES;

            // Row 0
            const unsigned char* bq3_0 = row0_weights + kbx * IQ3S_BYTES;
            partial[0] += dot_iq3s_q8_smem(bq3_0, smem_q8_for_block, iqs);

            // Row 1 (reuse same Q8_1 from shared memory!)
            partial[1] += dot_iq3s_q8_smem(row1_weights + kbx * IQ3S_BYTES,
                                           smem_q8_for_block, iqs);
        }

        __syncthreads();  // Before next iteration's smem write
    }

    // --- Cross-warp reduction (ggml-faithful, rows_per_cuda_block=2) ---
    if (warp_id > 0) {
        // tmp_shared[(warp_id-1)*2*32 + row*32 + lane_id]
        tmp_shared[(warp_id - 1) * 2 * WARP_SIZE_V3 + 0 * WARP_SIZE_V3 + lane_id] = partial[0];
        tmp_shared[(warp_id - 1) * 2 * WARP_SIZE_V3 + 1 * WARP_SIZE_V3 + lane_id] = partial[1];
    }
    __syncthreads();
    if (warp_id > 0) return;

    // Warp 0: accumulate from warps 1-3
    #pragma unroll
    for (int w = 0; w < 3; w++) {
        partial[0] += tmp_shared[w * 2 * WARP_SIZE_V3 + 0 * WARP_SIZE_V3 + lane_id];
        partial[1] += tmp_shared[w * 2 * WARP_SIZE_V3 + 1 * WARP_SIZE_V3 + lane_id];
    }

    partial[0] = warp_reduce_sum(partial[0]);
    partial[1] = warp_reduce_sum(partial[1]);

    if (lane_id == 0) {
        output[row0] = partial[0];
    }
    if (lane_id == 1) {
        output[row0 + 1] = partial[1];
    }
}

// =====================================================================
// V3 batched GEMV: shared Q8_1 input + 2 rows per block
// =====================================================================
extern "C" __global__
__launch_bounds__(128, 1)
void gemv_iq3s_q8_batched(
    const unsigned char* __restrict__ all_weights,
    float*               __restrict__ output,
    int                               cols,
    const unsigned char* __restrict__ q8_input,
    const int*           __restrict__ expert_ids,
    int                               rows,
    int                               expert_stride
) {
    // 2 rows per block
    const int global_pair = blockIdx.x;
    const int k           = global_pair / ((rows + 1) / 2);
    const int pair_idx    = global_pair % ((rows + 1) / 2);
    const int row0        = 2 * pair_idx;
    const int warp_id     = threadIdx.y;
    const int lane_id     = threadIdx.x;
    const int tid         = warp_id * 32 + lane_id;

    const int n_iq3s_blocks = cols >> 8;
    const int iqs = 2 * (tid & 7);

    int expert_id = expert_ids[k];
    const unsigned char* expert_weights = all_weights + (long long)expert_id * expert_stride;
    const unsigned char* row0_weights = expert_weights + (long long)row0 * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* row1_weights = expert_weights + (long long)(row0 + 1) * (long long)n_iq3s_blocks * IQ3S_BYTES;

    extern __shared__ char smem_raw[];
    unsigned char* smem_q8 = (unsigned char*)smem_raw;
    float* tmp_shared = (float*)(smem_raw + SMEM_Q8_BYTES);

    float partial[2] = {0.0f, 0.0f};
    const int do_row1 = (row0 + 1 < rows) ? 1 : 0;

    for (int kbx_base = 0; kbx_base < n_iq3s_blocks; kbx_base += BLOCKS_PER_ITER) {
        int n_iq3_this_iter = BLOCKS_PER_ITER;
        if (kbx_base + BLOCKS_PER_ITER > n_iq3s_blocks)
            n_iq3_this_iter = n_iq3s_blocks - kbx_base;
        int n_q8_blocks = n_iq3_this_iter * 8;

        for (int q8idx = tid; q8idx < n_q8_blocks; q8idx += 128) {
            const int src_q8_block = (kbx_base * 8) + q8idx;
            const unsigned char* src = q8_input + (long long)src_q8_block * Q8_1_BYTES;
            unsigned char* dst = smem_q8 + q8idx * Q8_1_BYTES;
            const int* src4 = (const int*)src;
            int* dst4 = (int*)dst;
            dst4[0] = src4[0]; dst4[1] = src4[1]; dst4[2] = src4[2];
            dst4[3] = src4[3]; dst4[4] = src4[4]; dst4[5] = src4[5];
            dst4[6] = src4[6]; dst4[7] = src4[7]; dst4[8] = src4[8];
        }
        __syncthreads();

        int local_kbx = tid / 8;
        if (kbx_base + local_kbx < n_iq3s_blocks) {
            int kbx = kbx_base + local_kbx;
            const unsigned char* smem_q8_for_block = smem_q8 + local_kbx * 8 * Q8_1_BYTES;

            partial[0] += dot_iq3s_q8_smem(row0_weights + kbx * IQ3S_BYTES,
                                           smem_q8_for_block, iqs);
            if (do_row1) {
                partial[1] += dot_iq3s_q8_smem(row1_weights + kbx * IQ3S_BYTES,
                                               smem_q8_for_block, iqs);
            }
        }
        __syncthreads();
    }

    if (warp_id > 0) {
        tmp_shared[(warp_id - 1) * 2 * WARP_SIZE_V3 + 0 * WARP_SIZE_V3 + lane_id] = partial[0];
        tmp_shared[(warp_id - 1) * 2 * WARP_SIZE_V3 + 1 * WARP_SIZE_V3 + lane_id] = partial[1];
    }
    __syncthreads();
    if (warp_id > 0) return;

    #pragma unroll
    for (int w = 0; w < 3; w++) {
        partial[0] += tmp_shared[w * 2 * WARP_SIZE_V3 + 0 * WARP_SIZE_V3 + lane_id];
        partial[1] += tmp_shared[w * 2 * WARP_SIZE_V3 + 1 * WARP_SIZE_V3 + lane_id];
    }

    partial[0] = warp_reduce_sum(partial[0]);
    partial[1] = warp_reduce_sum(partial[1]);

    if (lane_id == 0) {
        output[k * rows + row0] = partial[0];
    }
    if (lane_id == 1 && do_row1) {
        output[k * rows + row0 + 1] = partial[1];
    }
}

// =====================================================================
// V3 batched multi-input GEMV: different Q8_1 per expert + 2 rows/block
// =====================================================================
extern "C" __global__
__launch_bounds__(128, 1)
void gemv_iq3s_q8_batched_multi_input(
    const unsigned char* __restrict__ all_weights,
    const unsigned char* __restrict__ all_q8_inputs,
    float*               __restrict__ output,
    const int*           __restrict__ expert_ids,
    int                               cols,
    int                               rows,
    int                               expert_stride,
    int                               q8_stride
) {
    // Grid: (ceil(rows/2), top_k, 1)
    const int pair_idx = blockIdx.x;
    const int k        = blockIdx.y;
    const int row0     = 2 * pair_idx;
    const int warp_id  = threadIdx.y;
    const int lane_id  = threadIdx.x;
    const int tid      = warp_id * 32 + lane_id;

    const int n_iq3s_blocks = cols >> 8;
    const int iqs = 2 * (tid & 7);

    int expert_id = expert_ids[k];
    const unsigned char* expert_weights = all_weights + (long long)expert_id * expert_stride;
    const unsigned char* row0_weights = expert_weights + (long long)row0 * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* row1_weights = expert_weights + (long long)(row0 + 1) * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* q8_input = all_q8_inputs + (long long)k * q8_stride;

    extern __shared__ char smem_raw[];
    unsigned char* smem_q8 = (unsigned char*)smem_raw;
    float* tmp_shared = (float*)(smem_raw + SMEM_Q8_BYTES);

    float partial[2] = {0.0f, 0.0f};
    const int do_row1 = (row0 + 1 < rows) ? 1 : 0;

    for (int kbx_base = 0; kbx_base < n_iq3s_blocks; kbx_base += BLOCKS_PER_ITER) {
        int n_iq3_this_iter = BLOCKS_PER_ITER;
        if (kbx_base + BLOCKS_PER_ITER > n_iq3s_blocks)
            n_iq3_this_iter = n_iq3s_blocks - kbx_base;
        int n_q8_blocks = n_iq3_this_iter * 8;

        for (int q8idx = tid; q8idx < n_q8_blocks; q8idx += 128) {
            const int src_q8_block = (kbx_base * 8) + q8idx;
            const unsigned char* src = q8_input + (long long)src_q8_block * Q8_1_BYTES;
            unsigned char* dst = smem_q8 + q8idx * Q8_1_BYTES;
            const int* src4 = (const int*)src;
            int* dst4 = (int*)dst;
            dst4[0] = src4[0]; dst4[1] = src4[1]; dst4[2] = src4[2];
            dst4[3] = src4[3]; dst4[4] = src4[4]; dst4[5] = src4[5];
            dst4[6] = src4[6]; dst4[7] = src4[7]; dst4[8] = src4[8];
        }
        __syncthreads();

        int local_kbx = tid / 8;
        if (kbx_base + local_kbx < n_iq3s_blocks) {
            int kbx = kbx_base + local_kbx;
            const unsigned char* smem_q8_for_block = smem_q8 + local_kbx * 8 * Q8_1_BYTES;

            partial[0] += dot_iq3s_q8_smem(row0_weights + kbx * IQ3S_BYTES,
                                           smem_q8_for_block, iqs);
            if (do_row1) {
                partial[1] += dot_iq3s_q8_smem(row1_weights + kbx * IQ3S_BYTES,
                                               smem_q8_for_block, iqs);
            }
        }
        __syncthreads();
    }

    if (warp_id > 0) {
        tmp_shared[(warp_id - 1) * 2 * WARP_SIZE_V3 + 0 * WARP_SIZE_V3 + lane_id] = partial[0];
        tmp_shared[(warp_id - 1) * 2 * WARP_SIZE_V3 + 1 * WARP_SIZE_V3 + lane_id] = partial[1];
    }
    __syncthreads();
    if (warp_id > 0) return;

    #pragma unroll
    for (int w = 0; w < 3; w++) {
        partial[0] += tmp_shared[w * 2 * WARP_SIZE_V3 + 0 * WARP_SIZE_V3 + lane_id];
        partial[1] += tmp_shared[w * 2 * WARP_SIZE_V3 + 1 * WARP_SIZE_V3 + lane_id];
    }

    partial[0] = warp_reduce_sum(partial[0]);
    partial[1] = warp_reduce_sum(partial[1]);

    if (lane_id == 0) {
        output[k * rows + row0] = partial[0];
    }
    if (lane_id == 1 && do_row1) {
        output[k * rows + row0 + 1] = partial[1];
    }
}

// =====================================================================
// V3 fused gate+up+SiLU: shared-memory Q8_1 + 2 rows per block
//
// Grid:  (ceil(expert_ffn/2), top_k, 1)
// Block: (32, 4, 1) = 128 threads = 4 warps
//
// Each block computes 2 output rows for one expert:
//   output[k * expert_ffn + row0] = silu(gate_dot_0) * up_dot_0
//   output[k * expert_ffn + row0+1] = silu(gate_dot_1) * up_dot_1
//
// Q8_1 input loaded into shared memory ONCE, used for:
//   gate row0, gate row1, up row0, up row1 = 4 dot products from 1 load!
// =====================================================================

extern "C" __global__
__launch_bounds__(128, 1)
void fused_gate_up_silu_iq3s_v3(
    const unsigned char* __restrict__ gate_weights,
    const unsigned char* __restrict__ up_weights,
    float*               __restrict__ output,
    const unsigned char* __restrict__ q8_input,
    const int*           __restrict__ expert_ids,
    int                               cols,
    int                               expert_stride,
    int                               expert_ffn
) {
    const int pair_idx = blockIdx.x;
    const int k        = blockIdx.y;
    const int row0     = 2 * pair_idx;
    const int expert_id = expert_ids[k];
    const int warp_id  = threadIdx.y;
    const int lane_id  = threadIdx.x;
    const int tid      = warp_id * 32 + lane_id;

    const int n_iq3s_blocks = cols >> 8;
    const int iqs = 2 * (tid & 7);

    const unsigned char* gate_row0 = gate_weights
        + (long long)expert_id * expert_stride
        + (long long)row0 * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* gate_row1 = gate_weights
        + (long long)expert_id * expert_stride
        + (long long)(row0 + 1) * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* up_row0 = up_weights
        + (long long)expert_id * expert_stride
        + (long long)row0 * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* up_row1 = up_weights
        + (long long)expert_id * expert_stride
        + (long long)(row0 + 1) * (long long)n_iq3s_blocks * IQ3S_BYTES;

    extern __shared__ char smem_raw[];
    unsigned char* smem_q8 = (unsigned char*)smem_raw;
    // Reduction area: 4 accumulators x 2 rows x 3 warps x 32 lanes
    // We need: [warp-1][accum*2+row][lane] — but simpler: 4 separate arrays
    // Actually let's keep it simple: [3][4][32] floats for warps 1-3, 4 values, 32 lanes
    float* tmp_shared = (float*)(smem_raw + SMEM_Q8_BYTES);

    // 4 accumulators: gate_row0, gate_row1, up_row0, up_row1
    float gate0 = 0.0f, gate1 = 0.0f;
    float up0 = 0.0f, up1 = 0.0f;
    const int do_row1 = (row0 + 1 < expert_ffn) ? 1 : 0;

    for (int kbx_base = 0; kbx_base < n_iq3s_blocks; kbx_base += BLOCKS_PER_ITER) {
        int n_iq3_this_iter = BLOCKS_PER_ITER;
        if (kbx_base + BLOCKS_PER_ITER > n_iq3s_blocks)
            n_iq3_this_iter = n_iq3s_blocks - kbx_base;
        int n_q8_blocks = n_iq3_this_iter * 8;

        for (int q8idx = tid; q8idx < n_q8_blocks; q8idx += 128) {
            const int src_q8_block = (kbx_base * 8) + q8idx;
            const unsigned char* src = q8_input + (long long)src_q8_block * Q8_1_BYTES;
            unsigned char* dst = smem_q8 + q8idx * Q8_1_BYTES;
            const int* src4 = (const int*)src;
            int* dst4 = (int*)dst;
            dst4[0] = src4[0]; dst4[1] = src4[1]; dst4[2] = src4[2];
            dst4[3] = src4[3]; dst4[4] = src4[4]; dst4[5] = src4[5];
            dst4[6] = src4[6]; dst4[7] = src4[7]; dst4[8] = src4[8];
        }
        __syncthreads();

        int local_kbx = tid / 8;
        if (kbx_base + local_kbx < n_iq3s_blocks) {
            int kbx = kbx_base + local_kbx;
            const unsigned char* sq8 = smem_q8 + local_kbx * 8 * Q8_1_BYTES;

            gate0 += dot_iq3s_q8_smem(gate_row0 + kbx * IQ3S_BYTES, sq8, iqs);
            up0   += dot_iq3s_q8_smem(up_row0   + kbx * IQ3S_BYTES, sq8, iqs);

            if (do_row1) {
                gate1 += dot_iq3s_q8_smem(gate_row1 + kbx * IQ3S_BYTES, sq8, iqs);
                up1   += dot_iq3s_q8_smem(up_row1   + kbx * IQ3S_BYTES, sq8, iqs);
            }
        }
        __syncthreads();
    }

    // Warp-level reduction for all 4 accumulators
    gate0 = warp_reduce_sum(gate0);
    gate1 = warp_reduce_sum(gate1);
    up0   = warp_reduce_sum(up0);
    up1   = warp_reduce_sum(up1);

    // We only need the final sum from lane 0. Use a simpler cross-warp reduction.
    // Shared: [4] floats per warp for the 4 accumulators, only lane 0 writes.
    float* warp_results = (float*)(smem_raw + SMEM_Q8_BYTES);
    // Layout: warp_results[warp_id * 4 + accum_idx]

    if (lane_id == 0) {
        warp_results[warp_id * 4 + 0] = gate0;
        warp_results[warp_id * 4 + 1] = gate1;
        warp_results[warp_id * 4 + 2] = up0;
        warp_results[warp_id * 4 + 3] = up1;
    }
    __syncthreads();

    if (tid == 0) {
        float g0 = warp_results[0] + warp_results[4] + warp_results[8] + warp_results[12];
        float g1 = warp_results[1] + warp_results[5] + warp_results[9] + warp_results[13];
        float u0 = warp_results[2] + warp_results[6] + warp_results[10] + warp_results[14];
        float u1 = warp_results[3] + warp_results[7] + warp_results[11] + warp_results[15];

        float silu_g0 = g0 / (1.0f + expf(-g0));
        output[k * expert_ffn + row0] = silu_g0 * u0;

        if (do_row1) {
            float silu_g1 = g1 / (1.0f + expf(-g1));
            output[k * expert_ffn + row0 + 1] = silu_g1 * u1;
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Constants and caching
// ---------------------------------------------------------------------------

const MODULE_NAME_V3: &str = "chimere_iq3s_gemv_v3_02";

const DEQUANT_FUNC: &str = "dequant_iq3s";
const GEMV_FUNC: &str = "gemv_iq3s_q8";
const GEMV_BATCHED_FUNC: &str = "gemv_iq3s_q8_batched";
const GEMV_BATCHED_MULTI_INPUT_FUNC: &str = "gemv_iq3s_q8_batched_multi_input";
const QUANTIZE_FUNC: &str = "quantize_f32_to_q8_1";
const QUANTIZE_BATCHED_FUNC: &str = "quantize_q8_1_batched";
const FUSED_GATE_UP_FUNC: &str = "fused_gate_up_silu_iq3s_v3";

/// IQ3_S block size: imported from parent module.
#[allow(unused_imports)]
use super::IQ3S_BLOCK_BYTES;
const IQ3S_BLOCK_ELEMS: usize = 256;

/// Q8_1 block size.
const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_BLOCK_ELEMS: usize = 32;

/// Shared memory for v3 kernels (Q8_1 staging + reduction).
/// Q8_1 staging: 16 * 8 * 36 = 4608 bytes
/// Cross-warp reduction: 3 * 2 * 32 * 4 = 768 bytes
/// Total: 5376 bytes
const V3_SMEM_BYTES: u32 = 5376;

/// Shared memory for fused gate+up v3:
/// Q8_1 staging: 4608 bytes
/// Warp results: 4 * 4 * 4 = 64 bytes (4 warps x 4 accumulators x 4 bytes)
/// Total: 4672 bytes (use 5376 to be safe with alignment)
const V3_FUSED_SMEM_BYTES: u32 = 5376;

static V3_PTX_CACHE: OnceLock<String> = OnceLock::new();

/// Check if v3 is enabled via CHIMERE_IQ3S_V3=1 environment variable.
pub fn is_v3() -> bool {
    use once_cell::sync::Lazy;
    static V3: Lazy<bool> = Lazy::new(|| {
        let v = std::env::var("CHIMERE_IQ3S_V3")
            .map(|v| v == "1")
            .unwrap_or(false);
        if v {
            eprintln!("[IQ3S_V3] ENABLED — shared mem Q8_1 + 2 rows/block kernels");
        }
        v
    });
    *V3
}

fn load_func(
    dev: &CudaDevice,
    fn_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    super::nvrtc_compile::get_or_load_func(
        dev, fn_name, MODULE_NAME_V3, IQ3S_V3_KERNEL_SRC, &V3_PTX_CACHE,
    )
}

// ---------------------------------------------------------------------------
// Launch configs
// ---------------------------------------------------------------------------

/// GEMV v3: 2 rows per block, so grid = ceil(nrows/2)
fn gemv_launch_config(n_rows: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n_rows + 1) / 2, 1, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: V3_SMEM_BYTES,
    }
}

/// Batched GEMV v3: 2 rows per block
fn gemv_batched_launch_config(rows: u32, top_k: u32) -> LaunchConfig {
    let pairs_per_expert = (rows + 1) / 2;
    LaunchConfig {
        grid_dim: (pairs_per_expert * top_k, 1, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: V3_SMEM_BYTES,
    }
}

/// Batched multi-input GEMV v3: 2 rows per block
fn gemv_multi_input_launch_config(n_rows: u32, top_k: u32) -> LaunchConfig {
    let pairs = (n_rows + 1) / 2;
    LaunchConfig {
        grid_dim: (pairs, top_k, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: V3_SMEM_BYTES,
    }
}

// ---------------------------------------------------------------------------
// Q8_1 helpers
// ---------------------------------------------------------------------------

fn q8_byte_size(n_elements: usize) -> usize {
    debug_assert!(n_elements % Q8_1_BLOCK_ELEMS == 0);
    (n_elements / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES
}

// ---------------------------------------------------------------------------
// Public API: F32 -> Q8_1 quantization (same API as iq3s_gemv.rs)
// ---------------------------------------------------------------------------

pub fn quantize_f32_to_q8_1_gpu(
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<u8>,
    n_elements: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(n_elements % Q8_1_BLOCK_ELEMS == 0);
    let n_blocks = n_elements / Q8_1_BLOCK_ELEMS;
    if n_blocks == 0 { return Ok(()); }

    let (func, stream) = load_func(dev, QUANTIZE_FUNC)?;
    let threads = 256u32;
    let blocks = ((n_blocks as u32) + threads - 1) / threads;
    let n_blocks_i32 = n_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&n_blocks_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 quantize_f32_to_q8_1 launch: {e}")))?;
    Ok(())
}

pub fn quantize_f32_to_q8_1_batched_gpu(
    input_all: &CudaSlice<f32>,
    output_all: &mut CudaSlice<u8>,
    n_per_expert: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(n_per_expert % Q8_1_BLOCK_ELEMS == 0);
    let blocks_per_expert = n_per_expert / Q8_1_BLOCK_ELEMS;
    if blocks_per_expert == 0 || top_k == 0 { return Ok(()); }

    let q8_stride = blocks_per_expert * Q8_1_BLOCK_BYTES;
    let (func, stream) = load_func(dev, QUANTIZE_BATCHED_FUNC)?;

    let threads = 256u32;
    let grid_x = ((blocks_per_expert as u32) + threads - 1) / threads;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, top_k as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let blocks_per_expert_i32 = blocks_per_expert as i32;
    let n_per_expert_i32 = n_per_expert as i32;
    let q8_stride_i32 = q8_stride as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(input_all);
    builder.arg(output_all);
    builder.arg(&blocks_per_expert_i32);
    builder.arg(&n_per_expert_i32);
    builder.arg(&q8_stride_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 quantize_q8_1_batched launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: IQ3_S GEMV v3
// ---------------------------------------------------------------------------

/// IQ3_S GEMV v3: weights[rows, cols] @ input[cols] -> output[rows].
pub fn gemv_iq3s_fused(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0);

    let q8_size = q8_byte_size(cols);
    let stream = dev.cuda_stream();
    let mut q8_buf: CudaSlice<u8> = stream
        .alloc_zeros(q8_size)
        .map_err(|e| candle_core::Error::Msg(format!("v3 q8 alloc: {e}")))?;

    quantize_f32_to_q8_1_gpu(input, &mut q8_buf, cols, dev)?;
    gemv_iq3s_q8_precomputed(weights, &q8_buf, output, rows, cols, dev)
}

/// IQ3_S GEMV v3 at byte offset.
pub fn gemv_iq3s_fused_at_offset(
    base: &CudaView<'_, u8>,
    byte_offset: usize,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let weights = base.slice(byte_offset..);
    gemv_iq3s_fused(&weights, input, output, rows, cols, dev)
}

/// IQ3_S GEMV v3 with pre-quantized Q8_1 input at offset.
pub fn gemv_iq3s_fused_at_offset_q8(
    base: &CudaView<'_, u8>,
    byte_offset: usize,
    q8_buf: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let weights = base.slice(byte_offset..);
    gemv_iq3s_q8_precomputed(&weights, q8_buf, output, rows, cols, dev)
}

/// IQ3_S GEMV v3 with pre-quantized Q8_1 input.
pub fn gemv_iq3s_q8_precomputed(
    weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0);

    let (func, stream) = load_func(dev, GEMV_FUNC)?;
    let cols_i32 = cols as i32;
    let cfg = gemv_launch_config(rows as u32);

    let mut builder = stream.launch_builder(&func);
    builder.arg(weights);
    builder.arg(q8_input);  // dummy f32 ptr
    builder.arg(output);
    builder.arg(&cols_i32);
    builder.arg(q8_input);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 gemv_iq3s_q8 launch: {e}")))?;
    Ok(())
}

/// Batched IQ3_S GEMV v3.
pub fn gemv_iq3s_q8_batched(
    all_weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids_buf: &CudaSlice<i32>,
    cols: usize,
    rows: usize,
    top_k: usize,
    expert_stride: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0);
    if top_k == 0 || rows == 0 { return Ok(()); }

    let (func, stream) = load_func(dev, GEMV_BATCHED_FUNC)?;
    let cfg = gemv_batched_launch_config(rows as u32, top_k as u32);

    let cols_i32 = cols as i32;
    let rows_i32 = rows as i32;
    let expert_stride_i32 = expert_stride as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(all_weights);
    builder.arg(output);
    builder.arg(&cols_i32);
    builder.arg(q8_input);
    builder.arg(expert_ids_buf);
    builder.arg(&rows_i32);
    builder.arg(&expert_stride_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 gemv_iq3s_q8_batched launch: {e}")))?;
    Ok(())
}

/// Batched multi-input IQ3_S GEMV v3.
pub fn gemv_iq3s_q8_batched_multi_input(
    all_weights: &CudaView<'_, u8>,
    all_q8_inputs: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    expert_ids: &CudaSlice<i32>,
    cols: usize,
    rows: usize,
    top_k: usize,
    expert_stride: usize,
    q8_stride: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0);
    if top_k == 0 || rows == 0 { return Ok(()); }

    let (func, stream) = load_func(dev, GEMV_BATCHED_MULTI_INPUT_FUNC)?;
    let cfg = gemv_multi_input_launch_config(rows as u32, top_k as u32);

    let cols_i32 = cols as i32;
    let rows_i32 = rows as i32;
    let expert_stride_i32 = expert_stride as i32;
    let q8_stride_i32 = q8_stride as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(all_weights);
    builder.arg(all_q8_inputs);
    builder.arg(output);
    builder.arg(expert_ids);
    builder.arg(&cols_i32);
    builder.arg(&rows_i32);
    builder.arg(&expert_stride_i32);
    builder.arg(&q8_stride_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 gemv_iq3s_q8_batched_multi_input launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: Fused gate+up+SiLU v3
// ---------------------------------------------------------------------------

/// Fused gate+up+SiLU IQ3_S GEMV v3.
///
/// Computes `output[k * expert_ffn + row] = silu(gate_dot) * up_dot` for all
/// `top_k` experts and all `expert_ffn` output rows. Uses shared-memory Q8_1
/// and 2 rows per block for maximum bandwidth efficiency.
pub fn fused_gate_up_silu_iq3s_v3(
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
    assert!(cols % 256 == 0);
    if top_k == 0 || expert_ffn == 0 { return Ok(()); }

    let (func, stream) = load_func(dev, FUSED_GATE_UP_FUNC)?;

    let pairs = ((expert_ffn as u32) + 1) / 2;
    let cfg = LaunchConfig {
        grid_dim: (pairs, top_k as u32, 1),
        block_dim: (32, 4, 1),
        shared_mem_bytes: V3_FUSED_SMEM_BYTES,
    };

    let cols_i32 = cols as i32;
    let expert_stride_i32 = expert_stride as i32;
    let expert_ffn_i32 = expert_ffn as i32;
    let mut builder = stream.launch_builder(&func);
    builder.arg(gate_weights);
    builder.arg(up_weights);
    builder.arg(output);
    builder.arg(q8_input);
    builder.arg(expert_ids);
    builder.arg(&cols_i32);
    builder.arg(&expert_stride_i32);
    builder.arg(&expert_ffn_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 fused_gate_up_silu_iq3s launch: {e}")))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: Dequantization (pass-through to v3 module's copy)
// ---------------------------------------------------------------------------

pub fn dequant_iq3s_at_offset(
    base: &CudaSlice<u8>,
    byte_offset: usize,
    n_elements: usize,
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<candle_core::Tensor> {
    let candle_core::Device::Cuda(cuda_dev) = device else {
        candle_core::bail!("dequant_iq3s requires CUDA device");
    };
    assert!(n_elements % IQ3S_BLOCK_ELEMS == 0);
    let n_blocks = n_elements / IQ3S_BLOCK_ELEMS;

    let stream = cuda_dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(n_elements)
        .map_err(|e| candle_core::Error::Msg(format!("dequant alloc: {e}")))?;

    let data_view = base.slice(byte_offset..);
    launch_dequant(&data_view, &mut output, n_blocks, cuda_dev)?;

    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
        output, cuda_dev.clone(),
    );
    let tensor = candle_core::Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        candle_core::Shape::from_dims(shape),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

pub fn dequant_iq3s_gpu(
    data: &CudaSlice<u8>,
    n_elements: usize,
    shape: &[usize],
    device: &candle_core::Device,
) -> Result<candle_core::Tensor> {
    dequant_iq3s_at_offset(data, 0, n_elements, shape, device)
}

pub fn dequant_iq3s_into(
    base: &CudaSlice<u8>,
    byte_offset: usize,
    output: &mut CudaSlice<f32>,
    n_elements: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(n_elements % IQ3S_BLOCK_ELEMS == 0);
    let n_blocks = n_elements / IQ3S_BLOCK_ELEMS;
    let data_view = base.slice(byte_offset..);
    launch_dequant(&data_view, output, n_blocks, dev)
}

fn launch_dequant(
    data: &CudaView<'_, u8>,
    output: &mut CudaSlice<f32>,
    n_blocks: usize,
    dev: &CudaDevice,
) -> Result<()> {
    if n_blocks == 0 { return Ok(()); }

    let (func, stream) = load_func(dev, DEQUANT_FUNC)?;
    let threads = 256u32;
    let blocks = ((n_blocks as u32) + threads - 1) / threads;
    let n_blocks_i32 = n_blocks as i32;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(data);
    builder.arg(output);
    builder.arg(&n_blocks_i32);
    unsafe { builder.launch(cfg) }
        .map_err(|e| candle_core::Error::Msg(format!("v3 dequant_iq3s launch: {e}")))?;
    Ok(())
}
