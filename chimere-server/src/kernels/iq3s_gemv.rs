//! IQ3_S GEMV + Q8_1 quantization + dequantization CUDA kernels.
//!
//! This module provides:
//! - `gemv_iq3s_fused`: IQ3_S GEMV with on-the-fly Q8_1 quantization of input
//! - `gemv_iq3s_fused_at_offset`: Same but reading weights from a byte offset
//! - `gemv_iq3s_fused_at_offset_q8`: Same but with pre-quantized Q8_1 input
//! - `quantize_f32_to_q8_1_gpu`: F32 -> Q8_1 quantization kernel
//! - `gemv_iq3s_q8_precomputed`: GEMV with pre-quantized Q8_1 input
//! - `gemv_iq3s_q8_batched`: Batched GEMV for multiple experts in one launch (shared input)
//! - `gemv_iq3s_q8_batched_multi_input`: Batched down GEMV — different Q8_1 input per expert
//! - `dequant_iq3s_at_offset`: IQ3_S -> F32 dequantization at byte offset
//! - `dequant_iq3s_gpu`: IQ3_S -> F32 dequantization from CudaSlice
//! - `dequant_iq3s_into`: IQ3_S -> F32 dequantization into pre-allocated buffer
//!
//! All kernels use the cubin path when available (build.rs compiled), falling back
//! to NVRTC runtime compilation.

use candle_core::cuda_backend::cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg,
};
use candle_core::cuda_backend::CudaDevice;
use candle_core::{Device, Result, Shape, Tensor};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// CUDA source for IQ3_S kernels (dequant + GEMV + Q8_1 quantize)
//
// This is the subset of chimere_kernels.cu needed for IQ3_S operations.
// When the cubin is available (build.rs), this source is only used as NVRTC
// fallback.
// ---------------------------------------------------------------------------

const IQ3S_KERNEL_SRC: &str = r#"
// f16 -> f32 via hardware cvt.f32.f16 (single instruction, no branches)
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

// IQ3_S grid constant — 512 entries
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
// IQ3_S dequantisation kernel
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
            unsigned int sb = (unsigned int)block[signs_off + l];

            // SIMD sign expansion for g1 (bits 0-3) and g2 (bits 4-7)
            unsigned int signs_lo = ((sb & 0x01) << 7) | ((sb & 0x02) << 14) | ((sb & 0x04) << 21) | ((sb & 0x08) << 28);
            unsigned int signs_hi = ((sb & 0x10) << 3) | ((sb & 0x20) << 10) | ((sb & 0x40) << 17) | ((sb & 0x80) << 24);
            unsigned int mask_lo = __vcmpne4(signs_lo, 0);
            unsigned int mask_hi = __vcmpne4(signs_hi, 0);
            int sg1 = __vsub4(g1 ^ mask_lo, mask_lo);
            int sg2 = __vsub4(g2 ^ mask_hi, mask_hi);

            out[out_off+0] = db1 * (float)(signed char)(sg1);
            out[out_off+1] = db1 * (float)(signed char)(sg1 >> 8);
            out[out_off+2] = db1 * (float)(signed char)(sg1 >> 16);
            out[out_off+3] = db1 * (float)(signed char)(sg1 >> 24);
            out[out_off+4] = db1 * (float)(signed char)(sg2);
            out[out_off+5] = db1 * (float)(signed char)(sg2 >> 8);
            out[out_off+6] = db1 * (float)(signed char)(sg2 >> 16);
            out[out_off+7] = db1 * (float)(signed char)(sg2 >> 24);
            out_off += 8;
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
            unsigned int sb = (unsigned int)block[signs_off + l];

            // SIMD sign expansion for g1 (bits 0-3) and g2 (bits 4-7)
            unsigned int signs_lo = ((sb & 0x01) << 7) | ((sb & 0x02) << 14) | ((sb & 0x04) << 21) | ((sb & 0x08) << 28);
            unsigned int signs_hi = ((sb & 0x10) << 3) | ((sb & 0x20) << 10) | ((sb & 0x40) << 17) | ((sb & 0x80) << 24);
            unsigned int mask_lo = __vcmpne4(signs_lo, 0);
            unsigned int mask_hi = __vcmpne4(signs_hi, 0);
            int sg1 = __vsub4(g1 ^ mask_lo, mask_lo);
            int sg2 = __vsub4(g2 ^ mask_hi, mask_hi);

            out[out_off+0] = db2 * (float)(signed char)(sg1);
            out[out_off+1] = db2 * (float)(signed char)(sg1 >> 8);
            out[out_off+2] = db2 * (float)(signed char)(sg1 >> 16);
            out[out_off+3] = db2 * (float)(signed char)(sg1 >> 24);
            out[out_off+4] = db2 * (float)(signed char)(sg2);
            out[out_off+5] = db2 * (float)(signed char)(sg2 >> 8);
            out[out_off+6] = db2 * (float)(signed char)(sg2 >> 16);
            out[out_off+7] = db2 * (float)(signed char)(sg2 >> 24);
            out_off += 8;
        }
        qh_off    += 2;
        qs_off    += 8;
        signs_off += 4;
    }
}

// =====================================================================
// Q8_1 quantization kernel (for IQ3_S GEMV)
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

// Batched Q8_1 quantization: quantize top_k vectors in a single launch.
// input_all:  [top_k * n_per_expert] f32 contiguous
// q8_out_all: [top_k * q8_stride] u8 contiguous
// Grid: (ceil(blocks_per_expert / blockDim.x), top_k, 1)
// Each row (blockIdx.y = k) handles one expert's quantization.
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
// IQ3_S GEMV with Q8_1+dp4a
// =====================================================================

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}
__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// Exact ggml vec_dot_iq3_s_q8_1 port.
// 'iqs' = ggml-style sub-group index: 0,2,4,...,14.
// Defers f16 conversion to after the integer loop to reduce register pressure.
__device__ __forceinline__ float dot_iq3s_q8_ggml(
    const unsigned char* __restrict__ bq3,   // IQ3_S block pointer (110 bytes)
    const unsigned char* __restrict__ bq8,   // Q8_1 block pointer (36 bytes)
    int iqs
) {
    // qs: 2x int32 = 8 bytes (ggml make_int2 pattern)
    const int qs_lo = get_int_b2(bq3 + 2, iqs);
    const int qs_hi = get_int_b2(bq3 + 2, iqs + 1);

    const int qh = bq3[66 + iqs/2];

    const int signs32 = get_int_b2(bq3 + 74, iqs/2);
    const unsigned char* sp = (const unsigned char*)&signs32;

    int sumi = 0;

    // l=0: qs_lo bytes 0,1
    {
        const int grid0 = iq3s_grid[(qs_lo & 0xFF)        | ((qh << 8) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 8) & 0xFF) | ((qh << 7) & 0x100)];
        const int s0 = __vcmpne4(((sp[0] & 0x03) << 7) | ((sp[0] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[0] & 0x30) << 3) | ((sp[0] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 0), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 1), sumi);
    }
    // l=1: qs_lo bytes 2,3
    {
        const int grid0 = iq3s_grid[((qs_lo >> 16) & 0xFF) | ((qh << 6) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 24) & 0xFF) | ((qh << 5) & 0x100)];
        const int s0 = __vcmpne4(((sp[1] & 0x03) << 7) | ((sp[1] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[1] & 0x30) << 3) | ((sp[1] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 2), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 3), sumi);
    }
    // l=2: qs_hi bytes 0,1
    {
        const int grid0 = iq3s_grid[(qs_hi & 0xFF)        | ((qh << 4) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 8) & 0xFF) | ((qh << 3) & 0x100)];
        const int s0 = __vcmpne4(((sp[2] & 0x03) << 7) | ((sp[2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[2] & 0x30) << 3) | ((sp[2] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 4), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 5), sumi);
    }
    // l=3: qs_hi bytes 2,3
    {
        const int grid0 = iq3s_grid[((qs_hi >> 16) & 0xFF) | ((qh << 2) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 24) & 0xFF) | ((qh << 1) & 0x100)];
        const int s0 = __vcmpne4(((sp[3] & 0x03) << 7) | ((sp[3] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[3] & 0x30) << 3) | ((sp[3] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 6), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 7), sumi);
    }

    sumi *= 1 + 2 * ((bq3[106 + iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    // Defer f16 conversion: compute d_iq3 * d_q8 only after integer loop finishes.
    // This keeps f32 registers free during the dp4a work.
    const float d = f16_to_f32(*(const unsigned short*)bq3)
                  * f16_to_f32(*(const unsigned short*)bq8);
    return d * (float)sumi;
}

// IQ3_S GEMV kernel — ggml-compatible thread mapping.
// Uses 4 warps (128 threads), VDR=2, qi=16.
// Each thread handles one sub-group (32 elements) per IQ3_S block.
// Thread mapping: kbx = tid/8 (IQ3_S block), iqs = 2*(tid%8) (sub-group).
extern "C" __global__ __launch_bounds__(128)
void gemv_iq3s_q8(
    const unsigned char* __restrict__ weights,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    int                              cols,
    const unsigned char* __restrict__ q8_input
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;

    // ggml-style thread mapping: qi=16, vdr=2 -> qi/vdr=8
    // Each group of 8 consecutive threads processes one IQ3_S block
    const int iqs = 2 * (tid & 7);               // sub-group index (0,2,4,...,14)
    const int NWARPS = 4;
    const int WARP_SIZE = 32;
    const int blocks_per_iter = 2 * NWARPS * WARP_SIZE / 16;  // = 16

    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;

    const unsigned char* row_weights =
        weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;

    float sum = 0.0f;

    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += blocks_per_iter) {
        const unsigned char* bq3 = row_weights + kbx * IQ3S_BYTES;
        // Each IQ3_S block (256 elems) maps to 8 Q8_1 blocks (32 elems each)
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;

        sum += dot_iq3s_q8_ggml(bq3, bq8, iqs);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction via shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        output[row] = total;
    }
}

// Batched version: one row-block per (expert_k, output_row) pair.
// expert_ids[blockIdx.x / rows] gives the expert index.
// expert_stride is the byte offset between consecutive experts.
extern "C" __global__ __launch_bounds__(128)
void gemv_iq3s_q8_batched(
    const unsigned char* __restrict__ all_weights,
    float*               __restrict__ output,
    int                               cols,
    const unsigned char* __restrict__ q8_input,
    const int*           __restrict__ expert_ids,
    int                               rows,
    int                               expert_stride
) {
    const int global_row = blockIdx.x;
    const int k   = global_row / rows;
    const int row = global_row % rows;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;

    const int iqs = 2 * (tid & 7);
    const int NWARPS = 4;
    const int WARP_SIZE = 32;
    const int blocks_per_iter = 2 * NWARPS * WARP_SIZE / 16;

    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;

    int expert_id = expert_ids[k];
    const unsigned char* expert_weights = all_weights + (long long)expert_id * expert_stride;
    const unsigned char* row_weights = expert_weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;

    float sum = 0.0f;

    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += blocks_per_iter) {
        const unsigned char* bq3 = row_weights + kbx * IQ3S_BYTES;
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;

        sum += dot_iq3s_q8_ggml(bq3, bq8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        output[k * rows + row] = total;
    }
}

// Batched down GEMV: different Q8_1 input per expert, single launch.
// Grid: (rows, top_k, 1) = (hidden_size, top_k, 1)
// Block: (128, 1, 1)
//
// Each block computes: output[k][row] = down_weights[expert_id][row] . intermediate[k]
//
// all_weights      : concatenated IQ3_S down-projection weights for ALL experts
// all_q8_inputs    : concatenated Q8_1 buffers [top_k * q8_stride bytes]
//                    all_q8_inputs[k * q8_stride .. (k+1)*q8_stride] = Q8_1 of intermediate[k]
// output           : [top_k * rows] f32, stored as output[k * rows + row]
// expert_ids       : [top_k] int32, expert index for each k
// cols             : expert_ffn (intermediate dimension, must be multiple of 256)
// rows             : hidden_size (output dimension per expert)
// expert_stride    : bytes per expert in all_weights
// q8_stride        : bytes per Q8_1 input = (cols/32)*36
extern "C" __global__ __launch_bounds__(128)
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
    const int row = blockIdx.x;
    const int k   = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;

    const int iqs = 2 * (tid & 7);
    const int NWARPS = 4;
    const int WARP_SIZE = 32;
    const int blocks_per_iter = 2 * NWARPS * WARP_SIZE / 16;

    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;

    int expert_id = expert_ids[k];
    const unsigned char* expert_weights = all_weights + (long long)expert_id * expert_stride;
    const unsigned char* row_weights = expert_weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;
    // Per-expert Q8_1 input: each expert k has its own intermediate vector
    const unsigned char* q8_input = all_q8_inputs + (long long)k * q8_stride;

    float sum = 0.0f;

    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += blocks_per_iter) {
        const unsigned char* bq3 = row_weights + kbx * IQ3S_BYTES;
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;

        sum += dot_iq3s_q8_ggml(bq3, bq8, iqs);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        output[k * rows + row] = total;
    }
}
"#;

// ---------------------------------------------------------------------------
// V2 kernel source: ggml MMVQ-faithful IQ3_S GEMV
//
// Key differences from v1:
// - __launch_bounds__(128, 1) — the ",1" tells compiler to optimize for
//   1 block/SM, allowing up to 512 regs/thread (no spill)
// - 2D block dims (32, 4, 1): threadIdx.y = warp_id, threadIdx.x = lane_id
// - ggml-style cross-warp reduction: warps 1-3 write to shared[warp-1][lane],
//   warp 0 sums all, then warp_reduce_sum within warp 0
// - Same dot product core (already ggml-faithful in v1)
//
// Toggle: CHIMERE_GEMV_V2=1 (default: v1 for backward compatibility)
// ---------------------------------------------------------------------------

const IQ3S_KERNEL_V2_SRC: &str = r#"
// f16 -> f32 via hardware cvt.f32.f16 (single instruction, no branches)
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

// IQ3_S grid constant — 512 entries (identical to v1)
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
// Q8_1 quantization kernel (shared with v1 — identical)
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
// IQ3_S dequantisation kernel (shared with v1 — identical)
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
            unsigned int sb = (unsigned int)block[signs_off + l];

            // SIMD sign expansion for g1 (bits 0-3) and g2 (bits 4-7)
            unsigned int signs_lo = ((sb & 0x01) << 7) | ((sb & 0x02) << 14) | ((sb & 0x04) << 21) | ((sb & 0x08) << 28);
            unsigned int signs_hi = ((sb & 0x10) << 3) | ((sb & 0x20) << 10) | ((sb & 0x40) << 17) | ((sb & 0x80) << 24);
            unsigned int mask_lo = __vcmpne4(signs_lo, 0);
            unsigned int mask_hi = __vcmpne4(signs_hi, 0);
            int sg1 = __vsub4(g1 ^ mask_lo, mask_lo);
            int sg2 = __vsub4(g2 ^ mask_hi, mask_hi);

            out[out_off+0] = db1 * (float)(signed char)(sg1);
            out[out_off+1] = db1 * (float)(signed char)(sg1 >> 8);
            out[out_off+2] = db1 * (float)(signed char)(sg1 >> 16);
            out[out_off+3] = db1 * (float)(signed char)(sg1 >> 24);
            out[out_off+4] = db1 * (float)(signed char)(sg2);
            out[out_off+5] = db1 * (float)(signed char)(sg2 >> 8);
            out[out_off+6] = db1 * (float)(signed char)(sg2 >> 16);
            out[out_off+7] = db1 * (float)(signed char)(sg2 >> 24);
            out_off += 8;
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
            unsigned int sb = (unsigned int)block[signs_off + l];

            // SIMD sign expansion for g1 (bits 0-3) and g2 (bits 4-7)
            unsigned int signs_lo = ((sb & 0x01) << 7) | ((sb & 0x02) << 14) | ((sb & 0x04) << 21) | ((sb & 0x08) << 28);
            unsigned int signs_hi = ((sb & 0x10) << 3) | ((sb & 0x20) << 10) | ((sb & 0x40) << 17) | ((sb & 0x80) << 24);
            unsigned int mask_lo = __vcmpne4(signs_lo, 0);
            unsigned int mask_hi = __vcmpne4(signs_hi, 0);
            int sg1 = __vsub4(g1 ^ mask_lo, mask_lo);
            int sg2 = __vsub4(g2 ^ mask_hi, mask_hi);

            out[out_off+0] = db2 * (float)(signed char)(sg1);
            out[out_off+1] = db2 * (float)(signed char)(sg1 >> 8);
            out[out_off+2] = db2 * (float)(signed char)(sg1 >> 16);
            out[out_off+3] = db2 * (float)(signed char)(sg1 >> 24);
            out[out_off+4] = db2 * (float)(signed char)(sg2);
            out[out_off+5] = db2 * (float)(signed char)(sg2 >> 8);
            out[out_off+6] = db2 * (float)(signed char)(sg2 >> 16);
            out[out_off+7] = db2 * (float)(signed char)(sg2 >> 24);
            out_off += 8;
        }
        qh_off    += 2;
        qs_off    += 8;
        signs_off += 4;
    }
}

// =====================================================================
// IQ3_S GEMV v2 — ggml MMVQ faithful port
//
// Block: (32, 4, 1) = 128 threads = 4 warps
// __launch_bounds__(128, 1): optimize for 1 block/SM, max registers
// Thread mapping: 8 threads per IQ3_S block, 16 blocks per iteration
// Vectorized loads: get_int_b2 (2x u16 -> u32), get_int_b4 (direct u32)
// =====================================================================

__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}
__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

// Exact ggml vec_dot_iq3_s_q8_1 port.
// 'iqs' = ggml-style sub-group index: 0,2,4,...,14.
// Each call processes 32 elements (one sub-group pair) within an IQ3_S block.
__device__ __forceinline__ float dot_iq3s_q8_v2(
    const unsigned char* __restrict__ bq3,   // IQ3_S block pointer (110 bytes)
    const unsigned char* __restrict__ bq8,   // Q8_1 block for this sub-group (36 bytes)
    int iqs
) {
    // Vectorized qs loads: 2x int32 = 8 bytes (ggml make_int2 pattern)
    const int qs_lo = get_int_b2(bq3 + 2, iqs);
    const int qs_hi = get_int_b2(bq3 + 2, iqs + 1);

    const int qh = bq3[66 + iqs/2];

    // Vectorized signs load: 4 bytes as one int32
    const int signs32 = get_int_b2(bq3 + 74, iqs/2);
    const unsigned char* sp = (const unsigned char*)&signs32;

    int sumi = 0;

    // Unrolled loop: 4 iterations processing 2 grid entries each = 32 elements
    // l=0: qs_lo bytes 0,1
    {
        const int grid0 = iq3s_grid[(qs_lo & 0xFF)        | ((qh << 8) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 8) & 0xFF) | ((qh << 7) & 0x100)];
        const int s0 = __vcmpne4(((sp[0] & 0x03) << 7) | ((sp[0] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[0] & 0x30) << 3) | ((sp[0] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 0), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 1), sumi);
    }
    // l=1: qs_lo bytes 2,3
    {
        const int grid0 = iq3s_grid[((qs_lo >> 16) & 0xFF) | ((qh << 6) & 0x100)];
        const int grid1 = iq3s_grid[((qs_lo >> 24) & 0xFF) | ((qh << 5) & 0x100)];
        const int s0 = __vcmpne4(((sp[1] & 0x03) << 7) | ((sp[1] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[1] & 0x30) << 3) | ((sp[1] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 2), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 3), sumi);
    }
    // l=2: qs_hi bytes 0,1
    {
        const int grid0 = iq3s_grid[(qs_hi & 0xFF)        | ((qh << 4) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 8) & 0xFF) | ((qh << 3) & 0x100)];
        const int s0 = __vcmpne4(((sp[2] & 0x03) << 7) | ((sp[2] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[2] & 0x30) << 3) | ((sp[2] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 4), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 5), sumi);
    }
    // l=3: qs_hi bytes 2,3
    {
        const int grid0 = iq3s_grid[((qs_hi >> 16) & 0xFF) | ((qh << 2) & 0x100)];
        const int grid1 = iq3s_grid[((qs_hi >> 24) & 0xFF) | ((qh << 1) & 0x100)];
        const int s0 = __vcmpne4(((sp[3] & 0x03) << 7) | ((sp[3] & 0x0C) << 21), 0);
        const int s1 = __vcmpne4(((sp[3] & 0x30) << 3) | ((sp[3] & 0xC0) << 17), 0);
        sumi = __dp4a((int)__vsub4(grid0 ^ s0, s0), get_int_b4(bq8 + 4, 6), sumi);
        sumi = __dp4a((int)__vsub4(grid1 ^ s1, s1), get_int_b4(bq8 + 4, 7), sumi);
    }

    // Scale: nibble from scales[4] array
    sumi *= 1 + 2 * ((bq3[106 + iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    // d_iq3 * d_q8 — defer f16 conversion to after integer loop
    const float d = f16_to_f32(*(const unsigned short*)bq3)
                  * f16_to_f32(*(const unsigned short*)bq8);
    return d * (float)sumi;
}

// -----------------------------------------------------------------
// ggml MMVQ warp reduction helper
// -----------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_v2(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// -----------------------------------------------------------------
// V2 GEMV kernel: single input vector, single expert
//
// Signature matches v1 gemv_iq3s_q8 exactly (same name, same args).
// -----------------------------------------------------------------
extern "C" __global__
__launch_bounds__(128, 1)
void gemv_iq3s_q8(
    const unsigned char* __restrict__ weights,
    const float*         __restrict__ input,   // unused when q8_input is valid
    float*               __restrict__ output,
    int                              cols,
    const unsigned char* __restrict__ q8_input
) {
    const int row      = blockIdx.x;
    const int warp_id  = threadIdx.y;          // 0..3 (2D block: 32x4)
    const int lane_id  = threadIdx.x;          // 0..31
    const int tid      = warp_id * 32 + lane_id;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;       // cols / 256

    // ggml thread mapping: qi=16, vdr=2, qi/vdr=8
    // 8 threads cooperate on each IQ3_S block
    // 128 threads / 8 = 16 blocks per iteration
    const int iqs             = 2 * (tid & 7);   // 0,2,4,...,14
    const int BLOCKS_PER_ITER = 16;               // 2 * 4 * 32 / 16

    const unsigned char* row_weights =
        weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;

    float partial = 0.0f;

    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += BLOCKS_PER_ITER) {
        const unsigned char* bq3 = row_weights + kbx * IQ3S_BYTES;
        // Each IQ3_S block (256 elems) maps to 8 Q8_1 blocks (32 elems each)
        // iqs/2 selects which Q8_1 block within the group of 8
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;

        partial += dot_iq3s_q8_v2(bq3, bq8, iqs);
    }

    // --- ggml MMVQ cross-warp reduction ---
    // Shared memory: [nwarps-1][warp_size] = [3][32] floats = 384 bytes
    __shared__ float tmp_shared[3][32];

    // Warps 1-3 write their partial sums
    if (warp_id > 0) {
        tmp_shared[warp_id - 1][lane_id] = partial;
    }
    __syncthreads();

    // Only warp 0 continues
    if (warp_id > 0) return;

    // Warp 0: accumulate partials from warps 1-3
    #pragma unroll
    for (int w = 0; w < 3; w++) {
        partial += tmp_shared[w][lane_id];
    }

    // Butterfly warp reduction within warp 0
    partial = warp_reduce_sum_v2(partial);

    if (lane_id == 0) {
        output[row] = partial;
    }
}

// -----------------------------------------------------------------
// V2 batched GEMV: shared Q8_1 input, multiple experts
// -----------------------------------------------------------------
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
    const int global_row = blockIdx.x;
    const int k          = global_row / rows;
    const int row        = global_row % rows;
    const int warp_id    = threadIdx.y;
    const int lane_id    = threadIdx.x;
    const int tid        = warp_id * 32 + lane_id;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;

    const int iqs             = 2 * (tid & 7);
    const int BLOCKS_PER_ITER = 16;

    int expert_id = expert_ids[k];
    const unsigned char* expert_weights = all_weights + (long long)expert_id * expert_stride;
    const unsigned char* row_weights = expert_weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;

    float partial = 0.0f;

    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += BLOCKS_PER_ITER) {
        const unsigned char* bq3 = row_weights + kbx * IQ3S_BYTES;
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;
        partial += dot_iq3s_q8_v2(bq3, bq8, iqs);
    }

    __shared__ float tmp_shared[3][32];

    if (warp_id > 0) {
        tmp_shared[warp_id - 1][lane_id] = partial;
    }
    __syncthreads();
    if (warp_id > 0) return;

    #pragma unroll
    for (int w = 0; w < 3; w++) {
        partial += tmp_shared[w][lane_id];
    }
    partial = warp_reduce_sum_v2(partial);

    if (lane_id == 0) {
        output[k * rows + row] = partial;
    }
}

// -----------------------------------------------------------------
// V2 batched multi-input GEMV: different Q8_1 per expert
// -----------------------------------------------------------------
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
    const int row     = blockIdx.x;
    const int k       = blockIdx.y;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;
    const int tid     = warp_id * 32 + lane_id;

    const int IQ3S_BYTES    = 110;
    const int Q8_1_BYTES    = 36;
    const int n_iq3s_blocks = cols >> 8;

    const int iqs             = 2 * (tid & 7);
    const int BLOCKS_PER_ITER = 16;

    int expert_id = expert_ids[k];
    const unsigned char* expert_weights = all_weights + (long long)expert_id * expert_stride;
    const unsigned char* row_weights = expert_weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;
    const unsigned char* q8_input = all_q8_inputs + (long long)k * q8_stride;

    float partial = 0.0f;

    for (int kbx = tid / 8; kbx < n_iq3s_blocks; kbx += BLOCKS_PER_ITER) {
        const unsigned char* bq3 = row_weights + kbx * IQ3S_BYTES;
        const unsigned char* bq8 = q8_input + (long long)(kbx * 8 + iqs/2) * Q8_1_BYTES;
        partial += dot_iq3s_q8_v2(bq3, bq8, iqs);
    }

    __shared__ float tmp_shared[3][32];

    if (warp_id > 0) {
        tmp_shared[warp_id - 1][lane_id] = partial;
    }
    __syncthreads();
    if (warp_id > 0) return;

    #pragma unroll
    for (int w = 0; w < 3; w++) {
        partial += tmp_shared[w][lane_id];
    }
    partial = warp_reduce_sum_v2(partial);

    if (lane_id == 0) {
        output[k * rows + row] = partial;
    }
}
"#;

const MODULE_NAME: &str = "chimere_iq3s_gemv_v25";
const MODULE_NAME_V2: &str = "chimere_iq3s_gemv_v2_02";

// Kernel function names (must match the extern "C" names in CUDA source)
const DEQUANT_FUNC: &str = "dequant_iq3s";
const GEMV_FUNC: &str = "gemv_iq3s_q8";
const GEMV_BATCHED_FUNC: &str = "gemv_iq3s_q8_batched";
const GEMV_BATCHED_MULTI_INPUT_FUNC: &str = "gemv_iq3s_q8_batched_multi_input";
const QUANTIZE_FUNC: &str = "quantize_f32_to_q8_1";
const QUANTIZE_BATCHED_FUNC: &str = "quantize_q8_1_batched";

const IQ3S_BLOCK_ELEMS: usize = 256;
const IQ3S_BLOCK_BYTES: usize = 110;

/// Q8_1 block size: 36 bytes per 32 elements (2 bytes d + 2 bytes s + 32 bytes qs).
const Q8_1_BLOCK_BYTES: usize = 36;
const Q8_1_BLOCK_ELEMS: usize = 32;

/// V2 shared memory: statically allocated in kernel as __shared__ float[3][32].
/// No dynamic shared memory needed (set shared_mem_bytes=0 in launch config).
const V2_SHARED_MEM_BYTES: u32 = 0;

static PTX_CACHE: OnceLock<String> = OnceLock::new();
static V2_PTX_CACHE: OnceLock<String> = OnceLock::new();

/// Check if GEMV v2 is enabled via CHIMERE_GEMV_V2=1 environment variable.
fn is_v2() -> bool {
    use once_cell::sync::Lazy;
    static V2: Lazy<bool> = Lazy::new(|| {
        std::env::var("CHIMERE_GEMV_V2")
            .map(|v| v == "1")
            .unwrap_or(false)
    });
    *V2
}

#[allow(dead_code)]
fn get_ptx() -> &'static str {
    super::nvrtc_compile::compile_and_cache(IQ3S_KERNEL_SRC, &PTX_CACHE)
}

/// Load a kernel function from cubin (fast) or NVRTC (fallback).
/// When CHIMERE_GEMV_V2=1, GEMV kernels are loaded from the v2 source.
/// Non-GEMV functions (dequant, quantize) always use v1 source.
fn load_func(
    dev: &CudaDevice,
    fn_name: &str,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    // GEMV kernels: use v2 source when toggled on
    if is_v2() && (fn_name == GEMV_FUNC
                   || fn_name == GEMV_BATCHED_FUNC
                   || fn_name == GEMV_BATCHED_MULTI_INPUT_FUNC
                   || fn_name == QUANTIZE_FUNC
                   || fn_name == QUANTIZE_BATCHED_FUNC
                   || fn_name == DEQUANT_FUNC) {
        return super::nvrtc_compile::get_or_load_func(
            dev, fn_name, MODULE_NAME_V2, IQ3S_KERNEL_V2_SRC, &V2_PTX_CACHE,
        );
    }
    // Default: v1 source (or cubin)
    super::nvrtc_compile::get_or_load_func(dev, fn_name, MODULE_NAME, IQ3S_KERNEL_SRC, &PTX_CACHE)
}

/// Return the appropriate LaunchConfig for GEMV kernels.
///
/// V1: block_dim=(128,1,1), shared_mem=16 bytes (4 floats for warp_sums)
/// V2: block_dim=(32,4,1),  shared_mem=384 bytes (ggml cross-warp reduction)
fn gemv_launch_config(n_rows: u32) -> LaunchConfig {
    if is_v2() {
        LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: V2_SHARED_MEM_BYTES,
        }
    } else {
        LaunchConfig {
            grid_dim: (n_rows, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 16,
        }
    }
}

/// Return the appropriate LaunchConfig for batched GEMV (1D grid).
fn gemv_batched_launch_config(total_rows: u32) -> LaunchConfig {
    if is_v2() {
        LaunchConfig {
            grid_dim: (total_rows, 1, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: V2_SHARED_MEM_BYTES,
        }
    } else {
        LaunchConfig {
            grid_dim: (total_rows, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 16,
        }
    }
}

/// Return the appropriate LaunchConfig for batched multi-input GEMV (2D grid).
fn gemv_multi_input_launch_config(n_rows: u32, top_k: u32) -> LaunchConfig {
    if is_v2() {
        LaunchConfig {
            grid_dim: (n_rows, top_k, 1),
            block_dim: (32, 4, 1),
            shared_mem_bytes: V2_SHARED_MEM_BYTES,
        }
    } else {
        LaunchConfig {
            grid_dim: (n_rows, top_k, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 16,
        }
    }
}

// ---------------------------------------------------------------------------
// Q8_1 buffer cache — amortized allocation via thread_local
// ---------------------------------------------------------------------------

/// Return the Q8_1 byte size for `n_elements` f32 values (must be multiple of 32).
fn q8_byte_size(n_elements: usize) -> usize {
    debug_assert!(n_elements % Q8_1_BLOCK_ELEMS == 0);
    (n_elements / Q8_1_BLOCK_ELEMS) * Q8_1_BLOCK_BYTES
}

// ---------------------------------------------------------------------------
// Public API: F32 -> Q8_1 quantization
// ---------------------------------------------------------------------------

/// Quantize `n_elements` f32 values to Q8_1 format on GPU.
///
/// `input` must be a CudaView of at least `n_elements` f32 values.
/// `output` must be a CudaSlice<u8> of at least `q8_byte_size(n_elements)` bytes.
/// `n_elements` must be a multiple of 32.
pub fn quantize_f32_to_q8_1_gpu(
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<u8>,
    n_elements: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(n_elements % Q8_1_BLOCK_ELEMS == 0, "n_elements must be multiple of 32");
    let n_blocks = n_elements / Q8_1_BLOCK_ELEMS;
    if n_blocks == 0 {
        return Ok(());
    }

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
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("quantize_f32_to_q8_1 launch: {e}")))?;

    Ok(())
}

/// Batched Q8_1 quantization: quantize `top_k` contiguous vectors in a single launch.
///
/// `input_all`: contiguous f32 buffer of `[top_k * n_per_expert]` elements.
///   Layout: expert 0's n_per_expert floats, then expert 1's, etc.
/// `output_all`: contiguous u8 buffer of `[top_k * q8_stride]` bytes.
///   Each expert's Q8_1 output occupies `q8_stride = (n_per_expert / 32) * 36` bytes.
/// `n_per_expert`: number of f32 elements per expert (must be multiple of 32).
/// `top_k`: number of experts.
///
/// Replaces `top_k` individual `quantize_f32_to_q8_1_gpu` launches + dtod copies
/// with a single kernel launch.
pub fn quantize_f32_to_q8_1_batched_gpu(
    input_all: &CudaSlice<f32>,
    output_all: &mut CudaSlice<u8>,
    n_per_expert: usize,
    top_k: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(n_per_expert % Q8_1_BLOCK_ELEMS == 0, "n_per_expert must be multiple of 32");
    let blocks_per_expert = n_per_expert / Q8_1_BLOCK_ELEMS;
    if blocks_per_expert == 0 || top_k == 0 {
        return Ok(());
    }

    let q8_stride = blocks_per_expert * Q8_1_BLOCK_BYTES;  // bytes per expert in output

    let (func, stream) = load_func(dev, QUANTIZE_BATCHED_FUNC)?;

    let threads = 256u32;
    let grid_x = ((blocks_per_expert as u32) + threads - 1) / threads;
    let blocks_per_expert_i32 = blocks_per_expert as i32;
    let n_per_expert_i32 = n_per_expert as i32;
    let q8_stride_i32 = q8_stride as i32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, top_k as u32, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = stream.launch_builder(&func);
    builder.arg(input_all);
    builder.arg(output_all);
    builder.arg(&blocks_per_expert_i32);
    builder.arg(&n_per_expert_i32);
    builder.arg(&q8_stride_i32);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("quantize_q8_1_batched launch: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: IQ3_S GEMV (fused Q8_1 quantize + dp4a GEMV)
// ---------------------------------------------------------------------------

/// IQ3_S GEMV: weights[rows, cols] @ input[cols] -> output[rows].
///
/// The input is quantized to Q8_1 on-the-fly, then the dp4a dot product
/// is computed against the IQ3_S weights.
///
/// `weights`: flat IQ3_S-encoded weight bytes, row-major.
/// `input`: f32 input vector of length `cols`.
/// `output`: pre-allocated f32 output buffer of length `rows`.
/// `rows`, `cols`: matrix dimensions (cols must be multiple of 256).
pub fn gemv_iq3s_fused(
    weights: &CudaView<'_, u8>,
    input: &CudaView<'_, f32>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0, "cols must be multiple of 256");

    // Quantize input to Q8_1 into a temporary buffer
    let q8_size = q8_byte_size(cols);
    let stream = dev.cuda_stream();
    let mut q8_buf: CudaSlice<u8> = stream
        .alloc_zeros(q8_size)
        .map_err(|e| candle_core::Error::Msg(format!("q8 alloc: {e}")))?;

    quantize_f32_to_q8_1_gpu(input, &mut q8_buf, cols, dev)?;

    // Launch GEMV kernel
    let (func, stream) = load_func(dev, GEMV_FUNC)?;

    let cols_i32 = cols as i32;
    let cfg = gemv_launch_config(rows as u32);

    let mut builder = stream.launch_builder(&func);
    builder.arg(weights);
    builder.arg(input);
    builder.arg(output);
    builder.arg(&cols_i32);
    builder.arg(&q8_buf);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("gemv_iq3s_q8 launch: {e}")))?;

    Ok(())
}

/// IQ3_S GEMV reading weights from `base` at `byte_offset`.
///
/// Same as `gemv_iq3s_fused` but reads weights from `base[byte_offset..]`.
/// This avoids creating a new CudaView for each expert.
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

/// IQ3_S GEMV with pre-quantized Q8_1 input, reading weights at offset.
///
/// `q8_buf`: pre-quantized Q8_1 input (from `quantize_f32_to_q8_1_gpu`).
pub fn gemv_iq3s_fused_at_offset_q8(
    base: &CudaView<'_, u8>,
    byte_offset: usize,
    q8_buf: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0, "cols must be multiple of 256");

    let weights = base.slice(byte_offset..);
    let (func, stream) = load_func(dev, GEMV_FUNC)?;

    let cols_i32 = cols as i32;
    let cfg = gemv_launch_config(rows as u32);

    // The GEMV kernel signature is: (weights, input_f32, output, cols, q8_input)
    // When using pre-quantized Q8_1, the `input_f32` parameter is unused by the
    // kernel (it reads from q8_input). We pass the q8_buf pointer as the dummy
    // input_f32 since it's a valid device pointer and won't be dereferenced.
    let mut builder = stream.launch_builder(&func);
    builder.arg(&weights);
    builder.arg(q8_buf);  // dummy f32 ptr — kernel never reads this when q8_input is present
    builder.arg(output);
    builder.arg(&cols_i32);
    builder.arg(q8_buf);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("gemv_iq3s_q8 at offset launch: {e}")))?;

    Ok(())
}

/// IQ3_S GEMV with pre-quantized Q8_1 input (no offset).
pub fn gemv_iq3s_q8_precomputed(
    weights: &CudaView<'_, u8>,
    q8_input: &CudaSlice<u8>,
    output: &mut CudaSlice<f32>,
    rows: usize,
    cols: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(cols % IQ3S_BLOCK_ELEMS == 0, "cols must be multiple of 256");

    let (func, stream) = load_func(dev, GEMV_FUNC)?;

    let cols_i32 = cols as i32;
    let cfg = gemv_launch_config(rows as u32);

    let mut builder = stream.launch_builder(&func);
    builder.arg(weights);
    builder.arg(q8_input);  // dummy f32 ptr — kernel never reads this when q8_input is present
    builder.arg(output);
    builder.arg(&cols_i32);
    builder.arg(q8_input);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("gemv_iq3s_q8_precomputed launch: {e}")))?;

    Ok(())
}

/// Batched IQ3_S GEMV: launch one kernel for `top_k` experts, each producing
/// `rows` output values.
///
/// `all_weights`: flat IQ3_S bytes for ALL experts (concatenated).
/// `q8_input`: pre-quantized Q8_1 input vector (shared across experts).
/// `output`: pre-allocated buffer for `top_k * rows` f32 values.
/// `expert_ids_buf`: GPU buffer of `top_k` i32 expert indices.
/// `cols`: input dimension (must be multiple of 256).
/// `rows`: output dimension per expert.
/// `top_k`: number of experts.
/// `expert_stride`: byte stride between consecutive experts in `all_weights`.
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
    assert!(cols % IQ3S_BLOCK_ELEMS == 0, "cols must be multiple of 256");
    if top_k == 0 || rows == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev, GEMV_BATCHED_FUNC)?;

    let cols_i32 = cols as i32;
    let rows_i32 = rows as i32;
    let expert_stride_i32 = expert_stride as i32;
    let total_rows = (top_k * rows) as u32;

    let cfg = gemv_batched_launch_config(total_rows);

    let mut builder = stream.launch_builder(&func);
    builder.arg(all_weights);
    builder.arg(output);
    builder.arg(&cols_i32);
    builder.arg(q8_input);
    builder.arg(expert_ids_buf);
    builder.arg(&rows_i32);
    builder.arg(&expert_stride_i32);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("gemv_iq3s_q8_batched launch: {e}")))?;

    Ok(())
}

/// Batched down-projection IQ3_S GEMV: different Q8_1 input per expert, single kernel launch.
///
/// Computes `output[k][row] = down_weights[expert_ids[k]][row] . intermediate[k]`
/// for each `k` in `0..top_k` and each `row` in `0..rows`.
///
/// This variant is used for the MoE down-projection where each expert `k` has
/// a distinct intermediate activation vector (the silu_mul output).
///
/// # Arguments
/// - `all_weights`: concatenated IQ3_S down-projection weights for ALL experts
///   (each expert occupies `expert_stride` bytes).
/// - `all_q8_inputs`: concatenated Q8_1 buffers `[top_k * q8_stride bytes]`.
///   Slice `k * q8_stride .. (k+1) * q8_stride` is the Q8_1 encoding of
///   `intermediate[k]` (the silu_mul output for expert `k`).
/// - `output`: pre-allocated buffer for `top_k * rows` f32 values,
///   stored as `output[k * rows + row]`.
/// - `expert_ids`: GPU buffer of `top_k` i32 expert indices.
/// - `cols`: intermediate (ffn) dimension — must be multiple of 256.
/// - `rows`: hidden_size (output dimension per expert).
/// - `top_k`: number of active experts.
/// - `expert_stride`: byte stride between consecutive experts in `all_weights`.
/// - `q8_stride`: byte size of one Q8_1 input vector = `(cols / 32) * 36`.
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
    assert!(cols % IQ3S_BLOCK_ELEMS == 0, "cols must be multiple of 256");
    if top_k == 0 || rows == 0 {
        return Ok(());
    }

    let (func, stream) = load_func(dev, GEMV_BATCHED_MULTI_INPUT_FUNC)?;

    let cols_i32 = cols as i32;
    let rows_i32 = rows as i32;
    let expert_stride_i32 = expert_stride as i32;
    let q8_stride_i32 = q8_stride as i32;

    // Grid: (rows, top_k, 1) — one block per (output_row, expert_k) pair
    let cfg = gemv_multi_input_launch_config(rows as u32, top_k as u32);

    let mut builder = stream.launch_builder(&func);
    builder.arg(all_weights);
    builder.arg(all_q8_inputs);
    builder.arg(output);
    builder.arg(expert_ids);
    builder.arg(&cols_i32);
    builder.arg(&rows_i32);
    builder.arg(&expert_stride_i32);
    builder.arg(&q8_stride_i32);
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("gemv_iq3s_q8_batched_multi_input launch: {e}")))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API: IQ3_S dequantization
// ---------------------------------------------------------------------------

/// Dequantize IQ3_S data at a byte offset within a CudaSlice, returning a Tensor.
///
/// `base`: raw GPU bytes containing IQ3_S data.
/// `byte_offset`: starting byte offset within `base`.
/// `n_elements`: total number of f32 values to produce.
/// `shape`: desired output tensor shape (e.g. `[rows, cols]`).
/// `device`: the Candle device (must be CUDA).
pub fn dequant_iq3s_at_offset(
    base: &CudaSlice<u8>,
    byte_offset: usize,
    n_elements: usize,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let Device::Cuda(cuda_dev) = device else {
        candle_core::bail!("dequant_iq3s requires CUDA device");
    };

    assert!(n_elements % IQ3S_BLOCK_ELEMS == 0, "n_elements must be multiple of 256");
    let n_blocks = n_elements / IQ3S_BLOCK_ELEMS;

    let stream = cuda_dev.cuda_stream();
    let mut output: CudaSlice<f32> = stream
        .alloc_zeros(n_elements)
        .map_err(|e| candle_core::Error::Msg(format!("dequant alloc: {e}")))?;

    let data_view = base.slice(byte_offset..);
    launch_dequant(&data_view, &mut output, n_blocks, cuda_dev)?;

    // Wrap as Tensor
    let storage = candle_core::cuda_backend::CudaStorage::wrap_cuda_slice(
        output, cuda_dev.clone(),
    );
    let tensor = Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        Shape::from_dims(shape),
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok(tensor)
}

/// Dequantize IQ3_S data from a CudaSlice, returning a Tensor.
pub fn dequant_iq3s_gpu(
    data: &CudaSlice<u8>,
    n_elements: usize,
    shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    dequant_iq3s_at_offset(data, 0, n_elements, shape, device)
}

/// Dequantize IQ3_S data at offset into a pre-allocated f32 buffer.
pub fn dequant_iq3s_into(
    base: &CudaSlice<u8>,
    byte_offset: usize,
    output: &mut CudaSlice<f32>,
    n_elements: usize,
    dev: &CudaDevice,
) -> Result<()> {
    assert!(n_elements % IQ3S_BLOCK_ELEMS == 0, "n_elements must be multiple of 256");
    let n_blocks = n_elements / IQ3S_BLOCK_ELEMS;
    let data_view = base.slice(byte_offset..);
    launch_dequant(&data_view, output, n_blocks, dev)
}

/// Internal: launch the dequant_iq3s kernel.
fn launch_dequant(
    data: &CudaView<'_, u8>,
    output: &mut CudaSlice<f32>,
    n_blocks: usize,
    dev: &CudaDevice,
) -> Result<()> {
    if n_blocks == 0 {
        return Ok(());
    }

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
    unsafe {
        builder.launch(cfg)
    }
    .map_err(|e| candle_core::Error::Msg(format!("dequant_iq3s launch: {e}")))?;

    Ok(())
}

// ===========================================================================
// Micro-benchmark: kernel launch overhead measurement
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::backend::BackendDevice;
    use candle_core::Device;
    use std::time::Instant;

    /// Trivial CUDA kernel source — does absolutely nothing.
    const TRIVIAL_KERNEL_SRC: &str = r#"
extern "C" __global__ void trivial_nop() {}
"#;

    static TRIVIAL_PTX: OnceLock<String> = OnceLock::new();

    fn get_trivial_ptx() -> &'static str {
        super::super::nvrtc_compile::compile_and_cache(TRIVIAL_KERNEL_SRC, &TRIVIAL_PTX)
    }

    /// Create synthetic IQ3_S weight data (deterministic, not random).
    /// `n_rows` x `n_cols` matrix. n_cols must be multiple of 256.
    fn make_synthetic_iq3s_weights(n_rows: usize, n_cols: usize) -> Vec<u8> {
        let n_blocks_per_row = n_cols / IQ3S_BLOCK_ELEMS;
        let total_bytes = n_rows * n_blocks_per_row * IQ3S_BLOCK_BYTES;
        (0..total_bytes).map(|i| (i % 251) as u8).collect()
    }

    /// Create synthetic Q8_1 quantized input.
    /// `n_elements` must be multiple of 32.
    fn make_synthetic_q8_input(n_elements: usize) -> Vec<u8> {
        let n_blocks = n_elements / Q8_1_BLOCK_ELEMS;
        let total_bytes = n_blocks * Q8_1_BLOCK_BYTES;
        // Fill with deterministic pattern: small qs values, zero d/s
        let mut data = vec![0u8; total_bytes];
        for blk in 0..n_blocks {
            let base = blk * Q8_1_BLOCK_BYTES;
            // d_bits = 0 (scale = 0 → all zeros, fast path)
            // Actually set a small nonzero scale so the kernel does real work
            data[base] = 0x00;     // d_bits low byte (f16 ~= 0.0001)
            data[base + 1] = 0x10; // d_bits high byte
            data[base + 2] = 0x00; // s_bits
            data[base + 3] = 0x00;
            for j in 0..32 {
                data[base + 4 + j] = ((j % 5) as i8).to_le_bytes()[0];
            }
        }
        data
    }

    #[test]
    fn test_launch_overhead_benchmark() -> Result<()> {
        // -- Acquire CUDA device --
        let device = Device::cuda_if_available(0)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA init: {e}")))?;
        let cuda_dev = match &device {
            Device::Cuda(d) => d,
            _ => {
                eprintln!("[LAUNCH_BENCH] No CUDA device available, skipping.");
                return Ok(());
            }
        };

        let stream = cuda_dev.cuda_stream();

        // ===================================================================
        // 0. Compile all PTX up front (don't count compilation in benchmarks)
        // ===================================================================
        eprintln!("[LAUNCH_BENCH] Compiling PTX...");
        let trivial_ptx = get_trivial_ptx();
        let iq3s_ptx = get_ptx();
        eprintln!("[LAUNCH_BENCH] PTX compiled.");

        // ===================================================================
        // 1. Trivial kernel launch overhead
        // ===================================================================
        let trivial_us = {
            let func = cuda_dev.get_or_load_custom_func(
                "trivial_nop", "chimere_bench_trivial_v1", trivial_ptx,
            )?;

            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };

            // Warmup
            for _ in 0..20 {
                let mut builder = func.builder();
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("trivial launch: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }

            // Measure
            let n_iters = 1000;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let mut builder = func.builder();
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("trivial launch: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }
            let elapsed = t0.elapsed();
            let per_launch_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

            eprintln!("[LAUNCH_BENCH] Trivial kernel launch: {:.1} us", per_launch_us);
            per_launch_us
        };

        // ===================================================================
        // 2. GEMV launch overhead (single expert: 512 rows x 2048 cols)
        // ===================================================================
        let rows: usize = 512;
        let cols: usize = 2048;
        let n_iq3s_blocks_per_row = cols / IQ3S_BLOCK_ELEMS; // 8
        let weight_bytes_per_row = n_iq3s_blocks_per_row * IQ3S_BLOCK_BYTES;
        let total_weight_bytes = rows * weight_bytes_per_row;

        // Allocate synthetic data on GPU
        let weights_host = make_synthetic_iq3s_weights(rows, cols);
        let q8_host = make_synthetic_q8_input(cols);

        let mut weights_gpu: CudaSlice<u8> = stream.alloc_zeros(total_weight_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("alloc weights: {e}")))?;
        cuda_dev.memcpy_htod(&weights_host, &mut weights_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload weights: {e}")))?;

        let q8_bytes = q8_byte_size(cols);
        let mut q8_gpu: CudaSlice<u8> = stream.alloc_zeros(q8_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("alloc q8: {e}")))?;
        cuda_dev.memcpy_htod(&q8_host, &mut q8_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload q8: {e}")))?;

        let output_gpu: CudaSlice<f32> = stream.alloc_zeros(rows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc output: {e}")))?;

        {
            let func = cuda_dev.get_or_load_custom_func(
                GEMV_FUNC, MODULE_NAME, iq3s_ptx,
            )?;

            let cols_i32 = cols as i32;
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 16,
            };

            let weights_view = weights_gpu.slice(0..);

            // Warmup
            for _ in 0..20 {
                let mut builder = func.builder();
                builder.arg(&weights_view);
                builder.arg(&q8_gpu); // dummy f32 ptr
                builder.arg(&output_gpu);
                builder.arg(&cols_i32);
                builder.arg(&q8_gpu);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gemv launch: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }

            // Measure GEMV (launch + compute)
            let n_iters = 100;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let mut builder = func.builder();
                builder.arg(&weights_view);
                builder.arg(&q8_gpu);
                builder.arg(&output_gpu);
                builder.arg(&cols_i32);
                builder.arg(&q8_gpu);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("gemv launch: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }
            let elapsed = t0.elapsed();
            let per_gemv_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

            // The trivial kernel measures pure launch+sync overhead.
            // GEMV overhead = GEMV total - (GEMV total - trivial) ≈ trivial.
            // But GEMV has more args + larger grid, so overhead may differ.
            // Best estimate: GEMV_compute = GEMV_total - trivial_launch.
            let gemv_compute_us = (per_gemv_us - trivial_us).max(0.0);
            let gemv_overhead_us = per_gemv_us - gemv_compute_us;
            eprintln!("[LAUNCH_BENCH] GEMV launch (incl compute): {:.1} us", per_gemv_us);
            eprintln!("[LAUNCH_BENCH] GEMV compute only (estimated): {:.1} us", gemv_compute_us);
            eprintln!("[LAUNCH_BENCH] GEMV launch overhead: {:.1} us", gemv_overhead_us);
        }

        // ===================================================================
        // 3. Batched vs individual: 8 experts
        // ===================================================================
        let top_k: usize = 8;
        let expert_stride = total_weight_bytes; // each expert same size

        // Allocate weight buffer for 8 "experts" (same data repeated)
        let total_all_weights = top_k * total_weight_bytes;
        let mut all_weights_host = Vec::with_capacity(total_all_weights);
        for _ in 0..top_k {
            all_weights_host.extend_from_slice(&weights_host);
        }
        let mut all_weights_gpu: CudaSlice<u8> = stream.alloc_zeros(total_all_weights)
            .map_err(|e| candle_core::Error::Msg(format!("alloc all_weights: {e}")))?;
        cuda_dev.memcpy_htod(&all_weights_host, &mut all_weights_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload all_weights: {e}")))?;

        // Expert IDs: [0, 1, 2, ..., 7]
        let expert_ids: Vec<i32> = (0..top_k as i32).collect();
        let mut expert_ids_gpu: CudaSlice<i32> = cuda_dev.alloc_zeros::<i32>(top_k)
            .map_err(|e| candle_core::Error::Msg(format!("alloc expert_ids: {e}")))?;
        cuda_dev.memcpy_htod(&expert_ids, &mut expert_ids_gpu)
            .map_err(|e| candle_core::Error::Msg(format!("upload expert_ids: {e}")))?;

        // Output for batched: top_k * rows
        let batched_output_gpu: CudaSlice<f32> = stream.alloc_zeros(top_k * rows)
            .map_err(|e| candle_core::Error::Msg(format!("alloc batched_output: {e}")))?;

        // -- 3a. 8 individual GEMV launches --
        let individual_us = {
            let func = cuda_dev.get_or_load_custom_func(
                GEMV_FUNC, MODULE_NAME, iq3s_ptx,
            )?;

            let cols_i32 = cols as i32;
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 16,
            };

            let all_view = all_weights_gpu.slice(0..);

            // Warmup
            for _ in 0..10 {
                for k in 0..top_k {
                    let offset = k * expert_stride;
                    let w_view = all_view.slice(offset..);
                    let mut builder = func.builder();
                    builder.arg(&w_view);
                    builder.arg(&q8_gpu);
                    builder.arg(&output_gpu);
                    builder.arg(&cols_i32);
                    builder.arg(&q8_gpu);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("ind gemv: {e}")))?;
                }
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }

            // Measure
            let n_iters = 100;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                for k in 0..top_k {
                    let offset = k * expert_stride;
                    let w_view = all_view.slice(offset..);
                    let mut builder = func.builder();
                    builder.arg(&w_view);
                    builder.arg(&q8_gpu);
                    builder.arg(&output_gpu);
                    builder.arg(&cols_i32);
                    builder.arg(&q8_gpu);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("ind gemv: {e}")))?;
                }
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }
            let elapsed = t0.elapsed();
            elapsed.as_secs_f64() * 1e6 / n_iters as f64
        };

        // -- 3b. 1 batched GEMV launch (8 experts) --
        let batched_us = {
            let func = cuda_dev.get_or_load_custom_func(
                GEMV_BATCHED_FUNC, MODULE_NAME, iq3s_ptx,
            )?;

            let cols_i32 = cols as i32;
            let rows_i32 = rows as i32;
            let expert_stride_i32 = expert_stride as i32;
            let total_grid_rows = (top_k * rows) as u32;

            let cfg = LaunchConfig {
                grid_dim: (total_grid_rows, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 16,
            };

            let all_view = all_weights_gpu.slice(0..);

            // Warmup
            for _ in 0..10 {
                let mut builder = func.builder();
                builder.arg(&all_view);
                builder.arg(&batched_output_gpu);
                builder.arg(&cols_i32);
                builder.arg(&q8_gpu);
                builder.arg(&expert_ids_gpu);
                builder.arg(&rows_i32);
                builder.arg(&expert_stride_i32);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("batched launch: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }

            // Measure
            let n_iters = 100;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let mut builder = func.builder();
                builder.arg(&all_view);
                builder.arg(&batched_output_gpu);
                builder.arg(&cols_i32);
                builder.arg(&q8_gpu);
                builder.arg(&expert_ids_gpu);
                builder.arg(&rows_i32);
                builder.arg(&expert_stride_i32);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("batched launch: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }
            let elapsed = t0.elapsed();
            elapsed.as_secs_f64() * 1e6 / n_iters as f64
        };

        let speedup = if batched_us > 0.0 { individual_us / batched_us } else { 0.0 };
        eprintln!("[LAUNCH_BENCH] 8x individual GEMV: {:.1} us", individual_us);
        eprintln!("[LAUNCH_BENCH] 1x batched GEMV (8 experts): {:.1} us", batched_us);
        eprintln!("[LAUNCH_BENCH] Speedup: {:.2}x", speedup);

        // ===================================================================
        // 4. get_or_load_custom_func overhead (HashMap lookup, cached)
        // ===================================================================
        {
            // First call was done above (PTX already loaded). Measure cached path.
            let n_iters = 10_000;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let _func = cuda_dev.get_or_load_custom_func(
                    GEMV_FUNC, MODULE_NAME, iq3s_ptx,
                )?;
            }
            let elapsed = t0.elapsed();
            let per_lookup_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

            eprintln!(
                "[LAUNCH_BENCH] get_or_load_custom_func (cached): {:.2} us",
                per_lookup_us
            );
        }

        // ===================================================================
        // 5. Top-K softmax kernel overhead
        // ===================================================================
        {
            let num_experts: usize = 256;
            let top_k_val: usize = 8;

            // Allocate synthetic logits
            let logits_host: Vec<f32> = (0..num_experts).map(|i| (i as f32) * 0.01).collect();
            let mut logits_gpu: CudaSlice<f32> = stream.alloc_zeros(num_experts)
                .map_err(|e| candle_core::Error::Msg(format!("alloc logits: {e}")))?;
            cuda_dev.memcpy_htod(&logits_host, &mut logits_gpu)
                .map_err(|e| candle_core::Error::Msg(format!("upload logits: {e}")))?;

            let mut top_indices_gpu: CudaSlice<i32> = cuda_dev.alloc_zeros::<i32>(top_k_val)
                .map_err(|e| candle_core::Error::Msg(format!("alloc top_indices: {e}")))?;
            let mut top_weights_gpu: CudaSlice<f32> = stream.alloc_zeros(top_k_val)
                .map_err(|e| candle_core::Error::Msg(format!("alloc top_weights: {e}")))?;

            let logits_view = logits_gpu.slice(0..);

            // Warmup
            for _ in 0..20 {
                crate::kernels::topk_softmax::gpu_topk_softmax(
                    &logits_view,
                    &mut top_indices_gpu,
                    &mut top_weights_gpu,
                    num_experts,
                    top_k_val,
                    cuda_dev,
                )?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }

            // Measure
            let n_iters = 1000;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                crate::kernels::topk_softmax::gpu_topk_softmax(
                    &logits_view,
                    &mut top_indices_gpu,
                    &mut top_weights_gpu,
                    num_experts,
                    top_k_val,
                    cuda_dev,
                )?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }
            let elapsed = t0.elapsed();
            let per_topk_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

            eprintln!(
                "[LAUNCH_BENCH] Top-K softmax (256 experts, k=8): {:.1} us",
                per_topk_us
            );
        }

        // ===================================================================
        // 6. get_or_load_custom_func: first call (compile + load) vs cached
        // ===================================================================
        {
            // Use a unique module name to force a fresh load
            let unique_module = "chimere_bench_fresh_load_v1";
            let t0 = Instant::now();
            let _func = cuda_dev.get_or_load_custom_func(
                "trivial_nop", unique_module, trivial_ptx,
            )?;
            let first_call_us = t0.elapsed().as_secs_f64() * 1e6;

            // Second call (cached)
            let t1 = Instant::now();
            for _ in 0..1000 {
                let _func = cuda_dev.get_or_load_custom_func(
                    "trivial_nop", unique_module, trivial_ptx,
                )?;
            }
            let cached_us = t1.elapsed().as_secs_f64() * 1e6 / 1000.0;

            eprintln!(
                "[LAUNCH_BENCH] get_or_load_custom_func (first load): {:.0} us",
                first_call_us
            );
            eprintln!(
                "[LAUNCH_BENCH] get_or_load_custom_func (cached): {:.2} us",
                cached_us
            );
        }

        // ===================================================================
        // 7. load_func (cubin or NVRTC) overhead comparison
        // ===================================================================
        {
            // Warm up the cubin/NVRTC cache
            let _ = load_func(cuda_dev, GEMV_FUNC)?;

            let n_iters = 10_000;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let (_func, _stream) = load_func(cuda_dev, GEMV_FUNC)?;
            }
            let elapsed = t0.elapsed();
            let per_lookup_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;

            let path = if super::super::cubin_loader::has_cubin() {
                "cubin"
            } else {
                "NVRTC fallback"
            };
            eprintln!(
                "[LAUNCH_BENCH] load_func [{}] (cached): {:.2} us",
                path, per_lookup_us
            );

            // Full launch cycle: load_func + launch_builder + launch (trivial)
            let (func, stream) = load_func(cuda_dev, GEMV_FUNC)?;
            let cols_i32 = cols as i32;
            let weights_view = weights_gpu.slice(0..);
            let cfg = LaunchConfig {
                grid_dim: (rows as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 16,
            };

            // Warmup
            for _ in 0..20 {
                let mut builder = stream.launch_builder(&func);
                builder.arg(&weights_view);
                builder.arg(&q8_gpu);
                builder.arg(&output_gpu);
                builder.arg(&cols_i32);
                builder.arg(&q8_gpu);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("cubin gemv: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }

            let n_iters = 100;
            let t0 = Instant::now();
            for _ in 0..n_iters {
                let (func, stream) = load_func(cuda_dev, GEMV_FUNC)?;
                let mut builder = stream.launch_builder(&func);
                builder.arg(&weights_view);
                builder.arg(&q8_gpu);
                builder.arg(&output_gpu);
                builder.arg(&cols_i32);
                builder.arg(&q8_gpu);
                unsafe { builder.launch(cfg) }
                    .map_err(|e| candle_core::Error::Msg(format!("cubin gemv: {e}")))?;
                cuda_dev.synchronize()
                    .map_err(|e| candle_core::Error::Msg(format!("sync: {e}")))?;
            }
            let elapsed = t0.elapsed();
            let per_launch_us = elapsed.as_secs_f64() * 1e6 / n_iters as f64;
            eprintln!(
                "[LAUNCH_BENCH] load_func [{}] + GEMV launch+sync: {:.1} us",
                path, per_launch_us
            );
        }

        eprintln!("[LAUNCH_BENCH] Done.");
        Ok(())
    }

    /// Expose PTX for analysis.
    pub fn get_iq3s_gemv_ptx() -> Result<String> {
        Ok(get_ptx().to_string())
    }

    #[test]
    fn test_dump_ptx() {
        let ptx = get_iq3s_gemv_ptx().unwrap();
        std::fs::write("/tmp/chimere_iq3s_gemv.ptx", &ptx).unwrap();
        eprintln!("PTX dumped to /tmp/chimere_iq3s_gemv.ptx ({} bytes)", ptx.len());
    }
}
