// chimere_kernels.cu — All CUDA kernels for chimere-deltanet.
//
// Compiled at build time by nvcc into a cubin for sm_120 (RTX 5060 Ti).
// This eliminates NVRTC runtime compilation overhead (13.3us -> ~3-5us per launch).
//
// ALL kernels are in this single compilation unit. No NVRTC fallback needed.
//
// Kernel modules:
//   1. deltanet_step_kernel     (fused DeltaNet state update)
//   2. dequant_iq3s             (IQ3_S -> F32 dequantisation)
//   3. gemv_q5k                 (Q5_K fused dequant + GEMV)
//   4. gemv_q5k_q8 + quantize_f32_to_q8_1_q5k  (Q5_K x Q8_1 dp4a GEMV)
//   5. elemwise kernels         (rms_norm, silu, sigmoid, softplus, mul, add, etc.)
//   6. gemv_iq3s_q8 + quantize_f32_to_q8_1  (IQ3_S GEMV with Q8_1+dp4a)
//   7. fused_moe_iq3s           (fused MoE IQ3_S gate+up+silu+down)
//   8. topk_softmax             (GPU top-K for MoE routing)
//   9. fused_beta_alpha_gate + fused_rms_norm_silu_gate  (fused GDN elem ops)
//  10. ggml_mul_mat_vec_q5_K_q8_1 + ggml_quantize_q8_1  (ggml-compatible MMVQ)
//  11. silu_mul_batched + weighted_combine + l2_norm_groups +
//      expand_groups + scale + add_inplace  (batched MoE + GDN elementwise)
//  12. fused_conv1d_silu_update (fused conv1d + SiLU + state update)
//  13. gemv_q5k_q8_dual        (dual Q5_K GEMV with shared Q8_1 input)
//  14. deinterleave_q_gate + rms_norm_per_head + sigmoid_gate_mul +
//      mrope_apply              (attention layer kernels)

#include <cuda_fp16.h>
#include <cstdint>

// =====================================================================
// Shared helper: f16 <-> f32 conversion (avoids cuda_fp16.h dependency)
// =====================================================================

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

__device__ __forceinline__ unsigned short f32_to_f16_bits(float x) {
    unsigned int u = *(unsigned int*)&x;
    unsigned int sign = (u >> 31) & 1u;
    int exp_f32 = (int)((u >> 23) & 0xFFu) - 127;
    unsigned int mant_f32 = u & 0x7FFFFFu;
    unsigned short h;
    if (exp_f32 >= 16) h = (unsigned short)((sign << 15) | 0x7C00u);
    else if (exp_f32 < -14) h = (unsigned short)(sign << 15);
    else { int e16 = exp_f32 + 15; h = (unsigned short)((sign << 15) | ((unsigned int)e16 << 10) | (mant_f32 >> 13)); }
    return h;
}

// =====================================================================
// IQ3_S grid — 512 entries, each u32 encodes 4 unsigned byte levels
// Used by: dequant_iq3s, gemv_iq3s_q8, fused_moe_iq3s
// =====================================================================

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


// =====================================================================
// 1. DeltaNet fused step kernel
// =====================================================================

extern "C" __global__ __launch_bounds__(128, 1)
void deltanet_step_kernel(
    const float* __restrict__ s_in,
    float*       __restrict__ s_out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float*       __restrict__ output,
    int D
) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;

    extern __shared__ char smem[];
    float* k_s = (float*)smem;
    float* q_s = (float*)smem + D;

    if (j < D) {
        k_s[j] = k[h * D + j];
        q_s[j] = q[h * D + j];
    }
    __syncthreads();

    if (j >= D) return;

    const float g   = gate[h];
    const float b   = beta[h];
    const float v_j = v[h * D + j];

    const float* s_col_in  = s_in  + h * D * D;
    float*       s_col_out = s_out + h * D * D;

    float s_reg[128];
    double sk_j = 0.0;

    #pragma unroll 8
    for (int i = 0; i < D; i++) {
        float val = s_col_in[i * D + j] * g;
        s_reg[i] = val;
        sk_j += (double)val * (double)k_s[i];
    }

    float delta_j = (v_j - (float)sk_j) * b;

    double out_j = 0.0;

    #pragma unroll 8
    for (int i = 0; i < D; i++) {
        float val = s_reg[i] + k_s[i] * delta_j;
        // Clamp state to prevent divergence (matches llama.cpp)
        val = fminf(fmaxf(val, -1e6f), 1e6f);
        s_col_out[i * D + j] = val;
        out_j += (double)val * (double)q_s[i];
    }

    output[h * D + j] = (float)out_j;
}


// =====================================================================
// 2. IQ3_S dequantisation kernel
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
// 3. Q5_K fused dequant + GEMV kernel
// =====================================================================

__device__ void get_scale_min_k4(
    int j,
    const unsigned char* scales,
    float* sc_out,
    float* mn_out
) {
    if (j < 4) {
        *sc_out = (float)(scales[j]   & 63);
        *mn_out = (float)(scales[j+4] & 63);
    } else {
        *sc_out = (float)((scales[j+4] & 0xF)  | ((scales[j-4] >> 6) << 4));
        *mn_out = (float)((scales[j+4] >> 4)    | ((scales[j-0] >> 6) << 4));
    }
}

extern "C" __global__ void gemv_q5k(
    const unsigned char* __restrict__ weights,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    int                               in_features
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int BLOCK_ELEMS = 256;
    const int BLOCK_BYTES = 176;

    const int n_blocks = in_features / BLOCK_ELEMS;

    const unsigned char* row_data =
        weights + (long long)row * (long long)n_blocks * BLOCK_BYTES;

    float sum = 0.0f;

    for (int block_idx = tid; block_idx < n_blocks; block_idx += blockDim.x) {
        const unsigned char* block = row_data + block_idx * BLOCK_BYTES;

        unsigned short d_bits    = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
        unsigned short dmin_bits = (unsigned short)block[2] | ((unsigned short)block[3] << 8);
        float d    = f16_to_f32(d_bits);
        float dmin = f16_to_f32(dmin_bits);

        const unsigned char* scales = block + 4;
        const unsigned char* qh     = block + 16;
        const unsigned char* qs     = block + 48;

        int base_elem = block_idx * BLOCK_ELEMS;

        for (int j = 0; j < 8; j++) {
            float sc, mn;
            get_scale_min_k4(j, scales, &sc, &mn);
            float d_sc = d    * sc;
            float d_mn = dmin * mn;

            for (int k = 0; k < 32; k++) {
                int elem        = j * 32 + k;
                int global_elem = base_elem + elem;

                int qs_byte_idx = (j / 2) * 32 + k;
                int low4 = (j & 1)
                    ? (((int)qs[qs_byte_idx] >> 4) & 0xF)
                    : ( (int)qs[qs_byte_idx]       & 0xF);

                int high1 = ((int)qh[k] >> j) & 1;
                int q5    = low4 | (high1 << 4);

                float w = d_sc * (float)q5 - d_mn;
                sum += w * input[global_elem];
            }
        }
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[row] = sdata[0];
}


// =====================================================================
// 4. Q5_K + Q8_1 dp4a GEMV kernel
// =====================================================================

struct block_q8_1_q5k {
    unsigned short d_bits;
    unsigned short s_bits;
    signed char    qs[32];
};

__device__ void quantize_q8_1_block_q5k(
    const float* __restrict__ x,
    struct block_q8_1_q5k* b
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

extern "C" __global__ void quantize_f32_to_q8_1_q5k(
    const float* __restrict__ input,
    unsigned char* __restrict__ q8_out,
    int n_blocks
) {
    for (int blk = threadIdx.x + blockIdx.x * blockDim.x; blk < n_blocks; blk += blockDim.x * gridDim.x) {
        quantize_q8_1_block_q5k(&input[blk * 32], (struct block_q8_1_q5k*)(q8_out + blk * 36));
    }
}

__device__ __forceinline__ float dot_q5k_q8_subblock(
    const unsigned char* __restrict__ blk,
    const struct block_q8_1_q5k* __restrict__ q8b,
    int sg
) {
    unsigned short d_bits    = (unsigned short)blk[0] | ((unsigned short)blk[1] << 8);
    unsigned short dmin_bits = (unsigned short)blk[2] | ((unsigned short)blk[3] << 8);
    float d5    = f16_to_f32(d_bits);
    float dmin5 = f16_to_f32(dmin_bits);

    const unsigned char* scales = blk + 4;
    float sc_f, mn_f;
    if (sg < 4) {
        sc_f = (float)(scales[sg]   & 63);
        mn_f = (float)(scales[sg+4] & 63);
    } else {
        sc_f = (float)((scales[sg+4] & 0xF)  | ((scales[sg-4] >> 6) << 4));
        mn_f = (float)((scales[sg+4] >> 4)    | ((scales[sg]   >> 6) << 4));
    }

    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;

    float d_q8 = f16_to_f32(q8b->d_bits);
    float s_q8 = f16_to_f32(q8b->s_bits);

    const int* q8_qs_int = (const int*)(q8b->qs);

    int sumi = 0;

    for (int l = 0; l < 8; l++) {
        int k_base = l * 4;
        unsigned int v = 0;
        for (int b = 0; b < 4; b++) {
            int k = k_base + b;
            int qs_byte_idx = (sg / 2) * 32 + k;
            int low4 = (sg & 1)
                ? (((int)qs[qs_byte_idx] >> 4) & 0xF)
                : ( (int)qs[qs_byte_idx]       & 0xF);
            int high1 = ((int)qh[k] >> sg) & 1;
            int q5 = low4 | (high1 << 4);
            v |= ((unsigned int)(q5 & 0xFF)) << (b * 8);
        }
        sumi = __dp4a((int)v, q8_qs_int[l], sumi);
    }

    return d5 * sc_f * d_q8 * (float)sumi - dmin5 * mn_f * s_q8;
}

extern "C" __global__
void gemv_q5k_q8(
    const unsigned char* __restrict__ weights,
    float*               __restrict__ output,
    int                              cols,
    const unsigned char* __restrict__ q8_input
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int n_q5k_blocks = cols >> 8;
    const int Q5K_BYTES = 176;

    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;

    const struct block_q8_1_q5k* q8_blocks =
        (const struct block_q8_1_q5k*)q8_input;

    const unsigned char* row_weights =
        weights + (long long)row * (long long)n_q5k_blocks * Q5K_BYTES;

    float sum = 0.0f;

    const int total_subblocks = n_q5k_blocks * 8;

    for (int sg_idx = tid; sg_idx < total_subblocks; sg_idx += 128) {
        int kbx = sg_idx >> 3;
        int sg  = sg_idx & 7;

        const unsigned char* blk_ptr = row_weights + kbx * Q5K_BYTES;
        const struct block_q8_1_q5k* q8_ptr = &q8_blocks[kbx * 8 + sg];

        sum += dot_q5k_q8_subblock(blk_ptr, q8_ptr, sg);
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
        output[row] = total;
    }
}


// =====================================================================
// 5. Element-wise fused CUDA kernels
// =====================================================================

extern "C" __global__ void rms_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int n,
    float eps
) {
    __shared__ float sdata[256];
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = input[i];
        local_sq += v * v;
    }
    sdata[threadIdx.x] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

extern "C" __global__ void silu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

extern "C" __global__ void sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

extern "C" __global__ void softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = logf(1.0f + expf(input[i]));
    }
}

extern "C" __global__ void mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] * b[i];
}

extern "C" __global__ void silu_mul_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        float silu_x = x / (1.0f + expf(-x));
        output[i] = silu_x * b[i];
    }
}

extern "C" __global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = a[i] + b[i];
}

extern "C" __global__ void weighted_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float weight,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += input[i] * weight;
}

extern "C" __global__ void argmax_kernel(
    const float* __restrict__ input,
    int* __restrict__ output,
    int n
) {
    __shared__ float sval[256];
    __shared__ int sidx[256];

    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }

    sval[threadIdx.x] = max_val;
    sidx[threadIdx.x] = max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (sval[threadIdx.x + s] > sval[threadIdx.x]) {
                sval[threadIdx.x] = sval[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) output[0] = sidx[0];
}

extern "C" __global__ void bias_softplus_mul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i] + bias[i];
        output[i] = logf(1.0f + expf(x)) * scale[i];
    }
}

extern "C" __global__ void sigmoid_mul_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        output[i] += g * input[i];
    }
}

// Shared-memory GEMV: y = A @ x, A is [rows, cols] row-major.
// Grid: (rows, 1, 1), Block: (256, 1, 1).
// Each block computes one output element y[row].
// The input vector x is loaded into shared memory once per block.
extern "C" __global__ void f32_gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols
) {
    extern __shared__ float sx[];       // shared x[cols]
    __shared__ float sred[256];         // reduction scratch

    int row = blockIdx.x;
    if (row >= rows) return;

    // Cooperatively load x into shared memory
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        sx[j] = x[j];
    }
    __syncthreads();

    // Each thread computes a partial dot product
    float acc = 0.0f;
    const float* a_row = A + (long long)row * cols;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) {
        acc += a_row[j] * sx[j];
    }

    // Block-level reduction
    sred[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) y[row] = sred[0];
}


// =====================================================================
// 6. IQ3_S GEMV with Q8_1+dp4a (the main IQ3_S GEMV)
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

// Vectorized load helpers matching ggml
__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    return (int)x16[2*i32] | ((int)x16[2*i32 + 1] << 16);
}
__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}

__device__ __forceinline__ float dot_iq3s_q8_subgroup(
    const unsigned char* __restrict__ block,
    const struct block_q8_1_gemv* __restrict__ q8_block,
    int sg
) {
    unsigned short d_raw = (unsigned short)block[0] | ((unsigned short)block[1] << 8);
    float d_iq3 = f16_to_f32(d_raw);

    unsigned char scale_byte = block[106 + sg / 2];
    int scale_nibble = (sg & 1) == 0 ? (scale_byte & 0x0F) : (scale_byte >> 4);

    int qh = (int)block[66 + sg];

    float d_q8 = f16_to_f32(q8_block->d_bits);

    const unsigned char* qs_base = block + 2 + sg * 8;
    const unsigned int qs_lo = (unsigned int)get_int_b2(qs_base, 0);
    const unsigned int qs_hi = (unsigned int)get_int_b2(qs_base, 1);

    const unsigned int signs_packed = (unsigned int)get_int_b2(block + 74 + sg * 4, 0);

    int sumi = 0;

    // --- l=0 ---
    {
        int grid_idx0 = (int)(qs_lo & 0xFFu)         | ((qh << 8) & 0x100);
        int grid_idx1 = (int)((qs_lo >> 8) & 0xFFu)  | ((qh << 7) & 0x100);
        unsigned int grid0 = iq3s_grid[grid_idx0];
        unsigned int grid1 = iq3s_grid[grid_idx1];
        unsigned int sb0 = signs_packed & 0xFFu;
        int signs0 = __vcmpne4((int)(((sb0 & 0x03u) << 7) | ((sb0 & 0x0Cu) << 21)), 0);
        int signs1 = __vcmpne4((int)(((sb0 & 0x30u) << 3) | ((sb0 & 0xC0u) << 17)), 0);
        sumi = __dp4a((int)__vsub4((int)grid0 ^ signs0, signs0), get_int_b4(q8_block->qs, 0), sumi);
        sumi = __dp4a((int)__vsub4((int)grid1 ^ signs1, signs1), get_int_b4(q8_block->qs, 1), sumi);
    }
    // --- l=1 ---
    {
        int grid_idx0 = (int)((qs_lo >> 16) & 0xFFu) | ((qh << 6) & 0x100);
        int grid_idx1 = (int)((qs_lo >> 24) & 0xFFu) | ((qh << 5) & 0x100);
        unsigned int grid0 = iq3s_grid[grid_idx0];
        unsigned int grid1 = iq3s_grid[grid_idx1];
        unsigned int sb1 = (signs_packed >> 8) & 0xFFu;
        int signs0 = __vcmpne4((int)(((sb1 & 0x03u) << 7) | ((sb1 & 0x0Cu) << 21)), 0);
        int signs1 = __vcmpne4((int)(((sb1 & 0x30u) << 3) | ((sb1 & 0xC0u) << 17)), 0);
        sumi = __dp4a((int)__vsub4((int)grid0 ^ signs0, signs0), get_int_b4(q8_block->qs, 2), sumi);
        sumi = __dp4a((int)__vsub4((int)grid1 ^ signs1, signs1), get_int_b4(q8_block->qs, 3), sumi);
    }
    // --- l=2 ---
    {
        int grid_idx0 = (int)(qs_hi & 0xFFu)         | ((qh << 4) & 0x100);
        int grid_idx1 = (int)((qs_hi >> 8) & 0xFFu)  | ((qh << 3) & 0x100);
        unsigned int grid0 = iq3s_grid[grid_idx0];
        unsigned int grid1 = iq3s_grid[grid_idx1];
        unsigned int sb2 = (signs_packed >> 16) & 0xFFu;
        int signs0 = __vcmpne4((int)(((sb2 & 0x03u) << 7) | ((sb2 & 0x0Cu) << 21)), 0);
        int signs1 = __vcmpne4((int)(((sb2 & 0x30u) << 3) | ((sb2 & 0xC0u) << 17)), 0);
        sumi = __dp4a((int)__vsub4((int)grid0 ^ signs0, signs0), get_int_b4(q8_block->qs, 4), sumi);
        sumi = __dp4a((int)__vsub4((int)grid1 ^ signs1, signs1), get_int_b4(q8_block->qs, 5), sumi);
    }
    // --- l=3 ---
    {
        int grid_idx0 = (int)((qs_hi >> 16) & 0xFFu) | ((qh << 2) & 0x100);
        int grid_idx1 = (int)((qs_hi >> 24) & 0xFFu) | ((qh << 1) & 0x100);
        unsigned int grid0 = iq3s_grid[grid_idx0];
        unsigned int grid1 = iq3s_grid[grid_idx1];
        unsigned int sb3 = (signs_packed >> 24) & 0xFFu;
        int signs0 = __vcmpne4((int)(((sb3 & 0x03u) << 7) | ((sb3 & 0x0Cu) << 21)), 0);
        int signs1 = __vcmpne4((int)(((sb3 & 0x30u) << 3) | ((sb3 & 0xC0u) << 17)), 0);
        sumi = __dp4a((int)__vsub4((int)grid0 ^ signs0, signs0), get_int_b4(q8_block->qs, 6), sumi);
        sumi = __dp4a((int)__vsub4((int)grid1 ^ signs1, signs1), get_int_b4(q8_block->qs, 7), sumi);
    }

    sumi *= 1 + 2 * scale_nibble;
    return d_iq3 * d_q8 * (float)sumi;
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

extern "C" __global__
void gemv_iq3s_q8(
    const unsigned char* __restrict__ weights,
    const float*         __restrict__ input,
    float*               __restrict__ output,
    int                              cols,
    const unsigned char* __restrict__ q8_input
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int n_iq3s_blocks = cols >> 8;
    const int IQ3S_BYTES    = 110;

    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;

    const struct block_q8_1_gemv* q8_blocks = (const struct block_q8_1_gemv*)q8_input;

    const int total_subgroups = n_iq3s_blocks * 8;

    const unsigned char* row_weights =
        weights + (long long)row * (long long)n_iq3s_blocks * IQ3S_BYTES;

    float sum = 0.0f;

    for (int sg_idx = tid; sg_idx < total_subgroups; sg_idx += 128) {
        int kbx = sg_idx / 8;
        int sg  = sg_idx % 8;

        const unsigned char* blk_ptr = row_weights + kbx * IQ3S_BYTES;
        const struct block_q8_1_gemv* q8_ptr = &q8_blocks[kbx * 8 + sg];

        sum += dot_iq3s_q8_subgroup(blk_ptr, q8_ptr, sg);
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
        output[row] = total;
    }
}


// =====================================================================
// 7. Fused MoE IQ3_S kernel
// =====================================================================

// Device helper: IQ3_S GEMV from shared-memory input -> shared-memory output
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

// Device helper: Down projection with atomicAdd
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


// =====================================================================
// 8. Top-K softmax kernel
// =====================================================================

extern "C" __global__ void topk_softmax(
    const float* __restrict__ logits,
    int*         __restrict__ top_indices,
    float*       __restrict__ top_weights,
    int num_experts,
    int top_k
) {
    extern __shared__ char smem[];
    float* probs = (float*)smem;

    int tid = threadIdx.x;

    float val = (tid < num_experts) ? logits[tid] : -1.0e30f;

    float warp_max = val;
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_max = fmaxf(warp_max, __shfl_xor_sync(0xFFFFFFFF, warp_max, offset));

    if ((tid & 31) == 0)
        probs[tid >> 5] = warp_max;
    __syncthreads();

    float max_val = 0.0f;
    if (tid == 0) {
        max_val = probs[0];
        int n_warps = (num_experts + 31) >> 5;
        for (int w = 1; w < n_warps; w++)
            max_val = fmaxf(max_val, probs[w]);
        probs[0] = max_val;
    }
    __syncthreads();
    max_val = probs[0];
    __syncthreads();

    float exp_val = (tid < num_experts) ? expf(val - max_val) : 0.0f;
    if (tid < num_experts)
        probs[tid] = exp_val;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && (tid + stride) < num_experts)
            probs[tid] += probs[tid + stride];
        __syncthreads();
    }
    float sum_exp = probs[0];
    __syncthreads();

    if (tid < num_experts)
        probs[tid] = (sum_exp > 0.0f) ? (exp_val / sum_exp) : (1.0f / (float)num_experts);
    __syncthreads();

    if (tid == 0) {
        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int   best_idx = 0;
            float best_val = -1.0f;
            for (int i = 0; i < num_experts; i++) {
                if (probs[i] > best_val) {
                    best_val = probs[i];
                    best_idx = i;
                }
            }
            top_indices[k] = best_idx;
            top_weights[k] = best_val;
            weight_sum     += best_val;
            probs[best_idx] = -1.0f;
        }
        if (weight_sum > 0.0f) {
            float inv = 1.0f / weight_sum;
            for (int k = 0; k < top_k; k++)
                top_weights[k] *= inv;
        }
    }
}


// =====================================================================
// 9. Fused GDN elementwise kernels
// =====================================================================

extern "C" __global__ void fused_beta_alpha_gate(
    const float* __restrict__ beta_proj,
    const float* __restrict__ alpha_proj,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ beta_out,
    float* __restrict__ gate_exp_out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    beta_out[i] = 1.0f / (1.0f + expf(-beta_proj[i]));
    float ab = alpha_proj[i] + dt_bias[i];
    float sp = ab > 20.0f ? ab : logf(1.0f + expf(ab));
    gate_exp_out[i] = expf(sp * ssm_a[i]);
}

extern "C" __global__ void fused_rms_norm_silu_gate(
    const float* __restrict__ ssm_out,
    const float* __restrict__ weight,
    const float* __restrict__ z,
    float* __restrict__ output,
    int D,
    float eps
) {
    const int g = blockIdx.x;
    const int j = threadIdx.x;

    extern __shared__ char smem[];
    float* sdata = (float*)smem;

    const int base = g * D;

    float val = 0.0f;
    float sq = 0.0f;
    if (j < D) {
        val = ssm_out[base + j];
        sq = val * val;
    }
    sdata[j] = sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) sdata[j] += sdata[j + s];
        __syncthreads();
    }

    if (j >= D) return;

    float rms = sqrtf(sdata[0] / (float)D + eps);
    float inv_rms = 1.0f / rms;

    float normed = val * inv_rms * weight[j];

    float zv = z[base + j];
    float silu_z = zv / (1.0f + expf(-zv));

    output[base + j] = normed * silu_z;
}

// =====================================================================
// 10. ggml-compatible Q5_K MMVQ kernel (ported from ggml-cuda)
//
// Faithful port of ggml's mul_mat_vec_q<Q5_K, ncols_y=1, nwarps=4>.
// Uses cuda_fp16.h types (half, half2) for exact numerical parity.
// Self-contained: no ggml.h / ggml-cuda.h dependency.
// =====================================================================

// ---- ggml-compatible struct definitions ----

// Q5_K super-block: 256 elements, ~5.5 bits/weight, 176 bytes
// Layout: {half d, half dmin, uint8_t scales[12], uint8_t qh[32], uint8_t qs[128]}
struct ggml_block_q5_K {
    half     d;             // super-block scale
    half     dmin;          // super-block min
    uint8_t  scales[12];    // sub-block scales and mins, quantized with 6 bits
    uint8_t  qh[32];        // quants, high bit (256/8 = 32 bytes)
    uint8_t  qs[128];       // quants, low 4 bits (256/2 = 128 bytes)
};

// Q8_1 block: 32 elements, 36 bytes
// Layout: {half2 ds, int8_t qs[32]} where ds.x = d (scale), ds.y = s (sum)
struct ggml_block_q8_1 {
    half2    ds;            // d (scale) and s (weighted sum)
    int8_t   qs[32];        // quantized values
};

// ---- Constants (inlined from ggml) ----
#define GGML_QK_K        256
#define GGML_K_SCALE_SIZE 12
#define GGML_QK8_1        32
#define GGML_QI5_K       (GGML_QK_K / (4 * 2))  // = 32
#define GGML_QR5_K        2
#define GGML_VDR_Q5_K     2
#define GGML_WARP_SIZE    32

// ---- Helper: warp reduction (matches ggml common.cuh) ----
__device__ __forceinline__ float ggml_warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

// ---- Helper: warp-wide max reduction ----
__device__ __forceinline__ float ggml_warp_reduce_max(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

// ---- vec_dot_q5_K_q8_1_impl_vmmq (from vecdotq.cuh) ----
// Computes dot product of one Q5_K sub-group against Q8_1 blocks.
__device__ __forceinline__ float ggml_vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh,
    const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5,
    const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < GGML_QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = __dp4a(v0i, u[2*i+0], __dp4a(v1i, u[2*i+1], 0));
        const int dot2 = __dp4a(0x01010101, u[2*i+0], __dp4a(0x01010101, u[2*i+1], 0));

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }

    const float2 dm5f = __half22float2(dm5);

    return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

// ---- vec_dot_q5_K_q8_1 (from vecdotq.cuh) ----
// Extracts data from Q5_K + Q8_1 blocks and calls impl_vmmq.
__device__ __forceinline__ float ggml_vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const struct ggml_block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const struct ggml_block_q5_K * bq5_K = (const struct ggml_block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*GGML_QR5_K];
    float d8[GGML_QR5_K];

    // QI8_1 = QK8_1 / (4 * QR8_1) = 32 / (4 * 1) = 8
    // bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2)) = 2 * ((iqs/2) / 4)
    const int bq8_off = GGML_QR5_K * ((iqs/2) / 4);
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_off + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_off;
    vh[1] = qh[4] >> bq8_off;

    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    const int j = bq8_off/2;
    if (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * mn = sc + 2;

    // Construct dm as half2 from bq5_K->d and bq5_K->dmin
    half2 dm5 = make_half2(bq5_K->d, bq5_K->dmin);

#pragma unroll
    for (int i = 0; i < GGML_QR5_K; ++i) {
        const struct ggml_block_q8_1 * bq8i = bq8_1 + bq8_off + i;
        d8[i] = __low2float(bq8i->ds);

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return ggml_vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, mn, dm5, d8);
}

// ---- ggml MMVQ kernel: Q5_K x Q8_1, ncols_y=1, nwarps=4 ----
// Launch config: gridDim.x = nrows, blockDim = (32, 4, 1) = 128 threads
// Each block computes one output row.
extern "C" __launch_bounds__(128, 1)
__global__ void ggml_mul_mat_vec_q5_K_q8_1(
    const void * __restrict__ vx,   // Q5_K weight data [nrows * blocks_per_row * 176 bytes]
    const void * __restrict__ vy,   // Q8_1 quantized input [ncols/32 * 36 bytes]
    float * __restrict__ dst,        // output [nrows]
    const int ncols,                 // input dimension (must be multiple of 256)
    const int nrows                  // output dimension
) {
    // Constants for Q5_K, ncols_y=1, nwarps=4
    const int nwarps = 4;
    const int qi     = GGML_QI5_K;     // 32
    const int qk     = GGML_QK_K;      // 256
    const int vdr    = GGML_VDR_Q5_K;   // 2

    const int tid = GGML_WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;  // 1 row per block (rows_per_cuda_block = 1 when ncols_y < 4)

    if (row >= nrows) return;

    const int blocks_per_row = ncols / qk;
    const int blocks_per_iter = vdr * nwarps * GGML_WARP_SIZE / qi;  // 2*4*32/32 = 8

    float tmp = 0.0f;

    const struct ggml_block_q8_1 * y = (const struct ggml_block_q8_1 *) vy;

    for (int kbx = tid / (qi/vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / GGML_QK8_1);  // y block index aligned with kbx
        const int kqs = vdr * (tid % (qi/vdr));     // quant sub-index

        tmp += ggml_vec_dot_q5_K_q8_1(vx, &y[kby], row * blocks_per_row + kbx, kqs);
    }

    // Cross-warp reduction via shared memory
    __shared__ float tmp_shared[3][GGML_WARP_SIZE]; // nwarps-1 = 3

    if (threadIdx.y > 0) {
        tmp_shared[threadIdx.y - 1][threadIdx.x] = tmp;
    }
    __syncthreads();

    if (threadIdx.y > 0) return;

    // Warp 0 aggregates all partial sums
#pragma unroll
    for (int l = 0; l < 3; ++l) {
        tmp += tmp_shared[l][threadIdx.x];
    }

    tmp = ggml_warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[row] = tmp;
    }
}

// ---- ggml-compatible Q8_1 quantization kernel ----
// Quantizes a float vector to Q8_1 format.
// Launch config: gridDim.x = ceil(ncols/32), blockDim.x = 32
// Each warp of 32 threads quantizes one block of 32 floats.
extern "C" __global__ void ggml_quantize_q8_1(
    const float * __restrict__ x,    // input [ncols]
    void * __restrict__ vy,           // output Q8_1 [ncols/32 * 36 bytes]
    const int ncols
) {
    const int ib = blockIdx.x;                // block index
    const int iqs = threadIdx.x;              // quant index within block (0..31)

    if (ib * GGML_QK8_1 + iqs >= ncols) return;

    struct ggml_block_q8_1 * y = (struct ggml_block_q8_1 *) vy;

    const float xi = x[ib * GGML_QK8_1 + iqs];
    float amax = fabsf(xi);
    float sum = xi;

    // Warp-wide reductions
    amax = ggml_warp_reduce_max(amax);
    sum  = ggml_warp_reduce_sum(sum);

    const float d = amax / 127.0f;
    const int8_t q = (amax == 0.0f) ? (int8_t)0 : (int8_t)roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) return;

    y[ib].ds = make_half2(__float2half(d), __float2half(sum));
}


// =====================================================================
// 11. Additional elementwise kernels (from elementwise.rs)
// =====================================================================

// Batched silu_mul: processes top_k experts in a single launch.
// Grid: (ceil(expert_ffn/256), top_k, 1)  Block: (256, 1, 1)
// gate_all/up_all/inter_all are [top_k * expert_ffn] contiguous.
extern "C" __global__ void silu_mul_batched_kernel(
    const float* __restrict__ gate_all,
    const float* __restrict__ up_all,
    float* __restrict__ inter_all,
    int expert_ffn,
    int top_k
) {
    int k = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= expert_ffn || k >= top_k) return;
    int idx = k * expert_ffn + i;
    float g = gate_all[idx];
    float s = g / (1.0f + expf(-g));
    inter_all[idx] = s * up_all[idx];
}

// Batched weighted combine: replaces top_k weighted_add launches with 1.
// Grid: (ceil(hidden_size/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void weighted_combine_kernel(
    const float* __restrict__ expert_outs,
    const float* __restrict__ weights,
    float* __restrict__ combined,
    int hidden_size,
    int top_k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden_size) return;
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        sum += weights[k] * expert_outs[k * hidden_size + i];
    }
    combined[i] = sum;
}

// L2-normalize groups: input[g*d..g*d+d] /= ||input[g*d..g*d+d]||_2
// Grid: (n_group, 1, 1), Block: (blockDim, 1, 1) where blockDim >= d
extern "C" __global__ void l2_norm_groups_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int d_state,
    int n_group,
    float eps
) {
    int g = blockIdx.x;
    int j = threadIdx.x;
    extern __shared__ char smem_raw[];
    float* sdata = (float*)smem_raw;
    int base = g * d_state;
    float val = 0.0f;
    float sq = 0.0f;
    if (j < d_state) {
        val = input[base + j];
        sq = val * val;
    }
    sdata[j] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) sdata[j] += sdata[j + s];
        __syncthreads();
    }
    if (j >= d_state) return;
    float norm = sqrtf(sdata[0] + eps);
    output[base + j] = val / norm;
}

// Expand groups: repeat entire tensor 'repeats' times (TILED layout).
// Grid: (ceil(total/256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void expand_groups_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int d_state,
    int n_group,
    int repeats
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_group * repeats * d_state;
    if (i >= total) return;
    int head = i / d_state;
    int dim = i % d_state;
    int group = head % n_group;
    output[i] = input[group * d_state + dim];
}

// Scale: output[i] = input[i] * scale
extern "C" __global__ void scale_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n,
    float scale
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = input[i] * scale;
}

// In-place add: output[i] += input[i]
extern "C" __global__ void add_inplace_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] += input[i];
}


// =====================================================================
// 12. Fused conv1d + SiLU + state update kernel (from elementwise.rs)
// =====================================================================

extern "C" __global__ void fused_conv1d_silu_update(
    const float* __restrict__ conv_state,    // [channels, conv_kernel-1]
    const float* __restrict__ new_input,     // [channels]
    const float* __restrict__ conv_weight,   // [channels, conv_kernel]
    float* __restrict__ output,              // [channels] (after silu)
    float* __restrict__ new_state,           // [channels, conv_kernel-1]
    int channels,
    int conv_kernel
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    int km1 = conv_kernel - 1;

    float sum = 0.0f;
    for (int k = 0; k < km1; k++) {
        float s = conv_state[ch * km1 + k];
        sum += s * conv_weight[ch * conv_kernel + k];
        if (k < km1 - 1)
            new_state[ch * km1 + k] = conv_state[ch * km1 + k + 1];
        else
            new_state[ch * km1 + k] = new_input[ch];
    }
    sum += new_input[ch] * conv_weight[ch * conv_kernel + km1];

    float silu = sum / (1.0f + expf(-sum));
    output[ch] = silu;
}


// =====================================================================
// 13. Q5_K Q8_1 dual GEMV kernel (from q5k_gemv.rs)
// =====================================================================

// Dual GEMV: blockIdx.y selects projection (0 = A, 1 = B).
// Both projections share the same Q8_1 quantized input.
// Grid: (max(rows_a, rows_b), 2, 1),  Block: (128, 1, 1)
extern "C" __global__ void gemv_q5k_q8_dual(
    const unsigned char* __restrict__ weights_a,
    const unsigned char* __restrict__ weights_b,
    float*               __restrict__ output_a,
    float*               __restrict__ output_b,
    int                               cols,
    int                               rows_a,
    int                               rows_b,
    const unsigned char* __restrict__ q8_input
) {
    const int proj = blockIdx.y;
    const int row  = blockIdx.x;
    const int rows = (proj == 0) ? rows_a : rows_b;
    if (row >= rows) return;

    const unsigned char* weights = (proj == 0) ? weights_a : weights_b;
    float* output = (proj == 0) ? output_a : output_b;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int n_q5k_blocks = cols >> 8;
    const int Q5K_BYTES = 176;
    extern __shared__ char smem[];
    float* warp_sums = (float*)smem;
    const struct block_q8_1_q5k* q8_blocks = (const struct block_q8_1_q5k*)q8_input;
    const unsigned char* row_weights = weights + (long long)row * (long long)n_q5k_blocks * Q5K_BYTES;
    float sum = 0.0f;
    const int total_subblocks = n_q5k_blocks * 8;
    for (int sg_idx = tid; sg_idx < total_subblocks; sg_idx += 128) {
        int kbx = sg_idx >> 3;
        int sg  = sg_idx & 7;
        sum += dot_q5k_q8_subblock(row_weights + kbx * Q5K_BYTES, &q8_blocks[kbx * 8 + sg], sg);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane_id == 0) warp_sums[warp_id] = sum;
    __syncthreads();
    if (tid == 0) {
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    }
}


// =====================================================================
// 14. Attention raw kernels (from attention_raw.rs)
// =====================================================================

// Deinterleave Q+gate from Qwen3.5 interleaved layout.
// Grid: (num_heads, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void deinterleave_q_gate_kernel(
    const float* __restrict__ input,
    float*       __restrict__ q_out,
    float*       __restrict__ gate_out,
    int num_heads,
    int head_dim
) {
    int h = blockIdx.x;
    int d = threadIdx.x;

    int in_base = h * 2 * head_dim;
    int out_base = h * head_dim;

    for (int i = d; i < head_dim; i += blockDim.x) {
        q_out[out_base + i]    = input[in_base + i];
        gate_out[out_base + i] = input[in_base + head_dim + i];
    }
}

// Per-head RMSNorm.
// Grid: (num_heads, 1, 1)   Block: (block_size, 1, 1)
// Shared mem: block_size * sizeof(float)
extern "C" __global__ void rms_norm_per_head_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float*       __restrict__ output,
    int num_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ char smem_perhead[];
    float* sdata = (float*)smem_perhead;

    int base = h * head_dim;

    float local_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float v = input[base + i];
        local_sq += v * v;
    }
    sdata[tid] = local_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float rms = sqrtf(sdata[0] / (float)head_dim + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < head_dim; i += blockDim.x) {
        output[base + i] = input[base + i] * inv_rms * weight[i];
    }
}

// sigmoid(gate) * attn_output
// Grid: (ceil(n/256), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void sigmoid_gate_mul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gate,
    float*       __restrict__ output,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = 1.0f / (1.0f + expf(-gate[i]));
        output[i] = g * input[i];
    }
}

// MRoPE — Multi-Resolution Rotary Position Embedding.
// Grid: (num_heads, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void mrope_apply_kernel(
    const float* __restrict__ input,
    float*       __restrict__ output,
    const float* __restrict__ cos_s0,
    const float* __restrict__ sin_s0,
    const float* __restrict__ cos_s1,
    const float* __restrict__ sin_s1,
    const float* __restrict__ cos_s2,
    const float* __restrict__ sin_s2,
    int pos_s0, int pos_s1, int pos_s2,
    int num_heads, int head_dim,
    int pairs_s0, int pairs_s1, int pairs_s2,
    int n_rot
) {
    int h = blockIdx.x;
    int base = h * head_dim;

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
