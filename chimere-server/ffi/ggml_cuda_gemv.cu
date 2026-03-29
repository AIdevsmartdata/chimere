// ggml_cuda_gemv.cu -- Thin C++ wrapper exposing ggml's optimized MMVQ CUDA
// kernels as extern "C" functions callable from Rust via FFI.
//
// This replaces chimere's naive CUDA GEMV kernels with ggml's optimized
// implementations (shared memory tiling, warp shuffles, importance grid
// prefetch) for a ~4x compute speedup.
//
// The wrapper constructs the `mmvq_args` struct required by ggml's internal
// MMVQ template dispatch, then calls the pre-compiled kernel launchers from
// libggml.so.
//
// Supported quant types:
//   - IQ3_S (experts in custom-mix)
//   - Q5_K  (attention projections)
//   - Q8_0  (lm_head, embeddings)
//
// Compile: nvcc -c -arch=sm_120 -O3 -I<ggml_include> -I<ggml_src> ...
// Link:    -lggml

#include <cstdint>
#include <cstring>
#include <cmath>

// Defines needed by ggml CUDA headers
#ifndef GGML_CUDA_FUSION
#define GGML_CUDA_FUSION 0
#endif
#ifndef GGML_CUDA_MIN_BATCH_OFFLOAD
#define GGML_CUDA_MIN_BATCH_OFFLOAD 256
#endif

// ggml internal CUDA headers
#include "common.cuh"
#include "mmvq-args.h"
#include "quantize.cuh"

// ---------------------------------------------------------------------------
// MMVQ kernel launcher declarations (defined in libggml.so template instances)
// ---------------------------------------------------------------------------

extern void mul_mat_vec_iq3_s_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q5_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q8_0_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q4_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_q6_K_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);
extern void mul_mat_vec_mxfp4_q8_1_cuda(const mmvq_args & args, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Helper: construct a simple mmvq_args for 2D GEMV (no batching, no bias,
// no ids, single vector input)
//
// This is the "simple case": weights[nrows, ncols] @ input_q8[ncols] -> out[nrows]
// ---------------------------------------------------------------------------

static mmvq_args make_simple_args(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded  // = pad(ncols, 512) / QK8_1  for Q8_1 block count
) {
    // mmvq_args has const int members — must use brace initialization
    mmvq_args args = {
        /* .vx_u     = */ weights,
        /* .vx_g     = */ nullptr,
        /* .bias_u   = */ nullptr,
        /* .bias_g   = */ nullptr,
        /* .vy       = */ input_q8,
        /* .dst      = */ output,
        /* .ids_data = */ nullptr,
        /* .ncols_x  = */ ncols,
        /* .nrows_x  = */ nrows,
        /* .nrows_y  = */ nrows_y_padded,
        /* .ncols_y  = */ 1,
        /* .nrows_dst= */ nrows,
        /* .ne2      = */ 1,
        /* .nb02     = */ (int64_t)0,
        /* .nb12     = */ (int64_t)0,
        /* .nb2      = */ (int64_t)0,
        /* .ids_nb0  = */ (int64_t)0,
        /* .bias_nb1 = */ (int64_t)0,
        /* .unary_op = */ GGML_UNARY_OP_COUNT,
        /* .limit    = */ INFINITY,
    };
    return args;
}


// ---------------------------------------------------------------------------
// Extern "C" API: Simple GEMV wrappers
// ---------------------------------------------------------------------------

extern "C" {

/// IQ3_S GEMV: weights[nrows, ncols] @ input_q8[ncols] -> output[nrows]
///
/// @param weights:   IQ3_S quantized weight data on GPU
/// @param input_q8:  Q8_1 quantized input on GPU (from chimere_quantize_q8_1)
/// @param output:    F32 output buffer on GPU, size >= nrows
/// @param ncols:     number of columns (input dimension, must be multiple of 256)
/// @param nrows:     number of rows (output dimension)
/// @param nrows_y_padded: padded ncols / QK8_1 (= number of Q8_1 blocks)
/// @param stream:    CUDA stream (0 = default)
void chimere_mmvq_iq3s(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded,
    cudaStream_t stream
) {
    mmvq_args args = make_simple_args(weights, input_q8, output, ncols, nrows, nrows_y_padded);
    mul_mat_vec_iq3_s_q8_1_cuda(args, stream);
}

/// Q5_K GEMV: weights[nrows, ncols] @ input_q8[ncols] -> output[nrows]
void chimere_mmvq_q5k(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded,
    cudaStream_t stream
) {
    mmvq_args args = make_simple_args(weights, input_q8, output, ncols, nrows, nrows_y_padded);
    mul_mat_vec_q5_K_q8_1_cuda(args, stream);
}

/// Q8_0 GEMV: weights[nrows, ncols] @ input_q8[ncols] -> output[nrows]
void chimere_mmvq_q8_0(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded,
    cudaStream_t stream
) {
    mmvq_args args = make_simple_args(weights, input_q8, output, ncols, nrows, nrows_y_padded);
    mul_mat_vec_q8_0_q8_1_cuda(args, stream);
}

/// Q4_K GEMV: weights[nrows, ncols] @ input_q8[ncols] -> output[nrows]
void chimere_mmvq_q4k(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded,
    cudaStream_t stream
) {
    mmvq_args args = make_simple_args(weights, input_q8, output, ncols, nrows, nrows_y_padded);
    mul_mat_vec_q4_K_q8_1_cuda(args, stream);
}

/// Q6_K GEMV: weights[nrows, ncols] @ input_q8[ncols] -> output[nrows]
void chimere_mmvq_q6k(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded,
    cudaStream_t stream
) {
    mmvq_args args = make_simple_args(weights, input_q8, output, ncols, nrows, nrows_y_padded);
    mul_mat_vec_q6_K_q8_1_cuda(args, stream);
}

/// Q8_1 quantization of F32 input on GPU.
///
/// Quantizes `ncols` floats from `input` into Q8_1 blocks in `output`.
/// The output buffer must be at least `pad(ncols, 512) / 32 * 36` bytes.
///
/// @param input:   F32 input on GPU, size >= ncols
/// @param output:  Q8_1 output on GPU
/// @param ncols:   number of elements to quantize
/// @param ncols_padded: ncols padded to MATRIX_ROW_PADDING (512)
/// @param stream:  CUDA stream
void chimere_quantize_q8_1(
    const float * input,
    void * output,
    int64_t ncols,
    int64_t ncols_padded,
    cudaStream_t stream
) {
    // quantize_row_q8_1_cuda(x, vy, kx0, kx1, channels, kx0_padded, type_x, stream)
    // For single vector: kx0=ncols, kx1=1, channels=1, kx0_padded=ncols_padded
    quantize_row_q8_1_cuda(input, output, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);
}

/// Return the byte size of one Q8_1 block (36 bytes = 32 int8 quants + f16 d + f16 sum).
int chimere_q8_1_block_bytes(void) {
    return sizeof(block_q8_1);  // Should be 36
}

/// Return QK8_1 (number of elements per Q8_1 block = 32).
int chimere_q8_1_block_elems(void) {
    return QK8_1;  // 32
}

/// Return MATRIX_ROW_PADDING (512).
int chimere_matrix_row_padding(void) {
    return MATRIX_ROW_PADDING;  // 512
}

// ---------------------------------------------------------------------------
// Phase 1.2: Batched MMVQ with GPU-resident expert indirection
//
// These functions accept an `expert_ids` array on GPU (from topk_softmax)
// and dispatch all `top_k` expert GEMVs in a single kernel launch using
// ggml's native `ids_data` indirection in the MMVQ template.
//
// The MMVQ template uses blockIdx.y to iterate over experts:
//   int i02 = ids_data ? *(int*)(ids_data + blockIdx.y * ids_nb0) : blockIdx.y;
//   const char * cx = (char*)vx + i02 * nb02;  // weight slice for this expert
//   const char * cy = (char*)vy + blockIdx.y * nb12;  // input (0 = shared)
//   char * cdst = (char*)dst + blockIdx.y * nb2;  // output slice
//
// Zero CPU sync. Zero memcpy_dtoh. All expert routing stays on GPU.
// ---------------------------------------------------------------------------

/// Batched IQ3_S MMVQ with expert indirection.
///
/// Dispatches `top_k` expert GEMVs in a single kernel launch.
/// Expert IDs are read from GPU memory (`expert_ids[top_k]`).
///
/// @param all_weights:        All expert weights stacked contiguously [256 * expert_stride]
/// @param input_q8:           Q8_1 quantized input(s) on GPU
/// @param output:             F32 output buffer [top_k * nrows]
/// @param expert_ids:         int32 expert IDs on GPU [top_k] (from topk_softmax)
/// @param ncols:              Input dimension (must be multiple of 256)
/// @param nrows:              Output dimension per expert
/// @param nrows_y_padded:     pad(ncols, 512) / QK8_1 (Q8_1 block count)
/// @param expert_stride:      Bytes per expert in the weight tensor
/// @param input_stride:       Bytes between per-expert Q8_1 inputs (0 = shared input)
/// @param top_k:              Number of selected experts (8)
/// @param stream:             CUDA stream
void chimere_mmvq_iq3s_batched(
    const void * all_weights,
    const void * input_q8,
    float * output,
    const int * expert_ids,
    int ncols,
    int nrows,
    int nrows_y_padded,
    int64_t expert_stride,
    int64_t input_stride,
    int top_k,
    cudaStream_t stream
) {
    mmvq_args args = {
        /* .vx_u     = */ all_weights,
        /* .vx_g     = */ nullptr,
        /* .bias_u   = */ nullptr,
        /* .bias_g   = */ nullptr,
        /* .vy       = */ input_q8,
        /* .dst      = */ output,
        /* .ids_data = */ (const char *)expert_ids,
        /* .ncols_x  = */ ncols,
        /* .nrows_x  = */ nrows,
        /* .nrows_y  = */ nrows_y_padded,
        /* .ncols_y  = */ 1,
        /* .nrows_dst= */ nrows,
        /* .ne2      = */ top_k,
        /* .nb02     = */ (uint64_t)expert_stride,
        /* .nb12     = */ (uint64_t)input_stride,
        /* .nb2      = */ (uint64_t)(nrows * sizeof(float)),
        /* .ids_nb0  = */ (uint64_t)sizeof(int),
        /* .bias_nb1 = */ (uint64_t)0,
        /* .unary_op = */ GGML_UNARY_OP_COUNT,
        /* .limit    = */ INFINITY,
    };
    mul_mat_vec_iq3_s_q8_1_cuda(args, stream);
}

// ---------------------------------------------------------------------------
// Fused F32->Q8_1->MMVQ: Eliminates one Rust->C FFI round trip per GEMV.
//
// Each function takes F32 input directly, quantizes to Q8_1 into a
// caller-provided scratch buffer, then dispatches the MMVQ kernel.
// Both CUDA kernels serialize on the same stream -- no extra sync needed.
//
// Saves ~5us of FFI/launch overhead per call. With ~80 calls/token,
// this reclaims ~0.4ms/token.
// ---------------------------------------------------------------------------

/// Fused Q5_K GEMV from F32 input: quantize F32->Q8_1 + MMVQ in one FFI call.
///
/// @param weights:       Q5_K quantized weight data on GPU
/// @param input_f32:     F32 input vector on GPU [ncols]
/// @param output:        F32 output buffer on GPU [nrows]
/// @param ncols:         number of columns (input dimension)
/// @param nrows:         number of rows (output dimension)
/// @param ncols_padded:  ncols padded to MATRIX_ROW_PADDING (512)
/// @param q8_scratch:    pre-allocated Q8_1 scratch buffer on GPU
/// @param stream:        CUDA stream
void chimere_mmvq_q5k_f32(
    const void * weights,
    const float * input_f32,
    float * output,
    int ncols,
    int nrows,
    int ncols_padded,
    void * q8_scratch,
    cudaStream_t stream
) {
    // Step 1: Quantize F32 -> Q8_1 into scratch buffer
    quantize_row_q8_1_cuda(input_f32, q8_scratch, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);

    // Step 2: Run MMVQ with Q8_1 input from scratch
    int nrows_y = ncols_padded / QK8_1;
    mmvq_args args = make_simple_args(weights, q8_scratch, output, ncols, nrows, nrows_y);
    mul_mat_vec_q5_K_q8_1_cuda(args, stream);
}

/// Fused IQ3_S GEMV from F32 input.
void chimere_mmvq_iq3s_f32(
    const void * weights,
    const float * input_f32,
    float * output,
    int ncols,
    int nrows,
    int ncols_padded,
    void * q8_scratch,
    cudaStream_t stream
) {
    quantize_row_q8_1_cuda(input_f32, q8_scratch, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);
    int nrows_y = ncols_padded / QK8_1;
    mmvq_args args = make_simple_args(weights, q8_scratch, output, ncols, nrows, nrows_y);
    mul_mat_vec_iq3_s_q8_1_cuda(args, stream);
}

/// Fused Q8_0 GEMV from F32 input.
void chimere_mmvq_q8_0_f32(
    const void * weights,
    const float * input_f32,
    float * output,
    int ncols,
    int nrows,
    int ncols_padded,
    void * q8_scratch,
    cudaStream_t stream
) {
    quantize_row_q8_1_cuda(input_f32, q8_scratch, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);
    int nrows_y = ncols_padded / QK8_1;
    mmvq_args args = make_simple_args(weights, q8_scratch, output, ncols, nrows, nrows_y);
    mul_mat_vec_q8_0_q8_1_cuda(args, stream);
}

/// Fused Q4_K GEMV from F32 input.
void chimere_mmvq_q4k_f32(
    const void * weights,
    const float * input_f32,
    float * output,
    int ncols,
    int nrows,
    int ncols_padded,
    void * q8_scratch,
    cudaStream_t stream
) {
    quantize_row_q8_1_cuda(input_f32, q8_scratch, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);
    int nrows_y = ncols_padded / QK8_1;
    mmvq_args args = make_simple_args(weights, q8_scratch, output, ncols, nrows, nrows_y);
    mul_mat_vec_q4_K_q8_1_cuda(args, stream);
}

/// Fused Q6_K GEMV from F32 input.
void chimere_mmvq_q6k_f32(
    const void * weights,
    const float * input_f32,
    float * output,
    int ncols,
    int nrows,
    int ncols_padded,
    void * q8_scratch,
    cudaStream_t stream
) {
    quantize_row_q8_1_cuda(input_f32, q8_scratch, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);
    int nrows_y = ncols_padded / QK8_1;
    mmvq_args args = make_simple_args(weights, q8_scratch, output, ncols, nrows, nrows_y);
    mul_mat_vec_q6_K_q8_1_cuda(args, stream);
}

/// Batched Q8_1 quantization: quantize `batch` vectors of `ncols` floats each.
///
/// Input layout: input[batch * ncols] (contiguous)
/// Output layout: output[batch * q8_row_bytes] (contiguous)
void chimere_quantize_q8_1_batched(
    const float * input,
    void * output,
    int64_t ncols,
    int64_t ncols_padded,
    int batch,
    cudaStream_t stream
) {
    // quantize_row_q8_1_cuda(x, vy, kx0, kx1, channels, kx0_padded, type_x, stream)
    // kx0=ncols per row, kx1=batch (number of rows), channels=1
    quantize_row_q8_1_cuda(input, output, ncols, batch, 1, ncols_padded, GGML_TYPE_F32, stream);
}

// ---------------------------------------------------------------------------
// MXFP4 GEMV (4.25 BPW, E2M1 + E8M0 scale, Blackwell-optimized)
// ---------------------------------------------------------------------------

/// MXFP4 GEMV: weights[nrows, ncols] @ input_q8[ncols] -> output[nrows]
void chimere_mmvq_mxfp4(
    const void * weights,
    const void * input_q8,
    float * output,
    int ncols,
    int nrows,
    int nrows_y_padded,
    cudaStream_t stream
) {
    mmvq_args args = make_simple_args(weights, input_q8, output, ncols, nrows, nrows_y_padded);
    mul_mat_vec_mxfp4_q8_1_cuda(args, stream);
}

/// MXFP4 batched expert GEMV (for MoE layers)
void chimere_mmvq_mxfp4_batched(
    const void * all_weights,
    const void * input_q8,
    float * output,
    const int * expert_ids,
    int ncols,
    int nrows,
    int nrows_y_padded,
    int64_t expert_stride,
    int64_t input_stride,
    int top_k,
    cudaStream_t stream
) {
    for (int k = 0; k < top_k; k++) {
        int eid = expert_ids[k];
        const void * w = (const char *)all_weights + (int64_t)eid * expert_stride;
        const void * x = (const char *)input_q8 + (int64_t)k * input_stride;
        mmvq_args args = make_simple_args(w, x, output + k * nrows, ncols, nrows, nrows_y_padded);
        mul_mat_vec_mxfp4_q8_1_cuda(args, stream);
    }
}

/// Fused F32→Q8_1→MXFP4 GEMV (no intermediate buffer)
void chimere_mmvq_mxfp4_f32(
    const void * weights,
    const float * input_f32,
    void * q8_scratch,
    float * output,
    int ncols,
    int nrows,
    int ncols_padded,
    cudaStream_t stream
) {
    int nrows_y_padded = ncols_padded / QK8_1;
    quantize_row_q8_1_cuda(input_f32, q8_scratch, ncols, 1, 1, ncols_padded, GGML_TYPE_F32, stream);
    chimere_mmvq_mxfp4(weights, q8_scratch, output, ncols, nrows, nrows_y_padded, stream);
}

} // extern "C"
