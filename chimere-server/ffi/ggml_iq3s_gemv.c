// ggml_iq3s_gemv.c -- Thin C wrapper around ggml's IQ3_S CPU GEMV.
//
// Provides CPU-side IQ3_S matmul using ggml's AVX2-optimized kernels.
// Used for the "invert ncmoe" approach: instead of copying 10 MB of expert
// weights to GPU, copy 8 KB of hidden state to CPU and do the matmul here.
//
// Links against libggml.so from the ik_llama.cpp build.

#include "ggml_iq3s_gemv.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// IQ3_S block size: 110 bytes per 256 elements
// Q8_K  block size: 296 bytes per 256 elements
// ---------------------------------------------------------------------------

// sizeof(block_iq3_s) = 2 + 13*(256/32) + 256/64 = 110
#define IQ3S_BLOCK_BYTES 110
#define IQ3S_BLOCK_ELEMS 256

// sizeof(block_q8_K) = 4 + 4 + 256 + 32 = 296
#define Q8K_BLOCK_BYTES 296
#define Q8K_BLOCK_ELEMS 256

// ---------------------------------------------------------------------------
// Quantize f32 -> Q8_K
// ---------------------------------------------------------------------------

void ggml_ffi_quantize_q8_K(const float* x, void* y, int64_t k) {
    quantize_row_q8_K(x, y, k);
}

// ---------------------------------------------------------------------------
// IQ3_S dot product (wraps ggml_vec_dot_iq3_s_q8_K)
// ---------------------------------------------------------------------------

void ggml_ffi_vec_dot_iq3s_q8K(int n, float* s,
                                const void* vx, const void* vy) {
    ggml_vec_dot_iq3_s_q8_K(n, s, 0, vx, 0, vy, 0, 1);
}

// ---------------------------------------------------------------------------
// Full GEMV: y = W_iq3s @ x_f32
// ---------------------------------------------------------------------------

void ggml_ffi_gemv_iq3s(const void* w_iq3s, const float* x_f32,
                         float* y_f32, int nrows, int ncols) {
    // 1. Quantize input to Q8_K (once for all rows)
    int nblocks = ncols / Q8K_BLOCK_ELEMS;
    size_t q8k_size = (size_t)nblocks * Q8K_BLOCK_BYTES;
    void* q8k_buf = malloc(q8k_size);
    if (!q8k_buf) return;
    quantize_row_q8_K(x_f32, q8k_buf, (int64_t)ncols);

    // 2. Compute dot product for each row
    size_t row_bytes = (size_t)nblocks * IQ3S_BLOCK_BYTES;
    const uint8_t* w = (const uint8_t*)w_iq3s;

    for (int row = 0; row < nrows; row++) {
        ggml_vec_dot_iq3_s_q8_K(
            ncols,
            &y_f32[row],
            0,                          // bs (unused)
            w + (size_t)row * row_bytes, // IQ3_S row
            0,                          // bx (unused)
            q8k_buf,                    // Q8_K activation
            0,                          // by (unused)
            1                           // nrc
        );
    }

    free(q8k_buf);
}

// ---------------------------------------------------------------------------
// Parallel GEMV using OpenMP
// ---------------------------------------------------------------------------

void ggml_ffi_gemv_iq3s_parallel(const void* w_iq3s, const float* x_f32,
                                  float* y_f32, int nrows, int ncols,
                                  int nthreads) {
    // 1. Quantize input to Q8_K (once for all rows)
    int nblocks = ncols / Q8K_BLOCK_ELEMS;
    size_t q8k_size = (size_t)nblocks * Q8K_BLOCK_BYTES;
    void* q8k_buf = malloc(q8k_size);
    if (!q8k_buf) return;
    quantize_row_q8_K(x_f32, q8k_buf, (int64_t)ncols);

    // 2. Parallel dot products across rows
    size_t row_bytes = (size_t)nblocks * IQ3S_BLOCK_BYTES;
    const uint8_t* w = (const uint8_t*)w_iq3s;

    #ifdef _OPENMP
    if (nthreads > 0) {
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (int row = 0; row < nrows; row++) {
            ggml_vec_dot_iq3_s_q8_K(
                ncols, &y_f32[row], 0,
                w + (size_t)row * row_bytes, 0,
                q8k_buf, 0, 1
            );
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < nrows; row++) {
            ggml_vec_dot_iq3_s_q8_K(
                ncols, &y_f32[row], 0,
                w + (size_t)row * row_bytes, 0,
                q8k_buf, 0, 1
            );
        }
    }
    #else
    // Fallback: sequential
    for (int row = 0; row < nrows; row++) {
        ggml_vec_dot_iq3_s_q8_K(
            ncols, &y_f32[row], 0,
            w + (size_t)row * row_bytes, 0,
            q8k_buf, 0, 1
        );
    }
    #endif

    free(q8k_buf);
}

// ---------------------------------------------------------------------------
// Size helpers
// ---------------------------------------------------------------------------

size_t ggml_ffi_iq3s_row_bytes(int ncols) {
    return (size_t)(ncols / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
}

size_t ggml_ffi_q8k_row_bytes(int ncols) {
    return (size_t)(ncols / Q8K_BLOCK_ELEMS) * Q8K_BLOCK_BYTES;
}
