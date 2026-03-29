#ifndef GGML_IQ3S_GEMV_H
#define GGML_IQ3S_GEMV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Quantize an f32 vector to Q8_K format.
///
/// Q8_K block: 256 elements, 296 bytes:
///   float d (4) + float sum (4) + int8_t qs[256] (256) + int16_t bsums[16] (32)
///
/// @param x     Input f32 vector, length k
/// @param y     Output Q8_K buffer, (k/256)*296 bytes
/// @param k     Number of elements (must be multiple of 256)
void ggml_ffi_quantize_q8_K(const float* x, void* y, int64_t k);

/// Compute the dot product of one IQ3_S super-block with one Q8_K block.
///
/// Both blocks cover 256 elements. The IQ3_S block is the weight,
/// the Q8_K block is the quantized activation.
///
/// @param n     Number of elements (must be multiple of 256)
/// @param s     Output: scalar dot product result
/// @param vx    IQ3_S weight block(s)
/// @param vy    Q8_K activation block(s)
void ggml_ffi_vec_dot_iq3s_q8K(int n, float* s,
                                const void* vx, const void* vy);

/// Full GEMV: y = W @ x where W is IQ3_S [nrows, ncols] and x is f32 [ncols].
///
/// Internally quantizes x to Q8_K, then computes one dot product per row
/// using ggml's AVX2-optimized ggml_vec_dot_iq3_s_q8_K.
///
/// @param w_iq3s   Raw IQ3_S weight bytes, row-major [nrows, ncols]
/// @param x_f32    Input f32 vector, length ncols
/// @param y_f32    Output f32 vector, length nrows (written on return)
/// @param nrows    Number of output features
/// @param ncols    Number of input features (must be multiple of 256)
void ggml_ffi_gemv_iq3s(const void* w_iq3s, const float* x_f32,
                         float* y_f32, int nrows, int ncols);

/// Parallel GEMV: same as ggml_ffi_gemv_iq3s but uses OpenMP for rows.
///
/// @param w_iq3s   Raw IQ3_S weight bytes, row-major [nrows, ncols]
/// @param x_f32    Input f32 vector, length ncols
/// @param y_f32    Output f32 vector, length nrows
/// @param nrows    Number of output features
/// @param ncols    Number of input features (must be multiple of 256)
/// @param nthreads Number of OpenMP threads to use (0 = auto)
void ggml_ffi_gemv_iq3s_parallel(const void* w_iq3s, const float* x_f32,
                                  float* y_f32, int nrows, int ncols,
                                  int nthreads);

/// Return the byte size of one IQ3_S row of ncols elements.
/// ncols must be a multiple of 256.
size_t ggml_ffi_iq3s_row_bytes(int ncols);

/// Return the byte size of one Q8_K row of ncols elements.
/// ncols must be a multiple of 256.
size_t ggml_ffi_q8k_row_bytes(int ncols);

#ifdef __cplusplus
}
#endif

#endif // GGML_IQ3S_GEMV_H
