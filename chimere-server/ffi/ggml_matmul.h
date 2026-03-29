#ifndef GGML_MATMUL_H
#define GGML_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to ggml CUDA backend + scratch context
typedef struct ggml_ffi_ctx ggml_ffi_ctx_t;

// Initialize ggml CUDA backend for given device.
// Returns NULL on failure.
ggml_ffi_ctx_t* ggml_ffi_init_cuda(int device_id);

// Free the context and all ggml resources.
void ggml_ffi_free(ggml_ffi_ctx_t* ctx);

// Compute y = x @ W^T for quantized W on GPU.
//
// w_data: raw quantized weight bytes (host pointer, will be uploaded to GPU)
// w_type: ggml type enum value (e.g., GGML_TYPE_Q5_K=13, GGML_TYPE_IQ3_S=21)
// rows:   number of output features (W is [rows, cols])
// cols:   number of input features  (= dimension of x)
// x:      float32 input vector  [1, cols]  (host pointer)
// y:      float32 output vector [1, rows]  (host pointer, written on return)
void ggml_ffi_mul_mat_vec(
    ggml_ffi_ctx_t* ctx,
    const void* w_data, int w_type, int rows, int cols,
    const float* x, float* y);

#ifdef __cplusplus
}
#endif

#endif // GGML_MATMUL_H
