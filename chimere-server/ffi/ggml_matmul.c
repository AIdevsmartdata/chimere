// ggml_matmul.c -- Thin C wrapper around ggml for FFI from Rust.
//
// Provides a simple "mul_mat_vec" operation that:
//   1. Creates a ggml context + CUDA backend
//   2. Builds a tiny compute graph: result = W @ x  (quantized W, f32 x)
//   3. Runs it on the CUDA backend
//   4. Copies the result back to host
//
// This avoids exposing ggml's full API surface to Rust while still getting
// the GPU-accelerated quantized GEMV kernels.

#include "ggml_matmul.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

struct ggml_ffi_ctx {
    ggml_backend_t backend;
    int device_id;
};

ggml_ffi_ctx_t* ggml_ffi_init_cuda(int device_id) {
    ggml_ffi_ctx_t* ctx = (ggml_ffi_ctx_t*)calloc(1, sizeof(ggml_ffi_ctx_t));
    if (!ctx) return NULL;

    ctx->backend = ggml_backend_cuda_init(device_id, NULL);
    if (!ctx->backend) {
        fprintf(stderr, "ggml_ffi: failed to init CUDA backend for device %d\n", device_id);
        free(ctx);
        return NULL;
    }
    ctx->device_id = device_id;
    return ctx;
}

void ggml_ffi_free(ggml_ffi_ctx_t* ctx) {
    if (!ctx) return;
    if (ctx->backend) {
        ggml_backend_free(ctx->backend);
    }
    free(ctx);
}

void ggml_ffi_mul_mat_vec(
    ggml_ffi_ctx_t* ctx,
    const void* w_data, int w_type, int rows, int cols,
    const float* x, float* y)
{
    if (!ctx || !ctx->backend) return;

    // We need enough memory for the tensor metadata (not the data itself,
    // which will be allocated on the backend).
    // Each tensor needs ~250 bytes of metadata. We have 3 tensors + graph overhead.
    size_t ctx_size = ggml_tensor_overhead() * 3 + ggml_graph_overhead();

    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = true,  // we'll allocate on the CUDA backend
    };

    struct ggml_context* gctx = ggml_init(params);
    if (!gctx) {
        fprintf(stderr, "ggml_ffi: failed to init ggml context\n");
        return;
    }

    // Create weight tensor W: [rows, cols] in quantized format
    // ggml_new_tensor_2d(ctx, type, ne0=cols, ne1=rows)
    // In ggml, ne[0] is the innermost dimension (columns).
    struct ggml_tensor* W = ggml_new_tensor_2d(gctx, (enum ggml_type)w_type, cols, rows);
    ggml_set_name(W, "W");

    // Create input tensor x: [1, cols] as f32
    struct ggml_tensor* X = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, cols, 1);
    ggml_set_name(X, "X");

    // Compute result = W @ X^T  => shape [1, rows]
    // ggml_mul_mat(ctx, a, b):
    //   a = [ne03, ne02, n, k]    k columns, n rows
    //   b = [ne03*x, ne02*y, m, k]   k columns, m rows
    //   result = [ne03*x, ne02*y, m, n]   n columns, m rows
    // So: a=W [rows, cols], b=X [1, cols] => result [1, rows] ✓
    struct ggml_tensor* result = ggml_mul_mat(gctx, W, X);
    ggml_set_name(result, "result");

    // Build compute graph
    struct ggml_cgraph* graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, result);

    // Allocate tensors on CUDA backend
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gctx, ctx->backend);
    if (!buf) {
        fprintf(stderr, "ggml_ffi: failed to allocate tensors on CUDA backend\n");
        ggml_free(gctx);
        return;
    }

    // Upload weight data to GPU
    size_t w_size = ggml_row_size((enum ggml_type)w_type, cols) * rows;
    ggml_backend_tensor_set(W, w_data, 0, w_size);

    // Upload input data to GPU
    ggml_backend_tensor_set(X, x, 0, cols * sizeof(float));

    // Run the computation
    enum ggml_status status = ggml_backend_graph_compute(ctx->backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_ffi: graph compute failed with status %d\n", status);
    }

    // Download result from GPU
    ggml_backend_tensor_get(result, y, 0, rows * sizeof(float));

    // Cleanup
    ggml_backend_buffer_free(buf);
    ggml_free(gctx);
}
