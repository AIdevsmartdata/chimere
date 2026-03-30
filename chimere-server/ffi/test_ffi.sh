#!/bin/bash
# Build and test the ggml-ffi crate
set -e
cd "$(dirname "$0")"

export GGML_STATIC_DIR=${GGML_STATIC_DIR:-${IKLLAMACPP_DIR:-$HOME/ik_llama.cpp}/build_static/ggml/src}
export GGML_INCLUDE_DIR=${GGML_INCLUDE_DIR:-${IKLLAMACPP_DIR:-$HOME/ik_llama.cpp}/ggml/include}
export CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

echo "=== Building ==="
cargo build 2>&1

echo ""
echo "=== Running tests ==="
cargo test -- --nocapture 2>&1

echo ""
echo "=== Running binary ==="
cargo run --bin test_ggml 2>&1

echo ""
echo "=== ALL DONE ==="
