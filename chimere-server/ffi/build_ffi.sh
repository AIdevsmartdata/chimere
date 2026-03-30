#!/bin/bash
# Build the ggml-ffi crate with IQ3_S GEMV support.
#
# Requires:
#   - libggml.so from ik_llama.cpp build (for IQ3_S AVX2 dot product)
#   - CUDA 12.8 (libggml.so links against cudart)
#   - gcc with OpenMP support (for parallel GEMV)
set -e
cd "$(dirname "$0")"

export GGML_SO_DIR=${GGML_SO_DIR:-${IKLLAMACPP_DIR:-$HOME/ik_llama.cpp}/build_sm120/ggml/src}
export GGML_INCLUDE_DIR=${GGML_INCLUDE_DIR:-${IKLLAMACPP_DIR:-$HOME/ik_llama.cpp}/ggml/include}
export GGML_SRC_DIR=${GGML_SRC_DIR:-${IKLLAMACPP_DIR:-$HOME/ik_llama.cpp}/ggml/src}
export CUDA_LIB_DIR=/usr/local/cuda-12.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$GGML_SO_DIR:$CUDA_LIB_DIR:$LD_LIBRARY_PATH

cargo build 2>&1
echo "=== BUILD OK ==="
echo ""
echo "IQ3_S GEMV FFI enabled. To use:"
echo "  CHIMERE_NCMOE_CPU=1 cargo run ..."
echo ""
echo "To set thread count (default 4):"
echo "  CHIMERE_NCMOE_THREADS=14 CHIMERE_NCMOE_CPU=1 cargo run ..."
