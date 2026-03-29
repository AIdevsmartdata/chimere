#!/bin/bash
# Build the hidden states extractor against existing llama.cpp build
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$HOME/llama.cpp"
LLAMA_BUILD="$LLAMA_DIR/build"

if [ ! -d "$LLAMA_BUILD" ]; then
    echo "ERROR: llama.cpp build not found at $LLAMA_BUILD"
    echo "Build llama.cpp first: cd ~/llama.cpp && cmake -B build && cmake --build build"
    exit 1
fi

echo "Building extract_hidden_states + target_daemon..."
echo "  llama.cpp: $LLAMA_DIR"
echo "  build dir: $LLAMA_BUILD"

mkdir -p "$SCRIPT_DIR/build"
cd "$SCRIPT_DIR/build"

cmake .. \
    -DLLAMA_DIR="$LLAMA_DIR" \
    -DLLAMA_BUILD_DIR="$LLAMA_BUILD" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j$(nproc)

echo ""
echo "Build OK: $SCRIPT_DIR/build/extract_hidden_states"
echo "Build OK: $SCRIPT_DIR/build/target_daemon"
echo ""
echo "Usage example:"
echo "  ./build/extract_hidden_states \\"
echo "    -m ~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-MXFP4_MOE.gguf \\"
echo "    -ngl 99 -ot '.ffn_.*_exps.=CPU' \\"
echo "    --layers 2,11,20,29,37 \\"
echo "    --input ../data/training_samples.jsonl \\"
echo "    --output ../data/features/"
