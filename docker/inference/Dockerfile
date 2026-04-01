# ═══════════════════════════════════════════════════════════════
# Chimere Server — Build it and test it
# ═══════════════════════════════════════════════════════════════
# Rust-native MoE inference engine with Engram memory,
# custom sampling, and OpenAI-compatible API.
#
# Usage:
#   docker build -f docker/inference/Dockerfile -t chimere .
#   docker run --gpus all -v /path/to/model.gguf:/model.gguf \
#     -e CHIMERE_MODEL=/model.gguf \
#     -p 8081:8081 chimere
# ═══════════════════════════════════════════════════════════════

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    curl git cmake build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build optimized inference backend
WORKDIR /build
RUN git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git /build/backend
WORKDIR /build/backend
RUN mkdir build && cd build && cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="89;120" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . -j$(nproc) --target llama ggml

# Build Chimere server
COPY chimere-server /build/chimere-server
WORKDIR /build/chimere-server

ENV GGML_SO_DIR=/build/backend/build/ggml/src
ENV GGML_INCLUDE_DIR=/build/backend/ggml/include
ENV GGML_SRC_DIR=/build/backend/ggml/src
ENV GGML_COMMON_DIR=/build/backend/common
ENV GGML_COMMON_LIB_DIR=/build/backend/build/common
ENV IK_LLAMA_INCLUDE=/build/backend/include
ENV IK_LLAMA_SRC=/build/backend/src
ENV LLAMA_SO_DIR=/build/backend/build/src
ENV CUDA_HOME=/usr/local/cuda

RUN cargo build --release --features server

# ═══════════════════════════════════════════════════════════════
# Runtime — minimal image with just the binary + GPU libs
# ═══════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*
RUN useradd -r -s /usr/sbin/nologin chimere

COPY --from=builder /build/chimere-server/target/release/chimere-server /usr/local/bin/
COPY --from=builder /build/backend/build/src/libllama.so /usr/local/lib/
COPY --from=builder /build/backend/build/ggml/src/libggml.so /usr/local/lib/
RUN ldconfig

USER chimere

# Defaults — override with -e at runtime
ENV CHIMERE_PORT=8081
ENV CHIMERE_LLAMA_BACKEND=1
ENV CHIMERE_NCMOE=4
ENV CHIMERE_KV_MAX_SEQ=32768
ENV CHIMERE_KV_TYPE_K=8
ENV CHIMERE_KV_TYPE_V=2
ENV CHIMERE_FLASH_ATTN=1
ENV CHIMERE_BATCH=4096
ENV CHIMERE_UBATCH=512
ENV CHIMERE_THREADS=8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8081/health || exit 1

ENTRYPOINT ["chimere-server"]
