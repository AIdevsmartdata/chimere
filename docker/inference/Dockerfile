# =============================================================================
# Chimere Inference Container
# Runs ik_llama.cpp llama-server with full GPU acceleration.
#
# Build:
#   docker build -t chimere-inference -f docker/inference/Dockerfile .
#   docker build --build-arg GPU_ARCH=89 -t chimere-inference .
#
# Run:
#   docker run --gpus all -v /path/to/models:/models \
#     -e CHIMERE_MODEL=/models/model.gguf \
#     -p 8081:8081 chimere-inference
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build ik_llama.cpp
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ARG GPU_ARCH=120
ARG IK_LLAMA_REPO=https://github.com/ikawrakow/ik_llama.cpp
ARG IK_LLAMA_BRANCH=master

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        cmake \
        ninja-build \
        build-essential \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone --depth 1 --branch ${IK_LLAMA_BRANCH} ${IK_LLAMA_REPO} ik_llama

WORKDIR /build/ik_llama

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH} \
        -DGGML_NATIVE=OFF \
        -DLLAMA_SERVER=ON \
        -DLLAMA_CURL=ON \
    && cmake --build build --target llama-server -j "$(nproc)"

# Collect the binary and its shared library dependencies from the build tree
# so the runtime stage only needs the CUDA runtime base image.
RUN mkdir -p /dist/lib \
    && cp build/bin/llama-server /dist/ \
    && ldd /dist/llama-server \
        | grep -oP '/build/\S+' \
        | xargs -I{} cp {} /dist/lib/ 2>/dev/null || true

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libcurl4 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy server binary and any build-tree shared libs
COPY --from=builder /dist/llama-server /usr/local/bin/llama-server
COPY --from=builder /dist/lib/ /usr/local/lib/
RUN ldconfig

# Entrypoint script
COPY docker/inference/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Create model mount point
RUN mkdir -p /models

# Environment defaults
ENV CHIMERE_MODEL=/models/model.gguf \
    CHIMERE_PORT=8081 \
    CHIMERE_CTX=32768 \
    CHIMERE_NGL=99 \
    CHIMERE_NCMOE=4 \
    CHIMERE_FLASH_ATTN=on \
    CHIMERE_KV_K=q8_0 \
    CHIMERE_KV_V=q4_0 \
    CHIMERE_NP=1

# Expose the default server port
EXPOSE 8081

# Health check: llama-server /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${CHIMERE_PORT}/health || exit 1

ENTRYPOINT ["entrypoint.sh"]
