FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    curl git cmake build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Build ik_llama.cpp (FFI backend)
WORKDIR /build
RUN git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git
WORKDIR /build/ik_llama.cpp
RUN mkdir build && cd build && cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="89;120" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . -j$(nproc) --target llama ggml

# Copy chimere source
COPY chimere-server /build/chimere-server
WORKDIR /build/chimere-server

# Build chimere-server with ik_llama FFI
ENV GGML_SO_DIR=/build/ik_llama.cpp/build/ggml/src
ENV GGML_INCLUDE_DIR=/build/ik_llama.cpp/ggml/include
ENV GGML_SRC_DIR=/build/ik_llama.cpp/ggml/src
ENV GGML_COMMON_DIR=/build/ik_llama.cpp/common
ENV GGML_COMMON_LIB_DIR=/build/ik_llama.cpp/build/common
ENV IK_LLAMA_INCLUDE=/build/ik_llama.cpp/include
ENV IK_LLAMA_SRC=/build/ik_llama.cpp/src
ENV LLAMA_SO_DIR=/build/ik_llama.cpp/build/src
ENV CUDA_HOME=/usr/local/cuda

RUN cargo build --release --features server

# --- Runtime stage ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/chimere-server/target/release/chimere-server /usr/local/bin/
COPY --from=builder /build/ik_llama.cpp/build/src/libllama.so /usr/local/lib/
COPY --from=builder /build/ik_llama.cpp/build/ggml/src/libggml.so /usr/local/lib/
RUN ldconfig

ENV CHIMERE_PORT=8081
ENV CHIMERE_LLAMA_BACKEND=1
ENV CHIMERE_NCMOE=4
ENV CHIMERE_KV_MAX_SEQ=32768
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8081/health || exit 1

ENTRYPOINT ["chimere-server"]
