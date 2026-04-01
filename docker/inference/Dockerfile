# Chimere Server — Build it and test it
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    curl git cmake build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build
RUN git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git /build/backend
WORKDIR /build/backend/build
RUN cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="89;120" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . -j2

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

FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y libgomp1 curl && rm -rf /var/lib/apt/lists/*
RUN useradd -r -s /usr/sbin/nologin chimere

COPY --from=builder /build/chimere-server/target/release/chimere-server /usr/local/bin/
COPY --from=builder /build/backend/build/src/libllama.so /usr/local/lib/
COPY --from=builder /build/backend/build/ggml/src/libggml.so /usr/local/lib/
RUN ldconfig

USER chimere
ENV CHIMERE_PORT=8081 CHIMERE_LLAMA_BACKEND=1 CHIMERE_NCMOE=4 \
    CHIMERE_KV_MAX_SEQ=32768 CHIMERE_BATCH=4096 CHIMERE_UBATCH=512 \
    CHIMERE_THREADS=8 NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8081
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8081/health || exit 1
ENTRYPOINT ["chimere-server"]
