cmake_minimum_required(VERSION 3.14)
project(chimere-extract LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to llama.cpp source and build
set(LLAMA_DIR "$ENV{HOME}/llama.cpp" CACHE PATH "Path to llama.cpp source")
set(LLAMA_BUILD_DIR "${LLAMA_DIR}/build" CACHE PATH "Path to llama.cpp build")

# Include dirs
include_directories(
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/ggml/include
)

add_executable(extract_hidden_states extract_hidden_states.cpp)
add_executable(target_daemon target_daemon.cpp)
add_executable(extract_single_position extract_single_position.cpp)
add_executable(extract_batched extract_batched.cpp)

# All static libraries — order matters (dependents first, providers last)
target_link_libraries(extract_hidden_states PRIVATE
    # Application-level static libs
    ${LLAMA_BUILD_DIR}/common/libcommon.a
    ${LLAMA_BUILD_DIR}/vendor/cpp-httplib/libcpp-httplib.a
    ${LLAMA_BUILD_DIR}/src/libllama.a
    # ggml static libs
    ${LLAMA_BUILD_DIR}/ggml/src/libggml.a
    ${LLAMA_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-base.a
    # System deps
    ssl
    crypto
    curl
    gomp
    pthread
    dl
    m
    # CUDA runtime (needed by ggml-cuda)
    cuda
    cublas
    culibos
    cudart
    cublasLt
)

target_link_libraries(target_daemon PRIVATE
    # Application-level static libs
    ${LLAMA_BUILD_DIR}/common/libcommon.a
    ${LLAMA_BUILD_DIR}/vendor/cpp-httplib/libcpp-httplib.a
    ${LLAMA_BUILD_DIR}/src/libllama.a
    # ggml static libs
    ${LLAMA_BUILD_DIR}/ggml/src/libggml.a
    ${LLAMA_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-base.a
    # System deps
    ssl
    crypto
    curl
    gomp
    pthread
    dl
    m
    # CUDA runtime (needed by ggml-cuda)
    cuda
    cublas
    culibos
    cudart
    cublasLt
)

target_link_libraries(extract_single_position PRIVATE
    # Application-level static libs
    ${LLAMA_BUILD_DIR}/common/libcommon.a
    ${LLAMA_BUILD_DIR}/vendor/cpp-httplib/libcpp-httplib.a
    ${LLAMA_BUILD_DIR}/src/libllama.a
    # ggml static libs
    ${LLAMA_BUILD_DIR}/ggml/src/libggml.a
    ${LLAMA_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-base.a
    # System deps
    ssl
    crypto
    curl
    gomp
    pthread
    dl
    m
    # CUDA runtime (needed by ggml-cuda)
    cuda
    cublas
    culibos
    cudart
    cublasLt
)

target_link_libraries(extract_batched PRIVATE
    # Application-level static libs
    ${LLAMA_BUILD_DIR}/common/libcommon.a
    ${LLAMA_BUILD_DIR}/vendor/cpp-httplib/libcpp-httplib.a
    ${LLAMA_BUILD_DIR}/src/libllama.a
    # ggml static libs
    ${LLAMA_BUILD_DIR}/ggml/src/libggml.a
    ${LLAMA_BUILD_DIR}/ggml/src/ggml-cuda/libggml-cuda.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-cpu.a
    ${LLAMA_BUILD_DIR}/ggml/src/libggml-base.a
    # System deps
    ssl
    crypto
    curl
    gomp
    pthread
    dl
    m
    # CUDA runtime (needed by ggml-cuda)
    cuda
    cublas
    culibos
    cudart
    cublasLt
)

# Install
install(TARGETS extract_hidden_states target_daemon extract_single_position extract_batched RUNTIME DESTINATION bin)
