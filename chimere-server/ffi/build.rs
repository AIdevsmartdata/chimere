// build.rs -- Compile ggml FFI wrappers:
//
// 1. ggml_iq3s_gemv.c  -- CPU AVX2 IQ3_S dot product for ncmoe CPU GEMV
// 2. ggml_cuda_gemv.cu -- GPU MMVQ wrapper calling ggml's optimized CUDA kernels
//                         (IQ3_S, Q5_K, Q8_0, Q4_K, Q6_K)
//
// Both link against libggml.so from ik_llama.cpp.

use std::env;
use std::process::Command;

fn main() {
    let ggml_so_dir = env::var("GGML_SO_DIR")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/build_sm120/ggml/src".to_string());
    let ggml_include_dir = env::var("GGML_INCLUDE_DIR")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/ggml/include".to_string());
    let ggml_src_dir = env::var("GGML_SRC_DIR")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/ggml/src".to_string());

    // Check if the shared library exists
    let so_path = format!("{}/libggml.so", ggml_so_dir);
    if !std::path::Path::new(&so_path).exists() {
        println!("cargo:warning=libggml.so not found at {}, skipping ggml FFI", so_path);
        return;
    }

    // -----------------------------------------------------------------------
    // Part 1: Compile ggml_iq3s_gemv.c (CPU AVX2 path)
    // -----------------------------------------------------------------------
    cc::Build::new()
        .file("ggml_iq3s_gemv.c")
        .include(&ggml_include_dir)
        .include(&ggml_src_dir)
        .flag("-mavx2")
        .flag("-mfma")
        .flag("-mf16c")
        .flag("-O3")
        .flag("-fopenmp")
        .warnings(false)
        .compile("ggml_iq3s_gemv");

    // -----------------------------------------------------------------------
    // Part 1c: Compile chimere_sampler.cpp (C++ sampler wrapper — avoids 993KB logits copy)
    // -----------------------------------------------------------------------
    let common_dir = env::var("GGML_COMMON_DIR")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/common".to_string());

    let ik_llama_include = env::var("IK_LLAMA_INCLUDE")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/include".to_string());

    let ik_llama_src = env::var("IK_LLAMA_SRC")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/src".to_string());

    cc::Build::new()
        .cpp(true)
        .file("chimere_sampler.cpp")
        .include(&ggml_include_dir)
        .include(&ik_llama_include)
        .include(&common_dir)
        .include(&ik_llama_src)  // for llama-grammar.h etc.
        .flag("-std=c++17")
        .flag("-O3")
        .warnings(false)
        .compile("chimere_sampler");

    println!("cargo:rerun-if-changed=chimere_sampler.cpp");

    // Link libcommon.a from ik_llama for common_sampler_*
    let common_lib_dir = env::var("GGML_COMMON_LIB_DIR")
        .unwrap_or_else(|_| "{IKLLAMACPP_DIR}/build_sm120/common".to_string());
    println!("cargo:rustc-link-search=native={}", common_lib_dir);
    println!("cargo:rustc-link-lib=static=common");

    // -----------------------------------------------------------------------
    // Part 2: Compile ggml_cuda_gemv.cu (GPU MMVQ wrapper)
    // -----------------------------------------------------------------------
    let out_dir = env::var("OUT_DIR").unwrap();
    let cuda_obj = format!("{}/ggml_cuda_gemv.o", out_dir);
    let cuda_lib = format!("{}/libggml_cuda_gemv.a", out_dir);

    // Find nvcc
    let nvcc = if std::path::Path::new("/usr/local/cuda-12.8/bin/nvcc").exists() {
        "/usr/local/cuda-12.8/bin/nvcc".to_string()
    } else if let Ok(cuda_home) = env::var("CUDA_HOME") {
        format!("{}/bin/nvcc", cuda_home)
    } else {
        "nvcc".to_string()
    };

    let ggml_cuda_dir = format!("{}/ggml-cuda", ggml_src_dir);

    // Compile .cu -> .o
    let nvcc_status = Command::new(&nvcc)
        .args(&[
            "-c",
            "-arch=sm_120",
            "-O3",
            "-std=c++17",
            "--compiler-options", "-fPIC",
            // Suppress benign warnings
            "-diag-suppress=20012",  // __host__ __device__ annotation
            "-diag-suppress=20042",  // shared memory redeclarations
            &format!("-I{}", ggml_include_dir),
            &format!("-I{}", ggml_src_dir),
            &format!("-I{}", ggml_cuda_dir),
            "-o", &cuda_obj,
            "ggml_cuda_gemv.cu",
        ])
        .status();

    match nvcc_status {
        Ok(s) if s.success() => {
            eprintln!("ggml-ffi build.rs: ggml_cuda_gemv.cu compiled successfully");

            // Create static library from .o
            let ar_status = Command::new("ar")
                .args(&["rcs", &cuda_lib, &cuda_obj])
                .status();

            match ar_status {
                Ok(s) if s.success() => {
                    println!("cargo:rustc-link-search=native={}", out_dir);
                    println!("cargo:rustc-link-lib=static=ggml_cuda_gemv");
                    // Need CUDA runtime for the wrapper
                    let cuda_lib_dir = env::var("CUDA_LIB_DIR")
                        .unwrap_or_else(|_| "/usr/local/cuda-12.8/targets/x86_64-linux/lib".to_string());
                    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
                    println!("cargo:rustc-link-lib=dylib=cudart");
                    // Need libstdc++ for C++ runtime
                    println!("cargo:rustc-link-lib=dylib=stdc++");
                    // Set cfg flag for Rust conditional compilation
                    println!("cargo:rustc-cfg=feature=\"ggml_cuda_gemv\"");
                }
                _ => {
                    eprintln!("ggml-ffi build.rs: WARNING: ar failed, skipping CUDA GEMV wrapper");
                }
            }
        }
        Ok(s) => {
            eprintln!(
                "ggml-ffi build.rs: WARNING: nvcc exited with {:?}, skipping CUDA GEMV wrapper",
                s.code()
            );
        }
        Err(e) => {
            eprintln!(
                "ggml-ffi build.rs: WARNING: nvcc not found ({}), skipping CUDA GEMV wrapper",
                e
            );
        }
    }

    // -----------------------------------------------------------------------
    // Common linking
    // -----------------------------------------------------------------------

    // Link against libggml.so + libllama.so from ik_llama
    let llama_so_dir = env::var("LLAMA_SO_DIR")
        .unwrap_or_else(|_| format!("{}/../src", ggml_so_dir));
    println!("cargo:rustc-link-search=native={}", ggml_so_dir);
    println!("cargo:rustc-link-search=native={}", llama_so_dir);
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", llama_so_dir);

    // libggml.so depends on CUDA runtime
    let cuda_lib_dir = env::var("CUDA_LIB_DIR")
        .unwrap_or_else(|_| "/usr/local/cuda-12.8/targets/x86_64-linux/lib".to_string());
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);

    // Link OpenMP for parallel GEMV (CPU path)
    println!("cargo:rustc-link-lib=dylib=gomp");

    // Set rpath so the binary can find libggml.so at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ggml_so_dir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib_dir);

    // Tell cargo to re-run if sources change
    println!("cargo:rerun-if-changed=ggml_iq3s_gemv.c");
    println!("cargo:rerun-if-changed=ggml_iq3s_gemv.h");
    println!("cargo:rerun-if-changed=ggml_cuda_gemv.cu");

    // Set cfg flag for CPU IQ3_S path
    println!("cargo:rustc-cfg=feature=\"ggml_iq3s\"");

    // Register custom check-cfg to avoid warnings
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"ggml_iq3s\", \"ggml_cuda_gemv\"))");
}
