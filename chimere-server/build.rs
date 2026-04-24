// build.rs — Compile CUDA kernels to cubin at build time via nvcc.
//
// This eliminates NVRTC runtime compilation overhead:
//   - No NVRTC compilation on first kernel use (~200ms total saved at startup)
//   - Pre-compiled cubin enables cached CudaFunction handles (3-5us vs 13.3us per launch)
//
// The compiled cubin is embedded into the binary via include_bytes! at compile time.
// Set CHIMERE_NVRTC=1 to force the old NVRTC path (for debugging).

use std::process::Command;

fn main() {
    // Only recompile if .cu file changed
    println!("cargo:rerun-if-changed=kernels/chimere_kernels.cu");
    println!("cargo:rerun-if-env-changed=CHIMERE_NVRTC");

    // Register custom cfg so check-cfg doesn't warn
    println!("cargo:rustc-check-cfg=cfg(chimere_has_cubin)");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cubin_path = format!("{}/chimere_kernels.cubin", out_dir);

    // Find nvcc — prefer /usr/local/cuda-12.8 (CUDA 12.8 toolkit on this system)
    let nvcc = if std::path::Path::new("/usr/local/cuda-12.8/bin/nvcc").exists() {
        "/usr/local/cuda-12.8/bin/nvcc".to_string()
    } else if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        format!("{}/bin/nvcc", cuda_home)
    } else {
        "nvcc".to_string()
    };

    eprintln!("chimere-deltanet build.rs: compiling kernels with {}", nvcc);

    let status = Command::new(&nvcc)
        .args(&[
            "--gpu-architecture=sm_120",
            "-cubin",
            "-O3",
            // Suppress benign warnings about shared memory redeclarations
            "-diag-suppress=20042",
            "-o", &cubin_path,
            "kernels/chimere_kernels.cu",
        ])
        .status();

    match status {
        Ok(s) if s.success() => {
            eprintln!(
                "chimere-deltanet build.rs: cubin compiled successfully -> {}",
                cubin_path
            );
            // Tell Rust where to find the cubin at compile time
            println!("cargo:rustc-env=CHIMERE_CUBIN_PATH={}", cubin_path);
            // Set cfg flag so cubin_loader knows the cubin is real
            println!("cargo:rustc-cfg=chimere_has_cubin");
        }
        Ok(s) => {
            // nvcc failed — create empty placeholder so include_bytes! doesn't break
            eprintln!(
                "chimere-deltanet build.rs: WARNING: nvcc exited with {:?}, \
                 falling back to NVRTC at runtime",
                s.code()
            );
            std::fs::write(&cubin_path, b"").ok();
            println!("cargo:rustc-env=CHIMERE_CUBIN_PATH={}", cubin_path);
        }
        Err(e) => {
            // nvcc not found — create empty placeholder so include_bytes! doesn't break
            eprintln!(
                "chimere-deltanet build.rs: WARNING: nvcc not found ({}), \
                 falling back to NVRTC at runtime",
                e
            );
            std::fs::write(&cubin_path, b"").ok();
            println!("cargo:rustc-env=CHIMERE_CUBIN_PATH={}", cubin_path);
        }
    }

    // --- ik_llama.cpp path resolution (community-friendly) ---
    // Precedence:
    //   1. explicit env override (IKLLAMACPP_DIR, LLAMA_LIB_DIR, GGML_LIB_DIR)
    //   2. $HOME/ik_llama.cpp (default install location — matches install-chimere.sh)
    //
    // Build subdir is `build_sm120` on Blackwell and `build_sm89` on Ada;
    // users override via IK_LLAMA_BUILD_SUBDIR when they built against a
    // different SM target.
    println!("cargo:rerun-if-env-changed=IKLLAMACPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=GGML_LIB_DIR");
    println!("cargo:rerun-if-env-changed=IK_LLAMA_BUILD_SUBDIR");

    let iklama_dir = std::env::var("IKLLAMACPP_DIR")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
            format!("{}/ik_llama.cpp", home)
        });
    let build_subdir = std::env::var("IK_LLAMA_BUILD_SUBDIR")
        .unwrap_or_else(|_| "build_sm120".into());
    let llama_lib_dir = std::env::var("LLAMA_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/{}/src", iklama_dir, build_subdir));
    let ggml_lib_dir = std::env::var("GGML_LIB_DIR")
        .unwrap_or_else(|_| format!("{}/{}/ggml/src", iklama_dir, build_subdir));

    // Propagate ggml_cuda_gemv feature for src/kernels/ggml_gpu.rs
    let has_ggml_so = std::path::Path::new(&format!("{}/libggml.so", ggml_lib_dir)).exists();
    if has_ggml_so {
        println!("cargo:rustc-cfg=feature=\"ggml_cuda_gemv\"");
    }
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"ggml_cuda_gemv\"))");

    // --- libllama FFI linkage (for CHIMERE_LLAMA_BACKEND=1) ---
    // Link against ik_llama's libllama.so and libggml.so for the full forward pass backend.
    // These are always linked (the LlamaForward struct is only instantiated at runtime
    // when CHIMERE_LLAMA_BACKEND=1, so the dynamic linker resolves symbols lazily).
    let has_libllama = std::path::Path::new(&format!("{}/libllama.so", llama_lib_dir)).exists();
    let has_libggml = std::path::Path::new(&format!("{}/libggml.so", ggml_lib_dir)).exists();

    if has_libllama && has_libggml {
        println!("cargo:rustc-link-search=native={}", llama_lib_dir);
        println!("cargo:rustc-link-search=native={}", ggml_lib_dir);
        println!("cargo:rustc-link-lib=dylib=llama");
        println!("cargo:rustc-link-lib=dylib=ggml");
        // Set rpath so the binary finds libllama.so at runtime without LD_LIBRARY_PATH
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", llama_lib_dir);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ggml_lib_dir);
        println!("cargo:rustc-cfg=has_libllama");
        eprintln!(
            "chimere-deltanet build.rs: libllama.so + libggml.so found at {} / {}, llama_backend enabled",
            llama_lib_dir, ggml_lib_dir,
        );
    } else {
        eprintln!(
            "chimere-deltanet build.rs: WARNING: libllama.so ({}) or libggml.so ({}) not found. \
             Set IKLLAMACPP_DIR or run install-chimere.sh first. \
             llama_backend will fail at runtime if CHIMERE_LLAMA_BACKEND=1",
            llama_lib_dir, ggml_lib_dir,
        );
    }
    println!("cargo:rustc-check-cfg=cfg(has_libllama)");
}
