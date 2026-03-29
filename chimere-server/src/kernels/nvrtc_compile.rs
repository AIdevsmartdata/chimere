//! NVRTC compilation helpers for chimere CUDA kernels.
//!
//! Provides:
//! - `compile_and_cache`: compile CUDA source to PTX via NVRTC (sm_120 target)
//! - `get_or_load_func`: load a kernel function from the pre-compiled cubin, or
//!   fall back to NVRTC. This replaces direct calls to candle's
//!   `get_or_load_custom_func` and provides ~2x lower launch overhead when the
//!   cubin is available (bypasses HashMap lookup + RwLock in candle's module cache).
//!
//! ## Cubin loading
//!
//! When `cubin_loader::has_cubin()` is true, `get_or_load_func` loads the
//! pre-compiled cubin via cudarc's `CudaContext::load_module(Ptx::from_binary(...))`
//! and caches the resulting `CudaModule` + `CudaFunction` handles in a global
//! `OnceLock`. Subsequent calls resolve to an `Arc` clone + `HashMap::get` — no
//! CUDA driver calls, no candle RwLock contention.
//!
//! When the cubin is not available (or the requested function is not in the cubin),
//! falls back transparently to candle's `get_or_load_custom_func` which does
//! NVRTC compilation on first use.

use candle_core::cuda_backend::cudarc::driver::{CudaFunction, CudaStream};
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Compile CUDA source to PTX via NVRTC, caching the result in the provided `OnceLock`.
///
/// Uses sm_120 as primary target. If that fails (e.g. older NVRTC without sm_120 support),
/// falls back to default arch (auto-detect). Panics if both fail.
///
/// Returns a `&'static str` reference to the cached PTX string.
pub fn compile_and_cache(source: &str, cache: &'static OnceLock<String>) -> &'static str {
    cache.get_or_init(|| {
        // maxrregcount limits registers per thread to improve occupancy.
        // 128 regs → 4 blocks/SM = 50% occupancy (vs 236 regs → 2 blocks = 12.5%)
        // Toggle: CHIMERE_MAXREG=N (default 128, set to 0 to disable)
        let maxreg: Option<usize> = {
            use once_cell::sync::Lazy;
            static MR: Lazy<Option<usize>> = Lazy::new(|| {
                match std::env::var("CHIMERE_MAXREG") {
                    Ok(v) if v == "0" => None,  // disable
                    Ok(v) => v.parse().ok(),
                    Err(_) => Some(128),  // default: 128 regs
                }
            });
            *MR
        };
        let opts_sm120 = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
            arch: Some("sm_120"),
            maxrregcount: maxreg,
            ..Default::default()
        };
        let result = candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts(
            source, opts_sm120,
        );
        match result {
            Ok(ptx) => ptx.to_src(),
            Err(e) => {
                // Fallback: compile without explicit arch (let NVRTC auto-detect)
                candle_core::cuda_backend::cudarc::nvrtc::compile_ptx_with_opts(
                    source,
                    candle_core::cuda_backend::cudarc::nvrtc::CompileOptions::default(),
                )
                .map(|p| p.to_src())
                .unwrap_or_else(|e2| {
                    panic!(
                        "NVRTC compile failed:\n  primary (sm_120): {e}\n  fallback (default): {e2}"
                    )
                })
            }
        }
    });
    cache.get().expect("OnceLock initialised above")
}

// ---------------------------------------------------------------------------
// Cubin module cache
// ---------------------------------------------------------------------------

/// Cached cubin module + per-function lookup.
///
/// The cubin is loaded once via `CudaContext::load_module`. Individual kernel
/// functions are resolved lazily via `CudaModule::load_function` and cached in
/// a HashMap keyed by function name.
struct CubinCache {
    module: Arc<candle_core::cuda_backend::cudarc::driver::CudaModule>,
    funcs: HashMap<String, CudaFunction>,
}

/// Global cubin cache. Initialized once on first `get_or_load_func` call when
/// `has_cubin()` is true. `None` means cubin load was attempted but failed.
static CUBIN_CACHE: OnceLock<Mutex<Option<CubinCache>>> = OnceLock::new();

/// Load the cubin module into the CUDA context, or return the cached version.
///
/// Returns `None` if cubin is not available or loading failed.
fn get_cubin_cache(
    stream: &Arc<CudaStream>,
) -> &'static Mutex<Option<CubinCache>> {
    CUBIN_CACHE.get_or_init(|| {
        if !super::cubin_loader::has_cubin() {
            eprintln!("[cubin] No cubin available, using NVRTC fallback");
            return Mutex::new(None);
        }

        let bytes = super::cubin_loader::cubin_bytes();
        let ctx = stream.context();
        let ptx = candle_core::cuda_backend::cudarc::nvrtc::Ptx::from_binary(bytes.to_vec());

        match ctx.load_module(ptx) {
            Ok(module) => {
                eprintln!(
                    "[cubin] Loaded pre-compiled cubin ({} bytes) into CUDA context",
                    bytes.len()
                );
                Mutex::new(Some(CubinCache {
                    module,
                    funcs: HashMap::new(),
                }))
            }
            Err(e) => {
                eprintln!("[cubin] Failed to load cubin: {e:?}, falling back to NVRTC");
                Mutex::new(None)
            }
        }
    })
}

/// Try to get a kernel function from the cubin cache.
///
/// Returns `Some((CudaFunction, Arc<CudaStream>))` if the function was found
/// in the cubin, `None` otherwise.
fn try_cubin_func(
    stream: &Arc<CudaStream>,
    fn_name: &str,
) -> Option<CudaFunction> {
    let cache_mutex = get_cubin_cache(stream);
    let mut guard = cache_mutex.lock().ok()?;
    let cache = guard.as_mut()?;

    // Check if we already resolved this function
    if let Some(func) = cache.funcs.get(fn_name) {
        return Some(func.clone());
    }

    // Try to load from the module
    match cache.module.load_function(fn_name) {
        Ok(func) => {
            cache.funcs.insert(fn_name.to_string(), func.clone());
            Some(func)
        }
        Err(_) => None,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load a kernel function, preferring the pre-compiled cubin over NVRTC.
///
/// Returns `(CudaFunction, Arc<CudaStream>)`. The caller uses
/// `stream.launch_builder(&func)` to build and launch the kernel.
///
/// ## Cubin path (fast, ~2us)
///
/// When `cubin_loader::has_cubin()` is true and the requested function exists
/// in the cubin, loads from the cached cubin module. First call per function
/// does `CudaModule::load_function` (~5us); subsequent calls are a HashMap
/// lookup + Mutex lock (~0.1us).
///
/// ## NVRTC fallback (slower first call, ~4us cached)
///
/// When the cubin is not available (or the function is not in the cubin), falls
/// back to candle's `get_or_load_custom_func` which compiles via NVRTC on first
/// use and caches the result.
///
/// ## Usage
///
/// ```ignore
/// let (func, stream) = get_or_load_func(
///     dev, "gemv_iq3s_q8", "chimere_iq3s_gemv_v14",
///     IQ3S_KERNEL_SRC, &PTX_CACHE,
/// )?;
/// let mut builder = stream.launch_builder(&func);
/// builder.arg(&weights);
/// // ...
/// unsafe { builder.launch(cfg) }?;
/// ```
pub fn get_or_load_func(
    dev: &CudaDevice,
    fn_name: &str,
    module_name: &str,
    kernel_src: &str,
    ptx_cache: &'static OnceLock<String>,
) -> Result<(CudaFunction, Arc<CudaStream>)> {
    let stream = dev.cuda_stream();

    // Try the cubin path first
    if let Some(func) = try_cubin_func(&stream, fn_name) {
        return Ok((func, stream));
    }

    // NVRTC fallback: compile PTX and load via candle's cache
    let ptx = compile_and_cache(kernel_src, ptx_cache);
    let cuda_func = dev.get_or_load_custom_func(fn_name, module_name, ptx)?;
    // Extract the CudaFunction from candle's CudaFunc wrapper
    let func = cuda_func.into_cuda_function();
    Ok((func, stream))
}
