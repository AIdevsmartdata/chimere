//! # Debug and profiling utilities
//!
//! Environment-variable-gated helpers for tensor inspection and performance
//! profiling. All functions are no-ops unless the relevant env-var is set.
//!
//! | Env var               | Purpose                                          |
//! |-----------------------|--------------------------------------------------|
//! | `CHIMERE_VRAM_LOG=1`  | Log GPU VRAM usage at each `log_vram()` call     |
//! | `CHIMERE_DEBUG=1`     | Enable tensor dump via `debug_dump()`            |
//! | `CHIMERE_GDN_PROFILE` | Per-op GDN profiling via `gdn_profile_enabled()` |
//! | `CHIMERE_DISPATCH_PROF` | Dispatch-level profiling                       |
//! | `CHIMERE_ACT_DTYPE=f16` | Store inter-layer hiddens as F16               |
//! | `CHIMERE_L0_DUMP=1`   | Step-by-step layer-0 tensor dump                |

use candle_core::Tensor;

/// Log GPU VRAM usage. Toggle: `CHIMERE_VRAM_LOG=1`
#[allow(dead_code)]
pub(crate) fn log_vram(label: &str) {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_VRAM_LOG").is_ok());
    if !*ENABLED {
        return;
    }
    if let Ok((free, total)) =
        candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
    {
        let used_mb = (total - free) as f64 / (1024.0 * 1024.0);
        let free_mb = free as f64 / (1024.0 * 1024.0);
        eprintln!(
            "[VRAM] {}: {:.0} MB used, {:.0} MB free",
            label, used_mb, free_mb
        );
    }
}

/// Dump first N values + L2 norm of a tensor for debugging.
///
/// Activates when `CHIMERE_DEBUG` is set in the environment.
pub(crate) fn debug_dump(label: &str, t: &Tensor) {
    // Upcast to F32 if needed (e.g. F16 activations)
    let t_f32 = if t.dtype() != candle_core::DType::F32 {
        match t.to_dtype(candle_core::DType::F32) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[DEBUG] {}: dtype cast failed: {}", label, e);
                return;
            }
        }
    } else {
        t.clone()
    };
    let flat = match t_f32.flatten_all() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("[DEBUG] {}: flatten failed: {}", label, e);
            return;
        }
    };
    let data: Vec<f32> = match flat.to_vec1() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[DEBUG] {}: to_vec1 failed: {}", label, e);
            return;
        }
    };
    let n = data.len().min(10);
    let l2: f64 = data
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "[DEBUG] {} shape={:?} L2={:.6} first{}={:?}",
        label,
        t.dims(),
        l2,
        n,
        &data[..n]
    );
}

/// Check if debug mode is enabled (`CHIMERE_DEBUG` env var).
pub(crate) fn debug_enabled() -> bool {
    use once_cell::sync::Lazy;
    static DEBUG: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_DEBUG").is_ok());
    *DEBUG
}


/// Check if dispatch-level profiling is enabled (`CHIMERE_DISPATCH_PROF` env var).
pub(crate) fn dispatch_prof_enabled() -> bool {
    use once_cell::sync::Lazy;
    static DISPATCH_PROF: Lazy<bool> =
        Lazy::new(|| std::env::var("CHIMERE_DISPATCH_PROF").is_ok());
    *DISPATCH_PROF
}

/// Check if F16 activation precision is enabled (`CHIMERE_ACT_DTYPE=f16`).
///
/// When enabled, hidden states are stored in F16 between layers to halve
/// inter-layer memory usage. All computation inside layers remains F32.
pub(crate) fn act_f16_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ACT_F16: Lazy<bool> = Lazy::new(|| {
        let enabled = std::env::var("CHIMERE_ACT_DTYPE")
            .map(|v| v.eq_ignore_ascii_case("f16"))
            .unwrap_or(false);
        if enabled {
            eprintln!(
                "[ACT_DTYPE] F16 activation precision enabled \
                 — inter-layer hidden states stored as F16"
            );
        }
        enabled
    });
    *ACT_F16
}

/// Layer-0 step-by-step dump helper (activated by `CHIMERE_L0_DUMP=1`).
pub(crate) fn l0_dump(label: &str, t: &Tensor) {
    if std::env::var("CHIMERE_L0_DUMP").is_err() {
        return;
    }
    let l2: f32 = t
        .sqr()
        .and_then(|s| s.sum_all())
        .and_then(|s| s.to_scalar())
        .map(|v: f32| v.sqrt())
        .unwrap_or(0.0);
    let flat: Vec<f32> = t
        .flatten_all()
        .and_then(|f| f.to_vec1())
        .unwrap_or_default();
    let n = flat.len().min(5);
    eprintln!(
        "[L0] {} L2={:.4} shape={:?} first{}={:?}",
        label,
        l2,
        t.dims(),
        n,
        &flat[..n]
    );
}
