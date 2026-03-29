//! Optional cubin loader — provides pre-compiled kernel binary from build.rs.
//!
//! At build time, `build.rs` compiles `kernels/chimere_kernels.cu` into a cubin
//! via nvcc and embeds it into the binary. This module exposes that cubin so that
//! kernel modules can load functions from it instead of going through NVRTC.
//!
//! ## Usage (future — kernel loading not wired yet)
//!
//! ```ignore
//! if cubin_loader::has_cubin() {
//!     let bytes = cubin_loader::cubin_bytes();
//!     // Load module from cubin bytes via cudarc::nvrtc::Ptx::from_binary(...)
//! } else {
//!     // Fall back to NVRTC compile_and_cache()
//! }
//! ```
//!
//! ## Environment
//!
//! - `CHIMERE_NVRTC=1` — force NVRTC path even when cubin is available (for debugging)
//! - `CHIMERE_NO_CUBIN=1` — same as CHIMERE_NVRTC (alias)

/// The cubin bytes, embedded at compile time.
///
/// When build.rs succeeds with nvcc, this contains the real sm_120 cubin.
/// When nvcc fails, this is an empty slice (and `has_cubin()` returns false).
static CUBIN_BYTES: &[u8] = include_bytes!(env!("CHIMERE_CUBIN_PATH"));

/// Returns `true` if a valid pre-compiled cubin is available.
///
/// This checks both:
/// 1. The `chimere_has_cubin` cfg flag set by build.rs on successful nvcc compilation
/// 2. That the embedded bytes are non-empty (defense in depth)
/// 3. That neither `CHIMERE_NVRTC=1` nor `CHIMERE_NO_CUBIN=1` is set
pub fn has_cubin() -> bool {
    cfg!(chimere_has_cubin)
        && !CUBIN_BYTES.is_empty()
        && std::env::var("CHIMERE_NVRTC").as_deref() != Ok("1")
        && std::env::var("CHIMERE_NO_CUBIN").as_deref() != Ok("1")
}

/// Returns the raw cubin bytes.
///
/// The caller should check `has_cubin()` first. If `has_cubin()` is false,
/// this returns an empty slice.
pub fn cubin_bytes() -> &'static [u8] {
    CUBIN_BYTES
}
