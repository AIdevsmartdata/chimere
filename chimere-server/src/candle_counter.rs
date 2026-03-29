//! Candle tensor operation counter.
//!
//! Counts Candle tensor operations per forward pass to track
//! dispatch overhead reduction. Toggle: CHIMERE_COUNT_OPS=1
//!
//! Usage:
//!   candle_counter::reset();
//!   // ... forward pass ...
//!   let count = candle_counter::get();
//!   eprintln!("[CANDLE_OPS] {} total", count);

use std::sync::atomic::{AtomicU64, Ordering};

static OPS_COUNT: AtomicU64 = AtomicU64::new(0);

/// Check if counting is enabled (CHIMERE_COUNT_OPS=1).
pub fn enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_COUNT_OPS").is_ok());
    *ENABLED
}

/// Increment the counter by 1.
#[inline(always)]
pub fn tick() {
    if enabled() {
        OPS_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Increment by n ops (for bulk counting).
#[inline(always)]
pub fn tick_n(n: u64) {
    if enabled() {
        OPS_COUNT.fetch_add(n, Ordering::Relaxed);
    }
}

/// Reset counter to 0 (call before each forward pass).
pub fn reset() {
    OPS_COUNT.store(0, Ordering::Relaxed);
}

/// Get current count.
pub fn get() -> u64 {
    OPS_COUNT.load(Ordering::Relaxed)
}
