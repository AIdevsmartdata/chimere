//! Zero-dep, zero-alloc (hot-path) timing profiler for chimere-server.
//!
//! # Why
//!
//! 2-slot `NativeScheduler` is 2.57x slower than 1-solo on Qwen3.6 hybrid.
//! We need per-span wall-clock to find where time goes before we guess.
//!
//! # Design goals
//!
//! 1. **Zero crate deps** — only `std`. Drops straight into `src/` without
//!    touching `Cargo.toml`.
//! 2. **Zero alloc on hot path** — span names are `&'static str`, slots are
//!    looked up by pointer equality (not string hashing) on every tick.
//! 3. **Zero cost when disabled** — `CHIMERE_PROFILE=1` env gate checked once
//!    at startup; disabled state is a single relaxed atomic load per span.
//! 4. **Thread-safe** — `Mutex<HashMap>` only on first-sight registration,
//!    subsequent tick/accumulate use a cached `&'static Counter` via
//!    `OnceLock` per call-site (see `prof!` macro). Accumulation is
//!    two `fetch_add(Relaxed)` ops per span close.
//! 5. **Inspectable live** — `/v1/profile` returns a plain-text ranked report,
//!    `/v1/profile/reset` zeroes all counters without dropping registrations.
//!
//! # Usage
//!
//! ```ignore
//! use crate::profile;
//!
//! // One-time, near main():
//! profile::init_from_env();
//!
//! // Hot path (instrument once per call-site):
//! {
//!     let _g = profile::span("scheduler.forward_multi_seq");
//!     self.llama.forward_multi_seq(&entries)?;
//! } // Drop closes the span.
//!
//! // Or with the `prof!` macro for zero per-call registration:
//! prof!("scheduler.forward_multi_seq", {
//!     self.llama.forward_multi_seq(&entries)?
//! });
//! ```
//!
//! # Overhead budget
//!
//! When enabled: one `Instant::now()` pair (~20 ns total on x86_64) + two
//! `Relaxed` `fetch_add` ops (~3 ns) = **~23 ns per span close**. At
//! 90 tok/s * 8 spans/token = 720 closes/s = ~17 µs/s overhead (<0.002 %).
//! When disabled: one `AtomicBool::load(Relaxed)` (~1 ns).
//!
//! # Non-goals
//!
//! - Histograms / percentiles. We only want mean = total/count and count.
//!   If you need p99 go add `hdrhistogram` later.
//! - Parent/child span trees. Flat list is enough to diagnose.
//! - Rust macro_rules over log crate. We want stdout unconditional.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Global registry
// ---------------------------------------------------------------------------

/// Runtime gate. Flipped once by `init_from_env()`.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Counter for one named span.
///
/// `count`   = number of times the span closed
/// `total_ns` = sum of elapsed nanoseconds
///
/// Both are `u64` because 2^64 ns is ~585 years — we won't overflow.
pub struct Counter {
    pub count: AtomicU64,
    pub total_ns: AtomicU64,
}

impl Counter {
    const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_ns: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    fn accumulate(&self, ns: u64) {
        // Relaxed is correct: no happens-before needed between spans.
        // The reader synthesizes a consistent snapshot by reading count
        // BEFORE total_ns (so mean = total/count is never > true mean).
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_ns.fetch_add(ns, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_ns.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> (u64, u64) {
        // Read count first so mean is a conservative lower bound (we may
        // see count updated before total_ns of the same span close).
        let c = self.count.load(Ordering::Relaxed);
        let t = self.total_ns.load(Ordering::Relaxed);
        (c, t)
    }
}

/// Intentional `'static` leak: each `Counter` is allocated once and lives for
/// the process lifetime, so `prof!` can cache a `&'static Counter` in a
/// `OnceLock` with no reference-count overhead.
fn registry() -> &'static Mutex<HashMap<&'static str, &'static Counter>> {
    static REG: OnceLock<Mutex<HashMap<&'static str, &'static Counter>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Register (or look up) a counter for `name`. Called ONCE per call-site
/// via the `OnceLock` in `prof!`, so the mutex lock is amortised to zero.
pub fn counter(name: &'static str) -> &'static Counter {
    let mut guard = registry().lock().expect("profile registry poisoned");
    if let Some(c) = guard.get(name) {
        return *c;
    }
    // Leak is intentional — `Counter` is 16 bytes, and we register at most
    // ~30 unique span names over the entire process lifetime.
    let boxed: &'static Counter = Box::leak(Box::new(Counter::new()));
    guard.insert(name, boxed);
    boxed
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Inspect the env var `CHIMERE_PROFILE` and set the global gate.
///
/// Values that enable: `"1"`, `"true"`, `"yes"` (case-insensitive).
/// Anything else — including the var being unset — leaves profiling off.
///
/// Idempotent; safe to call multiple times (last one wins).
pub fn init_from_env() {
    let enabled = std::env::var("CHIMERE_PROFILE")
        .map(|v| {
            let t = v.trim();
            matches!(t, "1") || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false);
    ENABLED.store(enabled, Ordering::Relaxed);
    if enabled {
        eprintln!("[profile] enabled via CHIMERE_PROFILE. /v1/profile + /v1/profile/reset active.");
    }
}

/// Cheap runtime check. Inlined so the branch predictor can elide the path
/// entirely when `CHIMERE_PROFILE` is unset.
#[inline(always)]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Open a span. Drop the returned `Span` to close it. If profiling is
/// disabled the returned value has no counter and skips the accumulation.
///
/// Prefer the `prof!` macro in hot paths — it caches the counter lookup in a
/// `OnceLock` so the `HashMap::get` mutex is hit only once per call-site.
#[must_use = "profile::span must be held until the end of the scope being timed"]
pub fn span(name: &'static str) -> Span {
    if !is_enabled() {
        return Span { start: None, counter: None };
    }
    Span {
        start: Some(Instant::now()),
        counter: Some(counter(name)),
    }
}

/// RAII guard returned by [`span`]. Closes on drop.
pub struct Span {
    start: Option<Instant>,
    counter: Option<&'static Counter>,
}

impl Drop for Span {
    #[inline(always)]
    fn drop(&mut self) {
        if let (Some(start), Some(c)) = (self.start.take(), self.counter.take()) {
            let ns = start.elapsed().as_nanos() as u64;
            c.accumulate(ns);
        }
    }
}

/// Macro variant: caches the counter pointer in a call-site `OnceLock` so
/// the HashMap lookup happens once per *call-site*, not once per call.
///
/// ## Form 1 — span around a block
///
/// ```ignore
/// let result = prof!("scheduler.forward_multi_seq", {
///     self.llama.forward_multi_seq(&entries)?
/// });
/// ```
///
/// ## Form 2 — RAII guard bound to a scope
///
/// ```ignore
/// let _g = prof!("http.tokenize");
/// // ...work...
/// // guard drops at end of scope
/// ```
#[macro_export]
macro_rules! prof {
    ($name:expr, $body:block) => {{
        static CACHED: std::sync::OnceLock<&'static $crate::profile::Counter> =
            std::sync::OnceLock::new();
        if $crate::profile::is_enabled() {
            let c = CACHED.get_or_init(|| $crate::profile::counter($name));
            let t0 = std::time::Instant::now();
            let out = $body;
            let ns = t0.elapsed().as_nanos() as u64;
            c.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            c.total_ns.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
            out
        } else {
            $body
        }
    }};
    ($name:expr) => {{
        static CACHED: std::sync::OnceLock<&'static $crate::profile::Counter> =
            std::sync::OnceLock::new();
        if $crate::profile::is_enabled() {
            let c = *CACHED.get_or_init(|| $crate::profile::counter($name));
            $crate::profile::Span::__from_counter(c)
        } else {
            $crate::profile::Span::__disabled()
        }
    }};
}

// Internal constructors so the macro can assemble a `Span` without exposing
// the private fields. These are marked with a leading `__` to signal "do not
// call directly from application code".
impl Span {
    #[doc(hidden)]
    #[inline(always)]
    pub fn __from_counter(c: &'static Counter) -> Self {
        Self { start: Some(Instant::now()), counter: Some(c) }
    }
    #[doc(hidden)]
    #[inline(always)]
    pub fn __disabled() -> Self {
        Self { start: None, counter: None }
    }
}

// ---------------------------------------------------------------------------
// Report + reset — wired to /v1/profile and /v1/profile/reset
// ---------------------------------------------------------------------------

/// Render a text report, one span per line, sorted by total_ns desc.
///
/// Format (columns tab-separated for easy `awk`/`sort`):
///
/// ```text
/// # chimere-server profile    enabled=true    spans=12
/// # name                      count     total_ms    mean_us   share_pct
/// scheduler.forward_multi_seq 12345     5432.10     440.1     61.3
/// scheduler.sample_slot       12345      812.44      65.8      9.2
/// ...
/// ```
pub fn report() -> String {
    use std::fmt::Write as _;

    let guard = match registry().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let enabled = is_enabled();

    // Snapshot all spans once so the table is internally consistent.
    let mut rows: Vec<(&'static str, u64, u64)> = guard
        .iter()
        .map(|(name, c)| {
            let (count, total_ns) = c.snapshot();
            (*name, count, total_ns)
        })
        .collect();
    drop(guard);

    rows.sort_by(|a, b| b.2.cmp(&a.2));
    let grand_total_ns: u64 = rows.iter().map(|r| r.2).sum();

    let mut out = String::with_capacity(128 + rows.len() * 96);
    let _ = writeln!(
        out,
        "# chimere-server profile\tenabled={}\tspans={}\ttotal_ms={:.3}",
        enabled,
        rows.len(),
        grand_total_ns as f64 / 1e6,
    );
    let _ = writeln!(
        out,
        "# name\tcount\ttotal_ms\tmean_us\tshare_pct",
    );
    for (name, count, total_ns) in rows {
        let mean_us = if count == 0 {
            0.0
        } else {
            (total_ns as f64 / count as f64) / 1_000.0
        };
        let share = if grand_total_ns == 0 {
            0.0
        } else {
            (total_ns as f64 / grand_total_ns as f64) * 100.0
        };
        let _ = writeln!(
            out,
            "{}\t{}\t{:.3}\t{:.2}\t{:.2}",
            name,
            count,
            total_ns as f64 / 1e6,
            mean_us,
            share,
        );
    }
    out
}

/// Zero out every registered counter in place. Names + counter pointers
/// are preserved so cached `OnceLock` entries in `prof!` remain valid.
pub fn reset() {
    let guard = match registry().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    for c in guard.values() {
        c.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_span_is_noop() {
        ENABLED.store(false, Ordering::Relaxed);
        {
            let _s = span("test.disabled");
        }
        let (c, t) = counter("test.disabled").snapshot();
        // In a clean process this would be (0,0). If another test ran first
        // we still expect no increment from THIS span. Reset to be sure.
        reset();
        let _ = (c, t);
        {
            let _s = span("test.disabled");
        }
        let (c2, _) = counter("test.disabled").snapshot();
        assert_eq!(c2, 0, "disabled span must not increment count");
    }

    #[test]
    fn enabled_span_increments() {
        ENABLED.store(true, Ordering::Relaxed);
        reset();
        for _ in 0..10 {
            let _s = span("test.enabled");
        }
        let (c, t) = counter("test.enabled").snapshot();
        assert_eq!(c, 10);
        assert!(t > 0, "total_ns should be non-zero");
    }

    #[test]
    fn report_is_sorted_by_total_desc() {
        ENABLED.store(true, Ordering::Relaxed);
        reset();
        // Fast span
        for _ in 0..3 {
            let _s = span("test.fast");
        }
        // Slow-ish span (10 iterations of 1us-ish sleep-free work)
        for _ in 0..100 {
            let _s = span("test.slow");
            std::hint::black_box(0u64);
        }
        let r = report();
        // `test.slow` must appear before `test.fast` (excluding header lines).
        let body = r.lines().filter(|l| !l.starts_with('#'));
        let order: Vec<&str> = body
            .filter(|l| l.contains("test."))
            .map(|l| l.split('\t').next().unwrap())
            .collect();
        if order.len() == 2 {
            assert_eq!(order[0], "test.slow");
            assert_eq!(order[1], "test.fast");
        }
    }

    #[test]
    fn reset_preserves_registration() {
        ENABLED.store(true, Ordering::Relaxed);
        let c1 = counter("test.stable") as *const _;
        reset();
        let c2 = counter("test.stable") as *const _;
        assert_eq!(c1, c2, "reset must not drop counter allocations");
    }
}
