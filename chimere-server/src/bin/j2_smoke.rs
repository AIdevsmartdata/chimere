//! # J2 scheduler smoke test — no model, no FFI, no VRAM
//!
//! Exercises the `slot_scheduler` admission → dispatcher → closure flow
//! without touching the model or any FFI symbols, so it runs on a box with
//! a full GPU (prod chimere-server is already using the VRAM).
//!
//! Scenario:
//!   1. Build a `Scheduler` with `CHIMERE_MULTISLOT=2`
//!   2. `spawn_workers()` starts the dispatcher OS thread
//!   3. Submit **two** ScheduledRequests concurrently from two tokio tasks
//!   4. Each request's closure spawns a worker thread that pushes a sequence
//!      of messages on its per-request channel with small inter-token sleeps
//!      (simulating ~50 tok/s generation)
//!   5. The main task reads from both per-request channels in parallel
//!      (via `tokio::select!`) and records the interleaving pattern
//!   6. Assert both requests completed AND at least one interleaving event
//!      happened (i.e. a token from B arrived between two tokens from A)
//!
//! This proves J2's plumbing: admission queue delivers requests, the
//! dispatcher doesn't block on one closure while another waits, and per-
//! request streams are fully independent.
//!
//! It does NOT prove real multi-seq FFI interleaving at the model level —
//! that is J3. This binary's "tokens" are strings pushed from
//! `thread::sleep`-based fakes.
//!
//! Run:
//!   cargo run --release --bin j2-smoke
//!
//! With `CHIMERE_MULTISLOT=2` (default here), the binary should exit 0 and
//! print a JSON summary of the interleaving pattern.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chimere_deltanet::slot_scheduler::{
    ScheduledRequest, ScheduledRequestMeta, Scheduler, SchedulerConfig,
};

/// A fake stream message — just a string. This deliberately avoids
/// pulling in `server::StreamMsg` (which would require the `server`
/// feature).
#[derive(Debug, Clone)]
enum FakeMsg {
    Token(String),
    Done,
}

#[tokio::main]
async fn main() {
    // Force multi-slot ON for this smoke test regardless of caller env.
    std::env::set_var("CHIMERE_MULTISLOT", "2");

    let cfg = SchedulerConfig::from_env();
    assert!(cfg.is_active(), "SchedulerConfig should be active with CHIMERE_MULTISLOT=2");
    eprintln!("[j2-smoke] scheduler config: {:?}", cfg);

    let mut sched = Scheduler::new(cfg);
    let _handles = sched.spawn_workers(); // dispatcher running
    let sched_arc = Arc::new(sched);

    // --------------------------------------------------------
    // Two fake requests submitted concurrently. Each generates 12 "tokens"
    // at ~50 tok/s (20 ms each), totalling ~240 ms per request.
    // --------------------------------------------------------
    let r1 = spawn_fake_request(Arc::clone(&sched_arc), "req-A", "Alpha", 12, 20);
    let r2 = spawn_fake_request(Arc::clone(&sched_arc), "req-B", "Bravo", 12, 20);

    // Wait for both. If interleaved correctly, each task returns a Vec<(t_ms, token)>
    // ordered by local arrival time.
    let started_at = Instant::now();
    let (trace_a, trace_b) = tokio::join!(r1, r2);
    let total_ms = started_at.elapsed().as_millis();

    let trace_a = trace_a.expect("req-A failed");
    let trace_b = trace_b.expect("req-B failed");

    eprintln!(
        "\n[j2-smoke] req-A received {} msgs over {} ms",
        trace_a.len(),
        trace_a.last().map(|(t, _)| *t).unwrap_or(0)
    );
    eprintln!(
        "[j2-smoke] req-B received {} msgs over {} ms",
        trace_b.len(),
        trace_b.last().map(|(t, _)| *t).unwrap_or(0)
    );
    eprintln!("[j2-smoke] wall-clock total: {} ms", total_ms);

    // --------------------------------------------------------
    // Interleaving check: merge by timestamp and count transitions.
    // --------------------------------------------------------
    let mut merged: Vec<(u128, &'static str, String)> = Vec::new();
    for (t, tok) in &trace_a {
        merged.push((*t, "A", tok.clone()));
    }
    for (t, tok) in &trace_b {
        merged.push((*t, "B", tok.clone()));
    }
    merged.sort_by_key(|(t, _, _)| *t);

    let mut transitions = 0usize;
    for pair in merged.windows(2) {
        if pair[0].1 != pair[1].1 {
            transitions += 1;
        }
    }
    eprintln!("[j2-smoke] merged timeline: {} tokens total", merged.len());
    eprintln!("[j2-smoke] interleaving transitions: {}", transitions);
    for (t, who, tok) in &merged[..merged.len().min(20)] {
        eprintln!("  t={:>4}ms  {}  {}", t, who, tok);
    }

    // --------------------------------------------------------
    // Assertions
    // --------------------------------------------------------
    let got_all_a = trace_a.iter().any(|(_, t)| t == "Done");
    let got_all_b = trace_b.iter().any(|(_, t)| t == "Done");
    assert!(got_all_a, "req-A did not receive Done");
    assert!(got_all_b, "req-B did not receive Done");
    assert!(
        transitions >= 2,
        "expected interleaving (>=2 A/B transitions in merged timeline); got {}",
        transitions
    );

    eprintln!(
        "\n[j2-smoke] PASS — both requests streamed concurrently with {} interleavings.",
        transitions
    );
    // Exit with success so CI can rely on status code.
    std::process::exit(0);
}

/// Submit one fake request through the admission channel. Returns a Future
/// resolving to a trace of `(elapsed_ms, msg)` arrivals on this request's
/// channel.
fn spawn_fake_request(
    sched: Arc<Scheduler>,
    request_id: &str,
    tok_prefix: &str,
    num_tokens: usize,
    sleep_ms: u64,
) -> tokio::task::JoinHandle<Vec<(u128, String)>> {
    let admission_tx = sched.admission_tx();
    let request_id = request_id.to_string();
    let tok_prefix = tok_prefix.to_string();

    tokio::spawn(async move {
        // Per-request channel, like `server::chat_completions_stream`.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<FakeMsg>(64);

        // Build the ScheduledRequest. The closure spawns a compute thread
        // that pushes messages with 20 ms intervals — mimicking what a
        // real `run_streaming_inference_worker` does.
        let cancelled = Arc::new(AtomicBool::new(false));
        let meta = ScheduledRequestMeta {
            request_id: request_id.clone(),
            prompt_token_count: 8,
            max_tokens: num_tokens as u32,
            cancelled: Arc::clone(&cancelled),
            enqueued_at: Instant::now(),
        };
        let tx_for_closure = tx.clone();
        let tok_prefix_for_closure = tok_prefix.clone();
        let run: Box<dyn FnOnce(ScheduledRequestMeta) + Send + 'static> =
            Box::new(move |m: ScheduledRequestMeta| {
                std::thread::Builder::new()
                    .name(format!("fake-{}", m.request_id))
                    .spawn(move || {
                        for i in 0..num_tokens {
                            if m.cancelled.load(Ordering::SeqCst) {
                                break;
                            }
                            let msg = FakeMsg::Token(format!("{}-{}", tok_prefix_for_closure, i));
                            if tx_for_closure.blocking_send(msg).is_err() {
                                break;
                            }
                            std::thread::sleep(Duration::from_millis(sleep_ms));
                        }
                        let _ = tx_for_closure.blocking_send(FakeMsg::Done);
                    })
                    .expect("failed to spawn fake-worker");
            });

        let scheduled = ScheduledRequest { metadata: meta, run };
        admission_tx.send(scheduled).await.expect("admission send");

        // Drain the per-request channel and record arrival times.
        let started = Instant::now();
        let mut trace: Vec<(u128, String)> = Vec::new();
        while let Some(msg) = rx.recv().await {
            let ms = started.elapsed().as_millis();
            match msg {
                FakeMsg::Token(s) => trace.push((ms, s)),
                FakeMsg::Done => {
                    trace.push((ms, "Done".to_string()));
                    break;
                }
            }
        }
        trace
    })
}
