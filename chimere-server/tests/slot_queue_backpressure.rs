//! # M1 J7 — slot queue backpressure (FIFO under saturation)
//!
//! Model-free integration test. Proves that when 8 `ScheduledRequest`s are
//! pushed into a scheduler configured with `CHIMERE_MULTISLOT=2` and a small
//! queue capacity, the dispatcher drains them **in the order they were
//! admitted** — i.e. FIFO — and nothing deadlocks when senders block on a
//! full admission channel.
//!
//! Why this test exists
//! --------------------
//! J2 wired a `tokio::sync::mpsc::channel<ScheduledRequest>(queue_cap)` as
//! the admission path. Back-pressure there is "hope the channel bound plus
//! `send().await` do the right thing" — this test verifies two properties
//! the production path relies on:
//!
//! 1. **FIFO ordering at the dispatcher**: the dispatcher calls
//!    `req.run(meta)` in strict admission order. We assert by recording,
//!    inside each closure, a monotonically increasing sequence number.
//!    The closures execute fast (no real inference) so the order we
//!    observe is the order the dispatcher pulled them off the queue.
//!
//! 2. **No deadlock when the queue fills**: we size `queue_cap = 2` with
//!    8 concurrent senders; each sender `send().await`s. The slow drain
//!    forces 6 senders to suspend. If the dispatcher somehow back-pressured
//!    itself (e.g. the closure captured the same tx and tried to send again),
//!    this would hang forever. A 10-second wall-clock timeout guards that.
//!
//! This is NOT a throughput test (see `concurrent_two_slots.rs` / `concurrent_four_slots.rs`).
//! The closures here run in microseconds; the numbers observed are about
//! ordering, not tok/s.
//!
//! Running
//! -------
//! ```bash
//! cargo test --release --features server --test slot_queue_backpressure -- --nocapture
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use chimere_deltanet::slot_scheduler::{
    ScheduledRequest, ScheduledRequestMeta, Scheduler, SchedulerConfig,
};

const NUM_REQUESTS: usize = 8;
const QUEUE_CAP: usize = 2;
const OVERALL_TIMEOUT: Duration = Duration::from_secs(10);

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slot_queue_backpressure_fifo() {
    // Force multi-slot ON regardless of caller env, so J1's default
    // (N=1, scheduler disabled) does not silently skip the test.
    std::env::set_var("CHIMERE_MULTISLOT", "2");
    std::env::set_var("CHIMERE_ADMISSION_QUEUE", &QUEUE_CAP.to_string());
    // Belt-and-braces: CPU-only, not that it matters (no model is loaded).
    std::env::set_var("CHIMERE_N_GPU_LAYERS", "0");

    let cfg = SchedulerConfig::from_env();
    assert!(cfg.is_active(), "need multi-slot enabled for this test");
    assert_eq!(cfg.queue_cap, QUEUE_CAP, "queue_cap must be tight for backpressure");

    let mut sched = Scheduler::new(cfg);
    let _worker = sched.spawn_workers();
    let sched = Arc::new(sched);

    // Each closure records the order in which the dispatcher invoked it.
    // `dispatched_counter` is the global "# of closures already run" at the
    // moment the closure fires — this is the sequence number the dispatcher
    // would have assigned in a strict FIFO queue. Slot == index into the
    // recorded sequence array.
    let observed_order: Arc<parking_lot_proxy::RwLock> = Arc::new(parking_lot_proxy::RwLock::new());
    let dispatched_counter = Arc::new(AtomicU32::new(0));
    // Record when each request was admitted (from the sender's PoV) and
    // when it started running (from the dispatcher's PoV). Used only for
    // diagnostic prints on failure — the assertion is on order.
    let start = Instant::now();

    let mut send_tasks = Vec::with_capacity(NUM_REQUESTS);
    for req_idx in 0..NUM_REQUESTS {
        let admission_tx = sched.admission_tx();
        let observed = Arc::clone(&observed_order);
        let counter = Arc::clone(&dispatched_counter);
        send_tasks.push(tokio::spawn(async move {
            // Closure runs on the dispatcher thread. It writes the pair
            // (request_index, dispatch_sequence) so the test can later
            // assert the dispatcher pulled them in order 0..NUM_REQUESTS.
            let cancelled = Arc::new(AtomicBool::new(false));
            let enqueued_at = Instant::now();
            let meta = ScheduledRequestMeta {
                request_id: format!("j7-bp-{:02}", req_idx),
                prompt_token_count: 4,
                max_tokens: 1,
                cancelled,
                enqueued_at,
            };
            let obs_for_closure = Arc::clone(&observed);
            let ctr_for_closure = Arc::clone(&counter);
            let run: Box<dyn FnOnce(ScheduledRequestMeta) + Send + 'static> =
                Box::new(move |_m: ScheduledRequestMeta| {
                    let seq = ctr_for_closure.fetch_add(1, Ordering::SeqCst);
                    obs_for_closure.push((req_idx, seq));
                    // Keep the dispatcher busy for a small, deterministic
                    // interval so the admission channel genuinely fills.
                    // If closures return instantly, the 8 sends may each
                    // slip in before the dispatcher wakes up, defeating
                    // the backpressure scenario.
                    std::thread::sleep(Duration::from_millis(5));
                });
            let scheduled = ScheduledRequest { metadata: meta, run };
            // NOTE: .await on a full channel suspends this task. That's the
            // whole point — we want to demonstrate no deadlock while senders
            // wait for the dispatcher to drain the queue.
            admission_tx
                .send(scheduled)
                .await
                .expect("admission send must not fail");
            req_idx
        }));
        // Stagger by 1 ms so admission order is unambiguous (i.e. the
        // senders themselves observe a clear sequence; without the stagger
        // tokio could schedule the 8 spawns in any order, in which case
        // "FIFO at dispatcher" would be trivially satisfied but not
        // meaningfully tested).
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    // Wait for every sender to have handed its request off to the queue.
    // Individual sender tasks return as soon as `send().await` unblocks
    // (their request is then somewhere between "in the channel" and "being
    // executed on the dispatcher thread").
    let gate = Instant::now();
    for task in send_tasks {
        let remaining = OVERALL_TIMEOUT.saturating_sub(gate.elapsed());
        tokio::time::timeout(remaining, task)
            .await
            .expect("sender task timed out — dispatcher deadlocked?")
            .expect("sender panicked");
    }

    // Now spin until the dispatcher has run NUM_REQUESTS closures. We
    // cannot await a JoinHandle on the dispatcher thread (it runs until
    // shutdown) so we poll the counter.
    let drain_deadline = Instant::now() + OVERALL_TIMEOUT.saturating_sub(start.elapsed());
    while dispatched_counter.load(Ordering::SeqCst) < NUM_REQUESTS as u32 {
        if Instant::now() > drain_deadline {
            panic!(
                "dispatcher drained only {}/{} requests before timeout",
                dispatched_counter.load(Ordering::SeqCst),
                NUM_REQUESTS,
            );
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    sched.shutdown();

    // Assertion: the sequence of (req_idx, dispatch_seq) pairs, when
    // sorted by dispatch_seq, must yield req_idx == 0, 1, 2, ..., N-1.
    // That's FIFO.
    let mut observed = observed_order.drain();
    observed.sort_by_key(|&(_idx, seq)| seq);
    eprintln!("[j7-backpressure] observed = {:?}", observed);
    assert_eq!(observed.len(), NUM_REQUESTS, "missed a request");
    for (expected_idx, &(got_idx, _seq)) in observed.iter().enumerate() {
        assert_eq!(
            got_idx, expected_idx,
            "FIFO violation: expected req {} at dispatch position {}, got req {}",
            expected_idx, expected_idx, got_idx,
        );
    }
}

// ---------------------------------------------------------------------------
// Tiny lock-free "push-only then drain" recorder.
//
// We cannot use `std::sync::Mutex<Vec<_>>` under a tokio multi_thread runtime
// from inside a blocking-dispatcher thread without risking poisoning on panic;
// a `parking_lot::Mutex` would need an extra dep. Keep it simple: one Mutex
// on a plain Vec, owned by an Arc. Writes from the dispatcher are blocking
// and fast (push + unlock). Drain happens after shutdown so no contention
// remains.
// ---------------------------------------------------------------------------
mod parking_lot_proxy {
    use std::sync::Mutex;

    pub struct RwLock {
        inner: Mutex<Vec<(usize, u32)>>,
    }

    impl RwLock {
        pub fn new() -> Self {
            Self { inner: Mutex::new(Vec::new()) }
        }

        pub fn push(&self, pair: (usize, u32)) {
            let mut g = self.inner.lock().expect("record lock poisoned");
            g.push(pair);
        }

        pub fn drain(&self) -> Vec<(usize, u32)> {
            let mut g = self.inner.lock().expect("record lock poisoned");
            std::mem::take(&mut *g)
        }
    }
}
