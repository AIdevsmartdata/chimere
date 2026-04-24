//! # M1 J7 — 1000-iteration slot leak detector
//!
//! Model-free integration test. Allocates a `SlotPool`, then runs 1000
//! rounds of:
//!   - mark every slot Generating
//!   - push a few tokens / apply engram stub / emit on a live mpsc
//!   - mark every slot Free
//!
//! At the end, asserts that:
//!
//! 1. `pool.len()` has stayed at `NUM_SLOTS` (no `Slot` was added or
//!    pruned behind our back),
//! 2. every slot is Free,
//! 3. no slot still carries a `SamplerHandle` that was freed under the
//!    `Drop` impl — we check that the Option stays `Some` for the whole
//!    run (we install a null-handle "sampler" at boot that survives
//!    `mark_free`, mirroring production where `alloc_samplers_with_dry`
//!    preserves per-slot pointers across conversations).
//! 4. the EngramHandle `Arc` refcount has not runaway-grown. We stash one
//!    `Weak` reference before the loop and one after, and check the
//!    `strong_count` matches: `NUM_SLOTS + 1` (one per slot + our
//!    stash).
//!
//! The latter two catch the class of bugs where `mark_free()` accidentally
//! starts replacing / re-allocating the per-slot FFI handles on every
//! reuse, which would leak C++ memory in production.
//!
//! Running
//! -------
//! ```bash
//! cargo test --release --features server --test slot_leak_1000 -- --nocapture
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use chimere_deltanet::engram_lookup::{EngramLookup, MultiEngramLookup};
use chimere_deltanet::slot_scheduler::{
    SamplerHandle, Slot, SlotPool, SlotState, StreamMsg,
};
use tokio::sync::mpsc;

const NUM_SLOTS: usize = 4;
const NUM_ITER: usize = 1000;

#[test]
fn slot_pool_1000_iterations_no_leak() {
    // Build the pool and install placeholder samplers (null-handle) so we
    // can observe that the Option<SamplerHandle> is preserved across
    // `mark_free()`. In production the handle would be a real C++ pointer,
    // but the preservation invariant is independent of the pointer value.
    let mut pool = SlotPool::new(NUM_SLOTS);
    assert_eq!(pool.len(), NUM_SLOTS);
    for i in 0..NUM_SLOTS as u32 {
        let s = pool.get_mut(i).unwrap();
        // SAFETY: `from_raw(null)` is explicitly documented as the
        // "no-op sampler" idiom. All ops on the handle are no-ops,
        // and Drop is skipped because the pointer is null.
        s.sampler = Some(unsafe { SamplerHandle::from_raw(std::ptr::null_mut()) });
    }

    // Build ONE shared engram (mmap of /tmp file) and attach to each slot.
    // We Arc::downgrade() a Weak before and after the loop so we can check
    // the refcount is stable.
    let engram_path = "/tmp/j7_leak_engram.engr";
    let corpus: Vec<u32> = (0..64u32).collect();
    EngramLookup::build(&corpus, 2, engram_path).expect("build engram");
    let table = EngramLookup::from_file(engram_path).expect("load engram");
    let multi = Arc::new(MultiEngramLookup::from_single("j7-leak".into(), table));
    let weak_before = Arc::downgrade(&multi);
    pool.attach_engram(Arc::clone(&multi), 0.0);

    // Expected refcount after attach:
    //   - 1 for our `multi` local binding
    //   - NUM_SLOTS for the per-slot Arc::clone inside attach_engram
    let expected_strong = NUM_SLOTS + 1;
    assert_eq!(
        Arc::strong_count(&multi),
        expected_strong,
        "engram Arc strong_count after attach_engram",
    );

    // ---------------------------------------------------------------
    // Iteration loop — the core of the leak test.
    //
    // Each iteration:
    //   (a) marks every slot Generating
    //   (b) attaches a fresh mpsc tx to each slot, emits 2 tokens
    //   (c) pushes a few tokens into recent_context
    //   (d) applies engram bias (no-op because alpha=0 & null sampler)
    //   (e) drops the rx end (simulating a fast client)
    //   (f) mark_free for every slot, with an emit that returns false
    //
    // We also assert once per iteration that no slot was mutated to None.
    // ---------------------------------------------------------------
    let emit_success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    for iter in 0..NUM_ITER {
        // (a) + (b): wire each slot with a live channel, emit a couple of
        // Token frames, then drop the rx so the next try_emit returns
        // Closed. This exercises the per-slot tx set/clear path.
        let mut rxs: Vec<mpsc::Receiver<StreamMsg>> = Vec::with_capacity(NUM_SLOTS);
        for i in 0..NUM_SLOTS as u32 {
            let (tx, rx) = mpsc::channel::<StreamMsg>(4);
            let s = pool.get_mut(i).unwrap();
            s.state = SlotState::Generating;
            s.tx = Some(tx);
            // (c): push some tokens into recent_context. Capped at 256 by
            // the implementation, so this cannot grow unbounded.
            for t in 0u32..8 {
                s.push_context(t);
            }
            // (d): apply engram bias — with alpha=0 and null sampler,
            // this is a fast no-op, but it still exercises the Option
            // match paths so we'd notice if an accidental `.take()` were
            // added there.
            s.apply_engram_bias_to_sampler();
            // Emit two tokens on the live channel.
            for tok_val in 0..2u32 {
                let ok = s.try_emit(StreamMsg::Token {
                    text: format!("i{}_{}", iter, tok_val),
                    logprob: None,
                });
                if ok {
                    emit_success_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            rxs.push(rx);
        }

        // Sanity inside the loop — every iteration, every slot still
        // holds its sampler Option. Catches a regression where
        // mark_free() would .take() the sampler.
        for i in 0..NUM_SLOTS as u32 {
            let s = pool.get_mut(i).unwrap();
            assert!(
                s.sampler.is_some(),
                "iter {}: slot {} lost its sampler Option (would leak per-slot C++ pointer in prod)",
                iter, i,
            );
            assert!(
                s.engram.is_some(),
                "iter {}: slot {} lost its engram Option",
                iter, i,
            );
        }

        // (e): drop all receivers to simulate clients disconnecting.
        drop(rxs);

        // (f): mark_free every slot. This is the hot reclaim path the
        // dispatcher calls when a stream completes — the one we are
        // proving does not leak.
        for i in 0..NUM_SLOTS as u32 {
            let s = pool.get_mut(i).unwrap();
            s.mark_free();
            assert!(s.is_free(), "iter {}: slot {} not Free after mark_free", iter, i);
            // mark_free MUST preserve the sampler/engram Options
            // (production contract: the C++ pointers live for the whole
            // lifetime of the pool, not per-conversation).
            assert!(
                s.sampler.is_some(),
                "iter {}: mark_free dropped the per-slot sampler Option",
                iter,
            );
            assert!(
                s.engram.is_some(),
                "iter {}: mark_free dropped the per-slot engram Option",
                iter,
            );
        }
    }
    let elapsed = start.elapsed();
    eprintln!(
        "[j7-leak] {} iterations over {} slots: {:.2?} ({} successful emits)",
        NUM_ITER,
        NUM_SLOTS,
        elapsed,
        emit_success_count.load(Ordering::Relaxed),
    );

    // ---------------------------------------------------------------
    // Post-loop invariants.
    // ---------------------------------------------------------------
    assert_eq!(pool.len(), NUM_SLOTS, "pool.len() changed under load");
    assert_eq!(
        pool.num_active(),
        0,
        "every slot must be Free after the loop",
    );

    // Engram Arc refcount should be unchanged from before the loop:
    // attach_engram cloned once per slot, the loop never touched the
    // Arc, so `strong_count` should still be NUM_SLOTS + 1 (our local
    // + per-slot clones).
    assert_eq!(
        Arc::strong_count(&multi),
        expected_strong,
        "engram Arc strong_count drifted during the loop (leak!)",
    );
    // Weak can still upgrade — the mmap is alive.
    assert!(
        weak_before.upgrade().is_some(),
        "Weak<MultiEngramLookup> could not upgrade; mmap dropped unexpectedly",
    );

    // Drop the pool: now the per-slot Arc::clones should disappear and
    // strong_count should become 1 (just our local `multi`).
    drop(pool);
    assert_eq!(
        Arc::strong_count(&multi),
        1,
        "after dropping pool, engram Arc should have strong_count=1 (only our local)",
    );

    // Cleanup the mmap'd file.
    let _ = std::fs::remove_file(engram_path);
}
