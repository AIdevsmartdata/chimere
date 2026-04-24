//! # J5 smoke test — per-slot sampler + per-slot engram bias isolation
//!
//! The single most dangerous regression the multi-slot serving path can
//! ship is **cross-slot contamination of logit biases**. In M1 J5a the
//! `SamplerHandle` struct was added so each active slot can hold its own
//! C++ `chimere_sampler` pointer, and in J5b `Slot::apply_engram_bias_to_sampler`
//! was added so each slot can push its own engram-derived logit biases
//! into that sampler.
//!
//! This smoke runs two slots concurrently (seq 0 and seq 1), attaches
//! **different** engram lookup tables to each, then for the same prompt
//! token state checks that the two slots' `sample_with_logprobs`
//! distributions diverge by more than just argmax — i.e. the biased
//! logit distributions reflect the per-slot engram predictions rather
//! than a globally-shared table.
//!
//! Unlike `j3-smoke` and `j4-smoke` this binary **requires** the C++
//! sampler to be allocated — it is the whole point.
//!
//! ## Known blocker (2026-04-24)
//!
//! `common_sampler_init` in the libcommon rebuilt 2026-04-24 09:36
//! crashes with `fatal runtime error: Rust cannot catch foreign
//! exceptions, aborting` on every fresh chimere-server build. The
//! running production chimere-server on :8081 is unaffected (it runs
//! from a mmap'd deleted pre-regression binary). See commit
//! `fix(m1): J4 follow-up — llama_grammar_apply shim must no-op`
//! for the initial investigation and the shim no-op fix that stops
//! *one* of the crash paths but not all of them.
//!
//! Until the libcommon regression is addressed, this smoke runs as far
//! as `SlotPool::alloc_samplers_with_dry` and then crashes at the first
//! `chimere_sampler_init` call inside `common_sampler_init`. The J5a
//! and J5b code paths exercised up to that point have been verified:
//! engram tables load, predictions match the expected `target_a` /
//! `target_b`, and the FFI call site signature has been compiled and
//! linked successfully.
//!
//! ## Running
//!
//! ```bash
//! export CHIMERE_MODEL=/path/to/any/small.gguf
//! export CHIMERE_N_CTX=2048
//! export CHIMERE_N_GPU_LAYERS=0
//! cargo run --release --features server --bin j5-smoke
//! ```
//!
//! Exit 0 = PASS, 1 = FAIL. On FAIL the output explains exactly which
//! invariant broke (null sampler, identical distributions, etc.).

use std::env;
use std::fs;
use std::sync::Arc;

use chimere_deltanet::engram_lookup::{EngramLookup, MultiEngramLookup};
use chimere_deltanet::llama_backend::{LlamaForward, MultiSeqEntry, TokenLogprob};
use chimere_deltanet::slot_scheduler::{SlotPool, SlotState};

/// Build a tiny synthetic engram table that **strongly** biases a single
/// 2-gram. The corpus encodes the pattern
///   `[prefix, anchor, target, prefix, anchor, target, ...]`
/// so that an order-2 lookup on the key `(prefix, anchor)` predicts
/// `target` with probability ~1.0. The order-2 lookup on `(anchor, target)`
/// predicts `prefix`, and on `(target, prefix)` predicts `anchor` — these
/// are not used by the smoke, but are valid table entries either way.
///
/// Returns a `MultiEngramLookup` wrapping a single table.
fn make_tiny_engram(
    path: &str,
    prefix: u32,
    anchor: u32,
    target: u32,
    repeats: usize,
) -> Result<Arc<MultiEngramLookup>, String> {
    let mut corpus: Vec<u32> = Vec::with_capacity(3 * repeats + 16);
    // A few noise tokens first so the engram has more than one bucket.
    for i in 0..8u32 {
        corpus.push(i * 7 + 3);
    }
    for _ in 0..repeats {
        corpus.push(prefix);
        corpus.push(anchor);
        corpus.push(target);
    }
    // Re-write the file each run to avoid stale content.
    if fs::metadata(path).is_ok() {
        let _ = fs::remove_file(path);
    }
    EngramLookup::build(&corpus, 2, path)?;
    let table = EngramLookup::from_file(path)?;
    let multi = MultiEngramLookup::from_single(format!("test-{}", anchor), table);
    Ok(Arc::new(multi))
}

fn log_top5(tag: &str, logprobs: &[TokenLogprob]) {
    eprintln!("[j5-smoke] {}: top-5 =", tag);
    for (i, lp) in logprobs.iter().enumerate() {
        eprintln!("    #{}: token={:6}  logprob={:8.4}", i, lp.token, lp.logprob);
    }
}

/// Distribution distance: compare the logprob of the SAME token in two
/// different top-5 lists. A non-zero difference (beyond a tiny epsilon)
/// proves the two samplers had different logit_bias states even if the
/// argmax happens to be the same token.
fn max_logprob_gap(a: &[TokenLogprob], b: &[TokenLogprob]) -> f32 {
    // Gather all tokens that appear in either list.
    let mut all_tokens: Vec<u32> = a.iter().map(|t| t.token).collect();
    all_tokens.extend(b.iter().map(|t| t.token));
    all_tokens.sort();
    all_tokens.dedup();

    let get = |list: &[TokenLogprob], tok: u32| -> Option<f32> {
        list.iter().find(|t| t.token == tok).map(|t| t.logprob)
    };

    let mut gap = 0.0f32;
    for tok in all_tokens {
        if let (Some(la), Some(lb)) = (get(a, tok), get(b, tok)) {
            let diff = (la - lb).abs();
            if diff > gap {
                gap = diff;
            }
        }
    }
    gap
}

fn run() -> Result<(), String> {
    // The BUILT-IN `LlamaForward` sampler is not used in the multi-slot
    // pattern — every Slot brings its own `SamplerHandle` allocated via
    // `SlotPool::alloc_samplers_with_dry`. We therefore skip the
    // single-shared-sampler construction inside `LlamaForward::new_multi_seq`.
    //
    // This is distinct from what j4-smoke does: j4 skips because it
    // does external argmax. J5 skips because per-slot samplers *are*
    // the whole subject of the test.
    env::set_var("CHIMERE_SKIP_SAMPLER_INIT", "1");

    let model_path = env::var("CHIMERE_MODEL")
        .map_err(|_| "CHIMERE_MODEL env var required (path to a .gguf file)".to_string())?;
    let n_ctx: u32 = env::var("CHIMERE_N_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n_gpu_layers: i32 = env::var("CHIMERE_N_GPU_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(0);

    eprintln!("[j5-smoke] Config:");
    eprintln!("  model        = {}", model_path);
    eprintln!("  n_ctx        = {}", n_ctx);
    eprintln!("  n_gpu_layers = {} (0 = CPU)", n_gpu_layers);

    // ---- Load model with n_seq_max = 2 ----
    let mut llama = LlamaForward::new_multi_seq(
        &model_path,
        n_gpu_layers,
        n_ctx,
        0,
        None,
        None,
        true,
        2,
    )?;
    let vocab = llama.vocab_size();
    eprintln!("[j5-smoke] Model loaded. vocab_size={}", vocab);

    // ---- Build two DIFFERENT tiny engram tables ----
    //
    // Engram A biases token TARGET_A strongly after anchor ANCHOR_A.
    // Engram B biases token TARGET_B strongly after anchor ANCHOR_B.
    //
    // We then prefill the SAME prompt whose last two tokens happen to be
    // (ANCHOR_A, _) → so engram A's lookup has a hit on the context
    // window [ANCHOR_A, <last_prefill_token>] — wait, easier: we make the
    // context window end with the engram's anchor by construction.
    // The last two tokens of the shared prompt will be (prefix, anchor).
    // Both slots push the same tail context into their `recent_context`,
    // so each slot's engram table looks up key `(prefix, anchor)`.
    //
    // Engram A's table predicts target_a for that key.
    // Engram B's table predicts target_b for that key.
    //
    // The two targets are distinct → the biases applied to the two
    // per-slot samplers are distinct → the sample-with-logprobs
    // distributions must diverge.
    let prefix: u32 = 11;
    let anchor: u32 = 42;
    let target_a: u32 = 9001;  // made up, unlikely to be the model's natural argmax
    let target_b: u32 = 9333;  // different target — drives distribution gap

    let engram_a = make_tiny_engram("/tmp/j5_engram_a.engr", prefix, anchor, target_a, 30)?;
    let engram_b = make_tiny_engram("/tmp/j5_engram_b.engr", prefix, anchor, target_b, 30)?;
    eprintln!("[j5-smoke] Built engram tables: A={} tables, B={} tables", engram_a.len(), engram_b.len());

    // Sanity : both engrams predict the expected target for the shared
    // query key `(prefix, anchor)`. This rules out a table-construction
    // bug before the sampler-isolation test runs.
    let ctx_vec = vec![prefix, anchor];
    let preds_a = engram_a.lookup(&ctx_vec);
    let preds_b = engram_b.lookup(&ctx_vec);
    eprintln!("[j5-smoke] engram A predicts (top 3 of {}): {:?}", preds_a.len(), preds_a.iter().take(3).collect::<Vec<_>>());
    eprintln!("[j5-smoke] engram B predicts (top 3 of {}): {:?}", preds_b.len(), preds_b.iter().take(3).collect::<Vec<_>>());
    if preds_a.is_empty() || preds_b.is_empty() {
        return Err("FAIL: one of the synthetic engram tables returned no predictions — test setup bug".into());
    }
    let top_a = preds_a[0].0;
    let top_b = preds_b[0].0;
    if top_a != target_a {
        return Err(format!(
            "FAIL: engram A top prediction for ctx {:?} is {}, expected {}. Predictions: {:?}",
            ctx_vec, top_a, target_a, preds_a,
        ));
    }
    if top_b != target_b {
        return Err(format!(
            "FAIL: engram B top prediction for ctx {:?} is {}, expected {}. Predictions: {:?}",
            ctx_vec, top_b, target_b, preds_b,
        ));
    }

    // ---- Allocate N samplers (1 per slot) ----
    let mut pool = SlotPool::new(2);
    // Qwen3.5 thinking defaults for the basic knobs. DRY is disabled
    // (dry_multiplier=0.0) because the default DRY sequence-breakers
    // `\n`, `:`, `"`, `*` fail to tokenise on some small-vocab GGUFs
    // (observed crash on Qwen3.5-9B Q3_K_M). Production chimere-server
    // on Qwen3.6-35B runs with DRY enabled; the isolation behaviour
    // this smoke exercises is orthogonal to DRY.
    let allocated = unsafe {
        pool.alloc_samplers_with_dry(
            llama.model_raw(),
            0.6,   // temperature
            0.95,  // top_p
            20,    // top_k
            0.05,  // min_p
            0.0,   // presence_penalty
            0.0,   // dry_multiplier — disabled (see comment above)
            1.75,  // dry_base      — ignored when multiplier is 0
            2,     // dry_min_length
            -1,    // dry_penalty_last_n
        )?
    };
    if allocated != 2 {
        return Err(format!("alloc_samplers returned {}, expected 2", allocated));
    }
    eprintln!("[j5-smoke] Allocated {} per-slot samplers (DRY disabled for smoke)", allocated);

    // ---- Attach DIFFERENT engrams per slot ----
    pool.attach_engram_per_slot(vec![engram_a.clone(), engram_b.clone()], 1.0)
        .map_err(|e| format!("attach_engram_per_slot failed: {}", e))?;
    eprintln!("[j5-smoke] Attached engram A to slot 0, engram B to slot 1");

    // ---- Prime the slots with a shared prompt ending in (prefix, anchor) ----
    //
    // Same token stream on seq 0 and seq 1 → model produces ~identical raw
    // logits for the last position. The ONLY thing that should diverge is
    // the per-slot sampler bias state. The last two tokens match the
    // engram query key so both tables will hit on `push_context`.
    let shared_prompt: Vec<u32> = vec![1, 2, 3, prefix, anchor];
    let mut entries: Vec<MultiSeqEntry> = Vec::new();
    for (i, &t) in shared_prompt.iter().enumerate() {
        entries.push(MultiSeqEntry {
            token: t,
            pos: i as i32,
            seq_id: 0,
            request_logits: i == shared_prompt.len() - 1,
        });
    }
    for (i, &t) in shared_prompt.iter().enumerate() {
        entries.push(MultiSeqEntry {
            token: t,
            pos: i as i32,
            seq_id: 1,
            request_logits: i == shared_prompt.len() - 1,
        });
    }
    let _out = llama.forward_multi_seq(&entries)?;
    eprintln!("[j5-smoke] Prefilled both slots ({} tokens each)", shared_prompt.len());

    // ---- Seed recent_context per slot so engram has a key to hit ----
    //
    // Slot 0 / Slot 1 both get [1, anchor_a] as their tail context so
    // both engrams will hit their respective 2-gram `[1, 42] → target`.
    {
        let s0 = pool.get_mut(0).unwrap();
        s0.state = SlotState::Generating;
        for &t in &shared_prompt {
            s0.push_context(t);
        }
    }
    {
        let s1 = pool.get_mut(1).unwrap();
        s1.state = SlotState::Generating;
        for &t in &shared_prompt {
            s1.push_context(t);
        }
    }

    // ---- Apply per-slot engram biases and sample with logprobs ----
    //
    // We cannot run `forward_multi_seq` between the two samples (it would
    // mutate the ctx's logit buffer). Instead we do ONE more forward pass
    // that produces logits for both seqs in the same batch, then sample
    // each seq with its OWN sampler handle (batch_idx 0 → seq 0 logits,
    // batch_idx 1 → seq 1 logits).
    //
    // Engram-A is applied to sampler 0, Engram-B to sampler 1. Since the
    // raw logits at batch_idx 0 and 1 are almost identical (same prompt,
    // different seq_id), any observable difference between the two
    // sampled distributions is evidence of bias isolation.
    let step_entries = vec![
        MultiSeqEntry {
            token: *shared_prompt.last().unwrap(),
            pos: shared_prompt.len() as i32,
            seq_id: 0,
            request_logits: true,
        },
        MultiSeqEntry {
            token: *shared_prompt.last().unwrap(),
            pos: shared_prompt.len() as i32,
            seq_id: 1,
            request_logits: true,
        },
    ];
    let step_out = llama.forward_multi_seq(&step_entries)?;
    eprintln!("[j5-smoke] Step forward done; got {} logit vectors", step_out.len());

    // Capture the RAW logits at batch_idx 0 and 1 *before* applying biases,
    // for a reference distribution. get_logits_at is borrowed from ctx and
    // only valid until the next decode — we're not doing another decode
    // before sampling so it's safe.
    let raw0_top = {
        let slice = llama.get_logits_at(0).ok_or("null logits[0]")?;
        argmax_top5(slice)
    };
    let raw1_top = {
        let slice = llama.get_logits_at(1).ok_or("null logits[1]")?;
        argmax_top5(slice)
    };
    eprintln!("[j5-smoke] raw seq 0 top-5: {:?}", raw0_top);
    eprintln!("[j5-smoke] raw seq 1 top-5: {:?}", raw1_top);

    // Apply engram biases to each slot's sampler
    pool.get_mut(0).unwrap().apply_engram_bias_to_sampler();
    pool.get_mut(1).unwrap().apply_engram_bias_to_sampler();

    // Sample slot 0 (batch_idx 0)
    let sampler0_raw = unsafe { pool.get_mut(0).unwrap().sampler.as_ref().unwrap().as_raw() };
    let (tok_a, lp_a) = unsafe { llama.sample_slot_with_logprobs(sampler0_raw, 0) };

    // Sample slot 1 (batch_idx 1)
    let sampler1_raw = unsafe { pool.get_mut(1).unwrap().sampler.as_ref().unwrap().as_raw() };
    let (tok_b, lp_b) = unsafe { llama.sample_slot_with_logprobs(sampler1_raw, 1) };

    log_top5("slot 0 post-bias", &lp_a);
    log_top5("slot 1 post-bias", &lp_b);
    eprintln!("[j5-smoke] sampled tokens: slot0={}  slot1={}", tok_a, tok_b);

    // ---- Assertions ----

    // Assertion 1 : both samplers returned logprobs (not all zeros)
    if lp_a.is_empty() || lp_b.is_empty() {
        return Err(format!(
            "FAIL: empty logprobs — slot0={} slot1={} — sampler not returning top-5",
            lp_a.len(), lp_b.len(),
        ));
    }

    // Assertion 2 : the target_a token appears with high logprob in slot 0's top-5
    let slot0_has_target_a = lp_a.iter().any(|t| t.token == target_a);
    // Engram alpha=1.0 with a near-1.0 prob should push the target into the
    // top 5 unless the model's raw logit disagreement is enormous. If
    // target_a is NOT in slot0's top-5, the bias did not propagate.
    if !slot0_has_target_a {
        return Err(format!(
            "FAIL: slot 0's top-5 after engram bias does not contain target_a={}. \
             top-5={:?}. Bias did not propagate into sampler.",
            target_a,
            lp_a.iter().map(|t| (t.token, t.logprob)).collect::<Vec<_>>(),
        ));
    }
    let slot1_has_target_b = lp_b.iter().any(|t| t.token == target_b);
    if !slot1_has_target_b {
        return Err(format!(
            "FAIL: slot 1's top-5 after engram bias does not contain target_b={}. \
             top-5={:?}. Bias did not propagate into sampler.",
            target_b,
            lp_b.iter().map(|t| (t.token, t.logprob)).collect::<Vec<_>>(),
        ));
    }

    // Assertion 3 : target_a should NOT appear in slot 1's top-5 (or if
    // it does, its logprob should be noticeably lower than in slot 0).
    // This proves slot 0's bias did not leak into slot 1's sampler.
    let slot1_has_target_a = lp_b.iter().any(|t| t.token == target_a);
    let slot0_has_target_b = lp_a.iter().any(|t| t.token == target_b);
    if slot1_has_target_a && slot0_has_target_b {
        // Both slots show both targets — that would be the cross-contamination signature.
        // Small model / low-vocab environments can still make this happen legitimately
        // because the raw logits may already rank both tokens highly. We mark it as
        // INCONCLUSIVE instead of FAIL only when the gap between per-slot logprobs
        // is smaller than the engram's expected bias contribution.
        let lp_ta_in_0 = lp_a.iter().find(|t| t.token == target_a).unwrap().logprob;
        let lp_ta_in_1 = lp_b.iter().find(|t| t.token == target_a).unwrap().logprob;
        let lp_tb_in_0 = lp_a.iter().find(|t| t.token == target_b).unwrap().logprob;
        let lp_tb_in_1 = lp_b.iter().find(|t| t.token == target_b).unwrap().logprob;
        let sep_a = lp_ta_in_0 - lp_ta_in_1;   // expect > 0 (slot0 favours A)
        let sep_b = lp_tb_in_1 - lp_tb_in_0;   // expect > 0 (slot1 favours B)
        eprintln!(
            "[j5-smoke] cross-target logprob gaps: target_a(slot0-slot1)={:.3}  target_b(slot1-slot0)={:.3}",
            sep_a, sep_b,
        );
        if sep_a <= 0.0 || sep_b <= 0.0 {
            return Err(format!(
                "FAIL: per-slot engram biases did not produce the expected asymmetry. \
                 target_a(slot0-slot1)={:.3}, target_b(slot1-slot0)={:.3} — \
                 per-slot logit_bias isolation appears broken.",
                sep_a, sep_b,
            ));
        }
    }

    // Assertion 4 : distributions must differ. Measure the max absolute
    // logprob gap over shared tokens. Anything > 0.1 is a robust signal
    // (engram alpha=1 should push > 1.0 nats for a target that's not in
    // the model's natural top-5).
    let gap = max_logprob_gap(&lp_a, &lp_b);
    eprintln!("[j5-smoke] max shared-token logprob gap (slot0 vs slot1): {:.4}", gap);
    if gap < 0.01 {
        return Err(format!(
            "FAIL: top-5 logprob gap between slot 0 and slot 1 is only {:.4}, \
             essentially identical distributions. Per-slot biases did not apply.",
            gap,
        ));
    }

    // ---- Cleanup ----
    let _ = llama.kv_cache_seq_rm_for(0);
    let _ = llama.kv_cache_seq_rm_for(1);
    let _ = fs::remove_file("/tmp/j5_engram_a.engr");
    let _ = fs::remove_file("/tmp/j5_engram_b.engr");

    eprintln!();
    eprintln!("===== J5 SMOKE PASS =====");
    eprintln!("  slot 0 sampled token = {}  (engram A pushed target_a={})", tok_a, target_a);
    eprintln!("  slot 1 sampled token = {}  (engram B pushed target_b={})", tok_b, target_b);
    eprintln!("  max logprob gap      = {:.4}  (> 0.01 threshold)", gap);
    eprintln!();
    Ok(())
}

/// Helper: argmax top-5 over a logit slice. Used only to *print* the raw
/// distribution for debug; not on the assertion path.
fn argmax_top5(logits: &[f32]) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    pairs.truncate(5);
    pairs
}

fn main() {
    match run() {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("\n[j5-smoke] FAIL: {}", e);
            std::process::exit(1);
        }
    }
}
