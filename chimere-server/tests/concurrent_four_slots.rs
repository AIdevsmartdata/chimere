//! # M1 J7 — concurrent four-slot throughput
//!
//! Integration test. Same shape as `concurrent_two_slots.rs` but with
//! `n_seq_max = 4` and four distinct prompts, one per seq_id. Target
//! aggregate throughput ≥ 3× baseline (plan §7 states 300 vs 100 tok/s
//! on GPU; we relax to 3× on CPU where scheduler overhead dominates
//! and experts on the CPU serialize).
//!
//! Gating
//! ------
//! - Skips if `CHIMERE_MODEL` is unset (CPU-safe default).
//! - If `CHIMERE_J7_RELAX=1`, throughput assertion is **replaced** by
//!   the weakest useful check: "no deadlock, all 4 streams complete,
//!   at least two streams diverge from each other". This guarantees
//!   the test is still doing something on a box too slow to meet the
//!   3× bar.
//!
//! Running
//! -------
//! ```bash
//! export CHIMERE_MODEL=/path/to/tiny.gguf
//! export CHIMERE_N_GPU_LAYERS=0
//! export CHIMERE_N_CTX=2048
//! cargo test --release --features server --test concurrent_four_slots -- --nocapture
//! ```

use std::env;
use std::time::Instant;

use chimere_deltanet::llama_backend::{LlamaForward, MultiSeqEntry};

const NUM_SEQS: usize = 4;

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx: u32 = 0;
    let mut best_val = f32::MIN;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

fn model_path_or_skip() -> Option<String> {
    match env::var("CHIMERE_MODEL") {
        Ok(p) => Some(p),
        Err(_) => {
            eprintln!("[j7-concurrent-four] CHIMERE_MODEL unset — skipping.");
            None
        }
    }
}

#[test]
fn concurrent_four_slots_aggregate_throughput() {
    let model_path = match model_path_or_skip() {
        Some(p) => p,
        None => return,
    };
    let n_ctx: u32 = env::var("CHIMERE_N_CTX")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let n_gpu_layers: i32 = env::var("CHIMERE_N_GPU_LAYERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n_gen: usize = env::var("CHIMERE_J7_NGEN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15);
    let relax: bool = env::var("CHIMERE_J7_RELAX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    env::set_var("CHIMERE_SKIP_SAMPLER_INIT", "1");

    eprintln!(
        "[j7-concurrent-four] model={}, n_gpu_layers={}, n_gen={}, relax={}",
        model_path, n_gpu_layers, n_gen, relax,
    );

    // ---- Baseline : single seq in isolation ----
    let (baseline_tokps, vocab_size) = {
        let mut llama = LlamaForward::new_multi_seq(
            &model_path, n_gpu_layers, n_ctx, 0, None, None, true, NUM_SEQS as u32,
        ).expect("baseline new_multi_seq");
        let vocab = llama.vocab_size();
        let prompt: Vec<u32> = vec![1, 450, 8448, 374, 6548];
        let mut entries = Vec::new();
        for (i, &t) in prompt.iter().enumerate() {
            entries.push(MultiSeqEntry {
                token: t, pos: i as i32, seq_id: 0,
                request_logits: i == prompt.len() - 1,
            });
        }
        let prefill = llama.forward_multi_seq(&entries).expect("baseline prefill");
        let mut tok = argmax(&prefill[0].1);
        let mut pos = prompt.len() as i32;
        let t0 = Instant::now();
        for _ in 0..n_gen {
            let batch = vec![MultiSeqEntry {
                token: tok, pos, seq_id: 0, request_logits: true,
            }];
            let out = llama.forward_multi_seq(&batch).expect("baseline gen");
            tok = argmax(&out[0].1);
            pos += 1;
        }
        let elapsed = t0.elapsed().as_secs_f32().max(1e-6);
        let tps = n_gen as f32 / elapsed;
        let _ = llama.kv_cache_seq_rm_for(0);
        (tps, vocab)
    };
    eprintln!("[j7-concurrent-four] baseline single-seq = {:.2} tok/s", baseline_tokps);

    // ---- Mixed : 4 seqs, one batch per step ----
    let mut llama = LlamaForward::new_multi_seq(
        &model_path, n_gpu_layers, n_ctx, 0, None, None, true, NUM_SEQS as u32,
    ).expect("mixed new_multi_seq");
    assert_eq!(llama.vocab_size(), vocab_size, "vocab size mismatch");

    // 4 distinct prompts — just unique seed IDs so greedy argmax
    // produces divergent continuations. Not required to be English.
    let prompts: [Vec<u32>; NUM_SEQS] = [
        vec![1, 450, 8448, 374, 6548],  // seq 0
        vec![1, 512, 6534, 279, 2913],  // seq 1
        vec![1, 101, 202, 303, 404],    // seq 2
        vec![1, 707, 808, 909, 111],    // seq 3
    ];

    // Prefill all four in ONE batch.
    let mut entries: Vec<MultiSeqEntry> = Vec::new();
    for (seq, prompt) in prompts.iter().enumerate() {
        for (i, &t) in prompt.iter().enumerate() {
            entries.push(MultiSeqEntry {
                token: t, pos: i as i32, seq_id: seq as i32,
                request_logits: i == prompt.len() - 1,
            });
        }
    }
    let prefill = llama.forward_multi_seq(&entries).expect("4-seq prefill");
    assert_eq!(prefill.len(), NUM_SEQS, "prefill must return 4 logit vectors");

    // Per-seq state bookkeeping.
    let mut toks: [u32; NUM_SEQS] = [0; NUM_SEQS];
    let mut pos: [i32; NUM_SEQS] = [0; NUM_SEQS];
    let mut streams: Vec<Vec<u32>> = Vec::with_capacity(NUM_SEQS);
    for seq in 0..NUM_SEQS {
        let logits = prefill.iter()
            .find(|(s, _)| *s == seq as i32)
            .unwrap_or_else(|| panic!("prefill missing logits for seq {}", seq));
        toks[seq] = argmax(&logits.1);
        pos[seq] = prompts[seq].len() as i32;
        streams.push(vec![toks[seq]]);
    }

    // Generate loop — 4 tokens per step, one step per batch.
    let t0 = Instant::now();
    for step in 0..n_gen {
        let mut batch: Vec<MultiSeqEntry> = Vec::with_capacity(NUM_SEQS);
        for seq in 0..NUM_SEQS {
            batch.push(MultiSeqEntry {
                token: toks[seq],
                pos: pos[seq],
                seq_id: seq as i32,
                request_logits: true,
            });
        }
        let logits = llama.forward_multi_seq(&batch)
            .unwrap_or_else(|e| panic!("step {} forward_multi_seq: {}", step, e));
        assert_eq!(logits.len(), NUM_SEQS, "step {}: expected {} logits", step, NUM_SEQS);
        for seq in 0..NUM_SEQS {
            let l = logits.iter()
                .find(|(s, _)| *s == seq as i32)
                .unwrap_or_else(|| panic!("step {}: no logits for seq {}", step, seq));
            toks[seq] = argmax(&l.1);
            streams[seq].push(toks[seq]);
            pos[seq] += 1;
        }
    }
    let elapsed = t0.elapsed().as_secs_f32().max(1e-6);
    let total_tokens = (NUM_SEQS * n_gen) as f32;
    let aggregate_tokps = total_tokens / elapsed;
    let ratio = aggregate_tokps / baseline_tokps;
    eprintln!(
        "[j7-concurrent-four] mixed {}-seq = {:.2} tok/s aggregate (baseline {:.2}) → {:.2}× ratio",
        NUM_SEQS, aggregate_tokps, baseline_tokps, ratio,
    );

    // Cleanup.
    for seq in 0..NUM_SEQS as i32 {
        let _ = llama.kv_cache_seq_rm_for(seq);
    }

    // --- Always: completion + divergence ---
    for seq in 0..NUM_SEQS {
        assert_eq!(
            streams[seq].len(), n_gen + 1,
            "seq {} missing tokens (deadlock or early exit)", seq,
        );
    }
    // At least two of the 4 streams must differ somewhere — identical
    // greedy streams across all 4 seqs would strongly suggest
    // cross-seq contamination.
    let distinct_pairs = (0..NUM_SEQS).any(|i| {
        (i+1..NUM_SEQS).any(|j| streams[i] != streams[j])
    });
    assert!(
        distinct_pairs,
        "all 4 streams produced identical output — cross-contamination suspected",
    );

    // --- Throughput: relaxable ---
    if !relax {
        assert!(
            ratio >= 3.0,
            "4-seq aggregate / baseline = {:.2} (<3.0). Plan §7 target 3.0× on GPU. \
             Set CHIMERE_J7_RELAX=1 if running on slow CPU where scheduler overhead \
             dominates.",
            ratio,
        );
    } else {
        eprintln!(
            "[j7-concurrent-four] RELAX mode: skipping ≥3× assertion (observed {:.2}×). \
             Only deadlock + divergence checks are enforced.",
            ratio,
        );
    }
}
