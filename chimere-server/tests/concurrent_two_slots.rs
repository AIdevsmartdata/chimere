//! # M1 J7 — concurrent two-slot throughput
//!
//! Integration test. Opens a `LlamaForward::new_multi_seq(n_seq_max=2)`
//! context and drives two sequences concurrently by calling
//! `forward_multi_seq` with a 2-entry batch per step (one entry per seq).
//! Records aggregate tok/s and compares it to an isolated single-seq
//! baseline on the same model.
//!
//! Gating (CPU-only, no VRAM stealing from prod)
//! ---------------------------------------------
//! The test is SKIPPED (prints a warning and returns Ok) if `CHIMERE_MODEL`
//! is not set — so `cargo test --test concurrent_two_slots` runs green on
//! a box with no GGUF, and only activates when a path is provided. This
//! mirrors the pattern used by j3-smoke / j4-smoke / j5-smoke.
//!
//! On an unloaded laptop-class CPU with a tiny test GGUF, we expect:
//!   - baseline   ≈ 10–30 tok/s (single seq, tight loop)
//!   - 2-slot mix ≈ 15–45 tok/s (≥ 1.5× baseline)
//!
//! The 1.5× target comes from plan-M1-multislot-2026-04-24.md §7 KPI row
//! "aggregate tok/s, 2 slots = 170 vs 100 baseline = 1.7×". We relax to
//! 1.5× for CPU-only runs where scheduler overhead is a larger fraction
//! of per-step wall time than on GPU.
//!
//! If the CPU is extremely slow (e.g. CI container throttled), the test
//! falls back to "no deadlock, all tokens produced, streams differ". This
//! fallback is OPT-IN via `CHIMERE_J7_RELAX=1` — without it the ratio
//! assertion stays hard.
//!
//! Running
//! -------
//! ```bash
//! export CHIMERE_MODEL=/path/to/tiny.gguf
//! export CHIMERE_N_GPU_LAYERS=0
//! export CHIMERE_N_CTX=2048
//! cargo test --release --features server --test concurrent_two_slots -- --nocapture
//! ```

use std::env;
use std::time::Instant;

use chimere_deltanet::llama_backend::{LlamaForward, MultiSeqEntry};

/// Greedy argmax — no sampler needed. Matches j3-smoke.
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

/// If `CHIMERE_MODEL` is unset, skip. Returns the path otherwise.
fn model_path_or_skip() -> Option<String> {
    match env::var("CHIMERE_MODEL") {
        Ok(p) => Some(p),
        Err(_) => {
            eprintln!(
                "[j7-concurrent-two] CHIMERE_MODEL unset — skipping (set it to enable this integration test)."
            );
            None
        }
    }
}

#[test]
fn concurrent_two_slots_aggregate_throughput() {
    let model_path = match model_path_or_skip() {
        Some(p) => p,
        None => return,
    };
    let n_ctx: u32 = env::var("CHIMERE_N_CTX")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    // CPU by default, as noted in the module docstring.
    let n_gpu_layers: i32 = env::var("CHIMERE_N_GPU_LAYERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    let n_gen: usize = env::var("CHIMERE_J7_NGEN")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let relax: bool = env::var("CHIMERE_J7_RELAX")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Skip if the user forgot to point at CPU inference.
    env::set_var("CHIMERE_SKIP_SAMPLER_INIT", "1");

    eprintln!(
        "[j7-concurrent-two] model={}, n_ctx={}, n_gpu_layers={}, n_gen={}, relax={}",
        model_path, n_ctx, n_gpu_layers, n_gen, relax,
    );

    // -------- Baseline: one seq in isolation --------
    let (baseline_tokps, vocab_size) = {
        let mut llama = LlamaForward::new_multi_seq(
            &model_path, n_gpu_layers, n_ctx, 0, None, None, true, 2,
        ).expect("baseline LlamaForward::new_multi_seq");
        let vocab = llama.vocab_size();
        let prompt: Vec<u32> = vec![1, 450, 8448, 374, 6548];
        let mut entries: Vec<MultiSeqEntry> = Vec::with_capacity(prompt.len());
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
            let logits = llama.forward_multi_seq(&batch).expect("baseline gen");
            tok = argmax(&logits[0].1);
            pos += 1;
        }
        let elapsed = t0.elapsed().as_secs_f32().max(1e-6);
        let tokps = n_gen as f32 / elapsed;
        let _ = llama.kv_cache_seq_rm_for(0);
        (tokps, vocab)
    };
    eprintln!("[j7-concurrent-two] baseline single-seq = {:.2} tok/s", baseline_tokps);

    // -------- Mixed run: two seqs, one batch per step --------
    let mut llama = LlamaForward::new_multi_seq(
        &model_path, n_gpu_layers, n_ctx, 0, None, None, true, 2,
    ).expect("mixed LlamaForward::new_multi_seq");
    assert_eq!(
        llama.vocab_size(), vocab_size,
        "vocab size mismatch between baseline and mixed run",
    );

    let prompt_a: Vec<u32> = vec![1, 450, 8448, 374, 6548];
    let prompt_b: Vec<u32> = vec![1, 512, 6534, 279, 2913];
    let mut entries: Vec<MultiSeqEntry> = Vec::with_capacity(prompt_a.len() + prompt_b.len());
    for (i, &t) in prompt_a.iter().enumerate() {
        entries.push(MultiSeqEntry {
            token: t, pos: i as i32, seq_id: 0,
            request_logits: i == prompt_a.len() - 1,
        });
    }
    for (i, &t) in prompt_b.iter().enumerate() {
        entries.push(MultiSeqEntry {
            token: t, pos: i as i32, seq_id: 1,
            request_logits: i == prompt_b.len() - 1,
        });
    }
    let prefill = llama.forward_multi_seq(&entries).expect("mixed prefill");
    assert_eq!(prefill.len(), 2, "prefill must return 2 logit vectors");
    let mut tok_a = argmax(&prefill.iter().find(|(s, _)| *s == 0).unwrap().1);
    let mut tok_b = argmax(&prefill.iter().find(|(s, _)| *s == 1).unwrap().1);
    let mut pos_a = prompt_a.len() as i32;
    let mut pos_b = prompt_b.len() as i32;
    let mut stream_a: Vec<u32> = vec![tok_a];
    let mut stream_b: Vec<u32> = vec![tok_b];

    let t0 = Instant::now();
    for step in 0..n_gen {
        let batch = vec![
            MultiSeqEntry { token: tok_a, pos: pos_a, seq_id: 0, request_logits: true },
            MultiSeqEntry { token: tok_b, pos: pos_b, seq_id: 1, request_logits: true },
        ];
        let logits = llama.forward_multi_seq(&batch)
            .unwrap_or_else(|e| panic!("step {} forward_multi_seq: {}", step, e));
        assert_eq!(logits.len(), 2, "step {}: expected 2 logits", step);
        tok_a = argmax(&logits.iter().find(|(s, _)| *s == 0).unwrap().1);
        tok_b = argmax(&logits.iter().find(|(s, _)| *s == 1).unwrap().1);
        stream_a.push(tok_a);
        stream_b.push(tok_b);
        pos_a += 1;
        pos_b += 1;
    }
    let elapsed = t0.elapsed().as_secs_f32().max(1e-6);
    let aggregate_tokps = (2 * n_gen) as f32 / elapsed;
    let ratio = aggregate_tokps / baseline_tokps;
    eprintln!(
        "[j7-concurrent-two] mixed 2-seq = {:.2} tok/s aggregate (baseline {:.2}) → {:.2}× ratio",
        aggregate_tokps, baseline_tokps, ratio,
    );

    let _ = llama.kv_cache_seq_rm_for(0);
    let _ = llama.kv_cache_seq_rm_for(1);

    // Always: completion + divergence (no deadlock, no contamination).
    assert_eq!(stream_a.len(), n_gen + 1, "seq 0 missing tokens");
    assert_eq!(stream_b.len(), n_gen + 1, "seq 1 missing tokens");
    let diverged = stream_a.iter().zip(stream_b.iter()).any(|(a, b)| a != b);
    assert!(diverged, "seq 0 == seq 1 — cross-contamination or deterministic-prompt collision");

    // Throughput assertion — hard unless CHIMERE_J7_RELAX=1.
    if !relax {
        assert!(
            ratio >= 1.5,
            "aggregate / baseline = {:.2} (<1.5). Set CHIMERE_J7_RELAX=1 on slow CPU boxes. \
             Plan §7 target = 1.7×; CPU allows 1.5× floor.",
            ratio,
        );
    } else {
        eprintln!(
            "[j7-concurrent-two] RELAX mode: throughput check skipped (ratio = {:.2}×).",
            ratio,
        );
    }
}
