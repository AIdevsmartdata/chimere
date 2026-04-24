//! # J3 smoke test — real FFI multi-seq decode, 2 concurrent sequences
//!
//! Proves the core M1 J3 invariant: a single `llama_decode` call can carry
//! tokens from **multiple distinct `seq_id`s** without cross-contamination.
//! Two short prompts are prefilled on seq_id 0 and 1, then each step
//! generates one token per seq in a single multi-seq batch. The smoke test
//! asserts that the two generated token streams diverge (they should, since
//! the prompts are different).
//!
//! This binary does NOT depend on the scheduler, admission queue, HTTP
//! handlers, or sampler — it drills straight to `LlamaForward::forward_multi_seq`
//! so a failure points at the FFI layer, not at plumbing above it.
//!
//! ## Running
//!
//! Requires a model that fits in VRAM (or CPU memory with `n_gpu_layers=0`).
//! Prod chimere-server holds ~14 GB of VRAM on :8081 — use a different
//! model here or wait until prod is idle.
//!
//! ```bash
//! export CHIMERE_MODEL=/path/to/any/model.gguf
//! export CHIMERE_N_CTX=4096
//! export CHIMERE_N_GPU_LAYERS=0   # CPU-only (safe if prod holds VRAM)
//! cargo run --release --bin j3-smoke
//! ```
//!
//! Exit 0 = PASS (two seqs produced divergent token streams).
//! Exit 1 = FAIL (FFI error, or both seqs produced identical output,
//!                which would indicate cross-contamination).

use std::env;
use std::time::Instant;

use chimere_deltanet::llama_backend::{LlamaForward, MultiSeqEntry};

/// Greedy argmax over a logit vector — cheap, no sampler needed for smoke.
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

fn run() -> Result<(), String> {
    let model_path = env::var("CHIMERE_MODEL")
        .map_err(|_| "CHIMERE_MODEL env var required (path to a .gguf file)".to_string())?;
    let n_ctx: u32 = env::var("CHIMERE_N_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let n_gpu_layers: i32 = env::var("CHIMERE_N_GPU_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);  // default CPU-only so prod :8081 VRAM isn't disturbed
    let n_gen: usize = env::var("CHIMERE_J3_NGEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    eprintln!("[j3-smoke] Config:");
    eprintln!("  model        = {}", model_path);
    eprintln!("  n_ctx        = {}", n_ctx);
    eprintln!("  n_gpu_layers = {} (0 = CPU only)", n_gpu_layers);
    eprintln!("  n_gen        = {} tokens/seq", n_gen);
    eprintln!("  n_seq_max    = 2 (this is the multi-seq test)");

    let t_load = Instant::now();
    let mut llama = LlamaForward::new_multi_seq(
        &model_path,
        n_gpu_layers,
        n_ctx,
        0,         // ncmoe = 0 (no MoE CPU offload in smoke)
        None,
        None,
        true,      // flash_attn
        2,         // n_seq_max — the whole point of this binary
    )?;
    eprintln!("[j3-smoke] Model loaded in {:.2}s. vocab_size={}", t_load.elapsed().as_secs_f32(), llama.vocab_size());

    // Two distinct byte-level token sequences. These aren't English text —
    // the smoke test does NOT depend on the tokenizer being sane. We just
    // need two *different* prompt token-IDs so the model produces divergent
    // continuations. BOS (id=1) + a handful of mid-vocab IDs.
    //
    // If a specific model has BOS=151643 (Qwen-style), the first entry here
    // may not be BOS — that's fine, we're testing seq-isolation, not
    // linguistic plausibility.
    let prompt_a: Vec<u32> = vec![1, 450, 8448, 374, 6548];   // "seed A"
    let prompt_b: Vec<u32> = vec![1, 512, 6534, 279, 2913];   // "seed B"

    // ---- Prefill both seqs in ONE multi-seq batch ----
    let mut entries: Vec<MultiSeqEntry> = Vec::with_capacity(prompt_a.len() + prompt_b.len());
    for (i, &t) in prompt_a.iter().enumerate() {
        entries.push(MultiSeqEntry {
            token: t,
            pos: i as i32,
            seq_id: 0,
            request_logits: i == prompt_a.len() - 1,
        });
    }
    for (i, &t) in prompt_b.iter().enumerate() {
        entries.push(MultiSeqEntry {
            token: t,
            pos: i as i32,
            seq_id: 1,
            request_logits: i == prompt_b.len() - 1,
        });
    }

    let t_prefill = Instant::now();
    let prefill_logits = llama.forward_multi_seq(&entries)?;
    eprintln!(
        "[j3-smoke] Prefill batch (n_tokens={}) ran in {:.2}s; got {} logit vectors",
        entries.len(),
        t_prefill.elapsed().as_secs_f32(),
        prefill_logits.len(),
    );
    if prefill_logits.len() != 2 {
        return Err(format!(
            "prefill returned {} logit vectors, expected 2 (one per seq)",
            prefill_logits.len(),
        ));
    }

    let seq0_logits = prefill_logits.iter().find(|(s, _)| *s == 0)
        .ok_or("no logits returned for seq 0")?;
    let seq1_logits = prefill_logits.iter().find(|(s, _)| *s == 1)
        .ok_or("no logits returned for seq 1")?;

    let mut tok_a = argmax(&seq0_logits.1);
    let mut tok_b = argmax(&seq1_logits.1);
    let mut gen_a: Vec<u32> = vec![tok_a];
    let mut gen_b: Vec<u32> = vec![tok_b];
    let mut pos_a = prompt_a.len() as i32;
    let mut pos_b = prompt_b.len() as i32;

    eprintln!("[j3-smoke] Prefill argmax: seq0={}, seq1={}", tok_a, tok_b);

    // ---- Generate N more tokens, one per seq per step, single batch per step ----
    let t_gen = Instant::now();
    for step in 0..n_gen {
        let batch = vec![
            MultiSeqEntry {
                token: tok_a,
                pos: pos_a,
                seq_id: 0,
                request_logits: true,
            },
            MultiSeqEntry {
                token: tok_b,
                pos: pos_b,
                seq_id: 1,
                request_logits: true,
            },
        ];

        let logits = llama.forward_multi_seq(&batch)?;
        if logits.len() != 2 {
            return Err(format!(
                "step {}: got {} logit vectors, expected 2",
                step, logits.len(),
            ));
        }

        let s0 = logits.iter().find(|(s, _)| *s == 0)
            .ok_or_else(|| format!("step {}: no logits for seq 0", step))?;
        let s1 = logits.iter().find(|(s, _)| *s == 1)
            .ok_or_else(|| format!("step {}: no logits for seq 1", step))?;

        tok_a = argmax(&s0.1);
        tok_b = argmax(&s1.1);
        gen_a.push(tok_a);
        gen_b.push(tok_b);
        pos_a += 1;
        pos_b += 1;
        eprintln!("[step {:2}] seq0 → {:6}   seq1 → {:6}", step, tok_a, tok_b);
    }
    let gen_elapsed = t_gen.elapsed().as_secs_f32();
    let aggregate_tokps = (2.0 * n_gen as f32) / gen_elapsed;
    eprintln!(
        "[j3-smoke] Generated {} tokens/seq in {:.2}s  ({:.1} tok/s aggregate)",
        n_gen, gen_elapsed, aggregate_tokps,
    );

    // ---- Assert divergence ----
    let diverged = gen_a.iter().zip(gen_b.iter()).any(|(a, b)| a != b);
    if !diverged {
        return Err(format!(
            "FAIL: seq 0 and seq 1 produced identical outputs (possible cross-contamination). \
             seq 0={:?}  seq 1={:?}",
            gen_a, gen_b,
        ));
    }

    // ---- Free both seq_ids ----
    let removed_0 = llama.kv_cache_seq_rm_for(0);
    let removed_1 = llama.kv_cache_seq_rm_for(1);
    eprintln!(
        "[j3-smoke] KV cache freed: seq0={} seq1={}",
        removed_0, removed_1,
    );

    eprintln!();
    eprintln!("===== J3 SMOKE PASS =====");
    eprintln!("  seq 0 generated : {:?}", gen_a);
    eprintln!("  seq 1 generated : {:?}", gen_b);
    eprintln!("  divergence at   : {} steps",
        gen_a.iter().zip(gen_b.iter()).filter(|(a, b)| a != b).count());
    eprintln!("  aggregate tok/s : {:.1}", aggregate_tokps);
    eprintln!();
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("\n[j3-smoke] FAIL: {}", e);
            std::process::exit(1);
        }
    }
}
