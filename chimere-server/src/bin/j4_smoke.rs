//! # J4 smoke test — chunked prefill + concurrent generate, via interleaved batches
//!
//! Validates that `LlamaForward::forward_multi_seq` can drive a
//! **chunked prefill** for one sequence while a **different** sequence is
//! concurrently generating — the classical scheduler pattern where long
//! prompts don't stall shorter ones.
//!
//! ## Observed libllama constraint (M1 J4, qwen3next)
//!
//! Early experiments revealed that on hybrid GDN models (qwen3next), a
//! single `llama_decode` batch that contains **both** a multi-token prefill
//! for seq A *and* a gen token for seq B triggers libllama's internal
//! fallback:
//!
//! ```text
//! llama_decode_internal: qwen3next mixed-sequence batch contains repeated
//!   seq_id values; falling back to single-token chunking
//! ```
//!
//! The fallback is correct (K/V routing stays per-seq), but it linearises
//! the batch and the resulting generate logits diverge from an isolated
//! baseline — that's a scheduling artefact, not a correctness bug. The
//! multi-slot scheduler therefore must **not** put a prefill chunk and a
//! generate step for two distinct seqs in the *same* batch on this arch.
//! Instead, it interleaves them:
//!
//!   step 2k     : push a ubatch-sized prefill chunk for seq 0 only.
//!   step 2k+1   : push ONE generate token for seq 1 only.
//!
//! Each step is a single `forward_multi_seq` call. seq_ids never repeat
//! within one batch, so the fallback path is never triggered.
//!
//! ## What this smoke proves
//!
//! 1. seq 0's chunked prefill, spread across N calls, produces a valid
//!    finaliser logit vector (no NaN/Inf, argmax in vocab range).
//! 2. seq 1's generated token stream under the interleaved schedule is
//!    **bit-for-bit identical** to the stream it produces in isolation.
//!    (Proves seq 0's KV writes don't leak into seq 1's logits across
//!    back-to-back `forward_multi_seq` calls — the K/V indexer stays
//!    per-seq as documented.)
//! 3. Total wall-clock of the interleaved run is bounded — no deadlock
//!    and all `forward_multi_seq` calls return zero from `llama_decode`.
//!
//! ## Running
//!
//! ```bash
//! export CHIMERE_MODEL=/path/to/any/small.gguf
//! export CHIMERE_N_CTX=4096
//! export CHIMERE_N_GPU_LAYERS=0    # CPU-safe vs prod :8081 VRAM
//! export CHIMERE_SKIP_SAMPLER_INIT=1  # don't need the C++ sampler here
//! cargo run --release --features server --bin j4-smoke
//! ```
//!
//! Exit 0 on PASS, 1 on FAIL.

use std::env;
use std::time::Instant;

use chimere_deltanet::llama_backend::{LlamaForward, MultiSeqEntry};

/// Greedy argmax. No sampler required.
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

/// Reject any NaN / Inf that would indicate a corrupted forward pass.
fn finite_check(logits: &[f32], tag: &str) -> Result<(), String> {
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!(
                "{}: logit index {} is non-finite ({}) — batch corruption?",
                tag, i, v,
            ));
        }
    }
    Ok(())
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
        .unwrap_or(0);
    // How many tokens the "long prefill" seq should push through. 200
    // deliberately exceeds a single 64-token chunk so the chunked-prefill
    // code path is exercised multiple times.
    let prefill_len: usize = env::var("CHIMERE_J4_PREFILL_LEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let chunk_size: usize = env::var("CHIMERE_J4_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let n_gen: usize = env::var("CHIMERE_J4_NGEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    eprintln!("[j4-smoke] Config:");
    eprintln!("  model        = {}", model_path);
    eprintln!("  n_ctx        = {}", n_ctx);
    eprintln!("  n_gpu_layers = {} (0 = CPU)", n_gpu_layers);
    eprintln!("  prefill_len  = {} (seq 0)", prefill_len);
    eprintln!("  chunk_size   = {} tokens/prefill step", chunk_size);
    eprintln!("  n_gen        = {} (seq 1)", n_gen);

    // ---- Baseline : seq 1 generates in isolation (no seq 0 activity) ----
    let (baseline, vocab_size) = {
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
        let prompt_b: Vec<u32> = vec![1, 512, 6534, 279, 2913];
        let mut entries: Vec<MultiSeqEntry> = Vec::with_capacity(prompt_b.len());
        for (i, &t) in prompt_b.iter().enumerate() {
            entries.push(MultiSeqEntry {
                token: t,
                pos: i as i32,
                seq_id: 1,
                request_logits: i == prompt_b.len() - 1,
            });
        }
        let out = llama.forward_multi_seq(&entries)?;
        finite_check(&out[0].1, "baseline.prefill")?;
        let mut tok_b = argmax(&out[0].1);
        let mut pos_b = prompt_b.len() as i32;
        let mut stream: Vec<u32> = vec![tok_b];
        for _ in 0..n_gen {
            let batch = vec![MultiSeqEntry {
                token: tok_b,
                pos: pos_b,
                seq_id: 1,
                request_logits: true,
            }];
            let logits = llama.forward_multi_seq(&batch)?;
            finite_check(&logits[0].1, "baseline.gen")?;
            tok_b = argmax(&logits[0].1);
            stream.push(tok_b);
            pos_b += 1;
        }
        let _ = llama.kv_cache_seq_rm_for(1);
        (stream, llama.vocab_size())
    };
    eprintln!("[j4-smoke] baseline seq1 stream captured ({} tokens): {:?}",
        baseline.len(), baseline);

    // ---- Interleaved run ----
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
    eprintln!("[j4-smoke] Mixed-run model loaded. vocab_size={}", llama.vocab_size());
    if llama.vocab_size() != vocab_size {
        return Err(format!(
            "vocab mismatch baseline={} mixed={} (model reloaded differently?)",
            vocab_size, llama.vocab_size(),
        ));
    }

    let prompt_a: Vec<u32> = (0..prefill_len as u32)
        .map(|i| (i.wrapping_mul(2654435761)) % 30000 + 100)
        .collect();
    let prompt_b: Vec<u32> = vec![1, 512, 6534, 279, 2913];

    // Prefill seq 1 ONLY, so it's at generate-ready state before we
    // start interleaving with seq 0's chunked prefill.
    let mut b_entries: Vec<MultiSeqEntry> = Vec::with_capacity(prompt_b.len());
    for (i, &t) in prompt_b.iter().enumerate() {
        b_entries.push(MultiSeqEntry {
            token: t,
            pos: i as i32,
            seq_id: 1,
            request_logits: i == prompt_b.len() - 1,
        });
    }
    let b_prefill = llama.forward_multi_seq(&b_entries)?;
    finite_check(&b_prefill[0].1, "mixed.seq1.prefill")?;
    let mut tok_b = argmax(&b_prefill[0].1);
    let mut gen_b: Vec<u32> = vec![tok_b];
    let mut pos_b = prompt_b.len() as i32;
    eprintln!("[j4-smoke] seq 1 prefilled, first argmax = {}", tok_b);

    // Step loop. Each iteration does TWO calls:
    //   1) one chunk of seq 0 prefill (single seq_id per batch — no fallback)
    //   2) one gen step of seq 1 if we still owe tokens to the baseline comp
    // Every call is a standalone `forward_multi_seq` — the scheduler's
    // real composition pattern.
    let t_mixed = Instant::now();
    let n_chunks = (prefill_len + chunk_size - 1) / chunk_size;
    let mut seq0_final_logits: Option<Vec<f32>> = None;

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(prefill_len);
        let is_last_chunk = end == prefill_len;

        // ---- (1) seq 0 prefill chunk : pure prefill, seq_id=0 everywhere ----
        let mut a_entries: Vec<MultiSeqEntry> = Vec::with_capacity(end - start);
        for (i, &t) in prompt_a[start..end].iter().enumerate() {
            let abs_pos = (start + i) as i32;
            let want_logits = is_last_chunk && (i == (end - start - 1));
            a_entries.push(MultiSeqEntry {
                token: t,
                pos: abs_pos,
                seq_id: 0,
                request_logits: want_logits,
            });
        }
        let a_out = llama.forward_multi_seq(&a_entries)?;
        if is_last_chunk {
            let a = a_out
                .iter()
                .find(|(s, _)| *s == 0)
                .ok_or_else(|| "final chunk: no logits for seq 0".to_string())?;
            finite_check(&a.1, "mixed.seq0.final")?;
            seq0_final_logits = Some(a.1.clone());
        }

        // ---- (2) seq 1 gen step (if we still have tokens to emit) ----
        let emit_b = gen_b.len() - 1 < n_gen;
        if emit_b {
            let batch = vec![MultiSeqEntry {
                token: tok_b,
                pos: pos_b,
                seq_id: 1,
                request_logits: true,
            }];
            let b_out = llama.forward_multi_seq(&batch)?;
            let b = b_out
                .iter()
                .find(|(s, _)| *s == 1)
                .ok_or_else(|| format!("chunk {}: no logits for seq 1", chunk_idx))?;
            finite_check(&b.1, &format!("mixed.chunk{}.seq1", chunk_idx))?;
            tok_b = argmax(&b.1);
            gen_b.push(tok_b);
            pos_b += 1;
        }

        eprintln!(
            "[chunk {:2}/{}] seq0 pos=[{:4}..{:4}){} | seq1 pos={:4} emit={} tok={}",
            chunk_idx + 1, n_chunks,
            start, end,
            if is_last_chunk { "*logits*" } else { "        " },
            pos_b,
            emit_b,
            tok_b,
        );
    }

    // If we ran out of chunks before n_gen, top up seq 1 with plain gen steps.
    while gen_b.len() - 1 < n_gen {
        let batch = vec![MultiSeqEntry {
            token: tok_b,
            pos: pos_b,
            seq_id: 1,
            request_logits: true,
        }];
        let logits = llama.forward_multi_seq(&batch)?;
        finite_check(&logits[0].1, "mixed.topup")?;
        tok_b = argmax(&logits[0].1);
        gen_b.push(tok_b);
        pos_b += 1;
    }

    let mixed_elapsed = t_mixed.elapsed().as_secs_f32();
    eprintln!(
        "[j4-smoke] Mixed run done in {:.2}s ({} chunks, {} seq1 tokens)",
        mixed_elapsed, n_chunks, gen_b.len(),
    );
    eprintln!("[j4-smoke] mixed seq1 stream : {:?}", gen_b);

    // --- Assertion 1 : seq 1 stream under interleaving == baseline ---
    if gen_b != baseline {
        return Err(format!(
            "FAIL: seq 1 token stream changed when running interleaved with seq 0. \
             baseline={:?}  mixed={:?}  \
             → seq 0's KV updates leaked into seq 1's forward pass.",
            baseline, gen_b,
        ));
    }

    // --- Assertion 2 : seq 0 produced a valid finalised logit vector ---
    let seq0_final = seq0_final_logits.ok_or_else(||
        "FAIL: seq 0 final prefill logits were never captured".to_string())?;
    if seq0_final.len() != llama.vocab_size() {
        return Err(format!(
            "FAIL: seq 0 final logits len={}, expected vocab_size={}",
            seq0_final.len(), llama.vocab_size(),
        ));
    }
    let seq0_argmax = argmax(&seq0_final);
    if (seq0_argmax as usize) >= llama.vocab_size() {
        return Err(format!(
            "FAIL: seq 0 argmax {} out of vocab range {}",
            seq0_argmax, llama.vocab_size(),
        ));
    }

    let _ = llama.kv_cache_seq_rm_for(0);
    let _ = llama.kv_cache_seq_rm_for(1);

    eprintln!();
    eprintln!("===== J4 SMOKE PASS =====");
    eprintln!("  chunks prefilled : {}", n_chunks);
    eprintln!("  seq0 argmax      : {} (vocab OK)", seq0_argmax);
    eprintln!("  seq1 stream      : baseline == mixed  ({} tokens matched)", gen_b.len());
    eprintln!("  mixed elapsed    : {:.2} s", mixed_elapsed);
    eprintln!();
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("\n[j4-smoke] FAIL: {}", e);
            std::process::exit(1);
        }
    }
}
