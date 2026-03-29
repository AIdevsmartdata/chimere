//! MTP cost benchmark — measures single vs batch decode cost to determine
//! if MTP bypass speculation is viable for Qwen3.5-35B-A3B MoE.
//!
//! Usage: CHIMERE_LLAMA_BACKEND=1 CHIMERE_MODEL=...MTP.gguf cargo run --release --bin bench-mtp
//!
//! Measures:
//! 1. Single token decode cost (baseline)
//! 2. MTP draft decode cost (1 layer)
//! 3. Batch-of-2 decode cost
//! 4. Batch-of-4 decode cost
//! 5. MTP acceptance rate on sample text

use std::time::Instant;

fn main() {
    eprintln!("=== MTP Cost Benchmark ===\n");

    // Force llama backend
    std::env::set_var("CHIMERE_LLAMA_BACKEND", "1");

    // Use MTP model if available, fall back to custom-mix
    let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
    let mtp_model = format!(
        "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-MTP.gguf",
        home
    );
    let default_model = format!(
        "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
        home
    );

    let model_path = if std::path::Path::new(&mtp_model).exists() {
        eprintln!("Using MTP model: {}", mtp_model);
        mtp_model
    } else {
        eprintln!("MTP model not found, using default: {}", default_model);
        default_model
    };

    std::env::set_var("CHIMERE_MODEL", &model_path);
    // Use smaller context for benchmark (saves VRAM)
    std::env::set_var("CHIMERE_KV_MAX_SEQ", "8192");

    eprintln!("\nLoading model...");
    let mut llama = chimere_deltanet::llama_backend::from_env()
        .expect("Failed to create llama backend");

    eprintln!("MTP available: {}", llama.has_mtp());
    eprintln!("n_embd: {}", llama.n_embd());
    eprintln!("n_vocab: {}\n", llama.n_vocab());

    // Warmup: prefill a short prompt
    let prompt_tokens: Vec<u32> = vec![
        248045, // <|im_start|>
        882,    // user
        198,    // \n
        3838,   // What
        374,    // is
        279,    // the
        7290,   // capital
        315,    // of
        9822,   // France
        30,     // ?
        198,    // \n
        248046, // <|im_end|>
        198,    // \n
        248045, // <|im_start|>
        77091,  // assistant
        198,    // \n
    ];

    eprintln!("Prefilling {} tokens...", prompt_tokens.len());
    let t0 = Instant::now();
    llama.forward_prefill(&prompt_tokens).expect("prefill failed");
    let prefill_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("Prefill: {:.1}ms ({:.0} tok/s)\n", prefill_ms,
        prompt_tokens.len() as f64 / (prefill_ms / 1000.0));

    // === Benchmark 1: Single token decode ===
    eprintln!("--- Benchmark 1: Single token decode ---");
    let n_warmup = 5;
    let n_bench = 30;

    // Warmup
    for _ in 0..n_warmup {
        let t = llama.sample_token_fast().unwrap_or(0);
        llama.forward_token_no_logits(t).expect("decode failed");
    }

    llama.reset_timings();
    let mut single_times = Vec::with_capacity(n_bench);
    for _ in 0..n_bench {
        let t = llama.sample_token_fast().unwrap_or(0);
        let t0 = Instant::now();
        llama.forward_token_no_logits(t).expect("decode failed");
        single_times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let single_avg = single_times.iter().sum::<f64>() / single_times.len() as f64;
    let single_min = single_times.iter().cloned().fold(f64::MAX, f64::min);
    let single_max = single_times.iter().cloned().fold(f64::MIN, f64::max);
    eprintln!("  avg: {:.2}ms  min: {:.2}ms  max: {:.2}ms  ({:.1} tok/s)",
        single_avg, single_min, single_max, 1000.0 / single_avg);

    // === Benchmark 2: SKIPPED (MTP decode crash in ik_llama — KV cache issue for layer 41) ===
    eprintln!("\n--- Benchmark 2: MTP decode --- SKIPPED (crash in ik_llama MTP graph)");

    // === Benchmark 3: Batch-of-2 decode ===
    eprintln!("\n--- Benchmark 3: Batch-of-2 decode ---");
    let mut batch2_times = Vec::with_capacity(n_bench);
    for _ in 0..n_bench {
        let t1 = llama.sample_token_fast().unwrap_or(0);
        let t2 = t1.wrapping_add(1); // dummy second token

        let t0 = Instant::now();
        let result = llama.forward_batch_verify(&[t1, t2]);
        batch2_times.push(t0.elapsed().as_secs_f64() * 1000.0);

        match result {
            Ok(_logits) => {
                llama.accept_draft_tokens(2);
            }
            Err(e) => {
                eprintln!("  Batch-2 error: {}", e);
                break;
            }
        }
    }
    if !batch2_times.is_empty() {
        let batch2_avg = batch2_times.iter().sum::<f64>() / batch2_times.len() as f64;
        let batch2_min = batch2_times.iter().cloned().fold(f64::MAX, f64::min);
        let batch2_max = batch2_times.iter().cloned().fold(f64::MIN, f64::max);
        let ratio = batch2_avg / single_avg;
        eprintln!("  avg: {:.2}ms  min: {:.2}ms  max: {:.2}ms  (ratio: {:.2}x single)",
            batch2_avg, batch2_min, batch2_max, ratio);
    }

    // === Benchmark 4: Batch-of-4 decode ===
    eprintln!("\n--- Benchmark 4: Batch-of-4 decode ---");
    let mut batch4_times = Vec::with_capacity(n_bench);
    for _ in 0..n_bench {
        let t1 = llama.sample_token_fast().unwrap_or(0);
        let drafts = [t1, t1.wrapping_add(1), t1.wrapping_add(2), t1.wrapping_add(3)];

        let t0 = Instant::now();
        let result = llama.forward_batch_verify(&drafts);
        batch4_times.push(t0.elapsed().as_secs_f64() * 1000.0);

        match result {
            Ok(_) => {
                llama.accept_draft_tokens(4);
            }
            Err(e) => {
                eprintln!("  Batch-4 error: {}", e);
                break;
            }
        }
    }
    if !batch4_times.is_empty() {
        let batch4_avg = batch4_times.iter().sum::<f64>() / batch4_times.len() as f64;
        let batch4_min = batch4_times.iter().cloned().fold(f64::MAX, f64::min);
        let batch4_max = batch4_times.iter().cloned().fold(f64::MIN, f64::max);
        let ratio = batch4_avg / single_avg;
        eprintln!("  avg: {:.2}ms  min: {:.2}ms  max: {:.2}ms  (ratio: {:.2}x single)",
            batch4_avg, batch4_min, batch4_max, ratio);
    }

    // === Benchmark 5: MTP acceptance rate === SKIPPED (MTP decode crash)
    eprintln!("\n--- Benchmark 5: MTP acceptance rate --- SKIPPED (MTP decode crash)");

    // === Summary ===
    eprintln!("\n=== SUMMARY ===");
    if !single_times.is_empty() {
        let single_avg = single_times.iter().sum::<f64>() / single_times.len() as f64;
        eprintln!("Single decode:  {:.2}ms ({:.1} tok/s)", single_avg, 1000.0 / single_avg);

        if !batch2_times.is_empty() {
            let batch2_avg = batch2_times.iter().sum::<f64>() / batch2_times.len() as f64;
            let ratio = batch2_avg / single_avg;
            eprintln!("Batch-2 decode: {:.2}ms ({:.2}x single)", batch2_avg, ratio);

            // Viability analysis
            if ratio < 1.3 {
                eprintln!("\n>>> VIABLE: batch-2 < 1.3x → MTP bypass can give ~{:.0} tok/s",
                    1.5 * 1000.0 / (single_avg + batch2_avg / 2.0));
            } else if ratio < 1.5 {
                eprintln!("\n>>> MARGINAL: batch-2 = {:.2}x → ~{:.0} tok/s (marginal gain)",
                    ratio, 1.5 * 1000.0 / (single_avg + batch2_avg / 2.0));
            } else {
                eprintln!("\n>>> NOT VIABLE: batch-2 = {:.2}x → MTP bypass would be slower than baseline", ratio);
            }
        }
    }

    // Print llama.cpp timings
    eprintln!("\n--- llama.cpp timings ---");
    llama.print_timings();
}
