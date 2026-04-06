//! End-to-end test: load Nemotron-H via chimere's libllama FFI and generate.
//!
//! Proves that chimere-server's `LlamaForward` (the foundation of
//! `GenericModel`) is fully wired against the new `ik_llama` Mamba-2 +
//! `nemotron_h_moe` backport. No `Qwen35Model`, no `cudarc_shell`, no
//! Qwen-specific assumptions on the path.
//!
//! Usage:
//!     CHIMERE_MODEL=.../Nemotron-3-Nano-30B-A3B-Q4_0.gguf \
//!     CHIMERE_TOKENIZER=.../Nemotron-3-Nano-30B-A3B-GGUF/tokenizer.json \
//!     CHIMERE_NCMOE=4 \
//!     CHIMERE_KV_MAX_SEQ=4096 \
//!     CHIMERE_LLAMA_BACKEND=1 \
//!     cargo run --release --bin test-nemotron

use std::time::Instant;

fn main() {
    eprintln!("=== chimere-server: GenericModel Nemotron-H smoke test ===\n");
    std::env::set_var("CHIMERE_LLAMA_BACKEND", "1");

    let model_path = std::env::var("CHIMERE_MODEL")
        .expect("CHIMERE_MODEL must be set to a Nemotron GGUF path");
    eprintln!("model: {}", model_path);

    eprintln!("\nLoading via libllama FFI (this exercises the new ik_llama Mamba-2 backport)...");
    let t_load = Instant::now();
    let mut llama = chimere_deltanet::llama_backend::from_env()
        .expect("llama_backend::from_env() failed — is the new libllama.so on LD_LIBRARY_PATH?");
    let load_ms = t_load.elapsed().as_millis();
    eprintln!(
        "loaded in {} ms — n_vocab={} n_embd={} n_layer={}",
        load_ms,
        llama.n_vocab(),
        llama.n_embd(),
        llama.n_layer(),
    );

    let prompt = std::env::var("CHIMERE_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".to_string());
    eprintln!("\nprompt: {:?}", prompt);

    let tokens_i32 = llama
        .tokenize(&prompt, true, false)
        .expect("tokenize failed");
    eprintln!("prompt tokens ({}): {:?}", tokens_i32.len(), tokens_i32);
    let prompt_tokens: Vec<u32> = tokens_i32.iter().map(|&t| t as u32).collect();

    eprintln!("\nprefill...");
    let t_prefill = Instant::now();
    let logits = llama
        .forward_prefill(&prompt_tokens)
        .expect("forward_prefill failed");
    eprintln!("prefill: {} ms, last-token logits len = {}", t_prefill.elapsed().as_millis(), logits.len());

    // Greedy sample loop — 20 tokens, picks argmax of logits each step.
    let n_predict: usize = std::env::var("CHIMERE_N_PREDICT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    eprintln!("\ngenerating {} tokens (greedy)...\n", n_predict);

    let mut all_tokens: Vec<i32> = tokens_i32.clone();
    let mut current_logits = logits;

    let t_gen = Instant::now();
    for _ in 0..n_predict {
        // argmax
        let (next_id, _) = current_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .expect("empty logits");
        let next_token = next_id as u32;
        all_tokens.push(next_token as i32);

        // Stop on EOS-ish tokens (Nemotron-H uses <|endoftext|> = 2 typically;
        // we don't hard-code, just stop after n_predict).
        current_logits = llama
            .forward_token(next_token)
            .expect("forward_token failed");
    }
    let gen_ms = t_gen.elapsed().as_millis();
    let tps = (n_predict as f64) * 1000.0 / (gen_ms as f64);

    let text = llama
        .detokenize(&all_tokens, false)
        .expect("detokenize failed");

    eprintln!("=== generated ({:.1} tok/s, {} ms) ===", tps, gen_ms);
    println!("{}", text);
    eprintln!("\n=== OK ===");
}
