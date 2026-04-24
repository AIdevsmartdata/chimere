//! # Regression — `finish_reason="stop"` on EOG tokens
//!
//! Guards the fix for the bug observed in the 2026-04-24 stress test where
//! the native multi-slot scheduler never detected `<|im_end|>` (248046) and
//! kept decoding until `max_tokens`, yielding `finish_reason="length"` on
//! every single response (or `null` on timeout).
//!
//! Strategy
//! --------
//! Drive a short prompt that the model answers in 1–3 tokens
//! (`Réponds par "OK" uniquement.`) through the native scheduler and
//! verify:
//!
//!   1. the terminal `StreamMsg::Done` carries `finish_reason == "stop"`;
//!   2. `generated_tokens` stays small (≤ 16). A regression that re-breaks
//!      EOS handling would see this balloon to 100+ (up to `max_tokens`).
//!
//! The test is gated on `CHIMERE_MODEL` (same pattern as
//! `tests/concurrent_two_slots.rs`) so a machine without a GGUF still runs
//! green on `cargo test`. Activate with:
//!
//! ```bash
//! export CHIMERE_MODEL=/path/to/chimere-v3-ramp.gguf
//! export CHIMERE_N_GPU_LAYERS=99
//! export CHIMERE_N_CTX=8192
//! cargo test --release --features server --test stop_token_imend -- --nocapture
//! ```

use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chimere_deltanet::llama_backend::LlamaForward;
use chimere_deltanet::slot_scheduler::{
    NativeScheduledRequest, NativeScheduler, SamplingParams as NativeSamplingParams,
    SchedulerConfig, StreamMsg,
};

/// If `CHIMERE_MODEL` is unset, skip. Returns the path otherwise.
fn model_path_or_skip() -> Option<String> {
    match env::var("CHIMERE_MODEL") {
        Ok(p) => Some(p),
        Err(_) => {
            eprintln!(
                "[stop_token_imend] CHIMERE_MODEL unset — skipping \
                 (set to a GGUF path to run this integration test)."
            );
            None
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn finish_reason_stop_on_im_end() {
    let model_path = match model_path_or_skip() {
        Some(p) => p,
        None => return,
    };
    let n_ctx: u32 = env::var("CHIMERE_N_CTX")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(8192);
    let n_gpu_layers: i32 = env::var("CHIMERE_N_GPU_LAYERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(99);

    // Boot a 2-slot native context (minimum for NativeScheduler::is_native()).
    let llama = LlamaForward::new_multi_seq(
        &model_path, n_gpu_layers, n_ctx, /*ncmoe=*/ 0,
        /*type_k=*/ None, /*type_v=*/ None, /*flash_attn=*/ true,
        /*n_seq_max=*/ 2,
    ).expect("LlamaForward::new_multi_seq");

    // Tokenize the short prompt. We construct the chat template manually
    // to stay dependency-free of server.rs::messages_to_prompt.
    let prompt_text = "<|im_start|>system\nTu es concis.<|im_end|>\n\
                       <|im_start|>user\nRéponds par \"OK\" et rien d'autre.<|im_end|>\n\
                       <|im_start|>assistant\n";
    let prompt_tokens = llama
        .tokenize(prompt_text, /*add_special=*/ false, /*parse_special=*/ true)
        .expect("tokenize");

    // --- Build the scheduler ---
    let cfg = SchedulerConfig {
        num_slots: 2, queue_cap: 8, enabled: true, native: true,
    };
    let mut sched = NativeScheduler::new(cfg, /*engram=*/ None, /*default_alpha=*/ 0.0)
        .expect("NativeScheduler::new");
    let _driver = sched.spawn_native_driver(llama).expect("spawn_native_driver");

    // --- Send request ---
    let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamMsg>(128);
    let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let params = NativeSamplingParams {
        temperature: 0.0,      // greedy → fully deterministic
        top_p: 1.0, top_k: 1, min_p: 0.0,
        presence_penalty: 0.0,
        max_tokens: 256,       // large enough that a regression is obvious
        stop_tokens: Vec::new(), // EOG path is what we test
        enable_thinking: false,
    };
    let native_req = NativeScheduledRequest {
        request_id: "test-stop-token-imend".into(),
        prompt_tokens: prompt_tokens.iter().map(|&t| t as u32).collect(),
        params,
        engram_alpha: 0.0,
        engram_hint: None,
        tx,
        want_logprobs: false,
        top_logprobs_n: 0,
        enable_thinking: false,
        cancelled,
        enqueued_at: Instant::now(),
    };
    sched.admission_tx().send(native_req).await.expect("admission send");

    // --- Consume stream with a hard timeout ---
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut generated_tokens: usize = 0;
    let mut finish_reason: Option<String> = None;
    let mut saw_content = false;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            panic!("stream did not terminate within 30s — stop-token regression?");
        }
        match tokio::time::timeout(remaining, rx.recv()).await {
            Ok(Some(StreamMsg::Token { text, .. })) => {
                if !text.is_empty() { saw_content = true; }
                generated_tokens += 1;
            }
            Ok(Some(StreamMsg::Thinking { .. })) => {
                generated_tokens += 1;
            }
            Ok(Some(StreamMsg::ToolCall { .. })) => {
                // Shouldn't happen with this prompt, but don't fail on it.
            }
            Ok(Some(StreamMsg::Done { finish_reason: r })) => {
                finish_reason = Some(r);
                break;
            }
            Ok(Some(StreamMsg::Error { message })) => {
                panic!("stream errored: {}", message);
            }
            Ok(None) => panic!("stream channel closed without Done"),
            Err(_) => panic!("stream stalled (no frame in {:?})", remaining),
        }
    }

    assert!(saw_content, "expected at least one non-empty Token frame");
    assert_eq!(
        finish_reason.as_deref(),
        Some("stop"),
        "expected finish_reason=\"stop\" (EOG), got {:?} after {} tokens",
        finish_reason, generated_tokens,
    );
    assert!(
        generated_tokens <= 16,
        "expected <= 16 generated tokens for an 'OK' reply, got {} — \
         EOG detection likely regressed",
        generated_tokens,
    );

    sched.shutdown();
}
