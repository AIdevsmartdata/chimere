//! # Text Generation Pipeline — Chimere Engine
//!
//! End-to-end text generation using a Qwen3.5 model + HuggingFace tokenizer.
//!
//! ## Usage
//!
//! ```ignore
//! let tokenizer = load_tokenizer(None)?;
//! let text = generate_text(&model, &tokenizer, "Hello!", 50, 0.7, &mut state)?;
//! println!("{}", text);
//! ```
//!
//! The tokenizer is loaded from the HuggingFace `tokenizer.json` shipped with
//! the Jackrong BF16 model download. The Qwen3.5 chat template is applied
//! automatically when using `generate_chat`.

use tokenizers::Tokenizer;

use crate::chimere_model::{forward_prefill_gdn, ChimereModel, ForwardOutput};
use crate::engram_lookup::{EngramLookup, MultiEngramLookup};
use crate::mtp_scheduler::{generate_with_mtp, generate_with_mtp_engram, MtpStats, SamplingParams};
use crate::state::GdnRecurrentState;

/// Default tokenizer path (relative to $HOME).
const DEFAULT_TOKENIZER_REL: &str = ".chimere/models/qwopus-27b-bf16/tokenizer.json";

/// Special token IDs for Qwen3.5.
pub const TOKEN_IM_START: u32 = 248045;
pub const TOKEN_IM_END: u32 = 248046;
pub const TOKEN_ENDOFTEXT: u32 = 248044;

/// EOS tokens — generation stops when any of these are produced.
pub const EOS_TOKENS: &[u32] = &[TOKEN_IM_END, TOKEN_ENDOFTEXT];

// ---------------------------------------------------------------------------
// Tokenizer loading
// ---------------------------------------------------------------------------

/// Load a HuggingFace tokenizer from disk.
///
/// If `path` is None, uses the default location at
/// `$HOME/.chimere/models/qwopus-27b-bf16/tokenizer.json`.
pub fn load_tokenizer(path: Option<&str>) -> Result<Tokenizer, String> {
    let tokenizer_path = match path {
        Some(p) => p.to_string(),
        None => {
            let home = std::env::var("HOME")
                .unwrap_or_else(|_| "{HOME}".into());
            format!("{}/{}", home, DEFAULT_TOKENIZER_REL)
        }
    };

    if !std::path::Path::new(&tokenizer_path).exists() {
        return Err(format!(
            "Tokenizer not found at {}. Download the Qwen3.5 tokenizer.json \
             from HuggingFace or extract it from the GGUF metadata.",
            tokenizer_path
        ));
    }

    Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer from {}: {}", tokenizer_path, e))
}

// ---------------------------------------------------------------------------
// Text generation (raw prompt)
// ---------------------------------------------------------------------------

/// Result of a text generation run.
#[derive(Debug)]
pub struct GenerateResult {
    /// The generated text (decoded from tokens).
    pub text: String,
    /// The generated token IDs (excluding prompt tokens).
    pub token_ids: Vec<u32>,
    /// MTP acceleration statistics.
    pub mtp_stats: MtpStats,
    /// Time spent processing the prompt (seconds).
    pub prompt_time_s: f64,
    /// Time spent generating tokens (seconds).
    pub gen_time_s: f64,
    /// Tokens generated per second (gen phase only).
    pub tokens_per_sec: f64,
    /// Per-token packed logprobs (optional, collected when logprobs requested).
    /// Each entry is the packed format: [token_id, n_top, t0, lp0, t1, lp1, ..., t4, lp4].
    pub packed_logprobs: Vec<Vec<f32>>,
}

/// Generate text from a raw string prompt.
///
/// The prompt is tokenized, fed through the model to build up recurrent state,
/// then MTP-accelerated generation produces `max_tokens` new tokens (or stops
/// at EOS). The result is decoded back to text.
///
/// # Arguments
/// - `model`: The loaded Qwen3.5 model.
/// - `tokenizer`: A loaded HuggingFace tokenizer.
/// - `prompt`: Raw text prompt (not chat-formatted).
/// - `max_tokens`: Maximum number of tokens to generate.
/// - `temperature`: Sampling temperature (0.0 = greedy/argmax).
/// - `state`: Mutable model state (will be modified in place).
pub fn generate_text(
    model: &dyn ChimereModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    params: &SamplingParams,
    state: &mut GdnRecurrentState,
) -> Result<GenerateResult, String> {
    // 1. Encode prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("Tokenizer encode failed: {}", e))?;
    let prompt_ids = encoding.get_ids();

    if prompt_ids.is_empty() {
        return Err("Prompt encoded to zero tokens".into());
    }

    // 2. Process prompt tokens via batch prefill (all tokens in one pass).
    //    forward_prefill returns logits for the LAST prompt token.
    let prompt_start = std::time::Instant::now();
    let ForwardOutput { logits: prefill_logits, mtp_logits: _ } =
        forward_prefill_gdn(model, prompt_ids, state)
            .map_err(|e| format!("Prefill failed: {}", e))?;
    let prompt_time = prompt_start.elapsed();

    // 3. Sample first generated token from prefill logits.
    //    If C++ sampler is active, prefill_logits is a fake 1×1 tensor with the pre-sampled token ID.
    let gen_start = std::time::Instant::now();
    let dims = prefill_logits.dims();

    // Collect logprobs from the first token if available (fast-sampler path).
    let mut all_packed_logprobs: Vec<Vec<f32>> = Vec::new();

    let first_token = if dims.len() == 2 && dims[0] == 1 && dims[1] == 12 {
        // C++ fast path: token already sampled with logprobs
        let packed: Vec<f32> = prefill_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let tok = packed[0] as u32;
        all_packed_logprobs.push(packed);
        Ok(tok)
    } else if params.temperature <= 0.0 {
        crate::mtp_scheduler::argmax(&prefill_logits)
    } else {
        crate::mtp_scheduler::sample_advanced(&prefill_logits, params, &[])
    }.map_err(|e| format!("First token sampling failed: {}", e))?;

    let (gen_tokens, mtp_stats) = if first_token == 248046 || first_token == 248044 {
        // First token is EOS — return immediately
        (vec![first_token], crate::mtp_scheduler::MtpStats::default())
    } else {
        // Continue generation from first_token.
        // generate_with_mtp now collects per-token packed logprobs internally.
        let (mut rest, stats, mut rest_logprobs) = generate_with_mtp(model, first_token, max_tokens.saturating_sub(1), state, params, Some(tokenizer))
            .map_err(|e| format!("MTP generation failed: {}", e))?;
        rest.insert(0, first_token);
        all_packed_logprobs.append(&mut rest_logprobs);
        (rest, stats)
    };

    let gen_time = gen_start.elapsed();

    // 4. Decode generated tokens to text
    let text = tokenizer
        .decode(&gen_tokens, true)
        .map_err(|e| format!("Tokenizer decode failed: {}", e))?;

    let gen_secs = gen_time.as_secs_f64();
    let tps = if gen_secs > 0.0 {
        gen_tokens.len() as f64 / gen_secs
    } else {
        0.0
    };

    Ok(GenerateResult {
        text,
        token_ids: gen_tokens,
        mtp_stats,
        prompt_time_s: prompt_time.as_secs_f64(),
        gen_time_s: gen_secs,
        tokens_per_sec: tps,
        packed_logprobs: all_packed_logprobs,
    })
}

// ---------------------------------------------------------------------------
// Engram loading from environment variables
// ---------------------------------------------------------------------------

/// Configuration for EngramLookup integration in the generation loop.
///
/// Loaded from environment variables:
/// - `CHIMERE_ENGRAM_FILE` — path to the engram binary file.
/// - `CHIMERE_ENGRAM_ALPHA` — blending strength (default 0.5).
///
/// When `CHIMERE_ENGRAM_FILE` is not set, `engram` is `None` and generation
/// is identical to the baseline (no performance cost).
pub struct EngramConfig {
    /// Loaded multi-table EngramLookup, or `None` if no env var was set.
    pub engram: Option<MultiEngramLookup>,
    /// Blending strength α in `logits[t] += α * ln(p_engram[t])`.
    pub alpha: f32,
}

impl EngramConfig {
    /// Load an `EngramConfig` from environment variables.
    ///
    /// Supports both `CHIMERE_ENGRAM_DIR` (multi-table) and
    /// `CHIMERE_ENGRAM_FILE` (single file, backward compat).
    pub fn from_env() -> Self {
        let engram = MultiEngramLookup::from_env();
        if let Some(ref e) = engram {
            eprintln!("[ENGRAM] Config: {}", e.summary());
        }

        let alpha: f32 = std::env::var("CHIMERE_ENGRAM_ALPHA")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);

        Self { engram, alpha }
    }
}

// ---------------------------------------------------------------------------
// Engram-aware text generation (raw prompt)
// ---------------------------------------------------------------------------

/// Generate text from a raw string prompt, with optional EngramLookup biasing.
///
/// Identical to `generate_text` when `cfg.engram` is `None`.
///
/// # Arguments
/// - `model`: The loaded Qwen3.5 model.
/// - `tokenizer`: A loaded HuggingFace tokenizer.
/// - `prompt`: Raw text prompt (not chat-formatted).
/// - `max_tokens`: Maximum number of tokens to generate.
/// - `params`: Sampling parameters (temperature, top_p, top_k, penalties).
/// - `state`: Mutable model state (will be modified in place).
/// - `cfg`: Engram configuration (from `EngramConfig::from_env()` or constructed directly).
pub fn generate_text_with_engram(
    model: &dyn ChimereModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    params: &SamplingParams,
    state: &mut GdnRecurrentState,
    cfg: &EngramConfig,
) -> Result<GenerateResult, String> {
    // 1. Encode prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("Tokenizer encode failed: {}", e))?;
    let prompt_ids = encoding.get_ids();

    if prompt_ids.is_empty() {
        return Err("Prompt encoded to zero tokens".into());
    }

    // 2. Process prompt tokens via batch prefill
    let prompt_start = std::time::Instant::now();
    let _ = forward_prefill_gdn(model, prompt_ids, state)
        .map_err(|e| format!("Prefill failed: {}", e))?;
    let prompt_time = prompt_start.elapsed();

    // 3. Generate with optional engram biasing
    let last_tok = *prompt_ids.last().unwrap();
    let gen_start = std::time::Instant::now();
    let (gen_tokens, mtp_stats) = generate_with_mtp_engram(
        model,
        last_tok,
        max_tokens,
        state,
        cfg.engram.as_ref(),
        cfg.alpha,
        params,
    )
    .map_err(|e| format!("MTP+engram generation failed: {}", e))?;
    let gen_time = gen_start.elapsed();

    // 4. Decode
    let text = tokenizer
        .decode(&gen_tokens, true)
        .map_err(|e| format!("Tokenizer decode failed: {}", e))?;

    let gen_secs = gen_time.as_secs_f64();
    let tps = if gen_secs > 0.0 {
        gen_tokens.len() as f64 / gen_secs
    } else {
        0.0
    };

    Ok(GenerateResult {
        text,
        token_ids: gen_tokens,
        mtp_stats,
        prompt_time_s: prompt_time.as_secs_f64(),
        gen_time_s: gen_secs,
        tokens_per_sec: tps,
        packed_logprobs: vec![],
    })
}

// ---------------------------------------------------------------------------
// Chat-formatted generation
// ---------------------------------------------------------------------------

/// Format a user message using the Qwen3.5 chat template.
///
/// Format a user message using the Qwen3.5 chat template (thinking mode).
///
/// The `<think>\n` prefix after `assistant\n` matches the Qwen3.5 Jinja
/// chat template behavior: the model enters thinking mode and generates
/// its reasoning inside `<think>...</think>` before the final answer.
pub fn format_chat_prompt(user_message: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        user_message
    )
}

/// Format a system + user message using the Qwen3.5 chat template (thinking mode).
pub fn format_chat_prompt_with_system(system: &str, user_message: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n",
        system, user_message
    )
}

/// Generate a chat response. Wraps the user message in the Qwen3.5 chat
/// template, then generates text.
///
/// Equivalent to `generate_text` but with automatic chat formatting.
pub fn generate_chat(
    model: &dyn ChimereModel,
    tokenizer: &Tokenizer,
    user_message: &str,
    max_tokens: usize,
    params: &SamplingParams,
    state: &mut GdnRecurrentState,
) -> Result<GenerateResult, String> {
    let prompt = format_chat_prompt(user_message);
    generate_text(model, tokenizer, &prompt, max_tokens, params, state)
}

/// Generate a chat response with optional EngramLookup biasing.
///
/// Equivalent to `generate_chat` but routes through `generate_text_with_engram`.
/// When `cfg.engram` is `None`, behaviour is identical to `generate_chat`.
///
/// # Env-var quick start
/// ```bash
/// CHIMERE_ENGRAM_FILE=/path/to/table.bin CHIMERE_ENGRAM_ALPHA=0.5 ./chimere-server
/// ```
pub fn generate_chat_with_engram(
    model: &dyn ChimereModel,
    tokenizer: &Tokenizer,
    user_message: &str,
    max_tokens: usize,
    params: &SamplingParams,
    state: &mut GdnRecurrentState,
    cfg: &EngramConfig,
) -> Result<GenerateResult, String> {
    let prompt = format_chat_prompt(user_message);
    generate_text_with_engram(model, tokenizer, &prompt, max_tokens, params, state, cfg)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: get the GGUF model path, or None if the file doesn't exist.
    fn gguf_path() -> Option<String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
            home
        );
        if std::path::Path::new(&path).exists() {
            Some(path)
        } else {
            None
        }
    }

    // -------------------------------------------------------------------------
    // Engram pipeline tests (no model required)
    // -------------------------------------------------------------------------

    /// Verify the full engram pipeline without loading a real model:
    /// 1. Build a small engram file from a toy corpus.
    /// 2. Load it back via `EngramConfig`.
    /// 3. Confirm `lookup` returns valid predictions.
    /// 4. Confirm `bias_logits` correctly modifies a logit vector.
    ///
    /// This test exercises `EngramLookup::build`, `from_file`, `order`,
    /// `lookup`, and `bias_logits` end-to-end.  It does NOT require the
    /// GGUF model or tokenizer.
    #[test]
    fn test_engram_pipeline_no_model() {
        use crate::engram_lookup::{EngramLookup, MultiEngramLookup};

        let corpus: Vec<u32> = vec![
            10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30,
            1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4,
        ];
        let path = "/tmp/chimere_generate_test_engram.bin".to_string();

        // --- Step 1: Build ---
        EngramLookup::build(&corpus, 2, &path).expect("build failed");

        // --- Step 2: Load via EngramConfig ---
        let eng = EngramLookup::from_file(&path).expect("from_file failed");
        assert_eq!(eng.order(), 2, "Order should be 2");
        let multi = MultiEngramLookup::from_single("test".to_string(), eng);
        let cfg = super::EngramConfig {
            engram: Some(multi),
            alpha: 0.5,
        };

        // --- Step 3: Verify lookup ---
        let engram_ref = cfg.engram.as_ref().unwrap();

        // [10, 20] → should predict 30 with p=1.0
        let preds_a = engram_ref.lookup(&[10, 20]);
        assert!(!preds_a.is_empty(), "Expected predictions for [10,20]");
        assert_eq!(preds_a[0].0, 30, "Expected token 30");
        assert!(
            (preds_a[0].1 - 1.0).abs() < 1e-5,
            "Expected p=1.0, got {}",
            preds_a[0].1
        );

        // [1, 2] → should predict tokens 3 and 4, each with p=0.5
        let preds_b = engram_ref.lookup(&[1, 2]);
        assert_eq!(preds_b.len(), 2, "Expected 2 predictions for [1,2]");
        let total_p: f32 = preds_b.iter().map(|(_, p)| p).sum();
        assert!(
            (total_p - 1.0).abs() < 1e-5,
            "Probabilities must sum to 1.0, got {}",
            total_p
        );

        // Unknown context → empty
        let preds_c = engram_ref.lookup(&[99, 99]);
        assert!(preds_c.is_empty(), "Unknown context must return empty");

        // --- Step 4: Verify bias_logits ---
        // alpha=0 → no change
        let mut logits_zero = vec![0.0f32; 200];
        let baseline = logits_zero.clone();
        EngramLookup::bias_logits(&mut logits_zero, &preds_a, 0.0);
        assert_eq!(logits_zero, baseline, "alpha=0 must leave logits unchanged");

        // alpha=0.5, p=0.5 for tokens 3 and 4 → delta = 0.5 * ln(0.5)
        let mut logits_half = vec![0.0f32; 200];
        EngramLookup::bias_logits(&mut logits_half, &preds_b, 0.5);
        let expected_delta = 0.5f32 * (0.5f32).ln(); // ≈ -0.347
        for &(tok, _) in &preds_b {
            assert!(
                (logits_half[tok as usize] - expected_delta).abs() < 1e-5,
                "Token {tok}: expected bias {expected_delta:.6}, got {:.6}",
                logits_half[tok as usize]
            );
        }
        // Out-of-range and unrelated tokens untouched
        assert_eq!(logits_half[0], 0.0);
        assert_eq!(logits_half[10], 0.0);
        assert_eq!(logits_half[20], 0.0);

        eprintln!("[ENGRAM TEST] Pipeline verified: build → load → lookup → bias_logits OK");
        eprintln!(
            "[ENGRAM TEST] preds for [10,20]: {:?}",
            preds_a
        );
        eprintln!(
            "[ENGRAM TEST] preds for [1,2]: {:?}",
            preds_b
        );
        eprintln!(
            "[ENGRAM TEST] bias delta at alpha=0.5, p=0.5: {:.6}",
            expected_delta
        );
    }

    /// Test `EngramConfig::from_env` with env vars set.
    #[test]
    fn test_engram_config_from_env() {
        use crate::engram_lookup::EngramLookup;

        let corpus: Vec<u32> = vec![1u32, 2, 3, 1, 2, 3, 1, 2, 3];
        let path = "/tmp/chimere_envconfig_test_engram.bin".to_string();
        EngramLookup::build(&corpus, 2, &path).expect("build failed");

        // Set env vars for this test.
        std::env::set_var("CHIMERE_ENGRAM_FILE", &path);
        std::env::set_var("CHIMERE_ENGRAM_ALPHA", "0.7");
        std::env::remove_var("CHIMERE_ENGRAM_DIR");

        let cfg = super::EngramConfig::from_env();
        assert!(cfg.engram.is_some(), "Expected engram to load from env var");
        assert!(
            (cfg.alpha - 0.7).abs() < 1e-5,
            "Expected alpha=0.7, got {}",
            cfg.alpha
        );

        // Cleanup env vars so we don't leak state into other tests.
        std::env::remove_var("CHIMERE_ENGRAM_FILE");
        std::env::remove_var("CHIMERE_ENGRAM_ALPHA");

        eprintln!("[ENGRAM ENV TEST] EngramConfig::from_env() loaded table correctly");
    }

    /// Test `EngramConfig::from_env` with no env vars (default path).
    #[test]
    fn test_engram_config_from_env_missing() {
        // Ensure env vars are unset.
        std::env::remove_var("CHIMERE_ENGRAM_FILE");
        std::env::remove_var("CHIMERE_ENGRAM_ALPHA");
        std::env::remove_var("CHIMERE_ENGRAM_DIR");

        let cfg = super::EngramConfig::from_env();
        assert!(
            cfg.engram.is_none(),
            "No env var → engram should be None"
        );
        assert!(
            (cfg.alpha - 0.5).abs() < 1e-5,
            "Default alpha should be 0.5, got {}",
            cfg.alpha
        );
    }

    #[test]
    fn test_load_tokenizer() {
        let tok = load_tokenizer(None);
        match tok {
            Ok(t) => {
                // Verify it can encode/decode
                let enc = t.encode("Hello, world!", false).unwrap();
                let ids = enc.get_ids();
                assert!(!ids.is_empty(), "Encoding should produce tokens");
                let decoded = t.decode(ids, true).unwrap();
                assert!(
                    decoded.contains("Hello"),
                    "Decoded text should contain 'Hello', got: {}",
                    decoded
                );
                eprintln!("[TOKENIZER] Encoded 'Hello, world!' -> {} tokens: {:?}", ids.len(), ids);
                eprintln!("[TOKENIZER] Decoded back: '{}'", decoded);
            }
            Err(e) => {
                eprintln!("[TOKENIZER] Skipping: {}", e);
            }
        }
    }

    #[test]
    fn test_chat_template_encoding() {
        let tok = match load_tokenizer(None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[TOKENIZER] Skipping: {}", e);
                return;
            }
        };

        let prompt = format_chat_prompt("Hello, how are you?");
        eprintln!("[CHAT] Formatted prompt: '{}'", prompt);

        let enc = tok.encode(prompt.as_str(), false).unwrap();
        let ids = enc.get_ids();
        eprintln!("[CHAT] Token IDs ({} tokens): {:?}", ids.len(), ids);

        // Verify special tokens are present
        assert!(
            ids.contains(&TOKEN_IM_START),
            "Should contain <|im_start|> token ({})",
            TOKEN_IM_START
        );
        assert!(
            ids.contains(&TOKEN_IM_END),
            "Should contain <|im_end|> token ({})",
            TOKEN_IM_END
        );
    }

    #[test]
    fn test_generate_text() {
        // Skip if model or tokenizer not available
        let model_path = match gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[GENERATE] Skipping: GGUF model not found");
                return;
            }
        };
        let tokenizer = match load_tokenizer(None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[GENERATE] Skipping: {}", e);
                return;
            }
        };

        let device =
            candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        eprintln!("[GENERATE] Device: {:?}", device);

        let model = crate::qwen35_model::Qwen35Model::from_gguf(&model_path, &device, None)
            .expect("Failed to load model");

        let mut state =
            crate::state::GdnRecurrentState::new(&model.config, &device)
                .expect("Failed to create state");

        let result = generate_chat(
            &model,
            &tokenizer,
            "Write a Python function that checks if a number is prime. Include type hints and docstring.",
            200,
            &SamplingParams::default(),
            &mut state,
        )
        .expect("Generation failed");

        eprintln!("============================================");
        eprintln!("[GENERATE] Prompt processing: {:.2}s", result.prompt_time_s);
        eprintln!(
            "[GENERATE] Generation: {:.2}s ({:.1} tok/s)",
            result.gen_time_s, result.tokens_per_sec
        );
        eprintln!(
            "[GENERATE] MTP acceptance: {:.1}%",
            result.mtp_stats.acceptance_rate() * 100.0
        );
        eprintln!(
            "[GENERATE] Tokens ({}):",
            result.token_ids.len()
        );
        eprintln!("  IDs: {:?}", &result.token_ids[..result.token_ids.len().min(30)]);
        eprintln!("============================================");
        eprintln!("[GENERATE] Output text:");
        eprintln!("{}", result.text);
        eprintln!("============================================");

        // Basic assertions
        assert!(!result.text.is_empty(), "Generated text should not be empty");
        assert!(
            !result.token_ids.is_empty(),
            "Should generate at least 1 token"
        );
        assert!(result.token_ids.len() <= 200, "Should not exceed max_tokens");
    }

    #[test]
    fn test_generate_long_500() {
        let model_path = match gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[LONG] Skipping: GGUF model not found");
                return;
            }
        };
        let tokenizer = match load_tokenizer(None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[LONG] Skipping: {}", e);
                return;
            }
        };

        let device =
            candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);

        let model = crate::qwen35_model::Qwen35Model::from_gguf(&model_path, &device, None)
            .expect("Failed to load model");

        let mut state =
            crate::state::GdnRecurrentState::new(&model.config, &device)
                .expect("Failed to create state");

        // Longer prompt
        let prompt = "Write a comprehensive Python class for a binary search tree with insert, \
                      search, delete, and in-order traversal methods. Include type hints, \
                      docstrings, and handle edge cases properly.";

        let result = generate_chat(
            &model,
            &tokenizer,
            prompt,
            500,
            &SamplingParams::default(),
            &mut state,
        )
        .expect("Generation failed");

        // Measure tok/s in segments
        let total = result.token_ids.len();
        let gen_ms = result.gen_time_s * 1000.0;
        let ms_per_tok = if total > 0 { gen_ms / total as f64 } else { 0.0 };

        eprintln!("============================================");
        eprintln!("[LONG] Prompt processing: {:.2}s", result.prompt_time_s);
        eprintln!("[LONG] Total tokens: {}", total);
        eprintln!("[LONG] Generation: {:.2}s ({:.1} tok/s, {:.1}ms/tok)",
            result.gen_time_s, result.tokens_per_sec, ms_per_tok);
        eprintln!("[LONG] KV cache seq_len at end: {}", state.position);

        // VRAM check
        if let candle_core::Device::Cuda(_) = &device {
            eprintln!("[LONG] (check nvidia-smi for VRAM after test)");
        }

        // Print first and last 200 chars of output
        let text = &result.text;
        if text.len() > 400 {
            eprintln!("[LONG] First 200 chars: {}", &text[..200]);
            eprintln!("[LONG] Last 200 chars: {}", &text[text.len()-200..]);
        } else {
            eprintln!("[LONG] Full output: {}", text);
        }
        eprintln!("============================================");

        assert!(total >= 100, "Should generate at least 100 tokens, got {}", total);
        assert!(!text.is_empty(), "Text should not be empty");
    }

    /// Throughput benchmark across multiple prompt lengths and max_token budgets.
    ///
    /// Run with:
    ///   cargo test --release bench_throughput_comparison -- --nocapture
    ///
    /// Results are printed as a Markdown table to stderr.
    #[test]
    fn bench_throughput_comparison() {
        // Skip if model not available
        let model_path = match gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[BENCH] Skipping: GGUF model not found at expected path");
                return;
            }
        };
        let tokenizer = match load_tokenizer(None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[BENCH] Skipping: tokenizer not available: {}", e);
                return;
            }
        };

        let device =
            candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        eprintln!("[BENCH] Device: {:?}", device);
        eprintln!("[BENCH] Model: {}", model_path);

        let model = crate::qwen35_model::Qwen35Model::from_gguf(&model_path, &device, None)
            .expect("Failed to load model for benchmark");

        let prompts: &[(&str, &str)] = &[
            (
                "short",
                "What is 2+2?",
            ),
            (
                "medium",
                "Write a Python function to sort a list using quicksort.",
            ),
            (
                "long",
                "Write a comprehensive Python class for a binary search tree with insert, \
                 search, delete, and in-order traversal methods. Include type hints, \
                 docstrings, and handle edge cases properly.",
            ),
        ];

        // (config_name, max_tokens)
        let configs: &[(&str, usize)] = &[
            ("baseline", 200),
            ("long_gen",  500),
        ];

        eprintln!();
        eprintln!("=== CHIMERE-DELTANET THROUGHPUT BENCHMARK ===");
        eprintln!();
        eprintln!(
            "| {:<8} | {:<8} | {:<12} | {:<12} | {:<10} | {:<12} | {:<10} | {:<10} |",
            "Prompt", "Config", "prompt_toks", "gen_toks", "prompt_s",
            "gen_s", "tok/s", "mtp_acc%"
        );
        eprintln!(
            "|{:-<10}|{:-<10}|{:-<14}|{:-<14}|{:-<14}|{:-<14}|{:-<12}|{:-<12}|",
            "", "", "", "", "", "", "", ""
        );

        for (prompt_name, prompt_text) in prompts {
            for (config_name, max_tokens) in configs {
                // Fresh state for every run to avoid cross-contamination
                let mut state =
                    crate::state::GdnRecurrentState::new(&model.config, &device)
                        .expect("Failed to create recurrent state");

                match generate_chat(
                    &model,
                    &tokenizer,
                    prompt_text,
                    *max_tokens,
                    &SamplingParams::default(),
                    &mut state,
                ) {
                    Ok(result) => {
                        // Approximate prompt token count via tokenizer
                        let formatted = format_chat_prompt(prompt_text);
                        let prompt_tok_count = tokenizer
                            .encode(formatted.as_str(), false)
                            .map(|e| e.get_ids().len())
                            .unwrap_or(0);

                        let mtp_pct = result.mtp_stats.acceptance_rate() * 100.0;

                        eprintln!(
                            "| {:<8} | {:<8} | {:<12} | {:<12} | {:<12.3} | {:<12.3} | {:<10.1} | {:<10.1} |",
                            prompt_name,
                            config_name,
                            prompt_tok_count,
                            result.token_ids.len(),
                            result.prompt_time_s,
                            result.gen_time_s,
                            result.tokens_per_sec,
                            mtp_pct,
                        );
                    }
                    Err(e) => {
                        eprintln!(
                            "| {:<8} | {:<8} | {:>12} | {:>12} | {:>12} | {:>12} | {:>10} | {:>10} |",
                            prompt_name, config_name,
                            "ERR", "ERR", "ERR", "ERR", "ERR",
                            format!("error: {}", e),
                        );
                    }
                }
            }
        }

        eprintln!();
        eprintln!("=== END BENCHMARK ===");
        eprintln!();
        // No hard assertions — this is a measurement-only test.
    }
}
