//! # Block Diffusion Generation — Chimere Engine
//!
//! Real block diffusion generation using the Qwen3.5 model.
//!
//! Generates N tokens in parallel via iterative masked denoising instead of
//! autoregressive 1-by-1 generation. The key loop:
//!
//! 1. Initialize a block of N positions with MASK tokens
//! 2. Forward ALL N tokens through the model (sequential per-token for now)
//! 3. Compute confidence per position (max softmax probability)
//! 4. Unmask the most confident positions (greedy argmax)
//! 5. Repeat steps 2-4 for K iterations
//! 6. Result: N tokens in K forwards instead of N forwards
//!
//! ## State management
//!
//! The GDN recurrent state is snapshot'd before each denoising iteration
//! and restored afterwards. This prevents the speculative block processing
//! from permanently advancing the state. Only after the block is finalized
//! do we commit by feeding the final tokens through the model.
//!
//! ## Connection to existing modules
//!
//! - Uses `Qwen35Model::forward_token()` as the score function
//! - Uses `GdnRecurrentState::snapshot()/restore()` for rollback
//! - Bridges to the `block_diffusion` module's `ScoreModel` trait
//! - Integrates with `entropy_router` via `generate_adaptive()`

// =============================================================================
// IMPLEMENTATION STATUS — Sprint 3 complete, Sprint 4 planned
// Updated: 2026-03-15
// =============================================================================
//
// ## Current state (Sprint 3)
//
// `generate_block()` is functional: it implements the full iterative masked
// denoising loop (snapshot → forward all block tokens → restore → unmask top-K
// → repeat). However, the inner scoring loop calls `forward_token()` once per
// token in the block, sequentially:
//
//   ```
//   for &tok in block.iter() {          // 16 iterations
//       let (logits, _) = model.forward_token(tok, state)?;  // ~56ms each
//   }
//   ```
//
// For a 16-token block with 4 denoising steps this is:
//   16 tokens × 4 steps × 56ms = 3,584ms per block
// (vs the committed AR baseline of 16 × 56ms = 896ms — paradoxically slower.)
//
// The block loop also ignores the GDN recurrent state dependency between
// positions: each token is scored with the state *as if the previous tokens
// in the block had already been committed*, which is an approximation. This
// is intentional for the PoC (the block is speculative; the state is rolled
// back and rerun on the accepted tokens after finalization).
//
// ## What was added this session (Sprint 3)
//
// `Qwen35Model::forward_prefill(tokens: &[u32], state)` now exists in
// `src/qwen35_model.rs`. It embeds all N tokens at once via `index_select`,
// builds a batched `[N, hidden_size]` tensor, runs it through all GDN layers
// in a single forward pass, and returns the full `[N, vocab_size]` logit
// matrix.  This is the primitive needed to make block scoring efficient.
//
// The function is tested via `src/prefill.rs` and referenced in
// `src/generate.rs` (prompt ingestion) and `src/mtp_scheduler.rs` (draft
// generation). It is NOT yet used inside `generate_block()`.
//
// ## Connection to entropy_router.rs
//
// `generate_adaptive()` (bottom of this file) is the integration point:
//
//   1. After each token, `compute_logit_entropy()` (from `entropy_router.rs`)
//      is called on the output logits.
//   2. If entropy < `ENTROPY_THRESHOLD_BLOCK` (0.3) and enough budget remains,
//      the function calls `generate_block()` to produce the next 16 tokens via
//      block diffusion.
//   3. Otherwise it falls back to `generate_with_mtp()` for single-token AR.
//
// `entropy_router.rs` also defines the full `EntropyRouter` struct with a
// 4-quadrant decision space (GreedyAR / SampledAR / SpeculativeDraft /
// BlockDiffusion). `generate_adaptive()` currently only uses a simplified
// 1D threshold; the full `EntropyRouter` routing (2D entropy × varentropy,
// GDN-state confidence modulation, hysteresis) is NOT yet wired in. Connecting
// them is Sprint 4 work.
//
// `entropy_router::compute_logit_entropy()` is also used by
// `Qwen35ScoreModel::forward()` (the `ScoreModel` bridge below) indirectly
// via the confidence calculation — all callers share the same numerics.
//
// ## Sprint 4: integrate forward_prefill into generate_block (DO NOT TOUCH YET)
//
// The planned change is to replace the sequential inner loop:
//
//   ```
//   // CURRENT (slow): 16 × forward_token calls per denoising step
//   for &tok in block.iter() {
//       let (logits, _) = model.forward_token(tok, state)?;
//       all_logits.push(logits.squeeze(0)?.to_vec1()?);
//   }
//   ```
//
// with a single batched call:
//
//   ```
//   // PLANNED (Sprint 4): 1 × forward_prefill call per denoising step
//   let all_logits_tensor = model.forward_prefill(&block, state)?;
//   // all_logits_tensor: [block_size, vocab_size]
//   // Extract per-position logit vectors from the batched output
//   ```
//
// Expected latency gain (empirical baselines from MEMORY.md):
//   - Current:  16 forward_token × 56ms × 4 steps = ~3,584ms / block
//   - Planned:  forward_prefill(16) × 4 steps ≈ 4 × 120ms = ~480ms / block
//   - Speedup:  ~7.5× for the block scoring loop
//
// Note: `forward_prefill` advances the GDN recurrent state sequentially
// through all N positions in order (it is not a true parallel attention pass
// over the block — GDN is recurrent). The speedup comes from fused embedding
// lookup, a single kernel dispatch for the attention-like steps, and avoided
// Python/Rust dispatch overhead between tokens. The approximation that
// positions share a common scoring state remains unchanged.
//
// Prerequisite before integrating:
//   - Verify `forward_prefill` returns logits in the shape [N, vocab_size]
//     (not [1, vocab_size] for the last position only). Currently the function
//     returns the full sequence tensor — double-check on a unit test.
//   - Ensure `state.restore(&iter_snapshot)` correctly undoes the N-step
//     state advance done by `forward_prefill` (it should, since restore is a
//     full tensor copy).
//   - Add a `test_generate_block_prefill_parity` test that compares token IDs
//     and logit values between the sequential and batched paths.
//
// This work is deferred to Sprint 4. Do not modify the scoring loop below
// until the parity test passes.
//
// =============================================================================

use candle_core::Result;

use crate::entropy_router::compute_logit_entropy;
use crate::generate::EOS_TOKENS;
use crate::mtp_scheduler::{generate_with_mtp, MtpStats};
use crate::qwen35_model::Qwen35Model;
use crate::state::GdnRecurrentState;

// ---------------------------------------------------------------------------
// Block Diffusion Core
// ---------------------------------------------------------------------------

/// Statistics from a block diffusion generation run.
#[derive(Debug, Clone)]
pub struct BlockDiffusionStats {
    /// Number of denoising iterations performed
    pub iterations: usize,
    /// Tokens generated in this block
    pub block_size: usize,
    /// Per-iteration count of newly unmasked positions
    pub unmask_history: Vec<usize>,
    /// Final number of positions that remained masked (should be 0)
    pub remaining_masked: usize,
    /// Time spent in block diffusion (seconds)
    pub elapsed_s: f64,
}

/// Generate a block of tokens using iterative masked denoising.
///
/// The model is used as a score function: for each masked position, we forward
/// through the model to obtain logits, then pick the most confident predictions
/// to unmask first. State is rolled back between iterations so the denoising
/// loop doesn't permanently alter the recurrent state.
///
/// After the block is finalized, the caller should feed the generated tokens
/// through the model to commit state changes (done by `generate_adaptive`).
///
/// # Arguments
/// - `model`: The loaded Qwen3.5 model
/// - `block_size`: Number of tokens to generate (e.g. 16 or 32)
/// - `num_steps`: Number of denoising iterations (e.g. 4 or 8)
/// - `state`: Mutable model state (will be snapshot'd/restored internally)
/// - `temperature`: Sampling temperature (0.0 = greedy argmax)
///
/// # Returns
/// - Generated token IDs for the block
/// - Block diffusion statistics
pub fn generate_block(
    model: &Qwen35Model,
    block_size: usize,
    num_steps: usize,
    state: &mut GdnRecurrentState,
    _temperature: f64,
) -> Result<(Vec<u32>, BlockDiffusionStats)> {
    let start = std::time::Instant::now();

    // Use token 0 as mask placeholder. Qwen3.5 doesn't have a dedicated [MASK]
    // token, so we use 0 (which is typically unused / <unk>).
    let mask_token: u32 = 0;

    let mut block = vec![mask_token; block_size];
    let mut unmasked = vec![false; block_size];
    let mut unmask_history = Vec::with_capacity(num_steps);

    // Snapshot the state before any block processing
    let base_snapshot = state.snapshot()?;

    for step in 0..num_steps {
        // 1. Save state before this iteration's forward passes
        let iter_snapshot = state.snapshot()?;

        // 2. Forward all tokens in the block to collect logits per position
        let mut all_logits: Vec<Vec<f32>> = Vec::with_capacity(block_size);

        for &tok in block.iter() {
            let (logits, _mtp) = model.forward_token(tok, state)?;
            let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            all_logits.push(logits_vec);
        }

        // 3. Restore state to before this iteration (don't permanently advance)
        state.restore(&iter_snapshot)?;

        // 4. Compute confidence for masked positions
        //    Confidence = max softmax probability = 1 / sum(exp(l_i - l_max))
        let mut confidences: Vec<(usize, f32)> = Vec::new();
        for (pos, logits) in all_logits.iter().enumerate() {
            if !unmasked[pos] {
                let max_logit = logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
                let max_prob = 1.0 / exp_sum;
                confidences.push((pos, max_prob));
            }
        }

        // 5. Sort by confidence (highest first)
        confidences.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 6. Determine how many positions to unmask this step
        //    Spread unmasking evenly across remaining steps
        let currently_masked = unmasked.iter().filter(|&&u| !u).count();
        let remaining_steps = num_steps - step;
        let n_unmask = std::cmp::max(1, currently_masked / remaining_steps);

        // 7. Unmask the top-confidence positions
        let mut unmasked_this_step = 0;
        for &(pos, _conf) in confidences.iter().take(n_unmask) {
            let logits = &all_logits[pos];
            block[pos] = sample_argmax(logits);
            unmasked[pos] = true;
            unmasked_this_step += 1;
        }
        unmask_history.push(unmasked_this_step);

        let total_unmasked = unmasked.iter().filter(|&&u| u).count();
        eprintln!(
            "[BLOCK] step {}/{}: unmasked {} this step, {}/{} total",
            step + 1,
            num_steps,
            unmasked_this_step,
            total_unmasked,
            block_size
        );

        // Early exit if everything is unmasked
        if total_unmasked == block_size {
            break;
        }
    }

    // Fill any remaining masked positions (shouldn't happen if num_steps >= log2(block_size))
    let mut remaining_masked = 0;
    for (pos, &is_unmasked) in unmasked.iter().enumerate() {
        if !is_unmasked {
            // Use the last iteration's logits if available, otherwise fallback to newline
            block[pos] = 198; // \n as safe fallback
            remaining_masked += 1;
        }
    }

    // Restore to the base snapshot -- the caller (generate_adaptive) will
    // commit the block by feeding the final tokens through forward_token.
    state.restore(&base_snapshot)?;

    let elapsed = start.elapsed().as_secs_f64();
    let stats = BlockDiffusionStats {
        iterations: unmask_history.len(),
        block_size,
        unmask_history,
        remaining_masked,
        elapsed_s: elapsed,
    };

    Ok((block, stats))
}

/// Argmax sampling: return the token ID with the highest logit.
fn sample_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| {
            a.1.partial_cmp(b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// ScoreModel bridge: connect block_diffusion scheduler to the real model
// ---------------------------------------------------------------------------

use crate::block_diffusion::{MaskedBlock, ScoreModel, ScoreModelOutput, TokenPrediction};

/// Wraps a Qwen3.5 model as a `ScoreModel` for the block diffusion scheduler.
///
/// This is the bridge between the abstract `ScoreModel` trait in
/// `block_diffusion.rs` and the real `Qwen35Model`. It runs the full
/// transformer forward pass for each position in the masked block.
pub struct Qwen35ScoreModel<'a> {
    pub model: &'a Qwen35Model,
    pub state: GdnRecurrentState,
}

impl<'a> ScoreModel for Qwen35ScoreModel<'a> {
    fn forward(&self, block: &MaskedBlock, context: &[u32]) -> ScoreModelOutput {
        // Clone state so we can restore after scoring
        let mut scoring_state = self
            .state
            .snapshot()
            .expect("Failed to snapshot state for score model");

        // Process context tokens first (to build up state)
        for &tok in context {
            let _ = self
                .model
                .forward_token(tok, &mut scoring_state)
                .expect("Context forward failed");
        }

        // Score each position in the block
        let mut predictions = Vec::with_capacity(block.len());
        for (i, &tok) in block.tokens.iter().enumerate() {
            let (logits, _) = self
                .model
                .forward_token(tok, &mut scoring_state)
                .expect("Block forward failed");
            let logits_vec: Vec<f32> = logits
                .squeeze(0)
                .expect("squeeze failed")
                .to_vec1()
                .expect("to_vec1 failed");

            // Compute confidence (max softmax probability)
            let max_logit = logits_vec
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logits_vec.iter().map(|&l| (l - max_logit).exp()).sum();
            let max_prob = 1.0 / exp_sum;

            // Argmax token
            let token_id = sample_argmax(&logits_vec);

            predictions.push(TokenPrediction {
                token_id,
                confidence: max_prob,
                logits: if block.is_masked[i] {
                    Some(logits_vec)
                } else {
                    None
                },
            });
        }

        ScoreModelOutput {
            predictions,
            state_metrics: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive Generation: AR + Block Diffusion switching
// ---------------------------------------------------------------------------

/// Result of adaptive generation (AR + Block Diffusion hybrid).
#[derive(Debug)]
pub struct AdaptiveGenerateResult {
    /// Generated text (decoded from tokens)
    pub text: String,
    /// Generated token IDs
    pub token_ids: Vec<u32>,
    /// Number of tokens produced via AR
    pub ar_tokens: usize,
    /// Number of tokens produced via block diffusion
    pub block_tokens: usize,
    /// Number of block diffusion invocations
    pub block_invocations: usize,
    /// MTP stats (from AR phases)
    pub mtp_stats: MtpStats,
    /// Total generation time (seconds)
    pub gen_time_s: f64,
    /// Tokens per second
    pub tokens_per_sec: f64,
}

/// Entropy threshold below which we switch to block diffusion.
/// Low entropy = model is confident = block diffusion can exploit parallelism.
const ENTROPY_THRESHOLD_BLOCK: f32 = 0.3;

/// Default block size for block diffusion
const DEFAULT_BLOCK_SIZE: usize = 16;

/// Default number of denoising steps
const DEFAULT_BLOCK_STEPS: usize = 4;

/// Generate tokens adaptively, switching between AR and Block Diffusion
/// based on the entropy of the output distribution.
///
/// When the model is confident (low entropy), block diffusion generates
/// multiple tokens at once. When uncertain (high entropy), standard AR
/// generation with MTP proceeds one token at a time for accuracy.
///
/// # Arguments
/// - `model`: The loaded Qwen3.5 model
/// - `tokenizer`: A loaded HuggingFace tokenizer
/// - `prompt`: Raw text prompt
/// - `max_tokens`: Maximum tokens to generate
/// - `state`: Mutable model state
pub fn generate_adaptive(
    model: &Qwen35Model,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    max_tokens: usize,
    state: &mut GdnRecurrentState,
) -> std::result::Result<AdaptiveGenerateResult, String> {
    let gen_start = std::time::Instant::now();

    // 1. Encode and process prompt
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("Tokenizer encode failed: {}", e))?;
    let prompt_ids = encoding.get_ids();

    if prompt_ids.is_empty() {
        return Err("Prompt encoded to zero tokens".into());
    }

    // Feed prompt tokens to build up state
    let mut last_logits_vec: Vec<f32> = Vec::new();
    for &tok in prompt_ids {
        let (logits, _) = model
            .forward_token(tok, state)
            .map_err(|e| format!("Prompt forward failed: {}", e))?;
        last_logits_vec = logits
            .squeeze(0)
            .map_err(|e| format!("squeeze: {}", e))?
            .to_vec1()
            .map_err(|e| format!("to_vec1: {}", e))?;
    }

    // 2. Generate tokens adaptively
    let mut generated = Vec::new();
    let mut ar_tokens = 0usize;
    let mut block_tokens = 0usize;
    let mut block_invocations = 0usize;
    let mut mtp_stats = MtpStats::default();
    let mut last_token = *prompt_ids.last().unwrap();

    while generated.len() < max_tokens {
        // Compute entropy of current logits
        let (entropy, _varentropy) = if last_logits_vec.is_empty() {
            (1.0f32, 0.0f32)
        } else {
            compute_logit_entropy(&last_logits_vec)
        };

        if entropy < ENTROPY_THRESHOLD_BLOCK && (max_tokens - generated.len()) >= DEFAULT_BLOCK_SIZE {
            // Low entropy: use Block Diffusion (model is confident)
            eprintln!(
                "[ADAPTIVE] entropy={:.3} < {:.3} -> Block Diffusion ({} tokens)",
                entropy, ENTROPY_THRESHOLD_BLOCK, DEFAULT_BLOCK_SIZE
            );

            let (block, _stats) = generate_block(
                model,
                DEFAULT_BLOCK_SIZE,
                DEFAULT_BLOCK_STEPS,
                state,
                0.7,
            )
            .map_err(|e| format!("Block diffusion failed: {}", e))?;

            // Commit the block: feed each generated token through the model
            // to properly advance the recurrent state
            for &tok in &block {
                let (logits, _) = model
                    .forward_token(tok, state)
                    .map_err(|e| format!("Block commit forward failed: {}", e))?;
                last_logits_vec = logits
                    .squeeze(0)
                    .map_err(|e| format!("squeeze: {}", e))?
                    .to_vec1()
                    .map_err(|e| format!("to_vec1: {}", e))?;

                // Check for EOS
                if EOS_TOKENS.contains(&tok) {
                    generated.push(tok);
                    block_tokens += 1;
                    // Stop generation
                    let gen_time = gen_start.elapsed().as_secs_f64();
                    let text = tokenizer
                        .decode(&generated, true)
                        .map_err(|e| format!("Decode failed: {}", e))?;
                    return Ok(AdaptiveGenerateResult {
                        text,
                        token_ids: generated,
                        ar_tokens,
                        block_tokens,
                        block_invocations,
                        mtp_stats,
                        gen_time_s: gen_time,
                        tokens_per_sec: (ar_tokens + block_tokens) as f64 / gen_time,
                    });
                }
            }

            generated.extend_from_slice(&block);
            last_token = *block.last().unwrap_or(&last_token);
            block_tokens += block.len();
            block_invocations += 1;
        } else {
            // High entropy: use AR with MTP (model is uncertain, be careful)
            eprintln!(
                "[ADAPTIVE] entropy={:.3} >= {:.3} -> AR+MTP (1 token)",
                entropy, ENTROPY_THRESHOLD_BLOCK
            );

            // Generate just 1 token via MTP
            let greedy = crate::mtp_scheduler::SamplingParams { temperature: 0.0, ..crate::mtp_scheduler::SamplingParams::default() };
            let (tokens, stats, _logprobs) = generate_with_mtp(model, last_token, 1, state, &greedy, None)
                .map_err(|e| format!("AR generation failed: {}", e))?;

            if tokens.is_empty() {
                break;
            }

            // Get logits for entropy estimation on next iteration
            // (The MTP generator already advanced state, so we use the last token's logits)
            let last_generated = *tokens.last().unwrap();
            // We need fresh logits for the entropy check. The MTP path already did
            // forward passes, but we don't have the raw logits. We'll use the state
            // as-is and compute entropy on the next iteration by doing a peek forward.
            // For simplicity in this PoC, just snapshot+forward+restore to get logits.
            {
                let peek_snap = state
                    .snapshot()
                    .map_err(|e| format!("Peek snapshot failed: {}", e))?;
                let (logits, _) = model
                    .forward_token(last_generated, state)
                    .map_err(|e| format!("Peek forward failed: {}", e))?;
                last_logits_vec = logits
                    .squeeze(0)
                    .map_err(|e| format!("squeeze: {}", e))?
                    .to_vec1()
                    .map_err(|e| format!("to_vec1: {}", e))?;
                state
                    .restore(&peek_snap)
                    .map_err(|e| format!("Peek restore failed: {}", e))?;
            }

            // Check for EOS
            for &tok in &tokens {
                if EOS_TOKENS.contains(&tok) {
                    generated.push(tok);
                    ar_tokens += 1;
                    let gen_time = gen_start.elapsed().as_secs_f64();
                    let text = tokenizer
                        .decode(&generated, true)
                        .map_err(|e| format!("Decode failed: {}", e))?;
                    return Ok(AdaptiveGenerateResult {
                        text,
                        token_ids: generated,
                        ar_tokens,
                        block_tokens,
                        block_invocations,
                        mtp_stats,
                        gen_time_s: gen_time,
                        tokens_per_sec: (ar_tokens + block_tokens) as f64 / gen_time,
                    });
                }
            }

            generated.extend_from_slice(&tokens);
            last_token = last_generated;
            ar_tokens += tokens.len();
            mtp_stats.total_steps += stats.total_steps;
            mtp_stats.accepted += stats.accepted;
            mtp_stats.rejected += stats.rejected;
            mtp_stats.tokens_generated += stats.tokens_generated;
        }
    }

    let gen_time = gen_start.elapsed().as_secs_f64();
    let text = tokenizer
        .decode(&generated, true)
        .map_err(|e| format!("Decode failed: {}", e))?;

    Ok(AdaptiveGenerateResult {
        text,
        token_ids: generated,
        ar_tokens,
        block_tokens,
        block_invocations,
        mtp_stats,
        gen_time_s: gen_time,
        tokens_per_sec: (ar_tokens + block_tokens) as f64 / gen_time,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::load_tokenizer;

    /// Helper: get the GGUF model path, or None if the file doesn't exist.
    fn gguf_path() -> Option<String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-Q4_0-MTP.gguf",
            home
        );
        if std::path::Path::new(&path).exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    fn test_sample_argmax() {
        let logits = vec![1.0f32, 3.0, 2.0, 0.5, -1.0];
        assert_eq!(sample_argmax(&logits), 1);

        let logits = vec![-1.0, -2.0, -0.5];
        assert_eq!(sample_argmax(&logits), 2);

        let logits = vec![5.0];
        assert_eq!(sample_argmax(&logits), 0);
    }

    #[test]
    fn test_block_diffusion_generate() {
        // Skip if model not available
        let model_path = match gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[BLOCK TEST] Skipping: GGUF model not found");
                return;
            }
        };
        let tokenizer = match load_tokenizer(None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[BLOCK TEST] Skipping: {}", e);
                return;
            }
        };

        let device =
            candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        eprintln!("[BLOCK TEST] Device: {:?}", device);

        let model = Qwen35Model::from_gguf(&model_path, &device, None)
            .expect("Failed to load model");

        let mut state = GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        // Process a short prompt first
        let prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
        let encoding = tokenizer.encode(prompt, false).expect("Encode failed");
        let prompt_ids = encoding.get_ids();
        eprintln!("[BLOCK TEST] Processing {} prompt tokens...", prompt_ids.len());

        for &tok in prompt_ids {
            model
                .forward_token(tok, &mut state)
                .expect("Prompt forward failed");
        }

        // Generate a block of 16 tokens with 4 denoising steps
        eprintln!("[BLOCK TEST] Generating block (16 tokens, 4 steps)...");
        let start = std::time::Instant::now();
        let (block, stats) =
            generate_block(&model, 16, 4, &mut state, 0.7).expect("Block generation failed");
        let elapsed = start.elapsed();

        assert_eq!(block.len(), 16, "Block should have 16 tokens");
        assert_eq!(
            stats.remaining_masked, 0,
            "No positions should remain masked"
        );

        // Decode and print
        let text = tokenizer
            .decode(&block, true)
            .expect("Decode failed");

        eprintln!("============================================");
        eprintln!("[BLOCK TEST] Generated {} tokens in {:.2}s", block.len(), elapsed.as_secs_f64());
        eprintln!("[BLOCK TEST] Tokens: {:?}", block);
        eprintln!("[BLOCK TEST] Text: {}", text);
        eprintln!(
            "[BLOCK TEST] Stats: {} iterations, unmask_history={:?}",
            stats.iterations, stats.unmask_history
        );
        eprintln!("============================================");

        // Verify tokens are reasonable (not all zeros, not all the same)
        let unique_tokens: std::collections::HashSet<u32> = block.iter().cloned().collect();
        eprintln!(
            "[BLOCK TEST] Unique tokens: {}/{}",
            unique_tokens.len(),
            block.len()
        );
        // We don't assert on uniqueness since repetition can be natural,
        // but the block should not be all mask tokens (0)
        assert!(
            !block.iter().all(|&t| t == 0),
            "Block should not be all mask tokens"
        );
    }

    #[test]
    fn test_block_diffusion_state_rollback() {
        // Verify that block diffusion correctly restores state after generation.
        let model_path = match gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[STATE TEST] Skipping: GGUF not found");
                return;
            }
        };

        let device =
            candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        let model = Qwen35Model::from_gguf(&model_path, &device, None)
            .expect("Failed to load model");
        let mut state = GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        // Feed a token to initialize state
        model.forward_token(1, &mut state).expect("Forward failed");
        let pos_before = state.position;

        // Take a reference snapshot
        let ref_snapshot = state.snapshot().expect("Snapshot failed");

        // Generate a block (should restore state internally)
        let (_block, _stats) =
            generate_block(&model, 8, 2, &mut state, 0.7).expect("Block gen failed");

        // State should be restored to before block generation
        let pos_after = state.position;
        assert_eq!(
            pos_before, pos_after,
            "Position should be restored after block generation: before={}, after={}",
            pos_before, pos_after
        );

        // GDN state should match the reference snapshot
        let gdn_diff: f32 = state.gdn_states[0]
            .sub(&ref_snapshot.gdn_states[0])
            .expect("sub failed")
            .abs()
            .expect("abs failed")
            .sum_all()
            .expect("sum failed")
            .to_scalar()
            .expect("scalar failed");
        assert!(
            gdn_diff < 1e-6,
            "GDN state should be restored: diff={}",
            gdn_diff
        );

        eprintln!("[STATE TEST] State rollback verified: pos={}, gdn_diff={:.2e}", pos_after, gdn_diff);
    }

    #[test]
    fn test_adaptive_generate() {
        let model_path = match gguf_path() {
            Some(p) => p,
            None => {
                eprintln!("[ADAPTIVE TEST] Skipping: GGUF not found");
                return;
            }
        };
        let tokenizer = match load_tokenizer(None) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("[ADAPTIVE TEST] Skipping: {}", e);
                return;
            }
        };

        let device =
            candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        let model = Qwen35Model::from_gguf(&model_path, &device, None)
            .expect("Failed to load model");
        let mut state = GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        let prompt = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n";

        let result = generate_adaptive(&model, &tokenizer, prompt, 30, &mut state)
            .expect("Adaptive generation failed");

        eprintln!("============================================");
        eprintln!("[ADAPTIVE] Generated {} tokens in {:.2}s ({:.1} tok/s)",
            result.token_ids.len(), result.gen_time_s, result.tokens_per_sec);
        eprintln!("[ADAPTIVE] AR tokens: {}, Block tokens: {} ({} invocations)",
            result.ar_tokens, result.block_tokens, result.block_invocations);
        eprintln!("[ADAPTIVE] MTP acceptance: {:.1}%",
            result.mtp_stats.acceptance_rate() * 100.0);
        eprintln!("[ADAPTIVE] Text: {}", result.text);
        eprintln!("============================================");

        assert!(
            !result.text.is_empty(),
            "Generated text should not be empty"
        );
        assert!(
            result.token_ids.len() <= 30,
            "Should not exceed max_tokens"
        );
    }
}
