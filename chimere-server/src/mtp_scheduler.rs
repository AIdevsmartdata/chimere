//! MTP (Multi-Token Prediction) Scheduler for Qwen3.5.
//!
//! Sequential verify cycle — NO batch multi-token, NO seq_rm, NO rollback.
//!
//! ```text
//! Step 1: forward(token, pos=P) → logits_T + logits_mtp
//! Step 2: sample token_1 = argmax(logits_T)
//!         sample token_2_draft = argmax(logits_mtp)
//! Step 3: forward(token_1, pos=P+1) → logits_verify + logits_mtp_next
//! Step 4: actual_token_2 = argmax(logits_verify)
//!         if actual_token_2 == token_2_draft → ACCEPT (free token!)
//!         else → REJECT (zero cost, state already correct)
//! ```

use candle_core::{Result, Tensor};

use crate::chimere_model::{
    forward_token_gdn, forward_token_generic, ChimereModel, ForwardOutput,
};

// ---------------------------------------------------------------------------
// NEST-style per-token adaptive Engram gating
// ---------------------------------------------------------------------------

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute NEST-style adaptive alpha based on Relative Retrieval Confidence.
///
/// Instead of a fixed `base_alpha` for all tokens, we scale alpha by how
/// confident the Engram lookup is: if the top prediction dominates, we trust
/// the Engram more (high effective alpha); if predictions are spread, we
/// defer to the model (low effective alpha).
///
/// Formula (adapted from NEST's RRC for n-gram lookup):
///   confidence = max_prob / (max_prob + second_prob + eps)
///   effective_alpha = base_alpha * sigmoid((confidence - 0.5) / tau)
///
/// where tau=0.1 gives a sharp transition around confidence=0.5.
///
/// Returns `base_alpha` unchanged if NEST is disabled (`CHIMERE_ENGRAM_NEST=0`)
/// or if predictions have fewer than 1 entry.
#[inline]
/// NEST v2: adaptive alpha from BOTH Engram confidence AND model entropy.
///
/// Formula: α_eff = base × engram_conf × (1 - model_conf)
///   - model_conf from logprobs entropy: low entropy = confident = alpha down
///   - engram_conf from prediction spread: dominant top-1 = confident = alpha up
///   - model confident → alpha ~0 (Engram stays quiet, model knows)
///   - model uncertain + Engram confident → alpha max (Engram helps)
///
/// When model_entropy is 0.0 (logprobs unavailable), falls back to NEST v1.
pub fn nest_adaptive_alpha(preds: &[(u32, f32)], base_alpha: f32) -> f32 {
    nest_adaptive_alpha_v2(preds, base_alpha, 0.0)
}

pub fn nest_adaptive_alpha_v2(preds: &[(u32, f32)], base_alpha: f32, model_entropy: f32) -> f32 {
    if preds.is_empty() {
        return 0.0; // No Engram prediction → no bias
    }
    // Engram confidence: how dominant is the top prediction?
    let max_prob = preds[0].1;
    let second_prob = if preds.len() > 1 { preds[1].1 } else { 0.0 };
    let engram_conf = max_prob / (max_prob + second_prob + 1e-10);
    let engram_gate = sigmoid((engram_conf - 0.5) / 0.1);

    // Model confidence: low entropy = confident = don't need Engram
    // model_entropy ~0 → model_need ~0 (model is sure, Engram quiet)
    // model_entropy >2 → model_need ~1 (model uncertain, Engram speaks)
    let model_need = if model_entropy > 0.001 {
        sigmoid((model_entropy - 1.0) / 0.5) // center at entropy=1.0
    } else {
        0.5 // logprobs unavailable → neutral (NEST v1 behavior)
    };

    base_alpha * engram_gate * model_need
}

/// Check whether NEST adaptive gating is enabled (default: ON).
///
/// Set `CHIMERE_ENGRAM_NEST=0` to disable and revert to fixed alpha.
fn nest_enabled() -> bool {
    match std::env::var("CHIMERE_ENGRAM_NEST") {
        Ok(v) => v != "0",
        Err(_) => true, // default ON
    }
}

/// Check whether DART speculative decoding via Engram is enabled.
///
/// Set `CHIMERE_ENGRAM_DART=1` to enable. Default: OFF.
/// DART uses the Engram n-gram tables as a FREE drafter (no extra model).
/// Only active during response phase (not thinking). Requires llama backend.
fn dart_enabled() -> bool {
    match std::env::var("CHIMERE_ENGRAM_DART") {
        Ok(v) => v == "1",
        Err(_) => false,
    }
}

/// Maximum number of draft tokens per DART speculation attempt.
fn dart_max_draft() -> usize {
    std::env::var("CHIMERE_DART_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5)
}

/// MTP scheduler statistics.
#[derive(Debug, Clone, Default)]
pub struct MtpStats {
    pub total_steps: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub tokens_generated: usize,
    /// DART speculative decoding stats (Engram n-gram drafter).
    pub dart_attempts: usize,
    pub dart_accepted: usize,
    pub dart_total_drafted: usize,
}

impl MtpStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_steps == 0 {
            return 0.0;
        }
        self.accepted as f32 / self.total_steps as f32
    }

    pub fn throughput_multiplier(&self) -> f32 {
        if self.total_steps == 0 {
            return 1.0;
        }
        self.tokens_generated as f32 / (self.total_steps + self.tokens_generated) as f32 * 2.0
    }

    pub fn dart_acceptance_rate(&self) -> f32 {
        if self.dart_total_drafted == 0 {
            return 0.0;
        }
        self.dart_accepted as f32 / self.dart_total_drafted as f32
    }

    pub fn dart_summary(&self) -> String {
        if self.dart_attempts == 0 {
            return "DART: inactive".to_string();
        }
        format!(
            "DART: {} attempts, {}/{} accepted ({:.1}%), avg draft len {:.1}",
            self.dart_attempts,
            self.dart_accepted,
            self.dart_total_drafted,
            self.dart_acceptance_rate() * 100.0,
            if self.dart_attempts > 0 { self.dart_total_drafted as f32 / self.dart_attempts as f32 } else { 0.0 },
        )
    }
}

/// Sample the argmax token from logits [1, vocab_size].
pub fn argmax(logits: &Tensor) -> Result<u32> {
    let logits_1d = logits.squeeze(0)?; // [vocab_size]
    let max_idx = logits_1d.argmax(0)?;
    let token: u32 = max_idx.to_scalar()?;
    Ok(token)
}

/// Sampling parameters for text generation.
///
/// Defaults match Qwen3.5 official recommendations for thinking+code mode.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,              // nucleus sampling (0.95 = keep top 95% probability mass)
    pub top_k: usize,            // top-K filtering (20 = consider only top 20 tokens)
    pub min_p: f32,              // min-P filtering: drop tokens with prob < min_p * max_prob (0.05 = ik_llama default)
    pub repetition_penalty: f32, // multiplicative penalty on repeated tokens (1.0 = disabled)
    pub presence_penalty: f32,   // additive penalty on tokens already seen (0.0 = disabled, 1.5 = Qwen3.5 default)
    pub dry_multiplier: f32,     // DRY penalty multiplier (0.0 = disabled, 0.8 = default)
    pub dry_base: f32,           // DRY exponential base (1.75 = default). penalty = multiplier * base^(L - min_length)
    pub dry_min_length: usize,   // minimum n-gram length to penalize (2 = bigrams and up)
    pub dry_penalty_last_n: i32, // context window for DRY scan (-1 = whole sequence, 0 = disabled)
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_min_length: 2,
            dry_penalty_last_n: -1,
        }
    }
}

impl SamplingParams {
    pub fn thinking_general() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            repetition_penalty: 1.0,
            presence_penalty: 0.0, // was 1.5 — kills code gen & long reasoning
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_min_length: 2,
            dry_penalty_last_n: -1,
        }
    }

    pub fn thinking_code() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.05,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_min_length: 2,
            dry_penalty_last_n: -1,
        }
    }

    pub fn nothink_general() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.05,
            repetition_penalty: 1.0,
            presence_penalty: 1.5,
            dry_multiplier: 0.8,
            dry_base: 1.75,
            dry_min_length: 2,
            dry_penalty_last_n: -1,
        }
    }
}

/// Sample with temperature + top-k + top-p + repetition penalty + presence penalty + DRY.
///
/// Pipeline order matches vLLM/HF convention:
///   1. Repetition penalty (multiplicative, on seen tokens)
///   2. Presence penalty (additive, on seen tokens)
///   2b. DRY penalty (penalizes tokens continuing repeated n-grams, Z-algorithm)
///   3. Temperature scaling
///   4. Top-K filtering
///   5. Softmax
///   6. Top-p (nucleus) filtering
///   7. Multinomial sampling
pub fn sample_advanced(
    logits: &Tensor,
    params: &SamplingParams,
    recent_tokens: &[u32],
) -> Result<u32> {
    if params.temperature <= 0.0 {
        return argmax(logits);
    }
    let mut logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;

    // Build set of unique seen tokens for penalty application
    let seen: std::collections::HashSet<u32> = recent_tokens.iter().copied().collect();

    // 1. Repetition penalty: multiplicative on logits of seen tokens
    if params.repetition_penalty != 1.0 {
        for &tok in &seen {
            let idx = tok as usize;
            if idx < logits_vec.len() {
                if logits_vec[idx] > 0.0 {
                    logits_vec[idx] /= params.repetition_penalty;
                } else {
                    logits_vec[idx] *= params.repetition_penalty;
                }
            }
        }
    }

    // 2. Presence penalty: additive subtraction on logits of seen tokens
    //    (OpenAI/vLLM convention: subtract penalty from logits of all tokens that appeared)
    if params.presence_penalty != 0.0 {
        for &tok in &seen {
            let idx = tok as usize;
            if idx < logits_vec.len() {
                logits_vec[idx] -= params.presence_penalty;
            }
        }
    }

    // 2b. DRY (Don't Repeat Yourself) penalty: penalize tokens that would continue
    //     an n-gram pattern already seen in the recent context.
    //     Uses the Z-algorithm (reversed) from llama.cpp / koboldcpp.
    if params.dry_multiplier != 0.0 && params.dry_base >= 1.0 && params.dry_penalty_last_n != 0 {
        let effective_last_n = if params.dry_penalty_last_n < 0 {
            recent_tokens.len()
        } else {
            (params.dry_penalty_last_n as usize).min(recent_tokens.len())
        };
        let dry_allowed_length = if params.dry_min_length > 0 {
            params.dry_min_length - 1
        } else {
            0
        };

        if effective_last_n > dry_allowed_length {
            // We work with the last `effective_last_n` tokens from recent_tokens.
            let start = recent_tokens.len().saturating_sub(effective_last_n);
            let ctx = &recent_tokens[start..];
            let n = ctx.len();

            // Z-algorithm in reverse: find suffix matches.
            // dry_repeat_count[i] = length of the match between the suffix starting
            // at the end and the substring ending at position i.
            let mut dry_repeat_count = vec![0i32; n];

            {
                let last = n - 1;
                let mut rt: usize = 0;
                let mut lt: usize = 0;

                for k in 1..n {
                    if k > rt {
                        // Naive computation outside Z-box
                        let mut m = 0usize;
                        while m + k < n && ctx[n - 1 - m] == ctx[n - 1 - (m + k)] {
                            m += 1;
                        }
                        dry_repeat_count[last - k] = (m as i32).min(effective_last_n as i32);
                        if m > 0 {
                            lt = k;
                            rt = k + m - 1;
                        }
                    } else {
                        let p = k - lt;
                        let right_part_len = (rt - k + 1) as i32;

                        if dry_repeat_count[last - p] < right_part_len {
                            dry_repeat_count[last - k] = dry_repeat_count[last - p].min(effective_last_n as i32);
                        } else {
                            let mut i = rt + 1;
                            while i < n && ctx[n - 1 - i] == ctx[n - 1 - (i - k)] {
                                i += 1;
                            }
                            let m = (i - k) as i32;
                            dry_repeat_count[last - k] = m.min(effective_last_n as i32);
                            lt = k;
                            rt = i - 1;
                        }
                    }
                }
            }

            // For each position with a repeat >= dry_allowed_length, the token
            // one step ahead would extend that repeat. Track the max repeat per token.
            let mut max_token_repeat: std::collections::HashMap<u32, i32> =
                std::collections::HashMap::new();

            for i in 0..(n - 1) {
                let repeat_len = dry_repeat_count[i];
                if repeat_len >= dry_allowed_length as i32 {
                    // The token that would continue this repetition is at
                    // ctx[n - 2 - i] (one step ahead of where the repeat ends).
                    let continuing_token = ctx[n - 2 - i];
                    let entry = max_token_repeat.entry(continuing_token).or_insert(0);
                    if repeat_len > *entry {
                        *entry = repeat_len;
                    }
                }
            }

            // Apply penalties. Clamp exponent to prevent f32 overflow.
            const FLOAT_MAX_LOG: f32 = 88.722_84;
            let max_exponent = if params.dry_base > 1.000_001 {
                (FLOAT_MAX_LOG / params.dry_base.ln()) as i32
            } else {
                0
            };

            for (&token, &repeat_len) in &max_token_repeat {
                let idx = token as usize;
                if idx < logits_vec.len() {
                    let mut exponent = repeat_len - dry_allowed_length as i32;
                    if max_exponent > 0 && exponent > max_exponent {
                        exponent = max_exponent;
                    }
                    let penalty = params.dry_multiplier * params.dry_base.powi(exponent);
                    logits_vec[idx] -= penalty;
                }
            }
        }
    }

    // 3. Temperature scaling
    let temp = params.temperature as f32;
    for l in logits_vec.iter_mut() {
        *l /= temp;
    }

    // 4. Top-K filtering: keep only the K highest logits, set rest to -inf
    if params.top_k > 0 && params.top_k < logits_vec.len() {
        let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate()
            .map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find the K-th largest value as threshold
        let threshold = indexed[params.top_k - 1].1;
        for l in logits_vec.iter_mut() {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // 5. Softmax
    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits_vec.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // 5b. Min-P filtering: drop tokens with prob < min_p * max_prob
    // This is very effective at preventing repetition (ik_llama default: 0.05)
    let probs = if params.min_p > 0.0 {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = params.min_p * max_prob;
        probs.iter().map(|&p| if p < threshold { 0.0 } else { p }).collect::<Vec<_>>()
    } else {
        probs
    };

    // 6. Top-p (nucleus) filtering
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0f32;
    let mut cutoff_idx = indexed.len();
    for (i, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= params.top_p as f32 {
            cutoff_idx = i + 1;
            break;
        }
    }
    let filtered = &indexed[..cutoff_idx];

    // 7. Re-normalize and multinomial sample
    let filtered_sum: f32 = filtered.iter().map(|&(_, p)| p).sum();
    use rand::Rng;
    let mut rng = rand::rng();
    let r: f32 = rng.random::<f32>() * filtered_sum;
    let mut acc = 0.0f32;
    for &(tok_id, prob) in filtered {
        acc += prob;
        if acc >= r {
            return Ok(tok_id as u32);
        }
    }
    Ok(filtered.last().map(|&(id, _)| id as u32).unwrap_or(0))
}

/// Simple temperature-only sampling (backward compat, no penalties).
pub fn sample_with_temperature(logits: &Tensor, temperature: f64) -> Result<u32> {
    sample_advanced(logits, &SamplingParams {
        temperature,
        top_p: 0.95,
        top_k: 20,
        min_p: 0.05,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        dry_multiplier: 0.0, // disabled for backward compat
        dry_base: 1.75,
        dry_min_length: 2,
        dry_penalty_last_n: 0,
    }, &[])
}

/// Run MTP-accelerated generation with optional EngramLookup biasing.
///
/// When `engram` is `Some`, the N-gram table is consulted after every forward
/// pass and its predictions are blended into the raw logits before argmax
/// via additive log-probability biasing (`logits[t] += alpha * ln(p_engram[t])`).
///
/// The engram integration is entirely optional and zero-cost when `None`:
/// the hot path is identical to `generate_with_mtp`.
///
/// # Arguments
/// - `model`: Loaded Qwen3.5 model.
/// - `prompt_token`: Last token of the prompt (generation starts here).
/// - `max_tokens`: Maximum tokens to generate.
/// - `state`: Mutable model recurrent state.
/// - `engram`: Optional multi-table N-gram lookup.
/// - `engram_alpha`: Blending strength (0.0 = no effect, 1.0 = full blend).
/// - `recent_window`: Number of most recent tokens to use as engram context
///                    (must be >= engram order; typically `engram.order()`).
pub fn generate_with_mtp_engram(
    model: &dyn ChimereModel,
    prompt_token: u32,
    max_tokens: usize,
    state: &mut crate::state::GdnRecurrentState,
    engram: Option<&crate::engram_lookup::MultiEngramLookup>,
    engram_alpha: f32,
    params: &SamplingParams,
) -> Result<(Vec<u32>, MtpStats)> {
    use crate::engram_lookup::MultiEngramLookup;
    use candle_core::Tensor;

    let mut tokens: Vec<u32> = Vec::new();
    let mut stats = MtpStats::default();
    let mut current_token = prompt_token;
    // Track thinking phase to disable Engram during reasoning
    const THINK_START_TOKEN: u32 = 248068;
    const THINK_END_TOKEN: u32 = 248069;
    let mut in_thinking = current_token == THINK_START_TOKEN;

    // recent_context tracks the last N tokens for engram lookup.
    let engram_order = engram.map(|e| e.order()).unwrap_or(0);
    let context_capacity = engram_order.max(1) + max_tokens;
    let mut recent_context: Vec<u32> = Vec::with_capacity(context_capacity);
    recent_context.push(prompt_token);

    while tokens.len() < max_tokens {
        let ForwardOutput { logits: logits_tensor, mtp_logits: _mtp } =
            forward_token_gdn(model, current_token, state)?;

        // Engram bias: ONLY during response, NOT during thinking phase
        let sampled_token = if let Some(eng) = engram {
            let mut logits_vec: Vec<f32> = logits_tensor.squeeze(0)?.to_vec1()?;

            if !in_thinking {
                let predictions = eng.lookup(&recent_context);
                if !predictions.is_empty() {
                    // Compute model entropy from logits for NEST v2 gating
                    let model_entropy = {
                        let max_l = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let sum_exp: f64 = logits_vec.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
                        let log_z = max_l as f64 + sum_exp.ln();
                        let mut ent: f64 = 0.0;
                        for &l in logits_vec.iter() {
                            let p = ((l as f64) - log_z).exp();
                            if p > 1e-10 { ent -= p * p.ln(); }
                        }
                        ent as f32
                    };
                    let alpha = if nest_enabled() {
                        nest_adaptive_alpha_v2(&predictions, engram_alpha, model_entropy)
                    } else {
                        engram_alpha
                    };
                    MultiEngramLookup::bias_logits(&mut logits_vec, &predictions, alpha);
                }
            }

            // Re-wrap as Tensor for sampling.
            let biased = Tensor::from_vec(
                logits_vec,
                logits_tensor.squeeze(0)?.shape().clone(),
                logits_tensor.device(),
            )?
            .unsqueeze(0)?;

            if params.temperature <= 0.0 {
                argmax(&biased)?
            } else {
                sample_advanced(&biased, params, &tokens)?
            }
        } else if params.temperature <= 0.0 {
            argmax(&logits_tensor)?
        } else {
            sample_advanced(&logits_tensor, params, &tokens)?
        };

        tokens.push(sampled_token);
        stats.tokens_generated += 1;

        // Track thinking phase transitions
        if sampled_token == THINK_END_TOKEN { in_thinking = false; }
        if sampled_token == THINK_START_TOKEN { in_thinking = true; }

        // Update rolling context window (increased to 12 for better disambiguation).
        recent_context.push(sampled_token);
        let keep = (engram_order + 1).max(12);
        if recent_context.len() > keep {
            let drain_count = recent_context.len() - keep;
            recent_context.drain(..drain_count);
        }

        // Check for EOS
        if sampled_token == 248046 || sampled_token == 248044 {
            break;
        }

        current_token = sampled_token;

        if tokens.len() >= max_tokens {
            break;
        }
    }

    Ok((tokens, stats))
}

/// Run MTP-accelerated generation with full sampling parameters.
///
/// Supports temperature, top-k, top-p, repetition_penalty, and presence_penalty.
/// When `params.temperature <= 0.0`, uses greedy argmax (deterministic).
///
/// Thinking mode: detects `</think>` (token 248069) and resets the response
/// token counter. `max_tokens` applies to the RESPONSE only — thinking gets
/// a separate budget (`CHIMERE_THINK_BUDGET`, default 8192).
///
/// Returns generated tokens, MTP statistics, and per-token packed logprobs.
///
/// The packed logprobs vector contains one entry per generated token (when using
/// the fast C++ sampler path). Each entry is in packed format:
/// `[token_id, n_top, t0, lp0, t1, lp1, ..., t4, lp4]` (12 floats).
/// When the slow sampling path is used, entries may be missing.
pub fn generate_with_mtp(
    model: &dyn ChimereModel,
    prompt_token: u32,
    max_tokens: usize,
    state: &mut crate::state::GdnRecurrentState,
    params: &SamplingParams,
    tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<(Vec<u32>, MtpStats, Vec<Vec<f32>>)> {
    let total_budget = std::env::var("CHIMERE_THINK_BUDGET")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(max_tokens);

    let mut tokens = Vec::new();
    let mut stats = MtpStats::default();
    let mut collected_logprobs: Vec<Vec<f32>> = Vec::new();
    let mut current_token = prompt_token;
    // Start in thinking mode ONLY if prompt ended with <think> token (248068).
    const THINK_START: u32 = 248068;
    let mut thinking = prompt_token == THINK_START;

    // Engram: O(1) hash lookup for n-gram predictions → logit biasing
    // FIXES (Mar 27 ablation): disable during thinking, reduce alpha, longer context
    // Step 7: tables are keyed on Qwen3.5 token IDs — refuse to load on any
    // other arch (otherwise Nemotron tokens would index Qwen vocab → garbage
    // or crash if vocab sizes differ).
    let engram_lookup = if model.arch() == crate::chimere_model::ModelArch::Qwen35A3B {
        crate::engram_lookup::MultiEngramLookup::from_env()
    } else {
        if std::env::var("CHIMERE_ENGRAM_DIR").is_ok()
            || std::env::var("CHIMERE_ENGRAM_FILE").is_ok()
        {
            eprintln!(
                "[ENGRAM] Disabled for arch {} — tokenizer mismatch with \
                 Qwen3.5-only engram tables.",
                model.arch().name()
            );
        }
        None
    };
    let engram_alpha: f32 = std::env::var("CHIMERE_ENGRAM_ALPHA")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.1); // was 0.3 — too aggressive
    if let Some(ref e) = engram_lookup {
        eprintln!("[ENGRAM] Active in generate_with_mtp: {} alpha={} nest={}", e.summary(), engram_alpha, nest_enabled());
    }

    const THINK_END: u32 = 248069;
    const IM_END: u32 = 248046;
    const EOS: u32 = 248044;

    loop {
        // Engram bias: ONLY during response phase, NOT during thinking
        // Ablation showed thinking-phase bias degrades factual recall by ~8pp
        if !thinking {
            if let Some(ref engram) = engram_lookup {
                let ctx_start = tokens.len().saturating_sub(12); // was 4 — longer context
                let mut context: Vec<u32> = tokens[ctx_start..].to_vec();
                context.push(current_token);
                let preds = engram.lookup(&context);
                if !preds.is_empty() {
                    let alpha = if nest_enabled() {
                        nest_adaptive_alpha(&preds, engram_alpha)
                    } else {
                        engram_alpha
                    };
                    let biases: Vec<(u32, f32)> = preds.iter()
                        .map(|&(tid, prob)| (tid, alpha * (prob + 1e-10).ln()))
                        .collect();
                    model.llama_set_engram_bias(&biases);
                } else {
                    model.llama_clear_engram_bias();
                }
            }
        } else {
            // Clear any lingering bias during thinking
            model.llama_clear_engram_bias();
        }

        let ForwardOutput { logits, mtp_logits: _ } =
            forward_token_gdn(model, current_token, state)?;

        // Fast path: if logits shape is (1,12), it's a pre-sampled token with logprobs
        // Format: [token_id, n_top, t0, lp0, t1, lp1, ..., t4, lp4]
        let dims = logits.dims();
        let is_fast_sampled = dims.len() == 2 && dims[0] == 1 && dims[1] == 12;

        let main_token = if is_fast_sampled {
            let packed: Vec<f32> = logits.flatten_all()?.to_vec1()?;
            packed[0] as u32
        } else if !thinking {
            // In response phase, suppress <think> and </think> to prevent re-entry
            let logits_1d = logits.squeeze(0)?;
            let mut logits_cpu: Vec<f32> = logits_1d.to_vec1()?;
            logits_cpu[THINK_START as usize] = f32::NEG_INFINITY;
            logits_cpu[THINK_END as usize] = f32::NEG_INFINITY;
            if params.temperature <= 0.0 {
                logits_cpu.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32).unwrap_or(0)
            } else {
                let suppressed = candle_core::Tensor::new(&logits_cpu[..], logits.device())?
                    .unsqueeze(0)?;
                sample_advanced(&suppressed, params, &tokens)?
            }
        } else if params.temperature <= 0.0 {
            argmax(&logits)?
        } else {
            sample_advanced(&logits, params, &tokens)?
        };

        // ABF: force </think> after a thinking budget (non-streaming path)
        const THINKING_BUDGET_NS: usize = 4096;
        let main_token = if thinking && stats.tokens_generated >= THINKING_BUDGET_NS {
            eprintln!("[ABF] Thinking budget exhausted ({} tokens, non-stream), forcing </think>", THINKING_BUDGET_NS);
            THINK_END
        } else {
            main_token
        };

        tokens.push(main_token);
        stats.tokens_generated += 1;

        // Collect packed logprobs from fast-sampler path (stored by forward_token).
        if let Some(packed) = model.take_last_packed_logprobs() {
            collected_logprobs.push(packed);
        }

        // Detect </think> by token ID (fast) or text matching (fallback)
        if thinking {
            if main_token == THINK_END {
                thinking = false;
                // Suppress </think> via logit bias so C++ sampler won't re-generate it
                if is_fast_sampled {
                    model.llama_set_logit_bias(THINK_END, f32::NEG_INFINITY);
                }
            } else if let Some(tok) = tokenizer {
                let recent_start = tokens.len().saturating_sub(10);
                if let Ok(text) = tok.decode(&tokens[recent_start..], false) {
                    if text.contains("</think>") {
                        thinking = false;
                        if is_fast_sampled {
                            model.llama_set_logit_bias(THINK_END, f32::NEG_INFINITY);
                        }
                    }
                }
            }
        }

        // Total budget check: thinking + response combined (like ik_llama)
        if tokens.len() >= total_budget {
            break;
        }

        // Check for EOS
        if main_token == IM_END || main_token == EOS {
            break;
        }

        current_token = main_token;
    }

    Ok((tokens, stats, collected_logprobs))
}

/// DART speculative decoding: verify draft tokens from Engram n-gram tables.
///
/// After the main loop samples `main_token`, this function feeds
/// `[main_token, D1, D2, ..., DK]` through the model in a single batch decode.
/// The model has NOT yet processed `main_token`, so we include it in the batch.
///
/// Verification logic (standard speculative decoding):
/// - `logits[0]` = model output after processing `main_token` → predicts pos P+2
///   → Check: `argmax(logits[0]) == D1?`
/// - `logits[1]` = model output after processing `D1` → predicts pos P+3
///   → Check: `argmax(logits[1]) == D2?`
/// - First mismatch → correction token = argmax at that position.
/// - If all K drafts match → `logits[K]` gives a bonus token.
///
/// Returns `(accepted_draft_tokens, correction_token)`:
/// - `accepted_draft_tokens`: draft tokens the model confirmed (may be empty)
/// - `correction_token`: model's actual prediction at the first mismatch
///   (0 if all drafts accepted — caller gets bonus from last logits position)
///
/// On return, the KV cache and pos are updated correctly:
/// - Accepted tokens (main_token + drafts + correction) are in the KV cache.
/// - Rejected draft positions have been removed via `kv_cache_seq_rm`.
fn dart_verify_drafts(
    model: &dyn ChimereModel,
    main_token: u32,
    draft_tokens: &[u32],
    state: &mut crate::state::GdnRecurrentState,
) -> Result<(Vec<u32>, Option<u32>)> {
    if draft_tokens.is_empty() {
        return Err(candle_core::Error::Msg("dart_verify_drafts: empty draft".into()));
    }

    // Trait method returns Option<RefMut<Option<LlamaForward>>>: first the
    // outer Option flags whether the model has any libllama backend at all
    // (None for pure-Rust models), the inner Option whether init_llama_forward
    // has been called yet. Both must be Some for DART to run.
    let mut llama_ref = model
        .llama_forward_mut()
        .ok_or_else(|| candle_core::Error::Msg("DART requires llama backend".into()))?;
    let llama = match llama_ref.as_mut() {
        Some(l) => l,
        None => return Err(candle_core::Error::Msg("DART requires llama backend".into())),
    };

    let saved_pos = llama.pos();

    // Build verification batch: [main_token, D1, D2, ..., DK]
    // main_token hasn't been processed yet — include it so the model
    // advances through the correct sequence.
    let mut verify_batch: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
    verify_batch.push(main_token);
    verify_batch.extend_from_slice(draft_tokens);

    // Batch decode: returns logits at each position in the batch.
    // On failure, clean up any KV entries that were added.
    let all_logits = match llama.forward_batch_verify(&verify_batch) {
        Ok(logits) => logits,
        Err(e) => {
            // Batch decode failed — remove any KV entries that may have been added
            // and leave pos unchanged so the caller can retry with forward_token.
            let batch_end = saved_pos + verify_batch.len() as i32;
            llama.kv_cache_seq_rm(saved_pos, batch_end);
            return Err(candle_core::Error::Msg(e));
        }
    };

    // Verify draft tokens against model predictions.
    // logits[i] = model output after processing verify_batch[i],
    //             predicting what should come at position saved_pos + i + 1.
    // We check: argmax(logits[i]) == draft_tokens[i] (= verify_batch[i+1])
    let mut accepted = Vec::new();
    let mut correction: Option<u32> = None;

    for i in 0..draft_tokens.len() {
        if i >= all_logits.len() {
            break;
        }
        let logits = &all_logits[i]; // logits after verify_batch[i]
        let model_choice = logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);

        if draft_tokens[i] == model_choice {
            accepted.push(draft_tokens[i]);
        } else {
            correction = Some(model_choice);
            break;
        }
    }

    // If ALL drafts accepted and we have logits for the last position,
    // extract the bonus token from logits[K] (model predicts what follows all drafts).
    if accepted.len() == draft_tokens.len() && all_logits.len() > draft_tokens.len() {
        let bonus_logits = &all_logits[draft_tokens.len()];
        let bonus = bonus_logits.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        correction = Some(bonus);
    }

    // KV cache cleanup.
    //
    // We keep KV entries for: main_token + accepted draft tokens.
    // The correction token is NOT in the KV cache — it was only argmax'd from
    // logits, never actually processed as input. The caller must feed it through
    // forward_token() to create the correct KV entry.
    //
    // Remove everything from the first rejected position onwards.
    let keep_count = 1 + accepted.len(); // main_token + accepted drafts
    let batch_len = verify_batch.len();

    if keep_count < batch_len {
        let rm_start = saved_pos + keep_count as i32;
        let rm_end = saved_pos + batch_len as i32;
        llama.kv_cache_seq_rm(rm_start, rm_end);
    }

    // Advance pos past the tokens we're keeping.
    // The correction token will be processed at this new position by the caller.
    llama.accept_draft_tokens(keep_count);

    // Register main_token and accepted drafts in sampler for DRY/repetition.
    // The correction token is NOT registered here — the caller will register it
    // when it feeds it through forward_token() and samples from it.
    llama.sampler_accept(main_token);
    for &tok in &accepted {
        llama.sampler_accept(tok);
    }

    // Sync state position.
    state.position = llama.pos() as usize;

    Ok((accepted, correction))
}

/// Run MTP-accelerated generation with a per-token callback for streaming.
///
/// Identical to `generate_with_mtp` but calls `token_callback(token_id, decoded_text, is_thinking)`
/// after each generated token. If the callback returns `false`, generation stops early.
///
/// The `decoded_text` is the incremental text decoded from the latest token(s).
/// For tokens in the middle of a multi-byte UTF-8 sequence, decoded_text may be
/// empty until the sequence completes.
///
/// # Arguments
/// - `model`: Loaded Qwen3.5 model.
/// - `prompt_token`: Last token of the prompt (generation starts here).
/// - `max_tokens`: Maximum tokens to generate (response budget, excludes thinking).
/// - `state`: Mutable model recurrent state.
/// - `params`: Sampling parameters.
/// - `tokenizer`: Optional tokenizer for `</think>` detection and text decoding.
/// - `token_callback`: Called after each token. Args: (token_id, decoded_text, is_thinking).
///                      Return `false` to stop generation.
///
/// Returns MTP statistics (token IDs are delivered via the callback).
pub fn generate_with_mtp_streaming(
    model: &dyn ChimereModel,
    prompt_token: u32,
    max_tokens: usize,
    state: &mut crate::state::GdnRecurrentState,
    params: &SamplingParams,
    tokenizer: Option<&tokenizers::Tokenizer>,
    thinking_active: bool,
    token_callback: &mut dyn FnMut(u32, &str, bool) -> bool,
) -> Result<MtpStats> {
    let total_budget = std::env::var("CHIMERE_THINK_BUDGET")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(max_tokens);

    let mut tokens = Vec::new();
    let mut stats = MtpStats::default();
    let mut current_token = prompt_token;
    // Thinking mode: explicit parameter from the handler (no more per-request
    // env var mutation — that was a global race under multi-slot native mode).
    // Falls back to prompt_token detection for direct callers that pass false.
    const THINK_START: u32 = 248068; // <think>
    let mut thinking = thinking_active || prompt_token == THINK_START;

    // Engram: O(1) hash lookup for n-gram predictions → logit biasing
    // Step 7: gate on arch (Qwen3.5 only) — see generate_with_mtp for rationale.
    let engram_lookup = if model.arch() == crate::chimere_model::ModelArch::Qwen35A3B {
        crate::engram_lookup::MultiEngramLookup::from_env()
    } else {
        if std::env::var("CHIMERE_ENGRAM_DIR").is_ok()
            || std::env::var("CHIMERE_ENGRAM_FILE").is_ok()
        {
            eprintln!(
                "[ENGRAM] Disabled for arch {} — tokenizer mismatch with \
                 Qwen3.5-only engram tables.",
                model.arch().name()
            );
        }
        None
    };
    let engram_alpha: f32 = std::env::var("CHIMERE_ENGRAM_ALPHA")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0.1); // was 0.3
    if let Some(ref e) = engram_lookup {
        eprintln!("[ENGRAM] Active in generate_with_mtp_streaming: {} alpha={} nest={}", e.summary(), engram_alpha, nest_enabled());
    }

    // DART speculative decoding: use Engram n-gram tables as a FREE drafter.
    // Only enabled when CHIMERE_ENGRAM_DART=1 AND engram tables are loaded AND llama backend active.
    let dart_active = dart_enabled()
        && engram_lookup.is_some()
        && model.llama_forward_active();
    let dart_steps = dart_max_draft();
    if dart_active {
        eprintln!("[DART] Engram speculative decoding active: max_draft={}", dart_steps);
    }

    // Token IDs for stop/phase detection
    const THINK_END: u32 = 248069; // </think>
    const IM_END: u32 = 248046;    // <|im_end|>
    const EOS: u32 = 248044;       // <|endoftext|>

    loop {
        // Engram bias: ONLY during response, NOT thinking
        if !thinking {
            if let Some(ref engram) = engram_lookup {
                let ctx_start = tokens.len().saturating_sub(12); // was 4
                let mut context: Vec<u32> = tokens[ctx_start..].to_vec();
                context.push(current_token);
                let preds = engram.lookup(&context);
                if !preds.is_empty() {
                    let alpha = if nest_enabled() {
                        nest_adaptive_alpha(&preds, engram_alpha)
                    } else {
                        engram_alpha
                    };
                    let biases: Vec<(u32, f32)> = preds.iter()
                        .map(|&(tid, prob)| (tid, alpha * (prob + 1e-10).ln()))
                        .collect();
                    model.llama_set_engram_bias(&biases);
                } else {
                    model.llama_clear_engram_bias();
                }
            }
        } else {
            model.llama_clear_engram_bias();
        }

        let ForwardOutput { logits, mtp_logits: _ } =
            forward_token_gdn(model, current_token, state)?;

        // Fast path: if logits shape is (1,12), it's a pre-sampled token with logprobs
        // Format: [token_id, n_top, t0, lp0, t1, lp1, ..., t4, lp4]
        let dims = logits.dims();
        let is_fast_sampled = dims.len() == 2 && dims[0] == 1 && dims[1] == 12;

        let main_token = if is_fast_sampled {
            let packed: Vec<f32> = logits.flatten_all()?.to_vec1()?;
            packed[0] as u32
        } else if !thinking {
            // In response phase, suppress <think> and </think> to prevent re-entry
            let logits_1d = logits.squeeze(0)?;
            let mut logits_cpu: Vec<f32> = logits_1d.to_vec1()?;
            logits_cpu[THINK_START as usize] = f32::NEG_INFINITY;
            logits_cpu[THINK_END as usize] = f32::NEG_INFINITY;
            if params.temperature <= 0.0 {
                logits_cpu.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u32).unwrap_or(0)
            } else {
                let suppressed = candle_core::Tensor::new(&logits_cpu[..], logits.device())?
                    .unsqueeze(0)?;
                sample_advanced(&suppressed, params, &tokens)?
            }
        } else if params.temperature <= 0.0 {
            argmax(&logits)?
        } else {
            sample_advanced(&logits, params, &tokens)?
        };

        // ABF: force </think> after a thinking budget to prevent infinite think loops
        // IQ3_S quantization sometimes fails to generate the </think> token naturally
        const THINKING_BUDGET: usize = 4096;
        let main_token = if thinking && stats.tokens_generated >= THINKING_BUDGET {
            eprintln!("[ABF] Thinking budget exhausted ({} tokens), forcing </think>", THINKING_BUDGET);
            THINK_END // Force the </think> token
        } else {
            main_token
        };

        tokens.push(main_token);
        stats.tokens_generated += 1;

        // Detect </think> by token ID (fast) or text matching (fallback)
        let was_thinking = thinking;
        if thinking {
            if main_token == THINK_END {
                thinking = false;
                // Suppress </think> via logit bias so C++ sampler won't re-generate it
                if is_fast_sampled {
                    model.llama_set_logit_bias(THINK_END, f32::NEG_INFINITY);
                }
            } else if let Some(tok) = tokenizer {
                let recent_start = tokens.len().saturating_sub(10);
                if let Ok(text) = tok.decode(&tokens[recent_start..], false) {
                    if text.contains("</think>") {
                        thinking = false;
                        if is_fast_sampled {
                            model.llama_set_logit_bias(THINK_END, f32::NEG_INFINITY);
                        }
                    }
                }
            }
        }

        // Decode the incremental text for the callback
        let decoded = if let Some(tok) = tokenizer {
            // Decode the last few tokens and diff against previous to get incremental text.
            // This handles multi-byte UTF-8 sequences that span multiple tokens.
            let last_few_start = tokens.len().saturating_sub(3);
            let current_decode = tok.decode(&tokens[last_few_start..], true).unwrap_or_default();
            let prev_decode = if last_few_start < tokens.len() - 1 {
                tok.decode(&tokens[last_few_start..tokens.len()-1], true).unwrap_or_default()
            } else {
                String::new()
            };
            if current_decode.len() > prev_decode.len() {
                current_decode[prev_decode.len()..].to_string()
            } else {
                // Fallback: decode single token
                tok.decode(&[main_token], true).unwrap_or_default()
            }
        } else {
            String::new()
        };

        // Call the callback with token_id, decoded text, and thinking state.
        // Use was_thinking: if we JUST exited thinking on this token, this token
        // (the </think> closer) is still part of thinking content.
        let should_continue = token_callback(main_token, &decoded, was_thinking);
        if !should_continue {
            break;
        }

        // Check for EOS before attempting DART (no point drafting after EOS)
        if main_token == IM_END || main_token == EOS {
            break;
        }

        // Total budget check: thinking + response combined (like ik_llama)
        if tokens.len() >= total_budget {
            break;
        }

        // -----------------------------------------------------------
        // DART speculative decoding: attempt FREE multi-token drafting
        // from Engram n-gram tables during response phase only.
        // -----------------------------------------------------------
        if dart_active && !thinking {
            if let Some(ref engram) = engram_lookup {
                let ctx_start = tokens.len().saturating_sub(12);
                let mut dart_context: Vec<u32> = tokens[ctx_start..].to_vec();
                dart_context.push(main_token);

                let draft = engram.draft_sequence(&dart_context, dart_steps);

                if !draft.is_empty() {
                    stats.dart_attempts += 1;
                    stats.dart_total_drafted += draft.len();

                    // Clear engram bias during verification (pure model check)
                    model.llama_clear_engram_bias();

                    match dart_verify_drafts(model, main_token, &draft, state) {
                        Ok((accepted, correction)) => {
                            // dart_verify_drafts already processed main_token
                            // in the batch, so do NOT re-process it.

                            // Emit accepted draft tokens via callback
                            let mut dart_aborted = false;
                            for &tok in &accepted {
                                tokens.push(tok);
                                stats.tokens_generated += 1;
                                stats.dart_accepted += 1;

                                let dec = if let Some(t) = tokenizer {
                                    let s = tokens.len().saturating_sub(3);
                                    let cur = t.decode(&tokens[s..], true).unwrap_or_default();
                                    let prev = if s < tokens.len() - 1 {
                                        t.decode(&tokens[s..tokens.len()-1], true).unwrap_or_default()
                                    } else { String::new() };
                                    if cur.len() > prev.len() { cur[prev.len()..].to_string() }
                                    else { t.decode(&[tok], true).unwrap_or_default() }
                                } else { String::new() };

                                if !token_callback(tok, &dec, false) {
                                    dart_aborted = true;
                                    break;
                                }

                                if tok == IM_END || tok == EOS {
                                    dart_aborted = true;
                                    break;
                                }
                            }

                            if dart_aborted {
                                break;
                            }

                            // Set next token for the main loop.
                            // The correction token needs to go through forward_token()
                            // to create its KV entry, so we set it as current_token.
                            // correction is always Some in practice (batch has K+1 logits).
                            current_token = correction.unwrap_or_else(|| {
                                eprintln!("[DART] WARNING: no correction token (unexpected)");
                                // Safety fallback: use last accepted or main_token.
                                // This token is already in KV cache, so forward_token
                                // will effectively "re-process" it. Not ideal but safe.
                                accepted.last().copied().unwrap_or(main_token)
                            });

                            if tokens.len() >= total_budget {
                                break;
                            }
                            continue;
                        }
                        Err(e) => {
                            // DART verification failed — fall through to normal AR
                            eprintln!("[DART] Verification error (non-fatal): {}", e);
                        }
                    }
                }
            }
        }

        current_token = main_token;
    }

    if stats.dart_attempts > 0 {
        eprintln!("[DART] {}", stats.dart_summary());
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Generic-arch generation (libllama-only path, Step 7)
// ---------------------------------------------------------------------------
//
// Sibling of `generate_with_mtp` for architectures whose state lives
// entirely inside the libllama FFI context (Mamba-1, Mamba-2,
// Nemotron-H MoE, ...). Intentionally minimal: no MTP, no DART, no
// engram, no `</think>` detection. EOS is configurable via
// `CHIMERE_GENERIC_EOS` (comma-separated list, default `[2]`).

/// Run greedy / sampled generation on a libllama-backed model.
///
/// Returns `(generated_token_ids, MtpStats, Vec<Vec<f32>>)`. The third
/// field is always empty for the Generic path — packed logprobs are not
/// produced because Generic models do not go through the C++ fast
/// sampler. Returned for ABI parity with `generate_with_mtp`.
pub fn generate_with_mtp_generic(
    model: &dyn ChimereModel,
    prompt_token: u32,
    max_tokens: usize,
    params: &SamplingParams,
    _tokenizer: Option<&tokenizers::Tokenizer>,
) -> Result<(Vec<u32>, MtpStats, Vec<Vec<f32>>)> {
    let mut tokens: Vec<u32> = Vec::new();
    let mut stats = MtpStats::default();
    let logprobs: Vec<Vec<f32>> = Vec::new();
    let mut current = prompt_token;

    // EOS detection: caller can override the default per-arch via
    // CHIMERE_GENERIC_EOS (comma-separated u32 list). Default = 2
    // (Nemotron-H / GPT-2 `<|endoftext|>`). Mamba-1 typically uses 0.
    let eos_tokens: Vec<u32> = std::env::var("CHIMERE_GENERIC_EOS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect::<Vec<u32>>()
        })
        .unwrap_or_else(|| vec![2]);

    // Push the prompt's last token immediately so the caller observes the
    // same "first generated token = prompt_token" semantics as
    // `generate_text_generic`. The first forward call below uses
    // `current = prompt_token` to predict what comes after it.
    for _ in 0..max_tokens {
        let out = forward_token_generic(model, current)?;
        let next = if params.temperature <= 0.0 {
            argmax(&out.logits)?
        } else {
            sample_advanced(&out.logits, params, &tokens)?
        };
        tokens.push(next);
        stats.tokens_generated += 1;
        if eos_tokens.contains(&next) {
            break;
        }
        current = next;
    }

    Ok((tokens, stats, logprobs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mtp_stats_default() {
        let stats = MtpStats::default();
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.throughput_multiplier(), 1.0);
    }

    #[test]
    fn test_mtp_stats_tracking() {
        let mut stats = MtpStats::default();
        stats.total_steps = 10;
        stats.accepted = 8;
        stats.rejected = 2;
        stats.tokens_generated = 18; // 10 main + 8 accepted drafts
        assert!((stats.acceptance_rate() - 0.8).abs() < 0.01);
        assert!(stats.throughput_multiplier() > 1.0);
    }

    #[test]
    fn test_dart_stats() {
        let mut stats = MtpStats::default();
        assert_eq!(stats.dart_acceptance_rate(), 0.0);
        assert_eq!(stats.dart_summary(), "DART: inactive");

        stats.dart_attempts = 5;
        stats.dart_total_drafted = 25; // 5 drafts per attempt
        stats.dart_accepted = 15;
        assert!((stats.dart_acceptance_rate() - 0.6).abs() < 0.01);
        assert!(stats.dart_summary().contains("60.0%"));
        assert!(stats.dart_summary().contains("15/25"));
    }

    #[test]
    fn test_dart_disabled_by_default() {
        // CHIMERE_ENGRAM_DART not set → disabled
        std::env::remove_var("CHIMERE_ENGRAM_DART");
        assert!(!dart_enabled());
    }

    #[test]
    fn test_dart_max_draft_default() {
        std::env::remove_var("CHIMERE_DART_STEPS");
        assert_eq!(dart_max_draft(), 5);
    }

    #[test]
    fn test_mtp_generate_real() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-Q4_0-MTP.gguf",
            home
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: GGUF not found at {}", path);
            return;
        }

        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        eprintln!("[MTP TEST] Using device: {:?}", device);

        let model = crate::qwen35_model::Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");

        let mut state = crate::state::GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        let start = std::time::Instant::now();
        let (tokens, stats, _logprobs) = generate_with_mtp(&model, 1, 10, &mut state, &SamplingParams { temperature: 0.0, ..SamplingParams::default() }, None)
            .expect("MTP generation failed");
        let elapsed = start.elapsed();

        eprintln!("[MTP TEST] Generated {} tokens in {:.2}s", tokens.len(), elapsed.as_secs_f64());
        eprintln!("[MTP TEST] Tokens: {:?}", tokens);
        eprintln!("[MTP TEST] Stats: {:?}", stats);
        eprintln!("[MTP TEST] Acceptance rate: {:.1}%", stats.acceptance_rate() * 100.0);
        eprintln!("[MTP TEST] Throughput multiplier: {:.2}x", stats.throughput_multiplier());

        assert!(!tokens.is_empty(), "Should generate at least 1 token");
        assert!(tokens.len() <= 10, "Should not exceed max_tokens");
    }

    #[test]
    fn test_logits_dump() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-Q4_0-MTP.gguf",
            home
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: GGUF not found");
            return;
        }

        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        let model = crate::qwen35_model::Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");
        let mut state = crate::state::GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        let (logits, mtp_logits) = model.forward_token(1, &mut state)
            .expect("Forward failed");

        // Analyze logits
        let logits_cpu: Vec<f32> = logits.squeeze(0).unwrap().to_vec1().unwrap();
        let n = logits_cpu.len();
        let min = logits_cpu.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = logits_cpu.iter().sum::<f32>() / n as f32;
        let variance: f32 = logits_cpu.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let std = variance.sqrt();
        let n_pos = logits_cpu.iter().filter(|&&x| x > 0.0).count();
        let n_neg = logits_cpu.iter().filter(|&&x| x < 0.0).count();
        let n_zero = logits_cpu.iter().filter(|&&x| x == 0.0).count();
        let n_nan = logits_cpu.iter().filter(|x| x.is_nan()).count();
        let n_inf = logits_cpu.iter().filter(|x| x.is_infinite()).count();

        eprintln!("=== LOGITS ANALYSIS (token 1 → logits) ===");
        eprintln!("vocab_size: {}", n);
        eprintln!("min: {:.4}, max: {:.4}, mean: {:.4}, std: {:.4}", min, max, mean, std);
        eprintln!("positive: {}, negative: {}, zero: {}, NaN: {}, Inf: {}", n_pos, n_neg, n_zero, n_nan, n_inf);

        // Top-10 tokens
        let mut indexed: Vec<(usize, f32)> = logits_cpu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("\nTop-10 tokens:");
        for (i, (tok, logit)) in indexed.iter().take(10).enumerate() {
            eprintln!("  {}: token {} = {:.4}", i+1, tok, logit);
        }
        eprintln!("\nBottom-5 tokens:");
        for (i, (tok, logit)) in indexed.iter().rev().take(5).enumerate() {
            eprintln!("  {}: token {} = {:.4}", i+1, tok, logit);
        }

        // MTP logits
        if let Some(mtp) = mtp_logits {
            let mtp_cpu: Vec<f32> = mtp.squeeze(0).unwrap().to_vec1().unwrap();
            let mtp_max = mtp_cpu.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mtp_argmax = mtp_cpu.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            eprintln!("\nMTP top-1: token {} = {:.4}", mtp_argmax, mtp_max);
        }

        assert!(!logits_cpu.iter().any(|x| x.is_nan()), "NaN in logits!");
        assert!(std > 0.001, "Logits have near-zero variance: std={}", std);
    }

    #[test]
    fn test_logits_evolution() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-Q4_0-MTP.gguf",
            home
        );
        if !std::path::Path::new(&path).exists() {
            eprintln!("Skipping: GGUF not found");
            return;
        }

        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        let model = crate::qwen35_model::Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");
        let mut state = crate::state::GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        // Feed tokens and compare the argmax logits each time
        // token 1 = BOS-like, then diverse tokens to verify state evolution
        let test_tokens = [1u32, 328, 220, 198, 1535];
        for (step, &tok) in test_tokens.iter().enumerate() {
            let (logits, _) = model.forward_token(tok, &mut state)
                .expect("Forward failed");
            let logits_cpu: Vec<f32> = logits.squeeze(0).unwrap().to_vec1().unwrap();
            let mut indexed: Vec<(usize, f32)> = logits_cpu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            eprintln!("[STEP {}] token_in={} top5: {} {:.2} | {} {:.2} | {} {:.2} | {} {:.2} | {} {:.2}",
                step, tok,
                indexed[0].0, indexed[0].1,
                indexed[1].0, indexed[1].1,
                indexed[2].0, indexed[2].1,
                indexed[3].0, indexed[3].1,
                indexed[4].0, indexed[4].1);
        }
    }

    #[test]
    fn test_mtp_multi_token_prompt() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-Q4_0-MTP.gguf",
            home
        );
        if !std::path::Path::new(&path).exists() { return; }

        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        let model = crate::qwen35_model::Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");
        let mut state = crate::state::GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");

        // Process a multi-token "prompt" first (diverse tokens to prime the state)
        // These approximate: <|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n
        let prompt_tokens: Vec<u32> = vec![
            248045, // <|im_start|>
            882,    // user
            198,    // \n
            9707,   // Hello
            248046, // <|im_end|>
            198,    // \n
            248045, // <|im_start|>
            78191,  // assistant
            198,    // \n
        ];

        // Process prompt (no generation, just feed tokens to build state)
        let start = std::time::Instant::now();
        for &tok in &prompt_tokens {
            let _ = model.forward_token(tok, &mut state).expect("Prompt forward failed");
        }
        let prompt_time = start.elapsed();
        eprintln!("[PROMPT] Processed {} tokens in {:.2}s", prompt_tokens.len(), prompt_time.as_secs_f64());

        // Now generate with MTP
        let last_prompt_tok = *prompt_tokens.last().unwrap();
        let gen_start = std::time::Instant::now();
        let (tokens, stats, _logprobs) = generate_with_mtp(&model, last_prompt_tok, 20, &mut state, &SamplingParams::thinking_general(), None)
            .expect("Generation failed");
        let gen_time = gen_start.elapsed();

        eprintln!("[GEN] Generated {} tokens in {:.2}s ({:.1} tok/s)",
            tokens.len(), gen_time.as_secs_f64(),
            tokens.len() as f64 / gen_time.as_secs_f64());
        eprintln!("[GEN] Tokens: {:?}", &tokens[..tokens.len().min(20)]);
        eprintln!("[GEN] Unique tokens: {}", tokens.iter().collect::<std::collections::HashSet<_>>().len());
        eprintln!("[GEN] MTP acceptance: {:.1}%", stats.acceptance_rate() * 100.0);
        eprintln!("[GEN] Throughput multiplier: {:.2}x", stats.throughput_multiplier());

        // With a real prompt, we expect more diverse tokens
        let unique = tokens.iter().collect::<std::collections::HashSet<_>>().len();
        eprintln!("[GEN] Diversity: {}/{} unique tokens", unique, tokens.len());
    }

    #[test]
    fn test_prefill_vs_sequential_cost() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let path = format!(
            "{}/.chimere/models/qwopus-27b/Qwen3.5-27B-Opus-Q4_0-MTP.gguf",
            home
        );
        if !std::path::Path::new(&path).exists() { return; }

        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        let model = crate::qwen35_model::Qwen35Model::from_gguf(&path, &device, None)
            .expect("Failed to load model");

        // Warmup: 5 tokens
        let mut state = crate::state::GdnRecurrentState::new(&model.config, &device)
            .expect("Failed to create state");
        for &t in &[1u32, 328, 220, 198, 1535] {
            model.forward_token(t, &mut state).unwrap();
        }

        // Benchmark: forward_token × 1
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            model.forward_token(100, &mut state).unwrap();
        }
        let ft1 = t0.elapsed().as_secs_f64() / 10.0;

        // Benchmark: forward_prefill with 1 token
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            model.forward_prefill(&[100], &mut state).unwrap();
        }
        let fp1 = t0.elapsed().as_secs_f64() / 10.0;

        // Benchmark: forward_prefill with 2 tokens
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            model.forward_prefill(&[100, 200], &mut state).unwrap();
        }
        let fp2 = t0.elapsed().as_secs_f64() / 10.0;

        // Benchmark: forward_prefill with 3 tokens
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            model.forward_prefill(&[100, 200, 300], &mut state).unwrap();
        }
        let fp3 = t0.elapsed().as_secs_f64() / 10.0;

        eprintln!("=== PREFILL vs SEQUENTIAL BENCHMARK ===");
        eprintln!("forward_token(1 tok):     {:.1}ms", ft1 * 1000.0);
        eprintln!("forward_prefill(1 tok):   {:.1}ms", fp1 * 1000.0);
        eprintln!("forward_prefill(2 tok):   {:.1}ms  (vs 2×ft = {:.1}ms)", fp2 * 1000.0, ft1 * 2000.0);
        eprintln!("forward_prefill(3 tok):   {:.1}ms  (vs 3×ft = {:.1}ms)", fp3 * 1000.0, ft1 * 3000.0);
        eprintln!("Speedup 2 tok: {:.2}x", (ft1 * 2.0) / fp2);
        eprintln!("Speedup 3 tok: {:.2}x", (ft1 * 3.0) / fp3);
    }
}
