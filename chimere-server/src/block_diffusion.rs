//! # Block Diffusion Scheduler
//!
//! The block diffusion scheduler implements discrete masked diffusion for
//! text generation, following BD3-LM / MDLM principles:
//!
//! - **Forward process**: progressively mask tokens according to a noise schedule
//! - **Reverse process**: iteratively unmask tokens using a score model
//! - **Block-level generation**: autoregressive between blocks, diffusion within
//!
//! ## Connection to other Chimère modules
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │              BlockDiffusionScheduler                 │
//! │                                                     │
//! │  for each block:                                    │
//! │    1. Initialize: all [MASK] tokens                 │
//! │    2. for t = T..1:                                 │
//! │       a. score = score_model(masked_block, context) │
//! │          └─→ HybridAttentionLayer (DeltaNet + GQA)  │
//! │          └─→ MoERouter dispatches experts            │
//! │       b. unmask tokens where confidence > threshold  │
//! │       c. StateMetrics → Engram.maybe_update()       │
//! │    3. context ← concat(context, generated_block)    │
//! │    4. Repeat for next block                         │
//! └─────────────────────────────────────────────────────┘
//! ```
//!
//! ## Noise Schedule
//!
//! MDLM showed that the discrete diffusion ELBO simplifies to a weighted
//! average of masked language modeling losses. The noise schedule σ(t)
//! controls the masking rate:
//!
//! - σ(0) = 0  → clean data (no masks)
//! - σ(1) = 1  → fully masked (maximum entropy)
//! - Cosine schedule: σ(t) = cos(π/2 · (1-t)) for smooth interpolation
//!
//! ## Simplified version (v1)
//!
//! This implementation uses:
//! - Fixed block size (configurable, default 32)
//! - Cosine noise schedule
//! - Confidence-based unmasking (unmask highest-confidence tokens first)
//! - Deterministic greedy sampling (argmax of score model output)
//!
//! Future: entropy-adaptive block boundaries, dilated unmasking,
//! temperature-based sampling.

use crate::StateMetrics;

// -------------------------------------------------------------------------
// Configuration
// -------------------------------------------------------------------------

/// Noise schedule type for the forward/reverse process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseScheduleType {
    /// σ(t) = cos(π/2 · (1-t)) — smooth, proven in MDLM
    Cosine,
    /// σ(t) = t — linear interpolation
    Linear,
    /// σ(t) = 1 - (1-t)^2 — fast early masking, slow refinement
    Quadratic,
}

/// Configuration for the block diffusion scheduler.
#[derive(Debug, Clone)]
pub struct BlockDiffusionConfig {
    /// Number of tokens per block (fixed in v1)
    pub block_size: usize,
    /// Number of denoising steps in the reverse process
    pub num_steps: usize,
    /// Noise schedule type
    pub schedule: NoiseScheduleType,
    /// Vocabulary size (for score model output)
    pub vocab_size: usize,
    /// Token ID used for [MASK]
    pub mask_token_id: u32,
    /// Minimum confidence to unmask a token (0.0 = unmask everything)
    pub unmask_threshold: f32,
    /// Whether to use nucleus (top-p) sampling vs greedy
    pub greedy: bool,
    /// Temperature for sampling (only if !greedy)
    pub temperature: f32,
    /// Enable entropy-adaptive block sizing (v2 feature)
    pub adaptive_blocks: bool,
    /// Min block size for adaptive mode
    pub min_block_size: usize,
    /// Max block size for adaptive mode
    pub max_block_size: usize,
    /// Mean delta threshold for block boundary detection
    pub entropy_boundary_threshold: f32,
}

impl BlockDiffusionConfig {
    /// Default config matching BD3LM PoC training parameters
    pub fn default_bd3lm() -> Self {
        Self {
            block_size: 32,
            num_steps: 10,
            schedule: NoiseScheduleType::Cosine,
            vocab_size: 151936, // Qwen2/Nanbeige vocab
            mask_token_id: 151936 - 1, // Last token as mask
            unmask_threshold: 0.0,
            greedy: true,
            temperature: 1.0,
            adaptive_blocks: false,
            min_block_size: 16,
            max_block_size: 128,
            entropy_boundary_threshold: 0.5,
        }
    }

    /// Test config with small dimensions
    pub fn test() -> Self {
        Self {
            block_size: 8,
            num_steps: 5,
            schedule: NoiseScheduleType::Cosine,
            vocab_size: 100,
            mask_token_id: 99, // [MASK] = last token
            unmask_threshold: 0.0,
            greedy: true,
            temperature: 1.0,
            adaptive_blocks: false,
            min_block_size: 4,
            max_block_size: 16,
            entropy_boundary_threshold: 0.5,
        }
    }

    /// Chimère config with entropy-adaptive blocks
    pub fn chimere() -> Self {
        Self {
            block_size: 64,
            num_steps: 16,
            schedule: NoiseScheduleType::Cosine,
            vocab_size: 151936,
            mask_token_id: 151936 - 1,
            unmask_threshold: 0.1,
            greedy: false,
            temperature: 0.8,
            adaptive_blocks: true,
            min_block_size: 16,
            max_block_size: 128,
            entropy_boundary_threshold: 0.5,
        }
    }
}

// -------------------------------------------------------------------------
// Noise Schedule
// -------------------------------------------------------------------------

/// Compute the noise level σ(t) for a given timestep.
///
/// t ∈ [0, 1] where:
/// - t = 0 → σ = 0 (clean data)
/// - t = 1 → σ = 1 (fully masked)
///
/// Returns the fraction of tokens that should be masked at time t.
pub fn noise_level(t: f64, schedule: NoiseScheduleType) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match schedule {
        NoiseScheduleType::Cosine => {
            // σ(t) = cos(π/2 · (1-t))
            // At t=0: cos(π/2) = 0 (clean)
            // At t=1: cos(0) = 1 (fully masked)
            (std::f64::consts::FRAC_PI_2 * (1.0 - t)).cos()
        }
        NoiseScheduleType::Linear => t,
        NoiseScheduleType::Quadratic => {
            // σ(t) = 1 - (1-t)^2
            // Fast early masking, slow refinement at the end
            1.0 - (1.0 - t).powi(2)
        }
    }
}

/// Compute the discrete timestep schedule for the reverse process.
///
/// Returns a vector of (t_current, t_next) pairs from t=1 (fully masked)
/// to t=0 (clean), representing the denoising trajectory.
///
/// For num_steps=5: [(1.0, 0.8), (0.8, 0.6), (0.6, 0.4), (0.4, 0.2), (0.2, 0.0)]
pub fn reverse_timesteps(num_steps: usize) -> Vec<(f64, f64)> {
    assert!(num_steps > 0, "Need at least 1 denoising step");
    let mut steps = Vec::with_capacity(num_steps);
    for i in 0..num_steps {
        let t_current = 1.0 - (i as f64 / num_steps as f64);
        let t_next = 1.0 - ((i + 1) as f64 / num_steps as f64);
        steps.push((t_current, t_next));
    }
    steps
}

// -------------------------------------------------------------------------
// Forward Process (Noising)
// -------------------------------------------------------------------------

/// A block of tokens with mask state tracking.
#[derive(Debug, Clone)]
pub struct MaskedBlock {
    /// Current token IDs (may contain mask_token_id)
    pub tokens: Vec<u32>,
    /// Whether each position is currently masked
    pub is_masked: Vec<bool>,
    /// Original (clean) token IDs — only known during training/testing
    pub clean_tokens: Option<Vec<u32>>,
    /// Current noise level σ(t)
    pub noise_level: f64,
}

impl MaskedBlock {
    /// Create a fully masked block (starting point for generation)
    pub fn fully_masked(block_size: usize, mask_token_id: u32) -> Self {
        Self {
            tokens: vec![mask_token_id; block_size],
            is_masked: vec![true; block_size],
            clean_tokens: None,
            noise_level: 1.0,
        }
    }

    /// Create from clean tokens (for training / forward process testing)
    pub fn from_clean(clean: &[u32], _mask_token_id: u32) -> Self {
        Self {
            tokens: clean.to_vec(),
            is_masked: vec![false; clean.len()],
            clean_tokens: Some(clean.to_vec()),
            noise_level: 0.0,
        }
    }

    /// Number of tokens in this block
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Number of currently masked positions
    pub fn num_masked(&self) -> usize {
        self.is_masked.iter().filter(|&&m| m).count()
    }

    /// Fraction of masked positions
    pub fn mask_fraction(&self) -> f64 {
        if self.tokens.is_empty() {
            return 0.0;
        }
        self.num_masked() as f64 / self.len() as f64
    }
}

/// Apply forward noise to a clean block: mask a fraction σ(t) of tokens.
///
/// Uses deterministic masking based on position priority (for reproducibility).
/// In a real implementation, this would use random masking with a seed.
///
/// # Arguments
/// * `block` - Clean or partially masked block
/// * `target_noise` - Target noise level σ(t) ∈ [0, 1]
/// * `mask_token_id` - Token ID to use for masking
/// * `priority` - Per-position priority for masking order (higher = mask first).
///   If None, uses reverse position order (mask last positions first).
pub fn forward_noise(
    block: &MaskedBlock,
    target_noise: f64,
    mask_token_id: u32,
    priority: Option<&[f64]>,
) -> MaskedBlock {
    let n = block.len();
    let target_masked = ((n as f64 * target_noise).round() as usize).min(n);

    // Build priority-ordered indices
    let mut indices: Vec<usize> = (0..n).collect();
    if let Some(prio) = priority {
        // Sort by priority descending (highest priority masked first)
        indices.sort_by(|&a, &b| {
            prio[b]
                .partial_cmp(&prio[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        // Default: mask from the end (like BD3LM's left-to-right generation)
        indices.reverse();
    }

    let mut new_tokens = block
        .clean_tokens
        .as_ref()
        .unwrap_or(&block.tokens)
        .clone();
    let mut new_masked = vec![false; n];

    // Mask the top `target_masked` positions by priority
    for &idx in indices.iter().take(target_masked) {
        new_tokens[idx] = mask_token_id;
        new_masked[idx] = true;
    }

    MaskedBlock {
        tokens: new_tokens,
        is_masked: new_masked,
        clean_tokens: block.clean_tokens.clone(),
        noise_level: target_noise,
    }
}

// -------------------------------------------------------------------------
// Reverse Process (Denoising)
// -------------------------------------------------------------------------

/// Score model output for one position: probability distribution over vocab.
///
/// In a real model, this comes from the transformer's output logits
/// passed through softmax. Here we represent it as (token_id, confidence) pairs.
#[derive(Debug, Clone)]
pub struct TokenPrediction {
    /// Predicted token ID (argmax of logits)
    pub token_id: u32,
    /// Confidence (probability) of the prediction
    pub confidence: f32,
    /// Full distribution (optional, for entropy computation)
    pub logits: Option<Vec<f32>>,
}

/// Output of the score model for an entire block.
///
/// This is what `HybridAttentionLayer.forward()` + `MoERouter.route()` produce
/// after processing through the transformer stack.
#[derive(Debug, Clone)]
pub struct ScoreModelOutput {
    /// Per-position predictions
    pub predictions: Vec<TokenPrediction>,
    /// StateMetrics from the DeltaNet layers (for Engram update + block sizing)
    pub state_metrics: Option<StateMetrics>,
}

/// Perform one denoising step: unmask tokens based on score model predictions.
///
/// The unmasking strategy follows MDLM's approach:
/// 1. Score model predicts distribution for ALL positions
/// 2. For masked positions, rank by confidence
/// 3. Unmask the top (σ(t) - σ(t-1)) × block_size positions
/// 4. Keep already-unmasked positions unchanged
///
/// # Arguments
/// * `block` - Current masked block state
/// * `scores` - Score model predictions for each position
/// * `t_current` - Current noise level
/// * `t_next` - Target noise level (< t_current)
/// * `config` - Block diffusion configuration
pub fn denoise_step(
    block: &MaskedBlock,
    scores: &ScoreModelOutput,
    _t_current: f64,
    t_next: f64,
    config: &BlockDiffusionConfig,
) -> MaskedBlock {
    let n = block.len();
    assert_eq!(
        scores.predictions.len(),
        n,
        "Score model output must match block size"
    );

    let sigma_next = noise_level(t_next, config.schedule);

    // How many positions should remain masked after this step
    let target_masked = ((n as f64 * sigma_next).round() as usize).min(n);
    let currently_masked = block.num_masked();

    // How many to unmask this step
    let to_unmask = currently_masked.saturating_sub(target_masked);

    // Collect masked positions with their confidence scores
    let mut masked_with_confidence: Vec<(usize, f32, u32)> = Vec::new();
    for (i, pred) in scores.predictions.iter().enumerate() {
        if block.is_masked[i] {
            masked_with_confidence.push((i, pred.confidence, pred.token_id));
        }
    }

    // Sort by confidence descending — unmask most confident first
    masked_with_confidence.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build new block state
    let mut new_tokens = block.tokens.clone();
    let mut new_masked = block.is_masked.clone();

    for &(idx, confidence, token_id) in masked_with_confidence.iter().take(to_unmask) {
        if confidence >= config.unmask_threshold {
            new_tokens[idx] = token_id;
            new_masked[idx] = false;
        }
    }

    MaskedBlock {
        tokens: new_tokens,
        is_masked: new_masked,
        clean_tokens: block.clean_tokens.clone(),
        noise_level: t_next,
    }
}

// -------------------------------------------------------------------------
// Block Generation (Full Reverse Process)
// -------------------------------------------------------------------------

/// Result of generating one block via the reverse diffusion process.
#[derive(Debug, Clone)]
pub struct GeneratedBlock {
    /// Final token IDs after denoising
    pub tokens: Vec<u32>,
    /// Per-position confidence from the final denoising step
    pub confidences: Vec<f32>,
    /// Number of denoising steps actually used
    pub steps_used: usize,
    /// Positions that remained masked (if any — ideally 0)
    pub remaining_masked: usize,
    /// StateMetrics from the final denoising step (for Engram)
    pub final_state_metrics: Option<StateMetrics>,
    /// Per-step unmasking counts (for analysis)
    pub unmask_history: Vec<usize>,
}

/// Trait for the score model — abstracts the transformer forward pass.
///
/// In production, this wraps the full Chimère stack:
/// HybridAttentionLayer → MoERouter → Experts → output projection
///
/// For testing, we provide `MockScoreModel` below.
pub trait ScoreModel {
    /// Run the score model on a masked block with conditioning context.
    ///
    /// # Arguments
    /// * `block` - Current masked block state
    /// * `context` - Previously generated tokens (autoregressive context)
    ///
    /// # Returns
    /// Score model predictions for each position in the block.
    fn forward(&self, block: &MaskedBlock, context: &[u32]) -> ScoreModelOutput;
}

/// Generate one block via the full reverse diffusion process.
///
/// Starting from a fully masked block, iteratively denoise by:
/// 1. Running the score model to get predictions
/// 2. Unmasking the most confident positions
/// 3. Repeating until σ(t) → 0
///
/// The context (previously generated blocks) conditions the score model
/// in an autoregressive manner — this is the "block AR" part of BD3LM.
pub fn generate_block(
    config: &BlockDiffusionConfig,
    score_model: &dyn ScoreModel,
    context: &[u32],
) -> GeneratedBlock {
    let mut block = MaskedBlock::fully_masked(config.block_size, config.mask_token_id);
    let timesteps = reverse_timesteps(config.num_steps);
    let mut unmask_history = Vec::new();
    let mut last_scores: Option<ScoreModelOutput> = None;

    for &(t_current, t_next) in timesteps.iter() {
        let scores = score_model.forward(&block, context);
        let prev_masked = block.num_masked();
        block = denoise_step(&block, &scores, t_current, t_next, config);
        let unmasked_this_step = prev_masked - block.num_masked();
        unmask_history.push(unmasked_this_step);
        last_scores = Some(scores);
    }

    let confidences = last_scores
        .as_ref()
        .map(|s| s.predictions.iter().map(|p| p.confidence).collect())
        .unwrap_or_else(|| vec![0.0; config.block_size]);

    let final_state_metrics = last_scores.and_then(|s| s.state_metrics);

    GeneratedBlock {
        tokens: block.tokens.clone(),
        confidences,
        steps_used: config.num_steps,
        remaining_masked: block.num_masked(),
        final_state_metrics,
        unmask_history,
    }
}

// -------------------------------------------------------------------------
// Entropy-Adaptive Block Boundaries (v2)
// -------------------------------------------------------------------------

/// Compute adaptive block boundaries based on StateMetrics.
///
/// Instead of fixed block_size=32, adapt block size based on content:
/// - Low mean_delta (DeltaNet confident) → larger blocks (less compute)
/// - High mean_delta (DeltaNet uncertain) → smaller blocks (more precision)
///
/// This is the entropy-adaptive compute allocation applied to diffusion:
/// predictable text gets generated in bigger chunks, novel/complex text
/// gets finer-grained diffusion.
///
/// # Arguments
/// * `state_metrics` - StateMetrics from the previous block's final step
/// * `config` - Block diffusion configuration
///
/// # Returns
/// Block size to use for the next block
pub fn compute_adaptive_block_size(
    state_metrics: &StateMetrics,
    config: &BlockDiffusionConfig,
) -> usize {
    if !config.adaptive_blocks {
        return config.block_size;
    }

    // Normalize mean_delta to [0, 1] range
    let uncertainty = (state_metrics.mean_delta / config.entropy_boundary_threshold).min(1.0);

    // Low uncertainty → large blocks, high uncertainty → small blocks
    // Linear interpolation: size = max - uncertainty * (max - min)
    let range = config.max_block_size - config.min_block_size;
    let size = config.max_block_size as f64 - (uncertainty as f64 * range as f64);

    // Round to nearest multiple of 8 (for tensor alignment)
    let size = (size / 8.0).round() as usize * 8;
    size.clamp(config.min_block_size, config.max_block_size)
}

/// Compute the number of denoising steps adapted to block size.
///
/// Larger blocks need more steps (more tokens to unmask).
/// Rule: steps ∝ sqrt(block_size), clamped to [4, 32].
pub fn compute_adaptive_steps(block_size: usize) -> usize {
    let steps = (block_size as f64).sqrt().round() as usize;
    steps.clamp(4, 32)
}

// -------------------------------------------------------------------------
// Multi-Block Generation
// -------------------------------------------------------------------------

/// Generate a full sequence by generating blocks autoregressively.
///
/// Each block is generated via diffusion, then appended to the context
/// for the next block. This is the "block AR + intra-block diffusion"
/// pattern from BD3LM.
///
/// # Arguments
/// * `config` - Block diffusion configuration
/// * `score_model` - The transformer score model
/// * `prompt` - Initial context tokens (prompt)
/// * `num_blocks` - Number of blocks to generate
///
/// # Returns
/// Generated token sequence (prompt + generated blocks)
pub fn generate_sequence(
    config: &BlockDiffusionConfig,
    score_model: &dyn ScoreModel,
    prompt: &[u32],
    num_blocks: usize,
) -> Vec<u32> {
    let mut context = prompt.to_vec();
    let mut all_state_metrics: Option<StateMetrics> = None;

    for _block_idx in 0..num_blocks {
        // Adaptive block sizing based on previous block's state
        let effective_config = if config.adaptive_blocks {
            if let Some(ref metrics) = all_state_metrics {
                let adaptive_size = compute_adaptive_block_size(metrics, config);
                let adaptive_steps = compute_adaptive_steps(adaptive_size);
                let mut cfg = config.clone();
                cfg.block_size = adaptive_size;
                cfg.num_steps = adaptive_steps;
                cfg
            } else {
                config.clone()
            }
        } else {
            config.clone()
        };

        let result = generate_block(&effective_config, score_model, &context);
        context.extend_from_slice(&result.tokens);
        all_state_metrics = result.final_state_metrics;
    }

    context
}

// -------------------------------------------------------------------------
// Metrics
// -------------------------------------------------------------------------

/// Compute accuracy of generated block vs clean reference.
/// Only meaningful during testing when clean_tokens are known.
pub fn block_accuracy(generated: &[u32], clean: &[u32]) -> f64 {
    assert_eq!(generated.len(), clean.len());
    if generated.is_empty() {
        return 1.0;
    }
    let correct = generated
        .iter()
        .zip(clean.iter())
        .filter(|(g, c)| g == c)
        .count();
    correct as f64 / generated.len() as f64
}

/// Compute per-position accuracy at each denoising step.
/// Useful for analyzing the denoising trajectory.
pub fn step_accuracy(block: &MaskedBlock) -> f64 {
    if let Some(ref clean) = block.clean_tokens {
        block_accuracy(&block.tokens, clean)
    } else {
        // Can't compute without reference
        1.0 - block.mask_fraction()
    }
}

// -------------------------------------------------------------------------
// Mock Score Model (for testing)
// -------------------------------------------------------------------------

/// A mock score model that "knows" the correct tokens.
///
/// Given clean tokens, it returns the correct prediction with configurable
/// confidence. This tests the scheduler logic independently of the
/// actual transformer.
pub struct MockScoreModel {
    /// The "correct" tokens the model should predict
    pub target_tokens: Vec<u32>,
    /// Base confidence for correct predictions
    pub base_confidence: f32,
    /// Add position-dependent confidence variation
    pub position_decay: bool,
    /// StateMetrics to return (simulates DeltaNet output)
    pub state_metrics: Option<StateMetrics>,
}

impl ScoreModel for MockScoreModel {
    fn forward(&self, block: &MaskedBlock, _context: &[u32]) -> ScoreModelOutput {
        let predictions = (0..block.len())
            .map(|i| {
                let target_idx = i % self.target_tokens.len();
                let confidence = if self.position_decay {
                    // Earlier positions get higher confidence (like real models)
                    self.base_confidence * (1.0 - 0.3 * (i as f32 / block.len() as f32))
                } else {
                    self.base_confidence
                };
                TokenPrediction {
                    token_id: self.target_tokens[target_idx],
                    confidence,
                    logits: None,
                }
            })
            .collect();

        ScoreModelOutput {
            predictions,
            state_metrics: self.state_metrics.clone(),
        }
    }
}

/// A noisy mock that sometimes predicts wrong tokens.
/// Error rate controls how often it predicts incorrectly.
pub struct NoisyMockScoreModel {
    pub target_tokens: Vec<u32>,
    pub base_confidence: f32,
    /// Positions that will get wrong predictions (deterministic for testing)
    pub error_positions: Vec<usize>,
    pub wrong_token_id: u32,
    pub wrong_confidence: f32,
}

impl ScoreModel for NoisyMockScoreModel {
    fn forward(&self, block: &MaskedBlock, _context: &[u32]) -> ScoreModelOutput {
        let predictions = (0..block.len())
            .map(|i| {
                if self.error_positions.contains(&i) {
                    TokenPrediction {
                        token_id: self.wrong_token_id,
                        confidence: self.wrong_confidence,
                        logits: None,
                    }
                } else {
                    let target_idx = i % self.target_tokens.len();
                    TokenPrediction {
                        token_id: self.target_tokens[target_idx],
                        confidence: self.base_confidence,
                        logits: None,
                    }
                }
            })
            .collect();

        ScoreModelOutput {
            predictions,
            state_metrics: None,
        }
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Noise Schedule Tests ---

    #[test]
    fn test_cosine_schedule_boundaries() {
        // σ(0) should be 0 (clean)
        let s0 = noise_level(0.0, NoiseScheduleType::Cosine);
        assert!(
            s0.abs() < 1e-10,
            "σ(0) should be 0 (clean), got {:.6}",
            s0
        );

        // σ(1) should be 1 (fully masked)
        let s1 = noise_level(1.0, NoiseScheduleType::Cosine);
        assert!(
            (s1 - 1.0).abs() < 1e-10,
            "σ(1) should be 1 (masked), got {:.6}",
            s1
        );

        // σ(0.5) should be between 0 and 1
        let s_mid = noise_level(0.5, NoiseScheduleType::Cosine);
        assert!(
            s_mid > 0.0 && s_mid < 1.0,
            "σ(0.5) should be in (0,1), got {:.6}",
            s_mid
        );

        println!(
            "Cosine: σ(0)={:.4}, σ(0.25)={:.4}, σ(0.5)={:.4}, σ(0.75)={:.4}, σ(1)={:.4}",
            noise_level(0.0, NoiseScheduleType::Cosine),
            noise_level(0.25, NoiseScheduleType::Cosine),
            noise_level(0.5, NoiseScheduleType::Cosine),
            noise_level(0.75, NoiseScheduleType::Cosine),
            noise_level(1.0, NoiseScheduleType::Cosine),
        );
    }

    #[test]
    fn test_schedule_monotonicity() {
        // Noise should increase monotonically with t
        for schedule in [
            NoiseScheduleType::Cosine,
            NoiseScheduleType::Linear,
            NoiseScheduleType::Quadratic,
        ] {
            let mut prev = 0.0;
            for i in 0..=100 {
                let t = i as f64 / 100.0;
                let sigma = noise_level(t, schedule);
                assert!(
                    sigma >= prev - 1e-10,
                    "{:?}: σ({:.2})={:.4} < σ({:.2})={:.4} — not monotonic",
                    schedule,
                    t,
                    sigma,
                    (i - 1) as f64 / 100.0,
                    prev
                );
                prev = sigma;
            }
            println!("{:?}: monotonicity verified (101 points)", schedule);
        }
    }

    #[test]
    fn test_reverse_timesteps() {
        let steps = reverse_timesteps(5);
        assert_eq!(steps.len(), 5);

        // First step: t=1.0 → 0.8
        assert!((steps[0].0 - 1.0).abs() < 1e-10);
        assert!((steps[0].1 - 0.8).abs() < 1e-10);

        // Last step: t=0.2 → 0.0
        assert!((steps[4].0 - 0.2).abs() < 1e-10);
        assert!(steps[4].1.abs() < 1e-10);

        // All steps should decrease
        for (t_curr, t_next) in &steps {
            assert!(t_curr > t_next, "Steps should decrease: {} > {}", t_curr, t_next);
        }

        println!("Reverse timesteps (5 steps): {:?}", steps);
    }

    // --- Forward Process Tests ---

    #[test]
    fn test_forward_noise_levels() {
        let config = BlockDiffusionConfig::test();
        let clean_tokens: Vec<u32> = (0..8).collect(); // [0,1,2,3,4,5,6,7]
        let block = MaskedBlock::from_clean(&clean_tokens, config.mask_token_id);

        // No noise → no masking
        let noised_0 = forward_noise(&block, 0.0, config.mask_token_id, None);
        assert_eq!(noised_0.num_masked(), 0, "σ=0 should have 0 masks");

        // 50% noise → ~4 masked
        let noised_50 = forward_noise(&block, 0.5, config.mask_token_id, None);
        assert_eq!(noised_50.num_masked(), 4, "σ=0.5 should mask 4 of 8");

        // Full noise → all masked
        let noised_100 = forward_noise(&block, 1.0, config.mask_token_id, None);
        assert_eq!(noised_100.num_masked(), 8, "σ=1.0 should mask all 8");

        // Masked tokens should be mask_token_id
        for (i, &tok) in noised_100.tokens.iter().enumerate() {
            assert_eq!(
                tok, config.mask_token_id,
                "Position {} should be [MASK]",
                i
            );
        }

        println!(
            "Forward noise: σ=0→{}masked, σ=0.5→{}masked, σ=1.0→{}masked",
            noised_0.num_masked(),
            noised_50.num_masked(),
            noised_100.num_masked()
        );
    }

    #[test]
    fn test_forward_noise_preserves_clean() {
        let config = BlockDiffusionConfig::test();
        let clean: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let block = MaskedBlock::from_clean(&clean, config.mask_token_id);

        let noised = forward_noise(&block, 0.5, config.mask_token_id, None);

        // Unmasked positions should retain original tokens
        for (i, (&tok, &masked)) in noised.tokens.iter().zip(noised.is_masked.iter()).enumerate() {
            if !masked {
                assert_eq!(
                    tok, clean[i],
                    "Unmasked position {} should retain token {}",
                    i, clean[i]
                );
            } else {
                assert_eq!(
                    tok, config.mask_token_id,
                    "Masked position {} should be [MASK]",
                    i
                );
            }
        }

        println!("Forward noise preserves unmasked tokens: verified");
    }

    // --- Denoising Tests ---

    #[test]
    fn test_denoise_step_unmasks_confident() {
        // Use linear schedule so σ(0.5) = 0.5 exactly (4 of 8 masked)
        let mut config = BlockDiffusionConfig::test();
        config.schedule = NoiseScheduleType::Linear;

        // Start fully masked
        let block = MaskedBlock::fully_masked(8, config.mask_token_id);
        assert_eq!(block.num_masked(), 8);

        // Score model predicts tokens with varying confidence
        let scores = ScoreModelOutput {
            predictions: vec![
                TokenPrediction { token_id: 10, confidence: 0.95, logits: None }, // high
                TokenPrediction { token_id: 20, confidence: 0.30, logits: None }, // low
                TokenPrediction { token_id: 30, confidence: 0.85, logits: None }, // high
                TokenPrediction { token_id: 40, confidence: 0.10, logits: None }, // low
                TokenPrediction { token_id: 50, confidence: 0.90, logits: None }, // high
                TokenPrediction { token_id: 60, confidence: 0.20, logits: None }, // low
                TokenPrediction { token_id: 70, confidence: 0.80, logits: None }, // high
                TokenPrediction { token_id: 80, confidence: 0.15, logits: None }, // low
            ],
            state_metrics: None,
        };

        // Denoise from σ(1.0)→σ(0.5): should unmask ~4 positions
        let denoised = denoise_step(&block, &scores, 1.0, 0.5, &config);

        // The 4 highest confidence positions should be unmasked
        assert!(
            !denoised.is_masked[0],
            "Position 0 (conf=0.95) should be unmasked"
        );
        assert!(
            !denoised.is_masked[4],
            "Position 4 (conf=0.90) should be unmasked"
        );
        assert!(
            !denoised.is_masked[2],
            "Position 2 (conf=0.85) should be unmasked"
        );
        assert!(
            !denoised.is_masked[6],
            "Position 6 (conf=0.80) should be unmasked"
        );

        // Low confidence positions should remain masked
        assert!(
            denoised.is_masked[1],
            "Position 1 (conf=0.30) should stay masked"
        );
        assert!(
            denoised.is_masked[3],
            "Position 3 (conf=0.10) should stay masked"
        );

        println!(
            "Denoise step: {} → {} masked (unmasked {} most confident)",
            block.num_masked(),
            denoised.num_masked(),
            block.num_masked() - denoised.num_masked()
        );
    }

    #[test]
    fn test_full_generation_perfect_model() {
        let config = BlockDiffusionConfig::test();
        let target: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];

        let mock = MockScoreModel {
            target_tokens: target.clone(),
            base_confidence: 0.9,
            position_decay: false,
            state_metrics: None,
        };

        let result = generate_block(&config, &mock, &[]);

        // Perfect model should generate correct tokens
        let accuracy = block_accuracy(&result.tokens, &target);
        assert!(
            accuracy > 0.99,
            "Perfect model should achieve >99% accuracy, got {:.1}%",
            accuracy * 100.0
        );
        assert_eq!(
            result.remaining_masked, 0,
            "No positions should remain masked"
        );

        println!(
            "Full generation: accuracy={:.1}%, steps={}, remaining_masked={}",
            accuracy * 100.0,
            result.steps_used,
            result.remaining_masked
        );
        println!("Unmask history: {:?}", result.unmask_history);
    }

    #[test]
    fn test_generation_with_noise() {
        let config = BlockDiffusionConfig::test();
        let target: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];

        // Model that gets positions 2 and 5 wrong
        let noisy = NoisyMockScoreModel {
            target_tokens: target.clone(),
            base_confidence: 0.9,
            error_positions: vec![2, 5],
            wrong_token_id: 0,
            wrong_confidence: 0.85, // high confidence on wrong answer
        };

        let result = generate_block(&config, &noisy, &[]);
        let accuracy = block_accuracy(&result.tokens, &target);

        // Should get 6/8 correct (positions 2 and 5 wrong)
        assert!(
            (accuracy - 0.75).abs() < 0.01,
            "Noisy model should get 75% accuracy (6/8), got {:.1}%",
            accuracy * 100.0
        );

        // Wrong positions should have the wrong token
        assert_eq!(
            result.tokens[2], 0,
            "Position 2 should have wrong token"
        );
        assert_eq!(
            result.tokens[5], 0,
            "Position 5 should have wrong token"
        );

        println!(
            "Noisy generation: accuracy={:.1}%, errors at {:?}",
            accuracy * 100.0,
            vec![2, 5]
        );
    }

    // --- Adaptive Block Size Tests ---

    #[test]
    fn test_adaptive_block_size_confident() {
        let config = BlockDiffusionConfig::chimere();

        // Low delta = confident → large blocks
        let confident_metrics = StateMetrics {
            frobenius_norm: 1.0,
            effective_rank: 0.8,
            mean_delta: 0.05, // very low → predictable content
        };

        let size = compute_adaptive_block_size(&confident_metrics, &config);
        assert!(
            size >= 112,
            "Confident state should produce large blocks, got {}",
            size
        );
        assert_eq!(size % 8, 0, "Block size should be multiple of 8");

        println!("Confident (delta=0.05): block_size={}", size);
    }

    #[test]
    fn test_adaptive_block_size_uncertain() {
        let config = BlockDiffusionConfig::chimere();

        // High delta = uncertain → small blocks
        let uncertain_metrics = StateMetrics {
            frobenius_norm: 1.0,
            effective_rank: 0.3,
            mean_delta: 0.8, // high → novel/complex content
        };

        let size = compute_adaptive_block_size(&uncertain_metrics, &config);
        assert!(
            size <= 40,
            "Uncertain state should produce small blocks, got {}",
            size
        );
        assert_eq!(size % 8, 0, "Block size should be multiple of 8");

        println!("Uncertain (delta=0.8): block_size={}", size);
    }

    #[test]
    fn test_adaptive_steps_scale_with_size() {
        // Larger blocks need more steps
        let steps_16 = compute_adaptive_steps(16);
        let steps_64 = compute_adaptive_steps(64);
        let steps_128 = compute_adaptive_steps(128);

        assert!(
            steps_16 <= steps_64,
            "16 tokens should need <= steps than 64: {} vs {}",
            steps_16,
            steps_64
        );
        assert!(
            steps_64 <= steps_128,
            "64 tokens should need <= steps than 128: {} vs {}",
            steps_64,
            steps_128
        );

        println!(
            "Adaptive steps: size=16→{}, size=64→{}, size=128→{}",
            steps_16, steps_64, steps_128
        );
    }

    // --- Multi-Block Generation Test ---

    #[test]
    fn test_multi_block_generation() {
        let config = BlockDiffusionConfig::test(); // block_size=8
        let target: Vec<u32> = (0..24).collect(); // 3 blocks of 8

        let mock = MockScoreModel {
            target_tokens: target.clone(),
            base_confidence: 0.9,
            position_decay: false,
            state_metrics: None,
        };

        let prompt: Vec<u32> = vec![100, 101]; // 2 token prompt
        let result = generate_sequence(&config, &mock, &prompt, 3);

        // Result should be: prompt (2) + 3 blocks of 8 = 26 tokens
        assert_eq!(
            result.len(),
            26,
            "Should have prompt + 3 blocks = 26 tokens, got {}",
            result.len()
        );

        // Prompt should be preserved
        assert_eq!(result[0], 100, "Prompt token 0 preserved");
        assert_eq!(result[1], 101, "Prompt token 1 preserved");

        println!(
            "Multi-block: {} tokens generated ({} prompt + {} generated)",
            result.len(),
            2,
            result.len() - 2
        );
    }

    #[test]
    fn test_state_metrics_flow_to_adaptive() {
        // Verify that StateMetrics from score model influences block sizing
        let mut config = BlockDiffusionConfig::test();
        config.adaptive_blocks = true;
        config.min_block_size = 4;
        config.max_block_size = 16;

        // First block: model returns "confident" state
        let confident_mock = MockScoreModel {
            target_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
            base_confidence: 0.9,
            position_decay: false,
            state_metrics: Some(StateMetrics {
                frobenius_norm: 1.0,
                effective_rank: 0.8,
                mean_delta: 0.05, // low → confident → should trigger larger next block
            }),
        };

        let result = generate_block(&config, &confident_mock, &[]);

        // The StateMetrics should be available for the next block's adaptive sizing
        assert!(
            result.final_state_metrics.is_some(),
            "StateMetrics should flow through generation"
        );

        let metrics = result.final_state_metrics.unwrap();
        let next_size = compute_adaptive_block_size(&metrics, &config);

        assert!(
            next_size > config.block_size,
            "Confident state should produce larger next block: {} > {}",
            next_size,
            config.block_size
        );

        println!(
            "StateMetrics flow: mean_delta={:.2} → next_block_size={} (base={})",
            metrics.mean_delta, next_size, config.block_size
        );
    }

    // --- Schedule Comparison ---

    #[test]
    fn test_schedule_comparison() {
        println!("\nSchedule comparison (σ at each timestep):");
        println!("{:<6} {:<10} {:<10} {:<10}", "t", "Cosine", "Linear", "Quadratic");
        println!("{}", "-".repeat(36));
        for i in 0..=10 {
            let t = i as f64 / 10.0;
            println!(
                "{:<6.1} {:<10.4} {:<10.4} {:<10.4}",
                t,
                noise_level(t, NoiseScheduleType::Cosine),
                noise_level(t, NoiseScheduleType::Linear),
                noise_level(t, NoiseScheduleType::Quadratic),
            );
        }

        // Cosine σ(t) = cos(π/2·(1-t)) is concave: more aggressive than
        // linear at start (derivative at t=0 is π/2 ≈ 1.57 vs 1.0 for linear)
        let cos_25 = noise_level(0.25, NoiseScheduleType::Cosine);
        let lin_25 = noise_level(0.25, NoiseScheduleType::Linear);
        assert!(
            cos_25 > lin_25,
            "Cosine should be more aggressive at t=0.25: {:.4} > {:.4}",
            cos_25,
            lin_25
        );

        // Quadratic should also be more aggressive than linear at the start
        let quad_25 = noise_level(0.25, NoiseScheduleType::Quadratic);
        assert!(
            quad_25 > lin_25,
            "Quadratic should be more aggressive at t=0.25: {:.4} > {:.4}",
            quad_25,
            lin_25
        );

        // But cosine is less aggressive than quadratic at the start
        assert!(
            cos_25 < quad_25,
            "Cosine < Quadratic at t=0.25: {:.4} < {:.4}",
            cos_25,
            quad_25
        );
    }
}
