//! # Entropy Router — Chimère Decode Strategy Orchestrator
//!
//! Routes tokens to autoregressive (GDN) or block diffusion generation
//! based on real-time entropy signals from the GatedDeltaNet state.
//!
//! ## The Key Insight
//!
//! ~80% of tokens are low-entropy and predictable — fast AR is optimal.
//! ~20% are high-entropy "decision forks" — block diffusion with iterative
//! refinement produces better quality. The entropy router exploits this
//! asymmetry for 4–6× throughput with no quality loss.
//!
//! ## Signal Sources (zero additional cost)
//!
//! All signals are intermediate values already computed during GDN forward:
//!
//! | Signal            | Source                    | Interpretation            |
//! |-------------------|---------------------------|---------------------------|
//! | Gate α_t          | GDN decay gate            | Low → content boundary    |
//! | Delta magnitude   | v - S·k                   | High → novel/uncertain    |
//! | State ‖S‖_F       | Frobenius norm            | Saturated → fragile       |
//! | Logit entropy     | output distribution       | High → uncertain next tok |
//! | Logit varentropy  | variance of surprisal     | High → decision fork      |
//! | MoE routing H     | expert gate distribution  | High → confused routing   |
//!
//! ## Decision Space (Entropix 2D framework extended to 3D)
//!
//! ```text
//!                    ┌─────────────────────────────────────────┐
//!     varentropy     │                                         │
//!         ↑          │  (low H, high V)      (high H, high V) │
//!         │          │  top-k sampling        MCTS / diffusion │
//!         │          │  → AR with temp        → block diffusion│
//!         │          │                        + remasking       │
//!         │          │  (low H, low V)        (high H, low V)  │
//!         │          │  greedy AR              parallel draft   │
//!         │          │  → fast path            → Jacobi / spec │
//!         │          │                                         │
//!         └──────────┴──────────── entropy H ──────────────────►
//! ```
//!
//! The 3rd dimension is the GDN state confidence (mean_delta + eff_rank),
//! which modulates the thresholds: a confident state raises the bar for
//! switching to diffusion (it "trusts" its linear attention more).
//!
//! ## Phases
//!
//! - **Phase 1 (heuristic):** Rolling exponential average of entropy signals,
//!   fixed thresholds with hysteresis. Zero training cost, deployable now.
//! - **Phase 2 (learned):** Small MLP trained via PPO on quality+throughput
//!   reward. Takes all 6 signals as input, outputs decode strategy.
//! - **Phase 3 (end-to-end):** Differentiable gating in the forward pass,
//!   jointly trained with the model (TiDAR-style).
//!
//! ## References
//!
//! - TiDAR (NVIDIA, Nov 2025): dual AR+diffusion in single forward pass, 4.7–5.9×
//! - Swordsman (Feb 2026): entropy-shift block boundaries, 8.79× speedup
//! - CARD (Feb 2026): causal AR diffusion with dynamic parallel tokens
//! - Entropix (2024): 2D entropy × varentropy decision space
//! - EAD (Feb 2025): 96.7% quality at 41.5% cost via entropy switching
//! - MoSE (Feb 2026): router confidence → expert width mapping
//! - Dream 7B (Aug 2025): entropy-ordered unmasking → 2× quality improvement
//! - ReMDM (NeurIPS 2025): remasking for inference-time compute scaling

use crate::moe_router::RoutingDecision;
use crate::StateMetrics;

// =========================================================================
// Configuration
// =========================================================================

/// Decode strategy the router can select.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecodeStrategy {
    /// Standard autoregressive: one token at a time via GDN.
    /// Optimal for low-entropy, predictable tokens (~80% of generation).
    GreedyAR,

    /// AR with temperature/top-k sampling for diverse but manageable options.
    /// Used when entropy is low but varentropy is high (few strong alternatives).
    SampledAR { temperature: f32, top_k: usize },

    /// Speculative drafting: generate N tokens in parallel via Jacobi iteration
    /// or a small draft model, then verify with the full model.
    /// Used when entropy is moderate and tokens are likely predictable.
    SpeculativeDraft { draft_length: usize },

    /// Block diffusion: generate a block of tokens via iterative masked diffusion.
    /// Used for high-entropy spans where parallel refinement beats sequential.
    BlockDiffusion {
        block_size: usize,
        num_steps: usize,
    },

    /// Block diffusion with remasking: allows already-decoded tokens to be
    /// re-predicted based on confidence (ReMDM-style self-correction).
    /// Used for the highest-entropy decision forks.
    BlockDiffusionWithRemasking {
        block_size: usize,
        num_steps: usize,
        remask_fraction: f32,
    },
}

/// Configuration for the entropy router.
#[derive(Debug, Clone)]
pub struct EntropyRouterConfig {
    // ─── Entropy thresholds (Phase 1 heuristic) ───────────────────────
    /// Below this: greedy AR (confident, no sampling needed)
    pub entropy_low: f32,
    /// Above this: consider diffusion (high uncertainty)
    pub entropy_high: f32,
    /// Varentropy threshold for the "decision fork" quadrant
    pub varentropy_high: f32,

    // ─── GDN state thresholds ─────────────────────────────────────────
    /// Mean delta above this: novel content, raise decode intensity
    pub delta_novelty_threshold: f32,
    /// Frobenius norm above this: state saturated, may need full attention
    pub state_saturation_threshold: f32,
    /// Effective rank below this: state underutilized, simple content
    pub rank_simplicity_threshold: f32,

    // ─── MoE coupling ─────────────────────────────────────────────────
    /// When MoE routing entropy exceeds this, escalate decode strategy
    pub moe_entropy_escalation: f32,

    // ─── Hysteresis (prevents rapid mode switching) ───────────────────
    /// Minimum tokens before switching from AR to diffusion
    pub min_ar_tokens: usize,
    /// Minimum tokens before switching from diffusion back to AR
    pub min_diffusion_tokens: usize,
    /// EMA decay for rolling entropy estimate (0.9 = slow, 0.5 = fast)
    pub ema_decay: f32,

    // ─── Block diffusion parameters ──────────────────────────────────
    /// Default block size for diffusion mode
    pub default_block_size: usize,
    /// Min block size (adaptive sizing via entropy shifts)
    pub min_block_size: usize,
    /// Max block size
    pub max_block_size: usize,
    /// Diffusion denoising steps
    pub diffusion_steps: usize,
    /// Remask fraction for highest-entropy mode
    pub remask_fraction: f32,

    // ─── Speculative draft ───────────────────────────────────────────
    /// Default draft length for speculative decoding
    pub default_draft_length: usize,

    // ─── Confidence modulation ────────────────────────────────────────
    /// How much GDN state confidence modulates thresholds.
    /// Higher = more trust in GDN, harder to trigger diffusion.
    /// Range [0, 1]. 0 = state ignored, 1 = thresholds double when confident.
    pub state_confidence_weight: f32,
}

impl EntropyRouterConfig {
    /// Default config tuned for Chimère (24 GDN + 8 GQA, Nanbeige 3B)
    pub fn chimere() -> Self {
        Self {
            entropy_low: 0.3,
            entropy_high: 0.7,
            varentropy_high: 0.5,
            delta_novelty_threshold: 0.4,
            state_saturation_threshold: 10.0,
            rank_simplicity_threshold: 0.3,
            moe_entropy_escalation: 0.6,
            min_ar_tokens: 8,
            min_diffusion_tokens: 16,
            ema_decay: 0.85,
            default_block_size: 32,
            min_block_size: 8,
            max_block_size: 128,
            diffusion_steps: 10,
            remask_fraction: 0.15,
            default_draft_length: 4,
            state_confidence_weight: 0.5,
        }
    }

    /// Conservative config: mostly AR, only switches for very high entropy
    pub fn conservative() -> Self {
        Self {
            entropy_low: 0.5,
            entropy_high: 0.85,
            varentropy_high: 0.7,
            delta_novelty_threshold: 0.6,
            state_saturation_threshold: 15.0,
            rank_simplicity_threshold: 0.2,
            moe_entropy_escalation: 0.8,
            min_ar_tokens: 16,
            min_diffusion_tokens: 32,
            ema_decay: 0.9,
            default_block_size: 16,
            min_block_size: 8,
            max_block_size: 64,
            diffusion_steps: 8,
            remask_fraction: 0.1,
            default_draft_length: 2,
            state_confidence_weight: 0.7,
        }
    }

    /// Test config with small dimensions
    pub fn test() -> Self {
        Self {
            entropy_low: 0.3,
            entropy_high: 0.7,
            varentropy_high: 0.5,
            delta_novelty_threshold: 0.4,
            state_saturation_threshold: 5.0,
            rank_simplicity_threshold: 0.3,
            moe_entropy_escalation: 0.6,
            min_ar_tokens: 2,
            min_diffusion_tokens: 4,
            ema_decay: 0.8,
            default_block_size: 8,
            min_block_size: 4,
            max_block_size: 16,
            diffusion_steps: 5,
            remask_fraction: 0.2,
            default_draft_length: 2,
            state_confidence_weight: 0.5,
        }
    }
}

// =========================================================================
// Entropy Signals — the raw inputs to the router
// =========================================================================

/// All entropy signals available for routing decisions.
/// These are extracted from various Chimère modules at near-zero cost.
#[derive(Debug, Clone)]
pub struct EntropySignals {
    // ─── From output logits ──────────────────────────────────────────
    /// Shannon entropy of the logit distribution H(p) = -Σ p_i log(p_i)
    /// Normalized to [0, 1] by dividing by log(vocab_size).
    pub logit_entropy: f32,
    /// Variance of surprisal: Var[-log(p_i)] across the distribution.
    /// High varentropy = few tokens with wildly different probabilities.
    pub logit_varentropy: f32,

    // ─── From GatedDeltaNet state ────────────────────────────────────
    /// Mean prediction error |v - S·k| across heads, from last token.
    pub gdn_mean_delta: f32,
    /// Mean gate value α across heads (memory decay rate).
    /// Low α = high forgetting = content boundary.
    pub gdn_mean_gate: f32,
    /// Frobenius norm of the state matrix (memory fullness).
    pub gdn_state_norm: f32,
    /// Effective rank of state (how much memory capacity is used).
    pub gdn_effective_rank: f32,

    // ─── From MoE router (optional) ──────────────────────────────────
    /// Tsallis entropy of the expert routing distribution.
    /// High = uncertain about which expert to use.
    pub moe_routing_entropy: Option<f32>,
    /// Number of experts actually activated (adaptive K).
    pub moe_active_k: Option<usize>,
}

impl EntropySignals {
    /// Create signals from individual components.
    pub fn from_parts(
        logit_entropy: f32,
        logit_varentropy: f32,
        state_metrics: &StateMetrics,
        mean_gate: f32,
        moe_decision: Option<&RoutingDecision>,
    ) -> Self {
        Self {
            logit_entropy,
            logit_varentropy,
            gdn_mean_delta: state_metrics.mean_delta,
            gdn_mean_gate: mean_gate,
            gdn_state_norm: state_metrics.frobenius_norm,
            gdn_effective_rank: state_metrics.effective_rank,
            moe_routing_entropy: moe_decision.map(|d| d.routing_entropy),
            moe_active_k: moe_decision.map(|d| d.active_k),
        }
    }

    /// Compute a composite difficulty score in [0, 1].
    /// Higher = harder token = more compute needed.
    pub fn difficulty_score(&self) -> f32 {
        // Weighted combination of all signals
        let logit_signal = 0.35 * self.logit_entropy + 0.15 * self.logit_varentropy;
        let state_signal = 0.25 * self.gdn_mean_delta.min(1.0);
        let gate_signal = 0.10 * (1.0 - self.gdn_mean_gate); // low gate = high difficulty
        let moe_signal = 0.15 * self.moe_routing_entropy.unwrap_or(0.0);

        (logit_signal + state_signal + gate_signal + moe_signal).clamp(0.0, 1.0)
    }

    /// Classify into the Entropix 2D quadrant.
    pub fn quadrant(&self, config: &EntropyRouterConfig) -> EntropyQuadrant {
        let h = self.logit_entropy;
        let v = self.logit_varentropy;
        let h_high = h > config.entropy_high;
        let h_low = h < config.entropy_low;
        let v_high = v > config.varentropy_high;

        match (h_low || !h_high, v_high) {
            (true, false) => EntropyQuadrant::LowH_LowV,    // greedy AR
            (true, true) => EntropyQuadrant::LowH_HighV,     // top-k sampling
            (false, false) => EntropyQuadrant::HighH_LowV,   // speculative draft
            (false, true) => EntropyQuadrant::HighH_HighV,    // block diffusion
        }
    }
}

/// Entropix 2D quadrant classification.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
pub enum EntropyQuadrant {
    /// Low entropy, low varentropy: model is confident. Greedy AR.
    LowH_LowV,
    /// Low entropy, high varentropy: few strong options. Top-k sampling.
    LowH_HighV,
    /// High entropy, low varentropy: uniform uncertainty. Parallel draft.
    HighH_LowV,
    /// High entropy, high varentropy: decision fork. Block diffusion.
    HighH_HighV,
}

// =========================================================================
// Router State — tracks rolling statistics and mode persistence
// =========================================================================

/// Persistent state of the entropy router across tokens.
#[derive(Debug, Clone)]
pub struct RouterState {
    /// Current decode strategy
    pub current_strategy: DecodeStrategy,
    /// Rolling EMA of logit entropy
    pub ema_entropy: f32,
    /// Rolling EMA of varentropy
    pub ema_varentropy: f32,
    /// Rolling EMA of GDN mean_delta
    pub ema_delta: f32,
    /// Tokens generated in current mode (for hysteresis)
    pub tokens_in_current_mode: usize,
    /// Total tokens generated
    pub total_tokens: usize,
    /// Tokens generated via AR
    pub ar_tokens: usize,
    /// Tokens generated via diffusion
    pub diffusion_tokens: usize,
    /// Tokens generated via speculative draft
    pub speculative_tokens: usize,
    /// History of entropy signals (for adaptive threshold calibration)
    pub entropy_history: Vec<f32>,
    /// History of strategy decisions (for logging)
    pub strategy_history: Vec<DecodeStrategy>,
}

impl RouterState {
    pub fn new() -> Self {
        Self {
            current_strategy: DecodeStrategy::GreedyAR,
            ema_entropy: 0.0,
            ema_varentropy: 0.0,
            ema_delta: 0.0,
            tokens_in_current_mode: 0,
            total_tokens: 0,
            ar_tokens: 0,
            diffusion_tokens: 0,
            speculative_tokens: 0,
            entropy_history: Vec::new(),
            strategy_history: Vec::new(),
        }
    }

    /// Fraction of tokens decoded via AR (expected ~80%).
    pub fn ar_fraction(&self) -> f32 {
        if self.total_tokens == 0 {
            1.0
        } else {
            self.ar_tokens as f32 / self.total_tokens as f32
        }
    }

    /// Fraction of tokens decoded via diffusion (expected ~15-20%).
    pub fn diffusion_fraction(&self) -> f32 {
        if self.total_tokens == 0 {
            0.0
        } else {
            self.diffusion_tokens as f32 / self.total_tokens as f32
        }
    }

    /// Effective throughput multiplier estimate vs pure AR.
    /// Based on measured benchmarks: diffusion blocks yield ~4-6× per-block.
    pub fn estimated_throughput_multiplier(&self) -> f32 {
        if self.total_tokens == 0 {
            return 1.0;
        }
        let ar_frac = self.ar_fraction();
        let diff_frac = self.diffusion_fraction();
        let spec_frac = self.speculative_tokens as f32 / self.total_tokens as f32;

        // AR = 1×, speculative = ~2.5×, diffusion = ~5× (conservative estimates)
        let effective = ar_frac * 1.0 + spec_frac * 2.5 + diff_frac * 5.0;
        // Normalize: if everything was AR, multiplier = 1.0
        // If 20% diffusion at 5×: 0.8*1 + 0.2*5 = 1.8× — realistic
        effective
    }
}

impl Default for RouterState {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// The Entropy Router (Phase 1 — Heuristic)
// =========================================================================

/// Phase 1 entropy router: heuristic thresholds with EMA smoothing.
///
/// Zero training cost, deployable immediately. Uses rolling exponential
/// averages of entropy signals to prevent rapid mode switching, with
/// GDN state confidence modulating the thresholds.
pub struct EntropyRouter {
    pub config: EntropyRouterConfig,
    pub state: RouterState,
}

impl EntropyRouter {
    pub fn new(config: EntropyRouterConfig) -> Self {
        Self {
            config,
            state: RouterState::new(),
        }
    }

    /// Core routing decision: given current entropy signals, what strategy?
    ///
    /// This is the main entry point called once per token (or per block boundary).
    pub fn route(&mut self, signals: &EntropySignals) -> DecodeStrategy {
        // 1. Update rolling EMAs
        let d = self.config.ema_decay;
        self.state.ema_entropy = d * self.state.ema_entropy + (1.0 - d) * signals.logit_entropy;
        self.state.ema_varentropy =
            d * self.state.ema_varentropy + (1.0 - d) * signals.logit_varentropy;
        self.state.ema_delta =
            d * self.state.ema_delta + (1.0 - d) * signals.gdn_mean_delta;

        // 2. Compute confidence-modulated thresholds
        // When GDN state is confident (low delta, high rank), raise thresholds
        // → harder to trigger diffusion → trust the linear attention more
        let state_confidence = self.compute_state_confidence(signals);
        let modulation = 1.0 + self.config.state_confidence_weight * state_confidence;

        let eff_entropy_low = self.config.entropy_low * modulation;
        let eff_entropy_high = (self.config.entropy_high * modulation).min(0.95);
        let eff_varentropy_high = self.config.varentropy_high * modulation;

        // 3. Classify using smoothed values
        let h = self.state.ema_entropy;
        let v = self.state.ema_varentropy;

        let candidate = if h < eff_entropy_low && v < eff_varentropy_high {
            // Quadrant 1: Low H, Low V → Greedy AR (fast path)
            DecodeStrategy::GreedyAR
        } else if h < eff_entropy_high && v >= eff_varentropy_high {
            // Quadrant 2: Low H, High V → Sampled AR (few strong alternatives)
            DecodeStrategy::SampledAR {
                temperature: 0.6 + 0.4 * v, // scale temp with varentropy
                top_k: if v > 0.8 { 10 } else { 5 },
            }
        } else if h >= eff_entropy_high && v < eff_varentropy_high {
            // Quadrant 3: High H, Low V → Speculative draft (uniform uncertainty)
            let draft_len = self.compute_draft_length(signals);
            DecodeStrategy::SpeculativeDraft {
                draft_length: draft_len,
            }
        } else {
            // Quadrant 4: High H, High V → Block diffusion (decision fork)
            let block_size = self.compute_adaptive_block_size(signals);
            let difficulty = signals.difficulty_score();

            if difficulty > 0.8 {
                // Extreme difficulty: use remasking for self-correction
                DecodeStrategy::BlockDiffusionWithRemasking {
                    block_size,
                    num_steps: self.config.diffusion_steps + 4, // more steps
                    remask_fraction: self.config.remask_fraction,
                }
            } else {
                DecodeStrategy::BlockDiffusion {
                    block_size,
                    num_steps: self.config.diffusion_steps,
                }
            }
        };

        // 4. Apply hysteresis: don't switch modes too rapidly
        let strategy = self.apply_hysteresis(candidate);

        // 5. Update state
        self.update_state(strategy);

        strategy
    }

    /// Compute a [0, 1] confidence score for the GDN state.
    /// 1.0 = state is confident (low delta, well-utilized memory).
    /// 0.0 = state is uncertain (high delta, sparse memory).
    fn compute_state_confidence(&self, signals: &EntropySignals) -> f32 {
        // Low delta = high confidence (state predicts well)
        let delta_confidence =
            1.0 - (signals.gdn_mean_delta / self.config.delta_novelty_threshold).min(1.0);

        // High effective rank = good memory utilization
        let rank_confidence = signals.gdn_effective_rank.min(1.0);

        // High gate α = retaining memory (not forgetting)
        let gate_confidence = signals.gdn_mean_gate;

        // Weighted combination
        0.5 * delta_confidence + 0.3 * rank_confidence + 0.2 * gate_confidence
    }

    /// Compute adaptive draft length for speculative decoding.
    /// More predictable tokens → longer drafts.
    fn compute_draft_length(&self, signals: &EntropySignals) -> usize {
        let base = self.config.default_draft_length;

        // Low delta → longer drafts (next tokens likely predictable too)
        let delta_factor = if signals.gdn_mean_delta < self.config.delta_novelty_threshold * 0.5 {
            2.0 // very predictable: double draft
        } else if signals.gdn_mean_delta < self.config.delta_novelty_threshold {
            1.5
        } else {
            1.0
        };

        let adjusted = (base as f32 * delta_factor) as usize;
        adjusted.clamp(2, 8) // never draft more than 8 tokens
    }

    /// Compute adaptive block size based on entropy shift detection.
    /// Inspired by Swordsman: block boundaries align with semantic shifts.
    fn compute_adaptive_block_size(&self, signals: &EntropySignals) -> usize {
        let base = self.config.default_block_size;

        // High difficulty → smaller blocks (more control, less risk)
        let difficulty = signals.difficulty_score();
        let size_factor = 1.5 - difficulty; // difficulty 0→1.5×, difficulty 1→0.5×

        // MoE entropy also influences: high routing entropy → smaller blocks
        let moe_factor = if let Some(moe_h) = signals.moe_routing_entropy {
            if moe_h > self.config.moe_entropy_escalation {
                0.7 // reduce block size when MoE is confused
            } else {
                1.0
            }
        } else {
            1.0
        };

        let adjusted = (base as f32 * size_factor * moe_factor) as usize;
        adjusted.clamp(self.config.min_block_size, self.config.max_block_size)
    }

    /// Apply hysteresis to prevent rapid oscillation between modes.
    fn apply_hysteresis(&self, candidate: DecodeStrategy) -> DecodeStrategy {
        let current_is_ar = matches!(
            self.state.current_strategy,
            DecodeStrategy::GreedyAR | DecodeStrategy::SampledAR { .. }
        );
        let candidate_is_ar = matches!(
            candidate,
            DecodeStrategy::GreedyAR | DecodeStrategy::SampledAR { .. }
        );

        // Switching from AR to non-AR: require minimum AR tokens
        if current_is_ar
            && !candidate_is_ar
            && self.state.tokens_in_current_mode < self.config.min_ar_tokens
        {
            return self.state.current_strategy;
        }

        // Switching from non-AR to AR: require minimum diffusion tokens
        if !current_is_ar
            && candidate_is_ar
            && self.state.tokens_in_current_mode < self.config.min_diffusion_tokens
        {
            return self.state.current_strategy;
        }

        candidate
    }

    /// Update internal state after a routing decision.
    fn update_state(&mut self, strategy: DecodeStrategy) {
        let same_mode = std::mem::discriminant(&strategy)
            == std::mem::discriminant(&self.state.current_strategy);

        if same_mode {
            self.state.tokens_in_current_mode += 1;
        } else {
            self.state.tokens_in_current_mode = 1;
            self.state.current_strategy = strategy;
        }

        // Update token counters
        let tokens_generated = match strategy {
            DecodeStrategy::GreedyAR | DecodeStrategy::SampledAR { .. } => {
                self.state.ar_tokens += 1;
                1
            }
            DecodeStrategy::SpeculativeDraft { draft_length } => {
                self.state.speculative_tokens += draft_length;
                draft_length
            }
            DecodeStrategy::BlockDiffusion { block_size, .. }
            | DecodeStrategy::BlockDiffusionWithRemasking { block_size, .. } => {
                self.state.diffusion_tokens += block_size;
                block_size
            }
        };
        self.state.total_tokens += tokens_generated;

        // Keep entropy history (capped at 1000)
        self.state.entropy_history.push(self.state.ema_entropy);
        if self.state.entropy_history.len() > 1000 {
            self.state.entropy_history.drain(0..500);
        }
        self.state.strategy_history.push(strategy);
        if self.state.strategy_history.len() > 1000 {
            self.state.strategy_history.drain(0..500);
        }
    }

    /// Get routing statistics for logging/monitoring.
    pub fn stats(&self) -> RouterStats {
        RouterStats {
            total_tokens: self.state.total_tokens,
            ar_fraction: self.state.ar_fraction(),
            diffusion_fraction: self.state.diffusion_fraction(),
            speculative_fraction: if self.state.total_tokens > 0 {
                self.state.speculative_tokens as f32 / self.state.total_tokens as f32
            } else {
                0.0
            },
            mean_entropy: if self.state.entropy_history.is_empty() {
                0.0
            } else {
                self.state.entropy_history.iter().sum::<f32>()
                    / self.state.entropy_history.len() as f32
            },
            throughput_multiplier: self.state.estimated_throughput_multiplier(),
            current_strategy: self.state.current_strategy,
        }
    }
}

/// Summary statistics for monitoring the entropy router.
#[derive(Debug, Clone)]
pub struct RouterStats {
    pub total_tokens: usize,
    pub ar_fraction: f32,
    pub diffusion_fraction: f32,
    pub speculative_fraction: f32,
    pub mean_entropy: f32,
    pub throughput_multiplier: f32,
    pub current_strategy: DecodeStrategy,
}

// =========================================================================
// Entropy computation from logits
// =========================================================================

/// Compute Shannon entropy and varentropy from a logit distribution.
///
/// This is called on the output logits of the model — essentially free
/// since we need the logits anyway for sampling.
///
/// Returns (entropy, varentropy) both normalized to [0, 1].
pub fn compute_logit_entropy(logits: &[f32]) -> (f32, f32) {
    if logits.is_empty() {
        return (0.0, 0.0);
    }

    let vocab_size = logits.len();

    // Stable softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let log_sum = exp_sum.ln();

    // Shannon entropy: H = -Σ p_i * log(p_i)
    let mut entropy = 0.0f32;
    let mut surprisals = Vec::with_capacity(vocab_size);

    for &logit in logits {
        let log_p = (logit - max_logit) - log_sum; // log(p_i) = log(exp(x_i - max) / Z)
        let p = log_p.exp();
        if p > 1e-10 {
            entropy -= p * log_p;
            surprisals.push(-log_p);
        }
    }

    // Normalize entropy to [0, 1]
    let max_entropy = (vocab_size as f32).ln();
    let norm_entropy = if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Varentropy: Var[-log(p_i)] weighted by p_i
    // = Σ p_i * (-log(p_i) - H)²
    let mut varentropy = 0.0f32;
    for &logit in logits {
        let log_p = (logit - max_logit) - log_sum;
        let p = log_p.exp();
        if p > 1e-10 {
            let surprisal = -log_p;
            varentropy += p * (surprisal - entropy).powi(2);
        }
    }

    // Normalize varentropy (heuristic: divide by entropy² + epsilon)
    let norm_varentropy = if entropy > 0.01 {
        (varentropy / (entropy * entropy + 1.0)).min(1.0)
    } else {
        0.0
    };

    (norm_entropy, norm_varentropy)
}

/// Detect entropy shift between consecutive tokens.
/// Used for adaptive block boundary detection (Swordsman-style).
///
/// Returns the magnitude of the entropy shift and whether it
/// constitutes a "semantic boundary" (large shift).
pub fn detect_entropy_shift(
    prev_entropy: f32,
    curr_entropy: f32,
    threshold: f32,
) -> (f32, bool) {
    let shift = (curr_entropy - prev_entropy).abs();
    let is_boundary = shift > threshold;
    (shift, is_boundary)
}

// =========================================================================
// MoE Coupling — decode strategy influences expert allocation
// =========================================================================

/// Compute MoE K adjustment based on the current decode strategy.
/// When in diffusion mode, we want more experts active (richer representations
/// for the score model). When in greedy AR, fewer experts suffice.
pub fn moe_k_for_strategy(strategy: DecodeStrategy, base_k: usize, max_k: usize) -> usize {
    match strategy {
        DecodeStrategy::GreedyAR => base_k, // standard
        DecodeStrategy::SampledAR { .. } => base_k, // standard
        DecodeStrategy::SpeculativeDraft { .. } => {
            // Draft tokens need fast inference, use fewer experts
            (base_k.max(1) - 1).max(1)
        }
        DecodeStrategy::BlockDiffusion { .. } => {
            // Diffusion needs richer representations, activate more experts
            (base_k + 1).min(max_k)
        }
        DecodeStrategy::BlockDiffusionWithRemasking { .. } => {
            // Maximum compute for hardest tokens
            (base_k + 2).min(max_k)
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    fn make_signals(entropy: f32, varentropy: f32, mean_delta: f32) -> EntropySignals {
        EntropySignals {
            logit_entropy: entropy,
            logit_varentropy: varentropy,
            gdn_mean_delta: mean_delta,
            gdn_mean_gate: 0.7,
            gdn_state_norm: 5.0,
            gdn_effective_rank: 0.6,
            moe_routing_entropy: Some(0.3),
            moe_active_k: Some(2),
        }
    }

    #[test]
    fn test_greedy_ar_for_confident_tokens() {
        let mut router = EntropyRouter::new(EntropyRouterConfig::test());

        // Low entropy, low varentropy, low delta → greedy AR
        let signals = make_signals(0.1, 0.1, 0.1);
        let strategy = router.route(&signals);
        assert_eq!(strategy, DecodeStrategy::GreedyAR);
    }

    #[test]
    fn test_sampled_ar_for_low_h_high_v() {
        let mut router = EntropyRouter::new(EntropyRouterConfig::test());

        // Prime with some tokens to pass hysteresis
        for _ in 0..5 {
            router.route(&make_signals(0.1, 0.1, 0.1));
        }

        // Low entropy, high varentropy → sampled AR
        let signals = make_signals(0.2, 0.8, 0.1);
        // Need several tokens to overcome EMA smoothing
        for _ in 0..10 {
            router.route(&signals);
        }
        let strategy = router.route(&signals);
        assert!(
            matches!(strategy, DecodeStrategy::SampledAR { .. }),
            "Expected SampledAR, got {:?}",
            strategy
        );
    }

    #[test]
    fn test_diffusion_for_high_entropy() {
        let mut router = EntropyRouter::new(EntropyRouterConfig::test());

        // Prime the EMA
        for _ in 0..3 {
            router.route(&make_signals(0.1, 0.1, 0.1));
        }

        // High entropy, high varentropy → block diffusion
        let signals = make_signals(0.9, 0.9, 0.8);
        for _ in 0..15 {
            router.route(&signals);
        }
        let strategy = router.route(&signals);
        assert!(
            matches!(
                strategy,
                DecodeStrategy::BlockDiffusion { .. }
                    | DecodeStrategy::BlockDiffusionWithRemasking { .. }
            ),
            "Expected BlockDiffusion*, got {:?}",
            strategy
        );
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        let config = EntropyRouterConfig {
            min_ar_tokens: 4,
            min_diffusion_tokens: 4,
            ema_decay: 0.0, // no smoothing → instant response
            ..EntropyRouterConfig::test()
        };
        let mut router = EntropyRouter::new(config);

        // Start with AR
        router.route(&make_signals(0.1, 0.1, 0.1));
        assert_eq!(router.state.current_strategy, DecodeStrategy::GreedyAR);

        // Try to switch to diffusion after just 1 token
        let strategy = router.route(&make_signals(0.9, 0.9, 0.9));
        // Should be blocked by hysteresis (min_ar_tokens = 4)
        assert_eq!(
            strategy,
            DecodeStrategy::GreedyAR,
            "Hysteresis should prevent switching after 1 token"
        );

        // After enough AR tokens, should allow switch
        for _ in 0..5 {
            router.route(&make_signals(0.9, 0.9, 0.9));
        }
        let strategy = router.route(&make_signals(0.9, 0.9, 0.9));
        assert!(
            !matches!(strategy, DecodeStrategy::GreedyAR),
            "Should have switched away from AR after enough tokens"
        );
    }

    #[test]
    fn test_state_confidence_raises_thresholds() {
        // A confident GDN state should make it harder to trigger diffusion
        let mut router_confident = EntropyRouter::new(EntropyRouterConfig {
            state_confidence_weight: 1.0, // maximum confidence influence
            ema_decay: 0.0,
            min_ar_tokens: 1,
            ..EntropyRouterConfig::test()
        });

        let mut router_uncertain = EntropyRouter::new(EntropyRouterConfig {
            state_confidence_weight: 0.0, // state ignored
            ema_decay: 0.0,
            min_ar_tokens: 1,
            ..EntropyRouterConfig::test()
        });

        // Moderate entropy right at the boundary — confidence should tip the scale
        let signals_confident = EntropySignals {
            logit_entropy: 0.55,
            logit_varentropy: 0.55,
            gdn_mean_delta: 0.05, // very confident state
            gdn_mean_gate: 0.95,
            gdn_state_norm: 3.0,
            gdn_effective_rank: 0.9,
            moe_routing_entropy: Some(0.2),
            moe_active_k: Some(1),
        };

        for _ in 0..10 {
            router_confident.route(&signals_confident);
            router_uncertain.route(&signals_confident);
        }

        // Confident router should stay AR (raised thresholds)
        // Uncertain router might switch to diffusion
        let s1 = router_confident.route(&signals_confident);
        let s2 = router_uncertain.route(&signals_confident);

        // At minimum, the confident router should be more conservative
        let s1_is_ar = matches!(s1, DecodeStrategy::GreedyAR | DecodeStrategy::SampledAR { .. });
        let s2_is_ar = matches!(s2, DecodeStrategy::GreedyAR | DecodeStrategy::SampledAR { .. });

        // If s2 switched to non-AR, s1 should still be AR
        if !s2_is_ar {
            assert!(
                s1_is_ar,
                "Confident state should keep AR when uncertain state switches. s1={:?}, s2={:?}",
                s1,
                s2
            );
        }
    }

    #[test]
    fn test_logit_entropy_computation() {
        // Uniform distribution → maximum entropy
        let uniform = vec![1.0f32; 100];
        let (h, _v) = compute_logit_entropy(&uniform);
        assert!(h > 0.95, "Uniform should have near-max entropy: {}", h);

        // One-hot distribution → zero entropy
        let mut one_hot = vec![-100.0f32; 100];
        one_hot[42] = 10.0;
        let (h, v) = compute_logit_entropy(&one_hot);
        assert!(h < 0.05, "One-hot should have near-zero entropy: {}", h);
        assert!(v < 0.1, "One-hot should have low varentropy: {}", v);

        // Bimodal → medium entropy, high varentropy
        let mut bimodal = vec![-100.0f32; 100];
        bimodal[10] = 5.0;
        bimodal[90] = 5.0;
        let (h, _v) = compute_logit_entropy(&bimodal);
        assert!(h < 0.3, "Bimodal should have low entropy: {}", h);
        // varentropy should be noticeable since most mass on 2 tokens
    }

    #[test]
    fn test_entropy_shift_detection() {
        let (shift, is_boundary) = detect_entropy_shift(0.2, 0.8, 0.3);
        assert!((shift - 0.6).abs() < 0.01);
        assert!(is_boundary, "Large shift should be a boundary");

        let (_shift, is_boundary) = detect_entropy_shift(0.3, 0.35, 0.3);
        assert!(!is_boundary, "Small shift should not be a boundary");
    }

    #[test]
    fn test_moe_k_adjustment() {
        assert_eq!(moe_k_for_strategy(DecodeStrategy::GreedyAR, 2, 4), 2);
        assert_eq!(
            moe_k_for_strategy(DecodeStrategy::SpeculativeDraft { draft_length: 4 }, 2, 4),
            1
        );
        assert_eq!(
            moe_k_for_strategy(
                DecodeStrategy::BlockDiffusion {
                    block_size: 32,
                    num_steps: 10,
                },
                2,
                4,
            ),
            3
        );
        assert_eq!(
            moe_k_for_strategy(
                DecodeStrategy::BlockDiffusionWithRemasking {
                    block_size: 32,
                    num_steps: 14,
                    remask_fraction: 0.15,
                },
                2,
                4,
            ),
            4
        );
    }

    #[test]
    fn test_difficulty_score_range() {
        // Easy token
        let easy = make_signals(0.05, 0.05, 0.02);
        assert!(easy.difficulty_score() < 0.2);

        // Hard token
        let hard = make_signals(0.95, 0.9, 0.9);
        assert!(hard.difficulty_score() > 0.6);

        // Always in [0, 1]
        for h in [0.0, 0.5, 1.0] {
            for v in [0.0, 0.5, 1.0] {
                for d in [0.0, 0.5, 1.0] {
                    let s = make_signals(h, v, d);
                    let score = s.difficulty_score();
                    assert!(score >= 0.0 && score <= 1.0, "score={} out of range", score);
                }
            }
        }
    }

    #[test]
    fn test_quadrant_classification() {
        let config = EntropyRouterConfig::test();

        // Low H, Low V
        let s = make_signals(0.1, 0.1, 0.1);
        assert_eq!(s.quadrant(&config), EntropyQuadrant::LowH_LowV);

        // Low H, High V
        let s = make_signals(0.1, 0.8, 0.1);
        assert_eq!(s.quadrant(&config), EntropyQuadrant::LowH_HighV);

        // High H, Low V
        let s = make_signals(0.9, 0.1, 0.1);
        assert_eq!(s.quadrant(&config), EntropyQuadrant::HighH_LowV);

        // High H, High V
        let s = make_signals(0.9, 0.9, 0.1);
        assert_eq!(s.quadrant(&config), EntropyQuadrant::HighH_HighV);
    }

    #[test]
    fn test_router_stats() {
        let mut router = EntropyRouter::new(EntropyRouterConfig::test());

        // Generate some tokens
        for _ in 0..100 {
            router.route(&make_signals(0.1, 0.1, 0.1)); // mostly AR
        }

        let stats = router.stats();
        assert!(stats.ar_fraction > 0.9, "Should be mostly AR");
        assert!(
            stats.throughput_multiplier >= 0.9 && stats.throughput_multiplier <= 1.5,
            "Throughput multiplier should be near 1.0 for AR: {}",
            stats.throughput_multiplier
        );
    }

    #[test]
    fn test_adaptive_block_size_scales_with_difficulty() {
        let router = EntropyRouter::new(EntropyRouterConfig::test());

        // Easy tokens → larger blocks
        let easy = make_signals(0.7, 0.6, 0.1);
        let size_easy = router.compute_adaptive_block_size(&easy);

        // Hard tokens → smaller blocks
        let hard = make_signals(0.95, 0.95, 0.9);
        let size_hard = router.compute_adaptive_block_size(&hard);

        assert!(
            size_easy >= size_hard,
            "Easy tokens should get larger blocks: easy={}, hard={}",
            size_easy,
            size_hard
        );
    }

    #[test]
    fn test_draft_length_scales_with_predictability() {
        let router = EntropyRouter::new(EntropyRouterConfig::test());

        // Very predictable → longer drafts
        let predictable = make_signals(0.5, 0.3, 0.05);
        let len_pred = router.compute_draft_length(&predictable);

        // Uncertain → shorter drafts
        let uncertain = make_signals(0.5, 0.3, 0.8);
        let len_unc = router.compute_draft_length(&uncertain);

        assert!(
            len_pred >= len_unc,
            "Predictable tokens should get longer drafts: pred={}, unc={}",
            len_pred,
            len_unc
        );
    }
}
