//! # MoE Router — Chimère Engine
//!
//! Entropy-adaptive Mixture of Experts routing with Tsallis entropy
//! for dynamic expert selection and Sinkhorn OT for load balancing.
//!
//! ## Design
//!
//! Unlike Qwen3's fixed top-2 routing, this router adapts the number
//! of active experts per token based on the routing entropy:
//!
//! - **Low entropy** (confident routing) → K=1 expert, minimal compute
//! - **High entropy** (uncertain routing) → K=2..max_k experts, more compute
//!
//! This implements the "entropy-adaptive compute" principle from the
//! Chimère survey: easy tokens cost less than hard tokens.
//!
//! ## Integration with GatedDeltaNet
//!
//! The `StateMetrics` from `gated_deltanet.rs` provide an additional
//! entropy signal: when the DeltaNet state has low prediction error
//! (mean_delta → 0), the token is predictable and can use fewer experts.
//!
//! ## References
//!
//! - Adaptive-K routing: 62% of tokens use K=1, saving 52.5% compute
//! - Tsallis entropy (q=2): H_q(p) = (1 - Σ p_i^q) / (q - 1)
//! - Sinkhorn OT: auxiliary-loss-free load balancing (DeepSeek style)
//! - DRMoLE (arXiv:2504.00661): +9.6% improvement with entropy routing

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use crate::StateMetrics;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// MoE Router configuration.
#[derive(Debug, Clone)]
pub struct MoeRouterConfig {
    /// Model hidden dimension
    pub hidden_dim: usize,
    /// Number of routed experts (excluding shared expert)
    pub num_experts: usize,
    /// Minimum number of active experts per token
    pub min_k: usize,
    /// Maximum number of active experts per token
    pub max_k: usize,
    /// Whether to use a shared expert (always active)
    pub use_shared_expert: bool,
    /// Tsallis entropy order (q=2 is standard)
    pub tsallis_q: f32,
    /// Temperature for gating logits (lower = sharper routing)
    pub temperature: f32,
    /// Scaling factor for routed expert outputs
    pub routed_scaling_factor: f32,
    /// Number of Sinkhorn OT iterations for load balancing
    pub sinkhorn_iterations: usize,
    /// Entropy threshold: below this → use min_k experts
    pub entropy_threshold_low: f32,
    /// Entropy threshold: above this → use max_k experts
    pub entropy_threshold_high: f32,
}

impl MoeRouterConfig {
    /// Chimère default: 1 shared + 8 routed, Adaptive-K 1-4
    pub fn chimere() -> Self {
        Self {
            hidden_dim: 4096,
            num_experts: 8,
            min_k: 1,
            max_k: 4,
            use_shared_expert: true,
            tsallis_q: 2.0,
            temperature: 1.0,
            routed_scaling_factor: 2.5,
            sinkhorn_iterations: 20,
            entropy_threshold_low: 0.3,
            entropy_threshold_high: 0.7,
        }
    }

    /// Small config for unit tests
    pub fn test() -> Self {
        Self {
            hidden_dim: 64,
            num_experts: 4,
            min_k: 1,
            max_k: 3,
            use_shared_expert: true,
            tsallis_q: 2.0,
            temperature: 1.0,
            routed_scaling_factor: 1.0,
            sinkhorn_iterations: 5,
            entropy_threshold_low: 0.3,
            entropy_threshold_high: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Routing output
// ---------------------------------------------------------------------------

/// Result of routing a single token.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Indices of selected experts (length = adaptive K)
    pub expert_ids: Vec<usize>,
    /// Normalized weights for selected experts (sum to 1.0)
    pub expert_weights: Vec<f32>,
    /// Raw gating probabilities for all experts
    pub gate_probs: Vec<f32>,
    /// Tsallis entropy of the gating distribution
    pub routing_entropy: f32,
    /// Number of active experts chosen (adaptive K)
    pub active_k: usize,
}

/// Batch routing result with load balancing stats.
#[derive(Debug, Clone)]
pub struct BatchRoutingResult {
    /// Per-token routing decisions
    pub decisions: Vec<RoutingDecision>,
    /// Expert load distribution (fraction of tokens routed to each)
    pub expert_load: Vec<f32>,
    /// Mean routing entropy across the batch
    pub mean_entropy: f32,
    /// Mean active K across the batch
    pub mean_k: f32,
}

// ---------------------------------------------------------------------------
// Entropy functions (standalone, testable)
// ---------------------------------------------------------------------------

/// Tsallis entropy of order q for a probability distribution.
///
/// H_q(p) = (1 - Σ p_i^q) / (q - 1)
///
/// Properties:
/// - q=1 recovers Shannon entropy (in the limit)
/// - q=2 gives a quadratic form, cheaper to compute
/// - Higher q penalizes dominant probabilities more
///
/// Returns a value in [0, 1] when normalized by max entropy.
pub fn tsallis_entropy(probs: &[f32], q: f32) -> f32 {
    if (q - 1.0).abs() < 1e-6 {
        // Shannon entropy (limit q→1)
        return shannon_entropy(probs);
    }

    let sum_pq: f32 = probs.iter().map(|&p| p.powf(q)).sum();
    let h = (1.0 - sum_pq) / (q - 1.0);

    // Normalize by maximum entropy (uniform distribution)
    let n = probs.len() as f32;
    let h_max = (1.0 - n.powf(1.0 - q)) / (q - 1.0);

    if h_max.abs() < 1e-12 {
        0.0
    } else {
        (h / h_max).clamp(0.0, 1.0)
    }
}

/// Shannon entropy: H(p) = -Σ p_i * log(p_i)
pub fn shannon_entropy(probs: &[f32]) -> f32 {
    let h: f32 = probs
        .iter()
        .filter(|&&p| p > 1e-12)
        .map(|&p| -p * p.ln())
        .sum();

    // Normalize by max entropy
    let n = probs.len() as f32;
    let h_max = n.ln();
    if h_max.abs() < 1e-12 {
        0.0
    } else {
        (h / h_max).clamp(0.0, 1.0)
    }
}

/// Determine adaptive K from normalized entropy and state metrics.
///
/// The key insight: tokens with low routing entropy AND low DeltaNet
/// prediction error need fewer experts (the model is confident).
pub fn adaptive_k(
    routing_entropy: f32,
    state_metrics: Option<&StateMetrics>,
    config: &MoeRouterConfig,
) -> usize {
    // Base K from routing entropy (linear interpolation)
    let t = if config.entropy_threshold_high <= config.entropy_threshold_low {
        if routing_entropy > config.entropy_threshold_low {
            1.0
        } else {
            0.0
        }
    } else {
        ((routing_entropy - config.entropy_threshold_low)
            / (config.entropy_threshold_high - config.entropy_threshold_low))
            .clamp(0.0, 1.0)
    };

    let range = (config.max_k - config.min_k) as f32;
    let mut k = config.min_k as f32 + t * range;

    // Modulate by DeltaNet state: low mean_delta → reduce K
    if let Some(metrics) = state_metrics {
        // When prediction error is low, the state already "knows" this pattern
        // → fewer experts needed. Scale K down by (1 - confidence).
        let confidence = (-metrics.mean_delta).exp().clamp(0.0, 1.0);
        k *= 1.0 - 0.5 * confidence; // At most halve K from state signal
    }

    (k.round() as usize).clamp(config.min_k, config.max_k)
}

// ---------------------------------------------------------------------------
// Softmax gating (standalone, for testing)
// ---------------------------------------------------------------------------

/// Compute softmax gating probabilities from logits.
pub fn softmax_gate(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = if temperature.abs() < 1e-12 {
        1.0
    } else {
        temperature
    };

    // Scale by temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / t).collect();

    // Numerically stable softmax
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    exps.iter().map(|&e| e / sum).collect()
}

/// Select top-k experts from probabilities, renormalize weights.
pub fn topk_select(probs: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let k = k.min(probs.len());
    let ids: Vec<usize> = indexed[..k].iter().map(|&(i, _)| i).collect();
    let weights: Vec<f32> = indexed[..k].iter().map(|&(_, w)| w).collect();

    // Renormalize
    let sum: f32 = weights.iter().sum();
    let weights: Vec<f32> = if sum > 1e-12 {
        weights.iter().map(|&w| w / sum).collect()
    } else {
        vec![1.0 / k as f32; k]
    };

    (ids, weights)
}

// ---------------------------------------------------------------------------
// Sinkhorn OT load balancing
// ---------------------------------------------------------------------------

/// Apply Sinkhorn normalization to balance expert assignments.
///
/// Given a cost matrix (batch_size × num_experts) of gating logits,
/// iteratively normalize rows and columns to achieve a doubly-stochastic
/// matrix, which ensures uniform expert load.
///
/// This is the auxiliary-loss-free approach from DeepSeek:
/// instead of adding a loss term, we directly constrain the assignment.
pub fn sinkhorn_balance(
    logits: &[Vec<f32>],
    iterations: usize,
) -> Vec<Vec<f32>> {
    let n_tokens = logits.len();
    if n_tokens == 0 {
        return vec![];
    }
    let n_experts = logits[0].len();
    if n_experts == 0 {
        return logits.to_vec();
    }

    // Initialize with exp(logits) — the transport plan
    let mut plan: Vec<Vec<f32>> = logits
        .iter()
        .map(|row| {
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            row.iter().map(|&x| (x - max_val).exp()).collect()
        })
        .collect();

    for _ in 0..iterations {
        // Row normalization: each token's distribution sums to 1
        for row in plan.iter_mut() {
            let sum: f32 = row.iter().sum();
            if sum > 1e-12 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        // Column normalization: each expert gets equal share
        // Target: n_tokens / n_experts tokens per expert
        let target = n_tokens as f32 / n_experts as f32;
        for j in 0..n_experts {
            let col_sum: f32 = plan.iter().map(|row| row[j]).sum();
            if col_sum > 1e-12 {
                let scale = target / col_sum;
                for row in plan.iter_mut() {
                    row[j] *= scale;
                }
            }
        }
    }

    // Final row normalization to get valid probabilities
    for row in plan.iter_mut() {
        let sum: f32 = row.iter().sum();
        if sum > 1e-12 {
            for val in row.iter_mut() {
                *val /= sum;
            }
        }
    }

    plan
}

// ---------------------------------------------------------------------------
// MoE Router (full, with learned gate)
// ---------------------------------------------------------------------------

/// Mixture of Experts router with entropy-adaptive K selection.
pub struct MoeRouter {
    pub config: MoeRouterConfig,
    /// Gating projection: hidden_dim → num_experts
    gate: Linear,
}

impl MoeRouter {
    pub fn new(config: MoeRouterConfig, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear_no_bias(
            config.hidden_dim,
            config.num_experts,
            vb.pp("gate"),
        )?;
        Ok(Self { config, gate })
    }

    /// Route a single token (no load balancing, for inference).
    pub fn route_token(
        &self,
        hidden: &Tensor,
        state_metrics: Option<&StateMetrics>,
    ) -> Result<RoutingDecision> {
        // hidden: [hidden_dim]
        let logits = self.gate.forward(&hidden.unsqueeze(0)?)?; // [1, num_experts]
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;

        // Softmax gating with temperature
        let gate_probs = softmax_gate(&logits_vec, self.config.temperature);

        // Compute Tsallis entropy
        let routing_entropy = tsallis_entropy(&gate_probs, self.config.tsallis_q);

        // Adaptive K selection
        let active_k = adaptive_k(routing_entropy, state_metrics, &self.config);

        // Select top-K experts
        let (expert_ids, expert_weights) = topk_select(&gate_probs, active_k);

        Ok(RoutingDecision {
            expert_ids,
            expert_weights,
            gate_probs,
            routing_entropy,
            active_k,
        })
    }

    /// Route a batch of tokens with Sinkhorn load balancing.
    pub fn route_batch(
        &self,
        hidden_states: &Tensor,
        state_metrics: Option<&[StateMetrics]>,
    ) -> Result<BatchRoutingResult> {
        let (n_tokens, _hidden) = hidden_states.dims2()?;

        // Compute all gating logits at once: [n_tokens, num_experts]
        let all_logits = self.gate.forward(hidden_states)?;
        let all_logits_vec: Vec<Vec<f32>> = (0..n_tokens)
            .map(|i| {
                all_logits
                    .narrow(0, i, 1)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            })
            .collect();

        // Apply Sinkhorn balancing to logits
        let balanced = sinkhorn_balance(&all_logits_vec, self.config.sinkhorn_iterations);

        // Route each token
        let mut decisions = Vec::with_capacity(n_tokens);
        let mut total_entropy = 0.0f32;
        let mut total_k = 0.0f32;
        let mut expert_counts = vec![0.0f32; self.config.num_experts];

        for i in 0..n_tokens {
            let gate_probs = &balanced[i];
            let routing_entropy = tsallis_entropy(gate_probs, self.config.tsallis_q);

            let sm = state_metrics.and_then(|m| m.get(i));
            let active_k = adaptive_k(routing_entropy, sm, &self.config);

            let (expert_ids, expert_weights) = topk_select(gate_probs, active_k);

            // Track load
            for &eid in &expert_ids {
                expert_counts[eid] += 1.0;
            }

            total_entropy += routing_entropy;
            total_k += active_k as f32;

            decisions.push(RoutingDecision {
                expert_ids,
                expert_weights,
                gate_probs: gate_probs.clone(),
                routing_entropy,
                active_k,
            });
        }

        // Normalize expert load
        let total_assignments: f32 = expert_counts.iter().sum();
        let expert_load: Vec<f32> = if total_assignments > 0.0 {
            expert_counts
                .iter()
                .map(|&c| c / total_assignments)
                .collect()
        } else {
            vec![1.0 / self.config.num_experts as f32; self.config.num_experts]
        };

        Ok(BatchRoutingResult {
            decisions,
            expert_load,
            mean_entropy: total_entropy / n_tokens as f32,
            mean_k: total_k / n_tokens as f32,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsallis_entropy_uniform() {
        // Uniform distribution should have maximum entropy (normalized to 1.0)
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = tsallis_entropy(&uniform, 2.0);
        assert!(
            (h - 1.0).abs() < 0.01,
            "Uniform distribution entropy should be ~1.0, got {:.4}",
            h
        );
        println!("Uniform entropy (q=2): {:.4}", h);
    }

    #[test]
    fn test_tsallis_entropy_peaked() {
        // Peaked distribution should have low entropy
        let peaked = vec![0.97, 0.01, 0.01, 0.01];
        let h = tsallis_entropy(&peaked, 2.0);
        assert!(
            h < 0.1,
            "Peaked distribution entropy should be < 0.1, got {:.4}",
            h
        );
        println!("Peaked entropy (q=2): {:.4}", h);
    }

    #[test]
    fn test_tsallis_entropy_shannon_limit() {
        // q=1 should give Shannon entropy
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let h_tsallis = tsallis_entropy(&probs, 1.0);
        let h_shannon = shannon_entropy(&probs);
        assert!(
            (h_tsallis - h_shannon).abs() < 0.01,
            "Tsallis(q=1) should equal Shannon: {:.4} vs {:.4}",
            h_tsallis,
            h_shannon
        );
        println!(
            "Shannon limit: Tsallis={:.4}, Shannon={:.4}",
            h_tsallis, h_shannon
        );
    }

    #[test]
    fn test_adaptive_k_varies_with_entropy() {
        let config = MoeRouterConfig::test();

        // Low entropy → min_k
        let k_low = adaptive_k(0.1, None, &config);
        assert_eq!(k_low, config.min_k, "Low entropy should give min_k");

        // High entropy → max_k
        let k_high = adaptive_k(0.9, None, &config);
        assert_eq!(k_high, config.max_k, "High entropy should give max_k");

        // Medium entropy → somewhere in between
        let k_mid = adaptive_k(0.5, None, &config);
        assert!(
            k_mid >= config.min_k && k_mid <= config.max_k,
            "Medium entropy K={} should be in [{}, {}]",
            k_mid,
            config.min_k,
            config.max_k
        );

        println!(
            "Adaptive K: low={}, mid={}, high={}",
            k_low, k_mid, k_high
        );
    }

    #[test]
    fn test_adaptive_k_with_state_metrics() {
        let config = MoeRouterConfig::test();

        // High entropy but confident state → should reduce K
        let confident_state = StateMetrics {
            frobenius_norm: 1.0,
            mean_delta: 0.01, // very low prediction error
            effective_rank: 0.5,
        };

        let k_without = adaptive_k(0.9, None, &config);
        let k_with = adaptive_k(0.9, Some(&confident_state), &config);

        assert!(
            k_with <= k_without,
            "Confident state should reduce K: without={}, with={}",
            k_without,
            k_with
        );

        println!(
            "State modulation: K without state={}, with confident state={}",
            k_without, k_with
        );
    }

    #[test]
    fn test_softmax_gate_basic() {
        let logits = vec![2.0, 1.0, 0.0, -1.0];
        let probs = softmax_gate(&logits, 1.0);

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Softmax should sum to 1.0, got {:.6}",
            sum
        );

        // Should be monotonically decreasing
        for i in 0..probs.len() - 1 {
            assert!(
                probs[i] > probs[i + 1],
                "Softmax should be monotonic: p[{}]={:.4} > p[{}]={:.4}",
                i,
                probs[i],
                i + 1,
                probs[i + 1]
            );
        }

        println!("Softmax gate: {:?}", probs);
    }

    #[test]
    fn test_softmax_temperature() {
        let logits = vec![2.0, 1.0, 0.0, -1.0];

        let sharp = softmax_gate(&logits, 0.1); // low temp → sharp
        let smooth = softmax_gate(&logits, 10.0); // high temp → uniform

        // Sharp should be more peaked (first prob much higher)
        assert!(
            sharp[0] > smooth[0],
            "Low temp should be sharper: {:.4} > {:.4}",
            sharp[0],
            smooth[0]
        );

        // Smooth should be more uniform (last prob higher)
        assert!(
            smooth[3] > sharp[3],
            "High temp should be more uniform: {:.4} > {:.4}",
            smooth[3],
            sharp[3]
        );

        println!("Temperature: sharp={:?}, smooth={:?}", sharp, smooth);
    }

    #[test]
    fn test_topk_select() {
        let probs = vec![0.1, 0.5, 0.05, 0.35];

        let (ids, weights) = topk_select(&probs, 2);
        assert_eq!(ids, vec![1, 3], "Top-2 should be experts 1 and 3");

        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0, got {:.6}",
            sum
        );

        println!("Top-2: ids={:?}, weights={:?}", ids, weights);
    }

    #[test]
    fn test_sinkhorn_balances_load() {
        // Create biased logits: all tokens prefer expert 0
        let n_tokens = 100;
        let n_experts = 4;
        let logits: Vec<Vec<f32>> = (0..n_tokens)
            .map(|_| vec![5.0, 0.0, 0.0, 0.0]) // heavily biased toward expert 0
            .collect();

        let balanced = sinkhorn_balance(&logits, 20);

        // After Sinkhorn, each expert should get ~25% of probability mass
        let mut col_sums = vec![0.0f32; n_experts];
        for row in &balanced {
            for (j, &val) in row.iter().enumerate() {
                col_sums[j] += val;
            }
        }

        // Normalize
        let total: f32 = col_sums.iter().sum();
        let col_fracs: Vec<f32> = col_sums.iter().map(|&c| c / total).collect();

        println!("Sinkhorn load balance: {:?}", col_fracs);

        // Each expert should get roughly equal share (±10%)
        for (j, &frac) in col_fracs.iter().enumerate() {
            assert!(
                (frac - 0.25).abs() < 0.10,
                "Expert {} load {:.3} should be ~0.25 (±0.10)",
                j,
                frac
            );
        }
    }

    #[test]
    fn test_end_to_end_routing_pipeline() {
        // Full pipeline: logits → softmax → entropy → adaptive K → top-K
        let config = MoeRouterConfig::test(); // 4 experts, K=1..3

        // Simulate two scenarios:

        // 1. Confident token: one expert clearly dominates
        let confident_logits = vec![5.0, 0.1, 0.1, 0.1];
        let probs = softmax_gate(&confident_logits, config.temperature);
        let entropy = tsallis_entropy(&probs, config.tsallis_q);
        let k = adaptive_k(entropy, None, &config);
        let (ids, weights) = topk_select(&probs, k);

        println!(
            "Confident: entropy={:.4}, K={}, experts={:?}, weights={:?}",
            entropy, k, ids, weights
        );
        assert_eq!(k, 1, "Confident routing should use K=1");

        // 2. Uncertain token: multiple experts compete
        let uncertain_logits = vec![1.0, 0.9, 0.8, 0.7];
        let probs = softmax_gate(&uncertain_logits, config.temperature);
        let entropy = tsallis_entropy(&probs, config.tsallis_q);
        let k = adaptive_k(entropy, None, &config);
        let (ids, weights) = topk_select(&probs, k);

        println!(
            "Uncertain: entropy={:.4}, K={}, experts={:?}, weights={:?}",
            entropy, k, ids, weights
        );
        assert!(k > 1, "Uncertain routing should use K > 1");
    }
}
