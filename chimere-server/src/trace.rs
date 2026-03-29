//! # Tracing Module — Chimere Engine
//!
//! Comprehensive tracing for understanding model internal behavior and guiding
//! performance optimization.
//!
//! Toggle via `CHIMERE_TRACE_LEVEL=0|1|2|3`:
//! - **0** (default): Off. Zero cost — no GPU syncs, no computation.
//! - **1**: Activation statistics, MoE router analysis, layer deltas, GDN state.
//! - **2**: + kernel timing (per-operation wall-clock).
//! - **3**: + binary dumps (reserved for future file I/O).
//!
//! All output goes to stderr via `eprintln!` to avoid interfering with stdout.

use candle_core::{Device, Tensor};
use once_cell::sync::Lazy;

/// Global trace level, read once from `CHIMERE_TRACE_LEVEL` at startup.
static TRACE_LEVEL: Lazy<usize> = Lazy::new(|| {
    std::env::var("CHIMERE_TRACE_LEVEL")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
});

/// Lightweight tracing handle. The `level` field mirrors the global `TRACE_LEVEL`
/// so callers can branch without going through `Lazy` on every call.
pub struct Tracer {
    pub level: usize,
}

impl Tracer {
    /// Create a new tracer. Reads `CHIMERE_TRACE_LEVEL` from the environment
    /// (cached via `once_cell::Lazy` — only the first call pays the env lookup).
    pub fn new() -> Self {
        Self { level: *TRACE_LEVEL }
    }

    /// Returns true if any tracing is enabled (level >= 1).
    #[inline(always)]
    pub fn enabled(&self) -> bool {
        self.level > 0
    }

    /// Returns true if kernel timing is enabled (level >= 2).
    #[inline(always)]
    pub fn timing_enabled(&self) -> bool {
        self.level >= 2
    }

    // -----------------------------------------------------------------
    // Level 1: Activation statistics
    // -----------------------------------------------------------------

    /// Log activation statistics for a tensor at a given layer and component.
    ///
    /// Computes on CPU: min, max, mean, std, L2 norm, and the first 5 values.
    /// Triggers a GPU->CPU sync (acceptable for tracing, not for production).
    pub fn activation_stats(&self, layer: usize, component: &str, tensor: &Tensor) {
        if self.level < 1 {
            return;
        }
        match self.compute_activation_stats(tensor) {
            Ok((min, max, mean, std, l2, first5)) => {
                eprintln!(
                    "[TRACE L{:02}] {}: min={:.4} max={:.4} mean={:.4} std={:.4} L2={:.4} first5={:.4?}",
                    layer, component, min, max, mean, std, l2, first5
                );
            }
            Err(e) => {
                eprintln!("[TRACE L{:02}] {}: stats error: {}", layer, component, e);
            }
        }
    }

    fn compute_activation_stats(
        &self,
        tensor: &Tensor,
    ) -> candle_core::Result<(f32, f32, f32, f32, f32, Vec<f32>)> {
        let flat = tensor.flatten_all()?.to_device(&Device::Cpu)?;
        let data: Vec<f32> = flat.to_vec1()?;
        let n = data.len();
        if n == 0 {
            return Ok((0.0, 0.0, 0.0, 0.0, 0.0, vec![]));
        }

        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = data.iter().map(|&x| x as f64).sum();
        let mean = (sum / n as f64) as f32;
        let var: f64 = data.iter().map(|&x| {
            let d = x as f64 - sum / n as f64;
            d * d
        }).sum::<f64>() / n as f64;
        let std = var.sqrt() as f32;
        let l2: f64 = data.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        let l2 = l2.sqrt() as f32;
        let first5: Vec<f32> = data.iter().take(5).cloned().collect();

        Ok((min, max, mean, std, l2, first5))
    }

    // -----------------------------------------------------------------
    // Level 1: MoE router analysis
    // -----------------------------------------------------------------

    /// Log MoE router statistics: entropy, top-1 confidence, load balance, selected experts.
    ///
    /// # Arguments
    /// - `layer`: layer index
    /// - `probs`: full softmax probability vector over all experts (e.g., 256 values)
    /// - `indices`: selected expert indices (top-K)
    /// - `weights`: selected expert weights (after normalization)
    pub fn router_trace(
        &self,
        layer: usize,
        probs: &[f32],
        indices: &[usize],
        weights: &[f32],
    ) {
        if self.level < 1 {
            return;
        }

        // Entropy: H = -sum(p * ln(p)), skipping zero probabilities
        let entropy: f64 = probs.iter().filter(|&&p| p > 1e-12).map(|&p| {
            let p64 = p as f64;
            -p64 * p64.ln()
        }).sum();

        // Top-1 confidence: the highest probability
        let top1 = probs.iter().cloned().fold(0.0f32, f32::max);

        // Load balance: std(weights) / mean(weights) — coefficient of variation
        let balance = if !weights.is_empty() {
            let w_mean: f64 = weights.iter().map(|&w| w as f64).sum::<f64>() / weights.len() as f64;
            if w_mean > 1e-12 {
                let w_var: f64 = weights.iter().map(|&w| {
                    let d = w as f64 - w_mean;
                    d * d
                }).sum::<f64>() / weights.len() as f64;
                (w_var.sqrt() / w_mean) as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        eprintln!(
            "[TRACE L{:02}] router: entropy={:.3} top1={:.3} balance={:.3} experts={:?}",
            layer, entropy, top1, balance, indices
        );
    }

    // -----------------------------------------------------------------
    // Level 1: Layer delta (how much each layer changes the hidden state)
    // -----------------------------------------------------------------

    /// Log the delta between a layer's input and output hidden states.
    ///
    /// Computes: cosine similarity, residual L2 norm, and residual ratio
    /// (residual_norm / input_norm).
    pub fn layer_delta(&self, layer: usize, hidden_in: &Tensor, hidden_out: &Tensor) {
        if self.level < 1 {
            return;
        }
        match self.compute_layer_delta(hidden_in, hidden_out) {
            Ok((cos_sim, residual_norm, ratio)) => {
                eprintln!(
                    "[TRACE L{:02}] delta: cos_sim={:.4} residual_norm={:.4} ratio={:.4}",
                    layer, cos_sim, residual_norm, ratio
                );
            }
            Err(e) => {
                eprintln!("[TRACE L{:02}] delta: error: {}", layer, e);
            }
        }
    }

    fn compute_layer_delta(
        &self,
        hidden_in: &Tensor,
        hidden_out: &Tensor,
    ) -> candle_core::Result<(f32, f32, f32)> {
        // Move to CPU for scalar computation
        let h_in = hidden_in.flatten_all()?.to_device(&Device::Cpu)?;
        let h_out = hidden_out.flatten_all()?.to_device(&Device::Cpu)?;
        let in_vec: Vec<f32> = h_in.to_vec1()?;
        let out_vec: Vec<f32> = h_out.to_vec1()?;

        let n = in_vec.len();
        if n == 0 || n != out_vec.len() {
            return Ok((0.0, 0.0, 0.0));
        }

        let mut dot: f64 = 0.0;
        let mut in_sq: f64 = 0.0;
        let mut out_sq: f64 = 0.0;
        let mut res_sq: f64 = 0.0;

        for i in 0..n {
            let a = in_vec[i] as f64;
            let b = out_vec[i] as f64;
            dot += a * b;
            in_sq += a * a;
            out_sq += b * b;
            res_sq += (b - a) * (b - a);
        }

        let in_norm = in_sq.sqrt();
        let out_norm = out_sq.sqrt();
        let cos_sim = if in_norm > 1e-12 && out_norm > 1e-12 {
            (dot / (in_norm * out_norm)) as f32
        } else {
            0.0
        };
        let residual_norm = res_sq.sqrt() as f32;
        let ratio = if in_norm > 1e-12 {
            (res_sq.sqrt() / in_norm) as f32
        } else {
            0.0
        };

        Ok((cos_sim, residual_norm, ratio))
    }

    // -----------------------------------------------------------------
    // Level 1: GDN state analysis
    // -----------------------------------------------------------------

    /// Log GDN recurrent state metrics.
    ///
    /// # Arguments
    /// - `layer`: layer index
    /// - `state_frobenius`: Frobenius norm of the state matrix
    /// - `gate_mean`: mean of the gate (alpha/decay) values
    /// - `beta_mean`: mean of the beta (update) values
    pub fn gdn_state_trace(
        &self,
        layer: usize,
        state_frobenius: f32,
        gate_mean: f32,
        beta_mean: f32,
    ) {
        if self.level < 1 {
            return;
        }
        eprintln!(
            "[TRACE L{:02}] gdn_state: ||S||_F={:.2} gate={:.4} beta={:.4}",
            layer, state_frobenius, gate_mean, beta_mean
        );
    }

    // -----------------------------------------------------------------
    // Level 2: Kernel timing helpers
    // -----------------------------------------------------------------

    /// Start a timing measurement. Returns the instant (only meaningful at level >= 2).
    #[inline(always)]
    pub fn timer_start(&self) -> std::time::Instant {
        std::time::Instant::now()
    }

    /// Log elapsed time for a named operation at a given layer.
    /// Only prints at level >= 2.
    pub fn timer_log(&self, layer: usize, operation: &str, start: std::time::Instant) {
        if self.level < 2 {
            return;
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[TRACE L{:02}] timing {}: {:.3}ms",
            layer, operation, elapsed_ms
        );
    }
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}
