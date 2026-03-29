//! # Activations and normalization helpers
//!
//! Pure, stateless helper functions used by the forward pass in `qwen35_model`.
//! All functions dispatch to fused CUDA kernels where available.

use candle_core::{Result, Tensor, D};
use candle_nn::ops as nn_ops;

/// RMSNorm: x * weight / sqrt(mean(x²) + eps)
///
/// Uses candle_nn::ops::rms_norm which dispatches to a single fused CUDA kernel.
/// Forces contiguous input (required by the fused kernel).
pub(crate) fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.contiguous()?;
    nn_ops::rms_norm(&x, weight, eps as f32)
}

/// Sigmoid activation: 1 / (1 + exp(-x))
///
/// Uses candle_nn::ops::sigmoid which dispatches to a single fused CUDA kernel (usigmoid).
pub(crate) fn sigmoid(x: &Tensor) -> Result<Tensor> {
    nn_ops::sigmoid(x)
}

/// SiLU activation: x * sigmoid(x)
///
/// Uses Tensor::silu() which dispatches to a single fused CUDA kernel (usilu).
pub(crate) fn silu_activation(x: &Tensor) -> Result<Tensor> {
    x.silu()
}

/// Softplus: log(1 + exp(x)) — numerically stable version.
///
/// Uses the identity: softplus(x) = max(x, 0) + log(1 + exp(-|x|))
/// which avoids overflow for large positive x (exp(88+) = inf in f32).
pub(crate) fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let pos_part = x.maximum(&zeros)?;              // max(x, 0)
    let neg_abs = x.abs()?.neg()?;                  // -|x|
    let log_part = (neg_abs.exp()? + 1.0)?.log()?;  // log(1 + exp(-|x|))
    &pos_part + &log_part
}

/// L2-normalise along the last dimension: x * rsqrt(max(sum(x²), eps²))
///
/// Matches ggml's CUDA `l2_norm_f32` kernel exactly:
///   `scale = rsqrtf(fmaxf(tmp, eps*eps))` where `tmp = sum(x²)`
///   `dst[i] = x[i] * scale`
pub(crate) fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let sq = x.sqr()?;
    let sum_sq = sq.sum_keepdim(D::Minus1)?;
    let eps_sq = (eps * eps) as f32;
    let eps_tensor = Tensor::full(eps_sq, sum_sq.shape(), sum_sq.device())?;
    let clamped = sum_sq.maximum(&eps_tensor)?;
    let norm = clamped.sqrt()?;
    x.broadcast_div(&norm)
}
