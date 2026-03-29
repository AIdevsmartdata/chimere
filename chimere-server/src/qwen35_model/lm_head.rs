//! LM head forward + MTP (multi-token prediction) compute.
//!
//! Extracted from qwen35_model.rs — pure code movement, zero behavioral changes.

use candle_core::quantized::QMatMul;
use candle_core::{Device, Module, Result, Tensor};

use super::Qwen35Model;
use crate::activations::rms_norm;

impl Qwen35Model {
    /// Compute MTP logits using the hidden state from the last forward_token call
    /// and the embedding of the predicted next token.
    ///
    /// # MTP Architecture (DeepSeek/Qwen3.5)
    /// - e_norm = RMSNorm(embed(predicted_token), enorm_weight)
    /// - h_norm = RMSNorm(last_hidden_state, hnorm_weight)
    /// - projected = eh_proj([e_norm, h_norm])
    /// - mtp_logits = lm_head(shared_head_norm(projected))
    ///
    /// This predicts the token AFTER predicted_token (i.e., token N+2
    /// if predicted_token is the main head's prediction for N+1).
    pub fn compute_mtp(&self, predicted_token: u32) -> Result<Option<Tensor>> {
        if !self.has_mtp_head {
            return Ok(None);
        }

        let hidden = self.last_hidden.borrow();
        let hidden = match hidden.as_ref() {
            Some(h) => h,
            None => return Ok(None),
        };

        let eps = self.config.rms_norm_eps;

        if let (Some(mtp), Some(enorm), Some(hnorm), Some(shn)) =
            (&self.mtp_head, &self.mtp_enorm, &self.mtp_hnorm, &self.mtp_shared_head_norm)
        {
            let lm_head = self.lm_head.as_ref().unwrap();
            let embed_tokens = self.embed_tokens.as_ref().unwrap();

            // Embed the PREDICTED next token (not the input token)
            let tok_tensor = Tensor::new(&[predicted_token], &Device::Cpu)?;
            let predicted_embd = embed_tokens.index_select(&tok_tensor, 0)?;
            let predicted_embd = predicted_embd.to_device(&self.device)?;

            // e_norm: normalize the predicted token's embedding
            let e_norm = rms_norm(&predicted_embd, enorm, eps)?;
            // h_norm: normalize the hidden state from last forward pass
            let h_norm = rms_norm(hidden, hnorm, eps)?;
            // Concat [e, h] — "eh_proj" weight expects e first, h second
            let concat = Tensor::cat(&[&e_norm, &h_norm], 1)?;
            let projected = mtp.eh_proj.forward(&concat)?;
            let projected = rms_norm(&projected, shn, eps)?;
            let mtp_logits = Self::lm_head_forward(lm_head, &projected)?;
            Ok(Some(mtp_logits))
        } else {
            Ok(None)
        }
    }

    /// Helper: run lm_head forward, handling CPU/GPU transfer if lm_head is on CPU.
    pub(crate) fn lm_head_forward(lm_head: &QMatMul, hidden: &Tensor) -> Result<Tensor> {
        // Check if hidden is on GPU but lm_head expects CPU
        // QMatMul panics (unreachable!) if device mismatch, so we must check first.
        use once_cell::sync::Lazy;
        static LM_CPU: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_LM_HEAD_CPU").is_ok());
        if *LM_CPU {
            let h_cpu = hidden.to_device(&Device::Cpu)?;
            lm_head.forward(&h_cpu)
        } else {
            lm_head.forward(hidden)
        }
    }
}
