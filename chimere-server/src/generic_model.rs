//! # GenericModel — libllama-only path for non-Qwen architectures
//!
//! This module hosts the multi-arch entry point for any architecture that
//! libllama (ik_llama.cpp's `libllama.so`) can load: today Mamba-1, after the
//! Phase 1 backport Mamba-2 + `nemotron_h_moe`, and any future arch upstream
//! adds. None of the Qwen3.5-specific code paths (MTP, MRoPE, block diffusion,
//! Engram-aware sampling, entropy routing, custom CUDA graphs, cudarc) live
//! here — `Qwen35Model` keeps owning all of that.
//!
//! ## Architecture
//!
//! ```text
//! GenericModel
//!   ├── llama_forward: RefCell<Option<LlamaForward>>   ← libllama FFI handle
//!   ├── arch:          ModelArch                       ← Mamba1 | Mamba2 | NemotronHMoe | ...
//!   ├── n_vocab        usize                           ← cached at load
//!   ├── n_layers       usize                           ← cached at load
//!   ├── chat_template  Option<String>                  ← from GGUF metadata
//!   └── last_packed_logprobs: RefCell<Option<Vec<f32>>>
//! ```
//!
//! `GenericModel` does ONLY what libllama can do directly:
//! - `forward_token` / `forward_prefill` via `LlamaForward`
//! - Tokenization via the GGUF-embedded vocab (`LlamaForward::tokenize`)
//! - Chat templating via the GGUF-embedded template
//!   (`LlamaForward::apply_chat_template`)
//! - Logit biasing for `</think>` suppression / Engram (already supported by
//!   the C++ chimere_sampler)
//!
//! It does NOT support: MTP, DART speculation, block diffusion, entropy
//! routing, MRoPE, cudarc weights. Those flags return `false` from the
//! `ChimereModel::supports_*` methods so callers gracefully fall back.
//!
//! ## State
//!
//! libllama owns the KV cache + recurrent state internally. The
//! `InferenceState` parameter passed through the trait API is therefore
//! IGNORED on this path — we accept any variant. This matches the agent's
//! design intent: the trait state is for callers that hold their own state
//! (Qwen35Model with its `GdnRecurrentState`), and is a no-op for libllama-
//! native models. The `forward_token` impl just calls
//! `LlamaForward::forward_token(token)` and ignores `state`.
//!
//! ## Why borrow internal mutability?
//!
//! `LlamaForward::forward_token` and friends take `&mut self` (because they
//! mutate the libllama context pointer). But the `ChimereModel` trait method
//! takes `&self` to match `Qwen35Model`'s pattern, which uses `RefCell` for
//! the same reason. So `GenericModel` wraps `LlamaForward` in `RefCell` too
//! and borrows mutably inside each forward call.

use std::cell::RefCell;

use candle_core::{Device, Result, Tensor};

use crate::chimere_model::{ChimereModel, ForwardOutput, InferenceState, ModelArch};
use crate::llama_backend::{self, LlamaForward};

/// libllama-backed model for any architecture that ik_llama can load.
///
/// One per process. Construct via `GenericModel::from_env(arch)`.
pub struct GenericModel {
    arch: ModelArch,
    /// libllama FFI handle. `RefCell` so the trait's `&self` methods can
    /// borrow it mutably for forward passes.
    llama_forward: RefCell<Option<LlamaForward>>,
    n_vocab: usize,
    n_layers: usize,
    /// GGUF-embedded chat template (jinja-equivalent string), if any.
    /// Read once at load and stored.
    #[allow(dead_code)]
    chat_template: Option<String>,
    /// Buffer for fast-sampler packed logprobs (mirrors `Qwen35Model`).
    last_packed_logprobs: RefCell<Option<Vec<f32>>>,
}

// Raw pointer inside `LlamaForward` makes us `!Send` by default. The server
// wraps the model in `tokio::sync::Mutex` for thread safety, so a manual
// `unsafe impl Send` is correct here — same pattern as `Qwen35Model`.
unsafe impl Send for GenericModel {}

impl GenericModel {
    /// Load a model via libllama using the standard `CHIMERE_*` env vars
    /// (`CHIMERE_MODEL`, `CHIMERE_NCMOE`, etc. — see `llama_backend::from_env`).
    ///
    /// `arch` should be detected from the GGUF's `general.architecture`
    /// metadata field by the caller (typically `bin/chimere-server.rs`); we
    /// take it as a parameter rather than re-parsing the GGUF here.
    pub fn from_env(arch: ModelArch) -> std::result::Result<Self, String> {
        eprintln!(
            "[GENERIC_MODEL] Loading {} via libllama FFI...",
            arch.name()
        );
        let llama = llama_backend::from_env()?;
        let n_vocab = llama.n_vocab();
        let n_layers = llama.n_layer();
        let chat_template = llama.chat_template(None);
        eprintln!(
            "[GENERIC_MODEL] Loaded: arch={} vocab={} layers={} chat_template={}",
            arch.name(),
            n_vocab,
            n_layers,
            chat_template
                .as_deref()
                .map(|_| "yes")
                .unwrap_or("no (built-in default will be used)"),
        );
        Ok(Self {
            arch,
            llama_forward: RefCell::new(Some(llama)),
            n_vocab,
            n_layers,
            chat_template,
            last_packed_logprobs: RefCell::new(None),
        })
    }

    /// Borrow the underlying `LlamaForward` for direct use by callers that
    /// need the tokenizer / chat template helpers (`bin/chimere-server.rs`,
    /// future agent scheduler hooks).
    pub fn llama_forward(&self) -> std::cell::Ref<'_, Option<LlamaForward>> {
        self.llama_forward.borrow()
    }
}

impl ChimereModel for GenericModel {
    fn arch(&self) -> ModelArch {
        self.arch
    }

    fn num_layers(&self) -> usize {
        self.n_layers
    }

    fn vocab_size(&self) -> usize {
        self.n_vocab
    }

    // -- Capability flags: only what libllama natively supports -----------
    //
    // No MTP (Qwen3.5-only feature, requires the MTP head + custom Rust
    // glue). No block diffusion (Qwen3.5-only). No DART (depends on MTP).
    // No entropy routing (Qwen3.5-only). All four return `false` (the
    // default), which is what we want.

    fn forward_token(
        &self,
        token: u32,
        _state: &mut InferenceState<'_>,
    ) -> Result<ForwardOutput> {
        // Libllama owns the KV / SSM state internally; the `_state`
        // parameter is unused on this path. The borrow is dropped at the
        // end of the call.
        *self.last_packed_logprobs.borrow_mut() = None;

        let mut llama_ref = self.llama_forward.borrow_mut();
        let llama = llama_ref
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(
                "GenericModel: llama_forward not initialized".into(),
            ))?;

        let logits_cpu = llama
            .forward_token(token)
            .map_err(candle_core::Error::Msg)?;

        let n_vocab = logits_cpu.len();
        let logits = Tensor::from_vec(logits_cpu, (1, n_vocab), &Device::Cpu)?;
        Ok(ForwardOutput {
            logits,
            mtp_logits: None,
        })
    }

    fn forward_prefill(
        &self,
        tokens: &[u32],
        _state: &mut InferenceState<'_>,
    ) -> Result<ForwardOutput> {
        if tokens.is_empty() {
            return Err(candle_core::Error::Msg(
                "GenericModel::forward_prefill: tokens slice must not be empty".into(),
            ));
        }

        let mut llama_ref = self.llama_forward.borrow_mut();
        let llama = llama_ref
            .as_mut()
            .ok_or_else(|| candle_core::Error::Msg(
                "GenericModel: llama_forward not initialized".into(),
            ))?;

        let logits_cpu = llama
            .forward_prefill(tokens)
            .map_err(candle_core::Error::Msg)?;

        let n_vocab = logits_cpu.len();
        let logits = Tensor::from_vec(logits_cpu, (1, n_vocab), &Device::Cpu)?;
        Ok(ForwardOutput {
            logits,
            mtp_logits: None,
        })
    }

    fn reset_for_new_request(&self) {
        if let Some(llama) = self.llama_forward.borrow_mut().as_mut() {
            llama.reset();
        }
    }

    // -- libllama FFI hooks: same delegation pattern as Qwen35Model -------

    fn llama_forward_active(&self) -> bool {
        self.llama_forward.borrow().is_some()
    }

    fn llama_forward_mut(
        &self,
    ) -> Option<std::cell::RefMut<'_, Option<LlamaForward>>> {
        Some(self.llama_forward.borrow_mut())
    }

    fn llama_set_logit_bias(&self, token_id: u32, bias: f32) {
        if let Some(llama) = self.llama_forward.borrow_mut().as_mut() {
            llama.set_logit_bias(token_id, bias);
        }
    }

    fn llama_set_engram_bias(&self, predictions: &[(u32, f32)]) {
        if let Some(llama) = self.llama_forward.borrow().as_ref() {
            llama.set_engram_bias(predictions);
        }
    }

    fn llama_clear_engram_bias(&self) {
        if let Some(llama) = self.llama_forward.borrow().as_ref() {
            llama.clear_engram_bias();
        }
    }

    fn take_last_packed_logprobs(&self) -> Option<Vec<f32>> {
        self.last_packed_logprobs.borrow_mut().take()
    }
}
