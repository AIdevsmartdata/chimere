//! # Chimere multi-architecture trait
//!
//! This module defines the `ChimereModel` trait, the model-agnostic surface that
//! `chimere-server`'s generation, sampling and HTTP layers will call into so the
//! same engine can host Qwen3.5-35B-A3B, Mamba-1, Mamba-2, NemotronH-MoE, and any
//! future architecture loadable through libllama.
//!
//! ## Design constraints (from the multi-arch refactor plan)
//!
//! - **Zero regression on Qwen3.5.** All Qwen-specific machinery (MTP, MRoPE,
//!   block diffusion, Engram-aware sampling, entropy routing, custom CUDA graphs)
//!   stays exactly where it is in `qwen35_model::Qwen35Model`. The trait only
//!   exposes the methods the *outside* of `qwen35_model/` actually needs.
//!
//! - **Token-level granularity.** The trait exposes `forward_token` /
//!   `forward_prefill`, *not* a `generate()` method. Reason: `mtp_scheduler`'s
//!   ~500-line streaming loop interleaves forward + engram bias + MTP snapshot +
//!   DART speculation. Hoisting `generate()` onto the trait would force
//!   duplicating that loop for every architecture.
//!
//! - **Capabilities degrade gracefully.** Architectures without MTP, block
//!   diffusion, DART, or entropy routing return `false` from the corresponding
//!   `supports_*()` method. Callers MUST check before invoking arch-specific
//!   code paths.
//!
//! ## Status
//!
//! This is **Step 1** of the 7-step migration. At this step:
//!
//! - The trait exists and has a working `impl ChimereModel for Qwen35Model` that
//!   delegates to the existing inherent methods (zero behavior change).
//! - **No external caller uses the trait yet.** `generate.rs`, `mtp_scheduler.rs`,
//!   `block_generate.rs`, `server.rs` and `bin/chimere-server.rs` are unchanged
//!   and still take `&Qwen35Model` directly. They migrate in Steps 2-5.
//! - The `InferenceState::Generic` variant is intentionally absent — it lands in
//!   Step 7 alongside `GenericModel` for the Mamba/Nemotron path. The enum is
//!   marked `#[non_exhaustive]` so adding the variant later is non-breaking.

use candle_core::{Result, Tensor};

use crate::llama_backend::LlamaForward;
use crate::state::GdnRecurrentState;

// ---------------------------------------------------------------------------
// Model identity
// ---------------------------------------------------------------------------

/// Identifier for the loaded model architecture.
///
/// Used by callers to gate arch-specific code paths and by logs/metrics. The
/// variants below are the architectures we plan to support; new ones can be
/// added without breaking existing matches because the enum is
/// `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArch {
    /// Qwen3.5-35B-A3B (MoE 256 experts top-8, 48 GDN + 16 attention layers,
    /// MRoPE, optional MTP head). The Chimere production target.
    Qwen35A3B,
    /// Mamba-1 (state-spaces/mamba-*). Pure recurrent SSM, no attention.
    Mamba1,
    /// Mamba-2 (state-spaces/mamba2-*). Recurrent SSM with grouped state.
    Mamba2,
    /// NVIDIA Nemotron-3-Nano (`nemotron_h_moe`). Hybrid Mamba-2 + GQA + MoE.
    NemotronHMoe,
}

impl ModelArch {
    /// Short snake-case identifier (matches the GGUF `general.architecture`
    /// metadata field for the libllama-supported variants).
    pub fn name(&self) -> &'static str {
        match self {
            ModelArch::Qwen35A3B => "qwen3.5",
            ModelArch::Mamba1 => "mamba",
            ModelArch::Mamba2 => "mamba2",
            ModelArch::NemotronHMoe => "nemotron_h_moe",
        }
    }
}

// ---------------------------------------------------------------------------
// Forward output
// ---------------------------------------------------------------------------

/// The result of a single forward pass.
///
/// `mtp_logits` is `Some` only when the model has an MTP head AND the call site
/// requested MTP-aware decoding (Qwen3.5 with the speculative MTP scheduler).
/// Generic models always return `None`.
#[derive(Debug)]
pub struct ForwardOutput {
    /// Final logits, shape `[1, vocab_size]`.
    pub logits: Tensor,
    /// Optional MTP head logits, shape `[1, vocab_size]`. Only populated for
    /// architectures that override `supports_mtp()` to return `true`.
    pub mtp_logits: Option<Tensor>,
}

// ---------------------------------------------------------------------------
// Inference state
// ---------------------------------------------------------------------------

/// Per-request inference state, model-agnostic at the call-site level.
///
/// This is a **borrowing** enum: the variants hold mutable references to the
/// concrete state owned elsewhere (typically by the HTTP handler in
/// `server.rs::run_inference`). That choice is deliberate: it lets functions
/// like `mtp_scheduler::generate_with_mtp` keep their existing
/// `state: &mut GdnRecurrentState` parameter type unchanged, and only build a
/// short-lived `InferenceState::Gdn(state)` wrapper for the trait call. No
/// move, no `Box`, no ownership reshuffle through the call chain.
///
/// At Step 1-2 of the migration, only the `Gdn` variant exists (used by
/// Qwen3.5-35B-A3B). The `Generic` variant for libllama-only models lands in
/// Step 7. The enum is `#[non_exhaustive]` so adding it is non-breaking.
#[non_exhaustive]
pub enum InferenceState<'a> {
    /// Recurrent state for Qwen3.5's hybrid GDN + attention layers (DeltaNet
    /// matrix state, conv1d sliding window, KV cache, and assorted scratch
    /// buffers). See `state::GdnRecurrentState`.
    Gdn(&'a mut GdnRecurrentState),
    /// Opaque marker for architectures whose state lives ENTIRELY inside the
    /// libllama FFI context (Mamba-1, Mamba-2, Nemotron-H MoE, ...). The
    /// lifetime is carried solely to keep the enum generic over `'a` — there
    /// is no borrowed field. Callers pass `InferenceState::Generic` whenever
    /// they call a `ChimereModel::forward_*` method on a `GenericModel`.
    Generic(std::marker::PhantomData<&'a ()>),
}

impl<'a> InferenceState<'a> {
    /// Borrow the inner `GdnRecurrentState`. Errors if `self` is a different
    /// variant. Used by `Qwen35Model`'s `ChimereModel` impl.
    pub fn as_gdn_mut(&mut self) -> Result<&mut GdnRecurrentState> {
        match self {
            InferenceState::Gdn(s) => Ok(*s),
            InferenceState::Generic(_) => Err(candle_core::Error::Msg(
                "InferenceState::Generic has no GdnRecurrentState — caller \
                 misrouted a libllama-backed model into a Qwen-only code \
                 path"
                    .into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Model-agnostic surface used by `chimere-server`'s generation, sampling and
/// HTTP layers.
///
/// Most methods have sensible defaults so a new architecture only needs to
/// implement `arch`, `num_layers`, `vocab_size`, `forward_token` and
/// `forward_prefill`. Capability flags default to `false` — override only the
/// features your architecture actually supports.
///
/// ## Thread safety
///
/// Implementors typically hold raw pointers (libllama handles, cudarc buffers)
/// and are `!Send` / `!Sync` by default. The server wraps the trait object in a
/// `tokio::sync::Mutex` and adds an `unsafe impl Send` per concrete type, the
/// same pattern used today by `Qwen35Model`.
pub trait ChimereModel {
    // -- Identity ----------------------------------------------------------

    /// Architecture identifier of the loaded model.
    fn arch(&self) -> ModelArch;

    /// Number of main transformer/SSM layers (does not count MTP heads).
    fn num_layers(&self) -> usize;

    /// Vocabulary size, used to size logit tensors and validate token IDs.
    fn vocab_size(&self) -> usize;

    // -- Capability flags --------------------------------------------------
    //
    // All default to `false`. Implementors override the ones their arch
    // actually supports. Callers MUST gate arch-specific code paths on these.

    /// Multi-token prediction head present (currently Qwen3.5-35B-A3B only).
    fn supports_mtp(&self) -> bool {
        false
    }

    /// Block diffusion / multi-draft decoding supported by the model's
    /// internal forward path (currently Qwen3.5 only).
    fn supports_block_diffusion(&self) -> bool {
        false
    }

    /// DART speculative draft verification supported (depends on MTP +
    /// libllama backend; currently Qwen3.5 only).
    fn supports_dart(&self) -> bool {
        false
    }

    /// Entropy-adaptive routing between attention paths supported.
    fn supports_entropy_routing(&self) -> bool {
        false
    }

    // -- Forward pass ------------------------------------------------------

    /// Forward pass for a single token. Updates `state` in place.
    fn forward_token(
        &self,
        token: u32,
        state: &mut InferenceState<'_>,
    ) -> Result<ForwardOutput>;

    /// Batch prefill: process the entire prompt in a single call. Returns the
    /// logits for the **last** prompt token only. Updates `state` in place.
    fn forward_prefill(
        &self,
        tokens: &[u32],
        state: &mut InferenceState<'_>,
    ) -> Result<ForwardOutput>;

    // -- Lifecycle ---------------------------------------------------------

    /// Reset all per-request state (KV cache, recurrent states, conv windows,
    /// position counters) so the next request starts fresh. Default no-op for
    /// architectures whose state lives entirely in `InferenceState`.
    fn reset_for_new_request(&self) {}

    // -- libllama FFI hooks ------------------------------------------------
    //
    // These all default to a no-op / `None` so that architectures that do not
    // hold a libllama backend (synthetic test models, future pure-Rust
    // implementations) trivially satisfy the trait. Implementations that DO
    // wrap libllama (`Qwen35Model`, `GenericModel`) override every method.

    /// Whether this model is currently routing its forward pass through a
    /// `LlamaForward` FFI handle. Sampling pipelines use this to choose
    /// between the C++ fast sampler and the Rust slow path.
    fn llama_forward_active(&self) -> bool {
        false
    }

    /// Borrow the underlying `LlamaForward` for direct manipulation (used by
    /// the agent scheduler for context switching, and by `dart_verify_drafts`
    /// for batch token verification). Returns `None` for models without a
    /// libllama backend.
    fn llama_forward_mut(
        &self,
    ) -> Option<std::cell::RefMut<'_, Option<LlamaForward>>> {
        None
    }

    /// Immutable view of the underlying `LlamaForward` (read-only access for
    /// tokenization, n_vocab queries, debug introspection). Returns `None`
    /// for models without a libllama backend. Default impl returns `None`.
    fn llama_forward(
        &self,
    ) -> Option<std::cell::Ref<'_, Option<LlamaForward>>> {
        None
    }

    /// Push an additive logit bias for a single token to the C++ sampler. The
    /// canonical use case is `</think>` suppression while the model is still
    /// inside its reasoning block. No-op when no libllama backend is active.
    fn llama_set_logit_bias(&self, _token_id: u32, _bias: f32) {}

    /// Push a batch of n-gram Engram predictions as additive biases. No-op
    /// when no libllama backend is active.
    fn llama_set_engram_bias(&self, _predictions: &[(u32, f32)]) {}

    /// Clear the Engram bias overlay (keeps any manual `llama_set_logit_bias`
    /// values intact). No-op when no libllama backend is active.
    fn llama_clear_engram_bias(&self) {}

    /// Drain the packed-logprobs slot left behind by the last `forward_token`
    /// call when the C++ fast-sampler path was used. Returns `None` if the
    /// slow path was used (or if no libllama backend is active).
    /// Format: `[token_id, n_top, t0, lp0, t1, lp1, ..., t4, lp4]`.
    fn take_last_packed_logprobs(&self) -> Option<Vec<f32>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers for GDN-state callers
// ---------------------------------------------------------------------------
//
// Most production callers (`generate.rs`, `mtp_scheduler.rs`) currently hold a
// `&mut GdnRecurrentState` directly and would otherwise need to wrap-and-unwrap
// it into an `InferenceState::Gdn(...)` for every trait method call. These two
// helpers do the wrapping in one place so the call sites stay readable.
//
// They are intentionally free functions, not trait methods: keeping
// `GdnRecurrentState` out of the trait surface is what makes the trait
// model-agnostic. Code paths that work with `Generic` state will use the trait
// methods directly with their own `InferenceState::Generic` wrapper (Step 7).

/// Wrap `state` in `InferenceState::Gdn` and call `model.forward_token`. The
/// borrow on `state` is released as soon as the call returns.
#[inline]
pub fn forward_token_gdn(
    model: &dyn ChimereModel,
    token: u32,
    state: &mut GdnRecurrentState,
) -> Result<ForwardOutput> {
    let mut inf = InferenceState::Gdn(state);
    model.forward_token(token, &mut inf)
}

/// Wrap `state` in `InferenceState::Gdn` and call `model.forward_prefill`. The
/// borrow on `state` is released as soon as the call returns.
#[inline]
pub fn forward_prefill_gdn(
    model: &dyn ChimereModel,
    tokens: &[u32],
    state: &mut GdnRecurrentState,
) -> Result<ForwardOutput> {
    let mut inf = InferenceState::Gdn(state);
    model.forward_prefill(tokens, &mut inf)
}

// ---------------------------------------------------------------------------
// Convenience helpers for libllama-only callers (Step 7)
// ---------------------------------------------------------------------------
//
// These mirror the `*_gdn` helpers above but build a phantom
// `InferenceState::Generic` instead. Used by the Generic generation path
// (`generate.rs::generate_text_generic`,
// `mtp_scheduler.rs::generate_with_mtp_generic`) so the call sites do not
// have to know about `std::marker::PhantomData`.

/// Wrap a phantom `InferenceState::Generic` and call `model.forward_token`.
#[inline]
pub fn forward_token_generic(
    model: &dyn ChimereModel,
    token: u32,
) -> Result<ForwardOutput> {
    let mut inf = InferenceState::Generic(std::marker::PhantomData);
    model.forward_token(token, &mut inf)
}

/// Wrap a phantom `InferenceState::Generic` and call `model.forward_prefill`.
#[inline]
pub fn forward_prefill_generic(
    model: &dyn ChimereModel,
    tokens: &[u32],
) -> Result<ForwardOutput> {
    let mut inf = InferenceState::Generic(std::marker::PhantomData);
    model.forward_prefill(tokens, &mut inf)
}
