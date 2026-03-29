//! CUDA Graph capture and replay for GDN and Attention layer forward passes.
//!
//! Each GDN layer's kernel sequence is identical every token (fixed dimensions,
//! pre-allocated buffers with stable GPU addresses, in-place state updates).
//! This module captures the entire kernel sequence as a CUDA Graph on the
//! second token, then replays it on all subsequent tokens — eliminating
//! ~390 kernel launch overheads (30 GDN layers x ~13 kernels each).
//!
//! For Attention layers, only the post-attention part (ffn_norm RMSNorm + MoE)
//! is captured. The attention sub-layer itself stays inline because the KV
//! cache length varies each token.
//!
//! ## Design
//!
//! - Token 0+1: warmup (ensures ggml_cuda_init + all lazy CUDA allocations
//!   complete — cudaMalloc is NOT allowed during stream capture).
//! - Token 2: capture each GDN layer's forward pass into a CUDA Graph.
//! - Token 3+: replay the cached CUDA Graph for each GDN layer.
//!
//! ## Stream unification
//!
//! GDN forward mixes:
//! 1. ggml FFI calls (chimere_mmvq_q5k, chimere_quantize_q8_1) — normally
//!    use the CUDA default stream (NULL).
//! 2. cudarc kernel launches — use cudarc's CudaStream.
//!
//! CUDA Graph capture only records operations on the captured stream. To
//! capture both, we set the ggml FFI stream override to the cudarc stream
//! during capture and replay.
//!
//! ## Raw CUDA driver API
//!
//! We bypass cudarc's `begin_capture`/`end_capture` wrappers because cudarc's
//! `bind_to_thread()` calls `check_err()` which can poison the capture path
//! with stale errors from `SyncOnDrop::drop` event recordings. Instead we
//! call `cuStreamBeginCapture_v2`, `cuStreamEndCapture`, `cuGraphInstantiate`,
//! and `cuGraphLaunch` directly via the `sys` bindings.
//!
//! ## Toggle
//!
//! `CHIMERE_CUDA_GRAPH=1` to enable CUDA Graph capture for GDN layers
//! and Attention-layer MoE subgraphs. Default: disabled (for safety
//! during development).

use std::sync::{Arc, OnceLock};

use candle_core::cuda_backend::cudarc::driver::sys;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::cuda_backend::CudaDevice;
use candle_core::Result;

/// Check if CUDA Graph capture is enabled via environment variable.
pub fn cuda_graph_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CHIMERE_CUDA_GRAPH").is_ok())
}

/// CUDA Graph cache for GDN layers and Attention-layer MoE subgraphs.
///
/// Stores one captured CUDA Graph per GDN layer (full layer: RMSNorm+GDN+
/// RMSNorm+MoE) and one per Attention layer's post-attention part
/// (ffn_norm RMSNorm + MoE). The attention sub-layer itself stays inline
/// because the KV cache length varies each token.
pub(crate) struct GdnGraphCache {
    /// One captured CUDA Graph per GDN layer. `None` until captured.
    /// Indexed by GDN layer index (0..num_gdn_layers).
    pub graphs: Vec<Option<RawCudaGraph>>,

    /// One captured CUDA Graph per Attention layer's post-attention MoE.
    /// `None` until captured. Indexed by attention layer index (0..num_attn).
    pub attn_moe_graphs: Vec<Option<RawCudaGraph>>,

    /// Number of tokens processed. Used to decide when to capture.
    /// - token_count 0+1: warmup, run normally
    /// - token_count == 2: capture mode
    /// - token_count >= 3: replay mode
    pub token_count: usize,

    /// Raw CUstream handle for capture and replay.
    /// This is the SAME stream that cudarc kernels launch on.
    raw_stream: sys::CUstream,

    /// Keep the Arc alive so the stream isn't dropped.
    _stream_ref: Arc<CudaStream>,
}

/// Raw CUDA Graph handle — bypasses cudarc's wrapper to avoid Send/Sync issues
/// and check_err poisoning.
pub(crate) struct RawCudaGraph {
    graph: sys::CUgraph,
    graph_exec: sys::CUgraphExec,
    stream: sys::CUstream,
}

// SAFETY: CUDA graphs are thread-safe once instantiated — launch() is a
// GPU-side operation with no mutable CPU state. The graph is only ever
// accessed from behind a &mut ComputeGraph (single owner).
unsafe impl Send for RawCudaGraph {}
unsafe impl Sync for RawCudaGraph {}
unsafe impl Send for GdnGraphCache {}
unsafe impl Sync for GdnGraphCache {}

impl Drop for RawCudaGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.graph_exec.is_null() {
                sys::cuGraphExecDestroy(self.graph_exec);
            }
            if !self.graph.is_null() {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

impl RawCudaGraph {
    /// Replay the captured graph on its original stream.
    pub fn launch(&self) -> Result<()> {
        let err = unsafe { sys::cuGraphLaunch(self.graph_exec, self.stream) };
        if err != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(candle_core::Error::Msg(
                format!("cuGraphLaunch failed: {:?}", err),
            ));
        }
        Ok(())
    }
}

impl GdnGraphCache {
    /// Create a new cache for `num_gdn` GDN layers and `num_attn` Attention layers.
    ///
    /// Uses the cudarc device's own stream for capture. This is the same
    /// stream that all cudarc kernels (fused_ops, deltanet_step, etc.)
    /// launch on, so they are automatically captured.
    pub fn new(num_gdn: usize, num_attn: usize, dev: &CudaDevice) -> Result<Self> {
        let stream_arc = dev.cuda_stream();
        let raw_stream = stream_arc.cu_stream();

        let mut graphs = Vec::with_capacity(num_gdn);
        for _ in 0..num_gdn {
            graphs.push(None);
        }

        let mut attn_moe_graphs = Vec::with_capacity(num_attn);
        for _ in 0..num_attn {
            attn_moe_graphs.push(None);
        }

        eprintln!(
            "[CUDA_GRAPH] Initialized graph cache for {} GDN layers + {} attn MoE graphs (capture on token 2)",
            num_gdn, num_attn,
        );

        Ok(Self {
            graphs,
            attn_moe_graphs,
            token_count: 0,
            raw_stream,
            _stream_ref: stream_arc,
        })
    }

    /// Get the raw CUstream pointer for the capture stream.
    ///
    /// This is passed to ggml FFI calls during capture so they run on
    /// the same stream as the cudarc kernels.
    pub fn raw_stream_ptr(&self) -> *mut std::ffi::c_void {
        self.raw_stream as *mut std::ffi::c_void
    }

    /// Check if we're in capture mode (token_count == 2).
    ///
    /// Token 0+1 are warmup (ensures ggml_cuda_init and all lazy CUDA
    /// allocations complete before capture — cudaMalloc is NOT allowed
    /// during stream capture).
    #[inline]
    pub fn is_capture_token(&self) -> bool {
        self.token_count == 2
    }

    /// Check if we're in replay mode (token_count >= 3 and graphs are cached).
    #[inline]
    pub fn is_replay_mode(&self) -> bool {
        self.token_count >= 3
    }

    /// Begin capturing a GDN layer's kernel sequence.
    ///
    /// Uses raw CUDA driver API to bypass cudarc's `check_err` poisoning.
    pub fn begin_capture(&self) -> Result<()> {
        // Clear any accumulated cudarc error state before capture
        let ctx = self._stream_ref.context();
        let _ = ctx.check_err();

        let err = unsafe {
            sys::cuStreamBeginCapture_v2(
                self.raw_stream,
                sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if err != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(candle_core::Error::Msg(
                format!("cuStreamBeginCapture failed: {:?}", err),
            ));
        }
        Ok(())
    }

    /// End capturing and store the graph for `gdn_idx`.
    ///
    /// **Always exits capture mode**, even on error. This prevents the stream
    /// from being stuck in capture mode (which would cause all subsequent
    /// begin_capture calls to fail with CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED).
    ///
    /// Uses raw CUDA driver API to bypass cudarc's `check_err` poisoning.
    pub fn end_capture_safe(&mut self, gdn_idx: usize) -> Result<()> {
        // Clear any cudarc error state accumulated during capture
        let ctx = self._stream_ref.context();
        let _ = ctx.check_err();

        // End capture — this ALWAYS transitions the stream out of capture mode
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let end_err = unsafe {
            sys::cuStreamEndCapture(self.raw_stream, &mut cu_graph)
        };

        if end_err != sys::cudaError_enum::CUDA_SUCCESS {
            // Stream is out of capture mode even on error, but no graph was produced
            if !cu_graph.is_null() {
                unsafe { sys::cuGraphDestroy(cu_graph); }
            }
            return Err(candle_core::Error::Msg(
                format!("cuStreamEndCapture failed for GDN layer {}: {:?}", gdn_idx, end_err),
            ));
        }

        if cu_graph.is_null() {
            eprintln!(
                "[CUDA_GRAPH] WARNING: end_capture returned null graph for GDN layer {} (empty?)",
                gdn_idx,
            );
            return Ok(());
        }

        // Instantiate the graph
        let mut graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        let inst_err = unsafe {
            sys::cuGraphInstantiateWithFlags(
                &mut graph_exec,
                cu_graph,
                0, // no flags needed
            )
        };

        if inst_err != sys::cudaError_enum::CUDA_SUCCESS {
            unsafe { sys::cuGraphDestroy(cu_graph); }
            return Err(candle_core::Error::Msg(
                format!("cuGraphInstantiate failed for GDN layer {}: {:?}", gdn_idx, inst_err),
            ));
        }

        eprintln!("[CUDA_GRAPH] Captured GDN layer {} graph", gdn_idx);
        self.graphs[gdn_idx] = Some(RawCudaGraph {
            graph: cu_graph,
            graph_exec,
            stream: self.raw_stream,
        });
        Ok(())
    }

    /// Replay the cached graph for `gdn_idx`.
    ///
    /// Returns `Ok(true)` if the graph was replayed, `Ok(false)` if no
    /// graph is cached for this layer.
    pub fn replay(&self, gdn_idx: usize) -> Result<bool> {
        if let Some(ref graph) = self.graphs[gdn_idx] {
            graph.launch()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// End capturing and store the graph for attention layer `attn_idx`'s MoE.
    ///
    /// **Always exits capture mode**, even on error. Same safety guarantee
    /// as `end_capture_safe`.
    pub fn end_capture_attn_moe_safe(&mut self, attn_idx: usize) -> Result<()> {
        // Clear any cudarc error state accumulated during capture
        let ctx = self._stream_ref.context();
        let _ = ctx.check_err();

        // End capture — this ALWAYS transitions the stream out of capture mode
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let end_err = unsafe {
            sys::cuStreamEndCapture(self.raw_stream, &mut cu_graph)
        };

        if end_err != sys::cudaError_enum::CUDA_SUCCESS {
            if !cu_graph.is_null() {
                unsafe { sys::cuGraphDestroy(cu_graph); }
            }
            return Err(candle_core::Error::Msg(
                format!("cuStreamEndCapture failed for attn MoE layer {}: {:?}", attn_idx, end_err),
            ));
        }

        if cu_graph.is_null() {
            eprintln!(
                "[CUDA_GRAPH] WARNING: end_capture returned null graph for attn MoE layer {} (empty?)",
                attn_idx,
            );
            return Ok(());
        }

        // Instantiate the graph
        let mut graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        let inst_err = unsafe {
            sys::cuGraphInstantiateWithFlags(
                &mut graph_exec,
                cu_graph,
                0,
            )
        };

        if inst_err != sys::cudaError_enum::CUDA_SUCCESS {
            unsafe { sys::cuGraphDestroy(cu_graph); }
            return Err(candle_core::Error::Msg(
                format!("cuGraphInstantiate failed for attn MoE layer {}: {:?}", attn_idx, inst_err),
            ));
        }

        eprintln!("[CUDA_GRAPH] Captured attn MoE layer {} graph", attn_idx);
        self.attn_moe_graphs[attn_idx] = Some(RawCudaGraph {
            graph: cu_graph,
            graph_exec,
            stream: self.raw_stream,
        });
        Ok(())
    }

    /// Replay the cached graph for attention layer `attn_idx`'s MoE.
    ///
    /// Returns `Ok(true)` if the graph was replayed, `Ok(false)` if no
    /// graph is cached for this layer.
    pub fn replay_attn_moe(&self, attn_idx: usize) -> Result<bool> {
        if let Some(ref graph) = self.attn_moe_graphs[attn_idx] {
            graph.launch()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Increment the token counter. Call once at the start of each token.
    pub fn advance_token(&mut self) {
        self.token_count += 1;
    }

    /// Reset for a new request (multi-turn). Drops all cached graphs
    /// and resets the token counter.
    pub fn reset(&mut self) {
        for g in self.graphs.iter_mut() {
            *g = None;
        }
        for g in self.attn_moe_graphs.iter_mut() {
            *g = None;
        }
        self.token_count = 0;
    }
}
