//! # GDN Recurrent State — Chimere Engine
//!
//! Manages the recurrent hidden state for the Qwen3.5 hybrid architecture.
//!
//! The state is split into three parts:
//! - **GDN states**: per-GDN-layer recurrent SSM state
//! - **Conv states**: per-GDN-layer causal convolution sliding window
//! - **KV caches**: per-attention-layer key/value caches
//!
//! The `snapshot` / `restore` methods enable MTP (multi-token prediction)
//! branching: take a snapshot before speculative decoding, then restore
//! if the branch is rejected.
//!
//! State tensors live on `compute_device` (typically GPU) during inference.
//! Use `snapshot_to_cpu` / `restore_from_cpu` for cross-device MTP branching.
//!
//! ## KV Ring Buffer (`CHIMERE_KV_RING=1`)
//!
//! When enabled, the KV cache uses pre-allocated buffers with `slice_set`
//! for O(1) appends instead of `Tensor::cat` which copies the entire cache
//! (O(P) per token at position P).  Reads use `narrow()` zero-copy views.
//! The buffer starts at `KV_RING_INITIAL_CAP` positions and doubles when full.

use candle_core::{DType, Device, Result, Tensor};
use candle_core::cuda_backend::cudarc::driver::CudaSlice;

use crate::config::Qwen35Config;

// ---------------------------------------------------------------------------
// NcmoeBufs — pre-allocated staging buffers for ncmoe batch copy
// ---------------------------------------------------------------------------

/// IQ3_S block constants.
const IQ3S_BLOCK_BYTES: usize = 110;
const IQ3S_BLOCK_ELEMS: usize = 256;

/// Pre-allocated buffers for the ncmoe (CPU-offloaded experts) batch copy path.
///
/// Instead of copying each expert one-by-one via Candle `to_device()` (24 copies
/// per token × 4 ncmoe layers = timeout), we:
/// 1. Assemble all 8 active experts' IQ3_S bytes into `cpu_staging` (memcpy)
/// 2. Single `memcpy_htod` from `cpu_staging` to `gpu_expert_buf`
/// 3. Run GEMV kernels at offsets within `gpu_expert_buf`
///
/// Total staging size: `top_k * expert_bytes_per_matrix * 3` (~10.3 MB for 8 experts).
/// Buffers are reused across tokens — zero allocation at runtime.
pub struct NcmoeBufs {
    /// GPU-side staging buffer for all 8 experts' gate+up+down IQ3_S bytes.
    /// Layout: [gate_0, gate_1, ..., gate_7, up_0, ..., up_7, down_0, ..., down_7]
    pub gpu_expert_buf: CudaSlice<u8>,
    /// CPU-side staging buffer (same size as gpu_expert_buf).
    pub cpu_staging: Vec<u8>,
    /// Per-expert byte count for gate projection.
    pub expert_bytes_gate: usize,
    /// Per-expert byte count for up projection.
    pub expert_bytes_up: usize,
    /// Per-expert byte count for down projection.
    pub expert_bytes_down: usize,
    /// Number of experts batched (top_k).
    pub top_k: usize,
    /// GPU scratch: gate GEMV output [expert_ffn] — reused per expert.
    pub gate_out: CudaSlice<f32>,
    /// GPU scratch: up GEMV output [expert_ffn] — reused per expert.
    pub up_out: CudaSlice<f32>,
    /// GPU scratch: down GEMV output [hidden_size] — reused per expert.
    pub down_out: CudaSlice<f32>,
    /// GPU scratch: accumulated weighted expert outputs [hidden_size] — reused per token.
    pub combined_out: CudaSlice<f32>,
}

impl NcmoeBufs {
    /// Allocate ncmoe staging buffers for the given model dimensions.
    ///
    /// `expert_ffn` = 512, `hidden_size` = 2048, `top_k` = 8 for Qwen3.5-35B-A3B.
    /// `gate_elements`, `up_elements`, `down_elements` are per-expert element counts.
    pub fn new(
        gate_elements_per_expert: usize,
        up_elements_per_expert: usize,
        down_elements_per_expert: usize,
        expert_ffn: usize,
        hidden_size: usize,
        top_k: usize,
        device: &Device,
    ) -> Result<Self> {
        let Device::Cuda(dev) = device else {
            candle_core::bail!("NcmoeBufs::new: device must be CUDA");
        };

        let expert_bytes_gate = (gate_elements_per_expert / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_up = (up_elements_per_expert / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;
        let expert_bytes_down = (down_elements_per_expert / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES;

        let total_bytes = top_k * (expert_bytes_gate + expert_bytes_up + expert_bytes_down);

        let gpu_expert_buf = dev
            .alloc_zeros::<u8>(total_bytes)
            .map_err(|e| candle_core::Error::Msg(format!("NcmoeBufs gpu_expert_buf alloc: {e}")))?;
        let cpu_staging = vec![0u8; total_bytes];

        let gate_out = dev
            .alloc_zeros::<f32>(expert_ffn)
            .map_err(|e| candle_core::Error::Msg(format!("NcmoeBufs gate_out alloc: {e}")))?;
        let up_out = dev
            .alloc_zeros::<f32>(expert_ffn)
            .map_err(|e| candle_core::Error::Msg(format!("NcmoeBufs up_out alloc: {e}")))?;
        let down_out = dev
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| candle_core::Error::Msg(format!("NcmoeBufs down_out alloc: {e}")))?;
        let combined_out = dev
            .alloc_zeros::<f32>(hidden_size)
            .map_err(|e| candle_core::Error::Msg(format!("NcmoeBufs combined_out alloc: {e}")))?;

        eprintln!(
            "[NCMOE] Allocated batch copy buffers: {} bytes GPU + {} bytes CPU ({:.2} MB each), \
             gate={} up={} down={} bytes/expert, top_k={}",
            total_bytes, total_bytes,
            total_bytes as f64 / (1024.0 * 1024.0),
            expert_bytes_gate, expert_bytes_up, expert_bytes_down, top_k,
        );

        Ok(Self {
            gpu_expert_buf,
            cpu_staging,
            expert_bytes_gate,
            expert_bytes_up,
            expert_bytes_down,
            top_k,
            gate_out,
            up_out,
            down_out,
            combined_out,
        })
    }
}

// ---------------------------------------------------------------------------
// KV Ring Buffer — pre-allocated KV cache with O(1) append
// ---------------------------------------------------------------------------

/// Initial capacity (number of sequence positions) for the ring buffer.
/// Kept small to avoid wasting VRAM on short sequences.
/// Memory per layer at cap=4096: 2 * 4096 * num_kv_heads * head_dim * 2 bytes (F16).
/// For 35B-A3B (num_kv_heads=2, head_dim=256): 2 * 4096 * 2 * 256 * 2 = 8 MB/layer.
/// With 10 attention layers: 80 MB total initial allocation.
const KV_RING_INITIAL_CAP: usize = 4096;

/// Maximum KV cache sequence length before truncation (legacy path only).
/// When exceeded, only the last `KV_MAX_SEQ_LEN` positions are kept.
const KV_MAX_SEQ_LEN: usize = 8192;

/// Check (once) whether the ring-buffer KV cache is enabled.
fn kv_ring_enabled() -> bool {
    use once_cell::sync::Lazy;
    static ENABLED: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_KV_RING").is_ok());
    *ENABLED
}

// ---------------------------------------------------------------------------
// KV Cache Quantization Type
// ---------------------------------------------------------------------------

/// KV cache storage type, selected by `CHIMERE_KV_TYPE` environment variable.
///
/// - `F16` (default): 2 bytes per element, full half-precision.
/// - `Q8_0`: 34 bytes per 32 elements (~1.0625 bytes/element), ~47% savings vs F16.
///   Data is quantized on write and dequantized on read. The quantized bytes are
///   stored on CPU in `Vec<u8>` buffers; dequantized F32 tensors are produced on
///   the compute device when attention needs them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KvCacheType {
    F16,
    Q8_0,
}

impl std::fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheType::F16 => write!(f, "f16"),
            KvCacheType::Q8_0 => write!(f, "q8_0"),
        }
    }
}

/// Check (once) the configured KV cache type from `CHIMERE_KV_TYPE` env var.
///
/// Returns `KvCacheType::F16` by default. Set `CHIMERE_KV_TYPE=q8_0` for Q8_0.
fn kv_cache_type() -> KvCacheType {
    use once_cell::sync::Lazy;
    static KV_TYPE: Lazy<KvCacheType> = Lazy::new(|| {
        match std::env::var("CHIMERE_KV_TYPE") {
            Ok(v) if v.eq_ignore_ascii_case("q8_0") || v.eq_ignore_ascii_case("q8") => {
                KvCacheType::Q8_0
            }
            _ => KvCacheType::F16,
        }
    });
    *KV_TYPE
}

// ---------------------------------------------------------------------------
// Q8_0 KV Ring Buffer — quantized KV cache with ~47% memory savings
// ---------------------------------------------------------------------------

/// Pre-allocated Q8_0-quantized KV cache for a single attention layer.
///
/// Stores K and V as Q8_0-encoded byte buffers. On append, incoming F32/F16
/// tensors are quantized to Q8_0 bytes. On read, the bytes are dequantized
/// back to F32 tensors on the compute device.
///
/// Memory layout per head per position: 128 elements (head_dim) = 4 Q8_0
/// blocks = 136 bytes (vs 256 bytes F16).
///
/// The buffer is pre-allocated at `capacity` positions and grows by doubling.
pub struct KvQ8Cache {
    /// Q8_0-encoded key bytes: `[num_kv_heads * capacity * bytes_per_head_pos]`
    /// Layout is head-major: all positions for head 0, then head 1, etc.
    k_bytes: Vec<u8>,
    /// Q8_0-encoded value bytes (same layout as `k_bytes`).
    v_bytes: Vec<u8>,
    /// Number of sequence positions currently stored.
    len: usize,
    /// Current buffer capacity (number of positions).
    capacity: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
    /// Head dimension (must be a multiple of 32 for Q8_0).
    head_dim: usize,
    /// Q8_0 bytes per head per position = (head_dim / 32) * 34.
    bytes_per_head_pos: usize,
    /// The device to produce dequantized tensors on.
    device: Device,
}

impl KvQ8Cache {
    /// Create a new Q8_0 KV cache with the given initial capacity.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        initial_cap: usize,
        device: &Device,
    ) -> Result<Self> {
        assert!(
            head_dim % 32 == 0,
            "KvQ8Cache: head_dim ({head_dim}) must be a multiple of 32 for Q8_0"
        );
        let bytes_per_head_pos = crate::kernels::kv_q8_0::q8_0_byte_size(head_dim);
        let total_bytes = num_kv_heads * initial_cap * bytes_per_head_pos;
        Ok(Self {
            k_bytes: vec![0u8; total_bytes],
            v_bytes: vec![0u8; total_bytes],
            len: 0,
            capacity: initial_cap,
            num_kv_heads,
            head_dim,
            bytes_per_head_pos,
            device: device.clone(),
        })
    }

    /// Byte offset for position `pos` of head `h`.
    fn offset(&self, h: usize, pos: usize) -> usize {
        (h * self.capacity + pos) * self.bytes_per_head_pos
    }

    /// Append `n` new key/value positions to the cache.
    ///
    /// `k_new` and `v_new` must have shape `[1, num_kv_heads, n, head_dim]`.
    /// They are quantized to Q8_0 and stored. Returns dequantized `(K, V)`
    /// tensors covering the full cached sequence on `self.device`.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = k_new.dim(2)?;

        // Grow if needed
        while self.len + n > self.capacity {
            self.grow()?;
        }

        // Pull new K/V to CPU F32 for quantization
        let k_cpu = k_new
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?;
        let v_cpu = v_new
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .contiguous()?;

        // Quantize each head's new positions
        for h in 0..self.num_kv_heads {
            for p in 0..n {
                // Extract [head_dim] slice for this head and position
                let k_slice = k_cpu.narrow(1, h, 1)?.narrow(2, p, 1)?.flatten_all()?;
                let v_slice = v_cpu.narrow(1, h, 1)?.narrow(2, p, 1)?.flatten_all()?;
                let k_f32 = k_slice.to_vec1::<f32>()?;
                let v_f32 = v_slice.to_vec1::<f32>()?;

                let k_q8 = crate::kernels::kv_q8_0::quantize_q8_0_cpu(&k_f32);
                let v_q8 = crate::kernels::kv_q8_0::quantize_q8_0_cpu(&v_f32);

                let off = self.offset(h, self.len + p);
                self.k_bytes[off..off + self.bytes_per_head_pos]
                    .copy_from_slice(&k_q8);
                self.v_bytes[off..off + self.bytes_per_head_pos]
                    .copy_from_slice(&v_q8);
            }
        }

        self.len += n;

        // Dequantize full cache and return as tensors on compute device
        self.dequantize_full()
    }

    /// Dequantize the entire cached K and V into tensors.
    ///
    /// Returns `(K, V)` with shape `[1, num_kv_heads, len, head_dim]` on `self.device`.
    fn dequantize_full(&self) -> Result<(Tensor, Tensor)> {
        if self.len == 0 {
            let k = Tensor::zeros((1, self.num_kv_heads, 0, self.head_dim), DType::F32, &self.device)?;
            let v = Tensor::zeros((1, self.num_kv_heads, 0, self.head_dim), DType::F32, &self.device)?;
            return Ok((k, v));
        }

        // Dequantize all heads and positions
        let total_elements = self.num_kv_heads * self.len * self.head_dim;
        let mut k_f32 = Vec::with_capacity(total_elements);
        let mut v_f32 = Vec::with_capacity(total_elements);

        for h in 0..self.num_kv_heads {
            for p in 0..self.len {
                let off = self.offset(h, p);
                let k_block = &self.k_bytes[off..off + self.bytes_per_head_pos];
                let v_block = &self.v_bytes[off..off + self.bytes_per_head_pos];
                k_f32.extend_from_slice(
                    &crate::kernels::kv_q8_0::dequantize_q8_0_cpu(k_block, self.head_dim),
                );
                v_f32.extend_from_slice(
                    &crate::kernels::kv_q8_0::dequantize_q8_0_cpu(v_block, self.head_dim),
                );
            }
        }

        let k_tensor = Tensor::from_vec(
            k_f32,
            (1, self.num_kv_heads, self.len, self.head_dim),
            &Device::Cpu,
        )?;
        let v_tensor = Tensor::from_vec(
            v_f32,
            (1, self.num_kv_heads, self.len, self.head_dim),
            &Device::Cpu,
        )?;

        if matches!(self.device, Device::Cpu) {
            Ok((k_tensor, v_tensor))
        } else {
            Ok((k_tensor.to_device(&self.device)?, v_tensor.to_device(&self.device)?))
        }
    }

    /// Double the buffer capacity.
    fn grow(&mut self) -> Result<()> {
        let new_cap = self.capacity * 2;
        let new_total = self.num_kv_heads * new_cap * self.bytes_per_head_pos;

        let mut new_k = vec![0u8; new_total];
        let mut new_v = vec![0u8; new_total];

        // Copy existing data (head-major layout: need to adjust offsets)
        for h in 0..self.num_kv_heads {
            let old_head_start = h * self.capacity * self.bytes_per_head_pos;
            let new_head_start = h * new_cap * self.bytes_per_head_pos;
            let copy_bytes = self.len * self.bytes_per_head_pos;
            new_k[new_head_start..new_head_start + copy_bytes]
                .copy_from_slice(&self.k_bytes[old_head_start..old_head_start + copy_bytes]);
            new_v[new_head_start..new_head_start + copy_bytes]
                .copy_from_slice(&self.v_bytes[old_head_start..old_head_start + copy_bytes]);
        }

        self.k_bytes = new_k;
        self.v_bytes = new_v;
        self.capacity = new_cap;
        Ok(())
    }

    /// Current number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Deep-clone (snapshot) the Q8_0 cache.
    pub fn snapshot(&self) -> Result<Self> {
        if self.len == 0 {
            return Self::new(self.num_kv_heads, self.head_dim, KV_RING_INITIAL_CAP, &self.device);
        }
        // Clone with exact capacity = len
        let total_bytes = self.num_kv_heads * self.len * self.bytes_per_head_pos;
        let mut k_snap = vec![0u8; total_bytes];
        let mut v_snap = vec![0u8; total_bytes];

        for h in 0..self.num_kv_heads {
            let old_start = h * self.capacity * self.bytes_per_head_pos;
            let new_start = h * self.len * self.bytes_per_head_pos;
            let copy_bytes = self.len * self.bytes_per_head_pos;
            k_snap[new_start..new_start + copy_bytes]
                .copy_from_slice(&self.k_bytes[old_start..old_start + copy_bytes]);
            v_snap[new_start..new_start + copy_bytes]
                .copy_from_slice(&self.v_bytes[old_start..old_start + copy_bytes]);
        }

        Ok(Self {
            k_bytes: k_snap,
            v_bytes: v_snap,
            len: self.len,
            capacity: self.len,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            bytes_per_head_pos: self.bytes_per_head_pos,
            device: self.device.clone(),
        })
    }

    /// Restore from another Q8_0 cache (snapshot).
    pub fn restore_from(&mut self, other: &KvQ8Cache) -> Result<()> {
        if other.len > self.capacity {
            // Reallocate
            let total = self.num_kv_heads * other.len * self.bytes_per_head_pos;
            self.k_bytes = vec![0u8; total];
            self.v_bytes = vec![0u8; total];
            self.capacity = other.len;
        }
        for h in 0..self.num_kv_heads {
            let src_start = h * other.capacity * self.bytes_per_head_pos;
            let dst_start = h * self.capacity * self.bytes_per_head_pos;
            let copy_bytes = other.len * self.bytes_per_head_pos;
            if copy_bytes > 0 {
                self.k_bytes[dst_start..dst_start + copy_bytes]
                    .copy_from_slice(&other.k_bytes[src_start..src_start + copy_bytes]);
                self.v_bytes[dst_start..dst_start + copy_bytes]
                    .copy_from_slice(&other.v_bytes[src_start..src_start + copy_bytes]);
            }
        }
        self.len = other.len;
        Ok(())
    }

    /// Move to a target device (Q8_0 bytes are on CPU; only changes dequant target).
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.len == 0 {
            return Self::new(self.num_kv_heads, self.head_dim, KV_RING_INITIAL_CAP, device);
        }
        let mut clone = self.snapshot()?;
        clone.device = device.clone();
        Ok(clone)
    }

    /// Reset to empty without reallocating.
    pub fn reset(&mut self) {
        self.len = 0;
    }
}

/// Pre-allocated KV cache for a single attention layer.
///
/// Instead of growing via `Tensor::cat` (which copies the entire cache every
/// token), this struct pre-allocates a buffer of `capacity` sequence positions
/// and uses `Tensor::slice_set` for O(1) writes.  Reads via `narrow()` return
/// zero-copy views into the occupied prefix.
///
/// When `len` reaches `capacity`, the buffer is doubled (one-time O(N) copy,
/// amortised O(1) per token).
pub struct KvRingCache {
    /// Pre-allocated key buffer: `[1, num_kv_heads, capacity, head_dim]`
    k_buf: Tensor,
    /// Pre-allocated value buffer: same shape as `k_buf`
    v_buf: Tensor,
    /// Number of sequence positions currently written (always <= capacity).
    len: usize,
    /// Current buffer capacity (number of positions along dim 2).
    capacity: usize,
}

impl KvRingCache {
    /// Create a new ring buffer with the given initial capacity.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        initial_cap: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let k_buf = Tensor::zeros((1, num_kv_heads, initial_cap, head_dim), dtype, device)?;
        let v_buf = Tensor::zeros((1, num_kv_heads, initial_cap, head_dim), dtype, device)?;
        Ok(Self {
            k_buf,
            v_buf,
            len: 0,
            capacity: initial_cap,
        })
    }

    /// Append `n` new key/value positions to the cache.
    ///
    /// `k_new` and `v_new` must have shape `[1, num_kv_heads, n, head_dim]`.
    /// Returns `(k_view, v_view)` — zero-copy `narrow()` views over the
    /// filled prefix `[1, num_kv_heads, len+n, head_dim]`.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let n = k_new.dim(2)?;
        // Grow if needed
        while self.len + n > self.capacity {
            self.grow()?;
        }
        // Write k_new into k_buf at offset self.len on dim 2
        // slice_set requires both tensors to be contiguous and on the same device.
        let k_src = k_new.contiguous()?;
        let v_src = v_new.contiguous()?;
        self.k_buf.slice_set(&k_src, 2, self.len)?;
        self.v_buf.slice_set(&v_src, 2, self.len)?;
        self.len += n;
        // Return views over the filled prefix
        Ok((
            self.k_buf.narrow(2, 0, self.len)?,
            self.v_buf.narrow(2, 0, self.len)?,
        ))
    }

    /// Double the buffer capacity, copying existing data to the new buffer.
    fn grow(&mut self) -> Result<()> {
        let new_cap = self.capacity * 2;
        let (_, num_kv_heads, _, head_dim) = self.k_buf.dims4()?;
        let device = self.k_buf.device().clone();
        let dtype = self.k_buf.dtype();
        let new_k = Tensor::zeros((1, num_kv_heads, new_cap, head_dim), dtype, &device)?;
        let new_v = Tensor::zeros((1, num_kv_heads, new_cap, head_dim), dtype, &device)?;
        // Copy existing data into the new buffer
        if self.len > 0 {
            let old_k = self.k_buf.narrow(2, 0, self.len)?.contiguous()?;
            let old_v = self.v_buf.narrow(2, 0, self.len)?.contiguous()?;
            new_k.slice_set(&old_k, 2, 0)?;
            new_v.slice_set(&old_v, 2, 0)?;
        }
        self.k_buf = new_k;
        self.v_buf = new_v;
        self.capacity = new_cap;
        Ok(())
    }

    /// Current number of cached sequence positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Deep-clone the ring buffer (copies only the filled prefix, not the
    /// full capacity, to save memory in snapshots).
    pub fn snapshot(&self) -> Result<Self> {
        if self.len == 0 {
            // Clone empty buffer at minimum capacity
            let (_, num_kv_heads, _, head_dim) = self.k_buf.dims4()?;
            return Self::new(
                num_kv_heads,
                head_dim,
                KV_RING_INITIAL_CAP,
                self.k_buf.dtype(),
                self.k_buf.device(),
            );
        }
        // Allocate a buffer of exactly self.len capacity and copy data
        let (_, num_kv_heads, _, head_dim) = self.k_buf.dims4()?;
        let device = self.k_buf.device().clone();
        let dtype = self.k_buf.dtype();
        let snap_k = Tensor::zeros((1, num_kv_heads, self.len, head_dim), dtype, &device)?;
        let snap_v = Tensor::zeros((1, num_kv_heads, self.len, head_dim), dtype, &device)?;
        let filled_k = self.k_buf.narrow(2, 0, self.len)?.contiguous()?;
        let filled_v = self.v_buf.narrow(2, 0, self.len)?.contiguous()?;
        snap_k.slice_set(&filled_k, 2, 0)?;
        snap_v.slice_set(&filled_v, 2, 0)?;
        Ok(Self {
            k_buf: snap_k,
            v_buf: snap_v,
            len: self.len,
            capacity: self.len,
        })
    }

    /// Restore from another ring buffer (snapshot), resizing if needed.
    pub fn restore_from(&mut self, other: &KvRingCache) -> Result<()> {
        // If source is larger than our capacity, reallocate
        if other.len > self.capacity {
            let (_, num_kv_heads, _, head_dim) = self.k_buf.dims4()?;
            let device = self.k_buf.device().clone();
            let dtype = self.k_buf.dtype();
            self.k_buf = Tensor::zeros((1, num_kv_heads, other.len, head_dim), dtype, &device)?;
            self.v_buf = Tensor::zeros((1, num_kv_heads, other.len, head_dim), dtype, &device)?;
            self.capacity = other.len;
        }
        // Copy data
        if other.len > 0 {
            let src_k = other.k_buf.narrow(2, 0, other.len)?.contiguous()?;
            let src_v = other.v_buf.narrow(2, 0, other.len)?.contiguous()?;
            self.k_buf.slice_set(&src_k, 2, 0)?;
            self.v_buf.slice_set(&src_v, 2, 0)?;
        }
        self.len = other.len;
        Ok(())
    }

    /// Move data to a target device, returning a new KvRingCache on that device.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.len == 0 {
            let (_, num_kv_heads, _, head_dim) = self.k_buf.dims4()?;
            return Self::new(
                num_kv_heads,
                head_dim,
                KV_RING_INITIAL_CAP,
                self.k_buf.dtype(),
                device,
            );
        }
        // Only transfer the filled prefix
        let filled_k = self.k_buf.narrow(2, 0, self.len)?.contiguous()?.to_device(device)?;
        let filled_v = self.v_buf.narrow(2, 0, self.len)?.contiguous()?.to_device(device)?;
        // Allocate on target device with exact capacity = len
        let (_, num_kv_heads, _, head_dim) = self.k_buf.dims4()?;
        let dtype = self.k_buf.dtype();
        let new_k = Tensor::zeros((1, num_kv_heads, self.len, head_dim), dtype, device)?;
        let new_v = Tensor::zeros((1, num_kv_heads, self.len, head_dim), dtype, device)?;
        new_k.slice_set(&filled_k, 2, 0)?;
        new_v.slice_set(&filled_v, 2, 0)?;
        Ok(Self {
            k_buf: new_k,
            v_buf: new_v,
            len: self.len,
            capacity: self.len,
        })
    }

    /// Reset to empty (len=0) without reallocating the buffer.
    pub fn reset(&mut self) {
        self.len = 0;
        // Buffer contents become stale but are overwritten on next append.
    }
}

/// Complete inference state for a Qwen3.5 model.
///
/// Holds the per-layer recurrent states (GDN), convolution buffers, and
/// KV caches (attention layers).  All tensors live on `compute_device`
/// during inference (typically GPU), eliminating per-token PCIe transfers.
pub struct GdnRecurrentState {
    /// Per-GDN-layer recurrent state (DeltaNet matrix state).
    ///
    /// Shape per tensor: `[batch, ssm_dt_rank, ssm_d_state, ssm_d_state]`
    /// (e.g. `[1, 48, 128, 128]` for Qwen3.5-27B)
    ///
    /// Each head maintains a `[d_state, d_state]` matrix that is updated via
    /// the DeltaNet rule: `S = decay * S + outer(d, k)` where `d = beta * (v - S @ k)`.
    pub gdn_states: Vec<Tensor>,

    /// Conv1d sliding-window state per GDN layer.
    ///
    /// Shape per tensor: `[batch, conv_channels, conv_kernel - 1]`
    /// where `conv_channels = ssm_d_inner + 2 * ssm_n_group * ssm_d_state`
    /// (e.g. `[1, 10240, 3]` for Qwen3.5-27B)
    pub conv_states: Vec<Tensor>,

    /// KV cache per attention layer: `(K, V)`.
    ///
    /// Shape per tensor: `[batch, num_kv_heads, seq_len, head_dim]`
    /// where `seq_len` grows during generation (starts at 0).
    ///
    /// **Legacy path** (default): grows via `Tensor::cat`.
    /// When `CHIMERE_KV_RING=1`, this field is unused and `kv_ring_caches` is
    /// used instead.
    pub kv_caches: Vec<(Tensor, Tensor)>,

    /// Pre-allocated ring-buffer KV caches (one per attention layer).
    ///
    /// Enabled by `CHIMERE_KV_RING=1`.  Uses `slice_set` for O(1) appends
    /// and `narrow()` for zero-copy reads.  When disabled, this is `None`.
    pub kv_ring_caches: Option<Vec<KvRingCache>>,

    /// Q8_0-quantized KV caches (one per attention layer).
    ///
    /// Enabled by `CHIMERE_KV_TYPE=q8_0`.  Stores K/V as Q8_0 encoded bytes
    /// (~47% memory savings vs F16).  When enabled, takes precedence over both
    /// `kv_caches` (legacy) and `kv_ring_caches` (ring buffer).
    /// When disabled, this is `None`.
    pub kv_q8_caches: Option<Vec<KvQ8Cache>>,

    /// Current token position in the sequence.
    pub position: usize,

    /// The device where state tensors live during inference (typically GPU).
    pub compute_device: Device,

    /// Pre-allocated GPU buffers for the raw MoE FFN path (bypass Candle).
    /// Lazily initialized on first use. Stateless scratch — not snapshotted.
    pub raw_moe_buffers: Option<crate::raw_forward::RawGpuBuffers>,

    /// Pre-allocated buffers for the full raw forward path (`CHIMERE_RAW_FORWARD=1`).
    /// Lazily initialized on first use. Stateless scratch — not snapshotted.
    pub raw_forward_bufs: Option<crate::raw_forward::RawForwardBufs>,

    /// Pre-allocated staging buffers for ncmoe batch copy (CPU-offloaded experts).
    /// Lazily initialized on first use when `experts_on_cpu` is true.
    /// Stateless scratch — not snapshotted.
    pub ncmoe_bufs: Option<NcmoeBufs>,

    /// Pre-allocated scratch buffers for MoE shared expert path.
    /// Eliminates ~9 cudaMalloc per layer × 39 layers = 351 allocs/token.
    /// Lazily initialized on first use when CHIMERE_SCRATCH_POOL=1.
    /// Stateless scratch — not snapshotted.
    pub scratch_pool: Option<crate::scratch_pool::ScratchPool>,

    /// Pre-allocated scratch buffers for the pure cudarc MoE FFN path.
    /// Eliminates ALL Candle Tensor allocations in the MoE FFN.
    /// Lazily initialized on first use when CHIMERE_CUDARC_MOE=1.
    /// Stateless scratch — not snapshotted.
    pub moe_cudarc_bufs: Option<crate::qwen35_model::moe_cudarc::MoeCudarcBufs>,
}

impl GdnRecurrentState {
    /// Initialise all state tensors to zero for a batch of size 1.
    ///
    /// GDN and conv states are pre-allocated at full size; KV caches
    /// start with `seq_len = 0` (they grow during generation).
    ///
    /// The `device` parameter becomes the compute device — pass a GPU device
    /// to keep all state GPU-resident and avoid per-token PCIe transfers.
    pub fn new(config: &Qwen35Config, device: &Device) -> Result<Self> {
        let batch = 1;
        let num_gdn = config.num_gdn_layers();
        let num_attn = config.num_attn_layers();

        let mut gdn_states = Vec::with_capacity(num_gdn);
        for _ in 0..num_gdn {
            // DeltaNet state: [batch, dt_rank, d_state, d_state]
            // Each of the dt_rank=48 heads has a d_state x d_state = 128x128 matrix
            gdn_states.push(Tensor::zeros(
                (batch, config.ssm_dt_rank, config.ssm_d_state, config.ssm_d_state),
                DType::F32,
                device,
            )?);
        }

        let mut conv_states = Vec::with_capacity(num_gdn);
        let conv_buf_len = if config.ssm_conv_kernel > 0 {
            config.ssm_conv_kernel - 1
        } else {
            0
        };
        // Conv channels = QKV total = d_inner + 2 * n_group * d_state
        let conv_channels = config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state;
        for _ in 0..num_gdn {
            conv_states.push(Tensor::zeros(
                (batch, conv_channels, conv_buf_len),
                DType::F32,
                device,
            )?);
        }

        let use_ring = kv_ring_enabled();
        let kv_type = kv_cache_type();

        let mut kv_caches = Vec::with_capacity(num_attn);
        for _ in 0..num_attn {
            let k = Tensor::zeros(
                (batch, config.num_kv_heads, 0, config.head_dim),
                DType::F16,
                device,
            )?;
            let v = Tensor::zeros(
                (batch, config.num_kv_heads, 0, config.head_dim),
                DType::F16,
                device,
            )?;
            kv_caches.push((k, v));
        }

        // Q8_0 KV cache takes precedence over ring buffer and legacy paths
        let kv_q8_caches = if kv_type == KvCacheType::Q8_0 {
            let mut q8s = Vec::with_capacity(num_attn);
            for _ in 0..num_attn {
                q8s.push(KvQ8Cache::new(
                    config.num_kv_heads,
                    config.head_dim,
                    KV_RING_INITIAL_CAP,
                    device,
                )?);
            }
            let bytes_per_head_pos =
                crate::kernels::kv_q8_0::q8_0_byte_size(config.head_dim);
            let f16_bytes_per_head_pos = config.head_dim * 2;
            let savings_pct =
                (1.0 - bytes_per_head_pos as f64 / f16_bytes_per_head_pos as f64) * 100.0;
            eprintln!(
                "[KV_Q8_0] Enabled: {} layers, initial capacity {} positions, \
                 {:.0}% savings vs F16 ({} vs {} bytes/head/pos), \
                 {:.1} MB total initial",
                num_attn,
                KV_RING_INITIAL_CAP,
                savings_pct,
                bytes_per_head_pos,
                f16_bytes_per_head_pos,
                (2 * KV_RING_INITIAL_CAP * config.num_kv_heads * bytes_per_head_pos * num_attn)
                    as f64
                    / (1024.0 * 1024.0),
            );
            Some(q8s)
        } else {
            None
        };

        let kv_ring_caches = if use_ring && kv_type != KvCacheType::Q8_0 {
            let mut rings = Vec::with_capacity(num_attn);
            for _ in 0..num_attn {
                rings.push(KvRingCache::new(
                    config.num_kv_heads,
                    config.head_dim,
                    KV_RING_INITIAL_CAP,
                    DType::F16,
                    device,
                )?);
            }
            eprintln!(
                "[KV_RING] Enabled: {} layers, initial capacity {} positions, \
                 {:.1} MB/layer ({:.1} MB total)",
                num_attn,
                KV_RING_INITIAL_CAP,
                (2 * KV_RING_INITIAL_CAP * config.num_kv_heads * config.head_dim * 2) as f64
                    / (1024.0 * 1024.0),
                (2 * KV_RING_INITIAL_CAP * config.num_kv_heads * config.head_dim * 2 * num_attn)
                    as f64
                    / (1024.0 * 1024.0),
            );
            Some(rings)
        } else {
            None
        };

        // Pre-allocate MoE buffers if on CUDA (avoids 39× lazy-init check per token)
        let raw_moe_buffers = if matches!(device, Device::Cuda(_)) {
            crate::raw_forward::RawGpuBuffers::new(device).ok()
        } else {
            None
        };

        // Pre-allocate scratch pool if enabled
        let scratch_pool = if crate::scratch_pool::scratch_pool_enabled()
            && matches!(device, Device::Cuda(_))
        {
            crate::scratch_pool::ScratchPool::new(
                config.hidden_size, 512, device,
            ).ok()
        } else {
            None
        };

        // Pre-allocate cudarc MoE buffers if enabled (CHIMERE_CUDARC_MOE=1)
        // Only for MoE models (num_experts > 0).
        let moe_cudarc_bufs = if crate::qwen35_model::moe_cudarc::cudarc_moe_enabled()
            && config.num_experts > 0
        {
            if let Device::Cuda(cuda_dev) = device {
                crate::qwen35_model::moe_cudarc::MoeCudarcBufs::new(
                    config.hidden_size,
                    512, // expert_ffn for Qwen3.5-35B-A3B
                    config.num_experts,
                    config.experts_per_token,
                    cuda_dev,
                ).ok()
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            gdn_states,
            conv_states,
            kv_caches,
            kv_ring_caches,
            kv_q8_caches,
            position: 0,
            compute_device: device.clone(),
            raw_moe_buffers,
            raw_forward_bufs: None,
            ncmoe_bufs: None,
            scratch_pool,
            moe_cudarc_bufs,
        })
    }

    /// Deep-clone all state tensors for MTP branching (in-device copy).
    ///
    /// Both the original and the snapshot remain on `compute_device`.
    /// Use `snapshot_to_cpu` / `restore_from_cpu` when you need a
    /// cross-device (GPU→CPU→GPU) round-trip.
    pub fn snapshot(&self) -> Result<Self> {
        let gdn_states = self
            .gdn_states
            .iter()
            .map(|t| t.copy())
            .collect::<Result<Vec<_>>>()?;

        let conv_states = self
            .conv_states
            .iter()
            .map(|t| t.copy())
            .collect::<Result<Vec<_>>>()?;

        let kv_caches = self
            .kv_caches
            .iter()
            .map(|(k, v)| Ok((k.copy()?, v.copy()?)))
            .collect::<Result<Vec<_>>>()?;

        let kv_ring_caches = if let Some(ref rings) = self.kv_ring_caches {
            let snapped = rings
                .iter()
                .map(|r| r.snapshot())
                .collect::<Result<Vec<_>>>()?;
            Some(snapped)
        } else {
            None
        };

        let kv_q8_caches = if let Some(ref q8s) = self.kv_q8_caches {
            let snapped = q8s
                .iter()
                .map(|q| q.snapshot())
                .collect::<Result<Vec<_>>>()?;
            Some(snapped)
        } else {
            None
        };

        Ok(Self {
            gdn_states,
            conv_states,
            kv_caches,
            kv_ring_caches,
            kv_q8_caches,
            position: self.position,
            compute_device: self.compute_device.clone(),
            raw_moe_buffers: None, // scratch buffers are stateless, no need to snapshot
            raw_forward_bufs: None, // scratch buffers are stateless, no need to snapshot
            ncmoe_bufs: None,      // scratch buffers are stateless, no need to snapshot
            scratch_pool: None,    // scratch buffers are stateless, no need to snapshot
            moe_cudarc_bufs: None, // scratch buffers are stateless, no need to snapshot
        })
    }

    /// Restore state from a previously taken snapshot (in-device copy).
    ///
    /// Copies tensor data in-place so that the snapshot can be reused.
    pub fn restore(&mut self, snapshot: &GdnRecurrentState) -> Result<()> {
        for (dst, src) in self.gdn_states.iter_mut().zip(snapshot.gdn_states.iter()) {
            *dst = src.copy()?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(snapshot.conv_states.iter()) {
            *dst = src.copy()?;
        }
        for ((dk, dv), (sk, sv)) in self.kv_caches.iter_mut().zip(snapshot.kv_caches.iter()) {
            *dk = sk.copy()?;
            *dv = sv.copy()?;
        }
        if let (Some(ref mut dst_rings), Some(ref src_rings)) =
            (&mut self.kv_ring_caches, &snapshot.kv_ring_caches)
        {
            for (dst, src) in dst_rings.iter_mut().zip(src_rings.iter()) {
                dst.restore_from(src)?;
            }
        }
        if let (Some(ref mut dst_q8s), Some(ref src_q8s)) =
            (&mut self.kv_q8_caches, &snapshot.kv_q8_caches)
        {
            for (dst, src) in dst_q8s.iter_mut().zip(src_q8s.iter()) {
                dst.restore_from(src)?;
            }
        }
        self.position = snapshot.position;
        Ok(())
    }

    /// Snapshot all state tensors to CPU memory.
    ///
    /// Used **only** before MTP branching when GPU memory pressure requires
    /// keeping the rollback copy off-device.  The returned `CpuSnapshot`
    /// can be passed to `restore_from_cpu` to move data back to
    /// `compute_device` if the speculative branch is rejected.
    pub fn snapshot_to_cpu(&self) -> Result<CpuSnapshot> {
        let gdn_states = self
            .gdn_states
            .iter()
            .map(|t| t.to_device(&Device::Cpu))
            .collect::<Result<Vec<_>>>()?;
        let conv_states = self
            .conv_states
            .iter()
            .map(|t| t.to_device(&Device::Cpu))
            .collect::<Result<Vec<_>>>()?;
        let kv_caches = self
            .kv_caches
            .iter()
            .map(|(k, v)| Ok((k.to_device(&Device::Cpu)?, v.to_device(&Device::Cpu)?)))
            .collect::<Result<Vec<_>>>()?;
        let kv_ring_caches = if let Some(ref rings) = self.kv_ring_caches {
            let cpu_rings = rings
                .iter()
                .map(|r| r.to_device(&Device::Cpu))
                .collect::<Result<Vec<_>>>()?;
            Some(cpu_rings)
        } else {
            None
        };
        let kv_q8_caches = if let Some(ref q8s) = self.kv_q8_caches {
            let cpu_q8s = q8s
                .iter()
                .map(|q| q.to_device(&Device::Cpu))
                .collect::<Result<Vec<_>>>()?;
            Some(cpu_q8s)
        } else {
            None
        };
        Ok(CpuSnapshot {
            gdn_states,
            conv_states,
            kv_caches,
            kv_ring_caches,
            kv_q8_caches,
            position: self.position,
        })
    }

    /// Restore state from a CPU snapshot back to `compute_device`.
    ///
    /// Transfers all tensors from the snapshot (on CPU) to `self.compute_device`
    /// (typically GPU), then updates the position counter.
    pub fn restore_from_cpu(&mut self, snap: &CpuSnapshot) -> Result<()> {
        for (dst, src) in self.gdn_states.iter_mut().zip(snap.gdn_states.iter()) {
            *dst = src.to_device(&self.compute_device)?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(snap.conv_states.iter()) {
            *dst = src.to_device(&self.compute_device)?;
        }
        for ((dk, dv), (sk, sv)) in self.kv_caches.iter_mut().zip(snap.kv_caches.iter()) {
            *dk = sk.to_device(&self.compute_device)?;
            *dv = sv.to_device(&self.compute_device)?;
        }
        if let (Some(ref mut dst_rings), Some(ref src_rings)) =
            (&mut self.kv_ring_caches, &snap.kv_ring_caches)
        {
            for (dst, src) in dst_rings.iter_mut().zip(src_rings.iter()) {
                let on_device = src.to_device(&self.compute_device)?;
                dst.restore_from(&on_device)?;
            }
        }
        if let (Some(ref mut dst_q8s), Some(ref src_q8s)) =
            (&mut self.kv_q8_caches, &snap.kv_q8_caches)
        {
            for (dst, src) in dst_q8s.iter_mut().zip(src_q8s.iter()) {
                let on_device = src.to_device(&self.compute_device)?;
                dst.restore_from(&on_device)?;
            }
        }
        self.position = snap.position;
        Ok(())
    }

    /// Advance the position counter by `n` tokens.
    pub fn advance(&mut self, n: usize) {
        self.position += n;
    }

    /// Reset all state to zeros and position to 0.
    ///
    /// KV caches are reset to empty (seq_len=0) rather than zeroed at their
    /// current size, so the next generation starts fresh.
    pub fn reset(&mut self) -> Result<()> {
        for t in &mut self.gdn_states {
            *t = t.zeros_like()?;
        }
        for t in &mut self.conv_states {
            *t = t.zeros_like()?;
        }
        // Reset KV caches to empty (seq_len=0) — not just zeroed
        let device = &self.compute_device;
        for (k, v) in &mut self.kv_caches {
            let (batch, num_kv_heads, _seq_len, head_dim) = k.dims4()?;
            *k = Tensor::zeros((batch, num_kv_heads, 0, head_dim), k.dtype(), device)?;
            *v = Tensor::zeros((batch, num_kv_heads, 0, head_dim), v.dtype(), device)?;
        }
        // Reset ring caches (keeps buffer allocated, just resets len to 0)
        if let Some(ref mut rings) = self.kv_ring_caches {
            for r in rings.iter_mut() {
                r.reset();
            }
        }
        // Reset Q8_0 caches
        if let Some(ref mut q8s) = self.kv_q8_caches {
            for q in q8s.iter_mut() {
                q.reset();
            }
        }
        self.position = 0;
        Ok(())
    }

    /// Append to KV cache for attention layer `attn_idx`.
    ///
    /// `k_new` and `v_new` must have shape `[1, num_kv_heads, n, head_dim]`
    /// where `n` is the number of new positions (1 for decode, N for prefill).
    ///
    /// Returns `(k_full, v_full)` covering the entire cached sequence:
    /// - Q8_0 mode (`CHIMERE_KV_TYPE=q8_0`): quantized storage, ~47% savings.
    /// - Ring mode (`CHIMERE_KV_RING=1`): zero-copy `narrow()` view, O(1) append.
    /// - Legacy mode: `Tensor::cat` with full copy, O(P) append.
    pub fn kv_append(
        &mut self,
        attn_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if let Some(ref mut q8s) = self.kv_q8_caches {
            // Q8_0 quantized path — ~47% memory savings vs F16
            let q8 = &mut q8s[attn_idx];
            q8.append(k_new, v_new)
        } else if let Some(ref mut rings) = self.kv_ring_caches {
            // Ring buffer path — O(1) append via slice_set
            let ring = &mut rings[attn_idx];
            ring.append(k_new, v_new)
        } else {
            // Legacy path — Tensor::cat (O(P) copy)
            let (ref k_old, ref v_old) = self.kv_caches[attn_idx];
            let (mut k_cache, mut v_cache) = if k_old.dim(2)? > 0 {
                let k_old_dev = k_old.to_device(k_new.device())?;
                let v_old_dev = v_old.to_device(v_new.device())?;
                (
                    Tensor::cat(&[&k_old_dev, k_new], 2)?,
                    Tensor::cat(&[&v_old_dev, v_new], 2)?,
                )
            } else {
                (k_new.clone(), v_new.clone())
            };
            // Truncate to KV_MAX_SEQ_LEN to prevent OOM on long generations
            if k_cache.dim(2)? > KV_MAX_SEQ_LEN {
                let start = k_cache.dim(2)? - KV_MAX_SEQ_LEN;
                k_cache = k_cache.narrow(2, start, KV_MAX_SEQ_LEN)?.contiguous()?;
                v_cache = v_cache.narrow(2, start, KV_MAX_SEQ_LEN)?.contiguous()?;
            }
            self.kv_caches[attn_idx] = (k_cache.clone(), v_cache.clone());
            Ok((k_cache, v_cache))
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side snapshot for cross-device MTP branching
// ---------------------------------------------------------------------------

/// CPU-side snapshot of a `GdnRecurrentState` for MTP branching.
///
/// All tensors are guaranteed to reside on `Device::Cpu`.  Created by
/// `GdnRecurrentState::snapshot_to_cpu`; consumed by
/// `GdnRecurrentState::restore_from_cpu`.
pub struct CpuSnapshot {
    pub gdn_states: Vec<Tensor>,
    pub conv_states: Vec<Tensor>,
    pub kv_caches: Vec<(Tensor, Tensor)>,
    pub kv_ring_caches: Option<Vec<KvRingCache>>,
    pub kv_q8_caches: Option<Vec<KvQ8Cache>>,
    pub position: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Qwen35Config {
        Qwen35Config::qwen35_35b_a3b()
    }

    #[test]
    fn test_state_new() -> Result<()> {
        let cfg = test_config();
        let state = GdnRecurrentState::new(&cfg, &Device::Cpu)?;

        // GDN state count = 30 layers (40 main - 10 attention)
        assert_eq!(state.gdn_states.len(), 30, "Expected 30 GDN states");
        // Conv state count = 30 layers
        assert_eq!(state.conv_states.len(), 30, "Expected 30 conv states");
        // KV cache count = 10 layers
        assert_eq!(state.kv_caches.len(), 10, "Expected 10 KV caches");
        // Position starts at 0
        assert_eq!(state.position, 0);
        // compute_device is Cpu
        assert!(matches!(state.compute_device, Device::Cpu));

        // Verify GDN state shapes: [1, 48, 128, 128]
        for (i, t) in state.gdn_states.iter().enumerate() {
            assert_eq!(
                t.dims(),
                &[1, cfg.ssm_dt_rank, cfg.ssm_d_state, cfg.ssm_d_state],
                "GDN state {} shape mismatch",
                i
            );
        }

        // Verify conv state shapes: [1, conv_channels, 3] (conv_kernel=4, so buf=3)
        let conv_channels = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        for (i, t) in state.conv_states.iter().enumerate() {
            assert_eq!(
                t.dims(),
                &[1, conv_channels, cfg.ssm_conv_kernel - 1],
                "Conv state {} shape mismatch",
                i
            );
        }

        // Verify KV cache shapes: [1, num_kv_heads, 0, head_dim] (seq_len starts at 0)
        for (i, (k, v)) in state.kv_caches.iter().enumerate() {
            assert_eq!(
                k.dims(),
                &[1, cfg.num_kv_heads, 0, cfg.head_dim],
                "KV cache K {} shape mismatch",
                i
            );
            assert_eq!(
                v.dims(),
                &[1, cfg.num_kv_heads, 0, cfg.head_dim],
                "KV cache V {} shape mismatch",
                i
            );
        }

        println!(
            "State initialised: {} GDN, {} conv, {} KV, position={}",
            state.gdn_states.len(),
            state.conv_states.len(),
            state.kv_caches.len(),
            state.position
        );
        Ok(())
    }

    #[test]
    fn test_snapshot_restore() -> Result<()> {
        let cfg = test_config();
        let mut state = GdnRecurrentState::new(&cfg, &Device::Cpu)?;

        // Modify state: set gdn_states[0] to ones and advance position
        state.gdn_states[0] = Tensor::ones(
            (1, cfg.ssm_dt_rank, cfg.ssm_d_state, cfg.ssm_d_state),
            DType::F32,
            &Device::Cpu,
        )?;
        state.position = 42;

        // Take snapshot (in-device copy)
        let snap = state.snapshot()?;
        assert!(matches!(snap.compute_device, Device::Cpu));

        // Further modify state
        state.gdn_states[0] = Tensor::full(
            2.0f32,
            (1, cfg.ssm_dt_rank, cfg.ssm_d_state, cfg.ssm_d_state),
            &Device::Cpu,
        )?;
        state.position = 100;

        // Verify state was modified
        let val: f32 = state.gdn_states[0].flatten_all()?.get(0)?.to_scalar()?;
        assert!((val - 2.0).abs() < 1e-6, "State should be 2.0 after modification");
        assert_eq!(state.position, 100);

        // Restore from snapshot
        state.restore(&snap)?;

        // Verify restoration
        let val: f32 = state.gdn_states[0].flatten_all()?.get(0)?.to_scalar()?;
        assert!((val - 1.0).abs() < 1e-6, "State should be 1.0 after restore, got {}", val);
        assert_eq!(state.position, 42, "Position should be 42 after restore");

        // Verify other states are still zero
        let val2: f32 = state.gdn_states[1].flatten_all()?.get(0)?.to_scalar()?;
        assert!((val2).abs() < 1e-6, "Unmodified state should be 0.0, got {}", val2);

        println!("Snapshot/restore verified");
        Ok(())
    }

    #[test]
    fn test_snapshot_to_cpu_restore_from_cpu() -> Result<()> {
        let cfg = test_config();
        // compute_device = Cpu in tests; the GPU→CPU→GPU path still exercises
        // the to_device() round-trip (it's a no-op on Cpu but structurally correct).
        let mut state = GdnRecurrentState::new(&cfg, &Device::Cpu)?;

        // Set gdn_states[0] to ones and a known position
        state.gdn_states[0] = Tensor::ones(
            (1, cfg.ssm_dt_rank, cfg.ssm_d_state, cfg.ssm_d_state),
            DType::F32,
            &Device::Cpu,
        )?;
        state.position = 77;

        // Snapshot to CPU
        let snap = state.snapshot_to_cpu()?;
        assert_eq!(snap.position, 77);
        // All snapshot tensors must be on CPU
        for t in &snap.gdn_states {
            assert!(matches!(t.device(), Device::Cpu));
        }
        for t in &snap.conv_states {
            assert!(matches!(t.device(), Device::Cpu));
        }
        for (k, v) in &snap.kv_caches {
            assert!(matches!(k.device(), Device::Cpu));
            assert!(matches!(v.device(), Device::Cpu));
        }

        // Overwrite gdn_states[0] and position
        state.gdn_states[0] = Tensor::full(
            3.0f32,
            (1, cfg.ssm_dt_rank, cfg.ssm_d_state, cfg.ssm_d_state),
            &Device::Cpu,
        )?;
        state.position = 200;

        // Restore from CPU snapshot
        state.restore_from_cpu(&snap)?;

        // State should be back to ones, position = 77
        let val: f32 = state.gdn_states[0].flatten_all()?.get(0)?.to_scalar()?;
        assert!(
            (val - 1.0).abs() < 1e-6,
            "Expected 1.0 after restore_from_cpu, got {}",
            val
        );
        assert_eq!(state.position, 77, "Position should be 77 after restore_from_cpu");

        // compute_device must be unchanged
        assert!(matches!(state.compute_device, Device::Cpu));

        println!("snapshot_to_cpu / restore_from_cpu verified");
        Ok(())
    }

    #[test]
    fn test_advance_position() -> Result<()> {
        let cfg = test_config();
        let mut state = GdnRecurrentState::new(&cfg, &Device::Cpu)?;

        assert_eq!(state.position, 0);
        state.advance(10);
        assert_eq!(state.position, 10);
        state.advance(5);
        assert_eq!(state.position, 15);
        state.advance(0);
        assert_eq!(state.position, 15);

        println!("Position advance verified");
        Ok(())
    }

    #[test]
    fn test_reset() -> Result<()> {
        let cfg = test_config();
        let mut state = GdnRecurrentState::new(&cfg, &Device::Cpu)?;

        // Modify everything
        state.gdn_states[0] = Tensor::ones(
            (1, cfg.ssm_dt_rank, cfg.ssm_d_state, cfg.ssm_d_state),
            DType::F32,
            &Device::Cpu,
        )?;
        let conv_channels = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        state.conv_states[0] = Tensor::ones(
            (1, conv_channels, cfg.ssm_conv_kernel - 1),
            DType::F32,
            &Device::Cpu,
        )?;
        state.position = 999;

        // Reset
        state.reset()?;

        // Verify all zeros and compute_device preserved
        assert_eq!(state.position, 0, "Position should be 0 after reset");
        assert!(matches!(state.compute_device, Device::Cpu));

        let gdn_sum: f32 = state.gdn_states[0].abs()?.sum_all()?.to_scalar()?;
        assert!(gdn_sum < 1e-10, "GDN state should be zero after reset, got {}", gdn_sum);

        let conv_sum: f32 = state.conv_states[0].abs()?.sum_all()?.to_scalar()?;
        assert!(conv_sum < 1e-10, "Conv state should be zero after reset, got {}", conv_sum);

        println!("Reset verified: all states zeroed, position=0");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // KvRingCache unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_ring_basic_append() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 256;
        let mut ring = KvRingCache::new(num_kv_heads, head_dim, 8, DType::F32, &Device::Cpu)?;
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.capacity, 8);

        // Append a single token
        let k1 = Tensor::ones((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu)?;
        let v1 = Tensor::full(2.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, vv) = ring.append(&k1, &v1)?;

        assert_eq!(ring.len(), 1);
        assert_eq!(kv.dims(), &[1, num_kv_heads, 1, head_dim]);
        assert_eq!(vv.dims(), &[1, num_kv_heads, 1, head_dim]);

        // Verify values
        let k_val: f32 = kv.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k_val - 1.0).abs() < 1e-6, "K should be 1.0, got {}", k_val);
        let v_val: f32 = vv.flatten_all()?.get(0)?.to_scalar()?;
        assert!((v_val - 2.0).abs() < 1e-6, "V should be 2.0, got {}", v_val);

        // Append more tokens
        for i in 2..=5 {
            let ki = Tensor::full(i as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let vi = Tensor::full((i * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let (kv, vv) = ring.append(&ki, &vi)?;
            assert_eq!(ring.len(), i);
            assert_eq!(kv.dim(2)?, i);
            assert_eq!(vv.dim(2)?, i);
        }

        // Verify the first value is still correct after later appends
        let k_view = ring.k_buf.narrow(2, 0, 1)?;
        let first_k: f32 = k_view.flatten_all()?.get(0)?.to_scalar()?;
        assert!((first_k - 1.0).abs() < 1e-6, "First K should still be 1.0, got {}", first_k);

        println!("KvRingCache basic append verified: len={}", ring.len());
        Ok(())
    }

    #[test]
    fn test_kv_ring_grow() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 4; // small for testing
        let mut ring = KvRingCache::new(num_kv_heads, head_dim, 4, DType::F32, &Device::Cpu)?;
        assert_eq!(ring.capacity, 4);

        // Fill to capacity
        for i in 0..4 {
            let k = Tensor::full((i + 1) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let v = Tensor::full(((i + 1) * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            ring.append(&k, &v)?;
        }
        assert_eq!(ring.len(), 4);
        assert_eq!(ring.capacity, 4);

        // One more should trigger growth
        let k5 = Tensor::full(5.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let v5 = Tensor::full(50.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, _vv) = ring.append(&k5, &v5)?;

        assert_eq!(ring.len(), 5);
        assert_eq!(ring.capacity, 8, "Capacity should have doubled to 8");
        assert_eq!(kv.dim(2)?, 5);

        // Verify old data survived the growth
        // Position 0 should have value 1.0 for K
        let k0_view = kv.narrow(2, 0, 1)?.contiguous()?;
        let k0_val: f32 = k0_view.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k0_val - 1.0).abs() < 1e-6, "K[0] should be 1.0 after grow, got {}", k0_val);

        // Position 4 should have value 5.0 for K
        let k4_view = kv.narrow(2, 4, 1)?.contiguous()?;
        let k4_val: f32 = k4_view.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k4_val - 5.0).abs() < 1e-6, "K[4] should be 5.0, got {}", k4_val);

        println!("KvRingCache grow verified: cap={}", ring.capacity);
        Ok(())
    }

    #[test]
    fn test_kv_ring_batch_append() -> Result<()> {
        // Test appending multiple positions at once (prefill path)
        let num_kv_heads = 2;
        let head_dim = 4;
        let mut ring = KvRingCache::new(num_kv_heads, head_dim, 8, DType::F32, &Device::Cpu)?;

        // Append 5 positions at once
        let k_batch = Tensor::ones((1, num_kv_heads, 5, head_dim), DType::F32, &Device::Cpu)?;
        let v_batch = Tensor::full(3.0f32, (1, num_kv_heads, 5, head_dim), &Device::Cpu)?;
        let (kv, vv) = ring.append(&k_batch, &v_batch)?;

        assert_eq!(ring.len(), 5);
        assert_eq!(kv.dim(2)?, 5);
        assert_eq!(vv.dim(2)?, 5);

        // Then append 1 more (decode step)
        let k1 = Tensor::full(7.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let v1 = Tensor::full(9.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, _) = ring.append(&k1, &v1)?;
        assert_eq!(ring.len(), 6);
        assert_eq!(kv.dim(2)?, 6);

        // Verify the last position
        let k5 = kv.narrow(2, 5, 1)?.contiguous()?;
        let k5_val: f32 = k5.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k5_val - 7.0).abs() < 1e-6, "K[5] should be 7.0, got {}", k5_val);

        println!("KvRingCache batch append verified: len={}", ring.len());
        Ok(())
    }

    #[test]
    fn test_kv_ring_snapshot_restore() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 4;
        let mut ring = KvRingCache::new(num_kv_heads, head_dim, 8, DType::F32, &Device::Cpu)?;

        // Append 3 tokens
        for i in 1..=3 {
            let k = Tensor::full(i as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let v = Tensor::full((i * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            ring.append(&k, &v)?;
        }
        assert_eq!(ring.len(), 3);

        // Snapshot
        let snap = ring.snapshot()?;
        assert_eq!(snap.len(), 3);

        // Append more to original
        for i in 4..=6 {
            let k = Tensor::full(i as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let v = Tensor::full((i * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            ring.append(&k, &v)?;
        }
        assert_eq!(ring.len(), 6);

        // Restore from snapshot
        ring.restore_from(&snap)?;
        assert_eq!(ring.len(), 3);

        // Verify data is correct after restore
        let k_view = ring.k_buf.narrow(2, 0, 3)?.contiguous()?;
        let k0: f32 = k_view.narrow(2, 0, 1)?.flatten_all()?.get(0)?.to_scalar()?;
        let k2: f32 = k_view.narrow(2, 2, 1)?.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k0 - 1.0).abs() < 1e-6, "K[0] should be 1.0 after restore, got {}", k0);
        assert!((k2 - 3.0).abs() < 1e-6, "K[2] should be 3.0 after restore, got {}", k2);

        println!("KvRingCache snapshot/restore verified");
        Ok(())
    }

    #[test]
    fn test_kv_ring_reset() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 4;
        let mut ring = KvRingCache::new(num_kv_heads, head_dim, 8, DType::F32, &Device::Cpu)?;

        // Append some data
        let k = Tensor::ones((1, num_kv_heads, 3, head_dim), DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, num_kv_heads, 3, head_dim), DType::F32, &Device::Cpu)?;
        ring.append(&k, &v)?;
        assert_eq!(ring.len(), 3);

        // Reset
        ring.reset();
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.capacity, 8, "Capacity should be preserved after reset");

        // Should be able to append again
        let k1 = Tensor::full(5.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let v1 = Tensor::full(5.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, _) = ring.append(&k1, &v1)?;
        assert_eq!(ring.len(), 1);
        let val: f32 = kv.flatten_all()?.get(0)?.to_scalar()?;
        assert!((val - 5.0).abs() < 1e-6, "After reset+append, K should be 5.0, got {}", val);

        println!("KvRingCache reset verified");
        Ok(())
    }

    // -----------------------------------------------------------------------
    // KvQ8Cache unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_q8_basic_append() -> Result<()> {
        // head_dim must be multiple of 32 for Q8_0
        let num_kv_heads = 2;
        let head_dim = 128; // 4 Q8_0 blocks
        let mut cache = KvQ8Cache::new(num_kv_heads, head_dim, 8, &Device::Cpu)?;
        assert_eq!(cache.len(), 0);

        // Append a single token
        let k1 = Tensor::ones((1, num_kv_heads, 1, head_dim), DType::F32, &Device::Cpu)?;
        let v1 = Tensor::full(2.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, vv) = cache.append(&k1, &v1)?;

        assert_eq!(cache.len(), 1);
        assert_eq!(kv.dims(), &[1, num_kv_heads, 1, head_dim]);
        assert_eq!(vv.dims(), &[1, num_kv_heads, 1, head_dim]);

        // Verify values (Q8_0 has quantization noise, so use tolerance)
        let k_val: f32 = kv.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k_val - 1.0).abs() < 0.05, "K should be ~1.0, got {}", k_val);
        let v_val: f32 = vv.flatten_all()?.get(0)?.to_scalar()?;
        assert!((v_val - 2.0).abs() < 0.05, "V should be ~2.0, got {}", v_val);

        // Append more tokens
        for i in 2..=5 {
            let ki = Tensor::full(i as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let vi = Tensor::full((i * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let (kv, vv) = cache.append(&ki, &vi)?;
            assert_eq!(cache.len(), i);
            assert_eq!(kv.dim(2)?, i);
            assert_eq!(vv.dim(2)?, i);
        }

        println!("KvQ8Cache basic append verified: len={}", cache.len());
        Ok(())
    }

    #[test]
    fn test_kv_q8_grow() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 128;
        let mut cache = KvQ8Cache::new(num_kv_heads, head_dim, 4, &Device::Cpu)?;
        assert_eq!(cache.capacity, 4);

        // Fill to capacity
        for i in 0..4 {
            let k = Tensor::full((i + 1) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let v = Tensor::full(((i + 1) * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            cache.append(&k, &v)?;
        }
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.capacity, 4);

        // One more should trigger growth
        let k5 = Tensor::full(5.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let v5 = Tensor::full(50.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, _) = cache.append(&k5, &v5)?;

        assert_eq!(cache.len(), 5);
        assert_eq!(cache.capacity, 8, "Capacity should have doubled to 8");
        assert_eq!(kv.dim(2)?, 5);

        // Verify old data survived growth (Q8_0 tolerance)
        let k0_view = kv.narrow(2, 0, 1)?.contiguous()?;
        let k0_val: f32 = k0_view.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k0_val - 1.0).abs() < 0.05, "K[0] should be ~1.0 after grow, got {}", k0_val);

        let k4_view = kv.narrow(2, 4, 1)?.contiguous()?;
        let k4_val: f32 = k4_view.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k4_val - 5.0).abs() < 0.1, "K[4] should be ~5.0, got {}", k4_val);

        println!("KvQ8Cache grow verified: cap={}", cache.capacity);
        Ok(())
    }

    #[test]
    fn test_kv_q8_snapshot_restore() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 128;
        let mut cache = KvQ8Cache::new(num_kv_heads, head_dim, 8, &Device::Cpu)?;

        // Append 3 tokens
        for i in 1..=3 {
            let k = Tensor::full(i as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let v = Tensor::full((i * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            cache.append(&k, &v)?;
        }
        assert_eq!(cache.len(), 3);

        // Snapshot
        let snap = cache.snapshot()?;
        assert_eq!(snap.len(), 3);

        // Append more to original
        for i in 4..=6 {
            let k = Tensor::full(i as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            let v = Tensor::full((i * 10) as f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
            cache.append(&k, &v)?;
        }
        assert_eq!(cache.len(), 6);

        // Restore from snapshot
        cache.restore_from(&snap)?;
        assert_eq!(cache.len(), 3);

        // Verify data is correct after restore (Q8_0 tolerance)
        let (k_full, _) = cache.dequantize_full()?;
        let k0: f32 = k_full.narrow(2, 0, 1)?.flatten_all()?.get(0)?.to_scalar()?;
        let k2: f32 = k_full.narrow(2, 2, 1)?.flatten_all()?.get(0)?.to_scalar()?;
        assert!((k0 - 1.0).abs() < 0.05, "K[0] should be ~1.0 after restore, got {}", k0);
        assert!((k2 - 3.0).abs() < 0.1, "K[2] should be ~3.0 after restore, got {}", k2);

        println!("KvQ8Cache snapshot/restore verified");
        Ok(())
    }

    #[test]
    fn test_kv_q8_reset() -> Result<()> {
        let num_kv_heads = 2;
        let head_dim = 128;
        let mut cache = KvQ8Cache::new(num_kv_heads, head_dim, 8, &Device::Cpu)?;

        // Append some data
        let k = Tensor::ones((1, num_kv_heads, 3, head_dim), DType::F32, &Device::Cpu)?;
        let v = Tensor::ones((1, num_kv_heads, 3, head_dim), DType::F32, &Device::Cpu)?;
        cache.append(&k, &v)?;
        assert_eq!(cache.len(), 3);

        // Reset
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity, 8, "Capacity should be preserved after reset");

        // Should be able to append again
        let k1 = Tensor::full(5.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let v1 = Tensor::full(5.0f32, (1, num_kv_heads, 1, head_dim), &Device::Cpu)?;
        let (kv, _) = cache.append(&k1, &v1)?;
        assert_eq!(cache.len(), 1);
        let val: f32 = kv.flatten_all()?.get(0)?.to_scalar()?;
        assert!((val - 5.0).abs() < 0.1, "After reset+append, K should be ~5.0, got {}", val);

        println!("KvQ8Cache reset verified");
        Ok(())
    }

    #[test]
    fn test_kv_q8_memory_savings() -> Result<()> {
        // Verify Q8_0 uses ~47% less memory than F16
        let num_kv_heads = 4;
        let head_dim = 128; // Qwen3.5-35B-A3B head_dim
        let capacity = 4096;

        // F16 memory: 2 * capacity * num_kv_heads * head_dim * 2 bytes
        let f16_bytes = 2 * capacity * num_kv_heads * head_dim * 2;

        // Q8_0 memory: 2 * capacity * num_kv_heads * q8_0_byte_size(head_dim)
        let q8_bytes_per_head_pos = crate::kernels::kv_q8_0::q8_0_byte_size(head_dim);
        let q8_bytes = 2 * capacity * num_kv_heads * q8_bytes_per_head_pos;

        let savings = 1.0 - (q8_bytes as f64 / f16_bytes as f64);
        eprintln!(
            "KV cache memory per layer @ {} positions:\n  F16: {:.1} MB\n  Q8_0: {:.1} MB\n  Savings: {:.1}%",
            capacity,
            f16_bytes as f64 / (1024.0 * 1024.0),
            q8_bytes as f64 / (1024.0 * 1024.0),
            savings * 100.0,
        );

        assert!(
            savings > 0.45 && savings < 0.50,
            "Expected ~47% savings, got {:.1}%",
            savings * 100.0
        );
        Ok(())
    }
}
