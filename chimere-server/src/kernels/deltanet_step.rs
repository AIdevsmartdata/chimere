//! Fused DeltaNet recurrent step kernel — single CUDA launch per GDN layer.
//!
//! Processes the state update for all heads in parallel: one block per head,
//! one thread per column of the [D, D] state matrix.
//!
//! # State convention
//!
//! State is stored in S^T (transposed) convention:
//!   `state[h, i, j]` = S[h, j, i]
//!
//! The kernel reads S^T, computes `sk = S @ k`, applies the gated delta update,
//! then writes the updated S^T back. Output: `o[h, j] = sum_i( S_new^T[h, i, j] * q[i] )`
//!
//! # Kernel geometry
//!
//! - Grid: (num_heads,) — one block per head
//! - Block: (D,) — one thread per column j (D=128, d_state)
//! - Shared memory: 2 * D * sizeof(float) for k_s[] and q_s[]
//!
//! # Usage
//!
//! ```ignore
//! let (new_state, output) = deltanet_step_fused(
//!     &state,     // [1, H, D, D]  S^T convention, F32, contiguous
//!     &q_scaled,  // [1, H, D]     already scaled by 1/sqrt(D)
//!     &k,         // [1, H, D]
//!     &v,         // [1, H, D]
//!     &gate_exp,  // [1, H]        already exp(gate_value), in (0,1)
//!     &beta,      // [1, H]        in (0,1)
//! )?;
//! ```

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaView, DevicePtr, DeviceSlice, LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaDevice, CudaStorage};
use candle_core::{DType, Result, Shape, Storage, Tensor};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// CUDA kernel source (extracted from kernels/chimere_kernels.cu)
// ---------------------------------------------------------------------------

const KERNEL_SRC: &str = r#"
extern "C" __global__ __launch_bounds__(128, 1)
void deltanet_step_kernel(
    const float* __restrict__ s_in,
    float*       __restrict__ s_out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float*       __restrict__ output,
    int D
) {
    const int h = blockIdx.x;
    const int j = threadIdx.x;

    extern __shared__ char smem[];
    float* k_s = (float*)smem;
    float* q_s = (float*)smem + D;

    if (j < D) {
        k_s[j] = k[h * D + j];
        q_s[j] = q[h * D + j];
    }
    __syncthreads();

    if (j >= D) return;

    const float g   = gate[h];
    const float b   = beta[h];
    const float v_j = v[h * D + j];

    const float* s_col_in  = s_in  + h * D * D;
    float*       s_col_out = s_out + h * D * D;

    float s_reg[128];
    double sk_j = 0.0;

    #pragma unroll 8
    for (int i = 0; i < D; i++) {
        float val = s_col_in[i * D + j] * g;
        s_reg[i] = val;
        sk_j += (double)val * (double)k_s[i];
    }

    float delta_j = (v_j - (float)sk_j) * b;

    double out_j = 0.0;

    #pragma unroll 8
    for (int i = 0; i < D; i++) {
        float val = s_reg[i] + k_s[i] * delta_j;
        // Clamp state to prevent divergence (matches llama.cpp)
        val = fminf(fmaxf(val, -1e6f), 1e6f);
        s_col_out[i * D + j] = val;
        out_j += (double)val * (double)q_s[i];
    }

    output[h * D + j] = (float)out_j;
}
"#;

const MODULE_NAME: &str = "chimere_deltanet_v5";
const FUNC_NAME: &str = "deltanet_step_kernel";

/// Cached PTX assembly (compiled once from KERNEL_SRC via NVRTC, used as fallback
/// when cubin is not available).
static PTX_CACHE: OnceLock<String> = OnceLock::new();

// ---------------------------------------------------------------------------
// Helper: extract a CudaView<f32> from a contiguous F32 Tensor
// ---------------------------------------------------------------------------

/// Extract the CudaDevice from a CUDA-resident tensor.
fn get_cuda_device(tensor: &Tensor) -> Result<CudaDevice> {
    let (storage, _layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cuda(cuda) => Ok(cuda.device.clone()),
        _ => candle_core::bail!("deltanet_step_fused: tensor is not on CUDA device"),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fused DeltaNet recurrent step: single CUDA kernel launch for the full
/// state update across all heads.
///
/// # Arguments
///
/// - `old_state`: `[1, num_heads, d_state, d_state]` — S^T convention, F32, contiguous
/// - `q`:         `[1, num_heads, d_state]` — pre-scaled by 1/sqrt(d_state)
/// - `k`:         `[1, num_heads, d_state]`
/// - `v`:         `[1, num_heads, d_state]`
/// - `gate_exp`:  `[1, num_heads]` — already `exp(gate_value)`, in (0, 1)
/// - `beta`:      `[1, num_heads]` — in (0, 1)
///
/// # Returns
///
/// `(new_state, output)` where:
/// - `new_state`: `[1, num_heads, d_state, d_state]`
/// - `output`:    `[1, num_heads, d_state]`
pub fn deltanet_step_fused(
    old_state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    gate_exp: &Tensor,
    beta: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // --- Validate dtypes ---
    if old_state.dtype() != DType::F32 {
        candle_core::bail!(
            "deltanet_step_fused: old_state must be F32, got {:?}",
            old_state.dtype()
        );
    }

    // --- Extract shapes ---
    let state_dims = old_state.dims();
    if state_dims.len() != 4 || state_dims[0] != 1 {
        candle_core::bail!(
            "deltanet_step_fused: old_state must be [1, H, D, D], got {:?}",
            state_dims
        );
    }
    let num_heads = state_dims[1];
    let d_state = state_dims[2];
    debug_assert_eq!(
        state_dims[3], d_state,
        "state matrix must be square: D={} vs {}",
        state_dims[2], state_dims[3]
    );

    // --- Ensure contiguity ---
    let old_state = old_state.contiguous()?;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let gate_exp = gate_exp.contiguous()?;
    let beta = beta.contiguous()?;

    // --- Get CUDA device ---
    let cuda_dev = get_cuda_device(&old_state)?;

    // --- Load kernel function (cached after first call) ---
    let (cuda_func, _stream) = super::nvrtc_compile::get_or_load_func(
        &cuda_dev, FUNC_NAME, MODULE_NAME, KERNEL_SRC, &PTX_CACHE,
    )?;

    // --- Allocate output buffers ---
    let state_elems = num_heads * d_state * d_state;
    let output_elems = num_heads * d_state;
    let mut s_out_slice = cuda_dev.alloc_zeros::<f32>(state_elems)?;
    let mut out_slice = cuda_dev.alloc_zeros::<f32>(output_elems)?;

    // --- Extract CudaViews from input tensors ---
    // We hold the storage guards alive across the kernel launch.
    let (s_stor, s_lay) = old_state.storage_and_layout();
    let s_cuda = match &*s_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("unreachable: old_state not CUDA"),
    };
    let s_view = s_cuda.as_cuda_slice::<f32>()?.slice(s_lay.start_offset()..);

    let (q_stor, q_lay) = q.storage_and_layout();
    let q_cuda = match &*q_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("unreachable: q not CUDA"),
    };
    let q_view = q_cuda.as_cuda_slice::<f32>()?.slice(q_lay.start_offset()..);

    let (k_stor, k_lay) = k.storage_and_layout();
    let k_cuda = match &*k_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("unreachable: k not CUDA"),
    };
    let k_view = k_cuda.as_cuda_slice::<f32>()?.slice(k_lay.start_offset()..);

    let (v_stor, v_lay) = v.storage_and_layout();
    let v_cuda = match &*v_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("unreachable: v not CUDA"),
    };
    let v_view = v_cuda.as_cuda_slice::<f32>()?.slice(v_lay.start_offset()..);

    let (g_stor, g_lay) = gate_exp.storage_and_layout();
    let g_cuda = match &*g_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("unreachable: gate_exp not CUDA"),
    };
    let g_view = g_cuda.as_cuda_slice::<f32>()?.slice(g_lay.start_offset()..);

    let (b_stor, b_lay) = beta.storage_and_layout();
    let b_cuda = match &*b_stor {
        Storage::Cuda(c) => c,
        _ => candle_core::bail!("unreachable: beta not CUDA"),
    };
    let b_view = b_cuda.as_cuda_slice::<f32>()?.slice(b_lay.start_offset()..);

    // --- Launch kernel ---
    let stream = cuda_dev.cuda_stream();
    let d_state_i32 = d_state as i32;

    // Grid: (num_heads, 1, 1) — one block per head
    // Block: (d_state, 1, 1) — one thread per column
    // Shared memory: 2 * D * sizeof(float) for k_s[] and q_s[]
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (d_state as u32, 1, 1),
        shared_mem_bytes: (2 * d_state * std::mem::size_of::<f32>()) as u32,
    };

    unsafe {
        stream
            .launch_builder(&cuda_func)
            .arg(&s_view)         // s_in
            .arg(&mut s_out_slice) // s_out
            .arg(&q_view)         // q
            .arg(&k_view)         // k
            .arg(&v_view)         // v
            .arg(&g_view)         // gate
            .arg(&b_view)         // beta
            .arg(&mut out_slice)  // output
            .arg(&d_state_i32)    // D
            .launch(cfg)
            .map_err(|e| candle_core::Error::Msg(format!("deltanet_step kernel launch: {e}")))?;
    }

    // Drop storage guards before wrapping output (release read locks).
    drop(s_stor);
    drop(q_stor);
    drop(k_stor);
    drop(v_stor);
    drop(g_stor);
    drop(b_stor);

    // --- Wrap output CudaSlices into Tensors ---
    let new_state_storage =
        CudaStorage::wrap_cuda_slice(s_out_slice, cuda_dev.clone());
    let new_state = Tensor::from_storage(
        Storage::Cuda(new_state_storage),
        Shape::from((1, num_heads, d_state, d_state)),
        candle_core::op::BackpropOp::none(),
        false,
    );

    let output_storage = CudaStorage::wrap_cuda_slice(out_slice, cuda_dev);
    let output = Tensor::from_storage(
        Storage::Cuda(output_storage),
        Shape::from((1, num_heads, d_state)),
        candle_core::op::BackpropOp::none(),
        false,
    );

    Ok((new_state, output))
}

// ---------------------------------------------------------------------------
// Raw variant — for raw forward v4
// ---------------------------------------------------------------------------

/// Fused DeltaNet recurrent step operating directly on pre-allocated
/// `CudaSlice<f32>` buffers, avoiding all Tensor creation/allocation overhead.
///
/// This is the hot-path variant used by `raw_forward_v4`, which maintains
/// per-layer GDN state as raw `CudaSlice<f32>` handles. By writing outputs
/// into caller-provided buffers and taking inputs by `CudaView`, this variant
/// eliminates the 6 Tensor extractions + 2 Tensor allocations that
/// `deltanet_step_fused` performs on every GDN layer call.
///
/// Uses the **same PTX kernel** (`deltanet_step_kernel`) as the Tensor variant.
/// The kernel arguments are identical; only the Rust-side setup differs.
///
/// # Arguments
///
/// - `state`:        `[1, num_heads, d_state, d_state]` — S^T convention, contiguous
/// - `q_scaled`:     `[1, num_heads, d_state]` — pre-scaled by 1/sqrt(d_state)
/// - `k_expanded`:   `[1, num_heads, d_state]`
/// - `v_3d`:         `[1, num_heads, d_state]`
/// - `gate_exp`:     `[1, num_heads]` — already `exp(gate_value)`, in (0, 1)
/// - `beta`:         `[1, num_heads]` — in (0, 1)
/// - `new_state_out`: pre-allocated `[1, num_heads, d_state, d_state]` — written in place
/// - `output_out`:    pre-allocated `[1, num_heads, d_state]` — written in place
/// - `num_heads`:    number of GDN heads (dt_rank = 32 for Qwen3.5-35B-A3B)
/// - `d_state`:      state dimension (128 for Qwen3.5-35B-A3B)
/// - `dev`:          the `CudaDevice` owning all buffers
///
/// # Returns
///
/// `Ok(())` on success; `new_state_out` and `output_out` contain the results.
pub fn deltanet_step_fused_raw(
    state: &CudaView<'_, f32>,
    q_scaled: &CudaView<'_, f32>,
    k_expanded: &CudaView<'_, f32>,
    v_3d: &CudaView<'_, f32>,
    gate_exp: &CudaView<'_, f32>,
    beta: &CudaView<'_, f32>,
    new_state_out: &mut CudaSlice<f32>,
    output_out: &mut CudaSlice<f32>,
    num_heads: usize,
    d_state: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // --- Load kernel function (cached after first call, same PTX as Tensor variant) ---
    let (cuda_func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, FUNC_NAME, MODULE_NAME, KERNEL_SRC, &PTX_CACHE,
    )?;

    // --- Launch kernel ---
    let stream = dev.cuda_stream();
    let d_state_i32 = d_state as i32;

    // Grid: (num_heads, 1, 1) — one block per head
    // Block: (d_state, 1, 1) — one thread per column
    // Shared memory: 2 * D * sizeof(float) for k_s[] and q_s[]
    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (d_state as u32, 1, 1),
        shared_mem_bytes: (2 * d_state * std::mem::size_of::<f32>()) as u32,
    };

    unsafe {
        stream
            .launch_builder(&cuda_func)
            .arg(state)            // s_in
            .arg(new_state_out)    // s_out  (mutable, written in place)
            .arg(q_scaled)         // q
            .arg(k_expanded)       // k
            .arg(v_3d)             // v
            .arg(gate_exp)         // gate
            .arg(beta)             // beta
            .arg(output_out)       // output (mutable, written in place)
            .arg(&d_state_i32)     // D
            .launch(cfg)
            .map_err(|e| {
                candle_core::Error::Msg(format!("deltanet_step_fused_raw kernel launch: {e}"))
            })?;
    }

    Ok(())
}

/// In-place variant: state is updated in-place (s_in == s_out).
///
/// Safe because the kernel reads each column entirely into registers
/// before writing back. No inter-thread dependency on state.
/// Eliminates one memcpy_dtod per GDN layer (30 × 2MB = 60MB/token).
pub fn deltanet_step_fused_raw_inplace(
    state: &mut CudaSlice<f32>,
    q_scaled: &CudaView<'_, f32>,
    k_expanded: &CudaView<'_, f32>,
    v_3d: &CudaView<'_, f32>,
    gate_exp: &CudaView<'_, f32>,
    beta: &CudaView<'_, f32>,
    output_out: &mut CudaSlice<f32>,
    num_heads: usize,
    d_state: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (cuda_func, _stream) = super::nvrtc_compile::get_or_load_func(
        dev, FUNC_NAME, MODULE_NAME, KERNEL_SRC, &PTX_CACHE,
    )?;

    let stream = dev.cuda_stream();
    let d_state_i32 = d_state as i32;

    let cfg = LaunchConfig {
        grid_dim: (num_heads as u32, 1, 1),
        block_dim: (d_state as u32, 1, 1),
        shared_mem_bytes: (2 * d_state * std::mem::size_of::<f32>()) as u32,
    };

    // Pass state as both s_in and s_out via raw pointers to bypass borrow checker.
    // SAFETY: kernel reads entire column to registers before writing (no aliasing issue).
    let state_ptr = {
        let view: CudaView<'_, f32> = state.slice(..);
        let (ptr, _sync) = view.device_ptr(view.stream());
        ptr
    };

    unsafe {
        stream
            .launch_builder(&cuda_func)
            .arg(&state_ptr)       // s_in  (raw ptr)
            .arg(state)            // s_out (same buffer, via &mut)
            .arg(q_scaled)
            .arg(k_expanded)
            .arg(v_3d)
            .arg(gate_exp)
            .arg(beta)
            .arg(output_out)
            .arg(&d_state_i32)
            .launch(cfg)
            .map_err(|e| {
                candle_core::Error::Msg(format!("deltanet_step_fused_raw_inplace launch: {e}"))
            })?;
    }

    Ok(())
}
