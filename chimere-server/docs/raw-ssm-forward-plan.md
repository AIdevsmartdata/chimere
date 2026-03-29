# Raw SSM Forward Path Design

## Objective

Eliminate ~1.8 ms/token of Candle Tensor dispatch overhead from the SSM portion
of `forward_gdn_layer_moe` by replacing all intermediate Candle Tensor operations
with pre-allocated `CudaSlice<f32>` operations.

**Current cost:** ~30 Candle Tensor ops per SSM layer x 30 GDN layers = ~900 ops
x ~2 us/op = **1.8 ms pure dispatch overhead**.

**Target:** 0 Candle Tensor allocations in the SSM hot path (all buffers
pre-allocated, all computation via raw CUDA kernel launches on CudaSlice).

---

## Current SSM Forward Path (from `forward_gdn_layer_moe`)

Reference: `src/qwen35_model.rs` lines 1454-1693.

### Architectural Constants (Qwen3.5-35B-A3B)

| Symbol | Value | Derivation |
|---|---|---|
| hidden_size | 2048 | config |
| n_group (H_k) | 16 | config.ssm_n_group |
| d_state (S) | 128 | config.ssm_d_state |
| dt_rank (H_v) | 32 | config.ssm_dt_rank |
| key_dim | 2048 | n_group * d_state = 16*128 |
| value_dim | 4096 | dt_rank * d_state = 32*128 |
| conv_channels | 8192 | key_dim*2 + value_dim |
| conv_kernel | 4 | config.ssm_conv_kernel |
| repeats | 2 | dt_rank / n_group = 32/16 |
| scale | 1/sqrt(128) | 1/sqrt(d_state) |

---

## Operation-by-Operation Analysis

### Op 1: RMS Norm (hidden -> normed)

**Candle ops:** `rms_norm(hidden, w.attn_norm, eps)` -> `candle_nn::ops::rms_norm`
which internally calls Candle's fused CUDA rms_norm kernel. Still creates a new
Tensor for the output.

**Shapes:**
- Input: `hidden` [1, 2048] (or flat [2048])
- Weight: `w.attn_norm` [2048]
- Output: `normed` [2048]

**Raw equivalent:** `raw_rms_norm()` in `elementwise.rs` -- **EXISTS**.
Operates on `CudaSlice<f32>` input/weight/output.

**Candle ops eliminated:** 1 (Tensor allocation for rms_norm output)

---

### Op 2: QKV Projection (normed -> qkv)

**Candle ops:** `w.attn_qkv.forward(&normed)` -- QMatMul on Q5_K weights.
With `CHIMERE_FUSED_SSM_PROJ=1`, both qkv + gate are done in a single fused
Q5K GEMV call (`gemv_q5k_q8_dual_from_tensor`).

**Shapes:**
- Input: `normed` [1, 2048]
- Weight: Q5_K [8192, 2048]
- Output: `qkv` [1, 8192]

**Raw equivalent:** The fused dual GEMV already extracts CudaSlices internally.
The issue is that it returns Tensors (`qkv_1d.unsqueeze(0)?`, `z_1d.unsqueeze(0)?`).
We need a variant that writes directly into pre-allocated `CudaSlice<f32>` buffers.

**New function needed:** `gemv_q5k_q8_dual_raw()` that takes CudaSlice input and
writes to CudaSlice output buffers. The Q5_K weight bytes stay as-is (raw U8 Tensor
whose CudaSlice is extracted at init time).

Alternatively: keep QMatMul for the projection (it's bandwidth-bound, not
dispatch-bound), but immediately extract the CudaSlice from the result Tensor
and discard the Tensor wrapper. This avoids modifying the GEMV kernel.

**Strategy:** Extract CudaSlice from QMatMul output via
`tensor.storage_and_layout()` -> `as_cuda_slice::<f32>()`. Copy into
pre-allocated buffer with `dtod_copy`. The copy is on the same device so it's
just a memcpy (8192 * 4 = 32 KB, negligible at GPU speeds).

**Candle ops eliminated:** 2 (qkv Tensor + unsqueeze, gate Tensor + unsqueeze)
- With fused path: 4 ops (flatten_all, dual_from_tensor -> 2 unsqueeze)
- Net: ~4 Tensor allocations avoided

---

### Op 3: Gate Projection (normed -> z)

Handled together with Op 2 in the fused path. Same analysis applies.

**Shapes:**
- Weight: Q5_K [4096, 2048]
- Output: `z` [1, 4096]

---

### Op 4: Beta Projection (normed -> beta_proj)

**Candle ops:** `w.ssm_beta.forward(&normed)` -- QMatMul on Q8_0 weights.

**Shapes:**
- Input: `normed` [1, 2048]
- Weight: Q8_0 [32, 2048]
- Output: `beta_proj` [1, 32]

**Raw equivalent:** QMatMul stays (tiny matrix, latency-bound not dispatch-bound).
Extract CudaSlice from result immediately.

**Candle ops eliminated:** 1 (beta_proj Tensor allocation)

---

### Op 5: Alpha Projection (normed -> alpha_proj)

**Candle ops:** `w.ssm_alpha.forward(&normed)` -- QMatMul on Q8_0 weights.

**Shapes:**
- Input: `normed` [1, 2048]
- Weight: Q8_0 [32, 2048]
- Output: `alpha_proj` [1, 32]

**Raw equivalent:** Same as Op 4. Extract CudaSlice.

**Candle ops eliminated:** 1

---

### Op 6: Fused Beta/Alpha/Gate (beta_proj, alpha_proj, dt_bias, ssm_a -> beta, gate_exp)

**Candle ops with CHIMERE_FUSED_ELEM=1:**
`fused_beta_alpha_gate_tensor(beta_proj, alpha_proj, dt_bias, ssm_a)` --
already a custom CUDA kernel, but the `_tensor` variant:
1. Extracts CudaViews from input Tensors (4x `storage_and_layout` + `as_cuda_slice`)
2. Allocates 2 new CudaSlices for beta_out and gate_out
3. Launches kernel
4. Wraps results as new Tensors (2x `Tensor::from_storage`)

**Candle ops without fused path (reference):**
- `sigmoid(beta_proj)` -> 1 op
- `alpha_proj.broadcast_add(dt_bias.unsqueeze(0))` -> 2 ops (unsqueeze + broadcast_add)
- `softplus(alpha_biased)` -> 2 ops (exp + add + log = 3 ops internally)
- `alpha_sp.broadcast_mul(ssm_a.unsqueeze(0))` -> 2 ops
- `gate_value.exp()` -> 1 op
- Total: ~8-10 Tensor allocations

**Shapes:**
- beta_proj: [1, 32]
- alpha_proj: [1, 32]
- dt_bias: [32] (weight, persistent)
- ssm_a: [32] (weight, persistent)
- beta: [1, 32]
- gate_exp: [1, 32]

**Raw equivalent needed:** `raw_fused_beta_alpha_gate()` -- same kernel as
`fused_beta_alpha_gate`, but operating on CudaSlice inputs/outputs directly.
The CUDA kernel already exists in `FUSED_GDN_KERNEL_SRC`. We just need a new
Rust wrapper that takes `CudaSlice<f32>` instead of `Tensor`.

The weight tensors (dt_bias, ssm_a) are persistent -- extract their CudaSlice
at model load time once, store in the raw buffer struct.

**Candle ops eliminated:** 2-10 depending on fused vs reference path

---

### Op 7: Conv1d + SiLU (qkv + conv_state -> conv_activated)

**Candle ops (current):**
```rust
let qkv_col = qkv.reshape((1, conv_channels, 1))?;           // 1 op
let conv_window = Tensor::cat(&[conv_state, &qkv_col], 2)?;  // 1 op
let new_conv = conv_window.narrow(2, 1, conv_kernel - 1)?;    // 1 op (view)
state.conv_states[gdn_idx] = new_conv.contiguous()?;          // 1 op (copy)
let conv_window_2d = conv_window.squeeze(0)?;                 // 1 op (view)
let conv_out = (&conv_window_2d * &w.ssm_conv1d)?             // 1 op (mul)
    .sum(1)?                                                   // 1 op (reduce)
    .unsqueeze(0)?;                                            // 1 op (view)
let conv_activated = conv_out.silu()?;                         // 1 op
```
Total: **~9 Candle ops** -- this is the DENSEST cluster of dispatch overhead.

**Shapes:**
- qkv: [1, 8192] -> reshaped to [1, 8192, 1]
- conv_state: [1, 8192, 3] (sliding window, kernel_size-1 = 3)
- conv_window: [1, 8192, 4] (cat along dim 2)
- new_conv: [1, 8192, 3] (narrow -> contiguous for next step)
- ssm_conv1d: [8192, 4] (depthwise conv weights)
- conv_out: [1, 8192] (dot product per channel)
- conv_activated: [1, 8192] (silu applied)

**Raw equivalent needed:** `raw_conv1d_silu()` -- **DOES NOT EXIST. MUST BE CREATED.**

This is a depthwise 1D convolution with kernel_size=4 followed by SiLU activation.
For single-token inference (batch=1), this is trivially parallelizable:

```
For each channel c in 0..conv_channels:
    conv_out[c] = sum_{k=0}^{3} conv_window[c, k] * weight[c, k]
    output[c] = conv_out[c] * sigmoid(conv_out[c])   // SiLU
```

The conv_state management also needs to go raw:
1. Shift the sliding window: copy conv_state[:, :, 1:] to conv_state[:, :, 0:]
2. Write new qkv into conv_state[:, :, conv_kernel-2] (last column)
3. Read the full [8192, 4] window (3 from state + 1 new)
4. Dot product with weights + SiLU

**Proposed CUDA kernel:**

```cuda
// Grid: (ceil(conv_channels/256), 1, 1)  Block: (256, 1, 1)
// Each thread handles one channel.
extern "C" __global__ void conv1d_silu_kernel(
    float* __restrict__ conv_state,  // [conv_channels, 3] (kernel-1)
    const float* __restrict__ new_col,   // [conv_channels] (new qkv column)
    const float* __restrict__ weight,    // [conv_channels, 4]
    float* __restrict__ output,      // [conv_channels]
    int conv_channels,
    int kernel_size                   // 4
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_channels) return;

    // Read old state columns + new column, compute dot product
    float acc = 0.0f;
    for (int k = 0; k < kernel_size - 1; k++) {
        float s = conv_state[c * (kernel_size - 1) + k];
        acc += s * weight[c * kernel_size + k];
    }
    acc += new_col[c] * weight[c * kernel_size + (kernel_size - 1)];

    // SiLU activation
    float silu = acc / (1.0f + expf(-acc));
    output[c] = silu;

    // Update sliding window: shift left by 1, append new_col
    for (int k = 0; k < kernel_size - 2; k++) {
        conv_state[c * (kernel_size - 1) + k] =
            conv_state[c * (kernel_size - 1) + k + 1];
    }
    conv_state[c * (kernel_size - 1) + (kernel_size - 2)] = new_col[c];
}
```

This fuses 9 Candle ops into 1 kernel launch. Conv state is updated in-place
(no more Tensor::cat + narrow + contiguous).

**Conv state goes raw:** The conv_state must transition from `Tensor` to
`CudaSlice<f32>`. This requires changing `GdnRecurrentState.conv_states` from
`Vec<Tensor>` to `Vec<CudaSlice<f32>>` (or adding a parallel raw field).

**Candle ops eliminated:** 9

---

### Op 8: Split QKV -> Q, K, V (conv_activated -> q_raw, k_raw, v_raw)

**Candle ops:**
```rust
let q_raw = conv_activated.narrow(1, 0, key_dim)?;              // view
let k_raw = conv_activated.narrow(1, key_dim, key_dim)?;        // view
let v_raw = conv_activated.narrow(1, key_dim * 2, value_dim)?;  // view
```

**Shapes:**
- conv_activated: [1, 8192]
- q_raw: [1, 2048] (offset 0)
- k_raw: [1, 2048] (offset 2048)
- v_raw: [1, 4096] (offset 4096)

**Raw equivalent:** Pure pointer arithmetic on CudaSlice. Use `.slice(offset..offset+len)` to
get a `CudaView<f32>`. Zero cost, zero allocation.

```rust
let q_view = conv_out_slice.slice(0..key_dim);
let k_view = conv_out_slice.slice(key_dim..key_dim*2);
let v_view = conv_out_slice.slice(key_dim*2..conv_channels);
```

**Candle ops eliminated:** 3 (narrow is a view, but still creates a new Tensor struct)

---

### Op 9: L2 Norm on Q and K (q_raw, k_raw -> q_normed, k_normed)

**Candle ops:**
```rust
let q_3d = q_raw.reshape((1, n_group, d_state))?;     // 1 op
let k_3d = k_raw.reshape((1, n_group, d_state))?;     // 1 op
let q_normed = l2_norm(&q_3d, eps)?;                   // 4 ops (sqr + sum_keepdim + add + sqrt + broadcast_div)
let k_normed = l2_norm(&k_3d, eps)?;                   // 4 ops
```
Total: ~10 Candle ops

**Shapes:**
- q_raw: [2048] flat -> [16, 128] logically (n_group x d_state)
- k_raw: [2048] flat -> [16, 128] logically
- q_normed: [16, 128]
- k_normed: [16, 128]

**Raw equivalent needed:** `raw_l2_norm_groups()` -- **DOES NOT EXIST. MUST BE CREATED.**

```cuda
// Grid: (n_group, 1, 1)  Block: (128, 1, 1)  (one block per group)
// Shared memory: 128 floats for parallel reduction
extern "C" __global__ void l2_norm_groups_kernel(
    const float* __restrict__ input,   // [n_group * d_state]
    float* __restrict__ output,        // [n_group * d_state]
    int d_state,
    float eps
) {
    const int g = blockIdx.x;
    const int j = threadIdx.x;
    extern __shared__ float sdata[];

    int idx = g * d_state + j;
    float val = (j < d_state) ? input[idx] : 0.0f;
    sdata[j] = val * val;
    __syncthreads();

    // Parallel reduction for L2 norm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (j < s) sdata[j] += sdata[j + s];
        __syncthreads();
    }

    if (j < d_state) {
        float norm = sqrtf(sdata[0] + eps);
        output[idx] = val / norm;
    }
}
```

Since we need this for both Q and K, we can launch once with 2*n_group = 32 blocks
on a concatenated buffer, or launch twice on the respective slices. Two launches is
simpler and the overhead is minimal (1 kernel launch = ~5 us on sm_120).

**Candle ops eliminated:** 10

---

### Op 10: Head Expansion (q_normed, k_normed -> q_expanded, k_expanded)

**Candle ops:**
```rust
let repeats = dt_rank / n_group;  // = 2
let q_expanded = q_normed.repeat(&[1, repeats, 1])?;  // 1 op (creates [1, 32, 128])
let k_expanded = k_normed.repeat(&[1, repeats, 1])?;  // 1 op
```

**Shapes:**
- q_normed: [16, 128] -> q_expanded: [32, 128]
- k_normed: [16, 128] -> k_expanded: [32, 128]

**Raw equivalent:** This can be handled two ways:

**Option A: Explicit repeat kernel.** Simple but adds 2 kernel launches.

**Option B: Implicit in DeltaNet kernel.** The DeltaNet step kernel currently
reads `q[h * D + j]` and `k[h * D + j]` where h = 0..dt_rank (32 heads) and
D = d_state (128). If we pass the un-expanded [16, 128] arrays and add a
`head_map` parameter (or hardcode repeats=2), the kernel can read
`q[(h / repeats) * D + j]` instead. This eliminates the expansion entirely.

**Option C: Fuse into l2_norm output.** Write the l2_norm output with
interleaved repeats directly: output[h*D+j] = normed[(h/repeats)*D+j].
This is a trivial modification to the l2_norm kernel.

**Recommended:** Option B (implicit in DeltaNet kernel) for zero cost.
Failing that, Option C (fuse into l2_norm) to avoid a separate kernel.

**Candle ops eliminated:** 2

---

### Op 11: Scale Q (q_expanded -> q_scaled)

**Candle ops:**
```rust
let scale = 1.0 / (d_state as f64).sqrt();  // = 1/sqrt(128) = 0.08838...
let q_scaled = (&q_expanded * scale)?;       // 1 op
```

**Shapes:** [32, 128] -> [32, 128]

**Raw equivalent:** Fuse into the l2_norm kernel. After L2 normalization,
multiply by `scale`. Or fuse into the DeltaNet kernel (pass `scale` as a
parameter and apply `q[i] *= scale` when loading q into shared memory).

**Recommended:** Fuse into the DeltaNet kernel. The kernel already loads
`q_s[j] = q[h * D + j]` into shared memory -- just multiply by scale there.

**Candle ops eliminated:** 1

---

### Op 12: DeltaNet Step (q_scaled, k_expanded, v, gate_exp, beta, state -> new_state, output)

**Candle ops with CHIMERE_NO_FUSED unset (fused kernel):**
`deltanet_step_fused()` in `kernels/deltanet_step.rs` -- already a custom CUDA
kernel. But it currently:
1. Validates dtypes (cheap)
2. Calls `.contiguous()` on 6 inputs (6 potential Tensor allocations)
3. Extracts CudaViews from all inputs (6x `storage_and_layout`)
4. Allocates 2 new CudaSlices (s_out, output)
5. Launches kernel
6. Wraps results as Tensors (2x `Tensor::from_storage`)

**Shapes:**
- old_state: [1, 32, 128, 128] (S^T convention)
- q: [1, 32, 128]
- k: [1, 32, 128]
- v: [1, 32, 128]
- gate_exp: [1, 32]
- beta: [1, 32]
- new_state: [1, 32, 128, 128]
- output: [1, 32, 128]

**Raw equivalent needed:** `raw_deltanet_step()` -- same CUDA kernel, but:
- Takes CudaSlice/CudaView inputs directly (no Tensor unwrapping)
- Writes to pre-allocated output buffers (no allocation)
- Writes new_state in-place to the existing state buffer (the DeltaNet kernel
  already supports separate s_in and s_out pointers; just point s_out to the
  pre-allocated slot)

The gdn_states must transition from `Vec<Tensor>` to using CudaSlice.
We need two state buffers per GDN layer (double-buffering) since the kernel
reads s_in while writing s_out. Or use a single shared staging buffer since
only one layer runs at a time.

**Candle ops eliminated:** ~10 (6 contiguous + 2 alloc + 2 wrap)

---

### Op 13: Fused RMS Norm + SiLU Gate (output, ssm_norm, z -> gated)

**Candle ops with CHIMERE_FUSED_ELEM=1:**
`fused_rms_norm_silu_gate_tensor()` -- already a custom CUDA kernel, but
the `_tensor` variant creates Tensors for output.

**Shapes:**
- output: [1, 32, 128] (reshaped from DeltaNet output)
- ssm_norm: [128] (weight, persistent)
- z: [1, 4096] (gate from projection)
- gated: [1, 4096]

**Raw equivalent needed:** `raw_fused_rms_norm_silu_gate()` -- same kernel
(`fused_rms_norm_silu_gate` in `FUSED_GDN_KERNEL_SRC`), but taking CudaSlice
inputs/outputs.

**Candle ops eliminated:** ~3 (contiguous + alloc + wrap)

---

### Op 14: SSM Output Projection (gated -> projected)

**Candle ops:** `gemv_q5k_or_qmm(ssm_out_raw, ssm_out, gated, hidden_size, value_dim)`

**Shapes:**
- Input: `gated` [1, 4096]
- Weight: Q5_K [2048, 4096]
- Output: `projected` [1, 2048]

**Raw equivalent:** Same strategy as Op 2 -- keep QMatMul, extract CudaSlice
from result, copy into pre-allocated buffer.

**Candle ops eliminated:** 1

---

### Op 15: Residual Add (hidden + projected -> h_mid)

**Candle ops:** `(hidden + &projected)?` -- creates new Tensor.

**Shapes:** [1, 2048] + [1, 2048] -> [1, 2048]

**Raw equivalent needed:** `raw_add()` -- the `add_kernel` already exists in the
CUDA source (`elementwise.rs` line 103) but has no Rust wrapper. Must be added.

For the residual connection, the result should be written to the `hidden` buffer
itself (in-place add): `hidden[i] += projected[i]`. This is just `raw_weighted_add`
with weight=1.0, which **already exists**.

**Candle ops eliminated:** 1

---

## Summary: Operation Count

| Op | Description | Candle Ops | Raw Kernel | Status |
|---|---|---|---|---|
| 1 | RMS norm | 1 | raw_rms_norm | EXISTS |
| 2-3 | QKV + Gate projection | 4 | QMatMul + extract | EXTRACT ONLY |
| 4-5 | Beta + Alpha projection | 2 | QMatMul + extract | EXTRACT ONLY |
| 6 | Fused beta/alpha/gate | 2-10 | raw_fused_beta_alpha_gate | NEEDS RAW WRAPPER |
| 7 | Conv1d + SiLU | 9 | raw_conv1d_silu | MUST CREATE |
| 8 | Split QKV | 3 | CudaSlice::slice | FREE (ptr arith) |
| 9 | L2 norm (Q, K) | 10 | raw_l2_norm_groups | MUST CREATE |
| 10 | Head expansion | 2 | Fuse into DeltaNet | FUSE |
| 11 | Scale Q | 1 | Fuse into DeltaNet | FUSE |
| 12 | DeltaNet step | 10 | raw_deltanet_step | NEEDS RAW WRAPPER |
| 13 | RMS norm + SiLU gate | 3 | raw_fused_rms_norm_silu_gate | NEEDS RAW WRAPPER |
| 14 | SSM out projection | 1 | QMatMul + extract | EXTRACT ONLY |
| 15 | Residual add | 1 | raw_weighted_add (w=1.0) | EXISTS |
| **Total** | | **~49-57** | | |

**Dispatch savings:** ~49-57 Candle Tensor operations per layer x 30 GDN layers
= 1470-1710 ops x ~2 us = **2.9-3.4 ms saved**.

(This exceeds the initial estimate of 1.8 ms because the initial estimate
counted ~30 ops/layer; the actual count is closer to 50.)

---

## Pre-Allocated Buffers

### Per-Model Buffers (shared across all layers)

```rust
pub struct RawSsmBuffers {
    // --- Intermediate activations (reused every layer) ---
    /// RMS-normed hidden state. [hidden_size] = [2048]
    pub normed: CudaSlice<f32>,

    /// QKV projection output. [conv_channels] = [8192]
    pub qkv: CudaSlice<f32>,

    /// Gate (z) projection output. [value_dim] = [4096]
    pub gate: CudaSlice<f32>,

    /// Beta projection output (before sigmoid). [dt_rank] = [32]
    pub beta_proj: CudaSlice<f32>,

    /// Alpha projection output (before softplus). [dt_rank] = [32]
    pub alpha_proj: CudaSlice<f32>,

    /// Sigmoid(beta) output. [dt_rank] = [32]
    pub beta: CudaSlice<f32>,

    /// exp(softplus(alpha + bias) * ssm_a) output. [dt_rank] = [32]
    pub gate_exp: CudaSlice<f32>,

    /// Conv1d + SiLU output. [conv_channels] = [8192]
    pub conv_out: CudaSlice<f32>,

    /// L2-normed Q output (expanded to dt_rank heads). [dt_rank * d_state] = [32 * 128] = [4096]
    pub q_normed: CudaSlice<f32>,

    /// L2-normed K output (expanded to dt_rank heads). [dt_rank * d_state] = [4096]
    pub k_normed: CudaSlice<f32>,

    /// DeltaNet step output. [dt_rank * d_state] = [32 * 128] = [4096]
    pub ssm_out: CudaSlice<f32>,

    /// DeltaNet new state staging buffer.
    /// [dt_rank * d_state * d_state] = [32 * 128 * 128] = [524288]
    /// Used as s_out by the DeltaNet kernel while s_in reads from the
    /// persistent state. After the kernel, swap pointers.
    pub state_staging: CudaSlice<f32>,

    /// Fused RMS-norm + SiLU(gate) output. [value_dim] = [4096]
    pub gated_out: CudaSlice<f32>,

    /// SSM output projection result. [hidden_size] = [2048]
    pub projected: CudaSlice<f32>,
}
```

### Per-Layer Persistent State (raw versions)

```rust
/// Raw conv state per GDN layer. [conv_channels * (conv_kernel - 1)]
/// = [8192 * 3] = [24576] floats = 96 KB per layer.
/// 30 layers = 2.88 MB total.
pub raw_conv_states: Vec<CudaSlice<f32>>,

/// Raw GDN recurrent state per GDN layer.
/// [dt_rank * d_state * d_state] = [32 * 128 * 128] = [524288] floats = 2 MB per layer.
/// 30 layers = 60 MB total (same as existing Tensor-based state).
pub raw_gdn_states: Vec<CudaSlice<f32>>,
```

### Per-Layer Persistent Weights (CudaSlice extracted at load time)

```rust
/// Weight CudaSlices extracted from Tensors at model load time.
/// These are read-only views into the existing weight memory -- no extra allocation.
pub struct RawSsmWeights {
    /// RMS norm weight for pre-SSM norm. CudaSlice<f32>, [2048].
    pub attn_norm: CudaSlice<f32>,

    /// Conv1d depthwise weight. CudaSlice<f32>, [8192 * 4] = [32768].
    /// NOTE: current Tensor shape is [8192, 4] -- need to verify contiguity
    /// matches the raw kernel's expected layout [channel, kernel_size].
    pub conv1d_weight: CudaSlice<f32>,

    /// SSM norm weight (for output norm). CudaSlice<f32>, [128].
    pub ssm_norm: CudaSlice<f32>,

    /// dt_bias. CudaSlice<f32>, [32].
    pub dt_bias: CudaSlice<f32>,

    /// ssm_a (log gate parameter). CudaSlice<f32>, [32].
    pub ssm_a: CudaSlice<f32>,
}
```

### Memory Budget

| Buffer | Size (floats) | Size (bytes) | Count | Total |
|---|---|---|---|---|
| normed | 2048 | 8 KB | 1 | 8 KB |
| qkv | 8192 | 32 KB | 1 | 32 KB |
| gate | 4096 | 16 KB | 1 | 16 KB |
| beta_proj | 32 | 128 B | 1 | 128 B |
| alpha_proj | 32 | 128 B | 1 | 128 B |
| beta | 32 | 128 B | 1 | 128 B |
| gate_exp | 32 | 128 B | 1 | 128 B |
| conv_out | 8192 | 32 KB | 1 | 32 KB |
| q_normed | 4096 | 16 KB | 1 | 16 KB |
| k_normed | 4096 | 16 KB | 1 | 16 KB |
| ssm_out | 4096 | 16 KB | 1 | 16 KB |
| state_staging | 524288 | 2 MB | 1 | 2 MB |
| gated_out | 4096 | 16 KB | 1 | 16 KB |
| projected | 2048 | 8 KB | 1 | 8 KB |
| **Scratch total** | | | | **~2.16 MB** |
| raw_conv_states | 24576 | 96 KB | 30 | 2.88 MB |
| raw_gdn_states | 524288 | 2 MB | 30 | 60 MB |
| **Persistent total** | | | | **~62.88 MB** |

The persistent state is the same size as the existing Tensor-based state (it just
removes the Tensor wrapper overhead). The scratch buffers add ~2.16 MB, negligible.

---

## New Kernels Required

### 1. `raw_conv1d_silu` (CRITICAL -- largest dispatch cluster)

Fuses: reshape + cat + narrow + contiguous + squeeze + mul + sum + unsqueeze + silu
= 9 Candle ops into 1 kernel launch.

Also handles conv state update in-place (eliminates cat + narrow + contiguous for
state management = 3 more ops saved in state bookkeeping).

```
Input:  conv_state [conv_channels * (K-1)], new_col [conv_channels],
        weight [conv_channels * K]
Output: activated [conv_channels]
Side effect: conv_state updated in-place (shift left, append new_col)

Grid: (ceil(8192/256), 1, 1) = (32, 1, 1)
Block: (256, 1, 1)
Shared mem: 0
```

### 2. `raw_l2_norm_groups` (second largest cluster)

Fuses: reshape + sqr + sum_keepdim + add_eps + sqrt + broadcast_div
= 5 ops per call, 2 calls (Q and K) = 10 ops.

Can optionally fuse head expansion (repeat x2) into the output write,
eliminating 2 more ops.

```
Input:  data [n_group * d_state] = [2048]
Output: normed [dt_rank * d_state] = [4096] (with expansion if fused)

Grid: (dt_rank, 1, 1) = (32, 1, 1)   [with expansion]
  or: (n_group, 1, 1) = (16, 1, 1)   [without expansion]
Block: (128, 1, 1)  or (256, 1, 1) for reduction padding
Shared mem: 128 * 4 = 512 bytes
```

If fusing expansion, each block reads from group `g = blockIdx.x / repeats`
and writes to head `blockIdx.x`. Two blocks read the same group.

### 3. `raw_fused_beta_alpha_gate` (wrapper only -- kernel exists)

Same CUDA kernel as `fused_beta_alpha_gate` in `FUSED_GDN_KERNEL_SRC`.
New Rust wrapper that takes `CudaSlice<f32>` inputs and writes to
pre-allocated `CudaSlice<f32>` outputs.

### 4. `raw_fused_rms_norm_silu_gate` (wrapper only -- kernel exists)

Same CUDA kernel as `fused_rms_norm_silu_gate` in `FUSED_GDN_KERNEL_SRC`.
New Rust wrapper taking CudaSlice inputs/outputs.

### 5. `raw_deltanet_step` (wrapper only -- kernel exists)

Same CUDA kernel as `deltanet_step_kernel` in `deltanet_step.rs`.
New Rust wrapper that:
- Takes CudaSlice<f32> for state, q, k, v, gate, beta
- Writes to pre-allocated s_out and output CudaSlices
- No Tensor allocation or wrapping

### 6. `raw_add` (wrapper only -- kernel exists)

The `add_kernel` exists in the elementwise CUDA source. Just needs a Rust
`pub fn raw_add(a, b, output, n, dev)` wrapper. (Or reuse `raw_weighted_add`
with weight=1.0.)

---

## Existing Kernels Reused

| Kernel | File | Purpose |
|---|---|---|
| `raw_rms_norm` | elementwise.rs:323 | Pre-SSM norm (Op 1) |
| `raw_weighted_add` | elementwise.rs:405 | Residual add (Op 15) |
| `fused_beta_alpha_gate` (CUDA) | elementwise.rs:249 | Op 6 (needs raw wrapper) |
| `fused_rms_norm_silu_gate` (CUDA) | elementwise.rs:266 | Op 13 (needs raw wrapper) |
| `deltanet_step_kernel` (CUDA) | deltanet_step.rs:43 | Op 12 (needs raw wrapper) |

---

## Phased Implementation Plan

### Phase 1: Conv1d (highest impact, ~9 ops x 30 = 270 ops eliminated)

1. Write `conv1d_silu_kernel` CUDA source in `elementwise.rs`
2. Add `raw_conv1d_silu()` Rust wrapper
3. Convert conv_states to raw CudaSlice (parallel field in GdnRecurrentState,
   or new RawSsmState struct)
4. Wire into `forward_gdn_layer_moe` behind `CHIMERE_RAW_SSM=1` env toggle
5. Test: compare output numerically against reference path

**Estimated savings:** 270 ops x 2 us = **0.54 ms**

### Phase 2: L2 Norm + Head Expansion (second highest, ~12 ops x 30 = 360 ops)

1. Write `l2_norm_groups_kernel` CUDA source with optional head expansion
2. Add `raw_l2_norm_groups()` Rust wrapper
3. Wire in: after conv1d, split QKV via pointer arithmetic, run raw l2_norm
4. Fuse q_scale into the norm output (multiply by `1/sqrt(d_state)` inline)

**Estimated savings:** 360 ops x 2 us = **0.72 ms**

### Phase 3: Raw wrappers for existing kernels (~15 ops x 30 = 450 ops)

1. `raw_fused_beta_alpha_gate()` -- thin wrapper around existing CUDA kernel
2. `raw_fused_rms_norm_silu_gate()` -- thin wrapper
3. `raw_deltanet_step()` -- thin wrapper, convert gdn_states to CudaSlice
4. `raw_add()` or reuse `raw_weighted_add(w=1.0)` for residual

**Estimated savings:** 450 ops x 2 us = **0.90 ms**

### Phase 4: QMatMul extraction bridge (~8 ops x 30 = 240 ops)

1. For each QMatMul call (qkv, gate, beta, alpha, ssm_out), immediately
   extract the CudaSlice from the result Tensor and `dtod_copy` into the
   pre-allocated buffer
2. Drop the Tensor -- no further Candle Tensor refs in the SSM path
3. This is the "seam" between Candle (weight dequant/GEMV) and raw (everything else)

**Estimated savings:** 240 ops x 2 us = **0.48 ms**

### Phase 5: Full integration + cleanup

1. Remove all intermediate Tensor variables from the SSM path
2. Consolidate `RawSsmBuffers` into `RawGpuBuffers` (merge with MoE buffers)
3. Add CHIMERE_RAW_SSM env toggle (default on, CHIMERE_NO_RAW_SSM=1 to disable)
4. Update snapshot/restore to handle raw state
5. Benchmark: expected total savings **2.6-3.4 ms/token**

---

## Risk Assessment

| Risk | Mitigation |
|---|---|
| Conv state layout mismatch | Verify `ssm_conv1d` Tensor is [channels, kernel] contiguous before extracting CudaSlice. Print strides at load time. |
| DeltaNet state double-buffering | Use single staging buffer + swap. Only one layer active at a time, so one staging buffer suffices for all 30 layers. |
| Numerical divergence | Run both paths in parallel for first N tokens, compare max abs error. The raw path uses the exact same CUDA kernels, so divergence should be zero (same rounding). |
| Snapshot/restore breaks | Keep Tensor-based state as fallback. Raw state can be wrapped into Tensor on demand for snapshot (costs ~2 us, only during snapshot). |
| QMatMul output extraction | The QMatMul forward always returns a contiguous F32 Tensor on the same device. Extraction via `storage_and_layout` is safe. The only risk is if QMatMul returns a non-contiguous view (it does not). |

---

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| Candle Tensor ops per SSM layer | ~50 | ~5 (QMatMul outputs only) |
| Candle Tensor ops per token (30 GDN layers) | ~1500 | ~150 |
| SSM dispatch overhead | ~3.0 ms | ~0.3 ms |
| Net savings | -- | **~2.7 ms/token** |
| Token throughput impact (from 26.6 ms/tok) | 37.6 tok/s | ~41.8 tok/s (+11%) |

Combined with the existing raw MoE path, total Candle Tensor ops per token
would drop from ~3500 to ~500 (MoE already raw + SSM raw = only attention
layers + lm_head + embedding still use Candle).
