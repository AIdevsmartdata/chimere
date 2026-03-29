# Raw Forward v4 — Complete Design Document

## Goal

Eliminate ALL Candle Tensor overhead from the token generation hot path.

| Metric | Current (v3) | Target (v4) | ik_llama ref |
|--------|-------------|-------------|--------------|
| tok/s | 59 | ~100 | 89.7 |
| ms/token | 16.8 | ~10 | 11.2 |
| Tensor creates/tok | 712 | 0 | N/A |
| cudaMalloc/tok | ~200 | 0 | 0 |

The overhead breakdown (measured via CHIMERE_GRAN_PROF):
- 712 Tensor creations (Arc::new + Storage::Cuda + Shape): ~2ms
- 712 Tensor drops (Arc::drop + cudaFree sync): ~3ms
- ~200 cudaMalloc per token: ~1ms
- HashMap lookups for kernel cache: ~0.5ms
- storage_and_layout() extraction boilerplate: ~0.5ms
- **Total Candle overhead: ~7ms/token (42% of 16.8ms)**

---

## 1. Complete Buffer Layout

All dimensions are for **Qwen3.5-35B-A3B** (the production model):
- hidden_size = 2048
- num_attention_heads = 16, num_kv_heads = 2, head_dim = 256
- ssm_d_state = 128, ssm_d_inner = 4096, ssm_dt_rank = 32, ssm_n_group = 16
- ssm_conv_kernel = 4
- num_experts = 256, experts_per_token = 8, expert_ffn_hidden = 512
- vocab_size = 248320
- num_main_layers = 40 (30 GDN + 10 ATTN)
- key_dim = n_group * d_state = 2048
- value_dim = dt_rank * d_state = 4096
- conv_channels = key_dim * 2 + value_dim = 8192
- n_rot = 64, q_head_dim = 512 (Q+gate interleaved: 16 heads * 2 * 256 = 8192)

### 1.1 Permanent State Buffers (live for entire generation)

These are allocated once and persist across all tokens.

#### GDN Recurrent State (30 layers)
```
gdn_states[30]:    CudaSlice<f32> [dt_rank * d_state * d_state]  = [32 * 128 * 128] = 524,288 floats = 2 MB each
                   Total: 30 * 2 MB = 60 MB
gdn_states_tmp[30]: CudaSlice<f32> [same] — double-buffered for in-place update
                   Total: 30 * 2 MB = 60 MB
```
**Note:** deltanet_step_kernel writes to s_out (separate from s_in). We need two
buffers per layer and swap pointers after each step. Alternative: single buffer
with the kernel reading+writing the same buffer (requires kernel modification
to use registers as shown in the existing kernel). The current kernel already
reads into s_reg[] before writing, so a single buffer works IF we modify the
kernel to read/write the same pointer. **Decision: use single buffer + modified
kernel that operates in-place**, saving 60 MB. The kernel already loads the
full column into `s_reg[128]` before writing back. Just pass `s_in == s_out`.

Revised:
```
gdn_states[30]:    CudaSlice<f32> [524,288]   = 2 MB each, total 60 MB
```

#### Conv1d Sliding Window State (30 GDN layers)
```
conv_states[30]:   CudaSlice<f32> [conv_channels * (conv_kernel-1)]
                   = [8192 * 3] = 24,576 floats = 96 KB each
                   Total: 30 * 96 KB = 2.88 MB
```

#### KV Cache (10 attention layers) — Ring Buffer
```
kv_k_bufs[10]:     CudaSlice<f32> [num_kv_heads * max_seq * head_dim]
                   = [2 * 4096 * 256] = 2,097,152 floats = 8 MB each (initial cap=4096)
kv_v_bufs[10]:     CudaSlice<f32> [same]
                   Total: 10 * 2 * 8 MB = 160 MB (initial, grows with seq_len)
kv_lens[10]:       usize — current length per layer
kv_caps[10]:       usize — current capacity per layer
```

**Total permanent: ~223 MB** (dominated by KV cache, same as current Candle path)

### 1.2 Per-Layer Scratch Buffers (reused every layer)

All scratch buffers are allocated once and reused across layers within a token.

#### Shared activation buffers
```
hidden:            CudaSlice<f32> [2048]           = 8 KB
hidden_residual:   CudaSlice<f32> [2048]           = 8 KB
normed:            CudaSlice<f32> [2048]           = 8 KB
```

#### GDN SSM scratch (already exists as GdnScratchBuffers)
```
conv_output:       CudaSlice<f32> [8192]           = 32 KB   — conv1d+silu output
qkv_copy:          CudaSlice<f32> [8192]           = 32 KB   — QKV for conv window
cw_copy:           CudaSlice<f32> [8192 * 4]       = 128 KB  — conv window concatenated
new_conv_state:    CudaSlice<f32> [8192 * 3]       = 96 KB   — updated conv state
q_split:           CudaSlice<f32> [2048]           = 8 KB
k_split:           CudaSlice<f32> [2048]           = 8 KB
q_normed:          CudaSlice<f32> [2048]           = 8 KB
k_normed:          CudaSlice<f32> [2048]           = 8 KB
q_expanded:        CudaSlice<f32> [4096]           = 16 KB
k_expanded:        CudaSlice<f32> [4096]           = 16 KB
q_scaled:          CudaSlice<f32> [4096]           = 16 KB
v_copy:            CudaSlice<f32> [4096]           = 16 KB
```

#### SSM projection scratch
```
qkv_proj_out:      CudaSlice<f32> [8192]           = 32 KB   — attn_qkv output
gate_proj_out:     CudaSlice<f32> [4096]           = 16 KB   — attn_gate (z) output
beta_out:          CudaSlice<f32> [32]             = 128 B   — sigmoid(ssm_beta proj)
alpha_out:         CudaSlice<f32> [32]             = 128 B   — ssm_alpha proj
gate_exp_out:      CudaSlice<f32> [32]             = 128 B   — exp(softplus(alpha+bias)*a)
ssm_out_proj:      CudaSlice<f32> [2048]           = 8 KB    — ssm_out projection result
normed_gated:      CudaSlice<f32> [4096]           = 16 KB   — rms_norm(output) * silu(z)
deltanet_output:   CudaSlice<f32> [4096]           = 16 KB   — deltanet step output
```

#### Q8_1 quantization scratch (reused per projection)
```
q8_input:          CudaSlice<u8> [pad(4096,512)*36/32] = 4,608 B — Q8_1 of input vec
                   (max input dim = 4096 for ssm_out path)
```

#### Attention layer scratch
```
q_proj_out:        CudaSlice<f32> [8192]           = 32 KB   — Q+gate interleaved
k_proj_out:        CudaSlice<f32> [512]            = 2 KB    — K projection
v_proj_out:        CudaSlice<f32> [512]            = 2 KB    — V projection
q_heads:           CudaSlice<f32> [16 * 256]       = 16 KB   — Q after reshape
q_gate:            CudaSlice<f32> [16 * 256]       = 16 KB   — gate after reshape
q_normed_attn:     CudaSlice<f32> [16 * 256]       = 16 KB   — QK-normed Q
k_normed_attn:     CudaSlice<f32> [2 * 256]        = 2 KB    — QK-normed K
q_rotated:         CudaSlice<f32> [16 * 256]       = 16 KB   — after MRoPE
k_rotated:         CudaSlice<f32> [2 * 256]        = 2 KB    — after MRoPE
attn_scores:       CudaSlice<f32> [16 * max_seq]   = 64 KB (at 4096 seq)
attn_output:       CudaSlice<f32> [16 * 256]       = 16 KB
gated_output:      CudaSlice<f32> [16 * 256]       = 16 KB
wo_output:         CudaSlice<f32> [2048]           = 8 KB
```

#### MoE FFN scratch (already exists as RawGpuBuffers subset)
```
router_logits:     CudaSlice<f32> [256]            = 1 KB
topk_indices:      CudaSlice<i32> [8]              = 32 B
topk_weights:      CudaSlice<f32> [8]              = 32 B
q8_hidden:         CudaSlice<u8>  [2304]           = 2.3 KB
batched_gate_out:  CudaSlice<f32> [8*512]          = 16 KB
batched_up_out:    CudaSlice<f32> [8*512]          = 16 KB
batched_inter:     CudaSlice<f32> [8*512]          = 16 KB
batched_q8_inter:  CudaSlice<u8>  [8*576]          = 4.6 KB
batched_expert_out:CudaSlice<f32> [8*2048]         = 64 KB
combined_out:      CudaSlice<f32> [2048]           = 8 KB
```

#### Shared expert (Q5_K) scratch
```
shared_gate_out:   CudaSlice<f32> [512]            = 2 KB
shared_up_out:     CudaSlice<f32> [512]            = 2 KB
shared_inter:      CudaSlice<f32> [512]            = 2 KB
shared_down_out:   CudaSlice<f32> [2048]           = 8 KB
q8_shared_input:   CudaSlice<u8>  [2304]           = 2.3 KB  (reuses q8_hidden)
q8_shared_inter:   CudaSlice<u8>  [576]            = 576 B
```

#### LM Head scratch
```
logits_buf:        CudaSlice<f32> [248320]         = 993 KB
output_normed:     CudaSlice<f32> [2048]           = 8 KB
```

**Total scratch: ~1.7 MB** (negligible)

### 1.3 MRoPE Precomputed Tables (permanent, read-only)
```
cos_tables[3]:     CudaSlice<f32> [65536 * pairs]  — section 0: 65536*11, section 1: 65536*11, section 2: 65536*10
sin_tables[3]:     CudaSlice<f32> [same]
Total: 2 * 65536 * 32 * 4 = ~16 MB (already allocated by MRoPE, need raw pointers)
```

---

## 2. Weight Pointer Extraction

### 2.1 Current State

Currently, only 3 weights per GDN-MoE layer have raw bytes extracted:
- `attn_qkv_raw: Option<Tensor>` — flat U8 Q5_K bytes
- `attn_gate_raw: Option<Tensor>` — flat U8 Q5_K bytes
- `ssm_out_raw: Option<Tensor>` — flat U8 Q5_K bytes

All other weights go through `QMatMul::forward()` which:
1. Calls `QTensor.storage()` (Arc lock)
2. Calls `quantize_q8_1` (1 cudaMalloc for temp Q8_1 buffer)
3. Calls the GEMV kernel
4. Wraps result in new Tensor (1 cudaMalloc + Arc::new + Shape + Layout)

### 2.2 Design: RawWeights struct

At model load time (in `from_gguf`), extract `CudaSlice<u8>` from every QMatMul
that will be used in the raw forward path. Store these in a parallel struct.

```rust
/// Raw weight pointers for one GDN-MoE layer.
pub struct RawGdnWeights {
    // Norms (F32, read-only) — already CudaSlice-compatible
    attn_norm: CudaSlice<f32>,       // [2048]
    post_norm: CudaSlice<f32>,       // [2048]
    ssm_norm: CudaSlice<f32>,        // [128]

    // SSM projections — Q5_K raw bytes
    attn_qkv: CudaSlice<u8>,        // Q5_K bytes for [8192, 2048]
    attn_gate: CudaSlice<u8>,       // Q5_K bytes for [4096, 2048]
    ssm_out: CudaSlice<u8>,         // Q5_K bytes for [2048, 4096]
    ssm_beta: CudaSlice<u8>,        // Q5_K bytes for [32, 2048]
    ssm_alpha: CudaSlice<u8>,       // Q5_K bytes for [32, 2048]

    // Conv1d (F32, small)
    ssm_conv1d: CudaSlice<f32>,     // [8192, 4]

    // SSM scalars (F32, tiny)
    ssm_a: CudaSlice<f32>,          // [32]
    ssm_dt_bias: CudaSlice<f32>,    // [32]

    // MoE expert weights — IQ3_S raw bytes (already extracted)
    gate_inp_t: CudaSlice<f32>,     // [256, 2048] transposed router
    gate_exps_raw: CudaSlice<u8>,   // all 256 experts gate
    up_exps_raw: CudaSlice<u8>,     // all 256 experts up
    down_exps_raw: CudaSlice<u8>,   // all 256 experts down
    expert_bytes_gate: usize,       // byte stride per expert
    expert_bytes_up: usize,
    expert_bytes_down: usize,

    // Shared expert — Q5_K raw bytes
    gate_shexp: CudaSlice<u8>,
    up_shexp: CudaSlice<u8>,
    down_shexp: CudaSlice<u8>,

    // Gate bias for shared expert
    gate_inp_shexp: CudaSlice<f32>, // [2048]
}

/// Raw weight pointers for one attention-MoE layer.
pub struct RawAttnWeights {
    attn_norm: CudaSlice<f32>,       // [2048]
    post_norm: CudaSlice<f32>,       // [2048]
    q_norm: CudaSlice<f32>,          // [256]
    k_norm: CudaSlice<f32>,          // [256]

    // Attention projections — Q5_K raw bytes
    wq: CudaSlice<u8>,              // Q5_K bytes for [8192, 2048]
    wk: CudaSlice<u8>,              // Q5_K bytes for [512, 2048]
    wv: CudaSlice<u8>,              // Q5_K bytes for [512, 2048]
    wo: CudaSlice<u8>,              // Q5_K bytes for [2048, 4096]

    // MoE (same as GDN)
    gate_inp_t: CudaSlice<f32>,
    gate_exps_raw: CudaSlice<u8>,
    up_exps_raw: CudaSlice<u8>,
    down_exps_raw: CudaSlice<u8>,
    expert_bytes_gate: usize,
    expert_bytes_up: usize,
    expert_bytes_down: usize,
    gate_shexp: CudaSlice<u8>,
    up_shexp: CudaSlice<u8>,
    down_shexp: CudaSlice<u8>,
    gate_inp_shexp: CudaSlice<f32>,
}

/// All raw weight pointers for the full model.
pub struct RawWeights {
    embed_tokens: Tensor,            // CPU, F32 — kept as Tensor for index_select
    gdn_layers: Vec<RawGdnWeights>,  // 30 layers
    attn_layers: Vec<RawAttnWeights>, // 10 layers
    output_norm: CudaSlice<f32>,     // [2048]
    lm_head: CudaSlice<u8>,         // Q5_K bytes for [248320, 2048]
}
```

### 2.3 Extraction Method

For each `QMatMul` weight, extract raw bytes at load time:

```rust
fn extract_raw_bytes(qmm: &QMatMul, device: &CudaDevice) -> Result<CudaSlice<u8>> {
    match qmm {
        QMatMul::QTensor(qt) => {
            // QTensor already has raw quantized bytes on GPU
            // Access via qt.storage() -> CudaStorage -> as_cuda_slice::<u8>()
            let storage = qt.storage();
            // Clone the CudaSlice (just an Arc bump, no data copy)
            Ok(storage.as_cuda_slice::<u8>()?.try_clone()?)
        }
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => {
            // F32/F16 tensor — extract CudaSlice<f32> and reinterpret as u8
            // (used for small F32 tensors like norms)
            candle_core::bail!("F32/F16 weights not supported for raw extraction")
        }
    }
}
```

For F32 norm tensors, extract `CudaSlice<f32>` directly:

```rust
fn extract_f32_slice(tensor: &Tensor) -> Result<CudaSlice<f32>> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Cuda(cuda) => {
            let slice = cuda.as_cuda_slice::<f32>()?;
            let offset = layout.start_offset();
            let n = tensor.elem_count();
            // Clone the sub-slice (Arc bump + offset, no data copy)
            Ok(slice.slice(offset..offset + n).try_clone()?)
        }
        _ => candle_core::bail!("tensor not on CUDA"),
    }
}
```

**Key insight:** `CudaSlice::try_clone()` is not available in cudarc. Instead,
we use `dev.clone_dtod(src)` to get an owned copy, or we keep the original
Tensor alive and just store CudaView references. The cleanest approach:
keep the Qwen35Model alive (it owns all Tensors) and store raw pointers
(CudaView) in RawWeights that borrow from the model. Since the model lives
for the entire inference session, lifetimes are trivially satisfied.

**Alternative (preferred):** Extract raw pointers as `*const u8` / `*const f32`
at load time. These are device pointers that remain valid as long as the model
lives. No Arc overhead, no CudaSlice wrapper. The CUDA kernels just need the
device pointer anyway.

---

## 3. Raw Forward Function Signatures

```rust
/// Complete raw state for the forward pass.
pub struct RawState {
    /// GDN recurrent states: 30 layers, each [dt_rank * d_state * d_state] f32
    pub gdn_states: Vec<CudaSlice<f32>>,
    /// Conv1d sliding window: 30 layers, each [conv_channels * (conv_kernel-1)] f32
    pub conv_states: Vec<CudaSlice<f32>>,
    /// KV cache keys: 10 layers, each [num_kv_heads * capacity * head_dim] f32
    pub kv_k: Vec<CudaSlice<f32>>,
    /// KV cache values: 10 layers, same shape
    pub kv_v: Vec<CudaSlice<f32>>,
    /// Current length of each KV cache (positions filled)
    pub kv_lens: Vec<usize>,
    /// Current capacity of each KV cache buffer
    pub kv_caps: Vec<usize>,
    /// Current token position in the sequence
    pub position: usize,
}

/// Pre-allocated scratch buffers for the entire forward pass.
/// Allocated once, reused every token, never freed during inference.
pub struct RawBuffers {
    // -- Shared activation --
    pub hidden: CudaSlice<f32>,            // [2048]
    pub normed: CudaSlice<f32>,            // [2048]
    pub hidden_residual: CudaSlice<f32>,   // [2048] for saving residual

    // -- GDN SSM scratch --
    pub gdn: GdnScratchBuffers,           // existing struct, already complete

    // -- SSM projection outputs --
    pub qkv_proj: CudaSlice<f32>,          // [8192]
    pub gate_proj: CudaSlice<f32>,         // [4096]
    pub beta_proj: CudaSlice<f32>,         // [32]
    pub alpha_proj: CudaSlice<f32>,        // [32]
    pub beta_out: CudaSlice<f32>,          // [32]
    pub gate_exp: CudaSlice<f32>,          // [32]
    pub deltanet_out: CudaSlice<f32>,      // [4096]
    pub normed_gated: CudaSlice<f32>,      // [4096]
    pub ssm_out: CudaSlice<f32>,           // [2048]

    // -- Attention scratch --
    pub q_proj: CudaSlice<f32>,            // [8192] Q+gate interleaved
    pub k_proj: CudaSlice<f32>,            // [512]
    pub v_proj: CudaSlice<f32>,            // [512]
    pub q_normed_attn: CudaSlice<f32>,     // [4096] (16 * 256)
    pub k_normed_attn: CudaSlice<f32>,     // [512]  (2 * 256)
    pub q_rotated: CudaSlice<f32>,         // [4096]
    pub k_rotated: CudaSlice<f32>,         // [512]
    pub attn_scores: CudaSlice<f32>,       // [16 * MAX_SEQ_CAPACITY]
    pub attn_output: CudaSlice<f32>,       // [4096]
    pub gated_output: CudaSlice<f32>,      // [4096]
    pub wo_output: CudaSlice<f32>,         // [2048]

    // -- Q8_1 scratch --
    pub q8_input: CudaSlice<u8>,           // max(4096, 2048) padded Q8_1

    // -- MoE scratch (existing RawGpuBuffers fields) --
    pub moe: RawMoeBuffers,

    // -- Output --
    pub output_normed: CudaSlice<f32>,     // [2048]
    pub logits: CudaSlice<f32>,            // [248320]
}

/// Main entry point: raw forward pass for a single token.
/// Returns a reference to the logits buffer (no allocation).
pub fn raw_forward_token(
    token: u32,
    state: &mut RawState,
    weights: &RawWeights,
    buffers: &mut RawBuffers,
    config: &Qwen35Config,
    mrope_tables: &RawMRoPETables,
    dev: &CudaDevice,
) -> Result<&CudaSlice<f32>>   // points to buffers.logits
```

---

## 4. Per-Layer Call Sequence

### 4.1 Embedding

```
1. CPU: index_select(embed_tokens, token) -> f32[2048] on CPU
2. htod: memcpy_htod(cpu_row, &mut buffers.hidden)
   Cost: ~2us (8 KB over PCIe)
```

### 4.2 GDN Layer (30 layers)

```
 1. raw_rms_norm(&buffers.hidden, &weights.attn_norm, &mut buffers.normed, eps, dev)
 2. raw_quantize_q8_1(&buffers.normed, &mut buffers.q8_input, 2048, dev)
 3. raw_q5k_gemv_candle(&weights.attn_qkv, &buffers.q8_input, &mut buffers.qkv_proj, 2048, 8192, dev)
 4. raw_q5k_gemv_candle(&weights.attn_gate, &buffers.q8_input, &mut buffers.gate_proj, 2048, 4096, dev)
    // q8_input still valid (same normed), reused for steps 3-4
 5. raw_q5k_gemv_candle(&weights.ssm_beta, &buffers.q8_input, &mut buffers.beta_proj, 2048, 32, dev)
 6. raw_q5k_gemv_candle(&weights.ssm_alpha, &buffers.q8_input, &mut buffers.alpha_proj, 2048, 32, dev)
    // 4 projections, 1 Q8_1 quantization = 5 kernel launches
 7. fused_beta_alpha_gate_raw(&buffers.beta_proj, &buffers.alpha_proj,
        &weights.ssm_dt_bias, &weights.ssm_a, &mut buffers.beta_out, &mut buffers.gate_exp, dev)
    // 1 kernel: sigmoid(beta) + exp(softplus(alpha+bias)*a)
 8. raw_fused_conv1d_silu_update(&buffers.qkv_proj, &state.conv_states[gdn_idx],
        &weights.ssm_conv1d, &mut buffers.gdn.conv_output,
        &mut state.conv_states[gdn_idx], conv_channels, conv_kernel, dev)
    // 1 kernel: conv1d + silu + state update
 9. raw_split_l2norm_expand(&buffers.gdn.conv_output,
        &mut buffers.gdn.q_scaled, &mut buffers.gdn.k_expanded, &mut buffers.gdn.v_copy,
        key_dim, value_dim, n_group, d_state, dt_rank, eps, dev)
    // 1-3 kernels: split QKV, L2 norm, expand groups, scale q
10. deltanet_step_fused_raw(
        &state.gdn_states[gdn_idx],  // s_in (read)
        &buffers.gdn.q_scaled, &buffers.gdn.k_expanded, &buffers.gdn.v_copy,
        &buffers.gate_exp, &buffers.beta_out,
        &mut state.gdn_states[gdn_idx],  // s_out (SAME as s_in — in-place)
        &mut buffers.deltanet_out,
        dt_rank, d_state, dev)
    // 1 kernel: full deltanet state update
11. raw_rms_norm_silu_gate(&buffers.deltanet_out, &weights.ssm_norm, &buffers.gate_proj,
        &mut buffers.normed_gated, d_state, dt_rank, eps, dev)
    // 1 fused kernel: rms_norm(output) * silu(z)
12. raw_quantize_q8_1(&buffers.normed_gated, &mut buffers.q8_input, 4096, dev)
13. raw_q5k_gemv_candle(&weights.ssm_out, &buffers.q8_input, &mut buffers.ssm_out, 4096, 2048, dev)
    // ssm_out projection
14. raw_add_inplace(&mut buffers.hidden, &buffers.ssm_out, 2048, dev)
    // residual: hidden += ssm_out
15. memcpy_dtod(&buffers.hidden, &mut buffers.hidden_residual)
    // save residual for MoE FFN
16. raw_rms_norm(&buffers.hidden, &weights.post_norm, &mut buffers.normed, eps, dev)
17. [MoE FFN — see 4.4]
18. raw_add_inplace(&mut buffers.hidden, &buffers.moe.combined_out, 2048, dev)
    // residual: hidden += ffn_out
```

**Total kernel launches per GDN layer: ~15** (vs ~60+ Candle dispatch calls currently)

### 4.3 Attention Layer (10 layers)

```
 1. raw_rms_norm(&buffers.hidden, &weights.attn_norm, &mut buffers.normed, eps, dev)
 2. raw_quantize_q8_1(&buffers.normed, &mut buffers.q8_input, 2048, dev)
 3. raw_q5k_gemv_candle(&weights.wq, &buffers.q8_input, &mut buffers.q_proj, 2048, 8192, dev)
 4. raw_q5k_gemv_candle(&weights.wk, &buffers.q8_input, &mut buffers.k_proj, 2048, 512, dev)
 5. raw_q5k_gemv_candle(&weights.wv, &buffers.q8_input, &mut buffers.v_proj, 2048, 512, dev)
    // 3 projections, 1 Q8_1 quantization
 6. raw_deinterleave_q_gate(&buffers.q_proj, &mut buffers.q_normed_attn,
        &mut buffers.gated_output, num_heads, head_dim)
    // Split Q+gate interleaved into separate Q and gate buffers
 7. raw_rms_norm_per_head(&buffers.q_normed_attn, &weights.q_norm,
        &mut buffers.q_normed_attn, num_heads, head_dim, eps, dev)
 8. raw_rms_norm_per_head(&buffers.k_proj, &weights.k_norm,
        &mut buffers.k_normed_attn, num_kv_heads, head_dim, eps, dev)
    // Per-head QK norm (2 kernel launches)
 9. raw_mrope_apply(&buffers.q_normed_attn, &mut buffers.q_rotated,
        mrope_tables, state.position, num_heads, head_dim, n_rot, sections, dev)
10. raw_mrope_apply(&buffers.k_normed_attn, &mut buffers.k_rotated,
        mrope_tables, state.position, num_kv_heads, head_dim, n_rot, sections, dev)
    // 2 kernel launches for MRoPE
11. raw_kv_append(&buffers.k_rotated, &buffers.v_proj,
        &mut state.kv_k[attn_idx], &mut state.kv_v[attn_idx],
        &mut state.kv_lens[attn_idx], &mut state.kv_caps[attn_idx],
        num_kv_heads, head_dim, dev)
    // 1 memcpy_dtod (append single row to ring buffer)
12. raw_gqa_attention(&buffers.q_rotated,
        &state.kv_k[attn_idx], &state.kv_v[attn_idx], state.kv_lens[attn_idx],
        &mut buffers.attn_output,
        num_heads, num_kv_heads, head_dim, dev)
    // 1 fused GQA kernel (already exists as fused_gqa_attention)
13. raw_sigmoid_gate_mul(&buffers.attn_output, &buffers.gated_output,
        &mut buffers.gated_output, num_heads * head_dim, dev)
    // sigmoid(gate) * attn_output
14. raw_quantize_q8_1(&buffers.gated_output, &mut buffers.q8_input, 4096, dev)
15. raw_q5k_gemv_candle(&weights.wo, &buffers.q8_input, &mut buffers.wo_output, 4096, 2048, dev)
    // output projection
16. raw_add_inplace(&mut buffers.hidden, &buffers.wo_output, 2048, dev)
    // residual
17. memcpy_dtod(&buffers.hidden, &mut buffers.hidden_residual)
18. raw_rms_norm(&buffers.hidden, &weights.post_norm, &mut buffers.normed, eps, dev)
19. [MoE FFN — see 4.4]
20. raw_add_inplace(&mut buffers.hidden, &buffers.moe.combined_out, 2048, dev)
```

**Total kernel launches per attention layer: ~17** (vs ~50+ Candle dispatch currently)

### 4.4 MoE FFN (shared by both layer types)

This path is **already mostly raw** in the current codebase (`moe_ffn_forward_raw`).
The main change is eliminating the `storage_and_layout()` extraction overhead
(currently ~5 calls per MoE layer, each taking the Arc read lock).

```
 1. raw_f32_gemv(&buffers.normed, &weights.gate_inp_t, &mut buffers.moe.router_logits,
        2048, 256, dev)
    // Router: [256, 2048] x [2048] -> [256]
 2. gpu_topk_softmax(&buffers.moe.router_logits, &mut buffers.moe.topk_indices,
        &mut buffers.moe.topk_weights, 256, 8, dev)
    // GPU top-K: 1 kernel launch, 64 bytes dtoh
 3. dtoh: indices[8], weights[8] -> CPU (64 bytes, needed to compute byte offsets)
 4. raw_quantize_q8_1(&buffers.normed, &mut buffers.moe.q8_hidden, 2048, dev)
 5. gemv_iq3s_q8_batched(gate_exps_raw, q8_hidden, batched_gate_out,
        offsets_from_indices, 2048, 512, 8, dev)
 6. gemv_iq3s_q8_batched(up_exps_raw, q8_hidden, batched_up_out,
        offsets_from_indices, 2048, 512, 8, dev)
 7. raw_silu_mul_batched(&batched_gate_out, &batched_up_out, &mut batched_inter, 512*8, dev)
 8. [quantize each expert's intermediate to Q8_1 -> batched_q8_inter]
 9. gemv_iq3s_q8_batched_multi_input(down_exps_raw, batched_q8_inter, batched_expert_outs,
        offsets_from_indices, 512, 2048, 8, dev)
10. raw_weighted_combine(&batched_expert_outs, &topk_weights, &mut combined_out, 2048, 8, dev)
    // Shared expert (Q5_K path):
11. raw_q5k_gemv_candle(&weights.gate_shexp, &buffers.moe.q8_hidden, &mut shared_gate_out, 2048, 512)
12. raw_q5k_gemv_candle(&weights.up_shexp, &buffers.moe.q8_hidden, &mut shared_up_out, 2048, 512)
13. raw_silu_mul(&shared_gate_out, &shared_up_out, &mut shared_inter, 512, dev)
14. raw_quantize_q8_1(&shared_inter, &mut q8_shared_inter, 512, dev)
15. raw_q5k_gemv_candle(&weights.down_shexp, &q8_shared_inter, &mut shared_down_out, 512, 2048)
16. raw_add_inplace(&mut combined_out, &shared_down_out, 2048, dev)
```

### 4.5 LM Head

```
1. raw_rms_norm(&buffers.hidden, &weights.output_norm, &mut buffers.output_normed, eps, dev)
2. raw_quantize_q8_1(&buffers.output_normed, &mut buffers.q8_input, 2048, dev)
3. raw_q5k_gemv_candle(&weights.lm_head, &buffers.q8_input, &mut buffers.logits, 2048, 248320, dev)
   // LM head: [248320, 2048] x normed -> [248320] logits
4. raw_argmax(&buffers.logits, 248320, dev) -> u32 token  // or return logits for sampling
```

---

## 5. The QMatMul Problem

### 5.1 Problem Statement

Candle's `mul_mat_vec_q5_K_q8_1_cuda1` kernel is the fastest Q5_K GEMV
available. Our custom Q5_K GEMV (`gemv_q5k_fused`) is ~2ms slower over
the full model. We MUST use Candle's kernel but without Tensor wrapping.

### 5.2 Solution: Already Implemented

`raw_qmatmul.rs` already solves this completely:

1. **`raw_quantize_q8_1(input, buf, ncols, dev)`** — Calls Candle's
   `quantize_q8_1` kernel directly with pre-allocated `buf.q8_input`.
   Zero cudaMalloc.

2. **`raw_q5k_gemv_candle(weight_data, buf, ncols, nrows, dev)`** — Calls
   Candle's `mul_mat_vec_q5_K_q8_1_cuda1` kernel directly with pre-allocated
   `buf.output`. Zero cudaMalloc.

### 5.3 Key Optimization: Q8_1 Amortization

For GDN layers, 4 projections share the same input (`normed`):
- attn_qkv: [8192, 2048]
- attn_gate: [4096, 2048]
- ssm_beta: [32, 2048]
- ssm_alpha: [32, 2048]

**One Q8_1 quantization, four GEMVs.** Saves 3 kernel launches + 3 input reads.

For attention layers, 3 projections share `normed`:
- wq: [8192, 2048]
- wk: [512, 2048]
- wv: [512, 2048]

**One Q8_1 quantization, three GEMVs.**

### 5.4 The kernel cache problem

`dev.get_or_load_func("quantize_q8_1", module)` does a HashMap lookup every
call. Over 40 layers * ~8 GEMV calls = ~320 lookups/token. Each lookup is
~1.5us = ~0.5ms total.

**Solution:** Cache `CudaFunction` handles at init time:

```rust
pub struct CachedKernelFuncs {
    quantize_q8_1: CudaFunction,
    mul_mat_vec_q5k: CudaFunction,
    rms_norm: CudaFunction,
    silu_mul: CudaFunction,
    // ... all other kernels
}
```

Load all functions once in `RawBuffers::new()`. Pass `&CachedKernelFuncs` to
every raw kernel call instead of `&CudaDevice`. This eliminates ALL HashMap
lookups from the hot path.

---

## 6. The DeltaNet Problem

### 6.1 Current State

`deltanet_step_fused_raw()` already exists in `src/kernels/deltanet_step.rs`
and takes `CudaSlice<f32>` directly. It is production-ready.

### 6.2 Remaining Issue: In-Place State Update

The current kernel takes separate `s_in` and `s_out` pointers. For v4, we
want to avoid the 2 MB allocation for `s_out` per layer (or use double
buffering with pointer swaps).

**Analysis of kernel correctness for `s_in == s_out`:**

The kernel loads the entire column into `s_reg[128]` before writing back:
```cuda
for (int i = 0; i < D; i++) {
    float val = s_col_in[i * D + j] * g;   // READ from s_in
    s_reg[i] = val;                          // store in register
    sk_j += val * k_s[i];
}
// ... compute delta ...
for (int i = 0; i < D; i++) {
    float val = s_reg[i] + k_s[i] * delta_j;
    s_col_out[i * D + j] = val;             // WRITE to s_out
    out_j += val * q_s[i];
}
```

Each thread reads column `j` completely before writing. Threads operate on
independent columns (no cross-column dependencies). **Therefore `s_in == s_out`
is safe.** The kernel can update the state in-place.

**Decision:** Pass the same `CudaSlice` as both `state` and `new_state_out` to
`deltanet_step_fused_raw()`. The kernel handles it correctly. Zero additional
allocation.

### 6.3 Raw DeltaNet Call

```rust
deltanet_step_fused_raw(
    &state.gdn_states[gdn_idx],       // s_in
    &buffers.gdn.q_scaled,
    &buffers.gdn.k_expanded,
    &buffers.gdn.v_copy,
    &buffers.gate_exp,
    &buffers.beta_out,
    &mut state.gdn_states[gdn_idx],   // s_out = s_in (in-place)
    &mut buffers.deltanet_out,
    dt_rank, d_state, dev,
)?;
```

**BUT:** Rust's borrow checker will reject simultaneous `&` and `&mut` on
`state.gdn_states[gdn_idx]`. Solutions:

1. **Double buffer with swap:** Keep two CudaSlice per layer (`state_a`, `state_b`),
   alternate which is input/output, swap pointers after each step. +60 MB but
   zero unsafe.

2. **unsafe raw pointer:** Extract `*const f32` and `*mut f32` from the same
   CudaSlice. The kernel is proven safe for aliasing. Minimal overhead.

3. **Split the Vec:** Use `split_at_mut` — not applicable since we need
   immutable + mutable access to the same element.

**Decision:** Double buffer with swap. The 60 MB cost is acceptable (we have
849 MB free). No unsafe needed.

```rust
pub struct GdnStateDoubleBuffer {
    a: CudaSlice<f32>,
    b: CudaSlice<f32>,
    current_is_a: bool,
}

impl GdnStateDoubleBuffer {
    fn current(&self) -> &CudaSlice<f32> {
        if self.current_is_a { &self.a } else { &self.b }
    }
    fn next(&mut self) -> &mut CudaSlice<f32> {
        if self.current_is_a { &mut self.b } else { &mut self.a }
    }
    fn swap(&mut self) {
        self.current_is_a = !self.current_is_a;
    }
}
```

---

## 7. The MRoPE Problem

### 7.1 Current Overhead

`MRoPE::apply()` for a single token (decode mode) creates ~12 Tensors:
- 3 sections * (narrow cos + narrow sin + unsqueeze + expand) = 12 ops
- Plus narrow x1, x2, mul, sub, add, cat per section = ~18 more
- Total: ~30 Tensor ops per attention layer
- 10 attention layers * 2 (Q and K) = 600 Tensor ops/token just for RoPE

### 7.2 Solution: Fused Raw MRoPE Kernel

**Design a CUDA kernel** that applies partial sectioned MRoPE in a single launch:

```cuda
// Apply MRoPE to [num_heads, head_dim] in-place.
// Only the first n_rot dims are rotated; the rest are untouched.
// cos_table[pos, pair_idx], sin_table[pos, pair_idx] are precomputed on device.
extern "C" __global__ void mrope_apply_kernel(
    const float* __restrict__ input,   // [num_heads, head_dim]
    float*       __restrict__ output,  // [num_heads, head_dim]
    const float* __restrict__ cos_s0,  // [MAX_POS, pairs_s0] — section 0 cos
    const float* __restrict__ sin_s0,  // section 0 sin
    const float* __restrict__ cos_s1,  // section 1 cos
    const float* __restrict__ sin_s1,
    const float* __restrict__ cos_s2,  // section 2 cos
    const float* __restrict__ sin_s2,
    int pos_s0, int pos_s1, int pos_s2,  // position for each section
    int num_heads, int head_dim,
    int pairs_s0, int pairs_s1, int pairs_s2,
    int n_rot
) {
    // Grid: (num_heads,)  Block: (head_dim,) or (128,) if head_dim > 128
    int h = blockIdx.x;
    int d = threadIdx.x;  // dimension index within the head
    if (d >= head_dim) return;

    int base = h * head_dim;
    float x = input[base + d];

    if (d >= n_rot) {
        // Pass-through: no rotation
        output[base + d] = x;
        return;
    }

    // Determine which section and pair this dimension belongs to
    int dim_s0 = pairs_s0 * 2;
    int dim_s1 = pairs_s1 * 2;

    float cos_val, sin_val;
    int pair_idx, partner_d;
    bool is_first_half;  // whether this is x1 (first half of pair) or x2

    if (d < dim_s0) {
        // Section 0
        int local_d = d;
        pair_idx = local_d < pairs_s0 ? local_d : local_d - pairs_s0;
        is_first_half = local_d < pairs_s0;
        partner_d = is_first_half ? d + pairs_s0 : d - pairs_s0;
        cos_val = cos_s0[pos_s0 * pairs_s0 + pair_idx];
        sin_val = sin_s0[pos_s0 * pairs_s0 + pair_idx];
    } else if (d < dim_s0 + dim_s1) {
        // Section 1
        int local_d = d - dim_s0;
        pair_idx = local_d < pairs_s1 ? local_d : local_d - pairs_s1;
        is_first_half = local_d < pairs_s1;
        partner_d = is_first_half ? d + pairs_s1 : d - pairs_s1;
        cos_val = cos_s1[pos_s1 * pairs_s1 + pair_idx];
        sin_val = sin_s1[pos_s1 * pairs_s1 + pair_idx];
    } else {
        // Section 2
        int local_d = d - dim_s0 - dim_s1;
        pair_idx = local_d < pairs_s2 ? local_d : local_d - pairs_s2;
        is_first_half = local_d < pairs_s2;
        partner_d = is_first_half ? d + pairs_s2 : d - pairs_s2;
        cos_val = cos_s2[pos_s2 * pairs_s2 + pair_idx];
        sin_val = sin_s2[pos_s2 * pairs_s2 + pair_idx];
    }

    float partner = input[base + partner_d];

    if (is_first_half) {
        // x1' = x1 * cos - x2 * sin
        output[base + d] = x * cos_val - partner * sin_val;
    } else {
        // x2' = x1 * sin + x2 * cos
        output[base + d] = partner * sin_val + x * cos_val;
    }
}
```

### 7.3 Raw MRoPE Tables

```rust
pub struct RawMRoPETables {
    pub cos_s0: CudaSlice<f32>,  // [65536, 11]
    pub sin_s0: CudaSlice<f32>,
    pub cos_s1: CudaSlice<f32>,  // [65536, 11]
    pub sin_s1: CudaSlice<f32>,
    pub cos_s2: CudaSlice<f32>,  // [65536, 10]
    pub sin_s2: CudaSlice<f32>,
    pub pairs: [usize; 3],      // [11, 11, 10]
    pub n_rot: usize,           // 64
}
```

Extract these from the existing `MRoPE` struct's `device_cache` at init time.

### 7.4 Calling Convention

```rust
fn raw_mrope_apply(
    input: &CudaSlice<f32>,       // [num_heads * head_dim]
    output: &mut CudaSlice<f32>,  // [num_heads * head_dim]
    tables: &RawMRoPETables,
    position: usize,              // current token position
    num_heads: usize,
    head_dim: usize,
    dev: &CudaDevice,
) -> Result<()> {
    // For text-only: pos_s0 = position, pos_s1 = 0, pos_s2 = 0
    // 1 kernel launch, 0 Tensor allocations
}
```

**Saving:** 30 Tensor ops * 10 layers * 2 (Q+K) = 600 ops -> 20 kernel launches.

---

## 8. Implementation Order

Ordered by estimated impact (ms saved per step) and implementation complexity.

### Phase 1: Weight Extraction + Cached Kernel Functions (est. -1.5ms)

**What:** Extract all weight CudaSlice pointers at load time. Cache all
CudaFunction handles to eliminate HashMap lookups.

**Files:** New `src/raw_weights.rs`, modify `src/raw_forward.rs`

**Why first:** This is prerequisite infrastructure for all other phases.
The kernel function cache alone saves ~0.5ms (320 HashMap lookups/token).

**Estimated gain:** 0.5ms (cache) + 1.0ms (no more storage_and_layout)

### Phase 2: Raw GDN Layer (est. -3.0ms)

**What:** Implement `raw_forward_gdn_layer()` using only CudaSlice operations.
Use existing `raw_rms_norm`, `raw_quantize_q8_1`, `raw_q5k_gemv_candle`,
`fused_beta_alpha_gate_raw` (needs raw variant), `raw_fused_conv1d_silu_update`,
`raw_split_l2norm_expand` (from GdnScratchBuffers path), `deltanet_step_fused_raw`.

**This is the biggest win** because there are 30 GDN layers vs 10 attention layers.

**New kernels needed:**
- `fused_beta_alpha_gate_raw` — raw CudaSlice variant of existing Tensor version
- `raw_rms_norm_silu_gate` — raw variant of `fused_rms_norm_silu_gate_tensor`
- These are thin wrappers: the CUDA kernels are identical, just skip Tensor extraction

**Files:** Extend `src/raw_forward.rs`, add raw variants to `src/kernels/elementwise.rs`

**Estimated gain:** 30 layers * 0.10ms overhead/layer = 3.0ms

### Phase 3: Raw MoE FFN (est. -0.5ms)

**What:** The MoE path is already mostly raw. Eliminate remaining
`storage_and_layout()` calls (currently 6 per MoE layer) by passing
CudaSlice directly from RawWeights.

**Files:** Modify `moe_ffn_forward_raw` to take `&RawGdnWeights` or
`&RawAttnWeights` directly.

**Estimated gain:** 40 layers * 6 * 2us = 0.5ms

### Phase 4: Raw Attention Layer (est. -1.0ms)

**What:** Implement `raw_forward_attn_layer()`. The attention path has fewer
layers (10) but more operations per layer.

**New kernels needed:**
- `raw_mrope_apply` — fused MRoPE kernel (Section 7)
- `raw_deinterleave_q_gate` — split Q+gate from interleaved layout
- `raw_rms_norm_per_head` — per-head RMSNorm for QK normalization
- `raw_sigmoid_gate_mul` — sigmoid(gate) * attn_out
- Raw KV append — memcpy into ring buffer

**Files:** Extend `src/raw_forward.rs`, new `src/kernels/mrope_raw.rs`

**Estimated gain:** 10 layers * 0.10ms = 1.0ms

### Phase 5: Raw Token Loop + LM Head (est. -1.0ms)

**What:** Wire up the complete `raw_forward_token()` function that calls
phases 2-4 in sequence, plus embedding + LM head. Eliminate the last
remaining Tensor operations (embedding lookup, final norm, lm_head GEMV).

**Embedding:** The embed table is on CPU (to save VRAM). The lookup is
`index_select` on CPU -> `htod` 8 KB. This is fast (~2us) and cannot be
improved. Keep as-is but use `memcpy_htod` into `buffers.hidden` directly.

**LM Head:** One Q5_K GEMV with output size 248320. This is the single
largest GEMV and takes ~2ms. Using `raw_q5k_gemv_candle` saves ~0.1ms
allocation overhead.

**Files:** Complete `src/raw_forward.rs`

**Estimated gain:** 0.5ms (loop overhead) + 0.5ms (embed/lm_head Tensor ops)

### Phase 6: Double-Buffer Optimization + Final Polish (est. -0.5ms)

**What:** Implement GDN state double buffering (Section 6.3). Profile and
eliminate any remaining hot spots. Tune kernel launch configurations.

**Consider:** Moving the embed table to GPU (costs ~2 GB VRAM but saves
the htod per token). At 8 KB/token this is negligible — skip unless VRAM
allows.

**Estimated gain:** 0.5ms from reduced overhead + better pipeline

### Summary

| Phase | Component | Est. Gain | Cumulative |
|-------|-----------|-----------|------------|
| 1 | Weight extraction + kernel cache | 1.5ms | 1.5ms |
| 2 | Raw GDN layer (30 layers) | 3.0ms | 4.5ms |
| 3 | Raw MoE FFN | 0.5ms | 5.0ms |
| 4 | Raw attention layer (10 layers) | 1.0ms | 6.0ms |
| 5 | Raw token loop + LM head | 1.0ms | 7.0ms |
| 6 | Double buffer + polish | 0.5ms | 7.5ms |

**Projected result: 16.8ms - 7.5ms = 9.3ms/token = ~107 tok/s**

This exceeds the 89.7 tok/s ik_llama baseline and approaches the ~100 tok/s
target. The remaining 9.3ms is pure GPU compute time — the irreducible
minimum given the kernel execution costs (QMatMul, DeltaNet, MoE IQ3_S GEMVs).

---

## Appendix A: Kernel Function Census

Complete list of distinct CUDA kernel functions called per token (for caching):

1. `rms_norm_kernel` (elementwise.rs) — used ~82 times/token
2. `quantize_q8_1` (Candle QUANTIZED module) — ~46 times
3. `mul_mat_vec_q5_K_q8_1_cuda1` (Candle QUANTIZED module) — ~46 times (Q5_K GEMV)
4. `fused_beta_alpha_gate_kernel` (elementwise.rs) — 30 times
5. `fused_conv1d_silu_update_kernel` (elementwise.rs) — 30 times
6. `l2_norm_groups_kernel` (elementwise.rs) — 30 times
7. `expand_groups_kernel` (elementwise.rs) — 30 times
8. `scale_kernel` (elementwise.rs) — 30 times
9. `deltanet_step_kernel` (deltanet_step.rs) — 30 times
10. `fused_rms_norm_silu_gate_kernel` (elementwise.rs) — 30 times
11. `add_inplace_kernel` (elementwise.rs) — 80 times
12. `silu_mul_kernel` (elementwise.rs) — 40+ times
13. `f32_gemv_kernel` (elementwise.rs) — 40 times (router)
14. `topk_softmax_kernel` (topk_softmax.rs) — 40 times
15. `gemv_iq3s_q8_batched_kernel` (iq3s_gemv.rs) — 120 times (gate+up+down * 40)
16. `quantize_f32_to_q8_1_gpu` (iq3s_gemv.rs) — 40 times
17. `weighted_combine_kernel` (elementwise.rs) — 40 times
18. `mrope_apply_kernel` (NEW) — 20 times
19. `fused_gqa_attention_kernel` (gqa_attention.rs) — 10 times
20. `argmax_kernel` (elementwise.rs) — 1 time

**Total: ~20 distinct kernel functions to cache.**

## Appendix B: Memory Budget

```
Current VRAM usage:       14994 MiB (model weights + Candle overhead)
Total VRAM:               15843 MiB
Free:                        849 MiB

New permanent allocation:
  GDN double-buffer:      +60 MB (if used; 0 MB with in-place kernel)
  Raw scratch buffers:     ~2 MB (mostly already allocated)
  MRoPE raw tables:         0 MB (extracted from existing Tensor)
  KV ring buffers:          0 MB (already allocated via KvRingCache)

Net change:               +2 MB to +62 MB
Remaining free:           787-847 MiB  (sufficient)
```

## Appendix C: Compatibility

The raw forward path should be **opt-in** via `CHIMERE_RAW_V4=1` environment
variable. The existing Candle path remains the default and reference for
correctness testing. A single-token numerical comparison test should verify
that `raw_forward_token` and `forward_token_preloaded` produce identical
logits (within f32 epsilon).

The `RawState` can be initialized from an existing `GdnRecurrentState` by
extracting CudaSlice pointers, and vice versa for snapshot/restore (MTP
branching). Snapshot/restore requires `dev.clone_dtod()` to deep-copy the
CudaSlice contents.
