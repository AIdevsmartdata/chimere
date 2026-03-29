# Timing Analysis: Chimere-DeltaNet vs llama.cpp/ik_llama

## Model: Qwen3.5-35B-A3B IQ3_S custom-mix

**Measured performance:**
- Chimere: 37.6 tok/s (26.6 ms/token)
- llama.cpp (stock, sm120): 77 tok/s (13.0 ms/token)
- ik_llama (sm120): 114 tok/s (8.8 ms/token)

Note: 1000/37.6 = 26.6 ms, not 18.9 ms as quoted in the original prompt.

**GPU:** RTX 5060 Ti (16 GB GDDR7)
- Memory bandwidth (spec): 448 GB/s
- Realistic sustained bandwidth: ~350 GB/s (78% of peak, typical for GEMV workloads)
- L2 cache: 48 MB (sm_120 Blackwell)

---

## Architecture Summary

| Parameter | Value |
|---|---|
| Main layers | 40 (30 GDN + 10 attention) |
| hidden_size | 2048 |
| GDN: n_group (H_k) | 16 |
| GDN: d_state (S) | 128 |
| GDN: dt_rank (H_v) | 32 |
| GDN: key_dim (H_k * S) | 2048 |
| GDN: value_dim (H_v * S = d_inner) | 4096 |
| GDN: conv_channels (2*key + value) | 8192 |
| Attn: num_heads | 16 |
| Attn: num_kv_heads | 2 |
| Attn: Q head_dim | 512 (asymmetric) |
| Attn: KV head_dim | 256 |
| MoE: num_experts | 256 |
| MoE: experts_per_token (top-K) | 8 |
| MoE: expert_ffn_hidden | 512 |
| MoE: shared_expert_ffn_hidden | 512 |
| vocab_size | 248320 |

---

## Quantization Block Sizes

| Type | Block elements | Block bytes | Bits/weight |
|---|---|---|---|
| IQ3_S | 256 | 110 | 3.4375 |
| Q5_K | 256 | 176 | 5.5 |
| Q8_0 | 32 | 34 | 8.5 |
| F32 | 1 | 4 | 32.0 |

---

## Per-Component Weight Read Analysis

### 1. GDN SSM Projections (30 GDN layers)

All SSM weight matrices are **Q5_K** in the custom-mix (critical path override).

| Tensor | Shape (out, in) | Elements | Q5_K bytes | Per layer |
|---|---|---|---|---|
| attn_qkv | [8192, 2048] | 16,777,216 | 11,534,336 | 11.00 MB |
| attn_gate | [4096, 2048] | 8,388,608 | 5,767,168 | 5.50 MB |
| ssm_out | [2048, 4096] | 8,388,608 | 5,767,168 | 5.50 MB |
| ssm_alpha | [32, 2048] | 65,536 | 69,632 (Q8_0) | 0.07 MB |
| ssm_beta | [32, 2048] | 65,536 | 69,632 (Q8_0) | 0.07 MB |

**Per GDN layer SSM projections:** 23,207,936 bytes = **22.13 MB**
**30 GDN layers total:** 696,238,080 bytes = **663.9 MB**

### 2. MoE Routed Experts (40 layers: 30 GDN + 10 attention)

All routed experts are **IQ3_S**. Only 8 of 256 experts are activated per token.

| Tensor | Shape per expert | Elements | IQ3_S bytes/expert |
|---|---|---|---|
| gate_exps | [512, 2048] | 1,048,576 | 450,560 |
| up_exps | [512, 2048] | 1,048,576 | 450,560 |
| down_exps | [2048, 512] | 1,048,576 | 450,560 |

Per expert total: 1,351,680 bytes = **1.289 MB**
8 active experts per layer: 10,813,440 bytes = **10.31 MB**
**40 layers total:** 432,537,600 bytes = **412.5 MB**

### 3. MoE Router (40 layers)

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| gate_inp | [2048, 256] | F32 | 2,097,152 |

Per layer: **2.0 MB** (but fits in L2 cache after first access)
40 layers total: 83,886,080 bytes = **80.0 MB**

**Router is L2-cacheable:** At 2 MB per layer with a 48 MB L2, the router weights
almost certainly hit L2 after the first few layers (the active set cycles). For
analysis purposes, we assume router reads cost ~50% of DRAM bandwidth due to partial
L2 hits. Effective router cost: ~40 MB from DRAM.

### 4. Shared Expert (40 layers)

All shared expert weights are **Q5_K**.

| Tensor | Shape | Elements | Q5_K bytes |
|---|---|---|---|
| gate_shexp | [512, 2048] | 1,048,576 | 720,896 |
| up_shexp | [512, 2048] | 1,048,576 | 720,896 |
| down_shexp | [2048, 512] | 1,048,576 | 720,896 |

Per layer shared expert: 2,162,688 bytes = **2.06 MB**
40 layers total: 86,507,520 bytes = **82.5 MB**

### 5. Attention Projections (10 attention layers)

All attention weights are **Q5_K** in the custom-mix.

| Tensor | Shape (out, in) | Elements | Q5_K bytes |
|---|---|---|---|
| wq | [16384, 2048] | 33,554,432 | 23,068,672 |
| wk | [512, 2048] | 1,048,576 | 720,896 |
| wv | [512, 2048] | 1,048,576 | 720,896 |
| wo | [2048, 8192] | 16,777,216 | 11,534,336 |

Per attention layer: 36,044,800 bytes = **34.38 MB**
10 attention layers total: 360,448,000 bytes = **343.8 MB**

### 6. Norms and Small Tensors (40 layers)

| Tensor | Shape | Type | Bytes | Notes |
|---|---|---|---|---|
| attn_norm | [2048] | F32 | 8,192 | Per layer |
| post_attention_norm | [2048] | F32 | 8,192 | Per layer |
| ssm_norm | [128] | F32 | 512 | GDN only (30) |
| ssm_conv1d | [8192, 4] | F32 | 131,072 | GDN only (30) |
| ssm_a | [32] | F32 | 128 | GDN only (30) |
| ssm_dt_bias | [32] | F32 | 128 | GDN only (30) |
| q_norm | [256] | F32 | 1,024 | Attn only (10) |
| k_norm | [256] | F32 | 1,024 | Attn only (10) |
| gate_inp_shexp | [2048] | F32 | 8,192 | MoE layers (40) |

Norms per GDN layer: ~148 KB (dominated by conv1d weights)
Norms per attn layer: ~18 KB
**Total norms/small (all 40 layers):** ~4.9 MB

### 7. LM Head and Embedding

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| output.weight (lm_head) | [248320, 2048] | Q5_K | 349,634,560 |
| token_embd.weight | [248320, 2048] | (CPU, not counted) | 0 |

LM head: **333.4 MB** (read once per token)
Embedding: on CPU, single row (8 KB) transferred per token — negligible.

### 8. DeltaNet State Read/Write (30 GDN layers)

Each GDN layer reads AND writes the recurrent state.

| Buffer | Shape | Type | Bytes |
|---|---|---|---|
| gdn_state | [1, 32, 128, 128] | F32 | 2,097,152 |
| conv_state | [1, 8192, 3] | F32 | 98,304 |

Per GDN layer state I/O: 2 * (2,097,152 + 98,304) = 4,390,912 bytes = **4.19 MB** (read + write)
30 GDN layers total state I/O: 131,727,360 bytes = **125.6 MB**

### 9. KV Cache I/O (10 attention layers)

For early generation (position ~100), KV cache is small.
Per attention layer per token: append + read for scores.
At position P:
- K cache: [1, 2, P, 256] read + expand to [1, 16, P, 256]
- V cache: [1, 2, P, 256] read

KV cache is small relative to weights for short sequences. At P=100:
Per layer: 2 * 100 * 256 * 4 * 2 (K+V) = 409,600 bytes = 0.39 MB
10 layers: ~3.9 MB (negligible at short seq; grows linearly)

---

## Total Bytes Read Per Token

| Component | Bytes | MB | % of total |
|---|---|---|---|
| GDN SSM projections (30 layers) | 696,238,080 | 663.9 | 33.7% |
| MoE routed experts (40 layers, 8/256) | 432,537,600 | 412.5 | 20.9% |
| MoE router (40 layers, partial L2) | ~41,943,040 | ~40.0 | 2.0% |
| Shared expert (40 layers) | 86,507,520 | 82.5 | 4.2% |
| Attention projections (10 layers) | 360,448,000 | 343.8 | 17.5% |
| DeltaNet state I/O (30 layers) | 131,727,360 | 125.6 | 6.4% |
| LM head | 349,634,560 | 333.4 | 16.9% |
| Norms + small tensors | ~5,100,000 | ~4.9 | 0.2% |
| KV cache (short seq) | ~4,000,000 | ~3.9 | 0.2% |
| **TOTAL** | **~2,106,478,048** | **~2,009 MB** | **100%** |

**~2.0 GB read from VRAM per token.**

---

## Theoretical Minimum Time

At 350 GB/s sustained bandwidth:

**Theory = 2,009 MB / 350,000 MB/s = 5.74 ms/token = 174 tok/s**

At 448 GB/s peak bandwidth:

**Peak theory = 2,009 MB / 448,000 MB/s = 4.48 ms/token = 223 tok/s**

---

## Efficiency Analysis

### vs Chimere (37.6 tok/s = 26.6 ms/token)

**Bandwidth utilization = 2,009 MB / (26.6 ms * 350 GB/s) = 2,009 / 9,310 = 21.6%**

This is very low. The gap of 26.6 - 5.74 = 20.9 ms is pure overhead.

### vs llama.cpp stock (77 tok/s = 13.0 ms/token)

llama.cpp reads the same weights but uses:
- ggml MMVQ kernels (fused dequant + matmul in one kernel launch)
- Batched kernel launches (graph scheduling)
- No per-operation tensor allocation overhead

**llama.cpp efficiency = 2,009 / (13.0 * 350) = 2,009 / 4,550 = 44.2%**

### vs ik_llama (114 tok/s = 8.8 ms/token)

ik_llama additionally uses:
- sm_120 optimized kernel variants (tensor core mma.sync for Blackwell)
- Better occupancy tuning (more warps per SM)

**ik_llama efficiency = 2,009 / (8.8 * 350) = 2,009 / 3,080 = 65.2%**

Note: Even ik_llama at 65% is not at the bandwidth limit, because:
1. Kernel launch overhead (CPU dispatch between layers)
2. GPU idle between kernels (pipeline bubbles)
3. L2 cache misses and DRAM latency
4. Softmax/top-K sync points

---

## Where the 26.6 ms Goes (Chimere Breakdown)

Estimated breakdown based on GRAN_PROF data and code analysis:

```
Component                     Bytes read    Theory @350GB/s   Est. actual   Efficiency
──────────────────────────────────────────────────────────────────────────────────────
GDN SSM projections (30 lyr)  662.6 MB      1.89 ms           6.8 ms        28%
  attn_qkv (Q5_K GEMV)        330.0 MB      0.94 ms           3.0 ms        31%
  attn_gate (Q5_K GEMV)        165.0 MB      0.47 ms           1.5 ms        31%
  ssm_out (Q5_K GEMV)          165.0 MB      0.47 ms           1.5 ms        31%
  alpha+beta (Q8_0 GEMV, tiny)   2.6 MB      0.01 ms           0.8 ms         1%

MoE routed experts (40 lyr)   412.5 MB      1.18 ms           8.0 ms        15%
  gate+up GEMV (IQ3S, 8 exp)   274.9 MB      0.79 ms           3.2 ms        25%
  down GEMV (IQ3S, 8 exp)      137.6 MB      0.39 ms           1.6 ms        25%
  router matmul + topK          40.0 MB      0.11 ms           1.2 ms         9%
  silu_mul + weighted_combine    (compute)    ~0 ms             0.4 ms         -
  storage_and_layout overhead    (CPU)        0 ms              1.6 ms         -

Shared expert (40 lyr)          82.6 MB      0.24 ms           1.2 ms        20%
  gate+up+down (Q5_K, 3 GEMVs)  82.6 MB      0.24 ms           1.2 ms        20%

Attention projs (10 lyr)       343.8 MB      0.98 ms           3.2 ms        31%
  wq (Q5_K, [16384,2048])       220.0 MB      0.63 ms           1.6 ms        39%
  wk+wv (Q5_K, small)            13.7 MB      0.04 ms           0.4 ms        10%
  wo (Q5_K, [2048,8192])        110.0 MB      0.31 ms           0.8 ms        39%
  QK norm + RoPE + scores        (compute)    ~0 ms             0.4 ms         -

DeltaNet state I/O (30 lyr)   125.6 MB      0.36 ms           1.8 ms        20%
  state read+write               125.6 MB      0.36 ms           1.8 ms        20%

LM head                        333.1 MB      0.95 ms           1.8 ms        53%
  output.weight (Q5_K GEMV)     333.1 MB      0.95 ms           1.8 ms        53%

Norms + elementwise (40 lyr)     4.9 MB      0.01 ms           1.2 ms         1%
  rms_norm (2x per layer)        (compute)    ~0 ms             0.8 ms         -
  silu, sigmoid, residual add    (compute)    ~0 ms             0.4 ms         -

Conv1d (30 GDN layers)           3.9 MB      0.01 ms           0.6 ms         2%

Embedding (CPU->GPU transfer)    0.008 MB     ~0 ms             0.2 ms         -

Candle dispatch overhead         0 MB         0 ms              ~2.0 ms        -
  40 layers * ~50 us overhead                                   ~2.0 ms
──────────────────────────────────────────────────────────────────────────────────────
TOTAL                        ~2,009 MB       5.74 ms          ~26.6 ms       21.6%
```

---

## Root Cause Analysis

### Why Chimere is 3.0x slower than ik_llama (26.6 vs 8.8 ms)

The 17.8 ms gap decomposes into:

1. **Kernel efficiency (~8 ms, 45% of gap)**
   - Candle's QMatMul kernels are generic CUDA matmul, not the fused dequant+dot
     kernels that ggml uses (MMVQ pattern: read quantized blocks, dequant in registers,
     accumulate dot product — one kernel, one memory read)
   - Chimere's IQ3_S GEMV kernel reads Q8_1-quantized input + IQ3_S weights and does
     dp4a, but launches per-expert (8 launches for gate + 8 for up + 8 for down = 24
     GEMV launches per MoE layer, vs ggml's single batched launch)
   - Q5_K projections go through Candle QMatMul which dequants to F32 then does matmul
     in a second kernel (2 kernels instead of 1)

2. **Candle dispatch overhead (~6 ms, 34% of gap)**
   - Each Candle operation (rms_norm, matmul, silu, add, reshape, narrow, etc.) creates
     a Tensor object with Arc, Shape, Layout, Storage allocation + dispatch
   - ~80-100 Candle ops per GDN layer * 30 layers = ~2400-3000 Candle ops per token
   - ~50 Candle ops per attention layer * 10 layers = ~500 Candle ops
   - At ~2 us per Candle op dispatch: 3500 * 2 us = 7 ms
   - This is consistent with the raw_forward.rs comment: "~52 ms / 74% of 70 ms"
     measured at an earlier stage before optimizations reduced it to ~6 ms

3. **MoE routing sync point (~1.2 ms, 7% of gap)**
   - GPU top-K softmax eliminates the old 1 KB dtoh, but still requires 64 bytes
     dtoh (8 indices + 8 weights) per MoE layer = 40 sync points
   - Each sync costs ~25-30 us (PCIe round-trip + GPU drain)
   - 40 * 30 us = 1.2 ms

4. **DeltaNet state overhead (~1.4 ms, 8% of gap)**
   - 30 layers * 2 MB state read + 2 MB state write = 125.6 MB
   - But state access pattern is scattered (transpose, broadcast_mul) not sequential
   - ggml doesn't have this (it has different SSM state handling)

5. **Elementwise kernel launch overhead (~1.2 ms, 7% of gap)**
   - Each GDN layer has ~10 elementwise operations (sigmoid, softplus, exp, silu,
     broadcast_mul, add, l2_norm, etc.)
   - Each is a separate CUDA kernel launch (~3-5 us per launch)
   - 30 * 10 * 4 us = 1.2 ms
   - ggml fuses many of these into compound kernels

---

## Comparison: What llama.cpp Does Differently

| Aspect | Chimere | llama.cpp/ik_llama |
|---|---|---|
| GEMV kernel | 2-pass: dequant then matmul | 1-pass: fused dequant+dot (MMVQ) |
| Expert dispatch | 24 individual GEMV launches/layer | Batched/fused expert kernels |
| Tensor alloc | Per-op Tensor(Arc,Shape,Layout) | Zero alloc (ggml graph) |
| Elementwise | Separate CUDA kernels each | Fused compound kernels |
| Top-K routing | GPU kernel + 64B dtoh sync | Integrated into graph |
| State management | Candle Tensor clone/cat | In-place buffer reuse |
| Graph scheduling | Sequential CPU dispatch | Pre-built compute graph |

---

## What Would Close the Gap

### Phase 1: Low-hanging fruit (target: 50+ tok/s)
- **Fuse QMatMul into MMVQ-style kernels** for Q5_K projections
  - Eliminates the dequant intermediate buffer
  - Saves 1 kernel launch + 1 memory write + 1 memory read per projection
  - Expected: -3 ms on SSM projections alone

### Phase 2: Structural (target: 70+ tok/s)
- **Batch all 8 expert GEMVs into single kernel launch** (already partially done
  with `moe_ffn_fused`, but still using IQ3_S dequant + separate matmul)
- **Eliminate Candle dispatch entirely** for the hot path via raw_forward.rs
  - The infrastructure exists (RawGpuBuffers), extend to full token loop
  - Expected: -4 ms

### Phase 3: Matching ik_llama (target: 90+ tok/s)
- **Port ggml MMVQ grid pattern** (4 warps/block, 1 row/block) for sm_120
- **Use tensor cores (mma.sync)** for the larger projections (wq, lm_head)
- **Fuse elementwise chains** (sigmoid+softplus+exp+mul into 1 kernel)

---

## Summary

| Metric | Value |
|---|---|
| Total weight bytes per token | 2,009 MB |
| Theoretical minimum (350 GB/s) | 5.74 ms (174 tok/s) |
| Theoretical minimum (448 GB/s peak) | 4.48 ms (223 tok/s) |
| Chimere actual | 26.6 ms (37.6 tok/s) |
| Chimere bandwidth efficiency | 21.6% |
| llama.cpp stock actual | 13.0 ms (77 tok/s) |
| llama.cpp bandwidth efficiency | 44.2% |
| ik_llama sm120 actual | 8.8 ms (114 tok/s) |
| ik_llama bandwidth efficiency | 65.2% |
| **Gap: Chimere vs theory** | **4.6x** |
| **Gap: Chimere vs ik_llama** | **3.0x** |
| **Gap: ik_llama vs theory** | **1.5x** |

The dominant bottleneck is NOT memory bandwidth. At 21.6% bandwidth utilization,
Chimere is heavily **compute/dispatch bound**. The primary targets for optimization are:
1. Fused dequant+dot GEMV kernels (replace 2-pass with 1-pass) — ~8 ms savings
2. Eliminate Candle per-op dispatch overhead — ~6 ms savings
3. Reduce MoE routing sync points — ~1.2 ms savings

Together these would bring Chimere from 26.6 ms to ~11 ms (~91 tok/s), competitive
with llama.cpp stock and within 25% of ik_llama.
