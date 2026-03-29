# Profiling Analysis: llama.cpp vs chimere-deltanet
## Qwen3.5-35B-A3B IQ3_S custom-mix on RTX 5060 Ti (sm_120)
## 2026-03-15

---

## 1. llama.cpp Baseline (stock b8125)

| Test | tok/s | ms/token |
|------|------:|--------:|
| tg32 (3 runs) | 76.98 +/- 1.19 | 12.99 |
| tg64 (3 runs) | 78.42 +/- 0.86 | 12.75 |
| pp1 | 68.80 | 14.53 |
| pp512 | 2102.86 | 0.476 |

**Target: 13.0 ms/token @ tg32**

Note: nsys 2024.6.2 cannot profile sm_120 (Blackwell) GPUs. `perf_event_paranoid=4`
also blocks CPU sampling. Kernel-level profiling requires either:
- Upgrade nsys to 2025.x+ with sm_120 CUPTI support, OR
- `sudo sysctl -w kernel.perf_event_paranoid=2`

---

## 2. chimere-deltanet Profiling (CHIMERE_GRAN_PROF=1)

### Per-token breakdown (steady state, averaged over tokens 3-20):

**Architecture: 30 GDN layers + 10 attention layers = 40 total**

#### GDN Layer (single layer, steady state):

| Operation | Time (ms) | % of GDN layer |
|-----------|--------:|------:|
| rms_norm | 0.025 | 3.5% |
| qkv_forward | 0.042 | 5.9% |
| gate_forward | 0.024 | 3.4% |
| beta_sigmoid | 0.015 | 2.1% |
| alpha_gate | 0.030 | 4.2% |
| conv1d_silu | 0.047 | 6.6% |
| split_norm_expand | 0.057 | 8.0% |
| deltanet_step | 0.100 | 14.0% |
| norm_gated_ssm_out | 0.035 | 4.9% |
| **ssm_total** | **0.333** | **46.7%** |
| **moe_ffn_total** | **0.375** | **52.5%** |
| **GDN layer total** | **~0.714** | **100%** |

Note: ssm_total (0.333) + moe_ffn_total (0.375) = 0.708ms.
Residual ~0.006ms is the residual connection + overhead.

#### Attention Layer (single layer, steady state):

| Operation | Time (ms) | % of attn layer | Note |
|-----------|--------:|------:|------|
| attn_norm | 0.243 | 27.7% | **ARTIFACT** (see below) |
| q_proj | 0.038 | 4.3% | |
| k_proj | 0.012 | 1.4% | |
| v_proj | 0.013 | 1.5% | |
| qk_norm_reshape | 0.022 | 2.5% | |
| rope | 0.001 | 0.1% | |
| kv_cache_update | 0.016 | 1.8% | |
| attention_scores | 0.031 | 3.5% | |
| softmax | 0.028 | 3.2% | |
| attention_output | 0.012 | 1.4% | |
| o_proj_gate_residual | 0.034 | 3.9% | |
| ffn_norm | 0.008 | 0.9% | |
| **moe_ffn** | **0.384** | **43.8%** | |
| ffn_residual | 0.007 | 0.8% | |
| **attn_layer_total** | **~0.877** | **100%** | |

**PROFILING ARTIFACT**: `attn_norm = 0.243ms` is inflated. The profiling macro `aprof!`
calls `d.synchronize()` before reading the timestamp. Since `GRAN_PROF` only profiles
layer 0 (GDN) and attn_idx 0 (first attention layer = layer 3), the `attn_norm` sync
captures all pending async GPU work from GDN layers 1 and 2 that executed without any
sync. The actual RMS norm cost is ~0.025ms (same as GDN rms_norm). The "missing" 0.218ms
is GDN layers 1-2 async execution time that was never captured in the GDN profiling.

**Corrected attn_norm**: ~0.025ms
**Corrected attn_layer_total**: ~0.659ms (0.877 - 0.218)
**Unaccounted GDN layers 1-2 time**: 0.218ms = ~0.109ms/layer async tail

#### Full model per-token:

| Component | Time (ms) | % of total |
|-----------|--------:|------:|
| embed | 0.01 | 0.0% |
| 30x GDN layers | 17.62 | 77.5% |
| 10x attn layers | 5.09 | 22.4% |
| lm_head | 0.00 | 0.0% |
| **Total** | **22.72** | **100%** |

**chimere-deltanet: 22.72 ms/token = 44.0 tok/s**

---

## 3. Gap Analysis: chimere-deltanet vs llama.cpp

| | chimere | llama.cpp | Ratio | Gap |
|--|-------:|--------:|------:|----:|
| **Total per-token** | 22.72 ms | 13.0 ms | 1.75x | 9.72 ms |
| **Effective tok/s** | 44.0 | 77.0 | 0.57x | |

### Where is the 9.72ms gap?

#### 3a. MoE FFN (the dominant cost)

MoE FFN runs in every layer (30 GDN + 10 attn = 40 layers):
- chimere: 30 x 0.375 + 10 x 0.384 = **15.09 ms** (66.4% of total)
- llama.cpp estimated (66% of 13ms): **~8.6 ms**
- **MoE gap: ~6.5 ms**

The IQ3_S MoE dequant+matmul is the bottleneck. llama.cpp uses fused MMVQ kernels
that dequantize and multiply in a single pass. chimere uses separate dequant-to-F32
then cuBLAS GEMV, causing 2x memory traffic.

#### 3b. Attention layers

- chimere attn total (corrected, excl. MoE): 10 x (0.659 - 0.384) = **2.75 ms**
- chimere attn total (reported, excl. MoE): 10 x (0.877 - 0.384) = **4.93 ms**
  - The 2.18ms difference = async GDN tail from unprofiled layers 1-2 (artifact)
- llama.cpp estimated attention: ~2.5 ms
- **Corrected attention gap: ~0.25 ms** (small, attention is close to optimal)

#### 3c. GDN/SSM layers

- chimere SSM total: 30 x 0.333 = **9.99 ms** (44% of total)
- llama.cpp SSM total estimated: ~4.0 ms (mamba-like SSM kernels are highly optimized)
- **SSM gap: ~6.0 ms**

Key SSM sub-operations:
- `deltanet_step`: 0.100ms x 30 = 3.0ms -- the actual recurrence, already fast
- `qkv_forward`: 0.042ms x 30 = 1.26ms -- quantized matmul projection
- `split_norm_expand`: 0.057ms x 30 = 1.71ms -- head expansion overhead (see Mar 14 bug fix)
- `conv1d_silu`: 0.047ms x 30 = 1.41ms -- temporal convolution
- `alpha_gate`: 0.030ms x 30 = 0.90ms -- gating
- Everything else (gate, beta, norm): ~0.074ms x 30 = 2.22ms

#### 3d. Summary by category (corrected for profiling artifact)

| Category | chimere (ms) | llama.cpp est (ms) | Gap (ms) | Fix priority |
|----------|----------:|---------------:|---------:|:----------:|
| MoE FFN (40 layers) | 15.09 | ~8.6 | ~6.5 | **#1** |
| SSM/GDN (30 layers) | 9.99 | ~3.5 | ~6.5 | **#2** |
| Attention ops (10 layers) | 2.75 | ~2.5 | ~0.25 | low |
| embed + lm_head | 0.02 | ~0.1 | (ok) | -- |
| Overhead/async tail/sync | ~4.87 | ~2.3 | ~2.6 | #3 |

Note: "overhead/async tail/sync" = 22.72 - 15.09 - 9.99 - 2.75 - 0.02 = -5.13
This is wrong -- because the GDN GRAN profile only measures layer 0 ops,
the actual per-layer GDN time is higher than 0.333+0.375 = 0.708ms.
The corrected GDN per-layer from wall-clock: 17.62ms / 30 = **0.587ms**
The corrected attn per-layer from wall-clock: (5.09 - 2.18) / 10 = **0.291ms** (attn-only)
  + 0.384ms MoE = **0.675ms** total attn layer

**Reconciled wall-clock breakdown**:
- 30 GDN layers: 17.62ms (0.587ms/layer)
- 10 attn layers: 5.09ms (0.509ms/layer, includes attn_norm artifact of 0.218ms for layer 3)
  - Corrected: 2.91ms real + 2.18ms artifact = 5.09ms
- Total: 22.72ms = **44.0 tok/s**

**True GDN per-layer**: 0.587ms (vs profiled 0.714ms for layer 0)
  - Layer 0 has cold-cache overhead (+0.127ms = 22% overhead)
  - Steady-state GDN layers are actually faster than layer 0 profiling suggests

**True attn per-layer**: ~0.509ms wall-clock (but layer 3 absorbs 0.218ms async tail)
  - Corrected: ~0.291ms real per attn layer (excl. async artifact)
  - Very competitive with llama.cpp (~0.25ms/layer estimated)

---

## 4. Root Causes & Fix Priorities

### Priority 1: MoE FFN -- fused MMVQ kernel (saves ~3-4ms)
llama.cpp's `mul_mat_vec_q` reads IQ3_S bytes, dequantizes in registers, and accumulates
the dot product in a single kernel launch per expert. chimere does:
1. `dequant_iq3s_gpu` → writes F32 to VRAM
2. cuBLAS SGEMV → reads F32 from VRAM
This doubles memory traffic (write F32 + read F32 = 2x read IQ3_S).
**Fix: implement fused IQ3_S MMVQ kernel like ggml.**

### Priority 2: SSM/GDN -- fuse small elementwise ops (saves ~2-3ms)
30 layers x ~6 small kernel launches (gate, beta, alpha, norm, split, gated_out) = 180 launches.
Each is a tiny elementwise op on 2048-4096 elements.
Kernel launch overhead alone: 180 x ~5us = 0.9ms.
**Fix: fuse elementwise chain into 1-2 kernels per GDN layer.**

### Priority 3: attn_norm anomaly -- RESOLVED (profiling artifact)
The 0.243ms `attn_norm` is a profiling artifact: `aprof!` syncs the device but GRAN_PROF
only profiles GDN layer 0 and attn layer 0 (=layer 3). The sync captures pending async
work from GDN layers 1-2. Actual attn_norm is ~0.025ms. No fix needed.
**Real priority 3: batch expert GEMVs** -- concatenate 8 expert inputs into a single
batched GEMM call instead of 8 sequential SGEMV. Saves ~50% MoE latency on small matrices.

### Priority 4: Reduce kernel launch overhead globally (saves ~1ms)
Use CUDA graphs or kernel fusion to batch the ~400+ kernel launches per token
into fewer dispatches.

---

## 5. Theoretical Lower Bound

Model weights for 8 active experts per layer:
- Expert gate: 2048 x 8 x IQ3_S = ~7.3 KB per layer
- Expert up: 2048 x 512 x 8 x IQ3_S = ~3.7 MB per layer
- Expert down: 512 x 2048 x 8 x IQ3_S = ~3.7 MB per layer
- Shared expert: 2048 x 512 x 3 x IQ3_S = ~1.4 MB per layer
- SSM/Attn projections: ~2048 x 2048 x IQ3_S = ~1.9 MB per layer
- Total per layer: ~10.7 MB at IQ3_S (3.4375 bpw)
- 40 layers: ~428 MB per token

RTX 5060 Ti bandwidth: 576 GB/s
Theoretical minimum: 428 MB / 576 GB/s = **0.74 ms/token = 1351 tok/s**

Actual llama.cpp achieves 13.0 ms = 17.5x above theoretical minimum.
This means llama.cpp kernel utilization is ~5.7% of peak bandwidth.
The model is NOT bandwidth-bound -- it's **launch-overhead and occupancy bound**.

This is consistent with the known issue: 256 MoE experts with top-8 routing means
8 tiny GEMVs (2048x512) per layer, too small for good GPU utilization.

---

## 6. Actionable Fix Roadmap (ordered by expected ms saved)

### Fix 1: Fused IQ3_S MMVQ kernel (expected: -4 to -6 ms)
Port ggml's `mul_mat_vec_q` IQ3_S kernel to chimere. Single kernel reads IQ3_S bytes,
dequantizes in shared memory/registers, and accumulates dot product. Eliminates the
write-F32-then-read-F32 round-trip. This is the single biggest win.

### Fix 2: Fuse GDN elementwise ops (expected: -2 to -3 ms)
The GDN SSM path has 6-8 tiny kernel launches per layer (gate, beta, sigmoid, alpha,
norm, expand, gated_out). Fuse them into 1-2 custom CUDA kernels:
- Fuse: beta_sigmoid + alpha_gate + gate_forward into "gating_fused"
- Fuse: split_norm_expand + norm_gated_ssm_out into "ssm_output_fused"
This eliminates ~120 kernel launches (saves ~0.6ms launch overhead) and reduces
intermediate VRAM traffic (saves ~1-2ms bandwidth).

### Fix 3: Batch expert GEMVs (expected: -1 to -2 ms)
Instead of 8 sequential SGEMV calls for the 8 active experts, pad and batch them
into a single batched GEMM: [8, 2048] x [8, 2048, 512] via cuBLAS batched GEMM.
Better GPU utilization for tiny matrices.

### Fix 4: CUDA graphs for layer execution (expected: -0.5 to -1 ms)
Capture the entire layer execution as a CUDA graph. Eliminates per-kernel launch
overhead for the fixed-topology portion of each layer.

### Total expected improvement: -7.5 to -12 ms
Target: 22.72 - 10 = **12.72 ms/token = 78.6 tok/s** (matches llama.cpp)

---

## 7. Key Insight: Per-Layer Budget Comparison

| | chimere (ms/layer) | llama.cpp target (ms/layer) | Ratio |
|--|--------:|--------:|------:|
| GDN (30 layers) | 0.587 | 0.325 | 1.81x |
| Attn (10 layers) | 0.291 | 0.250 | 1.16x |
| MoE FFN (per layer) | 0.378 | 0.215 | 1.76x |

The attention path is already within 16% of llama.cpp. The two bottlenecks are
MoE FFN (1.76x) and GDN SSM (1.81x), both of which are dominated by the same
root cause: tiny kernel launches with low GPU occupancy.
