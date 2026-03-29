# Attention Layer Tensor Operation Audit

## Source: `forward_attn_layer_moe()` in `src/qwen35_model.rs` (lines 2159-2312)

Analysis for **single-token decode** (seq_len=1, the hot path).

Model params for attention layers:
- `num_heads` = 16, `num_kv_heads` = 2, `head_dim` = 256
- `q_head_dim` = 512 (asymmetric Q), `group_size` = 8 (16/2)
- MRoPE: `n_rot` = 64, `head_dim` = 256, sections = [11, 11, 10, 0]

---

## Section 1: Exact Tensor Operation Count

### 1.1 Attention Norm (line 2198)

```rust
let normed = rms_norm(hidden, &w.attn_norm, eps)?;
```

Inside `rms_norm`:
1. `x.contiguous()` -- view check, likely no-op if already contiguous, but still dispatches
2. `nn_ops::rms_norm(&x, weight, eps)` -- **1 fused CUDA kernel**, returns new Tensor

**Ops: 2** (1 contiguous check + 1 fused kernel)
**Allocations: 1** (output tensor [1, 2048])

### 1.2 Q/K/V Projections (lines 2202-2209)

```rust
let q_full = w.wq.forward(&normed)?;   // QMatMul [16384, 2048] x [2048] -> [16384]
let k_proj = w.wk.forward(&normed)?;   // QMatMul [512, 2048]   x [2048] -> [512]
let v_proj = w.wv.forward(&normed)?;   // QMatMul [512, 2048]   x [2048] -> [512]
```

Each `QMatMul::forward` does **2 ops**: dequantize to F32, then matmul.
(Candle QMatMul for quantized weights: dequant + GEMV = 2 kernel dispatches)

**Ops: 6** (3 projections x 2 kernels each)
**Allocations: 6** (3 dequant intermediates + 3 output tensors)

### 1.3 Q+Gate Split and Norms (lines 2213-2223)

```rust
let q_head_dim_x2 = q_full.dim(1)? / num_heads;     // CPU scalar, no tensor op
let q_head_dim = q_head_dim_x2 / 2;                   // CPU scalar
let q = q_full.reshape((1, num_heads, q_head_dim_x2))?;           // 1. reshape (view)
let q = q.narrow(2, 0, q_head_dim)?;                              // 2. narrow (view)
let q_gate_raw = q_full.reshape((1, num_heads, q_head_dim_x2))?   // 3. reshape (view, same as above)
    .narrow(2, q_head_dim, q_head_dim)?;                           // 4. narrow (view)
let q = rms_norm(&q, &w.q_norm, eps)?;     // 5. contiguous + 6. rms_norm
let k = k_proj.reshape((1, num_kv_heads, head_dim))?;  // 7. reshape (view)
let k = rms_norm(&k, &w.k_norm, eps)?;     // 8. contiguous + 9. rms_norm
```

**Ops: 9** (4 view ops + 2 contiguous + 2 rms_norm + 1 reshape)
**Allocations: 4** (q_norm contiguous copy + q_norm output + k contiguous copy + k_norm output)

Note: The `narrow` after `reshape` on interleaved Q+gate data means the narrow is NOT contiguous,
so the `contiguous()` inside `rms_norm` WILL force a copy. These are real allocations.

- q narrow region: [1, 16, 512] BF16 = 16 KB (tiny)
- k reshape: [1, 2, 256] BF16 = 1 KB (tiny)

### 1.4 MRoPE (lines 2226-2228)

```rust
let positions = MRoPE::text_positions(1, state.position);  // CPU Vec, no tensor op
let q_rotated = self.mrope.apply(&q, &positions)?;          // see breakdown below
let k_rotated = self.mrope.apply(&k, &positions)?;          // same
```

**Per MRoPE::apply call** (for text, sections [11, 11, 10, 0]):

For **each of 3 active sections** (s=0,1,2):
1. `x.narrow(2, dim_offset, section_dims)` -- view
2. `x_sec.narrow(2, 0, n_pairs)` -- view (x1)
3. `x_sec.narrow(2, n_pairs, n_pairs)` -- view (x2)
4. `Tensor::from_vec(cos_data, ...)` -- **CPU tensor creation**
5. `.to_dtype(dtype)` -- **CPU->BF16 conversion** (allocation)
6. `.to_device(device)` -- **CPU->GPU transfer** (allocation on GPU)
7. `Tensor::from_vec(sin_data, ...)` -- **CPU tensor creation**
8. `.to_dtype(dtype)` -- CPU->BF16 conversion (allocation)
9. `.to_device(device)` -- CPU->GPU transfer (allocation on GPU)
10. `cos_tok.unsqueeze(1)` -- view
11. `sin_tok.unsqueeze(1)` -- view
12. `cos_tok.expand(...)` -- view (lazy broadcast)
13. `sin_tok.expand(...)` -- view (lazy broadcast)
14. `x1.mul(&cos_tok)` -- **GPU kernel** (allocation)
15. `x2.mul(&sin_tok)` -- **GPU kernel** (allocation)
16. `(14) - (15)` i.e. `sub` -- **GPU kernel** (allocation) -> out1
17. `x1.mul(&sin_tok)` -- **GPU kernel** (allocation)
18. `x2.mul(&cos_tok)` -- **GPU kernel** (allocation)
19. `(17) + (18)` i.e. `add` -- **GPU kernel** (allocation) -> out2
20. `Tensor::cat(&[&out1, &out2], 2)` -- **GPU kernel** (allocation)

After all 3 sections:
21. `x.narrow(2, self.n_rot, pass_through_dims)` -- view (tail: 256-64=192 dims)
22. `Tensor::cat(&rotated_sections, 2)` -- **GPU cat of 4 slices** (allocation)

**Per MRoPE::apply call:**
- Tensor ops: 3*20 + 2 = **62 ops** (but many are views)
- GPU kernel launches: 3*(6 elementwise + 1 cat) + 1 final cat = **22 GPU kernels**
- GPU allocations: 3*(2 device transfers + 6 compute outputs + 1 cat) + 1 final cat = **28 allocations**
- CPU->GPU transfers: 3*2 = **6 transfers per call** (cos + sin for 3 sections)

**For both Q and K MRoPE calls:**
- Tensor ops: **124**
- GPU kernel launches: **44**
- GPU allocations: **56**
- CPU->GPU transfers: **12**

**THIS IS THE DOMINANT SOURCE OF CANDLE DISPATCH OVERHEAD IN ATTENTION.**

### 1.5 KV Cache Update (lines 2231-2247)

```rust
let q_attn = q_rotated.unsqueeze(2)?;                            // 1. view
let k_new = k_rotated.unsqueeze(2)?;                              // 2. view
let v_new = v_proj.reshape((1, num_kv_heads, 1, head_dim))?;      // 3. view

// KV cache cat (non-first-token path):
let k_old_dev = k_old.to_device(k_new.device())?;                 // 4. device transfer (maybe no-op)
let v_old_dev = v_old.to_device(v_new.device())?;                 // 5. device transfer (maybe no-op)
let k_cache = Tensor::cat(&[&k_old_dev, &k_new], 2)?;            // 6. cat (ALLOCATES NEW TENSOR)
let v_cache = Tensor::cat(&[&v_old_dev, &v_new], 2)?;            // 7. cat (ALLOCATES NEW TENSOR)
state.kv_caches[attn_idx] = (k_cache.clone(), v_cache.clone());  // 8-9. clone (Arc increment, no copy)
```

**Ops: 9** (3 views + 2 potential device transfers + 2 cats + 2 clones)
**Allocations: 2** (k_cache and v_cache, both grow by 1 token each step)

At position P, each cat copies P+1 tokens:
- K cache: [1, 2, P+1, 256] BF16 = 2*(P+1)*256*2 bytes
- V cache: same
- At P=100: 2 * 101 * 256 * 2 * 2 = 206 KB (small)
- At P=1000: 2 * 1001 * 256 * 2 * 2 = 2.05 MB (significant)
- At P=4000: 2 * 4001 * 256 * 2 * 2 = 8.2 MB (large, copies entire cache)

**KV cache Tensor::cat allocates a NEW tensor every token and copies the entire old cache.
This is O(P) memory bandwidth per token and grows linearly. This is a major optimization target.**

### 1.6 GQA Expansion + Attention Scores (lines 2249-2262)

```rust
let group_size = num_heads / num_kv_heads;                         // CPU scalar (= 8)
let cached_seq_len = k_cache.dim(2)?;                              // CPU scalar

// K expansion:
let k_exp = k_cache.unsqueeze(2)?                                  // 1. view [1,2,1,P,256]
    .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?  // 2. view (lazy broadcast)
    .contiguous()?                                                  // 3. FULL COPY [1,2,8,P,256] -> ALLOC
    .reshape((1, num_heads, cached_seq_len, head_dim))?;           // 4. view [1,16,P,256]

// V expansion (same pattern):
let v_exp = v_cache.unsqueeze(2)?                                  // 5. view
    .expand((1, num_kv_heads, group_size, cached_seq_len, head_dim))?  // 6. view
    .contiguous()?                                                  // 7. FULL COPY -> ALLOC
    .reshape((1, num_heads, cached_seq_len, head_dim))?;           // 8. view

// Attention scores:
let scores = q_attn.matmul(&k_exp.transpose(2, 3)?)?;             // 9. transpose (view) + 10. matmul -> ALLOC
let scale = 1.0 / (head_dim as f64).sqrt();
let scores = (scores * scale)?;                                     // 11. scalar mul -> ALLOC
```

**Ops: 11** (6 views + 2 contiguous copies + 1 transpose view + 1 matmul + 1 scalar mul)
**Allocations: 4** (k_exp contiguous, v_exp contiguous, matmul result, scaled scores)

Contiguous copy sizes at position P:
- k_exp: [1, 2, 8, P, 256] BF16 = 16 * P * 256 * 2 = 8192*P bytes
  - At P=100: 800 KB
  - At P=1000: 8 MB (!)
  - At P=4000: 32 MB (!!)
- v_exp: same size

**GQA expansion .contiguous() is the second-largest memory overhead, growing as O(P).**

### 1.7 Softmax (line 2265)

```rust
let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;  // 1 fused kernel
```

**Ops: 1**
**Allocations: 1** (attn_weights [1, 16, 1, P])

### 1.8 Attention Output (line 2269)

```rust
let attn_out = attn_weights.matmul(&v_exp)?  // 1. matmul [1,16,1,P] x [1,16,P,256] -> [1,16,1,256]
    .squeeze(2)?;                              // 2. view -> [1,16,256]
```

**Ops: 2** (1 matmul + 1 squeeze view)
**Allocations: 1** (matmul output [1, 16, 1, 256])

### 1.9 Gate + Output Projection + Residual (lines 2272-2277)

```rust
let gate = sigmoid(&q_gate_raw)?;                                     // 1. fused sigmoid kernel
let gated_out = (&attn_out * &gate)?                                  // 2. elementwise mul
    .reshape((1, num_heads * q_head_dim))?;                            // 3. view -> [1, 8192]
let attn_projected = w.wo.forward(&gated_out)?;                       // 4-5. QMatMul (dequant + matmul)
let h_mid = (hidden + &attn_projected)?;                              // 6. residual add
```

**Ops: 6** (1 sigmoid + 1 mul + 1 view + 2 QMatMul kernels + 1 add)
**Allocations: 5** (sigmoid output, mul output, dequant buffer, wo output, residual sum)

### 1.10 MoE FFN (lines 2280-2302)

Not part of attention, but included for completeness:
```rust
let normed_ffn = rms_norm(&h_mid, &w.post_norm, eps)?;  // 2 ops (contiguous + rms_norm)
// ... MoE forward (many more ops, covered by separate analysis)
let h_out = (&h_mid + &ffn_out)?;                        // 1 op (residual add)
```

**Attention-related MoE ops: 3** (contiguous + rms_norm + residual add)

---

## Section 2: Summary Table

| Phase | Line # | Tensor Ops | GPU Kernel Launches | GPU Allocations | Notes |
|---|---|---|---|---|---|
| attn_norm | 2198 | 2 | 1 | 1 | Fused RMSNorm |
| Q/K/V projections | 2202-2209 | 6 | 6 | 6 | QMatMul 2-pass |
| Q+gate split + norms | 2213-2223 | 9 | 2 | 4 | narrow+contiguous+rms_norm |
| **MRoPE (Q)** | **2227** | **62** | **22** | **28** | **CPU cos/sin + 12 GPU elementwise + 4 cats** |
| **MRoPE (K)** | **2228** | **62** | **22** | **28** | **Same pattern, smaller tensors** |
| KV cache update | 2231-2247 | 9 | 2 | 2 | Tensor::cat grows O(P) |
| GQA expand + scores | 2249-2262 | 11 | 4 | 4 | .contiguous() copies O(P) |
| Softmax | 2265 | 1 | 1 | 1 | Fused |
| Attention output | 2269 | 2 | 1 | 1 | matmul + squeeze |
| Gate + o_proj + residual | 2272-2277 | 6 | 5 | 5 | sigmoid, mul, QMatMul, add |
| **TOTAL (attention only)** | | **170** | **66** | **80** | Per attention layer |

### Per attention layer: **170 Candle tensor operations, 66 GPU kernel launches, 80 GPU memory allocations.**

### For 10 attention layers: **1700 tensor ops, 660 GPU kernel launches, 800 GPU allocations.**

---

## Section 3: Which Operations Allocate GPU Memory

### Heavyweight allocations (data-dependent, grow with sequence length P)

| Operation | Size at P=100 | Size at P=1000 | Size at P=4000 | Growth |
|---|---|---|---|---|
| KV cache cat (K) | 103 KB | 1.02 MB | 4.1 MB | O(P) per token |
| KV cache cat (V) | 103 KB | 1.02 MB | 4.1 MB | O(P) per token |
| GQA expand K (.contiguous) | 800 KB | 8.0 MB | 32 MB | O(P) per token |
| GQA expand V (.contiguous) | 800 KB | 8.0 MB | 32 MB | O(P) per token |
| Attention scores matmul | 12.8 KB | 128 KB | 512 KB | O(P) per token |
| Softmax output | 12.8 KB | 128 KB | 512 KB | O(P) per token |

**At P=1000, the GQA expansion alone copies 16 MB x 10 layers = 160 MB per token.**
**At P=4000, it's 640 MB per token just for GQA expansion copies.**

### Fixed-size allocations (independent of sequence length)

| Operation | Count per layer | Size each | Notes |
|---|---|---|---|
| MRoPE cos/sin CPU->GPU | 12 | ~88 bytes each | Tiny but 12 PCIe transfers! |
| MRoPE elementwise (mul, sub, add) | 12 per MRoPE call x 2 | ~1-4 KB | Many tiny GPU kernels |
| MRoPE cat (per section + final) | 4 per MRoPE call x 2 | ~1-8 KB | |
| QMatMul dequant buffers | 4 | 8-64 KB | Intermediate F32 |
| RMSNorm outputs | 4 | 4-16 KB | |
| Sigmoid, mul, add | 4 | 4-16 KB | |

---

## Section 4: Estimated ms Saved by Optimization

### Current: ~3.7 ms for 10 attention layers (GRAN_PROF measured)

Theoretical bandwidth cost for attention weights: ~1.0 ms (343.8 MB / 350 GB/s).
**Gap: 2.7 ms = pure Candle dispatch + allocation overhead.**

| Optimization | Expected Savings | Difficulty | Priority |
|---|---|---|---|
| **1. Precompute MRoPE cos/sin on GPU** | **0.8 ms** | Low | **P0** |
| Eliminate 12 CPU->GPU transfers per layer, 120 total. Replace 44 GPU kernels per layer with 2 (fused rotation). Precompute for all positions up to max_seq_len at model load. |
| **2. Pre-allocated KV cache ring buffer** | **0.4 ms** | Medium | **P1** |
| Replace `Tensor::cat` (copies entire cache) with pre-allocated buffer + index. At P=1000, saves 2 MB copy x 10 layers = 20 MB of unnecessary copies. At P=4000, saves 80 MB. |
| **3. Eliminate GQA .contiguous() via repeat_interleave** | **0.6 ms** | Medium | **P1** |
| The `expand().contiguous()` copies the full expanded KV cache. Alternative: use `repeat_interleave` on the head dim, or better yet, modify the matmul to work with strided/broadcast tensors directly. At P=1000, saves 16 MB x 10 layers. Or use ggml-style repeated KV pattern in a custom kernel. |
| **4. Fuse Q+K norm into single kernel** | **0.1 ms** | Low | **P2** |
| Two separate rms_norm calls for Q and K. Could be a single kernel that operates on both. Saves 1 kernel launch per layer. |
| **5. Fuse gate sigmoid + multiply** | **0.1 ms** | Low | **P2** |
| `sigmoid(gate) * attn_out` is 2 kernels. Fuse into 1 `x * sigmoid(g)` kernel. |
| **6. Use raw Q5K GEMV for wq/wk/wv/wo** | **0.4 ms** | Medium | **P1** |
| Replace QMatMul 2-pass (dequant+matmul) with fused MMVQ-style 1-pass. Saves 4 kernel launches per layer. |
| **7. Convert all views to raw pointer offsets** | **0.3 ms** | High | **P3** |
| Many reshape/narrow/unsqueeze ops go through Candle dispatch even though they're just metadata. In a raw path, these are free (pointer + stride arithmetic). |
| **TOTAL potential savings** | **~2.7 ms** | | |

**Estimated post-optimization: ~1.0 ms for 10 attention layers (matching bandwidth theory).**

---

## Section 5: Priority Implementation Order

### P0: MRoPE Precomputation (saves 0.8 ms, low difficulty)

The MRoPE implementation creates cos/sin tensors **on CPU**, converts them to BF16, transfers to
GPU, then does 6 elementwise operations per section, all **per token per layer**. For 10 attention
layers, that's 120 CPU->GPU transfers and 440 GPU kernel launches just for position encoding.

**Fix**: At model load, precompute `[max_seq_len, n_rot/2]` cos and sin tensors on GPU in BF16.
At inference, slice with `narrow(0, position, 1)` (a free view). Then the rotation becomes:
- 2 narrow (views, free)
- 4 elementwise (mul, sub, mul, add)
- 1 cat

Per MRoPE call: 7 ops instead of 62. For both Q and K: 14 ops instead of 124.
**Saves 110 Candle tensor ops per layer, 1100 total.**

### P1a: Pre-allocated KV Cache (saves 0.4 ms, medium difficulty)

Replace:
```rust
let k_cache = Tensor::cat(&[&k_old, &k_new], dim)?;  // copies P tokens + 1
```
With:
```rust
k_buffer[position] = k_new;  // writes 1 token, no copy
let k_cache = k_buffer.narrow(seq_dim, 0, position + 1);  // view, free
```

Requires pre-allocating `[1, num_kv_heads, max_seq_len, head_dim]` buffers per layer.
10 layers x 2 (K+V) x max_seq_len x 2 x 256 x 2 bytes.
At max_seq_len=4096: 10 * 2 * 4096 * 2 * 256 * 2 = 80 MB (acceptable).

### P1b: Eliminate GQA .contiguous() (saves 0.6 ms, medium difficulty)

The expand+contiguous pattern:
```rust
k.unsqueeze(2).expand(...).contiguous().reshape(...)
```
creates a full copy of the expanded KV cache (8x the original for group_size=8).

**Option A**: Custom CUDA kernel for GQA matmul that handles the repeat pattern internally
(like ggml's `ggml_mul_mat` with `ne02 != ne12`). The matmul reads each KV head 8 times
from the same memory location instead of copying.

**Option B**: Use `Tensor::repeat` instead of `expand+contiguous` - same effect but might
be slightly more efficient in Candle's backend.

**Option C** (best): In a raw CUDA path, the Q*K^T matmul can be written to index K with
`head_idx / group_size` instead of `head_idx`, eliminating the expansion entirely.

### P2: Fuse small ops (saves 0.2 ms combined)

Low priority, small wins. Fuse QK-norms and gate-sigmoid.

### P3: Raw path for views (saves 0.3 ms)

Convert all reshape/narrow/unsqueeze/transpose to raw pointer+stride math in the
eventual raw attention path. This is implicit in any "raw forward" rewrite.

---

## Section 6: Key Finding -- MRoPE is 73% of Attention Dispatch Overhead

| Component | Tensor Ops | % of 170 total |
|---|---|---|
| MRoPE (Q + K) | 124 | **73%** |
| Everything else | 46 | 27% |

The MRoPE implementation accounts for **124 out of 170 tensor operations** per attention layer.
It creates 56 GPU allocations and performs 12 CPU-to-GPU transfers. This single function is
responsible for the majority of the Candle dispatch overhead in the attention path.

Precomputing cos/sin on GPU at model load would reduce MRoPE from 124 ops to ~14 ops (89% reduction),
bringing the total per-layer attention ops from 170 down to ~60, which at ~2 us/op = 120 us per layer
= 1.2 ms for 10 layers -- much closer to the 1.0 ms bandwidth theoretical minimum.
