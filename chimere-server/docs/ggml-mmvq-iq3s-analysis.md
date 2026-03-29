# ggml MMVQ IQ3_S Kernel -- Deep Technical Analysis

## chimere-deltanet -- 2026-03-16

This document is a line-by-line analysis of how ggml (both stock llama.cpp and
ik_llama.cpp) implements the fused dequant+dot-product GEMV kernel for IQ3_S
weights against Q8_1-quantized activations, and how it compares to our current
chimere kernel. The goal is to produce a precise port plan with expected gains.

**Source files analyzed:**

| File | Location |
|------|----------|
| Stock MMVQ kernel | `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` |
| Stock vec_dot IQ3_S | `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` lines 1068-1105 |
| Stock type traits | `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/common.cuh` line 1026 |
| Stock Q8_1 quantization | `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/quantize.cu` lines 4-48 |
| Stock data structures | `/home/remondiere/llama.cpp/ggml/src/ggml-common.h` |
| ik_llama MMVQ template | `/home/remondiere/ik_llama.cpp/ggml/src/ggml-cuda/mmvq-templates.cuh` |
| ik_llama IQ3_S instance | `/home/remondiere/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/mmvq-instance-iq3_s.cu` |
| Our kernel | `/home/remondiere/.chimere/workspaces/chimere/chimere-deltanet/src/kernels/iq3s_gemv.rs` |
| Previous analysis | `/home/remondiere/.chimere/workspaces/chimere/chimere-deltanet/docs/ggml-mmvq-port-plan.md` |

---

## 1. ggml MMVQ Algorithm -- Step by Step

### 1.1 The Two-Phase Pipeline

ggml never dequantizes IQ3_S weights to f32. Instead, it uses a **two-phase pipeline**:

**Phase 1: Quantize activations to Q8_1** (separate kernel, launched before MMVQ)

```
quantize_row_q8_1_cuda() -> quantize_q8_1<<<grid, 256>>>
```

Each of the 256 threads in a block handles one element. Groups of 32 threads
(one warp) cooperate via `warp_reduce_max<32>` and `warp_reduce_sum<32>` to find
the scale `d` and sum `s` for a single Q8_1 block.

Result: `block_q8_1` structs in GPU global memory (36 bytes each: 4 bytes `ds`
as half2 + 32 bytes `qs` as int8).

**Phase 2: Fused dot product** (the MMVQ kernel)

```
mul_mat_vec_q<GGML_TYPE_IQ3_S, 1, false><<<(nrows, 1, 1), (32, 4, 1)>>>
```

Each thread block computes one output row. Within each block, 128 threads (4
warps) cooperate to compute the dot product of one IQ3_S weight row against the
Q8_1-quantized activation vector. The dot product is computed in integer
arithmetic (dp4a) and only converted to float at the very end.

### 1.2 Compile-Time Constants for IQ3_S + ncols_dst=1

These are the values that the C++ templates resolve to at compile time:

```
QK_K               = 256     elements per IQ3_S super-block
QK8_1              = 32      elements per Q8_1 block
QR3_S              = 4
QI3_S              = QK_K / (4 * QR3_S) = 16
VDR_IQ3_S_Q8_1_MMVQ = 2     vector dot ratio
nwarps             = 4       (MMVQ_PARAMETERS_GENERIC table, ncols_dst <= 4)
warp_size          = 32
rows_per_cuda_block = 1      (GENERIC, ncols_dst=1)
blocks_per_iter    = VDR * nwarps * warp_size / QI3_S
                   = 2 * 4 * 32 / 16 = 16
```

### 1.3 Thread-to-Work Mapping

The 128 threads are assigned to IQ3_S blocks using this formula
(mmvq.cu line 244, 248):

```c
for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kqs = vdr * (tid % (qi/vdr));
```

With qi=16, vdr=2: `qi/vdr = 8`. So:

- `tid / 8` = which IQ3_S block this thread starts on (0..15 in first iteration)
- `tid % 8` = which sub-group within the block (0..7)
- `kqs = 2 * (tid % 8)` = iqs parameter: 0, 2, 4, 6, 8, 10, 12, 14

**8 consecutive threads share one IQ3_S block.** Each thread processes 32
elements (one sub-group) via the `vec_dot_iq3_s_q8_1` function. Together, 8
threads cover all 256 elements of the super-block.

**Per iteration: 128 threads / 8 threads-per-block = 16 IQ3_S blocks processed.**
This matches `blocks_per_iter = 16`.

**Stride across iterations:** `kbx += 16`. For a 2048-element row with
2048/256 = 8 IQ3_S blocks, only threads 0-63 do any work (8 blocks x 8
threads). The remaining 64 threads contribute zero. For wider layers (e.g.,
4096-element ffn_up with 16 IQ3_S blocks), exactly one iteration is needed.

### 1.4 The Core: `vec_dot_iq3_s_q8_1` (vecdotq.cuh:1068-1105)

This is the fused dequant+dot kernel. It runs per-thread, processing 32
elements (one sub-group of one IQ3_S super-block) in pure integer arithmetic:

```c
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq,
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx,
    const int & iqs)
{
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;
```

**Step A: Vectorized data loading (2 loads instead of 8 byte loads)**

```c
    // Load 8 qs bytes as two int32 via uint16_t-pair loads
    const int2 qs_packed = make_int2(
        get_int_b2(bq3->qs, iqs + 0),    // 4 bytes: qs[4*iqs .. 4*iqs+3]
        get_int_b2(bq3->qs, iqs + 1));   // 4 bytes: qs[4*(iqs+1) .. 4*(iqs+1)+3]
    const uint8_t * qs = (const uint8_t *) &qs_packed;  // reinterpret as 8 bytes

    const int qh = bq3->qh[iqs/2];      // 1 high-bit byte

    // Load 4 sign bytes as one int32
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;
```

`get_int_b2(ptr, i)` loads `ptr[2*i]` and `ptr[2*i+1]` as uint16_t, then
combines them into a single int32. This compiles to two `ld.global.u16`
instructions, which the memory subsystem can coalesce when threads in a warp
access consecutive addresses.

**Step B: Unrolled 4-iteration loop (32 elements = 8 grid lookups = 4 pairs)**

```c
    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
```

Each iteration processes 2 grid entries = 8 elements:

```c
        // Grid lookup: combine 8-bit qs index with 1-bit qh extension
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0+0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0+1] | ((qh << (7 - l0)) & 0x100)]);
```

Each grid lookup uses a 9-bit index (8-bit qs + 1-bit qh) into the 512-entry
`iq3s_grid` table. The result is a packed uint32 containing 4 unsigned bytes
(the dequantized magnitudes, each in {1, 3, 5, 7, 9, 11, 13, 15}).

```c
        // Sign decoding: expand 2-bit packed signs into 4-byte sign masks
        const int signs0 = __vcmpne4(
            ((signs_packed_8[l0/2] & 0x03) << 7) |
            ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(
            ((signs_packed_8[l0/2] & 0x30) << 3) |
            ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
```

`__vcmpne4(a, b)` compares 4 bytes independently, producing 0xFF where a != b
and 0x00 where a == b. This converts 2-bit packed sign data into per-byte
0xFF/-1 or 0x00 masks.

```c
        // Apply signs: XOR + subtract implements conditional negation
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
```

`__vsub4(a ^ mask, mask)` is the standard two's-complement negation trick:
when mask=0xFF, this computes `(a ^ 0xFF) - 0xFF = ~a - (-1) = ~a + 1 = -a`.
When mask=0x00, it's a no-op. Result: signed int8x4 packed in an int32.

```c
        // Dot product: int8x4 * int8x4 with accumulation
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);  // 4 Q8_1 values
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);  // 4 Q8_1 values

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);  // __dp4a
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
```

`__dp4a(a, b, c)` = `c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]`
where a and b are int8x4 packed in int32. This is a single hardware instruction
on sm_61+ (including sm_120).

**Step C: Scale application**

```c
    // Per-subgroup 4-bit scale (packed 2 per byte)
    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
```

The IQ3_S scale is a 4-bit value (0..15) packed in pairs. The effective scale
multiplier is `1 + 2*s` = {1, 3, 5, ..., 31}.

**Step D: Float conversion (only at the very end)**

```c
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
```

`__half2float` converts the IQ3_S super-block f16 scale to f32.
`__low2float` extracts the low half from the Q8_1 `ds` half2 pair (the `d`
scale, not the `s` sum).

**Key insight: the ENTIRE inner loop runs in integer arithmetic.** Float
conversion happens exactly once, after all 32 elements have been accumulated.
This minimizes the use of f32 registers and maximizes throughput of the integer
pipeline (dp4a is 1 instruction per 4 multiply-accumulates).

### 1.5 Reduction: Warp Shuffle + Shared Memory

After the main loop, each of 128 threads holds a partial sum covering
different sub-groups and different IQ3_S blocks. The reduction proceeds in
two stages (mmvq.cu lines 266-351):

**Stage 1: Cross-warp via shared memory**

```c
__shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];
// For IQ3_S ncols_dst=1: float tmp_shared[3][1][1][32] = 384 bytes

// Warps 1-3 write their partials into shared memory
if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y-1][0][0][threadIdx.x] = tmp[0][0];
}
__syncthreads();
if (threadIdx.y > 0) return;  // Only warp 0 continues
```

**Stage 2: Warp 0 accumulates all partials, then butterfly reduction**

```c
// Warp 0: accumulate shared memory partials
for (int l = 0; l < nwarps-1; ++l) {
    tmp[0][0] += tmp_shared[l][0][0][threadIdx.x];
}
// Butterfly warp reduction
tmp[0][0] = warp_reduce_sum<warp_size>(tmp[0][0]);

// Thread 0 writes result
if (threadIdx.x == 0) {
    dst[row] = tmp[0][0];
}
```

`warp_reduce_sum<32>` is a standard butterfly reduction using `__shfl_xor_sync`
with offsets 16, 8, 4, 2, 1 -- 5 shuffle instructions.

### 1.6 Grid Lookup Table Storage

In the stock ggml CUDA backend, `iq3s_grid` is declared via:

```c
GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)
```

which on CUDA expands to:

```c
static const __device__ uint32_t iq3s_grid[512] = { ... };
```

This is `__device__` memory (NOT `__constant__`). The CUDA compiler may
cache it in the read-only / constant cache depending on access patterns.
512 entries x 4 bytes = 2048 bytes = 2 KB.

Our chimere kernel uses `__constant__` explicitly, which guarantees the data
goes through the constant cache (8 KB per SM on sm_120). For a 2 KB table
with broadcast access pattern (many threads reading the same entry), this is
slightly better than `__device__` -- but the difference is negligible since
the constant cache hit rate is high either way.

### 1.7 The ik_llama Variant

ik_llama's MMVQ for stock IQ3_S (i.e., GGML_TYPE_IQ3_S, NOT their custom
IQ3_K) uses **exactly the same algorithm** as stock llama.cpp. The dispatch
path is:

```
mmvq.cu: case GGML_TYPE_IQ3_S -> mul_mat_vec_iq3_s_q8_1_cuda()
mmvq-instance-iq3_s.cu: calls mul_mat_vec_q_cuda<GGML_TYPE_IQ3_S>()
mmvq-templates.cuh: template instantiation -> same vec_dot_iq3_s_q8_1
```

The only differences from stock are:

1. **ik_llama adds fused gate+bias+SwiGLU** support (`fused_mul_mat_vec_q`
   template in mmvq-templates.cuh lines 152-272), which fuses the MoE gate
   projection + up projection + SwiGLU activation into a single kernel launch.

2. **Warp count selection** (mmvq-templates.cuh line 438): ik_llama uses
   `nwarps = args.ncols_y <= 4 ? 4 : 2` when ne2 < 2 on NVIDIA. Same as
   stock for ncols_dst=1.

3. **Block dims:** 2D like stock: `(WARP_SIZE, nwarps, 1)` = `(32, 4, 1)`.

The IQ3_S dot product function is identical -- same `vec_dot_iq3_s_q8_1` from
the shared `vecdotq.cuh`.

**The +23% TG advantage of ik_llama over stock comes from other optimizations**
(improved attention kernels, MoE scheduling, fused ops) -- NOT from the MMVQ
kernel itself, which is identical for IQ3_S.

---

## 2. Key Differences from Our Current Kernel

### 2.1 Side-by-Side Comparison Table

| Aspect | ggml MMVQ | chimere current | Performance impact |
|--------|-----------|-----------------|-------------------|
| **Launch bounds** | `__launch_bounds__(128, 1)` | `__launch_bounds__(128)` (no minblocks) | **MEDIUM**: compiler may over-spill without the "1" hint |
| **Block dims** | `(32, 4, 1)` 2D | `(128, 1, 1)` flat | Equivalent 128 threads, but 2D enables `threadIdx.y` for warp_id |
| **Threads per IQ3_S block** | 8 (cooperative) | 8 (cooperative, same mapping) | **Same** -- our `iqs = 2*(tid&7)` matches ggml |
| **qs load** | `get_int_b2`: 2x u16 = u32 | `get_int_b2`: 2x u16 = u32 | **Same** -- we already ported this |
| **signs load** | `get_int_b2(signs, iqs/2)` = u32 | `get_int_b2(bq3+74, iqs/2)` = u32 | **Same** -- we already ported this |
| **Q8_1 qs load** | `get_int_b4(bq8_1[].qs, l0)` = aligned u32 | `get_int_b4(bq8+4, l0)` = aligned u32 | **Same** |
| **Grid LUT** | `static const __device__` (global mem) | `__constant__` | **Ours slightly better** (guaranteed constant cache) |
| **Sign decode** | `__vcmpne4` + `__vsub4` | `__vcmpne4` + `__vsub4` | **Same** |
| **Dot product** | `ggml_cuda_dp4a` = `__dp4a` | `__dp4a` | **Same** |
| **f16 decode** | `__half2float` intrinsic | `f16_to_f32` via PTX asm `cvt.f32.f16` | **Ours is equivalent** (PTX asm compiles to the same instruction) |
| **Reduction** | shm[3][1][1][32] + warp shuffle | shm warp_sums[4] + warp shuffle | **DIFFERENT** (see 2.2) |
| **Shared memory** | 384 bytes | 16 bytes | **DIFFERENT** (see 2.2) |
| **Output write** | `threadIdx.x == 0` writes | `tid == 0` writes | **Same** |
| **Loop stride** | `kbx += blocks_per_iter (16)` | `kbx += blocks_per_iter (16)` | **Same** |

### 2.2 The Reduction Difference

**ggml's reduction** (from `mmvq.cu`):

```c
// Stage 1: Warps 1-3 dump 32 partial sums each into shared memory
__shared__ float tmp_shared[3][1][1][32];  // 384 bytes
if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y-1][0][0][threadIdx.x] = partial;
}
__syncthreads();
if (threadIdx.y > 0) return;

// Stage 2: Warp 0 gathers and reduces
for (l = 0; l < 3; l++) {
    partial += tmp_shared[l][0][0][threadIdx.x];
}
partial = warp_reduce_sum<32>(partial);  // 5 shuffles

if (threadIdx.x == 0) dst[row] = partial;
```

Total operations: 1 syncthreads + 3 shared loads + 5 shuffles = **9 operations**.
Latency: ~20 cycles (dominated by syncthreads and shared memory).

**Our reduction** (from `iq3s_gemv.rs`):

```c
// Stage 1: Each warp's lane 0 writes one float to shared memory
if (lane_id == 0) {
    warp_sums[warp_id] = sum;  // 4 floats in shared memory
}
__syncthreads();

// Stage 2: Thread 0 loads all 4 and sums
if (tid == 0) {
    float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    output[row] = total;
}
```

But **we do the FULL warp reduction BEFORE writing to shared memory**, using
5 shuffles to get each warp's total into lane 0. This means our reduction is:

- Per warp: 5 shuffles (within-warp reduction to lane 0)
- Sync: 1 syncthreads
- Thread 0: 4 shared loads + 3 adds

Total: **5 shuffles + 1 syncthreads + 4 shared loads + 3 adds = 13 operations**.

**ggml's approach is smarter.** It does NOT reduce within each warp first.
Instead, it dumps ALL 32 partial sums per warp into shared memory (32 floats
per warp, 96 floats total), then warp 0 reads all 96 values, adds them to
its own 32 partials, and THEN does the single warp reduction.

This is better because:
1. The shared memory writes/reads are **fully coalesced** (32 consecutive floats).
2. The within-warp reduction happens only ONCE (for warp 0), not 4 times.
3. Total shuffles: 5 (only in warp 0) vs 20 (5 per warp x 4 warps in our approach).
4. Total shared memory operations: 96 writes + 96 reads = 192 vs 4 writes + 4 reads = 8.

**Net impact:** ggml trades 184 more shared memory operations for 15 fewer
shuffle operations. On sm_120, shared memory throughput is 128 bytes/cycle/SM
while shuffles cost ~2 cycles each. For 384 bytes of shared memory operations
at 128 B/cycle, that's ~3 cycles. The 15 saved shuffles are worth ~30 cycles.
**Net gain: ~27 cycles per block** = ~0.05 us at 600 MHz. Marginal.

### 2.3 What We Already Got Right

After the perf sprint on 2026-03-15, our kernel already matches ggml on:

- Thread mapping (8 threads per IQ3_S block, `kbx = tid/8`, `iqs = 2*(tid%8)`)
- Vectorized loads (`get_int_b2`, `get_int_b4`)
- dp4a dot product
- Sign decoding (`__vcmpne4` + `__vsub4`)
- Scale application
- Grid LUT (512 entries in constant memory)
- 4-warp, 128-thread configuration
- `__launch_bounds__(128)`

### 2.4 What We Still Differ On

1. **`__launch_bounds__(128, 1)` vs `__launch_bounds__(128)`**: The missing
   second argument "1" tells the compiler "at least 1 block per SM" which
   unlocks up to 512 registers per thread. Without it, the compiler targets
   higher occupancy and may spill registers. This is likely our biggest
   remaining gap.

2. **Reduction strategy**: As analyzed in 2.2, ggml's reduction saves ~15
   shuffle instructions by dumping all per-lane partials to shared memory
   instead of reducing per-warp first. Impact: ~27 cycles = ~0.05 us.

3. **2D block dims**: ggml uses `(32, 4, 1)` block dims, giving natural
   `threadIdx.y` for warp_id. We use `(128, 1, 1)` and compute
   `warp_id = tid >> 5`. This is functionally identical but may interact
   differently with the compiler's register allocator.

4. **Shared memory sizing**: ggml uses 384 bytes
   `(3 * 1 * 1 * 32 * sizeof(float))` vs our 16 bytes. The ggml pattern
   is more efficient despite using more shared memory (see 2.2).

---

## 3. Port Plan -- Exact Changes Needed

### 3.1 Change `__launch_bounds__`

**Current:**
```c
extern "C" __global__ __launch_bounds__(128)
void gemv_iq3s_q8(...)
```

**Port to:**
```c
extern "C" __global__ __launch_bounds__(128, 1)
void gemv_iq3s_q8(...)
```

Apply to ALL kernel variants: `gemv_iq3s_q8`, `gemv_iq3s_q8_batched`,
`gemv_iq3s_q8_batched_multi_input`.

### 3.2 Port the ggml Reduction Strategy

Replace the current two-stage reduction (per-warp shuffle + thread-0 sum)
with ggml's approach (shared memory dump + single warp reduction).

**Current (in each kernel variant):**
```c
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    // Cross-warp reduction via shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (tid == 0) {
        float total = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
        output[row] = total;
    }
```

**Replace with:**
```c
    // Cross-warp reduction: ggml pattern
    // Shared memory: 3 x 32 floats = 384 bytes
    // (warps 1-3 each dump their 32-lane partials)
    __shared__ float tmp_shared[3][32];

    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    if (warp_id > 0) {
        tmp_shared[warp_id - 1][lane_id] = sum;
    }
    __syncthreads();
    if (warp_id > 0) return;

    // Warp 0: accumulate partials from warps 1-3
    #pragma unroll
    for (int w = 0; w < 3; w++) {
        sum += tmp_shared[w][lane_id];
    }

    // Single warp reduction (butterfly)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        output[row] = sum;
    }
```

Update the Rust-side `shared_mem_bytes` from 16 to **384** in all launch configs.

### 3.3 Optionally Switch to 2D Block Dims

This is optional since it's functionally equivalent, but for cleanliness:

**Rust side:**
```rust
// Current:
block_dim: (128, 1, 1),

// Optional:
block_dim: (32, 4, 1),
```

If we switch to 2D, update the kernel to use `threadIdx.y` for warp_id
and `threadIdx.x` for lane_id instead of computing from flat `tid`. This
may help the compiler's register allocator slightly.

### 3.4 No Other Changes Needed

The dot product function `dot_iq3s_q8_ggml`, the main loop structure, and
all data loading patterns already match ggml. The existing analysis in
`ggml-mmvq-port-plan.md` correctly identified the vectorized loads and thread
mapping issues, and those have been fixed.

---

## 4. Expected Gain

### 4.1 Current Performance Baseline

From the perf sprint on 2026-03-15:
- Our GEMV: ~5-6 us per 512x2048 call (after dp4a port)
- ggml/ik_llama MMVQ: ~4 us per equivalent call
- Gap: ~1.5-2x remaining

### 4.2 Impact of Each Change

| Change | Expected impact | Confidence |
|--------|----------------|------------|
| `__launch_bounds__(128, 1)` | 10-20% speedup from reduced register spill | High |
| ggml reduction strategy | 2-5% from fewer shuffles | Medium |
| 2D block dims | 0-2% from compiler hints | Low |

### 4.3 Projected Numbers

With `__launch_bounds__(128, 1)`:
- Current ~5.5 us -> ~4.5-5.0 us per GEMV call
- This alone closes most of the gap

With all changes:
- Target ~4.5 us per GEMV call (matching ggml within ~10%)

**Remaining gap explanation (if any):**
1. Static NVCC compilation can do whole-program optimization that NVRTC cannot.
2. ggml's Q8_1 quantization kernel uses 32-thread cooperative quantization
   vs our 1-thread-per-block approach. While not on the GEMV critical path,
   the better Q8_1 quantization may produce slightly better cache behavior.
3. ggml's template instantiation allows the compiler to inline everything with
   known compile-time constants. NVRTC with `#define` macros should be
   equivalent, but there may be edge cases.

### 4.4 Model-Level Impact

For a 40-layer Qwen3.5-35B-A3B model with 8 active experts per token:
- MoE GEMVs per token: ~960 calls (8 experts x 3 projections x 40 layers)
- At 5.5 us: 960 x 5.5us = 5.28 ms in MoE GEMVs
- At 4.5 us: 960 x 4.5us = 4.32 ms in MoE GEMVs
- **Savings: ~0.96 ms per token** (~4% of total ~24 ms per token)
- **From ~42 tok/s to ~43.5 tok/s** (MoE improvement only)

The gain is modest because the GEMV is already well-optimized after the
perf sprint. The bigger gains (closing the full 4.4x gap with ggml) require
addressing launch overhead, which is ~93% of measured GEMV time and is an
orthogonal problem (batch multiple GEMVs into a single launch, or use
persistent kernels).

---

## 5. Why ggml's Approach is Faster Than a 2-Pass Approach

The original chimere design used a **2-pass approach**:
1. Pass 1: Dequantize IQ3_S to f32 in GPU memory
2. Pass 2: Multiply f32 weights by f32/Q8_1 activations

ggml's fused MMVQ avoids this by:

### 5.1 Memory Bandwidth Elimination

- **2-pass:** Writes 256 x 4 bytes = 1 KB of f32 per IQ3_S block to global
  memory, then reads it back. For a 2048-element row, that's 8 KB written
  and 8 KB read = 16 KB of extra global memory traffic.
- **Fused:** The dequantized values only exist in registers (as int8x4 packed
  in int32). Zero extra memory traffic.

At 512 GB/s (RTX 5060 Ti), 16 KB takes ~31 ns. For 960 GEMV calls per token,
that's ~30 us saved. But the real cost is higher because the dequant kernel
has its own launch overhead (~3-5 us per launch).

### 5.2 Reduced Kernel Launch Overhead

- **2-pass:** 2 kernel launches per GEMV (dequant + GEMM)
- **Fused:** 1 kernel launch per GEMV

On sm_120, kernel launch overhead is ~3-5 us via the CUDA driver API. Saving
one launch per GEMV is significant when the GEMV itself is ~4-5 us.

### 5.3 Integer Arithmetic Throughput

The fused kernel keeps everything in integer until the final scale multiply:
- `__dp4a`: 1 instruction for 4 multiply-adds (int8x4 x int8x4 -> int32)
- `__vcmpne4`, `__vsub4`: SIMD byte operations for sign application
- `iq3s_grid` lookup: returns packed int8x4, no float conversion needed

An f32 approach would need:
- f16->f32 conversion for every dequantized weight (256 per block)
- f32 fused-multiply-add for the dot product (256 FMAs per block)
- The fp32 FMA throughput on sm_120 is 1/warp/cycle for 32 elements, while
  dp4a achieves 4 multiply-adds per instruction = 4x more arithmetic density.

### 5.4 Register Pressure

- **2-pass f32:** The dequant buffer must live somewhere. If in registers,
  that's 256 x 4 bytes = 1 KB per thread. If in shared memory, it limits
  occupancy. If in global memory, it adds latency.
- **Fused int:** The entire state for one sub-group is: 2 int32 for qs_packed,
  1 int for qh, 1 int for signs, 1 int for sumi, plus loop variables. Total:
  ~10 registers. This leaves plenty of room for instruction-level parallelism.

---

## Appendix A: Complete IQ3_S Block Layout

```
Offset  Bytes  Field         Sub-groups covered
  0       2    d             Super-block f16 scale (1 per block)
  2      64    qs[64]        8 bytes per sub-group (8 x qs indices)
 66       8    qh[8]         1 byte per sub-group (high bits)
 74      32    signs[32]     4 bytes per sub-group (32 sign bits)
106       4    scales[4]     1 nibble per sub-group (packed in pairs)
-----
110 bytes total for 256 elements = 3.4375 bpw
```

Each sub-group = 32 elements. 8 sub-groups per super-block.

Thread iqs=N processes sub-group N (0..7), covering:
- `qs[8*N .. 8*N+7]`
- `qh[N]`
- `signs[4*N .. 4*N+3]`
- `scales[N/2]` (shared between pairs, selected by nibble shift)
- `bq8_1[N]` (the corresponding Q8_1 block)

## Appendix B: Q8_1 Block Layout

```
Offset  Bytes  Field
  0       4    ds          half2: (.x = d = scale, .y = s = d * sum(qs))
  4      32    qs[32]      int8 quantized values
-----
36 bytes total for 32 elements
```

One IQ3_S super-block (256 elements) maps to 8 Q8_1 blocks (256/32 = 8).

## Appendix C: Thread Scheduling Visualization (128 threads, 8 IQ3_S blocks)

For a row with 2048 columns = 8 IQ3_S blocks:

```
Iteration 0 (only iteration needed for 8 blocks):
  Threads  0- 7: IQ3_S block 0, iqs = 0,2,4,6,8,10,12,14
  Threads  8-15: IQ3_S block 1, iqs = 0,2,4,6,8,10,12,14
  Threads 16-23: IQ3_S block 2, iqs = 0,2,4,6,8,10,12,14
  Threads 24-31: IQ3_S block 3, iqs = 0,2,4,6,8,10,12,14
  Threads 32-39: IQ3_S block 4, iqs = 0,2,4,6,8,10,12,14
  Threads 40-47: IQ3_S block 5, iqs = 0,2,4,6,8,10,12,14
  Threads 48-55: IQ3_S block 6, iqs = 0,2,4,6,8,10,12,14
  Threads 56-63: IQ3_S block 7, iqs = 0,2,4,6,8,10,12,14
  Threads 64-127: kbx = 8..15 > blocks_per_row(8), IDLE (partial = 0)
```

For a row with 4096 columns = 16 IQ3_S blocks:

```
Iteration 0:
  Threads  0- 7: IQ3_S block  0
  Threads  8-15: IQ3_S block  1
  ...
  Threads 120-127: IQ3_S block 15
  (All 128 threads active, exactly one iteration)
```

For a row with 8192 columns = 32 IQ3_S blocks:

```
Iteration 0: Threads process blocks  0-15 (128 threads / 8 = 16 blocks)
Iteration 1: Threads process blocks 16-31
```

## Appendix D: ggml Source Code References

### `mul_mat_vec_q` kernel: stock llama.cpp

- File: `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/mmvq.cu` lines 140-356
- Template params: `<ggml_type type, int ncols_dst, bool has_fusion, bool is_multi_token_id>`
- Launch bounds: line 141: `__launch_bounds__(calc_nwarps(...)*warp_size, 1)`
- Main loop: line 244-264
- Reduction: lines 266-351

### `vec_dot_iq3_s_q8_1`: shared between stock and ik_llama

- File: `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` lines 1068-1105
- VDR: line 1064: `#define VDR_IQ3_S_Q8_1_MMVQ 2`
- Grid table: `/home/remondiere/llama.cpp/ggml/src/ggml-common.h` line 1020

### `get_int_b2` / `get_int_b4`: vectorized load helpers

- File: `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh` lines 18-29
- `get_int_b2`: loads 4 bytes via 2x uint16_t reads
- `get_int_b4`: loads 4 bytes via 1x int32 read

### ik_llama MMVQ template (same algorithm, adds fused SwiGLU):

- File: `/home/remondiere/ik_llama.cpp/ggml/src/ggml-cuda/mmvq-templates.cuh` lines 68-451
- IQ3_S instance: `/home/remondiere/ik_llama.cpp/ggml/src/ggml-cuda/template-instances/mmvq-instance-iq3_s.cu`

### Q8_1 quantization kernel:

- File: `/home/remondiere/llama.cpp/ggml/src/ggml-cuda/quantize.cu` lines 4-48
- 256 threads per block, cooperative (32 threads per Q8_1 block)
- Uses `warp_reduce_max<32>` and `warp_reduce_sum<32>`
