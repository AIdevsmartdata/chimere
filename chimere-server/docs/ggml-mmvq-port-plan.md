# ggml MMVQ Kernel Port Plan for IQ3_S
## chimere-deltanet -- 2026-03-15

---

## 1. ggml's MMVQ Architecture Overview

ggml's `mul_mat_vec_q` is a **single templated kernel** that handles all quantized
types (Q4_0 through IQ4_XS) via a compile-time type dispatch. For IQ3_S with
`ncols_dst=1` (single-token generation), the kernel:

1. Receives **pre-quantized Q8_1 activations** (quantized by a separate kernel).
2. Reads **IQ3_S weight blocks** from global memory.
3. Computes the dot product **in registers** using `__dp4a` (int8x4 dot product).
4. Reduces across warps via **shared memory** + **warp shuffle**.

### Key template parameters for IQ3_S, ncols_dst=1:

```
qk  = QK_K = 256          (elements per IQ3_S super-block)
qi  = QI3_S = QK_K / (4 * QR3_S) = 256 / (4 * 4) = 16
vdr = VDR_IQ3_S_Q8_1_MMVQ = 2
nwarps = 4                 (MMVQ_PARAMETERS_GENERIC, ncols_dst=1)
warp_size = 32
rows_per_cuda_block = 1    (GENERIC, ncols_dst=1)
blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 4 * 32 / 16 = 16
```

---

## 2. Activation Quantization (Q8_1)

### Q1: How does ggml quantize activations for MMVQ?

**Separate kernel, not in-kernel.** The quantization happens in `quantize_q8_1`
(file: `quantize.cu`, line 4-48), launched from `quantize_row_q8_1_cuda` (line 273-287)
**before** the MMVQ kernel runs.

```c
// Launch config for quantization:
block_num_x = (ne0 + 256 - 1) / 256;  // CUDA_QUANTIZE_BLOCK_SIZE = 256
grid  = (block_num_x, ne1, ne2*ne3)
block = (256, 1, 1)
```

**One thread quantizes one element.** The kernel:
1. Each thread loads one float (or zero if out of bounds).
2. `warp_reduce_max<QK8_1>(amax)` -- finds max abs over 32 threads (one Q8_1 block).
3. `warp_reduce_sum<QK8_1>(sum)` -- sums all 32 values for the `s` field.
4. Scale: `d = amax / 127.0f`, quantize: `q = roundf(xi / d)`.
5. Thread 0 in each group writes `ds = make_half2(d, sum)`.

**Result lives in global memory** as `block_q8_1` structs (36 bytes each: 4 bytes ds + 32 bytes qs).

**Chimere comparison:** Our `quantize_f32_to_q8_1` kernel is serial -- one thread
does all 32 elements of a block. ggml uses 32 threads per block with warp-level
reductions. This alone is a significant difference but is NOT the bottleneck since
quantization happens once and is reused across all expert GEMVs.

---

## 3. Grid/Block Configuration for IQ3_S ncols_dst=1

### Q2: Exact launch config

From `calc_launch_params` (mmvq.cu line 358-365):

```
grid  = (nrows_x, nchannels_dst, nsamples_dst)
block = (warp_size, nwarps, 1) = (32, 4, 1)
```

For a single MoE expert gate/up GEMV (e.g., 512 rows x 2048 cols):
```
grid  = (512, 1, 1)   -- one block per output row
block = (32, 4, 1)     -- 128 threads = 4 warps
```

**`__launch_bounds__`** is set at line 141:
```c
__launch_bounds__(calc_nwarps(ncols_dst, get_device_table_id()) *
                  ggml_cuda_get_physical_warp_size(), 1)
// = __launch_bounds__(4 * 32, 1) = __launch_bounds__(128, 1)
```

The **second argument is 1** = minimum blocks per SM. This tells the compiler
"optimize for 1 block per SM" which allows it to use more registers per thread
(up to 255). This is appropriate for MMVQ since the kernel is memory-latency-bound
and benefits more from register pressure reduction than from occupancy.

**Chimere comparison:** Our kernel uses `(128, 1, 1)` block dims (flat threadIdx.x)
and has **no `__launch_bounds__`**. The NVRTC compiler sees 128 threads but doesn't
know the minimum blocks constraint, so it may spill registers unnecessarily or
under-allocate. The flat `threadIdx.x` vs ggml's `(threadIdx.x, threadIdx.y)` is
just syntactic -- 128 threads either way.

---

## 4. The Main Loop: Thread Assignment and Memory Access

### Q3: Memory access pattern

The core loop (mmvq.cu line 244-264):

```c
constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
// = 2 * 4 * 32 / 16 = 16

for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);  // = kbx * 8
    const int kqs = vdr * (tid % (qi/vdr));  // = 2 * (tid % 8)
    // ...
    tmp[j][i] += vec_dot_q_cuda(vx, &y[kby], kbx_offset + kbx, kqs);
}
```

**Thread assignment (128 threads, IQ3_S):**
- `qi/vdr = 16/2 = 8` -- 8 threads share one IQ3_S block.
- `tid / 8` = which IQ3_S block this thread works on (0..15 within one iteration).
- `tid % 8` = which sub-group pair this thread handles.
- `kqs = 2 * (tid % 8)` = 0, 2, 4, 6, 8, 10, 12, 14.

**But wait -- IQ3_S has only 8 sub-groups (0..7).** With VDR=2, `kqs` ranges over
0, 2, 4, 6, 8, 10, 12, 14. Looking at `vec_dot_iq3_s_q8_1`, the parameter `iqs`
is used as:
- `bq3->qs[iqs + 0]` and `bq3->qs[iqs + 1]` via `get_int_b2(bq3->qs, iqs + 0/1)`
- `bq3->qh[iqs/2]`
- `bq3->signs[iqs/2]` via `get_int_b2(bq3->signs, iqs/2)`
- `bq8_1[iqs/2]` -- selects which Q8_1 block
- Scale: `bq3->scales[iqs/4]` with shift `(iqs << 1) & 0x04`

**The iqs sliding window:** With `iqs` = 0, 2, 4, 6, each call to `vec_dot_iq3_s_q8_1`
processes **32 elements** (one sub-group). With VDR=2, only values 0, 2, 4, 6 are
used (covering sub-groups 0-3), and then `iqs` = 8, 10, 12, 14 would index into
a SECOND call. But actually, looking more carefully:

The `iqs` parameter in `vec_dot_iq3_s_q8_1` does:
```c
const int2 qs_packed = make_int2(
    get_int_b2(bq3->qs, iqs + 0),   // loads 4 bytes from qs at offset 2*(iqs)
    get_int_b2(bq3->qs, iqs + 1));  // loads 4 bytes from qs at offset 2*(iqs+1)
const uint8_t * qs = (const uint8_t *) &qs_packed;  // 8 bytes = qs indices

const int qh = bq3->qh[iqs/2];           // 1 qh byte
const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);  // 4 sign bytes
const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;

// Loop over 4 pairs (l0 = 0, 2, 4, 6) processing 8 grid entries = 32 elements
for (int l0 = 0; l0 < 8; l0 += 2) { ... }

// Q8_1 block: bq8_1[iqs/2]
const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
```

So each `vec_dot` call with a given `iqs` value (0, 2, 4, or 6) processes
32 elements within one IQ3_S super-block. With 8 threads per block
(`tid % 8` giving `kqs` = 0, 2, 4, 6, 8, 10, 12, 14), only `kqs` values
0, 2, 4, 6 index valid sub-groups. Values 8..14 would access `qs[16..21]`
which are bytes 16-21 of the `qs[64]` array (valid since `qs` has 64 bytes).

**Mapping kqs to sub-groups:**
- `kqs=0`: qs bytes 0-7, qh[0], signs bytes 0-3 = sub-groups 0-1
  - Actually: `get_int_b2(qs, 0)` = qs[0..3], `get_int_b2(qs, 1)` = qs[4..7] = 8 qs bytes
  - These 8 qs bytes index 8 grid entries x 4 bytes = 32 elements
  - `qh[0]`, `signs` from `get_int_b2(signs, 0)` = signs[0..3]
  - Q8_1 block: `bq8_1[0]` (32 int8 values)
- `kqs=2`: qs bytes 8-15, qh[1], signs bytes 4-7 = sub-groups 2-3
- `kqs=4`: qs bytes 16-23, qh[2], signs bytes 8-11 = sub-groups 4-5
- `kqs=6`: qs bytes 24-31, qh[3], signs bytes 12-15 = sub-groups 6-7
- `kqs=8`: qs bytes 32-39, qh[4], signs bytes 16-19 = sub-groups 8-9 (INVALID -- overflow)

Wait. QK_K=256, so each IQ3_S block has 8 sub-groups. Let me recount:
- `qs[QK_K/4]` = `qs[64]` -- 64 bytes, 8 bytes per sub-group
- `qh[QK_K/32]` = `qh[8]` -- 8 bytes, 1 byte per sub-group
- `signs[QK_K/8]` = `signs[32]` -- 32 bytes, 4 bytes per sub-group
- Each sub-group = 32 elements

With `iqs` = 0..6 step 2, that's 4 values, each processing 32 elements = 128 elements.
With VDR=2 and 8 threads per IQ3_S block, each thread handles `kqs = 2*(tid%8)`:
- Threads 0-3: kqs = 0, 2, 4, 6 -- processes sub-groups 0-7 (all 256 elements)
- Threads 4-7: kqs = 8, 10, 12, 14 -- **these would overflow!**

Actually, with `qi=16` and `vdr=2`, `qi/vdr = 8` means 8 threads per IQ3_S block.
But `vec_dot_iq3_s_q8_1` already processes 32 elements per call, and with VDR=2
only 4 threads suffice for 256 elements. So **128 threads process
128/8 = 16 IQ3_S blocks per iteration**. Let me re-examine:

Actually `kqs` ranges 0..14 in steps of 2. The function `vec_dot_iq3_s_q8_1`
processes 32 elements for each `iqs` value. With `iqs` from 0 to 14 step 2,
that would be 8 calls = 256 elements. But each call already does a loop of 4
iterations x 2 grid entries x 4 bytes = 32 elements. So one function call = 32 elements.

With 8 threads per IQ3_S block (kqs = 0,2,4,6,8,10,12,14), each processing
32 elements = 256 elements total = exactly one IQ3_S super-block.

**This means:** ggml assigns 8 threads to one IQ3_S block, each thread doing
32 elements. Then partial sums are reduced across all threads.

**Our kernel:** We assign 1 thread to 1 sub-group (32 elements), striding by 128
over the total sub-groups. The key difference: ggml has 8 threads per block doing
independent sub-groups that get summed later; we have 1 thread per sub-group.

### Memory access: vectorized loads

**ggml uses `get_int_b2`** for qs and signs -- loads via `uint16_t*` cast, assembling
two 16-bit loads into one 32-bit value. **`get_int_b4`** for Q8_1 qs -- direct 32-bit
aligned load. These 16-bit and 32-bit loads compile to `ld.global.u16` or
`ld.global.u32` instructions, much better than byte-by-byte access.

**Our kernel uses byte access everywhere:**
```c
unsigned int qs0 = (unsigned int)block[qs_off + 2*l];     // byte load
unsigned int qs1 = (unsigned int)block[qs_off + 2*l + 1]; // byte load
unsigned char sb = b[signs_off + l];                       // byte load
```

This is the **#1 performance killer**. Each byte load is a separate `ld.global.u8`
instruction with no coalescing benefit. ggml's `get_int_b2` loads 4 bytes as two
`u16` loads, and `get_int_b4` loads 4 bytes as one `u32` load.

**Q8_1 data is in global memory** in both implementations (not shared memory for MMVQ).

---

## 5. Register Usage and `__launch_bounds__`

### Q4: Register pressure

**ggml's `__launch_bounds__(128, 1)`:**
- Tells NVCC: max 128 threads per block, minimum 1 block per SM.
- With 1 block per SM and 128 threads, the compiler can allocate up to
  `65536 / 128 = 512` registers per thread (RTX 5060 Ti has 65536 regs per SM).
- In practice, the kernel likely uses ~40-60 registers per thread for IQ3_S.

**Our kernel:** No `__launch_bounds__`. NVRTC defaults to assuming occupancy targets
that may spill registers to local memory. The reported 252 virtual b32 registers
suggests the compiler is trying to keep everything in registers but generating
excessive spills due to:
1. **Scalar byte loads** instead of packed int loads (more intermediate values).
2. **Serial quantization** function inlined into the kernel (now split out).
3. **No `__launch_bounds__`** hint -- compiler may target higher occupancy.

**VDR=2 register reduction:** VDR (Vector Dot product Ratio) = 2 means each thread
processes 2x the data per iteration, which means the outer loop runs 2x fewer
iterations. This DOUBLES the work per thread but HALVES the number of partial sums
to track. The net effect: fewer loop iterations = fewer live variables at any point
= lower register pressure. The compiler can schedule instructions more tightly.

**Expected register count after port:** ~40-50 registers per thread (matching ggml).
With `__launch_bounds__(128, 1)`, the compiler knows it can use up to 512 regs
without hurting occupancy, so it will NOT spill.

---

## 6. iqs Sliding Window vs Sequential Layout

### Q5: Performance reason for ggml's iqs-based layout

**ggml's iqs pattern:**
```
kqs = 2 * (tid % 8)   -- values: 0, 2, 4, 6, 8, 10, 12, 14
```

Each thread uses `kqs` to index into the IQ3_S block with:
- `get_int_b2(bq3->qs, kqs)` and `get_int_b2(bq3->qs, kqs+1)`
- `bq3->qh[kqs/2]`
- `bq8_1[kqs/2].qs[l0]`

**Our sequential sg pattern:**
```
sg = sg_idx % 8        -- values: 0, 1, 2, 3, 4, 5, 6, 7
qs_off = 2 + sg * 8    -- byte offsets: 2, 10, 18, 26, 34, 42, 50, 58
```

**Key difference: coalescing.** When 8 threads in the same warp access the same
IQ3_S block simultaneously:

- ggml threads 0-7 access `qs` at byte offsets: 0, 4, 8, 12, 16, 20, 24, 28
  (via `get_int_b2` with indices 0, 2, 4, 6, 8, 10, 12, 14 -- each loads 4 bytes
  starting at `2*iqs` bytes into qs). These are **consecutive 4-byte aligned** accesses
  within a 64-byte cache line. **Perfect coalescing.**

- Our threads 0-7 access `qs` at byte offsets: 2, 10, 18, 26, 34, 42, 50, 58
  (via `block[qs_off + 2*l]`). These are 8 bytes apart. **Also within a 128-byte
  cache line, but accessed as individual bytes** -- no vectorization.

**The real performance difference is NOT the layout pattern -- it's the load width.**
ggml loads 4 bytes at once via `get_int_b2` (two 16-bit loads assembled into 32 bits).
Our kernel loads 1 byte at a time. Both access patterns span the same cache lines,
but ggml issues 1/4 as many load instructions.

**Conclusion:** Both sequential and iqs-based layouts are valid. The performance
difference comes from:
1. **Vectorized loads** (get_int_b2/b4 vs byte access) -- this is the dominant factor.
2. **8 threads per IQ3_S block** vs 1 thread per sub-group -- better ILP within a warp.
3. **`__launch_bounds__`** controlling register allocation.

---

## 7. Detailed Side-by-Side Comparison

| Aspect | ggml MMVQ | chimere current | Impact |
|--------|-----------|-----------------|--------|
| Block dim | (32, 4, 1) | (128, 1, 1) | Equivalent: 128 threads |
| `__launch_bounds__` | (128, 1) | None | **HIGH**: compiler register hints |
| Threads per IQ3_S block | 8 | 1 (stride over sub-groups) | **MED**: ILP and coalescing |
| IQ3_S blocks per iteration | 16 | N/A (linear stride) | Different scheduling |
| qs load | `get_int_b2` (2x u16 = u32) | byte-by-byte | **CRITICAL**: 4x fewer loads |
| qh load | `bq3->qh[iqs/2]` (byte from struct) | `block[66+sg]` (byte from raw) | Similar |
| signs load | `get_int_b2(signs, iqs/2)` (u32) | `block[signs_off+l]` (byte) | **HIGH**: 4x fewer loads |
| Q8_1 load | `get_int_b4(bq8_1[].qs, l0)` (u32) | `q8_qs_int[2*l]` (u32 via cast) | **Similar** |
| Sign decode | `__vcmpne4` + `__vsub4` | Same | Same |
| Dot product | `ggml_cuda_dp4a` = `__dp4a` | `__dp4a` | Same |
| Scale multiply | `sumi *= 1 + 2*scale_nibble` | Same | Same |
| d multiply | `__half2float(d) * __low2float(ds)` | `f16_to_f32_gemv(d) * f16_to_f32_gemv(d_q8)` | **MED**: intrinsic vs manual |
| Warp reduction | `warp_reduce_sum<32>` (butterfly) | `__shfl_xor_sync` (butterfly) | Same |
| Cross-warp reduction | shared mem array + sequential add | shared mem + thread 0 add | Same |
| Grid LUT | `iq3s_grid[]` in const/texture mem | `__constant__ iq3s_grid_gemv[]` | **Same 512 entries** |
| Q8_1 location | Global memory (pre-quantized) | Global memory (pre-quantized) | Same |
| Activation quantization | Separate kernel, 32 threads/block | Separate kernel, 1 thread/block | Different but not bottleneck |

---

## 8. Step-by-Step Porting Plan

### Step 1: NVRTC kernel skeleton with correct launch config

Replace the current kernel with ggml's structure. The key change is moving from
"1 thread strides over sub-groups" to "8 threads cooperate on each IQ3_S block".

```c
extern "C" __global__
__launch_bounds__(128, 1)  // 4 warps, min 1 block per SM
void gemv_iq3s_q8_v2(
    const unsigned char* __restrict__ weights,
    float*               __restrict__ output,
    int                              cols,
    const unsigned char* __restrict__ q8_input,
    int                              n_rows_per_block  // always 1 for GEMV
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.y * 32 + threadIdx.x;  // 0..127

    // Thread assignment: 8 threads per IQ3_S block, 16 blocks per iter
    const int blocks_per_row = cols / 256;
    const int BLOCKS_PER_ITER = 16;  // = vdr * nwarps * warp_size / qi
```

NVRTC note: We CAN use `__launch_bounds__` in NVRTC (supported since CUDA 11+).
We CANNOT use C++ templates or `constexpr`, but we can use `#define` macros and
`const int` for compile-time constants.

### Step 2: Port `vec_dot_iq3_s_q8_1` with vectorized loads

The core dot product function, adapted for NVRTC (no templates, no struct member access):

```c
__device__ __forceinline__ float vec_dot_iq3s_q8(
    const unsigned char* __restrict__ vbq,  // IQ3_S raw bytes
    const unsigned char* __restrict__ q8,   // Q8_1 raw bytes
    int kbx,         // IQ3_S block index within this row
    int iqs,         // sub-group selector: 0, 2, 4, or 6
    int row_stride   // bytes per row in IQ3_S data = n_blocks * 110
) {
    const unsigned char* bq3 = vbq + kbx * 110;

    // Vectorized qs load: 4 bytes via uint16_t pair (get_int_b2)
    const unsigned short* qs16 = (const unsigned short*)(bq3 + 2);  // offset 2 = qs start
    int qs_packed0 = qs16[2*iqs + 0] | (qs16[2*iqs + 1] << 16);
    int qs_packed1 = qs16[2*(iqs+1) + 0] | (qs16[2*(iqs+1) + 1] << 16);
    // ... BUT this is exactly what get_int_b2 does.
    // Better: cast to int2 for explicit 8-byte load
    const unsigned char* qs = (const unsigned char*)&qs_packed0;
    // Actually, replicate ggml: load as int2 at once
    // ...
```

Actually, the cleanest port: replicate `get_int_b2` exactly:

```c
__device__ __forceinline__ int get_int_b2(const void* x, int i32) {
    const unsigned short* x16 = (const unsigned short*)x;
    return x16[2*i32] | (x16[2*i32 + 1] << 16);
}

__device__ __forceinline__ int get_int_b4(const void* x, int i32) {
    return ((const int*)x)[i32];
}
```

### Step 3: Port the dot product function faithfully

```c
__device__ __forceinline__ float vec_dot_iq3s_q8(
    const unsigned char* __restrict__ block_ptr,  // IQ3_S block, 110 bytes
    const unsigned char* __restrict__ q8_ptr,     // Q8_1 block array
    int iqs                                       // 0, 2, 4, or 6 (or 8..14 with VDR=2)
) {
    // IQ3_S block layout offsets:
    //   0: d (2 bytes, f16)
    //   2: qs[64]
    //  66: qh[8]
    //  74: signs[32]
    // 106: scales[4]

    // Load 8 qs bytes as two int (vectorized)
    int qs_lo = get_int_b2(block_ptr + 2, iqs + 0);  // qs[4*iqs .. 4*iqs+3]
    int qs_hi = get_int_b2(block_ptr + 2, iqs + 1);  // qs[4*(iqs+1) .. 4*(iqs+1)+3]
    const unsigned char* qs = ... ; // point to local vars

    int qh = block_ptr[66 + iqs/2];

    // Load 4 sign bytes as int (vectorized)
    int signs_packed = get_int_b2(block_ptr + 74, iqs/2);
    const unsigned char* signs_bytes = (const unsigned char*)&signs_packed;

    int sumi = 0;

    // Q8_1 block for this sub-group: q8_ptr + (iqs/2) * 36
    const unsigned char* q8_block = q8_ptr + (iqs/2) * 36;
    // Q8_1 qs start at offset 4 (after ds half2)
    const int* q8_qs_int = (const int*)(q8_block + 4);

    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        int grid_idx0 = qs[l0 + 0] | ((qh << (8 - l0)) & 0x100);
        int grid_idx1 = qs[l0 + 1] | ((qh << (7 - l0)) & 0x100);

        int grid0 = iq3s_grid_gemv[grid_idx0];
        int grid1 = iq3s_grid_gemv[grid_idx1];

        int s0 = __vcmpne4(
            ((signs_bytes[l0/2] & 0x03) << 7) | ((signs_bytes[l0/2] & 0x0C) << 21), 0);
        int s1 = __vcmpne4(
            ((signs_bytes[l0/2] & 0x30) << 3) | ((signs_bytes[l0/2] & 0xC0) << 17), 0);

        int g0 = __vsub4(grid0 ^ s0, s0);
        int g1 = __vsub4(grid1 ^ s1, s1);

        sumi = __dp4a(g0, q8_qs_int[l0 + 0], sumi);
        sumi = __dp4a(g1, q8_qs_int[l0 + 1], sumi);
    }

    // Scale
    unsigned char scale_byte = block_ptr[106 + iqs/4];
    int shift = (iqs << 1) & 0x04;
    sumi *= 1 + 2 * ((scale_byte >> shift) & 0x0F);

    // d_iq3 * d_q8
    unsigned short d_raw = block_ptr[0] | (block_ptr[1] << 8);
    float d_iq3 = f16_to_f32_gemv(d_raw);
    unsigned short d_q8_raw = q8_block[0] | (q8_block[1] << 8);
    float d_q8 = f16_to_f32_gemv(d_q8_raw);

    return d_iq3 * d_q8 * (float)sumi;
}
```

**Important note on qs indexing:** In ggml, `get_int_b2(bq3->qs, iqs)` loads 4 bytes
starting at `bq3->qs + 2*iqs` (since uint16_t indexing). In our raw byte layout,
`bq3->qs` starts at offset 2 in the block. So `get_int_b2(block_ptr + 2, iqs)`
loads 4 bytes starting at `block_ptr + 2 + 2*iqs` bytes. With `iqs` = 0, 2, 4, 6,
that's bytes 2, 6, 10, 14 -- loading 4 bytes each at stride 4.

Wait, let me recheck. `get_int_b2(ptr, i32)` loads `ptr[2*i32]` and `ptr[2*i32+1]`
as uint16_t, then combines them. So it reads 4 bytes starting at byte offset `4*i32`
from `ptr`. With `ptr = bq3->qs` and `iqs = 0`: bytes 0-3 of qs. `iqs = 1`: bytes 4-7.
`iqs = 2`: bytes 8-11, etc. So `iqs` steps by 1 give 4-byte steps.

In the function signature, `iqs` comes from `kqs = 2 * (tid % 8)` = 0, 2, 4, ..., 14.
The function uses `iqs + 0` and `iqs + 1`. So for `iqs=0`: loads qs bytes 0-3 and 4-7.
For `iqs=2`: loads qs bytes 8-11 and 12-15. For `iqs=6`: loads qs bytes 24-27 and 28-31.
For `iqs=8`: loads qs bytes 32-35 and 36-39. Up to `iqs=14`: bytes 56-59 and 60-63.
Total: 64 bytes of qs covered by 8 threads. Correct.

### Step 4: Port the main loop structure

```c
extern "C" __global__
__launch_bounds__(128, 1)
void gemv_iq3s_q8_v2(
    const unsigned char* __restrict__ weights,
    float*               __restrict__ output,
    int                              cols,
    const unsigned char* __restrict__ q8_input
) {
    const int row = blockIdx.x;
    const int warp_id = threadIdx.y;   // 0..3
    const int lane_id = threadIdx.x;   // 0..31
    const int tid = warp_id * 32 + lane_id;

    const int blocks_per_row = cols >> 8;  // cols / 256
    const int IQ3S_BYTES = 110;

    // Thread assignment:
    // qi = 16, vdr = 2, qi/vdr = 8
    // blocks_per_iter = 2 * 4 * 32 / 16 = 16
    const int QI_VDR = 8;              // qi / vdr
    const int BLOCKS_PER_ITER = 16;    // vdr * nwarps * warp_size / qi

    const unsigned char* row_weights = weights + (long long)row * blocks_per_row * IQ3S_BYTES;
    const unsigned char* q8_blocks = q8_input;

    float partial = 0.0f;

    for (int kbx = tid / QI_VDR; kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        int kby = kbx * 8;  // Q8_1 block index (8 Q8_1 blocks per IQ3_S block)
        int iqs = 2 * (tid % QI_VDR);  // 0, 2, 4, 6, 8, 10, 12, 14

        const unsigned char* iq3_block = row_weights + kbx * IQ3S_BYTES;
        const unsigned char* q8_block_base = q8_blocks + kby * 36;

        partial += vec_dot_iq3s_q8(iq3_block, q8_block_base, iqs);
    }

    // Warp reduction
    __shared__ float warp_sums[3][32];  // nwarps-1 = 3, indexed by warp_size
    // (ggml uses tmp_shared[nwarps-1][ncols_dst][rows_per_block][warp_size])

    if (warp_id > 0) {
        warp_sums[warp_id - 1][lane_id] = partial;
    }
    __syncthreads();
    if (warp_id > 0) return;

    // Warp 0: sum up partials from all warps
    for (int w = 0; w < 3; w++) {
        partial += warp_sums[w][lane_id];
    }
    partial = warp_reduce_sum_f32(partial);  // butterfly shuffle

    if (lane_id == 0) {
        output[row] = partial;
    }
}
```

### Step 5: NVRTC adaptations needed

| ggml feature | NVRTC adaptation |
|---|---|
| `template <ggml_type type, ...>` | Remove: hardcode IQ3_S constants |
| `constexpr int qk = ...` | `#define QK 256` or `const int qk = 256` |
| `__half2float(bq3->d)` | `f16_to_f32_gemv()` (our existing manual f16 decoder) |
| `__low2float(bq8_1->ds)` | `f16_to_f32_gemv(block[0] | (block[1] << 8))` |
| `struct block_iq3_s` member access | Raw byte offsets (already done in our kernel) |
| `struct block_q8_1` member access | Raw byte offsets: ds at 0-3, qs at 4-35 |
| `ggml_cuda_dp4a` | `__dp4a` (native on sm_61+, including sm_120) |
| `__vcmpne4` / `__vsub4` | Available in NVRTC for sm_120 (already used) |
| `warp_reduce_sum<32>` | Manual butterfly `__shfl_xor_sync` (already done) |
| `get_device_table_id()` | Always GENERIC for NVIDIA GPUs |
| `blockDim = (32, 4, 1)` | Launch with 2D block dims from Rust side |
| `threadIdx.y` for warp_id | Requires 2D block dims OR compute from flat threadIdx.x |
| `__launch_bounds__(128, 1)` | Supported in NVRTC, use directly |
| `iq3s_grid` (global table) | `__constant__` array (already done) |
| `const void* vbq + kbx` struct ptr | `const unsigned char*` + byte offset |

### Step 6: Launch configuration from Rust

```rust
// Current:
let block_dim = (128u32, 1u32, 1u32);
let grid_dim = (n_rows as u32, 1u32, 1u32);

// Ported:
let block_dim = (32u32, 4u32, 1u32);  // warp_size=32, nwarps=4
let grid_dim = (n_rows as u32, 1u32, 1u32);
let shared_mem = 3 * 32 * 4;  // 3 warps * 32 lanes * sizeof(float) = 384 bytes
```

### Step 7: Quantization kernel improvement (optional, lower priority)

The ggml quantization kernel uses 32 threads per Q8_1 block with warp reductions.
Our current kernel uses 1 thread per block (serial). Since quantization is done once
and reused, this is lower priority, but for completeness:

```c
// Port ggml's quantize_q8_1: 1 thread per element, warp reductions for scale/sum
// Launch: grid=(ceil(ne0/256), ne1, ne2*ne3), block=(256, 1, 1)
// Each group of 32 threads quantizes one Q8_1 block cooperatively.
```

---

## 9. Expected Performance After Port

### Current chimere IQ3_S GEMV:
- **~10 us per call** (single 512x2048 GEMV)
- 252 virtual registers, scalar byte access, no `__launch_bounds__`

### Expected after faithful ggml port:
- **~4-5 us per call** (matching ggml's MMVQ)
- ~40-50 registers per thread
- Vectorized loads via `get_int_b2` / `get_int_b4`
- `__launch_bounds__(128, 1)` controlling register allocation

### Per-token improvement estimate:
- MoE FFN does ~24 IQ3_S GEMVs per layer (8 experts x 3 projections)
- 40 layers = 960 GEMV calls per token
- Current: 960 x 10us = 9.6ms
- After port: 960 x 4.5us = 4.3ms
- **Savings: ~5.3ms per token** (23% of current 22.72ms total)
- **From 44 tok/s to ~57 tok/s** (MoE improvement only)

### Why the remaining gap vs ggml (4.5us vs 4us):
1. NVRTC compilation may be slightly less optimized than static nvcc.
2. Our `f16_to_f32_gemv` is software emulation vs ggml's `__half2float` intrinsic
   (though the compiler may optimize both equally on sm_120).
3. We don't have the fusion path (gate+bias+GLU fused into the MMVQ kernel).

---

## 10. Implementation Checklist

- [ ] **Phase 1: Core kernel** (~2 hours)
  - [ ] Write `get_int_b2` and `get_int_b4` helper functions
  - [ ] Port `vec_dot_iq3s_q8` with ggml's iqs-based indexing
  - [ ] Write main loop with `blocks_per_iter=16` striding
  - [ ] Add `__launch_bounds__(128, 1)`
  - [ ] Use 2D block dims (32, 4, 1)
  - [ ] Warp reduction via shared memory (ggml pattern)

- [ ] **Phase 2: Integration** (~1 hour)
  - [ ] Update Rust launch code in `gemv_iq3s_fused_at_offset_q8`
  - [ ] Change block_dim from (128,1,1) to (32,4,1)
  - [ ] Set shared memory size to 384 bytes
  - [ ] Verify correctness against reference dequant+GEMV path

- [ ] **Phase 3: Validation** (~1 hour)
  - [ ] Compare output values against ggml (max relative error < 1e-3)
  - [ ] Benchmark: target <5us per 512x2048 GEMV
  - [ ] Check register count via `--ptxas-options=-v` in NVRTC
  - [ ] Full model inference test: verify tok/s improvement

- [ ] **Phase 4: Polish** (optional)
  - [ ] Port the quantization kernel to parallel (32 threads per Q8_1 block)
  - [ ] Consider adding the GLU fusion path for gate+up
  - [ ] Tune `nwarps` for sm_120 (try 2 warps instead of 4)

---

## 11. Key Code Snippets from ggml (Reference)

### vec_dot_iq3_s_q8_1 (vecdotq.cuh:1068-1105)

```c
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2 qs_packed = make_int2(
        get_int_b2(bq3->qs, iqs + 0),      // 4 bytes via 2x u16
        get_int_b2(bq3->qs, iqs + 1));     // 4 bytes via 2x u16
    const uint8_t * qs = (const uint8_t *) &qs_packed;  // 8 qs bytes in registers

    const int qh = bq3->qh[iqs/2];         // 1 byte
    const int signs_packed_32 = get_int_b2(bq3->signs, iqs/2);  // 4 sign bytes
    const uint8_t * signs_packed_8 = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
    #pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0+0] | ((qh << (8-l0)) & 0x100)],
            iq3s_grid[qs[l0+1] | ((qh << (7-l0)) & 0x100)]);

        const int signs0 = __vcmpne4(
            ((signs_packed_8[l0/2] & 0x03) << 7) |
            ((signs_packed_8[l0/2] & 0x0C) << 21), 0);
        const int signs1 = __vcmpne4(
            ((signs_packed_8[l0/2] & 0x30) << 3) |
            ((signs_packed_8[l0/2] & 0xC0) << 17), 0);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        sumi = ggml_cuda_dp4a(grid_l, get_int_b4(bq8_1[iqs/2].qs, l0+0), sumi);
        sumi = ggml_cuda_dp4a(grid_h, get_int_b4(bq8_1[iqs/2].qs, l0+1), sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = __half2float(bq3->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
}
```

### Main loop (mmvq.cu:244-264)

```c
constexpr int blocks_per_iter = vdr * nwarps * warp_size / qi;
// IQ3_S: 2 * 4 * 32 / 16 = 16

for (int kbx = tid / (qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
    const int kby = kbx * (qk/QK8_1);  // = kbx * 8
    const int kqs = vdr * (tid % (qi/vdr));  // = 2 * (tid % 8)

    #pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        #pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
            tmp[j][i] += vec_dot_q_cuda(
                vx, &y[j*stride_col_y + kby],
                kbx_offset + i*stride_row_x + kbx, kqs);
        }
    }
}
```

### Shared memory reduction (mmvq.cu:266-351)

```c
__shared__ float tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size];
// IQ3_S ncols_dst=1: float tmp_shared[3][1][1][32] = 384 bytes

if (threadIdx.y > 0) {
    tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
}
__syncthreads();
if (threadIdx.y > 0) return;

// Warp 0 sums all partial results
for (int l = 0; l < nwarps-1; ++l) {
    tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
}
tmp[j][i] = warp_reduce_sum<warp_size>(tmp[j][i]);

if (threadIdx.x == 0) {
    dst[row] = tmp[0][0];
}
```

---

## 12. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| NVRTC can't compile `__launch_bounds__` | Low (supported since CUDA 11) | Fall back to `__attribute__((launch_bounds(128,1)))` |
| `get_int_b2` uint16_t cast alignment issues | Low (IQ3_S qs starts at byte 2, even-aligned) | Test on real data, add alignment asserts |
| sm_120 intrinsic availability | Low (`__dp4a`, `__vcmpne4`, `__vsub4` all available) | Already verified in current kernel |
| Register spill despite `__launch_bounds__` | Low | Check ptxas output, manually reduce live vars |
| Performance not matching ggml exactly | Medium | Accept 4-5us (vs ggml 4us) as success; the remaining gap is static compilation advantage |

---

## Appendix A: IQ3_S Block Layout (110 bytes, 256 elements)

```
Offset  Size  Field       Description
  0       2   d           f16 super-block scale
  2      64   qs[64]      8 bytes/subgroup, 8 qs indices per subgroup
 66       8   qh[8]       1 byte/subgroup, high bits for 8 grid indices
 74      32   signs[32]   4 bytes/subgroup, 32 sign bits per subgroup
106       4   scales[4]   1 nibble/subgroup (packed in pairs)
```

**Total: 2 + 64 + 8 + 32 + 4 = 110 bytes** for 256 elements = 3.4375 bpw.

## Appendix B: Q8_1 Block Layout (36 bytes, 32 elements)

```
Offset  Size  Field       Description
  0       4   ds          half2: (d=scale, s=d*sum(qs))
  4      32   qs[32]      int8 quantized values
```

**Total: 4 + 32 = 36 bytes** per 32 elements.

## Appendix C: ggml Constants for IQ3_S

```
QK_K   = 256     (elements per super-block)
QR3_S  = 4       (quantization ratio)
QI3_S  = QK_K / (4 * QR3_S) = 256 / 16 = 16
VDR    = 2       (vector dot ratio for MMVQ)
QK8_1  = 32      (elements per Q8_1 block)
nwarps = 4       (GENERIC table, ncols_dst=1)
rows_per_cuda_block = 1
blocks_per_iter = vdr * nwarps * warp_size / qi = 2 * 4 * 32 / 16 = 16
```
