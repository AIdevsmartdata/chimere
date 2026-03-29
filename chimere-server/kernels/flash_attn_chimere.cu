// flash_attn_chimere.cu — Flash Attention decode kernel for chimere-deltanet.
//
// Specialized for single-token decode (batch=1, q_len=1) with F16 KV cache
// and F32 Q/output. Uses online softmax over KV tiles to avoid materializing
// the full score vector (O(1) extra memory instead of O(seq_len)).
//
// Architecture: Qwen3.5-35B-A3B
//   num_heads=16, num_kv_heads=2, head_dim=256, GQA ratio=8:1
//   10 attention layers (every 4th layer)
//
// Grid:  (num_heads, 1, 1) — one block per Q head
// Block: (256, 1, 1)       — 8 warps, thread i handles dimension i
//
// Memory: constant shared memory = TILE_KV * head_dim * sizeof(half) * 2
//         (K tile + V tile). For TILE_KV=32, head_dim=256: 32*256*2*2 = 32 KB.

#include <cuda_fp16.h>

// Tile size for KV sequence dimension.
// 32 is a good balance: 32 KB shared mem per block, fits in L1.
#define TILE_KV 32

// ---------------------------------------------------------------------------
// Flash Attention decode kernel: fused score + online-softmax + weighted-sum.
//
// Q:      [num_heads, head_dim]                F32 (single token, post-RoPE)
// K:      [num_kv_heads, seq_len, head_dim]    F16 (KV cache)
// V:      [num_kv_heads, seq_len, head_dim]    F16 (KV cache)
// output: [num_heads, head_dim]                F32
//
// For each Q head h:
//   kv_h = h / (num_heads / num_kv_heads)   -- GQA mapping
//   Iterate over KV in tiles of TILE_KV positions.
//   For each tile:
//     Load K[kv_h, tile_start:tile_end, :] into shared memory (F16)
//     Compute scores: s[j] = dot(Q[h,:], K_tile[j,:]) * scale  (F32 accum)
//     Online softmax update:
//       m_new = max(m, max(s))
//       correction = exp(m_old - m_new)
//       l *= correction; o *= correction
//       for j: p = exp(s[j] - m_new); l += p; o += p * V[kv_h,j,:]
//     m = m_new
//   output[h,:] = o / l
// ---------------------------------------------------------------------------

extern "C" __global__ void flash_attn_decode_f16kv(
    const float*          __restrict__ Q,       // [num_heads * head_dim]
    const unsigned short* __restrict__ K,       // [num_kv_heads * seq_len * head_dim] (F16 as u16)
    const unsigned short* __restrict__ V,       // [num_kv_heads * seq_len * head_dim] (F16 as u16)
    float*                __restrict__ output,  // [num_heads * head_dim]
    int seq_len,
    int head_dim,        // expected 256
    int num_heads,       // expected 16
    int num_kv_heads,    // expected 2
    float scale          // 1/sqrt(head_dim)
) {
    const int h   = blockIdx.x;           // Q head index [0, num_heads)
    const int tid = threadIdx.x;          // [0, blockDim.x), expected 256
    const int kv_h = h * num_kv_heads / num_heads;  // GQA: which KV head

    // Load Q[h, tid] into register (each thread owns one dimension)
    float q_val = 0.0f;
    if (tid < head_dim) {
        q_val = Q[h * head_dim + tid];
    }

    // Shared memory for K and V tiles (F16).
    // Layout: K_tile[TILE_KV][head_dim] then V_tile[TILE_KV][head_dim]
    extern __shared__ char smem[];
    unsigned short* k_tile = (unsigned short*)smem;
    // Scores buffer: TILE_KV floats after K tile
    float* s_tile = (float*)(smem + TILE_KV * head_dim * sizeof(unsigned short));
    // Reduction workspace: 8 floats (one per warp)
    float* warp_reduce = (float*)(smem + TILE_KV * head_dim * sizeof(unsigned short)
                                      + TILE_KV * sizeof(float));

    // Online softmax accumulators (per thread, for its dimension)
    float o_acc = 0.0f;    // running weighted sum for output[h, tid]
    float m_run = -1e30f;  // running max score
    float l_run = 0.0f;    // running sum of exp(score - m)

    // Base pointers for this KV head
    const unsigned short* k_base = K + (long long)kv_h * seq_len * head_dim;
    const unsigned short* v_base = V + (long long)kv_h * seq_len * head_dim;

    // Number of warps in this block
    const int n_warps = (blockDim.x + 31) >> 5;

    // Iterate over KV sequence in tiles
    for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_KV) {
        int tile_end = tile_start + TILE_KV;
        if (tile_end > seq_len) tile_end = seq_len;
        int tile_len = tile_end - tile_start;

        // --- Load K tile into shared memory ---
        // K_tile[j][d] for j in [0, tile_len), d in [0, head_dim)
        // Total elements: tile_len * head_dim
        // With 256 threads and tile_len*head_dim = 32*256 = 8192, each thread loads 32 elements
        {
            int total_k = tile_len * head_dim;
            for (int i = tid; i < total_k; i += blockDim.x) {
                k_tile[i] = k_base[(tile_start * head_dim) + i];
            }
        }
        __syncthreads();

        // --- Compute scores for this tile ---
        // For each position j in the tile, compute:
        //   s[j] = dot(Q[h,:], K_tile[j,:]) * scale
        // Each thread contributes Q[h,tid] * K_tile[j,tid], then reduce across threads.

        for (int j = 0; j < tile_len; j++) {
            // Each thread computes its partial product
            float partial = 0.0f;
            if (tid < head_dim) {
                // Convert F16 K value to F32
                unsigned short k_bits = k_tile[j * head_dim + tid];
                // F16 to F32 conversion (same as chimere_kernels.cu)
                unsigned int sign = (k_bits >> 15) & 1u;
                unsigned int exp_val  = (k_bits >> 10) & 0x1Fu;
                unsigned int mant =  k_bits & 0x3FFu;
                float k_f32;
                if (exp_val == 0u) {
                    k_f32 = (float)mant * (1.0f / (1024.0f * 16384.0f));
                    if (sign) k_f32 = -k_f32;
                } else if (exp_val == 31u) {
                    unsigned int f32_bits = (sign << 31) | 0x7F800000u | (mant << 13);
                    k_f32 = *(float*)&f32_bits;
                } else {
                    unsigned int f32_bits = (sign << 31) | ((exp_val + 112u) << 23) | (mant << 13);
                    k_f32 = *(float*)&f32_bits;
                }
                partial = q_val * k_f32;
            }

            // Warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xFFFFFFFF, partial, offset);
            }

            // Cross-warp reduction via shared memory
            if ((tid & 31) == 0) {
                warp_reduce[tid >> 5] = partial;
            }
            __syncthreads();

            if (tid == 0) {
                float dot = 0.0f;
                for (int w = 0; w < n_warps; w++) {
                    dot += warp_reduce[w];
                }
                s_tile[j] = dot * scale;
            }
            __syncthreads();
        }

        // --- Online softmax update for this tile ---

        // Find tile max (cooperative across threads)
        float tile_max = -1e30f;
        for (int j = tid; j < tile_len; j += blockDim.x) {
            float s = s_tile[j];
            if (s > tile_max) tile_max = s;
        }
        // Warp reduce max
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, tile_max, offset);
            if (other > tile_max) tile_max = other;
        }
        if ((tid & 31) == 0) {
            warp_reduce[tid >> 5] = tile_max;
        }
        __syncthreads();
        if (tid == 0) {
            float m = warp_reduce[0];
            for (int w = 1; w < n_warps; w++) {
                if (warp_reduce[w] > m) m = warp_reduce[w];
            }
            warp_reduce[0] = m;
        }
        __syncthreads();
        tile_max = warp_reduce[0];
        __syncthreads();

        // New running max
        float m_new = (tile_max > m_run) ? tile_max : m_run;

        // Correction factor for previous accumulators
        float correction = expf(m_run - m_new);
        o_acc *= correction;
        l_run *= correction;

        // Accumulate this tile's contributions
        // Each thread handles one dimension of the output.
        // For each position j in the tile:
        //   p = exp(s[j] - m_new)
        //   l_run += p  (only one thread needs to do this, but all need p)
        //   o_acc += p * V[kv_h, tile_start+j, tid]

        // First compute exp(s[j] - m_new) for all j and sum into l_run
        // All threads need the same p values, so thread 0 could compute them,
        // but we can also have each thread compute them redundantly since s_tile
        // is in shared memory.

        for (int j = 0; j < tile_len; j++) {
            float p = expf(s_tile[j] - m_new);

            // Only tid==0 accumulates l_run (to avoid 256x over-counting)
            if (tid == 0) {
                l_run += p;
            }

            // All threads with tid < head_dim accumulate V contribution
            if (tid < head_dim) {
                unsigned short v_bits = v_base[(tile_start + j) * head_dim + tid];
                // F16 to F32
                unsigned int v_sign = (v_bits >> 15) & 1u;
                unsigned int v_exp  = (v_bits >> 10) & 0x1Fu;
                unsigned int v_mant =  v_bits & 0x3FFu;
                float v_f32;
                if (v_exp == 0u) {
                    v_f32 = (float)v_mant * (1.0f / (1024.0f * 16384.0f));
                    if (v_sign) v_f32 = -v_f32;
                } else if (v_exp == 31u) {
                    unsigned int f32_bits = (v_sign << 31) | 0x7F800000u | (v_mant << 13);
                    v_f32 = *(float*)&f32_bits;
                } else {
                    unsigned int f32_bits = (v_sign << 31) | ((v_exp + 112u) << 23) | (v_mant << 13);
                    v_f32 = *(float*)&f32_bits;
                }
                o_acc += p * v_f32;
            }
        }

        m_run = m_new;
        __syncthreads();
    }

    // --- Broadcast l_run from thread 0 to all threads ---
    // Thread 0 has the correct l_run; others have 0.
    // Use shared memory to broadcast.
    if (tid == 0) {
        warp_reduce[0] = l_run;
    }
    __syncthreads();
    float l_final = warp_reduce[0];

    // --- Write output ---
    if (tid < head_dim && l_final > 0.0f) {
        output[h * head_dim + tid] = o_acc / l_final;
    } else if (tid < head_dim) {
        output[h * head_dim + tid] = 0.0f;
    }
}
