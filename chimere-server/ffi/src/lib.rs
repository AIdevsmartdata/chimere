//! ggml-ffi: Quantized GEMV via ggml FFI + pure-Rust validation.
//!
//! This crate provides two GEMV backends:
//!
//! ## 1. Pure-Rust (validation path)
//!
//! Implements ggml-compatible quantized dequantization and matrix-vector
//! multiplication in pure Rust. No external C dependencies required.
//! The CPU round-trip is intentionally slow -- correctness is the only goal.
//!
//! - **Q8_0**: 32 elements/block, 34 bytes (f16 scale + 32 i8 quants)
//! - **Q5_K**: 256 elements/block, 176 bytes (f16 d + f16 dmin + 12 scales + 32 qh + 128 qs)
//! - **Q8_1**: 32 elements/block, 40 bytes (f32 d + f32 s + 32 i8 quants)
//!
//! ## 2. ggml FFI (performance path, requires libggml.so)
//!
//! Links against ggml's native IQ3_S implementation with AVX2 SIMD.
//! Used for the "invert ncmoe" approach: copy 8 KB hidden state to CPU
//! instead of 10 MB expert weights to GPU.
//!
//! - **IQ3_S**: 256 elements/block, 110 bytes -- ggml AVX2-optimized dot product
//! - **Q8_K**: 256 elements/block, 296 bytes -- activation quantization for IQ3_S
//!
//! Toggle: `CHIMERE_NCMOE_CPU=1` enables the CPU ggml path for ncmoe experts.
//!
//! ## Usage
//!
//! ```ignore
//! // Pure-Rust Q8_0 (validation):
//! let ctx = GgmlCpuContext::new();
//! let mut output = vec![0.0f32; nrows];
//! ctx.mul_mat_vec_q8_0(raw_bytes, nrows, ncols, &input_f32, &mut output);
//!
//! // ggml FFI IQ3_S (performance):
//! let output = iq3s_gemv_cpu(weight_iq3s, &input_f32, nrows, ncols);
//! // Or parallel (14 threads on i5-14600KF):
//! let output = iq3s_gemv_cpu_parallel(weight_iq3s, &input_f32, nrows, ncols, 14);
//! ```

/// Q8_0 block: 32 elements, 34 bytes total.
const QK8_0: usize = 32;
/// Bytes per Q8_0 block: 2 (f16 scale) + 32 (int8 values) = 34.
const Q8_0_BLOCK_BYTES: usize = 34;

/// Q8_1 block: 32 elements, 40 bytes total.
/// Layout: f32 d (4 bytes) + f32 s (4 bytes) + 32 i8 qs (32 bytes).
/// s = d * sum(qs[0..32]) -- weighted sum for the "min" contribution in K-quant dots.
const QK8_1: usize = 32;
const Q8_1_BLOCK_BYTES: usize = 40; // 4 + 4 + 32

/// Q5_K super-block: 256 elements, 176 bytes total.
/// Layout: f16 d (2) + f16 dmin (2) + scales[12] + qh[32] + qs[128].
const QK_K: usize = 256;
const Q5_K_BLOCK_BYTES: usize = 176; // 2 + 2 + 12 + 32 + 128
/// Number of Q8_1 blocks per Q5_K super-block: 256/32 = 8.
const Q8_1_PER_Q5K: usize = QK_K / QK8_1;

/// ggml quantization type enum (matching ggml.h values).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ3_S   = 21,
    BF16    = 30,
}

/// Pure-Rust CPU context for ggml-compatible quantized GEMV.
///
/// No GPU, no FFI -- just correct dequantization and matrix-vector multiply.
/// Used for validation against llama.cpp / ik_llama reference outputs.
pub struct GgmlCpuContext;

impl GgmlCpuContext {
    /// Create a new CPU context. No initialization needed.
    pub fn new() -> Self {
        Self
    }

    /// Compute y = W @ x where W is Q8_0 `[nrows, ncols]` and x is f32 `[ncols]`.
    ///
    /// This is ggml-compatible: same block layout, same dequantization formula.
    /// Each row of W has `ncols / 32` blocks of Q8_0 (34 bytes each).
    ///
    /// # Arguments
    /// * `w_data` - Raw Q8_0 bytes, row-major, `nrows * (ncols/32) * 34` bytes total
    /// * `nrows`  - Number of output features (vocab_size for lm_head)
    /// * `ncols`  - Number of input features (hidden_size for lm_head)
    /// * `x`      - Input vector, f32, length `ncols`
    /// * `y`      - Output vector, f32, length `nrows` (overwritten)
    ///
    /// # Panics
    /// Panics if dimensions are invalid or buffer sizes don't match.
    pub fn mul_mat_vec_q8_0(
        &self,
        w_data: &[u8],
        nrows: usize,
        ncols: usize,
        x: &[f32],
        y: &mut [f32],
    ) {
        assert_eq!(ncols % QK8_0, 0, "ncols must be multiple of {QK8_0}");
        assert!(x.len() >= ncols, "x too short: {} < {ncols}", x.len());
        assert!(y.len() >= nrows, "y too short: {} < {nrows}", y.len());

        let blocks_per_row = ncols / QK8_0;
        let row_bytes = blocks_per_row * Q8_0_BLOCK_BYTES;
        let expected_bytes = nrows * row_bytes;
        assert!(
            w_data.len() >= expected_bytes,
            "w_data too short: {} < {expected_bytes} (nrows={nrows}, ncols={ncols})",
            w_data.len()
        );

        for row in 0..nrows {
            let row_start = row * row_bytes;
            let mut sum = 0.0f64; // accumulate in f64 for precision

            for blk in 0..blocks_per_row {
                let blk_offset = row_start + blk * Q8_0_BLOCK_BYTES;

                // Read f16 scale (2 bytes, little-endian)
                let scale_u16 = u16::from_le_bytes([
                    w_data[blk_offset],
                    w_data[blk_offset + 1],
                ]);
                let scale = f16_to_f32(scale_u16) as f64;

                // Read 32 int8 quantized values
                let qs_start = blk_offset + 2;
                let x_start = blk * QK8_0;

                for i in 0..QK8_0 {
                    let qi = w_data[qs_start + i] as i8;
                    sum += scale * (qi as f64) * (x[x_start + i] as f64);
                }
            }

            y[row] = sum as f32;
        }
    }

    /// Compute y = W @ x where W is F32 `[nrows, ncols]` and x is f32 `[ncols]`.
    /// Used as a reference/test path.
    pub fn mul_mat_vec_f32(
        &self,
        w_data: &[u8],
        nrows: usize,
        ncols: usize,
        x: &[f32],
        y: &mut [f32],
    ) {
        assert!(x.len() >= ncols);
        assert!(y.len() >= nrows);
        let expected_bytes = nrows * ncols * 4;
        assert!(w_data.len() >= expected_bytes);

        // Reinterpret w_data as &[f32]
        let w_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(
                w_data.as_ptr() as *const f32,
                nrows * ncols,
            )
        };

        for row in 0..nrows {
            let row_start = row * ncols;
            let mut sum = 0.0f64;
            for col in 0..ncols {
                sum += (w_f32[row_start + col] as f64) * (x[col] as f64);
            }
            y[row] = sum as f32;
        }
    }

    /// Compute y = W @ x where W is Q5_K `[nrows, ncols]` and x is f32 `[ncols]`.
    ///
    /// Internally quantizes x to Q8_1 blocks, then computes the dot product
    /// using `vec_dot_q5_K_q8_1` for each row.
    ///
    /// # Arguments
    /// * `w_data` - Raw Q5_K bytes, row-major, `nrows * (ncols/256) * 176` bytes total
    /// * `nrows`  - Number of output features
    /// * `ncols`  - Number of input features (must be multiple of 256)
    /// * `x`      - Input vector, f32, length `ncols`
    /// * `y`      - Output vector, f32, length `nrows` (overwritten)
    pub fn mul_mat_vec_q5_k(
        &self,
        w_data: &[u8],
        nrows: usize,
        ncols: usize,
        x: &[f32],
        y: &mut [f32],
    ) {
        let result = q5k_matmul_cpu(w_data, x, nrows, ncols);
        y[..nrows].copy_from_slice(&result);
    }

    /// Dispatch mul_mat_vec for the given ggml type.
    /// Currently supports Q8_0, Q5_K, and F32. Returns Err for unsupported types.
    pub fn mul_mat_vec(
        &self,
        w_data: &[u8],
        w_type: GgmlType,
        nrows: usize,
        ncols: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<(), String> {
        match w_type {
            GgmlType::Q8_0 => {
                self.mul_mat_vec_q8_0(w_data, nrows, ncols, x, y);
                Ok(())
            }
            GgmlType::Q5_K => {
                self.mul_mat_vec_q5_k(w_data, nrows, ncols, x, y);
                Ok(())
            }
            GgmlType::F32 => {
                self.mul_mat_vec_f32(w_data, nrows, ncols, x, y);
                Ok(())
            }
            _ => Err(format!("Unsupported ggml type for mul_mat_vec: {:?}", w_type)),
        }
    }
}

/// Convert IEEE 754 half-precision (f16) to f32.
///
/// Handles normal, subnormal, infinity, and NaN values correctly.
/// This matches ggml's `GGML_FP16_TO_FP32` macro.
#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            f32::from_bits(sign << 31)
        } else {
            // Subnormal f16 -> normal f32
            // value = (-1)^sign * 2^(-14) * (mant / 1024)
            let mut m = mant;
            let mut e: i32 = -14 + 127; // f32 bias
            // Normalize: shift mantissa left until leading 1 is in bit 10
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF; // remove the leading 1
            let f32_bits = (sign << 31) | ((e as u32) << 23) | (m << 13);
            f32::from_bits(f32_bits)
        }
    } else if exp == 31 {
        // Infinity or NaN
        let f32_bits = (sign << 31) | (0xFF << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    } else {
        // Normal f16 -> f32
        // f16 exponent bias = 15, f32 bias = 127
        // Use i32 arithmetic to avoid u32 underflow when exp < 15
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        let f32_bits = (sign << 31) | (f32_exp << 23) | (mant << 13);
        f32::from_bits(f32_bits)
    }
}

/// Dequantize a Q8_0 tensor to f32.
///
/// Returns a Vec<f32> of length `n_elements`.
/// `raw` must contain `(n_elements / 32) * 34` bytes.
pub fn dequant_q8_0_to_f32(raw: &[u8], n_elements: usize) -> Vec<f32> {
    assert_eq!(n_elements % QK8_0, 0);
    let n_blocks = n_elements / QK8_0;
    assert_eq!(raw.len(), n_blocks * Q8_0_BLOCK_BYTES);

    let mut out = vec![0.0f32; n_elements];

    for blk in 0..n_blocks {
        let offset = blk * Q8_0_BLOCK_BYTES;
        let scale_u16 = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
        let scale = f16_to_f32(scale_u16);
        let qs_start = offset + 2;
        let out_start = blk * QK8_0;

        for i in 0..QK8_0 {
            let qi = raw[qs_start + i] as i8;
            out[out_start + i] = scale * (qi as f32);
        }
    }

    out
}

// ============================================================================
// Q8_1 quantization
// ============================================================================

/// Quantize a slice of f32 values into Q8_1 blocks.
///
/// Q8_1 block layout (40 bytes per 32 elements):
///   - d:     f32 (4 bytes) -- scale = amax / 127
///   - s:     f32 (4 bytes) -- s = d * sum(qs[0..32])
///   - qs[32]: i8 (32 bytes) -- quantized values
///
/// Ported from ggml `quantize_row_q8_1_ref`.
pub fn quantize_row_q8_1(input: &[f32]) -> Vec<u8> {
    assert_eq!(input.len() % QK8_1, 0, "input length must be multiple of {QK8_1}");
    let nb = input.len() / QK8_1;
    let mut output = vec![0u8; nb * Q8_1_BLOCK_BYTES];

    for i in 0..nb {
        let block_input = &input[i * QK8_1..(i + 1) * QK8_1];
        let blk_offset = i * Q8_1_BLOCK_BYTES;

        // Find absolute maximum
        let mut amax: f32 = 0.0;
        for &v in block_input {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        let d: f32 = amax / 127.0;
        let id: f32 = if d != 0.0 { 1.0 / d } else { 0.0 };

        // Write d as f32 LE at offset 0
        output[blk_offset..blk_offset + 4].copy_from_slice(&d.to_le_bytes());

        // Quantize and compute sum of quantized values
        let mut sum_qs: i32 = 0;
        let qs_offset = blk_offset + 8;
        for j in 0..QK8_1 {
            let v = (block_input[j] * id).round();
            let qi = v.max(-128.0).min(127.0) as i8;
            output[qs_offset + j] = qi as u8;
            sum_qs += qi as i32;
        }

        // s = d * sum(qs) -- the weighted sum for "min" contribution
        let s: f32 = d * (sum_qs as f32);
        output[blk_offset + 4..blk_offset + 8].copy_from_slice(&s.to_le_bytes());
    }

    output
}

// ============================================================================
// Q5_K helpers
// ============================================================================

/// Extract 6-bit scale and min from the packed scales[12] array in Q5_K/Q4_K.
///
/// For j < 4: lower 6 bits of scales[j] and scales[j+4].
/// For j >= 4: combined from high bits of scales[j-4]/scales[j] and scales[j+4].
///
/// Ported from ggml `get_scale_min_k4`.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

/// Compute the dot product of one Q5_K super-block (256 elements) with
/// 8 Q8_1 blocks (8 x 32 = 256 elements).
///
/// Faithfully ported from ggml's scalar `ggml_vec_dot_q5_K_q8_K` fallback,
/// adapted for Q8_1 blocks instead of Q8_K.
///
/// The formula per sub-block pair:
///   result += d_q5k * d_q8_1 * sum(q5_val[i] * q8_qs[i]) * sub_scale
///   result -= dmin_q5k * q8_1.s * sub_min
///
/// Uses f64 accumulation for maximum precision.
pub fn vec_dot_q5_k_q8_1(bx: &[u8], by: &[u8]) -> f32 {
    assert!(bx.len() >= Q5_K_BLOCK_BYTES, "Q5_K block too short: {}", bx.len());
    assert!(
        by.len() >= Q8_1_PER_Q5K * Q8_1_BLOCK_BYTES,
        "Q8_1 blocks too short: {} < {}",
        by.len(),
        Q8_1_PER_Q5K * Q8_1_BLOCK_BYTES
    );

    // Parse Q5_K super-block header
    let d_bits = u16::from_le_bytes([bx[0], bx[1]]);
    let d = f16_to_f32(d_bits) as f64;

    let dmin_bits = u16::from_le_bytes([bx[2], bx[3]]);
    let dmin = f16_to_f32(dmin_bits) as f64;

    let scales = &bx[4..16];    // 12 bytes of packed 6-bit scales+mins
    let qh = &bx[16..48];       // 32 bytes of high bits
    let qs = &bx[48..176];      // 128 bytes of low nibbles

    // First pass: decode all 256 Q5_K values into a temporary buffer.
    // This follows ggml's scalar fallback exactly.
    let mut aux8 = [0i8; QK_K]; // decoded 5-bit values (0..31 range, unsigned)
    {
        let mut a_off = 0usize;
        let mut ql_off = 0usize;
        let mut m: u8 = 1;
        for _j in 0..QK_K / 64 {
            // First 32 of this group: low nibble + high bit
            for l in 0..32 {
                let low = (qs[ql_off + l] & 0xF) as i8;
                let high = if qh[l] & m != 0 { 16i8 } else { 0i8 };
                aux8[a_off + l] = low + high;
            }
            a_off += 32;
            m <<= 1;
            // Second 32 of this group: high nibble + high bit
            for l in 0..32 {
                let low = (qs[ql_off + l] >> 4) as i8;
                let high = if qh[l] & m != 0 { 16i8 } else { 0i8 };
                aux8[a_off + l] = low + high;
            }
            a_off += 32;
            m <<= 1;
            ql_off += 32;
        }
    }

    // Unpack the 6-bit scales and mins using the utmp[] trick from ggml.
    // We use get_scale_min_k4 for clarity (same result, easier to verify).
    //
    // There are 8 sub-blocks of 32 elements each.
    // Sub-block j uses scale[j] and min[j].
    let mut unpacked_scales = [0u8; 8];
    let mut unpacked_mins = [0u8; 8];
    for j in 0..8 {
        let (sc, mi) = get_scale_min_k4(j, scales);
        unpacked_scales[j] = sc;
        unpacked_mins[j] = mi;
    }

    // Compute the dot product
    let mut sumf = 0.0f64;
    let mut sumi_mins = 0.0f64;

    let mut a_off = 0usize;
    for j in 0..8 {
        let q8_blk_offset = j * Q8_1_BLOCK_BYTES;

        // Read Q8_1 block header
        let d_q8 = f32::from_le_bytes([
            by[q8_blk_offset],
            by[q8_blk_offset + 1],
            by[q8_blk_offset + 2],
            by[q8_blk_offset + 3],
        ]) as f64;

        let s_q8 = f32::from_le_bytes([
            by[q8_blk_offset + 4],
            by[q8_blk_offset + 5],
            by[q8_blk_offset + 6],
            by[q8_blk_offset + 7],
        ]) as f64;

        // Dot product: sum(q5_val[i] * q8_qs[i]) for 32 elements
        let q8_qs_offset = q8_blk_offset + 8;
        let mut dot: i64 = 0;
        for l in 0..QK8_1 {
            let q5_val = aux8[a_off + l] as i32;
            let q8_val = by[q8_qs_offset + l] as i8 as i32;
            dot += (q5_val * q8_val) as i64;
        }

        let scale = unpacked_scales[j] as f64;
        let min = unpacked_mins[j] as f64;

        // Accumulate: d * d_q8 * scale * dot
        sumf += d * d_q8 * scale * (dot as f64);

        // Min contribution: dmin * min * s_q8  (s_q8 = d_q8 * sum(qs))
        sumi_mins += dmin * min * s_q8;

        a_off += QK8_1;
    }

    (sumf - sumi_mins) as f32
}

// ============================================================================
// Q5_K GEMV (mul_mat_vec)
// ============================================================================

/// Compute y = W @ x where W is Q5_K and x is pre-quantized Q8_1.
///
/// * `weight` - Raw Q5_K bytes, row-major
/// * `input_q8_1` - Pre-quantized Q8_1 bytes for the input vector
/// * `nrows` - Number of rows (output dimension)
/// * `ncols` - Number of columns (input dimension, must be multiple of 256)
fn mul_mat_vec_q5k(weight: &[u8], input_q8_1: &[u8], nrows: usize, ncols: usize) -> Vec<f32> {
    assert_eq!(ncols % QK_K, 0, "ncols must be multiple of {QK_K}");
    let blocks_per_row = ncols / QK_K;
    let row_bytes = blocks_per_row * Q5_K_BLOCK_BYTES;
    let q8_blocks_per_row = ncols / QK8_1;
    let q8_row_bytes = q8_blocks_per_row * Q8_1_BLOCK_BYTES;

    assert!(
        weight.len() >= nrows * row_bytes,
        "weight too short: {} < {}",
        weight.len(),
        nrows * row_bytes
    );
    assert!(
        input_q8_1.len() >= q8_row_bytes,
        "input_q8_1 too short: {} < {}",
        input_q8_1.len(),
        q8_row_bytes
    );

    let mut output = vec![0.0f32; nrows];

    for row in 0..nrows {
        let mut sum = 0.0f64;

        for blk in 0..blocks_per_row {
            let w_offset = row * row_bytes + blk * Q5_K_BLOCK_BYTES;
            let q8_offset = blk * Q8_1_PER_Q5K * Q8_1_BLOCK_BYTES;

            let dot = vec_dot_q5_k_q8_1(
                &weight[w_offset..w_offset + Q5_K_BLOCK_BYTES],
                &input_q8_1[q8_offset..q8_offset + Q8_1_PER_Q5K * Q8_1_BLOCK_BYTES],
            );
            sum += dot as f64;
        }

        output[row] = sum as f32;
    }

    output
}

/// High-level Q5_K matrix-vector multiply: W_q5k @ x_f32.
///
/// Quantizes the input vector to Q8_1, then computes the GEMV.
///
/// * `weight_q5k` - Raw Q5_K weight bytes, row-major `[nrows, ncols]`
/// * `input_f32` - Input vector, f32, length `ncols`
/// * `nrows` - Number of output features
/// * `ncols` - Number of input features (must be multiple of 256)
///
/// Returns a Vec<f32> of length `nrows`.
pub fn q5k_matmul_cpu(weight_q5k: &[u8], input_f32: &[f32], nrows: usize, ncols: usize) -> Vec<f32> {
    assert_eq!(ncols % QK_K, 0, "ncols must be multiple of {QK_K}");
    assert!(input_f32.len() >= ncols, "input too short: {} < {ncols}", input_f32.len());

    // Quantize the input vector to Q8_1 (once, reused for all rows)
    let input_q8_1 = quantize_row_q8_1(&input_f32[..ncols]);

    mul_mat_vec_q5k(weight_q5k, &input_q8_1, nrows, ncols)
}

/// Dequantize a Q5_K tensor to f32.
///
/// Returns a Vec<f32> of length `n_elements`.
pub fn dequant_q5_k_to_f32(raw: &[u8], n_elements: usize) -> Vec<f32> {
    assert_eq!(n_elements % QK_K, 0);
    let n_blocks = n_elements / QK_K;
    assert!(raw.len() >= n_blocks * Q5_K_BLOCK_BYTES);

    let mut result = Vec::with_capacity(n_elements);

    for block_idx in 0..n_blocks {
        let block = &raw[block_idx * Q5_K_BLOCK_BYTES..];

        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let d = f16_to_f32(d_bits);

        let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
        let dmin = f16_to_f32(dmin_bits);

        let scales = &block[4..16];
        let qh_base = 16;
        let qs_base = 48;

        let mut is = 0usize;
        let mut ql_off = qs_base;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;

        for _j in 0..4 {
            let (sc1, m1) = get_scale_min_k4(is, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;

            let (sc2, m2) = get_scale_min_k4(is + 1, scales);
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;

            for l in 0..32 {
                let ql = block[ql_off + l] & 0xF;
                let qh_high = if block[qh_base + l] & u1 != 0 { 16u8 } else { 0u8 };
                result.push(d1 * (ql + qh_high) as f32 - min1);
            }
            for l in 0..32 {
                let ql = block[ql_off + l] >> 4;
                let qh_high = if block[qh_base + l] & u2 != 0 { 16u8 } else { 0u8 };
                result.push(d2 * (ql + qh_high) as f32 - min2);
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    result
}

// ============================================================================
// IQ3_S FFI — ggml native AVX2 GEMV via linked libggml.so
// ============================================================================

/// IQ3_S super-block: 256 elements, 110 bytes total.
/// Layout: f16 d (2) + qs[64] + qh[8] + signs[32] + scales[4]
pub const IQ3S_BLOCK_BYTES: usize = 110;
pub const IQ3S_BLOCK_ELEMS: usize = 256;

/// Q8_K block: 256 elements, 296 bytes total.
/// Layout: f32 d (4) + f32 sum (4) + int8_t qs[256] (256) + int16_t bsums[16] (32)
pub const Q8K_BLOCK_BYTES: usize = 296;
pub const Q8K_BLOCK_ELEMS: usize = 256;

// FFI declarations -- these resolve to the C wrapper compiled in build.rs,
// which in turn calls ggml's AVX2-optimized functions from libggml.so.
#[cfg(feature = "ggml_iq3s")]
#[allow(dead_code)]
extern "C" {
    fn ggml_ffi_quantize_q8_K(x: *const f32, y: *mut u8, k: i64);

    fn ggml_ffi_vec_dot_iq3s_q8K(
        n: std::os::raw::c_int,
        s: *mut f32,
        vx: *const u8,
        vy: *const u8,
    );

    fn ggml_ffi_gemv_iq3s(
        w_iq3s: *const u8,
        x_f32: *const f32,
        y_f32: *mut f32,
        nrows: std::os::raw::c_int,
        ncols: std::os::raw::c_int,
    );

    fn ggml_ffi_gemv_iq3s_parallel(
        w_iq3s: *const u8,
        x_f32: *const f32,
        y_f32: *mut f32,
        nrows: std::os::raw::c_int,
        ncols: std::os::raw::c_int,
        nthreads: std::os::raw::c_int,
    );

    fn ggml_ffi_iq3s_row_bytes(ncols: std::os::raw::c_int) -> usize;
    fn ggml_ffi_q8k_row_bytes(ncols: std::os::raw::c_int) -> usize;
}

/// Return whether the ggml IQ3_S FFI backend is available (compiled and linked).
pub fn iq3s_ffi_available() -> bool {
    cfg!(feature = "ggml_iq3s")
}

/// Compute the byte size of one IQ3_S row of `ncols` elements.
/// `ncols` must be a multiple of 256.
pub fn iq3s_row_bytes(ncols: usize) -> usize {
    (ncols / IQ3S_BLOCK_ELEMS) * IQ3S_BLOCK_BYTES
}

/// CPU GEMV: y = W @ x where W is IQ3_S [nrows, ncols] and x is f32 [ncols].
///
/// Uses ggml's AVX2-optimized `ggml_vec_dot_iq3_s_q8_K` for each row.
/// Quantizes the input to Q8_K internally.
///
/// # Arguments
/// * `weights_iq3s` - Raw IQ3_S bytes, row-major [nrows, ncols]
/// * `input_f32`    - f32 input vector, length ncols
/// * `nrows`        - Number of output features
/// * `ncols`        - Number of input features (must be multiple of 256)
///
/// # Returns
/// Output f32 vector, length nrows.
///
/// # Panics
/// Panics if ggml_iq3s feature is not compiled, or if dimensions are invalid.
#[cfg(feature = "ggml_iq3s")]
pub fn iq3s_gemv_cpu(
    weights_iq3s: &[u8],
    input_f32: &[f32],
    nrows: usize,
    ncols: usize,
) -> Vec<f32> {
    assert_eq!(ncols % IQ3S_BLOCK_ELEMS, 0, "ncols must be multiple of {IQ3S_BLOCK_ELEMS}");
    assert!(input_f32.len() >= ncols, "input too short: {} < {ncols}", input_f32.len());

    let expected_bytes = nrows * iq3s_row_bytes(ncols);
    assert!(
        weights_iq3s.len() >= expected_bytes,
        "weights too short: {} < {expected_bytes} ({nrows} rows x {ncols} cols)",
        weights_iq3s.len()
    );

    let mut output = vec![0.0f32; nrows];
    unsafe {
        ggml_ffi_gemv_iq3s(
            weights_iq3s.as_ptr(),
            input_f32.as_ptr(),
            output.as_mut_ptr(),
            nrows as std::os::raw::c_int,
            ncols as std::os::raw::c_int,
        );
    }
    output
}

/// Parallel CPU GEMV: y = W @ x where W is IQ3_S [nrows, ncols] and x is f32 [ncols].
///
/// Same as `iq3s_gemv_cpu` but uses OpenMP for parallelism across rows.
/// `nthreads=0` lets OpenMP choose automatically.
///
/// For the i5-14600KF (14 threads), use `nthreads=14` for best throughput.
/// At 512x2048 per expert, this is ~0.3ms per expert sequentially, ~0.08ms with 4 threads.
#[cfg(feature = "ggml_iq3s")]
pub fn iq3s_gemv_cpu_parallel(
    weights_iq3s: &[u8],
    input_f32: &[f32],
    nrows: usize,
    ncols: usize,
    nthreads: usize,
) -> Vec<f32> {
    assert_eq!(ncols % IQ3S_BLOCK_ELEMS, 0, "ncols must be multiple of {IQ3S_BLOCK_ELEMS}");
    assert!(input_f32.len() >= ncols, "input too short: {} < {ncols}", input_f32.len());

    let expected_bytes = nrows * iq3s_row_bytes(ncols);
    assert!(
        weights_iq3s.len() >= expected_bytes,
        "weights too short: {} < {expected_bytes}",
        weights_iq3s.len()
    );

    let mut output = vec![0.0f32; nrows];
    unsafe {
        ggml_ffi_gemv_iq3s_parallel(
            weights_iq3s.as_ptr(),
            input_f32.as_ptr(),
            output.as_mut_ptr(),
            nrows as std::os::raw::c_int,
            ncols as std::os::raw::c_int,
            nthreads as std::os::raw::c_int,
        );
    }
    output
}

/// Stub when ggml_iq3s feature is not compiled.
#[cfg(not(feature = "ggml_iq3s"))]
pub fn iq3s_gemv_cpu(
    _weights_iq3s: &[u8],
    _input_f32: &[f32],
    _nrows: usize,
    _ncols: usize,
) -> Vec<f32> {
    panic!("IQ3_S GEMV FFI not available: ggml_iq3s feature not compiled. \
            Ensure libggml.so is available at build time.");
}

/// Stub when ggml_iq3s feature is not compiled.
#[cfg(not(feature = "ggml_iq3s"))]
pub fn iq3s_gemv_cpu_parallel(
    _weights_iq3s: &[u8],
    _input_f32: &[f32],
    _nrows: usize,
    _ncols: usize,
    _nthreads: usize,
) -> Vec<f32> {
    panic!("IQ3_S GEMV FFI not available: ggml_iq3s feature not compiled.");
}

/// SwiGLU expert forward pass on CPU using IQ3_S GEMV.
///
/// Computes: `down_proj( silu(gate_proj(x)) * up_proj(x) )`
///
/// All three weight matrices are IQ3_S quantized. The intermediate
/// activations are kept in f32 on CPU.
///
/// # Arguments
/// * `gate_w`       - IQ3_S gate weights [expert_ffn, hidden_size]
/// * `up_w`         - IQ3_S up weights [expert_ffn, hidden_size]
/// * `down_w`       - IQ3_S down weights [hidden_size, expert_ffn]
/// * `input_f32`    - f32 hidden state, length hidden_size
/// * `expert_ffn`   - Expert FFN hidden dimension (512 for Qwen3.5)
/// * `hidden_size`  - Model hidden dimension (2048 for Qwen3.5)
///
/// # Returns
/// f32 output vector, length hidden_size.
pub fn iq3s_swiglu_expert_cpu(
    gate_w: &[u8],
    up_w: &[u8],
    down_w: &[u8],
    input_f32: &[f32],
    expert_ffn: usize,
    hidden_size: usize,
) -> Vec<f32> {
    // gate = gate_w @ input  [expert_ffn]
    let gate_out = iq3s_gemv_cpu(gate_w, input_f32, expert_ffn, hidden_size);

    // up = up_w @ input  [expert_ffn]
    let up_out = iq3s_gemv_cpu(up_w, input_f32, expert_ffn, hidden_size);

    // SwiGLU: silu(gate) * up
    let mut intermediate = vec![0.0f32; expert_ffn];
    for i in 0..expert_ffn {
        let g = gate_out[i];
        let silu_g = g * (1.0 / (1.0 + (-g).exp())); // silu(x) = x * sigmoid(x)
        intermediate[i] = silu_g * up_out[i];
    }

    // down = down_w @ intermediate  [hidden_size]
    iq3s_gemv_cpu(down_w, &intermediate, hidden_size, expert_ffn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-7);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-7);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) + 1.0).abs() < 1e-7);
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -0.0
        assert_eq!(f16_to_f32(0x8000), -0.0);
        println!("f16 conversion tests passed");
    }

    #[test]
    fn test_q8_0_roundtrip() {
        // Create a simple Q8_0 block: scale=1.0, values=[0,1,2,...,31]
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        // f16 for 1.0 = 0x3C00
        block[0] = 0x00;
        block[1] = 0x3C;
        for i in 0..QK8_0 {
            block[2 + i] = i as u8; // 0..31 as unsigned, but read as signed
        }

        let dequant = dequant_q8_0_to_f32(&block, QK8_0);
        for i in 0..QK8_0 {
            let expected = 1.0 * (i as i8 as f32);
            assert!(
                (dequant[i] - expected).abs() < 1e-5,
                "dequant[{i}] = {}, expected {expected}",
                dequant[i]
            );
        }
        println!("Q8_0 roundtrip test passed");
    }

    #[test]
    fn test_mul_mat_vec_q8_0_identity() {
        // Create a Q8_0 "approximate identity" matrix.
        // For a 32x32 matrix, each row has 1 block.
        // Row i: scale = 1/127, qs[i] = 127, qs[j!=i] = 0
        // This gives W[i][i] ≈ 1.0, W[i][j] = 0.0
        let nrows = 32;
        let ncols = 32;
        let mut w_data = vec![0u8; nrows * Q8_0_BLOCK_BYTES];

        for row in 0..nrows {
            let offset = row * Q8_0_BLOCK_BYTES;
            // scale = 1.0/127.0 in f16
            // 1/127 ≈ 0.007874 → f16 ≈ 0x2009 (roughly)
            // Actually let's use scale=1.0 and qs[i]=1 for simplicity
            // W[row][row] = 1.0 * 1 = 1.0
            block_set_scale(&mut w_data[offset..offset + Q8_0_BLOCK_BYTES], 1.0);
            w_data[offset + 2 + row] = 1; // qs[row] = 1 (signed)
        }

        let x = vec![1.0f32; ncols];
        let mut y = vec![0.0f32; nrows];

        let ctx = GgmlCpuContext::new();
        ctx.mul_mat_vec_q8_0(&w_data, nrows, ncols, &x, &mut y);

        // Each row i should give: scale * 1 * x[i] = 1.0 * 1 * 1.0 = 1.0
        for i in 0..nrows {
            assert!(
                (y[i] - 1.0).abs() < 1e-4,
                "y[{i}] = {}, expected 1.0",
                y[i]
            );
        }
        println!("Q8_0 identity GEMV test passed");
    }

    /// Helper: set f16 scale in a Q8_0 block from an f32 value.
    fn block_set_scale(block: &mut [u8], scale: f32) {
        let h = f32_to_f16(scale);
        block[0] = (h & 0xFF) as u8;
        block[1] = (h >> 8) as u8;
    }

    /// Convert f32 to f16 (simple truncation, sufficient for tests).
    fn f32_to_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        if exp == 0 {
            // Zero or subnormal f32 -> zero f16
            (sign << 15) as u16
        } else if exp == 0xFF {
            // Inf or NaN
            let h_mant = mant >> 13;
            ((sign << 15) | (0x1F << 10) | h_mant) as u16
        } else {
            let h_exp = exp - 127 + 15;
            if h_exp <= 0 {
                // Underflow -> zero
                (sign << 15) as u16
            } else if h_exp >= 31 {
                // Overflow -> infinity
                ((sign << 15) | (0x1F << 10)) as u16
            } else {
                let h_mant = mant >> 13;
                ((sign << 15) | ((h_exp as u32) << 10) | h_mant) as u16
            }
        }
    }

    // ====================================================================
    // Q8_1 tests
    // ====================================================================

    #[test]
    fn test_q8_1_quantize() {
        // Test with known values: [1.0, 2.0, ..., 32.0]
        let input: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let q8_1 = quantize_row_q8_1(&input);
        assert_eq!(q8_1.len(), Q8_1_BLOCK_BYTES);

        // Read d (f32 LE at offset 0)
        let d = f32::from_le_bytes([q8_1[0], q8_1[1], q8_1[2], q8_1[3]]);
        // amax = 32.0, so d = 32.0 / 127.0
        let expected_d = 32.0f32 / 127.0;
        assert!(
            (d - expected_d).abs() < 1e-6,
            "d = {d}, expected {expected_d}"
        );

        // Read s (f32 LE at offset 4)
        let s = f32::from_le_bytes([q8_1[4], q8_1[5], q8_1[6], q8_1[7]]);
        // s = d * sum(qs), where qs[i] = round(input[i] / d)
        // Verify s matches
        let mut sum_qs: i32 = 0;
        for i in 0..32 {
            let qi = q8_1[8 + i] as i8;
            sum_qs += qi as i32;
        }
        let expected_s = d * (sum_qs as f32);
        assert!(
            (s - expected_s).abs() < 1e-5,
            "s = {s}, expected {expected_s} (sum_qs = {sum_qs})"
        );

        // Verify roundtrip: dequantize and compare with original
        for i in 0..32 {
            let qi = q8_1[8 + i] as i8;
            let deq = d * (qi as f32);
            let err = (deq - input[i]).abs();
            assert!(
                err < 0.3, // Q8 quantization error should be small
                "roundtrip[{i}]: deq={deq}, orig={}, err={err}",
                input[i]
            );
        }

        println!("Q8_1 quantize test passed: d={d:.6}, s={s:.4}, sum_qs={sum_qs}");
    }

    #[test]
    fn test_q8_1_quantize_zeros() {
        let input = vec![0.0f32; 32];
        let q8_1 = quantize_row_q8_1(&input);
        let d = f32::from_le_bytes([q8_1[0], q8_1[1], q8_1[2], q8_1[3]]);
        let s = f32::from_le_bytes([q8_1[4], q8_1[5], q8_1[6], q8_1[7]]);
        assert_eq!(d, 0.0);
        assert_eq!(s, 0.0);
        for i in 0..32 {
            assert_eq!(q8_1[8 + i], 0);
        }
        println!("Q8_1 quantize zeros test passed");
    }

    #[test]
    fn test_q8_1_quantize_negative() {
        // Test with negative values
        let input: Vec<f32> = (1..=32).map(|i| -(i as f32)).collect();
        let q8_1 = quantize_row_q8_1(&input);
        let d = f32::from_le_bytes([q8_1[0], q8_1[1], q8_1[2], q8_1[3]]);
        assert!(d > 0.0, "d should always be positive: {d}");

        // All quantized values should be negative
        for i in 0..32 {
            let qi = q8_1[8 + i] as i8;
            assert!(qi <= 0, "qs[{i}] = {qi}, expected <= 0");
        }

        // s should be negative (d positive * sum of negative qs)
        let s = f32::from_le_bytes([q8_1[4], q8_1[5], q8_1[6], q8_1[7]]);
        assert!(s <= 0.0, "s should be <= 0 for negative inputs: {s}");

        println!("Q8_1 quantize negative test passed");
    }

    // ====================================================================
    // Q5_K dot product tests
    // ====================================================================

    /// Helper: build a Q5_K block from known dequantized values.
    /// Sets d, dmin, scales, qh, qs to encode the given 256 f32 values.
    ///
    /// Uses a simple encoding: sc=1 for all sub-blocks, min=0,
    /// so value = d * q5_val - 0.
    fn build_q5k_block(values: &[f32; 256]) -> Vec<u8> {
        let mut block = vec![0u8; Q5_K_BLOCK_BYTES];

        // Find amax to set d
        let mut amax: f32 = 0.0;
        for &v in values {
            if v.abs() > amax {
                amax = v.abs();
            }
        }

        // For simplicity: use sc=1, min=0 for all sub-blocks.
        // value = d * sc * q5_val - dmin * min = d * q5_val
        // So q5_val = round(value / d), clamped to 0..31
        let d = if amax > 0.0 { amax / 31.0 } else { 0.0 };
        let dmin: f32 = 0.0;

        // Write d as f16 at offset 0
        let d_f16 = f32_to_f16(d);
        block[0] = (d_f16 & 0xFF) as u8;
        block[1] = (d_f16 >> 8) as u8;

        // Write dmin as f16 at offset 2
        let dmin_f16 = f32_to_f16(dmin);
        block[2] = (dmin_f16 & 0xFF) as u8;
        block[3] = (dmin_f16 >> 8) as u8;

        // scales[12] at offset 4: encode sc=1, min=0 for all 8 sub-blocks
        // For j < 4: scales[j] = sc & 63 = 1, scales[j+4] = min & 63 = 0
        // For j >= 4: more complex packing
        for j in 0..4 {
            block[4 + j] = 1; // sc=1 in lower 6 bits
        }
        for j in 4..8 {
            block[4 + j] = 0; // min=0 in lower 6 bits
        }
        // scales[8..12] encode high bits of j>=4, but with sc=1,min=0 they are 0
        for j in 8..12 {
            block[4 + j] = 0;
        }
        // For j>=4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4) = 0|0 = 0
        // That gives sc=0 for sub-blocks 4-7, which is wrong.
        // Let's fix: for j>=4, we need scales[j+4] low nibble = sc & 0xF = 1
        for j in 4..8 {
            // scales[j+4] = scales[8..12]
            // low nibble = sc for sub-block j
            // high nibble = min for sub-block j
            block[4 + j + 4] = 1; // sc=1 in low nibble, min=0 in high nibble
        }

        // Encode values into qs and qh
        // qh[32] at offset 16, qs[128] at offset 48
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };

        // Process 4 groups of 64
        let mut val_idx = 0usize;
        let mut ql_off = 48usize;
        let mut m: u8 = 1;

        for _group in 0..4 {
            // First 32: low nibble + high bit
            for l in 0..32 {
                let q5_val = (values[val_idx + l].abs() * id).round().max(0.0).min(31.0) as u8;
                let low4 = q5_val & 0xF;
                let high_bit = (q5_val >> 4) & 1;
                // Set low nibble of qs
                block[ql_off + l] = (block[ql_off + l] & 0xF0) | low4;
                // Set high bit in qh
                if high_bit != 0 {
                    block[16 + l] |= m;
                }
            }
            val_idx += 32;
            m <<= 1;

            // Second 32: high nibble + high bit
            for l in 0..32 {
                let q5_val = (values[val_idx + l].abs() * id).round().max(0.0).min(31.0) as u8;
                let low4 = q5_val & 0xF;
                let high_bit = (q5_val >> 4) & 1;
                // Set high nibble of qs
                block[ql_off + l] = (block[ql_off + l] & 0x0F) | (low4 << 4);
                // Set high bit in qh
                if high_bit != 0 {
                    block[16 + l] |= m;
                }
            }
            val_idx += 32;
            m <<= 1;
            ql_off += 32;
        }

        block
    }

    #[test]
    fn test_q5k_dot_vs_dequantize() {
        // Create a Q5_K block with known values, then compare:
        // 1) vec_dot_q5_k_q8_1(block, quantize(input))
        // 2) sum(dequantize(block)[i] * input[i])
        //
        // They should be very close (within quantization error).

        // Build Q5_K block with ramp values
        let mut q5_values = [0.0f32; 256];
        for i in 0..256 {
            q5_values[i] = (i as f32) * 0.1; // 0.0, 0.1, 0.2, ..., 25.5
        }
        let q5k_block = build_q5k_block(&q5_values);

        // Build input vector
        let mut input = [0.0f32; 256];
        for i in 0..256 {
            input[i] = 1.0 - (i as f32) * 0.005; // 1.0, 0.995, ..., 0.725...
        }

        // Method 1: vec_dot
        let q8_1 = quantize_row_q8_1(&input);
        let dot_result = vec_dot_q5_k_q8_1(&q5k_block, &q8_1);

        // Method 2: dequantize + f32 dot
        let deq = dequant_q5_k_to_f32(&q5k_block, 256);
        let mut ref_dot = 0.0f64;
        for i in 0..256 {
            ref_dot += (deq[i] as f64) * (input[i] as f64);
        }
        let ref_result = ref_dot as f32;

        let abs_err = (dot_result - ref_result).abs();
        let rel_err = if ref_result.abs() > 1e-6 {
            abs_err / ref_result.abs()
        } else {
            abs_err
        };

        println!(
            "Q5K dot test: vec_dot={dot_result:.6}, ref={ref_result:.6}, abs_err={abs_err:.6}, rel_err={rel_err:.6}"
        );

        // Allow up to 1% relative error (Q8_1 quantization introduces small errors)
        assert!(
            rel_err < 0.01,
            "Q5K dot product relative error too large: {rel_err:.6} (vec_dot={dot_result}, ref={ref_result})"
        );
    }

    #[test]
    fn test_q5k_dot_zero_input() {
        // Zero input should produce zero output
        let q5_values = [1.0f32; 256];
        let q5k_block = build_q5k_block(&q5_values);

        let input = [0.0f32; 256];
        let q8_1 = quantize_row_q8_1(&input);
        let result = vec_dot_q5_k_q8_1(&q5k_block, &q8_1);

        assert!(
            result.abs() < 1e-6,
            "zero input should give zero dot product, got {result}"
        );
        println!("Q5K dot zero input test passed: result={result}");
    }

    #[test]
    fn test_q5k_matmul() {
        // Small matrix: 4 rows x 256 cols (1 Q5_K block per row)
        let nrows = 4;
        let ncols = 256;

        // Build weight matrix: each row is a different ramp
        let mut weight = Vec::new();
        for row in 0..nrows {
            let mut vals = [0.0f32; 256];
            for i in 0..256 {
                vals[i] = ((row + 1) as f32) * (i as f32) * 0.01;
            }
            weight.extend_from_slice(&build_q5k_block(&vals));
        }

        // Input vector
        let mut input = vec![0.0f32; ncols];
        for i in 0..ncols {
            input[i] = 1.0 / (1.0 + i as f32);
        }

        // Q5K matmul
        let result = q5k_matmul_cpu(&weight, &input, nrows, ncols);
        assert_eq!(result.len(), nrows);

        // Reference: dequantize each row and compute f32 dot product
        for row in 0..nrows {
            let row_data = &weight[row * Q5_K_BLOCK_BYTES..(row + 1) * Q5_K_BLOCK_BYTES];
            let deq = dequant_q5_k_to_f32(row_data, ncols);
            let mut ref_val = 0.0f64;
            for i in 0..ncols {
                ref_val += (deq[i] as f64) * (input[i] as f64);
            }
            let ref_f32 = ref_val as f32;

            let abs_err = (result[row] - ref_f32).abs();
            let rel_err = if ref_f32.abs() > 1e-6 {
                abs_err / ref_f32.abs()
            } else {
                abs_err
            };

            println!(
                "row {row}: q5k_matmul={:.6}, ref={:.6}, rel_err={:.6}",
                result[row], ref_f32, rel_err
            );

            assert!(
                rel_err < 0.02,
                "row {row}: rel_err={rel_err:.6} too large (result={}, ref={ref_f32})",
                result[row]
            );
        }

        println!("Q5K matmul test passed for {nrows}x{ncols} matrix");
    }

    #[test]
    fn test_q5k_matmul_via_dispatch() {
        // Verify Q5_K works through the GgmlCpuContext dispatcher
        let nrows = 2;
        let ncols = 256;

        let mut weight = Vec::new();
        for row in 0..nrows {
            let mut vals = [0.0f32; 256];
            for i in 0..256 {
                vals[i] = ((row + 1) as f32) * (i as f32) * 0.05;
            }
            weight.extend_from_slice(&build_q5k_block(&vals));
        }

        let mut input = vec![0.0f32; ncols];
        for i in 0..ncols {
            input[i] = 0.5;
        }

        let ctx = GgmlCpuContext::new();
        let mut y = vec![0.0f32; nrows];
        let res = ctx.mul_mat_vec(&weight, GgmlType::Q5_K, nrows, ncols, &input, &mut y);
        assert!(res.is_ok(), "Q5_K dispatch failed: {:?}", res.err());

        // Verify non-zero results
        for row in 0..nrows {
            assert!(
                y[row].abs() > 1e-6,
                "row {row}: expected non-zero output, got {}",
                y[row]
            );
        }

        println!("Q5K dispatch test passed: y={:?}", y);
    }
}
