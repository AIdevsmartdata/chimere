//! TurboQuant KV cache compression (Google, ICLR 2026).
//!
//! Compresses KV cache vectors from fp16/q8_0 to 3-4 bits using:
//! 1. Random orthogonal rotation (Hadamard or QR) -- handled externally by
//!    ik_llama's `k_cache_hadamard` flag, NOT by this module.
//! 2. Lloyd-Max optimal scalar quantization per coordinate
//! 3. Bit-packing (3 or 4 bits per coordinate)
//!
//! Reference: "TurboQuant: Online Vector Quantization for KV Cache Compression"
//! (Google DeepMind, ICLR 2026)
//!
//! ## Usage in chimere-rewrite
//!
//! This module is the quantize/dequantize core. The rotation is applied
//! upstream (ik_llama Hadamard or the hybrid attention layer). Vectors
//! arriving here are already in rotated space, unit-normalized, so we
//! only do: searchsorted -> index -> bit-pack (quantize) or
//! unpack -> centroid-lookup -> rescale (dequantize).

// ---------------------------------------------------------------------------
// Lloyd-Max Codebook
// ---------------------------------------------------------------------------

/// Pre-computed Lloyd-Max codebook for a given (dimension, bits) pair.
///
/// The centroids and boundaries are computed offline by solving the
/// continuous 1-D k-means (Lloyd-Max) problem on the Beta distribution
/// that arises from random rotation of unit vectors on S^{d-1}.
///
/// For high d, each coordinate ~ N(0, 1/d); the codebook exploits the
/// exact distribution for tighter quantization.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Sorted centroid values, length = 2^bits.
    pub centroids: Vec<f32>,
    /// Sorted decision boundaries, length = 2^bits + 1.
    /// boundaries[0] = -1.0, boundaries[last] = 1.0.
    /// boundaries[i] for 0 < i < 2^bits is the midpoint between
    /// centroids[i-1] and centroids[i].
    pub boundaries: Vec<f32>,
    /// Embedding dimension this codebook was computed for.
    pub d: usize,
    /// Bits per coordinate.
    pub bits: usize,
}

impl Codebook {
    /// Lloyd-Max codebook for d=128, 3-bit (8 centroids, 9 boundaries).
    ///
    /// MSE per coord: 2.654e-4, total MSE: 0.03397
    pub fn d128_b3() -> Self {
        Self {
            centroids: vec![
                -0.188_390_613_802_078_0,
                -0.118_132_983_698_993_62,
                -0.066_580_595_315_956_85,
                -0.021_602_468_667_239_208,
                 0.021_602_468_667_239_208,
                 0.066_580_595_315_956_85,
                 0.118_132_983_698_993_62,
                 0.188_390_613_802_078_0,
            ],
            boundaries: vec![
                -1.0,
                -0.153_261_798_750_535_8,
                -0.092_356_789_507_475_24,
                -0.044_091_531_991_598_03,
                 0.0,
                 0.044_091_531_991_598_03,
                 0.092_356_789_507_475_24,
                 0.153_261_798_750_535_8,
                 1.0,
            ],
            d: 128,
            bits: 3,
        }
    }

    /// Lloyd-Max codebook for d=128, 4-bit (16 centroids, 17 boundaries).
    ///
    /// MSE per coord: 7.277e-5, total MSE: 0.00931
    pub fn d128_b4() -> Self {
        Self {
            centroids: vec![
                -0.237_627_186_730_953_57,
                -0.180_793_729_472_174_34,
                -0.141_761_654_296_733_1,
                -0.110_247_065_382_763_63,
                -0.082_792_566_673_095_79,
                -0.057_744_535_605_257_094,
                -0.034_134_028_231_120_876,
                -0.011_296_498_142_743_928,
                 0.011_296_498_142_743_841,
                 0.034_134_028_231_120_786,
                 0.057_744_535_605_257_05,
                 0.082_792_566_673_095_74,
                 0.110_247_065_382_763_59,
                 0.141_761_654_296_733_04,
                 0.180_793_729_472_174_26,
                 0.237_627_186_730_953_45,
            ],
            boundaries: vec![
                -1.0,
                -0.209_210_458_101_563_97,
                -0.161_277_691_884_453_73,
                -0.126_004_359_839_748_36,
                -0.096_519_816_027_929_71,
                -0.070_268_551_139_176_43,
                -0.045_939_281_918_188_98,
                -0.022_715_263_186_932_403,
                 0.0,
                 0.022_715_263_186_932_313,
                 0.045_939_281_918_188_92,
                 0.070_268_551_139_176_4,
                 0.096_519_816_027_929_67,
                 0.126_004_359_839_748_3,
                 0.161_277_691_884_453_65,
                 0.209_210_458_101_563_86,
                 1.0,
            ],
            d: 128,
            bits: 4,
        }
    }

    /// Lloyd-Max codebook for d=256, 3-bit (8 centroids, 9 boundaries).
    ///
    /// MSE per coord: 1.338e-4, total MSE: 0.03426
    pub fn d256_b3() -> Self {
        Self {
            centroids: vec![
                -0.133_847_944_032_317_53,
                -0.083_758_967_753_885_6,
                -0.047_161_934_537_344_92,
                -0.015_295_735_736_674_07,
                 0.015_295_735_736_674_029,
                 0.047_161_934_537_344_86,
                 0.083_758_967_753_885_53,
                 0.133_847_944_032_317_53,
            ],
            boundaries: vec![
                -1.0,
                -0.108_803_455_893_101_56,
                -0.065_460_451_145_615_26,
                -0.031_228_835_137_009_497,
                 0.0,
                 0.031_228_835_137_009_442,
                 0.065_460_451_145_615_19,
                 0.108_803_455_893_101_53,
                 1.0,
            ],
            d: 256,
            bits: 3,
        }
    }

    /// Lloyd-Max codebook for d=256, 4-bit (16 centroids, 17 boundaries).
    ///
    /// MSE per coord: 3.675e-5, total MSE: 0.00941
    pub fn d256_b4() -> Self {
        Self {
            centroids: vec![
                -0.169_383_402_376_314_94,
                -0.128_557_349_658_639_1,
                -0.100_666_365_777_972_7,
                -0.078_219_396_324_058_14,
                -0.058_706_167_466_790_915,
                -0.040_929_159_691_844_436,
                -0.024_188_228_109_530_408,
                -0.008_004_050_632_916_976,
                 0.008_004_050_632_916_976,
                 0.024_188_228_109_530_408,
                 0.040_929_159_691_844_436,
                 0.058_706_167_466_790_915,
                 0.078_219_396_324_058_14,
                 0.100_666_365_777_972_7,
                 0.128_557_349_658_639_1,
                 0.169_383_402_376_314_94,
            ],
            boundaries: vec![
                -1.0,
                -0.148_970_376_017_477_04,
                -0.114_611_857_718_305_9,
                -0.089_442_881_051_015_42,
                -0.068_462_781_895_424_53,
                -0.049_817_663_579_317_675,
                -0.032_558_693_900_687_42,
                -0.016_096_139_371_223_693,
                 0.0,
                 0.016_096_139_371_223_693,
                 0.032_558_693_900_687_42,
                 0.049_817_663_579_317_675,
                 0.068_462_781_895_424_53,
                 0.089_442_881_051_015_42,
                 0.114_611_857_718_305_9,
                 0.148_970_376_017_477_04,
                 1.0,
            ],
            d: 256,
            bits: 4,
        }
    }

    /// Select the appropriate codebook for a given (dimension, bits) pair.
    ///
    /// Returns `None` if no pre-computed codebook exists for the combination.
    pub fn for_config(d: usize, bits: usize) -> Option<Self> {
        match (d, bits) {
            (128, 3) => Some(Self::d128_b3()),
            (128, 4) => Some(Self::d128_b4()),
            (256, 3) => Some(Self::d256_b3()),
            (256, 4) => Some(Self::d256_b4()),
            _ => None,
            // TODO: add d=64 codebooks (for smaller head_dim models)
            // TODO: add d=576 codebook (Qwen3.5 Q head_dim=512 asymmetric case)
        }
    }

    /// Interior decision boundaries (excludes the -1.0 and 1.0 endpoints).
    /// These are the thresholds for `searchsorted` during quantization.
    pub fn decision_boundaries(&self) -> &[f32] {
        &self.boundaries[1..self.boundaries.len() - 1]
    }
}

// ---------------------------------------------------------------------------
// Quantized vector storage
// ---------------------------------------------------------------------------

/// A quantized vector: bit-packed indices + the original L2 norm.
///
/// The indices encode which centroid each coordinate maps to.
/// After dequantization, the centroid values are looked up and the
/// result is rescaled by `norm` to recover approximate magnitude.
#[derive(Debug, Clone)]
pub struct QuantizedVec {
    /// Bit-packed quantization indices.
    /// For 3-bit: every 8 indices packed into 3 bytes (24 bits).
    /// For 4-bit: every 2 indices packed into 1 byte.
    pub packed: Vec<u8>,
    /// L2 norm of the original (pre-normalization) vector.
    pub norm: f32,
    /// Number of elements in the original vector (needed for unpacking).
    pub n_elements: usize,
    /// Bits per coordinate (3 or 4).
    pub bits: usize,
}

// ---------------------------------------------------------------------------
// Bit-packing utilities
// ---------------------------------------------------------------------------

/// Pack 3-bit indices into bytes.
///
/// Layout: 8 indices -> 3 bytes (24 bits).
///   byte0 = idx[0] | (idx[1] << 3) | (idx[2] << 6)        -- idx[2] contributes 2 bits
///   byte1 = (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7)  -- idx[5] contributes 1 bit
///   byte2 = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5)
///
/// This is a tight packing with no wasted bits, unlike the Python reference
/// which rounds up to 4-bit for simplicity.
fn pack_3bit(indices: &[u8]) -> Vec<u8> {
    let n = indices.len();
    // Ceiling division: (n * 3 + 7) / 8
    let packed_len = (n * 3 + 7) / 8;
    let mut packed = vec![0u8; packed_len];

    let mut bit_offset: usize = 0;
    for &idx in indices {
        debug_assert!(idx < 8, "3-bit index must be < 8, got {}", idx);
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        // Write 3 bits starting at bit_pos in byte_pos.
        // May span two bytes if bit_pos > 5.
        packed[byte_pos] |= (idx & 0x07) << bit_pos;
        if bit_pos > 5 && byte_pos + 1 < packed_len {
            packed[byte_pos + 1] |= (idx & 0x07) >> (8 - bit_pos);
        }

        bit_offset += 3;
    }

    packed
}

/// Unpack 3-bit indices from packed bytes.
fn unpack_3bit(packed: &[u8], n_elements: usize) -> Vec<u8> {
    let mut indices = Vec::with_capacity(n_elements);

    let mut bit_offset: usize = 0;
    for _ in 0..n_elements {
        let byte_pos = bit_offset / 8;
        let bit_pos = bit_offset % 8;

        let mut val = (packed[byte_pos] >> bit_pos) & 0x07;
        if bit_pos > 5 && byte_pos + 1 < packed.len() {
            val |= (packed[byte_pos + 1] << (8 - bit_pos)) & 0x07;
        }

        indices.push(val);
        bit_offset += 3;
    }

    indices
}

/// Pack 4-bit indices into bytes (2 per byte, little-endian nibble order).
fn pack_4bit(indices: &[u8]) -> Vec<u8> {
    let packed_len = (indices.len() + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    for (i, &idx) in indices.iter().enumerate() {
        debug_assert!(idx < 16, "4-bit index must be < 16, got {}", idx);
        let byte_pos = i / 2;
        if i % 2 == 0 {
            packed[byte_pos] |= idx & 0x0F;
        } else {
            packed[byte_pos] |= (idx & 0x0F) << 4;
        }
    }

    packed
}

/// Unpack 4-bit indices from packed bytes.
fn unpack_4bit(packed: &[u8], n_elements: usize) -> Vec<u8> {
    let mut indices = Vec::with_capacity(n_elements);

    for i in 0..n_elements {
        let byte_pos = i / 2;
        let val = if i % 2 == 0 {
            packed[byte_pos] & 0x0F
        } else {
            (packed[byte_pos] >> 4) & 0x0F
        };
        indices.push(val);
    }

    indices
}

// ---------------------------------------------------------------------------
// TurboQuantizer
// ---------------------------------------------------------------------------

/// TurboQuant MSE quantizer for a specific (dimension, bits) configuration.
///
/// The rotation is handled externally (ik_llama `k_cache_hadamard` flag or
/// the hybrid attention layer). This quantizer only performs:
///   - L2 normalization + norm extraction
///   - Per-coordinate scalar quantization via Lloyd-Max codebook lookup
///   - Bit-packing
///
/// ## Example
///
/// ```ignore
/// let tq = TurboQuantizer::new(128, 3).unwrap();
/// let vec = vec![0.1f32; 128]; // already rotated
/// let qvec = tq.quantize(&vec);
/// let reconstructed = tq.dequantize(&qvec);
/// // reconstructed is an approximation of vec
/// ```
pub struct TurboQuantizer {
    codebook: Codebook,
}

impl TurboQuantizer {
    /// Create a new quantizer for the given dimension and bit-width.
    ///
    /// Returns `None` if no pre-computed codebook exists for this configuration.
    /// Currently supports d in {128, 256} and bits in {3, 4}.
    pub fn new(d: usize, bits: usize) -> Option<Self> {
        Codebook::for_config(d, bits).map(|codebook| Self { codebook })
    }

    /// The codebook used by this quantizer.
    pub fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    /// Quantize a single vector (assumed already rotated by Hadamard/QR).
    ///
    /// Steps:
    /// 1. Compute L2 norm and normalize to unit sphere
    /// 2. For each coordinate, find the nearest centroid via binary search
    ///    on decision boundaries (equivalent to `searchsorted`)
    /// 3. Bit-pack the resulting indices
    pub fn quantize(&self, vec: &[f32]) -> QuantizedVec {
        let d = self.codebook.d;
        assert_eq!(vec.len(), d, "vector length {} != codebook dimension {}", vec.len(), d);

        // Step 1: L2 norm + normalize
        let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let inv_norm = if norm > 1e-10 { 1.0 / norm } else { 0.0 };

        // Step 2: per-coordinate quantization
        let decision = self.codebook.decision_boundaries();
        let n_clusters = 1usize << self.codebook.bits;
        let mut indices = Vec::with_capacity(d);

        for &x in vec {
            let x_unit = x * inv_norm;

            // Binary search on interior boundaries to find the bucket.
            // decision.len() = n_clusters - 1.
            // If x_unit <= decision[0] -> index 0
            // If x_unit > decision[last] -> index n_clusters-1
            // Otherwise: index i such that decision[i-1] < x_unit <= decision[i]
            //
            // This is equivalent to searchsorted(decision_boundaries, x_unit)
            // which returns the insertion point.
            let idx = match decision.binary_search_by(|b| b.partial_cmp(&x_unit).unwrap()) {
                Ok(i) => i + 1, // exact match on boundary -> upper bucket
                Err(i) => i,    // insertion point = number of boundaries < x_unit
            };
            // Clamp to valid range
            let idx = idx.min(n_clusters - 1);
            indices.push(idx as u8);
        }

        // Step 3: bit-pack
        let packed = match self.codebook.bits {
            3 => pack_3bit(&indices),
            4 => pack_4bit(&indices),
            _ => indices.clone(), // fallback: 1 byte per index
        };

        QuantizedVec {
            packed,
            norm,
            n_elements: d,
            bits: self.codebook.bits,
        }
    }

    /// Dequantize back to float vector (still in rotated space).
    ///
    /// Steps:
    /// 1. Unpack bit-packed indices
    /// 2. Look up centroid value for each coordinate
    /// 3. Rescale by the stored L2 norm
    pub fn dequantize(&self, qvec: &QuantizedVec) -> Vec<f32> {
        assert_eq!(qvec.bits, self.codebook.bits);

        // Step 1: unpack
        let indices = match qvec.bits {
            3 => unpack_3bit(&qvec.packed, qvec.n_elements),
            4 => unpack_4bit(&qvec.packed, qvec.n_elements),
            _ => qvec.packed.clone(),
        };

        // Step 2 + 3: centroid lookup + rescale
        let centroids = &self.codebook.centroids;
        let norm = qvec.norm;

        indices
            .iter()
            .map(|&idx| centroids[idx as usize] * norm)
            .collect()
    }

    /// Compute the MSE between original and dequantized vectors.
    /// Useful for quality validation.
    pub fn round_trip_mse(&self, vec: &[f32]) -> f32 {
        let qvec = self.quantize(vec);
        let reconstructed = self.dequantize(&qvec);
        let d = vec.len() as f32;

        vec.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / d
    }
}

// TODO: Integration hooks for chimere KV cache pipeline:
//   - `quantize_kv_head(head: &Tensor, bits: usize) -> QuantizedVec`
//     wrapping candle Tensor -> slice -> quantize
//   - `dequantize_kv_head(qvec: &QuantizedVec, device: &Device) -> Tensor`
//     dequantize -> candle Tensor on device
//   - Batch quantize/dequantize for full KV cache layers
//   - CUDA kernel for fused searchsorted + pack (avoid CPU round-trip)

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Codebook sanity --

    #[test]
    fn codebook_d128_b3_shape() {
        let cb = Codebook::d128_b3();
        assert_eq!(cb.centroids.len(), 8);     // 2^3
        assert_eq!(cb.boundaries.len(), 9);    // 2^3 + 1
        assert_eq!(cb.d, 128);
        assert_eq!(cb.bits, 3);
    }

    #[test]
    fn codebook_d128_b4_shape() {
        let cb = Codebook::d128_b4();
        assert_eq!(cb.centroids.len(), 16);    // 2^4
        assert_eq!(cb.boundaries.len(), 17);   // 2^4 + 1
    }

    #[test]
    fn codebook_d256_b3_shape() {
        let cb = Codebook::d256_b3();
        assert_eq!(cb.centroids.len(), 8);
        assert_eq!(cb.boundaries.len(), 9);
        assert_eq!(cb.d, 256);
    }

    #[test]
    fn codebook_d256_b4_shape() {
        let cb = Codebook::d256_b4();
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 17);
    }

    #[test]
    fn codebook_centroids_sorted() {
        for cb in [Codebook::d128_b3(), Codebook::d128_b4(), Codebook::d256_b3(), Codebook::d256_b4()] {
            for w in cb.centroids.windows(2) {
                assert!(w[0] < w[1], "centroids not sorted: {} >= {}", w[0], w[1]);
            }
        }
    }

    #[test]
    fn codebook_boundaries_sorted() {
        for cb in [Codebook::d128_b3(), Codebook::d128_b4(), Codebook::d256_b3(), Codebook::d256_b4()] {
            for w in cb.boundaries.windows(2) {
                assert!(w[0] < w[1], "boundaries not sorted: {} >= {}", w[0], w[1]);
            }
            assert_eq!(cb.boundaries[0], -1.0);
            assert_eq!(*cb.boundaries.last().unwrap(), 1.0);
        }
    }

    #[test]
    fn codebook_symmetry() {
        // Lloyd-Max on symmetric distribution should produce antisymmetric centroids
        let cb = Codebook::d128_b3();
        let n = cb.centroids.len();
        for i in 0..n / 2 {
            let lo = cb.centroids[i];
            let hi = cb.centroids[n - 1 - i];
            assert!((lo + hi).abs() < 1e-6, "centroids not symmetric: {} + {} != 0", lo, hi);
        }
    }

    #[test]
    fn codebook_for_config_valid() {
        assert!(Codebook::for_config(128, 3).is_some());
        assert!(Codebook::for_config(128, 4).is_some());
        assert!(Codebook::for_config(256, 3).is_some());
        assert!(Codebook::for_config(256, 4).is_some());
    }

    #[test]
    fn codebook_for_config_invalid() {
        assert!(Codebook::for_config(64, 3).is_none());
        assert!(Codebook::for_config(128, 5).is_none());
        assert!(Codebook::for_config(512, 3).is_none());
    }

    // -- Bit-packing round-trips --

    #[test]
    fn pack_unpack_3bit_round_trip() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 8) as u8).collect();
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed, 128);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_3bit_all_values() {
        // Test all 8 possible 3-bit values
        let indices: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed, 8);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_3bit_odd_length() {
        // Non-multiple-of-8 length
        let indices: Vec<u8> = vec![3, 7, 1, 5, 2];
        let packed = pack_3bit(&indices);
        let unpacked = unpack_3bit(&packed, 5);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_round_trip() {
        let indices: Vec<u8> = (0..128).map(|i| (i % 16) as u8).collect();
        let packed = pack_4bit(&indices);
        let unpacked = unpack_4bit(&packed, 128);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_all_values() {
        let indices: Vec<u8> = (0..16).collect();
        let packed = pack_4bit(&indices);
        assert_eq!(packed.len(), 8); // 16 nibbles = 8 bytes
        let unpacked = unpack_4bit(&packed, 16);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_unpack_4bit_odd_length() {
        let indices: Vec<u8> = vec![0xA, 0x3, 0xF];
        let packed = pack_4bit(&indices);
        let unpacked = unpack_4bit(&packed, 3);
        assert_eq!(indices, unpacked);
    }

    #[test]
    fn pack_3bit_compression_ratio() {
        // 128 elements * 3 bits = 384 bits = 48 bytes (vs 128 raw)
        let indices: Vec<u8> = vec![0; 128];
        let packed = pack_3bit(&indices);
        assert_eq!(packed.len(), 48);
    }

    #[test]
    fn pack_4bit_compression_ratio() {
        // 128 elements * 4 bits = 512 bits = 64 bytes (vs 128 raw)
        let indices: Vec<u8> = vec![0; 128];
        let packed = pack_4bit(&indices);
        assert_eq!(packed.len(), 64);
    }

    // -- Quantizer round-trip --

    #[test]
    fn quantizer_new_valid() {
        assert!(TurboQuantizer::new(128, 3).is_some());
        assert!(TurboQuantizer::new(256, 4).is_some());
    }

    #[test]
    fn quantizer_new_invalid() {
        assert!(TurboQuantizer::new(64, 3).is_none());
    }

    #[test]
    fn quantize_dequantize_round_trip_d128_b3() {
        let tq = TurboQuantizer::new(128, 3).unwrap();

        // Simulate a unit-ish vector (already rotated)
        let vec: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01) - 0.64).collect();
        let qvec = tq.quantize(&vec);
        let recon = tq.dequantize(&qvec);

        assert_eq!(recon.len(), 128);
        // MSE should be reasonable (not zero, but small relative to signal)
        let mse: f32 = vec.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / 128.0;
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn quantize_dequantize_round_trip_d256_b4() {
        let tq = TurboQuantizer::new(256, 4).unwrap();

        let vec: Vec<f32> = (0..256).map(|i| (i as f32 * 0.005) - 0.64).collect();
        let qvec = tq.quantize(&vec);
        let recon = tq.dequantize(&qvec);

        assert_eq!(recon.len(), 256);
        let mse: f32 = vec.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / 256.0;
        assert!(mse < 0.01, "MSE too high: {}", mse);
    }

    #[test]
    fn quantize_preserves_norm() {
        let tq = TurboQuantizer::new(128, 3).unwrap();
        let vec: Vec<f32> = vec![0.5; 128]; // norm = sqrt(128 * 0.25) = sqrt(32)
        let qvec = tq.quantize(&vec);
        let expected_norm = (128.0_f32 * 0.25).sqrt();
        assert!((qvec.norm - expected_norm).abs() < 1e-5);
    }

    #[test]
    fn quantize_zero_vector() {
        let tq = TurboQuantizer::new(128, 3).unwrap();
        let vec = vec![0.0f32; 128];
        let qvec = tq.quantize(&vec);
        assert!(qvec.norm < 1e-10);
        let recon = tq.dequantize(&qvec);
        // Dequantized zero vector should also be all zeros (norm=0 rescale)
        for &v in &recon {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn quantize_negative_vector() {
        let tq = TurboQuantizer::new(128, 3).unwrap();
        let vec: Vec<f32> = vec![-0.3; 128];
        let qvec = tq.quantize(&vec);
        let recon = tq.dequantize(&qvec);
        // All reconstructed values should be negative (same sign as input)
        for &v in &recon {
            assert!(v < 0.0, "expected negative, got {}", v);
        }
    }

    #[test]
    fn round_trip_mse_method() {
        let tq = TurboQuantizer::new(128, 4).unwrap();
        let vec: Vec<f32> = (0..128).map(|i| ((i as f32) / 64.0 - 1.0) * 0.1).collect();
        let mse = tq.round_trip_mse(&vec);
        assert!(mse >= 0.0);
        assert!(mse < 0.01);
    }

    // -- Boundary assignment test --

    #[test]
    fn quantize_boundary_assignment() {
        // Values at exact centroids should quantize to themselves
        let tq = TurboQuantizer::new(128, 3).unwrap();
        let cb = Codebook::d128_b3();

        // Create a vector where all coordinates are the same centroid value,
        // scaled by a known norm. After normalization, they should all be
        // that centroid value.
        //
        // For a constant vector [c, c, ..., c] of length d:
        //   norm = |c| * sqrt(d)
        //   unit = c / (|c| * sqrt(d)) = sign(c) / sqrt(d)
        //
        // So we pick a centroid value and construct a vector whose unit form
        // has all coordinates equal to that centroid.
        let centroid_val = cb.centroids[3]; // e.g. -0.0216...
        let scale = centroid_val * (128.0_f32).sqrt();
        let vec = vec![scale; 128];

        let qvec = tq.quantize(&vec);
        let recon = tq.dequantize(&qvec);

        // Each reconstructed value should be close to the input
        let mse: f32 = vec.iter().zip(recon.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / 128.0;
        assert!(mse < 1e-8, "centroid vector should have near-zero MSE, got {}", mse);
    }
}
