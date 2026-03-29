//! # Engram — Hierarchical Memory Codebook
//!
//! The Engram is a shared memory layer that sits above the per-layer DeltaNet
//! state matrices. It provides:
//!
//! 1. **Hierarchical addressing** via the Poincare ball model of hyperbolic space
//!    - Entries near the origin = general concepts (high fan-out)
//!    - Entries near the boundary = specialized details (low fan-out)
//!    - Distance: d(x,y) = arcosh(1 + 2||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
//!
//! 2. **MDL-driven updates** — new entries are only added when the DeltaNet
//!    prediction error (mean_delta) exceeds a threshold, achieving minimum
//!    description length compression of the sequence history
//!
//! 3. **Entropy signal** — the codebook utilization (how many entries are active)
//!    feeds back to the MoE router as an additional entropy signal
//!
//! ## Connection to Other Modules
//!
//! ```text
//! GatedDeltaNet → StateMetrics.mean_delta → Engram (update gate)
//! Engram → codebook utilization → MoeRouter (entropy signal)
//! Engram → nearest entries → GatedDeltaNet (memory augmentation)
//! ```
//!
//! ## Hyperbolic Space Intuition
//!
//! The Poincare ball has exponentially more "room" near the boundary than
//! near the center. This naturally encodes hierarchies:
//! - "animal" sits near the center (general, connects to many things)
//! - "mammal" sits further out (more specific)
//! - "golden retriever" sits near the boundary (very specific)
//!
//! The hyperbolic distance respects this: d(animal, mammal) < d(mammal, retriever)
//! even if the Euclidean distances are similar, because the metric tensor
//! scales as 1/(1-||x||^2)^2.

use crate::StateMetrics;

// ---------------------------------------------------------------------------
// Poincare ball operations (standalone, testable)
// ---------------------------------------------------------------------------

/// Compute the Poincare disk distance between two points in the open unit ball.
///
/// d(x, y) = arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))
///
/// Both x and y must have ||.|| < 1 (inside the open unit ball).
pub fn poincare_distance(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "Vectors must have same dimension");

    let diff_sq: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    let x_sq: f32 = x.iter().map(|a| a.powi(2)).sum();
    let y_sq: f32 = y.iter().map(|a| a.powi(2)).sum();

    let denom = (1.0 - x_sq) * (1.0 - y_sq);
    if denom <= 0.0 {
        // Points on or outside the boundary — return large distance
        return f32::MAX;
    }

    let arg = 1.0 + 2.0 * diff_sq / denom;
    // arcosh(x) = ln(x + sqrt(x^2 - 1)), for x >= 1
    if arg < 1.0 {
        0.0
    } else {
        (arg + (arg * arg - 1.0).sqrt()).ln()
    }
}

/// Project a point into the Poincare ball (||x|| < 1 - epsilon).
/// This is necessary after gradient updates to keep points valid.
pub fn poincare_project(x: &mut [f32], max_norm: f32) {
    let norm_sq: f32 = x.iter().map(|a| a.powi(2)).sum();
    let norm = norm_sq.sqrt();
    if norm >= max_norm {
        let scale = (max_norm - 1e-5) / norm;
        for v in x.iter_mut() {
            *v *= scale;
        }
    }
}

/// Compute the norm of a point in the Poincare ball.
/// Points near 0 are "general", points near 1 are "specific".
pub fn poincare_depth(x: &[f32]) -> f32 {
    let norm_sq: f32 = x.iter().map(|a| a.powi(2)).sum();
    norm_sq.sqrt()
}

/// Mobius addition in the Poincare ball model.
/// This is the proper "translation" operation in hyperbolic space.
///
/// x (+) y = ((1 + 2<x,y> + ||y||^2) * x + (1 - ||x||^2) * y)
///           / (1 + 2<x,y> + ||x||^2 * ||y||^2)
pub fn mobius_add(x: &[f32], y: &[f32]) -> Vec<f32> {
    let d = x.len();
    assert_eq!(d, y.len());

    let xy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let x_sq: f32 = x.iter().map(|a| a.powi(2)).sum();
    let y_sq: f32 = y.iter().map(|a| a.powi(2)).sum();

    let num_x = 1.0 + 2.0 * xy + y_sq;
    let num_y = 1.0 - x_sq;
    let denom = 1.0 + 2.0 * xy + x_sq * y_sq;

    if denom.abs() < 1e-12 {
        return x.to_vec();
    }

    let mut result = vec![0.0f32; d];
    for i in 0..d {
        result[i] = (num_x * x[i] + num_y * y[i]) / denom;
    }

    // Safety projection
    poincare_project(&mut result, 0.999);
    result
}

// ---------------------------------------------------------------------------
// Engram Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Engram codebook.
#[derive(Debug, Clone)]
pub struct EngramConfig {
    /// Dimension of each codebook entry
    pub entry_dim: usize,
    /// Maximum number of entries in the codebook
    pub max_entries: usize,
    /// MDL threshold: only add entry if mean_delta > this value
    pub mdl_threshold: f32,
    /// Maximum norm in Poincare ball (entries must be inside this radius)
    pub max_poincare_norm: f32,
    /// Number of nearest entries to return on query
    pub top_k: usize,
    /// Minimum distance between entries (deduplication)
    pub min_distance: f32,
}

impl EngramConfig {
    pub fn chimere() -> Self {
        Self {
            entry_dim: 128, // matches DeltaNet head_dim
            max_entries: 4096,
            mdl_threshold: 0.1,
            max_poincare_norm: 0.999,
            top_k: 4,
            min_distance: 0.5,
        }
    }

    pub fn test() -> Self {
        Self {
            entry_dim: 8,
            max_entries: 32,
            mdl_threshold: 0.1,
            max_poincare_norm: 0.999,
            top_k: 3,
            min_distance: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Engram Entry
// ---------------------------------------------------------------------------

/// A single entry in the Engram codebook.
#[derive(Debug, Clone)]
pub struct EngramEntry {
    /// Position in the Poincare ball (||pos|| < 1)
    /// Near origin = general concept, near boundary = specific detail
    pub position: Vec<f32>,
    /// The value/content vector associated with this position
    pub value: Vec<f32>,
    /// Number of times this entry has been accessed (for LRU eviction)
    pub access_count: u32,
    /// Running mean of prediction error when this entry was last relevant
    pub mean_delta_at_write: f32,
}

// ---------------------------------------------------------------------------
// Engram Codebook
// ---------------------------------------------------------------------------

/// The Engram: a hierarchical memory codebook with hyperbolic addressing.
pub struct Engram {
    pub config: EngramConfig,
    /// Current entries in the codebook
    pub entries: Vec<EngramEntry>,
}

/// Result of querying the Engram.
#[derive(Debug, Clone)]
pub struct EngramQueryResult {
    /// Indices of nearest entries
    pub entry_indices: Vec<usize>,
    /// Hyperbolic distances to nearest entries
    pub distances: Vec<f32>,
    /// Values of nearest entries
    pub values: Vec<Vec<f32>>,
    /// Codebook utilization: num_entries / max_entries
    pub utilization: f32,
}

/// Metrics about the Engram state, for the MoE router.
#[derive(Debug, Clone)]
pub struct EngramMetrics {
    /// Fraction of codebook slots used
    pub utilization: f32,
    /// Mean depth (Poincare norm) of entries — high = many specific entries
    pub mean_depth: f32,
    /// Mean access count — high = entries are frequently reused
    pub mean_access_count: f32,
    /// Number of entries added recently (last N queries)
    pub recent_additions: usize,
}

impl Engram {
    pub fn new(config: EngramConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
        }
    }

    /// Query the codebook: find the top-k nearest entries to query_pos.
    pub fn query(&mut self, query_pos: &[f32]) -> EngramQueryResult {
        assert_eq!(query_pos.len(), self.config.entry_dim);

        if self.entries.is_empty() {
            return EngramQueryResult {
                entry_indices: vec![],
                distances: vec![],
                values: vec![],
                utilization: 0.0,
            };
        }

        // Compute distances to all entries
        let mut dists: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, poincare_distance(query_pos, &e.position)))
            .collect();

        // Sort by distance
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let k = self.config.top_k.min(dists.len());
        let entry_indices: Vec<usize> = dists[..k].iter().map(|&(i, _)| i).collect();
        let distances: Vec<f32> = dists[..k].iter().map(|&(_, d)| d).collect();
        let values: Vec<Vec<f32>> = entry_indices
            .iter()
            .map(|&i| self.entries[i].value.clone())
            .collect();

        // Update access counts
        for &i in &entry_indices {
            self.entries[i].access_count += 1;
        }

        let utilization = self.entries.len() as f32 / self.config.max_entries as f32;

        EngramQueryResult {
            entry_indices,
            distances,
            values,
            utilization,
        }
    }

    /// MDL-gated update: only add a new entry if the DeltaNet prediction
    /// error exceeds the threshold (the sequence contains new information
    /// not yet captured by the codebook).
    ///
    /// Returns true if an entry was added.
    pub fn maybe_update(
        &mut self,
        position: &[f32],
        value: &[f32],
        state_metrics: &StateMetrics,
    ) -> bool {
        assert_eq!(position.len(), self.config.entry_dim);
        assert_eq!(value.len(), self.config.entry_dim);

        // MDL gate: only store if prediction error is high enough
        if state_metrics.mean_delta < self.config.mdl_threshold {
            return false;
        }

        // Check for duplicates: if a nearby entry exists, skip
        for entry in &self.entries {
            let dist = poincare_distance(position, &entry.position);
            if dist < self.config.min_distance {
                return false;
            }
        }

        // Project position into the Poincare ball
        let mut pos = position.to_vec();
        poincare_project(&mut pos, self.config.max_poincare_norm);

        let entry = EngramEntry {
            position: pos,
            value: value.to_vec(),
            access_count: 0,
            mean_delta_at_write: state_metrics.mean_delta,
        };

        // Eviction: if at capacity, remove the least accessed entry
        if self.entries.len() >= self.config.max_entries {
            let min_idx = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(i, _)| i)
                .unwrap();
            self.entries.swap_remove(min_idx);
        }

        self.entries.push(entry);
        true
    }

    /// Get metrics about the codebook state.
    pub fn metrics(&self) -> EngramMetrics {
        if self.entries.is_empty() {
            return EngramMetrics {
                utilization: 0.0,
                mean_depth: 0.0,
                mean_access_count: 0.0,
                recent_additions: 0,
            };
        }

        let n = self.entries.len() as f32;
        let mean_depth: f32 = self
            .entries
            .iter()
            .map(|e| poincare_depth(&e.position))
            .sum::<f32>()
            / n;

        let mean_access: f32 = self
            .entries
            .iter()
            .map(|e| e.access_count as f32)
            .sum::<f32>()
            / n;

        EngramMetrics {
            utilization: self.entries.len() as f32 / self.config.max_entries as f32,
            mean_depth,
            mean_access_count: mean_access,
            recent_additions: self
                .entries
                .iter()
                .filter(|e| e.access_count == 0)
                .count(),
        }
    }

    /// Number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Poincare distance tests ---

    #[test]
    fn test_poincare_distance_origin() {
        // Distance from origin to a point
        let origin = vec![0.0; 4];
        let point = vec![0.5, 0.0, 0.0, 0.0];
        let d = poincare_distance(&origin, &point);

        assert!(d > 0.0, "Distance should be positive, got {:.4}", d);
        assert!(d < 5.0, "Distance should be finite, got {:.4}", d);

        println!("d(origin, [0.5,0,0,0]) = {:.4}", d);
    }

    #[test]
    fn test_poincare_distance_symmetry() {
        let a = vec![0.3, 0.1, 0.0, 0.0];
        let b = vec![0.1, 0.4, 0.0, 0.0];

        let d_ab = poincare_distance(&a, &b);
        let d_ba = poincare_distance(&b, &a);

        assert!(
            (d_ab - d_ba).abs() < 1e-6,
            "Distance should be symmetric: {:.6} vs {:.6}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_poincare_hierarchy() {
        // KEY TEST: Hyperbolic distance encodes hierarchy.
        // Points near the boundary are "deeper" in the hierarchy.
        // Moving the same Euclidean distance near the boundary costs MORE
        // in hyperbolic distance than near the origin.

        // "animal" near origin
        let animal = vec![0.1, 0.0, 0.0, 0.0];
        // "mammal" further out
        let mammal = vec![0.5, 0.0, 0.0, 0.0];
        // "golden_retriever" near boundary
        let retriever = vec![0.9, 0.0, 0.0, 0.0];

        let d_animal_mammal = poincare_distance(&animal, &mammal);
        let d_mammal_retriever = poincare_distance(&mammal, &retriever);

        println!(
            "Hierarchy: d(animal, mammal)={:.4}, d(mammal, retriever)={:.4}",
            d_animal_mammal, d_mammal_retriever
        );

        // The Euclidean distance is the same (0.4) but hyperbolic distance
        // should be LARGER near the boundary
        assert!(
            d_mammal_retriever > d_animal_mammal,
            "Boundary distance should be larger: {:.4} > {:.4}",
            d_mammal_retriever,
            d_animal_mammal
        );
    }

    #[test]
    fn test_poincare_depth_ordering() {
        let general = vec![0.1, 0.0, 0.0, 0.0]; // near center
        let specific = vec![0.8, 0.0, 0.0, 0.0]; // near boundary

        let d_general = poincare_depth(&general);
        let d_specific = poincare_depth(&specific);

        assert!(
            d_specific > d_general,
            "Specific concepts should have higher depth: {:.4} > {:.4}",
            d_specific,
            d_general
        );

        println!(
            "Depth: general={:.4}, specific={:.4}",
            d_general, d_specific
        );
    }

    // --- Mobius addition test ---

    #[test]
    fn test_mobius_add_stays_in_ball() {
        let a = vec![0.3, 0.4, 0.0, 0.0];
        let b = vec![0.2, -0.3, 0.0, 0.0];

        let result = mobius_add(&a, &b);
        let norm: f32 = result.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        assert!(
            norm < 1.0,
            "Mobius addition should stay in ball: norm={:.4}",
            norm
        );

        println!("Mobius add: {:?}, norm={:.4}", result, norm);
    }

    // --- Engram codebook tests ---

    #[test]
    fn test_engram_mdl_gate() {
        // The MDL gate should prevent storing entries when prediction error is low
        let config = EngramConfig::test();
        let mut engram = Engram::new(config.clone());

        let pos = vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let val = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Low prediction error → should NOT store
        let low_delta = StateMetrics {
            frobenius_norm: 0.5,
            mean_delta: 0.01, // below threshold (0.1)
            effective_rank: 0.5,
        };
        let stored = engram.maybe_update(&pos, &val, &low_delta);
        assert!(!stored, "Should NOT store when mean_delta < threshold");
        assert_eq!(engram.len(), 0);

        // High prediction error → SHOULD store
        let high_delta = StateMetrics {
            frobenius_norm: 0.5,
            mean_delta: 0.5, // above threshold
            effective_rank: 0.5,
        };
        let stored = engram.maybe_update(&pos, &val, &high_delta);
        assert!(stored, "Should store when mean_delta > threshold");
        assert_eq!(engram.len(), 1);

        println!(
            "MDL gate: low_delta→skipped, high_delta→stored (entries={})",
            engram.len()
        );
    }

    #[test]
    fn test_engram_deduplication() {
        // Nearby entries should be deduplicated
        let config = EngramConfig::test();
        let mut engram = Engram::new(config);

        let high_delta = StateMetrics {
            frobenius_norm: 0.5,
            mean_delta: 0.5,
            effective_rank: 0.5,
        };

        // Store first entry
        let pos1 = vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let val1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(engram.maybe_update(&pos1, &val1, &high_delta));

        // Try to store very close entry → should be deduplicated
        let pos2 = vec![0.11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let val2 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stored = engram.maybe_update(&pos2, &val2, &high_delta);
        assert!(!stored, "Nearby entry should be deduplicated");

        // Store distant entry → should succeed
        let pos3 = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8];
        let val3 = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let stored = engram.maybe_update(&pos3, &val3, &high_delta);
        assert!(stored, "Distant entry should be stored");

        assert_eq!(engram.len(), 2);
        println!("Deduplication: 3 attempts → {} entries", engram.len());
    }

    #[test]
    fn test_engram_query_nearest() {
        let config = EngramConfig::test();
        let mut engram = Engram::new(config);

        let high_delta = StateMetrics {
            frobenius_norm: 0.5,
            mean_delta: 0.5,
            effective_rank: 0.5,
        };

        // Add entries at different positions
        let positions = vec![
            vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // near origin
            vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // middle
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], // far, different direction
        ];

        for (i, pos) in positions.iter().enumerate() {
            let mut val = vec![0.0; 8];
            val[i] = 1.0;
            engram.maybe_update(pos, &val, &high_delta);
        }

        assert_eq!(engram.len(), 3);

        // Query near the first entry → should return it as nearest
        let query = vec![0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = engram.query(&query);

        assert!(!result.entry_indices.is_empty());
        assert_eq!(
            result.entry_indices[0], 0,
            "Nearest to [0.12,...] should be entry 0 at [0.1,...], got entry {}",
            result.entry_indices[0]
        );

        println!(
            "Query result: nearest={}, distances={:?}, utilization={:.2}",
            result.entry_indices[0], result.distances, result.utilization
        );
    }

    #[test]
    fn test_engram_compression() {
        // KEY TEST: A repetitive sequence should create FEW entries
        // (MDL compression), while a novel sequence creates many.
        let config = EngramConfig {
            mdl_threshold: 0.05, // lower threshold to allow some storage
            min_distance: 0.2,   // moderate dedup distance
            ..EngramConfig::test()
        };

        // Repetitive sequence: same 3 patterns cycling
        let mut engram_repetitive = Engram::new(config.clone());
        let patterns = vec![
            (
                vec![0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                vec![0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            (
                vec![0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ];

        // First pass: high delta (novel) → entries get created
        for (pos, val) in &patterns {
            let metrics = StateMetrics {
                frobenius_norm: 0.5,
                mean_delta: 0.3, // high → novel
                effective_rank: 0.5,
            };
            engram_repetitive.maybe_update(pos, val, &metrics);
        }
        let entries_after_first = engram_repetitive.len();

        // Subsequent passes: low delta (repetitive) → no new entries
        for _ in 0..5 {
            for (pos, val) in &patterns {
                let metrics = StateMetrics {
                    frobenius_norm: 0.5,
                    mean_delta: 0.02, // low → repetitive, below threshold
                    effective_rank: 0.5,
                };
                engram_repetitive.maybe_update(pos, val, &metrics);
            }
        }
        let entries_after_repeat = engram_repetitive.len();

        // Novel sequence: all different patterns, all high delta
        let mut engram_novel = Engram::new(config.clone());
        for i in 0..10 {
            let angle = (i as f32) * 0.6; // spread around the ball
            let mut pos = vec![0.0; 8];
            pos[0] = 0.3 * angle.cos();
            pos[1] = 0.3 * angle.sin();
            let mut val = vec![0.0; 8];
            val[i % 8] = 1.0;

            let metrics = StateMetrics {
                frobenius_norm: 0.5,
                mean_delta: 0.4, // always high → always novel
                effective_rank: 0.5,
            };
            engram_novel.maybe_update(&pos, &val, &metrics);
        }
        let entries_novel = engram_novel.len();

        println!(
            "Compression: repetitive={} (first pass: {}, after 5 repeats: same), novel={}",
            entries_after_repeat, entries_after_first, entries_novel
        );

        // Repetitive should have fewer entries than novel
        assert_eq!(
            entries_after_first, entries_after_repeat,
            "Repetitive sequence should NOT grow: {} -> {}",
            entries_after_first, entries_after_repeat
        );
        assert!(
            entries_novel > entries_after_repeat,
            "Novel sequence ({}) should have more entries than repetitive ({})",
            entries_novel,
            entries_after_repeat
        );
    }

    #[test]
    fn test_engram_eviction() {
        // When at capacity, LRU eviction should remove least-accessed entries.
        // Use top_k=1 to ensure only the nearest entry gets access_count bumped.
        let config = EngramConfig {
            max_entries: 3,
            mdl_threshold: 0.05,
            min_distance: 0.01,
            top_k: 1, // only bump nearest entry's access count
            ..EngramConfig::test()
        };
        let mut engram = Engram::new(config);

        let high_delta = StateMetrics {
            frobenius_norm: 0.5,
            mean_delta: 0.5,
            effective_rank: 0.5,
        };

        // Fill codebook with 3 entries at well-separated positions
        let positions = [
            vec![0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // entry A
            vec![0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // entry B
            vec![0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], // entry C (will be evicted)
        ];
        for (i, pos) in positions.iter().enumerate() {
            let mut val = vec![0.0; 8];
            val[i] = 1.0;
            engram.maybe_update(pos, &val, &high_delta);
        }
        assert_eq!(engram.len(), 3);

        // Access entries A and B (but not C) — C becomes least accessed
        let q_a = vec![0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let q_b = vec![0.0, 0.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        engram.query(&q_a);
        engram.query(&q_b);

        // Verify access counts: A=1, B=1, C=0
        let access_counts: Vec<u32> = engram.entries.iter().map(|e| e.access_count).collect();
        println!("Access counts before eviction: {:?}", access_counts);
        assert_eq!(access_counts, vec![1, 1, 0], "Only nearest should be bumped");

        // Add a 4th entry → should evict C (least accessed, count=0)
        let pos_d = vec![0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0];
        let val_d = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        engram.maybe_update(&pos_d, &val_d, &high_delta);
        assert_eq!(engram.len(), 3, "Should still be at capacity");

        // Verify: C evicted, A and B retained, D added
        let has_a = engram.entries.iter().any(|e| e.position[0] > 0.2);
        let has_b = engram.entries.iter().any(|e| e.position[1] > 0.2);
        let has_c = engram.entries.iter().any(|e| e.position[2] > 0.2);
        let has_d = engram.entries.iter().any(|e| e.position[3] > 0.2);

        assert!(has_a, "Entry A (accessed) should still exist");
        assert!(has_b, "Entry B (accessed) should still exist");
        assert!(!has_c, "Entry C (not accessed) should have been evicted");
        assert!(has_d, "New entry D should exist");

        println!(
            "Eviction: C evicted, A/B retained, D added. Entries: {}",
            engram.len()
        );
    }

    #[test]
    fn test_engram_metrics() {
        let config = EngramConfig::test();
        let mut engram = Engram::new(config);

        // Empty codebook
        let m0 = engram.metrics();
        assert_eq!(m0.utilization, 0.0);
        assert_eq!(m0.mean_depth, 0.0);

        let high_delta = StateMetrics {
            frobenius_norm: 0.5,
            mean_delta: 0.5,
            effective_rank: 0.5,
        };

        // Add some entries at different depths
        let positions = vec![
            vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // shallow
            vec![0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // medium
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8], // deep
        ];

        for (i, pos) in positions.iter().enumerate() {
            let mut val = vec![0.0; 8];
            val[i] = 1.0;
            engram.maybe_update(pos, &val, &high_delta);
        }

        let m1 = engram.metrics();
        assert!(m1.utilization > 0.0, "Should have non-zero utilization");
        assert!(m1.mean_depth > 0.0, "Should have non-zero mean depth");
        assert_eq!(m1.recent_additions, 3, "All entries are new (0 access)");

        println!(
            "Engram metrics: util={:.3}, depth={:.3}, access={:.1}, new={}",
            m1.utilization, m1.mean_depth, m1.mean_access_count, m1.recent_additions
        );
    }
}
