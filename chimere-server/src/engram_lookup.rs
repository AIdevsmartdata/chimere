//! # Engram Lookup — Fast N-gram Hash Table
//!
//! A runtime lookup table for n-grams (sequences of token IDs) that provides
//! next-token predictions in O(1) average time. The table is backed by a
//! memory-mapped file for zero-copy access.
//!
//! ## Purpose
//!
//! This module boosts model quality by injecting pre-computed next-token
//! statistics into the generation pipeline without additional model parameters.
//! During generation, `lookup` retrieves likely next tokens for the recent
//! context window, and `bias_logits` blends these predictions into the model's
//! own logit distribution.
//!
//! ## File Format
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │ Header (20 bytes)                               │
//! │   magic:      u32  = 0x454E4752  ("ENGR")       │
//! │   version:    u32  = 1                          │
//! │   order:      u32  (n-gram order)               │
//! │   table_size: u32  (number of hash-table slots) │
//! │   num_entries:u32  (occupied slots)             │
//! ├─────────────────────────────────────────────────┤
//! │ Hash Table  [table_size × 16 bytes]             │
//! │   hash:   u64  (FNV-1a of token sequence)       │
//! │   offset: u32  (byte offset into data section)  │
//! │   count:  u32  (total frequency — sum of all    │
//! │                 next-token frequencies at slot)  │
//! ├─────────────────────────────────────────────────┤
//! │ Data Section  [variable]                        │
//! │   per slot: num_nexts: u32                      │
//! │             [num_nexts × (token: u32, freq: u32)]│
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! An empty (tombstone) slot has hash == 0 and offset == 0.
//! The data section offset stored in each slot is relative to the start
//! of the data section itself (i.e. after header + hash table).

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// File magic: ASCII "ENGR"
const MAGIC: u32 = 0x454E_4752;
const VERSION: u32 = 1;

/// Header size in bytes.
const HEADER_SIZE: usize = 20;

/// Hash-table slot size in bytes: u64 hash + u32 offset + u32 count.
const SLOT_SIZE: usize = 16;

/// A sentinel hash value meaning "empty slot".
/// FNV-1a of an empty sequence is the offset basis itself, which is non-zero,
/// so 0 is safe as a sentinel.
const EMPTY_HASH: u64 = 0;

// ---------------------------------------------------------------------------
// CuckooFilter — Tier 0 probabilistic membership test
// ---------------------------------------------------------------------------

/// Cuckoo filter: ~1 byte per entry, ~3% false positive rate, zero false negatives.
///
/// Before doing the expensive linear-probe hash lookup in `EngramLookup::lookup()`,
/// we check this filter first. If the filter says "absent", the n-gram hash is
/// definitely not in the table, and we skip the lookup entirely.
///
/// For a 50M-entry engram table this uses ~50 MB of RAM (1 byte fingerprint per
/// bucket entry, 4 entries per bucket). Novel n-grams (~97% of lookups at
/// inference time) are rejected in O(1) with two cache-line reads.
struct CuckooFilter {
    /// Flat array of 8-bit fingerprints. Layout: `num_buckets` buckets,
    /// each containing `CUCKOO_BUCKET_SIZE` fingerprint slots.
    buckets: Vec<u8>,
    /// Number of buckets (must be a power of two).
    num_buckets: usize,
}

/// Entries per bucket. 4 gives ~95% utilisation and ~3% FPR with 8-bit fps.
const CUCKOO_BUCKET_SIZE: usize = 4;

/// Maximum kicks before giving up on insertion (filter is effectively full).
const CUCKOO_MAX_KICKS: usize = 500;

/// Empty fingerprint sentinel. A real fingerprint is always non-zero.
const CUCKOO_EMPTY: u8 = 0;

impl CuckooFilter {
    /// Derive an 8-bit fingerprint from a 64-bit hash. Never returns 0 (sentinel).
    #[inline(always)]
    fn fingerprint(hash: u64) -> u8 {
        // Use the upper byte, then force non-zero.
        let fp = (hash >> 56) as u8;
        if fp == 0 { 1 } else { fp }
    }

    /// Compute two candidate bucket indices for a given hash.
    #[inline(always)]
    fn bucket_indices(&self, hash: u64) -> (usize, usize) {
        let i1 = (hash as usize) & (self.num_buckets - 1);
        let fp = Self::fingerprint(hash) as u64;
        // XOR-based alternate index (standard cuckoo filter trick).
        // Hash the fingerprint to spread it across the bucket space.
        let i2 = (i1 ^ (fp.wrapping_mul(0x5bd1_e995) as usize)) & (self.num_buckets - 1);
        (i1, i2)
    }

    /// Build a cuckoo filter from an iterator of 64-bit hashes.
    ///
    /// Returns `None` if the filter cannot accommodate all entries (too full
    /// or too many collisions). Callers should fall back to no-filter mode.
    fn build(hashes: impl Iterator<Item = u64>, count_hint: usize) -> Option<Self> {
        // Size the filter: enough buckets so load factor is ~50%.
        let min_buckets = ((count_hint + CUCKOO_BUCKET_SIZE - 1) / CUCKOO_BUCKET_SIZE) * 2;
        let num_buckets = min_buckets.next_power_of_two().max(16);
        let buckets = vec![CUCKOO_EMPTY; num_buckets * CUCKOO_BUCKET_SIZE];

        let mut filter = CuckooFilter { buckets, num_buckets };
        for hash in hashes {
            if !filter.insert(hash) {
                // Insertion failed — filter too full, fall back gracefully.
                return None;
            }
        }
        Some(filter)
    }

    /// Insert a hash into the filter. Returns false if the filter is full.
    fn insert(&mut self, hash: u64) -> bool {
        let fp = Self::fingerprint(hash);
        let (i1, i2) = self.bucket_indices(hash);

        // Try bucket i1
        let base1 = i1 * CUCKOO_BUCKET_SIZE;
        for j in 0..CUCKOO_BUCKET_SIZE {
            if self.buckets[base1 + j] == CUCKOO_EMPTY {
                self.buckets[base1 + j] = fp;
                return true;
            }
        }

        // Try bucket i2
        let base2 = i2 * CUCKOO_BUCKET_SIZE;
        for j in 0..CUCKOO_BUCKET_SIZE {
            if self.buckets[base2 + j] == CUCKOO_EMPTY {
                self.buckets[base2 + j] = fp;
                return true;
            }
        }

        // Both full — kick an existing entry (cuckoo displacement).
        let mut cur_fp = fp;
        let mut cur_i = i1;
        for _ in 0..CUCKOO_MAX_KICKS {
            let base = cur_i * CUCKOO_BUCKET_SIZE;
            // Evict slot 0 (deterministic, simple).
            let evicted = self.buckets[base];
            self.buckets[base] = cur_fp;
            cur_fp = evicted;
            // Alternate bucket for the evicted fingerprint.
            cur_i = (cur_i ^ (cur_fp as usize).wrapping_mul(0x5bd1_e995)) & (self.num_buckets - 1);
            let base2 = cur_i * CUCKOO_BUCKET_SIZE;
            for j in 0..CUCKOO_BUCKET_SIZE {
                if self.buckets[base2 + j] == CUCKOO_EMPTY {
                    self.buckets[base2 + j] = cur_fp;
                    return true;
                }
            }
        }
        false // too many kicks
    }

    /// Check if a hash MAY be in the filter (probabilistic: false positives possible).
    #[inline]
    fn contains(&self, hash: u64) -> bool {
        let fp = Self::fingerprint(hash);
        let (i1, i2) = self.bucket_indices(hash);

        let base1 = i1 * CUCKOO_BUCKET_SIZE;
        for j in 0..CUCKOO_BUCKET_SIZE {
            if self.buckets[base1 + j] == fp {
                return true;
            }
        }

        let base2 = i2 * CUCKOO_BUCKET_SIZE;
        for j in 0..CUCKOO_BUCKET_SIZE {
            if self.buckets[base2 + j] == fp {
                return true;
            }
        }

        false
    }

    /// Memory usage in bytes.
    fn memory_bytes(&self) -> usize {
        self.buckets.len()
    }
}

// ---------------------------------------------------------------------------
// EngramLookup
// ---------------------------------------------------------------------------

/// Fast n-gram lookup table stored as a memory-mapped file.
///
/// Given a sequence of `order` token IDs, returns the most likely next
/// token(s) with their probabilities.
///
/// # Performance
/// - `lookup` is O(1) average with linear probing (open addressing).
/// - Zero allocations in the hot path: the returned `Vec` is the only
///   allocation and is bounded by the number of distinct next-tokens stored
///   for that n-gram (typically small, e.g. ≤ 16).
pub struct EngramLookup {
    /// Memory-mapped file backing the hash table and data section.
    data: Mmap,
    /// Number of slots in the hash table.
    table_size: usize,
    /// N-gram order: how many preceding tokens form the context key.
    order: usize,
    /// Tier-0 Cuckoo filter for fast negative lookups.
    /// If `Some`, we check the filter BEFORE probing the hash table.
    /// If the filter says "absent", the n-gram is definitely not in the table.
    /// `None` if the filter could not be built (graceful fallback).
    cuckoo: Option<CuckooFilter>,
}

// ---------------------------------------------------------------------------
// Raw slot helpers (no allocation)
// ---------------------------------------------------------------------------

/// Read a little-endian u32 from a byte slice at `offset`.
#[inline(always)]
fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let b = &bytes[offset..offset + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian u64 from a byte slice at `offset`.
#[inline(always)]
fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    let b = &bytes[offset..offset + 8];
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

// ---------------------------------------------------------------------------
// Hash function
// ---------------------------------------------------------------------------

impl EngramLookup {
    /// FNV-1a hash over a sequence of token IDs.
    ///
    /// Processes each token as 4 little-endian bytes.  The FNV offset basis
    /// (`0xcbf2_9ce4_8422_2325`) is non-zero, so an empty slice hashes to a
    /// non-zero value, keeping 0 available as a safe sentinel for empty slots.
    #[inline]
    pub fn hash_ngram(tokens: &[u32]) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        for &tok in tokens {
            let bytes = tok.to_le_bytes();
            for &b in &bytes {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x0000_0001_0000_01b3); // FNV prime
            }
        }
        hash
    }
}

// ---------------------------------------------------------------------------
// Byte offset arithmetic
// ---------------------------------------------------------------------------

impl EngramLookup {
    /// Byte offset of slot `i` within the memory-mapped file.
    #[inline(always)]
    fn slot_offset(&self, i: usize) -> usize {
        HEADER_SIZE + i * SLOT_SIZE
    }

    /// Byte offset of the start of the data section.
    #[inline(always)]
    fn data_section_start(&self) -> usize {
        HEADER_SIZE + self.table_size * SLOT_SIZE
    }

    /// Read the (hash, data_offset, total_count) triple for slot `i`.
    #[inline(always)]
    fn read_slot(&self, i: usize) -> (u64, u32, u32) {
        let base = self.slot_offset(i);
        let hash = read_u64(&self.data, base);
        let offset = read_u32(&self.data, base + 8);
        let count = read_u32(&self.data, base + 12);
        (hash, offset, count)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

impl EngramLookup {
    /// Load an engram file via memory-mapping.
    ///
    /// # Errors
    /// Returns a `String` describing the error (file not found, bad magic,
    /// version mismatch, truncated file, etc.).
    /// Load from CHIMERE_ENGRAM_FILE env var. Returns None if not set.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("CHIMERE_ENGRAM_FILE").ok()?;
        match Self::from_file(&path) {
            Ok(e) => {
                eprintln!("[ENGRAM] Loaded table from {} (order={})", path, e.order());
                Some(e)
            }
            Err(e) => {
                eprintln!("[ENGRAM] Failed to load {}: {}", path, e);
                None
            }
        }
    }

    pub fn from_file(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {path}: {e}"))?;

        // Safety: the file is read-only and we never write through the map.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap failed for {path}: {e}"))?;

        if mmap.len() < HEADER_SIZE {
            return Err(format!(
                "File {path} too small ({} bytes), expected at least {HEADER_SIZE}",
                mmap.len()
            ));
        }

        let magic = read_u32(&mmap, 0);
        if magic != MAGIC {
            return Err(format!(
                "Bad magic in {path}: got 0x{magic:08X}, expected 0x{MAGIC:08X}"
            ));
        }

        let version = read_u32(&mmap, 4);
        if version != VERSION {
            return Err(format!(
                "Unsupported version in {path}: {version} (expected {VERSION})"
            ));
        }

        let order = read_u32(&mmap, 8) as usize;
        let table_size = read_u32(&mmap, 12) as usize;
        // num_entries at offset 16 — parsed but not stored (only used in build/verify)

        let min_size = HEADER_SIZE + table_size * SLOT_SIZE;
        if mmap.len() < min_size {
            return Err(format!(
                "File {path} truncated: {bytes} bytes, need at least {min_size}",
                bytes = mmap.len()
            ));
        }

        // Build Tier-0 Cuckoo filter from all occupied hash-table slots.
        let num_entries = read_u32(&mmap, 16) as usize;
        let cuckoo = {
            // Iterator over non-empty hash values in the mmap'd hash table.
            let hash_iter = (0..table_size).filter_map(|i| {
                let base = HEADER_SIZE + i * SLOT_SIZE;
                let h = read_u64(&mmap, base);
                if h != EMPTY_HASH { Some(h) } else { None }
            });
            match CuckooFilter::build(hash_iter, num_entries) {
                Some(cf) => {
                    eprintln!(
                        "[ENGRAM] Cuckoo filter built: {} entries, {:.1} MB",
                        num_entries,
                        cf.memory_bytes() as f64 / (1024.0 * 1024.0)
                    );
                    Some(cf)
                }
                None => {
                    eprintln!("[ENGRAM] Cuckoo filter build failed, falling back to direct lookup");
                    None
                }
            }
        };

        Ok(Self {
            data: mmap,
            table_size,
            order,
            cuckoo,
        })
    }

    /// Return the n-gram order of this table.
    #[inline(always)]
    pub fn order(&self) -> usize {
        self.order
    }

    /// Look up next-token predictions for a context.
    ///
    /// Uses the last `self.order` tokens of `context` as the key.
    /// Returns `Vec<(token_id, probability)>` sorted by descending probability,
    /// or an empty `Vec` if the n-gram is not found or `context` is too short.
    ///
    /// # Hot-path allocations
    /// Only the returned `Vec` is allocated.  All reads are through the
    /// memory-mapped file (zero copy).
    pub fn lookup(&self, context: &[u32]) -> Vec<(u32, f32)> {
        if context.len() < self.order {
            return Vec::new();
        }

        let key = &context[context.len() - self.order..];
        let hash = Self::hash_ngram(key);

        // Tier 0: Cuckoo filter fast reject — skip ~97% of novel n-grams.
        if let Some(ref cf) = self.cuckoo {
            if !cf.contains(hash) {
                return Vec::new();
            }
        }

        // Linear probe
        let start = (hash as usize) % self.table_size;
        let mut probe = start;
        loop {
            let (slot_hash, data_offset, _total_count) = self.read_slot(probe);

            if slot_hash == EMPTY_HASH {
                // Empty slot → n-gram not in table
                return Vec::new();
            }

            if slot_hash == hash {
                // Found candidate slot — read the data entries
                return self.read_predictions(data_offset);
            }

            probe = (probe + 1) % self.table_size;
            if probe == start {
                // Full wrap-around without finding the key
                return Vec::new();
            }
        }
    }

    /// Read and normalise next-token predictions at a data-section offset.
    ///
    /// Data layout at offset: `num_nexts: u32` followed by
    /// `num_nexts` pairs of `(token: u32, frequency: u32)`.
    fn read_predictions(&self, data_offset: u32) -> Vec<(u32, f32)> {
        let base = self.data_section_start() + data_offset as usize;
        let bytes = &self.data;

        if base + 4 > bytes.len() {
            return Vec::new();
        }

        let num_nexts = read_u32(bytes, base) as usize;
        if num_nexts == 0 {
            return Vec::new();
        }

        let needed = base + 4 + num_nexts * 8;
        if needed > bytes.len() {
            return Vec::new();
        }

        let mut total_freq: u64 = 0;
        let mut entries: Vec<(u32, u32)> = Vec::with_capacity(num_nexts);

        for i in 0..num_nexts {
            let off = base + 4 + i * 8;
            let token = read_u32(bytes, off);
            let freq = read_u32(bytes, off + 4);
            total_freq += freq as u64;
            entries.push((token, freq));
        }

        if total_freq == 0 {
            return Vec::new();
        }

        let total_f = total_freq as f32;
        let mut result: Vec<(u32, f32)> = entries
            .into_iter()
            .map(|(tok, freq)| (tok, freq as f32 / total_f))
            .collect();

        // Sort by descending probability
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Bias a model's raw logit array using engram next-token predictions.
    ///
    /// Adds `alpha * log(p_engram)` to each logit where the engram has a
    /// non-zero probability.  `alpha = 0` → no effect; `alpha = 1` → full
    /// log-prob blending.
    ///
    /// # Arguments
    /// - `logits`: mutable slice of raw (pre-softmax) logit scores, indexed by token id.
    /// - `predictions`: output of `lookup`, sorted by probability.
    /// - `alpha`: blending strength in [0, 1].
    /// Get draft tokens from Engram n-gram lookup for speculative decoding.
    ///
    /// Given the recent token context, returns up to `max_drafts` likely
    /// continuation tokens sorted by probability. These can be passed to
    /// the model for batch verification (DART-style speculative decoding).
    ///
    /// Returns empty vec if no n-gram match is found (model generates normally).
    /// Cost: O(1) hash lookup + O(max_drafts) sort — typically < 0.01ms.
    pub fn draft_tokens(&self, context: &[u32], max_drafts: usize) -> Vec<u32> {
        let predictions = self.lookup(context);
        if predictions.is_empty() {
            return Vec::new();
        }
        // Already sorted by probability (highest first) from read_predictions
        predictions.iter()
            .take(max_drafts)
            .map(|(tok, _prob)| *tok)
            .collect()
    }

    /// Multi-step drafting: given context, predict a sequence of N tokens
    /// by chaining n-gram lookups.
    ///
    /// Each step appends the top-1 prediction to the context and looks up again.
    /// Returns the draft sequence (may be shorter than `steps` if a lookup misses).
    ///
    /// This is the DART (arXiv 2601.19278) approach: n-gram trie as FREE drafter.
    /// Cost: O(steps) hash lookups — typically < 0.05ms for steps=5.
    pub fn draft_sequence(&self, context: &[u32], steps: usize) -> Vec<u32> {
        let mut ctx = context.to_vec();
        let mut drafts = Vec::with_capacity(steps);

        for _ in 0..steps {
            let predictions = self.lookup(&ctx);
            if predictions.is_empty() {
                break;
            }
            let top_token = predictions[0].0;
            drafts.push(top_token);
            ctx.push(top_token);
        }

        drafts
    }

    pub fn bias_logits(logits: &mut [f32], predictions: &[(u32, f32)], alpha: f32) {
        // Additive log-probability blending:
        // logits[token_id] += alpha * ln(p_engram)
        //
        // This shifts the model's raw logits toward tokens that the N-gram
        // table predicts are likely.  Because logits are pre-softmax log-unnormalised
        // scores, adding log(p) is equivalent to multiplying the unnormalised
        // probability by p^alpha — a standard interpolation in log-space.
        //
        // alpha = 0.0 → no effect (predictions ignored)
        // alpha = 1.0 → full log-prob blend (N-gram equally weighted with model)
        // alpha > 1.0 → N-gram dominates (use with care)
        for &(token_id, prob) in predictions {
            if (token_id as usize) < logits.len() && prob > 0.0 {
                logits[token_id as usize] += alpha * prob.ln();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MultiEngramLookup — load all .engr tables from a directory
// ---------------------------------------------------------------------------

/// A collection of `EngramLookup` tables that are queried in parallel.
///
/// At lookup time, predictions from **all** tables are merged: token
/// frequencies are summed before normalisation.  This gives each domain
/// proportional influence based on how much evidence it has for the
/// n-gram in question.
///
/// # Loading
///
/// `MultiEngramLookup::from_env()` inspects two env vars (in order):
///
/// 1. `CHIMERE_ENGRAM_DIR`  — loads every `*.engr` file in the directory.
/// 2. `CHIMERE_ENGRAM_FILE` — single file fallback (backward compat).
///
/// Returns `None` if neither is set or no tables could be loaded.
pub struct MultiEngramLookup {
    tables: Vec<(String, EngramLookup)>, // (filename, table)
}

impl MultiEngramLookup {
    /// Load from environment variables.
    pub fn from_env() -> Option<Self> {
        // Priority 1: directory of .engr files
        if let Ok(dir) = std::env::var("CHIMERE_ENGRAM_DIR") {
            let loaded = Self::from_dir(&dir);
            if !loaded.tables.is_empty() {
                return Some(loaded);
            }
            eprintln!("[ENGRAM] CHIMERE_ENGRAM_DIR={dir} but no .engr files found");
        }

        // Priority 2: single file (backward compat)
        if let Ok(path) = std::env::var("CHIMERE_ENGRAM_FILE") {
            match EngramLookup::from_file(&path) {
                Ok(e) => {
                    let name = std::path::Path::new(&path)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.clone());
                    eprintln!(
                        "[ENGRAM] Loaded single table '{}' (order={})",
                        name,
                        e.order()
                    );
                    return Some(Self {
                        tables: vec![(name, e)],
                    });
                }
                Err(e) => {
                    eprintln!("[ENGRAM] Failed to load {path}: {e}");
                }
            }
        }

        None
    }

    /// Load all `.engr` files from a directory.
    pub fn from_dir(dir: &str) -> Self {
        let mut tables = Vec::new();
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("[ENGRAM] Cannot read dir {dir}: {e}");
                return Self { tables };
            }
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "engr").unwrap_or(false) {
                let path_str = path.to_string_lossy().into_owned();
                match EngramLookup::from_file(&path_str) {
                    Ok(e) => {
                        let name = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        eprintln!(
                            "[ENGRAM] Loaded '{}' — order={}, size={}",
                            name,
                            e.order(),
                            path.metadata().map(|m| m.len()).unwrap_or(0)
                        );
                        tables.push((name, e));
                    }
                    Err(e) => {
                        eprintln!("[ENGRAM] Skipping {}: {}", path_str, e);
                    }
                }
            }
        }

        // Sort by name for deterministic ordering
        tables.sort_by(|a, b| a.0.cmp(&b.0));
        eprintln!("[ENGRAM] Loaded {} tables from {dir}", tables.len());
        Self { tables }
    }

    /// Wrap a single `EngramLookup` into a `MultiEngramLookup`.
    pub fn from_single(name: String, table: EngramLookup) -> Self {
        Self {
            tables: vec![(name, table)],
        }
    }

    /// Number of loaded tables.
    pub fn len(&self) -> usize {
        self.tables.len()
    }

    /// True if no tables are loaded.
    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    /// N-gram order of the first table (all tables should have the same order).
    pub fn order(&self) -> usize {
        self.tables.first().map(|(_, t)| t.order()).unwrap_or(0)
    }

    /// Query all tables and merge predictions.
    ///
    /// Frequencies from each table are summed per token, then normalised.
    /// Tables with more evidence for an n-gram naturally dominate.
    /// Returns `Vec<(token_id, probability)>` sorted descending.
    pub fn lookup(&self, context: &[u32]) -> Vec<(u32, f32)> {
        if self.tables.len() == 1 {
            // Fast path: single table, no merge overhead
            return self.tables[0].1.lookup(context);
        }

        // Merge: sum raw predictions across all tables
        let mut merged: HashMap<u32, f32> = HashMap::new();
        let mut any_hit = false;

        for (_, table) in &self.tables {
            let preds = table.lookup(context);
            if !preds.is_empty() {
                any_hit = true;
                for (tok, prob) in preds {
                    *merged.entry(tok).or_insert(0.0) += prob;
                }
            }
        }

        if !any_hit {
            return Vec::new();
        }

        // Re-normalise
        let total: f32 = merged.values().sum();
        if total <= 0.0 {
            return Vec::new();
        }

        let mut result: Vec<(u32, f32)> = merged
            .into_iter()
            .map(|(tok, freq)| (tok, freq / total))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Draft tokens using the merged lookup (DART-style).
    pub fn draft_tokens(&self, context: &[u32], max_drafts: usize) -> Vec<u32> {
        let predictions = self.lookup(context);
        predictions
            .iter()
            .take(max_drafts)
            .map(|(tok, _)| *tok)
            .collect()
    }

    /// Multi-step draft sequence using chained merged lookups.
    pub fn draft_sequence(&self, context: &[u32], steps: usize) -> Vec<u32> {
        let mut ctx = context.to_vec();
        let mut drafts = Vec::with_capacity(steps);
        for _ in 0..steps {
            let predictions = self.lookup(&ctx);
            if predictions.is_empty() {
                break;
            }
            let top_token = predictions[0].0;
            drafts.push(top_token);
            ctx.push(top_token);
        }
        drafts
    }

    /// Bias logits using merged predictions (delegates to EngramLookup::bias_logits).
    pub fn bias_logits(logits: &mut [f32], predictions: &[(u32, f32)], alpha: f32) {
        EngramLookup::bias_logits(logits, predictions, alpha);
    }

    /// Summary string for logging.
    pub fn summary(&self) -> String {
        let names: Vec<&str> = self.tables.iter().map(|(n, _)| n.as_str()).collect();
        format!(
            "{} table(s): [{}]",
            self.tables.len(),
            names.join(", ")
        )
    }
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

impl EngramLookup {
    /// Build an engram binary file from a flat corpus of token IDs.
    ///
    /// # Algorithm
    /// 1. Slide a window of `order + 1` tokens across the corpus.
    ///    The first `order` tokens form the key; the last token is the next token.
    /// 2. Accumulate per-key → per-next-token frequency counts.
    /// 3. Choose a power-of-two table size with a load factor ≤ 0.5.
    /// 4. Write the header, hash table, and data section.
    ///
    /// # Arguments
    /// - `corpus`: flat array of token IDs (must be longer than `order`).
    /// - `order`: n-gram order (e.g. 3 for trigrams → context key is 3 tokens).
    /// - `output_path`: file path to write.
    ///
    /// # Errors
    /// Returns a description if the file cannot be created or written.
    pub fn build(corpus: &[u32], order: usize, output_path: &str) -> Result<(), String> {
        if order == 0 {
            return Err("order must be >= 1".to_string());
        }
        if corpus.len() <= order {
            return Err(format!(
                "Corpus too short ({} tokens) for order {order}",
                corpus.len()
            ));
        }

        // Step 1 — count next-token frequencies.
        // ngram_map: FNV hash → HashMap<next_token, frequency>
        let mut ngram_map: HashMap<u64, HashMap<u32, u32>> = HashMap::new();

        for window in corpus.windows(order + 1) {
            let key_tokens = &window[..order];
            let next_token = window[order];
            let hash = Self::hash_ngram(key_tokens);
            let nexts = ngram_map.entry(hash).or_default();
            *nexts.entry(next_token).or_insert(0) += 1;
        }

        let num_entries = ngram_map.len();

        // Step 2 — choose table size: next power of two, load factor <= 0.5.
        let table_size = next_power_of_two((num_entries * 2).max(16));

        // Step 3 — build the in-memory hash table (parallel arrays).
        // Each slot: (hash: u64, data_offset: u32, total_count: u32).
        let mut slots: Vec<(u64, u32, u32)> = vec![(EMPTY_HASH, 0, 0); table_size];

        // Data section: accumulate raw bytes here.
        let mut data_bytes: Vec<u8> = Vec::new();

        for (hash, nexts_map) in &ngram_map {
            // Compute total count for this n-gram.
            let total_count: u32 = nexts_map.values().copied().sum();

            // Write data entry: num_nexts u32, then (token u32, freq u32) pairs.
            let data_offset = data_bytes.len() as u32;
            let num_nexts = nexts_map.len() as u32;
            data_bytes.extend_from_slice(&num_nexts.to_le_bytes());
            // Sort by descending frequency for deterministic output.
            let mut nexts_sorted: Vec<(u32, u32)> = nexts_map
                .iter()
                .map(|(&tok, &freq)| (tok, freq))
                .collect();
            nexts_sorted.sort_by(|a, b| b.1.cmp(&a.1));
            for (tok, freq) in nexts_sorted {
                data_bytes.extend_from_slice(&tok.to_le_bytes());
                data_bytes.extend_from_slice(&freq.to_le_bytes());
            }

            // Insert into the open-addressing hash table.
            let mut probe = (*hash as usize) % table_size;
            loop {
                if slots[probe].0 == EMPTY_HASH {
                    slots[probe] = (*hash, data_offset, total_count);
                    break;
                }
                probe = (probe + 1) % table_size;
            }
        }

        // Step 4 — write file.
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(output_path)
            .map_err(|e| format!("Cannot create {output_path}: {e}"))?;

        let mut writer = BufWriter::new(file);

        // Header
        writer
            .write_all(&MAGIC.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&VERSION.to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(order as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(table_size as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;
        writer
            .write_all(&(num_entries as u32).to_le_bytes())
            .map_err(|e| e.to_string())?;

        // Hash table slots
        for (hash, offset, count) in &slots {
            writer
                .write_all(&hash.to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&offset.to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&count.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }

        // Data section
        writer
            .write_all(&data_bytes)
            .map_err(|e| e.to_string())?;

        writer.flush().map_err(|e| e.to_string())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Returns the smallest power of two >= n (minimum 1).
fn next_power_of_two(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // hash_ngram
    // -------------------------------------------------------------------------

    #[test]
    fn test_hash_ngram_deterministic() {
        let tokens = [1u32, 2, 3];
        let h1 = EngramLookup::hash_ngram(&tokens);
        let h2 = EngramLookup::hash_ngram(&tokens);
        assert_eq!(h1, h2, "hash must be deterministic");
    }

    #[test]
    fn test_hash_ngram_different_inputs() {
        // Different token sequences must (almost certainly) hash differently.
        let ha = EngramLookup::hash_ngram(&[1, 2, 3]);
        let hb = EngramLookup::hash_ngram(&[3, 2, 1]);
        let hc = EngramLookup::hash_ngram(&[1, 2, 4]);
        let hd = EngramLookup::hash_ngram(&[1]);
        assert_ne!(ha, hb, "order matters");
        assert_ne!(ha, hc, "last token matters");
        assert_ne!(ha, hd, "length matters");
    }

    #[test]
    fn test_hash_ngram_non_zero() {
        // Hash must never be EMPTY_HASH (0) for real inputs, since 0 is our sentinel.
        for seq in &[vec![0u32], vec![0, 0], vec![0, 0, 0], vec![1, 2, 3]] {
            let h = EngramLookup::hash_ngram(seq);
            assert_ne!(h, EMPTY_HASH, "hash({seq:?}) must not be 0 (sentinel)");
        }
    }

    // -------------------------------------------------------------------------
    // next_power_of_two
    // -------------------------------------------------------------------------

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(1024), 1024);
    }

    // -------------------------------------------------------------------------
    // build + lookup (round-trip)
    // -------------------------------------------------------------------------

    /// Build a small engram file, reload it, and check lookups.
    fn make_temp_path(suffix: &str) -> String {
        format!("/tmp/chimere_engram_test_{suffix}.bin")
    }

    #[test]
    fn test_build_and_lookup_basic() {
        // Corpus: a simple repeating pattern ABCABC…
        // Bigram (order=2): [A,B]->C, [B,C]->A, [C,A]->B (each seen twice)
        let corpus: Vec<u32> = vec![10, 20, 30, 10, 20, 30, 10, 20, 30];
        let path = make_temp_path("basic");

        EngramLookup::build(&corpus, 2, &path).expect("build failed");
        let lut = EngramLookup::from_file(&path).expect("from_file failed");

        // [10, 20] -> next should be 30 with probability 1.0
        let preds = lut.lookup(&[10, 20]);
        assert!(!preds.is_empty(), "Expected predictions for [10,20]");
        assert_eq!(preds[0].0, 30, "Expected next token 30");
        assert!(
            (preds[0].1 - 1.0).abs() < 1e-6,
            "Expected probability 1.0, got {}",
            preds[0].1
        );

        // [20, 30] -> next should be 10
        let preds2 = lut.lookup(&[20, 30]);
        assert!(!preds2.is_empty());
        assert_eq!(preds2[0].0, 10);

        // Unknown n-gram -> empty
        let preds3 = lut.lookup(&[99, 99]);
        assert!(preds3.is_empty(), "Unknown n-gram should return empty");

        // Context shorter than order -> empty
        let preds4 = lut.lookup(&[10]);
        assert!(preds4.is_empty(), "Short context should return empty");
    }

    #[test]
    fn test_build_and_lookup_multiple_next_tokens() {
        // Corpus: [1,2] appears before both 3 and 4, each twice.
        //   1,2,3,  1,2,4,  1,2,3,  1,2,4
        let corpus: Vec<u32> = vec![1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4];
        let path = make_temp_path("multi_next");

        EngramLookup::build(&corpus, 2, &path).expect("build failed");
        let lut = EngramLookup::from_file(&path).expect("from_file failed");

        let preds = lut.lookup(&[1, 2]);
        assert_eq!(preds.len(), 2, "Expected 2 distinct next tokens");

        // Probabilities should each be 0.5
        for (tok, prob) in &preds {
            assert!(
                (*tok == 3 || *tok == 4),
                "Unexpected token {tok}"
            );
            assert!(
                (prob - 0.5).abs() < 1e-6,
                "Expected prob 0.5 for token {tok}, got {prob}"
            );
        }

        // Results must be sorted by descending probability (equal here, order arbitrary)
        assert!(
            preds[0].1 >= preds[1].1,
            "Results must be sorted descending by probability"
        );
    }

    #[test]
    fn test_lookup_uses_tail_of_context() {
        // lookup() uses the LAST `order` tokens of the provided context slice.
        let corpus: Vec<u32> = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let path = make_temp_path("tail_ctx");

        EngramLookup::build(&corpus, 2, &path).expect("build failed");
        let lut = EngramLookup::from_file(&path).expect("from_file failed");

        // Providing a longer context — only last 2 tokens [1,2] should be used.
        let preds_long = lut.lookup(&[99, 99, 99, 1, 2]);
        let preds_exact = lut.lookup(&[1, 2]);
        assert_eq!(
            preds_long, preds_exact,
            "lookup must use only the tail of the context"
        );
    }

    #[test]
    fn test_build_order_3() {
        // Trigram order.  1,2,3 -> 4; 2,3,4 -> 1; 3,4,1 -> 2; 4,1,2 -> 3
        let corpus: Vec<u32> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let path = make_temp_path("order3");

        EngramLookup::build(&corpus, 3, &path).expect("build failed");
        let lut = EngramLookup::from_file(&path).expect("from_file failed");

        let p = lut.lookup(&[1, 2, 3]);
        assert!(!p.is_empty());
        assert_eq!(p[0].0, 4);

        let p2 = lut.lookup(&[4, 1, 2]);
        assert!(!p2.is_empty());
        assert_eq!(p2[0].0, 3);

        // Bigram sub-context should NOT match (order = 3)
        let p3 = lut.lookup(&[2, 3]);
        assert!(
            p3.is_empty(),
            "Bigram context should not match trigram table"
        );
    }

    #[test]
    fn test_probabilities_sum_to_one() {
        let corpus: Vec<u32> = vec![5, 6, 7, 5, 6, 8, 5, 6, 7, 5, 6, 9];
        let path = make_temp_path("prob_sum");

        EngramLookup::build(&corpus, 2, &path).expect("build");
        let lut = EngramLookup::from_file(&path).expect("from_file");

        let preds = lut.lookup(&[5, 6]);
        let total: f32 = preds.iter().map(|(_, p)| p).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "Probabilities must sum to 1.0, got {total}"
        );
    }

    #[test]
    fn test_from_file_bad_magic() {
        let path = make_temp_path("bad_magic");
        std::fs::write(&path, b"\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00")
            .expect("write");
        let result = EngramLookup::from_file(&path);
        assert!(result.is_err());
        let msg = result.err().unwrap();
        assert!(msg.contains("magic"), "Expected 'magic' in: {msg}");
    }

    #[test]
    fn test_build_error_corpus_too_short() {
        let err = EngramLookup::build(&[1u32, 2], 3, "/tmp/never.bin");
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_build_error_order_zero() {
        let err = EngramLookup::build(&[1u32, 2, 3], 0, "/tmp/never.bin");
        assert!(err.is_err());
    }

    // -------------------------------------------------------------------------
    // bias_logits
    // -------------------------------------------------------------------------

    #[test]
    fn test_bias_logits_alpha_zero_no_effect() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let predictions = vec![(0u32, 0.9f32), (1u32, 0.1f32)];
        let original = logits.clone();
        EngramLookup::bias_logits(&mut logits, &predictions, 0.0);
        assert_eq!(logits, original, "alpha=0 must leave logits unchanged");
    }

    #[test]
    fn test_bias_logits_modifies_only_predicted_tokens() {
        let mut logits = vec![0.0f32; 10];
        // Token 3 with probability 0.9.
        // ln(0.9) ≈ -0.1054; alpha*ln(p) is the expected delta.
        let predictions = vec![(3u32, 0.9f32)];
        EngramLookup::bias_logits(&mut logits, &predictions, 1.0);

        let expected = 1.0f32 * (0.9f32).ln();
        assert!(
            (logits[3] - expected).abs() < 1e-6,
            "logits[3] should be alpha*ln(p): expected {expected}, got {}",
            logits[3]
        );
        for i in 0..10usize {
            if i != 3 {
                assert_eq!(logits[i], 0.0, "Unpredicted token {i} should be unchanged");
            }
        }
    }

    #[test]
    fn test_bias_logits_out_of_bounds_token_ignored() {
        let mut logits = vec![0.0f32; 4];
        // Token 100 is beyond vocab_size — must not panic or write out of bounds.
        let predictions = vec![(100u32, 0.5f32), (2u32, 0.5f32)];
        EngramLookup::bias_logits(&mut logits, &predictions, 1.0);
        // Only token 2 should be modified.
        assert!(logits[2] != 0.0, "Token 2 should be biased");
        assert_eq!(logits[0], 0.0);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[3], 0.0);
    }

    #[test]
    fn test_bias_logits_zero_prob_skipped() {
        let mut logits = vec![0.0f32; 4];
        // prob=0.0 must be skipped (ln(0) = -inf would corrupt logits)
        let predictions = vec![(1u32, 0.0f32), (2u32, 0.5f32)];
        EngramLookup::bias_logits(&mut logits, &predictions, 1.0);
        assert_eq!(logits[1], 0.0, "Zero-prob token must not be biased");
        assert!(logits[2] != 0.0, "Non-zero prob token should be biased");
    }

    // -------------------------------------------------------------------------
    // Magicoder engram build (integration test — skipped if dataset absent)
    // -------------------------------------------------------------------------

    #[test]
    fn test_build_engram_from_magicoder() {
        let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());
        let dataset_path = format!(
            "{home}/.chimere/workspaces/chimere/datasets/magicoder/data-oss_instruct-decontaminated.jsonl"
        );
        let tokenizer_path = format!(
            "{home}/.chimere/models/qwopus-27b-bf16/tokenizer.json"
        );
        let output_path = format!(
            "{home}/.chimere/workspaces/chimere/chimere-deltanet/data/magicoder.engr"
        );

        if !std::path::Path::new(&dataset_path).exists() {
            eprintln!("[ENGRAM] Skipping: dataset not found at {dataset_path}");
            return;
        }
        if !std::path::Path::new(&tokenizer_path).exists() {
            eprintln!("[ENGRAM] Skipping: tokenizer not found at {tokenizer_path}");
            return;
        }

        // Load tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer");

        // Read and tokenize first 10 000 entries
        let file = std::fs::File::open(&dataset_path).expect("open dataset");
        let reader = std::io::BufReader::new(file);

        let mut corpus: Vec<u32> = Vec::new();
        let mut count = 0usize;
        let mut skipped = 0usize;

        use std::io::BufRead;
        for line in reader.lines() {
            if count >= 10_000 {
                break;
            }
            let line = line.expect("read line");
            if line.trim().is_empty() {
                continue;
            }
            let val: serde_json::Value =
                serde_json::from_str(&line).expect("parse json");

            // Magicoder OSS-Instruct schema: problem + solution fields.
            // Concatenate both so we capture instruction→code n-gram patterns.
            let problem = val
                .get("problem")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let solution = val
                .get("solution")
                .and_then(|v| v.as_str())
                // Fall back to other common field names just in case.
                .or_else(|| val.get("response").and_then(|v| v.as_str()))
                .or_else(|| val.get("content").and_then(|v| v.as_str()))
                .or_else(|| val.get("output").and_then(|v| v.as_str()))
                .unwrap_or("");

            let text = if problem.is_empty() && solution.is_empty() {
                skipped += 1;
                continue;
            } else if problem.is_empty() {
                solution.to_string()
            } else if solution.is_empty() {
                problem.to_string()
            } else {
                format!("{problem}\n{solution}")
            };

            let encoding = tokenizer
                .encode(text.as_str(), false)
                .expect("tokenize");
            corpus.extend(encoding.get_ids().iter().copied());
            count += 1;
        }

        eprintln!(
            "[ENGRAM] Tokenized {} entries ({} skipped) → {} tokens",
            count,
            skipped,
            corpus.len()
        );
        assert!(
            corpus.len() > 3,
            "Corpus too short to build trigrams: {} tokens",
            corpus.len()
        );

        // Create output directory
        std::fs::create_dir_all(
            std::path::Path::new(&output_path)
                .parent()
                .unwrap(),
        )
        .ok();

        // Build order-3 engram (trigram → next token)
        let t0 = std::time::Instant::now();
        EngramLookup::build(&corpus, 3, &output_path).expect("build engram");
        let build_ms = t0.elapsed().as_millis();

        // Print file size
        let file_size = std::fs::metadata(&output_path)
            .expect("stat output file")
            .len();
        let file_size_mb = file_size as f64 / 1_048_576.0;
        eprintln!(
            "[ENGRAM] Build done in {}ms — file size: {:.2} MB ({} bytes)",
            build_ms, file_size_mb, file_size
        );

        // Verify: load back and check header
        let engram = EngramLookup::from_file(&output_path).expect("load engram");
        assert_eq!(engram.order(), 3, "Loaded engram order must be 3");
        eprintln!("[ENGRAM] Loaded OK — order={}", engram.order());

        // Test a lookup on the first trigram in the corpus
        if corpus.len() >= 4 {
            let preds = engram.lookup(&corpus[0..3]);
            eprintln!(
                "[ENGRAM] Lookup [{},{},{}] → {} predictions",
                corpus[0],
                corpus[1],
                corpus[2],
                preds.len()
            );
            if !preds.is_empty() {
                eprintln!(
                    "[ENGRAM] Top prediction: token {} (p={:.4})",
                    preds[0].0, preds[0].1
                );
                // Probabilities of all predictions should sum to ≈1.0
                let total: f32 = preds.iter().map(|(_, p)| p).sum();
                assert!(
                    (total - 1.0).abs() < 1e-4,
                    "Prediction probabilities must sum to 1.0, got {total}"
                );
            }
        }

        // Sanity-check a few more lookups against the actual corpus
        let mut hits = 0usize;
        let sample_count = 100.min(corpus.len().saturating_sub(3));
        for i in (0..sample_count).step_by(10) {
            let preds = engram.lookup(&corpus[i..i + 3]);
            if !preds.is_empty() {
                hits += 1;
            }
        }
        eprintln!(
            "[ENGRAM] Spot-check: {hits}/{} sampled trigrams returned predictions",
            sample_count / 10
        );
        assert!(hits > 0, "At least some trigrams should have predictions");
    }

    #[test]
    fn test_bias_logits_alpha_scaling() {
        // logits[t] += alpha * ln(p). Verify different alpha values scale linearly.
        let predictions = vec![(0u32, 0.5f32)];
        let log_p = (0.5f32).ln();

        let mut l05 = vec![0.0f32];
        EngramLookup::bias_logits(&mut l05, &predictions, 0.5);
        assert!((l05[0] - 0.5 * log_p).abs() < 1e-6);

        let mut l10 = vec![0.0f32];
        EngramLookup::bias_logits(&mut l10, &predictions, 1.0);
        assert!((l10[0] - 1.0 * log_p).abs() < 1e-6);

        let mut l20 = vec![0.0f32];
        EngramLookup::bias_logits(&mut l20, &predictions, 2.0);
        assert!((l20[0] - 2.0 * log_p).abs() < 1e-6);
    }

    // -------------------------------------------------------------------------
    // Full pipeline: build → load → lookup → bias_logits
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_pipeline_build_lookup_bias() {
        // 1. Deterministic p=1.0 case: token 30 always follows [10,20].
        //    ln(1.0) = 0 → bias delta is zero. Verify no corruption.
        let corpus: Vec<u32> = vec![10, 20, 30, 10, 20, 30, 10, 20, 30];
        let path = make_temp_path("pipeline");
        EngramLookup::build(&corpus, 2, &path).expect("build failed");
        let lut = EngramLookup::from_file(&path).expect("from_file failed");

        let preds = lut.lookup(&[10, 20]);
        assert!(!preds.is_empty());
        assert_eq!(preds[0].0, 30);

        let mut logits = vec![0.0f32; 200];
        EngramLookup::bias_logits(&mut logits, &preds, 1.0);
        // p=1.0 → ln(1.0)=0 → logits[30] unchanged
        assert!(
            (logits[30]).abs() < 1e-6,
            "p=1.0 → no logit change, got {}",
            logits[30]
        );

        // 2. Balanced p=0.5 case: tokens 3 and 4 each appear twice after [1,2].
        let corpus2: Vec<u32> = vec![1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 4];
        let path2 = make_temp_path("pipeline2");
        EngramLookup::build(&corpus2, 2, &path2).expect("build2 failed");
        let lut2 = EngramLookup::from_file(&path2).expect("from_file2 failed");

        let preds2 = lut2.lookup(&[1, 2]);
        assert_eq!(preds2.len(), 2, "Expected 2 next tokens");

        let mut logits2 = vec![0.0f32; 200];
        EngramLookup::bias_logits(&mut logits2, &preds2, 0.5);

        // Both tokens have p=0.5 → bias = alpha * ln(0.5) = 0.5 * (-0.6931) ≈ -0.3466
        let expected_bias = 0.5f32 * (0.5f32).ln();
        for &(tok, _) in &preds2 {
            let idx = tok as usize;
            assert!(
                (logits2[idx] - expected_bias).abs() < 1e-5,
                "Token {tok}: expected bias {expected_bias:.6}, got {:.6}",
                logits2[idx]
            );
        }
        // Uninvolved tokens remain at zero.
        assert_eq!(logits2[0], 0.0);
        assert_eq!(logits2[1], 0.0);
        assert_eq!(logits2[5], 0.0);
    }
}
