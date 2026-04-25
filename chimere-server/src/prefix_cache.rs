//! # M2 — Prefix Cache (J1 PoC + J2 CacheConfig env-reader)
//!
//! J1 scaffolding (radix trie + LRU + stats + 11 unit tests) is unchanged
//! from `m2-prefix-cache` tip `64e4680`. J2 adds a `CacheConfig` env-reader
//! so the scheduler can gate the whole feature behind `CHIMERE_PREFIX_CACHE`
//! and respect a bounded byte budget via `CHIMERE_PREFIX_CACHE_MAX_BYTES`.
//!
//! ## Design summary (J1, preserved)
//!
//! - Keys are `&[u32]` token IDs (the exact sequence produced by the
//!   tokenizer on the request's `messages`). Since Qwen3.5's vocabulary is
//!   ~152k tokens, storing full token sequences in trie nodes is cheap
//!   relative to the KV footprint (~1 MB / 1k tokens at Q8 KV cache).
//! - Each trie node owns a compressed "edge label" (`Vec<u32>`) that lets
//!   us walk multiple tokens in one comparison — classic PATRICIA trie.
//! - Leaf nodes hold an `Arc<KVBlock>`: multiple callers can share the
//!   same cached KV state without copying.
//! - LRU is tracked per-entry: `last_hit` is updated on every successful
//!   `longest_prefix()` or `insert()` and used by `evict_lru()`.
//!
//! ## J2 additions
//!
//! - [`CacheConfig::from_env`] — reads `CHIMERE_PREFIX_CACHE`,
//!   `CHIMERE_PREFIX_CACHE_MAX_BYTES`, `CHIMERE_PREFIX_CACHE_MAX_NODES`.
//! - [`PrefixTrie::from_config`] — thin constructor that applies a
//!   `CacheConfig`; equivalent to `with_byte_budget` when enabled, `None`
//!   caller-side when disabled.
//!
//! ## Not in scope here
//!
//! - Actual FFI serialisation of KV (lives in `llama_backend.rs` aliases
//!   `save_seq_state` / `restore_seq_state` — M2-J2).
//! - Wiring into `slot_scheduler::NativeDriver` admission (M2-J2 — patches
//!   in `APPLY.md`).
//! - Engram compatibility (doc in the M2 plan; `Slot::push_context` loop
//!   is inside the scheduler patch, not here).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// KV block placeholder — byte-level handle for J2 FFI wiring
// ---------------------------------------------------------------------------

/// Opaque handle to a serialized KV cache block.
///
/// In J1 this was a pure PoC; in J2 the scheduler populates `seq_bytes`
/// via `LlamaForward::save_seq_state` and consumes it via
/// `LlamaForward::restore_seq_state`. The rest of the shape is unchanged
/// so the J1 tests still apply verbatim.
///
/// `KVBlock` is immutable once inserted. Sharing is cheap via `Arc`.
#[derive(Debug)]
pub struct KVBlock {
    /// Opaque ID for logging / stats correlation.
    pub id: u32,
    /// Serialized bytes from `llama_state_seq_get_data`.
    pub seq_bytes: Vec<u8>,
    /// Number of tokens this block represents (the "prefix length" it
    /// covers). Needed so the scheduler can compute `n_hit = tokens[..n]`
    /// and continue prefill from position `n`.
    pub token_count: usize,
}

impl KVBlock {
    pub fn new(id: u32, seq_bytes: Vec<u8>, token_count: usize) -> Self {
        Self { id, seq_bytes, token_count }
    }

    /// Byte size of the serialized state (for stats + eviction accounting).
    pub fn byte_size(&self) -> usize {
        self.seq_bytes.len()
    }
}

// ---------------------------------------------------------------------------
// M2-J2 — CacheConfig (env reader + kill switch)
// ---------------------------------------------------------------------------

/// Default maximum cached bytes when unset: 1 GB (per mission spec).
pub const DEFAULT_MAX_CACHED_BYTES: usize = 1024 * 1024 * 1024;

/// Default maximum trie nodes when unset.
pub const DEFAULT_MAX_NODES: usize = 256;

/// Runtime configuration for the prefix cache, read once at scheduler
/// construction time. Changes require a process restart — env vars are
/// not polled.
///
/// ## Env vars
///
/// | Var | Type | Default | Meaning |
/// |---|---|---|---|
/// | `CHIMERE_PREFIX_CACHE` | `bool` (0/1/true/false/on/off, case-insensitive) | `0` | Master kill-switch. When **off**, the scheduler behaves bit-identically to M1 — no trie touch, no FFI save/restore. |
/// | `CHIMERE_PREFIX_CACHE_MAX_BYTES` | `usize` | `1_073_741_824` (1 GB) | Soft upper bound on sum of `KVBlock::byte_size()`. The trie enforces this via LRU eviction at insert time. |
/// | `CHIMERE_PREFIX_CACHE_MAX_NODES` | `usize` | `256` | Upper bound on the number of value-bearing entries (independent of bytes). |
///
/// ## Precedence
///
/// If the kill-switch is off, the other two vars are still *parsed* (so
/// malformed values produce a clear error at startup) but the cache is
/// not built. The scheduler's `prefix_trie: None` short-circuits every
/// hot-path.
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    pub enabled: bool,
    pub max_bytes: usize,
    pub max_nodes: usize,
}

impl CacheConfig {
    /// Read all three env vars. Never panics — malformed integers fall
    /// back to the defaults with an eprintln warning.
    pub fn from_env() -> Self {
        let enabled = std::env::var("CHIMERE_PREFIX_CACHE")
            .map(|v| parse_bool_env(&v))
            .unwrap_or(false);

        let max_bytes = std::env::var("CHIMERE_PREFIX_CACHE_MAX_BYTES")
            .ok()
            .and_then(|v| {
                v.trim().parse::<usize>().map_err(|e| {
                    eprintln!(
                        "[prefix_cache] CHIMERE_PREFIX_CACHE_MAX_BYTES parse error: {} \
                         (falling back to {} B)",
                        e, DEFAULT_MAX_CACHED_BYTES,
                    );
                    e
                }).ok()
            })
            .unwrap_or(DEFAULT_MAX_CACHED_BYTES);

        let max_nodes = std::env::var("CHIMERE_PREFIX_CACHE_MAX_NODES")
            .ok()
            .and_then(|v| {
                v.trim().parse::<usize>().map_err(|e| {
                    eprintln!(
                        "[prefix_cache] CHIMERE_PREFIX_CACHE_MAX_NODES parse error: {} \
                         (falling back to {})",
                        e, DEFAULT_MAX_NODES,
                    );
                    e
                }).ok()
            })
            .unwrap_or(DEFAULT_MAX_NODES);

        // Zero budgets force the cache off even if the kill-switch is on.
        let effectively_enabled = enabled && max_bytes > 0 && max_nodes > 0;
        if enabled && !effectively_enabled {
            eprintln!(
                "[prefix_cache] CHIMERE_PREFIX_CACHE=1 but max_bytes={} max_nodes={} → \
                 treating as disabled",
                max_bytes, max_nodes,
            );
        }

        Self {
            enabled: effectively_enabled,
            max_bytes,
            max_nodes,
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_bytes: DEFAULT_MAX_CACHED_BYTES,
            max_nodes: DEFAULT_MAX_NODES,
        }
    }
}

/// Canonical boolean parser used by several env vars in this repo.
/// Accepts `"1"`, `"true"`, `"yes"`, `"on"` (case-insensitive) as true;
/// everything else (including empty string) as false.
fn parse_bool_env(s: &str) -> bool {
    let t = s.trim();
    if t.is_empty() {
        return false;
    }
    t == "1"
        || t.eq_ignore_ascii_case("true")
        || t.eq_ignore_ascii_case("yes")
        || t.eq_ignore_ascii_case("on")
}

// ---------------------------------------------------------------------------
// Cache statistics (J1 — unchanged)
// ---------------------------------------------------------------------------

/// Cumulative counters for `/v1/prefix_cache_stats`. Use atomics so the
/// endpoint can read without locking the trie.
#[derive(Debug, Default)]
pub struct CacheStats {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    /// Sum of `n_hit` across all hits — for average prefix reuse telemetry.
    pub total_hit_tokens: AtomicU64,
    /// Sum of `prompt_len` across all `longest_prefix` calls.
    pub total_query_tokens: AtomicU64,
}

impl CacheStats {
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            total_hit_tokens: self.total_hit_tokens.load(Ordering::Relaxed),
            total_query_tokens: self.total_query_tokens.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn record_hit(&self, n_hit: usize, prompt_len: usize) {
        self.hits.fetch_add(1, Ordering::Relaxed);
        self.total_hit_tokens.fetch_add(n_hit as u64, Ordering::Relaxed);
        self.total_query_tokens.fetch_add(prompt_len as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_miss(&self, prompt_len: usize) {
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.total_query_tokens.fetch_add(prompt_len as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_eviction(&self, count: u64) {
        self.evictions.fetch_add(count, Ordering::Relaxed);
    }
}

/// Cheap-copy snapshot for the stats endpoint.
#[derive(Debug, Clone, Copy)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_hit_tokens: u64,
    pub total_query_tokens: u64,
}

impl CacheStatsSnapshot {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
    pub fn avg_hit_tokens(&self) -> f64 {
        if self.hits == 0 { 0.0 } else { self.total_hit_tokens as f64 / self.hits as f64 }
    }
}

// ---------------------------------------------------------------------------
// Trie node (J1 — unchanged)
// ---------------------------------------------------------------------------

/// A PATRICIA-style trie node. The edge from parent→self is `edge_label`;
/// children are indexed by their first token.
struct TrieNode {
    /// Compressed edge label (sequence of token IDs consumed along this edge).
    /// Empty for the root node.
    edge_label: Vec<u32>,
    /// KV block stored at this node (if any). A node is a "cache entry" iff
    /// this is `Some`.
    value: Option<Arc<KVBlock>>,
    /// Child nodes, keyed by the first token of their edge label.
    children: HashMap<u32, TrieNode>,
    /// Last hit timestamp (updated on `insert` and successful `longest_prefix`
    /// descent through this node's value). Used for LRU eviction.
    last_hit: Instant,
}

impl TrieNode {
    fn new(edge_label: Vec<u32>) -> Self {
        Self {
            edge_label,
            value: None,
            children: HashMap::new(),
            last_hit: Instant::now(),
        }
    }

    /// Count the number of `value`-bearing descendants (including self).
    fn count_values(&self) -> usize {
        let mine = if self.value.is_some() { 1 } else { 0 };
        let kids: usize = self.children.values().map(|c| c.count_values()).sum();
        mine + kids
    }
}

// ---------------------------------------------------------------------------
// PrefixTrie (J1 API preserved; J2 adds from_config + byte-budget eviction on insert)
// ---------------------------------------------------------------------------

/// Token-keyed radix trie storing references to saved KV blocks.
///
/// Thread-safety: **not** `Sync`. Callers (the scheduler worker) must wrap
/// in a `Mutex` or `RwLock`. We expect writes to be rare (once per admission
/// that misses) and reads to be fast (once per admission), so a `RwLock`
/// is the natural choice when M2 J4 wires this in.
pub struct PrefixTrie {
    root: TrieNode,
    /// Upper bound on the number of value-bearing entries. Exceeding this
    /// triggers LRU eviction.
    max_nodes: usize,
    /// Soft byte budget for the serialized KV data (sum of `byte_size()`).
    /// 0 → no byte bound.
    max_cached_bytes: usize,
    pub stats: CacheStats,
    next_block_id: u32,
}

impl PrefixTrie {
    pub fn new(max_nodes: usize) -> Self {
        Self {
            root: TrieNode::new(Vec::new()),
            max_nodes,
            max_cached_bytes: 0,
            stats: CacheStats::default(),
            next_block_id: 0,
        }
    }

    pub fn with_byte_budget(max_nodes: usize, max_cached_bytes: usize) -> Self {
        let mut t = Self::new(max_nodes);
        t.max_cached_bytes = max_cached_bytes;
        t
    }

    /// M2-J2 — construct a `PrefixTrie` sized per the `CacheConfig`. The
    /// caller is responsible for not calling this when `cfg.enabled` is
    /// false (the scheduler just wires `prefix_trie: None` in that case).
    pub fn from_config(cfg: &CacheConfig) -> Self {
        Self::with_byte_budget(cfg.max_nodes, cfg.max_bytes)
    }

    /// Number of value-bearing entries currently in the trie.
    pub fn len(&self) -> usize {
        self.root.count_values()
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Insert `tokens → kv_handle`. Returns `true` if this is a fresh entry,
    /// `false` if it replaced an existing value at the same prefix.
    ///
    /// After insertion, evicts LRU entries until both `len() <= max_nodes`
    /// AND `cached_bytes() <= max_cached_bytes` (when the byte budget is
    /// non-zero).
    pub fn insert(&mut self, tokens: &[u32], kv_handle: Arc<KVBlock>) -> bool {
        let fresh = Self::insert_rec(&mut self.root, tokens, kv_handle);
        // Node-count bound.
        while self.len() > self.max_nodes {
            if !self.evict_one() { break; }
        }
        // Byte-count bound (M2-J2). Only active when max_cached_bytes > 0.
        if self.max_cached_bytes > 0 {
            while self.cached_bytes() > self.max_cached_bytes {
                if !self.evict_one() { break; }
            }
        }
        fresh
    }

    fn insert_rec(node: &mut TrieNode, tokens: &[u32], kv: Arc<KVBlock>) -> bool {
        node.last_hit = Instant::now();
        if tokens.is_empty() {
            let fresh = node.value.is_none();
            node.value = Some(kv);
            return fresh;
        }

        let first = tokens[0];
        if let Some(child) = node.children.get_mut(&first) {
            let common = common_prefix_len(tokens, &child.edge_label);
            if common == child.edge_label.len() {
                return Self::insert_rec(child, &tokens[common..], kv);
            }
            // Partial match → split the edge at `common`.
            let mut old_child = node.children.remove(&first).unwrap();
            let split_label: Vec<u32> = old_child.edge_label[common..].to_vec();
            old_child.edge_label = split_label.clone();

            let mut intermediate = TrieNode::new(tokens[..common].to_vec());
            intermediate.last_hit = Instant::now();

            if !split_label.is_empty() {
                intermediate.children.insert(split_label[0], old_child);
            } else {
                intermediate.value = old_child.value.take();
            }

            let remainder = &tokens[common..];
            let fresh = Self::insert_rec(&mut intermediate, remainder, kv);

            node.children.insert(first, intermediate);
            fresh
        } else {
            let mut fresh_node = TrieNode::new(tokens.to_vec());
            fresh_node.last_hit = Instant::now();
            fresh_node.value = Some(kv);
            node.children.insert(first, fresh_node);
            true
        }
    }

    /// Find the longest prefix of `tokens` that has an associated KV block.
    /// Returns `(n_hit, kv_handle)` where `n_hit` is the number of tokens
    /// covered by the cached block (≤ `tokens.len()`). Returns `None` if
    /// no prefix matches at all (not even the empty root).
    ///
    /// Side effect: updates `last_hit` on the winning node so it is fresh
    /// for LRU.
    pub fn longest_prefix(&mut self, tokens: &[u32]) -> Option<(usize, Arc<KVBlock>)> {
        let result = Self::longest_prefix_rec(&mut self.root, tokens, 0);
        match &result {
            Some((n, _)) => self.stats.record_hit(*n, tokens.len()),
            None => self.stats.record_miss(tokens.len()),
        }
        result
    }

    fn longest_prefix_rec(
        node: &mut TrieNode,
        tokens: &[u32],
        depth: usize,
    ) -> Option<(usize, Arc<KVBlock>)> {
        let mut best: Option<(usize, Arc<KVBlock>)> = node.value.as_ref()
            .map(|kv| (depth, Arc::clone(kv)));
        if best.is_some() {
            node.last_hit = Instant::now();
        }

        if let Some(first) = tokens.first().copied() {
            if let Some(child) = node.children.get_mut(&first) {
                let common = common_prefix_len(tokens, &child.edge_label);
                if common == child.edge_label.len() {
                    let child_depth = depth + common;
                    if let Some(deeper) =
                        Self::longest_prefix_rec(child, &tokens[common..], child_depth)
                    {
                        return Some(deeper);
                    }
                }
                // Partial edge match: `best` from above is the answer.
                // (Falls through to the return below.)
            }
        }
        best
    }

    /// Evict entries (oldest `last_hit` first) until `len() ≤ keep`.
    /// Returns the number of entries evicted.
    pub fn evict_lru(&mut self, keep: usize) -> u64 {
        let mut count = 0u64;
        while self.len() > keep {
            if !self.evict_one() { break; }
            count += 1;
        }
        self.stats.record_eviction(count);
        count
    }

    /// Evict the single oldest entry. Returns `false` if the trie is empty.
    fn evict_one(&mut self) -> bool {
        let victim_path = self.find_lru_path();
        let Some(path) = victim_path else { return false };
        let cleared = Self::clear_value_at(&mut self.root, &path);
        if cleared {
            self.stats.record_eviction(1);
        }
        cleared
    }

    /// DFS to find the path (sequence of first-tokens) leading to the
    /// value-bearing node with the oldest `last_hit`.
    fn find_lru_path(&self) -> Option<Vec<u32>> {
        fn dfs(
            node: &TrieNode,
            cur_path: &mut Vec<u32>,
            best: &mut Option<(Instant, Vec<u32>)>,
        ) {
            if node.value.is_some() {
                let replace = match best {
                    None => true,
                    Some((ts, _)) => node.last_hit < *ts,
                };
                if replace {
                    *best = Some((node.last_hit, cur_path.clone()));
                }
            }
            for (&key, child) in &node.children {
                cur_path.push(key);
                dfs(child, cur_path, best);
                cur_path.pop();
            }
        }
        let mut best: Option<(Instant, Vec<u32>)> = None;
        let mut cur = Vec::new();
        dfs(&self.root, &mut cur, &mut best);
        best.map(|(_, path)| path)
    }

    fn clear_value_at(node: &mut TrieNode, path: &[u32]) -> bool {
        if path.is_empty() {
            return node.value.take().is_some();
        }
        let Some(child) = node.children.get_mut(&path[0]) else { return false };
        Self::clear_value_at(child, &path[1..])
    }

    /// Allocate the next KVBlock ID. Used by callers that construct
    /// `KVBlock` instances (typically the scheduler after a successful
    /// prefill → `save_seq_state`).
    pub fn next_kv_id(&mut self) -> u32 {
        let id = self.next_block_id;
        self.next_block_id = self.next_block_id.wrapping_add(1);
        id
    }

    /// Total serialized bytes across all cached blocks.
    pub fn cached_bytes(&self) -> usize {
        fn walk(node: &TrieNode, acc: &mut usize) {
            if let Some(ref v) = node.value { *acc += v.byte_size(); }
            for c in node.children.values() { walk(c, acc); }
        }
        let mut acc = 0;
        walk(&self.root, &mut acc);
        acc
    }
}

/// Length of the shared prefix between two token slices.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------
// Tests — J1 set (11 tests) + J2 additions (CacheConfig + byte-budget eviction)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_block(id: u32, n_tokens: usize) -> Arc<KVBlock> {
        Arc::new(KVBlock::new(id, vec![0u8; n_tokens * 128], n_tokens))
    }

    fn mk_block_bytes(id: u32, n_bytes: usize) -> Arc<KVBlock> {
        Arc::new(KVBlock::new(id, vec![0u8; n_bytes], n_bytes / 128))
    }

    #[test]
    fn empty_trie_returns_none() {
        let mut t = PrefixTrie::new(16);
        assert!(t.is_empty());
        assert!(t.longest_prefix(&[1, 2, 3]).is_none());
        let s = t.stats.snapshot();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 0);
    }

    #[test]
    fn insert_and_longest_prefix_exact_match() {
        let mut t = PrefixTrie::new(16);
        let tokens = vec![10u32, 20, 30, 40];
        let kv = mk_block(1, 4);
        assert!(t.insert(&tokens, Arc::clone(&kv)));
        assert_eq!(t.len(), 1);

        let hit = t.longest_prefix(&tokens).expect("exact match");
        assert_eq!(hit.0, 4);
        assert_eq!(hit.1.id, 1);
    }

    #[test]
    fn shared_prefix_returns_longest() {
        let mut t = PrefixTrie::new(16);
        t.insert(&[1, 2, 3], mk_block(1, 3));
        t.insert(&[1, 2, 3, 4, 5], mk_block(2, 5));
        t.insert(&[1, 2, 3, 4, 5, 6, 7, 8], mk_block(3, 8));
        assert_eq!(t.len(), 3);

        let query = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9];
        let (n_hit, kv) = t.longest_prefix(&query).expect("should hit");
        assert_eq!(n_hit, 8);
        assert_eq!(kv.id, 3);
    }

    #[test]
    fn partial_prefix_match_returns_shortest_valid() {
        let mut t = PrefixTrie::new(16);
        t.insert(&[1, 2, 3, 4, 5], mk_block(1, 5));
        let hit = t.longest_prefix(&[1, 2, 3, 9, 9]);
        assert!(hit.is_none(), "partial edge should not count as a hit");

        t.insert(&[1, 2, 3], mk_block(2, 3));
        let (n_hit, kv) = t.longest_prefix(&[1, 2, 3, 9, 9]).unwrap();
        assert_eq!(n_hit, 3);
        assert_eq!(kv.id, 2);
    }

    #[test]
    fn no_match_returns_none() {
        let mut t = PrefixTrie::new(16);
        t.insert(&[100, 200, 300], mk_block(1, 3));
        let hit = t.longest_prefix(&[1, 2, 3]);
        assert!(hit.is_none());
    }

    #[test]
    fn insert_beyond_max_nodes_evicts_lru() {
        let mut t = PrefixTrie::new(2);
        t.insert(&[1, 2, 3], mk_block(1, 3));
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.insert(&[4, 5, 6], mk_block(2, 3));
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.insert(&[7, 8, 9], mk_block(3, 3));
        assert_eq!(t.len(), 2, "LRU eviction should cap at max_nodes");
        assert!(t.longest_prefix(&[1, 2, 3]).is_none());
        assert!(t.longest_prefix(&[4, 5, 6]).is_some());
        assert!(t.longest_prefix(&[7, 8, 9]).is_some());

        let s = t.stats.snapshot();
        assert!(s.evictions >= 1);
    }

    #[test]
    fn longest_prefix_refreshes_lru() {
        let mut t = PrefixTrie::new(2);
        t.insert(&[1, 2, 3], mk_block(1, 3));
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.insert(&[4, 5, 6], mk_block(2, 3));
        std::thread::sleep(std::time::Duration::from_millis(5));
        let _ = t.longest_prefix(&[1, 2, 3]);
        std::thread::sleep(std::time::Duration::from_millis(5));
        t.insert(&[7, 8, 9], mk_block(3, 3));
        assert_eq!(t.len(), 2);
        assert!(t.longest_prefix(&[1, 2, 3]).is_some(),
                "refreshed entry should survive");
        assert!(t.longest_prefix(&[4, 5, 6]).is_none(),
                "un-touched entry should be evicted");
    }

    #[test]
    fn evict_lru_keeps_exactly_n() {
        let mut t = PrefixTrie::new(100);
        for i in 0..10u32 {
            t.insert(&[i, i + 1, i + 2], mk_block(i, 3));
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(t.len(), 10);
        let evicted = t.evict_lru(3);
        assert_eq!(evicted, 7);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn edge_split_preserves_existing_entries() {
        let mut t = PrefixTrie::new(16);
        t.insert(&[1, 2, 3, 4, 5], mk_block(1, 5));
        t.insert(&[1, 2, 3], mk_block(2, 3));
        assert_eq!(t.len(), 2);

        let (n, kv) = t.longest_prefix(&[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(n, 5);
        assert_eq!(kv.id, 1);

        let (n, kv) = t.longest_prefix(&[1, 2, 3, 9]).unwrap();
        assert_eq!(n, 3);
        assert_eq!(kv.id, 2);
    }

    #[test]
    fn cache_stats_track_hit_rate() {
        let mut t = PrefixTrie::new(16);
        t.insert(&[1, 2, 3], mk_block(1, 3));

        let _ = t.longest_prefix(&[1, 2, 3]);
        let _ = t.longest_prefix(&[1, 2, 3, 4]);
        let _ = t.longest_prefix(&[99, 99]);

        let s = t.stats.snapshot();
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 1);
        assert!((s.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(s.total_hit_tokens, 6);
        assert!((s.avg_hit_tokens() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn cached_bytes_sums_all_blocks() {
        let mut t = PrefixTrie::new(16);
        t.insert(&[1, 2, 3], mk_block(1, 3));
        t.insert(&[4, 5, 6, 7, 8], mk_block(2, 5));
        assert_eq!(t.cached_bytes(), 384 + 640);
    }

    #[test]
    fn insert_replacement_is_not_fresh() {
        let mut t = PrefixTrie::new(16);
        assert!(t.insert(&[1, 2, 3], mk_block(1, 3)));
        assert!(!t.insert(&[1, 2, 3], mk_block(2, 3)));
        let (_, kv) = t.longest_prefix(&[1, 2, 3]).unwrap();
        assert_eq!(kv.id, 2, "second insert should replace the block");
    }

    #[test]
    fn common_prefix_len_basic() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[9]), 0);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
    }

    // ----------------------------------------------------------------
    // M2-J2 — CacheConfig + byte-budget eviction
    // ----------------------------------------------------------------

    #[test]
    fn cache_config_default_is_disabled() {
        let cfg = CacheConfig::default();
        assert!(!cfg.enabled);
        assert_eq!(cfg.max_bytes, DEFAULT_MAX_CACHED_BYTES);
        assert_eq!(cfg.max_nodes, DEFAULT_MAX_NODES);
    }

    #[test]
    fn parse_bool_env_accepts_common_forms() {
        for s in ["1", "true", "TRUE", "True", "yes", "on", "YeS", "On"] {
            assert!(parse_bool_env(s), "expected true for {:?}", s);
        }
        for s in ["0", "false", "no", "off", "", "   ", "garbage"] {
            assert!(!parse_bool_env(s), "expected false for {:?}", s);
        }
    }

    #[test]
    fn from_config_applies_byte_budget() {
        let cfg = CacheConfig { enabled: true, max_bytes: 1024, max_nodes: 100 };
        let t = PrefixTrie::from_config(&cfg);
        assert_eq!(t.max_cached_bytes, 1024);
        assert_eq!(t.max_nodes, 100);
    }

    #[test]
    fn byte_budget_evicts_on_overflow() {
        // max_bytes = 400 — room for exactly one 384-byte block (3 tokens
        // * 128 B mk_block), second insert must evict the first.
        let mut t = PrefixTrie::with_byte_budget(16, 400);
        t.insert(&[1, 2, 3], mk_block_bytes(1, 384));
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Inserting a second 384-byte block takes us to 768 > 400 → evict.
        t.insert(&[4, 5, 6], mk_block_bytes(2, 384));
        assert!(t.cached_bytes() <= 400, "byte budget must be honoured");
        assert_eq!(t.len(), 1, "byte budget should keep one entry");
        // The survivor is the newly-inserted (freshest) one.
        assert!(t.longest_prefix(&[4, 5, 6]).is_some());
        assert!(t.longest_prefix(&[1, 2, 3]).is_none());
    }

    #[test]
    fn zero_byte_budget_means_unbounded_bytes() {
        // with_byte_budget(..., 0) behaves like `new` — only node count matters.
        let mut t = PrefixTrie::with_byte_budget(16, 0);
        for i in 0..10u32 {
            t.insert(&[i, i + 1, i + 2], mk_block_bytes(i, 10_000));
        }
        assert_eq!(t.len(), 10);
        assert_eq!(t.cached_bytes(), 100_000);
    }
}
