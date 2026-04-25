# M2 — Prefix Cache (RadixAttention-style) for chimere-server

**Status:** DRAFT (M2 epic, successor to M1 multi-slot at tip `8fc079a`)
**Author:** kevin@openclaw, 2026-04-24
**References:** SGLang RadixAttention (Zheng et al., 2024), vLLM `PrefixCache`, llama.cpp `--prompt-cache`, ik_llama `llama_state_seq_{get,set}_data`.

## 1. Why (KPIs)

Chimere-server is at ~65% vLLM single-GPU parity after M1 (multi-slot + round-robin admission). The next gap is **repeated-prefix amortization**. In the typical OpenClaw workload, **80% of turns share the leading system prompt + SOUL.md envelope (~2–4k tokens)**, and the first user turn in every session re-prefills the same ~1500 tokens of Jinja-rendered boilerplate. That prefill burns 15–30 ms/turn at 498 tok/s prefill throughput, inflating P50 latency and robbing decode budget.

**M2 targets (measured on `bench_prefix_repeat.py`, 200 requests, Qwen3.5 IQ3_S custom-mix, 2 slots):**
- **≥ 2× P50 latency improvement** on the repeated-system-prompt workload (cold first turn pays full prefill; warm turns pay ~0 prefill).
- **≥ 1.5× aggregate throughput** at 2 slots (because freed decode budget flows into more concurrent slots).
- **Hit rate ≥ 70%** on a mixed workload (system-prompt repeats + fresh ad-hoc prompts).
- **Zero regression** on quality: every hit followed by `forward_prefill(tokens[n_hit..])` must produce token-for-token identical logits to a cold run (validated in J7 against a reference capture).

## 2. Data structure

Token-ID-keyed **radix trie** (PATRICIA-compressed). Canonical reference: SGLang's RadixAttention (Zheng et al., Dec 2024). Chosen over vLLM's fixed-block hash because (a) Qwen3.5's Jinja output produces variable-length shared prefixes that don't align cleanly to 16- or 64-token blocks, and (b) a radix trie gives O(|prompt|) exact `longest_prefix` without a second-pass block walk.

- **Keys** = exact token-ID sequences as produced by `LlamaForward::tokenize(messages, /*add_special*/true)`. We key on the post-template output, so any system-prompt or chat-template variation (even whitespace changes inside Jinja) cleanly misses — safer than keying on the rendered string.
- **Values** = `Arc<KVBlock>` (see PoC). `Arc` lets multiple concurrent requests share the same KV without copy.
- **Edge labels** are `Vec<u32>` compressed runs — insertion splits on divergence.
- **Leaf-only values** are **not** enforced: any internal node may hold a value (because a shorter prompt is a valid prefix of a longer one and both deserve caching).

Memory: per-token overhead ≈ 4 B (u32) + trie pointer share. Dominant cost is the KV payload itself (~1 MB / 1k tokens at `q8_0` keys + `q4_0` values on IQ3_S).

## 3. Integration with ik_llama KV cache

### 3.1 FFI surface (read from `llama_backend.rs`)

ik_llama exposes (lines 364–366):

```c
size_t llama_state_seq_get_size(ctx, seq_id, flags);
size_t llama_state_seq_get_data(ctx, dst, size, seq_id, flags);
size_t llama_state_seq_set_data(ctx, src, size, seq_id, flags);
```

These serialize **the full per-`seq_id` KV cache + GDN recurrent state** in one blob. Chimere already wraps them as `LlamaForward::state_seq_save/restore` (lines 1097–1113) and reuses them in `agent_scheduler.rs` for multi-agent context switching.

### 3.2 Observed constraints & caveats

1. **Whole-sequence granularity, not block-level.** `llama_state_seq_get_data` dumps the *entire* seq range `[0, pos]`; there is no `(p0, p1)` equivalent. **Consequence:** the cache stores state snapshots taken at position `p = len(tokens_cached)`. To serve a prefix hit of length `n_hit`, we must restore the *exact* length-`n_hit` snapshot, then continue with `forward_prefill(tokens[n_hit..])`. We **cannot** splice a block of length `n_hit` into the middle of an active sequence — so the trie value is always "the full saved KV at this exact token count."

2. **GDN recurrent state is serialized too.** The Qwen3.5 architecture has 30/40 GDN layers. `llama_state_seq_get_data` captures the recurrent matrices of every GDN layer, which is why `agent_scheduler.rs` works for agent switching. **This is exactly what we need for prefix caching** — a prefix snapshot restores both the KV-attention pages and the GDN state matrices, so resumption from position `n_hit` produces identical logits.

3. **`seq_id` is per-slot.** In the M1 multi-slot scheduler (`slot_scheduler.rs:203`), each `Slot.id` is used as the `seq_id` inside `LlamaForward::forward_multi_seq` / `forward_batch_multiseq`. **Consequence for M2:** when restoring a cached prefix into slot `S`, we call `state_seq_set_data(ctx, src, size, S, 0)`. The saved blob itself is `seq_id`-independent (the structure serialization writes per-layer buffers, not per-slot metadata), so a blob saved from `seq_id=0` can be restored to `seq_id=3` — **verified behavior in ik_llama `llama_state_io.cpp`, Jan 2026 fork**.

4. **`pos` is not in the blob.** The `LlamaForward.pos` counter is Rust-side book-keeping (line 1116). When we `state_seq_restore`, we must also `set_pos(cached.token_count)` — same pattern as `agent_scheduler.rs:190`.

5. **Blob size.** At IQ3_S q8_0/q4_0 KV, the state blob is ≈ **1.1 MB per 1000 tokens** (measured in `agent_scheduler` telemetry logs). At 32k total cached tokens we need ~36 MB of system RAM — trivial. We keep blobs in **CPU host RAM** (not VRAM) so the cache never competes with model weights on the 16 GB RTX 5060 Ti.

### 3.3 Hit path (worker side, post-J4 wiring)

```text
admission (tokens) → trie.longest_prefix(tokens)
  ↳ None       → miss: forward_prefill(tokens); state_seq_save; trie.insert(tokens, block)
  ↳ Some(n, k) → hit:  state_seq_restore(slot.seq_id, &k.seq_bytes); set_pos(n);
                        forward_prefill(tokens[n..]); state_seq_save; trie.insert(tokens, block_new)
```

Note the trailing `insert(tokens, block_new)`: every hit produces a *longer* snapshot and we overwrite the trie entry. This is how the cache "grows" on repeated shared prefixes.

## 4. Eviction

- **Primary:** LRU by `last_hit` timestamp, capped by `max_nodes` (default 256) **and** `max_cached_bytes` (default 128 MB system RAM).
- **Refresh:** every successful `longest_prefix` updates `last_hit` on the hit node (implemented in PoC `longest_prefix_rec`).
- **Tie-breaking:** when two entries have the same `last_hit` (common at first boot), we prefer to evict the **shorter** entry — the longer one amortizes more future prefill.
- **Scan cost:** `find_lru_path` is a full DFS (O(|trie|)). At 256 entries this is ~10 µs; acceptable since eviction is only triggered on miss-insert.

## 5. Engram interaction (the **critical** compatibility check)

Chimere's Engram codebook (`engram_lookup::MultiEngramLookup`) runs at **decode** via `chimere_sampler_set_engram_bias` (logit bias applied per sampling step — `llama_backend.rs:1187`). **Engram does NOT modify the KV cache** — it is a post-logit bias computed from the current decode token, looked up against a per-slot history buffer (`Slot.engram_history`).

**Therefore:**
- Restoring a cached KV snapshot into a slot does **not** restore the slot's engram history. The scheduler must **reset `Slot.engram_history` to `tokens[..n_hit]`** after `state_seq_restore`, so subsequent decode steps see the right 2-gram/3-gram history when biasing logits.
- This is a `push_context` loop over `tokens[..n_hit]` in the slot admission path (the existing method `Slot::push_context`, `slot_scheduler.rs:408`, already does the right thing — just needs to be called for cache-hit admission too).
- **No engram-codebook invalidation needed**: the codebook itself is global, read-only during serving.

## 6. J4-rewrite integration (scheduler admission)

The M1 J4 admission path is `Scheduler::spawn_workers → rx.blocking_recv → req.run(meta)`. M2 wraps the `req.run` invocation with a prefix-cache check **before** the closure calls `forward_prefill`.

Sketch:
```rust
// Inside the closure built by chat_completions_stream, before forward_prefill:
let (start_pos, restored) = {
    let mut trie = app.prefix_trie.write().unwrap();
    match trie.longest_prefix(&tokens) {
        Some((n, kv)) => (n, Some(kv)),
        None => (0, None),
    }
};
if let Some(kv) = restored {
    llama.state_seq_restore(slot.seq_id as i32, &kv.seq_bytes)?;
    llama.set_pos(start_pos as i32);
    // Rebuild engram history.
    for &t in &tokens[..start_pos] { slot.push_context(t); }
}
let logits = llama.forward_prefill(&tokens[start_pos..])?;
// ... decode loop ...
// After decode finishes, optionally promote this longer snapshot to the trie:
if should_cache(&tokens, start_pos) {
    let bytes = llama.state_seq_save(slot.seq_id as i32)?;
    let mut trie = app.prefix_trie.write().unwrap();
    let id = trie.next_kv_id();
    trie.insert(&tokens, Arc::new(KVBlock::new(id, bytes, tokens.len())));
}
```

The `should_cache()` gate avoids caching ultra-short prompts (`< 512` tokens) where the save/restore overhead (≈ 2 ms blob memcpy + 1 ms FFI) exceeds the saved prefill time.

## 7. J1–J7 impl plan (mirrors M1 cadence)

| Day | Deliverable | Atomic commits |
|---|---|---|
| **M2-J1** | Scaffolding: `PrefixTrie`, `KVBlock`, `CacheStats`, `pub mod prefix_cache` in `lib.rs`. Unit tests only. | 2 |
| **M2-J2** | Stronger correctness tests: random prompt mix, stress with 10k inserts, property-test `longest_prefix` against a naive HashMap. | 2 |
| **M2-J3** | FFI wrappers already exist (`state_seq_save/restore`). Add `KVBlock::from_llama(&LlamaForward, seq_id) -> Result<Arc<KVBlock>>` helper + `apply_to(&mut LlamaForward, seq_id)` inverse. | 3 |
| **M2-J4** | Wire into `Scheduler::spawn_workers` closure path. Add `AppState.prefix_trie: Arc<RwLock<PrefixTrie>>`. Gated by `CHIMERE_PREFIX_CACHE=1`. | 4 |
| **M2-J5** | Eviction tuning: implement `max_cached_bytes` budget, expose tuning env vars, add eviction metrics. | 2 |
| **M2-J6** | `/v1/prefix_cache_stats` endpoint + `/metrics` Prometheus lines. Include hit_rate, avg_hit_tokens, cached_bytes, evictions. | 2 |
| **M2-J7** | Stress test `bench_prefix_repeat.py`. Acceptance: ≥ 2× P50 latency on repeated-prefix workload; ≥ 1.5× throughput at 2 slots; logit-equivalence on 100 sampled token positions. | 3 |

Total: 18 atomic commits, mirroring M1's cadence.

## 8. Risks & mitigations

1. **Restore skews decode quality** (logit drift). Mitigation: J7 captures cold-path logits for 100 prompts, compares against cache-hit logits; tolerance `max |Δ| < 1e-4` on the sampled positions. If drift appears, likely cause is a GDN state-restore bug in our ik_llama fork — fix in `ik_llama` source, not in this crate.
2. **`state_seq_get_data` too slow** (observed ≈ 0.8 ms/MB in `agent_scheduler` logs). At 32k cached tokens → ~30 ms per cold save. Mitigation: save asynchronously on a dedicated serialization thread; the admission path does not block on save. Only `restore` is on the hot path.
3. **Trie lock contention** under high QPS. Mitigation: `RwLock` (reads parallel, write rare), bounded to one write per cache-miss admission.
4. **Cache poisoning from tokenizer non-determinism.** Mitigation: `tokenize()` is deterministic in ik_llama; sanity-check with `assert_eq!(retokenize(decoded_text), tokens)` in debug builds during J2.
5. **Memory pressure from very long prompts.** Mitigation: hard `max_prompt_tokens_to_cache` env (default 16k) — longer prompts still work but skip the cache.

---

**Acceptance for M2 = green on J7 bench + `/v1/prefix_cache_stats` reporting hit_rate ≥ 0.70 on the mixed workload.**
