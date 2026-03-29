# Chimere-DeltaNet Codebase Audit

**Date**: 2026-03-16
**Scope**: Post-optimization sprint cleanup. ~40 agents made changes.
**Total source**: ~26,500 lines across 28 .rs files + 1 .cu file + backups

---

## 1. Dead Code and Unused Imports

### Compiler warnings (cargo build --release)

Only 2 warnings — the codebase is cleaner than expected:

| Count | Location | What |
|-------|----------|------|
| 1 | `src/kernels/iq3s_gemv.rs:525` | `const IQ3S_BLOCK_BYTES: usize = 110` — never used (duplicate of inline constant in CUDA source) |
| 1 | `src/kernels/iq3s_gemv.rs:534` | `fn get_ptx()` — never called directly (superseded by `load_func()` at line 539) |

### Functions defined but never called from production code

These are exported via `deltanet_kernel.rs` or `kernels/mod.rs` but have zero call sites outside their own module:

| Function | File | Status |
|----------|------|--------|
| `gemv_q5k_fused()` | `kernels/q5k_gemv.rs:275` | **DEAD** — takes raw CudaSlice, no callers |
| `gemv_q5k_from_tensor()` | `kernels/q5k_gemv.rs:317` | **DEAD** — no callers |
| `gemv_iq3s_fused()` | `kernels/iq3s_gemv.rs:613` | **DEAD externally** — only called by `gemv_iq3s_fused_at_offset` (same file) |
| `gemv_iq3s_q8_precomputed()` | `kernels/iq3s_gemv.rs:716` | **DEAD** — no callers |
| `raw_argmax()` | `kernels/elementwise.rs:432` | **DEAD** — re-exported but never called |
| `raw_rms_norm()` | `kernels/elementwise.rs:323` | **DEAD** — re-exported but never called |
| `raw_silu_mul()` | `kernels/elementwise.rs:351` | **DEAD** — re-exported but never called |
| `raw_silu_mul_views()` | `kernels/elementwise.rs:378` | **DEAD** — re-exported but never called |
| `raw_weighted_add()` | `kernels/elementwise.rs:405` | **DEAD** — re-exported but never called |
| `dequant_iq3s_gpu()` | `kernels/iq3s_gemv.rs:922` | Thin wrapper around `dequant_iq3s_at_offset` — could be inlined |

**Note**: The `raw_*` elementwise functions in `elementwise.rs` were likely the raw cudarc API intended for the zero-allocation forward pass (`raw_forward.rs`), but that path uses different fused kernels now. All 5 are dead.

### Modules with no external callers (potential dead modules)

| Module | Lines | Used by |
|--------|-------|---------|
| `engram.rs` | 799 | **No callers** — `use crate::engram` appears nowhere. Future feature (Engram codebook). |
| `nvfp4_loader.rs` | 756 | **No callers** — `use crate::nvfp4_loader` appears nowhere. Safetensors NVFP4 loader for a format that was never shipped. |
| `model.rs` | ~325 (non-test) | Used by `config.rs` only for `ChimereModelConfig`. The `ChimereModel` struct and `forward()` are the **old assembly**, superseded by `qwen35_model.rs`. |

### Duplicate GDN layer implementations

`qwen35_model.rs` contains **two complete GDN forward paths**:

1. `forward_gdn_layer_q()` (line 1225) — dense weights path, no MoE, no fused kernels
2. `forward_gdn_layer_moe()` (line 1454) — MoE path with all fused kernel toggles

Both are called (line 1003 vs 1016). The "q" variant duplicates SSM logic without fused kernel support (no `CHIMERE_FUSED_ELEM`, no `CHIMERE_GRAN_PROF`, no `CHIMERE_Q5K_GEMV`). This is a maintenance burden — they will drift.

---

## 2. Feature Toggles Inventory

### Complete environment variable registry

| Variable | Default | Type | Where |
|----------|---------|------|-------|
| **CHIMERE_NO_FUSED** | OFF (fused=ON) | Negative toggle | `qwen35_model.rs:1349,1627` — DeltaNet step: fused CUDA kernel vs Candle reference |
| **CHIMERE_NO_RAW_MOE** | OFF (raw=ON) | Negative toggle | `qwen35_model.rs:1703,2288` — Raw cudarc MoE vs Candle MoE |
| **CHIMERE_NO_FUSED_MOE** | OFF (fused=ON) | Negative toggle | `qwen35_model.rs:1927` — Within raw MoE: fused 1-launch vs 24-launch per-expert |
| **CHIMERE_FUSED_MOE_V2** | OFF | Positive toggle | `raw_forward.rs:608` — Within per-expert MoE: fused gate+up+silu path |
| **CHIMERE_FUSED_ELEM** | OFF | Positive toggle | `qwen35_model.rs:1569` — Fused beta/alpha/gate and rms_norm/silu elementwise |
| **CHIMERE_FUSED_SSM_PROJ** | OFF | Positive toggle | `qwen35_model.rs:1518` — Dual Q5K GEMV for QKV+gate projection |
| **CHIMERE_Q5K_GEMV** | OFF | Positive toggle | `qwen35_model.rs:1510` — Q5K GEMV for individual projections |
| **CHIMERE_DEBUG** | OFF | Positive | `qwen35_model.rs:70,3640,3715` — Debug dump of tensor values |
| **CHIMERE_PROFILE** | OFF | Positive | `qwen35_model.rs:908` — Non-Lazy! Checked every token |
| **CHIMERE_GDN_PROFILE** | OFF | Positive | `qwen35_model.rs:77` — Per-operation GDN profiling |
| **CHIMERE_GRAN_PROF** | OFF | Positive | `qwen35_model.rs:1484,2182` — Ultra-granular profiling with GPU sync |
| **CHIMERE_DISPATCH_PROF** | OFF | Positive | `qwen35_model.rs:83` — MoE dispatch sub-operation profiling |
| **CHIMERE_MOE_PROFILE** | OFF | Positive | `qwen35_model.rs:2013` — Candle MoE timing |
| **CHIMERE_TRACE** | OFF | Positive | `qwen35_model.rs:1841` — Dequant reference comparison trace |
| **CHIMERE_TRACE_LEVEL** | 0 | Integer | `trace.rs:19`, `qwen35_model.rs:2034` — Router and activation tracing |
| **CHIMERE_SKIP_LAYERS** | none | Comma-list | `qwen35_model.rs:927` — Skip specified layer indices |
| **CHIMERE_EARLY_EXIT** | OFF | Positive | `qwen35_model.rs:950` — Early exit when hidden converges |
| **CHIMERE_EXIT_THRESH** | 0.05 | Float | `qwen35_model.rs:956` — Early exit cos_sim threshold |
| **CHIMERE_EXIT_MIN_LAYERS** | 32 | Integer | `qwen35_model.rs:967` — Minimum layers before early exit |
| **CHIMERE_ENGRAM_FILE** | none | Path | `generate.rs:179` — Engram lookup table file |
| **CHIMERE_ENGRAM_ALPHA** | 0.3 | Float | `generate.rs:196` — Engram bias strength |
| **CHIMERE_NO_CUBIN** | OFF | Positive | `kernels/cubin_loader.rs:39` — Skip pre-compiled cubin |
| **CHIMERE_NVRTC** | OFF | Positive | `kernels/cubin_loader.rs:38` — Force NVRTC compilation |
| **CHIMERE_MAXREG** | none | Integer | `kernels/nvrtc_compile.rs:42` — NVRTC max register count |
| **CHIMERE_PORT** | 8090 | Integer | `bin/chimere-server.rs:39` — Server port |
| **CHIMERE_MODEL** | hardcoded path | Path | `bin/chimere-server.rs:44` — Model path |
| **CHIMERE_TOKENIZER** | none | Path | `bin/chimere-server.rs:51` — Tokenizer path |
| **CHIMERE_NAME** | chimere-deltanet | String | `bin/chimere-server.rs:54` — Model name |

**Total: 28 environment variables.** 8 are debug/profiling, 7 are kernel toggles, 5 are config, 8 are operational.

### Critical issue: Contradictory toggle naming

The MoE path has **three overlapping toggles** with inconsistent polarity:

```
CHIMERE_NO_FUSED_MOE=<unset>  --> fused MoE ON  (single-launch kernel)
CHIMERE_NO_FUSED_MOE=1        --> per-expert path, THEN:
  CHIMERE_FUSED_MOE_V2=<unset>  --> v1 path (29 launches)
  CHIMERE_FUSED_MOE_V2=1        --> v2 path (5-6 launches, uses fused_gate_up)
```

In practice, production uses `CHIMERE_NO_FUSED_MOE=<unset>` (the single-launch `fused_moe_iq3s`). The `CHIMERE_FUSED_MOE_V2` toggle is **only reachable when CHIMERE_NO_FUSED_MOE=1**, making it effectively dead in production.

Similarly confusing: `CHIMERE_NO_FUSED` (DeltaNet) vs `CHIMERE_FUSED_ELEM` (elementwise) use opposite polarities.

---

## 3. Redundant Kernel Paths

### IQ3_S GEMV variants

| Function | Used in production? | Purpose |
|----------|-------------------|---------|
| `gemv_iq3s_fused()` | NO | F32 input, standalone. Only called by `_at_offset` wrapper |
| `gemv_iq3s_fused_at_offset()` | NO directly | Thin wrapper calling `gemv_iq3s_fused` |
| `gemv_iq3s_fused_at_offset_q8()` | YES | Q8_1 input, used by `moe_ffn_raw()` per-expert path |
| `gemv_iq3s_q8_precomputed()` | NO | Dead — was likely an intermediate experiment |
| `gemv_iq3s_q8_batched()` | YES | Used by `moe_ffn_raw()` for gate/up batched GEMV |
| `gemv_iq3s_q8_batched_multi_input()` | YES | Used by `moe_ffn_raw()` for down projection |

**Summary**: `gemv_iq3s_fused`, `gemv_iq3s_fused_at_offset`, and `gemv_iq3s_q8_precomputed` are dead. Keep the 3 that are called.

### Q5_K GEMV variants

| Function | Used? |
|----------|-------|
| `gemv_q5k_fused()` | NO — raw CudaSlice API, zero callers |
| `gemv_q5k_from_tensor()` | NO — zero callers |
| `gemv_q5k_q8_from_tensor()` | YES — via `gemv_q5k_or_qmm` closure (gated by `CHIMERE_Q5K_GEMV=1`) |
| `gemv_q5k_q8_dual_from_tensor()` | YES — via `CHIMERE_FUSED_SSM_PROJ=1` |

**Summary**: `gemv_q5k_fused` and `gemv_q5k_from_tensor` are dead. The F32-input Q5K path was superseded by Q8_1 dp4a.

### MoE forward paths

| Path | Launches | Gate | Status |
|------|----------|------|--------|
| `moe_ffn_forward()` | Candle ops | `CHIMERE_NO_RAW_MOE=1` | Fallback |
| `moe_ffn_raw()` v1 | ~29 | `CHIMERE_NO_FUSED_MOE=1` + no V2 | Superseded |
| `moe_ffn_raw()` v2 | ~11 | `CHIMERE_NO_FUSED_MOE=1` + `CHIMERE_FUSED_MOE_V2=1` | Superseded |
| `moe_ffn_fused()` | 1 | Default | **PRODUCTION** |

**v1 and v2 sub-paths inside `moe_ffn_raw()` are both dead in production.** The fused single-launch path is always used.

---

## 4. Consistency Check

### Kernel module names (OnceLock collision check)

All module names are unique -- no collision risk:

| Module Name | File |
|-------------|------|
| `chimere_iq3s_gemv_v23` | `kernels/iq3s_gemv.rs` |
| `chimere_fused_gate_up_v1` | `kernels/fused_gate_up.rs` |
| `chimere_deltanet_v5` | `kernels/deltanet_step.rs` |
| `chimere_q5k_gemv_v2` | `kernels/q5k_gemv.rs` (F32 path) |
| `chimere_q5k_q8_gemv_v2` | `kernels/q5k_gemv.rs` (Q8 path, separate PTX cache) |
| `chimere_bench_trivial_v1` | `kernels/iq3s_gemv.rs` (test only) |

No collisions. Each has its own `OnceLock<String>` cache.

### sm_120 target: CLEAN

All NVRTC compilation paths target `sm_120`. Zero references to `sm_89` or `sm_80` in any `.rs` or `.cu` file. The build.rs cubin also targets `sm_120`. This is correct for RTX 5060 Ti.

### IQ3_S grid table consistency

Three copies of the `iq3s_grid[512]` table exist in CUDA source strings:

1. `kernels/iq3s_gemv.rs` line 55 — `__constant__ unsigned int iq3s_grid[512]`
2. `kernels/fused_moe.rs` line 33 — `__constant__ unsigned int iq3s_grid[512]`
3. `kernels/fused_gate_up.rs` line 58 — `__constant__ unsigned int iq3s_grid_gemv[512]`

A fourth Rust copy exists in `gguf_loader.rs` line 935 (`const IQ3S_GRID: [u32; 512]`).

**Verified**: All three CUDA copies contain **identical data** (512 entries, same values). The `fused_gate_up.rs` copy uses a different symbol name (`iq3s_grid_gemv`) to avoid linker conflicts when multiple kernels are loaded, which is correct.

The Rust copy in `gguf_loader.rs` serves a different purpose (CPU dequantization for tests) but the values are the same.

**Risk**: If llama.cpp upstream ever updates the IQ3_S grid (unlikely but possible), all 4 copies must be updated in sync. Consider generating them from a single source.

### Q8_1 block format consistency

All kernels agree on: **36 bytes per block of 32 elements**.

| File | Format |
|------|--------|
| `raw_forward.rs:32` | `(n_elements / 32) * 36` |
| `kernels/iq3s_gemv.rs:529` | `Q8_1_BLOCK_BYTES: usize = 36` |
| `kernels/fused_gate_up.rs:384` | `Q8_1_BLOCK_BYTES: usize = 36` |
| `kernels/q5k_gemv.rs` | Uses `struct block_q8_1_q5k` (36 bytes implicit) |

Layout: `[f16 d (2B), f16 s (2B), i8 qs[32] (32B)]` = 36 bytes.

**Consistent across all kernels.**

---

## 5. Performance Config Recommendation

### Current state (confusing)

Production requires:
```bash
# These are the DEFAULTS (no env vars needed):
# CHIMERE_NO_FUSED=<unset>      → fused DeltaNet ON
# CHIMERE_NO_RAW_MOE=<unset>    → raw MoE ON
# CHIMERE_NO_FUSED_MOE=<unset>  → fused single-launch MoE ON
```

The user's suggestion to set `CHIMERE_FUSED_MOE_V2=1 CHIMERE_NO_FUSED_MOE=1 CHIMERE_FUSED_ELEM=1` is actually **suboptimal**: it disables the fused single-launch MoE in favor of the V2 multi-launch path. Only `CHIMERE_FUSED_ELEM=1` provides a benefit on top of defaults.

### Recommended simplification

**Immediate (zero-risk)**:
```bash
# Optimal production config — only 1 env var needed:
CHIMERE_FUSED_ELEM=1
```

Everything else is already default-on via negative toggles. The `CHIMERE_FUSED_ELEM` toggle should be flipped to default-on (it is the only one that requires explicit opt-in despite being stable).

**Proposed refactor**: Replace all 7 kernel toggles with a single level system:

```
CHIMERE_KERNEL_LEVEL=0  (all Candle reference — for debugging)
CHIMERE_KERNEL_LEVEL=1  (fused DeltaNet + raw MoE, no elem fusion)
CHIMERE_KERNEL_LEVEL=2  (= level 1 + fused elementwise) ← NEW DEFAULT
CHIMERE_KERNEL_LEVEL=3  (= level 2 + Q5K GEMV + fused SSM proj) ← experimental
```

This replaces `NO_FUSED`, `NO_RAW_MOE`, `NO_FUSED_MOE`, `FUSED_ELEM`, `Q5K_GEMV`, `FUSED_SSM_PROJ`, and `FUSED_MOE_V2` with one integer. Individual toggles can remain as overrides for debugging.

---

## 6. Files to Delete

### Confirmed dead files

| File | Size | Reason |
|------|------|--------|
| `src/deltanet_kernel.rs.destroyed` | ~200 lines | Old destroyed kernel, replaced by `src/kernels/deltanet_step.rs` |
| `kernels/chimere_kernels.cu.backup` | ~1000 lines | Backup of the .cu file. The live version is `chimere_kernels.cu` |
| `src/lib.rs.bak` | ~800 lines | Backup of lib.rs from before modules were added. Missing 13 of 19 current modules |

### Safe to delete after review

| File | Lines | Reason |
|------|-------|--------|
| `src/nvfp4_loader.rs` | 756 | NVFP4 safetensors loader. Zero callers. The project loads from GGUF only. Contains 17 tests that pass but test a format never used in production. |
| `src/model.rs` | ~325 (non-test) | Old ChimereModel assembly using synthetic/pretrained weights. Superseded entirely by `qwen35_model.rs`. Only external ref is `config.rs` importing types. |
| `profile-analysis-2026-03-15.md` | N/A | Should be in `docs/` not project root |
| `validate_logits.py` | N/A | One-off validation script, should be in a `scripts/` directory |

---

## 7. Refactoring Suggestions (prioritized by impact)

### P0 — High impact, low risk

1. **Flip `CHIMERE_FUSED_ELEM` to default-on** — It is stable and provides measurable speedup. One line change in `qwen35_model.rs:1569`:
   ```rust
   static FE: Lazy<bool> = Lazy::new(|| std::env::var("CHIMERE_NO_FUSED_ELEM").is_err());
   ```

2. **Delete the 3 backup/destroyed files** — They add confusion and nothing else. Zero risk.

3. **Remove dead re-exports from `deltanet_kernel.rs`** — Remove `gemv_q5k_fused`, `gemv_q5k_from_tensor`, `raw_argmax`, `raw_rms_norm`, `raw_silu_mul`, `raw_silu_mul_views`, `raw_weighted_add`. Then remove the dead functions from their source modules.

### P1 — Medium impact, medium risk

4. **Unify `forward_gdn_layer_q()` and `forward_gdn_layer_moe()`** — The SSM logic is duplicated across ~250 lines. Extract the shared SSM part into a helper, then branch only at the FFN step. This prevents future drift where a fix in one path is missed in the other.

5. **Fix `CHIMERE_PROFILE` to use Lazy** — At `qwen35_model.rs:908`, this is `std::env::var("CHIMERE_PROFILE").is_ok()` — called every single token. All other toggles use `Lazy<bool>`. This is a micro-optimization but also a consistency fix.

6. **Extract IQ3_S grid table to a shared header** — The 512-entry grid is duplicated 3 times in CUDA source strings. Create a shared CUDA header string that all three kernel modules include.

### P2 — Lower priority

7. **Remove `nvfp4_loader.rs`** — 756 lines of code and 17 tests for a format that is not used. Remove the module from `lib.rs`.

8. **Remove dead GEMV functions** — `gemv_iq3s_fused`, `gemv_iq3s_q8_precomputed`, `gemv_q5k_fused`, `gemv_q5k_from_tensor` — 4 dead public functions totaling ~200 lines.

9. **Consolidate profiling toggles** — 6 separate profiling env vars (`CHIMERE_PROFILE`, `CHIMERE_GDN_PROFILE`, `CHIMERE_GRAN_PROF`, `CHIMERE_DISPATCH_PROF`, `CHIMERE_MOE_PROFILE`, `CHIMERE_TRACE`) could be replaced by `CHIMERE_PROF_LEVEL=0..3`.

10. **Move `model.rs` to `model_legacy.rs`** or delete — The `ChimereModel` struct and its forward pass are dead code. Only the type imports from `expert.rs`/`hybrid_attention.rs`/`moe_router.rs` through `config.rs` are live.

---

## Summary

| Category | Count |
|----------|-------|
| Dead functions (public, zero callers) | 10 |
| Dead modules (zero external use) | 2 (engram.rs, nvfp4_loader.rs) |
| Backup/destroyed files to delete | 3 |
| Environment variables | 28 (7 kernel toggles, 8 debug, 5 config, 8 operational) |
| Duplicate code (GDN forward) | ~250 lines |
| Duplicate data (IQ3S grid) | 3 CUDA copies + 1 Rust copy |
| Consistency issues | 0 (sm_120 clean, Q8_1 consistent, module names unique) |

The codebase is in better shape than expected for ~40 agents. The main technical debt is toggle proliferation and the dual GDN forward paths. The kernel infrastructure (sm_120, grid tables, Q8_1 format) is consistent and correct.
