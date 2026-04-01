# Code Audit -- Chimere Public Repository

**Date**: 2026-04-01
**Auditor**: Claude Opus 4.6 (automated comprehensive audit)
**Scope**: Full repository at `github.com/AIdevsmartdata/chimere` (PUBLIC)
**Codebase**: ~60 Rust source files in `chimere-server/src/`, FFI layer, Docker stack, CI

---

## Executive Summary

Chimere is an impressive inference engine with genuine novel contributions (Engram memory, entropy routing, custom CUDA kernels, MTP speculation). The codebase is well-structured and extensively documented with high-quality module-level doc comments. However, there are several issues that should be addressed for a public repository: hardcoded local paths in `build.rs`, a SearXNG secret key committed in plaintext, numerous `unwrap()` calls in production paths, missing OpenAI API endpoints, and `std::env::set_var` usage that is unsound in multi-threaded contexts.

| Category | Grade | Issues |
|----------|-------|--------|
| Code quality | B | Good structure, but 80+ unwrap()s and 20+ panics in non-test code |
| Documentation | A- | Excellent READMEs and doc comments; minor env var mismatches |
| Consistency | B+ | Docker stack is well-orchestrated; minor naming inconsistencies |
| Git hygiene | B | Good .gitignore; one committed secret, hardcoded local paths |
| Build reproducibility | C+ | Hard-coded sm_120, local paths in build.rs, no CI for source builds |
| API correctness | B- | Core works well; missing standard OpenAI endpoints |
| Error handling | C+ | Too many bare unwrap()s in hot paths; panics in production code |
| Performance | A | Clearly optimized hot paths; zero-alloc staging buffers; CUDA graphs |
| Rust best practices | B | Good architecture; unsafe Send/Sync impls need justification |

---

## 1. Code Quality

### 1.1 Dead Code and `#[allow(dead_code)]`

10 instances of `#[allow(dead_code)]` in production code:

- `chimere-server/src/qwen35_model/mod.rs` (lines 175, 233, 265, 314) -- `MoeFFN`, `GdnLayerMoE`, `AttnLayerMoE` structs marked "unused until MoE forward pass is implemented" but the MoE forward IS implemented (in `moe_ffn.rs`). These suppression attributes appear to be stale and should be audited -- either the Candle-path MoE structs are genuinely dead (remove them) or the annotations are wrong (remove the `#[allow(dead_code)]`).
- `chimere-server/src/llama_backend.rs` (line 226) -- suppressed dead code in the FFI types.
- `chimere-server/src/rope.rs` (lines 240, 246) -- test-only helpers.
- `chimere-server/src/debug_utils.rs` (line 18) -- `log_vram()` marked dead but is a debug utility.
- `chimere-server/src/kernels/iq3s_gemv.rs` (line 1120) -- dead code in kernel module.
- `chimere-server/src/kernels/iq3s_gemv_v3.rs` (line 913) -- unused imports.

**Recommendation**: Audit each `#[allow(dead_code)]`. Remove genuinely dead structs (especially `GdnLayerMoE`/`AttnLayerMoE` if superseded by the raw-weights path). For debug utilities, keep `#[allow(dead_code)]` but add a comment explaining they are intentionally available.

### 1.2 TODO/FIXME

Only 1 TODO found in production code:

- `compute_graph.rs:2182`: `// TODO: need gemv_q8_0 or equivalent kernel for lm_head.`

This is a legitimate tracked item. No stray FIXMEs or HACKs were found.

### 1.3 Commented-Out Code Blocks

No significant commented-out code blocks were found. The codebase is clean in this regard.

### 1.4 Imports

Server module (`server.rs`) imports are clean -- all used. No unused imports detected in the main source files.

---

## 2. Documentation

### 2.1 README.md Accuracy

**Good**:
- Architecture diagram is accurate and reflects actual service topology.
- Benchmark numbers match the system described in memory (80 tok/s RTX 5060 Ti).
- HuggingFace links and model names are consistent.
- Docker quick-start instructions are complete.
- Env var documentation is comprehensive and mostly accurate.

**Issues**:

| Issue | Location | Details |
|-------|----------|---------|
| Port mismatch in code vs docs | README says `CHIMERE_PORT` default is 8090, Docker overrides to 8081 | The code default (8090) and Docker default (8081) differ, which is fine, but the README env table says "Default: 8090" while the Docker section shows 8081. Should clarify "8090 (standalone) / 8081 (Docker)". |
| Missing env vars in README | Multiple undocumented vars | `CHIMERE_CUDARC_FORWARD`, `CHIMERE_NAME`, `CHIMERE_FLASH_PREFILL`, `CHIMERE_FLASH_DEBUG`, `CHIMERE_KV_RING`, `CHIMERE_VRAM_LOG`, `CHIMERE_DEBUG`, `CHIMERE_GDN_PROFILE`, `CHIMERE_DISPATCH_PROF`, `CHIMERE_ACT_DTYPE`, `CHIMERE_L0_DUMP`, `CHIMERE_ENGRAM_FILE`, `CHIMERE_ENGRAM_ALPHA`, `CHIMERE_ENGRAM_DIR`, `CHIMERE_ENGRAM_DART`, `CHIMERE_DART_STEPS`, `CHIMERE_THINKING_ACTIVE`, `CHIMERE_NVRTC`, `CHIMERE_THREADS_BATCH`, `CHIMERE_FLASH_PREFILL` are used in code but not in the README. Debug vars are fine to omit, but `CHIMERE_CUDARC_FORWARD` and `CHIMERE_NAME` should be documented. |
| Stale tokenizer default path | `chimere-server.rs` line 22 | Default tokenizer path references `qwopus-27b-bf16/tokenizer.json` (an old model). The README correctly says "auto-detect" but the code hardcodes a fallback to an obsolete path. |
| 6 services vs 7 | README lists 6, docker-compose has 7 | The `scorer` service exists in docker-compose.yml but is not listed in the README Docker Stack table. |
| Referenced scripts missing | `docker/README.md` line 36 | References `scripts/detect-gpu.py` and `scripts/download-model.sh` -- these DO exist, so this is fine. But `docker/README.md` references `docker/.env` which is gitignored and won't exist on clone. Should mention running detect-gpu.py first. |

### 2.2 Docker README

The `docker/README.md` is thorough and well-written. The env var table there uses different var names than the main code (`CHIMERE_KV_K` vs `CHIMERE_KV_TYPE_K`, `CHIMERE_NGL` not found in code at all, `CHIMERE_NP` not found in code). These appear to be env vars consumed by an `entrypoint.sh` that translates them for llama-server, not by chimere-server directly. This should be clarified.

---

## 3. Consistency

### 3.1 docker-compose.yml vs Dockerfile

**Mostly consistent**. Both use:
- CUDA 12.8 base images
- Port 8081 for inference
- Same env var names

**Issues**:

| Issue | Details |
|-------|---------|
| `scorer` service reuses inference Dockerfile | `scorer` builds from `docker/inference/Dockerfile` but runs on CPU (`CHIMERE_NCMOE=0`). The Dockerfile hard-requires CUDA (nvidia/cuda base). A scorer running on CPU should use a lighter base image or the inference image should gracefully handle CPU-only. |
| `nightly` build context points outside repo | `context: ../../chimere-odo` -- requires chimere-odo repo to be checked out as a sibling. This is not documented in the main README quick-start. Running `docker compose up` will fail unless chimere-odo is also cloned. |
| ODO build context also outside repo | `context: ../../chimere-odo` -- same issue as nightly. |
| Service name inconsistency | docker-compose uses `chimere` (service) but docker/README says `inference` (container name). |

### 3.2 Package Name vs Binary Name

The Cargo.toml package name is `chimere-deltanet` but the binary is `chimere-server` and the lib is imported as `chimere_deltanet`. This is confusing for external users. Consider renaming the package to `chimere-server` or at minimum documenting why the name differs.

---

## 4. Git Hygiene

### 4.1 .gitignore

**Good coverage**:
- `target/`, `*.gguf`, `*.bin`, `*.pt`, `*.safetensors`, `*.engr` -- model files excluded.
- `.env`, `.env.*`, `credentials/` -- secrets excluded.
- `__pycache__/`, `.vscode/`, `.idea/`, `.claude/` -- IDE/temp files excluded.
- `libs/` excluded (used by Dockerfile.prebuilt for local .so files).

**Missing**:
- `Cargo.lock` for the library crate is committed (both `chimere-server/Cargo.lock` and `chimere-server/ffi/Cargo.lock`). For a binary, this is correct practice. For the library sub-crate (`ffi/`), it's debatable but acceptable.
- No explicit exclusion for `*.cubin` files (but they go into `target/` which is already excluded).
- The `chimere-server/ffi/target/` directory was not found in the glob results as committed files, so the gitignore is working correctly.

### 4.2 Sensitive Data

| Severity | File | Issue |
|----------|------|-------|
| **MEDIUM** | `docker/config/searxng/settings.yml:27` | `secret_key: "chimere-searxng-change-me-in-production"` -- While the comment says "change me," this is still a default secret committed to a public repo. SearXNG uses this for CSRF protection. Should use an env var or generate on first run. |
| **LOW** | `docker/docker-compose.yml:162` | `OPENAI_API_KEYS: chimere-local` -- This is a placeholder key for local-only auth, not a real secret. But it looks like one and may trigger automated secret scanners. |
| **CLEAN** | `.env` files | Not committed. `.gitignore` correctly excludes them. |
| **CLEAN** | API keys | No real API keys, passwords, or private keys found in the repository. |

### 4.3 Large Files

No files over 1 MB were found in the working tree. Model files (GGUF, PT, etc.) are properly gitignored. The repository does not appear to have large binary blobs in history (no `.gguf` or `.bin` files tracked).

---

## 5. Build Reproducibility

### 5.1 Critical Issues

| Severity | Issue | Details |
|----------|-------|---------|
| **HIGH** | Hardcoded local path in `build.rs` | Lines 80, 90-91: `"{IKLLAMACPP_DIR}/build_sm120/..."` -- These are literal string paths with curly braces, NOT env var expansions. `std::path::Path::new("{IKLLAMACPP_DIR}/build_sm120/ggml/src/libggml.so").exists()` will ALWAYS return false on any machine (no such directory). These checks are dead code. The actual libllama linkage works only in the Docker build where `GGML_SO_DIR`/`LLAMA_SO_DIR` are set via the Dockerfile. |
| **HIGH** | Hardcoded `sm_120` in `build.rs` | Line 37: `--gpu-architecture=sm_120` is hard-coded. The `.cargo/config.toml` also sets `CUDA_COMPUTE_CAP = "120"`. This means the build will fail or produce non-functional binaries on any GPU except Blackwell (sm_120). The Dockerfile correctly handles this with `CMAKE_CUDA_ARCHITECTURES="89;120"` for ik_llama, but chimere's own CUDA kernels target only sm_120. |
| **MEDIUM** | No env var for SM architecture | The `build.rs` should read `CUDA_COMPUTE_CAP` or `CMAKE_CUDA_ARCHITECTURES` from the environment instead of hardcoding sm_120. |
| **MEDIUM** | Candle git dependency | `Cargo.toml` line 8: `candle-core = { git = "https://github.com/huggingface/candle.git" }` -- No branch, tag, or rev pinned. This means builds are not reproducible; `cargo build` on different days may pull different Candle versions. Pin a specific rev or tag. |
| **LOW** | Missing `scripts/` from Docker context | The Docker quick-start says to run `detect-gpu.py` but this script is in `scripts/`, not in the Docker context. Works fine for host-side use but not if someone tries to run it inside the container. |

### 5.2 Docker Build

The Dockerfile is well-structured (multi-stage, non-root user, healthcheck). The build would succeed if the self-hosted GPU runner has CUDA 12.8. However:

- `RUN git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git` -- Clones HEAD without a tag. If ik_llama makes a breaking change, the Docker build breaks. Pin a commit hash.
- The build uses `-j2` which is conservative but safe for CI with limited RAM.
- The runtime image correctly copies only `libllama.so` and `libggml.so` (minimal footprint).

---

## 6. API Correctness (OpenAI Compatibility)

### 6.1 Implemented

- `/v1/chat/completions` (POST) -- streaming and non-streaming
- `/health` (GET)
- `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`, `logprobs`, `top_logprobs`, `model`, `tools`, `user`, `presence_penalty`, `repetition_penalty`
- `reasoning_content` in both message and delta (Qwen3.5/DeepSeek extension)
- Tool calling with Qwen3.5 XML format parsed to OpenAI JSON format
- Multimodal content array deserialization (text parts concatenated)
- SSE streaming with proper `[DONE]` termination

### 6.2 Missing OpenAI Endpoints

| Endpoint | Status | Impact |
|----------|--------|--------|
| `GET /v1/models` | **Missing** | Many OpenAI-compatible clients call this on startup. Without it, clients like Open WebUI, LiteLLM, or Cursor may fail to detect the model. Easy to add (static JSON response). |
| `GET /v1/chat/completions` | Missing | Some clients send GET for capability probing. Low impact. |
| `POST /v1/completions` | Missing | Legacy completions endpoint. Low impact for chat models. |

### 6.3 Missing Request Fields

| Field | Status | Impact |
|-------|--------|--------|
| `frequency_penalty` | **Missing** | Standard OpenAI field. Clients may send it; currently silently ignored (serde default). Should at minimum accept and apply or document as unsupported. |
| `stop` | **Missing** | Stop sequences. Standard field. Clients like Aider send custom stop tokens. |
| `n` | **Missing** | Number of completions. Low impact (single-completion is fine). |
| `response_format` | **Missing** | JSON mode. Growing in importance for tool-use clients. |
| `seed` | **Missing** | Reproducibility. Low priority but nice-to-have. |
| `stream_options` | **Missing** | `include_usage` option for streaming. Clients like litellm use this. |

### 6.4 Response Issues

- `usage` is returned in non-streaming but NOT in streaming (OpenAI returns usage in the final chunk when `stream_options.include_usage` is true).
- `system_fingerprint` field is missing (optional but returned by OpenAI).
- The `id` field uses timestamp-based hex (`chatcmpl-{nanos:x}`) which may collide under very fast sequential requests. Consider adding a random component.

---

## 7. Error Handling

### 7.1 Bare `unwrap()` in Production Code

**80+ instances** of `.unwrap()` across the codebase. Most are in test code or paths where failure indicates a bug, but several are in production hot paths:

**Critical (production server paths)**:

| File | Line | Code | Risk |
|------|------|------|------|
| `server.rs` | 289 | `tools.unwrap()` | Panics if `tools` is None when `has_tools` was computed from `.as_ref()`. Actually safe due to the guard above, but the pattern is fragile. Use `if let Some(tools) = &tools` instead. |
| `server.rs` | 340 | `v.as_str().unwrap()` | Panics if value is not a string despite the `is_string()` guard. Safe but brittle. |
| `server.rs` | 661 | `.next().unwrap()` | Panics if packed logprobs produce an empty iterator. |
| `server.rs` | 691 | `serde_json::to_value(resp).unwrap()` | Panics if ChatResponse cannot be serialized. Theoretically impossible but should use `unwrap_or_else`. |
| `server.rs` | 851 | `flatten_all().unwrap().to_vec1::<f32>().unwrap()[0]` | Double unwrap in the streaming path's prefill logit extraction. |
| `moe_ffn.rs` | 462-466 | 5 consecutive `.unwrap()` calls | Panics if MoE shared expert weights are None. These should be validated at model load time, not at inference time. |
| `mtp_scheduler.rs` | 1289-1323 | Multiple `.unwrap()` chains | Panics in MTP verification on tensor operations. |

**Recommendation**: Replace bare `.unwrap()` in server paths with `.expect("descriptive message")` or proper error propagation via `?`. The model initialization path already uses `expect()` well; apply the same pattern to the inference hot path.

### 7.2 Panics in Production Code

**23 `panic!()` calls** found outside test modules:

- **4 in `ggml_gpu.rs`** (lines 1709, 1712, 1862, 1865, 2007, 2010): `panic!("BUG: ggml produces NaN/zeros")` -- These are integrity checks in kernel output validation. Acceptable as they indicate data corruption, but should use `eprintln!` + graceful error return instead of crashing the server.
- **4 in `layer_gdn.rs`, `layer_attn.rs`, `moe_ffn.rs`**: `panic!("cubin fallback disabled")` -- These fire when a required feature is missing. Should return `Err(...)` instead of panicking.
- **2 in `mod.rs` line 3302-3305**: `panic!("")` with empty message -- These are particularly bad. An empty panic message gives zero diagnostic information.
- **1 in `compute_graph.rs` line 2256**: `panic!("EMBEDDING MISMATCH!")` -- Test-only, acceptable.
- **1 in `scratch_pool.rs` line 370**: `panic!("not CUDA")` -- Should return an error.
- **1 in `nvrtc_compile.rs` line 68**: Panic on NVRTC compilation failure -- Should return an error.

### 7.3 `std::env::set_var` in Multi-Threaded Context

`server.rs` lines 865-867:
```rust
std::env::set_var("CHIMERE_THINKING_ACTIVE", "1");
std::env::remove_var("CHIMERE_THINKING_ACTIVE");
```

**This is unsound.** `std::env::set_var` is marked `unsafe` since Rust 1.66 (when used without `unsafe` block, it compiles but with a deprecation warning; in future Rust editions it will be a hard error). In a multi-threaded server, concurrent calls to `set_var`/`remove_var` are a data race. The comment says "thread-local safe because we're on a dedicated OS thread per request" but `set_var` modifies the PROCESS-WIDE environment, not thread-local storage.

**Recommendation**: Replace with a thread-local variable (`thread_local!`) or pass the flag through the function signature.

Similarly, `generate.rs` lines 501-526 use `set_var`/`remove_var` in test code, which is acceptable.

---

## 8. Performance

### 8.1 Strengths

The codebase shows deep performance awareness:

- **Zero-allocation staging buffers** (`NcmoeBufs`): Pre-allocated CPU/GPU staging for MoE expert batch copies. Single `memcpy_htod` instead of per-expert copies.
- **CUDA graph capture** for repeated inference patterns (`cuda_graph.rs`).
- **Pre-compiled cubins** via `build.rs` (eliminates 200ms NVRTC startup).
- **Ring buffer KV cache** (`CHIMERE_KV_RING=1`) for O(1) appends vs O(P) `cat()`.
- **Streaming architecture**: Dedicated OS thread for inference with mpsc channel to SSE -- avoids blocking the tokio runtime.
- **Quantized GEMV kernels**: Custom IQ3_S, Q5_K, Q8_0 kernels for the specific tensor shapes in Qwen3.5.

### 8.2 Potential Issues

| Issue | Location | Impact |
|-------|----------|--------|
| String cloning in SSE loop | `server.rs` lines 948-1037 | `st.req_id.clone()` and `st.model_name.clone()` on every SSE event (~80/s). These are small strings so impact is minimal, but could use `Arc<str>` to avoid clones. |
| `serde_json::to_string` per SSE event | `server.rs` lines 963, 988, etc. | Allocates a new String per token. At 80 tok/s this is 80 allocations/s. Consider a reusable buffer or `serde_json::to_writer`. |
| `prompt.push_str(&format!(...))` | `server.rs` line 327 | Multiple format+push_str in message building. Could use `write!` directly on the String to avoid intermediate allocations. Minor. |
| Tokenizer encode for prompt_tokens count | `server.rs` line 551-555 | Encodes the full prompt just to count tokens, then the inference path encodes it again. The prompt_tokens count could be derived from the inference path to avoid double encoding. |

---

## 9. Rust Best Practices

### 9.1 `unsafe impl Send/Sync`

| File | Lines | Type | Justification |
|------|-------|------|---------------|
| `compute_graph.rs` | 1108-1109 | `CachedWeightPtrs` | No safety comment. Contains raw GPU pointers. Should document why this is safe (e.g., "pointers are valid for the lifetime of the model, access is serialized by Mutex"). |
| `cuda_graph.rs` | 98-101 | `RawCudaGraph`, `GdnGraphCache` | No safety comment. Contains raw CUDA driver handles. |
| `llama_backend.rs` | 359 | `LlamaForward` | No safety comment. Contains raw C pointers to llama model/context. |

**Recommendation**: Add `// SAFETY:` comments above each `unsafe impl` explaining why the invariants hold. This is standard Rust practice and helps reviewers (and future contributors) understand the threading model.

### 9.2 Error Types

The codebase uses `String` as the error type almost everywhere:
```rust
fn generate_text(...) -> Result<GenerateResult, String>
```

This is functional but not idiomatic. A proper error enum with `thiserror` (already in `Cargo.lock`) would enable:
- Pattern matching on error variants
- Better error messages for users
- `?` propagation without `.map_err(|e| format!(...))` boilerplate

This is a low-priority refactor but would significantly improve code quality for contributors.

### 9.3 `RefCell` in `Qwen35Model`

The model uses `RefCell<Option<Tensor>>` which makes it `!Sync`. The server wraps it in `tokio::sync::Mutex` which is correct. The comment at `server.rs:254` correctly explains this. However, `RefCell` in a GPU inference engine suggests interior mutability for caching -- consider whether `Cell` or atomic operations could replace it to simplify the threading story.

---

## 10. Additional Findings

### 10.1 Hardcoded Paths

| File | Line | Path | Issue |
|------|------|------|-------|
| `build.rs` | 24 | `/usr/local/cuda-12.8/bin/nvcc` | Hard-codes CUDA 12.8 path. Should prefer `CUDA_HOME` env var first. |
| `build.rs` | 37 | `--gpu-architecture=sm_120` | Hard-codes Blackwell. Should read from env. |
| `build.rs` | 80, 90-91 | `{IKLLAMACPP_DIR}/build_sm120/...` | Literal curly braces -- not an env var expansion. Dead code. |
| `generate.rs` | 25 | `.chimere/models/qwopus-27b-bf16/tokenizer.json` | References an old model name. |
| `chimere-server.rs` | 22 | Same stale tokenizer path | In the doc comment. |

### 10.2 Cargo.toml Issues

- Package name `chimere-deltanet` does not match the binary name `chimere-server` or the repository name `chimere`. This causes confusion when importing the library (`use chimere_deltanet::...`).
- Git dependency on `candle` without a pinned rev means builds are not reproducible.
- `rand = "0.9"` -- Consider pinning more tightly for reproducibility.

### 10.3 CI/CD

The GitHub Actions workflow only builds the Docker image -- there is no CI for:
- `cargo test`
- `cargo clippy`
- `cargo fmt --check`

Adding these would catch many of the issues identified in this audit automatically.

---

## Prioritized Recommendations

### Must Fix (before promoting publicly)

1. **Remove/template the SearXNG secret_key** -- Use `${SEARXNG_SECRET_KEY:-$(openssl rand -hex 32)}` or an entrypoint script.
2. **Fix `std::env::set_var` unsoundness** in `server.rs:865-867` -- Replace with thread-local or function parameter.
3. **Add `/v1/models` endpoint** -- Many clients require it. 10 lines of code.
4. **Fix empty `panic!("")`** in `mod.rs:3302-3305` -- At minimum add a message.
5. **Pin Candle git dependency** to a specific rev in `Cargo.toml`.

### Should Fix (quality)

6. **Make `build.rs` SM architecture configurable** via env var (fall back to sm_120).
7. **Remove dead `{IKLLAMACPP_DIR}` paths** from `build.rs` (lines 80, 90-91).
8. **Replace bare `unwrap()`** in server hot paths with `expect()` or `?`.
9. **Convert `panic!()`** in production code to `Err(...)` returns.
10. **Add `// SAFETY:` comments** to all `unsafe impl Send/Sync`.
11. **Document the scorer service** in the README Docker table.
12. **Document chimere-odo dependency** -- docker-compose will fail without it.
13. **Add `stop` and `frequency_penalty`** to ChatRequest for OpenAI compatibility.
14. **Update stale tokenizer default path** (`qwopus-27b-bf16` -> current model name).

### Nice to Have (polish)

15. Add `cargo test` and `cargo clippy` to GitHub Actions CI.
16. Rename package from `chimere-deltanet` to `chimere-server` for clarity.
17. Define a proper error enum with `thiserror`.
18. Audit and clean up `#[allow(dead_code)]` annotations.
19. Pin `ik_llama.cpp` clone to a specific commit in the Dockerfile.
20. Add `stream_options.include_usage` support for streaming responses.

---

## Files Reviewed

### Core Source (chimere-server/src/)
- `server.rs` (1091 lines) -- HTTP server, OpenAI API, SSE streaming
- `bin/chimere-server.rs` (213 lines) -- Binary entrypoint
- `lib.rs` (~100 lines) -- Module declarations, NormLayer, activations
- `config.rs` (~300 lines) -- Model configuration, Qwen35Config
- `generate.rs` (~500 lines) -- Text generation pipeline, tokenizer loading
- `state.rs` (large) -- GDN recurrent state, NcmoeBufs, KV ring buffer
- `agent_scheduler.rs` -- Multi-agent context switching
- `llama_backend.rs` -- libllama FFI bindings
- `engram.rs` -- Poincare hyperbolic memory codebook
- `entropy_router.rs` -- Entropy-adaptive decode strategy
- `block_diffusion.rs` -- Block diffusion scheduler
- `turboquant.rs` -- TurboQuant KV cache compression
- `mtp_scheduler.rs` -- Multi-token prediction
- `debug_utils.rs` -- Debug/profiling utilities
- `qwen35_model/mod.rs` -- Full Qwen3.5 model implementation
- `qwen35_model/compute_graph.rs` -- Compute graph with CUDA graphs
- `qwen35_model/moe_ffn.rs` -- MoE forward pass
- `qwen35_model/lm_head.rs` -- LM head + MTP
- All kernel modules in `kernels/`

### Infrastructure
- `Cargo.toml`, `build.rs`, `.cargo/config.toml`
- `docker/docker-compose.yml`
- `docker/inference/Dockerfile`, `docker/inference/Dockerfile.prebuilt`
- `docker/nightly/Dockerfile`
- `docker/config/searxng/settings.yml`
- `.github/workflows/docker-build.yml`
- `.gitignore`
- `README.md`, `docker/README.md`
- `scripts/detect-gpu.py`, `scripts/download-model.sh`
