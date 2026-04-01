# Security Audit -- Chimere Project

**Date**: 2026-04-01
**Auditor**: Claude Opus 4.6 (automated)
**Scope**: Full repository at `github-repos/chimere/` (public repo)
**Commit**: HEAD as of 2026-04-01

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH     | 3 |
| MEDIUM   | 8 |
| LOW      | 6 |

No hardcoded API keys, passwords, or private keys were found in the repository.
The `.gitignore` correctly excludes `.env`, `.env.*`, and `credentials/`.
No `.env` files are committed. The codebase is generally well-structured for
a local inference engine, but has several issues that should be addressed
before broader public use.

---

## Findings

### HIGH-1: SearXNG `secret_key` hardcoded in tracked config

**File**: `docker/config/searxng/settings.yml:27`
**Severity**: HIGH

```yaml
secret_key: "chimere-searxng-change-me-in-production"
```

The SearXNG secret key is committed to the repo with a placeholder value.
While the comment says "change me", users who deploy without changing it
share the same secret, enabling session forgery on their SearXNG instance.

**Fix**: Remove the hardcoded key. Generate it at container startup:
```yaml
secret_key: "${SEARXNG_SECRET_KEY}"
```
Add to `entrypoint.sh`:
```bash
export SEARXNG_SECRET_KEY="${SEARXNG_SECRET_KEY:-$(python3 -c 'import secrets; print(secrets.token_hex(32))')}"
```

---

### HIGH-2: Docker containers run as root

**Files**: `docker/inference/Dockerfile`, `docker/nightly/Dockerfile`
**Severity**: HIGH

Neither Dockerfile creates a non-root user. The inference container runs
`chimere-server` as root, and the nightly container runs Python scripts
as root. If an attacker exploits a vulnerability in the server or a
dependency, they gain root inside the container.

**Fix**: Add a non-root user in each Dockerfile:
```dockerfile
RUN useradd -r -s /bin/false chimere
USER chimere
```
For inference, ensure the `/models` volume is readable by the chimere user.
For nightly, ensure `/data/logs` and `/data/engram` are writable.

---

### HIGH-3: No `max_tokens` upper bound -- DoS via resource exhaustion

**File**: `chimere-server/src/server.rs:53-54`
**Severity**: HIGH

```rust
#[serde(default = "default_max_tokens")]
pub max_tokens: usize,
```

The `max_tokens` field from the request is a `usize` with no upper bound
enforcement. A client can send `max_tokens: 18446744073709551615` (usize::MAX),
causing the server to attempt generating tokens indefinitely. Combined with
the single-slot mutex design, this blocks all other requests.

The default is 2048, but no clamp is applied to user-supplied values.

**Fix**: Clamp `max_tokens` to a configurable maximum (e.g., 32768):
```rust
let total_budget = req.max_tokens.min(MAX_GENERATION_TOKENS);
```
Add `MAX_GENERATION_TOKENS` as a constant or env var (e.g., `CHIMERE_MAX_TOKENS`).

---

### MEDIUM-1: `engram_table` field allows arbitrary file path (path traversal)

**File**: `chimere-server/src/server.rs:84`
**Severity**: MEDIUM

```rust
pub engram_table: Option<String>,
```

The `ChatRequest` accepts an `engram_table` path override. While this field
is currently **unused** in the handler code (the actual Engram loading uses
`CHIMERE_ENGRAM_DIR` / `CHIMERE_ENGRAM_FILE` env vars), if it were connected,
a client could pass `"engram_table": "/etc/passwd"` or similar paths.

**Fix**: Either remove the unused field, or if it will be used, validate the
path against an allowed directory:
```rust
if let Some(ref table) = req.engram_table {
    let canonical = std::fs::canonicalize(table)?;
    if !canonical.starts_with("/data/engram") {
        return Err("engram_table path outside allowed directory".into());
    }
}
```

---

### MEDIUM-2: `CHIMERE_MODEL` env var controls file path without validation

**Files**: `chimere-server/src/bin/chimere-server.rs:51`, `chimere-server/src/llama_backend.rs:1007`
**Severity**: MEDIUM

The model path is read from `CHIMERE_MODEL` and passed directly to
`llama_model_load_from_file()` and `Qwen35Model::from_gguf()`. No
canonicalization or directory restriction is applied. An attacker who
can set environment variables could point this at arbitrary files.

In a Docker deployment the attack surface is limited (env vars are set
at compose time), but for bare-metal deployments this is a concern.

**Fix**: Validate the path exists and optionally restrict to an allowed
directory via `CHIMERE_MODEL_DIR`.

---

### MEDIUM-3: Server binds to `0.0.0.0` with no authentication

**File**: `chimere-server/src/bin/chimere-server.rs:47`
**Severity**: MEDIUM

```rust
let addr = format!("0.0.0.0:{}", port);
```

The server binds to all interfaces with no API key, bearer token, or
any authentication mechanism. Anyone on the network can send inference
requests, potentially exhausting GPU resources.

The Docker compose exposes port 8081 to the host as well.

**Fix**: Add optional API key authentication via `CHIMERE_API_KEY` env var:
```rust
if let Some(expected) = api_key {
    if req_key != expected { return 401 Unauthorized; }
}
```
Or bind to `127.0.0.1` by default and require an explicit `CHIMERE_BIND`
env var to expose externally.

---

### MEDIUM-4: Open WebUI `WEBUI_AUTH: "false"` disables authentication

**File**: `docker/docker-compose.yml:109`
**Severity**: MEDIUM

```yaml
WEBUI_AUTH: "false"
ENABLE_SIGNUP: "false"
```

Open WebUI authentication is disabled. Anyone who can reach port 3000 has
full access to the chat interface and can issue inference requests.

**Fix**: Enable authentication for non-localhost deployments. The current
setup is acceptable for a Tailscale-only deployment (as documented) but
should be clearly warned:
```yaml
# WARNING: Auth disabled -- only safe behind Tailscale/VPN
WEBUI_AUTH: "false"
```

---

### MEDIUM-5: Alignment assumptions in `gguf_loader.rs` unsafe pointer casts

**File**: `chimere-server/src/gguf_loader.rs:729,744,785`
**Severity**: MEDIUM

```rust
let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_elements) };
let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, n_elements) };
```

These casts assume the underlying `&[u8]` data is properly aligned for `f32`
(4-byte) and `u16` (2-byte) access. The data comes from a memory-mapped
GGUF file. While GGUF files typically have 32-byte alignment for tensor data,
the code does not verify alignment before casting.

Misaligned reads cause undefined behavior on some architectures and may
silently produce wrong results on x86.

**Fix**: Assert alignment before casting:
```rust
assert!(data.as_ptr() as usize % std::mem::align_of::<f32>() == 0,
        "F32 data is not 4-byte aligned");
```

---

### MEDIUM-6: `unsafe impl Send for LlamaForward` relies on external invariant

**File**: `chimere-server/src/llama_backend.rs:359`
**Severity**: MEDIUM

```rust
unsafe impl Send for LlamaForward {}
```

`LlamaForward` contains raw pointers (`*mut LlamaModel`, `*mut LlamaContext`,
`*mut c_void` for the sampler). The `Send` impl is correct only if single-
threaded access is enforced by the tokio Mutex in `server.rs`. The safety
comment documents this, but there is no compile-time enforcement.

Similarly: `gguf_loader.rs:326-327` (`unsafe impl Send/Sync for GgufFile`),
`cuda_graph.rs:98-101` (`unsafe impl Send/Sync for RawCudaGraph, GdnGraphCache`),
`ggml_gpu.rs:335-336` (`unsafe impl Send/Sync for SendPtr`),
`compute_graph.rs:1108-1109` (`unsafe impl Send/Sync for CachedWeightPtrs`).

These are all defensible for the current architecture but require careful
attention if the concurrency model changes.

**Fix**: Document the invariant on the struct itself:
```rust
/// SAFETY: Single-threaded access enforced by `tokio::sync::Mutex<LlamaForward>`
/// in `server.rs::AppState`. Do NOT use without a mutex.
unsafe impl Send for LlamaForward {}
```

---

### MEDIUM-7: `ffi/lib.rs` unsafe pointer cast without alignment check

**File**: `chimere-server/ffi/src/lib.rs:171`
**Severity**: MEDIUM

```rust
let w_f32: &[f32] = unsafe {
    std::slice::from_raw_parts(
        w_data.as_ptr() as *const f32,
        nrows * ncols,
    )
};
```

Same alignment issue as MEDIUM-5. The `w_data` slice is cast to `*const f32`
without verifying 4-byte alignment.

**Fix**: Add an alignment assertion before the cast.

---

### MEDIUM-8: Error messages leak internal paths

**File**: `chimere-server/src/llama_backend.rs:431`, `chimere-server/src/server.rs:692-696`
**Severity**: MEDIUM

```rust
return Err(format!("llama_model_load_from_file failed for {}", model_path));
```

```rust
Err(e) => {
    let body = serde_json::json!({
        "error": { "message": e, "type": "server_error" }
    });
```

Internal error messages (including file system paths and error details)
are returned directly to the client in JSON error responses. This leaks
information about the server's internal file structure.

**Fix**: Return generic error messages to clients and log details server-side:
```rust
eprintln!("[ERROR] Inference failed: {}", e);
let body = json!({"error": {"message": "Internal inference error", "type": "server_error"}});
```

---

### LOW-1: `ARCHITECTURE.md` states "CPU-only" and "Not yet connected to real model weights"

**File**: `chimere-server/ARCHITECTURE.md:9-10`
**Severity**: LOW (documentation)

```
Status: Executable specification. 53 tests passing, CPU-only.
Not yet connected to real model weights or CUDA.
```

This is outdated. The codebase has extensive CUDA support, FFI to libllama,
and runs on real model weights at 93 tok/s.

**Fix**: Update the status line to reflect the current state.

---

### LOW-2: CUDA kernel launches are `unsafe` but well-contained

**Files**: Multiple files in `chimere-server/src/kernels/*.rs`
**Severity**: LOW

There are ~100 `unsafe { builder.launch(cfg) }` calls for CUDA kernel
launches via cudarc. These are inherently unsafe (GPU code execution)
but follow a consistent pattern: parameters are validated before launch,
grid/block dimensions are computed from tensor shapes, and results are
checked. No obvious buffer overflow or out-of-bounds issues.

**Status**: Acceptable. No action needed.

---

### LOW-3: `Cargo.lock` not auditable (cargo-audit not available)

**File**: `chimere-server/Cargo.lock`
**Severity**: LOW

Could not run `cargo audit` to check for known vulnerable crate versions.
Manual inspection shows:
- `axum 0.8.8` (latest stable, no known CVEs)
- `tokenizers 0.21.4` / `0.22.2` (two versions in lock file -- potential confusion)
- `cudarc 0.17.8` and `0.19.4` (two major versions -- dependency conflict risk)
- `candle-core 0.10.1` (git dependency, not pinned to a specific commit)

**Fix**: Run `cargo audit` in CI. Pin git dependencies to specific commits.
Resolve duplicate versions where possible.

---

### LOW-4: `Mmap::map` is unsafe but correctly used

**Files**: `chimere-server/src/gguf_loader.rs:334`, `chimere-server/src/engram_lookup.rs:337`
**Severity**: LOW

Memory-mapped files are inherently unsafe (external modification can cause
UB). The code uses them correctly for read-only access to model weights
and engram tables. The files are not expected to change during runtime.

**Status**: Acceptable for the use case. Document that model files must
not be modified while the server is running.

---

### LOW-5: `pos` counter overflow in `LlamaForward`

**File**: `chimere-server/src/llama_backend.rs:565,595,622`
**Severity**: LOW

```rust
self.pos += 1;            // i32
self.pos += toks.len() as i32;
```

The position counter is `i32`, which overflows at ~2.1 billion tokens.
In practice this is unreachable (context is 32K-65K), but the `as i32`
cast from `usize` is unchecked.

**Fix**: Use `i32::try_from(toks.len()).map_err(...)` or switch to `i64`.

---

### LOW-6: `download-model.sh` sources `.env` file line-by-line

**File**: `scripts/download-model.sh:111-116`
**Severity**: LOW

```bash
while IFS='=' read -r key value; do
    ...
    export "$key=$value" 2>/dev/null || true
done < "$ENV_FILE"
```

This is safer than `source .env` (no command execution), but a malicious
`.env` with key names containing shell metacharacters could theoretically
cause issues. The `2>/dev/null || true` suppresses errors silently.

**Fix**: Validate key names match `^[A-Za-z_][A-Za-z0-9_]*$` before export.

---

## Items Verified (No Issues Found)

1. **No hardcoded secrets**: Grep for API keys, tokens, passwords, private
   keys, AWS keys, GitHub tokens -- none found. `.gitignore` correctly
   excludes secrets directories.

2. **No `.env` files committed**: Confirmed via glob search.

3. **FFI null pointer checks**: All `llama_model_load_from_file`,
   `llama_init_from_model`, `llama_get_logits_ith`, `llama_get_embeddings`,
   and `chimere_sampler_init` return values are checked for null before use.

4. **FFI lifetime management**: `LlamaForward::Drop` correctly calls
   `llama_free()`, `llama_free_model()`, and `llama_backend_free()`.
   The `_ncmoe_patterns` and `_ncmoe_overrides` fields keep CString
   patterns alive for the lifetime of the model (avoiding dangling pointers).

5. **Docker network isolation**: SearXNG uses `cap_drop: ALL` with minimal
   caps added. ChromaDB is not exposed to the host. Internal services
   communicate over `chimere-net` bridge.

6. **No code execution from env vars**: All `CHIMERE_*` env vars are parsed
   as strings, integers, or booleans. None are passed to `eval()`, `exec()`,
   `Command::new()`, or similar execution primitives.

7. **Serde deserialization**: Request parsing via `serde_json` is safe.
   The custom `deserialize_content` visitor handles both string and array
   formats without panicking.

8. **Docker build is multi-stage**: Inference Dockerfile uses a builder stage
   and copies only the binary and shared libs to the runtime image.
   Build tools, source code, and Rust toolchain are not in the final image.

9. **`build.rs` files**: Both build scripts use `cc::Build` and `Command::new("nvcc")`
   with hardcoded, non-user-controlled arguments. No injection vectors.

10. **`entrypoint.sh`**: Uses `set -eu`, properly quotes all variables,
    builds args via `set --` (no word-splitting), and uses `exec "$@"`.

---

## Documentation Accuracy

| File | Status | Issue |
|------|--------|-------|
| `README.md` | Current | Accurate performance numbers, download links, license |
| `docker/README.md` | Current | Comprehensive, matches docker-compose.yml |
| `chimere-server/ARCHITECTURE.md` | **Outdated** | Says "CPU-only, not connected to real weights" -- see LOW-1 |
| `chimere-server/ROADMAP-v2.md` | N/A | Brief roadmap, no accuracy concerns |
| `chimere-server/docs/*.md` | Current | Technical design docs, match code |

---

## Recommendations (Priority Order)

1. **Enforce `max_tokens` limit** (HIGH-3) -- Immediate, simple fix
2. **Add non-root Docker user** (HIGH-2) -- Standard Docker security
3. **Generate SearXNG secret at runtime** (HIGH-1) -- Remove from tracked file
4. **Add optional API key auth** (MEDIUM-3) -- Before any network exposure
5. **Add alignment assertions** (MEDIUM-5, MEDIUM-7) -- Prevents subtle UB
6. **Remove or validate `engram_table` field** (MEDIUM-1) -- Dead code cleanup
7. **Sanitize error messages** (MEDIUM-8) -- Before production use
8. **Update `ARCHITECTURE.md`** (LOW-1) -- Documentation hygiene
9. **Set up `cargo audit` in CI** (LOW-3) -- Ongoing dependency security
10. **Pin git dependencies** (LOW-3) -- Reproducible builds
