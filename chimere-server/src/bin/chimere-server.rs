//! chimere-server — OpenAI-compatible inference server for chimere-deltanet.
//!
//! # Usage
//!
//! ```sh
//! # Default port 8090, default model path
//! cargo run --release --features server --bin chimere-server
//!
//! # Custom port and model
//! CHIMERE_PORT=8091 \
//! CHIMERE_MODEL=~/.chimere/models/my-model.gguf \
//! CHIMERE_TOKENIZER=~/.chimere/models/my-model/tokenizer.json \
//! cargo run --release --features server --bin chimere-server
//! ```
//!
//! # Environment variables
//!
//! | Variable           | Default                                                                | Description              |
//! |--------------------|------------------------------------------------------------------------|--------------------------|
//! | `CHIMERE_PORT`     | `8090`                                                                 | Listening port           |
//! | `CHIMERE_MODEL`    | `$HOME/.chimere/models/Qwen3.5-35B-A3B-GGUF/...IQ3_S-custom-mix.gguf` | GGUF model path        |
//! | `CHIMERE_TOKENIZER`| `$HOME/.chimere/models/qwopus-27b-bf16/tokenizer.json`               | Tokenizer path (optional)|
//! | `CHIMERE_NAME`     | `chimere-deltanet`                                                     | Model name in responses  |
//! | `CHIMERE_LLAMA_BACKEND` | (unset)                                                           | Set to `1` to use libllama FFI (93 tok/s, recommended) |
//! | `CHIMERE_CUDARC_FORWARD` | (unset)                                                          | Set to `1` to use cudarc forward path (~39 tok/s) |
//! | `CHIMERE_NCMOE`    | `4`                                                                    | CPU-offloaded MoE layers (cudarc/llama path) |
//! | `CHIMERE_KV_MAX_SEQ` | `65536`                                                              | Max sequence length (context size) |
//! | `CHIMERE_KV_TYPE_K`| `8`                                                                    | KV cache key type (8=Q8_0) |
//! | `CHIMERE_KV_TYPE_V`| `2`                                                                    | KV cache value type (2=Q4_0) |
//! | `CHIMERE_FLASH_ATTN`| `1`                                                                   | Enable flash attention (default: on) |
//! | `CHIMERE_MULTISLOT`| `1`                                                                    | Number of scheduler slots (>=2 arms J2) |
//! | `CHIMERE_MULTISLOT_NATIVE` | (unset)                                                        | Set to `1` with `CHIMERE_MULTISLOT>=2` to arm the J4-rewrite NativeScheduler |
//! | `CHIMERE_NATIVE_ENGRAM_ALPHA` | `0.0`                                                       | Default engram bias alpha used by NativeScheduler when request does not override |
//! | `CHIMERE_MAX_PREFILL_CHUNK` | `256`                                                          | Native scheduler: max prompt tokens per `forward_multi_seq` prefill tick (alias of `CHIMERE_NATIVE_MAX_PREFILL_CHUNK`) |
//! | `CHIMERE_SKIP_LEGACY_LLAMA` | (unset)                                                       | Set to `1` to skip the legacy `Qwen35Model::init_llama_forward` when NativeScheduler is armed (saves ~KV cache VRAM) |
//! | `CHIMERE_PREFIX_CACHE`       | `0`                                                         | M2-J2: master kill-switch for the prompt-prefix cache. `0` -> bit-identical to M1. `1` -> lookup on admission, snapshot on reap (2-5x faster prefill on warm system prompts). |
//! | `CHIMERE_PREFIX_CACHE_MAX_BYTES` | `1073741824` (1 GB)                                      | Soft upper bound on total cached KV blob bytes; LRU-evicted on insert. |
//! | `CHIMERE_PREFIX_CACHE_MAX_NODES` | `256`                                                    | Upper bound on the number of cache entries (independent of bytes). |
//!
//! # Observability
//!
//! - `GET /health` — liveness (cheap, no lock).
//! - `GET /metrics` — Prometheus text exposition 0.0.4.
//! - `GET /v1/status` — JSON snapshot (envelope + metrics block).

use std::sync::Arc;

use candle_core::Device;
use chimere_deltanet::chimere_model::ModelArch;
use chimere_deltanet::generate::load_tokenizer;
use chimere_deltanet::generic_model::GenericModel;
use chimere_deltanet::gguf_loader::GgufFile;
use chimere_deltanet::metrics::Metrics;
use chimere_deltanet::qwen35_model::Qwen35Model;
use chimere_deltanet::server::{AppState, AppStateModel, build_router};
use chimere_deltanet::slot_scheduler::{NativeScheduler, Scheduler, SchedulerConfig};
use tokio::net::TcpListener;
use tokio::sync::Mutex;

/// Detect the architecture of a GGUF file by peeking at its
/// `general.architecture` metadata field. The mmap is opened, the
/// metadata read, then the file is dropped — total cost is well under
/// 100 ms for a 15 GB GGUF.
///
/// Tolerant matching: accepts the canonical strings emitted by current
/// llama.cpp / ik_llama (`qwen35moe`, `mamba`, `mamba2`, `nemotron_h_moe`)
/// plus a few common spellings observed in the wild.
///
/// Verified by PF-2 on 2026-04-07: chimere-v3-ramp.gguf reports
/// `qwen35moe`, Nemotron-3-Nano-30B-A3B-Q4_0.gguf reports
/// `nemotron_h_moe`. Both are matched below.
fn detect_arch(model_path: &str) -> Result<ModelArch, String> {
    let gguf = GgufFile::open(model_path)
        .map_err(|e| format!("cannot open GGUF {}: {}", model_path, e))?;
    let arch_str = gguf
        .get_metadata("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    drop(gguf); // release mmap before the heavy load below
    eprintln!(
        "[chimere-server] Detected GGUF architecture: '{}'",
        arch_str
    );
    let arch = match arch_str.to_ascii_lowercase().as_str() {
        // Qwen3.5 (production). PF-2 confirmed exact spelling = "qwen35moe".
        "qwen35" | "qwen3.5" | "qwen3_5" | "qwen35moe" | "qwen3.5moe"
        | "qwen3next" | "qwen3.5next" => ModelArch::Qwen35A3B,
        "mamba" | "mamba1" => ModelArch::Mamba1,
        "mamba2" | "mamba_2" => ModelArch::Mamba2,
        "nemotron_h_moe" | "nemotronh" | "nemotron-h" | "nemotron_h" => {
            ModelArch::NemotronHMoe
        }
        other => {
            return Err(format!(
                "unsupported architecture '{}'. Supported: qwen3.5, mamba, \
                 mamba2, nemotron_h_moe.",
                other
            ));
        }
    };
    Ok(arch)
}

#[tokio::main]
async fn main() {
    // -----------------------------------------------------------------
    // Configuration from environment
    // -----------------------------------------------------------------
    let port = std::env::var("CHIMERE_PORT").unwrap_or_else(|_| "8090".into());
    let addr = format!("0.0.0.0:{}", port);

    let home = std::env::var("HOME").unwrap_or_else(|_| "{HOME}".into());

    let model_path = std::env::var("CHIMERE_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.chimere/models/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-IQ3_S-custom-mix.gguf",
            home
        )
    });

    let tokenizer_path = std::env::var("CHIMERE_TOKENIZER").ok();

    let model_name =
        std::env::var("CHIMERE_NAME").unwrap_or_else(|_| "chimere-deltanet".into());

    let llama_backend = std::env::var("CHIMERE_LLAMA_BACKEND").is_ok();
    let cudarc_forward = std::env::var("CHIMERE_CUDARC_FORWARD").is_ok();

    // -----------------------------------------------------------------
    // Load tokenizer (CPU-only JSON, no VRAM)
    // -----------------------------------------------------------------
    eprintln!("[chimere-server] Loading tokenizer...");
    let tokenizer = match load_tokenizer(tokenizer_path.as_deref()) {
        Ok(t) => {
            eprintln!("[chimere-server] Tokenizer loaded.");
            Arc::new(t)
        }
        Err(e) => {
            eprintln!("[chimere-server] Fatal: tokenizer load failed: {}", e);
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------
    // Select compute device
    // -----------------------------------------------------------------
    // Use new_with_stream() for a non-blocking dedicated CUDA stream.
    // The legacy default stream (from cuda_if_available) does NOT support
    // CUDA Graph capture — cuStreamBeginCapture fails with UNSUPPORTED.
    // Use the default stream (cuda_if_available). new_with_stream() + disable_event_tracking
    // is needed for CUDA Graph capture (bench tests) but causes generation issues
    // when mixed with Candle tensor operations in Qwen35Model.
    let device = match Device::cuda_if_available(0) {
        Ok(d) => {
            eprintln!("[chimere-server] Using CUDA device.");
            d
        }
        Err(_) => {
            eprintln!("[chimere-server] CUDA unavailable, falling back to CPU.");
            Device::Cpu
        }
    };

    // -----------------------------------------------------------------
    // Step 7: detect architecture from GGUF metadata, dispatch to the
    // correct loader (Qwen35Model for the prod path, GenericModel for
    // libllama-only archs like Mamba-2 / Nemotron-H MoE).
    // -----------------------------------------------------------------
    let arch = match detect_arch(&model_path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[chimere-server] Fatal: {}", e);
            std::process::exit(1);
        }
    };

    // Belt-and-braces: a stray env change must NOT load a non-Qwen GGUF
    // into the prod slot. Set CHIMERE_FORCE_QWEN35=1 in the production
    // service unit to enforce this.
    if std::env::var("CHIMERE_FORCE_QWEN35").is_ok() && arch != ModelArch::Qwen35A3B {
        eprintln!(
            "[chimere-server] Fatal: CHIMERE_FORCE_QWEN35=1 but GGUF arch is {}",
            arch.name()
        );
        std::process::exit(1);
    }

    // -----------------------------------------------------------------
    // M1 J4-final: decide up-front whether the NativeScheduler will be
    // armed. This needs to happen BEFORE the model loader because a
    // second `LlamaForward` FFI context is built for the scheduler, and
    // the legacy `Qwen35Model::init_llama_forward` must be skipped when
    // the operator opts in to `CHIMERE_SKIP_LEGACY_LLAMA=1` (avoids
    // doubling KV cache VRAM).
    //
    // Default behaviour when `CHIMERE_MULTISLOT_NATIVE` is unset or `0`:
    //   - scheduler_cfg.is_native() == false
    //   - legacy Qwen35Model::init_llama_forward is called as before
    //   - `native_scheduler_arc` stays `None`
    //   - BIT-IDENTICAL to current prod.
    // -----------------------------------------------------------------
    let scheduler_cfg_preview = SchedulerConfig::from_env();
    let native_planned = scheduler_cfg_preview.is_native();
    let skip_legacy_llama = native_planned
        && std::env::var("CHIMERE_SKIP_LEGACY_LLAMA")
            .map(|v| {
                let t = v.trim();
                !(t.is_empty() || t == "0" || t.eq_ignore_ascii_case("false"))
            })
            .unwrap_or(false);
    if native_planned {
        eprintln!(
            "[chimere-server] M1 J4-final: NativeScheduler WILL be armed \
             (num_slots={}, queue_cap={}). skip_legacy_llama={}",
            scheduler_cfg_preview.num_slots,
            scheduler_cfg_preview.queue_cap,
            skip_legacy_llama,
        );
    }

    // -----------------------------------------------------------------
    // Load model according to arch.
    //
    // Qwen3.5 — three paths ordered by performance:
    //   1. CHIMERE_LLAMA_BACKEND=1 (93 tok/s, prod default)
    //   2. CHIMERE_CUDARC_FORWARD=1 (~39 tok/s, dev/debug)
    //   3. Default: full Candle path via from_gguf()
    //
    // Generic (Mamba/Nemotron) — single path: GenericModel::from_env(arch).
    // CHIMERE_LLAMA_BACKEND / CHIMERE_CUDARC_FORWARD are ignored on the
    // Generic path (libllama is implicit, cudarc is unsupported).
    // -----------------------------------------------------------------
    let app_model: AppStateModel = match arch {
        ModelArch::Qwen35A3B => {
            let qwen = if llama_backend {
                eprintln!("[chimere-server] LLAMA_BACKEND mode: loading via libllama.so (93 tok/s)...");
                let shell = match Qwen35Model::cudarc_shell(&model_path, &device) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[chimere-server] Fatal: shell creation failed: {}", e);
                        std::process::exit(1);
                    }
                };
                if skip_legacy_llama {
                    // J4-final: NativeScheduler armed + operator opted in.
                    // Skip the per-model libllama context — the scheduler
                    // below will build its own via `llama_backend::from_env`.
                    // The legacy `Mutex<AppStateModel>` path will refuse
                    // Qwen35 requests (it expects `llama_forward` to be
                    // initialised). The native SSE path does not touch it.
                    eprintln!(
                        "[chimere-server] CHIMERE_SKIP_LEGACY_LLAMA=1 → \
                         skipping Qwen35Model::init_llama_forward. Non-streaming \
                         requests WILL be rejected by the legacy path. Only \
                         `stream=true` requests are supported in this mode."
                    );
                } else {
                    if let Err(e) = shell.init_llama_forward() {
                        eprintln!("[chimere-server] Fatal: llama_backend init failed: {}", e);
                        std::process::exit(1);
                    }
                    eprintln!("[chimere-server] libllama backend ready.");
                }
                shell
            } else if cudarc_forward {
                eprintln!("[chimere-server] CUDARC mode: loading lightweight shell (no Candle weights)...");
                let shell = match Qwen35Model::cudarc_shell(&model_path, &device) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[chimere-server] Fatal: cudarc shell creation failed: {}", e);
                        std::process::exit(1);
                    }
                };
                eprintln!("[chimere-server] Loading cudarc weights from {} ...", model_path);
                if let Err(e) = shell.init_cudarc_forward() {
                    eprintln!("[chimere-server] Fatal: cudarc forward init failed: {}", e);
                    std::process::exit(1);
                }
                eprintln!("[chimere-server] Cudarc model ready (single load, ~14.7 GB VRAM).");
                shell
            } else {
                eprintln!("[chimere-server] Loading model from {} ...", model_path);
                let m = match Qwen35Model::from_gguf(&model_path, &device, None) {
                    Ok(m) => {
                        eprintln!("[chimere-server] Model loaded (Candle path).");
                        m
                    }
                    Err(e) => {
                        eprintln!("[chimere-server] Fatal: model load failed: {}", e);
                        std::process::exit(1);
                    }
                };
                m
            };
            AppStateModel::Qwen35(qwen)
        }
        ModelArch::Mamba1 | ModelArch::Mamba2 | ModelArch::NemotronHMoe => {
            eprintln!(
                "[chimere-server] GENERIC mode: loading {} via libllama FFI...",
                arch.name()
            );
            // Generic models REQUIRE an external HF tokenizer.json (Step 7
            // ships only the HF path; FFI fallback is Step 7.5).
            if tokenizer_path.is_none() {
                eprintln!(
                    "[chimere-server] Fatal: arch {} requires CHIMERE_TOKENIZER \
                     to point at a HuggingFace tokenizer.json. The built-in \
                     libllama tokenizer is available via the trait but not \
                     yet wired into the HTTP path.",
                    arch.name()
                );
                std::process::exit(1);
            }
            if llama_backend {
                eprintln!(
                    "[chimere-server] Note: CHIMERE_LLAMA_BACKEND=1 is implicit for Generic archs."
                );
            }
            if cudarc_forward {
                eprintln!(
                    "[chimere-server] Warning: CHIMERE_CUDARC_FORWARD=1 ignored for {} (no cudarc path).",
                    arch.name()
                );
            }
            // GenericModel::from_env reads CHIMERE_MODEL etc. internally.
            match GenericModel::from_env(arch) {
                Ok(gm) => {
                    eprintln!(
                        "[chimere-server] GenericModel loaded, arch={}, vocab={}, layers={}",
                        arch.name(),
                        chimere_deltanet::chimere_model::ChimereModel::vocab_size(&gm),
                        chimere_deltanet::chimere_model::ChimereModel::num_layers(&gm),
                    );
                    AppStateModel::Generic(gm)
                }
                Err(e) => {
                    eprintln!("[chimere-server] Fatal: Generic model load failed: {}", e);
                    std::process::exit(1);
                }
            }
        }
        // ModelArch is #[non_exhaustive] — future variants must be handled
        // explicitly when added (compile error rather than silent fallthrough).
        _ => {
            eprintln!(
                "[chimere-server] Fatal: arch '{}' is recognised by detect_arch but \
                 has no loader wired into bin/chimere-server.rs. Add a match arm.",
                arch.name()
            );
            std::process::exit(1);
        }
    };

    // -----------------------------------------------------------------
    // Build shared state and router
    //
    // Qwen35Model is !Sync (contains RefCell), so we wrap it in a Mutex.
    // The Mutex also serialises inference — one request at a time.
    // -----------------------------------------------------------------
    let max_agents: usize = std::env::var("CHIMERE_MAX_AGENTS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    eprintln!("[chimere-server] AgentScheduler: max_agents={}", max_agents);

    // ---------------------------------------------------------------------
    // M1 J2: multi-slot scheduler (optional). The scheduler is built iff
    // `CHIMERE_MULTISLOT` is explicitly set to a value >= 2. Otherwise
    // `AppState.scheduler = None` and the HTTP handlers take the legacy
    // direct-`thread::spawn` path — production behaviour is unchanged.
    //
    // When built, we spawn the dispatcher OS thread here so the admission
    // channel is live before the axum listener starts accepting requests.
    // The JoinHandles are leaked on purpose (process-lifetime workers).
    // ---------------------------------------------------------------------
    let scheduler_cfg = SchedulerConfig::from_env();
    let scheduler_arc: Option<Arc<Scheduler>> = if scheduler_cfg.is_active() {
        eprintln!(
            "[chimere-server] M1 multi-slot ENABLED: num_slots={}, queue_cap={} (CHIMERE_MULTISLOT)",
            scheduler_cfg.num_slots, scheduler_cfg.queue_cap,
        );
        let mut sched = Scheduler::new(scheduler_cfg.clone());
        let handles = sched.spawn_workers();
        let sched_arc = Arc::new(sched);
        // Detach the dispatcher JoinHandle — it lives for the process.
        for h in handles {
            std::mem::forget(h);
        }
        Some(sched_arc)
    } else {
        eprintln!(
            "[chimere-server] M1 multi-slot disabled (CHIMERE_MULTISLOT unset or =1). \
             Using legacy single-slot path."
        );
        None
    };

    // ---------------------------------------------------------------------
    // M1 J4-final: NativeScheduler construction.
    //
    // Only built when `scheduler_cfg.is_native()` — i.e. both
    // `CHIMERE_MULTISLOT>=2` AND `CHIMERE_MULTISLOT_NATIVE=1`. We build a
    // SEPARATE `LlamaForward` via `llama_backend::from_env()` (which reads
    // `CHIMERE_MULTISLOT` and automatically sets `n_seq_max` to the slot
    // count — see `llama_backend.rs::from_env`). This forward context is
    // owned by the scheduler driver thread for the process lifetime.
    //
    // The engram lookup is loaded from `CHIMERE_ENGRAM_DIR` /
    // `CHIMERE_ENGRAM_FILE` and shared with the native slot pool. When
    // `MultiEngramLookup::from_env()` returns `None`, slots fall back to
    // pure sampler behaviour (engram_alpha ignored).
    //
    // When NOT armed, `native_scheduler_arc = None` and the handlers route
    // every request through the legacy `Mutex<AppStateModel>` path.
    // BIT-IDENTICAL to prod today.
    // ---------------------------------------------------------------------
    let native_scheduler_arc: Option<Arc<NativeScheduler>> = if scheduler_cfg.is_native() {
        eprintln!(
            "[chimere-server] M1 J4-final: building NativeScheduler \
             (num_slots={}, queue_cap={}, CHIMERE_MULTISLOT_NATIVE=1)...",
            scheduler_cfg.num_slots, scheduler_cfg.queue_cap,
        );

        // 1. Build the dedicated LlamaForward for the scheduler.
        //    `from_env()` reads CHIMERE_MULTISLOT and sets n_seq_max>=num_slots
        //    when CHIMERE_MULTISLOT_NATIVE=1 (already patched — see
        //    `llama_backend.rs::from_env` lines 1795-1813).
        let llama_fwd = match chimere_deltanet::llama_backend::from_env() {
            Ok(f) => {
                eprintln!(
                    "[chimere-server] NativeScheduler: LlamaForward context built \
                     (vocab={})",
                    f.n_vocab(),
                );
                f
            }
            Err(e) => {
                eprintln!(
                    "[chimere-server] Fatal: NativeScheduler LlamaForward \
                     construction failed: {}",
                    e,
                );
                std::process::exit(1);
            }
        };

        // 2. Load the engram lookup (optional). Shared with all slots.
        let engram_global = chimere_deltanet::engram_lookup::MultiEngramLookup::from_env()
            .map(Arc::new);
        if let Some(eg) = &engram_global {
            eprintln!(
                "[chimere-server] NativeScheduler: engram_global attached ({} tables)",
                eg.len(),
            );
        } else {
            eprintln!(
                "[chimere-server] NativeScheduler: no engram tables loaded \
                 (CHIMERE_ENGRAM_DIR / CHIMERE_ENGRAM_FILE unset or empty)"
            );
        }

        // 3. Default engram alpha (overridable per-request).
        let default_engram_alpha: f32 = std::env::var("CHIMERE_NATIVE_ENGRAM_ALPHA")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0.0);

        // 4. M2-J2d -- optionally build the prompt-prefix cache trie.
        //    Gated on `CacheConfig::from_env().enabled` (i.e.
        //    `CHIMERE_PREFIX_CACHE=1` with non-zero budgets). When off,
        //    we pass `None` and the scheduler's hot paths stay bit-
        //    identical to M1 (no trie touch, no FFI save/restore).
        //
        //    Expected speedup on warm system-prompt repeats: 2-5x on
        //    TTFT. See README.md in `~/Bureau/chimere-drafts/m2-j2-main-wire/`.
        let prefix_cache_cfg = chimere_deltanet::prefix_cache::CacheConfig::from_env();
        let prefix_trie: Option<Arc<std::sync::RwLock<chimere_deltanet::prefix_cache::PrefixTrie>>> =
            if prefix_cache_cfg.enabled {
                eprintln!(
                    "[chimere-server] M2-J2 prompt-prefix cache ENABLED: \
                     max_bytes={} ({:.2} GB), max_nodes={}. \
                     Bypass with CHIMERE_PREFIX_CACHE=0.",
                    prefix_cache_cfg.max_bytes,
                    prefix_cache_cfg.max_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    prefix_cache_cfg.max_nodes,
                );
                let trie = chimere_deltanet::prefix_cache::PrefixTrie::from_config(
                    &prefix_cache_cfg,
                );
                Some(Arc::new(std::sync::RwLock::new(trie)))
            } else {
                eprintln!(
                    "[chimere-server] M2-J2 prompt-prefix cache DISABLED \
                     (CHIMERE_PREFIX_CACHE unset or =0) -- bit-identical to M1"
                );
                None
            };

        // 5. Construct the scheduler and spawn its driver.
        let mut native_sched = match NativeScheduler::new(
            scheduler_cfg.clone(),
            engram_global,
            default_engram_alpha,
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "[chimere-server] Fatal: NativeScheduler::new failed: {}",
                    e,
                );
                std::process::exit(1);
            }
        };

        // Attach the prefix-cache trie before spawning the driver. When
        // `prefix_trie == None` (CHIMERE_PREFIX_CACHE=0), this is a no-op
        // that leaves the scheduler in the M1 bit-identical configuration.
        native_sched = native_sched.with_prefix_cache(prefix_trie);

        // Spawn the driver BEFORE wrapping in Arc — spawn_native_driver
        // requires `&mut self` to consume the admission_rx end.
        let driver_handle = match native_sched.spawn_native_driver(llama_fwd) {
            Ok(h) => h,
            Err(e) => {
                eprintln!(
                    "[chimere-server] Fatal: NativeScheduler driver spawn failed: {}",
                    e,
                );
                std::process::exit(1);
            }
        };
        // Driver thread is process-lifetime — detach the JoinHandle.
        std::mem::forget(driver_handle);

        let sched_arc = Arc::new(native_sched);
        eprintln!(
            "[chimere-server] NativeScheduler ACTIVE: num_slots={}",
            scheduler_cfg.num_slots,
        );
        Some(sched_arc)
    } else {
        if scheduler_cfg.is_active() {
            eprintln!(
                "[chimere-server] M1 J4 NativeScheduler disabled \
                 (CHIMERE_MULTISLOT_NATIVE unset or =0). Using J2 closure path."
            );
        }
        None
    };

    let state = Arc::new(AppState {
        model: Mutex::new(app_model),
        tokenizer,
        model_name,
        agent_scheduler: Mutex::new(chimere_deltanet::agent_scheduler::AgentScheduler::new(max_agents)),
        user_agent_map: Mutex::new(std::collections::HashMap::new()),
        max_agents,
        scheduler: scheduler_arc,
        // M1 J4-final: NativeScheduler is opt-in via CHIMERE_MULTISLOT_NATIVE=1.
        // When disabled (prod default), stays None and all requests take the
        // legacy `Mutex<AppStateModel>` path.
        native_scheduler: native_scheduler_arc,
        // polish-prometheus: process-wide Metrics handle. Cheap to construct
        // (a handful of atomics + a Mutex<Vec<u64>>). Always present so the
        // /metrics scrape endpoint has something to render even before the
        // first request.
        metrics: Arc::new(Metrics::new()),
    });

    let app = build_router(state);

    // -----------------------------------------------------------------
    // Start server
    // -----------------------------------------------------------------
    eprintln!("[chimere-server] Listening on http://{}", addr);
    eprintln!("[chimere-server] Endpoints:");
    eprintln!("  POST http://{}/v1/chat/completions", addr);
    eprintln!("  GET  http://{}/health", addr);
    eprintln!("  GET  http://{}/metrics", addr);
    eprintln!("  GET  http://{}/v1/status", addr);

    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[chimere-server] Fatal: cannot bind {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("[chimere-server] Server error: {}", e);
        std::process::exit(1);
    }
}
