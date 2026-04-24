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

use std::sync::Arc;

use candle_core::Device;
use chimere_deltanet::chimere_model::ModelArch;
use chimere_deltanet::generate::load_tokenizer;
use chimere_deltanet::generic_model::GenericModel;
use chimere_deltanet::gguf_loader::GgufFile;
use chimere_deltanet::qwen35_model::Qwen35Model;
use chimere_deltanet::server::{AppState, AppStateModel, build_router};
use chimere_deltanet::slot_scheduler::{Scheduler, SchedulerConfig};
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
                if let Err(e) = shell.init_llama_forward() {
                    eprintln!("[chimere-server] Fatal: llama_backend init failed: {}", e);
                    std::process::exit(1);
                }
                eprintln!("[chimere-server] libllama backend ready.");
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
        let mut sched = Scheduler::new(scheduler_cfg);
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

    let state = Arc::new(AppState {
        model: Mutex::new(app_model),
        tokenizer,
        model_name,
        agent_scheduler: Mutex::new(chimere_deltanet::agent_scheduler::AgentScheduler::new(max_agents)),
        user_agent_map: Mutex::new(std::collections::HashMap::new()),
        max_agents,
        scheduler: scheduler_arc,
    });

    let app = build_router(state);

    // -----------------------------------------------------------------
    // Start server
    // -----------------------------------------------------------------
    eprintln!("[chimere-server] Listening on http://{}", addr);
    eprintln!("[chimere-server] Endpoints:");
    eprintln!("  POST http://{}/v1/chat/completions", addr);
    eprintln!("  GET  http://{}/health", addr);

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
