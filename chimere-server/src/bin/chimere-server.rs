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
    // Load model — three paths, ordered by performance:
    //
    // 1. CHIMERE_LLAMA_BACKEND=1 (recommended, 93 tok/s):
    //    Create a lightweight shell (config + MRoPE only, ~0 Candle VRAM).
    //    init_llama_forward() loads the model via libllama.so (~14.7 GB).
    //    The entire forward pass is delegated to ik_llama's CUDA kernels.
    //
    // 2. CHIMERE_CUDARC_FORWARD=1 (~39 tok/s):
    //    Shell + cudarc raw weights from GGUF.
    //
    // 3. Default: Full Candle path via from_gguf().
    // -----------------------------------------------------------------
    let model = if llama_backend {
        eprintln!("[chimere-server] LLAMA_BACKEND mode: loading via libllama.so (93 tok/s)...");
        // Create a lightweight shell — only config + MRoPE, zero Candle weight VRAM.
        // libllama loads weights independently via its own CUDA backend.
        let shell = match Qwen35Model::cudarc_shell(&model_path, &device) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[chimere-server] Fatal: shell creation failed: {}", e);
                std::process::exit(1);
            }
        };

        // Initialize libllama FFI — loads GGUF and creates context.
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

        // Load cudarc weights from GGUF — this is the ONLY GPU weight load.
        eprintln!("[chimere-server] Loading cudarc weights from {} ...", model_path);
        if let Err(e) = shell.init_cudarc_forward() {
            eprintln!("[chimere-server] Fatal: cudarc forward init failed: {}", e);
            std::process::exit(1);
        }
        eprintln!("[chimere-server] Cudarc model ready (single load, ~14.7 GB VRAM).");
        shell
    } else {
        // Legacy Candle path: load full model with QMatMul weights.
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

        // NOTE: Do NOT call init_cudarc_forward() here — it would load a
        // second copy of all weights (~14 GB) causing OOM on 16 GB GPUs.
        // The Candle path uses QMatMul directly for inference.
        m
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

    let state = Arc::new(AppState {
        model: Mutex::new(model),
        tokenizer,
        model_name,
        agent_scheduler: Mutex::new(chimere_deltanet::agent_scheduler::AgentScheduler::new(max_agents)),
        user_agent_map: Mutex::new(std::collections::HashMap::new()),
        max_agents,
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
