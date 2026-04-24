//! # J4-rewrite end-to-end smoke — concurrent HTTP → native scheduler path
//!
//! Spins up an in-process chimere-server on 127.0.0.1:NNNN (random port)
//! with `CHIMERE_MULTISLOT=2` and `CHIMERE_MULTISLOT_NATIVE=1`. Then POSTs
//! two `/v1/chat/completions` requests concurrently with `stream=true`
//! and different prompts. Asserts:
//!
//! 1. Both SSE streams produce at least `MIN_TOKENS_PER_REQ` tokens.
//! 2. The two streams **interleave**: there is at least one event from
//!    request B's timeline sandwiched between two events from request A
//!    (and vice versa). With the legacy Mutex<AppStateModel> serial path
//!    this is typically false — one request completes before the other
//!    starts making visible progress — so an interleave proves the
//!    native multi-seq dispatcher is running.
//! 3. The two streams are **distinct** — total aggregate text of stream
//!    A does not equal stream B. (With a random model on identical
//!    prompts they would anyway, but our prompts are deliberately
//!    different to make this assertion robust.)
//!
//! ## Running
//!
//! ```bash
//! export CHIMERE_MODEL=/path/to/any/small.gguf
//! export CHIMERE_TOKENIZER=/path/to/tokenizer.json   # required for Qwen35 path
//! export CHIMERE_LLAMA_BACKEND=1                      # forces libllama path
//! export CHIMERE_MULTISLOT=2
//! export CHIMERE_MULTISLOT_NATIVE=1
//! export CHIMERE_N_GPU_LAYERS=0                       # CPU — don't conflict with prod :8081
//! export CHIMERE_N_CTX=2048
//! cargo run --release --features server --bin j4-rewrite-smoke
//! ```
//!
//! Exit 0 = PASS, 1 = FAIL.
//!
//! ## Caveats
//!
//! - This binary **requires** a model. It is NOT a unit test; it exercises
//!   the full server/scheduler/FFI stack.
//! - It spawns the chimere-server main loop in a background tokio task
//!   rather than via `std::process::Command`. This keeps everything in
//!   the same address space so a panic in the server shows a clean
//!   backtrace.
//! - On FFI-failure (wrong tokenizer, missing libllama, etc.) the smoke
//!   will hang waiting for the first SSE chunk. A 30 s wall-clock
//!   timeout will kill it and print the observed failure mode.

use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::Client;
use serde_json::json;
use tokio::net::TcpListener;
use tokio::time::timeout;

const MIN_TOKENS_PER_REQ: usize = 5;
const WALLCLOCK_TIMEOUT: Duration = Duration::from_secs(30);

/// A single observed SSE event plus its arrival wall-clock.
#[derive(Debug, Clone)]
struct Observation {
    t_ms: u128,
    event_data: String,
}

/// Trace for one concurrent request.
#[derive(Debug, Clone, Default)]
struct Trace {
    label: &'static str,
    observations: Vec<Observation>,
    aggregate_text: String,
    got_done_marker: bool,
    error: Option<String>,
}

async fn run_one(
    label: &'static str,
    endpoint: String,
    prompt: String,
    started_at: Instant,
) -> Result<Trace, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| format!("client build: {}", e))?;

    let body = json!({
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 32,
        "temperature": 0.01,  // near-greedy so streams are semi-stable
        "top_p": 1.0,
        "top_k": 1,
        "stream": true,
        "chat_template_kwargs": { "enable_thinking": false },
    });

    let resp = client.post(&endpoint)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("{}: POST failed: {}", label, e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("{}: HTTP {} — body: {}", label, status, text));
    }

    let mut trace = Trace {
        label,
        observations: Vec::new(),
        aggregate_text: String::new(),
        got_done_marker: false,
        error: None,
    };

    // Stream the SSE body.
    let mut byte_stream = resp.bytes_stream();
    use futures::StreamExt;
    let mut buffer = Vec::<u8>::new();

    while let Some(chunk) = byte_stream.next().await {
        let chunk = match chunk {
            Ok(b) => b,
            Err(e) => {
                trace.error = Some(format!("body stream error: {}", e));
                break;
            }
        };
        buffer.extend_from_slice(&chunk);

        // Parse SSE events separated by "\n\n".
        loop {
            let delim_pos = match find_subslice(&buffer, b"\n\n") {
                Some(p) => p,
                None => break,
            };
            let event_block: Vec<u8> = buffer.drain(..delim_pos + 2).collect();
            // Extract "data: ..." line — SSE spec may have multiple "data:" lines.
            let block_str = String::from_utf8_lossy(&event_block).to_string();
            let mut event_data = String::new();
            for line in block_str.lines() {
                if let Some(rest) = line.strip_prefix("data:") {
                    let payload = rest.trim();
                    if !event_data.is_empty() {
                        event_data.push('\n');
                    }
                    event_data.push_str(payload);
                }
            }
            let t_ms = started_at.elapsed().as_millis();
            trace.observations.push(Observation {
                t_ms,
                event_data: event_data.clone(),
            });

            if event_data == "[DONE]" {
                trace.got_done_marker = true;
                // The server emits [DONE] only after the stop chunk.
                return Ok(trace);
            }

            // Try to JSON-parse the event_data and pull out `choices[0].delta.content`.
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&event_data) {
                if let Some(content) = v
                    .pointer("/choices/0/delta/content")
                    .and_then(|c| c.as_str())
                {
                    trace.aggregate_text.push_str(content);
                }
            }
        }
    }
    Ok(trace)
}

/// Helper: find subslice. Mirrors `[u8]::windows(n).position(...)` without
/// requiring the `memchr` crate.
fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    for i in 0..=haystack.len() - needle.len() {
        if &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

/// Count the number of "transitions" in the interleave merge of two
/// per-request traces. Each SSE event from both traces is sorted by
/// `t_ms`; a transition is a point where the current event's label differs
/// from the previous. 0 transitions means all of A arrived before any of B
/// (no interleave). ≥ 2 transitions means A and B interleaved.
fn count_transitions(a: &[Observation], b: &[Observation], label_a: &'static str, label_b: &'static str) -> usize {
    let mut merged: Vec<(u128, &'static str)> = Vec::new();
    merged.extend(a.iter().map(|o| (o.t_ms, label_a)));
    merged.extend(b.iter().map(|o| (o.t_ms, label_b)));
    merged.sort_by_key(|x| x.0);
    if merged.is_empty() {
        return 0;
    }
    let mut prev = merged[0].1;
    let mut transitions = 0;
    for (_, lbl) in &merged[1..] {
        if *lbl != prev {
            transitions += 1;
            prev = *lbl;
        }
    }
    transitions
}

#[tokio::main]
async fn main() {
    match run().await {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("\n[j4-rewrite-smoke] FAIL: {}", e);
            std::process::exit(1);
        }
    }
}

async fn run() -> Result<(), String> {
    // ---------------------------------------------------------------
    // 1. Environment sanity + prod-safety
    // ---------------------------------------------------------------
    std::env::set_var("CHIMERE_MULTISLOT", "2");
    std::env::set_var("CHIMERE_MULTISLOT_NATIVE", "1");
    if std::env::var("CHIMERE_MODEL").is_err() {
        return Err("CHIMERE_MODEL env var is required".to_string());
    }
    if std::env::var("CHIMERE_N_GPU_LAYERS").is_err() {
        // Default to CPU to avoid conflict with prod :8081 (14 GB VRAM held).
        std::env::set_var("CHIMERE_N_GPU_LAYERS", "0");
    }
    // Pick a random high port via ephemeral bind.
    let listener = TcpListener::bind("127.0.0.1:0").await
        .map_err(|e| format!("bind ephemeral: {}", e))?;
    let port = listener.local_addr()
        .map_err(|e| format!("local_addr: {}", e))?
        .port();
    let bind_addr = format!("127.0.0.1:{}", port);
    drop(listener);
    std::env::set_var("CHIMERE_PORT", port.to_string());
    eprintln!("[j4-rewrite-smoke] Will launch chimere-server on {}", bind_addr);

    // ---------------------------------------------------------------
    // 2. Spawn chimere-server in-process.
    //
    // We invoke the existing launcher (bin/chimere-server.rs does
    // everything: model load, scheduler init, axum bind). By importing
    // the lib types we can't directly re-run that `main`, so instead we
    // spawn it as a subprocess via std::process::Command. This keeps the
    // test hermetic and catches env-var regressions end-to-end.
    //
    // NOTE(OPEN QUESTION): a true in-process launch would require
    // refactoring bin/chimere-server.rs into a reusable lib entry. Out
    // of scope for this draft — subprocess launch is the pragmatic
    // choice (no lib boundary changes, uses the exact prod binary).
    // ---------------------------------------------------------------
    let server_bin = std::env::current_exe()
        .map_err(|e| format!("current_exe: {}", e))?;
    let server_bin_dir = server_bin
        .parent()
        .ok_or_else(|| "current_exe has no parent".to_string())?;
    let main_bin = server_bin_dir.join("chimere-server");
    if !main_bin.exists() {
        return Err(format!(
            "chimere-server binary not found at {}. Build it first: cargo build --release --features server --bin chimere-server",
            main_bin.display(),
        ));
    }

    eprintln!("[j4-rewrite-smoke] Launching subprocess: {}", main_bin.display());
    let mut child = std::process::Command::new(&main_bin)
        .envs(std::env::vars())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn chimere-server: {}", e))?;

    // Shared flag to stop the tail threads.
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Tail stdout/stderr in background so logs don't buffer fill.
    if let Some(stdout) = child.stdout.take() {
        let stop = Arc::clone(&stop_flag);
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                if let Ok(l) = line {
                    eprintln!("[server.out] {}", l);
                }
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let stop = Arc::clone(&stop_flag);
        std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if stop.load(std::sync::atomic::Ordering::SeqCst) {
                    break;
                }
                if let Ok(l) = line {
                    eprintln!("[server.err] {}", l);
                }
            }
        });
    }

    // Cleanup-on-exit guard.
    struct KillGuard {
        child: std::process::Child,
        stop: Arc<std::sync::atomic::AtomicBool>,
    }
    impl Drop for KillGuard {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
    let _guard = KillGuard { child, stop: stop_flag };

    // ---------------------------------------------------------------
    // 3. Wait for /health to respond. Up to 120 s for large model loads.
    // ---------------------------------------------------------------
    let health_endpoint = format!("http://{}/health", bind_addr);
    eprintln!("[j4-rewrite-smoke] Waiting for server health on {}", health_endpoint);
    let health_client = Client::new();
    let health_deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if Instant::now() > health_deadline {
            return Err("server did not respond on /health within 120 s".to_string());
        }
        match health_client.get(&health_endpoint).timeout(Duration::from_millis(500)).send().await {
            Ok(r) if r.status().is_success() => break,
            _ => {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
    }
    eprintln!("[j4-rewrite-smoke] Server healthy");

    // ---------------------------------------------------------------
    // 4. Fire two concurrent /v1/chat/completions requests.
    // ---------------------------------------------------------------
    let endpoint = format!("http://{}/v1/chat/completions", bind_addr);
    let started_at = Instant::now();

    // Distinct prompts to make the cross-contamination check robust.
    let prompt_a = "Count from one to five slowly.".to_string();
    let prompt_b = "List three colors: red, ".to_string();

    let task_a = tokio::spawn(run_one("A", endpoint.clone(), prompt_a, started_at));
    // Tiny offset so the second request enters the admission queue a beat later.
    tokio::time::sleep(Duration::from_millis(5)).await;
    let task_b = tokio::spawn(run_one("B", endpoint.clone(), prompt_b, started_at));

    let (trace_a_res, trace_b_res) = match timeout(WALLCLOCK_TIMEOUT, async {
        tokio::join!(task_a, task_b)
    }).await {
        Ok(pair) => pair,
        Err(_) => return Err(format!("requests timed out after {:?}", WALLCLOCK_TIMEOUT)),
    };

    let trace_a = trace_a_res.map_err(|e| format!("task A join: {}", e))??;
    let trace_b = trace_b_res.map_err(|e| format!("task B join: {}", e))??;

    eprintln!(
        "[j4-rewrite-smoke] Request A: {} SSE events, aggregate text len {}, done_marker={}, err={:?}",
        trace_a.observations.len(), trace_a.aggregate_text.len(), trace_a.got_done_marker, trace_a.error,
    );
    eprintln!(
        "[j4-rewrite-smoke] Request B: {} SSE events, aggregate text len {}, done_marker={}, err={:?}",
        trace_b.observations.len(), trace_b.aggregate_text.len(), trace_b.got_done_marker, trace_b.error,
    );

    // ---------------------------------------------------------------
    // 5. Assertions.
    // ---------------------------------------------------------------
    if let Some(e) = &trace_a.error {
        return Err(format!("request A error: {}", e));
    }
    if let Some(e) = &trace_b.error {
        return Err(format!("request B error: {}", e));
    }
    if trace_a.observations.len() < MIN_TOKENS_PER_REQ {
        return Err(format!(
            "request A produced only {} events, need >= {} (short-circuit?)",
            trace_a.observations.len(), MIN_TOKENS_PER_REQ,
        ));
    }
    if trace_b.observations.len() < MIN_TOKENS_PER_REQ {
        return Err(format!(
            "request B produced only {} events, need >= {} (short-circuit?)",
            trace_b.observations.len(), MIN_TOKENS_PER_REQ,
        ));
    }
    if !trace_a.got_done_marker {
        return Err("request A never received [DONE]".to_string());
    }
    if !trace_b.got_done_marker {
        return Err("request B never received [DONE]".to_string());
    }

    // Interleave check.
    let transitions = count_transitions(
        &trace_a.observations,
        &trace_b.observations,
        "A", "B",
    );
    eprintln!(
        "[j4-rewrite-smoke] Merge-sorted transitions count: {}",
        transitions,
    );
    if transitions < 2 {
        return Err(format!(
            "streams did not interleave (transitions={}); native multi-seq likely not active",
            transitions,
        ));
    }

    // Distinctness check. With deterministic (temp=0.01, top_k=1) sampling,
    // different prompts should produce different outputs.
    if trace_a.aggregate_text == trace_b.aggregate_text {
        return Err(format!(
            "streams A and B produced IDENTICAL aggregate text — possible cross-slot contamination. \
             Text: '{}'",
            trace_a.aggregate_text,
        ));
    }

    // ---------------------------------------------------------------
    // 6. PASS summary.
    // ---------------------------------------------------------------
    eprintln!();
    eprintln!("===== J4-REWRITE SMOKE PASS =====");
    eprintln!("  requests served       : 2 (A+B)");
    eprintln!("  events A              : {}", trace_a.observations.len());
    eprintln!("  events B              : {}", trace_b.observations.len());
    eprintln!("  interleave transitions: {}", transitions);
    eprintln!("  A distinct from B     : yes");
    eprintln!("  total elapsed         : {:?}", started_at.elapsed());
    eprintln!();

    Ok(())
}
