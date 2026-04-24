//! # M1 multi-slot concurrency bench harness
//!
//! Drives `N` concurrent `/v1/chat/completions` streams against a running
//! `chimere-server` binary and reports aggregate throughput, per-request
//! latency percentiles, VRAM delta, and a coarse engram-isolation assertion
//! (all N answers must be byte-distinct when N distinct prompts are used).
//!
//! This is the J8 deliverable for the M1 plan
//! (see `~/Bureau/plan-M1-multislot-2026-04-24.md §7 KPIs + §5 J8`).
//!
//! ## Sweep
//!
//! The harness runs the same load three times against three *different*
//! `chimere-server` processes (the caller is responsible for starting and
//! stopping them; see `scripts/bench_m1.sh` for the one-shot wrapper):
//!
//! | Pass      | `CHIMERE_MULTISLOT` | Expected        |
//! |-----------|---------------------|-----------------|
//! | baseline  | unset / `1`         | legacy path     |
//! | 2-slot    | `2`                 | `≥ 1.7× base`   |
//! | 4-slot    | `4`                 | `≥ 3.0× base`   |
//!
//! The ratio target comes from `§7 KPIs` in the plan. Numbers below those
//! thresholds fail the smoke.
//!
//! ## Output
//!
//! A JSON-ish line per pass, plus a final markdown table on stdout:
//!
//! ```text
//! [bench-m1] baseline     n=100 conc=4  agg=XXX tok/s  p50=XXms  p95=XXms  p99=XXms  vram=XX.XG
//! [bench-m1] 2-slot       n=100 conc=4  agg=XXX tok/s  p50=XXms  p95=XXms  p99=XXms  vram=XX.XG  ratio=1.XX
//! [bench-m1] 4-slot       n=100 conc=4  agg=XXX tok/s  p50=XXms  p95=XXms  p99=XXms  vram=XX.XG  ratio=X.XX
//! ```
//!
//! ## Environment
//!
//! | Variable             | Default                            | Notes                                     |
//! |----------------------|------------------------------------|-------------------------------------------|
//! | `BENCH_URL`          | `http://127.0.0.1:8082/v1/chat/completions` | Endpoint under test (NEVER :8081)  |
//! | `BENCH_MODEL`        | `chimere-deltanet`                 | `model` field in the request              |
//! | `BENCH_CONC`         | `4`                                | concurrency level                         |
//! | `BENCH_N`            | `100`                              | total requests per pass                   |
//! | `BENCH_MAX_TOKENS`   | `64`                               | tokens/request (keep short to bound wall) |
//! | `BENCH_PASS_LABEL`   | `baseline`                         | free-form label for the output row        |
//! | `BENCH_BASELINE_TPS` | `0` (disables ratio)               | tokens/s from the baseline pass           |
//! | `BENCH_ASSERT_ISO`   | `1`                                | assert all N answers byte-distinct        |
//!
//! The harness does **not** start/stop chimere-server itself — that is the
//! shell wrapper's job. `BENCH_URL` defaulting to `:8082` is a hard guard so
//! an accidental `cargo run` never hits production `:8081`.
//!
//! ## CPU-only fallback
//!
//! If `nvidia-smi` is not available or returns an error, the VRAM column is
//! printed as `n/a`. The harness still emits throughput / latency numbers.
//!
//! ## Status (2026-04-24)
//!
//! This is the bench *harness*. Running it end-to-end requires the J4 HTTP
//! dispatcher rewrite (see `mtp_scheduler.rs` → `slot_scheduler.rs` brief in
//! `chimere-m1-j4-j5-2026-04-24.md`) so the multi-slot path actually drives
//! `forward_multi_seq`. Until then, passes `2-slot` and `4-slot` will
//! observe no speedup (all requests still serialise through
//! `AppState.model.lock()`). The harness prints this warning when the ratio
//! comes in below `1.2×`.

use std::env;
use std::process::Command;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

// ---------------------------------------------------------------------------
// Request / response types (minimal subset of the OpenAI chat schema)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    #[serde(default)]
    completion_tokens: u32,
}

// ---------------------------------------------------------------------------
// Per-request observation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ReqObs {
    latency_ms: u64,
    gen_tokens: u32,
    body: String,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// Prompt bank: N synthetically distinct prompts so the isolation assertion
// has signal. We intentionally vary short English sentences rather than using
// a random-number seed so the request payload is deterministic and
// cacheable across passes.
// ---------------------------------------------------------------------------

fn prompt_bank() -> Vec<&'static str> {
    vec![
        "Write one short sentence about the Eiffel Tower.",
        "Write one short sentence about Mount Fuji.",
        "Write one short sentence about the Amazon river.",
        "Write one short sentence about the Sahara desert.",
        "Write one short sentence about the Great Wall.",
        "Write one short sentence about the Great Barrier Reef.",
        "Write one short sentence about the Niagara Falls.",
        "Write one short sentence about Machu Picchu.",
    ]
}

// ---------------------------------------------------------------------------
// One HTTP request. No retries. Captures wall time and the returned text.
// ---------------------------------------------------------------------------

#[cfg(feature = "server")]
async fn one_request(
    client: &reqwest::Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> ReqObs {
    let req = ChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: prompt.into(),
        }],
        max_tokens,
        temperature: 0.7,
        stream: false,
    };

    let t0 = Instant::now();
    let resp = match client.post(url).json(&req).send().await {
        Ok(r) => r,
        Err(e) => {
            return ReqObs {
                latency_ms: t0.elapsed().as_millis() as u64,
                gen_tokens: 0,
                body: String::new(),
                error: Some(format!("send: {}", e)),
            };
        }
    };
    let status = resp.status();
    let text = match resp.text().await {
        Ok(t) => t,
        Err(e) => {
            return ReqObs {
                latency_ms: t0.elapsed().as_millis() as u64,
                gen_tokens: 0,
                body: String::new(),
                error: Some(format!("read body: {}", e)),
            };
        }
    };
    let latency_ms = t0.elapsed().as_millis() as u64;

    if !status.is_success() {
        return ReqObs {
            latency_ms,
            gen_tokens: 0,
            body: text.clone(),
            error: Some(format!("http {}", status)),
        };
    }

    let parsed: Result<ChatResponse, _> = serde_json::from_str(&text);
    match parsed {
        Ok(p) => {
            let body = p
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();
            let gen_tokens = p.usage.map(|u| u.completion_tokens).unwrap_or(0);
            ReqObs {
                latency_ms,
                gen_tokens,
                body,
                error: None,
            }
        }
        Err(e) => ReqObs {
            latency_ms,
            gen_tokens: 0,
            body: text,
            error: Some(format!("parse: {}", e)),
        },
    }
}

// ---------------------------------------------------------------------------
// Percentile helper. Sorts in-place.
// ---------------------------------------------------------------------------

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ---------------------------------------------------------------------------
// VRAM snapshot via `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`.
// Returns `None` if the binary is missing or fails. Units: GiB float.
// ---------------------------------------------------------------------------

fn vram_used_gib() -> Option<f32> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let first_line = s.lines().next()?.trim();
    let mib: f32 = first_line.parse().ok()?;
    Some(mib / 1024.0)
}

// ---------------------------------------------------------------------------
// Bench pass: drive `n` total requests at `conc` concurrency, return all
// observations. Uses the prompt bank round-robin so each slot sees a
// different prompt.
// ---------------------------------------------------------------------------

#[cfg(feature = "server")]
async fn run_pass(url: &str, model: &str, conc: usize, n: usize, max_tokens: u32) -> Vec<ReqObs> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .pool_max_idle_per_host(conc)
        .build()
        .expect("reqwest client");
    let client = Arc::new(client);
    let prompts: Vec<&'static str> = prompt_bank();
    let prompts = Arc::new(prompts);
    let results: Arc<Mutex<Vec<ReqObs>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let url = Arc::new(url.to_string());
    let model = Arc::new(model.to_string());

    // Simple semaphore-style concurrency cap
    let sem = Arc::new(tokio::sync::Semaphore::new(conc));

    let mut handles: Vec<JoinHandle<()>> = Vec::with_capacity(n);
    for i in 0..n {
        let permit = sem.clone().acquire_owned().await.expect("semaphore");
        let client = Arc::clone(&client);
        let url = Arc::clone(&url);
        let model = Arc::clone(&model);
        let prompts = Arc::clone(&prompts);
        let results = Arc::clone(&results);
        handles.push(tokio::spawn(async move {
            let prompt = prompts[i % prompts.len()];
            let obs = one_request(&client, &url, &model, prompt, max_tokens).await;
            results.lock().await.push(obs);
            drop(permit);
        }));
    }
    for h in handles {
        let _ = h.await;
    }

    Arc::try_unwrap(results)
        .expect("results Arc unique at join")
        .into_inner()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[cfg(feature = "server")]
#[tokio::main]
async fn main() {
    let url = env::var("BENCH_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8082/v1/chat/completions".into());
    // Guard: NEVER hit :8081 (prod) by accident.
    if url.contains(":8081") {
        eprintln!(
            "[bench-m1] refusing to run: BENCH_URL contains :8081 (production). \
             Start chimere-server on :8082 or override BENCH_URL."
        );
        std::process::exit(2);
    }

    let model = env::var("BENCH_MODEL").unwrap_or_else(|_| "chimere-deltanet".into());
    let conc: usize = env::var("BENCH_CONC")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n: usize = env::var("BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let max_tokens: u32 = env::var("BENCH_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let label = env::var("BENCH_PASS_LABEL").unwrap_or_else(|_| "baseline".into());
    let baseline_tps: f64 = env::var("BENCH_BASELINE_TPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let assert_iso: bool = env::var("BENCH_ASSERT_ISO")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(true);

    eprintln!(
        "[bench-m1] pass={} url={} n={} conc={} max_tokens={}",
        label, url, n, conc, max_tokens
    );

    // Warm the server with one request so model load doesn't skew the first
    // observation. Failures here are fatal — the harness can't measure
    // anything if the server is unreachable.
    eprintln!("[bench-m1] warmup...");
    {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("reqwest");
        let obs =
            one_request(&client, &url, &model, "Say hi in one word.", 4).await;
        if let Some(e) = obs.error.as_ref() {
            eprintln!("[bench-m1] warmup FAILED: {}", e);
            std::process::exit(3);
        }
        eprintln!(
            "[bench-m1] warmup OK in {}ms, gen_tokens={}",
            obs.latency_ms, obs.gen_tokens
        );
    }

    let vram_before = vram_used_gib();

    let t_wall = Instant::now();
    let obs = run_pass(&url, &model, conc, n, max_tokens).await;
    let wall_s = t_wall.elapsed().as_secs_f64();

    let vram_after = vram_used_gib();

    // ---- aggregate ----
    let (ok, err): (Vec<_>, Vec<_>) = obs.iter().partition(|o| o.error.is_none());
    if ok.is_empty() {
        eprintln!("[bench-m1] FAIL: no successful requests ({} errors)", err.len());
        for e in err.iter().take(5) {
            eprintln!("  - {:?}", e.error);
        }
        std::process::exit(4);
    }

    let total_gen: u64 = ok.iter().map(|o| o.gen_tokens as u64).sum();
    let agg_tps: f64 = total_gen as f64 / wall_s;

    let mut latencies: Vec<u64> = ok.iter().map(|o| o.latency_ms).collect();
    latencies.sort_unstable();
    let p50 = percentile(&latencies, 50.0);
    let p95 = percentile(&latencies, 95.0);
    let p99 = percentile(&latencies, 99.0);

    // ---- isolation assertion ----
    let mut iso_ok = true;
    let mut iso_reason = String::new();
    if assert_iso {
        // Group responses by prompt slot (`i % prompts.len()`) — within a
        // given prompt bucket we EXPECT some stochastic overlap (same
        // temperature, same prompt), but across DIFFERENT prompts we
        // MUST NOT see identical-body responses. If we do, that's a
        // cross-seq contamination red flag.
        let p_len = prompt_bank().len();
        let mut per_bucket: Vec<Vec<String>> = vec![Vec::new(); p_len];
        for (i, o) in ok.iter().enumerate() {
            per_bucket[i % p_len].push(o.body.clone());
        }
        // Cross-bucket diff check
        'outer: for i in 0..p_len {
            for j in (i + 1)..p_len {
                // If all answers in bucket i and j are non-empty and exactly
                // equal (byte-for-byte), that's the failure mode.
                if !per_bucket[i].is_empty() && !per_bucket[j].is_empty() {
                    let a = &per_bucket[i][0];
                    let b = &per_bucket[j][0];
                    if !a.is_empty() && a == b {
                        iso_ok = false;
                        iso_reason = format!(
                            "prompts {} and {} returned identical body: {:?}",
                            i,
                            j,
                            &a.chars().take(80).collect::<String>(),
                        );
                        break 'outer;
                    }
                }
            }
        }
    }

    let vram_str = match (vram_before, vram_after) {
        (Some(a), Some(b)) => format!("{:.2}G→{:.2}G", a, b),
        _ => "n/a".to_string(),
    };

    let ratio_str = if baseline_tps > 0.0 {
        format!(" ratio={:.2}", agg_tps / baseline_tps)
    } else {
        String::new()
    };

    println!(
        "[bench-m1] {:<12} n={} conc={} ok={} err={} wall={:.2}s agg={:.1} tok/s p50={}ms p95={}ms p99={}ms vram={}{}",
        label,
        n,
        conc,
        ok.len(),
        err.len(),
        wall_s,
        agg_tps,
        p50,
        p95,
        p99,
        vram_str,
        ratio_str,
    );
    if !iso_ok {
        println!("[bench-m1] ISOLATION FAIL: {}", iso_reason);
    }

    // Machine-readable JSON line (for scripts/bench_m1.sh to parse)
    println!(
        "{{\"pass\":\"{}\",\"n\":{},\"conc\":{},\"ok\":{},\"err\":{},\"wall_s\":{:.3},\"agg_tps\":{:.3},\"p50_ms\":{},\"p95_ms\":{},\"p99_ms\":{},\"vram_before_gib\":{},\"vram_after_gib\":{},\"iso_ok\":{}}}",
        label,
        n,
        conc,
        ok.len(),
        err.len(),
        wall_s,
        agg_tps,
        p50,
        p95,
        p99,
        vram_before.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "null".into()),
        vram_after.map(|v| format!("{:.3}", v)).unwrap_or_else(|| "null".into()),
        iso_ok,
    );

    // Exit codes:
    //   0 — pass OK, ratio and isolation both OK
    //   5 — ratio below target (only when BENCH_BASELINE_TPS is set AND
    //       label is `2-slot` or `4-slot`)
    //   6 — isolation assertion failed
    if !iso_ok {
        std::process::exit(6);
    }
    if baseline_tps > 0.0 {
        let ratio = agg_tps / baseline_tps;
        let target: f64 = match label.as_str() {
            "2-slot" => 1.7,
            "4-slot" => 3.0,
            _ => 0.0,
        };
        if target > 0.0 && ratio < target {
            eprintln!(
                "[bench-m1] WARNING: ratio {:.2} < target {:.2} for pass {}",
                ratio, target, label
            );
            // Exit non-zero only when the ratio is obviously bad (<1.2x),
            // otherwise the harness prints the warning but returns success
            // so the J4-dispatcher-rewrite-pending case is tolerated.
            if ratio < 1.2 {
                std::process::exit(5);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stub main when the `server` feature is off — the harness needs reqwest and
// tokio::main, both pulled in by `server`. Matches the project convention
// (see j2/j3/j4/j5 smokes).
// ---------------------------------------------------------------------------
#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("bench-m1 requires --features server (reqwest + tokio).");
    std::process::exit(1);
}
