//! # M1 J7 — stress harness CLI (external, talks to running chimere-server)
//!
//! Usage:
//!   cargo run --release --features server --bin j7-stress -- \
//!       --concurrency 4 --requests 100 [--host 127.0.0.1:8081]
//!
//! Fires `requests` concurrent HTTP POSTs at `/v1/chat/completions`
//! (non-streaming), with up to `concurrency` in-flight at any time.
//! Reports:
//!   - P50 / P95 / P99 latency (per-request, wall-clock seconds)
//!   - aggregate generation tok/s (sum of completion_tokens / total walltime)
//!   - error count and error sample
//!
//! Intentionally lives outside the test framework because it:
//!   1. needs a real running server (network I/O, not an in-process ctx),
//!   2. runs for 10s-10min depending on scale,
//!   3. should be cheap to re-run manually with different knobs.
//!
//! The harness is self-contained: no `reqwest`, no extra deps — it speaks
//! HTTP/1.1 `POST` with `Content-Length` over a raw `tokio::net::TcpStream`,
//! then parses the JSON response with `serde_json`. The only crates
//! imported are `tokio` and `serde_json`, both already in Cargo.toml.
//!
//! ## Operational notes
//!
//! - The target server MUST be up and serving at `--host`. The smoke tests
//!   in `tests/*.rs` run in-process; *this* binary does not load a model,
//!   so it does not compete for VRAM.
//! - Point it at the `m1-multislot` branch build once J4-rewrite lands;
//!   the numbers on the single-slot prod server are the baseline to beat.
//! - Keep `--requests` ≤ 200 on local dev unless you want to wait.
//!
//! ## Expected output (example, illustrative)
//!
//! ```text
//! [j7-stress] concurrency=4 requests=100 host=127.0.0.1:8081
//! [j7-stress] starting…
//! [j7-stress] done: 100/100 success, 0 errors, wall 42.3s
//! [j7-stress] latencies (s): p50=1.41 p95=2.87 p99=3.11
//! [j7-stress] aggregate tok/s: 118.2 (completion), 164.5 (prompt+completion)
//! ```

use std::env;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Semaphore;

// ---------------------------------------------------------------------------
// CLI arg parsing — tiny, zero-dep.
// ---------------------------------------------------------------------------
struct Args {
    concurrency: usize,
    requests: usize,
    host: String,
    model: String,
    max_tokens: u32,
    prompt: String,
}

impl Args {
    fn parse_from_env() -> Result<Self, String> {
        let mut concurrency: usize = 4;
        let mut requests: usize = 20;
        let mut host: String = "127.0.0.1:8081".into();
        let mut model: String = "chimere".into();
        let mut max_tokens: u32 = 64;
        let mut prompt: String =
            "Write one short sentence about multi-slot batching.".into();

        let mut args: Vec<String> = env::args().collect();
        // Drop argv[0].
        if !args.is_empty() { args.remove(0); }

        let mut i = 0;
        while i < args.len() {
            let k = args[i].as_str();
            let v = args.get(i + 1).cloned();
            let get = || -> Result<String, String> {
                v.clone().ok_or_else(|| format!("missing value after {}", k))
            };
            match k {
                "--concurrency" | "-c" => {
                    concurrency = get()?.parse().map_err(|e| format!("concurrency: {}", e))?;
                    i += 2;
                }
                "--requests" | "-n" => {
                    requests = get()?.parse().map_err(|e| format!("requests: {}", e))?;
                    i += 2;
                }
                "--host" => {
                    host = get()?;
                    i += 2;
                }
                "--model" => {
                    model = get()?;
                    i += 2;
                }
                "--max-tokens" => {
                    max_tokens = get()?.parse().map_err(|e| format!("max-tokens: {}", e))?;
                    i += 2;
                }
                "--prompt" => {
                    prompt = get()?;
                    i += 2;
                }
                "--help" | "-h" => {
                    eprintln!("{}", USAGE);
                    std::process::exit(0);
                }
                other => {
                    return Err(format!("unknown argument {:?}\n\n{}", other, USAGE));
                }
            }
        }
        if concurrency == 0 {
            return Err("concurrency must be >= 1".into());
        }
        if requests == 0 {
            return Err("requests must be >= 1".into());
        }
        Ok(Args { concurrency, requests, host, model, max_tokens, prompt })
    }
}

const USAGE: &str = "\
Usage:
  j7-stress [flags]

Flags:
  --concurrency, -c  N   in-flight requests cap (default 4)
  --requests,    -n  M   total requests to send (default 20)
  --host         HOST    host:port of the chimere-server (default 127.0.0.1:8081)
  --model        NAME    OpenAI-style model field (default \"chimere\")
  --max-tokens   N       response budget per request (default 64)
  --prompt       STR     user prompt text (default demo sentence)
  --help,        -h      show this message";

// ---------------------------------------------------------------------------
// Per-request result
// ---------------------------------------------------------------------------
#[derive(Debug)]
struct ReqResult {
    latency: Duration,
    prompt_tokens: u32,
    completion_tokens: u32,
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// HTTP/1.1 POST client (non-streaming). Minimal, self-contained.
// ---------------------------------------------------------------------------
async fn post_chat_completions(
    host: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> io::Result<(u32, u32, String)> {
    // Build the JSON body. We manually serialize to avoid pulling serde
    // derive macros into this binary's dep graph.
    let body_json = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "user", "content": prompt },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": false,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    let body = serde_json::to_vec(&body_json)?;

    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\n\
         Host: {}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         Accept: application/json\r\n\
         \r\n",
        host, body.len(),
    );

    let mut stream = TcpStream::connect(host).await?;
    stream.write_all(request.as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.flush().await?;

    // Read the full response (Connection: close → server closes after body).
    let mut raw = Vec::with_capacity(8192);
    stream.read_to_end(&mut raw).await?;

    // Split status line + headers from body at the first CRLF CRLF.
    let sep = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "no CRLFCRLF in response")
    })?;
    let head = &raw[..sep];
    let body = &raw[sep + 4..];

    // Status check: parse "HTTP/1.1 <code> ..." from the first line.
    let head_str = std::str::from_utf8(head).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("non-utf8 head: {}", e))
    })?;
    let status_line = head_str.lines().next().unwrap_or("");
    let code: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if !(200..300).contains(&code) {
        let sample = std::str::from_utf8(body).unwrap_or("<binary>");
        return Err(io::Error::other(format!(
            "HTTP {}: {}",
            code,
            &sample[..sample.len().min(200)]
        )));
    }

    // Response may be chunked — but with Content-Length and
    // Connection: close we typically get a raw payload. Handle both:
    // if the body starts with a hex length token followed by \r\n,
    // treat it as chunked and strip the framing.
    let parsed_body: Vec<u8> = if head_str
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        dechunk(body).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "chunked body malformed")
        })?
    } else {
        body.to_vec()
    };

    // Parse JSON to extract usage.{prompt_tokens,completion_tokens}.
    let v: serde_json::Value = serde_json::from_slice(&parsed_body).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("json: {}", e))
    })?;
    let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    Ok((prompt_tokens, completion_tokens, content))
}

/// Strip HTTP/1.1 chunked framing. Returns `None` on malformed input.
fn dechunk(body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0usize;
    while i < body.len() {
        // Read the hex size line ending with \r\n.
        let line_end = body[i..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .map(|p| i + p)?;
        let size_hex = std::str::from_utf8(&body[i..line_end]).ok()?;
        // Some servers include a chunk extension ";foo=bar" — split at ';'.
        let size_hex = size_hex.split(';').next().unwrap_or("");
        let size = usize::from_str_radix(size_hex.trim(), 16).ok()?;
        i = line_end + 2;
        if size == 0 {
            break;
        }
        if i + size > body.len() {
            return None;
        }
        out.extend_from_slice(&body[i..i + size]);
        i += size + 2; // skip trailing CRLF after chunk
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Percentile helpers
// ---------------------------------------------------------------------------
fn percentile(sorted_micros: &[u64], p: f64) -> f64 {
    if sorted_micros.is_empty() {
        return 0.0;
    }
    // Nearest-rank method. Plenty accurate for a few hundred samples.
    let rank = (p * sorted_micros.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted_micros.len() - 1);
    sorted_micros[idx] as f64 / 1_000_000.0
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let args = match Args::parse_from_env() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[j7-stress] error: {}", e);
            std::process::exit(2);
        }
    };
    eprintln!(
        "[j7-stress] concurrency={} requests={} host={} model={} max_tokens={}",
        args.concurrency, args.requests, args.host, args.model, args.max_tokens,
    );

    let sem = Arc::new(Semaphore::new(args.concurrency));
    let completed = Arc::new(AtomicUsize::new(0));
    let total_prompt_toks = Arc::new(AtomicU64::new(0));
    let total_completion_toks = Arc::new(AtomicU64::new(0));

    let host = Arc::new(args.host.clone());
    let model = Arc::new(args.model.clone());
    let prompt = Arc::new(args.prompt.clone());

    let mut handles = Vec::with_capacity(args.requests);
    let wall_start = Instant::now();

    for i in 0..args.requests {
        let sem = Arc::clone(&sem);
        let host = Arc::clone(&host);
        let model = Arc::clone(&model);
        let prompt = Arc::clone(&prompt);
        let completed = Arc::clone(&completed);
        let total_p = Arc::clone(&total_prompt_toks);
        let total_c = Arc::clone(&total_completion_toks);
        let max_tokens = args.max_tokens;

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.expect("semaphore");
            let t0 = Instant::now();
            let res = post_chat_completions(&host, &model, &prompt, max_tokens).await;
            let latency = t0.elapsed();
            let result = match res {
                Ok((p, c, _content)) => {
                    total_p.fetch_add(p as u64, Ordering::Relaxed);
                    total_c.fetch_add(c as u64, Ordering::Relaxed);
                    ReqResult { latency, prompt_tokens: p, completion_tokens: c, error: None }
                }
                Err(e) => {
                    ReqResult {
                        latency, prompt_tokens: 0, completion_tokens: 0,
                        error: Some(e.to_string()),
                    }
                }
            };
            let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 10 == 0 || done == 1 {
                eprintln!(
                    "[j7-stress] progress: {} completed (latest latency {:.3}s, req #{})",
                    done, latency.as_secs_f64(), i,
                );
            }
            result
        }));
    }

    let mut results: Vec<ReqResult> = Vec::with_capacity(args.requests);
    for h in handles {
        match h.await {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("[j7-stress] task panicked: {}", e);
            }
        }
    }
    let wall = wall_start.elapsed();

    // Separate successes from errors.
    let mut ok_latencies_micros: Vec<u64> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    for r in &results {
        if r.error.is_none() {
            ok_latencies_micros.push(r.latency.as_micros() as u64);
        } else if let Some(e) = &r.error {
            errors.push(e.clone());
        }
    }
    ok_latencies_micros.sort_unstable();

    let p50 = percentile(&ok_latencies_micros, 0.50);
    let p95 = percentile(&ok_latencies_micros, 0.95);
    let p99 = percentile(&ok_latencies_micros, 0.99);
    let p_sum = total_prompt_toks.load(Ordering::Relaxed);
    let c_sum = total_completion_toks.load(Ordering::Relaxed);
    let wall_s = wall.as_secs_f64().max(1e-6);
    let completion_tokps = c_sum as f64 / wall_s;
    let total_tokps = (p_sum + c_sum) as f64 / wall_s;

    println!();
    println!("=== J7 stress summary ===");
    println!(
        "  requests           : {} success, {} errors (of {})",
        ok_latencies_micros.len(),
        errors.len(),
        args.requests,
    );
    println!("  wall-clock total   : {:.2} s", wall_s);
    println!("  latency p50 / p95 / p99 : {:.3}s  {:.3}s  {:.3}s", p50, p95, p99);
    println!("  prompt tokens      : {}", p_sum);
    println!("  completion tokens  : {}", c_sum);
    println!("  aggregate tok/s    : {:.1} completion / {:.1} total", completion_tokps, total_tokps);
    if !errors.is_empty() {
        println!("  first error sample :");
        for e in errors.iter().take(3) {
            println!("    - {}", e);
        }
    }
    // Exit non-zero if ALL requests failed (helpful for CI).
    if ok_latencies_micros.is_empty() {
        std::process::exit(1);
    }
}
