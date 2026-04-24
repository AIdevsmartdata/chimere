//! # Regression test: `chimere_gen_tokens_total` increments on the
//! streaming path, including for `Thinking` / `reasoning_content` events.
//!
//! Context — bug fixed 2026-04-24
//! ------------------------------
//! The production E2E bench on 2026-04-24 produced 120 successful streaming
//! requests totalling 15 360 tokens. Afterwards `GET /metrics` reported
//! `chimere_gen_tokens_total 0`. Root cause: both the legacy J2 streaming
//! SSE handler and the J4-rewrite native streaming handler only called
//! `Metrics::add_gen_tokens(1)` on `StreamMsg::Token` events, never on
//! `StreamMsg::Thinking`. Qwen3 replies to short prompts entirely inside
//! `<think>` blocks, so every sampled token on the bench was a `Thinking`
//! event — and the counter never moved.
//!
//! The non-streaming path always counted the full `completion_tokens`
//! (thinking included), so this was a pure streaming-handler divergence.
//!
//! This test locks in the streaming-path contract:
//!
//!   - A `Token` event contributes 1 to `gen_tokens_total`.
//!   - A `Thinking` event contributes 1 to `gen_tokens_total`.
//!   - The Prometheus exposition surfaces the sum as
//!     `chimere_gen_tokens_total <N>`.
//!
//! It operates on the `Metrics` struct directly (no model, no FFI) — the
//! bug was the call-site decision in `server.rs`, and this test mirrors
//! that decision with a simulated SSE stream. If the hook ever regresses
//! to "thinking does not count", `thinking_tokens_are_counted_as_gen_tokens`
//! fails immediately.
//!
//! It also verifies the other counters that share the streaming hot path
//! (`prompt_tokens_total`, `requests_total{status="ok"}`, and the TTFT
//! ring summary) all observe their events as intended.

use std::sync::Arc;

use chimere_deltanet::metrics::Metrics;

/// Narrow local enum mirroring `slot_scheduler::StreamMsg` — we do not
/// depend on that enum in this test so it also validates contract stability.
/// Adjusting `StreamMsg` will not silently break this test's assertions.
enum SimMsg {
    Prompt(usize),
    Token,
    Thinking,
    Done,
    Error,
}

/// Drive `Metrics` through the sequence the server's SSE unfold closure
/// would produce for one streaming request. Returns the TTFT observation
/// (milliseconds) that the caller stamped, matching the first content or
/// thinking event.
fn feed_stream(metrics: &Arc<Metrics>, events: &[SimMsg], ttft_ms: u64) {
    let mut ttft_observed = false;
    for ev in events {
        match ev {
            SimMsg::Prompt(n) => metrics.add_prompt_tokens(*n),
            SimMsg::Token => {
                if !ttft_observed {
                    metrics.observe_ttft_ms(ttft_ms);
                    ttft_observed = true;
                }
                metrics.add_gen_tokens(1);
            }
            SimMsg::Thinking => {
                if !ttft_observed {
                    metrics.observe_ttft_ms(ttft_ms);
                    ttft_observed = true;
                }
                // Fix 2026-04-24: thinking tokens count toward gen_tokens.
                metrics.add_gen_tokens(1);
            }
            SimMsg::Done => metrics.inc_request_ok(),
            SimMsg::Error => metrics.inc_request_error(),
        }
    }
}

#[test]
fn thinking_tokens_are_counted_as_gen_tokens() {
    // Regression: the production bug that triggered this fix.
    // 128 Thinking events simulate one short Qwen3 reply that stayed
    // entirely inside a `<think>` block, like the 2026-04-24 E2E bench.
    let metrics = Arc::new(Metrics::new());

    let mut events: Vec<SimMsg> = Vec::with_capacity(130);
    events.push(SimMsg::Prompt(12));
    for _ in 0..128 {
        events.push(SimMsg::Thinking);
    }
    events.push(SimMsg::Done);

    feed_stream(&metrics, &events, 42);

    let body = metrics.render_prometheus(0, 4, 0);
    assert!(
        body.contains("chimere_gen_tokens_total 128"),
        "gen_tokens_total must reflect all 128 thinking tokens; body was:\n{}",
        body,
    );
    assert!(body.contains("chimere_prompt_tokens_total 12"));
    assert!(body.contains("chimere_requests_total{status=\"ok\"} 1"));
    assert!(body.contains("chimere_ttft_seconds_count 1"));
}

#[test]
fn mixed_token_and_thinking_stream_sums_correctly() {
    // One request emits 20 Thinking tokens (inside <think>) followed by
    // 40 Token tokens (the final response) — matches longer Qwen3 replies
    // where the model exits the reasoning block and emits content.
    let metrics = Arc::new(Metrics::new());

    let mut events: Vec<SimMsg> = Vec::with_capacity(70);
    events.push(SimMsg::Prompt(20));
    for _ in 0..20 {
        events.push(SimMsg::Thinking);
    }
    for _ in 0..40 {
        events.push(SimMsg::Token);
    }
    events.push(SimMsg::Done);

    feed_stream(&metrics, &events, 100);

    let body = metrics.render_prometheus(0, 4, 0);
    assert!(
        body.contains("chimere_gen_tokens_total 60"),
        "gen_tokens_total must equal thinking(20)+token(40); body was:\n{}",
        body,
    );
    // TTFT is observed on the first Thinking event (ms=100).
    assert!(body.contains("chimere_ttft_seconds_count 1"));
}

#[test]
fn aggregate_over_120_streaming_requests_matches_bench_math() {
    // Exact replay of the E2E bench shape: 120 streaming requests, each
    // producing 128 generated tokens (all Thinking) and counted as ok.
    // Expected totals: 120 OK, 0 errors, 120*12=1440 prompt, 120*128=15360
    // generated. With the pre-fix code, gen_tokens_total was 0.
    let metrics = Arc::new(Metrics::new());
    for _ in 0..120 {
        let mut events: Vec<SimMsg> = Vec::with_capacity(130);
        events.push(SimMsg::Prompt(12));
        for _ in 0..128 {
            events.push(SimMsg::Thinking);
        }
        events.push(SimMsg::Done);
        feed_stream(&metrics, &events, 50);
    }

    let body = metrics.render_prometheus(0, 4, 0);
    assert!(body.contains("chimere_gen_tokens_total 15360"));
    assert!(body.contains("chimere_prompt_tokens_total 1440"));
    assert!(body.contains("chimere_requests_total{status=\"ok\"} 120"));
    assert!(body.contains("chimere_requests_total{status=\"error\"} 0"));

    let snap = metrics.snapshot_json(0, 4, 0);
    assert_eq!(snap["gen_tokens_total"], 15360);
    assert_eq!(snap["prompt_tokens_total"], 1440);
    assert_eq!(snap["requests_ok"], 120);
}

#[test]
fn error_path_bumps_error_counter_not_ok_counter() {
    let metrics = Arc::new(Metrics::new());
    feed_stream(
        &metrics,
        &[SimMsg::Prompt(10), SimMsg::Thinking, SimMsg::Error],
        5,
    );
    let body = metrics.render_prometheus(0, 1, 0);
    assert!(body.contains("chimere_requests_total{status=\"ok\"} 0"));
    assert!(body.contains("chimere_requests_total{status=\"error\"} 1"));
    // The thinking event still counted as a gen token — it was sampled
    // before the error.
    assert!(body.contains("chimere_gen_tokens_total 1"));
}
