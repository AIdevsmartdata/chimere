# chimere-server metrics — catalog

Status: draft (polish pass, April 2026). Implemented in
`chimere-server/src/metrics.rs`, exposed via:

- **`GET /metrics`** — Prometheus scrape endpoint.
  `Content-Type: text/plain; version=0.0.4; charset=utf-8`.
- **`GET /v1/status`** — structured JSON consumed by ODO and the
  OpenClaw dashboard; mirrors `/metrics` plus `engine`, `model`, and
  `scheduler` mode.

## Design principles

- **No new crate deps.** Atomics from `std::sync::atomic` + one
  `std::sync::Mutex` around a tiny ring buffer.
- **Hot path is lock-free.** The six counter hooks (`inc_request_ok`,
  `inc_request_error`, `add_prompt_tokens`, `add_gen_tokens`,
  `observe_ttft_ms`) are one atomic RMW each (the TTFT one takes a
  short-lived `Mutex` on a 100-element `Vec<u32>`).
- **Scrape cost ≈ 40 µs** on an idle server (format + quantile sort
  over at most 100 samples).
- **Additive.** Existing clients of `/health` and
  `/v1/chat/completions` are unchanged.

## Metric catalog

| Name                                   | Type     | Description                                                           |
|---------------------------------------|----------|-----------------------------------------------------------------------|
| `chimere_requests_total{status=...}`  | counter  | Total chat completions by terminal status (`ok`, `error`).            |
| `chimere_prompt_tokens_total`         | counter  | Prompt tokens tokenised across all accepted requests.                 |
| `chimere_gen_tokens_total`            | counter  | Tokens produced by the model across all requests.                     |
| `chimere_slot_occupancy`              | gauge    | Slots currently serving a request.                                    |
| `chimere_slot_pool_size`              | gauge    | Total slots configured on this server instance.                       |
| `chimere_admission_queue_depth`       | gauge    | Requests parked in the admission mpsc channel.                        |
| `chimere_ttft_seconds{quantile=...}`  | summary  | Time to first token (p50, p90, p95, p99) over last 100 requests.      |
| `chimere_ttft_seconds_sum`            | summary  | Sum of the observed TTFTs in the ring buffer (seconds).               |
| `chimere_ttft_seconds_count`          | summary  | Number of samples in the TTFT ring (≤100).                            |

Naming mirrors the vLLM / llama.cpp conventions where possible:

- vLLM has `vllm_num_requests_running` / `vllm_num_requests_waiting` —
  we use `chimere_slot_occupancy` / `chimere_admission_queue_depth` for
  the same concepts (mapped to our slot model).
- vLLM's `vllm_time_to_first_token_seconds` is a histogram; we use a
  summary because 100 samples give exact quantiles cheaper than
  bucketed histograms for a polish pass.
- llama.cpp's `n_prompt_tokens_processed` / `n_tokens_predicted` map
  cleanly to `chimere_prompt_tokens_total` / `chimere_gen_tokens_total`.

## Example scrape

```sh
$ curl -s http://127.0.0.1:8081/metrics
# HELP chimere_requests_total Total chat-completion requests by terminal status
# TYPE chimere_requests_total counter
chimere_requests_total{status="ok"} 1234
chimere_requests_total{status="error"} 2

# HELP chimere_prompt_tokens_total Prompt tokens tokenised across all accepted requests
# TYPE chimere_prompt_tokens_total counter
chimere_prompt_tokens_total 567890

# HELP chimere_gen_tokens_total Tokens produced by the model across all requests
# TYPE chimere_gen_tokens_total counter
chimere_gen_tokens_total 234567

# HELP chimere_slot_occupancy Slots currently serving a request
# TYPE chimere_slot_occupancy gauge
chimere_slot_occupancy 2

# HELP chimere_slot_pool_size Total slots configured on this server instance
# TYPE chimere_slot_pool_size gauge
chimere_slot_pool_size 4

# HELP chimere_admission_queue_depth Requests parked in the admission mpsc channel
# TYPE chimere_admission_queue_depth gauge
chimere_admission_queue_depth 0

# HELP chimere_ttft_seconds Time to first token per streaming request (ring buffer quantiles, cap=100)
# TYPE chimere_ttft_seconds summary
chimere_ttft_seconds{quantile="0.50"} 0.120000
chimere_ttft_seconds{quantile="0.90"} 0.410000
chimere_ttft_seconds{quantile="0.95"} 0.450000
chimere_ttft_seconds{quantile="0.99"} 0.830000
chimere_ttft_seconds_sum 42.118000
chimere_ttft_seconds_count 100
```

## Example `/v1/status`

```json
{
  "status": "ok",
  "engine": "chimere-deltanet",
  "model": "chimere-v3-ramp",
  "scheduler": "native-multislot",
  "metrics": {
    "requests_ok": 1234,
    "requests_error": 2,
    "prompt_tokens_total": 567890,
    "gen_tokens_total": 234567,
    "slot_occupancy": 2,
    "slot_pool_size": 4,
    "admission_queue_depth": 0,
    "ttft": {
      "count": 100,
      "p50_ms": 120,
      "p90_ms": 410,
      "p95_ms": 450,
      "p99_ms": 830
    }
  }
}
```

The `scheduler` field is one of:

- `"single-slot"` — legacy behaviour (`CHIMERE_MULTISLOT` unset or `1`).
- `"legacy-multislot"` — closure-based J2 path
  (`CHIMERE_MULTISLOT>=2`, `CHIMERE_MULTISLOT_NATIVE` not set).
- `"native-multislot"` — J4 rewrite driver (recommended in 2026).

## Dashboards & alerts

Suggested PromQL starters:

```promql
# Throughput — tokens per second over 1 minute.
rate(chimere_gen_tokens_total[1m])

# Error rate.
rate(chimere_requests_total{status="error"}[5m])
  /
rate(chimere_requests_total[5m])

# P95 TTFT — fire alert above 1s sustained for 5 min.
chimere_ttft_seconds{quantile="0.95"} > 1

# Slot saturation — fire when occupancy == pool_size for 2 min.
chimere_slot_occupancy == chimere_slot_pool_size

# Admission queue pressure — any depth for >30 s indicates tuning needed.
chimere_admission_queue_depth > 0
```

Grafana dashboard idea (one row, five panels):

1. Requests/sec stacked by status.
2. Tokens/sec (prompt + generated).
3. TTFT p50/p95/p99 lines.
4. Slot occupancy vs pool size (overlay).
5. Admission queue depth bar.

## Prometheus scrape config

Drop into `/etc/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: chimere-server
    scrape_interval: 15s
    metrics_path: /metrics
    static_configs:
      - targets: ['127.0.0.1:8081']
```

For systems with multiple model endpoints (qwen35-custom on 8081, and a
swap target on 8082 when GLM-OCR is loaded), add one target per port.

## Non-goals (future work)

- Per-route latency histograms (handler-level instrumentation).
- Per-slot stats (would require reaching into the NativeDriver pool,
  currently owned by the driver thread).
- Per-engram hit/miss counters.
- OpenTelemetry / W3C trace-context headers (Chimere is offline-first —
  tracing is lower priority than scrape-able metrics).
- HDR histograms for TTFT. The 100-sample ring is a polish-grade
  compromise; a real histogram crate would replace it when the metrics
  set grows past this first cut.

## References

- Prometheus exposition format (text 0.0.4):
  <https://prometheus.io/docs/instrumenting/exposition_formats/>
- vLLM metrics reference:
  <https://docs.vllm.ai/en/latest/usage/metrics.html>
- llama.cpp `/metrics` endpoint (b8125+):
  <https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#api-endpoints>
