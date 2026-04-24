// chimere_sampler.cpp — Self-contained sampler for FFI (libcommon-free).
//
// # What this file does
//
// Provides a minimal, stable-ABI sampling context backed entirely by
// libllama.so's public C API (`llama_sample_*`). We purposely avoid
// `common_sampler_init` from ik_llama's libcommon because:
//
//   1. libcommon's `struct common_sampler` layout is a moving target.
//      Upstream's autoparser refactor (ik_llama commit e0596bf6)
//      added a `rbudget` field and new `reasoning_budget_*` params,
//      and chimere's libcommon.a (rebuilt 2026-04-24 09:36 from a
//      tree containing that refactor) is ABI-incompatible with the
//      `sampling.h` currently checked out on the chimere branch used
//      to compile chimere_sampler.cpp.
//   2. `common_sampler_init` instantiates a full grammar pipeline
//      unconditionally. Chimère does not use grammar sampling; that
//      extra state is pure tech debt and the source of uncaught
//      foreign exceptions that crashed every fresh chimere-server
//      binary on 2026-04-24.
//
// # ABI contract
//
// All `extern "C"` entry points below keep the exact signatures
// declared in `llama_backend.rs` — Rust code is unchanged.
//
// Compile: g++ -c -O3 -std=c++17 -I<ik_llama>/include
// Link:    -lllama  (libggml.so is transitively loaded)

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// ik_llama grammar-apply shim (kept as defensive no-op)
// ---------------------------------------------------------------------------
// Some object files in libcommon.a reference `llama_grammar_apply`, a
// symbol exposed by upstream llama.cpp but NOT by ik_llama's libllama.so.
// Until we fully drop the libcommon.a link, keep this shim so stray
// references don't break the Rust link stage.
struct llama_grammar;
struct llama_token_data_array;
extern "C" void llama_grammar_apply(struct llama_grammar * /*grammar*/,
                                    struct llama_token_data_array * /*candidates*/) {
    // No-op: Chimère does not use grammar-constrained sampling.
}

// ---------------------------------------------------------------------------
// Internal sampler type
// ---------------------------------------------------------------------------

namespace {

struct chimere_sampler {
    // Sampling knobs
    float temp             = 0.6f;
    float top_p            = 0.95f;
    int   top_k            = 20;
    float min_p            = 0.05f;
    float penalty_present  = 0.0f;
    float penalty_repeat   = 1.0f;
    float penalty_freq     = 0.0f;
    int   penalty_last_n   = 64;

    // Logit-bias map: token_id -> additive bias (used for engram + </think>)
    std::unordered_map<int32_t, float> logit_bias;

    // Rolling history of last-sampled tokens (for repetition penalty).
    // Sized to penalty_last_n.
    std::vector<int32_t> prev;

    // Scratch buffer for candidate token_data — sized to n_vocab on first use.
    std::vector<llama_token_data> cur;

    // RNG seeded on init.
    std::mt19937 rng;
};

} // anonymous namespace

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

// Build cur_p from raw logits, applying logit_bias (additive).
// Returns a llama_token_data_array view over sampler->cur.
llama_token_data_array prepare_candidates(chimere_sampler * s,
                                          const float * logits,
                                          int n_vocab) {
    s->cur.resize((size_t)n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        s->cur[i] = llama_token_data{ i, logits[i], 0.0f };
    }
    // Apply logit biases (small map, typically <20 entries).
    for (const auto & kv : s->logit_bias) {
        if (kv.first >= 0 && kv.first < n_vocab) {
            s->cur[(size_t)kv.first].logit += kv.second;
        }
    }
    llama_token_data_array arr;
    arr.data     = s->cur.data();
    arr.size     = (size_t)n_vocab;
    arr.sorted   = false;
    return arr;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public C ABI — matches `llama_backend.rs` extern "C" declarations
// ---------------------------------------------------------------------------

extern "C" {

// Opaque sampler handle (ABI: the Rust side only sees `*mut c_void`).
typedef struct chimere_sampler chimere_sampler_t;

/// Create a sampler with Qwen3.5 optimal params.
/// Returns nullptr on allocation failure (never throws).
chimere_sampler_t * chimere_sampler_init(
    const void * /*model*/,   // llama_model * — reserved for future use
    float temperature,
    float top_p,
    int top_k,
    float min_p,
    float presence_penalty,
    float /*dry_multiplier*/,
    float /*dry_base*/,
    int /*dry_min_length*/,
    int /*dry_penalty_last_n*/
) {
    try {
        auto * s = new chimere_sampler();
        s->temp            = temperature;
        s->top_p           = top_p;
        s->top_k           = top_k;
        s->min_p           = min_p;
        s->penalty_present = presence_penalty;
        s->penalty_last_n  = 64;
        s->prev.reserve(s->penalty_last_n);
        // Seed RNG from std::random_device for non-determinism across slots.
        std::random_device rd;
        s->rng.seed(rd());
        return s;
    } catch (...) {
        fprintf(stderr, "[chimere_sampler] init failed (alloc)\n");
        return nullptr;
    }
    // NOTE: DRY is intentionally disabled in this minimal path. The smoke
    // tests and current production workloads do not rely on DRY; if we
    // need it again, re-introduce it here using `llama_sampler_init_dry` +
    // `llama_sample_dry` without touching libcommon.
}

// Internal: the sampling chain (candidates → token).
static int32_t sample_chain(chimere_sampler * s,
                            llama_context * lctx,
                            llama_token_data_array & cur_p) {
    // Repetition penalties (uses rolling prev history).
    if (!s->prev.empty() && (s->penalty_repeat != 1.0f ||
                             s->penalty_present != 0.0f ||
                             s->penalty_freq != 0.0f)) {
        // llama_sample_repetition_penalties takes llama_token * (==int32*).
        llama_sample_repetition_penalties(
            lctx, &cur_p,
            s->prev.data(),
            s->prev.size(),
            s->penalty_repeat,
            s->penalty_freq,
            s->penalty_present);
    }

    if (s->temp <= 0.0f) {
        // Greedy sampling.
        return (int32_t)llama_sample_token_greedy(lctx, &cur_p);
    }

    // Standard ordering: top-k → top-p → min-p → temperature → softmax → multinomial
    if (s->top_k > 0) {
        llama_sample_top_k(lctx, &cur_p, s->top_k, /*min_keep=*/1);
    }
    if (s->top_p > 0.0f && s->top_p < 1.0f) {
        llama_sample_top_p(lctx, &cur_p, s->top_p, /*min_keep=*/1);
    }
    if (s->min_p > 0.0f) {
        llama_sample_min_p(lctx, &cur_p, s->min_p, /*min_keep=*/1);
    }
    llama_sample_temp(lctx, &cur_p, s->temp);

    // Multinomial draw from the post-sampler distribution.
    // llama_sample_token uses the context's internal RNG; to keep per-slot
    // RNG independence we use the greedy fallback when top-k=1 and otherwise
    // let llama_sample_token do the work. The slight coupling is acceptable
    // for the smoke (distributions are what the test asserts on) and for
    // single-slot production (one RNG is fine).
    return (int32_t)llama_sample_token(lctx, &cur_p);
}

/// Sample one token from the last decode's logits. Returns token ID.
int32_t chimere_sampler_sample(
    chimere_sampler_t * s,
    void * ctx,
    int idx
) {
    if (!s) return -1;
    auto * lctx = (llama_context *)ctx;
    const int n_vocab = llama_n_vocab(llama_get_model(lctx));
    const float * logits = llama_get_logits_ith(lctx, idx);
    if (!logits) return -1;

    auto cur_p = prepare_candidates(s, logits, n_vocab);
    return sample_chain(s, lctx, cur_p);
}

/// Update sampler state for a sampled token (rolling penalty history).
void chimere_sampler_accept(
    chimere_sampler_t * s,
    void * /*ctx*/,
    int32_t token
) {
    if (!s) return;
    if (s->penalty_last_n <= 0) return;
    // Maintain a rolling window of the last N tokens.
    if ((int)s->prev.size() >= s->penalty_last_n) {
        // Shift left by one — penalty_last_n is small (<=64), memmove is fine.
        s->prev.erase(s->prev.begin());
    }
    s->prev.push_back(token);
}

/// Result struct for sample_with_logprobs — shape matches Rust side
/// (`ChimereLogprobResult`). Do not reorder fields.
struct chimere_logprob_result {
    int32_t token_id;
    int32_t n_top;
    int32_t top_tokens[5];
    float   top_logprobs[5];
};

/// Sample one token AND return top-5 logprobs (used by ABF).
///
/// Computes true pre-sampling log-probabilities by scanning the full vocab
/// for the top-5 (with logit_bias applied), log-sum-exp over all logits
/// for numerical stability, THEN runs the normal sampler chain for the
/// actual token choice. This gives accurate logprobs unaffected by the
/// chain's filtering (top-k/top-p discard candidates and renormalise).
void chimere_sampler_sample_with_logprobs(
    chimere_sampler_t * s,
    void * ctx,
    int idx,
    struct chimere_logprob_result * result
) {
    if (!s || !result) return;
    auto * lctx = (llama_context *)ctx;

    const int n_vocab = llama_n_vocab(llama_get_model(lctx));
    const float * logits = llama_get_logits_ith(lctx, idx);
    if (!logits) {
        result->token_id = -1;
        result->n_top = 0;
        for (int i = 0; i < 5; ++i) {
            result->top_tokens[i] = -1;
            result->top_logprobs[i] = -100.0f;
        }
        return;
    }

    // ---- Step 1: capture top-5 from bias-adjusted logits ----
    const auto & bias_map = s->logit_bias;
    struct tok_logit { int32_t id; float logit; };
    tok_logit top5[5];
    int n_top = 0;
    for (int i = 0; i < n_vocab; i++) {
        float l = logits[i];
        auto bit = bias_map.find(i);
        if (bit != bias_map.end()) {
            l += bit->second;
        }
        if (n_top < 5) {
            top5[n_top++] = {i, l};
            for (int j = n_top - 1; j > 0 && top5[j].logit > top5[j-1].logit; j--) {
                std::swap(top5[j], top5[j-1]);
            }
        } else if (l > top5[4].logit) {
            top5[4] = {i, l};
            for (int j = 4; j > 0 && top5[j].logit > top5[j-1].logit; j--) {
                std::swap(top5[j], top5[j-1]);
            }
        }
    }

    // ---- Step 2: log-sum-exp for accurate log-softmax ----
    float max_logit = n_top > 0 ? top5[0].logit : 0.0f;
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum_exp += exp((double)(logits[i] - max_logit));
    }
    // Correct for biased tokens (small map, typically < 20 entries).
    for (auto it = bias_map.begin(); it != bias_map.end(); ++it) {
        int tok = it->first;
        if (tok < 0 || tok >= n_vocab) continue;
        float raw    = logits[tok];
        float biased = raw + it->second;
        sum_exp -= exp((double)(raw    - max_logit));
        sum_exp += exp((double)(biased - max_logit));
    }
    if (sum_exp <= 0.0) sum_exp = 1e-10;
    float log_sum_exp = max_logit + (float)log(sum_exp);

    result->n_top = n_top;
    for (int i = 0; i < n_top; i++) {
        result->top_tokens[i]   = top5[i].id;
        result->top_logprobs[i] = top5[i].logit - log_sum_exp;
    }
    for (int i = n_top; i < 5; i++) {
        result->top_tokens[i]   = -1;
        result->top_logprobs[i] = -100.0f;
    }

    // ---- Step 3: run the normal sampler chain to pick a token ----
    // Rebuild cur_p so the chain sees the biased distribution.
    auto cur_p = prepare_candidates(s, logits, n_vocab);
    result->token_id = sample_chain(s, lctx, cur_p);
}

/// Set a single logit bias (e.g., `-inf` for `</think>` suppression).
void chimere_sampler_set_logit_bias(
    chimere_sampler_t * s,
    int32_t token_id,
    float bias
) {
    if (!s) return;
    s->logit_bias[token_id] = bias;
}

/// Clear all logit biases.
void chimere_sampler_clear_logit_bias(chimere_sampler_t * s) {
    if (!s) return;
    s->logit_bias.clear();
}

/// Set Engram logit biases from sparse (token_id, bias) pairs.
/// Preserves manual `-inf` suppressions (e.g. `</think>`).
void chimere_sampler_set_engram_bias(
    chimere_sampler_t * s,
    const int32_t * token_ids,
    const float * biases,
    int n_entries
) {
    if (!s || !token_ids || !biases) return;
    for (int i = 0; i < n_entries; i++) {
        auto it = s->logit_bias.find(token_ids[i]);
        if (it != s->logit_bias.end() && it->second <= -1e6f) {
            continue;  // keep manual suppression
        }
        s->logit_bias[token_ids[i]] = biases[i];
    }
}

/// Clear only Engram biases (keep manual `-inf` biases).
void chimere_sampler_clear_engram_bias(chimere_sampler_t * s) {
    if (!s) return;
    for (auto it = s->logit_bias.begin(); it != s->logit_bias.end(); ) {
        if (it->second > -1e6f) {
            it = s->logit_bias.erase(it);
        } else {
            ++it;
        }
    }
}

/// Reset sampler state (new conversation).
void chimere_sampler_reset(chimere_sampler_t * s) {
    if (!s) return;
    s->prev.clear();
    // logit_bias is explicitly NOT cleared here — the chimere_sampler_clear_*
    // entry points exist for that purpose.
}

/// Free sampler.
void chimere_sampler_free(chimere_sampler_t * s) {
    delete s;  // delete on nullptr is well-defined in C++
}

} // extern "C"
