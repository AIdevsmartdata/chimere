// chimere_sampler.cpp — Thin C++ wrapper around common_sampler for FFI.
// Avoids copying 993KB of logits per token to Rust.
// Instead: sample directly in C++ and return just the token ID.
//
// Compile: g++ -c -O3 -std=c++17 -I<ik_llama>/include -I<ik_llama>/common
// Link: -lcommon -lllama

#include "llama.h"
#include "sampling.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>

extern "C" {

// Opaque sampler handle
typedef struct common_sampler chimere_sampler_t;

/// Create a sampler with Qwen3.5 optimal params.
chimere_sampler_t * chimere_sampler_init(
    const void * model,  // llama_model *
    float temperature,
    float top_p,
    int top_k,
    float min_p,
    float presence_penalty,
    float dry_multiplier,
    float dry_base,
    int dry_min_length,
    int dry_penalty_last_n
) {
    common_params_sampling params;
    params.temp = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.min_p = min_p;
    params.penalty_present = presence_penalty;
    params.dry_multiplier = dry_multiplier;
    params.dry_base = dry_base;
    params.dry_allowed_length = dry_min_length;
    params.dry_penalty_last_n = dry_penalty_last_n;

    auto * smpl = common_sampler_init((const llama_model *)model, params);
    if (!smpl) {
        fprintf(stderr, "[chimere_sampler] init failed\n");
    }
    return smpl;
}

/// Sample one token from the last decode's logits. Returns token ID.
/// This is the hot path — replaces 993KB logits copy + Rust sort.
int32_t chimere_sampler_sample(
    chimere_sampler_t * smpl,
    void * ctx,  // llama_context *
    int idx      // batch index (usually 0 or -1 for last)
) {
    return (int32_t)common_sampler_sample(smpl, (llama_context *)ctx, idx);
}

/// Accept a token (update sampler state for repetition penalties etc.)
void chimere_sampler_accept(
    chimere_sampler_t * smpl,
    void * ctx,
    int32_t token
) {
    common_sampler_accept(smpl, (llama_context *)ctx, (llama_token)token, true);
}

/// Result struct for sample_with_logprobs
struct chimere_logprob_result {
    int32_t token_id;
    int32_t n_top;
    int32_t top_tokens[5];
    float   top_logprobs[5];
};

/// Sample one token AND return top-5 logprobs.
/// This is the key function for ABF — think_router needs entropy/confidence.
///
/// BUG FIX (2026-03-26): Previously called common_sampler_sample() first, then
/// common_sampler_get_candidates() — but after sampling, the candidate array is
/// filtered/consumed by the sampler chain (top-k, top-p, min-p, temperature),
/// leaving only 1 candidate with transformed logits → all logprobs were 0.0.
///
/// Fix: capture raw logits from llama_get_logits_ith() BEFORE sampling, find
/// top-5 by scanning the full vocab, compute log-softmax ourselves. Then sample
/// normally. This gives true pre-sampling log-probabilities.
void chimere_sampler_sample_with_logprobs(
    chimere_sampler_t * smpl,
    void * ctx,
    int idx,
    struct chimere_logprob_result * result
) {
    llama_context * lctx = (llama_context *)ctx;

    // ---- Step 1: Capture raw logits BEFORE sampling modifies them ----
    const int n_vocab = llama_n_vocab(llama_get_model(lctx));
    float * logits = llama_get_logits_ith(lctx, idx);

    // Build a fast lookup for logit biases (Engram, </think> suppression).
    // We apply biases during the scan rather than mutating the logits array,
    // because common_sampler_sample() will also apply them — mutating here
    // would cause double-application of finite biases.
    const auto & bias_map = smpl->params.logit_bias;

    // Find top-5 by partial sort on bias-adjusted logits (pre-temperature).
    // We use indices to avoid copying the full 993KB logits array.
    struct tok_logit { int32_t id; float logit; };
    tok_logit top5[5];
    int n_top = 0;

    for (int i = 0; i < n_vocab; i++) {
        float l = logits[i];
        // Apply logit bias if present (e.g. -inf for </think> suppression)
        auto bit = bias_map.find(i);
        if (bit != bias_map.end()) {
            l += bit->second;
        }
        if (n_top < 5) {
            top5[n_top++] = {i, l};
            // Bubble the new entry into sorted position (descending)
            for (int j = n_top - 1; j > 0 && top5[j].logit > top5[j-1].logit; j--) {
                std::swap(top5[j], top5[j-1]);
            }
        } else if (l > top5[4].logit) {
            top5[4] = {i, l};
            // Re-sort the last entry into position
            for (int j = 4; j > 0 && top5[j].logit > top5[j-1].logit; j--) {
                std::swap(top5[j], top5[j-1]);
            }
        }
    }

    // Compute log-softmax over the FULL vocab for accurate log-probabilities.
    // log_softmax(x_i) = x_i - log(sum(exp(x_j - max_x)))
    // We use the max from top5[0] which is the global max (with bias applied).
    float max_logit = top5[0].logit;

    // Numerically stable log-sum-exp over full vocab.
    // First pass: sum over raw logits (fast, no hash lookups for 151K tokens).
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        sum_exp += exp((double)(logits[i] - max_logit));
    }
    // Correction pass: adjust for biased tokens. For each biased token, subtract
    // its raw contribution and add its biased contribution. This is O(|bias_map|)
    // which is typically < 10 entries.
    for (auto it = bias_map.begin(); it != bias_map.end(); it++) {
        int tok = it->first;
        if (tok < 0 || tok >= n_vocab) continue;
        float raw = logits[tok];
        float biased = raw + it->second;
        sum_exp -= exp((double)(raw - max_logit));
        sum_exp += exp((double)(biased - max_logit));
    }
    if (sum_exp <= 0.0) sum_exp = 1e-10;  // safety
    float log_sum_exp = max_logit + (float)log(sum_exp);

    result->n_top = n_top;  // always 5 unless vocab < 5
    for (int i = 0; i < n_top; i++) {
        result->top_tokens[i]   = top5[i].id;
        result->top_logprobs[i] = top5[i].logit - log_sum_exp;
    }
    for (int i = n_top; i < 5; i++) {
        result->top_tokens[i]   = -1;
        result->top_logprobs[i] = -100.0f;
    }

    // ---- Step 2: Sample normally (this modifies cur_p, but we already have our logprobs) ----
    llama_token token = common_sampler_sample(smpl, lctx, idx);
    result->token_id = (int32_t)token;
}

/// Set a logit bias (e.g., suppress </think> token in response mode).
void chimere_sampler_set_logit_bias(
    chimere_sampler_t * smpl,
    int32_t token_id,
    float bias
) {
    smpl->params.logit_bias[token_id] = bias;
}

/// Clear all logit biases.
void chimere_sampler_clear_logit_bias(chimere_sampler_t * smpl) {
    smpl->params.logit_bias.clear();
}

/// Set Engram logit biases from sparse (token_id, bias) pairs.
/// MERGES with existing biases (does NOT clear — preserves </think> suppression).
/// Called once per token before sampling, with n-gram lookup predictions.
void chimere_sampler_set_engram_bias(
    chimere_sampler_t * smpl,
    const int32_t * token_ids,
    const float * biases,
    int n_entries
) {
    // Remove previous Engram biases (marked by being in the "engram range")
    // but keep any manually set biases (like </think> = -inf).
    // Strategy: Engram biases are additive, small values (0.1-2.0).
    // Manual biases are large (-inf). So we just overwrite Engram entries.
    for (int i = 0; i < n_entries; i++) {
        auto it = smpl->params.logit_bias.find(token_ids[i]);
        if (it != smpl->params.logit_bias.end() && it->second <= -1e6f) {
            // This token has a manual suppression (-inf) — don't override
            continue;
        }
        smpl->params.logit_bias[token_ids[i]] = biases[i];
    }
}

/// Clear only Engram biases (keep manual biases like </think> suppression).
void chimere_sampler_clear_engram_bias(chimere_sampler_t * smpl) {
    // Remove entries with small bias values (Engram), keep large negative (manual)
    for (auto it = smpl->params.logit_bias.begin(); it != smpl->params.logit_bias.end(); ) {
        if (it->second > -1e6f) {
            it = smpl->params.logit_bias.erase(it);
        } else {
            ++it;
        }
    }
}

/// Reset sampler state (new conversation).
void chimere_sampler_reset(chimere_sampler_t * smpl) {
    common_sampler_reset(smpl);
}

/// Free sampler.
void chimere_sampler_free(chimere_sampler_t * smpl) {
    common_sampler_free(smpl);
}

} // extern "C"
