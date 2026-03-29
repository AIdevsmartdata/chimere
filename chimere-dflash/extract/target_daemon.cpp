/**
 * target_daemon.cpp — Resident stdin/stdout daemon for online speculative decoding
 *
 * Loads a Qwen3.5-35B-A3B GGUF model once, stays resident, and serves binary
 * protocol requests over stdin/stdout. Uses the same eval callback as
 * extract_hidden_states.cpp to capture hidden states from "post_moe" tensors.
 *
 * Binary protocol: [cmd:u8][payload_len:u32_le][payload...] →
 *                  [status:u8][payload_len:u32_le][payload...]
 *
 * Usage:
 *   ./target_daemon \
 *     -m ~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/model.gguf \
 *     -ngl 99 -ot '.ffn_.*_exps.=CPU' \
 *     --layers 2,11,20,29,37 \
 *     --capture-routing --routing-layers 20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39 \
 *     --flash-attn on
 *
 * MoE routing capture (CMD_GET_ROUTING = 0x0A):
 *   Intercepts "ffn_moe_topk-{L}" tensors during eval for routing_layers.
 *   Shape: [n_expert_used=8, n_tokens], type I32.
 *   Stored per eval as uint8[n_routing_layers][n_tokens][8].
 *   CMD_GET_ROUTING response: [n_layers:u16][n_tokens:u16][data:uint8[n_layers*n_tokens*8]]
 */

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ── Protocol constants ──

static constexpr uint8_t CMD_TOKENIZE   = 0x01;
static constexpr uint8_t CMD_EVAL_FULL  = 0x02;
static constexpr uint8_t CMD_EVAL_INCR  = 0x03;
static constexpr uint8_t CMD_TRIM_KV    = 0x04;
static constexpr uint8_t CMD_CLEAR_KV   = 0x05;
static constexpr uint8_t CMD_DETOKENIZE = 0x06;
static constexpr uint8_t CMD_EVAL_LAST     = 0x07;
static constexpr uint8_t CMD_STATE_SAVE    = 0x08;
static constexpr uint8_t CMD_STATE_RESTORE = 0x09;
static constexpr uint8_t CMD_GET_ROUTING   = 0x0A;
static constexpr uint8_t CMD_STATE_SAVE_GDN    = 0x0B;
static constexpr uint8_t CMD_STATE_RESTORE_GDN = 0x0C;
static constexpr uint8_t CMD_QUIT          = 0xFF;

static constexpr uint8_t STATUS_OK    = 0x00;
static constexpr uint8_t STATUS_ERROR = 0x01;

// ── Callback data (reused from extract_hidden_states.cpp) ──

struct extract_callback_data {
    std::vector<uint8_t>              temp_buf;
    std::set<int>                     target_layers;
    std::string                       tensor_name;
    std::map<int, std::vector<float>> captured;
    int64_t                           hidden_dim = 0;
    int64_t                           seq_len    = 0;
    bool                              last_only  = false;  // only capture last position

    // ── MoE routing capture (optional, flag-gated) ──
    // Enabled when routing_layers is non-empty.
    // Intercepts "ffn_moe_topk-{L}" tensors: shape [n_expert_used, n_tokens], type I32.
    // routing_buf[layer_idx] = vector of uint8, length = n_tokens * n_expert_used (= n_tokens * 8).
    //   Layout per token t: bytes [t*8 .. t*8+7] = expert indices chosen (uint8, fit in [0,255]).
    bool                                   capture_routing = false;
    std::set<int>                          routing_layers;
    std::map<int, std::vector<uint8_t>>    routing_buf;    // layer_idx -> uint8[n_tokens * 8]
    int64_t                                routing_n_tokens  = 0;  // n_tokens from last routing capture
    int64_t                                routing_n_experts = 8;  // n_expert_used (default 8)
    std::vector<uint8_t>                   routing_temp_buf; // scratch for GPU tensor readback
};

// ── Eval callback ──
//
// Handles two tensor families:
//   1. Hidden states: "{tensor_name}-{L}" (e.g. "post_moe-20") — F32/F16/BF16
//   2. MoE routing:   "ffn_moe_topk-{L}"                       — I32, [n_expert_used, n_tokens]
//
// Routing tensors are only intercepted when cb->capture_routing is true and
// layer L is in cb->routing_layers.
//
// Note on ffn_moe_topk tensor layout (from llama-graph.cpp build_moe_ffn):
//   ggml_argsort_top_k returns a view of the full argsort result.
//   ne[0] = n_expert_used (8 for Qwen3.5-35B-A3B)
//   ne[1] = n_tokens
//   type  = GGML_TYPE_I32
//   Each element is an expert index in [0, n_experts).  For Qwen3.5-35B-A3B,
//   n_experts = 256, so indices fit in uint8.

static bool extract_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = static_cast<extract_callback_data *>(user_data);

    std::string name(t->name);
    size_t dash_pos = name.rfind('-');
    if (dash_pos == std::string::npos) {
        if (ask) return false;
        return true;
    }

    std::string base_name = name.substr(0, dash_pos);

    int layer_idx = -1;
    try {
        layer_idx = std::stoi(name.substr(dash_pos + 1));
    } catch (...) {
        if (ask) return false;
        return true;
    }

    // ── Branch 1: MoE routing (ffn_moe_topk-{L}) ──
    //
    // ffn_moe_topk is built by ggml_argsort_top_k → ggml_view_4d of the full
    // n_experts-wide argsort result.  Its strides are:
    //   nb[0] = sizeof(int32_t) = 4
    //   nb[1] = n_experts * sizeof(int32_t)  (stride across tokens in parent buffer)
    //
    // This means the tensor is NOT contiguous in memory for n_expert_used < n_experts.
    // We must use the nb[] strides to access elements correctly.
    if (cb->capture_routing && base_name == "ffn_moe_topk") {
        if (cb->routing_layers.find(layer_idx) == cb->routing_layers.end()) {
            if (ask) return false;
            return true;
        }
        if (ask) return true;

        // ffn_moe_topk: ne[0]=n_expert_used, ne[1]=n_tokens, type=I32
        int64_t n_expert_used = t->ne[0];  // should be 8
        int64_t n_tokens      = t->ne[1];
        // Strides in bytes: nb[0] = 4, nb[1] = n_experts * 4 (non-contiguous view)
        size_t  nb0           = t->nb[0];  // byte stride between expert indices
        size_t  nb1           = t->nb[1];  // byte stride between tokens

        cb->routing_n_tokens  = n_tokens;
        cb->routing_n_experts = n_expert_used;

        // Determine host vs device; for views, check view_src->buffer
        ggml_backend_buffer_t buf_to_check = t->view_src ? t->view_src->buffer : t->buffer;
        bool is_host = (buf_to_check != nullptr) && ggml_backend_buffer_is_host(buf_to_check);

        const uint8_t * base_ptr;
        if (is_host) {
            base_ptr = static_cast<const uint8_t *>(t->data);
        } else {
            // Read the raw strided region into temp_buf.
            // ggml_nbytes uses nb[] so it covers exactly the strided region.
            size_t n_bytes = ggml_nbytes(t);
            cb->routing_temp_buf.resize(n_bytes);
            ggml_backend_tensor_get(t, cb->routing_temp_buf.data(), 0, n_bytes);
            base_ptr = cb->routing_temp_buf.data();
        }

        // Pack strided [n_expert_used, n_tokens] I32 into dense uint8[n_tokens * n_expert_used].
        // Element at (e, tok): byte offset = tok * nb1 + e * nb0
        std::vector<uint8_t> routing_data(n_tokens * n_expert_used);
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            for (int64_t e = 0; e < n_expert_used; e++) {
                const uint8_t * elem_ptr = base_ptr + tok * nb1 + e * nb0;
                int32_t expert_id;
                memcpy(&expert_id, elem_ptr, sizeof(int32_t));
                routing_data[tok * n_expert_used + e] = static_cast<uint8_t>(expert_id & 0xFF);
            }
        }

        cb->routing_buf[layer_idx] = std::move(routing_data);
        return true;
    }

    // ── Branch 2: Hidden states ──
    if (base_name != cb->tensor_name) {
        if (ask) return false;
        return true;
    }

    if (cb->target_layers.find(layer_idx) == cb->target_layers.end()) {
        if (ask) return false;
        return true;
    }

    if (ask) {
        return true;
    }

    int64_t ne0 = t->ne[0];
    int64_t ne1 = t->ne[1];
    cb->hidden_dim = ne0;
    cb->seq_len    = ne1;

    size_t n_bytes = ggml_nbytes(t);
    ggml_backend_buffer_t hs_buf = t->view_src ? t->view_src->buffer : t->buffer;
    bool is_host = (hs_buf != nullptr) && ggml_backend_buffer_is_host(hs_buf);

    uint8_t * data_ptr;
    if (is_host) {
        data_ptr = static_cast<uint8_t *>(t->data);
    } else {
        cb->temp_buf.resize(n_bytes);
        ggml_backend_tensor_get(t, cb->temp_buf.data(), 0, n_bytes);
        data_ptr = cb->temp_buf.data();
    }

    // In last_only mode, only copy the last position (ne0 floats)
    // Otherwise copy all positions (ne0 * ne1 floats)
    int64_t copy_start = cb->last_only ? (ne1 - 1) : 0;
    int64_t copy_count = cb->last_only ? 1 : ne1;
    int64_t n_elems = ne0 * copy_count;

    std::vector<float> float_data(n_elems);

    if (t->type == GGML_TYPE_F32) {
        const float * src = reinterpret_cast<const float *>(data_ptr) + copy_start * ne0;
        memcpy(float_data.data(), src, n_elems * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(data_ptr) + copy_start * ne0;
        for (int64_t i = 0; i < n_elems; i++) {
            float_data[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(data_ptr) + copy_start * ne0;
        for (int64_t i = 0; i < n_elems; i++) {
            float_data[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else {
        fprintf(stderr, "[daemon] WARNING: unsupported tensor type %s for %s\n",
                ggml_type_name(t->type), t->name);
        return true;
    }

    cb->captured[layer_idx] = std::move(float_data);
    return true;
}

// ── Robust binary I/O helpers ──

static bool read_exact(void * buf, size_t n) {
    size_t total = 0;
    uint8_t * p = static_cast<uint8_t *>(buf);
    while (total < n) {
        size_t r = fread(p + total, 1, n - total, stdin);
        if (r == 0) return false;  // EOF or error
        total += r;
    }
    return true;
}

static bool write_exact(const void * buf, size_t n) {
    size_t total = 0;
    const uint8_t * p = static_cast<const uint8_t *>(buf);
    while (total < n) {
        size_t w = fwrite(p + total, 1, n - total, stdout);
        if (w == 0) return false;
        total += w;
    }
    return true;
}

static bool read_u8(uint8_t & v)   { return read_exact(&v, 1); }
static bool read_u32(uint32_t & v) { return read_exact(&v, 4); }
static bool read_i32(int32_t & v)  { return read_exact(&v, 4); }

static bool write_u8(uint8_t v)    { return write_exact(&v, 1); }
static bool write_u32(uint32_t v)  { return write_exact(&v, 4); }

static bool send_response(uint8_t status, const void * payload, uint32_t len) {
    if (!write_u8(status))  return false;
    if (!write_u32(len))    return false;
    if (len > 0 && !write_exact(payload, len)) return false;
    fflush(stdout);
    return true;
}

static bool send_ok(const void * payload = nullptr, uint32_t len = 0) {
    return send_response(STATUS_OK, payload, len);
}

static bool send_error(const std::string & msg) {
    return send_response(STATUS_ERROR, msg.data(), (uint32_t)msg.size());
}

// ── Parse layers string "2,11,20,29,37" ──

static std::set<int> parse_layers(const std::string & s) {
    std::set<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        try {
            result.insert(std::stoi(token));
        } catch (...) {
            fprintf(stderr, "[daemon] WARNING: invalid layer '%s'\n", token.c_str());
        }
    }
    return result;
}

// ── Eval helper (shared by EVAL_FULL and EVAL_INCR) ──

static bool do_eval(
    llama_context * ctx,
    const llama_vocab * vocab,
    extract_callback_data & cb_data,
    const std::vector<int32_t> & tokens,
    const std::vector<int32_t> & layer_ids,
    int kv_pos,
    int n_tokens
) {
    cb_data.captured.clear();
    cb_data.hidden_dim    = 0;
    cb_data.seq_len       = 0;
    cb_data.last_only     = false;  // capture all positions
    cb_data.routing_buf.clear();
    cb_data.routing_n_tokens  = 0;

    // Set target layers for this request
    cb_data.target_layers.clear();
    for (int32_t lid : layer_ids) {
        cb_data.target_layers.insert(lid);
    }

    // Build batch with per-position logits
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], kv_pos + i, {0}, true);
    }

    int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        send_error("llama_decode failed, ret=" + std::to_string(ret));
        return false;
    }

    // Collect argmax logits
    int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int32_t> argmax_tokens(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        float * logits = llama_get_logits_ith(ctx, i);
        int best = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > logits[best]) best = v;
        }
        argmax_tokens[i] = best;
    }

    // Build response payload:
    //   [n_layers:u32][seq_len:u32][hidden_dim:u32]
    //   [hidden_data:f32[n_layers * seq_len * hidden_dim]]
    //   [logits:i32[seq_len]]
    int n_layers   = (int)layer_ids.size();
    int hidden_dim = (int)cb_data.hidden_dim;

    if (hidden_dim == 0 && !layer_ids.empty()) {
        send_error("no hidden states captured (hidden_dim=0)");
        return false;
    }

    // Verify all requested layers were captured
    for (int32_t lid : layer_ids) {
        if (cb_data.captured.find(lid) == cb_data.captured.end()) {
            send_error("missing layer " + std::to_string(lid));
            return false;
        }
    }

    size_t header_bytes  = 3 * sizeof(uint32_t);
    size_t hidden_bytes  = (size_t)n_layers * n_tokens * hidden_dim * sizeof(float);
    size_t logits_bytes  = (size_t)n_tokens * sizeof(int32_t);
    size_t total_payload = header_bytes + hidden_bytes + logits_bytes;

    std::vector<uint8_t> payload(total_payload);
    uint8_t * p = payload.data();

    // Header
    uint32_t nl = (uint32_t)n_layers;
    uint32_t sl = (uint32_t)n_tokens;
    uint32_t hd = (uint32_t)hidden_dim;
    memcpy(p, &nl, 4); p += 4;
    memcpy(p, &sl, 4); p += 4;
    memcpy(p, &hd, 4); p += 4;

    // Hidden states — ordered by layer_ids request order
    for (int32_t lid : layer_ids) {
        const auto & data = cb_data.captured[lid];
        // data has hidden_dim * cb_data.seq_len floats, we need hidden_dim * n_tokens
        size_t copy_bytes = (size_t)n_tokens * hidden_dim * sizeof(float);
        memcpy(p, data.data(), copy_bytes);
        p += copy_bytes;
    }

    // Argmax logits
    memcpy(p, argmax_tokens.data(), logits_bytes);

    return send_ok(payload.data(), (uint32_t)total_payload);
}

// ── Eval helper for last-position-only extraction ──

static bool do_eval_last_pos(
    llama_context * ctx,
    const llama_vocab * vocab,
    extract_callback_data & cb_data,
    const std::vector<int32_t> & tokens,
    const std::vector<int32_t> & layer_ids,
    int kv_pos,
    int n_tokens
) {
    cb_data.captured.clear();
    cb_data.hidden_dim        = 0;
    cb_data.seq_len           = 0;
    cb_data.last_only         = true;  // only capture last position in callback
    cb_data.routing_buf.clear();
    cb_data.routing_n_tokens  = 0;

    cb_data.target_layers.clear();
    for (int32_t lid : layer_ids) {
        cb_data.target_layers.insert(lid);
    }

    // Build batch — only request logits for last position
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], kv_pos + i, {0}, i == n_tokens - 1);
    }

    int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        send_error("llama_decode failed, ret=" + std::to_string(ret));
        return false;
    }

    // Get argmax of last position only
    int n_vocab = llama_vocab_n_tokens(vocab);
    float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
    int32_t argmax_token = 0;
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > logits[argmax_token]) argmax_token = v;
    }

    int n_layers   = (int)layer_ids.size();
    int hidden_dim = (int)cb_data.hidden_dim;

    if (hidden_dim == 0 && !layer_ids.empty()) {
        send_error("no hidden states captured (hidden_dim=0)");
        return false;
    }

    for (int32_t lid : layer_ids) {
        if (cb_data.captured.find(lid) == cb_data.captured.end()) {
            send_error("missing layer " + std::to_string(lid));
            return false;
        }
    }

    // Response: [n_layers:u32][1:u32][hidden_dim:u32]
    //           [hidden_data:f32[n_layers * 1 * hidden_dim]]  (LAST position only)
    //           [argmax:i32[1]]
    size_t header_bytes  = 3 * sizeof(uint32_t);
    size_t hidden_bytes  = (size_t)n_layers * hidden_dim * sizeof(float);
    size_t logits_bytes  = sizeof(int32_t);
    size_t total_payload = header_bytes + hidden_bytes + logits_bytes;

    std::vector<uint8_t> payload(total_payload);
    uint8_t * p = payload.data();

    uint32_t nl = (uint32_t)n_layers;
    uint32_t sl = 1;  // only 1 position
    uint32_t hd = (uint32_t)hidden_dim;
    memcpy(p, &nl, 4); p += 4;
    memcpy(p, &sl, 4); p += 4;
    memcpy(p, &hd, 4); p += 4;

    // Copy last position's hidden state — callback already captured only 1 position
    for (int32_t lid : layer_ids) {
        const auto & data = cb_data.captured[lid];
        // data is [hidden_dim] (1 position, last_only mode)
        memcpy(p, data.data(), hidden_dim * sizeof(float));
        p += hidden_dim * sizeof(float);
    }

    memcpy(p, &argmax_token, sizeof(int32_t));

    return send_ok(payload.data(), (uint32_t)total_payload);
}

// ── Main ──

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [llama-args] --layers 2,11,20,29,37\n", prog);
    fprintf(stderr, "\nStdin/stdout binary protocol daemon for speculative decoding.\n");
    fprintf(stderr, "\nExtra arguments:\n");
    fprintf(stderr, "  --layers L1,L2,...         Layer indices to capture (required)\n");
    fprintf(stderr, "  --tensor NAME              Tensor name to capture (default: post_moe)\n");
    fprintf(stderr, "  --capture-routing          Enable MoE routing capture (default: off)\n");
    fprintf(stderr, "  --routing-layers L1,L2,... MoE layers to capture routing for (default: same as --layers)\n");
    fprintf(stderr, "\nAll standard llama.cpp arguments (-m, -ngl, -ot, etc.) are also supported.\n");
}

int main(int argc, char ** argv) {
    // Unbuffered binary I/O
    setbuf(stdin,  NULL);
    setbuf(stdout, NULL);

    std::set<int>  target_layers;
    std::string    tensor_name = "post_moe";
    bool           capture_routing  = false;
    std::set<int>  routing_layers;   // populated after arg parse

    // Pre-parse custom args
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    std::string routing_layers_str;  // deferred until after --layers is parsed

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--layers" && i + 1 < argc) {
            target_layers = parse_layers(argv[++i]);
        } else if (arg == "--tensor" && i + 1 < argc) {
            tensor_name = argv[++i];
        } else if (arg == "--capture-routing") {
            capture_routing = true;
        } else if (arg == "--routing-layers" && i + 1 < argc) {
            routing_layers_str = argv[++i];
        } else if (arg == "--help-daemon") {
            print_usage(argv[0]);
            return 0;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    // Resolve routing layers: explicit list, or fall back to target_layers
    if (capture_routing) {
        if (!routing_layers_str.empty()) {
            routing_layers = parse_layers(routing_layers_str);
        } else {
            routing_layers = target_layers;
        }
    }

    if (target_layers.empty()) {
        print_usage(argv[0]);
        fprintf(stderr, "\nERROR: --layers is required.\n");
        return 1;
    }

    // Add dummy prompt for common_params_parse
    bool has_prompt = false;
    for (auto * a : filtered_argv) {
        if (std::string(a) == "-p" || std::string(a) == "--prompt") {
            has_prompt = true;
            break;
        }
    }
    if (!has_prompt) {
        filtered_argv.push_back(const_cast<char *>("-p"));
        filtered_argv.push_back(const_cast<char *>("dummy"));
    }

    // Setup callback data
    extract_callback_data cb_data;
    cb_data.target_layers   = target_layers;
    cb_data.tensor_name     = tensor_name;
    cb_data.capture_routing = capture_routing;
    cb_data.routing_layers  = routing_layers;

    // Parse llama params
    common_params params;
    int fargc = static_cast<int>(filtered_argv.size());
    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.cb_eval           = extract_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    fprintf(stderr, "[daemon] Loading model...\n");
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        fprintf(stderr, "[daemon] ERROR: failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    fprintf(stderr, "[daemon] Model loaded. Target layers: ");
    for (int l : target_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "\n");
    if (capture_routing) {
        fprintf(stderr, "[daemon] Routing capture ENABLED. Routing layers: ");
        for (int l : routing_layers) fprintf(stderr, "%d ", l);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "[daemon] Entering command loop (binary protocol on stdin/stdout).\n");

    int kv_pos = 0;

    // State save/restore buffer (single slot) — full state
    std::vector<uint8_t> saved_state;
    int                  saved_kv_pos = 0;

    // GDN-only save/restore buffer — recurrent states only (~2 MB vs ~63 MB)
    std::vector<uint8_t> saved_gdn_state;
    int                  saved_gdn_kv_pos = 0;

    // ── Command loop ──
    for (;;) {
        uint8_t  cmd;
        uint32_t payload_len;

        if (!read_u8(cmd)) {
            fprintf(stderr, "[daemon] stdin closed, exiting.\n");
            break;
        }
        if (!read_u32(payload_len)) {
            fprintf(stderr, "[daemon] failed to read payload_len, exiting.\n");
            break;
        }

        // Read full payload
        std::vector<uint8_t> payload(payload_len);
        if (payload_len > 0 && !read_exact(payload.data(), payload_len)) {
            fprintf(stderr, "[daemon] failed to read payload (%u bytes), exiting.\n", payload_len);
            break;
        }

        switch (cmd) {

        // ── TOKENIZE ──
        case CMD_TOKENIZE: {
            std::string text(payload.begin(), payload.end());
            std::vector<llama_token> tokens = common_tokenize(ctx, text, add_bos);
            std::vector<int32_t> ids(tokens.begin(), tokens.end());
            send_ok(ids.data(), (uint32_t)(ids.size() * sizeof(int32_t)));
            fprintf(stderr, "[daemon] TOKENIZE: %zu chars -> %zu tokens\n", text.size(), tokens.size());
            break;
        }

        // ── EVAL_FULL ──
        case CMD_EVAL_FULL: {
            if (payload_len < 8) {
                send_error("EVAL_FULL payload too short");
                break;
            }
            const uint8_t * p = payload.data();

            uint32_t n_tokens;
            memcpy(&n_tokens, p, 4); p += 4;

            if (payload_len < 4 + n_tokens * 4 + 4) {
                send_error("EVAL_FULL payload truncated");
                break;
            }

            std::vector<int32_t> tokens(n_tokens);
            memcpy(tokens.data(), p, n_tokens * 4); p += n_tokens * 4;

            uint32_t n_layers;
            memcpy(&n_layers, p, 4); p += 4;

            if (payload_len < 4 + n_tokens * 4 + 4 + n_layers * 4) {
                send_error("EVAL_FULL layer_ids truncated");
                break;
            }

            std::vector<int32_t> layer_ids(n_layers);
            memcpy(layer_ids.data(), p, n_layers * 4);

            // Clear KV cache
            llama_memory_clear(llama_get_memory(ctx), true);
            kv_pos = 0;

            if (do_eval(ctx, vocab, cb_data, tokens, layer_ids, kv_pos, (int)n_tokens)) {
                kv_pos = (int)n_tokens;
            }
            fprintf(stderr, "[daemon] EVAL_FULL: %u tokens, kv_pos=%d\n", n_tokens, kv_pos);
            break;
        }

        // ── EVAL_INCR ──
        case CMD_EVAL_INCR: {
            if (payload_len < 8) {
                send_error("EVAL_INCR payload too short");
                break;
            }
            const uint8_t * p = payload.data();

            uint32_t n_tokens;
            memcpy(&n_tokens, p, 4); p += 4;

            if (payload_len < 4 + n_tokens * 4 + 4) {
                send_error("EVAL_INCR payload truncated");
                break;
            }

            std::vector<int32_t> tokens(n_tokens);
            memcpy(tokens.data(), p, n_tokens * 4); p += n_tokens * 4;

            uint32_t n_layers;
            memcpy(&n_layers, p, 4); p += 4;

            if (payload_len < 4 + n_tokens * 4 + 4 + n_layers * 4) {
                send_error("EVAL_INCR layer_ids truncated");
                break;
            }

            std::vector<int32_t> layer_ids(n_layers);
            memcpy(layer_ids.data(), p, n_layers * 4);

            // Do NOT clear KV cache — append
            if (do_eval(ctx, vocab, cb_data, tokens, layer_ids, kv_pos, (int)n_tokens)) {
                kv_pos += (int)n_tokens;
            }
            fprintf(stderr, "[daemon] EVAL_INCR: %u tokens, kv_pos=%d\n", n_tokens, kv_pos);
            break;
        }

        // ── TRIM_KV ──
        case CMD_TRIM_KV: {
            if (payload_len < 4) {
                send_error("TRIM_KV needs 4-byte payload");
                break;
            }
            int32_t keep_n;
            memcpy(&keep_n, payload.data(), 4);

            if (keep_n < 0 || keep_n > kv_pos) {
                send_error("TRIM_KV keep_n out of range [0, " + std::to_string(kv_pos) + "]");
                break;
            }

            llama_memory_seq_rm(llama_get_memory(ctx), 0, keep_n, -1);
            kv_pos = keep_n;
            send_ok();
            fprintf(stderr, "[daemon] TRIM_KV: keep_n=%d, kv_pos=%d\n", keep_n, kv_pos);
            break;
        }

        // ── CLEAR_KV ──
        case CMD_CLEAR_KV: {
            llama_memory_clear(llama_get_memory(ctx), true);
            kv_pos = 0;
            send_ok();
            fprintf(stderr, "[daemon] CLEAR_KV: kv_pos=0\n");
            break;
        }

        // ── EVAL_LAST (last position only — 500× less pipe I/O) ──
        case CMD_EVAL_LAST: {
            if (payload_len < 8) {
                send_error("EVAL_LAST payload too short");
                break;
            }
            const uint8_t * p = payload.data();

            uint32_t n_tokens;
            memcpy(&n_tokens, p, 4); p += 4;

            if (payload_len < 4 + n_tokens * 4 + 4) {
                send_error("EVAL_LAST payload truncated");
                break;
            }

            std::vector<int32_t> tokens(n_tokens);
            memcpy(tokens.data(), p, n_tokens * 4); p += n_tokens * 4;

            uint32_t n_layers;
            memcpy(&n_layers, p, 4); p += 4;

            if (payload_len < 4 + n_tokens * 4 + 4 + n_layers * 4) {
                send_error("EVAL_LAST layer_ids truncated");
                break;
            }

            std::vector<int32_t> layer_ids(n_layers);
            memcpy(layer_ids.data(), p, n_layers * 4);

            // Clear KV cache (full eval)
            llama_memory_clear(llama_get_memory(ctx), true);
            kv_pos = 0;

            if (do_eval_last_pos(ctx, vocab, cb_data, tokens, layer_ids, kv_pos, (int)n_tokens)) {
                kv_pos = (int)n_tokens;
            }
            fprintf(stderr, "[daemon] EVAL_LAST: %u tokens, kv_pos=%d\n", n_tokens, kv_pos);
            break;
        }

        // ── DETOKENIZE ──
        case CMD_DETOKENIZE: {
            if (payload_len % 4 != 0) {
                send_error("DETOKENIZE payload not aligned to 4 bytes");
                break;
            }
            int n = (int)(payload_len / 4);
            std::vector<int32_t> tokens(n);
            memcpy(tokens.data(), payload.data(), payload_len);

            std::string text;
            for (int i = 0; i < n; i++) {
                text += common_token_to_piece(ctx, (llama_token)tokens[i]);
            }
            send_ok(text.data(), (uint32_t)text.size());
            fprintf(stderr, "[daemon] DETOKENIZE: %d tokens -> %zu chars\n", n, text.size());
            break;
        }

        // ── STATE_SAVE ── save full recurrent+KV state (single slot)
        case CMD_STATE_SAVE: {
            size_t state_size = llama_state_seq_get_size(ctx, 0);
            saved_state.resize(state_size);
            size_t written = llama_state_seq_get_data(ctx, saved_state.data(), saved_state.size(), 0);
            if (written == 0) {
                send_error("llama_state_seq_get_data failed (returned 0)");
                break;
            }
            saved_state.resize(written);  // actual size may differ
            saved_kv_pos = kv_pos;
            uint32_t sz = (uint32_t)written;
            send_ok(&sz, sizeof(sz));
            fprintf(stderr, "[daemon] STATE_SAVE: %zu bytes, kv_pos=%d\n", written, kv_pos);
            break;
        }

        // ── STATE_RESTORE ── restore full recurrent+KV state
        case CMD_STATE_RESTORE: {
            if (saved_state.empty()) {
                send_error("no saved state to restore");
                break;
            }
            size_t read_n = llama_state_seq_set_data(ctx, saved_state.data(), saved_state.size(), 0);
            if (read_n == 0) {
                send_error("llama_state_seq_set_data failed (returned 0)");
                break;
            }
            kv_pos = saved_kv_pos;
            send_ok();
            fprintf(stderr, "[daemon] STATE_RESTORE: %zu bytes, kv_pos=%d\n", read_n, kv_pos);
            break;
        }

        // ── GET_ROUTING ──
        // Returns MoE routing decisions from the most recent eval.
        // No payload for request.
        //
        // Response payload layout:
        //   [n_routing_layers : u16 LE]
        //   [n_tokens         : u16 LE]
        //   [data             : uint8[n_routing_layers * n_tokens * n_expert_used]]
        //
        // Layers are returned in ascending sorted order (matching routing_layers set iteration).
        // n_expert_used is implicit (8 for Qwen3.5-35B-A3B).  Python side uses it directly.
        //
        // Returns STATUS_ERROR if routing was not enabled or no eval has been run yet.
        case CMD_GET_ROUTING: {
            if (!cb_data.capture_routing) {
                send_error("routing capture not enabled (use --capture-routing)");
                break;
            }
            if (cb_data.routing_buf.empty() || cb_data.routing_n_tokens == 0) {
                send_error("no routing data captured yet (run eval first)");
                break;
            }

            // Determine ordered list of captured routing layers
            // routing_buf is keyed by layer_idx; iterate in ascending order.
            std::vector<int> captured_routing_layers;
            for (const auto & kv : cb_data.routing_buf) {
                captured_routing_layers.push_back(kv.first);
            }
            // std::map is ordered, so already ascending — but be explicit
            std::sort(captured_routing_layers.begin(), captured_routing_layers.end());

            int n_rl    = (int)captured_routing_layers.size();
            int n_tok   = (int)cb_data.routing_n_tokens;
            int n_exp   = (int)cb_data.routing_n_experts;

            // Build response: u16 n_layers + u16 n_tokens + uint8[n_layers * n_tokens * n_exp]
            size_t data_bytes  = (size_t)n_rl * n_tok * n_exp;
            size_t total_bytes = 2 + 2 + data_bytes;  // two u16 headers + data

            std::vector<uint8_t> resp(total_bytes);
            uint8_t * p = resp.data();

            uint16_t nl16 = (uint16_t)n_rl;
            uint16_t nt16 = (uint16_t)n_tok;
            memcpy(p, &nl16, 2); p += 2;
            memcpy(p, &nt16, 2); p += 2;

            for (int layer_idx : captured_routing_layers) {
                const auto & buf = cb_data.routing_buf.at(layer_idx);
                // buf is uint8[n_tokens * n_expert_used]
                size_t expected = (size_t)n_tok * n_exp;
                if (buf.size() < expected) {
                    // Partial capture — zero-pad (shouldn't normally happen)
                    memcpy(p, buf.data(), buf.size());
                    memset(p + buf.size(), 0, expected - buf.size());
                } else {
                    memcpy(p, buf.data(), expected);
                }
                p += expected;
            }

            send_ok(resp.data(), (uint32_t)total_bytes);
            fprintf(stderr, "[daemon] GET_ROUTING: %d layers, %d tokens, %d experts/token\n",
                    n_rl, n_tok, n_exp);
            break;
        }

        // ── STATE_SAVE_GDN ── save ONLY recurrent (GDN) states, not attention KV cache
        // Uses LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY to skip attention KV cache.
        // For Qwen3.5: saves ~2 MB (30 GDN layers) instead of ~63 MB (full state).
        // The attention KV cache is handled separately via TRIM_KV (O(1)).
        case CMD_STATE_SAVE_GDN: {
            size_t state_size = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            saved_gdn_state.resize(state_size);
            size_t written = llama_state_seq_get_data_ext(
                ctx, saved_gdn_state.data(), saved_gdn_state.size(), 0,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (written == 0) {
                send_error("llama_state_seq_get_data_ext (GDN) failed (returned 0)");
                break;
            }
            saved_gdn_state.resize(written);
            saved_gdn_kv_pos = kv_pos;
            uint32_t sz = (uint32_t)written;
            send_ok(&sz, sizeof(sz));
            fprintf(stderr, "[daemon] STATE_SAVE_GDN: %zu bytes (vs full ~%zu), kv_pos=%d\n",
                    written, llama_state_seq_get_size(ctx, 0), kv_pos);
            break;
        }

        // ── STATE_RESTORE_GDN ── restore GDN states + trim attention KV cache
        // Payload: [keep_n:i32] — number of KV positions to keep in attention cache.
        // Steps:
        //   1. Restore GDN recurrent states from saved buffer
        //   2. Trim attention KV cache to keep_n positions (llama_memory_seq_rm)
        //   3. Update kv_pos
        case CMD_STATE_RESTORE_GDN: {
            if (saved_gdn_state.empty()) {
                send_error("no saved GDN state to restore");
                break;
            }
            if (payload_len < 4) {
                send_error("STATE_RESTORE_GDN needs 4-byte payload (keep_n)");
                break;
            }

            int32_t keep_n;
            memcpy(&keep_n, payload.data(), 4);

            // Step 1: Restore GDN recurrent states
            size_t read_n = llama_state_seq_set_data_ext(
                ctx, saved_gdn_state.data(), saved_gdn_state.size(), 0,
                LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (read_n == 0) {
                send_error("llama_state_seq_set_data_ext (GDN) failed (returned 0)");
                break;
            }

            // Step 2: Trim attention KV cache (only affects attention layers, GDN is recurrent)
            if (keep_n >= 0 && keep_n < kv_pos) {
                llama_memory_seq_rm(llama_get_memory(ctx), 0, keep_n, -1);
            }

            kv_pos = (keep_n >= 0) ? keep_n : saved_gdn_kv_pos;
            send_ok();
            fprintf(stderr, "[daemon] STATE_RESTORE_GDN: %zu bytes, kv_pos=%d\n", read_n, kv_pos);
            break;
        }

        // ── QUIT ──
        case CMD_QUIT: {
            fprintf(stderr, "[daemon] QUIT received, shutting down.\n");
            goto done;
        }

        default: {
            send_error("unknown command 0x" + std::to_string(cmd));
            fprintf(stderr, "[daemon] unknown command 0x%02x\n", cmd);
            break;
        }
        }  // switch
    }  // for

done:
    llama_perf_context_print(ctx);
    llama_backend_free();
    fprintf(stderr, "[daemon] exit.\n");
    return 0;
}
