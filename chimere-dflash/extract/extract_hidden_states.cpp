/**
 * extract_hidden_states.cpp — Extract per-layer hidden states from Qwen3.5
 *
 * Uses llama.cpp eval callback to intercept "post_moe" tensors at specific
 * layer indices during forward pass. Saves as binary float32 files.
 *
 * Usage:
 *   ./extract_hidden_states \
 *     -m ~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/model.gguf \
 *     --layers 2,11,20,29,37 \
 *     --input data/prompts.jsonl \
 *     --output data/features/ \
 *     -ngl 99 -ot ".ffn_.*_exps.=CPU"
 *
 * Output format per sample:
 *   data/features/sample_000000/
 *     input_ids.bin     — int32[seq_len]
 *     layer_02.bin      — float32[seq_len * hidden_dim]
 *     layer_11.bin      — float32[seq_len * hidden_dim]
 *     ...
 *     metadata.json     — {seq_len, hidden_dim, layers, prompt}
 */

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ── Configuration ──

struct extract_config {
    std::set<int>   target_layers;     // layer indices to extract
    std::string     input_path;        // JSONL input file
    std::string     output_dir;        // output directory
    std::string     tensor_name = "post_moe";  // tensor name to capture
    int             max_samples = -1;  // -1 = all
    int             max_seq_len = 2048;
};

// ── Callback data ──

struct extract_callback_data {
    std::vector<uint8_t>              temp_buf;     // GPU→CPU transfer buffer
    std::set<int>                     target_layers;
    std::string                       tensor_name;

    // Captured data for current decode: layer_id → float32 vector
    std::map<int, std::vector<float>> captured;
    int64_t                           hidden_dim = 0;
    int64_t                           seq_len    = 0;
};

// ── Eval callback ──

static bool extract_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = static_cast<extract_callback_data *>(user_data);

    // Parse tensor name: "post_moe-N" where N is layer index
    // The callback system names tensors as "name-layer_idx"
    // In ggml, tensor names from cb() are like "post_moe-2", "post_moe-11" etc.
    std::string name(t->name);

    // Check if this tensor matches our target pattern
    size_t dash_pos = name.rfind('-');
    if (dash_pos == std::string::npos) {
        if (ask) return false;
        return true;
    }

    std::string base_name = name.substr(0, dash_pos);
    if (base_name != cb->tensor_name) {
        if (ask) return false;
        return true;
    }

    int layer_idx = -1;
    try {
        layer_idx = std::stoi(name.substr(dash_pos + 1));
    } catch (...) {
        if (ask) return false;
        return true;
    }

    // Check if this is a layer we want
    if (cb->target_layers.find(layer_idx) == cb->target_layers.end()) {
        if (ask) return false;
        return true;
    }

    if (ask) {
        return true;  // Yes, we want this tensor's data
    }

    // ── Data extraction (ask == false) ──

    // Get dimensions: post_moe tensors are [n_embd, n_tokens] (2D) or similar
    int64_t ne0 = t->ne[0];  // hidden_dim (n_embd)
    int64_t ne1 = t->ne[1];  // n_tokens

    cb->hidden_dim = ne0;
    cb->seq_len    = ne1;

    // Copy data from GPU if needed
    size_t n_bytes = ggml_nbytes(t);
    bool is_host = ggml_backend_buffer_is_host(t->buffer);

    uint8_t * data_ptr;
    if (is_host) {
        data_ptr = static_cast<uint8_t *>(t->data);
    } else {
        cb->temp_buf.resize(n_bytes);
        ggml_backend_tensor_get(t, cb->temp_buf.data(), 0, n_bytes);
        data_ptr = cb->temp_buf.data();
    }

    // Convert to float32 and store
    std::vector<float> float_data(ne0 * ne1);

    if (t->type == GGML_TYPE_F32) {
        memcpy(float_data.data(), data_ptr, ne0 * ne1 * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(data_ptr);
        for (int64_t i = 0; i < ne0 * ne1; i++) {
            float_data[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else if (t->type == GGML_TYPE_BF16) {
        const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(data_ptr);
        for (int64_t i = 0; i < ne0 * ne1; i++) {
            float_data[i] = ggml_bf16_to_fp32(src[i]);
        }
    } else {
        fprintf(stderr, "[extract] WARNING: unsupported tensor type %s for %s, skipping\n",
                ggml_type_name(t->type), t->name);
        return true;
    }

    cb->captured[layer_idx] = std::move(float_data);

    fprintf(stderr, "[extract] captured layer %d: [%lld x %lld] %s\n",
            layer_idx, (long long)ne0, (long long)ne1, ggml_type_name(t->type));

    return true;
}

// ── JSON helpers (minimal, no dependency) ──

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

// Parse a simple JSON string value: "key": "value"
static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++; // skip opening quote

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case '"': result += '"';  break;
                case '\\': result += '\\'; break;
                default: result += json[pos];
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

// ── Save functions ──

static void save_binary_f32(const std::string & path, const std::vector<float> & data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
}

static void save_binary_i32(const std::string & path, const std::vector<int32_t> & data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(int32_t));
}

static void save_metadata(const std::string & path, int64_t seq_len, int64_t hidden_dim,
                           const std::set<int> & layers, const std::string & prompt) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"seq_len\": " << seq_len << ",\n";
    f << "  \"hidden_dim\": " << hidden_dim << ",\n";
    f << "  \"layers\": [";
    bool first = true;
    for (int l : layers) {
        if (!first) f << ", ";
        f << l;
        first = false;
    }
    f << "],\n";
    f << "  \"prompt\": \"" << json_escape(prompt) << "\"\n";
    f << "}\n";
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
            fprintf(stderr, "[extract] WARNING: invalid layer '%s'\n", token.c_str());
        }
    }
    return result;
}

// ── Load JSONL samples ──

struct sample {
    std::string prompt;
    std::string response;
};

static std::vector<sample> load_jsonl(const std::string & path, int max_samples) {
    std::vector<sample> samples;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[extract] ERROR: cannot open %s\n", path.c_str());
        return samples;
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        sample s;
        s.prompt   = json_get_string(line, "prompt");
        s.response = json_get_string(line, "response");

        // Fallback keys
        if (s.prompt.empty()) s.prompt = json_get_string(line, "text");
        if (s.response.empty()) s.response = json_get_string(line, "completion");

        if (!s.prompt.empty()) {
            samples.push_back(std::move(s));
        }

        if (max_samples > 0 && (int)samples.size() >= max_samples) break;
    }

    return samples;
}

// ── Main ──

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [llama-args] --layers 2,11,20,29,37 --input prompts.jsonl --output features/\n", prog);
    fprintf(stderr, "\nExtra arguments:\n");
    fprintf(stderr, "  --layers L1,L2,...   Layer indices to extract (required)\n");
    fprintf(stderr, "  --input FILE         JSONL input file with prompt/response pairs (required)\n");
    fprintf(stderr, "  --output DIR         Output directory for features (required)\n");
    fprintf(stderr, "  --tensor NAME        Tensor name to capture (default: post_moe)\n");
    fprintf(stderr, "  --max-samples N      Max samples to process (-1 = all)\n");
    fprintf(stderr, "  --max-seq-len N      Max sequence length (default: 2048)\n");
    fprintf(stderr, "\nAll standard llama.cpp arguments (-m, -ngl, -ot, etc.) are also supported.\n");
}

int main(int argc, char ** argv) {
    extract_config ecfg;

    // Pre-parse our custom args before passing to common_params_parse
    // We need to extract them and remove from argv so llama doesn't choke
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--layers" && i + 1 < argc) {
            ecfg.target_layers = parse_layers(argv[++i]);
        } else if (arg == "--input" && i + 1 < argc) {
            ecfg.input_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            ecfg.output_dir = argv[++i];
        } else if (arg == "--tensor" && i + 1 < argc) {
            ecfg.tensor_name = argv[++i];
        } else if (arg == "--max-samples" && i + 1 < argc) {
            ecfg.max_samples = std::stoi(argv[++i]);
        } else if (arg == "--max-seq-len" && i + 1 < argc) {
            ecfg.max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--help-extract") {
            print_usage(argv[0]);
            return 0;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    // Validate
    if (ecfg.target_layers.empty() || ecfg.input_path.empty() || ecfg.output_dir.empty()) {
        print_usage(argv[0]);
        fprintf(stderr, "\nERROR: --layers, --input, and --output are required.\n");
        return 1;
    }

    // We need a dummy prompt for common_params_parse (it will be overridden per sample)
    // Add -p "dummy" if no prompt was provided
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

    // Parse llama params
    common_params params;
    int fargc = static_cast<int>(filtered_argv.size());

    if (!common_params_parse(fargc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Setup callback
    extract_callback_data cb_data;
    cb_data.target_layers = ecfg.target_layers;
    cb_data.tensor_name   = ecfg.tensor_name;

    params.cb_eval           = extract_cb_eval;
    params.cb_eval_user_data = &cb_data;
    params.warmup            = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Load model
    fprintf(stderr, "[extract] Loading model...\n");
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model || !ctx) {
        fprintf(stderr, "[extract] ERROR: failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Load samples
    auto samples = load_jsonl(ecfg.input_path, ecfg.max_samples);
    fprintf(stderr, "[extract] Loaded %zu samples from %s\n", samples.size(), ecfg.input_path.c_str());
    fprintf(stderr, "[extract] Target layers: ");
    for (int l : ecfg.target_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "\n");
    fprintf(stderr, "[extract] Tensor name: %s\n", ecfg.tensor_name.c_str());

    fs::create_directories(ecfg.output_dir);

    int processed = 0;
    int skipped   = 0;

    for (size_t si = 0; si < samples.size(); si++) {
        const auto & sample = samples[si];

        // Tokenize full text (prompt + response)
        std::string full_text = sample.prompt + sample.response;
        std::vector<llama_token> tokens = common_tokenize(ctx, full_text, add_bos);

        if (tokens.empty()) {
            skipped++;
            continue;
        }

        // Truncate to max_seq_len
        if ((int)tokens.size() > ecfg.max_seq_len) {
            tokens.resize(ecfg.max_seq_len);
        }

        // Clear previous captures
        cb_data.captured.clear();
        cb_data.hidden_dim = 0;
        cb_data.seq_len    = 0;

        // Reset KV cache for clean extraction
        llama_memory_clear(llama_get_memory(ctx), true);

        // Decode
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "[extract] WARNING: decode failed for sample %zu (ret=%d), skipping\n", si, ret);
            skipped++;
            continue;
        }

        // Verify we captured all target layers
        if (cb_data.captured.size() != ecfg.target_layers.size()) {
            fprintf(stderr, "[extract] WARNING: sample %zu captured %zu/%zu layers, skipping\n",
                    si, cb_data.captured.size(), ecfg.target_layers.size());
            skipped++;
            continue;
        }

        // Save to disk
        char sample_dir_name[64];
        snprintf(sample_dir_name, sizeof(sample_dir_name), "sample_%06zu", si);
        std::string sample_dir = ecfg.output_dir + "/" + sample_dir_name;
        fs::create_directories(sample_dir);

        // Save input_ids
        std::vector<int32_t> token_ids(tokens.begin(), tokens.end());
        save_binary_i32(sample_dir + "/input_ids.bin", token_ids);

        // Save hidden states per layer
        for (const auto & [layer_id, data] : cb_data.captured) {
            char fname[64];
            snprintf(fname, sizeof(fname), "layer_%02d.bin", layer_id);
            save_binary_f32(sample_dir + "/" + fname, data);
        }

        // Save metadata
        save_metadata(
            sample_dir + "/metadata.json",
            cb_data.seq_len,
            cb_data.hidden_dim,
            ecfg.target_layers,
            sample.prompt
        );

        processed++;
        if (processed % 10 == 0) {
            fprintf(stderr, "[extract] %d/%zu samples processed (hidden_dim=%lld, seq_len=%lld)\n",
                    processed, samples.size(),
                    (long long)cb_data.hidden_dim, (long long)cb_data.seq_len);
        }
    }

    fprintf(stderr, "[extract] Done: %d processed, %d skipped, output in %s\n",
            processed, skipped, ecfg.output_dir.c_str());

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
