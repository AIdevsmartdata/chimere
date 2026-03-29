/**
 * extract_single_position.cpp — Native C++ hidden state extractor (single or multi-position)
 *
 * Replaces the Python extract_single_position.py + target_daemon IPC pipeline.
 * Directly loads the model, tokenizes, runs forward pass, and saves features.
 *
 * For each JSONL prompt+response:
 *   1. Tokenize full text
 *   2. Compute anchor_pos = min(seq_len, max_seq_len) - BLOCK_SIZE - 1
 *   3. Forward pass on tokens[0..anchor_pos] (inclusive)
 *   4. Extract hidden states at last ctx_len positions (default 1 for backward compat)
 *   5. Save context_hidden.bin (float16[n_layers, ctx_len, hidden_dim]) if ctx_len>1
 *      OR anchor_hidden.bin (float16[n_layers, hidden_dim]) if ctx_len=1
 *   6. Save block_tokens.bin (int32[BLOCK_SIZE]) = tokens after anchor
 *   7. Save metadata.json
 *
 * Output format:
 *   data/features_q5/sample_000000/
 *     context_hidden.bin  — float16[5 * ctx_len * 2048] (ctx_len>1)
 *     OR anchor_hidden.bin — float16[5 * 2048] (ctx_len=1, legacy)
 *     block_tokens.bin    — int32[16] = 64 bytes
 *     metadata.json       — {anchor_pos, seq_len, block_size, ctx_len, layers, hidden_dim, dtype, source_id}
 *
 * Usage:
 *   ./extract_single_position \
 *     -m ~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/model.gguf \
 *     --layers 1,10,19,28,37 \
 *     --input data/prompts_v6/combined_150k.jsonl \
 *     --output data/features_q5 \
 *     --max-seq-len 512 \
 *     --block-size 16 \
 *     --resume-from 0 \
 *     -ngl 99 -ot "blk.[2-3][0-9].ffn_.*_exps.weight=CPU" \
 *     --flash-attn on -b 4096 -ub 4096 \
 *     --cache-type-k q8_0 --cache-type-v q4_0
 */

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// NOTE: async_io.h is included AFTER sample_record definition (see below)

// ── Constants ──

static constexpr int    DEFAULT_BLOCK_SIZE   = 16;
static constexpr int    DEFAULT_MAX_SEQ_LEN  = 512;
static constexpr int    MASK_TOKEN_ID        = 248077;  // <|MASK|> z-lab convention
static constexpr size_t MAX_LINE_BYTES       = 16 * 1024 * 1024;  // 16 MB per JSONL line

// ── Signal handling ──

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int /*signum*/) {
    if (g_shutdown.load(std::memory_order_relaxed)) {
        // Second signal: force exit
        const char msg[] = "\nForce kill.\n";
        (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
        _exit(1);
    }
    g_shutdown.store(true, std::memory_order_relaxed);
    const char msg[] = "\nShutdown requested, finishing current sample...\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
}

static void install_signal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT,  &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);
}

// ── Configuration ──

struct extract_config {
    std::set<int>   target_layers;
    std::string     input_path;
    std::string     output_dir;
    std::string     tensor_name  = "post_moe";
    int             max_samples  = -1;
    int             max_seq_len  = DEFAULT_MAX_SEQ_LEN;
    int             block_size   = DEFAULT_BLOCK_SIZE;
    int             ctx_len      = 1;       // number of context positions to extract
    int             resume_from  = 0;       // skip first N prompts
    bool            sort_by_len  = true;
    bool            extract_all  = false;   // extract ALL positions (full-sequence mode)
};

// ── Callback data ──

struct extract_callback_data {
    std::vector<uint8_t>              temp_buf;
    std::set<int>                     target_layers;
    std::string                       tensor_name;
    std::map<int, std::vector<float>> captured;  // layer_id -> float32[n_positions * hidden_dim]
    int64_t                           hidden_dim    = 0;
    int64_t                           seq_len       = 0;
    int                               ctx_len       = 1;   // how many positions to extract
    int64_t                           n_positions   = 0;   // actual positions extracted (may be < ctx_len)
    bool                              extract_all   = false; // extract ALL positions
};

// ── Eval callback (last position only) ──

static bool extract_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = static_cast<extract_callback_data *>(user_data);

    std::string name(t->name);
    size_t dash_pos = name.rfind('-');
    if (dash_pos == std::string::npos) return ask ? false : true;

    std::string base_name = name.substr(0, dash_pos);
    if (base_name != cb->tensor_name) return ask ? false : true;

    int layer_idx = -1;
    try {
        layer_idx = std::stoi(name.substr(dash_pos + 1));
    } catch (...) {
        return ask ? false : true;
    }

    if (cb->target_layers.find(layer_idx) == cb->target_layers.end())
        return ask ? false : true;

    if (ask) return true;

    // ── Extract last ctx_len positions ──
    int64_t ne0 = t->ne[0];  // hidden_dim
    int64_t ne1 = t->ne[1];  // seq_len
    cb->hidden_dim = ne0;
    cb->seq_len    = ne1;

    int64_t start_pos = cb->extract_all ? 0 : std::max((int64_t)0, ne1 - (int64_t)cb->ctx_len);
    int64_t n_pos = ne1 - start_pos;
    cb->n_positions = n_pos;

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

    // Extract n_pos positions: flat array [n_pos * hidden_dim]
    std::vector<float> float_data(ne0 * n_pos);

    for (int64_t p = 0; p < n_pos; p++) {
        int64_t src_pos = start_pos + p;
        if (t->type == GGML_TYPE_F32) {
            const float * src = reinterpret_cast<const float *>(data_ptr) + src_pos * ne0;
            memcpy(float_data.data() + p * ne0, src, ne0 * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(data_ptr) + src_pos * ne0;
            for (int64_t i = 0; i < ne0; i++) {
                float_data[p * ne0 + i] = ggml_fp16_to_fp32(src[i]);
            }
        } else if (t->type == GGML_TYPE_BF16) {
            const ggml_bf16_t * src = reinterpret_cast<const ggml_bf16_t *>(data_ptr) + src_pos * ne0;
            for (int64_t i = 0; i < ne0; i++) {
                float_data[p * ne0 + i] = ggml_bf16_to_fp32(src[i]);
            }
        } else {
            fprintf(stderr, "[extract] WARNING: unsupported tensor type %s for %s\n",
                    ggml_type_name(t->type), t->name);
            return true;
        }
    }

    cb->captured[layer_idx] = std::move(float_data);
    return true;
}

// ── JSON helpers (zero-dependency) ──

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string json_get_string(const std::string & json, const std::string & key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    // Find colon after key
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    // Find opening quote
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++; // skip opening quote

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case 'r':  result += '\r'; break;
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case '/':  result += '/';  break;
                default:   result += json[pos];
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

// ── Parse layers "1,10,19,28,37" ──

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

// ── Sample record ──

struct sample_record {
    std::string id;
    std::string prompt;
    std::string response;
    size_t      text_len;  // for sorting
};

// ── Async I/O pipeline (must come after sample_record definition) ──
#include "async_io.h"

// ── Load and optionally sort JSONL ──

static std::vector<sample_record> load_jsonl(const std::string & path, int max_samples, bool sort_by_len) {
    std::vector<sample_record> samples;
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[extract] ERROR: cannot open %s\n", path.c_str());
        return samples;
    }

    std::string line;
    size_t line_no = 0;
    while (std::getline(f, line)) {
        line_no++;
        if (line.empty()) continue;
        if (line.size() > MAX_LINE_BYTES) {
            fprintf(stderr, "[extract] WARNING: line %zu exceeds max size (%zu bytes), skipping\n",
                    line_no, line.size());
            continue;
        }

        sample_record rec;
        rec.prompt   = json_get_string(line, "prompt");
        rec.response = json_get_string(line, "response");
        rec.id       = json_get_string(line, "id");

        // Fallback keys
        if (rec.prompt.empty()) rec.prompt = json_get_string(line, "text");
        if (rec.response.empty()) rec.response = json_get_string(line, "completion");
        if (rec.id.empty()) rec.id = "prompt_" + std::to_string(line_no);

        if (rec.prompt.empty()) continue;

        rec.text_len = rec.prompt.size() + rec.response.size();
        samples.push_back(std::move(rec));

        if (max_samples > 0 && (int)samples.size() >= max_samples) break;
    }

    if (sort_by_len) {
        std::sort(samples.begin(), samples.end(),
                  [](const sample_record & a, const sample_record & b) {
                      return a.text_len < b.text_len;
                  });
        fprintf(stderr, "[extract] Sorted %zu samples by text length (ascending)\n", samples.size());
    }

    return samples;
}

// ── Save helpers ──

static bool save_anchor_hidden_f16(const std::string & path,
                                    const std::map<int, std::vector<float>> & captured,
                                    const std::set<int> & target_layers,
                                    int64_t hidden_dim) {
    // Save as float16[n_layers, hidden_dim] in layer order (ctx_len=1 legacy)
    size_t n_layers = target_layers.size();
    size_t total_elements = n_layers * hidden_dim;

    std::vector<uint16_t> fp16_data(total_elements);
    size_t offset = 0;

    for (int layer_id : target_layers) {
        auto it = captured.find(layer_id);
        if (it == captured.end()) return false;

        const auto & fdata = it->second;
        if ((int64_t)fdata.size() < hidden_dim) return false;

        for (int64_t i = 0; i < hidden_dim; i++) {
            fp16_data[offset + i] = ggml_fp32_to_fp16(fdata[i]);
        }
        offset += hidden_dim;
    }

    // Write to .tmp then rename for atomicity
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(fp16_data.data()),
                total_elements * sizeof(uint16_t));
        f.flush();
        if (!f) {
            fs::remove(tmp_path);
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
        fs::remove(tmp_path);
        return false;
    }
    return true;
}

static bool save_context_hidden_f16(const std::string & path,
                                     const std::map<int, std::vector<float>> & captured,
                                     const std::set<int> & target_layers,
                                     int64_t hidden_dim, int64_t n_positions) {
    // Save as float16[n_layers, n_positions, hidden_dim] in layer order
    size_t n_layers = target_layers.size();
    size_t total_elements = n_layers * n_positions * hidden_dim;

    std::vector<uint16_t> fp16_data(total_elements);
    size_t offset = 0;

    for (int layer_id : target_layers) {
        auto it = captured.find(layer_id);
        if (it == captured.end()) return false;

        const auto & fdata = it->second;
        int64_t expected = n_positions * hidden_dim;
        if ((int64_t)fdata.size() < expected) return false;

        for (int64_t i = 0; i < expected; i++) {
            fp16_data[offset + i] = ggml_fp32_to_fp16(fdata[i]);
        }
        offset += expected;
    }

    // Write to .tmp then rename for atomicity
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(fp16_data.data()),
                total_elements * sizeof(uint16_t));
        f.flush();
        if (!f) {
            fs::remove(tmp_path);
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
        fs::remove(tmp_path);
        return false;
    }
    return true;
}

static bool save_block_tokens(const std::string & path,
                               const std::vector<int32_t> & block_tokens) {
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(block_tokens.data()),
                block_tokens.size() * sizeof(int32_t));
        f.flush();
        if (!f) {
            fs::remove(tmp_path);
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
        fs::remove(tmp_path);
        return false;
    }
    return true;
}

static bool save_tokens(const std::string & path,
                         const std::vector<llama_token> & tokens, int seq_len) {
    std::vector<int32_t> tok_i32(seq_len);
    for (int i = 0; i < seq_len; i++) {
        tok_i32[i] = (int32_t)tokens[i];
    }
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(tok_i32.data()),
                seq_len * sizeof(int32_t));
        f.flush();
        if (!f) {
            fs::remove(tmp_path);
            return false;
        }
    }
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
        fs::remove(tmp_path);
        return false;
    }
    return true;
}

static bool save_metadata(const std::string & path,
                           int anchor_pos, int seq_len, int block_size,
                           int ctx_len, int n_positions,
                           const std::set<int> & layers, int hidden_dim,
                           const std::string & source_id,
                           const std::string & mode = "anchor") {
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path);
        if (!f) return false;
        f << "{\n";
        f << "  \"anchor_pos\": " << anchor_pos << ",\n";
        f << "  \"seq_len\": " << seq_len << ",\n";
        f << "  \"block_size\": " << block_size << ",\n";
        f << "  \"ctx_len\": " << ctx_len << ",\n";
        f << "  \"n_positions\": " << n_positions << ",\n";
        f << "  \"layers\": [";
        bool first = true;
        for (int l : layers) {
            if (!first) f << ", ";
            f << l;
            first = false;
        }
        f << "],\n";
        f << "  \"hidden_dim\": " << hidden_dim << ",\n";
        f << "  \"dtype\": \"float16\",\n";
        f << "  \"mode\": \"" << json_escape(mode) << "\",\n";
        f << "  \"source_id\": \"" << json_escape(source_id) << "\"\n";
        f << "}\n";
        f.flush();
        if (!f) {
            fs::remove(tmp_path);
            return false;
        }
    }

    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) {
        fs::remove(tmp_path);
        return false;
    }
    return true;
}

// ── NaN/Inf check ──

static bool has_nan_or_inf(const std::vector<float> & data) {
    for (float v : data) {
        if (!std::isfinite(v)) return true;
    }
    return false;
}

// ── Progress checkpoint for resume ──

static void save_checkpoint(const std::string & output_dir, size_t prompt_idx, size_t n_written) {
    std::string path = output_dir + "/.extraction_checkpoint.json";
    std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp);
        f << "{\"prompt_idx\": " << prompt_idx << ", \"n_written\": " << n_written << "}\n";
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
}

// ── Main ──

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [llama-args] --layers 1,10,19,28,37 --input prompts.jsonl --output features/\n\n", prog);
    fprintf(stderr, "DFlash v6 single-position hidden state extractor.\n");
    fprintf(stderr, "Extracts anchor-point hidden states for block diffusion drafter training.\n\n");
    fprintf(stderr, "Extra arguments:\n");
    fprintf(stderr, "  --layers L1,L2,...   Layer indices to extract (required)\n");
    fprintf(stderr, "  --input FILE         JSONL input file with prompt/response pairs (required)\n");
    fprintf(stderr, "  --output DIR         Output directory for features (required)\n");
    fprintf(stderr, "  --tensor NAME        Tensor name to capture (default: post_moe)\n");
    fprintf(stderr, "  --max-samples N      Max samples to process (-1 = all)\n");
    fprintf(stderr, "  --max-seq-len N      Max sequence length (default: %d)\n", DEFAULT_MAX_SEQ_LEN);
    fprintf(stderr, "  --block-size N       Block size for DFlash (default: %d)\n", DEFAULT_BLOCK_SIZE);
    fprintf(stderr, "  --ctx-len N          Context positions to extract (default: 1)\n");
    fprintf(stderr, "  --resume-from N      Skip first N prompts (default: 0)\n");
    fprintf(stderr, "  --no-sort            Do not sort prompts by length\n");
    fprintf(stderr, "  --extract-all        Extract ALL positions (full-sequence mode for v8)\n");
    fprintf(stderr, "\nAll standard llama.cpp arguments (-m, -ngl, -ot, etc.) are also supported.\n");
}

int main(int argc, char ** argv) {
    install_signal_handlers();

    extract_config ecfg;

    // Pre-parse custom args
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
        } else if (arg == "--block-size" && i + 1 < argc) {
            ecfg.block_size = std::stoi(argv[++i]);
        } else if (arg == "--ctx-len" && i + 1 < argc) {
            ecfg.ctx_len = std::stoi(argv[++i]);
        } else if (arg == "--resume-from" && i + 1 < argc) {
            ecfg.resume_from = std::stoi(argv[++i]);
        } else if (arg == "--no-sort") {
            ecfg.sort_by_len = false;
        } else if (arg == "--extract-all") {
            ecfg.extract_all = true;
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
    cb_data.ctx_len       = ecfg.ctx_len;
    cb_data.extract_all   = ecfg.extract_all;

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
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Load samples
    fprintf(stderr, "[extract] Loading prompts from %s...\n", ecfg.input_path.c_str());
    auto samples = load_jsonl(ecfg.input_path, ecfg.max_samples, ecfg.sort_by_len);
    if (samples.empty()) {
        fprintf(stderr, "[extract] ERROR: no samples loaded\n");
        llama_backend_free();
        return 1;
    }

    fs::create_directories(ecfg.output_dir);

    // Count existing samples for resume
    size_t existing = 0;
    for (auto & entry : fs::directory_iterator(ecfg.output_dir)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            if (name.substr(0, 7) == "sample_") existing++;
        }
    }

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, " DFlash C++ Hidden State Extraction (ctx_len=%d)\n", ecfg.ctx_len);
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Model:            %s\n", params.model.path.c_str());
    fprintf(stderr, "  Layers:           ");
    for (int l : ecfg.target_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Block size:       %d\n", ecfg.block_size);
    fprintf(stderr, "  Context len:      %d\n", ecfg.ctx_len);
    fprintf(stderr, "  Max seq len:      %d\n", ecfg.max_seq_len);
    fprintf(stderr, "  Tensor:           %s\n", ecfg.tensor_name.c_str());
    fprintf(stderr, "  Prompts:          %zu\n", samples.size());
    fprintf(stderr, "  Resume from:      %d\n", ecfg.resume_from);
    fprintf(stderr, "  Existing samples: %zu\n", existing);
    fprintf(stderr, "  Extract all:      %s\n", ecfg.extract_all ? "yes (full-seq)" : "no");
    fprintf(stderr, "  Sorted by length: %s\n", ecfg.sort_by_len ? "yes" : "no");
    fprintf(stderr, "\n");

    // ── Async I/O pipeline ──

    AsyncWriter    writer(8);
    AsyncTokenizer tokenizer(vocab, add_bos, ecfg.max_seq_len);

    fprintf(stderr, "[extract] Async I/O pipeline enabled (writer queue=8, pre-tokenizer)\n");

    // ── Extraction loop ──

    size_t n_written = existing;
    size_t n_failed  = 0;
    size_t n_skipped = 0;
    uint64_t total_tokens = 0;

    auto t_start = std::chrono::steady_clock::now();

    // Pre-tokenize first prompt
    size_t start_si = (size_t)ecfg.resume_from;
    if (start_si < samples.size()) {
        tokenizer.submit_batch(samples, start_si, 1);
    }

    for (size_t si = start_si; si < samples.size(); si++) {
        if (g_shutdown.load(std::memory_order_relaxed)) {
            fprintf(stderr, "\n[extract] Graceful shutdown at prompt %zu\n", si);
            save_checkpoint(ecfg.output_dir, si, n_written);
            break;
        }

        // Get pre-tokenized result (should be ready — tokenized during previous GPU forward)
        auto tok_results = tokenizer.get_results();

        // Immediately submit next prompt for pre-tokenization (overlaps with GPU forward below)
        if (si + 1 < samples.size()) {
            tokenizer.submit_batch(samples, si + 1, 1);
        }

        // Progress reporting every 100 samples
        if ((si + 1) % 100 == 0 || si == start_si) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            size_t done = si + 1 - start_si;
            double rate = done / (elapsed + 1e-9);
            double tok_rate = total_tokens / (elapsed + 1e-9);
            double eta_h = (samples.size() - si - 1) / (rate + 1e-9) / 3600.0;
            fprintf(stderr, "  [%zu/%zu] %.1f prompts/s | %.0f tok/s | written=%zu | "
                            "failed=%zu | pending_io=%zu | ETA=%.1fh\n",
                    si + 1, samples.size(), rate, tok_rate, n_written,
                    n_failed + writer.failed(), writer.pending(), eta_h);
        }

        // Use pre-tokenized result
        if (tok_results.empty()) {
            n_skipped++;
            continue;
        }

        auto & tp = tok_results[0];
        std::vector<llama_token> tokens = std::move(tp.tokens);
        int seq_len = tp.seq_len;
        const std::string & source_id = tp.source_id;
        total_tokens += seq_len;

        // Need at least block_size + 2 tokens
        if (seq_len < ecfg.block_size + 2) {
            n_skipped++;
            continue;
        }

        // Compute anchor position
        int anchor_pos = seq_len - ecfg.block_size - 1;
        if (anchor_pos < 0) {
            n_skipped++;
            continue;
        }

        // Forward pass: full sequence if extract_all, else up to anchor_pos
        int eval_len = ecfg.extract_all ? seq_len : anchor_pos + 1;

        // Clear callback captures
        cb_data.captured.clear();
        cb_data.hidden_dim = 0;
        cb_data.seq_len    = 0;

        // Clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // Decode — request logits only for last position
        // (next prompt is being tokenized in background during this GPU forward)
        llama_batch batch = llama_batch_init(eval_len, 0, 1);
        for (int i = 0; i < eval_len; i++) {
            common_batch_add(batch, tokens[i], i, {0}, i == eval_len - 1);
        }

        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            fprintf(stderr, "[extract] WARNING: decode failed for prompt %zu (ret=%d)\n", si, ret);
            n_failed++;
            continue;
        }

        // Verify all layers captured
        if (cb_data.captured.size() != ecfg.target_layers.size()) {
            fprintf(stderr, "[extract] WARNING: prompt %zu captured %zu/%zu layers\n",
                    si, cb_data.captured.size(), ecfg.target_layers.size());
            n_failed++;
            continue;
        }

        // NaN check on captured hidden states
        bool has_bad = false;
        for (const auto & [lid, data] : cb_data.captured) {
            if (has_nan_or_inf(data)) {
                has_bad = true;
                break;
            }
        }
        if (has_bad) {
            fprintf(stderr, "[extract] WARNING: NaN/Inf in hidden states for prompt %zu\n", si);
            n_failed++;
            continue;
        }

        // Build async write job — moves captured data to writer thread (zero-copy)
        AsyncWriter::WriteJob job;
        char dir_name[64];
        snprintf(dir_name, sizeof(dir_name), "sample_%06zu", n_written);
        job.sample_dir    = ecfg.output_dir + "/" + dir_name;
        job.captured      = std::move(cb_data.captured);
        job.target_layers = ecfg.target_layers;
        job.hidden_dim    = cb_data.hidden_dim;
        job.n_positions   = cb_data.n_positions;
        job.anchor_pos    = anchor_pos;
        job.block_size    = ecfg.block_size;
        job.source_id     = source_id;

        int64_t actual_n_pos = cb_data.n_positions;

        if (ecfg.extract_all) {
            job.mode        = "full_seq";
            job.ctx_len     = 0;
            job.seq_len     = seq_len;
            job.full_tokens = std::move(tokens);
        } else {
            job.mode    = (ecfg.ctx_len == 1) ? "anchor" : "context";
            job.ctx_len = ecfg.ctx_len;
            job.seq_len = seq_len;
            // Block tokens (next block_size tokens after anchor)
            job.block_tokens.resize(ecfg.block_size, MASK_TOKEN_ID);
            for (int j = 0; j < ecfg.block_size; j++) {
                int tok_idx = anchor_pos + 1 + j;
                if (tok_idx < seq_len) {
                    job.block_tokens[j] = (int32_t)tokens[tok_idx];
                }
            }
        }

        writer.submit(std::move(job));
        n_written++;
    }

    // Flush async writer — wait for all pending disk writes
    fprintf(stderr, "[extract] Flushing async writer (%zu pending)...\n", writer.pending());
    writer.flush();
    n_failed += writer.failed();

    // ── Summary ──

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // Count total disk usage
    uint64_t total_bytes = 0;
    for (auto & entry : fs::recursive_directory_iterator(ecfg.output_dir)) {
        if (entry.is_regular_file()) {
            total_bytes += entry.file_size();
        }
    }

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, " Extraction Summary\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Samples written:  %zu\n", n_written);
    fprintf(stderr, "  Samples failed:   %zu\n", n_failed);
    fprintf(stderr, "  Samples skipped:  %zu (too short)\n", n_skipped);
    fprintf(stderr, "  Total tokens:     %llu\n", (unsigned long long)total_tokens);
    fprintf(stderr, "  Total size:       %.1f MB\n", total_bytes / 1e6);
    fprintf(stderr, "  Per sample:       %.1f KB\n", total_bytes / (double)std::max(n_written, (size_t)1) / 1024.0);
    fprintf(stderr, "  Time:             %.1fh\n", elapsed / 3600.0);
    fprintf(stderr, "  Rate:             %.1f prompts/s\n",
            (samples.size() - ecfg.resume_from) / (elapsed + 1e-9));
    fprintf(stderr, "  Token rate:       %.0f tok/s\n", total_tokens / (elapsed + 1e-9));
    fprintf(stderr, "============================================================\n\n");

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
