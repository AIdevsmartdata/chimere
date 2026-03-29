/**
 * extract_batched.cpp — Batched hidden state extractor (N sequences per llama_decode call)
 *
 * Extension of extract_single_position.cpp that packs N prompts into a single
 * llama_batch using distinct seq_id values (0..N-1), cutting per-prompt overhead
 * and improving GPU utilization for the selective-offload Qwen3.5-35B-A3B setup.
 *
 * Key differences from extract_single_position.cpp:
 *   - Groups prompts into batches of --batch-seqs N (default 4)
 *   - Sorts within each batch by token count to minimise wasted positions
 *   - Single llama_decode per batch; fallback to one-by-one on failure
 *   - Callback extracts per-seq ranges from the concatenated tensor
 *   - KV cache sized for N × max_seq_len tokens (-c must be >= that)
 *
 * Output format (identical to extract_single_position.cpp):
 *   data/features_q5/sample_000000/
 *     context_hidden.bin  — float16[n_layers, n_positions, hidden_dim]  (extract_all or ctx_len>1)
 *     OR anchor_hidden.bin — float16[n_layers, hidden_dim]              (ctx_len=1 legacy)
 *     block_tokens.bin    — int32[block_size]                           (non-extract_all)
 *     tokens.bin          — int32[seq_len]                              (extract_all)
 *     metadata.json
 *
 * Usage:
 *   ./extract_batched \
 *     -m ~/.openclaw/models/Qwen3.5-35B-A3B-GGUF/model.gguf \
 *     --layers 1,10,19,28,37 \
 *     --input data/prompts_v6/combined_150k.jsonl \
 *     --output data/features_q5 \
 *     --max-seq-len 512 \
 *     --block-size 16 \
 *     --batch-seqs 4 \
 *     --extract-all \
 *     -ngl 99 -ot "blk.[2-3][0-9].ffn_.*_exps.weight=CPU" \
 *     --flash-attn on -b 4096 -ub 4096 \
 *     -c 2048 \
 *     --cache-type-k q8_0 --cache-type-v q4_0
 *
 * VRAM note:
 *   With N=4, max_seq_len=512: context = 2048 tokens.
 *   At ~50 KB/token KV (q8_0 keys + q4_0 values, 40 layers): ~100 MB — well within 2 GB margin.
 *   Set -c >= batch_seqs * max_seq_len.
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

namespace fs = std::filesystem;

// ── Constants ──

static constexpr int    DEFAULT_BLOCK_SIZE   = 16;
static constexpr int    DEFAULT_MAX_SEQ_LEN  = 512;
static constexpr int    DEFAULT_BATCH_SEQS   = 4;
static constexpr int    MASK_TOKEN_ID        = 248077;   // <|MASK|> z-lab convention
static constexpr size_t MAX_LINE_BYTES       = 16 * 1024 * 1024;  // 16 MB per JSONL line

// ── Signal handling ──

static std::atomic<bool> g_shutdown{false};

static void signal_handler(int /*signum*/) {
    if (g_shutdown.load(std::memory_order_relaxed)) {
        const char msg[] = "\nForce kill.\n";
        (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
        _exit(1);
    }
    g_shutdown.store(true, std::memory_order_relaxed);
    const char msg[] = "\nShutdown requested, finishing current batch...\n";
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
    int             ctx_len      = 1;       // context positions to extract (non-extract_all)
    int             resume_from  = 0;       // skip first N prompts
    int             batch_seqs   = DEFAULT_BATCH_SEQS;  // sequences per llama_decode call
    bool            sort_by_len  = true;
    bool            extract_all  = false;   // extract ALL positions (full-sequence mode)
};

// ── Per-sequence info used during batched decode ──

// A single sequence within a batch
struct batch_seq_entry {
    size_t                  sample_idx;   // index into samples[]
    std::vector<llama_token> tokens;       // tokenized, truncated to eval_len
    int                     eval_len;     // tokens to feed (anchor+1 or full seq_len)
    int                     seq_len;      // full truncated sequence length (for metadata)
    int                     anchor_pos;   // last context position
    llama_seq_id            seq_id;       // 0..batch_seqs-1
    int64_t                 tensor_start; // first row in the callback tensor for this seq
};

// ── Callback data ──

struct extract_callback_data {
    std::vector<uint8_t>   temp_buf;
    std::set<int>          target_layers;
    std::string            tensor_name;

    // Per-sequence capture: seq_id -> (layer_id -> float32[n_positions * hidden_dim])
    // Populated during callback; keyed by seq_id so results survive multi-seq decode.
    std::vector<std::map<int, std::vector<float>>> captured;  // [seq_id][layer_id]

    // Per-sequence position info set BEFORE each decode
    // seq_ranges[i] = {tensor_start_row, eval_len} for seq_id i
    // In extract_all=true mode, start_pos_to_extract=0 and n_extract=eval_len.
    // In anchor mode: start_pos_to_extract = max(0, eval_len - ctx_len), n_extract = ctx_len.
    struct seq_range {
        int64_t tensor_start;       // first row in the flattened tensor belonging to this seq
        int64_t eval_len;           // number of rows for this seq
        int64_t extract_start;      // first row to extract (relative to tensor_start)
        int64_t n_extract;          // number of rows to extract
    };
    std::vector<seq_range>  seq_ranges;  // indexed by seq_id

    int64_t hidden_dim  = 0;
    int     ctx_len     = 1;
    bool    extract_all = false;
    int     n_seqs      = 0;   // number of sequences in current batch
};

// ── Eval callback — extracts per-sequence slices from the concatenated tensor ──

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

    // ── Read tensor data ──
    int64_t ne0 = t->ne[0];  // hidden_dim
    // ne1 = total positions across ALL sequences in this batch
    cb->hidden_dim = ne0;

    size_t n_bytes = ggml_nbytes(t);
    bool is_host = ggml_backend_buffer_is_host(t->buffer);

    const uint8_t * data_ptr;
    if (is_host) {
        data_ptr = static_cast<const uint8_t *>(t->data);
    } else {
        cb->temp_buf.resize(n_bytes);
        ggml_backend_tensor_get(t, cb->temp_buf.data(), 0, n_bytes);
        data_ptr = cb->temp_buf.data();
    }

    // ── Extract each sequence's slice ──
    for (int sid = 0; sid < cb->n_seqs; sid++) {
        const auto & sr = cb->seq_ranges[sid];

        int64_t abs_start = sr.tensor_start + sr.extract_start;
        int64_t n_pos     = sr.n_extract;
        if (n_pos <= 0) continue;

        std::vector<float> float_data(ne0 * n_pos);

        for (int64_t p = 0; p < n_pos; p++) {
            int64_t src_row = abs_start + p;

            if (t->type == GGML_TYPE_F32) {
                const float * src = reinterpret_cast<const float *>(data_ptr) + src_row * ne0;
                memcpy(float_data.data() + p * ne0, src, ne0 * sizeof(float));
            } else if (t->type == GGML_TYPE_F16) {
                const ggml_fp16_t * src =
                    reinterpret_cast<const ggml_fp16_t *>(data_ptr) + src_row * ne0;
                for (int64_t i = 0; i < ne0; i++) {
                    float_data[p * ne0 + i] = ggml_fp16_to_fp32(src[i]);
                }
            } else if (t->type == GGML_TYPE_BF16) {
                const ggml_bf16_t * src =
                    reinterpret_cast<const ggml_bf16_t *>(data_ptr) + src_row * ne0;
                for (int64_t i = 0; i < ne0; i++) {
                    float_data[p * ne0 + i] = ggml_bf16_to_fp32(src[i]);
                }
            } else {
                fprintf(stderr, "[extract] WARNING: unsupported tensor type %s for %s\n",
                        ggml_type_name(t->type), t->name);
                return true;
            }
        }

        cb->captured[sid][layer_idx] = std::move(float_data);
    }

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

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++;  // skip opening quote

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

        if (rec.prompt.empty()) rec.prompt   = json_get_string(line, "text");
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

// ── Save helpers (identical to extract_single_position.cpp) ──

static bool save_anchor_hidden_f16(const std::string & path,
                                    const std::map<int, std::vector<float>> & captured,
                                    const std::set<int> & target_layers,
                                    int64_t hidden_dim) {
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

    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(fp16_data.data()),
                total_elements * sizeof(uint16_t));
        f.flush();
        if (!f) { fs::remove(tmp_path); return false; }
    }
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) { fs::remove(tmp_path); return false; }
    return true;
}

static bool save_context_hidden_f16(const std::string & path,
                                     const std::map<int, std::vector<float>> & captured,
                                     const std::set<int> & target_layers,
                                     int64_t hidden_dim, int64_t n_positions) {
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

    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char *>(fp16_data.data()),
                total_elements * sizeof(uint16_t));
        f.flush();
        if (!f) { fs::remove(tmp_path); return false; }
    }
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) { fs::remove(tmp_path); return false; }
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
        if (!f) { fs::remove(tmp_path); return false; }
    }
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) { fs::remove(tmp_path); return false; }
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
        if (!f) { fs::remove(tmp_path); return false; }
    }
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) { fs::remove(tmp_path); return false; }
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
        if (!f) { fs::remove(tmp_path); return false; }
    }
    std::error_code ec;
    fs::rename(tmp_path, path, ec);
    if (ec) { fs::remove(tmp_path); return false; }
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
    std::string tmp  = path + ".tmp";
    {
        std::ofstream f(tmp);
        f << "{\"prompt_idx\": " << prompt_idx << ", \"n_written\": " << n_written << "}\n";
    }
    std::error_code ec;
    fs::rename(tmp, path, ec);
}

// ── Prepare a batch_seq_entry from a sample_record ──

static bool prepare_seq_entry(const sample_record & rec,
                               size_t sample_idx,
                               llama_seq_id seq_id,
                               const extract_config & ecfg,
                               llama_context * ctx,
                               bool add_bos,
                               batch_seq_entry & out) {
    std::string full_text = rec.prompt;
    if (!rec.response.empty()) full_text += "\n" + rec.response;

    std::vector<llama_token> tokens = common_tokenize(ctx, full_text, add_bos);
    int seq_len = std::min((int)tokens.size(), ecfg.max_seq_len);
    tokens.resize(seq_len);

    if (seq_len < ecfg.block_size + 2) return false;

    int anchor_pos = seq_len - ecfg.block_size - 1;
    if (anchor_pos < 0) return false;

    int eval_len = ecfg.extract_all ? seq_len : anchor_pos + 1;

    out.sample_idx  = sample_idx;
    out.tokens      = std::move(tokens);
    out.eval_len    = eval_len;
    out.seq_len     = seq_len;
    out.anchor_pos  = anchor_pos;
    out.seq_id      = seq_id;
    out.tensor_start = -1;  // filled later
    return true;
}

// ── Save one sequence's results to disk ──
// Returns true on success.

static bool save_seq_result(
    const batch_seq_entry & se,
    const std::map<int, std::vector<float>> & captured,
    int64_t hidden_dim,
    int64_t n_positions,
    const extract_config & ecfg,
    const std::string & source_id,
    size_t sample_out_idx)
{
    char dir_name[64];
    snprintf(dir_name, sizeof(dir_name), "sample_%06zu", sample_out_idx);
    std::string sample_dir = ecfg.output_dir + "/" + dir_name;
    fs::create_directories(sample_dir);

    bool ok;

    if (ecfg.extract_all) {
        // Full-sequence mode: context_hidden.bin + tokens.bin
        ok = save_context_hidden_f16(
            sample_dir + "/context_hidden.bin",
            captured, ecfg.target_layers, hidden_dim, n_positions);
        if (!ok) return false;

        ok = save_tokens(sample_dir + "/tokens.bin", se.tokens, se.seq_len);
        if (!ok) return false;
    } else {
        if (ecfg.ctx_len == 1) {
            ok = save_anchor_hidden_f16(
                sample_dir + "/anchor_hidden.bin",
                captured, ecfg.target_layers, hidden_dim);
        } else {
            ok = save_context_hidden_f16(
                sample_dir + "/context_hidden.bin",
                captured, ecfg.target_layers, hidden_dim, n_positions);
        }
        if (!ok) return false;

        // Block tokens
        std::vector<int32_t> block_tokens(ecfg.block_size, MASK_TOKEN_ID);
        for (int j = 0; j < ecfg.block_size; j++) {
            int tok_idx = se.anchor_pos + 1 + j;
            if (tok_idx < se.seq_len) {
                block_tokens[j] = (int32_t)se.tokens[tok_idx];
            }
        }
        ok = save_block_tokens(sample_dir + "/block_tokens.bin", block_tokens);
        if (!ok) return false;
    }

    std::string mode_str = ecfg.extract_all ? "full_seq" : "anchor";
    ok = save_metadata(
        sample_dir + "/metadata.json",
        se.anchor_pos, se.seq_len, ecfg.block_size,
        ecfg.extract_all ? 0 : ecfg.ctx_len, (int)n_positions,
        ecfg.target_layers, (int)hidden_dim, source_id, mode_str);
    return ok;
}

// ── Single-sequence fallback decode for one entry ──
// Used when a batched decode fails. Mirrors the logic of extract_single_position.
// Returns true if extraction + save succeeded.

static bool decode_and_save_single(
    const batch_seq_entry & se,
    const sample_record & rec,
    const extract_config & ecfg,
    llama_context * ctx,
    extract_callback_data & cb_data,
    size_t sample_out_idx,
    size_t prompt_idx)
{
    // Reset callback for single-seq use
    cb_data.n_seqs = 1;
    cb_data.captured.assign(1, {});
    cb_data.seq_ranges.resize(1);

    int64_t extract_start = ecfg.extract_all
        ? 0
        : std::max(0, se.eval_len - ecfg.ctx_len);
    int64_t n_extract = se.eval_len - extract_start;

    cb_data.seq_ranges[0] = {0, (int64_t)se.eval_len, extract_start, n_extract};
    cb_data.hidden_dim    = 0;

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(se.eval_len, 0, 1);
    for (int i = 0; i < se.eval_len; i++) {
        common_batch_add(batch, se.tokens[i], i, {0}, i == se.eval_len - 1);
    }

    int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        fprintf(stderr, "[extract] WARNING: single fallback decode failed for prompt %zu (ret=%d)\n",
                prompt_idx, ret);
        return false;
    }

    if (cb_data.captured[0].size() != ecfg.target_layers.size()) {
        fprintf(stderr, "[extract] WARNING: single fallback captured %zu/%zu layers for prompt %zu\n",
                cb_data.captured[0].size(), ecfg.target_layers.size(), prompt_idx);
        return false;
    }

    for (const auto & [lid, data] : cb_data.captured[0]) {
        if (has_nan_or_inf(data)) {
            fprintf(stderr, "[extract] WARNING: NaN/Inf in single fallback for prompt %zu\n", prompt_idx);
            return false;
        }
    }

    return save_seq_result(se, cb_data.captured[0], cb_data.hidden_dim,
                           n_extract, ecfg, rec.id, sample_out_idx);
}

// ── Usage ──

static void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s [llama-args] --layers 1,10,19,28,37 --input prompts.jsonl --output features/\n\n", prog);
    fprintf(stderr, "DFlash batched hidden state extractor.\n");
    fprintf(stderr, "Processes N prompts per llama_decode call using seq_id batching.\n\n");
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
    fprintf(stderr, "  --batch-seqs N       Sequences per llama_decode call (default: %d)\n", DEFAULT_BATCH_SEQS);
    fprintf(stderr, "  --no-sort            Do not sort prompts by length\n");
    fprintf(stderr, "  --extract-all        Extract ALL positions (full-sequence mode)\n");
    fprintf(stderr, "\nIMPORTANT: Set -c >= batch_seqs * max_seq_len (e.g. -c 2048 for 4×512).\n");
    fprintf(stderr, "All standard llama.cpp arguments (-m, -ngl, -ot, etc.) are also supported.\n");
}

// ── Main ──

int main(int argc, char ** argv) {
    install_signal_handlers();

    extract_config ecfg;

    // Pre-parse custom args, pass the rest to llama
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
        } else if (arg == "--batch-seqs" && i + 1 < argc) {
            ecfg.batch_seqs = std::stoi(argv[++i]);
            if (ecfg.batch_seqs < 1) {
                fprintf(stderr, "[extract] ERROR: --batch-seqs must be >= 1\n");
                return 1;
            }
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

    // Validate required args
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

    // Warn if context size might be too small
    {
        int ctx_size = (int)llama_n_ctx(ctx);
        int min_ctx  = ecfg.batch_seqs * ecfg.max_seq_len;
        if (ctx_size < min_ctx) {
            fprintf(stderr,
                    "[extract] WARNING: context size %d < batch_seqs(%d) * max_seq_len(%d) = %d.\n"
                    "          Batched decodes may fail. Use -c %d or reduce --batch-seqs.\n",
                    ctx_size, ecfg.batch_seqs, ecfg.max_seq_len, min_ctx, min_ctx);
        }
    }

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
    fprintf(stderr, " DFlash C++ Batched Hidden State Extraction\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Model:            %s\n", params.model.path.c_str());
    fprintf(stderr, "  Layers:           ");
    for (int l : ecfg.target_layers) fprintf(stderr, "%d ", l);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Block size:       %d\n", ecfg.block_size);
    fprintf(stderr, "  Context len:      %d\n", ecfg.ctx_len);
    fprintf(stderr, "  Max seq len:      %d\n", ecfg.max_seq_len);
    fprintf(stderr, "  Batch seqs:       %d\n", ecfg.batch_seqs);
    fprintf(stderr, "  Context size:     %d (need >= %d)\n",
            (int)llama_n_ctx(ctx), ecfg.batch_seqs * ecfg.max_seq_len);
    fprintf(stderr, "  Tensor:           %s\n", ecfg.tensor_name.c_str());
    fprintf(stderr, "  Prompts:          %zu\n", samples.size());
    fprintf(stderr, "  Resume from:      %d\n", ecfg.resume_from);
    fprintf(stderr, "  Existing samples: %zu\n", existing);
    fprintf(stderr, "  Extract all:      %s\n", ecfg.extract_all ? "yes (full-seq)" : "no");
    fprintf(stderr, "  Sorted by length: %s\n", ecfg.sort_by_len ? "yes" : "no");
    fprintf(stderr, "\n");

    // ── Extraction loop ──

    size_t n_written   = existing;
    size_t n_failed    = 0;
    size_t n_skipped   = 0;
    size_t n_batch_fallback = 0;
    uint64_t total_tokens = 0;

    auto t_start = std::chrono::steady_clock::now();

    size_t si = (size_t)ecfg.resume_from;

    while (si < samples.size()) {
        if (g_shutdown.load(std::memory_order_relaxed)) {
            fprintf(stderr, "\n[extract] Graceful shutdown at prompt %zu\n", si);
            save_checkpoint(ecfg.output_dir, si, n_written);
            break;
        }

        // Progress reporting every 100 samples (or first iteration)
        if (si % 100 == 0 || si == (size_t)ecfg.resume_from) {
            auto now     = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - t_start).count();
            size_t done  = si - ecfg.resume_from;
            double rate  = done / (elapsed + 1e-9);
            double tok_rate = total_tokens / (elapsed + 1e-9);
            double eta_h = (samples.size() - si) / (rate + 1e-9) / 3600.0;
            fprintf(stderr, "  [%zu/%zu] %.1f prompts/s | %.0f tok/s | written=%zu | "
                            "failed=%zu | fallback=%zu | ETA=%.1fh\n",
                    si, samples.size(), rate, tok_rate,
                    n_written, n_failed, n_batch_fallback, eta_h);
        }

        // ── Build a batch of up to batch_seqs sequences ──

        std::vector<batch_seq_entry> batch_entries;
        std::vector<size_t>          batch_sample_indices;  // original si for each entry
        batch_entries.reserve(ecfg.batch_seqs);

        size_t si_scan = si;
        while ((int)batch_entries.size() < ecfg.batch_seqs && si_scan < samples.size()) {
            batch_seq_entry se;
            llama_seq_id sid = (llama_seq_id)batch_entries.size();

            if (!prepare_seq_entry(samples[si_scan], si_scan, sid, ecfg,
                                   ctx, add_bos, se)) {
                // Too short — skip immediately
                n_skipped++;
                si_scan++;
                continue;
            }
            total_tokens += se.eval_len;
            batch_entries.push_back(std::move(se));
            batch_sample_indices.push_back(si_scan);
            si_scan++;
        }

        // Advance main cursor past everything we scanned (including skips)
        si = si_scan;

        if (batch_entries.empty()) {
            // All remaining samples in this window were too short; continue
            continue;
        }

        // Sort entries within the batch by eval_len (ascending) for minimal padding variance
        // (This does NOT affect which sample_idx/seq_id they carry.)
        // Note: seq_id was assigned above; sort won't break it since we use batch_entries[i].seq_id.
        // Actually we need stable seq_id assignment to match callback ranges, so we re-assign
        // seq_ids after sorting.
        std::sort(batch_entries.begin(), batch_entries.end(),
                  [](const batch_seq_entry & a, const batch_seq_entry & b) {
                      return a.eval_len < b.eval_len;
                  });
        for (int k = 0; k < (int)batch_entries.size(); k++) {
            batch_entries[k].seq_id = (llama_seq_id)k;
        }

        int n_seqs_this_batch = (int)batch_entries.size();

        // ── Compute tensor layout ──
        // Sequences are concatenated in seq_id order (0, 1, ..., n-1).
        // tensor_start for seq k = sum of eval_len of seqs 0..k-1.
        {
            int64_t running_start = 0;
            for (auto & se : batch_entries) {
                se.tensor_start = running_start;
                running_start  += se.eval_len;
            }
        }

        // ── Configure callback ──
        cb_data.n_seqs = n_seqs_this_batch;
        cb_data.captured.assign(n_seqs_this_batch, {});
        cb_data.seq_ranges.resize(n_seqs_this_batch);
        cb_data.hidden_dim = 0;

        for (int k = 0; k < n_seqs_this_batch; k++) {
            const auto & se = batch_entries[k];
            int64_t extract_start = ecfg.extract_all
                ? 0
                : std::max(0, se.eval_len - ecfg.ctx_len);
            int64_t n_extract = se.eval_len - extract_start;

            cb_data.seq_ranges[k] = {
                se.tensor_start,
                (int64_t)se.eval_len,
                extract_start,
                n_extract
            };
        }

        // ── Build the llama_batch ──
        int total_batch_tokens = 0;
        for (const auto & se : batch_entries) total_batch_tokens += se.eval_len;

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_batch batch = llama_batch_init(total_batch_tokens, 0, n_seqs_this_batch);

        for (int k = 0; k < n_seqs_this_batch; k++) {
            const auto & se = batch_entries[k];
            for (int p = 0; p < se.eval_len; p++) {
                bool is_last = (p == se.eval_len - 1);
                common_batch_add(batch, se.tokens[p], p, {se.seq_id}, is_last);
            }
        }

        // ── Decode ──
        int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (ret != 0) {
            // Batched decode failed — fall back to single-sequence for each entry
            fprintf(stderr,
                    "[extract] WARNING: batched decode failed (ret=%d) for batch starting at prompt %zu. "
                    "Falling back to single-sequence.\n",
                    ret, batch_sample_indices.empty() ? si : batch_sample_indices[0]);
            n_batch_fallback++;

            for (int k = 0; k < n_seqs_this_batch; k++) {
                const auto & se  = batch_entries[k];
                size_t orig_si   = batch_sample_indices[k];
                const auto & rec = samples[orig_si];

                bool ok = decode_and_save_single(se, rec, ecfg, ctx, cb_data, n_written, orig_si);
                if (ok) {
                    n_written++;
                } else {
                    n_failed++;
                }
            }
            continue;
        }

        // ── Save results for each sequence ──
        for (int k = 0; k < n_seqs_this_batch; k++) {
            const auto & se      = batch_entries[k];
            size_t orig_si       = batch_sample_indices[k];
            const auto & rec     = samples[orig_si];
            const auto & sr      = cb_data.seq_ranges[k];

            // Verify all layers captured
            if (cb_data.captured[k].size() != ecfg.target_layers.size()) {
                fprintf(stderr,
                        "[extract] WARNING: seq %d of batch (prompt %zu) captured %zu/%zu layers\n",
                        k, orig_si, cb_data.captured[k].size(), ecfg.target_layers.size());
                n_failed++;
                continue;
            }

            // NaN check
            bool has_bad = false;
            for (const auto & [lid, data] : cb_data.captured[k]) {
                if (has_nan_or_inf(data)) { has_bad = true; break; }
            }
            if (has_bad) {
                fprintf(stderr,
                        "[extract] WARNING: NaN/Inf in hidden states for prompt %zu (seq %d)\n",
                        orig_si, k);
                n_failed++;
                continue;
            }

            bool ok = save_seq_result(
                se, cb_data.captured[k],
                cb_data.hidden_dim, sr.n_extract,
                ecfg, rec.id, n_written);

            if (!ok) {
                fprintf(stderr,
                        "[extract] WARNING: failed to save prompt %zu (seq %d)\n",
                        orig_si, k);
                n_failed++;
                continue;
            }

            n_written++;
        }
    }

    // ── Summary ──

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    uint64_t total_bytes = 0;
    for (auto & entry : fs::recursive_directory_iterator(ecfg.output_dir)) {
        if (entry.is_regular_file()) {
            total_bytes += entry.file_size();
        }
    }

    size_t n_processed = samples.size() - ecfg.resume_from;

    fprintf(stderr, "\n============================================================\n");
    fprintf(stderr, " Extraction Summary\n");
    fprintf(stderr, "============================================================\n");
    fprintf(stderr, "  Samples written:  %zu\n", n_written);
    fprintf(stderr, "  Samples failed:   %zu\n", n_failed);
    fprintf(stderr, "  Samples skipped:  %zu (too short)\n", n_skipped);
    fprintf(stderr, "  Batch fallbacks:  %zu\n", n_batch_fallback);
    fprintf(stderr, "  Total tokens:     %llu\n", (unsigned long long)total_tokens);
    fprintf(stderr, "  Total size:       %.1f MB\n", total_bytes / 1e6);
    fprintf(stderr, "  Per sample:       %.1f KB\n",
            total_bytes / (double)std::max(n_written, (size_t)1) / 1024.0);
    fprintf(stderr, "  Time:             %.1fh\n", elapsed / 3600.0);
    fprintf(stderr, "  Rate:             %.1f prompts/s\n",
            n_processed / (elapsed + 1e-9));
    fprintf(stderr, "  Token rate:       %.0f tok/s\n",
            total_tokens / (elapsed + 1e-9));
    fprintf(stderr, "============================================================\n\n");

    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
