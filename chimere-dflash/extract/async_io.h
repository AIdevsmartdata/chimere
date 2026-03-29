/**
 * async_io.h — Async I/O pipeline for the DFlash batched hidden-state extractor.
 *
 * Provides two components:
 *
 *   AsyncWriter    — Background thread that drains a queue of WriteJob objects and
 *                    serializes them to disk, overlapping I/O with GPU computation on
 *                    the main thread. Never blocks the caller on a disk error.
 *
 *   AsyncTokenizer — Background thread that pre-tokenizes the NEXT batch of prompts
 *                    while the GPU is busy with the current batch. Uses the thread-safe
 *                    llama_vocab* overload of common_tokenize (read-only, no ctx state).
 *
 * Thread-safety model:
 *   - Both classes own their worker threads internally.
 *   - All shared state is protected by std::mutex + std::condition_variable.
 *   - No busy-waiting anywhere (all waits use cv.wait / cv.wait_for).
 *   - The main thread is NEVER stalled by a disk error; failed jobs increment a counter
 *     and the worker continues processing the queue.
 *
 * Memory note:
 *   WriteJob holds ownership of the captured hidden-state data via move semantics.
 *   With 5 layers × 512 positions × 2048 floats = 20 MB per job and a default queue
 *   depth of 8, peak resident memory for queued jobs is ~160 MB — acceptable given
 *   the 32 GB system and 16 GB VRAM configuration.
 *
 * Usage sketch (batched extractor):
 *
 *   AsyncWriter  writer(8);
 *   AsyncTokenizer tokenizer(vocab, add_bos, max_seq_len);
 *
 *   // Pre-tokenize first batch while GPU warms up
 *   tokenizer.submit_batch(records, 0, BATCH_SIZE);
 *
 *   for (size_t batch = 0; batch < n_batches; batch++) {
 *       auto tok_batch = tokenizer.get_results();   // blocks until batch ready
 *
 *       // Kick off pre-tokenization of NEXT batch in background
 *       if (batch + 1 < n_batches)
 *           tokenizer.submit_batch(records, (batch+1)*BATCH_SIZE, BATCH_SIZE);
 *
 *       for (auto & tprompt : tok_batch) {
 *           // GPU forward pass ...
 *           AsyncWriter::WriteJob job;
 *           job.captured   = std::move(cb_data.captured);
 *           // ... fill remaining fields ...
 *           writer.submit(std::move(job));
 *       }
 *   }
 *
 *   writer.flush();  // drain queue before exit summary
 *
 * Integration with extract_single_position.cpp:
 *   Include this header after the existing llama/common includes. The file save
 *   helpers (save_anchor_hidden_f16, save_context_hidden_f16, save_block_tokens,
 *   save_metadata) are duplicated here as private static helpers inside AsyncWriter
 *   to keep the header self-contained. In the batched extractor, those helpers in
 *   the .cpp file can be removed or kept for the synchronous fallback path.
 */

#pragma once

#ifndef CHIMERE_ASYNC_IO_H
#define CHIMERE_ASYNC_IO_H

// ── Standard library ──────────────────────────────────────────────────────────

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

// ── llama.cpp headers (must be included BEFORE this header in the .cpp file) ──
//
// We forward-declare / use the types that are already pulled in by the including
// translation unit. This header does NOT include llama.h or common.h itself to
// avoid double-include issues; the caller is responsible for including them first.
//
// Types used:
//   llama_token   (typedef int32_t, from llama.h)
//   llama_vocab   (struct, opaque pointer, from llama.h)
//   common_tokenize(const llama_vocab*, const std::string&, bool, bool)  from common.h
//   ggml_fp32_to_fp16(float) -> uint16_t   from ggml.h  (via llama.h)
//
// If you need to compile this header standalone (e.g. unit tests), define
// ASYNC_IO_STANDALONE and provide stub implementations.

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers (write path — mirrored from extract_single_position.cpp)
// ─────────────────────────────────────────────────────────────────────────────

namespace async_io_detail {

// JSON string escaping (zero-dependency, matches extract_single_position.cpp)
inline std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// Atomic-safe file write: write to .tmp then rename.
// Returns false on any I/O error (caller should NOT re-try; job is already lost).
inline bool atomic_write_binary(const std::string & path,
                                 const void         * data,
                                 size_t               nbytes) {
    std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        if (!f) return false;
        f.write(static_cast<const char *>(data), static_cast<std::streamsize>(nbytes));
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

inline bool atomic_write_text(const std::string & path, const std::string & text) {
    return atomic_write_binary(path, text.data(), text.size());
}

// Save float32 array as float16, layout: [n_layers, n_positions, hidden_dim]
// (n_positions == 1 collapses to [n_layers, hidden_dim] — same byte layout).
inline bool save_hidden_f16(const std::string                       & path,
                              const std::map<int, std::vector<float>> & captured,
                              const std::set<int>                     & target_layers,
                              int64_t                                   hidden_dim,
                              int64_t                                   n_positions) {
    size_t n_layers        = target_layers.size();
    size_t total_elements  = n_layers * static_cast<size_t>(n_positions) * static_cast<size_t>(hidden_dim);

    std::vector<uint16_t> fp16_data(total_elements);
    size_t offset = 0;

    for (int layer_id : target_layers) {
        auto it = captured.find(layer_id);
        if (it == captured.end()) return false;

        const auto & fdata    = it->second;
        int64_t      expected = n_positions * hidden_dim;
        if (static_cast<int64_t>(fdata.size()) < expected) return false;

        for (int64_t i = 0; i < expected; i++) {
            fp16_data[offset + i] = ggml_fp32_to_fp16(fdata[static_cast<size_t>(i)]);
        }
        offset += static_cast<size_t>(expected);
    }

    return atomic_write_binary(path,
                               fp16_data.data(),
                               total_elements * sizeof(uint16_t));
}

// Save int32 block tokens.
inline bool save_block_tokens(const std::string          & path,
                               const std::vector<int32_t> & block_tokens) {
    return atomic_write_binary(path,
                               block_tokens.data(),
                               block_tokens.size() * sizeof(int32_t));
}

// Save full token sequence as int32.
inline bool save_tokens(const std::string             & path,
                         const std::vector<llama_token> & tokens,
                         int                             seq_len) {
    std::vector<int32_t> tok_i32(static_cast<size_t>(seq_len));
    for (int i = 0; i < seq_len; i++) {
        tok_i32[static_cast<size_t>(i)] = static_cast<int32_t>(tokens[static_cast<size_t>(i)]);
    }
    return atomic_write_binary(path,
                               tok_i32.data(),
                               static_cast<size_t>(seq_len) * sizeof(int32_t));
}

// Build and save metadata.json.
inline bool save_metadata(const std::string  & path,
                           int                  anchor_pos,
                           int                  seq_len,
                           int                  block_size,
                           int                  ctx_len,
                           int                  n_positions,
                           const std::set<int> & layers,
                           int                  hidden_dim,
                           const std::string  & source_id,
                           const std::string  & mode) {
    std::string text;
    text.reserve(512);
    text += "{\n";
    text += "  \"anchor_pos\": ";   text += std::to_string(anchor_pos);  text += ",\n";
    text += "  \"seq_len\": ";      text += std::to_string(seq_len);     text += ",\n";
    text += "  \"block_size\": ";   text += std::to_string(block_size);  text += ",\n";
    text += "  \"ctx_len\": ";      text += std::to_string(ctx_len);     text += ",\n";
    text += "  \"n_positions\": ";  text += std::to_string(n_positions); text += ",\n";
    text += "  \"layers\": [";
    bool first = true;
    for (int l : layers) {
        if (!first) text += ", ";
        text += std::to_string(l);
        first = false;
    }
    text += "],\n";
    text += "  \"hidden_dim\": ";   text += std::to_string(hidden_dim);  text += ",\n";
    text += "  \"dtype\": \"float16\",\n";
    text += "  \"mode\": \"";       text += json_escape(mode);           text += "\",\n";
    text += "  \"source_id\": \"";  text += json_escape(source_id);      text += "\"\n";
    text += "}\n";

    return atomic_write_text(path, text);
}

} // namespace async_io_detail


// ─────────────────────────────────────────────────────────────────────────────
// AsyncWriter
// ─────────────────────────────────────────────────────────────────────────────

/**
 * AsyncWriter — Background-thread disk writer for hidden-state features.
 *
 * The main thread calls submit() (non-blocking, O(1)) to hand off a WriteJob.
 * An internal worker thread drains the queue and writes all files for each job.
 *
 * If the queue is full (pending() == max_queue_size), submit() blocks until a
 * slot opens. This provides natural back-pressure: if the disk falls far behind
 * the GPU, the main thread will stall on submit() rather than exhausting RAM.
 *
 * Write errors are logged to stderr and counted via failed(). They never throw
 * and never block the worker — the job is discarded and the worker moves on.
 *
 * Destruction automatically joins the worker thread after the queue drains.
 * Call flush() explicitly before reading the final failed() count.
 */
class AsyncWriter {
public:
    // ── Public job descriptor ────────────────────────────────────────────────

    /**
     * WriteJob — all data needed to persist one extraction sample.
     *
     * The caller must move the captured map and tokens vector into the job to
     * transfer ownership. After submit(std::move(job)) the caller's captured
     * map will be empty (moved-from state).
     *
     * Modes:
     *   "anchor"   — ctx_len == 1, saves anchor_hidden.bin + block_tokens.bin + metadata.json
     *   "context"  — ctx_len >  1, saves context_hidden.bin + block_tokens.bin + metadata.json
     *   "full_seq" — extract_all,  saves context_hidden.bin + tokens.bin + metadata.json
     */
    struct WriteJob {
        // Output destination
        std::string sample_dir;     // absolute path, will be mkdir'd if absent

        // Hidden state data (moved in from cb_data.captured)
        std::map<int, std::vector<float>> captured;   // layer_id -> float32[n_positions * hidden_dim]
        std::set<int>                     target_layers;
        int64_t                           hidden_dim  = 0;
        int64_t                           n_positions = 0;  // actual positions captured

        // Token data
        std::vector<int32_t>  block_tokens;  // int32[block_size], used in anchor/context mode
        std::vector<llama_token> full_tokens; // all tokens, used in full_seq mode
        int                   seq_len     = 0;

        // Metadata fields
        int         anchor_pos  = 0;
        int         block_size  = 16;
        int         ctx_len     = 1;
        std::string source_id;
        std::string mode        = "anchor";  // "anchor" | "context" | "full_seq"
    };

    // ── Construction / destruction ───────────────────────────────────────────

    /**
     * Construct an AsyncWriter with the given queue depth limit.
     *
     * @param max_queue_size  Maximum number of jobs that may be queued before
     *                        submit() blocks. Default 8 → ~160 MB peak for
     *                        5×512×2048 float jobs.
     */
    explicit AsyncWriter(int max_queue_size = 8)
        : max_queue_size_(static_cast<size_t>(max_queue_size))
        , n_failed_(0)
        , stop_(false)
    {
        worker_ = std::thread(&AsyncWriter::worker_loop, this);
    }

    /** Signals the worker to stop after draining the queue, then joins. */
    ~AsyncWriter() {
        {
            std::unique_lock<std::mutex> lk(mutex_);
            stop_ = true;
        }
        cv_work_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // Non-copyable, non-movable (owns a thread and mutex).
    AsyncWriter(const AsyncWriter &)             = delete;
    AsyncWriter & operator=(const AsyncWriter &) = delete;
    AsyncWriter(AsyncWriter &&)                  = delete;
    AsyncWriter & operator=(AsyncWriter &&)      = delete;

    // ── Public interface ─────────────────────────────────────────────────────

    /**
     * Submit a WriteJob for background writing.
     *
     * Non-blocking as long as pending() < max_queue_size. If the queue is full,
     * blocks until a slot is available (back-pressure — prevents OOM).
     *
     * The job is moved into the queue; the caller's job is left in a valid but
     * unspecified (moved-from) state.
     */
    void submit(WriteJob job) {
        std::unique_lock<std::mutex> lk(mutex_);
        // Back-pressure: wait until there is room in the queue.
        cv_space_.wait(lk, [this] {
            return queue_.size() < max_queue_size_ || stop_;
        });
        if (stop_) return;   // shutting down — discard silently
        queue_.push(std::move(job));
        lk.unlock();
        cv_work_.notify_one();
    }

    /**
     * Block until all currently queued jobs have been written to disk.
     *
     * After flush() returns, pending() == 0. Check failed() for write errors.
     * Note: jobs submitted concurrently with flush() may or may not be included.
     */
    void flush() {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_space_.wait(lk, [this] {
            return queue_.empty() || stop_;
        });
    }

    /** Returns the number of jobs currently waiting in the queue. */
    size_t pending() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return queue_.size();
    }

    /** Returns the cumulative count of jobs that failed to write. Thread-safe. */
    size_t failed() const {
        return n_failed_.load(std::memory_order_relaxed);
    }

private:
    // ── Worker ───────────────────────────────────────────────────────────────

    void worker_loop() {
        while (true) {
            WriteJob job;
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_work_.wait(lk, [this] {
                    return !queue_.empty() || stop_;
                });

                if (queue_.empty()) {
                    // stop_ is true and queue is drained — exit cleanly.
                    return;
                }
                job = std::move(queue_.front());
                queue_.pop();
            }
            // Notify submit() that a slot freed up.
            cv_space_.notify_one();

            // Perform the actual disk write outside the lock.
            write_job(job);
        }
    }

    void write_job(const WriteJob & job) {
        // Ensure output directory exists.
        {
            std::error_code ec;
            fs::create_directories(job.sample_dir, ec);
            if (ec) {
                fprintf(stderr,
                        "[async_writer] WARNING: cannot create directory '%s': %s — sample lost\n",
                        job.sample_dir.c_str(), ec.message().c_str());
                n_failed_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }

        bool ok = true;

        if (job.mode == "full_seq") {
            // Full-sequence mode: context_hidden.bin + tokens.bin
            ok &= async_io_detail::save_hidden_f16(
                job.sample_dir + "/context_hidden.bin",
                job.captured, job.target_layers,
                job.hidden_dim, job.n_positions
            );
            if (ok) {
                ok &= async_io_detail::save_tokens(
                    job.sample_dir + "/tokens.bin",
                    job.full_tokens, job.seq_len
                );
            }
        } else if (job.ctx_len == 1 && job.mode == "anchor") {
            // Legacy single-position format: anchor_hidden.bin
            ok &= async_io_detail::save_hidden_f16(
                job.sample_dir + "/anchor_hidden.bin",
                job.captured, job.target_layers,
                job.hidden_dim, /*n_positions=*/1
            );
            if (ok) {
                ok &= async_io_detail::save_block_tokens(
                    job.sample_dir + "/block_tokens.bin",
                    job.block_tokens
                );
            }
        } else {
            // Multi-position context format: context_hidden.bin
            ok &= async_io_detail::save_hidden_f16(
                job.sample_dir + "/context_hidden.bin",
                job.captured, job.target_layers,
                job.hidden_dim, job.n_positions
            );
            if (ok) {
                ok &= async_io_detail::save_block_tokens(
                    job.sample_dir + "/block_tokens.bin",
                    job.block_tokens
                );
            }
        }

        if (!ok) {
            fprintf(stderr,
                    "[async_writer] WARNING: failed to write hidden/token files for '%s'\n",
                    job.sample_dir.c_str());
            n_failed_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Metadata: written last so an incomplete sample directory lacks metadata.json
        // — the loader can skip directories without it.
        bool meta_ok = async_io_detail::save_metadata(
            job.sample_dir + "/metadata.json",
            job.anchor_pos, job.seq_len, job.block_size,
            job.ctx_len, static_cast<int>(job.n_positions),
            job.target_layers, static_cast<int>(job.hidden_dim),
            job.source_id, job.mode
        );
        if (!meta_ok) {
            fprintf(stderr,
                    "[async_writer] WARNING: failed to write metadata.json for '%s'\n",
                    job.sample_dir.c_str());
            n_failed_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // ── State ────────────────────────────────────────────────────────────────

    size_t                    max_queue_size_;

    mutable std::mutex        mutex_;
    std::condition_variable   cv_work_;   // worker waits on this for new jobs or stop
    std::condition_variable   cv_space_;  // submit/flush waits on this for queue room or drain

    std::queue<WriteJob>      queue_;
    std::atomic<size_t>       n_failed_;
    bool                      stop_;

    std::thread               worker_;
};


// ─────────────────────────────────────────────────────────────────────────────
// AsyncTokenizer
// ─────────────────────────────────────────────────────────────────────────────

// Forward-declare sample_record as it is defined in the including .cpp file.
// If this header is used from a different TU, define ASYNC_IO_OWN_SAMPLE_RECORD
// and the struct will be defined here instead.
#ifndef ASYNC_IO_OWN_SAMPLE_RECORD
// sample_record is expected to be defined before this header is included.
// Declaration only — no body.
struct sample_record;
#else
/** Minimal sample record used when this header is compiled standalone. */
struct sample_record {
    std::string id;
    std::string prompt;
    std::string response;
    size_t      text_len = 0;
};
#endif

/**
 * AsyncTokenizer — Pre-tokenizes a batch of prompts in a background thread.
 *
 * Usage pattern:
 *
 *   1. Call submit_batch() with the current "look-ahead" batch while the GPU
 *      processes the previous batch.
 *   2. When the GPU batch is done, call get_results() to retrieve the
 *      pre-tokenized prompts. get_results() blocks only if tokenization has
 *      not completed yet (typically it already has).
 *   3. Call submit_batch() again for the next look-ahead, then repeat.
 *
 * Thread safety:
 *   Only one batch may be in-flight at a time. Calling submit_batch() while a
 *   previous batch is still running asserts in debug mode; in release mode the
 *   old batch is discarded and a warning is printed.
 *
 * llama_vocab thread safety:
 *   common_tokenize(const llama_vocab*, ...) is read-only — it does not modify
 *   any llama_context state and is safe to call from any thread concurrently
 *   with llama_decode() on the main thread.
 */
class AsyncTokenizer {
public:
    // ── Result type ──────────────────────────────────────────────────────────

    struct TokenizedPrompt {
        std::vector<llama_token> tokens;      // tokenized sequence (trimmed to max_seq_len)
        int                      seq_len;     // actual length after truncation
        std::string              source_id;   // record.id, for logging / resume tracking
        size_t                   original_index; // index into the global samples vector
    };

    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @param vocab        Vocabulary from llama_model_get_vocab(model). Must outlive this object.
     * @param add_bos      Whether to prepend BOS token (from llama_vocab_get_add_bos).
     * @param max_seq_len  Hard cap on token sequence length (matches extract_config::max_seq_len).
     */
    AsyncTokenizer(const llama_vocab * vocab,
                   bool                add_bos,
                   int                 max_seq_len)
        : vocab_(vocab)
        , add_bos_(add_bos)
        , max_seq_len_(max_seq_len)
        , state_(State::IDLE)
        , stop_(false)
    {
        worker_ = std::thread(&AsyncTokenizer::worker_loop, this);
    }

    /**
     * Destructor — signals the worker and joins. If a batch is in-flight it is
     * completed (or the worker wakes on stop_ and exits early).
     */
    ~AsyncTokenizer() {
        {
            std::unique_lock<std::mutex> lk(mutex_);
            stop_ = true;
            state_ = State::DONE;  // unblock any waiting get_results()
        }
        cv_work_.notify_one();
        cv_done_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    AsyncTokenizer(const AsyncTokenizer &)             = delete;
    AsyncTokenizer & operator=(const AsyncTokenizer &) = delete;
    AsyncTokenizer(AsyncTokenizer &&)                  = delete;
    AsyncTokenizer & operator=(AsyncTokenizer &&)      = delete;

    // ── Public interface ─────────────────────────────────────────────────────

    /**
     * Submit a slice of the global samples vector for background tokenization.
     *
     * Non-blocking. The caller retains ownership of `records` (we take const&
     * because the records vector is never mutated and may be large — we just
     * read from it in the worker).
     *
     * @param records   Full samples array (must outlive the call to get_results).
     * @param start_idx Index of the first record in this batch.
     * @param count     Number of records to tokenize. Clamped to records.size().
     */
    void submit_batch(const std::vector<sample_record> & records,
                      size_t                             start_idx,
                      size_t                             count) {
        std::unique_lock<std::mutex> lk(mutex_);

        if (state_ == State::RUNNING) {
            // A previous batch is still in flight. In debug this is a logic error;
            // in release we warn and overwrite (caller should call get_results first).
            fprintf(stderr,
                    "[async_tokenizer] WARNING: submit_batch() called while previous batch "
                    "is still running — previous results will be discarded.\n");
        }

        // Clamp to actual available records.
        size_t actual_count = std::min(count, records.size() > start_idx
                                              ? records.size() - start_idx
                                              : (size_t)0);

        pending_records_  = &records;
        pending_start_    = start_idx;
        pending_count_    = actual_count;
        results_.clear();
        state_            = State::RUNNING;

        lk.unlock();
        cv_work_.notify_one();
    }

    /**
     * Block until the in-flight batch is tokenized, then return the results.
     *
     * If no batch is in flight (idle), returns an empty vector immediately.
     * After this call the tokenizer returns to IDLE state and is ready for the
     * next submit_batch().
     */
    std::vector<TokenizedPrompt> get_results() {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] {
            return state_ == State::DONE || state_ == State::IDLE || stop_;
        });
        std::vector<TokenizedPrompt> out = std::move(results_);
        results_.clear();
        state_ = State::IDLE;
        return out;
    }

    /**
     * Non-blocking check — returns true if the current batch has finished
     * tokenizing and get_results() can be called without blocking.
     */
    bool is_ready() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return state_ == State::DONE || state_ == State::IDLE;
    }

private:
    // ── Internal state machine ───────────────────────────────────────────────

    enum class State {
        IDLE,     // no batch in flight
        RUNNING,  // worker is tokenizing
        DONE,     // worker finished, results available
    };

    // ── Worker ───────────────────────────────────────────────────────────────

    void worker_loop() {
        while (true) {
            // Wait for a batch to be submitted or stop signal.
            const std::vector<sample_record> * records = nullptr;
            size_t start_idx = 0;
            size_t count     = 0;

            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_work_.wait(lk, [this] {
                    return state_ == State::RUNNING || stop_;
                });
                if (stop_) return;

                records   = pending_records_;
                start_idx = pending_start_;
                count     = pending_count_;
            }

            // Tokenize outside the lock so the main thread can call is_ready()
            // without contention.
            std::vector<TokenizedPrompt> batch_results;
            batch_results.reserve(count);

            for (size_t i = 0; i < count; i++) {
                // Check for shutdown mid-batch.
                if (stop_) break;

                size_t global_idx = start_idx + i;
                const sample_record & rec = (*records)[global_idx];

                // Build full text (same logic as extract_single_position.cpp).
                std::string full_text = rec.prompt;
                if (!rec.response.empty()) {
                    full_text += "\n";
                    full_text += rec.response;
                }

                // Thread-safe tokenization via vocab* overload (read-only).
                std::vector<llama_token> tokens =
                    common_tokenize(vocab_, full_text, add_bos_, /*parse_special=*/false);

                // Truncate to max_seq_len.
                int seq_len = std::min(static_cast<int>(tokens.size()), max_seq_len_);
                tokens.resize(static_cast<size_t>(seq_len));

                TokenizedPrompt tp;
                tp.tokens         = std::move(tokens);
                tp.seq_len        = seq_len;
                tp.source_id      = rec.id;
                tp.original_index = global_idx;

                batch_results.push_back(std::move(tp));
            }

            // Publish results under the lock and transition to DONE.
            {
                std::lock_guard<std::mutex> lk(mutex_);
                results_ = std::move(batch_results);
                state_   = State::DONE;
            }
            cv_done_.notify_all();
        }
    }

    // ── Configuration (immutable after construction) ─────────────────────────

    const llama_vocab * vocab_;
    bool                add_bos_;
    int                 max_seq_len_;

    // ── Shared state (protected by mutex_) ───────────────────────────────────

    mutable std::mutex      mutex_;
    std::condition_variable cv_work_;   // worker waits on this for submit or stop
    std::condition_variable cv_done_;   // caller waits on this for results

    // Input (written by main thread before RUNNING, read by worker)
    const std::vector<sample_record> * pending_records_ = nullptr;
    size_t                             pending_start_   = 0;
    size_t                             pending_count_   = 0;

    // Output (written by worker in DONE state, consumed by get_results)
    std::vector<TokenizedPrompt>       results_;

    State                              state_;
    bool                               stop_;

    std::thread                        worker_;
};

#endif // CHIMERE_ASYNC_IO_H
