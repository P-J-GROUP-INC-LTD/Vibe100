// include/strata/ds41/session/trace.hpp - DS1-E: the engine's side of DS1-F's trace format (ref/ds41/trace_io.py, docs/deepseek/DS1_VERIFY.md section 1) and the logits dump of
// tools/volta/golden_compare.py.
//
//   TraceWriter    one directory: `trace.json` (format ds41-trace v1, producer "engine", geometry, tokens, n_prompt, quant flags, positions, stages: rewritten by flush()) and
//                  one little-endian .npy per (stage, layer, position): `<stage>.p<POS>.npy` / `<stage>.L<LL>.p<POS>.npy`, f4 or i4, numpy format version 1.0, C order.
//   LogitsDump     int32 n_vocab, int32 n_rows, then rows of n_vocab float32 (the row count in the header is kept current after every row).
#pragma once

#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace strata::ds41::session {

/// QuantConfig's flags as trace.json records them (the oracle replays an engine trace with the flags the engine ran with).
struct TraceQuant {
    bool int8_act = true;
    bool window_kv = true;
    bool compressed_kv = true;
    bool index = true;
    bool linear_act = false;
};

class TraceWriter {
public:
    /// Creates `dir` (and parents) and writes an initial trace.json.  Throws std::runtime_error when the directory cannot be made.
    TraceWriter(const std::string& dir, const std::string& geometry, const TraceQuant& quant, int n_prompt);
    /// A float stage: `layer` < 0 for a stage that exists once per position (embed, final_hidden, logits).  shape: the array's dimensions (their product is n).
    void put_f32(const char* stage, int pos, int layer, const float* data, const std::vector<int64_t>& shape);
    void put_i32(const char* stage, int pos, int layer, const int32_t* data, int64_t n);
    /// The tokens of the sequence so far (prompt, then generated) and the prompt length: rewritten into trace.json by flush().
    void set_tokens(const std::vector<int32_t>& tokens, int n_prompt);
    /// Rewrites trace.json (the file is small; the engine does it after every token so that a crashed run leaves a readable trace).
    void flush();
    const std::string& dir() const { return dir_; }
    uint64_t files() const { return files_; }
    uint64_t bytes() const { return bytes_; }

private:
    void write_npy(const std::string& name, const char* descr, const void* data, size_t elem, const std::vector<int64_t>& shape);
    std::string dir_, geometry_;
    TraceQuant quant_;
    int n_prompt_ = 0;
    std::vector<int32_t> tokens_;
    std::set<int> positions_;
    std::set<std::string> stages_;
    uint64_t files_ = 0, bytes_ = 0;
};

class LogitsDump {
public:
    /// Opens `path` for writing; `n_vocab` is the row width.  Throws std::runtime_error when the file cannot be created.
    LogitsDump(const std::string& path, int n_vocab);
    ~LogitsDump();
    LogitsDump(const LogitsDump&) = delete;
    LogitsDump& operator=(const LogitsDump&) = delete;
    void add_row(const float* row);
    int rows() const { return rows_; }

private:
    void write_header();
    std::FILE* f_ = nullptr;
    int n_vocab_ = 0, rows_ = 0;
    std::string path_;
};

}  // namespace strata::ds41::session
