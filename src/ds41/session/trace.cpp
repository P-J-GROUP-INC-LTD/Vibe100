// src/ds41/session/trace.cpp - DS1-E: the trace writer and the logits dump (include/strata/ds41/session/trace.hpp).
#include "strata/ds41/session/trace.hpp"

#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace strata::ds41::session {

namespace {
std::string shape_text(const std::vector<int64_t>& shape) {
    // numpy's repr: a 1-D shape is "(n,)", a 2-D one "(a, b)"
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
    if (shape.size() == 1) s += ",";
    return s + ")";
}
std::string json_escape(const std::string& s) {
    std::string r;
    for (char c : s) {
        if (c == '"' || c == '\\') r += '\\';
        r += c;
    }
    return r;
}
}  // namespace

TraceWriter::TraceWriter(const std::string& dir, const std::string& geometry, const TraceQuant& quant, int n_prompt) : dir_(dir), geometry_(geometry), quant_(quant), n_prompt_(n_prompt) {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec || !std::filesystem::is_directory(dir_)) throw std::runtime_error("trace: cannot create directory " + dir_ + (ec ? ": " + ec.message() : std::string()));
    flush();
}

void TraceWriter::write_npy(const std::string& name, const char* descr, const void* data, size_t elem, const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    std::string h = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape_text(shape) + ", }";
    // version 1.0: magic (6) + version (2) + header length (2) + header, the whole padded with spaces to a multiple of 64 and ended by a newline
    const size_t total = ((10 + h.size() + 1 + 63) / 64) * 64;
    h.append(total - 10 - h.size() - 1, ' ');
    h += '\n';
    const std::string path = dir_ + "/" + name;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("trace: cannot create " + path);
    const unsigned char magic[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char) (h.size() & 0xFF), (unsigned char) (h.size() >> 8)};
    bool ok = std::fwrite(magic, 1, 10, f) == 10 && std::fwrite(h.data(), 1, h.size(), f) == h.size();
    if (n > 0) ok = ok && std::fwrite(data, elem, (size_t) n, f) == (size_t) n;
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) throw std::runtime_error("trace: short write to " + path);
    ++files_;
    bytes_ += total + (uint64_t) n * elem;
}

static std::string stage_file(const char* stage, int pos, int layer) {
    char b[160];
    if (layer >= 0) std::snprintf(b, sizeof b, "%s.L%02d.p%05d.npy", stage, layer, pos);
    else std::snprintf(b, sizeof b, "%s.p%05d.npy", stage, pos);
    return b;
}

void TraceWriter::put_f32(const char* stage, int pos, int layer, const float* data, const std::vector<int64_t>& shape) {
    write_npy(stage_file(stage, pos, layer), "<f4", data, 4, shape);
    positions_.insert(pos);
    stages_.insert(stage);
}

void TraceWriter::put_i32(const char* stage, int pos, int layer, const int32_t* data, int64_t n) {
    write_npy(stage_file(stage, pos, layer), "<i4", data, 4, {n});
    positions_.insert(pos);
    stages_.insert(stage);
}

void TraceWriter::set_tokens(const std::vector<int32_t>& tokens, int n_prompt) {
    tokens_ = tokens;
    n_prompt_ = n_prompt;
}

void TraceWriter::flush() {
    std::string j = "{\n \"format\": \"ds41-trace\",\n \"version\": 1,\n \"producer\": \"engine\",\n \"geometry\": \"" + json_escape(geometry_) + "\",\n";
    j += " \"n_prompt\": " + std::to_string(n_prompt_) + ",\n";
    j += std::string(" \"quant\": {\"int8_act\": ") + (quant_.int8_act ? "true" : "false") + ", \"window_kv\": " + (quant_.window_kv ? "true" : "false") +
         ", \"compressed_kv\": " + (quant_.compressed_kv ? "true" : "false") + ", \"index\": " + (quant_.index ? "true" : "false") +
         ", \"linear_act\": " + (quant_.linear_act ? "true" : "false") + "},\n";
    j += " \"tokens\": [";
    for (size_t i = 0; i < tokens_.size(); ++i) j += (i ? ", " : "") + std::to_string(tokens_[i]);
    j += "],\n \"positions\": [";
    bool first = true;
    for (int p : positions_) {
        j += (first ? "" : ", ") + std::to_string(p);
        first = false;
    }
    j += "],\n \"stages\": [";
    first = true;
    for (const std::string& s : stages_) {
        j += std::string(first ? "" : ", ") + "\"" + s + "\"";
        first = false;
    }
    j += "]\n}\n";
    const std::string path = dir_ + "/trace.json";
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("trace: cannot write " + tmp);
    const bool ok = std::fwrite(j.data(), 1, j.size(), f) == j.size();
    if (std::fclose(f) != 0 || !ok) throw std::runtime_error("trace: short write to " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);                                // atomic: a reader never sees half a trace.json
    if (ec) throw std::runtime_error("trace: cannot rename " + tmp + ": " + ec.message());
}

// ================================================================================================ the logits dump
LogitsDump::LogitsDump(const std::string& path, int n_vocab) : n_vocab_(n_vocab), path_(path) {
    f_ = std::fopen(path.c_str(), "wb");
    if (!f_) throw std::runtime_error("cannot create the logits dump " + path);
    write_header();
}

void LogitsDump::write_header() {
    const int32_t h[2] = {n_vocab_, rows_};
    std::fseek(f_, 0, SEEK_SET);
    std::fwrite(h, 4, 2, f_);
    std::fseek(f_, 0, SEEK_END);
}

void LogitsDump::add_row(const float* row) {
    if (std::fwrite(row, 4, (size_t) n_vocab_, f_) != (size_t) n_vocab_) throw std::runtime_error("short write to the logits dump " + path_);
    ++rows_;
    write_header();
    std::fflush(f_);
}

LogitsDump::~LogitsDump() {
    if (f_) {
        write_header();
        std::fclose(f_);
    }
}

}  // namespace strata::ds41::session
