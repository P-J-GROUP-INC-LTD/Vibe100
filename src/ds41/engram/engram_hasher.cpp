// src/ds41/engram/engram_hasher.cpp - DS1-D: the Engram n-gram hasher and the row gather (host side, plain C++, no CUDA).
// Oracle: ref/ds41/engram.py NgramHasher.__call__ and constants_from_gguf_metadata; official: inference/engram.py NgramHashState.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "strata/ds41/cuda/engram.hpp"

namespace strata::ds41::cuda {

namespace {
[[noreturn]] void bad(const std::string& what) { throw std::invalid_argument("Engram constants: " + what); }
}  // namespace

NgramHasher::NgramHasher(EngramConstants c) : c_(std::move(c)) {
    const int64_t L = (int64_t) c_.layer_ids.size();
    if (c_.n_heads < 1) bad("n_heads < 1");
    if (c_.max_ngram_size < 2) bad("max_ngram_size < 2");
    if (c_.compressed_vocab_size < 1) bad("compressed_vocab_size < 1");
    const int64_t cols = (int64_t) (c_.max_ngram_size - 1) * c_.n_heads;
    if ((int64_t) c_.primes.size() != L * cols) bad("primes has " + std::to_string(c_.primes.size()) + " entries, expected " + std::to_string(L * cols));
    if ((int64_t) c_.offsets.size() != L * cols) bad("offsets has " + std::to_string(c_.offsets.size()) + " entries, expected " + std::to_string(L * cols));
    if ((int64_t) c_.multipliers.size() != L * c_.max_ngram_size)
        bad("multipliers has " + std::to_string(c_.multipliers.size()) + " entries, expected " + std::to_string(L * c_.max_ngram_size));
    if (c_.token_map.empty()) bad("token_map is empty");
    if (c_.pad_token_id < 0 || (size_t) c_.pad_token_id >= c_.token_map.size()) bad("pad_token_id outside the vocabulary");
    for (int32_t v : c_.token_map)
        if (v < 0 || v >= c_.compressed_vocab_size) bad("token_map holds an id outside [0, compressed_vocab_size)");
    if (!c_.num_embeddings.empty() && (int64_t) c_.num_embeddings.size() != L) bad("num_embeddings must have one entry per layer");
    const int64_t max_mult = std::numeric_limits<int64_t>::max() / c_.compressed_vocab_size;       // token * multiplier must not overflow int64
    for (int64_t m : c_.multipliers)
        if (m <= 0 || m > max_mult) bad("a multiplier is not in (0, INT64_MAX / compressed_vocab_size]");
    rows_.assign((size_t) L, 0);
    for (int64_t l = 0; l < L; ++l) {
        int64_t acc = 0;
        for (int64_t j = 0; j < cols; ++j) {                                      // offsets = running sums of the layer's own primes (engram.py EngramLayout.offsets)
            const int64_t p = c_.primes[(size_t) (l * cols + j)];
            if (p < 2) bad("a prime is < 2");
            if (c_.offsets[(size_t) (l * cols + j)] != acc) bad("offsets are not the running sums of the primes (layer index " + std::to_string(l) + ")");
            acc += p;
        }
        rows_[(size_t) l] = acc;
        if (!c_.num_embeddings.empty() && c_.num_embeddings[(size_t) l] != acc) bad("num_embeddings disagrees with the sum of the primes (layer index " + std::to_string(l) + ")");
    }
    pad_c_ = c_.token_map[(size_t) c_.pad_token_id];
    hist_.assign((size_t) c_.max_ngram_size - 1, pad_c_);
}

int NgramHasher::layer_index(int layer_id) const {
    for (size_t i = 0; i < c_.layer_ids.size(); ++i)
        if (c_.layer_ids[i] == layer_id) return (int) i;
    return -1;
}

void NgramHasher::reset() {
    std::fill(hist_.begin(), hist_.end(), pad_c_);
    pos_ = 0;
}

void NgramHasher::push(int32_t token_id, int64_t* out) {
    if (token_id < 0 || (size_t) token_id >= c_.token_map.size()) throw std::out_of_range("NgramHasher::push: token id outside the vocabulary");
    const int ng = c_.max_ngram_size, heads = c_.n_heads;
    const int64_t cols = (int64_t) (ng - 1) * heads;
    constexpr int kMaxNgram = 16;
    if (ng > kMaxNgram) throw std::invalid_argument("NgramHasher: max_ngram_size > 16");
    int64_t tok[kMaxNgram];
    tok[0] = c_.token_map[(size_t) token_id];
    for (int s = 1; s < ng; ++s) tok[s] = hist_[(size_t) s - 1];                    // position p - s (the pad's compressed id before the start)
    for (size_t l = 0; l < c_.layer_ids.size(); ++l) {
        const int64_t* mult = &c_.multipliers[l * (size_t) ng];
        const int64_t* prime = &c_.primes[l * (size_t) cols];
        const int64_t* off = &c_.offsets[l * (size_t) cols];
        int64_t* o = out + l * (size_t) cols;
        int64_t rolling = tok[0] * mult[0];
        for (int i = 1; i < ng; ++i) {                                              // the running value after step i hashes the (i + 1)-gram
            rolling ^= tok[i] * mult[i];
            for (int h = 0; h < heads; ++h) {
                const int64_t col = (int64_t) (i - 1) * heads + h;
                o[col] = rolling % prime[col] + off[col];
            }
        }
    }
    for (int s = ng - 2; s >= 1; --s) hist_[(size_t) s] = hist_[(size_t) s - 1];
    hist_[0] = (int32_t) tok[0];
    ++pos_;
}

void NgramHasher::push_many(const int32_t* ids, int n, int64_t* out) {
    const size_t per = (size_t) n_layers() * (size_t) rows_per_layer();
    for (int i = 0; i < n; ++i) push(ids[i], out + (size_t) i * per);
}

void engram_gather_rows(const EngramTableView& table, const int64_t* idx, int64_t n, size_t row_bytes, uint8_t* out) {
    if (table.base == nullptr && n > 0) throw std::invalid_argument("engram_gather_rows: null table");
    for (int64_t i = 0; i < n; ++i) {
        const int64_t r = idx[i];
        if (r < 0 || r >= table.rows) throw std::out_of_range("engram_gather_rows: row " + std::to_string(r) + " outside [0, " + std::to_string(table.rows) + ")");
        if (i + 1 < n && idx[i + 1] >= 0 && idx[i + 1] < table.rows) __builtin_prefetch(table.base + (size_t) idx[i + 1] * row_bytes);
        std::memcpy(out + (size_t) i * row_bytes, table.base + (size_t) r * row_bytes, row_bytes);
    }
}

}  // namespace strata::ds41::cuda
