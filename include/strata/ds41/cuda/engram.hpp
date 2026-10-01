// include/strata/ds41/cuda/engram.hpp - DS1-D: Engram (hashed n-gram embeddings added into the mHC stream) for the V100: the host n-gram hasher, the row gather
// from the mmapped MXFP4 tables, the device dequantisation and the combine kernel, and the per-layer runner that strings them together.
//
// ORACLE: ref/ds41/engram.py (`NgramHasher`, `constants_from_gguf_metadata`, `engram_layer`), model.py `Model.forward` (where Engram sits: BEFORE
// block l on the Engram layers, rewriting the 4-copy stream; the mHC lag `pre` is not touched by it), official engram.py / model.py `Engram`.
//
// ONE ENGRAM LAYER, STEP BY STEP (T tokens; DS-1 is synchronous, DS-2 will prefetch the rows: their addresses depend on the token ids only):
//   host    NgramHasher::push(token, out)      token id -> compressed id -> rolling XOR of id * multiplier over the last max_ngram_size tokens -> % prime +
//                                              offset: kEngramRows = (ngram - 1) * heads = 24 table row indices per Engram layer per token
//   host    engram_gather_rows                 the 24 rows of each token (136 B each: kEngramHeadDim / 32 MXFP4 blocks of 17 B) from the mmapped table into a
//                                              pinned staging buffer; then one h2d copy
//   device  ds41_engram_dequant_rows           MXFP4 -> FP32 rows [T][kEngramIn] (bit-exact: kvalues * 2^(e - 128))
//   device  wkv (NOT here: Q8_0 GEMV, int8 activations: a `linear_act` site)    kv [T][kEngramOut] = [key (kHc * kHidden) | value (kHidden)]
//   device  ds41_engram_combine                gate = sigmoid(signed sqrt of the normalised key . stream dot) per (token, copy); stream[c] += gate * value
// The wkv hook is a std::function: engram_wkv_q8<G> builds it from DS1-G's ds41_quantize_acts (natural order) and a Q8_0 GEMV callback (DS1-B's / DS1-C's gemv_q8).  Everything FP32 except the Q8_0 weights of that GEMV and the BF16 q / k weights.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/ds41/cuda/ds41_cuda.hpp"       // ds41_quantize_acts, ActOrder (DS1-G)
#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"

namespace strata::ds41::cuda {

inline constexpr float kEngramClampValue = 1e-6f;       // model.py Engram.clamp_value: gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))

// ---------------------------------------------------------------------------------------------------------------------------------
// host: the n-gram hasher
// ---------------------------------------------------------------------------------------------------------------------------------
/// The Engram constants of ONE GGUF, as the loader reads them from the `deepseek41.engram.*` metadata (ref/ds41/engram.py constants_from_gguf_metadata):
/// authoritative for that file (the primes are the next primes above engram_vocab_size - 1, never reused; offsets are the running sums of a layer's primes;
/// multipliers are odd int64 from default_rng(10007 * layer_id); token_map folds the 129,280 tokens into 99,092 compressed ids).
struct EngramConstants {
    std::vector<int32_t> layer_ids;      // engram.layer_ids           [L]            e.g. {1, 14}
    int32_t n_heads = 0;                 // engram.n_heads             (8)
    int32_t max_ngram_size = 0;          // engram.max_ngram_size      (4)
    int32_t pad_token_id = 0;            // engram.pad_token_id        (2): a TOKEN id; its compressed id stands in for positions before the sequence
    int32_t compressed_vocab_size = 0;   // engram.compressed_vocab_size (99,092)
    std::vector<int64_t> primes;         // engram.primes              [L][max_ngram_size - 1][n_heads], row-major
    std::vector<int64_t> offsets;        // engram.offsets             [L][(max_ngram_size - 1) * n_heads]
    std::vector<int64_t> multipliers;    // engram.multipliers         [L][max_ngram_size]
    std::vector<int32_t> token_map;      // engram.token_map           [vocab] -> compressed id in [0, compressed_vocab_size)
    std::vector<int64_t> num_embeddings; // engram.num_embeddings      [L] table rows (optional: checked against the primes when present)
};

/// The constants from DS1-A's `strata::ds41::model::EngramConfig` (include/strata/ds41/model/config.hpp), or any struct with the same members (layers, heads,
/// ngram, pad_id, cvocab, num_embeddings, primes, offsets, multipliers, token_map).  A template, so that this header does not include the loader's.
template <class C>
EngramConstants engram_constants_from(const C& e) {
    EngramConstants c;
    c.layer_ids.assign(e.layers.begin(), e.layers.end());
    c.n_heads = (int32_t) e.heads;
    c.max_ngram_size = (int32_t) e.ngram;
    c.pad_token_id = (int32_t) e.pad_id;
    c.compressed_vocab_size = (int32_t) e.cvocab;
    c.primes.assign(e.primes.begin(), e.primes.end());
    c.offsets.assign(e.offsets.begin(), e.offsets.end());
    c.multipliers.assign(e.multipliers.begin(), e.multipliers.end());
    c.token_map.assign(e.token_map.begin(), e.token_map.end());
    c.num_embeddings.assign(e.num_embeddings.begin(), e.num_embeddings.end());
    return c;
}

/// engram.py NgramHashState, text only, one sequence.  Position p hashes the compressed ids of tokens p, p-1, p-2, p-3 (the pad token's compressed id where the
/// sequence has not started).  Incremental: keeps the last max_ngram_size - 1 compressed ids, so a prompt fed one token at a time, in chunks or all at once
/// gives the same rows as the oracle's prefill + decode.  Row index of (layer li, order i = 2..max_ngram, head h) = rolling_xor_i % primes[li][i-2][h] + offsets,
/// stored at column (i - 2) * n_heads + h: exactly the oracle's `[s, n_engram_layers, n_hash_cols]` int64 array, which is also the order the rows are
/// concatenated in for wkv.
class NgramHasher {
   public:
    /// Validates shapes and ranges (every multiplier odd-or-positive and small enough that token * multiplier fits int64, primes > 0, offsets = running sums of
    /// the primes, token_map values in range, num_embeddings = sum of a layer's primes if given); throws std::invalid_argument.
    explicit NgramHasher(EngramConstants c);

    int n_layers() const { return (int) c_.layer_ids.size(); }               // Engram layers
    int rows_per_layer() const { return (c_.max_ngram_size - 1) * c_.n_heads; }   // kEngramRows (24)
    int64_t position() const { return pos_; }                                 // tokens pushed since reset()
    /// Table rows of Engram layer index li (sum of its primes).
    int64_t table_rows(int li) const { return rows_.at((size_t) li); }
    /// Index of `layer_id` in layer_ids, or -1.
    int layer_index(int layer_id) const;
    const EngramConstants& constants() const { return c_; }

    /// Start a new sequence (position 0, empty look-back).
    void reset();
    /// Append one token (id in [0, vocab); throws std::out_of_range otherwise) and write its rows: out[n_layers()][rows_per_layer()] int64.
    void push(int32_t token_id, int64_t* out);
    /// Append n tokens: out[n][n_layers()][rows_per_layer()].
    void push_many(const int32_t* ids, int n, int64_t* out);

   private:
    EngramConstants c_;
    std::vector<int64_t> rows_;
    std::vector<int32_t> hist_;      // compressed ids of the last max_ngram_size - 1 tokens, most recent first (pad where missing)
    int32_t pad_c_ = 0;              // token_map[pad_token_id]
    int64_t pos_ = 0;
};

// ---------------------------------------------------------------------------------------------------------------------------------
// host: the tables
// ---------------------------------------------------------------------------------------------------------------------------------
/// One Engram layer's table as the GGUF stores it: `rows` rows of Derived<G>::kEngramRowBytes bytes (kEngramHeadDim / 32 GGML MXFP4 blocks of 17 B: e, qs[16]),
/// contiguous, in host memory (the mmap of the shard).  DS1-A fills it; nothing here owns the memory.
struct EngramTableView {
    const uint8_t* base = nullptr;
    int64_t rows = 0;
};

/// Copy rows idx[0..n) of `table` (row_bytes each) to out[n][row_bytes]; throws std::out_of_range for an index outside [0, rows).  On an mmapped table the
/// first touch of a page is the SSD read (DS-1: synchronous); the next row is prefetched while the current one is copied.
void engram_gather_rows(const EngramTableView& table, const int64_t* idx, int64_t n, size_t row_bytes, uint8_t* out);

// ---------------------------------------------------------------------------------------------------------------------------------
// device ops (templates over G; RealGeom is instantiated in engram.cu, MiniGeom in the emulator build)
// ---------------------------------------------------------------------------------------------------------------------------------
/// MXFP4 rows -> FP32: rows = n_rows * kEngramRowBytes device bytes (no alignment needed), out = [n_rows][kEngramHeadDim] floats, 16-byte aligned.  Bit-exact
/// against ref/ds41/quant.py dequant_mxfp4 (and ggml's dequantize_row_mxfp4).  Rows of one token are consecutive: out viewed [T][kEngramIn] is wkv's input.
template <class G>
void ds41_engram_dequant_rows(Dev& dev, const uint8_t* rows, int n_rows, float* out, Stream stream = nullptr);

/// The combine (engram.py engram_layer after wkv), in place on the stream x [T][kHc][kHidden] (FP32, 16-byte aligned): kv [T][kEngramOut] = wkv's output
/// (key [kHc][kHidden] then value [kHidden] per token); q_bf16 / k_bf16 = engram_{q,k}.weight, [kHc][kHidden] BF16 bits (8-byte aligned); norm_eps = 1e-20
/// (the model's rms_norm_eps).  weight = q * k (FP32 product); per (token, copy): rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps),
/// dot = sum_d h * weight * key * rstd * kHidden^-0.5, gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot)), h += gate * value.
template <class G>
void ds41_engram_combine(Dev& dev, float* x, const float* kv, const uint16_t* q_bf16, const uint16_t* k_bf16, int T, float norm_eps, Stream stream = nullptr);

/// Registers / static shared memory of the two kernels (which: 0 = dequant, 1 = combine); zeros in the emulator build.
struct EngramKernelInfo {
    int regs = 0;
    int static_smem = 0;
};
template <class G> EngramKernelInfo ds41_engram_kernel_info(int which);

// ---------------------------------------------------------------------------------------------------------------------------------
// the runner: gather -> upload -> dequantise -> wkv -> combine, for one Engram layer at a time
// ---------------------------------------------------------------------------------------------------------------------------------
/// The wkv linear (Q8_0 [kEngramOut][kEngramIn], int8 activations: a `linear_act` site, DS1.md section 2): rows = device FP32 [T][kEngramIn] -> kv = device FP32
/// [T][kEngramOut].  The session binds it to DS1-B's GEMV after DS1-G's activation quantiser; enqueue on `stream`.
using EngramWkvFn = std::function<void(const float* rows, int T, float* kv, Stream stream)>;

/// The Q8_0 GEMV a session binds: y[t][r] = sum over the k / 32 blocks of d_w[r][b] * xs[t][b] * (sum_j wq[r][b][j] * xq[t][b][j]) (the integer sum exact; DS1.md section 2), with
/// w = GGML Q8_0 rows (fp16 d + int8 qs[32], 34 bytes, row r at w + r * (k / 32) * 34), xq / xs in the NATURAL layout of ds41_quantize_acts, y fp32 [T][n].  DS1-C's
/// AttnDenseOps::gemv_q8 has this shape, and DS1-B's dense.hpp will.
using EngramGemvQ8Fn = std::function<void(const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream stream)>;

/// The wkv hook of a layer: DS1-G's activation quantiser (`ds41_quantize_acts<G>`, ActOrder::kNatural: int8 per 32 + one FP32 scale, CONTRACTS.md's rule) on the dequantised rows,
/// then `gemv` with the layer's Q8_0 weight `wkv_q8` [kEngramOut][kEngramIn] (device).  `xq` / `xs` = scratch int8 [t_max][kEngramIn] and float [t_max][kEngramIn / 32]
/// (EngramRunner::xq() / xs()).  `dev` is held by reference: it must outlive the returned function.
template <class G>
EngramWkvFn engram_wkv_q8(Dev& dev, const void* wkv_q8, int8_t* xq, float* xs, EngramGemvQ8Fn gemv) {
    return [&dev, wkv_q8, xq, xs, gemv = std::move(gemv)](const float* rows, int T, float* kv, Stream s) {
        ds41_quantize_acts<G>(dev, rows, T, Derived<G>::kEngramIn, xq, xs, s, ActOrder::kNatural);
        gemv(wkv_q8, Derived<G>::kEngramOut, Derived<G>::kEngramIn, xq, xs, T, kv, s);
    };
}

/// One Engram layer's weights: the host table and the two BF16 device vectors (the Q8_0 wkv belongs to the hook).
struct EngramLayerWeights {
    EngramTableView table;
    const uint16_t* q_bf16 = nullptr;    // device [kHc][kHidden] BF16 bits
    const uint16_t* k_bf16 = nullptr;    // device [kHc][kHidden] BF16 bits
};

/// Owns the staging memory: a pinned host buffer for the gathered rows and device buffers (raw rows, FP32 rows, kv, int8 scratch for the wkv hook), sized for t_max tokens.  Not
/// thread-safe; one per session.  DS-1 is synchronous: the h2d copy is blocking (Dev::h2d), so the staging buffer is free again when run() returns its upload.
template <class G>
class EngramRunner {
   public:
    EngramRunner(Dev& dev, int t_max) : dev_(dev), t_max_(t_max) {
        if (t_max < 1) throw std::invalid_argument("EngramRunner: t_max < 1");
        staging_ = static_cast<uint8_t*>(dev_.alloc_mapped(raw_bytes(t_max)));
        raw_ = static_cast<uint8_t*>(dev_.alloc(raw_bytes(t_max)));
        rows_ = static_cast<float*>(dev_.alloc((size_t) t_max * Derived<G>::kEngramIn * sizeof(float)));
        kv_ = static_cast<float*>(dev_.alloc((size_t) t_max * Derived<G>::kEngramOut * sizeof(float)));
        xq_ = static_cast<int8_t*>(dev_.alloc((size_t) t_max * Derived<G>::kEngramIn));
        xs_ = static_cast<float*>(dev_.alloc((size_t) t_max * (Derived<G>::kEngramIn / 32) * sizeof(float)));
    }
    ~EngramRunner() {
        dev_.release(xs_);
        dev_.release(xq_);
        dev_.release(kv_);
        dev_.release(rows_);
        dev_.release(raw_);
        dev_.release_mapped(staging_);
    }
    EngramRunner(const EngramRunner&) = delete;
    EngramRunner& operator=(const EngramRunner&) = delete;

    int t_max() const { return t_max_; }
    float* rows_f32() const { return rows_; }        // device [t_max][kEngramIn]: the dequantised rows of the last run (tests, tracing)
    float* kv() const { return kv_; }                // device [t_max][kEngramOut]: wkv's output of the last run
    int8_t* xq() const { return xq_; }               // device [t_max][kEngramIn] / [t_max][kEngramIn / 32]: scratch for the wkv hook's activation quantiser
    float* xs() const { return xs_; }                // (engram_wkv_q8)

    /// Gather the kEngramRows rows of each of T tokens (row_idx[t * token_stride + r], from NgramHasher: token_stride = n_layers * rows_per_layer when the
    /// pointer is `out + layer_index * rows_per_layer`) into the staging buffer and copy them to the device.  Blocking.
    void upload_rows(const EngramTableView& table, const int64_t* row_idx, int T, int64_t token_stride) {
        check_t(T);
        constexpr size_t rb = Derived<G>::kEngramRowBytes;
        for (int t = 0; t < T; ++t) engram_gather_rows(table, row_idx + (int64_t) t * token_stride, G::kEngramRows, rb, staging_ + (size_t) t * G::kEngramRows * rb);
        dev_.h2d(raw_, staging_, raw_bytes(T));
    }

    /// The whole layer: upload, dequantise, wkv (hook), combine in place on the stream x [T][kHc][kHidden].
    void run(const EngramLayerWeights& w, const int64_t* row_idx, int T, int64_t token_stride, float* x, const EngramWkvFn& wkv, float norm_eps,
             Stream stream = nullptr) {
        upload_rows(w.table, row_idx, T, token_stride);
        const Stream s = stream_or_default(dev_, stream);
        ds41_engram_dequant_rows<G>(dev_, raw_, T * G::kEngramRows, rows_, s);
        wkv(rows_, T, kv_, s);
        ds41_engram_combine<G>(dev_, x, kv_, w.q_bf16, w.k_bf16, T, norm_eps, s);
    }

   private:
    static size_t raw_bytes(int T) { return (size_t) T * G::kEngramRows * Derived<G>::kEngramRowBytes; }
    void check_t(int T) const {
        if (T < 1 || T > t_max_) throw std::invalid_argument("EngramRunner: T outside 1 .. t_max");
    }
    Dev& dev_;
    int t_max_;
    uint8_t* staging_ = nullptr;
    uint8_t* raw_ = nullptr;
    float* rows_ = nullptr;
    float* kv_ = nullptr;
    int8_t* xq_ = nullptr;
    float* xs_ = nullptr;
};

}  // namespace strata::ds41::cuda
