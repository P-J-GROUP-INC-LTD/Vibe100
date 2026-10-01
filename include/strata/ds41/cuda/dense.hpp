// include/strata/ds41/cuda/dense.hpp - DS1-B: the dense GPU kernels of DeepSeek-V4.1-Flash for the V100 (sm_70): Q8_0 / BF16 / F32 GEMV, RMSNorm, RoPE, the
// shared expert, argmax / top-k over the vocabulary, and the small host helpers around them (embedding rows, the RoPE table).
//
// Oracle: ref/ds41 (ops.py rmsnorm / linear, rope.py, moe.py expert, quant.py dequant_q8_0 / act_quant_int8, attention.py ~302-350 for which linear calls
// quantise their input).  Contract: docs/deepseek/DS1.md sections 1, 2, 5; docs/deepseek/CONTRACTS.md (the activation quantiser, the SwiGLU clamps).
//
// EVERY OP IS `template <class G> void ds41_<op>(Dev&, args..., Stream = nullptr)` (DS1.md section 5): G (RealGeom / MiniGeom, geom.hpp) supplies the shapes
// of the geometry-specific ops (vocabulary, shared expert, wo_a, RoPE width) and is only a compile-time check for the shape-generic ones (the GEMVs take
// n / k at run time: every shape of RESEARCH.md section 10 is a multiple of 32).  The nvcc build (src/ds41/cuda/dense.cu, library strata_ds41_dense)
// instantiates RealGeom; the CPU emulation (src/ds41/cuda/dense_emu_impl.cpp, library strata_ds41_dense_emu, target ds41_dense_emu_test) instantiates RealGeom
// and MiniGeom from the same source (dense_impl.cuh) - link exactly one of the two libraries into a program.  Calling a G the build did not instantiate is
// a LINK error.  Pointers are device pointers from `Dev` (HostDev's malloc in the emulation) except where a parameter says "host".  Every op is
// stream-ordered on `stream` (nullptr = dev.stream()) and allocation-free: scratch comes from the caller (the *_scratch_bytes / *_carve helpers).
// Errors (bad shape, a misaligned pointer, too little scratch) throw std::invalid_argument; a failed launch std::runtime_error.
//
// ---------------------------------------------------------------------------------------------------------------------------------------------------
// THE NUMERICS (what is promised, and what the tests pin)
// ---------------------------------------------------------------------------------------------------------------------------------------------------
//  * ONE REDUCTION ORDER PER OUTPUT, a function of (k, op) only - never of T, of n, of the launch grid, of the SM count, of the block / fibre scheduling:
//    a token's row of Y is bit-identical whether it is computed alone (T = 1), in a window (T <= 8) or in a larger T (tested, in three scheduling orders).
//    There are no atomics.  The order is documented per op below.
//  * Q8_0 weights are read VERBATIM as the GGUF stores them: blocks of 34 bytes = fp16 d + 32 int8 (no padding between blocks or rows; row r starts at
//    r * (k / 32) * 34).  fp16 -> FP32 is exact (zero, subnormals, Inf, NaN).
//  * INT8-ACTIVATION GEMV (the `linear(act_quant=True)` sites: wq_a, wq_b, wkv, wo_b, the shared expert, indexer.wq_b, engram.wkv): the input is quantised
//    by CONTRACTS.md's rule (DS-D's ds41_quantize_acts<G>(..., ActOrder::kNatural)); per 32-block b the integer
//    dot sum_j wq*xq is EXACT (dp4a) and the FP32 accumulation is   acc = fma(d_w * d_x, float(isum), acc)   (the rule of DS-D's expert kernels).  Order:
//    a row has S = ceil(k / 256) "super-blocks" of 8 blocks; P = min(32, next power of two >= S) lanes share a row, lane l takes super-blocks l, l + P, ...
//    and adds their blocks in increasing order into one accumulator, then the P lanes are summed by the xor butterfly (offsets P/2 .. 1).
//    A NaN / Inf block of x (d = NaN) or of w reaches every output of the row.
//  * FP32-ACTIVATION Q8_0 GEMV (wo_a, and every site in the oracle's `exact` mode): value = float(q) * d_w is exact in FP32; acc = fma(x, value, acc) in
//    increasing k within a lane, same lane layout as above (P lanes per row, super-blocks l, l + P, ...), then the butterfly.
//  * BF16 / F32 GEMV (the head, the router-like projections): one warp per row, lane l takes the 16-byte chunks l, l + 32, ... (8 BF16 or 4 F32 each),
//    acc = fma(x, w, acc) in element order, then the 32-lane butterfly (16, 8, 4, 2, 1).
//  * RMSNorm: out = w * (x / sqrt(mean(x^2) + eps)), the oracle's formula, FP32 (the sum of squares in the lane / butterfly order, divisions and sqrt
//    IEEE; eps = 1e-20 is the real model's value).  RoPE: separately rounded products (no FMA), exactly numpy's `a * c - b * s`, `a * s + b * c`.
//  * NaN: propagates through the GEMVs, the norm, RoPE, and through the SwiGLU clamps `g > 10 ? 10 : g`, `u > 10 ? 10 : (u < -10 ? -10 : u)` (NOT fminf /
//    fmaxf).  argmax / top-k rank a NaN ABOVE every number (numpy / torch semantics: a failure upstream shows as the top-1), ties go to the LOWER index.
//  * Weights are read ONCE from HBM for all T <= 8 (T tokens share one pass over the rows).  When T x k would not fit the 96 KB of shared memory the
//    token tiles run as separate blocks over the same rows (adjacent in the grid, so the second tile hits L2): the numbers do not change.
//
// A DECODE STEP IN CALLS (T = 1; every call on the same stream; G = RealGeom):
//     ds41_rmsnorm<G>(dev, h, attn_norm_w, 1, kHidden, xn, 1e-20f, s);                                    // x = attn_norm(hc_pre(stream))
//     ds41_quantize_acts<G>(dev, xn, 1, kHidden, xq, xs, s, ActOrder::kNatural);                          // ONE quantisation, read by both GEMVs below
//     ds41_gemv_q8_int8<G>(dev, wq_a, kQLora, kHidden, xq, xs, 1, qa, s);   ds41_gemv_q8_int8<G>(dev, wkv, kHeadDim, kHidden, xq, xs, 1, kv, s);
//     ds41_rmsnorm<G>(dev, qa, q_norm_w, 1, kQLora, qa, eps, s);   ds41_quantize_acts<G>(dev, qa, 1, kQLora, qxq, qxs, s, ActOrder::kNatural);
//     ds41_gemv_q8_int8<G>(dev, wq_b, kHeads * kHeadDim, kQLora, qxq, qxs, 1, q, s);   ds41_rope<G>(dev, q, q, 1, kHeads, kHeadDim, table, pos, false, s);
//     ... attention (DS1-C) ... ds41_rope<G>(.., true ..);  ds41_wo_a<G>(dev, wo_a, o, 1, mid, s);  quantise mid; ds41_gemv_q8_int8<G>(dev, wo_b, kHidden, kOMid, ...);
//     FFN: ds41_rmsnorm; router (DS-D); ds41_shared_expert<G>(dev, w, xn, 1, y_shared, scratch, true, 10.f, s); ... ds41_f32_add<G>(...);
//     head: ds41_rmsnorm<G>(final hidden); ds41_head<G>(dev, output_bf16, xn, 1, logits, s); ds41_argmax<G>(dev, logits, 1, tok, val, s);
//
// NO OVERLAP: y (and any scratch) must not overlap an input of the same call (except where an op says `out may equal x`).
//
// ALIGNMENT (the emulator asserts it): FP32 activations and every FP32 / BF16 weight array 16 bytes; Q8_0 weights 2 bytes (16 when k % 256 == 0: the fast
// path; any other alignment or k takes the generic path, bit-identical, slower); xq 16 bytes (k % 32 == 0); xs, y 4 bytes.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "strata/ds41/cuda/ds41_cuda.hpp"   // ActOrder, ds41_quantize_acts<G>, Dev
#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"

namespace strata::ds41::cuda {

inline constexpr int kQ8Block = 32;          // values per Q8_0 block
inline constexpr int kQ8BlockBytes = 34;     // fp16 d + 32 int8
/// Bytes of one Q8_0 row of k values (k a multiple of 32).
constexpr size_t q8_row_bytes(int k) { return (size_t) (k / kQ8Block) * kQ8BlockBytes; }
inline constexpr int kDenseMaxT = 8;         // the tested window; any T >= 1 works (more token tiles)

// ---- the activation quantiser: DS-D / DS1-G's -------------------------------------------------------------------------------------------------
// The int8 GEMVs take x ALREADY QUANTISED in the natural per-32 layout: ds41_quantize_acts<G>(dev, x, T, width, xq, xs, stream, ActOrder::kNatural) of
// ds41_cuda.hpp (xq int8 [T][width/32][32], byte j of block b = element 32 b + j; xs fp32 [T][width/32]; CONTRACTS.md's rule bit for bit).  One call per input,
// shared by every GEMV that reads the same input.  This package has no quantiser of its own: the shared expert and its tests call that one (link
// strata_ds41_cuda, or compile src/ds41/cuda/ds41_emu_impl.cpp into an emulated program).

// ---- Q8_0 GEMV --------------------------------------------------------------------------------------------------------------------------------
/// Y[T][n] = X[T][k] . W^T, W = n rows of k values in GGUF Q8_0 blocks, X given as the natural-order quantised activations (xq / xs of ds41_quantize_acts).
/// k, n >= 1, k a multiple of 32.  y[t * n + row].
template <class G>
void ds41_gemv_q8_int8(Dev& dev, const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream stream = nullptr);

/// Two matrices of the same shape that read the SAME activations (the shared expert's gate and up): one launch, y_a = X . Wa^T, y_b = X . Wb^T.  Bit-identical
/// to two separate ds41_gemv_q8_int8 calls.
template <class G>
void ds41_gemv_q8_int8_pair(Dev& dev, const void* w_a, const void* w_b, int n, int k, const int8_t* xq, const float* xs, int T, float* y_a, float* y_b,
                            Stream stream = nullptr);

/// The same product with FP32 activations (no quantisation): y = X . dequant(W)^T, FP32 accumulation.  x fp32 [T][k].
template <class G>
void ds41_gemv_q8_f32(Dev& dev, const void* w, int n, int k, const float* x, int T, float* y, Stream stream = nullptr);

/// The grouped (block-diagonal) GEMV: W = groups * rows_per_group rows of k_per_group values (Q8_0), row g * rows_per_group + r multiplies the slice
/// x[t][g * k_per_group .. (g + 1) * k_per_group) of the token's row (x fp32 [T][groups * k_per_group]); y[t][g * rows_per_group + r].  FP32 activations
/// (the reference keeps wo_a bf16).  GGUF layout check: attn_output_a is [ne0 = 4096 (kOGroupIn), ne1 = 8192 (kOGroups * kOLora)], i.e. 8192 consecutive Q8_0
/// rows of 4096 values = the oracle's `wo_a.reshape(g, o_lora_rank, -1)`: rows 0..1023 are group 0, and so on - this call reads it as stored.
template <class G>
void ds41_gemv_q8_grouped_f32(Dev& dev, const void* w, int groups, int rows_per_group, int k_per_group, const float* x, int T, float* y,
                              Stream stream = nullptr);
/// wo_a of G: groups = kOGroups, rows_per_group = kOLora, k_per_group = kOGroupIn (x = the attention output [T][kHeads * kHeadDim], y [T][kOGroups * kOLora]).
template <class G>
void ds41_wo_a(Dev& dev, const void* w, const float* x, int T, float* y, Stream stream = nullptr);

// ---- BF16 / F32 GEMV --------------------------------------------------------------------------------------------------------------------------
/// y[t][r] = sum_d x[t][d] * bf16(w[r][d]); w = raw BF16 bits [n][k] (16-byte aligned, k % 8 == 0), FP32 activations and accumulation.
template <class G>
void ds41_gemv_bf16(Dev& dev, const uint16_t* w, int n, int k, const float* x, int T, float* y, Stream stream = nullptr);
/// The same with FP32 weights (k % 4 == 0).
template <class G>
void ds41_gemv_f32(Dev& dev, const float* w, int n, int k, const float* x, int T, float* y, Stream stream = nullptr);
/// The head: logits[t][v] for the kVocab rows of the BF16 `output` matrix [kVocab][kHidden] (RealGeom: 129,280 x 5120, 1.32 GB: one coalesced pass).
template <class G>
void ds41_head(Dev& dev, const uint16_t* w, const float* x, int T, float* logits, Stream stream = nullptr);

// ---- RMSNorm ----------------------------------------------------------------------------------------------------------------------------------
/// out[r] = w * (x[r] / sqrt(mean(x[r]^2) + eps)) for `rows` rows of `width` (any multiple of 4: kHidden, kQLora, kHeadDim, kIdxDim ...; rows = T for a
/// hidden / latent norm, T * heads for a per-head one).  `w` fp32 [width] or nullptr (weightless).  out may equal x.  Mirrors ops.py rmsnorm.
template <class G>
void ds41_rmsnorm(Dev& dev, const float* x, const float* w, int rows, int width, float* out, float eps = 1e-20f, Stream stream = nullptr);

// ---- RoPE -------------------------------------------------------------------------------------------------------------------------------------
/// The device table of one layer: cos / sin fp32 [rows][kRopeDim / 2] (the same two arrays as DS1-C's AttnRope), built on the host by ds41_rope_table_host.
struct RopeTable {
    const float* cos = nullptr;
    const float* sin = nullptr;
    int rows = 0;   // positions in the table (a call checks pos0 + T <= rows)
};
/// Rotates the LAST kRopeDim channels (adjacent pairs, (a + ib)(cos + i sin); `inverse` conjugates) of every row of x [T * heads][width]; row r is token
/// r / heads at position pos0 + r / heads.  The first width - kRopeDim channels are copied when out != x.  width = kHeadDim (q, kv, o) or kIdxDim (the
/// indexer); out may equal x.  Mirrors rope.py apply_rope_tail.
template <class G>
void ds41_rope(Dev& dev, const float* x, float* out, int T, int heads, int width, const RopeTable& table, int pos0, bool inverse, Stream stream = nullptr);

/// HOST: the table of rope.py `rope_table` computed in double and rounded to FP32: cos / sin [seqlen][dim / 2] (dim = kRopeDim).  YaRN is applied when
/// original_seq_len > 0 (rope.py layer_rope_params: layers 0-1: original_seq_len 0, base 10000; layers >= 2: original_seq_len 65536, base 160000;
/// factor 16, beta_fast 32, beta_slow 1 for the whole model).
struct RopeParams {
    int original_seq_len = 0;
    double base = 10000.0;
    double factor = 16.0;
    double beta_fast = 32.0;
    double beta_slow = 1.0;
};
/// rope.py layer_rope_params: a layer with a compress ratio uses theta = compress_rope_theta with YaRN (original_seq_len, factor, beta_fast, beta_slow of the
/// config), a layer without (L0, L1) theta = rope_theta and no YaRN.
inline RopeParams ds41_layer_rope_params(int compress_ratio, double rope_theta, double compress_rope_theta, int original_seq_len, double factor = 16.0,
                                         double beta_fast = 32.0, double beta_slow = 1.0) {
    RopeParams p;
    p.original_seq_len = compress_ratio != 0 ? original_seq_len : 0;
    p.base = compress_ratio != 0 ? compress_rope_theta : rope_theta;
    p.factor = factor;
    p.beta_fast = beta_fast;
    p.beta_slow = beta_slow;
    return p;
}
inline void ds41_rope_table_host(int dim, int seqlen, const RopeParams& p, std::vector<float>& cos_out, std::vector<float>& sin_out) {
    if (dim < 2 || dim % 2 != 0 || seqlen < 0) throw std::invalid_argument("ds41_rope_table_host: bad dim / seqlen");
    const int half = dim / 2;
    std::vector<double> freqs((size_t) half);
    for (int i = 0; i < half; ++i) freqs[(size_t) i] = 1.0 / std::pow(p.base, (double) (2 * i) / (double) dim);
    if (p.original_seq_len > 0) {
        auto corrected_dim = [&](double rotations) {
            return dim * std::log((double) p.original_seq_len / (rotations * 2.0 * 3.14159265358979323846)) / (2.0 * std::log(p.base));
        };
        const double low = std::max(std::floor(corrected_dim(p.beta_fast)), 0.0);
        const double high = std::min(std::ceil(corrected_dim(p.beta_slow)), (double) (dim - 1));
        const double den = std::max(high - low, 1e-3);
        for (int i = 0; i < half; ++i) {
            double ramp = ((double) i - low) / den;
            ramp = ramp < 0.0 ? 0.0 : (ramp > 1.0 ? 1.0 : ramp);
            const double smooth = 1.0 - ramp;
            freqs[(size_t) i] = freqs[(size_t) i] / p.factor * (1.0 - smooth) + freqs[(size_t) i] * smooth;
        }
    }
    cos_out.assign((size_t) seqlen * half, 0.0f);
    sin_out.assign((size_t) seqlen * half, 0.0f);
    for (int pos = 0; pos < seqlen; ++pos)
        for (int i = 0; i < half; ++i) {
            const double ang = (double) pos * freqs[(size_t) i];
            cos_out[(size_t) pos * half + i] = (float) std::cos(ang);
            sin_out[(size_t) pos * half + i] = (float) std::sin(ang);
        }
}

// ---- the shared expert ------------------------------------------------------------------------------------------------------------------------
/// Q8_0 weights of one layer's shared expert (GGUF ffn_{gate,up,down}_shexp): w1 = gate [kFF][kHidden], w3 = up [kFF][kHidden], w2 = down [kHidden][kFF].
struct SharedExpertWeights {
    const void* w1 = nullptr;
    const void* w3 = nullptr;
    const void* w2 = nullptr;
};
/// Device scratch of one call (carve with shared_expert_scratch_carve): xq / xs the quantised x, g / u the two projections [T][kFF] (h overwrites g in the
/// FP32 mode), hq / hs the quantised h.
struct SharedExpertScratch {
    int8_t* xq = nullptr;   // [T][kHidden]
    float* xs = nullptr;    // [T][kHidden / 32]
    float* g = nullptr;     // [T][kFF]
    float* u = nullptr;     // [T][kFF]
    int8_t* hq = nullptr;   // [T][kFF]
    float* hs = nullptr;    // [T][kFF / 32]
};
namespace detail {
constexpr size_t dense_up256(size_t v) { return (v + 255) & ~(size_t) 255; }
}
template <class G>
constexpr size_t shared_expert_scratch_bytes(int T) {
    return detail::dense_up256((size_t) T * G::kHidden) + detail::dense_up256((size_t) T * (G::kHidden / 32) * 4) + 2 * detail::dense_up256((size_t) T * G::kFF * 4) +
           detail::dense_up256((size_t) T * G::kFF) + detail::dense_up256((size_t) T * (G::kFF / 32) * 4);
}
template <class G>
inline SharedExpertScratch shared_expert_scratch_carve(void* base, int T) {
    auto* p = static_cast<unsigned char*>(base);
    SharedExpertScratch s;
    s.xq = reinterpret_cast<int8_t*>(p), p += detail::dense_up256((size_t) T * G::kHidden);
    s.xs = reinterpret_cast<float*>(p), p += detail::dense_up256((size_t) T * (G::kHidden / 32) * 4);
    s.g = reinterpret_cast<float*>(p), p += detail::dense_up256((size_t) T * G::kFF * 4);
    s.u = reinterpret_cast<float*>(p), p += detail::dense_up256((size_t) T * G::kFF * 4);
    s.hq = reinterpret_cast<int8_t*>(p), p += detail::dense_up256((size_t) T * G::kFF);
    s.hs = reinterpret_cast<float*>(p);
    return s;
}

/// y[T][kHidden] = W2 . ( silu(min(W1.x, L)) * clamp(W3.x, -L, L) ), L = swiglu_limit (10; <= 0: no clamp), mirroring ref/ds41/moe.py expert with weights = None
/// (the routing weight is 1: the shared expert).  int8_act = true (the engine's mode, QuantConfig.int8_act): x and h are quantised (CONTRACTS.md), the three
/// GEMVs are Q8_0 x int8 dp4a; false (QuantConfig.exact): FP32 activations throughout.  x fp32 [T][kHidden].  int8 mode: four launches (quantise x, gate + up in one
/// launch, SwiGLU + quantise h, down).  y is OVERWRITTEN (the caller adds the routed experts' sum).
template <class G>
void ds41_shared_expert(Dev& dev, const SharedExpertWeights& w, const float* x, int T, float* y, const SharedExpertScratch& scratch, bool int8_act = true,
                        float swiglu_limit = 10.0f, Stream stream = nullptr);
/// The int8 mode starting from x already quantised in the NATURAL order (xq / xs of ds41_quantize_acts<G>(..., ActOrder::kNatural), width kHidden);
/// scratch.xq / xs are not used.
template <class G>
void ds41_shared_expert_q(Dev& dev, const SharedExpertWeights& w, const int8_t* xq, const float* xs, int T, float* y, const SharedExpertScratch& scratch,
                          float swiglu_limit = 10.0f, Stream stream = nullptr);

/// The SwiGLU epilogue alone: h = silu(min(g, L)) * clamp(u, -L, L) over `n` = T * width values; writes h over g.  (With NaN propagation.)
template <class G>
void ds41_swiglu(Dev& dev, float* g_h, const float* u, int n, float swiglu_limit = 10.0f, Stream stream = nullptr);

// ---- the vocabulary ---------------------------------------------------------------------------------------------------------------------------
/// idx[t] = the index of the largest of logits[t][0 .. kVocab) (the LOWEST index among equals; a NaN counts as the largest), val[t] = logits[t][idx[t]].
/// logits fp32 [T][kVocab] (16-byte aligned, kVocab % 4 == 0).  One block per token: ~10 us at kVocab = 129,280.
template <class G>
void ds41_argmax(Dev& dev, const float* logits, int T, int32_t* idx, float* val, Stream stream = nullptr);

/// Candidate scratch of ds41_topk: T * slices * k (key, index) pairs.
template <class G>
constexpr size_t topk_scratch_bytes(int T, int k) {
    return (size_t) T * (((size_t) G::kVocab + 2047) / 2048) * (size_t) k * 8;
}
/// The k largest (k = 1..64) of each logits row, in descending order (ties: lower index first): idx[t][i], val[t][i] (a NaN is reported as NaN, -0 as +0).
/// scratch = topk_scratch_bytes<G>(T, k) device bytes.  Two launches (per-slice top-k, then a merge).
template <class G>
void ds41_topk(Dev& dev, const float* logits, int T, int k, int32_t* idx, float* val, void* scratch, size_t scratch_bytes, Stream stream = nullptr);

// ---- the embedding row (host -> device) --------------------------------------------------------------------------------------------------------
/// HOST: out[t][d] = float(table[ids[t]][d]) for the BF16 `token_embd` matrix [vocab][hidden] kept in host memory (raw bits).  Throws on an id outside
/// [0, vocab).
inline void ds41_embed_rows_host(const uint16_t* table, int vocab, int hidden, const int32_t* ids, int T, float* out) {
    for (int t = 0; t < T; ++t) {
        const int32_t id = ids[t];
        if (id < 0 || id >= vocab) throw std::invalid_argument("ds41_embed_rows: token id out of range");
        const uint16_t* row = table + (size_t) id * hidden;
        for (int d = 0; d < hidden; ++d) {
            const uint32_t b = (uint32_t) row[d] << 16;
            float f;
            static_assert(sizeof(f) == sizeof(b));
            std::memcpy(&f, &b, 4);
            out[(size_t) t * hidden + d] = f;
        }
    }
}
/// BF16 rows -> FP32 -> device: dev_out [T][kHidden].  `host_stage` (caller-owned, resized to T * kHidden) holds the FP32 rows; the copy is the Dev's blocking
/// h2d on dev.stream() (so it is complete, and the stage reusable, when the call returns; use dev.stream() for `stream` consumers or synchronise).
template <class G>
void ds41_embed_rows(Dev& dev, const uint16_t* table_host, const int32_t* ids_host, int T, float* dev_out, std::vector<float>& host_stage) {
    host_stage.resize((size_t) T * G::kHidden);
    ds41_embed_rows_host(table_host, G::kVocab, G::kHidden, ids_host, T, host_stage.data());
    dev.h2d(dev_out, host_stage.data(), host_stage.size() * sizeof(float));
}

// ---- FP32 elementwise helpers -----------------------------------------------------------------------------------------------------------------
/// out[i] = a[i] + b[i] for i < n (out may equal a or b); n a multiple of 4, 16-byte aligned pointers.
template <class G>
void ds41_f32_add(Dev& dev, const float* a, const float* b, int n, float* out, Stream stream = nullptr);
/// out[i] = a[i] * s (out may equal a).
template <class G>
void ds41_f32_scale(Dev& dev, const float* a, float s, int n, float* out, Stream stream = nullptr);

}  // namespace strata::ds41::cuda
