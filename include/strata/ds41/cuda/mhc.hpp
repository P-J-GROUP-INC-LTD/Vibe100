// include/strata/ds41/cuda/mhc.hpp - DS1-D: mHC (manifold-constrained hyper-connections), the 4-copy FP32 residual stream, for the V100.
//
// ORACLE: ref/ds41/mhc.py (sinkhorn_split, hc_mixes, hc_pre, hc_post) and the "single-pass one-block lag" of ref/ds41/model.py
// (Model.block / Model.forward); official: third_party/deepseek-v41-flash-reference/inference/model.py Block.hc_mixes / hc_pre / hc_post and
// kernel.py hc_split_sinkhorn_kernel.  Everything here is FP32 (docs/deepseek/DS1.md section 2: the stream is FP32 [T][kHc][kHidden]).
//
// THE MATH (per token; x = the stream [kHc][kHidden], xf = flatten(x), K = kHc * kHidden = Derived<G>::kHcFlat):
//     mixes = (hc_fn[kHcMixes][K] @ xf) * rsqrt(mean(xf^2) + norm_eps)             norm_eps = 1e-20 (NOT hc_eps)
//     pre   = sigmoid(mixes[0:kHc]      * scale[0] + base[0:kHc]) + hc_eps          hc_eps = 1e-6
//     post  = 2 * sigmoid(mixes[kHc:2kHc] * scale[1] + base[kHc:2kHc])
//     comb  = mixes[2kHc:] * scale[2] + base[2kHc:]   viewed [kHc(j)][kHc(k)]
//     comb  = softmax(comb over k) + hc_eps; comb /= (colsum_j(comb) + hc_eps);  then (kHcIters - 1) x { comb /= (rowsum_k + hc_eps);
//             comb /= (colsum_j + hc_eps) }
//     hc_pre : y[d]       = sum_c pre[c] * x[c][d]                                  ([T][kHc][kHidden] -> [T][kHidden])
//     hc_post: new[k][d]  = post[k] * f[d] + sum_j comb[j][k] * x[j][d]             (f = the sub-layer's output [T][kHidden])
//
// THE COEFFICIENT RECORD.  hc_mixes writes, per token, kHcMixes = (2 + kHc) * kHc floats in the layout of `mixes` itself:
//     coef[t][0 .. kHc)               pre
//     coef[t][kHc .. 2 kHc)           post
//     coef[t][2 kHc .. 2 kHc + kHc^2) comb, row-major comb[j][k] at 2 kHc + j * kHc + k
// hc_pre reads `pre` and hc_post reads `post` and `comb` out of a record, so the one-block lag is nothing but WHICH record hc_pre is pointed at.
//
// THE ONE-BLOCK LAG (what the session keeps; docs of Model.block): a block's attention sub-layer collapses the stream with the `pre` that the
// PREVIOUS block's FFN mixes produced (block 0: pre = [1, 0, 0, 0]); its FFN sub-layer collapses with the `pre` of its OWN attention mixes;
// after the last block the head fold is hc_pre with the last block's FFN `pre`.  The caller keeps three records (HcRecords) per forward pass and
// calls the per-sub-layer functions below, which do the bookkeeping (which record hc_pre reads, which one hc_mixes writes, when the lag moves):
//
//     HcRecords<G> rec(carve of hc_records_bytes<G>(T), T);      ds41_hc_begin<G>(dev, rec, T);                 // once per token batch: lag = [1, 0, 0, 0]
//     for each block l:
//         [Engram, on its layers, rewrites the stream x in place: the records are untouched by it]
//         ds41_hc_attn_in<G>(dev, x, attn_hc, T, prm, rec, y, ws, ws_bytes);   // mixes(x) -> rec.a ;  y = sum_c rec.lag.pre[c] x[c]   (the LAG)
//         ... y = attn_norm(y); y = attention(y) ...
//         ds41_hc_attn_out<G>(dev, y, x, T, rec);                              // x <- post_a * y + comb_a^T x                    (in place)
//         ds41_hc_ffn_in<G>(dev, x, ffn_hc, T, prm, rec, y, ws, ws_bytes);     // mixes(NEW x) -> rec.f ; y = sum_c rec.a.pre[c] x[c]  (the attention's OWN pre)
//         ... y = ffn_norm(y); y = moe(y) ...
//         ds41_hc_ffn_out<G>(dev, y, x, T, rec);                               // x <- post_f * y + comb_f^T x ;  then lag <- f (two pointers swap; no copy)
//     ds41_hc_head_fold<G>(dev, x, rec, T, h);                                 // the final fold with the last FFN's pre, then norm -> head
//
// (Each is two or one launch of the primitives below; the primitives stay public for tests and tracing.)  After ds41_hc_ffn_out the record `rec.lag`
// holds this block's FFN mixes: its first kHc floats per token are the trace stage `pre_mix.L`.
//
// Tokens are independent: T tokens (a prompt chunk, a verify window) go through the same calls; a token's results are bit-identical for every
// T (the per-token reduction order is fixed) and for every block / thread scheduling order (no atomics anywhere).  DS-1 prefill feeds one
// token at a time (T = 1).
//
// All pointers are device pointers; every op is `template <class G> void ds41_<op>(Dev&, args..., Stream = nullptr)` (docs/deepseek/DS1.md section 5;
// nullptr = dev.stream()), stream-ordered and allocation-free; scratch comes from the caller.  Alignment: every float pointer 16 bytes (hc_fn rows are 16-byte multiples).  Errors (bad
// alignment, too little scratch) throw std::invalid_argument; a failed launch std::runtime_error.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"

namespace strata::ds41::cuda {

/// The constants the GGUF carries (hyper_connection.epsilon, attention.layer_norm_rms_epsilon); the defaults are the released model's.
struct HcParams {
    float hc_eps = 1e-6f;     // pre += eps, comb softmax + eps, every Sinkhorn division is by (sum + eps)
    float norm_eps = 1e-20f;  // rsqrt(mean(x^2) + norm_eps) of hc_mixes (RMSNorm's 1e-20, deliberately not hc_eps)
};

/// One sub-layer's mHC parameters (hc_{attn,ffn}_{fn,scale,base} in the GGUF, all F32).
struct HcWeights {
    const float* fn;     // [kHcMixes][kHc * kHidden] row-major (GGUF dims 20480 x 24), 16-byte aligned
    const float* scale;  // [3]
    const float* base;   // [kHcMixes]
};

/// Number of K-slices hc_mixes splits the flattened stream into (one thread block each, 256 floats of K per block).
template <class G> constexpr int hc_slices() { return Derived<G>::kHcFlat / 256; }

/// Floats / bytes of the coefficient record of T tokens: [T][kHcMixes].
template <class G> constexpr size_t hc_coef_floats(int T) { return (size_t) T * G::kHcMixes; }
/// Bytes of hc_mixes scratch for T tokens (the per-slice partial sums): [T][hc_slices][32] floats.
template <class G> constexpr size_t hc_scratch_bytes(int T) { return (size_t) T * hc_slices<G>() * 32 * sizeof(float); }
/// Bytes of the three coefficient records HcRecords needs (256-byte aligned parts).
template <class G> constexpr size_t hc_records_bytes(int T) {
    return 3 * (((size_t) T * G::kHcMixes * sizeof(float) + 255) / 256 * 256);
}

/// Three coefficient records and the lag bookkeeping (see the header comment).  Pure pointer arithmetic: no allocation, no device calls.
template <class G> struct HcRecords {
    float* a = nullptr;    // this block's attention mixes
    float* f = nullptr;    // this block's FFN mixes
    float* lag = nullptr;  // the record whose `pre` the next attention hc_pre uses (identity at block 0)
    HcRecords() = default;
    /// Carve three records out of `base` (device memory, hc_records_bytes<G>(T) bytes, 16-byte aligned).
    HcRecords(void* base, int T) {
        const size_t part = ((size_t) T * G::kHcMixes * sizeof(float) + 255) / 256 * 256;
        unsigned char* p = static_cast<unsigned char*>(base);
        a = reinterpret_cast<float*>(p);
        f = reinterpret_cast<float*>(p + part);
        lag = reinterpret_cast<float*>(p + 2 * part);
    }
    /// After a block: the FFN record becomes the lag (the old lag record is free to be the next block's FFN record).
    void next_block() {
        float* t = lag;
        lag = f;
        f = t;
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------
// ops (templates over G; RealGeom is instantiated in mhc.cu, MiniGeom in the emulator build)
// ---------------------------------------------------------------------------------------------------------------------------------

/// mixes -> pre / post / comb for T tokens.  `x` = the stream [T][kHc][kHidden]; `coef` [T][kHcMixes] (the record layout above); `ws` = hc_scratch_bytes<G>(T)
/// device bytes (`ws_bytes` is checked); `mixes_out` (optional, [T][kHcMixes]) receives the mixes BEFORE the split (tests, tracing).
/// Two launches: per-slice partial dot products (hc_fn is 1.97 MB of FP32: 80 thread blocks stream it once; every weight vector element
/// is read once per call, whatever T <= 8), then one warp per token adds the 80 partials in slice order, applies rsqrt(mean(x^2) + norm_eps) and
/// runs the split / Sinkhorn on lanes.  Mirrors mhc.py hc_mixes + sinkhorn_split.
template <class G>
void ds41_hc_mixes(Dev& dev, const float* x, const HcWeights& w, int T, const HcParams& prm, float* coef, void* ws, size_t ws_bytes,
                   float* mixes_out = nullptr, Stream stream = nullptr);

/// The split / Sinkhorn alone: mixes [T][kHcMixes] -> coef [T][kHcMixes] (one warp per token; the same device code ds41_hc_mixes finishes with).
/// mhc.py sinkhorn_split.
template <class G>
void ds41_hc_split(Dev& dev, const float* mixes, const float* scale, const float* base, int T, const HcParams& prm, float* coef,
                   Stream stream = nullptr);

/// y[t][d] = sum_c pre[c] * x[t][c][d] with pre = coef[t][0 .. kHc) (a record from ds41_hc_mixes or ds41_hc_set_identity_pre).  mhc.py hc_pre.
/// Also the final head fold (pre = the last block's FFN record).  The sum is in copy order, each product rounded separately (as numpy).
template <class G>
void ds41_hc_pre(Dev& dev, const float* x, const float* coef, int T, float* y, Stream stream = nullptr);

/// out[t][k][d] = post[k] * f[t][d] + sum_j comb[j][k] * res[t][j][d], post / comb from the record.  mhc.py hc_post.
/// `out` may EQUAL `res` (every thread reads all kHc residual copies of its columns before it writes any); partial overlap is not allowed.
template <class G>
void ds41_hc_post(Dev& dev, const float* f, const float* res, const float* coef, int T, float* out, Stream stream = nullptr);

/// coef[t] = { pre = [1, 0, ..., 0], post = 0, comb = 0 }: block 0's lag (model.py make_identity_pre_mix).  Only `pre` is meaningful.
template <class G>
void ds41_hc_set_identity_pre(Dev& dev, float* coef, int T, Stream stream = nullptr);

/// Embedding -> stream: out[t][c][d] = emb[t][d] for every copy c (model.py: np.repeat(h[:, None, :], hc_mult, axis=1)).
template <class G>
void ds41_hc_expand(Dev& dev, const float* emb, int T, float* x_out, Stream stream = nullptr);

// ---- the per-sub-layer API: the primitives above in the order of the one-block lag (header comment) -------------------------------------
/// Start of a token batch: the lag record becomes pre = [1, 0, 0, 0] (model.py make_identity_pre_mix).
template <class G>
void ds41_hc_begin(Dev& dev, HcRecords<G>& rec, int T, Stream stream = nullptr) {
    ds41_hc_set_identity_pre<G>(dev, rec.lag, T, stream);
}
/// Attention sub-layer input: mixes of x -> rec.a; y = hc_pre(x, the LAG record's pre).
template <class G>
void ds41_hc_attn_in(Dev& dev, const float* x, const HcWeights& w, int T, const HcParams& prm, HcRecords<G>& rec, float* y, void* ws, size_t ws_bytes,
                     Stream stream = nullptr) {
    ds41_hc_mixes<G>(dev, x, w, T, prm, rec.a, ws, ws_bytes, nullptr, stream);
    ds41_hc_pre<G>(dev, x, rec.lag, T, y, stream);
}
/// Attention sub-layer output: x <- hc_post(f, x, rec.a), in place.
template <class G>
void ds41_hc_attn_out(Dev& dev, const float* f, float* x, int T, const HcRecords<G>& rec, Stream stream = nullptr) {
    ds41_hc_post<G>(dev, f, x, rec.a, T, x, stream);
}
/// FFN sub-layer input: mixes of the NEW x -> rec.f; y = hc_pre(x, the attention's OWN pre: rec.a).
template <class G>
void ds41_hc_ffn_in(Dev& dev, const float* x, const HcWeights& w, int T, const HcParams& prm, HcRecords<G>& rec, float* y, void* ws, size_t ws_bytes,
                    Stream stream = nullptr) {
    ds41_hc_mixes<G>(dev, x, w, T, prm, rec.f, ws, ws_bytes, nullptr, stream);
    ds41_hc_pre<G>(dev, x, rec.a, T, y, stream);
}
/// FFN sub-layer output: x <- hc_post(f, x, rec.f), in place; then the lag moves on (rec.lag = this block's FFN record).
template <class G>
void ds41_hc_ffn_out(Dev& dev, const float* f, float* x, int T, HcRecords<G>& rec, Stream stream = nullptr) {
    ds41_hc_post<G>(dev, f, x, rec.f, T, x, stream);
    rec.next_block();
}
/// After the last block: h = hc_pre(x, the last FFN's pre) (the head fold); h [T][kHidden] feeds the final RMSNorm and the head.
template <class G>
void ds41_hc_head_fold(Dev& dev, const float* x, const HcRecords<G>& rec, int T, float* h, Stream stream = nullptr) {
    ds41_hc_pre<G>(dev, x, rec.lag, T, h, stream);
}

// ---- test / tooling hooks -----------------------------------------------------------------------------------------------------------
/// What the compiler made of the mHC kernels (registers per thread, static smem); zeros in the emulator build.  Which: 0 = partial GEMV, 1 = finalize,
/// 2 = hc_pre, 3 = hc_post.
struct HcKernelInfo {
    int regs = 0;
    int static_smem = 0;
};
template <class G> HcKernelInfo ds41_hc_kernel_info(int which);

}  // namespace strata::ds41::cuda
