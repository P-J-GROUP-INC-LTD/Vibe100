// src/ds41/cuda/mhc_impl.cuh - DS1-D: mHC (hc_mixes, Sinkhorn split, hc_pre, hc_post) for the V100 (sm_70).  Included by mhc.cu (nvcc, RealGeom) and
// by the CPU emulation test (-DDS41_EMU, MiniGeom and RealGeom shapes).  The API and the math are include/strata/ds41/cuda/mhc.hpp; the oracle is
// ref/ds41/mhc.py.
//
// HOW (the GEMV is the only thing with bytes: hc_fn is 24 x 20480 FP32 = 1.97 MB per sub-layer, 80 sub-layers = 157 MB per token):
//   * hc_partial_kernel<G>: grid (80 K-slices, ceil(T / 8)), 4 warps.  A slice is 256 floats of the flattened stream; warp w owns hc_fn rows
//     [6w, 6w + 6): a lane holds 2 float4 of each of its 6 rows (12 x 16 B loads in flight per lane; the whole 1.97 MB is requested at once, one
//     wave over 80 SMs), then walks the block's tokens (up to 8; the next token's x is requested while the current one is multiplied): 6 chains of 8 FMAs each.  The 32 lanes
//     are summed with a vector butterfly (V = 8 values for 6 rows: 9 shuffles per token, not 30).  Warp 0 also sums x^2.  The per-slice partial
//     sums go to a [T][80][32] FP32 scratch: no atomics, so every sum has ONE fixed order.
//   * hc_finalize_kernel<G, FROM_PARTIALS>: one warp per token.  Lane l (l <= 24) adds the 80 partials of its output in slice order (lane 24 = the
//     sum of squares), rsqrt(mean + norm_eps), mixes = dot * rs, then the split on lanes: lanes 0-3 pre, 4-7 post, 8-23 the 4x4 comb; every row / column
//     sum of the Sinkhorn is a handful of shuffles added in the order numpy adds them (((a + b) + c) + d), every division an IEEE division of
//     the oracle's operands.
//   * hc_pre_kernel / hc_post_kernel: elementwise over float4 columns, the 4 copies of a column read by one thread.  Every product is rounded
//     separately and the sums are in copy order (fmul_rn / fadd_rn: no contraction), i.e. numpy's float32 evaluation of mhc.py hc_pre / hc_post; hc_post
//     may write over its residual input (all reads of a thread precede its writes; no restrict, no __ldg on the aliased input).
// REDUCTION ORDER (the definition; independent of T and of the block / fiber schedule):
//   dot[r] of token t   per lane: i = 0, 1 (float4 columns lane + 32 i of the slice), components x, y, z, w:  acc = fma(w, x, acc);
//                       lane total: butterfly offsets 16, 8, 4 (vector stages) then 2, 1;  slice total = that;  dot = ((p_0 + p_1) + ...) + p_79
//   sumsq               the same with x * x
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "mhc_common.cuh"
#include "strata/ds41/cuda/mhc.hpp"

namespace strata::ds41::cuda::dev {

inline constexpr int kHcKB = 256;       // floats of the flattened stream per partial block (= hc_slices<G>() slices of K)
inline constexpr int kHcPartWarps = 4;
inline constexpr int kHcPartTokens = 8;       // tokens per block of the partial kernel (T > 8: more blocks along y)
inline constexpr int kHcPartThreads = 32 * kHcPartWarps;
inline constexpr int kHcColsThreads = 128;      // hc_pre / hc_post / expand: threads per block, one float4 column each

// ---- kernel 1: per-slice partial dot products and sums of squares -------------------------------------------------------------------------
// One kernel for every T: a block walks up to kHcPartTokens tokens (a runtime loop, not unrolled: the weights stay in registers, the next token's x is
// requested while the current one is multiplied), so the 1.97 MB of hc_fn are read once per call whatever T <= 8 and the code is one token long.
template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kHcPartThreads) void hc_partial_kernel(const float* DS41_RESTRICT x, const float* DS41_RESTRICT fn, int T,
                                                                    float* DS41_RESTRICT part) {
    constexpr int K = Derived<G>::kHcFlat;
    constexpr int S = K / kHcKB;
    constexpr int NR = G::kHcMixes;
    constexpr int RPW = (NR + kHcPartWarps - 1) / kHcPartWarps;       // rows per warp (6)
    constexpr int V = hc_pow2_ceil(RPW);                               // 8
    constexpr int I = kHcKB / 4 / 32;                                  // float4 columns per lane and row (2)
    static_assert(K % kHcKB == 0 && NR <= 31 && RPW <= 32 && I >= 1);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int s = blockIdx.x;
    const int t0 = blockIdx.y * kHcPartTokens;
    const int nt = T - t0 < kHcPartTokens ? T - t0 : kHcPartTokens;    // tokens of this block, >= 1 (uniform)
    const int col0 = s * (kHcKB / 4) + lane;                           // float4 column of this lane's first element in the flattened stream

    float4 w[RPW][I];                                                   // all of this lane's weights, requested up front
    DS41_UNROLL
    for (int r = 0; r < RPW; ++r) {
        const int row = warp * RPW + r;
        DS41_UNROLL
        for (int i = 0; i < I; ++i)
            w[r][i] = row < NR ? ldgf4(fn + (size_t) row * K + 4 * (col0 + 32 * i)) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float4 xv[I];
    DS41_UNROLL
    for (int i = 0; i < I; ++i) xv[i] = ldgf4(x + (size_t) t0 * K + 4 * (col0 + 32 * i));
    DS41_UNROLL1
    for (int t = 0; t < nt; ++t) {
        float4 xn[I];
        const int tn = t + 1 < nt ? t + 1 : t;
        DS41_UNROLL
        for (int i = 0; i < I; ++i) xn[i] = ldgf4(x + (size_t) (t0 + tn) * K + 4 * (col0 + 32 * i));
        float acc[V];
        DS41_UNROLL
        for (int r = 0; r < V; ++r) {
            float a = 0.0f;
            if (r < RPW) {
                DS41_UNROLL
                for (int i = 0; i < I; ++i) {
                    a = fmaf(w[r][i].x, xv[i].x, a);
                    a = fmaf(w[r][i].y, xv[i].y, a);
                    a = fmaf(w[r][i].z, xv[i].z, a);
                    a = fmaf(w[r][i].w, xv[i].w, a);
                }
            }
            acc[r] = a;
        }
        const float mine = hc_vsum<V>(acc);
        const int idx = lane >> (5 - hc_log2(V));
        const int row = warp * RPW + idx;
        float* out = part + ((size_t) (t0 + t) * S + s) * 32;
        if ((lane & ((1 << (5 - hc_log2(V))) - 1)) == 0 && idx < RPW && row < NR) out[row] = mine;
        if (warp == 0) {                                                // sum of squares of the slice (once per token, not per warp)
            float q = 0.0f;
            DS41_UNROLL
            for (int i = 0; i < I; ++i) {
                q = fmaf(xv[i].x, xv[i].x, q);
                q = fmaf(xv[i].y, xv[i].y, q);
                q = fmaf(xv[i].z, xv[i].z, q);
                q = fmaf(xv[i].w, xv[i].w, q);
            }
            q = hc_warp_sum(q);
            if (lane == 0) out[NR] = q;
        }
        DS41_UNROLL
        for (int i = 0; i < I; ++i) xv[i] = xn[i];
    }
}

// ---- kernel 2: finish the sums, rsqrt, split / Sinkhorn on lanes ------------------------------------------------------------------------------
// Lane roles (kHc = 4): 0..3 pre, 4..7 post, 8..23 comb[j][k] at lane 8 + 4 j + k, 24 = the sum of squares (FROM_PARTIALS only).
template <class G, bool FROM_PARTIALS>
DS41_KERNEL DS41_LAUNCH_BOUNDS(32) void hc_finalize_kernel(const float* DS41_RESTRICT in, int T, const float* DS41_RESTRICT scale,
                                                         const float* DS41_RESTRICT base, float norm_eps, float hc_eps, float* DS41_RESTRICT coef,
                                                         float* DS41_RESTRICT mixes_out) {
    constexpr int H = G::kHc;
    constexpr int NR = G::kHcMixes;
    constexpr int S = Derived<G>::kHcFlat / kHcKB;
    constexpr int K = Derived<G>::kHcFlat;
    constexpr int C0 = 2 * H;                                            // first comb lane
    static_assert(NR <= 31 && NR == (2 + H) * H);
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x;

    float m = 0.0f;                                                      // this lane's mixes value (lanes < NR)
    if (FROM_PARTIALS) {
        float v = 0.0f;
        if (lane <= NR) {
            const float* p = in + (size_t) t * S * 32 + lane;
            DS41_UNROLL                                                   // all S loads in flight, then the adds in slice order (the definition of the sum)
            for (int s = 0; s < S; ++s) v += ldgf(p + (size_t) s * 32);
        }
        const float ss = shfl(v, NR);
        const float mean = fdiv_rn(ss, (float) K);
        const float rs = fdiv_rn(1.0f, sqrtf(fadd_rn(mean, norm_eps)));
        m = fmul_rn(v, rs);
    } else {
        m = lane < NR ? ldgf(in + (size_t) t * NR + lane) : 0.0f;
    }
    if (mixes_out != nullptr && lane < NR) mixes_out[(size_t) t * NR + lane] = m;

    const float s0 = ldgf(scale), s1 = ldgf(scale + 1), s2 = ldgf(scale + 2);
    const float b = lane < NR ? ldgf(base + lane) : 0.0f;

    // comb: lane -> (j, k); the lanes outside the comb range compute a copy of lane C0's problem (every lane takes part in every shuffle)
    const int cl = (lane >= C0 && lane < NR) ? lane - C0 : 0;
    const int j = cl / H, k = cl % H;
    const int rowlane = C0 + j * H;                                      // the lane of comb[j][0]
    const int collane = C0 + k;                                          // the lane of comb[0][k]

    float c = fadd_rn(fmul_rn(m, s2), b);                                // comb pre-activation
    {   // row softmax over k, then + eps
        float mx = shfl(c, rowlane);
        DS41_UNROLL
        for (int kk = 1; kk < H; ++kk) {
            const float o = shfl(c, rowlane + kk);
            mx = o > mx ? o : mx;
        }
        const float e = expf(c - mx);
        float sum = shfl(e, rowlane);
        DS41_UNROLL
        for (int kk = 1; kk < H; ++kk) sum = fadd_rn(sum, shfl(e, rowlane + kk));
        c = fadd_rn(fdiv_rn(e, sum), hc_eps);
    }
    auto col_normalise = [&]() {                                          // comb /= (sum over j + eps)
        float cs = shfl(c, collane);
        DS41_UNROLL
        for (int jj = 1; jj < H; ++jj) cs = fadd_rn(cs, shfl(c, collane + jj * H));
        c = fdiv_rn(c, fadd_rn(cs, hc_eps));
    };
    auto row_normalise = [&]() {                                          // comb /= (sum over k + eps)
        float rsum = shfl(c, rowlane);
        DS41_UNROLL
        for (int kk = 1; kk < H; ++kk) rsum = fadd_rn(rsum, shfl(c, rowlane + kk));
        c = fdiv_rn(c, fadd_rn(rsum, hc_eps));
    };
    col_normalise();
    DS41_UNROLL1
    for (int it = 1; it < G::kHcIters; ++it) {
        row_normalise();
        col_normalise();
    }

    float y = 0.0f;
    if (lane < H) y = fadd_rn(hc_sigmoid(fadd_rn(fmul_rn(m, s0), b)), hc_eps);                 // pre
    else if (lane < 2 * H) y = fmul_rn(2.0f, hc_sigmoid(fadd_rn(fmul_rn(m, s1), b)));          // post
    else y = c;                                                                                   // comb
    if (lane < NR) coef[(size_t) t * NR + lane] = y;
}

// ---- hc_pre / hc_post / identity / expand ---------------------------------------------------------------------------------------------------
template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kHcColsThreads) void hc_pre_kernel(const float* DS41_RESTRICT x, const float* DS41_RESTRICT coef, float* DS41_RESTRICT y) {
    constexpr int H = G::kHc, D4 = G::kHidden / 4, NR = G::kHcMixes;
    const int d4 = blockIdx.x * kHcColsThreads + threadIdx.x;
    const int t = blockIdx.y;
    if (d4 >= D4) return;
    float p[H];
    DS41_UNROLL
    for (int c = 0; c < H; ++c) p[c] = ldgf(coef + (size_t) t * NR + c);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    DS41_UNROLL
    for (int c = 0; c < H; ++c) {
        const float4 v = ldgf4(x + ((size_t) t * H + c) * G::kHidden + 4 * d4);
        if (c == 0) {
            a = make_float4(fmul_rn(p[0], v.x), fmul_rn(p[0], v.y), fmul_rn(p[0], v.z), fmul_rn(p[0], v.w));
        } else {
            a.x = fadd_rn(a.x, fmul_rn(p[c], v.x));
            a.y = fadd_rn(a.y, fmul_rn(p[c], v.y));
            a.z = fadd_rn(a.z, fmul_rn(p[c], v.z));
            a.w = fadd_rn(a.w, fmul_rn(p[c], v.w));
        }
    }
    hc_st4(y + (size_t) t * G::kHidden + 4 * d4, a);
}

/// `res` and `out` may be the same array (no restrict, plain loads): a thread reads its kHc residual float4 and the sub-layer output before it
/// writes its kHc outputs, and no other thread touches those columns.
template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kHcColsThreads) void hc_post_kernel(const float* DS41_RESTRICT f, const float* res, const float* DS41_RESTRICT coef, float* out) {
    constexpr int H = G::kHc, D4 = G::kHidden / 4, NR = G::kHcMixes;
    const int d4 = blockIdx.x * kHcColsThreads + threadIdx.x;
    const int t = blockIdx.y;
    if (d4 >= D4) return;
    float post[H], comb[H][H];
    DS41_UNROLL
    for (int k = 0; k < H; ++k) post[k] = ldgf(coef + (size_t) t * NR + H + k);
    DS41_UNROLL
    for (int j = 0; j < H; ++j) {
        DS41_UNROLL
        for (int k = 0; k < H; ++k) comb[j][k] = ldgf(coef + (size_t) t * NR + 2 * H + j * H + k);
    }
    const float4 fv = ldgf4(f + (size_t) t * G::kHidden + 4 * d4);
    float4 r[H];
    DS41_UNROLL
    for (int j = 0; j < H; ++j) r[j] = hc_ld4(res + ((size_t) t * H + j) * G::kHidden + 4 * d4);
    DS41_UNROLL
    for (int k = 0; k < H; ++k) {
        float4 o;
        float sx = fmul_rn(comb[0][k], r[0].x), sy = fmul_rn(comb[0][k], r[0].y), sz = fmul_rn(comb[0][k], r[0].z), sw = fmul_rn(comb[0][k], r[0].w);
        DS41_UNROLL
        for (int j = 1; j < H; ++j) {
            sx = fadd_rn(sx, fmul_rn(comb[j][k], r[j].x));
            sy = fadd_rn(sy, fmul_rn(comb[j][k], r[j].y));
            sz = fadd_rn(sz, fmul_rn(comb[j][k], r[j].z));
            sw = fadd_rn(sw, fmul_rn(comb[j][k], r[j].w));
        }
        o.x = fadd_rn(fmul_rn(post[k], fv.x), sx);
        o.y = fadd_rn(fmul_rn(post[k], fv.y), sy);
        o.z = fadd_rn(fmul_rn(post[k], fv.z), sz);
        o.w = fadd_rn(fmul_rn(post[k], fv.w), sw);
        hc_st4(out + ((size_t) t * H + k) * G::kHidden + 4 * d4, o);
    }
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(32) void hc_identity_kernel(float* coef) {
    constexpr int NR = G::kHcMixes;
    const int lane = threadIdx.x & 31;
    if (lane < NR) coef[(size_t) blockIdx.x * NR + lane] = lane == 0 ? 1.0f : 0.0f;
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kHcColsThreads) void hc_expand_kernel(const float* DS41_RESTRICT emb, float* DS41_RESTRICT out) {
    constexpr int H = G::kHc, D4 = G::kHidden / 4;
    const int d4 = blockIdx.x * kHcColsThreads + threadIdx.x;
    const int t = blockIdx.y;
    if (d4 >= D4) return;
    const float4 v = ldgf4(emb + (size_t) t * G::kHidden + 4 * d4);
    DS41_UNROLL
    for (int c = 0; c < H; ++c) hc_st4(out + ((size_t) t * H + c) * G::kHidden + 4 * d4, v);
}

}  // namespace strata::ds41::cuda::dev

// =====================================================================================================================================
// host side
// =====================================================================================================================================
namespace strata::ds41::cuda {

namespace hc_detail {
inline void need_aligned(const void* p, size_t a, const char* what) {
    if (reinterpret_cast<uintptr_t>(p) % a != 0) throw std::invalid_argument(std::string("ds41 mHC: ") + what + " must be " + std::to_string(a) + "-byte aligned");
}
inline unsigned cols_blocks(int hidden) { return (unsigned) ((hidden / 4 + dev::kHcColsThreads - 1) / dev::kHcColsThreads); }
}  // namespace hc_detail

template <class G>
void ds41_hc_mixes(Dev& d, const float* x, const HcWeights& w, int T, const HcParams& prm, float* coef, void* ws, size_t ws_bytes, float* mixes_out,
                   Stream stream) {
    static_assert(Derived<G>::kHcFlat % dev::kHcKB == 0);
    if (T < 1) return;
    hc_detail::need_aligned(x, 16, "stream");
    hc_detail::need_aligned(w.fn, 16, "hc_fn");
    hc_detail::need_aligned(ws, 16, "scratch");
    if (ws_bytes < hc_scratch_bytes<G>(T)) throw std::invalid_argument("ds41 hc_mixes: scratch smaller than hc_scratch_bytes<G>(T)");
    const Stream s = stream_or_default(d, stream);
    constexpr unsigned S = (unsigned) hc_slices<G>();
    float* part = static_cast<float*>(ws);
    dev::launch(dev::hc_partial_kernel<G>, dim3(S, (unsigned) ((T + dev::kHcPartTokens - 1) / dev::kHcPartTokens)), dim3(dev::kHcPartThreads), 0, s, x, w.fn, T, part);
    dev::check_launch("hc_partial");
    dev::launch(dev::hc_finalize_kernel<G, true>, dim3((unsigned) T), dim3(32), 0, s, (const float*) part, T, w.scale, w.base, prm.norm_eps, prm.hc_eps, coef,
                mixes_out);
    dev::check_launch("hc_finalize");
}

template <class G>
void ds41_hc_split(Dev& d, const float* mixes, const float* scale, const float* base, int T, const HcParams& prm, float* coef, Stream stream) {
    if (T < 1) return;
    const Stream s = stream_or_default(d, stream);
    dev::launch(dev::hc_finalize_kernel<G, false>, dim3((unsigned) T), dim3(32), 0, s, mixes, T, scale, base, prm.norm_eps, prm.hc_eps, coef, (float*) nullptr);
    dev::check_launch("hc_split");
}

template <class G>
void ds41_hc_pre(Dev& d, const float* x, const float* coef, int T, float* y, Stream stream) {
    if (T < 1) return;
    hc_detail::need_aligned(x, 16, "stream");
    hc_detail::need_aligned(y, 16, "y");
    const Stream s = stream_or_default(d, stream);
    dev::launch(dev::hc_pre_kernel<G>, dim3(hc_detail::cols_blocks(G::kHidden), (unsigned) T), dim3(dev::kHcColsThreads), 0, s, x, coef, y);
    dev::check_launch("hc_pre");
}

template <class G>
void ds41_hc_post(Dev& d, const float* f, const float* res, const float* coef, int T, float* out, Stream stream) {
    if (T < 1) return;
    hc_detail::need_aligned(f, 16, "sub-layer output");
    hc_detail::need_aligned(res, 16, "residual stream");
    hc_detail::need_aligned(out, 16, "output stream");
    if (out != res) {                                    // partial overlap would let one thread read what another has written
        const size_t n = (size_t) T * G::kHc * G::kHidden * sizeof(float);
        const auto a = reinterpret_cast<uintptr_t>(res), b = reinterpret_cast<uintptr_t>(out);
        if (a < b + n && b < a + n) throw std::invalid_argument("ds41 hc_post: out must equal res or not overlap it");
    }
    const Stream s = stream_or_default(d, stream);
    dev::launch(dev::hc_post_kernel<G>, dim3(hc_detail::cols_blocks(G::kHidden), (unsigned) T), dim3(dev::kHcColsThreads), 0, s, f, res, coef, out);
    dev::check_launch("hc_post");
}

template <class G>
void ds41_hc_set_identity_pre(Dev& d, float* coef, int T, Stream stream) {
    if (T < 1) return;
    const Stream s = stream_or_default(d, stream);
    dev::launch(dev::hc_identity_kernel<G>, dim3((unsigned) T), dim3(32), 0, s, coef);
    dev::check_launch("hc_identity");
}

template <class G>
void ds41_hc_expand(Dev& d, const float* emb, int T, float* x_out, Stream stream) {
    if (T < 1) return;
    hc_detail::need_aligned(emb, 16, "embedding rows");
    hc_detail::need_aligned(x_out, 16, "stream");
    const Stream s = stream_or_default(d, stream);
    dev::launch(dev::hc_expand_kernel<G>, dim3(hc_detail::cols_blocks(G::kHidden), (unsigned) T), dim3(dev::kHcColsThreads), 0, s, emb, x_out);
    dev::check_launch("hc_expand");
}

template <class G>
HcKernelInfo ds41_hc_kernel_info(int which) {
    HcKernelInfo info;
#if !defined(DS41_EMU)
    cudaFuncAttributes a{};
    cudaError_t e = cudaErrorInvalidValue;
    switch (which) {
        case 0: e = cudaFuncGetAttributes(&a, (const void*) dev::hc_partial_kernel<G>); break;
        case 1: e = cudaFuncGetAttributes(&a, (const void*) dev::hc_finalize_kernel<G, true>); break;
        case 2: e = cudaFuncGetAttributes(&a, (const void*) dev::hc_pre_kernel<G>); break;
        case 3: e = cudaFuncGetAttributes(&a, (const void*) dev::hc_post_kernel<G>); break;
        default: break;
    }
    if (e == cudaSuccess) {
        info.regs = a.numRegs;
        info.static_smem = (int) a.sharedSizeBytes;
    }
#else
    (void) which;
#endif
    return info;
}

}  // namespace strata::ds41::cuda
