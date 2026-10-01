// src/ds41/cuda/ds41_router_impl.cuh - DS-D: the sqrt-softplus router (V100, sm_70).  Included by ds41_router.cu (nvcc) and by the
// CPU emulation test (-DDS41_EMU).
//
// WHAT (docs/deepseek/CONTRACTS.md, "Router"; reference: third_party/deepseek-v41-flash-reference/inference/model.py Gate):
//     logit[e] = x . Wg[e]            FP32 accumulation, Wg is BF16 [384][5120] read as raw bits (bf16 -> fp32 is a 16-bit shift:
//                                     Volta has no bf16 math and needs none)
//     s[e]     = sqrt(softplus(logit[e]))   softplus(x) = x for x > 20 (torch's threshold), log1p(exp(x)) below: no overflow
//     idx      = top-6 of (s[e] + bias[e])  by value, descending; ties -> the LOWEST index (the bias only picks, never weighs)
//     w[r]     = s[idx[r]];  w = w / (sum(w) + 1e-20);  w = w * 1.5       (two separate roundings, as the reference)
//
// HOW.  Two kernels per call, the logits in between in a [T][384] FP32 workspace (the parity program reads it to report near-ties):
//   * router_logits_gemv_kernel (T <= 32): one BLOCK of 10 warps per expert (the K axis is split in 10 slices, one per warp, 2 x 512 B of
//     the bf16 row each), the loads of weights AND activations all issued up front, each weight chunk converted to 8 floats ONCE and
//     multiplied with every token of the group (NT = 1, 2 or 4 tokens per block; more tokens = more blocks along y, re-reading the
//     3.9 MB weight matrix from L2).
//     Bytes: 3.93 MB of weights per call = ~4.4 us at 900 GB/s; the FP32 FMAs (T x 2 M) are far below the FMA roof.
//   * router_logits_prefill_kernel (T > 32): the SAME per-(token, expert) reduction as the GEMV above - bit for bit - with the token loop
//     inside the block: a block of 10 warps holds the weights of 4 experts in registers (converted to FP32 once) and walks 16 tokens, so
//     Wg is read once per 16 tokens and each token's x once per 4 experts, from L2 (prefill: 2 x T x 384 x 5120 FMA; FP32 on purpose: the
//     logits decide the routing, so they are not rounded to fp16 for the tensor cores).  It replaces the shared-memory tiled GEMM this file
//     had, whose K order (one sequential chain over all 5120) differed from the GEMV's: a token routed in a prompt chunk and the same
//     token routed alone in decode got logits that differed in the last bits, so a near-tie could pick different experts (CONTRACTS.md:
//     "Routing must not depend on T").
//   * router_select_kernel: one warp per token: 12 experts per lane, 6 rounds of warp argmax with a (value, index) total order.
// ONE REDUCTION ORDER FOR EVERY T (the definition, shared by both logits kernels; the GEMV's NT = 1, 2, 4 and the prefill kernel give the same bits):
//   lane partial   s = 0; for the lane's 2 chunk groups g = 0, 1 (chunk c = lane + 32 * (2 * warp + g), 8 consecutive k each):
//                  s = fma(bf16(wg[e][k]), x[t][k], s)   for the 8 k of the chunk, in order      (16 chained FMAs)
//   warp total     a 32-lane butterfly, offsets 16, 8, 4, 2, 1:  s += shfl_xor(s, off)            (lanes_vector_sum: any number of values
//                  per lane gives the same tree; every lane ends with the same bits)
//   logit          tot = 0; tot += total of warp 0, 1, ..., 9 in this order
// Every reduction has a fixed order: the same input gives the same bits, run to run, and for every T.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "ds41_math.cuh"
#include "strata/ds41/cuda/ds41_cuda.hpp"

namespace strata::ds41::cuda::dev {

inline constexpr int kRtChunks = kHidden / 8;                 // 640 chunks of 8 bf16 (16 B) per expert row
inline constexpr int kRtPerLane = kRtChunks / 32;             // 20
inline constexpr int kRtWarps = 4;
static_assert(kRtChunks % 32 == 0 && kExperts % kRtWarps == 0 && kExperts % 32 == 0);

// ---- logits, small T ------------------------------------------------------------------------------------------------------------
// One BLOCK per expert and token group: 10 warps, each takes 2 of the row's 20 "chunk groups" (a chunk group = 32 lanes x 16 bytes =
// 256 bf16 = 512 B of the row), so a lane owns 2 weight chunks and the matching 2 x 2 float4 of each token's x: everything it needs
// is loaded up front (~1 KB per warp in flight, 50 warps per SM), multiplied, reduced over the warp with a shuffle tree, and the 10
// warp totals are added in a fixed order through shared memory.
inline constexpr int kRtSliceWarps = 10;                      // warps per block = slices of the K axis
inline constexpr int kRtGroupsPerWarp = kRtPerLane / kRtSliceWarps;   // 2
static_assert(kRtPerLane % kRtSliceWarps == 0);

template <int NT>
DS41_KERNEL DS41_LAUNCH_BOUNDS(320) void router_logits_gemv_kernel(const float* DS41_RESTRICT x, const uint16_t* DS41_RESTRICT wg, int T,
                                                                   float* DS41_RESTRICT logits) {
    DS41_SHARED float s_part[kRtSliceWarps][NT];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e = blockIdx.x;
    const int t0 = blockIdx.y * NT;
    const unsigned char* wrow = reinterpret_cast<const unsigned char*>(wg + (size_t) e * kHidden);

    uint4 wc[kRtGroupsPerWarp];
    float4 xa[NT][kRtGroupsPerWarp][2];
    DS41_UNROLL
    for (int g = 0; g < kRtGroupsPerWarp; ++g) {
        const int c = lane + 32 * (warp * kRtGroupsPerWarp + g);       // chunk index 0..639 (8 elements each)
        wc[g] = ldg4(wrow + 16 * c);
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) {
            if (t0 + t < T) {
                const float* xp = x + (size_t) (t0 + t) * kHidden + 8 * c;
                xa[t][g][0] = ldgf4(xp);
                xa[t][g][1] = ldgf4(xp + 4);
            } else {
                xa[t][g][0] = xa[t][g][1] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }
    // Keep every load at the top: without a dependency of the first FMA on ALL of them, ptxas sinks each load next to its use and the
    // warp has ~3 loads in flight instead of ~10.  `zero` is 0 at run time (T < 2^20) but opaque to the compiler.
    uint32_t anyw = 0;
    DS41_UNROLL
    for (int g = 0; g < kRtGroupsPerWarp; ++g) {
        anyw |= wc[g].x | wc[g].w;
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) anyw |= f2u(xa[t][g][0].x) | f2u(xa[t][g][1].w);
    }
    const float zero = u2f(anyw & (uint32_t) (T >> 20));

    float acc[NT];
    DS41_UNROLL
    for (int t = 0; t < NT; ++t) {
        float s = zero;
        DS41_UNROLL
        for (int g = 0; g < kRtGroupsPerWarp; ++g) {
            const float w0 = u2f(wc[g].x << 16), w1 = u2f(wc[g].x & 0xFFFF0000u);   // bf16 -> fp32: the bits are the top half
            const float w2 = u2f(wc[g].y << 16), w3 = u2f(wc[g].y & 0xFFFF0000u);
            const float w4 = u2f(wc[g].z << 16), w5 = u2f(wc[g].z & 0xFFFF0000u);
            const float w6 = u2f(wc[g].w << 16), w7 = u2f(wc[g].w & 0xFFFF0000u);
            const float4 a = xa[t][g][0], b = xa[t][g][1];
            s = fmaf(w0, a.x, s);
            s = fmaf(w1, a.y, s);
            s = fmaf(w2, a.z, s);
            s = fmaf(w3, a.w, s);
            s = fmaf(w4, b.x, s);
            s = fmaf(w5, b.y, s);
            s = fmaf(w6, b.z, s);
            s = fmaf(w7, b.w, s);
        }
        acc[t] = s;
    }
    const float mine = lanes_vector_sum<32, NT>(acc);
    constexpr int S = ilog2(NT);
    const int idx = lane >> (5 - S);
    if ((lane & ((1 << (5 - S)) - 1)) == 0) s_part[warp][idx] = mine;
    sync_block();
    if (threadIdx.x < NT && t0 + (int) threadIdx.x < T) {
        float tot = 0.0f;
        DS41_UNROLL
        for (int w = 0; w < kRtSliceWarps; ++w) tot += s_part[w][threadIdx.x];       // fixed order
        logits[(size_t) (t0 + threadIdx.x) * kExperts + e] = tot;
    }
}

// ---- logits, large T (prefill): the GEMV's reduction, tokens looped inside the block --------------------------------------------
// Block = 10 warps (the K axis in 10 slices, exactly as the GEMV), kRtPfExperts experts and up to kRtPfTokens tokens.  Each lane converts its
// 2 weight chunks of every expert to FP32 once, then for every token of the block loads its 2 x 2 float4 of x (the next token's loads are issued
// before the current token is multiplied), runs the 16 chained FMAs per expert, butterflies the kRtPfExperts values over the warp and parks the
// warp totals in shared memory; after the last token the ten totals of every (token, expert) are added in order by one thread each.
inline constexpr int kRtPfExperts = 4;                         // experts per block (grid.x = 384 / 4)
inline constexpr int kRtPfTokens = 16;                         // tokens per block (grid.y = ceil(T / 16))
static_assert(kExperts % kRtPfExperts == 0 && (kRtPfExperts & (kRtPfExperts - 1)) == 0);

DS41_KERNEL DS41_LAUNCH_BOUNDS(32 * kRtSliceWarps) void router_logits_prefill_kernel(const float* DS41_RESTRICT x, const uint16_t* DS41_RESTRICT wg, int T,
                                                                                    float* DS41_RESTRICT logits) {
    constexpr int NE = kRtPfExperts, TB = kRtPfTokens, G = kRtGroupsPerWarp;
    DS41_SHARED float s_part[TB][kRtSliceWarps][NE];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int e0 = blockIdx.x * NE;
    const int t0 = blockIdx.y * TB;
    const int nt = T - t0 < TB ? T - t0 : TB;                      // tokens of this block, >= 1 (uniform)

    float w[NE][G][8];                                             // this lane's weights, FP32, for the whole block
    DS41_UNROLL
    for (int ee = 0; ee < NE; ++ee) {
        const unsigned char* wrow = reinterpret_cast<const unsigned char*>(wg + (size_t) (e0 + ee) * kHidden);
        DS41_UNROLL
        for (int g = 0; g < G; ++g) {
            const int c = lane + 32 * (warp * G + g);              // chunk index 0..639 (8 elements each), as in the GEMV
            const uint4 q = ldg4(wrow + 16 * c);
            w[ee][g][0] = u2f(q.x << 16);                          // bf16 -> fp32: the bits are the top half
            w[ee][g][1] = u2f(q.x & 0xFFFF0000u);
            w[ee][g][2] = u2f(q.y << 16);
            w[ee][g][3] = u2f(q.y & 0xFFFF0000u);
            w[ee][g][4] = u2f(q.z << 16);
            w[ee][g][5] = u2f(q.z & 0xFFFF0000u);
            w[ee][g][6] = u2f(q.w << 16);
            w[ee][g][7] = u2f(q.w & 0xFFFF0000u);
        }
    }
    auto load_x = [&](float4 (&xa)[G][2], int tt) {
        DS41_UNROLL
        for (int g = 0; g < G; ++g) {
            const int c = lane + 32 * (warp * G + g);
            const float* xp = x + (size_t) (t0 + tt) * kHidden + 8 * c;
            xa[g][0] = ldgf4(xp);
            xa[g][1] = ldgf4(xp + 4);
        }
    };
    float4 xc[G][2];
    load_x(xc, 0);
    DS41_UNROLL1
    for (int tt = 0; tt < nt; ++tt) {
        float4 xn[G][2];
        load_x(xn, tt + 1 < nt ? tt + 1 : tt);                     // the next token's x flies while this one is multiplied
        float acc[NE];
        DS41_UNROLL
        for (int ee = 0; ee < NE; ++ee) {
            float sacc = 0.0f;
            DS41_UNROLL
            for (int g = 0; g < G; ++g) {
                const float4 a = xc[g][0], b = xc[g][1];
                sacc = fmaf(w[ee][g][0], a.x, sacc);
                sacc = fmaf(w[ee][g][1], a.y, sacc);
                sacc = fmaf(w[ee][g][2], a.z, sacc);
                sacc = fmaf(w[ee][g][3], a.w, sacc);
                sacc = fmaf(w[ee][g][4], b.x, sacc);
                sacc = fmaf(w[ee][g][5], b.y, sacc);
                sacc = fmaf(w[ee][g][6], b.z, sacc);
                sacc = fmaf(w[ee][g][7], b.w, sacc);
            }
            acc[ee] = sacc;
        }
        const float mine = lanes_vector_sum<32, NE>(acc);
        constexpr int S = ilog2(NE);
        const int idx = lane >> (5 - S);
        if ((lane & ((1 << (5 - S)) - 1)) == 0) s_part[tt][warp][idx] = mine;
        DS41_UNROLL
        for (int g = 0; g < G; ++g) {
            xc[g][0] = xn[g][0];
            xc[g][1] = xn[g][1];
        }
    }
    sync_block();
    for (int i = threadIdx.x; i < nt * NE; i += 32 * kRtSliceWarps) {
        const int tt = i / NE, ee = i - tt * NE;
        float tot = 0.0f;
        DS41_UNROLL
        for (int wi = 0; wi < kRtSliceWarps; ++wi) tot += s_part[tt][wi][ee];       // fixed order, as the GEMV
        logits[(size_t) (t0 + tt) * kExperts + e0 + ee] = tot;
    }
}

// ---- selection --------------------------------------------------------------------------------------------------------------------
/// (value descending, index ascending) is a total order on distinct indices.
DS41_FI bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

DS41_KERNEL DS41_LAUNCH_BOUNDS(128) void router_select_kernel(const float* DS41_RESTRICT logits, const float* DS41_RESTRICT bias, int T,
                                                              int32_t* DS41_RESTRICT ids, float* DS41_RESTRICT weights) {
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * kRtWarps + (threadIdx.x >> 5);
    if (t >= T) return;                                       // whole warp
    constexpr int PER = kExperts / 32;                        // 12 experts per lane: e = lane + 32 i
    float s[PER], v[PER];
    DS41_UNROLL
    for (int i = 0; i < PER; ++i) {
        const int e = lane + 32 * i;
        const float l = ldgf(logits + (size_t) t * kExperts + e);
        const float sp = l > 20.0f ? l : log1pf(expf(l));     // torch softplus (beta 1, threshold 20)
        s[i] = sqrtf(sp);
        const float val = s[i] + ldgf(bias + e);
        v[i] = val != val ? -INFINITY : val;                  // NaN never wins
    }
    uint32_t taken = 0;
    int isel[kTopK];
    float wsel[kTopK];
    DS41_UNROLL
    for (int r = 0; r < kTopK; ++r) {
        float bv = -INFINITY, bs = 0.0f;
        int bi = 0x7fffffff;
        DS41_UNROLL
        for (int i = 0; i < PER; ++i)
            if (!((taken >> i) & 1u) && better(v[i], lane + 32 * i, bv, bi)) {
                bv = v[i];
                bi = lane + 32 * i;
                bs = s[i];
            }
        DS41_UNROLL
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = shfl_xor(bv, off), os = shfl_xor(bs, off);
            const int oi = shfl_xor(bi, off);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bs = os;
            }
        }
        isel[r] = bi;
        wsel[r] = bs;
        if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
    }
    float sum = 0.0f;
    DS41_UNROLL
    for (int r = 0; r < kTopK; ++r) sum += wsel[r];
    const float denom = sum + kRouteNormEps;                  // weights /= (sum + 1e-20) ...
    if (lane == 0) {
        DS41_UNROLL
        for (int r = 0; r < kTopK; ++r) {
            ids[(size_t) t * kTopK + r] = isel[r];
            weights[(size_t) t * kTopK + r] = fmul_rn(fdiv_rn(wsel[r], denom), kRouteScale);   // ... then * 1.5
        }
    }
}

// ---- host side --------------------------------------------------------------------------------------------------------------------
inline void rt_check_launch(const char* what) {
#if !defined(DS41_EMU)
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 ") + what + ": " + cudaGetErrorString(e));
#else
    (void) what;
#endif
}

template <int NT>
inline void launch_logits_gemv(const float* x, const uint16_t* wg, int T, float* logits, void* stream) {
    const dim3 grid(kExperts, (unsigned) ((T + NT - 1) / NT));
#if defined(DS41_EMU)
    (void) stream;
    ds41_emu::launch(grid, dim3(32 * kRtSliceWarps), 0, [&] { router_logits_gemv_kernel<NT>(x, wg, T, logits); });
#else
    router_logits_gemv_kernel<NT><<<grid, 32 * kRtSliceWarps, 0, (cudaStream_t) stream>>>(x, wg, T, logits);
#endif
    rt_check_launch("router_logits");
}

}  // namespace strata::ds41::cuda::dev

namespace strata::ds41::cuda {

void router_logits(const float* x, const uint16_t* wg_bf16, int T, float* logits, void* stream) {
    if (T < 1) return;
    if (reinterpret_cast<uintptr_t>(x) % 16 != 0 || reinterpret_cast<uintptr_t>(wg_bf16) % 16 != 0 || reinterpret_cast<uintptr_t>(logits) % 16 != 0)
        throw std::invalid_argument("ds41 router_logits: x, wg and logits must be 16-byte aligned");
    if (T > kRouterSmallT) {
        const dim3 grid((unsigned) (kExperts / dev::kRtPfExperts), (unsigned) ((T + dev::kRtPfTokens - 1) / dev::kRtPfTokens));
#if defined(DS41_EMU)
        (void) stream;
        ds41_emu::launch(grid, dim3(32 * dev::kRtSliceWarps), 0, [&] { dev::router_logits_prefill_kernel(x, wg_bf16, T, logits); });
#else
        dev::router_logits_prefill_kernel<<<grid, 32 * dev::kRtSliceWarps, 0, (cudaStream_t) stream>>>(x, wg_bf16, T, logits);
#endif
        dev::rt_check_launch("router_logits(prefill)");
    } else if (T == 1) {
        dev::launch_logits_gemv<1>(x, wg_bf16, T, logits, stream);
    } else if (T == 2) {
        dev::launch_logits_gemv<2>(x, wg_bf16, T, logits, stream);
    } else {
        dev::launch_logits_gemv<4>(x, wg_bf16, T, logits, stream);
    }
}

void router_select(const float* logits, const float* bias, int T, int32_t* ids, float* weights, void* stream) {
    if (T < 1) return;
    const dim3 grid((unsigned) ((T + dev::kRtWarps - 1) / dev::kRtWarps));
#if defined(DS41_EMU)
    (void) stream;
    ds41_emu::launch(grid, dim3(128), 0, [&] { dev::router_select_kernel(logits, bias, T, ids, weights); });
#else
    dev::router_select_kernel<<<grid, 128, 0, (cudaStream_t) stream>>>(logits, bias, T, ids, weights);
#endif
    dev::rt_check_launch("router_select");
}

void router_forward(const float* x, const uint16_t* wg_bf16, const float* bias, int T, float* logits_ws, int32_t* ids, float* weights,
                    void* stream) {
    router_logits(x, wg_bf16, T, logits_ws, stream);
    router_select(logits_ws, bias, T, ids, weights, stream);
}

}  // namespace strata::ds41::cuda
