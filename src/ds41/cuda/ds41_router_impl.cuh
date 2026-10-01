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
//   * router_logits_tiled_kernel (T > 32): a plain shared-memory tiled FP32 GEMM, 64 tokens x 64 experts x 16 per step, 4x4 outputs
//     per thread (prefill: 2 x T x 384 x 5120 FMA; ~5-8 TFLOPS on a V100, i.e. ~2-3 ms per 4096 tokens per layer).  FP32 on purpose:
//     the logits decide the routing, so they are not rounded to fp16 for the tensor cores.
//   * router_select_kernel: one warp per token: 12 experts per lane, 6 rounds of warp argmax with a (value, index) total order.
// Every reduction has a fixed order: the same input gives the same bits, run to run.
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

// ---- logits, large T (prefill): FP32 tiled GEMM ---------------------------------------------------------------------------------
inline constexpr int kTbm = 64, kTbn = 64, kTbk = 16, kTpad = 4;

DS41_KERNEL DS41_LAUNCH_BOUNDS(256) void router_logits_tiled_kernel(const float* DS41_RESTRICT x, const uint16_t* DS41_RESTRICT wg, int T,
                                                                    float* DS41_RESTRICT logits) {
    DS41_SHARED float Xs[2][kTbk][kTbm + kTpad];             // [k][token]
    DS41_SHARED float Ws[2][kTbk][kTbn + kTpad];             // [k][expert]
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * kTbm, n0 = blockIdx.y * kTbn;
    const int lr = tid >> 2, k4 = (tid & 3) * 4;             // this thread's tile row and its 4 consecutive k
    const bool xin = m0 + lr < T;
    const float* xrow = x + (size_t) (m0 + lr) * kHidden;
    const uint16_t* wrow = wg + (size_t) (n0 + lr) * kHidden;

    float acc[4][4];
    DS41_UNROLL
    for (int i = 0; i < 4; ++i)
        DS41_UNROLL
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;

    constexpr int KT = kHidden / kTbk;                        // 320
    float4 xr = xin ? ldgf4(xrow + k4) : make_float4(0.f, 0.f, 0.f, 0.f);
    uint2 wr = ldg2(wrow + k4);
    DS41_UNROLL1
    for (int kt = 0; kt < KT; ++kt) {
        const int buf = kt & 1;
        Xs[buf][k4 + 0][lr] = xr.x;
        Xs[buf][k4 + 1][lr] = xr.y;
        Xs[buf][k4 + 2][lr] = xr.z;
        Xs[buf][k4 + 3][lr] = xr.w;
        Ws[buf][k4 + 0][lr] = u2f(wr.x << 16);
        Ws[buf][k4 + 1][lr] = u2f(wr.x & 0xFFFF0000u);
        Ws[buf][k4 + 2][lr] = u2f(wr.y << 16);
        Ws[buf][k4 + 3][lr] = u2f(wr.y & 0xFFFF0000u);
        sync_block();
        if (kt + 1 < KT) {                                    // the next step's loads fly while this one is multiplied
            const int k0 = (kt + 1) * kTbk + k4;
            xr = xin ? ldgf4(xrow + k0) : make_float4(0.f, 0.f, 0.f, 0.f);
            wr = ldg2(wrow + k0);
        }
        DS41_UNROLL
        for (int kk = 0; kk < kTbk; ++kk) {
            const float4 a = *reinterpret_cast<const float4*>(&Xs[buf][kk][ty * 4]);
            const float4 b = *reinterpret_cast<const float4*>(&Ws[buf][kk][tx * 4]);
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
            DS41_UNROLL
            for (int i = 0; i < 4; ++i)
                DS41_UNROLL
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
    }
    DS41_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m < T)
            *reinterpret_cast<float4*>(logits + (size_t) m * kExperts + n0 + tx * 4) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
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
        const dim3 grid((unsigned) ((T + dev::kTbm - 1) / dev::kTbm), (unsigned) (kExperts / dev::kTbn));
#if defined(DS41_EMU)
        (void) stream;
        ds41_emu::launch(grid, dim3(256), 0, [&] { dev::router_logits_tiled_kernel(x, wg_bf16, T, logits); });
#else
        dev::router_logits_tiled_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(x, wg_bf16, T, logits);
#endif
        dev::rt_check_launch("router_logits(tiled)");
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
