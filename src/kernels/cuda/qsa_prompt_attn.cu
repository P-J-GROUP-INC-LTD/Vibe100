// src/kernels/cuda/qsa_prompt_attn.cu - see include/strata/kernels/qsa_prompt_attn.hpp.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

// The Volta kernel (further down) is written with nvcuda::wmma, which HIP does not have: AMD builds compile it out
// (the host returns false there before anything is launched).
#if !defined(__HIPCC__)
#include <mma.h>
#endif

namespace strata::kernels {
namespace {

constexpr int HD = 256;           // head_dim
constexpr int G = 12;             // query heads per KV head
#ifndef D1_CH
#define D1_CH 32
#endif
constexpr int CH = D1_CH;         // cells per chunk
constexpr int THREADS = 128;      // 4 warps: scores by cell (8 each), p.v by dimension (64 each = one int8 scale group)
constexpr int QS = HD + 8;        // q row stride in halves (bank-conflict-free fragment loads)

// The MMA below needs sm_75 or newer (Turing runs it as two k=8 steps); cp.async needs sm_80. Builds for pre-sm_75
// cards compile the MMA to a trap (that includes Volta, sm_70/72: its m8n8k4 tensor cores have no m16n8k8 / m16n8k16
// form); qsa_prompt_attn_batch never launches these kernels there - Volta has its own WMMA kernel below and every
// older card runs the old FP32 kernel.  Turing compiles cp_async16 to a trap as well and takes the v1 kernel instead
// of launch_i8.
#if defined(__HIPCC__)          // AMD: no mma.sync / cp.async; the host keeps the old kernel (below)
#define STRATA_PA_SM80 0
#elif !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define STRATA_PA_SM80 1
#else
#define STRATA_PA_SM80 0
#endif

// The Volta kernel needs nvcuda::wmma (CUDA only, sm_70 or newer).  STRATA_PA_VOLTA says whether this compilation
// has it at all (AMD: no, the kernel and its launcher are compiled out); STRATA_PA_VOLTA_BODY whether the device
// pass for the current arch gets the real body (the host pass compiles it too, it is never run there; sm_60/61
// device passes get a trap, and qsa_prompt_attn_batch never launches the kernel below sm_70).
#if defined(__HIPCC__)
#define STRATA_PA_VOLTA 0
#else
#define STRATA_PA_VOLTA 1
#endif
#if STRATA_PA_VOLTA && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700)
#define STRATA_PA_VOLTA_BODY 1
#else
#define STRATA_PA_VOLTA_BODY 0
#endif

// m16n8k16 with f16 inputs needs sm_80.  Turing (sm_75) has m16n8k8 with the SAME A/B/C register mapping, so the
// k=16 step is two k=8 steps on the fragments as they are already laid out: a[0]/a[1] are rows gid/gid+8 at k columns
// 2*tig..2*tig+1 (b[0]'s k rows), a[2]/a[3] the same rows at k columns 2*tig+8..2*tig+9 (b[1]'s k rows).  The
// products then add into the same FP32 C registers in the order hi-part-0, hi-part-1, which is the order the k16
// instruction accumulates in as well - but the sum now rounds twice, so the two paths do not agree bit for bit.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if !STRATA_PA_SM80 && (defined(__HIPCC__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 750)
    __trap();   // AMD and pre-Turing builds: no mma.sync (the host keeps the old kernel there)
#elif !STRATA_PA_SM80
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[2]), "r"(a[3]), "r"(b[1]));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// Two int8 codes (low byte first) as an exact half2: 1024 + (c + 128) built in the mantissa, minus 1152.
__device__ __forceinline__ uint32_t i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t pack_h2(float lo_k, float hi_k) {   // element k in the low half
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

// KV_MODE 1: int8 codes + fp16 scale per 64 values. KV_MODE 0: fp16 values (scales 1).
// KV_MODE 3 (hybrid K8V4): K as mode 1, V as mode 0 - the row's q4_0 blocks are dequantized to fp16 at
// gather, so everything downstream of the load is the mode-0 V path; the caller un-rotates the output.
template <int KV_MODE>
struct Smem {
    using KElem = typename std::conditional<KV_MODE == 0, __half, int8_t>::type;
    using VElem = typename std::conditional<KV_MODE == 1, int8_t, __half>::type;
    static constexpr int KROW = KV_MODE == 0 ? HD + 8 : HD + 16;   // elements; 16-byte aligned rows, banks spread
    static constexpr int VROW = KV_MODE == 1 ? HD + 16 : HD + 8;
    __half qh[16][QS];
    __half ql[16][QS];
    KElem k[CH][KROW];
    VElem v[CH][VROW];
    float ks[CH][4];
    float vs[CH][4];
    float s[16][CH + 1];
    float qmax[THREADS / 32];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    long long row[CH];
};

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) prompt_attn_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                              const int32_t* __restrict__ ids,
                                                              const int32_t* __restrict__ steps, int n_kv_heads,
                                                              int page_size, float scale_log2, float* __restrict__ attn,
                                                              int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem<KV_MODE>& S = *reinterpret_cast<Smem<KV_MODE>*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact, and the
    // lo halves stay out of FP16's subnormal range), then split into hi + lo halves
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD, d = i % HD;
        const float x = h < G ? q[(size_t) h * HD + d] * qup : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        if (t < CH) {
            long long r = -1;
            if (t < nh) {
                const int cell = ids[c0 + t];
                const long long page = (long long) p.page_table[cell / page_size];
                r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
            }
            S.row[t] = r;
        }
        __syncthreads();   // rows ready; the previous chunk's p.v is done with k, v, s
        // gather the chunk's K and V rows (16-byte pieces; K8V4's V as q4_0 blocks dequantized to fp16)
        // and their scales
        {
            constexpr int KPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::KElem) / 16;   // per K row
            for (int i = t; i < CH * KPIECES; i += THREADS) {
                const int c = i / KPIECES, pc = i % KPIECES;
                const long long r = S.row[c];
                uint4 kx = make_uint4(0, 0, 0, 0);
                if (r >= 0) {
                    if constexpr (KV_MODE == 0)
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_pool + r * HD) + pc);
                    else   // modes 1 and 3: the K side is INT8
                        kx = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD) + pc);
                }
                *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.k[c][0]) + pc * 16) = kx;
            }
            if constexpr (KV_MODE == 3) {   // V: dequantize the row's q4_0 blocks straight into the fp16 V row
                constexpr int BLKS = HD / QK4_0;
                constexpr int BYTES = BLKS * (int) sizeof(block_q4_0);
                for (int i = t; i < CH * BLKS; i += THREADS) {
                    const int c = i / BLKS, b = i % BLKS;
                    const long long r = S.row[c];
#pragma unroll
                    for (int j = 0; j < QK4_0; ++j) S.v[c][b * QK4_0 + j] = __half(0);
                    if (r >= 0) {
                        const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(p.v_q4 + r * BYTES) + b;
                        const float d = __half2float(__ushort_as_half(blk->d));
#pragma unroll
                        for (int j = 0; j < QK4_0 / 2; ++j) {
                            S.v[c][b * QK4_0 + j] = __float2half_rn((float) ((int)(blk->qs[j] & 0x0F) - 8) * d);
                            S.v[c][b * QK4_0 + j + QK4_0 / 2] =
                                __float2half_rn((float) ((int)(blk->qs[j] >> 4) - 8) * d);
                        }
                    }
                }
            } else {
                constexpr int VPIECES = HD * (int) sizeof(typename Smem<KV_MODE>::VElem) / 16;   // per V row
                for (int i = t; i < CH * VPIECES; i += THREADS) {
                    const int c = i / VPIECES, pc = i % VPIECES;
                    const long long r = S.row[c];
                    uint4 vx = make_uint4(0, 0, 0, 0);
                    if (r >= 0) {
                        if constexpr (KV_MODE == 1)
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_q + r * HD) + pc);
                        else
                            vx = __ldg(reinterpret_cast<const uint4*>(p.v_pool + r * HD) + pc);
                    }
                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(&S.v[c][0]) + pc * 16) = vx;
                }
            }
            for (int i = t; i < CH * 4; i += THREADS) {
                const int c = i / 4, g = i % 4;
                const long long r = S.row[c];
                float a = 0.0f, b = 0.0f;
                if (r >= 0) {
                    if constexpr (KV_MODE == 1) {
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = __half2float(__ushort_as_half(p.v_scale[r * (HD / KV_Q8_GROUP) + g]));
                    } else if constexpr (KV_MODE == 3) {   // K as int8, V dequantized to fp16 (scale 1)
                        a = __half2float(__ushort_as_half(p.k_scale[r * (HD / KV_Q8_GROUP) + g]));
                        b = 1.0f;
                    } else {
                        a = b = 1.0f;
                    }
                }
                S.ks[c][g] = a;
                S.vs[c][g] = b;
            }
        }
        __syncthreads();
        // scores: warp w takes cells 8w..8w+7 (one n-tile) over all 256 dims, per 64-dim scale group
#pragma unroll
        for (int nt = 0; nt < CH / 32; ++nt) {
            const int cb = (warp + 4 * nt) * 8;
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int g = 0; g < 4; ++g) {
                float tg[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    const int k0 = (g * 4 + kk) * 16;
                    uint32_t ah[4], al[4], b[2];
                    ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                    ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                    ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                    ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                    al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                    al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                    al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                    al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                    if constexpr (KV_MODE != 0) {   // modes 1 and 3: the K side is INT8 codes
                        b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig]));
                        b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]));
                    } else {
                        b[0] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig]);
                        b[1] = *reinterpret_cast<const uint32_t*>(&S.k[cb + gid][k0 + 2 * tig + 8]);
                    }
                    mma16816(tg, ah, b);
#ifndef D1_NO_QLO
                    mma16816(tg, al, b);
#endif
                }
                const float s0 = S.ks[cb + 2 * tig][g], s1 = S.ks[cb + 2 * tig + 1][g];
                sc[0] = fmaf(tg[0], s0, sc[0]);
                sc[1] = fmaf(tg[1], s1, sc[1]);
                sc[2] = fmaf(tg[2], s0, sc[2]);
                sc[3] = fmaf(tg[3], s1, sc[3]);
            }
            const int c = cb + 2 * tig;
            S.s[gid][c] = c < nh ? sc[0] * qdown : -INFINITY;
            S.s[gid][c + 1] = c + 1 < nh ? sc[1] * qdown : -INFINITY;
            S.s[gid + 8][c] = c < nh ? sc[2] * qdown : -INFINITY;
            S.s[gid + 8][c + 1] = c + 1 < nh ? sc[3] * qdown : -INFINITY;
        }
        __syncthreads();
        // online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = CH / 8;
            const int r = t >> 3, sub = t & 7;
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) { x[j] = S.s[r][sub * PER + j]; mx = fmaxf(mx, x[j]); }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v: warp w owns dims [64w, 64w+64), which is int8 scale group w. The scale is folded into p relative to
        // the chunk's largest, times 2^14 (p' <= 2^14: its lo half stays out of FP16's subnormal range); the chunk's
        // sum is then added to the running one in FP32 with the factor taken back out
        {
            float vmax = 0.0f;
#pragma unroll
            for (int c = lane; c < CH; c += 32) vmax = fmaxf(vmax, S.vs[c][warp]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.vs[cA][warp] * vup, w1 = S.vs[cA + 1][warp] * vup, w2 = S.vs[cB][warp] * vup,
                            w3 = S.vs[cB + 1][warp] * vup;
                const float p00 = S.s[gid][cA] * w0, p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2, p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    if constexpr (KV_MODE == 1) {
                        const uint32_t x0 = (uint8_t) S.v[cA][d] | ((uint32_t) (uint8_t) S.v[cA + 1][d] << 8);
                        const uint32_t x1 = (uint8_t) S.v[cB][d] | ((uint32_t) (uint8_t) S.v[cB + 1][d] << 8);
                        b[0] = i8x2_to_h2(x0);
                        b[1] = i8x2_to_h2(x1);
                    } else {
                        const __half2 h0 = __halves2half2(S.v[cA][d], S.v[cA + 1][d]);
                        const __half2 h1 = __halves2half2(S.v[cB][d], S.v[cB + 1][d]);
                        b[0] = *reinterpret_cast<const uint32_t*>(&h0);
                        b[1] = *reinterpret_cast<const uint32_t*>(&h1);
                    }
                    mma16816(tmp[j], ah, b);
#ifndef D1_NO_PLO
                    mma16816(tmp[j], al, b);
#endif
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

// ---- v2 (int8 KV): warp w owns dims [64w, 64w+64) for both q.k and p.v, which is also int8 scale group w. So a
// warp needs only its own 64-byte slice of each K and V row: it gathers it itself with cp.async into its own
// double-buffered stage while it computes the previous chunk, and q stays in registers. Only the q.k partial sums
// (one per dim group) cross warps, and they are added in a fixed order: the result is deterministic.
constexpr int CH2 = 32;

struct Smem2 {
    int8_t kv[2][4][2][CH2][64];   // stage, warp, K/V, cell, 64 dims in 16-byte pieces XOR-swizzled by the cell
    float sc[2][4][2][CH2];        // stage, warp, K/V scale of the cell for the warp's group
    float part[4][16][CH2 + 1];    // q.k per dim group
    float p[16][CH2 + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ int swz(int cell, int byte) {   // byte offset of (cell, byte) in a stage slice
    return cell * 64 + ((((byte >> 4) ^ (cell >> 1)) & 3) << 4) + (byte & 15);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
#if !STRATA_PA_SM80
    __trap();
#else
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(valid ? 16 : 0));
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if STRATA_PA_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
__device__ __forceinline__ void cp_async_wait1() {
#if STRATA_PA_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__global__ void __launch_bounds__(THREADS) prompt_attn_i8_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                 const int32_t* __restrict__ ids,
                                                                 const int32_t* __restrict__ steps, int n_kv_heads,
                                                                 int page_size, float scale_log2,
                                                                 float* __restrict__ attn, int cap) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem2& S = *reinterpret_cast<Smem2*>(smem_raw);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int gid = lane >> 2, tig = lane & 3;
    const int dim0 = warp * 64;

    // q: the power-of-two prescale over all 12 heads (as v1), then this warp's 64 dims as hi/lo A fragments
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    uint32_t qh[4][4], ql[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = gid + (r & 1) * 8, col = dim0 + kk * 16 + 2 * tig + (r >> 1) * 8;
            float2 x = make_float2(0.f, 0.f);
            if (row < G) x = *reinterpret_cast<const float2*>(q + (size_t) row * HD + col);
            x.x *= qup;
            x.y *= qup;
            const __half2 hi = __floats2half2_rn(x.x, x.y);
            const float2 hf = __half22float2(hi);
            const __half2 lo = __floats2half2_rn(x.x - hf.x, x.y - hf.y);
            qh[kk][r] = *reinterpret_cast<const uint32_t*>(&hi);
            ql[kk][r] = *reinterpret_cast<const uint32_t*>(&lo);
        }
    }

    // the chunk pipeline: cells two chunks ahead, their pool rows one chunk ahead, the data (cp.async) one ahead
    const int n_chunks = (n + CH2 - 1) / CH2;
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return (page * n_kv_heads + kvh) * page_size + (cell % page_size);
    };
    auto issue = [&](long long r, int st, float& ksr, float& vsr) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + 32 * j, cell = idx >> 2, pc = idx & 3;
            const long long rr = __shfl_sync(0xffffffffu, r, cell);
            const bool ok = rr >= 0;
            const size_t off = ok ? (size_t) rr * HD + dim0 + pc * 16 : 0;
            cp_async16(&S.kv[st][warp][0][0][0] + swz(cell, pc * 16), p.k_q + off, ok);
            cp_async16(&S.kv[st][warp][1][0][0] + swz(cell, pc * 16), p.v_q + off, ok);
        }
        ksr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
        vsr = r >= 0 ? __half2float(__ushort_as_half(__ldg(p.v_scale + r * (HD / KV_Q8_GROUP) + warp))) : 0.0f;
    };
    float ksn, vsn;
    issue(row_of(cell_of(lane)), 0, ksn, vsn);
    cp_async_commit();
    S.sc[0][warp][0][lane] = ksn;
    S.sc[0][warp][1][lane] = vsn;
    long long r_next = row_of(cell_of(CH2 + lane));
    int cell_next2 = cell_of(2 * CH2 + lane);

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    for (int ci = 0; ci < n_chunks; ++ci) {
        const int st = ci & 1, c0 = ci * CH2;
        const bool more = ci + 1 < n_chunks;
        if (more) issue(r_next, st ^ 1, ksn, vsn);
        cp_async_commit();
        r_next = row_of(cell_next2);
        cell_next2 = cell_of((ci + 3) * CH2 + lane);
        cp_async_wait1();
        __syncwarp();
        const int8_t* K = &S.kv[st][warp][0][0][0];
        const int8_t* V = &S.kv[st][warp][1][0][0];
        // q.k over this warp's 64 dims, times the cell's K scale for this group
#pragma unroll
        for (int nt = 0; nt < CH2 / 8; ++nt) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            const int cell = nt * 8 + gid;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t b[2];
                b[0] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig)));
                b[1] = i8x2_to_h2(*reinterpret_cast<const uint16_t*>(K + swz(cell, kk * 16 + 2 * tig + 8)));
                mma16816(tg, qh[kk], b);
                mma16816(tg, ql[kk], b);
            }
            const int c = nt * 8 + 2 * tig;
            const float s0 = S.sc[st][warp][0][c], s1 = S.sc[st][warp][0][c + 1];
            S.part[warp][gid][c] = tg[0] * s0;
            S.part[warp][gid][c + 1] = tg[1] * s1;
            S.part[warp][gid + 8][c] = tg[2] * s0;
            S.part[warp][gid + 8][c + 1] = tg[3] * s1;
        }
        __syncthreads();
        // online softmax over the four groups' sum (fixed order): row t/8, 4 cells per thread
        {
            const int r = t >> 3, sub = t & 7;
            float x[4], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                x[j] = c0 + c < n ? (((S.part[0][r][c] + S.part[1][r][c]) + S.part[2][r][c]) + S.part[3][r][c]) * qdown
                                  : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                S.p[r][sub * 4 + j] = e;
                sum += e;
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();
        // p.v over this warp's 64 dims (as v1)
        {
            float vmax = S.sc[st][warp][1][lane];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
            const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
            float tmp[8][4];
#pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < CH2 / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig, cB = cA + 8;
                const float w0 = S.sc[st][warp][1][cA] * vup, w1 = S.sc[st][warp][1][cA + 1] * vup,
                            w2 = S.sc[st][warp][1][cB] * vup, w3 = S.sc[st][warp][1][cB + 1] * vup;
                const float p00 = S.p[gid][cA] * w0, p01 = S.p[gid][cA + 1] * w1;
                const float p10 = S.p[gid + 8][cA] * w0, p11 = S.p[gid + 8][cA + 1] * w1;
                const float p02 = S.p[gid][cB] * w2, p03 = S.p[gid][cB + 1] * w3;
                const float p12 = S.p[gid + 8][cB] * w2, p13 = S.p[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = pack_h2(p00, p01);
                ah[1] = pack_h2(p10, p11);
                ah[2] = pack_h2(p02, p03);
                ah[3] = pack_h2(p12, p13);
                {
                    const __half2* h = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h[0]); al[0] = pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h[1]); al[1] = pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h[2]); al[2] = pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h[3]); al[3] = pack_h2(p12 - f.x, p13 - f.y);
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = j * 8 + gid;
                    const uint32_t x0 = (uint8_t) V[swz(cA, d)] | ((uint32_t) (uint8_t) V[swz(cA + 1, d)] << 8);
                    const uint32_t x1 = (uint8_t) V[swz(cB, d)] | ((uint32_t) (uint8_t) V[swz(cB + 1, d)] << 8);
                    uint32_t b[2];
                    b[0] = i8x2_to_h2(x0);
                    b[1] = i8x2_to_h2(x1);
                    mma16816(tmp[j], ah, b);
                    mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
        if (more) {
            S.sc[st ^ 1][warp][0][lane] = ksn;
            S.sc[st ^ 1][warp][1][lane] = vsn;
        }
        __syncwarp();
    }
    __syncthreads();
    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = dim0 + j * 8 + 2 * tig;
        *reinterpret_cast<float2*>(attn + (size_t) gid * HD + d) = make_float2(acc[j][0] * i0, acc[j][1] * i0);
        if (gid + 8 < G)
            *reinterpret_cast<float2*>(attn + (size_t) (gid + 8) * HD + d) = make_float2(acc[j][2] * i1, acc[j][3] * i1);
    }
}

bool launch_i8(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
               const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // the shared-memory opt-in is per device (a layer split runs this on several)
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem2);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_i8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_i8_kernel<<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

template <int KV_MODE>
bool launch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
            const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // per device, as above
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(Smem<KV_MODE>);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes) !=
            cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

#if STRATA_PA_VOLTA
// ---- Volta (sm_70/72): prompt_attn_volta_kernel - the v1 kernel's math on nvcuda::wmma ----------------------------
//
// Volta's tensor cores only do m8n8k4 (HMMA.884): the m16n8k8 / m16n8k16 forms the kernels above use do not exist
// there (on sm_70 they are BPT.TRAP stubs).  There are two ways to drive m8n8k4: raw `mma.sync.aligned.m8n8k4` with
// Volta's quad-pair fragment layout (hand-permuted - llama.cpp's mma.cuh shows how much care that takes), or
// nvcuda::wmma m16n16k16, which the compiler turns into 16 HMMA.884 per product and whose fragment loads / stores
// are part of the API.  This kernel was written without a Volta card at hand, so it takes the route whose semantics
// are guaranteed by an API instead of by a layout table: every operand fragment comes from shared memory through
// load_matrix_sync (A row-major; B column-major for K^T and row-major for V - Volta's HMMA has both forms natively,
// neither costs a transpose) and every result leaves through store_matrix_sync.  The ONLY place that depends on
// which (row, col) a fragment element holds is the per-row rescale of O, and that is probed at run time (below),
// not assumed.
//
// Same math as v1, FP32 wherever v1 is FP32: q is split into FP16 hi + lo parts (after the same power-of-two
// prescale, so q.k keeps ~22 bits), K and V enter exactly (fp16 as is; int8 codes are exact in FP16; K8V4's q4_0 V
// is dequantized to fp16 at gather, as in v1), the int8 scales are applied in FP32 (to the q.k partial of each
// 64-dim group: sc = fma(tg[g], ks[g], sc) in the order g = 0..3; folded into p, times 2^14 / the chunk's largest,
// for p.v), p' is split into hi + lo, and the online softmax is v1's (scale_log2, exp2f, running max and sum,
// alpha = exp2(m_old - m_new), acc = fma(acc, alpha, tmp * vdown), out = acc * (1 / l)).  Two deliberate differences.
// (1) The FP32 accumulation INSIDE a tensor-core product is re-associated: Volta's HMMA adds four products per step
// and truncates, so the hi part of each 16-wide k-step gets its own accumulator (summed with fp32 adds) and the lo
// parts one chain of their own - see q.k and p.v below.  On the CPU, with a model of that truncation (round toward
// zero after every 4-product step), the largest absolute error against FP64 at 2,051 cells (outputs up to ~2.7) was
// 4.8e-6 with one chain per score and per product, 1.6-1.8e-6 as written; the old FP32 kernel's arithmetic: 1.1-1.5e-6
// (the parity program allows 4x the old kernel's).  (2) A cell whose page is not resident
// (page table entry < 0, KV streaming) is MASKED (score -inf, weight 0), as the old FP32 kernel does; v1 reads it as
// a zero row with score 0 and lets it dilute the softmax.  The padding of the last chunk (cells >= n) is masked the
// same way, and contributes exactly what v1's masking does: nothing (e = 0, zero K and V rows, zero scales).
//
// Work split (the i8 kernel's): 4 warps, warp w owns dims [64w, 64w + 64) - int8 scale group w - for q.k AND p.v.
//   * q lives in registers: 4 k-steps x (hi, lo) A fragments over the warp's 64 dims, loaded once;
//   * a warp gathers only its own 64-dim slice of the chunk's K and V rows (lane c owns cell c, so the pool row of
//     each piece arrives by __shfl) into its OWN shared tile: no block barrier on the gather;
//   * q.k: 2 n-tiles of 16 cells x 4 k-steps x (hi, lo) -> raw partial sums per group, stored to `part`;
//   * a block barrier, then the softmax for all 16 rows (the 4 pad rows included: they see zero q, finish, and are
//     never written) over the four groups' partials combined in v1's order, which also builds every warp's p'
//     hi/lo tiles (it knows e and the V scales of all four groups);
//   * a second barrier, then p.v per warp: 2 k-steps x 4 n-tiles x (hi, lo), the product of THIS chunk in a fresh
//     fragment `tmp` and the running output rescaled in registers.
//
// O stays in 4 FP32 accumulator fragments per warp for the whole kernel.  The rescale needs alpha PER ROW, and an
// accumulator fragment does not say which row its x[i] holds (nor does the CUDA documentation; it is the hardware's
// layout).  But the layout is a property of the fragment TYPE: load_matrix_sync puts the element at (r, c) of the
// memory tile into the slot the hardware assigns to (r, c), and mma_sync reads and writes the very same slots (C in,
// D out).  So each warp loads one accumulator fragment from a 16 x 16 shared matrix whose entry is its row number,
// and reads x[i] once: rowi[i] is the row of x[i] of EVERY FragC on this device, whatever the layout is.  (The
// alternative - store each chunk's p.v product to shared memory and add it to registers in a layout of our own -
// costs a shared-memory round trip of 4 tiles per warp per chunk; the probe costs 8 loads per chunk.)  Nothing
// else needs element access: Q, K, V and p' are built in shared memory in a layout of our own and loaded through
// the API, the q.k partials are stored with store_matrix_sync and combined from shared memory.
//
// Shared memory (sizeof(SmemV), the same for every KV mode - K and V are always staged as exact FP16): per warp
// one 32 x 72 FP16 tile that holds, in turn, the Q staging (start), K (scores), V (p.v) and the output staging (end);
// the K tile is dead by the first barrier, so V reuses it.  V is only loaded into REGISTERS together with K (its
// global latency hides under the q.k, the barriers and the softmax) and written to the tile after the second barrier.
// ~39 KB: two blocks per SM on a V100 (96 KB), where the carve-out is asked for explicitly by the launcher.
//
// Hazards, per chunk (B = block barrier, W = __syncwarp): ks, vw, valid and the K tile are written by the owning
// warp(s) after B2 of the previous chunk (everyone is done reading them); part is written before B1 and read between
// B1 and B2; p' is written between B1 and B2 and read after B2; the V tile is written after B2 and read by the
// same warp, and a W separates its last read from the next chunk's K write; alpha / lsum / mrow are written between
// B1 and B2 and read after it (alpha) or after the loop (lsum).
constexpr int VCH = 32;                  // cells per chunk: lane c of every warp owns cell c
constexpr int VWARPS = THREADS / 32;     // 4: warp w owns dims [64w, 64w + 64) = int8 scale group w
constexpr int VKLD = 64 + 8;             // halves per tile row: 64 dims + 8 (144-byte rows: 16-byte aligned, no
                                         // bank conflicts)
constexpr int VPLD = VCH + 4;            // floats per row of the q.k partials (wmma: a multiple of 4)
constexpr int VPLH = VCH + 8;            // halves per row of p' (wmma: a multiple of 8)
constexpr int VOLD = 64 + 4;             // floats per row of the output staging
static_assert(THREADS == 128 && VCH == 32 && HD == 4 * 64 && G <= 16,
              "the Volta kernel is written for 4 warps x 64 dims, 32 cells per chunk, 12 (+4 pad) query rows");

struct alignas(32) SmemV {
    __half tile[VWARPS][VCH * VKLD];     // per warp: Q staging, K, V, output staging (all of them <= 32 x 72 halves)
    float part[VWARPS][16 * VPLD];       // per warp: q.k over its 64 dims, hi + lo, before the K scale; then the probe
    __half pp[VWARPS][2][16 * VPLH];     // per warp: p' hi and lo halves, [row][cell]
    float ks[VCH][4];                    // K scale of cell c for group g (0 for a missing cell)
    float vw[VCH][4];                    // V scale of cell c for group g times that group's 2^14 / vmax
    float qmax[VWARPS];
    float alpha[16];
    float lsum[16];
    float mrow[16];
    unsigned valid;                      // bit c: cell c of the chunk exists (inside n, page resident)
};
static_assert(sizeof(SmemV) <= 98304, "the V100 opt-in limit is 96 KB per block");
static_assert(sizeof(float) * 16 * VOLD <= sizeof(__half) * VCH * VKLD && 2 * 16 * VKLD <= VCH * VKLD,
              "Q staging (hi + lo) and the output staging must fit a warp's tile");

// Four int8 codes (low byte first) as two exact half2: byte c ^ 0x80 = c + 128 goes into the mantissa of 1024
// (0x6400 | (c + 128) is the half 1024 + c + 128), minus 1152 is c.  The same trick as i8x2_to_h2, byte-permuted:
// 1 xor, 2 prmt and 2 hsub2 per four codes.
[[maybe_unused]] __device__ __forceinline__ void i8x4_to_h2x2(uint32_t x, uint32_t& lo, uint32_t& hi) {
    x ^= 0x80808080u;
    uint32_t a = __byte_perm(x, 0x64646464u, 0x5140);   // bytes: x0, 0x64, x1, 0x64
    uint32_t b = __byte_perm(x, 0x64646464u, 0x5342);   // bytes: x2, 0x64, x3, 0x64
    uint32_t m = 0x64806480u;                           // half2(1152, 1152)
    const __half2 ha = __hsub2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&m));
    const __half2 hb = __hsub2(*reinterpret_cast<__half2*>(&b), *reinterpret_cast<__half2*>(&m));
    lo = *reinterpret_cast<const uint32_t*>(&ha);
    hi = *reinterpret_cast<const uint32_t*>(&hb);
}

// This warp's 64-dim slice (dims [64*warp, 64*warp + 64)) of the chunk's 32 pool rows, as 16-byte pieces into
// registers: lane l takes pieces l, l + 32, ...; piece i is piece (i % NP) of cell i / NP.  `row` is lane c's pool row
// for cell c (-1: no such cell, zeros).  I8: int8 codes (4 pieces per cell), else fp16 values (8 per cell).
template <bool I8>
__device__ __forceinline__ void vg_load(uint4* x, const void* pool, long long row, int lane, int warp) {
    constexpr int NP = I8 ? 4 : 8, SH = I8 ? 2 : 3;
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        const int idx = lane + 32 * j, cell = idx >> SH, pc = idx & (NP - 1);
        const long long rr = __shfl_sync(0xffffffffu, row, cell);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (rr >= 0)
            v = __ldg(reinterpret_cast<const uint4*>(static_cast<const char*>(pool) +
                                                     (rr * HD + warp * 64) * (I8 ? 1 : 2) + pc * 16));
        x[j] = v;
    }
}
// ... and those pieces into the warp's tile as FP16: fp16 as is, int8 codes expanded exactly.
template <bool I8>
__device__ __forceinline__ void vg_store(__half* tile, const uint4* x, int lane) {
    constexpr int NP = I8 ? 4 : 8, SH = I8 ? 2 : 3;
#pragma unroll
    for (int j = 0; j < NP; ++j) {
        const int idx = lane + 32 * j, cell = idx >> SH, pc = idx & (NP - 1);
        if constexpr (!I8) {
            *reinterpret_cast<uint4*>(tile + cell * VKLD + pc * 8) = x[j];
        } else {
            uint4 a, b;
            i8x4_to_h2x2(x[j].x, a.x, a.y);
            i8x4_to_h2x2(x[j].y, a.z, a.w);
            i8x4_to_h2x2(x[j].z, b.x, b.y);
            i8x4_to_h2x2(x[j].w, b.z, b.w);
            *reinterpret_cast<uint4*>(tile + cell * VKLD + pc * 16) = a;
            *reinterpret_cast<uint4*>(tile + cell * VKLD + pc * 16 + 8) = b;
        }
    }
}
// K8V4's V: the warp's slice of a row is two q4_0 blocks (18 bytes each, 2-byte aligned: nine 16-bit words per
// block, word 0 the fp16 scale).  Lane l takes (cell, block) pairs l and l + 32: cell = pair / 2.
[[maybe_unused]] __device__ __forceinline__ void vg_load_q4(uint32_t (&w)[2][9], const uint8_t* v_q4, long long row,
                                                            int lane, int warp) {
    constexpr int BYTES = (HD / QK4_0) * (int) sizeof(block_q4_0);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int idx = lane + 32 * j, cell = idx >> 1, b = idx & 1;
        const long long rr = __shfl_sync(0xffffffffu, row, cell);
        const uint16_t* bp =
            reinterpret_cast<const uint16_t*>(v_q4 + rr * BYTES + (2 * warp + b) * (int) sizeof(block_q4_0));
#pragma unroll
        for (int k = 0; k < 9; ++k) w[j][k] = rr >= 0 ? (uint32_t) __ldg(bp + k) : 0u;
    }
}
// ... dequantized as v1 does: fp16((nibble - 8) * d), element j of a block in the low nibble of byte j, j + 16 in
// the high one
[[maybe_unused]] __device__ __forceinline__ void vg_store_q4(__half* tile, const uint32_t (&w)[2][9], int lane) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int idx = lane + 32 * j, cell = idx >> 1, b = idx & 1;
        const float d = __half2float(__ushort_as_half((unsigned short) w[j][0]));
#pragma unroll
        for (int m = 0; m < 4; ++m) {   // 8 halves each: dims 0-7, 8-15 from the low nibbles; 16-23, 24-31 high ones
            const int sh = (m >> 1) * 4, w0 = 1 + (m & 1) * 4;
            uint32_t o[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t wd = w[j][w0 + k];   // two qs bytes: the elements 2k and 2k + 1 of this piece
                o[k] = pack_h2((float) ((int) ((wd >> sh) & 0xFu) - 8) * d,
                               (float) ((int) ((wd >> (8 + sh)) & 0xFu) - 8) * d);
            }
            *reinterpret_cast<uint4*>(tile + cell * VKLD + b * 32 + m * 8) = make_uint4(o[0], o[1], o[2], o[3]);
        }
    }
}

template <int KV_MODE>
__global__ void __launch_bounds__(THREADS, 2) prompt_attn_volta_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                       const int32_t* __restrict__ ids,
                                                                       const int32_t* __restrict__ steps,
                                                                       int n_kv_heads, int page_size, float scale_log2,
                                                                       float* __restrict__ attn, int cap) {
#if STRATA_PA_VOLTA_BODY
    using namespace nvcuda;
    typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> FragA;
    // K^T: B(dim, cell) = K[cell][dim], column-major; V: B(cell, dim) = V[cell][dim], row-major
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> FragBk;
    typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> FragBv;
    typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> FragC;
    constexpr bool K_I8 = KV_MODE != 0;       // modes 1 and 3: int8 K codes
    constexpr bool V_I8 = KV_MODE == 1;       // int8 V codes
    constexpr bool V_Q4 = KV_MODE == 3;       // q4_0 V blocks (K8V4)
    constexpr int NPK = K_I8 ? 4 : 8;         // 16-byte pieces of K per lane per chunk
    constexpr int NPV = V_I8 ? 4 : 8;

    extern __shared__ __align__(128) unsigned char smem_volta[];
    SmemV& S = *reinterpret_cast<SmemV*>(smem_volta);
    const int qi = blockIdx.x, kvh = blockIdx.y;
    const int n_head = n_kv_heads * G;
    q += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    attn += (size_t) qi * n_head * HD + (size_t) kvh * G * HD;
    ids += (size_t) qi * cap;
    const int n = __ldg(steps + (size_t) qi * kStepCount + kStepWidth);
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    __half* const tile = S.tile[warp];
    float* const part = S.part[warp];

    // q: 12 heads + 4 zero rows, scaled by a power of two that puts its largest value near 2^14 (exact), then split
    // into hi + lo halves - as v1.  The maximum is over all 12 heads (a block-wide quantity), then each warp stages
    // its own 64 dims (hi at tile row 0, lo at row 16) and loads them as A fragments, which stay in registers.
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o));
    if (lane == 0) S.qmax[warp] = qm;
    if (t < 16) { S.mrow[t] = -INFINITY; S.lsum[t] = 0.0f; }
    // the row probe: a 16 x 16 matrix whose entry is its row number, loaded as an accumulator fragment
    int rowi[8];
    {
        for (int e = lane; e < 256; e += 32) part[(e >> 4) * VPLD + (e & 15)] = (float) (e >> 4);
        __syncwarp();
        FragC probe;
        wmma::load_matrix_sync(probe, part, VPLD, wmma::mem_row_major);
#pragma unroll
        for (int e = 0; e < 8; ++e) rowi[e] = (int) probe.x[e];
        __syncwarp();
    }
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);                 // qm < 2^qe
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale_log2, qe - 14);
    FragA qh[4], ql[4];
    {
        __half* const sh = tile;                    // hi: rows 0..15 of the tile, lo: rows 16..31
        __half* const sl = tile + 16 * VKLD;
#pragma unroll
        for (int i = 0; i < 32; ++i) {              // 16 rows x 64 dims, 32 dims (one row half) per sweep
            const int row = i >> 1, d = (i & 1) * 32 + lane;
            const float x = row < G ? q[(size_t) row * HD + warp * 64 + d] * qup : 0.0f;
            const __half hi = __float2half_rn(x);
            sh[row * VKLD + d] = hi;
            sl[row * VKLD + d] = __float2half_rn(x - __half2float(hi));
        }
        __syncwarp();
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma::load_matrix_sync(qh[kk], sh + kk * 16, VKLD);
            wmma::load_matrix_sync(ql[kk], sl + kk * 16, VKLD);
        }
        __syncwarp();                               // the tile is free for K
    }

    FragC acc[4];                                   // O[16 rows][this warp's 64 dims]
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[j], 0.0f);

    // the chunk pipeline: lane c resolves the pool row of cell c; the cells are read two chunks ahead and their rows
    // one chunk ahead, so the two dependent loads (ids, page table) are off the critical path
    auto cell_of = [&](int c) -> int { return c < n ? __ldg(ids + c) : -1; };
    auto row_of = [&](int cell) -> long long {
        if (cell < 0) return -1;
        const long long page = (long long) __ldg(p.page_table + cell / page_size);
        return page < 0 ? -1 : (page * n_kv_heads + kvh) * page_size + (cell % page_size);   // no page: masked
    };
    long long r_cur = row_of(cell_of(lane));
    int cell_nx = cell_of(VCH + lane);

    for (int c0 = 0; c0 < n; c0 += VCH) {
        const long long r_nx = row_of(cell_nx);     // the next chunk's rows: their loads fly during this chunk
        cell_nx = cell_of(c0 + 2 * VCH + lane);
        // --- gather: K and V of this warp's 64 dims into registers (global latency starts here), the scales
        uint4 kx[NPK];
        uint4 vx[V_Q4 ? 1 : NPV];
        uint32_t vq[2][9];
        if constexpr (K_I8) vg_load<true>(kx, p.k_q, r_cur, lane, warp);
        else vg_load<false>(kx, p.k_pool, r_cur, lane, warp);
        if constexpr (V_Q4) vg_load_q4(vq, p.v_q4, r_cur, lane, warp);
        else if constexpr (V_I8) vg_load<true>(vx, p.v_q, r_cur, lane, warp);
        else vg_load<false>(vx, p.v_pool, r_cur, lane, warp);
        float ksf = 0.0f, vsf = 0.0f;               // this cell's scales for this warp's group
        if (r_cur >= 0) {
            if constexpr (KV_MODE == 1) {
                ksf = __half2float(__ushort_as_half(__ldg(p.k_scale + r_cur * (HD / KV_Q8_GROUP) + warp)));
                vsf = __half2float(__ushort_as_half(__ldg(p.v_scale + r_cur * (HD / KV_Q8_GROUP) + warp)));
            } else if constexpr (KV_MODE == 3) {    // K as int8, V dequantized to fp16 (scale 1)
                ksf = __half2float(__ushort_as_half(__ldg(p.k_scale + r_cur * (HD / KV_Q8_GROUP) + warp)));
                vsf = 1.0f;
            } else {
                ksf = vsf = 1.0f;
            }
        }
        // p.v's scale handling: the V scale folded into p relative to the chunk's largest, times 2^14 (as v1)
        float vmax = vsf;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, o));
        const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
        S.ks[lane][warp] = ksf;
        S.vw[lane][warp] = vsf * vup;
        const unsigned vmask = __ballot_sync(0xffffffffu, r_cur >= 0);
        if (t == 0) S.valid = vmask;
        if constexpr (K_I8) vg_store<true>(tile, kx, lane);
        else vg_store<false>(tile, kx, lane);
        __syncwarp();                               // the warp's K tile is complete

        // --- q.k over this warp's 64 dims: raw partial sums (hi + lo, before the K scale).  Volta's HMMA adds only
        // FOUR products per step and truncates its fp32 result, so a 32-step chain over all 64 dims would lose far more
        // than the 16-step-per-instruction chains of Turing / Ampere.  Each k-step of 16 dims therefore gets a fresh
        // 4-step accumulator for its hi part, summed into the group's total with ordinary fp32 adds (element-wise
        // on same-type fragments: no layout knowledge needed), and the lo parts (2^-11 smaller, so their own
        // truncation is irrelevant) form one chain of their own, added last.
#pragma unroll
        for (int nt = 0; nt < VCH / 16; ++nt) {
            FragC tg, tl, th;
            wmma::fill_fragment(tl, 0.0f);
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                FragBk b;
                wmma::load_matrix_sync(b, tile + nt * 16 * VKLD + kk * 16, VKLD);
                wmma::fill_fragment(th, 0.0f);
                wmma::mma_sync(th, qh[kk], b, th);
                wmma::mma_sync(tl, ql[kk], b, tl);
                if (kk == 0) {
                    tg = th;
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) tg.x[e] += th.x[e];
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) tg.x[e] += tl.x[e];
            wmma::store_matrix_sync(part + nt * 16, tg, VPLD, wmma::mem_row_major);
        }
        __syncthreads();   // B1: every warp's partials, scales and the validity mask are in shared memory

        // --- online softmax: row t/8, 4 cells per thread, 8 threads per row (lanes 8r..8r+7 of a warp)
        {
            constexpr int PER = VCH / 8;
            static_assert(PER == 4, "float4 accesses below");
            const int r = t >> 3, sub = t & 7;
            const unsigned vm = S.valid;
            float sc[PER] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int g = 0; g < 4; ++g) {           // v1's order: sc = fma(tg[g], ks[g], sc), g = 0..3
                const float4 pg = *reinterpret_cast<const float4*>(&S.part[g][r * VPLD + sub * PER]);
                const float pj[PER] = {pg.x, pg.y, pg.z, pg.w};
#pragma unroll
                for (int j = 0; j < PER; ++j) sc[j] = fmaf(pj[j], S.ks[sub * PER + j][g], sc[j]);
            }
            float x[PER], mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = ((vm >> (sub * PER + j)) & 1u) ? sc[j] * qdown : -INFINITY;
                mx = fmaxf(mx, x[j]);
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float e[PER], sum = 0.0f;
#pragma unroll
            for (int j = 0; j < PER; ++j) {
                e[j] = x[j] == -INFINITY ? 0.0f : exp2f(x[j] - m_new);
                sum += e[j];
            }
#pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            // p' = e * (V scale * 2^14 / vmax) of each group, split into hi + lo halves, for all four warps' p.v
            float4 vwc[PER];
#pragma unroll
            for (int j = 0; j < PER; ++j) vwc[j] = *reinterpret_cast<const float4*>(S.vw[sub * PER + j]);
#pragma unroll
            for (int w = 0; w < VWARPS; ++w) {
                uint32_t ph[PER / 2], pl[PER / 2];
#pragma unroll
                for (int j = 0; j < PER; j += 2) {
                    const float wa = w == 0 ? vwc[j].x : w == 1 ? vwc[j].y : w == 2 ? vwc[j].z : vwc[j].w;
                    const float wb = w == 0   ? vwc[j + 1].x
                                     : w == 1 ? vwc[j + 1].y
                                     : w == 2 ? vwc[j + 1].z : vwc[j + 1].w;
                    const float pa = e[j] * wa, pb = e[j + 1] * wb;
                    const __half2 h = __floats2half2_rn(pa, pb);
                    const float2 f = __half22float2(h);
                    ph[j / 2] = *reinterpret_cast<const uint32_t*>(&h);
                    pl[j / 2] = pack_h2(pa - f.x, pb - f.y);
                }
                *reinterpret_cast<uint2*>(&S.pp[w][0][r * VPLH + sub * PER]) = make_uint2(ph[0], ph[1]);
                *reinterpret_cast<uint2*>(&S.pp[w][1][r * VPLH + sub * PER]) = make_uint2(pl[0], pl[1]);
            }
            __syncwarp();
            if (sub == 0) {
                const float a = m_old == -INFINITY ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();   // B2: p' and alpha are ready; everyone is done with part, ks, vw, valid

        // --- p.v over this warp's 64 dims: V into the tile that held K, then 2 k-steps (16 cells) x 4 n-tiles x
        // (hi, lo).
        // Accumulated as q.k is (Volta truncates every 4-product HMMA step): the hi part of each k-step in a fresh
        // accumulator, summed with fp32 adds; the (tiny) lo parts in one chain of their own, added last.
        if constexpr (V_Q4) vg_store_q4(tile, vq, lane);
        else if constexpr (V_I8) vg_store<true>(tile, vx, lane);
        else vg_store<false>(tile, vx, lane);
        __syncwarp();
        FragA ah[VCH / 16], al[VCH / 16];
#pragma unroll
        for (int ks = 0; ks < VCH / 16; ++ks) {
            wmma::load_matrix_sync(ah[ks], &S.pp[warp][0][ks * 16], VPLH);
            wmma::load_matrix_sync(al[ks], &S.pp[warp][1][ks * 16], VPLH);
        }
        float am[8];                                // alpha of the row of fragment element e (the row: from the probe)
#pragma unroll
        for (int e = 0; e < 8; ++e) am[e] = S.alpha[rowi[e]];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            FragC tmp, th, tl;
            wmma::fill_fragment(tl, 0.0f);
#pragma unroll
            for (int ks = 0; ks < VCH / 16; ++ks) {
                FragBv b;
                wmma::load_matrix_sync(b, tile + ks * 16 * VKLD + j * 16, VKLD);
                wmma::fill_fragment(th, 0.0f);
                wmma::mma_sync(th, ah[ks], b, th);
                wmma::mma_sync(tl, al[ks], b, tl);
                if (ks == 0) {
                    tmp = th;
                } else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) tmp.x[e] += th.x[e];
                }
            }
            // acc = fma(acc, alpha[row], tmp * vdown)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j].x[e] = fmaf(acc[j].x[e], am[e], (tmp.x[e] + tl.x[e]) * vdown);
        }
        __syncwarp();      // this warp's reads of the V tile are done before the next chunk's K goes in
        r_cur = r_nx;
    }

    // --- out = acc * (1 / l) for the 12 real rows (the pad rows 12..15 are never written), through the warp's own
    // tile, row by row.  lsum needs no barrier: the last chunk's B2 (or, with no chunk, the first barrier) is behind
    // us.
    float* const stage = reinterpret_cast<float*>(tile);
#pragma unroll
    for (int j = 0; j < 4; ++j) wmma::store_matrix_sync(stage + j * 16, acc[j], VOLD, wmma::mem_row_major);
    __syncwarp();
#pragma unroll
    for (int row = 0; row < G; ++row) {
        const float l = S.lsum[row];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        const float2 v = *reinterpret_cast<const float2*>(stage + row * VOLD + 2 * lane);
        *reinterpret_cast<float2*>(attn + (size_t) row * HD + warp * 64 + 2 * lane) = make_float2(v.x * inv, v.y * inv);
    }
#else
    __trap();   // pre-Volta device pass: qsa_prompt_attn_batch never launches this below sm_70
#endif
}

template <int KV_MODE>
bool launch_volta(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps, int64_t cap,
                  const QsaShapes& s, float* attn, int64_t n_q, cudaStream_t st) {
    static bool attr[64] = {};   // per device, as launch<>
    int dev = 0;
    cudaGetDevice(&dev);
    const int bytes = (int) sizeof(SmemV);
    if (dev < 0 || dev >= 64) return false;
    if (!attr[dev]) {
        if (cudaFuncSetAttribute(prompt_attn_volta_kernel<KV_MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 bytes) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        // two ~39 KB blocks per SM need the 96 KB shared-memory carve-out; ask for it (a hint: failure is harmless)
        cudaFuncSetAttribute(prompt_attn_volta_kernel<KV_MODE>, cudaFuncAttributePreferredSharedMemoryCarveout,
                             (int) cudaSharedmemCarveoutMaxShared);
        cudaGetLastError();
        attr[dev] = true;
    }
    const float scale_log2 = 1.4426950408889634f / sqrtf((float) HD);
    for (int64_t q0 = 0; q0 < n_q; q0 += 65535) {
        const int64_t nb = n_q - q0 < 65535 ? n_q - q0 : 65535;
        prompt_attn_volta_kernel<KV_MODE><<<dim3((unsigned) nb, (unsigned) s.n_head_kv), THREADS, bytes, st>>>(
            q + q0 * s.n_head * HD, pools, ids + q0 * cap, steps + q0 * kStepCount, (int) s.n_head_kv,
            (int) s.page_size, scale_log2, attn + q0 * s.n_head * HD, (int) cap);
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_prompt_attn_batch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}
#endif  // STRATA_PA_VOLTA

}  // namespace

namespace {

// Which implementation runs for these pools on the CURRENT device.  qsa_prompt_attn_batch and
// qsa_prompt_attn_variant both go through select_impl, so a test can say what ran without guessing.
enum class PaImpl {
    None,        // false: the caller keeps the old FP32 kernel (qsa_decode_attn_batch)
    I8CpAsync,   // sm_80+, int8 KV: prompt_attn_i8_kernel (cp.async, q in registers)
    V1,          // sm_75+: prompt_attn_kernel<KV_MODE> (m16n8k8 / m16n8k16 mma.sync)
    Volta,       // sm_70/72: prompt_attn_volta_kernel<KV_MODE> (nvcuda::wmma m16n16k16)
};

// major * 10 + minor of the current device, cached per device (a layer split can mix architectures); 0 when it
// cannot be read.  The minor matters: Volta (7.0, 7.2) and Turing (7.5) share a major version, and the Turing
// kernels are BPT.TRAP stubs in the sm_70 code.
int current_cc() {
    static int cc[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return 0; }
    if (cc[dev] == 0) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
#if !defined(__HIPCC__)   // (the HIP compat header has no minor attribute; select_impl refuses AMD devices anyway)
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
#endif
        cc[dev] = major * 10 + minor;
    }
    return cc[dev];
}

// STRATA_VOLTA_ATTN, read once: unset / 1 = on (sm_70/72 run the WMMA kernel), 0 = off (they return false and the
// caller's FP32 kernel runs: the pre-port behaviour minus the crash, and the reference the Volta kernel is checked
// against), 2 / "force" = the WMMA kernel on ANY sm_70+ card (a test aid for a dev box without a V100; the default
// behaviour of every other architecture is untouched without it).
int volta_attn_mode() {
    static const int mode = [] {
        const char* e = std::getenv("STRATA_VOLTA_ATTN");
        if (e == nullptr || *e == 0) return 1;
        if (std::strcmp(e, "0") == 0) return 0;
        if (std::strcmp(e, "2") == 0 || std::strcmp(e, "force") == 0) return 2;
        return 1;
    }();
    return mode;
}

PaImpl select_impl(const QsaAttnPools& pools, const QsaShapes& s, int& kv_mode) {
    kv_mode = 0;
    const int cc = current_cc();
    if (cc < 70) return PaImpl::None;       // Pascal and older, or a device that cannot be queried: the old kernel
#if defined(__HIPCC__)
    return PaImpl::None;   // the tensor-core kernels are compiled out on AMD (its major version is not a CUDA sm)
#endif
    if (pools.k_q4 != nullptr || s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv) return PaImpl::None;
    if (pools.k_q != nullptr && pools.v_q4 != nullptr) {   // hybrid K8V4: int8 K + dequantized-q4 V
        if (!pools.k_scale) return PaImpl::None;
        kv_mode = 3;
    } else if (pools.k_q != nullptr) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale) return PaImpl::None;
        kv_mode = 1;
    } else {
        if (!pools.k_pool || !pools.v_pool) return PaImpl::None;
        kv_mode = 0;
    }
    // sm_70/72: no m16n8k8 / m16n8k16, so none of the kernels above exist there (they are traps); the WMMA kernel
    // does, unless STRATA_VOLTA_ATTN=0 asks for the FP32 fallback.
    const int volta = volta_attn_mode();
    if (cc < 75) return volta != 0 ? PaImpl::Volta : PaImpl::None;
    if (volta == 2) return PaImpl::Volta;   // STRATA_VOLTA_ATTN=force
    // sm_75 or newer: the MMA above compiles for both.  sm_80+ runs the cp.async kernel (launch_i8); Turing has
    // no cp.async, so it runs the v1 kernel (launch<1>, same accuracy, another summation order).
    if (cc < 80) return PaImpl::V1;
    if (kv_mode == 1) {
        // STRATA_PROMPT_ATTN_V1=1 (debug): the first version, same accuracy, another summation order - the control
        // for how far the model amplifies an FP32-level change.  Turing always takes it: v2's cp.async does not
        // exist before sm_80.  (No effect on Volta, where the v1 kernel does not exist.)
        static const bool v1 = std::getenv("STRATA_PROMPT_ATTN_V1") != nullptr;
        return v1 ? PaImpl::V1 : PaImpl::I8CpAsync;
    }
    return PaImpl::V1;
}

}  // namespace

const char* qsa_prompt_attn_variant(const QsaAttnPools& pools, const QsaShapes& s) {
    int kv_mode = 0;
    switch (select_impl(pools, s, kv_mode)) {
        case PaImpl::Volta: return "volta-wmma";
        case PaImpl::I8CpAsync: return "ampere-i8-cpasync";
        case PaImpl::V1: return "mma-v1";
        case PaImpl::None: break;
    }
    return "fallback-fp32";
}

bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    int kv_mode = 0;
    const PaImpl impl = select_impl(pools, s, kv_mode);
    if (impl == PaImpl::None || cap <= 0 || !ids || !steps || !pools.page_table) return false;
    cudaStream_t st = (cudaStream_t) stream;
    switch (impl) {
        case PaImpl::Volta:
#if STRATA_PA_VOLTA
            if (kv_mode == 3) return launch_volta<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
            if (kv_mode == 1) return launch_volta<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
            return launch_volta<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
#else
            return false;
#endif
        case PaImpl::I8CpAsync: return launch_i8(q, pools, ids, steps, cap, s, attn, n_q, st);
        case PaImpl::V1:
            if (kv_mode == 3) return launch<3>(q, pools, ids, steps, cap, s, attn, n_q, st);
            if (kv_mode == 1) return launch<1>(q, pools, ids, steps, cap, s, attn, n_q, st);
            return launch<0>(q, pools, ids, steps, cap, s, attn, n_q, st);
        case PaImpl::None: break;
    }
    return false;
}

}  // namespace strata::kernels
