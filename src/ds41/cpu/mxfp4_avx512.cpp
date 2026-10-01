// src/ds41/cpu/mxfp4_avx512.cpp - DS-C: the MXFP4 dot product on AVX-512 + VNNI (Cascade Lake: F/BW/VL/DQ/VNNI, NO VBMI).
//
// Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mfma -mf16c and nothing else; the portable
// file picks it at run time (CPUID).  Read include/strata/ds41/cpu/mxfp4_expert.hpp for the method; this file is the
// instruction-level story.
//
// ONE GROUP (4 blocks, 68 bytes, 128 weights) PER STEP, for each of NR weight rows at once and for each of NT tokens:
//
//   weights   4 x `vmovdqu xmm` + 3 x `vinserti32x4` put the four 16-byte `qs` fields into the four 128-bit lanes of one
//             zmm (the blocks are 17 bytes apart, so nothing wider than a 16-byte load lines up).  `vpandd` and
//             `vpsrlw` + `vpandd` split the nibbles, two `vpshufb` look all 128 codes up at once in the table
//             kvalues + 12 (unsigned, 0..24): lane k of `wlo` holds block k's values 0..15, `whi` its values 16..31.
//   exponents the four E8M0 bytes of the group sit at byte offsets 0, 17, 34, 51, i.e. at position k of 128-bit lane k of
//             the SAME 64 bytes loaded from the group start; one `vpshufb` with a constant control spreads them to
//             the four dwords of their lane.  2^(e-128) is then `vscalefps(1.0, e - 128)`: exact for every e,
//             denormals (e = 0, 1) and 2^127 (e = 255) included, with no branch and no table.
//   products  `vpdpbusd` multiplies 64 unsigned weight bytes by 64 signed activation bytes and adds groups of four into
//             16 int32 lanes: lane 4k + l is a partial sum of block k.  It is chained twice (low halves, then high
//             halves) onto the activation's own correction vector -12 * sum(x) (the weights are offset by +12 to be
//             unsigned; ActQ::corr removes that exactly), so the int32 lanes come out as the true signed partial sums.
//   scale     cvtdq2ps, times the (e8m0 * activation scale) vector, accumulated with an FMA into one float vector per
//             (row, token) that lives for the whole row and is reduced once at the end.
//
// Per row and group that is ~17 vector instructions of weight decoding shared by all NT tokens plus 5 per token (two
// `vpdpbusd`, the convert, the scale multiply, the FMA): measured with the weights in L2 on the dev VM (one core,
// ~2.8 GHz Cascade Lake class) it consumes ~21 GB/s of weights at T = 1, 13 at T = 2, 10 at T = 3, 8 at T = 4 - several
// times what one core can pull from DRAM when twelve of them share a socket's ~50 GB/s, so the kernel is meant to sit
// at the memory limit, not the ALU limit, for T <= 4.  Tiles of NR rows share each activation load (see dot_rows_avx512).
//
// PREFETCH: every row is its own sequential stream; besides the hardware prefetcher each step touches the two cache
// lines `pf` bytes ahead of the group it is decoding (a group is 68 bytes, so one prefetch per step would skip a line
// every 16 steps), clamped to the end of the call's range, with the first `pf` bytes of the range prefetched in a burst
// (see mxfp4_internal.hpp).  On the dev VM a distance of 6-12 KB lifted one-core streaming from 7.6 GB/s (1 KB) to ~10
// (none: 6): a core's own line-fill buffers, not the DRAM, are what limits a single stream.  The distance is a
// run-time knob (set_prefetch_bytes) because the Xeon will want its own value; the benchmark's --sweep-pf finds it.
//
// Attribution: the activation quantiser at the end follows ggml's x86 SIMD quantize_row_q8_0 (ggml-cpu/arch/x86/quants.c) - llama.cpp, MIT License,
// Copyright (c) 2023-2026 The ggml authors (notice: src/ds41/cuda/ds41_math.cuh, third_party/ggml/LICENSE).
#include "mxfp4_internal.hpp"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <algorithm>

namespace strata::ds41::cpu::detail {
namespace {

constexpr int kGroupBytes = kGroupBlocks * kBlockBytes;   // 68

alignas(16) const uint8_t kTableU[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};   // kvalues_fp4 + 12
// lane k: the four dwords [k, 0x80, 0x80, 0x80]: byte k of the lane, zero-extended
#define DS41_CTRL_LANE(k) (int8_t)(k), -128, -128, -128, (int8_t)(k), -128, -128, -128, (int8_t)(k), -128, -128, -128, (int8_t)(k), -128, -128, -128
alignas(64) const int8_t kCtrlE[64] = {DS41_CTRL_LANE(0), DS41_CTRL_LANE(1), DS41_CTRL_LANE(2), DS41_CTRL_LANE(3)};
#undef DS41_CTRL_LANE

struct Consts {
    __m512i table, m0f, ctrl_e, c128;
    __m512 one;
};

inline Consts make_consts() {
    Consts c;
    c.table = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i*) kTableU));
    c.m0f = _mm512_set1_epi8(0x0F);
    c.ctrl_e = _mm512_load_si512((const void*) kCtrlE);
    c.c128 = _mm512_set1_epi32(128);
    c.one = _mm512_set1_ps(1.0f);
    return c;
}

struct RowDec {
    __m512i wlo, whi;   // unsigned weight bytes (kvalues + 12), lane k = block k
    __m512 ws;          // 2^(e_k - 128) in the four dword lanes of block k
};

inline RowDec decode_group(const uint8_t* p, const Consts& c) {
    __m512i z = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i*) (p + 1)));
    z = _mm512_inserti32x4(z, _mm_loadu_si128((const __m128i*) (p + 18)), 1);
    z = _mm512_inserti32x4(z, _mm_loadu_si128((const __m128i*) (p + 35)), 2);
    z = _mm512_inserti32x4(z, _mm_loadu_si128((const __m128i*) (p + 52)), 3);
    RowDec d;
    d.wlo = _mm512_shuffle_epi8(c.table, _mm512_and_si512(z, c.m0f));
    d.whi = _mm512_shuffle_epi8(c.table, _mm512_and_si512(_mm512_srli_epi16(z, 4), c.m0f));
    const __m512i raw = _mm512_loadu_si512((const void*) p);
    const __m512i e = _mm512_shuffle_epi8(raw, c.ctrl_e);
    d.ws = _mm512_scalef_ps(c.one, _mm512_cvtepi32_ps(_mm512_sub_epi32(e, c.c128)));
    return d;
}

// NR rows x NT tokens.  `rows[r]` points at row r's first block; `out` at the (token 0, first row) element.
template <int NR, int NT>
inline __attribute__((always_inline)) void tile(const uint8_t* const* rows, const uint8_t* const* lim, int ng,
                                                const ActQ* x, float* out, size_t out_stride, bool accumulate, int pf) {
    const Consts c = make_consts();
    __m512 acc[NR][NT];
#pragma GCC unroll 8
    for (int r = 0; r < NR; ++r)
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) acc[r][t] = _mm512_setzero_ps();

    for (int g = 0; g < ng; ++g) {
        RowDec d[NR];
#pragma GCC unroll 8
        for (int r = 0; r < NR; ++r) {
            const uint8_t* p = rows[r] + (size_t) g * kGroupBytes;
            if (pf) prefetch_ahead(p, pf, lim[r]);
            d[r] = decode_group(p, c);
        }
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) {
            const ActQ& a = x[t];
            const __m512i xlo = _mm512_load_si512((const void*) (a.q + g * kGroupValues));
            const __m512i xhi = _mm512_load_si512((const void*) (a.q + g * kGroupValues + 64));
            const __m512i corr = _mm512_load_si512((const void*) (a.corr + g * 16));
            const __m512 sc = _mm512_load_ps(a.sc4 + g * 16);
#pragma GCC unroll 8
            for (int r = 0; r < NR; ++r) {
                __m512i s = _mm512_dpbusd_epi32(corr, d[r].wlo, xlo);
                s = _mm512_dpbusd_epi32(s, d[r].whi, xhi);
                acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_mul_ps(d[r].ws, sc), acc[r][t]);
            }
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t)
#pragma GCC unroll 8
        for (int r = 0; r < NR; ++r) {
            const float v = _mm512_reduce_add_ps(acc[r][t]);
            float& o = out[(size_t) t * out_stride + r];
            o = accumulate ? o + v : v;
        }
}

template <int NR, int NT>
void rows_nt(const uint8_t* w, size_t stride, int ng, int nrows, const ActQ* x, float* out, size_t os, bool accumulate,
             int pf) {
    const size_t row_bytes = (size_t) ng * kGroupBytes;
    const bool contiguous = stride == row_bytes;
    const uint8_t* end = w + (size_t) (nrows - 1) * stride + row_bytes;
    if (pf && contiguous) prefetch_prologue(w, std::min<size_t>((size_t) pf, (size_t) (end - w)));
    const uint8_t* rp[NR];
    const uint8_t* lim[NR];
    int r = 0;
    for (; r + NR <= nrows; r += NR) {
        for (int i = 0; i < NR; ++i) {
            rp[i] = w + (size_t) (r + i) * stride;
            lim[i] = contiguous ? end : rp[i] + row_bytes;
        }
        tile<NR, NT>(rp, lim, ng, x, out + r, os, accumulate, pf);
    }
    for (; r < nrows; ++r) {
        rp[0] = w + (size_t) r * stride;
        lim[0] = contiguous ? end : rp[0] + row_bytes;
        tile<1, NT>(rp, lim, ng, x, out + r, os, accumulate, pf);
    }
}

// Rows per tile for each token count: the accumulators (NR x NT zmm) plus the decoded rows (3 zmm each) must stay inside the
// 32 registers, and a tile of NR rows reads each token's activation once per NR rows instead of once per row - which is what
// matters from T = 5 on, where the activations (11 KB per token) no longer fit L1 and one row per tile makes the kernel
// L2-bandwidth-bound (cache-resident, dev VM: T = 8 3.1 GB/s with NR = 1, 4.1 with NR = 2; T = 2 12.0 with NR = 2, 13.5 with 4).
// (Overridable at build time for tuning: -DDS41_NR1=4 ... -DDS41_NR8=2.)
#ifndef DS41_NR1
#define DS41_NR1 4
#endif
#ifndef DS41_NR2
#define DS41_NR2 4
#endif
#ifndef DS41_NR3
#define DS41_NR3 2
#endif
#ifndef DS41_NR4
#define DS41_NR4 2
#endif
#ifndef DS41_NR5
#define DS41_NR5 2
#endif
#ifndef DS41_NR6
#define DS41_NR6 2
#endif
#ifndef DS41_NR7
#define DS41_NR7 2
#endif
#ifndef DS41_NR8
#define DS41_NR8 2
#endif

void dot_rows_avx512(const uint8_t* w, size_t stride, int ng, int nrows, const ActQ* x, int T, float* out, size_t os,
                     bool accumulate, int pf) {
    switch (T) {
        case 1: rows_nt<DS41_NR1, 1>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 2: rows_nt<DS41_NR2, 2>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 3: rows_nt<DS41_NR3, 3>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 4: rows_nt<DS41_NR4, 4>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 5: rows_nt<DS41_NR5, 5>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 6: rows_nt<DS41_NR6, 6>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 7: rows_nt<DS41_NR7, 7>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        default: rows_nt<DS41_NR8, 8>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
    }
}

// 32 floats -> block `blk` of `a`, bit-identical to quantize_block_scalar (same FP32 operations, same rounding; the rule: quant_scale() in
// mxfp4_internal.hpp).  The largest magnitude is the INTEGER maximum of the magnitude bits (order-independent with NaN operands); the scale
// and the Inf / NaN / tiny cases come from the shared scalar quant_scale(); the codes are vcvtps2dq (round half to EVEN, the default MXCSR
// mode) of x * (127 / amax), and vpmovsdb saturates at -128..127, never reached: |x * id| <= 127 (1 + 2^-23) for a finite, non-tiny block.
void quantize_block_avx512(const float* x, ActQ& a, int blk) {
    const __m512 v0 = _mm512_loadu_ps(x), v1 = _mm512_loadu_ps(x + 16);
    const __m512i magmask = _mm512_set1_epi32(0x7FFFFFFF);
    const __m512i mag = _mm512_max_epu32(_mm512_and_si512(_mm512_castps_si512(v0), magmask), _mm512_and_si512(_mm512_castps_si512(v1), magmask));
    const QuantScale qs = quant_scale(_mm512_reduce_max_epu32(mag));
    const float d = qs.d;
    __m128i lo = _mm_setzero_si128(), hi = _mm_setzero_si128();      // every q = 0 for an all-zero / tiny / non-finite block
    if (!qs.zero) {
        const __m512 vid = _mm512_set1_ps(qs.id);
        lo = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v0, vid)));
        hi = _mm512_cvtsepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(v1, vid)));
    }
    const int g = blk >> 2, k = blk & 3;
    _mm_storeu_si128((__m128i*) (a.q + g * kGroupValues + k * 16), lo);
    _mm_storeu_si128((__m128i*) (a.q + g * kGroupValues + 64 + k * 16), hi);
    const __m128i ones = _mm_set1_epi8(1);
    __m128i s = _mm_dpbusd_epi32(_mm_setzero_si128(), ones, lo);
    s = _mm_dpbusd_epi32(s, ones, hi);
    _mm_store_si128((__m128i*) (a.corr + g * 16 + k * 4), _mm_mullo_epi32(s, _mm_set1_epi32(-12)));
    _mm_store_ps(a.sc4 + g * 16 + k * 4, _mm_set1_ps(d));
    a.scale[blk] = d;
}

const Kernels kAvx512 = {dot_rows_avx512, quantize_block_avx512, "avx512"};

}  // namespace

const Kernels* kernels_avx512() { return &kAvx512; }

}  // namespace strata::ds41::cpu::detail

#else

namespace strata::ds41::cpu::detail {
const Kernels* kernels_avx512() { return nullptr; }
}  // namespace strata::ds41::cpu::detail

#endif
