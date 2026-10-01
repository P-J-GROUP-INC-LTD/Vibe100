// src/ds41/cpu/mxfp4_avx2.cpp - DS-C: the MXFP4 dot product on AVX2 + FMA (no AVX-512, no VNNI).
//
// Compiled with -mavx2 -mfma -mf16c.  The same method and the same activation layout as the AVX-512 file (read its
// header and include/strata/ds41/cpu/mxfp4_expert.hpp), at half the width: a step is half a group, two blocks, one ymm.
//
//   weights   2 x `vmovdqu xmm` + `vinserti128` put two blocks' 16-byte `qs` into the two 128-bit lanes of a ymm; the nibbles
//             are split and looked up with `vpshufb` in the unsigned table kvalues + 12.
//   products  `vpmaddubsw` (unsigned weights x signed activations -> int16 pair sums, at most 2 * 24 * 127 = 6096 each, so
//             the saturating instruction cannot saturate; the low and the high halves add to at most 12192) and
//             `vpmaddwd` with ones give the same 4-lanes-per-block int32 partial sums `vpdpbusd` would, and the activation's
//             correction vector (-12 * sum(x) per lane) is ADDED to make them signed.
//   exponents the E8M0 bytes are picked out of the group's first 64 bytes by one `vpshufb` (byte k of lane k, see the
//             AVX-512 file) and turned into floats with integer operations: (e - 1) << 23 for e >= 2, and the two denormals
//             0x00200000 << e for e = 0, 1 (ggml's own formula), blended by a compare.
//
// There are only 16 ymm registers, so the tiles are small: the decode is repeated per half group and the accumulators
// are the budget.
#include "mxfp4_internal.hpp"

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <algorithm>

namespace strata::ds41::cpu::detail {
namespace {

constexpr int kGroupBytes = kGroupBlocks * kBlockBytes;   // 68

alignas(16) const uint8_t kTableU[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};   // kvalues_fp4 + 12
// half 0 picks e0 (byte 0 of lane 0) and e1 (byte 1 of lane 1) of the first 32 group bytes; half 1 picks e2 (byte 2 of
// lane 0) and e3 (byte 3 of lane 1) of the second 32 bytes
#define DS41_CTRL_LANE(k) (int8_t)(k), -128, -128, -128, (int8_t)(k), -128, -128, -128, (int8_t)(k), -128, -128, -128, (int8_t)(k), -128, -128, -128
alignas(32) const int8_t kCtrlE0[32] = {DS41_CTRL_LANE(0), DS41_CTRL_LANE(1)};
alignas(32) const int8_t kCtrlE1[32] = {DS41_CTRL_LANE(2), DS41_CTRL_LANE(3)};
#undef DS41_CTRL_LANE

struct Consts {
    __m256i table, m0f, ones16, ctrl0, ctrl1, one, two, denorm;
};

inline Consts make_consts() {
    Consts c;
    c.table = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i*) kTableU));
    c.m0f = _mm256_set1_epi8(0x0F);
    c.ones16 = _mm256_set1_epi16(1);
    c.ctrl0 = _mm256_load_si256((const __m256i*) kCtrlE0);
    c.ctrl1 = _mm256_load_si256((const __m256i*) kCtrlE1);
    c.one = _mm256_set1_epi32(1);
    c.two = _mm256_set1_epi32(2);
    c.denorm = _mm256_set1_epi32(0x00200000);
    return c;
}

struct HalfDec {
    __m256i ulo, uhi;   // unsigned weight bytes, lane 0 = block 2h, lane 1 = block 2h + 1
    __m256 ws;          // 2^(e - 128), four dwords per lane
};

// p = group start, h2 = which half (blocks 2 h2, 2 h2 + 1)
template <int H2>
inline HalfDec decode_half(const uint8_t* p, const Consts& c) {
    const __m128i a = _mm_loadu_si128((const __m128i*) (p + 1 + 34 * H2));
    const __m128i b = _mm_loadu_si128((const __m128i*) (p + 18 + 34 * H2));
    const __m256i z = _mm256_inserti128_si256(_mm256_castsi128_si256(a), b, 1);
    HalfDec d;
    d.ulo = _mm256_shuffle_epi8(c.table, _mm256_and_si256(z, c.m0f));
    d.uhi = _mm256_shuffle_epi8(c.table, _mm256_and_si256(_mm256_srli_epi16(z, 4), c.m0f));
    const __m256i raw = _mm256_loadu_si256((const __m256i*) (p + 32 * H2));
    const __m256i e = _mm256_shuffle_epi8(raw, H2 ? c.ctrl1 : c.ctrl0);
    const __m256i normal = _mm256_slli_epi32(_mm256_sub_epi32(e, c.one), 23);
    const __m256i denorm = _mm256_sllv_epi32(c.denorm, e);
    const __m256i is_denorm = _mm256_cmpgt_epi32(c.two, e);
    d.ws = _mm256_castsi256_ps(_mm256_blendv_epi8(normal, denorm, is_denorm));
    return d;
}

inline float hsum(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

template <int NR, int NT>
inline __attribute__((always_inline)) void tile(const uint8_t* const* rows, const uint8_t* const* lim, int ng,
                                                const ActQ* x, float* out, size_t out_stride, bool accumulate, int pf) {
    const Consts c = make_consts();
    __m256 acc[NR][NT];
#pragma GCC unroll 8
    for (int r = 0; r < NR; ++r)
#pragma GCC unroll 8
        for (int t = 0; t < NT; ++t) acc[r][t] = _mm256_setzero_ps();

    for (int g = 0; g < ng; ++g) {
#pragma GCC unroll 8
        for (int r = 0; r < NR; ++r) {
            const uint8_t* p = rows[r] + (size_t) g * kGroupBytes;
            if (pf) prefetch_ahead(p, pf, lim[r]);
        }
#pragma GCC unroll 2
        for (int h2 = 0; h2 < 2; ++h2) {
            HalfDec d[NR];
#pragma GCC unroll 8
            for (int r = 0; r < NR; ++r) {
                const uint8_t* p = rows[r] + (size_t) g * kGroupBytes;
                d[r] = h2 == 0 ? decode_half<0>(p, c) : decode_half<1>(p, c);
            }
#pragma GCC unroll 8
            for (int t = 0; t < NT; ++t) {
                const ActQ& a = x[t];
                const __m256i xlo = _mm256_load_si256((const __m256i*) (a.q + g * kGroupValues + 32 * h2));
                const __m256i xhi = _mm256_load_si256((const __m256i*) (a.q + g * kGroupValues + 64 + 32 * h2));
                const __m256i corr = _mm256_load_si256((const __m256i*) (a.corr + g * 16 + 8 * h2));
                const __m256 sc = _mm256_load_ps(a.sc4 + g * 16 + 8 * h2);
#pragma GCC unroll 8
                for (int r = 0; r < NR; ++r) {
                    const __m256i p16 = _mm256_add_epi16(_mm256_maddubs_epi16(d[r].ulo, xlo),
                                                         _mm256_maddubs_epi16(d[r].uhi, xhi));
                    const __m256i s = _mm256_add_epi32(_mm256_madd_epi16(p16, c.ones16), corr);
                    acc[r][t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(d[r].ws, sc), acc[r][t]);
                }
            }
        }
    }
#pragma GCC unroll 8
    for (int t = 0; t < NT; ++t)
#pragma GCC unroll 8
        for (int r = 0; r < NR; ++r) {
            const float v = hsum(acc[r][t]);
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

// Rows per tile for each token count (16 ymm registers: the accumulators NR x NT, the decoded half group of each row, the
// constants).  Overridable at build time for tuning: -DDS41_A2_NR1=2 ...
#ifndef DS41_A2_NR1
#define DS41_A2_NR1 2
#endif
#ifndef DS41_A2_NR2
#define DS41_A2_NR2 2
#endif
#ifndef DS41_A2_NR3
#define DS41_A2_NR3 1
#endif
#ifndef DS41_A2_NR4
#define DS41_A2_NR4 1
#endif
#ifndef DS41_A2_NR5
#define DS41_A2_NR5 1
#endif

void dot_rows_avx2(const uint8_t* w, size_t stride, int ng, int nrows, const ActQ* x, int T, float* out, size_t os,
                   bool accumulate, int pf) {
    switch (T) {
        case 1: rows_nt<DS41_A2_NR1, 1>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 2: rows_nt<DS41_A2_NR2, 2>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 3: rows_nt<DS41_A2_NR3, 3>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 4: rows_nt<DS41_A2_NR4, 4>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 5: rows_nt<DS41_A2_NR5, 5>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 6: rows_nt<1, 6>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        case 7: rows_nt<1, 7>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
        default: rows_nt<1, 8>(w, stride, ng, nrows, x, out, os, accumulate, pf); break;
    }
}

// 32 floats -> block `blk` of `a`; bit-identical to quantize_block_scalar.
void quantize_block_avx2(const float* x, ActQ& a, int blk) {
    const __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8), v2 = _mm256_loadu_ps(x + 16),
                 v3 = _mm256_loadu_ps(x + 24);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 m = _mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1));
    m = _mm256_max_ps(m, _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    const float amax = _mm_cvtss_f32(m4);
    const float d = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    const __m256 vid = _mm256_set1_ps(id);
    const __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vid)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vid)),
                  i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vid)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vid));
    // packs works within 128-bit lanes: the permute restores the natural order of the 32 bytes
    __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
    p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    const __m128i lo = _mm256_castsi256_si128(p), hi = _mm256_extracti128_si256(p, 1);
    const int g = blk >> 2, k = blk & 3;
    _mm_storeu_si128((__m128i*) (a.q + g * kGroupValues + k * 16), lo);
    _mm_storeu_si128((__m128i*) (a.q + g * kGroupValues + 64 + k * 16), hi);
    const __m128i ones8 = _mm_set1_epi8(1);
    const __m128i s16 = _mm_add_epi16(_mm_maddubs_epi16(ones8, lo), _mm_maddubs_epi16(ones8, hi));
    const __m128i s = _mm_madd_epi16(s16, _mm_set1_epi16(1));
    _mm_store_si128((__m128i*) (a.corr + g * 16 + k * 4), _mm_mullo_epi32(s, _mm_set1_epi32(-12)));
    _mm_store_ps(a.sc4 + g * 16 + k * 4, _mm_set1_ps(d));
    a.scale[blk] = d;
}

const Kernels kAvx2 = {dot_rows_avx2, quantize_block_avx2, "avx2"};

}  // namespace

const Kernels* kernels_avx2() { return &kAvx2; }

}  // namespace strata::ds41::cpu::detail

#else

namespace strata::ds41::cpu::detail {
const Kernels* kernels_avx2() { return nullptr; }
}  // namespace strata::ds41::cpu::detail

#endif
