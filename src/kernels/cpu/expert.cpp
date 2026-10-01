// src/kernels/cpu/expert.cpp - P2.S3: the Q2_0 expert kernel, promoted from bench/micro/cpu_s2.cpp.
//
// The body is the validated kernel, moved rather than rewritten: it carries the P0.T2 parity result
// (rel 1.461e-06 against the ggml formula) and the three performance findings recorded in
// include/strata/kernels/cpu/expert.hpp.  Read that header first; it says why each piece is shaped this way.
//
// ---- TWO BUILDS OF THIS FILE (Vibe100: Cascade Lake has AVX512-VNNI but no AVX512-VBMI) ----
//
// This file is compiled TWICE and nothing else in it differs between the two builds:
//
//   expert.cpp          with -mavx512vbmi         namespace `vbmi`    2-bit unpack by `vpmultishiftqb` (upstream's)
//   expert_novbmi.cpp   without it, and it defines STRATA_EXPERT_NO_VBMI before including this file
//                                                 namespace `novbmi`  2-bit unpack by shuffle / shift / mask
//
// The two unpack helpers (`unpack_q2_0`, `unpack64_q2_0`) are the only code the macro selects.  Everything the
// kernels do with the unpacked codes - the VNNI dot, the scale products, the float accumulation order - is the
// same source, so the same instructions, and the two builds are bit-identical as long as the two unpacks produce the
// same bytes, which `expert_variant_test` checks (exhaustively over every possible input dword) and the comments at
// the helpers prove lane by lane.  The public entry points of expert.hpp are NOT defined here: they are the
// dispatchers in expert_dispatch.cpp, which pick a build once, at run time, from the CPU's feature bits.
#include "strata/kernels/cpu/expert.hpp"
#include "expert_variant.hpp"

#include <immintrin.h>

#include <cmath>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(STRATA_EXPERT_NO_VBMI)
#define STRATA_EXPERT_NS novbmi
#define STRATA_EXPERT_KERNELS expert_kernels_novbmi
#define STRATA_EXPERT_NAME "AVX-512 VNNI (no VBMI)"
#else
#define STRATA_EXPERT_NS vbmi
#define STRATA_EXPERT_KERNELS expert_kernels_vbmi
#define STRATA_EXPERT_NAME "AVX-512 VNNI + VBMI"
#endif

namespace strata::kernels::cpu {
namespace STRATA_EXPERT_NS {
namespace {

// The flag itself lives in expert_dispatch.cpp (one copy for both builds); the name is kept so the kernels below read
// as they always did.
std::atomic<bool>& oracle_q8_0 = g_oracle_q8_0;

/// fp16 -> fp32, written out rather than using `_cvtsh_ss`, because the F16C intrinsic's behaviour on
/// subnormals is the one place the two can differ and the scales in this artifact are small.
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

// The x86 Q8_0 quantization and generic Q2_0 dot sequences are adapted from llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d,
// ggml/src/ggml-cpu/arch/x86/quants.c: quantize_row_q8_0 and
// ggml/src/ggml-cpu/quants.c: ggml_vec_dot_q2_0_q8_0_generic.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
void quantize_oracle_q8_0(const float* x, int n, ActQ& a) {
    a.nchunks = n / QKA;
    for (int chunk = 0; chunk < a.nchunks; ++chunk) {
        const float* xb = x + chunk * QKA;
        __m256 v0 = _mm256_loadu_ps(xb);
        __m256 v1 = _mm256_loadu_ps(xb + 8);
        __m256 v2 = _mm256_loadu_ps(xb + 16);
        __m256 v3 = _mm256_loadu_ps(xb + 24);
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 maximum = _mm256_andnot_ps(sign, v0);
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, v1));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, v2));
        maximum = _mm256_max_ps(maximum, _mm256_andnot_ps(sign, v3));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(maximum, 1), _mm256_castps256_ps128(maximum));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float amax = _mm_cvtss_f32(max4);
        const float d = amax / 127.0f;
        const uint16_t half_d = (uint16_t) _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(d), 0));
        const __m256 inverse = _mm256_set1_ps(amax != 0.0f ? 127.0f / amax : 0.0f);
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, inverse), _MM_FROUND_TO_NEAREST_INT);
        v1 = _mm256_round_ps(_mm256_mul_ps(v1, inverse), _MM_FROUND_TO_NEAREST_INT);
        v2 = _mm256_round_ps(_mm256_mul_ps(v2, inverse), _MM_FROUND_TO_NEAREST_INT);
        v3 = _mm256_round_ps(_mm256_mul_ps(v3, inverse), _MM_FROUND_TO_NEAREST_INT);
        __m256i i0 = _mm256_packs_epi32(_mm256_cvtps_epi32(v0), _mm256_cvtps_epi32(v1));
        const __m256i i2 = _mm256_packs_epi32(_mm256_cvtps_epi32(v2), _mm256_cvtps_epi32(v3));
        i0 = _mm256_packs_epi16(i0, i2);
        i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        int8_t* q = a.q + chunk * QKA;
        _mm256_storeu_si256((__m256i*) q, i0);
        int32_t sum = 0;
        for (int j = 0; j < QKA; ++j) sum += q[j];
        a.scale[chunk] = h2f((const uint8_t*) &half_d);
        a.sum[chunk] = sum;
        a.hx[chunk] = a.scale[chunk] * float(sum);
    }
}

/// 2-bit unpack via VPMULTISHIFTQB.
///
/// ARGUMENT ORDER IS (control, data) AND WAS DETERMINED EMPIRICALLY (`bench/micro/probe_multishift.cpp`).
/// The Intel guide states (a=data, b=control); using that produced 43 of 64 wrong codes.  `plan.md`'s sketch
/// passes the shift pattern first, which is correct - so a "fix" that swapped it to match the guide was the
/// bug, not the fix.  Within each 64-bit lane, output byte i is `(u16 >> 2i) & 0xFF`, so AND 3 yields the
/// eight 2-bit codes in order.
///
/// The result is the 32 codes of ONE ACTIVATION CHUNK (4 qword lanes), which is the unit the Q8_1 scale is
/// defined on - not the 64-weight block.
#if !defined(STRATA_EXPERT_NO_VBMI)
inline __m256i unpack_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadl_epi64((const __m128i*) codes);   // 8 bytes -> 4 u16 in the low half
    const __m256i lanes = _mm256_cvtepu16_epi64(packed);              // 4 qwords, 8 codes each
    const __m256i ctrl = _mm256_set1_epi64x((long long) 0x0E0C0A0806040200ULL);
    return _mm256_and_si256(_mm256_multishift_epi64_epi8(ctrl, lanes), _mm256_set1_epi8(3));
}
#else
// The same 32 codes without VBMI (AVX2 instructions only; the 256-bit kernels are the debug / oracle path).
//
// The output is two 128-bit lanes of 16 codes.  Lane L holds codes 16L..16L+15, which come from the four bytes
// codes[4L .. 4L+3] - ONE DWORD of the packed input - and nothing else, so each lane is an independent 4x4 problem:
//     out[L][4*m + k] = (codes[4L + m] >> 2k) & 3        m = byte of the dword, k = code within the byte.
// Four steps, each a single instruction:
//   1. broadcast the 8 input bytes into both lanes (a memory-operand `vpbroadcastq`);
//   2. `vpshufb`: lane L's four dwords all become input dword L.  Control byte (4t + u) = 4L + u, so dword t of the lane is
//      [codes[4L], codes[4L+1], codes[4L+2], codes[4L+3]] for every t.  vpshufb indexes inside its own 128-bit lane
//      and 4L + u <= 7 < 16, which the broadcast made valid in both lanes;
//   3. `vpsrlvd` by (0, 2, 4, 6) for dwords t = 0..3, then AND 0x03 per byte: byte u of dword t becomes
//      (codes[4L + u] >> 2t) & 3, because bit (8u + 2t) of the dword is bit 2t of byte u and the bits that cross in
//      from byte u+1 land at positions 8 - 2t .. 7 >= 2, above the mask (2t <= 6);
//   4. `vpshufb` with control byte (4m + k) = 4k + m: the 4x4 byte transpose that puts dword-major (t = k, u = m) data
//      into value order out[4m + k].
// Each output byte is therefore exactly `(codes[j / 4] >> (2 * (j % 4))) & 3` for j = 16L + 4m + k, which is what
// `vpmultishiftqb` + AND 3 gives in the VBMI build (below, and in the byte-for-byte comment on unpack64_q2_0).
inline __m256i unpack_q2_0(const uint8_t* codes) {
    const __m256i src = _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i*) codes));
    const __m256i rep = _mm256_shuffle_epi8(src, _mm256_setr_epi32(0x03020100, 0x03020100, 0x03020100, 0x03020100,
                                                                   0x07060504, 0x07060504, 0x07060504, 0x07060504));
    const __m256i sh = _mm256_srlv_epi32(rep, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
    const __m256i plane = _mm256_and_si256(sh, _mm256_set1_epi8(3));
    return _mm256_shuffle_epi8(plane, _mm256_setr_epi32(0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703,
                                                        0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703));
}
#endif

/// 32 codes against 32 int8 activations. Codes 0 contribute nothing to `sum(c*xhat)`, so the zero lanes the
/// narrow load leaves behind are harmless.
inline __m256i block_dot(const uint8_t* codes, const int8_t* xq) {
    return _mm256_dpbusd_epi32(_mm256_setzero_si256(), unpack_q2_0(codes),
                               _mm256_load_si256((const __m256i*) xq));
}

inline float hsum_ps(__m256 v) {
    const __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    return _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(s, s), s));
}

// ================================ plan v0.3 P6: the row dot in 512-bit lanes ================================
//
// The 256-bit row dot below spent ~12 cycles per 64-weight block: two unpacks, two VNNI dots, two converts, two
// scalar scale products broadcast into two FMAs, an fp16 conversion and a scalar correction - compute-bound at
// ~39 GB/s while the machine streams 52.  Here a block is ONE unpack (16 code bytes -> 64 codes), ONE VNNI dot
// over 64 activations, ONE convert and ONE FMA whose per-chunk scales come from a permute of a vector of scale
// products built for eight blocks at a time; the weight-independent correction is one FMA per eight blocks.
// The sum order differs from the 256-bit kernel (last-bit differences); the verify window and plain decode both
// use this kernel, so speculative decode still reproduces plain decode exactly.
static const bool kZmm = std::getenv("STRATA_CPU_YMM") == nullptr;

#if !defined(STRATA_EXPERT_NO_VBMI)
inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m128i packed = _mm_loadu_si128((const __m128i*) codes);          // 8 u16 = 64 codes
    const __m512i lanes = _mm512_cvtepu16_epi64(packed);                    // 8 qwords, 8 codes each
    const __m512i ctrl = _mm512_set1_epi64((long long) 0x0E0C0A0806040200ULL);
    return _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, lanes), _mm512_set1_epi8(3));
}
#else
// The same 64 codes without VBMI: the 256-bit scheme above, four lanes wide.
//
// THE SPECIFICATION BOTH BUILDS MEET: output byte j (0..63) is `(codes[j / 4] >> (2 * (j % 4))) & 3`.
//   VBMI build:   qword i of `lanes` is the u16 codes[2i..2i+1] (zero-extended); control byte c of the multishift
//                 selects the 8 bits starting at bit 2c, i.e. byte c of the qword is (u16 >> 2c) & 0xFF for c <= 7
//                 (no wrap-around: 2c + 8 <= 22 < 64); AND 3 keeps the two low bits.  Output byte 8i + c is then
//                 (u16_i >> 2c) & 3 = code number 8i + c of the stream (code n lives in byte n/4 at bit 2(n%4), and
//                 u16_i holds bytes 2i and 2i+1, so its 2c-bit offset covers codes 8i..8i+7 in order).
//   this build:   lane L (0..3) holds output bytes 16L..16L+15 and reads only input bytes 4L..4L+3 (one dword):
//                   1. `vbroadcasti32x4` puts all 16 input bytes in every lane (a pure load, no ALU uop);
//                   2. `vpshufb`, control byte (4t + u) of lane L = 4L + u: dword t of lane L becomes input dword L
//                      for t = 0..3 (the index stays inside the lane: 4L + u <= 15);
//                   3. `vpsrlvd` by 2t on dword t (counts 0,2,4,6), AND 0x03 per byte: byte u of dword t
//                      = (codes[4L + u] >> 2t) & 3.  Bits shifted in from byte u+1 land at positions 8 - 2t .. 7, all
//                      >= 2, so the mask removes them; and for u = 3 the shift brings in zeros;
//                   4. `vpshufb`, control byte (4m + k) = 4k + m: output byte 16L + 4m + k takes the byte at lane
//                      position 4k + m, i.e. dword t = k, byte u = m: (codes[4L + m] >> 2k) & 3
//                      = (codes[(16L + 4m + k) / 4] >> 2((16L + 4m + k) % 4)) & 3.   QED for every j.
// Four ALU uops (vpshufb p5, vpsrlvd p0, vpand p05, vpshufb p5) against the VBMI build's three (vpmovzxwq p5,
// vpmultishiftqb p0, vpand p05), all single-uop on Skylake-X / Cascade Lake.
inline __m512i unpack64_q2_0(const uint8_t* codes) {
    const __m512i src = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*) codes));
    const __m512i rep = _mm512_shuffle_epi8(src, _mm512_setr_epi32(0x03020100, 0x03020100, 0x03020100, 0x03020100,
                                                                   0x07060504, 0x07060504, 0x07060504, 0x07060504,
                                                                   0x0B0A0908, 0x0B0A0908, 0x0B0A0908, 0x0B0A0908,
                                                                   0x0F0E0D0C, 0x0F0E0D0C, 0x0F0E0D0C, 0x0F0E0D0C));
    const __m512i sh = _mm512_srlv_epi32(rep, _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6));
    const __m512i plane = _mm512_and_si512(sh, _mm512_set1_epi8(3));
    return _mm512_shuffle_epi8(plane, _mm512_setr_epi32(0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703,
                                                        0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703,
                                                        0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703,
                                                        0x0C080400, 0x0D090501, 0x0E0A0602, 0x0F0B0703));
}
#endif

// For blocks [b0, b0+nb) (nb <= 8): p[2i+h] = d_{b0+i} * xscale[2(b0+i)+h], dd[2i+h] = d_{b0+i}.
inline void scales8(const uint8_t* wscales, const ActQ& a, int b0, int nb, __m512& p, __m512& dd) {
    const __mmask8 m8 = nb >= 8 ? (__mmask8) 0xFF : (__mmask8) ((1u << nb) - 1u);
    const __m128i h = _mm_maskz_loadu_epi16(m8, wscales + 2 * b0);
    const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(h));
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    dd = _mm512_permutexvar_ps(dup, d8);
    const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
    dd = _mm512_maskz_mov_ps(m16, dd);
    p = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a.scale + 2 * b0));
}

inline float row_dot_z(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    __m512 acc = _mm512_setzero_ps(), corr = _mm512_setzero_ps();
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        __m512 p, dd;
        scales8(scales, a, b0, nb, p, dd);
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        corr = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a.hx + 2 * b0), corr);
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), unpack64_q2_0(codes + b * 16),
                                                    _mm512_load_si512((const void*) (a.q + b * QK)));
            const __m512 sv = _mm512_permutexvar_ps(_mm512_add_epi32(base, _mm512_set1_epi32(2 * i)), p);
            acc = _mm512_fmadd_ps(sv, _mm512_cvtepi32_ps(dot), acc);
        }
    }
    return _mm512_reduce_add_ps(acc) - _mm512_reduce_add_ps(corr);
}

template<int NT>
inline void row_dot_multi_z(const uint8_t* codes, const uint8_t* scales, const ActQ* const* a, int nblocks,
                            float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            __m512 dd;
            scales8(scales, *a[t], b0, nb, p[t], dd);
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(codes + b * 16);
            const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(2 * i));
            for (int t = 0; t < NT; ++t) {
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                        _mm512_load_si512((const void*) (a[t]->q + b * QK)));
                acc[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx, p[t]), _mm512_cvtepi32_ps(dot), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - _mm512_reduce_add_ps(corr[t]);
}

/// One weight row against a quantized activation.
///
///     sum_j (c_j - 1) d_w xhat_j d_x  ==  d_w * ( d_x*sum(c*xhat) - d_x*sum(xhat) )
///
/// split per 32-chunk because `d_x` changes at the chunk boundary while `d_w` does not.
///
/// The accumulator stays a FLOAT VECTOR for the whole row and is reduced once at the end.  Reducing per
/// 64-weight block (two 8-lane horizontal sums: a store plus 16 scalar adds) cost far more than the dot
/// products themselves - it made the kernel compute-bound at 21 GB/s instead of DRAM-bound at 32.
inline float row_dot(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    if (kZmm) return row_dot_z(codes, scales, a, nblocks);
    __m256 acc = _mm256_setzero_ps();
    // The -sum(xhat) term is a SCALAR: broadcasting it into all 8 lanes and letting hsum_ps add them would
    // apply it eight times per block.
    float corr = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);
        const __m256 slo = _mm256_cvtepi32_ps(block_dot(codes + b * 16, a.q + b * QK));
        const __m256 shi = _mm256_cvtepi32_ps(block_dot(codes + b * 16 + 8, a.q + b * QK + QKA));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * a.scale[2 * b]), slo, acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d * a.scale[2 * b + 1]), shi, acc);
        corr += d * (a.hx[2 * b] + a.hx[2 * b + 1]);
    }
    return hsum_ps(acc) - corr;
}

// VNNI's integer partial sums are exact. Remove the Q2_0 code offset BEFORE
// conversion to float, then follow the pinned generic CPU dot's two-chunk and
// per-64-block reduction. Its x86 type trait selects this generic reduction even
// in the AVX-512 oracle build. The legacy eight-lane FP32 reduction above has a
// different rounding order and remains the default.
inline int chunk_dot_oracle(const uint8_t* codes, const int8_t* q, int sum) {
    const __m256i lanes = block_dot(codes, q);
    const __m128i halves = _mm_add_epi32(_mm256_castsi256_si128(lanes), _mm256_extracti128_si256(lanes, 1));
    const __m128i pairs = _mm_hadd_epi32(halves, halves);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(pairs, pairs)) - sum;
}

inline float row_dot_oracle(const uint8_t* codes, const uint8_t* scales, const ActQ& a, int nblocks) {
    float result = 0.0f;
    for (int b = 0; b < nblocks; ++b) {
        float block = 0.0f;
        for (int half = 0; half < 2; ++half) {
            const int chunk = 2 * b + half;
            const int dot = chunk_dot_oracle(codes + b * 16 + half * 8, a.q + chunk * QKA, a.sum[chunk]);
            block += a.scale[chunk] * float(dot);
        }
        result += h2f(scales + 2 * b) * block;
    }
    return result;
}

/// `row_dot` for NT activations at once: the codes of each block are unpacked once, and each token's
/// accumulator sees exactly the operations `row_dot` would apply to it, in the same order.
template<int NT>
inline void row_dot_multi(const uint8_t* codes, const uint8_t* scales, const ActQ* const* a, int nblocks,
                          float* res) {
    if (kZmm) { row_dot_multi_z<NT>(codes, scales, a, nblocks, res); return; }
    __m256 acc[NT];
    float corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm256_setzero_ps(); corr[t] = 0.f; }
    for (int b = 0; b < nblocks; ++b) {
        const float d = h2f(scales + 2 * b);
        const __m256i lo = unpack_q2_0(codes + b * 16);
        const __m256i hi = unpack_q2_0(codes + b * 16 + 8);
        for (int t = 0; t < NT; ++t) {
            const __m256 slo = _mm256_cvtepi32_ps(
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, _mm256_load_si256((const __m256i*) (a[t]->q + b * QK))));
            const __m256 shi = _mm256_cvtepi32_ps(
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), hi,
                                    _mm256_load_si256((const __m256i*) (a[t]->q + b * QK + QKA))));
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b]), slo, acc[t]);
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * a[t]->scale[2 * b + 1]), shi, acc[t]);
            corr[t] += d * (a[t]->hx[2 * b] + a[t]->hx[2 * b + 1]);
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = hsum_ps(acc[t]) - corr[t];
}

template<int NT>
void expert_multi(const uint8_t* blob, const ActQ* const* a1, float* const* out, ExpertScratchMulti& ws) {
    float g[NT], u[NT];
    for (int r = 0; r < FF; ++r) {
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU, g);
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU, u);
        for (int t = 0; t < NT; ++t) ws.ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
    const ActQ* a2[NT];
    for (int t = 0; t < NT; ++t) {
        act_quant_q8_1(ws.ff[t], FF, ws.a2[t]);
        a2[t] = &ws.a2[t];
    }
    float o[NT];
    for (int r = 0; r < H; ++r) {
        row_dot_multi<NT>(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2,
                          SC_D, o);
        for (int t = 0; t < NT; ++t) out[t][r] = o[t];
    }
}

void expert_oracle_q8_0(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    for (int r = 0; r < FF; ++r) {
        const float g = row_dot_oracle(blob + O_GU_CODES + size_t(2 * r) * ROW_GU,
                                      blob + O_GU_SCALES + size_t(2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot_oracle(blob + O_GU_CODES + size_t(2 * r + 1) * ROW_GU,
                                      blob + O_GU_SCALES + size_t(2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ws.ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    quantize_oracle_q8_0(ws.ff, FF, ws.a2);
    for (int r = 0; r < H; ++r)
        out[r] = row_dot_oracle(blob + O_D_CODES + size_t(r) * ROW_D,
                               blob + O_D_SCALES + size_t(r) * SC_D * 2, ws.a2, SC_D);
}

}  // namespace

// ---- the entry points (kernel side).  `act_quant_q8_1`, `s2_expert_vnni_q`, ... of expert.hpp are the dispatchers in
// expert_dispatch.cpp; they reach these through the table at the end of this file.  The `k_` prefix is not decoration:
// with the same name and signature as the public functions, argument-dependent lookup would find both and the calls
// below would be ambiguous.
void k_act_quant_q8_1(const float* x, int n, ActQ& a) {
    if (oracle_q8_0.load(std::memory_order_relaxed)) {
        quantize_oracle_q8_0(x, n, a);
        return;
    }
    a.nchunks = n / QKA;
    // Plan v0.3 P6: AVX-512, the same operations per element as the scalar loop below (max of |x|, one multiply,
    // +-0.5 away from zero, truncation, clamp), so the result is bitwise the scalar one.  The scalar loop took
    // ~22 us per 2560 values - 3.2 ms of every speculative round.
    {
        const __m512 half = _mm512_set1_ps(0.5f), mhalf = _mm512_set1_ps(-0.5f), zero = _mm512_setzero_ps();
        const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
        const __m512 absmask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        for (int k = 0; k < a.nchunks; ++k) {
            const float* xb = x + k * QKA;
            const __m512 x0 = _mm512_loadu_ps(xb), x1 = _mm512_loadu_ps(xb + 16);
            const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_and_ps(x0, absmask), _mm512_and_ps(x1, absmask)));
            const float s = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = s > 0.f ? 1.f / s : 0.f;
            const __m512 vinv = _mm512_set1_ps(inv);
            const __m512 t0 = _mm512_mul_ps(x0, vinv), t1 = _mm512_mul_ps(x1, vinv);
            const __m512 r0 = _mm512_add_ps(t0, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t0, zero, _CMP_GE_OQ), mhalf, half));
            const __m512 r1 = _mm512_add_ps(t1, _mm512_mask_blend_ps(_mm512_cmp_ps_mask(t1, zero, _CMP_GE_OQ), mhalf, half));
            __m512i v0 = _mm512_cvttps_epi32(r0), v1 = _mm512_cvttps_epi32(r1);
            v0 = _mm512_min_epi32(_mm512_max_epi32(v0, lo), hi);
            v1 = _mm512_min_epi32(_mm512_max_epi32(v1, lo), hi);
            _mm_storeu_si128((__m128i*) (a.q + k * QKA), _mm512_cvtepi32_epi8(v0));
            _mm_storeu_si128((__m128i*) (a.q + k * QKA + 16), _mm512_cvtepi32_epi8(v1));
            const int32_t sum = _mm512_reduce_add_epi32(_mm512_add_epi32(v0, v1));
            a.scale[k] = s;
            a.sum[k] = sum;
            a.hx[k] = s * (float) sum;
        }
        return;
    }
    for (int k = 0; k < a.nchunks; ++k) {
        const float* xb = x + k * QKA;
        float amax = 0.f;
        for (int j = 0; j < QKA; ++j) amax = std::fmax(amax, std::fabs(xb[j]));
        const float s = amax > 0.f ? amax / 127.f : 0.f;
        const float inv = s > 0.f ? 1.f / s : 0.f;
        int32_t sum = 0;
        int8_t* q = a.q + k * QKA;
        for (int j = 0; j < QKA; ++j) {
            // **`std::lround` IS A FUNCTION CALL AND IT COST 51 us PER LAYER.**  Measured: quantizing 2560
            // floats took 51.3 us - 20 ns per element, about 60 cycles for a fabs, a multiply and a round.
            // `lround` respects the current rounding mode, so MSVC cannot inline it to a single instruction
            // and emits a call per element.  48 layers x 51.3 us = 2.46 ms/token, which is 8.6% of the whole
            // CPU expert term and is pure overhead.
            //
            // `t + copysign(0.5, t)` is EXACTLY `lround`'s rule - round half AWAY FROM ZERO - and it is
            // branchless and vectorizable.  It is NOT `_mm256_round_ps`, which rounds half to EVEN and would
            // quietly change the activation on the ties.  The cast is safe because `|t| <= 127` by construction
            // (`s = amax/127`, so `|x_j|/s <= 127`), well inside int32.
            const float t = xb[j] * inv;
            const float r = t + (t >= 0.f ? 0.5f : -0.5f);
            int v = (int) r;
            v = v < -127 ? -127 : (v > 127 ? 127 : v);
            q[j] = (int8_t) v;
            sum += v;
        }
        a.scale[k] = s;
        a.sum[k] = sum;
        a.hx[k] = s * (float) sum;
    }
}

void k_s2_expert_vnni_q(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    if (oracle_q8_0.load(std::memory_order_relaxed)) {
        expert_oracle_q8_0(blob, a1, out, ws);
        return;
    }
    for (int r = 0; r < FF; ++r) {
        const float g = row_dot(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        // SiLU on the GATE, multiplied by up - the same reading `docs/semantics.md` records for the shared
        // expert, and the one that is wrong the other way round in a way that still produces a number.
        ws.ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
    act_quant_q8_1(ws.ff, FF, ws.a2);
    for (int r = 0; r < H; ++r)
        out[r] = row_dot(blob + O_D_CODES + (size_t) r * ROW_D,
                         blob + O_D_SCALES + (size_t) r * SC_D * 2, ws.a2, SC_D);
}

void k_s2_expert_gu_rows(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
        const float g = row_dot(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU);
        const float u = row_dot(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                                blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

void k_s2_expert_down_rows(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    for (int r = r0; r < r1; ++r)
        out[r] = row_dot(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2, SC_D);
}

namespace {
template<int NT>
void gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, float* const* ff, int r0, int r1) {
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2, a1, SC_GU, g);
        row_dot_multi<NT>(blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU,
                          blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2, a1, SC_GU, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}
template<int NT>
void down_rows_multi(const uint8_t* blob, const ActQ* const* a2, float* const* out, int r0, int r1) {
    float o[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot_multi<NT>(blob + O_D_CODES + (size_t) r * ROW_D, blob + O_D_SCALES + (size_t) r * SC_D * 2, a2,
                          SC_D, o);
        for (int t = 0; t < NT; ++t) out[t][r] = o[t];
    }
}
}  // namespace

void k_s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                               int r1) {
    switch (n_tokens) {
        case 1: gu_rows_multi<1>(blob, a1, ff, r0, r1); break;
        case 2: gu_rows_multi<2>(blob, a1, ff, r0, r1); break;
        case 3: gu_rows_multi<3>(blob, a1, ff, r0, r1); break;
        case 4: gu_rows_multi<4>(blob, a1, ff, r0, r1); break;
        case 5: gu_rows_multi<5>(blob, a1, ff, r0, r1); break;
        case 6: gu_rows_multi<6>(blob, a1, ff, r0, r1); break;
        case 7: gu_rows_multi<7>(blob, a1, ff, r0, r1); break;
        default: gu_rows_multi<8>(blob, a1, ff, r0, r1); break;
    }
}

void k_s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0,
                                 int r1) {
    switch (n_tokens) {
        case 1: down_rows_multi<1>(blob, a2, out, r0, r1); break;
        case 2: down_rows_multi<2>(blob, a2, out, r0, r1); break;
        case 3: down_rows_multi<3>(blob, a2, out, r0, r1); break;
        case 4: down_rows_multi<4>(blob, a2, out, r0, r1); break;
        case 5: down_rows_multi<5>(blob, a2, out, r0, r1); break;
        case 6: down_rows_multi<6>(blob, a2, out, r0, r1); break;
        case 7: down_rows_multi<7>(blob, a2, out, r0, r1); break;
        default: down_rows_multi<8>(blob, a2, out, r0, r1); break;
    }
}

// ================================ plan v0.3 P6: Q2_0 rows in the GGUF block layout ================================
//
// The IQ model files keep their down projections in Q2_0 GGUF blocks (fp16 d, then 16 code bytes: 18 bytes per
// 64 weights, interleaved), which ggml-cpu computes with a scalar loop on x86.  This is `row_dot_multi_z` with
// the block stride of the GGUF layout: the same unpack, the same VNNI dot, the same correction.
namespace {
template<int NT>
inline void q2g_row_multi(const uint8_t* row, const ActQ* const* a, int nblocks, float* res) {
    __m512 acc[NT], corr[NT];
    for (int t = 0; t < NT; ++t) { acc[t] = _mm512_setzero_ps(); corr[t] = _mm512_setzero_ps(); }
    const __m512i base = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    for (int b0 = 0; b0 < nblocks; b0 += 8) {
        const int nb = nblocks - b0 < 8 ? nblocks - b0 : 8;
        const __mmask16 m16 = nb >= 8 ? (__mmask16) 0xFFFF : (__mmask16) ((1u << (2 * nb)) - 1u);
        alignas(16) uint16_t sc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < nb; ++i) std::memcpy(&sc[i], row + (size_t) (b0 + i) * 18, 2);
        const __m512 d8 = _mm512_castps256_ps512(_mm256_cvtph_ps(_mm_load_si128((const __m128i*) sc)));
        const __m512 dd = _mm512_maskz_mov_ps(m16, _mm512_permutexvar_ps(dup, d8));
        __m512 p[NT];
        for (int t = 0; t < NT; ++t) {
            p[t] = _mm512_mul_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->scale + 2 * b0));
            corr[t] = _mm512_fmadd_ps(dd, _mm512_maskz_loadu_ps(m16, a[t]->hx + 2 * b0), corr[t]);
        }
        for (int i = 0; i < nb; ++i) {
            const int b = b0 + i;
            const __m512i w = unpack64_q2_0(row + (size_t) b * 18 + 2);
            const __m512i idx = _mm512_add_epi32(base, _mm512_set1_epi32(2 * i));
            for (int t = 0; t < NT; ++t) {
                const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w,
                                                        _mm512_load_si512((const void*) (a[t]->q + b * QK)));
                acc[t] = _mm512_fmadd_ps(_mm512_permutexvar_ps(idx, p[t]), _mm512_cvtepi32_ps(dot), acc[t]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(acc[t]) - _mm512_reduce_add_ps(corr[t]);
}
template<int NT>
void q2g_rows(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, float* const* out, int r0, int r1) {
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        q2g_row_multi<NT>(w + (size_t) r * row_bytes, a, nblocks, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}
}  // namespace

void k_q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                            float* const* out, int r0, int r1) {
    switch (nt) {
        case 1: q2g_rows<1>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 2: q2g_rows<2>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 3: q2g_rows<3>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 4: q2g_rows<4>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 5: q2g_rows<5>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 6: q2g_rows<6>(w, row_bytes, nblocks, a, out, r0, r1); break;
        case 7: q2g_rows<7>(w, row_bytes, nblocks, a, out, r0, r1); break;
        default: q2g_rows<8>(w, row_bytes, nblocks, a, out, r0, r1); break;
    }
}

void k_s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                            ExpertScratchMulti& ws) {
    if (oracle_q8_0.load(std::memory_order_relaxed) || n_tokens < 1 || n_tokens > MAXT) {
        for (int t = 0; t < n_tokens; ++t) k_s2_expert_vnni_q(blob, *a1[t], out[t], ws.single);
        return;
    }
    switch (n_tokens) {
        case 1: expert_multi<1>(blob, a1, out, ws); break;
        case 2: expert_multi<2>(blob, a1, out, ws); break;
        case 3: expert_multi<3>(blob, a1, out, ws); break;
        case 4: expert_multi<4>(blob, a1, out, ws); break;
        case 5: expert_multi<5>(blob, a1, out, ws); break;
        case 6: expert_multi<6>(blob, a1, out, ws); break;
        case 7: expert_multi<7>(blob, a1, out, ws); break;
        default: expert_multi<8>(blob, a1, out, ws); break;
    }
}

void k_unpack64(const uint8_t* codes16, uint8_t* out64) { _mm512_storeu_si512((void*) out64, unpack64_q2_0(codes16)); }
void k_unpack32(const uint8_t* codes8, uint8_t* out32) { _mm256_storeu_si256((__m256i*) out32, unpack_q2_0(codes8)); }

}  // namespace STRATA_EXPERT_NS

const ExpertKernels& STRATA_EXPERT_KERNELS() {
    static const ExpertKernels k = {STRATA_EXPERT_NAME,
                                    &STRATA_EXPERT_NS::k_act_quant_q8_1,
                                    &STRATA_EXPERT_NS::k_s2_expert_vnni_q,
                                    &STRATA_EXPERT_NS::k_s2_expert_gu_rows,
                                    &STRATA_EXPERT_NS::k_s2_expert_down_rows,
                                    &STRATA_EXPERT_NS::k_s2_expert_gu_rows_multi,
                                    &STRATA_EXPERT_NS::k_s2_expert_down_rows_multi,
                                    &STRATA_EXPERT_NS::k_s2_expert_vnni_multi,
                                    &STRATA_EXPERT_NS::k_q2_0_gguf_rows_multi,
                                    &STRATA_EXPERT_NS::k_unpack64,
                                    &STRATA_EXPERT_NS::k_unpack32};
    return k;
}

}  // namespace strata::kernels::cpu
