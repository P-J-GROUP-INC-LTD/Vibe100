// src/ds41/cuda/ds41_math.cuh - DS-D: the arithmetic every DS-D kernel shares: the E8M0 scale, the MXFP4 code -> int8
// decode, the activation quantiser, the vector shuffle-tree reduction.
//
// Attribution.  The MXFP4 decode (lut8 below) follows the technique of llama.cpp's get_int_from_table_16 / vec_dot_mxfp4_q8_1
// (ggml/src/ggml-cuda/vecdotq.cuh, commit 3cf03257f219afbe7334045ff7c6a06ac68c627d): a 16-entry byte table looked up with prmt (3-bit
// index) in two halves merged by the code's sign bit, and the doubled e2m1 table kvalues_fp4 with the E8M0 scale applied in FP32 on the
// int8 dot product (ggml_e8m0_to_fp32_half).  The code here is a rewrite for this layout (inline-PTX prmt without the compiler's
// selector mask, pre-interleaved activations so that the two final byte shuffles disappear), not a copy.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
// FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include "ds41_dev.cuh"
#include "strata/ds41/cuda/ds41_cuda.hpp"

namespace strata::ds41::cuda::dev {

DS41_HD constexpr int ilog2(int v) { return v <= 1 ? 0 : 1 + ilog2(v >> 1); }
DS41_HD constexpr int round16(int v) { return (v + 15) & ~15; }

/// The kernels' copy of act_perm_pos (include/strata/ds41/cuda/ds41_cuda.hpp), usable on the device; they must agree.
DS41_HD constexpr int perm_pos(int j) { return ((j & 15) >> 2) * 8 + (j & 3) * 2 + (j >> 4); }
static_assert(perm_pos(0) == 0 && perm_pos(16) == 1 && perm_pos(1) == 2 && perm_pos(17) == 3 && perm_pos(4) == 8 &&
              perm_pos(19) == 7 && perm_pos(31) == 31);
static_assert(perm_pos(5) == act_perm_pos(5) && perm_pos(27) == act_perm_pos(27) && perm_pos(13) == act_perm_pos(13));

// ---- E8M0 -------------------------------------------------------------------------------------------------------------
/// 2^(e - 128) EXACTLY as ggml_e8m0_to_fp32_half does it (ggml-impl.h): e < 2 -> the denormals 2^-128 (0x00200000) and
/// 2^-127 (0x00400000); otherwise the float with biased exponent e - 1.  e = 255 is 2^127 (finite, NOT NaN).  The doubled
/// e2m1 values times this = the real MXFP4 value, so the integer dot products below are exact and the 1/2 lives here.
DS41_FI float e8m0_half(uint32_t e) {
    const uint32_t norm = (e - 1u) << 23;
    const uint32_t sub = 0x00200000u << (e & 1u);     // e = 0 -> 0x00200000, e = 1 -> 0x00400000 (only used when e < 2)
    return u2f(e >= 2u ? norm : sub);
}

// ---- MXFP4 code -> int8 -----------------------------------------------------------------------------------------------------
// kvalues_fp4 = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12} (the e2m1 table doubled).  Two 8-byte tables, positive and negative
// magnitude, looked up with prmt (3-bit index) and merged by the code's sign bit:
inline constexpr uint32_t kPosLo = 0x03020100u, kPosHi = 0x0C080604u;    // 0,1,2,3 | 4,6,8,12
inline constexpr uint32_t kNegLo = 0xFDFEFF00u, kNegHi = 0xF4F8FAFCu;    // 0,-1,-2,-3 | -4,-6,-8,-12

/// One 32-bit word of qs (8 e2m1 codes, nibbles 0..7 in stream order = low nibble of byte 0, high nibble of byte 0, low of
/// byte 1, ...) -> two int8x4 words: lo4 = values of nibbles 0..3, hi4 = nibbles 4..7, byte i = value of nibble i.
/// 11 instructions for 8 weights.
DS41_FI void lut8(uint32_t q, int& lo4, int& hi4) {
    const uint32_t t = q & 0x77777777u;                                    // 3-bit magnitude index per nibble
    const uint32_t s = ((q >> 1) & 0x44444444u) | 0x32103210u;             // output byte i <- pos[i] or neg[i] (4 + i)
    uint32_t pos = prmt(kPosLo, kPosHi, t);
    uint32_t neg = prmt(kNegLo, kNegHi, t);
    lo4 = (int) prmt(pos, neg, s);
    const uint32_t t2 = t >> 16;
    const uint32_t s2 = s >> 16;
    pos = prmt(kPosLo, kPosHi, t2);
    neg = prmt(kNegLo, kNegHi, t2);
    hi4 = (int) prmt(pos, neg, s2);
}

// ---- activation quantiser (CONTRACTS.md "Activations"; bit-identical to strata::ds41::cpu::quantize_act on EVERY input) -------------
// The rule (ggml's x86 SIMD quantize_row_q8_0 with an FP32 d; the CPU's copy of it is quant_scale() in src/ds41/cpu/mxfp4_internal.hpp):
//   m = the INTEGER maximum of (bits & 0x7FFFFFFF) over the 32 values - order-independent, unlike a float max with a NaN operand;
//   m >= 0x7F800000 (an Inf or a NaN in the block):  d = the canonical NaN 0x7FC00000, every q = 0   (the NaN reaches y)
//   m <  0x0D800000 (amax < 2^-100, zero included):  d = 0, every q = 0                              (127 / amax would overflow)
//   otherwise  amax = float(m), d = amax / 127, id = 127 / amax (two IEEE divisions), q = rint(x * id) (FP32 product, round half to EVEN
//   = cvt.rni), clamped to [-127, 127].
// Nothing here relies on what a conversion does with a NaN or an out-of-range float (the old code reached cvt.rni with x * inf).
inline constexpr uint32_t kQuantInfBits = 0x7F800000u;
inline constexpr uint32_t kQuantTinyBits = 0x0D800000u;     // 2^-100
inline constexpr uint32_t kQuantNaNBits = 0x7FC00000u;

/// One warp = one 32-block: lane l holds element l.  Writes the 32 int8 at `dst` (the block's 32 bytes) and the fp32 scale to *scale_dst
/// (lane 0).  kInterleaved (the default): the dp4a-friendly order act_perm_pos of ds41_cuda.hpp "ACTIVATION AND h LAYOUT" (the hot-expert
/// kernels and the MXFP4 decode); false: natural order, byte j = element j (a Q8_0 GEMV that reads its weights as they are stored).
template <bool kInterleaved = true>
DS41_FI void quantize_block_warp(float v, int lane, int8_t* DS41_RESTRICT dst, float* DS41_RESTRICT scale_dst) {
    int m = (int) (f2u(v) & 0x7FFFFFFFu);                  // magnitude bits, < 2^31: the signed max is the unsigned one
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) {
        const int o = shfl_xor(m, off);
        m = o > m ? o : m;
    }
    const bool special = (uint32_t) m >= kQuantInfBits || (uint32_t) m < kQuantTinyBits;     // every q = 0
    const float amax = u2f((uint32_t) m);
    const float d = (uint32_t) m >= kQuantInfBits ? u2f(kQuantNaNBits) : ((uint32_t) m < kQuantTinyBits ? 0.0f : fdiv_rn(amax, 127.0f));
    const float id = special ? 0.0f : fdiv_rn(127.0f, amax);
    int q = special ? 0 : f2i_rn(fmul_rn(v, id));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    dst[kInterleaved ? perm_pos(lane) : lane] = (int8_t) q;
    if (lane == 0) *scale_dst = d;
}

// ---- vector reduction over a group of lanes ----------------------------------------------------------------------------------
/// V partial sums per lane (V = 1, 2, 4, 8, 16 or 32, a power of two <= LANES) are summed over the LANES lanes of the lane's
/// group (LANES = 16: a half-warp, 32: the warp) with V - 1 + (log2 LANES - log2 V) shuffles instead of log2(LANES) V: each stage
/// keeps half of the values and trades the other half with the partner lane.  Returns this lane's value; it is the total of value
/// index  idx = (lane & (LANES-1)) >> (log2 LANES - log2 V)  (so with V = LANES lane kk holds value kk; with V = 1 every lane holds
/// the total).  Groups are independent.  The order of additions is fixed (deterministic).
template <int LANES, int V>
DS41_FI float lanes_vector_sum(float (&v)[V]) {
    static_assert(V >= 1 && (V & (V - 1)) == 0 && V <= LANES && (LANES == 16 || LANES == 32));
    const int kk = threadIdx.x & (LANES - 1);
    int off = LANES / 2;
    DS41_UNROLL
    for (int n = V; n > 1; n >>= 1, off >>= 1) {
        const bool upper = (kk & off) != 0;                    // this lane keeps the upper half of the n values
        DS41_UNROLL
        for (int i = 0; i < n / 2; ++i) {
            const float send = upper ? v[i] : v[i + n / 2];
            const float keep = upper ? v[i + n / 2] : v[i];
            v[i] = keep + shfl_xor(send, off);
        }
    }
    float r = v[0];
    DS41_UNROLL
    for (; off > 0; off >>= 1) r += shfl_xor(r, off);
    return r;
}
template <int V>
DS41_FI float halfwarp_vector_sum(float (&v)[V]) {
    return lanes_vector_sum<16, V>(v);
}

}  // namespace strata::ds41::cuda::dev
