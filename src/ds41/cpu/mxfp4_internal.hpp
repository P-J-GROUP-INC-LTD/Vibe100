// src/ds41/cpu/mxfp4_internal.hpp - DS-C: what the per-ISA translation units hand to the portable one.
//
// The portable file (mxfp4_expert.cpp) owns everything that is not an inner loop: the phases, the row ranges, the
// SwiGLU, CPUID, the scalar reference.  The AVX2 and AVX-512 files each export one table of the two functions that
// are compiled with their own ISA flags.  Keeping the split here means the SwiGLU and the phase logic exist ONCE,
// so the three ISAs cannot disagree about anything but the last bits of an FP32 sum.
#pragma once

#include "strata/ds41/cpu/mxfp4_expert.hpp"

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

// The cache level software prefetches target: T1 = L2 (measured on the dev VM: as fast as T0 and it cannot evict the activation
// from L1; build-time knob for the Xeon: -DDS41_PF_HINT=_MM_HINT_T0).
#ifndef DS41_PF_HINT
#define DS41_PF_HINT _MM_HINT_T1
#endif

namespace strata::ds41::cpu::detail {

#if defined(__x86_64__) || defined(_M_X64)
// One weight row is a sequential stream and rows of a matrix are contiguous (stride == row bytes) in every layout this
// code reads in production, so "pf bytes ahead of the group being decoded" simply runs on into the next row.  The two
// lines per group are because a group is 68 bytes: one prefetch per group would skip a line every 16 groups.  The
// address is clamped to `limit` (the last byte the CALL will read) so the end of a range does not prefetch into
// another thread's rows; for a strided layout (a half of a GPU blob: its down rows have a 612-byte gap) the limit is
// the end of the row.  A prefetch never faults, so the clamp is about bandwidth, not safety.
inline void prefetch_ahead(const uint8_t* p, int pf, const uint8_t* limit) {
    const uint8_t* a = p + pf;
    const uint8_t* b = p + pf + 64;
    a = a < limit ? a : limit - 1;
    b = b < limit ? b : limit - 1;
    _mm_prefetch((const char*) a, DS41_PF_HINT);
    _mm_prefetch((const char*) b, DS41_PF_HINT);
}
// The first `bytes` of a call's range, before its first tile: the hardware prefetcher needs a few misses to find a stream and
// every call starts a new one.
inline void prefetch_prologue(const uint8_t* w, size_t bytes) {
    for (size_t o = 0; o < bytes; o += 64) _mm_prefetch((const char*) (w + o), DS41_PF_HINT);
}
#endif

/// out[t * out_stride + r] (= or +=) the dot of weight row r with x[t]; `ng` = groups (4 blocks) per row;
/// `pf` = software prefetch distance in bytes (0 = none).  Contract of mxfp4_dot_rows without the checks.
using DotRowsFn = void (*)(const uint8_t* w, size_t row_stride, int ng, int nrows, const ActQ* x, int T, float* out,
                           size_t out_stride, bool accumulate, int pf);

/// Quantise 32 floats into block `blk` of `a` (q, sc4, corr, scale), bit-identical to the scalar version.
using QuantBlockFn = void (*)(const float* x32, ActQ& a, int blk);

// ---- the activation quantiser's rule, in ONE place (docs/deepseek/CONTRACTS.md "Activations") ------------------------------
// Every implementation (scalar, AVX2, AVX-512; the GPU kernel and the oracle say the same thing in their own languages)
// finds the block's largest magnitude as an INTEGER maximum of `bits & 0x7FFFFFFF` (order-independent, unlike a float max
// with a NaN operand; for finite values the integer order of the magnitude bits is the float order), and then calls
// quant_scale() on it; only the codes q = rint(x * id) differ in how they are computed (a loop, 8 or 16 lanes at a time).
//   bits >= 0x7F800000      the block holds an Inf or a NaN: d = the canonical quiet NaN 0x7FC00000 (a pinned payload, so the
//                           bytes are the same everywhere), every q = 0 - the NaN reaches y instead of being laundered
//   bits <  2^-100          (zero and every denormal included; 127 / amax would overflow below ~3.7e-37): d = 0, every q = 0
//   otherwise               d = amax / 127 and id = 127 / amax, two FP32 divisions; q = rint(x * id) (FP32 product, round half
//                           to EVEN: lrintf / vcvtps2dq in the default rounding mode), clamped to [-127, 127] (the clamp is
//                           provably dead - |x * id| <= 127 (1 + 2^-23) - but it is part of the rule; vpmovsdb saturates at -128 only)
inline constexpr uint32_t kQuantInfBits = 0x7F800000u;
inline constexpr uint32_t kQuantTinyBits = 0x0D800000u;     // 2^-100: exponent field 127 - 100 = 27
inline constexpr uint32_t kQuantNaNBits = 0x7FC00000u;
struct QuantScale {
    float d;          // the block's scale
    float id;         // 127 / amax (0 when `zero`)
    bool zero;        // every q is 0 (all-zero / tiny / non-finite block)
};
inline float quant_bits_to_float(uint32_t b) {
    float f;
    std::memcpy(&f, &b, sizeof f);
    return f;
}
inline QuantScale quant_scale(uint32_t max_mag_bits) {
    QuantScale s;
    if (max_mag_bits >= kQuantInfBits) {
        s.d = quant_bits_to_float(kQuantNaNBits);
        s.id = 0.0f;
        s.zero = true;
    } else if (max_mag_bits < kQuantTinyBits) {
        s.d = 0.0f;
        s.id = 0.0f;
        s.zero = true;
    } else {
        const float amax = quant_bits_to_float(max_mag_bits);
        s.d = amax / 127.0f;
        s.id = 127.0f / amax;
        s.zero = false;
    }
    return s;
}

struct Kernels {
    DotRowsFn dot_rows;
    QuantBlockFn quantize_block;
    const char* name;
};

const Kernels& kernels_scalar();
const Kernels* kernels_avx2();      // nullptr on a build that has no x86 code
const Kernels* kernels_avx512();

}  // namespace strata::ds41::cpu::detail
