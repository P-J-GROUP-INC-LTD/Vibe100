// src/ds41/cpu/mxfp4_internal.hpp - DS-C: what the per-ISA translation units hand to the portable one.
//
// The portable file (mxfp4_expert.cpp) owns everything that is not an inner loop: the phases, the row ranges, the
// SwiGLU, CPUID, the scalar reference.  The AVX2 and AVX-512 files each export one table of the two functions that
// are compiled with their own ISA flags.  Keeping the split here means the SwiGLU and the phase logic exist ONCE,
// so the three ISAs cannot disagree about anything but the last bits of an FP32 sum.
#pragma once

#include "strata/ds41/cpu/mxfp4_expert.hpp"

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

struct Kernels {
    DotRowsFn dot_rows;
    QuantBlockFn quantize_block;
    const char* name;
};

const Kernels& kernels_scalar();
const Kernels* kernels_avx2();      // nullptr on a build that has no x86 code
const Kernels* kernels_avx512();

}  // namespace strata::ds41::cpu::detail
