// src/ds41/cuda/dense_dev.cuh - DS1-B: the small device helpers the dense kernels share (GEMV, norm, RoPE, vocabulary ops).
//
// The dense kernels are written once and compiled twice, as DS-D's are (ds41_dev.cuh): by nvcc for sm_70 and by the host compiler with
// -DDS41_EMU against the CPU emulation of the thread model (ds41_emu.hpp).  They use only the names of ds41_dev.cuh for intrinsics.
// This header deliberately includes ONLY ds41_dev.cuh (not ds41_math.cuh / ds41_cuda.hpp): the dense package owns its copy of the few
// arithmetic rules it needs (the activation quantiser, the butterfly), pinned by its own tests, so it does not move when DS-D's headers do.
#pragma once

#include <cstdint>

#include "ds41_dev.cuh"

namespace strata::ds41::cuda::dense {

inline constexpr int kNW = 8;                       // warps per block of the GEMV kernels (and of the batched row kernels)
inline constexpr int kThreads = 32 * kNW;           // 256

// ---- IEEE half -> float (the d of a Q8_0 block) ------------------------------------------------------------------------------------
/// Exact for every half bit pattern (zero, subnormals, Inf, NaN).  nvcc: one cvt.f32.f16; emulator: the same function in integer arithmetic.
DS41_FI float f16_to_f32(uint32_t h) {
#if defined(DS41_EMU)
    const uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t b;
    if (e == 0) {
        if (m == 0) {
            b = s;
        } else {                                         // subnormal: m * 2^-24, normalised
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 0x400u)) {
                mm <<= 1;
                ++sh;
            }
            b = s | ((uint32_t) (113 - sh) << 23) | ((mm & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        b = s | 0x7F800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    return dev::u2f(b);
#else
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"((unsigned short) h));
    return f;
#endif
}

// ---- the warp butterfly ---------------------------------------------------------------------------------------------------------------
/// Sum over the 32 lanes in the fixed xor-butterfly order (offsets 16, 8, 4, 2, 1); every lane ends with the same bits.  Float addition is
/// commutative, so lane l computes the same bits as its partner: the tree is a function of the lane layout only.
DS41_FI float warp_sum(float v) {
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) v += dev::shfl_xor(v, off);
    return v;
}

// ---- the activation quantiser, NATURAL byte order ---------------------------------------------------------------------------------------
// docs/deepseek/CONTRACTS.md "Activations" (ggml's x86 SIMD quantize_row_q8_0 with an FP32 d), bit for bit the rule of ds41_math.cuh's
// quantize_block_warp<false> (DS-D / DS1-G): m = the INTEGER maximum of (bits & 0x7FFFFFFF) over the 32 values;
//   m >= 0x7F800000 (an Inf or a NaN in the block):  d = the canonical NaN 0x7FC00000, every q = 0   (the NaN reaches y)
//   m <  0x0D800000 (amax < 2^-100, zero included):  d = 0, every q = 0
//   otherwise  amax = float(m), d = amax / 127, id = 127 / amax (two IEEE divisions), q = rint(x * id) (FP32 product, ties to EVEN), +-127.
// This copy is the "small local adapter" for ds41_quantize_acts<G>(..., ActOrder::kNatural): swap the call in dense_impl.cuh when that lands.
inline constexpr uint32_t kQuantInfBits = 0x7F800000u;
inline constexpr uint32_t kQuantTinyBits = 0x0D800000u;
inline constexpr uint32_t kQuantNaNBits = 0x7FC00000u;

/// One warp = one 32-block, lane l holds element l.  Writes the 32 int8 at `dst` (natural order: byte j = element j) and the scale (lane 0).
DS41_FI void quantize_block_nat(float v, int lane, int8_t* DS41_RESTRICT dst, float* DS41_RESTRICT scale_dst) {
    int m = (int) (dev::f2u(v) & 0x7FFFFFFFu);             // magnitude bits < 2^31: the signed maximum is the unsigned one
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) {
        const int o = dev::shfl_xor(m, off);
        m = o > m ? o : m;
    }
    const bool nonfinite = (uint32_t) m >= kQuantInfBits;
    const bool special = nonfinite || (uint32_t) m < kQuantTinyBits;
    const float amax = dev::u2f((uint32_t) m);
    const float d = nonfinite ? dev::u2f(kQuantNaNBits) : ((uint32_t) m < kQuantTinyBits ? 0.0f : dev::fdiv_rn(amax, 127.0f));
    const float id = special ? 0.0f : dev::fdiv_rn(127.0f, amax);
    int q = special ? 0 : dev::f2i_rn(dev::fmul_rn(v, id));
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    dst[lane] = (int8_t) q;
    if (lane == 0) *scale_dst = d;
}

// ---- a total order for the vocabulary ops -----------------------------------------------------------------------------------------------
/// (value descending, index ascending) is a total order on distinct indices; a NaN value must be mapped to -inf by the caller.
DS41_FI bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

// ---- the card ---------------------------------------------------------------------------------------------------------------------------
/// Streaming multiprocessors of the current device (the launch heuristics only: how many rows one block takes).  80 in the emulation.
inline int sm_count() {
#if defined(DS41_EMU)
    return 80;
#else
    static const int n = [] {
        int d = 0, v = 80;
        if (cudaGetDevice(&d) == cudaSuccess && cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d) == cudaSuccess && v > 0) return v;
        (void) cudaGetLastError();
        return 80;
    }();
    return n;
#endif
}

}  // namespace strata::ds41::cuda::dense
