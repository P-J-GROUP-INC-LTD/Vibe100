// src/ds41/cuda/dense_dev.cuh - DS1-B: the small device helpers the dense kernels share (GEMV, norm, RoPE, vocabulary ops): the half -> float
// conversion, the pinned FP32 arithmetic, the warp butterfly, the natural-order activation quantiser, the total order of the vocabulary ops.
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
// numpy / torch semantics: a NaN is LARGER than every number (so a failure upstream is visible as the argmax / top-1), -0 == +0, ties go to the LOWER index.
// f2key maps a float to a uint32 whose unsigned order is that order (the classic sign-flip of the bit pattern, with the two canonicalisations).
DS41_FI uint32_t f2key(float v) {
    if (v != v) return 0xFFFFFFFFu;                                     // every NaN (any sign / payload) above +inf
    const uint32_t b = dev::f2u(v == 0.0f ? 0.0f : v);                  // -0 -> +0
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
/// Inverse of f2key (a NaN key gives a quiet NaN, a zero key the +0).
DS41_FI float key2f(uint32_t k) { return dev::u2f((k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k); }
/// (key descending, index ascending) is a total order on distinct indices.  The "absent" element is key 0 with index INT_MAX: below every real one.
DS41_FI bool key_better(uint32_t k, int i, uint32_t bk, int bi) { return k > bk || (k == bk && i < bi); }
inline constexpr int kNoIndex = 0x7FFFFFFF;

// ---- more pinned arithmetic ------------------------------------------------------------------------------------------------------------
/// a * b + c with ONE rounding (FFMA).  The emulator's std::fmaf is the same correctly rounded operation, so both builds agree to the bit.
DS41_FI float fma_rn(float a, float b, float c) {
#if defined(DS41_EMU)
    return std::fmaf(a, b, c);
#else
    return __fmaf_rn(a, b, c);
#endif
}
/// IEEE square root (round to nearest): __fsqrt_rn on the device, the host's correctly rounded sqrtf in the emulator.
DS41_FI float sqrt_rn(float a) {
#if defined(DS41_EMU)
    return std::sqrt(a);
#else
    return __fsqrt_rn(a);
#endif
}
/// expf (the accurate one: this project never builds with --use_fast_math); device and host libm may differ in the last bit, which no test pins.
DS41_FI float exp_f(float a) { return expf(a); }
/// Signed byte `i` (0..3) of a packed word, sign-extended: one BFE on the device.
DS41_FI int sext8(uint32_t w, int i) { return (int) (int8_t) (uint8_t) ((w >> (8 * i)) & 0xFFu); }
/// SwiGLU's sigmoid of ops.py (the split form: no overflow, a NaN stays a NaN): e = exp(-|x|); x >= 0 ? 1 / (1 + e) : e / (1 + e).
DS41_FI float sigmoid_split(float x) {
    const float e = exp_f(-fabsf(x));
    const float d = dev::fadd_rn(1.0f, e);
    return x >= 0.0f ? dev::fdiv_rn(1.0f, d) : dev::fdiv_rn(e, d);
}

// ---- loads and stores with the alignment the access needs (the emulator asserts it) ---------------------------------------------------------
/// A 16-bit global load (the Q8_0 block scale and quants are only 2-byte aligned: 34-byte blocks).
DS41_FI uint32_t ldg16(const void* p) {
#if defined(DS41_EMU)
    dev::emu_check_align(p, 2, "ldg16 (LDG.U16)");
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
#else
    return (uint32_t) __ldg(reinterpret_cast<const unsigned short*>(p));
#endif
}
/// Plain (coherent) 16-byte load / store: for data a kernel may overwrite itself (in-place norm / RoPE) and for shared memory.
DS41_FI uint4 ld16b(const void* p) {
    DS41_ASSERT_ALIGNED(p, 16);
    return *reinterpret_cast<const uint4*>(p);
}
DS41_FI void st16b(void* p, uint4 v) {
    DS41_ASSERT_ALIGNED(p, 16);
    *reinterpret_cast<uint4*>(p) = v;
}
DS41_FI float4 ldf4(const void* p) {
    DS41_ASSERT_ALIGNED(p, 16);
    return *reinterpret_cast<const float4*>(p);
}
DS41_FI void stf4(void* p, float4 v) {
    DS41_ASSERT_ALIGNED(p, 16);
    *reinterpret_cast<float4*>(p) = v;
}

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
