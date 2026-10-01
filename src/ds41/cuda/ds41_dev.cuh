// src/ds41/cuda/ds41_dev.cuh - DS-D: the thin layer between the kernels and the compiler that builds them.
//
// The kernels in this directory are written once and compiled twice: by nvcc for sm_70 (the real thing), and by the host
// compiler with -DDS41_EMU against ds41_emu.hpp (a CPU emulation of the CUDA thread model, so the logic of every kernel can
// be run and checked on a machine with no GPU).  Everything that differs between the two is behind the names below; the
// kernel bodies use only these names for intrinsics.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(DS41_EMU)
#include "ds41_emu.hpp"
#define DS41_KERNEL inline
#define DS41_HD
#define DS41_FI inline
#define DS41_LAUNCH_BOUNDS(...)
#define DS41_SHARED static
#define DS41_UNROLL
#define DS41_UNROLL1
#define DS41_RESTRICT __restrict__
#define DS41_DYN_SMEM(name) unsigned char* name = (unsigned char*) ds41_emu::dyn_smem()
#else
#include <cuda_runtime.h>

#include "strata/kernels/dp4a.hpp"
#define DS41_KERNEL __global__
#define DS41_HD __host__ __device__
#define DS41_FI __device__ __forceinline__
#define DS41_LAUNCH_BOUNDS(...) __launch_bounds__(__VA_ARGS__)
#define DS41_SHARED __shared__
#define DS41_UNROLL _Pragma("unroll")
#define DS41_UNROLL1 _Pragma("unroll 1")
#define DS41_RESTRICT __restrict__
#define DS41_DYN_SMEM(name) extern __shared__ __align__(16) unsigned char name[]
#endif

namespace strata::ds41::cuda::dev {

// ---- byte permute (PTX prmt.b32, generic mode) ------------------------------------------------------------------------
// result byte i = byte (sel nibble i & 7) of the 8-byte value {b : a} (a = bytes 0..3, b = bytes 4..7); if the nibble's
// bit 3 is set the byte is instead the replicated sign bit of the selected byte.  The kernels never set bit 3.
// (Not __byte_perm: the compiler masks its selector with 0x7777 first, one extra instruction per use.)
DS41_FI uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
#if defined(DS41_EMU)
    const uint64_t v = ((uint64_t) b << 32) | a;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t s = (sel >> (4 * i)) & 0xF;
        uint32_t byte = (uint32_t) ((v >> (8 * (s & 7))) & 0xFF);
        if (s & 8) byte = (byte & 0x80) ? 0xFF : 0x00;
        r |= byte << (8 * i);
    }
    return r;
#else
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
#endif
}

/// Four signed bytes of a times four signed bytes of b, summed into c.
DS41_FI int dp4a(int a, int b, int c) {
#if defined(DS41_EMU)
    int s = c;
    for (int i = 0; i < 4; ++i) s += (int) (int8_t) ((a >> (8 * i)) & 0xFF) * (int) (int8_t) ((b >> (8 * i)) & 0xFF);
    return s;
#else
    return STRATA_DP4A(a, b, c);
#endif
}

// ---- bit casts ---------------------------------------------------------------------------------------------------------
DS41_FI float u2f(uint32_t u) {
#if defined(DS41_EMU)
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#else
    return __uint_as_float(u);
#endif
}
DS41_FI uint32_t f2u(float f) {
#if defined(DS41_EMU)
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#else
    return __float_as_uint(f);
#endif
}

// ---- IEEE-pinned arithmetic (no FMA contraction, no fast division) -----------------------------------------------------
DS41_FI float fmul_rn(float a, float b) {
#if defined(DS41_EMU)
    volatile float r = a * b;
    return r;
#else
    return __fmul_rn(a, b);
#endif
}
DS41_FI float fadd_rn(float a, float b) {
#if defined(DS41_EMU)
    volatile float r = a + b;
    return r;
#else
    return __fadd_rn(a, b);
#endif
}
DS41_FI float fdiv_rn(float a, float b) {
#if defined(DS41_EMU)
    volatile float r = a / b;
    return r;
#else
    return __fdiv_rn(a, b);
#endif
}
/// float -> int, round to nearest EVEN (cvt.rni), the CPU's lrintf in the default rounding mode.
DS41_FI int f2i_rn(float a) {
#if defined(DS41_EMU)
    return (int) std::lrintf(a);
#else
    return __float2int_rn(a);
#endif
}

// ---- warp / block synchronisation and exchange (always the full warp) -----------------------------------------------------
DS41_FI void sync_block() {
#if defined(DS41_EMU)
    ds41_emu::sync_block();
#else
    __syncthreads();
#endif
}
DS41_FI void sync_warp() {
#if defined(DS41_EMU)
    ds41_emu::sync_warp();
#else
    __syncwarp();
#endif
}
DS41_FI float shfl_xor(float v, int m) {
#if defined(DS41_EMU)
    return ds41_emu::shfl_xor<float>(v, m);
#else
    return __shfl_xor_sync(0xffffffffu, v, m);
#endif
}
DS41_FI int shfl_xor(int v, int m) {
#if defined(DS41_EMU)
    return ds41_emu::shfl_xor<int>(v, m);
#else
    return __shfl_xor_sync(0xffffffffu, v, m);
#endif
}
DS41_FI float shfl(float v, int src) {
#if defined(DS41_EMU)
    return ds41_emu::shfl<float>(v, src);
#else
    return __shfl_sync(0xffffffffu, v, src);
#endif
}
DS41_FI int shfl(int v, int src) {
#if defined(DS41_EMU)
    return ds41_emu::shfl<int>(v, src);
#else
    return __shfl_sync(0xffffffffu, v, src);
#endif
}
DS41_FI uint32_t ballot(bool p) {
#if defined(DS41_EMU)
    return ds41_emu::ballot(p);
#else
    return __ballot_sync(0xffffffffu, p);
#endif
}
DS41_FI int popc(uint32_t v) {
#if defined(DS41_EMU)
    return __builtin_popcount(v);
#else
    return __popc(v);
#endif
}
DS41_FI void threadfence_system() {
#if defined(DS41_EMU)
#else
    __threadfence_system();
#endif
}

// ---- global loads -----------------------------------------------------------------------------------------------------
// Read-only data (weights, activations, tables): the non-coherent path.  The blob bytes are never written while a kernel
// that reads them runs (the caller orders cache fills on the same stream).
DS41_FI uint4 ldg4(const void* p) {
#if defined(DS41_EMU)
    uint4 v;
    std::memcpy(&v, p, 16);
    return v;
#else
    return __ldg(reinterpret_cast<const uint4*>(p));
#endif
}
DS41_FI uint2 ldg2(const void* p) {
#if defined(DS41_EMU)
    uint2 v;
    std::memcpy(&v, p, 8);
    return v;
#else
    return __ldg(reinterpret_cast<const uint2*>(p));
#endif
}
DS41_FI float4 ldgf4(const void* p) {
#if defined(DS41_EMU)
    float4 v;
    std::memcpy(&v, p, 16);
    return v;
#else
    return __ldg(reinterpret_cast<const float4*>(p));
#endif
}
DS41_FI float ldgf(const void* p) {
#if defined(DS41_EMU)
    return *reinterpret_cast<const float*>(p);
#else
    return __ldg(reinterpret_cast<const float*>(p));
#endif
}

// ---- small helpers shared by the kernels ----------------------------------------------------------------------------------
DS41_FI int lane_id() {
#if defined(DS41_EMU)
    return threadIdx.x & 31;
#else
    return threadIdx.x & 31;
#endif
}

}  // namespace strata::ds41::cuda::dev
