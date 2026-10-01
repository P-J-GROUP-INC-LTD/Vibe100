// src/ds41/cuda/ds41_dev.cuh - DS-D: the thin layer between the kernels and the compiler that builds them.
//
// The kernels in this directory are written once and compiled twice: by nvcc for sm_70 (the real thing), and by the host
// compiler with -DDS41_EMU against ds41_emu.hpp (a CPU emulation of the CUDA thread model, so the logic of every kernel can
// be run and checked on a machine with no GPU).  Everything that differs between the two is behind the names below; the
// kernel bodies use only these names for intrinsics.
#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(DS41_EMU)
#include "ds41_emu.hpp"
#define DS41_KERNEL inline
#define DS41_HD
#define DS41_FI inline
#define DS41_LAUNCH_BOUNDS(...)
// static __shared__ arrays: statics in the named section ds41_smem, which the emulator fills with 0xFF at the start of every block (ds41_emu.hpp)
#if defined(__GNUC__) && defined(__linux__)
#define DS41_SHARED static __attribute__((section("ds41_smem")))
#else
#define DS41_SHARED static
#endif
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
/// float -> int, round to nearest EVEN (cvt.rni.s32.f32), the CPU's lrintf in the default rounding mode.  NaN -> 0 and out-of-range values
/// SATURATE to INT_MIN / INT_MAX on the device; the emulator does the same (a bare lrintf of a NaN is undefined / INT_MIN on the host,
/// which would make the emulator disagree with the hardware).  The quantiser does not rely on either: it never converts a non-finite value.
DS41_FI int f2i_rn(float a) {
#if defined(DS41_EMU)
    if (a != a) return 0;
    if (a >= 2147483648.0f) return 2147483647;
    if (a <= -2147483648.0f) return (int) 0x80000000u;
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
/// `fence.acq_rel.sys`: orders this thread's earlier writes (and, after a __syncthreads, those of the block it synchronised with: PTX
/// release patterns are cumulative) before its later ones, at SYSTEM scope - what the host sees through mapped memory.  One thread issues
/// it, not every thread (the old __threadfence_system() in every thread was fence.sc.sys each).  The emulator runs one thread of execution:
/// a C++ fence.
DS41_FI void fence_acq_rel_sys() {
#if defined(DS41_EMU)
    std::atomic_thread_fence(std::memory_order_acq_rel);
#else
    asm volatile("fence.acq_rel.sys;" ::: "memory");
#endif
}
/// `st.release.sys.global.u32`: a 32-bit store with RELEASE semantics at system scope - every write the storing thread (and the block
/// it synchronised with) made before it is visible to a host thread that reads the value with an acquire load.  The doorbell of a
/// host-visible record (sm_70+: the .release qualifier needs PTX 6.0 / Volta).  In the emulator: a plain store behind a release fence.
DS41_FI void store_release_sys(uint32_t* p, uint32_t v) {
#if defined(DS41_EMU)
    std::atomic_thread_fence(std::memory_order_release);
    *reinterpret_cast<volatile uint32_t*>(p) = v;
#else
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(__cvta_generic_to_global(p)), "r"(v) : "memory");
#endif
}

// ---- alignment (emulator only) ----------------------------------------------------------------------------------------
// A vector load or store (ld.global.v4 / ld.shared.v2 ...) whose address is not a multiple of its size FAULTS on the hardware
// ("misaligned address"); the host memcpy the emulator would otherwise use accepts it silently.  Every vector access in the kernels
// goes through ldg* below or through DS41_ASSERT_ALIGNED, and the emulator aborts with the address instead.
#if defined(DS41_EMU)
inline void emu_check_align(const void* p, unsigned bytes, const char* what) {
    if (reinterpret_cast<uintptr_t>(p) % bytes != 0) {
        std::fprintf(stderr, "ds41_emu: MISALIGNED %u-byte %s at %p (it faults on a GPU)\n", bytes, what, p);
        std::abort();
    }
}
#define DS41_ASSERT_ALIGNED(p, bytes) ::strata::ds41::cuda::dev::emu_check_align((p), (bytes), "vector access")
#else
#define DS41_ASSERT_ALIGNED(p, bytes) ((void) 0)
#endif

// ---- global loads -----------------------------------------------------------------------------------------------------
// Read-only data (weights, activations, tables): the non-coherent path.  The blob bytes are never written while a kernel
// that reads them runs (the caller orders cache fills on the same stream).
DS41_FI uint4 ldg4(const void* p) {
#if defined(DS41_EMU)
    emu_check_align(p, 16, "ldg4 (LDG.E.128)");
    uint4 v;
    std::memcpy(&v, p, 16);
    return v;
#else
    return __ldg(reinterpret_cast<const uint4*>(p));
#endif
}
DS41_FI uint2 ldg2(const void* p) {
#if defined(DS41_EMU)
    emu_check_align(p, 8, "ldg2 (LDG.E.64)");
    uint2 v;
    std::memcpy(&v, p, 8);
    return v;
#else
    return __ldg(reinterpret_cast<const uint2*>(p));
#endif
}
DS41_FI float4 ldgf4(const void* p) {
#if defined(DS41_EMU)
    emu_check_align(p, 16, "ldgf4 (LDG.E.128)");
    float4 v;
    std::memcpy(&v, p, 16);
    return v;
#else
    return __ldg(reinterpret_cast<const float4*>(p));
#endif
}
DS41_FI float ldgf(const void* p) {
#if defined(DS41_EMU)
    emu_check_align(p, 4, "ldgf (LDG.E.32)");
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
