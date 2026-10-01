// src/ds41/cuda/ds41_emu.hpp - DS-D: a CPU emulation of the CUDA execution model, just big enough to run the DS-D
// kernels (src/ds41/cuda/*.cuh) on a machine WITHOUT a GPU.
//
// WHY.  The V100 parity programs can only run on the user's box, and the expert kernels are intricate (17-byte blocks read
// through shared memory, a lane-constant byte permute, a shuffle-tree reduction, a quantised-h hand-off between two
// launches).  This header lets the SAME kernel source be compiled by the host compiler (-DDS41_EMU, no nvcc) and run
// block by block, thread by thread, against the FP64 references.  That checks the logic - indexing, layouts, reductions,
// arithmetic - not the performance, not the sm_70 code generation, not the intrinsics' hardware behaviour (each
// intrinsic is re-implemented here from the PTX / CUDA documentation).
//
// HOW.  Every CUDA thread is a fiber (ucontext) with its own stack; one OS thread runs them round robin.  A fiber yields
// only at __syncthreads / __syncwarp / a warp shuffle, so execution is deterministic.  Shared memory is a plain static
// (`DS41_SHARED` = `static`) or the launch's dynamic buffer; blocks run one after the other.  Threads that `return` early
// leave the barriers (a block-uniform early exit, which is all the kernels do).  Deadlock = a diagnostic + abort.
#pragma once

#include <ucontext.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

// ---- the CUDA vector / dim types the kernels use (global names, like CUDA) ---------------------------------------------
struct uint3 {
    unsigned x, y, z;
};
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct alignas(8) uint2 {
    unsigned x, y;
};
struct alignas(16) uint4 {
    unsigned x, y, z, w;
};
struct alignas(16) float4 {
    float x, y, z, w;
};
inline uint4 make_uint4(unsigned x, unsigned y, unsigned z, unsigned w) { return uint4{x, y, z, w}; }
inline uint2 make_uint2(unsigned x, unsigned y) { return uint2{x, y}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }

namespace ds41_emu {

struct Fiber;
struct Ctx {
    uint3 threadIdx{0, 0, 0}, blockIdx{0, 0, 0}, blockDim{1, 1, 1}, gridDim{1, 1, 1};
    Fiber* fiber = nullptr;
};
enum State { kRunnable = 0, kWaitBlock = 1, kWaitWarp = 2, kDone = 3 };
struct Fiber {
    ucontext_t uctx;
    std::vector<char> stack;
    State state = kRunnable;
    Ctx ctx;
    int tid = 0;
    uint32_t slot = 0;   // the value this lane offers to the warp's current exchange
};

inline Ctx* g_cur = nullptr;                        // the running thread's context
inline ucontext_t g_main;                           // the scheduler
inline std::vector<Fiber>* g_fibers = nullptr;
inline const std::function<void()>* g_kernel = nullptr;
inline unsigned char* g_dyn_smem = nullptr;
inline size_t g_launch_count = 0, g_block_count = 0;

inline void yield(State s) {
    Fiber* f = g_cur->fiber;
    f->state = s;
    swapcontext(&f->uctx, &g_main);
}
inline void fiber_entry() {
    (*g_kernel)();
    Fiber* f = g_cur->fiber;
    f->state = kDone;
    swapcontext(&f->uctx, &g_main);
}

inline void sync_block() { yield(kWaitBlock); }
inline void sync_warp() { yield(kWaitWarp); }

inline int lane_id() { return g_cur->fiber->tid & 31; }
inline std::vector<Fiber>& fibers() { return *g_fibers; }

template <class T>
inline uint32_t to_bits(T v) {
    static_assert(sizeof(T) == 4);
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}
template <class T>
inline T from_bits(uint32_t b) {
    T v;
    std::memcpy(&v, &b, 4);
    return v;
}
/// Every live lane of the warp must call this together (the kernels' shuffles are always full-warp).
template <class T>
inline T shfl_xor(T v, int mask) {
    Fiber* f = g_cur->fiber;
    f->slot = to_bits(v);
    sync_warp();
    const int src = (f->tid & 31) ^ mask;
    Fiber& s = fibers()[(f->tid & ~31) + src];
    const T r = from_bits<T>(s.slot);
    sync_warp();
    return r;
}
template <class T>
inline T shfl(T v, int src_lane) {
    Fiber* f = g_cur->fiber;
    f->slot = to_bits(v);
    sync_warp();
    Fiber& s = fibers()[(f->tid & ~31) + (src_lane & 31)];
    const T r = from_bits<T>(s.slot);
    sync_warp();
    return r;
}
inline uint32_t ballot(bool p) {
    Fiber* f = g_cur->fiber;
    f->slot = p ? 1u : 0u;
    sync_warp();
    uint32_t m = 0;
    const int base = f->tid & ~31;
    for (int l = 0; l < 32 && base + l < (int) fibers().size(); ++l) m |= (fibers()[base + l].slot & 1u) << l;
    sync_warp();
    return m;
}
inline void* dyn_smem() { return g_dyn_smem; }

/// Run `kernel` (the body every thread executes) over grid x block.  `smem_bytes` of dynamic shared memory.
inline void launch(dim3 grid, dim3 block, size_t smem_bytes, const std::function<void()>& kernel) {
    const int nthreads = (int) (block.x * block.y * block.z);
    std::vector<Fiber> fibers(nthreads);
    const size_t kStack = 256 * 1024;
    for (auto& f : fibers) f.stack.resize(kStack);
    g_fibers = &fibers;
    g_kernel = &kernel;
    constexpr size_t kGuard = 256;
    std::vector<unsigned char> smem(smem_bytes + 2 * kGuard + 16, 0xCD);
    unsigned char* dyn = smem.data() + kGuard;
    dyn = (unsigned char*) (((uintptr_t) dyn + 15) & ~(uintptr_t) 15);
    g_dyn_smem = dyn;
    ++g_launch_count;
    for (unsigned bz = 0; bz < grid.z; ++bz)
        for (unsigned by = 0; by < grid.y; ++by)
            for (unsigned bx = 0; bx < grid.x; ++bx) {
                ++g_block_count;
                std::memset(dyn, 0xFF, smem_bytes);          // uninitialised shared memory = NaN bits
                for (int t = 0; t < nthreads; ++t) {
                    Fiber& f = fibers[t];
                    f.tid = t;
                    f.state = kRunnable;
                    f.ctx.fiber = &f;
                    f.ctx.threadIdx = uint3{(unsigned) (t % block.x), (unsigned) ((t / block.x) % block.y),
                                            (unsigned) (t / (block.x * block.y))};
                    f.ctx.blockIdx = uint3{bx, by, bz};
                    f.ctx.blockDim = uint3{block.x, block.y, block.z};
                    f.ctx.gridDim = uint3{grid.x, grid.y, grid.z};
                    getcontext(&f.uctx);
                    f.uctx.uc_stack.ss_sp = f.stack.data();
                    f.uctx.uc_stack.ss_size = kStack;
                    f.uctx.uc_link = &g_main;
                    makecontext(&f.uctx, (void (*)()) fiber_entry, 0);
                }
                for (;;) {
                    int ran = 0;
                    for (auto& f : fibers)
                        if (f.state == kRunnable) {
                            g_cur = &f.ctx;
                            swapcontext(&g_main, &f.uctx);
                            ++ran;
                        }
                    int live = 0, wait_block = 0;
                    for (auto& f : fibers) {
                        live += f.state != kDone;
                        wait_block += f.state == kWaitBlock;
                    }
                    if (live == 0) break;
                    bool released = false;
                    if (wait_block == live) {
                        for (auto& f : fibers) f.state = kRunnable;
                        released = true;
                    } else {
                        for (int w = 0; w * 32 < nthreads; ++w) {
                            int wl = 0, ww = 0;
                            for (int l = 0; l < 32 && w * 32 + l < nthreads; ++l) {
                                const State s = fibers[w * 32 + l].state;
                                wl += s != kDone;
                                ww += s == kWaitWarp;
                            }
                            if (wl > 0 && ww == wl) {
                                for (int l = 0; l < 32 && w * 32 + l < nthreads; ++l)
                                    if (fibers[w * 32 + l].state == kWaitWarp) fibers[w * 32 + l].state = kRunnable;
                                released = true;
                            }
                        }
                    }
                    if (!ran && !released) {
                        std::fprintf(stderr, "ds41_emu: DEADLOCK in block (%u,%u,%u): a barrier that not every thread reaches\n", bx, by,
                                     bz);
                        std::abort();
                    }
                }
            }
    // shared-memory guard bytes: a store past either end of the dynamic buffer is a bug in the kernel
    for (unsigned char* q = smem.data(); q < dyn; ++q)
        if (*q != 0xCD) {
            std::fprintf(stderr, "ds41_emu: dynamic shared memory underflow (%ld bytes before the buffer)\n", (long) (dyn - q));
            std::abort();
        }
    for (unsigned char* q = dyn + smem_bytes; q < smem.data() + smem.size(); ++q)
        if (*q != 0xCD) {
            std::fprintf(stderr, "ds41_emu: dynamic shared memory overflow (%ld bytes past the launch's size)\n", (long) (q - (dyn + smem_bytes)) + 1);
            std::abort();
        }
    g_fibers = nullptr;
    g_kernel = nullptr;
}

}  // namespace ds41_emu

#define threadIdx (ds41_emu::g_cur->threadIdx)
#define blockIdx (ds41_emu::g_cur->blockIdx)
#define blockDim (ds41_emu::g_cur->blockDim)
#define gridDim (ds41_emu::g_cur->gridDim)
