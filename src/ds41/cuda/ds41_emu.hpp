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
// (`DS41_SHARED` = a static in the section `ds41_smem`) or the launch's dynamic buffer; blocks run one after the other.  Threads that
// `return` early leave the barriers (a block-uniform early exit, which is all the kernels do).  Deadlock = a diagnostic + abort.
//
// WHAT IT CATCHES THAT A PLAIN RUN WOULD NOT (each is a bug the hardware punishes and a host run forgives):
//   * uninitialised shared memory: the dynamic buffer AND every static `__shared__` array are filled with 0xFF (float NaN, int -1) at the
//     start of EVERY block, so a read before the write shows up as NaN / garbage in the results;
//   * misaligned vector accesses: ldg4 / ldg2 / ldgf4 / ldgf and DS41_ASSERT_ALIGNED abort on an address that is not a multiple of the access
//     size (the GPU faults);
//   * a warp vote or shuffle that involves a lane that has already exited (ballot only counts the lanes still active; a shuffle FROM an
//     exited lane aborts: its register is undefined on the hardware);
//   * ordering assumptions: the scheduling order of the fibers of a block (and of the blocks of a launch) is selectable.  A kernel that is only
//     right when thread 3 runs before thread 17 between two barriers passes in one direction and fails in the other.
//         DS41_EMU_ORDER=forward | reverse | shuffle[:SEED]      (environment; the default is forward)
//         ds41_emu::set_order(...)                                 (what `ds41_cuda_emu_test --order ...` calls)
//     The tests are run in all three; "shuffle" re-draws the order at every scheduling round.
#pragma once

#include <ucontext.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <string>
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
    // The value this lane offers to the warp's current exchange (shuffle / vote).  TWO slots used alternately, so that one barrier per collective
    // is enough: a lane can write slot[p] of exchange n + 2 only after every lane has passed the barrier of exchange n + 1, i.e. has already
    // read slot[p] of exchange n.  `xn` counts this lane's exchanges (all active lanes of a warp call the collectives in lockstep).
    uint32_t slot[2] = {0, 0};
    uint32_t xn = 0;
};

inline Ctx* g_cur = nullptr;                        // the running thread's context
inline ucontext_t g_main;                           // the scheduler
inline std::vector<Fiber>* g_fibers = nullptr;
inline const std::function<void()>* g_kernel = nullptr;
inline unsigned char* g_dyn_smem = nullptr;
inline size_t g_launch_count = 0, g_block_count = 0;

// ---- the scheduling order (see the header comment) ----------------------------------------------------------------------
enum class Order { kForward, kReverse, kShuffle };
inline Order g_order = Order::kForward;
inline uint64_t g_order_seed = 1;
inline std::mt19937_64& order_rng() {
    static std::mt19937_64 g(1);
    return g;
}
inline void set_order(Order o, uint64_t seed = 1) {
    g_order = o;
    g_order_seed = seed;
    order_rng().seed(seed);
}
inline const char* order_name() { return g_order == Order::kForward ? "forward" : g_order == Order::kReverse ? "reverse" : "shuffle"; }
/// "forward", "reverse", "shuffle" or "shuffle:SEED"; false if the text is none of them.
inline bool set_order_from_string(const std::string& v) {
    if (v == "forward") set_order(Order::kForward);
    else if (v == "reverse") set_order(Order::kReverse);
    else if (v.compare(0, 7, "shuffle") == 0 && (v.size() == 7 || v[7] == ':')) set_order(Order::kShuffle, v.size() > 8 ? std::strtoull(v.c_str() + 8, nullptr, 10) : 1);
    else return false;
    return true;
}
inline bool g_order_env_read = [] {
    if (const char* e = std::getenv("DS41_EMU_ORDER"))
        if (!set_order_from_string(e)) std::fprintf(stderr, "ds41_emu: DS41_EMU_ORDER='%s' is not forward | reverse | shuffle[:SEED]; using forward\n", e);
    return true;
}();
/// `n` indices in the order the current setting asks for (a fresh random permutation for every call in shuffle mode).
inline void make_order(std::vector<int>& idx, int n) {
    idx.resize((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    if (g_order == Order::kReverse) std::reverse(idx.begin(), idx.end());
    else if (g_order == Order::kShuffle) std::shuffle(idx.begin(), idx.end(), order_rng());
}

// ---- static shared memory: one named section, poisoned at the start of every block --------------------------------------------
// `DS41_SHARED` (ds41_dev.cuh) puts every static __shared__ array into the section "ds41_smem"; the GNU linker provides the bounds.
#if defined(__linux__) && defined(__GNUC__)
extern "C" {
extern char __start_ds41_smem[] __attribute__((weak));
extern char __stop_ds41_smem[] __attribute__((weak));
}
inline void poison_static_shared() {
    char* const b = __start_ds41_smem;                         // (null when no kernel instantiated a static __shared__ array: weak symbols)
    char* const e = __stop_ds41_smem;
    if (b && e > b) std::memset(b, 0xFF, (size_t) (e - b));
}
#else
inline void poison_static_shared() {}
#endif

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
// A lane takes part in the exchange `n` (the caller's xn - 1) iff it counted it: s.xn >= me.xn.  A lane that EXITED before reaching it has a smaller
// count (a lane that offered and then exited is fine: its slot is still valid).  Reading from one that never offered is undefined on the hardware.
inline void check_source_lane(const Fiber& s, const Fiber& me, const char* what) {
    if (s.xn < me.xn) {
        std::fprintf(stderr, "ds41_emu: %s reads lane %d of a warp, which exited before this exchange: its register is undefined on the hardware\n", what, s.tid & 31);
        std::abort();
    }
}
/// Offer `bits` to the warp and wait until every active lane has offered its own; returns the parity of the slot the offers are in.
/// Every live lane of the warp must call the collectives together (the kernels' shuffles and votes are always full-warp).
inline int warp_offer(uint32_t bits) {
    Fiber* f = g_cur->fiber;
    const int p = (int) (f->xn++ & 1u);
    f->slot[p] = bits;
    sync_warp();
    return p;
}
template <class T>
inline T shfl_xor(T v, int mask) {
    Fiber* f = g_cur->fiber;
    const int p = warp_offer(to_bits(v));
    const int src = (f->tid & 31) ^ mask;
    Fiber& s = fibers()[(f->tid & ~31) + src];
    check_source_lane(s, *f, "shfl_xor");
    return from_bits<T>(s.slot[p]);
}
template <class T>
inline T shfl(T v, int src_lane) {
    Fiber* f = g_cur->fiber;
    const int p = warp_offer(to_bits(v));
    Fiber& s = fibers()[(f->tid & ~31) + (src_lane & 31)];
    check_source_lane(s, *f, "shfl");
    return from_bits<T>(s.slot[p]);
}
/// __ballot_sync(full mask, p): the bit of lane l is its predicate, for the lanes still ACTIVE (an exited lane contributes 0 - its slot would
/// otherwise hold the stale predicate of an earlier vote).
inline uint32_t ballot(bool pred) {
    Fiber* f = g_cur->fiber;
    const int p = warp_offer(pred ? 1u : 0u);
    uint32_t m = 0;
    const int base = f->tid & ~31;
    for (int l = 0; l < 32 && base + l < (int) fibers().size(); ++l)
        if (fibers()[base + l].xn >= f->xn) m |= (fibers()[base + l].slot[p] & 1u) << l;      // only the lanes that took part (not the exited ones)
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
    const int nblocks = (int) (grid.x * grid.y * grid.z);
    std::vector<int> border, forder;
    make_order(border, nblocks);                             // the blocks of a launch are independent: any order is legal on the hardware
    for (const int lin : border) {
        const unsigned bx = (unsigned) lin % grid.x, by = ((unsigned) lin / grid.x) % grid.y, bz = (unsigned) lin / (grid.x * grid.y);
        ++g_block_count;
        std::memset(dyn, 0xFF, smem_bytes);          // uninitialised shared memory = NaN bits
        poison_static_shared();                      // ... in the static __shared__ arrays too
        for (int t = 0; t < nthreads; ++t) {
            Fiber& f = fibers[t];
            f.tid = t;
            f.state = kRunnable;
            f.xn = 0;
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
            make_order(forder, nthreads);            // the order the runnable fibers get their turn in (a new draw per round when shuffling)
            for (const int fi : forder) {
                Fiber& f = fibers[(size_t) fi];
                if (f.state == kRunnable) {
                    g_cur = &f.ctx;
                    swapcontext(&g_main, &f.uctx);
                    ++ran;
                }
            }
            int live = 0, wait_block = 0;
            for (auto& f : fibers) {
                live += f.state != kDone;
                wait_block += f.state == kWaitBlock;
            }
            if (live == 0) break;
            bool released = false;
            if (wait_block == live) {
                for (auto& f : fibers)
                    if (f.state == kWaitBlock) f.state = kRunnable;      // (a finished fiber stays finished)
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
