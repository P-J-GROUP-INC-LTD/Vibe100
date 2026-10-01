// src/ds41/cuda/ds41_emu_test.cpp - DS-D: runs the SAME kernels the V100 runs (src/ds41/cuda/*.cuh, compiled for the host with
// -DDS41_EMU, see ds41_emu.hpp) through the parity drivers on the CPU.  No GPU needed.
//
//   ds41_cuda_emu_test [--router] [--split] [--experts] [--all] [--seed N]
//
// What it covers: the logic of every kernel (indexing, the 17-byte block reads, the byte permute, the shuffle reductions, the h hand-off
// between phases, the lists).  What it does NOT: speed, register/shared-memory budgets, instruction scheduling, the hardware behaviour
// of the intrinsics (they are re-implemented in ds41_dev.cuh from the PTX documentation) - those are the V100 programs' job.
// Slow by design (one fiber per CUDA thread): `--quick` (default) runs a reduced set, `--all` the full one.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <map>

#include "ds41_parity_lib.hpp"

namespace {
using namespace strata::ds41::cuda::parity;

struct HostDev : Dev {
    // every allocation sits between two 256-byte guard zones that are checked when it is released: a kernel that stores outside its
    // buffer is a failure here (the V100 would silently corrupt a neighbour)
    static constexpr size_t kGuard = 256;
    std::map<void*, size_t> live;
    void* alloc(size_t bytes) override {
        if (bytes == 0) bytes = 16;
        const size_t body = (bytes + 255) & ~(size_t) 255;
        unsigned char* base = static_cast<unsigned char*>(std::aligned_alloc(256, body + 2 * kGuard));
        std::memset(base, 0xAB, body + 2 * kGuard);
        std::memset(base + kGuard, 0xCD, body);                        // fresh "device memory" is not zero
        live[base + kGuard] = bytes;
        return base + kGuard;
    }
    void release(void* p) override {
        if (!p) return;
        const size_t bytes = live.at(p);
        const size_t body = (bytes + 255) & ~(size_t) 255;
        unsigned char* base = static_cast<unsigned char*>(p) - kGuard;
        bool ok = true;
        for (size_t i = 0; i < kGuard; ++i) ok = ok && base[i] == 0xAB;
        for (size_t i = bytes; i < body; ++i) ok = ok && base[kGuard + i] == 0xCD;       // the padding of the last 256 bytes
        for (size_t i = 0; i < kGuard; ++i) ok = ok && base[kGuard + body + i] == 0xAB;
        if (!ok) {
            std::printf("FAIL guard: a kernel or the test wrote outside a device buffer of %zu bytes\n", bytes);
            std::exit(3);
        }
        live.erase(p);
        std::free(base);
    }
    void h2d(void* dst, const void* src, size_t n) override { std::memcpy(dst, src, n); }
    void d2h(void* dst, const void* src, size_t n) override { std::memcpy(dst, src, n); }
    void fill(void* p, int byte, size_t n) override { std::memset(p, byte, n); }
    void sync() override {}
    void* stream() override { return nullptr; }
    double time_us(const std::function<void()>& fn, int reps) override {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
    }
    bool is_emulation() const override { return true; }
};
}  // namespace

int main(int argc, char** argv) {
    bool router = false, split = false, experts = false, all = false;
    uint64_t seed = 0;                                       // 0: each driver's own default seed
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        if (!std::strcmp(argv[i], "--router")) router = true;
        else if (!std::strcmp(argv[i], "--split")) split = true;
        else if (!std::strcmp(argv[i], "--experts")) experts = true;
        else if (!std::strcmp(argv[i], "--all")) all = true;
    }
    if (!router && !split && !experts) router = split = experts = true;
    HostDev dev;
    Report rep;
    const auto t0 = std::chrono::steady_clock::now();
    if (router) {
        RouterOpts o;
        if (seed) o.seed = seed;
        o.Ts = all ? std::vector<int>{1, 2, 3, 4, 5, 8, 17, 32, 33, 70} : std::vector<int>{1, 2, 3, 5, 33};
        o.timing_reps = 0;
        run_router_parity(dev, o, rep);
    }
    if (split) {
        SplitOpts o;
        if (seed) o.seed = seed;
        o.Ts = all ? std::vector<int>{1, 2, 3, 5, 8, 9, 40, 1000} : std::vector<int>{1, 3, 8, 9, 40};
        run_split_parity(dev, o, rep);
    }
    if (experts) {
        ExpertOpts o;
        if (seed) o.seed = seed;
        o.slots = all ? 16 : 8;
        o.Ts = all ? std::vector<int>{1, 2, 3, 4, 5, 8} : std::vector<int>{1, 3};
        o.max_hits_checked = all ? 1000000 : 3;
        o.timing_reps = 0;
        run_expert_parity(dev, o, rep);
    }
    std::printf("INFO emulation wall time %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return rep.summary();
}
