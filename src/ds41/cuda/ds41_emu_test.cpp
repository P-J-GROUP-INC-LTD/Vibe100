// src/ds41/cuda/ds41_emu_test.cpp - DS-D: runs the SAME kernels the V100 runs (src/ds41/cuda/*.cuh, compiled for the host with
// -DDS41_EMU, see ds41_emu.hpp) through the parity drivers on the CPU.  No GPU needed.
//
//   ds41_cuda_emu_test [--router] [--split] [--experts] [--quant] [--emu] [--all] [--seed N] [--order forward|reverse|shuffle[:SEED]] [--geom real|mini|both]
//
// --geom picks the geometry the kernels are instantiated at (include/strata/ds41/geom.hpp): `real` (default; the V100 engine's, slow in emulation: one fiber
// per CUDA thread), `mini` (the tiny model of tools/ds41/make_mini_gguf.py: hidden 256, 16 experts, top-2; every suite in seconds, with the full test sets)
// or `both`.  The emulator's own checks (--emu) do not depend on the geometry and run once.
//
// --order selects the order in which the emulator schedules the fibers of a block and the blocks of a launch (also: the environment variable
// DS41_EMU_ORDER, see ds41_emu.hpp): the suites are run in all three, because a kernel that is only right when thread 3 runs before thread 17
// (or block 0 before block 1) passes in one direction and fails in the other.  The emulator also poisons shared memory at the start of every
// block (0xFF), aborts on a misaligned vector load and on a shuffle from an exited lane.
// --emu: the emulator's own checks of what it promises (a vote does not count exited lanes; static shared memory is poisoned at the start of every
// block; the scheduling order really is the one asked for).
// --quant: the activation quantiser's kernel against the CPU library itself (scalar, AVX2, AVX-512) on the contract's special blocks and on
// random bit patterns: identical bytes (the GPU kernel, the CPU's three implementations and ref::quantize_block are one rule).
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

#include <algorithm>
#include <string>
#include <type_traits>

#include "ds41_emu.hpp"
#include "ds41_parity_lib.hpp"
#include "strata/ds41/cpu/mxfp4_expert.hpp"

namespace {
using namespace strata::ds41::cuda::parity;

using strata::ds41::cuda::HostDev;     // the shared emulation Dev (include/strata/ds41/cuda/ds41_dev.hpp): guard zones checked at release, 0xCD-filled
// The emulator's promises, checked on tiny kernels (a regression here would silently weaken every other suite).
void run_emu_selftest(Report& rep) {
    namespace E = ds41_emu;
    // 1. ballot counts only the lanes still active: lanes 16..31 leave after the first vote
    {
        unsigned first = 0, second = 0;
        E::launch(dim3(1), dim3(32), 0, [&] {
            const unsigned m1 = E::ballot(true);
            if ((threadIdx.x & 31) >= 16) return;
            const unsigned m2 = E::ballot(true);
            if (threadIdx.x == 0) {
                first = m1;
                second = m2;
            }
        });
        rep.line(first == 0xFFFFFFFFu && second == 0x0000FFFFu, "emulator: ballot ignores exited lanes", "full vote 0x%08x (expect 0xffffffff), vote after lanes 16..31 exited 0x%08x (expect 0x0000ffff)", first,
                 second);
    }
    // 2. static __shared__ storage is poisoned (0xFF) at the start of every block: block 1 must not see what block 0 left behind
    {
        unsigned seen[3] = {0, 0, 0};
        E::launch(dim3(3), dim3(32), 0, [&] {
            static __attribute__((section("ds41_smem"))) unsigned s_arr[16];
            if (threadIdx.x == 0) {
                seen[blockIdx.x] = s_arr[5];
                s_arr[5] = 123;
            }
        });
        rep.line(seen[0] == 0xFFFFFFFFu && seen[1] == 0xFFFFFFFFu && seen[2] == 0xFFFFFFFFu, "emulator: static shared memory is poisoned per block",
                 "first read of a never-written static __shared__ word in blocks 0, 1, 2: 0x%08x 0x%08x 0x%08x (expect 0xffffffff)", seen[0], seen[1], seen[2]);
    }
    // 3. the scheduling order is the one asked for: blocks of a launch, and the fibers of a block between two barriers
    {
        const int nb = 8, nt = 8;
        std::vector<int> block_order, thread_order;
        E::launch(dim3(nb), dim3(nt), 0, [&] {
            if (threadIdx.x == 0) block_order.push_back((int) blockIdx.x);
            if (blockIdx.x == 0) thread_order.push_back((int) threadIdx.x);
        });
        std::vector<int> asc(nb), desc(nb);
        for (int i = 0; i < nb; ++i) asc[i] = i, desc[i] = nb - 1 - i;
        std::vector<int> sorted = block_order;
        std::sort(sorted.begin(), sorted.end());
        const bool perm = sorted == asc;
        bool ok;
        const char* want;
        switch (E::g_order) {
            case E::Order::kForward: ok = block_order == asc; want = "ascending"; break;
            case E::Order::kReverse: ok = block_order == desc; want = "descending"; break;
            default: ok = perm && block_order != asc && block_order != desc; want = "a shuffled permutation"; break;
        }
        std::vector<int> t_asc(nt), t_desc(nt);
        for (int i = 0; i < nt; ++i) t_asc[i] = i, t_desc[i] = nt - 1 - i;
        bool tok = true;
        if (E::g_order == E::Order::kForward) tok = thread_order == t_asc;
        else if (E::g_order == E::Order::kReverse) tok = thread_order == t_desc;
        else tok = thread_order.size() == (size_t) nt;
        rep.line(ok && tok, "emulator: scheduling order", "order '%s': blocks ran %s (%s), threads of block 0 ran %s", E::order_name(), ok ? "as asked" : "NOT as asked", want, tok ? "as asked" : "NOT as asked");
    }
}

// The emulated GPU quantiser against the CPU library: the same bytes on the contract's special blocks and on random bit patterns.
template <class G>
void run_quant_vs_cpu(Dev& dev, uint64_t seed, int reps, Report& rep) {
    using namespace strata::ds41;
    namespace cpu = strata::ds41::cpu;
    constexpr int kHidden = G::kHidden, kActBlocks = cuda::ExpertDims<G>::kActBlocks;
    cuda::ref::Rng rng(seed);
    const auto edges = cuda::parity::quant_edge_blocks();
    const int T = std::max(2, (int) ((edges.size() + kActBlocks - 1) / kActBlocks)), nblocks = T * kActBlocks;      // every special block fits (RealGeom: T = 2)
    for (int rep_i = 0; rep_i < reps; ++rep_i) {
        std::vector<float> x((size_t) T * kHidden);
        for (int b = 0; b < nblocks; ++b) {
            float* xb = &x[(size_t) b * 32];
            const int idx = rep_i == 0 ? b : -1;
            if (idx >= 0 && idx < (int) edges.size()) {
                for (int j = 0; j < 32; ++j) xb[j] = j < (int) edges[idx].head.size() ? edges[idx].head[j] : edges[idx].fill;
                continue;
            }
            const int kind = rng.range(0, 5);
            for (int j = 0; j < 32; ++j) {
                float v;
                switch (kind) {
                    case 0: v = cuda::ref::bits_f32(rng.u32()); break;
                    case 1: v = (float) (rng.normal() * std::ldexp(1.0, rng.range(-140, 100))); break;
                    case 2: v = (rng.range(0, 40) == 0) ? cuda::ref::bits_f32(0x7F800000u | (rng.u32() & 0x807FFFFFu)) : (float) rng.normal(); break;
                    case 3: v = (float) rng.range(-300, 300) * 0.5f; break;
                    case 4: v = (float) (rng.normal() * std::ldexp(1.0, -100) * (1.0 + 1e-3 * rng.range(-5, 5))); break;
                    default: v = (float) rng.normal(); break;
                }
                xb[j] = v;
            }
        }
        cuda::DevBuf<float> d_x(dev, x.size());
        d_x.up(x);
        cuda::DevBuf<int8_t> d_xq(dev, x.size());
        cuda::DevBuf<float> d_xs(dev, (size_t) nblocks);
        cuda::ds41_quantize_acts<G>(dev, d_x.p, T, kHidden, d_xq.p, d_xs.p, dev.stream());
        dev.sync();
        const auto xq_perm = d_xq.down();
        const auto xs = d_xs.down();
        std::vector<int8_t> gpu_nat(x.size());
        cuda::ref::from_perm(xq_perm.data(), gpu_nat.data(), nblocks);
        for (cpu::Isa isa : {cpu::Isa::kScalar, cpu::Isa::kAvx2, cpu::Isa::kAvx512}) {
            if (!cpu::isa_supported(isa)) continue;
            long bad = 0;
            static cpu::ActQ a;
            std::vector<int8_t> cq((size_t) kHidden);
            for (int t = 0; t < T; ++t) {
                cpu::quantize_act(&x[(size_t) t * kHidden], kHidden, a, isa);
                cpu::act_unpack(a, kHidden, cq.data());
                bad += std::memcmp(cq.data(), &gpu_nat[(size_t) t * kHidden], kHidden) != 0;
                for (int b = 0; b < kActBlocks; ++b) bad += std::memcmp(&a.scale[b], &xs[(size_t) t * kActBlocks + b], 4) != 0;
            }
            rep.line(bad == 0, ("quantize GPU kernel vs CPU " + std::string(cpu::isa_name(isa)) + " rep " + std::to_string(rep_i)).c_str(),
                     "%d tokens x %d values (%s): int8 values and fp32 scales (as bits) identical: %s", T, kHidden, rep_i == 0 ? "the special blocks first, then random bit patterns" : "random bit patterns",
                     bad == 0 ? "yes" : "NO");
        }
    }
}

// One geometry's suites.  RealGeom: the reduced sets (--quick, the default) or the full ones (--all) as always.  MiniGeom: always the full sets (they are cheap).
template <class G>
void run_geometry(HostDev& dev, bool router, bool split, bool experts, bool quant, bool all, uint64_t seed, Report& rep) {
    const bool full = all || !std::is_same_v<G, strata::ds41::RealGeom>;
    std::printf("INFO geometry: %s (hidden %d, experts %d, top-%d, ff %d)\n", G::kName, G::kHidden, G::kExperts, G::kTopK, G::kFF);
    if (router) {
        RouterOpts o;
        if (seed) o.seed = seed;
        o.Ts = full ? std::vector<int>{1, 2, 3, 4, 5, 8, 17, 32, 33, 70} : std::vector<int>{1, 2, 3, 5, 33};
        o.indep_Ts = full ? std::vector<int>{2, 3, 5, 8, 17, 32, 33, 40, 100} : std::vector<int>{8, 32, 33, 100};
        o.indep_alone = full ? std::vector<int>{0, 1, 7, 8, 31, 32, 33, 99} : std::vector<int>{0, 7, 33, 99};
        if (!full) o.indep_windows = {{37, 8}, {5, 4}};
        o.timing_reps = 0;
        run_router_parity<G>(dev, o, rep);
    }
    if (split) {
        SplitOpts o;
        if (seed) o.seed = seed;
        o.Ts = full ? std::vector<int>{1, 2, 3, 5, 8, 9, 40, 1000} : std::vector<int>{1, 3, 8, 9, 40};
        run_split_parity<G>(dev, o, rep);
    }
    if (experts) {
        ExpertOpts o;
        if (seed) o.seed = seed;
        o.slots = !std::is_same_v<G, strata::ds41::RealGeom> ? 12 : (all ? 16 : 8);       // MiniGeom: every expert id of the resident pool (kExperts * 25 / 32 = 12)
        o.Ts = full ? std::vector<int>{1, 2, 3, 4, 5, 8} : std::vector<int>{1, 3};
        o.max_hits_checked = full ? 1000000 : 3;
        o.timing_reps = 0;
        run_expert_parity<G>(dev, o, rep);
    }
    if (quant) {
        // the kernel on the contract's special blocks, then at other widths and in both byte orders, then against the CPU library (all ISAs this machine has)
        strata::ds41::cuda::ref::Rng rng(seed ? seed : 7);
        run_quant_edge<G>(dev, rng, rep);
        run_quant_widths<G>(dev, rng, rep);
        run_quant_vs_cpu<G>(dev, seed ? seed : 7, full ? 8 : 3, rep);
    }
}
}  // namespace

int main(int argc, char** argv) {
    bool router = false, split = false, experts = false, quant = false, emu = false, all = false;
    bool run_real = true, run_mini = false;
    uint64_t seed = 0;                                       // 0: each driver's own default seed
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        if (!std::strcmp(argv[i], "--router")) router = true;
        else if (!std::strcmp(argv[i], "--split")) split = true;
        else if (!std::strcmp(argv[i], "--experts")) experts = true;
        else if (!std::strcmp(argv[i], "--quant")) quant = true;
        else if (!std::strcmp(argv[i], "--emu")) emu = true;
        else if (!std::strcmp(argv[i], "--all")) all = true;
        else if (!std::strcmp(argv[i], "--geom") && i + 1 < argc) {
            const std::string g = argv[++i];
            if (g == "real") run_real = true, run_mini = false;
            else if (g == "mini") run_real = false, run_mini = true;
            else if (g == "both") run_real = run_mini = true;
            else {
                std::printf("FAIL: --geom wants real | mini | both\n");
                return 2;
            }
        } else if (!std::strcmp(argv[i], "--order") && i + 1 < argc) {
            if (!ds41_emu::set_order_from_string(argv[++i])) {
                std::printf("FAIL: --order wants forward | reverse | shuffle[:SEED]\n");
                return 2;
            }
        }
    }
    if (!router && !split && !experts && !quant && !emu) router = split = experts = quant = emu = true;
    std::printf("INFO emulator scheduling order: %s (seed %llu)\n", ds41_emu::order_name(), (unsigned long long) ds41_emu::g_order_seed);
    HostDev dev;
    Report rep;
    const auto t0 = std::chrono::steady_clock::now();
    if (run_real) run_geometry<strata::ds41::RealGeom>(dev, router, split, experts, quant, all, seed, rep);
    if (run_mini) run_geometry<strata::ds41::MiniGeom>(dev, router, split, experts, quant, all, seed, rep);
    if (emu) run_emu_selftest(rep);
    std::printf("INFO emulation wall time %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return rep.summary();
}
