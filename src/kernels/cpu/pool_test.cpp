// src/kernels/cpu/pool_test.cpp - P2.S3's test for the expert pool.
//
// The pool's correctness claims are small and specific, so they are checked directly rather than through a
// timing number:
//
//   1. THE SAME ANSWER AS SERIAL.  Every job's output must be bit-identical to running that expert serially.
//      Each worker owns its own `ExpertScratch`, and the jobs share one read-only activation, so there is no
//      legitimate source of difference - "close enough" here would be hiding a data race.
//   2. EVERY JOB RUNS EXACTLY ONCE.  Claiming is `head.fetch_add`, and an off-by-one in the bound is the
//      classic pool bug: it either drops a job or runs one twice.  Checked with a sentinel-filled output
//      buffer, so a dropped job is visible as an untouched slot rather than as a slightly wrong number.
//   3. REPEATED BATCHES.  A pool that works once and hangs or corrupts on the second `run()` is the failure
//      mode the park protocol exists to prevent, so `run()` is called many times in a row.
//   4. A BATCH BIGGER AND SMALLER THAN THE WORKER COUNT, because `n < workers` leaves most workers claiming
//      nothing and `n > workers` is the real case (10 experts, 5 workers).
//
// `--synthetic` uses ten random experts instead of reading them from the pack (the checks are about the pool, not
// the weights), which is how it runs on a machine without the 66 GB file.
//
// `--numa-mirror` (Vibe100 WP-F, implies --synthetic) is the pool's half of the NUMA mirroring, on a machine with ONE
// node: the CPUs are split in two halves and called node 0 and node 1 (`PoolNuma`), the arena gets a replica built by the
// real code (`NumaReplicas`, on a fake sysfs tree), and then
//   1. MIRRORING ON vs OFF: every pool API (run, run_split, run_split_multi, and the native i-quant run_split_multi_native
//      where ggml is built in) is bit-identical to the same call with no mirror, whatever node each worker is on;
//   2. WHICH COPY A WORKER READS: after the replica is built the PRIMARY is overwritten with other experts ("poisoned"), so
//      the two copies differ.  With every thread on "node 1" every output must be the replica's, with every thread on
//      "node 0" the primary's, and with the threads split every whole-expert job must be entirely one copy's or the other's
//      (never a mix) and both must occur - so node-1 workers read the replica's addresses and node-0 workers do not;
//   3. A BLOB OUTSIDE THE ARENA (a staged buffer) is left alone, and the host's own drain obeys the same rule;
//   4. THE CORE ORDER: the host core is on the GPU's node, the workers alternate between the nodes.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/platform/numa.hpp"

#if defined(STRATA_NATIVE_EXPERTS)
#include "ggml.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace cpu = strata::kernels::cpu;
namespace plat = strata::platform;

namespace {

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

#if defined(__linux__)
// ================================ --numa-mirror (WP-F) ================================

// the synthetic Q2_0 expert of the main test, as a function of a seed
void synth_blob(uint8_t* b, uint32_t seed) {
    std::mt19937 rng(seed);
    std::memset(b, 0, cpu::BLOB);
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

constexpr int kNexp = 10;   // experts in an arena; an 11th ("stray") lives outside it

// One kind of expert (the canonical Q2_0 blob, or a native i-quant one): how to fill an arena with experts of a given seed
// and how to run every pool API of that kind over an arena.  `exec` returns one output vector per API.
struct Kind {
    std::string name;
    size_t blob = 0;                                                      // bytes per expert
    std::function<void(uint8_t*, uint32_t)> fill_one;                     // one expert from a seed
    std::function<std::vector<std::vector<float>>(cpu::ExpertPool&, const uint8_t* arena, const uint8_t* stray)> exec;
    bool whole_expert_api0 = false;                                       // api 0 = `run`: one worker computes a whole expert
    size_t api0_floats_per_job = 0;
    int variants = kNexp;                                                 // distinct experts per arena (ggml's quantizer is slow: the native kind repeats a few)
};

size_t arena_bytes(const Kind& k) { return (size_t) (kNexp + 1) * k.blob; }   // kNexp experts + one spare, as the engine's arena is one blob longer

void fill_arena(const Kind& k, std::vector<uint8_t>& a, uint32_t seed) {
    a.assign(arena_bytes(k), 0);
    for (int e = 0; e < kNexp; ++e) {
        uint8_t* dst = a.data() + (size_t) e * k.blob;
        if (e < k.variants) k.fill_one(dst, seed * 131u + (uint32_t) e);
        else std::memcpy(dst, a.data() + (size_t) (e % k.variants) * k.blob, k.blob);
    }
}

Kind make_q2_kind() {
    Kind k;
    k.name = "canonical Q2_0";
    k.blob = cpu::BLOB;
    k.fill_one = [](uint8_t* b, uint32_t seed) { synth_blob(b, seed); };
    k.whole_expert_api0 = true;
    k.api0_floats_per_job = cpu::H;
    k.exec = [](cpu::ExpertPool& pool, const uint8_t* arena, const uint8_t* stray) {
        std::mt19937 rng(4242);
        std::normal_distribution<float> gauss(0.f, 1.f);
        std::vector<cpu::ActQ> acts(3);
        for (auto& a : acts) {
            std::vector<float> x(cpu::H);
            for (auto& v : x) v = gauss(rng);
            cpu::act_quant_q8_1(x.data(), cpu::H, a);
        }
        const float SENT = -1.2345e33f;
        std::vector<std::vector<float>> outs;
        auto base_of = [&](int e) { return e < kNexp ? arena + (size_t) e * cpu::BLOB : stray; };
        for (int api = 0; api < 2; ++api) {   // run, then run_split
            std::vector<cpu::ExpertJob> jobs(kNexp + 1);
            std::vector<float> out((size_t) (kNexp + 1) * cpu::H, SENT);
            for (int e = 0; e <= kNexp; ++e) {
                jobs[(size_t) e].blob = base_of(e);
                jobs[(size_t) e].act = &acts[0];
                jobs[(size_t) e].out = out.data() + (size_t) e * cpu::H;
                jobs[(size_t) e].slot = e;
            }
            if (api == 0) pool.run(jobs.data(), kNexp + 1);
            else pool.run_split(jobs.data(), kNexp + 1);
            outs.push_back(std::move(out));
        }
        {   // run_split_multi: expert e carries 1 + e % 3 tokens
            std::vector<cpu::ExpertJobMulti> jobs(kNexp + 1);
            std::vector<float> out((size_t) (kNexp + 1) * 3 * cpu::H, SENT);
            for (int e = 0; e <= kNexp; ++e) {
                jobs[(size_t) e].blob = base_of(e);
                jobs[(size_t) e].nt = 1 + e % 3;
                for (int t = 0; t < jobs[(size_t) e].nt; ++t) {
                    jobs[(size_t) e].act[t] = &acts[(size_t) t];
                    jobs[(size_t) e].out[t] = out.data() + ((size_t) e * 3 + (size_t) t) * cpu::H;
                }
            }
            pool.run_split_multi(jobs.data(), kNexp + 1);
            outs.push_back(std::move(out));
        }
        return outs;
    };
    return k;
}

#if defined(STRATA_NATIVE_EXPERTS)
// A native pack's expert: IQ3_XXS gate/up rows then IQ4_NL down rows (ggml quantizes them), the same layout the engine's
// native packs use: [gate rows | up rows | down rows].
bool make_native_kind(Kind& k, cpu::NativeFmt& f) {
    std::string err;
    if (!cpu::native_experts_available() || !cpu::native_fmt((int) GGML_TYPE_IQ3_XXS, (int) GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, f, err))
        return false;
    k.name = "native IQ3_XXS / IQ4_NL";
    k.blob = f.bytes;
    k.variants = 3;
    k.fill_one = [f](uint8_t* b, uint32_t seed) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> src((size_t) cpu::H * cpu::FF), imat((size_t) cpu::H, 1.0f);
        auto quant = [&](ggml_type t, int64_t rows, int64_t cols, uint8_t* dst, size_t want) {
            for (auto& v : src) v = nd(rng) * (0.5f + 0.5f * nd(rng) * nd(rng));
            const size_t got = ggml_quantize_chunk(t, src.data(), dst, 0, rows, cols,
                                                   ggml_quantize_requires_imatrix(t) ? imat.data() : nullptr);
            if (got != want) { std::fprintf(stderr, "native fixture: quantize gave %zu, wanted %zu\n", got, want); std::abort(); }
        };
        quant(GGML_TYPE_IQ3_XXS, cpu::FF, cpu::H, b, (size_t) cpu::FF * f.gu_row);
        quant(GGML_TYPE_IQ3_XXS, cpu::FF, cpu::H, b + f.up_off, (size_t) cpu::FF * f.gu_row);
        quant(GGML_TYPE_IQ4_NL, cpu::H, cpu::FF, b + f.down_off, (size_t) cpu::H * f.d_row);
    };
    k.exec = [f](cpu::ExpertPool& pool, const uint8_t* arena, const uint8_t* stray) {
        std::mt19937 rng(777);
        std::normal_distribution<float> gauss(0.f, 1.f);
        std::vector<std::vector<uint8_t>> nact(3, std::vector<uint8_t>(cpu::kNativeActBytes));
        for (auto& a : nact) {
            std::vector<float> x(cpu::H);
            for (auto& v : x) v = gauss(rng);
            cpu::native_quant_act(f, x.data(), a.data());
        }
        const float SENT = -1.2345e33f;
        std::vector<cpu::ExpertJobMulti> jobs(kNexp + 1);
        std::vector<float> out((size_t) (kNexp + 1) * 3 * cpu::H, SENT);
        for (int e = 0; e <= kNexp; ++e) {
            jobs[(size_t) e].blob = e < kNexp ? arena + (size_t) e * f.bytes : stray;
            jobs[(size_t) e].nt = 1 + e % 3;
            for (int t = 0; t < jobs[(size_t) e].nt; ++t) {
                jobs[(size_t) e].nact[t] = nact[(size_t) t].data();
                jobs[(size_t) e].out[t] = out.data() + ((size_t) e * 3 + (size_t) t) * cpu::H;
            }
        }
        pool.run_split_multi_native(f, jobs.data(), kNexp + 1);
        return std::vector<std::vector<float>>{std::move(out)};
    };
    return true;
}
#endif

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}
bool all_same(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i])) return false;
    return true;
}

struct FakeBox {
    std::filesystem::path root;
    std::vector<int> node0, node1;
    plat::NumaTopology topo;
    plat::MirrorPlan plan;
    ~FakeBox() { std::error_code ec; std::filesystem::remove_all(root, ec); }
};

// CPUs 0..n/2-1 of the physical-core list are "node 0" (the GPU's), the rest "node 1"; a fake sysfs tree says so and the
// real planner makes the plan, so the replica below is built by the engine's own code.
bool make_fake_box(FakeBox& b, uint64_t arena_cap) {
    const std::vector<int> cores = cpu::physical_cores(false);
    if (cores.size() < 2) return false;
    const size_t k = cores.size() / 2;
    b.node0.assign(cores.begin(), cores.begin() + (long) k);
    b.node1.assign(cores.begin() + (long) k, cores.end());
    std::sort(b.node0.begin(), b.node0.end());
    std::sort(b.node1.begin(), b.node1.end());
    b.root = std::filesystem::temp_directory_path() / ("strata_pool_test_numa_" + std::to_string((long) getpid()));
    std::filesystem::remove_all(b.root);
    auto put = [&](const std::filesystem::path& p, const std::string& t) {
        std::filesystem::create_directories(p.parent_path());
        std::ofstream(p) << t;
    };
    const std::string mem = "Node %d MemTotal: 67108864 kB\nNode %d MemFree: 52428800 kB\nNode %d Active(file): 1048576 kB\nNode %d Inactive(file): 1048576 kB\n";
    for (int id = 0; id < 2; ++id) {
        char t[256];
        std::snprintf(t, sizeof t, mem.c_str(), id, id, id, id);
        const auto d = b.root / "devices/system/node" / ("node" + std::to_string(id));
        put(d / "cpulist", plat::format_cpulist(id == 0 ? b.node0 : b.node1) + "\n");
        put(d / "meminfo", t);
    }
    put(b.root / "bus/pci/devices/0000:3b:00.0/numa_node", "0\n");
    b.topo = plat::numa_discover(b.root.string());
    plat::numa_set_gpu(b.topo, "0000:3B:00.0");
    plat::MirrorInputs in;
    in.mode = plat::NumaMode::Mirror;
    in.topo = &b.topo;
    in.arena_bytes = arena_cap;
    in.headroom = 1ull << 30;
    b.plan = plat::plan_arena_mirror(in);
    return b.plan.mirror;
}

// A fake sysfs tree of `cpus.size()` nodes (an empty cpu list = a memory-only node), the cpu topology files when `pkg` is given, the
// GPU on node `gpu_node`; the topology, and the plan the engine's own planner makes for it.
struct FakeTree {
    std::filesystem::path root;
    plat::NumaTopology topo;
    plat::MirrorPlan plan;
    ~FakeTree() { std::error_code ec; std::filesystem::remove_all(root, ec); }
};
bool make_fake_tree(FakeTree& f, const std::string& tag, const std::vector<std::vector<int>>& cpus, const std::vector<int>& pkg,
                    int gpu_node, plat::NumaMode mode, uint64_t arena_cap) {
    f.root = std::filesystem::temp_directory_path() / ("strata_pool_test_" + tag + "_" + std::to_string((long) getpid()));
    std::filesystem::remove_all(f.root);
    auto put = [&](const std::filesystem::path& p, const std::string& t) {
        std::filesystem::create_directories(p.parent_path());
        std::ofstream(p) << t;
    };
    const std::string mem = "Node %d MemTotal: 67108864 kB\nNode %d MemFree: 52428800 kB\nNode %d Active(file): 1048576 kB\nNode %d Inactive(file): 1048576 kB\n";
    for (size_t id = 0; id < cpus.size(); ++id) {
        char t[256];
        std::snprintf(t, sizeof t, mem.c_str(), (int) id, (int) id, (int) id, (int) id);
        const auto d = f.root / "devices/system/node" / ("node" + std::to_string(id));
        put(d / "cpulist", plat::format_cpulist(cpus[id]) + "\n");
        put(d / "meminfo", t);
        if (!pkg.empty())
            for (int c : cpus[id]) put(f.root / "devices/system/cpu" / ("cpu" + std::to_string(c)) / "topology/physical_package_id", std::to_string(pkg[id]) + "\n");
    }
    put(f.root / "bus/pci/devices/0000:3b:00.0/numa_node", std::to_string(gpu_node) + "\n");
    f.topo = plat::numa_discover(f.root.string());
    plat::numa_set_gpu(f.topo, "0000:3B:00.0");
    plat::MirrorInputs in;
    in.mode = mode;
    in.topo = &f.topo;
    in.arena_bytes = arena_cap;
    in.headroom = 1ull << 30;
    f.plan = plat::plan_arena_mirror(in);
    return f.plan.mirror;
}

cpu::PoolNuma layout(const std::vector<int>& n0, const std::vector<int>& n1, int host_node) {
    cpu::PoolNuma p;
    p.node_cpus = {n0, n1};
    p.host_node = host_node;
    return p;
}

int numa_mirror_test() {
    int bad = 0;
    auto expect = [&](bool ok, const std::string& what) {
        std::printf("  %-78s %s\n", what.c_str(), ok ? "yes" : "*** NO ***");
        if (!ok) ++bad;
    };
    std::vector<Kind> kinds;
    kinds.push_back(make_q2_kind());
#if defined(STRATA_NATIVE_EXPERTS)
    {
        Kind nk;
        cpu::NativeFmt nf;
        if (make_native_kind(nk, nf)) kinds.push_back(std::move(nk));
        else std::printf("  (native i-quant experts not available in this build: skipped)\n");
    }
#endif
    FakeBox box;
    if (!make_fake_box(box, arena_bytes(kinds[0]))) {
        std::printf("  needs at least two physical cores to call one of them a second node: SKIPPED\n");
        return 0;
    }
    std::printf("  fake two-node box: node 0 (the GPU's) = cpus %s, node 1 = cpus %s\n", plat::format_cpulist(box.node0).c_str(),
                plat::format_cpulist(box.node1).c_str());

    // ---- 4. the core order
    {
        const std::vector<int> legacy = cpu::physical_cores(false);
        cpu::set_pool_numa(layout(box.node0, box.node1, 1));
        const std::vector<int> a = cpu::physical_cores(false), b = cpu::physical_cores(true);
        auto sorted = [](std::vector<int> v) { std::sort(v.begin(), v.end()); return v; };
        expect(sorted(a) == sorted(legacy), "the same cores, reordered");
        expect(!a.empty() && std::find(box.node1.begin(), box.node1.end(), a[0]) != box.node1.end(),
               "host node 1: the first core (the host loop's) is on node 1");
        expect(b.size() + 1 == a.size() && std::find(b.begin(), b.end(), a[0]) == b.end(), "physical_cores(true) leaves out exactly that core");
        bool alt = true;
        for (size_t i = 1; i + 1 < a.size() && i < 2 * std::min(box.node0.size(), box.node1.size()); ++i) {
            const bool n0 = std::find(box.node0.begin(), box.node0.end(), a[i]) != box.node0.end();
            const bool m0 = std::find(box.node0.begin(), box.node0.end(), a[i + 1]) != box.node0.end();
            alt = alt && n0 != m0;
        }
        expect(alt, "the workers alternate between the nodes");
        cpu::set_pool_numa(layout(box.node0, box.node1, 0));
        const std::vector<int> c = cpu::physical_cores(false);
        expect(!c.empty() && std::find(box.node0.begin(), box.node0.end(), c[0]) != box.node0.end(), "host node 0: the first core is on node 0");
        cpu::clear_pool_numa();
        expect(cpu::physical_cores(false) == legacy, "cleared: the legacy order again");
    }

    for (Kind& k : kinds) {
        std::printf("\n  --- %s experts (%.2f MB each) ---\n", k.name.c_str(), (double) k.blob / 1e6);
        cpu::clear_pool_numa();
        const size_t cap = arena_bytes(k);
        // two sets of experts that differ: A is the primary's, B is what the poison puts there
        std::vector<uint8_t> A, B, stray(k.blob);
        fill_arena(k, A, 1);
        fill_arena(k, B, 2);
        k.fill_one(stray.data(), 99);
        FakeBox kbox;
        if (!make_fake_box(kbox, cap)) { std::printf("  (no fake box)\n"); return bad + 1; }

        // references: no PoolNuma, no mirror - the pool as it was before WP-F, on A and on B
        cpu::clear_pool_numa();
        std::vector<std::vector<float>> refA, refB;
        {
            cpu::ExpertPool plain;
            expect(!plain.mirrored(), "a pool with no mirror says so");
            refA = k.exec(plain, A.data(), stray.data());
            refB = k.exec(plain, B.data(), stray.data());
        }
        expect(!all_same(refA, refB), "the two expert sets give different outputs (so the copies can be told apart)");

        // ---- 1. ON vs OFF, nodes mixed: the replica is an exact copy built by the engine's code
        {
            cpu::set_pool_numa(layout(kbox.node0, kbox.node1, 0));
            cpu::ExpertPool pool;
            plat::NumaReplicas rep;
            plat::MirrorOptions mo;
            mo.faked = true; mo.threads = 4; mo.lock = false; mo.samples = 8;
            std::vector<std::string> log;
            const int kept = rep.build(A.data(), cap, cap, kbox.plan, kbox.topo, mo, log);
            expect(kept == 1 && rep.mirror().active(), "NumaReplicas built the replica on fake node 1");
            pool.set_mirror(rep.mirror());
            expect(pool.mirrored(), "the pool took the mirror");
            int on0 = 0, on1 = 0;
            for (int i = 0; i < pool.workers(); ++i) (pool.worker_node(i) == 0 ? on0 : on1) += 1;
            std::printf("  workers on node 0: %d, on node 1: %d, host on node %d\n", on0, on1, pool.host_node());
            expect(on1 > 0 && pool.host_node() == 0, "workers on both nodes, the host on the GPU's node");
            int worse = 0;
            for (int r = 0; r < 10; ++r) worse += all_same(k.exec(pool, A.data(), stray.data()), refA) ? 0 : 1;
            expect(worse == 0, "mirroring ON is bit-identical to OFF (run, run_split, run_split_multi; 10 repeats)");
            pool.set_mirror(plat::ArenaMirror{});
            expect(!pool.mirrored() && all_same(k.exec(pool, A.data(), stray.data()), refA), "set_mirror(inactive) switches it off, still identical");
        }

        // ---- 2. which copy does a worker read?  Build the replica from A, THEN poison the primary with B.
        auto poisoned = [&](const std::vector<int>& n0, const std::vector<int>& n1, int host_node, auto&& body) {
            cpu::set_pool_numa(layout(n0, n1, host_node));
            cpu::ExpertPool pool;
            std::vector<uint8_t> primary = A;   // fresh primary holding A
            plat::NumaReplicas rep;
            plat::MirrorOptions mo;
            mo.faked = true; mo.threads = 2; mo.lock = false; mo.samples = 4;
            std::vector<std::string> log;
            rep.build(primary.data(), cap, cap, kbox.plan, kbox.topo, mo, log);   // replica = A
            std::memcpy(primary.data(), B.data(), cap);                           // primary = B: POISONED
            pool.set_mirror(rep.mirror());
            body(pool, primary.data());
        };
        std::vector<int> every = kbox.node0;
        every.insert(every.end(), kbox.node1.begin(), kbox.node1.end());
        std::sort(every.begin(), every.end());
        poisoned({}, every, 1,
                 [&](cpu::ExpertPool& pool, const uint8_t* primary) {
                     bool all1 = true;
                     for (int i = 0; i < pool.workers(); ++i) all1 = all1 && pool.worker_node(i) == 1;
                     expect(all1 && pool.host_node() == 1, "every worker and the host are on node 1");
                     bool ok = true;
                     for (int r = 0; r < 10 && ok; ++r) ok = all_same(k.exec(pool, primary, stray.data()), refA);
                     expect(ok, "all on node 1, primary POISONED: every output is the REPLICA's (so node-1 workers read the replica)");
                 });
        poisoned(every, {}, 0,
                 [&](cpu::ExpertPool& pool, const uint8_t* primary) {
                     bool all0 = true;
                     for (int i = 0; i < pool.workers(); ++i) all0 = all0 && pool.worker_node(i) == 0;
                     expect(all0 && pool.host_node() == 0, "every worker and the host are on node 0");
                     bool ok = true;
                     for (int r = 0; r < 10 && ok; ++r) ok = all_same(k.exec(pool, primary, stray.data()), refB);
                     expect(ok, "all on node 0, primary POISONED: every output is the poisoned PRIMARY's (node-0 workers read the primary)");
                     expect(!all_same(k.exec(pool, primary, stray.data()), refA), "...and so differ from the replica's: the control");
                 });
        if (k.whole_expert_api0) {
            // mixed: `run` jobs are whole experts, so each is the replica's (A) or the primary's (B), never a blend - and a
            // stray blob (job kNexp) is neither: it is read where it is, by everyone
            poisoned(kbox.node0, kbox.node1, 0, [&](cpu::ExpertPool& pool, const uint8_t* primary) {
                const size_t per = k.api0_floats_per_job;
                long from_a = 0, from_b = 0, blended = 0, stray_bad = 0;
                for (int r = 0; r < 40; ++r) {
                    const auto outs = k.exec(pool, primary, stray.data());
                    for (int e = 0; e < kNexp; ++e) {
                        const bool a = std::memcmp(&outs[0][(size_t) e * per], &refA[0][(size_t) e * per], per * 4) == 0;
                        const bool b = std::memcmp(&outs[0][(size_t) e * per], &refB[0][(size_t) e * per], per * 4) == 0;
                        from_a += a; from_b += b; blended += (!a && !b);
                    }
                    // stray: identical in refA and refB (same blob), so either reference will do
                    stray_bad += std::memcmp(&outs[0][(size_t) kNexp * per], &refA[0][(size_t) kNexp * per], per * 4) != 0;
                }
                std::printf("  mixed nodes, 40 batches x %d experts: %ld read the replica (node 1), %ld the primary (node 0), %ld blended\n",
                            kNexp, from_a, from_b, blended);
                expect(blended == 0, "every whole-expert job read ONE copy only");
                expect(stray_bad == 0, "a blob outside the arena is left alone, on every thread");
                expect(from_a > 0 && from_b > 0, "both nodes' workers took jobs, each from its own node's copy");
            });
        }
    }
    // ---- 5. (audit A4) more nodes than the pool used to track, and one copy per socket (sub-NUMA clustering).  Q2_0 experts only:
    // the pool's translation does not depend on the kernel family.
    {
        Kind& k = kinds[0];
        std::printf("\n  --- many nodes, shared copies (%s experts) ---\n", k.name.c_str());
        cpu::clear_pool_numa();
        const size_t cap = arena_bytes(k);
        std::vector<uint8_t> A, B, stray(k.blob);
        fill_arena(k, A, 1);
        fill_arena(k, B, 2);
        k.fill_one(stray.data(), 99);
        std::vector<std::vector<float>> refA, refB;
        {
            cpu::ExpertPool plain;
            refA = k.exec(plain, A.data(), stray.data());
            refB = k.exec(plain, B.data(), stray.data());
        }
        // build the replicas of `ft.plan` from A, then poison the primary with B, and hand the body a pool laid out as `pn`
        auto poisoned_on = [&](FakeTree& ft, const cpu::PoolNuma& pn, auto&& body) {
            cpu::set_pool_numa(pn);
            cpu::ExpertPool pool;
            std::vector<uint8_t> primary = A;
            plat::NumaReplicas rep;
            plat::MirrorOptions mo;
            mo.faked = true; mo.threads = 2; mo.lock = false; mo.samples = 4;
            std::vector<std::string> log;
            rep.build(primary.data(), cap, cap, ft.plan, ft.topo, mo, log);
            std::memcpy(primary.data(), B.data(), cap);   // primary = B: POISONED
            pool.set_mirror(rep.mirror());
            body(pool, primary.data(), rep.mirror());
        };
        const std::vector<int> cores = cpu::physical_cores(false);
        std::vector<int> every = cores;
        std::sort(every.begin(), every.end());
        {   // TEN nodes, the workers on node index 9 (nodes 1-8 memory-only): it used to read the primary - the replica of index >= 8 was built and never read
            FakeTree ft;
            std::vector<std::vector<int>> cpus(10);
            cpus[0] = {every.front()};
            cpus[9] = std::vector<int>(every.begin() + 1, every.end());
            const bool made = every.size() >= 2 && make_fake_tree(ft, "n10", cpus, {}, 0, plat::NumaMode::Mirror, cap);
            expect(made && ft.plan.replica_nodes == std::vector<int>{9}, "ten nodes: the plan puts a replica on node 9 (the only other node with CPUs)");
            if (made) {
                cpu::PoolNuma pn;
                pn.node_cpus.assign(10, {});
                pn.node_cpus[9] = every;       // every worker (and the host) on node index 9
                pn.host_node = 9;
                poisoned_on(ft, pn, [&](cpu::ExpertPool& pool, const uint8_t* primary, const plat::ArenaMirror& m) {
                    expect(m.copy.size() == 10 && m.copy[9] != nullptr && m.copy[9] != m.primary, "the mirror has a slot, and a replica, for node index 9");
                    bool all9 = true;
                    for (int i = 0; i < pool.workers(); ++i) all9 = all9 && pool.worker_node(i) == 9;
                    expect(all9 && pool.host_node() == 9, "every worker and the host are on node index 9");
                    bool ok = true;
                    for (int r = 0; r < 10 && ok; ++r) ok = all_same(k.exec(pool, primary, stray.data()), refA);
                    expect(ok, "primary POISONED: every output is node 9's REPLICA's (index >= 8 is read now)");
                });
            }
        }
        {   // SNC: four nodes, sockets {0,1} and {2,3}, the GPU on node 1; auto mode makes ONE replica, on node 2, that node 3 reads too
            FakeTree ft;
            const bool four = every.size() >= 4;
            std::vector<std::vector<int>> cpus(4);
            if (four) {
                const size_t q = every.size() / 4;
                for (size_t n = 0; n < 4; ++n)
                    cpus[n].assign(every.begin() + (long) (n * q), n == 3 ? every.end() : every.begin() + (long) ((n + 1) * q));
            }
            const bool made = four && make_fake_tree(ft, "snc", cpus, {0, 0, 1, 1}, 1, plat::NumaMode::Auto, cap);
            if (!four) std::printf("  (fewer than 4 physical cores: the SNC case is SKIPPED)\n");
            if (four) {
                expect(made && ft.plan.replica_nodes == std::vector<int>{2}, "SNC auto: one replica, on node 2 (not one per node)");
                const std::vector<std::pair<int, int>> want{{0, 1}, {1, 1}, {2, 2}, {3, 2}};
                expect(ft.plan.reads == want, "...nodes 0,1 read the primary, nodes 2,3 read the replica");
            }
            if (made) {
                auto only = [&](int node, int host) {
                    cpu::PoolNuma pn;
                    pn.node_cpus.assign(4, {});
                    pn.node_cpus[(size_t) node] = every;
                    pn.host_node = host;
                    return pn;
                };
                poisoned_on(ft, only(3, 3), [&](cpu::ExpertPool& pool, const uint8_t* primary, const plat::ArenaMirror& m) {
                    expect(m.copy.size() == 4 && m.copy[3] == m.copy[2] && m.copy[3] != m.primary && m.copy[0] == m.primary && m.copy[1] == m.primary,
                           "the mirror: nodes 0,1 -> primary, nodes 2,3 -> the one replica");
                    bool ok = true;
                    for (int r = 0; r < 10 && ok; ++r) ok = all_same(k.exec(pool, primary, stray.data()), refA);
                    expect(ok, "everything on node 3, primary POISONED: node 3 reads node 2's replica (its socket's copy)");
                });
                poisoned_on(ft, only(0, 0), [&](cpu::ExpertPool& pool, const uint8_t* primary, const plat::ArenaMirror&) {
                    bool ok = true;
                    for (int r = 0; r < 10 && ok; ++r) ok = all_same(k.exec(pool, primary, stray.data()), refB);
                    expect(ok, "everything on node 0, primary POISONED: node 0 reads the primary (its socket's copy is on node 1)");
                });
            }
        }
        {   // set_mirror between batches, repeatedly, from the host thread: the wait at its entry returns at once when the workers are parked
            FakeBox kbox;
            if (make_fake_box(kbox, cap)) {
                cpu::set_pool_numa(layout(kbox.node0, kbox.node1, 0));
                cpu::ExpertPool pool;
                plat::NumaReplicas rep;
                plat::MirrorOptions mo;
                mo.faked = true; mo.threads = 2; mo.lock = false; mo.samples = 4;
                std::vector<std::string> log;
                rep.build(A.data(), cap, cap, kbox.plan, kbox.topo, mo, log);
                bool ok = true;
                for (int r = 0; r < 200 && ok; ++r) {
                    pool.set_mirror(r % 2 ? rep.mirror() : plat::ArenaMirror{});
                    ok = pool.mirrored() == (r % 2 == 1);
                    if (r % 20 == 0) ok = ok && all_same(k.exec(pool, A.data(), stray.data()), refA);
                }
                expect(ok, "set_mirror on and off 200 times between batches (it waits for the workers to be parked)");
            }
        }
        (void) refB;
    }
    cpu::clear_pool_numa();
    return bad;
}

#else
int numa_mirror_test() {
    std::printf("  the NUMA mirror test is Linux-only: SKIPPED\n");
    return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, synthetic = false, numa_mirror = false;
    const char* path = "pack/full/experts.bin";
    long long layer = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--synthetic") synthetic = true;
        else if (a == "--numa-mirror") numa_mirror = synthetic = true;
        else if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "usage: pool_test [--selftest] [--synthetic] [--numa-mirror] [--file P] [--layer N]\n"); return 2; }
    }

    const cpu::CpuFeatures feat = cpu::cpu_features();
    if (feat.usable())
        std::printf("  %-44s %s\n", "Q2_0 kernel build under test", cpu::expert_isa_name(cpu::cpu_q2_expert_isa()));
    if (!feat.usable()) {
        std::printf("  CPU lacks %s - the VNNI path cannot run here; pool test SKIPPED, not passed.\n",
                    feat.reason());
        std::printf("\npool: 0 failures, 1 SKIPPED\n");
        return 0;
    }

    int bad = 0;

    if (numa_mirror) {
        std::printf("\nNUMA mirror (WP-F):\n");
        bad = numa_mirror_test();
        std::printf("\npool: %d failures\n", bad);
        if (bad) return 1;
        if (selftest) std::printf("pool_test OK\n");
        return 0;
    }

    // ---- ten experts off one layer, which is exactly what a token uses
    const int NEXP = 10;
    std::vector<std::vector<uint8_t>> blobs((size_t) NEXP);
    std::mt19937 blob_rng(31337);
    for (int e = 0; e < NEXP; ++e)
        if (synthetic) {
            auto& b = blobs[(size_t) e];
            b.assign(cpu::BLOB, 0);
            for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) blob_rng();
            for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
                const uint16_t h = (uint16_t) (0x1C00 + blob_rng() % 0x0800);
                std::memcpy(b.data() + i, &h, 2);
            }
        } else if (!read_blob(path, layer * 512 + e, blobs[(size_t) e])) {
            std::fprintf(stderr, "cannot read expert %d of layer %lld from %s\n", e, layer, path);
            return 2;
        }

    std::mt19937 rng(99);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ActQ act;
    cpu::act_quant_q8_1(x.data(), cpu::H, act);

    // ---- serial reference
    std::vector<float> ref((size_t) NEXP * cpu::H);
    {
        cpu::ExpertScratch ws;
        for (int e = 0; e < NEXP; ++e)
            cpu::s2_expert_vnni_q(blobs[(size_t) e].data(), act, ref.data() + (size_t) e * cpu::H, ws);
    }

    // ---- the pool
    const std::vector<int> cores = cpu::physical_cores(true);
    const int hw = (int) std::thread::hardware_concurrency();
    std::printf("  %-44s %d logical, %d physical (skipping the first)\n", "cores the pool will use",
                hw, (int) cores.size());

    cpu::ExpertPool pool;
    std::printf("  %-44s %d\n", "workers", pool.workers());

    std::vector<cpu::ExpertJob> jobs((size_t) NEXP);
    std::vector<float> got((size_t) NEXP * cpu::H);
    std::vector<float> weights((size_t) NEXP);
    for (int e = 0; e < NEXP; ++e) weights[(size_t) e] = 0.1f * (float) (e + 1);

    // ---- 1 + 2: same answer as serial, and every job exactly once.
    // The output buffer starts at a sentinel no real result can equal, so a job that never runs shows up as
    // an untouched slot instead of as a plausible number.
    const float SENTINEL = -1.2345e33f;
    for (int e = 0; e < NEXP; ++e) {
        jobs[(size_t) e].blob = blobs[(size_t) e].data();
        jobs[(size_t) e].act = &act;
        jobs[(size_t) e].out = got.data() + (size_t) e * cpu::H;
        jobs[(size_t) e].weight = weights[(size_t) e];
        jobs[(size_t) e].slot = e;
        std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
    }
    pool.run(jobs.data(), NEXP);

    long long not_run = 0, diff = 0;
    float worst = 0.f;
    for (int e = 0; e < NEXP; ++e)
        for (int i = 0; i < cpu::H; ++i) {
            const float a = ref[(size_t) e * cpu::H + i], b = got[(size_t) e * cpu::H + i];
            if (b == SENTINEL) { ++not_run; continue; }
            if (std::memcmp(&a, &b, 4) != 0) ++diff;
            worst = std::fmax(worst, std::fabs(a - b));
        }
    std::printf("  %-44s %s (%lld of %d outputs untouched)\n", "every job ran exactly once",
                not_run ? "*** NO ***" : "yes", not_run, NEXP * cpu::H);
    if (not_run) ++bad;
    // BIT-IDENTICAL, not "close".  Each worker has a private scratch and the activation is shared read-only,
    // so any difference at all is a race or a scratch collision - and a tolerance would hide exactly that.
    std::printf("  %-44s %s (%lld of %d differ, worst |d| %.3e)\n", "identical to the serial run",
                diff ? "*** NO ***" : "yes", diff, NEXP * cpu::H, (double) worst);
    if (diff) ++bad;

    // ---- 3: repeated batches.  A park protocol that works once and corrupts on the second call is the
    //        exact failure this loop is here to catch.
    {
        int repeats_bad = 0;
        for (int r = 0; r < 200; ++r) {
            for (int e = 0; e < NEXP; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), NEXP);
            for (int e = 0; e < NEXP && !repeats_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++repeats_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (200 consecutive batches)\n", "repeated batches stay correct",
                    repeats_bad ? "*** NO ***" : "yes");
        if (repeats_bad) ++bad;
    }

    // ---- 4: batch sizes either side of the worker count
    {
        int size_bad = 0;
        for (int n : {1, 2, pool.workers() - 1 > 0 ? pool.workers() - 1 : 1, pool.workers(),
                      pool.workers() + 1, NEXP}) {
            if (n < 1 || n > NEXP) continue;
            for (int e = 0; e < n; ++e) std::fill(jobs[(size_t) e].out, jobs[(size_t) e].out + cpu::H, SENTINEL);
            pool.run(jobs.data(), n);
            for (int e = 0; e < n && !size_bad; ++e)
                for (int i = 0; i < cpu::H; ++i)
                    if (std::memcmp(&ref[(size_t) e * cpu::H + i], &got[(size_t) e * cpu::H + i], 4) != 0) {
                        ++size_bad;
                        break;
                    }
        }
        std::printf("  %-44s %s (1, w-1, w, w+1, 10)\n", "batch sizes around the worker count",
                    size_bad ? "*** NO ***" : "yes");
        if (size_bad) ++bad;
    }

    // ---- and the number that matters for the ledger: c for one layer, 10 experts over 48 layers.
    //
    // BEST OF N, not a mean.  The first version divided a 10-run total by 10 and reported 6.604 ms per layer -
    // 2.1 GB/s against L9's measured 42.55 - purely because a CUDA build was running on the same 6-core
    // machine at the time.  A mean over a contended machine measures the contention; a best-of measures the
    // instrument, and every other timing tool in this project already takes the best.  The spread is printed
    // so a contended run is visible rather than being read as a regression.
    {
        const int REPS = 20;
        std::vector<double> t((size_t) REPS, 0.0);
        for (int r = 0; r < REPS; ++r) {
            const double t0 = now_ms();
            pool.run(jobs.data(), NEXP);
            t[(size_t) r] = now_ms() - t0;
        }
        std::vector<double> sorted = t;
        std::sort(sorted.begin(), sorted.end());
        const double best = sorted.front(), median = sorted[sorted.size() / 2], worst = sorted.back();
        const double gbs = (double) NEXP * cpu::BLOB / (best * 1e-3) / 1e9;
        std::printf("\n  %-44s %7.3f ms   (%.1f GB/s)\n", "10 experts, one layer (best of 20)",
                    best, gbs);
        std::printf("  %-44s %7.3f / %7.3f ms   (spread %.2fx)\n", "median / worst", median, worst,
                    median / (best > 0 ? best : 1));
        std::printf("  %-44s %7.3f ms   -> %.1f tok/s for the full 48 layers\n",
                    "extrapolated to 48 layers", best * 48, 1000.0 / (best * 48));
        std::printf("  L9 measured 42.55 GB/s on 6 cores; this pool uses %d workers (core 0 is left to the\n",
                    pool.workers());
        std::printf("  host loop), so %.1f GB/s x 6/%d = %.1f GB/s is the per-core comparison.\n",
                    gbs, pool.workers(), gbs * 6.0 / pool.workers());
        if (median / (best > 0 ? best : 1) > 1.5)
            std::printf("  *** the spread is over 1.5x: this machine was CONTENDED and the median is not a\n"
                        "      property of the pool.  Re-run on a quiet machine before quoting it. ***\n");
    }

    std::printf("\npool: %d failures\n", bad);
    if (bad) return 1;
    if (selftest) std::printf("pool_test OK\n");
    return 0;
}
