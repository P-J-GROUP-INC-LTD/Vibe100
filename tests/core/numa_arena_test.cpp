// tests/core/numa_arena_test.cpp - Vibe100 WP-F: the mirrored resident arena end to end (ArenaExpertSource + ExpertPool),
// on a one-node machine, with or without a GPU.
//
// The topology is FAKE: a sysfs tree under a temp directory (STRATA_SYSFS_ROOT) that calls the first half of this machine's
// cores node 0 (where it puts the "GPU") and the second half node 1.  Node 1 does not exist for the kernel, so the bind to
// it fails and the placement check finds the pages on node 0: with a fake topology that is REPORTED and the build goes on
// (a real topology would drop the replica; platform/numa_test.cpp checks that), which is exactly what lets the rest -
// ArenaExpertSource's wiring, the notes it prints, the replica's contents, the pool reading it - run here.
//
//   1. --numa off: one copy, an inactive mirror, a note that says why.
//   2. --numa mirror: the arena opens, `blob()` still names the PRIMARY (the only copy a GPU DMA may see), one replica holds the
//      primary's bytes at every expert, the notes carry the decision and the placement lines, and the pool built from the
//      source's layout and mirror computes bit-identically to a pool with no mirror.
//   3. The refusals ArenaExpertSource makes: a one-node box, --shared-expert-arena, and (numa_single_copy_notes) the mmap /
//      low-RAM modes - each says why, and keeps one copy.
//
// It needs no GPU: cudaHostRegister failing is a path the arena handles (the note says the arena is not pinned).
#include "strata/core/expert_source.hpp"
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/platform/numa.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
namespace cpu = strata::kernels::cpu;
namespace plat = strata::platform;
using strata::core::ArenaExpertSource;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-96s %s\n", what.c_str(), ok ? "ok" : "*** FAIL ***");
    if (!ok) ++g_fail;
}

bool has(const std::vector<std::string>& notes, const std::string& needle) {
    for (const auto& n : notes)
        if (n.find(needle) != std::string::npos) return true;
    return false;
}

void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

void write_node(const fs::path& root, int id, const std::vector<int>& cpus) {
    const fs::path d = root / "devices/system/node" / ("node" + std::to_string(id));
    put(d / "cpulist", plat::format_cpulist(cpus) + "\n");
    char t[256];
    std::snprintf(t, sizeof t,
                  "Node %d MemTotal: 67108864 kB\nNode %d MemFree: 52428800 kB\nNode %d Active(file): 1048576 kB\n"
                  "Node %d Inactive(file): 1048576 kB\n", id, id, id, id);
    put(d / "meminfo", t);
}

struct Temp {
    fs::path root;
    Temp() {
        root = fs::temp_directory_path() / ("strata_numa_arena_test_" + std::to_string((long) getpid()));
        fs::remove_all(root);
        fs::create_directories(root);
    }
    ~Temp() { std::error_code ec; fs::remove_all(root, ec); }
};

constexpr int kLayers = 2, kExperts = 4;

}  // namespace

int main() {
    unsetenv("STRATA_NUMA_MIRROR");
    unsetenv("STRATA_NUMA_HEADROOM_GIB");
    Temp tmp;

    // ---- a synthetic pack: 2 layers x 4 experts at the real blob size, random bytes (scales kept to sane fp16)
    const fs::path pack = tmp.root / "pack";
    fs::create_directories(pack);
    std::string err;
    if (!cpu::expert_layout_load(pack.string(), kLayers, kExperts, err)) { std::printf("layout: %s\n", err.c_str()); return 2; }
    const uint64_t total = cpu::expert_layout().total;
    {
        std::vector<uint8_t> all((size_t) total);
        std::mt19937 rng(5);
        for (int e = 0; e < kLayers * kExperts; ++e) {
            uint8_t* b = all.data() + (size_t) e * cpu::BLOB;
            for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
            for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
                const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
                std::memcpy(b + i, &h, 2);
            }
        }
        std::ofstream(pack / "experts.bin", std::ios::binary).write((const char*) all.data(), (std::streamsize) all.size());
    }

    // ---- the fake boxes
    const std::vector<int> cores = cpu::physical_cores(false);
    if (cores.size() < 2) { std::printf("needs two physical cores to call one of them a second node: SKIPPED\n"); return 0; }
    const size_t half = cores.size() / 2;
    std::vector<int> n0(cores.begin(), cores.begin() + (long) half), n1(cores.begin() + (long) half, cores.end());
    std::sort(n0.begin(), n0.end());
    std::sort(n1.begin(), n1.end());
    const fs::path two = tmp.root / "sys2", one = tmp.root / "sys1";
    write_node(two, 0, n0);
    write_node(two, 1, n1);
    {   // the GPU (if there is one) is on node 0; without one the node is unknown and assumed to be 0 anyway
        const std::string bdf = strata::core::gpu_pci_bus_id(0);
        std::string low = bdf;
        for (char& c : low) c = (char) std::tolower((unsigned char) c);
        if (!low.empty()) put(two / "bus/pci/devices" / low / "numa_node", "0\n");
        std::printf("  GPU bus id: %s\n", bdf.empty() ? "(no GPU)" : bdf.c_str());
    }
    write_node(one, 0, cores);
    const std::string two_s = two.string(), one_s = one.string();

    // ---- 1. off
    setenv("STRATA_SYSFS_ROOT", two_s.c_str(), 1);
    {
        ArenaExpertSource a;
        a.set_numa(plat::NumaMode::Off);
        check(a.open(pack.string(), kLayers, kExperts, 2, err), "off: the arena opens");
        check(!a.mirror().active() && a.replicas().count() == 0, "off: one copy, no mirror");
        check(has(a.numa_notes(), "--numa off"), "off: the notes say so");
        for (const auto& n : a.numa_notes()) std::printf("    %s\n", n.c_str());
    }

    // ---- 2. mirror
    {
        ArenaExpertSource a;
        a.set_numa(plat::NumaMode::Mirror);
        check(a.open(pack.string(), kLayers, kExperts, 2, err), "mirror: the arena opens");
        for (const auto& n : a.numa_notes()) std::printf("    %s\n", n.c_str());
        check(a.mirror().active() && a.replicas().count() == 1, "mirror: one replica kept (a fake topology reports its failed bind and carries on)");
        check(has(a.numa_notes(), "MIRRORED") && has(a.numa_notes(), "primary") && has(a.numa_notes(), "replica on node 1") &&
                  has(a.numa_notes(), "only copy any GPU DMA reads"),
              "mirror: the notes name the primary, the replica and who reads which");
        check(has(a.numa_notes(), "sampled pages"), "mirror: ...and the placement that was checked");
        const uint8_t* b00 = a.blob(0, 0);
        check(b00 == a.mirror().primary && a.mirror().bytes >= total, "blob() names the PRIMARY; the mirror covers the whole arena");
        const uint8_t* rep = a.mirror().copy.size() == 2 ? a.mirror().copy[1] : nullptr;
        check(rep != nullptr && rep != b00 && std::memcmp(rep, b00, total) == 0, "the replica holds the primary's bytes (every expert)");
        check(a.mirror().copy[0] == b00, "the primary's own slot of the mirror is the primary");
        check(a.pool_numa().active() && a.pool_numa().host_node == 0, "pool layout: two nodes, the host on the GPU's node 0");

        // the pool, with this source's layout and mirror, against a pool with none: every API, bit for bit
        std::mt19937 rng(9);
        std::normal_distribution<float> gauss(0.f, 1.f);
        std::vector<float> x(cpu::H);
        for (auto& v : x) v = gauss(rng);
        cpu::ActQ act;
        cpu::act_quant_q8_1(x.data(), cpu::H, act);
        const int N = kLayers * kExperts;
        auto run_all = [&](cpu::ExpertPool& pool) {
            std::vector<float> out((size_t) N * cpu::H, 0.f);
            std::vector<cpu::ExpertJob> jobs((size_t) N);
            for (int i = 0; i < N; ++i) {
                jobs[(size_t) i].blob = a.blob(i / kExperts, i % kExperts);
                jobs[(size_t) i].act = &act;
                jobs[(size_t) i].out = out.data() + (size_t) i * cpu::H;
            }
            pool.run(jobs.data(), N);
            std::vector<float> out2((size_t) N * cpu::H, 0.f);
            for (int i = 0; i < N; ++i) jobs[(size_t) i].out = out2.data() + (size_t) i * cpu::H;
            pool.run_split(jobs.data(), N);
            out.insert(out.end(), out2.begin(), out2.end());
            return out;
        };
        cpu::clear_pool_numa();
        std::vector<float> plain;
        {
            cpu::ExpertPool pool;
            plain = run_all(pool);
        }
        cpu::set_pool_numa(a.pool_numa());
        std::vector<float> mirrored;
        {
            cpu::ExpertPool pool;
            pool.set_mirror(a.mirror());
            check(pool.mirrored(), "the pool took the source's mirror");
            for (int r = 0; r < 5; ++r) {
                mirrored = run_all(pool);
                if (mirrored.size() != plain.size() || std::memcmp(mirrored.data(), plain.data(), plain.size() * 4) != 0) break;
            }
        }
        cpu::clear_pool_numa();
        check(mirrored.size() == plain.size() && std::memcmp(mirrored.data(), plain.data(), plain.size() * 4) == 0,
              "the mirrored pool is bit-identical to the pool without a mirror (run and run_split)");
    }

    // ---- 3. the refusals
    {
        ArenaExpertSource a;
        a.set_numa(plat::NumaMode::Auto);
        const std::string shared = (tmp.root / "arena.shm").string();
        const bool opened = a.open(pack.string(), kLayers, kExperts, 2, err, 0, shared);
        for (const auto& n : a.numa_notes()) std::printf("    %s\n", n.c_str());
        check(opened && !a.mirror().active() && has(a.numa_notes(), "--shared-expert-arena"),
              "--shared-expert-arena: one copy, and the note names the flag");
    }
    setenv("STRATA_SYSFS_ROOT", one_s.c_str(), 1);
    {
        ArenaExpertSource a;
        a.set_numa(plat::NumaMode::Auto);
        check(a.open(pack.string(), kLayers, kExperts, 2, err) && !a.mirror().active() && has(a.numa_notes(), "one NUMA node"),
              "a one-node box: one copy, the note says so");
        ArenaExpertSource m;
        m.set_numa(plat::NumaMode::Mirror);
        check(m.open(pack.string(), kLayers, kExperts, 2, err) && !m.mirror().active() && has(m.numa_notes(), "WARNING"),
              "--numa mirror on a one-node box: refused, as a warning");
    }
    setenv("STRATA_SYSFS_ROOT", two_s.c_str(), 1);
    {
        const auto notes = strata::core::numa_single_copy_notes(plat::NumaMode::Auto, "--mmap-experts, the low-RAM mode");
        for (const auto& n : notes) std::printf("    %s\n", n.c_str());
        check(has(notes, "--mmap-experts, the low-RAM mode") && has(notes, "NOT mirrored") && has(notes, "one copy"),
              "mmap / low-RAM: one copy, the note names the mode");
    }
    {   // headroom: a node must keep STRATA_NUMA_HEADROOM_GIB beyond the copy - a 60 GiB headroom cannot be met by 50 GiB nodes
        setenv("STRATA_NUMA_HEADROOM_GIB", "60", 1);
        ArenaExpertSource a;
        a.set_numa(plat::NumaMode::Auto);
        check(a.open(pack.string(), kLayers, kExperts, 2, err) && !a.mirror().active() && has(a.numa_notes(), "usable"),
              "not enough room on a node: one copy, the note gives the numbers");
        for (const auto& n : a.numa_notes()) std::printf("    %s\n", n.c_str());
        unsetenv("STRATA_NUMA_HEADROOM_GIB");
    }
    unsetenv("STRATA_SYSFS_ROOT");

    std::printf("\nnuma_arena: %d failure(s)\n", g_fail);
    if (g_fail == 0) std::printf("numa_arena_test OK\n");
    return g_fail == 0 ? 0 : 1;
}
