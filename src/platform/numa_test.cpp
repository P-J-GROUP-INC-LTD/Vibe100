// src/platform/numa_test.cpp - Vibe100 WP-F: the NUMA layer, on a machine with ONE node and no GPU.
//
//   (a) the topology parser on fake sysfs trees built in a temp directory: one node, two nodes, cpu lists with ranges
//       ("0-11,24-35"), non-contiguous node ids, a memory-only node, a GPU whose numa_node is -1 / 0 / 1 / absent / a node
//       that does not exist, malformed files - and STRATA_SYSFS_ROOT / the argument picking the tree;
//   (b) the mirror planner's refusal paths (`plan_arena_mirror`, a pure function): --numa off, no topology, one node, the
//       mmap / low-RAM / budget modes, a shared arena, not enough room on a node (with the headroom and the file cache in
//       the arithmetic), a taskset that leaves a node out, and the environment override;
//   (c) the real syscalls on this machine: a node-bound buffer lands where it was bound (checked with move_pages), a bind
//       to a node that does not exist FAILS and is reported, the pinned copy is exact;
//   (d) the replica builder with a fake two-node topology: a faked topology reports a failed bind and carries on (the replica
//       is a good copy, `copy[]` indexes by node), a real one DROPS a replica whose pages are not on its node.
//
//   (e) WP-F audit A4: the 2 MiB hugetlb pool per node in the topology and the planner (a pool that holds a copy takes no normal
//       memory; half a copy helps nobody), THP as the fallback and page-size reporting, the SIGBUS guard (a prefault that fails
//       remaps the copy as THP), pinning a replica through a callback (and mlock where that fails), and sub-NUMA clustering:
//       4 nodes on 2 packages (one copy per socket in auto mode, per node in mirror mode), a socket that is all one package,
//       and 10 nodes (nothing is capped at 8).
//
// The pool's half (a worker reads the copy of its node) is in pool_test.cpp (--numa-mirror).
#include "strata/platform/numa.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace fs = std::filesystem;
namespace plat = strata::platform;

namespace {

int g_fail = 0, g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("  *** FAIL: %s\n", what.c_str()); }
}

constexpr uint64_t GiB = 1ull << 30;

// ---------------------------------------------------------------- fake sysfs trees
void put(const fs::path& p, const std::string& text) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << text;
}

// A node with `total`/`free`/`file` GiB (file = Active(file) + Inactive(file), split 1:3).
void node(const fs::path& root, int id, const std::string& cpulist, double total, double free_, double file = 0) {
    const fs::path d = root / "devices/system/node" / ("node" + std::to_string(id));
    put(d / "cpulist", cpulist + "\n");
    auto kb = [](double g) { return (unsigned long long) (g * 1024.0 * 1024.0); };
    char t[1024];
    std::snprintf(t, sizeof t,
                  "Node %d MemTotal:       %llu kB\nNode %d MemFree:        %llu kB\nNode %d MemUsed:        %llu kB\n"
                  "Node %d Active(file):   %llu kB\nNode %d Inactive(file): %llu kB\nNode %d Shmem:  0 kB\n",
                  id, kb(total), id, kb(free_), id, kb(total - free_), id, kb(file) / 4, id, kb(file) - kb(file) / 4, id);
    put(d / "meminfo", t);
}

void gpu(const fs::path& root, const std::string& bdf_lower, int numa_node) {
    put(root / "bus/pci/devices" / bdf_lower / "numa_node", std::to_string(numa_node) + "\n");
}

// `cpu<N>/topology/physical_package_id` for each CPU of a cpulist; `<node>/hugepages/hugepages-2048kB/{free,nr}_hugepages`
void packages(const fs::path& root, const std::vector<int>& cpus, int pkg) {
    for (int c : cpus) put(root / "devices/system/cpu" / ("cpu" + std::to_string(c)) / "topology/physical_package_id", std::to_string(pkg) + "\n");
}
void hugepool(const fs::path& root, int node, unsigned long free_pages, unsigned long total_pages) {
    const fs::path d = root / "devices/system/node" / ("node" + std::to_string(node)) / "hugepages/hugepages-2048kB";
    put(d / "free_hugepages", std::to_string(free_pages) + "\n");
    put(d / "nr_hugepages", std::to_string(total_pages) + "\n");
}
void thp_mode(const fs::path& root, const std::string& mode) {
    const std::string all[] = {"always", "madvise", "never"};
    std::string text;
    for (const auto& m : all) text += (text.empty() ? "" : " ") + (m == mode ? "[" + m + "]" : m);
    put(root / "kernel/mm/transparent_hugepage/enabled", text + "\n");
}
std::vector<int> range(int a, int b) { std::vector<int> v; for (int i = a; i < b; ++i) v.push_back(i); return v; }

struct TempRoot {
    fs::path root;
    explicit TempRoot(const std::string& tag) {
        root = fs::temp_directory_path() / ("strata_numa_test_" + std::to_string((long) getpid()) + "_" + tag);
        fs::remove_all(root);
        fs::create_directories(root);
    }
    ~TempRoot() { std::error_code ec; fs::remove_all(root, ec); }
};

// the target box: 2 sockets, 12 cores + 12 SMT siblings each, 192 GiB each
void target_box(const fs::path& r, double free0 = 150, double free1 = 170, double file = 20) {
    node(r, 0, "0-11,24-35", 192, free0, file);
    node(r, 1, "12-23,36-47", 192, free1, file);
}

// ---------------------------------------------------------------- (a)
void test_cpulist() {
    std::printf("[cpulist]\n");
    bool ok = false;
    auto v = plat::parse_cpulist("0-11,24-35", &ok);
    check(ok && v.size() == 24 && v.front() == 0 && v[11] == 11 && v[12] == 24 && v.back() == 35, "0-11,24-35 -> 24 cpus");
    check(plat::format_cpulist(v) == "0-11,24-35", "format round trip");
    v = plat::parse_cpulist("3", &ok);
    check(ok && v.size() == 1 && v[0] == 3, "single cpu");
    v = plat::parse_cpulist("", &ok);
    check(ok && v.empty(), "empty list is a valid (memory-only node)");
    v = plat::parse_cpulist("0-3\n", &ok);
    check(ok && v.size() == 4, "trailing newline");
    v = plat::parse_cpulist("2,0-1,1-2", &ok);
    check(ok && v.size() == 3 && v[0] == 0 && v[2] == 2, "unordered, overlapping -> sorted, unique");
    for (const char* bad : {"5-3", "a", "1,,2", "1-", "-3", "1 2", "0-3,", "9999999999"}) {
        plat::parse_cpulist(bad, &ok);
        check(!ok, std::string("rejects \"") + bad + "\"");
    }
    check(plat::format_cpulist({0, 2, 3, 4, 9}) == "0,2-4,9", "format ranges");
}

void test_topology() {
    std::printf("[topology on fake sysfs trees]\n");
    {   // one node, the VM
        TempRoot t("one");
        node(t.root, 0, "0-3", 15.7, 4.0, 8.0);
        const auto topo = plat::numa_discover(t.root.string());
        check(topo.available && topo.nodes.size() == 1 && !topo.multi(), "one node: available, not multi");
        check(topo.faked && topo.root == t.root.string(), "an explicit root is a faked topology");
        check(topo.nodes[0].cpus.size() == 4 && topo.nodes[0].has_meminfo, "cpus and meminfo read");
        check(topo.nodes[0].mem_total == (uint64_t) (15.7 * 1024 * 1024) * 1024, "MemTotal in bytes");
        const uint64_t want_usable = (uint64_t) ((4.0 + 8.0) * 1024 * 1024) * 1024;
        check(topo.nodes[0].usable_bytes() == want_usable, "usable = MemFree + Active(file) + Inactive(file)");
    }
    {   // the target: two sockets with SMT siblings numbered 24-47
        TempRoot t("two");
        target_box(t.root);
        auto topo = plat::numa_discover(t.root.string());
        check(topo.available && topo.multi() && topo.nodes.size() == 2, "two nodes");
        check(topo.nodes[0].cpus.size() == 24 && topo.nodes[1].cpus.size() == 24, "24 logical cpus each");
        check(topo.index_of_cpu(0) == 0 && topo.index_of_cpu(30) == 0 && topo.index_of_cpu(12) == 1 &&
                  topo.index_of_cpu(47) == 1 && topo.index_of_cpu(48) == -1, "index_of_cpu over the ranges");
        check(plat::format_cpulist(topo.nodes[1].cpus) == "12-23,36-47", "node 1 cpulist");

        // the GPU: -1 / 0 / 1 / absent / a node that is not there / upper-case address
        gpu(t.root, "0000:3b:00.0", -1);
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(!topo.gpu_node_known && topo.gpu_node == 0 && topo.gpu_bdf == "0000:3b:00.0", "numa_node -1: unknown, assumed node 0");
        check(topo.gpu_note.find("-1") != std::string::npos && plat::numa_describe(topo).find("ASSUMED") != std::string::npos,
              "...and it says so");
        gpu(t.root, "0000:3b:00.0", 0);
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(topo.gpu_node_known && topo.gpu_node == 0, "numa_node 0");
        gpu(t.root, "0000:3b:00.0", 1);
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(topo.gpu_node_known && topo.gpu_node == 1, "numa_node 1 (address lower-cased for the lookup)");
        check(plat::numa_describe(topo).find("GPU 0000:3b:00.0 on node 1") != std::string::npos, "describe names the GPU's node");
        plat::numa_set_gpu(topo, "0000:af:00.0");
        check(!topo.gpu_node_known && topo.gpu_node == 0, "no such device: unknown, node 0");
        gpu(t.root, "0000:5e:00.0", 7);
        plat::numa_set_gpu(topo, "0000:5E:00.0");
        check(!topo.gpu_node_known && topo.gpu_node == 0, "a node id with no node directory: unknown");
        plat::numa_set_gpu(topo, "");
        check(!topo.gpu_node_known, "no PCI address: unknown");
        gpu(t.root, "0000:3b:00.0", -1);
        setenv("STRATA_NUMA_GPU_NODE", "1", 1);
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(topo.gpu_node_known && topo.gpu_node == 1 && topo.gpu_note.find("STRATA_NUMA_GPU_NODE") != std::string::npos,
              "STRATA_NUMA_GPU_NODE answers where sysfs says -1");
        plat::numa_set_gpu(topo, "");
        check(topo.gpu_node_known && topo.gpu_node == 1, "...and without a PCI address too");
        setenv("STRATA_NUMA_GPU_NODE", "9", 1);
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(!topo.gpu_node_known && topo.gpu_node == 0 && topo.gpu_note.find("ignored") != std::string::npos,
              "a node that does not exist is ignored, and said so");
        unsetenv("STRATA_NUMA_GPU_NODE");
    }
    {   // non-contiguous ids, a memory-only node
        TempRoot t("gappy");
        node(t.root, 0, "0-7", 64, 40);
        node(t.root, 2, "8-15", 64, 40);
        node(t.root, 3, "", 128, 120);   // e.g. CXL / persistent memory: no CPUs
        auto topo = plat::numa_discover(t.root.string());
        check(topo.available && topo.nodes.size() == 3 && topo.nodes[1].id == 2 && topo.nodes[2].id == 3, "ids 0,2,3 ascending");
        check(topo.nodes[2].cpus.empty(), "memory-only node has no cpus");
        check(topo.index_of_node(2) == 1 && topo.index_of_node(1) == -1, "index_of_node");
        gpu(t.root, "0000:01:00.0", 2);
        plat::numa_set_gpu(topo, "0000:01:00.0");
        check(topo.gpu_node_known && topo.gpu_node == 2, "GPU on node 2");
    }
    {   // not a NUMA machine / broken files
        TempRoot t("none");
        auto topo = plat::numa_discover(t.root.string());
        check(!topo.available && !topo.why.empty(), "no node directory: not available, with a reason");
        check(plat::numa_describe(topo).find("not available") != std::string::npos, "describe says so");
        put(t.root / "devices/system/node/node0/cpulist", "0-x\n");
        topo = plat::numa_discover(t.root.string());
        check(!topo.available && topo.why.find("cpu list") != std::string::npos, "malformed cpulist: not available");
        fs::remove_all(t.root / "devices");
        node(t.root, 0, "0-3", 8, 4);
        fs::remove(t.root / "devices/system/node/node0/meminfo");
        topo = plat::numa_discover(t.root.string());
        check(topo.available && !topo.nodes[0].has_meminfo, "no meminfo: node kept, flagged");
    }
    {   // STRATA_SYSFS_ROOT, and the argument beating it
        TempRoot a("envA"), b("envB");
        node(a.root, 0, "0-3", 8, 4);
        target_box(b.root);
        setenv("STRATA_SYSFS_ROOT", b.root.c_str(), 1);
        auto topo = plat::numa_discover();
        check(topo.available && topo.nodes.size() == 2 && topo.faked && topo.root == b.root.string(), "STRATA_SYSFS_ROOT picks the tree");
        topo = plat::numa_discover(a.root.string());
        check(topo.nodes.size() == 1, "the argument beats the environment");
        unsetenv("STRATA_SYSFS_ROOT");
        topo = plat::numa_discover();
        check(!topo.faked && topo.root == "/sys", "default root is /sys");
        if (topo.available) std::printf("  this machine: %s\n", plat::numa_describe(topo).c_str());
    }
}

// ---------------------------------------------------------------- (b)
plat::NumaTopology two_nodes(TempRoot& t, double free0, double free1, double file0, double file1, int gpu_node) {
    fs::remove_all(t.root);
    node(t.root, 0, "0-11,24-35", 192, free0, file0);
    node(t.root, 1, "12-23,36-47", 192, free1, file1);
    if (gpu_node != -2) gpu(t.root, "0000:3b:00.0", gpu_node);
    auto topo = plat::numa_discover(t.root.string());
    plat::numa_set_gpu(topo, "0000:3B:00.0");
    return topo;
}

void test_planner() {
    std::printf("[the mirror planner: yes and every refusal]\n");
    TempRoot t("plan");
    auto topo = two_nodes(t, 150, 170, 20, 5, 1);
    const uint64_t arena = 50 * GiB;   // the biggest pack
    auto in = [&]() {
        plat::MirrorInputs m;
        m.mode = plat::NumaMode::Auto;
        m.topo = &topo;
        m.arena_bytes = arena;
        m.headroom = 6 * GiB;
        return m;
    };
    {
        const auto p = plat::plan_arena_mirror(in());
        check(p.mirror && p.primary_node == 1 && p.replica_nodes == std::vector<int>{0} && !p.gpu_node_assumed,
              "auto, 2 nodes, room: mirror, primary on the GPU's node 1, replica on node 0");
        check(p.why_not.empty(), "no refusal text when mirroring");
        std::printf("  plan: primary node %d, replicas on %zu node(s)\n", p.primary_node, p.replica_nodes.size());
    }
    {
        auto m = in(); m.mode = plat::NumaMode::Mirror;
        check(plat::plan_arena_mirror(m).mirror, "mirror mode, same");
        m.mode = plat::NumaMode::Off;
        const auto p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("--numa off") != std::string::npos, "off: refused, says why");
    }
    {   // GPU node unknown -> assumed node 0, flagged
        auto topo2 = two_nodes(t, 150, 170, 20, 5, -1);
        auto m = in(); m.topo = &topo2;
        const auto p = plat::plan_arena_mirror(m);
        check(p.mirror && p.primary_node == 0 && p.gpu_node_assumed && p.replica_nodes == std::vector<int>{1},
              "GPU node -1: mirrors, primary assumed on node 0, flagged");
        auto topo3 = two_nodes(t, 150, 170, 20, 5, -2);   // no numa_node file at all
        m.topo = &topo3;
        check(plat::plan_arena_mirror(m).gpu_node_assumed, "no numa_node file: assumed too");
        two_nodes(t, 150, 170, 20, 5, 1);                 // restore the tree `topo` was read from
    }
    {   // no topology / one node
        plat::NumaTopology none;
        none.why = "not Linux";
        auto m = in(); m.topo = &none;
        auto p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("not available") != std::string::npos && p.why_not.find("not Linux") != std::string::npos,
              "no topology: refused with the reason");
        m.topo = nullptr;
        check(!plat::plan_arena_mirror(m).mirror, "null topology: refused");
        TempRoot one("plan1");
        node(one.root, 0, "0-3", 16, 8);
        auto t1 = plat::numa_discover(one.root.string());
        m.topo = &t1;
        p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("one NUMA node") != std::string::npos, "one node: refused");
        std::printf("  one node  -> %s\n", p.why_not.c_str());
    }
    {   // the modes that keep one copy
        auto m = in(); m.full_ram = false; m.not_full_ram_why = "--mmap-experts, the low-RAM mode";
        auto p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("--mmap-experts, the low-RAM mode") != std::string::npos && p.why_not.find("one copy") != std::string::npos,
              "low-RAM / mmap: refused, names the mode, says one copy is kept");
        std::printf("  low-RAM   -> %s\n", p.why_not.c_str());
        m.not_full_ram_why.clear();
        check(!plat::plan_arena_mirror(m).mirror, "...without a reason string too");
        m = in(); m.shared_file = true;
        p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("--shared-expert-arena") != std::string::npos, "shared arena: refused");
        m = in(); m.arena_bytes = 0;
        check(!plat::plan_arena_mirror(m).mirror, "an empty arena: refused");
    }
    {   // room: arena + headroom on EVERY node that holds a copy
        auto lowmem = two_nodes(t, 30, 170, 10, 5, 1);   // node 0: 40 GiB usable, needs 56
        auto m = in(); m.topo = &lowmem;
        auto p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("node 0 has 40.0 GiB usable") != std::string::npos &&
                  p.why_not.find("needs 56.0 GiB") != std::string::npos && p.why_not.find("STRATA_NUMA_HEADROOM_GIB") != std::string::npos,
              "not enough room on the replica's node: refused, naming the node and both numbers");
        std::printf("  no room   -> %s\n", p.why_not.c_str());
        auto lowgpu = two_nodes(t, 150, 20, 5, 5, 1);    // the primary's node (1) is short
        m.topo = &lowgpu;
        p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("node 1 has") != std::string::npos, "...or on the primary's node");
        // exactly enough: usable = arena + headroom -> yes; one GiB less -> no
        auto edge = two_nodes(t, 40, 170, 16, 5, 1);     // node 0: 40 + 16 = 56 GiB usable
        m.topo = &edge;
        check(plat::plan_arena_mirror(m).mirror, "usable == arena + headroom: enough (and the file cache counts)");
        edge = two_nodes(t, 40, 170, 15, 5, 1);
        m.topo = &edge;
        check(!plat::plan_arena_mirror(m).mirror, "one GiB short: refused");
        m.headroom = 5 * GiB;
        check(plat::plan_arena_mirror(m).mirror, "...a smaller headroom admits it");
        setenv("STRATA_NUMA_HEADROOM_GIB", "9.5", 1);
        check(plat::numa_headroom_bytes() == (uint64_t) (9.5 * (double) GiB), "STRATA_NUMA_HEADROOM_GIB");
        unsetenv("STRATA_NUMA_HEADROOM_GIB");
        check(plat::numa_headroom_bytes() == 6 * GiB, "headroom default 6 GiB");
        auto nomem = two_nodes(t, 150, 170, 5, 5, 1);
        fs::remove(t.root / "devices/system/node/node0/meminfo");
        nomem = plat::numa_discover(t.root.string());
        plat::numa_set_gpu(nomem, "0000:3B:00.0");
        m = in(); m.topo = &nomem;
        p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("cannot be read") != std::string::npos, "unreadable node memory: refused");
        two_nodes(t, 150, 170, 20, 5, 1);
        m = in();
        m.global_available = 80 * GiB;   // two 50 GiB copies + headroom do not fit the whole system
        p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("MemAvailable") != std::string::npos, "whole-system room (cgroup limit): refused");
        m.global_available = 120 * GiB;
        check(plat::plan_arena_mirror(m).mirror, "...and enough of it: mirrors");
    }
    {   // taskset / numactl --cpunodebind
        auto m = in();
        for (int c = 0; c < 12; ++c) m.allowed_cpus.push_back(c);          // node 0 only; the GPU is on node 1
        auto p = plat::plan_arena_mirror(m);
        check(p.mirror && p.cpu_nodes == std::vector<int>{0} && p.primary_node == 1 && p.replica_nodes == std::vector<int>{0},
              "CPUs on node 0 only: the replica goes where the readers are, the primary stays on the GPU's node");
        m.allowed_cpus.clear();
        for (int c = 12; c < 24; ++c) m.allowed_cpus.push_back(c);         // node 1 = the GPU's node only
        p = plat::plan_arena_mirror(m);
        check(!p.mirror && p.why_not.find("every CPU this process may use is on node 1") != std::string::npos,
              "CPUs on the GPU's node only: nothing to mirror");
        std::printf("  taskset   -> %s\n", p.why_not.c_str());
    }
    {   // the environment override
        plat::NumaMode m = plat::NumaMode::Auto;
        std::string note;
        unsetenv("STRATA_NUMA_MIRROR");
        check(plat::numa_mode_with_env(plat::NumaMode::Auto, &note) == plat::NumaMode::Auto && note.empty(), "env unset: --numa stands");
        setenv("STRATA_NUMA_MIRROR", "0", 1);
        check(plat::numa_mode_with_env(plat::NumaMode::Mirror, &note) == plat::NumaMode::Off && !note.empty(), "STRATA_NUMA_MIRROR=0 -> off");
        setenv("STRATA_NUMA_MIRROR", "1", 1);
        check(plat::numa_mode_with_env(plat::NumaMode::Off, &note) == plat::NumaMode::Mirror, "STRATA_NUMA_MIRROR=1 -> mirror");
        setenv("STRATA_NUMA_MIRROR", "maybe", 1);
        check(plat::numa_mode_with_env(plat::NumaMode::Auto, &note) == plat::NumaMode::Auto && note.find("ignored") != std::string::npos,
              "anything else is ignored, and says so");
        unsetenv("STRATA_NUMA_MIRROR");
        check(plat::parse_numa_mode("auto", m) && m == plat::NumaMode::Auto && plat::parse_numa_mode("mirror", m) &&
                  m == plat::NumaMode::Mirror && plat::parse_numa_mode("OFF", m) && m == plat::NumaMode::Off &&
                  !plat::parse_numa_mode("interleave", m),
              "--numa values");
    }
}

// ---------------------------------------------------------------- (c)
void fill(std::vector<uint8_t>& v, uint32_t seed) {
    uint32_t x = seed;
    for (auto& b : v) { x = x * 1664525u + 1013904223u; b = (uint8_t) (x >> 24); }
}

void test_syscalls() {
    std::printf("[the real syscalls on this machine]\n");
    const auto real = plat::numa_discover("/sys");
    if (!plat::numa_syscalls_available() || !real.available || real.nodes.empty()) {
        std::printf("  mbind / move_pages not available here: SKIPPED\n");
        return;
    }
    const int node0 = real.nodes.front().id;
    const uint64_t n = 8ull << 20;
    {
        plat::NumaBuffer b;
        check(b.allocate(n, node0, /*use_hugepages=*/true), "allocate on a real node");
        std::printf("  buffer on node %d: %s\n", node0, b.note().c_str());
        check(b.bound(), "mbind(MPOL_BIND) to a real node succeeds");
        std::memset(b.data(), 0x5a, n);
        const auto rep = plat::numa_sample_placement(b.data(), n, 32);
        std::printf("  placement: %s\n", rep.text().c_str());
        check(rep.ok && rep.sampled > 0 && rep.on_node(node0) == rep.sampled, "every sampled page is on the bound node");
        // an untouched buffer reports its pages as not present, not as a node
        plat::NumaBuffer c;
        check(c.allocate(n, node0, false), "allocate (4 KiB pages)");
        const auto rep2 = plat::numa_sample_placement(c.data(), n, 16);
        check(rep2.ok && rep2.not_present == rep2.sampled, "untouched pages: not present (a query does not touch)");
        check(!c.hugepages(), "no hugepages requested -> none");
        b.lock();   // may be refused by ulimit -l; either way it is a note
        std::printf("  lock: %s\n", b.lock_note().c_str());
    }
    {   // a bind to a node that does not exist fails and is REPORTED; the buffer is still there
        plat::NumaBuffer b;
        check(b.allocate(n, 63, true), "the mapping succeeds");
        check(!b.bound() && b.note().find("FAILED") != std::string::npos, "...but the bind to a missing node fails, and says so");
        std::printf("  node 63: %s\n", b.note().c_str());
        std::memset(b.data(), 1, n);
    }
    {   // the pinned copy is exact, at an awkward size, on more threads than parts
        std::vector<uint8_t> src((size_t) (5 * (1 << 20) + 12345));
        fill(src, 7);
        plat::NumaBuffer d;
        check(d.allocate(src.size(), node0, false), "allocate the destination");
        const double sec = plat::numa_copy(d.data(), src.data(), src.size(), real.nodes.front().cpus, 6);
        check(sec >= 0 && std::memcmp(d.data(), src.data(), src.size()) == 0, "numa_copy is exact");
        std::vector<uint8_t> dst(src.size());
        plat::numa_copy(dst.data(), src.data(), src.size(), {}, 3);
        check(dst == src, "...with no cpu list (unpinned)");
        plat::numa_copy(dst.data(), src.data(), 0, {}, 3);   // nothing
    }
    std::string det;
    const bool inter = plat::numa_task_policy_interleaved(det);
    std::printf("  this process's memory policy: %s%s\n", det.c_str(), inter ? "  (INTERLEAVE)" : "");
    check(!det.empty(), "get_mempolicy answers");
#if defined(SYS_set_mempolicy)
    {   // what `numactl --interleave=all` does: an interleave TASK policy.  The detector must see it, and a node-bound buffer
        // must still land on its node (a per-mapping policy outranks the task's).
        unsigned long mask[16] = {0};
        mask[(size_t) node0 / 64] |= 1ul << (node0 % 64);
        if (syscall(SYS_set_mempolicy, 3 /*MPOL_INTERLEAVE*/, mask, 1025ul) == 0) {
            std::string d2;
            check(plat::numa_task_policy_interleaved(d2) && d2.find("interleave") != std::string::npos, "an interleave task policy is detected");
            plat::NumaBuffer b;
            b.allocate(n, node0, false);
            std::memset(b.data(), 3, n);
            const auto rep = plat::numa_sample_placement(b.data(), n, 16);
            check(rep.ok && rep.on_node(node0) == rep.sampled, "under that policy a bound buffer still lands on its node");
            syscall(SYS_set_mempolicy, 0 /*MPOL_DEFAULT*/, (void*) nullptr, 0ul);
            std::string d3;
            check(!plat::numa_task_policy_interleaved(d3), "back to the default policy: not interleaved");
        } else {
            std::printf("  set_mempolicy refused here: interleave detection not exercised\n");
        }
    }
#endif
}

// ---------------------------------------------------------------- (d)
void test_replicas() {
    std::printf("[the replica builder, fake two-node topology on a one-node machine]\n");
    if (!plat::numa_syscalls_available()) { std::printf("  no mbind here: SKIPPED\n"); return; }
    // fake: CPUs 0-1 are node 0, 2-3 are node 1, the GPU is on node 0
    TempRoot t("repl");
    node(t.root, 0, "0-1", 64, 50, 4);
    node(t.root, 1, "2-3", 64, 50, 4);
    gpu(t.root, "0000:3b:00.0", 0);
    auto topo = plat::numa_discover(t.root.string());
    plat::numa_set_gpu(topo, "0000:3B:00.0");
    plat::MirrorInputs in;
    in.mode = plat::NumaMode::Mirror;
    in.topo = &topo;
    in.arena_bytes = 6 << 20;
    in.headroom = 1 * GiB;
    const auto plan = plat::plan_arena_mirror(in);
    check(plan.mirror && plan.primary_node == 0 && plan.replica_nodes == std::vector<int>{1}, "plan: primary node 0, replica node 1");

    const uint64_t valid = 5 * (1ull << 20) + 4097, cap = 6ull << 20;   // an arena longer than what was loaded
    std::vector<uint8_t> src((size_t) cap, 0);
    fill(src, 11);
    {   // a FAKE topology: node 1 does not exist, the bind fails, the replica is still a good copy, `copy[]` is by node index
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = true; mo.threads = 4; mo.lock = false; mo.samples = 16;
        std::vector<std::string> log;
        const int kept = rep.build(src.data(), valid, cap, plan, topo, mo, log);
        for (const auto& l : log) std::printf("    %s\n", l.c_str());
        check(kept == 1 && rep.count() == 1, "faked: the replica is kept");
        const auto& m = rep.mirror();
        check(m.active() && m.primary == src.data() && m.bytes == cap && m.primary_index == 0 && m.copy.size() == 2 &&
                  m.copy[0] == src.data() && m.copy[1] != nullptr && m.copy[1] != src.data(),
              "mirror(): primary, one copy per node index, the primary's own slot is the primary");
        check(m.copy[1] != nullptr && std::memcmp(m.copy[1], src.data(), valid) == 0, "the replica holds the primary's bytes");
        bool logged_fail = false, logged_cont = false;
        for (const auto& l : log) {
            logged_fail |= l.find("FAILED") != std::string::npos;
            logged_cont |= l.find("test topology: continuing") != std::string::npos;
        }
        check(logged_fail && logged_cont, "the failed bind and the placement mismatch are reported, and it carries on");
        check(rep.primary_placement().ok, "the primary's placement was sampled");
        rep.release();
        check(!rep.mirror().active() && rep.count() == 0, "release() empties it");
    }
    {   // a REAL topology (faked = false) with the same unplaceable node: the replica is DROPPED
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = false; mo.threads = 2; mo.lock = false;
        std::vector<std::string> log;
        const int kept = rep.build(src.data(), valid, cap, plan, topo, mo, log);
        bool dropped = false;
        for (const auto& l : log) dropped |= l.find("DROPPED") != std::string::npos;
        check(kept == 0 && !rep.mirror().active() && dropped, "real topology: a replica whose pages are not on its node is dropped, and it says so");
    }
    {   // an inactive plan builds nothing
        plat::NumaReplicas rep;
        plat::MirrorPlan off;
        std::vector<std::string> log;
        check(rep.build(src.data(), valid, cap, off, topo, plat::MirrorOptions{}, log) == 0 && !rep.mirror().active(), "no plan, no replica");
    }
}

// ---------------------------------------------------------------- (e) WP-F audit A4
constexpr uint64_t MiB2 = 2ull << 20;

void test_hugepool_and_planner() {
    std::printf("[the hugetlb pool per node: topology and planner]\n");
    TempRoot t("pool");
    // vm.nr_hugepages = 26000 (51 GiB, sized for one 50 GiB arena) is spread EVENLY: 13000 pages = 25.4 GiB on each node
    auto make = [&](double free0, double free1, unsigned long pool0, unsigned long pool1_free, unsigned long pool1_total) {
        fs::remove_all(t.root);
        node(t.root, 0, "0-11,24-35", 192, free0, 5);
        node(t.root, 1, "12-23,36-47", 192, free1, 5);
        gpu(t.root, "0000:3b:00.0", 1);
        if (pool0 != 0) hugepool(t.root, 0, pool0, pool0);
        if (pool1_total != 0) hugepool(t.root, 1, pool1_free, pool1_total);
        thp_mode(t.root, "madvise");
        auto topo = plat::numa_discover(t.root.string());
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        return topo;
    };
    auto topo = make(150, 170, 13000, 13000, 13000);
    check(topo.nodes[0].has_huge && topo.nodes[0].huge_free == 13000 * MiB2 && topo.nodes[0].huge_total == 13000 * MiB2 &&
              topo.nodes[1].has_huge && topo.nodes[1].huge_free == 13000 * MiB2,
          "discover reads each node's free_hugepages / nr_hugepages (bytes)");
    check(topo.thp_mode == "madvise", "...and the THP mode");
    std::printf("  %s\n", plat::numa_describe(topo).c_str());
    check(plat::numa_describe(topo).find("free in its 2 MiB hugepage pool of 25.4 GiB") != std::string::npos, "describe names the pool");
    check(!topo.nodes[0].pool_holds(50 * GiB) && topo.nodes[0].pool_holds(25 * GiB), "a 25.4 GiB pool holds a 25 GiB copy, not a 50 GiB one");
    topo = make(150, 170, 0, 0, 0);
    check(!topo.nodes[0].has_huge && !topo.nodes[0].pool_holds(1), "no pool files: no pool");
    topo = make(150, 170, 0, 0, 0);
    thp_mode(t.root, "never");
    check(plat::numa_discover(t.root.string()).thp_mode == "never", "THP mode never");

    auto in = [&](const plat::NumaTopology& tp) {
        plat::MirrorInputs m;
        m.mode = plat::NumaMode::Auto;
        m.topo = &tp;
        m.arena_bytes = 50 * GiB;
        m.headroom = 6 * GiB;
        return m;
    };
    {   // half a copy per node in the pool: nothing changes for the planner - the copy takes normal memory (THP), and node 1 has it
        topo = make(150, 170, 13000, 13000, 13000);
        const auto p = plat::plan_arena_mirror(in(topo));
        check(p.mirror, "pools of half a copy, plenty of normal memory: mirrors (the copies go to THP)");
    }
    {   // a node short of NORMAL memory whose own pool holds the whole copy: the copy takes pool pages, the headroom is all it needs
        topo = make(150, 8, 13000, 30720, 30720);   // node 1 (the GPU's): 8 + 5 = 13 GiB usable, a 60 GiB pool
        auto p = plat::plan_arena_mirror(in(topo));
        check(p.mirror, "node 1 has 13 GiB usable but its free 60 GiB pool holds the copy: mirrors (before: refused)");
        topo = make(150, 8, 13000, 0, 0);
        p = plat::plan_arena_mirror(in(topo));
        check(!p.mirror && p.why_not.find("node 1 has 13.0 GiB usable") != std::string::npos, "...the same node without the pool: refused");
        topo = make(150, 8, 13000, 13000, 13000);
        p = plat::plan_arena_mirror(in(topo));
        check(!p.mirror && p.why_not.find("not a whole copy") != std::string::npos,
              "...with a pool of half a copy: refused, and the message says the pool is not a whole copy");
        std::printf("  half pool -> %s\n", p.why_not.c_str());
        topo = make(150, 8, 13000, 30720, 30720);
        auto m = in(topo);
        m.global_available = 62 * GiB;   // node 0: a 50 GiB copy from normal memory + 6 headroom; node 1's copy is in its pool
        check(plat::plan_arena_mirror(m).mirror, "pooled copies are not charged to MemAvailable");
        m.global_available = 50 * GiB;
        const auto q = plat::plan_arena_mirror(m);
        check(!q.mirror && q.why_not.find("hugepage pools excluded") != std::string::npos, "...and the refusal says so");
    }
}

void test_sub_numa_clustering() {
    std::printf("[sub-NUMA clustering: 4 nodes on 2 sockets, one socket, 10 nodes]\n");
    TempRoot t("snc");
    auto snc = [&](double f0, double f1, double f2, double f3, int pkg23) {
        fs::remove_all(t.root);
        node(t.root, 0, "0-5", 96, f0, 5);
        node(t.root, 1, "6-11", 96, f1, 5);
        node(t.root, 2, "12-17", 96, f2, 5);
        node(t.root, 3, "18-23", 96, f3, 5);
        packages(t.root, range(0, 12), 0);
        packages(t.root, range(12, 24), pkg23);
        gpu(t.root, "0000:3b:00.0", 1);
        auto topo = plat::numa_discover(t.root.string());
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        return topo;
    };
    auto in = [&](const plat::NumaTopology& tp, plat::NumaMode mode) {
        plat::MirrorInputs m;
        m.mode = mode;
        m.topo = &tp;
        m.arena_bytes = 50 * GiB;
        m.headroom = 6 * GiB;
        return m;
    };
    {
        auto topo = snc(80, 80, 70, 85, 1);
        check(topo.nodes[0].package == 0 && topo.nodes[1].package == 0 && topo.nodes[2].package == 1 && topo.nodes[3].package == 1,
              "the package of each node, from its CPUs' physical_package_id");
        std::printf("  %s\n", plat::numa_describe(topo).c_str());
        const auto p = plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto));
        check(p.mirror && p.primary_node == 1 && p.replica_nodes == std::vector<int>{3},
              "auto: ONE replica for the other socket (not one per node), on its node with the most room (3)");
        const std::vector<std::pair<int, int>> want{{0, 1}, {1, 1}, {2, 3}, {3, 3}};
        check(p.reads == want, "...nodes 0,1 read the primary on node 1, nodes 2,3 read the replica on node 3");
        std::printf("  layout: %s\n", p.layout.c_str());
        check(p.layout.find("primary on node 1 is read by nodes 0,1") != std::string::npos &&
                  p.layout.find("replica on node 3 is read by nodes 2,3") != std::string::npos, "...and the layout line says it");
        const auto pm = plat::plan_arena_mirror(in(topo, plat::NumaMode::Mirror));
        check(pm.mirror && pm.replica_nodes == std::vector<int>{0, 2, 3}, "mirror (asked for): one replica per node, 3");
        // the same four nodes with the other socket's node 2 roomier: the holder follows the room
        topo = snc(80, 80, 90, 60, 1);
        check(plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto)).replica_nodes == std::vector<int>{2}, "the holder is the roomier node (2)");
        // no room on the roomiest node of the other socket: refused, naming that node
        topo = snc(80, 80, 20, 25, 1);
        const auto pr = plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto));
        check(!pr.mirror && pr.why_not.find("node 3 has") != std::string::npos, "no room on the holder: refused, naming it");
    }
    {   // one socket cut into four by SNC: nothing to mirror in auto mode
        auto topo = snc(80, 80, 70, 85, 0);
        const auto p = plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto));
        check(!p.mirror && p.why_not.find("own socket") != std::string::npos && p.why_not.find("--numa mirror") != std::string::npos,
              "all four nodes in the GPU's package: auto refuses, and names --numa mirror");
        std::printf("  one socket -> %s\n", p.why_not.c_str());
        check(plat::plan_arena_mirror(in(topo, plat::NumaMode::Mirror)).replica_nodes == std::vector<int>{0, 2, 3}, "...mirror still makes one per node");
    }
    {   // no cpu topology files: every node is a package of its own (the pre-SNC behaviour)
        auto topo = snc(80, 80, 70, 85, 1);
        fs::remove_all(t.root / "devices/system/cpu");
        topo = plat::numa_discover(t.root.string());
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(topo.nodes[0].package == -1, "no physical_package_id files: package unknown");
        const auto p = plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto));
        check(p.mirror && p.replica_nodes == std::vector<int>{0, 2, 3}, "...so auto makes one replica per node, as before");
    }
    {   // a node whose CPUs span two packages counts as its own
        auto topo = snc(80, 80, 70, 85, 1);
        packages(t.root, {13}, 0);   // a CPU of node 2 claims package 0
        topo = plat::numa_discover(t.root.string());
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        check(topo.nodes[2].package == -2, "CPUs of one node in two packages: -2");
        const auto p = plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto));
        check(p.mirror && p.replica_nodes == std::vector<int>{2, 3}, "...it gets a replica of its own, apart from node 3's");
    }
    if (!plat::numa_syscalls_available()) { std::printf("  no mbind here: the replica builds SKIPPED\n"); return; }

    // the replicas of an SNC plan: ONE buffer, the primary's node and its socket-mate read the primary, the other two read it
    {
        auto topo = snc(80, 80, 70, 85, 1);
        const auto plan = plat::plan_arena_mirror(in(topo, plat::NumaMode::Auto));
        const uint64_t cap = 6ull << 20, valid = 5ull << 20;
        std::vector<uint8_t> src((size_t) cap);
        fill(src, 3);
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = true; mo.threads = 3; mo.lock = false; mo.samples = 8;
        std::vector<std::string> log;
        auto plan2 = plan;
        plan2.copy_bytes = cap;
        const int kept = rep.build(src.data(), valid, cap, plan2, topo, mo, log);
        const auto& m = rep.mirror();
        check(kept == 1 && rep.count() == 1 && m.active() && m.copy.size() == 4, "SNC auto: one replica kept, a slot per node");
        check(m.copy[1] == src.data() && m.copy[0] == src.data(), "nodes 0 and 1 read the primary");
        check(m.copy[3] != nullptr && m.copy[3] != src.data() && m.copy[2] == m.copy[3], "nodes 2 and 3 read the SAME replica");
        check(m.copy[3] != nullptr && std::memcmp(m.copy[3], src.data(), valid) == 0, "...which holds the primary's bytes");
        for (const auto& l : log) std::printf("    %s\n", l.c_str());
    }
    {   // 10 nodes (two sockets of five): nothing is capped at 8
        fs::remove_all(t.root);
        for (int i = 0; i < 10; ++i) {
            node(t.root, i, std::to_string(2 * i) + "-" + std::to_string(2 * i + 1), 96, 80 + i, 5);
            packages(t.root, {2 * i, 2 * i + 1}, i < 5 ? 0 : 1);
        }
        gpu(t.root, "0000:3b:00.0", 7);
        auto topo = plat::numa_discover(t.root.string());
        plat::numa_set_gpu(topo, "0000:3B:00.0");
        auto ma = in(topo, plat::NumaMode::Auto);
        ma.arena_bytes = 6ull << 20;
        ma.headroom = 1 * GiB;
        const auto pa = plat::plan_arena_mirror(ma);
        check(pa.mirror && pa.primary_node == 7 && pa.replica_nodes == std::vector<int>{4}, "10 nodes, auto: one replica (the other socket's roomiest node, 4)");
        check(pa.reads.size() == 10, "...and all ten nodes have a copy to read");
        auto mm = ma;
        mm.mode = plat::NumaMode::Mirror;
        const auto pm = plat::plan_arena_mirror(mm);
        check(pm.mirror && pm.replica_nodes.size() == 9, "10 nodes, mirror: 9 replicas");
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = true; mo.threads = 2; mo.lock = false; mo.samples = 4;
        const uint64_t cap = 6ull << 20;
        std::vector<uint8_t> src((size_t) cap);
        fill(src, 5);
        std::vector<std::string> log;
        const int kept = rep.build(src.data(), cap, cap, pm, topo, mo, log);
        bool all = kept == 9 && rep.mirror().copy.size() == 10;
        for (size_t i = 0; all && i < 10; ++i) all = rep.mirror().copy[i] != nullptr;
        check(all, "10 nodes: 9 replicas built, every node index (9 included) has a copy");
        check(rep.mirror().copy[9] != nullptr && rep.mirror().copy[9] != src.data() && std::memcmp(rep.mirror().copy[9], src.data(), cap) == 0,
              "...node index 9's copy is its own replica of the primary's bytes");
    }
}

void test_page_backing() {
    std::printf("[page backing: hugetlb, THP, 4 KiB, the SIGBUS guard]\n");
    const auto real = plat::numa_discover("/sys");
    if (!plat::numa_syscalls_available() || !real.available || real.nodes.empty()) {
        std::printf("  mbind not available here: SKIPPED\n");
        return;
    }
    const int node0 = real.nodes.front().id;
    const uint64_t n = 8ull << 20;
    const std::string mode = plat::numa_thp_mode();
    std::printf("  system THP mode: %s\n", mode.empty() ? "(unreadable)" : mode.c_str());
    bool known = false;
    const uint64_t real_pool = plat::numa_node_free_hugepage_bytes(node0, known, "/sys");
    const bool pool_ok = known && real_pool >= n;
    std::printf("  this machine's node %d hugepage pool: %s\n", node0, known ? (std::to_string(real_pool >> 20) + " MiB free").c_str() : "none");
    auto touch = [&](void* p, uint64_t bytes) { std::memset(p, 0x6b, bytes); };

    {   // no pool anywhere: THP (when the system allows it), else 4 KiB; 2 MiB-aligned, a whole number of 2 MiB; bound; resident as asked
        plat::ArenaMapOptions o;
        o.node = node0;
        o.hugetlb = false;
        plat::ArenaMap m;
        check(plat::numa_map_arena(n + 12345, o, m), "map an awkward size");
        check(((uintptr_t) m.base & (MiB2 - 1)) == 0 && m.bytes == 5 * MiB2 && m.bound, "2 MiB-aligned, rounded to 2 MiB, bound");
        touch(m.base, n + 12345);
        plat::PageStats ps = plat::numa_page_stats(m.base, n + 12345);
        std::printf("  %s -> %s\n", m.note.c_str(), ps.text().c_str());
        if (mode == "madvise" || mode == "always") {
            check(m.kind == plat::PageKind::Thp, "THP asked for, the system allows it: kind Thp");
            // the kernel may be out of 2 MiB blocks right now (fragmented memory): give it three tries before calling it a failure
            for (int attempt = 1; attempt < 3 && !(ps.ok && ps.thp > 0); ++attempt) {
                plat::numa_unmap_arena(m.base, m.bytes);
                plat::numa_map_arena(n + 12345, o, m);
                touch(m.base, n + 12345);
                ps = plat::numa_page_stats(m.base, n + 12345);
            }
            check(ps.ok && ps.thp > 0, "...and the kernel really gave 2 MiB pages (smaps AnonHugePages > 0)");
        } else if (mode == "never") {
            check(m.kind == plat::PageKind::Small && m.note.find("never") != std::string::npos, "system THP off: 4 KiB pages, and the note says why");
        }
        plat::numa_unmap_arena(m.base, m.bytes);
        o.thp = false;
        check(plat::numa_map_arena(n, o, m) && m.kind == plat::PageKind::Small, "hugetlb and THP both off: 4 KiB pages");
        touch(m.base, n);
        const auto ps2 = plat::numa_page_stats(m.base, n);
        check(!ps2.ok || ps2.thp == 0 || mode == "always", "...and no THP was used (unless the system is `always`)");
        plat::numa_unmap_arena(m.base, m.bytes);
        setenv("STRATA_NO_THP", "1", 1);
        o.thp = true;
        check(plat::numa_map_arena(n, o, m) && m.kind == plat::PageKind::Small && m.note.find("STRATA_NO_THP") != std::string::npos,
              "STRATA_NO_THP=1: no THP request (the A/B arm)");
        plat::numa_unmap_arena(m.base, m.bytes);
        unsetenv("STRATA_NO_THP");
    }
    {   // the pool of the node is too small / absent: the decision comes from the NODE's files, so a fake root decides it
        TempRoot r("pb");
        node(r.root, node0, "0-3", 16, 8);
        hugepool(r.root, node0, 2, 2);   // 4 MiB free, 8 MiB wanted
        thp_mode(r.root, mode.empty() ? "madvise" : mode);
        plat::ArenaMapOptions o;
        o.node = node0;
        o.sysfs_root = r.root.string();
        plat::ArenaMap m;
        check(plat::numa_map_arena(n, o, m) && m.kind != plat::PageKind::Hugetlb &&
                  m.note.find("pool has 4.0 MiB free, 8.0 MiB needed") != std::string::npos && m.note.find("size every node's pool") != std::string::npos,
              "a pool of 4 MiB for an 8 MiB copy: not used, the note gives both numbers and the sizing advice");
        std::printf("  %s\n", m.note.c_str());
        plat::numa_unmap_arena(m.base, m.bytes);
        std::error_code ec;
        fs::remove_all(r.root / "devices/system/node" / ("node" + std::to_string(node0)) / "hugepages", ec);
        check(plat::numa_map_arena(n, o, m) && m.kind != plat::PageKind::Hugetlb && m.note.find("no 2 MiB hugepage pool on node") != std::string::npos,
              "no pool files: no hugetlb attempt, and it says so");
        plat::numa_unmap_arena(m.base, m.bytes);
    }
    {   // the node's files (fake root) claim plenty but the kernel's reservation is what decides: mmap refused -> THP, with the reason
        TempRoot r("pc");
        node(r.root, node0, "0-3", 16, 8);
        hugepool(r.root, node0, 100000, 100000);
        thp_mode(r.root, mode.empty() ? "madvise" : mode);
        plat::ArenaMapOptions o;
        o.node = node0;
        o.sysfs_root = r.root.string();
        plat::ArenaMap m;
        check(plat::numa_map_arena(n, o, m), "map against a fake pool that is bigger than the real one");
        if (!pool_ok) {
            check(m.kind != plat::PageKind::Hugetlb && m.note.find("MAP_HUGETLB refused") != std::string::npos,
                  "the node's files say the pool is there, the kernel's mmap says no: falls back, the note has the errno");
            std::printf("  %s\n", m.note.c_str());
        }
        plat::numa_unmap_arena(m.base, m.bytes);
    }
    if (!pool_ok) {
        std::printf("  no free hugepage pool on this machine: the hugetlb mapping, its prefault and the SIGBUS guard are SKIPPED\n"
                    "  (as root: echo 64 > /proc/sys/vm/nr_hugepages, then run again)\n");
    } else {
        plat::ArenaMapOptions o;
        o.node = node0;
        o.cpus = real.nodes.front().cpus;
        o.threads = 3;
        plat::ArenaMap m;
        check(plat::numa_map_arena(n, o, m) && m.kind == plat::PageKind::Hugetlb && m.bound, "a real pool that holds it: hugetlb, bound");
        check(m.note.find("prefaulted") != std::string::npos, "...prefaulted (MADV_POPULATE_WRITE) before anything touched it");
        std::printf("  %s; %s\n", m.note.c_str(), m.bind_note.c_str());
        const auto rep = plat::numa_sample_placement(m.base, n, 4, MiB2);
        check(rep.ok && rep.on_node(node0) == rep.sampled && rep.not_present == 0, "all its pages are resident on the bound node already");
        touch(m.base, n);
        const auto ps = plat::numa_page_stats(m.base, n);
        std::printf("  %s\n", ps.text().c_str());
        check(ps.ok && ps.hugetlb >= n / 2, "smaps agrees: hugetlb pages");
        plat::numa_unmap_arena(m.base, m.bytes);
        {   // an UNBOUND hugetlb mapping is prefaulted as well (numactl --membind / --interleave restrict a hugetlb fault the same way)
            plat::ArenaMapOptions u;
            u.node = -1;
            plat::ArenaMap um;
            check(plat::numa_map_arena(n, u, um) && um.kind == plat::PageKind::Hugetlb && um.note.find("prefaulted") != std::string::npos,
                  "an unbound hugetlb mapping is prefaulted too");
            plat::numa_unmap_arena(um.base, um.bytes);
        }
        // the guard: a prefault that fails (injected: a pool that was promised elsewhere cannot be had at this point) gives the
        // hugetlb mapping back and remaps the copy as THP / 4 KiB, still bound, instead of leaving a SIGBUS for the first touch
        o.fail_prefault_for_test = true;
        const uint64_t free_before = plat::numa_node_free_hugepage_bytes(node0, known, "/sys");
        check(plat::numa_map_arena(n, o, m) && m.kind != plat::PageKind::Hugetlb && m.bound && m.note.find("prefault") != std::string::npos,
              "a failed prefault: the copy is remapped without hugetlb, still bound, and the note says why");
        std::printf("  %s\n", m.note.c_str());
        touch(m.base, n);   // no SIGBUS
        plat::numa_unmap_arena(m.base, m.bytes);
        check(plat::numa_node_free_hugepage_bytes(node0, known, "/sys") == free_before, "...and the pool pages it took went back");
    }
    {   // numa_populate_write reports what it cannot do: a read-only mapping cannot be populated for write
        void* p = mmap(nullptr, 4 * MiB2, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        std::string err;
        const auto r = plat::numa_populate_write(p, 4 * MiB2, 3, {}, err);
        check(r == plat::Populate::Failed || r == plat::Populate::Unsupported, "populate of a read-only mapping: an error, not a crash");
        std::printf("  populate(read-only): %s\n", err.c_str());
        munmap(p, 4 * MiB2);
        void* q = mmap(nullptr, 8 * MiB2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        const auto r2 = plat::numa_populate_write(q, 8 * MiB2, 3, {}, err);
        check(r2 == plat::Populate::Ok || r2 == plat::Populate::Unsupported, "populate of a normal mapping works (or the kernel is older than 5.14)");
        if (r2 == plat::Populate::Ok) {
            const auto rep = plat::numa_sample_placement(q, 8 * MiB2, 4);
            check(rep.ok && rep.not_present == 0, "...every page is resident afterwards");
        }
        munmap(q, 8 * MiB2);
    }
}

void test_pin_callbacks() {
    std::printf("[pinning a replica through a callback]\n");
    const auto real = plat::numa_discover("/sys");
    if (!plat::numa_syscalls_available() || !real.available || real.nodes.empty()) { std::printf("  no mbind / no NUMA topology here: SKIPPED\n"); return; }
    const int real_node0 = real.nodes.front().id;
    const uint64_t cap = 6ull << 20;
    int pins = 0, unpins = 0;
    void* pinned_p = nullptr;
    plat::PinFn ok = [&](void* p, uint64_t bytes, std::string& how) { ++pins; pinned_p = p; how = "fake register of " + std::to_string(bytes >> 20) + " MiB"; return true; };
    plat::PinFn no = [&](void*, uint64_t, std::string& how) { ++pins; how = "register refused (fake)"; return false; };
    plat::UnpinFn un = [&](void* p, uint64_t) { ++unpins; check(p == pinned_p, "unpin gets the pointer that was pinned"); };
    {
        plat::NumaBuffer b;
        check(b.allocate(cap, real_node0, false), "allocate");
        check(b.pin(ok, un) && b.pinned() && b.pin_note().find("fake register of 6 MiB") != std::string::npos, "pin: pinned, with the callback's note");
        check(unpins == 0, "still pinned");
    }
    check(unpins == 1, "destroying the buffer unpins it BEFORE the memory goes");
    {
        plat::NumaBuffer b;
        b.allocate(cap, real_node0, false);
        check(!b.pin(no, un) && !b.pinned() && b.pin_note() == "register refused (fake)", "a refusal is not pinned, with the reason");
        check(!b.pin(plat::PinFn{}, un), "no callback: nothing to do");
    }
    check(unpins == 1, "...and a buffer that was not pinned is not unpinned");

    // the replica builder: pin first, mlock only where the pin fails
    TempRoot t("pinrep");
    node(t.root, 0, "0-1", 64, 50, 4);
    node(t.root, 1, "2-3", 64, 50, 4);
    gpu(t.root, "0000:3b:00.0", 0);
    auto topo = plat::numa_discover(t.root.string());
    plat::numa_set_gpu(topo, "0000:3B:00.0");
    plat::MirrorInputs in;
    in.mode = plat::NumaMode::Mirror;
    in.topo = &topo;
    in.arena_bytes = cap;
    in.headroom = 1 * GiB;
    const auto plan = plat::plan_arena_mirror(in);
    std::vector<uint8_t> src((size_t) cap);
    fill(src, 9);
    pins = unpins = 0;
    {
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = true; mo.threads = 2; mo.lock = true; mo.samples = 8;
        mo.pin = ok; mo.unpin = un;
        std::vector<std::string> log;
        check(rep.build(src.data(), cap, cap, plan, topo, mo, log) == 1 && pins == 1, "build pins the replica once");
        bool said = false, mlocked = false;
        for (const auto& l : log) { said |= l.find("pinned: fake register") != std::string::npos; mlocked |= l.find("mlock") != std::string::npos; }
        check(said && !mlocked, "...the log says pinned, and nothing was mlock-ed");
        check(rep.buffer(0).pinned() && rep.buffer(0).locked_bytes() == 0, "the buffer is pinned, not locked");
        rep.release();
        check(unpins == 1, "release() of the replicas unpins it");
    }
    pins = unpins = 0;
    {
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = true; mo.threads = 2; mo.lock = true; mo.samples = 8;
        mo.pin = no; mo.unpin = un;
        std::vector<std::string> log;
        rep.build(src.data(), cap, cap, plan, topo, mo, log);
        bool refused = false, mlock_tried = false;
        for (const auto& l : log) { refused |= l.find("not pinned: register refused (fake)") != std::string::npos; mlock_tried |= l.find("mlock") != std::string::npos; }
        check(pins == 1 && refused && mlock_tried, "a refused pin falls back to mlock (which may itself be refused by ulimit -l: a note)");
        for (const auto& l : log) std::printf("    %s\n", l.c_str());
    }
    pins = unpins = 0;
    {
        plat::NumaReplicas rep;
        plat::MirrorOptions mo;
        mo.faked = true; mo.threads = 2; mo.lock = false; mo.samples = 8;
        mo.pin = no; mo.unpin = un;
        std::vector<std::string> log;
        rep.build(src.data(), cap, cap, plan, topo, mo, log);
        bool unlocked = false;
        for (const auto& l : log) unlocked |= l.find("not locked") != std::string::npos;
        check(unlocked, "STRATA_ARENA_LOCK=0 (lock = false): a refused pin is not followed by an mlock");
    }
}

}  // namespace

int main() {
    unsetenv("STRATA_SYSFS_ROOT");
    unsetenv("STRATA_NUMA_MIRROR");
    unsetenv("STRATA_NUMA_HEADROOM_GIB");
    unsetenv("STRATA_NUMA_GPU_NODE");
    test_cpulist();
    test_topology();
    test_planner();
    test_syscalls();
    test_replicas();
    test_hugepool_and_planner();
    test_sub_numa_clustering();
    test_page_backing();
    test_pin_callbacks();
    std::printf("\nnuma: %d failures of %d checks\n", g_fail, g_checks);
    if (g_fail == 0) std::printf("numa_test OK\n");
    return g_fail == 0 ? 0 : 1;
}
