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
// The pool's half (a worker reads the copy of its node) is in pool_test.cpp (--numa-mirror).
#include "strata/platform/numa.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

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
    std::printf("\nnuma: %d failures of %d checks\n", g_fail, g_checks);
    if (g_fail == 0) std::printf("numa_test OK\n");
    return g_fail == 0 ? 0 : 1;
}
