// src/platform/numa.cpp - see include/strata/platform/numa.hpp.
#include "strata/platform/numa.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE 23   // Linux 5.14; older libc headers lack the name, older kernels answer EINVAL
#endif
#endif

namespace strata::platform {

namespace {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kHuge = 2ull << 20;

// "31.6 GiB", or "6.0 MiB" for what is under a GiB (the tests' buffers, a dry hugepage pool)
std::string gib(uint64_t bytes) {
    char b[48];
    if (bytes < kGiB) std::snprintf(b, sizeof b, "%.1f MiB", (double) bytes / (double) (1 << 20));
    else std::snprintf(b, sizeof b, "%.1f GiB", (double) bytes / (double) kGiB);
    return b;
}

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// The whole file as text with trailing white space cut; false when it cannot be opened.
bool slurp(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '\t')) out.pop_back();
    return true;
}

std::string resolve_root(const std::string& arg, bool& faked) {
    faked = false;
    if (!arg.empty()) { faked = true; return arg; }
    if (const char* e = std::getenv("STRATA_SYSFS_ROOT"); e != nullptr && *e != '\0') { faked = true; return e; }
    return "/sys";
}

// "always [madvise] never" -> "madvise"; empty when unreadable
std::string thp_mode_of(const std::string& root) {
    std::string text;
    if (!slurp(root + "/kernel/mm/transparent_hugepage/enabled", text)) return {};
    const size_t a = text.find('['), b = text.find(']');
    if (a == std::string::npos || b == std::string::npos || b < a + 2) return {};
    return text.substr(a + 1, b - a - 1);
}

}  // namespace

// ================================ TOPOLOGY ================================

int NumaTopology::index_of_node(int id) const {
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].id == id) return (int) i;
    return -1;
}

int NumaTopology::index_of_cpu(int cpu) const {
    for (size_t i = 0; i < nodes.size(); ++i)
        if (std::binary_search(nodes[i].cpus.begin(), nodes[i].cpus.end(), cpu)) return (int) i;
    return -1;
}

std::vector<int> parse_cpulist(const std::string& text, bool* ok) {
    std::vector<int> out;
    bool good = true;
    size_t i = 0;
    const size_t n = text.size();
    auto skip_ws = [&] { while (i < n && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i; };
    auto number = [&](int& v) -> bool {
        if (i >= n || text[i] < '0' || text[i] > '9') return false;
        long long x = 0;
        while (i < n && text[i] >= '0' && text[i] <= '9') {
            x = x * 10 + (text[i] - '0');
            if (x > 1000000) return false;   // no machine has a million CPUs: a corrupt file, not a list
            ++i;
        }
        v = (int) x;
        return true;
    };
    skip_ws();
    while (i < n && good) {
        int a = 0, b = 0;
        if (!number(a)) { good = false; break; }
        b = a;
        if (i < n && text[i] == '-') {
            ++i;
            if (!number(b) || b < a) { good = false; break; }
        }
        for (int c = a; c <= b; ++c) out.push_back(c);
        skip_ws();
        if (i < n && text[i] == ',') { ++i; skip_ws(); if (i >= n) good = false; }
        else if (i < n) good = false;
    }
    if (!good) out.clear();
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (ok != nullptr) *ok = good;
    return out;
}

std::string format_cpulist(const std::vector<int>& cpus) {
    std::string s;
    for (size_t i = 0; i < cpus.size();) {
        size_t j = i;
        while (j + 1 < cpus.size() && cpus[j + 1] == cpus[j] + 1) ++j;
        if (!s.empty()) s += ',';
        s += std::to_string(cpus[i]);
        if (j > i) s += "-" + std::to_string(cpus[j]);
        i = j + 1;
    }
    return s;
}

NumaTopology numa_discover(const std::string& sysfs_root) {
    NumaTopology t;
#if defined(__linux__)
    t.root = resolve_root(sysfs_root, t.faked);
    namespace fs = std::filesystem;
    const std::string dir = t.root + "/devices/system/node";
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        t.why = "no " + dir + " (this kernel exposes no NUMA topology)";
        return t;
    }
    std::vector<int> ids;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.size() < 5 || name.compare(0, 4, "node") != 0) continue;
        bool digits = true;
        for (size_t k = 4; k < name.size(); ++k) digits = digits && name[k] >= '0' && name[k] <= '9';
        if (digits && name.size() <= 9) ids.push_back(std::atoi(name.c_str() + 4));
    }
    std::sort(ids.begin(), ids.end());
    if (ids.empty()) {
        t.why = "no node<N> directory under " + dir;
        return t;
    }
    for (int id : ids) {
        NumaNode n;
        n.id = id;
        const std::string base = dir + "/node" + std::to_string(id);
        std::string text;
        if (!slurp(base + "/cpulist", text)) {
            t.nodes.clear();
            t.why = "cannot read " + base + "/cpulist";
            return t;
        }
        bool ok = true;
        n.cpus = parse_cpulist(text, &ok);
        if (!ok) {
            t.nodes.clear();
            t.why = base + "/cpulist is not a cpu list: \"" + text + "\"";
            return t;
        }
        // meminfo lines look like "Node 0 MemFree:       4235608 kB"
        std::ifstream mi(base + "/meminfo");
        std::string line;
        uint64_t active_file = 0, inactive_file = 0;
        bool saw_total = false, saw_free = false;
        while (mi && std::getline(mi, line)) {
            char key[64] = {0};
            int nid = 0;
            unsigned long long kb = 0;
            if (std::sscanf(line.c_str(), "Node %d %63[^:]: %llu", &nid, key, &kb) != 3) continue;
            const std::string k = key;
            if (k == "MemTotal") { n.mem_total = kb << 10; saw_total = true; }
            else if (k == "MemFree") { n.mem_free = kb << 10; saw_free = true; }
            else if (k == "Active(file)") active_file = kb << 10;
            else if (k == "Inactive(file)") inactive_file = kb << 10;
        }
        n.mem_file = active_file + inactive_file;
        n.has_meminfo = saw_total && saw_free;
        // the node's own 2 MiB hugetlb pool: pages `vm.nr_hugepages` reserved are not in MemFree, and a bound MAP_HUGETLB
        // mapping draws on THIS node's pool only
        {
            const std::string hp = base + "/hugepages/hugepages-2048kB/";
            std::string f, nr;
            if (slurp(hp + "free_hugepages", f) && slurp(hp + "nr_hugepages", nr)) {
                char *e1 = nullptr, *e2 = nullptr;
                const unsigned long long vf = std::strtoull(f.c_str(), &e1, 10), vn = std::strtoull(nr.c_str(), &e2, 10);
                if (e1 != f.c_str() && e2 != nr.c_str()) {
                    n.has_huge = true;
                    n.huge_free = (uint64_t) vf * kHuge;
                    n.huge_total = (uint64_t) vn * kHuge;
                }
            }
        }
        // the package (socket) of the node's CPUs: sub-NUMA clustering makes several nodes of one
        for (int c : n.cpus) {
            std::string pk;
            if (!slurp(t.root + "/devices/system/cpu/cpu" + std::to_string(c) + "/topology/physical_package_id", pk)) continue;
            char* end = nullptr;
            const long v = std::strtol(pk.c_str(), &end, 10);
            if (end == pk.c_str() || v < 0) continue;
            if (n.package == -1) n.package = (int) v;
            else if (n.package != (int) v) { n.package = -2; break; }
        }
        t.nodes.push_back(std::move(n));
    }
    t.available = true;
    t.gpu_node = t.nodes.front().id;
    t.thp_mode = thp_mode_of(t.root);
#else
    (void) sysfs_root;
    t.why = "not Linux: the topology comes from sysfs";
#endif
    return t;
}

void numa_set_gpu(NumaTopology& t, const std::string& pci_bus_id) {
    t.gpu_node_known = false;
    t.gpu_bdf.clear();
    t.gpu_note.clear();
    if (!t.available || t.nodes.empty()) return;
    t.gpu_node = t.nodes.front().id;
    std::string bdf = pci_bus_id;
    for (char& c : bdf) c = (char) std::tolower((unsigned char) c);
    t.gpu_bdf = bdf;
    // STRATA_NUMA_GPU_NODE=N: the user knows better (`nvidia-smi topo -m` says which socket the card hangs off) than a BIOS
    // that reports -1.  A node that does not exist is ignored, and said so.
    if (const char* e = std::getenv("STRATA_NUMA_GPU_NODE"); e != nullptr && *e != '\0') {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && v >= 0 && t.index_of_node((int) v) >= 0) {
            t.gpu_node = (int) v;
            t.gpu_node_known = true;
            t.gpu_note = "STRATA_NUMA_GPU_NODE=" + std::string(e);
            return;
        }
        t.gpu_note = "STRATA_NUMA_GPU_NODE=" + std::string(e) + " is not a node of this machine (ignored); ";
    }
    if (bdf.empty()) { t.gpu_note += "the GPU's PCI address is unknown: assuming node " + std::to_string(t.gpu_node); return; }
    std::string text;
    const std::string path = t.root + "/bus/pci/devices/" + bdf + "/numa_node";
    if (!slurp(path, text)) {
        t.gpu_note += "cannot read " + path + ": assuming the GPU is on node " + std::to_string(t.gpu_node);
        return;
    }
    char* end = nullptr;
    const long v = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str()) {
        t.gpu_note += path + " says \"" + text + "\": assuming the GPU is on node " + std::to_string(t.gpu_node);
        return;
    }
    if (v < 0) {   // -1: the platform does not say which socket the slot belongs to
        t.gpu_note += "sysfs numa_node is -1 for " + bdf + " (unknown): assuming node " + std::to_string(t.gpu_node);
        return;
    }
    if (t.index_of_node((int) v) < 0) {
        t.gpu_note += "sysfs puts " + bdf + " on node " + std::to_string(v) + ", which has no node directory: assuming node " +
                     std::to_string(t.gpu_node);
        return;
    }
    t.gpu_node = (int) v;
    t.gpu_node_known = true;
    t.gpu_note = "sysfs numa_node of " + bdf;   // (a known node: any earlier note was about an override that did not apply)
}

std::string numa_describe(const NumaTopology& t) {
    if (!t.available) return "NUMA topology not available: " + t.why;
    std::string s = std::to_string(t.nodes.size()) + " NUMA node" + (t.nodes.size() == 1 ? "" : "s") + ": ";
    for (size_t i = 0; i < t.nodes.size(); ++i) {
        const NumaNode& n = t.nodes[i];
        if (i) s += ", ";
        s += "node" + std::to_string(n.id) + " cpus " + (n.cpus.empty() ? std::string("none") : format_cpulist(n.cpus));
        if (n.package >= 0) s += " socket " + std::to_string(n.package);
        if (n.has_meminfo) {
            s += " (" + gib(n.usable_bytes()) + " usable";
            if (n.has_huge && n.huge_total > 0) s += ", " + gib(n.huge_free) + " free in its 2 MiB hugepage pool of " + gib(n.huge_total);
            s += ")";
        } else if (n.has_huge && n.huge_total > 0) {
            s += " (" + gib(n.huge_free) + " free in its 2 MiB hugepage pool)";
        }
    }
    if (!t.gpu_bdf.empty())
        s += std::string("; GPU ") + t.gpu_bdf + " on node " + std::to_string(t.gpu_node) +
             (t.gpu_node_known ? "" : " (ASSUMED: " + t.gpu_note + ")");
    if (t.faked) s += " [fake sysfs: " + t.root + "]";
    return s;
}

std::vector<int> numa_allowed_cpus() {
    std::vector<int> out;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0)
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) out.push_back(i);
#endif
    return out;
}

// ================================ MEMORY POLICY ================================

#if defined(__linux__) && defined(SYS_mbind) && defined(SYS_move_pages) && defined(SYS_get_mempolicy)
namespace {
constexpr int kMpolDefault = 0, kMpolBind = 2, kMpolInterleave = 3, kMpolWeighted = 6;
constexpr unsigned long kFNode = 1, kFAddr = 2;
constexpr int kMaskBits = 1024;   // one 1024-bit node mask; the syscalls take `maxnode` bits + 1 (a historical quirk)
}  // namespace
bool numa_syscalls_available() { return true; }

bool numa_bind_memory(void* p, uint64_t bytes, int node, std::string& err, uint64_t page) {
    if (node < 0 || node >= kMaskBits) { err = "node " + std::to_string(node) + " out of range"; return false; }
    if (p == nullptr || bytes == 0) { err = "nothing to bind"; return false; }
    unsigned long mask[kMaskBits / (8 * sizeof(unsigned long))] = {0};
    mask[(size_t) node / (8 * sizeof(unsigned long))] |= 1ul << ((size_t) node % (8 * sizeof(unsigned long)));
    const uint64_t len = round_up(bytes, page == 0 ? 4096 : page);
    const long r = syscall(SYS_mbind, p, (unsigned long) len, kMpolBind, mask, (unsigned long) kMaskBits + 1, 0u);
    if (r != 0) { err = std::strerror(errno); return false; }
    return true;
}

PlacementReport numa_sample_placement(const void* p, uint64_t bytes, int samples, uint64_t page) {
    PlacementReport rep;
    if (p == nullptr || bytes == 0 || samples < 1) { rep.note = "nothing to sample"; return rep; }
    if (page == 0) page = 4096;
    std::vector<void*> pages;
    const uint64_t n = (uint64_t) samples;
    for (uint64_t i = 0; i < n; ++i) {
        uint64_t off = n == 1 ? 0 : (bytes - 1) / (n - 1) * i;
        if (i + 1 == n && n > 1) off = bytes - 1;
        off = off / page * page;
        void* a = (void*) ((uintptr_t) p + off);
        if (pages.empty() || pages.back() != a) pages.push_back(a);
    }
    std::vector<int> status(pages.size(), -1);
    for (size_t at = 0; at < pages.size(); at += 256) {
        const size_t cnt = std::min<size_t>(256, pages.size() - at);
        const long r = syscall(SYS_move_pages, 0, (unsigned long) cnt, pages.data() + at, (const int*) nullptr,
                               status.data() + at, 0);
        if (r != 0) { rep.note = std::string("move_pages: ") + std::strerror(errno); return rep; }
    }
    rep.ok = true;
    rep.sampled = pages.size();
    for (int s : status) {
        if (s < 0) { ++rep.not_present; continue; }
        bool found = false;
        for (auto& e : rep.by_node)
            if (e.first == s) { ++e.second; found = true; break; }
        if (!found) rep.by_node.emplace_back(s, 1);
    }
    std::sort(rep.by_node.begin(), rep.by_node.end());
    return rep;
}

bool numa_task_policy_interleaved(std::string& detail) {
    int mode = -1;
    unsigned long mask[kMaskBits / (8 * sizeof(unsigned long))] = {0};
    if (syscall(SYS_get_mempolicy, &mode, mask, (unsigned long) kMaskBits + 1, (void*) nullptr, 0ul) != 0) {
        detail = std::string("get_mempolicy: ") + std::strerror(errno);
        return false;
    }
    const int m = mode & 0xff;   // the high bits are mode flags (MPOL_F_STATIC_NODES ...)
    std::vector<int> nodes;
    for (int b = 0; b < kMaskBits; ++b)
        if (mask[(size_t) b / (8 * sizeof(unsigned long))] >> ((size_t) b % (8 * sizeof(unsigned long))) & 1ul) nodes.push_back(b);
    detail = m == kMpolDefault ? "default (local)" : m == kMpolBind ? "bind" : m == kMpolInterleave ? "interleave" :
             m == kMpolWeighted ? "weighted interleave" : "policy " + std::to_string(m);
    if (!nodes.empty()) detail += " over node(s) " + format_cpulist(nodes);
    (void) kFNode; (void) kFAddr;
    return m == kMpolInterleave || m == kMpolWeighted;
}
#else
bool numa_syscalls_available() { return false; }
bool numa_bind_memory(void*, uint64_t, int, std::string& err, uint64_t) { err = "no mbind on this OS"; return false; }
PlacementReport numa_sample_placement(const void*, uint64_t, int, uint64_t) {
    PlacementReport r;
    r.note = "no move_pages on this OS";
    return r;
}
bool numa_task_policy_interleaved(std::string& detail) { detail = "not available on this OS"; return false; }
#endif

uint64_t PlacementReport::on_node(int id) const {
    for (const auto& e : by_node)
        if (e.first == id) return e.second;
    return 0;
}

std::string PlacementReport::text() const {
    if (!ok) return "placement not checked (" + note + ")";
    std::string s;
    if (by_node.size() == 1 && not_present == 0) {
        s = std::to_string(by_node[0].second) + " of " + std::to_string(sampled) + " sampled pages on node " +
            std::to_string(by_node[0].first);
        return s;
    }
    for (const auto& e : by_node) {
        if (!s.empty()) s += ", ";
        s += std::to_string(e.second) + " on node " + std::to_string(e.first);
    }
    if (not_present) s += std::string(s.empty() ? "" : ", ") + std::to_string(not_present) + " not present";
    return s + " (of " + std::to_string(sampled) + " sampled)";
}

uint64_t numa_node_free_hugepage_bytes(int node, bool& known, const std::string& sysfs_root) {
    known = false;
#if defined(__linux__)
    bool faked = false;
    const std::string root = resolve_root(sysfs_root, faked);
    std::string text;
    if (!slurp(root + "/devices/system/node/node" + std::to_string(node) + "/hugepages/hugepages-2048kB/free_hugepages",
               text))
        return 0;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str()) return 0;
    known = true;
    return (uint64_t) v * kHuge;
#else
    (void) node; (void) sysfs_root;
    return 0;
#endif
}

// ================================ PAGE BACKING ================================

const char* page_kind_name(PageKind k) {
    return k == PageKind::Hugetlb ? "2 MiB hugetlb pages" : k == PageKind::Thp ? "THP (2 MiB pages, madvise)" : "4 KiB pages";
}

std::string numa_thp_mode(const std::string& sysfs_root) {
    bool faked = false;
    return thp_mode_of(resolve_root(sysfs_root, faked));
}

#if defined(__linux__)
namespace {
// Whether the kernel knows MADV_POPULATE_WRITE (>= 5.14): probed once on a scratch page, EINVAL = no.
bool populate_supported() {
    static const bool ok = [] {
        void* p = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) return false;
        const int r = madvise(p, 4096, MADV_POPULATE_WRITE);
        const int e = errno;
        munmap(p, 4096);
        return r == 0 || e != EINVAL;
    }();
    return ok;
}
}  // namespace

Populate numa_populate_write(void* p, uint64_t bytes, int threads, const std::vector<int>& cpus, std::string& err) {
    if (p == nullptr || bytes == 0) return Populate::Ok;
    if (!populate_supported()) {
        err = "madvise(MADV_POPULATE_WRITE) needs Linux >= 5.14";
        return Populate::Unsupported;
    }
    const uint64_t chunk = 64ull << 20;   // a multiple of 2 MiB: a hugetlb page is populated by exactly one thread
    const uint64_t nchunks = (bytes + chunk - 1) / chunk;
    const int n = (int) std::max<uint64_t>(1, std::min<uint64_t>((uint64_t) std::max(1, threads), nchunks));
    std::atomic<uint64_t> next{0};
    std::atomic<int> bad{0};
    auto run = [&](int t) {
        if (!cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[(size_t) t % cpus.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);   // best effort: the page is zeroed by a CPU of its node
        }
        for (;;) {
            if (bad.load(std::memory_order_relaxed) != 0) return;
            const uint64_t c = next.fetch_add(1);
            if (c >= nchunks) return;
            const uint64_t a = c * chunk, len = std::min(chunk, bytes - a);
            while (madvise((uint8_t*) p + a, (size_t) len, MADV_POPULATE_WRITE) != 0) {
                if (errno == EINTR) continue;
                bad.store(errno);
                return;
            }
        }
    };
    std::vector<std::thread> th;
    th.reserve((size_t) n);
    for (int t = 0; t < n; ++t) {
        try { th.emplace_back(run, t); }
        catch (...) { run(t); }   // no thread to be had: do its share here
    }
    for (auto& x : th) x.join();
    if (const int e = bad.load(); e != 0) {
        err = std::string(std::strerror(e)) +
              (e == EFAULT ? " (a page fault there would have been a SIGBUS: the node's hugepage pool was promised elsewhere)" : "");
        return Populate::Failed;
    }
    return Populate::Ok;
}

bool numa_map_arena(uint64_t requested, const ArenaMapOptions& opt, ArenaMap& out) {
    out = ArenaMap{};
    if (requested == 0) { out.note = "zero bytes"; return false; }
    bool faked = false;
    const std::string root = resolve_root(opt.sysfs_root, faked);
    const std::string nd = std::to_string(opt.node);
    const uint64_t len = round_up(requested, kHuge);
    std::string why;   // what kept hugetlb from being the answer
    auto bind = [&](void* p, uint64_t page) {
        if (opt.node < 0) return;
        std::string err;
        out.bound = numa_bind_memory(p, len, opt.node, err, page);
        out.bind_note = out.bound ? "mbind(MPOL_BIND) to node " + nd : "mbind to node " + nd + " FAILED (" + err + ")";
    };

    // ---- 1. hugetlb.  A mapping bound to a node draws its pages from THAT node's pool, and a pool that runs dry is a SIGBUS
    // at the first touch, not a fallback: so the pool is asked first (an unreadable one counts as none), and a mapping that
    // got through is prefaulted at once, which turns that SIGBUS into an error here.  (`free_hugepages` also counts pages another
    // reservation has been promised, and the mmap's own reservation is global, not per node: neither is a guarantee for THIS node.)
    if (opt.hugetlb) {
        bool try_huge = true, known = false;
        uint64_t room = 0;
        if (opt.node >= 0) {
            room = numa_node_free_hugepage_bytes(opt.node, known, root);
            if (!known) { try_huge = false; why = "no 2 MiB hugepage pool on node " + nd; }
            else if (room < len) {
                try_huge = false;
                why = room == 0 ? "node " + nd + "'s 2 MiB hugepage pool is empty"
                                : "node " + nd + "'s 2 MiB hugepage pool has " + gib(room) + " free, " + gib(len) +
                                      " needed (Linux spreads vm.nr_hugepages evenly over the nodes: size every node's pool for a whole copy)";
            }
        }
        if (try_huge) {
            void* p = mmap(nullptr, (size_t) len, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
            if (p == MAP_FAILED) {
                why = std::string("MAP_HUGETLB refused (") + std::strerror(errno) + (opt.node < 0 ? "; no hugetlb pool configured?)" : ")");
            } else {
                out.base = p;
                out.bytes = len;
                out.kind = PageKind::Hugetlb;
                bind(p, kHuge);
                std::string extra;
                bool keep = true;
                // (an unbound mapping too: the task policy `numactl --membind` / `--interleave` restricts a hugetlb fault to the nodes it
                // names just as a bind does, and their pools are what runs dry)
                if (opt.prefault_hugetlb && (opt.node < 0 || out.bound)) {
                    std::string perr;
                    Populate r = numa_populate_write(p, len, opt.threads, opt.cpus, perr);
                    if (opt.fail_prefault_for_test) { r = Populate::Failed; perr = "injected by the test"; }
                    if (r == Populate::Failed) {
                        keep = false;
                        why = "the prefault of the hugetlb mapping failed (" + perr + ")";
                    } else if (r == Populate::Unsupported) {
                        extra = "; not prefaulted (" + perr + "): a pool that runs dry would SIGBUS at the first touch";
                    } else {
                        extra = "; prefaulted";
                    }
                }
                if (keep) {
                    out.note = std::string(page_kind_name(PageKind::Hugetlb)) +
                               (opt.node >= 0 && known ? " (node " + nd + "'s pool had " + gib(room) + " free)" : std::string()) + extra;
                    return true;
                }
                munmap(p, (size_t) len);   // the pages it did get go back to the pool
                out.base = nullptr;
                out.bytes = 0;
                out.bound = false;
                out.bind_note.clear();
                out.kind = PageKind::Small;
            }
        }
    }

    // ---- 2. a 2 MiB-ALIGNED mapping (THP only backs aligned 2 MiB ranges: an unaligned 50 GB mapping gets 4 KiB pages at both
    // ends and, on kernels before 6.7, wherever the kernel put the start), then MADV_HUGEPAGE before the first touch
    const uint64_t span = len + kHuge;
    void* raw = mmap(nullptr, (size_t) span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) {
        out.note = "mmap of " + gib(requested) + " failed: " + std::strerror(errno);
        return false;
    }
    const uintptr_t r0 = (uintptr_t) raw, a0 = round_up(r0, kHuge), a1 = a0 + len, r1 = r0 + span;
    if (a0 > r0) munmap(raw, (size_t) (a0 - r0));
    if (r1 > a1) munmap((void*) a1, (size_t) (r1 - a1));
    out.base = (void*) a0;
    out.bytes = len;
    out.kind = PageKind::Small;
    std::string thp_note;
    const bool no_thp = [] { const char* e = std::getenv("STRATA_NO_THP"); return e != nullptr && e[0] == '1'; }();   // A/B arm
    if (opt.thp && !no_thp) {
        const std::string mode = thp_mode_of(root);
        if (madvise(out.base, (size_t) len, MADV_HUGEPAGE) != 0)
            thp_note = std::string("madvise(MADV_HUGEPAGE) refused (") + std::strerror(errno) + ")";
        else if (mode == "never")
            thp_note = "THP is off system-wide (transparent_hugepage/enabled = never)";
        else {
            out.kind = PageKind::Thp;
            if (!mode.empty()) thp_note = "system THP mode " + mode;
        }
    } else if (opt.thp) {
        thp_note = "THP off (STRATA_NO_THP=1)";
    }
    bind(out.base, 4096);
    std::string tail = why;
    if (!thp_note.empty()) tail += (tail.empty() ? "" : "; ") + thp_note;
    out.note = std::string(page_kind_name(out.kind)) + (tail.empty() ? "" : " [" + tail + "]");
    return true;
}

void numa_unmap_arena(void* base, uint64_t bytes) {
    if (base != nullptr) munmap(base, (size_t) round_up(bytes, kHuge));
}

PageStats numa_page_stats(const void* p, uint64_t bytes) {
    PageStats st;
    std::ifstream f("/proc/self/smaps");
    if (!f || p == nullptr || bytes == 0) return st;
    const uint64_t lo = (uint64_t) (uintptr_t) p, hi = lo + bytes;
    double frac = 0.0, rss = 0.0, thp = 0.0, huge = 0.0;
    std::string line;
    while (std::getline(f, line)) {
        unsigned long long a = 0, b = 0;
        if (std::sscanf(line.c_str(), "%llx-%llx ", &a, &b) == 2) {   // a VMA header; the field lines never match this
            const uint64_t s = std::max<uint64_t>(a, lo), e = std::min<uint64_t>(b, hi);
            frac = e > s && b > a ? (double) (e - s) / (double) (b - a) : 0.0;
            continue;
        }
        if (frac <= 0.0) continue;
        unsigned long long kb = 0;
        if (line.compare(0, 4, "Rss:") == 0 && std::sscanf(line.c_str() + 4, " %llu", &kb) == 1) rss += frac * (double) kb;
        else if (line.compare(0, 14, "AnonHugePages:") == 0 && std::sscanf(line.c_str() + 14, " %llu", &kb) == 1) thp += frac * (double) kb;
        else if (line.compare(0, 16, "Private_Hugetlb:") == 0 && std::sscanf(line.c_str() + 16, " %llu", &kb) == 1) huge += frac * (double) kb;
        else if (line.compare(0, 15, "Shared_Hugetlb:") == 0 && std::sscanf(line.c_str() + 15, " %llu", &kb) == 1) huge += frac * (double) kb;
    }
    st.ok = true;
    st.rss = (uint64_t) (rss * 1024.0);
    st.thp = (uint64_t) (thp * 1024.0);
    st.hugetlb = (uint64_t) (huge * 1024.0);
    return st;
}
#else
Populate numa_populate_write(void*, uint64_t, int, const std::vector<int>&, std::string& err) {
    err = "no madvise on this OS";
    return Populate::Unsupported;
}
bool numa_map_arena(uint64_t, const ArenaMapOptions&, ArenaMap& out) {
    out = ArenaMap{};
    out.note = "no NUMA mapping on this OS";
    return false;
}
void numa_unmap_arena(void*, uint64_t) {}
PageStats numa_page_stats(const void*, uint64_t) { return PageStats{}; }
#endif

std::string PageStats::text() const {
    if (!ok) return "page sizes not checked";
    if (hugetlb > 0 && hugetlb >= rss) return "2 MiB hugetlb pages: " + gib(hugetlb) + " resident";   // smaps' Rss leaves hugetlb out
    if (rss == 0) return "no page resident yet";
    if (thp > 0) return "THP: " + gib(thp) + " of " + gib(rss) + " resident in 2 MiB pages";
    return "4 KiB pages: " + gib(rss) + " resident (no THP)";
}

// ================================ NumaBuffer ================================

NumaBuffer& NumaBuffer::operator=(NumaBuffer&& o) noexcept {
    if (this != &o) {
        release();
        base_ = o.base_; bytes_ = o.bytes_; map_bytes_ = o.map_bytes_; locked_ = o.locked_; node_ = o.node_;
        bound_ = o.bound_; pinned_ = o.pinned_; kind_ = o.kind_; unpin_ = std::move(o.unpin_);
        note_ = std::move(o.note_); lock_note_ = std::move(o.lock_note_); pin_note_ = std::move(o.pin_note_);
        o.base_ = nullptr; o.bytes_ = o.map_bytes_ = o.locked_ = 0; o.bound_ = o.pinned_ = false; o.kind_ = PageKind::Small;
        o.unpin_ = nullptr;
    }
    return *this;
}

NumaBuffer::~NumaBuffer() { release(); }

void NumaBuffer::release() {
#if defined(__linux__)
    if (base_ != nullptr) {
        if (pinned_ && unpin_) unpin_(base_, bytes_);   // the registration must not outlive the pages
        if (locked_ > 0) munlock(base_, (size_t) locked_);
        numa_unmap_arena(base_, map_bytes_);
    }
#endif
    base_ = nullptr;
    bytes_ = map_bytes_ = locked_ = 0;
    bound_ = pinned_ = false;
    kind_ = PageKind::Small;
    unpin_ = nullptr;
}

bool NumaBuffer::allocate(uint64_t bytes, int node, bool use_hugepages, const std::vector<int>& cpus) {
    release();
    note_.clear();
    lock_note_.clear();
    pin_note_.clear();
    node_ = node;
#if defined(__linux__)
    if (bytes == 0) { note_ = "zero bytes"; return false; }
    ArenaMapOptions o;
    o.node = node;
    o.hugetlb = use_hugepages;
    o.thp = use_hugepages;
    o.cpus = cpus;
    ArenaMap m;
    if (!numa_map_arena(bytes, o, m)) { note_ = m.note; return false; }
    base_ = (uint8_t*) m.base;
    bytes_ = bytes;
    map_bytes_ = m.bytes;
    kind_ = m.kind;
    bound_ = m.bound;
    note_ = m.note + "; " + m.bind_note;
    return true;
#else
    (void) bytes; (void) use_hugepages; (void) cpus;
    note_ = "no NUMA binding on this OS";
    return false;
#endif
}

bool NumaBuffer::pin(const PinFn& pin_fn, const UnpinFn& unpin_fn) {
    pin_note_.clear();
    if (base_ == nullptr || !pin_fn) { pin_note_ = "nothing to pin"; return false; }
    std::string how;
    if (!pin_fn(base_, bytes_, how)) { pin_note_ = how; return false; }
    pinned_ = true;
    unpin_ = unpin_fn;
    pin_note_ = how;
    return true;
}

bool NumaBuffer::lock() {
#if defined(__linux__)
    if (base_ == nullptr) { lock_note_ = "nothing to lock"; return false; }
    if (mlock(base_, (size_t) bytes_) != 0) {
        lock_note_ = std::string("mlock failed (") + std::strerror(errno) + "; raise ulimit -l)";
        return false;
    }
    locked_ = bytes_;
    lock_note_ = "mlock";
    return true;
#else
    lock_note_ = "no mlock on this OS";
    return false;
#endif
}

double numa_copy(void* dst, const void* src, uint64_t bytes, const std::vector<int>& cpus, int threads) {
    const auto t0 = std::chrono::steady_clock::now();
    if (bytes == 0) return 0.0;
    int n = std::max(1, threads);
    // contiguous parts, 2 MiB aligned, so a hugetlb page is faulted by exactly one thread
    uint64_t part = round_up((bytes + (uint64_t) n - 1) / (uint64_t) n, kHuge);
    n = (int) ((bytes + part - 1) / part);
    auto run = [&](int t) {
#if defined(__linux__)
        if (!cpus.empty()) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpus[(size_t) t % cpus.size()], &set);
            pthread_setaffinity_np(pthread_self(), sizeof set, &set);   // best effort: a refusal only costs speed
        }
#else
        (void) cpus;
#endif
        const uint64_t a = (uint64_t) t * part, b = std::min<uint64_t>(bytes, a + part);
        if (a < b) std::memcpy((uint8_t*) dst + a, (const uint8_t*) src + a, (size_t) (b - a));
    };
    std::vector<std::thread> th;
    th.reserve((size_t) n);
    bool spawn_failed = false;
    for (int t = 0; t < n; ++t) {
        try { th.emplace_back(run, t); }
        catch (...) { spawn_failed = true; const uint64_t a = (uint64_t) t * part, b = std::min<uint64_t>(bytes, a + part);
                      if (a < b) std::memcpy((uint8_t*) dst + a, (const uint8_t*) src + a, (size_t) (b - a)); }
    }
    for (auto& x : th) x.join();
    (void) spawn_failed;
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ================================ THE PLAN ================================

bool parse_numa_mode(const std::string& text, NumaMode& out) {
    std::string s = text;
    for (char& c : s) c = (char) std::tolower((unsigned char) c);
    if (s == "auto") { out = NumaMode::Auto; return true; }
    if (s == "mirror" || s == "on" || s == "1") { out = NumaMode::Mirror; return true; }
    if (s == "off" || s == "0") { out = NumaMode::Off; return true; }
    return false;
}

const char* numa_mode_name(NumaMode m) { return m == NumaMode::Mirror ? "mirror" : m == NumaMode::Off ? "off" : "auto"; }

NumaMode numa_mode_with_env(NumaMode cli, std::string* note) {
    const char* e = std::getenv("STRATA_NUMA_MIRROR");
    if (e == nullptr || *e == '\0') return cli;
    const std::string v = e;
    if (v == "0") {
        if (note) *note = "STRATA_NUMA_MIRROR=0 overrides --numa";
        return NumaMode::Off;
    }
    if (v == "1") {
        if (note) *note = "STRATA_NUMA_MIRROR=1 overrides --numa";
        return NumaMode::Mirror;
    }
    if (note) *note = "STRATA_NUMA_MIRROR=" + v + " ignored (0 or 1)";
    return cli;
}

uint64_t numa_headroom_bytes() {
    if (const char* e = std::getenv("STRATA_NUMA_HEADROOM_GIB"); e != nullptr && *e != '\0') {
        const double v = std::atof(e);
        if (v >= 0.0) return (uint64_t) (v * (double) kGiB);
    }
    return 6 * kGiB;
}

MirrorPlan plan_arena_mirror(const MirrorInputs& in) {
    MirrorPlan p;
    p.mode = in.mode;
    p.copy_bytes = in.arena_bytes;
    auto no = [&](std::string why) {
        p.mirror = false;
        p.why_not = std::move(why);
        return p;
    };
    if (in.mode == NumaMode::Off) return no("--numa off (or STRATA_NUMA_MIRROR=0): one copy, placed by first touch");
    if (in.topo == nullptr || !in.topo->available)
        return no("the NUMA topology is not available (" + (in.topo ? in.topo->why : std::string("not probed")) + "): one copy");
    const NumaTopology& t = *in.topo;
    if (t.nodes.size() < 2) return no("this machine has one NUMA node: nothing to mirror");
    if (!in.full_ram)
        return no("the experts are not all resident in one RAM arena (" +
                  (in.not_full_ram_why.empty() ? std::string("mmap / low-RAM / RAM-budget mode") : in.not_full_ram_why) +
                  "): one copy is kept, placed by the OS");
    if (in.shared_file)
        return no("--shared-expert-arena backs the arena with a file other processes share: its placement is not this "
                  "process's to choose, one copy is kept");
    if (in.arena_bytes == 0) return no("the arena is empty");

    // Which nodes can run a worker at all (taskset / numactl --cpunodebind / a cpuset may leave a node out).
    for (const NumaNode& n : t.nodes) {
        bool has = false;
        for (int c : n.cpus)
            if (in.allowed_cpus.empty() || std::binary_search(in.allowed_cpus.begin(), in.allowed_cpus.end(), c)) { has = true; break; }
        if (has) p.cpu_nodes.push_back(n.id);
    }
    p.primary_node = t.gpu_node;
    p.gpu_node_assumed = !t.gpu_node_known;
    const int pidx = t.index_of_node(p.primary_node);
    if (pidx < 0) return no("the GPU's node " + std::to_string(p.primary_node) + " is not in the topology: one copy");
    auto node_of = [&](int id) -> const NumaNode& { return t.nodes[(size_t) t.index_of_node(id)]; };
    // The package (socket) of a node for the one-copy-per-socket rule; a node whose package is unknown (no cpu topology
    // files) or spans several counts as a package of its own, which is the one-copy-per-node rule.
    auto pkg = [&](const NumaNode& n) { return n.package >= 0 ? n.package : -1000 - n.id; };
    // Room for a copy on node `n`: a node whose own free hugepage pool holds the whole copy takes it from the pool and needs
    // only the headroom in normal memory; every other node takes the copy from normal memory (hugetlb needs the whole copy).
    auto room_for = [&](const NumaNode& n) { return n.usable_bytes() + (n.pool_holds(in.arena_bytes) ? in.arena_bytes : 0); };

    int others = 0;
    for (int id : p.cpu_nodes) others += id != p.primary_node ? 1 : 0;
    if (in.mode == NumaMode::Mirror) {
        // asked for: one replica on every other node that has CPUs, however the sockets divide into nodes
        for (int id : p.cpu_nodes)
            if (id != p.primary_node) p.replica_nodes.push_back(id);
    } else {
        // Auto: ONE copy per package.  The GPU's package is served by the primary; every other package gets one replica, on its
        // node with the most room, and its other nodes (sub-NUMA clustering) read that one.
        std::vector<int> pkgs;
        for (int id : p.cpu_nodes) {
            const int k = pkg(node_of(id));
            if (k != pkg(t.nodes[(size_t) pidx]) && std::find(pkgs.begin(), pkgs.end(), k) == pkgs.end()) pkgs.push_back(k);
        }
        for (int k : pkgs) {
            const NumaNode* best = nullptr;
            for (int id : p.cpu_nodes) {
                const NumaNode& n = node_of(id);
                if (pkg(n) != k) continue;
                if (best == nullptr || room_for(n) > room_for(*best)) best = &n;   // ascending ids: the lowest wins a tie
            }
            p.replica_nodes.push_back(best->id);
        }
    }
    if (p.replica_nodes.empty()) {
        if (others == 0)
            return no("every CPU this process may use is on node " + std::to_string(p.primary_node) +
                      (p.gpu_node_assumed ? " (assumed to be the GPU's)" : " (the GPU's)") +
                      " (taskset / numactl --cpunodebind?): nothing would read a replica");
        std::string ids;
        for (int id : p.cpu_nodes)
            if (id != p.primary_node) ids += (ids.empty() ? "" : ",") + std::to_string(id);
        return no("the other NUMA node" + std::string(others == 1 ? " with CPUs (node " : "s with CPUs (nodes ") + ids +
                  ") sit in the GPU's own socket (sub-NUMA clustering / NPS): --numa auto keeps one copy per socket, so "
                  "there is nothing to mirror (--numa mirror makes one per node)");
    }
    // who reads which copy: a holder reads its own, a node of a package without a holder... every package has one by now
    for (int id : p.cpu_nodes) {
        int src = id;
        const bool holder = id == p.primary_node || std::find(p.replica_nodes.begin(), p.replica_nodes.end(), id) != p.replica_nodes.end();
        if (!holder) {
            const int k = pkg(node_of(id));
            src = k == pkg(t.nodes[(size_t) pidx]) ? p.primary_node : id;
            for (int h : p.replica_nodes)
                if (pkg(node_of(h)) == k) { src = h; break; }
        }
        p.reads.emplace_back(id, src);
    }
    {
        std::vector<int> hold{p.primary_node};
        hold.insert(hold.end(), p.replica_nodes.begin(), p.replica_nodes.end());
        for (int h : hold) {
            std::string rd;
            for (const auto& r : p.reads)
                if (r.second == h) rd += (rd.empty() ? "" : ",") + std::to_string(r.first);
            p.layout += (p.layout.empty() ? "" : "; ") + std::string(h == p.primary_node ? "the primary on node " : "a replica on node ") +
                        std::to_string(h) + " is read by node" + (rd.find(',') != std::string::npos ? "s " : " ") + (rd.empty() ? std::string("none") : rd);
        }
    }

    // The room: each node that gets a copy must have the copy plus the headroom, counting the file cache the kernel
    // reclaims for a bound allocation - or just the headroom where its own free hugepage pool holds the whole copy.
    std::vector<int> holders{p.primary_node};
    holders.insert(holders.end(), p.replica_nodes.begin(), p.replica_nodes.end());
    uint64_t normal_copies = 0;   // bytes of copies that will come from normal memory (a pooled copy is not in MemAvailable)
    for (int id : holders) {
        const NumaNode& n = node_of(id);
        if (!n.has_meminfo) return no("node " + std::to_string(id) + "'s free memory cannot be read: one copy is kept");
        const bool pooled = n.pool_holds(in.arena_bytes);
        const uint64_t own = pooled ? 0 : in.arena_bytes;
        normal_copies += own;
        if (n.usable_bytes() < own + in.headroom) {
            std::string pool;
            if (n.has_huge && n.huge_total > 0)
                pool = pooled ? "; its " + gib(n.huge_free) + " free hugepage pool holds the copy"
                              : "; its 2 MiB hugepage pool has " + gib(n.huge_free) + " free, not a whole copy, so the copy takes normal memory";
            return no("node " + std::to_string(id) + " has " + gib(n.usable_bytes()) + " usable (" + gib(n.mem_free) + " free + " +
                      gib(n.mem_file) + " file cache" + pool + ") but " +
                      (pooled ? std::string("the headroom") : "a " + gib(in.arena_bytes) + " copy plus " + "the headroom") + " of " +
                      gib(in.headroom) + " (STRATA_NUMA_HEADROOM_GIB) needs " + gib(own + in.headroom) + ": one copy is kept");
        }
    }
    const uint64_t all = normal_copies + in.headroom;
    if (in.global_available != 0 && in.global_available < all)
        return no("the system has " + gib(in.global_available) + " available (MemAvailable, cgroup limit included) but " +
                  std::to_string(holders.size()) + " copies of " + gib(in.arena_bytes) + " plus " + gib(in.headroom) +
                  " of headroom need " + gib(all) + (normal_copies != in.arena_bytes * holders.size() ? " (copies in the hugepage pools excluded)" : "") +
                  ": one copy is kept");
    p.mirror = true;
    return p;
}

// ================================ THE MIRROR ================================

uint64_t NumaReplicas::replica_bytes() const {
    uint64_t b = 0;
    for (const auto& x : bufs_) b += x.bytes();
    return b;
}

void NumaReplicas::release() {
    bufs_.clear();
    mirror_ = ArenaMirror{};
    primary_placement_ = PlacementReport{};
}

int NumaReplicas::build(const uint8_t* primary, uint64_t bytes, uint64_t capacity, const MirrorPlan& plan,
                        const NumaTopology& topo, const MirrorOptions& opt, std::vector<std::string>& log) {
    release();
    if (!plan.mirror || primary == nullptr || bytes == 0 || capacity < bytes) return 0;
    const int pidx = topo.index_of_node(plan.primary_node);
    if (pidx < 0) { log.push_back("mirror: the primary's node " + std::to_string(plan.primary_node) + " is not in the topology"); return 0; }

    primary_placement_ = numa_sample_placement(primary, bytes, opt.samples);
    {
        const PlacementReport& r = primary_placement_;
        const bool on = r.ok && r.sampled > 0 && r.on_node(plan.primary_node) * 10 >= r.sampled * 9;
        log.push_back("primary on node " + std::to_string(plan.primary_node) + ": " + r.text() +
                      (r.ok && !on ? "; WARNING: not on the GPU's node - cache fills and PCIe reads cross the socket link, "
                                     "and the mirror's gain is smaller" : ""));
    }

    ArenaMirror m;
    m.primary = primary;
    m.bytes = capacity;
    m.primary_index = pidx;
    m.copy.assign(topo.nodes.size(), nullptr);
    m.copy[(size_t) pidx] = primary;

    for (int id : plan.replica_nodes) {
        const int ni = topo.index_of_node(id);
        if (ni < 0) { log.push_back("replica for node " + std::to_string(id) + " skipped: not in the topology"); continue; }
        // THE FIRST TOUCH IS THE COPY, and it runs on CPUs of the node: the bind below is the guarantee, this is the
        // second line of defence (and what places the pages where the bind failed, or where the node does not exist in
        // a test).
        // the node's CPUs this process may actually run on (a taskset that leaves some out would refuse the pin)
        std::vector<int> cpus = topo.nodes[(size_t) ni].cpus;
        if (const std::vector<int> allowed = numa_allowed_cpus(); !allowed.empty()) {
            std::vector<int> ok;
            for (int c : cpus)
                if (std::binary_search(allowed.begin(), allowed.end(), c)) ok.push_back(c);
            if (!ok.empty()) cpus.swap(ok);
        }
        NumaBuffer b;
        if (!b.allocate(capacity, id, opt.hugepages, cpus)) {
            log.push_back("replica on node " + std::to_string(id) + " not made: " + b.note());
            continue;
        }
        const double sec = numa_copy(b.data(), primary, bytes, cpus, opt.threads);
        const PlacementReport rep = numa_sample_placement(b.data(), bytes, opt.samples);
        char tbuf[96];
        std::snprintf(tbuf, sizeof tbuf, "%s copied in %.2f s (%.1f GiB/s)", gib(bytes).c_str(), sec,
                      sec > 0 ? (double) bytes / (double) kGiB / sec : 0.0);
        // b.note() says which page size was asked for and got ("THP ... [hugetlb: node 0's pool has 25 GiB free, 50 GiB needed]");
        // the smaps check after the copy says what the kernel really delivered
        std::string line = "replica on node " + std::to_string(id) + ": " + b.note() + "; " + tbuf + "; " + rep.text() + "; " +
                           numa_page_stats(b.data(), bytes).text();
        const bool landed = rep.ok && rep.sampled > 0 && rep.on_node(id) * 10 >= rep.sampled * 9;
        if (!landed && !opt.faked) {
            // a replica that is not on its node costs 30-50 GB and buys nothing (its readers would cross UPI to it)
            log.push_back(line + "; DROPPED: the pages are not on node " + std::to_string(id) +
                          (rep.ok ? "" : " or cannot be checked"));
            continue;
        }
        if (!landed) line += "; (test topology: continuing)";
        // KEEP IT RESIDENT.  The primary is pinned by the CUDA registration, which is not charged to RLIMIT_MEMLOCK; mlock is
        // (8 MiB for a user by default), so a replica that only mlock-ed was pageable.  Register it the same way (pinning
        // only: nothing but the CPU pool reads a replica), and mlock - under the arena lock policy - only where that fails.
        bool kept_resident = false;
        if (opt.pin) {
            kept_resident = b.pin(opt.pin, opt.unpin);
            line += kept_resident ? "; pinned: " + b.pin_note() : "; not pinned: " + b.pin_note();
        }
        if (!kept_resident) {
            if (opt.lock) {
                b.lock();
                line += "; " + b.lock_note();
            } else {
                line += "; not locked (the arena lock policy is off: STRATA_ARENA_LOCK=0)";
            }
        }
        log.push_back(line);
        m.copy[(size_t) ni] = b.data();
        bufs_.push_back(std::move(b));
    }
    if (bufs_.empty()) return 0;
    // the nodes that share a socket's copy (sub-NUMA clustering) read the same one; a node whose holder was not kept reads the
    // primary (null -> the pool's identity translation)
    for (const auto& r : plan.reads) {
        const int ri = topo.index_of_node(r.first), hi = topo.index_of_node(r.second);
        if (ri < 0 || hi < 0 || ri == hi) continue;
        if (m.copy[(size_t) hi] != nullptr) m.copy[(size_t) ri] = m.copy[(size_t) hi];
        else log.push_back("node " + std::to_string(r.first) + "'s workers read the primary: the copy on node " + std::to_string(r.second) + " was not kept");
    }
    mirror_ = std::move(m);
    return (int) bufs_.size();
}

}  // namespace strata::platform
