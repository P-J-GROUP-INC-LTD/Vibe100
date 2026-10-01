// include/strata/platform/numa.hpp - Vibe100 WP-F: NUMA topology, memory binding and the mirrored expert arena.
//
// WHY THIS EXISTS.  A dual-socket box (the target: a Dell Precision 7920, 2x Xeon Gold 6226, two NUMA nodes, a V100 on one
// socket's PCIe) has two memory controllers and a UPI link between the sockets.  The expert pool's workers are pinned to
// cores on BOTH sockets, but the expert arena is ONE allocation whose pages sit on whichever node touched them first, so
// about half of every worker's reads cross UPI - and the pool is DRAM-bound (pool.hpp), so that is the whole speed of the
// CPU path.  The user measured ~2x decode from MIRRORING the experts, one full copy per node, with each worker reading the
// copy of its own node.  Strata's arena is 23-50 GB and the box has 384 GB, so two copies fit easily.  This header is the
// platform half: it finds the nodes, decides whether a mirror is possible, binds memory to a node, and checks where pages
// actually landed.  `strata::kernels::cpu::ExpertPool` (pool.hpp) does the other half - translating a worker's expert
// pointer to its node's copy - and `ArenaExpertSource` (core/expert_source.hpp) wires the two.
//
// PAGE BACKING IS PART OF THE DESIGN.  Each copy is 23-50 GB read at DRAM speed by every core, so 4 KiB pages (one TLB entry
// per 4 KiB: 12 million for 50 GB) cost bandwidth.  A copy therefore takes, in this order: 2 MiB hugetlb pages when ITS
// node's own pool (`.../node<N>/hugepages/hugepages-2048kB/free_hugepages`) holds the whole copy - Linux spreads
// `vm.nr_hugepages` EVENLY over the nodes, so a pool sized for one arena is half a copy per node - else transparent
// hugepages (`madvise(MADV_HUGEPAGE)` on a 2 MiB-aligned mapping, before the first touch; the default THP mode of Ubuntu is
// `madvise`, which is OFF for memory that does not ask), else 4 KiB pages; and the line that reports a copy says which it
// got.  A hugetlb mapping bound to a node whose pool was promised elsewhere is a SIGBUS at the first touch, so it is
// prefaulted at once with `MADV_POPULATE_WRITE` (Linux >= 5.14), which returns an error instead, and a failure remaps the
// copy as THP.
//
// NO libnuma.  Everything below is sysfs reads, three raw syscalls (mbind, get_mempolicy, move_pages) and libc's madvise /
// mmap, so the engine still has nothing to install and the prebuilt release needs no new shared library.  Linux only; elsewhere topology
// reports "not available" and the engine keeps its single copy.
//
// TESTABLE ON ONE NODE.  Every sysfs read goes through a root that `STRATA_SYSFS_ROOT` (or the `sysfs_root` argument)
// replaces, so a test can build a fake two-node tree under a temp directory and exercise the parser and the planner on a
// one-node VM; binding to a node that does not exist fails, and the builder reports that and continues when it is told
// the topology is fake (`MirrorOptions::faked`).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::platform {

// ================================ TOPOLOGY ================================

struct NumaNode {
    int id = 0;                       ///< the kernel's node number (`node<id>` in sysfs; ids need not be contiguous)
    std::vector<int> cpus;            ///< logical CPUs of the node, ascending (a memory-only node has none)
    uint64_t mem_total = 0;           ///< bytes, from the node's meminfo
    uint64_t mem_free = 0;            ///< MemFree
    uint64_t mem_file = 0;            ///< Active(file) + Inactive(file): clean page cache the kernel reclaims for a bound allocation
    bool has_meminfo = false;         ///< false: the node's meminfo could not be read (the planner then refuses it)
    /// The node's 2 MiB hugetlb pool (`hugepages/hugepages-2048kB/{free,nr}_hugepages`).  Pages reserved by
    /// `vm.nr_hugepages` are NOT in MemFree, and a MAP_HUGETLB mapping bound to the node draws on THIS pool alone.
    bool has_huge = false;            ///< the pool's files were readable
    uint64_t huge_free = 0;           ///< bytes of FREE pages (may include pages another reservation has promised)
    uint64_t huge_total = 0;          ///< bytes of the whole pool
    /// The physical package (socket) of the node's CPUs, from `cpu<N>/topology/physical_package_id`: -1 unknown (no
    /// such file), -2 the CPUs span several packages.  Sub-NUMA clustering (SNC / NPS) makes several nodes of one socket.
    int package = -1;
    /// What a MPOL_BIND allocation on this node can count on in NORMAL memory: free pages plus the file cache it will
    /// reclaim (the experts were just read through that cache, so MemFree alone would call a node full that is not).
    /// The hugetlb pool is not in it: a copy the pool holds whole takes pool pages instead (`pool_holds`).
    uint64_t usable_bytes() const { return mem_free + mem_file; }
    /// Whether the node's free 2 MiB pool holds a whole copy of `copy` bytes: that copy then takes no normal memory.
    bool pool_holds(uint64_t copy) const { return has_huge && copy > 0 && huge_free >= (copy + (2ull << 20) - 1) / (2ull << 20) * (2ull << 20); }
};

struct NumaTopology {
    bool available = false;           ///< the nodes could be read at all
    std::string why;                  ///< when `!available`: the one-line reason ("not Linux", "no .../node directory")
    std::string root;                 ///< the sysfs root that was read ("/sys" or STRATA_SYSFS_ROOT)
    bool faked = false;               ///< the root came from STRATA_SYSFS_ROOT / the argument: a test topology
    std::vector<NumaNode> nodes;      ///< ascending by id
    std::string gpu_bdf;              ///< the GPU's PCI address, lower case ("0000:3b:00.0"), empty when not asked
    int gpu_node = 0;                 ///< the node ID the GPU hangs off (the lowest node when unknown)
    bool gpu_node_known = false;      ///< false: no GPU asked, sysfs said -1, or the file was unreadable - `gpu_node` is a guess
    std::string gpu_note;             ///< how `gpu_node` was found, for the startup line

    /// Index into `nodes` of node id `id`, or -1.
    int index_of_node(int id) const;
    /// Index into `nodes` of the node that owns logical CPU `cpu`, or -1.
    int index_of_cpu(int cpu) const;
    bool multi() const { return available && nodes.size() >= 2; }
    /// The `transparent_hugepage/enabled` mode ("always" / "madvise" / "never"; empty = unreadable), from the same root.
    std::string thp_mode;
};

/// "0-11,24-35" -> {0..11, 24..35}.  `ok` (when given) is false for anything that is not a cpulist; an empty string is a
/// valid empty list (a memory-only node).  Ascending, duplicates removed.
std::vector<int> parse_cpulist(const std::string& text, bool* ok = nullptr);
/// The reverse, for messages: {0..11, 24..35} -> "0-11,24-35".
std::string format_cpulist(const std::vector<int>& cpus);

/// Reads `<root>/devices/system/node/node*/{cpulist,meminfo,hugepages/hugepages-2048kB/*}`, `cpu<N>/topology/
/// physical_package_id` of every node's CPUs and `kernel/mm/transparent_hugepage/enabled`.  `sysfs_root` empty ->
/// `STRATA_SYSFS_ROOT`, else "/sys".  Never throws; a failure is `available == false` with `why`.
NumaTopology numa_discover(const std::string& sysfs_root = {});

/// Finds the GPU's node: `pci_bus_id` is what `cudaDeviceGetPCIBusId` returns ("0000:3B:00.0"); the node is read from
/// `<root>/bus/pci/devices/<lower-cased id>/numa_node`.  A `-1` there (BIOS without SRAT affinity, a VM) or an unreadable
/// file means UNKNOWN: the GPU is then treated as being on the first node and `gpu_note` says so.
/// `STRATA_NUMA_GPU_NODE=N` overrides all of it (the user read `nvidia-smi topo -m`); a node that does not exist is ignored.
void numa_set_gpu(NumaTopology& topo, const std::string& pci_bus_id);

/// "2 NUMA nodes: node0 cpus 0-11 (181.2 GiB usable), node1 cpus 12-23 (190.4 GiB usable); GPU 0000:3b:00.0 on node 1"
std::string numa_describe(const NumaTopology& topo);

/// The logical CPUs this process may run on (sched_getaffinity); empty when it cannot be read or off Linux.
std::vector<int> numa_allowed_cpus();

// ================================ MEMORY POLICY ================================

/// Whether the mbind / move_pages / get_mempolicy syscalls exist in this build (Linux).
bool numa_syscalls_available();

/// `mbind(p, bytes, MPOL_BIND, {node})` - the pages of [p, p + bytes) are allocated on `node` (id), and ONLY there, when
/// they are first touched.  Must run BEFORE the first touch (no MPOL_MF_MOVE: nothing is migrated).  `p` must be page
/// aligned; `bytes` is rounded up to `page` (4 KiB, or 2 MiB for a hugetlb mapping).  False + `err` (strerror) on failure.
bool numa_bind_memory(void* p, uint64_t bytes, int node, std::string& err, uint64_t page = 4096);

/// Where a sample of pages of [p, p + bytes) actually sit, from `move_pages` in QUERY mode (nodes == NULL: nothing moves).
struct PlacementReport {
    bool ok = false;                  ///< the query ran
    std::string note;                 ///< why not, when !ok
    uint64_t sampled = 0;             ///< pages asked about
    uint64_t not_present = 0;         ///< pages that had no frame yet (never touched) or the query could not place
    std::vector<std::pair<int, uint64_t>> by_node;   ///< (node id, pages) for the pages that were found
    uint64_t on_node(int id) const;
    /// "64 of 64 sampled pages on node 1" / "31 on node 0, 33 on node 1 (of 64 sampled)"
    std::string text() const;
};
/// Samples `samples` pages spread evenly over the range (first and last included), `page` bytes apart at the finest.
PlacementReport numa_sample_placement(const void* p, uint64_t bytes, int samples = 64, uint64_t page = 4096);

/// Whether this thread's (and so, with numactl, the process's) default memory policy is INTERLEAVE: `numactl
/// --interleave=all` sets it for every allocation, including whatever the arena's `mbind` does not name.  `detail` says
/// the mode and node set.  The arena's own mbind (a per-mapping policy) overrides the task policy, so the mirror still
/// lands where it was told, but the combination is pointless and the log says so.
bool numa_task_policy_interleaved(std::string& detail);

/// Free 2 MiB hugepages on `node` in bytes (the per-node pool: a MAP_HUGETLB mapping bound to a node whose pool is short
/// takes SIGBUS at the first touch, it does not fall back).  `known` false when the file is unreadable.
uint64_t numa_node_free_hugepage_bytes(int node, bool& known, const std::string& sysfs_root = {});

// ================================ PAGE BACKING ================================

enum class PageKind { Small, Thp, Hugetlb };
/// "4 KiB pages" / "THP (2 MiB pages, madvise)" / "2 MiB hugetlb pages"
const char* page_kind_name(PageKind k);

/// `transparent_hugepage/enabled` of `sysfs_root` (empty -> STRATA_SYSFS_ROOT, else /sys): "always", "madvise", "never",
/// or empty when it cannot be read.  With "never" a MADV_HUGEPAGE request is accepted and ignored.
std::string numa_thp_mode(const std::string& sysfs_root = {});

/// `madvise(MADV_POPULATE_WRITE)` (Linux >= 5.14) over [p, p + bytes) from `threads` threads (pinned round-robin to `cpus`
/// when given): every page is allocated NOW, with the mapping's policy, and a page that cannot be had - the case that is a
/// SIGBUS at the first touch of a hugetlb mapping whose node pool was promised elsewhere - is an error here instead.
/// Unsupported: the kernel lacks it (EINVAL), nothing was done.  Failed: `err` says why; some pages may be populated.
enum class Populate { Ok, Unsupported, Failed };
Populate numa_populate_write(void* p, uint64_t bytes, int threads, const std::vector<int>& cpus, std::string& err);

struct ArenaMapOptions {
    int node = -1;                    ///< bind to this node (id) before the first touch; -1 = no binding
    bool hugetlb = true;              ///< try MAP_HUGETLB (bound: only when the node's own pool holds the whole mapping)
    bool thp = true;                  ///< else (or when hugetlb is not asked) `madvise(MADV_HUGEPAGE)` on the aligned mapping
    bool prefault_hugetlb = true;     ///< a hugetlb mapping is prefaulted (numa_populate_write) - bound or under a task policy; failure -> THP
    std::vector<int> cpus;            ///< CPUs of the node, for the prefault threads (empty: unpinned)
    int threads = 8;                  ///< prefault threads
    std::string sysfs_root;           ///< "" = STRATA_SYSFS_ROOT / /sys (the pool and THP files)
    bool fail_prefault_for_test = false;   ///< test hook: treat the hugetlb prefault as failed (a pool promised elsewhere)
};
struct ArenaMap {
    void* base = nullptr;             ///< 2 MiB-aligned; the mapping is exactly [base, base + bytes) (munmap releases it)
    uint64_t bytes = 0;               ///< `requested` rounded up to 2 MiB
    PageKind kind = PageKind::Small;
    bool bound = false;               ///< mbind(MPOL_BIND) to `node` succeeded (false also when none was asked)
    std::string bind_note;            ///< "mbind(MPOL_BIND) to node 1" / "mbind to node 1 FAILED (...)"; empty when no bind was asked
    std::string note;                 ///< why the page kind is what it is ("hugetlb: node 1 pool 26.0 GiB < 50.0 GiB; THP madvise ...")
};
/// ONE ANONYMOUS PRIVATE MAPPING for a big arena, in the order above (hugetlb, THP, 4 KiB), bound to `opt.node` before any
/// page is touched.  False only when no mapping could be made (`out.note` says why).  Unmap with `numa_unmap_arena`.
bool numa_map_arena(uint64_t requested, const ArenaMapOptions& opt, ArenaMap& out);
void numa_unmap_arena(void* base, uint64_t bytes);

/// What the kernel actually gave a mapping: from /proc/self/smaps (AnonHugePages / Private_Hugetlb / Rss of the VMAs that
/// overlap [p, p + bytes), pro rata where a VMA is larger).  Call it AFTER the pages were touched.
struct PageStats {
    bool ok = false;
    uint64_t rss = 0, thp = 0, hugetlb = 0;    ///< bytes resident / of them in THP / in hugetlb pages
    /// "THP: 49.8 of 50.0 GiB in 2 MiB pages" / "2 MiB hugetlb pages: 50.0 of 50.0 GiB" / "4 KiB pages: 50.0 GiB resident"
    std::string text() const;
};
PageStats numa_page_stats(const void* p, uint64_t bytes);

/// Pins a finished range for the GPU runtime (cudaHostRegister, from the CUDA side: this library has no CUDA); `how` says
/// what was done or why not.  `UnpinFn` undoes it and must run BEFORE the memory is unmapped.
using PinFn = std::function<bool(void* p, uint64_t bytes, std::string& how)>;
using UnpinFn = std::function<void(void* p, uint64_t bytes)>;

/// An anonymous mapping bound to one node, released on destruction.  Backed by 2 MiB hugetlb pages when the node's own
/// pool holds the whole buffer (as the primary arena does), else by THP, else 4 KiB pages (`numa_map_arena`).
class NumaBuffer {
public:
    NumaBuffer() = default;
    ~NumaBuffer();
    NumaBuffer(const NumaBuffer&) = delete;
    NumaBuffer& operator=(const NumaBuffer&) = delete;
    NumaBuffer(NumaBuffer&& o) noexcept { *this = static_cast<NumaBuffer&&>(o); }
    NumaBuffer& operator=(NumaBuffer&& o) noexcept;

    /// Maps `bytes` and binds them to `node` BEFORE any page is touched.  Returns false only when the MAPPING failed;
    /// a failed bind is reported through `bound()` / `note()` and the buffer is still usable (first touch from a thread
    /// pinned to the node - `numa_copy` - then places it).  `use_hugepages` is a request (hugetlb, else THP), `page_kind()`
    /// the outcome.  `cpus`: the node's CPUs, for the prefault threads of a hugetlb mapping.
    bool allocate(uint64_t bytes, int node, bool use_hugepages = true, const std::vector<int>& cpus = {});
    uint8_t* data() const { return base_; }
    uint64_t bytes() const { return bytes_; }
    int node() const { return node_; }
    bool bound() const { return bound_; }
    PageKind page_kind() const { return kind_; }
    bool hugepages() const { return kind_ == PageKind::Hugetlb; }
    /// What allocate() did, for the startup line ("2 MiB hugetlb pages; mbind(MPOL_BIND) to node 0" / "THP ...; mbind FAILED").
    const std::string& note() const { return note_; }
    /// Pin the buffer for the GPU runtime (call it AFTER the copy: it finds every page resident).  Pinned pages are
    /// unswappable like an mlock but are not charged to RLIMIT_MEMLOCK (8 MiB for a user by default), which is why a
    /// replica is pinned the way the primary is and `lock()` is only the fallback.  The buffer unpins before it unmaps.
    bool pin(const PinFn& pin, const UnpinFn& unpin);
    bool pinned() const { return pinned_; }
    const std::string& pin_note() const { return pin_note_; }
    /// mlock the whole buffer (after the copy: it makes every page resident).  Subject to RLIMIT_MEMLOCK; the caller applies
    /// the arena's lock policy (`strata::platform::arena_lock_allowed`, memory.hpp) and a refusal (ulimit -l) is a note,
    /// not an error.
    bool lock();
    uint64_t locked_bytes() const { return locked_; }
    const std::string& lock_note() const { return lock_note_; }
    void release();

private:
    uint8_t* base_ = nullptr;
    uint64_t bytes_ = 0, map_bytes_ = 0, locked_ = 0;
    int node_ = -1;
    bool bound_ = false, pinned_ = false;
    PageKind kind_ = PageKind::Small;
    UnpinFn unpin_;
    std::string note_, lock_note_, pin_note_;
};

/// `memcpy(dst, src, bytes)` on `threads` threads, each pinned to one of `cpus` (round-robin) for the duration, over
/// contiguous 2 MiB-aligned parts.  Pinned so the replica's pages are FIRST-TOUCHED by CPUs of the node they belong to
/// even where the bind failed (the second line of defence), and so the copy runs at the node's own write bandwidth.
/// Returns seconds, or a negative number when a thread could not be started.
double numa_copy(void* dst, const void* src, uint64_t bytes, const std::vector<int>& cpus, int threads);

// ================================ THE PLAN ================================

enum class NumaMode { Auto, Mirror, Off };
/// "auto" | "mirror" | "off" (also "on"/"1" = mirror, "0" = off); false for anything else.
bool parse_numa_mode(const std::string& text, NumaMode& out);
const char* numa_mode_name(NumaMode m);
/// The engine option `--numa`, overridden by the environment for A/B: `STRATA_NUMA_MIRROR=0` -> Off, `=1` -> Mirror,
/// anything else leaves `cli` alone.  `note` (when given) says when the environment decided.
NumaMode numa_mode_with_env(NumaMode cli, std::string* note = nullptr);
/// `STRATA_NUMA_HEADROOM_GIB` (default 6): what a node must keep free beyond a copy for everything else the process
/// and the OS put there.
uint64_t numa_headroom_bytes();

struct MirrorInputs {
    NumaMode mode = NumaMode::Auto;
    const NumaTopology* topo = nullptr;
    /// CPUs the process may use (numa_allowed_cpus()); empty = all of them.  A node with none gets no replica: nothing
    /// would read it.
    std::vector<int> allowed_cpus;
    uint64_t arena_bytes = 0;         ///< ONE copy, in bytes (the arena's capacity)
    bool full_ram = true;             ///< the experts are all resident in one RAM arena (not mmap / low-RAM / a RAM budget)
    std::string not_full_ram_why;     ///< when !full_ram: "--mmap-experts (the low-RAM mode)" ...
    bool shared_file = false;         ///< --shared-expert-arena: the backing file's placement is not this process's to choose
    uint64_t headroom = 6ull << 30;
    /// Whole-system room (MemAvailable, cgroup-limited) for all the copies together, 0 = unknown.
    uint64_t global_available = 0;
};

struct MirrorPlan {
    NumaMode mode = NumaMode::Auto;
    bool mirror = false;
    std::string why_not;              ///< when !mirror: ONE line, the reason (printed once at startup)
    int primary_node = -1;            ///< node id of the primary copy (the GPU's node): CUDA-registered, the only copy a GPU DMA sees
    /// Node ids that get a replica.  `Mirror`: every other node that has CPUs to read it.  `Auto`: ONE per physical
    /// package (socket) other than the GPU's - with sub-NUMA clustering (SNC / NPS) a socket is several nodes, and a copy
    /// per node would be 4 x 50 GB on a dual Xeon Gold; the node's package-mates read that copy.
    std::vector<int> replica_nodes;
    std::vector<int> cpu_nodes;       ///< node ids with at least one allowed CPU
    /// (node id, id of the node whose copy its workers read) for every node of `cpu_nodes`: itself when it holds a copy
    /// (the primary or a replica), else the copy of its package.  What `NumaReplicas::build` turns into `ArenaMirror::copy`.
    std::vector<std::pair<int, int>> reads;
    std::string layout;               ///< when mirroring: which copy serves which nodes ("socket 0: nodes 0,1 read the primary on node 0; ...")
    bool gpu_node_assumed = false;    ///< the GPU's node was unknown and node `primary_node` is a guess
    uint64_t copy_bytes = 0;
};
/// The decision, as a pure function of its inputs: Off -> no; no topology / one node / one node with CPUs -> no; not the
/// full-RAM mode -> no (one copy, and why); shared arena -> no; a node without room for its copy (`arena + headroom` of
/// usable memory, or just the headroom when its own free hugepage pool holds the whole copy) -> no (naming the node and
/// the numbers).  Otherwise: primary on the GPU's node, replicas as `MirrorPlan::replica_nodes` says.
MirrorPlan plan_arena_mirror(const MirrorInputs& in);

// ================================ THE MIRROR ================================

/// What the pool needs: the primary arena and, per node, the copy that node's workers read, indexed by NODE INDEX in
/// `NumaTopology::nodes` (`copy[primary_index] == primary`; a node that HOLDS a replica has it; a node served by its
/// socket's copy (sub-NUMA clustering) has the same pointer as its package-mate; null = no copy, read the primary).
struct ArenaMirror {
    const uint8_t* primary = nullptr;
    uint64_t bytes = 0;               ///< length of the primary range every copy covers
    std::vector<const uint8_t*> copy;
    int primary_index = -1;
    bool active() const { return primary != nullptr && bytes > 0 && copy.size() >= 2; }
};

struct MirrorOptions {
    /// A test topology: a node that does not exist, a bind that fails and pages that land elsewhere are REPORTED and the
    /// build goes on.  Real topology (false): a replica whose bind failed or whose pages are not on its node is dropped
    /// (it would cost memory and buy nothing) and `built` says so.
    bool faked = false;
    int threads = 8;                  ///< copy threads per replica
    int samples = 64;                 ///< pages sampled per placement check
    bool lock = true;                 ///< apply the arena lock policy (mlock) to a replica that could not be pinned
    bool hugepages = true;            ///< hugetlb when the node's pool holds the copy, else THP (false: 4 KiB pages)
    /// Pin each replica for the GPU runtime (cudaHostRegister: pinning only, the GPU never reads a replica) instead of
    /// mlock-ing it: pinned memory is not charged to RLIMIT_MEMLOCK, so a replica is no longer pageable on a stock user
    /// limit while the primary is pinned.  Set by the CUDA side (this library has no CUDA); a failure falls back to mlock.
    PinFn pin;
    UnpinFn unpin;
};

/// The replicas of one primary arena.  `build` allocates each replica bound to its node (hugetlb, else THP, else 4 KiB
/// pages), copies the primary into it from threads pinned to the node, verifies placement of a sample of pages of the
/// primary AND the replica, pins it (`MirrorOptions::pin`, else mlock); `log` gets one line per fact, the page size each
/// copy got among them.  After it, `mirror()` is what the pool takes.
class NumaReplicas {
public:
    NumaReplicas() = default;
    NumaReplicas(const NumaReplicas&) = delete;
    NumaReplicas& operator=(const NumaReplicas&) = delete;

    /// `primary` holds `bytes` valid bytes (the arena's loaded part) in a mapping of `capacity` >= bytes.  Replicas cover
    /// `capacity`.  Returns the number of replicas kept (0: the mirror is not active; `log` says why).
    int build(const uint8_t* primary, uint64_t bytes, uint64_t capacity, const MirrorPlan& plan, const NumaTopology& topo,
              const MirrorOptions& opt, std::vector<std::string>& log);
    const ArenaMirror& mirror() const { return mirror_; }
    uint64_t replica_bytes() const;
    size_t count() const { return bufs_.size(); }
    const NumaBuffer& buffer(size_t i) const { return bufs_[i]; }
    /// The primary's placement as sampled by `build`.
    const PlacementReport& primary_placement() const { return primary_placement_; }
    void release();

private:
    std::vector<NumaBuffer> bufs_;
    ArenaMirror mirror_;
    PlacementReport primary_placement_;
};

}  // namespace strata::platform
