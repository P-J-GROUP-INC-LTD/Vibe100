// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace strata::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

namespace {
std::mutex g_numa_mu;
PoolNuma g_numa;   // empty (inactive) unless the engine found a NUMA machine and is mirroring the arena

PoolNuma numa_snapshot() {
    std::lock_guard<std::mutex> lk(g_numa_mu);
    return g_numa;
}

// The node INDEX that owns logical CPU `cpu` in `pn`, or -1.
int numa_node_of(const PoolNuma& pn, int cpu) {
    for (size_t i = 0; i < pn.node_cpus.size(); ++i)
        if (std::find(pn.node_cpus[i].begin(), pn.node_cpus[i].end(), cpu) != pn.node_cpus[i].end()) return (int) i;
    return -1;
}
}  // namespace

void set_pool_numa(const PoolNuma& p) {
    std::lock_guard<std::mutex> lk(g_numa_mu);
    g_numa = p;
}

void clear_pool_numa() {
    std::lock_guard<std::mutex> lk(g_numa_mu);
    g_numa = PoolNuma{};
}

std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> cores;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back((int) i);
    } else {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            const char* p = buf.data();
            const char* end = p + len;
            while (p < end) {
                const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
                if (e->Relationship == RelationProcessorCore) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                    for (int bit = 0; bit < 64; ++bit)
                        if (g.Mask & (1ull << bit)) { cores.push_back((int) (g.Group * 64 + bit)); break; }
                }
                p += e->Size;
            }
        }
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  sysfs names each CPU's (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    std::vector<int> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) allowed.push_back(i);
    } else {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }
    std::vector<std::pair<long, long>> seen;
    for (int cpu : allowed) {
        const long pkg = topo(cpu, "physical_package_id"), core = topo(cpu, "core_id");
        if (pkg >= 0 && core >= 0) {
            const std::pair<long, long> key{pkg, core};
            if (std::find(seen.begin(), seen.end(), key) != seen.end()) continue;   // an SMT sibling
            seen.push_back(key);
        }
        cores.push_back(cpu);
    }
#endif
    // ---- NUMA (WP-F): with a mirrored arena the order is part of the design.  The FIRST core is the host loop's (the
    // session pins itself to `physical_cores(false)[0]` and the pool leaves `physical_cores(true)` without it), and it must
    // be on the GPU's node: the host drives the GPU's doorbell and reads the pinned buffers the card writes.  The rest then
    // ALTERNATE between the nodes, starting with a node other than the host's (the host also drains, so it already counts
    // for its own node), so that `--pool-workers N` spreads N workers over both memory controllers instead of filling the
    // first socket.  With no mirror the legacy order (ascending, one per physical core) is untouched.
    const PoolNuma pn = numa_snapshot();
    if (pn.active() && !cores.empty()) {
        std::vector<std::vector<int>> by(pn.node_cpus.size());
        std::vector<int> rest;
        for (int c : cores) {
            const int ni = numa_node_of(pn, c);
            (ni >= 0 ? by[(size_t) ni] : rest).push_back(c);
        }
        std::vector<int> ordered;
        int h = pn.host_node;
        if (h < 0 || h >= (int) by.size() || by[(size_t) h].empty()) h = -1;
        if (h >= 0) {
            ordered.push_back(by[(size_t) h].front());
            by[(size_t) h].erase(by[(size_t) h].begin());
        } else {
            h = 0;   // no usable preference: the legacy first core stays the host's
            ordered.push_back(cores.front());
            for (auto& b : by) b.erase(std::remove(b.begin(), b.end(), cores.front()), b.end());
            rest.erase(std::remove(rest.begin(), rest.end(), cores.front()), rest.end());
        }
        std::vector<size_t> next(by.size(), 0);
        for (bool took = true; took;) {
            took = false;
            for (size_t k = 1; k <= by.size(); ++k) {
                const size_t ni = ((size_t) h + k) % by.size();
                if (next[ni] < by[ni].size()) { ordered.push_back(by[ni][next[ni]++]); took = true; }
            }
        }
        ordered.insert(ordered.end(), rest.begin(), rest.end());
        cores.swap(ordered);
    }
    if (skip_first && !cores.empty()) cores.erase(cores.begin());
    return cores;
}

namespace {

void pin_this_thread(int core) {
    if (core < 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

}  // namespace

namespace {
// The calling thread runs on `core` for the life of the scope (core < 0: not at all), then gets its affinity back.  Used for the
// pool's own first touches (below): with a mirror the node a page lands on is the node of the thread that first writes it.
class ThreadPinScope {
public:
    explicit ThreadPinScope(int core) {
        if (core < 0) return;
#if defined(_WIN32)
        prev_ = pin_current_thread(core);
        active_ = prev_ > 0;
#else
        if (pthread_getaffinity_np(pthread_self(), sizeof prev_, &prev_) != 0) return;
        pin_this_thread(core);
        active_ = true;
#endif
    }
    ~ThreadPinScope() {
        if (!active_) return;
#if defined(_WIN32)
        restore_thread_affinity(prev_);
#else
        pthread_setaffinity_np(pthread_self(), sizeof prev_, &prev_);
#endif
    }
    ThreadPinScope(const ThreadPinScope&) = delete;
    ThreadPinScope& operator=(const ThreadPinScope&) = delete;

private:
    bool active_ = false;
#if defined(_WIN32)
    long long prev_ = -1;
#else
    cpu_set_t prev_;
#endif
};
}  // namespace

long long pin_current_thread(int core) {
    if (core < 0) return -1;
#if defined(_WIN32)
    // `SetThreadAffinityMask` RETURNS the previous mask, or 0 on failure - so 0 doubles as the error, which is
    // why the caller must not treat it as a restorable value.
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
    return prev == 0 ? -1 : (long long) prev;
#else
    cpu_set_t prev;
    CPU_ZERO(&prev);
    if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return -1;
    unsigned long mask = 0;
    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i)
        if (CPU_ISSET(i, &prev)) mask |= 1ul << i;
    pin_this_thread(core);
    return (long long) mask;
#endif
}

void restore_thread_affinity(long long previous) {
    if (previous <= 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) previous);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; ++i)
        if ((previous >> i) & 1) CPU_SET(i, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works) : host_works_(host_works) {
    if (const char* e = std::getenv("STRATA_POOL_SPIN_US"))   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
    const std::vector<int> cores = physical_cores(true);
    n_ = n_workers > 0 ? n_workers : (int) cores.size();
    if (n_ < 1) n_ = 1;
    // NUMA (WP-F): each worker's node, from the core it is pinned to.  Without a `PoolNuma` every entry is -1 and nothing
    // below ever reads it.
    int host_core = -1;   // the core the session pins the host loop to (the GPU's node) - only with a mirror
    {
        const PoolNuma pn = numa_snapshot();
        wnode_.assign((size_t) n_, -1);
        weff_.assign((size_t) n_, 0);
        wcpu_.assign((size_t) n_, -1);
        for (int i = 0; i < n_; ++i) {
            const int core = pin && i < (int) cores.size() ? cores[(size_t) i] : -1;
            wcpu_[(size_t) i] = core;
            if (pn.active() && core >= 0) wnode_[(size_t) i] = numa_node_of(pn, core);
        }
        if (pn.active()) {
            // the host loop pins itself to `physical_cores(false)[0]` (session.cpp), so ITS node is the node of that core -
            // which is `pn.host_node` unless that node had no core to give (then the legacy first core stays the host's)
            const std::vector<int> all = physical_cores(false);
            hnode_ = !all.empty() ? numa_node_of(pn, all.front()) : pn.host_node;
            if (!all.empty()) host_core = all.front();
        }
    }
    // The per-worker scratch and the split buffers (several MB, shared by all the threads of a split job) are first-touched by the
    // value-initialisation in `resize`, i.e. by THIS thread, which is not pinned yet and may be running on the socket the GPU is
    // NOT on.  Make them on the host's core, so with a mirror they land on the GPU's node like the session's other host buffers.
    // (The scope ends before any worker is created: a std::thread inherits its creator's affinity mask.)
    {
        ThreadPinScope on_host(host_core);
        scratch_.resize((size_t) n_);
        split_.resize((size_t) kMaxSplit);
        split_multi_.resize((size_t) kMaxSplitMulti);
    }
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    strata::core::diag_pool_fn().store(&diag_active_pool);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) cores.size() ? cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::set_mirror(const strata::platform::ArenaMirror& m) {
    // The workers read `mir_*` / `weff_` with plain loads after the epoch's acquire: write them only while every worker is
    // parked (the contract in pool.hpp).  `wait_parked` is the wait `run` starts with: the workers' own release (a worker
    // re-parks with `parked_.fetch_add`) is what orders their last reads before these writes, and the next `begin_batch`'s
    // release is what orders these writes before their next reads.
    wait_parked("before set_mirror");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    if (!m.active()) {
        mir_len_ = 0;
        mir_lo_ = 0;
        mir_nodes_ = 0;
        mir_copy_.clear();   // the copies are about to be unmapped by their owner: nothing may keep their addresses
        return;
    }
    mir_nodes_ = (int) m.copy.size();
    mir_primary_ = m.primary_index >= 0 && m.primary_index < mir_nodes_ ? m.primary_index : 0;
    mir_copy_.assign((size_t) mir_nodes_, nullptr);
    for (int i = 0; i < mir_nodes_; ++i) mir_copy_[(size_t) i] = m.copy[(size_t) i] != nullptr ? m.copy[(size_t) i] : m.primary;
    // A thread whose node holds no copy (or is unknown) reads the primary: the identity translation.
    auto eff = [&](int node) { return node >= 0 && node < mir_nodes_ ? node : mir_primary_; };
    for (int i = 0; i < n_; ++i) weff_[(size_t) i] = eff(wnode_[(size_t) i]);
    heff_ = eff(hnode_);
    mir_lo_ = (uintptr_t) m.primary;
    mir_len_ = m.bytes;   // last: `xl` is the identity until the whole description is in place
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "strata: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267: the GPU may be spinning on this layer's flag
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        // The copy this thread reads (WP-F): its node's, or the primary when no mirror is set.  Read AFTER the claim: a claim
        // that succeeds belongs to the current batch, which `set_mirror` (host, workers parked) ordered its writes before; a
        // worker that woke late for an old epoch claims nothing and never looks at what a `set_mirror` may be writing.
        const int node = id >= 0 ? weff_[(size_t) id] : heff_;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(xl(j.blob, node), *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(xl(jobs_[e].blob, node), *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(xl(jobs_[e].blob, node), split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                // ONE translation per job: every address below (gate, up at `up_off`, down at `down_off`) is an offset
                // from this base, so a job never reads half from one node's copy and half from another's.
                const uint8_t* mb = xl(mjobs_[e].blob, node);
                if (mode_ == 5 && nfmt_->gu_type == 42) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mb, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mb + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mb, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (nfmt_->d_type == 42) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mb + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mb, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                const uint8_t* mb = xl(mjobs_[e].blob, node);   // see mode 5: one translation per job
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mb, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mb, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, host_scratch_, e);
    wait_done(n_tasks);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    mrows_ = (int64_t) n * FF;
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        const int threads = n_ + (host_works_ ? 1 : 0);
        mtasks_ = 3 * threads;
        mrows_ = (int64_t) nb * FF;
        const auto a = std::chrono::steady_clock::now();
        run_phase(5, mtasks_);
        const auto b = std::chrono::steady_clock::now();
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (f.d_type == 42) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        mrows_ = (int64_t) nb * H;
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(xl(jobs[i].blob, heff_), *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(xl(j.blob, heff_), *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
