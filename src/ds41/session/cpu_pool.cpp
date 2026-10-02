// src/ds41/session/cpu_pool.cpp - DS1-E: the CPU expert pool (include/strata/ds41/session/cpu_pool.hpp: read its header comment first).
#include "strata/ds41/session/cpu_pool.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <set>
#include <stdexcept>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace strata::ds41::session {

namespace {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
}

bool pin_this_thread(int cpu) {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
#else
    (void) cpu;
    return false;
#endif
}

/// A sense-reversing barrier for the n workers of one socket: spin a little, then yield (a pinned worker whose partner shares its CPU must let it run).
struct Barrier {
    std::atomic<int> count{0};
    std::atomic<uint32_t> sense{0};
    int n = 1;
    void arrive_and_wait() {
        if (n <= 1) return;
        const uint32_t s = sense.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            sense.fetch_add(1, std::memory_order_release);
            return;
        }
        for (unsigned spins = 0; sense.load(std::memory_order_acquire) == s; ++spins) {
            if (spins < 2000) cpu_relax();
            else std::this_thread::yield();
        }
    }
};

template <class T> struct AlignedFree {
    void operator()(T* p) const { std::free(p); }
};
template <class T> std::unique_ptr<T[], AlignedFree<T>> aligned_array(size_t n) {
    const size_t bytes = std::max<size_t>(64, (n * sizeof(T) + 63) / 64 * 64);
    void* p = std::aligned_alloc(64, bytes);
    if (!p) throw std::bad_alloc();
    return std::unique_ptr<T[], AlignedFree<T>>(static_cast<T*>(p));
}

}  // namespace

// ================================================================================================ the plan
CpuPoolOptions plan_cpu_pool(const platform::NumaTopology& topo, bool numa, int threads_per_socket, const std::vector<int>& allowed) {
    CpuPoolOptions o;
    model::ArenaNodes an;
    if (numa) an = model::choose_arena_nodes(topo);
    else an.note = "NUMA binding switched off";
    const std::set<int> ok(allowed.begin(), allowed.end());
    auto filtered = [&](const std::vector<int>& v) {
        std::vector<int> r;
        for (int c : v)
            if (ok.empty() || ok.count(c)) r.push_back(c);
        return r;
    };
    bool bound = an.bind;
    if (bound) {
        for (int h = 0; h < 2; ++h) o.cpus[h] = filtered(an.cpus[h]);
        if (o.cpus[0].empty() || o.cpus[1].empty()) {
            an.note += "; but this process may not run on every node's CPUs: the workers are not pinned";
            o.cpus[0].clear();
            o.cpus[1].clear();
            bound = false;
        }
    }
    int ncpu = (int) (ok.empty() ? std::max(1u, std::thread::hardware_concurrency()) : ok.size());
    if (threads_per_socket > 0) o.threads_per_group = threads_per_socket;
    else if (bound) o.threads_per_group = std::max(1, (int) std::min(o.cpus[0].size(), o.cpus[1].size()) - 1);      // one CPU per node stays free: the thread that drives the GPU
    else o.threads_per_group = std::max(1, ncpu / 2);
    o.pin = bound;
    o.note = std::string("CPU expert pool: ") + (bound ? "one worker group per socket, pinned" : "two worker groups, not pinned") + " (" + an.note + "), " +
             std::to_string(o.threads_per_group) + " threads per group";
    return o;
}

// ================================================================================================ the pool
struct CpuExpertPool::Impl {
    struct Group {
        int expert = 0;
        int count = 0;                      // tokens (misses) in the group
        int pos = 0;                        // first y row of the group (in both halves' partial buffers)
        const cpu::ActQ* x = nullptr;       // count quantised activations, contiguous
        float w[cpu::kMaxTokens] = {};
    };
    struct HalfState {
        std::unique_ptr<cpu::ExpertScratch[]> scratch;               // one per group
        std::unique_ptr<float[], AlignedFree<float>> y;              // [max_misses][hidden]
        Barrier barrier;
    };

    const model::ExpertArena& arena;
    model::ExpertDims d;
    CpuPoolOptions opt;
    cpu::Isa isa;
    int tpg = 1;
    int max_misses = 1;
    int hidden = 0, half_ff = 0;

    // the current job
    int layer = 0, n_groups = 0, n_miss = 0, job_T = 0;
    std::vector<Group> groups;
    std::vector<int> pos_of_miss;
    std::vector<cpu::ActQ> gx;                                        // copies of the activations of multi-token groups
    double t_start = 0;
    bool active = false;
    std::chrono::steady_clock::time_point t0;

    HalfState half[2];
    std::vector<std::thread> threads;
    std::atomic<uint32_t> gen{0};
    std::atomic<bool> stop{false};
    std::atomic<int> done{0};
    std::atomic<int> pinned{0};
    std::atomic<int> failed{0};

    Impl(const model::ExpertArena& a, const CpuPoolOptions& o) : arena(a), d(a.dims()), opt(o) {
        if (!arena.built()) throw std::invalid_argument("CpuExpertPool: the expert arena is not built");
        if (opt.max_tokens < 1 || opt.max_tokens > cpu::kMaxTokens) throw std::invalid_argument("CpuExpertPool: max_tokens must be 1..8");
        if (opt.top_k < 1) throw std::invalid_argument("CpuExpertPool: top_k < 1");
        isa = cpu::resolve_isa(opt.isa);
        hidden = d.hidden;
        half_ff = d.half_ff();
        tpg = std::max(1, opt.threads_per_group > 0 ? opt.threads_per_group
                                                    : (!opt.cpus[0].empty() && !opt.cpus[1].empty() ? (int) std::min(opt.cpus[0].size(), opt.cpus[1].size()) : 1));
        max_misses = opt.max_tokens * opt.top_k;
        groups.resize((size_t) max_misses);
        pos_of_miss.resize((size_t) max_misses);
        gx.resize((size_t) max_misses);
        for (int h = 0; h < 2; ++h) {
            half[h].scratch.reset(new cpu::ExpertScratch[(size_t) max_misses]);
            half[h].y = aligned_array<float>((size_t) max_misses * (size_t) hidden);
            half[h].barrier.n = tpg;
        }
        threads.reserve((size_t) 2 * (size_t) tpg);
        for (int h = 0; h < 2; ++h)
            for (int i = 0; i < tpg; ++i) threads.emplace_back([this, h, i] { worker(h, i); });
    }

    ~Impl() {
        stop.store(true, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_release);
        gen.notify_all();
        for (std::thread& t : threads) t.join();
    }

    cpu::ExpertView view(int h, int expert) const {
        const uint8_t* base = arena.half(h, layer, expert);
        cpu::ExpertView v;
        v.gate = base;
        v.up = base + d.half_gate_bytes();
        v.down = base + 2 * d.half_gate_bytes();
        v.down_row_stride = (size_t) d.half_down_row_bytes();
        v.hidden = hidden;
        v.ff = half_ff;
        return v;
    }

    uint32_t wait_job(uint32_t seen) {
        for (int i = 0; i < opt.spin_iterations; ++i) {
            const uint32_t g = gen.load(std::memory_order_acquire);
            if (g != seen) return g;
            cpu_relax();
        }
        for (;;) {
            const uint32_t g = gen.load(std::memory_order_acquire);
            if (g != seen) return g;
            gen.wait(seen, std::memory_order_acquire);
        }
    }

    void worker(int h, int i) {
        if (opt.pin && !opt.cpus[h].empty() && pin_this_thread(opt.cpus[h][(size_t) i % opt.cpus[h].size()])) pinned.fetch_add(1);
        uint32_t seen = 0;
        for (;;) {
            seen = wait_job(seen);
            if (stop.load(std::memory_order_acquire)) return;
            try {
                run_job(h, i);
            } catch (...) {
                failed.fetch_add(1);
            }
            if (done.fetch_add(1, std::memory_order_acq_rel) + 1 == 2 * tpg) done.notify_one();       // the last worker wakes the thread that waits in wait()
        }
    }

    /// The work of a phase is `per_group` units for each of the n_groups experts, cut into `tpg` contiguous parts of the FLATTENED list (group-major): worker i gets the units
    /// [i * total / tpg, (i + 1) * total / tpg), as at most one range per group.  A worker's share is balanced to one unit however the units divide (36 chunks over 23 workers
    /// would leave some with 1 and some with 2; 216 chunks of six experts over 23 give 9 or 10), and which worker computes a unit cannot change the unit's bits.
    template <class Fn> void for_my_units(int per_group, int i, Fn&& fn) const {
        const long long total = (long long) per_group * n_groups;
        const long long f0 = total * i / tpg, f1 = total * (i + 1) / tpg;
        for (int g = 0; g < n_groups; ++g) {
            const long long base = (long long) g * per_group;
            const long long lo = std::max(f0, base) - base, hi = std::min(f1, base + per_group) - base;
            if (hi > lo) fn(g, (int) lo, (int) hi);
        }
    }

    static constexpr int kRowUnit = 16;                                  // output rows per phase-2 unit: one 64-byte line of y

    void run_job(int h, int i) {
        HalfState& hs = half[h];
        // phase 1: the 32-row chunks of the half's intermediate (gate / up rows, SwiGLU, the int8 image of h) of this worker's share
        const int chunks = half_ff / cpu::kChunkRows;
        for_my_units(chunks, i, [&](int g, int c0, int c1) {
            const Group& gr = groups[(size_t) g];
            cpu::expert_gate_up(isa, view(h, gr.expert), gr.x, gr.count, gr.w, hs.scratch[(size_t) g], c0, c1);
        });
        hs.barrier.arrive_and_wait();                                   // every chunk of h of every group is needed by every down row
        // phase 2: this worker's 16-row units of every group's y.  Each unit is zeroed and then ADDED to by the same worker (static ownership: no other thread, and not the other
        // socket, ever touches these elements), so no zeroing pass and no cross-thread hand-off is needed
        const int units = (hidden + kRowUnit - 1) / kRowUnit;
        for_my_units(units, i, [&](int g, int u0, int u1) {
            const Group& gr = groups[(size_t) g];
            const int r0 = u0 * kRowUnit, r1 = std::min(hidden, u1 * kRowUnit);
            for (int j = 0; j < gr.count; ++j) std::memset(hs.y.get() + (size_t) (gr.pos + j) * hidden + r0, 0, (size_t) (r1 - r0) * sizeof(float));
            cpu::expert_down(isa, view(h, gr.expert), hs.scratch[(size_t) g], gr.count, hs.y.get() + (size_t) gr.pos * hidden, r0, r1);
        });
    }
};

CpuExpertPool::CpuExpertPool(const model::ExpertArena& arena, const CpuPoolOptions& opt) : impl_(new Impl(arena, opt)) {
    tpg_ = impl_->tpg;
    isa_ = impl_->isa;
    // the workers pin themselves as they start: give them a moment so that pinned_workers() is meaningful right after construction
    for (int spin = 0; spin < 2000 && impl_->pinned.load() < (impl_->opt.pin && !impl_->opt.cpus[0].empty() && !impl_->opt.cpus[1].empty() ? 2 * tpg_ : 0); ++spin)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    pinned_ = impl_->pinned.load();
}

CpuExpertPool::~CpuExpertPool() = default;

std::string CpuExpertPool::describe() const {
    char b[512];
    std::snprintf(b, sizeof b, "CPU expert pool: 2 groups x %d threads (half 0, half 1), %d pinned, %s kernels", tpg_, pinned_, cpu::isa_name(isa_));
    return b;
}

void CpuExpertPool::start(int layer, const cpu::ActQ* x, int T, const CpuMiss* miss, int n) {
    Impl& m = *impl_;
    if (m.active) throw std::logic_error("CpuExpertPool::start: a job is already running");
    if (layer < 0 || layer >= m.d.n_layer) throw std::invalid_argument("CpuExpertPool::start: layer out of range");
    if (n < 0 || n > m.max_misses) throw std::invalid_argument("CpuExpertPool::start: more misses than max_tokens * top_k");
    if (T < 1 || T > cpu::kMaxTokens) throw std::invalid_argument("CpuExpertPool::start: T must be 1..8");
    m.layer = layer;
    m.job_T = T;
    m.n_miss = n;
    m.n_groups = 0;
    int gx_used = 0;
    // group the misses by expert, in order of first appearance; a group's tokens in the order they come
    std::vector<std::vector<int>> members;
    members.reserve((size_t) n);
    for (int i = 0; i < n; ++i) {
        if (miss[i].token < 0 || miss[i].token >= T) throw std::invalid_argument("CpuExpertPool::start: a miss names a token outside the window");
        if (miss[i].expert < 0 || miss[i].expert >= m.d.n_expert) throw std::invalid_argument("CpuExpertPool::start: a miss names an expert outside the model");
        int g = -1;
        for (int q = 0; q < m.n_groups; ++q)
            if (m.groups[(size_t) q].expert == miss[i].expert && members[(size_t) q].size() < (size_t) cpu::kMaxTokens) {
                g = q;
                break;
            }
        if (g < 0) {
            g = m.n_groups++;
            m.groups[(size_t) g].expert = miss[i].expert;
            members.emplace_back();
        }
        members[(size_t) g].push_back(i);
    }
    int pos = 0;
    for (int g = 0; g < m.n_groups; ++g) {
        Impl::Group& gr = m.groups[(size_t) g];
        const std::vector<int>& mem = members[(size_t) g];
        gr.count = (int) mem.size();
        gr.pos = pos;
        for (int j = 0; j < gr.count; ++j) {
            m.pos_of_miss[(size_t) mem[(size_t) j]] = pos + j;
            gr.w[j] = miss[mem[(size_t) j]].weight;
        }
        if (gr.count == 1) {
            gr.x = x + miss[mem[0]].token;
        } else {
            gr.x = m.gx.data() + gx_used;
            for (int j = 0; j < gr.count; ++j) m.gx[(size_t) (gx_used + j)] = x[miss[mem[(size_t) j]].token];
            gx_used += gr.count;
        }
        pos += gr.count;
    }
    m.active = true;
    m.t0 = std::chrono::steady_clock::now();
    if (n == 0) return;                                                // nothing to run: wait() returns at once
    m.done.store(0, std::memory_order_relaxed);
    m.gen.fetch_add(1, std::memory_order_release);
    m.gen.notify_all();
}

int CpuExpertPool::wait(float* out) {
    Impl& m = *impl_;
    if (!m.active) throw std::logic_error("CpuExpertPool::wait: no job");
    const int nworkers = 2 * m.tpg;
    if (m.n_miss > 0) {
        // spin briefly (a layer's CPU work is ~1 ms on the real model, a wake-up ~50 us), then BLOCK: this thread shares a CPU with a pinned worker when the pool uses every CPU of a
        // node, and a spinning waiter would steal cycles from the one worker everyone then waits for at the barrier
        for (unsigned spins = 0;; ++spins) {
            const int cur = m.done.load(std::memory_order_acquire);
            if (cur == nworkers) break;
            if (spins < 2000) cpu_relax();
            else m.done.wait(cur, std::memory_order_acquire);
        }
        if (m.failed.load() != 0) throw std::runtime_error("CpuExpertPool: a worker threw an exception");
        // the two sockets' partials, added once: h0 + h1
        const float* y0 = m.half[0].y.get();
        const float* y1 = m.half[1].y.get();
        for (int i = 0; i < m.n_miss; ++i) {
            const size_t row = (size_t) m.pos_of_miss[(size_t) i] * (size_t) m.hidden;
            float* o = out + (size_t) i * (size_t) m.hidden;
            for (int c = 0; c < m.hidden; ++c) o[c] = y0[row + (size_t) c] + y1[row + (size_t) c];
        }
    }
    m.active = false;
    stats_.jobs += 1;
    stats_.experts += (uint64_t) m.n_groups;
    stats_.entries += (uint64_t) m.n_miss;
    stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - m.t0).count();
    return m.n_groups;
}

}  // namespace strata::ds41::session
