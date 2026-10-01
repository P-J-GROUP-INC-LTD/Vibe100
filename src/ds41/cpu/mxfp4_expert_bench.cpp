// src/ds41/cpu/mxfp4_expert_bench.cpp - DS-C: how fast do the MXFP4 expert kernels consume expert weights, against what the
// memory system can deliver?  A program, not a test.
//
//   ds41_cpu_mxfp4_bench [--halves N] [--seconds S] [--threads a,b,..] [--tokens a,b,..] [--sweep-pf] [--scalar]
//                        [--box-gbps R] [--no-l2] [--no-thp] [--no-pin] [--static] [--reps N]
//
// What is measured: the whole half-expert pipeline (phase 1: gate/up + SwiGLU + h quantisation, barrier, phase 2: down) on the
// real geometry (one CPU half = 9,400,320 B), cycling through N different halves so that the weights come from DRAM, not the
// cache (N x 9.4 MB should be several times the last-level cache; the default is 40 = 376 MB).  GB/s is expert-weight bytes
// consumed per second; the wall time includes the barrier, the SwiGLU and the activation quantisation.  Against it, the
// CEILING: the same threads reading the same arena with plain 512-bit loads and doing nothing else - efficiency is the ratio.
//
// THE NUMBERS ARE THIS MACHINE'S.  The target box is a dual Xeon Gold 6226 with 12 cores and ~6 DDR4-2666 channels per socket;
// this container is a shared 4-vCPU VM.  The tok/s lines convert the VM's GB/s into "what the CPU experts alone would allow"
// with the arithmetic of docs/deepseek/PLAN.md (240 expert activations per token x miss rate x 18.8 MB per expert); they are
// illustrations of the formula with these measurements, not predictions for the Xeon - the projection lines say what they assume.
#include "strata/ds41/cpu/mxfp4_expert.hpp"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#endif

using namespace strata::ds41;
using namespace strata::ds41::cpu;

// (what the library's software prefetch targets; a build-time choice, see DS41_PF_HINT in src/ds41/cpu/mxfp4_internal.hpp)
#ifndef DS41_BENCH_HINT_NAME
#define DS41_BENCH_HINT_NAME "T0 unless the library was built with -DDS41_PF_HINT=..."
#endif

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// The CPUs this process may use, captured once at start-up: a thread inherits its creator's affinity, so asking again after
// the main thread has pinned itself would see one CPU.
std::vector<int> g_cpus;
int g_reps = 3;          // measurement windows per configuration (the best one is reported)
bool g_pin = true;
bool g_static = false;   // static row ranges per thread instead of dynamic claiming
void capture_cpus() {
#if defined(__linux__)
    cpu_set_t avail;
    if (sched_getaffinity(0, sizeof avail, &avail) == 0)
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &avail)) g_cpus.push_back(i);
#endif
}
void pin_to_cpu(int idx) {
#if defined(__linux__)
    if (g_cpus.empty() || !g_pin) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(g_cpus[(size_t) idx % g_cpus.size()], &set);
    sched_setaffinity(0, sizeof set, &set);
#else
    (void) idx;
#endif
}

// A sense-reversing spin barrier (the kernel threads are CPU-bound and pinned; a futex wake per phase would be measured too).
struct SpinBarrier {
    explicit SpinBarrier(int n) : n_(n) {}
    void wait() {
        const int s = sense_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            sense_.store(s ^ 1, std::memory_order_release);
        } else {
            unsigned spins = 0;
            while (sense_.load(std::memory_order_acquire) == s) {
                _mm_pause();
                if (++spins > 256) { std::this_thread::yield(); spins = 0; }   // a preempted peer must not be starved of its vCPU
            }
        }
    }
    int n_;
    std::atomic<int> count_{0}, sense_{0};
};

// ---- the arena of halves -----------------------------------------------------------------------------------------------
struct Arena {
    uint8_t* base = nullptr;
    size_t bytes = 0;
    int halves = 0;
    const uint8_t* half(int i) const { return base + (size_t) i * kHalfBytes; }
    explicit Arena(int n, bool thp) : halves(n), bytes((size_t) n * kHalfBytes) {
        const size_t align = 2u << 20;
        const size_t rounded = (bytes + align - 1) / align * align;
        base = (uint8_t*) std::aligned_alloc(align, rounded);
        if (!base) { std::fprintf(stderr, "cannot allocate %zu MB\n", rounded >> 20); std::exit(1); }
#if defined(__linux__)
        if (thp) madvise(base, rounded, MADV_HUGEPAGE);
#else
        (void) thp;
#endif
        // random qs bytes, realistic exponents (so no denormals or overflow slow the arithmetic down)
        const int nthr = (int) std::max(1u, std::thread::hardware_concurrency());
        std::vector<std::thread> th;
        for (int t = 0; t < nthr; ++t)
            th.emplace_back([&, t] {
                std::mt19937_64 g(1234 + t);
                const size_t blocks = bytes / kBlockBytes;      // kHalfBytes is a multiple of 17: blocks never straddle halves
                const size_t b0 = blocks * t / nthr, b1 = blocks * (t + 1) / nthr;
                uint64_t buf[3];
                for (size_t b = b0; b < b1; ++b) {
                    buf[0] = g(); buf[1] = g(); buf[2] = g();
                    uint8_t* p = base + b * kBlockBytes;
                    std::memcpy(p, buf, kBlockBytes);
                    p[0] = (uint8_t) (116 + (buf[2] >> 60) % 8);
                }
            });
        for (auto& t : th) t.join();
    }
    ~Arena() { std::free(base); }
};

// ---- the read-bandwidth ceiling --------------------------------------------------------------------------------------------
double read_ceiling_gbps(const Arena& a, int nthr, double min_seconds, int pf) {
    std::atomic<uint64_t> sink{0};
    SpinBarrier bar(nthr);
    std::atomic<int> target{0};
    double best = 0;
    const size_t lines = a.bytes / 64;
    auto body = [&](int tid) {
        pin_to_cpu(tid);
        const size_t l0 = lines * tid / nthr, l1 = lines * (tid + 1) / nthr;
        __m512i acc[8];
        for (int k = 0; k < 8; ++k) acc[k] = _mm512_setzero_si512();
        auto pass = [&] {
            const __m512i* p = (const __m512i*) a.base + l0;
            const size_t n = l1 - l0;
            for (size_t i = 0; i + 8 <= n; i += 8) {
                if (pf) _mm_prefetch((const char*) (p + i) + pf, _MM_HINT_T1), _mm_prefetch((const char*) (p + i) + pf + 256, _MM_HINT_T1);
                for (int k = 0; k < 8; ++k) acc[k] = _mm512_add_epi64(acc[k], _mm512_load_si512((const void*) (p + i + k)));
            }
        };
        for (int rep = 0; rep < g_reps; ++rep) {
            bar.wait();
            auto t0 = Clock::now();
            pass();                                   // one pass calibrates how many fit in `min_seconds`
            bar.wait();
            const double one = seconds_since(t0);
            if (tid == 0) target.store(std::max(1, (int) std::ceil(min_seconds / std::max(one, 1e-6))));
            bar.wait();
            t0 = Clock::now();
            const int n = target.load();
            for (int i = 0; i < n; ++i) pass();
            bar.wait();
            if (tid == 0) best = std::max(best, (double) n * (double) a.bytes / seconds_since(t0) / 1e9);
        }
        uint64_t s = 0;
        alignas(64) uint64_t tmp[8];
        for (int k = 0; k < 8; ++k) { _mm512_store_si512((void*) tmp, acc[k]); for (int j = 0; j < 8; ++j) s += tmp[j]; }
        sink.fetch_add(s);
    };
    std::vector<std::thread> th;
    for (int t = 1; t < nthr; ++t) th.emplace_back(body, t);
    body(0);
    for (auto& t : th) t.join();
    if (sink.load() == 1) std::printf(" ");   // keep the loads alive
    return best;
}

// ---- the kernel -------------------------------------------------------------------------------------------------------------
struct Result {
    double gbps = 0, us_per_half = 0;
};

Result run_kernel(const Arena& a, Isa isa, int nthr, int T, double min_seconds) {
    static ActQ xq[kMaxTokens];
    static ExpertScratch scratch[2];
    static float y[kMaxTokens * kHidden];
    {
        std::mt19937 g(7);
        std::normal_distribution<float> nd;
        std::vector<float> x((size_t) T * kHidden);
        for (auto& v : x) v = nd(g);
        quantize_acts(x.data(), kHidden, T, xq, isa);
    }
    float w[kMaxTokens];
    for (int t = 0; t < kMaxTokens; ++t) w[t] = 0.5f + 0.1f * t;
    std::memset(y, 0, sizeof y);
    SpinBarrier bar(nthr), sync(nthr);
    std::atomic<long> iters_done{0};
    Result best;
    std::vector<double> rep_gbps((size_t) g_reps, 0.0), rep_us((size_t) g_reps, 0.0);
    std::atomic<int> stop_at{-1};
    struct Ctr { alignas(64) std::atomic<int> g{0}, d{0}; };
    Ctr ctr[4];
    auto body = [&](int tid) {
        pin_to_cpu(tid);
        long it = 0;       // the half counter persists across windows: the ring of work counters below follows it
        for (int rep = 0; rep < g_reps; ++rep) {
            // warm-up: touch the code and the activation, 2 halves
            auto one = [&](long i) {
                const ExpertView v = view_cpu_half(a.half((int) (i % a.halves)));
                ExpertScratch& s = scratch[i & 1];
                if (nthr == 1 || g_static) {
                    int c0, c1, r0, r1;
                    split_range(v.chunks(), nthr, tid, 1, c0, c1);
                    expert_gate_up(isa, v, xq, T, w, s, c0, c1);
                    if (nthr > 1) bar.wait();
                    split_range(v.hidden, nthr, tid, 4, r0, r1);
                    expert_down(isa, v, s, T, y, r0, r1);
                } else {
                    // dynamic: claim one 32-row chunk (phase 1) / 64 output rows (phase 2) at a time from this half's counters,
                    // so a preempted thread delays only the unit it holds.  The counter slots of half i + 2 are reset by thread 0
                    // right after this half's barrier (nobody can be that far ahead: they would have to pass half i + 1's barrier).
                    Ctr& cc = ctr[i & 3];
                    for (;;) {
                        const int c = cc.g.fetch_add(1, std::memory_order_relaxed);
                        if (c >= v.chunks()) break;
                        expert_gate_up(isa, v, xq, T, w, s, c, c + 1);
                    }
                    bar.wait();
                    if (tid == 0) { ctr[(i + 2) & 3].g.store(0, std::memory_order_relaxed); ctr[(i + 2) & 3].d.store(0, std::memory_order_relaxed); }
                    for (;;) {
                        const int r0 = cc.d.fetch_add(1, std::memory_order_relaxed) * 64;
                        if (r0 >= v.hidden) break;
                        expert_down(isa, v, s, T, y, r0, std::min(v.hidden, r0 + 64));
                    }
                }
            };
            for (int i = 0; i < 2; ++i) one(it++);
            sync.wait();
            if (tid == 0) stop_at.store(-1);
            sync.wait();
            const auto t0 = Clock::now();
            long done = 0;
            for (;;) {
                // check the clock every 8 halves, on thread 0 only; the decision reaches the others through the barrier
                for (int k = 0; k < 8; ++k) one(it++), ++done;
                if (tid == 0 && seconds_since(t0) >= min_seconds) stop_at.store((int) done);
                bar.wait();
                if (stop_at.load() >= 0) break;
            }
            sync.wait();
            if (tid == 0) {
                const double secs = seconds_since(t0);
                rep_gbps[rep] = (double) done * (double) kHalfBytes / secs / 1e9;
                rep_us[rep] = secs / (double) done * 1e6;
            }
            sync.wait();
            if (tid == 0) for (auto& v : y) if (!(std::fabs(v) < 1e20f)) v = 0;   // keep y finite
            sync.wait();
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < nthr; ++t) th.emplace_back(body, t);
    body(0);
    for (auto& t : th) t.join();
    for (int rep = 0; rep < g_reps; ++rep)
        if (rep_gbps[rep] > best.gbps) { best.gbps = rep_gbps[rep]; best.us_per_half = rep_us[rep]; }
    return best;
}

// Cache-resident kernel throughput: the compute ceiling of one core (the rows of a 160-block gate matrix that fit in L2).
double l2_resident_gbps(Isa isa, int T, double min_seconds) {
    static ActQ xq[kMaxTokens];
    const int rows = 64;
    std::vector<uint8_t> w((size_t) rows * kGateRowBytes);
    std::mt19937_64 g(5);
    for (auto& b : w) b = (uint8_t) g();
    for (size_t i = 0; i < w.size(); i += kBlockBytes) w[i] = (uint8_t) (118 + (g() & 3));
    {
        std::normal_distribution<float> nd;
        std::vector<float> x((size_t) T * kHidden);
        for (auto& v : x) v = nd(g);
        quantize_acts(x.data(), kHidden, T, xq, isa);
    }
    pin_to_cpu(0);
    std::vector<float> out((size_t) T * rows);
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        long n = 0;
        while (seconds_since(t0) < min_seconds / 2) {
            for (int k = 0; k < 64; ++k) mxfp4_dot_rows(isa, w.data(), kGateRowBytes, kGateRowBlocks, rows, xq, T, out.data(), (size_t) rows, false), ++n;
        }
        best = std::max(best, (double) n * (double) w.size() / seconds_since(t0) / 1e9);
    }
    return best;
}

std::vector<int> parse_list(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    capture_cpus();
    int nhalves = 40;
    double secs = 0.7, box_gbps = 100.0;
    std::vector<int> threads = {1, 4}, tokens = {1, 4};
    bool sweep = false, scalar = false, l2 = true, thp = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--halves" && i + 1 < argc) nhalves = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) secs = std::atof(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = parse_list(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) tokens = parse_list(argv[++i]);
        else if (a == "--box-gbps" && i + 1 < argc) box_gbps = std::atof(argv[++i]);
        else if (a == "--sweep-pf") sweep = true;
        else if (a == "--scalar") scalar = true;
        else if (a == "--no-l2") l2 = false;
        else if (a == "--no-thp") thp = false;
        else if (a == "--no-pin") g_pin = false;
        else if (a == "--static") g_static = true;
        else if (a == "--reps" && i + 1 < argc) g_reps = std::max(1, std::atoi(argv[++i]));
        else {
            std::printf("usage: %s [--halves N] [--seconds S] [--threads a,b] [--tokens a,b] [--sweep-pf] [--scalar] [--box-gbps R] [--no-l2] [--no-thp] [--no-pin] [--static] [--reps N]\n", argv[0]);
            return 2;
        }
    }
    std::vector<Isa> isas;
    if (isa_supported(Isa::kAvx2)) isas.push_back(Isa::kAvx2);
    if (isa_supported(Isa::kAvx512)) isas.push_back(Isa::kAvx512);
    if (scalar) isas.insert(isas.begin(), Isa::kScalar);
    if (!isa_supported(Isa::kAvx512)) { std::printf("this CPU has no AVX-512 F/BW/VL/DQ: the read-bandwidth probe needs it\n"); return 1; }

    std::printf("ds41 MXFP4 CPU expert benchmark - THIS MACHINE (%u hardware threads), not the target Xeon\n", std::thread::hardware_concurrency());
    std::printf("arena: %d halves x %.2f MB = %.0f MB (cycled, so weights come from DRAM); prefetch distance %d B; %.1f s per measurement window, best of %d windows\n",
                nhalves, kHalfBytes / 1e6, nhalves * kHalfBytes / 1e6, prefetch_bytes(), secs, g_reps);
    Arena arena(nhalves, thp);

    std::printf("\nread-bandwidth ceiling (zmm loads over the same arena, nothing else; best of no / 4 KB / 12 KB prefetch ahead):\n");
    std::vector<double> ceiling(threads.size());
    for (size_t i = 0; i < threads.size(); ++i) {
        // the best of plain loads and loads with a software prefetch ahead (a plain loop leaves bandwidth on the table)
        double best = 0;
        int best_pf = 0;
        for (int pf : {0, 4096, 12288}) {
            const double g = read_ceiling_gbps(arena, threads[i], secs, pf);
            if (g > best) { best = g; best_pf = pf; }
        }
        ceiling[i] = best;
        std::printf("  %d thread%s: %7.2f GB/s%s\n", threads[i], threads[i] == 1 ? " " : "s", ceiling[i], best_pf ? "  (with software prefetch)" : "");
    }

    struct Row { Isa isa; int thr, T; Result r; double ceil; };
    std::vector<Row> rows;
    std::printf("\nhalf-expert (gate/up + SwiGLU + quantise + down), GB/s of expert weights consumed:\n");
    std::printf("  %-7s %7s %3s | %9s %12s | %8s\n", "isa", "threads", "T", "GB/s", "us / half", "of read");
    for (Isa isa : isas)
        for (size_t ti = 0; ti < threads.size(); ++ti)
            for (int T : tokens) {
                Row r{isa, threads[ti], T, run_kernel(arena, isa, threads[ti], T, secs), ceiling[ti]};
                rows.push_back(r);
                std::printf("  %-7s %7d %3d | %9.2f %12.1f | %7.0f%%\n", isa_name(isa), r.thr, r.T, r.r.gbps, r.r.us_per_half, 100 * r.r.gbps / r.ceil);
                std::fflush(stdout);
            }

    if (l2) {
        std::printf("\ncache-resident (64 gate rows = 174 KB, one core): the compute ceiling of the dot kernel, GB/s of weights:\n");
        for (Isa isa : isas)
            for (int T : tokens) std::printf("  %-7s T=%d: %7.2f GB/s\n", isa_name(isa), T, l2_resident_gbps(isa, T, secs));
    }

    if (sweep) {
        std::printf("\nprefetch-distance sweep (avx512), GB/s of expert weights; the default is %d B (software prefetch hint %s):\n", prefetch_bytes(), DS41_BENCH_HINT_NAME);
        std::printf("  %-8s", "pf bytes");
        for (int T : tokens)
            for (int th : threads) std::printf(" %10s", ("T" + std::to_string(T) + "/" + std::to_string(th) + "thr").c_str());
        std::printf("\n");
        const int saved = prefetch_bytes();
        for (int pf : {0, 256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 32768}) {
            set_prefetch_bytes(pf);
            std::printf("  %-8d", pf);
            for (int T : tokens)
                for (int th : threads) std::printf(" %10.2f", run_kernel(arena, Isa::kAvx512, th, T, secs).gbps);
            std::printf("\n");
            std::fflush(stdout);
        }
        set_prefetch_bytes(saved);
    }

    // ---- what the CPU experts alone would allow: PLAN.md's arithmetic with the VM's measurements ----
    const double expert_bytes = 2.0 * kHalfBytes;      // one routed expert = both halves
    std::printf("\ntok/s equivalent - decode, T = 1, from THIS VM's measurements (one socket's worth of threads does BOTH halves in turn):\n");
    std::printf("  tokens/s = 1 / (240 activations x miss rate x time per expert); bytes per expert %.1f MB, 240 x 18.8 MB = %.2f GB at miss rate 1\n",
                expert_bytes / 1e6, 240 * expert_bytes / 1e9);
    std::printf("  %-7s %7s | %10s | %s\n", "isa", "threads", "ms/expert", "tok/s at miss rate  25%     50%     75%    100%");
    for (const Row& r : rows) {
        if (r.T != 1) continue;
        const double ms = 2.0 * r.r.us_per_half / 1e3;
        std::printf("  %-7s %7d | %10.3f |                    ", isa_name(r.isa), r.thr, ms);
        for (double miss : {0.25, 0.5, 0.75, 1.0}) std::printf(" %7.1f", 1e3 / (240 * miss * ms));
        std::printf("\n");
    }
    for (const Row& r : rows) {
        if (r.T == 1 || r.isa != Isa::kAvx512) continue;
        const double ms = 2.0 * r.r.us_per_half / 1e3;
        std::printf("  avx512 T=%d, %d thr: %.3f ms/expert; if the %d tokens of a verify window all hit the experts this read serves, token-equivalents/s at 50%% miss: %.1f\n",
                    r.T, r.thr, ms, r.T, r.T * 1e3 / (240 * 0.5 * ms));
    }
    std::printf("\nprojection for the target box (an ASSUMPTION, not a measurement): the kernel keeps the efficiency measured here\n");
    std::printf("(GB/s consumed / plain-read GB/s, avx512, T=1, widest thread count = %d) and the box's two sockets read %.0f GB/s together\n",
                threads.back(), box_gbps);
    for (const Row& r : rows)
        if (r.isa == Isa::kAvx512 && r.T == 1 && r.thr == threads.back()) {
            const double eff = r.r.gbps / r.ceil, kernel = eff * box_gbps;
            std::printf("  efficiency %.0f%% -> %.0f GB/s of expert weights -> ms/expert %.3f (both sockets in parallel on their halves) -> tok/s at miss 25/50/75/100%%:",
                        100 * eff, kernel, expert_bytes / (kernel * 1e9) * 1e3);
            for (double miss : {0.25, 0.5, 0.75, 1.0}) std::printf(" %.1f", 1.0 / (240 * miss * expert_bytes / (kernel * 1e9)));
            std::printf("\n");
        }
    return 0;
}
