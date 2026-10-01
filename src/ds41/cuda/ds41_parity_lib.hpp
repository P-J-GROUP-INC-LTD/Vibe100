// src/ds41/cuda/ds41_parity_lib.hpp - DS-D: the parity drivers (router, hit/miss split, hit experts) behind the V100 programs and
// the CPU emulation test.  Header only; the device is reached through the small `Dev` interface, so the same checks run
//   * on a V100: ds41_router_parity / ds41_split_parity / ds41_expert_parity (src/ds41/cuda/ds41_*_parity.cpp, cudaMalloc + events),
//   * on the host: ds41_cuda_emu_test (the kernels compiled for the CPU, "device memory" = malloc, see ds41_emu.hpp).
//
// Output: one line per check, "PASS <name>: <numbers>" or "FAIL <name>: <numbers>"; near-ties and informational numbers are
// printed with "INFO"; the summary line is "ALL PASS (n checks)" or "FAILED (k of n)".  Exit status of the programs: 0 iff all pass.
//
// What is covered beyond the arithmetic (CONTRACTS.md, "Decided after the audit"): the router's logits are bit-identical for every T; the
// split treats a residency value outside [0, n_slots) as a counted miss; its result reaches the host through a per-layer record whose
// sequence number is stored last with release semantics (layers queued ahead included); the activation quantiser gives the contract's
// bytes on zero / tiny / Inf / NaN / tie blocks; NaN propagates through the SwiGLU clamps into y.
#pragma once

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ds41_ref.hpp"
#include "strata/ds41/cuda/ds41_cuda.hpp"

namespace strata::ds41::cuda::parity {

// ---------------------------------------------------------------------------------------------------------------------
// device access
// ---------------------------------------------------------------------------------------------------------------------
struct Dev {
    virtual ~Dev() = default;
    virtual void* alloc(size_t bytes) = 0;
    virtual void release(void* p) = 0;
    virtual void h2d(void* dst, const void* src, size_t n) = 0;
    virtual void d2h(void* dst, const void* src, size_t n) = 0;
    virtual void fill(void* p, int byte, size_t n) = 0;
    virtual void sync() = 0;
    virtual void* stream() = 0;
    /// Memory the HOST can read while the device runs (a GPU: mapped pinned memory, cudaHostAlloc(cudaHostAllocMapped), the same pointer
    /// is valid in kernels under UVA; the emulation: plain memory).  Zero-filled.
    virtual void* alloc_mapped(size_t bytes) = 0;
    virtual void release_mapped(void* p) = 0;
    /// average microseconds of fn() over `reps` calls (after one warm-up), the device synchronised around the whole loop
    virtual double time_us(const std::function<void()>& fn, int reps) = 0;
    virtual bool is_emulation() const = 0;
};

template <class T>
struct DevBuf {
    Dev& dev;
    T* p = nullptr;
    size_t n = 0;
    DevBuf(Dev& d, size_t count) : dev(d), n(count) { p = static_cast<T*>(dev.alloc(count * sizeof(T))); }
    ~DevBuf() { dev.release(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void up(const std::vector<T>& v) { dev.h2d(p, v.data(), v.size() * sizeof(T)); }
    std::vector<T> down(size_t count) const {
        std::vector<T> v(count);
        dev.d2h(v.data(), p, count * sizeof(T));
        return v;
    }
    std::vector<T> down() const { return down(n); }
    void zero() { dev.fill(p, 0, n * sizeof(T)); }
};

/// A zero-filled host-visible array (see Dev::alloc_mapped); read it through the pointer, no copy.
template <class T>
struct MappedBuf {
    Dev& dev;
    T* p = nullptr;
    size_t n = 0;
    MappedBuf(Dev& d, size_t count) : dev(d), n(count) { p = static_cast<T*>(dev.alloc_mapped(count * sizeof(T))); }
    ~MappedBuf() { dev.release_mapped(p); }
    MappedBuf(const MappedBuf&) = delete;
    MappedBuf& operator=(const MappedBuf&) = delete;
};

struct Report {
    int checks = 0, failures = 0;
    void line(bool ok, const char* name, const char* fmt, ...) {
        ++checks;
        failures += ok ? 0 : 1;
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        std::printf("%s %s: %s\n", ok ? "PASS" : "FAIL", name, buf);
        std::fflush(stdout);
    }
    static void info(const char* fmt, ...) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        std::printf("INFO %s\n", buf);
        std::fflush(stdout);
    }
    int summary() const {
        if (failures == 0) std::printf("ALL PASS (%d checks)\n", checks);
        else std::printf("FAILED (%d of %d checks)\n", failures, checks);
        return failures == 0 ? 0 : 1;
    }
};

inline std::string fmt_str(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return buf;
}

// =====================================================================================================================
// router
// =====================================================================================================================
struct RouterOpts {
    uint64_t seed = 1;
    std::vector<int> Ts = {1, 2, 3, 4, 5, 8, 17, 32, 33, 70};
    int timing_reps = 100;                 // 0: no timing
    std::vector<int> timing_Ts = {1, 2, 4, 8, 32, 1024, 4096};
    bool tie_test = true;
    // "Routing must not depend on T" (CONTRACTS.md): the first T tokens of one pool routed as a T-token batch, against the same tokens in the
    // largest batch, and single tokens routed ALONE against their rows of that batch: logits compared as BITS, ids and weights exactly.
    std::vector<int> indep_Ts = {2, 3, 5, 8, 17, 32, 33, 40, 100};
    std::vector<int> indep_alone = {0, 1, 7, 8, 31, 32, 33, 99};
    std::vector<std::pair<int, int>> indep_windows = {{37, 8}, {60, 33}, {5, 4}};     // (first token, count): a window that starts mid-batch
};

inline int run_router_parity(Dev& dev, const RouterOpts& o, Report& rep) {
    using namespace ref;
    Rng rng(o.seed);
    const int Tmax = *std::max_element(o.Ts.begin(), o.Ts.end());
    const int Tindep = o.indep_Ts.empty() ? 0 : *std::max_element(o.indep_Ts.begin(), o.indep_Ts.end());
    const int Tpool = std::max(Tmax, Tindep);                // tokens in the pool of inputs

    std::vector<uint16_t> wg((size_t) kExperts * kHidden);
    std::vector<double> wgd(wg.size());
    for (size_t i = 0; i < wg.size(); ++i) {
        wg[i] = f32_to_bf16((float) (rng.normal() * 0.02));
        wgd[i] = (double) bf16_to_f32(wg[i]);
    }
    std::vector<float> bias(kExperts);
    for (auto& b : bias) b = (float) (rng.normal() * 0.1);

    // a pool of Tmax tokens of four kinds: ordinary, with outlier channels, small, and big (logits of +-100: the softplus linear branch)
    std::vector<float> xpool((size_t) Tpool * kHidden);
    for (int t = 0; t < Tpool; ++t) {
        const int kind = t & 3;
        for (int k = 0; k < kHidden; ++k) {
            double v = rng.normal();
            if (kind == 1 && (k % 640) == 7) v *= 30.0;
            if (kind == 2) v *= 0.2;
            if (kind == 3) v *= 25.0;
            xpool[(size_t) t * kHidden + k] = (float) v;
        }
    }

    DevBuf<uint16_t> d_wg(dev, wg.size());
    d_wg.up(wg);
    DevBuf<float> d_bias(dev, kExperts);
    d_bias.up(bias);

    // reference for the whole pool, once
    std::vector<RouterRef> refs(Tmax);
    {
        std::vector<double> xd(kHidden);
        for (int t = 0; t < Tmax; ++t) {
            RouterRef& r = refs[t];
            r.logits.assign(kExperts, 0.0);
            r.mass.assign(kExperts, 0.0);
            r.s.assign(kExperts, 0.0);
            r.v.assign(kExperts, 0.0);
            for (int k = 0; k < kHidden; ++k) xd[k] = (double) xpool[(size_t) t * kHidden + k];
            for (int e = 0; e < kExperts; ++e) {
                double acc = 0.0, m = 0.0;
                const double* w = &wgd[(size_t) e * kHidden];
                for (int k = 0; k < kHidden; ++k) {
                    const double p = xd[k] * w[k];
                    acc += p;
                    m += std::fabs(p);
                }
                r.logits[e] = acc;
                r.mass[e] = m;
                r.s[e] = softplus_sqrt(acc);
                r.v[e] = r.s[e] + (double) bias[e];
            }
            std::vector<int> order(kExperts);
            for (int e = 0; e < kExperts; ++e) order[e] = e;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return r.v[a] > r.v[b]; });     // ties: lowest index first
            for (int i = 0; i < kTopK; ++i) r.ids[i] = order[i];
        }
    }

    for (int T : o.Ts) {
        DevBuf<float> d_x(dev, (size_t) T * kHidden);
        d_x.dev.h2d(d_x.p, xpool.data(), (size_t) T * kHidden * sizeof(float));
        DevBuf<float> d_logits(dev, (size_t) T * kExperts);
        DevBuf<int32_t> d_ids(dev, (size_t) T * kTopK);
        DevBuf<float> d_w(dev, (size_t) T * kTopK);
        router_forward(d_x.p, d_wg.p, d_bias.p, T, d_logits.p, d_ids.p, d_w.p, dev.stream());
        dev.sync();
        const auto logits = d_logits.down();
        const auto ids = d_ids.down();
        const auto w = d_w.down();

        double worst_logit = 0.0;      // max |gpu - ref| / (3e-6 * mass + 1e-6)  (<= 1 passes)
        int sets_equal = 0, order_diff = 0, near_ties = 0, hard_fail = 0, w_fail = 0;
        double worst_w = 0.0;
        for (int t = 0; t < T; ++t) {
            const RouterRef& r = refs[t];
            for (int e = 0; e < kExperts; ++e) {
                const double err = std::fabs((double) logits[(size_t) t * kExperts + e] - r.logits[e]);
                worst_logit = std::max(worst_logit, err / (3e-6 * r.mass[e] + 1e-6));
            }
            // selection
            std::vector<double> sorted = r.v;
            std::sort(sorted.begin(), sorted.end(), std::greater<double>());
            const double kth = sorted[kTopK - 1];
            const double tol = 2e-5 * std::max(1.0, std::fabs(kth));
            std::set<int> gset(ids.begin() + (size_t) t * kTopK, ids.begin() + (size_t) (t + 1) * kTopK), rset(r.ids, r.ids + kTopK);
            bool ok = (int) gset.size() == kTopK;
            for (int e : gset) ok = ok && e >= 0 && e < kExperts;
            if (ok && gset == rset) {
                ++sets_equal;
                for (int i = 0; i < kTopK; ++i)
                    if (ids[(size_t) t * kTopK + i] != r.ids[i]) {
                        const double a = r.v[ids[(size_t) t * kTopK + i]], b = r.v[r.ids[i]];
                        if (std::fabs(a - b) <= 2e-5 * std::max(1.0, std::fabs(a))) {
                            ++near_ties;
                            Report::info("router T=%d token %d: ranks %d differ by a near-tie (|dv| = %.3g)", T, t, i, std::fabs(a - b));
                        } else {
                            ++order_diff;
                        }
                        break;
                    }
            } else if (ok) {
                bool near = true;
                for (int e : gset)
                    if (!rset.count(e) && std::fabs(r.v[e] - kth) > tol) near = false;
                if (near) {
                    ++near_ties;
                    Report::info("router T=%d token %d: a different expert set by a NEAR-TIE at the 6th place (tol %.3g) - reported, not failed", T, t, tol);
                } else {
                    ++hard_fail;
                }
            } else {
                ++hard_fail;
            }
            if (!ok || gset != rset) {
                std::printf("INFO router T=%d token %d: gpu ids", T, t);
                for (int i = 0; i < kTopK; ++i) std::printf(" %d(%.4f)", ids[(size_t) t * kTopK + i], r.v[ids[(size_t) t * kTopK + i] % kExperts]);
                std::printf("  ref ids");
                for (int i = 0; i < kTopK; ++i) std::printf(" %d(%.4f)", r.ids[i], r.v[r.ids[i]]);
                std::printf("\n");
            }
            // weights: from the GPU's own ids, in double
            if (ok) {
                double sum = 0.0;
                for (int i = 0; i < kTopK; ++i) sum += r.s[ids[(size_t) t * kTopK + i]];
                for (int i = 0; i < kTopK; ++i) {
                    const double wr = r.s[ids[(size_t) t * kTopK + i]] / (sum + 1e-20) * 1.5;
                    const double err = std::fabs((double) w[(size_t) t * kTopK + i] - wr) / (5e-5 * std::fabs(wr) + 1e-7);
                    worst_w = std::max(worst_w, err);
                    if (err > 1.0) ++w_fail;
                }
            }
        }
        rep.line(worst_logit <= 1.0, fmt_str("router logits T=%d", T).c_str(), "worst error %.3f of the allowed (3e-6 x sum|x w|)", worst_logit);
        rep.line(hard_fail == 0 && order_diff == 0, fmt_str("router ids T=%d", T).c_str(),
                 "%d of %d tokens identical sets, %d near-tie differences reported, %d hard mismatches, %d order mismatches", sets_equal, T,
                 near_ties, hard_fail, order_diff);
        rep.line(w_fail == 0, fmt_str("router weights T=%d", T).c_str(), "worst error %.3f of the allowed (5e-5 relative)", worst_w);
        if (T == o.Ts.front() || T == Tmax) {   // determinism: the second run must be bit identical
            router_forward(d_x.p, d_wg.p, d_bias.p, T, d_logits.p, d_ids.p, d_w.p, dev.stream());
            dev.sync();
            const auto logits2 = d_logits.down();
            const auto w2 = d_w.down();
            rep.line(logits2 == logits && w2 == w, fmt_str("router determinism T=%d", T).c_str(), "second run bit identical: %s",
                     (logits2 == logits && w2 == w) ? "yes" : "NO");
        }
    }

    if (Tindep > 0) {
        // ---- the router's result for a token must not depend on how many tokens are routed with it ----
        struct Run {
            std::vector<float> logits, w;
            std::vector<int32_t> ids;
        };
        auto route = [&](int first, int T) {
            DevBuf<float> d_x(dev, (size_t) T * kHidden);
            dev.h2d(d_x.p, xpool.data() + (size_t) first * kHidden, (size_t) T * kHidden * sizeof(float));
            DevBuf<float> d_logits(dev, (size_t) T * kExperts);
            DevBuf<int32_t> d_ids(dev, (size_t) T * kTopK);
            DevBuf<float> d_w(dev, (size_t) T * kTopK);
            router_forward(d_x.p, d_wg.p, d_bias.p, T, d_logits.p, d_ids.p, d_w.p, dev.stream());
            dev.sync();
            return Run{d_logits.down(), d_w.down(), d_ids.down()};
        };
        // do tokens [a, a + n) of run `r` (whose first token is r_first) equal tokens [b, b + n) of `big` (whose first token is big_first), bit for bit?
        auto same_tokens = [&](const Run& r, int r_first, const Run& big, int big_first, int a0, int n) {
            long bad_logit = 0, bad_ids = 0, bad_w = 0;
            for (int t = a0; t < a0 + n; ++t) {
                const int bt = t + r_first - big_first;
                bad_logit += std::memcmp(&r.logits[(size_t) t * kExperts], &big.logits[(size_t) bt * kExperts], kExperts * sizeof(float)) != 0;
                bad_ids += std::memcmp(&r.ids[(size_t) t * kTopK], &big.ids[(size_t) bt * kTopK], kTopK * sizeof(int32_t)) != 0;
                bad_w += std::memcmp(&r.w[(size_t) t * kTopK], &big.w[(size_t) bt * kTopK], kTopK * sizeof(float)) != 0;
            }
            return std::make_tuple(bad_logit, bad_ids, bad_w);
        };
        const Run big = route(0, Tindep);
        for (int T : o.indep_Ts) {
            if (T == Tindep) continue;
            const Run r = route(0, T);
            const auto [bl, bi, bw] = same_tokens(r, 0, big, 0, 0, T);
            rep.line(bl == 0 && bi == 0 && bw == 0, fmt_str("router T-independence T=%d vs T=%d", T, Tindep).c_str(),
                     "%d tokens: logits bit-identical (%ld rows differ), ids identical (%ld rows differ), weights bit-identical (%ld rows differ)", T, bl, bi, bw);
        }
        for (const auto& wnd : o.indep_windows) {
            if (wnd.first + wnd.second > Tindep) continue;
            const Run r = route(wnd.first, wnd.second);
            const auto [bl, bi, bw] = same_tokens(r, wnd.first, big, 0, 0, wnd.second);
            rep.line(bl == 0 && bi == 0 && bw == 0, fmt_str("router T-independence window [%d, %d) vs T=%d", wnd.first, wnd.first + wnd.second, Tindep).c_str(),
                     "logits bit-identical (%ld rows differ), ids (%ld), weights (%ld)", bl, bi, bw);
        }
        long al = 0, ai = 0, aw = 0;
        int n_alone = 0;
        for (int t : o.indep_alone) {
            if (t >= Tindep) continue;
            const Run r = route(t, 1);
            const auto [bl, bi, bw] = same_tokens(r, t, big, 0, 0, 1);
            al += bl;
            ai += bi;
            aw += bw;
            ++n_alone;
        }
        rep.line(al == 0 && ai == 0 && aw == 0, fmt_str("router T-independence: %d tokens alone (T=1) vs T=%d", n_alone, Tindep).c_str(),
                 "logits bit-identical (%ld rows differ), ids (%ld), weights (%ld)", al, ai, aw);
    }

    if (o.tie_test) {
        // x = 0: every logit is 0, s is the same for all experts, the choice is made by the bias alone; the bias has big ties.
        std::vector<float> b2(kExperts);
        for (int e = 0; e < kExperts; ++e) b2[e] = 0.25f * (float) (e % 5) + (e == 377 ? 1.0f : 0.0f);      // max bias group: e%5==4 (1.0); expert 377 = 0.5 + 1.0
        DevBuf<float> d_b2(dev, kExperts);
        d_b2.up(b2);
        for (int T : {1, 3}) {
            std::vector<float> zero((size_t) T * kHidden, 0.0f);
            DevBuf<float> d_x(dev, zero.size());
            d_x.up(zero);
            DevBuf<float> d_logits(dev, (size_t) T * kExperts);
            DevBuf<int32_t> d_ids(dev, (size_t) T * kTopK);
            DevBuf<float> d_w(dev, (size_t) T * kTopK);
            router_forward(d_x.p, d_wg.p, d_b2.p, T, d_logits.p, d_ids.p, d_w.p, dev.stream());
            dev.sync();
            const auto ids = d_ids.down();
            const auto w = d_w.down();
            // expected: 377 (bias 1.5 > 1.0), then the lowest indices of the bias-1.0 group (e % 5 == 4): 4, 9, 14, 19, 24
            const int expect[kTopK] = {377, 4, 9, 14, 19, 24};
            bool ok = true;
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < kTopK; ++i) ok = ok && ids[(size_t) t * kTopK + i] == expect[i];
            const double s0 = std::sqrt(std::log(2.0));
            bool wok = true;
            for (int i = 0; i < kTopK; ++i) wok = wok && std::fabs((double) w[i] - s0 / (6 * s0) * 1.5) < 1e-5;
            rep.line(ok && wok, fmt_str("router tie-break T=%d", T).c_str(), "ids %d %d %d %d %d %d (expect 377 4 9 14 19 24), weights %.6f (expect 0.25)",
                     ids[0], ids[1], ids[2], ids[3], ids[4], ids[5], (double) w[0]);
        }
    }

    if (o.timing_reps > 0 && !dev.is_emulation()) {
        for (int T : o.timing_Ts) {
            std::vector<float> xs((size_t) T * kHidden);
            for (size_t i = 0; i < xs.size(); ++i) xs[i] = xpool[i % xpool.size()];
            DevBuf<float> d_x(dev, xs.size());
            d_x.up(xs);
            DevBuf<float> d_logits(dev, (size_t) T * kExperts);
            DevBuf<int32_t> d_ids(dev, (size_t) T * kTopK);
            DevBuf<float> d_w(dev, (size_t) T * kTopK);
            const int reps = T >= 1024 ? std::max(5, o.timing_reps / 10) : o.timing_reps;
            const double us = dev.time_us([&] { router_forward(d_x.p, d_wg.p, d_bias.p, T, d_logits.p, d_ids.p, d_w.p, dev.stream()); }, reps);
            const double flops = 2.0 * T * kExperts * kHidden;
            Report::info("router T=%-5d %9.2f us per call   weights 3.93 MB -> %.0f GB/s,  %.2f TFLOPS", T, us, 3.93216e6 / (us * 1e3), flops / (us * 1e6));
        }
    }
    return rep.failures == 0 ? 0 : 1;
}

// =====================================================================================================================
// split
// =====================================================================================================================
struct SplitOpts {
    uint64_t seed = 2;
    std::vector<int> Ts = {1, 2, 3, 5, 8, 9, 40, 1000};
    int queued_layers = 6;                 // launches queued back to back in the "layers in flight" test (0: skip it)
};

namespace detail {

/// The host reference of one routed row of ids against a residency table: the slot (or -1) and whether the table entry was corrupt.
struct SplitRef {
    std::vector<HitEntry> hits;
    std::vector<MissEntry> misses;
    int n_bad = 0;
};
inline SplitRef split_reference(const std::vector<int32_t>& ids, const std::vector<float>& w, int T, const std::vector<int32_t>& res, int n_slots, int layer) {
    SplitRef r;
    for (int i = 0; i < T * kTopK; ++i) {
        const int id = ids[i];
        const int32_t v = (id >= 0 && id < kExperts) ? res[(size_t) layer * kExperts + id] : -1;
        const bool ok = v >= 0 && v < n_slots;
        r.n_bad += (v != -1 && !ok) ? 1 : 0;
        if (ok) r.hits.push_back({i / kTopK, i % kTopK, v, w[i]});
        else r.misses.push_back({i / kTopK, i % kTopK, id, w[i]});
    }
    return r;
}
inline bool lists_equal(const SplitRef& ref, const SplitCounts& c, const std::vector<HitEntry>& hits, const std::vector<MissEntry>& miss) {
    bool ok = c.n_hits == (int) ref.hits.size() && c.n_misses == (int) ref.misses.size() && c.n_bad_slots == ref.n_bad;
    for (size_t i = 0; ok && i < ref.hits.size(); ++i)
        ok = hits[i].token == ref.hits[i].token && hits[i].k == ref.hits[i].k && hits[i].slot == ref.hits[i].slot && hits[i].weight == ref.hits[i].weight;
    for (size_t i = 0; ok && i < ref.misses.size(); ++i)
        ok = miss[i].token == ref.misses[i].token && miss[i].k == ref.misses[i].k && miss[i].expert == ref.misses[i].expert && miss[i].weight == ref.misses[i].weight;
    return ok;
}
inline std::vector<MissEntry> read_mapped_misses(const MissEntry* p, int n, size_t capacity) { return std::vector<MissEntry>(p, p + (n > 0 ? std::min((size_t) n, capacity) : 0)); }

}  // namespace detail

inline int run_split_parity(Dev& dev, const SplitOpts& o, Report& rep) {
    using namespace ref;
    Rng rng(o.seed);
    const int32_t junk_values[] = {-1, -1, -2, -1000, 0, 5, 299, 300, 301, 999, 1000, 1001, 2147483647, (int32_t) 0x80000000u};
    for (int variant = 0; variant < 8; ++variant) {
        // residency table [40][384].  0: ~40 % resident; 1: nothing resident; 2: everything; 3: a handful; 4: ~40 % + a hot expert;
        // 5: ~50 % resident but n_slots = 300, so every slot >= 300 is out of range (a miss, and counted); 6: junk values (negative, == n_slots,
        // beyond it, INT_MAX, INT_MIN) around n_slots = 300; 7: a table that was zero-initialised instead of -1 with n_slots = 0 (every
        // entry "names" slot 0 of an empty cache: all misses, all counted).
        int n_slots = 1000;
        std::vector<int32_t> res((size_t) kLayers * kExperts, -1);
        for (int l = 0; l < kLayers; ++l) {
            int next = 0;
            for (int e = 0; e < kExperts; ++e) {
                bool in = false;
                switch (variant) {
                    case 0: in = rng.uniform() < 0.4; break;
                    case 1: in = false; break;
                    case 2: in = true; break;
                    case 3: in = (e % 61) == 5; break;
                    case 4: in = e == 100 || rng.uniform() < 0.4; break;
                    case 5: in = rng.uniform() < 0.5; break;
                    default: break;
                }
                if (in) res[(size_t) l * kExperts + e] = (next++ * 7 + l) % 1000;
                if (variant == 6) res[(size_t) l * kExperts + e] = junk_values[rng.range(0, (int) (sizeof junk_values / sizeof junk_values[0]) - 1)];
                if (variant == 7) res[(size_t) l * kExperts + e] = 0;
            }
        }
        if (variant == 5 || variant == 6) n_slots = 300;
        if (variant == 7) n_slots = 0;
        DevBuf<int32_t> d_res(dev, res.size());
        d_res.up(res);
        for (int T : o.Ts) {
            const int layer = (variant * 11 + T) % kLayers;
            std::vector<int32_t> ids((size_t) T * kTopK);
            std::vector<float> w((size_t) T * kTopK);
            for (int t = 0; t < T; ++t) {
                std::set<int> used;
                for (int k = 0; k < kTopK; ++k) {
                    int e;
                    do {
                        // a small pool of popular experts so that several tokens share experts
                        e = (rng.uniform() < 0.5) ? rng.range(95, 110) : rng.range(0, kExperts - 1);
                    } while (used.count(e));
                    used.insert(e);
                    ids[(size_t) t * kTopK + k] = e;
                    w[(size_t) t * kTopK + k] = (float) (0.01 + rng.uniform());
                }
            }
            if (T == 9 || T == 5) {                 // edge ids: out of range -> misses
                ids[3] = -1;
                ids[kTopK + 2] = 384;
                ids[kTopK + 4] = 1 << 30;
            }
            DevBuf<int32_t> d_ids(dev, ids.size());
            d_ids.up(ids);
            DevBuf<float> d_w(dev, w.size());
            d_w.up(w);
            DevBuf<HitEntry> d_hits(dev, ids.size());
            DevBuf<MissEntry> d_miss(dev, ids.size());
            DevBuf<HitGroup> d_groups(dev, ids.size());
            DevBuf<SplitCounts> d_counts(dev, 1);
            const bool grouped = T <= kMaxExpertTokens;
            const bool hosted = (variant & 1) != 0;          // odd variants hand the result to the host through a mapped record and miss list
            MappedBuf<SplitHostRecord> m_rec(dev, 1);
            MappedBuf<MissEntry> m_miss(dev, ids.size());
            const uint32_t seq = 1;
            split_hits_misses(d_ids.p, d_w.p, T, d_res.p, n_slots, layer, d_hits.p, hosted ? m_miss.p : d_miss.p, grouped ? d_groups.p : nullptr, d_counts.p,
                              hosted ? m_rec.p : nullptr, seq, dev.stream());
            bool doorbell = true;
            SplitHostRecord hrec{};
            std::vector<MissEntry> hmiss;
            if (hosted) {                                     // the host waits for the doorbell BEFORE the stream is synchronised
                doorbell = split_host_wait(*m_rec.p, seq);
                hrec = *m_rec.p;
                hmiss = detail::read_mapped_misses(m_miss.p, hrec.n_misses, ids.size());
            }
            dev.sync();
            const SplitCounts c = d_counts.down(1)[0];
            const auto hits = d_hits.down(c.n_hits > 0 ? c.n_hits : 1);
            const auto miss = hosted ? hmiss : d_miss.down(c.n_misses > 0 ? c.n_misses : 1);

            // host reference
            const detail::SplitRef ref = detail::split_reference(ids, w, T, res, n_slots, layer);
            const bool ok = detail::lists_equal(ref, c, hits, miss);
            rep.line(ok, fmt_str("split lists variant=%d T=%d", variant, T).c_str(),
                     "%d hits, %d misses (expect %zu, %zu), %d corrupt residency entries (expect %d), n_slots %d, order and fields identical: %s", c.n_hits, c.n_misses,
                     ref.hits.size(), ref.misses.size(), c.n_bad_slots, ref.n_bad, n_slots, ok ? "yes" : "NO");
            if (hosted) {
                const bool hok = doorbell && hrec.seq == seq && hrec.n_hits == c.n_hits && hrec.n_misses == c.n_misses && hrec.n_groups == c.n_groups &&
                                 hrec.n_bad_slots == c.n_bad_slots && detail::lists_equal(ref, c, hits, hmiss);
                rep.line(hok, fmt_str("split host record variant=%d T=%d", variant, T).c_str(),
                         "doorbell rang: %s, seq %u (expect %u), host counts %d/%d/%d/%d equal the device counts: %s", doorbell ? "yes" : "NO", hrec.seq, seq, hrec.n_hits,
                         hrec.n_misses, hrec.n_groups, hrec.n_bad_slots, hok ? "yes" : "NO");
            }
            if (grouped) {
                // expected groups: by decreasing size, ties by first appearance of the slot
                std::vector<HitGroup> eg;
                const std::vector<HitEntry>& eh = ref.hits;
                for (size_t i = 0; i < eh.size(); ++i) {
                    int g = -1;
                    for (size_t j = 0; j < eg.size(); ++j)
                        if (eg[j].slot == eh[i].slot) g = (int) j;
                    if (g < 0) {
                        HitGroup n{};
                        n.slot = eh[i].slot;
                        n.n = 0;
                        eg.push_back(n);
                        g = (int) eg.size() - 1;
                    }
                    eg[g].hit[eg[g].n++] = (int) i;
                }
                std::stable_sort(eg.begin(), eg.end(), [](const HitGroup& a, const HitGroup& b) { return a.n > b.n; });     // larger groups first
                const auto groups = d_groups.down(eg.empty() ? 1 : eg.size());
                bool gok = c.n_groups == (int) eg.size();
                for (size_t g = 0; gok && g < eg.size(); ++g) {
                    gok = groups[g].slot == eg[g].slot && groups[g].n == eg[g].n;
                    for (int j = 0; gok && j < eg[g].n; ++j) gok = groups[g].hit[j] == eg[g].hit[j];
                }
                int biggest = 0;
                for (auto& g : eg) biggest = std::max(biggest, g.n);
                rep.line(gok, fmt_str("split groups variant=%d T=%d", variant, T).c_str(), "%d groups (expect %zu), largest has %d hits: %s", c.n_groups,
                         eg.size(), biggest, gok ? "identical" : "MISMATCH");
            }
        }
    }

    // ---- layers in flight: K splits queued back to back, one record + miss list each, then waited for; twice, with the SAME records (seq 1, then 2):
    // the second round's wait must return the second round's results, never the first's.
    if (o.queued_layers > 0) {
        const int K = o.queued_layers;
        std::vector<int32_t> res((size_t) kLayers * kExperts, -1);
        const int n_slots = 400;
        for (int l = 0; l < kLayers; ++l) {
            int next = 0;
            for (int e = 0; e < kExperts; ++e)
                if (rng.uniform() < 0.45) res[(size_t) l * kExperts + e] = (next++ * 5 + l) % 400;
        }
        DevBuf<int32_t> d_res(dev, res.size());
        d_res.up(res);
        DevBuf<HitEntry> d_hits(dev, 6 * kMaxExpertTokens);                  // device scratch shared by all the launches (stream order protects it)
        DevBuf<HitGroup> d_groups(dev, 6 * kMaxExpertTokens);
        DevBuf<SplitCounts> d_counts(dev, 1);
        std::vector<std::unique_ptr<MappedBuf<SplitHostRecord>>> recs;
        std::vector<std::unique_ptr<MappedBuf<MissEntry>>> miss;
        for (int k = 0; k < K; ++k) {
            recs.emplace_back(new MappedBuf<SplitHostRecord>(dev, 1));
            miss.emplace_back(new MappedBuf<MissEntry>(dev, 6 * kMaxExpertTokens));
        }
        uint32_t seq = 0;
        for (int round = 0; round < 2; ++round) {
            seq = split_next_seq(seq);
            std::vector<std::vector<int32_t>> ids(K);
            std::vector<std::vector<float>> w(K);
            std::vector<int> Tk(K), layer(K);
            std::vector<std::unique_ptr<DevBuf<int32_t>>> d_ids;
            std::vector<std::unique_ptr<DevBuf<float>>> d_w;
            for (int k = 0; k < K; ++k) {
                Tk[k] = 1 + (k + round) % kMaxExpertTokens;
                layer[k] = (k * 7 + round * 3) % kLayers;
                ids[k].resize((size_t) Tk[k] * kTopK);
                w[k].resize(ids[k].size());
                for (int t = 0; t < Tk[k]; ++t) {
                    std::set<int> used;
                    for (int j = 0; j < kTopK; ++j) {
                        int e;
                        do e = rng.range(0, kExperts - 1);
                        while (used.count(e));
                        used.insert(e);
                        ids[k][(size_t) t * kTopK + j] = e;
                        w[k][(size_t) t * kTopK + j] = (float) (0.01 + rng.uniform());
                    }
                }
                d_ids.emplace_back(new DevBuf<int32_t>(dev, ids[k].size()));
                d_ids.back()->up(ids[k]);
                d_w.emplace_back(new DevBuf<float>(dev, w[k].size()));
                d_w.back()->up(w[k]);
            }
            for (int k = 0; k < K; ++k)       // all launched before the host looks at any record
                split_hits_misses(d_ids[k]->p, d_w[k]->p, Tk[k], d_res.p, n_slots, layer[k], d_hits.p, miss[k]->p, d_groups.p, d_counts.p, recs[k]->p, seq, dev.stream());
            int bad = 0, stale = 0;
            for (int k = K - 1; k >= 0; --k) {                  // wait in the REVERSE order: the host does not have to follow the stream's order
                const bool rang = split_host_wait(*recs[k]->p, seq);
                const SplitHostRecord r = *recs[k]->p;
                const detail::SplitRef ref = detail::split_reference(ids[k], w[k], Tk[k], res, n_slots, layer[k]);
                const std::vector<MissEntry> hm = detail::read_mapped_misses(miss[k]->p, r.n_misses, 6 * kMaxExpertTokens);
                bool ok = rang && r.seq == seq && r.n_hits == (int) ref.hits.size() && r.n_misses == (int) ref.misses.size() && r.n_bad_slots == ref.n_bad &&
                          r.n_misses == (int) hm.size();
                for (size_t i = 0; ok && i < ref.misses.size(); ++i)
                    ok = hm[i].token == ref.misses[i].token && hm[i].k == ref.misses[i].k && hm[i].expert == ref.misses[i].expert && hm[i].weight == ref.misses[i].weight;
                bad += ok ? 0 : 1;
                stale += (r.seq != seq) ? 1 : 0;
            }
            dev.sync();
            rep.line(bad == 0, fmt_str("split layers in flight round %d (seq %u)", round + 1, seq).c_str(),
                     "%d launches queued ahead, each with its own record + miss list, waited for in reverse order: %d wrong, %d stale", K, bad, stale);
        }
    }
    return rep.failures == 0 ? 0 : 1;
}

// =====================================================================================================================
// hit experts
// =====================================================================================================================
struct ExpertOpts {
    uint64_t seed = 3;
    int slots = 24;                        // resident experts (cache slots) to create
    std::vector<int> Ts = {1, 2, 3, 4, 5, 6, 7, 8};
    int max_hits_checked = 1000000;        // limit the (slow) FP64 reference per case (emulation uses a few)
    int timing_reps = 100;                 // 0: no timing
    bool edge = true;
    bool full_cases = true;
};

namespace detail {

inline bool is_sentinel(const float* p, int n) {
    for (int i = 0; i < n; ++i) {
        uint32_t b;
        std::memcpy(&b, p + i, 4);
        if (b != 0x7FC00001u) return false;
    }
    return true;
}

struct HitCheck {
    double tight = 0.0;          // worst |y_gpu - y_ref(q)| / (5e-6 * mass + 1e-30)           (<= 1 passes)
    double h_dev = 0.0;          // worst  |deq(h_gpu) - h_ref| / (0.5001 d + 4e-6 max|h|)      (<= 1 passes)
    double h_scale = 0.0;        // worst relative scale difference
    long h_q_mismatch = 0;       // int8 values that differ from the reference quantiser
    long h_total = 0;
    double fp32_norm = 0.0;      // ||y_gpu - y_ref_fp32|| / ||y_ref_fp32||
};

}  // namespace detail

/// A context that owns the cache (random blobs) and the per-case buffers.
struct ExpertRig {
    Dev& dev;
    int slots;
    std::vector<std::vector<uint8_t>> blobs;       // host copies
    std::unique_ptr<DevBuf<uint8_t>> cache;
    int layer = 17;
    std::vector<int> pool;                         // pool[i] = the expert id resident in slot i
    DevBuf<int32_t>* residency = nullptr;
    std::vector<int32_t> res_host;

    ExpertRig(Dev& d, int nslots, uint64_t seed) : dev(d), slots(nslots) {
        using namespace ref;
        cache.reset(new DevBuf<uint8_t>(dev, (size_t) slots * kBlobBytes));
        blobs.resize(slots);
        for (int s = 0; s < slots; ++s) {
            Rng rng(seed * 1000003ull + (uint64_t) s);
            blobs[s].resize(kBlobBytes);
            gen_blob(blobs[s].data(), rng);
            // every code at every position of some block: the first blocks of the first rows of each matrix
            for (int b = 0; b < 16; ++b) {
                put_all_codes(blobs[s].data() + kBlobGate + (size_t) b * kBlockBytes, b);
                put_all_codes(blobs[s].data() + kBlobUp + (size_t) b * kBlockBytes, b + 5);
                put_all_codes(blobs[s].data() + kBlobDown + (size_t) b * kBlockBytes, b + 9);
            }
            dev.h2d(cache->p + (size_t) s * kBlobBytes, blobs[s].data(), kBlobBytes);
            pool.push_back((7 + 13 * s) % 300);        // distinct (13 is coprime with 300) and below 340, where the miss ids start
        }
        res_host.assign((size_t) kLayers * kExperts, -1);
        for (int s = 0; s < slots; ++s) res_host[(size_t) layer * kExperts + pool[s]] = s;
        residency = new DevBuf<int32_t>(dev, res_host.size());
        residency->up(res_host);
    }
    ~ExpertRig() { delete residency; }
};

struct CaseSpec {
    std::string name;
    int T = 1;
    std::vector<std::vector<int>> pool_idx;        // per token: 6 entries; >= 0: index into the resident pool, < 0: a non-resident expert (-1 - n)
};

/// Fits a hand-written case to a pool of P resident experts: pool indices >= P are folded back (and kept distinct within a token).
inline void fit_pool(CaseSpec& cs, int P) {
    for (auto& row : cs.pool_idx) {
        std::set<int> used;
        for (int& v : row) {
            if (v < 0) continue;
            v %= P;
            while (used.count(v)) v = (v + 1) % P;
            used.insert(v);
        }
    }
}

/// Runs one routed case end to end (split -> quantise -> gate/up -> down) and checks everything.  Returns the number of hits.
inline int run_expert_case(ExpertRig& rig, const CaseSpec& cs, ref::Rng& rng, int max_hits_checked, Report& rep, bool verbose_phases = true) {
    using namespace ref;
    Dev& dev = rig.dev;
    const int T = cs.T;
    std::vector<int32_t> ids((size_t) T * kTopK);
    std::vector<float> wts((size_t) T * kTopK);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < kTopK; ++k) {
            const int pi = cs.pool_idx[t][k];
            ids[(size_t) t * kTopK + k] = pi >= 0 ? rig.pool[pi] : 340 + (-1 - pi) % 40;       // 340.. are never in the pool (pool ids <= 306)
            wts[(size_t) t * kTopK + k] = (float) (0.05 + 1.2 * rng.uniform());
        }
    std::vector<float> x((size_t) T * kHidden);
    for (int t = 0; t < T; ++t)
        for (int b = 0; b < kActBlocks; ++b) {
            const double bs = std::pow(2.0, rng.range(-2, 2));            // per-block magnitude changes: the scales matter
            for (int j = 0; j < 32; ++j) x[(size_t) t * kHidden + 32 * b + j] = (float) (rng.normal() * bs);
        }
    DevBuf<float> d_x(dev, x.size());
    d_x.up(x);
    DevBuf<int32_t> d_ids(dev, ids.size());
    d_ids.up(ids);
    DevBuf<float> d_w(dev, wts.size());
    d_w.up(wts);
    DevBuf<HitEntry> d_hits(dev, ids.size());
    DevBuf<HitGroup> d_groups(dev, ids.size());
    DevBuf<SplitCounts> d_counts(dev, 1);
    MappedBuf<SplitHostRecord> m_rec(dev, 1);       // the host's view of the split: one record + miss list per layer in flight
    MappedBuf<MissEntry> m_miss(dev, ids.size());
    DevBuf<unsigned char> d_scr(dev, expert_scratch_bytes(T));
    const ExpertScratch scr = expert_scratch_carve(d_scr.p, T);
    DevBuf<float> d_parts(dev, (size_t) T * kTopK * kHidden);
    {   // the sentinel: the quiet NaN 0x7FC00001 everywhere; a miss row must still hold it after the kernels ran
        std::vector<uint32_t> sent(d_parts.n, 0x7FC00001u);
        dev.h2d(d_parts.p, sent.data(), sent.size() * 4);
    }

    const uint32_t seq = 1;
    split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.slots, rig.layer, d_hits.p, m_miss.p, d_groups.p, d_counts.p, m_rec.p, seq, dev.stream());
    experts_hits(rig.cache->p, d_x.p, d_hits.p, d_groups.p, d_counts.p, T, scr, d_parts.p, dev.stream());
    // the host takes the split's result through the doorbell while the expert kernels are still queued / running (on a GPU)
    const bool doorbell = split_host_wait(*m_rec.p, seq);
    const SplitHostRecord hrec = *m_rec.p;
    const std::vector<MissEntry> miss = detail::read_mapped_misses(m_miss.p, hrec.n_misses, ids.size());
    dev.sync();

    const SplitCounts c = d_counts.down(1)[0];
    const auto hits = d_hits.down(c.n_hits > 0 ? c.n_hits : 1);
    const auto groups = d_groups.down(c.n_groups > 0 ? c.n_groups : 1);
    const auto parts = d_parts.down();

    // device-side scratch back to the host
    std::vector<int8_t> xq_perm((size_t) T * kHidden), hq_perm((size_t) kTopK * T * kFF);
    std::vector<float> xs((size_t) T * kActBlocks), hs((size_t) kTopK * T * kHBlocks);
    dev.d2h(xq_perm.data(), scr.xq, xq_perm.size());
    dev.d2h(xs.data(), scr.xs, xs.size() * 4);
    dev.d2h(hq_perm.data(), scr.hq, hq_perm.size());
    dev.d2h(hs.data(), scr.hs, hs.size() * 4);

    // ---- the activation quantiser: bit exact against the CPU's rule
    {
        long bad = 0;
        std::vector<int8_t> q, nat((size_t) kHidden);
        std::vector<float> d;
        for (int t = 0; t < T; ++t) {
            quantize_vec(x.data() + (size_t) t * kHidden, kHidden, q, d);
            from_perm(xq_perm.data() + (size_t) t * kHidden, nat.data(), kActBlocks);
            for (int i = 0; i < kHidden; ++i) bad += nat[i] != q[i];
            for (int b = 0; b < kActBlocks; ++b) bad += std::memcmp(&xs[(size_t) t * kActBlocks + b], &d[b], 4) != 0;
        }
        rep.line(bad == 0, (cs.name + " quantize x").c_str(), "%d tokens, int8 values and fp32 scales identical to the CPU quantiser: %s (%ld differences)", T,
                 bad == 0 ? "yes" : "NO", bad);
    }

    // ---- lists
    int expect_hits = 0;
    for (int i = 0; i < T * kTopK; ++i) expect_hits += cs.pool_idx[i / kTopK][i % kTopK] >= 0;
    {
        bool ok = c.n_hits == expect_hits && c.n_misses == T * kTopK - expect_hits;
        std::set<int> slots;
        for (int i = 0; i < c.n_hits; ++i) slots.insert(hits[i].slot);
        ok = ok && c.n_groups == (int) slots.size();
        int biggest = 0;
        for (int g = 0; g < c.n_groups; ++g) biggest = std::max(biggest, groups[g].n);
        ok = ok && c.n_bad_slots == 0 && doorbell && hrec.seq == seq && hrec.n_hits == c.n_hits && hrec.n_misses == c.n_misses && hrec.n_groups == c.n_groups &&
             hrec.n_bad_slots == 0;
        rep.line(ok, (cs.name + " split").c_str(), "%d hits in %d groups (largest %d tokens), %d misses (expect %d hits), host record %s (seq %u), %d corrupt slots", c.n_hits,
                 c.n_groups, biggest, c.n_misses, expect_hits, doorbell ? "agrees" : "DOORBELL NEVER RANG", hrec.seq, c.n_bad_slots);
    }

    // ---- per hit
    detail::HitCheck worst;
    long checked = 0;
    double fp32_err2 = 0.0, fp32_ref2 = 0.0;
    std::vector<int8_t> xq_nat(kHidden), hq_nat(kFF);
    std::vector<double> h_ref, y_q, mass_q, h_f, y_f;
    bool first_nan = false;
    for (int i = 0; i < c.n_hits && checked < max_hits_checked; ++i, ++checked) {
        const HitEntry& h = hits[i];
        const uint8_t* blob = rig.blobs[h.slot].data();
        from_perm(xq_perm.data() + (size_t) h.token * kHidden, xq_nat.data(), kActBlocks);
        const float* xs_t = &xs[(size_t) h.token * kActBlocks];
        from_perm(hq_perm.data() + (size_t) i * kFF, hq_nat.data(), kHBlocks);
        const float* hs_i = &hs[(size_t) i * kHBlocks];
        const float* y_gpu = &parts[((size_t) h.token * kTopK + h.k) * kHidden];

        // phase 1 against the reference computed from the same int8 x
        expert_h_q(blob, xq_nat.data(), xs_t, (double) h.weight, h_ref);
        for (int b = 0; b < kHBlocks; ++b) {
            double hmax = 0.0;
            for (int j = 0; j < 32; ++j) hmax = std::max(hmax, std::fabs(h_ref[32 * b + j]));
            const double d_gpu = hs_i[b];
            const double d_ref = hmax / 127.0;
            if (hmax > 0) worst.h_scale = std::max(worst.h_scale, std::fabs(d_gpu - d_ref) / d_ref);
            float hf[32];
            int8_t qr[32];
            float dr;
            for (int j = 0; j < 32; ++j) hf[j] = (float) h_ref[32 * b + j];
            quantize_block(hf, qr, dr);
            for (int j = 0; j < 32; ++j) {
                const double deq = (double) hq_nat[32 * b + j] * d_gpu;
                const double allowed = 0.5001 * d_gpu + 4e-6 * hmax;
                worst.h_dev = std::max(worst.h_dev, std::fabs(deq - h_ref[32 * b + j]) / (allowed > 0 ? allowed : 1e-30));
                worst.h_q_mismatch += hq_nat[32 * b + j] != qr[j];
                ++worst.h_total;
            }
        }
        // phase 2 against the reference computed from the GPU's own h (exact integer arithmetic on both sides)
        expert_down_q(blob, hq_nat.data(), hs_i, y_q, mass_q);
        for (int r = 0; r < kHidden; ++r) {
            if (std::isnan(y_gpu[r])) first_nan = true;
            const double err = std::fabs((double) y_gpu[r] - y_q[r]) / (5e-6 * mass_q[r] + 1e-30);
            worst.tight = std::max(worst.tight, err);
        }
        // the FP32 pipeline (no quantisation anywhere)
        expert_h_f(blob, x.data() + (size_t) h.token * kHidden, (double) h.weight, h_f);
        expert_down_f(blob, h_f, y_f);
        for (int r = 0; r < kHidden; ++r) {
            const double dlt = (double) y_gpu[r] - y_f[r];
            fp32_err2 += dlt * dlt;
            fp32_ref2 += y_f[r] * y_f[r];
        }
    }
    worst.fp32_norm = std::sqrt(fp32_err2 / (fp32_ref2 > 0 ? fp32_ref2 : 1.0));
    const bool phase1_ok = worst.h_dev <= 1.0 && worst.h_scale <= 4e-6 && worst.h_q_mismatch <= worst.h_total / 500 + 2;
    rep.line(phase1_ok, (cs.name + " phase 1 (gate/up -> h)").c_str(),
             "%ld hits: dequantised h within the quantiser bound (worst %.3f of 1), scale rel err %.2e, %ld of %ld int8 differ from the reference quantiser (+-1 flips at rounding ties)",
             checked, worst.h_dev, worst.h_scale, worst.h_q_mismatch, worst.h_total);
    rep.line(worst.tight <= 1.0 && !first_nan, (cs.name + " phase 2 (down, tight)").c_str(),
             "%ld hits: W2.h vs FP64 on the same int8 h, worst error %.3f of the allowed (5e-6 x sum of |block terms|)", checked, worst.tight);
    rep.line(worst.fp32_norm <= 0.03, (cs.name + " vs FP32 activations").c_str(), "relative L2 error of the layer-expert output vs the unquantised FP64 pipeline: %.3e (bound 3e-2)",
             worst.fp32_norm);

    // ---- misses untouched, hits written
    {
        bool ok = true;
        int nmiss_checked = 0;
        for (int i = 0; i < c.n_misses; ++i) {
            const float* row = &parts[((size_t) miss[i].token * kTopK + miss[i].k) * kHidden];
            ok = ok && detail::is_sentinel(row, kHidden);
            ++nmiss_checked;
        }
        bool wrote = true;
        for (int i = 0; i < c.n_hits; ++i) {
            const float* row = &parts[((size_t) hits[i].token * kTopK + hits[i].k) * kHidden];
            wrote = wrote && !detail::is_sentinel(row, kHidden);
        }
        rep.line(ok && wrote, (cs.name + " parts").c_str(), "%d miss rows untouched: %s; %d hit rows written: %s", nmiss_checked, ok ? "yes" : "NO", c.n_hits, wrote ? "yes" : "NO");
    }
    (void) verbose_phases;
    return c.n_hits;
}

inline CaseSpec make_case(const std::string& name, int T, int seed_variant, int pool_size, int misses_per_token, ref::Rng& rng) {
    CaseSpec cs;
    cs.name = name;
    cs.T = T;
    (void) seed_variant;
    for (int t = 0; t < T; ++t) {
        std::vector<int> row;
        std::set<int> used;
        for (int k = 0; k < kTopK; ++k) {
            if (k >= kTopK - misses_per_token) {
                row.push_back(-1 - (t * 7 + k));
                continue;
            }
            int e;
            do e = rng.range(0, pool_size - 1);
            while (used.count(e));
            used.insert(e);
            row.push_back(e);
        }
        // shuffle so the misses are not always last
        for (int i = kTopK - 1; i > 0; --i) std::swap(row[i], row[rng.range(0, i)]);
        cs.pool_idx.push_back(row);
    }
    return cs;
}

/// Edge scale bytes: e = 0, 1, 127, 254, 255 (and normal ones) through the kernels with hand-made activations whose scales keep every
/// product finite (the E8M0 range is wider than FP32's), compared against FP64.  Also the decode table of all 256 bytes.
inline void run_expert_edge(Dev& dev, ref::Rng& rng, Report& rep) {
    using namespace ref;
    // ---- all 256 scale bytes
    {
        DevBuf<float> d_tab(dev, 256);
        test_e8m0_table(d_tab.p, dev.stream());
        dev.sync();
        const auto tab = d_tab.down();
        int bad = 0;
        for (int e = 0; e < 256; ++e) {
            const float want = e8m0_half((uint8_t) e);
            bad += std::memcmp(&tab[e], &want, 4) != 0;
        }
        rep.line(bad == 0, "e8m0 decode (256 values)", "bit-identical to ggml_e8m0_to_fp32_half: %s; e=0 -> %a, e=1 -> %a, e=127 -> %a, e=254 -> %a, e=255 -> %a",
                 bad == 0 ? "yes" : "NO", (double) tab[0], (double) tab[1], (double) tab[127], (double) tab[254], (double) tab[255]);
    }
    // ---- a one-expert blob whose scale byte is column structured, and hand-made activations
    const int edge_e[5] = {0, 1, 127, 254, 255};
    std::vector<uint8_t> blob(kBlobBytes);
    auto col_e = [&](int b, int nblk, int period) -> int {
        (void) nblk;
        if (b % period == 3) return edge_e[(b / period) % 5];
        return -1;
    };
    // gate/up: columns b % 16 == 3 are edge columns (each of the 5 values appears in 160/16 = 10 columns)
    std::vector<int> ecol_g(kGateRowBlocks), ecol_d(kDownRowBlocks);
    for (int b = 0; b < kGateRowBlocks; ++b) ecol_g[b] = col_e(b, kGateRowBlocks, 16);
    for (int b = 0; b < kDownRowBlocks; ++b) ecol_d[b] = col_e(b, kDownRowBlocks, 8);
    auto fill = [&](uint8_t* base, int rows, int nblk, const std::vector<int>& ecol, int elo, int ehi) {
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nblk; ++b) {
                uint8_t* blk = base + ((size_t) r * nblk + b) * kBlockBytes;
                blk[0] = (uint8_t) (ecol[b] >= 0 ? ecol[b] : rng.range(elo, ehi));
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) rng.u32();
            }
    };
    fill(blob.data() + kBlobGate, kFF, kGateRowBlocks, ecol_g, 119, 123);
    fill(blob.data() + kBlobUp, kFF, kGateRowBlocks, ecol_g, 119, 123);
    fill(blob.data() + kBlobDown, kHidden, kDownRowBlocks, ecol_d, 118, 122);
    DevBuf<uint8_t> d_blob(dev, kBlobBytes);
    d_blob.up(blob);

    const int T = 2;                                  // two tokens share the expert: one group of 2
    auto xscale = [&](int e) -> float {               // activation scale that keeps (2^(e-128) * scale) ~ 2^-9 for an edge column
        if (e < 0) return (float) (std::ldexp(1.0, -7) * (0.5 + 0.5 * rng.uniform()));
        return (float) (std::ldexp(1.0, 127 - e - 8) * (0.5 + 0.5 * rng.uniform()));
    };
    std::vector<int8_t> xq_nat((size_t) T * kHidden), hq_nat((size_t) T * kFF);
    std::vector<float> xs((size_t) T * kActBlocks), hs((size_t) T * kHBlocks);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < kHidden; ++i) xq_nat[(size_t) t * kHidden + i] = (int8_t) rng.range(-7, 7);
        for (int b = 0; b < kActBlocks; ++b) xs[(size_t) t * kActBlocks + b] = xscale(ecol_g[b]);
        for (int i = 0; i < kFF; ++i) hq_nat[(size_t) t * kFF + i] = (int8_t) rng.range(-7, 7);
        for (int b = 0; b < kHBlocks; ++b) hs[(size_t) t * kHBlocks + b] = xscale(ecol_d[b]);
    }
    std::vector<int8_t> xq_perm((size_t) T * kHidden), hq_perm((size_t) T * kFF);
    to_perm(xq_nat.data(), xq_perm.data(), T * kActBlocks);
    to_perm(hq_nat.data(), hq_perm.data(), T * kHBlocks);
    // lists: tokens 0 and 1 (each with k = 0) -> one group {slot 0, hits 0, 1}
    HitEntry h0{0, 0, 0, 1.0f}, h1{1, 0, 0, 0.75f};
    std::vector<HitEntry> hits = {h0, h1};
    HitGroup grp{};
    grp.slot = 0;
    grp.n = 2;
    grp.hit[0] = 0;
    grp.hit[1] = 1;
    SplitCounts cnt{2, 0, 1, 0};
    DevBuf<HitEntry> d_hits(dev, 2);
    d_hits.up(hits);
    DevBuf<HitGroup> d_groups(dev, 1);
    d_groups.up({grp});
    DevBuf<SplitCounts> d_cnt(dev, 1);
    d_cnt.up({cnt});
    DevBuf<unsigned char> d_scr(dev, expert_scratch_bytes(T));
    const ExpertScratch scr = expert_scratch_carve(d_scr.p, T);
    dev.h2d(scr.xq, xq_perm.data(), xq_perm.size());
    dev.h2d(scr.xs, xs.data(), xs.size() * 4);
    DevBuf<float> d_parts(dev, (size_t) T * kTopK * kHidden);
    d_parts.zero();

    experts_gate_up(d_blob.p, d_hits.p, d_groups.p, d_cnt.p, T, scr.xq, scr.xs, scr.hq, scr.hs, dev.stream());
    dev.sync();
    std::vector<int8_t> hq_gpu_perm((size_t) 2 * kFF), hq_gpu((size_t) 2 * kFF);
    std::vector<float> hs_gpu(2 * kHBlocks);
    dev.d2h(hq_gpu_perm.data(), scr.hq, hq_gpu_perm.size());
    dev.d2h(hs_gpu.data(), scr.hs, hs_gpu.size() * 4);
    from_perm(hq_gpu_perm.data(), hq_gpu.data(), 2 * kHBlocks);
    double worst1 = 0.0, hmaxall = 0.0;
    for (int t = 0; t < 2; ++t) {
        std::vector<double> h_ref;
        expert_h_q(blob.data(), xq_nat.data() + (size_t) t * kHidden, xs.data() + (size_t) t * kActBlocks, (double) hits[t].weight, h_ref);
        for (int b = 0; b < kHBlocks; ++b) {
            double hmax = 0.0;
            for (int j = 0; j < 32; ++j) hmax = std::max(hmax, std::fabs(h_ref[32 * b + j]));
            hmaxall = std::max(hmaxall, hmax);
            const double d = hs_gpu[(size_t) t * kHBlocks + b];
            for (int j = 0; j < 32; ++j) {
                const double deq = (double) hq_gpu[(size_t) t * kFF + 32 * b + j] * d;
                const double allowed = 0.5001 * d + 4e-6 * hmax;
                worst1 = std::max(worst1, std::fabs(deq - h_ref[32 * b + j]) / (allowed > 0 ? allowed : 1e-30));
            }
        }
    }
    rep.line(worst1 <= 1.0 && hmaxall > 0.5, "edge e8m0 gate/up", "scale bytes 0/1/127/254/255 in 10 columns each (all rows), dequantised h within the bound (worst %.3f of 1), max|h| %.3g", worst1,
             hmaxall);

    // down, on the hand-made h
    dev.h2d(scr.hq, hq_perm.data(), hq_perm.size());
    dev.h2d(scr.hs, hs.data(), hs.size() * 4);
    experts_down(d_blob.p, d_hits.p, d_groups.p, d_cnt.p, T, scr.hq, scr.hs, d_parts.p, dev.stream());
    dev.sync();
    const auto parts = d_parts.down();
    double worst2 = 0.0, ymax = 0.0;
    for (int t = 0; t < 2; ++t) {
        std::vector<double> y, mass;
        expert_down_q(blob.data(), hq_nat.data() + (size_t) t * kFF, hs.data() + (size_t) t * kHBlocks, y, mass);
        const float* g = &parts[((size_t) t * kTopK + 0) * kHidden];
        for (int r = 0; r < kHidden; ++r) {
            worst2 = std::max(worst2, std::fabs((double) g[r] - y[r]) / (5e-6 * mass[r] + 1e-30));
            ymax = std::max(ymax, std::fabs(y[r]));
        }
    }
    rep.line(worst2 <= 1.0 && ymax > 0.1, "edge e8m0 down", "scale bytes 0/1/127/254/255 in 9 columns each (all rows), W2.h vs FP64: worst %.3f of the allowed, max|y| %.3g", worst2, ymax);
}

/// The activation quantiser's rule on the blocks that used to differ between implementations (CONTRACTS.md "Activations"): the kernels'
/// bytes against hand-written expectations (zero, 1e-37 everywhere, the 2^-100 boundary, ties to even, Inf and NaN at every position) and,
/// for random bit patterns and scaled mixtures, against ref::quantize_block, which is the CPU rule.  The emulation test also compares with
/// the CPU library itself (ds41_cuda_emu_test --quant).
struct QuantEdge {
    std::string what;
    std::vector<float> head;      // the first elements; the rest of the block is `fill`
    float fill;
    uint32_t d_bits;              // the expected scale, as bits
    std::vector<int> q;           // the expected int8 values from element 0 on (zeros after)
};
inline std::vector<QuantEdge> quant_edge_blocks() {
    using namespace ref;
    const float tiny = 1e-37f, inf = INFINITY, qnan = bits_f32(0x7FC00000u), nnan = bits_f32(0xFFC00000u), snan = bits_f32(0x7FA00000u);
    std::vector<QuantEdge> e;
    auto add = [&](const char* what, std::vector<float> head, float fill, uint32_t d_bits, std::vector<int> q) { e.push_back({what, head, fill, d_bits, q}); };
    add("all zero", {}, 0.0f, 0u, {});
    add("all -0.0", {}, -0.0f, 0u, {});
    add("1e-37 everywhere (127/amax overflows)", {}, tiny, 0u, {});
    add("-1e-37 everywhere", {}, -tiny, 0u, {});
    add("denormals", {1e-40f, -3e-39f, 1e-45f}, 0.0f, 0u, {});
    add("just below 2^-100", {bits_f32(0x0D7FFFFFu), -bits_f32(0x0D7FFFFFu)}, 0.0f, 0u, {});
    {
        const float a = bits_f32(0x0D800000u);
        add("exactly 2^-100", {a, -a, a * 0.5f}, 0.0f, f32_bits(a / 127.0f), {127, -127, 64});
    }
    add("ties to even [254, 5, 1, -5]", {254.0f, 5.0f, 1.0f, -5.0f}, 0.0f, f32_bits(2.0f), {127, 2, 0, -2});
    add("ties to even, odd side", {254.0f, 3.0f, -3.0f, 7.0f, -1.0f, 9.0f}, 0.0f, f32_bits(2.0f), {127, 2, -2, 4, 0, 4});
    add("NaN first", {qnan, 1.0f, 2.0f}, 0.5f, 0x7FC00000u, {});
    add("Inf first", {inf, 1.0f, 2.0f}, 0.5f, 0x7FC00000u, {});
    add("-Inf", {-inf}, 1.0f, 0x7FC00000u, {});
    add("negative NaN", {nnan}, 1.0f, 0x7FC00000u, {});
    add("signalling NaN", {snan}, 1.0f, 0x7FC00000u, {});
    add("NaN and Inf", {qnan, inf, -inf}, 3.0f, 0x7FC00000u, {});
    {
        const float big = 3.4028234663852886e38f;
        add("FLT_MAX", {big, -big, big * 0.25f}, 0.0f, f32_bits(big / 127.0f), {127, -127, 32});
    }
    for (int pos : {1, 7, 8, 15, 16, 24, 31}) {
        QuantEdge a{"NaN at one position", {}, 0.0f, 0x7FC00000u, {}};
        for (int j = 0; j < 32; ++j) a.head.push_back((float) (j - 15) * 3.0f);
        a.head[pos] = qnan;
        e.push_back(a);
        QuantEdge b = a;
        b.what = "Inf at one position";
        b.head[pos] = (pos & 1) ? inf : -inf;
        e.push_back(b);
    }
    return e;
}

inline void run_quant_edge(Dev& dev, ref::Rng& rng, Report& rep) {
    using namespace ref;
    const std::vector<QuantEdge> edges = quant_edge_blocks();
    const int T = 3;
    const int nblocks = T * kActBlocks;
    std::vector<float> x((size_t) T * kHidden);
    for (int b = 0; b < nblocks; ++b) {
        float* xb = &x[(size_t) b * 32];
        if (b < (int) edges.size()) {
            for (int j = 0; j < 32; ++j) xb[j] = j < (int) edges[b].head.size() ? edges[b].head[j] : edges[b].fill;
            continue;
        }
        const int kind = rng.range(0, 5);
        for (int j = 0; j < 32; ++j) {
            float v;
            switch (kind) {
                case 0: v = bits_f32(rng.u32()); break;                                                               // any bit pattern
                case 1: v = (float) (rng.normal() * std::ldexp(1.0, rng.range(-140, 100))); break;                    // all magnitudes, 2^-100 included
                case 2: v = (rng.range(0, 40) == 0) ? bits_f32(0x7F800000u | (rng.u32() & 0x807FFFFFu)) : (float) rng.normal(); break;   // rare Inf / NaN
                case 3: v = (float) rng.range(-300, 300) * 0.5f; break;                                               // integer and half-integer ties
                case 4: v = (float) (rng.normal() * std::ldexp(1.0, -100) * (1.0 + 1e-3 * rng.range(-5, 5))); break;  // straddling 2^-100
                default: v = (float) rng.normal(); break;
            }
            xb[j] = v;
        }
    }
    DevBuf<float> d_x(dev, x.size());
    d_x.up(x);
    DevBuf<int8_t> d_xq(dev, x.size());
    DevBuf<float> d_xs(dev, (size_t) nblocks);
    quantize_acts(d_x.p, T, d_xq.p, d_xs.p, dev.stream());
    dev.sync();
    const auto xq_perm = d_xq.down();
    const auto xs = d_xs.down();
    std::vector<int8_t> nat(x.size());
    from_perm(xq_perm.data(), nat.data(), nblocks);
    long bad_edge = 0, bad_ref = 0;
    std::string first_bad;
    for (int b = 0; b < nblocks; ++b) {
        int8_t rq[32];
        float rd;
        quantize_block(&x[(size_t) b * 32], rq, rd);
        bool same = std::memcmp(&rd, &xs[b], 4) == 0 && std::memcmp(rq, &nat[(size_t) b * 32], 32) == 0;
        bad_ref += !same;
        if (b < (int) edges.size()) {
            bool ok = f32_bits(xs[b]) == edges[b].d_bits;
            for (int j = 0; j < 32; ++j) ok = ok && nat[(size_t) b * 32 + j] == (j < (int) edges[b].q.size() ? edges[b].q[j] : 0);
            if (!ok) {
                ++bad_edge;
                if (first_bad.empty()) first_bad = fmt_str("block %d '%s': d %08x want %08x, q[0..3] %d %d %d %d", b, edges[b].what.c_str(), f32_bits(xs[b]), edges[b].d_bits,
                                                          nat[(size_t) b * 32], nat[(size_t) b * 32 + 1], nat[(size_t) b * 32 + 2], nat[(size_t) b * 32 + 3]);
            }
        }
    }
    rep.line(bad_edge == 0, "quantize: the contract's special blocks", "%zu hand-written blocks (zero, 1e-37, 2^-100 boundary, ties to even, NaN / Inf at every position): %ld differ%s%s",
             edges.size(), bad_edge, first_bad.empty() ? "" : "; first: ", first_bad.c_str());
    rep.line(bad_ref == 0, "quantize: random bit patterns", "%d blocks (random bit patterns, all magnitudes, rare Inf / NaN, ties) bit-identical to the CPU rule: %ld differ", nblocks,
             bad_ref);
}

/// NaN through the SwiGLU clamps.  A one-expert blob whose block 0 of every gate / up row holds the code 0 (value 0) or 7 (+12), and an
/// activation whose block-0 scale is +Inf:  zero row -> Inf * 0 = NaN;  +12 row -> +Inf, which the clamps turn into 10.  So
///   variant 0: gate rows zero, up rows +12   -> g = NaN, u = 10      variant 1: gate +12, up zero  -> g = 10, u = NaN
///   variant 2: a NaN in x itself (through the quantiser kernel; every g and u is NaN)
/// h must be NaN (scale NaN, int8 zero), and so y: for token 1 every element is NaN; token 0 (a finite control) stays finite.  A clamp that
/// returned its constant for a NaN (fminf / fmaxf) would give finite h and finite y.
inline void run_expert_nan(Dev& dev, ref::Rng& rng, Report& rep) {
    using namespace ref;
    const int T = 2;
    for (int variant = 0; variant < 3; ++variant) {
        std::vector<uint8_t> blob(kBlobBytes);
        gen_blob(blob.data(), rng);
        if (variant < 2)
            for (int r = 0; r < kFF; ++r) {
                uint8_t* g = blob.data() + kBlobGate + (size_t) r * kGateRowBytes;
                uint8_t* u = blob.data() + kBlobUp + (size_t) r * kGateRowBytes;
                g[0] = u[0] = 120;
                std::memset(g + 1, variant == 0 ? 0x00 : 0x77, 16);
                std::memset(u + 1, variant == 1 ? 0x00 : 0x77, 16);
            }
        DevBuf<uint8_t> d_blob(dev, kBlobBytes);
        d_blob.up(blob);
        HitEntry h0{0, 0, 0, 1.0f}, h1{1, 0, 0, 0.75f};
        HitGroup grp{};
        grp.slot = 0;
        grp.n = 2;
        grp.hit[0] = 0;
        grp.hit[1] = 1;
        DevBuf<HitEntry> d_hits(dev, 2);
        d_hits.up({h0, h1});
        DevBuf<HitGroup> d_groups(dev, 1);
        d_groups.up({grp});
        DevBuf<SplitCounts> d_cnt(dev, 1);
        d_cnt.up({SplitCounts{2, 0, 1, 0}});
        DevBuf<unsigned char> d_scr(dev, expert_scratch_bytes(T));
        const ExpertScratch scr = expert_scratch_carve(d_scr.p, T);
        DevBuf<float> d_parts(dev, (size_t) T * kTopK * kHidden);
        d_parts.zero();
        if (variant < 2) {
            std::vector<int8_t> q((size_t) T * kHidden, 1), qp((size_t) T * kHidden);
            std::vector<float> xs((size_t) T * kActBlocks, 0.01f);
            xs[kActBlocks] = INFINITY;                                    // token 1, block 0
            to_perm(q.data(), qp.data(), T * kActBlocks);
            dev.h2d(scr.xq, qp.data(), qp.size());
            dev.h2d(scr.xs, xs.data(), xs.size() * 4);
        } else {
            std::vector<float> x((size_t) T * kHidden);
            for (auto& v : x) v = (float) (0.5 * rng.normal());
            x[(size_t) kHidden + 77] = bits_f32(0x7FC00000u);
            DevBuf<float> d_x(dev, x.size());
            d_x.up(x);
            quantize_acts(d_x.p, T, scr.xq, scr.xs, dev.stream());
        }
        experts_gate_up(d_blob.p, d_hits.p, d_groups.p, d_cnt.p, T, scr.xq, scr.xs, scr.hq, scr.hs, dev.stream());
        experts_down(d_blob.p, d_hits.p, d_groups.p, d_cnt.p, T, scr.hq, scr.hs, d_parts.p, dev.stream());
        dev.sync();
        std::vector<float> hs_all((size_t) kTopK * T * kHBlocks);
        dev.d2h(hs_all.data(), scr.hs, hs_all.size() * 4);
        const auto parts = d_parts.down();
        int nan_scale = 0, finite_scale = 0, nan1 = 0, nan0 = 0;
        for (int b = 0; b < kHBlocks; ++b) {
            nan_scale += f32_bits(hs_all[(size_t) 1 * kHBlocks + b]) == 0x7FC00000u;       // hit index 1 = token 1
            finite_scale += std::isfinite(hs_all[b]);                                     // hit index 0 = token 0
        }
        for (int r = 0; r < kHidden; ++r) {
            nan0 += std::isnan(parts[((size_t) 0 * kTopK + 0) * kHidden + r]);
            nan1 += std::isnan(parts[((size_t) 1 * kTopK + 0) * kHidden + r]);
        }
        const bool ok = nan_scale == kHBlocks && finite_scale == kHBlocks && nan1 == kHidden && nan0 == 0;
        rep.line(ok, fmt_str("NaN through the clamps: %s", variant == 0 ? "g only" : variant == 1 ? "u only" : "x").c_str(),
                 "token 1: %d of %d h scales are the canonical NaN, %d of %d y elements NaN; the finite control token 0: %d of %d h scales finite, %d NaN in y", nan_scale, kHBlocks, nan1,
                 kHidden, finite_scale, kHBlocks, nan0);
    }
}

inline void run_expert_timing(ExpertRig& rig, int reps, Report& rep) {
    using namespace ref;
    (void) rep;
    Dev& dev = rig.dev;
    for (int NT : {1, 2, 4, 8}) {
        const ExpertKernelInfo g = expert_kernel_info(NT, false), d = expert_kernel_info(NT, true);
        Report::info("kernel NT=%d: gate/up %d regs, %d B dynamic smem, %d blocks/SM (%d warps/SM);  down %d regs, %d B dynamic smem, %d blocks/SM (%d warps/SM)", NT, g.regs,
                     g.dyn_smem, g.blocks_per_sm, 8 * g.blocks_per_sm, d.regs, d.dyn_smem, d.blocks_per_sm, 8 * d.blocks_per_sm);
    }
    struct Sc {
        int T;
        int hits;
    };
    for (Sc sc : {Sc{1, 6}, Sc{4, 24}, Sc{8, 48}}) {
        if (sc.hits > rig.slots) {
            Report::info("timing: T=%d needs %d slots (have %d), skipped", sc.T, sc.hits, rig.slots);
            continue;
        }
        const int T = sc.T;
        std::vector<int32_t> ids((size_t) T * kTopK);
        std::vector<float> wts((size_t) T * kTopK, 0.25f);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < kTopK; ++k) ids[(size_t) t * kTopK + k] = rig.pool[t * kTopK + k];       // all distinct experts: sc.hits distinct slots
        std::vector<float> x((size_t) T * kHidden);
        Rng rng(99);
        for (auto& v : x) v = (float) rng.normal();
        DevBuf<float> d_x(dev, x.size());
        d_x.up(x);
        DevBuf<int32_t> d_ids(dev, ids.size());
        d_ids.up(ids);
        DevBuf<float> d_w(dev, wts.size());
        d_w.up(wts);
        DevBuf<HitEntry> d_hits(dev, ids.size());
        DevBuf<MissEntry> d_miss(dev, ids.size());
        DevBuf<HitGroup> d_groups(dev, ids.size());
        DevBuf<SplitCounts> d_counts(dev, 1);
        MappedBuf<SplitHostRecord> m_rec(dev, 1);
        MappedBuf<MissEntry> m_miss(dev, ids.size());
        DevBuf<unsigned char> d_scr(dev, expert_scratch_bytes(T));
        const ExpertScratch scr = expert_scratch_carve(d_scr.p, T);
        DevBuf<float> d_parts(dev, (size_t) T * kTopK * kHidden);
        split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.slots, rig.layer, d_hits.p, d_miss.p, d_groups.p, d_counts.p, nullptr, 0, dev.stream());
        dev.sync();
        const double bytes = (double) sc.hits * (double) kBlobBytes;
        const double t_all = dev.time_us([&] { experts_hits(rig.cache->p, d_x.p, d_hits.p, d_groups.p, d_counts.p, T, scr, d_parts.p, dev.stream()); }, reps);
        const double t_q = dev.time_us([&] { quantize_acts(d_x.p, T, scr.xq, scr.xs, dev.stream()); }, reps);
        const double t_gu = dev.time_us([&] { experts_gate_up(rig.cache->p, d_hits.p, d_groups.p, d_counts.p, T, scr.xq, scr.xs, scr.hq, scr.hs, dev.stream()); }, reps);
        const double t_dn = dev.time_us([&] { experts_down(rig.cache->p, d_hits.p, d_groups.p, d_counts.p, T, scr.hq, scr.hs, d_parts.p, dev.stream()); }, reps);
        const double t_split = dev.time_us([&] { split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.slots, rig.layer, d_hits.p, d_miss.p, d_groups.p, d_counts.p, nullptr, 0, dev.stream()); }, reps);
        uint32_t timing_seq = 0;
        const double t_split_host = dev.time_us([&] {       // + a host record: the fence and the release store of the doorbell
            timing_seq = split_next_seq(timing_seq);
            split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.slots, rig.layer, d_hits.p, m_miss.p, d_groups.p, d_counts.p, m_rec.p, timing_seq, dev.stream());
        }, reps);
        Report::info("timing T=%d, %d hits (%d distinct experts, %.1f MB of blobs): layer experts %.1f us = %.0f GB/s (%.0f%% of 900);  gate/up %.1f us (%.0f GB/s of %.1f MB), down %.1f us (%.0f GB/s of %.1f MB), quantise x %.1f us, split %.1f us (%.1f us with a host record: fence + release doorbell)",
                     T, sc.hits, sc.hits, bytes / 1e6, t_all, bytes / (t_all * 1e3), 100.0 * bytes / (t_all * 1e3) / 900.0, t_gu, sc.hits * 2.0 * kGateBytes / (t_gu * 1e3),
                     sc.hits * 2.0 * kGateBytes / 1e6, t_dn, sc.hits * (double) kDownBytes / (t_dn * 1e3), sc.hits * (double) kDownBytes / 1e6, t_q, t_split, t_split_host);
    }
}

inline int run_expert_parity(Dev& dev, const ExpertOpts& o, Report& rep) {
    using namespace ref;
    Rng rng(o.seed);
    ExpertRig rig(dev, o.slots, o.seed);
    const int P = std::min(o.slots, 16);
    if (o.full_cases) {
        for (int T : o.Ts) {
            const int misses = (T % 3);                                    // 0, 1 or 2 misses per token
            CaseSpec cs = make_case(fmt_str("experts T=%d (pool %d, %d miss/token)", T, P, misses), T, 0, P, misses, rng);
            run_expert_case(rig, cs, rng, o.max_hits_checked, rep);
        }
        {   // the SAME expert for two tokens, and for all of them
            CaseSpec cs;
            cs.name = "experts T=3 same expert for several tokens";
            cs.T = 3;
            cs.pool_idx = {{0, 1, 2, 3, 4, 5}, {0, 1, 6, 7, 8, 9}, {0, 2, 6, 10, -1, -2}};
            fit_pool(cs, P);
            run_expert_case(rig, cs, rng, o.max_hits_checked, rep);
        }
        {
            CaseSpec cs;
            cs.name = "experts T=8 one expert shared by all 8 tokens";
            cs.T = 8;
            for (int t = 0; t < 8; ++t) cs.pool_idx.push_back({0, 1 + t % (P - 1), 1 + (t + 3) % (P - 1), -1 - t, -20 - t, -40 - t});
            fit_pool(cs, P);
            run_expert_case(rig, cs, rng, o.max_hits_checked, rep);
        }
        {   // groups of 6, 5 and 4 tokens: the boundary between the lean (n <= 4) and the wide (n >= 5) specialisation
            CaseSpec cs;
            cs.name = "experts T=6 groups of 6, 5 and 4 tokens";
            cs.T = 6;
            for (int t = 0; t < 6; ++t) {
                std::vector<int> row = {0, t < 5 ? 1 : 3 + t, t < 4 ? 2 : 4 + t, 5 + t, -1 - t, -10 - t};
                cs.pool_idx.push_back(row);
            }
            fit_pool(cs, P);
            run_expert_case(rig, cs, rng, o.max_hits_checked, rep);
        }
        {
            CaseSpec cs;
            cs.name = "experts T=2 all misses";
            cs.T = 2;
            cs.pool_idx = {{-1, -2, -3, -4, -5, -6}, {-7, -8, -9, -10, -11, -12}};
            run_expert_case(rig, cs, rng, o.max_hits_checked, rep);
        }
    }
    if (o.edge) {
        run_expert_edge(dev, rng, rep);
        run_expert_nan(dev, rng, rep);
        run_quant_edge(dev, rng, rep);
    }
    if (o.timing_reps > 0 && !dev.is_emulation()) run_expert_timing(rig, o.timing_reps, rep);
    return rep.failures == 0 ? 0 : 1;
}

}  // namespace strata::ds41::cuda::parity
