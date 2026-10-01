// src/ds41/cuda/ds41_parity_lib.hpp - DS-D: the parity drivers (router, hit/miss split, hit experts) behind the V100 programs and
// the CPU emulation test.  Header only; the device is reached through the small `Dev` interface, so the same checks run
//   * on a V100: ds41_router_parity / ds41_split_parity / ds41_expert_parity (src/ds41/cuda/ds41_*_parity.cpp, cudaMalloc + events),
//   * on the host: ds41_cuda_emu_test (the kernels compiled for the CPU, "device memory" = malloc, see ds41_emu.hpp).
//
// Output: one line per check, "PASS <name>: <numbers>" or "FAIL <name>: <numbers>"; near-ties and informational numbers are
// printed with "INFO"; the summary line is "ALL PASS (n checks)" or "FAILED (k of n)".  Exit status of the programs: 0 iff all pass.
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
};

inline int run_router_parity(Dev& dev, const RouterOpts& o, Report& rep) {
    using namespace ref;
    Rng rng(o.seed);
    const int Tmax = *std::max_element(o.Ts.begin(), o.Ts.end());

    std::vector<uint16_t> wg((size_t) kExperts * kHidden);
    std::vector<double> wgd(wg.size());
    for (size_t i = 0; i < wg.size(); ++i) {
        wg[i] = f32_to_bf16((float) (rng.normal() * 0.02));
        wgd[i] = (double) bf16_to_f32(wg[i]);
    }
    std::vector<float> bias(kExperts);
    for (auto& b : bias) b = (float) (rng.normal() * 0.1);

    // a pool of Tmax tokens of four kinds: ordinary, with outlier channels, small, and big (logits of +-100: the softplus linear branch)
    std::vector<float> xpool((size_t) Tmax * kHidden);
    for (int t = 0; t < Tmax; ++t) {
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
};

inline int run_split_parity(Dev& dev, const SplitOpts& o, Report& rep) {
    using namespace ref;
    Rng rng(o.seed);
    for (int variant = 0; variant < 5; ++variant) {
        // residency table [40][384]: variant 0 = ~40 % resident, 1 = nothing resident, 2 = everything, 3 = a handful, 4 = ~40 % + a hot expert
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
                    default: in = e == 100 || rng.uniform() < 0.4; break;
                }
                if (in) res[(size_t) l * kExperts + e] = (next++ * 7 + l) % 1000;
            }
        }
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
            split_hits_misses(d_ids.p, d_w.p, T, d_res.p, layer, d_hits.p, d_miss.p, grouped ? d_groups.p : nullptr, d_counts.p, (variant & 1) != 0,
                              dev.stream());
            dev.sync();
            const SplitCounts c = d_counts.down(1)[0];
            const auto hits = d_hits.down(c.n_hits > 0 ? c.n_hits : 1);
            const auto miss = d_miss.down(c.n_misses > 0 ? c.n_misses : 1);

            // host reference
            std::vector<HitEntry> eh;
            std::vector<MissEntry> em;
            for (int i = 0; i < T * kTopK; ++i) {
                const int id = ids[i];
                const int slot = (id >= 0 && id < kExperts) ? res[(size_t) layer * kExperts + id] : -1;
                if (slot >= 0) eh.push_back({i / kTopK, i % kTopK, slot, w[i]});
                else em.push_back({i / kTopK, i % kTopK, id, w[i]});
            }
            bool ok = c.n_hits == (int) eh.size() && c.n_misses == (int) em.size();
            for (size_t i = 0; ok && i < eh.size(); ++i)
                ok = hits[i].token == eh[i].token && hits[i].k == eh[i].k && hits[i].slot == eh[i].slot && hits[i].weight == eh[i].weight;
            for (size_t i = 0; ok && i < em.size(); ++i)
                ok = miss[i].token == em[i].token && miss[i].k == em[i].k && miss[i].expert == em[i].expert && miss[i].weight == em[i].weight;
            rep.line(ok, fmt_str("split lists variant=%d T=%d", variant, T).c_str(), "%d hits, %d misses (expect %zu, %zu), order and fields identical: %s", c.n_hits,
                     c.n_misses, eh.size(), em.size(), ok ? "yes" : "NO");
            if (grouped) {
                // expected groups: by decreasing size, ties by first appearance of the slot
                std::vector<HitGroup> eg;
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
    DevBuf<MissEntry> d_miss(dev, ids.size());
    DevBuf<HitGroup> d_groups(dev, ids.size());
    DevBuf<SplitCounts> d_counts(dev, 1);
    DevBuf<unsigned char> d_scr(dev, expert_scratch_bytes(T));
    const ExpertScratch scr = expert_scratch_carve(d_scr.p, T);
    DevBuf<float> d_parts(dev, (size_t) T * kTopK * kHidden);
    {   // the sentinel: the quiet NaN 0x7FC00001 everywhere; a miss row must still hold it after the kernels ran
        std::vector<uint32_t> sent(d_parts.n, 0x7FC00001u);
        dev.h2d(d_parts.p, sent.data(), sent.size() * 4);
    }

    split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.layer, d_hits.p, d_miss.p, d_groups.p, d_counts.p, false, dev.stream());
    experts_hits(rig.cache->p, d_x.p, d_hits.p, d_groups.p, d_counts.p, T, scr, d_parts.p, dev.stream());
    dev.sync();

    const SplitCounts c = d_counts.down(1)[0];
    const auto hits = d_hits.down(c.n_hits > 0 ? c.n_hits : 1);
    const auto miss = d_miss.down(c.n_misses > 0 ? c.n_misses : 1);
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
        rep.line(ok, (cs.name + " split").c_str(), "%d hits in %d groups (largest %d tokens), %d misses (expect %d hits)", c.n_hits, c.n_groups, biggest, c.n_misses,
                 expect_hits);
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
        DevBuf<unsigned char> d_scr(dev, expert_scratch_bytes(T));
        const ExpertScratch scr = expert_scratch_carve(d_scr.p, T);
        DevBuf<float> d_parts(dev, (size_t) T * kTopK * kHidden);
        split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.layer, d_hits.p, d_miss.p, d_groups.p, d_counts.p, false, dev.stream());
        dev.sync();
        const double bytes = (double) sc.hits * (double) kBlobBytes;
        const double t_all = dev.time_us([&] { experts_hits(rig.cache->p, d_x.p, d_hits.p, d_groups.p, d_counts.p, T, scr, d_parts.p, dev.stream()); }, reps);
        const double t_q = dev.time_us([&] { quantize_acts(d_x.p, T, scr.xq, scr.xs, dev.stream()); }, reps);
        const double t_gu = dev.time_us([&] { experts_gate_up(rig.cache->p, d_hits.p, d_groups.p, d_counts.p, T, scr.xq, scr.xs, scr.hq, scr.hs, dev.stream()); }, reps);
        const double t_dn = dev.time_us([&] { experts_down(rig.cache->p, d_hits.p, d_groups.p, d_counts.p, T, scr.hq, scr.hs, d_parts.p, dev.stream()); }, reps);
        const double t_split = dev.time_us([&] { split_hits_misses(d_ids.p, d_w.p, T, rig.residency->p, rig.layer, d_hits.p, d_miss.p, d_groups.p, d_counts.p, false, dev.stream()); }, reps);
        Report::info("timing T=%d, %d hits (%d distinct experts, %.1f MB of blobs): layer experts %.1f us = %.0f GB/s (%.0f%% of 900);  gate/up %.1f us (%.0f GB/s of %.1f MB), down %.1f us (%.0f GB/s of %.1f MB), quantise x %.1f us, split %.1f us",
                     T, sc.hits, sc.hits, bytes / 1e6, t_all, bytes / (t_all * 1e3), 100.0 * bytes / (t_all * 1e3) / 900.0, t_gu, sc.hits * 2.0 * kGateBytes / (t_gu * 1e3),
                     sc.hits * 2.0 * kGateBytes / 1e6, t_dn, sc.hits * (double) kDownBytes / (t_dn * 1e3), sc.hits * (double) kDownBytes / 1e6, t_q, t_split);
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
    if (o.edge) run_expert_edge(dev, rng, rep);
    if (o.timing_reps > 0 && !dev.is_emulation()) run_expert_timing(rig, o.timing_reps, rep);
    return rep.failures == 0 ? 0 : 1;
}

}  // namespace strata::ds41::cuda::parity
