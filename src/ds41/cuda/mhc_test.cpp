// src/ds41/cuda/mhc_emu_test.cpp - DS1-D: the mHC kernels (src/ds41/cuda/mhc_impl.cuh, the SAME source nvcc compiles for sm_70) run on the CPU through the
// thread-model emulation (ds41_emu.hpp), against the NumPy oracle's golden data (src/ds41/engram/golden/gen_golden.py: ref/ds41/mhc.py run in FLOAT64 and in
// FLOAT32) at MiniGeom's and RealGeom's shapes.  No GPU needed.
//
//   ds41_mhc_emu_test --golden DIR [--suite NAME ...] [--order forward|reverse|shuffle[:SEED]]
//
// Suites (default: all):
//   mixes    hc_mixes (GEMV over the flattened stream, rsqrt scale, split + Sinkhorn) against the FLOAT64 oracle, two parameter regimes, T = 5 tokens.
//   split    the split / Sinkhorn alone on mixes of every magnitude (saturated sigmoids and softmaxes, zeros, +-80 patterns).
//   prepost  hc_pre / hc_post: bit-identical to the oracle's FLOAT32 evaluation (both evaluate in the same order) and close to FLOAT64; in-place == out of place.
//   chain    the per-sub-layer API (HcRecords: the one-block lag, the identity pre at block 0, the head fold) in model.py Model.block order against the oracle.
//   tinv     T-invariance: a token's coefficients are bit-identical whatever T it is computed with (T = 1 .. 8, 9, 17: one and several blocks along y).
//   edge     NaN propagation, zero streams, the identity record, expand, argument checks (alignment, scratch size, overlapping in / out).
// TOLERANCES (documented, from the oracle's own FLOAT32 noise): the oracle evaluated in float32 differs from the float64 evaluation by e32 (the golden carries both).  The
// kernel must be within max(4 * e32, floor) of the float64 value, floor = 4e-7 * max|ref| for sums over the stream, 1e-6 for the (bounded, <= 2) coefficients.  The kernel's sums have
// a different (fixed) order from numpy's pairwise ones, so it is not bit-identical there; hc_pre / hc_post, whose order is numpy's, ARE.  The scheduling order (--order) changes no bit.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ds41_emu.hpp"
#include "mhc_test_util.hpp"
#include "strata/ds41/cuda/mhc.hpp"

using namespace strata::ds41;
using namespace strata::ds41::cuda;
using namespace ds1d;

namespace {

struct Ctx {
    HostDev dev;
    Report rep;
    Golden* gold = nullptr;
};

template <class T>
struct Up {                                   // a golden array on the "device" (guard-zone checked memory)
    DevBuf<T> b;
    Up(Dev& d, const Arr<T>& a) : b(d, a.size()) { b.up(a.v); }
    Up(Dev& d, const std::vector<T>& v) : b(d, v.size()) { b.up(v); }
    T* p() const { return b.p; }
};

const char* geom_name(const MiniGeom*) { return "mini"; }
const char* geom_name(const RealGeom*) { return "real"; }
template <class G> const char* gname() { return geom_name((const G*) nullptr); }

double tol_of(const Err& oracle32, double floor_abs) { return std::max(4.0 * oracle32.max_abs, floor_abs); }

// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
void suite_mixes(Ctx& c) {
    constexpr int K = Derived<G>::kHcFlat, NR = G::kHcMixes;
    for (const char* cs : {"a", "b"}) {
        const std::string d = std::string("mhc_") + gname<G>() + "/" + cs + "/";
        const auto x = c.gold->get<float>(d + "x"), fn = c.gold->get<float>(d + "fn"), scale = c.gold->get<float>(d + "scale"), base = c.gold->get<float>(d + "base");
        const auto m64 = c.gold->get<double>(d + "mixes64"), c64 = c.gold->get<double>(d + "coef64");
        const auto m32 = c.gold->get<float>(d + "mixes32o"), c32 = c.gold->get<float>(d + "coef32o");
        const int T = (int) x.dim(0);
        Up<float> dx(c.dev, x), dfn(c.dev, fn), dsc(c.dev, scale), dba(c.dev, base);
        DevBuf<float> coef(c.dev, (size_t) T * NR), mixes(c.dev, (size_t) T * NR);
        DevBuf<unsigned char> ws(c.dev, hc_scratch_bytes<G>(T));
        HcWeights w{dfn.p(), dsc.p(), dba.p()};
        ds41_hc_mixes<G>(c.dev, dx.p(), w, T, HcParams{}, coef.p, ws.p, hc_scratch_bytes<G>(T), mixes.p);
        const auto gm = mixes.down(), gc = coef.down();
        const Err em = compare(gm.data(), m64.data(), gm.size()), em32 = compare(m32.data(), m64.data(), gm.size());
        const Err ec = compare(gc.data(), c64.data(), gc.size()), ec32 = compare(c32.data(), c64.data(), gc.size());
        const double tm = tol_of(em32, 4e-7 * em.max_ref), tc = tol_of(ec32, 1e-6);
        c.rep.check(em.nan_mismatch == 0 && em.max_abs <= tm, fmt("mixes  %s/%s  T=%d K=%d", gname<G>(), cs, T, K),
                    fmt("%s  (tol %.2e; oracle fp32 %.2e)", describe(em).c_str(), tm, em32.max_abs));
        c.rep.check(ec.nan_mismatch == 0 && ec.max_abs <= tc, fmt("coef   %s/%s  pre/post/comb", gname<G>(), cs),
                    fmt("%s  (tol %.2e; oracle fp32 %.2e)", describe(ec).c_str(), tc, ec32.max_abs));
        // structure of the record: the last operation of the split is the column normalisation, so every column of comb sums to 1 up to eps
        double worst = 0;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < G::kHc; ++k) {
                double s = 0;
                for (int j = 0; j < G::kHc; ++j) s += gc[(size_t) t * NR + 2 * G::kHc + j * G::kHc + k];
                worst = std::max(worst, std::fabs(s - 1.0));
            }
        c.rep.check(worst < 1e-5, fmt("comb   %s/%s  column sums = 1", gname<G>(), cs), fmt("max |colsum - 1| = %.2e", worst));
    }
}

template <class G>
void suite_split(Ctx& c) {
    constexpr int NR = G::kHcMixes;
    const std::string d = std::string("split_") + gname<G>() + "/";
    const auto mx = c.gold->get<float>(d + "mixes"), scale = c.gold->get<float>(d + "scale"), base = c.gold->get<float>(d + "base");
    const auto c64 = c.gold->get<double>(d + "coef64");
    const auto c32 = c.gold->get<float>(d + "coef32o");
    const int N = (int) mx.dim(0);
    Up<float> dm(c.dev, mx), dsc(c.dev, scale), dba(c.dev, base);
    DevBuf<float> coef(c.dev, (size_t) N * NR);
    ds41_hc_split<G>(c.dev, dm.p(), dsc.p(), dba.p(), N, HcParams{}, coef.p);
    const auto got = coef.down();
    const Err e = compare(got.data(), c64.data(), got.size()), e32 = compare(c32.data(), c64.data(), got.size());
    const double tol = tol_of(e32, 2e-6);
    c.rep.check(e.nan_mismatch == 0 && e.max_abs <= tol, fmt("split  %s  %d rows of mixes (|m| up to 1000, zeros, +-80)", gname<G>(), N),
                fmt("%s  (tol %.2e; oracle fp32 %.2e)", describe(e).c_str(), tol, e32.max_abs));
    // every row finite, pre in (0, 1 + eps], post in [0, 2], comb in [0, 1]
    bool ok = true;
    for (int t = 0; t < N; ++t)
        for (int i = 0; i < NR; ++i) {
            const float v = got[(size_t) t * NR + i];
            const float hi = i < G::kHc ? 1.0f + 2e-6f : (i < 2 * G::kHc ? 2.0f : 1.0f + 1e-5f);
            ok = ok && std::isfinite(v) && v >= 0.0f && v <= hi;
        }
    c.rep.check(ok, fmt("split  %s  ranges: pre <= 1 + eps, post <= 2, comb <= 1, all finite", gname<G>()));
}

// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
void suite_prepost(Ctx& c) {
    constexpr int NR = G::kHcMixes, H = G::kHidden, HC = G::kHc;
    for (const char* cs : {"a", "b"}) {
        const std::string d = std::string("mhc_") + gname<G>() + "/" + cs + "/";
        const auto x = c.gold->get<float>(d + "x"), f = c.gold->get<float>(d + "f"), coef = c.gold->get<float>(d + "coef32");
        const auto y64 = c.gold->get<double>(d + "y_pre64");
        const auto y32 = c.gold->get<float>(d + "y_pre32o");
        const auto p64 = c.gold->get<double>(d + "post64");
        const auto p32 = c.gold->get<float>(d + "post32o");
        const int T = (int) x.dim(0);
        Up<float> dx(c.dev, x), df(c.dev, f), dcoef(c.dev, coef);
        DevBuf<float> y(c.dev, (size_t) T * H), out(c.dev, (size_t) T * HC * H), inplace(c.dev, (size_t) T * HC * H);
        ds41_hc_pre<G>(c.dev, dx.p(), dcoef.p(), T, y.p);
        const auto gy = y.down();
        const Err ey = compare(gy.data(), y64.data(), gy.size()), ey32 = compare(y32.data(), y64.data(), gy.size());
        c.rep.check(first_bit_diff(gy.data(), y32.data(), gy.size()) == gy.size(), fmt("hc_pre  %s/%s  bit-identical to the oracle's float32", gname<G>(), cs));
        c.rep.check(ey.max_abs <= tol_of(ey32, 2e-7 * ey.max_ref), fmt("hc_pre  %s/%s  vs float64", gname<G>(), cs), describe(ey));
        ds41_hc_post<G>(c.dev, df.p(), dx.p(), dcoef.p(), T, out.p);
        const auto go = out.down();
        const Err eo = compare(go.data(), p64.data(), go.size()), eo32 = compare(p32.data(), p64.data(), go.size());
        c.rep.check(first_bit_diff(go.data(), p32.data(), go.size()) == go.size(), fmt("hc_post %s/%s  bit-identical to the oracle's float32", gname<G>(), cs));
        c.rep.check(eo.max_abs <= tol_of(eo32, 2e-7 * eo.max_ref), fmt("hc_post %s/%s  vs float64", gname<G>(), cs), describe(eo));
        // in place: out == res must give the same bits as the separate output
        c.dev.d2d_async(inplace.p, dx.p(), (size_t) T * HC * H * sizeof(float), nullptr);
        ds41_hc_post<G>(c.dev, df.p(), inplace.p, dcoef.p(), T, inplace.p);
        const auto gi = inplace.down();
        c.rep.check(first_bit_diff(gi.data(), go.data(), go.size()) == go.size(), fmt("hc_post %s/%s  in place (out == res) == out of place", gname<G>(), cs));
        (void) NR;
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// chain: model.py Model.block order through the per-sub-layer API, with elementwise toy sub-layers (exact in binary: the oracle runs the same functions)
template <class G>
void toy(float* y, size_t n, float a, float b) {
    for (size_t i = 0; i < n; ++i) y[i] = a * y[i] + b;           // a, b powers of two: exact; the emulation's device memory is host memory
}

template <class G>
std::vector<float> run_chain(Ctx& c, const Arr<float>& x0, const Arr<float>& fn, const Arr<float>& scale, const Arr<float>& base, int L, int T, std::vector<float>* final_out,
                             std::vector<float>* pre_trace = nullptr) {
    constexpr int NR = G::kHcMixes, H = G::kHidden, HC = G::kHc, K = Derived<G>::kHcFlat;
    Up<float> dx(c.dev, x0);
    DevBuf<float> y(c.dev, (size_t) T * H), h(c.dev, (size_t) T * H);
    DevBuf<unsigned char> ws(c.dev, hc_scratch_bytes<G>(T)), recmem(c.dev, hc_records_bytes<G>(T));
    HcRecords<G> rec(recmem.p, T);
    ds41_hc_begin<G>(c.dev, rec, T);
    for (int l = 0; l < L; ++l) {
        for (int sub = 0; sub < 2; ++sub) {
            const int i = 2 * l + sub;
            DevBuf<float> dfn(c.dev, (size_t) NR * K), dsc(c.dev, 3), dba(c.dev, NR);
            std::vector<float> fnv(fn.v.begin() + (size_t) i * NR * K, fn.v.begin() + (size_t) (i + 1) * NR * K);
            dfn.up(fnv);
            dsc.up(std::vector<float>(scale.v.begin() + 3 * i, scale.v.begin() + 3 * i + 3));
            dba.up(std::vector<float>(base.v.begin() + (size_t) NR * i, base.v.begin() + (size_t) NR * (i + 1)));
            HcWeights w{dfn.p, dsc.p, dba.p};
            if (sub == 0) {
                ds41_hc_attn_in<G>(c.dev, dx.p(), w, T, HcParams{}, rec, y.p, ws.p, hc_scratch_bytes<G>(T));
                toy<G>(y.p, (size_t) T * H, 0.5f, 0.0f);
                ds41_hc_attn_out<G>(c.dev, y.p, dx.p(), T, rec);
            } else {
                ds41_hc_ffn_in<G>(c.dev, dx.p(), w, T, HcParams{}, rec, y.p, ws.p, hc_scratch_bytes<G>(T));
                toy<G>(y.p, (size_t) T * H, 0.25f, 0.5f);
                ds41_hc_ffn_out<G>(c.dev, y.p, dx.p(), T, rec);
            }
        }
        if (pre_trace) {                                       // the `pre_mix.L` trace stage: the first kHc floats of every token's record
            std::vector<float> rv((size_t) T * NR);
            c.dev.d2h(rv.data(), rec.lag, rv.size() * sizeof(float));
            for (int t = 0; t < T; ++t)
                for (int k = 0; k < HC; ++k) pre_trace->push_back(rv[(size_t) t * NR + k]);
        }
    }
    ds41_hc_head_fold<G>(c.dev, dx.p(), rec, T, h.p);
    if (final_out) *final_out = h.down();
    std::vector<float> xs((size_t) T * HC * H);
    c.dev.d2h(xs.data(), dx.p(), xs.size() * sizeof(float));
    return xs;
}

template <class G>
void suite_chain(Ctx& c) {
    const std::string d = std::string("chain_") + gname<G>() + "/";
    const auto x0 = c.gold->get<float>(d + "x0"), fn = c.gold->get<float>(d + "fn"), scale = c.gold->get<float>(d + "scale"), base = c.gold->get<float>(d + "base");
    const auto s64 = c.gold->get<double>(d + "stream64"), f64 = c.gold->get<double>(d + "final64");
    const auto s32 = c.gold->get<float>(d + "stream32o"), f32 = c.gold->get<float>(d + "final32o");
    const auto par = c.gold->get<int64_t>(d + "params");
    const int L = (int) par[0], T = (int) par[1];
    std::vector<float> fin, pre_trace;
    const auto xs = run_chain<G>(c, x0, fn, scale, base, L, T, &fin, &pre_trace);
    const Err es = compare(xs.data(), s64.data(), xs.size()), es32 = compare(s32.data(), s64.data(), xs.size());
    const Err ef = compare(fin.data(), f64.data(), fin.size()), ef32 = compare(f32.data(), f64.data(), fin.size());
    c.rep.check(es.nan_mismatch == 0 && es.max_abs <= tol_of(es32, 1e-6 * es.max_ref), fmt("chain  %s  %d blocks x 2 sub-layers, T=%d: stream after the last block", gname<G>(), L, T),
                fmt("%s  (oracle fp32 %.2e)", describe(es).c_str(), es32.max_abs));
    c.rep.check(ef.nan_mismatch == 0 && ef.max_abs <= tol_of(ef32, 1e-6 * ef.max_ref), fmt("chain  %s  final head fold (the last FFN's pre)", gname<G>()),
                fmt("%s  (oracle fp32 %.2e)", describe(ef).c_str(), ef32.max_abs));
    c.rep.check((int) pre_trace.size() == L * T * G::kHc, fmt("chain  %s  pre_mix trace has %d blocks", gname<G>(), L));
    // the lag is what the chain tests: with the lag broken (block 0 attention collapsing with an FFN pre, or the head fold with the attention's pre) the result differs by far more
    // than the tolerance; check that the tolerance is tight enough to see it by comparing against the float32 oracle of a block-0-only run
    // T-invariance of the whole chain: every token alone gives the same bits as in the batch
    bool same = true;
    for (int t = 0; t < T && same; ++t) {
        Arr<float> x1;
        x1.shape = {1, (size_t) G::kHc, (size_t) G::kHidden};
        x1.v.assign(x0.v.begin() + (size_t) t * G::kHc * G::kHidden, x0.v.begin() + (size_t) (t + 1) * G::kHc * G::kHidden);
        std::vector<float> fin1;
        const auto xs1 = run_chain<G>(c, x1, fn, scale, base, L, 1, &fin1);
        same = same && first_bit_diff(xs1.data(), xs.data() + (size_t) t * G::kHc * G::kHidden, xs1.size()) == xs1.size() &&
               first_bit_diff(fin1.data(), fin.data() + (size_t) t * G::kHidden, fin1.size()) == fin1.size();
    }
    c.rep.check(same, fmt("chain  %s  T-invariant: each token alone == in the batch of %d (bit-identical)", gname<G>(), T));
}

// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
std::vector<float> mixes_of(Ctx& c, const std::vector<float>& xs_tokens, const Up<float>& fn, const Up<float>& sc, const Up<float>& ba, int T) {
    constexpr int NR = G::kHcMixes;
    DevBuf<float> x(c.dev, xs_tokens.size()), coef(c.dev, (size_t) T * NR);
    x.up(xs_tokens);
    DevBuf<unsigned char> ws(c.dev, hc_scratch_bytes<G>(T));
    ds41_hc_mixes<G>(c.dev, x.p, HcWeights{fn.p(), sc.p(), ba.p()}, T, HcParams{}, coef.p, ws.p, hc_scratch_bytes<G>(T));
    return coef.down();
}

template <class G>
void suite_tinv(Ctx& c) {
    constexpr int NR = G::kHcMixes, TOK = G::kHc * G::kHidden;
    const std::string d = std::string("mhc_") + gname<G>() + "/b/";                 // the stress regime
    const auto x = c.gold->get<float>(d + "x"), fn = c.gold->get<float>(d + "fn"), scale = c.gold->get<float>(d + "scale"), base = c.gold->get<float>(d + "base");
    const int NG = (int) x.dim(0);
    Up<float> dfn(c.dev, fn), dsc(c.dev, scale), dba(c.dev, base);
    // reference: every token alone
    std::vector<std::vector<float>> alone(NG);
    for (int t = 0; t < NG; ++t) alone[t] = mixes_of<G>(c, std::vector<float>(x.v.begin() + (size_t) t * TOK, x.v.begin() + (size_t) (t + 1) * TOK), dfn, dsc, dba, 1);
    bool all = true;
    std::string bad;
    for (int T : {2, 3, 4, 5, 6, 7, 8, 9, 17}) {
        std::vector<float> xs;
        for (int t = 0; t < T; ++t) xs.insert(xs.end(), x.v.begin() + (size_t) (t % NG) * TOK, x.v.begin() + (size_t) (t % NG + 1) * TOK);
        const auto coef = mixes_of<G>(c, xs, dfn, dsc, dba, T);
        for (int t = 0; t < T; ++t)
            if (first_bit_diff(coef.data() + (size_t) t * NR, alone[t % NG].data(), NR) != NR) {
                all = false;
                bad += fmt(" T=%d token %d", T, t);
            }
    }
    c.rep.check(all, fmt("tinv   %s  hc_mixes: coefficients of a token are bit-identical for T = 2..9, 17 and alone", gname<G>()), bad);
}

template <class G>
void suite_order(Ctx& c) {
    // the same call under the three scheduling orders of the emulator must give the same bits (no atomics, fixed reduction order)
    constexpr int NR = G::kHcMixes, H = G::kHidden, HC = G::kHc;
    const std::string d = std::string("mhc_") + gname<G>() + "/b/";
    const auto x = c.gold->get<float>(d + "x"), fn = c.gold->get<float>(d + "fn"), scale = c.gold->get<float>(d + "scale"), base = c.gold->get<float>(d + "base"),
               f = c.gold->get<float>(d + "f"), cf = c.gold->get<float>(d + "coef32");
    const int T = (int) x.dim(0);
    Up<float> dx(c.dev, x), dfn(c.dev, fn), dsc(c.dev, scale), dba(c.dev, base), df(c.dev, f), dcf(c.dev, cf);
    const auto saved_order = ds41_emu::g_order;
    std::vector<float> ref_coef, ref_pre, ref_post;
    bool ok = true;
    std::string bad;
    const char* orders[] = {"forward", "reverse", "shuffle:7", "shuffle:8"};
    for (int oi = 0; oi < 4; ++oi) {
        ds41_emu::set_order_from_string(orders[oi]);
        DevBuf<float> coef(c.dev, (size_t) T * NR), y(c.dev, (size_t) T * H), out(c.dev, (size_t) T * HC * H);
        DevBuf<unsigned char> ws(c.dev, hc_scratch_bytes<G>(T));
        ds41_hc_mixes<G>(c.dev, dx.p(), HcWeights{dfn.p(), dsc.p(), dba.p()}, T, HcParams{}, coef.p, ws.p, hc_scratch_bytes<G>(T));
        ds41_hc_pre<G>(c.dev, dx.p(), dcf.p(), T, y.p);
        ds41_hc_post<G>(c.dev, df.p(), dx.p(), dcf.p(), T, out.p);
        const auto gc = coef.down(), gy = y.down(), go = out.down();
        if (oi == 0) {
            ref_coef = gc, ref_pre = gy, ref_post = go;
        } else {
            const bool same = first_bit_diff(gc.data(), ref_coef.data(), gc.size()) == gc.size() && first_bit_diff(gy.data(), ref_pre.data(), gy.size()) == gy.size() &&
                              first_bit_diff(go.data(), ref_post.data(), go.size()) == go.size();
            ok = ok && same;
            if (!same) bad += std::string(" ") + orders[oi];
        }
    }
    ds41_emu::g_order = saved_order;
    c.rep.check(ok, fmt("order  %s  hc_mixes / hc_pre / hc_post bit-identical in forward, reverse and two shuffled schedules", gname<G>()), bad);
}

// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
void suite_edge(Ctx& c) {
    constexpr int NR = G::kHcMixes, H = G::kHidden, HC = G::kHc, K = Derived<G>::kHcFlat;
    Rng rng(99);
    std::vector<float> fnv((size_t) NR * K), sc = {0.7f, 1.1f, 0.9f}, ba(NR);
    for (auto& v : fnv) v = (float) (rng.normal() / std::sqrt((double) K));
    for (auto& v : ba) v = (float) (0.5 * rng.normal());
    Up<float> dfn(c.dev, fnv), dsc(c.dev, sc), dba(c.dev, ba);
    HcWeights w{dfn.p(), dsc.p(), dba.p()};

    // zero stream: mixes = 0 * rsqrt(0 + 1e-20) = 0 -> pre = sigmoid(base) + eps, post = 2 sigmoid(base), exactly the formulas of mhc.py
    {
        std::vector<float> xz((size_t) HC * H, 0.0f);
        const auto coef = mixes_of<G>(c, xz, dfn, dsc, dba, 1);
        double worst = 0;
        bool finite = true;
        for (int i = 0; i < NR; ++i) finite = finite && std::isfinite(coef[i]);
        for (int k = 0; k < HC; ++k) {
            const double pre = 1.0 / (1.0 + std::exp(-(double) ba[k])) + 1e-6, post = 2.0 / (1.0 + std::exp(-(double) ba[HC + k]));
            worst = std::max({worst, std::fabs(coef[k] - pre), std::fabs(coef[HC + k] - post)});
        }
        c.rep.check(finite && worst < 2e-7, fmt("edge   %s  zero stream: finite, pre = sigmoid(base) + eps, post = 2 sigmoid(base)", gname<G>()), fmt("max err %.2e", worst));
    }
    // NaN / Inf in the stream propagate (a failure upstream is not laundered into a finite coefficient): the sum of squares is NaN, so every mixes lane is NaN
    {
        std::vector<float> xn((size_t) HC * H, 1.0f);
        xn[17] = std::nanf("");
        const auto coef = mixes_of<G>(c, xn, dfn, dsc, dba, 1);
        bool all_nan = true;
        for (int i = 0; i < NR; ++i) all_nan = all_nan && std::isnan(coef[i]);
        c.rep.check(all_nan, fmt("edge   %s  a NaN in the stream makes every coefficient NaN", gname<G>()));
        // ... and only that token's: the neighbour in the same batch is untouched
        std::vector<float> two((size_t) 2 * HC * H, 0.5f);
        two[5] = std::nanf("");
        const auto cf2 = mixes_of<G>(c, two, dfn, dsc, dba, 2);
        const auto clean = mixes_of<G>(c, std::vector<float>((size_t) HC * H, 0.5f), dfn, dsc, dba, 1);
        c.rep.check(std::isnan(cf2[0]) && first_bit_diff(cf2.data() + NR, clean.data(), NR) == NR, fmt("edge   %s  NaN stays inside its token (T = 2)", gname<G>()));
    }
    // expand: every copy equals the embedding row (bitwise)
    {
        const int T = 3;
        std::vector<float> emb((size_t) T * H);
        for (auto& v : emb) v = (float) rng.normal();
        DevBuf<float> e(c.dev, emb.size()), o(c.dev, (size_t) T * HC * H);
        e.up(emb);
        ds41_hc_expand<G>(c.dev, e.p, T, o.p);
        const auto g = o.down();
        bool ok = true;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < HC; ++k) ok = ok && first_bit_diff(g.data() + ((size_t) t * HC + k) * H, emb.data() + (size_t) t * H, H) == (size_t) H;
        c.rep.check(ok, fmt("edge   %s  expand: the embedding row in every copy", gname<G>()));
    }
    // the identity record
    {
        const int T = 5;
        DevBuf<float> r(c.dev, (size_t) T * NR);
        ds41_hc_set_identity_pre<G>(c.dev, r.p, T);
        const auto g = r.down();
        bool ok = true;
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < NR; ++i) ok = ok && g[(size_t) t * NR + i] == (i == 0 ? 1.0f : 0.0f);
        c.rep.check(ok, fmt("edge   %s  identity pre = [1, 0, 0, 0] (post / comb zero)", gname<G>()));
        // block 0's attention collapse with the identity record is the first copy, bitwise
        std::vector<float> xs((size_t) T * HC * H);
        for (auto& v : xs) v = (float) rng.normal();
        DevBuf<float> x(c.dev, xs.size()), y(c.dev, (size_t) T * H);
        x.up(xs);
        ds41_hc_pre<G>(c.dev, x.p, r.p, T, y.p);
        const auto gy = y.down();
        bool same = true;
        for (int t = 0; t < T; ++t) same = same && first_bit_diff(gy.data() + (size_t) t * H, xs.data() + (size_t) t * HC * H, H) == (size_t) H;
        c.rep.check(same, fmt("edge   %s  hc_pre with the identity record returns copy 0 unchanged", gname<G>()));
    }
    // argument checks
    {
        DevBuf<float> x(c.dev, (size_t) 2 * HC * H + 8), coef(c.dev, 2 * NR + 8), y(c.dev, (size_t) 2 * H + 8);
        DevBuf<unsigned char> ws(c.dev, hc_scratch_bytes<G>(2));
        auto throws = [&](auto fn_) {
            try {
                fn_();
            } catch (const std::invalid_argument&) {
                return true;
            } catch (...) {
            }
            return false;
        };
        c.rep.check(throws([&] { ds41_hc_mixes<G>(c.dev, x.p + 1, w, 2, HcParams{}, coef.p, ws.p, hc_scratch_bytes<G>(2)); }), fmt("edge   %s  hc_mixes refuses a misaligned stream", gname<G>()));
        c.rep.check(throws([&] { ds41_hc_mixes<G>(c.dev, x.p, w, 2, HcParams{}, coef.p, ws.p, hc_scratch_bytes<G>(2) - 1); }), fmt("edge   %s  hc_mixes refuses too little scratch", gname<G>()));
        c.rep.check(throws([&] { ds41_hc_pre<G>(c.dev, x.p, coef.p, 2, y.p + 2); }), fmt("edge   %s  hc_pre refuses a misaligned output", gname<G>()));
        c.rep.check(throws([&] { ds41_hc_post<G>(c.dev, y.p, x.p, coef.p, 2, x.p + 4); }), fmt("edge   %s  hc_post refuses a partially overlapping output", gname<G>()));
        ds41_hc_mixes<G>(c.dev, x.p, w, 0, HcParams{}, coef.p, ws.p, 0);        // T = 0: nothing happens, nothing throws
        c.rep.check(true, fmt("edge   %s  T = 0 is a no-op", gname<G>()));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string golden;
    std::vector<std::string> suites;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--golden" && i + 1 < argc) golden = argv[++i];
        else if (a == "--suite" && i + 1 < argc) suites.push_back(argv[++i]);
        else if (a == "--order" && i + 1 < argc) {
            if (!ds41_emu::set_order_from_string(argv[++i])) {
                std::fprintf(stderr, "bad --order\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "usage: %s --golden DIR [--suite mixes|split|prepost|chain|tinv|order|edge]... [--order forward|reverse|shuffle[:SEED]]\n", argv[0]);
            return 2;
        }
    }
    if (golden.empty()) {
        std::fprintf(stderr, "--golden DIR is required (python3 src/ds41/engram/golden/gen_golden.py --out DIR)\n");
        return 2;
    }
    auto want = [&](const char* s) { return suites.empty() || std::find(suites.begin(), suites.end(), s) != suites.end(); };
    try {
        Golden g(golden);
        Ctx c;
        c.gold = &g;
        std::printf("ds41_mhc_emu_test: scheduling order %s, golden %s\n", ds41_emu::order_name(), golden.c_str());
        if (want("mixes")) { suite_mixes<MiniGeom>(c); suite_mixes<RealGeom>(c); }
        if (want("split")) { suite_split<MiniGeom>(c); suite_split<RealGeom>(c); }
        if (want("prepost")) { suite_prepost<MiniGeom>(c); suite_prepost<RealGeom>(c); }
        if (want("chain")) { suite_chain<MiniGeom>(c); suite_chain<RealGeom>(c); }
        if (want("tinv")) { suite_tinv<MiniGeom>(c); suite_tinv<RealGeom>(c); }
        if (want("order")) { suite_order<MiniGeom>(c); suite_order<RealGeom>(c); }
        if (want("edge")) { suite_edge<MiniGeom>(c); suite_edge<RealGeom>(c); }
        return c.rep.finish("ds41_mhc_emu_test");
    } catch (const std::exception& e) {
        std::printf("FAIL  exception: %s\n", e.what());
        return 1;
    }
}
