// src/ds41/cuda/engram_test.cpp - DS1-D: Engram.  The host side (the n-gram hasher, the row gather) runs natively; the device side (MXFP4 row dequantisation,
// the combine kernel; src/ds41/cuda/engram_impl.cuh, the SAME source nvcc compiles for sm_70) runs on the CPU through the thread-model emulation (ds41_emu.hpp), or on the V100.
// The oracle is the NumPy one: golden data from src/ds41/engram/golden/gen_golden.py (ref/ds41/engram.py NgramHasher / engram_layer, quant.dequant_mxfp4), at MiniGeom's
// and RealGeom's shapes, with the mini GGUF's constants and with the REAL model's constants (the saved shard header and the extracted token map).  One test source, two programs:
//
//   ds41_engram_emu_test --golden DIR [--suite NAME ...] [--order forward|reverse|shuffle[:SEED]]     CPU emulation of the kernels, Mini + Real shapes, no GPU needed
//   ds41_engram_gpu_test --golden DIR [--suite NAME ...]        (-DDS1D_ON_GPU) the sm_70 kernels on the V100, RealGeom only, plus a `perf` suite (information); not in ctest
//
// Suites (default: all):
//   hasher    NgramHasher EXACTLY equals the oracle's int64 rows: whole stream in one call, one token at a time (decode), chunked, after reset(); mini and REAL constants; the
//             constants' validation; the DS1-A EngramConfig adapter.
//   gather    engram_gather_rows copies the right 136-byte (mini: 34-byte) rows, refuses an index outside the table.
//   dequant   ds41_engram_dequant_rows is BIT-identical to quant.dequant_mxfp4 over every E8M0 scale that matters (0 and 1 denormal, 255 = 2^127) and every code.
//   combine   ds41_engram_combine against the oracle's engram_layer code (given a kv): float64 reference, tolerance from the oracle's float32 noise (below); T-invariance;
//             scheduling-order invariance; the clamp (|dot| < 1e-6 gives sqrt(1e-6) with the sign of dot) analytically; NaN propagation per (token, copy).
//   runner    the whole path gather -> upload -> dequantise -> wkv -> combine: on the mini GGUF's real tables and Q8_0 wkv against engram_layer in QuantConfig.int8() (the wkv hook is
//             a host stand-in: ds41 int8 quantiser + Q8_0 dot, standing in for DS1-B's GEMV), decode (T = 1, incremental hasher) == prefill (T = 12) bit for bit; RealGeom shapes
//             with a synthetic table and the oracle's kv injected.
// TOLERANCES: the combine's sums (h^2, key^2, the dot over 5120 or 256 values) have a fixed order of their own, not numpy's pairwise one.  Documented bound: |kernel - float64| <=
// max(4 * |oracle float32 - float64|, 2e-7 * max|ref|) over the whole output; the end-to-end mini layers (the wkv GEMV differs by accumulation order too): 5e-5 * max|ref|.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "mhc_test_util.hpp"
#include "strata/ds41/cuda/engram.hpp"
#ifdef DS1D_ON_GPU
#include "strata/ds41/cuda/ds41_cuda_runtime.hpp"
#else
#include "ds41_emu.hpp"
#endif

using namespace strata::ds41;
using namespace strata::ds41::cuda;
using namespace ds1d;

namespace {

#ifdef DS1D_ON_GPU
using TestDev = CudaDev;
#define DS1D_GEOMS(fn, c) fn<RealGeom>(c)
#else
using TestDev = HostDev;
#define DS1D_GEOMS(fn, c) do { fn<MiniGeom>(c); fn<RealGeom>(c); } while (0)
#endif

struct Ctx {
    TestDev dev;
    Report rep;
    Golden* gold = nullptr;
};

template <class T>
struct Up {
    DevBuf<T> b;
    Up(Dev& d, const Arr<T>& a) : b(d, a.size()) { b.up(a.v); }
    Up(Dev& d, const std::vector<T>& v) : b(d, v.size()) { b.up(v); }
    T* p() const { return b.p; }
};

const char* geom_name(const MiniGeom*) { return "mini"; }
const char* geom_name(const RealGeom*) { return "real"; }
template <class G> const char* gname() { return geom_name((const G*) nullptr); }

bool throws_invalid(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
    }
    return false;
}
bool throws_range(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::out_of_range&) {
        return true;
    } catch (...) {
    }
    return false;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// hasher
// ---------------------------------------------------------------------------------------------------------------------------------
EngramConstants load_constants(const Golden& g, const std::string& name) {
    const std::string d = "hasher_" + name + "/";
    const auto layers = g.get<int32_t>(d + "layer_ids"), tm = g.get<int32_t>(d + "token_map");
    const auto par = g.get<int64_t>(d + "params");
    const auto primes = g.get<int64_t>(d + "primes"), offsets = g.get<int64_t>(d + "offsets"), mult = g.get<int64_t>(d + "multipliers"), ne = g.get<int64_t>(d + "num_embeddings");
    EngramConstants c;
    c.layer_ids = layers.v;
    c.n_heads = (int32_t) par[0];
    c.max_ngram_size = (int32_t) par[1];
    c.pad_token_id = (int32_t) par[2];
    c.compressed_vocab_size = (int32_t) par[3];
    c.primes = primes.v;
    c.offsets = offsets.v;
    c.multipliers = mult.v;
    c.token_map = tm.v;
    c.num_embeddings = ne.v;
    return c;
}

// the shape of DS1-A's strata::ds41::model::EngramConfig (include/strata/ds41/model/config.hpp), to check the adapter without including the loader
struct LoaderEngramConfig {
    std::vector<int> layers;
    int heads = 0, head_dim = 0, ngram = 0, pad_id = 0, cvocab = 0;
    std::vector<int64_t> num_embeddings, primes, offsets, multipliers;
    std::vector<int32_t> token_map;
};

void suite_hasher(Ctx& c, const std::string& name) {
    const EngramConstants k = load_constants(*c.gold, name);
    NgramHasher h(k);
    const int L = h.n_layers(), cols = h.rows_per_layer();
    const std::string d = "hasher_" + name + "/";
    c.rep.check(L == (int) k.layer_ids.size() && cols == (k.max_ngram_size - 1) * k.n_heads, fmt("hasher %s  %d Engram layers x %d rows per token", name.c_str(), L, cols));
    bool rows_ok = true;
    for (int li = 0; li < L; ++li) rows_ok = rows_ok && h.table_rows(li) == k.num_embeddings[(size_t) li];
    c.rep.check(rows_ok, fmt("hasher %s  table rows = sum of the layer's primes = engram.num_embeddings (%lld, %lld)", name.c_str(), (long long) h.table_rows(0),
                             (long long) h.table_rows(L - 1)));
    for (const char* s : {"s0", "s1", "s2", "s3"}) {
        const auto toks = c.gold->get<int32_t>(d + "tokens_" + s);
        const auto want = c.gold->get<int64_t>(d + "rows_" + s);
        const int n = (int) toks.size();
        const size_t per = (size_t) L * cols;
        auto same = [&](const std::vector<int64_t>& got) { return got.size() == want.size() && std::memcmp(got.data(), want.v.data(), got.size() * 8) == 0; };
        h.reset();
        std::vector<int64_t> all((size_t) n * per);
        h.push_many(toks.data(), n, all.data());
        c.rep.check(same(all) && h.position() == n, fmt("hasher %s/%s  %d tokens in one call == oracle (EXACT)", name.c_str(), s, n));
        h.reset();
        std::vector<int64_t> one((size_t) n * per);
        for (int i = 0; i < n; ++i) h.push(toks[i], one.data() + (size_t) i * per);
        c.rep.check(same(one), fmt("hasher %s/%s  one token at a time (decode) == oracle", name.c_str(), s));
        h.reset();
        std::vector<int64_t> chunked((size_t) n * per);
        int i = 0;
        const int steps[] = {1, 2, 3, 5, 7, 11, 13, 9};
        for (int si = 0; i < n; ++si) {
            const int m = std::min(n - i, steps[si % 8]);
            h.push_many(toks.data() + i, m, chunked.data() + (size_t) i * per);
            i += m;
        }
        c.rep.check(same(chunked), fmt("hasher %s/%s  chunked 1,2,3,5,7,11,13,9,... == oracle", name.c_str(), s));
        h.reset();
        h.push(toks[0], one.data());                        // a second sequence after reset(): the look-back does not leak
        h.reset();
        std::vector<int64_t> again((size_t) n * per);
        h.push_many(toks.data(), n, again.data());
        c.rep.check(same(again), fmt("hasher %s/%s  reset() starts a fresh sequence", name.c_str(), s));
    }
    // the DS1-A adapter
    {
        LoaderEngramConfig lc;
        lc.layers.assign(k.layer_ids.begin(), k.layer_ids.end());
        lc.heads = k.n_heads;
        lc.head_dim = 64;
        lc.ngram = k.max_ngram_size;
        lc.pad_id = k.pad_token_id;
        lc.cvocab = k.compressed_vocab_size;
        lc.num_embeddings = k.num_embeddings;
        lc.primes = k.primes;
        lc.offsets = k.offsets;
        lc.multipliers = k.multipliers;
        lc.token_map = k.token_map;
        NgramHasher h2(engram_constants_from(lc));
        const auto toks = c.gold->get<int32_t>(d + "tokens_s0");
        const size_t per = (size_t) L * cols;
        std::vector<int64_t> a((size_t) toks.size() * per), b((size_t) toks.size() * per);
        h.reset();
        h.push_many(toks.data(), (int) toks.size(), a.data());
        h2.push_many(toks.data(), (int) toks.size(), b.data());
        c.rep.check(a == b, fmt("hasher %s  engram_constants_from(DS1-A's EngramConfig) hashes the same", name.c_str()));
    }
    // validation: each violated invariant is refused
    {
        auto bad = [&](const char* what, const std::function<void(EngramConstants&)>& mut) {
            EngramConstants m = k;
            mut(m);
            c.rep.check(throws_invalid([&] { NgramHasher x(m); }), fmt("hasher %s  refuses constants with %s", name.c_str(), what));
        };
        bad("an offset that is not the running sum of the primes", [](EngramConstants& m) { m.offsets[1] += 1; });
        bad("a missing prime", [](EngramConstants& m) { m.primes.pop_back(); });
        bad("a prime < 2", [](EngramConstants& m) { m.primes[0] = 1; });
        bad("a missing multiplier", [](EngramConstants& m) { m.multipliers.pop_back(); });
        bad("a multiplier that overflows int64 with the compressed vocabulary", [](EngramConstants& m) { m.multipliers[0] = INT64_MAX / 2; });
        bad("a non-positive multiplier", [](EngramConstants& m) { m.multipliers[0] = 0; });
        bad("a token_map entry outside the compressed vocabulary", [](EngramConstants& m) { m.token_map[3] = m.compressed_vocab_size; });
        bad("a pad token outside the vocabulary", [](EngramConstants& m) { m.pad_token_id = (int32_t) m.token_map.size(); });
        bad("num_embeddings that disagree with the primes", [](EngramConstants& m) { m.num_embeddings[0] += 1; });
        bad("max_ngram_size 1", [](EngramConstants& m) { m.max_ngram_size = 1; });
        std::vector<int64_t> out((size_t) h.n_layers() * h.rows_per_layer());
        c.rep.check(throws_range([&] { h.push((int32_t) k.token_map.size(), out.data()); }), fmt("hasher %s  push() refuses a token id outside the vocabulary", name.c_str()));
        c.rep.check(throws_range([&] { h.push(-1, out.data()); }), fmt("hasher %s  push() refuses a negative token id", name.c_str()));
        c.rep.check(h.layer_index(k.layer_ids[0]) == 0 && h.layer_index(9999) == -1, fmt("hasher %s  layer_index", name.c_str()));
    }
}


// ---------------------------------------------------------------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
void suite_gather(Ctx& c) {
    constexpr size_t RB = Derived<G>::kEngramRowBytes;
    const int64_t R = 257;
    Rng rng(5);
    std::vector<uint8_t> tbl((size_t) R * RB);
    for (auto& b : tbl) b = (uint8_t) rng.next();
    const EngramTableView tv{tbl.data(), R};
    std::vector<int64_t> idx = {0, R - 1, 5, 5, 100};
    for (int i = 0; i < G::kEngramRows; ++i) idx.push_back((int64_t) (rng.next() % (uint64_t) R));
    std::vector<uint8_t> out(idx.size() * RB, 0xEE);
    engram_gather_rows(tv, idx.data(), (int64_t) idx.size(), RB, out.data());
    bool ok = true;
    for (size_t i = 0; i < idx.size(); ++i) ok = ok && std::memcmp(out.data() + i * RB, tbl.data() + (size_t) idx[i] * RB, RB) == 0;
    c.rep.check(ok, fmt("gather %s  %zu rows of %zu bytes copied from the table", gname<G>(), idx.size(), RB));
    int64_t bad = R;
    c.rep.check(throws_range([&] { engram_gather_rows(tv, &bad, 1, RB, out.data()); }), fmt("gather %s  refuses row == rows", gname<G>()));
    bad = -1;
    c.rep.check(throws_range([&] { engram_gather_rows(tv, &bad, 1, RB, out.data()); }), fmt("gather %s  refuses a negative row", gname<G>()));
    bad = 3;
    c.rep.check(throws_invalid([&] { engram_gather_rows(EngramTableView{}, &bad, 1, RB, out.data()); }), fmt("gather %s  refuses a null table", gname<G>()));
    engram_gather_rows(tv, &bad, 0, RB, out.data());
    c.rep.check(true, fmt("gather %s  n = 0 is a no-op", gname<G>()));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// dequant
// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
void suite_dequant(Ctx& c) {
    constexpr int HD = G::kEngramHeadDim;
    constexpr size_t RB = Derived<G>::kEngramRowBytes;
    const std::string d = std::string("dequant_") + gname<G>() + "/";
    const auto rows = c.gold->get<uint8_t>(d + "rows");
    const auto want = c.gold->get<float>(d + "out");
    const int n = (int) rows.dim(0);
    c.rep.check(rows.dim(1) == RB && want.dim(1) == (size_t) HD, fmt("dequant %s  golden row = %zu bytes -> %d values", gname<G>(), RB, HD));
    Up<uint8_t> dr(c.dev, rows);
    DevBuf<float> out(c.dev, (size_t) n * HD);
    ds41_engram_dequant_rows<G>(c.dev, dr.p(), n, out.p);
    auto got = out.down();
    size_t bd = first_bit_diff(got.data(), want.data(), got.size());
    c.rep.check(bd == got.size(), fmt("dequant %s  %d rows (all E8M0 scales 0, 1, 255..., every code): bit-identical to quant.dequant_mxfp4", gname<G>(), n),
                bd == got.size() ? "" : fmt("first difference at %zu: got %.9g (0x%08x) want %.9g (0x%08x)", bd, got[bd], bits_of(got[bd]), want[bd], bits_of(want[bd])));
    // a row count that does not fill the last thread block (the kernel's bounds), and a source that is not aligned (it is read byte by byte)
    const int m = 5;
    DevBuf<unsigned char> shifted(c.dev, (size_t) m * RB + 8);
    c.dev.h2d(shifted.p + 3, rows.data(), (size_t) m * RB);
    DevBuf<float> out5(c.dev, (size_t) m * HD);
    ds41_engram_dequant_rows<G>(c.dev, shifted.p + 3, m, out5.p);
    got = out5.down();
    bd = first_bit_diff(got.data(), want.data(), got.size());
    c.rep.check(bd == got.size(), fmt("dequant %s  5 rows from an unaligned source: bit-identical", gname<G>()));
    c.rep.check(throws_invalid([&] { ds41_engram_dequant_rows<G>(c.dev, dr.p(), n, out.p + 1); }), fmt("dequant %s  refuses a misaligned output", gname<G>()));
    ds41_engram_dequant_rows<G>(c.dev, dr.p(), 0, out.p);
    c.rep.check(true, fmt("dequant %s  0 rows is a no-op", gname<G>()));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// combine
// ---------------------------------------------------------------------------------------------------------------------------------
double tol_exp(const Err& oracle32, double floor_abs) { return std::max(4.0 * oracle32.max_abs, kLibmFactor * floor_abs); }       // the gate goes through expf

uint16_t bf16_bits(float v) {                           // round to nearest even (finite inputs)
    uint32_t u = bits_of(v);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
float bf16_to_f(uint16_t b) { return f_of_bits((uint32_t) b << 16); }

template <class G>
std::vector<float> run_combine(Ctx& c, const std::vector<float>& x, const std::vector<float>& kv, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k, int T) {
    DevBuf<float> dx(c.dev, x.size());
    dx.up(x);
    Up<float> dkv(c.dev, kv);
    Up<uint16_t> dq(c.dev, q), dk(c.dev, k);
    ds41_engram_combine<G>(c.dev, dx.p, dkv.p(), dq.p(), dk.p(), T, 1e-20f);
    return dx.down();
}

template <class G>
void suite_combine(Ctx& c) {
    constexpr int HC = G::kHc, H = G::kHidden, OUT = Derived<G>::kEngramOut;
    for (const char* cs : {"a", "b"}) {
        const std::string d = std::string("combine_") + gname<G>() + "/" + cs + "/";
        const auto x = c.gold->get<float>(d + "x"), kv = c.gold->get<float>(d + "kv");
        const auto qb = c.gold->get<uint16_t>(d + "q_bf16"), kb = c.gold->get<uint16_t>(d + "k_bf16");
        const auto o64 = c.gold->get<double>(d + "out64");
        const auto o32 = c.gold->get<float>(d + "out32o");
        const int T = (int) x.dim(0);
        const auto got = run_combine<G>(c, x.v, kv.v, qb.v, kb.v, T);
        const Err e = compare(got.data(), o64.data(), got.size()), e32 = compare(o32.data(), o64.data(), got.size());
        const double tol = tol_exp(e32, 2e-7 * e.max_ref);
        c.rep.check(e.nan_mismatch == 0 && e.max_abs <= tol, fmt("combine %s/%s  T=%d vs the oracle's engram_layer (float64)", gname<G>(), cs, T),
                    fmt("%s  (tol %.2e; oracle fp32 %.2e)", describe(e).c_str(), tol, e32.max_abs));
        // T-invariance: every token alone gives the bits it has in the batch
        bool same = true;
        for (int t = 0; t < T && same; ++t) {
            const auto one = run_combine<G>(c, std::vector<float>(x.v.begin() + (size_t) t * HC * H, x.v.begin() + (size_t) (t + 1) * HC * H),
                                           std::vector<float>(kv.v.begin() + (size_t) t * OUT, kv.v.begin() + (size_t) (t + 1) * OUT), qb.v, kb.v, 1);
            same = first_bit_diff(one.data(), got.data() + (size_t) t * HC * H, one.size()) == one.size();
        }
        c.rep.check(same, fmt("combine %s/%s  T-invariant: each token alone == in the batch (bit-identical)", gname<G>(), cs));
#ifndef DS1D_ON_GPU
        // the scheduling order changes no bit
        const auto saved = ds41_emu::g_order;
        bool ord = true;
        for (const char* o : {"reverse", "shuffle:3", "shuffle:4"}) {
            ds41_emu::set_order_from_string(o);
            const auto again = run_combine<G>(c, x.v, kv.v, qb.v, kb.v, T);
            ord = ord && first_bit_diff(again.data(), got.data(), got.size()) == got.size();
        }
        ds41_emu::g_order = saved;
        c.rep.check(ord, fmt("combine %s/%s  bit-identical in reverse and shuffled block / thread order", gname<G>(), cs));
#else
        const auto again = run_combine<G>(c, x.v, kv.v, qb.v, kb.v, T);              // on the card: the same call twice gives the same bits (no atomics, fixed order)
        c.rep.check(first_bit_diff(again.data(), got.data(), got.size()) == got.size(), fmt("combine %s/%s  repeatable (bit-identical on a second call)", gname<G>(), cs));
#endif
    }
}

template <class G>
void suite_combine_edge(Ctx& c) {
    constexpr int HC = G::kHc, H = G::kHidden, OUT = Derived<G>::kEngramOut;
    Rng rng(321);
    // h = 1 everywhere (rms 1), key = 1 (rms 1): rstd = 1, dot = sum(weight) / sqrt(H).  Copy 0: h alternates +-1 so that the sum is exactly 0; copies 1 / 2: weight -+1e-8
    // (|dot| < 1e-6: the clamp, with the sign of dot); copy 3: a large positive dot.
    std::vector<float> x((size_t) HC * H, 1.0f), kv((size_t) OUT, 1.0f);
    for (int d = 0; d < H; ++d) x[d] = (d % 2 == 0) ? 1.0f : -1.0f;
    std::vector<uint16_t> q((size_t) HC * H, bf16_bits(1.0f)), k((size_t) HC * H, bf16_bits(1.0f));
    const float w1 = -1e-8f, w2 = 1e-8f, w3 = 4.0f;
    for (int d = 0; d < H; ++d) {
        k[(size_t) H + d] = bf16_bits(w1);
        k[(size_t) 2 * H + d] = bf16_bits(w2);
        k[(size_t) 3 * H + d] = bf16_bits(w3);
    }
    for (int d = 0; d < H; ++d) kv[(size_t) HC * H + d] = (float) rng.normal();           // the value
    const auto got = run_combine<G>(c, x, kv, q, k, 1);
    const double wk[HC] = {1.0, (double) bf16_to_f(bf16_bits(w1)), (double) bf16_to_f(bf16_bits(w2)), (double) bf16_to_f(bf16_bits(w3))};
    double worst = 0;
    double gates[HC];
    for (int cp = 0; cp < HC; ++cp) {
        double dot = 0;
        for (int d = 0; d < H; ++d) dot += (double) x[(size_t) cp * H + d] * wk[cp] * 1.0;
        dot = dot / std::sqrt((double) H);                                              // rstd = 1
        const double mag = std::sqrt(std::max(std::fabs(dot), 1e-6));
        const double s = (std::signbit(dot) ? -mag : mag);
        gates[cp] = 1.0 / (1.0 + std::exp(-s));
        for (int d = 0; d < H; ++d) worst = std::max(worst, std::fabs((double) got[(size_t) cp * H + d] - ((double) x[(size_t) cp * H + d] + gates[cp] * (double) kv[(size_t) HC * H + d])));
    }
    c.rep.check(worst < kLibmFactor * 2e-6, fmt("combine %s  dot = 0 -> gate sigmoid(+1e-3); |dot| < 1e-6 keeps its sign; large dot", gname<G>()),
                fmt("gates %.9f %.9f %.9f %.9f, max err %.2e", gates[0], gates[1], gates[2], gates[3], worst));
    c.rep.check(gates[0] > 0.5 && gates[1] < 0.5 && gates[2] > 0.5 && gates[3] > 0.9, fmt("combine %s  (the cases really are the clamp: gate of copy 1 < 0.5 < gate of copy 2)", gname<G>()));

    // NaN propagation: a NaN in the key of copy 1 poisons that (token, copy) only; a NaN in the value poisons every copy of that token only
    {
        std::vector<float> x2, kv2;
        for (int t = 0; t < 3; ++t) {
            x2.insert(x2.end(), x.begin(), x.end());
            kv2.insert(kv2.end(), kv.begin(), kv.end());
        }
        kv2[(size_t) 0 * OUT + (size_t) 1 * H + 7] = std::nanf("");                      // token 0, key copy 1
        kv2[(size_t) 1 * OUT + (size_t) HC * H + 9] = std::nanf("");                     // token 1, value
        const auto g2 = run_combine<G>(c, x2, kv2, q, k, 3);
        bool ok = true;
        for (int t = 0; t < 3; ++t)
            for (int cp = 0; cp < HC; ++cp) {
                bool any_nan = false, all_nan = true;
                for (int d = 0; d < H; ++d) {
                    const bool n = std::isnan(g2[((size_t) t * HC + cp) * H + d]);
                    any_nan = any_nan || n;
                    all_nan = all_nan && n;
                }
                const bool expect_nan = (t == 0 && cp == 1) || t == 1;
                if (t == 1) ok = ok && any_nan && !all_nan;          // value NaN at one element: gate finite, only that element of the value is NaN
                else if (expect_nan) ok = ok && all_nan;
                else ok = ok && !any_nan && first_bit_diff(g2.data() + ((size_t) t * HC + cp) * H, got.data() + (size_t) cp * H, H) == (size_t) H;
            }
        c.rep.check(ok, fmt("combine %s  NaN in a key poisons its (token, copy); in a value only that element; neighbours untouched", gname<G>()));
        // a NaN in the stream itself: that copy's rstd, dot, gate and output are NaN, other copies are not
        std::vector<float> x3 = x;
        x3[(size_t) 2 * H + 4] = std::nanf("");
        const auto g3 = run_combine<G>(c, x3, kv, q, k, 1);
        bool ok3 = true;
        for (int cp = 0; cp < HC; ++cp)
            for (int d = 0; d < H; ++d) ok3 = ok3 && (cp == 2 ? std::isnan(g3[(size_t) cp * H + d]) : !std::isnan(g3[(size_t) cp * H + d]));
        c.rep.check(ok3, fmt("combine %s  NaN in the stream poisons its copy only", gname<G>()));
    }
    // argument checks
    {
        DevBuf<float> dx(c.dev, (size_t) HC * H + 8), dkv(c.dev, (size_t) OUT + 8);
        DevBuf<uint16_t> dq(c.dev, (size_t) HC * H + 8), dk(c.dev, (size_t) HC * H + 8);
        c.rep.check(throws_invalid([&] { ds41_engram_combine<G>(c.dev, dx.p + 1, dkv.p, dq.p, dk.p, 1, 1e-20f); }), fmt("combine %s  refuses a misaligned stream", gname<G>()));
        c.rep.check(throws_invalid([&] { ds41_engram_combine<G>(c.dev, dx.p, dkv.p + 1, dq.p, dk.p, 1, 1e-20f); }), fmt("combine %s  refuses a misaligned kv", gname<G>()));
        c.rep.check(throws_invalid([&] { ds41_engram_combine<G>(c.dev, dx.p, dkv.p, dq.p + 1, dk.p, 1, 1e-20f); }), fmt("combine %s  refuses a misaligned q", gname<G>()));
        ds41_engram_combine<G>(c.dev, dx.p, dkv.p, dq.p, dk.p, 0, 1e-20f);
        c.rep.check(true, fmt("combine %s  T = 0 is a no-op", gname<G>()));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------
// runner
// ---------------------------------------------------------------------------------------------------------------------------------
// Host stand-in for DS1-B's Q8_0 GEMV with int8 activations (docs/deepseek/DS1.md section 2): the activation quantiser of CONTRACTS.md (ggml's x86 q8_0 arithmetic with an FP32
// scale), acc += (d_w * d_x) * (integer sum of the int8 products).  Copies through the Dev (d2h / h2d), so it runs the same on the emulation and on a card.
float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) {
        if (m == 0) return f_of_bits(s);
        float f = (float) m * (1.0f / 16777216.0f);                    // 2^-24
        return s ? -f : f;
    }
    if (e == 31) return f_of_bits(s | 0x7F800000u | (m << 13));
    return f_of_bits(s | ((e + 112u) << 23) | (m << 13));
}

void quantize_block_contract(const float* x, int8_t* q, float& d) {
    uint32_t m = 0;
    for (int j = 0; j < 32; ++j) m = std::max(m, bits_of(x[j]) & 0x7FFFFFFFu);
    for (int j = 0; j < 32; ++j) q[j] = 0;
    if (m >= 0x7F800000u) {
        d = f_of_bits(0x7FC00000u);
        return;
    }
    if (m < 0x0D800000u) {
        d = 0.0f;
        return;
    }
    const float amax = f_of_bits(m);
    d = amax / 127.0f;
    const float id = 127.0f / amax;
    for (int j = 0; j < 32; ++j) {
        volatile float t = x[j] * id;
        const long v = std::lrintf(t);
        q[j] = (int8_t) std::min(127L, std::max(-127L, v));
    }
}

// The Q8_0 GEMV a session binds (DS1-B's / DS1-C's gemv_q8 later) as host code over the Dev: copied down, the integer sum per 32-block exact, acc += (d_w * d_x) * isum.  The activation
// quantiser in front of it is DS1-G's ds41_quantize_acts (engram_wkv_q8 calls it); the test cross-checks its bytes against the CONTRACTS.md rule above.
struct HostGemvQ8 {
    Dev& dev;
    int calls = 0;
    explicit HostGemvQ8(Dev& d) : dev(d) {}
    void operator()(const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream) {
        ++calls;
        const int nb = k / 32;
        std::vector<uint8_t> q8((size_t) n * nb * 34);
        std::vector<int8_t> hq((size_t) T * k);
        std::vector<float> hs((size_t) T * nb), hy((size_t) T * n);
        dev.d2h(q8.data(), w, q8.size());
        dev.d2h(hq.data(), xq, hq.size());
        dev.d2h(hs.data(), xs, hs.size() * sizeof(float));
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < n; ++o) {
                float acc = 0.0f;
                for (int b = 0; b < nb; ++b) {
                    const uint8_t* blk = q8.data() + ((size_t) o * nb + b) * 34;
                    uint16_t hd;
                    std::memcpy(&hd, blk, 2);
                    int isum = 0;
                    for (int j = 0; j < 32; ++j) isum += (int) (int8_t) blk[2 + j] * (int) hq[(size_t) t * k + 32 * b + j];
                    volatile float term = (half_to_float(hd) * hs[(size_t) t * nb + b]) * (float) isum;
                    volatile float nacc = acc + term;
                    acc = nacc;
                }
                hy[(size_t) t * n + o] = acc;
            }
        dev.h2d(y, hy.data(), hy.size() * sizeof(float));
    }
};

#ifndef DS1D_ON_GPU
void suite_runner_mini(Ctx& c) {
    using G = MiniGeom;
    constexpr int HC = G::kHc, H = G::kHidden, IN = Derived<G>::kEngramIn, OUT = Derived<G>::kEngramOut;
    const auto toks = c.gold->get<int32_t>("layer_mini/tokens");
    const auto nl = c.gold->get<int64_t>("layer_mini/num_layers")[0];
    const int S = (int) toks.size();
    const EngramConstants k = load_constants(*c.gold, "mini");
    for (int j = 0; j < (int) nl; ++j) {
        const std::string d = fmt("layer_mini/%d/", j);
        const auto table = c.gold->get<uint8_t>(d + "table"), wkv = c.gold->get<uint8_t>(d + "wkv_q8");
        const auto x = c.gold->get<float>(d + "x"), rows = c.gold->get<float>(d + "rows_f32"), kv_int8 = c.gold->get<float>(d + "kv_int8"),
                   out_int8 = c.gold->get<float>(d + "out_int8"), out_exact = c.gold->get<float>(d + "out_exact");
        const auto qb = c.gold->get<uint16_t>(d + "q_bf16"), kb = c.gold->get<uint16_t>(d + "k_bf16");
        const auto layer_id = (int) c.gold->get<int64_t>(d + "layer_id")[0];
        const std::string tag = fmt("runner mini  layer %d", layer_id);
        c.rep.check(table.dim(1) == Derived<G>::kEngramRowBytes && wkv.dim(0) == (size_t) OUT && x.dim(0) == (size_t) S, tag + "  golden shapes");
        NgramHasher hasher(k);
        const int L = hasher.n_layers(), cols = hasher.rows_per_layer();
        std::vector<int64_t> hr((size_t) S * L * cols);
        hasher.push_many(toks.data(), S, hr.data());
        c.rep.check(hr == [&] {
            const auto g = c.gold->get<int64_t>("layer_mini/rows_idx");
            return g.v;
        }(), tag + "  hasher rows == the oracle's for this stream (EXACT)");

        EngramRunner<G> runner(c.dev, S);
        Up<uint16_t> dq(c.dev, qb), dk(c.dev, kb);
        EngramLayerWeights w{EngramTableView{table.data(), (int64_t) table.dim(0)}, dq.p(), dk.p()};
        Up<uint8_t> dwkv(c.dev, wkv);
        HostGemvQ8 gemv(c.dev);
        const EngramWkvFn host_wkv = engram_wkv_q8<G>(c.dev, dwkv.p(), runner.xq(), runner.xs(),
                                                      [&gemv](const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream st) { gemv(w, n, k, xq, xs, T, y, st); });

        // prefill: T = S in one call, with the stand-in wkv
        DevBuf<float> dx(c.dev, x.size());
        dx.up(x.v);
        runner.run(w, hr.data() + (size_t) j * cols, S, (int64_t) L * cols, dx.p, host_wkv, 1e-20f);
        const auto got = dx.down();
        std::vector<float> got_rows((size_t) S * IN);
        c.dev.d2h(got_rows.data(), runner.rows_f32(), got_rows.size() * sizeof(float));
        const size_t bd = first_bit_diff(got_rows.data(), rows.data(), got_rows.size());
        c.rep.check(bd == got_rows.size(), tag + fmt("  gather -> upload -> dequantise: the %d x %d rows are bit-identical to the oracle's dequantised table rows", S, G::kEngramRows));
        std::vector<float> got_kv((size_t) S * OUT);
        c.dev.d2h(got_kv.data(), runner.kv(), got_kv.size() * sizeof(float));
        {   // DS1-G's ds41_quantize_acts (natural order) wrote runner.xq() / xs(): the same bytes as the CONTRACTS.md rule applied to the oracle's dequantised rows
            std::vector<int8_t> hq((size_t) S * IN), eq((size_t) S * IN);
            std::vector<float> hs((size_t) S * (IN / 32)), es((size_t) S * (IN / 32));
            c.dev.d2h(hq.data(), runner.xq(), hq.size());
            c.dev.d2h(hs.data(), runner.xs(), hs.size() * sizeof(float));
            for (size_t b = 0; b < (size_t) S * (IN / 32); ++b) quantize_block_contract(rows.data() + 32 * b, eq.data() + 32 * b, es[b]);
            c.rep.check(hq == eq && first_bit_diff(hs.data(), es.data(), hs.size()) == hs.size(), tag + "  ds41_quantize_acts (natural) on the Engram rows == the CONTRACTS.md quantiser (bit-identical)");
        }
        const Err ek = compare(got_kv.data(), kv_int8.data(), got_kv.size());
        c.rep.check(ek.nan_mismatch == 0 && ek.max_abs <= 2e-5 * ek.max_ref, tag + "  wkv stand-in (int8 activations x Q8_0) vs the oracle's kv in int8 mode", describe(ek));
        const Err eo = compare(got.data(), out_int8.data(), got.size());
        c.rep.check(eo.nan_mismatch == 0 && eo.max_abs <= 5e-5 * eo.max_ref, tag + "  layer output vs engram_layer(QuantConfig.int8())", describe(eo));
        const Err ex = compare(got.data(), out_exact.data(), got.size());
        c.rep.check(ex.nan_mismatch == 0, tag + "  (information) distance of the int8 path from the exact-activation oracle", describe(ex));

        // the combine alone, fed the oracle's own kv: only the combine's arithmetic is left
        DevBuf<float> dx2(c.dev, x.size());
        dx2.up(x.v);
        const auto inject = [&](const float*, int T, float* kv, Stream) { c.dev.h2d(kv, kv_int8.data(), (size_t) T * OUT * sizeof(float)); };
        runner.run(w, hr.data() + (size_t) j * cols, S, (int64_t) L * cols, dx2.p, inject, 1e-20f);
        const auto got2 = dx2.down();
        const Err e2 = compare(got2.data(), out_int8.data(), got2.size());
        c.rep.check(e2.nan_mismatch == 0 && e2.max_abs <= 2e-6 * e2.max_ref, tag + "  combine on the oracle's kv vs the oracle's output", describe(e2));

        // decode: one token at a time through an incremental hasher == the prefill, bit for bit (T-invariance of the whole path)
        NgramHasher h2(k);
        DevBuf<float> dx3(c.dev, x.size());
        dx3.up(x.v);
        for (int t = 0; t < S; ++t) {
            std::vector<int64_t> row((size_t) L * cols);
            h2.push(toks[(size_t) t], row.data());
            runner.run(w, row.data() + (size_t) j * cols, 1, (int64_t) L * cols, dx3.p + (size_t) t * HC * H, host_wkv, 1e-20f);
        }
        const auto got3 = dx3.down();
        c.rep.check(first_bit_diff(got3.data(), got.data(), got.size()) == got.size(), tag + fmt("  decode (T=1, incremental hasher) == prefill (T=%d): bit-identical", S));
        c.rep.check(gemv.calls == 1 + S, tag + fmt("  the wkv hook called the Q8_0 GEMV once per run (%d)", gemv.calls));
        // the runner refuses what it cannot hold
        c.rep.check(throws_invalid([&] { runner.upload_rows(w.table, hr.data(), S + 1, (int64_t) L * cols); }), tag + "  refuses T > t_max");
    }
}

#endif

void suite_runner_real(Ctx& c) {
    using G = RealGeom;
    constexpr int OUT = Derived<G>::kEngramOut, IN = Derived<G>::kEngramIn;
    const auto table = c.gold->get<uint8_t>("layer_real/table");
    const auto q = c.gold->get<uint16_t>("layer_real/q_bf16"), kb = c.gold->get<uint16_t>("layer_real/k_bf16");
    const auto idx = c.gold->get<int64_t>("layer_real/idx");
    const auto rows = c.gold->get<float>("layer_real/rows_f32"), x = c.gold->get<float>("layer_real/x"), kv = c.gold->get<float>("layer_real/kv");
    const auto o64 = c.gold->get<double>("layer_real/out64");
    const int T = (int) x.dim(0);
    EngramRunner<G> runner(c.dev, T);
    Up<uint16_t> dq(c.dev, q), dk(c.dev, kb);
    EngramLayerWeights w{EngramTableView{table.data(), (int64_t) table.dim(0)}, dq.p(), dk.p()};
    DevBuf<float> dx(c.dev, x.size());
    dx.up(x.v);
    const auto inject = [&](const float*, int Tn, float* kvd, Stream) { c.dev.h2d(kvd, kv.data(), (size_t) Tn * OUT * sizeof(float)); };
    runner.run(w, idx.data(), T, G::kEngramRows, dx.p, inject, 1e-20f);
    std::vector<float> got_rows((size_t) T * IN);
    c.dev.d2h(got_rows.data(), runner.rows_f32(), got_rows.size() * sizeof(float));
    c.rep.check(first_bit_diff(got_rows.data(), rows.data(), got_rows.size()) == got_rows.size(),
                fmt("runner real  %d x %d rows of %d bytes (a %lld-row synthetic table) gathered and dequantised: bit-identical", T, G::kEngramRows, (int) Derived<G>::kEngramRowBytes,
                    (long long) table.dim(0)));
    const auto got = dx.down();
    const Err e = compare(got.data(), o64.data(), got.size());
    c.rep.check(e.nan_mismatch == 0 && e.max_abs <= kLibmFactor * (4e-6 * e.max_ref + 1e-6), "runner real  combine on the real-shape stream vs the oracle (float64)", describe(e));
}

#ifdef DS1D_ON_GPU
template <class G>
void suite_perf(Ctx& c) {
    constexpr int H = G::kHidden, HC = G::kHc, OUT = Derived<G>::kEngramOut, ROWS = G::kEngramRows;
    constexpr size_t RB = Derived<G>::kEngramRowBytes;
    for (int i = 0; i < 2; ++i) {
        const EngramKernelInfo ki = ds41_engram_kernel_info<G>(i);
        c.rep.info(fmt("engram kernel %d (%s): %d registers, %d B static shared memory", i, i == 0 ? "MXFP4 dequantise" : "combine", ki.regs, ki.static_smem));
    }
    Rng rng(4);
    for (const int T : {1, 8}) {
        std::vector<uint8_t> raw((size_t) T * ROWS * RB);
        for (auto& b : raw) b = (uint8_t) rng.next();
        for (size_t i = 0; i < raw.size(); i += 17) raw[i] = (uint8_t) rng.range(110, 130);          // scale bytes (the buffer is a multiple of 17 only per row; the values do not matter here)
        std::vector<float> xv((size_t) T * HC * H), kvv((size_t) T * OUT);
        for (auto& v : xv) v = (float) rng.normal();
        for (auto& v : kvv) v = (float) rng.normal();
        std::vector<uint16_t> qk((size_t) HC * H, bf16_bits(1.0f));
        Up<uint8_t> draw(c.dev, raw);
        Up<float> dkv(c.dev, kvv);
        Up<uint16_t> dq(c.dev, qk), dk(c.dev, qk);
        DevBuf<float> rows(c.dev, (size_t) T * ROWS * G::kEngramHeadDim), dx(c.dev, xv.size());
        dx.up(xv);
        auto line = [&](const char* what, double bytes, const std::function<void()>& fn_) {
            const double us = c.dev.time_us(fn_, 200);
            c.rep.info(fmt("perf   T=%d  %-34s %8.2f us   %7.1f GB/s effective (%.2f MB moved)", T, what, us, bytes / us * 1e-3, bytes * 1e-6));
        };
        line("engram dequantise (MXFP4 -> f32)", (double) T * ROWS * (RB + (double) G::kEngramHeadDim * 4), [&] { ds41_engram_dequant_rows<G>(c.dev, draw.p(), T * ROWS, rows.p); });
        line("engram combine (in place)", (double) T * ((2 * HC + 1) * H + (double) HC * H) * 4 + (double) HC * H * 4, [&] { ds41_engram_combine<G>(c.dev, dx.p, dkv.p(), dq.p(), dk.p(), T, 1e-20f); });
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    std::string golden;
    std::vector<std::string> suites;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--golden" && i + 1 < argc) golden = argv[++i];
        else if (a == "--suite" && i + 1 < argc) suites.push_back(argv[++i]);
#ifndef DS1D_ON_GPU
        else if (a == "--order" && i + 1 < argc) {
            if (!ds41_emu::set_order_from_string(argv[++i])) {
                std::fprintf(stderr, "bad --order\n");
                return 2;
            }
        }
#endif
        else {
#ifdef DS1D_ON_GPU
            std::fprintf(stderr, "usage: %s --golden DIR [--suite hasher|gather|dequant|combine|runner|perf]...\n", argv[0]);
#else
            std::fprintf(stderr, "usage: %s --golden DIR [--suite hasher|gather|dequant|combine|runner]... [--order forward|reverse|shuffle[:SEED]]\n", argv[0]);
#endif
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
#ifdef DS1D_ON_GPU
        std::printf("ds41_engram_gpu_test: the sm_70 kernels, golden %s\n", golden.c_str());
#else
        std::printf("ds41_engram_emu_test: scheduling order %s, golden %s\n", ds41_emu::order_name(), golden.c_str());
#endif
        if (want("hasher")) { suite_hasher(c, "mini"); suite_hasher(c, "real"); }
        if (want("gather")) DS1D_GEOMS(suite_gather, c);
        if (want("dequant")) DS1D_GEOMS(suite_dequant, c);
        if (want("combine")) {
            DS1D_GEOMS(suite_combine, c);
            DS1D_GEOMS(suite_combine_edge, c);
        }
        if (want("runner")) {
#ifndef DS1D_ON_GPU
            suite_runner_mini(c);
#endif
            suite_runner_real(c);
        }
#ifdef DS1D_ON_GPU
        if (suites.empty() || want("perf")) suite_perf<RealGeom>(c);
        c.dev.sync();
        return c.rep.finish("ds41_engram_gpu_test");
#else
        return c.rep.finish("ds41_engram_emu_test");
#endif
    } catch (const std::exception& e) {
        std::printf("FAIL  exception: %s\n", e.what());
        return 1;
    }
}
