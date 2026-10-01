// src/ds41/attn/attn_emu_test.cpp - DS1-C: runs the attention kernels and the per-layer driver the V100 runs (src/ds41/cuda/attn_impl.cuh,
// src/ds41/attn/attn_host_impl.hpp, compiled for the host with -DDS41_EMU: every GPU thread is a fiber, see src/ds41/cuda/ds41_emu.hpp) on the CPU.
//
//   ds41_attn_emu_test [--kernels] [--oracle DIR] [--order forward|reverse|shuffle[:SEED]] [--seed N] [--scenario NAME]
//
//   --kernels   every kernel against an independent C++ reference, at MiniGeom AND RealGeom shapes, random data: the three fake-quantisers (bit-exact,
//               against a table-based implementation), RMSNorm, RoPE (bit-exact), swa_kv, the compressor (ratios 1, 2, 4), index_k / iq, the indexer
//               scores (+ candidate mask), block scores, the top-k select (exact, ties to the lower index, -inf / NaN never selected), the sparse
//               attention with the sink (a ring that is part filled and wrapped, padded top-k rows, SWA-only) and the guard zones of every buffer.
//   --oracle    DIR = the output of src/ds41/attn/golden/gen_golden.py: the NumPy oracle's `attention_layer` on a MiniGeom-shaped 8-layer model over every
//               layer role (SWA, FULL ratio 2, REUSE, FULL ratio 1 + candidate pool, REINDEX), 100 positions, KV quantisation on / off / mixed, decode
//               and prefill-then-decode.  The engine replays token by token and every stage is compared: q, SWA KV row, group latent, compressed-KV and
//               index-K cache rows, indexer scores, candidate pool, top-k indices, o, the layer output.  Also: T = 1..8 windows give bit-identical rows
//               to T = 1; a second pass after reset() and under two other scheduling orders is bit-identical; the dense stand-ins below are the only
//               non-package code (DS1-B / DS1-G replace them).
//   TOLERANCES (relative to the largest magnitude of the compared row): 3e-5 for every FP32 stage (the oracle's reductions are numpy's, the engine's are
//   fixed butterflies; observed ~1e-6).  The KV caches are FAKE-QUANTISED: an element whose pre-quantisation value differs by 1 ulp from the oracle's
//   can round to the neighbouring grid point (a "flip": probability ~1e-6 per element), so those rows are compared bit for bit and a flip is an EVENT, not
//   a failure: it is counted (budget below), the engine's row is then overwritten by the oracle's (so the run stays on the oracle's trajectory) and the
//   outputs of that step, which consumed the flipped value, are compared with the loose tolerance 5e-2 instead.  The same for a top-k / pool decision
//   that differs from the oracle's: allowed only when every differing position is a near-tie in the ORACLE's scores (<= 3e-5 of the score scale) or the
//   scores themselves had an event; the engine's selection must always be exactly the top-k of ITS OWN scores (checked on the host).
//   Everything is deterministic for a given oracle run; the emulator poisons shared memory, checks alignment and guards every buffer.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ds41_emu.hpp"
#include "strata/ds41/cuda/attn.hpp"

using namespace strata::ds41;
using namespace strata::ds41::cuda;

namespace {

int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (cond) {                                       \
            ++g_pass;                                     \
        } else {                                          \
            ++g_fail;                                     \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }          // [0, 1)
    int below(int n) { return (int) (next() % (uint64_t) n); }
    float normal() {
        const double u = uni() + 1e-300, v = uni();
        return (float) (std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
    }
};

template <class T>
struct Up {                                  // a device array filled from a vector
    DevBuf<T> b;
    Up(Dev& d, const std::vector<T>& v) : b(d, v.size() ? v.size() : 1) {
        if (!v.empty()) b.up(v);
    }
    T* p() { return b.p; }
};

uint32_t bits_of(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
float float_of(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// independent references
// ---------------------------------------------------------------------------------------------------------------------------------
const std::vector<float>& e4m3_grid() {                       // the finite non-negative e4m3fn values (codes 0 .. 0x7E), ascending
    static const std::vector<float> g = [] {
        std::vector<float> v;
        for (int c = 0; c < 127; ++c) {
            const int e = c >> 3, m = c & 7;
            v.push_back(e == 0 ? (float) m * std::ldexp(1.0f, -9) : (1.0f + (float) m / 8.0f) * std::ldexp(1.0f, e - 7));
        }
        return v;
    }();
    return g;
}
const std::vector<float>& e2m1_grid() {
    static const std::vector<float> g = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return g;
}
/// round-to-nearest, ties to the EVEN code, of a >= 0 onto an ascending grid (saturating at the top)
float rne_grid(const std::vector<float>& g, float a) {
    const size_t hi = (size_t) (std::upper_bound(g.begin(), g.end(), a) - g.begin());
    if (hi == g.size()) return g.back();
    const size_t lo = hi - 1;
    const double dl = (double) a - (double) g[lo], dh = (double) g[hi] - (double) a;
    if (dl < dh) return g[lo];
    if (dh < dl) return g[hi];
    return lo % 2 == 0 ? g[lo] : g[hi];
}
float pow2_ceil_ref(float t) {
    int e;
    const float m = std::frexp(t, &e);
    return std::ldexp(1.0f, m == 0.5f ? e - 1 : e);
}
/// kind 0: act_quant_fp8 (e4m3, pow2 scale, block 32); 1: fp4_quant_e8m0 (e2m1, pow2 scale, block 32); 2: fp4_quant_e4m3 (e2m1, e4m3 scale, block 16)
void fq_ref(int kind, const float* x, int n, float* y) {
    const int blk = kind == 2 ? 16 : 32;
    for (int b = 0; b < n; b += blk) {
        float amax = 0.0f;
        for (int j = 0; j < blk; ++j) amax = std::max(amax, std::fabs(x[b + j]));
        float s;
        if (kind == 0) {
            amax = std::max(amax, 1e-4f);
            s = pow2_ceil_ref(amax * (float) (1.0 / 448.0));
        } else if (kind == 1) {
            amax = std::max(amax, 6.0f * std::ldexp(1.0f, -126));
            s = pow2_ceil_ref(amax * (float) (1.0 / 6.0));
        } else {
            amax = std::max(amax, 6.0f * std::ldexp(1.0f, -9));
            s = std::min(rne_grid(e4m3_grid(), amax / 6.0f), 448.0f);
        }
        for (int j = 0; j < blk; ++j) {
            float v = x[b + j] / s;
            const float lim = kind == 0 ? 448.0f : 6.0f;
            v = std::min(std::max(v, -lim), lim);
            const float r = rne_grid(kind == 0 ? e4m3_grid() : e2m1_grid(), std::fabs(v));
            y[b + j] = std::copysign(r, v) * s;
        }
    }
}

/// RoPE of the last `rd` channels of a row of `d` (adjacent pairs), each product rounded to float as the oracle's numpy does; inverse conjugates.
void rope_tail_ref(float* row, int d, int rd, const float* cos, const float* sin, bool inverse) {
    for (int i = 0; i < rd / 2; ++i) {
        const float a = row[d - rd + 2 * i], b = row[d - rd + 2 * i + 1], c = cos[i];
        const float s = inverse ? -sin[i] : sin[i];
        const float ac = a * c, bs = b * s, as = a * s, bc = b * c;
        row[d - rd + 2 * i] = ac - bs;
        row[d - rd + 2 * i + 1] = as + bc;
    }
}

void topk_ref(const std::vector<float>& s, int k, std::vector<int>& sel, std::vector<uint8_t>& flags) {
    std::vector<int> el;
    for (int i = 0; i < (int) s.size(); ++i)
        if (s[i] == s[i] && bits_of(s[i]) != 0xFF800000u) el.push_back(i);
    std::stable_sort(el.begin(), el.end(), [&](int a, int b) { return s[a] > s[b]; });
    if ((int) el.size() > k) el.resize((size_t) k);
    flags.assign(s.size(), 0);
    for (int i : el) flags[(size_t) i] = 1;
    sel = el;
    std::sort(sel.begin(), sel.end());
}

double rel_err(const float* a, const float* ref, int n) {          // max |a - ref| / max |ref|   (inf if a holds a NaN where ref does not)
    double mx = 0.0, md = 0.0;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(ref[i])) continue;
        mx = std::max(mx, (double) std::fabs(ref[i]));
        if (std::isnan(a[i])) return INFINITY;
        if (std::isinf(ref[i]) || std::isinf(a[i])) {
            if (a[i] != ref[i]) return INFINITY;
            continue;
        }
        md = std::max(md, std::fabs((double) a[i] - (double) ref[i]));
    }
    return md / std::max(mx, 1e-30);
}

std::vector<float> down(Dev& dev, const float* p, size_t n) {
    std::vector<float> v(n);
    dev.d2h(v.data(), p, n * sizeof(float));
    return v;
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the kernel tests
// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
struct KernelTests {
    using K = AttnKernels<G>;
    Dev& dev;
    Rng rng;
    std::string tag;
    KernelTests(Dev& d, uint64_t seed) : dev(d), rng(seed), tag(std::string("[") + G::kName + "] ") {}

    std::vector<float> randv(int n, float scale = 1.0f) {
        std::vector<float> v((size_t) n);
        for (auto& x : v) x = rng.normal() * scale;
        return v;
    }
    void rope_row(std::vector<float>& cos, std::vector<float>& sin) {
        cos.resize((size_t) G::kRopeDim / 2);
        sin.resize((size_t) G::kRopeDim / 2);
        for (size_t i = 0; i < cos.size(); ++i) {
            const double a = rng.uni() * 6.0 - 3.0;
            cos[i] = (float) std::cos(a);
            sin[i] = (float) std::sin(a);
        }
    }

    void fakequant() {
        const int n = 32 * 48;
        std::vector<float> x((size_t) n);
        for (int b = 0; b < n / 32; ++b) {
            const double mag = std::pow(10.0, -6.0 + 10.0 * rng.uni());
            for (int j = 0; j < 32; ++j) x[(size_t) (b * 32 + j)] = (float) (rng.normal() * mag);
        }
        auto block = [&](int b) { return x.data() + (size_t) b * 32; };
        for (int j = 0; j < 32; ++j) block(0)[j] = 0.0f;                                         // all zero
        for (int j = 0; j < 32; ++j) block(1)[j] = (float) rng.normal() * 1e-30f;                // below every clamp
        for (int j = 0; j < 32; ++j) block(2)[j] = (float) rng.normal() * 1e-9f;                 // denormal-ish scales
        block(3)[5] = 1e30f;                                                                      // a huge element: the rest rounds to 0
        block(4)[0] = -3.0e38f;
        // exact midpoints between neighbouring grid values (ties to even): amax 448 / 6 pins the scale at 1
        const auto& g4 = e4m3_grid();
        block(5)[0] = 448.0f;
        for (int j = 1; j < 32; ++j) {
            const int i = rng.below((int) g4.size() - 2);
            block(5)[j] = (rng.below(2) ? -1.0f : 1.0f) * 0.5f * (g4[(size_t) i] + g4[(size_t) i + 1]);
        }
        const auto& g2 = e2m1_grid();
        for (int b = 6; b <= 9; ++b) {
            block(b)[0] = b % 2 ? -6.0f : 6.0f;
            for (int j = 1; j < 32; ++j) {
                const int i = rng.below(7);
                block(b)[j] = (rng.below(2) ? -1.0f : 1.0f) * 0.5f * (g2[(size_t) i] + g2[(size_t) i + 1]) * (b >= 8 ? 1.0f : 1.0f);
            }
        }
        Up<float> dx(dev, x);
        DevBuf<float> dy(dev, (size_t) n);
        for (int kind = 0; kind < 3; ++kind) {
            K::fakequant(kind, dx.p(), n, dy.p, nullptr);
            const std::vector<float> got = dy.down();
            std::vector<float> ref((size_t) n);
            fq_ref(kind, x.data(), n, ref.data());
            int bad = 0;
            for (int i = 0; i < n; ++i) {
                if (bits_of(got[(size_t) i]) != bits_of(ref[(size_t) i]) && !(got[(size_t) i] == 0.0f && ref[(size_t) i] == 0.0f)) {
                    if (++bad <= 3) std::printf("  kind %d element %d: x %.9g kernel %.9g reference %.9g\n", kind, i, x[(size_t) i], got[(size_t) i], ref[(size_t) i]);
                }
            }
            CHECK(bad == 0, "%sfakequant kind %d: %d of %d elements differ from the table-based reference", tag.c_str(), kind, bad, n);
        }
    }

    void rmsnorm() {
        for (int n : {64, 96, 256, 1280, 2048}) {
            const std::vector<float> x = randv(n, 3.0f), w = randv(n, 1.0f);
            Up<float> dx(dev, x), dw(dev, w);
            DevBuf<float> dy(dev, (size_t) n);
            K::rmsnorm_row(dx.p(), dw.p(), n, 1e-20f, dy.p, nullptr);
            const std::vector<float> got = dy.down();
            double ss = 0;
            for (float v : x) ss += (double) v * v;
            const double den = std::sqrt(ss / n + 1e-20);
            std::vector<float> ref((size_t) n);
            for (int i = 0; i < n; ++i) ref[(size_t) i] = (float) ((double) w[(size_t) i] * x[(size_t) i] / den);
            const double e = rel_err(got.data(), ref.data(), n);
            CHECK(e < 3e-6, "%srmsnorm n=%d: relative error %.3g", tag.c_str(), n, e);
        }
        // eps matters for an all-zero row: y = w * 0 / sqrt(eps) = 0, not NaN
        {
            const int n = 128;
            std::vector<float> x((size_t) n, 0.0f);
            const std::vector<float> w = randv(n);
            Up<float> dx(dev, x), dw(dev, w);
            DevBuf<float> dy(dev, (size_t) n);
            K::rmsnorm_row(dx.p(), dw.p(), n, 1e-20f, dy.p, nullptr);
            const auto got = dy.down();
            bool ok = true;
            for (float v : got) ok = ok && v == 0.0f;
            CHECK(ok, "%srmsnorm of a zero row is not zero", tag.c_str());
        }
    }

    void q_rope() {
        const int D = G::kHeadDim, R = G::kRopeDim, H = G::kHeads;
        std::vector<float> q = randv(H * D, 2.0f), cos, sin;
        rope_row(cos, sin);
        Up<float> dq(dev, q), dc(dev, cos), ds(dev, sin);
        K::q_rope(dq.p(), dc.p(), ds.p(), nullptr);
        const auto got = dq.b.down();
        std::vector<float> ref = q;
        for (int h = 0; h < H; ++h) rope_tail_ref(ref.data() + (size_t) h * D, D, R, cos.data(), sin.data(), false);
        int bad = 0;
        for (size_t i = 0; i < q.size(); ++i) bad += bits_of(got[i]) != bits_of(ref[i]);
        CHECK(bad == 0, "%sq_rope: %d elements differ bitwise from the reference (the nope part must be untouched)", tag.c_str(), bad);
    }

    /// after-the-fact check of a row that went through a fake-quantiser: elements may differ from the reference computed on slightly different
    /// pre-quantisation values only by a flip to the neighbouring grid point (<= max_flips elements of the row)
    int count_diff(const std::vector<float>& got, const std::vector<float>& ref) {
        int d = 0;
        for (size_t i = 0; i < got.size(); ++i) d += bits_of(got[i]) != bits_of(ref[i]) && !(got[i] == 0.0f && ref[i] == 0.0f);
        return d;
    }

    void swa_kv_and_index_k() {
        const int D = G::kHeadDim, R = G::kRopeDim;
        // swa_kv: rmsnorm -> rope -> (fp8) -> ring row;  index_k_finish: rmsnorm -> rope -> (fp4 e8m0)
        for (int which = 0; which < 2; ++which) {
            const int n = which == 0 ? D : G::kIdxDim;
            for (int quant = 0; quant < 2; ++quant) {
                const std::vector<float> raw = randv(n, 2.0f), nw = randv(n, 1.0f);
                std::vector<float> cos, sin;
                rope_row(cos, sin);
                Up<float> draw(dev, raw), dnw(dev, nw), dc(dev, cos), ds(dev, sin);
                DevBuf<float> dout(dev, (size_t) n);
                if (which == 0) K::swa_kv(draw.p(), dnw.p(), 1e-20f, dc.p(), ds.p(), quant != 0, dout.p, nullptr);
                else K::index_k_finish(draw.p(), dnw.p(), 1e-20f, dc.p(), ds.p(), quant != 0, dout.p, nullptr);
                const auto got = dout.down();
                double ss = 0;
                for (float v : raw) ss += (double) v * v;
                const double den = std::sqrt(ss / n + 1e-20);
                std::vector<float> ref((size_t) n);
                for (int i = 0; i < n; ++i) ref[(size_t) i] = (float) ((double) nw[(size_t) i] * raw[(size_t) i] / den);
                rope_tail_ref(ref.data(), n, R, cos.data(), sin.data(), false);
                if (!quant) {
                    const double e = rel_err(got.data(), ref.data(), n);
                    CHECK(e < 3e-6, "%s%s (no quant): relative error %.3g", tag.c_str(), which == 0 ? "swa_kv" : "index_k_finish", e);
                } else {
                    std::vector<float> q((size_t) n);
                    fq_ref(which == 0 ? 0 : 1, ref.data(), n, q.data());
                    const int d = count_diff(got, q);
                    CHECK(d <= 2, "%s%s (quant): %d elements differ from fq(reference) (> 2: not rounding noise)", tag.c_str(), which == 0 ? "swa_kv" : "index_k_finish", d);
                }
            }
        }
        // index_q_finish is exact: rope (bit-exact) then the quantiser (bit-exact) on identical inputs
        for (int quant = 0; quant < 2; ++quant) {
            const int H = G::kIdxHeads, ID = G::kIdxDim;
            std::vector<float> iq = randv(H * ID, 2.0f), cos, sin;
            rope_row(cos, sin);
            Up<float> diq(dev, iq), dc(dev, cos), ds(dev, sin);
            K::index_q_finish(diq.p(), dc.p(), ds.p(), quant != 0, nullptr);
            const auto got = diq.b.down();
            std::vector<float> ref = iq;
            for (int h = 0; h < H; ++h) {
                rope_tail_ref(ref.data() + (size_t) h * ID, ID, R, cos.data(), sin.data(), false);
                if (quant) {
                    std::vector<float> t(ref.begin() + (size_t) h * ID, ref.begin() + (size_t) (h + 1) * ID);
                    fq_ref(1, t.data(), ID, ref.data() + (size_t) h * ID);
                }
            }
            CHECK(count_diff(got, ref) == 0, "%sindex_q_finish (quant %d): not bit-exact", tag.c_str(), quant);
        }
    }

    void compress() {
        const int D = G::kHeadDim, R = G::kRopeDim;
        for (int ratio : {1, 2, 4}) {
            for (int quant = 0; quant < 2; ++quant) {
                const std::vector<float> nw = randv(D, 1.0f);
                std::vector<float> cos, sin;
                rope_row(cos, sin);
                Up<float> dnw(dev, nw), dc(dev, cos), ds(dev, sin);
                std::vector<float> st_kv((size_t) ratio * D, 0.0f), st_sc((size_t) ratio * D, -INFINITY);
                Up<float> dkvs(dev, st_kv), dscs(dev, st_sc);
                DevBuf<float> dlat(dev, (size_t) D), dcomp(dev, (size_t) D);
                std::vector<float> ref_kv((size_t) ratio * D), ref_sc((size_t) ratio * D);
                int fq_bad = 0;
                for (int step = 0; step < 3 * ratio; ++step) {
                    const std::vector<float> kvc = randv(D, 1.5f), scc = randv(D, 2.0f);
                    Up<float> dkv(dev, kvc), dsc(dev, scc);
                    const int slot = step % ratio;
                    std::copy(kvc.begin(), kvc.end(), ref_kv.begin() + (size_t) slot * D);
                    std::copy(scc.begin(), scc.end(), ref_sc.begin() + (size_t) slot * D);
                    AttnCompressArgs a;
                    a.kv_c = dkv.p();
                    a.score_c = ratio > 1 ? dsc.p() : nullptr;
                    a.kv_state = ratio > 1 ? dkvs.p() : nullptr;
                    a.score_state = ratio > 1 ? dscs.p() : nullptr;
                    a.ratio = ratio;
                    a.slot = slot;
                    a.complete = slot == ratio - 1 ? 1 : 0;
                    a.norm_w = dnw.p();
                    a.eps = 1e-20f;
                    a.cos_row = dc.p();
                    a.sin_row = ds.p();
                    a.fp4 = quant;
                    a.latent_out = dlat.p;
                    a.comp_row = dcomp.p;
                    dlat.zero();
                    dcomp.zero();
                    K::compress_step(a, nullptr);
                    const auto lat = dlat.down(), comp = dcomp.down();
                    if (!a.complete) {
                        bool untouched = true;
                        for (float v : lat) untouched = untouched && v == 0.0f;
                        for (float v : comp) untouched = untouched && v == 0.0f;
                        CHECK(untouched, "%scompress ratio %d step %d: an incomplete group wrote an output", tag.c_str(), ratio, step);
                        continue;
                    }
                    // reference in double
                    std::vector<float> ref_lat((size_t) D);
                    {
                        std::vector<double> c((size_t) D);
                        for (int e = 0; e < D; ++e) {
                            double m = -INFINITY;
                            if (ratio == 1) {
                                c[(size_t) e] = ref_kv[(size_t) e];
                                continue;
                            }
                            for (int r = 0; r < ratio; ++r) m = std::max(m, (double) ref_sc[(size_t) r * D + e]);
                            double sum = 0, acc = 0;
                            for (int r = 0; r < ratio; ++r) {
                                const double ex = std::exp((double) ref_sc[(size_t) r * D + e] - m);
                                sum += ex;
                                acc += ex * ref_kv[(size_t) r * D + e];
                            }
                            c[(size_t) e] = acc / sum;
                        }
                        double ss = 0;
                        for (double v : c) ss += v * v;
                        const double den = std::sqrt(ss / D + 1e-20);
                        for (int e = 0; e < D; ++e) ref_lat[(size_t) e] = (float) ((double) nw[(size_t) e] * c[(size_t) e] / den);
                    }
                    const double el = rel_err(lat.data(), ref_lat.data(), D);
                    CHECK(el < 4e-6, "%scompress ratio %d: latent relative error %.3g", tag.c_str(), ratio, el);
                    std::vector<float> ref_comp = ref_lat;
                    rope_tail_ref(ref_comp.data(), D, R, cos.data(), sin.data(), false);
                    if (!quant) {
                        const double ec = rel_err(comp.data(), ref_comp.data(), D);
                        CHECK(ec < 4e-6, "%scompress ratio %d: comp row relative error %.3g", tag.c_str(), ratio, ec);
                    } else {
                        std::vector<float> q((size_t) D);
                        fq_ref(2, ref_comp.data(), D, q.data());
                        fq_bad += count_diff(comp, q);
                    }
                }
                if (quant) CHECK(fq_bad <= 4, "%scompress ratio %d: %d cache elements differ from fp4(reference) (> 4: not rounding noise)", tag.c_str(), ratio, fq_bad);
            }
        }
    }

    void index_scores_and_blocks() {
        const int H = G::kIdxHeads, D = G::kIdxDim;
        for (int n : {1, 7, 128, 129, 333, G::kIdxTopK * 8 + 5, (G::kHeadDim >= 512 ? 2049 : 700)}) {
            const std::vector<float> iq = randv(H * D, 1.0f), wp = randv(H, 1.0f), ik = randv(n * D, 1.0f);
            const float wscale = (float) (std::pow((double) D, -0.5) * std::pow((double) H, -0.5));
            const int cb = G::kCandBlock, nblk = (n + cb - 1) / cb;
            std::vector<uint8_t> flags((size_t) nblk);
            for (auto& f : flags) f = rng.below(3) != 0;
            Up<float> diq(dev, iq), dwp(dev, wp), dik(dev, ik);
            Up<uint8_t> dfl(dev, flags);
            DevBuf<float> dsc(dev, (size_t) n), dsc2(dev, (size_t) n);
            K::index_scores(diq.p(), dwp.p(), wscale, dik.p(), n, nullptr, cb, dsc.p, nullptr);
            K::index_scores(diq.p(), dwp.p(), wscale, dik.p(), n, dfl.p(), cb, dsc2.p, nullptr);
            const auto got = dsc.down(), got2 = dsc2.down();
            std::vector<float> ref((size_t) n), ref2((size_t) n);
            for (int t = 0; t < n; ++t) {
                double s = 0;
                for (int h = 0; h < H; ++h) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) dot += (double) iq[(size_t) h * D + d] * ik[(size_t) t * D + d];
                    s += std::max(dot, 0.0) * (double) (wp[(size_t) h] * wscale);
                }
                ref[(size_t) t] = (float) s;
                ref2[(size_t) t] = flags[(size_t) (t / cb)] ? (float) s : -INFINITY;
            }
            const double e = rel_err(got.data(), ref.data(), n), e2 = rel_err(got2.data(), ref2.data(), n);
            CHECK(e < 5e-6, "%sindex_scores n=%d: relative error %.3g", tag.c_str(), n, e);
            CHECK(e2 < 5e-6, "%sindex_scores (candidate mask) n=%d: error %.3g (a masked position must be exactly -inf)", tag.c_str(), n, e2);
            // block scores of the unmasked row, with some -inf holes
            std::vector<float> sc = got;
            for (int t = 0; t < n; t += 5) sc[(size_t) t] = -INFINITY;
            Up<float> dsc3(dev, sc);
            DevBuf<float> dbs(dev, (size_t) nblk);
            K::block_scores(dsc3.p(), n, cb, dbs.p, nullptr);
            const auto bs = dbs.down();
            bool ok = true;
            for (int b = 0; b < nblk; ++b) {
                float m = -INFINITY;
                for (int t = b * cb; t < std::min(n, (b + 1) * cb); ++t) m = std::max(m, sc[(size_t) t]);
                if (b == (n - 1) / cb) m = INFINITY;
                ok = ok && bits_of(m) == bits_of(bs[(size_t) b]);
            }
            CHECK(ok, "%sblock_scores n=%d", tag.c_str(), n);
        }
    }

    void topk() {
        struct Case { int n, k, kind; };
        std::vector<Case> cases = {{0, 0, 0}, {1, 1, 0}, {5, 8, 0}, {8, 8, 0}, {9, 8, 0}, {300, 8, 0}, {300, 8, 1}, {300, 8, 2}, {1000, 512, 0}, {1000, 512, 1},
                                   {1000, 512, 3}, {777, 0, 0}, {2049, 2048, 1}, {256, 256, 4}, {257, 256, 4}, {4096, G::kIdxTopK, 2}, {4096, G::kCandTopBlocks, 1}};
        if (G::kIdxTopK >= 512) {
            cases.push_back({16384, 2048, 0});
            cases.push_back({16384, 512, 2});
            cases.push_back({9000, 2048, 3});
        }
        for (const Case& c : cases) {
            // kind: 0 random floats; 1 values from a small set (many ties); 2 ties + -inf holes + +inf pins; 3 NaN and +-0 and -inf mixed; 4 all equal
            std::vector<float> s((size_t) c.n);
            for (int i = 0; i < c.n; ++i) {
                float v = rng.normal();
                if (c.kind == 1 || c.kind == 2) v = (float) rng.below(9) * 0.125f - 0.5f;
                if (c.kind == 2 && rng.below(6) == 0) v = -INFINITY;
                if (c.kind == 2 && rng.below(50) == 0) v = INFINITY;
                if (c.kind == 3) {
                    const int r = rng.below(10);
                    v = r == 0 ? NAN : r == 1 ? -INFINITY : r == 2 ? 0.0f : r == 3 ? -0.0f : r == 4 ? INFINITY : v;
                }
                if (c.kind == 4) v = 1.0f;
                s[(size_t) i] = v;
            }
            std::vector<int> sel;
            std::vector<uint8_t> flags;
            topk_ref(s, c.k, sel, flags);
            const int out_len = std::max(c.k, 1) + 3;
            Up<float> ds(dev, s);
            DevBuf<int32_t> didx(dev, (size_t) out_len);
            DevBuf<uint8_t> dfl(dev, (size_t) std::max(c.n, 1));
            std::vector<int32_t> fill((size_t) out_len, 12345);
            didx.up(fill);
            std::vector<uint8_t> ffill((size_t) std::max(c.n, 1), 77);
            dfl.up(ffill);
            K::topk_select(ds.p(), c.n, c.k, didx.p, out_len, dfl.p, nullptr);
            const auto gi = didx.down();
            const auto gf = dfl.down();
            std::vector<int32_t> want((size_t) out_len, -1);
            for (size_t i = 0; i < sel.size(); ++i) want[i] = sel[i];
            CHECK(gi == want, "%stopk n=%d k=%d kind=%d: indices differ (%zu selected expected)", tag.c_str(), c.n, c.k, c.kind, sel.size());
            bool fl = true;
            for (int i = 0; i < c.n; ++i) fl = fl && gf[(size_t) i] == flags[(size_t) i];
            CHECK(fl, "%stopk n=%d k=%d kind=%d: flags differ", tag.c_str(), c.n, c.k, c.kind);
            // each output on its own (the other pointer null)
            DevBuf<int32_t> didx2(dev, (size_t) out_len);
            K::topk_select(ds.p(), c.n, c.k, didx2.p, out_len, nullptr, nullptr);
            CHECK(didx2.down() == want, "%stopk n=%d k=%d kind=%d: indices-only call differs", tag.c_str(), c.n, c.k, c.kind);
        }
    }

    void sparse_attn() {
        const int D = G::kHeadDim, R = G::kRopeDim, H = G::kHeads, W = G::kWindow, K_ = G::kIdxTopK;
        struct Case { int pos; int n_comp; int n_sel; bool swa; float qscale; float sink; };
        std::vector<Case> cases = {{0, 0, 0, true, 1.0f, 0.0f},        {3, 0, 0, true, 1.0f, -2.0f},    {W - 1, 0, 0, true, 1.0f, 0.5f},     {W, 0, 0, true, 2.0f, 0.0f},
                                   {3 * W + 5, 0, 0, true, 1.0f, 1.0f}, {W + 2, 40, 7, false, 1.0f, 0.0f}, {2, 100, K_, false, 1.5f, 0.3f}, {5 * W - 1, 300, K_, false, 1.0f, 8.0f},
                                   {W + 1, 25, 0, false, 1.0f, 0.0f},   {7 * W, 600, K_ - 3, false, 3.0f, -30.0f}};
        for (const Case& c : cases) {
            const std::vector<float> q = randv(H * D, c.qscale), win = randv(W * D, 1.0f), sink = randv(H, 2.0f);
            std::vector<float> sk = sink;
            for (auto& v : sk) v += c.sink;
            const std::vector<float> comp = randv(std::max(c.n_comp, 1) * D, 1.0f);
            std::vector<int32_t> tk((size_t) K_, -1);
            {
                std::vector<int> pool(c.n_comp);
                for (int i = 0; i < c.n_comp; ++i) pool[(size_t) i] = i;
                for (int i = 0; i < c.n_sel && i < c.n_comp; ++i) std::swap(pool[(size_t) i], pool[(size_t) (i + rng.below(c.n_comp - i))]);
                std::vector<int> pick(pool.begin(), pool.begin() + std::min(c.n_sel, c.n_comp));
                std::sort(pick.begin(), pick.end());
                for (size_t i = 0; i < pick.size(); ++i) tk[i] = pick[i];
            }
            std::vector<float> cos, sin;
            rope_row(cos, sin);
            Up<float> dq(dev, q), dwin(dev, win), dsink(dev, sk), dcomp(dev, comp), dc(dev, cos), ds(dev, sin);
            Up<int32_t> dtk(dev, tk);
            DevBuf<float> dout(dev, (size_t) H * D);
            K::sparse_attn(dq.p(), dwin.p(), c.pos, c.swa ? nullptr : dcomp.p(), c.swa ? nullptr : dtk.p(), dsink.p(), dc.p(), ds.p(), dout.p, nullptr);
            const auto got = dout.down();
            // reference
            std::vector<const float*> rows;
            for (int j = 0; j < W; ++j) {
                const int slot = (c.pos % W + 1 + j) % W;
                if (slot <= c.pos) rows.push_back(win.data() + (size_t) slot * D);
            }
            if (!c.swa)
                for (int i = 0; i < K_; ++i)
                    if (tk[(size_t) i] >= 0) rows.push_back(comp.data() + (size_t) tk[(size_t) i] * D);
            const double scale = (double) (float) (1.0 / std::sqrt((double) D));
            std::vector<float> ref((size_t) H * D);
            for (int h = 0; h < H; ++h) {
                std::vector<double> l(rows.size());
                double m = sk[(size_t) h];
                for (size_t j = 0; j < rows.size(); ++j) {
                    double dot = 0;
                    for (int d = 0; d < D; ++d) dot += (double) q[(size_t) h * D + d] * rows[j][d];
                    l[j] = dot * scale;
                    m = std::max(m, l[j]);
                }
                double den = std::exp((double) sk[(size_t) h] - m);
                for (double& v : l) {
                    v = std::exp(v - m);
                    den += v;
                }
                std::vector<float> o((size_t) D, 0.0f);
                for (int d = 0; d < D; ++d) {
                    double acc = 0;
                    for (size_t j = 0; j < rows.size(); ++j) acc += l[j] * rows[j][d];
                    o[(size_t) d] = (float) (acc / den);
                }
                rope_tail_ref(o.data(), D, R, cos.data(), sin.data(), true);
                std::copy(o.begin(), o.end(), ref.begin() + (size_t) h * D);
            }
            const double e = rel_err(got.data(), ref.data(), H * D);
            CHECK(e < 3e-5, "%ssparse_attn pos=%d comp=%d sel=%d%s sink+%.1f: relative error %.3g (%zu keys)", tag.c_str(), c.pos, c.n_comp, c.n_sel, c.swa ? " swa" : "", c.sink, e, rows.size());
        }
    }

    void run_all() {
        fakequant();
        rmsnorm();
        q_rope();
        swa_kv_and_index_k();
        compress();
        index_scores_and_blocks();
        topk();
        sparse_attn();
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------
// the stand-ins for DS1-B's dense ops (plain C++ on "device" pointers, valid for HostDev); FP64 accumulation, the integer sums exact
// ---------------------------------------------------------------------------------------------------------------------------------
float half_to_float(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t b;
    if (e == 0) {
        if (m == 0) {
            b = s;
        } else {
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 0x400u)) {
                mm <<= 1;
                ++sh;
            }
            b = s | ((uint32_t) (113 - sh) << 23) | ((mm & 0x3FFu) << 13);
        }
    } else if (e == 31) {
        b = s | 0x7F800000u | (m << 13);
    } else {
        b = s | ((e + 112u) << 23) | (m << 13);
    }
    return float_of(b);
}
float bf16_to_float(uint16_t h) { return float_of((uint32_t) h << 16); }

struct TestDense : AttnDenseOps {
    void quantize_acts(const float* x, int T, int k, int8_t* xq, float* xs, Stream) override {
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < k / 32; ++b) {
                const float* v = x + (size_t) t * k + (size_t) b * 32;
                uint32_t m = 0;
                for (int j = 0; j < 32; ++j) m = std::max(m, bits_of(v[j]) & 0x7FFFFFFFu);
                const bool nonfinite = m >= 0x7F800000u, tiny = m < 0x0D800000u, special = nonfinite || tiny;
                const float amax = float_of(m);
                const float d = nonfinite ? float_of(0x7FC00000u) : tiny ? 0.0f : amax / 127.0f;
                const float id = special ? 0.0f : 127.0f / amax;
                xs[(size_t) t * (k / 32) + b] = d;
                for (int j = 0; j < 32; ++j) {
                    int q = special ? 0 : (int) std::nearbyintf(v[j] * id);
                    q = std::min(127, std::max(-127, q));
                    xq[(size_t) t * k + (size_t) b * 32 + j] = (int8_t) q;
                }
            }
    }
    void gemv_q8(const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream) override {
        const unsigned char* wb = static_cast<const unsigned char*>(w);
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < n; ++r) {
                double acc = 0;
                for (int b = 0; b < k / 32; ++b) {
                    const unsigned char* blk = wb + ((size_t) r * (k / 32) + (size_t) b) * 34;
                    uint16_t h;
                    std::memcpy(&h, blk, 2);
                    int isum = 0;
                    for (int j = 0; j < 32; ++j) isum += (int) (int8_t) blk[2 + j] * (int) xq[(size_t) t * k + (size_t) b * 32 + j];
                    acc += (double) half_to_float(h) * (double) xs[(size_t) t * (k / 32) + b] * (double) isum;
                }
                y[(size_t) t * n + r] = (float) acc;
            }
    }
    void gemv_q8_grouped_f32(const void* w, int groups, int rows_per_group, int k_per_group, const float* x, int T, float* y, Stream) override {
        const unsigned char* wb = static_cast<const unsigned char*>(w);
        for (int t = 0; t < T; ++t)
            for (int g = 0; g < groups; ++g)
                for (int r = 0; r < rows_per_group; ++r) {
                    double acc = 0;
                    const size_t row = (size_t) g * rows_per_group + r;
                    for (int d = 0; d < k_per_group; ++d) {
                        const unsigned char* blk = wb + (row * (k_per_group / 32) + (size_t) (d / 32)) * 34;
                        uint16_t h;
                        std::memcpy(&h, blk, 2);
                        const float wv = half_to_float(h) * (float) (int8_t) blk[2 + d % 32];
                        acc += (double) x[(size_t) t * groups * k_per_group + (size_t) g * k_per_group + d] * (double) wv;
                    }
                    y[(size_t) t * groups * rows_per_group + row] = (float) acc;
                }
    }
    void gemv_bf16(const uint16_t* w, int n, int k, const float* x, int T, float* y, Stream) override {
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < n; ++r) {
                double acc = 0;
                for (int d = 0; d < k; ++d) acc += (double) x[(size_t) t * k + d] * (double) bf16_to_float(w[(size_t) r * k + d]);
                y[(size_t) t * n + r] = (float) acc;
            }
    }
};

// ---------------------------------------------------------------------------------------------------------------------------------
// the oracle's golden data
// ---------------------------------------------------------------------------------------------------------------------------------
struct Rec {
    int dtype = 0;                      // 0 f32, 1 i32, 2 u8, 3 u16
    std::vector<uint32_t> dims;
    std::vector<unsigned char> data;
    size_t count() const {
        size_t n = 1;
        for (uint32_t d : dims) n *= d;
        return n;
    }
    const float* f() const { return reinterpret_cast<const float*>(data.data()); }
    const int32_t* i() const { return reinterpret_cast<const int32_t*>(data.data()); }
    const uint8_t* u8() const { return data.data(); }
};
using Golden = std::map<std::string, Rec>;

bool load_golden(const std::string& path, Golden& g) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    uint32_t ver = 0, n = 0;
    bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, "DS1CGOLD", 8) == 0 && std::fread(&ver, 4, 1, f) == 1 && std::fread(&n, 4, 1, f) == 1 && ver == 1;
    for (uint32_t r = 0; ok && r < n; ++r) {
        uint32_t nl = 0;
        ok = std::fread(&nl, 4, 1, f) == 1;
        std::string name(nl, '\0');
        ok = ok && std::fread(name.data(), 1, nl, f) == nl;
        Rec rec;
        uint32_t dt = 0, nd = 0;
        ok = ok && std::fread(&dt, 4, 1, f) == 1 && std::fread(&nd, 4, 1, f) == 1;
        rec.dtype = (int) dt;
        rec.dims.resize(nd);
        for (uint32_t i = 0; ok && i < nd; ++i) ok = std::fread(&rec.dims[i], 4, 1, f) == 1;
        const size_t esize = dt == 0 || dt == 1 ? 4 : dt == 2 ? 1 : 2;
        rec.data.resize(rec.count() * esize + 16);
        rec.data.resize(rec.count() * esize);
        ok = ok && (rec.data.empty() || std::fread(rec.data.data(), 1, rec.data.size(), f) == rec.data.size());
        g[name] = std::move(rec);
    }
    std::fclose(f);
    return ok;
}

const Rec& need(const Golden& g, const std::string& k) {
    auto it = g.find(k);
    if (it == g.end()) {
        std::printf("FAIL golden record '%s' is missing\n", k.c_str());
        std::exit(2);
    }
    return it->second;
}
std::string key(const char* what, int l) { return std::string(what) + "." + std::to_string(l); }

// ---------------------------------------------------------------------------------------------------------------------------------
// the replay
// ---------------------------------------------------------------------------------------------------------------------------------
constexpr double kTight = 3e-5, kLoose = 5e-2;

struct StageStat {
    double max_err = 0;
    long n = 0, events = 0;
    void add(double e) {
        max_err = std::max(max_err, e);
        ++n;
    }
};

struct Replay {
    using G = MiniGeom;
    static constexpr int HID = G::kHidden, D = G::kHeadDim, Q = G::kHeads * G::kHeadDim, TK = G::kIdxTopK, ID = G::kIdxDim;

    Dev& dev;
    const Golden& g;
    std::string name;
    TestDense dense;
    int nl = 0, n_pos = 0;
    Ds41LayerRoles roles;
    AttnQuantFlags qf;
    float eps = 1e-20f;
    std::vector<AttnLayerWeights> W;
    std::vector<void*> allocs;
    std::map<std::string, StageStat> stats;
    std::vector<std::vector<float>> out_all;            // [layer][pos * HID + i]  the engine's outputs (T = 1 run)
    std::vector<std::vector<int32_t>> topk_all;         // [layer][pos * TK + i]
    long events_total = 0, rows_total = 0;

    Replay(Dev& d, const Golden& gg, std::string nm) : dev(d), g(gg), name(std::move(nm)) {}
    ~Replay() {
        for (void* p : allocs) dev.release(p);
    }

    void* up(const Rec& r) {
        void* p = dev.alloc(r.data.size() + 16);
        if (!r.data.empty()) dev.h2d(p, r.data.data(), r.data.size());
        allocs.push_back(p);
        return p;
    }
    const void* opt(int l, const char* nm) {
        const std::string k = "w." + std::to_string(l) + "." + nm;
        auto it = g.find(k);
        return it == g.end() ? nullptr : up(it->second);
    }

    void setup() {
        const Rec& ratios = need(g, "cfg.ratios");
        nl = (int) ratios.count();
        n_pos = need(g, "cfg.n_pos").i()[0];
        roles.compress_ratio.assign(ratios.i(), ratios.i() + nl);
        const Rec& kvs = need(g, "cfg.kv_source");
        roles.kv_source_layers.assign(kvs.i(), kvs.i() + kvs.count());
        const Rec& ixs = need(g, "cfg.index_source");
        roles.index_source_layers.assign(ixs.i(), ixs.i() + ixs.count());
        roles.candidate_source_layer = need(g, "cfg.cand_source").i()[0];
        const int32_t* q = need(g, "cfg.quant").i();
        qf.window_kv = q[1] != 0;
        qf.compressed_kv = q[2] != 0;
        qf.index = q[3] != 0;
        eps = need(g, "cfg.eps").f()[0];
        W.assign((size_t) nl, AttnLayerWeights{});
        for (int l = 0; l < nl; ++l) {
            AttnLayerWeights& w = W[(size_t) l];
            w.wq_a = opt(l, "wq_a.weight");
            w.q_norm = static_cast<const float*>(opt(l, "q_norm.weight"));
            w.wq_b = opt(l, "wq_b.weight");
            w.wkv = opt(l, "wkv.weight");
            w.kv_norm = static_cast<const float*>(opt(l, "kv_norm.weight"));
            w.sink = static_cast<const float*>(opt(l, "attn_sink"));
            w.wo_a = opt(l, "wo_a.weight");
            w.wo_b = opt(l, "wo_b.weight");
            w.rope.cos = static_cast<const float*>(up(need(g, "rope." + std::to_string(l) + ".cos")));
            w.rope.sin = static_cast<const float*>(up(need(g, "rope." + std::to_string(l) + ".sin")));
            w.comp_wkv = static_cast<const uint16_t*>(opt(l, "compressor.wkv.weight"));
            w.comp_wgate = static_cast<const uint16_t*>(opt(l, "compressor.wgate.weight"));
            w.comp_norm = static_cast<const float*>(opt(l, "compressor.norm.weight"));
            w.idx_wk = static_cast<const uint16_t*>(opt(l, "indexer.wk.weight"));
            w.idx_k_norm = static_cast<const float*>(opt(l, "indexer.k_norm.weight"));
            w.idx_wq_b = opt(l, "indexer.wq_b.weight");
            w.idx_weights_proj = static_cast<const uint16_t*>(opt(l, "indexer.weights_proj.weight"));
        }
        out_all.assign((size_t) nl, std::vector<float>((size_t) n_pos * HID, NAN));
        topk_all.assign((size_t) nl, std::vector<int32_t>((size_t) n_pos * TK, -2));
    }

    void init(Ds41Attention<G>& att) {
        att.init(roles, n_pos + 8, qf, eps);
        for (int l = 0; l < nl; ++l) att.set_weights(l, W[(size_t) l]);
    }

    // ---- one quantised cache row: bit-exact, or a counted flip that is then overwritten by the oracle's row ----
    // returns true on a flip
    bool quant_row(const char* stage, const float* got, const float* ref, int n, int max_diff, bool quantised, float* resync_dst) {
        StageStat& st = stats[stage];
        if (!quantised) {
            const double e = rel_err(got, ref, n);
            st.add(e);
            if (e > kTight) {
                ++st.events;
                CHECK(e <= kLoose, "[%s] %s: relative error %.3g", name.c_str(), stage, e);
                if (resync_dst) dev.h2d(resync_dst, ref, (size_t) n * sizeof(float));
                return true;
            }
            return false;
        }
        int diff = 0;
        double mx = 0, md = 0;
        for (int i = 0; i < n; ++i) {
            mx = std::max(mx, (double) std::fabs(ref[i]));
            if (bits_of(got[i]) != bits_of(ref[i]) && !(got[i] == 0.0f && ref[i] == 0.0f)) {
                ++diff;
                md = std::max(md, (double) std::fabs((double) got[i] - (double) ref[i]));
            }
        }
        st.add(diff ? md / std::max(mx, 1e-30) : 0.0);
        if (diff == 0) return false;
        ++st.events;
        CHECK(diff <= max_diff && md <= 0.45 * mx, "[%s] %s: %d elements differ from the oracle's (> %d or the step is > 45%% of the row max: not a rounding flip)", name.c_str(), stage, diff, max_diff);
        if (resync_dst) dev.h2d(resync_dst, ref, (size_t) n * sizeof(float));
        return true;
    }

    /// near-tie / event explanation of a selection that differs from the oracle's
    bool explained_diff(const std::vector<int>& got, const std::vector<int>& want, const float* oracle_scores, int n, double scale, int k) {
        std::vector<float> s(oracle_scores, oracle_scores + n);
        std::vector<float> srt;
        for (float v : s)
            if (v == v && bits_of(v) != 0xFF800000u) srt.push_back(v);
        std::sort(srt.begin(), srt.end(), std::greater<float>());
        const double b = srt.empty() ? 0.0 : (double) srt[(size_t) std::min<int>((int) srt.size(), k) - 1];
        std::vector<int> d;
        std::set_symmetric_difference(got.begin(), got.end(), want.begin(), want.end(), std::back_inserter(d));
        for (int i : d) {
            if (i < 0 || i >= n) return false;
            const double v = (double) s[(size_t) i];
            if (!(std::fabs(v - b) <= 3e-5 * scale)) return false;
        }
        return true;
    }

    std::vector<int> to_list(const int32_t* a, int n) {
        std::vector<int> v;
        for (int i = 0; i < n; ++i)
            if (a[i] >= 0) v.push_back(a[i]);
        return v;
    }

    // ---- compare every stage of (layer l, position p) right after forward() ----
    void compare(Ds41Attention<G>& att, int l, int p, const float* out_dev, bool& topk_div, bool& cand_div) {
        const int ratio = roles.compress_ratio[(size_t) l];
        const AttnMode mode = att.mode(l);
        bool explained = false;
        auto ref_row = [&](const char* what, int width) { return need(g, key(what, l)).f() + (size_t) p * width; };
        ++rows_total;

        // q
        {
            const auto q = down(dev, att.trace_q(0), Q);
            const double e = rel_err(q.data(), ref_row("q", Q), Q);
            StageStat& st = stats["q"];
            st.add(e);
            if (e > kTight) {
                ++st.events;
                explained = true;
                CHECK(e <= kLoose, "[%s] L%d p%d q: relative error %.3g", name.c_str(), l, p, e);
            }
        }
        // SWA KV row
        {
            const float* ring = att.kv_win_row(l, p);
            const auto row = down(dev, ring, D);
            if (quant_row("kv_win", row.data(), ref_row("kvwin", D), D, 4, qf.window_kv, const_cast<float*>(ring))) explained = true;
        }
        // compressor: the latent, the cache rows
        if (mode == AttnMode::kFull) {
            const float* lat = att.trace_latent(0);
            const float* rl = ref_row("latent", D);
            const bool ref_valid = !std::isnan(rl[0]);
            CHECK((lat != nullptr) == ref_valid, "[%s] L%d p%d: the group latent is %s, the oracle's is %s", name.c_str(), l, p, lat ? "present" : "absent", ref_valid ? "present" : "absent");
            if (lat && ref_valid) {
                const auto v = down(dev, lat, D);
                const double e = rel_err(v.data(), rl, D);
                stats["latent"].add(e);
                CHECK(e <= kTight, "[%s] L%d p%d latent: relative error %.3g", name.c_str(), l, p, e);
                const int gi = p / ratio;
                const float* cr = att.comp_kv_row(l, gi);
                const float* ir = att.index_k_row(l, gi);
                const auto comp = down(dev, cr, D), ik = down(dev, ir, ID);
                if (quant_row("comp_kv", comp.data(), ref_row("comp", D), D, 16, qf.compressed_kv, const_cast<float*>(cr))) explained = true;
                if (quant_row("index_k", ik.data(), ref_row("idxk", ID), ID, 8, qf.index, const_cast<float*>(ir))) explained = true;
            }
        }
        // indexer: scores, candidate pool, top-k
        const bool is_index = mode == AttnMode::kFull || mode == AttnMode::kReindex;
        bool scores_event = false;
        if (ratio > 0) {
            const int clen = (p + 1) / ratio;
            std::vector<float> sc;
            int sn = 0;
            if (is_index && clen > 0) {
                const float* sp = att.trace_scores(&sn);
                CHECK(sp != nullptr && sn == clen, "[%s] L%d p%d: trace_scores length %d, expected %d", name.c_str(), l, p, sn, clen);
                if (sp && sn == clen) {
                    sc = down(dev, sp, (size_t) sn);
                    const float* os = need(g, key("idx_score", l)).f() + (size_t) p * n_pos;
                    double scale = 0;
                    for (int i = 0; i < sn; ++i)
                        if (std::isfinite(os[i])) scale = std::max(scale, (double) std::fabs(os[i]));
                    const double e = rel_err(sc.data(), os, sn);
                    stats["idx_score"].add(e);
                    if (e > kTight) {
                        ++stats["idx_score"].events;
                        scores_event = true;
                        explained = true;
                        CHECK(e <= 0.5, "[%s] L%d p%d idx_score: relative error %.3g", name.c_str(), l, p, e);
                    }
                    for (int i = 0; i < sn; ++i)
                        CHECK(std::isinf(sc[(size_t) i]) == std::isinf(os[i]) && (!std::isinf(os[i]) || sc[(size_t) i] == os[i]), "[%s] L%d p%d idx_score[%d]: reachability differs (%g vs %g)", name.c_str(), l, p, i, sc[(size_t) i], os[i]);
                    // candidate pool of the source layer
                    if (l == roles.candidate_source_layer) {
                        int nb = 0, bn = 0;
                        const uint8_t* cf = att.trace_cand(0, &nb);
                        const float* bsp = att.trace_block_scores(&bn);
                        CHECK(cf && bsp && nb == bn && nb == (clen + G::kCandBlock - 1) / G::kCandBlock, "[%s] L%d p%d: candidate trace lengths %d / %d", name.c_str(), l, p, nb, bn);
                        if (cf && bsp && nb == bn) {
                            std::vector<uint8_t> flags((size_t) nb);
                            dev.d2h(flags.data(), cf, (size_t) nb);
                            const auto bs = down(dev, bsp, (size_t) nb);
                            // self-consistency: the flags are the top blocks of the engine's OWN block scores
                            std::vector<int> sel;
                            std::vector<uint8_t> wf;
                            topk_ref(bs, std::min(G::kCandTopBlocks, nb), sel, wf);
                            CHECK(wf == flags, "[%s] L%d p%d: the candidate flags are not the top blocks of the engine's own block scores", name.c_str(), l, p);
                            const float* ob = need(g, key("blk_score", l)).f() + (size_t) p * need(g, key("blk_score", l)).dims[1];
                            const uint8_t* oc = need(g, key("cand", l)).u8() + (size_t) p * need(g, key("cand", l)).dims[1];
                            const double eb = rel_err(bs.data(), ob, nb);
                            stats["blk_score"].add(eb);
                            if (eb > kTight) ++stats["blk_score"].events;
                            std::vector<int> a, b;
                            for (int i = 0; i < nb; ++i) {
                                if (flags[(size_t) i]) a.push_back(i);
                                if (oc[i]) b.push_back(i);
                            }
                            cand_div = a != b;
                            if (cand_div) {
                                ++stats["cand"].events;
                                CHECK(eb > kTight || explained_diff(a, b, ob, nb, scale, std::min(G::kCandTopBlocks, nb)), "[%s] L%d p%d: the candidate pool differs from the oracle's and it is not a near-tie", name.c_str(), l, p);
                                explained = true;
                            }
                            stats["cand"].add(cand_div ? 1.0 : 0.0);
                        }
                    }
                    // self-consistency of the top-k: exactly the top-k of the engine's own scores
                    std::vector<int> sel;
                    std::vector<uint8_t> wf;
                    topk_ref(sc, std::min(TK, clen), sel, wf);
                    std::vector<int32_t> mine((size_t) TK);
                    dev.d2h(mine.data(), att.trace_topk(0), (size_t) TK * 4);
                    std::vector<int32_t> want((size_t) TK, -1);
                    for (size_t i = 0; i < sel.size(); ++i) want[i] = sel[i];
                    CHECK(mine == want, "[%s] L%d p%d: the top-k is not the top-k of the engine's own scores", name.c_str(), l, p);
                    // against the oracle
                    const int32_t* ot = need(g, key("topk", l)).i() + (size_t) p * TK;
                    const std::vector<int> a = to_list(mine.data(), TK), b = to_list(ot, TK);
                    stats["topk"].add(a == b ? 0.0 : 1.0);
                    topk_div = a != b;
                    if (topk_div) {
                        ++stats["topk"].events;
                        const bool ok = scores_event || (l == roles.candidate_source_layer ? false : cand_div) || explained_diff(a, b, os, sn, scale, std::min(TK, clen));
                        CHECK(ok, "[%s] L%d p%d: the top-k differs from the oracle's and it is not a near-tie (engine %zu entries, oracle %zu)", name.c_str(), l, p, a.size(), b.size());
                        explained = true;
                    }
                }
            } else if (clen == 0 && ratio > 0) {
                std::vector<int32_t> mine((size_t) TK);
                dev.d2h(mine.data(), att.trace_topk(0), (size_t) TK * 4);
                bool allneg = true;
                for (int32_t v : mine) allneg = allneg && v < 0;
                CHECK(allneg, "[%s] L%d p%d: no compressed entry yet but the top-k row is not all -1", name.c_str(), l, p);
                topk_div = false;
            } else if (mode == AttnMode::kReuse) {
                std::vector<int32_t> mine((size_t) TK);
                dev.d2h(mine.data(), att.trace_topk(0), (size_t) TK * 4);
                const int32_t* ot = need(g, key("topk", l)).i() + (size_t) p * TK;
                const bool same = to_list(mine.data(), TK) == to_list(ot, TK);
                stats["topk_reuse"].add(same ? 0.0 : 1.0);
                CHECK(same || topk_div, "[%s] L%d p%d: a REUSE layer's top-k differs from the oracle's although its index source agreed", name.c_str(), l, p);
                if (!same) explained = true;
            }
            if (is_index || mode == AttnMode::kReuse) {
                std::vector<int32_t> mine((size_t) TK);
                dev.d2h(mine.data(), att.trace_topk(0), (size_t) TK * 4);
                std::copy(mine.begin(), mine.end(), topk_all[(size_t) l].begin() + (size_t) p * TK);
            }
        }
        if (topk_div && mode == AttnMode::kReuse) explained = true;
        // o (after the inverse RoPE) and the layer output
        {
            const auto o = down(dev, att.trace_o(0), Q);
            const double e = rel_err(o.data(), ref_row("o", Q), Q);
            stats["o"].add(e);
            if (explained) {
                if (e > kTight) ++stats["o"].events;
                CHECK(e <= 0.5, "[%s] L%d p%d o: relative error %.3g even with an explained upstream event", name.c_str(), l, p, e);
            } else {
                CHECK(e <= kTight, "[%s] L%d p%d o: relative error %.3g", name.c_str(), l, p, e);
            }
            const auto out = down(dev, out_dev, HID);
            std::copy(out.begin(), out.end(), out_all[(size_t) l].begin() + (size_t) p * HID);
            const double eo = rel_err(out.data(), ref_row("out", HID), HID);
            stats["out"].add(eo);
            if (explained) {
                if (eo > kTight) ++stats["out"].events;
                CHECK(eo <= 0.5, "[%s] L%d p%d out: relative error %.3g even with an explained upstream event", name.c_str(), l, p, eo);
            } else {
                CHECK(eo <= kTight, "[%s] L%d p%d out: relative error %.3g", name.c_str(), l, p, eo);
            }
        }
        if (explained) ++events_total;
    }

    /// the whole run, T = 1, compared against the oracle stage by stage
    void run_t1(Ds41Attention<G>& att, int first, int count, bool do_compare, std::vector<float>* outs = nullptr) {
        DevBuf<float> dx(dev, HID), dout(dev, HID);
        for (int p = first; p < first + count; ++p) {
            bool topk_div = false, cand_div = false;
            for (int l = 0; l < nl; ++l) {
                dev.h2d(dx.p, need(g, key("x", l)).f() + (size_t) p * HID, HID * sizeof(float));
                att.forward(l, dx.p, 1, p, dout.p, nullptr);
                if (do_compare) compare(att, l, p, dout.p, topk_div, cand_div);
                if (outs) {
                    const auto o = dout.down();
                    outs->insert(outs->end(), o.begin(), o.end());
                }
            }
        }
    }

    /// windows of T = 1..8 positions, layer by layer: rows must be BIT-identical to the T = 1 run
    void run_windows(Ds41Attention<G>& att) {
        DevBuf<float> dx(dev, 8 * HID), dout(dev, 8 * HID);
        const int pattern[] = {4, 8, 3, 1, 5, 2, 7, 6};
        int p = 0, w = 0;
        long bad = 0, badk = 0, rows = 0;
        while (p < n_pos) {
            const int T = std::min(pattern[w++ % 8], n_pos - p);
            for (int l = 0; l < nl; ++l) {
                std::vector<float> xin((size_t) T * HID);
                std::memcpy(xin.data(), need(g, key("x", l)).f() + (size_t) p * HID, xin.size() * 4);
                dev.h2d(dx.p, xin.data(), xin.size() * 4);
                att.forward(l, dx.p, T, p, dout.p, nullptr);
                const auto o = dout.down((size_t) T * HID);
                for (int t = 0; t < T; ++t) {
                    ++rows;
                    bad += std::memcmp(o.data() + (size_t) t * HID, out_all[(size_t) l].data() + (size_t) (p + t) * HID, HID * 4) != 0;
                    if (roles.compress_ratio[(size_t) l] > 0) {
                        std::vector<int32_t> tk((size_t) TK);
                        dev.d2h(tk.data(), att.trace_topk(t), (size_t) TK * 4);
                        const int32_t* want = topk_all[(size_t) l].data() + (size_t) (p + t) * TK;
                        if (want[0] != -2) badk += std::memcmp(tk.data(), want, (size_t) TK * 4) != 0;
                    }
                }
            }
            p += T;
        }
        CHECK(bad == 0, "[%s] windows T=1..8: %ld of %ld rows differ bitwise from the T=1 run", name.c_str(), bad, rows);
        CHECK(badk == 0, "[%s] windows T=1..8: %ld top-k rows differ from the T=1 run", name.c_str(), badk);
    }

    static uint64_t hash_of(const std::vector<float>& v) {
        uint64_t h = 1469598103934665603ull;
        for (float f : v) {
            const uint32_t b = bits_of(f);
            for (int i = 0; i < 4; ++i) h = (h ^ ((b >> (8 * i)) & 0xFF)) * 1099511628211ull;
        }
        return h;
    }

    void report() {
        std::printf("INFO [%s] %d layers x %d positions: %ld layer-steps, %ld with an explained event (a quantiser flip or a near-tie, see header)\n", name.c_str(), nl, n_pos, rows_total, events_total);
        for (const auto& kv : stats)
            std::printf("INFO [%s]   %-10s compared %6ld  max error %.3g  events %ld\n", name.c_str(), kv.first.c_str(), kv.second.n, kv.second.max_err, kv.second.events);
        const long budget = std::max(4L, rows_total / 100);
        CHECK(events_total <= budget, "[%s] %ld layer-steps had an event (budget %ld): the quantiser flips are not rare", name.c_str(), events_total, budget);
    }
};

void run_oracle(Dev& dev, const std::string& dir, const std::vector<std::string>& only, bool first_extras) {
    const char* names[] = {"decode_all_on", "decode_all_off", "prefill_all_on", "prefill_all_off", "decode_win_only", "decode_ckv_idx"};
    bool first = true;
    int ran = 0;
    for (const char* nm : names) {
        if (!only.empty() && std::find(only.begin(), only.end(), nm) == only.end()) continue;
        Golden g;
        const std::string path = dir + "/scn_" + nm + ".bin";
        if (!load_golden(path, g)) {
            std::printf("FAIL cannot read the golden file %s (run src/ds41/attn/golden/gen_golden.py --out %s)\n", path.c_str(), dir.c_str());
            ++g_fail;
            continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        {
            Replay r(dev, g, nm);
            r.setup();
            Ds41Attention<MiniGeom> att(dev, r.dense);
            r.init(att);
            r.run_t1(att, 0, r.n_pos, true);
            r.report();
            if (first && first_extras) {
                first = false;
                // (1) windows of T = 1..8 on a second instance: bit-identical rows
                {
                    Ds41Attention<MiniGeom> att2(dev, r.dense);
                    r.init(att2);
                    r.run_windows(att2);
                }
                // (2) the same instance after reset(), and two other scheduling orders: bit-identical outputs
                {
                    const int n = std::min(r.n_pos, 40);
                    std::vector<float> a, b, c;
                    const auto order0 = ds41_emu::g_order;
                    const auto seed0 = ds41_emu::g_order_seed;
                    att.reset();
                    r.run_t1(att, 0, n, false, &a);
                    ds41_emu::set_order(ds41_emu::Order::kReverse);
                    att.reset();
                    r.run_t1(att, 0, n, false, &b);
                    ds41_emu::set_order(ds41_emu::Order::kShuffle, 9);
                    att.reset();
                    r.run_t1(att, 0, n, false, &c);
                    ds41_emu::set_order(order0, seed0);
                    // the first n positions of the main run are in out_all
                    std::vector<float> m;
                    for (int p = 0; p < n; ++p)
                        for (int l = 0; l < r.nl; ++l) m.insert(m.end(), r.out_all[(size_t) l].begin() + (size_t) p * Replay::HID, r.out_all[(size_t) l].begin() + (size_t) (p + 1) * Replay::HID);
                    const uint64_t hm = Replay::hash_of(m), ha = Replay::hash_of(a), hb = Replay::hash_of(b), hc = Replay::hash_of(c);
                    CHECK(hm == ha, "[%s] a second pass after reset() is not bit-identical to the first", nm);
                    CHECK(hm == hb && hm == hc, "[%s] the outputs depend on the emulator's scheduling order", nm);
                    std::printf("INFO [%s] output hash of the first %d positions: %016llx (first run), after reset %016llx, reverse %016llx, shuffle %016llx\n", nm, n, (unsigned long long) hm,
                                (unsigned long long) ha, (unsigned long long) hb, (unsigned long long) hc);
                }
            }
        }
        std::printf("INFO [%s] %.1f s\n", nm, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        ++ran;
    }
    CHECK(ran > 0, "no golden scenario was run");
}

}  // namespace

int main(int argc, char** argv) {
    bool kernels = false;
    std::string oracle_dir;
    std::vector<std::string> only;
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--kernels")) kernels = true;
        else if (!std::strcmp(argv[i], "--oracle") && i + 1 < argc) oracle_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--scenario") && i + 1 < argc) only.push_back(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--order") && i + 1 < argc) {
            if (!ds41_emu::set_order_from_string(argv[++i])) {
                std::printf("FAIL: --order wants forward | reverse | shuffle[:SEED]\n");
                return 2;
            }
        } else {
            std::printf("usage: ds41_attn_emu_test [--kernels] [--oracle DIR [--scenario NAME]...] [--order forward|reverse|shuffle[:SEED]] [--seed N]\n");
            return 2;
        }
    }
    if (!kernels && oracle_dir.empty()) kernels = true;
    std::printf("INFO emulator scheduling order: %s (seed %llu)\n", ds41_emu::order_name(), (unsigned long long) ds41_emu::g_order_seed);
    const auto t0 = std::chrono::steady_clock::now();
    {
        HostDev dev;
        if (kernels) {
            KernelTests<MiniGeom>(dev, seed).run_all();
            std::printf("INFO MiniGeom kernels done (%d checks, %d failed)\n", g_pass + g_fail, g_fail);
            KernelTests<RealGeom>(dev, seed + 100).run_all();
            std::printf("INFO RealGeom kernels done (%d checks, %d failed)\n", g_pass + g_fail, g_fail);
        }
        if (!oracle_dir.empty()) run_oracle(dev, oracle_dir, only, true);
    }
    std::printf("INFO emulation wall time %.1f s, %d checks passed, %d failed\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), g_pass, g_fail);
    if (g_fail) {
        std::printf("FAIL: %d checks failed\n", g_fail);
        return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
