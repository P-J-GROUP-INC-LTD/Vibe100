// src/ds41/cuda/dense_test.cpp - DS1-B: the dense kernels (src/ds41/cuda/dense_impl.cuh) against C++ references.  Two programs are built from this one source:
//   ds41_dense_emu_test   the kernels compiled for the HOST with -DDS41_EMU (the thread-model emulation, ds41_emu.hpp), at both geometries, on the CPU.  No GPU
//                         needed; registered with ctest.
//   ds41_dense_parity     (-DDS41_DENSE_GPU, STRATA_ENABLE_CUDA) the sm_70 kernels on the V100 through CudaDev, RealGeom only, plus --bench (GB/s per shape).  Run
//                         by hand on the box; not registered with ctest.  The comparisons that are bit-exact in the emulator stay bit-exact on the card except the ones
//                         that go through expf (SwiGLU's sigmoid), which compare within 4e-6 there (the device libm and the host's may differ in the last bit).
//
//   ds41_dense_emu_test [--quant] [--gemv] [--wide] [--norm] [--rope] [--shared] [--vocab] [--args] [--all] [--geom real|mini|both] [--seed N]
//                       [--order forward|reverse|shuffle[:SEED]] [--big]
//   ds41_dense_parity   [the same suites, --geom is ignored: real] [--big] [--bench]
//
// Every suite prints PASS / FAIL lines and "ALL PASS (n checks)".  What each one pins:
//   --quant   the natural-order activation quantiser the GEMVs consume (DS-D / DS1-G's ds41_quantize_acts<G>, ActOrder::kNatural): byte for byte against an
//             independent scalar implementation of CONTRACTS.md (zeros, tiny, huge, Inf, NaN of both signs, exact ties) and against the CPU library's
//             quantize_act (DS-C) on random data;
//   --gemv    Q8_0 GEMV, int8 activations (dp4a) and FP32 activations, plain / paired / grouped (wo_a): every output equal BIT FOR BIT to a C++ model of the
//             documented reduction order (dense.hpp: lanes, super-blocks, butterfly) and within a tolerance of an FP64 reference; T = 1..9 rows identical to the
//             T = 1 calls; the 2-byte-aligned generic path identical to the fast path; the three scheduling orders identical;
//   --wide    BF16 / F32 GEMV, the same checks (the head shapes, K = 5120 and 256);
//   --norm    RMSNorm against the oracle's formula (FP64) and a float model, in place / out of place / weightless, NaN rows, the tiny-variance eps case;
//   --rope    the host table (long double + the numpy oracle's golden values) and the kernel (forward, inverse, in place, nope channels copied), bit-exact
//             against the separately rounded float formula;
//   --shared  the shared expert in both modes against FP64 (stage by stage), the clamps with NaN / Inf, NaN propagation end to end;
//   --vocab   argmax and top-k (ties to the lower index, NaN above everything, -0 == +0), the embedding rows, the elementwise helpers;
//   --args    what the entry points refuse (shapes, alignment, k limits, scratch) and accept at the limits.
// --big adds the largest real shapes (wq_b 32768 x 1280, wo_a, the head's 5120-wide rows at more rows): minutes in the emulator.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "strata/ds41/cpu/mxfp4_expert.hpp"
#include "strata/ds41/cuda/dense.hpp"
#include "strata/ds41/cuda/ds41_cuda.hpp"
#include "strata/ds41/cuda/ds41_dev.hpp"
#if defined(DS41_DENSE_GPU)
#include "strata/ds41/cuda/ds41_cuda_runtime.hpp"
#else
#include "ds41_emu.hpp"
#endif

namespace {

using namespace strata::ds41;
using namespace strata::ds41::cuda;

// ---- tiny harness ---------------------------------------------------------------------------------------------------------------------------------
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
[[maybe_unused]] std::string fmt_s(const char* f, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint32_t u32() { return (uint32_t) (next() >> 32); }
    int below(int n) { return (int) (next() % (uint64_t) n); }
    double uni() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }
    double gauss() {
        const double u1 = uni() + 1e-300, u2 = uni();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

float bits_f(uint32_t b) {
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
uint32_t f_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}
bool same_f(float a, float b) { return (a != a && b != b) || f_bits(a) == f_bits(b); }       // equal bits, any NaN == any NaN
size_t count_diff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += !same_f(a[i], b[i]);
    return n;
}

// the scheduling order the kernels run in (the emulator): a result must not depend on it
#if defined(DS41_DENSE_GPU)
constexpr bool kEmu = false;                           // the card: no scheduling order to choose (the sweeps below repeat the run), expf is the device's
inline bool set_sched(const std::string&) { return true; }
#else
constexpr bool kEmu = true;
inline bool set_sched(const std::string& o) { return ds41_emu::set_order_from_string(o); }
#endif
const char* const kOrders[3] = {"forward", "reverse", "shuffle:5"};
std::string g_order = "forward";                       // the order the command line asked for (restored after the internal order sweeps)
void set_cmdline_order() { set_sched(g_order); }

// ---- half precision (independent of the kernels' conversion) ------------------------------------------------------------------------------------
double half_to_double(uint16_t h) {
    const int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) v = std::ldexp((double) m, -24);
    else if (e == 31) v = m ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    else v = std::ldexp((double) (1024 + m), e - 25);
    return s ? -v : v;
}
float half_to_float(uint16_t h) { return (float) half_to_double(h); }          // exact: every half is a float
uint16_t random_half(Rng& r, int style) {
    // style 0: ordinary scales; 1: also zero / subnormal / the largest; 2: small scales (2^-7 .. 2^-4: pre-activations of a few units, so the SwiGLU clamps bite only now and then)
    if (style == 2) return (uint16_t) (((r.u32() & 1) << 15) | ((8 + r.below(4)) << 10) | r.below(1024));
    const int pick = r.below(100);
    if (style == 1 && pick < 3) return (uint16_t) ((r.u32() & 1) << 15);                                   // +-0
    if (style == 1 && pick < 6) return (uint16_t) (((r.u32() & 1) << 15) | (1 + r.below(1023)));           // subnormal
    if (style == 1 && pick < 8) return (uint16_t) (((r.u32() & 1) << 15) | 0x7BFF);                        // +-65504
    const int e = 6 + r.below(12);                                                                          // 2^-9 .. 2^2
    return (uint16_t) (((r.u32() & 1) << 15) | (e << 10) | r.below(1024));
}

// ---- Q8_0 matrices ----------------------------------------------------------------------------------------------------------------------------------
struct Q8Mat {
    int rows = 0, k = 0;
    std::vector<uint8_t> bytes;                          // rows * k / 32 * 34
    const uint8_t* row(int r) const { return bytes.data() + (size_t) r * (k / 32) * 34; }
};
Q8Mat make_q8(Rng& rng, int rows, int k, int style) {
    Q8Mat m;
    m.rows = rows;
    m.k = k;
    m.bytes.resize((size_t) rows * (k / 32) * 34);
    for (size_t b = 0; b < m.bytes.size() / 34; ++b) {
        uint8_t* p = m.bytes.data() + b * 34;
        const uint16_t d = random_half(rng, style);
        p[0] = (uint8_t) (d & 255);
        p[1] = (uint8_t) (d >> 8);
        const int kind = rng.below(10);
        for (int j = 0; j < 32; ++j) {
            int q;
            if (kind == 0) q = (rng.below(2) ? 127 : -128);                         // extremes (-128 is legal in a stored block)
            else if (kind == 1) q = 0;
            else q = (int) std::lround(rng.gauss() * 40.0);
            q = std::max(-128, std::min(127, q));
            p[2 + j] = (uint8_t) (int8_t) q;
        }
    }
    return m;
}

// ---- the activation quantiser, independently (CONTRACTS.md "Activations") -----------------------------------------------------------------------------
void ref_quant(const float* x, int n, int8_t* q, float* d) {
    for (int b = 0; b < n / 32; ++b) {
        uint32_t m = 0;
        for (int j = 0; j < 32; ++j) m = std::max(m, f_bits(x[b * 32 + j]) & 0x7FFFFFFFu);
        if (m >= 0x7F800000u) {
            d[b] = bits_f(0x7FC00000u);
            for (int j = 0; j < 32; ++j) q[b * 32 + j] = 0;
        } else if (m < 0x0D800000u) {
            d[b] = 0.0f;
            for (int j = 0; j < 32; ++j) q[b * 32 + j] = 0;
        } else {
            const float amax = bits_f(m);
            volatile float dd = amax / 127.0f;
            volatile float id = 127.0f / amax;
            d[b] = dd;
            for (int j = 0; j < 32; ++j) {
                volatile float p = x[b * 32 + j] * id;
                float r = std::nearbyintf(p);                                          // ties to even (the default rounding mode)
                r = std::max(-127.0f, std::min(127.0f, r));
                q[b * 32 + j] = (int8_t) r;
            }
        }
    }
}

// ---- device helpers -----------------------------------------------------------------------------------------------------------------------------------
/// The activation quantiser the int8 GEMVs consume: DS-D / DS1-G's, natural per-32 layout.
template <class G>
void quantize_nat(Dev& dev, const float* x, int T, int width, int8_t* xq, float* xs) {
    ds41_quantize_acts<G>(dev, x, T, width, xq, xs, nullptr, ActOrder::kNatural);
}
template <class T>
void upload(Dev& dev, DevBuf<T>& b, const std::vector<T>& v) {
    dev.h2d(b.p, v.data(), v.size() * sizeof(T));
}
template <class T>
std::vector<T> download(Dev& dev, const T* p, size_t n) {
    std::vector<T> v(n);
    dev.d2h(v.data(), p, n * sizeof(T));
    return v;
}

// =====================================================================================================================================================
// --quant
// =====================================================================================================================================================
template <class G>
void run_quant(Dev& dev, Report& rep, Rng& rng) {
    const std::string g = G::kName;
    // 1. the contract's special blocks, bit patterns included
    {
        std::vector<float> x;
        auto add_block = [&](const std::vector<float>& b) {
            for (int j = 0; j < 32; ++j) x.push_back(b[(size_t) j % b.size()]);
        };
        add_block({0.0f});
        add_block({-0.0f});
        add_block({1e-32f, -3e-33f});                                                    // amax < 2^-100: d = 0, q = 0
        add_block({1e-30f, -5e-31f, 2e-30f});                                            // just above 2^-100: a real block
        add_block({bits_f(0x0D7FFFFFu), bits_f(0x0D800000u)});                           // the threshold itself: below / at
        add_block({1.0f / 0.0f, 1.0f});
        add_block({-1.0f / 0.0f, 5.0f});
        add_block({bits_f(0x7FC00000u), 2.0f});
        add_block({bits_f(0xFFC12345u), 2.0f});                                           // NaN with sign and payload
        add_block({bits_f(0x7F800001u), 2.0f});                                           // signalling NaN
        add_block({127.0f, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 3.5f});                // d = 1, id = 1: exact ties go to even
        add_block({1e30f, -1e30f, 3e29f});
        add_block({3.4e38f, -3.4e38f, 1.0f});
        add_block({1.0f, -1.0f});
        std::vector<float> rnd(32);
        for (int j = 0; j < 32; ++j) rnd[(size_t) j] = (float) rng.gauss();
        add_block(rnd);
        const int nb = (int) x.size() / 32;
        const int T = 1;
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<int8_t> dq(dev, x.size());
        DevBuf<float> ds(dev, (size_t) nb);
        quantize_nat<G>(dev, dx.p, T, nb * 32, dq.p, ds.p);
        const auto q = dq.down();
        const auto d = ds.down();
        std::vector<int8_t> rq(x.size());
        std::vector<float> rd((size_t) nb);
        ref_quant(x.data(), nb * 32, rq.data(), rd.data());
        size_t bad = 0;
        for (size_t i = 0; i < q.size(); ++i) bad += q[i] != rq[i];
        for (size_t i = 0; i < d.size(); ++i) bad += !(f_bits(d[i]) == f_bits(rd[i]));
        rep.line(bad == 0, (g + " quantiser: contract specials").c_str(), "%d blocks, %zu mismatching bytes / scales (scales compared as bits: the NaN is 0x7FC00000)", nb, bad);
    }
    // 2. random data of every width the model uses, T = 1..3, against the scalar reference and against DS-C's CPU quantiser
    std::vector<int> widths = {32, 64, 128, 256, 1280, G::kQLora, G::kHeadDim, G::kHidden, G::kFF, Derived<G>::kOGroupIn};
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    for (int width : widths) {
        if (width % 32 != 0 || width < 32) continue;
        for (int T : {1, 2, 3, 8}) {
            std::vector<float> x((size_t) T * width);
            for (auto& v : x) {
                const int kind = rng.below(40);
                if (kind == 0) v = 0.0f;
                else if (kind == 1) v = (float) (rng.gauss() * 1e6);
                else if (kind == 2) v = bits_f(rng.u32() & 0xBFFFFFFFu);                 // arbitrary patterns, NaN / Inf included (rarely a whole block's worth)
                else v = (float) (rng.gauss() * std::pow(10.0, rng.below(7) - 3));
            }
            DevBuf<float> dx(dev, x.size());
            upload(dev, dx, x);
            DevBuf<int8_t> dq(dev, x.size());
            DevBuf<float> ds(dev, x.size() / 32);
            quantize_nat<G>(dev, dx.p, T, width, dq.p, ds.p);
            const auto q = dq.down();
            const auto d = ds.down();
            std::vector<int8_t> rq(x.size());
            std::vector<float> rd(x.size() / 32);
            ref_quant(x.data(), (int) x.size(), rq.data(), rd.data());
            size_t bad = 0;
            for (size_t i = 0; i < q.size(); ++i) bad += q[i] != rq[i];
            for (size_t i = 0; i < d.size(); ++i) bad += f_bits(d[i]) != f_bits(rd[i]);
            // DS-C's quantiser (the CPU engine's), for the widths it takes (a multiple of 128, at most 5120)
            size_t bad_cpu = 0;
            if (width % 128 == 0 && width <= cpu::kActMaxBlocks * 32) {
                for (int t = 0; t < T; ++t) {
                    cpu::ActQ aq;
                    cpu::quantize_act(x.data() + (size_t) t * width, width, aq);
                    std::vector<int8_t> cq((size_t) width);
                    cpu::act_unpack(aq, width, cq.data());
                    for (int i = 0; i < width; ++i) bad_cpu += cq[(size_t) i] != q[(size_t) t * width + i];
                    for (int b = 0; b < width / 32; ++b) bad_cpu += f_bits(aq.scale[b]) != f_bits(d[(size_t) t * (width / 32) + b]);
                }
            }
            rep.line(bad == 0 && bad_cpu == 0, (g + " quantiser: random").c_str(), "width %d, T %d: %zu mismatches vs the scalar reference, %zu vs the CPU library", width, T, bad, bad_cpu);
        }
    }
}

// =====================================================================================================================================================
// GEMV models (the documented reduction order, in float, bit for bit) and FP64 references
// =====================================================================================================================================================
int model_p(int k) {
    const int nsb = (k / 32 + 7) / 8;
    int p = 1;
    while (p < nsb) p <<= 1;
    return std::min(32, p);
}
// the xor butterfly over the P lanes of a row: every lane ends with the same bits; return lane 0's
float butterfly(std::vector<float> v) {
    for (size_t off = v.size() / 2; off > 0; off >>= 1) {
        std::vector<float> n(v.size());
        for (size_t l = 0; l < v.size(); ++l) n[l] = v[l] + v[l ^ off];
        v = n;
    }
    return v[0];
}
// int8 mode: per block dw[b], isum[b] (exact), then the lane / super-block order
struct Int8Row {
    std::vector<float> dw;
    std::vector<int> isum;
};
Int8Row int8_blocks(const uint8_t* rowp, int k, const int8_t* xq) {
    Int8Row r;
    const int nb = k / 32;
    r.dw.resize((size_t) nb);
    r.isum.resize((size_t) nb);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* p = rowp + (size_t) b * 34;
        r.dw[(size_t) b] = half_to_float((uint16_t) (p[0] | (p[1] << 8)));
        int s = 0;
        for (int j = 0; j < 32; ++j) s += (int) (int8_t) p[2 + j] * (int) xq[b * 32 + j];
        r.isum[(size_t) b] = s;
    }
    return r;
}
float model_int8(const Int8Row& r, const float* xs, int k) {
    const int nb = k / 32, nsb = (nb + 7) / 8, P = model_p(k);
    std::vector<float> acc((size_t) P, 0.0f);
    for (int l = 0; l < P; ++l)
        for (int s = l; s < nsb; s += P)
            for (int c = 0; c < 8 && 8 * s + c < nb; ++c) {
                const int b = 8 * s + c;
                volatile float prod = r.dw[(size_t) b] * xs[b];
                acc[(size_t) l] = std::fmaf(prod, (float) r.isum[(size_t) b], acc[(size_t) l]);
            }
    return butterfly(acc);
}
void ref_int8(const Int8Row& r, const float* xs, int k, double& value, double& abssum) {
    value = abssum = 0.0;
    for (int b = 0; b < k / 32; ++b) {
        const double t = (double) r.dw[(size_t) b] * (double) xs[b] * (double) r.isum[(size_t) b];
        value += t;
        abssum += std::fabs(t);
    }
}
// FP32-activation mode
float model_f32(const uint8_t* rowp, const float* x, int k) {
    const int nb = k / 32, nsb = (nb + 7) / 8, P = model_p(k);
    std::vector<float> acc((size_t) P, 0.0f);
    for (int l = 0; l < P; ++l)
        for (int s = l; s < nsb; s += P)
            for (int c = 0; c < 8 && 8 * s + c < nb; ++c) {
                const int b = 8 * s + c;
                const uint8_t* p = rowp + (size_t) b * 34;
                const float dw = half_to_float((uint16_t) (p[0] | (p[1] << 8)));
                for (int j = 0; j < 32; ++j) {
                    volatile float f = dw * (float) (int8_t) p[2 + j];
                    acc[(size_t) l] = std::fmaf(x[b * 32 + j], f, acc[(size_t) l]);
                }
            }
    return butterfly(acc);
}
void ref_f32(const uint8_t* rowp, const float* x, int k, double& value, double& abssum) {
    value = abssum = 0.0;
    for (int b = 0; b < k / 32; ++b) {
        const uint8_t* p = rowp + (size_t) b * 34;
        const double dw = half_to_double((uint16_t) (p[0] | (p[1] << 8)));
        for (int j = 0; j < 32; ++j) {
            const double t = (double) x[b * 32 + j] * (dw * (double) (int8_t) p[2 + j]);
            value += t;
            abssum += std::fabs(t);
        }
    }
}
bool close(double got, double want, double abssum, double rel) {
    if (std::isnan(want)) return std::isnan(got);
    if (std::isnan(got)) return false;
    if (std::isinf(want)) return got == want;
    return std::fabs(got - want) <= rel * abssum + 1e-30;
}

std::vector<float> make_x(Rng& rng, int T, int k, int style) {
    std::vector<float> x((size_t) T * k);
    for (auto& v : x) {
        const int kind = rng.below(50);
        if (style == 1 && kind == 0) v = 0.0f;
        else if (style == 1 && kind == 1) v = (float) (rng.gauss() * 1e4);
        else v = (float) (rng.gauss() * (style == 1 ? std::pow(10.0, rng.below(5) - 2) : 1.0));
    }
    if (style == 1 && T > 0 && k >= 64)                                                       // one all-zero block per token
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < 32; ++j) x[(size_t) t * k + 32 + j] = 0.0f;
    return x;
}

// =====================================================================================================================================================
// --gemv
// =====================================================================================================================================================
struct GemvCase {
    const char* what;
    int n, k;
};

template <class G>
void check_q8_int8(Dev& dev, Report& rep, Rng& rng, const GemvCase& cs, const std::vector<int>& Ts) {
    const std::string g = G::kName;
    const Q8Mat m = make_q8(rng, cs.n, cs.k, 1);
    DevBuf<uint8_t> dw(dev, m.bytes.size());
    upload(dev, dw, m.bytes);
    const int nb = cs.k / 32;
    for (int T : Ts) {
        const std::vector<float> x = make_x(rng, T, cs.k, 1);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<int8_t> dq(dev, (size_t) T * cs.k);
        DevBuf<float> ds(dev, (size_t) T * nb);
        DevBuf<float> dy(dev, (size_t) T * cs.n);
        quantize_nat<G>(dev, dx.p, T, cs.k, dq.p, ds.p);
        ds41_gemv_q8_int8<G>(dev, dw.p, cs.n, cs.k, dq.p, ds.p, T, dy.p);
        const auto y = dy.down();
        const auto q = dq.down();
        const auto s = ds.down();
        size_t bad_model = 0, bad_ref = 0;
        double worst = 0.0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < cs.n; ++r) {
                const Int8Row br = int8_blocks(m.row(r), cs.k, q.data() + (size_t) t * cs.k);
                const float want = model_int8(br, s.data() + (size_t) t * nb, cs.k);
                const float got = y[(size_t) t * cs.n + r];
                bad_model += !same_f(got, want);
                double v, a;
                ref_int8(br, s.data() + (size_t) t * nb, cs.k, v, a);
                if (!close(got, v, a, 4e-6)) ++bad_ref;
                if (a > 0 && !std::isnan(v)) worst = std::max(worst, std::fabs(got - v) / a);
            }
        rep.line(bad_model == 0 && bad_ref == 0, (g + " q8 int8 " + cs.what).c_str(), "n %d k %d T %d (P %d): %zu rows differ from the order model (bits), %zu from FP64 (worst rel %.2e of sum|t|)", cs.n, cs.k,
                 T, model_p(cs.k), bad_model, bad_ref, worst);
        // T-invariance: every row of this T call equals the T = 1 call of that token
        if (T > 1) {
            size_t bad = 0;
            for (int t = 0; t < T; ++t) {
                DevBuf<float> dy1(dev, (size_t) cs.n);
                ds41_gemv_q8_int8<G>(dev, dw.p, cs.n, cs.k, dq.p + (size_t) t * cs.k, ds.p + (size_t) t * nb, 1, dy1.p);
                const auto y1 = dy1.down();
                for (int r = 0; r < cs.n; ++r) bad += !same_f(y1[(size_t) r], y[(size_t) t * cs.n + r]);
            }
            rep.line(bad == 0, (g + " q8 int8 T-invariance " + cs.what).c_str(), "n %d k %d T %d: %zu rows differ from the T = 1 calls", cs.n, cs.k, T, bad);
        }
    }
}

template <class G>
void check_q8_int8_extra(Dev& dev, Report& rep, Rng& rng, const GemvCase& cs) {
    const std::string g = G::kName;
    const int T = 3, nb = cs.k / 32;
    const Q8Mat ma = make_q8(rng, cs.n, cs.k, 1), mb = make_q8(rng, cs.n, cs.k, 0);
    DevBuf<uint8_t> wa(dev, ma.bytes.size()), wb(dev, mb.bytes.size());
    upload(dev, wa, ma.bytes);
    upload(dev, wb, mb.bytes);
    const std::vector<float> x = make_x(rng, T, cs.k, 1);
    DevBuf<float> dx(dev, x.size());
    upload(dev, dx, x);
    DevBuf<int8_t> dq(dev, (size_t) T * cs.k);
    DevBuf<float> ds(dev, (size_t) T * nb);
    quantize_nat<G>(dev, dx.p, T, cs.k, dq.p, ds.p);
    DevBuf<float> y1(dev, (size_t) T * cs.n), y2(dev, (size_t) T * cs.n), ya(dev, (size_t) T * cs.n), yb(dev, (size_t) T * cs.n);
    ds41_gemv_q8_int8<G>(dev, wa.p, cs.n, cs.k, dq.p, ds.p, T, y1.p);
    ds41_gemv_q8_int8<G>(dev, wb.p, cs.n, cs.k, dq.p, ds.p, T, y2.p);
    ds41_gemv_q8_int8_pair<G>(dev, wa.p, wb.p, cs.n, cs.k, dq.p, ds.p, T, ya.p, yb.p);
    rep.line(count_diff(y1.down(), ya.down()) == 0 && count_diff(y2.down(), yb.down()) == 0, (g + " q8 int8 pair == two calls").c_str(), "n %d k %d T %d", cs.n, cs.k, T);

    // the generic path (weights at an odd 2-byte offset) is bit-identical to the fast path
    if (cs.k % 256 == 0) {
        DevBuf<uint8_t> wo(dev, ma.bytes.size() + 16);
        std::vector<uint8_t> shifted(ma.bytes.size() + 16, 0);
        std::memcpy(shifted.data() + 2, ma.bytes.data(), ma.bytes.size());
        upload(dev, wo, shifted);
        DevBuf<float> yg(dev, (size_t) T * cs.n);
        ds41_gemv_q8_int8<G>(dev, wo.p + 2, cs.n, cs.k, dq.p, ds.p, T, yg.p);
        rep.line(count_diff(y1.down(), yg.down()) == 0, (g + " q8 int8 generic path == fast path").c_str(), "n %d k %d (weights 2-byte aligned only)", cs.n, cs.k);
    }
    // the three scheduling orders: identical bits
    {
        const auto base = y1.down();
        size_t bad = 0;
        for (const char* o : kOrders) {
            set_sched(o);
            DevBuf<float> yo(dev, (size_t) T * cs.n);
            ds41_gemv_q8_int8<G>(dev, wa.p, cs.n, cs.k, dq.p, ds.p, T, yo.p);
            bad += count_diff(base, yo.down());
        }
        set_cmdline_order();
        rep.line(bad == 0, (g + " q8 int8 scheduling orders").c_str(), "n %d k %d: %zu differing outputs over forward / reverse / shuffle", cs.n, cs.k, bad);
    }
    // NaN: one NaN activation poisons every output of its token and no other token
    {
        std::vector<float> xn = x;
        xn[(size_t) 1 * cs.k + 5] = bits_f(0x7FC00000u);
        DevBuf<float> dxn(dev, xn.size());
        upload(dev, dxn, xn);
        DevBuf<int8_t> qn(dev, (size_t) T * cs.k);
        DevBuf<float> sn(dev, (size_t) T * nb);
        quantize_nat<G>(dev, dxn.p, T, cs.k, qn.p, sn.p);
        DevBuf<float> yn(dev, (size_t) T * cs.n);
        ds41_gemv_q8_int8<G>(dev, wa.p, cs.n, cs.k, qn.p, sn.p, T, yn.p);
        const auto y = yn.down(), yref = y1.down();
        size_t nan_t1 = 0, bad_other = 0;
        for (int r = 0; r < cs.n; ++r) {
            nan_t1 += std::isnan(y[(size_t) cs.n + r]);
            bad_other += !same_f(y[(size_t) r], yref[(size_t) r]) + !same_f(y[(size_t) 2 * cs.n + r], yref[(size_t) 2 * cs.n + r]);
        }
        rep.line(nan_t1 == (size_t) cs.n && bad_other == 0, (g + " q8 int8 NaN propagation").c_str(), "n %d k %d: %zu of %d outputs of the NaN token are NaN, %zu other outputs changed", cs.n, cs.k, nan_t1, cs.n, bad_other);
    }
}

template <class G>
void check_q8_f32(Dev& dev, Report& rep, Rng& rng, const GemvCase& cs, const std::vector<int>& Ts) {
    const std::string g = G::kName;
    const Q8Mat m = make_q8(rng, cs.n, cs.k, 1);
    DevBuf<uint8_t> dw(dev, m.bytes.size());
    upload(dev, dw, m.bytes);
    std::vector<std::vector<float>> y1s;                                 // the T = 1 results per token (T-invariance)
    for (int T : Ts) {
        const std::vector<float> x = make_x(rng, T, cs.k, 1);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<float> dy(dev, (size_t) T * cs.n);
        ds41_gemv_q8_f32<G>(dev, dw.p, cs.n, cs.k, dx.p, T, dy.p);
        const auto y = dy.down();
        size_t bad_model = 0, bad_ref = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < cs.n; ++r) {
                const float want = model_f32(m.row(r), x.data() + (size_t) t * cs.k, cs.k);
                const float got = y[(size_t) t * cs.n + r];
                bad_model += !same_f(got, want);
                double v, a;
                ref_f32(m.row(r), x.data() + (size_t) t * cs.k, cs.k, v, a);
                if (!close(got, v, a, 2e-5)) ++bad_ref;
            }
        rep.line(bad_model == 0 && bad_ref == 0, (g + " q8 f32 " + cs.what).c_str(), "n %d k %d T %d (P %d): %zu rows differ from the order model (bits), %zu from FP64", cs.n, cs.k, T, model_p(cs.k),
                 bad_model, bad_ref);
        if (T > 1) {
            size_t bad = 0;
            for (int t = 0; t < T; ++t) {
                DevBuf<float> dy1(dev, (size_t) cs.n);
                ds41_gemv_q8_f32<G>(dev, dw.p, cs.n, cs.k, dx.p + (size_t) t * cs.k, 1, dy1.p);
                const auto y1 = dy1.down();
                for (int r = 0; r < cs.n; ++r) bad += !same_f(y1[(size_t) r], y[(size_t) t * cs.n + r]);
            }
            rep.line(bad == 0, (g + " q8 f32 T-invariance " + cs.what).c_str(), "n %d k %d T %d: %zu rows differ from the T = 1 calls", cs.n, cs.k, T, bad);
        }
    }
}

template <class G>
void check_q8_f32_extra(Dev& dev, Report& rep, Rng& rng, const GemvCase& cs) {
    const std::string g = G::kName;
    const int T = 2;
    const Q8Mat m = make_q8(rng, cs.n, cs.k, 1);
    DevBuf<uint8_t> dw(dev, m.bytes.size());
    upload(dev, dw, m.bytes);
    const std::vector<float> x = make_x(rng, T, cs.k, 1);
    DevBuf<float> dx(dev, x.size());
    upload(dev, dx, x);
    DevBuf<float> y1(dev, (size_t) T * cs.n);
    ds41_gemv_q8_f32<G>(dev, dw.p, cs.n, cs.k, dx.p, T, y1.p);
    if (cs.k % 256 == 0) {
        std::vector<uint8_t> shifted(m.bytes.size() + 16, 0);
        std::memcpy(shifted.data() + 2, m.bytes.data(), m.bytes.size());
        DevBuf<uint8_t> wo(dev, shifted.size());
        upload(dev, wo, shifted);
        DevBuf<float> yg(dev, (size_t) T * cs.n);
        ds41_gemv_q8_f32<G>(dev, wo.p + 2, cs.n, cs.k, dx.p, T, yg.p);
        rep.line(count_diff(y1.down(), yg.down()) == 0, (g + " q8 f32 generic path == fast path").c_str(), "n %d k %d", cs.n, cs.k);
    }
    const auto base = y1.down();
    size_t bad = 0;
    for (const char* o : kOrders) {
        set_sched(o);
        DevBuf<float> yo(dev, (size_t) T * cs.n);
        ds41_gemv_q8_f32<G>(dev, dw.p, cs.n, cs.k, dx.p, T, yo.p);
        bad += count_diff(base, yo.down());
    }
    set_cmdline_order();
    rep.line(bad == 0, (g + " q8 f32 scheduling orders").c_str(), "n %d k %d: %zu differing outputs", cs.n, cs.k, bad);
}

template <class G>
void check_grouped(Dev& dev, Report& rep, Rng& rng, int groups, int R, int K, const std::vector<int>& Ts) {
    const std::string g = G::kName;
    const Q8Mat m = make_q8(rng, groups * R, K, 1);
    DevBuf<uint8_t> dw(dev, m.bytes.size());
    upload(dev, dw, m.bytes);
    for (int T : Ts) {
        const std::vector<float> x = make_x(rng, T, groups * K, 1);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<float> dy(dev, (size_t) T * groups * R);
        ds41_gemv_q8_grouped_f32<G>(dev, dw.p, groups, R, K, dx.p, T, dy.p);
        const auto y = dy.down();
        size_t bad_model = 0, bad_ref = 0;
        for (int t = 0; t < T; ++t)
            for (int gr = 0; gr < groups; ++gr)
                for (int r = 0; r < R; ++r) {
                    const float* xs = x.data() + (size_t) t * groups * K + (size_t) gr * K;
                    const uint8_t* rowp = m.row(gr * R + r);
                    const float want = model_f32(rowp, xs, K);
                    const float got = y[(size_t) t * groups * R + (size_t) gr * R + r];
                    bad_model += !same_f(got, want);
                    double v, a;
                    ref_f32(rowp, xs, K, v, a);
                    if (!close(got, v, a, 2e-5)) ++bad_ref;
                }
        rep.line(bad_model == 0 && bad_ref == 0, (g + " grouped q8 f32 (wo_a)").c_str(), "groups %d, %d rows of %d, T %d: %zu rows differ from the order model, %zu from FP64", groups, R, K, T, bad_model,
                 bad_ref);
        if (T > 1) {
            size_t bad = 0;
            for (int t = 0; t < T; ++t) {
                DevBuf<float> dy1(dev, (size_t) groups * R);
                ds41_gemv_q8_grouped_f32<G>(dev, dw.p, groups, R, K, dx.p + (size_t) t * groups * K, 1, dy1.p);
                const auto y1 = dy1.down();
                for (int i = 0; i < groups * R; ++i) bad += !same_f(y1[(size_t) i], y[(size_t) t * groups * R + i]);
            }
            rep.line(bad == 0, (g + " grouped q8 f32 T-invariance").c_str(), "T %d: %zu rows differ from the T = 1 calls", T, bad);
        }
    }
}

template <class G>
void run_gemv(Dev& dev, Report& rep, Rng& rng, bool big) {
    const std::string g = G::kName;
    const bool real = std::strcmp(G::kName, "real") == 0;
    using D = Derived<G>;
    // every Q8_0 shape of the model (RESEARCH.md section 10), n reduced for the real geometry where a full pass is too slow for an emulator (n does not change the
    // per-row structure; odd n exercises the tail of the last row block)
    std::vector<GemvCase> cases = {
        {"wq_a", G::kQLora, G::kHidden},
        {"wkv", G::kHeadDim, G::kHidden},
        {"wo_b", G::kHidden, D::kOMid},
        {"shexp gate/up", G::kFF, G::kHidden},
        {"shexp down", G::kHidden, G::kFF},
        {"indexer wq_b", D::kIdxQ, G::kQLora},
        {"engram wkv", D::kEngramOut, D::kEngramIn},
        {"wq_b", D::kQ, G::kQLora},
    };
    if (real) {
        for (auto& c : cases) {
            if (!big) c.n = std::min(c.n, std::strcmp(c.what, "wq_b") == 0 ? 2048 : (std::strcmp(c.what, "engram wkv") == 0 ? 512 : 1280));
        }
    }
    const std::vector<int> Ts_all = {1, 2, 3, 5, 8, 9};
    for (const auto& c : cases) {
        check_q8_int8<G>(dev, rep, rng, c, c.n * c.k > 3000000 ? std::vector<int>{1, 3, 8} : Ts_all);
        check_q8_f32<G>(dev, rep, rng, c, c.n * c.k > 3000000 ? std::vector<int>{1, 3} : std::vector<int>{1, 2, 5, 8});
    }
    // odd shapes: partial super-blocks, odd block counts, the generic path
    for (const GemvCase& c : std::vector<GemvCase>{{"k=32", 37, 32}, {"k=96 (3 blocks)", 33, 96}, {"k=224 (7 blocks)", 35, 224}, {"k=288 (9 blocks)", 41, 288}, {"k=2304 (72 blocks)", 29, 2304},
                                                   {"k=384", 50, 384}, {"k=512", 17, 512}, {"k=4096", 12, 4096}}) {
        check_q8_int8<G>(dev, rep, rng, c, {1, 4, 7});
        check_q8_f32<G>(dev, rep, rng, c, {1, 3});
    }
    check_q8_int8_extra<G>(dev, rep, rng, {"k=256", 70, 256});
    check_q8_int8_extra<G>(dev, rep, rng, {"k=1280", 45, 1280});
    check_q8_int8_extra<G>(dev, rep, rng, {"k=384", 31, 384});
    check_q8_f32_extra<G>(dev, rep, rng, {"k=256", 70, 256});
    check_q8_f32_extra<G>(dev, rep, rng, {"k=96", 31, 96});
    // wo_a: the grouped GEMV at the geometry's own shape (groups x rows x k per group)
    {
        const int groups = G::kOGroups, R = G::kOLora, K = D::kOGroupIn;
        check_grouped<G>(dev, rep, rng, groups, R, K, real && !big ? std::vector<int>{1} : std::vector<int>{1, 2, 5});
        check_grouped<G>(dev, rep, rng, 3, 20, 96, {1, 4});                     // a ragged grouped shape
        // ds41_wo_a is the same call with the geometry's shape: one check that it does not mis-wire the arguments (mini only: the full real shape was just run)
        if (!real) {
            const Q8Mat m = make_q8(rng, groups * R, K, 0);
            DevBuf<uint8_t> dw(dev, m.bytes.size());
            upload(dev, dw, m.bytes);
            const auto x = make_x(rng, 2, groups * K, 0);
            DevBuf<float> dx(dev, x.size());
            upload(dev, dx, x);
            DevBuf<float> a(dev, (size_t) 2 * groups * R), b(dev, (size_t) 2 * groups * R);
            ds41_wo_a<G>(dev, dw.p, dx.p, 2, a.p);
            ds41_gemv_q8_grouped_f32<G>(dev, dw.p, groups, R, K, dx.p, 2, b.p);
            rep.line(count_diff(a.down(), b.down()) == 0, (g + " ds41_wo_a == the grouped call").c_str(), "%d groups x %d rows x %d", groups, R, K);
        }
    }
}

// =====================================================================================================================================================
// --wide (BF16 / F32)
// =====================================================================================================================================================
float model_wide(const float* w_row_f /* the weights as floats */, const float* x, int k, int elems_per_chunk) {
    std::vector<float> acc(32, 0.0f);
    const int nchunks = k / elems_per_chunk;
    for (int l = 0; l < 32; ++l)
        for (int c = l; c < nchunks; c += 32)
            for (int u = 0; u < elems_per_chunk; ++u) {
                const int e = c * elems_per_chunk + u;
                acc[(size_t) l] = std::fmaf(x[e], w_row_f[e], acc[(size_t) l]);
            }
    return butterfly(acc);
}
template <class G, bool BF16>
void check_wide(Dev& dev, Report& rep, Rng& rng, const char* what, int n, int k, const std::vector<int>& Ts) {
    const std::string g = G::kName;
    std::vector<float> wf((size_t) n * k);
    std::vector<uint16_t> wb;
    if (BF16) wb.resize(wf.size());
    for (size_t i = 0; i < wf.size(); ++i) {
        const int kind = rng.below(100);
        float v = kind < 3 ? 0.0f : (float) (rng.gauss() * (kind < 5 ? 100.0 : 0.05));
        if (BF16) {
            const uint32_t b = f_bits(v) & 0xFFFF0000u;                  // truncate to a BF16 value (exactly representable)
            wb[i] = (uint16_t) (b >> 16);
            v = bits_f(b);
        }
        wf[i] = v;
    }
    DevBuf<uint16_t> dwb(dev, BF16 ? wb.size() : 1);
    DevBuf<float> dwf(dev, BF16 ? 1 : wf.size());
    if (BF16) upload(dev, dwb, wb);
    else upload(dev, dwf, wf);
    for (int T : Ts) {
        const std::vector<float> x = make_x(rng, T, k, 1);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<float> dy(dev, (size_t) T * n);
        if (BF16) ds41_gemv_bf16<G>(dev, dwb.p, n, k, dx.p, T, dy.p);
        else ds41_gemv_f32<G>(dev, dwf.p, n, k, dx.p, T, dy.p);
        const auto y = dy.down();
        size_t bad_model = 0, bad_ref = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < n; ++r) {
                const float want = model_wide(wf.data() + (size_t) r * k, x.data() + (size_t) t * k, k, BF16 ? 8 : 4);
                const float got = y[(size_t) t * n + r];
                bad_model += !same_f(got, want);
                double v = 0, a = 0;
                for (int i = 0; i < k; ++i) {
                    const double p = (double) x[(size_t) t * k + i] * (double) wf[(size_t) r * k + i];
                    v += p;
                    a += std::fabs(p);
                }
                if (!close(got, v, a, 2e-5)) ++bad_ref;
            }
        rep.line(bad_model == 0 && bad_ref == 0, (g + (BF16 ? " bf16 " : " f32 ") + what).c_str(), "n %d k %d T %d: %zu rows differ from the order model (bits), %zu from FP64", n, k, T, bad_model, bad_ref);
        if (T > 1) {
            size_t bad = 0;
            for (int t = 0; t < T; ++t) {
                DevBuf<float> dy1(dev, (size_t) n);
                if (BF16) ds41_gemv_bf16<G>(dev, dwb.p, n, k, dx.p + (size_t) t * k, 1, dy1.p);
                else ds41_gemv_f32<G>(dev, dwf.p, n, k, dx.p + (size_t) t * k, 1, dy1.p);
                const auto y1 = dy1.down();
                for (int r = 0; r < n; ++r) bad += !same_f(y1[(size_t) r], y[(size_t) t * n + r]);
            }
            rep.line(bad == 0, (g + (BF16 ? " bf16 T-invariance " : " f32 T-invariance ") + what).c_str(), "n %d k %d T %d: %zu rows differ from the T = 1 calls", n, k, T, bad);
        }
    }
    // scheduling orders + NaN
    {
        const int T = 3;
        const std::vector<float> x = make_x(rng, T, k, 1);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        std::vector<float> base;
        size_t bad = 0;
        for (const char* o : kOrders) {
            set_sched(o);
            DevBuf<float> dy(dev, (size_t) T * n);
            if (BF16) ds41_gemv_bf16<G>(dev, dwb.p, n, k, dx.p, T, dy.p);
            else ds41_gemv_f32<G>(dev, dwf.p, n, k, dx.p, T, dy.p);
            const auto y = dy.down();
            if (base.empty()) base = y;
            else bad += count_diff(base, y);
        }
        set_cmdline_order();
        std::vector<float> xn = x;
        xn[(size_t) k + 3] = bits_f(0x7FC00000u);
        DevBuf<float> dxn(dev, xn.size());
        upload(dev, dxn, xn);
        DevBuf<float> dy(dev, (size_t) T * n);
        if (BF16) ds41_gemv_bf16<G>(dev, dwb.p, n, k, dxn.p, T, dy.p);
        else ds41_gemv_f32<G>(dev, dwf.p, n, k, dxn.p, T, dy.p);
        const auto y = dy.down();
        size_t nan1 = 0, bad_other = 0;
        for (int r = 0; r < n; ++r) {
            nan1 += std::isnan(y[(size_t) n + r]);
            bad_other += !same_f(y[(size_t) r], base[(size_t) r]) + !same_f(y[(size_t) 2 * n + r], base[(size_t) 2 * n + r]);
        }
        rep.line(bad == 0 && nan1 == (size_t) n && bad_other == 0, (g + (BF16 ? " bf16 orders / NaN " : " f32 orders / NaN ") + what).c_str(),
                 "n %d k %d: %zu bit differences over the scheduling orders; NaN token %zu / %d NaN outputs, %zu other outputs changed", n, k, bad, nan1, n, bad_other);
    }
}

template <class G>
void run_wide(Dev& dev, Report& rep, Rng& rng, bool big) {
    const bool real = std::strcmp(G::kName, "real") == 0;
    // the head (kVocab x kHidden): n reduced for the real geometry (a full pass is 662 M weights), the router-like / compressor / indexer projections
    const int head_n = real ? (big ? 6000 : 700) : G::kVocab;
    check_wide<G, true>(dev, rep, rng, "head rows", head_n, G::kHidden, real ? std::vector<int>{1, 3, 8} : std::vector<int>{1, 2, 3, 5, 8, 9});
    check_wide<G, true>(dev, rep, rng, "indexer proj", G::kIdxHeads, G::kHidden, {1, 4});
    check_wide<G, true>(dev, rep, rng, "compressor", G::kHeadDim, G::kHidden, {1, 2});
    check_wide<G, true>(dev, rep, rng, "idx compressor", G::kIdxDim, G::kHeadDim, {1, 3});
    check_wide<G, true>(dev, rep, rng, "k=8", 13, 8, {1, 2});
    check_wide<G, true>(dev, rep, rng, "k=2056", 21, 2056, {1, 5});
    check_wide<G, false>(dev, rep, rng, "hc-like", 24, 4 * G::kHidden > 8192 ? 8192 : 4 * G::kHidden, {1, 3, 8});
    check_wide<G, false>(dev, rep, rng, "k=4", 9, 4, {1});
    check_wide<G, false>(dev, rep, rng, "k=1292", 17, 1292, {1, 2});
    if (real) {
        // the real head's token tiles: T = 8 at K = 5120 does not fit one tile of shared memory (the tiles run as separate blocks over the same rows)
        check_wide<G, true>(dev, rep, rng, "head rows (T = 8, 9 tiles)", 160, G::kHidden, {8, 9});
    }
}

// =====================================================================================================================================================
// --norm
// =====================================================================================================================================================
float model_rmsnorm_row(const float* x, const float* w, int width, float eps, float* out) {
    float lane[32] = {0};
    for (int l = 0; l < 32; ++l)
        for (int c = l; c < width / 4; c += 32) {
            volatile float a = x[4 * c] * x[4 * c];
            volatile float b = x[4 * c + 1] * x[4 * c + 1];
            volatile float cc = x[4 * c + 2] * x[4 * c + 2];
            volatile float d = x[4 * c + 3] * x[4 * c + 3];
            volatile float s4 = a;
            s4 = s4 + b;
            s4 = s4 + cc;
            s4 = s4 + d;
            lane[l] = lane[l] + s4;
        }
    std::vector<float> v(lane, lane + 32);
    const float ss = butterfly(v);
    volatile float ms = ss / (float) width;
    volatile float t = ms + eps;
    const float r = std::sqrt((float) t);
    for (int i = 0; i < width; ++i) {
        volatile float q = x[i] / r;
        out[i] = w ? w[i] * q : (float) q;
    }
    return r;
}
template <class G>
void check_norm(Dev& dev, Report& rep, Rng& rng, const char* what, int rows, int width) {
    const std::string g = G::kName;
    std::vector<float> x((size_t) rows * width), w((size_t) width);
    for (auto& v : x) v = (float) (rng.gauss() * std::pow(10.0, rng.below(5) - 2));
    for (auto& v : w) v = (float) (0.5 + rng.uni());
    if (rows > 1) {                                                                  // a zero row, a tiny row (eps matters), a huge row
        for (int i = 0; i < width; ++i) {
            x[(size_t) i] = 0.0f;
            x[(size_t) width + i] = 1e-12f * (float) (1 + (i % 3));
            if (rows > 2) x[(size_t) 2 * width + i] = 1e9f * (float) (1 + (i % 5));
        }
    }
    DevBuf<float> dx(dev, x.size()), dw(dev, w.size()), dy(dev, x.size());
    upload(dev, dx, x);
    upload(dev, dw, w);
    ds41_rmsnorm<G>(dev, dx.p, dw.p, rows, width, dy.p);
    const auto y = dy.down();
    size_t bad_model = 0, bad_ref = 0;
    std::vector<float> m((size_t) width);
    for (int r = 0; r < rows; ++r) {
        model_rmsnorm_row(x.data() + (size_t) r * width, w.data(), width, 1e-20f, m.data());
        double ss = 0;
        for (int i = 0; i < width; ++i) ss += (double) x[(size_t) r * width + i] * x[(size_t) r * width + i];
        const double inv = 1.0 / std::sqrt(ss / width + 1e-20);                       // the oracle: weight * (x / sqrt(var + eps))
        for (int i = 0; i < width; ++i) {
            bad_model += !same_f(y[(size_t) r * width + i], m[(size_t) i]);
            const double want = (double) w[(size_t) i] * (double) x[(size_t) r * width + i] * inv;
            if (std::fabs(y[(size_t) r * width + i] - want) > 3e-6 * std::fabs(want) + 1e-30) ++bad_ref;
        }
    }
    rep.line(bad_model == 0 && bad_ref == 0, (g + " rmsnorm " + what).c_str(), "rows %d width %d: %zu elements differ from the float model (bits), %zu from FP64 (3e-6)", rows, width, bad_model, bad_ref);
    // in place and weightless
    {
        DevBuf<float> dip(dev, x.size());
        upload(dev, dip, x);
        ds41_rmsnorm<G>(dev, dip.p, dw.p, rows, width, dip.p);
        rep.line(count_diff(dip.down(), y) == 0, (g + " rmsnorm in place == out of place").c_str(), "rows %d width %d", rows, width);
        DevBuf<float> dnw(dev, x.size());
        ds41_rmsnorm<G>(dev, dx.p, nullptr, rows, width, dnw.p);
        const auto yn = dnw.down();
        size_t bad = 0;
        for (int r = 0; r < rows; ++r) {
            model_rmsnorm_row(x.data() + (size_t) r * width, nullptr, width, 1e-20f, m.data());
            for (int i = 0; i < width; ++i) bad += !same_f(yn[(size_t) r * width + i], m[(size_t) i]);
        }
        rep.line(bad == 0, (g + " rmsnorm weightless").c_str(), "rows %d width %d: %zu elements differ", rows, width, bad);
    }
    // a NaN row is NaN everywhere, its neighbours are untouched
    if (rows > 2) {
        std::vector<float> xn = x;
        xn[(size_t) width + 1] = bits_f(0x7FC00000u);
        DevBuf<float> dxn(dev, xn.size()), dyn(dev, xn.size());
        upload(dev, dxn, xn);
        ds41_rmsnorm<G>(dev, dxn.p, dw.p, rows, width, dyn.p);
        const auto yn = dyn.down();
        size_t nan1 = 0, bad = 0;
        for (int i = 0; i < width; ++i) {
            nan1 += std::isnan(yn[(size_t) width + i]);
            bad += !same_f(yn[(size_t) i], y[(size_t) i]) + !same_f(yn[(size_t) 2 * width + i], y[(size_t) 2 * width + i]);
        }
        rep.line(nan1 == (size_t) width && bad == 0, (g + " rmsnorm NaN row").c_str(), "width %d: %zu of %d NaN, %zu neighbour elements changed", width, nan1, width, bad);
    }
}
template <class G>
void run_norm(Dev& dev, Report& rep, Rng& rng) {
    check_norm<G>(dev, rep, rng, "hidden", 3, G::kHidden);
    check_norm<G>(dev, rep, rng, "hidden T=8", 8, G::kHidden);
    check_norm<G>(dev, rep, rng, "q lora", 3, G::kQLora);
    check_norm<G>(dev, rep, rng, "head dim x heads", 3 * 4, G::kHeadDim);
    check_norm<G>(dev, rep, rng, "idx dim x heads", 4 * G::kIdxHeads, G::kIdxDim);
    check_norm<G>(dev, rep, rng, "width 4", 5, 4);
    check_norm<G>(dev, rep, rng, "width 36", 4, 36);
}

// =====================================================================================================================================================
// --rope
// =====================================================================================================================================================
// golden values from the numpy oracle (ref/ds41/rope.py: rope_table, float64, rounded to float32): {pos, pair, cos bits, sin bits}.  Generated by
//   python3 -c "..." (see the comment block in the test's source); regenerate with src/ds41/cuda/dense_golden/gen_rope_golden.py
#include "dense_rope_golden.inc"

template <class G>
void run_rope(Dev& dev, Report& rep, Rng& rng) {
    const std::string g = G::kName;
    const int rd = G::kRopeDim;
    // 1. the host table against an independent long double evaluation (any parameters), <= 1 ulp
    {
        struct P {
            const char* name;
            RopeParams p;
        } sets[] = {{"plain 10000", {0, 10000.0, 16.0, 32.0, 1.0}}, {"yarn 160000", {65536, 160000.0, 16.0, 32.0, 1.0}}, {"yarn small", {512, 1000.0, 4.0, 8.0, 2.0}}};
        for (const auto& s : sets) {
            const int seq = 600;
            std::vector<float> c, sn;
            ds41_rope_table_host(rd, seq, s.p, c, sn);
            const int half = rd / 2;
            // long double reference
            std::vector<long double> fr((size_t) half);
            for (int i = 0; i < half; ++i) fr[(size_t) i] = 1.0L / powl((long double) s.p.base, (long double) (2 * i) / (long double) rd);
            if (s.p.original_seq_len > 0) {
                auto cd = [&](long double rot) { return rd * logl((long double) s.p.original_seq_len / (rot * 2.0L * 3.14159265358979323846264338327950288L)) / (2.0L * logl((long double) s.p.base)); };
                const long double low = std::max(floorl(cd((long double) s.p.beta_fast)), 0.0L), high = std::min(ceill(cd((long double) s.p.beta_slow)), (long double) (rd - 1));
                for (int i = 0; i < half; ++i) {
                    long double ramp = ((long double) i - low) / std::max(high - low, 1e-3L);
                    ramp = std::min(1.0L, std::max(0.0L, ramp));
                    const long double smooth = 1.0L - ramp;
                    fr[(size_t) i] = fr[(size_t) i] / (long double) s.p.factor * (1.0L - smooth) + fr[(size_t) i] * smooth;
                }
            }
            double worst = 0;
            for (int pos = 0; pos < seq; ++pos)
                for (int i = 0; i < half; ++i) {
                    const long double ang = (long double) pos * fr[(size_t) i];
                    const double wc = (double) cosl(ang), ws = (double) sinl(ang);
                    worst = std::max(worst, std::fabs(c[(size_t) pos * half + i] - wc));
                    worst = std::max(worst, std::fabs(sn[(size_t) pos * half + i] - ws));
                }
            rep.line(worst < 1.3e-7, (g + " rope table " + s.name).c_str(), "dim %d, %d positions: worst |float - long double| = %.2e (<= 1 ulp of a value <= 1)", rd, seq, worst);
        }
    }
    // 2. the oracle's golden values (real-model parameters, dim 64: independent of G's kRopeDim)
    {
        std::vector<float> c, sn;
        ds41_rope_table_host(64, 70000, {65536, 160000.0, 16.0, 32.0, 1.0}, c, sn);
        std::vector<float> c0, s0;
        ds41_rope_table_host(64, 70000, {0, 10000.0, 16.0, 32.0, 1.0}, c0, s0);
        size_t n = 0, bad = 0, ulp1 = 0;
        for (const auto& e : kRopeGolden) {
            const auto& cc = e.plain ? c0 : c;
            const auto& ss = e.plain ? s0 : sn;
            const float gc = cc[(size_t) e.pos * 32 + e.pair], gs = ss[(size_t) e.pos * 32 + e.pair];
            const uint32_t dc = f_bits(gc) > e.cos ? f_bits(gc) - e.cos : e.cos - f_bits(gc);
            const uint32_t ds = f_bits(gs) > e.sin ? f_bits(gs) - e.sin : e.sin - f_bits(gs);
            ++n;
            if (dc > 1 || ds > 1) ++bad;
            ulp1 += (dc == 1) + (ds == 1);
        }
        rep.line(bad == 0, (g + " rope table vs the numpy oracle").c_str(), "%zu golden entries (YaRN theta 160000 and plain 10000, positions up to 69999): %zu beyond 1 ulp, %zu at exactly 1 ulp", n, bad, ulp1);
    }
    // 3. the kernel, bit for bit against the separately rounded float formula, forward / inverse, in place / out of place
    for (int width : {G::kHeadDim, G::kIdxDim}) {
        const int T = 3, heads = 4, pos0 = 17, seq = 40;
        RopeParams prm{65536, 160000.0, 16.0, 32.0, 1.0};
        std::vector<float> c, sn;
        ds41_rope_table_host(rd, seq, prm, c, sn);
        DevBuf<float> dc(dev, c.size()), ds(dev, sn.size());
        upload(dev, dc, c);
        upload(dev, ds, sn);
        RopeTable tab{dc.p, ds.p, seq};
        std::vector<float> x((size_t) T * heads * width);
        for (auto& v : x) v = (float) (rng.gauss() * 3.0);
        for (int inverse = 0; inverse < 2; ++inverse) {
            std::vector<float> want = x;
            for (int r = 0; r < T * heads; ++r)
                for (int i = 0; i < rd / 2; ++i) {
                    const size_t ti = (size_t) (pos0 + r / heads) * (rd / 2) + i;
                    const float cs = c[ti];
                    const float sg = inverse ? -sn[ti] : sn[ti];
                    float* p = want.data() + (size_t) r * width + (width - rd) + 2 * i;
                    volatile float a_c = p[0] * cs, b_s = p[1] * sg, a_s = p[0] * sg, b_c = p[1] * cs;
                    p[0] = a_c - b_s;
                    p[1] = a_s + b_c;
                }
            DevBuf<float> dx(dev, x.size()), dy(dev, x.size()), dip(dev, x.size());
            upload(dev, dx, x);
            upload(dev, dip, x);
            ds41_rope<G>(dev, dx.p, dy.p, T, heads, width, tab, pos0, inverse != 0);
            ds41_rope<G>(dev, dip.p, dip.p, T, heads, width, tab, pos0, inverse != 0);
            const auto y = dy.down(), yi = dip.down();
            rep.line(count_diff(y, want) == 0 && count_diff(yi, want) == 0, (g + " rope kernel " + (inverse ? "inverse" : "forward")).c_str(),
                     "width %d, T %d x %d heads, pos0 %d: %zu / %zu elements differ from the float formula (out of place / in place), nope channels copied", width, T, heads, pos0, count_diff(y, want),
                     count_diff(yi, want));
            if (inverse == 0) {                                                       // forward then inverse = identity (to FP32 rounding)
                DevBuf<float> dz(dev, x.size());
                ds41_rope<G>(dev, dy.p, dz.p, T, heads, width, tab, pos0, true);
                const auto z = dz.down();
                double worst = 0;
                for (size_t i = 0; i < z.size(); ++i) worst = std::max(worst, (double) std::fabs(z[i] - x[i]));
                rep.line(worst < 2e-6, (g + " rope inverse(forward(x)) = x").c_str(), "worst |error| %.2e", worst);
            }
        }
    }
}

// =====================================================================================================================================================
// --shared (the shared expert)
// =====================================================================================================================================================
double ref_silu_double(double g) {
    const double e = std::exp(-std::fabs(g));
    const double sig = g >= 0 ? 1.0 / (1.0 + e) : e / (1.0 + e);
    return g * sig;
}
// the kernel's h in float, operation by operation (the host libm's expf is the one the emulator calls, so this is bit-exact there; on the card the device's
// expf may differ in the last bit: h_same compares within 4e-6 then)
bool h_same(float a, float b) {
    if (kEmu) return same_f(a, b);
    if (a != a || b != b) return a != a && b != b;
    return std::fabs(a - b) <= 4e-6 * std::max(std::fabs(a), std::fabs(b)) + 1e-30;
}
float model_h(float g, float u, float limit) {
    if (limit > 0.0f) {
        u = u > limit ? limit : (u < -limit ? -limit : u);
        g = g > limit ? limit : g;
    }
    const float e = std::exp(-std::fabs(g));
    volatile float d = 1.0f + e;
    volatile float sig = g >= 0.0f ? 1.0f / d : e / d;
    volatile float silu = g * sig;
    volatile float h = silu * u;
    return h;
}

template <class G>
void run_shared(Dev& dev, Report& rep, Rng& rng, bool big) {
    const std::string g = G::kName;
    (void) big;
    const bool real = std::strcmp(G::kName, "real") == 0;
    // 1. the SwiGLU epilogue on crafted values: the clamps, Inf, NaN
    {
        std::vector<float> gv, uv;
        const float gs[] = {-100.0f, -10.0f, -1.0f, 0.0f, -0.0f, 1.0f, 9.99f, 10.0f, 10.5f, 1e30f, 1.0f / 0.0f, -1.0f / 0.0f, bits_f(0x7FC00000u), bits_f(0xFFC00001u)};
        const float us[] = {-1e30f, -11.0f, -10.0f, -9.0f, 0.0f, 0.5f, 10.0f, 11.0f, 1.0f / 0.0f, -1.0f / 0.0f, bits_f(0x7FC00000u)};
        for (float a : gs)
            for (float b : us) {
                gv.push_back(a);
                uv.push_back(b);
            }
        while (gv.size() % 4) {
            gv.push_back(1.0f);
            uv.push_back(1.0f);
        }
        const int n = (int) gv.size();
        DevBuf<float> dg(dev, gv.size()), du(dev, uv.size());
        upload(dev, dg, gv);
        upload(dev, du, uv);
        ds41_swiglu<G>(dev, dg.p, du.p, n, 10.0f);
        const auto h = dg.down();
        size_t bad = 0, bad_ref = 0;
        for (int i = 0; i < n; ++i) {
            bad += !h_same(h[(size_t) i], model_h(gv[(size_t) i], uv[(size_t) i], 10.0f));
            // the oracle's semantics in double: clamp with NaN propagation, silu of ops.py
            double gg = gv[(size_t) i], uu = uv[(size_t) i];
            if (!std::isnan(uu)) uu = std::min(10.0, std::max(-10.0, uu));
            if (!std::isnan(gg)) gg = std::min(10.0, gg);
            const double want = ref_silu_double(gg) * uu;
            const bool ok = std::isnan(want) ? std::isnan(h[(size_t) i]) : std::fabs(h[(size_t) i] - want) <= 2e-6 * std::fabs(want) + 1e-30;
            bad_ref += !ok;
        }
        rep.line(bad == 0 && bad_ref == 0, (g + " swiglu clamps").c_str(), "%d (g, u) pairs incl. +-Inf / NaN: %zu differ from the float model (bits), %zu from the oracle's semantics (FP64)", n, bad, bad_ref);
        // limit <= 0: no clamp
        ds41_swiglu<G>(dev, dg.p, du.p, n, 10.0f);                                      // (dg now holds h: reload)
        upload(dev, dg, gv);
        ds41_swiglu<G>(dev, dg.p, du.p, n, 0.0f);
        const auto h0 = dg.down();
        size_t bad0 = 0;
        for (int i = 0; i < n; ++i) bad0 += !h_same(h0[(size_t) i], model_h(gv[(size_t) i], uv[(size_t) i], 0.0f));
        rep.line(bad0 == 0, (g + " swiglu without a limit").c_str(), "%zu differ", bad0);
    }
    // 2. the whole shared expert, both modes
    const int kFF = G::kFF, kH = G::kHidden;
    const Q8Mat w1 = make_q8(rng, kFF, kH, 2), w3 = make_q8(rng, kFF, kH, 2), w2 = make_q8(rng, kH, kFF, 2);
    // keep the activations moderate: scales around 2^-9..2^2 make |g| of a few units, so the clamps are exercised without everything saturating
    DevBuf<uint8_t> d1(dev, w1.bytes.size()), d3(dev, w3.bytes.size()), d2(dev, w2.bytes.size());
    upload(dev, d1, w1.bytes);
    upload(dev, d3, w3.bytes);
    upload(dev, d2, w2.bytes);
    SharedExpertWeights sw{d1.p, d3.p, d2.p};
    for (int T : {1, 3, real ? 3 : 8}) {
        std::vector<float> x((size_t) T * kH);
        for (auto& v : x) v = (float) (rng.gauss() * 0.3);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<unsigned char> scratch(dev, shared_expert_scratch_bytes<G>(T));
        const SharedExpertScratch sc = shared_expert_scratch_carve<G>(scratch.p, T);
        DevBuf<float> dy(dev, (size_t) T * kH);
        // ---- int8 mode, stage by stage
        ds41_shared_expert<G>(dev, sw, dx.p, T, dy.p, sc, true, 10.0f);
        const auto xq = download(dev, sc.xq, (size_t) T * kH);
        const auto xs = download(dev, sc.xs, (size_t) T * kH / 32);
        const auto gdev = download(dev, sc.g, (size_t) T * kFF), udev = download(dev, sc.u, (size_t) T * kFF);
        const auto hq = download(dev, sc.hq, (size_t) T * kFF);
        const auto hs = download(dev, sc.hs, (size_t) T * kFF / 32);
        const auto y = dy.down();
        // (a) x -> xq / xs by the contract's rule
        std::vector<int8_t> rq(x.size());
        std::vector<float> rs(x.size() / 32);
        ref_quant(x.data(), (int) x.size(), rq.data(), rs.data());
        size_t bad = 0;
        for (size_t i = 0; i < xq.size(); ++i) bad += xq[i] != rq[i];
        for (size_t i = 0; i < xs.size(); ++i) bad += f_bits(xs[i]) != f_bits(rs[i]);
        rep.line(bad == 0, (g + " shared expert: input quantisation").c_str(), "T %d: %zu mismatches", T, bad);
        // (b) gate / up against the order model and FP64
        size_t bad_model = 0, bad_ref = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < kFF; ++r)
                for (int which = 0; which < 2; ++which) {
                    const Q8Mat& m = which ? w3 : w1;
                    const Int8Row br = int8_blocks(m.row(r), kH, xq.data() + (size_t) t * kH);
                    const float want = model_int8(br, xs.data() + (size_t) t * (kH / 32), kH);
                    const float got = (which ? udev : gdev)[(size_t) t * kFF + r];
                    bad_model += !same_f(got, want);
                    double v, a;
                    ref_int8(br, xs.data() + (size_t) t * (kH / 32), kH, v, a);
                    bad_ref += !close(got, v, a, 4e-6);
                }
        rep.line(bad_model == 0 && bad_ref == 0, (g + " shared expert: gate / up").c_str(), "T %d: %zu differ from the order model, %zu from FP64", T, bad_model, bad_ref);
        // (c) h = swiglu(g, u) quantised: the device's g, u pushed through the float model and the reference quantiser
        std::vector<float> h((size_t) T * kFF);
        for (size_t i = 0; i < h.size(); ++i) h[i] = model_h(gdev[i], udev[i], 10.0f);
        std::vector<int8_t> rhq(h.size());
        std::vector<float> rhs(h.size() / 32);
        ref_quant(h.data(), (int) h.size(), rhq.data(), rhs.data());
        size_t badh = 0, off_by_one = 0;
        for (size_t i = 0; i < hq.size(); ++i) {
            if (kEmu) badh += hq[i] != rhq[i];
            else {                                                                      // on the card a last-bit difference of h may move a code by one
                const int dq = std::abs((int) hq[i] - (int) rhq[i]);
                badh += dq > 1;
                off_by_one += dq == 1;
            }
        }
        for (size_t i = 0; i < hs.size(); ++i) badh += kEmu ? f_bits(hs[i]) != f_bits(rhs[i]) : !h_same(hs[i], rhs[i]);
        if (!kEmu && off_by_one * 100 > hq.size()) badh += off_by_one;                  // more than 1 % of the codes off by one: not a last-bit effect
        rep.line(badh == 0, (g + " shared expert: swiglu + quantise").c_str(), "T %d: %zu mismatches of h (int8 / scale) against the float model (%zu codes off by one)", T, badh, off_by_one);
        // (d) down projection of the device's hq / hs
        size_t bady = 0, bady_ref = 0;
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < kH; ++r) {
                const Int8Row br = int8_blocks(w2.row(r), kFF, hq.data() + (size_t) t * kFF);
                const float want = model_int8(br, hs.data() + (size_t) t * (kFF / 32), kFF);
                bady += !same_f(y[(size_t) t * kH + r], want);
                double v, a;
                ref_int8(br, hs.data() + (size_t) t * (kFF / 32), kFF, v, a);
                bady_ref += !close(y[(size_t) t * kH + r], v, a, 4e-6);
            }
        rep.line(bady == 0 && bady_ref == 0, (g + " shared expert: down (int8)").c_str(), "T %d: %zu differ from the order model, %zu from FP64", T, bady, bady_ref);
        // (e) the pre-quantised entry point gives the same bits
        DevBuf<float> dy2(dev, (size_t) T * kH);
        ds41_shared_expert_q<G>(dev, sw, sc.xq, sc.xs, T, dy2.p, sc);
        rep.line(count_diff(y, dy2.down()) == 0, (g + " shared expert: _q entry == full call").c_str(), "T %d", T);
        // (f) the full int8 result against the oracle's maths, end to end (FP64 from x): the quantisation noise bounds the difference, so a loose bound
        {
            double num = 0, den = 0;
            for (int t = 0; t < T; ++t) {
                std::vector<double> hh((size_t) kFF);
                for (int r = 0; r < kFF; ++r) {
                    double a1 = 0, a3 = 0;
                    for (int b = 0; b < kH / 32; ++b) {
                        const double d1v = half_to_double((uint16_t) (w1.row(r)[b * 34] | (w1.row(r)[b * 34 + 1] << 8))), d3v = half_to_double((uint16_t) (w3.row(r)[b * 34] | (w3.row(r)[b * 34 + 1] << 8)));
                        for (int j = 0; j < 32; ++j) {
                            const double xv = x[(size_t) t * kH + b * 32 + j];
                            a1 += xv * d1v * (int8_t) w1.row(r)[b * 34 + 2 + j];
                            a3 += xv * d3v * (int8_t) w3.row(r)[b * 34 + 2 + j];
                        }
                    }
                    a3 = std::min(10.0, std::max(-10.0, a3));
                    a1 = std::min(10.0, a1);
                    hh[(size_t) r] = ref_silu_double(a1) * a3;
                }
                for (int r = 0; r < kH; ++r) {
                    double a = 0;
                    for (int b = 0; b < kFF / 32; ++b) {
                        const double dv = half_to_double((uint16_t) (w2.row(r)[b * 34] | (w2.row(r)[b * 34 + 1] << 8)));
                        for (int j = 0; j < 32; ++j) a += hh[(size_t) (b * 32 + j)] * dv * (int8_t) w2.row(r)[b * 34 + 2 + j];
                    }
                    num += (y[(size_t) t * kH + r] - a) * (y[(size_t) t * kH + r] - a);
                    den += a * a;
                }
            }
            const double rel = std::sqrt(num / std::max(den, 1e-300));
            rep.line(rel < 0.03, (g + " shared expert: int8 vs exact maths").c_str(), "T %d: relative RMS difference %.3e (activation quantisation noise, bound 3e-2)", T, rel);
        }
        // ---- FP32 mode: the exact semantics
        DevBuf<float> dyf(dev, (size_t) T * kH);
        ds41_shared_expert<G>(dev, sw, dx.p, T, dyf.p, sc, false, 10.0f);
        const auto yf = dyf.down();
        {
            double num = 0, den = 0;
            size_t bad_f = 0;
            const auto gf = download(dev, sc.g, (size_t) T * kFF);                       // h (over g)
            for (int t = 0; t < T; ++t) {
                std::vector<float> hm((size_t) kFF);
                for (int r = 0; r < kFF; ++r) {
                    const float a1 = model_f32(w1.row(r), x.data() + (size_t) t * kH, kH), a3 = model_f32(w3.row(r), x.data() + (size_t) t * kH, kH);
                    hm[(size_t) r] = model_h(a1, a3, 10.0f);
                    bad_f += !h_same(gf[(size_t) t * kFF + r], hm[(size_t) r]);
                }
                for (int r = 0; r < kH; ++r) {                                           // the down projection of the DEVICE's h: bit-exact against the order model
                    const float want = model_f32(w2.row(r), gf.data() + (size_t) t * kFF, kFF);
                    bad_f += !same_f(yf[(size_t) t * kH + r], want);
                    double v, a;
                    ref_f32(w2.row(r), gf.data() + (size_t) t * kFF, kFF, v, a);
                    num += (yf[(size_t) t * kH + r] - v) * (yf[(size_t) t * kH + r] - v);
                    den += v * v;
                }
            }
            rep.line(bad_f == 0, (g + " shared expert: FP32 mode == the float model of every stage").c_str(), "T %d: %zu elements differ (h and y, bits)", T, bad_f);
            (void) num;
            (void) den;
        }
        // ---- T-invariance of the whole expert
        if (T > 1) {
            size_t bad_t = 0;
            for (int t = 0; t < T; ++t) {
                DevBuf<unsigned char> s1(dev, shared_expert_scratch_bytes<G>(1));
                const SharedExpertScratch sc1 = shared_expert_scratch_carve<G>(s1.p, 1);
                DevBuf<float> d1y(dev, kH);
                ds41_shared_expert<G>(dev, sw, dx.p + (size_t) t * kH, 1, d1y.p, sc1, true, 10.0f);
                const auto y1 = d1y.down();
                for (int r = 0; r < kH; ++r) bad_t += !same_f(y1[(size_t) r], y[(size_t) t * kH + r]);
            }
            rep.line(bad_t == 0, (g + " shared expert: T-invariance (int8)").c_str(), "T %d: %zu outputs differ from the T = 1 calls", T, bad_t);
        }
    }
    // 3. NaN end to end: a NaN in x reaches every output of its token (both modes), other tokens are untouched
    {
        const int T = 2;
        std::vector<float> x((size_t) T * kH);
        for (auto& v : x) v = (float) (rng.gauss() * 0.3);
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<unsigned char> scratch(dev, shared_expert_scratch_bytes<G>(T));
        const SharedExpertScratch sc = shared_expert_scratch_carve<G>(scratch.p, T);
        DevBuf<float> dy0(dev, (size_t) T * kH), dy(dev, (size_t) T * kH);
        for (int mode = 0; mode < 2; ++mode) {
            ds41_shared_expert<G>(dev, sw, dx.p, T, dy0.p, sc, mode == 0, 10.0f);
            x[(size_t) kH + 7] = bits_f(0x7FC00000u);
            upload(dev, dx, x);
            ds41_shared_expert<G>(dev, sw, dx.p, T, dy.p, sc, mode == 0, 10.0f);
            x[(size_t) kH + 7] = 0.25f;
            upload(dev, dx, x);
            const auto a = dy0.down(), b = dy.down();
            size_t nan1 = 0, bad = 0;
            for (int r = 0; r < kH; ++r) {
                nan1 += std::isnan(b[(size_t) kH + r]);
                bad += !same_f(a[(size_t) r], b[(size_t) r]);
            }
            rep.line(nan1 == (size_t) kH && bad == 0, (g + (mode == 0 ? " shared expert NaN (int8)" : " shared expert NaN (f32)")).c_str(), "%zu of %d outputs of the NaN token are NaN; %zu outputs of the clean token changed", nan1, kH, bad);
        }
    }
}

// =====================================================================================================================================================
// --vocab (argmax, top-k, embedding rows, elementwise)
// =====================================================================================================================================================
uint32_t ref_key(float v) {
    if (v != v) return 0xFFFFFFFFu;
    const uint32_t b = f_bits(v == 0.0f ? 0.0f : v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
template <class G>
void run_vocab(Dev& dev, Report& rep, Rng& rng) {
    const std::string g = G::kName;
    const int V = G::kVocab;
    for (int T : {1, 3}) {
        std::vector<float> x((size_t) T * V);
        for (auto& v : x) v = (float) std::lround(rng.gauss() * 4.0) * 0.25f;            // heavy ties
        // plant specials: a NaN, +/- Inf, a zero pair, a block of -inf
        if (T >= 1) {
            x[(size_t) 0 * V + V / 3] = 1.0f / 0.0f;
            x[(size_t) 0 * V + V - 1] = 1.0f / 0.0f;                                       // a later tie of the maximum: the lower index must win
        }
        if (T >= 2) {
            x[(size_t) 1 * V + 5] = bits_f(0x7FC00000u);
            x[(size_t) 1 * V + 9] = bits_f(0xFFC00001u);                                    // a second NaN (negative sign): the first NaN index wins
            x[(size_t) 1 * V + 1] = 1.0f / 0.0f;
        }
        if (T >= 3) {
            for (int i = 0; i < V; ++i) x[(size_t) 2 * V + i] = -1.0f / 0.0f;
            x[(size_t) 2 * V + V / 2] = -0.0f;
            x[(size_t) 2 * V + V / 2 + 1] = 0.0f;                                           // -0 == +0: the lower index wins
        }
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<int32_t> di(dev, (size_t) T);
        DevBuf<float> dv(dev, (size_t) T);
        std::vector<int32_t> base;
        size_t bad_order = 0;
        for (const char* o : kOrders) {
            set_sched(o);
            ds41_argmax<G>(dev, dx.p, T, di.p, dv.p);
            const auto idx = di.down();
            if (base.empty()) base = idx;
            else bad_order += idx != base;
        }
        set_cmdline_order();
        size_t bad = 0;
        for (int t = 0; t < T; ++t) {
            int bi = 0;
            for (int i = 1; i < V; ++i)
                if (ref_key(x[(size_t) t * V + i]) > ref_key(x[(size_t) t * V + bi])) bi = i;
            bad += base[(size_t) t] != bi;
            const auto vals = dv.down();
            bad += !same_f(vals[(size_t) t], x[(size_t) t * V + bi]);
        }
        rep.line(bad == 0 && bad_order == 0, (g + " argmax").c_str(), "T %d, vocab %d, many ties / Inf / NaN: %zu wrong results, %zu differences between scheduling orders", T, V, bad, bad_order);
    }
    for (int k : {1, 2, 7, 64}) {
        const int T = 2;
        std::vector<float> x((size_t) T * V);
        for (auto& v : x) v = (float) std::lround(rng.gauss() * 6.0) * 0.5f;             // ties across slices
        x[(size_t) 3] = 1.0f / 0.0f;
        x[(size_t) V - 2] = 1.0f / 0.0f;
        x[(size_t) V + 11] = bits_f(0x7FC00000u);
        x[(size_t) V + V / 2] = -1.0f / 0.0f;
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<unsigned char> scratch(dev, topk_scratch_bytes<G>(T, k));
        DevBuf<int32_t> di(dev, (size_t) T * k);
        DevBuf<float> dv(dev, (size_t) T * k);
        std::vector<int32_t> base;
        size_t bad_order = 0;
        for (const char* o : kOrders) {
            set_sched(o);
            ds41_topk<G>(dev, dx.p, T, k, di.p, dv.p, scratch.p, topk_scratch_bytes<G>(T, k));
            const auto idx = di.down();
            if (base.empty()) base = idx;
            else bad_order += idx != base;
        }
        set_cmdline_order();
        const auto vals = dv.down();
        size_t bad = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<int> order((size_t) V);
            for (int i = 0; i < V; ++i) order[(size_t) i] = i;
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ref_key(x[(size_t) t * V + a]) > ref_key(x[(size_t) t * V + b]); });
            for (int i = 0; i < k; ++i) {
                bad += base[(size_t) t * k + i] != order[(size_t) i];
                bad += !same_f(vals[(size_t) t * k + i], x[(size_t) t * V + order[(size_t) i]]) && !(x[(size_t) t * V + order[(size_t) i]] == 0.0f && vals[(size_t) t * k + i] == 0.0f);
            }
        }
        rep.line(bad == 0 && bad_order == 0, (g + " top-k").c_str(), "k %d, T %d, vocab %d: %zu wrong entries vs a stable sort, %zu differences between scheduling orders", k, T, V, bad, bad_order);
    }
    // a vocabulary smaller than a slice and k == 64 on the mini geometry is covered above (V = 512); a plain top-1 == argmax check
    {
        const int T = 2;
        std::vector<float> x((size_t) T * V);
        for (auto& v : x) v = (float) rng.gauss();
        DevBuf<float> dx(dev, x.size());
        upload(dev, dx, x);
        DevBuf<unsigned char> scratch(dev, topk_scratch_bytes<G>(T, 1));
        DevBuf<int32_t> di(dev, T), da(dev, T);
        DevBuf<float> dv(dev, T), dva(dev, T);
        ds41_topk<G>(dev, dx.p, T, 1, di.p, dv.p, scratch.p, topk_scratch_bytes<G>(T, 1));
        ds41_argmax<G>(dev, dx.p, T, da.p, dva.p);
        rep.line(di.down() == da.down(), (g + " top-1 == argmax").c_str(), "random normal logits, T %d", T);
    }
    // embedding rows (host -> device)
    {
        const int H = G::kHidden;
        std::vector<uint16_t> table((size_t) V * H);
        for (auto& v : table) v = (uint16_t) rng.u32();
        const int32_t ids[3] = {0, V - 1, V / 2};
        DevBuf<float> dout(dev, (size_t) 3 * H);
        std::vector<float> stage;
        ds41_embed_rows<G>(dev, table.data(), ids, 3, dout.p, stage);
        const auto out = dout.down();
        size_t bad = 0;
        for (int t = 0; t < 3; ++t)
            for (int d = 0; d < H; ++d) bad += !same_f(out[(size_t) t * H + d], bits_f((uint32_t) table[(size_t) ids[t] * H + d] << 16));
        bool threw = false;
        try {
            const int32_t badid[1] = {V};
            ds41_embed_rows<G>(dev, table.data(), badid, 1, dout.p, stage);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        rep.line(bad == 0 && threw, (g + " embedding rows").c_str(), "3 rows of %d BF16 -> FP32 -> device: %zu wrong elements; an out-of-range id %s", H, bad, threw ? "is refused" : "was NOT refused");
    }
    // elementwise
    {
        const int n = 1000 * 4 + 4;
        std::vector<float> a((size_t) n), b((size_t) n);
        for (int i = 0; i < n; ++i) {
            a[(size_t) i] = (float) rng.gauss();
            b[(size_t) i] = (float) rng.gauss();
        }
        DevBuf<float> da(dev, a.size()), db(dev, b.size()), dc(dev, a.size());
        upload(dev, da, a);
        upload(dev, db, b);
        ds41_f32_add<G>(dev, da.p, db.p, n, dc.p);
        auto c = dc.down();
        size_t bad = 0;
        for (int i = 0; i < n; ++i) bad += f_bits(c[(size_t) i]) != f_bits(a[(size_t) i] + b[(size_t) i]);
        ds41_f32_add<G>(dev, da.p, db.p, n, da.p);                                        // in place
        bad += count_diff(da.down(), c);
        upload(dev, da, a);
        ds41_f32_scale<G>(dev, da.p, 0.75f, n, dc.p);
        c = dc.down();
        for (int i = 0; i < n; ++i) bad += f_bits(c[(size_t) i]) != f_bits(a[(size_t) i] * 0.75f);
        rep.line(bad == 0, (g + " elementwise add / scale").c_str(), "n %d: %zu wrong", n, bad);
    }
}


// =====================================================================================================================================================
// --args (what the entry points refuse)
// =====================================================================================================================================================
template <class G>
void run_args(Dev& dev, Report& rep, Rng& rng) {
    const std::string g = G::kName;
    (void) rng;
    auto refused = [&](const char* what, const std::function<void()>& f) {
        bool threw = false;
        try {
            f();
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        rep.line(threw, (g + " refuses: " + what).c_str(), "std::invalid_argument");
    };
    auto accepted = [&](const char* what, const std::function<void()>& f) {
        bool threw = false;
        try {
            f();
        } catch (const std::exception& e) {
            threw = true;
            std::printf("  unexpected: %s\n", e.what());
        }
        rep.line(!threw, (g + " accepts: " + what).c_str(), "no exception");
    };
    DevBuf<uint8_t> w(dev, 64 * 34 * 4 + 64);
    DevBuf<float> x(dev, 8192), y(dev, 8192), z(dev, 8192);
    DevBuf<int8_t> xq(dev, 8192);
    DevBuf<float> xs(dev, 256);
    DevBuf<uint16_t> wb(dev, 4096);
    DevBuf<unsigned char> scratch(dev, 1 << 20);
    DevBuf<float> lg(dev, (size_t) G::kVocab);                                       // one row of logits (zeros: DevBuf memory is 0xCD-filled, finite)
    lg.zero();
    DevBuf<int32_t> di(dev, 128);
    DevBuf<float> dv(dev, 128);
    accepted("a valid int8 GEMV", [&] { ds41_gemv_q8_int8<G>(dev, w.p, 64, 128, xq.p, xs.p, 2, y.p); });
    refused("q8 int8: k not a multiple of 32", [&] { ds41_gemv_q8_int8<G>(dev, w.p, 64, 100, xq.p, xs.p, 1, y.p); });
    refused("q8 int8: n = 0", [&] { ds41_gemv_q8_int8<G>(dev, w.p, 0, 128, xq.p, xs.p, 1, y.p); });
    refused("q8 int8: T = 0", [&] { ds41_gemv_q8_int8<G>(dev, w.p, 64, 128, xq.p, xs.p, 0, y.p); });
    refused("q8 int8: misaligned xq", [&] { ds41_gemv_q8_int8<G>(dev, w.p, 64, 128, xq.p + 1, xs.p, 1, y.p); });
    refused("q8 int8: null weights", [&] { ds41_gemv_q8_int8<G>(dev, nullptr, 64, 128, xq.p, xs.p, 1, y.p); });
    refused("q8 int8: odd weight pointer", [&] { ds41_gemv_q8_int8<G>(dev, w.p + 1, 64, 128, xq.p, xs.p, 1, y.p); });
    accepted("q8 int8 weights at a 2-byte offset (the generic path)", [&] { ds41_gemv_q8_int8<G>(dev, w.p + 2, 64, 128, xq.p, xs.p, 1, y.p); });
    refused("q8 f32: misaligned x", [&] { ds41_gemv_q8_f32<G>(dev, w.p, 64, 128, x.p + 1, 1, y.p); });
    refused("grouped: groups = 0", [&] { ds41_gemv_q8_grouped_f32<G>(dev, w.p, 0, 8, 128, x.p, 1, y.p); });
    accepted("a valid BF16 GEMV", [&] { ds41_gemv_bf16<G>(dev, wb.p, 16, 64, x.p, 1, y.p); });
    refused("bf16: k % 8 != 0", [&] { ds41_gemv_bf16<G>(dev, wb.p, 16, 60, x.p, 1, y.p); });
    refused("bf16: misaligned weights", [&] { ds41_gemv_bf16<G>(dev, wb.p + 1, 16, 64, x.p, 1, y.p); });
    refused("f32: k % 4 != 0", [&] { ds41_gemv_f32<G>(dev, x.p, 16, 62, y.p, 1, z.p); });
    refused("rmsnorm: width % 4 != 0", [&] { ds41_rmsnorm<G>(dev, x.p, nullptr, 2, 6, y.p); });
    refused("rmsnorm: misaligned x", [&] { ds41_rmsnorm<G>(dev, x.p + 1, nullptr, 2, 8, y.p); });
    std::vector<float> c, sn;
    ds41_rope_table_host(G::kRopeDim, 8, RopeParams{}, c, sn);
    DevBuf<float> dc(dev, c.size()), ds(dev, sn.size());
    upload(dev, dc, c);
    upload(dev, ds, sn);
    const RopeTable tab{dc.p, ds.p, 8};
    accepted("a valid RoPE", [&] { ds41_rope<G>(dev, x.p, y.p, 2, 1, G::kHeadDim, tab, 3, false); });
    refused("rope: positions past the table", [&] { ds41_rope<G>(dev, x.p, y.p, 2, 1, G::kHeadDim, tab, 7, false); });
    refused("rope: width < kRopeDim", [&] { ds41_rope<G>(dev, x.p, y.p, 1, 1, G::kRopeDim - 2, tab, 0, false); });
    refused("rope: out partially overlaps x", [&] { ds41_rope<G>(dev, x.p, x.p + 2, 1, 1, G::kHeadDim, tab, 0, false); });
    accepted("top-k at the limit (k = 64)", [&] { ds41_topk<G>(dev, lg.p, 1, 64, di.p, dv.p, scratch.p, topk_scratch_bytes<G>(1, 64)); });
    refused("top-k: k = 65", [&] { ds41_topk<G>(dev, lg.p, 1, 65, di.p, dv.p, scratch.p, topk_scratch_bytes<G>(1, 65)); });
    refused("top-k: k = 0", [&] { ds41_topk<G>(dev, lg.p, 1, 0, di.p, dv.p, scratch.p, 1 << 20); });
    refused("top-k: scratch too small", [&] { ds41_topk<G>(dev, lg.p, 1, 8, di.p, dv.p, scratch.p, topk_scratch_bytes<G>(1, 8) - 1); });
    refused("argmax: T = 0", [&] { ds41_argmax<G>(dev, lg.p, 0, di.p, dv.p); });
    refused("f32 add: n % 4 != 0", [&] { ds41_f32_add<G>(dev, x.p, y.p, 6, z.p); });
    SharedExpertWeights sw{w.p, w.p, w.p};
    SharedExpertScratch sc;
    refused("shared expert: T = 0", [&] { ds41_shared_expert<G>(dev, sw, x.p, 0, y.p, sc); });
    // an out-of-range token id never reaches the device
    refused("embedding: id out of range", [&] {
        std::vector<uint16_t> table((size_t) 4 * 8, 0x3F80);
        std::vector<float> out(8);
        const int32_t ids[1] = {4};
        ds41_embed_rows_host(table.data(), 4, 8, ids, 1, out.data());
    });
}

// =====================================================================================================================================================
// --bench (the card only): GB/s of weights per call, against the card's peak
// =====================================================================================================================================================
#if defined(DS41_DENSE_GPU)
template <class G>
void run_bench(Dev& dev, Report& rep, Rng& rng) {
    int mem_clock = 0, bus = 0, sms = 0;
    cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const double peak = 2.0 * mem_clock * 1e3 * (bus / 8) / 1e9;                           // GB/s (HBM2: double data rate)
    Report::info("BENCH device: %d SMs, peak %.0f GB/s (memory clock %d kHz, bus %d bits); weights are cycled through enough copies that no call finds its matrix in L2", sms, peak, mem_clock, bus);
    auto say = [&](const char* name, const std::string& shape, double us, double bytes) {
        const double gbs = bytes / (us * 1e-6) / 1e9;
        Report::info("BENCH %-12s %-34s %9.1f us %8.1f GB/s %5.1f%% of peak", name, shape.c_str(), us, gbs, 100.0 * gbs / peak);
    };
    using D = Derived<G>;
    struct S {
        const char* name;
        int n, k;
    };
    const S shapes[] = {{"wq_a", G::kQLora, G::kHidden},   {"wq_b", D::kQ, G::kQLora},       {"wkv", G::kHeadDim, G::kHidden}, {"wo_b", G::kHidden, D::kOMid},
                        {"shexp gate", G::kFF, G::kHidden}, {"shexp down", G::kHidden, G::kFF}, {"idx wq_b", D::kIdxQ, G::kQLora}, {"engram wkv", D::kEngramOut, D::kEngramIn}};
    for (const S& sh : shapes) {
        const Q8Mat m = make_q8(rng, sh.n, sh.k, 0);
        const double bytes = (double) m.bytes.size();
        const int copies = (int) std::max(1.0, std::min(8.0, std::ceil(96e6 / bytes)));
        std::vector<std::unique_ptr<DevBuf<uint8_t>>> w;
        for (int c = 0; c < copies; ++c) {
            w.emplace_back(new DevBuf<uint8_t>(dev, m.bytes.size()));
            upload(dev, *w.back(), m.bytes);
        }
        for (int T : {1, 2, 4, 8}) {
            const std::vector<float> x = make_x(rng, T, sh.k, 0);
            DevBuf<float> dx(dev, x.size()), dy(dev, (size_t) T * sh.n);
            upload(dev, dx, x);
            DevBuf<int8_t> dq(dev, (size_t) T * sh.k);
            DevBuf<float> ds(dev, (size_t) T * sh.k / 32);
            quantize_nat<G>(dev, dx.p, T, sh.k, dq.p, ds.p);
            int i = 0;
            const double us = dev.time_us([&] { ds41_gemv_q8_int8<G>(dev, w[(size_t) (i++ % copies)]->p, sh.n, sh.k, dq.p, ds.p, T, dy.p); }, 30);
            say("q8 int8", fmt_s("%s n=%d k=%d T=%d", sh.name, sh.n, sh.k, T), us, bytes);
        }
        {
            const int T = 1;
            const std::vector<float> x = make_x(rng, T, sh.k, 0);
            DevBuf<float> dx(dev, x.size()), dy(dev, (size_t) T * sh.n);
            upload(dev, dx, x);
            int i = 0;
            const double us = dev.time_us([&] { ds41_gemv_q8_f32<G>(dev, w[(size_t) (i++ % copies)]->p, sh.n, sh.k, dx.p, T, dy.p); }, 30);
            say("q8 f32", fmt_s("%s n=%d k=%d T=%d", sh.name, sh.n, sh.k, T), us, bytes);
        }
    }
    {   // wo_a: the grouped GEMV
        const int groups = G::kOGroups, R = G::kOLora, K = D::kOGroupIn;
        const Q8Mat m = make_q8(rng, groups * R, K, 0);
        const double bytes = (double) m.bytes.size();
        const int copies = 3;
        std::vector<std::unique_ptr<DevBuf<uint8_t>>> w;
        for (int c = 0; c < copies; ++c) {
            w.emplace_back(new DevBuf<uint8_t>(dev, m.bytes.size()));
            upload(dev, *w.back(), m.bytes);
        }
        for (int T : {1, 2, 4, 8}) {
            const auto x = make_x(rng, T, groups * K, 0);
            DevBuf<float> dx(dev, x.size()), dy(dev, (size_t) T * groups * R);
            upload(dev, dx, x);
            int i = 0;
            const double us = dev.time_us([&] { ds41_wo_a<G>(dev, w[(size_t) (i++ % copies)]->p, dx.p, T, dy.p); }, 30);
            say("wo_a", fmt_s("%d groups x %d x %d T=%d", groups, R, K, T), us, bytes);
        }
    }
    {   // the head: BF16, 1.3 GB
        const int N = G::kVocab, K = G::kHidden;
        std::vector<uint16_t> pat(1 << 20);
        for (auto& v : pat) v = (uint16_t) (0x3C00u + (rng.u32() & 0x3FFu) + ((rng.u32() & 1u) << 15));        // finite BF16 values around 2^-7
        std::vector<uint16_t> host((size_t) N * K);
        for (size_t i = 0; i < host.size(); ++i) host[i] = pat[i & (pat.size() - 1)];
        DevBuf<uint16_t> w(dev, host.size());
        upload(dev, w, host);
        for (int T : {1, 2, 4, 8}) {
            const auto x = make_x(rng, T, K, 0);
            DevBuf<float> dx(dev, x.size()), dy(dev, (size_t) T * N);
            upload(dev, dx, x);
            const double us = dev.time_us([&] { ds41_head<G>(dev, w.p, dx.p, T, dy.p); }, 10);
            say("head bf16", fmt_s("n=%d k=%d T=%d", N, K, T), us, (double) host.size() * 2);
            DevBuf<int32_t> di(dev, T);
            DevBuf<float> dv(dev, T);
            const double ua = dev.time_us([&] { ds41_argmax<G>(dev, dy.p, T, di.p, dv.p); }, 20);
            Report::info("BENCH %-12s %-34s %9.1f us", "argmax", fmt_s("T=%d vocab=%d", T, N).c_str(), ua);
            DevBuf<unsigned char> scratch(dev, topk_scratch_bytes<G>(T, 64));
            DevBuf<int32_t> ti(dev, (size_t) T * 64);
            DevBuf<float> tv(dev, (size_t) T * 64);
            const double ut = dev.time_us([&] { ds41_topk<G>(dev, dy.p, T, 64, ti.p, tv.p, scratch.p, topk_scratch_bytes<G>(T, 64)); }, 10);
            Report::info("BENCH %-12s %-34s %9.1f us", "top-64", fmt_s("T=%d vocab=%d", T, N).c_str(), ut);
        }
    }
    rep.line(true, "bench", "timings printed above (INFO BENCH lines)");
}
#endif

// =====================================================================================================================================================
// main
// =====================================================================================================================================================
template <class G>
void run_geom(Dev& dev, Report& rep, uint64_t seed, bool big, bool bench, bool quant, bool gemv, bool wide, bool norm, bool rope, bool shared, bool vocab, bool argsuite) {
    Rng rng(seed);
    const auto t0 = std::chrono::steady_clock::now();
    auto lap = [&](const char* what) {
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        Report::info("%s %s done (%.1f s since the geometry started)", G::kName, what, s);
    };
    if (quant) run_quant<G>(dev, rep, rng), lap("quant");
    if (gemv) run_gemv<G>(dev, rep, rng, big), lap("gemv");
    if (wide) run_wide<G>(dev, rep, rng, big), lap("wide");
    if (norm) run_norm<G>(dev, rep, rng), lap("norm");
    if (rope) run_rope<G>(dev, rep, rng), lap("rope");
    if (shared) run_shared<G>(dev, rep, rng, big), lap("shared");
    if (vocab) run_vocab<G>(dev, rep, rng), lap("vocab");
    if (argsuite) run_args<G>(dev, rep, rng), lap("args");
#if defined(DS41_DENSE_GPU)
    if (bench) run_bench<G>(dev, rep, rng), lap("bench");
#else
    (void) bench;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    bool quant = false, gemv = false, wide = false, norm = false, rope = false, shared = false, vocab = false, big = false, bench = false, argsuite = false;
    std::string geom = "both";
    uint64_t seed = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quant") quant = true;
        else if (a == "--gemv") gemv = true;
        else if (a == "--wide") wide = true;
        else if (a == "--norm") norm = true;
        else if (a == "--rope") rope = true;
        else if (a == "--shared") shared = true;
        else if (a == "--vocab") vocab = true;
        else if (a == "--args") argsuite = true;
        else if (a == "--all") quant = gemv = wide = norm = rope = shared = vocab = argsuite = true;
        else if (a == "--big") big = true;
        else if (a == "--bench") bench = true;
        else if (a == "--geom" && i + 1 < argc) geom = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--order" && i + 1 < argc) {
            g_order = argv[++i];
            if (!set_sched(g_order)) {
                std::fprintf(stderr, "--order: forward | reverse | shuffle[:SEED]\n");
                return 2;
            }
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (!(quant || gemv || wide || norm || rope || shared || vocab || bench || argsuite)) quant = gemv = wide = norm = rope = shared = vocab = argsuite = true;
    Report rep;
#if defined(DS41_DENSE_GPU)
    CudaDev dev;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess)
        Report::info("device: %s (cc %d.%d, %d SMs, %.1f GB)%s", prop.name, prop.major, prop.minor, prop.multiProcessorCount, (double) prop.totalGlobalMem / 1e9,
                     (prop.major == 7 && prop.minor == 0) ? "" : "  <-- NOT a Volta (sm_70) card: the numbers say nothing about the V100");
    Report::info("dense kernels on the card, geometry real, seed %llu", (unsigned long long) seed);
    run_geom<RealGeom>(dev, rep, seed, big, bench, quant, gemv, wide, norm, rope, shared, vocab, argsuite);          // the nvcc library instantiates RealGeom only
#else
    HostDev dev;
    Report::info("dense kernels, emulated, order %s, geometry %s, seed %llu", g_order.c_str(), geom.c_str(), (unsigned long long) seed);
    if (geom == "mini" || geom == "both") run_geom<MiniGeom>(dev, rep, seed, big, bench, quant, gemv, wide, norm, rope, shared, vocab, argsuite);
    if (geom == "real" || geom == "both") run_geom<RealGeom>(dev, rep, seed, big, bench, quant, gemv, wide, norm, rope, shared, vocab, argsuite);
#endif
    return rep.summary();
}
