// src/ds41/cpu/mxfp4_expert_test.cpp - DS-C: tests of the MXFP4 CPU expert kernels.  No weights, no GPU, no ggml.
//
//   ds41_cpu_mxfp4_test [--quick] [--require-avx512] [--require-avx2] [--seed N]
//
// What is compared with what, and with which tolerance (the tolerances are the claims):
//
//   1. constants   kvalues, e8m0_half(e) = 2^(e-128) for ALL 256 e (denormals e = 0, 1 and 2^127 at e = 255 included).
//   2. quantiser   the library's (scalar, AVX2, AVX-512) against an independent natural-order one written here: EXACTLY
//                  equal (q, d), and the layout metadata (corr, sc4) recomputed from the natural values: exactly.
//   3. dot rows    the library scalar against an independent double-precision transcription of the definition: the integer
//                  part is exact on both sides, so only the FP32-vs-double order of the sum differs: 2e-7 of M (a rounding of the result
//                  to FP32 is 6e-8), where
//                  M = sum over all products of |scale * w * q| (the sum of absolute terms: the FP32 error of a sum scales
//                  with it, not with the size of the result).  AVX2 / AVX-512 against the same
//                  reference: 1e-6 of M (FP32 lane accumulation over up to 160 blocks; observed 1e-7).  Every code, all 256 exponents,
//                  the exponent edge cases 0 / 1 / 2 / 127 / 128 / 254 / 255 with activations scaled to keep the products
//                  finite, T = 1..8, ragged row counts, strided output, accumulate.
//   4. pipeline    expert_gate_up + expert_down against an FP64 transcription of CONTRACTS.md, in stages:
//                    a. gate/up on the quantised x (dequantised, in FP64): 1e-5 of the largest value;
//                    b. the quantised intermediate h against the FP64 h (which includes the routing weight): within half a
//                       quantisation step of its block (the definition of round-to-nearest), and its SCALE equal to
//                       max|h * w| / 127 - this is the test that fails if the routing weight is applied after W2 instead
//                       of before (the int8 quantisation is scale-invariant, so the final y alone cannot tell);
//                    c. y from the kernel's own quantised h against FP64 W2 . h_hat: 1e-5 of the largest value;
//                    d. end to end against FP64 with the original FP32 activations: every y element within the DETERMINISTIC
//                       bound that propagates the two int8 roundings (x -> g, u -> h -> y, worst-case sums of absolute
//                       values, no statistics); plus a sanity bound of 5 % relative RMS (typically 0.5-1.5 %, ~3 % in tiny
//                       heavy-tailed cases - the int8 step of a block is set by its largest element).
//   5. ISAs        AVX2 / AVX-512 pipeline against the scalar one: the FP32 sums differ in the last bits, so the rounding of
//                  an h element can flip by one step in rare elements (a few per 100,000).  The intermediate may differ by
//                  at most one step in at most 1 + n/2000 elements, and every y element by 3e-5 of the RMS plus, exactly,
//                  |W2[r,k]| * step for each flipped element k - so with no flips the agreement is 3e-5.
//   6. halves      the CPU-half layout against the in-place view of the GPU blob: BITWISE; half0 + half1 against the
//                  whole-expert pass: 3e-5 of the RMS (only the FP32 order of the down sum differs).
//   7. threads     ragged row-range splits on real threads with a barrier, and T tokens against T single-token calls:
//                  BITWISE equal to the single-thread run (per (row, token) the arithmetic is one fixed sequence).
//   8. accumulate  y is added to, not overwritten.
#include "strata/ds41/cpu/mxfp4_expert.hpp"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace strata::ds41;
using namespace strata::ds41::cpu;

namespace {

int g_checks = 0, g_fail = 0;
bool g_quick = false;
uint64_t g_seed = 20261001;

#define CHECK(cond, ...)                                                                   \
    do {                                                                                   \
        ++g_checks;                                                                        \
        if (!(cond)) {                                                                     \
            ++g_fail;                                                                      \
            std::printf("FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);                  \
            std::printf(__VA_ARGS__);                                                      \
            std::printf("\n");                                                             \
        }                                                                                  \
    } while (0)

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s) {}
    uint32_t u32() { return (uint32_t) g(); }
    int range(int lo, int hi) { return lo + (int) (g() % (uint64_t) (hi - lo + 1)); }   // inclusive
    float normal() { return std::normal_distribution<float>(0.f, 1.f)(g); }
    float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g); }
};

std::vector<Isa> simd_isas() {
    std::vector<Isa> v;
    if (isa_supported(Isa::kAvx2)) v.push_back(Isa::kAvx2);
    if (isa_supported(Isa::kAvx512)) v.push_back(Isa::kAvx512);
    return v;
}
std::vector<Isa> all_isas() {
    std::vector<Isa> v = {Isa::kScalar};
    for (Isa i : simd_isas()) v.push_back(i);
    return v;
}

// ---- random weights ------------------------------------------------------------------------------------------------
enum class EMode { kRealistic, kAll256, kFixed };

// `rows` rows of `nb` blocks.  qs bytes are uniform random (every code pair); row 0 starts with the bytes 0..255 so that
// every (low, high) code pair is present even in tiny shapes.
void fill_rows(Rng& rng, uint8_t* w, size_t rows, int nb, EMode mode, int efixed, int elo = 112, int ehi = 126) {
    int cyc = 0;
    for (size_t r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = w + (r * (size_t) nb + (size_t) b) * kBlockBytes;
            for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) rng.u32();
            if (r == 0)
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) ((b * 16 + j) & 255);
            switch (mode) {
                case EMode::kRealistic: blk[0] = (uint8_t) rng.range(elo, ehi); break;
                case EMode::kAll256: blk[0] = (uint8_t) (cyc++ & 255); break;
                case EMode::kFixed: blk[0] = (uint8_t) efixed; break;
            }
        }
}

// ---- independent references ----------------------------------------------------------------------------------------
// The natural-order quantiser of CONTRACTS.md with the rounding spelled out; no layout, no SIMD.
void ref_quantize(const float* x, int n, std::vector<int8_t>& q, std::vector<float>& d) {
    q.assign((size_t) n, 0);
    d.assign((size_t) n / 32, 0.f);
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
        d[b] = amax / 127.0f;
        const float id = amax != 0 ? 127.0f / amax : 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float t = x[b * 32 + j] * id;
            q[(size_t) b * 32 + j] = (int8_t) std::clamp((int) std::nearbyintf(t), -127, 127);   // ties to even (FE_TONEAREST)
        }
    }
}

double ref_e8m0(uint8_t e) { return std::ldexp(1.0, (int) e - 128); }

// Decode one row of `nb` blocks to doubles (exact: kvalues * 2^(e-128)).
void decode_row(const uint8_t* row, int nb, std::vector<double>& out) {
    out.resize((size_t) nb * 32);
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * kBlockBytes;
        const double s = ref_e8m0(blk[0]);
        for (int j = 0; j < 16; ++j) {
            out[(size_t) b * 32 + j] = kMxfp4Values[blk[1 + j] & 15] * s;
            out[(size_t) b * 32 + 16 + j] = kMxfp4Values[blk[1 + j] >> 4] * s;
        }
    }
}

struct DotRef {
    double v = 0, mag = 0;
};
// The definition: sum over blocks of (float(e8m0 * d_b)) * (exact integer dot), in double.
DotRef ref_dot_int(const uint8_t* row, int nb, const int8_t* q, const float* d) {
    DotRef r;
    for (int b = 0; b < nb; ++b) {
        const uint8_t* blk = row + (size_t) b * kBlockBytes;
        const float s = std::ldexp(1.0f, (int) blk[0] - 128) * d[b];
        long isum = 0, iabs = 0;
        for (int j = 0; j < 16; ++j) {
            const long a = (long) kMxfp4Values[blk[1 + j] & 15] * q[b * 32 + j], c = (long) kMxfp4Values[blk[1 + j] >> 4] * q[b * 32 + 16 + j];
            isum += a + c;
            iabs += std::labs(a) + std::labs(c);
        }
        r.v += (double) s * (double) isum;
        r.mag += std::fabs((double) s) * (double) iabs;     // the sum of |terms| at the level of single products
    }
    return r;
}

// ---- 1. constants ----------------------------------------------------------------------------------------------------
void test_constants() {
    std::printf("[1] constants\n");
    const int8_t expect[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int i = 0; i < 16; ++i) CHECK(kMxfp4Values[i] == expect[i], "kvalues[%d]", i);
    for (int e = 0; e < 256; ++e) {
        const float f = e8m0_half((uint8_t) e);
        CHECK((double) f == std::ldexp(1.0, e - 128), "e8m0_half(%d) = %g", e, (double) f);
    }
    CHECK(e8m0_half(0) > 0 && e8m0_half(0) < 1e-38f, "e = 0 is the denormal 2^-128");
    CHECK(e8m0_half(255) == 1.7014118e38f, "e = 255 is 2^127, not NaN");
    static_assert(kHalfBytes == 9400320 && kBlobBytes == 18800640, "geometry");
    static_assert(kHalfDownRowBlocks * kBlockBytes * kHidden == kHalfDownBytes, "half down rows are contiguous");
    CHECK(act_index(0) == 0 && act_index(15) == 15 && act_index(16) == 64 && act_index(31) == 79 && act_index(32) == 16 &&
              act_index(128) == 128,
          "act_index");
    // act_index is a permutation of [0, 5120)
    std::vector<char> seen(kHidden, 0);
    for (int i = 0; i < kHidden; ++i) seen[act_index(i)]++;
    bool perm = true;
    for (char c : seen) perm = perm && c == 1;
    CHECK(perm, "act_index is a permutation");
    for (Isa isa : all_isas()) CHECK(isa_supported(isa), "isa_supported");
    std::printf("    scalar: yes   avx2: %s   avx512: %s\n", isa_supported(Isa::kAvx2) ? "yes" : "NO (not tested)",
                isa_supported(Isa::kAvx512) ? "yes" : "NO (not tested)");
}

// ---- 2. the quantiser --------------------------------------------------------------------------------------------------
void check_act_equals_reference(const ActQ& a, const float* x, int n, const char* what, Isa isa) {
    std::vector<int8_t> rq, got((size_t) n);
    std::vector<float> rd;
    ref_quantize(x, n, rq, rd);
    act_unpack(a, n, got.data());
    bool ok = true;
    for (int i = 0; i < n && ok; ++i)
        if (rq[i] != got[i]) {
            ok = false;
            CHECK(false, "%s/%s: q[%d] ref %d got %d (x = %g)", isa_name(isa), what, i, rq[i], got[i], (double) x[i]);
        }
    for (int b = 0; b < n / 32 && ok; ++b)
        if (std::memcmp(&rd[b], &a.scale[b], 4) != 0) {
            ok = false;
            CHECK(false, "%s/%s: scale[%d] ref %g got %g", isa_name(isa), what, b, (double) rd[b], (double) a.scale[b]);
        }
    // the layout metadata, recomputed from the natural values
    for (int b = 0; b < n / 32 && ok; ++b) {
        const int g = b / 4, k = b % 4;
        for (int l = 0; l < 4; ++l) {
            int s = 0;
            for (int i = 0; i < 4; ++i) s += rq[b * 32 + 4 * l + i] + rq[b * 32 + 16 + 4 * l + i];
            if (a.corr[g * 16 + k * 4 + l] != -12 * s || std::memcmp(&a.sc4[g * 16 + k * 4 + l], &rd[b], 4) != 0) {
                ok = false;
                CHECK(false, "%s/%s: corr/sc4 block %d lane %d", isa_name(isa), what, b, l);
            }
        }
    }
    CHECK(ok, "%s/%s", isa_name(isa), what);
}

void test_quantiser(Rng& rng) {
    std::printf("[2] activation quantiser (exact against the independent natural-order one)\n");
    const int n = kHidden;
    std::vector<float> x((size_t) n);
    auto run_case = [&](const char* what) {
        for (Isa isa : all_isas()) {
            static ActQ a;
            std::memset(&a, 0xAB, sizeof a);   // nothing may be left to chance
            quantize_act(x.data(), n, a, isa);
            check_act_equals_reference(a, x.data(), n, what, isa);
        }
    };
    for (int rep = 0; rep < (g_quick ? 3 : 12); ++rep) {
        for (auto& v : x) v = rng.normal() * std::ldexp(1.0f, rng.range(-3, 3));
        run_case("normal");
    }
    // every block shape a router could produce: zeros, a lone spike, ties, huge, tiny, mixed sign
    for (int b = 0; b < n / 32; ++b) {
        float* xb = &x[(size_t) b * 32];
        for (int j = 0; j < 32; ++j) xb[j] = 0;
        switch (b % 8) {
            case 0: break;                                                       // all zero: d = 0, q = 0
            case 1: xb[rng.range(0, 31)] = rng.normal() * 100; break;           // a lone spike
            case 2:                                                              // ties at x.5 (amax = 127 -> id = 1)
                xb[0] = 127.0f;
                for (int j = 1; j < 32; ++j) xb[j] = (float) (j - 16) + 0.5f;
                break;
            case 3: for (int j = 0; j < 32; ++j) xb[j] = rng.normal() * 1e-30f; break;
            case 4: for (int j = 0; j < 32; ++j) xb[j] = rng.normal() * 1e30f; break;
            case 5: for (int j = 0; j < 32; ++j) xb[j] = (j & 1) ? -3.0f : 3.0f; break;
            case 6: for (int j = 0; j < 32; ++j) xb[j] = rng.normal() * std::ldexp(1.0f, -100); break;
            default: for (int j = 0; j < 32; ++j) xb[j] = -127.0f + (float) j * 0.4f; break;
        }
    }
    run_case("edge blocks");
    // act_from_q8 round trip: rebuilding the metadata from natural values gives the same ActQ as the quantiser
    {
        static ActQ a, b;
        for (auto& v : x) v = rng.normal();
        quantize_act(x.data(), n, a, Isa::kScalar);
        std::vector<int8_t> q((size_t) n);
        act_unpack(a, n, q.data());
        std::memset(&b, 0, sizeof b);
        act_from_q8(q.data(), a.scale, n, b);
        CHECK(std::memcmp(a.q, b.q, sizeof a.q) == 0 && std::memcmp(a.corr, b.corr, sizeof a.corr) == 0 &&
                  std::memcmp(a.sc4, b.sc4, sizeof a.sc4) == 0 && std::memcmp(a.scale, b.scale, sizeof a.scale) == 0,
              "act_from_q8 reproduces quantize_act");
    }
}

// ---- 3. dot rows -------------------------------------------------------------------------------------------------------
struct DotCase {
    int nb, nrows, T;
    EMode mode;
    int efixed;
    float xscale_log2;     // activations ~ N(0,1) * 2^xscale_log2
};

// Runs one case on every ISA against the independent reference; returns the largest |err| / M seen.
void run_dot_case(Rng& rng, const DotCase& c, bool accumulate, size_t extra_stride, double* worst_scalar,
                  double* worst_simd) {
    const size_t row_bytes = (size_t) c.nb * kBlockBytes;
    std::vector<uint8_t> w(row_bytes * (size_t) c.nrows + 64);
    fill_rows(rng, w.data(), (size_t) c.nrows, c.nb, c.mode, c.efixed);
    const int n = c.nb * 32;
    std::vector<float> x((size_t) n * c.T);
    for (auto& v : x) v = rng.normal() * std::ldexp(1.0f, (int) c.xscale_log2);
    static ActQ xq[kMaxTokens];
    quantize_acts(x.data(), n, c.T, xq, Isa::kScalar);
    // reference per (t, r)
    std::vector<DotRef> ref((size_t) c.T * c.nrows);
    for (int t = 0; t < c.T; ++t) {
        std::vector<int8_t> q((size_t) n);
        act_unpack(xq[t], n, q.data());
        for (int r = 0; r < c.nrows; ++r) ref[(size_t) t * c.nrows + r] = ref_dot_int(&w[(size_t) r * row_bytes], c.nb, q.data(), xq[t].scale);
    }
    const size_t os = (size_t) c.nrows + extra_stride;
    for (Isa isa : all_isas()) {
        std::vector<float> out((size_t) c.T * os, 0.f), seed((size_t) c.T * os, 0.f);
        for (size_t i = 0; i < out.size(); ++i) seed[i] = out[i] = accumulate ? rng.normal() : -7.5f;
        mxfp4_dot_rows(isa, w.data(), row_bytes, c.nb, c.nrows, xq, c.T, out.data(), os, accumulate);
        const double tol = isa == Isa::kScalar ? 2e-7 : 1e-6;
        for (int t = 0; t < c.T; ++t)
            for (int r = 0; r < c.nrows; ++r) {
                const DotRef& d = ref[(size_t) t * c.nrows + r];
                const double base = accumulate ? (double) seed[(size_t) t * os + r] : 0.0;
                const double got = out[(size_t) t * os + r];
                const double err = std::fabs(got - (base + d.v));
                // allow the FP32 rounding of `base + v` (accumulate) and of the final cast
                const double allow = tol * d.mag + std::fabs(base) * 1.2e-7 + std::fabs(d.v) * 1.2e-7 + 1e-37;
                const bool ok = std::isfinite(got) && err <= allow;
                if (!ok)
                    CHECK(false, "%s dot nb=%d nrows=%d T=%d mode=%d e=%d: (t=%d r=%d) got %.9g want %.9g (mag %.3g, err/mag %.3g)",
                          isa_name(isa), c.nb, c.nrows, c.T, (int) c.mode, c.efixed, t, r, got, base + d.v, d.mag,
                          d.mag > 0 ? err / d.mag : 0.0);
                else ++g_checks;
                double* worst = isa == Isa::kScalar ? worst_scalar : worst_simd;
                if (!accumulate && d.mag > 1e-25) *worst = std::max(*worst, err / d.mag);   // (below ~1e-25 the sums are denormal-level)
                if (extra_stride)   // the padding between rows of the output must be untouched
                    for (size_t p = c.nrows; p < os; ++p)
                        if (out[(size_t) t * os + p] != seed[(size_t) t * os + p]) CHECK(false, "output padding written");
            }
    }
}

void test_dot_rows(Rng& rng) {
    std::printf("[3] dot rows against the double-precision definition (scalar 2e-7, SIMD 1e-6 of the sum of |terms|)\n");
    double worst_scalar = 0, worst_simd = 0;
    const int nbs[] = {4, 8, 36, 72, 160};
    const int rowss[] = {1, 2, 3, 4, 5, 7, 8, 9, 31, 33};
    for (int nb : nbs)
        for (int T = 1; T <= kMaxTokens; ++T)
            for (int rows : rowss) {
                if (g_quick && ((nb + T + rows) % 3) != 0) continue;
                run_dot_case(rng, {nb, rows, T, EMode::kRealistic, 0, 0}, false, 0, &worst_scalar, &worst_simd);
            }
    // every exponent in one matrix (activations small enough that nothing overflows), strided + accumulating output
    for (int T : {1, 2, 5, 8}) {
        run_dot_case(rng, {36, 9, T, EMode::kAll256, 0, -40}, false, 3, &worst_scalar, &worst_simd);
        run_dot_case(rng, {72, 8, T, EMode::kAll256, 0, -40}, true, 5, &worst_scalar, &worst_simd);
        run_dot_case(rng, {160, 5, T, EMode::kRealistic, 0, 0}, true, 0, &worst_scalar, &worst_simd);
    }
    // the exponent edges: e = 0, 1 (denormal scales), 2, 127, 128, 254, 255 (2^127); activations scaled so the products stay
    // finite and representable
    for (int e : {0, 1, 2, 127, 128, 253, 254, 255}) {
        const int xs = e < 8 ? 100 : (e > 248 ? -100 : 0);
        for (int T : {1, 3, 8}) run_dot_case(rng, {36, 6, T, EMode::kFixed, e, (float) xs}, false, 0, &worst_scalar, &worst_simd);
        // mixed: the edge exponent and ordinary ones in the same row
    }
    std::printf("    worst |err|/M: scalar %.2e   SIMD %.2e\n", worst_scalar, worst_simd);
    CHECK(worst_scalar < 2e-7 && worst_simd < 1e-6, "dot tolerance");

    // e = 255 and e = 0 must give exactly the values the definition gives (not zero, not NaN): a one-block row
    {
        std::vector<uint8_t> w(4 * kBlockBytes, 0);
        for (int b = 0; b < 4; ++b) {
            w[b * 17] = (uint8_t) (b == 0 ? 255 : b == 1 ? 0 : b == 2 ? 1 : 128);
            for (int j = 0; j < 16; ++j) w[b * 17 + 1 + j] = 0x77;    // codes 7 / 7: +12 for all 32 values
        }
        std::vector<float> x(128);
        for (int i = 0; i < 128; ++i) x[i] = (i < 32) ? std::ldexp(1.0f, -100) : (i < 96 ? std::ldexp(1.0f, 100) : 1.0f);
        static ActQ xq[1];
        quantize_act(x.data(), 128, xq[0], Isa::kScalar);   // d = 2^-100/127, 2^100/127 (x2), 1/127
        for (Isa isa : all_isas()) {
            float out = 0;
            mxfp4_dot_rows(isa, w.data(), 4 * kBlockBytes, 4, 1, xq, 1, &out, 1, false);
            // block b: 2^(e-128) * d_b * (32 values * 12 * 127)
            const double want = ref_e8m0(255) * (double) xq[0].scale[0] * 32 * 12 * 127 +
                                ref_e8m0(0) * (double) xq[0].scale[1] * 32 * 12 * 127 +
                                ref_e8m0(1) * (double) xq[0].scale[2] * 32 * 12 * 127 +
                                ref_e8m0(128) * (double) xq[0].scale[3] * 32 * 12 * 127;
            CHECK(std::isfinite(out) && std::fabs(out - want) <= 1e-5 * std::fabs(want), "%s e-edge sum got %.9g want %.9g",
                  isa_name(isa), (double) out, want);
        }
    }
}

// ---- 4..7. the pipeline ------------------------------------------------------------------------------------------------
struct Expert {              // owns a blob-shaped (or tiny) expert and its views
    std::vector<uint8_t> mem;
    ExpertView v;
};

// A tiny expert in the GPU-blob arrangement: [gate ff rows][up ff rows][down hidden rows of ff/32 blocks].
Expert make_tiny_expert(Rng& rng, int hidden, int ff, EMode mode, int elo, int ehi) {
    Expert e;
    const int hb = hidden / 32, fb = ff / 32;
    const size_t gate = (size_t) ff * hb * kBlockBytes, down = (size_t) hidden * fb * kBlockBytes;
    e.mem.assign(2 * gate + down + 64, 0);
    fill_rows(rng, e.mem.data(), (size_t) ff, hb, mode, 0, elo, ehi);
    fill_rows(rng, e.mem.data() + gate, (size_t) ff, hb, mode, 0, elo, ehi);
    fill_rows(rng, e.mem.data() + 2 * gate, (size_t) hidden, fb, mode, 0, elo, ehi);
    e.v.gate = e.mem.data();
    e.v.up = e.mem.data() + gate;
    e.v.down = e.mem.data() + 2 * gate;
    e.v.down_row_stride = (size_t) fb * kBlockBytes;
    e.v.hidden = hidden;
    e.v.ff = ff;
    return e;
}

// FP64 transcription of CONTRACTS.md for one view.
struct RefOut {
    std::vector<double> g, u, h, y;   // [T][ff], [T][ff], [T][ff], [T][hidden]
};
RefOut ref_expert(const ExpertView& v, const float* x /*[T][hidden]*/, int T, const float* w) {
    RefOut o;
    o.g.assign((size_t) T * v.ff, 0);
    o.u = o.g;
    o.h = o.g;
    o.y.assign((size_t) T * v.hidden, 0);
    std::vector<double> row;
    const int hb = v.hidden / 32;
    for (int r = 0; r < v.ff; ++r) {
        decode_row(v.gate + (size_t) r * hb * kBlockBytes, hb, row);
        for (int t = 0; t < T; ++t) {
            double s = 0;
            for (int k = 0; k < v.hidden; ++k) s += row[k] * (double) x[(size_t) t * v.hidden + k];
            o.g[(size_t) t * v.ff + r] = s;
        }
        decode_row(v.up + (size_t) r * hb * kBlockBytes, hb, row);
        for (int t = 0; t < T; ++t) {
            double s = 0;
            for (int k = 0; k < v.hidden; ++k) s += row[k] * (double) x[(size_t) t * v.hidden + k];
            o.u[(size_t) t * v.ff + r] = s;
        }
    }
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < v.ff; ++r) {
            double g = std::min(o.g[(size_t) t * v.ff + r], 10.0);
            double u = std::clamp(o.u[(size_t) t * v.ff + r], -10.0, 10.0);
            o.h[(size_t) t * v.ff + r] = g / (1.0 + std::exp(-g)) * u * (double) w[t];       // the routing weight BEFORE W2
        }
    const int fb = v.ff / 32;
    for (int r = 0; r < v.hidden; ++r) {
        decode_row(v.down + (size_t) r * v.down_row_stride, fb, row);
        for (int t = 0; t < T; ++t) {
            double s = 0;
            for (int k = 0; k < v.ff; ++k) s += row[k] * o.h[(size_t) t * v.ff + k];
            o.y[(size_t) t * v.hidden + r] = s;
        }
    }
    return o;
}

// y_ref2 = W2 . hhat for the kernel's own quantised intermediate (stage c)
std::vector<double> ref_down(const ExpertView& v, const std::vector<double>& hhat, int T) {
    std::vector<double> y((size_t) T * v.hidden, 0), row;
    const int fb = v.ff / 32;
    for (int r = 0; r < v.hidden; ++r) {
        decode_row(v.down + (size_t) r * v.down_row_stride, fb, row);
        for (int t = 0; t < T; ++t) {
            double s = 0;
            for (int k = 0; k < v.ff; ++k) s += row[k] * hhat[(size_t) t * v.ff + k];
            y[(size_t) t * v.hidden + r] = s;
        }
    }
    return y;
}

double maxabs(const std::vector<double>& v) {
    double m = 0;
    for (double d : v) m = std::max(m, std::fabs(d));
    return m;
}
double rms(const std::vector<double>& v) {
    double s = 0;
    for (double d : v) s += d * d;
    return std::sqrt(s / std::max<size_t>(1, v.size()));
}

// out[t * hidden + r] = sum_k |W2[r, k]| * E[t][k]  (E: [T][ff]); used for the deterministic error bounds
std::vector<double> abs_down(const ExpertView& v, const std::vector<double>& E, int T) {
    std::vector<double> out((size_t) T * v.hidden, 0), row;
    const int fb = v.ff / 32;
    for (int r = 0; r < v.hidden; ++r) {
        decode_row(v.down + (size_t) r * v.down_row_stride, fb, row);
        for (int t = 0; t < T; ++t) {
            double s = 0;
            for (int k = 0; k < v.ff; ++k) s += std::fabs(row[k]) * E[(size_t) t * v.ff + k];
            out[(size_t) t * v.hidden + r] = s;
        }
    }
    return out;
}

// The deterministic bound on |y_kernel - y_FP64(fp32 activations)| per element.  Two sources, both exact in the sense that
// nothing here is statistical:
//   * the quantised x moves g and u by at most eg = sum_j |W1[k,j]| dx(j) / 2 (resp. eu); that moves h = silu(g) clamp(u) w by
//     at most |w| (1.1 eg |clamp(u_hat)| + |silu(g)| eu)  (silu' is within [-0.1, 1.1], the clamps are 1-Lipschitz);
//   * the int8 rounding of h moves each element by at most half its block's step.
// y moves by sum_k |W2[r,k]| times the sum of the two.
std::vector<double> ref_error_bound(const ExpertView& v, const ActQ* xq, int T, const float* w, const RefOut& ref,
                                    const ExpertScratch& s) {
    std::vector<double> E((size_t) T * v.ff, 0), row;
    const int hb = v.hidden / 32;
    std::vector<double> eg((size_t) T * v.ff), eu((size_t) T * v.ff);
    for (int which = 0; which < 2; ++which)
        for (int k = 0; k < v.ff; ++k) {
            decode_row((which ? v.up : v.gate) + (size_t) k * hb * kBlockBytes, hb, row);
            for (int t = 0; t < T; ++t) {
                double e = 0;
                for (int b = 0; b < hb; ++b) {
                    double sa = 0;
                    for (int j = 0; j < 32; ++j) sa += std::fabs(row[(size_t) b * 32 + j]);
                    e += sa * (double) xq[t].scale[b] * 0.5;
                }
                (which ? eu : eg)[(size_t) t * v.ff + k] = e;
            }
        }
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < v.ff; ++k) {
            const size_t i = (size_t) t * v.ff + k;
            const double g = std::min(ref.g[i], 10.0), u = ref.u[i];
            const double silu = g / (1.0 + std::exp(-g));
            const double uc = std::min(10.0, std::fabs(u) + eu[i]);
            const double dh = std::fabs((double) w[t]) * (1.1 * eg[i] * uc + std::fabs(silu) * eu[i]);
            E[i] = dh + 0.5 * (double) s.h[t].scale[k / 32];
        }
    return abs_down(v, E, T);
}

struct Stats {
    double worst_stage_c = 0, worst_e2e = 0, worst_bound_use = 0, worst_isa_y = 0;
    int flips = 0, elements = 0;
};

// One expert instance through every ISA; checks stages a-d and the ISA agreement.
void run_pipeline_case(Rng& rng, const ExpertView& v, int T, const std::vector<float>& route_w, double xscale, Stats& st,
                       const char* label) {
    std::vector<float> x((size_t) T * v.hidden);
    for (auto& e : x) e = rng.normal() * (float) xscale;
    static ExpertScratch scratch[3];
    static ActQ xq[kMaxTokens];
    const RefOut ref = ref_expert(v, x.data(), T, route_w.data());      // FP32 activations, FP64 math: the contract
    std::vector<std::vector<double>> hhat_by_isa;
    std::vector<std::vector<float>> y_by_isa;
    std::vector<std::vector<int8_t>> hq_by_isa;
    std::vector<Isa> isas = all_isas();
    for (size_t ii = 0; ii < isas.size(); ++ii) {
        const Isa isa = isas[ii];
        quantize_acts(x.data(), v.hidden, T, xq, isa);
        std::vector<float> y((size_t) T * v.hidden, 0.f);
        ExpertScratch& s = scratch[ii];
        std::memset(&s, 0, sizeof s);
        expert_gate_up(isa, v, xq, T, route_w.data(), s, 0, v.chunks());
        // ---- stage a/b: the intermediate
        std::vector<double> xhat((size_t) T * v.hidden);
        {
            std::vector<float> xd((size_t) v.hidden);
            for (int t = 0; t < T; ++t) {
                act_dequant(xq[t], v.hidden, xd.data());
                for (int k = 0; k < v.hidden; ++k) xhat[(size_t) t * v.hidden + k] = xd[k];
            }
        }
        std::vector<float> xhatf(xhat.begin(), xhat.end());
        const RefOut refq = ref_expert(v, xhatf.data(), T, route_w.data());   // FP64 on the quantised x: what the kernel sees
        std::vector<double> hhat((size_t) T * v.ff);
        std::vector<int8_t> hq((size_t) T * v.ff);
        for (int t = 0; t < T; ++t) {
            act_unpack(s.h[t], v.ff, &hq[(size_t) t * v.ff]);
            for (int k = 0; k < v.ff; ++k) hhat[(size_t) t * v.ff + k] = (double) hq[(size_t) t * v.ff + k] * (double) s.h[t].scale[k / 32];
            for (int b = 0; b < v.ff / 32; ++b) {
                double amax = 0;
                for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(refq.h[(size_t) t * v.ff + b * 32 + j]));
                const double d = (double) s.h[t].scale[b];
                // the SCALE carries the routing weight: d = max|h * w| / 127 (relative 2e-4: g, u are FP32 sums)
                CHECK(std::fabs(d - amax / 127.0) <= 2e-4 * amax / 127.0 + 1e-30,
                      "%s/%s: h scale t=%d b=%d: kernel %.9g, max|h w|/127 = %.9g (routing weight not applied before W2?)",
                      label, isa_name(isa), t, b, d, amax / 127.0);
                for (int j = 0; j < 32; ++j) {
                    const double hv = refq.h[(size_t) t * v.ff + b * 32 + j], hk = hhat[(size_t) t * v.ff + b * 32 + j];
                    // round-to-nearest: within half a step (plus the FP32 noise of g, u)
                    if (std::fabs(hk - hv) > 0.5 * d * (1 + 1e-3) + 1e-3 * amax / 127.0 * 0.01 + 1e-30)
                        CHECK(false, "%s/%s: h t=%d [%d]: kernel %.9g ref %.9g step %.3g", label, isa_name(isa), t, b * 32 + j, hk, hv, d);
                }
            }
        }
        hhat_by_isa.push_back(hhat);
        hq_by_isa.push_back(hq);
        // ---- stage a (gate/up values themselves): the dot rows on x_hat against FP64, checked through h above; the
        // direct check is done in [3].  Stage c: down on the kernel's own h_hat.
        expert_down(isa, v, s, T, y.data(), 0, v.hidden);
        const std::vector<double> yc = ref_down(v, hhat, T);
        const double mc = maxabs(yc);
        double werr = 0;
        for (size_t i = 0; i < y.size(); ++i) werr = std::max(werr, std::fabs((double) y[i] - yc[i]));
        st.worst_stage_c = std::max(st.worst_stage_c, mc > 0 ? werr / mc : 0.0);
        CHECK(werr <= 1e-5 * mc + 1e-30, "%s/%s: stage c (down on the kernel's h): max err %.3g of max %.3g", label, isa_name(isa), werr, mc);
        // ---- stage d: end to end against the contract with FP32 activations
        double num = 0, den = 0;
        for (size_t i = 0; i < y.size(); ++i) {
            num += ((double) y[i] - ref.y[i]) * ((double) y[i] - ref.y[i]);
            den += ref.y[i] * ref.y[i];
        }
        const double rel = den > 0 ? std::sqrt(num / den) : 0;
        st.worst_e2e = std::max(st.worst_e2e, rel);
        // an empirical sanity bound (typically 0.5-1.5 %, up to ~3 % for tiny heavy-tailed cases) ...
        CHECK(rel < 0.05, "%s/%s: end-to-end relative RMS error %.4f (sanity bound 5%%)", label, isa_name(isa), rel);
        // ... and the deterministic one: every element within the propagated int8 rounding error
        const std::vector<double> bound = ref_error_bound(v, xq, T, route_w.data(), ref, s);
        const double ry = rms(ref.y);
        bool inside = true;
        for (size_t i = 0; i < y.size(); ++i) {
            const double err = std::fabs((double) y[i] - ref.y[i]), allow = 1.01 * bound[i] + 1e-5 * ry;
            if (err > allow) {
                inside = false;
                CHECK(false, "%s/%s: y[%zu] = %.9g, FP64 %.9g: error %.3g exceeds the int8 bound %.3g", label, isa_name(isa), i, (double) y[i], ref.y[i], err, allow);
                break;
            }
            if (bound[i] > 1e-30 * ry) st.worst_bound_use = std::max(st.worst_bound_use, err / bound[i]);
        }
        CHECK(inside, "%s/%s: deterministic error bound", label, isa_name(isa));
        y_by_isa.push_back(y);
    }
    // ---- 5. the ISAs against the scalar pipeline
    for (size_t ii = 1; ii < isas.size(); ++ii) {
        int flips = 0, bad = 0;
        for (size_t i = 0; i < hq_by_isa[0].size(); ++i) {
            const int d = std::abs((int) hq_by_isa[0][i] - (int) hq_by_isa[ii][i]);
            if (d) ++flips;
            if (d > 1) ++bad;
        }
        st.flips += flips;
        st.elements += (int) hq_by_isa[0].size();
        CHECK(bad == 0, "%s/%s: intermediate differs from the scalar one by more than one step in %d elements", label, isa_name(isas[ii]), bad);
        CHECK(flips <= 1 + (int) hq_by_isa[0].size() / 2000, "%s/%s: %d of %zu intermediate elements flipped", label, isa_name(isas[ii]), flips, hq_by_isa[0].size());
        const double r0 = rms(std::vector<double>(y_by_isa[0].begin(), y_by_isa[0].end()));
        // y may differ by the FP32 order of the sums (3e-5 of the RMS) plus, per flipped h element k (+-1 step of its
        // block), |W2[r, k]| * step: exactly the bound of a rounding flip
        std::vector<double> flipE((size_t) T * v.ff, 0);
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < v.ff; ++k)
                if (hq_by_isa[0][(size_t) t * v.ff + k] != hq_by_isa[ii][(size_t) t * v.ff + k])
                    flipE[(size_t) t * v.ff + k] = 1.001 * (double) std::max(scratch[0].h[t].scale[k / 32], scratch[ii].h[t].scale[k / 32]);
        const std::vector<double> fb = flips ? abs_down(v, flipE, T) : std::vector<double>((size_t) T * v.hidden, 0.0);
        double worst_ratio = 0;
        bool within = true;
        for (size_t i = 0; i < y_by_isa[0].size() && within; ++i) {
            const double err = std::fabs((double) y_by_isa[0][i] - (double) y_by_isa[ii][i]), allow = 3e-5 * r0 + fb[i];
            worst_ratio = std::max(worst_ratio, r0 > 0 ? err / r0 : 0.0);
            if (err > allow + 1e-30) {
                within = false;
                CHECK(false, "%s/%s: y[%zu] differs from scalar by %.3g (allowed %.3g, rms %.3g, flips %d)", label, isa_name(isas[ii]), i, err, allow, r0, flips);
            }
        }
        st.worst_isa_y = std::max(st.worst_isa_y, worst_ratio);
        CHECK(within, "%s/%s: ISA agreement", label, isa_name(isas[ii]));
    }
}

void test_pipeline_tiny(Rng& rng) {
    std::printf("[4,5] pipeline on tiny shapes: stages against FP64, ISAs against scalar\n");
    Stats st;
    const int hiddens[] = {128, 256, 384, 512, 1024};
    const int ffs[] = {128, 256, 384, 640};
    const int cases = g_quick ? 24 : 120;
    for (int c = 0; c < cases; ++c) {
        const int hidden = hiddens[rng.range(0, 4)], ff = ffs[rng.range(0, 3)];
        const int T = rng.range(1, kMaxTokens);
        // e range chosen per case so that g, u are sometimes small and sometimes far beyond the +-10 clamps
        const int elo = rng.range(112, 124);
        Expert e = make_tiny_expert(rng, hidden, ff, EMode::kRealistic, elo, elo + 3);
        std::vector<float> w((size_t) T);
        for (auto& v : w) v = rng.range(0, 3) == 0 ? rng.uniform(-2.0f, 2.0f) : rng.uniform(0.05f, 2.5f);
        run_pipeline_case(rng, e.v, T, w, std::ldexp(1.0, rng.range(-1, 2)), st, "tiny");
    }
    std::printf("    worst: stage c %.2e of max, end-to-end rel RMS %.4f (deterministic bound used: at most %.1f%%), SIMD-vs-scalar y/rms %.2e; "
                "intermediate flips %d of %d\n", st.worst_stage_c, st.worst_e2e, 100 * st.worst_bound_use, st.worst_isa_y, st.flips, st.elements);
}

// The routing weight, on its own: a non-trivial weight must change the quantisation scale by exactly |w|.
void test_routing_weight(Rng& rng) {
    std::printf("[4b] routing weight applied before W2\n");
    const int hidden = 512, ff = 256, T = 3;
    Expert e = make_tiny_expert(rng, hidden, ff, EMode::kRealistic, 118, 121);
    std::vector<float> x((size_t) T * hidden), w1(T, 1.0f), w2 = {1.37f, -0.61f, 0.0123f};
    for (auto& v : x) v = rng.normal();
    static ActQ xq[kMaxTokens];
    static ExpertScratch s1, s2;
    for (Isa isa : all_isas()) {
        quantize_acts(x.data(), hidden, T, xq, isa);
        expert_gate_up(isa, e.v, xq, T, w1.data(), s1, 0, e.v.chunks());
        expert_gate_up(isa, e.v, xq, T, w2.data(), s2, 0, e.v.chunks());
        for (int t = 0; t < T; ++t)
            for (int b = 0; b < ff / 32; ++b) {
                const double ratio = (double) s2.h[t].scale[b] / std::max(1e-30, (double) s1.h[t].scale[b]);
                CHECK(std::fabs(ratio - std::fabs((double) w2[t])) <= 2e-5 * std::fabs(w2[t]),
                      "%s: scale ratio %.9g for w = %g (t=%d b=%d)", isa_name(isa), ratio, (double) w2[t], t, b);
            }
        // and the weight is not applied a second time after W2: y(w) = y(1) * w up to the int8 rounding of h (|w| scaling
        // preserves the codes except for ties; a negative weight flips them)
        std::vector<float> y1((size_t) T * hidden, 0.f), y2((size_t) T * hidden, 0.f);
        expert_down(isa, e.v, s1, T, y1.data(), 0, hidden);
        expert_down(isa, e.v, s2, T, y2.data(), 0, hidden);
        for (int t = 0; t < T; ++t) {
            double num = 0, den = 0;
            for (int r = 0; r < hidden; ++r) {
                const double a = (double) y2[(size_t) t * hidden + r], b = (double) y1[(size_t) t * hidden + r] * w2[t];
                num += (a - b) * (a - b);
                den += b * b;
            }
            CHECK(den > 0 && std::sqrt(num / den) < 5e-3, "%s: y(w) vs w * y(1): %.3g (t=%d)", isa_name(isa), std::sqrt(num / den), t);
        }
    }
}

// Real geometry: one expert blob, a CPU half, T tokens.
struct BigExpert {
    std::vector<uint8_t> blob, half0, half1;
    BigExpert(Rng& rng, int elo, int ehi) : blob(kBlobBytes), half0(kHalfBytes), half1(kHalfBytes) {
        fill_rows(rng, blob.data(), (size_t) 2 * kFF, kGateRowBlocks, EMode::kRealistic, 0, elo, ehi);
        fill_rows(rng, blob.data() + kBlobDown, (size_t) kHidden, kDownRowBlocks, EMode::kRealistic, 0, elo, ehi);
        pack_cpu_half(blob.data(), 0, half0.data());
        pack_cpu_half(blob.data(), 1, half1.data());
    }
};

void test_pipeline_real(Rng& rng) {
    std::printf("[4,5] pipeline on the real geometry (one CPU half, 5120 x 1152)\n");
    BigExpert be(rng, 117, 122);
    Stats st;
    for (int T : g_quick ? std::vector<int>{1, 4} : std::vector<int>{1, 3, 8}) {
        std::vector<float> w((size_t) T);
        for (auto& v : w) v = rng.uniform(0.2f, 2.0f);
        run_pipeline_case(rng, view_cpu_half(be.half0.data()), T, w, 1.0, st, "real");
    }
    std::printf("    worst: stage c %.2e of max, end-to-end rel RMS %.4f (deterministic bound used: at most %.1f%%), SIMD-vs-scalar y/rms %.2e; "
                "intermediate flips %d of %d\n", st.worst_stage_c, st.worst_e2e, 100 * st.worst_bound_use, st.worst_isa_y, st.flips, st.elements);
}

// ---- 6. halves ----------------------------------------------------------------------------------------------------------
void test_halves(Rng& rng) {
    std::printf("[6] CPU-half layout vs the in-place blob view (bitwise); half0 + half1 vs the whole expert\n");
    BigExpert be(rng, 117, 122);
    // the CPU half is a pure re-layout of the blob
    {
        const ExpertView a = view_cpu_half(be.half1.data()), b = view_blob_half(be.blob.data(), 1);
        CHECK(a.ff == b.ff && a.hidden == b.hidden && a.down_row_stride == 612 && b.down_row_stride == 1224, "view strides");
        CHECK(std::memcmp(a.gate, b.gate, kHalfGateBytes) == 0 && std::memcmp(a.up, b.up, kHalfGateBytes) == 0, "half gate/up bytes");
        CHECK(std::memcmp(be.half0.data() + kHalfDown + 612, be.blob.data() + kBlobDown + 1224, 612) == 0, "half 0 down row 1");
        CHECK(std::memcmp(be.half1.data() + kHalfDown + 612, be.blob.data() + kBlobDown + 1224 + 612, 612) == 0, "half 1 down row 1");
    }
    for (int T : {1, 4}) {
        std::vector<float> x((size_t) T * kHidden), w((size_t) T);
        for (auto& v : x) v = rng.normal();
        for (auto& v : w) v = rng.uniform(0.3f, 1.8f);
        static ActQ xq[kMaxTokens];
        static ExpertScratch sa, sb, sc;
        for (Isa isa : all_isas()) {
            quantize_acts(x.data(), kHidden, T, xq, isa);
            std::vector<float> ya((size_t) T * kHidden, 0.f), yb = ya, yfull = ya;
            for (int h = 0; h < 2; ++h) {
                expert_run(isa, view_cpu_half(h ? be.half1.data() : be.half0.data()), xq, T, w.data(), sa, ya.data());
                expert_run(isa, view_blob_half(be.blob.data(), h), xq, T, w.data(), sb, yb.data());
            }
            CHECK(std::memcmp(ya.data(), yb.data(), ya.size() * 4) == 0, "%s T=%d: CPU halves and blob halves are not bitwise equal", isa_name(isa), T);
            expert_run_blob(isa, be.blob.data(), xq, T, w.data(), sc, yfull.data());
            std::vector<double> yd(yfull.begin(), yfull.end());
            const double r = rms(yd);
            double werr = 0;
            for (size_t i = 0; i < ya.size(); ++i) werr = std::max(werr, std::fabs((double) ya[i] - (double) yfull[i]));
            std::printf("    %-6s T=%d: |half0 + half1 - full| max %.3e (rms of y %.3e, ratio %.2e)\n", isa_name(isa), T, werr, r, werr / r);
            CHECK(werr <= 3e-5 * r, "%s T=%d: half0 + half1 vs full: %.3g (rms %.3g)", isa_name(isa), T, werr, r);
            // the intermediate of the full pass is exactly the two halves' intermediates (same rows, same chunks)
            ExpertScratch& sf = sc;
            static ExpertScratch s0h, s1h;
            expert_gate_up(isa, view_blob_half(be.blob.data(), 0), xq, T, w.data(), s0h, 0, 36);
            expert_gate_up(isa, view_blob_half(be.blob.data(), 1), xq, T, w.data(), s1h, 0, 36);
            bool same = true;
            for (int t = 0; t < T && same; ++t) {
                std::vector<int8_t> f(kFF), a(kHalfFF), b(kHalfFF);
                act_unpack(sf.h[t], kFF, f.data());
                act_unpack(s0h.h[t], kHalfFF, a.data());
                act_unpack(s1h.h[t], kHalfFF, b.data());
                same = std::memcmp(f.data(), a.data(), kHalfFF) == 0 && std::memcmp(f.data() + kHalfFF, b.data(), kHalfFF) == 0 &&
                       std::memcmp(sf.h[t].scale, s0h.h[t].scale, 36 * 4) == 0 && std::memcmp(sf.h[t].scale + 36, s1h.h[t].scale, 36 * 4) == 0;
            }
            CHECK(same, "%s: the full intermediate is the two halves' intermediates", isa_name(isa));
        }
    }
}

// ---- 7. threads ----------------------------------------------------------------------------------------------------------
void test_threads(Rng& rng) {
    std::printf("[7] ragged thread splits, T tokens vs single tokens (bitwise)\n");
    BigExpert be(rng, 117, 122);
    const ExpertView v = view_cpu_half(be.half0.data());
    // split_range properties
    for (int n : {36, 5120, 1, 7}) for (int parts : {1, 2, 3, 5, 12, 24}) for (int align : {1, 4, 32}) {
        int prev = 0;
        for (int p = 0; p < parts; ++p) {
            int lo, hi;
            split_range(n, parts, p, align, lo, hi);
            CHECK(lo == prev && hi >= lo && hi <= n && (hi == n || hi % align == 0), "split_range(%d,%d,%d,%d) = [%d,%d)", n, parts, p, align, lo, hi);
            prev = hi;
        }
        CHECK(prev == n, "split_range covers %d (parts %d align %d)", n, parts, align);
    }
    for (Isa isa : all_isas()) {
        for (int T : {1, 3, 8}) {
            std::vector<float> x((size_t) T * kHidden), w((size_t) T);
            for (auto& e : x) e = rng.normal();
            for (auto& e : w) e = rng.uniform(0.2f, 1.9f);
            static ActQ xq[kMaxTokens];
            quantize_acts(x.data(), kHidden, T, xq, isa);
            static ExpertScratch s_ref, s_thr;
            std::vector<float> y_ref((size_t) T * kHidden, 0.f);
            expert_run(isa, v, xq, T, w.data(), s_ref, y_ref.data());
            // (a) token by token == the batch, bitwise
            {
                std::vector<float> y_single((size_t) T * kHidden, 0.f);
                static ExpertScratch s1;
                for (int t = 0; t < T; ++t) {
                    std::vector<float> yt(kHidden, 0.f);
                    expert_run(isa, v, &xq[t], 1, &w[t], s1, yt.data());
                    std::memcpy(&y_single[(size_t) t * kHidden], yt.data(), kHidden * 4);
                }
                CHECK(std::memcmp(y_single.data(), y_ref.data(), y_ref.size() * 4) == 0, "%s T=%d: token-by-token != batch (bitwise)", isa_name(isa), T);
            }
            // (b) ragged splits over real threads with a barrier between the phases
            for (int nthr : {2, 3, 5, 7, 12}) {
                // random cut points (not aligned to the tile sizes): chunk ranges for phase 1, row ranges for phase 2
                auto cuts = [&](int n) {
                    std::vector<int> c = {0, n};
                    for (int i = 1; i < nthr; ++i) c.push_back(rng.range(0, n));
                    std::sort(c.begin(), c.end());
                    return c;
                };
                const std::vector<int> c1 = cuts(v.chunks()), c2 = cuts(v.hidden);
                std::vector<float> y((size_t) T * kHidden, 0.f);
                std::memset(&s_thr, 0x5A, sizeof s_thr);    // stale garbage: everything must be rewritten by phase 1
                std::barrier bar(nthr);
                std::vector<std::thread> th;
                for (int i = 0; i < nthr; ++i)
                    th.emplace_back([&, i] {
                        expert_gate_up(isa, v, xq, T, w.data(), s_thr, c1[i], c1[i + 1]);
                        bar.arrive_and_wait();
                        expert_down(isa, v, s_thr, T, y.data(), c2[i], c2[i + 1]);
                    });
                for (auto& t : th) t.join();
                CHECK(std::memcmp(y.data(), y_ref.data(), y.size() * 4) == 0, "%s T=%d threads=%d: ragged split != single thread (bitwise)", isa_name(isa), T, nthr);
            }
        }
    }
}

// ---- 8. accumulate ---------------------------------------------------------------------------------------------------------
void test_accumulate(Rng& rng) {
    std::printf("[8] y is accumulated, not overwritten\n");
    Expert e = make_tiny_expert(rng, 256, 256, EMode::kRealistic, 118, 121);
    const int T = 2;
    std::vector<float> x((size_t) T * 256), w = {1.1f, 0.7f};
    for (auto& v : x) v = rng.normal();
    static ActQ xq[kMaxTokens];
    static ExpertScratch s;
    for (Isa isa : all_isas()) {
        quantize_acts(x.data(), 256, T, xq, isa);
        std::vector<float> y1((size_t) T * 256, 0.f), y0((size_t) T * 256), y2;
        for (auto& v : y0) v = rng.normal() * 100;
        y2 = y0;
        expert_run(isa, e.v, xq, T, w.data(), s, y1.data());
        expert_run(isa, e.v, xq, T, w.data(), s, y2.data());
        bool ok = true;
        for (size_t i = 0; i < y0.size(); ++i) ok = ok && y2[i] == y0[i] + y1[i];
        CHECK(ok, "%s: y = y_old + partial, bitwise", isa_name(isa));
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool req512 = false, req2 = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") g_quick = true;
        else if (a == "--require-avx512") req512 = true;
        else if (a == "--require-avx2") req2 = true;
        else if (a == "--seed" && i + 1 < argc) g_seed = std::strtoull(argv[++i], nullptr, 10);
        else {
            std::printf("usage: %s [--quick] [--require-avx512] [--require-avx2] [--seed N]\n", argv[0]);
            return 2;
        }
    }
    if (req512 && !isa_supported(Isa::kAvx512)) {
        std::printf("FAIL: --require-avx512 and this CPU (or build) has no AVX-512 F/BW/VL/DQ + VNNI\n");
        return 1;
    }
    if (req2 && !isa_supported(Isa::kAvx2)) {
        std::printf("FAIL: --require-avx2 and this CPU (or build) has no AVX2 + FMA + F16C\n");
        return 1;
    }
    std::printf("ds41 cpu mxfp4 test: seed %llu%s, auto ISA = %s\n", (unsigned long long) g_seed, g_quick ? " (quick)" : "",
                isa_name(resolve_isa(Isa::kAuto)));
    const auto t0 = std::chrono::steady_clock::now();
    Rng rng(g_seed);
    test_constants();
    test_quantiser(rng);
    test_dot_rows(rng);
    test_pipeline_tiny(rng);
    test_routing_weight(rng);
    test_pipeline_real(rng);
    test_halves(rng);
    test_threads(rng);
    test_accumulate(rng);
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (!isa_supported(Isa::kAvx512)) std::printf("NOTE: AVX-512 was NOT tested (CPU lacks it): run --require-avx512 on the target box\n");
    if (!isa_supported(Isa::kAvx2)) std::printf("NOTE: AVX2 was NOT tested (CPU lacks it)\n");
    std::printf("%s: %d checks, %d failed, %.1f s\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail, secs);
    return g_fail ? 1 : 0;
}
