// src/ds41/cpu/mxfp4_ggml_xcheck.cpp - DS-C: the MXFP4 semantics of this library against ggml's own implementation.
//
// The vendored llama.cpp (STRATA_GGML_DIR / the pinned commit) is the definition of "GGML MXFP4" that CONTRACTS.md points
// at, so this program asks it directly:
//
//   1. ggml's to_float for MXFP4 (dequantize_row_mxfp4) against kvalues * e8m0_half(e) of this library: EXACTLY equal, for
//      every code pair and every one of the 256 exponent bytes (e = 0 and 1 denormal, e = 255 = 2^127).  This is the check of
//      the nibble order (value j = low nibble of qs[j], value j + 16 = high nibble), of the doubled table and of the E8M0
//      decoding at once.
//   2. ggml's CPU dot product for MXFP4 x Q8_0 (the type trait `vec_dot`, which on this machine is its AVX2 kernel) against
//      mxfp4_dot_rows on every ISA, fed the SAME Q8_0 operand: the int8 values of a quantised vector with the scales rounded to
//      fp16 as Q8_0 stores them.  The integer parts are identical; the FP32 sums are ordered differently, so the tolerance is
//      1e-5 of the sum of |terms|.  T = 1..8 tokens, ragged row counts, every exponent edge.
//   3. ggml's own activation quantiser (the Q8_0 `from_float` of the CPU traits = ggml-cpu/arch/x86/quants.c quantize_row_q8_0, the x86 SIMD one) against
//      quantize_act of this library: CONTRACTS.md "Activations" says our rule IS that code with an FP32 d, so for every finite block above 2^-100 the
//      int8 values must be identical and ggml's fp16 scale must be our FP32 scale rounded to fp16.  (Below 2^-100 and for Inf / NaN blocks the
//      contract pins its own results - d = 0, q = 0 / d = NaN, q = 0 - where ggml's SIMD code produces INT_MIN-saturated garbage: not compared.)
//
// Needs the ggml-cpu target (the top-level build has it unless STRATA_NATIVE_EXPERTS=OFF).
#include "strata/ds41/cpu/mxfp4_expert.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ds41;
using namespace strata::ds41::cpu;

namespace {

int g_checks = 0, g_fail = 0;
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

struct BlockQ8 {            // ggml block_q8_0: fp16 scale, 32 int8
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 34, "block_q8_0");

}  // namespace

int main() {
    ggml_cpu_init();
    const ggml_type_traits* tt = ggml_get_type_traits(GGML_TYPE_MXFP4);
    const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);
    if (!tt || !tt->to_float || !tc || !tc->vec_dot || tc->vec_dot_type != GGML_TYPE_Q8_0) {
        std::printf("FAIL: ggml has no MXFP4 to_float / vec_dot (x) Q8_0\n");
        return 1;
    }
    std::printf("ggml MXFP4: block %zu bytes, %lld values, vec_dot_type %s\n", (size_t) tt->type_size, (long long) tt->blck_size,
                ggml_type_name(tc->vec_dot_type));
    CHECK(tt->type_size == kBlockBytes && tt->blck_size == kQK, "ggml block geometry");
    std::mt19937_64 rng(77);

    // ---- 1. to_float: every exponent, every code pair ----
    {
        std::vector<uint8_t> blocks(256 * kBlockBytes);
        for (int e = 0; e < 256; ++e) {
            uint8_t* b = &blocks[(size_t) e * kBlockBytes];
            b[0] = (uint8_t) e;
            for (int j = 0; j < 16; ++j) b[1 + j] = (uint8_t) ((e + j * 16) & 255);       // sweeps all 256 byte values
        }
        // plus every byte value in every position, at a fixed exponent
        std::vector<float> got(256 * 32);
        tt->to_float(blocks.data(), got.data(), 256 * 32);
        int bad = 0;
        for (int e = 0; e < 256; ++e)
            for (int j = 0; j < 16; ++j) {
                const uint8_t q = blocks[(size_t) e * kBlockBytes + 1 + j];
                const float lo = (float) kMxfp4Values[q & 15] * e8m0_half((uint8_t) e);
                const float hi = (float) kMxfp4Values[q >> 4] * e8m0_half((uint8_t) e);
                if (std::memcmp(&lo, &got[(size_t) e * 32 + j], 4) != 0 || std::memcmp(&hi, &got[(size_t) e * 32 + 16 + j], 4) != 0) ++bad;
            }
        CHECK(bad == 0, "ggml to_float differs from kvalues * e8m0_half in %d of %d (e, j) pairs", bad, 256 * 16);
        // every byte value x every position
        std::vector<uint8_t> all(16 * 256 * kBlockBytes);
        for (int pos = 0; pos < 16; ++pos)
            for (int v = 0; v < 256; ++v) {
                uint8_t* b = &all[((size_t) pos * 256 + v) * kBlockBytes];
                b[0] = 127;
                std::memset(b + 1, 0, 16);
                b[1 + pos] = (uint8_t) v;
            }
        std::vector<float> g2(16 * 256 * 32);
        tt->to_float(all.data(), g2.data(), (int64_t) g2.size());
        int bad2 = 0;
        for (int pos = 0; pos < 16; ++pos)
            for (int v = 0; v < 256; ++v)
                for (int j = 0; j < 32; ++j) {
                    float want = 0;
                    if (j == pos) want = (float) kMxfp4Values[v & 15] * e8m0_half(127);
                    else if (j == pos + 16) want = (float) kMxfp4Values[v >> 4] * e8m0_half(127);
                    if (want != g2[((size_t) pos * 256 + v) * 32 + j]) ++bad2;
                }
        CHECK(bad2 == 0, "ggml to_float: nibble placement differs in %d elements", bad2);
    }

    // ---- 2. vec_dot ----
    double worst = 0;
    auto run = [&](int nb, int nrows, int T, int efixed, float xlog2) {
        const int n = nb * 32;
        std::vector<uint8_t> w((size_t) nrows * nb * kBlockBytes);
        for (size_t r = 0; r < (size_t) nrows; ++r)
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = &w[(r * nb + b) * kBlockBytes];
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) rng();
                blk[0] = efixed >= 0 ? (uint8_t) efixed : (uint8_t) (112 + rng() % 15);
            }
        std::vector<float> x((size_t) T * n);
        std::normal_distribution<float> nd;
        for (auto& v : x) v = nd(rng) * std::ldexp(1.0f, (int) xlog2);
        std::vector<ActQ> xq((size_t) T);
        std::vector<std::vector<BlockQ8>> q8((size_t) T, std::vector<BlockQ8>((size_t) nb));
        for (int t = 0; t < T; ++t) {
            quantize_act(&x[(size_t) t * n], n, xq[t], Isa::kScalar);
            // the Q8_0 operand: the same int8 values, the scale rounded to fp16 like Q8_0 stores it; and the same operand
            // handed to our kernels (scales widened back to fp32)
            std::vector<int8_t> q((size_t) n);
            std::vector<float> d16((size_t) nb);
            act_unpack(xq[t], n, q.data());
            for (int b = 0; b < nb; ++b) {
                q8[t][b].d = ggml_fp32_to_fp16(xq[t].scale[b]);
                d16[b] = ggml_fp16_to_fp32(q8[t][b].d);
                std::memcpy(q8[t][b].qs, &q[(size_t) b * 32], 32);
            }
            act_from_q8(q.data(), d16.data(), n, xq[t]);
        }
        for (Isa isa : {Isa::kScalar, Isa::kAvx2, Isa::kAvx512}) {
            if (!isa_supported(isa)) continue;
            std::vector<float> mine((size_t) T * nrows);
            mxfp4_dot_rows(isa, w.data(), (size_t) nb * kBlockBytes, nb, nrows, xq.data(), T, mine.data(), (size_t) nrows, false);
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < nrows; ++r) {
                    float ref = 0;
                    tc->vec_dot(n, &ref, 0, &w[(size_t) r * nb * kBlockBytes], 0, q8[t].data(), 0, 1);
                    // sum of |terms| for the tolerance
                    double mag = 0;
                    for (int b = 0; b < nb; ++b) {
                        const uint8_t* blk = &w[((size_t) r * nb + b) * kBlockBytes];
                        long isum = 0, iabs = 0;
                        for (int j = 0; j < 16; ++j) {
                            const long a = (long) kMxfp4Values[blk[1 + j] & 15] * q8[t][b].qs[j];
                            const long c = (long) kMxfp4Values[blk[1 + j] >> 4] * q8[t][b].qs[16 + j];
                            isum += a + c;
                            iabs += std::labs(a) + std::labs(c);
                        }
                        (void) isum;
                        mag += (double) e8m0_half(blk[0]) * (double) ggml_fp16_to_fp32(q8[t][b].d) * (double) iabs;
                    }
                    const double got = mine[(size_t) t * nrows + r];
                    const double err = std::fabs(got - ref);
                    if (mag > 1e-25) worst = std::max(worst, err / mag);
                    CHECK(std::isfinite(got) && std::isfinite(ref) && err <= 1e-5 * mag + 1e-37,
                          "%s nb=%d rows=%d T=%d e=%d: (t=%d r=%d) ours %.9g ggml %.9g mag %.3g", isa_name(isa), nb, nrows, T,
                          efixed, t, r, got, (double) ref, mag);
                }
        }
    };
    for (int nb : {4, 36, 72, 160})
        for (int T : {1, 2, 3, 5, 8})
            for (int rows : {1, 4, 7, 16}) run(nb, rows, T, -1, 0);
    for (int e : {0, 1, 2, 126, 127, 128, 129, 253, 254, 255}) {
        const float xl = e < 8 ? 12.0f : (e > 248 ? -9.0f : 0.0f);   // keep every product finite and representable
        for (int T : {1, 4}) run(8, 6, T, e, xl);
    }
    std::printf("ggml vec_dot (MXFP4 x Q8_0) vs ours: worst |err| / sum|terms| = %.2e (limit 1e-5)\n", worst);

    // ---- 3. the activation quantiser against ggml's quantize_row_q8_0 ----
    {
        const ggml_type_traits_cpu* q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (!q8 || !q8->from_float) {
            std::printf("FAIL: ggml has no Q8_0 from_float\n");
            return 1;
        }
        const int nblk = 160, n = nblk * 32;
        long blocks = 0, bad_q = 0, bad_d = 0;
        std::normal_distribution<float> nd;
        std::vector<float> x((size_t) n);
        std::vector<BlockQ8> g((size_t) nblk);
        for (int rep = 0; rep < 60; ++rep) {
            for (int b = 0; b < nblk; ++b) {
                const int kind = (int) (rng() % 4);
                const float sc = std::ldexp(1.0f, (int) (rng() % 21) - 10);
                for (int j = 0; j < 32; ++j) {
                    float v = nd(rng) * sc;
                    if (kind == 1) v = (float) ((int) (rng() % 601) - 300) * 0.5f;               // integer / half-integer: rounding ties
                    if (kind == 2) v = j == 0 ? 254.0f * sc : j == 1 ? 5.0f * sc : j == 2 ? 1.0f * sc : j == 3 ? -5.0f * sc : 0.0f;   // the [254, 5, 1, -5, 0 ...] tie pattern
                    if (kind == 3) v = (j & 1) ? -3.0f * sc : 3.0f * sc;
                    x[(size_t) b * 32 + j] = v;
                }
            }
            q8->from_float(x.data(), g.data(), n);
            static ActQ a[3];
            std::vector<int8_t> q((size_t) n);
            int ni = 0;
            for (Isa isa : {Isa::kScalar, Isa::kAvx2, Isa::kAvx512}) {
                if (!isa_supported(isa)) continue;
                ActQ& ai = a[ni++];
                quantize_act(x.data(), n, ai, isa);
                act_unpack(ai, n, q.data());
                for (int b = 0; b < nblk; ++b) {
                    ++blocks;
                    bad_q += std::memcmp(&q[(size_t) b * 32], g[b].qs, 32) != 0;
                    bad_d += ggml_fp32_to_fp16(ai.scale[b]) != g[b].d;
                }
            }
        }
        CHECK(bad_q == 0 && bad_d == 0, "ggml quantize_row_q8_0 differs from quantize_act: %ld of %ld blocks' int8 values, %ld scales", bad_q, blocks, bad_d);
        std::printf("ggml quantize_row_q8_0 vs quantize_act (every ISA): %ld blocks, int8 values and fp16-rounded scales identical: %s\n", blocks,
                    (bad_q == 0 && bad_d == 0) ? "yes" : "NO");
    }
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAILED" : "PASSED", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
