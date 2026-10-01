// src/kernels/cpu/iq_avx512_test.cpp - the i-quant AVX-512 rows against ggml's own dot products (CPU only, synthetic).
//
//   iq_avx512_test            parity of iq512_rows / iq512_gu_rows (and the AVX2 iq256 ones) against ggml-cpu's vec_dot
//                             for IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S on random rows quantized by ggml, 1..4 tokens
//   iq_avx512_test --bench    one-thread and T-thread weight throughput of ggml vec_dot, iq256 (AVX2) and iq512 (AVX-512),
//                             best of --reps windows (--experts 2 makes it cache-resident: the compute-bound ceiling)
//
// Why this exists: iq_avx512.cpp is compiled WITHOUT -mavx512vbmi (it uses nothing newer than AVX512-BW), so a Cascade Lake
// CPU can run it, and `cpu_avx512_ok()` now says yes there.  native_expert_parity.cpp checks the same thing but needs a
// GPU build; this one needs only ggml.  The reference is ggml's own `vec_dot` over the same quantized activation, and
// the two differ only in the order of float additions (measured ~3e-8; the bound is 1e-5, as in native_expert_parity).
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include "ggml-cpu.h"
#include "ggml.h"

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

namespace c = strata::kernels::cpu;

namespace {

constexpr int64_t H = 2560, FF = 640;
constexpr int MAXNT = 4;

struct Fixture {
    ggml_type type;
    size_t row_bytes = 0;
    std::vector<uint8_t> blob;      // gate rows, then up rows (FF each)
    size_t up_off = 0;
    std::vector<std::vector<uint8_t>> act;   // MAXNT quantized activations (the type's vec_dot_type)
};

double rel(const std::vector<float>& a, const std::vector<float>& ref) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - ref[i]); d += std::fabs((double) ref[i]); }
    return n / (d + 1e-30);
}

bool make_fixture(ggml_type type, int64_t rows, Fixture& f, std::mt19937& rng) {
    f.type = type;
    f.row_bytes = ggml_row_size(type, H);
    f.up_off = (size_t) rows * f.row_bytes;
    f.blob.assign(2 * f.up_off, 0);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> src((size_t) rows * H), imat((size_t) H, 1.0f);
    for (int part = 0; part < 2; ++part) {
        for (auto& v : src) v = nd(rng) * (0.5f + 0.5f * nd(rng) * nd(rng));
        const size_t got = ggml_quantize_chunk(type, src.data(), f.blob.data() + (size_t) part * f.up_off, 0, rows, H,
                                               ggml_quantize_requires_imatrix(type) ? imat.data() : nullptr);
        if (got != f.up_off) return false;
    }
    const ggml_type vt = ggml_get_type_traits_cpu(type)->vec_dot_type;
    const auto* from = ggml_get_type_traits_cpu(vt);
    f.act.assign(MAXNT, std::vector<uint8_t>(ggml_row_size(vt, H)));
    std::vector<float> x((size_t) H);
    for (int t = 0; t < MAXNT; ++t) {
        for (auto& v : x) v = nd(rng) * (t == 3 ? 20.f : 1.f);
        from->from_float(x.data(), f.act[(size_t) t].data(), H);
    }
    return true;
}

int g_fail = 0;
void check(bool ok, const std::string& what, double r) {
    std::printf("  %-60s rel %.2e  %s\n", what.c_str(), r, ok ? "ok" : "*** FAILED ***");
    if (!ok) ++g_fail;
}

void parity(ggml_type type) {
    std::mt19937 rng(100 + (int) type);
    Fixture f;
    const int64_t rows = 48;
    if (!make_fixture(type, rows, f, rng)) { std::printf("  %s: quantize failed\n", ggml_type_name(type)); ++g_fail; return; }
    const auto* tc = ggml_get_type_traits_cpu(type);
    for (int nt : {1, 2, 3, 4}) {
        const void* act[MAXNT];
        for (int t = 0; t < MAXNT; ++t) act[t] = f.act[(size_t) t].data();
        std::vector<std::vector<float>> got512(MAXNT, std::vector<float>((size_t) rows)), got256 = got512;
        float* p512[MAXNT];
        float* p256[MAXNT];
        for (int t = 0; t < MAXNT; ++t) { p512[t] = got512[(size_t) t].data(); p256[t] = got256[(size_t) t].data(); }
        c::iq512_rows((int) type, f.blob.data(), f.row_bytes, (int) H, act, nt, p512, 0, (int) rows);
        c::iq256_rows((int) type, f.blob.data(), f.row_bytes, (int) H, act, nt, p256, 0, (int) rows);
        std::vector<float> ref((size_t) nt * rows), a512((size_t) nt * rows), a256((size_t) nt * rows);
        for (int t = 0; t < nt; ++t)
            for (int64_t r = 0; r < rows; ++r) {
                tc->vec_dot((int) H, &ref[(size_t) (t * rows + r)], 0, f.blob.data() + (size_t) r * f.row_bytes, 0, act[t], 0, 1);
                a512[(size_t) (t * rows + r)] = got512[(size_t) t][(size_t) r];
                a256[(size_t) (t * rows + r)] = got256[(size_t) t][(size_t) r];
            }
        char what[96];
        const double r512 = rel(a512, ref), r256 = rel(a256, ref);
        std::snprintf(what, sizeof what, "%s rows, %d token(s): AVX-512 vs ggml vec_dot", ggml_type_name(type), nt);
        check(r512 <= 1e-5, what, r512);
        std::snprintf(what, sizeof what, "%s rows, %d token(s): AVX2 vs ggml vec_dot", ggml_type_name(type), nt);
        check(r256 <= 1e-5, what, r256);

        // gate/up with the SiLU: ff = silu(g) * u
        std::vector<std::vector<float>> ff512(MAXNT, std::vector<float>((size_t) rows));
        float* pf[MAXNT];
        for (int t = 0; t < MAXNT; ++t) pf[t] = ff512[(size_t) t].data();
        c::iq512_gu_rows((int) type, f.blob.data(), f.row_bytes, f.up_off, (int) H, act, nt, pf, 0, (int) rows);
        std::vector<float> ffref((size_t) nt * rows), ffgot((size_t) nt * rows);
        for (int t = 0; t < nt; ++t)
            for (int64_t r = 0; r < rows; ++r) {
                float g = 0, u = 0;
                tc->vec_dot((int) H, &g, 0, f.blob.data() + (size_t) r * f.row_bytes, 0, act[t], 0, 1);
                tc->vec_dot((int) H, &u, 0, f.blob.data() + f.up_off + (size_t) r * f.row_bytes, 0, act[t], 0, 1);
                ffref[(size_t) (t * rows + r)] = (g / (1.f + std::exp(-g))) * u;
                ffgot[(size_t) (t * rows + r)] = ff512[(size_t) t][(size_t) r];
            }
        const double rg = rel(ffgot, ffref);
        std::snprintf(what, sizeof what, "%s gate/up, %d token(s): AVX-512 vs ggml + SiLU", ggml_type_name(type), nt);
        check(rg <= 1e-5, what, rg);

        // and through the engine's own entry point, which picks the kernel by tier (STRATA_IQ512 / STRATA_NO_IQ512 /
        // STRATA_FORCE_* apply): whichever it picked must agree with ggml too
        if (nt >= 2) {
            c::NativeFmt nf;
            nf.gu_type = (int) type;
            nf.gu_row = f.row_bytes;
            nf.up_off = f.up_off;
            nf.n_embd = H;
            std::vector<std::vector<float>> ffn(MAXNT, std::vector<float>((size_t) rows));
            float* pn[MAXNT];
            for (int t = 0; t < MAXNT; ++t) pn[t] = ffn[(size_t) t].data();
            c::native_gu_rows(nf, f.blob.data(), act, nt, pn, 0, (int) rows);
            std::vector<float> nget((size_t) nt * rows);
            for (int t = 0; t < nt; ++t)
                for (int64_t r = 0; r < rows; ++r) nget[(size_t) (t * rows + r)] = ffn[(size_t) t][(size_t) r];
            const double rn = rel(nget, ffref);
            std::snprintf(what, sizeof what, "%s gate/up, %d token(s): native_gu_rows (the engine's pick) vs ggml", ggml_type_name(type), nt);
            check(rn <= 1e-5, what, rn);
        }
    }
}

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void bench(ggml_type type, int threads, double seconds, int reps, int experts) {
    std::mt19937 rng(7);
    Fixture f;
    const int64_t rows = FF;   // one expert's gate and up: 2 x 640 rows
    if (!make_fixture(type, rows, f, rng)) return;
    const size_t one = f.blob.size();
    std::vector<uint8_t> arena((size_t) experts * one);
    for (int e = 0; e < experts; ++e) std::memcpy(&arena[(size_t) e * one], f.blob.data(), one);
    const void* act[MAXNT];
    for (int t = 0; t < MAXNT; ++t) act[t] = f.act[(size_t) t].data();
    const auto* tc = ggml_get_type_traits_cpu(type);

    for (int nt : {1, 2, 3}) {
        for (int impl = 0; impl < 3; ++impl) {
            if (impl == 1 && !c::iq256_supported((int) type)) continue;
            auto run = [&](int e) {
                static thread_local float o[MAXNT][FF + 8];
                float* po[MAXNT];
                for (int t = 0; t < MAXNT; ++t) po[t] = o[t];
                const uint8_t* b = &arena[(size_t) e * one];
                if (impl == 0) {
                    for (int64_t r = 0; r < rows; ++r)
                        for (int t = 0; t < nt; ++t) {
                            float g, u;
                            tc->vec_dot((int) H, &g, 0, b + (size_t) r * f.row_bytes, 0, act[t], 0, 1);
                            tc->vec_dot((int) H, &u, 0, b + f.up_off + (size_t) r * f.row_bytes, 0, act[t], 0, 1);
                            o[t][r] = g * u;
                        }
                } else if (impl == 1) {
                    c::iq256_gu_rows((int) type, b, f.row_bytes, f.up_off, (int) H, act, nt, po, 0, (int) rows);
                } else {
                    c::iq512_gu_rows((int) type, b, f.row_bytes, f.up_off, (int) H, act, nt, po, 0, (int) rows);
                }
            };
            // BEST OF `reps` windows (as pool_test does): on a shared machine a mean measures the neighbours, the best
            // window measures the kernel.
            double gb[2] = {0, 0};
            const int tcount[2] = {1, threads};
            const double window = seconds / reps;
            for (int k = 0; k < 2; ++k)
                for (int rep = 0; rep < reps; ++rep) {
                    std::atomic<bool> go{false}, stop{false};
                    std::atomic<int> ready{0};
                    std::vector<long long> done((size_t) tcount[k], 0);
                    std::vector<std::thread> pool;
                    for (int t = 0; t < tcount[k]; ++t)
                        pool.emplace_back([&, t] {
                            ready.fetch_add(1);
                            while (!go.load()) std::this_thread::yield();
                            long long n = 0;
                            while (!stop.load(std::memory_order_relaxed))
                                for (int e = t; e < experts && !stop.load(std::memory_order_relaxed); e += tcount[k]) { run(e); ++n; }
                            done[(size_t) t] = n;
                        });
                    while (ready.load() < tcount[k]) std::this_thread::yield();
                    const double t0 = now_s();
                    go = true;
                    std::this_thread::sleep_for(std::chrono::duration<double>(window));
                    stop = true;
                    for (auto& th : pool) th.join();
                    const double dt = now_s() - t0;
                    long long total = 0;
                    for (auto n : done) total += n;
                    gb[k] = std::max(gb[k], (double) total * (double) one / dt / 1e9);   // weight bytes streamed (gate + up)
                }
            std::printf("  %-9s %d token%s  %-22s %8.2f GB/s (1 thread) %8.2f GB/s (%d threads)\n", ggml_type_name(type), nt,
                        nt == 1 ? " " : "s", impl == 0 ? "ggml vec_dot" : impl == 1 ? "AVX2 (iq256)" : "AVX-512 (iq512)", gb[0], gb[1], threads);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    int threads = 4, reps = 5, experts = 96;   // 96 experts: ~ 270 MB for IQ3_S, far beyond the caches; --experts 2 = cache-resident
    double seconds = 1.5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") { do_bench = true; if (i + 1 < argc && argv[i + 1][0] != '-') threads = std::atoi(argv[++i]); }
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) experts = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: iq_avx512_test [--bench [T]] [--seconds S] [--reps N] [--experts E]\n"); return 2; }
    }
    ggml_cpu_init();
    std::printf("CPU: %s\n  AVX2 kernels %s; AVX-512 kernels %s (tier: %s)\n", c::cpu_name().c_str(),
                c::cpu_avx2_ok() ? "usable" : "NOT usable", c::cpu_avx512_ok() ? "usable" : "NOT usable",
                c::expert_isa_name(c::cpu_expert_isa()));
    if (!c::cpu_avx512_ok() || !c::cpu_avx2_ok()) {
        std::printf("  needs AVX2 and AVX-512: SKIPPED, not passed.\n");
        return 0;
    }
    const ggml_type types[] = {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S};
    if (do_bench) {
        std::printf("gate+up of one expert, weight bytes streamed per second; indicative (shared VM)\n");
        for (ggml_type t : types) bench(t, threads, seconds, reps, experts);
        return 0;
    }
    for (ggml_type t : types) {
        std::printf("%s\n", ggml_type_name(t));
        parity(t);
    }
    std::printf("\niq_avx512_test: %d failures\n", g_fail);
    if (g_fail == 0) std::printf("iq_avx512_test OK\n");
    return g_fail ? 1 : 0;
}
