// src/kernels/cpu/expert_variant_test.cpp - the two builds of the AVX-512 Q2_0 kernels (CPU only, synthetic data).
//
//   expert_variant_test               unpack helpers against a scalar unpack (single-byte, every byte value x every byte
//                                     position, structured and random blocks), the kernels of each runnable build against
//                                     the scalar oracle, and - on a CPU that has VBMI - the two builds against each other
//                                     BIT FOR BIT (every output of every kernel entry point)
//   expert_variant_test --exhaustive  the unpack helpers over ALL 2^32 values of a packed dword in every lane (a minute or two
//                                     per build and width; --threads N, default 2)
//   expert_variant_test --bench [T]   weight-streaming throughput of the AVX2 rows, the AVX-512 rows (each build the CPU can
//                                     run) and the canonical expert kernel, 1 thread and T threads (default 4)
//
// WHAT "THE SAME" MEANS.  The VBMI and no-VBMI builds differ only in the 2-bit unpack; every other instruction is
// the same source.  So they are bit-identical if and only if the two unpacks give the same 64 (or 32) bytes for every
// input, and the exhaustive mode proves that for the no-VBMI one by enumeration: each output lane depends on ONE input
// dword (4 bytes, 2^32 values), the four lanes are independent, and the loop below feeds every value to every lane
// (the four dwords of a call are four different bijections of the loop counter).  On a VBMI machine the same test runs on
// the VBMI build too, and the cross-build comparison covers whole kernels.  On a machine without VBMI the VBMI half
// reports SKIPPED - it cannot be executed there, and the tool says so instead of passing it.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "expert_variant.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "*** FAILED ***");
    if (!ok) ++g_fail;
}

// ---- the specification: output byte j is (codes[j / 4] >> (2 * (j % 4))) & 3
void scalar_unpack(const uint8_t* codes, uint8_t* out, int n_out) {
    for (int j = 0; j < n_out; ++j) out[j] = (uint8_t) ((codes[j / 4] >> (2 * (j % 4))) & 3);
}

struct Unpack {
    const char* name;
    void (*fn)(const uint8_t*, uint8_t*);
    int in_bytes, out_bytes;
};

uint32_t xorshift(uint64_t& s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (uint32_t) (s >> 16);
}

/// Every (byte position, byte value) pair alone, every value in all positions, a few fixed patterns, and `n_random`
/// random blocks.  Each output byte depends on exactly one input byte, so the first group already covers the function
/// for all inputs; the others catch an unintended coupling between bytes.
long long check_unpack(const Unpack& u, long long n_random, bool& ok) {
    long long cases = 0;
    uint8_t in[16] = {}, out[64], ref[64];
    auto one = [&]() {
        std::memset(out, 0xCC, sizeof out);
        u.fn(in, out);
        scalar_unpack(in, ref, u.out_bytes);
        ++cases;
        if (std::memcmp(out, ref, (size_t) u.out_bytes) != 0) ok = false;
    };
    for (int p = 0; p < u.in_bytes; ++p)
        for (int v = 0; v < 256; ++v) { std::memset(in, 0, sizeof in); in[p] = (uint8_t) v; one(); }
    for (int p = 0; p < u.in_bytes; ++p)
        for (int v = 0; v < 256; ++v) { std::memset(in, 0xFF, sizeof in); in[p] = (uint8_t) v; one(); }
    for (int v = 0; v < 256; ++v) { std::memset(in, v, sizeof in); one(); }
    for (int v = 0; v < 256; ++v)
        for (int w = 0; w < 256; w += 15) { for (int i = 0; i < 16; ++i) in[i] = (uint8_t) ((i & 1) ? w : v); one(); }
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (long long i = 0; i < n_random; ++i) {
        for (int k = 0; k < 4; ++k) { const uint32_t r = xorshift(s); std::memcpy(in + 4 * k, &r, 4); }
        one();
    }
    return cases;
}

/// All 2^32 packed dwords in every lane: call i feeds dword (i, i ^ A, ~i, rotl(i, 13)) - four bijections of i - and the
/// expected output is the lookup table of one byte's four codes.  `lo..hi` is this thread's share of i.
bool exhaustive_range(const Unpack& u, uint64_t lo, uint64_t hi) {
    static uint32_t lut[256];
    static std::once_flag once;
    std::call_once(once, [] {
        for (int b = 0; b < 256; ++b)
            lut[b] = (uint32_t) (b & 3) | (uint32_t) ((b >> 2) & 3) << 8 | (uint32_t) ((b >> 4) & 3) << 16 |
                     (uint32_t) ((b >> 6) & 3) << 24;
    });
    const int lanes = u.in_bytes / 4;
    alignas(64) uint8_t in[16], out[64];
    uint32_t want[16];
    for (uint64_t i = lo; i < hi; ++i) {
        const uint32_t x = (uint32_t) i;
        const uint32_t d[4] = {x, x ^ 0xA5A5C3C3u, ~x, (x << 13) | (x >> 19)};
        std::memcpy(in, d, 16);
        u.fn(in, out);
        for (int l = 0; l < lanes; ++l)
            for (int m = 0; m < 4; ++m) want[4 * l + m] = lut[(d[l] >> (8 * m)) & 255];
        if (std::memcmp(out, want, (size_t) u.out_bytes) != 0) {
            std::printf("  MISMATCH %s at i = 0x%08llx\n", u.name, (unsigned long long) i);
            return false;
        }
    }
    return true;
}

bool exhaustive(const Unpack& u, int threads) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    std::atomic<bool> ok{true};
    const uint64_t total = 1ull << 32;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            if (!exhaustive_range(u, total * (uint64_t) t / (uint64_t) threads, total * (uint64_t) (t + 1) / (uint64_t) threads))
                ok = false;
        });
    for (auto& th : pool) th.join();
    std::printf("      %s: 2^32 dwords x all lanes in %.1f s\n", u.name,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return ok.load();
}

// ---- synthetic experts
void make_blob(uint8_t* b, std::mt19937& rng) {
    for (size_t i = 0; i < c::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();               // codes
    for (size_t i = c::O_GU_SCALES; i < c::BLOB; i += 2) {                             // fp16 scales ~0.004-0.016
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b + i, &h, 2);
    }
}

double rel_l1(const float* a, const float* b, int n) {
    double d = 0, m = 0;
    for (int i = 0; i < n; ++i) { d += std::fabs((double) a[i] - b[i]); m += std::fabs((double) a[i]); }
    return d / (m > 1e-30 ? m : 1e-30);
}

/// Every entry point of two kernel tables on the same inputs: outputs must be byte-identical.
int compare_builds(const c::ExpertKernels& A, const c::ExpertKernels& B) {
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int E = 3;
    std::vector<uint8_t> blobs((size_t) E * c::BLOB);
    for (int e = 0; e < E; ++e) make_blob(&blobs[(size_t) e * c::BLOB], rng);
    static c::ActQ acts[2][c::MAXT];
    static float xs[c::MAXT][c::H];
    for (int t = 0; t < c::MAXT; ++t) {
        for (float& v : xs[t]) v = nd(rng) * (t == 5 ? 1e-3f : t == 6 ? 40.f : 1.f);   // a tiny and a large token too
        A.act_quant(xs[t], c::H, acts[0][t]);
        B.act_quant(xs[t], c::H, acts[1][t]);
    }
    int bad = 0;
    for (int t = 0; t < c::MAXT; ++t)
        if (std::memcmp(acts[0][t].q, acts[1][t].q, c::H) || std::memcmp(acts[0][t].scale, acts[1][t].scale, sizeof acts[0][t].scale) ||
            std::memcmp(acts[0][t].sum, acts[1][t].sum, sizeof acts[0][t].sum))
            ++bad;
    std::printf("  %-66s %s\n", "act_quant: identical activations", bad ? "*** FAILED ***" : "ok");
    if (bad) ++g_fail;

    static c::ExpertScratch wsA, wsB;
    static c::ExpertScratchMulti wmA, wmB;
    static float oA[c::MAXT][c::H], oB[c::MAXT][c::H];
    float* pA[c::MAXT];
    float* pB[c::MAXT];
    const c::ActQ* aA[c::MAXT];
    const c::ActQ* aB[c::MAXT];
    for (int t = 0; t < c::MAXT; ++t) { pA[t] = oA[t]; pB[t] = oB[t]; aA[t] = &acts[0][t]; aB[t] = &acts[1][t]; }
    long long n_cmp = 0, n_diff = 0;
    auto same = [&](const float* x, const float* y, size_t n) {
        ++n_cmp;
        if (std::memcmp(x, y, n * sizeof(float)) != 0) ++n_diff;
    };
    for (int e = 0; e < E; ++e) {
        const uint8_t* blob = &blobs[(size_t) e * c::BLOB];
        A.expert_q(blob, acts[0][0], oA[0], wsA);
        B.expert_q(blob, acts[1][0], oB[0], wsB);
        same(oA[0], oB[0], c::H);
        for (int n = 1; n <= c::MAXT; ++n) {
            A.expert_multi(blob, aA, n, pA, wmA);
            B.expert_multi(blob, aB, n, pB, wmB);
            for (int t = 0; t < n; ++t) same(oA[t], oB[t], c::H);
        }
        // the row-range entry points, an odd split like the pool's
        static float ffA[c::MAXT][c::FF], ffB[c::MAXT][c::FF];
        float* fA[c::MAXT];
        float* fB[c::MAXT];
        for (int t = 0; t < c::MAXT; ++t) { fA[t] = ffA[t]; fB[t] = ffB[t]; }
        for (int n : {1, 3, 8}) {
            A.gu_rows_multi(blob, aA, n, fA, 7, 613);
            B.gu_rows_multi(blob, aB, n, fB, 7, 613);
            for (int t = 0; t < n; ++t) same(ffA[t] + 7, ffB[t] + 7, 606);
            A.down_rows_multi(blob, aA, n, pA, 5, 2501);
            B.down_rows_multi(blob, aB, n, pB, 5, 2501);
            for (int t = 0; t < n; ++t) same(oA[t] + 5, oB[t] + 5, 2496);
        }
        A.gu_rows(blob, acts[0][1], ffA[0], 0, c::FF);
        B.gu_rows(blob, acts[1][1], ffB[0], 0, c::FF);
        same(ffA[0], ffB[0], c::FF);
        A.down_rows(blob, acts[0][2], oA[0], 0, c::H);
        B.down_rows(blob, acts[1][2], oB[0], 0, c::H);
        same(oA[0], oB[0], c::H);
        // the GGUF-layout rows (18 B blocks), gate/up shaped and down shaped, 1..8 tokens
        for (int n : {1, 2, 5, 8}) {
            A.q2g_rows_multi(blob, 720, 40, aA, n, pA, 0, 1280);
            B.q2g_rows_multi(blob, 720, 40, aB, n, pB, 0, 1280);
            for (int t = 0; t < n; ++t) same(oA[t], oB[t], 1280 < c::H ? 1280 : c::H);
            A.q2g_rows_multi(blob + 921600, 180, 10, aA, n, pA, 0, c::H);
            B.q2g_rows_multi(blob + 921600, 180, 10, aB, n, pB, 0, c::H);
            for (int t = 0; t < n; ++t) same(oA[t], oB[t], c::H);
        }
    }
    std::printf("  %-66s %s (%lld buffers)\n", "kernels: every output buffer bit-identical between the builds",
                n_diff ? "*** FAILED ***" : "ok", n_cmp);
    if (n_diff) ++g_fail;
    return bad + (int) n_diff;
}

/// One build's expert kernel against the scalar transcription of the formula (the same bound as expert_parity).
void check_vs_scalar(const c::ExpertKernels& K) {
    std::mt19937 rng(77);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint8_t> blob(c::BLOB);
    make_blob(blob.data(), rng);
    std::vector<float> x(c::H), got(c::H), ref(c::H);
    for (float& v : x) v = nd(rng);
    static c::ExpertScratch ws;
    K.act_quant(x.data(), c::H, ws.a1);
    K.expert_q(blob.data(), ws.a1, got.data(), ws);
    c::s2_expert_scalar(blob.data(), x.data(), ref.data(), true);
    const double r = rel_l1(ref.data(), got.data(), c::H);
    char msg[120];
    std::snprintf(msg, sizeof msg, "%s: expert vs the scalar oracle, rel %.2e (<= 3e-3)", K.name, r);
    check(r <= 3e-3, msg);
}

// ---- benchmark
double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// `work(expert_index, thread_state)` over the arena by T threads for about `seconds`; GB/s of weights streamed.
template <class Work>
double stream_gbs(int threads, int experts, double seconds, Work work) {
    std::atomic<int> ready{0};
    std::atomic<bool> go{false}, stop{false};
    std::vector<long long> done((size_t) threads, 0);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            long long n = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (int e = t; e < experts; e += threads) { work(e, t); ++n; }
            }
            done[(size_t) t] = n;
        });
    while (ready.load() < threads) std::this_thread::yield();
    const double t0 = now_s();
    go = true;
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto& th : pool) th.join();
    const double dt = now_s() - t0;
    long long total = 0;
    for (long long n : done) total += n;
    return (double) total * (double) c::BLOB / dt / 1e9;
}

/// Best of `reps` windows of `seconds / reps` (as pool_test does): on a shared machine a mean measures the neighbours.
template <class Work>
double best_gbs(int threads, int experts, double seconds, int reps, Work work) {
    double best = 0;
    for (int r = 0; r < reps; ++r) best = std::max(best, stream_gbs(threads, experts, seconds / reps, work));
    return best;
}

void bench(int threads, int experts, double seconds) {
    std::mt19937 rng(5);
    std::vector<uint8_t> arena((size_t) experts * c::BLOB);
    for (int e = 0; e < experts; ++e) make_blob(&arena[(size_t) e * c::BLOB], rng);
    std::normal_distribution<float> nd(0.f, 1.f);
    static c::ActQ a1, a2;
    std::vector<float> x(c::H), h(c::FF);
    for (float& v : x) v = nd(rng);
    for (float& v : h) v = nd(rng);
    c::act_quant_q8_1_avx2(x.data(), c::H, a1);
    c::act_quant_q8_1_avx2(h.data(), c::FF, a2);
    std::printf("%d experts x %.2f MB = %.0f MB of weights, best of 5 windows in %.1f s per cell; GB/s = weight bytes streamed (codes + scales)\n",
                experts, c::BLOB / 1e6, (double) experts * c::BLOB / 1e6, seconds);
    std::printf("%-44s %10s %10s\n", "kernel (one token, GGUF-layout Q2_0 rows)", "1 thread", (std::to_string(threads) + " threads").c_str());

    struct Row {
        std::string name;
        std::function<void(const uint8_t*)> run;
    };
    std::vector<Row> rows;
    const c::ActQ* pa1[1] = {&a1};
    const c::ActQ* pa2[1] = {&a2};
    // per-thread outputs: a plain thread_local so the lambdas stay stateless
    static thread_local float out_gu[1280 + 64], out_d[c::H + 64];
    rows.push_back({"AVX2   (q2_avx2.cpp)", [&](const uint8_t* b) {
        float* o1[1] = {out_gu};
        float* o2[1] = {out_d};
        c::q2_0_gguf_rows_multi_avx2(b, 720, 40, pa1, 1, o1, 0, 1280);
        c::q2_0_gguf_rows_multi_avx2(b + 921600, 180, 10, pa2, 1, o2, 0, c::H);
    }});
    const c::ExpertKernels* builds[2] = {nullptr, nullptr};
    if (c::cpu_features().usable()) builds[0] = &c::expert_kernels_novbmi();
    if (c::cpu_features().vbmi()) builds[1] = &c::expert_kernels_vbmi();
    for (const c::ExpertKernels* K : builds) {
        if (!K) continue;
        rows.push_back({std::string("AVX-512 ") + (K == builds[0] ? "no-VBMI build" : "VBMI build   "), [&, K](const uint8_t* b) {
            float* o1[1] = {out_gu};
            float* o2[1] = {out_d};
            K->q2g_rows_multi(b, 720, 40, pa1, 1, o1, 0, 1280);
            K->q2g_rows_multi(b + 921600, 180, 10, pa2, 1, o2, 0, c::H);
        }});
    }
    for (const Row& r : rows) {
        auto work = [&](int e, int) { r.run(&arena[(size_t) e * c::BLOB]); };
        stream_gbs(1, experts, 0.3, work);   // warm
        const double g1 = best_gbs(1, experts, seconds, 5, work);
        const double gn = best_gbs(threads, experts, seconds, 5, work);
        std::printf("%-44s %10.2f %10.2f\n", r.name.c_str(), g1, gn);
    }
    std::printf("\n%-44s %10s %10s\n", "canonical Q2_0 pack expert (AVX-512 only)", "1 thread", (std::to_string(threads) + " threads").c_str());
    for (const c::ExpertKernels* K : builds) {
        if (!K) continue;
        static thread_local c::ExpertScratchMulti ws;
        static thread_local float oo[c::H];
        auto work = [&, K](int e, int) {
            float* o[1] = {oo};
            K->expert_multi(&arena[(size_t) e * c::BLOB], pa1, 1, o, ws);
        };
        stream_gbs(1, experts, 0.3, work);
        const double g1 = best_gbs(1, experts, seconds, 5, work);
        const double gn = best_gbs(threads, experts, seconds, 5, work);
        std::printf("%-44s %10.2f %10.2f\n", K == builds[0] ? "AVX-512 no-VBMI build (s2_expert_vnni_multi)" : "AVX-512 VBMI build    (s2_expert_vnni_multi)", g1, gn);
    }
    std::printf("\nIndicative only: %u hardware threads, shared with other work; the %d-thread figure is bounded by the VM's memory bandwidth.\n",
                std::thread::hardware_concurrency(), threads);
}

}  // namespace

int main(int argc, char** argv) {
    bool do_exhaustive = false, do_bench = false;
    int threads = 0;
    int bench_experts = 192;
    double seconds = 1.5;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--exhaustive") do_exhaustive = true;
        else if (a == "--bench") { do_bench = true; if (i + 1 < argc && argv[i + 1][0] != '-') threads = std::atoi(argv[++i]); }
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--experts" && i + 1 < argc) bench_experts = std::atoi(argv[++i]);
        else if (a == "--seconds" && i + 1 < argc) seconds = std::atof(argv[++i]);
        else { std::fprintf(stderr, "usage: expert_variant_test [--exhaustive] [--bench [T]] [--threads N] [--experts E] [--seconds S]\n"); return 2; }
    }
    const c::CpuFeatures f = c::cpu_features();
    std::printf("CPU: %s\n", c::cpu_name().c_str());
    std::printf("  AVX512 F/BW/VL/DQ/VNNI %s, VBMI %s -> tier %s%s\n", f.usable() ? "present" : "NOT present",
                f.avx512_vbmi ? "present" : "absent", c::expert_isa_name(c::cpu_q2_expert_isa()),
                std::getenv("STRATA_FORCE_AVX512_NOVBMI") ? " (STRATA_FORCE_AVX512_NOVBMI set)" : "");
    if (!f.usable()) {
        std::printf("  neither AVX-512 build can run here (%s): SKIPPED, not passed.\n", f.reason());
        return 0;
    }

    if (do_bench) { bench(threads > 0 ? threads : 4, bench_experts, seconds); return 0; }

    const c::ExpertKernels& nov = c::expert_kernels_novbmi();
    const c::ExpertKernels* vb = f.vbmi() ? &c::expert_kernels_vbmi() : nullptr;

    std::printf("\n2-bit unpack helpers against the scalar definition\n");
    for (const c::ExpertKernels* K : {&nov, vb}) {
        if (!K) { std::printf("  AVX-512 VNNI + VBMI build: SKIPPED (this CPU has no VBMI; run this test on an Ice Lake / Zen 4 machine)\n"); continue; }
        const Unpack u64{"unpack64", K->unpack64, 16, 64}, u32{"unpack32", K->unpack32, 8, 32};
        for (const Unpack& u : {u64, u32}) {
            bool ok = true;
            const long long n = check_unpack(u, 1 << 21, ok);
            char msg[120];
            std::snprintf(msg, sizeof msg, "%s %s: %lld cases (every byte value x position, patterns, random)", K->name, u.name, n);
            check(ok, msg);
        }
    }

    {   // The VBMI build's unpack as the instruction is SPECIFIED, run here as plain C so it can be checked on any CPU: qword i
        // of `lanes` is the zero-extended u16 codes[2i..2i+1]; output byte c of the multishift is the 8 bits of that qword starting
        // at bit ctrl[c] = 2c (a rotate, so it wraps - it never does for ctrl <= 14, data < 2^16); then & 3.  Exhaustive over every
        // u16 and every qword position, against the same scalar definition the hardware builds are held to.  (The hardware run
        // of the VBMI build is the `AVX-512 VNNI + VBMI build` lines above, on a CPU that has it.)
        long long bad = 0;
        for (uint32_t v = 0; v < 65536; ++v) {
            for (int i = 0; i < 8; ++i) {
                uint8_t in[16] = {};
                in[2 * i] = (uint8_t) v;
                in[2 * i + 1] = (uint8_t) (v >> 8);
                uint8_t ref[64];
                scalar_unpack(in, ref, 64);
                const uint64_t q = v;                                    // vpmovzxwq
                for (int cbyte = 0; cbyte < 8; ++cbyte) {
                    const int start = 2 * cbyte;                         // control byte 0x0E0C0A0806040200, byte cbyte
                    const uint64_t rot = start ? (q >> start) | (q << (64 - start)) : q;
                    if ((uint8_t) ((rot & 0xFF) & 3) != ref[8 * i + cbyte]) ++bad;
                }
            }
        }
        check(bad == 0, "vpmultishiftqb unpack (software model of the instruction): 2^16 values x 8 qwords");
    }

    {   // the checker itself: a unpack that gets the order within a byte backwards must be CAUGHT
        const Unpack bad{"reversed-order unpack", [](const uint8_t* in, uint8_t* out) {
            for (int j = 0; j < 64; ++j) out[j] = (uint8_t) ((in[j / 4] >> (2 * (3 - j % 4))) & 3);
        }, 16, 64};
        bool ok = true;
        check_unpack(bad, 1 << 10, ok);
        check(!ok, "harness: a deliberately wrong unpack is detected");
    }

    if (do_exhaustive) {
        std::printf("\nexhaustive: every packed dword value in every lane\n");
        if (threads <= 0) threads = 2;
        for (const c::ExpertKernels* K : {&nov, vb}) {
            if (!K) { std::printf("  AVX-512 VNNI + VBMI build: SKIPPED (no VBMI here)\n"); continue; }
            for (const Unpack& u : {Unpack{"unpack64", K->unpack64, 16, 64}, Unpack{"unpack32", K->unpack32, 8, 32}}) {
                char msg[120];
                std::snprintf(msg, sizeof msg, "%s %s: all 2^32 dwords x 4 (2 for unpack32) lanes", K->name, u.name);
                check(exhaustive(u, threads), msg);
            }
        }
    }

    std::printf("\nthe kernels of each build against the scalar oracle\n");
    check_vs_scalar(nov);
    if (vb) check_vs_scalar(*vb);

    std::printf("\nbuild against build, bit for bit\n");
    if (vb) compare_builds(nov, *vb);
    else std::printf("  SKIPPED (this CPU has no VBMI, so the VBMI build cannot run to be compared)\n");
    // and the no-VBMI build against itself, so the comparison harness itself is known to run clean
    std::printf("  (harness check: the no-VBMI build against itself)\n");
    compare_builds(nov, nov);

    std::printf("\nexpert_variant_test: %d failures\n", g_fail);
    if (g_fail == 0) std::printf("expert_variant_test OK\n");
    return g_fail ? 1 : 0;
}
