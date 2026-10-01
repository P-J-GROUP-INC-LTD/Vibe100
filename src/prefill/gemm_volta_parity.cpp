// src/prefill/gemm_volta_parity.cpp - Volta port, work package C: the FP16 tensor-core route of Gemm::bf16, on a GPU.
//
// usage: gemm_volta_parity [--selftest | --bench] [--seed S] [--scratch-mib M]
//
//   (no flag)        the correctness set, then the timing
//   --selftest       the correctness set only; exit status 1 if any check fails, 2 on a CUDA / usage error, 3 when
//                    there is no CUDA device; prints "gemm_volta_parity OK" when everything passed
//   --bench          the timing only: T = 8192, K = 2560, N = 2560 / 5120 / 10240 and the engine's own narrow shapes,
//                    the upstream bf16 route against the FP16 route, with TFLOPS
//   --seed S         the generator's seed (default 20260930)
//   --scratch-mib M  the Gemm scratch of the timing run (default 64: the engine's GEMM_SCRATCH)
//   STRATA_PREFILL_F16_GEMM=auto|0|1  is honoured by the "default route" check (a Gemm that follows the variable);
//                    every other check names its route explicitly (Gemm::set_f16_route), so the program runs on any
//                    CUDA card, not only a V100.
//
// WHAT IS CHECKED.  `Gemm::bf16` on cc 7.x converts its bf16 operands to FP16 on the device (exactly, with a per-chunk
// power-of-two scale so nothing can overflow) and runs the FP16 tensor-core GEMM with FP32 accumulation (gemm.cu has the
// design).  This program calls it on synthetic data with the route forced on (set_f16_route(2)) and then forced off (the
// upstream cuBLAS bf16 call, the same object, the same buffers, right after) and compares both with a host reference in
// FP64 over the SAME bf16 values, and with each other:
//
//   * shapes with N, K, T that are not multiples of 8, 16 or any chunk; ldy > N (the padding columns of Y and the
//     guard words around it must come back bit for bit); beta = 0 (over a Y full of NaN: it must not be read) and 1;
//   * the engine's own shapes (router, indexer, the hyper-connection read and write, the PLE value);
//   * chunking: a 2 MiB scratch forces T and N chunks, with an outlier in one chunk only (each chunk is scaled alone);
//   * adversarial values: an X row and a W row far above 65504, both huge, values in fp16's subnormal range, one huge
//     outlier among ordinary values, Inf and NaN in both operands;
//   * where the operands live: W, X or Y INSIDE the Gemm scratch (the route may only use the part it does not occupy,
//     and must leave the operands untouched), X at one end and W at the other (no room: it must fall back);
//   * a scratch too small / none at all (fallback), two calls back to back with no synchronization (the FP16 copies
//     and the device scalars are reused in stream order), the bf16 call right after an FP16 one (the handle's pointer
//     mode is restored), and the default route of a Gemm that follows STRATA_PREFILL_F16_GEMM.
//   Every case also checks that X and W come back unchanged and that the route counters say what was taken.
//
// THE THRESHOLD.  Both routes multiply the very same bf16 values: a bf16 has 8 significant bits and an fp16 11, so in
// fp16's normal range the conversion is exact, and each product is exact in FP32.  What is left is the FP32
// accumulation, whose error is proportional to what is being added, S = sum_k |x_ik w_jk| (+ |beta y_ij|) - not to the
// output, which cancels (an outlier row or column makes some outputs tiny next to the numbers that made them).
// Against S the noise is expected at ~1e-7 for the CUDA-core bf16 kernel and up to ~1e-6 for the tensor cores, whose
// accumulation truncates (an estimate, not a measurement - nothing here has run on a V100 yet: 8e-7 at K = 2560 and
// 1.5e-6 at K = 10240 if every truncation loses half an ulp).  An element passes if
//         |y - ref|  <=  1e-5 S  +  qb
// where qb = 2^-25 (2^kx sum_k|w_jk| + 2^kw sum_k|x_ik|) + K 2^-50 2^(kx+kw) is what rounding every element to fp16's
// subnormal grid after the scale 2^-kx / 2^-kw can cost (an element below 2^-14 after scaling keeps no more than the
// 2^-24 grid; chunks of ordinary values convert exactly, and their elements cost nothing).  1e-5 is above the noise by
// a factor of 6 or more, and below every error that would mean something: FP16 accumulation or output (>= 1e-2 of S),
// a lost mantissa bit (4e-3), a wrong scale, chunk offset or non-zero pad (>= 1e-2).  The two routes are compared with
// twice that.  Each line prints the error against S, against the largest output, and the share of the budget used, so
// the margin is visible.
//
// THE TIMING prints ms and TFLOPS (2 T N K) of the bf16 route and the FP16 route (conversions included) and of a bare
// cublasGemmEx on FP16 data (the HMMA ceiling for the shape).  More than ~35 TFLOPS can only be tensor cores (a V100's
// FP16 CUDA-core peak is 31 and its FP32 peak 15.7); ~80-100 is what a V100 reaches.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/prefill/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using strata::kernels::bf16_from_f32;
using strata::kernels::f32_from_bf16;
using strata::prefill::Gemm;

constexpr double kTolS = 1e-5;                 // the accumulation-noise budget, relative to S = sum_k |x w|
constexpr uint32_t kPoison = 0x7FC0DEADu;      // a NaN: what beta = 0 must never read
constexpr uint32_t kPad = 0xA5A5A5A5u;         // the padding columns of Y (a finite, tiny value)
constexpr uint32_t kGuard = 0xC3C3C3C3u;       // the words around Y
constexpr int64_t kGuardWords = 4096;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gemm_volta_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <class T>
struct Dev {
    T* p = nullptr;
    size_t n = 0;
    explicit Dev(size_t count) : n(count) { ck(cudaMalloc((void**) &p, std::max<size_t>(count, 1) * sizeof(T)), "cudaMalloc"); }
    ~Dev() { if (p) cudaFree(p); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    void up(const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); }
    std::vector<T> down() const {
        std::vector<T> v(n);
        ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
        return v;
    }
};

// ------------------------------------------------------------------------------------------------ data
std::vector<uint16_t> gauss(size_t n, double sigma, std::mt19937_64& rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint16_t> v(n);
    for (auto& x : v) x = bf16_from_f32((float) (nd(rng) * sigma));
    return v;
}
void scale_row(std::vector<uint16_t>& m, int64_t row, int64_t K, double f) {
    for (int64_t k = 0; k < K; ++k) m[(size_t) (row * K + k)] = bf16_from_f32((float) (f32_from_bf16(m[(size_t) (row * K + k)]) * f));
}
uint16_t bf(float f) { return bf16_from_f32(f); }

// k of the chunk (the route's rule, on the host): the smallest k >= 0 with max finite |v| * 2^-k < 2^15
int scale_k_of(const std::vector<uint16_t>& v) {
    uint32_t m = 0;
    for (uint16_t b : v) {
        const uint32_t mag = b & 0x7FFFu;
        if (mag < 0x7F80u) m = std::max(m, mag);
    }
    const uint32_t e = m >> 7;
    return e > 141u ? (int) (e - 141u) : 0;
}

// ------------------------------------------------------------------------------------------------ reference
// ref[t, n] = (beta != 0 ? beta * Y0[t, n] : 0) + sum_k X[t, k] W[n, k], in DOUBLE over the bf16 values (the products of
// two bf16 are exact in double; Inf and NaN follow IEEE), and sabs[t, n] = |beta Y0| + sum_k |X[t, k] W[n, k]|: what the
// rounding error of the sum scales with.  Threaded over the rows of X.
void reference(const std::vector<uint16_t>& X, const std::vector<uint16_t>& W, const std::vector<float>& Y0, float beta,
               int64_t T, int64_t N, int64_t K, std::vector<double>& ref, std::vector<double>& sabs) {
    std::vector<double> xd((size_t) (T * K)), wd((size_t) (N * K));
    for (size_t i = 0; i < xd.size(); ++i) xd[i] = f32_from_bf16(X[i]);
    for (size_t i = 0; i < wd.size(); ++i) wd[i] = f32_from_bf16(W[i]);
    ref.assign((size_t) (T * N), 0.0);
    sabs.assign((size_t) (T * N), 0.0);
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    auto work = [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; ++t) {
            const double* x = &xd[(size_t) (t * K)];
            for (int64_t n = 0; n < N; ++n) {
                const double* w = &wd[(size_t) (n * K)];
                double acc = 0.0, sa = 0.0;
                for (int64_t k = 0; k < K; ++k) {
                    const double p = x[k] * w[k];
                    acc += p;
                    sa += std::fabs(p);
                }
                if (beta != 0.0f) {
                    const double b = (double) beta * (double) Y0[(size_t) (t * N + n)];
                    acc += b;
                    sa += std::fabs(b);
                }
                ref[(size_t) (t * N + n)] = acc;
                sabs[(size_t) (t * N + n)] = sa;
            }
        }
    };
    std::vector<std::thread> th;
    const int64_t per = (T + nt - 1) / nt;
    for (unsigned i = 0; i < nt && (int64_t) i * per < T; ++i) th.emplace_back(work, (int64_t) i * per, std::min(T, (int64_t) (i + 1) * per));
    for (auto& t : th) t.join();
}

struct Scales {
    std::vector<double> sabs;         // [T, N]: |beta y| + sum_k |x w|
    std::vector<double> xsum, wsum;   // sum_k |x_ik| and sum_k |w_jk| over the finite values
    double gmax = 0;                  // the largest finite |ref|
    double qb_scale_x = 0, qb_scale_w = 0, qb_const = 0;   // 2^(kx-25), 2^(kw-25), K 2^-50 2^(kx+kw)
};
Scales make_scales(const std::vector<uint16_t>& X, const std::vector<uint16_t>& W, const std::vector<double>& ref,
                   const std::vector<double>& sabs, int64_t T, int64_t N, int64_t K) {
    Scales s;
    s.sabs = sabs;
    for (double r : ref)
        if (std::isfinite(r)) s.gmax = std::max(s.gmax, std::fabs(r));
    s.xsum.assign((size_t) T, 0.0);
    s.wsum.assign((size_t) N, 0.0);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t k = 0; k < K; ++k) {
            const double v = f32_from_bf16(X[(size_t) (t * K + k)]);
            if (std::isfinite(v)) s.xsum[(size_t) t] += std::fabs(v);
        }
    for (int64_t n = 0; n < N; ++n)
        for (int64_t k = 0; k < K; ++k) {
            const double v = f32_from_bf16(W[(size_t) (n * K + k)]);
            if (std::isfinite(v)) s.wsum[(size_t) n] += std::fabs(v);
        }
    const int kx = scale_k_of(X), kw = scale_k_of(W);
    s.qb_scale_x = std::ldexp(1.0, kx - 25);
    s.qb_scale_w = std::ldexp(1.0, kw - 25);
    s.qb_const = (double) K * std::ldexp(1.0, -50 + kx + kw);
    return s;
}
// the rounding-to-the-fp16-grid budget of element (t, n); zero for a route that does not convert
double qb_at(const Scales& s, int64_t t, int64_t n) {
    return s.qb_scale_x * s.wsum[(size_t) n] + s.qb_scale_w * s.xsum[(size_t) t] + s.qb_const;
}

struct Cmp {
    double rel_s = 0, rel_global = 0;       // max |y - ref| / S, and / the largest |ref|
    double use = 0;                         // max |y - ref| / budget: <= 1 passes
    int64_t class_bad = 0;                  // elements finite in one and not in the other
};
// y against `ref` (the FP64 reference, or the other route's output): `budget_k` times kTolS of S, plus qb when `with_qb`.
// Non-finite elements must be non-finite in both (Inf against NaN is not distinguished).
Cmp compare(const std::vector<float>& y, const std::vector<double>& ref, const Scales& s, int64_t T, int64_t N,
            double budget_k, bool with_qb) {
    Cmp c;
    const double inf = std::numeric_limits<double>::infinity();
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) {
            const size_t i = (size_t) (t * N + n);
            const double r = ref[i], v = y[i];
            const bool rf = std::isfinite(r), vf = std::isfinite(v);
            if (rf != vf) { ++c.class_bad; continue; }
            if (!rf) continue;
            const double err = std::fabs(v - r), sa = s.sabs[i];
            c.rel_s = std::max(c.rel_s, sa > 0 ? err / sa : (err > 0 ? inf : 0.0));
            c.rel_global = std::max(c.rel_global, s.gmax > 0 ? err / s.gmax : (err > 0 ? inf : 0.0));
            const double budget = budget_k * kTolS * sa + (with_qb ? qb_at(s, t, n) : 0.0);
            c.use = std::max(c.use, budget > 0 ? err / budget : (err > 0 ? inf : 0.0));
        }
    return c;
}

// ------------------------------------------------------------------------------------------------ the cases
enum Place { kPlain, kOddPtr, kWStart, kWEnd, kXMid, kXStartWEnd, kYStart };
enum Which { kBig, kSmall, kTiny, kNone };   // which Gemm: 64 MiB scratch, 2 MiB, 8 KiB, none

struct Case {
    std::string name;
    int64_t T, N, K, ldy;                   // ldy 0: the default (N)
    float beta;
    Which which;
    Place place;
    bool expect_route = true;               // the FP16 route is taken (else: the bf16 call, counted as a fallback)
    int64_t min_tiles = 1;
    double sx = 1.0, sw = 0.05;             // the standard deviations of X and W
    std::function<void(std::vector<uint16_t>&, std::vector<uint16_t>&)> shape;   // edits the operands afterwards
    const char* note = "";
    Case(std::string name_, int64_t T_, int64_t N_, int64_t K_, int64_t ldy_, float beta_, Which which_ = kBig,
         Place place_ = kPlain)
        : name(std::move(name_)), T(T_), N(N_), K(K_), ldy(ldy_), beta(beta_), which(which_), place(place_) {}
};

struct Rig {
    cudaStream_t stream = nullptr;
    Gemm gemm[4];
    int64_t scratch_elems[4] = {32ll << 20, 1ll << 20, 4096, 0};
    void init() {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");   // non-blocking: a launch that strays to the
        for (int i = 0; i < 4; ++i) {                                              // default stream would race, not serialize
            std::string err;
            if (!gemm[i].init(stream, scratch_elems[i], err)) {
                std::fprintf(stderr, "gemm_volta_parity: Gemm init: %s\n", err.c_str());
                std::exit(2);
            }
        }
    }
};

const char* place_name(Place p) {
    switch (p) {
        case kOddPtr: return "X and W 6 bytes off 16-byte alignment";
        case kWStart: return "W at the scratch start";
        case kWEnd: return "W at the scratch end";
        case kXMid: return "X in the scratch middle";
        case kXStartWEnd: return "X at the start and W at the end";
        case kYStart: return "Y at the scratch start";
        default: return "";
    }
}

// one route: Y initialized to `yinit`, bf16() called, then the counters, the words around Y, the padding columns and the
// operands checked.  `ybuf` is the first element of Y (kGuardWords words of guard precede it).  false: a check failed
// here (`why` says which); `out` is the compact [T, N] result.
bool run_route(Rig& rig, Gemm& g, int route, const Case& c, const uint16_t* dX, const uint16_t* dW, uint32_t* ybuf,
               const std::vector<uint32_t>& yinit, int64_t ldy, std::vector<float>& out, const std::vector<uint16_t>& hX,
               const std::vector<uint16_t>& hW, std::string& why) {
    const int64_t T = c.T, N = c.N, K = c.K;
    ck(cudaMemcpy(ybuf - kGuardWords, yinit.data(), yinit.size() * 4, cudaMemcpyHostToDevice), "upload Y");
    g.set_f16_route(route);
    const int64_t calls0 = g.f16_calls(), fb0 = g.f16_fallbacks(), tiles0 = g.f16_tiles();
    g.bf16(dX, dW, (float*) ybuf, T, N, K, c.ldy, c.beta);
    ck(cudaStreamSynchronize(rig.stream), "bf16");
    ck(cudaGetLastError(), "bf16 launch");
    const int64_t calls = g.f16_calls() - calls0, fb = g.f16_fallbacks() - fb0, tiles = g.f16_tiles() - tiles0;
    if (route == 2) {
        if (c.expect_route && (calls != 1 || fb != 0 || tiles < c.min_tiles)) {
            why += " [route counters: calls " + std::to_string(calls) + " fallbacks " + std::to_string(fb) + " tiles " +
                   std::to_string(tiles) + ", expected the FP16 route with >= " + std::to_string(c.min_tiles) + " tiles]";
            return false;
        }
        if (!c.expect_route && (calls != 0 || fb != 1 || tiles != 0)) {
            why += " [route counters: calls " + std::to_string(calls) + " fallbacks " + std::to_string(fb) +
                   ", expected a fallback to the bf16 call]";
            return false;
        }
    } else if (calls != 0 || fb != 0 || tiles != 0) {
        why += " [the bf16 route moved the FP16 counters]";
        return false;
    }
    // Y: the words around it and the padding columns come back bit for bit; the block's own values go out
    std::vector<uint32_t> y(yinit.size());
    ck(cudaMemcpy(y.data(), ybuf - kGuardWords, y.size() * 4, cudaMemcpyDeviceToHost), "download Y");
    bool ok = true;
    // Y inside the scratch: the words after the last row's N columns are the stretch the route may use
    const bool in_scratch = c.place == kYStart;
    if (!in_scratch) {
        for (int64_t i = 0; i < kGuardWords && ok; ++i)
            if (y[(size_t) i] != yinit[(size_t) i] || y[(size_t) (kGuardWords + T * ldy + i)] != yinit[(size_t) (kGuardWords + T * ldy + i)]) {
                ok = false;
                why += " [the guard words around Y changed]";
            }
    }
    for (int64_t t = 0; t < (in_scratch ? T - 1 : T) && ok; ++t)
        for (int64_t n = N; n < ldy; ++n)
            if (y[(size_t) (kGuardWords + t * ldy + n)] != yinit[(size_t) (kGuardWords + t * ldy + n)]) {
                ok = false;
                why += " [a padding column of Y was written (row " + std::to_string(t) + ")]";
                break;
            }
    out.assign((size_t) (T * N), 0.0f);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < N; ++n) std::memcpy(&out[(size_t) (t * N + n)], &y[(size_t) (kGuardWords + t * ldy + n)], 4);
    // X and W are only read: bit for bit (they may live IN the scratch)
    std::vector<uint16_t> x(hX.size()), w(hW.size());
    ck(cudaMemcpy(x.data(), dX, x.size() * 2, cudaMemcpyDeviceToHost), "download X");
    ck(cudaMemcpy(w.data(), dW, w.size() * 2, cudaMemcpyDeviceToHost), "download W");
    if (std::memcmp(x.data(), hX.data(), x.size() * 2) != 0) { ok = false; why += " [X was modified]"; }
    if (std::memcmp(w.data(), hW.data(), w.size() * 2) != 0) { ok = false; why += " [W was modified]"; }
    return ok;
}

bool run_case(Rig& rig, const Case& c, std::mt19937_64& rng) {
    const int64_t T = c.T, N = c.N, K = c.K;
    const int64_t ldy = c.ldy > 0 ? c.ldy : N;
    Gemm& g = rig.gemm[c.which];
    std::vector<uint16_t> hX = gauss((size_t) (T * K), c.sx, rng), hW = gauss((size_t) (N * K), c.sw, rng);
    if (c.shape) c.shape(hX, hW);
    std::vector<float> Y0((size_t) (T * N));
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : Y0) v = nd(rng);
    }
    // Y: [guard | T rows of ldy | guard] as raw words
    std::vector<uint32_t> yinit((size_t) (T * ldy + 2 * kGuardWords), kGuard);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t n = 0; n < ldy; ++n) {
            uint32_t w = kPad;
            if (n < N) {
                if (c.beta == 0.0f) w = kPoison;
                else std::memcpy(&w, &Y0[(size_t) (t * N + n)], 4);
            }
            yinit[(size_t) (kGuardWords + t * ldy + n)] = w;
        }
    std::vector<double> ref, sabs;
    reference(hX, hW, Y0, c.beta, T, N, K, ref, sabs);
    const Scales sc = make_scales(hX, hW, ref, sabs, T, N, K);

    // where the operands live
    uint16_t* ws = g.scratch();
    const int64_t se = g.scratch_elems();
    std::unique_ptr<Dev<uint16_t>> bx, bw;
    std::unique_ptr<Dev<uint32_t>> by;
    uint16_t *dX = nullptr, *dW = nullptr;
    uint32_t* ybuf = nullptr;
    const int64_t xe = T * K, we = N * K, ye = (T * ldy + 2 * kGuardWords) * 2;   // in 16-bit elements
    switch (c.place) {
        case kPlain: case kOddPtr: break;
        case kWStart: dW = ws; break;
        case kWEnd: dW = ws + se - we - 3; break;           // 3 elements from the end: not 16-byte aligned, and the room is before it
        case kXMid: dX = ws + (se - xe) / 2 + 1; break;     // odd: 2-byte aligned only
        case kXStartWEnd: dX = ws; dW = ws + se - we; break;
        case kYStart: ybuf = (uint32_t*) ws + kGuardWords; break;
    }
    if ((dX && (xe > se)) || (dW && (we > se)) || (ybuf && ye > se)) {
        std::fprintf(stderr, "gemm_volta_parity: case '%s' does not fit the scratch\n", c.name.c_str());
        return false;
    }
    const size_t odd = c.place == kOddPtr ? 3 : 0;      // elements: 6 bytes
    if (!dX) { bx = std::make_unique<Dev<uint16_t>>((size_t) xe + 8); dX = bx->p + odd; }
    if (!dW) { bw = std::make_unique<Dev<uint16_t>>((size_t) we + 8); dW = bw->p + odd; }
    if (!ybuf) { by = std::make_unique<Dev<uint32_t>>(yinit.size()); ybuf = by->p + kGuardWords; }
    ck(cudaMemcpy(dX, hX.data(), hX.size() * 2, cudaMemcpyHostToDevice), "upload X");
    ck(cudaMemcpy(dW, hW.data(), hW.size() * 2, cudaMemcpyHostToDevice), "upload W");

    std::string why;
    std::vector<float> y16, ybf;
    bool ok = true;
    // the FP16 route first, then the upstream route on the same handle: if the FP16 route had left the handle in device
    // pointer mode, the bf16 call (a host &alpha) would fail or read garbage
    ok &= run_route(rig, g, 2, c, dX, dW, ybuf, yinit, ldy, y16, hX, hW, why);
    ok &= run_route(rig, g, 0, c, dX, dW, ybuf, yinit, ldy, ybf, hX, hW, why);
    if (ok) {
        const Cmp a = compare(y16, ref, sc, T, N, 1.0, true);
        const Cmp b = compare(ybf, ref, sc, T, N, 1.0, false);
        std::vector<double> bref(ybf.begin(), ybf.end());
        const Cmp d = compare(y16, bref, sc, T, N, 2.0, true);
        const bool pass = a.use <= 1.0 && b.use <= 1.0 && d.use <= 1.0 && a.class_bad == 0 && b.class_bad == 0 && d.class_bad == 0;
        std::printf("%s  %-34s T=%-5lld N=%-5lld K=%-5lld ldy=%-5lld beta=%g%s%s\n"
                    "      error / S, / max|y| (budget used): fp16 route %.1e, %.1e (%.3f) | bf16 route %.1e, %.1e (%.3f) | fp16 vs bf16 %.1e, %.1e (%.3f)%s%s\n",
                    pass ? "PASS" : "FAIL", c.name.c_str(), (long long) T, (long long) N, (long long) K, (long long) ldy,
                    (double) c.beta, c.place != kPlain ? "  " : "", place_name(c.place), a.rel_s, a.rel_global, a.use,
                    b.rel_s, b.rel_global, b.use, d.rel_s, d.rel_global, d.use, c.note[0] ? "  # " : "", c.note);
        if (a.class_bad || b.class_bad || d.class_bad)
            std::printf("      non-finite mismatch: fp16 %lld, bf16 %lld, between %lld\n", (long long) a.class_bad, (long long) b.class_bad, (long long) d.class_bad);
        return pass;
    }
    std::printf("FAIL  %-34s T=%lld N=%lld K=%lld ldy=%lld beta=%g:%s\n", c.name.c_str(), (long long) T, (long long) N,
                (long long) K, (long long) ldy, (double) c.beta, why.c_str());
    return false;
}

// two FP16-route calls with no synchronization between them: the second reuses the first's FP16 copies and device
// scalars in stream order (the first one needs a scale, the second does not)
bool run_back_to_back(Rig& rig, std::mt19937_64& rng) {
    const int64_t T = 200, N = 150, K = 600;
    Gemm& g = rig.gemm[kBig];
    std::vector<uint16_t> X1 = gauss((size_t) (T * K), 1.0, rng), W1 = gauss((size_t) (N * K), 0.05, rng);
    std::vector<uint16_t> X2 = gauss((size_t) (T * K), 1.0, rng), W2 = gauss((size_t) (N * K), 0.05, rng);
    scale_row(X1, 11, K, 4e6);
    scale_row(W1, 9, K, 3e6);
    std::vector<float> Y0((size_t) (T * N), 0.0f);
    std::vector<double> r1, r2, a1, a2;
    reference(X1, W1, Y0, 0.0f, T, N, K, r1, a1);
    reference(X2, W2, Y0, 0.0f, T, N, K, r2, a2);
    Dev<uint16_t> dX1((size_t) (T * K)), dW1((size_t) (N * K)), dX2((size_t) (T * K)), dW2((size_t) (N * K));
    Dev<float> dY1((size_t) (T * N)), dY2((size_t) (T * N));
    dX1.up(X1); dW1.up(W1); dX2.up(X2); dW2.up(W2);
    g.set_f16_route(2);
    g.bf16(dX1.p, dW1.p, dY1.p, T, N, K);
    g.bf16(dX2.p, dW2.p, dY2.p, T, N, K);
    ck(cudaStreamSynchronize(rig.stream), "back to back");
    ck(cudaGetLastError(), "back to back launch");
    const Scales s1 = make_scales(X1, W1, r1, a1, T, N, K), s2 = make_scales(X2, W2, r2, a2, T, N, K);
    const Cmp a = compare(dY1.down(), r1, s1, T, N, 1.0, true), b = compare(dY2.down(), r2, s2, T, N, 1.0, true);
    const bool pass = a.use <= 1.0 && b.use <= 1.0 && a.class_bad == 0 && b.class_bad == 0;
    std::printf("%s  %-34s T=%lld N=%lld K=%lld\n      error / S: the scaled call %.1e (budget used %.3f) | the call after it, unscaled, %.1e (%.3f)\n",
                pass ? "PASS" : "FAIL", "two calls, no sync between", (long long) T, (long long) N, (long long) K, a.rel_s, a.use, b.rel_s, b.use);
    return pass;
}

// the Gemm that follows STRATA_PREFILL_F16_GEMM: which route does a big shape take, a small one, and are the results right
bool run_default_route(std::mt19937_64& rng) {
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    Gemm g;
    std::string err;
    if (!g.init(st, 32ll << 20, err)) { std::fprintf(stderr, "gemm_volta_parity: Gemm init: %s\n", err.c_str()); std::exit(2); }
    int dev = 0, major = 0, minor = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
    ck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "cc");
    const int cc = major * 10 + minor;
    const char* env = std::getenv("STRATA_PREFILL_F16_GEMM");
    const std::string e = env ? env : "";
    bool expect_big;
    if (e == "0") expect_big = false;
    else if (e == "1") expect_big = true;
    else expect_big = cc >= 70 && cc < 80;
    struct Shape { int64_t T, N, K; bool expect; const char* what; };
    const Shape shapes[] = {{1024, 1024, 1024, expect_big, "a big shape"},
                            {1024, 1, 2560, false, "the N = 1 router gate (stays upstream whatever the variable)"},
                            {32, 4096, 2560, false, "T = 32 (a draft layer: stays upstream)"},
                            {1024, 48, 2560, false, "N = 48 (GDN gates: stays upstream)"}};
    bool pass = true;
    for (const Shape& s : shapes) {
        std::vector<uint16_t> X = gauss((size_t) (s.T * s.K), 1.0, rng), W = gauss((size_t) (s.N * s.K), 0.05, rng);
        std::vector<float> Y0;
        std::vector<double> ref, sabs;
        reference(X, W, Y0, 0.0f, s.T, s.N, s.K, ref, sabs);
        Dev<uint16_t> dX(X.size()), dW(W.size());
        Dev<float> dY((size_t) (s.T * s.N));
        dX.up(X); dW.up(W);
        const int64_t c0 = g.f16_calls();
        g.bf16(dX.p, dW.p, dY.p, s.T, s.N, s.K);
        ck(cudaStreamSynchronize(st), "bf16");
        const bool routed = g.f16_calls() - c0 == 1;
        const Scales sc = make_scales(X, W, ref, sabs, s.T, s.N, s.K);
        const Cmp a = compare(dY.down(), ref, sc, s.T, s.N, 1.0, true);
        const bool ok = routed == s.expect && a.use <= 1.0 && a.class_bad == 0;
        pass &= ok;
        std::printf("%s  default route, %-58s: %s (expected %s)  error / S %.1e\n", ok ? "PASS" : "FAIL",
                    s.what, routed ? "FP16" : "bf16", s.expect ? "FP16" : "bf16", a.rel_s);
    }
    std::printf("      cc %d.%d, STRATA_PREFILL_F16_GEMM=%s\n", major, minor, env ? env : "(unset)");
    cudaStreamDestroy(st);
    return pass;
}

std::vector<Case> make_cases() {
    std::vector<Case> v;
    auto add = [&](Case c) { v.push_back(std::move(c)); };
    // ---- shapes: nothing a multiple of anything, ldy > N, beta 0 and 1
    add({"aligned", 256, 192, 512, 0, 0.0f});
    add({"odd, ldy > N", 203, 131, 517, 136, 0.0f});
    add({"odd, ldy > N, beta = 1", 203, 131, 517, 140, 1.0f});
    add({"degenerate 1 x 1 x 1", 1, 1, 1, 0, 0.0f});
    add({"degenerate 3 x 2 x 9, ldy 5, beta 1", 3, 2, 9, 5, 1.0f});
    add({"T = 65 N = 64 K = 8", 65, 64, 8, 0, 0.0f});
    // ---- the engine's shapes (router, indexer, GDN gates' stride, PLE value, hyper-connection down / up)
    add({"router [512, 2560]", 1024, 512, 2560, 0, 0.0f});
    add({"indexer.k_proj [128, 2560]", 777, 128, 2560, 0, 0.0f});
    add({"ssm_alpha [48, 2560], ldy 96", 1000, 48, 2560, 96, 0.0f});
    add({"ple_value [2560, 2560]", 256, 2560, 2560, 0, 0.0f});
    add({"hc down [320, 10240]", 256, 320, 10240, 0, 0.0f});
    add({"hc up [10240, 320], beta = 1", 256, 10240, 320, 10240, 1.0f});
    // ---- chunking: a 2 MiB scratch holds ~1000 rows of K = 1000; an outlier in one chunk only
    {
        Case c{"chunked T and N, beta = 1", 1500, 900, 1000, 903, 1.0f, kSmall};
        c.min_tiles = 4;
        c.shape = [](std::vector<uint16_t>& X, std::vector<uint16_t>& W) { scale_row(X, 1400, 1000, 3e6); scale_row(W, 800, 1000, 5e6); };
        c.note = "X row 1400 and W row 800 are outliers: only their chunks are scaled";
        add(c);
    }
    {
        Case c{"chunked T only", 3000, 100, 1000, 0, 0.0f, kSmall};
        c.min_tiles = 3;
        c.shape = [](std::vector<uint16_t>& X, std::vector<uint16_t>&) { scale_row(X, 2900, 1000, 1e7); };
        add(c);
    }
    {
        Case c{"chunked N only, beta = 1", 100, 3000, 1000, 3010, 1.0f, kSmall};
        c.min_tiles = 3;
        c.shape = [](std::vector<uint16_t>&, std::vector<uint16_t>& W) { scale_row(W, 2950, 1000, 1e7); };
        add(c);
    }
    // ---- adversarial values
    {
        Case c{"X row above 65504", 96, 130, 520, 0, 0.0f};
        c.shape = [](std::vector<uint16_t>& X, std::vector<uint16_t>&) { scale_row(X, 5, 520, 5e6); };
        c.note = "row 5 ~ 5e6: an unscaled fp16 copy would be all Inf";
        add(c);
    }
    {
        Case c{"W row above 65504", 96, 130, 520, 0, 0.0f};
        c.shape = [](std::vector<uint16_t>&, std::vector<uint16_t>& W) { scale_row(W, 7, 520, 2e7); };
        add(c);
    }
    {
        Case c{"X and W both large", 96, 130, 520, 0, 0.0f};
        c.sx = 3e4;
        c.sw = 1e5;
        c.note = "both chunks scaled: alpha = 2^(kx+kw) = 2^6";
        add(c);
    }
    {
        Case c{"tiny: fp16 subnormal range", 96, 130, 520, 0, 0.0f};
        c.sx = 3e-6;
        c.sw = 2e-5;
        c.note = "every element is on the 2^-24 grid: the error is bounded by qb, not small";
        add(c);
    }
    {
        Case c{"one outlier among ordinary", 96, 130, 520, 0, 0.0f};
        c.shape = [](std::vector<uint16_t>& X, std::vector<uint16_t>&) { X[(size_t) (3 * 520 + 17)] = bf(1e8f); };
        c.note = "X[3][17] = 1e8: k = 12, the chunk's small elements lose bits";
        add(c);
    }
    {
        Case c{"Inf and NaN in both operands", 96, 130, 520, 0, 0.0f};
        c.shape = [](std::vector<uint16_t>& X, std::vector<uint16_t>& W) {
            X[(size_t) (3 * 520 + 11)] = 0x7F80;                       // +Inf
            X[(size_t) (9 * 520 + 20)] = 0x7FC1;                       // NaN
            X[(size_t) (20 * 520 + 5)] = bf(3e7f);                     // a finite value that sets the scale
            W[(size_t) (40 * 520 + 13)] = 0xFF80;                      // -Inf
        };
        c.note = "rows 3, 9 and column 40 non-finite in both routes; the other rows exact";
        add(c);
    }
    {
        Case c{"X zero, beta = 1", 64, 64, 64, 80, 1.0f};
        c.sx = 0.0;
        c.note = "max = 0, k = 0: Y comes back as beta * Y";
        add(c);
    }
    // ---- where the operands live
    add({"operands off 16-byte alignment", 150, 131, 520, 0, 0.0f, kBig, kOddPtr});
    add({"W inside the scratch (start)", 300, 200, 700, 0, 0.0f, kBig, kWStart});
    add({"W inside the scratch (end)", 300, 200, 700, 205, 1.0f, kBig, kWEnd});
    add({"X inside the scratch (middle)", 300, 200, 700, 0, 0.0f, kBig, kXMid});
    add({"Y inside the scratch (start)", 300, 200, 700, 210, 0.0f, kBig, kYStart});
    {
        Case c{"X at one end, W at the other", 300, 200, 700, 0, 0.0f, kBig, kXStartWEnd};
        c.expect_route = false;
        c.note = "no room left: falls back to the bf16 call";
        add(c);
    }
    {
        Case c{"scratch of 8 KiB", 300, 200, 700, 0, 0.0f, kTiny};
        c.expect_route = false;
        add(c);
    }
    {
        Case c{"no scratch", 300, 200, 700, 0, 1.0f, kNone};
        c.expect_route = false;
        add(c);
    }
    return v;
}

int correctness(uint64_t seed) {
    Rig rig;
    rig.init();
    std::mt19937_64 rng(seed);
    int bad = 0, total = 0;
    for (const Case& c : make_cases()) {
        ++total;
        if (!run_case(rig, c, rng)) ++bad;
    }
    ++total;
    if (!run_back_to_back(rig, rng)) ++bad;
    ++total;
    if (!run_default_route(rng)) ++bad;
    std::printf("%d of %d correctness checks failed\n", bad, total);
    return bad;
}

// ------------------------------------------------------------------------------------------------ timing
double time_ms(cudaStream_t st, int iters, const std::function<void()>& fn) {
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    for (int i = 0; i < 2; ++i) fn();
    ck(cudaStreamSynchronize(st), "warm up");
    ck(cudaEventRecord(e0, st), "event");
    for (int i = 0; i < iters; ++i) fn();
    ck(cudaEventRecord(e1, st), "event");
    ck(cudaEventSynchronize(e1), "timing");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, e0, e1), "timing");
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

int bench(uint64_t seed, int64_t scratch_mib) {
    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "stream");
    Gemm g;
    std::string err;
    if (!g.init(st, scratch_mib << 19, err)) { std::fprintf(stderr, "gemm_volta_parity: Gemm init: %s\n", err.c_str()); return 2; }
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS || cublasSetStream(h, st) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "gemm_volta_parity: cublasCreate failed\n");
        return 2;
    }
    struct Shape { int64_t T, N, K; const char* what; };
    const Shape shapes[] = {{8192, 2560, 2560, "square"},
                            {8192, 5120, 2560, ""},
                            {8192, 10240, 2560, ""},
                            {8192, 512, 2560, "router"},
                            {8192, 320, 10240, "hc down"},
                            {8192, 10240, 320, "hc up"}};
    // a pool of random values, tiled: the timing does not depend on the data (much), the host generator would
    std::mt19937_64 rng(seed);
    const std::vector<uint16_t> pool_bf = gauss(1 << 20, 1.0, rng);
    std::vector<uint16_t> pool_f16(pool_bf.size());
    for (size_t i = 0; i < pool_bf.size(); ++i) pool_f16[i] = strata::kernels::f16_from_f32(f32_from_bf16(pool_bf[i]));
    auto fill = [&](Dev<uint16_t>& d, const std::vector<uint16_t>& pool, size_t off) {
        std::vector<uint16_t> v(d.n);
        for (size_t i = 0; i < v.size(); ++i) v[i] = pool[(i * 7919 + off) % pool.size()];
        d.up(v);
    };
    std::printf("%-28s | %-21s | %-40s | %-24s | %s\n", "shape (T, N, K)", "bf16 route (cuBLAS)",
                "FP16 route (converted + HMMA), speed-up", "bare FP16 cublasGemmEx", "first 64 rows, max |diff| / max |y|");
    bool hmma_seen = false;
    int bad = 0;
    for (const Shape& s : shapes) {
        const int64_t T = s.T, N = s.N, K = s.K;
        Dev<uint16_t> dX((size_t) (T * K)), dW((size_t) (N * K)), dX16((size_t) (T * K)), dW16((size_t) (N * K));
        Dev<float> dY0((size_t) (T * N)), dY1((size_t) (T * N));
        fill(dX, pool_bf, 0); fill(dW, pool_bf, 5);
        fill(dX16, pool_f16, 0); fill(dW16, pool_f16, 5);
        const double flop = 2.0 * (double) T * (double) N * (double) K;
        g.set_f16_route(0);
        const double ms_bf = time_ms(st, 6, [&] { g.bf16(dX.p, dW.p, dY0.p, T, N, K); });
        g.set_f16_route(1);
        const int64_t tiles0 = g.f16_tiles(), calls0 = g.f16_calls();
        const double ms_f16 = time_ms(st, 6, [&] { g.bf16(dX.p, dW.p, dY1.p, T, N, K); });
        const int64_t routed = g.f16_calls() - calls0;     // 0: the FP16 route was not taken (a shape below the threshold, or no room)
        const int64_t tiles = (g.f16_tiles() - tiles0) / std::max<int64_t>(routed, 1);
        const float one = 1.0f, zero = 0.0f;
        const double ms_raw = time_ms(st, 6, [&] {
            if (cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &one, dW16.p, CUDA_R_16F, (int) K, dX16.p,
                             CUDA_R_16F, (int) K, &zero, dY1.p, CUDA_R_32F, (int) N, CUBLAS_COMPUTE_32F,
                             CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "gemm_volta_parity: cublasGemmEx failed\n");
                std::exit(2);
            }
        });
        // the two routes' outputs on the first rows (bf16 route's Y0, FP16 route's Y1 - recompute it: the raw call overwrote it)
        g.set_f16_route(1);
        g.bf16(dX.p, dW.p, dY1.p, T, N, K);
        ck(cudaStreamSynchronize(st), "compare");
        const int64_t rows = std::min<int64_t>(T, 64);
        std::vector<float> a((size_t) (rows * N)), b((size_t) (rows * N));
        ck(cudaMemcpy(a.data(), dY0.p, a.size() * 4, cudaMemcpyDeviceToHost), "download");
        ck(cudaMemcpy(b.data(), dY1.p, b.size() * 4, cudaMemcpyDeviceToHost), "download");
        double diff = 0, mx = 0;
        for (size_t i = 0; i < a.size(); ++i) { diff = std::max(diff, (double) std::fabs(a[i] - b[i])); mx = std::max(mx, (double) std::fabs(a[i])); }
        const double tf_bf = flop / (ms_bf * 1e-3) / 1e12, tf_f16 = flop / (ms_f16 * 1e-3) / 1e12, tf_raw = flop / (ms_raw * 1e-3) / 1e12;
        hmma_seen |= tf_raw > 35.0;
        const bool agree = diff <= 2e-4 * mx;
        bad += agree ? 0 : 1;
        char shape[64];
        std::snprintf(shape, sizeof shape, "%lld, %lld, %lld", (long long) T, (long long) N, (long long) K);
        char tl[32];
        if (routed == 0) std::snprintf(tl, sizeof tl, "NOT TAKEN");
        else std::snprintf(tl, sizeof tl, "%2lld tile%s", (long long) tiles, tiles == 1 ? " " : "s");
        std::printf("%-28s | %7.2f ms %6.1f TFLOPS | %7.2f ms %6.1f TFLOPS %-9s %4.2fx | %7.2f ms %6.1f TFLOPS | %s %.1e%s%s\n",
                    shape, ms_bf, tf_bf, ms_f16, tf_f16, tl, ms_bf / ms_f16, ms_raw, tf_raw,
                    agree ? "ok" : "DIFFER", mx > 0 ? diff / mx : 0.0, s.what[0] ? "  " : "", s.what);
    }
    std::printf("bare FP16 cublasGemmEx %s 35 TFLOPS: %s\n", hmma_seen ? ">" : "<=",
                hmma_seen ? "the tensor cores are in use" : "NOT the tensor-core rate - check the card and the cuBLAS build");
    cublasDestroy(h);
    cudaStreamDestroy(st);
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, do_bench = false;
    uint64_t seed = 20260930ull;
    int64_t scratch_mib = 64;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--seed" && i + 1 < argc) seed = (uint64_t) std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--scratch-mib" && i + 1 < argc) scratch_mib = std::max<int64_t>(1, std::atoll(argv[++i]));
        else {
            std::fprintf(stderr, "usage: gemm_volta_parity [--selftest | --bench] [--seed S] [--scratch-mib M]\n");
            return 2;
        }
    }
    if (selftest && do_bench) {
        std::fprintf(stderr, "usage: gemm_volta_parity [--selftest | --bench] [--seed S] [--scratch-mib M]\n");
        return 2;
    }
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        std::fprintf(stderr, "gemm_volta_parity: no CUDA device\n");
        return 3;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "device properties");
    std::printf("device 0: %s, cc %d.%d, %d SMs, %.1f GiB\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                (double) prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    int bad = 0;
    if (!do_bench) bad += correctness(seed);
    if (!selftest && bad == 0) bad += bench(seed, scratch_mib) != 0 ? 1 : 0;
    if (bad) {
        std::printf("gemm_volta_parity FAILED\n");
        return 1;
    }
    std::printf("gemm_volta_parity OK\n");
    return 0;
}
