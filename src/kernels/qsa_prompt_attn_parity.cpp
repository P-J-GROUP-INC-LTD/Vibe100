// src/kernels/qsa_prompt_attn_parity.cpp - perf-review D-1: the tensor-core prompt attention (qsa_prompt_attn.hpp)
// against the FP32 kernel it replaces (`qsa_decode_attn_batch`) and an FP64 host reference (GPU, synthetic, no model).
//
// Int8, FP16 and hybrid K8V4 pools (KV modes 1, 0 and 3) with random codes, scales and queries; selections shaped like
// the prompt path's (the 2,051 widest, a recent window plus older cells that drift slowly from one query to the next,
// so neighbours share most of them as they do in a real prompt; short contexts take every cell). Checks:
//   1. against FP64, the new kernel's error is no larger than a small multiple of the old kernel's (both FP32 math);
//   2. the new and old outputs agree to a relative 1e-4 of the output scale;
//   3. every output is finite (the output buffer is NaN-filled first, so a row the kernel failed to write shows up);
// then times both over a prompt chunk (the old one in batches of 32, as prefill.cpp calls it).
//
// K8V4's V does not enter the kernel exactly: it rounds (n - 8) * d to fp16 where the FP32 kernel and the host
// reference keep it in FP32 (qsa_prompt_attn.cu above `Smem`; upstream's v1 kernel does the same).  Two K8V4 cases:
//   k8v4                    q4_0 scales d = k / 1024, k <= 31: (n - 8) * d has <= 8 significant bits, so it is exact in
//                           fp16 and the rounding adds nothing: the checks above stay as tight as for the other modes.
//   k8v4 (real q4_0 scales) d from real-looking data: blocks of 32 N(0, sigma) values (sigma log-uniform over 0.1..2 per
//                           row), quantized by kv_q4.hpp's rule (d = max / -8, rounded to fp16, so an arbitrary 11-bit
//                           significand; codes (int) (x / d + 8.5)).  (n - 8) * d then needs up to 15 bits and the fp16
//                           rounding is real.  Budget, derived from its bound: round-to-nearest gives |v^ - v| <=
//                           2^-11 |v| (2^-25 absolute below 2^-14, the fp16 subnormals), an output element is
//                           sum_c p_c v_c with the softmax weights p_c (sum 1), so it can move by at most
//                           B = sum_c p_c (2^-11 |v_c| + 2^-25)  <=  2^-11 max|v| + 2^-25
//                           - a hard bound (every rounding aligned with its weight's sign), nearly reached when one cell
//                           dominates the softmax (as with this data: random keys make it peaked) and ~1/sqrt(N_eff) of
//                           it when N_eff cells share the weight.  Check 1 then holds the part of the new kernel's error
//                           that is BEYOND its own B (B computed on the host per sampled element from the FP64 weights)
//                           to the same limit as the other modes - the kernel's FP32-level error is what remains, and
//                           what 4x the old kernel's covers - and check 2 gets 2^-11 max|v| + 2^-25 (max over the pool)
//                           added to its 1e-4.  The case therefore bounds the rounding rather than resolving FP32-level
//                           error under it; the exact-scale case above checks the rest of the kernel at full tightness.
//
// Which kernel ran is printed in brackets, from `qsa_prompt_attn_variant` (the decision `qsa_prompt_attn_batch` itself
// takes): "volta-wmma" on a V100 (sm_70/72) and on any card running a 70-only build, "mma-v1" on Turing,
// "ampere-i8-cpasync" / "mma-v1" on Ampere and newer.  When it says "fallback-fp32" - STRATA_VOLTA_ATTN=0 on a Volta, a
// pre-Volta card - `qsa_prompt_attn_batch` returns false, the caller would use the FP32 kernel, and there is nothing to
// compare: those runs print SKIP and do not fail (counted as "SKIPPED (fallback)").  A card whose variant is a kernel
// but whose batch call refuses is a FAIL.  Two cases below can also be SKIPPED because they do not apply (page holes on
// a kernel that does not mask them, a split run that does not fit in memory): "SKIPPED (not applicable)".  The last line
// of the output, `PASSED: n  SKIPPED (fallback): n  FAILURES: n  SKIPPED (not applicable): n`, is read by
// tools/volta/run_parity.sh (a V100 run must have no fallback skip, STRATA_VOLTA_ATTN=0 no pass).
//
// Edge cases, each for the three KV modes (all PASS or SKIP the same way):
//   page holes   a page-table entry of -1 (a page KV streaming could not make resident; every 7th page and the last,
//                so page 0's queries have no resident cell at all).  The Volta kernel MASKS such a cell (score -inf, weight
//                0) as the FP32 kernel does - the deliberate change from upstream's v1, which reads it as a zero row with
//                score 0 and lets it dilute the softmax (so does the cp.async kernel); the FP64 reference masks.  Run on
//                "volta-wmma" only; the other kernels print SKIP (their behaviour is upstream's, not ours to bound).
//   n = 0        every 5th query selects no cell (steps width 0): every kernel writes zeros (l = 0), the reference too.
//   split        65,541 queries (> 65,535: the launch is split into grids of 65,535, the second one starting at query
//                65,535 with every pointer advanced): the pools are tiny (200 cells) and the queries periodic - query i is a
//                copy of query i % 251 (251 is prime, so a pointer left unadvanced or advanced by the wrong amount puts a
//                different query there) - so only the buffers that must be big are (3.2 GB of q and output on the device;
//                SKIP when that does not fit).  Checked: FP64 and the old kernel on the first launch's start and end, the
//                split's boundary and the tail (single-query calls of the old kernel), every output finite, and every
//                output equal to the first period's (the kernels are deterministic) to 1e-6 of the scale.
//
// Usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5]
//   then always the short cases (every cell selected): context 1500 with 1500 queries, and the 2100 / 256 edge, for each
//   of the three KV modes, then the real-scale K8V4 and the edge cases above.  Exit status 0 = every run passed or was
//   skipped, 1 = a check failed, 2 = a CUDA error.
// Environment: STRATA_VOLTA_ATTN=0 (V100: the FP32 fallback, runs are skipped), =force (the Volta kernel on any
// sm_70+ card), STRATA_PROMPT_ATTN_V1=1 (Ampere+: the v1 kernel instead of the cp.async one).
// On a V100:  qsa_prompt_attn_parity               (32K context, every mode, plus the short cases)
//             qsa_prompt_attn_parity 2048 2048 3   (a 2K context: every selection width 1..2048, so every chunk tail)
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
float h2f(uint16_t b) { __half h; *reinterpret_cast<uint16_t*>(&h) = b; return __half2float(h); }
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

constexpr double kRound16 = 1.0 / 2048.0;       // 2^-11: the relative error of rounding to fp16 (round to nearest)
constexpr double kSub16 = 1.0 / 33554432.0;     // 2^-25: the absolute error below 2^-14 (fp16 subnormals)

// A device buffer that frees itself (the runs return from many places).
template <typename T> struct DevBuf {
    T* p = nullptr;
    DevBuf() = default;
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { cudaFree(p); }
    bool try_alloc(size_t n) {   // false: it did not fit (the error is cleared)
        if (cudaMalloc(&p, n * sizeof(T) + 64) == cudaSuccess) return true;
        cudaGetLastError();
        p = nullptr;
        return false;
    }
    void alloc(size_t n) { ck(try_alloc(n) ? cudaSuccess : cudaErrorMemoryAllocation, "malloc"); }
    void upload(const std::vector<T>& h) {
        alloc(h.size());
        ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    // `n` copies of `period` (per_query elements per query, `period_q` queries), back to back: query i is query
    // i % period_q
    void replicate(const std::vector<T>& period, int64_t per_query, int64_t period_q, int64_t n) {
        if (!p) alloc((size_t) (n * per_query));   // (try_alloc may have done it already)
        for (int64_t b = 0; b < n; b += period_q)
            ck(cudaMemcpy(p + b * per_query, period.data(), (size_t) (std::min(period_q, n - b) * per_query) * sizeof(T),
                          cudaMemcpyHostToDevice), "upload");
    }
};

enum Result { kPass = 0, kFail = 1, kSkip = 2, kSkipNA = 3 };   // kSkip: the FP32 fallback; kSkipNA: does not apply here

struct Case {
    int fmt = 1;                 // 1 int8, 0 fp16, 3 hybrid K8V4 (int8 K, q4_0 V)
    int64_t ctx = 0, nq = 0;     // the queries are the positions ctx - nq .. ctx - 1
    bool real_q4 = false;        // fmt 3: q4_0 scales from real-looking data (the fp16 rounding of V is real)
    bool holes = false;          // page-table entries of -1
    bool zero_width = false;     // every 5th query selects no cell
    bool split = false;          // > 65,535 queries, periodic (run_split)
    bool timed = true;
};
Case make(int fmt, int64_t ctx, int64_t nq) { Case c; c.fmt = fmt; c.ctx = ctx; c.nq = nq; return c; }

std::string label(const Case& c) {
    std::string s = c.fmt == 1 ? "int8" : c.fmt == 0 ? "fp16" : c.real_q4 ? "k8v4 (real q4_0 scales)" : "k8v4";
    if (c.holes) s += ", page-table holes";
    if (c.zero_width) s += ", n = 0 queries";
    if (c.split) s += ", split launch";
    return s;
}

// One synthetic KV cache: the host copies the reference reads, and the device pools.
struct Kv {
    int fmt = 1;
    bool rounded = false;         // K8V4 with real q4_0 scales: the kernel's fp16 rounding of V changes values
    double vmax = 0;              // rounded: max |dequantized V| over the pool
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> v4;      // fmt 3: block_q4_0 [row][8 blocks of 18 bytes]
    std::vector<float> vdq;       // fmt 3: the dequantized V (exact in FP32), for the reference
    std::vector<int32_t> table;   // page table: a shuffled permutation (the readers must follow it); -1 = no page
    DevBuf<int8_t> d_kq, d_vq;
    DevBuf<uint16_t> d_ks, d_vs, d_kh, d_vh;
    DevBuf<uint8_t> d_v4;
    DevBuf<int32_t> d_table;
    k::QsaAttnPools pl;
    void upload() {
        if (fmt == 1) {
            d_kq.upload(kq); d_vq.upload(vq); d_ks.upload(ks); d_vs.upload(vs);
            pl.k_q = d_kq.p; pl.v_q = d_vq.p; pl.k_scale = d_ks.p; pl.v_scale = d_vs.p;
        } else if (fmt == 3) {
            d_kq.upload(kq); d_ks.upload(ks); d_v4.upload(v4);
            pl.k_q = d_kq.p; pl.k_scale = d_ks.p; pl.v_q4 = d_v4.p;
        } else {
            d_kh.upload(kh); d_vh.upload(vh);
            pl.k_pool = d_kh.p; pl.v_pool = d_vh.p;
        }
        d_table.upload(table);
        pl.page_table = d_table.p;
    }
};

void build_kv(Kv& kv, int fmt, int64_t ctx, bool real_q4, bool holes, std::mt19937& rng) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    kv.fmt = fmt;
    kv.rounded = fmt == 3 && real_q4;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    if (fmt == 1 || fmt == 3) {
        kv.kq.resize(rows * HD); kv.ks.resize(rows * 4);
        for (auto& x : kv.kq) x = (int8_t) code(rng);
        for (auto& x : kv.ks) x = f2h(sc(rng));
    }
    if (fmt == 1) {
        kv.vq.resize(rows * HD); kv.vs.resize(rows * 4);
        for (auto& x : kv.vq) x = (int8_t) code(rng);
        for (auto& x : kv.vs) x = f2h(sc(rng));
    } else if (fmt == 3) {
        const int64_t blocks = HD / 32, bytes = blocks * 18;
        kv.v4.assign(rows * bytes, 0);
        kv.vdq.resize(rows * HD);
        std::uniform_int_distribution<int> kd(6, 31), byte(0, 255);
        std::uniform_real_distribution<float> lsig(std::log(0.1f), std::log(2.0f));
        for (int64_t r = 0; r < rows; ++r) {
            const float sigma = real_q4 ? std::exp(lsig(rng)) : 1.0f;   // (real scales only) this row's spread
            for (int64_t b = 0; b < blocks; ++b) {
                uint8_t* blk = kv.v4.data() + r * bytes + b * 18;
                float* out = kv.vdq.data() + r * HD + b * 32;
                if (!real_q4) {
                    const float d = kd(rng) / 1024.0f;   // exact in FP16 times any (nibble - 8): see the header comment
                    const uint16_t db = f2h(d);
                    std::memcpy(blk, &db, 2);
                    for (int j = 0; j < 16; ++j) {
                        blk[2 + j] = (uint8_t) byte(rng);
                        out[j] = (float) ((blk[2 + j] & 15) - 8) * d;          // element j: low nibble
                        out[j + 16] = (float) ((blk[2 + j] >> 4) - 8) * d;     // element j + 16: high nibble
                    }
                } else {
                    float x[32], amax = 0.f, mx = 0.f;
                    for (int j = 0; j < 32; ++j) {
                        x[j] = nd(rng) * sigma;
                        if (amax < std::fabs(x[j])) { amax = std::fabs(x[j]); mx = x[j]; }
                    }
                    const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;   // kv_q4.hpp's rule
                    const uint16_t db = f2h(d);
                    const float dr = h2f(db);                                        // the scale as stored
                    std::memcpy(blk, &db, 2);
                    for (int j = 0; j < 16; ++j) {
                        const int q0 = std::min(15, std::max(0, (int) (x[j] * id + 8.5f)));
                        const int q1 = std::min(15, std::max(0, (int) (x[j + 16] * id + 8.5f)));
                        blk[2 + j] = (uint8_t) (q0 | (q1 << 4));
                        out[j] = (float) (q0 - 8) * dr;
                        out[j + 16] = (float) (q1 - 8) * dr;
                    }
                }
            }
        }
        if (kv.rounded) for (float v : kv.vdq) kv.vmax = std::max(kv.vmax, (double) std::fabs(v));
    }
    if (fmt == 0) {
        kv.kh.resize(rows * HD); kv.vh.resize(rows * HD);
        for (auto& x : kv.kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : kv.vh) x = f2h(nd(rng));
    }
    kv.table.resize(pages);
    for (int64_t i = 0; i < pages; ++i) kv.table[i] = (int32_t) i;
    std::shuffle(kv.table.begin(), kv.table.end(), rng);
    if (holes)   // pages that are not resident: every 7th and the last (page 0 included: its queries see nothing)
        for (int64_t i = 0; i < pages; ++i)
            if (i % 7 == 0 || i == pages - 1) kv.table[i] = -1;
}

// FP64 attention of one query head over the selected cells whose page is resident (a page-table entry < 0 masks the
// cell: weight 0, as the FP32 kernel and the Volta kernel do); out = 0 when there is none (n = 0, or every cell masked),
// which is what every kernel writes then.  bound: the most the fp16 rounding of K8V4's V can move each element
// (sum_c p_c (2^-11 |v_c| + 2^-25), see the header); 0 for the modes whose V enters exactly.
void ref_head(const Kv& kv, const float* q, int64_t kvh, const int32_t* sel, int64_t w, double* out, double* bound) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, PS = s.page_size;
    std::fill(out, out + HD, 0.0);
    std::fill(bound, bound + HD, 0.0);
    std::vector<int64_t> rows;
    std::vector<double> p;
    double mx = -1e300;
    for (int64_t c = 0; c < w; ++c) {
        const int64_t cell = sel[c], page = kv.table[cell / PS];
        if (page < 0) continue;
        const int64_t row = (page * NKV + kvh) * PS + cell % PS;
        double a = 0;
        for (int64_t d = 0; d < HD; ++d) {
            const double kvv = kv.fmt != 0 ? (double) kv.kq[row * HD + d] * h2f(kv.ks[row * 4 + d / 64])
                                           : h2f(kv.kh[row * HD + d]);
            a += (double) q[d] * kvv;
        }
        a /= 16.0;
        rows.push_back(row);
        p.push_back(a);
        mx = std::max(mx, a);
    }
    if (rows.empty()) return;
    double l = 0;
    for (auto& x : p) { x = std::exp(x - mx); l += x; }
    for (size_t c = 0; c < rows.size(); ++c) {
        const int64_t row = rows[c];
        const double wgt = p[c] / l;
        for (int64_t d = 0; d < HD; ++d) {
            const double v = kv.fmt == 1   ? (double) kv.vq[row * HD + d] * h2f(kv.vs[row * 4 + d / 64])
                             : kv.fmt == 3 ? (double) kv.vdq[row * HD + d]
                                           : h2f(kv.vh[row * HD + d]);
            out[d] += wgt * v;
            if (kv.rounded) bound[d] += wgt * (kRound16 * std::fabs(v) + kSub16);
        }
    }
}

struct Err {
    double old_err = 0, new_err = 0, excess = 0, ref_scale = 0, budget = 0;
    // old_err / new_err: max |kernel - FP64|; excess: max (|new - FP64| - B) (>= 0), the part of the new kernel's error
    // that the fp16 rounding of K8V4's V cannot explain (= new_err where B is 0); budget: max B
};

// Folds one query (all 24 heads) into `e`: q [n_head * HD] and its selection, the old and new kernels' outputs.
void check_query(const Kv& kv, const float* q, const int32_t* sel, int64_t w, const float* o_old, const float* o_new,
                 Err& e) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NH = s.n_head, NKV = s.n_head_kv;
    std::vector<double> out((size_t) HD), bound((size_t) HD);
    for (int64_t h = 0; h < NH; ++h) {
        ref_head(kv, q + h * HD, h / (NH / NKV), sel, w, out.data(), bound.data());
        for (int64_t d = 0; d < HD; ++d) {
            const double r = out[d], en = std::fabs(o_new[h * HD + d] - r);
            e.ref_scale = std::max(e.ref_scale, std::fabs(r));
            e.old_err = std::max(e.old_err, (double) std::fabs(o_old[h * HD + d] - r));
            e.new_err = std::max(e.new_err, en);
            e.excess = std::max(e.excess, en - bound[d]);
            e.budget = std::max(e.budget, bound[d]);
        }
    }
}

// Prints a skip/fail line for a variant that has nothing to compare ("fallback-fp32": the caller's FP32 kernel runs).
Result fallback_result(const char* kind, const char* variant, int64_t ctx, int64_t nq, bool took) {
    std::printf("%s %s [%s] ctx %lld, %lld queries: qsa_prompt_attn_batch returns false here (the caller's FP32 "
                "kernel runs): fallback, skipped\n",
                took ? "FAIL" : "SKIP", kind, variant, (long long) ctx, (long long) nq);
    return took ? kFail : kSkip;
}

Result run(const Case& c, int reps) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NH = s.n_head;
    const int64_t ctx = c.ctx, nq = c.nq;
    const std::string kind_s = label(c);
    const char* kind = kind_s.c_str();
    std::mt19937 rng(1234 + c.fmt + (c.real_q4 ? 100 : 0));
    std::normal_distribution<float> nd(0.f, 1.f);
    Kv kv;
    build_kv(kv, c.fmt, ctx, c.real_q4, c.holes, rng);
    // queries at positions ctx - nq .. ctx - 1
    const int64_t cap = k::qsa_selection_width(ctx, s);
    std::vector<int32_t> ids((size_t) (nq * cap), 0), steps((size_t) (nq * k::kStepCount), 0);
    std::vector<float> q((size_t) (nq * NH * HD));
    for (auto& x : q) x = nd(rng) * 2.0f;
    std::vector<int32_t> old_cells;
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t pos = ctx - nq + i, nkv = pos + 1;
        const int64_t w = k::qsa_selection_width(nkv, s);
        int32_t* sel = ids.data() + i * cap;
        steps[i * k::kStepCount + k::kStepWidth] = (int32_t) w;
        if (w == nkv) {
            for (int64_t cc = 0; cc < w; ++cc) sel[cc] = (int32_t) cc;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {   // the older cells: a set that drifts ~3% per query
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t cc = 0; cc < nkv - recent; ++cc) all[cc] = (int32_t) cc;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t cc = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), cc) == old_cells.end()) old_cells[pick(rng)] = cc;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t cc = nkv - recent; cc < nkv; ++cc) v.push_back((int32_t) cc);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
    if (c.zero_width)   // every 5th query selects nothing (its ids are never read)
        for (int64_t i = 2; i < nq; i += 5) steps[i * k::kStepCount + k::kStepWidth] = 0;
    kv.upload();
    DevBuf<int32_t> d_ids, d_steps;
    DevBuf<float> d_q, d_old, d_new, scratch;
    d_ids.upload(ids);
    d_steps.upload(steps);
    d_q.upload(q);
    const int64_t batch = 32;
    d_old.alloc((size_t) (nq * NH * HD));
    d_new.alloc((size_t) (nq * NH * HD));
    scratch.alloc((size_t) (batch * k::qsa_decode_attn_scratch_floats(cap, s)));
    ck(cudaMemset(d_new.p, 0xFF, nq * NH * HD * 4), "memset");   // NaN: an element the new kernel does not write stays NaN
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q.p + t0 * NH * HD, kv.pl, d_ids.p + t0 * cap, d_steps.p + t0 * k::kStepCount, cap, s,
                                     scratch.p, d_old.p + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };
    auto new_run = [&]() {
        return k::qsa_prompt_attn_batch(d_q.p, kv.pl, d_ids.p, d_steps.p, cap, s, d_new.p, nq, nullptr);
    };
    // which kernel: a "fallback" is not a failure (nothing to compare); a kernel that refuses is
    const char* variant = k::qsa_prompt_attn_variant(kv.pl, s);
    if (std::strcmp(variant, "fallback-fp32") == 0) return fallback_result(kind, variant, ctx, nq, new_run());
    if (c.holes && std::strcmp(variant, "volta-wmma") != 0) {
        std::printf("SKIP %s [%s] ctx %lld, %lld queries: page-table entries of -1 are read as zero rows with score 0 "
                    "by this kernel (upstream's behaviour); only the Volta kernel masks them as the FP32 kernel does: "
                    "nothing to compare\n", kind, variant, (long long) ctx, (long long) nq);
        return kSkipNA;
    }
    old_run();
    if (!new_run()) {
        std::printf("FAIL %s [%s] ctx %lld, %lld queries: qsa_prompt_attn_batch refused the pools although the variant "
                    "is a kernel\n", kind, variant, (long long) ctx, (long long) nq);
        return kFail;
    }
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size());
    ck(cudaMemcpy(o.data(), d_old.p, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(nw.data(), d_new.p, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
    // 0. nothing left unwritten or non-finite
    int64_t nonfinite = 0;
    for (size_t i = 0; i < nw.size(); ++i) nonfinite += !std::isfinite(nw[i]) || !std::isfinite(o[i]);
    // 1. FP64 reference on a sample of queries (and every n = 0 one: they are cheap)
    Err e;
    const int64_t stride = std::max<int64_t>(1, nq / 16);
    for (int64_t i = 0; i < nq; ++i) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        if (i % stride != 0 && w != 0) continue;
        check_query(kv, q.data() + i * NH * HD, ids.data() + i * cap, w, o.data() + i * NH * HD, nw.data() + i * NH * HD, e);
    }
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    // 3. speed
    float ms_old = 0, ms_new = 0;
    if (c.timed) {
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) old_run();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "time");
        cudaEventElapsedTime(&ms_old, e0, e1);
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) new_run();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "time");
        cudaEventElapsedTime(&ms_new, e0, e1);
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    const double v4_budget = kv.rounded ? kRound16 * kv.vmax + kSub16 : 0.0;   // check 2's share of the fp16 rounding
    const bool ok1 = e.excess <= std::max(4.0 * e.old_err, 1e-6 * e.ref_scale);
    const bool ok2 = diff <= 1e-4 * scale + v4_budget;
    const bool ok3 = nonfinite == 0;
    char cmp[192], l2[64] = "", tm[96] = "";
    const double ratio = e.excess / std::max(e.old_err, 1e-300);
    if (kv.rounded) {
        std::snprintf(cmp, sizeof cmp, "new %.3g (q4_0 fp16 rounding bound %.3g; beyond it %.3g = %.2fx old; limit 4x)",
                      e.new_err, e.budget, e.excess, ratio);
        std::snprintf(l2, sizeof l2, " + %.3g fp16 rounding", v4_budget);
    } else {
        std::snprintf(cmp, sizeof cmp, "new %.3g (%.2fx; limit 4x)", e.new_err, ratio);
    }
    if (c.timed)
        std::snprintf(tm, sizeof tm, "; %.3f -> %.3f ms per chunk (%.2fx)", ms_old / reps, ms_new / reps, ms_old / ms_new);
    std::printf("%s %s [%s] ctx %lld, %lld queries: vs FP64 old %.3g %s (output scale %.3g); new vs old %.3g (%.2g of "
                "scale; limit 1e-4%s); %lld non-finite%s\n",
                ok1 && ok2 && ok3 ? "PASS" : "FAIL", kind, variant, (long long) ctx, (long long) nq, e.old_err, cmp,
                e.ref_scale, diff, diff / std::max(scale, 1e-300), l2, (long long) nonfinite, tm);
    return ok1 && ok2 && ok3 ? kPass : kFail;
}

// More than 65,535 queries: the launch is split (grid.x is capped at 65,535 per launch here) and every pointer advances
// by the queries already done.  See the header for the construction (tiny pools, queries periodic with period 251).
Result run_split(int fmt) {
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NH = s.n_head, QN = NH * HD;
    const int64_t P = 251, ctx = 200, nq = 65535 + 6, cap = ctx;
    Case c = make(fmt, ctx, nq);
    c.split = true;
    c.timed = false;
    const std::string kind_s = label(c);
    const char* kind = kind_s.c_str();
    std::mt19937 rng(777 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    Kv kv;
    build_kv(kv, fmt, ctx, false, false, rng);
    std::vector<float> hq((size_t) (P * QN));
    for (auto& x : hq) x = nd(rng) * 2.0f;
    std::vector<int32_t> hids((size_t) (P * cap), 0), hsteps((size_t) (P * k::kStepCount), 0);
    for (int64_t j = 0; j < P; ++j) {   // widths 1..200 (every chunk tail of 32), every 5th query selects nothing
        const int64_t w = j % 5 == 2 ? 0 : 1 + (j * 37) % ctx;
        hsteps[j * k::kStepCount + k::kStepWidth] = (int32_t) w;
        for (int64_t cc = 0; cc < w; ++cc) hids[j * cap + cc] = (int32_t) cc;
    }
    kv.upload();
    DevBuf<int32_t> d_ids, d_steps;
    d_ids.replicate(hids, cap, P, nq);
    d_steps.replicate(hsteps, k::kStepCount, P, nq);
    const char* variant = k::qsa_prompt_attn_variant(kv.pl, s);
    DevBuf<float> d_q, d_new;
    if (std::strcmp(variant, "fallback-fp32") == 0) {   // returns false before it touches a buffer: tiny ones will do
        d_q.alloc(QN);
        d_new.alloc(QN);
        return fallback_result(kind, variant, ctx, nq,
                               k::qsa_prompt_attn_batch(d_q.p, kv.pl, d_ids.p, d_steps.p, cap, s, d_new.p, nq, nullptr));
    }
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "meminfo");
    const size_t need = 2 * (size_t) nq * QN * 4 + ((size_t) 256 << 20);
    if (free_b < need || !d_q.try_alloc((size_t) (nq * QN)) || !d_new.try_alloc((size_t) (nq * QN))) {
        std::printf("SKIP %s [%s] ctx %lld, %lld queries: needs %.1f GB of device memory for q and the output "
                    "(%.1f GB free)\n", kind, variant, (long long) ctx, (long long) nq, need / 1e9, free_b / 1e9);
        return kSkipNA;
    }
    d_q.replicate(hq, QN, P, nq);
    ck(cudaMemset(d_new.p, 0xFF, (size_t) nq * QN * 4), "memset");   // NaN: an output never written stays NaN
    if (!k::qsa_prompt_attn_batch(d_q.p, kv.pl, d_ids.p, d_steps.p, cap, s, d_new.p, nq, nullptr)) {
        std::printf("FAIL %s [%s] ctx %lld, %lld queries: qsa_prompt_attn_batch refused the pools although the variant is a "
                    "kernel\n", kind, variant, (long long) ctx, (long long) nq);
        return kFail;
    }
    ck(cudaDeviceSynchronize(), "run");
    // the old kernel and FP64 on queries around the boundaries: the start, the end of the first launch, the start and
    // the tail of the second (65,538 selects nothing)
    std::vector<int64_t> samp;
    for (int64_t i : {(int64_t) 0, (int64_t) 2, (int64_t) 3, P - 1, (int64_t) 65534, (int64_t) 65535, (int64_t) 65536,
                      (int64_t) 65537, (int64_t) 65538, nq - 1})
        if (i < nq) samp.push_back(i);
    DevBuf<float> d_old, scratch;
    d_old.alloc(samp.size() * (size_t) QN);
    scratch.alloc((size_t) k::qsa_decode_attn_scratch_floats(cap, s));
    for (size_t si = 0; si < samp.size(); ++si)
        k::qsa_decode_attn_batch(d_q.p + samp[si] * QN, kv.pl, d_ids.p + samp[si] * cap,
                                 d_steps.p + samp[si] * k::kStepCount, cap, s, scratch.p, d_old.p + si * QN, 1, nullptr);
    ck(cudaDeviceSynchronize(), "old");
    std::vector<float> o(samp.size() * (size_t) QN), nw(QN);
    ck(cudaMemcpy(o.data(), d_old.p, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    Err e;
    double diff = 0, scale = 0;
    for (size_t si = 0; si < samp.size(); ++si) {
        const int64_t i = samp[si], j = i % P;
        ck(cudaMemcpy(nw.data(), d_new.p + i * QN, (size_t) QN * 4, cudaMemcpyDeviceToHost), "down");
        check_query(kv, hq.data() + j * QN, hids.data() + j * cap, hsteps[j * k::kStepCount + k::kStepWidth],
                    o.data() + si * QN, nw.data(), e);
        for (int64_t x = 0; x < QN; ++x) {
            diff = std::max(diff, (double) std::fabs(o[si * QN + x] - nw[x]));
            scale = std::max(scale, (double) std::fabs(o[si * QN + x]));
        }
    }
    // every output finite, and equal to the first period's
    std::vector<float> rep((size_t) (P * QN));
    ck(cudaMemcpy(rep.data(), d_new.p, rep.size() * 4, cudaMemcpyDeviceToHost), "down");
    double period_diff = 0, rep_scale = 0;
    int64_t nonfinite = 0;
    for (float x : rep) if (std::isfinite(x)) rep_scale = std::max(rep_scale, (double) std::fabs(x));
    const int64_t slab = 8 * P;
    std::vector<float> buf((size_t) (slab * QN));
    for (int64_t b = 0; b < nq; b += slab) {
        const int64_t m = std::min(slab, nq - b);
        ck(cudaMemcpy(buf.data(), d_new.p + b * QN, (size_t) (m * QN) * 4, cudaMemcpyDeviceToHost), "down");
        for (int64_t i = 0; i < m; ++i) {
            const float* got = buf.data() + i * QN;
            const float* want = rep.data() + ((b + i) % P) * QN;
            for (int64_t x = 0; x < QN; ++x) {
                if (!std::isfinite(got[x])) ++nonfinite;
                else period_diff = std::max(period_diff, (double) std::fabs(got[x] - want[x]));
            }
        }
    }
    const bool ok1 = e.excess <= std::max(4.0 * e.old_err, 1e-6 * e.ref_scale);
    const bool ok2 = diff <= 1e-4 * scale;
    const bool ok3 = nonfinite == 0;
    const bool ok4 = period_diff <= 1e-6 * rep_scale;
    std::printf("%s %s [%s] ctx %lld, %lld queries (> 65,535: the launch is split): vs FP64 old %.3g new %.3g (%.2fx; "
                "limit 4x) (output scale %.3g); new vs old %.3g (%.2g of scale; limit 1e-4); %lld non-finite; vs the "
                "first period (query i vs i %% %lld) %.3g (limit 1e-6 of scale)\n",
                ok1 && ok2 && ok3 && ok4 ? "PASS" : "FAIL", kind, variant, (long long) ctx, (long long) nq, e.old_err,
                e.new_err, e.excess / std::max(e.old_err, 1e-300), e.ref_scale, diff, diff / std::max(scale, 1e-300),
                (long long) nonfinite, (long long) P, period_diff);
    return ok1 && ok2 && ok3 && ok4 ? kPass : kFail;
}
}  // namespace

int main(int argc, char** argv) {
    int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 32768;
    int64_t nq = argc > 2 ? std::atoll(argv[2]) : 2048;
    const int reps = argc > 3 ? std::max(1, std::atoi(argv[3])) : 5;
    if (ctx < 1 || nq < 1) {
        std::fprintf(stderr, "usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5]\n");
        return 2;
    }
    nq = std::min(nq, ctx);   // the queries are the last `nq` positions of the context
    int dev = 0;
    cudaDeviceProp prop{};
    ck(cudaGetDevice(&dev), "device");
    ck(cudaGetDeviceProperties(&prop, dev), "properties");
    const char* env = std::getenv("STRATA_VOLTA_ATTN");
    std::printf("device %d: %s (sm_%d%d), STRATA_VOLTA_ATTN=%s, context %lld, %lld queries, %d reps\n", dev, prop.name,
                prop.major, prop.minor, env ? env : "(unset)", (long long) ctx, (long long) nq, reps);
    int fails = 0, skips = 0, na = 0, passes = 0;
    auto tally = [&](Result r) { (r == kFail ? fails : r == kSkip ? skips : r == kSkipNA ? na : passes) += 1; };
    for (int fmt : {1, 0, 3}) {   // int8, fp16, hybrid K8V4: the three KV modes of the tensor-core kernels
        tally(run(make(fmt, ctx, nq), reps));
        tally(run(make(fmt, 1500, std::min<int64_t>(nq, 1500)), reps));   // short context: the selection is every cell
        tally(run(make(fmt, 2100, std::min<int64_t>(nq, 256)), reps));    // the identity-to-sparse edge
    }
    {   // K8V4 again with real q4_0 scales: the fp16 rounding of V is real (and budgeted)
        Case c = make(3, ctx, nq);
        c.real_q4 = true;
        tally(run(c, reps));
        c = make(3, 1500, std::min<int64_t>(nq, 1500));
        c.real_q4 = true;
        tally(run(c, reps));
        c = make(3, 2100, std::min<int64_t>(nq, 256));
        c.real_q4 = true;
        tally(run(c, reps));
    }
    const int64_t edge[2][2] = {{1500, 1500}, {2100, 256}};   // (context, queries): identity selection; the edge
    for (int fmt : {1, 0, 3}) {   // page-table entries of -1 (Volta kernel only: see the header)
        for (const auto& cn : edge) {
            Case c = make(fmt, cn[0], std::min<int64_t>(nq, cn[1]));
            c.holes = true;
            c.timed = false;
            tally(run(c, reps));
        }
    }
    for (int fmt : {1, 0, 3}) {   // queries that select no cell
        Case c = make(fmt, 1500, std::min<int64_t>(nq, 1500));
        c.zero_width = true;
        c.timed = false;
        tally(run(c, reps));
    }
    for (int fmt : {1, 0, 3}) tally(run_split(fmt));   // more than 65,535 queries: the split launch
    // (tools/volta/run_parity.sh reads the prefix of this line: keep it)
    std::printf("PASSED: %d  SKIPPED (fallback): %d  FAILURES: %d  SKIPPED (not applicable): %d\n", passes, skips, fails,
                na);
    return fails ? 1 : 0;
}
