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
// K8V4's q4_0 V scales are multiples of 2^-10 (k / 1024, k <= 31) so every dequantized value is exact in FP16: the
// kernel's fp16 dequantization (`qsa_prompt_attn.cu`, the same as v1's) then adds nothing, and the checks above stay as
// tight as for the other modes.
//
// Which kernel ran is printed in brackets, from `qsa_prompt_attn_variant` (the decision `qsa_prompt_attn_batch` itself
// takes): "volta-wmma" on a V100 (sm_70/72), "mma-v1" on Turing, "ampere-i8-cpasync" / "mma-v1" on Ampere and newer.
// When it says "fallback-fp32" - STRATA_VOLTA_ATTN=0 on a Volta, a pre-Volta card - `qsa_prompt_attn_batch` returns
// false, the caller would use the FP32 kernel, and there is nothing to compare: those runs print SKIP and do not fail.
// A card whose variant is a kernel but whose batch call refuses is a FAIL.
//
// Usage: qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5]
//   then always the short cases (every cell selected): context 1500 with 1500 queries, and the 2100 / 256 edge, for each
//   of the three KV modes.  Exit status 0 = every run passed or was skipped, 1 = a check failed, 2 = a CUDA error.
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc(&d, h.size() * sizeof(T) + 64), "malloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}
float h2f(uint16_t b) { __half h; *reinterpret_cast<uint16_t*>(&h) = b; return __half2float(h); }
uint16_t f2h(float f) { __half h = __float2half(f); return *reinterpret_cast<uint16_t*>(&h); }

enum Result { kPass = 0, kFail = 1, kSkip = 2 };

Result run(int fmt, int64_t ctx, int64_t nq, int reps) {   // fmt 1 int8, 0 fp16, 3 hybrid K8V4 (int8 K, q4_0 V)
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t HD = s.head_dim, NKV = s.n_head_kv, NH = s.n_head, PS = s.page_size;
    const int64_t pages = (ctx + PS - 1) / PS, rows = pages * NKV * PS;
    const char* kind = fmt == 1 ? "int8" : fmt == 0 ? "fp16" : "k8v4";
    std::mt19937 rng(1234 + fmt);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.03f);
    // pools; page table a shuffled permutation (the readers must follow it)
    std::vector<int8_t> kq, vq;
    std::vector<uint16_t> ks, vs, kh, vh;
    std::vector<uint8_t> v4;       // fmt 3: block_q4_0 [row][8 blocks of 18 bytes]
    std::vector<float> vdq;        // fmt 3: the dequantized V, for the reference
    if (fmt == 1 || fmt == 3) {
        kq.resize(rows * HD); ks.resize(rows * 4);
        for (auto& x : kq) x = (int8_t) code(rng);
        for (auto& x : ks) x = f2h(sc(rng));
    }
    if (fmt == 1) {
        vq.resize(rows * HD); vs.resize(rows * 4);
        for (auto& x : vq) x = (int8_t) code(rng);
        for (auto& x : vs) x = f2h(sc(rng));
    } else if (fmt == 3) {
        const int64_t blocks = HD / 32, bytes = blocks * 18;
        v4.assign(rows * bytes, 0);
        vdq.resize(rows * HD);
        std::uniform_int_distribution<int> kd(6, 31), byte(0, 255);
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t b = 0; b < blocks; ++b) {
                uint8_t* blk = v4.data() + r * bytes + b * 18;
                const float d = kd(rng) / 1024.0f;   // exact in FP16 times any (nibble - 8): see the header comment
                const uint16_t db = f2h(d);
                std::memcpy(blk, &db, 2);
                for (int j = 0; j < 16; ++j) {
                    blk[2 + j] = (uint8_t) byte(rng);
                    vdq[r * HD + b * 32 + j] = (float) ((blk[2 + j] & 15) - 8) * d;          // element j: low nibble
                    vdq[r * HD + b * 32 + j + 16] = (float) ((blk[2 + j] >> 4) - 8) * d;     // element j + 16: high nibble
                }
            }
    }
    if (fmt == 0) {
        kh.resize(rows * HD); vh.resize(rows * HD);
        for (auto& x : kh) x = f2h(nd(rng) * 1.5f);
        for (auto& x : vh) x = f2h(nd(rng));
    }
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
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
            for (int64_t c = 0; c < w; ++c) sel[c] = (int32_t) c;
            continue;
        }
        const int64_t recent = 512, older = w - recent;
        if ((int64_t) old_cells.size() != older) {   // the older cells: a set that drifts ~3% per query
            std::vector<int32_t> all((size_t) (nkv - recent));
            for (int64_t c = 0; c < nkv - recent; ++c) all[c] = (int32_t) c;
            std::shuffle(all.begin(), all.end(), rng);
            old_cells.assign(all.begin(), all.begin() + older);
        } else {
            std::uniform_int_distribution<int64_t> pick(0, older - 1), any(0, nkv - recent - 1);
            for (int r = 0; r < older / 32; ++r) {
                const int32_t c = (int32_t) any(rng);
                if (std::find(old_cells.begin(), old_cells.end(), c) == old_cells.end()) old_cells[pick(rng)] = c;
            }
        }
        std::vector<int32_t> v(old_cells);
        for (int64_t c = nkv - recent; c < nkv; ++c) v.push_back((int32_t) c);
        std::sort(v.begin(), v.end());
        std::copy(v.begin(), v.end(), sel);
    }
    k::QsaAttnPools pl;
    if (fmt == 1) { pl.k_q = up(kq); pl.v_q = up(vq); pl.k_scale = up(ks); pl.v_scale = up(vs); }
    else if (fmt == 3) { pl.k_q = up(kq); pl.k_scale = up(ks); pl.v_q4 = up(v4); }
    else { pl.k_pool = up(kh); pl.v_pool = up(vh); }
    pl.page_table = up(table);
    const int32_t* d_ids = up(ids);
    const int32_t* d_steps = up(steps);
    const float* d_q = up(q);
    float *d_old = nullptr, *d_new = nullptr, *scratch = nullptr;
    const int64_t batch = 32;
    ck(cudaMalloc(&d_old, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&d_new, nq * NH * HD * 4), "malloc");
    ck(cudaMalloc(&scratch, batch * k::qsa_decode_attn_scratch_floats(cap, s) * 4), "malloc");
    ck(cudaMemset(d_new, 0xFF, nq * NH * HD * 4), "memset");   // NaN: an element the new kernel does not write stays NaN
    auto cleanup = [&]() {
        cudaFree((void*) d_ids); cudaFree((void*) d_steps); cudaFree((void*) d_q); cudaFree(d_old); cudaFree(d_new);
        cudaFree(scratch);
        cudaFree((void*) pl.k_q); cudaFree((void*) pl.v_q); cudaFree((void*) pl.k_scale); cudaFree((void*) pl.v_scale);
        cudaFree((void*) pl.k_pool); cudaFree((void*) pl.v_pool); cudaFree((void*) pl.v_q4); cudaFree((void*) pl.page_table);
    };
    auto old_run = [&]() {
        for (int64_t t0 = 0; t0 < nq; t0 += batch)
            k::qsa_decode_attn_batch(d_q + t0 * NH * HD, pl, d_ids + t0 * cap, d_steps + t0 * k::kStepCount, cap, s,
                                     scratch, d_old + t0 * NH * HD, std::min(batch, nq - t0), nullptr);
    };
    auto new_run = [&]() { return k::qsa_prompt_attn_batch(d_q, pl, d_ids, d_steps, cap, s, d_new, nq, nullptr); };
    // which kernel: a "fallback" is not a failure (nothing to compare); a kernel that refuses is
    const char* variant = k::qsa_prompt_attn_variant(pl, s);
    const bool fallback = std::strcmp(variant, "fallback-fp32") == 0;
    if (fallback) {
        const bool took = new_run();
        std::printf("%s %s [%s] ctx %lld, %lld queries: qsa_prompt_attn_batch returns false here (the caller's FP32 "
                    "kernel runs): fallback, skipped\n",
                    took ? "FAIL" : "SKIP", kind, variant, (long long) ctx, (long long) nq);
        cleanup();
        return took ? kFail : kSkip;
    }
    old_run();
    if (!new_run()) {
        std::printf("FAIL %s [%s] ctx %lld, %lld queries: qsa_prompt_attn_batch refused the pools although the variant "
                    "is a kernel\n", kind, variant, (long long) ctx, (long long) nq);
        cleanup();
        return kFail;
    }
    ck(cudaDeviceSynchronize(), "run");
    std::vector<float> o((size_t) (nq * NH * HD)), nw(o.size());
    ck(cudaMemcpy(o.data(), d_old, o.size() * 4, cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(nw.data(), d_new, nw.size() * 4, cudaMemcpyDeviceToHost), "down");
    // 0. nothing left unwritten or non-finite
    int64_t nonfinite = 0;
    for (size_t i = 0; i < nw.size(); ++i) nonfinite += !std::isfinite(nw[i]) || !std::isfinite(o[i]);
    // 1. FP64 reference on a sample of queries
    double err_old = 0, err_new = 0, ref_scale = 0;
    for (int64_t i = 0; i < nq; i += std::max<int64_t>(1, nq / 16)) {
        const int64_t w = steps[i * k::kStepCount + k::kStepWidth];
        const int32_t* sel = ids.data() + i * cap;
        for (int64_t h = 0; h < NH; ++h) {
            const int64_t kvh = h / (NH / NKV);
            std::vector<double> sco((size_t) w);
            double mx = -1e300;
            for (int64_t c = 0; c < w; ++c) {
                const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                double a = 0;
                for (int64_t d = 0; d < HD; ++d) {
                    const double kv = fmt != 0 ? (double) kq[row * HD + d] * h2f(ks[row * 4 + d / 64]) : h2f(kh[row * HD + d]);
                    a += (double) q[(i * NH + h) * HD + d] * kv;
                }
                sco[c] = a / 16.0;
                mx = std::max(mx, sco[c]);
            }
            double l = 0;
            for (auto& x : sco) { x = std::exp(x - mx); l += x; }
            for (int64_t d = 0; d < HD; ++d) {
                double a = 0;
                for (int64_t c = 0; c < w; ++c) {
                    const int64_t cell = sel[c], row = ((int64_t) table[cell / PS] * NKV + kvh) * PS + cell % PS;
                    const double vv = fmt == 1   ? (double) vq[row * HD + d] * h2f(vs[row * 4 + d / 64])
                                      : fmt == 3 ? (double) vdq[row * HD + d]
                                                 : h2f(vh[row * HD + d]);
                    a += sco[c] * vv;
                }
                const double r = a / l;
                const size_t at = (size_t) ((i * NH + h) * HD + d);
                ref_scale = std::max(ref_scale, std::fabs(r));
                err_old = std::max(err_old, std::fabs(o[at] - r));
                err_new = std::max(err_new, std::fabs(nw[at] - r));
            }
        }
    }
    // 2. new vs old everywhere
    double diff = 0, scale = 0;
    for (size_t i = 0; i < o.size(); ++i) {
        diff = std::max(diff, (double) std::fabs(o[i] - nw[i]));
        scale = std::max(scale, (double) std::fabs(o[i]));
    }
    // 3. speed
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    float ms_old = 0, ms_new = 0;
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
    const bool ok1 = err_new <= std::max(4.0 * err_old, 1e-6 * ref_scale);
    const bool ok2 = diff <= 1e-4 * scale;
    const bool ok3 = nonfinite == 0;
    std::printf("%s %s [%s] ctx %lld, %lld queries: vs FP64 old %.3g new %.3g (%.2fx; limit 4x) (output scale %.3g); new vs "
                "old %.3g (%.2g of scale; limit 1e-4); %lld non-finite; %.3f -> %.3f ms per chunk (%.2fx)\n",
                ok1 && ok2 && ok3 ? "PASS" : "FAIL", kind, variant, (long long) ctx, (long long) nq, err_old, err_new,
                err_new / std::max(err_old, 1e-300), ref_scale, diff, diff / scale, (long long) nonfinite, ms_old / reps,
                ms_new / reps, ms_old / ms_new);
    cleanup();
    return ok1 && ok2 && ok3 ? kPass : kFail;
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
    int fails = 0, skips = 0, passes = 0;
    auto tally = [&](Result r) { (r == kFail ? fails : r == kSkip ? skips : passes) += 1; };
    for (int fmt : {1, 0, 3}) {   // int8, fp16, hybrid K8V4: the three KV modes of the tensor-core kernels
        tally(run(fmt, ctx, nq, reps));
        tally(run(fmt, 1500, std::min<int64_t>(nq, 1500), reps));   // short context: the selection is every cell
        tally(run(fmt, 2100, std::min<int64_t>(nq, 256), reps));    // the identity-to-sparse edge
    }
    std::printf("PASSED: %d  SKIPPED (fallback): %d  FAILURES: %d\n", passes, skips, fails);
    return fails ? 1 : 0;
}
