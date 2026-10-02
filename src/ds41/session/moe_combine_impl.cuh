// src/ds41/session/moe_combine_impl.cuh - DS1-E: the MoE layer's FP32 sum (include/strata/ds41/session/moe_combine.hpp), written once and compiled twice: by nvcc for
// sm_70 (moe_combine.cu: RealGeom) and by the host compiler with -DDS41_EMU against the emulator (moe_combine_emu_impl.cpp: RealGeom and MiniGeom).
#pragma once

#include <stdexcept>

#include "../cuda/ds41_dev.cuh"
#include "strata/ds41/session/moe_combine.hpp"

namespace strata::ds41::cuda {
namespace session_kernels {

inline constexpr int kThreads = 256;

/// One thread per output element (token t, column d).  The token's kTopK (id, k) keys are visited in increasing order of id * kTopK + k (ids are distinct for a
/// healthy router; the k breaks a tie so that a corrupt duplicate is still added exactly once): `prev` is the last key taken, the next one is the smallest key
/// above it.  No array is indexed by a run-time value: every access is a global load with a compile-time unrolled k.
template <int K>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void moe_combine_kernel(const int32_t* DS41_RESTRICT ids, const float* DS41_RESTRICT parts, const int32_t* DS41_RESTRICT cpu_row,
                                                              const float* DS41_RESTRICT cpu_rows, const float* DS41_RESTRICT shared, int hidden, int total, float* DS41_RESTRICT out) {
    const int i = (int) (blockIdx.x * kThreads + threadIdx.x);
    if (i >= total) return;
    const int t = i / hidden, d = i - t * hidden;
    int prev = -1;
    float acc = 0.0f;
    DS41_UNROLL1
    for (int r = 0; r < K; ++r) {
        int best_key = 0x7fffffff, best_k = -1;
        DS41_UNROLL
        for (int k = 0; k < K; ++k) {
            const int key = ids[t * K + k] * K + k;
            if (key > prev && key < best_key) {
                best_key = key;
                best_k = k;
            }
        }
        if (best_k < 0) break;                                   // (cannot happen: K distinct keys)
        prev = best_key;
        const int row = t * K + best_k;
        const int c = cpu_row[row];
        const float v = c >= 0 ? cpu_rows[(size_t) c * hidden + d] : parts[(size_t) row * hidden + d];
        acc = dev::fadd_rn(acc, v);
    }
    out[(size_t) i] = dev::fadd_rn(acc, shared[(size_t) i]);
}

inline void check_ptr(const void* p, unsigned align, const char* what) {
    if (!p || reinterpret_cast<uintptr_t>(p) % align != 0) throw std::invalid_argument(std::string("ds41_moe_combine: ") + what + " is null or misaligned");
}

}  // namespace session_kernels

template <class G>
void ds41_moe_combine(Dev& dev, const int32_t* ids, const float* parts, const int32_t* cpu_row, const float* cpu_rows, const float* shared, int T, float* out, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (T < 1) throw std::invalid_argument("ds41_moe_combine: T < 1");
    session_kernels::check_ptr(ids, 4, "ids");
    session_kernels::check_ptr(parts, 16, "parts");
    session_kernels::check_ptr(cpu_row, 4, "cpu_row");
    session_kernels::check_ptr(cpu_rows, 16, "cpu_rows");
    session_kernels::check_ptr(shared, 16, "shared");
    session_kernels::check_ptr(out, 16, "out");
    stream = stream_or_default(dev, stream);
    const int total = T * G::kHidden;
    dev::launch(session_kernels::moe_combine_kernel<G::kTopK>, dim3((unsigned) ((total + session_kernels::kThreads - 1) / session_kernels::kThreads)),
                dim3(session_kernels::kThreads), 0, stream, ids, parts, cpu_row, cpu_rows, shared, (int) G::kHidden, total, out);
    dev::check_launch("ds41_moe_combine");
}

}  // namespace strata::ds41::cuda

#define DS41_INSTANTIATE_MOE_COMBINE(G)                                                                                                                        \
    namespace strata::ds41::cuda {                                                                                                                              \
    template void ds41_moe_combine<G>(Dev&, const int32_t*, const float*, const int32_t*, const float*, const float*, int, float*, Stream);                     \
    }
