// src/ds41/cuda/mhc_common.cuh - DS1-D: small device helpers shared by the mHC and Engram kernels (own copies: nothing here depends on the DS-D
// math header, so the reduction order they define cannot move under them).  Names from ds41_dev.cuh only.
#pragma once

#include <cmath>
#include <cstdint>

#include "ds41_dev.cuh"

namespace strata::ds41::cuda::dev {

DS41_HD constexpr int hc_pow2_ceil(int v) { return v <= 1 ? 1 : 2 * hc_pow2_ceil((v + 1) / 2); }
DS41_HD constexpr int hc_log2(int v) { return v <= 1 ? 0 : 1 + hc_log2(v >> 1); }

/// V partial sums per lane (V a power of two <= 32) summed over the 32 lanes with V - 1 + (5 - log2 V) shuffles: each stage keeps half of the
/// values and trades the other half with the partner lane.  Lane kk returns the total of value index (kk >> (5 - log2 V)); with V = 1 every lane
/// holds the total.  Fixed order: the same bits for every call.
template <int V>
DS41_FI float hc_vsum(float (&v)[V]) {
    static_assert(V >= 1 && V <= 32 && (V & (V - 1)) == 0);
    const int kk = threadIdx.x & 31;
    int off = 16;
    DS41_UNROLL
    for (int n = V; n > 1; n >>= 1, off >>= 1) {
        const bool upper = (kk & off) != 0;
        DS41_UNROLL
        for (int i = 0; i < n / 2; ++i) {
            const float send = upper ? v[i] : v[i + n / 2];
            const float keep = upper ? v[i + n / 2] : v[i];
            v[i] = keep + shfl_xor(send, off);
        }
    }
    float r = v[0];
    DS41_UNROLL
    for (; off > 0; off >>= 1) r += shfl_xor(r, off);
    return r;
}

/// Sum over the 32 lanes (butterfly 16, 8, 4, 2, 1); every lane returns the total.
DS41_FI float hc_warp_sum(float v) {
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) v += shfl_xor(v, off);
    return v;
}

/// mhc.py / ops.py sigmoid: the split form, no overflow: e = exp(-|x|); x >= 0: 1 / (1 + e), else e / (1 + e).  A NaN stays NaN.
DS41_FI float hc_sigmoid(float x) {
    const float e = expf(-fabsf(x));
    const float d = fadd_rn(1.0f, e);
    return x >= 0.0f ? fdiv_rn(1.0f, d) : fdiv_rn(e, d);
}

DS41_FI float4 hc_ld4(const float* p) {          // plain (coherent) 16-byte load: for data a kernel may overwrite itself
    DS41_ASSERT_ALIGNED(p, 16);
    return *reinterpret_cast<const float4*>(p);
}
DS41_FI void hc_st4(float* p, float4 v) {
    DS41_ASSERT_ALIGNED(p, 16);
    *reinterpret_cast<float4*>(p) = v;
}

}  // namespace strata::ds41::cuda::dev
