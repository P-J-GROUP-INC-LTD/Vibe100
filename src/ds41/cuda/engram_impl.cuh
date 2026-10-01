// src/ds41/cuda/engram_impl.cuh - DS1-D: the Engram device side for the V100 (sm_70): MXFP4 row dequantisation and the combine kernel.  Included by
// engram.cu (nvcc, RealGeom) and by the CPU emulation test (-DDS41_EMU).  API: include/strata/ds41/cuda/engram.hpp; oracle: ref/ds41/engram.py
// engram_layer (= model.py Engram.forward) and ref/ds41/quant.py dequant_mxfp4.
//
// engram_dequant_kernel   GGML block_mxfp4 (17 B: E8M0 byte e, 16 bytes of codes; value j = kvalues[qs[j] & 15] * d, value j + 16 = kvalues[qs[j] >> 4] * d,
//                         d = 2^(e - 128) EXACTLY as ggml_e8m0_to_fp32_half, so e = 0, 1 are denormals and e = 255 is the finite 2^127) -> FP32.  One warp per
//                         block, lane = value index: one 17-byte span per warp.  Bit-exact against the oracle (small integers times a power of two).
// engram_combine_kernel   grid (kHc, T), 256 threads, one (token, copy) per block, in place on the stream:
//                           weight = q_w[c] * k_w[c]                        (BF16 -> FP32, product rounded once, as numpy)
//                           rstd   = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)           over kHidden, per (token, copy)
//                           dot    = sum_d h * weight * key  *  rstd * kHidden^-0.5              ((h * weight) * key, summed)
//                           gate   = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
//                           h     += gate * value                                                 (value = kv[kHc * kHidden ..], shared by the copies)
//                         The three sums (h^2, key^2, the dot) have one order: per thread i = tid, tid + 256, ... (float4 columns, components in order),
//                         warp butterfly 16..1, then the 8 warp totals added in warp order.  No atomics; independent of T and of the schedule.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "mhc_common.cuh"
#include "strata/ds41/cuda/engram.hpp"

namespace strata::ds41::cuda::dev {

inline constexpr int kEgThreads = 256;
inline constexpr int kEgWarps = kEgThreads / 32;

/// 2^(e - 128) as ggml_e8m0_to_fp32_half has it: e < 2 -> the denormals 2^-128 (0x00200000) and 2^-127 (0x00400000); otherwise biased exponent e - 1.
DS41_FI float eg_e8m0_half(uint32_t e) {
    const uint32_t norm = (e - 1u) << 23;
    const uint32_t sub = 0x00200000u << (e & 1u);
    return u2f(e >= 2u ? norm : sub);
}

/// kvalues_fp4 = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}: magnitude {0,1,2,3,4,6,8,12}[code & 7], sign from bit 3 (code 8 is +0, as ggml).
DS41_FI int eg_kvalue(uint32_t code) {
    const int m = (int) (code & 7u);
    const int mag = m < 4 ? m : (m == 7 ? 12 : 2 * (m - 2));
    return (code & 8u) ? -mag : mag;
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(128) void engram_dequant_kernel(const uint8_t* DS41_RESTRICT rows, int n_blocks, float* DS41_RESTRICT out) {
    const int lane = threadIdx.x & 31;
    const int gb = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (gb >= n_blocks) return;                                        // warp-uniform
    const uint8_t* blk = rows + (size_t) gb * kBlockBytes;
    const uint32_t e = blk[0];
    const uint32_t q = blk[1 + (lane & 15)];
    const uint32_t code = lane < 16 ? (q & 15u) : (q >> 4);
    out[(size_t) gb * kQK + lane] = fmul_rn((float) eg_kvalue(code), eg_e8m0_half(e));
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kEgThreads) void engram_combine_kernel(float* x, const float* DS41_RESTRICT kv, const uint16_t* DS41_RESTRICT qw,
                                                                    const uint16_t* DS41_RESTRICT kw, float eps, float inv_sqrt_dim, float clamp_value) {
    constexpr int H = G::kHidden, C = G::kHc, D4 = G::kHidden / 4;
    constexpr int OUT = Derived<G>::kEngramOut;
    DS41_SHARED float s_red[3][kEgWarps];
    const int c = blockIdx.x, t = blockIdx.y;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float* h = x + ((size_t) t * C + c) * H;                          // read AND written by this block: plain loads
    const float* key = kv + (size_t) t * OUT + (size_t) c * H;
    const float* val = kv + (size_t) t * OUT + (size_t) C * H;
    const uint16_t* q = qw + (size_t) c * H;
    const uint16_t* kk = kw + (size_t) c * H;

    float shh = 0.0f, skk = 0.0f, sdot = 0.0f;
    for (int i = tid; i < D4; i += kEgThreads) {
        const float4 hv = hc_ld4(h + 4 * i);
        const float4 kv4 = ldgf4(key + 4 * i);
        const uint2 qb = ldg2(q + 4 * i);
        const uint2 kb = ldg2(kk + 4 * i);
        const float hs[4] = {hv.x, hv.y, hv.z, hv.w};
        const float ks[4] = {kv4.x, kv4.y, kv4.z, kv4.w};
        const float qs[4] = {u2f(qb.x << 16), u2f(qb.x & 0xFFFF0000u), u2f(qb.y << 16), u2f(qb.y & 0xFFFF0000u)};   // bf16 -> fp32: the bits are the top half
        const float ws[4] = {u2f(kb.x << 16), u2f(kb.x & 0xFFFF0000u), u2f(kb.y << 16), u2f(kb.y & 0xFFFF0000u)};
        DS41_UNROLL
        for (int j = 0; j < 4; ++j) {
            const float weight = fmul_rn(qs[j], ws[j]);
            sdot = fadd_rn(sdot, fmul_rn(fmul_rn(hs[j], weight), ks[j]));
            shh = fadd_rn(shh, fmul_rn(hs[j], hs[j]));
            skk = fadd_rn(skk, fmul_rn(ks[j], ks[j]));
        }
    }
    shh = hc_warp_sum(shh);
    skk = hc_warp_sum(skk);
    sdot = hc_warp_sum(sdot);
    if (lane == 0) {
        s_red[0][warp] = shh;
        s_red[1][warp] = skk;
        s_red[2][warp] = sdot;
    }
    sync_block();
    float th = 0.0f, tk = 0.0f, td = 0.0f;
    DS41_UNROLL
    for (int w = 0; w < kEgWarps; ++w) {                              // fixed order
        th = fadd_rn(th, s_red[0][w]);
        tk = fadd_rn(tk, s_red[1][w]);
        td = fadd_rn(td, s_red[2][w]);
    }
    const float mean_h = fdiv_rn(th, (float) H), mean_k = fdiv_rn(tk, (float) H);
    const float rstd = fmul_rn(fdiv_rn(1.0f, sqrtf(fadd_rn(mean_h, eps))), fdiv_rn(1.0f, sqrtf(fadd_rn(mean_k, eps))));
    const float dot = fmul_rn(fmul_rn(td, rstd), inv_sqrt_dim);
    float a = fabsf(dot);
    a = a < clamp_value ? clamp_value : a;                            // np.maximum: a NaN stays NaN
    const float gate = hc_sigmoid(copysignf(sqrtf(a), dot));

    for (int i = tid; i < D4; i += kEgThreads) {
        const float4 hv = hc_ld4(h + 4 * i);
        const float4 vv = ldgf4(val + 4 * i);
        hc_st4(h + 4 * i, make_float4(fadd_rn(hv.x, fmul_rn(gate, vv.x)), fadd_rn(hv.y, fmul_rn(gate, vv.y)), fadd_rn(hv.z, fmul_rn(gate, vv.z)),
                                      fadd_rn(hv.w, fmul_rn(gate, vv.w))));
    }
}

}  // namespace strata::ds41::cuda::dev

namespace strata::ds41::cuda {

template <class G>
void ds41_engram_dequant_rows(Dev& d, const uint8_t* rows, int n_rows, float* out, Stream stream) {
    if (n_rows < 1) return;
    if (reinterpret_cast<uintptr_t>(out) % 16 != 0) throw std::invalid_argument("ds41 engram_dequant_rows: out must be 16-byte aligned");
    const Stream s = stream_or_default(d, stream);
    const int nb = n_rows * (G::kEngramHeadDim / kQK);
    dev::launch(dev::engram_dequant_kernel<G>, dim3((unsigned) ((nb + 3) / 4)), dim3(128), 0, s, rows, nb, out);
    dev::check_launch("engram_dequant");
}

template <class G>
void ds41_engram_combine(Dev& d, float* x, const float* kv, const uint16_t* q_bf16, const uint16_t* k_bf16, int T, float norm_eps, Stream stream) {
    if (T < 1) return;
    if (reinterpret_cast<uintptr_t>(x) % 16 != 0 || reinterpret_cast<uintptr_t>(kv) % 16 != 0 || reinterpret_cast<uintptr_t>(q_bf16) % 8 != 0 ||
        reinterpret_cast<uintptr_t>(k_bf16) % 8 != 0)
        throw std::invalid_argument("ds41 engram_combine: x and kv must be 16-byte, q / k 8-byte aligned");
    const Stream s = stream_or_default(d, stream);
    const float inv_sqrt_dim = (float) (1.0 / std::sqrt((double) G::kHidden));       // `dim ** -0.5` rounded to float32
    dev::launch(dev::engram_combine_kernel<G>, dim3((unsigned) G::kHc, (unsigned) T), dim3(dev::kEgThreads), 0, s, x, kv, q_bf16, k_bf16, norm_eps,
                inv_sqrt_dim, kEngramClampValue);
    dev::check_launch("engram_combine");
}

template <class G>
EngramKernelInfo ds41_engram_kernel_info(int which) {
    EngramKernelInfo info;
#if !defined(DS41_EMU)
    cudaFuncAttributes a{};
    cudaError_t e = cudaErrorInvalidValue;
    if (which == 0) e = cudaFuncGetAttributes(&a, (const void*) dev::engram_dequant_kernel<G>);
    else if (which == 1) e = cudaFuncGetAttributes(&a, (const void*) dev::engram_combine_kernel<G>);
    if (e == cudaSuccess) {
        info.regs = a.numRegs;
        info.static_smem = (int) a.sharedSizeBytes;
    }
#else
    (void) which;
#endif
    return info;
}

}  // namespace strata::ds41::cuda
