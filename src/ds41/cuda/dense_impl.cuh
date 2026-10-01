// src/ds41/cuda/dense_impl.cuh - DS1-B: the dense kernels (Q8_0 / BF16 / F32 GEMV, RMSNorm, RoPE, SwiGLU, argmax / top-k, the activation quantiser) and their host
// entry points, written ONCE and compiled twice: by nvcc for sm_70 (dense.cu: RealGeom) and by the host compiler with -DDS41_EMU against the emulator
// (dense_emu_impl.cpp: RealGeom and MiniGeom).  The API and the numerics are in include/strata/ds41/cuda/dense.hpp; read its header first.
//
// HOW THE Q8_0 GEMV READS ITS WEIGHTS.  A Q8_0 block is 34 bytes (fp16 d, 32 int8): rows are only 2-byte aligned in general, but 8 blocks = 272 bytes = 17 x 16
// bytes, so when k % 256 == 0 (every real shape) a "super-block" of 8 blocks is 16-byte aligned and one lane loads it with 17 LDG.128.  Inside a super-block
// (68 words) pair q of blocks (A even, B odd) is words 17q .. 17q + 16: d_A = low half of word 17q, the quants of A are bytes 2..33 of the pair (misaligned by
// 2: one byte-permute per word, shared by every token), d_B = high half of word 17q + 8, the quants of B are words 17q + 9 .. 17q + 16 (aligned).
// One lane = one super-block of one row; P lanes (P = a power of two >= ceil(k / 256), at most 32) share a row and 32 / P rows share a warp.  The activations
// of the block's tokens are staged in shared memory with 16 bytes of padding per super-block, so the 8 lanes of a LDS.128 quarter-warp hit 8 different bank groups.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "dense_dev.cuh"
#include "ds41_math.cuh"                      // DS-D: dev::quantize_block_warp (the activation quantiser, shared with the expert path)
#include "strata/ds41/cuda/dense.hpp"

namespace strata::ds41::cuda::dense {

// ---- launch constants -------------------------------------------------------------------------------------------------------------------------
inline constexpr int kGemvWarps = 4;                    // warps per block of the Q8_0 GEMV
inline constexpr int kGemvThreads = 32 * kGemvWarps;
inline constexpr int kWideWarps = 4;                    // warps per block of the BF16 / F32 GEMV (one warp per row)
inline constexpr int kWideThreads = 32 * kWideWarps;
#ifndef DS41_DENSE_WIDE_MINB
#define DS41_DENSE_WIDE_MINB 3
#endif
#ifndef DS41_DENSE_GEMV_MINB
#define DS41_DENSE_GEMV_MINB 2
#endif
inline constexpr int kMaxSmem = 96 * 1024;              // dynamic shared memory per block on sm_70 (opt-in above 48 KB)
inline constexpr int kActSb = 304;                      // int8 activations: 256 quants + 8 fp32 scales + 16 pad, per super-block
inline constexpr int kXSb = 1040;                       // fp32 activations: 256 floats + 4 pad, per super-block
inline constexpr int kWideUnroll = 8;                   // 16-byte chunks a lane keeps in flight (BF16 / F32 GEMV)
inline constexpr int kTopSlice = 2048;                  // vocabulary elements one top-k block of phase 1 owns (8 per thread)
inline constexpr int kTopMaxK = 64;

// =====================================================================================================================================================
// kernels
// =====================================================================================================================================================

// ---- Q8_0 super-block helpers ------------------------------------------------------------------------------------------------------------------
/// The 68 words of super-block data of one row: `nblk` (1..8) blocks starting at `p`.  FAST: p is 16-byte aligned and the super-block is complete: 17 LDG.128.
/// Otherwise (any alignment, a partial super-block): 16-bit loads of exactly the bytes that exist (a 2-byte aligned p; never past the row's end).
template <bool FAST>
DS41_FI void load_sb(const uint8_t* DS41_RESTRICT p, int nblk, uint32_t (&w)[68]) {
    if (FAST) {
        (void) nblk;
        DS41_UNROLL
        for (int i = 0; i < 17; ++i) {
            const uint4 v = dev::ldg4(p + 16 * i);
            w[4 * i + 0] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else {
        DS41_UNROLL
        for (int k = 0; k < 68; ++k) w[k] = 0u;
        const int nh = 17 * nblk;                          // 16-bit halves that exist
        DS41_UNROLL
        for (int i = 0; i < 136; ++i)
            if (i < nh) w[i >> 1] |= ldg16(p + 2 * i) << (16 * (i & 1));
    }
}

/// Block c (0..7) of a super-block: its scale and its 8 quant words aligned to the activation words (word j = quants 4j .. 4j + 3).
DS41_FI void block_words(const uint32_t (&w)[68], int c, float& dw, uint32_t (&wq)[8]) {
    const int q = c >> 1;
    if ((c & 1) == 0) {                                    // pair q's block A: d in the low half of word 17q, quants at bytes 2..33 of the pair
        dw = f16_to_f32(w[17 * q] & 0xFFFFu);
        DS41_UNROLL
        for (int j = 0; j < 8; ++j) wq[j] = dev::prmt(w[17 * q + j], w[17 * q + j + 1], 0x5432u);   // bytes 2,3 of the first word, 0,1 of the next
    } else {                                               // block B: d in the high half of word 17q + 8, quants at words 17q + 9 .. 17q + 16
        dw = f16_to_f32(w[17 * q + 8] >> 16);
        DS41_UNROLL
        for (int j = 0; j < 8; ++j) wq[j] = w[17 * q + 9 + j];
    }
}

/// One lane's super-block, int8 activations: acc[t] = fma(d_w * d_x, float(sum of dp4a), acc[t]) block by block (increasing c).  `act` = the super-block's
/// first byte of token 0 in shared memory; token t is `tok_bytes` further.
template <int NT>
DS41_FI void sb_int8(const uint32_t (&w)[68], int nblk, const unsigned char* act, int tok_bytes, float (&acc)[NT]) {
    DS41_UNROLL
    for (int c = 0; c < 8; ++c) {
        if (c < nblk) {
            float dw;
            uint32_t wq[8];
            block_words(w, c, dw, wq);
            DS41_UNROLL
            for (int t = 0; t < NT; ++t) {
                const unsigned char* a = act + (size_t) t * tok_bytes;
                const uint4 a0 = ld16b(a + 32 * c), a1 = ld16b(a + 32 * c + 16);
                const float sc = *reinterpret_cast<const float*>(a + 256 + 4 * c);
                int s = 0;
                s = dev::dp4a((int) wq[0], (int) a0.x, s);
                s = dev::dp4a((int) wq[1], (int) a0.y, s);
                s = dev::dp4a((int) wq[2], (int) a0.z, s);
                s = dev::dp4a((int) wq[3], (int) a0.w, s);
                s = dev::dp4a((int) wq[4], (int) a1.x, s);
                s = dev::dp4a((int) wq[5], (int) a1.y, s);
                s = dev::dp4a((int) wq[6], (int) a1.z, s);
                s = dev::dp4a((int) wq[7], (int) a1.w, s);
                acc[t] = fma_rn(dev::fmul_rn(dw, sc), (float) s, acc[t]);
            }
        }
    }
}

/// One lane's super-block, FP32 activations: value = float(q) * d_w (exact), acc[t] = fma(x, value, acc[t]) in increasing element order.
template <int NT>
DS41_FI void sb_f32(const uint32_t (&w)[68], int nblk, const unsigned char* xsb, int tok_bytes, float (&acc)[NT]) {
    DS41_UNROLL
    for (int c = 0; c < 8; ++c) {
        if (c < nblk) {
            float dw;
            uint32_t wq[8];
            block_words(w, c, dw, wq);
            DS41_UNROLL
            for (int j = 0; j < 8; ++j) {
                const float f0 = dev::fmul_rn(dw, (float) sext8(wq[j], 0));
                const float f1 = dev::fmul_rn(dw, (float) sext8(wq[j], 1));
                const float f2 = dev::fmul_rn(dw, (float) sext8(wq[j], 2));
                const float f3 = dev::fmul_rn(dw, (float) sext8(wq[j], 3));
                DS41_UNROLL
                for (int t = 0; t < NT; ++t) {
                    const float4 xv = ldf4(xsb + (size_t) t * tok_bytes + (size_t) (8 * c + j) * 16);
                    float a = acc[t];
                    a = fma_rn(xv.x, f0, a);
                    a = fma_rn(xv.y, f1, a);
                    a = fma_rn(xv.z, f2, a);
                    a = fma_rn(xv.w, f3, a);
                    acc[t] = a;
                }
            }
        }
    }
}

// ---- the Q8_0 GEMV, int8 activations -----------------------------------------------------------------------------------------------------------
// grid.x = ntiles * row_blocks * nmat (token tile fastest, so that the tiles of one row range run side by side and share the weights through L2).
template <int NT, bool FAST>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kGemvThreads, DS41_DENSE_GEMV_MINB) void q8_int8_kernel(const uint8_t* DS41_RESTRICT wa, const uint8_t* DS41_RESTRICT wb, int N, int K,
                                                                 const int8_t* DS41_RESTRICT xq, const float* DS41_RESTRICT xs, int T, float* DS41_RESTRICT ya,
                                                                 float* DS41_RESTRICT yb, int p_log2, int rows_per_block, int row_blocks, int ntiles) {
    DS41_DYN_SMEM(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb = K >> 5, nsb = (nb + 7) >> 3;
    const int tok_bytes = nsb * kActSb;
    int bid = blockIdx.x;
    const int tt = bid % ntiles;
    bid /= ntiles;
    const int rbk = bid % row_blocks;
    const int mat = bid / row_blocks;
    const uint8_t* W = mat ? wb : wa;
    float* y = mat ? yb : ya;
    const int t0 = tt * NT;
    const int tn = (T - t0) < NT ? (T - t0) : NT;

    // ---- the tile's activations -> shared memory (the tokens past the end of the window are zero: their results are not stored)
    for (int t = 0; t < NT; ++t) {
        unsigned char* base = smem + (size_t) t * tok_bytes;
        if (t < tn) {
            const unsigned char* src = reinterpret_cast<const unsigned char*>(xq) + (size_t) (t0 + t) * K;
            for (int u = tid; u < (K >> 4); u += kGemvThreads) st16b(base + (u >> 4) * kActSb + (u & 15) * 16, dev::ldg4(src + 16 * u));
            for (int b = tid; b < nb; b += kGemvThreads)
                *reinterpret_cast<float*>(base + (b >> 3) * kActSb + 256 + (b & 7) * 4) = dev::ldgf(xs + (size_t) (t0 + t) * nb + b);
        } else {
            for (int u = tid; u < tok_bytes / 16; u += kGemvThreads) st16b(base + 16 * u, make_uint4(0u, 0u, 0u, 0u));
        }
    }
    dev::sync_block();

    const int p = 1 << p_log2, gpw = 32 >> p_log2;
    const int ls = lane & (p - 1), gi = lane >> p_log2;
    const int r0 = rbk * rows_per_block;
    const int r1 = (r0 + rows_per_block < N) ? r0 + rows_per_block : N;
    const int ngrp = (r1 - r0 + gpw - 1) / gpw;
    const size_t row_bytes = (size_t) nb * kQ8BlockBytes;
    for (int rg = warp; rg < ngrp; rg += kGemvWarps) {
        const int row = r0 + rg * gpw + gi;
        const bool valid = row < r1;
        float acc[NT];
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        if (valid) {
            const uint8_t* rp = W + (size_t) row * row_bytes;
            for (int s = ls; s < nsb; s += p) {
                const int nblk = (nb - 8 * s) < 8 ? (nb - 8 * s) : 8;
                uint32_t w[68];
                load_sb<FAST>(rp + (size_t) s * 272, nblk, w);
                sb_int8<NT>(w, nblk, smem + (size_t) s * kActSb, tok_bytes, acc);
            }
        }
        for (int off = p >> 1; off > 0; off >>= 1) {
            DS41_UNROLL
            for (int t = 0; t < NT; ++t) acc[t] = acc[t] + dev::shfl_xor(acc[t], off);
        }
        if (valid && ls == 0) {
            DS41_UNROLL
            for (int t = 0; t < NT; ++t)
                if (t < tn) y[(size_t) (t0 + t) * N + row] = acc[t];
        }
    }
}

// ---- the Q8_0 GEMV, FP32 activations (plain and grouped) --------------------------------------------------------------------------------------------
// grid.x = ntiles * row_blocks_per_group * groups.  Row n = g * R + r of the matrix multiplies x[t][g * x_group_stride ..].
template <int NT, bool FAST>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kGemvThreads, DS41_DENSE_GEMV_MINB) void q8_f32_kernel(const uint8_t* DS41_RESTRICT W, int R, int K, const float* DS41_RESTRICT x, int x_tok_stride,
                                                                int x_group_stride, int T, float* DS41_RESTRICT y, int p_log2, int rows_per_block,
                                                                int row_blocks, int ntiles, int Ntot) {
    DS41_DYN_SMEM(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nb = K >> 5, nsb = (nb + 7) >> 3;
    const int tok_bytes = nsb * kXSb;
    int bid = blockIdx.x;
    const int tt = bid % ntiles;
    bid /= ntiles;
    const int rbk = bid % row_blocks;
    const int g = bid / row_blocks;
    const int t0 = tt * NT;
    const int tn = (T - t0) < NT ? (T - t0) : NT;

    for (int t = 0; t < NT; ++t) {
        unsigned char* base = smem + (size_t) t * tok_bytes;
        if (t < tn) {
            const float* src = x + (size_t) (t0 + t) * x_tok_stride + (size_t) g * x_group_stride;
            for (int u = tid; u < (K >> 2); u += kGemvThreads) stf4(base + (u >> 6) * kXSb + (u & 63) * 16, dev::ldgf4(src + 4 * u));
        } else {
            for (int u = tid; u < tok_bytes / 16; u += kGemvThreads) st16b(base + 16 * u, make_uint4(0u, 0u, 0u, 0u));
        }
    }
    dev::sync_block();

    const int p = 1 << p_log2, gpw = 32 >> p_log2;
    const int ls = lane & (p - 1), gi = lane >> p_log2;
    const int r0 = rbk * rows_per_block;
    const int r1 = (r0 + rows_per_block < R) ? r0 + rows_per_block : R;
    const int ngrp = (r1 - r0 + gpw - 1) / gpw;
    const size_t row_bytes = (size_t) nb * kQ8BlockBytes;
    for (int rg = warp; rg < ngrp; rg += kGemvWarps) {
        const int r = r0 + rg * gpw + gi;
        const bool valid = r < r1;
        const int row = g * R + r;
        float acc[NT];
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        if (valid) {
            const uint8_t* rp = W + (size_t) row * row_bytes;
            for (int s = ls; s < nsb; s += p) {
                const int nblk = (nb - 8 * s) < 8 ? (nb - 8 * s) : 8;
                uint32_t w[68];
                load_sb<FAST>(rp + (size_t) s * 272, nblk, w);
                sb_f32<NT>(w, nblk, smem + (size_t) s * kXSb, tok_bytes, acc);
            }
        }
        for (int off = p >> 1; off > 0; off >>= 1) {
            DS41_UNROLL
            for (int t = 0; t < NT; ++t) acc[t] = acc[t] + dev::shfl_xor(acc[t], off);
        }
        if (valid && ls == 0) {
            DS41_UNROLL
            for (int t = 0; t < NT; ++t)
                if (t < tn) y[(size_t) (t0 + t) * Ntot + row] = acc[t];
        }
    }
}

// ---- the BF16 / F32 GEMV: one warp per row, coalesced 16-byte chunks -----------------------------------------------------------------------------
/// One 16-byte chunk c of the row (8 BF16 or 4 F32 elements, element index e = c * kElems + u): acc[t] = fma(x, w, acc[t]) in element order.  The FP32
/// activations of the tile are at 16-byte unit u + (u >> 3) (one padding unit per 8: conflict-free LDS.128 for a warp reading consecutive chunks).
template <int NT, bool BF16>
DS41_FI void wide_chunk(const uint4& wv, int c, const unsigned char* tile, int tok_bytes, float (&acc)[NT]) {
    if (BF16) {
        float wf[8];
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
        DS41_UNROLL
        for (int i = 0; i < 4; ++i) {
            wf[2 * i] = dev::u2f(ww[i] << 16);
            wf[2 * i + 1] = dev::u2f(ww[i] & 0xFFFF0000u);
        }
        const int u0 = 2 * c;                              // the 8 elements are the 16-byte units 2c and 2c + 1 (contiguous after the padding)
        const int ph = u0 + (u0 >> 3);
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) {
            const unsigned char* b = tile + (size_t) t * tok_bytes + (size_t) ph * 16;
            const float4 xa = ldf4(b), xb = ldf4(b + 16);
            float a = acc[t];
            a = fma_rn(xa.x, wf[0], a);
            a = fma_rn(xa.y, wf[1], a);
            a = fma_rn(xa.z, wf[2], a);
            a = fma_rn(xa.w, wf[3], a);
            a = fma_rn(xb.x, wf[4], a);
            a = fma_rn(xb.y, wf[5], a);
            a = fma_rn(xb.z, wf[6], a);
            a = fma_rn(xb.w, wf[7], a);
            acc[t] = a;
        }
    } else {
        const float w0 = dev::u2f(wv.x), w1 = dev::u2f(wv.y), w2 = dev::u2f(wv.z), w3 = dev::u2f(wv.w);
        const int ph = c + (c >> 3);
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) {
            const float4 xa = ldf4(tile + (size_t) t * tok_bytes + (size_t) ph * 16);
            float a = acc[t];
            a = fma_rn(xa.x, w0, a);
            a = fma_rn(xa.y, w1, a);
            a = fma_rn(xa.z, w2, a);
            a = fma_rn(xa.w, w3, a);
            acc[t] = a;
        }
    }
}

template <int NT, bool BF16>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kWideThreads, DS41_DENSE_WIDE_MINB) void wide_kernel(const uint8_t* DS41_RESTRICT W, int N, int K, const float* DS41_RESTRICT x, int T, float* DS41_RESTRICT y,
                                                              int rows_per_block, int ntiles) {
    DS41_DYN_SMEM(smem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nu = K >> 2;                                 // 16-byte units of one token's activations
    const int tok_bytes = (nu + ((nu + 7) >> 3)) * 16;
    const int tt = blockIdx.x % ntiles, rbk = blockIdx.x / ntiles;
    const int t0 = tt * NT;
    const int tn = (T - t0) < NT ? (T - t0) : NT;
    for (int t = 0; t < NT; ++t) {
        unsigned char* base = smem + (size_t) t * tok_bytes;
        if (t < tn) {
            const float* src = x + (size_t) (t0 + t) * K;
            for (int u = tid; u < nu; u += kWideThreads) stf4(base + (size_t) (u + (u >> 3)) * 16, dev::ldgf4(src + 4 * u));
        } else {
            for (int u = tid; u < tok_bytes / 16; u += kWideThreads) st16b(base + 16 * u, make_uint4(0u, 0u, 0u, 0u));
        }
    }
    dev::sync_block();

    constexpr int kElems = BF16 ? 8 : 4;
    const int nchunks = K / kElems;
    const size_t row_bytes = (size_t) K * (BF16 ? 2 : 4);
    const int r0 = rbk * rows_per_block;
    const int r1 = (r0 + rows_per_block < N) ? r0 + rows_per_block : N;
    for (int row = r0 + warp; row < r1; row += kWideWarps) {
        const uint8_t* rp = W + (size_t) row * row_bytes;
        float acc[NT];
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        for (int c0 = lane; c0 < nchunks; c0 += 32 * kWideUnroll) {
            uint4 wv[kWideUnroll];
            DS41_UNROLL
            for (int i = 0; i < kWideUnroll; ++i) {
                const int c = c0 + 32 * i;
                wv[i] = c < nchunks ? dev::ldg4(rp + (size_t) c * 16) : make_uint4(0u, 0u, 0u, 0u);
            }
            DS41_UNROLL
            for (int i = 0; i < kWideUnroll; ++i) {
                const int c = c0 + 32 * i;
                if (c < nchunks) wide_chunk<NT, BF16>(wv[i], c, smem, tok_bytes, acc);
            }
        }
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) {
            const float v = warp_sum(acc[t]);
            if (lane == 0 && t < tn) y[(size_t) (t0 + t) * N + row] = v;
        }
    }
}

// ---- RMSNorm: one warp per row ---------------------------------------------------------------------------------------------------------------------
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void rmsnorm_kernel(const float* x, const float* DS41_RESTRICT w, float* out, int rows, int width, float eps) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * kNW + warp;
    if (row >= rows) return;                               // warp-uniform
    const float* xr = x + (size_t) row * width;
    float* orow = out + (size_t) row * width;
    const int nc = width >> 2;
    float ss = 0.0f;
    for (int c = lane; c < nc; c += 32) {
        const float4 v = ldf4(xr + 4 * c);
        float s4 = dev::fmul_rn(v.x, v.x);
        s4 = dev::fadd_rn(s4, dev::fmul_rn(v.y, v.y));
        s4 = dev::fadd_rn(s4, dev::fmul_rn(v.z, v.z));
        s4 = dev::fadd_rn(s4, dev::fmul_rn(v.w, v.w));
        ss = dev::fadd_rn(ss, s4);
    }
    ss = warp_sum(ss);
    const float ms = dev::fdiv_rn(ss, (float) width);
    const float r = sqrt_rn(dev::fadd_rn(ms, eps));
    for (int c = lane; c < nc; c += 32) {
        const float4 v = ldf4(xr + 4 * c);
        float4 o;
        if (w != nullptr) {
            const float4 wv = dev::ldgf4(w + 4 * c);
            o.x = dev::fmul_rn(wv.x, dev::fdiv_rn(v.x, r));
            o.y = dev::fmul_rn(wv.y, dev::fdiv_rn(v.y, r));
            o.z = dev::fmul_rn(wv.z, dev::fdiv_rn(v.z, r));
            o.w = dev::fmul_rn(wv.w, dev::fdiv_rn(v.w, r));
        } else {
            o.x = dev::fdiv_rn(v.x, r);
            o.y = dev::fdiv_rn(v.y, r);
            o.z = dev::fdiv_rn(v.z, r);
            o.w = dev::fdiv_rn(v.w, r);
        }
        stf4(orow + 4 * c, o);
    }
}

// ---- RoPE: one thread per adjacent pair of a row ---------------------------------------------------------------------------------------------------
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void rope_kernel(const float* x, float* out, int rows, int heads, int width, int rd, const float* DS41_RESTRICT cos_t,
                                                          const float* DS41_RESTRICT sin_t, int pos0, int inverse) {
    const int pairs = width >> 1;
    const long long gid = (long long) blockIdx.x * kThreads + threadIdx.x;
    if (gid >= (long long) rows * pairs) return;
    const int row = (int) (gid / pairs), p = (int) (gid - (long long) row * pairs);
    const int nope_pairs = (width - rd) >> 1;
    const size_t e = (size_t) row * width + 2 * (size_t) p;
    const float a = x[e], b = x[e + 1];
    if (p < nope_pairs) {                                  // the channels that are not rotated
        if (out != x) {
            out[e] = a;
            out[e + 1] = b;
        }
        return;
    }
    const int i = p - nope_pairs;
    const size_t ti = (size_t) (pos0 + row / heads) * (size_t) (rd >> 1) + (size_t) i;
    const float c = dev::ldgf(cos_t + ti);
    float s = dev::ldgf(sin_t + ti);
    if (inverse) s = -s;
    out[e] = dev::fadd_rn(dev::fmul_rn(a, c), -dev::fmul_rn(b, s));     // a * c - b * s   (two roundings: numpy's arithmetic)
    out[e + 1] = dev::fadd_rn(dev::fmul_rn(a, s), dev::fmul_rn(b, c));  // a * s + b * c
}

// ---- SwiGLU: h = silu(min(g, L)) * clamp(u, -L, L) ------------------------------------------------------------------------------------------------
DS41_FI float swiglu_h(float g, float u, float limit) {
    if (limit > 0.0f) {
        u = u > limit ? limit : (u < -limit ? -limit : u);   // NaN fails both tests and stays NaN (torch.clamp), unlike fminf / fmaxf
        g = g > limit ? limit : g;
    }
    const float silu = dev::fmul_rn(g, sigmoid_split(g));
    return dev::fmul_rn(silu, u);
}
/// QUANT = false: thread per element, h written over g.  QUANT = true: warp per (token, 32-block), h quantised (natural order) into hq / hs; g, u are read only.
template <bool QUANT>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void swiglu_kernel(float* g, const float* DS41_RESTRICT u, int n_or_T, int width, float limit, int8_t* DS41_RESTRICT hq,
                                                            float* DS41_RESTRICT hs) {
    if (!QUANT) {
        const int i = blockIdx.x * kThreads + threadIdx.x;
        if (i >= n_or_T) return;
        g[i] = swiglu_h(g[i], dev::ldgf(u + i), limit);
    } else {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int nb = width >> 5;
        const int gw = blockIdx.x * kNW + warp;
        if (gw >= n_or_T * nb) return;                     // warp-uniform
        const int t = gw / nb, b = gw - t * nb;
        const size_t i = (size_t) t * width + (size_t) b * 32 + lane;
        const float h = swiglu_h(dev::ldgf(g + i), dev::ldgf(u + i), limit);
        dev::quantize_block_warp<false>(h, lane, hq + ((size_t) t * nb + b) * 32, hs + (size_t) t * nb + b);   // DS-D's quantiser, natural order
    }
}

// ---- elementwise ------------------------------------------------------------------------------------------------------------------------------------
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void add_kernel(const float* a, const float* b, int n4, float* out) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n4) return;
    const float4 x = ldf4(a + 4 * i), y = ldf4(b + 4 * i);
    stf4(out + 4 * i, make_float4(dev::fadd_rn(x.x, y.x), dev::fadd_rn(x.y, y.y), dev::fadd_rn(x.z, y.z), dev::fadd_rn(x.w, y.w)));
}
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void scale_kernel(const float* a, float s, int n4, float* out) {
    const int i = blockIdx.x * kThreads + threadIdx.x;
    if (i >= n4) return;
    const float4 x = ldf4(a + 4 * i);
    stf4(out + 4 * i, make_float4(dev::fmul_rn(x.x, s), dev::fmul_rn(x.y, s), dev::fmul_rn(x.z, s), dev::fmul_rn(x.w, s)));
}

// ---- argmax: one block per token ---------------------------------------------------------------------------------------------------------------------
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void argmax_kernel(const float* DS41_RESTRICT logits, int vocab, int32_t* DS41_RESTRICT idx_out, float* DS41_RESTRICT val_out) {
    DS41_SHARED uint32_t s_key[kNW];
    DS41_SHARED int s_idx[kNW];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const float* row = logits + (size_t) t * vocab;
    uint32_t bk = 0u;
    int bi = kNoIndex;
    const int nv4 = vocab >> 2;
    for (int f = tid; f < nv4; f += kThreads) {
        const float4 v = dev::ldgf4(row + 4 * f);
        const float vv[4] = {v.x, v.y, v.z, v.w};
        DS41_UNROLL
        for (int e = 0; e < 4; ++e) {
            const uint32_t k = f2key(vv[e]);
            const int i = 4 * f + e;
            if (key_better(k, i, bk, bi)) {
                bk = k;
                bi = i;
            }
        }
    }
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) {
        const uint32_t ok = (uint32_t) dev::shfl_xor((int) bk, off);
        const int oi = dev::shfl_xor(bi, off);
        if (key_better(ok, oi, bk, bi)) {
            bk = ok;
            bi = oi;
        }
    }
    if (lane == 0) {
        s_key[warp] = bk;
        s_idx[warp] = bi;
    }
    dev::sync_block();
    uint32_t fk = s_key[0];
    int fi = s_idx[0];
    for (int w = 1; w < kNW; ++w)
        if (key_better(s_key[w], s_idx[w], fk, fi)) {
            fk = s_key[w];
            fi = s_idx[w];
        }
    if (tid == 0) {
        idx_out[t] = fi;
        val_out[t] = fi == kNoIndex ? key2f(fk) : row[fi];
    }
}

// ---- top-k: per-slice selection, then a merge --------------------------------------------------------------------------------------------------------
/// k rounds of "the block's best (key descending, index ascending) among the elements still alive"; every thread owns NE elements (key / idx / alive bit).  Thread 0
/// stores round r's winner at out_key[r] / out_idx[r] (and the decoded value at out_val[r] when given).  One barrier per round (the shared slots alternate).
template <int NE>
DS41_FI void select_rounds(const uint32_t (&key)[NE], const int (&idx)[NE], uint32_t alive, int k, uint32_t* s_key, int* s_idx, uint32_t* out_key, int32_t* out_idx,
                           float* out_val) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int r = 0; r < k; ++r) {
        uint32_t bk = 0u;
        int bi = kNoIndex, be = -1;
        DS41_UNROLL
        for (int e = 0; e < NE; ++e)
            if (((alive >> e) & 1u) != 0u && key_better(key[e], idx[e], bk, bi)) {
                bk = key[e];
                bi = idx[e];
                be = e;
            }
        uint32_t wk = bk;
        int wi = bi;
        DS41_UNROLL
        for (int off = 16; off > 0; off >>= 1) {
            const uint32_t ok = (uint32_t) dev::shfl_xor((int) wk, off);
            const int oi = dev::shfl_xor(wi, off);
            if (key_better(ok, oi, wk, wi)) {
                wk = ok;
                wi = oi;
            }
        }
        const int par = (r & 1) * kNW;
        if (lane == 0) {
            s_key[par + warp] = wk;
            s_idx[par + warp] = wi;
        }
        dev::sync_block();
        uint32_t fk = s_key[par];
        int fi = s_idx[par];
        for (int w = 1; w < kNW; ++w)
            if (key_better(s_key[par + w], s_idx[par + w], fk, fi)) {
                fk = s_key[par + w];
                fi = s_idx[par + w];
            }
        if (tid == 0) {
            if (out_key != nullptr) out_key[r] = fk;
            out_idx[r] = fi;
            if (out_val != nullptr) out_val[r] = key2f(fk);
        }
        if (be >= 0 && bi == fi) alive &= ~(1u << be);     // the owner retires the winner (indices are unique)
    }
}

/// Phase 1: grid (slices, T).  Block (s, t) selects the k best of vocabulary slice s of token t and writes them (sorted) to cand[(t * slices + s) * k + r].
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void topk_slice_kernel(const float* DS41_RESTRICT logits, int vocab, int k, int slices, uint32_t* DS41_RESTRICT cand_key,
                                                                int32_t* DS41_RESTRICT cand_idx) {
    constexpr int NE = kTopSlice / kThreads;
    DS41_SHARED uint32_t s_key[2 * kNW];
    DS41_SHARED int s_idx[2 * kNW];
    const int sl = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    const float* row = logits + (size_t) t * vocab;
    uint32_t key[NE];
    int idx[NE];
    uint32_t alive = 0u;
    DS41_UNROLL
    for (int e = 0; e < NE; ++e) {
        const int i = sl * kTopSlice + e * kThreads + tid;
        if (i < vocab) {
            key[e] = f2key(dev::ldgf(row + i));
            idx[e] = i;
            alive |= 1u << e;
        } else {
            key[e] = 0u;
            idx[e] = kNoIndex;
        }
    }
    const size_t o = ((size_t) t * slices + sl) * k;
    select_rounds<NE>(key, idx, alive, k, s_key, s_idx, cand_key + o, cand_idx + o, nullptr);
}

/// Phase 2: one block per token merges its slices * k candidates (at most kThreads * kMergeNE) into the final k.
inline constexpr int kMergeNE = 32;
DS41_KERNEL DS41_LAUNCH_BOUNDS(kThreads) void topk_merge_kernel(const uint32_t* DS41_RESTRICT cand_key, const int32_t* DS41_RESTRICT cand_idx, int m, int k,
                                                                int32_t* DS41_RESTRICT out_idx, float* DS41_RESTRICT out_val) {
    DS41_SHARED uint32_t s_key[2 * kNW];
    DS41_SHARED int s_idx[2 * kNW];
    const int t = blockIdx.x, tid = threadIdx.x;
    uint32_t key[kMergeNE];
    int idx[kMergeNE];
    uint32_t alive = 0u;
    DS41_UNROLL
    for (int e = 0; e < kMergeNE; ++e) {
        const int j = e * kThreads + tid;
        if (j < m) {
            key[e] = cand_key[(size_t) t * m + j];
            idx[e] = cand_idx[(size_t) t * m + j];
            alive |= 1u << e;
        } else {
            key[e] = 0u;
            idx[e] = kNoIndex;
        }
    }
    select_rounds<kMergeNE>(key, idx, alive, k, s_key, s_idx, nullptr, out_idx + (size_t) t * k, out_val + (size_t) t * k);
}

}  // namespace strata::ds41::cuda::dense

// =====================================================================================================================================================
// host side
// =====================================================================================================================================================
namespace strata::ds41::cuda::dense {

inline void check_ptr(const void* p, size_t align, const char* what) {
    if (p == nullptr) throw std::invalid_argument(std::string("ds41 dense: ") + what + " is null");
    if (reinterpret_cast<uintptr_t>(p) % align != 0)
        throw std::invalid_argument(std::string("ds41 dense: ") + what + " is not " + std::to_string(align) + "-byte aligned");
}
inline int pow2_ceil(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}
inline int ilog2_exact(int v) {
    int l = 0;
    while ((1 << l) < v) ++l;
    return l;
}
inline int round_up(int v, int m) { return (v + m - 1) / m * m; }
/// Blocks of this much shared memory one SM can hold (at most 4: 128 threads x ~100-150 registers).  Launch heuristic only.
inline int resident_blocks(size_t smem) { return std::max<int>(1, std::min<int>(4, (int) ((size_t) kMaxSmem / std::max<size_t>(smem, 1)))); }

/// The geometry of one Q8_0 GEMV launch.  Everything that touches the NUMBERS (p_log2) is a function of k alone; the rest (tile width, rows per block) only
/// decides which block computes which row.
struct Q8Plan {
    int nsb = 0;             // super-blocks per row
    int p_log2 = 0;          // log2 of the lanes per row
    int nt = 1;              // tokens per tile (1, 2, 4, 8)
    int ntiles = 1;
    int rows_per_block = 0;  // rows of ONE group a block takes
    int row_blocks = 0;      // per group (and matrix)
    size_t smem = 0;
    bool fast = false;
};
inline Q8Plan plan_q8(int rows_per_group, int groups_x_mats, int k, int T, int tok_sb_bytes, bool fast_ok) {
    Q8Plan pl;
    const int nb = k / 32;
    pl.nsb = (nb + 7) / 8;
    pl.p_log2 = ilog2_exact(std::min(32, pow2_ceil(pl.nsb)));
    const size_t tok = (size_t) pl.nsb * tok_sb_bytes;
    if (tok > (size_t) kMaxSmem) throw std::invalid_argument("ds41 dense: k too large for the activation tile in shared memory");
    pl.nt = pow2_ceil(std::min(T, kDenseMaxT));
    while (pl.nt > 1 && (size_t) pl.nt * tok > (size_t) kMaxSmem) pl.nt >>= 1;
    pl.ntiles = (T + pl.nt - 1) / pl.nt;
    pl.smem = (size_t) pl.nt * tok;
    const int gpw = 32 >> pl.p_log2;
    const int min_rows = kGemvWarps * gpw;
    // one wave: as many row blocks (per group / matrix) as the card holds at once, each with an equal share of the rows (launch heuristic only: it decides which block
    // computes which row, never what a row's value is)
    const int target = std::max(1, (resident_blocks(pl.smem) * sm_count()) / (pl.ntiles * groups_x_mats));
    pl.rows_per_block = std::max(min_rows, round_up((rows_per_group + target - 1) / target, gpw));
    pl.row_blocks = (rows_per_group + pl.rows_per_block - 1) / pl.rows_per_block;
    pl.fast = fast_ok && (k % 256 == 0);
    return pl;
}

template <class K>
inline void raise_smem_limit(K kernel, size_t smem) {
#if !defined(DS41_EMU)
    if (smem > 48 * 1024) cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSmem);
#else
    (void) kernel;
    (void) smem;
#endif
}

template <int NT, bool FAST>
inline void launch_q8_int8(const Q8Plan& pl, const void* wa, const void* wb, int nmat, int n, int k, const int8_t* xq, const float* xs, int T, float* ya, float* yb,
                           Stream stream) {
    auto* kern = q8_int8_kernel<NT, FAST>;
    raise_smem_limit(kern, pl.smem);
    const unsigned grid = (unsigned) ((size_t) pl.ntiles * pl.row_blocks * nmat);
    dev::launch(kern, dim3(grid), dim3(kGemvThreads), pl.smem, stream, static_cast<const uint8_t*>(wa), static_cast<const uint8_t*>(wb), n, k, xq, xs, T, ya, yb,
                pl.p_log2, pl.rows_per_block, pl.row_blocks, pl.ntiles);
    dev::check_launch("ds41_gemv_q8_int8");
}
template <int NT, bool FAST>
inline void launch_q8_f32(const Q8Plan& pl, const void* w, int groups, int R, int k, const float* x, int x_tok_stride, int x_group_stride, int T, float* y, Stream stream) {
    auto* kern = q8_f32_kernel<NT, FAST>;
    raise_smem_limit(kern, pl.smem);
    const unsigned grid = (unsigned) ((size_t) pl.ntiles * pl.row_blocks * groups);
    dev::launch(kern, dim3(grid), dim3(kGemvThreads), pl.smem, stream, static_cast<const uint8_t*>(w), R, k, x, x_tok_stride, x_group_stride, T, y, pl.p_log2,
                pl.rows_per_block, pl.row_blocks, pl.ntiles, groups * R);
    dev::check_launch("ds41_gemv_q8_f32");
}
template <int NT, bool BF16>
inline void launch_wide(const void* w, int n, int k, const float* x, int T, float* y, int rows_per_block, int ntiles, size_t smem, Stream stream) {
    auto* kern = wide_kernel<NT, BF16>;
    raise_smem_limit(kern, smem);
    const unsigned grid = (unsigned) ((size_t) ntiles * ((n + rows_per_block - 1) / rows_per_block));
    dev::launch(kern, dim3(grid), dim3(kWideThreads), smem, stream, static_cast<const uint8_t*>(w), n, k, x, T, y, rows_per_block, ntiles);
    dev::check_launch("ds41_gemv_wide");
}

inline void check_q8_args(int n, int k, int T, const char* what) {
    if (n < 1 || k < 32 || k % 32 != 0 || T < 1) throw std::invalid_argument(std::string("ds41 ") + what + ": need n >= 1, k a positive multiple of 32, T >= 1");
}
inline bool q8_fast_ok(const void* w, int k) { return k % 256 == 0 && reinterpret_cast<uintptr_t>(w) % 16 == 0; }

}  // namespace strata::ds41::cuda::dense

namespace strata::ds41::cuda {

template <class G>
void ds41_gemv_q8_int8_pair(Dev& dev, const void* w_a, const void* w_b, int n, int k, const int8_t* xq, const float* xs, int T, float* y_a, float* y_b, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    dense::check_q8_args(n, k, T, "gemv_q8_int8");
    const int nmat = y_b != nullptr ? 2 : 1;
    dense::check_ptr(w_a, 2, "w");
    if (nmat == 2) dense::check_ptr(w_b, 2, "w_b");
    dense::check_ptr(xq, 16, "xq");
    dense::check_ptr(xs, 4, "xs");
    dense::check_ptr(y_a, 4, "y");
    stream = stream_or_default(dev, stream);
    const bool fast = dense::q8_fast_ok(w_a, k) && (nmat == 1 || dense::q8_fast_ok(w_b, k));
    const dense::Q8Plan pl = dense::plan_q8(n, nmat, k, T, dense::kActSb, fast);
    if (nmat == 1) w_b = w_a, y_b = y_a;
    switch (pl.nt) {
#define DS41_DENSE_CASE(NT)                                                                                                                              \
    case NT:                                                                                                                                             \
        if (pl.fast) dense::launch_q8_int8<NT, true>(pl, w_a, w_b, nmat, n, k, xq, xs, T, y_a, y_b, stream);                                             \
        else dense::launch_q8_int8<NT, false>(pl, w_a, w_b, nmat, n, k, xq, xs, T, y_a, y_b, stream);                                                    \
        break;
        DS41_DENSE_CASE(1)
        DS41_DENSE_CASE(2)
        DS41_DENSE_CASE(4)
        DS41_DENSE_CASE(8)
#undef DS41_DENSE_CASE
        default: throw std::logic_error("ds41_gemv_q8_int8: bad tile");
    }
}

template <class G>
void ds41_gemv_q8_int8(Dev& dev, const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream stream) {
    ds41_gemv_q8_int8_pair<G>(dev, w, nullptr, n, k, xq, xs, T, y, nullptr, stream);
}

template <class G>
void ds41_gemv_q8_grouped_f32(Dev& dev, const void* w, int groups, int rows_per_group, int k_per_group, const float* x, int T, float* y, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (groups < 1) throw std::invalid_argument("ds41_gemv_q8_grouped_f32: groups < 1");
    dense::check_q8_args(rows_per_group, k_per_group, T, "gemv_q8_f32");
    dense::check_ptr(w, 2, "w");
    dense::check_ptr(x, 16, "x");
    dense::check_ptr(y, 4, "y");
    stream = stream_or_default(dev, stream);
    const dense::Q8Plan pl = dense::plan_q8(rows_per_group, groups, k_per_group, T, dense::kXSb, dense::q8_fast_ok(w, k_per_group));
    const int x_tok = groups * k_per_group;
    switch (pl.nt) {
#define DS41_DENSE_CASE(NT)                                                                                                                              \
    case NT:                                                                                                                                             \
        if (pl.fast) dense::launch_q8_f32<NT, true>(pl, w, groups, rows_per_group, k_per_group, x, x_tok, k_per_group, T, y, stream);                    \
        else dense::launch_q8_f32<NT, false>(pl, w, groups, rows_per_group, k_per_group, x, x_tok, k_per_group, T, y, stream);                           \
        break;
        DS41_DENSE_CASE(1)
        DS41_DENSE_CASE(2)
        DS41_DENSE_CASE(4)
        DS41_DENSE_CASE(8)
#undef DS41_DENSE_CASE
        default: throw std::logic_error("ds41_gemv_q8_grouped_f32: bad tile");
    }
}

template <class G>
void ds41_gemv_q8_f32(Dev& dev, const void* w, int n, int k, const float* x, int T, float* y, Stream stream) {
    ds41_gemv_q8_grouped_f32<G>(dev, w, 1, n, k, x, T, y, stream);
}

template <class G>
void ds41_wo_a(Dev& dev, const void* w, const float* x, int T, float* y, Stream stream) {
    ds41_gemv_q8_grouped_f32<G>(dev, w, G::kOGroups, G::kOLora, Derived<G>::kOGroupIn, x, T, y, stream);
}

namespace dense {
template <bool BF16>
inline void gemv_wide(Dev& dev, const void* w, int n, int k, const float* x, int T, float* y, Stream stream) {
    constexpr int kElems = BF16 ? 8 : 4;
    if (n < 1 || k < kElems || k % kElems != 0 || T < 1) throw std::invalid_argument("ds41_gemv_bf16/f32: need n >= 1, T >= 1, k a multiple of 8 (BF16) / 4 (F32)");
    check_ptr(w, 16, "w");
    check_ptr(x, 16, "x");
    check_ptr(y, 4, "y");
    stream = stream_or_default(dev, stream);
    const int nu = k / 4;
    const size_t tok = (size_t) (nu + (nu + 7) / 8) * 16;
    if (tok > (size_t) kMaxSmem) throw std::invalid_argument("ds41_gemv_bf16/f32: k too large for the activation tile in shared memory");
    int nt = pow2_ceil(std::min(T, kDenseMaxT));
    while (nt > 1 && (size_t) nt * tok > (size_t) kMaxSmem) nt >>= 1;
    const int ntiles = (T + nt - 1) / nt;
    const size_t smem = (size_t) nt * tok;
    const int target = std::max(1, (resident_blocks(smem) * sm_count()) / ntiles);
    const int rpb = std::max(kWideWarps, round_up((n + target - 1) / target, kWideWarps));
    switch (nt) {
#define DS41_DENSE_CASE(NT)                                                                                                                              \
    case NT:                                                                                                                                             \
        launch_wide<NT, BF16>(w, n, k, x, T, y, rpb, ntiles, smem, stream);                                                                              \
        break;
        DS41_DENSE_CASE(1)
        DS41_DENSE_CASE(2)
        DS41_DENSE_CASE(4)
        DS41_DENSE_CASE(8)
#undef DS41_DENSE_CASE
        default: throw std::logic_error("ds41_gemv_wide: bad tile");
    }
}
}  // namespace dense

template <class G>
void ds41_gemv_bf16(Dev& dev, const uint16_t* w, int n, int k, const float* x, int T, float* y, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    dense::gemv_wide<true>(dev, w, n, k, x, T, y, stream);
}
template <class G>
void ds41_gemv_f32(Dev& dev, const float* w, int n, int k, const float* x, int T, float* y, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    dense::gemv_wide<false>(dev, w, n, k, x, T, y, stream);
}
template <class G>
void ds41_head(Dev& dev, const uint16_t* w, const float* x, int T, float* logits, Stream stream) {
    ds41_gemv_bf16<G>(dev, w, G::kVocab, G::kHidden, x, T, logits, stream);
}

template <class G>
void ds41_rmsnorm(Dev& dev, const float* x, const float* w, int rows, int width, float* out, float eps, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (rows < 1 || width < 4 || width % 4 != 0) throw std::invalid_argument("ds41_rmsnorm: need rows >= 1 and width a positive multiple of 4");
    dense::check_ptr(x, 16, "x");
    dense::check_ptr(out, 16, "out");
    if (w != nullptr) dense::check_ptr(w, 16, "w");
    stream = stream_or_default(dev, stream);
    const unsigned grid = (unsigned) ((rows + dense::kNW - 1) / dense::kNW);
    dev::launch(dense::rmsnorm_kernel, dim3(grid), dim3(dense::kThreads), 0, stream, x, w, out, rows, width, eps);
    dev::check_launch("ds41_rmsnorm");
}

template <class G>
void ds41_rope(Dev& dev, const float* x, float* out, int T, int heads, int width, const RopeTable& table, int pos0, bool inverse, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    const int rd = G::kRopeDim;
    if (T < 1 || heads < 1 || width < rd || width % 2 != 0) throw std::invalid_argument("ds41_rope: need T, heads >= 1 and an even width >= kRopeDim");
    if (pos0 < 0 || pos0 + T > table.rows) throw std::invalid_argument("ds41_rope: positions outside the table");
    dense::check_ptr(x, 8, "x");
    dense::check_ptr(out, 8, "out");
    dense::check_ptr(table.cos, 4, "table.cos");
    dense::check_ptr(table.sin, 4, "table.sin");
    if (out != x && ((out < x + (size_t) T * heads * width) && (x < out + (size_t) T * heads * width))) throw std::invalid_argument("ds41_rope: out partially overlaps x");
    stream = stream_or_default(dev, stream);
    const int rows = T * heads;
    const long long total = (long long) rows * (width / 2);
    const unsigned grid = (unsigned) ((total + dense::kThreads - 1) / dense::kThreads);
    dev::launch(dense::rope_kernel, dim3(grid), dim3(dense::kThreads), 0, stream, x, out, rows, heads, width, rd, table.cos, table.sin, pos0, (int) inverse);
    dev::check_launch("ds41_rope");
}

template <class G>
void ds41_swiglu(Dev& dev, float* g_h, const float* u, int n, float swiglu_limit, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (n < 1) throw std::invalid_argument("ds41_swiglu: n < 1");
    dense::check_ptr(g_h, 4, "g");
    dense::check_ptr(u, 4, "u");
    stream = stream_or_default(dev, stream);
    const unsigned grid = (unsigned) (((size_t) n + dense::kThreads - 1) / dense::kThreads);
    dev::launch(dense::swiglu_kernel<false>, dim3(grid), dim3(dense::kThreads), 0, stream, g_h, u, n, 0, swiglu_limit, static_cast<int8_t*>(nullptr),
                static_cast<float*>(nullptr));
    dev::check_launch("ds41_swiglu");
}

template <class G>
void ds41_shared_expert_q(Dev& dev, const SharedExpertWeights& w, const int8_t* xq, const float* xs, int T, float* y, const SharedExpertScratch& s, float swiglu_limit,
                          Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (T < 1) throw std::invalid_argument("ds41_shared_expert: T < 1");
    stream = stream_or_default(dev, stream);
    // gate and up in ONE launch (the same activations, two matrices), then SwiGLU + the quantisation of h, then down
    ds41_gemv_q8_int8_pair<G>(dev, w.w1, w.w3, G::kFF, G::kHidden, xq, xs, T, s.g, s.u, stream);
    dense::check_ptr(s.hq, 16, "scratch.hq");
    dense::check_ptr(s.hs, 4, "scratch.hs");
    const int nb = G::kFF / 32;
    const unsigned grid = (unsigned) (((size_t) T * nb + dense::kNW - 1) / dense::kNW);
    dev::launch(dense::swiglu_kernel<true>, dim3(grid), dim3(dense::kThreads), 0, stream, s.g, static_cast<const float*>(s.u), T, G::kFF, swiglu_limit, s.hq, s.hs);
    dev::check_launch("ds41_shared_expert swiglu");
    ds41_gemv_q8_int8<G>(dev, w.w2, G::kHidden, G::kFF, s.hq, s.hs, T, y, stream);
}

template <class G>
void ds41_shared_expert(Dev& dev, const SharedExpertWeights& w, const float* x, int T, float* y, const SharedExpertScratch& s, bool int8_act, float swiglu_limit,
                        Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (T < 1) throw std::invalid_argument("ds41_shared_expert: T < 1");
    stream = stream_or_default(dev, stream);
    if (int8_act) {
        ds41_quantize_acts<G>(dev, x, T, G::kHidden, s.xq, s.xs, stream, ActOrder::kNatural);       // DS-D / DS1-G's quantiser, plain per-32 layout
        ds41_shared_expert_q<G>(dev, w, s.xq, s.xs, T, y, s, swiglu_limit, stream);
    } else {
        ds41_gemv_q8_f32<G>(dev, w.w1, G::kFF, G::kHidden, x, T, s.g, stream);
        ds41_gemv_q8_f32<G>(dev, w.w3, G::kFF, G::kHidden, x, T, s.u, stream);
        ds41_swiglu<G>(dev, s.g, s.u, T * G::kFF, swiglu_limit, stream);                  // h over g
        ds41_gemv_q8_f32<G>(dev, w.w2, G::kHidden, G::kFF, s.g, T, y, stream);
    }
}

template <class G>
void ds41_argmax(Dev& dev, const float* logits, int T, int32_t* idx, float* val, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (T < 1 || G::kVocab % 4 != 0) throw std::invalid_argument("ds41_argmax: T >= 1 and kVocab % 4 == 0 required");
    dense::check_ptr(logits, 16, "logits");
    dense::check_ptr(idx, 4, "idx");
    dense::check_ptr(val, 4, "val");
    stream = stream_or_default(dev, stream);
    dev::launch(dense::argmax_kernel, dim3((unsigned) T), dim3(dense::kThreads), 0, stream, logits, G::kVocab, idx, val);
    dev::check_launch("ds41_argmax");
}

template <class G>
void ds41_topk(Dev& dev, const float* logits, int T, int k, int32_t* idx, float* val, void* scratch, size_t scratch_bytes, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (T < 1 || k < 1 || k > dense::kTopMaxK || k > G::kVocab) throw std::invalid_argument("ds41_topk: need T >= 1 and 1 <= k <= 64 (and <= kVocab)");
    const int slices = (G::kVocab + dense::kTopSlice - 1) / dense::kTopSlice;
    const int m = slices * k;
    if (m > dense::kThreads * dense::kMergeNE) throw std::invalid_argument("ds41_topk: too many candidates (slices * k)");
    if (scratch_bytes < topk_scratch_bytes<G>(T, k)) throw std::invalid_argument("ds41_topk: scratch too small");
    dense::check_ptr(logits, 4, "logits");
    dense::check_ptr(idx, 4, "idx");
    dense::check_ptr(val, 4, "val");
    dense::check_ptr(scratch, 16, "scratch");
    stream = stream_or_default(dev, stream);
    auto* cand_key = static_cast<uint32_t*>(scratch);
    auto* cand_idx = reinterpret_cast<int32_t*>(cand_key + (size_t) T * m);
    dev::launch(dense::topk_slice_kernel, dim3((unsigned) slices, (unsigned) T), dim3(dense::kThreads), 0, stream, logits, G::kVocab, k, slices, cand_key, cand_idx);
    dev::check_launch("ds41_topk slice");
    dev::launch(dense::topk_merge_kernel, dim3((unsigned) T), dim3(dense::kThreads), 0, stream, static_cast<const uint32_t*>(cand_key),
                static_cast<const int32_t*>(cand_idx), m, k, idx, val);
    dev::check_launch("ds41_topk merge");
}

template <class G>
void ds41_f32_add(Dev& dev, const float* a, const float* b, int n, float* out, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (n < 4 || n % 4 != 0) throw std::invalid_argument("ds41_f32_add: n must be a positive multiple of 4");
    dense::check_ptr(a, 16, "a");
    dense::check_ptr(b, 16, "b");
    dense::check_ptr(out, 16, "out");
    stream = stream_or_default(dev, stream);
    dev::launch(dense::add_kernel, dim3((unsigned) ((n / 4 + dense::kThreads - 1) / dense::kThreads)), dim3(dense::kThreads), 0, stream, a, b, n / 4, out);
    dev::check_launch("ds41_f32_add");
}
template <class G>
void ds41_f32_scale(Dev& dev, const float* a, float s, int n, float* out, Stream stream) {
    static_assert(geom_ok<G>(), "G violates the kernel constraints of geom.hpp");
    if (n < 4 || n % 4 != 0) throw std::invalid_argument("ds41_f32_scale: n must be a positive multiple of 4");
    dense::check_ptr(a, 16, "a");
    dense::check_ptr(out, 16, "out");
    stream = stream_or_default(dev, stream);
    dev::launch(dense::scale_kernel, dim3((unsigned) ((n / 4 + dense::kThreads - 1) / dense::kThreads)), dim3(dense::kThreads), 0, stream, a, s, n / 4, out);
    dev::check_launch("ds41_f32_scale");
}

}  // namespace strata::ds41::cuda

/// Explicit instantiation of every host entry point for geometry G (dense.cu: RealGeom; dense_emu_impl.cpp: RealGeom and MiniGeom).
#define DS41_INSTANTIATE_DENSE(G)                                                                                                                        \
    namespace strata::ds41::cuda {                                                                                                                       \
    template void ds41_gemv_q8_int8<G>(Dev&, const void*, int, int, const int8_t*, const float*, int, float*, Stream);                                   \
    template void ds41_gemv_q8_int8_pair<G>(Dev&, const void*, const void*, int, int, const int8_t*, const float*, int, float*, float*, Stream);        \
    template void ds41_gemv_q8_f32<G>(Dev&, const void*, int, int, const float*, int, float*, Stream);                                                   \
    template void ds41_gemv_q8_grouped_f32<G>(Dev&, const void*, int, int, int, const float*, int, float*, Stream);                                      \
    template void ds41_wo_a<G>(Dev&, const void*, const float*, int, float*, Stream);                                                                    \
    template void ds41_gemv_bf16<G>(Dev&, const uint16_t*, int, int, const float*, int, float*, Stream);                                                 \
    template void ds41_gemv_f32<G>(Dev&, const float*, int, int, const float*, int, float*, Stream);                                                     \
    template void ds41_head<G>(Dev&, const uint16_t*, const float*, int, float*, Stream);                                                                \
    template void ds41_rmsnorm<G>(Dev&, const float*, const float*, int, int, float*, float, Stream);                                                    \
    template void ds41_rope<G>(Dev&, const float*, float*, int, int, int, const RopeTable&, int, bool, Stream);                                          \
    template void ds41_swiglu<G>(Dev&, float*, const float*, int, float, Stream);                                                                        \
    template void ds41_shared_expert<G>(Dev&, const SharedExpertWeights&, const float*, int, float*, const SharedExpertScratch&, bool, float, Stream);   \
    template void ds41_shared_expert_q<G>(Dev&, const SharedExpertWeights&, const int8_t*, const float*, int, float*, const SharedExpertScratch&, float,  \
                                          Stream);                                                                                                       \
    template void ds41_argmax<G>(Dev&, const float*, int, int32_t*, float*, Stream);                                                                     \
    template void ds41_topk<G>(Dev&, const float*, int, int, int32_t*, float*, void*, size_t, Stream);                                                   \
    template void ds41_f32_add<G>(Dev&, const float*, const float*, int, float*, Stream);                                                                \
    template void ds41_f32_scale<G>(Dev&, const float*, float, int, float*, Stream);                                                                     \
    }
