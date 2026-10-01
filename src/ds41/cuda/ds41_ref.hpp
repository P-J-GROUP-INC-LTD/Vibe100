// src/ds41/cuda/ds41_ref.hpp - DS-D: host references (FP64) and generators for the GPU router / split / expert parity programs.
// Plain C++, no CUDA.  Shared by the V100 parity programs and by the CPU emulation test.
//
// Semantics are docs/deepseek/CONTRACTS.md; the MXFP4 decode is ggml's (dequantize_row_mxfp4: kvalues_fp4 * ggml_e8m0_to_fp32_half,
// written out below including e = 0, 1 and 255).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "strata/ds41/cuda/ds41_cuda.hpp"
#include "strata/ds41/geometry.hpp"

namespace strata::ds41::cuda::ref {

inline constexpr int8_t kFp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/// ggml_e8m0_to_fp32_half (ggml-impl.h): 2^(x-128); x = 0 and 1 are the denormals 2^-128 and 2^-127, x = 255 is 2^127.
inline float e8m0_half(uint8_t x) {
    uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t) (x - 1) << 23);
    float r;
    std::memcpy(&r, &bits, 4);
    return r;
}

inline float bf16_to_f32(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
inline uint16_t f32_to_bf16(float f) {                  // round to nearest even (finite inputs)
    uint32_t i;
    std::memcpy(&i, &f, 4);
    i += 0x7FFFu + ((i >> 16) & 1u);
    return (uint16_t) (i >> 16);
}

// ---- deterministic random numbers ------------------------------------------------------------------------------------
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint32_t u32() { return (uint32_t) g(); }
    double uniform() { return (double) (g() >> 11) * (1.0 / 9007199254740992.0); }           // [0, 1)
    int range(int lo, int hi) { return lo + (int) (g() % (uint64_t) (hi - lo + 1)); }         // inclusive
    double normal() {
        const double u1 = uniform() + 1e-300, u2 = uniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    }
};

// ---- the activation quantiser (the CPU's rule: strata::ds41::cpu::quantize_act; CONTRACTS.md "Activations") ------------------------
inline uint32_t f32_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
inline float bits_f32(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
/// x[32] -> q[32] natural order, scale d.  The largest magnitude is the INTEGER maximum of (bits & 0x7FFFFFFF);
///   >= 0x7F800000 (an Inf or a NaN in the block):  d = the canonical NaN 0x7FC00000, every q = 0
///   <  0x0D800000 (amax < 2^-100, zero included):   d = 0, every q = 0
///   otherwise d = amax / 127, id = 127 / amax (FP32 divisions), q = lrintf(x * id) (FP32 product, ties to EVEN) clamped to +-127.
inline void quantize_block(const float* x, int8_t* q, float& d) {
    uint32_t m = 0;
    for (int j = 0; j < 32; ++j) m = std::max(m, f32_bits(x[j]) & 0x7FFFFFFFu);
    for (int j = 0; j < 32; ++j) q[j] = 0;
    if (m >= 0x7F800000u) {
        d = bits_f32(0x7FC00000u);
        return;
    }
    if (m < 0x0D800000u) {
        d = 0.0f;
        return;
    }
    const float amax = bits_f32(m);
    d = amax / 127.0f;
    const float id = 127.0f / amax;
    for (int j = 0; j < 32; ++j) {
        volatile float t = x[j] * id;
        long v = std::lrintf(t);
        q[j] = (int8_t) std::min(127L, std::max(-127L, v));
    }
}
/// n values (multiple of 32): q natural [n], d [n/32].
inline void quantize_vec(const float* x, int n, std::vector<int8_t>& q, std::vector<float>& d) {
    q.assign(n, 0);
    d.assign(n / 32, 0.0f);
    for (int b = 0; b < n / 32; ++b) quantize_block(x + 32 * b, q.data() + 32 * b, d[b]);
}
/// the kernels' interleaved block layout <-> natural order
inline void to_perm(const int8_t* nat, int8_t* perm, int nblocks) {
    for (int b = 0; b < nblocks; ++b)
        for (int j = 0; j < 32; ++j) perm[32 * b + act_perm_pos(j)] = nat[32 * b + j];
}
inline void from_perm(const int8_t* perm, int8_t* nat, int nblocks) {
    for (int b = 0; b < nblocks; ++b)
        for (int j = 0; j < 32; ++j) nat[32 * b + j] = perm[32 * b + act_perm_pos(j)];
}

// ---- MXFP4 blobs -----------------------------------------------------------------------------------------------------
/// How the random blob's scale bytes and codes are drawn.
struct BlobStyle {
    int gate_e_lo = 119, gate_e_hi = 123;       // gate / up scale bytes (uniform)
    int down_e_lo = 118, down_e_hi = 122;       // down scale bytes
};

/// A random expert blob [gate][up][down] (kBlobBytes): every nibble uniform over the 16 codes, scale bytes per the style.
inline void gen_blob(uint8_t* blob, Rng& rng, const BlobStyle& st = BlobStyle()) {
    auto fill = [&](uint8_t* base, int rows, int nblk, int elo, int ehi) {
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nblk; ++b) {
                uint8_t* blk = base + ((size_t) r * nblk + b) * kBlockBytes;
                blk[0] = (uint8_t) rng.range(elo, ehi);
                for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) rng.u32();
            }
    };
    fill(blob + kBlobGate, kFF, kGateRowBlocks, st.gate_e_lo, st.gate_e_hi);
    fill(blob + kBlobUp, kFF, kGateRowBlocks, st.gate_e_lo, st.gate_e_hi);
    fill(blob + kBlobDown, kHidden, kDownRowBlocks, st.down_e_lo, st.down_e_hi);
}

/// Block `b` of a row with the 16 codes in order (0..15 in the low nibbles of bytes 0..7, 15..0 in bytes 8..15): used to make
/// sure every code appears at every position of some block.
inline void put_all_codes(uint8_t* blk, int rot) {
    for (int j = 0; j < 16; ++j) blk[1 + j] = (uint8_t) ((((j + rot) & 15)) | ((((j + 7 + rot) & 15)) << 4));
}

/// dot of one MXFP4 row (nblk blocks) with natural-order int8 activations q[nblk*32] and scales d[nblk] -> exact integer partial sums
/// times 2^(e-128) * d, in double.  `mass` = the sum of the absolute block terms (for the error bound).
inline double dot_q(const uint8_t* row, int nblk, const int8_t* q, const float* d, double* mass = nullptr) {
    double acc = 0.0, m = 0.0;
    for (int b = 0; b < nblk; ++b) {
        const uint8_t* blk = row + (size_t) b * kBlockBytes;
        int isum = 0;
        for (int j = 0; j < 16; ++j) {
            isum += (int) kFp4[blk[1 + j] & 15] * q[32 * b + j];
            isum += (int) kFp4[blk[1 + j] >> 4] * q[32 * b + 16 + j];
        }
        const double term = (double) e8m0_half(blk[0]) * (double) d[b] * (double) isum;
        acc += term;
        m += std::fabs(term);
    }
    if (mass) *mass = m;
    return acc;
}
/// the same with exact FP32 activations x[nblk*32].
inline double dot_f(const uint8_t* row, int nblk, const float* x, double* mass = nullptr) {
    double acc = 0.0, m = 0.0;
    for (int b = 0; b < nblk; ++b) {
        const uint8_t* blk = row + (size_t) b * kBlockBytes;
        double s = 0.0;
        for (int j = 0; j < 16; ++j) {
            s += (double) kFp4[blk[1 + j] & 15] * (double) x[32 * b + j];
            s += (double) kFp4[blk[1 + j] >> 4] * (double) x[32 * b + 16 + j];
        }
        const double term = (double) e8m0_half(blk[0]) * s;
        acc += term;
        m += std::fabs(term);
    }
    if (mass) *mass = m;
    return acc;
}

inline double silu(double x) { return x / (1.0 + std::exp(-x)); }

/// The clamps of CONTRACTS.md with NaN PROPAGATING (as torch.clamp does): g = g > 10 ? 10 : g, u = u > 10 ? 10 : (u < -10 ? -10 : u).
inline double clamp_g(double g) { return g > (double) kSwigluLimit ? (double) kSwigluLimit : g; }
inline double clamp_u(double u) { return u > (double) kSwigluLimit ? (double) kSwigluLimit : (u < -(double) kSwigluLimit ? -(double) kSwigluLimit : u); }
/// h[r] = silu(min(g,10)) * clamp(u,-10,10) * w for the 2304 intermediate rows, from natural-order int8 activations.
inline void expert_h_q(const uint8_t* blob, const int8_t* xq, const float* xs, double w, std::vector<double>& h) {
    h.assign(kFF, 0.0);
    for (int r = 0; r < kFF; ++r) {
        const double g = dot_q(blob + kBlobGate + (size_t) r * kGateRowBytes, kGateRowBlocks, xq, xs);
        const double u = dot_q(blob + kBlobUp + (size_t) r * kGateRowBytes, kGateRowBlocks, xq, xs);
        h[r] = silu(clamp_g(g)) * clamp_u(u) * w;
    }
}
/// the same with exact FP32 x.
inline void expert_h_f(const uint8_t* blob, const float* x, double w, std::vector<double>& h) {
    h.assign(kFF, 0.0);
    for (int r = 0; r < kFF; ++r) {
        const double g = dot_f(blob + kBlobGate + (size_t) r * kGateRowBytes, kGateRowBlocks, x);
        const double u = dot_f(blob + kBlobUp + (size_t) r * kGateRowBytes, kGateRowBlocks, x);
        h[r] = silu(clamp_g(g)) * clamp_u(u) * w;
    }
}
/// y[row] = W2 . h, h given as natural-order int8 + scales; also the error-bound mass per row.
inline void expert_down_q(const uint8_t* blob, const int8_t* hq, const float* hs, std::vector<double>& y, std::vector<double>& mass) {
    y.assign(kHidden, 0.0);
    mass.assign(kHidden, 0.0);
    for (int r = 0; r < kHidden; ++r) y[r] = dot_q(blob + kBlobDown + (size_t) r * kDownRowBytes, kDownRowBlocks, hq, hs, &mass[r]);
}
/// y[row] = W2 . h with h in double (exact; the dot is accumulated in double from float-rounded h).
inline void expert_down_f(const uint8_t* blob, const std::vector<double>& h, std::vector<double>& y) {
    y.assign(kHidden, 0.0);
    std::vector<float> hf(kFF);
    for (int i = 0; i < kFF; ++i) hf[i] = (float) h[i];
    for (int r = 0; r < kHidden; ++r) y[r] = dot_f(blob + kBlobDown + (size_t) r * kDownRowBytes, kDownRowBlocks, hf.data());
}

// ---- router ------------------------------------------------------------------------------------------------------------
inline double softplus_sqrt(double l) {
    const double sp = std::max(l, 0.0) + std::log1p(std::exp(-std::fabs(l)));
    return std::sqrt(sp);
}
struct RouterRef {
    std::vector<double> logits, mass;        // [384]
    std::vector<double> s, v;                // [384] sqrt(softplus), s + bias
    int ids[kTopK];                          // top-6, (value desc, index asc)
    double w[kTopK];
};
inline void router_ref(const float* x, const uint16_t* wg, const float* bias, RouterRef& r) {
    r.logits.assign(kExperts, 0.0);
    r.mass.assign(kExperts, 0.0);
    r.s.assign(kExperts, 0.0);
    r.v.assign(kExperts, 0.0);
    for (int e = 0; e < kExperts; ++e) {
        double acc = 0.0, m = 0.0;
        for (int k = 0; k < kHidden; ++k) {
            const double t = (double) x[k] * (double) bf16_to_f32(wg[(size_t) e * kHidden + k]);
            acc += t;
            m += std::fabs(t);
        }
        r.logits[e] = acc;
        r.mass[e] = m;
        r.s[e] = softplus_sqrt(acc);
        r.v[e] = r.s[e] + (double) bias[e];
    }
    std::vector<int> order(kExperts);
    for (int e = 0; e < kExperts; ++e) order[e] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return r.v[a] > r.v[b]; });     // ties: lowest index first
    double sum = 0.0;
    for (int i = 0; i < kTopK; ++i) {
        r.ids[i] = order[i];
        sum += r.s[order[i]];
    }
    for (int i = 0; i < kTopK; ++i) r.w[i] = r.s[order[i]] / (sum + 1e-20) * 1.5;
}

}  // namespace strata::ds41::cuda::ref
