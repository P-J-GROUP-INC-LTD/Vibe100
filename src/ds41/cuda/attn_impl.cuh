// src/ds41/cuda/attn_impl.cuh - DS1-C: the attention kernels of DeepSeek-V4.1-Flash (V100, sm_70) and their host wrappers.  Included by attn_real.cu
// (nvcc, AttnKernels<RealGeom>) and by the CPU-emulation tests (-DDS41_EMU, any G).  Declarations: include/strata/ds41/cuda/attn.hpp.
//
// WHAT EACH KERNEL MIRRORS (ref/ds41/attention.py, itself a transcription of the reference's model.py / kernel.py):
//   rmsnorm_row      ops.rmsnorm                      x * (1 / sqrt(mean(x^2) + eps)) * w, FP32 (division by the square root, as the oracle)
//   q_rope           attention_layer, rope.apply_rope_tail on q   adjacent pairs of the LAST kRopeDim channels: (a, b) -> (a c - b s, a s + b c)
//   swa_kv           window_kv                        rmsnorm(wkv x) -> RoPE -> act_quant_fp8(32) -> the ring row
//   compress_step    compressor                       group state, softmax over the group, sum, RMSNorm; then attention_layer: RoPE at group * ratio,
//                                                     fp4_quant_e4m3(16) -> the comp_kv row.  Also hands the PRE-RoPE latent to the indexer.
//   index_k_finish   indexer (owns_k branch)          rmsnorm(wk latent) -> RoPE -> fp4_quant_e8m0(32) -> the index_k row
//   index_q_finish   indexer                          RoPE on iq, fp4_quant_e8m0(32)
//   index_scores     indexer                          score[t] = sum_h relu(iq[h] . index_k[t]) * w[h]   (heads summed in order 0..H-1, like np.sum)
//                                                     + `np.where(shared.candidates, score, -inf)` on the layers after the candidate source
//   block_scores     select_candidate_blocks          block max over `block` positions, newest reachable block pinned to +inf
//   topk_select      select_candidate_blocks / indexer   top-k with the oracle's tie rule (np.argsort(-score, kind="stable"): ties -> the LOWER index),
//                                                     -inf never selected, ascending order of the selected indices (the oracle's np.sort(pick))
//   sparse_attn      sparse_attn + the inverse RoPE   window ring in the oracle's slot order, then the selected compressed rows; logits * scale;
//                                                     m = max(max logit, sink); denominator + exp(sink - m); o = sum p kv / denom
// NUMERICS: FP32 throughout; every operation the oracle rounds separately is a separate IEEE operation here (fmul_rn / fadd_rn / fdiv_rn: no
// contraction), the dot products use fmaf in a fixed order.  The three fake-quantisers are BIT-EXACT against ref/ds41/quant.py (the test checks
// the bits).  Differences from the oracle: the order of the dot-product / sum reductions (the oracle's are numpy's), and expf.
// DETERMINISM: no atomics on floats; the only shared atomics are integer counters in the radix histogram of topk_select.
//
// BLOCK SHAPES (all block sizes are multiples of 32: every warp is full, every shuffle is a full-warp one):
//   rmsnorm_row 256 | q_rope kHeads x round32(kRopeDim) | swa_kv, compress_step kHeadDim | index_k_finish kIdxDim | index_q_finish kIdxHeads x kIdxDim |
//   index_scores 128 (one position per thread) | block_scores 128 | topk_select kAttnTopkThreads, ONE block | sparse_attn 256 (8 warps), kHeads / 8 blocks.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "ds41_dev.cuh"
#include "strata/ds41/cuda/attn.hpp"

#define ATTN_ALIGN16 __attribute__((aligned(16)))

namespace strata::ds41::cuda::adev {

using namespace ::strata::ds41::cuda::dev;

// ---------------------------------------------------------------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------------------------------------------------------------
/// 2^e as a float, e in [-126, 127] (a normal number).
DS41_FI float exp2i(int e) { return u2f((uint32_t) (e + 127) << 23); }

DS41_FI void smem_add(int* p, int v) {
#if defined(DS41_EMU)
    *p += v;                       // the emulator runs one fiber at a time and yields only at barriers: a plain add is atomic there
#else
    atomicAdd(p, v);
#endif
}

/// Sum over the NT threads of the block (NT a multiple of 32): a butterfly per warp (every lane ends with the same bits), then the warp totals added
/// in warp order by every thread.  Includes two barriers; s_red holds NT / 32 floats.  The result is the same for every thread and every run.
template <int NT>
DS41_FI float block_sum(float v, float* s_red) {
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) v = fadd_rn(v, shfl_xor(v, off));
    if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = v;
    sync_block();
    float t = s_red[0];
    DS41_UNROLL
    for (int w = 1; w < NT / 32; ++w) t = fadd_rn(t, s_red[w]);
    sync_block();
    return t;
}

/// sqrt(mean(x^2) + eps) from the sum of squares of n values (the oracle divides by it: `x / np.sqrt(var + eps)`).
DS41_FI float rms_den(float sum_sq, int n, float eps) { return sqrtf(fadd_rn(fdiv_rn(sum_sq, (float) n), eps)); }

/// RoPE of one element of the rotated tail.  v = this element, partner = shfl_xor(v, 1) (the other element of its adjacent pair), j = index within
/// the tail (0 .. kRopeDim - 1).  Forward: (a, b) -> (a c - b s, a s + b c); inverse: s -> -s.  Each product rounded, as numpy.
template <bool INV>
DS41_FI float rope_elem(float v, float partner, int j, const float* DS41_RESTRICT cosr, const float* DS41_RESTRICT sinr) {
    const float c = cosr[j >> 1];
    float s = sinr[j >> 1];
    if (INV) s = -s;
    const bool odd = (j & 1) != 0;
    const float a = odd ? partner : v, b = odd ? v : partner;
    return odd ? fadd_rn(fmul_rn(a, s), fmul_rn(b, c)) : fadd_rn(fmul_rn(a, c), -fmul_rn(b, s));
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the three fake-quantisers (ref/ds41/quant.py: act_quant_fp8, fp4_quant_e8m0, fp4_quant_e4m3), bit-exact.
// Each is called by a FULL warp (fq_fp8_32, fq_fp4_e8m0_32: lane l holds element l of the 32-block) or by a half-warp group
// (fq_fp4_e4m3_16: lanes 0-15 and 16-31 are two independent 16-blocks).
// ---------------------------------------------------------------------------------------------------------------------------------
/// RNE onto the e4m3fn grid (quant.py: round_e4m3), no saturation (the caller clamps to +-448 first).
DS41_FI float round_e4m3(float v) {
    const float a = fabsf(v);
    const int E = (int) ((f2u(a) >> 23) & 0xFFu);                   // biased exponent (0 for zero and denormals)
    const int ex = E - 127 > -6 ? E - 127 : -6;                     // floor(log2 a), clamped at the subnormal binade
    const int q = f2i_rn(fmul_rn(a, exp2i(3 - ex)));                // a / 2^(ex-3): exact
    const float r = fmul_rn((float) q, exp2i(ex - 3));
    return u2f(f2u(r) | (f2u(v) & 0x80000000u));                    // copysign
}
/// RNE onto the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating at +-6 (quant.py: round_e2m1).
DS41_FI float round_e2m1(float v) {
    const float a = fminf(fabsf(v), 6.0f);
    const int E = (int) ((f2u(a) >> 23) & 0xFFu);
    const int ex = E - 127 > 0 ? E - 127 : 0;
    const int q = f2i_rn(fmul_rn(a, exp2i(1 - ex)));
    const float r = fmul_rn((float) q, exp2i(ex - 1));
    return u2f(f2u(r) | (f2u(v) & 0x80000000u));
}
/// 2^ceil(log2 t) for a positive normal t (quant.py: _round_scale_pow2).
DS41_FI float pow2_ceil(float t) {
    const uint32_t u = f2u(t);
    const int E = (int) ((u >> 23) & 0xFFu);
    const int c = E - 127 + ((u & 0x7FFFFFu) != 0 ? 1 : 0);
    return exp2i(c);
}
DS41_FI float warp_absmax(float v) {
    v = fabsf(v);
    DS41_UNROLL
    for (int off = 16; off > 0; off >>= 1) {
        const float o = shfl_xor(v, off);
        v = o > v ? o : v;
    }
    return v;
}
DS41_FI float halfwarp_absmax(float v) {
    v = fabsf(v);
    DS41_UNROLL
    for (int off = 8; off > 0; off >>= 1) {
        const float o = shfl_xor(v, off);
        v = o > v ? o : v;
    }
    return v;
}
/// act_quant_fp8(x, 32): s = 2^ceil(log2(max(amax, 1e-4) * (1/448)));  y = e4m3(clamp(x / s, +-448)) * s
DS41_FI float fq_fp8_32(float v) {
    const float amax = fmaxf(warp_absmax(v), 1e-4f);
    const float s = pow2_ceil(fmul_rn(amax, (float) (1.0 / 448.0)));
    const float t = fminf(fmaxf(fdiv_rn(v, s), -448.0f), 448.0f);
    return fmul_rn(round_e4m3(t), s);
}
/// fp4_quant_e8m0(x, 32): s = 2^ceil(log2(max(amax, 6 * 2^-126) * (1/6)));  y = e2m1(clamp(x / s, +-6)) * s
DS41_FI float fq_fp4_e8m0_32(float v) {
    const float amax = fmaxf(warp_absmax(v), 6.0f * exp2i(-126));
    const float s = pow2_ceil(fmul_rn(amax, (float) (1.0 / 6.0)));
    const float t = fminf(fmaxf(fdiv_rn(v, s), -6.0f), 6.0f);
    return fmul_rn(round_e2m1(t), s);
}
/// fp4_quant_e4m3(x, 16): s = min(e4m3(max(amax, 6 * 2^-9) / 6), 448);  y = e2m1(clamp(x / s, +-6)) * s     (s is an ordinary e4m3 number)
DS41_FI float fq_fp4_e4m3_16(float v) {
    const float amax = fmaxf(halfwarp_absmax(v), 6.0f * exp2i(-9));
    const float s = fminf(round_e4m3(fdiv_rn(amax, 6.0f)), 448.0f);
    const float t = fminf(fmaxf(fdiv_rn(v, s), -6.0f), 6.0f);
    return fmul_rn(round_e2m1(t), s);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------------------------------------------------------------
template <int NT>
DS41_KERNEL DS41_LAUNCH_BOUNDS(NT) void rmsnorm_row_kernel(const float* DS41_RESTRICT x, const float* DS41_RESTRICT w, int n, float eps,
                                                           float* DS41_RESTRICT y) {
    DS41_SHARED float s_red[NT / 32];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += NT) ss = fadd_rn(ss, fmul_rn(x[i], x[i]));
    const float den = rms_den(block_sum<NT>(ss, s_red), n, eps);
    for (int i = threadIdx.x; i < n; i += NT) y[i] = fmul_rn(w[i], fdiv_rn(x[i], den));
}

constexpr int round32(int v) { return (v + 31) & ~31; }

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(round32(G::kRopeDim)) void q_rope_kernel(float* DS41_RESTRICT q, const float* DS41_RESTRICT cosr,
                                                                        const float* DS41_RESTRICT sinr) {
    constexpr int D = G::kHeadDim, R = G::kRopeDim;
    const int j = threadIdx.x;
    float* row = q + (size_t) blockIdx.x * D + (D - R);
    const float v = j < R ? row[j] : 0.0f;
    const float partner = shfl_xor(v, 1);
    if (j < R) row[j] = rope_elem<false>(v, partner, j, cosr, sinr);
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(G::kHeadDim) void swa_kv_kernel(const float* DS41_RESTRICT kv_raw, const float* DS41_RESTRICT nw, float eps,
                                                               const float* DS41_RESTRICT cosr, const float* DS41_RESTRICT sinr, int fp8,
                                                               float* DS41_RESTRICT ring_row) {
    constexpr int D = G::kHeadDim, R = G::kRopeDim;
    static_assert(D % 32 == 0 && D <= 1024);
    DS41_SHARED float s_red[D / 32];
    const int e = threadIdx.x;
    const float raw = kv_raw[e];
    const float den = rms_den(block_sum<D>(fmul_rn(raw, raw), s_red), D, eps);
    float v = fmul_rn(nw[e], fdiv_rn(raw, den));
    const float partner = shfl_xor(v, 1);
    if (e >= D - R) v = rope_elem<false>(v, partner, e - (D - R), cosr, sinr);
    if (fp8) v = fq_fp8_32(v);
    ring_row[e] = v;
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(G::kHeadDim) void compress_step_kernel(AttnCompressArgs a) {
    constexpr int D = G::kHeadDim, R = G::kRopeDim;
    DS41_SHARED float s_red[D / 32];
    const int e = threadIdx.x;
    float lat;
    if (a.ratio > 1) {
        a.kv_state[a.slot * D + e] = a.kv_c[e];
        a.score_state[a.slot * D + e] = a.score_c[e];
        if (!a.complete) return;                                     // block-uniform: this thread's channel only ever touches its own state
        float sc[kAttnMaxCompressRatio], kv[kAttnMaxCompressRatio], ex[kAttnMaxCompressRatio];
        DS41_UNROLL
        for (int r = 0; r < kAttnMaxCompressRatio; ++r) {
            sc[r] = r < a.ratio ? a.score_state[r * D + e] : 0.0f;
            kv[r] = r < a.ratio ? a.kv_state[r * D + e] : 0.0f;
        }
        float mx = sc[0];
        DS41_UNROLL
        for (int r = 1; r < kAttnMaxCompressRatio; ++r) mx = (r < a.ratio && sc[r] > mx) ? sc[r] : mx;
        float sum = 0.0f;
        DS41_UNROLL
        for (int r = 0; r < kAttnMaxCompressRatio; ++r) {
            ex[r] = expf(fadd_rn(sc[r], -mx));
            if (r < a.ratio) sum = r == 0 ? ex[r] : fadd_rn(sum, ex[r]);
        }
        lat = fmul_rn(kv[0], fdiv_rn(ex[0], sum));                   // softmax over the group, then sum over the group in order (np.sum axis 0)
        DS41_UNROLL
        for (int r = 1; r < kAttnMaxCompressRatio; ++r)
            if (r < a.ratio) lat = fadd_rn(lat, fmul_rn(kv[r], fdiv_rn(ex[r], sum)));
    } else {
        lat = a.kv_c[e];
    }
    const float den = rms_den(block_sum<D>(fmul_rn(lat, lat), s_red), D, a.eps);
    float v = fmul_rn(a.norm_w[e], fdiv_rn(lat, den));
    a.latent_out[e] = v;                                             // PRE-RoPE, normalised: the indexer's input
    const float partner = shfl_xor(v, 1);
    if (e >= D - R) v = rope_elem<false>(v, partner, e - (D - R), a.cos_row, a.sin_row);
    if (a.fp4) v = fq_fp4_e4m3_16(v);
    a.comp_row[e] = v;
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(G::kIdxDim) void index_k_finish_kernel(const float* DS41_RESTRICT ik_raw, const float* DS41_RESTRICT nw, float eps,
                                                                      const float* DS41_RESTRICT cosr, const float* DS41_RESTRICT sinr, int fp4,
                                                                      float* DS41_RESTRICT out_row) {
    constexpr int D = G::kIdxDim, R = G::kRopeDim;
    static_assert(D % 32 == 0 && D <= 1024 && R <= D);
    DS41_SHARED float s_red[D / 32];
    const int e = threadIdx.x;
    const float raw = ik_raw[e];
    const float den = rms_den(block_sum<D>(fmul_rn(raw, raw), s_red), D, eps);
    float v = fmul_rn(nw[e], fdiv_rn(raw, den));
    const float partner = shfl_xor(v, 1);
    if (e >= D - R) v = rope_elem<false>(v, partner, e - (D - R), cosr, sinr);
    if (fp4) v = fq_fp4_e8m0_32(v);
    out_row[e] = v;
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(G::kIdxDim) void index_q_finish_kernel(float* DS41_RESTRICT iq, const float* DS41_RESTRICT cosr,
                                                                      const float* DS41_RESTRICT sinr, int fp4) {
    constexpr int D = G::kIdxDim, R = G::kRopeDim;
    const int e = threadIdx.x;
    float* row = iq + (size_t) blockIdx.x * D;
    float v = row[e];
    const float partner = shfl_xor(v, 1);
    if (e >= D - R) v = rope_elem<false>(v, partner, e - (D - R), cosr, sinr);
    if (fp4) v = fq_fp4_e8m0_32(v);
    row[e] = v;
}

constexpr int kScoreThreads = 128;

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kScoreThreads) void index_scores_kernel(const float* DS41_RESTRICT iq, const float* DS41_RESTRICT wproj, float wscale,
                                                                       const float* DS41_RESTRICT index_k, int n, const uint8_t* DS41_RESTRICT cand,
                                                                       int cand_block, float* DS41_RESTRICT score) {
    constexpr int H = G::kIdxHeads, D = G::kIdxDim;
    static_assert(D % 4 == 0 && H * D * 4 <= 32 * 1024, "iq lives in shared memory");
    DS41_SHARED float s_iq[H * D] ATTN_ALIGN16;
    DS41_SHARED float s_w[H];
    for (int i = threadIdx.x; i < H * D; i += kScoreThreads) s_iq[i] = iq[i];
    if (threadIdx.x < H) s_w[threadIdx.x] = fmul_rn(wproj[threadIdx.x], wscale);
    sync_block();
    const int t = blockIdx.x * kScoreThreads + threadIdx.x;
    if (t >= n) return;
    float acc[H];
    DS41_UNROLL
    for (int h = 0; h < H; ++h) acc[h] = 0.0f;
    const float* krow = index_k + (size_t) t * D;
    DS41_UNROLL1
    for (int d4 = 0; d4 < D / 4; ++d4) {
        const float4 kk = ldgf4(krow + 4 * d4);
        DS41_UNROLL
        for (int h = 0; h < H; ++h) {
            const float4 qq = *reinterpret_cast<const float4*>(&s_iq[h * D + 4 * d4]);
            float a = acc[h];
            a = fmaf(qq.x, kk.x, a);
            a = fmaf(qq.y, kk.y, a);
            a = fmaf(qq.z, kk.z, a);
            a = fmaf(qq.w, kk.w, a);
            acc[h] = a;
        }
    }
    float sc = fmul_rn(fmaxf(acc[0], 0.0f), s_w[0]);
    DS41_UNROLL
    for (int h = 1; h < H; ++h) sc = fadd_rn(sc, fmul_rn(fmaxf(acc[h], 0.0f), s_w[h]));
    if (cand != nullptr && cand[t / cand_block] == 0) sc = -INFINITY;
    score[t] = sc;
}

DS41_KERNEL DS41_LAUNCH_BOUNDS(kScoreThreads) void block_scores_kernel(const float* DS41_RESTRICT score, int n, int block, float* DS41_RESTRICT bs) {
    const int nb = (n + block - 1) / block;
    const int b = blockIdx.x * kScoreThreads + threadIdx.x;
    if (b >= nb) return;
    float m = -INFINITY;
    const int lo = b * block, hi = lo + block < n ? lo + block : n;
    for (int i = lo; i < hi; ++i) m = score[i] > m ? score[i] : m;
    if (b == (n - 1) / block) m = INFINITY;                          // the block of the newest reachable position is always kept
    bs[b] = m;
}

/// An order-preserving 32-bit key of a float (ascending key = ascending value), -0 == +0; `elig` = a finite number or +inf (not NaN, not -inf).
DS41_FI uint32_t score_key(float x, bool& elig) {
    const uint32_t raw = f2u(x);
    elig = (x == x) && raw != 0xFF800000u;
    const uint32_t u = x == 0.0f ? 0u : raw;
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

template <int NT>
DS41_KERNEL DS41_LAUNCH_BOUNDS(NT) void topk_select_kernel(const float* DS41_RESTRICT scores, int n, int k, int32_t* DS41_RESTRICT out_idx, int out_len,
                                                           uint8_t* DS41_RESTRICT out_flags) {
    DS41_SHARED int s_hist[256];
    DS41_SHARED int s_cnt0[NT / 32];
    DS41_SHARED int s_cnt1[NT / 32];
    DS41_SHARED int s_ctl[2];                  // 0: the number to select, 1: how many of the entries equal to the threshold are still needed
    DS41_SHARED uint32_t s_prefix[1];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    // ---- the threshold: the k-th largest key, found a byte at a time from the top (4 histogram passes) ----
    uint32_t prefix = 0, pmask = 0;
    int kk = 0, rem = 0;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int b = tid; b < 256; b += NT) s_hist[b] = 0;
        sync_block();
        for (int i = tid; i < n; i += NT) {
            bool el;
            const uint32_t key = score_key(scores[i], el);
            if (el && (key & pmask) == prefix) smem_add(&s_hist[(key >> shift) & 255u], 1);
        }
        sync_block();
        if (tid == 0) {
            if (pass == 0) {
                int tot = 0;
                for (int b = 0; b < 256; ++b) tot += s_hist[b];
                s_ctl[0] = tot < k ? tot : k;
                s_ctl[1] = s_ctl[0];
            }
            const int r = s_ctl[1];
            uint32_t digit = 0;
            if (r > 0) {
                int cum = 0;
                for (int b = 255; b >= 0; --b) {
                    if (cum + s_hist[b] >= r) {
                        digit = (uint32_t) b;
                        break;
                    }
                    cum += s_hist[b];
                }
                s_ctl[1] = r - cum;
            }
            s_prefix[0] = prefix | (digit << shift);
        }
        sync_block();
        kk = s_ctl[0];
        rem = s_ctl[1];
        prefix = s_prefix[0];
        pmask |= 0xFFu << shift;
        sync_block();
        if (kk == 0) break;                    // uniform: every thread read the same shared values
    }

    // ---- one ordered pass: everything above the threshold, then the first `rem` entries equal to it (ties -> the lower index) ----
    int eq_run = 0, sel_run = 0;
    const uint32_t lane_lt = (1u << lane) - 1u;
    for (int base = 0; base < n; base += NT) {
        const int i = base + tid;
        bool gt = false, eq = false;
        if (kk > 0 && i < n) {
            bool el;
            const uint32_t key = score_key(scores[i], el);
            if (el) {
                gt = key > prefix;
                eq = key == prefix;
            }
        }
        const uint32_t m_eq = ballot(eq);
        if (lane == 0) s_cnt0[warp] = popc(m_eq);
        sync_block();
        int eq_before = 0, eq_total = 0;
        for (int w = 0; w < NT / 32; ++w) {
            const int c = s_cnt0[w];
            eq_before += w < warp ? c : 0;
            eq_total += c;
        }
        const bool sel = gt || (eq && eq_run + eq_before + popc(m_eq & lane_lt) < rem);
        const uint32_t m_sel = ballot(sel);
        if (lane == 0) s_cnt1[warp] = popc(m_sel);
        sync_block();
        int sel_before = 0, sel_total = 0;
        for (int w = 0; w < NT / 32; ++w) {
            const int c = s_cnt1[w];
            sel_before += w < warp ? c : 0;
            sel_total += c;
        }
        if (sel && out_idx != nullptr) out_idx[sel_run + sel_before + popc(m_sel & lane_lt)] = i;
        if (out_flags != nullptr && i < n) out_flags[i] = sel ? 1 : 0;
        eq_run += eq_total;
        sel_run += sel_total;
        sync_block();
    }
    if (out_idx != nullptr)
        for (int j = sel_run + tid; j < out_len; j += NT) out_idx[j] = -1;
}

constexpr int kSparseThreads = 256;

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(kSparseThreads) void sparse_attn_kernel(const float* DS41_RESTRICT q, const float* DS41_RESTRICT win, int pos,
                                                                       const float* DS41_RESTRICT comp, const int32_t* DS41_RESTRICT topk,
                                                                       const float* DS41_RESTRICT sink, const float* DS41_RESTRICT cosr,
                                                                       const float* DS41_RESTRICT sinr, float* DS41_RESTRICT o_rot, float scale) {
    constexpr int D = G::kHeadDim, R = G::kRopeDim, W = G::kWindow, K = G::kIdxTopK, NK = W + K;
    constexpr int HPB = AttnKernels<G>::kHeadsPerBlock, NT = kSparseThreads, NWARP = NT / 32, DPT = (D + NT - 1) / NT;
    static_assert(HPB <= NWARP && G::kHeads % HPB == 0 && D % 4 == 0);
    static_assert(sizeof(float) * (HPB * D + HPB * NK + HPB) + sizeof(int) * NK <= 48 * 1024, "static shared memory");
    DS41_SHARED float s_q[HPB * D] ATTN_ALIGN16;
    DS41_SHARED float s_p[HPB * NK];            // logits, then probabilities
    DS41_SHARED int s_row[NK];                  // the row of key j in the ring (j < W) / in the compressed cache (j >= W); -1 = no such key
    DS41_SHARED float s_den[HPB];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int h0 = blockIdx.x * HPB;

    for (int i = tid; i < HPB * D; i += NT) s_q[i] = q[(size_t) h0 * D + i];
    for (int j = tid; j < NK; j += NT) {
        int r = -1;
        if (j < W) {
            const int slot = (pos % W + 1 + j) % W;                  // the oracle's window_topk_idxs order: the oldest slot first, the newest last
            r = slot <= pos ? slot : -1;                              // slots that hold nothing yet (pos < W - 1)
        } else if (topk != nullptr) {
            r = topk[j - W];
        }
        s_row[j] = r;
    }
    sync_block();

    // ---- logits: a warp per key, the 32 lanes split the head dimension, all HPB heads of the block at once ----
    for (int j = warp; j < NK; j += NWARP) {
        const int r = s_row[j];
        if (r < 0) {
            if (lane == 0)
                for (int hh = 0; hh < HPB; ++hh) s_p[hh * NK + j] = -INFINITY;
            continue;
        }
        const float* row = (j < W ? win : comp) + (size_t) r * D;
        float part[HPB];
        DS41_UNROLL
        for (int hh = 0; hh < HPB; ++hh) part[hh] = 0.0f;
        for (int c4 = lane; c4 < D / 4; c4 += 32) {
            const float4 kv = ldgf4(row + 4 * c4);
            DS41_UNROLL
            for (int hh = 0; hh < HPB; ++hh) {
                const float4 qq = *reinterpret_cast<const float4*>(&s_q[hh * D + 4 * c4]);
                float a = part[hh];
                a = fmaf(qq.x, kv.x, a);
                a = fmaf(qq.y, kv.y, a);
                a = fmaf(qq.z, kv.z, a);
                a = fmaf(qq.w, kv.w, a);
                part[hh] = a;
            }
        }
        DS41_UNROLL
        for (int hh = 0; hh < HPB; ++hh) {
            float v = part[hh];
            DS41_UNROLL
            for (int off = 16; off > 0; off >>= 1) v = fadd_rn(v, shfl_xor(v, off));
            if (lane == 0) s_p[hh * NK + j] = fmul_rn(v, scale);
        }
    }
    sync_block();

    // ---- softmax with the sink, a warp per head: m = max(max logit, sink); p = exp(l - m); denominator = sum p + exp(sink - m) ----
    if (warp < HPB) {
        float* pl = s_p + warp * NK;
        const float sk = sink[h0 + warp];
        float mx = -INFINITY;
        for (int j = lane; j < NK; j += 32) mx = pl[j] > mx ? pl[j] : mx;
        DS41_UNROLL
        for (int off = 16; off > 0; off >>= 1) {
            const float o = shfl_xor(mx, off);
            mx = o > mx ? o : mx;
        }
        const float m = sk > mx ? sk : mx;
        float sum = 0.0f;
        for (int j = lane; j < NK; j += 32) {
            const float l = pl[j];
            const float p = l == -INFINITY ? 0.0f : expf(fadd_rn(l, -m));
            pl[j] = p;
            sum = fadd_rn(sum, p);
        }
        DS41_UNROLL
        for (int off = 16; off > 0; off >>= 1) sum = fadd_rn(sum, shfl_xor(sum, off));
        if (lane == 0) s_den[warp] = fadd_rn(sum, expf(fadd_rn(sk, -m)));
    }
    sync_block();

    // ---- o[h][d] = sum_j p[h][j] kv[j][d] / denom, a thread per d (and d + 256 ...), the keys in index order ----
    float acc[HPB][DPT];
    DS41_UNROLL
    for (int hh = 0; hh < HPB; ++hh)
        DS41_UNROLL
        for (int i = 0; i < DPT; ++i) acc[hh][i] = 0.0f;
    DS41_UNROLL1
    for (int j = 0; j < NK; ++j) {
        const int r = s_row[j];
        if (r < 0) continue;                                          // block-uniform
        const float* row = (j < W ? win : comp) + (size_t) r * D;
        DS41_UNROLL
        for (int i = 0; i < DPT; ++i) {
            const int d = tid + NT * i;
            const float x = d < D ? row[d] : 0.0f;
            DS41_UNROLL
            for (int hh = 0; hh < HPB; ++hh) acc[hh][i] = fmaf(s_p[hh * NK + j], x, acc[hh][i]);
        }
    }
    DS41_UNROLL
    for (int hh = 0; hh < HPB; ++hh) {
        DS41_UNROLL
        for (int i = 0; i < DPT; ++i) {
            const int d = tid + NT * i;
            float o = fdiv_rn(acc[hh][i], s_den[hh]);
            const float partner = shfl_xor(o, 1);
            if (d < D) {
                if (d >= D - R) o = rope_elem<true>(o, partner, d - (D - R), cosr, sinr);       // the inverse RoPE on the tail
                o_rot[(size_t) (h0 + hh) * D + d] = o;
            }
        }
    }
}

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(32) void fakequant_kernel(int kind, const float* DS41_RESTRICT x, int n, float* DS41_RESTRICT y) {
    const int lane = threadIdx.x;
    for (int base = 0; base < n; base += 32) {
        const float v = x[base + lane];
        y[base + lane] = kind == 0 ? fq_fp8_32(v) : (kind == 1 ? fq_fp4_e8m0_32(v) : fq_fp4_e4m3_16(v));
    }
}

}  // namespace strata::ds41::cuda::adev

// ---------------------------------------------------------------------------------------------------------------------------------
// host wrappers
// ---------------------------------------------------------------------------------------------------------------------------------
namespace strata::ds41::cuda {

template <class G>
void AttnKernels<G>::rmsnorm_row(const float* x, const float* w, int n, float eps, float* y, Stream s) {
    dev::launch(adev::rmsnorm_row_kernel<256>, dim3(1), dim3(256), 0, s, x, w, n, eps, y);
    dev::check_launch("attn rmsnorm_row");
}

template <class G>
void AttnKernels<G>::q_rope(float* q, const float* cos_row, const float* sin_row, Stream s) {
    dev::launch(adev::q_rope_kernel<G>, dim3(G::kHeads), dim3(adev::round32(G::kRopeDim)), 0, s, q, cos_row, sin_row);
    dev::check_launch("attn q_rope");
}

template <class G>
void AttnKernels<G>::swa_kv(const float* kv_raw, const float* norm_w, float eps, const float* cos_row, const float* sin_row, bool fp8, float* ring_row,
                            Stream s) {
    dev::launch(adev::swa_kv_kernel<G>, dim3(1), dim3(G::kHeadDim), 0, s, kv_raw, norm_w, eps, cos_row, sin_row, fp8 ? 1 : 0, ring_row);
    dev::check_launch("attn swa_kv");
}

template <class G>
void AttnKernels<G>::compress_step(const AttnCompressArgs& a, Stream s) {
    if (a.ratio < 1 || a.ratio > kAttnMaxCompressRatio) throw std::invalid_argument("attn compress_step: ratio must be 1.." + std::to_string(kAttnMaxCompressRatio));
    dev::launch(adev::compress_step_kernel<G>, dim3(1), dim3(G::kHeadDim), 0, s, a);
    dev::check_launch("attn compress_step");
}

template <class G>
void AttnKernels<G>::index_k_finish(const float* ik_raw, const float* norm_w, float eps, const float* cos_row, const float* sin_row, bool fp4,
                                    float* out_row, Stream s) {
    dev::launch(adev::index_k_finish_kernel<G>, dim3(1), dim3(G::kIdxDim), 0, s, ik_raw, norm_w, eps, cos_row, sin_row, fp4 ? 1 : 0, out_row);
    dev::check_launch("attn index_k_finish");
}

template <class G>
void AttnKernels<G>::index_q_finish(float* iq, const float* cos_row, const float* sin_row, bool fp4, Stream s) {
    dev::launch(adev::index_q_finish_kernel<G>, dim3(G::kIdxHeads), dim3(G::kIdxDim), 0, s, iq, cos_row, sin_row, fp4 ? 1 : 0);
    dev::check_launch("attn index_q_finish");
}

template <class G>
void AttnKernels<G>::index_scores(const float* iq, const float* wproj, float wscale, const float* index_k, int n, const uint8_t* cand_flags,
                                  int cand_block, float* score, Stream s) {
    if (n <= 0) return;
    dev::launch(adev::index_scores_kernel<G>, dim3((unsigned) ((n + adev::kScoreThreads - 1) / adev::kScoreThreads)), dim3(adev::kScoreThreads), 0, s, iq,
                wproj, wscale, index_k, n, cand_flags, cand_block, score);
    dev::check_launch("attn index_scores");
}

template <class G>
void AttnKernels<G>::block_scores(const float* score, int n, int block, float* bs, Stream s) {
    if (n <= 0) return;
    const int nb = (n + block - 1) / block;
    dev::launch(adev::block_scores_kernel, dim3((unsigned) ((nb + adev::kScoreThreads - 1) / adev::kScoreThreads)), dim3(adev::kScoreThreads), 0, s, score, n,
                block, bs);
    dev::check_launch("attn block_scores");
}

template <class G>
void AttnKernels<G>::topk_select(const float* scores, int n, int k, int32_t* out_idx, int out_len, uint8_t* out_flags, Stream s) {
    dev::launch(adev::topk_select_kernel<kAttnTopkThreads>, dim3(1), dim3(kAttnTopkThreads), 0, s, scores, n < 0 ? 0 : n, k, out_idx, out_len, out_flags);
    dev::check_launch("attn topk_select");
}

template <class G>
void AttnKernels<G>::sparse_attn(const float* q, const float* win_kv, int pos, const float* comp_kv, const int32_t* topk_row, const float* sink,
                                 const float* cos_row, const float* sin_row, float* o_rot, Stream s) {
    const float scale = (float) (1.0 / std::sqrt((double) G::kHeadDim));          // cfg.softmax_scale = head_dim ** -0.5 (a python double, used as an fp32 scalar)
    dev::launch(adev::sparse_attn_kernel<G>, dim3(G::kHeads / kHeadsPerBlock), dim3(adev::kSparseThreads), 0, s, q, win_kv, pos, comp_kv, topk_row, sink,
                cos_row, sin_row, o_rot, scale);
    dev::check_launch("attn sparse_attn");
}

template <class G>
void AttnKernels<G>::fakequant(int kind, const float* x, int n, float* y, Stream s) {
    dev::launch(adev::fakequant_kernel<G>, dim3(1), dim3(32), 0, s, kind, x, n, y);
    dev::check_launch("attn fakequant");
}

}  // namespace strata::ds41::cuda
