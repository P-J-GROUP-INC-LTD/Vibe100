// src/ds41/cuda/ds41_experts_impl.cuh - DS-D / DS1-G: the MXFP4 hit-expert kernels (V100, sm_70), templates over the geometry G (geom.hpp).
// Included by ds41_experts.cu (nvcc: RealGeom) and by ds41_emu_impl.cpp (-DDS41_EMU: RealGeom and MiniGeom).  The numbers in the comments are RealGeom's
// (hidden 5120 = 160 blocks, ff 2304 = 72 blocks); every one of them is derived from G in ExCfg below, see "SHAPES AT OTHER G" there.
//
// =====================================================================================================================
// WHAT THE THREE KERNELS DO (docs/deepseek/CONTRACTS.md has the math)
// =====================================================================================================================
//   quantize_acts_kernel   x fp32 -> int8 per 32 + fp32 scale (the CPU's rule, bit for bit), in the dp4a-friendly byte order.
//   gate_up_kernel         per GROUP of hits sharing a cache slot (= one expert, 1..NT tokens): for a tile of 32 intermediate
//                          rows, stream the rows of W1 and W3 ONCE, dot them with every token of the group, then
//                          h = silu(min(g,10)) * clamp(u,-10,10) * w (NaN propagates through the clamps) and quantise h per 32 rows (exactly one tile) to int8 + scale.
//   down_kernel            per group: for a tile of 64 output rows stream W2's rows once and dot them with every token's h:
//                          parts[(token*6+k)][row] = W2.h.
//   One launch per phase covers EVERY hit of the layer: grid.y = 6*T group slots, the blocks past counts->n_groups return
//   at once, so the grid is sized from T alone and the host never learns how many hits there were.
//
// =====================================================================================================================
// HOW A 17-BYTE BLOCK IS READ FAST
// =====================================================================================================================
//   A row is 160 (W1/W3) or 72 (W2) blocks of 17 bytes = 2720 / 1224 bytes: blocks are not 16-byte aligned (block b starts at
//   17 b), so a lane cannot load one with a vector load.  Instead a warp stages the bytes in shared memory with COALESCED
//   16-byte (W1/W3: rows are 16-byte aligned, 2720 = 170 x 16) or 8-byte (W2: 1224 = 153 x 8) loads, and each lane then picks
//   out its own blocks with 32-bit shared loads and one prmt per word:
//     * lane (rr, kk) = (lane >> 4, lane & 15) handles row `rr` of a PAIR of rows and the blocks b = kk, kk + 16, kk + 32, ...
//       of it.  Because 16 is a multiple of 4, 17 b mod 4 = kk mod 4 is the SAME for every block of a lane: the byte alignment
//       of the lane's blocks is a lane constant, so the "realign by a" selector is computed once.
//     * block bytes [17 b, 17 b + 17) lie in 32-bit words w[17b/4 .. 17b/4 + 4]; e = byte a of w0; qs word k = bytes
//       (a + 1 + 4k ..) of {w[k+1] : w[k]} = prmt(w[k], w[k+1], 0x3210 + 0x1111 (a + 1)).  5 LDS.32 + 5 prmt per block.
//     * the 16 lanes of a row read 16 different banks (the word index 17kk/4 is distinct mod 32 for kk = 0..15), the second row
//       sits 16 words (mod 32) further on (padded segment strides), so every one of those loads is bank-conflict free.
//   The staged "stage" is 2 rows x half a row of W1/W3 (2720 B) or 2 rows of W2 (2448 B) per warp, filled from registers that were
//   loaded one stage AHEAD (software pipeline: the loads for stage s+1 are in flight while stage s is multiplied).
//
//   The weights are decoded ONCE per block into int8x4 words (11 instructions per 8 weights, see lut8 in ds41_math.cuh) and
//   used for every token of the group: per token 8 dp4a + 2 LDS.128 (the token's int8 block) + 1 LDS.32 (its scale).  The
//   activations live in shared memory for the whole block, split in two 16-byte halves so that the 16 lanes read consecutive
//   16-byte slots (conflict free), and the two row-groups of the warp read the same slot (a broadcast).
//
//   The block scale is applied in FP32 per (block, token): acc += (2^(e-128) * d_x) * (float) isum, so the E8M0 range is
//   exact and there is no fp16 anywhere.
//
// =====================================================================================================================
// BYTES AND TIME (per hit; 1 hit = 1 expert for 1..8 tokens)
// =====================================================================================================================
//   W1 + W3 = 2 * 6,266,880 B = 12,533,760 B;  W2 = 6,266,880 B;  total 18,800,640 B per hit = kBlobBytes.  The activations
//   (5,760 B per token for x, 2,592 B for h) come from L2: < 4 % of the weight bytes per block at T = 1, ~13 % at 4 tokens per group.
//   At the V100's 900 GB/s: 20.9 us per hit  (gate/up 13.9 us, down 7.0 us); streaming reads sustain ~800-830 GB/s on this card:
//   ~23 us per hit.   6 hits (T = 1) = 112.8 MB = 125 us (900 GB/s) .. 141 us (800 GB/s);  24 hits (T = 4) = 451 MB = 501 us .. 564 us;
//   48 hits (T = 8, all experts distinct) = 902 MB = 1.00 .. 1.13 ms.  Expect the kernels at 85-92 % of the bandwidth bound for T <= 4.
//   Integer-pipe budget (the V100 does 64 integer lanes/clk/SM = 7.07e12 lane-ops/s): decode + realign ~ 53 ops per 17-byte block and
//   lane, plus 8 dp4a per token: 61 (T = 1), 85 (T = 4), 117 (T = 8) per block -> 6.4, 8.9, 12.2 us per hit for gate/up (the bandwidth
//   time is 13.9 us): the integer pipe is ~45 %, ~65 % and ~88 % busy at T = 1, 4, 8 if memory runs flat out.  T = 8 with large groups is
//   therefore issue bound (~75 % of the bandwidth bound is what to expect); this is where a tensor-core path would pay, see the notes
//   in the report: the fixed GGUF block layout means every weight must be decoded to fp16 first, which costs about what the dp4a saves.
//   Occupancy (V100, 96 KB shared / 64 K registers per SM; numbers from ptxas and the formulas below):
//       specialisation   gate/up: regs  dyn smem  blocks/SM    down: regs  dyn smem  blocks/SM
//       NT = 1                    80    29,568 B   3 (24 warps)       74    23,648 B   3 (24 warps)
//       NT = 2                   127    35,456 B   2 (16)             96    26,496 B   2 (16)
//       NT = 4                   128    47,232 B   2 (16)             97    32,192 B   2 (16)
//       NT = 8                   128    70,784 B   1 (8)             106    43,584 B   2 (16)
//   (ds41_expert_parity prints the same table from the occupancy API, expert_kernel_info().)  Little's law needs ~10-15 KB of loads in
//   flight per SM at 900 GB/s; every warp keeps one 2.7 KB stage in flight, so even NT = 8 (8 warps = 22 KB) has enough.
//   WHICH SPECIALISATION RUNS WHAT: T = 1 NT = 1; T = 2 NT = 2; T = 3, 4 NT = 4; T = 5..8: NT = 4 for the groups of 1..4 tokens and
//   NT = 8 for the (at most floor(6T/5)) groups of 5..8 tokens, two launches over the same group list (split sorts it by decreasing
//   size); a launch's blocks whose group has a different size return at once.
//
// =====================================================================================================================
// ARITHMETIC (the contract; every step has a reason)
// =====================================================================================================================
//   isum = sum_j  kvalues_fp4[code_j] * q_j            exact int32 (|isum| <= 12 * 127 * 32)
//   acc  += (2^(e-128) * d_x) * (float) isum           per 32-block, FP32, in a fixed order -> run-to-run deterministic
//   h    = silu(min(g,10)) * clamp(u,-10,10) * w       (w = the routing weight, BEFORE the down projection)
//   hq   = quantise(h) per 32 rows (same rule as x)    so a hit and a miss (CPU) of the same expert agree to the FP32 sums
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "ds41_math.cuh"
#include "strata/ds41/cuda/ds41_cuda.hpp"

namespace strata::ds41::cuda::dev {

// ---------------------------------------------------------------------------------------------------------------------
// geometry shared by the two phases
// ---------------------------------------------------------------------------------------------------------------------
inline constexpr int kExThreads = 256;                    // 8 warps
inline constexpr int kExWarps = 8;
inline constexpr int kDnTileRows = 64;                    // rows per block (down)

/// Stage-buffer padding: the smallest word count >= `w` that is 16 (mod 32), so that the second row of a pair sits 16 banks away from the first (the
/// 16 lanes of a row read 16 different banks, the other 16 lanes of the warp read the other 16).  w = 340 -> 368, 306 -> 336.
constexpr int pad_bank16(int w) { return ((w - 16 + 31) / 32) * 32 + 16; }

/// SHAPES AT OTHER G.  Everything the kernels index with, derived from G.  RealGeom: hidden 5120 (160 blocks, rows of 2720 B = 170 x 16), ff 2304 (72 blocks,
/// down rows of 1224 B = 153 x 8).  A row of W2 must be a whole number of 8-byte pieces (kFF % 256 == 0); a pair of gate/up rows a whole number of 16-byte
/// chunks (kHidden % 256 == 0).  When the gate/up row has a multiple of 32 blocks (kHidden % 1024 == 0, RealGeom) a stage is HALF a row of two rows
/// (rows are 16-byte aligned, the halves too: 1360 B = 85 x 16) as always; otherwise (MiniGeom: 8 blocks = 136 B per row, not a multiple of 16) the whole
/// PAIR of rows is contiguous in global memory (2 x 136 = 17 chunks) and is staged as one run: the second row follows the first in shared memory with no
/// padding.  Same kernel, same arithmetic; only the staging constants differ (kGuSplits, kGuRowSplit, kGuGap, kGuSeg).
template <class G>
struct ExCfg {
    using D = ExpertDims<G>;
    static constexpr int kHidden = G::kHidden, kFF = G::kFF, kTopK = G::kTopK;
    static constexpr int kActBlocks = D::kActBlocks;          // 160 blocks of 32 in x
    static constexpr int kHBlocks = D::kHBlocks;              // 72 blocks of 32 in h
    static constexpr size_t kGateRowBytes = D::kGateRowBytes, kDownRowBytes = D::kDownRowBytes;
    static constexpr size_t kBlobGate = D::kBlobGate, kBlobUp = D::kBlobUp, kBlobDown = D::kBlobDown, kBlobBytes = D::kBlobBytes;
    static constexpr int kGuTiles = kFF / 32;                 // 72 tiles of 32 rows per group (gate/up)
    static constexpr int kDnTiles = kHidden / kDnTileRows;    // 80 tiles per group (down)

    // ---- gate/up: a stage = 2 rows x (half a row | a whole row), the two rows of a pair
    static constexpr bool kGuHalves = kActBlocks % 32 == 0;
    static constexpr int kGuSplits = kGuHalves ? 2 : 1;       // stages per row
    static constexpr int kGuSplitLog2 = kGuHalves ? 1 : 0;
    static constexpr int kGuSegBlocks = kActBlocks / kGuSplits;                        // 80 blocks of a row per stage
    static constexpr int kGuSegBytes = kGuSegBlocks * kBlockBytes;                     // 1360
    static constexpr int kGuChunks = 2 * kGuSegBytes / 16;                             // 170 chunks of 16 B per stage (both rows)
    static constexpr int kGuRowSplit = kGuHalves ? kGuSegBytes / 16 : kGuChunks;       // 85: the first chunk of row 1 (never reached when contiguous)
    static constexpr int kGuGap = kGuHalves ? (int) kGateRowBytes - kGuSegBytes : 0;   // 1360: global bytes between row 0's segment and row 1's
    static constexpr int kGuPF = (kGuChunks + 31) / 32;                                // 6 registers of 16 B per lane
    static constexpr int kGuSeg = kGuHalves ? 4 * pad_bank16(kGuSegBytes / 4) : kGuSegBytes;   // 1472: row 1's segment in shared memory
    static constexpr int kGuStageBytes = 2 * kGuSeg + 16;

    // ---- down: a stage = 2 whole rows (contiguous in global memory), 8-byte pieces
    static constexpr int kDnPieces = (int) (kDownRowBytes / 8);                        // 153 pieces of 8 B per row
    static constexpr int kDnTotal = 2 * kDnPieces;                                     // 306
    static constexpr int kDnPF = (kDnTotal + 31) / 32;                                 // 10 registers of 8 B per lane
    static constexpr int kDnSeg = 4 * pad_bank16((int) (kDownRowBytes / 4));           // 1344
    static constexpr int kDnStageBytes = kDnSeg + (int) kDownRowBytes + 32;

    static constexpr int kXBytesPerTok = kActBlocks * 36;     // 160 x (16 + 16 + 4) = 5760 smem bytes of activations per token (x)
    static constexpr int kHBytesPerTok = kHBlocks * 36;       // 72 x 36 = 2592 (h)

    static_assert(kFF % 32 == 0 && kHidden % kDnTileRows == 0, "experts: kFF % 32 == 0 (an h block is a tile of 32 rows), kHidden % 64 == 0 (the down tile)");
    static_assert(kHidden % 256 == 0, "experts: a pair of gate/up rows is a whole number of 16-byte chunks (kHidden % 256 == 0)");
    static_assert(kFF % 256 == 0, "experts: a row of W2 is a whole number of 8-byte pieces (kFF % 256 == 0)");
    static_assert(kActBlocks % 4 == 0 && kHBlocks % 4 == 0, "experts: the block scales are copied as float4");
    static_assert(!kGuHalves || kGuSegBytes % 16 == 0, "experts: half rows are 16-byte aligned");
    static_assert(kGateRowBytes % 16 == 0 || !kGuHalves, "experts: gate/up rows are 16-byte aligned when they are staged in halves");
    static_assert((2 * kGateRowBytes) % 16 == 0, "experts: a pair of gate/up rows starts on a 16-byte boundary");
    static_assert(kGuStageBytes % 16 == 0, "experts: the gate/up stage buffers are 16-byte aligned (ST.E.128)");
    static_assert(kDnStageBytes % 8 == 0, "experts: the down stage buffers are 8-byte aligned (ST.E.64)");
    static_assert(kBlobBytes % 256 == 0 && kBlobUp % 16 == 0 && kBlobDown % 16 == 0, "experts: slots and matrices keep their alignment");
};

template <class G, int NT>
struct GuLayout {
    using C = ExCfg<G>;
    static constexpr int lo = 0;                                  // uint4 [NT][160]
    static constexpr int hi = lo + NT * C::kActBlocks * 16;       // uint4 [NT][160]
    static constexpr int sc = hi + NT * C::kActBlocks * 16;       // float [NT][160]
    static constexpr int h = sc + NT * C::kActBlocks * 4;         // float [NT][32]
    static constexpr int stage = round16(h + NT * 32 * 4);        // 8 warps x kGuStageBytes
    static constexpr int total = stage + kExWarps * C::kGuStageBytes;
};
template <class G, int NT>
struct DnLayout {
    using C = ExCfg<G>;
    static constexpr int lo = 0;                                  // uint4 [NT][72]
    static constexpr int hi = lo + NT * C::kHBlocks * 16;
    static constexpr int sc = hi + NT * C::kHBlocks * 16;         // float [NT][72]
    static constexpr int y = sc + NT * C::kHBlocks * 4;           // float [NT][64]
    static constexpr int stage = round16(y + NT * kDnTileRows * 4);
    static constexpr int total = stage + kExWarps * C::kDnStageBytes;
};

// ---------------------------------------------------------------------------------------------------------------------
// stage movement: global -> registers (one stage ahead) -> shared memory
// ---------------------------------------------------------------------------------------------------------------------
/// W1/W3: `base` = first byte of the stage's first row (+ half * 1360).  Chunk u of 16 B (0..169): row (u >= 85), offset.
template <class G>
DS41_FI void gu_load(uint4 (&pf)[ExCfg<G>::kGuPF], const uint8_t* base, int lane) {
    using C = ExCfg<G>;
    DS41_UNROLL
    for (int j = 0; j < C::kGuPF; ++j) {
        const int u = lane + 32 * j;
        if (u < C::kGuChunks) pf[j] = ldg4(base + 16 * u + (u >= C::kGuRowSplit ? C::kGuGap : 0));       // row 1 starts 2720 B on: 16u + 1360 = 2720 + 16 (u - 85)
    }
}
template <class G>
DS41_FI void gu_store(const uint4 (&pf)[ExCfg<G>::kGuPF], unsigned char* stage, int lane) {
    using C = ExCfg<G>;
    DS41_UNROLL
    for (int j = 0; j < C::kGuPF; ++j) {
        const int u = lane + 32 * j;
        if (u < C::kGuChunks) {
            DS41_ASSERT_ALIGNED(stage + 16 * u + (u >= C::kGuRowSplit ? C::kGuSeg - C::kGuSegBytes : 0), 16);          // ST.E.128 to shared memory: misaligned = a fault
            *reinterpret_cast<uint4*>(stage + 16 * u + (u >= C::kGuRowSplit ? C::kGuSeg - C::kGuSegBytes : 0)) = pf[j];
        }
    }
}
/// W2: `base` = first byte of the stage's first row; the two rows are contiguous (2448 B = 306 pieces of 8 B).
template <class G>
DS41_FI void dn_load(uint2 (&pf)[ExCfg<G>::kDnPF], const uint8_t* base, int lane) {
    using C = ExCfg<G>;
    DS41_UNROLL
    for (int j = 0; j < C::kDnPF; ++j) {
        const int p = lane + 32 * j;
        if (p < C::kDnTotal) pf[j] = ldg2(base + 8 * p);
    }
}
template <class G>
DS41_FI void dn_store(const uint2 (&pf)[ExCfg<G>::kDnPF], unsigned char* stage, int lane) {
    using C = ExCfg<G>;
    DS41_UNROLL
    for (int j = 0; j < C::kDnPF; ++j) {
        const int p = lane + 32 * j;
        if (p < C::kDnTotal) {
            DS41_ASSERT_ALIGNED(stage + 8 * p + (p >= C::kDnPieces ? C::kDnSeg - (int) C::kDownRowBytes : 0), 8);           // ST.E.64 to shared memory
            *reinterpret_cast<uint2*>(stage + 8 * p + (p >= C::kDnPieces ? C::kDnSeg - (int) C::kDownRowBytes : 0)) = pf[j];
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// the inner loop: one lane, one row segment in shared memory, NB blocks, all NT tokens
// ---------------------------------------------------------------------------------------------------------------------
/// seg: this lane's ROW segment in shared memory (4-byte aligned).  The lane takes blocks kk, kk+16, ... (< NB); block bl of
/// the segment is activation block (blk0 + bl) of every token.  Activations: lo/hi (uint4) and sc (float) arrays with
/// `STRIDE` blocks per token.  Tokens t >= n are skipped (n is uniform across the block).
template <int NT, int NB, int STRIDE>
DS41_FI void stage_dot(const unsigned char* seg, int kk, int blk0, const uint4* act_lo, const uint4* act_hi, const float* act_sc,
                       int n, float (&acc)[NT]) {
    const int a = kk & 3;                                  // byte alignment of every block of this lane (17 b mod 4)
    const uint32_t sel_e = 0x4440u + (uint32_t) a;         // byte a of w0 -> byte 0, zero elsewhere
    const uint32_t sel_q = 0x3210u + 0x1111u * (uint32_t) (a + 1);   // bytes a+1 .. a+4 of {w[k+1] : w[k]}
    const uint32_t* wbase = reinterpret_cast<const uint32_t*>(seg);
    DS41_UNROLL
    for (int m = 0; m < (NB + 15) / 16; ++m) {
        const int bl = kk + 16 * m;
        if (NB % 16 == 0 || bl < NB) {                     // only the 72-block rows of W2 have a ragged last step (and rows of fewer than 16 blocks)
            const uint32_t* wp = wbase + ((17 * bl) >> 2);
            const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3], w4 = wp[4];
            const float d = e8m0_half(prmt(w0, 0u, sel_e));
            int A0, B0, A1, B1, A2, B2, A3, B3;
            lut8(prmt(w0, w1, sel_q), A0, B0);
            lut8(prmt(w1, w2, sel_q), A1, B1);
            lut8(prmt(w2, w3, sel_q), A2, B2);
            lut8(prmt(w3, w4, sel_q), A3, B3);
            const int b = blk0 + bl;
            DS41_UNROLL
            for (int t = 0; t < NT; ++t) {
                if (t == 0 || t < n) {                      // token 0 always exists (n >= 1): no test, no branch
                    DS41_ASSERT_ALIGNED(act_lo + t * STRIDE + b, 16);                  // LDS.128
                    DS41_ASSERT_ALIGNED(act_hi + t * STRIDE + b, 16);
                    const uint4 x0 = act_lo[t * STRIDE + b];
                    const uint4 x1 = act_hi[t * STRIDE + b];
                    const float dx = act_sc[t * STRIDE + b];
                    int s0 = dp4a(A0, (int) x0.x, 0);
                    int s1 = dp4a(A2, (int) x1.x, 0);
                    s0 = dp4a(B0, (int) x0.y, s0);
                    s1 = dp4a(B2, (int) x1.y, s1);
                    s0 = dp4a(A1, (int) x0.z, s0);
                    s1 = dp4a(A3, (int) x1.z, s1);
                    s0 = dp4a(B1, (int) x0.w, s0);
                    s1 = dp4a(B3, (int) x1.w, s1);
                    acc[t] = fmaf(d * dx, (float) (s0 + s1), acc[t]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// x -> int8
// ---------------------------------------------------------------------------------------------------------------------
/// One warp per (token, 32-block).  NB > 0: the row has NB blocks at COMPILE time (the expert path: ExpertDims<G>::kActBlocks, the same code as ever);
/// NB == 0: `nb_rt` blocks per row at run time (the generic ds41_quantize_acts).  kInter: the interleaved (dp4a / MXFP4) byte order, else natural.
template <int NB, bool kInter>
DS41_KERNEL DS41_LAUNCH_BOUNDS(256) void quantize_acts_kernel(const float* DS41_RESTRICT x, int T, int8_t* DS41_RESTRICT xq,
                                                              float* DS41_RESTRICT xs, int nb_rt) {
    const int nb = NB > 0 ? NB : nb_rt;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int gw = blockIdx.x * kExWarps + warp;           // one warp per (token, 32-block)
    if (gw >= T * nb) return;
    const int t = gw / nb, b = gw - t * nb;
    const float v = ldgf(x + (size_t) t * (nb * 32) + b * 32 + lane);
    quantize_block_warp<kInter>(v, lane, xq + ((size_t) t * nb + b) * 32, xs + (size_t) t * nb + b);
}

DS41_KERNEL void e8m0_table_kernel(float* out) { out[threadIdx.x] = e8m0_half(threadIdx.x); }

// ---------------------------------------------------------------------------------------------------------------------
// phase 1: gate / up -> h (quantised)
// ---------------------------------------------------------------------------------------------------------------------
template <class G, int NT>
DS41_KERNEL DS41_LAUNCH_BOUNDS(256, (NT == 1 ? 3 : 2)) void gate_up_kernel(const uint8_t* DS41_RESTRICT cache, const HitEntry* DS41_RESTRICT hits,
                                                           const HitGroup* DS41_RESTRICT groups,
                                                           const SplitCounts* DS41_RESTRICT counts,
                                                           const int8_t* DS41_RESTRICT xq, const float* DS41_RESTRICT xs,
                                                           int8_t* DS41_RESTRICT hq, float* DS41_RESTRICT hs, int n_lo,
                                                           int n_hi) {
    using C = ExCfg<G>;
    using L = GuLayout<G, NT>;
    constexpr int kHidden = C::kHidden, kActBlocks = C::kActBlocks, kHBlocks = C::kHBlocks;
    const int g = blockIdx.y;
    if (g >= counts->n_groups) return;                     // uniform for the block
    const int tile = blockIdx.x;
    const HitGroup* gp = groups + g;
    const int n = gp->n;
    if (n < n_lo || n > n_hi) return;                      // another specialisation's group (uniform)
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int rr = lane >> 4, kk = lane & 15;

    DS41_DYN_SMEM(smem);
    uint4* act_lo = reinterpret_cast<uint4*>(smem + L::lo);
    uint4* act_hi = reinterpret_cast<uint4*>(smem + L::hi);
    float* act_sc = reinterpret_cast<float*>(smem + L::sc);
    float* s_h = reinterpret_cast<float*>(smem + L::h);
    unsigned char* stage = smem + L::stage + warp * C::kGuStageBytes;

    // this warp's 4 rows = two row PAIRS; the stage sequence is (pair p, half h) x (gate, up)
    const uint8_t* blob = cache + (size_t) gp->slot * C::kBlobBytes;
    const int row_base = tile * 32 + warp * 4;
    const uint8_t* gate_rows = blob + C::kBlobGate + (size_t) row_base * C::kGateRowBytes;
    const uint8_t* up_rows = blob + C::kBlobUp + (size_t) row_base * C::kGateRowBytes;

    uint4 pf[C::kGuPF];
    gu_load<G>(pf, gate_rows, lane);                       // stage (p0, h0, gate): in flight while the activations are copied

    // activations of the group's tokens -> shared memory (lo / hi 16-byte halves of every block, scales)
    for (int j = 0; j < n; ++j) {
        const int token = hits[gp->hit[j]].token;
        const unsigned char* src = reinterpret_cast<const unsigned char*>(xq + (size_t) token * kHidden);
        for (int c = threadIdx.x; c < 2 * kActBlocks; c += kExThreads) {
            const uint4 v = ldg4(src + 16 * c);
            (c & 1 ? act_hi : act_lo)[j * kActBlocks + (c >> 1)] = v;
        }
        for (int c = threadIdx.x; c < kActBlocks / 4; c += kExThreads) {
            DS41_ASSERT_ALIGNED(act_sc + j * kActBlocks + 4 * c, 16);
            reinterpret_cast<float4*>(act_sc + j * kActBlocks)[c] = ldgf4(xs + (size_t) token * kActBlocks + 4 * c);
        }
    }
    sync_block();

    float acc_g[NT], acc_u[NT];
    DS41_UNROLL
    for (int t = 0; t < NT; ++t) acc_g[t] = acc_u[t] = 0.0f;
    const unsigned char* seg = stage + rr * C::kGuSeg;

    DS41_UNROLL1
    for (int ph = 0; ph < 2 * C::kGuSplits; ++ph) {
        const int p = ph >> C::kGuSplitLog2, half = ph & (C::kGuSplits - 1);
        const size_t off = (size_t) (2 * p) * C::kGateRowBytes + (size_t) half * C::kGuSegBytes;
        // ---- gate stage of (p, half)
        gu_store<G>(pf, stage, lane);
        sync_warp();
        gu_load<G>(pf, up_rows + off, lane);
        stage_dot<NT, C::kGuSegBlocks, kActBlocks>(seg, kk, half * C::kGuSegBlocks, act_lo, act_hi, act_sc, n, acc_g);
        sync_warp();
        // ---- up stage of (p, half)
        gu_store<G>(pf, stage, lane);
        sync_warp();
        if (ph < 2 * C::kGuSplits - 1) {
            const int pn = (ph + 1) >> C::kGuSplitLog2, hn = (ph + 1) & (C::kGuSplits - 1);
            gu_load<G>(pf, gate_rows + (size_t) (2 * pn) * C::kGateRowBytes + (size_t) hn * C::kGuSegBytes, lane);
        }
        stage_dot<NT, C::kGuSegBlocks, kActBlocks>(seg, kk, half * C::kGuSegBlocks, act_lo, act_hi, act_sc, n, acc_u);
        sync_warp();

        if (half == C::kGuSplits - 1) {
            // ---- the pair's rows are complete: reduce over the 16 lanes of each row, then h = silu(g) * u * w
            constexpr int S = ilog2(2 * NT);
            float v[2 * NT];
            DS41_UNROLL
            for (int t = 0; t < NT; ++t) {
                v[t] = acc_g[t];
                v[NT + t] = acc_u[t];
                acc_g[t] = acc_u[t] = 0.0f;
            }
            const float mine = halfwarp_vector_sum<2 * NT>(v);       // lane kk holds value index kk >> (4 - S): gate for kk < 8
            const float other = shfl_xor(mine, 8);                   // lane kk < 8 gets the matching up value
            const int idx = kk >> (4 - S);
            if (kk < 8 && (kk & ((1 << (4 - S)) - 1)) == 0 && idx < n) {
                // NaN PROPAGATES through the clamps, as through torch.clamp (CONTRACTS.md): the ternaries compare false for a NaN and return it.
                // Not fminf / fmaxf (they return the constant for a NaN operand and would launder a failure upstream into a finite number).
                const float gv = mine > kSwigluLimit ? kSwigluLimit : mine;
                const float uv = other > kSwigluLimit ? kSwigluLimit : (other < -kSwigluLimit ? -kSwigluLimit : other);
                const float w = hits[gp->hit[idx]].weight;
                s_h[idx * 32 + warp * 4 + 2 * p + rr] = (gv / (1.0f + expf(-gv))) * uv * w;
            }
        }
    }
    sync_block();

    // ---- one warp per token quantises the tile's 32 rows (= one h block) and stores them at the HIT's index
    if (warp < NT && warp < n) {
        const int hit = gp->hit[warp];
        quantize_block_warp(s_h[warp * 32 + lane], lane, hq + ((size_t) hit * kHBlocks + tile) * 32,
                            hs + (size_t) hit * kHBlocks + tile);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// phase 2: down
// ---------------------------------------------------------------------------------------------------------------------
template <class G, int NT>
DS41_KERNEL DS41_LAUNCH_BOUNDS(256, (NT == 1 ? 3 : 2)) void down_kernel(const uint8_t* DS41_RESTRICT cache, const HitEntry* DS41_RESTRICT hits,
                                                        const HitGroup* DS41_RESTRICT groups,
                                                        const SplitCounts* DS41_RESTRICT counts,
                                                        const int8_t* DS41_RESTRICT hq, const float* DS41_RESTRICT hs,
                                                        float* DS41_RESTRICT parts, int n_lo, int n_hi) {
    using C = ExCfg<G>;
    using L = DnLayout<G, NT>;
    constexpr int kHidden = C::kHidden, kFF = C::kFF, kTopK = C::kTopK, kHBlocks = C::kHBlocks;
    const int g = blockIdx.y;
    if (g >= counts->n_groups) return;
    const int tile = blockIdx.x;
    const HitGroup* gp = groups + g;
    const int n = gp->n;
    if (n < n_lo || n > n_hi) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int rr = lane >> 4, kk = lane & 15;

    DS41_DYN_SMEM(smem);
    uint4* act_lo = reinterpret_cast<uint4*>(smem + L::lo);
    uint4* act_hi = reinterpret_cast<uint4*>(smem + L::hi);
    float* act_sc = reinterpret_cast<float*>(smem + L::sc);
    float* s_y = reinterpret_cast<float*>(smem + L::y);
    unsigned char* stage = smem + L::stage + warp * C::kDnStageBytes;

    const uint8_t* blob = cache + (size_t) gp->slot * C::kBlobBytes;
    const int row0 = tile * kDnTileRows + warp * 8;        // this warp: 4 row pairs
    const uint8_t* rows = blob + C::kBlobDown + (size_t) row0 * C::kDownRowBytes;

    uint2 pf[C::kDnPF];
    dn_load<G>(pf, rows, lane);

    for (int j = 0; j < n; ++j) {
        const int hit = gp->hit[j];
        const unsigned char* src = reinterpret_cast<const unsigned char*>(hq + (size_t) hit * kFF);
        for (int c = threadIdx.x; c < 2 * kHBlocks; c += kExThreads) {
            const uint4 v = ldg4(src + 16 * c);
            (c & 1 ? act_hi : act_lo)[j * kHBlocks + (c >> 1)] = v;
        }
        for (int c = threadIdx.x; c < kHBlocks / 4; c += kExThreads) {
            DS41_ASSERT_ALIGNED(act_sc + j * kHBlocks + 4 * c, 16);
            reinterpret_cast<float4*>(act_sc + j * kHBlocks)[c] = ldgf4(hs + (size_t) hit * kHBlocks + 4 * c);
        }
    }
    sync_block();

    const unsigned char* seg = stage + rr * C::kDnSeg;
    DS41_UNROLL1
    for (int p = 0; p < 4; ++p) {
        dn_store<G>(pf, stage, lane);
        sync_warp();
        if (p < 3) dn_load<G>(pf, rows + (size_t) (2 * (p + 1)) * C::kDownRowBytes, lane);
        float acc[NT];
        DS41_UNROLL
        for (int t = 0; t < NT; ++t) acc[t] = 0.0f;
        stage_dot<NT, kHBlocks, kHBlocks>(seg, kk, 0, act_lo, act_hi, act_sc, n, acc);
        sync_warp();
        constexpr int S = ilog2(NT);
        const float mine = halfwarp_vector_sum<NT>(acc);
        const int idx = kk >> (4 - S);
        if ((kk & ((1 << (4 - S)) - 1)) == 0 && idx < n) s_y[idx * kDnTileRows + warp * 8 + 2 * p + rr] = mine;
    }
    sync_block();

    // the tile's 64 outputs of every token, coalesced: parts[(token*6 + k)][64 tile .. 64 tile + 63]
    for (int i = threadIdx.x; i < n * kDnTileRows; i += kExThreads) {
        const int t = i / kDnTileRows, r = i - t * kDnTileRows;
        const HitEntry e = hits[gp->hit[t]];
        parts[((size_t) e.token * kTopK + e.k) * kHidden + (size_t) tile * kDnTileRows + r] = s_y[t * kDnTileRows + r];
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// host side
// ---------------------------------------------------------------------------------------------------------------------
/// `groups_grid` = how many group slots the launch covers (grid.y); blocks whose group has n outside [n_lo, n_hi] return at once.
template <class G, int NT>
inline void launch_gate_up(const uint8_t* cache, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int groups_grid,
                           int n_lo, int n_hi, const int8_t* xq, const float* xs, int8_t* hq, float* hs, void* stream) {
    const size_t smem = GuLayout<G, NT>::total;
    const dim3 grid(ExCfg<G>::kGuTiles, (unsigned) groups_grid);
#if !defined(DS41_EMU)
    static const bool once = (prepare_kernel(gate_up_kernel<G, NT>, smem), true);
    (void) once;
#endif
    launch(gate_up_kernel<G, NT>, grid, dim3(kExThreads), smem, stream, cache, hits, groups, counts, xq, xs, hq, hs, n_lo, n_hi);
    check_launch("gate_up");
}

template <class G, int NT>
inline void launch_down(const uint8_t* cache, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int groups_grid,
                        int n_lo, int n_hi, const int8_t* hq, const float* hs, float* parts, void* stream) {
    const size_t smem = DnLayout<G, NT>::total;
    const dim3 grid(ExCfg<G>::kDnTiles, (unsigned) groups_grid);
#if !defined(DS41_EMU)
    static const bool once = (prepare_kernel(down_kernel<G, NT>, smem), true);
    (void) once;
#endif
    launch(down_kernel<G, NT>, grid, dim3(kExThreads), smem, stream, cache, hits, groups, counts, hq, hs, parts, n_lo, n_hi);
    check_launch("down");
}

inline void check_T(int T, const char* who) {
    if (T < 1 || T > kMaxExpertTokens) throw std::invalid_argument(std::string(who) + ": T must be 1..8");
}
inline void check_align(const void* p, size_t a, const char* what) {
    if (reinterpret_cast<uintptr_t>(p) % a != 0) throw std::invalid_argument(std::string("ds41: ") + what + " must be " + std::to_string(a) + "-byte aligned");
}

/// The expert path's quantiser: x [T][kHidden] -> xq, xs, INTERLEAVED, compile-time width.
template <class G>
inline void quantize_x(const float* x, int T, int8_t* xq, float* xs, void* stream) {
    if (T < 1) return;
    check_align(x, 4, "x");
    constexpr int NB = ExCfg<G>::kActBlocks;
    const dim3 grid((unsigned) ((T * NB + kExWarps - 1) / kExWarps));
    launch(quantize_acts_kernel<NB, true>, grid, dim3(kExThreads), 0, stream, x, T, xq, xs, 0);
    check_launch("quantize_acts");
}

}  // namespace strata::ds41::cuda::dev

namespace strata::ds41::cuda {

template <class G>
void ds41_quantize_acts(Dev& dev, const float* x, int T, int width, int8_t* xq, float* xs, Stream stream, ActOrder order) {
    if (T < 1) throw std::invalid_argument("ds41_quantize_acts: T must be >= 1");
    if (width < 32 || width % 32 != 0) throw std::invalid_argument("ds41_quantize_acts: width must be a positive multiple of 32");
    dev::check_align(x, 4, "x");
    dev::check_align(xq, 16, "xq");
    dev::check_align(xs, (width / 32) % 4 == 0 ? 16 : 4, "xs");
    stream = stream_or_default(dev, stream);
    const int nb = width / 32;
    const dim3 grid((unsigned) (((size_t) T * nb + dev::kExWarps - 1) / dev::kExWarps));
    if (order == ActOrder::kInterleaved && width == G::kHidden) {
        dev::quantize_x<G>(x, T, xq, xs, stream);
        return;
    }
    if (order == ActOrder::kInterleaved) dev::launch(dev::quantize_acts_kernel<0, true>, grid, dim3(dev::kExThreads), 0, stream, x, T, xq, xs, nb);
    else dev::launch(dev::quantize_acts_kernel<0, false>, grid, dim3(dev::kExThreads), 0, stream, x, T, xq, xs, nb);
    dev::check_launch("ds41_quantize_acts");
}

// Which specialisation runs which groups.  T = 1: NT = 1;  T = 2: NT = 2;  T = 3, 4: NT = 4;  T = 5..8: two launches over the SAME group
// list (sorted by decreasing size, see split): NT = 4 for the groups of 1..4 tokens (47 KB shared memory per block, 2 blocks per SM) and
// NT = 8 for the rare groups of 5..8 tokens (71 KB, 1 block per SM) - at most floor(6T/5) of them, so that launch covers only that many
// group slots.  A group never has more tokens than T.
template <class G>
void experts_gate_up(const uint8_t* cache_base, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int T,
                     const int8_t* xq, const float* xs, int8_t* hq, float* hs, void* stream) {
    dev::check_T(T, "experts_gate_up");
    dev::check_align(cache_base, 256, "cache_base");
    dev::check_align(xq, 16, "xq");
    dev::check_align(xs, 16, "xs");
    dev::check_align(hq, 16, "hq");
    dev::check_align(hs, 16, "hs");
    const int GR = G::kTopK * T;
    if (T == 1) dev::launch_gate_up<G, 1>(cache_base, hits, groups, counts, GR, 1, 1, xq, xs, hq, hs, stream);
    else if (T == 2) dev::launch_gate_up<G, 2>(cache_base, hits, groups, counts, GR, 1, 2, xq, xs, hq, hs, stream);
    else if (T <= 4) dev::launch_gate_up<G, 4>(cache_base, hits, groups, counts, GR, 1, 4, xq, xs, hq, hs, stream);
    else {
        dev::launch_gate_up<G, 4>(cache_base, hits, groups, counts, GR, 1, 4, xq, xs, hq, hs, stream);
        dev::launch_gate_up<G, 8>(cache_base, hits, groups, counts, GR / 5, 5, 8, xq, xs, hq, hs, stream);
    }
}

template <class G>
void experts_down(const uint8_t* cache_base, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts, int T,
                  const int8_t* hq, const float* hs, float* parts, void* stream) {
    dev::check_T(T, "experts_down");
    dev::check_align(cache_base, 256, "cache_base");
    dev::check_align(hq, 16, "hq");
    dev::check_align(hs, 16, "hs");
    const int GR = G::kTopK * T;
    if (T == 1) dev::launch_down<G, 1>(cache_base, hits, groups, counts, GR, 1, 1, hq, hs, parts, stream);
    else if (T == 2) dev::launch_down<G, 2>(cache_base, hits, groups, counts, GR, 1, 2, hq, hs, parts, stream);
    else if (T <= 4) dev::launch_down<G, 4>(cache_base, hits, groups, counts, GR, 1, 4, hq, hs, parts, stream);
    else {
        dev::launch_down<G, 4>(cache_base, hits, groups, counts, GR, 1, 4, hq, hs, parts, stream);
        dev::launch_down<G, 8>(cache_base, hits, groups, counts, GR / 5, 5, 8, hq, hs, parts, stream);
    }
}

template <class G>
void experts_hits(const uint8_t* cache_base, const float* x, const HitEntry* hits, const HitGroup* groups,
                  const SplitCounts* counts, int T, const ExpertScratch& s, float* parts, void* stream) {
    dev::quantize_x<G>(x, T, s.xq, s.xs, stream);
    experts_gate_up<G>(cache_base, hits, groups, counts, T, s.xq, s.xs, s.hq, s.hs, stream);
    experts_down<G>(cache_base, hits, groups, counts, T, s.hq, s.hs, parts, stream);
}

namespace dev {
template <class G, int NT>
inline ExpertKernelInfo kernel_info(bool down) {
    ExpertKernelInfo r;
#if !defined(DS41_EMU)
    cudaFuncAttributes a{};
    const size_t smem = down ? DnLayout<G, NT>::total : GuLayout<G, NT>::total;
    cudaError_t e = down ? cudaFuncGetAttributes(&a, down_kernel<G, NT>) : cudaFuncGetAttributes(&a, gate_up_kernel<G, NT>);
    if (e != cudaSuccess) return r;
    r.regs = a.numRegs;
    r.static_smem = (int) a.sharedSizeBytes;
    r.dyn_smem = (int) smem;
    int blocks = 0;
    if (down) {
        prepare_kernel(down_kernel<G, NT>, smem);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, down_kernel<G, NT>, kExThreads, smem);
    } else {
        prepare_kernel(gate_up_kernel<G, NT>, smem);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, gate_up_kernel<G, NT>, kExThreads, smem);
    }
    r.blocks_per_sm = e == cudaSuccess ? blocks : 0;
#else
    (void) down;
#endif
    return r;
}
}  // namespace dev

template <class G>
void experts_prepare() {
#if !defined(DS41_EMU)
    dev::prepare_kernel(dev::gate_up_kernel<G, 1>, dev::GuLayout<G, 1>::total);
    dev::prepare_kernel(dev::gate_up_kernel<G, 2>, dev::GuLayout<G, 2>::total);
    dev::prepare_kernel(dev::gate_up_kernel<G, 4>, dev::GuLayout<G, 4>::total);
    dev::prepare_kernel(dev::gate_up_kernel<G, 8>, dev::GuLayout<G, 8>::total);
    dev::prepare_kernel(dev::down_kernel<G, 1>, dev::DnLayout<G, 1>::total);
    dev::prepare_kernel(dev::down_kernel<G, 2>, dev::DnLayout<G, 2>::total);
    dev::prepare_kernel(dev::down_kernel<G, 4>, dev::DnLayout<G, 4>::total);
    dev::prepare_kernel(dev::down_kernel<G, 8>, dev::DnLayout<G, 8>::total);
#endif
}

template <class G>
ExpertKernelInfo expert_kernel_info(int NT, bool down) {
    switch (NT) {
        case 1: return dev::kernel_info<G, 1>(down);
        case 2: return dev::kernel_info<G, 2>(down);
        case 4: return dev::kernel_info<G, 4>(down);
        case 8: return dev::kernel_info<G, 8>(down);
        default: throw std::invalid_argument("expert_kernel_info: NT must be 1, 2, 4 or 8");
    }
}

void test_e8m0_table(float* out256, void* stream) {
    dev::launch(dev::e8m0_table_kernel, dim3(1), dim3(256), 0, stream, out256);
    dev::check_launch("test_e8m0_table");
}

}  // namespace strata::ds41::cuda

/// Explicit instantiation of the hot-expert host entry points for geometry G (the .cu: RealGeom; the emulator build: RealGeom and MiniGeom).
#define DS41_INSTANTIATE_EXPERTS(G)                                                                                                                      \
    template void ::strata::ds41::cuda::ds41_quantize_acts<G>(::strata::ds41::cuda::Dev&, const float*, int, int, int8_t*, float*, void*,                \
                                                              ::strata::ds41::cuda::ActOrder);                                                          \
    template void ::strata::ds41::cuda::experts_prepare<G>();                                                                                            \
    template void ::strata::ds41::cuda::experts_gate_up<G>(const uint8_t*, const ::strata::ds41::cuda::HitEntry*, const ::strata::ds41::cuda::HitGroup*,   \
                                                           const ::strata::ds41::cuda::SplitCounts*, int, const int8_t*, const float*, int8_t*, float*, void*); \
    template void ::strata::ds41::cuda::experts_down<G>(const uint8_t*, const ::strata::ds41::cuda::HitEntry*, const ::strata::ds41::cuda::HitGroup*,      \
                                                        const ::strata::ds41::cuda::SplitCounts*, int, const int8_t*, const float*, float*, void*);       \
    template void ::strata::ds41::cuda::experts_hits<G>(const uint8_t*, const float*, const ::strata::ds41::cuda::HitEntry*,                              \
                                                        const ::strata::ds41::cuda::HitGroup*, const ::strata::ds41::cuda::SplitCounts*, int,             \
                                                        const ::strata::ds41::cuda::ExpertScratch&, float*, void*);                                        \
    template auto ::strata::ds41::cuda::expert_kernel_info<G>(int, bool) -> ::strata::ds41::cuda::ExpertKernelInfo;
