// include/strata/ds41/cpu/mxfp4_expert.hpp - DS-C: the CPU path for the routed experts the GPU cache misses.
//
// WHAT IS COMPUTED (docs/deepseek/CONTRACTS.md, "The math"), for a token with activation x (5120) and routing weight w:
//
//     g = W1 . x        u = W3 . x                      (MXFP4 weights, int8 activations, FP32 results)
//     g = min(g, 10)    u = clamp(u, -10, 10)
//     h = silu(g) * u * w                               (the routing weight goes in HERE, before W2)
//     y += W2 . h                                       (h is re-quantised to int8 per 32 first)
//
// and the expert lives on the CPU as TWO HALVES (PLAN.md section 2), one per socket.  Half k owns intermediate rows
// [1152 k, 1152 k + 1152): it computes h for those rows from its own gate/up rows and multiplies them by its own
// slice of W2's columns, giving a partial y; the expert's output is partial_0 + partial_1.  This header is the
// kernel layer under that: a "half" is the unit the NUMA pool schedules, and it is cut into ROW RANGES so that
// several threads can share one half without a lock or an atomic.
//
// THE THREADING SCHEME (the whole contract of the row-range functions; the pool follows it, the kernels do not
// synchronise):
//
//     phase 1   expert_gate_up(..., chunk0, chunk1)   thread i takes a range of 32-row CHUNKS of the intermediate
//               (1152 rows = 36 chunks; split_range() cuts them).  For its rows it computes g and u for all T
//               tokens, h = silu(g) * u * w, and QUANTISES h: a chunk of 32 rows is exactly one int8 block of the
//               input of W2, so the owner of the chunk also writes its quantised image into the shared
//               ExpertScratch.  No separate quantisation step, no overlap between threads.
//     --------  barrier (every chunk's h is needed by every down row)
//     phase 2   expert_down(..., row0, row1)          thread i takes a range of the 5120 output rows, reads the
//               whole quantised h, and ADDS its rows of y.  Rows are disjoint, so there is nothing to merge.
//
// Within a phase each output element is produced by exactly one thread, and its value does not depend on how the
// ranges were cut or on how many tokens ride along: every (row, token) pair is one fixed sequence of operations.
// That makes the ragged splits, the single-thread run and the token-by-token run BITWISE identical, which is what
// src/ds41/cpu/mxfp4_expert_test.cpp checks (for one ISA; different ISAs differ in the last bits of the FP32 sums).
// It also means the ranges can be claimed dynamically (a thread takes the next 32-row chunk / 64 output rows from an
// atomic counter), which is what the benchmark does so that one slow or preempted core delays only the unit it holds.
//
// Several experts at once.  Each in-flight expert needs its own ExpertScratch.  A thread that finishes its share of
// expert A's phase 2 may start expert B's phase 1 while others are still on A: the only dependency is phase 1 -> phase 2
// OF THE SAME EXPERT.  The one thing to watch is `y`: phase 2 of different experts adds into the same rows, so either give
// every thread the same row range in phase 2 for all experts of the layer (static ownership: no two threads ever touch a
// row), or use one `y` per expert and add them afterwards.
//
// THE WEIGHTS ARE READ ONCE FOR ALL T TOKENS (T = 1..8, a verify window).  A 17-byte block is decoded to int8 once
// per group of four blocks and applied to every token's activation from registers, so a verify window costs one
// pass over the expert's bytes plus T times the (much cheaper) integer work.  That is the whole reason the
// functions take T instead of being called once per token.
//
// HOW THE MXFP4 DOT PRODUCT IS DONE (see src/ds41/cpu/mxfp4_avx512.cpp for the same story with the instructions):
//
//   * A block is [e][qs 16]: value j (0..15) is the LOW nibble of qs[j], value j + 16 the HIGH nibble, and the value
//     is kvalues_fp4[code] * 2^(e - 128) with kvalues_fp4 = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12} (the e2m1
//     table DOUBLED, so the integers are exact and the 0.5 lives in the exponent).  The exponent byte is E8M0 and is
//     decoded exactly as ggml does (`ggml_e8m0_to_fp32_half`): e = 0 and 1 are the denormals 2^-128 and 2^-127,
//     e = 255 is 2^127 (not NaN).  Every exponent is exact in FP32, so the scale is just a float.
//   * Decoding is a table lookup: `vpshufb` turns sixteen 4-bit codes into sixteen bytes in ONE instruction.  The
//     table holds kvalues + 12 (0..24), UNSIGNED, because the integer dot `vpdpbusd` (AVX-512 VNNI) multiplies
//     unsigned by signed bytes.  The offset is removed exactly with a per-lane correction -12 * sum(x) that depends
//     only on the activation and is stored with it (ActQ::corr), so the signed product costs no extra instruction
//     per weight block and the same correction serves every row.  (AVX2 uses `vpmaddubsw` + `vpmaddwd` on the same
//     operands: 24 * 127 * 4 < 32767, so the 16-bit stage cannot saturate.)
//   * One scale per 32-block: e8m0 * activation scale is applied in FP32 to the exact int32 partial sums; the
//     accumulator stays a FLOAT VECTOR for the whole row and is reduced once at its end (as Strata's own expert
//     kernel found: reducing per block costs more than the dot products).
//
// THE ACTIVATION LAYOUT IS THE KERNEL'S, NOT THE NATURAL ORDER.  Four consecutive 32-blocks (a "group", 128 values)
// are stored as [low-nibble halves of blocks 0..3 (4 x 16 B)][high-nibble halves of blocks 0..3 (4 x 16 B)], which is
// exactly the order in which one 64-byte register of looked-up weights lines up with one 64-byte load of
// activations.  Use act_unpack() / act_q() to read the values in natural order; quantize_act() is the only producer.
// A weight row therefore must have a multiple of 4 blocks (all of this model's rows do: 160, 72, 36).
//
// ACTIVATION QUANTISATION (CONTRACTS.md): per 32 values d = max|x| / 127, q = round-to-nearest-even(x * (127 / max|x|))
// (all zero -> d = 0, q = 0).  All three implementations are bit-identical on finite input.  This is more precise than
// the reference's FP8-e4m3 fake quantisation; DS-A's oracle provides both.
//
// WHICH CODE RUNS: Isa::kAuto picks the best the CPU has (CPUID + OS state).  Scalar is the exact-semantics
// reference (slow); AVX2 needs AVX2 + FMA + F16C; AVX-512 needs F/BW/VL/DQ + VNNI and does NOT need VBMI (the
// target Xeon Gold 6226 is Cascade Lake).  Requesting an unsupported ISA is a programming error and aborts.
#pragma once

#include "strata/ds41/geometry.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::ds41::cpu {

// ---- constants -----------------------------------------------------------------------------------------------------
inline constexpr int kMaxTokens = 8;                       // a verify window: T = 1..8
inline constexpr int kGroupBlocks = 4;                     // blocks per kernel group (the layout unit)
inline constexpr int kGroupValues = kGroupBlocks * kQK;    // 128
inline constexpr int kActMaxBlocks = kHidden / kQK;        // 160: the widest vector quantised here (x); h needs <= 72
inline constexpr int kChunkRows = kQK;                     // gate/up row ranges are cut on multiples of 32 rows

/// kvalues_fp4: the e2m1 table doubled (ggml `kvalues_fp4` / `kvalues_mxfp4`).
extern const int8_t kMxfp4Values[16];

/// ggml's `ggml_e8m0_to_fp32_half(e)` = 2^(e - 128), exactly, for all 256 values (e = 0, 1 denormal; e = 255 = 2^127).
float e8m0_half(uint8_t e);

// ---- which instruction set ----------------------------------------------------------------------------------------
enum class Isa { kAuto, kScalar, kAvx2, kAvx512 };
bool isa_supported(Isa isa);          // kAuto and kScalar are always supported
Isa resolve_isa(Isa isa);             // kAuto -> the best supported one; anything else is returned as is
const char* isa_name(Isa isa);

/// Software prefetch distance in bytes, per weight-row stream (0 = none; the hardware prefetcher still runs).  Default 8192:
/// the dev VM's optimum is flat between 6 and 12 KB, and the Xeon's will differ - `ds41_cpu_mxfp4_bench --sweep-pf` finds it.
/// A tuning knob, global, read at the start of every kernel call: set it before starting worker threads.
void set_prefetch_bytes(int bytes);
int prefetch_bytes();

// ---- quantised activations ---------------------------------------------------------------------------------------
/// One quantised vector of up to 5120 values: int8 per 32-block plus an FP32 scale.  Besides the int8 values and
/// the scales it carries what the kernels would otherwise recompute for every weight row: the scale repeated over
/// the four lanes of each block, and the offset correction.  ~10.9 KB per vector; a verify window of 8 is 87 KB.
/// Always produced by quantize_act() (x) or by expert_gate_up() (the intermediate h); never filled by hand.
struct ActQ {
    alignas(64) int8_t q[kActMaxBlocks * kQK];      // kernel layout (see the header comment), SIGNED int8
    alignas(64) float sc4[kActMaxBlocks * 4];       // per group: 16 floats = [d0 x4, d1 x4, d2 x4, d3 x4]
    alignas(64) int32_t corr[kActMaxBlocks * 4];    // per group: 16 lanes = -12 * (the 8 q values the lane covers)
    float scale[kActMaxBlocks];                     // d per block, natural order
};
static_assert(sizeof(ActQ) % 64 == 0, "an array of ActQ keeps every member 64-byte aligned");

/// Index of natural element `i` (block i / 32, position i % 32) inside ActQ::q.
inline constexpr int act_index(int i) {
    const int b = i >> 5, j = i & 31;
    return (b >> 2) * kGroupValues + (j >> 4) * 64 + (b & 3) * 16 + (j & 15);
}
inline int8_t act_q(const ActQ& a, int i) { return a.q[act_index(i)]; }
/// Natural-order int8 values of the first `n` elements.
void act_unpack(const ActQ& a, int n, int8_t* out);
/// The dequantised values q * d, natural order.
void act_dequant(const ActQ& a, int n, float* out);

/// Build an ActQ from values quantised elsewhere: natural-order int8 `q[n]` and one scale per 32-block `scale[n / 32]`
/// (the oracle's quantiser, a Q8_0 vector with its fp16 scales widened, a test's hand-made input).  Values in [-127, 127].
void act_from_q8(const int8_t* q, const float* scale, int n, ActQ& out);
/// x[n] -> out.  `n` must be a multiple of 128 (a group) and at most 5120.
void quantize_act(const float* x, int n, ActQ& out, Isa isa = Isa::kAuto);
/// x[T][n] (row stride n) -> out[T].
void quantize_acts(const float* x, int n, int T, ActQ* out, Isa isa = Isa::kAuto);

// ---- where an expert's weights are -----------------------------------------------------------------------------
/// Three row-major MXFP4 matrices.  gate and up have `hidden / 32` blocks per row (stride hidden / 32 * 17 B); down has
/// `hidden` rows of `ff / 32` blocks, `down_row_stride` bytes apart.  A view carries no ownership.
///
/// The three builders below describe the real layouts; the fields are public so the tests can also run tiny shapes
/// (any hidden and ff that are multiples of 128; ff <= 5120).
struct ExpertView {
    const uint8_t* gate = nullptr;     // `ff` rows
    const uint8_t* up = nullptr;       // `ff` rows
    const uint8_t* down = nullptr;     // `hidden` rows
    size_t down_row_stride = 0;
    int hidden = kHidden;              // activation width = rows of down
    int ff = kHalfFF;                  // intermediate rows this view covers
    int gate_row_blocks() const { return hidden / kQK; }
    size_t gate_row_bytes() const { return (size_t) (hidden / kQK) * kBlockBytes; }
    int down_row_blocks() const { return ff / kQK; }
    int chunks() const { return ff / kChunkRows; }
};
/// A CPU half (9,400,320 B, geometry.hpp): rows of down are 36 blocks, 612 B apart.
ExpertView view_cpu_half(const uint8_t* half);
/// Half `h` of a GPU-layout blob (18,800,640 B), in place: down rows are 36 blocks of the 72, 1224 B apart.
ExpertView view_blob_half(const uint8_t* blob, int h);
/// The whole expert of a GPU-layout blob: ff = 2304, down rows of 72 blocks.
ExpertView view_blob(const uint8_t* blob);
/// Copy half `h` of a GPU-layout blob into the CPU-half layout (what DS-B's tool and the engine's loader produce).
void pack_cpu_half(const uint8_t* blob, int h, uint8_t* out);

// ---- the expert, in row ranges ---------------------------------------------------------------------------------
/// The quantised intermediate of one expert for up to kMaxTokens tokens.  Shared by all threads working on the
/// same expert (phase 1 writes disjoint blocks of it, phase 2 reads all of it); one per concurrently running expert.
struct ExpertScratch {
    ActQ h[kMaxTokens];
};

/// `n` items cut into `parts` ranges for `part` in [0, parts), each boundary a multiple of `align` (except `n`).
/// The ranges are contiguous, cover [0, n) and differ in size by at most one `align`.
void split_range(int n, int parts, int part, int align, int& lo, int& hi);

/// PHASE 1 for the chunks [chunk0, chunk1) of the intermediate (32 rows each): g and u for all T tokens, the clamps,
/// h = silu(g) * u * route_w[t], and h's int8 image written into s.h[t] (its blocks chunk0..chunk1).
/// `x` holds T quantised activations (width view.hidden), `route_w` T routing weights.  Reads the weights of the
/// rows exactly once.  Threads call it with disjoint chunk ranges; the barrier before phase 2 is the caller's.
void expert_gate_up(Isa isa, const ExpertView& v, const ActQ* x, int T, const float* route_w, ExpertScratch& s,
                    int chunk0, int chunk1);

/// PHASE 2 for the output rows [row0, row1): y[t * view.hidden + r] += W2[r, :] . h_t.  `y` is [T][hidden], FP32,
/// ADDED to (zero it before the first expert of a layer; the halves and the experts of a layer all accumulate into it).
/// Needs every chunk of phase 1 to be done.  Threads call it with disjoint row ranges.
void expert_down(Isa isa, const ExpertView& v, const ExpertScratch& s, int T, float* y, int row0, int row1);

/// Both phases on the calling thread: the whole view, T tokens.  The reference arrangement of the two functions
/// above, the non-NUMA single-thread path, and what the tests compare the multi-threaded schedules with.
void expert_run(Isa isa, const ExpertView& v, const ActQ* x, int T, const float* route_w, ExpertScratch& s, float* y);

/// The whole expert of a GPU-layout blob in one pass (ff = 2304, so one down row is 72 blocks and there is no
/// partial to add): y[t][r] += expert(x_t) * route_w[t].  For tests and a non-NUMA path.  The sum of the two
/// halves' partials equals this up to the order of FP32 additions.
void expert_run_blob(Isa isa, const uint8_t* blob, const ActQ* x, int T, const float* route_w, ExpertScratch& s,
                     float* y);

// ---- the primitive under both phases (exposed for the tests and the benchmark) -------------------------------------
/// out[t * out_stride + r] (= or +=) dot(row r of W, x_t) for r < nrows, t < T:
///   sum over blocks b of [ e8m0_half(e_b) * x_t.scale[b] ] * [ sum_j kvalues[code_j] * x_t.q[32 b + j] ]
/// with the inner sum an exact int32 and the bracketed scale one FP32 product.  `w` points at row 0, rows are
/// `row_stride` bytes apart and have `nblocks` (a multiple of 4) blocks.  The scalar version sums in double, the
/// SIMD versions in FP32 lanes: they agree to ~1e-6 of the sum of |terms| (the integer part is identical).
void mxfp4_dot_rows(Isa isa, const uint8_t* w, size_t row_stride, int nblocks, int nrows, const ActQ* x, int T,
                    float* out, size_t out_stride, bool accumulate);

}  // namespace strata::ds41::cpu
