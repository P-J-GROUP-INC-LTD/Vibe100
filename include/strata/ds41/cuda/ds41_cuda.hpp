// include/strata/ds41/cuda/ds41_cuda.hpp - DS-D: the GPU half of the DeepSeek-V4.1-Flash MoE layer, for the V100 (sm_70).
//
// THREE JOBS, IN THE ORDER A LAYER NEEDS THEM (shapes: include/strata/ds41/geometry.hpp, math: docs/deepseek/CONTRACTS.md):
//
//   1. ROUTER        router_forward():  x fp32 [T][5120], Wg bf16 [384][5120], bias fp32 [384]
//                    -> ids int32 [T][6], weights fp32 [T][6]
//                    logits in FP32, s = sqrt(softplus(logit)), top-6 of (s + bias) with the LOWEST index winning ties,
//                    w = s[ids], w = w / (sum(w) + 1e-20) * 1.5.   Any T from 1 to a few thousand.
//   2. SPLIT         split_hits_misses():  the router's ids against the residency table of the layer ->
//                    HIT list (GPU computes), MISS list (the CPU pool computes), counts, and the hits grouped by
//                    cache slot.  Entirely on the device: the router's output never visits the host.
//   3. HIT EXPERTS   quantize_acts() -> experts_gate_up() -> experts_down()   (experts_hits() runs all three):
//                    the MXFP4 blobs sitting in VRAM cache slots, T = 1..8 tokens, ALL hits of the layer in one launch
//                    per phase (two for T >= 5: a lean and a wide specialisation over the same list), device-side lists only
//                    (no host synchronisation between the router and the experts).
//
// ---------------------------------------------------------------------------------------------------------------------
// DEVICE-SIDE LIST FORMATS (all plain structs, 16-byte multiples, little endian; the miss list is meant to be read by
// the host)
// ---------------------------------------------------------------------------------------------------------------------
//   HitEntry   { int32 token; int32 k; int32 slot; float weight; }   token = 0..T-1, k = 0..5 (rank in the router's
//              output, i.e. ids[token*6 + k]), slot = the cache slot holding that expert's blob, weight = the routing
//              weight (already normalised and scaled).  Entries are in (token, k) order: deterministic.
//   MissEntry  { int32 token; int32 k; int32 expert; float weight; } the same, but with the expert id (0..383) instead of
//              a slot; this is what the host's CPU expert pool reads.  Also in (token, k) order.
//   SplitCounts{ int32 n_hits; int32 n_misses; int32 n_groups; int32 reserved; }   written LAST (see below).
//   HitGroup   { int32 slot; int32 n; int32 hit[8]; }   (T <= 8 only)  the hits that share a cache slot, i.e. the same
//              expert for several tokens: n = how many (1..8), hit[j] = index into the HIT list (members in hit order).  The
//              expert kernels stream the blob ONCE per group and use it for all n tokens.  Groups are numbered by DECREASING n
//              (ties: order of first appearance): the expert kernels run the groups with 5..8 tokens in a wide specialisation.
//   Capacity: hits, misses and groups each hold up to 6*T entries.  Nothing is zeroed beyond the counts.
//
//   Host visibility.  Pass host_visible = true when `misses` and `counts` live in mapped (zero-copy) pinned memory: the kernel then
//   issues __threadfence_system() after the last list entry and again between the first three counts and `reserved`, so a host that
//   polls for a doorbell sees complete data: pre-set counts->reserved to a nonzero sentinel before the launch (and the lists/counts to
//   whatever you like), launch, and wait until `reserved` reads 0 (volatile read, then an acquire fence): n_hits, n_misses, n_groups and
//   every entry of the miss list are then valid.  With a device buffer, just copy `counts`, then `misses[0 .. n_misses)`, after the
//   stream's work (cudaMemcpyAsync + event).
//
// ---------------------------------------------------------------------------------------------------------------------
// EXPERT KERNELS: ACTIVATION AND h LAYOUT (internal, but the parity programs and the CPU-side checks need it)
// ---------------------------------------------------------------------------------------------------------------------
//   An activation vector is quantised per 32 values (CONTRACTS.md; bit-identical to the CPU's quantize_act()):
//   d = amax/127 (fp32), id = 127/amax, q = rint(x*id) clamped to [-127,127]; an all-zero block has d = 0.
//   The int8 values of one 32-block are stored in the "interleaved" order that the dp4a kernel wants:
//       byte position of natural element j (0..31) = 8*((j&15)>>2) + 2*(j&3) + (j>>4)
//   i.e. 8-byte groups [x[4k], x[4k+16], x[4k+1], x[4k+17], x[4k+2], x[4k+18], x[4k+3], x[4k+19]].
//   xq: int8 [T][160][32] (x), xs: fp32 [T][160]; hq: int8 [6T][72][32] (h, indexed by HIT index), hs: fp32 [6T][72].
//   (act_perm_pos() below is the single definition of the order.)
//
// ---------------------------------------------------------------------------------------------------------------------
// A DECODE LAYER (one stream; T = 1..8), and what the caller guarantees
// ---------------------------------------------------------------------------------------------------------------------
//   router_forward(x, wg, bias, T, logits_ws, ids, weights, s);
//   split_hits_misses(ids, weights, T, residency, layer, hits, misses, groups, counts, host_visible, s);
//   ... the host CPU pool reads `counts` and `misses` (copy them out, or map them) and computes the misses ...
//   experts_hits(cache_base, x, hits, groups, counts, T, scratch, parts, s);     // parts rows of the misses stay untouched
//   ... out[t] = shared expert + sum over k of parts[t*6+k] (hits) + the CPU's partials (misses), FP32 ...
//   * every pointer is a device pointer (the miss list and counts may be mapped host memory); a null stream is the default stream;
//   * alignment: x, wg, logits, xq, xs, hq, hs, parts: 16 bytes; cache_base: 256 bytes (kBlobBytes is a multiple of 256, so every slot is);
//   * the groups given to experts_gate_up / experts_down must be the ones split_hits_misses produced (sorted by decreasing size);
//   * the cache slots must be complete and ordered before the experts kernels on the stream (fills on the same stream, or an event);
//   * the blobs are the GGUF layout of CONTRACTS.md: [gate][up][down], MXFP4 blocks of 17 bytes, rows 2720 / 1224 bytes;
//   * experts_* work for T = 1..8 only; router_* and split_hits_misses (without groups) for any T (prefill).
// Errors (bad T, a failed launch) throw std::invalid_argument / std::runtime_error.
//
// Everything here is plain C++ (void* streams) so that host code needs no CUDA headers.
#pragma once

#include "strata/ds41/geometry.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::ds41::cuda {

inline constexpr int kMaxExpertTokens = 8;               // expert kernels: T = 1..8 (decode / a verify window)
inline constexpr int kActBlocks = kHidden / kQK;         // 160 blocks of 32 in x
inline constexpr int kHBlocks = kFF / kQK;               // 72 blocks of 32 in h
inline constexpr int kRouterSmallT = 32;                 // router: T <= 32 -> GEMV kernel, above -> tiled FP32 GEMM

/// Byte position of natural element j (0..31) of a 32-block inside the kernel's int8 layout (see above).
constexpr int act_perm_pos(int j) { return ((j & 15) >> 2) * 8 + (j & 3) * 2 + (j >> 4); }

struct HitEntry {
    int32_t token;
    int32_t k;
    int32_t slot;
    float weight;
};
struct MissEntry {
    int32_t token;
    int32_t k;
    int32_t expert;
    float weight;
};
struct SplitCounts {
    int32_t n_hits;
    int32_t n_misses;
    int32_t n_groups;
    int32_t reserved;
};
struct HitGroup {
    int32_t slot;
    int32_t n;
    int32_t hit[kMaxExpertTokens];
};
static_assert(sizeof(HitEntry) == 16 && sizeof(MissEntry) == 16 && sizeof(SplitCounts) == 16 && sizeof(HitGroup) == 40);

// ---- router -------------------------------------------------------------------------------------------------------
/// Bytes of the logits workspace for T tokens (fp32 [T][384]).
inline constexpr size_t router_workspace_bytes(int T) { return (size_t) T * kExperts * sizeof(float); }

/// logits[t][e] = sum_k x[t][k] * bf16_to_f32(wg[e][k]), FP32 accumulation in a fixed order (deterministic run to run).
/// x [T][5120] fp32, wg [384][5120] raw bf16 bits, logits [T][384].
void router_logits(const float* x, const uint16_t* wg_bf16, int T, float* logits, void* stream);

/// ids/weights from logits (see the header).  NaN logits are treated as -inf for selection.
void router_select(const float* logits, const float* bias, int T, int32_t* ids, float* weights, void* stream);

/// router_logits + router_select.  `logits_ws` = router_workspace_bytes(T) of device memory (kept: the parity program
/// reads it to report near-ties).
void router_forward(const float* x, const uint16_t* wg_bf16, const float* bias, int T, float* logits_ws, int32_t* ids,
                    float* weights, void* stream);

// ---- hit / miss split ---------------------------------------------------------------------------------------------
/// ids int32 [T][6], weights fp32 [T][6] (router output), residency_table int32 [40][384] (slot or -1; device),
/// `layer` 0..39.  Fills hits, misses, counts and (only when `groups` != nullptr, requires T <= 8) groups.
/// An id outside 0..383 is counted as a miss (and passed on unchanged).
void split_hits_misses(const int32_t* ids, const float* weights, int T, const int32_t* residency_table, int layer,
                       HitEntry* hits, MissEntry* misses, HitGroup* groups, SplitCounts* counts, bool host_visible,
                       void* stream);

// ---- hit experts --------------------------------------------------------------------------------------------------
struct ExpertScratch {
    int8_t* xq;   // [T][160][32]
    float* xs;    // [T][160]
    int8_t* hq;   // [6T][72][32]
    float* hs;    // [6T][72]
};
/// Device bytes of scratch for T tokens (T = 1..8), and the carving of one allocation (256-byte aligned parts).
size_t expert_scratch_bytes(int T);
ExpertScratch expert_scratch_carve(void* device_base, int T);

/// x fp32 [T][5120] -> xq, xs (T = 1..8; any T works, the other expert kernels need T <= 8).
void quantize_acts(const float* x, int T, int8_t* xq, float* xs, void* stream);

/// Optional: sets the shared-memory attributes of all expert kernel specialisations (what the first launch does lazily), so that nothing
/// but kernel launches happens inside a CUDA graph capture.  Idempotent.
void experts_prepare();

/// Phase 1.  For every group in `groups` (device count in counts->n_groups): h = silu(min(W1.x,10)) * clamp(W3.x,+-10) * w
/// for every token of the group, quantised into hq/hs at the HIT index.  `cache_base`: slot s is the blob at
/// cache_base + s * kBlobBytes (256-byte aligned).  `T` only sizes the grid and picks the specialisation
/// (the maximum number of tokens per group).
void experts_gate_up(const uint8_t* cache_base, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts,
                     int T, const int8_t* xq, const float* xs, int8_t* hq, float* hs, void* stream);

/// Phase 2.  parts[(token*6 + k)][5120] = W2 . h for every hit; miss entries of `parts` are not touched.
void experts_down(const uint8_t* cache_base, const HitEntry* hits, const HitGroup* groups, const SplitCounts* counts,
                  int T, const int8_t* hq, const float* hs, float* parts, void* stream);

/// quantize_acts + experts_gate_up + experts_down.
void experts_hits(const uint8_t* cache_base, const float* x, const HitEntry* hits, const HitGroup* groups,
                  const SplitCounts* counts, int T, const ExpertScratch& scratch, float* parts, void* stream);

// ---- test hooks ---------------------------------------------------------------------------------------------------
/// What the compiler made of one expert kernel specialisation (NT = 1, 2, 4 or 8 tokens per group): registers per thread, static and
/// dynamic shared memory per block and how many blocks the SM can hold.  Zeros in the CPU emulation.
struct ExpertKernelInfo {
    int regs = 0;
    int static_smem = 0;
    int dyn_smem = 0;
    int blocks_per_sm = 0;
};
ExpertKernelInfo expert_kernel_info(int NT, bool down);

/// out[e] = the kernels' decode of E8M0 byte e (0..255), as the float 2^(e-128) the way ggml_e8m0_to_fp32_half has it.
void test_e8m0_table(float* out256, void* stream);

}  // namespace strata::ds41::cuda
