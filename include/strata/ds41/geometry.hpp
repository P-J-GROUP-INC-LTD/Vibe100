// include/strata/ds41/geometry.hpp - DeepSeek-V4.1-Flash: the shapes and byte layouts every DS41 work package shares.
//
// Read from the target GGUF's own metadata and tensor table (mxxm-t/DeepSeek-V4.1-Flash-GGUF, MXFP4; see
// docs/deepseek/RESEARCH.md section 10) and from the official reference (third_party/deepseek-v41-flash-reference).
// As in the Qwen path, the geometry is a compile-time contract: a loader that finds anything else REFUSES rather than
// mis-indexes.  The semantics behind these numbers are docs/deepseek/CONTRACTS.md; change both together.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::ds41 {

// ---- model ----
inline constexpr int kLayers = 40;
inline constexpr int kHidden = 5120;            // n_embd
inline constexpr int kExperts = 384;            // routed experts per layer
inline constexpr int kTopK = 6;                 // routed experts per token
inline constexpr int kFF = 2304;                // expert (and shared expert) intermediate width
inline constexpr float kRouteScale = 1.5f;      // routed_scaling_factor
inline constexpr float kSwigluLimit = 10.0f;    // up clamped to [-10, 10], gate to (-inf, 10]
inline constexpr float kRouteNormEps = 1e-20f;  // weights /= sum + 1e-20 (matches training, not norm_eps)

// ---- GGML MXFP4 (ggml-common.h block_mxfp4) ----
// 32 values per block: one E8M0 scale byte, then 16 bytes of e2m1 codes; value j (0..15) is the LOW nibble of qs[j],
// value j + 16 the HIGH nibble.  value = kvalues_fp4[code] * 2^(e - 128), where kvalues_fp4 =
// {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12} is the e2m1 table doubled (ggml: GGML_E8M0_TO_FP32_HALF).
inline constexpr int kQK = 32;
inline constexpr int kBlockBytes = 17;
struct BlockMxfp4 {
    uint8_t e;
    uint8_t qs[kQK / 2];
};
static_assert(sizeof(BlockMxfp4) == kBlockBytes, "block_mxfp4 is 17 bytes, unpadded");

// ---- one routed expert as the GGUF stores it, and as a GPU cache slot holds it ----
// ffn_gate_exps / ffn_up_exps: [ne0 = 5120, ne1 = 2304, ne2 = 384] -> expert e is 2304 rows of 160 blocks;
// ffn_down_exps: [ne0 = 2304, ne1 = 5120, ne2 = 384] -> expert e is 5120 rows of 72 blocks.
inline constexpr int kGateRowBlocks = kHidden / kQK;                                  // 160
inline constexpr int kDownRowBlocks = kFF / kQK;                                      // 72
inline constexpr size_t kGateRowBytes = (size_t) kGateRowBlocks * kBlockBytes;        // 2,720
inline constexpr size_t kDownRowBytes = (size_t) kDownRowBlocks * kBlockBytes;        // 1,224
inline constexpr size_t kGateBytes = (size_t) kFF * kGateRowBytes;                    // 6,266,880 (gate; up the same)
inline constexpr size_t kDownBytes = (size_t) kHidden * kDownRowBytes;                // 6,266,880
// GPU blob: [gate][up][down], exactly the three GGUF slices of the expert concatenated
inline constexpr size_t kBlobGate = 0;
inline constexpr size_t kBlobUp = kGateBytes;
inline constexpr size_t kBlobDown = 2 * kGateBytes;
inline constexpr size_t kBlobBytes = 2 * kGateBytes + kDownBytes;                     // 18,800,640
static_assert(kBlobBytes == 18800640, "one MXFP4 expert of DeepSeek-V4.1-Flash");

// ---- one routed expert split across the two CPU sockets (tensor-parallel, docs/deepseek/PLAN.md section 2) ----
// Half h (0 or 1): gate/up rows [h * 1152, h * 1152 + 1152) and, of every down row, the blocks [h * 36, h * 36 + 36)
// (the matching intermediate columns).  Each half computes its 1152 intermediate values and a partial 5120-vector;
// the expert's output is the sum of the two halves' partials.  Layout of a half: [gate rows][up rows][down rows],
// row-major, blocks contiguous within a row.
inline constexpr int kHalves = 2;
inline constexpr int kHalfFF = kFF / kHalves;                                         // 1,152 = 36 blocks
inline constexpr int kHalfDownRowBlocks = kDownRowBlocks / kHalves;                   // 36
inline constexpr size_t kHalfGateBytes = (size_t) kHalfFF * kGateRowBytes;            // 3,133,440 (gate; up the same)
inline constexpr size_t kHalfDownBytes = (size_t) kHidden * kHalfDownRowBlocks * kBlockBytes;   // 3,133,440
inline constexpr size_t kHalfGate = 0;
inline constexpr size_t kHalfUp = kHalfGateBytes;
inline constexpr size_t kHalfDown = 2 * kHalfGateBytes;
inline constexpr size_t kHalfBytes = 2 * kHalfGateBytes + kHalfDownBytes;            // 9,400,320
static_assert(2 * kHalfBytes == kBlobBytes, "the two halves hold exactly one expert");
static_assert(kFF % (kHalves * kQK) == 0, "the split falls on MXFP4 block boundaries");

}  // namespace strata::ds41
