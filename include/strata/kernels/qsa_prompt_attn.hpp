// include/strata/kernels/qsa_prompt_attn.hpp - perf-review D-1: the prompt path's QSA attention on tensor cores.
//
// `qsa_decode_attn_batch` serves a prompt one query at a time with the decode kernel: FP32 dot products with a warp
// reduction per head and cell, 64-cell chunks whose partial sums go through global scratch to a second (merge)
// kernel. At 32K that is 21% of the prompt (5.6 s of 26 s on a 5070, Q2_0, int8 KV), and it is limited by
// instruction issue, not memory (~150 GB/s of logical reads, 3.4 TFLOP/s).
//
// This kernel keeps the same per-query selection (no masking, no union) and reads the same pools, but:
//   * one block per (query, KV head) walks all its selected cells in chunks of 32 with an online softmax: no
//     split-K scratch and no merge kernel;
//   * q.k and p.v are FP16 MMAs with FP32 accumulation over the 12 query heads of the KV head (+4 pad rows):
//     m16n8k16 on Ampere and newer, m16n8k8 on Turing, and nvcuda::wmma m16n16k16 on Volta (sm_70/72, whose
//     tensor cores only do m8n8k4);
//   * the stored values enter exactly: int8 codes are exact in FP16 and their per-64 scales are applied in FP32
//     (to the q.k partial of each 64-dim group, and folded into p for p.v); FP16 KV is used as is;
//   * q and p are split into FP16 hi + lo parts (two MMAs each), so they keep ~22 bits: the result differs from
//     the FP32 kernel by summation order and the exp2 rounding, not by an FP16 cast.
// Not bitwise equal to `qsa_decode_attn_batch`; `qsa_prompt_attn_parity` bounds the difference and the prompt
// quality gate (needles, teacher-forced top-1) checks it end to end. Q4_0 KV is not handled (returns false).
//
// Which kernel runs is decided per call from the CURRENT device's compute capability (major * 10 + minor):
//     cc >= 80          Ampere and newer: the cp.async kernel for int8 KV, the v1 kernel for fp16 and K8V4 KV
//     75 <= cc < 80     Turing: the v1 kernel
//     cc 70, 72         Volta: the WMMA kernel, unless STRATA_VOLTA_ATTN=0 (then false: the caller's FP32 fallback)
//     cc < 70           false
// STRATA_VOLTA_ATTN (read once): unset / 1 = the Volta kernel on sm_70/72, 0 = off, 2 or "force" = the Volta kernel
// on any sm_70+ card (a test aid; nothing else changes without it).
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Same arguments and output as `qsa_decode_attn_batch` minus the scratch. Returns false (nothing launched) when the
/// pools are Q4_0, the geometry is not 24 heads / 2 KV heads / 256, the device is older than sm_70, or it is a Volta
/// with STRATA_VOLTA_ATTN=0: the caller then uses the old kernel.
bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream);

/// Which implementation `qsa_prompt_attn_batch` would run for these pools on the current device (the same decision
/// code, so it cannot disagree): "volta-wmma", "ampere-i8-cpasync", "mma-v1" (Turing; the fp16 and K8V4 modes
/// everywhere), or "fallback-fp32" when it would return false. For tests and logs.
const char* qsa_prompt_attn_variant(const QsaAttnPools& pools, const QsaShapes& s);

}  // namespace strata::kernels
