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
//   * the stored values enter exactly - int8 codes are exact in FP16 and their per-64 scales are applied in FP32 (to
//     the q.k partial of each 64-dim group, and folded into p for p.v); FP16 KV is used as is - with ONE exception,
//     the V side of the hybrid K8V4 pools: its q4_0 values (n - 8) * d (exact in FP32: a 4-bit integer times an 11-bit
//     significand) are rounded to FP16 when the kernel dequantizes them (as upstream's v1 kernel does), whereas the
//     FP32 fallback keeps them in FP32.  The relative error is at most 2^-11 per V element, so an output element
//     moves by at most 2^-11 * sum_c p_c |v_c| <= 2^-11 * max|v| (about 5e-4 of the V magnitude; nearly reached when
//     one cell dominates the softmax, ~1/sqrt(N) of it when N cells share it): up to 2-3 orders above the FP32-level
//     differences of the other modes.
//     K8V4's int8 K side is exact.  (Derivation: qsa_prompt_attn.cu above `Smem`.)  With q4_0 scales that happen to
//     make every product exact in FP16 (e.g. d = k / 1024, k <= 31) the rounding adds nothing;
//   * q and p are split into FP16 hi + lo parts (two MMAs each), so they keep ~22 bits: apart from K8V4's V above,
//     the result differs from the FP32 kernel by summation order and the exp2 rounding, not by an FP16 cast.
// Not bitwise equal to `qsa_decode_attn_batch`; `qsa_prompt_attn_parity` bounds the difference and the prompt
// quality gate (needles, teacher-forced top-1) checks it end to end. Q4_0 KV is not handled (returns false).
//
// Which kernel runs is decided per call from the CURRENT device's compute capability (major * 10 + minor) AND from
// the code this binary carries for that device.  The MMA kernels exist only in code built for a virtual arch >= sm_75
// (the v1 kernel) / >= sm_80 (cp.async); in code for an older arch they are `trap` stubs.  The driver runs, on a
// given card, the best code the binary has for it - a -DCMAKE_CUDA_ARCHITECTURES=70 build (sm_70 + compute_70) runs
// its sm_70 cubin on a cc 7.5 card and JIT-compiles the compute_70 PTX on cc 8.x, trap stubs both - so the decision
// asks the runtime which arch each kernel's code was compiled for (cudaFuncGetAttributes: ptxVersion), not just the
// card.  Where the binary's code is as new as the card (a build for 75 on Turing, for 80 / 86 / 89 on Ampere and Ada,
// ...; every normal build) the table is the upstream one:
//     cc >= 80          Ampere and newer: the cp.async kernel for int8 KV, the v1 kernel for fp16 and K8V4 KV
//     75 <= cc < 80     Turing: the v1 kernel
//     cc 70, 72         Volta: the WMMA kernel, unless STRATA_VOLTA_ATTN=0 (then false: the caller's FP32 fallback)
//     cc < 70           false
// and where it is not (the code the driver runs for a card >= 7.5 was built for an arch below 75, or for the
// cp.async kernel below 80) the card takes what its code has: the v1 kernel if the binary has sm_75 code for it,
// else the Volta WMMA kernel, which runs on any sm_70+ card - e.g. the 70-only engine of a layer split that pairs a V100
// with a newer card; there STRATA_VOLTA_ATTN=0 gives false, as on a Volta.  The code is only ever asked about on cc >=
// 7.5 (a cc 7.0 / 7.2 card has nothing but the Volta kernel), and a kernel is only taken away from the card on positive
// evidence that its code is a trap: if the runtime cannot say, the card's compute capability decides, as upstream.
// STRATA_VOLTA_ATTN (read once, case-insensitive): unset / empty / 1 / on / true / yes = the Volta kernel where the
// table above says so, 0 / off / false / no = off, 2 or "force" = the Volta kernel on any sm_70+ card (a test aid;
// nothing else changes without it); anything else warns once on stderr and means the default.
#pragma once

#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// Same arguments and output as `qsa_decode_attn_batch` minus the scratch; q [n_q, n_head, 256], ids [n_q, cap], steps
/// [n_q, kStepCount] (the selection width of query i is steps[i * kStepCount + kStepWidth], 0 allowed: that query's
/// output is zeros), attn [n_q, n_head, 256].  A page-table entry < 0 (a page KV streaming could not make resident)
/// masks its cells on the Volta kernel (score -inf, weight 0, as the FP32 kernel does); the v1 and cp.async kernels,
/// as upstream, read such a cell as a zero row with score 0.  Any n_q is fine (the launch is split at 65,535 queries).
///
/// Returns false, with nothing launched, when
///   * `qsa_prompt_attn_variant(pools, s)` is "fallback-fp32" - the pools are Q4_0, the geometry is not 24 heads / 2
///     KV heads / 256, the pools are incomplete or have no page table, the device is older than sm_70 or cannot be
///     queried, it is a Volta (or a card whose code here has no MMA kernels) with STRATA_VOLTA_ATTN=0, or the
///     chosen kernel's shared-memory opt-in is refused on this device - the caller then uses the old kernel; or
///   * `cap <= 0`, `ids` or `steps` is null: the three arguments the variant never sees.
/// Nothing else makes it return false: once a kernel is chosen it is launched (a CUDA error after that is fatal, as
/// in the old kernel).  `n_q <= 0` returns true without launching, whatever the variant is.
bool qsa_prompt_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* attn, int64_t n_q, void* stream);

/// Which implementation `qsa_prompt_attn_batch` would run for these pools on the current device: "volta-wmma",
/// "ampere-i8-cpasync", "mma-v1" (Turing; the fp16 and K8V4 modes everywhere), or "fallback-fp32" when it would return
/// false.  It is the decision `qsa_prompt_attn_batch` itself takes (the same code, the same per-device state, the
/// shared-memory opt-in included), so for the same pools, shapes and device it cannot disagree, except in the
/// arguments it does not receive: `cap <= 0` / null `ids` / `steps` make the batch call return false for a
/// kernel variant, and `n_q <= 0` returns true for any variant.  For tests and logs.
const char* qsa_prompt_attn_variant(const QsaAttnPools& pools, const QsaShapes& s);

}  // namespace strata::kernels
