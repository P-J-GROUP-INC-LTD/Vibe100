# Vibe100 — DeepSeek-V4.1-Flash port plan (V100 + big-RAM host, hot experts on the GPU)

Spec: `docs/deepseek/RESEARCH.md`. Ground truth: the official reference, vendored in
`third_party/deepseek-v41-flash-reference/` (MIT). Prerequisite: the Volta port (`docs/volta/PLAN.md`).

## 1. Corrections to the original brief

The brief planned for a "DeepSeek-family" model (MLA, DSA, no SSD table). The released V4.1-Flash is different:

| Brief assumed | V4.1-Flash actually has | Consequence |
|---|---|---|
| MLA (compressed latent KV) | **MQA**: 64 q-heads × 512, one KV head, K == V, partial RoPE, attention sink, grouped output projection | Simpler than MLA. No latent up-projection. |
| DSA indexer | **CSA2**: per-layer modes (SWA-only / Full / Reuse / Reindex), ratio-2 gated compressor (encoder), ratio-1 projection of the last encoder state (decoder), 32-head FP4 indexer, top-512 + 128-token window, candidate pool past 16K | ≤ 640 keys per query at any context. Only 8 of 40 layers run an indexer; only 4 produce global KV. |
| "SSD lookup table is Qwen-specific, remove it" | **Engram**: 203 GB hashed n-gram table, 24 rows × 2 layers per token, deterministic addresses | **Reuse** Strata's PLE/n-gram SSD reader pattern instead of removing it. |
| Standard residual | **mHC**: 4 residual copies, 24×20480 fp32 mixing per sub-layer, 20-step Sinkhorn | Small fused fp32 kernel. |
| MTP head | **DSpark**: 3 extra blocks, block-size-5 drafts, Markov + confidence heads | Optional; small measured gain on offload rigs. Last milestone. |
| Experts "hundreds per layer", Q4 | 384 per layer, 6 active + 1 shared, **MXFP4 = 18.8 MB per expert** (lossless GGML-compatible) | ~1,000 experts fit a 20 GB cache (≈ 26/layer, 6.9 %). |
| llama.cpp has a reference | llama.cpp **cannot run V4.1 correctly** yet (PR #28696 open, no sparse attention) | Reference = the official PyTorch code, run on CPU on small shapes and per layer on real weights. |

The brief's principles stand: correctness before speed, change one variable at a time, gates.

**Decided (2026-10-01): the routed experts run in MXFP4** — the format DeepSeek released them in, so the experts are
used bit for bit as trained, with no re-quantisation step and no quality question to measure. Every expert kernel
(CPU and GPU), the pack layout and the cache sizing target exactly one format: per expert, `w1`/`w3` as 2304 rows of
2560 packed e2m1 bytes + 160 E8M0 scales, `w2` as 5120 rows of 1152 bytes + 72 scales — 18,800,640 bytes, identical
to GGML's MXFP4 blocks (17 bytes per 32 values), so an MXFP4 GGUF of the experts maps onto the same kernels.
Q4_K / Q2_K / i-quant experts are out of scope.

## 2. Memory map on one V100 32 GB + host RAM

| Where | What | Size |
|---|---|---|
| GPU | attention (FP8 bytes, dequantised in-kernel — Volta has no FP8), shared experts, gates, mHC, indexers/compressors, Engram projections, head | ≈ 8.5 GB (head could drop to 0.66 GB at fp8/Q8) |
| GPU | KV: per-layer 128-token SWA ring + global KV of 4 layers + index-K, fp16 (no quantised KV — see §1 and issue #25382 notes) | ≈ 3.2 KB/token → 0.4 GB at 128K |
| GPU | activations, prompt buffers (borrowed from cache slots, as Strata does) | 1-2 GB |
| GPU | **hot-expert cache** (MXFP4 blobs, residency table, adaptive swaps) | the rest: ~18-20 GB ≈ 1,000 experts |
| RAM | all 15,360 routed experts, MXFP4, pinned where the driver allows | 288.8 GB |
| RAM / SSD | Engram tables (mmapped in place from the shards; rows prefetched from token ids) | 202.8 GB |
| CPU | computes missed experts in place (AVX-512/AVX2 MXFP4 kernels), overlapped with the GPU | |

With less RAM than ~520 GB, the routed experts use Strata 0.1.31's RAM-budget + file tier with routing prefetch
(`--resident-budget-gib`) and Engram stays on SSD. MXFP4 fixes the expert footprint at 288.8 GB, so RAM below
~300 GB means some experts are read from the SSD on a miss. **Needed from you: RAM size, CPU model (AVX-512?
VNNI?), SSD, and whether the weights come from the official safetensors or an MXFP4 GGUF (which one).**

Decode budget [estimate, from RESEARCH §6-7]: GPU reads ≈ 7.2 GB of dense weights + 240·h × 18.8 MB of hit
experts per token (≈ 10-11 ms at 900 GB/s); CPU reads 240·(1-h) × 18.8 MB (h = 0.5 → 2.26 GB → 16.5 ms at
137 GB/s). Overlapped, the ceiling is ~50 tok/s at h = 0.5; expect 40-60 % of it before speculation. That is the
case for the port versus the 4-5 tok/s measured today; it is not a prediction.

## 3. Architecture of the port

A second model path, `ds41`, beside the Qwen one — same philosophy as upstream (geometry fixed at compile time by
the artifact, refuse rather than mis-index), sharing the model-agnostic infrastructure:

| Reused as is | Reused as a pattern (new DS41 code) | New |
|---|---|---|
| `strata_core` (device, pinned, graph, arenas), `ExpertCache` (slots, residency, profiles, adaptive swaps), CPU thread pool, `platform/direct_file`, sampler kernels, server protocol (`serve/`), conversation cache | expert source/pool contract (`expert_source.hpp`), PLE reader → Engram reader, prefill slot borrowing, MTP verify loop → DSpark | MXFP4 expert kernels (CPU + GPU), FP8-block GEMV/GEMM (Volta: dequant to fp16), MQA sparse attention, compressor, indexer + top-k + candidate pool, mHC, Engram hashing + combine, sqrtsoftplus router, DS41 session loop, safetensors → pack converter, DeepSeek tokenizer |

## 4. Milestones and gates

**DS-0 — foundations, testable without the weights or a GPU (starts now).** Sonnet agents, disjoint files:

| WP | Files | Deliverable | Verified here by |
|---|---|---|---|
| DS-A oracle | `ref/ds41/` | numpy reference of every op (router, expert with clamps, MXFP4/FP8 dequant, RMSNorm 1e-20, partial RoPE + YaRN + inverse, MQA with sink over window + selection, compressor r1/r2, indexer + top-k + candidate pool, mHC incl. single-pass lag, Engram normaliser/hash/combine, block forward) | unit tests against the vendored **official torch code** on small random shapes (CPU torch); Engram primes sum to `engram_num_embeddings` |
| DS-B pack | `tools/ds41/` | stdlib safetensors reader; pack layout (expert blobs w1·w3·w2 + scales, aligned; dense tensors as stored; Engram mapped in place via an index); converter with `--dry-run` that plans the whole pack from `tensors.json.gz`; validator | dry-run on the vendored inventory; byte totals = 510,286,023,000; round-trip on a synthetic mini-checkpoint |
| DS-C CPU experts | `src/ds41/cpu/`, `include/strata/ds41/` | MXFP4 expert kernel (gate/up 5120→2304, clamp-SwiGLU, down) AVX-512 / AVX2 / scalar, single- and multi-token, Q8 activations | parity vs the numpy oracle and a scalar fp64 path; GB/s microbenchmark (this CPU; re-run on the Xeon) |
| DS-D GPU hot experts | `src/ds41/cuda/`, `include/strata/ds41/` | sm_70 MXFP4 grouped GEMV for cached experts (dp4a + int8 LUT), sqrtsoftplus/noaux router (top-6, renorm, ×1.5), hit/miss split on `ExpertCache` residency | `compile_one.py` (no spills, no traps); GPU parity program for the V100 |

**DS-1 — one correct token.** FP8-block GEMV (dequant in registers), MQA attention decode (window + selected
compressed KV, sink), compressor/indexer (exact top-k by brute force first), mHC, Engram (RAM-resident first),
DS41 session loop for decode, layer-by-layer comparison with the oracle on real weights (user's box).
**Gate DS-1:** per-layer max relative error within thresholds set from fp16-vs-fp32 noise; top-1 ≥ 99 % vs the
reference on 500 tokens; no NaN/inf (fp16 range of V4.1 activations is UNVERIFIED — guard every fp16 conversion
like the Volta GEMM does).

**DS-2 — usable.** Prefill (FP8 → fp16 dequant + cuBLAS HMMA, chunked; encoder-only prefill is an approximation —
start with all 40 layers), Engram on SSD with token-driven prefetch, expert cache sized to VRAM, adaptive swaps,
server integration. **Gate DS-2 (= the brief's Gate 3): measured hit rate**; below ~40 % the cache does not beat
plain `--n-cpu-moe` style offload by much — stop and reassess.

**DS-3 — fast.** Routing traces → profiles per workload (coding vs general sets overlap little), NUMA placement of
the expert arena, DSpark drafting (CPU-resident first), PCIe Gen3-aware miss policy (compute misses on the CPU;
never stream them on a miss).

## 5. Honest scope

DS-0 is days; DS-1 and DS-2 are weeks of CUDA work (the brief's estimate holds). Nothing here has touched the real
weights or a V100. DeepSeek tuning only makes sense after the Volta port passes its Gate 1 on your card.
