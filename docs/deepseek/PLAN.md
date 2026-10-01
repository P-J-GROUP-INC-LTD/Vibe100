# Vibe100 — DeepSeek-V4.1-Flash port plan (V100 + big-RAM host, hot experts on the GPU)

Spec: `docs/deepseek/RESEARCH.md`. Ground truth: the official reference, vendored in
`third_party/deepseek-v41-flash-reference/` (MIT). Prerequisite for the engine (DS-1 on): Gate 1 of the Volta port
(`docs/volta/PLAN.md`).

## Status (2026-10-01)

| Package | State |
|---|---|
| DS-0 foundations: shared geometry header, contracts, per-package CMake hook | done (`d9721e8`) |
| DS-A NumPy oracle, validated against the official reference | done (`d37b3d5`) |
| DS-B GGUF tooling (`tools/ds41/`): manifest / validator, expert layout, memory plan, mini-GGUF fixture | done (`7cda903`) |
| DS-C MXFP4 CPU expert kernels (scalar, AVX2, AVX-512 VNNI) | done (`86ad959`) |
| DS-D V100 router, hit/miss split, MXFP4 hot-expert kernels (dp4a) | done (`7f8158b`); parity programs written, **not yet run on a V100** |
| the 2026-10-01 audit round over DS-0 to DS-D | done; its decisions are in `CONTRACTS.md` |
| **DS-1 the engine: one correct token** (section 4) | **next**; gated by Gate 1 of the Volta port |
| usage ledger (`docs/USAGE_LEDGER.md`) | later (DS-2); a design only |

Everything up to DS-D is testable without the weights or a V100 and has been tested that way. Nothing has touched the real weights
or a GPU.

## 1. Corrections to the original brief

The brief planned for a "DeepSeek-family" model (MLA, DSA, no SSD table). The released V4.1-Flash is different:

| Brief assumed | V4.1-Flash actually has | Consequence |
|---|---|---|
| MLA (compressed latent KV) | **MQA**: 64 q-heads × 512, one KV head, K == V, partial RoPE, attention sink, grouped output projection | Simpler than MLA. No latent up-projection. |
| DSA indexer | **CSA2**: per-layer modes (SWA-only / Full / Reuse / Reindex), ratio-2 gated compressor (encoder), ratio-1 projection of the last encoder state (decoder), 32-head FP4 indexer, top-512 + 128-token window, candidate pool past 16K | ≤ 640 keys per query at any context. Only 8 of 40 layers run an indexer; only 4 produce global KV. |
| "SSD lookup table is Qwen-specific, remove it" | **Engram**: two hashed n-gram tables (202.8 GB as DeepSeek's FP8 rows; **97.28 GiB = 104,451,107,600 B** in the MXFP4 GGUF this port reads), 24 rows per layer per token, deterministic addresses | **Reuse** Strata's PLE/n-gram SSD reader pattern instead of removing it. |
| Standard residual | **mHC**: 4 residual copies, 24×20480 fp32 mixing per sub-layer, 20-step Sinkhorn | Small fused fp32 kernel. |
| MTP head | **DSpark**: 3 extra blocks, block-size-5 drafts, Markov + confidence heads | Optional; small measured gain on offload rigs. Last milestone. |
| Experts "hundreds per layer", Q4 | 384 per layer, 6 active + 1 shared, **MXFP4 = 18.8 MB per expert** (lossless GGML-compatible) | ~1,000 experts fit a 20 GB cache (≈ 26/layer, 6.9 %). |
| llama.cpp has a reference | llama.cpp **cannot run V4.1 correctly** yet (PR #28696 open, no sparse attention) | Reference = the official PyTorch code, run on CPU on small shapes and per layer on real weights. |

The brief's principles stand: correctness before speed, change one variable at a time, gates.

**Decided (2026-10-01): the routed experts run in MXFP4** — the format DeepSeek released them in, so the experts are
used bit for bit as trained, with no re-quantisation step and no quality question to measure. Every expert kernel
(CPU and GPU), the pack layout and the cache sizing target exactly one format: per expert, `w1`/`w3` as 2304 rows of
2560 packed e2m1 bytes + 160 E8M0 scales, `w2` as 5120 rows of 1152 bytes + 72 scales — 18,800,640 bytes. The
kernels use GGML's block layout (17 bytes per 32 values, scale first, element j / j+16 in byte j), which the MXFP4
GGUF already has; DeepSeek's safetensors hold the same codes and scales in another byte order.
Q4_K / Q2_K / i-quant experts are out of scope.

## 2. Memory map on one V100 32 GB + host RAM

| Where | What | Size |
|---|---|---|
| GPU | attention (FP8 bytes, dequantised in-kernel — Volta has no FP8), shared experts, gates, mHC, indexers/compressors, Engram projections, head | ≈ 8.5 GB (head could drop to 0.66 GB at fp8/Q8) |
| GPU | KV: per-layer 128-token SWA ring + global KV of 4 layers + index-K, fp16 (no quantised KV — see §1 and issue #25382 notes) | ≈ 3.2 KB/token → 0.4 GB at 128K |
| GPU | activations, prompt buffers (borrowed from cache slots, as Strata does) | 1-2 GB |
| GPU | **hot-expert cache** (MXFP4 blobs, residency table, adaptive swaps) | the rest: ~18-20 GB ≈ 1,000 experts |
| RAM | all 15,360 routed experts, MXFP4, resident, row-split over the two NUMA nodes; only a small staging pool is page-locked (for GPU cache fills), not the whole arena | 288.8 GB (268.95 GiB) |
| RAM / SSD | Engram tables (mmapped in place from the GGUF shards; rows prefetched from token ids) | 97.28 GiB (104.5 GB; 202.8 GB in the official FP8 form) |
| CPU | computes missed experts in place (AVX-512/AVX2 MXFP4 kernels), overlapped with the GPU | |

**The target box (from the user, 2026-10-01):** dual Xeon Cascade Lake, 384 GB DDR4-2666 in 24 slots (6 channels
per socket on Cascade Lake-SP → 12 channels, ≈ 256 GB/s theoretical, ~200 GB/s if both sockets read locally — to
be measured), one V100 32 GB, weights from an **MXFP4 GGUF**. Consequences:

- **RAM: the experts (268.95 GiB) and Engram (97.28 GiB) together are 366 GiB of 377 usable.** Experts resident
  in RAM (split across the two NUMA nodes, see below); Engram stays mapped from the SSD and ~96 GiB of page cache
  holds ~98 % of it (the RAM budget table below). Per token Engram reads 48 rows of 136 B (MXFP4 in this GGUF);
  a miss costs a 4 KB page read (~2,400 IOPS at 50 tok/s — easy for NVMe). Layer 1's rows are needed right after
  sampling, so the reads are issued the moment a token is known (and for the prompt, all at once). Only a small
  staging pool is page-locked for GPU fills, not the whole 269 GiB.
- **RAM budget** (24 × 16 GiB = 384 GiB, ~377 GiB usable; sizes from the GGUF's tensor table):

  | Item | GiB | Where |
  |---|---|---|
  | routed experts, MXFP4 | 268.95 | resident, row-split: 134.5 per node |
  | OS, services, server | ~6-10 | |
  | engine host buffers (staging for cache fills, activations, miss lists) | ~2-4 | GPU's node |
  | `token_embd` (BF16; rows looked up per token) | 1.23 | host |
  | dense weights (attention, shared experts, mHC, router, indexer, head) | 8.33 | **GPU**; the host copy is dropped after upload |
  | DSpark sidecar (optional, if drafting runs on the CPU) | 7.42 | |
  | **left for Engram's page cache** | **~85-95** | of 97.28 GiB of Engram tables |

  So nearly all of Engram stays in RAM (90-100 %), and the "spare" ~100 GB is exactly what holds it.
- **Mirroring** (a full copy of experts on each node, as the user ran Qwen Next with ~2× decode): its gain comes
  from every CPU read becoming node-local. The row-split above gets the same locality with ONE copy, so mirroring
  adds no bandwidth on top of it — and a full mirror needs 538 GiB, a partial mirror of the next-hottest experts
  would evict Engram to the SSD (layer 1's Engram rows are needed right after sampling, so that is on the critical
  path). Kept as a measured experiment, off by default: `mirror N GiB of the next-hottest experts, computed whole
  on each socket`, only worth it if the per-layer cross-socket reduction of the split measures expensive.
- **CPU kernels: AVX-512 + VNNI (`vpdpbusd`), no VBMI** (Cascade Lake). MXFP4 needs a nibble → int8 lookup
  (`vpshufb` on 4-bit indices), which AVX-512BW has; AVX2 fallback for other machines.
- **NUMA and the GPU's socket, from day one** (Dell Precision 7920, 2x Xeon Gold 6226: 12 cores each, 6 channels
  each; the user measures **90-120 GB/s**, under half of the ~256 GB/s theoretical — the signature of reads
  crossing the UPI link, which carries roughly 40 GB/s per direction on this platform; confirm with Intel MLC
  `mlc --bandwidth_matrix`, which prints local vs remote bandwidth per node pair).

  Who moves how many bytes per token decides the layout:

  | Traffic | Bytes per token (h = 0.5) | Bound |
  |---|---|---|
  | CPU cores reading missed experts | **~2.3 GB** | DRAM / UPI |
  | GPU cache promotions (adaptive swaps) | rate-limited, ~0.2 GB/s | PCIe Gen3 (~12 GB/s) |
  | activations / expert results CPU ↔ GPU | 40 layers × ~20 KB | latency |
  | Engram rows | 48 × 136 B (+ page reads on a miss) | SSD / page cache |

  So the cross-socket traffic that matters is the CPU's own expert reads, not the GPU's. The layout:
  1. **Detect the GPU's node** at start (`/sys/bus/pci/devices/<bdf>/numa_node` from `cudaDeviceGetPCIBusId`;
     `nvidia-smi topo -m` shows the same) and the NVMe drive's (`/sys/block/nvme*/device/numa_node`).
  2. **GPU-node duties:** the host thread that drives the GPU, the page-locked staging pool for cache fills, the
     mapped doorbell/activation buffers, and the PCIe-fed queue of promotion candidates live on the GPU's socket;
     Engram I/O threads on the drive's socket.
  3. **Every expert is split across both sockets the way tensor-parallel MLPs are**: socket A holds rows
     0-1151 of w1/w3 and the matching columns 0-1151 of w2, socket B the other halves (1152 = 36 MXFP4 blocks,
     so the split falls on block boundaries). Each socket computes its half of the intermediate from its own
     gate/up rows and multiplies it by its own w2 columns into a partial 5120-vector; the two partials are added
     once per layer, together with the other experts' (an add the layer needs anyway to hand its result to the
     GPU). No mid-expert exchange, no UPI traffic for compute, and exactly 50/50 work on every token whatever
     experts it routes to.
  4. **"Secondary hot" experts next to the GPU** (the user's idea): putting the next-hottest experts *only* on the
     GPU's socket would send most CPU misses to that one socket — half the box's bandwidth doing most of the work
     while the other socket idles — so it loses to (3) for compute. Where GPU-locality does pay is the bytes that
     go to the GPU: promotions into VRAM and any misses the GPU fetches itself (`--pcie-frac`). With (3) half of
     such a copy crosses UPI, at no more than PCIe Gen3's ~12 GB/s and only while it lasts. Option, off by
     default and measured before it is kept: a full extra copy of the top promotion candidates on the GPU's node,
     paid for in RAM that otherwise caches Engram.
  5. Baseline to beat: the OS-level `numactl --interleave=all` (no code change). Never BIOS Node Interleaving: it makes Linux
     see one node, which hides the two nodes this layout (and the Qwen mirror) needs; keep it Disabled (`docs/volta/VOLTA.md`, "BIOS").
- **Which GGUF (decided 2026-10-01):** `mxxm-t/DeepSeek-V4.1-Flash-GGUF` (MXFP4, 12 shards + a DSpark sidecar,
  ~411 GB; headers read and recorded in RESEARCH §10). DS-B reads GGUF (Strata already has a reader) as the primary input and the
  official safetensors as the reference; `tools/ds41/manifest.py` validates the files against the `deepseek41` contract and refuses
  the other published family (vcruz305: Q2_K to Q8_0, no MXFP4 experts) unless told to look anyway. **Still needed from you: ≥ 411 GB
  of free NVMe** (the Engram tables stay mapped from it).

Decode budget [estimate, from RESEARCH §6-7]: GPU reads ≈ 7.2 GB of dense weights + 240·h × 18.8 MB of hit
experts per token (≈ 10-11 ms at 900 GB/s); CPU reads 240·(1-h) × 18.8 MB (h = 0.5 → 2.26 GB → ~11 ms at
~200 GB/s NUMA-local on the target box, ~23 ms if half of it crosses UPI). Overlapped, the bandwidth ceiling is
~60-90 tok/s at h = 0.5; expect well under half of it in practice (kernel efficiency, synchronisation, attention,
Engram). That is the case for the port versus the 4-5 tok/s the box's owner reports measuring today with another engine (their figure; other people's V4.1 offload rigs report 5-8, RESEARCH §8); it is not a prediction.

## 3. Architecture of the port

A second model path, `ds41`, beside the Qwen one — same philosophy as upstream (geometry fixed at compile time by
the artifact, refuse rather than mis-index), sharing the model-agnostic infrastructure:

| Reused as is | Reused as a pattern (new DS41 code) | New |
|---|---|---|
| `strata_core` (device, pinned, graph, arenas), `ExpertCache` (slots, residency, profiles, adaptive swaps), CPU thread pool, `platform/direct_file`, sampler kernels, server protocol (`serve/`), conversation cache | expert source/pool contract (`expert_source.hpp`), PLE reader → Engram reader, prefill slot borrowing, MTP verify loop → DSpark | MXFP4 expert kernels (CPU + GPU), FP8-block GEMV/GEMM (Volta: dequant to fp16), MQA sparse attention, compressor, indexer + top-k + candidate pool, mHC, Engram hashing + combine, sqrtsoftplus router, DS41 session loop, safetensors → pack converter, DeepSeek tokenizer |

## 4. Milestones and gates

**DS-0 — foundations, testable without the weights or a GPU (DONE: DS-0 `d9721e8`, DS-A `d37b3d5`, DS-B `7cda903`, DS-C `86ad959`,
DS-D `7f8158b`; audited).** Sonnet agents, disjoint files. What was planned, and what was built where it differs:

| WP | Files | Deliverable | Verified here by |
|---|---|---|---|
| DS-A oracle | `ref/ds41/` | numpy reference of every op (router, expert with clamps, MXFP4/FP8 dequant, RMSNorm 1e-20, partial RoPE + YaRN + inverse, MQA with sink over window + selection, compressor r1/r2, indexer + top-k + candidate pool, mHC incl. single-pass lag, Engram normaliser/hash/combine, block forward) | unit tests against the vendored **official torch code** on small random shapes (CPU torch); Engram primes sum to `engram_num_embeddings` |
| DS-B GGUF tooling | `tools/ds41/` | **Built (the plan said "safetensors converter"; once the GGUF was chosen, the work became manifest tooling):** a validator / manifest writer for the mxxm-t GGUF (every tensor's shard, absolute offset, type, dims and bytes; the CSA2 layer maps and Engram constants from the metadata); expert extraction (the GPU blob `[gate][up][down]`, 18,800,640 B, and the two CPU halves, 9,400,320 B each); the memory plan of the target box; a tiny deterministic `deepseek41` GGUF for loader tests; NumPy MXFP4 / Q8_0 / BF16 codecs bit-identical to GGML's C code. No converter: the GGUF's layout is used as it is. | the saved headers of the real files (`third_party/deepseek-v41-flash-reference/gguf-headers-*.json.gz`), the mini GGUF, GGML's own `ggml-quants.c` and DS-C's `pack_cpu_half` as oracles; nothing has read real expert bytes yet |
| DS-C CPU experts | `src/ds41/cpu/`, `include/strata/ds41/` | MXFP4 expert kernel (gate/up 5120→2304, clamp-SwiGLU, down) AVX-512 / AVX2 / scalar, single- and multi-token, Q8 activations | parity vs the numpy oracle and a scalar fp64 path; GB/s microbenchmark (this CPU; re-run on the Xeon) |
| DS-D GPU hot experts | `src/ds41/cuda/`, `include/strata/ds41/` | sm_70 MXFP4 grouped GEMV for cached experts, one launch for all hit experts with device-side ids; sqrtsoftplus/noaux router (top-6, renorm, ×1.5); hit/miss split on `ExpertCache` residency. **Built: one kernel candidate, the dp4a + prmt-table decode (scales in fp32).** The plan promised two candidates benchmarked against each other; **the m8n8k4 tensor-core GEMV after 1Cat-vLLM's `mxfp4_qpn_m1_sm70` (`docs/volta/NINFER_STUDY.md`) was not built** - it is an experiment for after DS-1, if the V100 timings say the dp4a kernel is compute-bound rather than bandwidth-bound. | `compile_one.py` (no spills, no traps); the kernel source also runs on the CPU through an emulator against FP64 references; parity programs for the V100 (`ds41_router_parity`, `ds41_split_parity`, `ds41_expert_parity`, run by `tools/volta/run_parity.sh`) - not yet run on a GPU |

**DS-1 — one correct token.** FP8-block GEMV (dequant in registers), MQA attention decode (window + selected
compressed KV, sink), compressor/indexer (exact top-k by brute force first), mHC, Engram (RAM-resident first),
DS41 session loop for decode, layer-by-layer comparison with the oracle on real weights (user's box).
**Gate DS-1:** per-layer max relative error within thresholds set from fp16-vs-fp32 noise; top-1 ≥ 99 % vs the
reference on 500 tokens; no NaN/inf (fp16 range of V4.1 activations is UNVERIFIED — guard every fp16 conversion
like the Volta GEMM does).

**DS-2 — usable.** Prefill (dense Q8_0 / BF16 → fp16 dequant + cuBLAS HMMA, chunked, experts streamed: the prefill design below; encoder-only prefill is an approximation —
start with all 40 layers), Engram on SSD with token-driven prefetch, expert cache sized to VRAM, adaptive swaps,
server integration, the **usage ledger** (`docs/USAGE_LEDGER.md`: weeks of the user's own routing decide the GPU
core set; the short-term tier follows the current task). **Gate DS-2 (= the brief's Gate 3): measured hit rate**; below ~40 % the cache does not beat
plain `--n-cpu-moe` style offload by much — stop and reassess.

### DS-2 prefill design (decided 2026-10-01; estimates, not measurements)

Upstream Strata already has the two mechanisms this builds on, for Qwen: from 1,024-token chunks on, every
non-resident expert streams over PCIe through a ring of borrowed expert-cache slots, overlapped with the attention
(`src/prefill/prefill.cpp`, `stream_all_min`, `ring_slots`); and the server keeps conversation checkpoints, so a
request that starts with tokens the engine already holds does not read them again (`cached_tokens` in the API).

**Why PCIe sets the speed.** A prompt chunk of more than ~1,000 tokens routes nearly every expert of every layer
(6 of 384 per token: an expert is unused with probability (1 - 6/384)^T, ~e^-16 at T = 1,024). The ~1,150 experts the
V100 caches cover 7.5 % of the 15,360; the other ~267 GB must reach the GPU (or be computed by the CPU) once per
chunk. PCIe Gen3 x16 moves ~12 GB/s from pinned memory, so one pass costs ~22 s whatever the chunk's length: about
180 tokens/s for a 4K-token prompt, about 700 for a 16K chunk. The GPU's own expert compute at those sizes is small
next to that (MXFP4 at 900 GB/s), and the CPUs alone are compute-bound at large T (17 GFLOP of expert work per token).

**The levers, in order of value:**
1. **Prefix reuse, saved to NVMe.** Port the conversation checkpoints, and persist them: DeepSeek-V4.1's KV is small
   (one shared KV head, compressed 2:1 or 1:1, sliding-window layers), so a checkpoint of a long system prompt,
   document or code base costs little disk and turns its prefill into a read. For chat and agent use - where most of
   every prompt was seen before - this is the largest win, and it survives server restarts.
2. **Big chunks.** One expert pass serves the whole chunk, so lend most of the hot-expert cache to the staging ring and
   the activations during prefill (as upstream does) and aim for 16K-32K tokens per chunk; the dense weights (8.3 GB)
   stay resident.
3. **Split each layer's experts between PCIe and the CPUs.** While the copy engine streams, the two sockets compute the
   experts with the fewest tokens of the chunk (their cost grows with the token count; a transfer's does not). Pick
   the split per layer from the routing counts so both finish together: estimated 1.2-1.4x on 2K-16K prompts. Short
   prompts (< ~1,000 tokens) take the decode path's rule instead: hits on the GPU, misses on the CPU, nothing streamed.
4. **Stream in routing order, not file order**, starting with the layer's most-routed experts, so the GPU starts
   computing while the tail of the layer is still in flight; read each expert's two socket halves from both nodes'
   pinned memory at once (UPI has headroom over Gen3's 12 GB/s).
5. **Hardware:** a second V100 is a second x16 link: roughly twice the prefill, independently of everything above.

Gate for DS-2's prefill: measured tokens/s at 2K / 8K / 32K prompts against this model (PCIe-bound pass time per
chunk), and a checkpoint round trip (save, restart, reuse) that reproduces the logits of a fresh prefill bit for bit.

**DS-3 — fast.** Routing traces → profiles per workload (coding vs general sets overlap little), NUMA placement of
the expert arena, DSpark drafting (CPU-resident first), PCIe Gen3-aware miss policy (compute misses on the CPU;
never stream them on a miss).

## 5. Honest scope

DS-0 (DS-A to DS-D) took days and is done; DS-1 and DS-2 are weeks of CUDA work (the brief's estimate holds). Nothing here has
touched the real weights or a V100. The engine (DS-1 on) and DeepSeek tuning only make sense after the Volta port passes its Gate 1 on
your card; the weights' download, `tools/ds41/manifest.py`, the DS-D parity programs and the CPU benchmark can run on the box any
time (`docs/volta/RUNBOOK.md`, step 9).
