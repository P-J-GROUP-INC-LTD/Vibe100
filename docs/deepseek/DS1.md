# DS-1 — one correct token: the contract between the DS-1 work packages

Goal: `strata-ds41` reads the mxxm-t GGUF and decodes DeepSeek-V4.1-Flash token by token on the V100 + the two Xeons, with logits
that match the NumPy oracle (`ref/ds41`) layer by layer. Speed is DS-2/DS-3; DS-1 is correctness with the right structure
(hot experts on the GPU, misses on the CPU, per-socket halves). Prefill in DS-1 = the prompt fed one token at a time through the
decode path (equal to the oracle's prefill by construction: CONTRACTS.md, index-K decision).

Read first: `docs/deepseek/RESEARCH.md` (§1-4, §10), `docs/deepseek/CONTRACTS.md` (expert math, quantiser, split protocol),
`include/strata/ds41/geom.hpp` (geometry policy), the oracle (`ref/ds41/model.py` `Model.block` / `forward`, `attention.py`, `mhc.py`,
`engram.py`, `moe.py`, `ops.py`, `rope.py`, `quant.py`), and DS-D's kernels as the style to follow (`src/ds41/cuda/*`).

## 1. How the code is built and tested here (no GPU in the cloud)

- **Geometry policy.** Every kernel and host module is a template over `G` (`RealGeom`, `MiniGeom` in `geom.hpp`). The engine binary
  instantiates `RealGeom`; tests instantiate `MiniGeom`. Shapes come from `G` and `Derived<G>`, never from literals.
- **Kernels are written once, compiled twice** (DS-D's pattern, `src/ds41/cuda/ds41_dev.cuh`): by nvcc for sm_70, and by the host
  compiler with `-DDS41_EMU` against the emulator (`ds41_emu.hpp`), which runs every GPU thread on the CPU (shared memory poisoned per
  block, block/fiber order forward / reverse / shuffled, alignment asserts). Kernel bodies use only `ds41_dev.cuh` names for intrinsics.
  Device memory and launches go through DS-D's `Dev` interface (`ds41_parity_lib.hpp`: `CudaDev` / the emulator's), so host code runs
  unchanged in both builds.
- **The end-to-end test**: `tools/ds41/make_mini_gguf.py` writes a tiny GGUF with the real tensor names, ggml types and metadata keys at
  `MiniGeom`'s shape (8 layers covering every layer role, 2 Engram layers, MXFP4 experts). The oracle loads it
  (`ref/ds41/model.py: model_from_gguf`); the engine (MiniGeom, emulated) loads the same file; per-layer traces are compared (§6).
- **The real model** is only on the owner's box: there the same comparison runs layer by layer on real weights (DS1-F's tool).
- **sm_70 gate** for every CUDA file: `tools/volta/compile_one.py` (registers, spills, traps) and `tools/volta/sass_audit.py`: no spills
  in hot kernels, no trap, no local-memory stack frames.

## 2. Numerics (bit-level rules everyone follows)

| What | Rule | Oracle mode it matches |
|---|---|---|
| residual / mHC stream | FP32 `[T][kHc][kHidden]` | all |
| GEMV with a weight the reference stores FP8/FP4 (`linear(..., act_quant=True)` sites: `wq_a`, `wq_b`, `wkv`, `wo_b`, shared expert w1/w3/w2, `indexer.wq_b`, `engram.wkv`) | activations quantised to **int8 per 32 + FP32 scale** with CONTRACTS.md's quantiser (DS-D's `quantize_block_warp`), Q8_0 weights, `dp4a` per 32-block: `acc += d_w(fp16→fp32) * d_x * Σ int8·int8` (the integer sum is exact) | `QuantConfig.int8_act` |
| every other GEMV (BF16 / F32 weights, and `wo_a`, which the reference keeps bf16 although the GGUF stores Q8_0) | FP32 activations × dequantised weights, FP32 accumulation | exact |
| reduction order | fixed per output, **independent of T** (T = 1..8 give bit-identical rows; DS-D's router rule) and of the block/fiber order | - |
| KV fake-quantisation (the model's KV is QAT'd) | runtime flags mirroring the oracle's `QuantConfig.window_kv` (fp8 e4m3 per 32, `act_quant_fp8`), `.compressed_kv` (fp4 e2m1, e4m3 scale per 16, `fp4_quant_e4m3`), `.index` (fp4, e8m0 per 32, `fp4_quant_e8m0`); **default all on**; values stored in FP32 caches (the fake-quantised values are exact there) | same flags |
| RMSNorm | FP32, `eps = 1e-20` (the real value), `x * rsqrt(mean(x²) + eps) * w` | `ops.rmsnorm` |
| RoPE | tables computed on the host in double (`rope.py: rope_table`, YaRN per layer, theta 160000 + YaRN on the CSA2 layers, 10000 plain on L0-1), stored FP32; applied to the LAST `kRopeDim` channels, adjacent pairs; inverse on the attention output | `rope.py` |
| softmax / sigmoid / softplus / exp | FP32 with `expf` etc.; softplus threshold 20 like torch; attention sink adds `exp(sink - max)` to the denominator only | `ops.py`, `attention.sparse_attn` |
| routed experts | DS-C (CPU) / DS-D (GPU) as committed: CONTRACTS.md | `QuantConfig.int8_act` |
| determinism | no float atomics; every reduction has one fixed order | - |

Comparison mode for the oracle: `QuantConfig(int8_act=True, window_kv=True, compressed_kv=True, index=True)` (engine default), and the
engine must also run with the three KV flags off to match `QuantConfig.int8()`.

## 3. Layer roles (runtime, from the GGUF metadata; never hard-coded)

From `attention.compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`, `candidate_source_layer_id`, `engram_layer_ids`
(the exact keys: `ref/ds41/config.py: Config.from_gguf_metadata`, `layer_modes`): each layer is **SWA** (ratio 0), **FULL** (owns a
compressor + indexer; writes comp_kv / index_k; the candidate source also builds the pool), **REUSE** (reads the owner's comp_kv and
the last top-k indices) or **REINDEX** (own indexer q, the owner's index_k and comp_kv, restricted to the candidate pool). Every layer has
its own Q and its own `kWindow` SWA ring. Shared state follows `attention.SharedState` (compress owner, index-K owner, top-k indices,
candidate mask) with the port's index-K rule (each owner scores its own cache; `stale_index_k` exists only in the oracle).

## 4. Work packages and file ownership (one owner per path; read anything, edit only yours)

| WP | Owns | Delivers |
|---|---|---|
| **DS1-G** geometry templating of DS-C/DS-D | `src/ds41/cuda/ds41_router*`, `ds41_split*`, `ds41_experts*`, `ds41_math.cuh`, `ds41_dev.cuh`, `ds41_emu*`, `ds41_parity_lib.hpp`, `ds41_ref.hpp`, `ds41_cuda_dev.hpp`, `include/strata/ds41/cuda/ds41_cuda.hpp`, `src/ds41/cpu/**`, `include/strata/ds41/cpu/**`, `cmake/ds41_cuda.cmake`, `cmake/ds41_cpu.cmake` | router / split / hot-expert kernels and the CPU expert API as templates over `G` (RealGeom output bit-identical to today: SASS-diff and the existing parity programs prove it); `MiniGeom` instantiations with emulator tests at mini shapes; shared helpers other WPs may call (quantise, `Dev`, launch helpers) exposed in a header the others include |
| **DS1-A** loader and model state | `include/strata/ds41/model/**`, `src/ds41/model/**`, `cmake/ds41_model.cmake` | multi-shard GGUF reader (reuse `strata/artifact/gguf_reader.hpp` if it fits) → `Ds41Config` (every key the oracle's `Config.from_gguf_metadata` reads) validated against `G`; tensor directory; `Ds41Weights<G>`: dense tensors uploaded to the device in their GGUF formats (Q8_0 blocks verbatim, BF16, F32), per layer; `token_embd` stays in host memory (row lookup); the head on the device; Engram tables and expert slices mmapped; the CPU expert arena as per-socket halves (DS-C's packer; NUMA binding through `src/platform/numa.cpp` when there are 2 nodes); the GPU expert cache (slots, residency table −1-initialised, a static initial fill: DS-1 may fill by expert index); the Engram metadata (primes, offsets, multipliers, token_map) |
| **DS1-B** dense kernels | `src/ds41/cuda/dense_*`, `include/strata/ds41/cuda/dense.hpp`, `cmake/ds41_dense.cmake` | Q8_0 GEMV, int8-activation (dp4a) and FP32-activation variants, T = 1..8, rows = any multiple of 32; the grouped (block-diagonal) GEMV for `wo_a`; BF16 and F32 GEMV (head 129,280 × 5120 included); RMSNorm; RoPE / inverse RoPE on the tail channels; embedding-row upload; argmax and top-k over the vocabulary; the shared expert (Q8_0 w1/w3 → clamps → silu·u → w2, int8 activations, CONTRACTS.md clamps with NaN propagation); FP32 elementwise helpers. Each with an emulator test against a C++ FP64 reference at both geometries |
| **DS1-C** attention (CSA2) | `src/ds41/cuda/attn_*`, `include/strata/ds41/cuda/attn.hpp`, `src/ds41/attn/**`, `cmake/ds41_attn.cmake` | per layer: q path (`wq_a` → q_norm → `wq_b` → RoPE), SWA KV (`wkv` → kv_norm → RoPE → fp8 fake-quant → ring), compressor (ratio 2 with its group state, ratio 1), indexer (index_k from the latent, iq, weights_proj, relu-dot scores, top-k position-sorted, candidate pool: 8-block max, newest pinned, top blocks; REINDEX mask), sparse attention (≤ `kWindow` + ≤ `kIdxTopK` positions, `kHeads` × `kHeadDim`, sink), inverse RoPE, grouped `wo_a`, `wo_b`; the caches (FP32, sized by a runtime max context) and the shared state across layers. Calls DS1-B's GEMV/norm/RoPE through `dense.hpp` (until it lands, a local stub with the same signature) |
| **DS1-D** mHC and Engram | `src/ds41/cuda/mhc_*`, `src/ds41/cuda/engram_*`, `include/strata/ds41/cuda/mhc.hpp`, `include/strata/ds41/cuda/engram.hpp`, `src/ds41/engram/**`, `cmake/ds41_mhc.cmake` | `hc_mixes` (F32 `hc_fn` GEMV over the flattened stream, rsqrt-mean scale, `scale` and `base`), Sinkhorn split (`kHcIters` iterations, eps), `hc_pre`, `hc_post` (`mhc.py`); Engram: the host hasher (rolling XOR with the GGUF's primes / offsets / multipliers / token_map; `engram.py: NgramHasher`), the row gather from the mmapped MXFP4 tables (host → device, 136-B rows), the combine kernel (`engram.py: engram_layer`) |
| **DS1-F** verification tools | `tools/ds41/ds1_*.py`, `tools/ds41/test_ds1_*.py`, `ref/ds41/trace_io.py` (new; the only ref/ds41 file it may add), `docs/deepseek/DS1_VERIFY.md` | the trace format both sides write (§6); the comparison tool (oracle vs engine per layer and per stage, with tolerances derived from the numerics above); a layer-by-layer mode for real weights (feed the engine's input of layer L to the oracle's layer L, so error does not accumulate across 40 layers); the DeepSeek tokenizer from the GGUF metadata (text ↔ ids, checked against the vendored `tokenizer.json` if present); mini-GGUF fixtures for ctest |
| **DS1-E** session and CLI (after A-D) | `src/ds41/session/**`, `src/ds41/program/**`, `cmake/ds41_engine.cmake` | the decode step (embedding → per layer: Engram (on its layers) → attention sub-layer with mHC → FFN sub-layer: norm → router → split → GPU hits ∥ CPU misses (two sockets, one partial y each, DS-D's doorbell) + shared expert → mHC → … → final `hc_pre` → norm → head → logits), the CPU expert pool for two sockets, the `strata-ds41` CLI (`--gguf`, `--tokens`, `--max-new`, `--max-context`, `--dump-logits` in golden_compare's format, `--trace DIR`, KV quant flags), the MiniGeom end-to-end ctest against the oracle |

Integrator (cloud session): this file, `geom.hpp`, `geometry.hpp`, `CONTRACTS.md`, `CMakeLists.txt`.

## 5. Interfaces (each WP's public header is its contract to the others)

- Host functions, not kernels, are the API: `template <class G> void ds41_<op>(Dev&, const Args&..., Stream)` in the WP's header;
  launches stay inside the WP's `.cu` / `_impl.cuh` (explicit instantiation for `RealGeom` in the `.cu`, for `MiniGeom` in the emulator
  build). Pointers are device pointers from `Dev`; shapes come from `G`; T (tokens per call, 1..8) is a runtime argument.
- Activations between ops: FP32 row-major `[T][width]`. The int8 copy for dp4a GEMVs: DS-D's quantised-activation layout
  (`ds41_cuda.hpp`, "ACTIVATION AND h LAYOUT"), produced by one quantise call and reused by every GEMV that reads the same input.
- Each op is stream-ordered and allocation-free on its hot path; scratch comes from a per-session arena the caller passes.

## 6. The trace (what DS1-F compares)

One directory per run: `trace.json` (geometry name, quant flags, token ids, positions) and one little-endian FP32 `.npy` per
(stage, layer, position): `embed`, `engram_out.L`, `attn_in.L` (after hc_pre + norm), `q.L`, `kv_win.L`, `latent.L`, `topk.L` (int32),
`attn_out.L`, `ffn_in.L`, `router_idx.L` (int32), `router_w.L`, `ffn_out.L`, `block_out.L`, `pre_mix.L`, `final_hidden`, `logits`.
The oracle side writes the same names from `Model.forward(..., trace=)` (DS1-F adds what is missing through `trace_io.py` without changing
the oracle's numerics). Tolerances per stage are DS1-F's to derive and document; bit-exact where both sides do the same integer math
(router ids at non-near-ties, top-k positions at non-near-ties).

## 7. Gate DS-1

On the mini model (here, emulated): every stage within its documented tolerance for 64 tokens; greedy continuations identical.
On the real model (the owner's box): layer-by-layer within tolerance for 32 tokens of a real prompt; top-1 ≥ 99 % over 500 tokens
against the oracle; no NaN / inf; then the llama.cpp comparison (tier 2 of `tools/volta/logit_identity.sh`, with the owner's
mx-llama.cpp as reference) as information.
