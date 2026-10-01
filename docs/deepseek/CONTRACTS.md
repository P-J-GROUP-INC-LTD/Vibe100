# DS-0 contracts — what the DeepSeek work packages agree on

Shapes and byte layouts: `include/strata/ds41/geometry.hpp` (owned by the integrator; ask before changing).
Spec: `docs/deepseek/RESEARCH.md` (§10 = the target GGUF). Ground truth: the vendored official reference
(`third_party/deepseek-v41-flash-reference/inference/model.py`, `engram.py`). Plan: `docs/deepseek/PLAN.md`.

## Ownership (one package per path; nobody edits another's files)

| WP | Paths |
|---|---|
| DS-A oracle | `ref/ds41/**` |
| DS-B GGUF tooling | `tools/ds41/**` (may absorb `tools/ds41_gguf_remote_headers.py`) |
| DS-C CPU experts | `include/strata/ds41/cpu/**`, `src/ds41/cpu/**`, `cmake/ds41_cpu.cmake` |
| DS-D GPU experts + router | `include/strata/ds41/cuda/**`, `src/ds41/cuda/**`, `cmake/ds41_cuda.cmake` |
| integrator | `include/strata/ds41/geometry.hpp`, this file, `CMakeLists.txt` (it includes every `cmake/ds41_*.cmake`) |

## The math (from `model.py`: `Gate`, `Expert`, `MoE`)

- **Router**, per token, in FP32: `s = sqrt(softplus(x · Wgᵀ))` (Wg BF16 [384, 5120], `gate_temp` = 1);
  `idx = top6(s + bias)` (bias F32 [384], `exp_probs_b.bias`; it selects only); `w = s[idx]`;
  `w /= (Σw + 1e-20)`; `w *= 1.5`. Ties: the reference uses `torch.topk`; tests compare sets and weights with a
  tolerance, and flag near-ties instead of failing on them.
- **Routed expert** with weight `w`: `g = W1·x`, `u = W3·x` (FP32 accumulation); `u = clamp(u, -10, 10)`;
  `g = min(g, 10)`; `h = silu(g) · u`; **`h *= w` before the down projection**; `y = W2·h`.
  Shared expert: the same with `w = 1` (Q8_0 weights in the GGUF). Layer output: `Σ routed y + shared y`, FP32.
- **Tensor-parallel halves** (CPU): half `k` computes `h` for intermediate rows `[1152k, 1152k + 1152)` from its gate/up
  rows, then `y_k = W2[:, those columns] · h_k`; `y = y_0 + y_1`.
- **Activations** for the integer dot products: int8 per 32 values with an FP32 scale (`d = max|x| / 127`,
  round-to-nearest). That is more precise than the reference's FP8-e4m3 fake-quantised activations; DS-A's oracle
  provides both (exact FP32 activations, and the reference's `act_quant` path) so the difference can be measured.
  `h` (the input of W2) is quantised the same way, after the routing weight is applied.

## MXFP4 (GGML `block_mxfp4`, 17 bytes per 32 values)

`e` (E8M0) then `qs[16]`; value `j` = low nibble of `qs[j]`, value `j + 16` = high nibble; value =
`kvalues_fp4[code] · 2^(e − 128)` with `kvalues_fp4 = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}` (doubled
e2m1). Check against `ggml-quants.c:dequantize_row_mxfp4` and `ggml-impl.h:ggml_e8m0_to_fp32_half` in the fetched
llama.cpp (`<build>/_deps/strata_llamacpp-src/ggml/src/`), including `e` = 0, 1 and 255.

Layouts:
- **GPU blob** (a cache slot): the expert's three GGUF slices concatenated, `[gate][up][down]`, 18,800,640 B.
- **CPU half** (socket `k`): `[gate rows 1152k..][up rows 1152k..][down: all 5120 rows, blocks 36k..36k+35]`,
  9,400,320 B. Built from the blob by DS-B's tool (Python, for tests) and later by the engine's loader.

## Testing without the weights or a GPU

- C++ tests generate random MXFP4 blocks (all 256 `e` values that matter, every code) and compare against a scalar
  FP64 implementation of the semantics above, in the same program. No test needs the 411 GB GGUF.
- DS-B's mini-GGUF generator writes a `deepseek41`-shaped file with tiny dims for loader tests; DS-A's oracle can
  read it.
- CPU benchmarks run on this container (AVX-512 F/BW/VL/DQ + VNNI, **no VBMI** — the same feature set as the target
  Xeon Gold 6226; 4 vCPUs, one socket, numbers are indicative only).
- GPU code is compiled for sm_70 with `tools/volta/compile_one.py` and gated with `tools/volta/sass_audit.py`;
  its parity programs run on the V100.

## Decided: the index-K cache a ratio-2 indexer scores against (2026-10-01)

DS-A found that the official reference (`model.py` `Indexer.forward`, lines 537-554) publishes an owner layer's
index-K cache to `shared_attn.index_k` only on steps where its compressor completed a group. On the other
decode steps of the ratio-2 owners (layers 2, 8, 14 — every other token) the layer scores against whatever was
published last: layer 20's ratio-1 cache from the previous token, sliced to `end_pos // 2` entries. This
contradicts the class's own invariant ("every source writes before its consumers read"), breaks prefill(N) ≡
prefill + decode, and does not occur in prefill or training (full sequences). It changes the top-512 selection only
once a layer has more than 512 compressed positions (context above ~1,024 tokens).

**The port follows the intent: each owner always scores against its own cache.** The oracle implements both
(`ref/ds41`: default = intent; `stale_index_k=True` reproduces the shipped reference bit for bit), so either can be
compared against. To revisit if DeepSeek or llama.cpp PR #28696 show their production decode does otherwise.
