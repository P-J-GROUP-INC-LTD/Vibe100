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
  rows, then `y_k = W2[:, those columns] · h_k`; `y = y_0 + y_1`. **Each socket writes its own partial `y_k`** (FP32
  `[T][5120]`, in its own node's memory) and the two are added once per layer, after both are complete (on the GPU,
  with the hits' output). The two halves' down rows overlap completely (each covers all 5120 rows), so two sockets
  must never add into one buffer — that loses updates (audit A5: 510 of 2,000 concurrent runs). Within a socket, the
  experts of a layer may share `y_k` only with static row ownership (every thread owns the same output rows in phase 2
  for all experts) or one buffer per expert.
- **Activations** for the integer dot products: int8 per 32 values with an FP32 scale. That is more precise than the
  reference's FP8-e4m3 fake-quantised activations; DS-A's oracle provides both (exact FP32 activations, and the
  reference's `act_quant` path) so the difference can be measured. `h` (the input of W2) is quantised the same way,
  after the routing weight is applied. **The rule, bit for bit, everywhere (CPU scalar / AVX2 / AVX-512, GPU, the GPU
  emulator, the oracle's int8 mode)** — it is ggml's x86 SIMD `quantize_row_q8_0` (`ggml-cpu/arch/x86/quants.c`),
  except that `d` stays FP32 (ggml stores it as fp16), and NOT ggml's `quantize_row_q8_0_ref`, which divides and
  rounds half away from zero:
  - `amax` = the largest `|x|` of the block; "non-finite" = the block holds an Inf or a NaN (test the magnitude BITS:
    the integer maximum of `bits & 0x7FFFFFFF` is `>= 0x7F800000` exactly then — order-independent, unlike a float max
    with a NaN operand).
  - a non-finite block: `d = NaN`, every `q = 0` (the NaN reaches `y`, so a failure upstream is not laundered into a
    finite value);
  - `amax < 2^-100` (zero included — `127 / amax` would overflow below ~3.7e-37): `d = 0`, every `q = 0`;
  - otherwise `d = amax / 127` and `id = 127 / amax` (two FP32 divisions, round-to-nearest), `q = rint(x · id)` (FP32
    product, round half to EVEN), clamped to `[-127, 127]`.
- **NaN propagates through the clamps** as through `torch.clamp`: `u = u > 10 ? 10 : (u < -10 ? -10 : u)`,
  `g = g > 10 ? 10 : g` (C `fminf`/`fmaxf` and x86 `minps`/`maxps` with the constant as the first operand return the
  constant for a NaN — don't).

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
once a layer has more than 512 compressed positions (context above ~1,024 tokens; below that every position is
selected) — and then for all 18 ratio-2 layers, not 3: the Reuse layers 3-7, 9-13 and 15-19 take their owner's
`topk_idxs` (audit A6).

**The port follows the intent: each owner always scores against its own cache.** The oracle implements both
(`ref/ds41`: default = intent; `stale_index_k=True` reproduces the shipped reference bit for bit), so either can be
compared against. To revisit if DeepSeek or llama.cpp PR #28696 show their production decode does otherwise.

## Decided after the audit (2026-10-01)

- **The activation quantiser** is the rule above (ggml's x86 SIMD q8_0 arithmetic). Audits A5 and A6 found the
  oracle's int8 mode on the `_ref` rule (`1 / (amax / 127)`, ties away from zero) while the CPU and GPU kernels used
  `127 / amax` and ties-to-even: ±1 code on ~1e-3 of bf16-valued inputs. The oracle follows the kernels.
- **GPU expert cache slots.** A residency value `r` is a hit only if `0 <= r < n_slots`; the split kernel takes
  `n_slots` and treats anything else as a miss (so the CPU computes it) and counts it in `SplitCounts` so the caller
  can assert. A residency table is initialised to -1 (bytes 0xFF), never to 0.
- **The host-visible split result is per layer and carries a sequence number.** The expert kernels read
  `n_groups` from DEVICE memory (never from the mapped record: every block would otherwise read it over PCIe); the
  host-visible record (counts + miss list) is one per layer in flight, and its doorbell is a sequence number passed to
  the launch and stored last with release semantics (`st.release.sys` / `cuda::atomic_ref`), after one
  `fence` by the writing thread — so launches can be queued ahead (or captured in a graph) without a re-arm race.
- **Routing must not depend on T.** The router's logits for a token are bit-identical whether it is routed alone
  (decode), in a verify window (T <= 8) or in a prompt chunk (T > 32): one reduction order for every T. Otherwise a
  near-tie routes differently in prefill and in decode, and prefill(N) ≡ prefill + decode cannot hold.
- **The oracle is bit-exact against an FP32 run of the reference**, not the shipped bf16 one (bf16 residual stream,
  `w2(h.to(bf16))`, bf16 index-score einsums). DS-1's thresholds against the official model carry that difference.
