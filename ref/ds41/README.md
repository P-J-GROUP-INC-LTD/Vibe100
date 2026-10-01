# ref/ds41 - NumPy oracle of the DeepSeek-V4.1-Flash forward pass (work package DS-A)

A transcription of the official PyTorch reference (`third_party/deepseek-v41-flash-reference/inference/{model,engram,kernel}.py`)
into NumPy, validated against that reference run on CPU. It is the ground truth every later C++/CUDA kernel of the port is
compared against: it favours traceability (every function names the `model.py` / `kernel.py` lines it transcribes) and
correctness over speed. **The oracle itself needs only NumPy** (+ the repo's `tools/gguf_reader.py` to read GGUFs); torch, sympy
and `tokenizers` are needed only by the tests that run the official code.

Scope: the 40-layer backbone, text only, batch 1: prefill of a whole prompt (one chunk starting at position 0, as in the
reference) and incremental decode with caches. Not implemented: DSpark drafting (`mtp.*`, 3 blocks), the vision tower, batching,
chunked prefill (the reference has none either).

```python
from ref.ds41 import model_from_gguf, QuantConfig
m = model_from_gguf(shard_paths, dtype="float32", max_seq_len=4096)      # lazy: config, Engram constants, token map from the file
cache = m.new_cache()
logits = m.forward(prompt_ids, 0, cache)                                  # prefill -> logits of the last position
logits = m.decode(next_id, cache)                                         # one token per call
m.forward(ids, 0, cache, full_logits=True, trace=tr)                      # all positions; tr[...] = per-layer block/attn/ffn outputs, router picks
# command line:  python -m ref.ds41 {selfcheck | info GGUF... | run GGUF... --ids 1,2,3 --decode 4 --npz out.npz}
```

## What transcribes what

| module | contents | transcribes |
|---|---|---|
| `config.py` | `Config` (= `ModelArgs` field names), from `inference/config.json`, from GGUF metadata, `Config.tiny()`; `layer_modes()` = SWA / FULL / REUSE / REINDEX map from `compress_ratios` + kv/index source ids | model.py:45-137, 613-660 |
| `quant.py` | GGML MXFP4 + Q8_0 + BF16 decoders, FP8-e4m3 32x32-block dequant, official packed-FP4 decode and its repack to GGML blocks, the reference's activation fake-quant (`act_quant_fp8`, `fp4_quant_e8m0`, `fp4_quant_e4m3`), the engine's `act_quant_int8`, `QuantConfig`; GGML-reference encoders for fixtures | kernel.py:22-205; ggml-quants.c `dequantize_row_mxfp4/q8_0`, ggml-impl.h `ggml_e8m0_to_fp32_half` |
| `ops.py` | RMSNorm (fp32, eps 1e-20), softplus/sigmoid/silu, `linear` (+ optional e4m3/int8 activation quant) | model.py:181-207, 281-293 |
| `rope.py` | partial RoPE (last 64 of 512, adjacent pairs) with YaRN, inverse; which theta/YaRN each layer uses | model.py:369-407, 682-691 |
| `moe.py` | router (sqrt-softplus, noaux_tc top-6 of s+bias, weights from unbiased s, /(sum+1e-20), x1.5), Expert (clamps, silu, routing weight before W2), MoE (routed + shared) | model.py:792-904, CONTRACTS.md |
| `attention.py` | window idxs + SWA ring, ratio-2/ratio-1 compressor, indexer (32x128, relu-weighted, top-512 position-sorted), candidate pool (8-blocks, newest pinned, top-2048), sparse attention with sink, grouped output projection, per-layer caches and the cross-layer hand-off | model.py:410-789, kernel.py:311-404 |
| `mhc.py` | hc_mixes, 20-iteration Sinkhorn, hc_pre / hc_post | model.py:948-966, kernel.py:407-475 |
| `engram.py` | primes / offsets / multipliers, token -> compressed-id map (stdlib re-implementation of the `tokenizers` normaliser), `NgramHasher`, the Engram layer | engram.py (all), model.py:296-366 |
| `model.py` | `Block` (the single-pass one-block lag in `pre`), whole-model `forward` (prefill / decode), `model_from_gguf` | model.py:907-994, 1242-1273 |
| `weights.py` | `Weights` providers: `DictWeights`, `GGUFWeights` (lazy, mmap), `load_from_gguf`, GGUF -> canonical-name map, shape contract | RESEARCH.md s10 |
| `selfcheck.py` | Engram constants vs the target GGUF's metadata; token map vs the GGUF's own map | RESEARCH.md s3 |

Parameters are addressed by the official state-dict names (`layers.3.attn.wq_a.weight`, `layers.3.ffn.gate.bias`,
`layers.3.hc_attn_fn`, `embed.weight`, ...); `weights.canonical_name()` maps the mxxm-t GGUF names onto them
(e.g. `blk.3.attn_q_a.weight`, `blk.3.ffn_gate_exps.weight` -> experts' `w1`, `blk.3.indexer.proj.weight` -> `weights_proj`).

## Running the tests

```bash
SP=...scratchpad;  $SP/venv-ref/bin/python -m pytest ref/ds41                 # all 102 tests, ~2.5 min on 4 shared vCPUs (needs torch>=2.10 CPU, numpy, sympy, pytest; tokenizers optional)
$SP/venv-ref/bin/python -m pytest ref/ds41 -m "not slow"                       # skip the real-dimension attention test (~80 s, ~3 GB RAM)
$SP/venv-ref/bin/python -m pytest ref/ds41 -s -k "tiny or real_dims"           # prints the error numbers quoted below
$SP/venv-ref/bin/python -m ref.ds41.selfcheck                                  # constants self-check (PASS/FAIL list)
```

Network: `tokenizer.json` (6.4 MB, `deepseek-ai/DeepSeek-V4.1-Flash`) is needed by the token-map tests; it is taken from
`$DS41_TOKENIZER_JSON`, else `$DS41_CACHE/ds41_cache/` (default: the temp dir), else downloaded; the tests skip if unavailable.
The loader tests use `tools/ds41/make_mini_gguf.py` (DS-B) and skip if it is absent.

Test files: `test_quant.py` (decoders and fake-quantisers, bit for bit), `test_ops.py` (every op vs the official function/module),
`test_tiny_model.py` (official `Transformer` vs the oracle, prefill + decode), `test_real_dims.py` (real shapes and YaRN, one layer
per mode), `test_constants.py` (Engram constants, token map), `test_config.py` (layer-mode map, all 1006 GGUF tensor names and
shapes), `test_loader.py` (mini GGUF). `tests/official.py` + `tests/kernel_stub.py` run the vendored reference on CPU.

## How the official code is run on CPU (and what could not be)

* `kernel.py` imports TileLang and needs a Hopper/Blackwell GPU: it is replaced (`sys.modules['kernel']`) by
  `tests/kernel_stub.py`, pure-torch versions of the six functions model.py uses, written from the kernels' code and comments
  (`act_quant`, `fp4_act_quant`, `fp8_gemm`, `fp4_gemm`, `hc_split_sinkhorn`, `sparse_attn`). They share no code with the oracle's
  NumPy versions. **This is the part that could not be run as shipped**; it is covered by (a) second formulations in the tests
  (sparse attention as concatenated-sink softmax; Sinkhorn as scalar loops following the kernel; the e4m3/fp4 rounding by brute-force
  nearest grid point; `fast_log2_ceil` by float32 bit manipulation; FP4 nibble decode taken from `convert.py`), (b) the stubs and the
  oracle agreeing bit for bit on 2382 quantiser calls of a full run. Residual risk = hardware behaviour a CPU cannot show: tie
  rounding of float->e2m1/e4m3 casts (round-to-nearest-even assumed), FP8 tensor-core accumulation precision, bf16 rounding of
  the probabilities in the attention kernel, e4m3 cast overflow of the compressed-KV scale (never reached by RMS-normed data).
* `vision.py` and `image_processor.py` (imported by model.py) are not vendored: stubbed; the vision path is off.
* `torch.distributed` is never initialised (world_size = rank = 1). `generate.py`, the chat encoder, the DSpark head
  (`DSparkBlock`, `forward_spec`) were not run.
* The released model runs with torch's default dtype bfloat16; the CPU tests run it in float32 (weights widened from their
  bf16-representable values; fp8/fp4 storage formats kept) - see "bf16" below.

## Tolerances (and why)

All errors are *relative to the tensor's max magnitude*, `max|a-b| / max|b|`.

| comparison | observed | bound asserted | why |
|---|---|---|---|
| each op vs official (RMSNorm, router, expert, MoE, compressor r1/r2, mHC, sparse attn, Engram layer; exact mode) | 4e-8 ... 4e-6 (softmax router 3.6e-6) | 1e-5 | the official is float32, the oracle float64: round-off only |
| RoPE tables vs official (positions < 512) | float32 table 1.5e-5, float64 table 2.4e-5 (absolute) | 5e-5 / 2e-4 | torch's table is itself float32: its angles carry ~3e-5 rad rounding at position 500 |
| quantisers (e4m3 act_quant, fp4 e8m0, fp4 e4m3, MXFP4/Q8_0/FP8 decode) | **0 (bit-exact)** | equality | same arithmetic, deterministic |
| tiny model, exact mode, every layer output, every position, prefill + 12 decode steps | 1.6e-6 (logits 2.1e-6, argmax identical) | 1e-4 | float32 round-off through 10 layers; 100x margin |
| tiny model, float32 oracle | 2.1e-6 | 5e-4 | |
| real dimensions, one layer per mode (SWA, FULL r2, REUSE, FULL r1, REINDEX), prefill + 2 decode | 1.5e-6 | 1e-4 | |
| reference-quantisation mode, every quantiser call on the official's own inputs | 0 (2382 calls) | equality | validates sites, order, shapes and arithmetic with no flip ambiguity |
| reference-quantisation mode, end to end | flip-free seeds 3e-7; flipped ~0.01-0.9 | best of up to 8 seeds <= 1e-4 | see below |
| bf16-as-deployed official vs oracle (layers 0-1, exact) | 0.6-1.2 % | 3 % | 2^-8 per bf16 rounding, a few in series |

**Why reference-quantisation mode is not compared elementwise end to end.** e4m3 / fp4 rounding is a step function. A 1e-7
difference between two float32 evaluations (or float64 vs float32) flips a rounding decision with probability of order 1e-6 per quantised
element; the tiny scenario (prefill 21 + 12 decode) quantises 3.6e5 elements (2382 calls), so roughly half of the random models contain a flip. A flip moves one
element by 6-30 % of its quantisation step, and the random tiny network amplifies it (top-k selections change). I verified the first
divergence in failing seeds: an input difference of 7.7e-7 and a single flipped element. A systematic bug would break every seed, so
the test requires one flip-free seed (it reports how many were) and the call-by-call test above removes the ambiguity.

**bf16.** Run as deployed (bfloat16 activations), the official model's sub-layers differ from the oracle by 0.4-1.2 % (bf16 round-off);
whole-model logits of a *random* tiny network then differ by tens of percent (discrete selections amplify noise; real, trained
models are presumably better conditioned - unmeasured). Consequence for the engine's tests: validate per op / per layer against
this oracle at the engine's own precision, and whole-model output statistically (top-1 agreement, KL), not elementwise.

## Quantisation modes (`QuantConfig`)

* `exact()` (default): weights as stored (dequantised), activations in the compute dtype, no KV quantisation.
* `reference()`: everything the reference does: e4m3 activation fake-quant (`act_quant`, 32, power-of-two scale) before every GEMM
  whose weight is fp8/fp4 in the reference (attention q/kv/o, indexer wq_b, shared and routed experts incl. W2's input, Engram wkv),
  fp8 SWA KV, fp4 (block 32, E8M0) indexer q and index-K, fp4 (block 16, E4M3 scale) compressed KV. Not applied to the bf16 tensors
  (gate, compressor, wo_a, weights_proj, indexer wk, head).
* `int8()`: the engine's choice (CONTRACTS.md): int8 per 32 with fp32 scale `d = max|x|/127`, ties away from zero, at the same GEMM sites,
  fp16-equivalent KV. On Gaussian data its rms error is 0.5 % against 2.7 % for e4m3 (heavy-tailed: 1.0 % vs 2.6 %).

## Known deviations from the reference, and reference quirks

1. **Reference bug (reproducible switch `stale_index_k`).** `model.py:560` scores the indexer against `shared_attn.index_k`, which an
   index-K owner (layers 2, 8, 14, 20) refreshes only when its compressor emitted a latent this step (`:548-549`). On ratio-2
   decode steps where the group is still incomplete (every other token), layers 2/8/14 therefore score against layer 20's cache (a
   different ratio) instead of their own: wrong top-512 selections, only visible once a layer has more than 512 compressed positions
   (context > 1024 tokens). Prefill is unaffected. The oracle's default is the evident intent (own cache); `Model(..., stale_index_k=True)`
   reproduces the reference exactly and the tiny-model test proves both statements (quirk on == stock official to 1.6e-6; quirk off ==
   the official with a one-line wrapper fix, `tests/official.py:patch_index_k_fix`, to 1.6e-6; stock vs fixed differ by up to 0.8 in
   logits in the tiny model). With the default, prefill of N tokens == prefill of N-k + k decode steps to 1e-15 (asserted); with the quirk it does not.
   **Needs a decision**: follow the intent (recommended; matches how it was trained, presumably) or the shipped code, and ask DeepSeek / check llama.cpp #28696.
2. RoPE tables are float64 by default (truth); the reference's are float32, whose angles are inaccurate at long positions (error ~position x 6e-8 rad:
   0.06 rad at 1M). `rope_dtype=np.float32` gives the reference's table.
3. Top-k ties (router, indexer, candidate blocks): the reference uses `torch.topk` (unspecified tie order); the oracle takes the lower index.
   `RouterOut.margin` and `SharedState.diag` (index / block selection margins) expose near-ties; the tests prove their scenarios tie-free.
4. Indexer selections whose score is -inf (unreachable, or outside the candidate pool) are always -1 here; the reference can emit an arbitrary
   in-range position for them when fewer than `index_topk` positions are reachable inside the pool (only when pool < index_topk).
5. The pooling state of the ratio-2 compressor is reset at prefill (the reference overwrites every slot before use anyway).
6. Scales of the fake-quantisers are computed in float32 (as the kernels do); `fp4_quant_e4m3` saturates the scale at 448.
7. Engram table values are not rounded to bf16 (the reference does `.to(bfloat16)`): exact for fp8 x 2^k and fp4 x 2^k rows, i.e. for both the official
   FP8 tables and the GGUF's MXFP4 tables. The GGUF's MXFP4 re-quantisation of Engram rows (lossy vs the official FP8) is *not* measured here (needs the weights).
8. No image/DEAD-token logic in the Engram n-gram hashing (text only).

## Constants self-check (`python -m ref.ds41.selfcheck`)

All pass against the saved header of the target GGUF (`gguf-headers-mxxm-t-MXFP4.json.gz`) and the real shard 1 token map:
* the 48 primes (first `16000057`, last `16000889`), the 48 offsets and the 8 multipliers (seeds `10007*1`, `10007*14`) derived as engram.py does
  equal `deepseek41.engram.{primes,offsets,multipliers}`; each layer's 24 primes sum to `engram_num_embeddings` = `[384006168, 384016682]`
  (also equal to config.json and the GGUF); `is_prime` agrees with sympy;
* `deepseek41.engram.token_map` (129,280 entries; only its length is in the saved header, so its contents were read from the real shard-1 header over HTTP
  range requests and kept as `tests/data/engram_token_map_mxxm.u32.gz`, 250 KB: gzip of the raw uint32 array): 99,092 distinct ids, first-appearance order, and **identical, entry for entry,
  to the map rebuilt from `tokenizer.json`** by both the oracle's stdlib normaliser and the official `engram.build_compressed_token_map` (HF `tokenizers`).
  (StripAccents drops every combining mark, Mn *and* Mc - Indic vowel signs - which the first draft missed; the full-vocabulary comparison caught it.)
* `config.json` == GGUF metadata for every field the port uses; all 1006 GGUF tensors map to canonical names with the shapes the config implies.

## Open questions / risks

1. The stale-index-K behaviour above (decide, then freeze the default).
2. The real GGUF has not been read here (not downloaded): the loader is validated on the mini GGUF (real names/types, small dims) and its name map and shape contract
   on all 1006 tensor records of the real header; dequantisation is cross-checked against the independent codecs of `tools/ds41/ggml_codecs.py` and a scalar transcription
   of GGML's C. Q8_0 dense tensors are decoded whole (up to 168 MB as fp32), experts and Engram rows lazily; speed is ~0.1 s per expert matrix, so a full-model
   token on real weights takes minutes: use it for single layers / few tokens (`--npz` dumps per-layer outputs).
3. GPU-only behaviours listed in "what could not be run".
4. bf16 end-to-end comparability (above) - the acceptance criteria of DS-1 (top-1 >= 99 %) should be statistical.
