# Provenance - where everything in this repository comes from

Vibe100 is a port of [Strata](https://github.com/Niko1221/Strata). This page lists every external source in the tree,
what was taken from it, under which license, and what was only studied. Written 2026-10-01 for commit `f1fac23`
(HEAD of the port at the time); the checks behind each statement are listed in section 7. Where a fact comes from
this repository's own docs and was not re-checked, the row says so.

**License of the port's own code: MIT, like upstream.** `LICENSE` is upstream's file, unchanged (`git diff 3906943 -- LICENSE`
is empty): "MIT License - Copyright (c) 2026 Niko1221 and the Strata contributors". The code added after the
first commit (the Volta kernels, the CPU kernel variants, the NUMA mirror, the installer changes, `tools/volta`,
`tools/ds41`, `ref/ds41`, `src/ds41`, the docs) is released under that same license. Third-party parts keep their own:
see the table in section 1.

## 1. Licenses in the tree

| License | Where | Holder |
|---|---|---|
| MIT | everything not listed below (upstream Strata and the port's additions) | `LICENSE`: Niko1221 and the Strata contributors (the file has no separate line for the port's additions) |
| MIT | `third_party/ggml/` (`LICENSE`, `ggml-common.h`); the 17 source files whose headers carry the ggml notice (section 2.3, 2.4) | The ggml authors, 2023-2026 |
| MIT | `third_party/deepseek-v41-flash-reference/` (`LICENSE`, `config.json`, `inference/*`) | DeepSeek, 2023 |
| SIL Open Font License 1.1 | `serve/web/fonts/` (Outfit; `OFL.txt`) | The Outfit Project Authors, 2021 |
| Qwen Community License 1.0 | `data/experimental-speed-projection/` (a control vector made from the model's activations; its README) | Qwen / the model's license |

Not in the tree, fetched when you install or build: the model files (section 3), llama.cpp (section 2.2), the NVIDIA CUDA
Toolkit and libraries (NVIDIA's own license), and the Python packages in `requirements.txt` (not reviewed here).

## 2. Code and data copied or vendored

### 2.1 Upstream Strata - the first commit

| | |
|---|---|
| Path | the whole tree of commit `3906943` (522 files plus `UPSTREAM.md`) |
| Origin | https://github.com/Niko1221/Strata, commit `9259cad4cfa3543cd3b8decab5962672b968c649` (engine 0.1.31: `project(strata VERSION 0.1.31)` in `CMakeLists.txt`) |
| License | MIT, Copyright (c) 2026 Niko1221 and the Strata contributors |
| What was taken | everything: the engine, the installer, the web app and server, the packs and tools, the upstream docs, benchmark results, the paper |
| What the port changed | 34 of those files were modified in later commits (CMake, installer, device check, CPU expert kernels and pool, prompt attention, prefill GEMM, `generate.cpp`, a few headers, three docs); the other files are untouched. `git diff --stat 3906943 HEAD` is the authoritative list |
| Checked | all 522 files are byte-identical to the files GitHub serves for that commit (section 7) |

### 2.2 llama.cpp / ggml fetched at build time (not in the tree)

| | |
|---|---|
| Origin | https://github.com/ggml-org/llama.cpp, commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (pinned in `CMakeLists.txt` `FetchContent_Declare(strata_llamacpp ...)`, in `setup.py` `LLAMA_CPP_COMMIT`, and in `third_party/ggml/VERSION.txt`, which also gives its date, Sun Sep 20 16:08:11 2026 +0800) |
| License | MIT, Copyright (c) 2023-2026 The ggml authors |
| What is used | `ggml-base` and `ggml-cpu` linked as static libraries (the CPU dot products for the i-quant experts; the quantisers in tests); the ggml-cuda MMQ kernel sources (`mmq.cuh`, `template-instances/mmq-instance-*.cu`, `quantize.cu`) compiled into the `strata_mmq` target for the prompt path's expert GEMMs; `mtmd` (the image encoder, `tools/vision/`); `gguf-py` (`tools/_paths.py`, the pack tools). It is compiled and linked, not copied into the tree |
| Introduced by | upstream (all of it); the port only reuses it as a test oracle (section 2.4) |

### 2.3 llama.cpp / ggml code vendored or adapted inside upstream's tree (inherited)

| Path | What | Notice |
|---|---|---|
| `third_party/ggml/ggml-common.h` | block structs and codebook grids of the i-quant formats, copied unchanged | `third_party/ggml/LICENSE` |
| `src/kernels/cuda/native_{bf16,flash_attn,gdn,gdn_preprocess,gr_norm,gr_postops,mmvq,moe,ple_postops,qsa,qsa_indexer,qsa_score,rope,router}.cu`, `src/kernels/cuda/dequant_bf16.cu` | CUDA arithmetic adapted from ggml-cuda at the pinned commit (the headers name the source files, e.g. `native_qsa_score.cu`: `mmf.cuh`, `mma.cuh`, `unary.cu`, `binbcast.cu`) | MIT notice in each header |
| `src/kernels/cpu/expert.cpp` | the x86 Q8_0 quantiser and the generic Q2_0 dot sequence, adapted from `ggml-cpu/arch/x86/quants.c` and `ggml-cpu/quants.c`. The port edited this file (WP-E, the no-VBMI unpack); the adapted parts are upstream's | MIT notice in the header |
| `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cuda/s_gemv.cu`, `include/strata/artifact/dequant.hpp`, `include/strata/kernels/{dequant_bf16,rope_scaling}.hpp`, `src/kernels/cuda/sampler.cu`, `tools/strata_tokenizer.py`, `tools/canonical_xcheck.py` | dot products, dequantisers, the `kvalues_iq4nl` codebook, a RoPE extension formula, sampler penalties and a tokenizer pattern, transcribed from llama.cpp / ggml (the comments say so) | a pointer to `third_party/ggml/LICENSE` or none |

`native_qsa_score.cu` and the other `native_*.cu` files above were already in the upstream snapshot; the port did not edit
them. The only file written in the port that carries a ggml notice is `src/ds41/cuda/ds41_math.cuh` (section 2.4).

### 2.4 Code written in the port that follows ggml

| Path | Relation to ggml | Notice |
|---|---|---|
| `src/ds41/cuda/ds41_math.cuh` | the MXFP4 decode follows the technique of llama.cpp's `get_int_from_table_16` / `vec_dot_mxfp4_q8_1` (`ggml-cuda/vecdotq.cuh`, pinned commit): a 16-entry byte table looked up with `prmt`, the doubled e2m1 table `kvalues_fp4`, the E8M0 scale applied in FP32. The header says it is a rewrite for this layout, not a copy | MIT notice (The ggml authors) in the header |
| `src/ds41/cpu/mxfp4_expert.cpp` (`e8m0_half`, `kMxfp4Values`), `src/ds41/cpu/mxfp4_avx2.cpp`, `src/ds41/cuda/ds41_ref.hpp`, `include/strata/ds41/{geometry,cpu/mxfp4_expert,cuda/ds41_cuda}.hpp` | the semantics of `ggml_e8m0_to_fp32_half` (a few lines, marked "verbatim" in a comment), the `kvalues_fp4` table and the `block_mxfp4` layout (17 bytes per 32 values) | no notice in these files; the ggml MIT license is `third_party/ggml/LICENSE` |
| `tools/ds41/ggml_codecs.py`, `ref/ds41/quant.py` | NumPy re-implementations of ggml's `dequantize_row_mxfp4`, `quantize_row_mxfp4_ref`, `dequantize_row_q8_0`, `quantize_row_q8_0_ref` (the oracle's engine-int8 mode follows the arithmetic of ggml's x86 SIMD `quantize_row_q8_0` instead - `127 / amax`, ties to even - as `docs/deepseek/CONTRACTS.md` decides) and the BF16 / E8M0 helpers ("line-for-line port of the semantics"), checked bit for bit against ggml's C code | no notice in these files |
| `tools/ds41/ggml_c_oracle.py`, `src/ds41/cpu/mxfp4_ggml_xcheck.cpp`, `src/kernels/cpu/iq_avx512_test.cpp`, `src/kernels/cpu/pool_test.cpp` | use ggml only as a test oracle: they compile or link the fetched ggml (`ggml-quants.c`, `ggml-cpu`) and compare against it. The only text of ggml's own in `ggml_c_oracle.py` is a four-line wrapper of its public helpers | - |

The Volta WMMA prompt-attention kernel (`prompt_attn_volta_kernel` in `src/kernels/cuda/qsa_prompt_attn.cu`) is a port of
upstream Strata's own v1 kernel onto `nvcuda::wmma`. A comment mentions llama.cpp's `mma.cuh` only to say how much
care a raw m8n8k4 layout needs; no code is taken from it. The FP16 prefill GEMM route (`src/prefill/gemm.cu`), the
no-VBMI expert build, the NUMA mirror (`src/platform/numa.cpp`) and the V100 router/expert kernels' structure are the
port's own.

### 2.5 DeepSeek-V4.1-Flash reference and what was derived from it

| Path | What | Origin / license | Checked |
|---|---|---|---|
| `third_party/deepseek-v41-flash-reference/{LICENSE, config.json, inference/{README.md, config.json, convert.py, engram.py, generate.py, kernel.py, model.py, requirements.txt}}` | the official reference implementation (PyTorch + TileLang), vendored unmodified; nothing in the engine imports it | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash, copied 2026-10-01 per its README; MIT, Copyright (c) 2023 DeepSeek | all 10 files are byte-identical to the repository's `main` as served on 2026-10-01 |
| `third_party/deepseek-v41-flash-reference/tensors.json.gz` | name -> [dtype, shape, bytes] of the 96,085 tensors of the 48 safetensors shards (sum 510,286,023,000 bytes), parsed from the shard headers so pack tools can be tested without the 475 GiB checkpoint. Not a DeepSeek file | derived data (facts about the MIT-licensed checkpoint) | the entry count and byte sum were recomputed; not re-fetched from the shards |
| `third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz`, `...-vcruz305-Q2_K.json.gz` | metadata and tensor tables (names, types, dims, offsets; long arrays summarised) of 12 shards + the DSpark sidecar (1,084 tensors) and of 7 shards (1,046 tensors), read by HTTP range requests with `tools/ds41_gguf_remote_headers.py` | derived data from the two GGUF repositories (both tagged `license:mit` on Hugging Face) | counts recomputed; not re-fetched |
| `third_party/deepseek-v41-flash-reference/README.md` | written by the port | - | - |
| `ref/ds41/` (NumPy oracle) | a transcription of `model.py`, `engram.py` and `kernel.py` into NumPy; every function cites the lines it transcribes. A derivative of the MIT reference | DeepSeek MIT (`third_party/deepseek-v41-flash-reference/LICENSE`) | - |
| `ref/ds41/tests/kernel_stub.py`, `ref/ds41/tests/official.py` | pure-torch stand-ins for the six TileLang functions of `kernel.py` (written from its code and comments) and a harness that runs the vendored `model.py` on CPU | same | - |
| `ref/ds41/tests/data/engram_token_map_mxxm.u32.gz` | the 129,280-entry Engram token map (99,092 distinct ids) as a gzipped little-endian uint32 array, read from the `deepseek41.engram.token_map` metadata of shard 1 of the mxxm-t GGUF. The tests rebuild the same map from DeepSeek's `tokenizer.json` (downloaded at test time, not stored) and compare entry by entry (`ref/ds41/README.md`) | derived from the MIT GGUF / tokenizer | array length and distinct count recomputed; the comparison with `tokenizer.json` is the README's claim, not re-run |

### 2.6 Fonts and other data inherited from upstream

| Path | What | License |
|---|---|---|
| `serve/web/fonts/outfit-*.woff2`, `OFL.txt` | the web app's font, Outfit | SIL OFL 1.1 |
| `data/experimental-speed-projection/` | a 480 KB control vector (refusal-direction projection) made from Qwen3.8-Flash-Next activations, off by default | Qwen Community License 1.0 (its README) |
| `data/expert-profile.bin`, `expert-profile-coder.bin`, `draft_vocab*.bin` | expert-usage profiles (from the upstream author's prompts, per `docs/USAGE_LEDGER.md`) and draft vocabularies (`tools/draft_vocab.py`) | upstream; covered by `LICENSE` |
| `docs/paper/Strata-Paper.pdf`, `docs/media/*`, `bench/results/*` | upstream's paper, figures and benchmark data | upstream (MIT) |
| `tools/volta/prompts/kld_text.txt` (added by the port) | the default text of the llama.cpp comparison: excerpts of *Alice's Adventures in Wonderland*, *A Tale of Two Cities*, *The Adventures of Sherlock Holmes* and *Moby Dick* (Project Gutenberg #11, #98, #1661, #2701, licence header and footer removed) mixed with four upstream source files and `docs/DETAILS.md` from `3906943`; `tools/volta/prompts/make_kld_text.py` rebuilds it byte for byte | the books: public domain; the rest: upstream (MIT) |

## 3. Models and data it runs

| What | Where it comes from | Used by | License |
|---|---|---|---|
| Qwen3.8-Flash-Next (the base model) | https://huggingface.co/Qwen/Qwen3.8-Flash-Next (Qwen team); `tools/mtp_fetch.py` reads its MTP draft layer at revision `de4b8e4...` | the Qwen path (upstream) | upstream: "their licenses apply to the model files". Hugging Face on 2026-10-01: `license: other`, `qwen-community-1.0` |
| Original-model quantisations (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S) and the vision projector | https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF, revision `ed59f92...` (pinned in `setup.py`) | installer menu (upstream) | Hugging Face on 2026-10-01: `apache-2.0` |
| Coder (expert-pruned, IQ1_M) | https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, revision `5348543...`; upstream credits its support to a contribution (PR #54) | installer menu (upstream) | Apache-2.0 per its card (upstream `docs/DETAILS.md`; Hugging Face on 2026-10-01 agrees) |
| Swift 1.5 (a fine-tune) | https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF, revision `b22d729...` | installer menu (upstream) | its own: Swift Open License 1.0 (`setup.py`; Hugging Face agrees) |
| Manual imports, not offered by the installer | OrcaRouter's Flash-Next Uncensored IQ3_XXS (`docs/ORCA.md`); Unsloth's UD-Q4_K_XL (`docs/UNSLOTH_Q4.md`) | upstream's docs | Hugging Face on 2026-10-01: `apache-2.0` and `qwen-community-1.0` respectively; upstream's docs do not state them |
| DeepSeek-V4.1-Flash MXFP4 GGUF (12 shards + a DSpark sidecar, 375.8 GiB + 7.43 GiB) | https://huggingface.co/mxxm-t/DeepSeek-V4.1-Flash-GGUF (built for the `mx-llama.cpp` fork), the file the DeepSeek port targets | `tools/ds41`, `ref/ds41`, `src/ds41` tests (shard headers only so far; no tensor data has been read) | `license:mit` on Hugging Face on 2026-10-01 |
| DeepSeek-V4.1-Flash GGUFs in Q2_K...Q8_0 | https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF | only its header inventory is kept; `tools/ds41/manifest.py` refuses it (experts not MXFP4) unless `--allow-non-mxfp4` | `license:mit` on Hugging Face on 2026-10-01 |
| DeepSeek-V4.1-Flash safetensors (475 GiB) | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash | the reference only; not downloaded by the port | MIT |

The Qwen models and the DeepSeek GGUFs are not part of this repository. A license tag on a quantisation does not replace
the base model's license; upstream's wording ("Their licenses apply to the weights") is the safe reading.

## 4. Studied, not copied

No file in this tree contains code from these projects: a search of `src`, `include`, `tools`, `ref`, `cmake`,
`CMakeLists.txt` and `setup.py` for "ninfer", "1Cat" and "v100-skinny" finds nothing. They were read to decide what to
build; the ideas are recorded in the docs named below, and each "worth taking" item stays a candidate until someone
writes it.

| Source | License | What was learned (where) |
|---|---|---|
| [Neroued/ninfer](https://github.com/Neroued/ninfer) and its V100 forks (geoffwatts, dollarwong, JimmyMax, andreasknopke, justxiami tp4, mylordmonkeyman's Flash-Next backport) | Apache-2.0 (ninfer's `LICENSE` checked; the study says the forks are Apache-2.0 too) | The headline V100 numbers decomposed: ~30 tok/s is base decode of a 27B NVFP4 model, 45-123 with MTP speculation on real text, 219-236 only on a tiled corpus. On a V100 speculation, not the weight format, is the lever, and Strata already has it. Apache code cannot be relicensed to MIT, so the useful tricks (small-T attention tile, ReplaySSM, M=2..8 GEMV) are to be reimplemented if wanted. The Flash-Next backport streams every selected expert over PCIe, the opposite of Strata's design. `docs/volta/NINFER_STUDY.md` |
| [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) | MIT for `kernels/`, `benchmarks/`, `scripts/`, `docs/`, `results/`; Apache-2.0 for its `fork_patches/` (taken from 1Cat-vLLM) - its `LICENSE`, checked | the QPN M=2..8 skinny GEMV (only speculative-verify widths gain; at M=1 plain SIMT is near the bandwidth ceiling) and the MXFP4 layout the 1Cat kernel uses. `docs/volta/NINFER_STUDY.md` |
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | Apache-2.0 (its `LICENSE`, checked) | an M=1 MXFP4 tensor-core expert GEMV for DeepSeek V4 on a V100 (all routed experts in one launch, device-side ids), a compact grouped launch of only the active experts, E8M0-to-fp16 by exponent rebasing, repack-on-fill. The V100 expert kernels in `src/ds41/cuda` are a dp4a + `prmt` design with FP32 scales; the m8n8k4 candidate is not written. `docs/volta/NINFER_STUDY.md`, `docs/deepseek/PLAN.md` section 4 |
| llama.cpp's Volta m8n8k4 fragment maps (`mma.cuh`; ninfer's `volta_mma.cuh` is a transcription of them, MIT per the study) | MIT | suggested as a cross-check for the WMMA kernel, and a port of llama.cpp's Volta flash-attention kernel "only if the WMMA kernel underperforms"; neither done. `docs/volta/NINFER_STUDY.md` |
| llama.cpp pull requests and issues for DeepSeek V4 / V4.1 (#28696 `deepseek41` arch, open, no sparse attention; #24162 V4, merged; #28887 V100 sparse-attention enablement, closed unmerged; #25382 quantised KV gives garbage) | MIT (the project) | llama.cpp cannot run V4.1 correctly yet, so the reference is DeepSeek's PyTorch code; keep the KV cache in fp16; a V100 data point (V4-Flash on 8 V100, ~12.5 tok/s). Recorded in `docs/deepseek/RESEARCH.md` section 8 as third-party reports, not re-checked |
| Published expert hit-rate measurements: 0xBakeer (10,760 teacher-forced tokens), ik_llama.cpp #2449 (RTX 3090), JigSawPT (RTX 5090) | as published | every layer uses nearly all experts, but the top quarter covers 59-83% of routed slots; hot sets differ by task (Jaccard 0.18-0.31); hit rates of 0.41-0.54 (32 slots per layer) and 0.627 (about 25 per layer). They give the estimate of 0.5-0.65 for a V100 32 GB and the case for adaptive caching and the usage ledger. `docs/deepseek/RESEARCH.md` section 7 (flagged [3RD]) |
| DeepSeek-V4.1 technical report (arXiv 2609.19969) and API news page | - | the architecture facts in `docs/deepseek/RESEARCH.md` (MQA + CSA2 + Engram + mHC, QAT'd KV formats); the official code and weights are the primary sources, the report cross-checks them |
| Upstream Strata's issue tracker (#236, #220, #224) | - | #236: a V100 32 GB user ran the Coder and Swift IQ3_XXS with upstream's experimental build; #220 / #224: an RTX 50 engine built with CUDA 12.8 crashed on long prompts, which is why a V100 and an RTX 50 cannot share an engine. Cited in `docs/volta/PLAN.md` and `docs/volta/VOLTA.md`; the issues themselves were not re-read |
| The owner's measurements on the target box | - | 90-120 GB/s memory bandwidth against ~256 GB/s theoretical; a reported ~2x decode from mirroring Qwen Next's experts on both sockets (`docs/volta/VOLTA.md`: in another engine). `docs/volta/PLAN.md`, `docs/deepseek/PLAN.md` |

## 5. Upstream's own credits

As upstream states them (`docs/DETAILS.md`, "Credits and licenses", and the README, now `docs/STRATA_README.md`):

- Model: Qwen3.8-Flash-Next by the Qwen team; quantisations by ISTA-DASLab; the Coder (Apache-2.0 per its card; its
  support in Strata came from PR #54); Swift 1.5 by UkisAI. Their licenses apply to the weights.
- llama.cpp / ggml (MIT): the i-quant formats, the GPU dot products and dequantisers transcribed in
  `src/kernels/cuda/iq_kernels.cu`, the CPU backend linked for the i-quant experts, the `mtmd` library behind the image
  encoder, and `gguf-py` used by the tools.
- Ideas from [Splash](https://github.com/incoai/splash), [ninfer](https://github.com/Neroued/ninfer) and
  [HyperQwen](https://github.com/syv-ai/HyperQwen); references in upstream's paper. (Both Splash's and HyperQwen's
  repositories carry an Apache-2.0 `LICENSE`; upstream says ideas only.)
- The web app's font: Outfit (SIL OFL 1.1). Its Monitor tab started from a contributor's dashboard idea (PR #22).
- The experimental speed projection's vector: Qwen Community License 1.0.

Not in upstream's credits list, but stated in its code comments: the draft layer's conventions (`include/strata/core/mtp.hpp`,
`tools/mtp_rt.py`) follow vLLM 0.30.0's `qwen4_exp` MTP implementation (the header says "transcribed and validated in
`tools/mtp_probe.py`", a file that is not in this tree; vLLM is Apache-2.0 - not re-checked here).

## 6. What is the port's own

Everything added after `3906943` that is not listed above: `src/ds41`, `include/strata/ds41`, `ref/ds41` (apart from the
transcription noted in 2.5), `tools/volta`, `tools/ds41`, `tools/test_setup_volta.py`, `tools/test_setup_cpu.py`,
`cmake/ds41_*.cmake`, `cmake/check_no_vbmi.cmake`, `src/platform/numa.cpp`, `include/strata/platform/numa.hpp`, the Volta and
no-VBMI changes in the files listed in `git diff --name-status 3906943 HEAD`, and the docs under `docs/volta`,
`docs/deepseek` and `docs/USAGE_LEDGER.md` (a design note; no code implements it). MIT, as above.

## 7. How this page was checked, and what was not

Checked on 2026-10-01 from the build machine:

- Files of the first commit against upstream: each of the 522 files (all but `UPSTREAM.md`) compared with
  `https://raw.githubusercontent.com/Niko1221/Strata/9259cad4cfa3543cd3b8decab5962672b968c649/<path>`: 522 identical.
  Not checked: that upstream's tree at that commit has no further files (the GitHub API and tarballs are not reachable
  from here).
- `third_party/ggml/ggml-common.h` and `third_party/ggml/LICENSE` against llama.cpp at `3cf03257...`: identical.
- The ten DeepSeek files against `huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash` (`main`): identical.
- License tags and names of the models in section 3 from the Hugging Face API; licenses of ninfer, v100-skinny,
  1Cat-vLLM, Splash and HyperQwen from their `LICENSE` files on GitHub.
- Which files carry a ggml notice, and which port files mention ggml, llama.cpp, ninfer, 1Cat or v100-skinny: by search
  of the tree at `f1fac23`.

Not checked: the llama.cpp, vLLM-fork and hit-rate sources in section 4 beyond what the docs say; the Python packages'
licenses; that the installer's downloads still resolve to the pinned revisions.
