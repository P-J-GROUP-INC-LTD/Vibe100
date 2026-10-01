# Vibe100

**[Strata](https://github.com/Niko1221/Strata) ported to the NVIDIA V100 (Volta), plus the start of a DeepSeek-V4.1-Flash
port, for a dual-Xeon V100 workstation.**

> **Credit first.** Vibe100 is a port of **Strata** by Niko1221 and the Strata contributors (MIT). The engine design - the
> expert cache on the GPU, the CPU computing the experts the GPU does not hold, speculative decoding, the pack format, the
> server and web app - is theirs. This repository's first commit, `3906943`, is a byte-for-byte snapshot of Strata
> 0.1.31 (upstream commit `9259cad`, see [UPSTREAM.md](UPSTREAM.md)); everything after it is the port, and
> `git diff 3906943` shows exactly what changed. Upstream's own README, install guide and user manual are kept in full in
> [docs/STRATA_README.md](docs/STRATA_README.md).

> **Status: nothing has run on a V100 yet.** The code was compiled for sm_70 with CUDA 12.8 and its machine code
> inspected; the CPU code was tested on a Cascade-Lake-class VM; GPU kernel logic was checked by running the kernel
> sources on the CPU. Correctness, parity and speed have to be established on the owner's box (below). Every speed
> figure in this repository's docs is an estimate unless it says "measured".

## What this is

Strata runs Qwen3.8-Flash-Next, a ~125B-parameter mixture-of-experts model, on one consumer GPU plus system RAM: the GPU
holds the dense layers and a cache of the most-used experts, RAM holds every expert, and the CPU computes the experts the
GPU does not hold while the GPU works on the others. Upstream supports NVIDIA RTX 20 to 50 (and AMD on Linux).

The target here is a Dell Precision 7920: two Xeon Gold 6226 (Cascade Lake), 384 GiB DDR4-2666, one V100 32 GB. Four things
in upstream did not fit that box, and this port changes them:

1. The ready-made engine is built with CUDA 13, which dropped Volta; the installer could not target sm_70; and upstream's
   experimental sm_70 build would launch a prompt-attention kernel whose sm_70 body is a trap (seen in the machine code).
2. Upstream's AVX-512 expert kernels required AVX-512 VBMI, which Cascade Lake lacks.
3. The CPU expert arena was not NUMA-aware, on a two-socket machine.
4. A DeepSeek-V4.1-Flash port is planned for the same box; its first, hardware-independent part is here.

## Status by work package

| WP | What it does | Commit | Verified without a V100 | Still to run on the hardware |
|---|---|---|---|---|
| **A** platform | sm_70 is a supported CMake / installer / Docker target. A V100 compiles its engine with a CUDA 12.x toolkit (installs 12.8 if missing), needs driver 570+, is refused beside an RTX 50. The device check admits cc 7.0 and says so when the binary has no code for the card | `b529651` | built for sm_70 with CUDA 12.8; `tools/test_setup_volta.py` (39 checks, no GPU, re-run for this page: pass) | `./setup.sh` end to end; `build/strata-device --selftest`. Windows and Docker paths never run |
| **B** audit and harness | `tools/volta`: SASS gate, dispatch audit, parity runner, golden-logits comparison, decode profiler | `04a9333` | SASS gate passed on the integrated sm_70 build (0 unexplained traps; the three Volta kernels have 512 HMMA each); 74 unit tests (re-run: pass) | the tools themselves, on the V100 |
| **C** prefill GEMM | dense bf16 projections of the prompt path run on FP16 tensor cores (exact conversion, power-of-two scaling, FP32 accumulate) | `2a6b6ff` | compiled for sm_70, no spills, no traps | `gemm_volta_parity` (error vs FP64, TFLOPS) |
| **D** prompt attention | fixes the sm_70 trap-kernel dispatch; new WMMA prompt-attention kernel for KV modes fp16 / int8 / K8V4 | `bf08826` | compiled for sm_70 (512 HMMA per kernel, 245-248 registers, no spills); logic checked on the CPU (kernel source emulated, ThreadSanitizer) | `qsa_prompt_attn_parity` (32K, 1500 and the 2100/256 edge) |
| **E** CPU kernels | AVX-512 expert kernels on CPUs with VNNI but no VBMI (second build of `expert.cpp`, chosen at run time); canonical Q2_0 pack offered on Cascade Lake | `4e04395` | no-VBMI unpack proven bit-identical (exhaustive over 2^32 inputs); a ctest greps the objects for VBMI instructions; 10/10 CPU tests on a Cascade-Lake-class VM | speed on the Xeon; the `STRATA_IQ512` A/B |
| **F** NUMA mirror | one copy of the expert arena per NUMA node; each CPU worker reads its own node's copy; the GPU reads the copy on its node | `f1fac23` | faked 2-node topology: mirrored = single copy bit for bit in every pool mode; a poison test shows each worker reads its own copy; 32/32 non-GPU tests, SASS gate pass | a real two-socket box: placement, `mbind` allowed, replica pages on their node, the A/B speed |
| **DS-0** foundations | shared geometry header, contracts, per-package CMake hook | `d9721e8` | - | - |
| **DS-A** oracle | NumPy forward of DeepSeek-V4.1-Flash, validated against DeepSeek's official reference on CPU | `d37b3d5` | 102 tests (per the commit; the tests need torch, not re-run here) | real weights, layer by layer |
| **DS-B** GGUF tooling | validates the mxxm-t MXFP4 GGUF, manifest, memory plan, expert extraction, mini-GGUF fixture | `7cda903` | 99 tests (re-run: pass, 7 skipped without a ggml source) on saved headers and the mini file | `manifest.py` on the real shards |
| **DS-C** CPU experts | MXFP4 expert kernels: scalar, AVX2, AVX-512 VNNI (no VBMI) | `86ad959` | 81,000 checks against an FP64 contract and ggml's own MXFP4; ASan / UBSan / TSan clean | the benchmark on the Xeon |
| **DS-D** V100 experts | router, hit/miss split and MXFP4 hot-expert kernels for sm_70 | `7f8158b` | 16 kernels, no spills or traps; same source run on the CPU through an emulator, 162 checks | `ds41_{router,split,expert}_parity` on the V100 |

Plans and notes (no code): [docs/volta/PLAN.md](docs/volta/PLAN.md), [docs/deepseek/PLAN.md](docs/deepseek/PLAN.md),
[docs/USAGE_LEDGER.md](docs/USAGE_LEDGER.md) (a design for placing experts from weeks of use; nothing implements it),
[docs/volta/NINFER_STUDY.md](docs/volta/NINFER_STUDY.md).

## Same as upstream, and different

**Unchanged:** the model and its packs, the server, web app and API, speculative decoding, the expert cache with its
adaptive swaps and profiles, the decode kernels (integer dot products and FP32/FP16 math, which Volta has), ggml's MMQ
for the prompt's expert GEMMs (on a V100 the CUDA-core int8 path; [dispatch_audit.md](tools/volta/dispatch_audit.md)
R12b suggests an A/B with `STRATA_PREFILL_MMQ=0`), multi-GPU layer split, AMD HIP, the low-RAM modes, and everything for RTX cards: the
prebuilt engine, CUDA 13.0, driver 580. The Ampere and Turing paths behave as before (WP-D: the sm_75 to sm_120
prompt-attention machine code is unchanged; the FP16 GEMM route below is Volta-only by default).

| | Upstream Strata 0.1.31 | Vibe100 |
|---|---|---|
| GPUs | RTX 20 to 50; AMD (Linux, experimental); Pascal only with a community flag | adds **V100 / Titan V (sm_70)**. Pascal is still `-DSTRATA_EXPERIMENTAL_SM60=ON` only |
| CUDA, driver | prebuilt engine, CUDA 13.0, driver 580+ | RTX unchanged. A V100 compiles locally with CUDA 12.x (12.8 checked), driver 570+; per [VOLTA.md](docs/volta/VOLTA.md) R580 is the last branch with Volta. CMake refuses a CUDA 13 compiler for sm_70 with the reason |
| Several GPUs | RTX cards share one engine | a V100 can share one with RTX 20/30/40; **not** with an RTX 50 (CUDA 12 vs 13): the setup refuses |
| Prompt attention on sm_70 | dispatcher keyed on `cc_major` alone, so a V100 launched a trap kernel | keyed on `major*10+minor`; new `prompt_attn_volta_kernel` (WMMA m16n16k16, fp16 in, fp32 accumulate); a 4-bit KV cache still takes the FP32 kernel on every card |
| Prefill dense GEMMs | cuBLAS bf16 (CUDA cores below sm_80) | FP16 tensor-core route on Volta (7.0 <= cc < 7.5); RTX 20 keeps upstream's call unless `STRATA_PREFILL_F16_GEMM=1`; small shapes keep upstream's call everywhere; `=0` restores it |
| Long-context block scoring | tensor-core kernel on sm_80+ | unchanged: the FP32 warp kernel on Volta (correct, slower; optimise only if the profile says so) |
| CPU expert kernels | AVX-512 needed VBMI (Ice Lake / Zen 4 and newer), else AVX2 | tiers `Avx512Vbmi` / `Avx512Vnni` / `Avx2`; needs F/BW/VL/DQ/VNNI; Cascade Lake runs the no-VBMI build. i-quant AVX-512 rows stay off by default on the VNNI tier except IQ2_S (measured slower there) |
| NUMA | none: workers pinned to cores, arena placed by first touch | `--numa auto` mirrors the arena per node (Linux, 2+ nodes, full-RAM arena, room on every node) |
| Installer | prebuilt engine, CUDA 13.0 | a V100 never gets the prebuilt engine; Ubuntu 22.04/24.04 `cuda-toolkit-12-8` or `winget` 12.8 after asking; `--numa auto\|mirror\|off`; AVX-512 test without VBMI so Cascade Lake is offered the canonical Q2_0 pack when the engine is compiled from this source (a V100 always; `./setup.sh --setup --build` otherwise - the ready-made upstream engine still needs VBMI for it, so setup prepares the AVX2 native pack there and says how to get the faster one); Docker `BASE_IMAGE` build argument |
| Tools | upstream's | `tools/volta`, `tools/ds41`, `ref/ds41`, `tools/test_setup_volta.py` |
| Models | Qwen3.8-Flash-Next | the same, plus the start of DeepSeek-V4.1-Flash (no end-to-end engine) |

### New run-time switches

Environment variables unless noted; defaults checked in the source.

| Switch | Default | Effect |
|---|---|---|
| `STRATA_VOLTA_ATTN` | unset = on | cc 7.0 / 7.2: the WMMA prompt attention. `0`: the FP32 kernel (the reference for the checks). `2` / `force` (code only): the WMMA kernel on any sm_70+ card |
| `STRATA_PREFILL_F16_GEMM` | `auto` | `auto`: the FP16 tensor-core route on Volta (7.0 <= cc < 7.5). `0`: upstream's bf16 cuBLAS call. `1`: the FP16 route on any card (how an RTX 20 opts in) |
| `--numa auto\|mirror\|off` (engine and `setup.py`) | `auto` | `mirror` asks for it (the engine still says why when it cannot); `off` keeps one copy |
| `STRATA_NUMA_MIRROR` | unset | `0` / `1` overrides `--numa` for one run: the A/B switch |
| `STRATA_NUMA_GPU_NODE` | unset (read from sysfs; if the BIOS gives none the first node is assumed, and the log says so) | `N` names the GPU's NUMA node |
| `STRATA_NUMA_HEADROOM_GIB` | 6 | free memory each node must keep beyond its arena copy, or the engine keeps one copy |
| `STRATA_FORCE_AVX512_NOVBMI` | unset | `1` runs the no-VBMI build on a CPU that has VBMI, to compare the two |
| `STRATA_IQ512` | unset | `1`: the AVX-512 i-quant rows for every format on any AVX-512 CPU (the A/B on Cascade Lake) |

## Quick start on a V100 box

Details, the A/B for each switch and the known limits: [docs/volta/VOLTA.md](docs/volta/VOLTA.md).

**1. Check the card.** `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` must say **7.0** and a
driver of **570 or newer** (VOLTA.md: not past the 580 series, the last branch that lists Volta).

**2. BIOS (Dell Precision 7920 and similar).** Node Interleaving: Disabled; Sub-NUMA Clustering: Disabled; System Profile:
Performance. With interleaving on, Linux sees one node and neither the mirror nor the DeepSeek per-socket split can work.

**3. Install.** `./setup.sh` (Linux; `START-HERE.bat` on Windows is untested). It skips the prebuilt engine, finds or installs
CUDA 12.x (asks before installing 12.8), compiles the engine for sm_70 (10-20 minutes, once; VOLTA.md's figure), then
downloads and prepares the model like for any card. `./setup.sh --check` only checks card, driver and RAM and says
whether a CUDA 12 toolkit was found. CUDA 12.8 is the version the port was compiled with; not 13.

**4. Verify, in this order** — the step-by-step version, with exact commands and what to send back after each step, is
[docs/volta/RUNBOOK.md](docs/volta/RUNBOOK.md); the gates are defined once in [docs/volta/PLAN.md](docs/volta/PLAN.md) section 3:

| Phase | Tool | Gate |
|---|---|---|
| 0 the box and the card: BIOS, OS settings, the R580 proprietary driver (not `-open`: NVIDIA's open modules do not drive Volta) | `nvidia-smi --query-gpu=name,compute_cap --format=csv`, `numactl -H` | **Gate 0:** the card says 7.0, Linux sees two NUMA nodes |
| 1 build for sm_70 and run every GPU parity program and the SASS audit | `.venv/bin/cmake --build build --target strata-device && build/strata-device --selftest`, then `tools/volta/run_parity.sh --ggml-dir third_party/llama.cpp --require-v100` | **Gate P:** the script prints `GATE P (...): PASS` |
| 2 correctness: Volta paths vs FP32 reference paths on the same card | `.venv/bin/python tools/volta/golden_compare.py --engine-config strata-<model>.json --prompt-name long --tail 512` | **Gate Q:** top-1 >= 99%, perplexity within 2%, no NaN/inf |
| 3 profile one decode step, then tune (A/B switches, `./setup.sh --calibrate`) | `tools/volta/profile_decode.sh --config strata-<model>.json --strict` | **Gate 1:** IQ3 decode >= 40 tok/s with Gates P and Q passed; if not, stop and reassess Volta before any DeepSeek engine work |

Each tool explains itself with `--help`. `tools/volta/sass_audit.py` gates a build's sm_70 machine code;
`tools/volta/compile_one.py` compiles one file for sm_70.

**5. NUMA A/B.** The startup log's `NUMA:` lines say what the mirror did. Compare the default, `STRATA_NUMA_MIRROR=0`, and
`STRATA_NUMA_MIRROR=0 numactl --interleave=all`; do not combine the mirror with `numactl --interleave`.

## Expected performance

None of these is a measurement on a V100 by this port.

| Figure | What it is | Basis | Source |
|---|---|---|---|
| IQ3 decode >= 40 tok/s | the pass mark of Gate 1 | the owner's brief; "a target, not a prediction" | [volta/PLAN.md](docs/volta/PLAN.md) section 4 |
| ~2x decode from the NUMA mirror | someone mirrored Qwen3.8-Flash-Next's experts on both sockets of this class of box | their report, in another engine; not this engine | [VOLTA.md](docs/volta/VOLTA.md) |
| ~15 vs ~125 TFLOPS | bf16 GEMM on CUDA cores vs FP16 tensor cores, V100 (VOLTA.md: "about 8x") | quoted in the commit message, not measured; `gemm_volta_parity` prints the achieved figure | `2a6b6ff` |
| 5.8-6.2 GB/s per thread vs 3.3 | canonical Q2_0 expert kernel, AVX-512 no-VBMI vs AVX2 rows | measured on a Cascade-Lake-class VM, indicative | `4e04395` |
| ~60-90 tok/s ceiling for DeepSeek at 50% expert hit rate; "well under half of it in practice" | bandwidth budget: GPU ~900 GB/s, CPU ~200 GB/s NUMA-local | arithmetic from the model's sizes; "not a prediction" | [deepseek/PLAN.md](docs/deepseek/PLAN.md) section 2 |
| 0.5-0.65 expert hit rate for ~26 experts per layer on a V100 32 GB | extrapolated from third-party hit-rate reports | "Measure." | [RESEARCH.md](docs/deepseek/RESEARCH.md) section 7 |
| ~150 us per layer for 6 hit experts at T=1 | MXFP4 hot-expert kernels, bandwidth-bound | estimate in the commit message | `7f8158b` |
| ~30 tok/s base, 45-123 with MTP on real text | other people's V100 runs of Qwen3.8-27B NVFP4 (ninfer forks), not Strata | the authors' own numbers | [NINFER_STUDY.md](docs/volta/NINFER_STUDY.md) |

Upstream's figures (for example 93 tokens/s on an RTX 5070) are RTX numbers and say nothing about a V100.

## DeepSeek-V4.1-Flash (started: DS-0; the engine is not written)

DeepSeek-V4.1-Flash is a 40-layer MoE (384 routed experts per layer, 6 active, MQA attention with compressed sparse
selection, Engram n-gram tables, 4-copy hyper-connections; MIT-licensed weights and code). The plan keeps Strata's idea:
hot experts in V100 VRAM, the rest computed by the CPU from RAM, in the format DeepSeek released them, **MXFP4**
(18,800,640 bytes per expert, GGML-compatible, no re-quantisation).

**Target file:** [mxxm-t/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/mxxm-t/DeepSeek-V4.1-Flash-GGUF) (MXFP4 experts, Q8_0
attention, 12 shards + a DSpark sidecar). The vcruz305 Q2_K...Q8_0 GGUFs are refused by the tooling (experts not MXFP4).

**What exists (DS-0):** the contract ([CONTRACTS.md](docs/deepseek/CONTRACTS.md), `include/strata/ds41/geometry.hpp`);
the NumPy oracle `ref/ds41`; GGUF tooling `tools/ds41` (manifest, expert layout, memory plan: on a V100 32 GB with 384 GiB it
plans ~1,150 cached experts, 269 GiB of experts in RAM split across the sockets, ~96 GiB left for Engram's page cache;
the reserves are the plan's numbers, not measurements); MXFP4 CPU kernels `src/ds41/cpu`; V100 router and expert kernels
`src/ds41/cuda`. Nothing in `src/program`, `src/core`, `src/prefill`, `serve` or `setup.py` refers to it: the `strata` binary
cannot run DeepSeek, and no DeepSeek tensor data has been read (only GGUF shard headers).

**Next ([PLAN.md](docs/deepseek/PLAN.md) section 4):** *DS-1*, one correct token (dense GEMVs, MQA decode, compressor /
indexer, mHC, Engram, a decode loop, layer-by-layer comparison with the oracle on real weights; gate: top-1 >= 99% vs the
reference over 500 tokens, no NaN/inf). *DS-2*, usable (prefill, Engram on SSD with prefetch, expert cache and adaptive
swaps, server integration, the usage ledger; gate: a measured hit rate, and below ~40% reassess). *DS-3*, fast (routing
traces, NUMA placement, DSpark drafting). The plan itself says DeepSeek tuning only makes sense after the Volta port
passes Gate 1 on the card.

**A bug in DeepSeek's reference, found by the oracle (not confirmed with DeepSeek).** On decode steps where a ratio-2 index-K owner's compressor group is
incomplete (layers 2, 8, 14: every other token), the official `model.py` scores the indexer against the last published
index-K cache, which is layer 20's, instead of the layer's own, and the Reuse layers that take their owner's selection
(`model.py:722-736`) inherit it, 18 ratio-2 layers in all. It changes the top-512 selection only once a layer holds more
than 512 compressed positions (context above ~1,024 tokens); prefill and training are unaffected. The port follows the
evident intent (each owner scores against its own cache); the oracle implements both (`stale_index_k=True` reproduces the
shipped code bit for bit). Recorded as decided in [CONTRACTS.md](docs/deepseek/CONTRACTS.md); to revisit if DeepSeek's or
llama.cpp's production decode does otherwise.

## Repository map

| Path | Contents |
|---|---|
| `src/`, `include/strata/` | the engine (upstream) with the port's changes; **new:** `src/ds41`, `include/strata/ds41`, `src/platform/numa.cpp` |
| `setup.py`, `setup.sh`, `SETUP.bat`, `START-HERE.bat`, `Dockerfile`, `docker-entrypoint.sh` | installers (V100 path added in `setup.py`, `Dockerfile`) |
| `serve/`, `chat.py` | server, web app, terminal chat (upstream, unchanged) |
| `tools/` | pack and benchmark tools (upstream); **new:** `tools/volta` (audit, parity, correctness, profiling), `tools/ds41` (GGUF tooling), `tools/test_setup_volta.py` |
| `ref/` | `ref/load.py` (upstream); **new:** `ref/ds41`, the DeepSeek NumPy oracle |
| `tests/`, `bench/`, `data/`, `cmake/` | upstream tests, benchmark results, data, CMake modules; **new:** `tests/core/numa_arena_test.cpp`, `cmake/ds41_*.cmake`, `cmake/check_no_vbmi.cmake` |
| `third_party/` | `ggml/` (upstream, MIT); **new:** `deepseek-v41-flash-reference/` (DeepSeek's official reference, MIT) |
| `docs/` | upstream docs plus `volta/`, `deepseek/`, `USAGE_LEDGER.md`, `PROVENANCE.md`, `STRATA_README.md` |

## Documentation

| Document | What it is |
|---|---|
| [docs/STRATA_README.md](docs/STRATA_README.md) | upstream's README: install, models, usage, troubleshooting |
| [docs/DETAILS.md](docs/DETAILS.md) | upstream's reference, with the port's "CPU kernels" and "NUMA mirroring" sections |
| [docs/volta/VOLTA.md](docs/volta/VOLTA.md) | the V100 user guide: install, switches, NUMA, checks, known limits |
| [docs/volta/PLAN.md](docs/volta/PLAN.md) | what upstream does on sm_70, the work packages, phases and gates |
| [docs/volta/NINFER_STUDY.md](docs/volta/NINFER_STUDY.md) | what the ninfer V100 forks offer, and licensing |
| [tools/volta/dispatch_audit.md](tools/volta/dispatch_audit.md) | every compute-capability decision in the code and what a V100 gets |
| [docs/deepseek/PLAN.md](docs/deepseek/PLAN.md), [RESEARCH.md](docs/deepseek/RESEARCH.md), [CONTRACTS.md](docs/deepseek/CONTRACTS.md) | the DeepSeek port: plan and gates, the spec read from the primary files and the GGUF, the shared contracts |
| [tools/ds41/README.md](tools/ds41/README.md), [ref/ds41/README.md](ref/ds41/README.md) | the GGUF tooling; the oracle and its tolerances |
| [third_party/deepseek-v41-flash-reference/README.md](third_party/deepseek-v41-flash-reference/README.md) | what was vendored from DeepSeek |
| [docs/USAGE_LEDGER.md](docs/USAGE_LEDGER.md) | design note: expert placement learned from use |
| [docs/PROVENANCE.md](docs/PROVENANCE.md), [UPSTREAM.md](UPSTREAM.md) | where everything comes from |
| [MULTI_GPU](docs/MULTI_GPU.md), [AMD_HIP](docs/AMD_HIP.md), [ORCA](docs/ORCA.md), [UNSLOTH_Q4](docs/UNSLOTH_Q4.md), [COMMUNITY_BENCHMARKS](docs/COMMUNITY_BENCHMARKS.md), [paper](docs/paper/Strata-Paper.pdf) | upstream's other guides, benchmarks and paper |

## Provenance and license

The tree is upstream Strata 0.1.31 (MIT) plus the port. llama.cpp / ggml (MIT, pinned commit `3cf0325`) is fetched at build
time; `third_party/ggml/` and 17 source files carry ggml's MIT notice. DeepSeek's reference implementation is vendored
unmodified under its MIT license. The ninfer forks, v100-skinny, 1Cat-vLLM and the llama.cpp DeepSeek pull requests were
studied and nothing was copied from them. The models are not in the repository; each model's own license applies. The full
table, with versions and what was checked: [docs/PROVENANCE.md](docs/PROVENANCE.md).

**License:** [MIT](LICENSE), upstream's file unchanged; the port's additions are released under it too. Exceptions with
their own licenses: `third_party/ggml` (MIT, ggml authors), `third_party/deepseek-v41-flash-reference` (MIT, DeepSeek),
the web app's font (SIL OFL 1.1) and the experimental speed projection's vector (Qwen Community License 1.0).
