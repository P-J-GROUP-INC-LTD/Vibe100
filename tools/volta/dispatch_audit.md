# Compute-capability dispatch audit - what a V100 (cc 7.0, `__CUDA_ARCH__` 700) gets

Scope: every place the code base decides something from the GPU architecture, at run time (`cudaDevAttrComputeCapability*`,
`cudaGetDeviceProperties().major/.minor`, `cc_major[]` caches, ggml's `cc` in `src/prefill/ggml_cuda_host.cu`, `setup.py`'s
arch checks) and at compile time (`__CUDA_ARCH__` branches, the `STRATA_*_SM80` macros, `__HIP_DEVICE_COMPILE__`).

Method.  Greps over `src/ include/ tests/ bench/ tools/ serve/ setup.py CMakeLists.txt cmake/ Dockerfile *.sh`:
`ComputeCapability|cudaGetDeviceProperties|\.major|\.minor|cc_major|compute_cap|__CUDA_ARCH__|STRATA_[A-Z0-9_]*SM[0-9]+|
STRATA_EXPERIMENTAL_SM60|cudaFuncSetAttribute|MaxSharedMemory|asm volatile|__shfl\(|__ballot\(|__activemask|volatile`, then each hit read
in context.  The result was checked against the machine code, not only the source: `strata_sm70.sass` (the upstream 0.1.31 full build for
sm_70 with CUDA 12.8, `cuobjdump -sass -arch sm_70`) and `tools/volta/sass_audit.py` over the same build.  ggml claims were checked in the
pinned llama.cpp (`3cf03257f219afbe7334045ff7c6a06ac68c627d`, the pin in `CMakeLists.txt`).

Line numbers are those of the upstream snapshot the port started from (commit `3906943`, "Import upstream Strata 0.1.31") unless marked "now".
WP-A's changes were committed while this was written (`b529651`); WP-C and WP-D were still editing their files, so for those the status
column says what their working tree contained when it was read.  Re-run the greps after the merge.

Verdicts: **OK** correct on sm_70 as is, **BUG** wrong or crashing on sm_70, **SLOW** correct but not what a V100 should be doing,
**UNUSED** the branch is never taken / the code is never called.  Path: **decode** hot path, **prefill**, **cold** (startup, installer,
logging), **test**.

## Summary

| Verdict | Count | Items |
|---|---:|---|
| BUG (upstream) | 7 | R1 device check refuses cc < 7.5; R7 the prompt-attention dispatcher launches a trap stub on a V100; R7b (found by the audit round) an sm_70-only build run on a cc >= 7.5 card reaches the same trap stubs; R14 / R14c / R15 / R16 installer, CMake and Docker refuse or cannot build sm_70 |
| SLOW | 3 | R8 QSA block scores on the FP32 warp kernel (correct, by design); R13 cuBLAS bf16 GEMM on CUDA cores; R12b ggml MMQ is always used for the prompt's expert GEMMs although llama.cpp itself would not on a V100 (an A/B, `SLOW?`) |
| UNUSED | 4 | R3 (Windows-only code); C6 `native_qsa_score.cu` (no caller); C12 / C13 the nvfp4 and unused-width ggml MMQ instantiations (trap stubs the host never selects) |
| OK | the rest | decode kernels, `dp4a.hpp`, `fused_gr.cu` shared-memory opt-in, ggml's host/device configuration, bf16/f16 bit helpers, warp-synchronous code |

Every BUG has an owner and is fixed in that owner's files: R1, R14-R16 by WP-A (committed, `b529651`), R7 by WP-D and R13 by WP-C (their working
trees, read at audit time), R7b in the audit round (below).  **No BUG was found in a file nobody owns**, so this audit changed no source file.

Two findings that are not arch dispatch but shape how the port can be checked on the target box (a dual Cascade Lake Xeon, per the plan):

* **Native (IQ) packs write no `--dump-logits` rows** (`generate.cpp:5410`: `if (native_pack) { spec_pos = pos; break; }` skips the token loop, the only
  writer of the dump), and upstream's Cascade Lake could not run the Q2_0 pack (its kernels required AVX-512 VBMI; WP-E removed that
  requirement - the canonical pack now runs there and DOES write the dump).  The native packs remain the common choice and have no logits dump.  `golden_compare.py` therefore has a second teacher source that works for them: `strata --serve` with
  `STRATA_LOGPOS` (`generate.cpp:4757-4765`, `Verifier::window_logprobs`), which writes per-position log-probabilities for a prompt part read through the
  verify windows - see its docstring.
* A native pack generates only through the verify-window loop (the token loop is skipped); `setup.py` always passes `--spec 4 --mtp ...` for it, so the profile
  script and the harness keep the config's speculative-decoding flags for a native pack.

## Runtime decisions

| # | file:line (HEAD) | condition | V100 (cc 7.0) gets | verdict | path | owner / status |
|---|---|---|---|---|---|---|
| R1 | `src/core/device.cu:147` (now `:187`) | `cc_major*10 + cc_minor < kMinCc`; `kMinCc` = 75, or 60 under `STRATA_EXPERIMENTAL_SM60` (`:140-146`) | in the plain build: `CudaError "Strata needs compute capability 7.5 or newer"` - the engine will not start on a V100 | **BUG** | cold | WP-A, **fixed in `b529651`**: `kMinCc = 70` (60 with the flag) and a new `missing_code_problem()` that detects "binary has no code for this GPU" via `cudaFuncGetAttributes` (extended in the audit round to refuse code that is *older* than the card, R7b) |
| R2 | `src/core/device.cu:115-119`, `include/strata/core/device.hpp:24` | `DeviceInfo.cc_major/.cc_minor` filled from `cudaDeviceProp` | consumed only by R1 and `src/core/device_main.cpp:40` (prints it) | OK | cold | - |
| R3 | `src/core/pinned.cu:263` | `cudaGetDeviceProperties(&p, dev)` for `p.luid` | inside `#ifdef _WIN32` (`sliced_pin_limit`): not compiled on Linux | UNUSED | cold | - |
| R4 | `src/program/generate.cpp:2026` | `cudaGetDeviceProperties(&dp, 0)` | only `dp.name` in a `--split-skip-if-fits` log line | OK | cold | - |
| R5 | `src/program/generate.cpp:2176-2177` | `cudaDevAttrMultiProcessorCount` x `cudaDevAttrClockRate` | speed estimate for `--layer-split auto` (multi-GPU only): 80 SMs x 1.38 GHz = 110; single V100: not executed | OK | cold | - |
| R6 | `src/kernels/s2_expert_grouped_parity.cpp:531` | `prop.l2CacheSize` | `--bench` only: sizes the blob rotation (V100: 6 MB L2) | OK | test | - |
| R7 | `src/kernels/cuda/qsa_prompt_attn.cu:693-709, 728` | `cc_major < 7 -> false; turing = cc_major < 8; turing -> launch<1>` (and `launch<0>`, `launch<3>` for fp16 / K8V4 KV) | **`major == 7` includes Volta**, so a V100 takes the "Turing" route and launches `prompt_attn_kernel<N>`, whose `mma16816` compiles to `__trap()` for sm_70 (STRATA_PA_SM80 = 0 and `__CUDA_ARCH__ < 750`, lines `:30-45`). All four kernels contain `BPT.TRAP` in the sm_70 SASS (checked). The first prompt chunk of a QSA layer with an int8 / fp16 / K8V4 KV cache kills the process with "unspecified launch failure" | **BUG** | prefill | WP-D, fixed in its working tree: `select_impl()` keys on `major*10 + minor` (`cc < 75` -> `prompt_attn_volta_kernel<KV>` (WMMA) or, with `STRATA_VOLTA_ATTN=0`, `false` -> the caller's FP32 `qsa_decode_attn_batch`; `cc < 80` -> the v1 mma kernel; else cp.async). Read at audit time; `sass_audit` on its object: the three Volta kernels have 512 HMMA each and no trap, the four old kernels keep their trap and are allowlisted because the dispatcher no longer reaches them below 7.5 / 8.0 |
| R7b | `src/kernels/cuda/qsa_prompt_attn.cu` `select_impl` (the cc >= 7.5 branches), `src/core/device.cu` `missing_code_problem` | **found by the audit round.** A build whose code is **sm_70-only** (`-DCMAKE_CUDA_ARCHITECTURES=70`: sm_70 SASS + compute_70 PTX) run on a card of **cc >= 7.5**. The runtime picks the best code the binary has for the card: on a **7.5** card the sm_70 SASS runs natively (SASS is forward-compatible within a major version), on an **8.x** card there is no sm_8x code, so the driver JIT-compiles the compute_70 PTX. Either way every kernel is the `__CUDA_ARCH__ == 700` body | `select_impl` keyed on the card's compute capability (`cc >= 75` -> the v1 kernel, `cc >= 80` -> the cp.async one) and launched `prompt_attn_kernel<N>` / `prompt_attn_i8_kernel`, whose Turing / Ampere bodies are `__trap()` stubs in sm_70 code (R7 / C2 / C3): the same "unspecified launch failure" on the first long prompt as R7, now on an RTX card with a V100-only engine. | **BUG** | prefill | **Fixed in the audit round, twice over:** (1) the attention dispatcher asks the runtime which code each kernel got (`cudaFuncGetAttributes`: `ptxVersion`) and gates the v1 / i8 kernels on it, taking the Volta WMMA kernel when the Turing / Ampere body is not what would run; (2) `device.cu` refuses at start-up to run a build whose code is older than the card, naming the architectures it was built for (`STRATA_CUDA_ARCHS`). Read both against the final source; `qsa_prompt_attn_parity` has the matching cases. Remedy for a mixed box: build `"70;75"` (V100 + RTX 20) or `"70;86"` etc. |
| R8 | `src/kernels/cuda/qsa_select.cu:511-522` | `qsa_block_scores_tc`: `cc_major < 8 -> return false` | the FP32 warp kernel `qsa_block_scores` (one warp per (query, block)) - correct; `block_scores_tc_kernel` (3xTF32 `mma.sync`, needs sm_80, `:163-190`) is a trap stub in the sm_70 code and is never launched (caller `src/prefill/prefill.cpp:1399-1403` falls back on `false`) | **SLOW** (by design) | prefill | nobody; Phase-3 candidate "Volta HMMA block-score scorer", only if Phase 0 shows it matters (the select cost grows with context x queries: look at `STRATA_PREFILL_TIMING` at 32K+) |
| R8b | `src/core/layer.cpp:933`, `src/core/verify.cpp:631` | decode / verify windows call `qsa_block_scores` (never the `_tc` one) | the same FP32 kernel on every arch | OK | decode | - |
| R9 | `src/kernels/cuda/fused_gr.cu:565-600` | `small_tile = cc_maj*10 + cc_min == 75` (TILEV 1280, Turing's 64 KB opt-in); `usable = (cc >= 7 && optin > 0) ? optin : per_block`; `capacity = usable / (tile*4)`; clamp to `kFusedGrMaxT` = 8 | V100: `small_tile` false -> TILEV 2560; `cudaDevAttrMaxSharedMemoryPerBlockOptin` = 98304 (Volta: 96 KB) >= `8*2560*4 = 81920` -> `cudaFuncSetAttribute(gr_down_multi_kernel<2560>, ..., 81920)` succeeds, capacity = 9 -> clamped to 8 -> **one launch of 8 tokens, no slicing**. One 80 KB block per SM (96 KB per SM): the grid is `DOWN_BLOCKS + 1` = 41 blocks on 80 SMs - fine. The `STRATA_GR_V3=1` variant (`:520-531`): `need1 = 81920 <= 98304` -> S = 1 | OK | decode (`fused_gr_read_multi`: `core/verify.cpp:496`, `core/mtp.cpp:461,519`) | - (the 98304 is NVIDIA's documented Volta limit; `strata-device` / the first run prints the real attribute) |
| R10 | `src/kernels/cuda/qsa.cu:675` | `cudaDevAttrMaxSharedMemoryPerBlockOptin` vs `(max_ids + 32) * 4` bytes | `max_ids` = `qsa_selection_width(kTopkMaxCells)` = 2051 -> 8.3 KB: below the 48 KB default, opt-in is a no-op | OK | decode (reference path `--no-fast-attn`) | - |
| R11 | all `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` sites: `fused_gr.cu`, `qsa.cu`, `qsa_select.cu:532`, `qsa_prompt_attn.cu`, ggml `CUDA_SET_SHARED_MEMORY_LIMIT` | requested dynamic shared memory vs the device opt-in | every one fits 96 KB: fused_gr 80 KB; qsa_select 49,920 B; `prompt_attn_volta_kernel` `static_assert(sizeof(SmemV) <= 98304)` (about 39 KB, two blocks per SM); ggml MMQ `mmq_get_nbytes_shared <= smpbo` is checked per tile width by the host (R12) | OK | decode / prefill | - |
| R12 | `src/prefill/ggml_cuda_host.cu:83` (+ `:84-88`) | `d.cc = 100*prop.major + 10*prop.minor`; `d.smpbo = sharedMemPerBlockOptin`; `d.nsm`, `d.warp_size` | `cc = 700`, `smpbo = 98304`, `nsm = 80`. **Evidence section below**: ggml's host selection (`ggml_cuda_mmq_get_config`, `mul_mat_q_switch_J`) and device config use the same "ampere" table for any cc >= 700; MMQ on sm_70 uses the dp4a data layout (no `TURING_MMA_AVAILABLE`), 0 HMMA in the whole sm_70 build; the 144 trap instantiations are exactly the (type, width, fallback) triples absent from that table, which the host never picks | OK | prefill | - |
| R12b | `src/prefill/prefill.cpp:450-451` (`mmq_plan()`), `src/prefill/moe_mmq.cu:Context::run` | `STRATA_PREFILL_MMQ` unset or != 0 -> MMQ for every supported expert type (the Strata Q2_0 pack always) | Strata calls `mul_mat_q_case` directly. llama.cpp's own gate `ggml_cuda_should_use_mmq` (`ggml-cuda/mmq.cu:266-335`) would, for an NVIDIA card with FP16 tensor cores but no int8 ones (`!turing_mma_available`, `fp16_mma_hardware_available`), use MMQ only for batches `< MMQ_DP4A_MAX_BATCH_SIZE` (64) and cuBLAS FP16 otherwise. On a V100 Strata therefore always runs the prompt's expert GEMMs on the **dp4a CUDA-core path** (V100 has no int8 tensor cores: IMMA starts at Turing), where `STRATA_PREFILL_MMQ=0` (dequantize to FP16 + cuBLAS, HMMA: `Gemm::f16`, `prefill.cpp:1708-1711`) may be faster for the large per-expert batches. Unmeasurable here | **SLOW?** | prefill | nobody; **A/B on the box**: `STRATA_PREFILL_MMQ=0` vs default on the long prompt, check prefill tok/s and PPL (golden_compare) |
| R13 | `src/prefill/gemm.cu:370-373` | `Gemm::bf16`: `cublasGemmEx(..., CUDA_R_16BF, ..., CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT)` | bf16 has no tensor-core GEMM below sm_80: the dense prefill projections (attention, SSM, shared expert, router, PLE, hyper-connection read/write: `prefill.cpp:820,822,899,1249,1255,1531`) run on CUDA cores (~8x below HMMA peak), and on some cuBLAS versions the call may be refused (unverified without a card) | **SLOW** | prefill | WP-C, fixed in 2a6b6ff: **Volta only**, `7.0 <= cc < 7.5` (everything else keeps the upstream call - Pascal, **Turing**, Ampere and newer - and so do small shapes; Turing opts in with `STRATA_PREFILL_F16_GEMM=1`) -> bf16 -> FP16 (exact for in-range values, per-chunk power-of-two scaling) -> FP16 HMMA with FP32 accumulate; `STRATA_PREFILL_F16_GEMM` auto / 0 / 1; Turing and Ampere+ unchanged bit for bit. **Audit round:** the scale is now signed (`k = E_max - 141`, clamped at -63: a chunk of small values is scaled UP, so every element within 2^28 of the chunk's maximum converts exactly - no fp16-subnormal loss for small-valued weights), and the `auto` default no longer switches Turing. `Gemm::f16` (`:387`, `CUDA_R_16F`) is already tensor-core on Volta |
| R14 | `setup.py:364-365, 484` | `int(g["arch"]) < 75` -> "not supported - older than the RTX 20 series"; `fail("none of your GPUs can run Strata")` | the installer rejects a V100 | **BUG** | cold | WP-A, **fixed in `b529651`** (a V100 is admitted; setup builds locally) |
| R14b | `setup.py:432-456, 1041-1091` (`engine_archs`, prebuilt-engine check) | the prebuilt engine's `BUILD.json` `archs` lack 70 | falls through to building from source (correct) - but see R14c | OK | cold | WP-A |
| R14c | `setup.py:1183-1231` | `need_cuda = (13,0) if max(archs) >= 120 else (12,0)`, but the install branch always runs `cuda-toolkit-13-0` (`:1231`) / `winget Nvidia.CUDA 13.0` (`:1210`); `:1302-1309` passes `-DCMAKE_CUDA_ARCHITECTURES=70` without the flag upstream's CMake needs | CUDA 13 has no `compute_70`: the auto-installed toolkit cannot build for a V100 | **BUG** | cold | WP-A |
| R15 | `CMakeLists.txt:89-93` | `_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM60 -> FATAL_ERROR` | configure refuses `-DCMAKE_CUDA_ARCHITECTURES=70` | **BUG** | cold | WP-A, **fixed in `b529651`**: 70 accepted without the flag; CUDA >= 13 refused for < 75 with the remedy; `STRATA_CUDA_ARCHS` passed to `device.cu` |
| R16 | `Dockerfile:58, 76` | `ARG CUDA_ARCHITECTURES=75;80;86;89;120`, CUDA 13 base only | no sm_70 code, no CUDA 12 toolchain | **BUG** | cold | WP-A, **fixed in `b529651`**: `BASE_IMAGE` / `CUDA_ARCHITECTURES` build-args select a CUDA 12.8 base and `70` (the default image stays CUDA 13 for RTX) |
| R17 | `src/kernels/qsa_prompt_attn_parity.cpp`, `src/prefill/gemm_volta_parity.cpp` | print / honour the arch and `STRATA_VOLTA_ATTN` / `STRATA_PREFILL_F16_GEMM` | test programs (WP-D, WP-C) | OK | test | - |
| R18 | `serve/server.py`, `chat.py`, `tools/*.py`, `docker-entrypoint.sh`, `setup.sh` | no compute-capability decisions (only `nvidia-smi` numbering for `CUDA_VISIBLE_DEVICES`) | - | OK | cold | - |

## Compile-time decisions

| # | file:line (HEAD) | condition | sm_70 build gets | verdict | path | owner / status |
|---|---|---|---|---|---|---|
| C1 | `qsa_prompt_attn.cu:30-34` | `STRATA_PA_SM80` = `__CUDA_ARCH__ >= 800` (or host pass) | 0 | - | - | - |
| C2 | `qsa_prompt_attn.cu:43-52` (`mma16816`) | `!STRATA_PA_SM80 && __CUDA_ARCH__ < 750` -> `__trap()`; `750..799` -> two `mma.m16n8k8`; `>= 800` -> `mma.m16n8k16` | `__trap()`: `prompt_attn_kernel<0/1/3>` and `prompt_attn_i8_kernel` are trap-carrying kernels in sm_70 SASS (confirmed: 4 of them in the 153-kernel trap list) | OK **if** R7 holds | prefill | guarded by R7's dispatcher (WP-D); allowlisted in `trap_allowlist.txt` |
| C3 | `qsa_prompt_attn.cu:394-408` (`cp_async16` etc.) | `!STRATA_PA_SM80` -> `__trap()` / no-op | same kernels (the i8 kernel needs sm_80) | OK **if** R7 holds | prefill | WP-D |
| C4 | `qsa_prompt_attn.cu` now: `STRATA_PA_VOLTA` (`nvcuda::wmma`, CUDA only) and `:1128` `__trap()` for a pre-Volta device pass | sm_70: the WMMA kernel with real code; the trap is in a branch that is not compiled for sm_70 | OK | prefill | WP-D (SASS-verified: no `BPT.TRAP` in `prompt_attn_volta_kernel<0/1/3>`) |
| C5 | `qsa_select.cu:163-190` (`STRATA_SEL_SM80`) | `tf32_hi` = identity, `mma_tf32` = `__trap()` when `__CUDA_ARCH__ < 800` | `block_scores_tc_kernel` is a trap stub (1 of the 153) | OK **if** R8 holds | prefill | guarded by `qsa_block_scores_tc` (R8); allowlisted |
| C6 | `native_qsa_score.cu:36, 64` | `__CUDA_ARCH__ >= 800`: ldmatrix + tf32 `mma`; `< 800`: one warp, FP32 FMAs, one row per lane | the FP32 fallback; **no trap** | UNUSED | - | nobody. Verified: `native_qsa_score()` / `_enabled()` / `_set_enabled()` have **no caller** in `src/` or `include/` (only the declaration, the definition and the HIP-only `tests/hip/native_qsa_score.cpp`); `nm -C strata` has no `native_qsa_score` symbol and the `strata` SASS has no `score_kernel` (the object is only in `libstrata_kernels.a`, and the linker did not pull it in). Out of scope |
| C7 | `include/strata/kernels/dp4a.hpp:23` | `__CUDA_ARCH__ < 610` -> software `strata_dp4a` | 700 -> hardware `__dp4a` (the sm_70 SASS has ~144 k `IDP` instructions: `IDP.4A`), every i-quant MMVQ / MMQ-adjacent kernel | OK | decode | - |
| C8 | `include/strata/kernels/dp4a.hpp:37` | `__CUDA_ARCH__ < 700` -> `strata_spin_pause()` spins without sleeping | 700 -> `__nanosleep(100)` (doorbell polls: `elementwise.cu:214-218`, `verify_kernels.cu:427,475`: all single-thread `<<<1,1>>>` kernels) | OK | decode | - |
| C9 | `include/strata/kernels/bf16_bits.hpp:39,53`, `f16_bits.hpp:39,96` | `__HIP_DEVICE_COMPILE__` only | CUDA: raw-bit bf16 -> fp32 (no `__nv_bfloat16` anywhere in `src/`: grep) and the f16 helpers - arch-independent | OK | decode / prefill | - |
| C10 | `src/core/device.cu:140` | `#if defined(STRATA_EXPERIMENTAL_SM60)` | R1 | - | - | WP-A |
| C11 | ggml `ggml-cuda/mmq.cuh:252-258, 276-285` | host `ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA` / device `__CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA` -> `ggml_cuda_mmq_get_config_ampere` | the same table on both sides (host and device agree) | OK | prefill | - |
| C12 | ggml `quantize.cu:132, 329` | `#if defined(BLACKWELL_MMA_AVAILABLE)` else `NO_DEVICE_CODE` | `quantize_mmq_nvfp4<...>` x 4 are trap stubs; Strata never launches them (`moe_mmq.cu: supported()` lists no NVFP4; `quantize()` calls `quantize_mmq_q8_1_cuda` only) | UNUSED | prefill | allowlisted |
| C13 | ggml `mmq.cuh` kernel prologue (`:965`) | `ggml_cuda_mmq_get_config(type, J, fallback).type == GGML_TYPE_COUNT` -> `NO_DEVICE_CODE; return;` | 144 `mul_mat_q<type, J, fallback>` instantiations (9 types x 16 widths) are trap stubs; never launched (host skips them, see evidence) | UNUSED | prefill | allowlisted with the exact width sets |
| C14 | `CMakeLists.txt` `--use_fast_math` for the `native_*.cu` files and `strata_mmq` (`-use_fast_math`) | arch-independent | - | OK | - | - |
| C15 | `src/kernels/cuda/gr.cu:175,188` (comments only) | FP64 reductions are slow on GeForce Blackwell | V100 has 1:2 FP64 - no issue | OK | decode | - |

## ggml MMQ through the shim: the evidence

The question: Strata's shim reports `cc = 100*major + 10*minor` (R12), so a V100 is `cc = 700`; the MMQ templates were compiled for tile widths
`J` = 8, 16, ..., 128 in two variants (`fallback` = true / false) for 9 quantization types, and 144 of those instantiations are trap stubs in
the sm_70 code.  Does the host ever launch one?

1. **Host selection** - `ggml-cuda/mmq.cuh` `mul_mat_q_switch_J` (`:1478-1560`): for `J = 8..128 step 8` it fetches
   `ggml_cuda_mmq_get_config(type, J, fallback, cc)` and `continue`s when `config.type == GGML_TYPE_COUNT` or when
   `mmq_get_nbytes_shared(config, cc) > smpbo`; it launches the `J` with the fewest column tiles among the rest (`default:` aborts, so an empty
   set would be loud, not silent).  `mul_mat_q_case` picks `fallback` from `nrows_x % 128`.
2. **The config for cc 700** - `ggml_cuda_mmq_get_config` (`:230-259`): not AMD, not Blackwell, and
   `ggml_cuda_highest_compiled_arch(700) >= GGML_CUDA_CC_VOLTA` -> `ggml_cuda_mmq_get_config_ampere` (`mmq-config-ampere.cuh`).  The build defines
   `__CUDA_ARCH_LIST__ = 700` (`--generate-code=arch=compute_70,code=[compute_70,sm_70]`), so `ggml_cuda_highest_compiled_arch(700) = 700`.
   The device-side `ggml_cuda_mmq_get_config` (`:261-286`) uses `__CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA` -> the same table: host and device agree.
3. **The table** lists, for each of the nine types Strata instantiates (Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_0 - the
   `foreach` in `CMakeLists.txt`), `fallback = true`: J in {8, 16, 32, 64, 128} and `fallback = false`: J in {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128}
   (identical for all nine - checked by script).  Everything else has `type == GGML_TYPE_COUNT`.
4. **The match.**  Parsing the table and the sm_70 SASS of the upstream build: the set of `mul_mat_q<(ggml_type)T, J, fallback>` kernels that contain
   `BPT.TRAP` is **exactly** the set of (T, J, fallback) not in the table: expected 144, observed 144, symmetric difference empty.
   (J in {24, 40, 48, 56, 72, 80, 88, 96, 104, 112, 120} for `fallback = true`; {56, 72, 88, 104, 120} for `false`; 9 x 16 = 144.)  The allowlist regexes in
   `trap_allowlist.txt` encode those width sets, so a trap in a width the host *does* launch (8, 16, 32, 64, 128 / 8..48, 64, 80, 96, 112, 128) fails the gate
   (`test_sass_audit.py` asserts it).
5. **Shared memory** - the dp4a tile for the largest type/width (J = 128) needs about 60 KB (x tile `(txs.qs + txs.dm + txs.sc) * 4`, y tile `J * 144` B, ids
   `J * 4`) < `smpbo` 98304; `CUDA_SET_SHARED_MEMORY_LIMIT` raises the per-kernel limit once per device.  On a 64 KB Turing the same selection loop would
   skip the largest widths; on a V100 none is skipped.
6. **No HMMA anywhere in MMQ on sm_70**: `use_mma_data_layout()` requires `TURING_MMA_AVAILABLE` (`mmq.cuh:197`), which is `ggml_cuda_highest_compiled_arch >= 750`; sm_70 has
   `VOLTA_MMA_AVAILABLE`, which only the flash-attention / mmf code uses (`mma.cuh`), not MMQ.  The upstream sm_70 SASS has zero HMMA kernels in total
   (`sass_audit.py` output below) - which is also why R12b matters: this is a CUDA-core int8 path.

Conclusion: the 144 + 4 stub kernels are never launched; no source change is needed; `trap_allowlist.txt` documents it, and `sass_audit.py` keeps it
true across llama.cpp pin bumps (a new trap, or a table row that disappears, shows up as an unexplained trap).

## CPU-side capability decisions that matter on the target box (not GPU, listed because they shape the tools)

| file:line | condition | Cascade Lake (AVX2, AVX-512 F/BW/VL/VNNI, **no VBMI**) gets | verdict |
|---|---|---|---|
| `src/kernels/cpu/expert_layout.cpp:24-54` `cpu_avx512_ok()` | CPUID: AVX-512 F, BW, VL, DQ, VNNI (VBMI optional since WP-E: it selects the VBMI build), OS saves the AVX-512 state; `STRATA_FORCE_AVX2=1` forces false | true (no-VBMI build) | OK (WP-E) |
| `src/program/generate.cpp:1628-1632` | non-native pack: `cpu_require_expert_support()` (`expert.cpp:375`) exits unless AVX-512 F/BW/VL/DQ + VNNI; native pack: AVX-512 -> else a notice and the AVX-2 kernels | WP-E: the Q2_0 pack runs on the no-VBMI build (`CPU expert kernels: AVX-512 VNNI (no VBMI)`); IQ packs: AVX2 rows by default except IQ2_S (`STRATA_IQ512=1` forces AVX-512) | OK |
| `src/program/generate.cpp:5410` | `if (native_pack) { spec_pos = pos; break; }` | a native pack never runs the token loop, so `--dump-logits` writes a header and no rows | OK, but it removes the engine's own logits oracle for every pack this box can run: `golden_compare.py --teacher-source logpos` |

## Warp-synchronous code and independent thread scheduling

Independent thread scheduling is Volta's (Turing and Ampere have it too), so only Volta-specific hazards would matter.  Checked:

* legacy warp intrinsics without `_sync` (`__shfl(`, `__shfl_down(`, `__ballot(`, `__any(`, `__all(`): none in CUDA sources (only the HIP shim in
  `include/strata/hip_compat/intrinsics.hpp`);
* `__activemask`, `__match_*_sync`, `__reduce_*_sync`, cooperative groups: none;
* every `__shfl_*_sync` / `__ballot_sync` uses a full mask and sits where the whole warp arrives (`if (tid < 32)` blocks, warp-uniform loops: `router_top10.cu:94,160`,
  `cvec.cu:77`, `native_bf16.cu:52-111`, `sampler.cu`, `gr.cu`, ...);
* `volatile` appears only on host-mapped flags and buffers (`elementwise.cu:211-303`, `verify_kernels.cu:291,426-515`, `sampler.cu:680-684`) - none is the
  implicit-warp-synchronous shared-memory reduction idiom;
* the spin-wait kernels (`doorbell_wait_kernel`, `wait_flag_ge*`, `resident_plan_kernel`) are `<<<1, 1>>>`: no divergent spin inside a warp.

Not a proof (nothing was run), but nothing Volta-specific was found.

## What the machine code says (upstream, sm_70)

`python3 tools/volta/sass_audit.py --build <upstream build>`: 1199 distinct kernels, **153 with `BPT.TRAP`** = 144 `mul_mat_q` + 4 `quantize_mmq_nvfp4` + 1
`block_scores_tc_kernel` + `prompt_attn_kernel<0/1/3>` + `prompt_attn_i8_kernel`; **0 kernels with HMMA** (cuBLAS, which holds the tensor-core GEMMs, is a shared
library and not scanned).  All 153 are allowlisted with the guard that keeps them from launching; the gate then fails only because the port's own
`prompt_attn_volta` kernel is not in the upstream build.

## Open items for the integrator

1. After the merge, re-read the owner-side fixes against this table (R1, R7, R7b, R13, R14-R16) and re-run `tools/volta/sass_audit.py --build <build>`: the
   allowlist entries for `prompt_attn_kernel<N>` / `prompt_attn_i8_kernel` are only true while `qsa_prompt_attn_batch` stays as in WP-D's tree.
2. R12b and R8 are performance questions the V100 has to answer (`STRATA_PREFILL_MMQ=0` A/B; `STRATA_PREFILL_TIMING=1` for the QSA select share at 32K+).
3. `src/kernels/qsa_select_bench.cpp` has no CMake target (upstream's `bench/micro` and this bench are not built): to time the select on the box it would need one.
