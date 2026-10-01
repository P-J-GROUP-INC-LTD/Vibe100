# Vibe100 — Volta (V100, sm_70) port plan

Base: upstream Strata 0.1.31 (`UPSTREAM.md`). Model: Qwen3.8-Flash-Next (all Strata packs). Second port
(DeepSeek V4.1) is planned separately in `docs/deepseek/PLAN.md` and starts only after Gate 1 below.

This plan supersedes the Volta half of the original "STRATA-VOLTA-DEEPSEEK-PORT" brief. It keeps that brief's
principles (one variable at a time, guard rather than replace, correctness harness before tuning, gates) and
corrects its premises against the actual upstream source.

## 1. What the source actually says (inventory, done)

Measured on upstream 0.1.31 by building it for sm_70 with CUDA 12.8 (`-DSTRATA_EXPERIMENTAL_SM60=ON`) and
disassembling the result (`cuobjdump -sass -arch sm_70`):

| Finding | Consequence |
|---|---|
| Upstream is **not Ampere-only**: RTX 20 (sm_75) has been supported since 0.1.27, and an experimental Pascal/Volta build exists (issue #236 — a V100 32 GB user ran the Coder and Swift IQ3_XXS with it). | The port is small and targeted, not a rewrite. |
| Kernels treat bf16 as raw bits → fp32 (`bf16_bits.hpp`), so no `__nv_bfloat16` hardware dependency in kernels. | Nothing to do for bf16 inside kernels. |
| The decode hot path (MMVQ i-quant GEMVs, GDN, gated residual, routers, sampler) uses `dp4a`/FP32/FP16 CUDA-core math. | Works unmodified on sm_70. V100 decode speed is a bandwidth / expert-hit-rate / CPU question → Phase 0 profiling, not a port. |
| Inline tensor-core PTX exists in **three** kernels only: `qsa_prompt_attn.cu` (m16n8k16/m16n8k8 + cp.async), `qsa_select.cu` (3xTF32 m16n8k8), `native_qsa_score.cu` (TF32 + ldmatrix). | Only these need Volta work. |
| **Bug:** `qsa_prompt_attn_batch()` treats every `cc_major == 7` device as Turing and launches `prompt_attn_kernel<1>`; on sm_70 that kernel body is `__trap()` (confirmed: all four prompt-attn kernels contain `BPT.TRAP` in the sm_70 SASS). | The experimental build **crashes on V100 whenever the tensor-core prompt attention is used** (prompt processing with an int8/fp16 KV cache). Must fix. |
| `qsa_select.cu` block-scores TC kernel: host refuses `cc_major < 8` → warp kernel. | Correct on V100 (FP32 CUDA cores). Optimise only if Phase 0 shows it matters. |
| `native_qsa_score.cu`: sm_70 compiles a single-warp FP32 fallback. | Not called by the engine today (only declared/tested). Out of scope. |
| Prefill dense projections go through `cublasGemmEx` with **`CUDA_R_16BF`** (`Gemm::bf16`). Volta has no bf16 tensor cores. | On V100 these run on CUDA cores (~8x below HMMA peak). Route to FP16 HMMA on sm_7x. |
| Expert prefill: `Gemm::native` (dequant → FP16 → cuBLAS fp16, HMMA on Volta) or ggml MMQ. ggml MMQ has trap stubs for tile widths ggml's own host selector never picks on cc 700 (Strata's shim reports `cc = 100*major + 10*minor`). | Already fine; record in the trap allowlist. |
| CUDA 13 dropped sm_70. `setup.py` refuses cc < 7.5, auto-installs CUDA 13.0, and never passes the sm_70 options. Dockerfile is CUDA 13 only. | Installer/build work needed for a one-command V100 setup. |
| Shared memory: Volta opt-in max is 96 KB/block (Turing 64 KB). | Every existing kernel already fits Turing, so it fits Volta. New kernels: ≤ 96 KB. |

## 2. Work packages (Sonnet agents, disjoint file ownership)

Shared contract for all packages:

* Runtime switches (env vars, read once per process):
  * `STRATA_VOLTA_ATTN` — prompt attention on sm_70: unset/`1` = the new Volta tensor-core kernel, `0` = the FP32
    fallback (the pre-port behaviour minus the crash). Used as the harness reference.
  * `STRATA_PREFILL_F16_GEMM` — `auto` (default: FP16 tensor-core route when the current device has cc < 8.0),
    `0` = upstream cuBLAS bf16 everywhere, `1` = force the FP16 route on any arch (testing on newer cards).
* Guard, don't replace: Ampere+/Turing code paths stay byte-identical in behaviour; Volta paths are added beside them.
* No package edits a file owned by another. No `git commit` by agents — the integrator commits after review.
* Verify with `tools/volta/compile_one.py` (per-file sm_70 compile + HMMA/TRAP/spill report). Full builds are
  done by the integrator (4 CPUs are shared).

| WP | Owner | Files | Deliverable |
|---|---|---|---|
| A — platform | Sonnet #1 | `CMakeLists.txt`, `src/core/device.cu`, `setup.py`, `Dockerfile`, `docker-entrypoint.sh`, `setup.sh`, `README.md`, `docs/volta/VOLTA.md` | sm_70 is a first-class target: CMake accepts 70 without the experimental flag and refuses CUDA ≥ 13 for it with a clear message; device check admits cc 7.0 and detects "binary has no code for this GPU"; setup.py builds from source with CUDA 12.8 for V100 (auto-install on Ubuntu/Windows), refuses V100 + RTX 50 in one engine; Docker build-arg for a CUDA 12.8 base; CMake target for `gemm_volta_parity`; user guide. |
| B — audit & harness | Sonnet #2 | `tools/volta/*` (except `compile_one.py`), arch-dispatch fixes in files nobody else owns | `sass_audit.py` + `trap_allowlist.txt` (CI-style gate: no unexplained BPT.TRAP on sm_70), `dispatch_audit.md` (every compute-capability decision, what sm_70 gets), `golden_compare.py` (top-1 agreement, max\|Δlogit\|, KL, perplexity, NaN scan: Volta fast paths vs FP32 reference on the same card, or vs a reference logits file), `run_parity.sh`, `profile_decode.sh` + `summarize_profile.py` (Phase 0). |
| C — prefill GEMM | Sonnet #3 | `src/prefill/gemm.cu`, `include/strata/prefill/gemm.hpp`, `src/prefill/gemm_volta_parity.cpp` | `Gemm::bf16` on cc < 8.0 converts bf16→fp16 (exact in range; power-of-two per-call scaling with device-side alpha so activations can never overflow FP16) and runs cuBLAS FP16 HMMA with FP32 accumulate, chunked to a bounded scratch; falls back to upstream bf16 if the scratch cannot be had. Parity program vs FP64. |
| D — prompt attention | Sonnet #4 | `src/kernels/cuda/qsa_prompt_attn.cu`, `include/strata/kernels/qsa_prompt_attn.hpp`, `src/kernels/qsa_prompt_attn_parity.cpp` | Fix the dispatcher (cc = major·10+minor; never launch a trap stub). New WMMA (`nvcuda::wmma` m16n16k16, fp16 in / fp32 acc) port of the v1 kernel for KV modes 0, 1, 3 with identical math; HMMA verified in SASS; ≤ 96 KB smem; parity program covers it. |

Integrator (Opus): review every diff against this plan, full sm_70 build + `sass_audit.py`, CPU-side tests that
can run without a GPU, commit, push.

## 3. Phases and gates (run on the V100 box — this container has no GPU)

| Phase | What | Tool | Gate |
|---|---|---|---|
| 0 | Confirm the card (`nvidia-smi --query-gpu=name,compute_cap --format=csv` must say 7.0), fix NUMA (`numactl --interleave=all` or BIOS node interleaving), profile one decode step. | `tools/volta/profile_decode.sh` | **Gate 0**: numbers are from a V100. A P4000 (6.1) baseline says nothing about V100. |
| 1 | Build for sm_70 with CUDA 12.8, run every GPU parity program. | `tools/volta/run_parity.sh` | All pass. |
| 2 | Correctness: Volta fast paths vs FP32 reference paths (and vs a reference logits file from another GPU / llama.cpp if available). | `tools/volta/golden_compare.py` | top-1 ≥ 99 % over 500 tokens, PPL within 1–2 %, no NaN/inf. |
| 3 | Tuning, data-driven from Phase 0: expert-cache size for 32 GB, `--calibrate`, `--pcie-frac` for PCIe Gen3, and only then kernel work where the profile points. | `--calibrate`, `--stats` | **Gate 1**: IQ3 decode ≥ 40 tok/s on one V100 and Phase 2 passing. If not, stop and reassess Volta as the target before any DeepSeek work. |

Candidate Phase-3 kernel work, only if the profile shows it: Volta HMMA version of the QSA block-score select
(currently FP32 warp kernel on sm_70), tile/occupancy retune of decode GEMVs for 80 SMs / 6 MB L2.

## 4. Honest limits

* Nothing here has run on a V100. The integrator can compile for sm_70, inspect SASS, check resource usage and
  run CPU-only tests; parity, golden-compare and speed numbers must come from the real card.
* The 40 tok/s target is the brief's, not a prediction. Decode on V100 is dominated by expert cache hit rate and
  CPU expert throughput, which the port does not change; the 32 GB of VRAM is the lever.
