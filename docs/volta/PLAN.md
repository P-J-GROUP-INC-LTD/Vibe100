# Vibe100 — Volta (V100, sm_70) port plan

Base: upstream Strata 0.1.31 (`UPSTREAM.md`). Model: Qwen3.8-Flash-Next (all Strata packs). Second port
(DeepSeek V4.1) is planned separately in `docs/deepseek/PLAN.md`: its foundations (DS-0 to DS-D: contracts, oracle, GGUF
tooling, CPU and GPU expert kernels) need no V100 and are done; its engine (DS-1 on) starts only after Gate 1 below.

The day-one checklist for the target box (BIOS, OS, driver, install, every tool below, in order, with what to send back) is
[RUNBOOK.md](RUNBOOK.md); this page is the plan it executes.

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
| The decode hot path (MMVQ i-quant GEMVs, GDN, gated residual, routers, sampler) uses `dp4a`/FP32/FP16 CUDA-core math. | Works unmodified on sm_70. V100 decode speed is a bandwidth / expert-hit-rate / CPU question → Phase 3 profiling, not a port. |
| Inline tensor-core PTX exists in **three** kernels only: `qsa_prompt_attn.cu` (m16n8k16/m16n8k8 + cp.async), `qsa_select.cu` (3xTF32 m16n8k8), `native_qsa_score.cu` (TF32 + ldmatrix). | Only these need Volta work. |
| **Bug:** `qsa_prompt_attn_batch()` treats every `cc_major == 7` device as Turing and launches `prompt_attn_kernel<1>`; on sm_70 that kernel body is `__trap()` (confirmed: all four prompt-attn kernels contain `BPT.TRAP` in the sm_70 SASS). | The experimental build **crashes on V100 whenever the tensor-core prompt attention is used** (prompt processing with an int8/fp16 KV cache). Must fix. |
| `qsa_select.cu` block-scores TC kernel: host refuses `cc_major < 8` → warp kernel. | Correct on V100 (FP32 CUDA cores). Optimise only if the Phase 3 profile shows it matters. |
| `native_qsa_score.cu`: sm_70 compiles a single-warp FP32 fallback. | Not called by the engine today (only declared/tested). Out of scope. |
| Prefill dense projections go through `cublasGemmEx` with **`CUDA_R_16BF`** (`Gemm::bf16`). Volta has no bf16 tensor cores. | On V100 these run on CUDA cores (~8x below HMMA peak). Route to FP16 HMMA on sm_7x. |
| Expert prefill: `Gemm::native` (dequant → FP16 → cuBLAS fp16, HMMA on Volta) or ggml MMQ. ggml MMQ has trap stubs for tile widths ggml's own host selector never picks on cc 700 (Strata's shim reports `cc = 100*major + 10*minor`). | Already fine; record in the trap allowlist. |
| CUDA 13 dropped sm_70. `setup.py` refuses cc < 7.5, auto-installs CUDA 13.0, and never passes the sm_70 options. Dockerfile is CUDA 13 only. | Installer/build work needed for a one-command V100 setup. |
| Shared memory: Volta opt-in max is 96 KB/block (Turing 64 KB). | Every existing kernel already fits Turing, so it fits Volta. New kernels: ≤ 96 KB. |

## 2. Work packages (Sonnet agents, disjoint file ownership)

Shared contract for all packages:

* Runtime switches (env vars, read once per process):
  * `STRATA_VOLTA_ATTN` — prompt attention on sm_70: unset/`1` (`on`, `true`, `yes`) = the new Volta tensor-core kernel,
    `0` (`off`, `false`, `no`) = the FP32 fallback (the pre-port behaviour minus the crash), used as the harness reference;
    `2`/`force` = the WMMA kernel on any sm_70+ card (a test aid); anything else warns and means the default.
  * `STRATA_PREFILL_F16_GEMM` — `auto` (default: the FP16 tensor-core route **only on Volta**, 7.0 ≤ cc < 7.5 — FP16
    tensor cores, no BF16 ones; Turing keeps upstream's bf16 cuBLAS numerics, and so does everything else),
    `0` = upstream cuBLAS bf16 everywhere, `1` = force the FP16 route on any arch (testing on newer cards, or opting a
    Turing card in).
* Guard, don't replace: Ampere+/Turing code paths stay byte-identical in behaviour (the FP16 GEMM default is Volta-only);
  Volta paths are added beside them.
* No package edits a file owned by another. No `git commit` by agents — the integrator commits after review.
* Verify with `tools/volta/compile_one.py` (per-file sm_70 compile + HMMA/TRAP/spill report). Full builds are
  done by the integrator (4 CPUs are shared).

| WP | Owner | Files | Deliverable |
|---|---|---|---|
| A — platform | Sonnet #1 | `CMakeLists.txt`, `src/core/device.cu`, `setup.py`, `Dockerfile`, `docker-entrypoint.sh`, `setup.sh`, `README.md`, `docs/volta/VOLTA.md` | sm_70 is a first-class target: CMake accepts 70 without the experimental flag and refuses CUDA ≥ 13 for it with a clear message; device check admits cc 7.0 and detects "binary has no code for this GPU"; setup.py builds from source with CUDA 12.8 for V100 (auto-install on Ubuntu/Windows), refuses V100 + RTX 50 in one engine; Docker build-arg for a CUDA 12.8 base; CMake target for `gemm_volta_parity`; user guide. |
| B — audit & harness | Sonnet #2 | `tools/volta/*` (except `compile_one.py`), arch-dispatch fixes in files nobody else owns | `sass_audit.py` + `trap_allowlist.txt` (CI-style gate: no unexplained BPT.TRAP on sm_70), `dispatch_audit.md` (every compute-capability decision, what sm_70 gets), `golden_compare.py` (top-1 agreement, max\|Δlogit\|, KL, perplexity, NaN scan: Volta fast paths vs FP32 reference on the same card, or vs a reference logits file), `run_parity.sh`, `profile_decode.sh` + `summarize_profile.py` (Phase 3). |
| C — prefill GEMM | Sonnet #3 | `src/prefill/gemm.cu`, `include/strata/prefill/gemm.hpp`, `src/prefill/gemm_volta_parity.cpp` | `Gemm::bf16` on a Volta (7.0 ≤ cc < 7.5) converts bf16→fp16 (exact in range) with a **signed** power-of-two scale per operand chunk (k = E_max − 141, clamped at −63, applied back through a device-side alpha): a chunk with huge values is scaled down so activations can never overflow FP16, a chunk of small values is scaled UP, so every element within 2^28 of its chunk's maximum converts exactly (no fp16-subnormal loss for small-valued weights); cuBLAS FP16 HMMA with FP32 accumulate, chunked to a bounded scratch; falls back to upstream bf16 if the scratch cannot be had. Turing and newer keep upstream's call unless `STRATA_PREFILL_F16_GEMM=1`. Parity program vs FP64. |
| D — prompt attention | Sonnet #4 | `src/kernels/cuda/qsa_prompt_attn.cu`, `include/strata/kernels/qsa_prompt_attn.hpp`, `src/kernels/qsa_prompt_attn_parity.cpp` | Fix the dispatcher (cc = major·10+minor; never launch a trap stub). New WMMA (`nvcuda::wmma` m16n16k16, fp16 in / fp32 acc) port of the v1 kernel for KV modes 0, 1, 3 with identical math; HMMA verified in SASS; ≤ 96 KB smem; parity program covers it. |

Integrator (Opus): review every diff against this plan, full sm_70 build + `sass_audit.py`, CPU-side tests that
can run without a GPU, commit, push.

## 3. Phases and gates (run on the V100 box — this container has no GPU)

### The gates, defined once (the tools print these names)

| Gate | Means | Established by |
|---|---|---|
| **Gate 0** | the numbers come from a V100: `nvidia-smi --query-gpu=name,compute_cap --format=csv` says 7.0. A P4000 (6.1) baseline says nothing about Volta. | printed by every tool (`run_parity.sh --require-v100`, `profile_decode.sh --strict`) |
| **Gate P** (parity) | every GPU parity program passes on the V100 — `qsa_prompt_attn_parity`, `gemm_volta_parity`, `kv_hybrid_parity`, the DeepSeek `ds41_router_parity` / `ds41_split_parity` / `ds41_expert_parity`, and every registered ctest — and the SASS audit finds no unexplained `BPT.TRAP` and HMMA in the Volta kernels | `tools/volta/run_parity.sh` prints `GATE P (parity, ...): PASS` |
| **Gate Q** (quality) | top-1 ≥ 99 % over 500 tokens, perplexity within 1–2 % (the tool's default limit is 2 %), no NaN/inf: the Volta fast paths against the FP32 reference paths on the same card, on a prompt long enough to use the batched path and the sparse selection | `tools/volta/golden_compare.py` prints `GATE Q (...): PASS` (or `not established - <what is missing>` when the run was smaller than the gate) |
| **Gate 1** (go on) | IQ3 decode ≥ 40 tok/s on one V100, with Gates P and Q passed. If not, stop and reassess Volta as the target before any DeepSeek engine work (DS-1 on). | measured with `--stats` after the Phase 3 tuning |

### The phases, in the order they can run

Each phase needs the product of the one before it: the profile runs the engine, so the engine has to be built (Phase 1) and
checked (Phases 1–2) first.

| Phase | What | Tool | Gate |
|---|---|---|---|
| 0 | **The box and the card**, nothing built yet: BIOS (Node Interleaving, Sub-NUMA Clustering and System Profile — see VOLTA.md), OS settings, the R580 proprietary driver; confirm the card (`nvidia-smi --query-gpu=name,compute_cap --format=csv` must say 7.0) and the NUMA layout (`numactl -H`: **two** nodes). | `nvidia-smi`, `numactl -H`, `lscpu` | **Gate 0** |
| 1 | **Build for sm_70 with CUDA 12.8** (`./setup.sh`, and `run_parity.sh`'s own build with the parity programs), `strata-device --selftest`, and run every GPU parity program and the SASS audit. | `tools/volta/run_parity.sh` | **Gate P** |
| 2 | **Correctness**: Volta fast paths vs FP32 reference paths on the same card (and vs a reference logits file from another GPU / llama.cpp if available). | `tools/volta/golden_compare.py` | **Gate Q** |
| 3 | **Profile one decode step, then tune**, data-driven: where the step goes (`profile_decode.sh`: GPU kernels, CPU experts, PCIe, idle share; per-token figures cut to the decode window), then expert-cache size for 32 GB, `--calibrate`, `--pcie-frac` for PCIe Gen3, the A/B switches (MMQ vs cuBLAS, NUMA mirror, `STRATA_IQ512`), and only then kernel work where the profile points. | `tools/volta/profile_decode.sh`, `--calibrate`, `--stats` | **Gate 1** |

(Until 2026-10-01 this plan listed the profile as "Phase 0", before the build it needs; VOLTA.md listed it last. The profile is
now Phase 3 everywhere, and the tools' headers and output files — `profile.txt`, `profile.json` — say so.)

Candidate Phase-3 kernel work, only if the profile shows it: Volta HMMA version of the QSA block-score select
(currently FP32 warp kernel on sm_70), tile/occupancy retune of decode GEMVs for 80 SMs / 6 MB L2.

Phase-3 placement work (not a kernel change, independent of Volta): the **usage ledger**
(`docs/USAGE_LEDGER.md`, a design only — nothing of it is implemented) — long-term, persisted per-expert routing counts
that seed the 32 GB expert cache from the user's own weeks of use, beside the existing short-term adaptive swaps.

### Audit round (2026-10-01)

An independent audit of the first integration found, and the round fixed:

* the prefill GEMM's scaling was unsigned (small-valued chunks lost bits to fp16 subnormals) and its `auto` default also
  switched Turing: now signed, and Volta-only (WP-C above);
* a build whose code is sm_70-only run on a card of cc ≥ 7.5 (the sm_70 cubin runs natively on a 7.5, the compute_70 PTX is
  JIT-compiled on 8.x, and either way upstream's v1 / cp.async prompt-attention kernels are `__trap()` stubs): the attention
  dispatcher now asks the runtime which code each kernel got (`ptxVersion`) and never launches a trap stub, and `device.cu`
  refuses at start-up to run a build whose code is older than the card, naming the architectures it was built for
  (`tools/volta/dispatch_audit.md`, R7b);
* the harness: the profile's PCIe and kernel-time figures were summed over the whole trace (start-up included) and divided by
  the token count; relative output paths were handed to an engine running in another directory; `golden_compare.py --teacher-source
  logpos` dropped the flags `strata --serve` refuses to start without; the gate names and the phase order disagreed (this page).

### The target box's CPU side (from the user, 2026-10-01)

Dell Precision 7920, 2x Xeon **Gold 6226** (Cascade Lake, 12 cores each), 24 DIMM slots, **384 GB DDR4-2666**
(6 channels per socket: 12 channels, 2 DIMMs each, ≈ 256 GB/s theoretical; **measured 90-120 GB/s** — likely
reads crossing the UPI link; the Phase 3 profile runs `mlc --bandwidth_matrix` (if Intel MLC is installed) and
`nvidia-smi topo -m` to see local vs remote bandwidth and which socket the V100 hangs off). Two facts
from the source that matter on this box, both CPU-side and independent of the GPU port:

1. **Strata's own AVX-512 expert kernels were off on Cascade Lake** (`cpu_avx512_ok()` required AVX512-VBMI for one
   instruction, `vpmultishiftqb`). **Fixed by WP-E:** `expert.cpp` is compiled a second time without VBMI (a
   shuffle / shift / mask unpack proven bit-identical over all 2^32 inputs) and picked at run time; setup now offers
   the canonical Q2_0 pack on Cascade Lake. Measured on a Cascade-Lake-class VM: canonical Q2_0 kernel 5.8-6.2 GB/s
   per thread (AVX2 rows: 3.3). The i-quant packs keep AVX2 by default on this tier except IQ2_S (AVX-512 measured
   slower there); `STRATA_IQ512=1` A/Bs it on the real Xeon.
2. **The CPU expert pool is not NUMA-aware.** Workers are pinned to cores (`kernels/cpu/pool.cpp`) but the expert
   arena is placed by first touch, so half the workers read it across UPI. Not a fix: BIOS Node Interleaving — it makes
   Linux see ONE node, which silently disables the mirror below and the DeepSeek port's per-socket split; keep it
   **Disabled** (VOLTA.md, "BIOS"). A quick baseline when the mirror cannot be used (low-RAM modes): the OS-level
   `numactl --interleave=all` (never together with the mirror: it is the A/B baseline). Real fix for Qwen: **mirror the expert arena on both
   nodes** — the arena is 23-50 GB, so two copies fit easily in 384 GB; each socket's workers read their own node's
   copy and the GPU's DMA reads the copy on the GPU's node. The user measured ~2x decode from full mirroring of Qwen
   Next on this box. (Mirroring is simpler than re-partitioning Strata's compile-time-geometry kernels by rows; the
   DeepSeek port, whose 269 GiB of experts cannot be mirrored, splits rows instead — `docs/deepseek/PLAN.md` §2.)
   **Done: WP-F** — `--numa auto` (default) mirrors the arena (primary on the GPU's node, CUDA-registered; one replica
   per other node; workers read their node's copy; placement verified at start); `STRATA_NUMA_MIRROR=0` for the A/B.

## 4. Honest limits

* Nothing here has run on a V100. The integrator can compile for sm_70, inspect SASS, check resource usage and
  run CPU-only tests; parity, golden-compare and speed numbers must come from the real card.
* The 40 tok/s target is the brief's, not a prediction. Decode on V100 is dominated by expert cache hit rate and
  CPU expert throughput, which the port does not change; the 32 GB of VRAM is the lever.
