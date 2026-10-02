# Handoff: working on Vibe100 on the target box

For the AI agent (or person) who takes over **on the owner's machine**: a Dell Precision 7920, 2× Xeon Gold 6226 (Cascade Lake,
12 cores each, AVX-512 VNNI, **no VBMI**), 384 GiB DDR4-2666 in two NUMA nodes, **one Tesla V100 32 GB** on PCIe Gen3, Ubuntu.
Written 2026-10-01 by the cloud session that built the port. Everything below was built and checked **without a GPU**: this
box is where it runs for the first time. Read sections 1-3 before touching anything; use section 5 as the playbook when a
step fails. The goal is that you rarely need to ask the cloud session anything.

---

## 1. What this repository is

**Vibe100 = upstream Strata 0.1.31 (MIT, github.com/Niko1221/Strata, upstream commit 9259cad) ported to the V100, plus the start
of a DeepSeek-V4.1-Flash port.** The first commit `3906943` is the byte-identical upstream snapshot; `git diff 3906943` is the whole
port. Strata runs Qwen3.8-Flash-Next (~125B MoE: 48 layers, 512 experts/layer, top-10) with the dense layers and a cache of hot experts
on the GPU and every expert in RAM; the CPU computes the experts the GPU does not hold.

Branch: `claude/volta-v100-conversion-8f17vp` (all work so far). State at handoff, newest first:

| Commit | What |
|---|---|
| `8ded8f0` | logit-identity matrix: bit-exact vs upstream, KL/top-1/PPL vs llama.cpp (`tools/volta/logit_identity.sh`) |
| `18db594` | prompt-attention dispatcher chooses kernels by the code the binary contains (audit A1) |
| `21f0236` | DeepSeek CPU/GPU expert kernel fixes (audits A5/A6) |
| `7e9aba0` | NUMA arena robustness (per-node huge pages, pinned replica, SNC), Q2_0 CPU prefetch +31%/thread |
| `43bb4bf` | installer/build/device-check fixes (audit A3) |
| `23423d7` | day-one runbook, gates, harness fixes |
| `51ba20b` | landing page README, PROVENANCE |
| `bfddd12` | prefill GEMM signed scaling, Volta-only default |
| earlier | WP-A..F (platform, harness, prefill GEMM, prompt attention, AVX-512 without VBMI, NUMA mirror), DS-0..DS-D |

The full audit (55 items, 52 fixed) and the ranked optimisation backlog: the "Vibe100 Audit Round" report (the owner has the link),
and in-tree: `tools/volta/dispatch_audit.md` (every compute-capability decision), `docs/volta/PLAN.md` §3 (gates).

### What the port changed (where to look when editing)

| Area | Files | What it does | Off switch |
|---|---|---|---|
| Build / platform (WP-A) | `CMakeLists.txt`, `setup.py`, `Dockerfile`, `src/core/device.cu`, `include/strata/core/device.hpp` | sm_70 is a supported target; CUDA ≥ 13 refused for it; a V100 compiles locally with CUDA 12.8; the engine refuses a binary with no (or too old) code for a card | - |
| Prompt attention (WP-D) | `src/kernels/cuda/qsa_prompt_attn.cu` (+ `.hpp`, `qsa_prompt_attn_parity.cpp`) | `select_impl`: V100 → `prompt_attn_volta_kernel<KV>` (WMMA m16n16k16, 512 HMMA/loop, runtime accumulator-row probe); upstream's v1 / cp.async kernels are trap stubs on sm_70 | `STRATA_VOLTA_ATTN=0` (FP32 fallback, upstream's) |
| Prefill dense GEMMs (WP-C) | `src/prefill/gemm.cu` (`bf16_via_f16`), `gemm_volta_parity.cpp` | bf16 → fp16 exact conversion with a signed per-chunk power-of-two scale, FP16 HMMA, FP32 accumulate, staged in the GEMM scratch; Volta only by default | `STRATA_PREFILL_F16_GEMM=0` |
| CPU expert kernels (WP-E) | `src/kernels/cpu/expert.cpp` (compiled twice: + `expert_novbmi.cpp`), `expert_dispatch.cpp`, `native_expert.cpp` | AVX-512 kernels without VBMI for Cascade Lake; tiers Avx512Vbmi / Avx512Vnni / Avx2; Q2_0 prefetch | `STRATA_FORCE_AVX2=1`, `STRATA_Q2_PREFETCH=0` |
| NUMA mirror (WP-F) | `src/platform/numa.cpp`, `include/strata/platform/numa.hpp`, `src/kernels/cpu/pool.cpp`, `src/core/pinned.cu`, `src/core/expert_source.cpp`, `src/core/session.cpp` | one copy of the expert arena per socket; each worker reads its own node's copy (`ExpertPool::xl()`); primary on the GPU's node | `--numa off` or `STRATA_NUMA_MIRROR=0` |
| Harness | `tools/volta/*` | SASS audit, parity runner, golden compare, profiler, logit identity | - |
| DeepSeek (not runnable yet) | `src/ds41/**`, `include/strata/ds41/**`, `ref/ds41/**`, `tools/ds41/**`, `cmake/ds41_*.cmake`, `docs/deepseek/**` | kernels + oracle + GGUF tooling; the engine (DS-1) is being written by the cloud session | - |

Nothing else in upstream's engine was changed: decode kernels, ggml MMQ, server, packs, speculative decoding are upstream's.

---

## 2. Ground rules for editing on the box

1. **Branch.** Do box work on its own branch, created from commit **`da0ae45`** - the Qwen-ready state (a clean full sm_70 build of
   all 268 targets, SASS audit PASS, every CPU-runnable test PASS):
   `git fetch origin && git checkout -b box/qwen-v100 da0ae45`. Push it; the cloud session (or the owner) merges it back.
   **Do not merge newer commits of `claude/volta-v100-conversion-8f17vp` while testing Qwen**: the cloud session pushes DeepSeek
   work-in-progress checkpoints there (often, so nothing is lost), and they may not compile. If a later Qwen fix is announced, take that
   one commit (`git cherry-pick <hash>`).
2. **Do not edit** `src/ds41/**`, `include/strata/ds41/**`, `ref/ds41/**`, `tools/ds41/**`, `cmake/ds41_*.cmake`, `docs/deepseek/**`:
   the cloud session owns them while it writes DS-1. If a DeepSeek parity program fails on the box, record it (section 6), do not fix it.
3. **Write down what you see.** Append every result and every change to `docs/volta/BOX_LOG.md` (create it; newest at the bottom; date,
   command, the lines that matter, conclusion). It is how the next session - cloud or local - picks up without asking.
4. **Never weaken a check to get green.** Do not skip, disable or widen a test's budget without a written reason in the code comment and
   in BOX_LOG (a measured error and why it is acceptable). Do not "fix" a parity failure by changing the reference.
5. **A switch before a patch.** Every port path has an environment switch back to upstream's behaviour (table in section 4). When something
   fails, first confirm with the switch that the port path is the cause, keep the box usable with the switch, then patch.
6. **Numerics are a contract.** A change to a kernel's arithmetic needs: the parity program passing, `golden_compare` (Gate Q) passing, and
   the SASS audit. A change that should be bit-identical (prefetch, layout, dispatch) needs a before/after logits comparison:
   `golden_compare.py --exact` (or `logit_identity.sh --only 1c,1b`).
7. **Commit messages**: what changed and why, and the measurements. Small, one subject per commit.

### How to build and check a change

```
# the engine, as setup builds it (the installed engine is engine/strata, copied from build/)
.venv/bin/cmake --build build --target strata -j24
./setup.sh                                  # rebuilds and reinstalls when the engine's source changed (fingerprint)

# one file for sm_70, with registers / spills / HMMA / traps per kernel (seconds; no full build). Needs a build directory with
# compile_commands.json: run_parity.sh's build-sm70/ has one (setup's build/ does not)
.venv/bin/python tools/volta/compile_one.py --build build-sm70 src/prefill/gemm.cu

# the full sm_70 build with every parity program, ctest and the SASS audit (Gate P)
tools/volta/run_parity.sh --ggml-dir third_party/llama.cpp --require-v100           # --skip-build to re-run checks only

# tool unit tests (no GPU), after editing tools/volta
.venv/bin/python -m unittest discover -s tools/volta
for f in tools/test_setup_*.py; do .venv/bin/python $f; done                           # after editing setup.py
```

---

## 3. The plan on the box, and what "done" means

Follow **`docs/volta/RUNBOOK.md`** step by step (0 BIOS → 1 OS → 2 driver → 3 setup → 4 strata-device → 5 run_parity → 6 golden_compare →
6b logit_identity → 7 profile → 8 A/B tuning → 9 DeepSeek checks). Gates (`docs/volta/PLAN.md` §3):

| Gate | Pass means | Printed by |
|---|---|---|
| 0 | `nvidia-smi` says compute capability 7.0, Linux sees 2 NUMA nodes | runbook steps 1-2 |
| P | every GPU parity program passes, ctest passes, SASS audit finds no unexplained trap | `run_parity.sh`: `GATE P (...): PASS` |
| Q | Volta fast paths vs FP32 reference paths on the same card: top-1 ≥ 99 %, PPL within 2 %, no NaN/inf | `golden_compare.py`: `GATE Q (...): PASS` |
| 1 | IQ3_XXS decode ≥ 40 tok/s on the V100 with P and Q passed | `--stats` / `profile_decode.sh` |

Plus the owner's requirement: **no quality loss against llama.cpp** on the same GGUF - `logit_identity.sh` tier 2 (`docs/volta/LOGIT_IDENTITY.md`).

**Expectations (estimates by the porting session, nothing measured on a V100):** Qwen IQ2 decode ~80-130 tok/s, IQ3 ~60-100 tok/s,
prefill ~1,000-2,000 tok/s on long prompts. Gate 1's 40 tok/s is the pass mark, not the estimate. If decode is far below 40, it is almost
certainly a placement/NUMA/CPU problem, not the GPU kernels (section 5.8).

---

## 4. Switches (environment unless noted)

| Switch | Default | Effect / when to use |
|---|---|---|
| `STRATA_VOLTA_ATTN` | on | `0`/`off`: prompt attention on upstream's FP32 kernel (slower prefill, the reference). `2`/`force`: WMMA kernel on any sm_70+ card (tests) |
| `STRATA_PREFILL_F16_GEMM` | `auto` (= on for Volta) | `0`: upstream's bf16 cuBLAS call for the prefill projections. `1`: route on any card |
| `STRATA_PREFILL_MMQ` | upstream (on) | `0`: prompt expert GEMMs as dequant-FP16 + cuBLAS HMMA instead of ggml MMQ (dp4a). **The top prefill A/B on Volta** (audit backlog #3); needs ~0.5 GB more at 8K-token chunks |
| `--numa auto\|mirror\|off` (engine; `./setup.sh --numa X` saves it) | `auto` | mirror the expert arena per socket |
| `STRATA_NUMA_MIRROR` | unset | `0`/`1` overrides `--numa` for one run (A/B) |
| `STRATA_NUMA_GPU_NODE` | from sysfs | set `N` when the log says the GPU's node was ASSUMED (`nvidia-smi topo -m` shows the card's CPU affinity) |
| `STRATA_NUMA_HEADROOM_GIB` | 6 | free memory each node keeps beyond its copy; lower it only if the mirror is refused for a few GiB |
| `STRATA_NUMA_PIN_REPLICA` | on | `0`: mlock the replica instead of `cudaHostRegister` (if registration costs VRAM or start-up time) |
| `STRATA_ARENA_LOCK` | on | `0`: no mlock at all |
| `STRATA_NO_THP` | unset | `1`: no transparent huge pages for the arena copies (A/B) |
| `STRATA_Q2_PREFETCH` | 3072 | bytes of prefetch in the Q2_0 CPU rows; `0` = off (bit-identical either way) |
| `STRATA_IQ512` | unset | `1`: AVX-512 i-quant rows for every format (A/B on Cascade Lake; default keeps AVX2 for IQ2_XS/IQ3_XXS/IQ3_S) |
| `STRATA_FORCE_AVX2` / `STRATA_FORCE_AVX512_NOVBMI` | unset | `1`: force that CPU tier (diagnosis, bit-exact comparisons) |
| `STRATA_IQ_MT_MIN` | upstream | `1`: fixed CPU thread split (determinism in comparisons) |
| `STRATA_PROMPT_ATTN_OLD` | upstream | `1`: upstream's own FP32 prompt attention (what the upstream build needs on a V100) |
| `STRATA_LOGPOS` / `STRATA_LOGITS_DUMP` | unset | per-position log-probs / full logits from the verify windows (harness use) |

---

## 5. Playbook: when a step fails

Each entry: **symptom → likely cause → what to do.** Always save the failing command's full output into BOX_LOG first.

### 5.1 BIOS / OS (runbook 0-1)

- **`lscpu` says `NUMA node(s): 1`** → BIOS Node Interleaving is enabled (or SNC on a one-socket view). Disable Node Interleaving
  (Memory Settings). The mirror and DeepSeek's per-socket split silently do nothing on one node.
- **4 NUMA nodes** → Sub-NUMA Clustering on. Disable it (Processor Settings). (The engine copes - one copy per socket - but the runbook
  assumes 2.)
- **`free -g` total well under ~377** → Memory Operating Mode is mirror/spare; set Optimizer Mode.
- **`ulimit -l` not unlimited after re-login** → `/etc/security/limits.d/90-strata.conf` not applied (needs a new login session; for
  systemd services set `LimitMEMLOCK=infinity`). Not fatal: the replica is pinned through CUDA instead; mlock is only the fallback.
- **THP `never`** → `echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled`. Not fatal (4 KiB pages, a few % slower).

### 5.2 Driver (runbook 2)

- **`nvidia-smi` lists no V100 / "No devices were found"**:
  - `cat /proc/driver/nvidia/version` says "Open Kernel Module" → the `-open` driver is installed: NVIDIA's open modules support
    Turing and newer only. `sudo apt purge 'nvidia-*open*' && sudo apt install nvidia-driver-580`, reboot. (`setup.py` also says this.)
  - Driver branch > 580 → R580 is the last branch with Volta. Install `nvidia-driver-580`, then `apt-mark hold` as in the runbook.
  - `dmesg | grep -i nvrm` shows BAR/resource errors → BIOS "Memory Mapped I/O Above 4 GB" must be Enabled.
  - Secure Boot: the DKMS module must be signed (MOK enrolment at the next boot) or Secure Boot off.
- **`nvidia-smi` works but CUDA says "driver too old"** → driver < 570; CUDA 12.8 needs ≥ 570.
- **PCIe link not x16 Gen3** (`nvidia-smi -q | grep -A3 "Link"`): the card is in an x8 slot, or a riser downgrades it - move it; it halves
  expert streaming (prefill) bandwidth.

### 5.3 `./setup.sh` (runbook 3)

- **"CUDA 13 dropped Volta" (CMake)** → it found a CUDA 13 nvcc. Setup normally passes the 12.8 nvcc itself; by hand use
  `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`.
- **CMake "3.24 or higher is required"** → use `.venv/bin/cmake` (pip), not the apt one (22.04 ships 3.22).
- **cuda-toolkit-12-8 install fails** → NVIDIA's apt repo key/network. Manual: download the 12.8 runfile from NVIDIA, install the toolkit only
  (no driver) to `/usr/local/cuda-12.8`, re-run setup (it finds it).
- **nvcc errors about the host compiler** → CUDA 12.8 supports the gcc of Ubuntu 22.04 (11) and 24.04 (13). An older CUDA 12.0 refuses
  gcc 13; setup should have chosen 12.8 - check `engine/BUILD.json` `"cuda"`.
- **Compile error in a port file** (`qsa_prompt_attn.cu`, `gemm.cu`, `device.cu`, `numa.cpp`, `pool.cpp`, `expert.cpp`) → these were all
  compiled for sm_70 with CUDA 12.8 in the cloud; a failure means a different toolkit or compiler. Record `nvcc --version`, `gcc --version`,
  the first error. Try `CUDA 12.8` exactly.
- **Setup picked the AVX2 "native" Q2_0 pack instead of the canonical one** → the engine is not recognised as compiled from this source
  (BUILD.json `source` not `local` or the source hash changed). Re-run `./setup.sh` so it rebuilds; on a V100 the engine is always local.
- **Start-up log has no `NUMA:` mirror line / says it kept ONE copy** → read the reason in that line:
  - "headroom" / not enough free memory on node N → close other processes, release hugetlb pools you do not use, or
    `STRATA_NUMA_HEADROOM_GIB=3`.
  - "GPU node ASSUMED" → `STRATA_NUMA_GPU_NODE=<node from nvidia-smi topo -m>`.
  - low-RAM / streaming mode → mirroring needs the full-RAM arena (the model's experts resident). Expected in low-RAM modes.
  - `mbind`/`move_pages` EPERM → running in Docker without `--cap-add SYS_NICE` + a seccomp profile allowing `move_pages`.
- **Start-up takes much longer than upstream** → registering a 23-50 GB replica with CUDA plus populating it. Compare with
  `STRATA_NUMA_PIN_REPLICA=0`; if registration is the cost and decode is the same, keep `=0` (write it into the config's `env`).

### 5.4 `strata-device --selftest` (runbook 4)

- **"this Strata engine has no code it can run"** → built without 70 (e.g. the ready-made engine). `./setup.sh` builds for the V100.
- **"compiled for sm_70 only ... card cc 7.5+"** → wrong GPU selected (another card visible). Use `CUDA_VISIBLE_DEVICES` / `--gpu`.
- **VRAM budget far below 32 GB** → other processes on the card (`nvidia-smi`), or ECC/persistence; check `nvidia-smi -q -d MEMORY`.

### 5.5 `run_parity.sh` (runbook 5, Gate P)

Read the summary table; each FAIL has a log in `build-sm70/parity-logs/<time>/`. Program by program:

- **`qsa_prompt_attn_parity`** (Volta WMMA prompt attention). Expected: variant `[volta-wmma]`, `PASSED: n  SKIPPED (fallback): 0
  FAILURES: 0`.
  - Variant shows `fallback-fp32` → the WMMA kernel was not selected: `STRATA_VOLTA_ATTN` set? cc not 7.0? Check `select_impl`.
  - **FAIL in every mode / every case** → the accumulator-layout probe or the fragment strides. The kernel probes which accumulator
    slot holds which row at run time (`prompt_attn_volta_kernel`, search "probe"); the auditor confirmed the mapping from SASS but it has
    never run. Keep the box usable with `STRATA_VOLTA_ATTN=0`, save the log, record the max error per case. Then debug from the log: it
    prints every case's max error and the failing query; start with the smallest failing case and compare its output against the FP64
    reference the program computes.
  - **FAIL only in the K8V4 "real q4_0 scales" case, error a little over budget** → K8V4 V values are rounded to fp16 (≤ 2^-11 relative,
    inherited from upstream's v1; the budget derivation is in the source above `Smem`). If the error is within ~2× the budget and Gate Q
    passes, widening is acceptable - with the measurement in the comment. Exact-scale and fp16/int8 cases must stay tight.
  - **FAIL only in the 65,541-query split case** → the grid-split launch offset (`launch_volta`); or out of device memory (it needs ~3.3 GB:
    then it should SKIP - check).
  - **"unspecified launch failure"** → a kernel trapped. Run with `CUDA_LAUNCH_BLOCKING=1` and
    `/usr/local/cuda-12.8/bin/compute-sanitizer --tool memcheck` on the program; the faulting kernel name tells you which. A trap in
    `prompt_attn_kernel` / `prompt_attn_i8_kernel` means the dispatcher let a trap stub through (should be impossible on a V100: report).
- **`gemm_volta_parity`** (FP16 prefill GEMM). Expected: `gemm_volta_parity OK`, route TFLOPS > 35 (only HMMA reaches that).
  - **NaN in a beta = 0 case** → cuBLAS read the (NaN-poisoned) output although beta = 0 is passed as a DEVICE scalar
    (CUBLAS_POINTER_MODE_DEVICE). Fix in `src/prefill/gemm.cu` `bf16_via_f16`: when the caller's `beta == 0.0f` (known on the host), zero the
    Y sub-block with `cudaMemset2DAsync` before the GEMM (pointer mode covers alpha and beta together, so beta cannot stay a host value
    while alpha is a device one). Meanwhile `STRATA_PREFILL_F16_GEMM=0`.
  - **TFLOPS ~10-15** → cuBLAS did not pick tensor-core kernels: check alignment (lda multiple of 8 halves; the code pads K to 8), math
    mode (`CUBLAS_DEFAULT_MATH`), and the cuBLAS version (`ldd build/... | grep cublas` → 12.x). Try `CUBLAS_TENSOR_OP_MATH` on the handle as
    a test.
  - **Error over budget by a small factor** → Volta's HMMA accumulates with truncation; the budget assumed ~1e-6 at K = 10240. If Gate Q
    passes, widen the budget with the measured figure in the comment.
  - **cuBLAS "not supported" on the bf16 call** (small shapes still use upstream's bf16 cuBLAS) → make the route take every shape on
    Volta: in `bf16_via_f16`, treat `auto` on cc 7.0/7.2 like `kRouteAll` (skip the `T < 64 || N < 64 || T*N*K < 2^29` early return).
- **`kv_hybrid_parity`** → same kernel family as qsa; read its [n/5] step; K8V4 tolerance is derived from the same 2^-11 bound.
- **`ds41_router_parity` / `ds41_split_parity` / `ds41_expert_parity`** (DeepSeek kernels, synthetic data) → do not block Qwen. Record
  output and stop there. One known risk: the split's host doorbell (mapped memory, `st.release.sys`) showing a torn read - the fallback
  is a `__threadfence_system()` in every miss-writing thread; leave it to the cloud session.
- **ctest failures** → CPU tests run on the real Xeon here for the first time: `expert_novbmi_no_vbmi` (greps objects for VBMI - must
  pass), `expert_variant_test`, `numa_test` (fake sysfs - must pass), `pool_test_numa_mirror` (real 2-node box now). Record which.
- **SASS audit "UNEXPLAINED trap"** → a kernel with `BPT.TRAP` not in `tools/volta/trap_allowlist.txt`. Only allowlist it if you can name
  the host guard that never launches it on cc 7.0 (see the existing entries' wording); otherwise it is a crash waiting to happen.

### 5.6 `golden_compare.py` (runbook 6, Gate Q)

- **`not established - ...`** → the run was smaller than the gate (fewer than 500 scored rows, or the batched part shorter than 2,051
  tokens). Use `--prompt-name long --tail 512`.
- **top-1 < 99 % or PPL off > 2 %** → bisect with the candidate's switches, one at a time:
  `--cand-env "STRATA_VOLTA_ATTN=0"` (attention alone off) and `--cand-env "STRATA_PREFILL_F16_GEMM=0"` (GEMM alone off). The path whose
  removal fixes it is the culprit → its parity log (5.5) has the detail. If neither fixes it, the difference is in the reference setup
  (fixed experts? same pack?): run `--exact` of the reference against itself first.
- **NaN / inf** → `STRATA_DBG_NAN=1` (upstream's debug switch) reports the first prompt-chunk layer whose MoE output is non-finite and
  checks the state the prompt leaves for the token path. The FP16 GEMM route cannot overflow by design
  (scaled); attention's fp16 KV path can if K/V are huge. Bisect as above.
- **The engine refuses to start in serve mode (logpos)** → `--spec` / `--mtp` must be in the config (IQ packs); the harness keeps them now.
- **"cannot open token file"** → old relative-path bug; the harness now makes paths absolute - make sure you pulled.

### 5.7 `logit_identity.sh` (runbook 6b)

- **1c FAIL (same command twice differs)** → nondeterminism: adaptive expert swaps or a different CPU thread split. The script pins
  `--pcie-frac 0 --adapt-swaps 0`, `STRATA_IQ_MT_MIN=1`, `STRATA_FORCE_AVX2=1`; if it still differs, `golden_compare.py --exact` names the first differing
  position and the size of the difference; record both and report.
- **1a FAIL (port fast-paths-off vs upstream)** → the port changed arithmetic it should not have. Expected differences: none with the
  pins above. Upstream may also fail to *run* on cc 7.0 (`STRATA_EXPERIMENTAL_SM60` build; its bf16 cuBLAS on Volta is unverified) - then
  it is SKIP/FAIL with upstream's log, not a port bug. Rows 1a for Q2_0 skip on a CPU without VBMI (expected here).
- **1b FAIL (mirror on vs off differ)** → serious: a replica is not byte-identical or `xl()` translates wrong. Check the start-up log
  ("64 of 64 sampled pages on node ...", replica built after the primary was filled). Run with `--numa off` until fixed.
- **Tier 2: llama.cpp OOM** → `--llama-cuda-args "-ngl 99 --cpu-moe"` (experts on CPU; the floor then reflects dense math only).
- **"n_vocab differs"** → `--kld-trim-vocab` (golden_compare option).
- **2e FAIL but 2f PASS** → a Volta fast path costs measurable quality: bisect as in 5.6 with the KL report.
- **2e and 2f both FAIL** → Strata (upstream's design) differs from llama.cpp beyond the floor: compare with upstream on another GPU if
  possible; record the KL, top-1 and PPL; it is not a port regression if 1a passes.
- **Thresholds**: factor 2 × floor, 5e-4 nats, 1 % top-1 are provisional. After the first real floor, write the measured values into
  BOX_LOG and adjust `--abs-kl` / `--abs-top1` defaults only with that evidence.
- **Disk**: the reference file ~4 GB (`-c 4096 --chunks 4`); `--ctx 8192 --chunks 3` (a second pass that reaches the sparse selection)
  ~6 GB.

### 5.8 Speed (runbook 7-8)

Decode below expectations → check in this order (each is a one-line test):
1. `cat /proc/sys/kernel/numa_balancing` = 0; governor = performance (`cpupower frequency-info`).
2. The start-up `NUMA:` line says MIRRORED with a replica on the other node; `numastat -p $(pgrep -f engine/strata)` shows one arena per node.
3. The page-size line says THP / hugetlb, not 4 KiB.
4. `stats.txt` (`--stats`): CPU expert compute vs GPU time vs the pool's `drain` ms. CPU-bound → NUMA/tier/prefetch; GPU-bound → kernels
   (profile `decode` window in `profile.txt`); idle → synchronisation/PCIe.
5. A/B one at a time: `STRATA_IQ512=1` (IQ packs), `STRATA_NUMA_MIRROR=0`, `STRATA_Q2_PREFETCH=0` (Q2_0: should be slower without),
   `STRATA_NO_THP=1`, then `./setup.sh --calibrate` (searches `--pcie-frac`, `--spec-min-p`, `--pool-workers`; run it after the NUMA choice).
6. Prefill: `STRATA_PREFILL_MMQ=0` A/B on the 33,000-token prompt (runbook 8 has the loop), then check quality with golden_compare
   `--cand-env "STRATA_PREFILL_MMQ=0"`.

**VRAM OOM** at start or on long prompts → smaller expert cache / prefill chunk (setup's config), `STRATA_PREFILL_MMQ=0` needs extra,
another process on the card, or the replica registration (`STRATA_NUMA_PIN_REPLICA=0` to test).

### 5.9 Runtime crash in normal use

`unspecified launch failure` / `illegal memory access` → reproduce with the smallest prompt, then `CUDA_LAUNCH_BLOCKING=1` and
`compute-sanitizer --tool memcheck` to get the kernel. Port kernels: `prompt_attn_volta_kernel*` (→ `STRATA_VOLTA_ATTN=0`),
`bf16_absmax_kernel` / `bf16_to_f16_*` / `f16_scale_kernel` (→ `STRATA_PREFILL_F16_GEMM=0`). Any other kernel is upstream's: look it up in
`tools/volta/dispatch_audit.md` (every compute-capability decision and what a V100 gets) - a row that keys on `cc_major == 7` as
"Turing" is the classic Volta bug (R7 was one).

---

## 6. DeepSeek on this box (runbook 9)

DS-1 exists: `strata-ds41`, a separate engine (`docs/deepseek/DS1.md`), correct on a tiny model in the cloud's CPU emulation, never
run on a GPU or the real weights. It needs a checkout newer than the Qwen-ready `da0ae45`: after the Qwen tests, make a separate branch
from the current `claude/volta-v100-conversion-8f17vp` (`git checkout -b box/deepseek origin/claude/volta-v100-conversion-8f17vp`) and
follow runbook 9a-9e: download (~411 GB, NVMe), validate (`manifest.py`, `ds41_model_info`), the GPU parity programs (also in
`run_parity.sh` now), the CPU benchmark, then `strata-ds41` on a short prompt and `ds1_compare.py layers` against the oracle. It is slow
by design in DS-1 (one token at a time, prompts included); correctness is the point. Failure playbook for 9e:
- `ds41_model_info` refuses the file -> its message names the key / tensor / shard; a truncated download is the usual cause (re-run the
  download, it resumes).
- `strata-ds41` out of GPU memory -> `--n-slots 0` (every expert on the CPU) first, then a smaller `--n-slots N`; `--max-context` small.
- "the split's doorbell never rang" -> a GPU kernel failed: re-run with `CUDA_LAUNCH_BLOCKING=1` and
  `compute-sanitizer --tool memcheck`; record the kernel name.
- `ds1_compare.py layers` reports a FIRST FAILURE at (position, layer, stage) -> record it with the report; the DS1-x package that owns
  that stage is in `DS1.md` §4. Rounding "flips" counted against a budget are expected, not failures (`DS1_VERIFY.md`).
Record everything in BOX_LOG; do not edit DeepSeek code (send the results back instead). Limits and expectations: `docs/deepseek/PLAN.md` (memory map, PCIe-bound prefill ~22 s per
expert pass, decode bounded by the hit rate and CPU bandwidth: roughly 20-25 tok/s first, 30-45 tuned - estimates).

---

## 7. Sending results back

Everything the cloud session needs is in the runbook's "send back" lines plus BOX_LOG. Commit them on `box/qwen-v100`
(`docs/volta/BOX_LOG.md`, small logs under `docs/volta/box-results/<date>/`; never the multi-GB logits/trace files) and push. A short
summary at the top of BOX_LOG: gates passed, the decode/prefill numbers, every switch you had to set, every patch you made and why.
