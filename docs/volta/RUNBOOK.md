# Day-one runbook: Dell Precision 7920, one Tesla V100 32 GB

One ordered checklist for the first day on the real box: BIOS, OS settings, the driver, `./setup.sh`, then every tool in
`tools/volta/` in the plan's order ([PLAN.md](PLAN.md), section 3), then DeepSeek. Each step has the exact commands and a
**Send back** line: what to copy out of the terminal so the next decision can be made from it. Stop at the first step that fails
and send back what it printed; later steps depend on it.

**The box** (from its owner): Dell Precision 7920, 2x Xeon Gold 6226 (Cascade Lake, 12 cores each, AVX-512 VNNI, no VBMI), 24 x 16 GiB
DDR4-2666 = 384 GiB (6 channels per socket; measured 90-120 GB/s), one Tesla V100 32 GB on PCIe Gen3, Ubuntu 22.04 or 24.04.
Qwen3.8-Flash-Next (Strata) first; DeepSeek-V4.1-Flash (MXFP4 GGUF, ~411 GB) after.

**Nothing here has run on a V100 yet.** Every number the tools print on this box is the first one. The reference behind each step
is [VOLTA.md](VOLTA.md).

Conventions: the repository root is `~/Vibe100` (the engine runs there; the commands below start from it). `.venv/bin/python` is the
Python `./setup.sh` makes (it holds `numpy` and `regex`, which the tools need; with another Python: `pip install numpy regex`).
The gates are the plan's: **Gate 0** the card is a V100 (7.0), **Gate P** the parity programs pass, **Gate Q** the quality
comparison passes, **Gate 1** IQ3 decode reaches 40 tok/s.

| Step | What | Takes | Gate |
|---|---|---|---|
| 0 | BIOS | 10 min | |
| 1 | OS settings | 10 min | |
| 2 | NVIDIA driver (R580, proprietary, held) | 10 min + reboot | Gate 0 |
| 3 | `./setup.sh` | 1-3 h (80 GB download, 10-20 min compile) | |
| 4 | `strata-device --selftest` | 5 min | |
| 5 | `tools/volta/run_parity.sh` | 30-60 min (a full build) | Gate P |
| 6 | `tools/volta/golden_compare.py` | 20-60 min per pack | Gate Q |
| 7 | `tools/volta/profile_decode.sh` | 15 min | |
| 8 | A/B switches | an afternoon | Gate 1 |
| 9 | DeepSeek: download, parity programs, CPU benchmark | download time + 30 min | |

---

## 0. BIOS (power on, F2 -> System Setup)

The menu names below are Dell's 14th-generation ones; another BIOS version may word them differently.

| Setting | Value | Why |
|---|---|---|
| Memory Settings -> **Node Interleaving** | **Disabled** | Enabled makes Linux see ONE blended node. The engine's NUMA mirror (and DeepSeek's per-socket split) then cannot place a copy next to each socket, and silently turns itself off. |
| Processor Settings -> **Sub NUMA Cluster** | **Disabled** | Enabled gives 4 nodes instead of 2. |
| System Profile Settings -> **System Profile** | **Performance** | (or the highest-performance profile offered) no clock ramp-up delays. |
| Memory Settings -> Memory Operating Mode | Optimizer Mode | Mirror / fault-resilient modes cut the usable RAM; `free -g` must show ~377 for 384 GiB installed. |
| Integrated Devices -> Memory Mapped I/O Above 4 GB (a.k.a. Above 4G Decoding) | Enabled | A 32 GB card needs a large PCIe window; with it off the card may not initialise (`nvidia-smi` finds no device). It is normally on already. |

The 7920's PCIe slots are split between the two CPUs: the slot decides which socket owns the card. Step 2's `nvidia-smi topo -m` (and step 7's `topo.txt`) shows it;
nothing needs to be moved.

**Send back:** nothing yet; step 1 checks the result.

## 1. OS settings

```
sudo apt update
sudo apt install -y numactl hwloc pciutils build-essential git curl

# 1a. the two nodes must be visible
lscpu | grep -E "Model name|Socket|NUMA node"      # Socket(s): 2    NUMA node(s): 2
numactl -H                                         # node 0 and node 1, about 192 GB each
free -g | head -2                                  # about 377 total
```

If `NUMA node(s)` says 1, go back to step 0: Node Interleaving is still enabled.

```
# 1b. no page migration behind the engine's back (now, and at every boot)
echo 'kernel.numa_balancing = 0' | sudo tee /etc/sysctl.d/90-strata.conf
sudo sysctl --system | grep numa_balancing

# 1c. the performance governor on every core (a reboot resets it; the unit below makes it stick)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
sort /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | uniq -c           # every CPU (48 with Hyper-Threading on): performance
sudo tee /etc/systemd/system/cpu-performance.service > /dev/null <<'EOF'
[Unit]
Description=CPU governor: performance
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now cpu-performance.service

# 1d. locked memory: the NUMA replicas are mlock'ed; without this the log says "mlock failed (... raise ulimit -l)"
echo '* - memlock unlimited' | sudo tee /etc/security/limits.d/90-strata.conf     # takes effect at the next login
ulimit -l                                                                        # unlimited (after logging in again)

# 1e. huge pages (look; the choice is in 1f)
cat /sys/kernel/mm/transparent_hugepage/enabled                                  # [madvise] is fine; "never" is not
grep -H . /sys/devices/system/node/node*/hugepages/hugepages-2048kB/nr_hugepages
```

**1f. Per-node huge-page pools (optional; skip on the first pass).** Without a pool the engine requests transparent huge pages for the
arena copies (`madvise`), which is enough. A hugetlb pool is the alternative to A/B in step 8. Linux splits `vm.nr_hugepages`
evenly over the nodes, so **each node needs its own pool holding one whole copy** of the arena, set through the node's own sysfs
entry. 2 MiB pages: a copy of G GiB needs G x 512 pages. The arenas: Q2_0 34.0 GB (31.7 GiB, 16,200 pages), IQ2_XS 35.5 GB, IQ3_XXS
42.9 GB (40.0 GiB, 20,500 pages), IQ3_S 50.3 GB (46.8 GiB, 24,000 pages). For IQ3_XXS:

```
echo 21000 | sudo tee /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 21000 | sudo tee /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
grep -H . /sys/devices/system/node/node*/hugepages/hugepages-2048kB/{nr,free}_hugepages     # both nodes 21000
```

A pool that comes up short (memory already fragmented) needs a reboot with the reservation on the kernel command line. A pool is RAM
nothing else can use (82 GiB here): size it for the model you run and release it (`echo 0 | sudo tee ...`) when done.

**Send back:** the `lscpu | grep ...` lines, `numactl -H`, `free -g`, `cat /proc/sys/kernel/numa_balancing`, the `enabled` THP line,
and `ulimit -l`.

## 2. NVIDIA driver: the proprietary R580, held

R580 is the **last** NVIDIA driver branch that supports Volta; CUDA 12.8 needs 570 or newer and CUDA 13 needs 580 or newer, so R580
is the one driver that fits everything. Install the **proprietary** package, `nvidia-driver-580` - **not** `nvidia-driver-580-open`:
NVIDIA's open kernel modules support Turing and newer only, and do not drive a V100.

```
sudo apt install -y dkms linux-headers-$(uname -r)
ubuntu-drivers list --gpgpu                 # nvidia-driver-580 is listed (and -open and -server variants: do not take those)
sudo apt install -y nvidia-driver-580
sudo reboot
```

If the box has Secure Boot on, the DKMS module must be signed: the installer offers to enrol a key (MOK) at the next boot; or turn
Secure Boot off in the BIOS.

After the reboot, **hold** the driver packages so no `apt upgrade`, "Additional Drivers" click or `ubuntu-drivers autoinstall` moves the
box to a branch that no longer lists the card:

```
dpkg-query -W -f='${Package}\n' 'nvidia-*-580' 'libnvidia-*-580' 'xserver-xorg-video-nvidia-580' 2>/dev/null | xargs -r sudo apt-mark hold
apt-mark showhold
sudo nvidia-smi -pm 1                       # persistence mode: the first CUDA call is not delayed by a driver reload
```

Do **not** `apt install cuda` or `cuda-drivers`: those meta-packages pull the newest driver. `./setup.sh` installs only
`cuda-toolkit-12-8` (no driver).

**Gate 0:**

```
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,pci.bus_id --format=csv     # Tesla V100..., 7.0, 580.x, 32768 MiB
nvidia-smi topo -m                                                                              # the card's CPU / NUMA affinity
nvidia-smi -q > ~/nvidia-smi-q.txt
```

**Send back:** the first `nvidia-smi` line, the `topo -m` table, and from `~/nvidia-smi-q.txt` the lines with `Product Name`,
`Driver Version`, `CUDA Version`, the PCI section (`Link Width` and `PCIe Generation`, Max and Current: expect x16 and Gen3) and
`Clocks Event Reasons`.

## 3. `./setup.sh`

```
cd ~/Vibe100
./setup.sh --check              # card, driver and RAM only; says whether a CUDA 12 toolkit was found
./setup.sh                      # the first run: asks a few questions, then installs everything
```

On a V100 the setup skips the ready-made engine (it is built with CUDA 13, which dropped Volta) and:

1. creates `.venv` and pip-installs `numpy`, `regex`, `cmake`, `ninja` and the other packages into it (the pip `cmake` matters:
   `CMakeLists.txt` needs 3.24 and Ubuntu 22.04's apt has 3.22);
2. asks before installing `build-essential` and NVIDIA's apt repository key + **`cuda-toolkit-12-8`** (sudo; the toolkit only, no driver);
3. downloads llama.cpp at the pinned commit into `third_party/llama.cpp` (ggml for the build);
4. compiles the engine for **sm_70** with CUDA 12.8 into `build/` (10-20 minutes, once) and copies it to `engine/strata`;
5. downloads the model into `Strata-data/` next to the repository (66-84 GB depending on the size), prepares the pack, fetches the
   MTP draft layer (~5 GB), writes `strata-<model>.json` and `run-<model>.sh`, and starts the server.

Answers: the original Qwen3.8-Flash-Next; size **IQ3_XXS** first (the size Gate 1 is stated for) and, if you have the disk, **Q2_0** second (`./setup.sh --setup --model Q2_0 --no-start`): the Cascade Lake VNNI path runs the
canonical Q2_0 pack since WP-E, and it is the pack for which the golden comparison can dump full logits (step 6). Context 131072 (the
recommendation for 32 GB), no vision. Add `--no-start` to stop after the install. The config files are `strata-iq3_xxs.json` and
`strata-q2_0.json` in the repository root; the engine's log is `strata-<model>.log`.

Not in a container. In Docker the NUMA mirror needs `--cap-add SYS_NICE` and a seccomp profile that allows `move_pages` (or
`--security-opt seccomp=unconfined`): [VOLTA.md](VOLTA.md), "Docker".

After the model has started once (the end of `./setup.sh`, or `./run-iq3_xxs.sh`), look at what the engine did with the two sockets:

```
grep -n "NUMA" strata-iq3_xxs.log | head -20
```

It should say 2 NUMA nodes, the GPU's node, `expert arena MIRRORED: primary ... on node N ... + 1 replica ...`, the page sizes
(`THP` / `2 MiB hugetlb pages` / `4 KiB pages`) and `64 of 64 sampled pages on node ...` for each copy. A line that says it kept ONE
copy gives the reason.

**Send back:** the last 30 lines of the setup output if anything failed; otherwise `engine/BUILD.json`, the `NUMA:` lines of the log, and
`numastat -p $(pgrep -f 'engine/strata')` while the model is up (about one arena's worth of memory on EACH node).

## 4. `strata-device --selftest`

`./setup.sh` builds only the `strata` target, so the quick card check has to be built first:

```
.venv/bin/cmake --build build --target strata-device -j4
build/strata-device --selftest              # ends with: strata-device selftest OK
```

It prints the card (compute capability 7.0), its multiprocessors, VRAM and driver / runtime versions, the planner's budget against
the free VRAM, and allocates and checks real memory on the card. An engine with no code for the card says so here, in one sentence,
and names the `-DCMAKE_CUDA_ARCHITECTURES` to rebuild with. (Step 5 builds it again in its own directory:
`build-sm70/strata-device`.)

**Send back:** the whole output.

## 5. `tools/volta/run_parity.sh` (Gate P)

Builds for sm_70 into `build-sm70/` with CUDA 12.8 and runs every GPU parity program, every registered test and the SASS audit
(`--ggml-dir third_party/llama.cpp` reuses the llama.cpp that setup downloaded at the pinned commit instead of fetching one):

```
tools/volta/run_parity.sh --ggml-dir third_party/llama.cpp --require-v100
```

It uses `.venv/bin/cmake` (and stops with a message if the cmake it finds is older than 3.24), `/usr/local/cuda-12.8/bin/nvcc`, and the
`cuobjdump` next to that nvcc. It runs: `qsa_prompt_attn_parity` (default, long-context, and `STRATA_VOLTA_ATTN=0`), `gemm_volta_parity`,
`kv_hybrid_parity`, the DeepSeek GPU programs `ds41_router_parity`, `ds41_split_parity` and `ds41_expert_parity` (each `--selftest`;
skipped with a note if the target does not exist), ctest, and `sass_audit.py`. The table ends with
`GATE P (parity, docs/volta/PLAN.md): PASS` or `FAIL (n failing)`; `SKIPPED (unverified!)` lines are checks that did not run. With the
build already done, `--skip-build` re-runs only the checks.

**Send back:** the whole summary table, and the logs: `tar czf ~/parity-logs.tgz -C build-sm70 parity-logs`. If anything failed, the
failing program's log in that folder is what matters.

## 6. `tools/volta/golden_compare.py` (Gate Q)

Compares the Volta fast paths (the WMMA prompt attention, the FP16 GEMM route) against the FP32 reference paths of the **same engine on the
same card** (`STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0`) over a 33,000-token prompt built from this repository's own sources, so the batched
path and the sparse selection are both exercised (both are invisible on a short prompt). Run it from the repository root, or with absolute
paths: every path handed to the engine is made absolute.

```
.venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --tail 512 --dry-run
.venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --tail 512 --workdir ~/golden/iq3_xxs
```

The first command only prints the plan and the two engine commands. The second runs the engine twice (reference first, the slow one) and takes
tens of minutes.

- **IQ packs** (`strata-iq3_xxs.json`, `strata-iq2_xs.json`, `strata-iq3_s.json`) are *native* packs: the engine's token loop, the only writer of
  `--dump-logits`, never runs for them. The harness therefore uses the `logpos` source on its own: the resident engine (`strata --serve`, which
  needs the config's `--spec 4 --mtp ...`, kept) with `STRATA_LOGPOS`, scoring the last 512 tokens through the verify windows after the rest went
  through the batched path. Metrics: top-1 agreement, perplexity, paired NLL delta, max |d log p|, NaN / inf; no KL.
- **Q2_0** (`strata-q2_0.json`) uses the logits dump (`--dump-logits`, the `dump` source): full logits per position, so top-1, max |dlogit|, KL
  (mean / p99 / max), perplexity and the NaN scan:

  ```
  .venv/bin/python tools/volta/golden_compare.py --engine-config strata-q2_0.json --prompt-name long --tail 512 --workdir ~/golden/q2_0
  ```

- The last line says `GATE Q (docs/volta/PLAN.md): PASS` when the run is the gate (top-1 >= 99%, perplexity within 2%, no NaN / inf, >= 500 rows,
  a batched part longer than 2,051 tokens); `not established - ...` says what the run lacked; `FAIL` lists which threshold failed. Exit status 0 / 1 / 2 =
  pass / fail / the harness could not run.
- `--mode greedy --max-new 256` shows what a user would see (how long the two runs stay identical); it is not the gate.
- A crash of the candidate run (`unspecified launch failure`) is reported as a failure with the candidate's log: a `BPT.TRAP` on the V100 looks like that.

**Send back:** `~/golden/<pack>/report.json` and `report.txt`, and `cand.log` and `ref.log` from the same folder (they hold the engine's NUMA
start-up lines and the batched-prefill speed of each run). Not the `*.bin` logits files (`--delete-logits` removes them).

## 7. `tools/volta/profile_decode.sh` (the Phase 3 profile)

Where one decode step goes. It runs the engine, so it comes after the build and the checks of steps 4-6. It runs the engine exactly as the server does (`--stats`, 128 tokens, the config's arguments), then the
pure-GPU floor (`--gpu-only-full`), then Nsight Systems on the same decode. It saves `numactl -H`, `nvidia-smi topo -m` and, when Intel MLC
is installed, `mlc --bandwidth_matrix`.

```
ls /usr/local/cuda-12.8/bin/nsys     # CUDA 12.8's own Nsight Systems; the script prefers it over any other
# missing?  sudo apt install -y cuda-nsight-systems-12-8 cuda-nsight-compute-12-8
```

Intel MLC is optional: download "Intel Memory Latency Checker" (free) from Intel's site, unpack it, `sudo modprobe msr` (it uses the MSRs to
switch the hardware prefetchers off) and give the script its path. Without it the script says so and skips that one measurement.

```
tools/volta/profile_decode.sh --config strata-iq3_xxs.json --n-tokens 128 --strict --mlc ~/mlc/Linux/mlc --outdir ~/profile/iq3_xxs
```

`--strict` stops unless the card reports 7.0 (Gate 0). The Nsight trace covers the whole process, start-up included (the dense-weight upload
and the expert-cache fill are tens of GB of PCIe traffic that is not decoding), so the table cuts every per-token figure to the **decode window**
(the last "decode N tokens in X ms" of the profiled run, from the trace's SQLite export) and labels the others "whole run". If `nsys`
complains about CPU sampling: `sudo sysctl kernel.perf_event_paranoid=2`. `--ncu` adds Nsight Compute (slow); it needs GPU counter access
(`ERR_NVGPUCTRPERM`: run the script with `sudo`, or `options nvidia NVreg_RestrictProfilingToAdminUsers=0` in `/etc/modprobe.d/`).

**Send back:** `~/profile/iq3_xxs/profile.json` and `profile.txt` (the table), `numa.txt`, `topo.txt`, `mlc_bandwidth_matrix.txt`, `stats.txt` (the
engine's `--stats` output; its first lines are the NUMA start-up lines) and `nsys_run.txt`. The `decode.nsys-rep` is large: only if asked.

## 8. A/B switches (Phase 3 tuning)

One variable at a time, the same prompt, and **discard the first run after a boot** (it reads the model from disk). Prefill speed comes from the
long prompt that step 6 wrote (`~/golden/iq3_xxs/long.ids`); decode speed from the default chat prompt. The table rows to compare are `prefill tok/s`,
`decode tok/s`, `CPU expert compute` and the pool's `drain` time in `stats.txt`.

```
# prefill A/B: one engine run each, prompt = the 33,000-token one, 64 tokens generated
for v in "" "STRATA_PREFILL_MMQ=0"; do
  env $v tools/volta/profile_decode.sh --config strata-iq3_xxs.json --tokens-file ~/golden/iq3_xxs/long.ids \
      --n-tokens 64 --no-nsys --no-floor --outdir ~/ab/mmq_${v:-default}
done
# decode A/B: the same, with the default chat prompt
for v in "" "STRATA_IQ512=1"; do
  env $v tools/volta/profile_decode.sh --config strata-iq3_xxs.json --no-nsys --no-floor --outdir ~/ab/iq512_${v:-default}
done
```

| Switch | Compares | Look at |
|---|---|---|
| `STRATA_PREFILL_MMQ=0` vs default | the prompt's expert GEMMs as ggml's MMQ on the dp4a **CUDA cores** (the default; the V100 has no int8 tensor cores) vs dequantise to FP16 + cuBLAS **HMMA**. Upstream's own rule would pick cuBLAS above a batch of 64 on this card. | prefill tok/s; then quality: `golden_compare.py ... --cand-env "STRATA_PREFILL_MMQ=0"` |
| `STRATA_PREFILL_F16_GEMM=0` vs default (`auto` = FP16 tensor cores on a Volta) | what the Volta dense-GEMM route buys over upstream's bf16 call on the CUDA cores (`1` is the same as `auto` on a V100) | prefill tok/s |
| `STRATA_VOLTA_ATTN=0` vs default | what the WMMA prompt attention buys over the FP32 fallback | prefill tok/s on the long prompt |
| `--engine-args "--numa off"` / `STRATA_NUMA_MIRROR=0` vs default (mirror) vs `profile_decode.sh --numa-interleave` | one copy by first touch vs one copy per socket vs `numactl --interleave=all`. Same prompt each time. | decode tok/s, the pool's `drain` ms, `numastat -p`; the log's `NUMA:` lines say what each run did |
| `STRATA_IQ512=1` | AVX-512 i-quant expert rows on Cascade Lake (the default keeps AVX2 for IQ2_XS / IQ3_XXS / IQ3_S: AVX-512 measured slower on a Cascade-Lake-class VM; the real Xeon decides) | decode tok/s on IQ3_XXS and IQ2_XS |
| page size: default (THP) vs the hugetlb pools of step 1f vs `STRATA_NO_THP=1` | TLB cost of the arena copies | decode tok/s; the log's page-kind line |
| `./setup.sh --calibrate` | the engine's own search: `--pcie-frac` (PCIe Gen3), `--spec-min-p`, `--pool-workers`; keeps a setting only if > 3% faster. Run it after the NUMA choice is final. | 5-10 minutes; it remembers the result per PC and model |

Gate 1 is read from the decode tok/s of the IQ3_XXS pack after this tuning (>= 40 tok/s, with Gates P and Q passed); if it is not reached, the
plan says to stop and reassess before any DeepSeek engine work.

**Send back:** for each A/B the two `profile.txt` (or just the `decode tok/s` / `prefill tok/s` lines of the two `stats.txt`) and which switch
was flipped; the `--calibrate` result.

## 9. DeepSeek-V4.1-Flash

Everything in this step except the engine itself already exists: the NumPy oracle, the GGUF tooling, the CPU and GPU expert kernels
(`docs/deepseek/PLAN.md`). The engine (DS-1) comes after Gate 1, so this step only checks the weights and the kernels on the real hardware.
Start the download early if the link is slow: it needs no GPU.

**9a. Download** (~411 GB: 12 shards of 375.8 GiB plus the 7.43 GiB DSpark sidecar; on the **NVMe** drive: the Engram tables, 97.28 GiB, stay
memory-mapped and are read at random, 48 rows of 136 B per token):

```
.venv/bin/python -m pip install -U "huggingface_hub[cli]"
.venv/bin/huggingface-cli download mxxm-t/DeepSeek-V4.1-Flash-GGUF --local-dir /nvme/models/DeepSeek-V4.1-Flash-GGUF
# newer huggingface_hub releases call the command `hf`:  .venv/bin/hf download mxxm-t/DeepSeek-V4.1-Flash-GGUF --local-dir ...
```

Re-running the same command resumes. The folder must end up with `DeepSeek-V4.1-Flash-MXFP4-000NN-of-00012.gguf` for NN = 01..12 and
`DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf`.

**9b. Validate the files** (names shard 1; its siblings and the DSpark file are found beside it; needs only numpy):

```
GGUF=/nvme/models/DeepSeek-V4.1-Flash-GGUF
.venv/bin/python tools/ds41/manifest.py $GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf --out ~/ds41.manifest.json
.venv/bin/python tools/ds41/expert_layout.py verify ~/ds41.manifest.json --sample 8 --c-oracle
```

Exit status 0 = the contract is satisfied (warnings allowed), 2 = refused (it says why: a truncated shard, a wrong expert type, ...).

**9c. The GPU parity programs** (built by step 5 into `build-sm70/`; synthetic data, no model; PASS looks like `ALL PASS (n checks)`):

```
build-sm70/ds41_router_parity --selftest
build-sm70/ds41_split_parity --selftest
build-sm70/ds41_expert_parity --selftest
build-sm70/ds41_router_parity --bench          # timing only: the router at T = 1
build-sm70/ds41_expert_parity --bench          # timing only: GB/s and microseconds per layer, 6 hits at T = 1, 24 at T = 4
```

(`ds41_split_parity` has only `--selftest`, `--seed` and `--T`; run without a flag a program does its checks and its timing.) The kernels'
own estimate is about 150 us per layer for 6 hits at T = 1, bandwidth-bound; this is where it is measured.

**9d. The CPU expert benchmark** (`ds41_cpu_mxfp4_bench`, also in `build-sm70/`): GB/s of expert weights consumed by the AVX-512 / AVX2 MXFP4
kernels against a plain-read ceiling, on this CPU. Run it per socket with the memory on the same node, and once across the link:

```
numactl --cpunodebind=0 --membind=0 build-sm70/ds41_cpu_mxfp4_bench --threads 1,4,8,12,24 --tokens 1,4
numactl --cpunodebind=1 --membind=1 build-sm70/ds41_cpu_mxfp4_bench --threads 1,4,8,12,24 --tokens 1,4
numactl --cpunodebind=0 --membind=1 build-sm70/ds41_cpu_mxfp4_bench --threads 12 --tokens 1,4        # across UPI: the penalty
numactl --cpunodebind=0 --membind=0 build-sm70/ds41_cpu_mxfp4_bench --threads 12 --tokens 1,4 --box-gbps 100
```

Each node has 12 cores (24 hardware threads with Hyper-Threading on). The bench pins its threads to the CPUs it is allowed (so `--cpunodebind` works), cycles through
40 half-experts (376 MB: far past the 19 MB of L3) so the weights come from DRAM, and its tok/s lines use `--box-gbps` (default 100): give it
the node-local read bandwidth that `mlc_bandwidth_matrix.txt` shows once you have it.

**Send back:** the manifest's `RESULT:` line and printed findings and memory plan, the output of each `ds41_*_parity` run, and the three
`ds41_cpu_mxfp4_bench` outputs.

---

## What to send back, in one go

```
cd ~
tar czf ~/send-back.tgz --exclude='*.bin' --exclude='*.nsys-rep' --exclude='*.sqlite' \
    nvidia-smi-q.txt parity-logs.tgz golden profile ab ds41.manifest.json Vibe100/engine/BUILD.json Vibe100/strata-*.log 2>/dev/null
ls -lh ~/send-back.tgz
```

(The large logits files, Nsight traces and SQLite exports stay behind; send them only if asked.)
