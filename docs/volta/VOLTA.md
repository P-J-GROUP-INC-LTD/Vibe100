# Strata on an NVIDIA V100 (Volta) - the Vibe100 port

Strata was written for RTX 20 and newer cards. This port makes the **V100 / Titan V (Volta, compute capability 7.0)**
a normal target: one command installs it, and the three places where Strata used instructions Volta does not have
got Volta versions. Everything else is upstream Strata 0.1.31 (see [UPSTREAM.md](../../UPSTREAM.md)); what was found
in its source and what was changed is in [PLAN.md](PLAN.md).

> **Read this first: nothing here has run on a V100 yet.** The port was written and compiled for sm_70 on a machine
> without a GPU. The machine code was inspected and the CPU-side checks run, but the first start on your card is
> the first real test - which is why this page has a section on [checking it](#checking-it-on-your-v100) and one
> on [what is not known](#known-limits). Upstream's speed numbers are RTX numbers; they say nothing about a V100.

## Why a V100 needs its own steps

- **The ready-made engine has no Volta code.** It is built with CUDA 13, and CUDA 13 dropped Volta (nvcc 13 cannot
  make sm_70 code at all). A V100 therefore always **compiles the engine on its own PC, with CUDA 12.8** - the
  installer does it for you.
- **Volta's tensor cores are an older kind.** Strata's tensor-core kernels (prompt attention, the long-context
  selection, the dense prompt GEMMs) use instructions from Turing and Ampere. On a V100 the prompt attention has its own
  kernel, the dense GEMMs go through FP16 tensor cores (Volta has no BF16 ones), and the long-context selection runs on the
  ordinary cores. Reading a prompt works the same way as before; only how the card does it differs.
- **Reading the answer out (decoding) needs no port.** It runs on integer dot products and ordinary FP32/FP16 math that
  Volta has. How fast it is depends on the GPU's memory, how many of the model's experts fit in VRAM, and on the CPU -
  the same things as on any card.

## What you need

| | |
| --- | --- |
| **The card** | Tesla V100 16 GB or 32 GB (PCIe or SXM2), Quadro GV100, Titan V (12 GB). `nvidia-smi --query-gpu=name,compute_cap --format=csv` must say **7.0**. 32 GB is the one worth having: the GPU's VRAM holds a copy of the most-used experts, so more VRAM means more answers served from the card. |
| **The driver** | **570 or newer** (CUDA 12.8 needs it) and **no newer than the 580 series**: R580 is the last NVIDIA driver branch that supports Volta. |
| **CUDA toolkit** | **12.8** (any 12.x from 12.0 to 12.9 is accepted; 12.8 is the one this port was checked with). **Not 13.** The installer finds an existing 12.x or installs 12.8 next to whatever CUDA you have; it passes nvcc to CMake itself, so your default CUDA is not touched. |
| **RAM and disk** | The same as for any card - [the table in the README](../../README.md#which-model-should-i-pick). A V100 server usually has plenty; with 64 GB of RAM every size fits. |
| **A compiler** | Linux: `g++` (installed with `build-essential`). Windows: Visual Studio 2022 Build Tools (C++). The installer offers to install them. |
| **The OS** | Linux (Ubuntu 22.04 / 24.04 gets the CUDA 12.8 install automatically; other distributions: install CUDA 12.8 yourself, the installer finds it in `/usr/local/cuda-12.*` or `/opt/cuda*`). Windows is possible for a Titan V / Quadro GV100 through the same installer; that path is untested. |

## Install

### One command (recommended)

```
./setup.sh                 # Linux
START-HERE.bat             # Windows (double-click)
```

The questions are the usual ones. On a V100 the setup:

1. lists your GPUs and says that the V100 can be used (a card below Volta, such as a P4000, is refused);
2. checks the driver (570 or newer for a V100);
3. **skips the ready-made engine** and says why;
4. looks for a CUDA **12.x** toolkit - even when a CUDA 13 is first on your PATH, the 12.x beside it is used. With none,
   it asks before installing one: `cuda-toolkit-12-8` from NVIDIA's apt repository on Ubuntu 22.04 / 24.04 (sudo asks
   for your password), `winget install --id Nvidia.CUDA --version 12.8` on Windows. Elsewhere it tells you to install
   CUDA 12.8 yourself (https://developer.nvidia.com/cuda-12-8-1-download-archive) and run it again;
5. compiles the engine for sm_70 (10-20 minutes, once), then downloads and prepares the model like for any card.

`./setup.sh --check` does only the card, driver and RAM checks and tells you whether a CUDA 12 toolkit was found.

Later runs start the model right away. After a `git pull` that changes the engine's source, the engine is compiled
again (only what changed); it is never replaced by a ready-made one.

**Several cards.** A V100 can share a model with RTX 20 / 30 / 40 cards (`--gpus 0,2`): one engine is built for
all of them with CUDA 12.8. A V100 **cannot** share one with an **RTX 50** card: CUDA 13 cannot build for Volta, and an
RTX 50 engine built with CUDA 12.8 crashed on long prompts (upstream issues #220, #224). The setup does not offer that
pair and stops with a message if you ask for it; pick one card (`--gpu N`) or keep a second Strata folder for the
other card. (A mixed V100 + RTX 40 box is allowed but nobody has run it.)

### By hand (developers)

```
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --target strata -j4
```

- No `-DSTRATA_EXPERIMENTAL_SM60=ON` is needed any more: sm_70 is a supported architecture. That flag is only for
  Pascal (sm_60 / 61 / 62), which this port does not cover.
- `-DCMAKE_CUDA_ARCHITECTURES="70;75;80;86;89"` builds one engine for a V100 and RTX 20 / 30 / 40 cards.
- With a CUDA 13 compiler, CMake stops at once and says that CUDA 13 dropped Volta and which nvcc to give it.
- The GPU test programs: `-DSTRATA_BUILD_TESTS=ON`, or just the ones for this port with
  `-DSTRATA_PARITY_PROMPT_ATTN=ON -DSTRATA_PARITY_PREFILL_GEMM=ON` (targets `qsa_prompt_attn_parity` and
  `gemm_volta_parity`). `strata-device` (`cmake --build build --target strata-device`) is the quickest check of the card.
- To run the model you still need the prepared model files; the one-command install makes them (`./setup.sh --no-start`
  compiles the engine with the same options and prepares everything).

### Docker

The image needs a CUDA 12 base - a build argument - and the architecture named:

```
docker build -t vibe100 \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  --build-arg CUDA_ARCHITECTURES=70 .
docker run --rm --gpus all -p 8080:8080 --ulimit memlock=-1 -v strata-data:/data vibe100
```

The host needs the NVIDIA Container Toolkit and a driver 570 or newer (and not past the 580 series). The default
base (CUDA 13) is unchanged for RTX cards; with it `CUDA_ARCHITECTURES=70` stops the build with the reason. The
other options (`-e MODEL=...`, `-e LOW_RAM=on`, ...) are the same as in the
[README](../../README.md#install). (The Docker build has not been run on a V100 host either.)

## The two run-time switches

Both are environment variables, read once when the engine starts. Leave them alone for normal use; they exist so the
Volta paths can be switched off to compare against, and for testing.

| Variable | Values | What it does |
| --- | --- | --- |
| `STRATA_VOLTA_ATTN` | unset or `1` (default) / `0` | **Prompt attention on a V100.** `1`: the new Volta tensor-core kernel. `0`: the older per-query FP32 kernel (what upstream falls back to when the tensor-core kernel cannot be used) - the reference the new kernel is checked against. Only matters while a prompt is read with a 16-bit, 8-bit or K8V4 KV cache; has no effect on other cards. |
| `STRATA_PREFILL_F16_GEMM` | `auto` or unset (default) / `0` / `1` | **The dense GEMMs that read a prompt.** `auto`: on a card with FP16 but no BF16 tensor cores (compute capability 7.0 to 7.5: the V100, and also an RTX 20; small GEMMs and Pascal cards keep the upstream call) they run as FP16 tensor-core GEMMs with FP32 sums - the BF16 values are converted exactly, with a power-of-two scale per piece so nothing can overflow FP16. `0`: always upstream's BF16 cuBLAS call (on a V100 that runs on the ordinary cores, about 8x slower than the tensor cores). `1`: use the FP16 route on every card, to test it on a newer one. |

Set them for one run on the command line:

```
STRATA_VOLTA_ATTN=0 ./run-iq2_xs.sh
```

or for good in the model's config file `strata-<model>.json` (the server passes `env` to the engine):

```
"env": {"STRATA_VOLTA_ATTN": "0"}
```

## Checking it on your V100

These tools are in `tools/volta/`; each one explains itself with `--help` (or its header). The order matters: each step
answers a question the next one depends on.

| Step | Tool | What it tells you |
| --- | --- | --- |
| 0 | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv` | The card really is 7.0 and the driver is in range. A P4000 (6.1) or an RTX card tells you nothing about a V100. |
| 1 | `build/strata-device --selftest` | The engine starts on the card: it knows the card is allowed (7.0) and that **this build has code for it**. If it does not, it says so in a sentence and names the `-DCMAKE_CUDA_ARCHITECTURES` to rebuild with. |
| 2 | `tools/volta/run_parity.sh` | Runs every GPU parity program (prompt attention and the prefill GEMM against FP64 / FP32 references, on synthetic data, no model needed) and prints one pass / fail table. All must pass before anything else means anything. |
| 3 | `tools/volta/golden_compare.py` | Correctness of the whole model, not one kernel: top-1 agreement, largest logit difference, KL and perplexity, and a NaN / inf scan - the Volta fast paths against the FP32 reference paths on the same card (that is what the two switches above are for), or against a reference-logits file made on another GPU. |
| 4 | `tools/volta/profile_decode.sh` (with `summarize_profile.py`) | Where one decode step spends its time: GPU kernels, PCIe copies or the CPU's expert work. Run this before trying to tune anything. |
| - | `tools/volta/sass_audit.py` | For people building the engine: reads the sm_70 machine code and fails when a kernel that must not trap does, or when a tensor-core kernel compiled to plain math. `tools/volta/compile_one.py` compiles one file for sm_70 and reports it. |

The pass marks the plan sets are in [PLAN.md](PLAN.md) (section 3):
top-1 agreement of at least 99% over 500 tokens, perplexity within 1-2%, no NaN or inf; and, as a goal rather than a
prediction, 40 tokens/s or more on the IQ3 size on one V100. If those fail, the plan says to stop and rethink before
building anything on top.

## Getting the speed

**NUMA (two-socket servers).** Most V100 boxes are dual-socket Xeons, and the CPU's share of the work is limited by
memory bandwidth. Linux puts a program's memory on the socket that first touched it; the engine loads 23-50 GB of experts
into RAM, and CPU threads on both sockets then read them, so about half of those reads cross the link between the sockets
(UPI on a Xeon), and the pinned memory the card copies from may sit far from the card's PCIe lanes. The engine now fixes this
itself by **mirroring the experts: one full copy in each socket's RAM, every CPU thread reading its own socket's copy**
(`--numa auto`, the default). Someone who tried full mirroring of Qwen3.8-Flash-Next in another engine on a dual-Xeon
V100 box (Dell Precision 7920, 2x Xeon Gold 6226) reported about 2x decode; that figure is theirs, not measured with
this engine - the A/B below is how to find out what it is on yours.

What `--numa auto` does, on Linux with two or more NUMA nodes, when the experts are the normal full-RAM arena (the default):

- the **primary copy** goes on the GPU's node (read from `/sys/bus/pci/devices/<card>/numa_node`; a BIOS that says `-1`
  gives no answer, and the engine then assumes node 0 and says so in the log). It is bound to that node before its first
  page is touched, registered with CUDA as before, and it is the **only copy the card ever reads** (expert cache fills,
  `--pcie-frac` misses, prompt streaming);
- one **replica** goes on every other node that has CPUs: bound to that node, filled by threads running there, locked in RAM
  like the arena (`STRATA_ARENA_LOCK=0` leaves it pageable), never registered with CUDA;
- every CPU worker, and the host thread's own share of the drain, reads the copy of **its own node**; the host thread that
  drives the card is put on a core of the **GPU's node**, and the workers are spread over both nodes;
- it costs one more arena of RAM per extra node (23-50 GB, 384 GB on this box) and a few seconds at start: the replica is
  copied from the primary by threads pinned to the replica's node. A node must have room for its copy plus
  `STRATA_NUMA_HEADROOM_GIB` (default 6) of other memory, counting the file cache it can reclaim, or the engine keeps one copy.

It keeps **one copy, and prints one line saying why**, when: the PC has one NUMA node (or BIOS node interleaving hides the
second one - turn that off for this); the experts are not the full-RAM arena (`--mmap-experts`, `--resident-experts`,
`--resident-budget-gib`: they sit in the OS file cache, which the engine does not place; `--shared-expert-arena`); every CPU
the process may use is on the GPU's node (`taskset`, `numactl --cpunodebind`); a node lacks the room; or `mbind` is refused
(some containers). The startup log shows the whole decision:

```
strata generate: NUMA: 2 NUMA nodes: node0 cpus 0-11,24-35 (...), node1 cpus 12-23,36-47 (...); GPU 0000:3b:00.0 on node 1
strata generate: NUMA: expert arena MIRRORED: primary 31.64 GiB on node 1 [the GPU's node; CUDA-registered 31.64 GiB; bound; the only copy any GPU DMA reads] + 1 replica of 31.64 GiB on node 0
strata generate: NUMA:   primary on node 1: 64 of 64 sampled pages on node 1
strata generate: NUMA:   replica on node 0: 4 KiB pages; mbind(MPOL_BIND) to node 0; 31.64 GiB copied in 2.9 s; 64 of 64 sampled pages on node 0; mlock
strata generate: NUMA: expert pool workers per node: node0=11, node1=12; the host thread goes to core 12 on node 1 (the GPU's); ...
```

(The lines are the shape, not output from a V100: the placement lines are `move_pages` queries on a sample of pages after
the copy; a replica whose pages are not on its node is dropped and the line says `DROPPED`. If a replica's line ends in
`mlock failed (... raise ulimit -l)`, the replica stays pageable: `ulimit -l unlimited` (a systemd unit: `LimitMEMLOCK=infinity`)
locks it; with no swap configured that changes nothing.)

Controls: `--numa auto|mirror|off` (`mirror` asks for it; the engine still refuses, with the reason, where it cannot), set
at setup or later with `./setup.sh --numa off` (saved for the model), and the environment `STRATA_NUMA_MIRROR=0` / `=1`,
which overrides the option for one run - the A/B switch. If the log says the GPU's node was ASSUMED (the BIOS reports
`numa_node -1`), tell it with `STRATA_NUMA_GPU_NODE=N` (what `nvidia-smi topo -m` shows: the card's CPU affinity).

**Do not combine it with `numactl --interleave=all`** (or BIOS node interleaving): interleaving spreads every page of every
allocation over both nodes, which is the opposite of placing a copy on each. The mirror binds its copies explicitly, and a
per-mapping policy outranks the process's, so they should still land where they belong - but then the interleave only
spreads everything else, and the engine prints a warning when it sees a process-wide interleave policy. The interleave
remains the right quick fix when the mirror cannot be used (low-RAM modes, not enough RAM for two copies):

```
STRATA_NUMA_MIRROR=0 numactl --interleave=all ./setup.sh            # or ./run-<model>.sh
```

**Checking it on the box.** `nvidia-smi topo -m` shows which socket the GPU hangs off (the engine's log must agree). While the
model runs, `numastat -p $(pgrep -f strata)` should show roughly one arena's worth of memory on EACH node (the primary on the
GPU's node, the replica on the other); with `STRATA_NUMA_MIRROR=0` the same command shows the arena split by whichever threads
touched it first. To measure the gain, run the same prompt three times and compare tokens/s and the pool's `drain` time in
`--stats` (the CPU path is memory-bound, so the drain is what shrinks): `STRATA_NUMA_MIRROR=0` (before), the default (mirror),
and `STRATA_NUMA_MIRROR=0 numactl --interleave=all` (the old quick fix). The signs of it working: the mirror's drain time is
clearly below both, and `pcm-memory` / `mlc` shows UPI traffic during decode falling to a fraction of the other runs'.
If mirror is not faster than interleave, look first at whether the log says the primary landed on the GPU's node.

**BIOS (Dell Precision 7920 and similar dual-Xeon workstations).** In the system setup: *Memory Settings → Node
Interleaving: **Disabled*** (with it enabled Linux sees one blended node and the engine cannot put a copy next to each
socket — the mirror and the DeepSeek port's per-socket split both need the two nodes); *Sub-NUMA Clustering: Disabled*
(2 nodes, not 4); *System Profile: Performance* (or the highest-performance profile offered). The 7920's PCIe slots are
divided between the two CPUs, so the slot decides which socket owns the card: `nvidia-smi topo -m` shows it, and a
second GPU is best placed on the other CPU's slots. On Linux, `kernel.numa_balancing=0` (`sysctl`) keeps the kernel from
migrating pages behind the engine's back, and the `performance` CPU governor avoids clock ramp-up delays.

**PCIe Gen3.** A V100 talks to the host at PCIe 3.0 x16 (about 12 GB/s in practice), slower than the Gen4 / Gen5 of
newer cards. Strata copies the experts the card is missing over PCIe, or computes them on the CPU; how much goes each way
(`--pcie-frac`), how sure the draft layer must be (`--spec-min-p`) and the number of CPU threads (`--pool-workers`)
are best measured on the machine:

```
./setup.sh --calibrate        # Windows: START-HERE.bat --calibrate    (about 5-10 minutes)
```

It keeps a setting only when it is more than 3% faster, and remembers the result per PC and model. Run it once after the
install, and again after a NUMA change.

**VRAM.** On a 32 GB card the setup suggests a 128K context and keeps the rest of VRAM for experts; 16 GB gets 64K. A
longer context takes VRAM from the experts' cache (more in [DETAILS.md](../DETAILS.md)).

## Known limits

- **Nothing is measured on a V100 by this port.** No speed, no quality, no memory number. Gate 0 in the plan is
  exactly that: numbers from a V100.
- **The NUMA mirror has only run against FAKE two-node topologies** (a one-node VM with a made-up sysfs tree: the parser, the
  planner, the replica copy, and every pool path reading the right copy are tested, bit for bit). On a real two-socket box
  it is unverified: that the primary lands on the GPU's node, that `mbind` is allowed where you run it, that the replica's
  pages are found on their node, and the speed - the log's placement lines and the A/B above are how to find out.
- **The long-context selection runs on the ordinary FP32 cores.** Strata's block-score selection has a tensor-core kernel
  for Ampere and newer; on Volta the engine uses its older warp kernel (correct, slower). It only matters for long
  prompts, and whether it shows in the profile is for step 4 above to say.
- **A 4-bit KV cache (`--kv q4_0`) is read by the slow prompt attention on every card.** The tensor-core prompt
  attention - upstream's and the Volta one - does not handle it, so the older per-query FP32 kernel runs. The 16-bit
  (8K and below), 8-bit (the default above 8K) and hybrid K8V4 caches use the Volta kernel. If prompts matter more than the
  extra memory, stay on 8-bit.
- **The answers are not bit-for-bit those of an RTX card.** The FP16 GEMM route and the Volta attention kernel add up
  in a different order; `golden_compare.py` bounds the difference.
- **A V100 and an RTX 50 cannot be in one engine** (see above).
- **Only Volta, not Pascal.** A GTX 10 / P4000 / P100 is refused by the setup. A hand build for Pascal is still
  `-DSTRATA_EXPERIMENTAL_SM60=ON`, community-tested and not part of this port.
- **The Windows and Docker paths are unexercised** here (no Windows machine, no Docker daemon, no GPU).

## When it says something

- *CMake: "CUDA 13 dropped Volta"* - you pointed it at a CUDA 13 compiler. Use 12.8: `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`.
- *"...this Strata engine has no code it can run (it was built for CMAKE_CUDA_ARCHITECTURES=...)"* - the binary was built
  for other cards (the ready-made engine, or a build without 70). Run `./setup.sh`, or rebuild with
  `-DCMAKE_CUDA_ARCHITECTURES=70` and a CUDA 12 toolkit.
- *"the NVIDIA driver is too old (...; 570 or newer is needed)"* - update it, but stay on the 580 series or older: newer
  branches no longer list Volta cards, so `nvidia-smi` would not show the V100 at all.
- *"one engine cannot be built for sm_70, sm_120"* - a V100 and an RTX 50 in one engine; choose with `--gpu` / `--gpus`.
- *A crash or `unspecified launch failure` on a V100* - run `tools/volta/run_parity.sh` and attach its output with
  `strata-<model>.log` and the output of `build/strata-device` when you report it.
