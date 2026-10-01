#!/usr/bin/env bash
# tools/volta/profile_decode.sh - Phase 3 of docs/volta/PLAN.md on the V100 box: where does one decode step go?
# (After Phase 1, `tools/volta/run_parity.sh`, and Phase 2, `golden_compare.py`: it runs the engine, so the engine has to be built
# - `./setup.sh` - and the model prepared.)
#
#   1. confirm the card: nvidia-smi must say compute capability 7.0 (a Quadro P4000 is 6.1 - Pascal, no tensor cores -
#      and a baseline from it says nothing about a V100: Gate 0), with a loud banner when it does not
#   2. the machine's NUMA picture, saved to the output directory: `numactl -H` (numa.txt), `nvidia-smi topo -m` (topo.txt: which
#      socket the card hangs off), and - when Intel MLC is installed - `mlc --bandwidth_matrix` (mlc_bandwidth_matrix.txt: local vs
#      remote bandwidth per node pair), plus the advice for a dual-socket Xeon
#      (the engine keeps 23-50 GB of experts in RAM that the CPU pool streams every token; since WP-F it MIRRORS them,
#      one copy per socket (--numa auto, the default) - keep BIOS node interleaving OFF so both nodes are visible;
#      --numa-interleave runs the old quick fix instead, `numactl --interleave=all` with the mirror off, for the A/B)
#   3. one engine run with `--stats` on a fixed prompt, N tokens
#   4. the pure-GPU floor (`--gpu-only-full`: pre + post graphs + LM head, no CPU pool) so the GPU's idle share is a number
#   5. `nsys profile --trace=cuda,nvtx,osrt` of the same decode (the CUDA 12.x toolkit's own nsys when there is one: Nsight of a
#      CUDA 13 toolkit no longer profiles Volta), then its SQLite export (decode.sqlite) and `nsys stats --report cuda_gpu_kern_sum,
#      cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum --format csv` (one invocation per report: the CSVs are parsed one per file)
#   6. optionally `ncu --set full --launch-count 50` (--ncu; slow: every kernel replayed ~40x)
#   7. tools/volta/summarize_profile.py turns it all into the profile table (kernel time by kernel, CPU expert time, GPU idle
#      share, expert cache hit rate, MTP tokens per pass, PCIe MB per token, tok/s).  The trace covers the whole process, start-up
#      included (the dense weights' upload, the expert-cache fill: tens of GB of PCIe traffic that is not decoding), so every
#      per-token and "decode window" figure is cut from the SQLite export to the engine's own decode time at the end of the run;
#      a figure over the whole trace is labelled "whole run".
#
# The engine is run exactly as the server runs it (the arguments of the config setup.py wrote, minus --serve), so the numbers
# are the product's.  Everything goes to --outdir (made absolute: the engine and nsys run inside the config's `cwd`, the
# repository root, so a relative path would name two different places); nothing is installed or changed on the machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: tools/volta/profile_decode.sh [options]

Engine (one of):
  --config FILE          the strata-*.json that setup.py wrote, in the repository root (exe, args, cwd, lib_dirs, env, tokenizer)
  --exe PATH             the engine binary; its arguments with --engine-args "..."
  --engine-args "..."    extra / all engine arguments, one shell-quoted string (added after the config's)
  --gpu N                CUDA_VISIBLE_DEVICES for every run (numbered as nvidia-smi does)

Prompt (default: the built-in chat prompt, tokenized with the pack's tokenizer):
  --tokens-file FILE     token ids (commas or whitespace)
  --prompt-name NAME     chat | code (tools/volta/prompts/)
  --prompt-file FILE     a text file to tokenize
  --tokenizer DIR        the pack's tokenizer/ directory (default: the config's, else pack/full/tokenizer)

Run:
  --n-tokens N           tokens to generate (default 128)
  --outdir DIR           where everything goes (default ./profile_out/<time>; relative paths are taken from where you run this)
  --numa-interleave      run the engine under `numactl --interleave=all` (see the NUMA report)
  --nsys-prompt-tokens N the nsys run uses only the first N prompt tokens, so the trace is decode and not the prompt's prefill
                         (default 64; 0 = the same prompt as the --stats run)
  --no-floor             skip the --gpu-only-full run
  --mlc PATH             Intel Memory Latency Checker (default: `mlc` on PATH, else skipped with a note); its --bandwidth_matrix needs
                         root and the msr module (`sudo modprobe msr`) to switch the prefetchers off - run as root or accept a warning
  --no-mlc               skip it even when it is installed
  --cuda-root DIR        take nsys / ncu from this CUDA toolkit (default: /usr/local/cuda-12.8, the other 12.x, then PATH)
  --no-nsys              skip Nsight Systems
  --ncu                  also run Nsight Compute: --set full --launch-count 50 (slow)
  --ncu-skip N           ncu --launch-skip N (default 4000: past the model load and the first tokens)
  --strict               exit with status 3 unless the GPU reports compute capability 7.0
  --dry-run              print the commands, run nothing
  -h, --help             this text

Python: the repository's .venv/bin/python (what setup.sh installs: numpy, regex, ...) when it exists, else python3; tokenizing a prompt
needs numpy and regex (`pip install numpy regex`), the rest of this script needs only the standard library.

Outputs in --outdir: stats.txt (engine --stats), gpu_floor.txt, nsys_run.txt, decode.nsys-rep, decode.sqlite, cuda_gpu_*.csv,
ncu_decode.ncu-rep, numa.txt (lscpu, numactl -H, CPU features), topo.txt (nvidia-smi topo -m), mlc_bandwidth_matrix.txt, gpu.txt,
profile.txt (the table) and profile.json.
EOF
}

CONFIG="" EXE_ARG="" ENGINE_EXTRA="" GPU="" TOKENS_FILE="" PROMPT_NAME="" PROMPT_FILE="" TOKENIZER=""
N_TOKENS=128 OUTDIR="" NUMA_INTERLEAVE=0 NSYS_PROMPT=64 DO_FLOOR=1 DO_NSYS=1 DO_NCU=0 NCU_SKIP=4000 STRICT=0 DRY=0
MLC_ARG="" DO_MLC=1 CUDA_ROOT=""

# abspath PATH : PATH made absolute against the directory this script was started in.  The engine and nsys run inside the config's
# `cwd` (the repository root), so every path handed to them must not depend on where we stand.
START_DIR="$PWD"
abspath() {
  case "$1" in
    /*) printf '%s\n' "$1";;
    *) printf '%s/%s\n' "$START_DIR" "${1#./}";;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$(abspath "$2")"; shift 2;;
    --exe) EXE_ARG="$(abspath "$2")"; shift 2;;
    --engine-args) ENGINE_EXTRA="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --tokens-file) TOKENS_FILE="$(abspath "$2")"; shift 2;;
    --prompt-name) PROMPT_NAME="$2"; shift 2;;
    --prompt-file) PROMPT_FILE="$(abspath "$2")"; shift 2;;
    --tokenizer) TOKENIZER="$(abspath "$2")"; shift 2;;
    --n-tokens) N_TOKENS="$2"; shift 2;;
    --outdir) OUTDIR="$(abspath "$2")"; shift 2;;
    --mlc) MLC_ARG="$2"; shift 2;;
    --no-mlc) DO_MLC=0; shift;;
    --cuda-root) CUDA_ROOT="$(abspath "$2")"; shift 2;;
    --numa-interleave) NUMA_INTERLEAVE=1; shift;;
    --nsys-prompt-tokens) NSYS_PROMPT="$2"; shift 2;;
    --no-floor) DO_FLOOR=0; shift;;
    --no-nsys) DO_NSYS=0; shift;;
    --ncu) DO_NCU=1; shift;;
    --ncu-skip) NCU_SKIP="$2"; shift 2;;
    --strict) STRICT=1; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "profile_decode: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

if [[ -z "$CONFIG" && -z "$EXE_ARG" ]]; then
  echo "profile_decode: give --config strata-*.json, or --exe PATH [--engine-args ...]" >&2
  exit 2
fi
OUTDIR="${OUTDIR:-$START_DIR/profile_out/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

# the Python that runs the helper scripts: the repository's .venv (what setup.sh installs: numpy, regex, ...), else python3
if [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3 || true)"; fi
if [[ -z "$PY" ]]; then echo "profile_decode: no python3 found (and no $ROOT/.venv/bin/python)" >&2; exit 2; fi

# need_py_modules WHY MODULE... : stop with the remedy when $PY cannot import them (a --dry-run only warns)
need_py_modules() {
  local why="$1"; shift
  local mods; mods="$(IFS=,; echo "$*")"
  if ! "$PY" -c "import $mods" 2>/dev/null; then
    echo "profile_decode: $why needs the Python packages: $* (python used: $PY)" >&2
    echo "  fix: ./setup.sh (it creates $ROOT/.venv with them), or  $PY -m pip install $*" >&2
    echo "  (or skip tokenizing: --tokens-file FILE with token ids)" >&2
    [[ $DRY == 1 ]] || exit 2
    echo "  (dry run: continuing)" >&2
  fi
}

# find_cuda_tool NAME : the CUDA 12.x toolkit's own copy of an Nsight tool (the Nsight of a CUDA 13 toolkit no longer profiles
# Volta), else the one on PATH.  Prints nothing when there is none.
find_cuda_tool() {
  local n="$1" d
  for d in ${CUDA_ROOT:+"$CUDA_ROOT"} /usr/local/cuda-12.8 /usr/local/cuda-12.9 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda; do
    if [[ -x "$d/bin/$n" ]]; then printf '%s\n' "$d/bin/$n"; return 0; fi
  done
  command -v "$n" || true
}

banner() {   # banner LINE...  - a box that cannot be missed in a terminal
  local line w=0
  for line in "$@"; do (( ${#line} > w )) && w=${#line}; done
  printf '\n%s\n' "$(printf '#%.0s' $(seq 1 $((w + 4))))"
  for line in "$@"; do printf '# %-*s #\n' "$w" "$line"; done
  printf '%s\n\n' "$(printf '#%.0s' $(seq 1 $((w + 4))))"
}

# ---------------------------------------------------------------------------------------------------- 1. the card
check_gpu() {
  {
    echo "== nvidia-smi --query-gpu=name,compute_cap --format=csv =="
  } > "$OUTDIR/gpu.txt"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi: not found - cannot confirm the GPU (is the driver installed on this machine?)" | tee -a "$OUTDIR/gpu.txt"
    [[ $STRICT == 1 ]] && exit 3
    GPU_DESC="unknown (no nvidia-smi)"; return 0
  fi
  local q rc=0
  q="$(nvidia-smi --query-gpu=index,name,compute_cap,memory.total,pci.bus_id,driver_version --format=csv,noheader 2>&1)" || rc=$?
  if [[ $rc -ne 0 || "$q" == *"not a valid field"* ]]; then
    # a driver older than ~510 has no compute_cap field: fall back to the name
    q="$(nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id,driver_version --format=csv,noheader 2>&1)" || true
    echo "$q" | tee -a "$OUTDIR/gpu.txt"
    banner "this driver cannot report compute_cap: check by hand that the card is a V100 (7.0)"
    GPU_DESC="$(echo "$q" | head -1)"; return 0
  fi
  nvidia-smi --query-gpu=name,compute_cap --format=csv | tee -a "$OUTDIR/gpu.txt"
  echo "$q" >> "$OUTDIR/gpu.txt"
  local sel="${GPU:-0}" idx name cc mem bus drv found=0
  while IFS=',' read -r idx name cc mem bus drv; do
    idx="${idx// /}"; cc="${cc// /}"; name="${name# }"; bus="${bus// /}"
    [[ "$idx" == "$sel" ]] || continue
    found=1
    GPU_DESC="$name, cc $cc"; GPU_BUS="$bus"
    if [[ "$cc" == "7.0" ]]; then
      echo "GPU $idx: $name, compute capability $cc - a Volta part, as the profile needs."
    else
      banner "WARNING: GPU $idx is \"$name\", compute capability $cc - NOT a V100 (7.0)." \
             "Gate 0: the numbers from this run are NOT V100 numbers and must not be used as the Volta baseline." \
             "(A Quadro P4000 is 6.1: Pascal, no tensor cores, a different memory system.)" \
             "Run this on the V100 box, or pick the card with --gpu N."
      [[ $STRICT == 1 ]] && { echo "profile_decode: --strict and the GPU is not a V100: stopping" >&2; exit 3; }
    fi
  done <<< "$q"
  if [[ $found == 0 ]]; then
    echo "profile_decode: no GPU with index $sel in nvidia-smi's list" >&2
    [[ $STRICT == 1 ]] && exit 3
    GPU_DESC="GPU $sel not found"
  fi
}

# ------------------------------------------------------------------------------------------- 2a. the CPU's vector units
# The engine's own AVX-512 expert kernels need F, BW, VL, DQ and VNNI (src/kernels/cpu/expert_layout.cpp: cpu_avx512_ok);
# VBMI is optional.  Ice Lake / Zen 4 and newer have it and run the VBMI build; Cascade Lake has VNNI but not VBMI and runs
# the no-VBMI build of the same kernels (bit-identical results).  Without VNNI (Skylake-X, Zen 2/3) the engine runs its AVX-2
# kernels (ggml-cpu for the i-quants) and the canonical Q2_0 pack is refused.  Said here so the profile's numbers are read for
# what they are.
cpu_report() {
  local flags need=(avx2 fma f16c avx512f avx512bw avx512vl avx512dq avx512_vnni avx512_vbmi) f line="" missing=()
  flags=" $(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | cut -d: -f2) "
  for f in "${need[@]}"; do
    if [[ "$flags" == *" $f "* ]]; then line+=" $f=yes"; else line+=" $f=NO"; missing+=("$f"); fi
  done
  {
    echo "== CPU vector features (what the engine's kernel selection looks at) =="
    echo "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "$line"
    local need512=" avx512f avx512bw avx512vl avx512dq avx512_vnni " m512=0
    for f in ${missing[@]+"${missing[@]}"}; do [[ "$need512" == *" $f "* ]] && m512=1; done
    if [[ $m512 == 1 ]]; then
      echo "  -> no AVX-512 with VNNI: the engine runs its AVX-2 expert kernels (the log says 'this CPU has no AVX-512: the expert kernels"
      echo "     run on AVX-2'); the canonical Q2_0 pack is refused, the i-quant (native) packs run.  STRATA_FORCE_AVX2=1 forces this on any CPU."
    elif [[ " ${missing[*]-} " == *" avx512_vbmi "* ]]; then
      echo "  -> AVX-512 + VNNI, no VBMI (Cascade Lake): the AVX-512 expert kernels run in their no-VBMI build (the log says"
      echo "     'CPU expert kernels: AVX-512 VNNI (no VBMI)').  STRATA_FORCE_AVX2=1 forces the AVX-2 kernels."
    else
      echo "  -> AVX-512 + VNNI + VBMI present: the AVX-512 expert kernels run in their VBMI build.  STRATA_FORCE_AVX512_NOVBMI=1 forces"
      echo "     the Cascade Lake build for an A/B test; STRATA_FORCE_AVX2=1 forces the AVX-2 kernels."
    fi
  } | tee -a "$OUTDIR/numa.txt"
}

# --------------------------------------------------------------------------------------------------- 2. NUMA
numa_report() {
  {
    echo "== lscpu =="; lscpu 2>/dev/null | grep -E "^(Model name|Socket|Thread|Core|NUMA|CPU\(s\))" || true
    if command -v numactl >/dev/null 2>&1; then
      echo; echo "== numactl -H =="; numactl -H || true
      echo; echo "== numactl --show (this shell's policy) =="; numactl --show || true
    else
      echo; echo "numactl is not installed (apt install numactl): the NUMA layout below comes from sysfs only"
    fi
  } > "$OUTDIR/numa.txt" 2>&1
  cat "$OUTDIR/numa.txt"
  local nodes=1
  if [[ -d /sys/devices/system/node ]]; then
    nodes="$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)"
  fi
  local gpu_node=""
  if [[ -n "${GPU_BUS:-}" ]]; then
    local sys="${GPU_BUS,,}"; sys="${sys#0000}"
    [[ -r "/sys/bus/pci/devices/$sys/numa_node" ]] && gpu_node="$(cat "/sys/bus/pci/devices/$sys/numa_node")"
  fi
  echo
  echo "NUMA nodes: $nodes${gpu_node:+; the GPU ($GPU_BUS) is attached to node $gpu_node}"
  local sockets; sockets="$(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/ {gsub(/ /, "", $2); print $2}')"
  if [[ "${sockets:-1}" -gt 1 && "$nodes" -le 1 ]]; then
    echo "  WARNING: $sockets CPU sockets but ONE NUMA node: BIOS 'Node Interleaving' looks ENABLED.  Linux then cannot tell the sockets'"
    echo "  memory apart, so the engine's NUMA mirror (and the DeepSeek port's per-socket split) is silently off.  Set Node Interleaving to"
    echo "  Disabled (and Sub-NUMA Clustering to Disabled) in the BIOS: docs/volta/VOLTA.md, 'BIOS'."
  fi
  if [[ "$nodes" -gt 1 ]]; then
    local policy; policy="$(numactl --show 2>/dev/null | awk '/^policy:/ {print $2}')"
    if [[ "$NUMA_INTERLEAVE" == 1 ]]; then
      echo "  -> this run uses the old quick fix: 'numactl --interleave=all' with the engine's mirror OFF (--numa-interleave)."
    elif [[ "$policy" == "interleave" ]]; then
      echo "  WARNING: this shell already runs with an interleave policy; the engine's NUMA mirror (the default) binds its"
      echo "  copies explicitly, but for a clean measurement run without numactl (mirror) or with --numa-interleave (A/B)."
    else
      echo "  $nodes NUMA nodes (a dual-socket machine).  The engine mirrors its expert arena on every node by default"
      echo "  (--numa auto: the startup log's 'NUMA:' lines say what it did).  A/B it: this script as is (mirror), with"
      echo "  STRATA_NUMA_MIRROR=0 (one copy, first touch), and with --numa-interleave (numactl --interleave=all, mirror off)."
      echo "  Keep BIOS 'Node Interleaving' disabled: with it on the OS sees one node and nothing can be placed."
    fi
  fi
}

# nvidia-smi topo -m: which socket (CPU affinity / NUMA affinity) the card hangs off, saved beside the numactl output
topo_report() {
  {
    echo "== nvidia-smi topo -m =="
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi topo -m || true; else echo "nvidia-smi not found"; fi
  } > "$OUTDIR/topo.txt" 2>&1
  cat "$OUTDIR/topo.txt"
}

# Intel MLC, optional: local vs remote bandwidth per node pair (the UPI link's share of the box's 90-120 GB/s).  Skipped with a note
# when it is not installed; nothing else runs while it measures (it uses every core for a minute or two).
mlc_report() {
  [[ $DO_MLC == 1 ]] || return 0
  local mlc="${MLC_ARG:-$(command -v mlc || true)}"
  if [[ -z "$mlc" ]]; then
    echo "mlc (Intel Memory Latency Checker) not found: skipping mlc --bandwidth_matrix.  Optional: download it from Intel's Memory Latency" \
         "Checker page, unpack it, and pass --mlc PATH (the tool shows local vs remote bandwidth per node pair)." | tee "$OUTDIR/mlc_bandwidth_matrix.txt"
    return 0
  fi
  echo "+ $mlc --bandwidth_matrix > $OUTDIR/mlc_bandwidth_matrix.txt" >&2
  [[ $DRY == 1 ]] && return 0
  "$mlc" --bandwidth_matrix 2>&1 | tee "$OUTDIR/mlc_bandwidth_matrix.txt" \
    || echo "profile_decode: mlc --bandwidth_matrix failed (continuing; it needs root and the msr module for the prefetcher switch, see its output)" >&2
}

# ------------------------------------------------------------------------------------------------ the engine setup
load_engine() {
  EXE="" CFG_CWD="" CFG_LIBDIRS="" CFG_TOKENIZER="" ENGINE_ARGS=()
  if [[ -n "$CONFIG" ]]; then
    eval "$("$PY" - "$CONFIG" <<'PY'
import json, os, shlex, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
args = [a for a in cfg["args"] if a != "--serve"]
cwd = cfg.get("cwd") or ""
exe = cfg["exe"] if os.path.isabs(cfg["exe"]) else os.path.join(cwd or os.getcwd(), cfg["exe"])    # the engine runs in `cwd`
print("EXE=%s" % shlex.quote(exe))
print("CFG_CWD=%s" % shlex.quote(cwd))
print("ENGINE_ARGS=(%s)" % " ".join(shlex.quote(a) for a in args))
print("CFG_LIBDIRS=%s" % shlex.quote(":".join(d for d in (cfg.get("lib_dirs") or []) if os.path.isdir(d))))
print("CFG_TOKENIZER=%s" % shlex.quote(cfg.get("tokenizer") or ""))
for k, v in (cfg.get("env") or {}).items():
    print("export %s=%s" % (k, shlex.quote(str(v))))
PY
)"
  fi
  [[ -n "$EXE_ARG" ]] && EXE="$EXE_ARG"
  if [[ -n "$ENGINE_EXTRA" ]]; then
    local extra; eval "extra=($ENGINE_EXTRA)"
    ENGINE_ARGS+=("${extra[@]}")
  fi
  [[ -x "$EXE" || "$DRY" == 1 ]] || { echo "profile_decode: the engine $EXE is not an executable file" >&2; exit 2; }
  [[ -n "$CFG_LIBDIRS" ]] && export LD_LIBRARY_PATH="$CFG_LIBDIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  if [[ -n "$GPU" ]]; then export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"; fi
  CWD="${CFG_CWD:-$PWD}"
}

# fix_max_context : the engine refuses a prompt that does not fit its --max-context (prompt + generated tokens + slack);
# a config written for chat sizes may be smaller than a long --tokens-file, so raise it to what this run needs
fix_max_context() {
  local need have i out=()
  [[ -r "$PROMPT_IDS" ]] || return 0      # --dry-run: the prompt was not built
  need="$("$PY" - "$PROMPT_IDS" "$N_TOKENS" <<'PY'
import sys
n = len(open(sys.argv[1]).read().replace(",", " ").split()) + int(sys.argv[2]) + 64
print((n + 1023) // 1024 * 1024)
PY
)"
  have=0
  for ((i = 0; i < ${#ENGINE_ARGS[@]}; i++)); do
    if [[ "${ENGINE_ARGS[$i]}" == "--max-context" ]]; then have="${ENGINE_ARGS[$((i + 1))]:-0}"; i=$((i + 1)); else out+=("${ENGINE_ARGS[$i]}"); fi
  done
  ENGINE_ARGS=(${out[@]+"${out[@]}"} --max-context "$(( have > need ? have : need ))")
}

make_prompt() {
  if [[ -n "$TOKENS_FILE" ]]; then
    PROMPT_IDS="$TOKENS_FILE"
    return
  fi
  local tok="${TOKENIZER:-$CFG_TOKENIZER}"
  [[ -z "$tok" ]] && tok="$CWD/pack/full/tokenizer"
  case "$tok" in /*) ;; *) tok="$CWD/$tok";; esac     # a relative tokenizer path in a config means "relative to its cwd"
  PROMPT_IDS="$OUTDIR/prompt.ids"
  local pa=(--tokenizer "$tok" --out "$PROMPT_IDS")
  if [[ -n "$PROMPT_FILE" ]]; then pa+=(--prompt-file "$PROMPT_FILE" --chat); else pa+=(--prompt-name "${PROMPT_NAME:-chat}"); fi
  need_py_modules "tokenizing the prompt" numpy regex
  run "$PY" "$HERE/tokenize_prompt.py" "${pa[@]}"
}

# run CMD... : echo it, and run it unless --dry-run
run() {
  echo "+ $*" >&2
  [[ $DRY == 1 ]] && return 0
  "$@"
}

# engine_cmd EXTRA... : the command array for one engine run (numactl wrapper, config args, then EXTRA)
engine_cmd() {
  CMD=()
  # the interleave A/B: the mirror off (STRATA_NUMA_MIRROR=0), so the two policies are not stacked
  [[ $NUMA_INTERLEAVE == 1 ]] && CMD+=(env STRATA_NUMA_MIRROR=0 numactl --interleave=all)
  CMD+=("$EXE" ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"} "$@")
}

run_logged() {   # run_logged LOGFILE CMD... : stdout+stderr to the file and the terminal
  local log="$1"; shift
  echo "+ $* > $log" >&2
  [[ $DRY == 1 ]] && return 0
  local rc=0
  ( cd "$CWD" && "$@" ) 2>&1 | tee "$log" || rc=$?
  return "$rc"
}

# ------------------------------------------------------------------------------------------------------------ main
GPU_DESC="" GPU_BUS=""
echo "profile_decode: output in $OUTDIR"
check_gpu
numa_report
cpu_report
topo_report
mlc_report
load_engine
make_prompt
fix_max_context
count_ids() {
  "$PY" - "$1" <<'PY' 2>/dev/null || echo "?"
import sys
print(len(open(sys.argv[1]).read().replace(",", " ").split()))
PY
}
echo "prompt: $PROMPT_IDS ($([[ -r "$PROMPT_IDS" ]] && count_ids "$PROMPT_IDS" || echo '?') tokens), generating $N_TOKENS"

# 3. the --stats run (the product's own configuration: speculative decoding, expert cache, ...)
engine_cmd --tokens-file "$PROMPT_IDS" --max-new "$N_TOKENS" --greedy --stats
run_logged "$OUTDIR/stats.txt" "${CMD[@]}" || { echo "profile_decode: the --stats run failed (see $OUTDIR/stats.txt)" >&2; exit 1; }

# 4. the GPU floor
if [[ $DO_FLOOR == 1 ]]; then
  engine_cmd --tokens-file "$PROMPT_IDS" --max-new 1 --gpu-only-full
  run_logged "$OUTDIR/gpu_floor.txt" "${CMD[@]}" || echo "profile_decode: the --gpu-only-full run failed (continuing; see $OUTDIR/gpu_floor.txt)" >&2
fi

# 5. Nsight Systems
NSYS_IDS="$PROMPT_IDS"
if [[ $DO_NSYS == 1 || $DO_NCU == 1 ]] && [[ "$NSYS_PROMPT" -gt 0 ]]; then
  NSYS_IDS="$OUTDIR/prompt_short.ids"
  need_py_modules "shortening the prompt (tokenize_prompt.py imports golden_compare)" numpy regex
  run "$PY" "$HERE/tokenize_prompt.py" --ids-file "$PROMPT_IDS" --first "$NSYS_PROMPT" --out "$NSYS_IDS"
fi
if [[ $DO_NSYS == 1 ]]; then
  NSYS_BIN="$(find_cuda_tool nsys)"
  if [[ -z "$NSYS_BIN" ]]; then
    echo "profile_decode: nsys not found - skipping Nsight Systems (it ships with the CUDA toolkit: /usr/local/cuda-12.8/bin/nsys; or pass --cuda-root, or --no-nsys)" >&2
  else
    echo "profile_decode: Nsight Systems: $NSYS_BIN" >&2
    engine_cmd --tokens-file "$NSYS_IDS" --max-new "$N_TOKENS" --greedy
    NSYS=("$NSYS_BIN" profile --trace=cuda,nvtx,osrt --output "$OUTDIR/decode" --force-overwrite=true)
    run_logged "$OUTDIR/nsys_run.txt" "${NSYS[@]}" "${CMD[@]}" || echo "profile_decode: the nsys run failed (continuing)" >&2
    # the first `nsys stats` (re)writes decode.sqlite from the .nsys-rep; the others reuse it
    force_export=true
    for rep in cuda_gpu_kern_sum cuda_gpu_mem_size_sum cuda_gpu_mem_time_sum; do
      echo "+ $NSYS_BIN stats --report $rep --format csv > $OUTDIR/$rep.csv" >&2
      [[ $DRY == 1 ]] && continue
      "$NSYS_BIN" stats --report "$rep" --format csv --force-export="$force_export" "$OUTDIR/decode.nsys-rep" > "$OUTDIR/$rep.csv" 2> "$OUTDIR/$rep.err" \
        || echo "profile_decode: nsys stats $rep failed (see $OUTDIR/$rep.err)" >&2
      force_export=false
    done
    # the decode window and the per-token figures are cut from the trace's SQLite export (the CSV reports cannot be cut by time)
    if [[ $DRY == 1 ]]; then
      echo "+ $NSYS_BIN export --type sqlite --output $OUTDIR/decode.sqlite $OUTDIR/decode.nsys-rep   (unless nsys stats made it)" >&2
    elif [[ ! -s "$OUTDIR/decode.sqlite" && -s "$OUTDIR/decode.nsys-rep" ]]; then
      "$NSYS_BIN" export --type sqlite --force-overwrite=true --output "$OUTDIR/decode.sqlite" "$OUTDIR/decode.nsys-rep" > "$OUTDIR/nsys_export.log" 2>&1 \
        || echo "profile_decode: nsys export failed (see $OUTDIR/nsys_export.log): the table will only have whole-run nsys figures" >&2
    fi
  fi
fi

# 6. Nsight Compute (opt-in)
if [[ $DO_NCU == 1 ]]; then
  NCU_BIN="$(find_cuda_tool ncu)"
  if [[ -z "$NCU_BIN" ]]; then
    echo "profile_decode: ncu not found - skipping Nsight Compute" >&2
  else
    engine_cmd --tokens-file "$NSYS_IDS" --max-new "$N_TOKENS" --greedy
    run_logged "$OUTDIR/ncu_run.txt" "$NCU_BIN" --set full --launch-skip "$NCU_SKIP" --launch-count 50 --export "$OUTDIR/ncu_decode" --force-overwrite \
      "${CMD[@]}" || echo "profile_decode: the ncu run failed (continuing)" >&2
  fi
fi

# 7. the table
if [[ $DRY == 1 ]]; then
  echo "(dry run: nothing executed)"
  exit 0
fi
SA=(--stats "$OUTDIR/stats.txt" --tokens "$N_TOKENS" --gpu "$GPU_DESC" --json "$OUTDIR/profile.json")
[[ -s "$OUTDIR/gpu_floor.txt" ]] && SA+=(--gpu-floor "$OUTDIR/gpu_floor.txt")
[[ -s "$OUTDIR/decode.sqlite" ]] && SA+=(--nsys-sqlite "$OUTDIR/decode.sqlite")
[[ -s "$OUTDIR/cuda_gpu_kern_sum.csv" ]] && SA+=(--nsys-kern "$OUTDIR/cuda_gpu_kern_sum.csv")
[[ -s "$OUTDIR/cuda_gpu_mem_size_sum.csv" ]] && SA+=(--nsys-memsize "$OUTDIR/cuda_gpu_mem_size_sum.csv")
[[ -s "$OUTDIR/cuda_gpu_mem_time_sum.csv" ]] && SA+=(--nsys-memtime "$OUTDIR/cuda_gpu_mem_time_sum.csv")
[[ -s "$OUTDIR/nsys_run.txt" ]] && SA+=(--nsys-run "$OUTDIR/nsys_run.txt")
"$PY" "$HERE/summarize_profile.py" "${SA[@]}" | tee "$OUTDIR/profile.txt"
echo
echo "profile_decode: done.  Table: $OUTDIR/profile.txt   raw: $OUTDIR/   (send back: profile.json, profile.txt, numa.txt, topo.txt, mlc_bandwidth_matrix.txt, stats.txt)"
