#!/usr/bin/env bash
# tools/volta/profile_decode.sh - Phase 0 of docs/volta/PLAN.md on the V100 box: where does one decode step go?
#
#   1. confirm the card: nvidia-smi must say compute capability 7.0 (a Quadro P4000 is 6.1 - Pascal, no tensor cores -
#      and a baseline from it says nothing about a V100: Gate 0), with a loud banner when it does not
#   2. the NUMA picture: `numactl -H`, which node the GPU hangs off, and the advice for a dual-socket Xeon
#      (the engine pins ~35-60 GB of experts in RAM that the CPU pool streams every token; first-touch puts them all on
#      one socket and leaves half the memory bandwidth idle: run under `numactl --interleave=all`, or turn on BIOS
#      node interleaving)
#   3. one engine run with `--stats` on a fixed prompt, N tokens
#   4. the pure-GPU floor (`--gpu-only-full`: pre + post graphs + LM head, no CPU pool) so the GPU's idle share is a number
#   5. `nsys profile --trace=cuda,nvtx,osrt` of the same decode, then `nsys stats --report cuda_gpu_kern_sum,
#      cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum --format csv` (one invocation per report: the CSVs are parsed one per file)
#   6. optionally `ncu --set full --launch-count 50` (--ncu; slow: every kernel replayed ~40x)
#   7. tools/volta/summarize_profile.py turns it all into the Phase-0 table (kernel time by kernel, CPU expert time, GPU idle
#      share, expert cache hit rate, MTP tokens per pass, PCIe MB per token, tok/s)
#
# The engine is run exactly as the server runs it (the arguments of the config setup.py wrote, minus --serve), so the numbers
# are the product's.  Everything goes to --outdir; nothing is installed or changed on the machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: tools/volta/profile_decode.sh [options]

Engine (one of):
  --config FILE          the strata-*.json that setup.py wrote (exe, args, cwd, lib_dirs, env, tokenizer)
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
  --outdir DIR           where everything goes (default ./profile_out/<time>)
  --numa-interleave      run the engine under `numactl --interleave=all` (see the NUMA report)
  --nsys-prompt-tokens N the nsys run uses only the first N prompt tokens, so the trace is decode and not the prompt's prefill
                         (default 64; 0 = the same prompt as the --stats run)
  --no-floor             skip the --gpu-only-full run
  --no-nsys              skip Nsight Systems
  --ncu                  also run Nsight Compute: --set full --launch-count 50 (slow)
  --ncu-skip N           ncu --launch-skip N (default 4000: past the model load and the first tokens)
  --strict               exit with status 3 unless the GPU reports compute capability 7.0
  --dry-run              print the commands, run nothing
  -h, --help             this text

Outputs in --outdir: stats.txt (engine --stats), gpu_floor.txt, nsys_run.txt, decode.nsys-rep, cuda_gpu_*.csv, ncu_decode.ncu-rep,
numa.txt, gpu.txt, phase0.txt (the table) and phase0.json.
EOF
}

CONFIG="" EXE_ARG="" ENGINE_EXTRA="" GPU="" TOKENS_FILE="" PROMPT_NAME="" PROMPT_FILE="" TOKENIZER=""
N_TOKENS=128 OUTDIR="" NUMA_INTERLEAVE=0 NSYS_PROMPT=64 DO_FLOOR=1 DO_NSYS=1 DO_NCU=0 NCU_SKIP=4000 STRICT=0 DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --exe) EXE_ARG="$2"; shift 2;;
    --engine-args) ENGINE_EXTRA="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --tokens-file) TOKENS_FILE="$2"; shift 2;;
    --prompt-name) PROMPT_NAME="$2"; shift 2;;
    --prompt-file) PROMPT_FILE="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;
    --n-tokens) N_TOKENS="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
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
OUTDIR="${OUTDIR:-profile_out/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTDIR"

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
      echo "GPU $idx: $name, compute capability $cc - a Volta part, as Phase 0 needs."
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
# The engine's own AVX-512 expert kernels need F, BW, VL, VNNI *and VBMI* (src/kernels/cpu/expert_layout.cpp: cpu_avx512_ok):
# Ice Lake / Zen 4 and newer.  Cascade Lake has VNNI but not VBMI, so it runs the AVX-2 kernels (ggml-cpu for the i-quants) and
# the canonical Q2_0 pack is refused.  Said here so the Phase-0 numbers are read for what they are.
cpu_report() {
  local flags need=(avx2 fma f16c avx512f avx512bw avx512vl avx512_vnni avx512_vbmi) f line="" missing=()
  flags=" $(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | cut -d: -f2) "
  for f in "${need[@]}"; do
    if [[ "$flags" == *" $f "* ]]; then line+=" $f=yes"; else line+=" $f=NO"; missing+=("$f"); fi
  done
  {
    echo "== CPU vector features (what the engine's kernel selection looks at) =="
    echo "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "$line"
    if [[ " ${missing[*]-} " == *" avx512_vbmi "* || " ${missing[*]-} " == *" avx512f "* ]]; then
      echo "  -> no AVX-512 VBMI: the engine runs its AVX-2 expert kernels (the log says 'this CPU has no AVX-512: the expert kernels"
      echo "     run on AVX-2'); the canonical Q2_0 pack is refused, the i-quant (native) packs run.  STRATA_FORCE_AVX2=1 forces this on any CPU."
    else
      echo "  -> AVX-512 + VBMI present: the AVX-512 expert kernels are used."
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
  if [[ "$nodes" -gt 1 ]]; then
    local policy; policy="$(numactl --show 2>/dev/null | awk '/^policy:/ {print $2}')"
    if [[ "$NUMA_INTERLEAVE" == 1 ]]; then
      echo "  -> the engine will run under 'numactl --interleave=all' (--numa-interleave)."
    elif [[ "$policy" == "interleave" ]]; then
      echo "  -> this shell already runs with an interleave policy: good."
    else
      echo "  ADVICE: $nodes NUMA nodes (a dual-socket machine).  The engine's expert arena is written by one thread and"
      echo "  lands on one node ('first touch'), so the CPU pool reads it all through one socket's memory controllers."
      echo "  Run under 'numactl --interleave=all' (this script: --numa-interleave) or enable node interleaving in the BIOS,"
      echo "  and compare tok/s.  The alternative, 'numactl --cpunodebind=${gpu_node:-N} --membind=${gpu_node:-N}', only helps"
      echo "  when the experts fit one socket's RAM and its cores suffice."
    fi
  fi
}

# ------------------------------------------------------------------------------------------------ the engine setup
load_engine() {
  EXE="" CFG_CWD="" CFG_LIBDIRS="" CFG_TOKENIZER="" ENGINE_ARGS=()
  if [[ -n "$CONFIG" ]]; then
    eval "$(python3 - "$CONFIG" <<'PY'
import json, os, shlex, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
args = [a for a in cfg["args"] if a != "--serve"]
print("EXE=%s" % shlex.quote(cfg["exe"]))
print("CFG_CWD=%s" % shlex.quote(cfg.get("cwd") or ""))
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
  need="$(python3 - "$PROMPT_IDS" "$N_TOKENS" <<'PY'
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
  PROMPT_IDS="$OUTDIR/prompt.ids"
  local pa=(--tokenizer "$tok" --out "$PROMPT_IDS")
  if [[ -n "$PROMPT_FILE" ]]; then pa+=(--prompt-file "$PROMPT_FILE" --chat); else pa+=(--prompt-name "${PROMPT_NAME:-chat}"); fi
  run python3 "$HERE/tokenize_prompt.py" "${pa[@]}"
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
  [[ $NUMA_INTERLEAVE == 1 ]] && CMD+=(numactl --interleave=all)
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
load_engine
make_prompt
fix_max_context
count_ids() {
  python3 - "$1" <<'PY' 2>/dev/null || echo "?"
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
  run python3 "$HERE/tokenize_prompt.py" --ids-file "$PROMPT_IDS" --first "$NSYS_PROMPT" --out "$NSYS_IDS"
fi
if [[ $DO_NSYS == 1 ]]; then
  if ! command -v nsys >/dev/null 2>&1; then
    echo "profile_decode: nsys not found - skipping Nsight Systems (install Nsight Systems, or pass --no-nsys)" >&2
  else
    engine_cmd --tokens-file "$NSYS_IDS" --max-new "$N_TOKENS" --greedy
    NSYS=(nsys profile --trace=cuda,nvtx,osrt --output "$OUTDIR/decode" --force-overwrite=true)
    run_logged "$OUTDIR/nsys_run.txt" "${NSYS[@]}" "${CMD[@]}" || echo "profile_decode: the nsys run failed (continuing)" >&2
    for rep in cuda_gpu_kern_sum cuda_gpu_mem_size_sum cuda_gpu_mem_time_sum; do
      echo "+ nsys stats --report $rep --format csv > $OUTDIR/$rep.csv" >&2
      [[ $DRY == 1 ]] && continue
      nsys stats --report "$rep" --format csv --force-export=true "$OUTDIR/decode.nsys-rep" > "$OUTDIR/$rep.csv" 2> "$OUTDIR/$rep.err" \
        || echo "profile_decode: nsys stats $rep failed (see $OUTDIR/$rep.err)" >&2
    done
  fi
fi

# 6. Nsight Compute (opt-in)
if [[ $DO_NCU == 1 ]]; then
  if ! command -v ncu >/dev/null 2>&1; then
    echo "profile_decode: ncu not found - skipping Nsight Compute" >&2
  else
    engine_cmd --tokens-file "$NSYS_IDS" --max-new "$N_TOKENS" --greedy
    run_logged "$OUTDIR/ncu_run.txt" ncu --set full --launch-skip "$NCU_SKIP" --launch-count 50 --export "$OUTDIR/ncu_decode" --force-overwrite \
      "${CMD[@]}" || echo "profile_decode: the ncu run failed (continuing)" >&2
  fi
fi

# 7. the table
if [[ $DRY == 1 ]]; then
  echo "(dry run: nothing executed)"
  exit 0
fi
SA=(--stats "$OUTDIR/stats.txt" --tokens "$N_TOKENS" --gpu "$GPU_DESC" --json "$OUTDIR/phase0.json")
[[ -s "$OUTDIR/gpu_floor.txt" ]] && SA+=(--gpu-floor "$OUTDIR/gpu_floor.txt")
[[ -s "$OUTDIR/cuda_gpu_kern_sum.csv" ]] && SA+=(--nsys-kern "$OUTDIR/cuda_gpu_kern_sum.csv")
[[ -s "$OUTDIR/cuda_gpu_mem_size_sum.csv" ]] && SA+=(--nsys-memsize "$OUTDIR/cuda_gpu_mem_size_sum.csv")
[[ -s "$OUTDIR/cuda_gpu_mem_time_sum.csv" ]] && SA+=(--nsys-memtime "$OUTDIR/cuda_gpu_mem_time_sum.csv")
[[ -s "$OUTDIR/nsys_run.txt" ]] && SA+=(--nsys-run "$OUTDIR/nsys_run.txt")
python3 "$HERE/summarize_profile.py" "${SA[@]}" | tee "$OUTDIR/phase0.txt"
echo
echo "profile_decode: done.  Table: $OUTDIR/phase0.txt   raw: $OUTDIR/"
