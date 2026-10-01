#!/usr/bin/env bash
# tools/volta/run_parity.sh - Phase 1 of docs/volta/PLAN.md on the V100 box: build for sm_70 with CUDA 12.x and run every GPU
# parity program and test.  Gate 1: all pass.
#
#   1. preflight: nvcc must be CUDA 12.x (CUDA 13 cannot target sm_70); the GPU should report compute capability 7.0
#      (warned loudly if not: then the sm_70 SASS is not what runs, and "all pass" proves nothing about Volta)
#   2. configure   cmake -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_BUILD_TESTS=ON
#                        -DSTRATA_PARITY_PREFILL_GEMM=ON -DSTRATA_PARITY_PROMPT_ATTN=ON
#      (the targets that need sources this tree does not publish - bench/micro/*, the llama.cpp oracle - are not configured;
#       the script lists which expected ones are absent so a missing test is visible, not silent)
#   3. build every target (the engine included: the SASS audit scans it)
#   4. ctest: every registered test, with the arguments CMake gave it (the parity programs run `--selftest`)
#   5. the programs CMake does not register, with the arguments their own usage comments give:
#        qsa_prompt_attn_parity [context=32768] [queries=2048] [reps=5]   at the default, at a long context, and with
#                               STRATA_VOLTA_ATTN=0 (the FP32 fallback: every case must then report SKIPPED, not run the kernel)
#        gemm_volta_parity [--selftest | --bench] [--seed S]              the FP16 tensor-core GEMM route vs an FP64 reference
#   6. tools/volta/sass_audit.py over the whole build: no unexplained BPT.TRAP, the Volta kernels really contain HMMA
#
# Everything is logged under <build>/parity-logs/<time>/; a summary table ends the run; the exit status is 1 if anything FAILED
# (a SKIPPED check - no device, missing fixture - is listed loudly but is not a failure).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: tools/volta/run_parity.sh [options]

  --build DIR          build directory (default: <repo>/build-sm70)
  --cuda-root DIR      CUDA 12.x toolkit root (default: /usr/local/cuda-12.8, else the nvcc on PATH)
  --ggml-dir DIR       a llama.cpp checkout to take ggml from (default: CMake fetches the pinned commit)
  --jobs N             build parallelism (default: number of CPUs)
  --skip-build         reuse the build directory as it is (no configure, no build)
  --tests-only         build only the parity / test targets (not the engine); the SASS audit then scans those
  --long-context N     the long context of the qsa_prompt_attn_parity run (default 131072)
  --ctest-regex RE     run only the ctest tests matching RE (default: all)
  --ctest-timeout S    per-test timeout of ctest (default 1200)
  --skip-ctest         skip step 4
  --skip-extra         skip step 5
  --skip-audit         skip step 6
  --require-v100       stop unless the GPU reports compute capability 7.0
  --logs DIR           where the logs go (default <build>/parity-logs/<time>)
  -h, --help           this text

Exit status: 0 = everything passed (skips are listed), 1 = a check failed, 2 = could not start (no CUDA 12.x, configure/build failed).
EOF
}

BUILD="$ROOT/build-sm70" CUDA_ROOT="" GGML_DIR="" JOBS="$(nproc 2>/dev/null || echo 4)" SKIP_BUILD=0 TESTS_ONLY=0
LONG_CTX=131072 CTEST_RE="" CTEST_TIMEOUT=1200 SKIP_CTEST=0 SKIP_EXTRA=0 SKIP_AUDIT=0 REQUIRE_V100=0 LOGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build) BUILD="$2"; shift 2;;
    --cuda-root) CUDA_ROOT="$2"; shift 2;;
    --ggml-dir) GGML_DIR="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --skip-build) SKIP_BUILD=1; shift;;
    --tests-only) TESTS_ONLY=1; shift;;
    --long-context) LONG_CTX="$2"; shift 2;;
    --ctest-regex) CTEST_RE="$2"; shift 2;;
    --ctest-timeout) CTEST_TIMEOUT="$2"; shift 2;;
    --skip-ctest) SKIP_CTEST=1; shift;;
    --skip-extra) SKIP_EXTRA=1; shift;;
    --skip-audit) SKIP_AUDIT=1; shift;;
    --require-v100) REQUIRE_V100=1; shift;;
    --logs) LOGS="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "run_parity: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

BUILD="$(mkdir -p "$BUILD" && cd "$BUILD" && pwd)"
LOGS="${LOGS:-$BUILD/parity-logs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOGS"
TSV="$LOGS/results.tsv"
: > "$TSV"

banner() {
  local line w=0
  for line in "$@"; do (( ${#line} > w )) && w=${#line}; done
  printf '\n%s\n' "$(printf '#%.0s' $(seq 1 $((w + 4))))"
  for line in "$@"; do printf '# %-*s #\n' "$w" "$line"; done
  printf '%s\n\n' "$(printf '#%.0s' $(seq 1 $((w + 4))))"
}

# record NAME STATUS SECONDS NOTE
record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$TSV"; }

# ------------------------------------------------------------------------------------------------ 1. preflight
NVCC=""
if [[ -n "$CUDA_ROOT" ]]; then
  NVCC="$CUDA_ROOT/bin/nvcc"
elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
  NVCC=/usr/local/cuda-12.8/bin/nvcc
else
  NVCC="$(command -v nvcc || true)"
fi
if [[ "$SKIP_BUILD" == 0 ]]; then
  if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
    echo "run_parity: no nvcc found (install a CUDA 12.x toolkit - 12.8 or 12.9 - or pass --cuda-root)" >&2
    exit 2
  fi
  NVCC_V="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
  echo "nvcc: $NVCC (CUDA $NVCC_V)"
  case "${NVCC_V%%.*}" in
    12) :;;
    13|1[3-9]) echo "run_parity: CUDA $NVCC_V cannot generate code for sm_70 (CUDA 13 dropped Volta): use a 12.x toolkit, --cuda-root /usr/local/cuda-12.8" >&2; exit 2;;
    *) echo "run_parity: CUDA $NVCC_V is too old: the engine needs C++20 device code (CUDA 12.x)" >&2; exit 2;;
  esac
fi

GPU_NAME="" GPU_CC=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gi=(); first="${CUDA_VISIBLE_DEVICES:-}"; first="${first%%,*}"
  [[ "$first" =~ ^[0-9]+$ ]] && gi=(-i "$first")        # the card the tests will actually use
  IFS=',' read -r GPU_NAME GPU_CC < <(nvidia-smi ${gi[@]+"${gi[@]}"} --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 || true) || true
  GPU_NAME="${GPU_NAME# }"; GPU_CC="${GPU_CC// /}"
fi
if [[ "$GPU_CC" == "7.0" ]]; then
  echo "GPU: $GPU_NAME, compute capability 7.0 (Volta)"
else
  banner "WARNING: the GPU is \"${GPU_NAME:-unknown}\", compute capability ${GPU_CC:-unknown} - NOT a V100 (7.0)." \
         "A pass here says nothing about Volta: the sm_70 SASS is not what runs on another card (the PTX is re-compiled)." \
         "Gate 1 needs these results from the V100."
  [[ "$REQUIRE_V100" == 1 ]] && { echo "run_parity: --require-v100 and the GPU is not a V100: stopping" >&2; exit 2; }
fi
record "GPU" "$([[ "$GPU_CC" == "7.0" ]] && echo INFO || echo WARN)" 0 "${GPU_NAME:-unknown}, cc ${GPU_CC:-unknown}"

# ----------------------------------------------------------------------------------------- 2/3. configure and build
if [[ "$SKIP_BUILD" == 0 ]]; then
  CM=(cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
      -DCMAKE_CUDA_COMPILER="$NVCC" -DSTRATA_BUILD_TESTS=ON -DSTRATA_PARITY_PREFILL_GEMM=ON -DSTRATA_PARITY_PROMPT_ATTN=ON
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
  [[ -n "$GGML_DIR" ]] && CM+=(-DSTRATA_GGML_DIR="$GGML_DIR")
  echo "+ ${CM[*]}"
  t0=$SECONDS
  if ! "${CM[@]}" > "$LOGS/configure.log" 2>&1; then
    tail -25 "$LOGS/configure.log" >&2
    echo "run_parity: configure failed (see $LOGS/configure.log)" >&2
    record "cmake configure" FAIL $((SECONDS - t0)) "$LOGS/configure.log"
    exit 2
  fi
  record "cmake configure" PASS $((SECONDS - t0)) "sm_70, CUDA $NVCC_V"

  # which of the targets the tests need are absent from this tree (EXISTS guards in CMakeLists.txt)?
  ABSENT=()
  while IFS= read -r rel; do
    [[ -e "$ROOT/$rel" ]] || ABSENT+=("$rel")
  done < <(grep -o 'EXISTS "\${CMAKE_CURRENT_SOURCE_DIR}/[^"]*"' "$ROOT/CMakeLists.txt" | sed 's/.*CMAKE_CURRENT_SOURCE_DIR}\/\([^"]*\)"/\1/' | sort -u)
  for rel in "${ABSENT[@]+"${ABSENT[@]}"}"; do
    echo "note: $rel does not exist in this tree: the target that needs it is not built (bench/micro sources are not published)"
    record "absent source" SKIP 0 "$rel"
  done

  BT=(cmake --build "$BUILD" -j "$JOBS")
  if [[ "$TESTS_ONLY" == 1 ]]; then
    TARGETS="$(cmake --build "$BUILD" --target help 2>/dev/null | python3 -c '
import re, sys
names = []
for ln in sys.stdin:
    m = re.match(r"^(?:\.\.\. )?([A-Za-z0-9_+.\-]+)(?::| |$)", ln.strip().replace("... ", "... "))
    if m: names.append(m.group(1))
keep = [n for n in dict.fromkeys(names) if re.search(r"(_parity|_test|_selftest)$", n) or n in ("pinned_capture", "strata-device", "strata-load")]
print(" ".join(keep))
')"
    [[ -n "$TARGETS" ]] || { echo "run_parity: could not list the test targets" >&2; exit 2; }
    BT+=(--target $TARGETS)
  fi
  echo "+ ${BT[*]}  (log: $LOGS/build.log)"
  t0=$SECONDS
  if ! "${BT[@]}" > "$LOGS/build.log" 2>&1; then
    grep -E "error|Error" "$LOGS/build.log" | head -20 >&2 || true
    echo "run_parity: the build failed (see $LOGS/build.log)" >&2
    record "build" FAIL $((SECONDS - t0)) "$LOGS/build.log"
    exit 2
  fi
  record "build" PASS $((SECONDS - t0)) "$([[ "$TESTS_ONLY" == 1 ]] && echo 'tests only' || echo 'all targets, engine included')"
fi

# ------------------------------------------------------------------------------------------------------ 4. ctest
if [[ "$SKIP_CTEST" == 0 ]]; then
  CT=(ctest --test-dir "$BUILD" --output-on-failure --timeout "$CTEST_TIMEOUT" --output-junit "$LOGS/ctest.xml")
  [[ -n "$CTEST_RE" ]] && CT+=(-R "$CTEST_RE")
  echo "+ ${CT[*]}"
  t0=$SECONDS
  rc=0
  "${CT[@]}" > "$LOGS/ctest.log" 2>&1 || rc=$?
  tail -8 "$LOGS/ctest.log"
  if [[ -s "$LOGS/ctest.xml" ]]; then
    python3 - "$LOGS/ctest.xml" "$TSV" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
rows = []
for tc in root.iter("testcase"):
    name, secs = tc.get("name", "?"), tc.get("time", "0")
    status = tc.get("status", "")
    if tc.find("skipped") is not None or status == "notrun":
        st, note = "SKIP", "skipped / not run"
    elif tc.find("failure") is not None or tc.find("error") is not None or status == "fail":
        st, note = "FAIL", "see ctest.log"
    else:
        st, note = "PASS", ""
    rows.append((f"ctest {name}", st, f"{float(secs):.0f}", note))
with open(sys.argv[2], "a") as f:
    for r in rows:
        f.write("\t".join(r) + "\n")
PY
  else
    record "ctest" "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" $((SECONDS - t0)) "no JUnit file; see $LOGS/ctest.log"
  fi
fi

# ----------------------------------------------------------------------------- 5. programs CMake does not register
# run_check NAME LOGNAME [VAR=val ...] -- CMD ARGS...    PASS on exit 0, SKIP on exit 3 (no device / fixture), else FAIL
run_check() {
  local name="$1" logname="$2"; shift 2
  local envs=()
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  local t0=$SECONDS rc=0 log="$LOGS/$logname.log"
  echo "+ ${envs[*]-} $*"
  env ${envs[@]+"${envs[@]}"} "$@" > "$log" 2>&1 || rc=$?
  tail -4 "$log"
  CHECK_RC=$rc CHECK_LOG="$log" CHECK_SECS=$((SECONDS - t0))
  case $rc in
    0) CHECK_STATUS=PASS;;
    3) CHECK_STATUS=SKIP;;
    *) CHECK_STATUS=FAIL;;
  esac
}

if [[ "$SKIP_EXTRA" == 0 ]]; then
  PA="$BUILD/qsa_prompt_attn_parity"
  if [[ -x "$PA" ]]; then
    # the Volta tensor-core kernel against FP64 and the FP32 kernel; the program itself also runs short (1,500) and edge
    # (2,100) contexts, int8 / fp16 / K8V4 KV.  A SKIPPED case means the dispatcher refused and the kernel never ran.
    for spec in "default:32768 2048 5" "long:$LONG_CTX 512 3" "short:1500 256 3"; do
      label="${spec%%:*}"; args="${spec#*:}"
      run_check "qsa_prompt_attn_parity ($label: $args)" "qsa_prompt_attn_parity_$label" -- "$PA" $args
      note="$(grep -E '^PASSED:' "$LOGS/qsa_prompt_attn_parity_$label.log" | tail -1 || true)"
      if [[ "$CHECK_STATUS" == PASS && "$note" == *"SKIPPED (fallback): 0 "* ]]; then :;
      elif [[ "$CHECK_STATUS" == PASS ]]; then
        CHECK_STATUS=FAIL; note="$note  <- the Volta kernel did not run (the dispatcher fell back): nothing was verified"
      fi
      record "qsa_prompt_attn_parity ($label: $args)" "$CHECK_STATUS" "$CHECK_SECS" "${note:-see log}"
    done
    run_check "qsa_prompt_attn_parity STRATA_VOLTA_ATTN=0" "qsa_prompt_attn_parity_fp32" STRATA_VOLTA_ATTN=0 -- "$PA" 32768 2048 3
    note="$(grep -E '^PASSED:' "$LOGS/qsa_prompt_attn_parity_fp32.log" | tail -1 || true)"
    # with the switch off every case must fall back (SKIPPED) and none may fail; a PASSED case means the switch is ignored
    if [[ "$CHECK_STATUS" == PASS && "$note" == *"PASSED: 0 "* ]]; then :;
    elif [[ "$CHECK_STATUS" == PASS ]]; then CHECK_STATUS=FAIL; note="$note  <- STRATA_VOLTA_ATTN=0 did not disable the Volta kernel"; fi
    record "qsa_prompt_attn_parity (STRATA_VOLTA_ATTN=0)" "$CHECK_STATUS" "$CHECK_SECS" "${note:-see log}"
  else
    echo "note: $PA is not built"
    record "qsa_prompt_attn_parity" FAIL 0 "not built (-DSTRATA_PARITY_PROMPT_ATTN=ON)"
  fi

  GP="$BUILD/gemm_volta_parity"
  if [[ -x "$GP" ]]; then
    run_check "gemm_volta_parity (correctness + timing)" "gemm_volta_parity" -- "$GP"
    record "gemm_volta_parity (correctness + timing)" "$CHECK_STATUS" "$CHECK_SECS" "$(grep -iE 'TFLOPS' "$LOGS/gemm_volta_parity.log" | tail -1 | cut -c1-110 || true)"
    run_check "gemm_volta_parity --selftest --seed 7" "gemm_volta_parity_seed7" -- "$GP" --selftest --seed 7
    ok="$(grep -c 'gemm_volta_parity OK' "$LOGS/gemm_volta_parity_seed7.log" || true)"
    [[ "$CHECK_STATUS" == PASS && "$ok" == 0 ]] && CHECK_STATUS=FAIL
    record "gemm_volta_parity --selftest --seed 7" "$CHECK_STATUS" "$CHECK_SECS" "see log"
  else
    echo "note: $GP is not built"
    record "gemm_volta_parity" FAIL 0 "not built (-DSTRATA_PARITY_PREFILL_GEMM=ON and src/prefill/gemm_volta_parity.cpp)"
  fi
fi

# ----------------------------------------------------------------------------------------------- 6. the SASS audit
if [[ "$SKIP_AUDIT" == 0 ]]; then
  t0=$SECONDS
  rc=0
  python3 "$HERE/sass_audit.py" --build "$BUILD" --json "$LOGS/sass_audit.json" > "$LOGS/sass_audit.txt" 2> "$LOGS/sass_audit.err" || rc=$?
  tail -30 "$LOGS/sass_audit.txt"
  case $rc in
    0) record "sass_audit (sm_70: traps explained, HMMA present)" PASS $((SECONDS - t0)) "$LOGS/sass_audit.txt";;
    1) record "sass_audit (sm_70: traps explained, HMMA present)" FAIL $((SECONDS - t0)) "$(grep -E '^  - ' "$LOGS/sass_audit.txt" | head -2 | tr '\n' ' ')";;
    *) record "sass_audit" FAIL $((SECONDS - t0)) "could not run: $(tail -1 "$LOGS/sass_audit.err")";;
  esac
fi

# ------------------------------------------------------------------------------------------------------- summary
echo
python3 - "$TSV" "$LOGS" "$GPU_NAME" "$GPU_CC" <<'PY'
import json, sys
tsv, logs, gpu, cc = sys.argv[1:5]
rows = [l.rstrip("\n").split("\t") for l in open(tsv) if l.strip()]
rows = [(r + [""] * 4)[:4] for r in rows]
w = max(len(r[0]) for r in rows)
print(f"Phase 1 summary - {gpu or 'unknown GPU'} (cc {cc or '?'}){'' if cc == '7.0' else '   *** NOT A V100: Gate 1 NOT established ***'}")
print(f"  {'status':<6}  {'check'.ljust(w)}  {'sec':>5}  note")
for name, st, secs, note in rows:
    if st == "INFO":
        continue
    print(f"  {st:<6}  {name.ljust(w)}  {secs:>5}  {note}")
counts = {}
for r in rows:
    counts[r[1]] = counts.get(r[1], 0) + 1
fail = counts.get("FAIL", 0)
print()
print("  " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()) if k != "INFO"))
skips = [r[0] for r in rows if r[1] == "SKIP" and not r[0].startswith("absent")]
if skips:
    print(f"  SKIPPED (unverified!): {', '.join(skips[:8])}{' ...' if len(skips) > 8 else ''}")
print(f"  logs: {logs}")
print("  GATE 1: " + ("PASS" if not fail else f"FAIL ({fail} failing)") + ("" if cc == "7.0" else "  (not on a V100: does not count)"))
json.dump({"gpu": gpu, "cc": cc, "results": [dict(zip(("name", "status", "seconds", "note"), r)) for r in rows]},
          open(logs + "/summary.json", "w"), indent=1)
sys.exit(1 if fail else 0)
PY
