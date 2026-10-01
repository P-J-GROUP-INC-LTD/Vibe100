#!/usr/bin/env bash
# tools/volta/logit_identity.sh - "does the V100 port give the model's answers?": the logit-identity / quality matrix of docs/volta/LOGIT_IDENTITY.md,
# one script, one row per question, each row PASS / FAIL / SKIP (or INFO: a measurement with no pass mark) with its reason.
#
#   TIER 1 - bit-exact.  Same card, same pack, same prompt, GPU expert set pinned (--pcie-frac 0 --adapt-swaps 0, the cache size of the first run),
#            STRATA_IQ_MT_MIN=1 (expert rows round the same whatever the window held).  golden_compare.py --exact: every logit compared bit for bit.
#       1c  the same command twice                                     identical  (the control: if this fails nothing below can be read as exact)
#       1a  the port with its fast paths off (STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0, CPU tier AVX2, --numa off)
#           against UPSTREAM Strata 3906943 built for sm_70 here (git worktree + CUDA 12.8; STRATA_PROMPT_ATTN_OLD=1)    identical
#       1b  NUMA mirror on (--numa mirror) against off                 identical   (same bytes, same kernels)
#       1d  CPU expert tier AVX-512 (STRATA_IQ512=1) against AVX2      INFO: identical, or how far apart (then it belongs to tier 2)
#   TIER 2 - quality against llama.cpp on the SAME GGUF.  Bit-exact against llama.cpp is impossible (different kernels sum in different orders; its
#            own CPU and CUDA builds differ), so: llama.cpp's CPU build against its CUDA build is the NOISE FLOOR, and the engine must be no further
#            from llama.cpp than that.  llama-perplexity --kl-divergence-base (CUDA build) is the reference; golden_compare.py --ref-kld scores the engine.
#       2a/2b  build llama.cpp (third_party/llama.cpp, the pinned commit): CUDA for sm_70, and CPU only
#       2c  llama-perplexity CUDA writes the reference (the text's chunks and its next-token distributions)
#       2d  llama-perplexity CPU against it = the noise floor                        INFO (FAIL when the two disagree beyond 0.05 nats: broken reference)
#       2e  Strata, port defaults (all fast paths on)         PASS when mean KL and top-1 mismatch <= max(2 x floor, absolute floor) and the paired
#       2f  Strata, fast paths off (STRATA_VOLTA_ATTN=0 ...)  ln-PPL change is within 2 standard errors of zero (or of the floor's own).  PROVISIONAL:
#                                                              the first real run on the box confirms the rule.
#       2g  (--kv-fp16-row) Strata with a 16-bit KV cache: separates the engine's int8 KV quantisation from its kernels
#
# Everything is written to --out (default build-logit-identity/results/<time>): results.tsv, summary.txt, one folder per row with the tool's own report.json /
# report.txt and the engine logs.  `--dry-run` prints every command and runs nothing (it works on a machine with no GPU, no model and no llama.cpp).
# Needs: ./setup.sh done (.venv, third_party/llama.cpp, the GGUF in Strata-data/models/, strata-<model>.json), CUDA 12.8 at /usr/local/cuda-12.8, git.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage: tools/volta/logit_identity.sh [options]

  --config FILE        engine config setup.py wrote (default: strata-iq3_xxs.json in the repository root)
  --gguf FILE          the GGUF llama.cpp loads (default: the config's --native: setup.py keeps the original GGUF shards in Strata-data/models/)
  --out DIR            results directory (default: <work>/results/<time>)
  --work DIR           where builds are cached (default: <repo>/build-logit-identity; gitignored)
  --text FILE          the text for tier 2 (default: tools/volta/prompts/kld_text.txt: public-domain prose, code and technical prose, ~29k tokens)
  --ctx N              llama-perplexity context, tokens per chunk (default 4096; 8192 also reaches the QSA sparse selection in the batched path)
  --chunks N           chunks to use (default 4; the shipped text holds 7 of 4096 tokens)
  --only LIST          run only these rows, e.g. 1c,1a or 2e,2f (default: all; --tier 1 / --tier 2 pick a tier)
  --tier N             1 or 2: only that tier
  --llama-dir DIR      llama.cpp checkout at the pinned commit (default: <repo>/third_party/llama.cpp)
  --cuda-root DIR      CUDA 12.x toolkit (default /usr/local/cuda-12.8)
  --cmake PATH         cmake 3.24+ (default: <repo>/.venv/bin/cmake, else cmake)
  --python PATH        python with numpy (default: <repo>/.venv/bin/python, else python3)
  --jobs N             build parallelism (default: nproc)
  --gpu N              CUDA device, numbered as nvidia-smi does (default: the first)
  --upstream-ref REF   the upstream snapshot for row 1a (default 3906943 = github.com/Niko1221/Strata 9259cad, this repository's first commit)
  --upstream-src DIR   an existing checkout of upstream Strata instead of a git worktree of --upstream-ref
  --upstream-hook M    patch (default): apply tools/volta/upstream_logits_dump.patch to the upstream checkout - it ONLY adds the STRATA_LOGITS_DUMP
                       measurement hook (full logits per scored position, off unless the variable is set), so row 1a can compare bits;
                       none: build upstream untouched (row 1a then compares printed log-probabilities, 9 decimals)
  --prompt-name NAME   tier 1 prompt: long (default, 33,000 tokens: exercises the batched path AND the sparse selection), chat, code
  --ids-file FILE      tier 1 prompt as token ids (instead of --prompt-name; e.g. a long.ids that golden_compare.py wrote)
  --tail N             tier 1: positions scored after the batched prefill (default 512)
  --llama-cuda-args S  extra llama-perplexity arguments for the CUDA run (default none: llama.cpp's own --fit puts what fits on the card;
                       e.g. "-ngl 99 --cpu-moe" keeps every expert on the CPU)
  --llama-cpu-args S   extra llama-perplexity arguments for the CPU run (default "-ngl 0")
  --abs-kl X           absolute floor of the tier 2 rule, mean KL in nats (default 5e-4)
  --abs-top1 X         absolute floor, share of positions whose top-1 differs (default 0.01)
  --kv-fp16-row         also run row 2g (the engine with --kv fp16)
  --skip-build         use the builds in --work as they are; rows that need a missing one are SKIPPED
  --require-v100       stop unless nvidia-smi says compute capability 7.0
  --dry-run            print every command, run nothing
  -h, --help

Exit status: 0 = no row FAILED (SKIP / INFO are listed), 1 = a row FAILED, 2 = could not start (usage, no config, no llama.cpp).
EOF
}

CONFIG="" GGUF_ARG="" OUT="" WORK="" TEXT="" CTX=4096 CHUNKS=4 ONLY="" TIER="" LLAMA_DIR="" CUDA_ROOT="" CMAKE_ARG="" PY_ARG="" JOBS=""
GPU="" UP_REF=3906943 UP_SRC="" UP_HOOK=patch PROMPT_NAME=long IDS_FILE="" TAIL=512 LLAMA_CUDA_ARGS="" LLAMA_CPU_ARGS="-ngl 0" ABS_KL=5e-4 ABS_TOP1=0.01
KVFP16=0 SKIP_BUILD=0 REQUIRE_V100=0 DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --gguf) GGUF_ARG="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    --text) TEXT="$2"; shift 2;;
    --ctx) CTX="$2"; shift 2;;
    --chunks) CHUNKS="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --tier) TIER="$2"; shift 2;;
    --llama-dir) LLAMA_DIR="$2"; shift 2;;
    --cuda-root) CUDA_ROOT="$2"; shift 2;;
    --cmake) CMAKE_ARG="$2"; shift 2;;
    --python) PY_ARG="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --upstream-ref) UP_REF="$2"; shift 2;;
    --upstream-src) UP_SRC="$2"; shift 2;;
    --upstream-hook) UP_HOOK="$2"; shift 2;;
    --prompt-name) PROMPT_NAME="$2"; shift 2;;
    --ids-file) IDS_FILE="$2"; shift 2;;
    --tail) TAIL="$2"; shift 2;;
    --llama-cuda-args) LLAMA_CUDA_ARGS="$2"; shift 2;;
    --llama-cpu-args) LLAMA_CPU_ARGS="$2"; shift 2;;
    --abs-kl) ABS_KL="$2"; shift 2;;
    --abs-top1) ABS_TOP1="$2"; shift 2;;
    --kv-fp16-row) KVFP16=1; shift;;
    --skip-build) SKIP_BUILD=1; shift;;
    --require-v100) REQUIRE_V100=1; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "logit_identity: unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done
case "$UP_HOOK" in patch|none) :;; *) echo "logit_identity: --upstream-hook is patch or none" >&2; exit 2;; esac
case "$TIER" in ""|1|2) :;; *) echo "logit_identity: --tier is 1 or 2" >&2; exit 2;; esac

[[ -n "$CONFIG" ]] || CONFIG="$ROOT/strata-iq3_xxs.json"
[[ -n "$WORK" ]] || WORK="$ROOT/build-logit-identity"
[[ -n "$OUT" ]] || OUT="$WORK/results/$(date +%Y%m%d-%H%M%S)"        # under build-logit-identity/, which .gitignore covers (build*/)
[[ -n "$TEXT" ]] || TEXT="$HERE/prompts/kld_text.txt"
[[ -n "$LLAMA_DIR" ]] || LLAMA_DIR="$ROOT/third_party/llama.cpp"
[[ -n "$CUDA_ROOT" ]] || CUDA_ROOT=/usr/local/cuda-12.8
[[ -n "$JOBS" ]] || JOBS="$(nproc 2>/dev/null || echo 4)"
if [[ -n "$PY_ARG" ]]; then PY="$PY_ARG"; elif [[ -x "$ROOT/.venv/bin/python" ]]; then PY="$ROOT/.venv/bin/python"; else PY="$(command -v python3 || echo python3)"; fi
if [[ -n "$CMAKE_ARG" ]]; then CMAKE="$CMAKE_ARG"; elif [[ -x "$ROOT/.venv/bin/cmake" ]]; then CMAKE="$ROOT/.venv/bin/cmake"; else CMAKE="$(command -v cmake || echo cmake)"; fi
GC="$HERE/golden_compare.py"
KF="$HERE/kld_format.py"

# ------------------------------------------------------------------------------------------------ plumbing: rows, commands, logs
ROW_IDS=(); declare -A ST REASON SECS
TSV=""
if [[ "$DRY" == 0 ]]; then mkdir -p "$OUT" "$WORK"; OUT="$(cd "$OUT" && pwd)"; WORK="$(cd "$WORK" && pwd)"; TSV="$OUT/results.tsv"; : > "$TSV"; fi

banner() { printf '\n%s\n# %s\n%s\n' "$(printf '#%.0s' $(seq 1 100))" "$*" "$(printf '#%.0s' $(seq 1 100))"; }
note() { echo "  $*"; }
quote_cmd() { local a out=""; for a in "$@"; do out+="$(printf '%q ' "$a")"; done; printf '%s' "${out% }"; }
now() { date +%s; }

# record ROW STATUS SECONDS REASON
record() {
  local id="$1" st="$2" secs="$3" why="$4"
  ROW_IDS+=("$id"); ST[$id]="$st"; SECS[$id]="$secs"; REASON[$id]="$why"
  [[ -n "$TSV" ]] && printf '%s\t%s\t%s\t%s\n' "$id" "$st" "$secs" "$why" >> "$TSV"
  printf '  => row %s: %s - %s\n' "$id" "$st" "$why"
}

# wanted ROW: is this row selected by --only / --tier?  (2g only on request)
wanted() {
  local id="$1"
  if [[ -n "$ONLY" ]]; then [[ ",$ONLY," == *",$id,"* ]]; return; fi
  [[ "$id" == 2g && "$KVFP16" == 0 ]] && return 1
  [[ -z "$TIER" || "${id:0:1}" == "$TIER" ]]
}

# run LOG [NAME=VALUE ...] -- COMMAND...   prints the command; --dry-run stops there; the output goes to the terminal and to LOG; returns the command's status
run() {
  local log="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  local shown=""
  [[ ${#envs[@]} -gt 0 ]] && shown="$(quote_cmd "${envs[@]}") "
  echo "  + ${shown}$(quote_cmd "$@")"
  [[ "$DRY" == 1 ]] && return 0
  if [[ -n "$log" ]]; then
    mkdir -p "$(dirname "$log")"
    echo "# ${shown}$(quote_cmd "$@")" > "$log"
    env "${envs[@]+"${envs[@]}"}" "$@" 2>&1 | tee -a "$log"
    return "${PIPESTATUS[0]}"
  fi
  env "${envs[@]+"${envs[@]}"}" "$@"
}

# one line about a golden_compare report.json: "STATUS<TAB>reason"
summarize() { "$PY" "$GC" --summarize-report "$1" 2>/dev/null || printf 'UNKNOWN\tno readable report (%s)\n' "$1"; }

# ------------------------------------------------------------------------------------------------ preflight: the engine config, the GGUF, the card
banner "logit_identity: preflight"
echo "  repository $ROOT"
echo "  results    $OUT$([[ "$DRY" == 1 ]] && echo '  (dry run: nothing is created)')"
echo "  builds     $WORK"
EXE="" GGUF="" PACK="" NATIVE_PACK=1 TOKENIZER="" CWD="" GGUF_URLS="" CFG_OK=0
if [[ -f "$CONFIG" ]]; then
  CFG_OK=1
  while IFS='=' read -r k v; do
    case "$k" in EXE) EXE="$v";; GGUF) GGUF="$v";; PACK) PACK="$v";; NATIVE_PACK) NATIVE_PACK="$v";; TOKENIZER) TOKENIZER="$v";; CWD) CWD="$v";; GGUF_URLS) GGUF_URLS="$v";; esac
  done < <("$PY" "$GC" --engine-config "$CONFIG" --print-config 2>/dev/null)
  if [[ -z "$EXE" ]]; then echo "logit_identity: $CONFIG is not an engine config that $PY can read" >&2; [[ "$DRY" == 1 ]] || exit 2; fi
elif [[ "$DRY" == 1 ]]; then
  echo "  note: $CONFIG does not exist (./setup.sh writes it): the commands below use placeholders"
  EXE="<the engine of $CONFIG>"; GGUF="<the config's --native GGUF>"; NATIVE_PACK=1
else
  echo "logit_identity: $CONFIG not found: run ./setup.sh first (it writes strata-<model>.json), or pass --config" >&2; exit 2
fi
[[ -n "$GGUF_ARG" ]] && GGUF="$GGUF_ARG"
echo "  engine     $EXE"
echo "  pack       ${PACK:-?}  ($([[ "$NATIVE_PACK" == 1 ]] && echo 'native (IQ): full logits / log-probabilities through strata --serve' || echo 'canonical Q2_0: full logits through --dump-logits'))"
echo "  GGUF       $GGUF"
GGUF_MISSING=0
if [[ "$DRY" == 0 && ! -f "$GGUF" ]]; then
  GGUF_MISSING=1
  {
    echo "  The GGUF llama.cpp must load is not there: ${GGUF:-(the config has no --native)}"
    echo "  setup.py keeps the original model files in Strata-data/models/ and gives the engine the first shard as --native, so ./setup.sh (or"
    echo "  ./setup.sh --setup --model <size> --no-start) puts it there.  The same two files at the revision setup.py pins (HF_REVISIONS), into the same folder:"
    if [[ -n "$GGUF_URLS" ]]; then for u in $GGUF_URLS; do echo "    curl -L -C - -o \"$(dirname "${GGUF:-.}")/$(basename "$u")\" \"$u\""; done
    else echo "    (the file name is not an original-model shard name: take its URL from setup.py FAMILIES / HF_REVISIONS)"; fi
    echo "  or pass --gguf FILE.  The rows that need it are SKIPPED."
  } >&2
fi

if [[ -n "$EXE" && -f "$EXE" ]] && ! grep -aq STRATA_LOGITS_DUMP "$EXE" 2>/dev/null; then
  echo "  WARNING: $EXE has no STRATA_LOGITS_DUMP hook (it was built before verify.cpp got it): rebuild it (./setup.sh recompiles a changed source)." \
       "Without it the IQ-pack rows can only compare printed log-probabilities (9 decimals) and tier 2 has no KL, only the target token's log-probability."
fi
GPU_LINE=""
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader ${GPU:+-i "$GPU"} 2>/dev/null | head -1)"
fi
echo "  card       ${GPU_LINE:-(nvidia-smi not found)}"
if [[ "$DRY" == 0 && "$GPU_LINE" != *", 7.0,"* ]]; then
  echo "  WARNING (Gate 0): the card is not a compute capability 7.0 V100: what runs here is not what the port is for"
  [[ "$REQUIRE_V100" == 1 ]] && { echo "logit_identity: --require-v100 and the card is not a V100" >&2; exit 2; }
fi
CPU_FLAGS=" ${LOGIT_IDENTITY_CPU_FLAGS:-$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)} "     # (the variable: for tests)
HAVE_AVX512=0; [[ "$CPU_FLAGS" == *" avx512f "* && "$CPU_FLAGS" == *" avx512bw "* && "$CPU_FLAGS" == *" avx512vl "* ]] && HAVE_AVX512=1
HAVE_VBMI=0; [[ "$CPU_FLAGS" == *" avx512_vbmi "* ]] && HAVE_VBMI=1
NUMA_NODES="${LOGIT_IDENTITY_NUMA_NODES:-$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l)}"
echo "  CPU        AVX-512 $([[ $HAVE_AVX512 == 1 ]] && echo yes || echo no), VBMI $([[ $HAVE_VBMI == 1 ]] && echo yes || echo no); NUMA nodes $NUMA_NODES"
if [[ "$DRY" == 0 ]]; then
  { echo "date $(date -Is)"; echo "repo $(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo ?)"; echo "config $CONFIG"; echo "gguf $GGUF"; echo "card $GPU_LINE"
    echo "cpu $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null)"; echo "numa_nodes $NUMA_NODES"; echo "pack native=$NATIVE_PACK"; } > "$OUT/env.txt"
fi

# ------------------------------------------------------------------------------------------------ builds
export PATH="$ROOT/.venv/bin:$PATH"
GEN=(); command -v ninja >/dev/null 2>&1 && GEN=(-G Ninja)
NVCC="$CUDA_ROOT/bin/nvcc"
LLAMA_CUDA_BUILD="$WORK/llama-cuda" LLAMA_CPU_BUILD="$WORK/llama-cpu"
UP_BUILD="$WORK/upstream-build" UP_WT="$WORK/upstream-src"
LP_CUDA="$LLAMA_CUDA_BUILD/bin/llama-perplexity" LP_CPU="$LLAMA_CPU_BUILD/bin/llama-perplexity"
UP_BIN="$UP_BUILD/strata"
LLAMA_COMMON=(-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON)

# build_llama ROW DIR BIN CUDA(0|1): llama.cpp's llama-perplexity at the pinned commit
build_llama() {
  local id="$1" dir="$2" bin="$3" cuda="$4" t0; t0="$(now)"
  wanted "$id" || return 0
  banner "row $id: build llama.cpp ($([[ $cuda == 1 ]] && echo 'CUDA, sm_70, CUDA 12.8' || echo 'CPU only')) -> $bin"
  if [[ "$DRY" == 0 && "$SKIP_BUILD" == 1 ]]; then
    if [[ -x "$bin" ]]; then record "$id" PASS 0 "reusing $bin (--skip-build)"; else record "$id" SKIP 0 "--skip-build and $bin does not exist"; fi
    return
  fi
  if [[ "$GGUF_MISSING" == 1 ]]; then record "$id" SKIP 0 "not built: the GGUF it would load is not there (see the note at the top); a build takes a long time, so run again once it is"; return; fi
  if [[ "$DRY" == 0 && ! -d "$LLAMA_DIR/ggml" ]]; then record "$id" FAIL 0 "no llama.cpp at $LLAMA_DIR (./setup.sh downloads the pinned commit there; or --llama-dir)"; return; fi
  local defs=("${LLAMA_COMMON[@]}")
  if [[ "$cuda" == 1 ]]; then
    if [[ "$DRY" == 0 && ! -x "$NVCC" ]]; then record "$id" FAIL 0 "no nvcc at $NVCC (CUDA 12.8: ./setup.sh installs it; or --cuda-root)"; return; fi
    defs+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 "-DCMAKE_CUDA_COMPILER=$NVCC")
  else
    defs+=(-DGGML_CUDA=OFF)
  fi
  run "$OUT/logs/$id-configure.log" -- "$CMAKE" -S "$LLAMA_DIR" -B "$dir" "${GEN[@]+"${GEN[@]}"}" "${defs[@]}" \
      || { record "$id" FAIL "$(( $(now) - t0 ))" "cmake configure failed: see logs/$id-configure.log"; return; }
  run "$OUT/logs/$id-build.log" -- "$CMAKE" --build "$dir" --target llama-perplexity -j "$JOBS" \
      || { record "$id" FAIL "$(( $(now) - t0 ))" "build failed: see logs/$id-build.log"; return; }
  if [[ "$DRY" == 1 ]]; then record "$id" DRYRUN 0 "$bin"; else record "$id" PASS "$(( $(now) - t0 ))" "$bin"; fi
}

# build_upstream: row 1a's engine - upstream Strata in a git worktree, built for sm_70 with CUDA 12.8.  Sets UP_STATE (ok / skip / fail) and UP_NOTE.
UP_STATE=unbuilt UP_NOTE=""
build_upstream() {
  local t0 src="$UP_WT"; t0="$(now)"
  [[ -n "$UP_SRC" ]] && src="$UP_SRC"
  banner "build upstream Strata ($UP_REF) for sm_70 with CUDA 12.8 -> $UP_BIN"
  if [[ "$DRY" == 0 && "$SKIP_BUILD" == 1 ]]; then
    if [[ -x "$UP_BIN" ]]; then UP_STATE=ok; UP_NOTE="reusing $UP_BIN (--skip-build)"; else UP_STATE=skip; UP_NOTE="--skip-build and $UP_BIN does not exist"; fi
    return
  fi
  if [[ -z "$UP_SRC" ]]; then
    if [[ "$DRY" == 0 && ! -e "$UP_WT/CMakeLists.txt" ]] && ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
      UP_STATE=skip; UP_NOTE="$ROOT is not a git checkout, so there is no $UP_REF to make a worktree of: give --upstream-src DIR"; return
    fi
    if [[ "$DRY" == 1 || ! -e "$UP_WT/CMakeLists.txt" ]]; then
      run "$OUT/logs/up-worktree.log" -- git -C "$ROOT" worktree add --detach "$UP_WT" "$UP_REF" \
          || { UP_STATE=fail; UP_NOTE="git worktree add $UP_REF failed: see logs/up-worktree.log"; return; }
    else
      note "(the worktree $UP_WT is already there)"
    fi
  fi
  if [[ "$UP_HOOK" == patch ]]; then
    # adds ONLY the STRATA_LOGITS_DUMP measurement hook to verify.cpp (nothing happens unless the variable is set); idempotent
    if [[ "$DRY" == 1 ]] || ! git -C "$src" apply --reverse --check "$HERE/upstream_logits_dump.patch" >/dev/null 2>&1; then
      run "$OUT/logs/up-patch.log" -- git -C "$src" apply "$HERE/upstream_logits_dump.patch" \
          || { UP_STATE=fail; UP_NOTE="the measurement patch does not apply to $src (--upstream-hook none builds it untouched)"; return; }
    else
      note "(the STRATA_LOGITS_DUMP patch is already applied to $src)"
    fi
  fi
  if [[ "$DRY" == 0 && ! -x "$NVCC" ]]; then UP_STATE=fail; UP_NOTE="no nvcc at $NVCC (CUDA 12.8)"; return; fi
  # upstream's own setup.py flags (build_engine) plus its experimental-sm_70 switch (CMakeLists.txt: STRATA_EXPERIMENTAL_SM60); nothing else is patched
  run "$OUT/logs/up-configure.log" -- "$CMAKE" -S "$src" -B "$UP_BUILD" "${GEN[@]+"${GEN[@]}"}" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
      -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70 "-DCMAKE_CUDA_COMPILER=$NVCC" "-DSTRATA_GGML_DIR=$LLAMA_DIR" \
      || { UP_STATE=fail; UP_NOTE="upstream cmake configure failed: see logs/up-configure.log"; return; }
  run "$OUT/logs/up-build.log" -- "$CMAKE" --build "$UP_BUILD" --target strata -j "$JOBS" \
      || { UP_STATE=fail; UP_NOTE="upstream build failed: see logs/up-build.log"; return; }
  UP_STATE=ok; UP_NOTE="upstream $UP_REF built in $(( $(now) - t0 )) s"
  if [[ "$DRY" == 0 && "$UP_HOOK" == patch ]] && ! grep -aq STRATA_LOGITS_DUMP "$UP_BIN" 2>/dev/null; then
    note "WARNING: the upstream binary has no STRATA_LOGITS_DUMP hook although the patch was applied: it was built before the patch; row 1a compares printed log-probabilities"
  fi
}

# ------------------------------------------------------------------------------------------------ tier 1: bit-exact
# the pinned baseline B0: the port with its fast paths off, upstream's CPU tier, no NUMA mirror, expert rows rounding independent of the window
B0_ENV="STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0 STRATA_PROMPT_ATTN_OLD=1 STRATA_FORCE_AVX2=1 STRATA_IQ_MT_MIN=1"
UP_ENV="STRATA_PROMPT_ATTN_OLD=1 STRATA_FORCE_AVX2=1 STRATA_IQ_MT_MIN=1"
IQ512_ENV="STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0 STRATA_PROMPT_ATTN_OLD=1 STRATA_IQ_MT_MIN=1 STRATA_IQ512=1"
GC_GPU=(); [[ -n "$GPU" ]] && GC_GPU=(--gpu "$GPU")
T1_PROMPT=(--prompt-name "$PROMPT_NAME" --tail "$TAIL"); [[ -n "$IDS_FILE" ]] && T1_PROMPT=(--ids-file "$IDS_FILE" --tail "$TAIL")
T1_DIFF_STATUS=FAIL T1_NEED_MIRROR=0

# tier1_row ROW "what" [golden_compare arguments...]: runs golden_compare.py --exact and records the row from its report
tier1_row() {
  local id="$1" what="$2"; shift 2
  local dir="$OUT/tier1/$id" t0; t0="$(now)"
  banner "row $id: $what"
  run "$OUT/tier1/$id.log" -- "$PY" "$GC" --engine-config "$CONFIG" "${GC_GPU[@]+"${GC_GPU[@]}"}" "${T1_PROMPT[@]}" --exact --workdir "$dir" \
      --json "$OUT/tier1/$id.json" "$@"
  local rc=$?
  if [[ "$DRY" == 1 ]]; then record "$id" DRYRUN 0 "$what"; return 0; fi
  local line st why; line="$(summarize "$OUT/tier1/$id.json")"; st="${line%%$'\t'*}"; why="${line#*$'\t'}"
  if [[ "$T1_NEED_MIRROR" == 1 && ( $rc == 0 || $rc == 1 ) ]] && ! grep -q "expert arena MIRRORED" "$dir/cand.log" 2>/dev/null; then
    record "$id" SKIP "$(( $(now) - t0 ))" "the engine kept ONE copy of the experts (no 'expert arena MIRRORED' line in tier1/$id/cand.log), so the mirror was not exercised; the run itself said: $st, $why"
    return 0
  fi
  case "$rc:$st" in
    0:IDENTICAL) record "$id" PASS "$(( $(now) - t0 ))" "$why";;
    1:DIFFERENT) record "$id" "$T1_DIFF_STATUS" "$(( $(now) - t0 ))" "$why";;
    *) record "$id" FAIL "$(( $(now) - t0 ))" "golden_compare could not compare (exit $rc): $why; see tier1/$id.log";;
  esac
}

run_tier1() {
  local r
  if [[ "$DRY" == 0 && "$CFG_OK" == 1 && ! -x "$EXE" ]]; then
    for r in 1c 1a 1b 1d; do wanted "$r" && record "$r" SKIP 0 "the engine $EXE does not exist (./setup.sh builds it)"; done
    return
  fi
  # 1c is the control AND the baseline run (B0) that the other rows reuse as their reference (--reuse-ref)
  if wanted 1c || wanted 1a || wanted 1b || wanted 1d; then
    T1_DIFF_STATUS=FAIL T1_NEED_MIRROR=0
    tier1_row 1c "the same command twice (the port, fast paths off): bit-identical?" \
        --ref-env "$B0_ENV" --cand-env "$B0_ENV" --ref-args "--numa off" --cand-args "--numa off"
  fi
  local base=(--reuse-ref "$OUT/tier1/1c" --ref-env "$B0_ENV" --ref-args "--numa off")
  # the long prompt was tokenized (slowly, once) into 1c's folder: the other rows read those ids, which is also what --reuse-ref checks
  if [[ "$DRY" == 0 && -z "$IDS_FILE" && -f "$OUT/tier1/1c/long.ids" ]]; then T1_PROMPT=(--ids-file "$OUT/tier1/1c/long.ids" --tail "$TAIL"); fi
  if [[ "$DRY" == 0 && ! -f "$OUT/tier1/1c/ref.log" ]]; then
    for r in 1a 1b 1d; do wanted "$r" && record "$r" SKIP 0 "needs the baseline run of row 1c, which did not produce one"; done
    return
  fi

  if wanted 1a; then
    if [[ "$DRY" == 0 && "$NATIVE_PACK" != 1 && "$HAVE_VBMI" == 0 ]]; then
      record 1a SKIP 0 "upstream cannot run the canonical Q2_0 pack without AVX512-VBMI (this CPU has none): its AVX-512 kernel is the only one for that pack; use an IQ pack"
    elif [[ "$DRY" == 0 && "$UP_STATE" != ok ]]; then
      record 1a SKIP 0 "no upstream engine: ${UP_NOTE:-not built}"
    else
      T1_DIFF_STATUS=FAIL T1_NEED_MIRROR=0
      tier1_row 1a "the port (fast paths off, AVX2, --numa off) against upstream $UP_REF built for sm_70 (STRATA_PROMPT_ATTN_OLD=1): bit-identical?" \
          "${base[@]}" --cand-exe "$UP_BIN" --cand-env "$UP_ENV"
      [[ "$UP_HOOK" == none && "$DRY" == 0 ]] && note "(upstream was built without the STRATA_LOGITS_DUMP hook: row 1a compared printed log-probabilities, not bits)"
    fi
  fi
  if wanted 1b; then
    if [[ "$DRY" == 0 && "$NUMA_NODES" -lt 2 ]]; then
      record 1b SKIP 0 "this machine has $NUMA_NODES NUMA node: the mirror cannot be on (BIOS Node Interleaving must be Disabled)"
    else
      T1_DIFF_STATUS=FAIL T1_NEED_MIRROR=1
      tier1_row 1b "NUMA mirror on (--numa mirror) against off: bit-identical?" "${base[@]}" --cand-env "$B0_ENV" --cand-args "--numa mirror"
      T1_NEED_MIRROR=0
    fi
  fi
  if wanted 1d; then
    if [[ "$DRY" == 0 && "$NATIVE_PACK" != 1 ]]; then
      record 1d SKIP 0 "the canonical Q2_0 pack has no AVX2 kernel to compare its AVX-512 one with"
    elif [[ "$DRY" == 0 && "$HAVE_AVX512" == 0 ]]; then
      record 1d SKIP 0 "this CPU has no AVX-512: only the AVX2 tier exists"
    else
      # a difference here is a measurement (the tiers sum a row's products in another order), not a failure: it belongs to tier 2's noise budget
      T1_DIFF_STATUS=INFO T1_NEED_MIRROR=0
      tier1_row 1d "CPU expert tier: AVX-512 rows (STRATA_IQ512=1) against AVX2: identical, or how far apart?" "${base[@]}" \
          --cand-env "$IQ512_ENV" --cand-args "--numa off"
      T1_DIFF_STATUS=FAIL
    fi
  fi
}

# ------------------------------------------------------------------------------------------------ tier 2: against llama.cpp
T2="$OUT/tier2"
KLD_BASE="$T2/cuda.kld"
FLOOR_LOG="$T2/llama_cpu_vs_cuda.log"
BOS=""

# tier2_strata ROW "what" [golden_compare arguments...]: the engine against the llama.cpp reference, with the noise floor as the pass mark
tier2_strata() {
  local id="$1" what="$2"; shift 2
  local dir="$T2/$id" t0; t0="$(now)"
  banner "row $id: $what"
  if [[ "$DRY" == 0 && ! -f "$KLD_BASE" ]]; then record "$id" SKIP 0 "needs the reference of row 2c"; return; fi
  local floor=(--kld-floor "$FLOOR_LOG")
  if [[ "$DRY" == 0 && ! -f "$FLOOR_LOG" ]]; then floor=(); note "(no noise floor - row 2d did not run: the result is only MEASURED, there is no pass mark)"; fi
  run "$T2/$id.log" -- "$PY" "$GC" --engine-config "$CONFIG" "${GC_GPU[@]+"${GC_GPU[@]}"}" --ref-kld "$KLD_BASE" --kld-bos "${BOS:-none}" --kld-gguf "$GGUF" \
      "${floor[@]+"${floor[@]}"}" --kld-abs-kl "$ABS_KL" --kld-abs-top1 "$ABS_TOP1" --workdir "$dir" --json "$T2/$id.json" "$@"
  local rc=$?
  if [[ "$DRY" == 1 ]]; then record "$id" DRYRUN 0 "$what"; return 0; fi
  local line st why; line="$(summarize "$T2/$id.json")"; st="${line%%$'\t'*}"; why="${line#*$'\t'}"
  case "$rc:$st" in
    0:PASS) record "$id" PASS "$(( $(now) - t0 ))" "$why";;
    0:MEASURED) record "$id" INFO "$(( $(now) - t0 ))" "$why";;
    1:FAIL) record "$id" FAIL "$(( $(now) - t0 ))" "$why";;
    *) record "$id" FAIL "$(( $(now) - t0 ))" "golden_compare could not score (exit $rc): $why; see tier2/$id.log";;
  esac
}

run_tier2() {
  local r t0 rc gpu_env=() any=0
  [[ -n "$GPU" ]] && gpu_env=("CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$GPU")
  for r in 2c 2d 2e 2f 2g; do wanted "$r" && any=1; done
  [[ "$any" == 0 ]] && return
  local skip_all=""
  if [[ "$GGUF_MISSING" == 1 ]]; then skip_all="the GGUF ${GGUF:-?} is not there (the engine's --native): see the note at the top"
  elif [[ "$DRY" == 0 && ! -f "$TEXT" ]]; then skip_all="the text $TEXT does not exist"; fi
  if [[ -n "$skip_all" ]]; then for r in 2c 2d 2e 2f 2g; do wanted "$r" && record "$r" SKIP 0 "$skip_all"; done; return; fi
  [[ "$DRY" == 0 ]] && mkdir -p "$T2"

  # what llama.cpp writes over the first token of every chunk (nothing for a Qwen / GPT-2 BPE vocabulary)
  if [[ "$DRY" == 1 ]]; then BOS="<read from the GGUF>"; echo "  + $(quote_cmd "$PY" "$KF" bos "$GGUF")"
  else BOS="$("$PY" "$KF" bos "$GGUF" 2>/dev/null)" || BOS=""; fi
  if [[ -z "$BOS" ]]; then
    for r in 2c 2d 2e 2f 2g; do wanted "$r" && record "$r" FAIL 0 "cannot read the vocabulary's BOS settings from $GGUF (kld_format.py bos)"; done
    return
  fi
  note "llama.cpp's BOS handling for this GGUF: $BOS"

  if wanted 2c; then
    banner "row 2c: llama-perplexity (CUDA build) writes the reference: $CHUNKS chunks of $CTX tokens of $(basename "$TEXT")"
    t0="$(now)"
    if [[ "$DRY" == 0 && ! -x "$LP_CUDA" ]]; then
      record 2c SKIP 0 "no CUDA build of llama.cpp ($LP_CUDA): row 2a"
    else
      [[ "$DRY" == 0 ]] && rm -f "$KLD_BASE"
      # shellcheck disable=SC2086
      run "$T2/llama_cuda.log" "${gpu_env[@]+"${gpu_env[@]}"}" -- "$LP_CUDA" -m "$GGUF" -f "$TEXT" -c "$CTX" --chunks "$CHUNKS" --kl-divergence-base "$KLD_BASE" $LLAMA_CUDA_ARGS
      rc=$?
      if [[ "$DRY" == 1 ]]; then record 2c DRYRUN 0 "$KLD_BASE"
      elif [[ $rc != 0 ]]; then
        record 2c FAIL "$(( $(now) - t0 ))" "llama-perplexity (CUDA) exited $rc: see tier2/llama_cuda.log (out of memory? --llama-cuda-args \"-ngl 99 --cpu-moe\" keeps every expert on the CPU)"
      elif info="$("$PY" "$KF" info "$KLD_BASE" 2>&1)"; then
        record 2c PASS "$(( $(now) - t0 ))" "$(echo "$info" | head -1)"
      else
        record 2c FAIL "$(( $(now) - t0 ))" "the reference file is unusable: $info"
      fi
    fi
  fi

  if wanted 2d; then
    banner "row 2d: the noise floor: llama-perplexity (CPU build) against the CUDA reference"
    t0="$(now)"
    if [[ "$DRY" == 0 && ! -x "$LP_CPU" ]]; then record 2d SKIP 0 "no CPU build of llama.cpp ($LP_CPU): row 2b"
    elif [[ "$DRY" == 0 && ! -f "$KLD_BASE" ]]; then record 2d SKIP 0 "needs the reference of row 2c"
    else
      # shellcheck disable=SC2086
      run "$FLOOR_LOG" -- "$LP_CPU" -m "$GGUF" -f "$TEXT" -c "$CTX" --kl-divergence --kl-divergence-base "$KLD_BASE" $LLAMA_CPU_ARGS
      rc=$?
      if [[ "$DRY" == 1 ]]; then
        echo "  + $(quote_cmd "$PY" "$KF" floor-summary "$FLOOR_LOG")"; record 2d DRYRUN 0 "$FLOOR_LOG"
      elif [[ $rc != 0 ]]; then
        record 2d FAIL "$(( $(now) - t0 ))" "llama-perplexity (CPU) exited $rc: see tier2/llama_cpu_vs_cuda.log"
      else
        local line st why; line="$("$PY" "$KF" floor-summary "$FLOOR_LOG" 2>&1)"; st="${line%%$'\t'*}"; why="${line#*$'\t'}"
        if [[ "$st" == OK ]]; then record 2d INFO "$(( $(now) - t0 ))" "llama.cpp CPU against CUDA, the noise floor: $why"
        else record 2d FAIL "$(( $(now) - t0 ))" "the two llama.cpp backends disagree too much (or the log is unreadable) to be a reference: $why"; fi
      fi
    fi
  fi

  wanted 2e && tier2_strata 2e "Strata, port defaults (every fast path on) against llama.cpp CUDA, with the CPU-vs-CUDA floor"
  wanted 2f && tier2_strata 2f "Strata, fast paths off (STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0) against llama.cpp CUDA" \
      --cand-env "STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0"
  wanted 2g && tier2_strata 2g "Strata with a 16-bit KV cache (--kv fp16) against llama.cpp CUDA: the engine's int8 KV quantisation taken out" --cand-args "--kv fp16"
}

# ------------------------------------------------------------------------------------------------ main
if [[ "$DRY" == 0 && "$SKIP_BUILD" == 0 ]] && { wanted 2a || wanted 2b || wanted 1a; } && [[ ! -d "$LLAMA_DIR/ggml" ]]; then
  echo "logit_identity: no llama.cpp at $LLAMA_DIR: run ./setup.sh (it downloads the pinned commit there) or pass --llama-dir" >&2
  exit 2
fi
# builds first: a failure should show in the first minutes, not after a 40-minute llama.cpp run
build_llama 2a "$LLAMA_CUDA_BUILD" "$LP_CUDA" 1
build_llama 2b "$LLAMA_CPU_BUILD" "$LP_CPU" 0
wanted 1a && build_upstream

if [[ -z "$TIER" || "$TIER" == 1 || -n "$ONLY" ]]; then run_tier1; fi
if [[ -z "$TIER" || "$TIER" == 2 || -n "$ONLY" ]]; then run_tier2; fi

# ------------------------------------------------------------------------------------------------ the table
banner "logit_identity: results"
table() {
  printf '%-5s %-7s %7s  %s\n' ROW STATUS SECONDS REASON
  local id; for id in "${ROW_IDS[@]+"${ROW_IDS[@]}"}"; do printf '%-5s %-7s %7s  %s\n' "$id" "${ST[$id]}" "${SECS[$id]}" "${REASON[$id]}"; done
}
table
npass=0 nfail=0 nskip=0 ninfo=0
for id in "${ROW_IDS[@]+"${ROW_IDS[@]}"}"; do
  case "${ST[$id]}" in PASS) npass=$((npass+1));; FAIL) nfail=$((nfail+1));; SKIP) nskip=$((nskip+1));; INFO) ninfo=$((ninfo+1));; esac
done
echo
echo "Tier 2's pass rule is PROVISIONAL (mean KL and top-1 mismatch <= max(2 x llama.cpp's own CPU-vs-CUDA floor, an absolute floor); paired ln-PPL change within"
echo "2 standard errors of zero): the first real run on the box confirms or adjusts it.  docs/volta/LOGIT_IDENTITY.md explains every row."
if [[ "$DRY" == 0 ]]; then
  table > "$OUT/summary.txt"
  echo "results: $OUT/results.tsv  $OUT/summary.txt"
  echo "send back: summary.txt, results.tsv, env.txt, tier1/*.json, tier2/*.json, tier2/llama_cpu_vs_cuda.log and every report.txt under tier1/ and tier2/"
fi
echo "ROWS: $npass PASS, $nfail FAIL, $nskip SKIP, $ninfo INFO$([[ "$DRY" == 1 ]] && echo ' (dry run: nothing was run)')"
[[ "$nfail" == 0 ]] || exit 1
exit 0
