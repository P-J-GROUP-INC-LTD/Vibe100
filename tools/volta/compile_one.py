#!/usr/bin/env python3
"""tools/volta/compile_one.py - compile ONE translation unit exactly as the build does, for sm_70, and report.

A full build of the engine takes 10-20 minutes; a kernel change needs only its own file recompiled to know whether
it builds for Volta, what it costs in registers / shared memory / spills, and whether its sm_70 SASS uses the tensor
cores (HMMA) or contains a trap (BPT.TRAP - an `__trap()` or NO_DEVICE_CODE stub the host must never launch on
sm_70).  This reads the command for the file from a CMake build's compile_commands.json (configure with
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON), writes the object to a temporary directory (so several people can compile in
parallel against one build tree without touching its objects), and summarises.

    python3 tools/volta/compile_one.py src/kernels/cuda/qsa_prompt_attn.cu
    python3 tools/volta/compile_one.py --build /path/to/build --kernel prompt_attn src/kernels/cuda/qsa_prompt_attn.cu
    python3 tools/volta/compile_one.py --sass src/prefill/gemm.cu        # also print the filtered SASS

Exit status: the compiler's on failure; 0 otherwise (traps are reported, not fatal - see tools/volta/sass_audit.py
for the gate).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD = os.environ.get("VIBE_BUILD_DIR", "")


def find_nvcc_tool(name: str) -> str:
    for cand in (shutil.which(name), f"/usr/local/cuda-12.8/bin/{name}", f"/usr/local/cuda/bin/{name}"):
        if cand and Path(cand).exists():
            return cand
    sys.exit(f"compile_one: {name} not found (install a CUDA 12.x toolkit; CUDA 13 cannot target sm_70)")


def demangle(names: list[str]) -> list[str]:
    cxxfilt = shutil.which("c++filt")
    if not cxxfilt or not names:
        return names
    out = subprocess.run([cxxfilt], input="\n".join(names), capture_output=True, text=True).stdout.splitlines()
    return out if len(out) == len(names) else names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="source file (path relative to the repository root, or absolute)")
    ap.add_argument("--build", default=DEFAULT_BUILD, help="CMake build dir with compile_commands.json "
                    "(default: $VIBE_BUILD_DIR)")
    ap.add_argument("--kernel", default="", help="only report kernels whose demangled name contains this")
    ap.add_argument("--sass", action="store_true", help="print the sm_70 SASS of the reported kernels")
    ap.add_argument("--keep", default="", help="copy the object file here")
    a = ap.parse_args()

    if not a.build:
        sys.exit("compile_one: pass --build DIR or set VIBE_BUILD_DIR")
    db = Path(a.build) / "compile_commands.json"
    if not db.exists():
        sys.exit(f"compile_one: {db} missing (configure with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)")
    src = Path(a.file)
    src = (src if src.is_absolute() else ROOT / src).resolve()
    entries = [e for e in json.loads(db.read_text()) if Path(e["file"]).resolve() == src]
    if not entries:
        sys.exit(f"compile_one: {src} is not in {db} (a new file needs a CMake target - re-run cmake after adding it)")
    e = entries[0]
    argv = e["arguments"] if "arguments" in e else shlex.split(e["command"])

    tmp = Path(tempfile.mkdtemp(prefix="compile_one_"))
    obj = tmp / (src.name + ".o")
    out_argv, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x == "-o":
            out_argv += ["-o", str(obj)]
            skip = True
            continue
        if x.startswith("-MF") or x in ("-MD", "-MMD"):
            if x == "-MF":
                skip = True
            continue
        out_argv.append(x)
    is_cuda = src.suffix == ".cu"
    if is_cuda:
        out_argv.insert(1, "-Xptxas=-v")
    r = subprocess.run(out_argv, cwd=e.get("directory", a.build), capture_output=True, text=True)
    log = r.stdout + r.stderr
    if r.returncode != 0:
        sys.stdout.write(log)
        print(f"\ncompile_one: FAILED ({r.returncode})")
        return r.returncode
    warnings = [l for l in log.splitlines() if "warning" in l.lower() and "deprecated-gpu-targets" not in l
                and "prior to '<compute/sm/lto>_75'" not in l]
    for w in warnings:
        print(w)
    if a.keep:
        shutil.copy2(obj, a.keep)
    if not is_cuda:
        print(f"compile_one: OK (host code) {src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}")
        return 0

    # ptxas -v: per-function registers / spills / smem, keyed by the mangled name
    res: dict[str, dict] = {}
    cur = None
    for line in log.splitlines():
        m = re.search(r"Compiling entry function '(\S+)' for 'sm_(\d+)'", line)
        if m:
            cur = m.group(1) if m.group(2) == "70" else None
            if cur:
                res[cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m:
            res[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = re.search(r"Used (\d+) registers", line)
        if m:
            res[cur]["regs"] = int(m.group(1))
            sm = re.search(r"(\d+) bytes smem", line)
            res[cur]["smem"] = int(sm.group(1)) if sm else 0

    sass = subprocess.run([find_nvcc_tool("cuobjdump"), "-sass", "-arch", "sm_70", str(obj)],
                          capture_output=True, text=True).stdout
    funcs: dict[str, list[str]] = {}
    fn = None
    for line in sass.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = m.group(1)
            funcs[fn] = []
        elif fn is not None:
            funcs[fn].append(line)
    names = sorted(set(funcs) | set(res))
    pretty = dict(zip(names, demangle(names)))
    rows = []
    for n in names:
        p = pretty[n]
        if a.kernel and a.kernel not in p:
            continue
        body = funcs.get(n, [])
        hmma = sum("HMMA" in l for l in body)
        trap = sum("BPT.TRAP" in l for l in body)
        r_ = res.get(n, {})
        rows.append((p, r_.get("regs", "?"), r_.get("smem", "?"), r_.get("spill_st", 0), r_.get("spill_ld", 0),
                     hmma, trap))
        if a.sass:
            print(f"\n==== {p}")
            print("\n".join(body))
    print(f"{'regs':>5} {'smem':>7} {'spillS':>6} {'spillL':>6} {'HMMA':>5} {'TRAP':>4}  kernel (sm_70)")
    for p, regs, smem, sst, sld, hmma, trap in rows:
        flag = "  <-- spills" if (sst or sld) else ""
        print(f"{regs:>5} {smem:>7} {sst:>6} {sld:>6} {hmma:>5} {trap:>4}  {p[:150]}{flag}")
    print(f"compile_one: OK {len(rows)} sm_70 function(s); object {'kept at ' + a.keep if a.keep else 'discarded'}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
