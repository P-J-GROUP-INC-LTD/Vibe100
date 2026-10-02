#!/usr/bin/env python3
"""e2e_mini.py - DS1-E: the end-to-end tests of the emulated mini engine (strata-ds41-mini-emu) against the NumPy oracle.  Called by ctest (cmake/ds41_engine.cmake).

    e2e_mini.py --engine EXE --work DIR --kv on|off [--n-slots N] [--threads-per-socket N]
        1. tools/ds41/ds1_e2e.py prepare: the mini GGUF + the oracle's trace of the prompt plus its greedy continuation (8 + 56 = 64 positions) with
           QuantConfig(int8_act, window_kv, compressed_kv, index) for --kv on, QuantConfig.int8() (the three KV flags off) for --kv off;
        2. EXE <the arguments ds1_e2e.py prints> --kv-quant on|off --n-slots N --trace DIR/engine: the engine decodes the same tokens (default N = 64 of the 128 experts: about
           half of the routed experts are served by the GPU cache (the DS-D kernels, emulated), the other half by the two-socket CPU pool);
        3. tools/ds41/ds1_e2e.py check: stage by stage, layer mode (stage-isolated replay: the strict criterion), whole-run mode and the greedy rule of DS1.md section 7.
    e2e_mini.py --engine EXE --work DIR --variants
        the engine's own invariants on the mini model: two runs give identical logits (determinism); the CPU pool's thread count, --numa and the prompt window (--window 4)
        change nothing, bit for bit; CPU-only, GPU-only and mixed expert placement agree to the float noise of the sums and quantisers.

Exit status: 0 pass, 1 fail, 2 could not run (no numpy / the engine or the fixture tools are missing): ctest turns 2 into "skipped".
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
TOOLS = REPO / "tools" / "ds41"
for p in (str(REPO), str(TOOLS)):
    if p not in sys.path:
        sys.path.insert(0, p)

PASS, FAIL, CANNOT = 0, 1, 2


def skip(msg: str) -> int:
    print(f"SKIPPED: {msg}")
    return CANNOT


def run(cmd, log: pathlib.Path | None = None, env=None, quiet=False) -> int:
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if log is not None:
        log.write_text(p.stdout)
    if not quiet:
        print(f"$ {' '.join(str(c) for c in cmd)}   [{time.time() - t0:.1f} s, exit {p.returncode}]")
        if p.returncode != 0 and log is None:
            print(p.stdout[-4000:])
    return p.returncode


def tool(*args, **kw):
    return [sys.executable, str(TOOLS / "ds1_e2e.py"), *args]


def prepare(work: pathlib.Path, quant: str, n_prompt: int = 8, max_new: int = 56) -> dict:
    out = work / "fixture"
    rc = run(tool("prepare", "--out", out, "--quant", quant, "--n-prompt", n_prompt, "--max-new", max_new))
    if rc != 0:
        raise SystemExit(skip("ds1_e2e.py prepare could not build the fixture") if rc == 2 else FAIL)
    fx = json.loads((out / "fixture.json").read_text())
    return {"dir": out, "fx": fx, "gguf": out / fx["gguf"], "prompt": fx["prompt"]}


def engine(exe, fixture: dict, work: pathlib.Path, name: str, extra=(), tokens=None, max_new=None, env=None) -> pathlib.Path:
    """Runs the engine on the fixture's prompt; returns the directory of the run's files (stdout in out.txt)."""
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    toks = tokens if tokens is not None else fixture["prompt"]
    mn = max_new if max_new is not None else fixture["fx"]["max_new"]
    cmd = [exe, "--gguf", fixture["gguf"], "--tokens", ",".join(str(t) for t in toks), "--max-new", mn, "-q", *extra]
    rc = run(cmd, log=d / "out.txt", env=env)
    if rc != 0:
        print((d / "out.txt").read_text()[-3000:])
        raise RuntimeError(f"the engine failed (exit {rc}) in run '{name}'")
    return d


def generated(d: pathlib.Path) -> list:
    for line in (d / "out.txt").read_text().splitlines():
        if line.startswith("generated ("):
            return [int(t) for t in line.split(":", 1)[1].split()]
    raise RuntimeError(f"{d}/out.txt: no 'generated' line")


# ---------------------------------------------------------------------------------------------------------------------------------
def e2e(args) -> int:
    quant = "int8-kv" if args.kv == "on" else "int8"
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    fix = prepare(work, quant)
    eng = work / "engine"
    if eng.exists():
        import shutil
        shutil.rmtree(eng)
    extra = ["--kv-quant", args.kv, "--n-slots", str(args.n_slots), "--threads-per-socket", str(args.threads_per_socket), "--stats-async", "--trace", eng]
    d = engine(args.engine, fix, work, "run", extra)
    print((d / "out.txt").read_text())
    rc = run(tool("check", "--fixture", fix["dir"], "--engine", eng, "-v", "--json", work / "check.json"), log=work / "check.txt")
    print((work / "check.txt").read_text())
    print(f"engine kv={args.kv}: tokens {generated(d)[:8]}... (generated {len(generated(d))}); checker exit {rc}")
    print("RESULT: " + ("PASS" if rc == 0 else "FAIL" if rc == 1 else "SKIPPED (the checker could not run)"))
    return rc


# ---------------------------------------------------------------------------------------------------------------------------------
def variants(args) -> int:
    import numpy as np
    from ref.ds41 import trace_io as TI
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    fix = prepare(work, "int8-kv")
    prompt = fix["prompt"]
    n_new = 4
    results = []

    def check(name: str, ok: bool, text: str = "") -> bool:
        results.append((name, bool(ok), text))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {text}" if text else ""))
        return bool(ok)

    def logits(name: str, extra, **kw) -> tuple:
        d = engine(args.engine, fix, work, name, ["--dump-logits", work / name / "logits.bin", *extra], max_new=n_new, **kw)
        return TI.read_logits_dump(d / "logits.bin"), generated(d)

    base_args = ["--n-slots", "64", "--threads-per-socket", "2"]
    a, ta = logits("base", base_args)
    b, tb = logits("again", base_args)
    check("determinism: the same run twice gives bit-identical logits (%d rows)" % a.shape[0], a.shape == b.shape and a.tobytes() == b.tobytes() and ta == tb)

    c, tc = logits("tps1", ["--n-slots", "64", "--threads-per-socket", "1"])
    d3, td = logits("tps3", ["--n-slots", "64", "--threads-per-socket", "3"])
    check("the CPU pool's thread count (1 / 2 / 3 per socket) does not change one bit", c.tobytes() == a.tobytes() and d3.tobytes() == a.tobytes())
    e, _ = logits("numa_off", ["--n-slots", "64", "--threads-per-socket", "2", "--numa", "off"])
    check("--numa off changes nothing", e.tobytes() == a.tobytes())
    w4, tw = logits("window4", ["--n-slots", "64", "--threads-per-socket", "2", "--window", "4"])
    check("a prompt window of 4 tokens gives the same logits as single tokens, bit for bit (T-invariance end to end)", w4.shape == a.shape and w4.tobytes() == a.tobytes(),
          "" if w4.tobytes() == a.tobytes() else f"max |diff| {np.max(np.abs(w4 - a)):.3g}")

    cpu_only, tcpu = logits("cpu_only", ["--n-slots", "0", "--threads-per-socket", "2"])
    gpu_only, tgpu = logits("gpu_only", ["--n-slots", "128"])
    scale = float(np.max(np.abs(gpu_only)))
    for nm, x in (("CPU-only", cpu_only), ("mixed (64 slots)", a)):
        rel = float(np.max(np.abs(x - gpu_only))) / scale
        rms = float(np.sqrt(np.mean((x - gpu_only) ** 2)) / np.sqrt(np.mean(gpu_only ** 2)))
        check(f"expert placement: {nm} agrees with GPU-only to the noise of the sums (max {rel:.2e}, rms {rms:.2e} of the logit scale)", rel < 5e-3 and x.shape == gpu_only.shape)
    check("expert placement: the same greedy tokens CPU-only / mixed / GPU-only", tcpu == tgpu == ta, f"{tcpu} {ta} {tgpu}")

    fails = [n for n, ok, _ in results if not ok]
    print("RESULT: " + ("PASS" if not fails else "FAIL"))
    return PASS if not fails else FAIL


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--kv", choices=("on", "off"), default="on")
    ap.add_argument("--n-slots", type=int, default=64)
    ap.add_argument("--threads-per-socket", type=int, default=2)
    ap.add_argument("--variants", action="store_true")
    args = ap.parse_args()
    try:
        import numpy  # noqa: F401
    except ImportError:
        return skip("python3 has no numpy: the end-to-end test needs it (the oracle is NumPy)")
    if not os.path.isfile(args.engine):
        return skip(f"the engine {args.engine} is not built")
    try:
        return variants(args) if args.variants else e2e(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else FAIL
    except Exception as e:   # noqa: BLE001
        print(f"e2e_mini: {type(e).__name__}: {e}")
        print("RESULT: FAIL")
        return FAIL


if __name__ == "__main__":
    sys.exit(main())
