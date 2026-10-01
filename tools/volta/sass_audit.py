#!/usr/bin/env python3
"""tools/volta/sass_audit.py - the sm_70 SASS gate: no unexplained BPT.TRAP, and the tensor-core kernels really use HMMA.

WHY THIS EXISTS.  Upstream Strata is written for Turing and newer.  A kernel compiled for sm_70 from a body that
needs a newer instruction (mma.m16n8k16, cp.async, tf32 mma, a Blackwell-only path) does not fail to build: the
compiler replaces the body with `__trap()` (or ggml's NO_DEVICE_CODE) and the object file is perfectly valid.  The
engine then dies with "unspecified launch failure" (a trap is a sticky device error) the first time the host launches it - and the host
only does so on the path that needed the kernel, which can be hours into a long prompt.  The cure is to read the
machine code: `cuobjdump -sass` shows a `BPT.TRAP` in every kernel that can trap, and the host-side dispatch can
then be checked against that list ONCE, in `trap_allowlist.txt`, by a person who writes down why the host never
launches that kernel on this arch (and in which function the guard is).

The converse failure is silent too: a port that is supposed to use the V100's tensor cores but compiles to plain
FMAs (a wrong fragment type, an `#if` that picked the scalar fallback) still passes every parity test - it is just
8x slower.  `hmma_required.txt` lists kernels that MUST contain HMMA; zero HMMA in a matching kernel fails, and so
does a pattern that matches no kernel at all (the kernel was renamed, or never built: the gate must not pass
because the thing it guards has gone missing).

WHAT IT DOES, for every file given (or found in a build directory):

    cuobjdump -sass       -arch sm_70 FILE     per kernel: HMMA / IMMA count, BPT.TRAP count, instruction count,
                                               STL/LDL count (local-memory traffic: spills and local arrays)
    cuobjdump -res-usage  -arch sm_70 FILE     per kernel: registers, stack frame, static shared, local, constant

then demangles (c++filt), merges a kernel that appears in several files (a kernel linked into `strata` is also in
libstrata_kernels.a), and applies two rules:

    FAIL  a kernel with a BPT.TRAP matches no entry of the trap allowlist
    FAIL  a kernel pattern of the HMMA-required list matches nothing, or matches a kernel with no HMMA

    python3 tools/volta/sass_audit.py --build build-sm70                  # the engine, every parity program, every lib
    python3 tools/volta/sass_audit.py build-sm70/strata                   # one file
    python3 tools/volta/sass_audit.py /tmp/qsa_prompt_attn.o              # one object (what compile_one.py makes)
    python3 tools/volta/sass_audit.py --build build-sm70 --json audit.json --top 25

Files may be executables, static libraries (.a), objects (.o) or cubins.  A directory argument is treated as a build
directory.  Exit status: 0 = pass, 1 = a rule failed, 2 = could not run (no cuobjdump, nothing to scan).

THE ALLOWLIST (`trap_allowlist.txt`) is one entry per line, `regex<TAB>justification`; `#` starts a comment.  The
regex is searched in the DEMANGLED name (with its parameter list, e.g. `void mul_mat_q<(ggml_type)21, 72, false>(...)`).
The justification must name the host-side guard (file:function) that keeps the kernel from being launched on sm_70 -
an entry without one is a promise nobody can check.  Entries that matched no kernel are reported (a stale entry
usually means a kernel was renamed and its guard should be looked at again) but do not fail the gate.

THE HMMA-REQUIRED LIST (`hmma_required.txt`): `regex<TAB>justification`, or `regex<TAB>MIN<TAB>justification` for a
minimum HMMA count other than 1.

Local-memory note: `cuobjdump -res-usage` has no spill counters (ptxas -v does).  STACK is the bytes of stack frame
per thread (spills AND local arrays), `local ops` counts STL + LDL instructions; a kernel high on both is the one to
look at.  compile_one.py prints ptxas's own spill figures for a single file.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ALLOWLIST = HERE / "trap_allowlist.txt"
DEFAULT_REQUIRED = HERE / "hmma_required.txt"

# ---------------------------------------------------------------------------------------------- tool discovery


def find_tool(name: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit if Path(explicit).exists() else None
    cands = [shutil.which(name)]
    for root in ("/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.9/bin", "/usr/local/cuda-12.6/bin",
                 "/usr/local/cuda-12.4/bin", "/usr/local/cuda/bin", "/opt/cuda/bin"):
        cands.append(f"{root}/{name}")
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


# ------------------------------------------------------------------------------------------------- the parsers
# `cuobjdump -sass` prints, per kernel:
#     \t\tFunction : <mangled>
#     \t.headerflags ...
#             /*0010*/              @!PT SHFL.IDX PT, RZ, RZ, RZ, RZ ;                 /* 0x000000fffffff389 */
#                                                                                       /* 0x000fe200000e00ff */
# i.e. one line per instruction (offset, optional predicate, opcode, operands) followed by a line that is only the
# control-word encoding.  For an archive it also prints `member <path>:<name.o>:` before each member's section.

_FUNC_SASS = re.compile(r"^\s*Function : (\S+)")
_MEMBER = re.compile(r"^member (.+):$")
_PRED = re.compile(r"^@!?U?P[0-9T]\s+")

TENSOR_OPS = {"HMMA": "hmma", "IMMA": "imma", "BMMA": "bmma", "DMMA": "dmma", "QMMA": "qmma"}


class Kernel:
    """Everything the audit knows about one kernel, merged across the files it appears in."""
    __slots__ = ("name", "demangled", "files", "hmma", "imma", "other_mma", "trap", "instr", "local_ops",
                 "reg", "stack", "shared", "local", "const0", "has_sass", "has_res")

    def __init__(self, name: str):
        self.name = name
        self.demangled = name
        self.files: list[str] = []
        self.hmma = self.imma = self.other_mma = self.trap = self.instr = self.local_ops = 0
        self.reg = self.stack = self.shared = self.local = self.const0 = -1
        self.has_sass = self.has_res = False

    def to_json(self) -> dict:
        return {"name": self.name, "demangled": self.demangled, "files": self.files, "hmma": self.hmma,
                "imma": self.imma, "other_mma": self.other_mma, "trap": self.trap, "instructions": self.instr,
                "local_ops": self.local_ops, "registers": self.reg, "stack": self.stack, "shared": self.shared,
                "local": self.local, "const0": self.const0}


def parse_sass(lines, source: str = "") -> dict[str, dict]:
    """The per-kernel instruction counts of one `cuobjdump -sass` output (an iterable of lines)."""
    out: dict[str, dict] = {}
    cur: dict | None = None
    member = ""
    for line in lines:
        s = line.lstrip()
        if not s:
            continue
        c0 = s[0]
        if c0 == "/":
            # an instruction line starts `/*0010*/`; the encoding-only line below it starts `/* 0x`
            if s.startswith("/* 0x") or cur is None:
                continue
            end = s.find("*/")
            if end < 0:
                continue
            rest = s[end + 2:].lstrip()
            if rest.startswith("@"):
                rest = _PRED.sub("", rest, count=1)
            # the opcode token ends at whitespace or `;`
            tok = rest.split(None, 1)[0] if rest else ""
            tok = tok.rstrip(";")
            if not tok:
                continue
            cur["instr"] += 1 if not tok.startswith("NOP") else 0
            base = tok.split(".", 1)[0]
            key = TENSOR_OPS.get(base)
            if key is not None:
                cur[key] += 1
            elif base == "BPT":
                if tok == "BPT.TRAP":
                    cur["trap"] += 1
            elif base == "STL" or base == "LDL":
                cur["local_ops"] += 1
            continue
        if c0 == "F":
            m = _FUNC_SASS.match(line)
            if m:
                fresh = {"hmma": 0, "imma": 0, "bmma": 0, "dmma": 0, "qmma": 0, "trap": 0, "instr": 0, "local_ops": 0, "member": member}
                # the same kernel in two members of an archive (a template instantiated in several objects): the first
                # copy counts, the later ones are read into a scratch dict and dropped, not added up
                cur = fresh if m.group(1) in out else out.setdefault(m.group(1), fresh)
            continue
        if c0 == "m":
            m = _MEMBER.match(line.rstrip())
            if m:
                member = m.group(1).rsplit(":", 1)[-1] if ":" in m.group(1) else m.group(1)
                member = Path(member).name
                cur = None
    return out


_FUNC_RES = re.compile(r"^\s*Function (\S+):\s*$")
_KV = re.compile(r"([A-Z]+)(?:\[(\d+)\])?:(\d+)")


def parse_res_usage(lines) -> dict[str, dict]:
    """REG / STACK / SHARED / LOCAL / CONSTANT[0] per kernel from `cuobjdump -res-usage`."""
    out: dict[str, dict] = {}
    cur: str | None = None
    for line in lines:
        m = _FUNC_RES.match(line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        if cur is not None and line.startswith("  REG:"):
            d = out[cur]
            for k, idx, v in _KV.findall(line):
                if k == "CONSTANT":
                    if idx == "0":
                        d["const0"] = int(v)
                else:
                    d[k.lower()] = int(v)
            cur = None
    return out


def demangle(names: list[str]) -> dict[str, str]:
    cxxfilt = shutil.which("c++filt")
    if not cxxfilt or not names:
        return {n: n for n in names}
    # c++filt reads one symbol per line; the names are plain identifiers (no whitespace), so a batch is exact
    p = subprocess.run([cxxfilt], input="\n".join(names) + "\n", capture_output=True, text=True)
    out = p.stdout.splitlines()
    if len(out) != len(names):
        return {n: n for n in names}
    return dict(zip(names, out))


def short_name(demangled: str) -> str:
    """`void strata::kernels::(anonymous namespace)::prompt_attn_kernel<1>(float const*, ...)` -> `prompt_attn_kernel<1>`."""
    s = demangled
    if s.endswith(")") or s.endswith(" const"):
        depth, i = 0, len(s) - 1
        while i >= 0:
            ch = s[i]
            if ch == ")":
                depth += 1
            elif ch == "(":
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        # an anonymous-namespace marker also contains parentheses: only strip when the paren group is not it
        if i > 0 and not s[:i].endswith("anonymous namespace"):
            s = s[:i]
    if s.startswith("void "):
        s = s[5:]
    s = s.replace("(anonymous namespace)::", "")
    s = re.sub(r"^(strata::(kernels|prefill|core)::)+", "", s)
    s = s.replace("(ggml_type)", "t")
    return s


# ------------------------------------------------------------------------------------------- running cuobjdump


class AuditError(Exception):
    pass


def _cuobjdump_lines(tool: str, args: list[str]):
    """Yield the stdout lines of a cuobjdump run; raise AuditError(stderr) on a failure other than 'no code'."""
    p = subprocess.Popen([tool, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         errors="replace", bufsize=1 << 20)
    assert p.stdout is not None
    for line in p.stdout:
        yield line
    err = p.stderr.read() if p.stderr else ""
    rc = p.wait()
    if rc != 0:
        low = err.lower()
        if "does not contain device code" in low or "no sass" in low or "does not contain code" in low:
            return
        raise AuditError(f"cuobjdump {' '.join(args)} failed ({rc}): {err.strip()[:300]}")


def scan_file(tool: str, arch: str, path: Path) -> dict:
    """Run the two cuobjdump passes over one file.  Returns {'path', 'sass': {...}, 'res': {...}, 'note'}."""
    note = ""
    try:
        sass = parse_sass(_cuobjdump_lines(tool, ["-sass", "-arch", arch, str(path)]))
        res = parse_res_usage(_cuobjdump_lines(tool, ["-res-usage", "-arch", arch, str(path)]))
    except AuditError as e:
        return {"path": str(path), "sass": {}, "res": {}, "note": str(e)}
    if not sass and not res:
        note = f"no {arch} device code"
    return {"path": str(path), "sass": sass, "res": res, "note": note}


# ------------------------------------------------------------------------------------------- finding the files


def _is_elf(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def _is_archive(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return f.read(8) == b"!<arch>\n"
    except OSError:
        return False


def discover(build: Path) -> list[Path]:
    """The engine, the parity / test programs (ELF executables at the top of the build dir or in bin/) first, then
    the static libraries.  Shared libraries and CMake's own files are left alone."""
    exes, libs = [], []
    for d in (build, build / "bin"):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if not p.is_file() or p.is_symlink() and not p.exists():
                continue
            n = p.name
            if n.endswith((".so", ".o", ".txt", ".cmake", ".json", ".ninja", ".log")) or ".so." in n:
                continue
            if n.endswith(".a"):
                if _is_archive(p):
                    libs.append(p)
            elif os.access(p, os.X_OK) and _is_elf(p):
                exes.append(p)
    # `strata` first: it is the one that matters and the one that takes longest
    exes.sort(key=lambda p: (p.name != "strata", p.name))
    return exes + libs


# --------------------------------------------------------------------------------------------- the rule files


def read_rules(path: Path) -> list[tuple[re.Pattern, str, list[str]]]:
    """(compiled regex, raw regex text, fields[1:]) per non-comment line; fields are TAB-separated."""
    rules = []
    if not path.exists():
        raise AuditError(f"{path} does not exist")
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) < 2 or not fields[0].strip():
            raise AuditError(f"{path}:{n}: expected `regex<TAB>justification`, got {raw[:80]!r}")
        try:
            rx = re.compile(fields[0])
        except re.error as e:
            raise AuditError(f"{path}:{n}: bad regex {fields[0]!r}: {e}") from e
        rules.append((rx, fields[0], [f.strip() for f in fields[1:]]))
    return rules


# ------------------------------------------------------------------------------------------------------ the gate


def merge(scans: list[dict]) -> dict[str, Kernel]:
    kernels: dict[str, Kernel] = {}
    for sc in scans:
        fname = Path(sc["path"]).name
        for name, d in sc["sass"].items():
            k = kernels.setdefault(name, Kernel(name))
            if fname not in k.files:
                k.files.append(fname)
            if not k.has_sass:   # the same kernel in a library and in the program that links it: first one wins
                k.has_sass = True
                k.hmma, k.imma, k.trap, k.instr, k.local_ops = d["hmma"], d["imma"], d["trap"], d["instr"], d["local_ops"]
                k.other_mma = d["bmma"] + d["dmma"] + d["qmma"]
        for name, d in sc["res"].items():
            k = kernels.setdefault(name, Kernel(name))
            if fname not in k.files:
                k.files.append(fname)
            if not k.has_res:
                k.has_res = True
                k.reg, k.stack, k.shared, k.local = d.get("reg", -1), d.get("stack", -1), d.get("shared", -1), d.get("local", -1)
                k.const0 = d.get("const0", -1)
    dem = demangle(list(kernels))
    for n, k in kernels.items():
        k.demangled = dem.get(n, n)
    return kernels


def evaluate(kernels: dict[str, Kernel], allow, required) -> dict:
    """Apply the two rules.  Returns the result dict (also the JSON's `gate` section)."""
    unexplained, by_entry = [], {rx_text: [] for _, rx_text, _ in allow}
    for k in kernels.values():
        if k.trap <= 0:
            continue
        for rx, rx_text, _ in allow:
            if rx.search(k.demangled):
                by_entry[rx_text].append(k.name)
                break
        else:
            unexplained.append(k.name)
    reasons: list[str] = []
    req_out = []
    for rx, rx_text, fields in required:
        mn = int(fields[0]) if fields and fields[0].isdigit() else 1
        matched = [k for k in kernels.values() if rx.search(k.demangled)]
        failing = [k.name for k in matched if k.hmma < mn]
        req_out.append({"pattern": rx_text, "min_hmma": mn, "matched": [k.name for k in matched], "failing": failing})
        if not matched:
            reasons.append(f"hmma-required pattern {rx_text!r} matched NO kernel (renamed, or not built?)")
        for n in failing:
            reasons.append(f"kernel {short_name(kernels[n].demangled)} matches {rx_text!r} but has "
                           f"{kernels[n].hmma} HMMA (< {mn}): it runs on CUDA cores")
    if unexplained:
        reasons.append(f"{len(unexplained)} kernel(s) with BPT.TRAP match no allowlist entry")
    return {"unexplained_traps": unexplained, "allowlisted_traps": by_entry, "required": req_out,
            "reasons": reasons, "pass": not reasons}


# ------------------------------------------------------------------------------------------------------ the report


def fmt_table(rows: list[list[str]], header: list[str], right: set[int] = frozenset()) -> str:
    cols = list(zip(*([header] + rows))) if rows else [[h] for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]
    def line(r):
        return "  ".join((str(c).rjust(w) if i in right else str(c).ljust(w)) for i, (c, w) in enumerate(zip(r, widths))).rstrip()
    out = [line(header), line(["-" * w for w in widths])]
    out += [line(r) for r in rows]
    return "\n".join(out)


def report(kernels: dict[str, Kernel], gate: dict, allow, scans: list[dict], arch: str, top: int, out=sys.stdout,
           list_traps: bool = False, list_hmma_max: int = 60, verbose: bool = False) -> None:
    p = lambda *a: print(*a, file=out)
    ks = list(kernels.values())
    ok_files = [s for s in scans if s["sass"] or s["res"]]
    p(f"sass_audit: {arch}, {len(scans)} file(s) scanned ({len(ok_files)} with device code), {len(ks)} distinct kernels")
    nocode = [Path(s["path"]).name for s in scans if s["note"].startswith("no ")]
    if nocode:
        p(f"  no {arch} device code (host-only programs, fine): {', '.join(nocode)}")
    for s in scans:
        if s["note"] and not s["note"].startswith("no "):
            p(f"  note: {Path(s['path']).name}: {s['note']}")

    # ---- traps
    traps = [k for k in ks if k.trap > 0]
    p("")
    p(f"== BPT.TRAP: {len(traps)} kernel(s) ==")
    justif = {rx_text: fields for _, rx_text, fields in allow}
    rows = []
    for rx_text, names in gate["allowlisted_traps"].items():
        if names:
            rows.append([str(len(names)), "allowlisted", rx_text if len(rx_text) <= 64 else rx_text[:61] + "..."])
    if rows:
        p(fmt_table(rows, ["kernels", "status", "allowlist entry"], right={0}))
    if list_traps:
        for rx_text, names in gate["allowlisted_traps"].items():
            for n in names:
                p(f"    trap  {short_name(kernels[n].demangled)}")
    unexp = gate["unexplained_traps"]
    p(f"  UNEXPLAINED: {len(unexp)}")
    for n in unexp:
        k = kernels[n]
        p(f"    !! {short_name(k.demangled)}   (in {', '.join(k.files[:3])}; {k.trap} x BPT.TRAP)")
    stale = [rx_text for rx_text, names in gate["allowlisted_traps"].items() if not names]
    if stale:
        p(f"  {len(stale)} allowlist entr{'y' if len(stale) == 1 else 'ies'} matched nothing (informational"
          f"{'' if verbose else '; --verbose lists them'})")
        for rx_text in stale if verbose else []:
            p(f"    - {rx_text}")

    # ---- HMMA
    hm = sorted((k for k in ks if k.hmma > 0), key=lambda k: -k.hmma)
    p("")
    p(f"== tensor-core instructions: {len(hm)} kernel(s) with HMMA ==")
    if hm:
        rows = [[short_name(k.demangled)[:70], str(k.hmma), str(k.reg), str(k.stack), str(k.shared),
                 (k.files[0] if k.files else "")[:34]] for k in hm[:list_hmma_max]]
        p(fmt_table(rows, ["kernel", "HMMA", "regs", "stack", "smem", "first seen in"], right={1, 2, 3, 4}))
        if len(hm) > list_hmma_max:
            p(f"  ... and {len(hm) - list_hmma_max} more (see --json)")
    other = [k for k in ks if k.imma or k.other_mma]
    if other:
        p(f"  ({len(other)} kernel(s) with IMMA/other MMA instructions - not expected on sm_70)")

    # ---- local memory
    p("")
    p(f"== top {top} kernels by local memory (stack bytes per thread; STL+LDL = spill / local-array traffic; trap stubs left out) ==")
    stubs = {n for ns in gate["allowlisted_traps"].values() for n in ns}   # trap stubs: their frame is not a spill
    heavy = sorted((k for k in ks if k.name not in stubs and (k.stack > 0 or k.local > 0 or k.local_ops > 0)),
                   key=lambda k: (-(max(k.stack, 0) + max(k.local, 0)), -k.local_ops, k.demangled))[:top]
    if heavy:
        rows = [[short_name(k.demangled)[:60], str(k.stack), str(k.local), str(k.local_ops), str(k.reg), str(k.shared),
                 (k.files[0] if k.files else "")[:34]] for k in heavy]
        p(fmt_table(rows, ["kernel", "stack", "local", "STL+LDL", "regs", "smem", "first seen in"], right={1, 2, 3, 4, 5}))
    else:
        p("  none")
    n_stack = sum(1 for k in ks if k.stack > 0 and k.name not in stubs)
    n_hi_reg = sum(1 for k in ks if k.reg >= 200)
    p(f"  {n_stack} of {len(ks) - len(stubs)} real kernels have a non-empty stack frame; {n_hi_reg} use >= 200 registers")

    # ---- required
    p("")
    p("== HMMA-required kernels ==")
    if not gate["required"]:
        p("  (no patterns configured)")
    for r in gate["required"]:
        if not r["matched"]:
            p(f"  FAIL  {r['pattern']!r}: NO kernel matched")
        else:
            for n in r["matched"]:
                k = kernels[n]
                tag = "ok  " if k.hmma >= r["min_hmma"] else "FAIL"
                p(f"  {tag}  {short_name(k.demangled)[:80]}: {k.hmma} HMMA, {k.reg} regs, stack {k.stack}, smem {k.shared}")

    p("")
    if gate["pass"]:
        p("RESULT: PASS")
    else:
        p("RESULT: FAIL")
        for r in gate["reasons"]:
            p(f"  - {r}")


# ----------------------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="files (executables, .a, .o, .cubin) or build directories to scan")
    ap.add_argument("--build", action="append", default=[], metavar="DIR",
                    help="a CMake build directory: scans its executables and static libraries")
    ap.add_argument("--arch", default="sm_70", help="the SASS architecture to disassemble (default sm_70)")
    ap.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST), help="trap allowlist (regex<TAB>justification)")
    ap.add_argument("--hmma-required", default=str(DEFAULT_REQUIRED), help="kernels that must contain HMMA")
    ap.add_argument("--no-required", action="store_true", help="skip the HMMA-required rule (e.g. one unrelated object)")
    ap.add_argument("--json", metavar="FILE", help="write the full result as JSON ('-' = stdout, which silences the table)")
    ap.add_argument("--top", type=int, default=15, help="rows of the local-memory table (default 15)")
    ap.add_argument("--list-traps", action="store_true", help="print every allowlisted trap kernel, not only the counts")
    ap.add_argument("-v", "--verbose", action="store_true", help="also list the allowlist entries that matched nothing")
    ap.add_argument("--jobs", type=int, default=2, help="files scanned in parallel (default 2: cuobjdump is CPU-bound)")
    ap.add_argument("--cuobjdump", help="path to cuobjdump (default: PATH, then /usr/local/cuda-12.*/bin)")
    ap.add_argument("--max-stack", type=int, default=0,
                    help="also FAIL when a kernel that is not allowlisted has a stack frame above this many bytes "
                         "(default 0 = off)")
    a = ap.parse_args(argv)

    tool = find_tool("cuobjdump", a.cuobjdump)
    if not tool:
        print("sass_audit: cuobjdump not found (install a CUDA 12.x toolkit, or pass --cuobjdump)", file=sys.stderr)
        return 2

    files: list[Path] = []
    for b in a.build:
        d = Path(b)
        if not d.is_dir():
            print(f"sass_audit: --build {b}: not a directory", file=sys.stderr)
            return 2
        files += discover(d)
    for s in a.paths:
        p = Path(s)
        if p.is_dir():
            files += discover(p)
        elif p.exists():
            files.append(p)
        else:
            print(f"sass_audit: {s}: no such file", file=sys.stderr)
            return 2
    seen, uniq = set(), []
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(f)
    files = uniq
    if not files:
        print("sass_audit: nothing to scan (give files, or --build DIR containing executables / .a files)", file=sys.stderr)
        return 2

    try:
        allow = read_rules(Path(a.allowlist))
        required = [] if a.no_required else read_rules(Path(a.hmma_required))
    except AuditError as e:
        print(f"sass_audit: {e}", file=sys.stderr)
        return 2

    quiet = a.json == "-"
    if not quiet:
        print(f"sass_audit: scanning {len(files)} file(s) with {tool} -arch {a.arch} ...", file=sys.stderr)
    scans: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(scan_file, tool, a.arch, f): f for f in files}
        results = {}
        for fut in concurrent.futures.as_completed(futs):
            results[futs[fut]] = fut.result()
    scans = [results[f] for f in files]
    errors = [s for s in scans if s["note"].startswith("cuobjdump")]
    if len(errors) == len(scans):
        print("sass_audit: cuobjdump failed on every file:\n  " + "\n  ".join(s["note"] for s in errors[:3]), file=sys.stderr)
        return 2
    if not any(s["sass"] or s["res"] for s in scans):
        print(f"sass_audit: none of the {len(scans)} file(s) contains {a.arch} device code", file=sys.stderr)
        return 2

    kernels = merge(scans)
    gate = evaluate(kernels, allow, required)
    if a.max_stack > 0:
        for k in kernels.values():
            if k.stack > a.max_stack and k.name not in {n for ns in gate["allowlisted_traps"].values() for n in ns}:
                gate["reasons"].append(f"kernel {short_name(k.demangled)} has a {k.stack} B stack frame (> --max-stack {a.max_stack})")
        gate["pass"] = not gate["reasons"]

    if a.json:
        doc = {"arch": a.arch, "files": [{"path": s["path"], "kernels": len(s["sass"] or s["res"]), "note": s["note"]} for s in scans],
               "summary": {"kernels": len(kernels), "trap_kernels": sum(1 for k in kernels.values() if k.trap),
                           "hmma_kernels": sum(1 for k in kernels.values() if k.hmma)},
               "gate": gate, "kernels": [k.to_json() for k in sorted(kernels.values(), key=lambda k: k.demangled)]}
        text = json.dumps(doc, indent=1)
        if a.json == "-":
            print(text)
        else:
            Path(a.json).write_text(text, encoding="utf-8")
            print(f"sass_audit: wrote {a.json}", file=sys.stderr)
    if not quiet:
        report(kernels, gate, allow, scans, a.arch, a.top, list_traps=a.list_traps, verbose=a.verbose)
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
