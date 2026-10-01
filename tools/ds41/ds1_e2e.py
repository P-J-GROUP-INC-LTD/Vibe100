#!/usr/bin/env python3
"""ds1_e2e.py - the mini end-to-end fixture of Gate DS-1 and the checker the C++ end-to-end ctest calls (docs/deepseek/DS1_VERIFY.md section 7).

    ds1_e2e.py prepare --out DIR [--seed 0] [--n-prompt 8] [--max-new 56]
        writes DIR/gguf/*.gguf (the mini model of tools/ds41/make_mini_gguf.py, fixed seed, byte-identical on every run), DIR/oracle/ (the oracle's trace of the
        prompt plus its greedy continuation: 64 positions by default) and DIR/fixture.json.  Idempotent: an existing, matching fixture is kept.
    ds1_e2e.py args  --fixture DIR          prints "--gguf FILE --tokens a,b,c --max-new N": the engine command line (strata-ds41 ... --trace OUT)
    ds1_e2e.py check --fixture DIR --engine TRACE_DIR [--json FILE] [-v]
        compares the engine's trace with the oracle, exit status 0 pass / 1 fail / 2 could not run (no fixture, no numpy, the oracle failed to load)

WHAT `check` DECIDES (the Gate DS-1 criteria on the mini model, DS1.md section 7):
  1. the trace is complete: the engine's tokens start with the prompt, it has every required stage (trace_io.STAGES) for every layer and position it should;
  2. LAYER MODE (stage-isolated): for every position and layer the oracle's layer is run on the ENGINE's own inputs (ds1_replay.py) and each stage must
     be within its documented tolerance (ds1_compare.Tolerances): the strict criterion, immune to error accumulation;
  3. WHOLE-RUN MODE: the oracle is run on the engine's tokens (its own trajectory) and compared stage by stage; strict up to the first deviation
     above float noise, the chaos ceiling after it (ds1_compare.compare_sources); no NaN / Inf anywhere;
  4. GREEDY: every generated token is the argmax of the engine's own logits at the position before it, and the continuation equals the oracle's greedy
     continuation - or the first difference is a near-tie of the oracle's logits, or comes at or after a numerical deviation of step 3 that the quantisers
     amplified (a difference with NO earlier deviation is a bug: the logits agreed to 1e-6 and the argmax still differed).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

FIXTURE_FORMAT = "ds41-e2e-fixture"
EXIT_PASS, EXIT_FAIL, EXIT_CANNOT = 0, 1, 2
DEFAULT_QUANT = "int8-kv"                     # the engine's default (DS1.md section 2)


class CannotRun(Exception):
    """The checker itself cannot run (exit status 2): no fixture, no GGUF, a library missing."""


# ---------------------------------------------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------------------------------------------


def _fixture_params(seed: int, n_prompt: int, max_new: int, quant: str) -> dict:
    return {"format": FIXTURE_FORMAT, "version": 1, "seed": seed, "n_prompt": n_prompt, "max_new": max_new, "quant_preset": quant}


def prepare(out, *, seed: int = 0, n_prompt: int = 8, max_new: int = 56, quant: str = DEFAULT_QUANT, force: bool = False) -> dict:
    """Build the GGUF and the oracle's trace in `out`; -> the fixture dict (also written to out/fixture.json)."""
    import ds1_compare as C
    import make_mini_gguf as MM
    from ref.ds41 import trace_io as TI
    out = pathlib.Path(out)
    want = _fixture_params(seed, n_prompt, max_new, quant)
    fj = out / "fixture.json"
    if fj.is_file() and not force:
        try:
            have = json.loads(fj.read_text())
            if all(have.get(k) == v for k, v in want.items()) and (out / have["gguf"]).is_file() and (out / "oracle" / "trace.json").is_file():
                return have
        except (ValueError, KeyError):
            pass
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(MM.MiniConfig(), seed=seed)
    gdir = out / "gguf"
    for old in gdir.glob("*.gguf") if gdir.is_dir() else []:
        old.unlink()
    MM.build_mini(gdir, cfg)
    shard1 = sorted(gdir.glob("*00001-of-*.gguf"))[0]
    q = C.quant_from_name(quant)
    model = C.load_oracle_model(shard1, q, dtype="float32", max_seq_len=n_prompt + max_new + 8)
    # a prompt whose greedy continuation never hits EOS (the engine would stop there and the trace would be shorter)
    eos = 1
    prompt = gen = None
    for attempt in range(200):
        rng = np.random.default_rng([seed, attempt, 20261001])
        prompt = [int(t) for t in rng.integers(3, cfg.vocab, n_prompt)]
        gen = TI.greedy_continue(model, prompt, max_new, token_by_token=True)
        if eos not in gen:
            break
    else:
        raise CannotRun("no prompt found whose greedy continuation avoids EOS")
    tokens = prompt + gen
    tr = TI.run_oracle_trace(model, tokens, out / "oracle", mode="token_by_token", n_prompt=n_prompt, geometry="MiniGeom",
                             extra_meta={"fixture_seed": seed})
    fx = {**want, "geometry": "MiniGeom", "gguf": str(shard1.relative_to(out)), "prompt": prompt, "oracle_tokens": tokens, "quant": TI.quant_to_dict(q),
          "n_positions": len(tr.positions()), "n_layers": int(model.cfg.n_layers), "vocab": int(cfg.vocab)}
    fj.write_text(json.dumps(fx, indent=1))
    return fx


def load_fixture(path) -> tuple:
    path = pathlib.Path(path)
    fj = path / "fixture.json" if path.is_dir() else path
    if not fj.is_file():
        raise CannotRun(f"{fj}: no fixture (run `ds1_e2e.py prepare --out {path}` first)")
    fx = json.loads(fj.read_text())
    if fx.get("format") != FIXTURE_FORMAT:
        raise CannotRun(f"{fj}: not a {FIXTURE_FORMAT}")
    root = fj.parent
    if not (root / fx["gguf"]).is_file():
        raise CannotRun(f"{root / fx['gguf']}: the fixture's GGUF is missing")
    return fx, root


def engine_args(fx: dict, root) -> str:
    return f"--gguf {pathlib.Path(root) / fx['gguf']} --tokens {','.join(str(t) for t in fx['prompt'])} --max-new {fx['max_new']}"


# ---------------------------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Verdict:
    ok: bool
    checks: list = dataclasses.field(default_factory=list)       # (name, ok, text)
    reports: dict = dataclasses.field(default_factory=dict)

    def add(self, name: str, ok: bool, text: str = "") -> bool:
        self.checks.append((name, bool(ok), text))
        self.ok = self.ok and bool(ok)
        return bool(ok)

    def text(self, verbose: bool = False) -> str:
        L = []
        for name, ok, text in self.checks:
            L.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {text}" if text and (verbose or not ok) else ""))
        for k, rep in self.reports.items():
            if verbose or not rep.ok:
                L.append(f"--- {k}")
                L.append(rep.text(verbose=verbose))
        L.append("RESULT: " + ("PASS" if self.ok else "FAIL"))
        return "\n".join(L)

    def to_json(self) -> dict:
        return {"ok": self.ok, "checks": [{"name": n, "ok": o, "text": t} for n, o, t in self.checks], "reports": {k: r.to_json() for k, r in self.reports.items()}}


def _expected_positions(n_tokens: int) -> list:
    """The engine feeds tokens[0 .. n-2] through the model (the last generated token is an output only)."""
    return list(range(max(n_tokens - 1, 0)))


def greedy_check(eng, ref, fx: dict, full_report, v: Verdict) -> None:
    """Criterion 4 (module docstring)."""
    toks, n_prompt = eng.tokens, int(fx["n_prompt"])
    # (a) every token after the prompt is the argmax of the engine's own logits at the position before it
    bad = []
    for i in range(n_prompt, len(toks)):
        lg = eng.get("logits", i - 1)
        if lg is None:
            bad.append((i, "no logits"))
            continue
        lg = np.asarray(lg, dtype=np.float64).reshape(-1)
        if not np.isfinite(lg).all():
            bad.append((i, "NaN/Inf in the logits"))
        elif lg[toks[i]] < lg.max():
            bad.append((i, f"token {toks[i]} but argmax {int(np.argmax(lg))} (gap {lg.max() - lg[toks[i]]:.3g})"))
    v.add("greedy: every generated token is the argmax of the engine's own logits", not bad,
          "; ".join(f"position {i - 1}: {why}" for i, why in bad[:5]) or f"{len(toks) - n_prompt} tokens")
    # (b) the continuation equals the oracle's
    oracle_tokens = fx["oracle_tokens"]
    d = next((i for i in range(n_prompt, min(len(toks), len(oracle_tokens))) if toks[i] != oracle_tokens[i]), None)
    if d is None:
        v.add("greedy: continuation identical to the oracle's", len(toks) == len(oracle_tokens) or len(toks) < len(oracle_tokens),
              f"{len(toks) - n_prompt} generated tokens, oracle {len(oracle_tokens) - n_prompt}")
        return
    # first difference at token index d, chosen from the logits of position d - 1
    explained = None
    ol = ref.get("logits", d - 1)
    if ol is not None:
        ol = np.asarray(ol, dtype=np.float64).reshape(-1)
        gap = float(ol[oracle_tokens[d]] - ol[toks[d]])
        el = np.asarray(eng.get("logits", d - 1), dtype=np.float64).reshape(-1) if eng.has("logits", d - 1) else ol
        err = float(np.max(np.abs(el - ol)))
        if gap <= 10 * err + 1e-5 * float(np.max(np.abs(ol))):
            explained = f"near-tie: the oracle's logits of tokens {oracle_tokens[d]} and {toks[d]} differ by {gap:.3g}, the engine's logits differ from the oracle's by up to {err:.3g}"
    if explained is None and full_report is not None:
        early = [o for o in full_report.onsets if o[0] <= d - 1]
        if early:
            p, l, st, txt = early[0]
            explained = (f"the first deviation above float noise is at position {p}" + (f", layer {l}" if l is not None else "") + f", stage {st} ({txt[:90]}): "
                         "the quantisers amplify it, two correct implementations diverge from here on")
    if explained is None:
        v.add("greedy: continuation identical to the oracle's", False, f"token {d} (position {d - 1}): engine {toks[d]}, oracle {oracle_tokens[d]}; no near-tie, and no numerical "
              "deviation before it: the logits agreed to float noise and the argmax still differed")
    else:
        v.add("greedy: continuation identical to the oracle's, or differing only where explained", True,
              f"first difference at token {d}: engine {toks[d]}, oracle {oracle_tokens[d]}; {explained}")


def check(fixture, engine_dir, *, workdir=None, verbose: bool = False, do_layers: bool = True, do_full: bool = True) -> Verdict:
    import ds1_compare as C
    import ds1_replay as R
    from ref.ds41 import trace_io as TI
    fx, root = load_fixture(fixture)
    v = Verdict(True)
    try:
        eng = TI.Trace(engine_dir)
    except TI.TraceError as e:
        v.add("the engine wrote a trace", False, str(e))
        return v
    n_prompt = int(fx["n_prompt"])
    toks = eng.tokens
    v.add("trace: tokens start with the prompt", toks[:n_prompt] == fx["prompt"], f"engine {toks[:n_prompt]} vs fixture {fx['prompt']}")
    want_n = n_prompt + int(fx["max_new"])
    v.add("trace: the engine generated the requested number of tokens", len(toks) == want_n or (len(toks) < want_n and toks[-1] == 1), f"{len(toks)} tokens, expected {want_n}")
    pos_need = _expected_positions(len(toks))
    have = set(eng.positions())
    missing_pos = [p for p in pos_need if p not in have]
    v.add("trace: a file for every position", not missing_pos, f"missing positions {missing_pos[:10]}" if missing_pos else f"{len(pos_need)} positions")
    if not eng.has("embed", 0) or not pos_need:
        return v

    q = eng.quant()
    shard1 = root / fx["gguf"]
    try:
        model = C.load_oracle_model(shard1, q, dtype="float32", max_seq_len=max(len(toks), len(fx["oracle_tokens"])) + 8)
    except Exception as ex:                       # noqa: BLE001
        raise CannotRun(f"the oracle could not load {shard1}: {type(ex).__name__}: {ex}") from ex
    cfg = model.cfg
    positions = [p for p in pos_need if p in have]

    # ---- 2. layer mode
    if do_layers:
        lrep = R.replay_compare(model, eng, list(range(cfg.n_layers)), positions, strict=True, include_head=True)
        v.reports["layer mode (stage-isolated replay)"] = lrep
        v.add("layer mode: every stage of every layer and position within its tolerance", lrep.ok,
              (lrep.text().splitlines()[-1] if not lrep.ok else f"{len(lrep.samples)} samples, {sum(1 for s in lrep.samples if s.level == C.Level.FLIP)} at flip level"))

    # ---- 3. whole-run mode
    full = None
    ref = None
    if do_full:
        same = toks == fx["oracle_tokens"] and q == TI.quant_from_dict(fx["quant"])
        if same:
            ref = TI.Trace(root / "oracle")
        else:
            wd = pathlib.Path(workdir) if workdir else pathlib.Path(tempfile.mkdtemp(prefix="ds1_e2e_oracle_"))
            ref = TI.run_oracle_trace(model, toks, wd / "oracle_on_engine_tokens", mode="token_by_token", n_prompt=n_prompt, geometry="MiniGeom")
        full = C.compare_traces(eng, ref, positions=positions, strict=True)
        v.reports["whole run (the oracle's own trajectory)"] = full
        v.add("whole run: strict before the first deviation, chaos ceiling after it, no NaN/Inf", full.ok, full.text().splitlines()[-1])
        lg = [s for s in full.samples if s.stage == "logits" and "top1_match" in s.extra]
        if lg:
            v.add("whole run: logits finite (info: top-1 agreement %d/%d, KL max %.2e)" % (sum(s.extra["top1_match"] for s in lg), len(lg), max(s.extra["kl"] for s in lg)),
                  all(math.isfinite(s.rms_rel) for s in lg))

    # ---- 4. greedy
    if do_full and ref is not None:
        greedy_check(eng, ref, fx, full, v)
    return v


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=0); p.add_argument("--n-prompt", type=int, default=8)
    p.add_argument("--max-new", type=int, default=56); p.add_argument("--quant", default=DEFAULT_QUANT); p.add_argument("--force", action="store_true")
    p = sub.add_parser("args")
    p.add_argument("--fixture", required=True)
    p = sub.add_parser("check")
    p.add_argument("--fixture", required=True); p.add_argument("--engine", required=True); p.add_argument("--json"); p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--workdir"); p.add_argument("--no-layers", action="store_true"); p.add_argument("--no-full", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "prepare":
            fx = prepare(a.out, seed=a.seed, n_prompt=a.n_prompt, max_new=a.max_new, quant=a.quant, force=a.force)
            print(f"fixture in {a.out}: {fx['n_positions']} positions, prompt {fx['n_prompt']} tokens, {fx['max_new']} to generate")
            print("engine arguments: " + engine_args(fx, a.out))
            return EXIT_PASS
        if a.cmd == "args":
            fx, root = load_fixture(a.fixture)
            print(engine_args(fx, root))
            return EXIT_PASS
        v = check(a.fixture, a.engine, workdir=a.workdir, verbose=a.verbose, do_layers=not a.no_layers, do_full=not a.no_full)
        print(v.text(a.verbose))
        if a.json:
            pathlib.Path(a.json).write_text(json.dumps(v.to_json(), indent=1, default=float))
        return EXIT_PASS if v.ok else EXIT_FAIL
    except CannotRun as e:
        print(f"ds1_e2e: cannot run: {e}", file=sys.stderr)
        return EXIT_CANNOT
    except (ImportError, OSError) as e:
        print(f"ds1_e2e: cannot run: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_CANNOT


if __name__ == "__main__":
    sys.exit(main())
