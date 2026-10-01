#!/usr/bin/env python3
"""ds1_compare.py - compare an engine trace with the NumPy oracle, stage by stage (DS-1 verification, docs/deepseek/DS1_VERIFY.md).

    ds1_compare.py trace   --engine DIR (--oracle DIR | --gguf G... [--quant int8-kv|int8|exact|reference])   whole-run comparison
    ds1_compare.py layers  --engine DIR --gguf G... --layers 0,5,39,head --positions 3,17,31             layer-by-layer replay (real weights)
    ds1_compare.py oracle  --gguf G... --tokens 1,2,3 --out DIR [--quant ...] [--max-new N]               write the oracle's trace
    ds1_compare.py logits  --engine FILE --oracle FILE                                                    DS-1 gate statistics on two logits dumps

Exit status: 0 pass, 1 a stage is outside its tolerance (or, with --strict, a required stage is missing), 2 the tool could not run.

WHAT IS COMPARED.  The trace format is ref/ds41/trace_io.py (one .npy per stage, layer and position).  For every (position, layer, stage) in
execution order the engine's array is compared with the oracle's:
  * float stages: rms_rel = rms(E - O) / rms(O), max_rel = max|E - O| / max|O| (relative to the TENSOR'S OWN scale, like ref/ds41/README.md),
    cosine; NaN / Inf in the engine where the oracle is finite is always a failure;
  * integer stages (`topk`, `router_idx`, `cand_blocks`): EXACT set equality; a difference is excused only when the oracle's recorded
    selection margin (gap between the k-th and (k+1)-th candidate) is within the stage's near-tie threshold - a "near-tie", never a pass;
  * router_w: compared per expert id when the two selections agree; logits: the float metrics plus argmax agreement, KL(oracle || engine).
Each sample gets a level: OK (within the soft tolerance), FLIP (above soft, within hard: a rounding decision of the int8 / fp8 / fp4
quantisers fell the other way on one element - expected now and then, see DS1_VERIFY.md section "Flips") or FAIL (beyond hard).  A stage fails when
any sample fails or when the fraction of FLIP samples exceeds the stage's budget (a systematic error shows up there long before it is large).
The tolerances (TOLERANCES below) are derived in docs/deepseek/DS1_VERIFY.md and re-measured by tools/ds41/ds1_noise.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import math
import pathlib
import sys
from typing import Callable, Iterable

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ref.ds41 import trace_io as TI  # noqa: E402
from ref.ds41.trace_io import STAGE, STAGE_ORDER, Trace, TraceError  # noqa: E402


class Level(enum.IntEnum):
    OK = 0
    FLIP = 1
    FAIL = 2


# ---------------------------------------------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FloatTol:
    soft_rms: float        # a sample at or below both soft values is OK
    soft_max: float
    hard_rms: float        # a sample above either hard value is a FAIL
    hard_max: float
    budget: float          # max fraction of the stage's samples that may sit between soft and hard


@dataclasses.dataclass(frozen=True)
class SetTol:
    tie_rel: float         # a selection mismatch is a near-tie when margin <= tie_rel * scale
    max_diff_frac: float   # ... and at most this fraction of the selection may differ (at least one entry)
    budget: float


# Filled from the measurements of tools/ds41/ds1_noise.py (docs/deepseek/DS1_VERIFY.md has the derivation).
# "layer": stage-isolated replay (every stage fed with the engine's own inputs) - only the stage's own float noise and its own quantiser flips.
# "full":  whole-run comparison - the noise of every earlier stage and position has accumulated.
TOLERANCES: dict = {"layer": {}, "full": {}}


def _fill_tolerances() -> None:
    def tab(mode, rows):
        TOLERANCES[mode].update(rows)

    f = FloatTol
    # soft_rms soft_max hard_rms hard_max budget
    tab("layer", {
        "embed": f(1e-12, 1e-12, 1e-9, 1e-9, 0.0),
        "engram_out": f(2e-6, 2e-6, 2e-3, 2e-3, 0.05),
        "attn_in": f(2e-6, 2e-6, 2e-3, 2e-3, 0.05),
        "q": f(2e-5, 2e-5, 2e-2, 2e-2, 0.10),
        "kv_win": f(2e-5, 2e-5, 5e-2, 0.30, 0.10),
        "latent": f(2e-5, 2e-5, 0.15, 0.40, 0.10),
        "index_k": f(2e-5, 2e-5, 0.15, 0.40, 0.10),
        "attn_out": f(2e-5, 2e-5, 2e-2, 2e-2, 0.10),
        "ffn_in": f(2e-6, 2e-6, 2e-3, 2e-3, 0.05),
        "router_w": f(1e-5, 1e-5, 1e-2, 1e-2, 0.05),
        "ffn_out": f(2e-5, 2e-5, 2e-2, 2e-2, 0.10),
        "block_out": f(2e-6, 2e-6, 2e-3, 2e-3, 0.05),
        "pre_mix": f(2e-6, 2e-6, 2e-3, 2e-3, 0.05),
        "final_hidden": f(2e-6, 2e-6, 2e-3, 2e-3, 0.05),
        "logits": f(2e-6, 2e-6, 5e-3, 5e-3, 0.05),
        "topk": SetTol(1e-4, 0.05, 0.05),
        "router_idx": SetTol(1e-5, 0.5, 0.05),
        "cand_blocks": SetTol(1e-4, 0.25, 0.05),
    })
    tab("full", {k: v for k, v in TOLERANCES["layer"].items()})


_fill_tolerances()


def tolerance(mode: str, stage: str, scale: float = 1.0):
    t = TOLERANCES[mode][stage]
    if scale == 1.0 or isinstance(t, SetTol):
        return t
    return FloatTol(*(x * scale for x in dataclasses.astuple(t)[:4]), t.budget)


# ---------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class Sample:
    stage: str
    layer: int | None
    pos: int
    level: Level
    rms_rel: float = 0.0
    max_rel: float = 0.0
    max_abs: float = 0.0
    cos: float = 1.0
    note: str = ""
    extra: dict = dataclasses.field(default_factory=dict)

    @property
    def key(self) -> tuple:
        return exec_key(self.pos, self.layer, self.stage)

    def where(self) -> str:
        return f"position {self.pos}, " + (f"layer {self.layer}, " if self.layer is not None else "") + f"stage {self.stage}"


def exec_key(pos: int, layer: int | None, stage: str) -> tuple:
    """Order of execution inside one token step: embed, then layer by layer in STAGES order, then final_hidden, logits."""
    if stage == "embed":
        return (pos, -1, 0)
    if stage in ("final_hidden", "logits"):
        return (pos, 1 << 20, STAGE_ORDER[stage])
    return (pos, -1 if layer is None else layer, STAGE_ORDER.get(stage, 999))


def float_metrics(e, o) -> dict:
    """rms_rel / max_rel / cos of the engine array e against the oracle array o (flattened, float64)."""
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    o = np.asarray(o, dtype=np.float64).reshape(-1)
    out = {"nonfinite": False}
    if e.shape != o.shape:
        raise ValueError(f"shape {e.shape} vs {o.shape}")
    if o.size == 0:
        return {**out, "rms_rel": 0.0, "max_rel": 0.0, "max_abs": 0.0, "cos": 1.0}
    if not np.isfinite(e).all() and np.isfinite(o).all():
        return {**out, "nonfinite": True, "rms_rel": math.inf, "max_rel": math.inf, "max_abs": math.inf, "cos": float("nan")}
    fin = np.isfinite(o) & np.isfinite(e)
    e, o = e[fin], o[fin]
    if o.size == 0:
        return {**out, "rms_rel": 0.0, "max_rel": 0.0, "max_abs": 0.0, "cos": 1.0}
    d = e - o
    amax, rms_o = float(np.max(np.abs(o))), float(np.sqrt(np.mean(o * o)))
    rms_d, max_d = float(np.sqrt(np.mean(d * d))), float(np.max(np.abs(d)))
    tiny = 1e-30
    ne, no = float(np.linalg.norm(e)), float(np.linalg.norm(o))
    return {**out, "rms_rel": rms_d / max(rms_o, tiny), "max_rel": max_d / max(amax, tiny), "max_abs": max_d,
            "cos": float(e @ o / (ne * no)) if ne > 0 and no > 0 else (1.0 if ne == no else 0.0)}


def float_level(m: dict, tol: FloatTol) -> Level:
    if m["nonfinite"] or m["rms_rel"] > tol.hard_rms or m["max_rel"] > tol.hard_max:
        return Level.FAIL
    if m["rms_rel"] > tol.soft_rms or m["max_rel"] > tol.soft_max:
        return Level.FLIP
    return Level.OK


def selection(a) -> list:
    """An integer selection as a sorted list of its valid (>= 0) entries."""
    return sorted({int(x) for x in np.asarray(a).reshape(-1) if x >= 0})


def set_sample(stage, layer, pos, e, o, margin, scale, tol: SetTol) -> Sample:
    es, os_ = selection(e), selection(o)
    if len(np.asarray(e).reshape(-1)) and len(es) != len([x for x in np.asarray(e).reshape(-1) if x >= 0]):
        return Sample(stage, layer, pos, Level.FAIL, note="duplicate entries in the engine's selection")
    if es == os_:
        return Sample(stage, layer, pos, Level.OK)
    diff = len(set(es) ^ set(os_))
    swaps = max(diff // 2, 1)
    limit = max(1, math.ceil(tol.max_diff_frac * max(len(os_), 1)))
    gap = None if margin is None else float(margin)
    near = gap is not None and math.isfinite(gap) and scale is not None and gap <= tol.tie_rel * max(float(scale), 1e-30)
    detail = (f"selections differ in {swaps} entr{'y' if swaps == 1 else 'ies'} (engine only {sorted(set(es) - set(os_))[:6]}, "
              f"oracle only {sorted(set(os_) - set(es))[:6]}); oracle margin {gap if gap is not None else 'n/a'} of scale "
              f"{scale if scale is not None else 'n/a'}")
    if near and swaps <= limit:
        s = Sample(stage, layer, pos, Level.FLIP, note="near-tie: " + detail)
        s.extra["near_tie"] = True
        return s
    return Sample(stage, layer, pos, Level.FAIL, note=detail)


# ---------------------------------------------------------------------------------------------------------------
# sources: where the two sides' arrays come from
# ---------------------------------------------------------------------------------------------------------------


class Source:
    """Anything with get(stage, pos, layer) -> ndarray | None (a Trace, or the dict a replay produces)."""

    def get(self, stage, pos, layer=None):
        raise NotImplementedError


class DictSource(Source):
    def __init__(self):
        self.d: dict = {}

    def put(self, stage, pos, layer, arr):
        self.d[(stage, pos, layer)] = np.asarray(arr)

    def get(self, stage, pos, layer=None):
        return self.d.get((stage, pos, layer))


# ---------------------------------------------------------------------------------------------------------------
# the comparison
# ---------------------------------------------------------------------------------------------------------------


def _softmax_kl(o_logits, e_logits) -> float:
    o = np.asarray(o_logits, dtype=np.float64)
    e = np.asarray(e_logits, dtype=np.float64)
    lo = o - (o.max() + np.log(np.exp(o - o.max()).sum()))
    le = e - (e.max() + np.log(np.exp(e - e.max()).sum()))
    return float(np.sum(np.exp(lo) * (lo - le)))


def compare_one(stage: str, layer, pos: int, e, o, ref: Source, mode: str, tol_scale: float = 1.0) -> Sample:
    """One (stage, layer, position): engine array e against oracle array o."""
    st = STAGE[stage]
    if st.kind in ("float", "kvq", "logits", "weights"):
        tol = tolerance(mode, stage, tol_scale)
        try:
            m = float_metrics(e, o)
        except ValueError as ex:
            return Sample(stage, layer, pos, Level.FAIL, note=f"shape mismatch: {ex}")
        s = Sample(stage, layer, pos, float_level(m, tol), m["rms_rel"], m["max_rel"], m["max_abs"], m["cos"],
                   "NaN/Inf in the engine's values" if m["nonfinite"] else "")
        if stage == "logits":
            ea, oa = np.asarray(e, dtype=np.float64).reshape(-1), np.asarray(o, dtype=np.float64).reshape(-1)
            if np.isfinite(ea).all() and np.isfinite(oa).all():
                top2 = np.partition(oa, -2)[-2:]
                s.extra.update({"top1_match": int(np.argmax(ea)) == int(np.argmax(oa)), "gap": float(top2[1] - top2[0]),
                                "kl": _softmax_kl(oa, ea), "max_logit": float(np.max(np.abs(oa)))})
        return s
    if st.kind == "set":
        tol = TOLERANCES[mode][stage]
        mstage = {"router_idx": "router_margin", "topk": "index_margin", "cand_blocks": "cand_margin"}[stage]
        mg = ref.get(mstage, pos, layer)
        margin, scale = (None, None) if mg is None else (float(mg[0]), float(mg[1]))
        return set_sample(stage, layer, pos, e, o, margin, scale, tol)
    raise AssertionError(stage)


def _router_w_sample(eng: Source, ref: Source, pos, layer, mode, tol_scale) -> Sample | None:
    ew, ow = eng.get("router_w", pos, layer), ref.get("router_w", pos, layer)
    ei, oi = eng.get("router_idx", pos, layer), ref.get("router_idx", pos, layer)
    if ew is None or ow is None or ei is None or oi is None:
        return None
    ew, ow, ei, oi = (np.asarray(a).reshape(-1) for a in (ew, ow, ei, oi))
    if sorted(ei.tolist()) != sorted(oi.tolist()) or len(set(ei.tolist())) != len(ei):
        return Sample("router_w", layer, pos, Level.OK, note="skipped: the selections differ (router_idx reports it)")
    emap, omap = dict(zip(ei.tolist(), ew.tolist())), dict(zip(oi.tolist(), ow.tolist()))
    ids = sorted(omap)
    return compare_one("router_w", layer, pos, [emap[i] for i in ids], [omap[i] for i in ids], ref, mode, tol_scale)


def tie_taints(tie, pos: int, layer, stage: str) -> bool:
    """Is a sample at (pos, layer, stage) downstream of the near-tie event `tie` = (pos0, layer0, stage0)?  A flipped selection at layer L0 changes the
    stream from there on at that position, hence every later layer's caches at every later position (layer L0's own caches are written before the
    selection, so later positions at layers <= L0 stay clean)."""
    p0, l0, s0 = tie
    if layer is None:
        return pos >= p0 and (stage in ("final_hidden", "logits") or pos > p0)
    if pos == p0:
        return layer > l0 or (layer == l0 and STAGE_ORDER[stage] > STAGE_ORDER[s0])
    return pos > p0 and layer > l0


@dataclasses.dataclass
class Report:
    mode: str
    samples: list = dataclasses.field(default_factory=list)
    missing: list = dataclasses.field(default_factory=list)         # (stage, layer, pos): in the oracle, not in the engine
    notes: list = dataclasses.field(default_factory=list)
    tie_events: list = dataclasses.field(default_factory=list)      # (pos, layer, stage, text)
    tainted: int = 0                                                  # failures excused as downstream of a tie
    errors: list = dataclasses.field(default_factory=list)          # conditions that fail the run whatever the samples say (cannot replay, other tokens)
    strict: bool = False
    budgets: dict = dataclasses.field(default_factory=dict)         # stage -> (n_flip, n, budget)
    title: str = ""

    # -- verdict
    def stage_rows(self) -> dict:
        rows: dict = {}
        for s in self.samples:
            r = rows.setdefault(s.stage, {"n": 0, "ok": 0, "flip": 0, "fail": 0, "worst_rms": (0.0, None), "worst_max": (0.0, None), "min_cos": (1.0, None)})
            r["n"] += 1
            r[("ok", "flip", "fail")[int(s.level)]] += 1
            if s.rms_rel > r["worst_rms"][0]:
                r["worst_rms"] = (s.rms_rel, (s.layer, s.pos))
            if s.max_rel > r["worst_max"][0]:
                r["worst_max"] = (s.max_rel, (s.layer, s.pos))
            if s.cos < r["min_cos"][0]:
                r["min_cos"] = (s.cos, (s.layer, s.pos))
        return rows

    def budget_violations(self) -> list:
        out = []
        for stage, r in self.stage_rows().items():
            if stage in TOLERANCES[self.mode]:
                b = TOLERANCES[self.mode][stage].budget
                allowed = max(b * r["n"], 1.0 if r["n"] < 10 else 0.0)        # a handful of samples can't establish a rate: one FLIP is tolerated
                if r["flip"] > allowed:
                    out.append((stage, r["flip"], r["n"], b))
        return out

    def failures(self) -> list:
        return sorted((s for s in self.samples if s.level == Level.FAIL), key=lambda s: s.key)

    def first_failure(self) -> Sample | None:
        f = self.failures()
        return f[0] if f else None

    @property
    def ok(self) -> bool:
        if self.errors or self.failures() or self.budget_violations():
            return False
        return not (self.strict and self.missing_required)

    @property
    def missing_required(self) -> list:
        return [m for m in self.missing if STAGE[m[0]].required]

    # -- text
    def text(self, verbose: bool = False) -> str:
        L = []
        if self.title:
            L.append(self.title)
        rows = self.stage_rows()
        order = sorted(rows, key=lambda n: STAGE_ORDER.get(n, 999))
        L.append(f"{'stage':14s} {'n':>6s} {'ok':>6s} {'flip':>5s} {'FAIL':>5s}   {'worst rms_rel':>13s} {'(layer,pos)':>12s}   {'worst max_rel':>13s}   {'min cos':>9s}")
        for n in order:
            r = rows[n]
            wr, wm = r["worst_rms"], r["worst_max"]
            L.append(f"{n:14s} {r['n']:6d} {r['ok']:6d} {r['flip']:5d} {r['fail']:5d}   {wr[0]:13.3e} {str(wr[1]) if wr[1] else '-':>12s}   {wm[0]:13.3e}   {r['min_cos'][0]:9.7f}")
        lg = [s for s in self.samples if s.stage == "logits" and "top1_match" in s.extra]
        if lg:
            top1 = sum(1 for s in lg if s.extra["top1_match"])
            kls = [s.extra["kl"] for s in lg]
            L.append(f"logits: top-1 agreement {top1}/{len(lg)} ({100.0 * top1 / len(lg):.2f} %), KL(oracle||engine) mean {np.mean(kls):.3e} max {np.max(kls):.3e}"
                     f"; mismatches: " + (", ".join(f"p{s.pos} (oracle gap {s.extra['gap']:.3g})" for s in lg if not s.extra["top1_match"])[:300] or "none"))
        if self.tie_events:
            L.append(f"near-ties (a selection that differs where the oracle's margin is within the tie threshold): {len(self.tie_events)}")
            for (p, l, st, txt) in self.tie_events[:8]:
                L.append(f"    position {p}, layer {l}, {st}: {txt}")
            if self.tainted:
                L.append(f"  {self.tainted} later failure(s) lie downstream of a near-tie and are not counted as failures (re-run with `layers` to check them in isolation)")
        if self.missing:
            by: dict = {}
            for (st, l, p) in self.missing:
                by[st] = by.get(st, 0) + 1
            L.append(("MISSING in the engine trace (required stages: an error under --strict): " if self.strict else "missing in the engine trace (not compared): ")
                     + ", ".join(f"{k} x{v}" + ("" if STAGE[k].required else " (optional)") for k, v in sorted(by.items(), key=lambda kv: STAGE_ORDER.get(kv[0], 999))))
        for n in self.notes:
            L.append("note: " + n)
        for n in self.errors:
            L.append("ERROR: " + n)
        bv = self.budget_violations()
        for (st, nf, n, b) in bv:
            L.append(f"STAGE {st}: {nf} of {n} samples are above the soft tolerance (budget {100 * b:.0f} %): a systematic difference, not isolated flips")
        ff = self.first_failure()
        if ff is None and bv:
            st = min(bv, key=lambda v: STAGE_ORDER.get(v[0], 999))[0]
            ff = min((s for s in self.samples if s.stage == st and s.level == Level.FLIP), key=lambda s: s.key, default=None)
        if ff is not None:
            L.append(f"FIRST FAILURE: {ff.where()}: rms_rel {ff.rms_rel:.3e}, max_rel {ff.max_rel:.3e}, max_abs {ff.max_abs:.3e}, cos {ff.cos:.8f}" + (f"  [{ff.note}]" if ff.note else ""))
            if verbose:
                for s in self.failures()[1:21]:
                    L.append(f"   also: {s.where()}: rms_rel {s.rms_rel:.3e}, max_rel {s.max_rel:.3e}" + (f"  [{s.note}]" if s.note else ""))
        L.append("RESULT: " + ("PASS" if self.ok else "FAIL") + f"  ({len(self.samples)} samples, {sum(1 for s in self.samples if s.level == Level.FLIP)} flip-level, "
                 f"{len(self.failures())} failed, mode {self.mode})")
        return "\n".join(L)

    def to_json(self) -> dict:
        rows = self.stage_rows()
        return {"mode": self.mode, "ok": self.ok, "samples": len(self.samples), "stages": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in rows.items()},
                "first_failure": (lambda f: None if f is None else {"pos": f.pos, "layer": f.layer, "stage": f.stage, "rms_rel": f.rms_rel, "max_rel": f.max_rel, "note": f.note})(self.first_failure()),
                "budget_violations": [list(v) for v in self.budget_violations()], "tie_events": [list(t) for t in self.tie_events],
                "missing": len(self.missing), "notes": self.notes, "errors": self.errors}


def execution_keys(ref: Trace, positions: Iterable[int] | None, layers: Iterable[int] | None, stages: Iterable[str] | None) -> list:
    """Every (pos, layer, stage) the oracle trace has (oracle-only stages excluded), in execution order."""
    pset = None if positions is None else set(positions)
    lset = None if layers is None else set(layers)
    sset = None if stages is None else set(stages)
    keys = []
    for stage in ref.stages():
        st = STAGE.get(stage)
        if st is None or st.role == "oracle":
            continue
        if sset is not None and stage not in sset:
            continue
        for (layer, pos) in ref._by_stage[stage]:
            if pset is not None and pos not in pset:
                continue
            if lset is not None and layer is not None and layer not in lset:
                continue
            keys.append((pos, layer, stage))
    keys.sort(key=lambda k: exec_key(k[0], k[1], k[2]))
    return keys


def compare_sources(eng: Source, ref: Source, keys: list, mode: str, *, tol_scale: float = 1.0, strict: bool = False, title: str = "",
                    taint: bool = True) -> Report:
    """Compare the arrays of two sources over `keys` ((pos, layer, stage) in execution order)."""
    rep = Report(mode=mode, strict=strict, title=title)
    for (pos, layer, stage) in keys:
        o = ref.get(stage, pos, layer)
        if o is None:
            continue
        e = eng.get(stage, pos, layer)
        if e is None:
            rep.missing.append((stage, layer, pos))
            continue
        if stage == "router_w":
            s = _router_w_sample(eng, ref, pos, layer, mode, tol_scale)
            if s is None:
                continue
        else:
            s = compare_one(stage, layer, pos, e, o, ref, mode, tol_scale)
        if s.extra.get("near_tie"):
            rep.tie_events.append((pos, layer, stage, s.note))
        if s.level == Level.FAIL and taint and mode == "full" and any(tie_taints((p0, l0, st0), pos, layer, stage) for (p0, l0, st0, _) in rep.tie_events):
            s.level = Level.FLIP
            s.note = "downstream of a near-tie: " + s.note
            rep.tainted += 1
        rep.samples.append(s)
    return rep


def compare_traces(eng: Trace, ref: Trace, *, positions=None, layers=None, stages=None, tol_scale: float = 1.0, strict: bool = False) -> Report:
    """Whole-run comparison of an engine trace with an oracle trace (mode "full")."""
    pos = sorted(set(eng.positions()) & set(ref.positions())) if positions is None else [p for p in positions if p in set(eng.positions())]
    rep = compare_sources(eng, ref, execution_keys(ref, pos, layers, stages), "full", tol_scale=tol_scale, strict=strict,
                          title=f"engine {eng.path} ({eng.producer}) vs oracle {ref.path}")
    if set(ref.positions()) - set(eng.positions()) and positions is None:
        rep.notes.append(f"the engine trace lacks {len(set(ref.positions()) - set(eng.positions()))} of the oracle's {len(ref.positions())} positions: only the common ones were compared")
    et, rt = eng.tokens, ref.tokens
    if et and rt and et[: len(rt)] != rt[: len(et)]:
        rep.errors.append("the two traces were made on DIFFERENT TOKENS: every difference above is meaningless")
    eq, rq = eng.quant(), ref.quant()
    if eq != rq:
        rep.notes.append(f"quantisation flags differ: engine {TI.quant_to_dict(eq)} vs oracle {TI.quant_to_dict(rq)}")
    return rep


# ---------------------------------------------------------------------------------------------------------------
# the oracle
# ---------------------------------------------------------------------------------------------------------------

QUANT_PRESETS = {
    "int8-kv": dict(int8_act=True, window_kv=True, compressed_kv=True, index=True),     # the engine's default (DS1.md section 2)
    "int8": dict(int8_act=True),                                                          # engine with the three KV flags off
    "exact": {},
    "reference": dict(linear_act=True, window_kv=True, compressed_kv=True, index=True),
}


def quant_from_name(name: str):
    from ref.ds41.quant import QuantConfig
    if name not in QUANT_PRESETS:
        raise SystemExit(f"--quant {name!r}: one of {sorted(QUANT_PRESETS)}")
    return QuantConfig(**QUANT_PRESETS[name])


def load_oracle_model(gguf, quant, *, dtype="float32", max_seq_len: int = 4096, cache_bytes: int = 2 << 30):
    from ref.ds41.model import model_from_gguf
    paths = _gguf_paths(gguf)
    return model_from_gguf(paths, quant=quant, dtype=np.dtype(dtype), max_seq_len=max_seq_len, cache_bytes=cache_bytes)


def _gguf_paths(gguf) -> list:
    """Shard 1 (or a directory, or all shards) -> every shard path of the split file."""
    import gguf_io
    if isinstance(gguf, (str, pathlib.Path)):
        gguf = [gguf]
    out = [str(p) for p in gguf_io.expand_split(gguf)]
    # the DSpark sidecar lives next to the main shards but is a different model
    return [p for p in out if "dspark" not in pathlib.Path(p).name.lower()]


# ---------------------------------------------------------------------------------------------------------------
# logits statistics (the DS-1 gate: top-1 over a long run, no NaN/Inf)
# ---------------------------------------------------------------------------------------------------------------


def logits_report(eng_rows: np.ndarray, ref_rows: np.ndarray, *, near_tie_gap: float = 0.0) -> dict:
    """Statistics of engine logits [n, V] against oracle logits [n, V]: the numbers golden_compare.py uses for Gate Q (top-1 agreement, max |dlogit|,
    KL(ref||cand), NaN/Inf), plus top-1 agreement over the rows whose oracle top-2 gap exceeds `near_tie_gap` (a flip there is a legitimate near-tie)."""
    n = min(len(eng_rows), len(ref_rows))
    e, o = np.asarray(eng_rows[:n], dtype=np.float64), np.asarray(ref_rows[:n], dtype=np.float64)
    finite = np.isfinite(e).all(axis=1)
    top1 = e.argmax(axis=1) == o.argmax(axis=1)
    srt = np.sort(o, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    kl = np.array([_softmax_kl(o[i], e[i]) if finite[i] else np.nan for i in range(n)])
    far = gap > near_tie_gap
    return {"rows": n, "nonfinite_rows": int((~finite).sum()), "top1": float(top1.mean()) if n else 0.0,
            "top1_excluding_near_ties": float(top1[far].mean()) if far.any() else 1.0, "near_tie_rows": int((~far).sum()),
            "max_abs_dlogit": float(np.max(np.abs(e[finite] - o[finite]))) if finite.any() else math.inf,
            "kl_mean": float(np.nanmean(kl)) if finite.any() else math.nan, "kl_max": float(np.nanmax(kl)) if finite.any() else math.nan,
            "mismatch_positions": [int(i) for i in np.nonzero(~top1)[0]][:50]}


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------


def _ints(s: str | None):
    if s is None or s == "":
        return None
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _cmd_oracle(a) -> int:
    quant = quant_from_name(a.quant)
    m = load_oracle_model(a.gguf, quant, dtype=a.dtype, max_seq_len=a.max_seq_len)
    ids = _ints(a.tokens)
    n_prompt = len(ids)
    if a.max_new:
        ids = ids + TI.greedy_continue(m, ids, a.max_new)
    tr = TI.run_oracle_trace(m, ids, a.out, mode=a.mode, n_prompt=n_prompt, geometry="oracle")
    if a.logits_dump:
        TI.write_logits_dump(a.logits_dump, np.stack([tr.get("logits", p) for p in tr.positions()]))
    print(f"wrote {len(tr.positions())} positions to {a.out} ({len(ids)} tokens)")
    return 0


def _cmd_trace(a) -> int:
    eng = Trace(a.engine)
    if a.oracle:
        ref = Trace(a.oracle)
    else:
        import tempfile
        quant = TI.quant_from_dict(eng.meta.get("quant")) if a.quant == "from-engine" else quant_from_name(a.quant)
        m = load_oracle_model(a.gguf, quant, dtype=a.dtype, max_seq_len=max(len(eng.tokens) + 8, 64))
        tmp = tempfile.mkdtemp(prefix="ds1_oracle_")
        print(f"running the oracle on {len(eng.tokens)} tokens ({a.quant}, {a.dtype}) ...", file=sys.stderr)
        ref = TI.run_oracle_trace(m, eng.tokens, tmp, mode=a.mode, n_prompt=eng.n_prompt)
        print(f"oracle trace in {tmp}", file=sys.stderr)
    rep = compare_traces(eng, ref, positions=_ints(a.positions), layers=_ints(a.layers), stages=a.stages.split(",") if a.stages else None,
                         tol_scale=a.tol_scale, strict=a.strict)
    print(rep.text(verbose=a.verbose))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rep.to_json(), indent=1))
    return 0 if rep.ok else 1


def _cmd_logits(a) -> int:
    def rows(p):
        p = pathlib.Path(p)
        if p.is_dir():
            t = Trace(p)
            return np.stack([t.get("logits", q) for q in t.positions_of("logits")])
        if p.suffix == ".npy":
            return np.load(p)
        return TI.read_logits_dump(p)

    e, o = rows(a.engine), rows(a.oracle)
    if a.first is not None:
        e, o = e[a.first:], o[a.first:]
    r = logits_report(e, o, near_tie_gap=a.near_tie_gap)
    print(json.dumps(r, indent=1))
    ok = r["nonfinite_rows"] == 0 and r["top1_excluding_near_ties"] >= a.min_top1
    print(("PASS" if ok else "FAIL") + f": top-1 {100 * r['top1']:.2f} % ({100 * r['top1_excluding_near_ties']:.2f} % excluding {r['near_tie_rows']} near-tie rows), "
          f"{r['nonfinite_rows']} rows with NaN/Inf; gate: top-1 >= {100 * a.min_top1:.1f} %")
    return 0 if ok else 1


def _cmd_layers(a) -> int:
    import ds1_replay
    return ds1_replay.main_layers(a)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_oracle(p):
        p.add_argument("--quant", default="int8-kv", help=f"oracle QuantConfig: {sorted(QUANT_PRESETS)} (default int8-kv, the engine's default)")
        p.add_argument("--dtype", default="float32", choices=("float32", "float64"))
        p.add_argument("--mode", default="token_by_token", choices=("token_by_token", "prefill"))

    p = sub.add_parser("trace", help="whole-run comparison of an engine trace with an oracle trace (or with the oracle run on the spot)")
    p.add_argument("--engine", required=True)
    p.add_argument("--oracle", help="an oracle trace directory (ds1_compare.py oracle ...)")
    p.add_argument("--gguf", nargs="+", help="run the oracle on the engine's tokens from this GGUF (shard 1 is enough)")
    common_oracle(p)
    p.set_defaults(quant="from-engine")
    p.add_argument("--positions"); p.add_argument("--layers"); p.add_argument("--stages")
    p.add_argument("--tol-scale", type=float, default=1.0, help="multiply every float tolerance (a loosening you must justify)")
    p.add_argument("--strict", action="store_true", help="a required stage missing from the engine trace is an error")
    p.add_argument("--json"); p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=_cmd_trace)

    p = sub.add_parser("layers", help="layer-by-layer replay: the oracle runs ONLY layer L on the engine's input at position p")
    p.add_argument("--engine", required=True)
    p.add_argument("--gguf", nargs="+", required=True)
    p.add_argument("--layers", help="comma list / ranges of layers; `head` adds the final norm + output head; default all present")
    p.add_argument("--positions", help="positions to check (default: the last position and the first)")
    common_oracle(p)
    p.add_argument("--cache-gib", type=float, default=4.0, help="oracle weight cache (GiB of decoded float32 tensors)")
    p.add_argument("--tol-scale", type=float, default=1.0)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--no-force", action="store_true", help="do not re-run with the engine's selection when a selection differs (default: re-run)")
    p.add_argument("--json"); p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=_cmd_layers)

    p = sub.add_parser("oracle", help="run the oracle on tokens and write its trace directory")
    p.add_argument("--gguf", nargs="+", required=True)
    p.add_argument("--tokens", required=True, help="comma-separated token ids")
    p.add_argument("--max-new", type=int, default=0, help="then append N greedy tokens")
    p.add_argument("--out", required=True)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--logits-dump", help="also write the logits in golden_compare.py's dump format")
    common_oracle(p)
    p.set_defaults(fn=_cmd_oracle)

    p = sub.add_parser("logits", help="DS-1 gate statistics (top-1 agreement, KL, NaN/Inf) of an engine logits dump against the oracle's")
    p.add_argument("--engine", required=True, help="a logits dump (golden_compare format), .npy, or a trace directory")
    p.add_argument("--oracle", required=True)
    p.add_argument("--first", type=int, help="skip this many leading rows of both (e.g. the prompt)")
    p.add_argument("--min-top1", type=float, default=0.99)
    p.add_argument("--near-tie-gap", type=float, default=0.0, help="rows whose oracle top-2 logit gap is at most this are reported as near-ties and left out of the gate")
    p.set_defaults(fn=_cmd_logits)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        if a.cmd == "trace" and not a.oracle and not a.gguf:
            ap.error("trace: give --oracle DIR or --gguf FILE")
        return a.fn(a)
    except (TraceError, OSError, KeyError, AssertionError, ValueError) as e:
        print(f"ds1_compare: {type(e).__name__}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
