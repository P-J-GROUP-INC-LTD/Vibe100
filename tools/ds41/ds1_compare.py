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
The tolerances (class Tolerances below) are derived in docs/deepseek/DS1_VERIFY.md and re-measured by tools/ds41/ds1_noise.py.
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
# tolerances (derivation and measurements: docs/deepseek/DS1_VERIFY.md, re-measured by tools/ds41/ds1_noise.py)
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
    tie_rel: float         # a selection mismatch is a near-tie when margin <= (tie_rel + 10 * observed input error) * scale
    max_diff_frac: float   # ... and at most this fraction of the selection may differ (at least one entry)
    budget: float


@dataclasses.dataclass(frozen=True)
class Dims:
    """The model dimensions the tolerance model needs (defaults: the real model, include/strata/ds41/geometry.hpp)."""
    dim: int = 5120
    hc: int = 4
    n_heads: int = 64
    head_dim: int = 512
    index_n_heads: int = 32
    index_head_dim: int = 128
    q_lora: int = 1280
    o_groups: int = 8
    o_lora: int = 1024
    ff: int = 2304                 # moe_inter_dim
    top_k: int = 6                 # routed experts per token

    @classmethod
    def from_summary(cls, d: dict | None) -> "Dims":
        """From the `model` block of trace.json (trace_io.model_summary); keys that are absent keep the real model's value."""
        d = d or {}
        names = {"dim": "dim", "hc": "hc_mult", "n_heads": "n_heads", "head_dim": "head_dim", "index_n_heads": "index_n_heads",
                 "index_head_dim": "index_head_dim", "q_lora": "q_lora_rank", "o_groups": "o_groups", "o_lora": "o_lora_rank",
                 "ff": "moe_inter_dim", "top_k": "n_activated_experts"}
        return cls(**{k: int(d[v]) for k, v in names.items() if v in d})


REAL_DIMS = Dims()

# --- the model behind the numbers (all measured; DS1_VERIFY.md section 3) -----------------------------------------
SUM_NOISE = 4.0e-8        # soft_rms = SUM_NOISE * sqrt(K) for a float32 sum of K terms: 2.2x the measured naive left-to-right sequential error
                          # (1.8e-8 * sqrt(K), K = 256 .. 20480); the engine's lane-strided sums are 3-6x better, numpy BLAS 4-7x
SOFT_FLOOR = 2.0e-6       # nothing is held tighter than this (the measured float32-vs-float64 stage error on the mini model is <= 5e-7)
SOFT_MAX_OVER_RMS = 2.5   # max_rel (max |E-O| / max |O|) vs rms_rel of a float32 sum error: measured 1.2 - 1.6
HARD_NOFLIP = 50.0        # a stage without a quantiser inside has no excuse for a sample above 50 x soft (hard = 50 x soft)
FLIP_RATE = 2.2e-5        # P(an int8 rounding decision flips) per element at the engine's pre-quantiser noise (5e-7 relative); 4.4e-5 at 1e-6
FLIP_RMS = 1.9e-2         # one int8 flip in a K-element GEMV input moves the output by 1.9e-2 / sqrt(K) of its rms (K = 256 .. 8192, Gaussian)
FLIP_MULT = 8.0           # hard = FLIP_MULT single-flip errors (up to 3 flips and outlier blocks: measured p99.9 is 4x the median)
FLIP_FLOOR = 5.0e-3       # ... never below this (heavy-tailed activations: measured p99.9 up to 1.4e-3 at K = 2304 .. 8192)
KV_FLIP_RATE = 1.0e-5     # per element, fp8 / fp4 fake-quant of a cache row (measured 2.4e-6 at 3e-7 noise, 9e-6 at 1e-6)
KV_FLIP_RMS = {"fp8": 0.25, "fp4e4m3": 0.65, "fp4e8m0": 0.45}      # one flip moves the row by this / sqrt(K) of its rms
KV_FLIP_MAX = {"fp8": 0.10, "fp4e4m3": 0.25, "fp4e8m0": 0.15}      # ... and one element by this much of the row's max |.|
KV_MULT = 2.0
BUDGET_BASE = 0.02        # a stage may always have this fraction of FLIP-level samples
BUDGET_MULT = 2.0         # budget = BASE + MULT * P(at least one flip in a sample), capped
BUDGET_CAP = 0.9
CHAOS_CEILING_RMS = 4.0   # full mode, downstream of the first deviation: rms_rel above this is a failure (an expert swap or a flipped selection gives ~1.0 - 1.4:
                          # two unrelated vectors of the same size; 4 means a blow-up, NaN / Inf always fails)
TIE_FLOAT = 2.0e-5        # float32 noise of a router / indexer score relative to its scale, x ~10 safety (no quantiser flip involved)
TIE_INDEX_FP4 = 5.0e-2    # a flip in the fp4 fake-quant of the indexer's q moves its scores by up to ~1 - 4 % of their scale (analysis in DS1_VERIFY.md)


@dataclasses.dataclass
class StageModel:
    """What can make an engine value of this stage differ from the oracle's when both are handed the same inputs."""
    k_float: int = 0                          # longest float32 sum feeding the stage (sets the soft tolerance)
    int8_sites: tuple = ()                    # K of every int8 activation quantisation INSIDE the stage (its input is the engine's, bit-identical)
    kv: str | None = None                     # fake-quantised cache row written by the stage: "fp8" | "fp4e4m3" | "fp4e8m0"
    kv_k: int = 0


def stage_models(d: Dims, q) -> dict:
    int8 = bool(q.int8_act or q.linear_act)
    S = StageModel
    return {
        "embed": S(0),
        "engram_out": S(d.dim),
        "attn_in": S(d.dim),
        # wq_a (int8 in: exact) -> q_norm -> int8 q_lora (the site) -> wq_b -> RoPE
        "q": S(d.q_lora, (d.q_lora,) if int8 else ()),
        # wkv -> kv_norm -> RoPE -> fp8 fake-quant (window_kv)
        "kv_win": S(d.head_dim, (), "fp8" if q.window_kv else None, d.head_dim),
        # compressor wkv / wgate are BF16 weights with float activations: a float32 sum of K = dim terms (DS1.md section 2)
        "latent": S(d.dim, (), "fp4e4m3" if q.compressed_kv else None, d.head_dim),
        "index_k": S(d.dim, (), "fp4e8m0" if q.index else None, d.index_head_dim),
        # sparse attention (<= window + top-k terms), inverse RoPE, wo_a (float act, K = heads*head_dim/groups), int8 o_groups*o_lora (the site), wo_b
        "attn_out": S(max(d.n_heads * d.head_dim // max(d.o_groups, 1), 640), (d.o_groups * d.o_lora,) if int8 else ()),
        "ffn_in": S(d.dim),
        "router_w": S(d.dim),
        # (top_k routed + 1 shared) x int8 of silu(g)*u (the site, K = ff) -> w2
        "ffn_out": S(d.ff, (d.ff,) * (d.top_k + 1) if int8 else ()),
        "block_out": S(d.hc),
        "pre_mix": S(d.hc * d.dim),
        "final_hidden": S(d.dim),
        "logits": S(d.dim),
    }


def _budget(lam: float) -> float:
    return min(BUDGET_CAP, BUDGET_BASE + BUDGET_MULT * (1.0 - math.exp(-lam)))


@dataclasses.dataclass
class Tolerances:
    """Per-stage tolerances for one (dimensions, quantisation flags) pair, `scale` multiplies the float values (a loosening you must justify)."""
    dims: Dims
    quant: object
    scale: float = 1.0
    stages: dict = dataclasses.field(default_factory=dict)       # stage -> FloatTol | SetTol
    lam: dict = dataclasses.field(default_factory=dict)          # stage -> expected flips per sample (documentation)

    def __post_init__(self):
        d, q = self.dims, self.quant
        for stage, m in stage_models(d, q).items():
            soft = max(SOFT_FLOOR, SUM_NOISE * math.sqrt(m.k_float)) if m.k_float else 1e-12
            soft_max = soft * SOFT_MAX_OVER_RMS if m.k_float else 1e-12
            lam, hard, hard_max = 0.0, soft * HARD_NOFLIP if m.k_float else 1e-9, 0.0
            hard_max = hard * SOFT_MAX_OVER_RMS if m.k_float else 1e-9
            if m.int8_sites:
                lam += FLIP_RATE * sum(m.int8_sites)
                single = FLIP_RMS / math.sqrt(min(m.int8_sites))
                hard = max(hard, FLIP_FLOOR, FLIP_MULT * single * (4.0 if q.linear_act else 1.0))
                hard_max = max(hard_max, 2.0 * hard)
            if m.kv:
                lam += KV_FLIP_RATE * m.kv_k
                hard = max(hard, KV_MULT * KV_FLIP_RMS[m.kv] / math.sqrt(m.kv_k))
                hard_max = max(hard_max, KV_MULT * KV_FLIP_MAX[m.kv])
            budget = _budget(lam) if stage != "embed" else 0.0
            self.lam[stage] = lam
            self.stages[stage] = FloatTol(soft * self.scale, soft_max * self.scale, hard * self.scale, hard_max * self.scale, budget)
        tie_idx = TIE_INDEX_FP4 if q.index else TIE_FLOAT
        self.stages["router_idx"] = SetTol(TIE_FLOAT, 0.2, 0.02)
        self.stages["topk"] = SetTol(tie_idx, 0.10, _budget(0.0) + (0.03 if q.index else 0.0))
        self.stages["cand_blocks"] = SetTol(tie_idx, 0.25, _budget(0.0) + (0.03 if q.index else 0.0))

    def __getitem__(self, stage: str):
        return self.stages[stage]

    def __contains__(self, stage: str) -> bool:
        return stage in self.stages

    def table(self) -> str:
        L = [f"{'stage':13s} {'soft rms':>9s} {'soft max':>9s} {'hard rms':>9s} {'hard max':>9s} {'flips/sample':>12s} {'budget':>7s}"]
        for st in sorted(self.stages, key=lambda n: STAGE_ORDER[n]):
            t = self.stages[st]
            if isinstance(t, FloatTol):
                L.append(f"{st:13s} {t.soft_rms:9.1e} {t.soft_max:9.1e} {t.hard_rms:9.1e} {t.hard_max:9.1e} {self.lam.get(st, 0.0):12.3f} {t.budget:7.2f}")
            else:
                L.append(f"{st:13s} near-tie window {t.tie_rel:.0e} of the score scale, <= {t.max_diff_frac:.0%} of the selection, budget {t.budget:.2f}")
        return "\n".join(L)


def default_tolerances(quant=None, dims: Dims | None = None, scale: float = 1.0) -> Tolerances:
    from ref.ds41.quant import QuantConfig
    return Tolerances(dims or REAL_DIMS, quant if quant is not None else QuantConfig(int8_act=True, window_kv=True, compressed_kv=True, index=True), scale)


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


def set_sample(stage, layer, pos, e, o, margin, scale, tol: SetTol, extra_noise: float = 0.0) -> Sample:
    """An integer selection (router experts, top-k positions, candidate blocks): EXACT set equality, or a near-tie.

    A different selection is excused only when the ORACLE's recorded margin (gap between its k-th and (k+1)-th candidate) is within the noise the two sides'
    scores can differ by: a swap across the boundary needs |s_a - s_b| <= 2 noise, and the gap is at most |s_a - s_b|, so a gap above the noise window
    proves the difference is not rounding.  The window is `tol.tie_rel` (the stage's own float / quantiser noise) plus `extra_noise` (10 x the observed
    error of the stage's input, for whole-run comparisons).  The excuse is a FLIP-level sample (counted against the budget), never a plain pass."""
    ea = np.asarray(e).reshape(-1)
    es, os_ = selection(e), selection(o)
    if len(ea) and len(es) != len([x for x in ea if x >= 0]):
        return Sample(stage, layer, pos, Level.FAIL, note="duplicate entries in the engine's selection")
    if es == os_:
        return Sample(stage, layer, pos, Level.OK)
    diff = len(set(es) ^ set(os_))
    swaps = max(diff // 2, 1)
    limit = max(1, math.ceil(tol.max_diff_frac * max(len(os_), 1)))
    gap = None if margin is None else float(margin)
    window = (tol.tie_rel + extra_noise) * max(float(scale), 1e-30) if scale is not None else 0.0
    near = gap is not None and math.isfinite(gap) and scale is not None and gap <= window
    detail = (f"selections differ in {swaps} entr{'y' if swaps == 1 else 'ies'} (engine only {sorted(set(es) - set(os_))[:6]}, "
              f"oracle only {sorted(set(os_) - set(es))[:6]}); oracle margin {gap if gap is not None else 'n/a'} of scale "
              f"{scale if scale is not None else 'n/a'} (tie window {window:.3g})")
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


# the stage whose error moves a selection's scores (for the whole-run near-tie window)
SELECTION_INPUTS = {"router_idx": ("ffn_in",), "topk": ("attn_in", "q"), "cand_blocks": ("attn_in", "q")}
SELECTION_MARGIN = {"router_idx": "router_margin", "topk": "index_margin", "cand_blocks": "cand_margin"}


def compare_one(stage: str, layer, pos: int, e, o, ref: Source, tols: Tolerances, extra_noise: float = 0.0) -> Sample:
    """One (stage, layer, position): engine array e against oracle array o, at the soft / hard levels of `tols`."""
    st = STAGE[stage]
    if st.kind in ("float", "kvq", "logits", "weights"):
        tol = tols[stage]
        try:
            m = float_metrics(e, o)
        except ValueError as ex:
            return Sample(stage, layer, pos, Level.FAIL, note=f"shape mismatch: {ex}")
        s = Sample(stage, layer, pos, float_level(m, tol), m["rms_rel"], m["max_rel"], m["max_abs"], m["cos"],
                   "NaN/Inf in the engine's values" if m["nonfinite"] else "")
        if stage == "logits":
            ea, oa = np.asarray(e, dtype=np.float64).reshape(-1), np.asarray(o, dtype=np.float64).reshape(-1)
            if np.isfinite(ea).all() and np.isfinite(oa).all() and oa.size > 1:
                top2 = np.partition(oa, -2)[-2:]
                s.extra.update({"top1_match": int(np.argmax(ea)) == int(np.argmax(oa)), "gap": float(top2[1] - top2[0]),
                                "kl": _softmax_kl(oa, ea), "max_logit": float(np.max(np.abs(oa)))})
        return s
    if st.kind == "set":
        mg = ref.get(SELECTION_MARGIN[stage], pos, layer)
        margin, scale = (None, None) if mg is None else (float(mg[0]), float(mg[1]))
        return set_sample(stage, layer, pos, e, o, margin, scale, tols[stage], extra_noise)
    raise AssertionError(stage)


def _router_w_sample(eng: Source, ref: Source, pos, layer, tols: Tolerances) -> Sample | None:
    ew, ow = eng.get("router_w", pos, layer), ref.get("router_w", pos, layer)
    ei, oi = eng.get("router_idx", pos, layer), ref.get("router_idx", pos, layer)
    if ew is None or ow is None or ei is None or oi is None:
        return None
    ew, ow, ei, oi = (np.asarray(a).reshape(-1) for a in (ew, ow, ei, oi))
    if sorted(ei.tolist()) != sorted(oi.tolist()) or len(set(ei.tolist())) != len(ei):
        return Sample("router_w", layer, pos, Level.OK, note="skipped: the selections differ (router_idx reports it)")
    emap, omap = dict(zip(ei.tolist(), ew.tolist())), dict(zip(oi.tolist(), ow.tolist()))
    ids = sorted(omap)
    return compare_one("router_w", layer, pos, [emap[i] for i in ids], [omap[i] for i in ids], ref, tols)


# stages whose error persists in the cache: later positions of the same layer read them
CACHE_FEEDERS = ("engram_out", "attn_in", "kv_win", "latent", "index_k")


def downstream_of(onset, pos: int, layer, stage: str) -> bool:
    """Can a sample at (pos, layer, stage) carry the error of the deviation `onset` = (pos0, layer0, stage0)?  Same position: everything later in
    execution order.  Later positions: every layer above layer0 (its caches at pos0 are contaminated), layer0 itself only when the deviation sits in
    a stage that feeds the layer's own cache rows; layers below layer0 stay clean (their rows at pos0 were written before the deviation)."""
    p0, l0, s0 = onset
    if stage == "embed":
        return False
    if layer is None:                                      # final_hidden, logits: after every layer of the same position
        if l0 is None:
            return pos == p0 and STAGE_ORDER[stage] > STAGE_ORDER[s0]
        return pos >= p0
    if l0 is None:
        return False
    if pos == p0:
        return layer > l0 or (layer == l0 and STAGE_ORDER[stage] > STAGE_ORDER[s0])
    return pos > p0 and (layer > l0 or (layer == l0 and s0 in CACHE_FEEDERS))


def _dominated(a, b) -> bool:
    """Is onset b implied by onset a (everything b taints, a taints)?"""
    (pa, la, sa), (pb, lb, sb) = a, b
    if la is None or lb is None:
        return False
    if pa == pb:
        return la < lb or (la == lb and STAGE_ORDER[sa] <= STAGE_ORDER[sb])
    return pa < pb and (la < lb or (la == lb and sa in CACHE_FEEDERS))


@dataclasses.dataclass
class Report:
    mode: str
    tols: Tolerances | None = None
    samples: list = dataclasses.field(default_factory=list)
    missing: list = dataclasses.field(default_factory=list)         # (stage, layer, pos): in the oracle, not in the engine
    notes: list = dataclasses.field(default_factory=list)
    tie_events: list = dataclasses.field(default_factory=list)      # (pos, layer, stage, text)
    onsets: list = dataclasses.field(default_factory=list)          # full mode: (pos, layer, stage, text) the deviations above float noise that taint what follows
    downstream: int = 0                                               # full mode: samples judged only against the chaos ceiling
    errors: list = dataclasses.field(default_factory=list)          # conditions that fail the run whatever the samples say (cannot replay, other tokens)
    strict: bool = False
    title: str = ""

    # -- verdict
    def stage_rows(self) -> dict:
        rows: dict = {}
        for s in self.samples:
            r = rows.setdefault(s.stage, {"n": 0, "ok": 0, "flip": 0, "fail": 0, "down": 0, "worst_rms": (0.0, None), "worst_max": (0.0, None), "min_cos": (1.0, None)})
            r["n"] += 1
            r[("ok", "flip", "fail")[int(s.level)]] += 1
            r["down"] += 1 if s.extra.get("downstream") else 0
            if s.rms_rel > r["worst_rms"][0]:
                r["worst_rms"] = (s.rms_rel, (s.layer, s.pos))
            if s.max_rel > r["worst_max"][0]:
                r["worst_max"] = (s.max_rel, (s.layer, s.pos))
            if s.cos < r["min_cos"][0]:
                r["min_cos"] = (s.cos, (s.layer, s.pos))
        return rows

    def budget_violations(self) -> list:
        """Stages whose FLIP-level samples are more than the quantiser-flip model allows: a systematic difference, not isolated flips."""
        out = []
        if self.tols is None:
            return out
        for stage, r in self.stage_rows().items():
            if stage in self.tols:
                b = self.tols[stage].budget
                n = r["n"] - r["down"]
                allowed = max(b * n, 1.0 if n < 10 else 0.0)        # a handful of samples can't establish a rate: one FLIP is tolerated
                if r["flip"] > allowed:
                    out.append((stage, r["flip"], n, b))
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
            L.append(f"near-ties (a selection that differs where the oracle's margin is within the tie window): {len(self.tie_events)}")
            for (p, l, st, txt) in self.tie_events[:8]:
                L.append(f"    position {p}, layer {l}, {st}: {txt}")
        if self.mode == "full":
            n = len(self.samples)
            if self.onsets:
                p, l, st, txt = self.onsets[0]
                L.append(f"first deviation above float noise: position {p}, " + (f"layer {l}, " if l is not None else "") + f"stage {st}: {txt}")
                L.append(f"  {self.downstream} of {n} samples lie downstream of a deviation: an int8 / fp8 / fp4 rounding flip or a near-tie is amplified by the quantisers of "
                         f"every later layer, so they are checked against the chaos ceiling only; use `layers` (stage-isolated replay) to check them strictly")
            L.append(f"strictly checked: {n - self.downstream} of {n} samples ({100.0 * (n - self.downstream) / max(n, 1):.0f} %)")
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
            ff = min((s for s in self.samples if s.stage == st and s.level == Level.FLIP and not s.extra.get("downstream")), key=lambda s: s.key, default=None)
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
        ff = self.first_failure()
        return {"mode": self.mode, "ok": self.ok, "samples": len(self.samples), "downstream": self.downstream,
                "stages": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in rows.items()},
                "first_failure": None if ff is None else {"pos": ff.pos, "layer": ff.layer, "stage": ff.stage, "rms_rel": ff.rms_rel, "max_rel": ff.max_rel, "note": ff.note},
                "budget_violations": [list(v) for v in self.budget_violations()], "tie_events": [list(t) for t in self.tie_events],
                "onsets": [list(o) for o in self.onsets[:20]], "missing": len(self.missing), "notes": self.notes, "errors": self.errors}


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


def compare_sources(eng: Source, ref: Source, keys: list, mode: str, *, tols: Tolerances | None = None, strict: bool = False, title: str = "",
                    chaos: bool = True) -> Report:
    """Compare the arrays of two sources over `keys` ((pos, layer, stage) in execution order).

    mode "layer": every sample at the stage tolerances (the oracle was handed the engine's inputs: no error accumulates).
    mode "full":  the oracle ran its own trajectory.  The first sample above the float noise (soft tolerance) is the ONSET of a deviation; everything
                  downstream of it (`downstream_of`) is judged against the chaos ceiling only (finite, rms_rel <= CHAOS_CEILING_RMS), because the
                  int8 / fp8 / fp4 quantisers turn a 1e-4 input difference into 1e-3 .. 1e-2 differences in every later layer (DS1_VERIFY.md section 4);
                  everything before the first onset, and the layers below a deviation at later positions, stay strictly checked."""
    tols = tols or default_tolerances()
    rep = Report(mode=mode, tols=tols, strict=strict, title=title)
    seen: dict = {}
    onsets: list = []
    for (pos, layer, stage) in keys:
        o = ref.get(stage, pos, layer)
        if o is None:
            continue
        e = eng.get(stage, pos, layer)
        if e is None:
            rep.missing.append((stage, layer, pos))
            continue
        if stage == "router_w":
            s = _router_w_sample(eng, ref, pos, layer, tols)
            if s is None:
                continue
        else:
            extra = 0.0
            if mode == "full" and stage in SELECTION_INPUTS:
                extra = 10.0 * max([seen[(i, layer, pos)].max_rel for i in SELECTION_INPUTS[stage] if (i, layer, pos) in seen] or [0.0])
            s = compare_one(stage, layer, pos, e, o, ref, tols, extra)
        seen[(stage, layer, pos)] = s
        if s.extra.get("near_tie"):
            rep.tie_events.append((pos, layer, stage, s.note))
        if mode == "full" and chaos:
            if any(downstream_of(on, pos, layer, stage) for on in onsets):
                s.extra["downstream"] = True
                rep.downstream += 1
                nonfinite = s.note.startswith("NaN/Inf") or not math.isfinite(s.rms_rel)
                if STAGE[stage].kind in ("float", "kvq", "logits", "weights"):
                    s.level = Level.FAIL if (nonfinite or s.rms_rel > CHAOS_CEILING_RMS or s.note.startswith("shape")) else Level.OK
                    if s.level == Level.FAIL:
                        s.note = (s.note + "; " if s.note else "") + f"beyond the chaos ceiling ({CHAOS_CEILING_RMS})"
                else:                                                   # a selection: judged by nothing but the near-tie bookkeeping above
                    if s.level == Level.FAIL:
                        s.note = "downstream of a deviation: " + s.note
                    s.level = Level.OK
                rep.samples.append(s)
                continue
            if s.level >= Level.FLIP:                                  # the first deviation above float noise
                new = (pos, layer, stage)
                if not any(_dominated(on, new) for on in onsets):
                    onsets = [on for on in onsets if not _dominated(new, on)] + [new]
                    txt = (f"rms_rel {s.rms_rel:.2e}, max_rel {s.max_rel:.2e}" + (f"  [{s.note}]" if s.note else "")
                           + ("" if s.level == Level.FAIL else "  (flip level: expected at this rate, not a defect by itself)"))
                    rep.onsets.append((pos, layer, stage, txt))
                if s.level == Level.FLIP:
                    s.level = Level.OK                                 # not counted against the stage budget in whole-run mode: the onset is reported instead
        rep.samples.append(s)
    return rep


def tolerances_of(eng: Trace, ref: Trace | None = None, scale: float = 1.0) -> Tolerances:
    """The tolerance model for a trace pair: dimensions from the oracle's trace.json `model` block (else the engine's, else the real model's),
    quantisation flags from the engine's."""
    meta = (ref.meta.get("model") if ref is not None else None) or eng.meta.get("model")
    return Tolerances(Dims.from_summary(meta), eng.quant(), scale)


def compare_traces(eng: Trace, ref: Trace, *, positions=None, layers=None, stages=None, tol_scale: float = 1.0, strict: bool = False) -> Report:
    """Whole-run comparison of an engine trace with an oracle trace (mode "full")."""
    pos = sorted(set(eng.positions()) & set(ref.positions())) if positions is None else [p for p in positions if p in set(eng.positions())]
    rep = compare_sources(eng, ref, execution_keys(ref, pos, layers, stages), "full", tols=tolerances_of(eng, ref, tol_scale), strict=strict,
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
