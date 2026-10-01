#!/usr/bin/env python3
"""ds1_noise.py - measure the numerical noise two CORRECT implementations of the DeepSeek-V4.1-Flash forward pass disagree by, and check that the
tolerances of ds1_compare.py cover it (derivation: docs/deepseek/DS1_VERIFY.md section 3).

    ds1_noise.py sums                         float32 summation noise (sequential / lane-strided / BLAS) against float64, by number of terms K
    ds1_noise.py flips                        quantiser flip statistics (int8 per 32, fp8, fp4): flips per vector and the error one flip causes, by K
    ds1_noise.py mini [--seeds 0,1,2 --tokens 64 --quant int8-kv,exact]
                                              the oracle in float32 against the oracle in float64 on mini models: stage-isolated ("layer": the float64
                                              oracle replays the float32 run's own inputs) and whole-run ("full"), per stage, against the tolerances
    ds1_noise.py all                          everything above; --json FILE writes the numbers

The float64 oracle plays the engine's role: it is the same mathematics with different rounding.  What it cannot reproduce is the engine's own
reduction order, which is why the tolerances are set from the naive sequential sum (the worst reasonable order, `sums`), not from the numpy BLAS order
the float32 oracle happens to use.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ds1_compare as C  # noqa: E402
from ref.ds41 import trace_io as TI  # noqa: E402
from ref.ds41.quant import (act_quant_fp8, fp4_quant_e4m3, fp4_quant_e8m0, quantize_int8_blocks)  # noqa: E402


# ---------------------------------------------------------------------------------------------------------------
# float32 summation noise
# ---------------------------------------------------------------------------------------------------------------


def _seq32(p):
    return np.cumsum(p, axis=-1, dtype=np.float32)[..., -1]


def _lanes32(p, lanes=32):
    n = p.shape[-1]
    pad = (-n) % lanes
    if pad:
        p = np.concatenate([p, np.zeros(p.shape[:-1] + (pad,), np.float32)], -1)
    parts = np.cumsum(p.reshape(*p.shape[:-1], -1, lanes), axis=-2, dtype=np.float32)[..., -1, :]
    while parts.shape[-1] > 1:
        parts = (parts[..., 0::2] + parts[..., 1::2]).astype(np.float32)
    return parts[..., 0]


def sum_noise(ks=(256, 1280, 2304, 4096, 5120, 8192, 20480), n_out: int = 256, seed: int = 0) -> list:
    """Relative error (rms over outputs / rms of the result, and max / max) of a float32 dot product of K terms against float64."""
    rng = np.random.default_rng(seed)
    rows = []
    for K in ks:
        x = rng.standard_normal(K).astype(np.float32)
        W = (rng.standard_normal((n_out, K)) / np.sqrt(K)).astype(np.float32)
        P = (W * x[None, :]).astype(np.float32)
        ref = (W.astype(np.float64) * x.astype(np.float64)[None, :]).sum(-1)
        row = {"K": K}
        for name, f in (("sequential", _seq32), ("lanes32", _lanes32), ("blas", lambda p: W @ x)):
            y = f(P).astype(np.float64)
            row[name] = (float(np.sqrt(((y - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean())), float(np.abs(y - ref).max() / np.abs(ref).max()))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# quantiser flips
# ---------------------------------------------------------------------------------------------------------------


def int8_flip_stats(K: int, eps: float, n: int = 2000, heavy: bool = False, seed: int = 0, n_out: int = 512) -> dict:
    """Perturb a [n, K] activation by relative noise eps (the float noise of the stage that produced it), quantise both with the engine's int8 rule,
    and report: flips per vector, P(vector has a flip), the relative change of a K -> n_out GEMV output (rms over outputs / rms) given a flip."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, K)).astype(np.float32)
    if heavy:
        x *= np.where(rng.random((n, K)) < 0.01, 10.0, 1.0).astype(np.float32)
    W = (rng.standard_normal((n_out, K)) / np.sqrt(K)).astype(np.float32)
    xe = (x * (1 + eps * rng.standard_normal(x.shape))).astype(np.float32)
    q1, d1 = quantize_int8_blocks(x)
    q2, d2 = quantize_int8_blocks(xe)
    flips = (q1 != q2).reshape(n, -1).sum(1)
    dq1 = (q1.astype(np.float32) * d1[..., None]).reshape(n, K).astype(np.float64)
    dq2 = (q2.astype(np.float32) * d2[..., None]).reshape(n, K).astype(np.float64)
    y1, y2 = dq1 @ W.T, dq2 @ W.T
    rel = np.sqrt(((y1 - y2) ** 2).mean(1)) / np.sqrt((y1 ** 2).mean(1))
    sel = flips > 0
    return {"K": K, "eps": eps, "heavy": heavy, "flips_per_vector": float(flips.mean()), "per_element": float(flips.mean() / K), "p_any": float(sel.mean()),
            "rms_given_flip_median": float(np.median(rel[sel])) if sel.any() else 0.0, "rms_p999": float(np.percentile(rel, 99.9)),
            "rms_times_sqrtK": float(np.median(rel[sel]) * math.sqrt(K)) if sel.any() else 0.0, "max_flips": int(flips.max())}


def kv_flip_stats(kind: str, K: int, eps: float, n: int = 20000, seed: int = 0) -> dict:
    fn = {"fp8": act_quant_fp8, "fp4e4m3": fp4_quant_e4m3, "fp4e8m0": fp4_quant_e8m0}[kind]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, K)).astype(np.float32)
    xe = (x * (1 + eps * rng.standard_normal(x.shape))).astype(np.float32)
    a, b = fn(x), fn(xe)
    fl = (a != b).reshape(n, -1).sum(1)
    d = (a - b).astype(np.float64)
    rms = np.sqrt((d ** 2).mean(1)) / np.sqrt((a.astype(np.float64) ** 2).mean(1))
    mx = np.abs(d).max(1) / np.abs(a).max(1)
    sel = fl > 0
    return {"kind": kind, "K": K, "eps": eps, "per_element": float(fl.mean() / K), "p_any": float(sel.mean()),
            "rms_given_flip_max": float(rms.max()), "rms_times_sqrtK": float(np.percentile(rms[sel], 99) * math.sqrt(K)) if sel.any() else 0.0,
            "max_rel_given_flip_max": float(mx.max())}


# ---------------------------------------------------------------------------------------------------------------
# the oracle against itself
# ---------------------------------------------------------------------------------------------------------------


def _percentiles(vals) -> tuple:
    v = np.asarray(vals, dtype=np.float64)
    return (float(np.median(v)), float(np.percentile(v, 99)), float(v.max())) if v.size else (0.0, 0.0, 0.0)


def measure_mini(seeds=(0,), n_tokens: int = 64, quants=("int8-kv",), workdir=None, progress=None) -> dict:
    """float32 oracle trace vs float64 oracle, per stage, for every (model seed, quant preset): -> {"layer": rows, "full": rows, "coverage": rows}.

    `layer`: the float64 model replays every (layer, position) of the float32 trace on that trace's own inputs (stage-isolated).  A sample above the soft
    tolerance is a FLIP (an int8 / fp8 / fp4 rounding decision that fell the other way, or a near-tie), above the hard tolerance a FAIL.
    `full`:  the two whole-run traces compared (errors accumulate; see ds1_compare.compare_sources)."""
    import dataclasses
    import make_mini_gguf as MM
    import ds1_replay as R
    work = pathlib.Path(workdir) if workdir else pathlib.Path(tempfile.mkdtemp(prefix="ds1_noise_"))
    out: dict = {"layer": [], "full": [], "meta": {"seeds": list(seeds), "tokens": n_tokens, "quants": list(quants)}}
    for seed in seeds:
        gdir = work / f"mini_s{seed}"
        info = MM.build_mini(gdir, dataclasses.replace(MM.MiniConfig(), seed=seed))
        shard1 = sorted(pathlib.Path(gdir).glob("*00001-of-*.gguf"))[0]
        ids = [int(t) for t in np.random.default_rng(1000 + seed).integers(3, MM.MiniConfig().vocab, n_tokens)]
        for qname in quants:
            quant = C.quant_from_name(qname)
            m32 = C.load_oracle_model(shard1, quant, dtype="float32", max_seq_len=n_tokens + 8)
            m64 = C.load_oracle_model(shard1, quant, dtype="float64", max_seq_len=n_tokens + 8)
            t0 = time.time()
            tr32 = TI.run_oracle_trace(m32, ids, work / f"t32_s{seed}_{qname}")
            tr64 = TI.run_oracle_trace(m64, ids, work / f"t64_s{seed}_{qname}")
            tols = C.Tolerances(C.Dims.from_summary(TI.model_summary(m64)), quant)
            rep = R.replay_compare(m64, tr32, list(range(m64.cfg.n_layers)), list(range(n_tokens)), include_head=True)
            full = C.compare_traces(tr32, tr64)
            if progress:
                progress(f"seed {seed} {qname}: traces + replay in {time.time() - t0:.0f} s")
            for mode, r in (("layer", rep), ("full", full)):
                by = collections.defaultdict(list)
                for s in r.samples:
                    by[s.stage].append(s)
                for st, ss in by.items():
                    clean = [s for s in ss if s.level == C.Level.OK and not s.extra.get("downstream")]
                    t = tols.stages.get(st)
                    out[mode].append({
                        "seed": seed, "quant": qname, "stage": st, "n": len(ss), "n_ok": len(clean), "n_flip": sum(s.level == C.Level.FLIP for s in ss),
                        "n_fail": sum(s.level == C.Level.FAIL for s in ss), "n_downstream": sum(bool(s.extra.get("downstream")) for s in ss),
                        "rms_ok": _percentiles([s.rms_rel for s in clean]), "max_ok": _percentiles([s.max_rel for s in clean]),
                        "rms_all": _percentiles([s.rms_rel for s in ss]), "max_all": _percentiles([s.max_rel for s in ss]),
                        "flip_rms": [s.rms_rel for s in ss if s.level == C.Level.FLIP and mode == "layer"],
                        "tol": None if not isinstance(t, C.FloatTol) else dataclasses.asdict(t),
                        "ok": bool(r.ok),
                    })
            out.setdefault("verdict", []).append({"seed": seed, "quant": qname, "layer_ok": bool(rep.ok), "full_ok": bool(full.ok),
                                                  "layer_text": rep.text().splitlines()[-1], "full_text": full.text().splitlines()[-1]})
    return out


def margin_table(res: dict) -> str:
    """Per stage: worst noise seen among the samples without a flip (ok), the soft tolerance and the factor between them; the worst flip against hard."""
    L = [f"{'stage':13s} {'n':>6s} {'noise max rms':>13s} {'noise max max':>13s} {'soft rms':>9s} {'margin':>7s}  {'flips':>5s} {'worst flip rms':>14s} {'hard rms':>9s} {'margin':>7s}"]
    agg: dict = {}
    for r in res["layer"]:
        a = agg.setdefault(r["stage"], {"n": 0, "rms": 0.0, "max": 0.0, "flips": [], "tol": r["tol"], "fail": 0})
        a["n"] += r["n"]
        a["rms"] = max(a["rms"], r["rms_ok"][2])
        a["max"] = max(a["max"], r["max_ok"][2])
        a["flips"] += r["flip_rms"]
        a["fail"] += r["n_fail"]
    for st in sorted(agg, key=lambda n: TI.STAGE_ORDER[n]):
        a = agg[st]
        t = a["tol"]
        if t is None:
            L.append(f"{st:13s} {a['n']:6d}  (a selection: exact; {len(a['flips'])} near-ties, {a['fail']} failures)")
            continue
        wf = max(a["flips"]) if a["flips"] else 0.0
        L.append(f"{st:13s} {a['n']:6d} {a['rms']:13.2e} {a['max']:13.2e} {t['soft_rms']:9.1e} {t['soft_rms'] / max(a['rms'], 1e-30):7.1f}  {len(a['flips']):5d} {wf:14.2e} {t['hard_rms']:9.1e} "
                 f"{(t['hard_rms'] / wf) if wf else float('inf'):7.1f}")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------------------------------------------


def _print_sums(rows):
    print("float32 dot product of K terms against float64 (rms / max of the relative error over 256 outputs)")
    print(f"{'K':>6s}  {'sequential':>21s}  {'lane-strided':>21s}  {'numpy BLAS':>21s}   model soft_rms = 4e-8 sqrt(K)")
    for r in rows:
        f = lambda t: f"{t[0]:9.2e} {t[1]:9.2e}"  # noqa: E731
        print(f"{r['K']:6d}  {f(r['sequential'])}  {f(r['lanes32'])}  {f(r['blas'])}   {max(C.SOFT_FLOOR, C.SUM_NOISE * math.sqrt(r['K'])):9.2e}")


def _print_flips(i8, kv):
    print("int8 per 32 + fp32 scale (the engine's quantiser): flips caused by relative noise eps on the input")
    print(f"{'K':>6s} {'acts':>6s} {'eps':>7s} {'flips/vec':>10s} {'per elem':>9s} {'P(any)':>7s} {'rms | flip':>11s} {'x sqrt(K)':>10s} {'p99.9 rms':>10s}  model: rate {C.FLIP_RATE:.1e}, rms {C.FLIP_RMS:.1e}/sqrt(K)")
    for r in i8:
        print(f"{r['K']:6d} {'heavy' if r['heavy'] else 'gauss':>6s} {r['eps']:7.0e} {r['flips_per_vector']:10.4f} {r['per_element']:9.2e} {r['p_any']:7.4f} {r['rms_given_flip_median']:11.2e} "
              f"{r['rms_times_sqrtK']:10.4f} {r['rms_p999']:10.2e}")
    print("\nfake-quantised cache rows: flips caused by relative noise eps")
    print(f"{'kind':8s} {'K':>5s} {'eps':>7s} {'per elem':>9s} {'P(any)':>7s} {'worst rms | flip':>16s} {'p99 x sqrt(K)':>14s} {'worst max_rel':>14s}")
    for r in kv:
        print(f"{r['kind']:8s} {r['K']:5d} {r['eps']:7.0e} {r['per_element']:9.2e} {r['p_any']:7.4f} {r['rms_given_flip_max']:16.2e} {r['rms_times_sqrtK']:14.3f} {r['max_rel_given_flip_max']:14.2e}")


def _flips_tables():
    i8 = [int8_flip_stats(K, eps, heavy=h) for K in (256, 1280, 2304, 5120, 8192) for h in (False, True) for eps in (3e-7, 1e-6)]
    kv = [kv_flip_stats(k, K, eps, n=8000) for (k, K) in (("fp8", 512), ("fp8", 64), ("fp4e4m3", 512), ("fp4e4m3", 64), ("fp4e8m0", 128), ("fp4e8m0", 32)) for eps in (3e-7, 1e-6)]
    return i8, kv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("sums", "flips", "mini", "all", "tolerances"))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--quant", default="int8-kv,int8,exact")
    ap.add_argument("--json")
    ap.add_argument("--real", action="store_true", help="tolerances: print the table for the real model's dimensions (default: the mini model's)")
    a = ap.parse_args(argv)
    res: dict = {}
    if a.what in ("sums", "all"):
        res["sums"] = sum_noise()
        _print_sums(res["sums"])
        print()
    if a.what in ("flips", "all"):
        i8, kv = _flips_tables()
        res["int8"], res["kv"] = i8, kv
        _print_flips(i8, kv)
        print()
    if a.what in ("mini", "all"):
        seeds = [int(x) for x in a.seeds.split(",") if x]
        res["mini"] = measure_mini(seeds, a.tokens, a.quant.split(","), progress=lambda s: print(s, file=sys.stderr))
        print("stage-isolated (layer mode): float64 oracle replaying the float32 oracle's own inputs, measured against the tolerances")
        for q in a.quant.split(","):
            sub = dict(res["mini"], layer=[r for r in res["mini"]["layer"] if r["quant"] == q])
            print(f"\n--- quant {q}, seeds {a.seeds}, {a.tokens} tokens")
            print(margin_table(sub))
        print()
        for v in res["mini"]["verdict"]:
            print(f"seed {v['seed']} {v['quant']}: layer: {v['layer_text']}\n                full : {v['full_text']}")
    if a.what == "tolerances":
        from ref.ds41.quant import QuantConfig
        import make_mini_gguf as MM
        mc = MM.MiniConfig()
        dims = C.REAL_DIMS if a.real else C.Dims(dim=mc.hidden, hc=mc.hc, n_heads=mc.n_head, head_dim=mc.head_dim, index_n_heads=mc.idx_heads, index_head_dim=mc.idx_dim,
                                                  q_lora=mc.q_lora, o_groups=mc.o_groups, o_lora=mc.o_lora, ff=mc.ff, top_k=mc.n_used)
        print(C.Tolerances(dims, QuantConfig(int8_act=True, window_kv=True, compressed_kv=True, index=True)).table())
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
