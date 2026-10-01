"""Tests for tools/ds41/ds1_compare.py: tolerances, metrics, near-ties, injected errors named by stage / layer / position, whole-run vs stage-isolated."""
import contextlib
import io
import json
import math
import pathlib
import tempfile
import unittest

import numpy as np

import ds1_compare as C
import ds1_replay as R
import ds1_testlib as T
from ref.ds41 import trace_io as TI
from ref.ds41.quant import QuantConfig

MINI = C.Dims(dim=256, hc=4, n_heads=4, head_dim=64, index_n_heads=4, index_head_dim=32, q_lora=64, o_groups=2, o_lora=32, ff=256, top_k=2)
ENGINE_QUANT = QuantConfig(int8_act=True, window_kv=True, compressed_kv=True, index=True)


class ToleranceModel(unittest.TestCase):
    def test_every_compared_stage_has_a_tolerance(self):
        tols = C.Tolerances(MINI, ENGINE_QUANT)
        for name, st in TI.STAGE.items():
            if st.role != "oracle":
                self.assertIn(name, tols, name)
        for name in ("router_margin", "index_margin", "cand_margin"):
            self.assertNotIn(name, tols)

    def test_values_follow_the_documented_formulas(self):
        mini, real = C.Tolerances(MINI, ENGINE_QUANT), C.Tolerances(C.REAL_DIMS, ENGINE_QUANT)
        # float sums: soft = max(2e-6, 4e-8 sqrt(K))
        self.assertAlmostEqual(mini["attn_in"].soft_rms, 2.0e-6)
        self.assertAlmostEqual(real["attn_in"].soft_rms, 4.0e-8 * math.sqrt(5120), delta=1e-12)
        self.assertAlmostEqual(real["pre_mix"].soft_rms, 4.0e-8 * math.sqrt(4 * 5120), delta=1e-12)
        self.assertAlmostEqual(real["attn_in"].soft_max / real["attn_in"].soft_rms, C.SOFT_MAX_OVER_RMS)
        # a stage without a quantiser inside: hard = 50 x soft, tiny budget
        self.assertAlmostEqual(real["ffn_in"].hard_rms, 50 * real["ffn_in"].soft_rms)
        self.assertEqual(real["ffn_in"].budget, C.BUDGET_BASE)
        # a stage with int8 sites: hard from the single-flip error, budget from the flip rate
        self.assertAlmostEqual(mini["ffn_out"].hard_rms, C.FLIP_MULT * C.FLIP_RMS / math.sqrt(256), places=9)
        self.assertGreater(real["ffn_out"].budget, mini["ffn_out"].budget)               # more sites of more elements: more flips per sample
        self.assertAlmostEqual(real.lam["ffn_out"], C.FLIP_RATE * 7 * 2304, places=9)
        self.assertGreaterEqual(real["ffn_out"].hard_rms, C.FLIP_FLOOR)
        self.assertEqual(real["embed"].budget, 0.0)
        self.assertLessEqual(max(t.budget for t in real.stages.values()), C.BUDGET_CAP)

    def test_flags_decide_where_a_flip_can_happen(self):
        exact = C.Tolerances(MINI, QuantConfig.exact())
        for st in ("q", "kv_win", "latent", "index_k", "attn_out", "ffn_out"):
            self.assertAlmostEqual(exact[st].hard_rms, 50 * exact[st].soft_rms, msg=st)
            self.assertEqual(exact.lam[st], 0.0)
        no_kv = C.Tolerances(MINI, QuantConfig(int8_act=True))                           # the engine with the three KV flags off
        self.assertEqual(no_kv.lam["kv_win"], 0.0)
        self.assertGreater(no_kv.lam["ffn_out"], 0.0)
        self.assertLess(no_kv["kv_win"].hard_rms, C.Tolerances(MINI, ENGINE_QUANT)["kv_win"].hard_rms)
        self.assertGreater(C.Tolerances(MINI, ENGINE_QUANT)["topk"].tie_rel, C.Tolerances(MINI, QuantConfig(int8_act=True))["topk"].tie_rel)

    def test_scale_loosens_float_values_only(self):
        a, b = C.Tolerances(MINI, ENGINE_QUANT), C.Tolerances(MINI, ENGINE_QUANT, scale=3.0)
        self.assertAlmostEqual(b["q"].soft_rms, 3 * a["q"].soft_rms)
        self.assertAlmostEqual(b["q"].hard_rms, 3 * a["q"].hard_rms)
        self.assertEqual(b["q"].budget, a["q"].budget)
        self.assertEqual(b["topk"], a["topk"])

    def test_dims_from_summary(self):
        d = C.Dims.from_summary({"dim": 256, "hc_mult": 4, "moe_inter_dim": 128, "n_activated_experts": 2})
        self.assertEqual((d.dim, d.ff, d.top_k, d.n_heads), (256, 128, 2, C.REAL_DIMS.n_heads))     # absent keys keep the real model's value
        self.assertEqual(C.Dims.from_summary(None), C.REAL_DIMS)
        self.assertEqual(C.Dims.from_summary(T.oracle_trace().meta["model"]), MINI)

    def test_mini_noise_is_covered_with_margin(self):
        """The float32 oracle against the float64 one, stage-isolated, without quantisers: every stage's error is a factor >= 2 below its soft tolerance."""
        tr32 = T.oracle_trace("exact", "float32", 20)
        m64 = T.model("exact", "float64")
        rep = R.replay_compare(m64, tr32, list(range(8)), list(range(20)), include_head=True)
        self.assertTrue(rep.ok, rep.text())
        tols = C.Tolerances(MINI, QuantConfig.exact())
        for st, r in rep.stage_rows().items():
            if st in tols and isinstance(tols[st], C.FloatTol) and st != "embed":
                self.assertLess(2 * r["worst_rms"][0], tols[st].soft_rms, f"{st}: {r['worst_rms']} vs soft {tols[st].soft_rms}")
                self.assertEqual(r["flip"], 0, st)


class Metrics(unittest.TestCase):
    def test_float_metrics(self):
        o = np.linspace(-1, 1, 100)
        m = C.float_metrics(o, o)
        self.assertEqual((m["rms_rel"], m["max_rel"], m["cos"]), (0.0, 0.0, 1.0))
        m = C.float_metrics(o * 1.01, o)
        self.assertAlmostEqual(m["rms_rel"], 0.01, places=6)
        self.assertAlmostEqual(m["max_rel"], 0.01, places=6)
        self.assertAlmostEqual(m["cos"], 1.0, places=9)
        e = o.copy()
        e[3] = np.nan
        m = C.float_metrics(e, o)
        self.assertTrue(m["nonfinite"] and math.isinf(m["rms_rel"]))
        with self.assertRaises(ValueError):
            C.float_metrics(o[:5], o)
        self.assertEqual(C.float_metrics(np.zeros(4), np.zeros(4))["cos"], 1.0)
        # a NaN the oracle has too is not the engine's fault
        o2 = o.copy()
        o2[3] = np.nan
        self.assertFalse(C.float_metrics(e, o2)["nonfinite"])

    def test_levels(self):
        t = C.FloatTol(1e-6, 2e-6, 1e-3, 2e-3, 0.1)
        self.assertEqual(C.float_level({"nonfinite": False, "rms_rel": 5e-7, "max_rel": 1e-6}, t), C.Level.OK)
        self.assertEqual(C.float_level({"nonfinite": False, "rms_rel": 5e-7, "max_rel": 3e-6}, t), C.Level.FLIP)
        self.assertEqual(C.float_level({"nonfinite": False, "rms_rel": 5e-4, "max_rel": 1e-3}, t), C.Level.FLIP)
        self.assertEqual(C.float_level({"nonfinite": False, "rms_rel": 2e-3, "max_rel": 1e-3}, t), C.Level.FAIL)
        self.assertEqual(C.float_level({"nonfinite": False, "rms_rel": 5e-4, "max_rel": 3e-3}, t), C.Level.FAIL)
        self.assertEqual(C.float_level({"nonfinite": True, "rms_rel": math.inf, "max_rel": math.inf}, t), C.Level.FAIL)

    def test_selection_sets(self):
        tol = C.SetTol(1e-5, 0.2, 0.02)
        S = lambda e, o, margin=None, scale=1.0, extra=0.0: C.set_sample("router_idx", 1, 2, e, o, margin, scale, tol, extra)   # noqa: E731
        self.assertEqual(S([1, 2, 3], [3, 1, 2]).level, C.Level.OK)                            # order is irrelevant
        self.assertEqual(S([1, 2, 3, -1, -1], [3, 2, 1]).level, C.Level.OK)                    # padding entries are ignored
        self.assertEqual(S([1, 2, 2], [1, 2, 3]).level, C.Level.FAIL)                          # duplicates
        s = S([1, 2, 4], [1, 2, 3], margin=0.5)
        self.assertEqual(s.level, C.Level.FAIL)                                                # a different expert, far from a tie
        self.assertIn("margin 0.5", s.note)
        s = S([1, 2, 4], [1, 2, 3], margin=1e-9)
        self.assertEqual((s.level, s.extra.get("near_tie")), (C.Level.FLIP, True))              # a near-tie is a FLIP, never a pass
        self.assertEqual(S([1, 2, 4], [1, 2, 3], margin=1e-4).level, C.Level.FAIL)             # 1e-4 > the 1e-5 window ...
        self.assertEqual(S([1, 2, 4], [1, 2, 3], margin=1e-4, extra=1e-3).level, C.Level.FLIP)  # ... unless the observed input error widens it
        self.assertEqual(S([1, 2, 4], [1, 2, 3], margin=None).level, C.Level.FAIL)             # no margin recorded: not excusable
        self.assertEqual(S([1, 2, 4], [1, 2, 3], margin=math.inf).level, C.Level.FAIL)
        many = C.set_sample("topk", 1, 2, list(range(20, 30)), list(range(10)), 1e-9, 1.0, C.SetTol(1e-5, 0.1, 0.05))
        self.assertEqual(many.level, C.Level.FAIL)                                             # too many entries differ for one rounding decision

    def test_taint_rule(self):
        d = C.downstream_of
        on = (5, 3, "ffn_out")
        self.assertTrue(d(on, 5, 3, "block_out") and d(on, 5, 4, "attn_in") and d(on, 5, None, "logits") and d(on, 5, None, "final_hidden"))
        self.assertFalse(d(on, 5, 3, "attn_out") or d(on, 5, 2, "block_out") or d(on, 4, 7, "ffn_out"))
        self.assertTrue(d(on, 6, 4, "attn_in") and d(on, 9, None, "logits"))
        self.assertFalse(d(on, 6, 3, "kv_win") or d(on, 6, 2, "attn_out") or d(on, 6, None, "embed") or d(on, 4, None, "logits"))
        # a deviation that feeds the layer's own cache rows taints that layer at later positions
        self.assertTrue(d((5, 3, "attn_in"), 6, 3, "kv_win") and d((5, 3, "kv_win"), 6, 3, "attn_out"))
        self.assertFalse(d((5, 3, "q"), 6, 3, "kv_win"))
        # a deviation in the head
        self.assertTrue(d((5, None, "final_hidden"), 5, None, "logits"))
        self.assertFalse(d((5, None, "final_hidden"), 6, None, "logits") or d((5, None, "final_hidden"), 5, 3, "attn_in"))


class Runs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        cls.ref = T.oracle_trace("int8-kv", "float32", cls.n)

    def test_identical_traces_pass_every_sample(self):
        rep = C.compare_traces(self.ref, self.ref, strict=True)
        self.assertTrue(rep.ok, rep.text())
        self.assertEqual(rep.downstream, 0)
        self.assertFalse(rep.missing or rep.onsets or rep.tie_events)
        self.assertTrue(all(s.level == C.Level.OK and s.rms_rel == 0.0 for s in rep.samples))
        rows = rep.stage_rows()
        self.assertEqual({s for s in rows}, {n for n, st in TI.STAGE.items() if st.role != "oracle"})
        self.assertEqual(rows["attn_in"]["n"], self.n * 8)
        self.assertEqual(rows["latent"]["n"], self.n // 2 + self.n)                  # layer 2 (ratio 2: every other position) + layer 4 (ratio 1)
        self.assertIn("RESULT: PASS", rep.text())

    def test_float32_against_float64_oracle_int8_kv(self):
        """Two correct implementations: stage-isolated every sample is within its budget; whole-run the first deviation is reported, not failed."""
        m64 = T.model("int8-kv", "float64")
        rep = R.replay_compare(m64, self.ref, list(range(8)), list(range(self.n)), include_head=True)
        self.assertTrue(rep.ok, rep.text())
        self.assertFalse(rep.failures())
        t64 = T.oracle_trace("int8-kv", "float64", self.n)
        full = C.compare_traces(self.ref, t64)
        self.assertTrue(full.ok, full.text())

    def test_whole_run_names_the_first_deviation_after_a_flip(self):
        """Seed 0, 48 tokens: the quantisers flip a rounding decision somewhere; whole-run mode reports where, and the rest is checked against the ceiling."""
        a = T.oracle_trace("int8-kv", "float32", 48, seed=3)
        b = T.oracle_trace("int8-kv", "float64", 48, seed=3)
        rep = C.compare_traces(a, b)
        self.assertTrue(rep.ok, rep.text())
        text = rep.text()
        self.assertIn("strictly checked", text)
        if rep.onsets:
            self.assertIn("first deviation above float noise", text)
            self.assertGreater(rep.downstream, 0)
            self.assertTrue(all(s.extra.get("downstream") for s in rep.samples if s.level == C.Level.OK and s.rms_rel > 0.5))

    def test_different_tokens_are_an_error(self):
        other = T.oracle_trace("int8-kv", "float32", self.n, seed=2)
        rep = C.compare_traces(other, self.ref)
        self.assertFalse(rep.ok)
        self.assertTrue(any("DIFFERENT TOKENS" in e for e in rep.errors))


class Injection(unittest.TestCase):
    """An error injected into the engine's copy of one stage, layer and position must fail and be NAMED."""
    @classmethod
    def setUpClass(cls):
        cls.n = 16
        cls.ref = T.oracle_trace("int8-kv", "float32", cls.n)
        cls.m = T.model("int8-kv", "float32")

    def layer_report(self, eng):
        return R.replay_compare(self.m, eng, list(range(8)), list(range(self.n)), include_head=True, strict=True)

    CASES = [("attn_in", 6, 5), ("q", 6, 5), ("kv_win", 6, 2), ("attn_out", 6, 3), ("ffn_in", 9, 1), ("router_w", 6, 5), ("ffn_out", 6, 6), ("block_out", 6, 4),
             ("pre_mix", 6, 4), ("latent", 7, 2), ("index_k", 7, 2), ("embed", 6, None), ("final_hidden", 6, None), ("logits", 6, None)]

    def test_each_stage_is_named_in_layer_mode_and_in_whole_run_mode(self):
        for stage, pos, layer in self.CASES:
            with self.subTest(stage=stage, pos=pos, layer=layer):
                eng = T.engine_copy(self.ref, f"inj_{stage}", mutate={(stage, pos, layer): T.noise(0.4 if stage in ("latent", "index_k") else 0.2)})
                full = C.compare_traces(eng, self.ref)
                self.assertFalse(full.ok)
                ff = full.first_failure()
                self.assertEqual((ff.stage, ff.layer, ff.pos), (stage, layer, pos), full.text())
                self.assertIn(f"stage {stage}", full.text())
                lay = self.layer_report(eng)
                self.assertFalse(lay.ok)
                ff = lay.first_failure()
                self.assertEqual((ff.stage, ff.layer, ff.pos), (stage, layer, pos), lay.text())

    def test_layer_mode_isolates_the_error(self):
        """A wrong q fails q and NOTHING else: attn_out is recomputed from the engine's attn_in, not from its q."""
        eng = T.engine_copy(self.ref, "inj_q_only", mutate={("q", 6, 5): T.noise(0.2)})
        lay = self.layer_report(eng)
        self.assertEqual({(s.stage, s.layer, s.pos) for s in lay.failures()}, {("q", 5, 6)})
        eng = T.engine_copy(self.ref, "inj_ffn_in_only", mutate={("ffn_in", 9, 1): T.noise(0.2)})
        lay = self.layer_report(eng)
        failed = {(s.stage, s.layer, s.pos) for s in lay.failures()}
        self.assertIn(("ffn_in", 1, 9), failed)
        self.assertTrue(failed <= {("ffn_in", 1, 9), ("router_idx", 1, 9), ("router_w", 1, 9), ("ffn_out", 1, 9)}, failed)       # what reads it, nothing upstream

    def test_a_small_error_is_a_flip_not_a_failure_a_systematic_one_exceeds_the_budget(self):
        """q is read by nothing downstream in layer mode, so a 1e-3 error in it is one isolated sample in the flip band (what one int8 flip inside the stage looks
        like); the same error at every position is a systematic difference and exceeds the stage's budget."""
        one = T.engine_copy(self.ref, "inj_small_one", mutate={("q", 6, 3): lambda a: a * np.float32(1.0 + 1e-3)})
        lay = self.layer_report(one)
        self.assertTrue(lay.ok, lay.text())
        self.assertEqual([(s.stage, s.layer, s.pos) for s in lay.samples if s.level == C.Level.FLIP], [("q", 3, 6)])
        every = {("q", p, 3): (lambda a: a * np.float32(1.0 + 1e-3)) for p in range(self.n)}
        many = T.engine_copy(self.ref, "inj_small_all", mutate=every)
        lay = self.layer_report(many)
        self.assertFalse(lay.ok)
        self.assertEqual([v[0] for v in lay.budget_violations()], ["q"], lay.text())
        self.assertIn("a systematic difference", lay.text())
        self.assertFalse(lay.failures())                                                    # none of them is beyond hard: the verdict comes from the budget
        # an error beyond the hard tolerance fails at once
        big = T.engine_copy(self.ref, "inj_big_one", mutate={("q", 6, 3): lambda a: a * np.float32(1.05)})
        self.assertFalse(self.layer_report(big).ok)

    def test_nan_and_inf_always_fail(self):
        def poison(a):
            a = a.copy()
            a.flat[5] = np.nan
            return a
        eng = T.engine_copy(self.ref, "inj_nan", mutate={("ffn_out", 4, 3): poison})
        full = C.compare_traces(eng, self.ref)
        self.assertFalse(full.ok)
        ff = full.first_failure()
        self.assertEqual((ff.stage, ff.layer, ff.pos), ("ffn_out", 3, 4))
        self.assertIn("NaN/Inf", ff.note)
        lay = self.layer_report(eng)
        self.assertFalse(lay.ok)
        inf = T.engine_copy(self.ref, "inj_inf", mutate={("logits", 4, None): lambda a: np.where(np.arange(a.size) == 7, np.inf, a).astype(a.dtype)})
        self.assertFalse(C.compare_traces(inf, self.ref).ok)

    def test_wrong_selection_is_named(self):
        def other_expert(a):
            a = a.copy()
            a[0] = (int(a[0]) + 1) % 16 if (int(a[0]) + 1) % 16 not in a else (int(a[0]) + 2) % 16
            return a
        eng = T.engine_copy(self.ref, "inj_router", mutate={("router_idx", 5, 4): other_expert})
        full = C.compare_traces(eng, self.ref)
        ff = full.first_failure()
        self.assertEqual((ff.stage, ff.layer, ff.pos), ("router_idx", 4, 5), full.text())
        self.assertIn("selections differ", ff.note)
        lay = self.layer_report(eng)
        self.assertEqual((lay.first_failure().stage, lay.first_failure().layer, lay.first_failure().pos), ("router_idx", 4, 5))
        # a dropped / foreign top-k position
        tk = self.ref.get("topk", 12, 6)
        self.assertGreater(len(tk), 2)
        def foreign(a):                                                       # three entries replaced by valid positions that were not selected
            fresh = sorted(set(range(13)) - set(a.tolist()))[:3]
            return np.sort(np.concatenate([a[3:], fresh])).astype("<i4")
        eng = T.engine_copy(self.ref, "inj_topk", mutate={("topk", 12, 6): foreign})
        ff = C.compare_traces(eng, self.ref).first_failure()
        self.assertEqual((ff.stage, ff.layer, ff.pos), ("topk", 6, 12))

    def test_near_tie_is_reported_and_excused_only_with_a_small_oracle_margin(self):
        def other_expert(a):
            a = a.copy()
            a[0] = (int(a[0]) + 1) % 16 if (int(a[0]) + 1) % 16 not in a else (int(a[0]) + 2) % 16
            return a
        eng = T.engine_copy(self.ref, "tie_eng", mutate={("router_idx", 5, 4): other_expert})
        tight = T.engine_copy(self.ref, "tie_ref", mutate={("router_margin", 5, 4): lambda a: np.array([1e-9, a[1]], dtype=a.dtype)})
        rep = C.compare_traces(eng, tight)
        self.assertEqual(len(rep.tie_events), 1, rep.text())
        self.assertEqual(rep.tie_events[0][:3], (5, 4, "router_idx"))
        self.assertIn("near-ties", rep.text())
        self.assertFalse([s for s in rep.failures() if s.stage == "router_idx"])
        loose = T.engine_copy(self.ref, "tie_ref2", mutate={("router_margin", 5, 4): lambda a: np.array([0.5, a[1]], dtype=a.dtype)})
        self.assertTrue(any(s.stage == "router_idx" for s in C.compare_traces(eng, loose).failures()))

    def test_missing_stages(self):
        eng = T.engine_copy(self.ref, "missing", drop=[("ffn_out", 3, 2), ("q", 3, 2)])
        rep = C.compare_traces(eng, self.ref)
        self.assertTrue(rep.ok)                                                              # not compared, reported
        self.assertEqual({(m[0], m[1], m[2]) for m in rep.missing}, {("ffn_out", 2, 3), ("q", 2, 3)})
        self.assertIn("missing in the engine trace", rep.text())
        strict = C.compare_traces(eng, self.ref, strict=True)
        self.assertFalse(strict.ok)
        self.assertIn("MISSING in the engine trace", strict.text())
        # an optional stage (index_margin is oracle-only) is never required
        self.assertFalse(any(not TI.STAGE[m[0]].required for m in strict.missing_required))

    def test_positions_layers_stages_filters(self):
        rep = C.compare_traces(self.ref, self.ref, positions=[1, 2], layers=[3], stages=["attn_in", "logits"])
        self.assertEqual({(s.stage, s.layer, s.pos) for s in rep.samples}, {("attn_in", 3, 1), ("attn_in", 3, 2), ("logits", None, 1), ("logits", None, 2)})


class Logits(unittest.TestCase):
    def test_logits_report(self):
        rng = np.random.default_rng(0)
        o = rng.standard_normal((50, 32)).astype(np.float32) * 3
        e = o + rng.standard_normal(o.shape).astype(np.float32) * 1e-4
        r = C.logits_report(e, o)
        self.assertEqual((r["rows"], r["nonfinite_rows"], r["top1"]), (50, 0, 1.0))
        self.assertLess(r["kl_max"], 1e-6)
        bad = e.copy()
        bad[7] = np.roll(bad[7], 1)
        r = C.logits_report(bad, o)
        self.assertEqual(r["mismatch_positions"], [7])
        self.assertAlmostEqual(r["top1"], 0.98)
        # a row whose oracle top-2 gap is tiny is a near-tie and can be left out of the gate
        o2, e3 = o.copy(), o.copy()
        i1, i2 = np.argsort(o2[7])[-1], np.argsort(o2[7])[-2]
        o2[7, i1] = o2[7, i2] + np.float32(1e-6)
        e3[:] = o2
        e3[7, i1], e3[7, i2] = o2[7, i2], o2[7, i1]
        r = C.logits_report(e3, o2, near_tie_gap=1e-3)
        self.assertEqual((r["mismatch_positions"], r["near_tie_rows"], r["top1"], r["top1_excluding_near_ties"]), ([7], 1, 0.98, 1.0))
        e2 = e.copy()
        e2[3, 4] = np.inf
        self.assertEqual(C.logits_report(e2, o)["nonfinite_rows"], 1)

    def test_logits_command(self):
        rng = np.random.default_rng(1)
        o = rng.standard_normal((6, 40)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            TI.write_logits_dump(d / "o.bin", o)
            TI.write_logits_dump(d / "e.bin", o + np.float32(1e-6))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(C.main(["logits", "--engine", str(d / "e.bin"), "--oracle", str(d / "o.bin")]), 0)
            self.assertIn("PASS", buf.getvalue())
            TI.write_logits_dump(d / "bad.bin", np.roll(o, 1, axis=1))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main(["logits", "--engine", str(d / "bad.bin"), "--oracle", str(d / "o.bin")]), 1)


class CommandLine(unittest.TestCase):
    def test_trace_command_exit_codes_and_json(self):
        ref = T.oracle_trace("int8-kv", "float32", 12)
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "r.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main(["trace", "--engine", str(ref.path), "--oracle", str(ref.path), "--strict", "--json", str(out)]), 0)
            j = json.loads(out.read_text())
            self.assertTrue(j["ok"])
            self.assertEqual(j["mode"], "full")
            bad = T.engine_copy(ref, "cli_bad", mutate={("attn_out", 4, 3): T.noise(0.3)})
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(C.main(["trace", "--engine", str(bad.path), "--oracle", str(ref.path), "--json", str(out)]), 1)
            self.assertIn("FIRST FAILURE: position 4, layer 3, stage attn_out", buf.getvalue())
            self.assertEqual(json.loads(out.read_text())["first_failure"]["stage"], "attn_out")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(C.main(["trace", "--engine", str(pathlib.Path(d) / "nothing"), "--oracle", str(ref.path)]), 2)

    def test_layers_command_on_the_mini_model(self):
        ref = T.oracle_trace("int8-kv", "float32", 12)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = C.main(["layers", "--engine", str(ref.path), "--gguf", str(T.gguf_shard1()), "--positions", "0,5,11", "--layers", "2,6,head"])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn("RESULT: PASS", buf.getvalue())
        self.assertIn("layer-by-layer", buf.getvalue())

    def test_oracle_command_writes_a_trace_and_a_logits_dump(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            with contextlib.redirect_stdout(io.StringIO()):
                rc = C.main(["oracle", "--gguf", str(T.gguf_shard1()), "--tokens", "5,17,200", "--max-new", "3", "--out", str(d / "t"),
                             "--logits-dump", str(d / "l.bin"), "--max-seq-len", "32"])
            self.assertEqual(rc, 0)
            t = TI.Trace(d / "t")
            self.assertEqual(len(t.tokens), 6)
            self.assertEqual(t.n_prompt, 3)
            self.assertEqual(TI.read_logits_dump(d / "l.bin").shape, (6, T.VOCAB))
            self.assertEqual(TI.read_logits_dump(d / "l.bin")[2].tolist(), t.get("logits", 2).tolist())


if __name__ == "__main__":
    unittest.main()
