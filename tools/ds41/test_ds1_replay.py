"""Tests for tools/ds41/ds1_replay.py: layer-by-layer replay equals the full run, and the minimal set of files it needs (the engine's dump protocol)."""
import shutil
import unittest

import numpy as np

import ds1_compare as C
import ds1_replay as R
import ds1_testlib as T
from ref.ds41 import trace_io as TI
from ref.ds41.config import Mode


class ReplayEqualsFullRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        cls.tr = T.oracle_trace("int8-kv", "float32", cls.n)
        cls.m = T.model("int8-kv", "float32")
        cls.cfg = cls.m.cfg

    def test_every_layer_and_position_bit_exact(self):
        """The orchestration repeats Model.block call for call: fed the oracle's own trace as the 'engine', every stage of every layer and position of the replay
        equals the full run's bit for bit (not within a tolerance)."""
        rp = R.Replayer(self.m, self.tr)
        n_checked = 0
        for p in range(self.n):
            for L in range(self.cfg.n_layers):
                out = rp.replay(L, p)
                self.assertEqual(out.pop("_notes"), [], (L, p))
                for stage, arr in out.items():
                    layer = None if stage == "embed" else L
                    if TI.STAGE[stage].role == "oracle":
                        continue
                    want = self.tr.get(stage, p, layer)
                    self.assertIsNotNone(want, (stage, L, p))
                    np.testing.assert_array_equal(np.asarray(arr, dtype=want.dtype), want, err_msg=f"{stage} layer {L} position {p}")
                    n_checked += 1
        self.assertGreater(n_checked, 1800)

    def test_report_is_clean_and_covers_the_head(self):
        rep = R.replay_compare(self.m, self.tr, list(range(self.cfg.n_layers)), list(range(self.n)), include_head=True, strict=True)
        self.assertTrue(rep.ok, rep.text())
        rows = rep.stage_rows()
        self.assertEqual(rows["logits"]["n"], self.n)
        self.assertEqual(rows["final_hidden"]["n"], self.n)
        self.assertEqual(rows["final_hidden"]["worst_rms"][0], 0.0)
        self.assertLess(rows["logits"]["worst_rms"][0], 2e-6)           # the head matmul over a batch of rows may differ from the one-row matmul in the last bits
        self.assertEqual(sum(r["flip"] + r["fail"] for r in rows.values()), 0)

    def test_replay_of_one_layer_does_not_depend_on_the_other_layers(self):
        a = R.Replayer(self.m, self.tr).replay(5, 17)
        b = R.Replayer(self.m, self.tr)
        for p in (3, 4):
            for L in (7, 2):
                b.replay(L, p)                                           # other work first: no state is carried between calls
        c = b.replay(5, 17)
        for k in a:
            if k != "_notes":
                np.testing.assert_array_equal(a[k], c[k], err_msg=k)


def needed_files(cfg, modes, L: int, p: int, *, with_cand_blocks: bool = True) -> set:
    """The files of the engine's trace that replaying layer L at position p reads: the DS1_VERIFY.md protocol, written out.  (stage, position, layer)."""
    need = set()
    W, ratio, mode = cfg.window_size, cfg.compress_ratios[L], modes[L]
    if L == 0:
        need.add(("embed", p, None))
    else:
        need |= {("block_out", p, L - 1), ("pre_mix", p, L - 1)}
    if L in cfg.engram_layer_ids:
        need.add(("engram_out", p, L))
    need |= {("attn_in", p, L), ("attn_out", p, L), ("ffn_in", p, L), ("ffn_out", p, L), ("router_idx", p, L), ("router_w", p, L), ("block_out", p, L), ("pre_mix", p, L),
             ("q", p, L), ("kv_win", p, L)}
    need |= {("kv_win", q, L) for q in range(max(0, p - W + 1), p)}                                 # the SWA ring
    if ratio:
        owner = L if mode is Mode.FULL else max(l for l in range(L) if modes[l] is Mode.FULL)
        before = p // ratio if owner == L else (p + 1) // ratio
        for j in range(before):
            q = (j + 1) * ratio - 1
            need |= {("latent", q, owner), ("index_k", q, owner)}                                   # the compressed cache of the owner
        if owner == L and ratio > 1:
            need |= {("attn_in", q, L) for q in range(p - p % ratio, p)}                            # the compressor's open group
        need.add(("topk", p, L))
        if mode is Mode.FULL and (p + 1) % ratio == 0:
            need |= {("latent", p, L), ("index_k", p, L)}
        if mode is Mode.REINDEX and with_cand_blocks:
            need.add(("cand_blocks", p, cfg.candidate_source_layer))
    return need


class MinimalDump(unittest.TestCase):
    """What the C++ engine has to write for a layer-by-layer check of (layer, position): nothing but the stage files below, no cache snapshots."""
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        cls.tr = T.oracle_trace("int8-kv", "float32", cls.n)
        cls.m = T.model("int8-kv", "float32")
        cls.cfg = cls.m.cfg

    def trimmed(self, name, files):
        d = T.root() / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        shutil.copy(self.tr.path / "trace.json", d / "trace.json")
        for stage, pos, layer in files:
            path = self.tr.path_of(stage, pos, layer)
            self.assertIsNotNone(path, (stage, pos, layer))
            shutil.copy(path, d / TI.stage_filename(stage, pos, layer))
        return TI.Trace(d)

    def test_replay_from_the_minimal_files_equals_the_full_run(self):
        for L, p in ((0, 3), (1, 12), (2, 9), (3, 15), (4, 15), (5, 19), (6, 15), (7, 14), (6, 4)):
            with self.subTest(layer=L, position=p, mode=self.m.modes[L].value):
                need = needed_files(self.cfg, self.m.modes, L, p)
                eng = self.trimmed(f"min_{L}_{p}", need)
                out = R.Replayer(self.m, eng).replay(L, p)
                out.pop("_notes")
                for stage, arr in out.items():
                    layer = None if stage == "embed" else L
                    if TI.STAGE[stage].role == "oracle":
                        continue
                    np.testing.assert_array_equal(np.asarray(arr, dtype=np.float32 if TI.STAGE[stage].dtype == "f4" else np.int32), self.tr.get(stage, p, layer),
                                                  err_msg=stage)

    def test_without_cand_blocks_the_pool_is_recomputed_from_the_candidate_source_layer(self):
        """cand_blocks is optional: a REINDEX layer then re-runs the candidate-source layer's indexer on the engine's state (more files, same answer)."""
        L, p = 6, 15
        need = needed_files(self.cfg, self.m.modes, L, p, with_cand_blocks=False)
        C0 = self.cfg.candidate_source_layer
        need |= needed_files(self.cfg, self.m.modes, C0, p) - {("cand_blocks", p, C0)}
        eng = self.trimmed("min_nocand", need)
        out = R.Replayer(self.m, eng).replay(L, p)
        np.testing.assert_array_equal(out["topk"], self.tr.get("topk", p, L))
        np.testing.assert_array_equal(out["attn_out"], self.tr.get("attn_out", p, L))

    def test_every_listed_file_is_necessary(self):
        L, p = 6, 15
        need = needed_files(self.cfg, self.m.modes, L, p)
        for drop in (("kv_win", p - 1, L), ("kv_win", 0, L), ("latent", 3, 4), ("index_k", 15, 4), ("attn_in", p, L), ("block_out", p, L - 1), ("pre_mix", p, L - 1),
                     ("cand_blocks", p, 4), ("attn_out", p, L), ("ffn_in", p, L), ("ffn_out", p, L)):
            with self.subTest(dropped=drop):
                eng = self.trimmed("min_drop", need - {drop})
                with self.assertRaises(R.ReplayError):
                    R.Replayer(self.m, eng).replay(L, p)

    def test_a_missing_input_is_an_error_in_the_report_not_a_pass(self):
        eng = T.engine_copy(self.tr, "replay_missing", drop=[("kv_win", 14, 3)])
        rep = R.replay_compare(self.m, eng, [3], [15], strict=True)
        self.assertFalse(rep.ok)
        self.assertTrue(any("cannot replay" in e and "kv_win.L03.p00014" in e for e in rep.errors), rep.errors)
        self.assertIn("ERROR:", rep.text())

    def test_reuse_layers_replay_with_the_engines_selection(self):
        """Layer 7 reuses layer 6's top-k: the engine's own topk file decides what the replay attends to, so a selection the engine got wrong fails attn_out."""
        L, p = 7, 15
        base = R.replay_compare(self.m, self.tr, [L], [p], strict=True)
        self.assertTrue(base.ok, base.text())

        def foreign(a):
            fresh = sorted(set(range(p + 1)) - set(a.tolist()))[:2]
            return np.sort(np.concatenate([a[2:], fresh])).astype("<i4")
        eng = T.engine_copy(self.tr, "replay_reuse_topk", mutate={("topk", p, L): foreign})
        rep = R.replay_compare(self.m, eng, [L], [p], strict=True)
        self.assertFalse(rep.ok)
        self.assertIn(("attn_out", L, p), {(s.stage, s.layer, s.pos) for s in rep.failures()}, rep.text())

    def test_head(self):
        rp = R.Replayer(self.m, self.tr)
        res = rp.replay_head([2, 9])
        for p in (2, 9):
            np.testing.assert_array_equal(res[p]["final_hidden"], self.tr.get("final_hidden", p))
            self.assertLess(np.abs(res[p]["logits"] - self.tr.get("logits", p)).max() / np.abs(res[p]["logits"]).max(), 2e-6)


class Flags(unittest.TestCase):
    def test_replay_takes_the_engines_quantisation_flags(self):
        """An engine with the three KV flags off (QuantConfig.int8) is checked against an oracle with those flags off."""
        tr = T.oracle_trace("int8", "float32", 10)
        self.assertFalse(tr.quant().window_kv)
        m = T.model("int8", "float32")
        rep = R.replay_compare(m, tr, list(range(8)), list(range(10)), include_head=True, strict=True)
        self.assertTrue(rep.ok, rep.text())
        # against the wrong flags the KV rows differ (fp8 / fp4 rounding): the check notices
        wrong = R.replay_compare(T.model("int8-kv", "float32"), tr, list(range(8)), list(range(10)), include_head=True)
        self.assertFalse(wrong.ok)


if __name__ == "__main__":
    unittest.main()
