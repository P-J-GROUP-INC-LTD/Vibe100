"""Tests for ref/ds41/trace_io.py: the DS-1 trace format and the oracle's tracing (python3 -m unittest discover -s tools/ds41, or pytest)."""
import json
import pathlib
import tempfile
import unittest

import numpy as np

import ds1_testlib as T
from ref.ds41 import attention as A
from ref.ds41 import model as M
from ref.ds41 import trace_io as TI
from ref.ds41.config import Mode


class Format(unittest.TestCase):
    def test_file_names(self):
        self.assertEqual(TI.stage_filename("attn_in", 17, 3), "attn_in.L03.p00017.npy")
        self.assertEqual(TI.stage_filename("logits", 0), "logits.p00000.npy")
        self.assertEqual(TI.stage_filename("embed", 123456), "embed.p123456.npy")
        with self.assertRaises(TI.TraceError):
            TI.stage_filename("attn_in", 1)                     # a per-layer stage needs a layer
        with self.assertRaises(TI.TraceError):
            TI.stage_filename("logits", 1, 2)                   # a global stage has none

    def test_stage_table_is_the_ds1_contract(self):
        """DS1.md section 6 lists these; the engine must write all of them (index_k.L in addition: the indexer layers' replay needs it)."""
        contract = ["embed", "engram_out", "attn_in", "q", "kv_win", "latent", "topk", "attn_out", "ffn_in", "router_idx", "router_w", "ffn_out", "block_out",
                    "pre_mix", "final_hidden", "logits"]
        for name in contract:
            self.assertIn(name, TI.STAGE, name)
            self.assertTrue(TI.STAGE[name].required, name)
        self.assertTrue(TI.STAGE["index_k"].required)
        self.assertEqual({n for n, s in TI.STAGE.items() if s.role == "oracle"}, {"router_margin", "index_margin", "cand_margin"})
        self.assertEqual({s.dtype for s in TI.STAGES if s.kind == "set"}, {"i4"})
        self.assertEqual(set(TI.PER_POSITION), {"embed", "final_hidden", "logits"})

    def test_round_trip_every_dtype(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            with TI.TraceWriter(d, {"tokens": [3, 4, 5], "geometry": "MiniGeom"}) as w:
                w.put("embed", 0, rng.standard_normal(8))
                w.put("attn_in", 2, rng.standard_normal(8), layer=1)
                w.put("topk", 2, np.array([0, 3, 9, -1]), layer=4)
                w.put("router_idx", 2, np.array([7, 1]), layer=0)
                w.put("logits", 2, rng.standard_normal(16))
            t = TI.Trace(d)
            self.assertEqual(t.tokens, [3, 4, 5])
            self.assertEqual(t.positions(), [0, 2])
            self.assertEqual(t.stages(), {"embed", "attn_in", "topk", "router_idx", "logits"})
            self.assertEqual(t.layers_of("attn_in"), [1])
            self.assertEqual(t.positions_of("attn_in", 1), [2])
            self.assertTrue(t.has("topk", 2, 4) and not t.has("topk", 2, 3) and not t.has("topk", 1, 4))
            self.assertIsNone(t.get("attn_in", 0, 1))
            self.assertEqual(t.get("topk", 2, 4).tolist(), [0, 3, 9, -1])
            self.assertEqual(t.get("topk", 2, 4).dtype, np.dtype("<i4"))
            self.assertEqual(t.get("embed", 0).dtype, np.dtype("<f4"))
            with self.assertRaises(TI.TraceError):
                t.need("embed", 5)
            meta = json.loads((pathlib.Path(d) / "trace.json").read_text())
            self.assertEqual(meta["positions"], [0, 2])
            self.assertEqual(meta["format"], "ds41-trace")

    def test_files_are_plain_little_endian_npy(self):
        """What the C++ engine has to produce: a version 1.0 .npy header, '<f4' / '<i4', C order."""
        with tempfile.TemporaryDirectory() as d:
            with TI.TraceWriter(d) as w:
                w.put("ffn_out", 5, np.arange(6, dtype=np.float64).reshape(2, 3), layer=2)
                w.put("router_idx", 5, np.array([4, 2, 9]), layer=2)
            raw = (pathlib.Path(d) / "ffn_out.L02.p00005.npy").read_bytes()
            self.assertEqual(raw[:6], b"\x93NUMPY")
            self.assertEqual(raw[6:8], b"\x01\x00")
            hdr_len = int.from_bytes(raw[8:10], "little")
            hdr = raw[10:10 + hdr_len].decode("latin1")
            self.assertIn("'descr': '<f4'", hdr)
            self.assertIn("'fortran_order': False", hdr)
            self.assertIn("(2, 3)", hdr)
            self.assertEqual((10 + hdr_len) % 64, 0)                          # numpy's 64-byte alignment of the data
            self.assertEqual(len(raw), 10 + hdr_len + 24)
            self.assertIn("'descr': '<i4'", (pathlib.Path(d) / "router_idx.L02.p00005.npy").read_bytes()[10:200].decode("latin1"))

    def test_errors(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            with self.assertRaises(TI.TraceError):
                TI.Trace(d)                                                   # no trace.json
            (d / "trace.json").write_text("{not json")
            with self.assertRaises(TI.TraceError):
                TI.Trace(d)
            (d / "trace.json").write_text(json.dumps({"format": "something-else"}))
            with self.assertRaises(TI.TraceError):
                TI.Trace(d)
            (d / "trace.json").write_text(json.dumps({"format": "ds41-trace", "version": 99, "tokens": [], "quant": {}}))
            with self.assertRaises(TI.TraceError):
                TI.Trace(d)
            (d / "trace.json").write_text(json.dumps({"format": "ds41-trace", "version": 1, "tokens": [1], "quant": {}}))
            np.save(d / "embed.p00000.npy", np.zeros(4, np.float32))
            (d / "embed.p00000.npy").write_bytes((d / "embed.p00000.npy").read_bytes()[:-9])         # a write cut short by a crash
            t = TI.Trace(d)
            self.assertTrue(t.has("embed", 0))
            with self.assertRaises(TI.TraceError):
                t.get("embed", 0)

    def test_int32_overflow_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TI.TraceError):
                TI.TraceWriter(d).put("topk", 0, np.array([2 ** 40]), layer=2)

    def test_quant_dict(self):
        q = TI.quant_from_dict({"int8_act": True, "window_kv": True, "unknown_key": 1})
        self.assertTrue(q.int8_act and q.window_kv and not q.index)
        self.assertEqual(TI.quant_from_dict(TI.quant_to_dict(q)), q)
        self.assertEqual(TI.quant_from_dict(None), TI.QuantConfig.exact())

    def test_logits_dump_round_trip(self):
        rows = np.random.default_rng(0).standard_normal((3, 7)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "l.bin"
            TI.write_logits_dump(p, rows)
            self.assertEqual(p.stat().st_size, 8 + 3 * 7 * 4)
            np.testing.assert_array_equal(TI.read_logits_dump(p), rows)
            with open(p, "ab") as f:
                f.write(b"\0" * 5)
            with self.assertRaises(TI.TraceError):
                TI.read_logits_dump(p)


class OracleTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n = 20
        cls.tr = T.oracle_trace("int8-kv", "float32", cls.n)
        cls.m = T.model("int8-kv", "float32")
        cls.cfg = cls.m.cfg

    def test_metadata(self):
        t = self.tr
        self.assertEqual(t.producer, "oracle")
        self.assertEqual(t.tokens, T.tokens(self.n))
        self.assertEqual(t.positions(), list(range(self.n)))
        self.assertEqual(t.quant(), self.m.quant)
        mm = t.meta["model"]
        self.assertEqual(mm["n_layers"], 8)
        self.assertEqual(mm["layer_modes"], ["swa", "swa", "full", "reuse", "full", "reuse", "reindex", "reuse"])
        self.assertEqual(mm["engram_layer_ids"], [1, 3])

    def test_every_required_stage_is_present_where_it_exists(self):
        t, cfg, modes = self.tr, self.cfg, self.m.modes
        for p in range(self.n):
            self.assertTrue(t.has("embed", p) and t.has("final_hidden", p) and t.has("logits", p))
            for l in range(cfg.n_layers):
                for st in ("attn_in", "q", "kv_win", "attn_out", "ffn_in", "router_idx", "router_w", "ffn_out", "block_out", "pre_mix", "router_margin"):
                    self.assertTrue(t.has(st, p, l), (st, p, l))
                self.assertEqual(t.has("engram_out", p, l), l in cfg.engram_layer_ids)
                ratio = cfg.compress_ratios[l]
                self.assertEqual(t.has("topk", p, l), ratio > 0, (p, l))          # every layer with a compressed cache, REUSE included
                published = modes[l] is Mode.FULL and (p + 1) % ratio == 0 if ratio else False
                self.assertEqual(t.has("latent", p, l), published, (p, l))
                self.assertEqual(t.has("index_k", p, l), published, (p, l))

    def test_shapes_and_dtypes(self):
        t, c = self.tr, self.cfg
        self.assertEqual(t.get("embed", 3).shape, (c.dim,))
        self.assertEqual(t.get("attn_in", 3, 2).shape, (c.dim,))
        self.assertEqual(t.get("q", 3, 2).shape, (c.n_heads, c.head_dim))
        self.assertEqual(t.get("kv_win", 3, 2).shape, (c.head_dim,))
        self.assertEqual(t.get("latent", 3, 2).shape, (c.head_dim,))
        self.assertEqual(t.get("index_k", 3, 2).shape, (c.index_head_dim,))
        self.assertEqual(t.get("engram_out", 3, 1).shape, (c.hc_mult, c.dim))
        self.assertEqual(t.get("block_out", 3, 2).shape, (c.hc_mult, c.dim))
        self.assertEqual(t.get("pre_mix", 3, 2).shape, (c.hc_mult,))
        self.assertEqual(t.get("router_idx", 3, 2).shape, (c.n_activated_experts,))
        self.assertEqual(t.get("router_idx", 3, 2).dtype, np.dtype("<i4"))
        self.assertEqual(t.get("router_w", 3, 2).shape, (c.n_activated_experts,))
        self.assertEqual(t.get("logits", 3).shape, (T.VOCAB,))
        tk = t.get("topk", 10, 6)                                                # REINDEX layer at position 10
        self.assertEqual(tk.dtype, np.dtype("<i4"))
        self.assertTrue((tk >= 0).all() and (np.diff(tk) > 0).all() and tk.max() < (10 + 1) // c.compress_ratios[6])
        self.assertLessEqual(len(tk), c.index_topk)

    def test_traced_run_is_bit_identical_to_the_untraced_one(self):
        """The wrappers around the oracle's functions must not change a single bit (the capture is the only difference)."""
        m = self.m
        cache = m.new_cache()
        ids = T.tokens(self.n)
        for p, tkn in enumerate(ids):
            lg = m.forward(np.array([tkn]), p, cache)
            np.testing.assert_array_equal(lg.astype(np.float32), self.tr.get("logits", p), err_msg=f"position {p}")

    def test_stages_match_what_the_oracle_computed(self):
        """kv_win / latent / index_k are exactly the cache rows the oracle holds; q and topk are what the sparse attention read."""
        m, t, cfg = self.m, self.tr, self.cfg
        cache = m.new_cache()
        for p, tkn in enumerate(T.tokens(self.n)):
            m.forward(np.array([tkn]), p, cache)
        W = cfg.window_size
        for l in range(cfg.n_layers):
            lc = cache.layers[l]
            for p in range(max(0, self.n - W), self.n):
                np.testing.assert_array_equal(lc.win_kv[p % W], t.get("kv_win", p, l), err_msg=f"kv_win layer {l} position {p}")
            ratio = cfg.compress_ratios[l]
            if ratio and self.m.modes[l] is Mode.FULL:
                for j in range(self.n // ratio):
                    q = (j + 1) * ratio - 1
                    np.testing.assert_array_equal(lc.comp_kv[j], t.get("latent", q, l), err_msg=f"latent layer {l} group {j}")
                    np.testing.assert_array_equal(lc.index_k[j], t.get("index_k", q, l), err_msg=f"index_k layer {l} group {j}")

    def test_attn_in_is_the_normed_pre_mixed_stream(self):
        from ref.ds41.mhc import hc_pre
        from ref.ds41.ops import rmsnorm
        m, t = self.m, self.tr
        for l in (2, 5):
            for p in (0, 7):
                x = t.get("block_out", p, l - 1)[None].astype(np.float32)
                pm = t.get("pre_mix", p, l - 1)[None].astype(np.float32)
                want = rmsnorm(hc_pre(x, pm), m.p(f"layers.{l}.attn_norm.weight"), m.cfg.norm_eps)[0]
                np.testing.assert_allclose(t.get("attn_in", p, l), want, rtol=1e-6, atol=1e-6)

    def test_prefill_mode_agrees_with_token_by_token_without_quantisers(self):
        a = T.oracle_trace("exact", "float32", 12, mode="token_by_token")
        b = T.oracle_trace("exact", "float32", 12, mode="prefill")
        self.assertEqual(a.positions(), b.positions())
        for p in range(12):
            la, lb = a.get("logits", p), b.get("logits", p)
            self.assertLess(np.abs(la - lb).max() / np.abs(la).max(), 1e-4, f"position {p}")
            self.assertTrue(np.array_equal(a.get("router_idx", p, 4), b.get("router_idx", p, 4)))
            self.assertLess(np.abs(a.get("kv_win", p, 3) - b.get("kv_win", p, 3)).max(), 1e-5)

    def test_selection_margins_are_recorded(self):
        t = self.tr
        mg = t.get("router_margin", 5, 3)
        self.assertEqual(mg.shape, (2,))
        self.assertGreater(mg[0], 0)
        self.assertGreater(mg[1], 0)
        self.assertTrue(t.has("index_margin", 12, 4) and t.has("cand_margin", 12, self.cfg.candidate_source_layer))

    def test_wrappers_are_removed_even_after_an_exception(self):
        before = (A.window_kv, A.sparse_attn, A.note_margin, M.attention_layer, M.moe, M.router)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                TI.run_oracle_trace(self.m, [5, 10 ** 9], d)                       # an id outside the vocabulary
        self.assertEqual(before, (A.window_kv, A.sparse_attn, A.note_margin, M.attention_layer, M.moe, M.router))

    def test_greedy_continue_token_by_token(self):
        a = TI.greedy_continue(self.m, T.tokens(5), 6, token_by_token=True)
        self.assertEqual(len(a), 6)
        self.assertEqual(a, TI.greedy_continue(self.m, T.tokens(5), 6, token_by_token=True))


if __name__ == "__main__":
    unittest.main()
