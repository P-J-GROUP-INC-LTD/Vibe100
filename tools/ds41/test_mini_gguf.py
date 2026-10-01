"""Tests for make_mini_gguf.py and gguf_io.py: the mini deepseek41 GGUF round-trips through tools/gguf_reader.py."""
import dataclasses
import math
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fixtures as T
import ggml_codecs as C
import gguf_io as G
import make_mini_gguf as MM
from gguf_reader import GGUFFile


class WriterRoundTrip(unittest.TestCase):
    def test_every_metadata_type_and_tensor_type(self):
        rng = np.random.default_rng(1)
        arrs = {"a.f32": ("F32", [64, 3]), "a.bf16": ("BF16", [40, 2]), "a.q8": ("Q8_0", [64, 5]),
                "a.mx": ("MXFP4", [96, 7]), "a.f16": ("F16", [10])}
        tensors, vals = [], {}
        for name, (ty, dims) in arrs.items():
            n = int(np.prod(dims))
            x = (rng.standard_normal(n) * 0.5).astype(np.float32)
            vals[name] = C.decode(ty, C.encode(ty, x))
            tensors.append((name, dims, ty, C.encode(ty, x)))
        kv = [("general.architecture", "string", "test"), ("t.u8", "u8", 200), ("t.i8", "i8", -5),
              ("t.u16", "u16", 60000), ("t.i16", "i16", -3000), ("t.u32", "u32", 4_000_000_000),
              ("t.i32", "i32", -2_000_000_000), ("t.u64", "u64", 2 ** 63 + 5), ("t.i64", "i64", -2 ** 62),
              ("t.f32", "f32", 1.5), ("t.f64", "f64", math.pi), ("t.bool", "bool", True), ("t.str", "string", "héllo"),
              ("t.arr_u32", "array:u32", [1, 2, 3]), ("t.arr_f32", "array:f32", [0.5, 2.0]),
              ("t.arr_str", "array:string", ["a", "bc", ""]), ("t.arr_i64", "array:i64", [-1, 2 ** 40]),
              ("t.arr_empty", "array:u32", [])]
        for align in (32, 64, 128):
            with tempfile.TemporaryDirectory() as d:
                info = G.write_gguf(Path(d) / "x.gguf", kv, tensors, alignment=align)
                g = GGUFFile(Path(d) / "x.gguf")
                self.assertEqual(g.version, 3)
                self.assertEqual(g.alignment, align)
                self.assertEqual(g.data_start, info["data_start"])
                self.assertEqual(g.data_start % align, 0)
                for k, t, v in kv:
                    self.assertEqual(g.metadata[k], v, k)
                self.assertEqual([t.name for t in g.tensors], list(arrs))
                for t in g.tensors:
                    ty, dims = arrs[t.name]
                    self.assertEqual((t.type_name, t.shape), (ty, dims))
                    self.assertEqual(t.offset % align, 0)
                    self.assertEqual(t.expected_bytes(), C.tensor_nbytes(ty, dims))
                    got = G.read_tensor_f32(Path(d) / "x.gguf", g.data_start + t.offset, t.type_name, t.shape)
                    self.assertEqual(got.shape, tuple(dims[::-1]))
                    self.assertTrue(np.array_equal(got.reshape(-1).view(np.uint32), vals[t.name].view(np.uint32)))

    def test_wrong_size_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                G.write_gguf(Path(d) / "x.gguf", [], [("t", [32], "MXFP4", bytes(16))])

    def test_expand_split(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for i in (1, 2, 3):
                (d / f"m-{i:05d}-of-00003.gguf").write_bytes(b"x")
            (d / "other-DSpark.gguf").write_bytes(b"x")
            got = G.expand_split([d / "m-00002-of-00003.gguf"])
            self.assertEqual([p.name for p in got], [f"m-{i:05d}-of-00003.gguf" for i in (1, 2, 3)])
            self.assertEqual(len(G.expand_split([d])), 4)


class MiniFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = T.mini()
        cls.cfg = cls.res["config"]
        cls.files = [GGUFFile(p) for p in cls.res["paths"]]

    def test_shards_and_split_keys(self):
        self.assertEqual(len(self.files), 3)
        total = sum(len(g.tensors) for g in self.files)
        self.assertEqual(total, len(self.res["tensors"]))
        for i, g in enumerate(self.files):
            self.assertEqual(g.metadata["split.no"], i)
            self.assertEqual(g.metadata["split.count"], 3)
            self.assertEqual(g.metadata["split.tensors.count"], total)
            self.assertEqual(g.version, 3)
        self.assertEqual(self.files[0].metadata["general.architecture"], "deepseek41")
        self.assertNotIn("general.architecture", self.files[1].metadata)      # only shard 1 carries the metadata
        self.assertRegex(self.res["paths"][0].name, r"-00001-of-00003\.gguf$")

    def test_tensor_table_matches_what_was_planned(self):
        seen = {}
        for g in self.files:
            prev_end = 0
            for t in sorted(g.tensors, key=lambda t: t.offset):
                self.assertEqual(t.offset % g.alignment, 0)
                self.assertEqual(t.offset, prev_end, t.name)               # tightly packed, aligned
                prev_end = G.align_up(t.offset + t.expected_bytes(), g.alignment)
                seen[t.name] = (t.type_name, tuple(t.shape))
        want = {n: (ty, tuple(d)) for n, (ty, d) in self.res["tensors"].items()}
        self.assertEqual(seen, want)

    def test_tensor_bytes_are_the_encoded_weights(self):
        """Read each tensor back from the file and compare with an independent regeneration."""
        import ds41_spec as S
        c = MM._as_cfg(self.cfg)
        modes = S.layer_modes(c, S.Findings())
        _, _, _, num_emb = MM.engram_tables(self.cfg)
        specs = {sp.name: sp for sp in S.tensor_specs(c, modes, self.cfg.vocab,
                                                      {L: num_emb[i] for i, L in enumerate(self.cfg.engram_layers)})}
        n = 0
        for g in self.files:
            data = Path(g.path).read_bytes()
            for t in g.tensors:
                nbytes = t.expected_bytes()
                raw = data[g.data_start + t.offset: g.data_start + t.offset + nbytes]
                want = C.encode(t.type_name, MM._weights(self.cfg, specs[t.name]))
                self.assertEqual(raw, want, t.name)
                n += 1
        self.assertEqual(n, len(specs))

    def test_metadata_keys_are_the_real_files(self):
        """Same metadata keys as mxxm-t shard 1 (minus the split bookkeeping, which the shard writer adds)."""
        real = T.headers(T.MXXM_HEADERS)[0].kv
        mini_keys = set(self.files[0].metadata)
        self.assertEqual(set(real) - mini_keys, set())
        self.assertEqual(mini_keys - set(real), set())
        a = "deepseek41."
        self.assertEqual(self.files[0].metadata[a + "attention.compress_ratios"], self.cfg.ratios)
        self.assertEqual(len(self.files[0].metadata[a + "attention.compress_ratios"]), self.cfg.n_layer + 3)
        self.assertEqual(self.files[0].metadata[a + "attention.kv_source_layer_ids"], [2, 4])
        self.assertEqual(len(self.files[0].metadata[a + "engram.primes"]), 2 * 3 * self.cfg.e_heads)
        self.assertEqual(len(self.files[0].metadata[a + "engram.multipliers"]), 2 * self.cfg.e_ngram)
        self.assertEqual(len(self.files[0].metadata[a + "engram.token_map"]), self.cfg.vocab)
        self.assertAlmostEqual(self.files[0].metadata[a + "expert_weights_scale"], 1.5)

    def test_tensor_names_and_types_are_the_real_files(self):
        """Every tensor name pattern and ggml type of the mini file occurs in mxxm-t with the same type."""
        real = {}
        for s in T.headers(T.MXXM_HEADERS)[:12]:
            for t in s.tensors:
                real[re.sub(r"^blk\.\d+\.", "blk.N.", t.name)] = t.type
        mini = {}
        for g in self.files:
            for t in g.tensors:
                mini[re.sub(r"^blk\.\d+\.", "blk.N.", t.name)] = t.type_name
        self.assertEqual(mini, real)

    def test_layer_modes_present(self):
        names = {t.name for g in self.files for t in g.tensors}
        # L2: ratio-2 Full owner: compressor kv + norm + gate, index compressor, indexer
        for n in ("attn_compressor_kv", "attn_compressor_norm", "attn_compressor_gate", "indexer_compressor_kv",
                  "indexer_compressor_norm", "indexer.attn_q_b", "indexer.proj"):
            self.assertIn(f"blk.2.{n}.weight", names)
        # L4: ratio-1 Full (no gate), L6: Reindex (indexer only), L3/L5/L7 Reuse and L0/L1 SWA: none of them
        self.assertNotIn("blk.4.attn_compressor_gate.weight", names)
        self.assertIn("blk.4.attn_compressor_kv.weight", names)
        self.assertIn("blk.6.indexer.proj.weight", names)
        self.assertNotIn("blk.6.attn_compressor_kv.weight", names)
        for L in (0, 1, 3, 5, 7):
            self.assertNotIn(f"blk.{L}.indexer.proj.weight", names)
            self.assertNotIn(f"blk.{L}.attn_compressor_kv.weight", names)
        self.assertIn("blk.1.engram_embed.weight", names)
        self.assertIn("blk.3.engram_wkv.weight", names)
        self.assertNotIn("blk.2.engram_embed.weight", names)

    def test_expert_slices_are_whole_blocks(self):
        cfg = self.cfg
        gate = next(t for g in self.files for t in g.tensors if t.name == "blk.0.ffn_gate_exps.weight")
        self.assertEqual(gate.shape, [cfg.hidden, cfg.ff, cfg.n_expert])
        self.assertEqual(gate.expected_bytes(), cfg.n_expert * cfg.ff * (cfg.hidden // 32) * 17)
        self.assertEqual(cfg.ff % 64, 0)


class Determinism(unittest.TestCase):
    def test_same_seed_same_bytes_other_seed_other_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            a = MM.build_mini(Path(d) / "a", MM.MiniConfig(seed=3))
            b = MM.build_mini(Path(d) / "b", MM.MiniConfig(seed=3))
            c = MM.build_mini(Path(d) / "c", MM.MiniConfig(seed=4))
            for pa, pb in zip(a["paths"], b["paths"]):
                self.assertEqual(pa.read_bytes(), pb.read_bytes())
            self.assertNotEqual(a["paths"][0].read_bytes(), c["paths"][0].read_bytes())

    def test_shard_counts_and_alignment(self):
        for n_shards, align in ((1, 32), (2, 64), (5, 32)):
            with tempfile.TemporaryDirectory() as d:
                res = MM.build_mini(Path(d), MM.MiniConfig(n_shards=n_shards, alignment=align))
                self.assertEqual(len(res["paths"]), n_shards)
                names = 0
                for p in res["paths"]:
                    g = GGUFFile(p)
                    self.assertEqual(g.alignment, align)
                    self.assertEqual(g.data_start % align, 0)
                    names += len(g.tensors)
                    if n_shards == 1:
                        self.assertNotIn("split.no", g.metadata)
                self.assertEqual(names, len(res["tensors"]))

    def test_config_validation(self):
        for bad in (dict(ff=48), dict(hidden=250), dict(e_head_dim=48), dict(n_used=99), dict(vocab=100),
                    dict(cand_source=3)):
            with self.assertRaises(ValueError, msg=str(bad)):
                dataclasses.replace(MM.MiniConfig(), **bad).validate()

    def test_other_dims_build(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = MM.MiniConfig(hidden=128, ff=128, n_expert=8, n_layer=6, kv_source=(2, 3), index_source=(2, 3, 5),
                                cand_source=3, engram_layers=(1, 2), n_shards=2)
            res = MM.build_mini(Path(d), cfg)
            self.assertEqual(len(res["paths"]), 2)


if __name__ == "__main__":
    unittest.main()
