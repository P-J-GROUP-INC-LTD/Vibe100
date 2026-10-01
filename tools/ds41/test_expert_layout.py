"""Tests for expert_layout.py: GPU blob and CPU halves of a routed expert, and the checks between them."""
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

import fixtures as T
import ds41_spec as S
import expert_layout as X
import cpu_pack_xcheck as CP
import ggml_c_oracle as O
import ggml_codecs as C
import gguf_io as G
import make_mini_gguf as MM
import manifest as M
from gguf_reader import GGUFFile

ORACLE = O.available()


def geometry_hpp_constants():
    """Evaluate the `inline constexpr` lines of geometry.hpp (integer arithmetic only)."""
    text = (T.REPO / "include" / "strata" / "ds41" / "geometry.hpp").read_text()
    env: dict = {}
    for m in re.finditer(r"inline constexpr\s+[\w:]+\s+(k\w+)\s*=\s*([^;]+);", text):
        expr = re.sub(r"\(\s*(size_t|int)\s*\)", "", m.group(2))
        expr = re.sub(r"(?<=[0-9.])f\b", "", expr).replace("/", "//")
        try:
            env[m.group(1)] = eval(expr, {}, dict(env))
        except Exception:  # noqa: BLE001 - struct definitions etc.
            pass
    return env


class RealGeometry(unittest.TestCase):
    def test_matches_geometry_hpp(self):
        if not (T.REPO / "include" / "strata" / "ds41" / "geometry.hpp").is_file():
            self.skipTest("geometry.hpp not present")
        k = geometry_hpp_constants()
        g = X.REAL
        for name, got in (("kGateRowBytes", g.gate_row_bytes), ("kDownRowBytes", g.down_row_bytes),
                          ("kGateBytes", g.gate_bytes), ("kDownBytes", g.down_bytes), ("kBlobBytes", g.blob_bytes),
                          ("kHalfFF", g.half_ff), ("kHalfDownRowBlocks", g.half_down_blocks),
                          ("kHalfGateBytes", g.half_gate_bytes), ("kHalfDownBytes", g.half_down_bytes),
                          ("kHalfBytes", g.half_bytes), ("kGateRowBlocks", g.gate_row_blocks),
                          ("kDownRowBlocks", g.down_row_blocks)):
            self.assertEqual(k[name], got, name)
        self.assertEqual((k["kHidden"], k["kFF"], k["kExperts"]), (g.hidden, g.ff, g.n_expert))
        # blob / half section offsets
        self.assertEqual((k["kBlobUp"], k["kBlobDown"]), (g.gate_bytes, 2 * g.gate_bytes))
        self.assertEqual((k["kHalfUp"], k["kHalfDown"]), (g.half_gate_bytes, 2 * g.half_gate_bytes))

    def test_numbers(self):
        g = X.REAL
        self.assertEqual(g.blob_bytes, 18_800_640)
        self.assertEqual(g.half_bytes, 9_400_320)
        self.assertEqual(g.gate_bytes, 6_266_880)
        self.assertEqual((g.half_ff, g.half_down_blocks), (1152, 36))

    def test_halves_on_block_boundaries(self):
        with self.assertRaises(ValueError):
            X.ExpertGeom(hidden=256, ff=96, n_expert=4)       # 48 values per half: not whole blocks

    def test_geometry_validation_is_the_cpu_kernels_rule(self):
        """expert_layout says what the C++ kernels' check_view says: hidden % 128 and (ff / 2) % 128, i.e. ff % 256."""
        self.assertEqual((X.KERNEL_HIDDEN_MULTIPLE, X.KERNEL_FF_MULTIPLE), (128, 256))
        X.ExpertGeom(hidden=128, ff=256, n_expert=2)            # the smallest geometry the kernels accept
        X.ExpertGeom(hidden=256, ff=512, n_expert=2)
        for hidden, ff in ((256, 64), (256, 128), (256, 192), (96, 256), (160, 256), (64, 256)):
            with self.assertRaises(ValueError, msg=f"hidden={hidden} ff={ff}") as cm:
                X.ExpertGeom(hidden=hidden, ff=ff, n_expert=2)
            self.assertIn("256", str(cm.exception))             # the message names the kernels' multiples
        # ... and the source of the rule is the C++ check: both numbers appear in it
        src = (T.REPO / "src" / "ds41" / "cpu" / "mxfp4_expert.cpp")
        if src.is_file():
            text = src.read_text()
            self.assertIn("v.hidden % kGroupValues != 0", text)
            self.assertIn("(v.ff / kQK) % kGroupBlocks != 0", text)
            hpp = (T.REPO / "include" / "strata" / "ds41" / "cpu" / "mxfp4_expert.hpp").read_text()
            self.assertIn("kGroupBlocks = 4", hpp)
            self.assertIn("kGroupValues = kGroupBlocks * kQK", hpp)

    def test_random_blob_at_real_shape(self):
        """No GGUF needed: a random valid expert at the real dimensions goes through the whole chain."""
        g = X.REAL
        rng = np.random.default_rng(3)
        blob = rng.integers(0, 256, size=g.blob_bytes, dtype=np.uint8)
        blob.reshape(-1, 17)[:, 0] = rng.integers(110, 130, size=g.blob_bytes // 17)       # sane scales
        h0, h1 = X.blob_to_halves(blob, g)
        self.assertEqual((len(h0), len(h1)), (9_400_320, 9_400_320))
        self.assertEqual(X.halves_to_blob(h0, h1, g), blob.tobytes())
        # spot-check the CONTRACTS.md placement byte by byte: half 1 starts at gate row 1152; its down rows hold blocks 36..71
        gate, up, down = X.blob_parts(blob, g)
        self.assertEqual(h1[:g.gate_row_bytes], gate[1152].tobytes())
        self.assertEqual(h1[g.half_gate_bytes:g.half_gate_bytes + g.gate_row_bytes], up[1152].tobytes())
        row0 = down[0].tobytes()
        self.assertEqual(h1[2 * g.half_gate_bytes:2 * g.half_gate_bytes + 36 * 17], row0[36 * 17:])
        self.assertEqual(h0[2 * g.half_gate_bytes:2 * g.half_gate_bytes + 36 * 17], row0[:36 * 17])
        rep = X.verify_expert(blob, h0, h1, g, seed=1, n_inputs=2, c_oracle=ORACLE)
        self.assertTrue(rep["ok"], rep)


class CpuPackCrossCheck(unittest.TestCase):
    """DS-C's C++ pack_cpu_half and the Python halves are two implementations of CONTRACTS.md: they must agree."""

    @unittest.skipUnless(CP.available(), "DS-C sources / C++ compiler not available: " + (CP.why_not() or ""))
    def test_python_halves_equal_cpp_pack_cpu_half(self):
        g = X.REAL
        rng = np.random.default_rng(21)
        for _ in range(2):
            blob = rng.integers(0, 256, size=g.blob_bytes, dtype=np.uint8)
            h0, h1 = X.blob_to_halves(blob, g)
            self.assertEqual(CP.pack_cpu_half(blob, 0), h0)
            self.assertEqual(CP.pack_cpu_half(blob, 1), h1)


class MiniExpertsRunThroughTheCppKernels(unittest.TestCase):
    """The mini GGUF's experts (runtime dims 256 x 256, not the compiled-in 5120 x 2304) go through DS-C's kernels: the default dims satisfy
    their check_view (hidden % 128, expert FF % 256), which the old defaults (FF 64) did not - the kernels abort on those."""

    @classmethod
    def setUpClass(cls):
        cls.res = T.mini()
        recs = [G.load_shard(p, i) for i, p in enumerate(cls.res["paths"])]
        cls.manifest, F = M.analyze(recs, geometry="self")
        assert not F.errors
        cls.src = X.ExpertSource(cls.manifest)

    @classmethod
    def tearDownClass(cls):
        cls.src.close()

    @unittest.skipUnless(CP.available(), "DS-C sources / C++ compiler not available: " + (CP.why_not() or ""))
    def test_mini_experts_run_and_match_float64(self):
        g = self.src.geom
        rng = np.random.default_rng(8)
        for layer, e in ((0, 0), (3, 9), (7, 15)):
            blob = self.src.blob(layer, e)
            Wg, Wu, Wd = X.dequant_blob(blob, g)
            x = (rng.standard_normal((3, g.hidden)) * 3.0).astype(np.float32)        # loud enough to reach the clamps
            w = rng.uniform(0.2, 1.5, size=3).astype(np.float32)
            ref = np.stack([X.expert_forward(x[t].astype(np.float64), Wg, Wu, Wd, float(w[t]))[0] for t in range(3)])
            for isa in (n for n in CP.ISAS if CP.isa_supported(n)):
                y_whole = CP.expert_run(blob, g.hidden, g.ff, -1, x, w, isa)
                y0 = CP.expert_run(blob, g.hidden, g.ff, 0, x, w, isa)      # one partial per half (per socket): never one shared buffer
                y1 = CP.expert_run(blob, g.hidden, g.ff, 1, x, w, isa)
                for name, y in (("whole", y_whole), ("halves", y0 + y1)):
                    rel = np.linalg.norm(y - ref) / np.linalg.norm(ref)
                    self.assertTrue(np.isfinite(y).all(), (layer, e, isa, name))
                    self.assertLess(rel, 0.05, f"layer {layer} expert {e} {isa} {name}: relative error {rel:.3g}")     # the int8 activation error (~1 %)
                self.assertLess(np.linalg.norm(y_whole - (y0 + y1)) / np.linalg.norm(y_whole), 1e-4)        # only the FP32 order of the down sum differs

    @unittest.skipUnless(CP.available(), "DS-C sources / C++ compiler not available: " + (CP.why_not() or ""))
    def test_the_old_default_dims_are_refused_before_the_kernels_could_abort(self):
        with self.assertRaises(ValueError):                    # FF 64 (the old mini default): the kernels' check_view would abort the process
            CP.expert_run(bytes(100), 256, 64, -1, np.zeros((1, 256), np.float32), [1.0])


class MiniExperts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = T.mini()
        cls.cfg = cls.res["config"]
        recs = [G.load_shard(p, i) for i, p in enumerate(cls.res["paths"])]
        cls.manifest, F = M.analyze(recs, geometry="self")
        assert not F.errors
        cls.src = X.ExpertSource(cls.manifest)
        cls.g = cls.src.geom
        cls.files = {Path(p).name: GGUFFile(p) for p in cls.res["paths"]}

    @classmethod
    def tearDownClass(cls):
        cls.src.close()

    def raw_slice(self, layer, which, e):
        """Independent of ExpertSource: find the tensor with the reference reader and cut the expert's bytes."""
        name = f"blk.{layer}.ffn_{which}_exps.weight"
        for fname, gf in self.files.items():
            for t in gf.tensors:
                if t.name == name:
                    per = t.expected_bytes() // t.shape[2]
                    data = Path(gf.path).read_bytes()
                    start = gf.data_start + t.offset + e * per
                    return data[start:start + per]
        raise KeyError(name)

    def test_geometry_from_the_file(self):
        g = self.g
        self.assertEqual((g.hidden, g.ff, g.n_expert), (256, 256, 16))
        self.assertEqual(g.blob_bytes, 2 * 256 * 8 * 17 + 256 * 8 * 17)
        self.assertEqual(self.src.layers, list(range(8)))

    def test_blob_is_the_three_gguf_slices(self):
        for layer, e in ((0, 0), (2, 7), (7, 15), (4, 3)):
            want = self.raw_slice(layer, "gate", e) + self.raw_slice(layer, "up", e) + self.raw_slice(layer, "down", e)
            blob = self.src.blob(layer, e)
            self.assertEqual(len(blob), self.g.blob_bytes)
            self.assertEqual(blob, want)

    def test_halves_by_explicit_loops(self):
        """CONTRACTS.md placement, re-derived with plain Python loops: gate/up rows ff/2*k.., down blocks (ff/64)*k.."""
        g = self.g
        for layer, e in ((1, 4), (6, 9)):
            gate = self.raw_slice(layer, "gate", e)
            up = self.raw_slice(layer, "up", e)
            down = self.raw_slice(layer, "down", e)
            h0, h1 = self.src.halves(layer, e)
            for k, half in enumerate((h0, h1)):
                exp = bytearray()
                for r in range(k * g.half_ff, (k + 1) * g.half_ff):
                    exp += gate[r * g.gate_row_bytes:(r + 1) * g.gate_row_bytes]
                for r in range(k * g.half_ff, (k + 1) * g.half_ff):
                    exp += up[r * g.gate_row_bytes:(r + 1) * g.gate_row_bytes]
                for r in range(g.hidden):
                    row = down[r * g.down_row_bytes:(r + 1) * g.down_row_bytes]
                    for b in range(k * g.half_down_blocks, (k + 1) * g.half_down_blocks):
                        exp += row[b * 17:(b + 1) * 17]
                self.assertEqual(len(half), g.half_bytes)
                self.assertEqual(half, bytes(exp), (layer, e, k))

    def test_every_expert_verifies(self):
        n = 0
        for layer in self.src.layers:
            for e in range(self.g.n_expert):
                blob = self.src.blob(layer, e)
                h0, h1 = X.blob_to_halves(blob, self.g)
                rep = X.verify_expert(blob, h0, h1, self.g, seed=layer * 100 + e, n_inputs=3)
                self.assertTrue(rep["ok"], (layer, e, rep))
                self.assertLess(rep["max_rel_err"], 1e-12)
                n += 1
        self.assertEqual(n, 8 * 16)

    @unittest.skipUnless(ORACLE, "ggml C source or compiler not available")
    def test_dequant_matches_ggml_c(self):
        blob = self.src.blob(3, 5)
        h0, h1 = X.blob_to_halves(blob, self.g)
        rep = X.verify_expert(blob, h0, h1, self.g, c_oracle=True)
        self.assertTrue(rep["ok"], rep)
        self.assertIn("numpy dequant == GGML C dequantize_row_mxfp4", rep["checks"])

    def test_clamps_are_exercised(self):
        blob = self.src.blob(0, 0)
        h0, h1 = X.blob_to_halves(blob, self.g)
        rep = X.verify_expert(blob, h0, h1, self.g, seed=5, n_inputs=6)
        detail = rep["checks"]["y_0 + y_1 == y on random inputs"]["detail"]
        self.assertRegex(detail, r"clamp hits [1-9]")

    def test_file_matches_the_quantised_source_weights(self):
        """File layout oracle: encode the generator's float weights with the codec alone, slice expert e out of the
        (E, rows, cols) array, and run the contract's math; the halves read from the GGUF must give the same y."""
        cfg, g = self.cfg, self.g
        c = MM._as_cfg(cfg)
        modes = S.layer_modes(c, S.Findings())
        specs = {sp.name: sp for sp in S.tensor_specs(c, modes, cfg.vocab, {1: 6126, 3: 6332})}
        rng = np.random.default_rng(0)
        for layer, e in ((0, 0), (2, 9), (5, 15), (6, 1)):
            def q(which, rows, cols):
                w = MM._weights(cfg, specs[f"blk.{layer}.ffn_{which}_exps.weight"])
                return C.decode("MXFP4", C.encode("MXFP4", w)).reshape(g.n_expert, rows, cols)[e]
            Wg, Wu, Wd = q("gate", g.ff, g.hidden), q("up", g.ff, g.hidden), q("down", g.hidden, g.ff)
            x = rng.standard_normal(g.hidden)
            y_ref, _ = X.expert_forward(x, Wg, Wu, Wd, 0.9)
            h0, h1 = self.src.halves(layer, e)
            parts = [X.dequant_half(h, g) for h in (h0, h1)]
            y = sum(p[2].astype(np.float64) @ X.expert_hidden(x, p[0], p[1], 0.9)[0] for p in parts)
            self.assertLess(float(np.abs(y - y_ref).max() / np.abs(y_ref).max()), 1e-12, (layer, e))
            # and a mis-assembled expert (down columns of the other half) gives a clearly different y
            self.assertEqual(X.halves_to_blob(h0, h1, g), self.src.blob(layer, e))
            bad = sum(parts[1 - i][2].astype(np.float64) @ X.expert_hidden(x, parts[i][0], parts[i][1], 0.9)[0]
                      for i in range(2))
            self.assertGreater(float(np.abs(bad - y_ref).max() / np.abs(y_ref).max()), 1e-3)

    def test_verify_detects_a_wrong_half(self):
        g = self.g
        blob = self.src.blob(2, 2)
        h0, h1 = X.blob_to_halves(blob, g)
        # down columns of the two halves swapped, gate/up left alone
        o = 2 * g.half_gate_bytes
        d, e = bytearray(h0), bytearray(h1)
        d[o:], e[o:] = h1[o:], h0[o:]
        rep = X.verify_expert(blob, bytes(d), bytes(e), g)
        self.assertFalse(rep["ok"])
        self.assertFalse(rep["checks"]["halves_to_blob(blob_to_halves(blob)) == blob"]["ok"])
        # a flipped bit in one half
        h0b = bytearray(h0)
        h0b[7] ^= 0x10
        rep = X.verify_expert(blob, bytes(h0b), h1, g)
        self.assertFalse(rep["ok"])

    def test_size_checks(self):
        with self.assertRaises(ValueError):
            X.blob_to_halves(bytes(10), self.g)
        with self.assertRaises(ValueError):
            X.halves_to_blob(bytes(10), bytes(10), self.g)
        with self.assertRaises(IndexError):
            self.src.slices(0, 16)

    def test_refuses_headers_json_manifest_and_non_mxfp4(self):
        m, _ = M.analyze(T.headers(T.MXXM_HEADERS), source="x")
        with self.assertRaises(ValueError):
            X.ExpertSource(m)
        m2 = dict(self.manifest)
        m2["tensors"] = {k: dict(v) for k, v in self.manifest["tensors"].items()}
        for k in ("gate", "up", "down"):
            m2["tensors"][f"blk.0.ffn_{k}_exps.weight"]["type"] = "Q4_K"
        with self.assertRaises(ValueError):
            X.ExpertSource(m2)

    def test_cli_extract_and_verify(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = X.main(["extract", str(self.res["paths"][0]), "--geometry", "self", "--layer", "3", "--expert", "7",
                             "--out", d])
            self.assertEqual(rc, 0, buf.getvalue())
            blob = (Path(d) / "l03_e007.blob").read_bytes()
            self.assertEqual(blob, self.src.blob(3, 7))
            h0, h1 = X.blob_to_halves(blob, self.g)
            self.assertEqual((Path(d) / "l03_e007.half0").read_bytes(), h0)
            self.assertEqual((Path(d) / "l03_e007.half1").read_bytes(), h1)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = X.main(["verify", str(self.res["paths"][0]), "--geometry", "self", "--sample", "3", "--seed", "2"])
            self.assertEqual(rc, 0, buf.getvalue())
            self.assertEqual(buf.getvalue().count("OK"), 3)


if __name__ == "__main__":
    unittest.main()
