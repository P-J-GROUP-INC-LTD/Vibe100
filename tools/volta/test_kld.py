"""tools/volta/test_kld.py - tests of kld_format.py (llama.cpp's --kl-divergence-base file and scoring rule) and of golden_compare.py's
`--ref-kld` and `--exact`.

  * the REAL llama.cpp (commit 3cf03257, the one setup.py pins) wrote the fixtures in testdata/tiny_kld/ - a base file, the logits of a second
    model at the scored positions, and the statistics `llama-perplexity --kl-divergence` printed for it (testdata/make_kld_fixture.py makes
    them): the reader must read that file, the scorer must reproduce those printed numbers from those logits, and the NumPy writer must
    write the same bytes llama.cpp's C++ writes;
  * the scoring math on cases computed by hand and against a plain scalar port of the C++ loop, including the edge rows (constant logits,
    a spread wider than the 16-nat window, a non-finite logit);
  * `--exact`: ULP distances, identical / one differing ULP / NaN, rows and values counted;
  * the whole `--ref-kld` flow and `--exact` flow against the FAKE ENGINE of test_golden_compare.py (one engine launch per chunk, the
    per-chunk split token, BOS substitution, the verify-window source with and without STRATA_LOGITS_DUMP, a crash, a NaN, the verdict).

    python3 -m unittest discover -s tools/volta        (needs numpy)
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_compare as G  # noqa: E402
import kld_format as K  # noqa: E402
import test_golden_compare as T  # noqa: E402  (the fake engine)

FIX = Path(__file__).resolve().parent / "testdata" / "tiny_kld"
V = T.V   # the fake model's vocabulary (64)


# ===================================================================================================== the file and the real numbers


class RealLlamaCppFiles(unittest.TestCase):
    """The fixtures were written by llama.cpp itself."""

    @classmethod
    def setUpClass(cls):
        cls.base = K.read_base(FIX / "base.kld")

    def test_header_and_layout(self):
        b = self.base
        self.assertEqual((b.n_ctx, b.n_vocab, b.n_chunk, b.chunks_present), (64, 121, 4, 4))
        self.assertEqual((b.first, b.n_scored, b.nv), (32, 31, 126))            # nv = 2 * ((121 + 1) / 2) + 4: the vocabulary is odd
        self.assertEqual(b.tokens.shape, (4, 64))
        self.assertEqual(int(b.tokens[0, 0]), 1)                                # the tokenizer put BOS (<s>, id 1) in front of the text
        self.assertEqual(b.notes, [])
        self.assertEqual(b.positions()[0], 32)
        self.assertEqual(b.positions()[-1], 62)                                 # position n_ctx - 1 is not scored: it has no target
        self.assertEqual(list(b.targets(1)), [int(t) for t in b.tokens[1, 33:]])
        self.assertEqual(Path(FIX / "base.kld").stat().st_size, 20 + 4 * 64 * 4 + 4 * 31 * 126 * 2)

    def logits(self, name):
        return np.fromfile(FIX / name, dtype="<f4", offset=8).reshape(4, 31, 121)

    def score(self, name):
        b, sc = self.base, K.KldScorer()
        rows = self.logits(name)
        for c in range(4):
            blk = b.block(c)
            scale, mlp = b.header_of(blk)
            sc.add_logits(rows[c], b.targets(c), scale, mlp, blk[:, 4:4 + b.n_vocab])
        return sc.summary()

    def test_scoring_reproduces_the_numbers_llama_cpp_printed(self):
        s = self.score("q.logits.bin")
        printed = K.parse_llama_log((FIX / "q_kl.txt").read_text(encoding="utf-8"))
        self.assertEqual(s["count"], 124)
        # llama.cpp prints 6 decimals (4 for a few); the logits here come from a separate llama_decode, so the last printed digit may differ
        self.assertAlmostEqual(s["kld_mean"], printed["kld"][0], delta=2e-6)
        self.assertAlmostEqual(s["kld_unc"], printed["kld"][1], delta=2e-6)
        self.assertAlmostEqual(s["kld_max"], printed["kld_max"], delta=2e-6)
        self.assertAlmostEqual(s["kld_median"], printed["kld_median"], delta=2e-6)
        self.assertAlmostEqual(s["kld_percentiles"]["0.99"], printed["kld_p99"], delta=2e-6)
        self.assertAlmostEqual(s["kld_percentiles"]["0.999"], printed["kld_p999"], delta=2e-6)
        self.assertAlmostEqual(100 * s["same_top"], printed["same_top"][0], delta=1e-3)
        self.assertAlmostEqual(100 * s["same_top_unc"], printed["same_top"][1], delta=1e-3)
        self.assertAlmostEqual(s["ppl"], printed["ppl_q"][0], delta=2e-5)
        self.assertAlmostEqual(s["ppl_base"], printed["ppl_base"][0], delta=2e-5)
        self.assertAlmostEqual(s["ln_ppl_ratio"], printed["ln_ppl_ratio"][0], delta=2e-6)
        self.assertAlmostEqual(s["ln_ppl_ratio_unc"], printed["ln_ppl_ratio"][1], delta=2e-6)
        self.assertAlmostEqual(s["ppl_diff"], printed["ppl_diff"][0], delta=2e-5)
        self.assertAlmostEqual(100 * s["dp_rms"], printed["dp_rms"][0], delta=1e-3)
        self.assertAlmostEqual(100 * s["dp_mean"], printed["dp_mean"][0], delta=1e-3)
        self.assertAlmostEqual(100 * s["dp_max"], printed["dp_max"], delta=1e-3)
        self.assertAlmostEqual(100 * s["ppl_cor"], printed["ppl_cor"], delta=1e-2)
        self.assertGreater(s["kld_mean"], 0.01)         # the second model really is different: this is not a trivial zero == zero

    def test_a_model_scored_against_its_own_base_is_perfect(self):
        s = self.score("base.logits.bin")
        self.assertAlmostEqual(s["kld_mean"], 0.0, delta=1e-5)        # not exactly zero: the base's 16-bit rounding
        self.assertEqual(s["same_top"], 1.0)
        self.assertAlmostEqual(s["ln_ppl_ratio"], 0.0, delta=1e-4)
        self.assertLess(s["dp_max_abs"], 1e-3)

    def test_writer_writes_the_bytes_llama_cpp_writes(self):
        b = self.base
        d = Path(tempfile.mkdtemp())
        try:
            K.write_base(d / "mine.kld", b.n_ctx, b.tokens, [self.logits("base.logits.bin")[c] for c in range(4)])
            self.assertEqual((d / "mine.kld").read_bytes(), (FIX / "base.kld").read_bytes())
            again = K.read_base(d / "mine.kld")
            self.assertEqual((again.n_ctx, again.n_vocab, again.n_chunk), (64, 121, 4))
        finally:
            shutil.rmtree(d)

    def test_parse_log_and_the_failure_mode(self):
        p = K.parse_llama_log((FIX / "q_kl.txt").read_text(encoding="utf-8"))
        self.assertEqual(set(["kld", "same_top", "ppl_q", "ppl_base", "ln_ppl_ratio", "dp_rms", "kld_max"]) - set(p), set())
        with self.assertRaises(K.KldError):
            K.parse_llama_log("perplexity: calculating perplexity over 4 chunks\nFinal estimate: PPL = 12.3 +/- 0.1\n")

    def test_a_log_that_lost_its_last_lines_is_read_from_the_chunk_table(self):
        # llama.cpp's log thread can lose the last lines at exit; the table printed before them has the same totals in its last row
        text = (FIX / "q_kl.txt").read_text(encoding="utf-8")
        full = K.parse_llama_log(text)
        cut = K.parse_llama_log(text[:text.index("====== Perplexity statistics")])
        self.assertTrue(cut["from_table"])
        self.assertNotIn("from_table", full)
        self.assertAlmostEqual(cut["kld"][0], full["kld"][0], delta=6e-6)
        self.assertAlmostEqual(cut["same_top"][0], full["same_top"][0], delta=1e-3)
        self.assertAlmostEqual(cut["ln_ppl_ratio"][0], full["ln_ppl_ratio"][0], delta=6e-6)
        self.assertAlmostEqual(cut["ln_ppl_ratio"][1], full["ln_ppl_ratio"][1], delta=6e-6)

    def test_format_summary_has_llama_cpps_lines(self):
        text = K.format_summary(self.score("q.logits.bin"))
        for line in ("Mean PPL(Q)", "Mean ln(PPL(Q)/PPL(base))", "Mean    KLD:", "Same top p:", "RMS dp", "99.0%   KLD"):
            self.assertIn(line, text)


class ReaderErrors(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.good = (FIX / "base.kld").read_bytes()

    def tearDown(self):
        shutil.rmtree(self.d)

    def put(self, data: bytes) -> Path:
        p = self.d / "x.kld"
        p.write_bytes(data)
        return p

    def test_not_a_base_file(self):
        with self.assertRaises(K.KldError) as cm:
            K.read_base(self.put(b"GGUF" + bytes(100)))
        self.assertIn("_logits_", str(cm.exception))
        with self.assertRaises(K.KldError):
            K.read_base(self.d / "missing")

    def test_truncated_tokens_and_an_interrupted_run(self):
        with self.assertRaises(K.KldError):
            K.read_base(self.put(self.good[:100]))
        per_chunk = 31 * 126 * 2
        head = 20 + 4 * 64 * 4
        b = K.read_base(self.put(self.good[:head + 2 * per_chunk + 77]))     # llama-perplexity killed in the middle of chunk 2
        self.assertEqual(b.chunks_present, 2)
        self.assertTrue(any("2 of the 4" in n for n in b.notes))
        with self.assertRaises(K.KldError):
            K.read_base(self.put(self.good[:head + 10]))                      # tokens, but no row at all

    def test_ids_outside_the_vocabulary(self):
        bad = bytearray(self.good)
        struct.pack_into("<i", bad, 20, 5000)
        with self.assertRaises(K.KldError):
            K.read_base(self.put(bytes(bad)))


# ===================================================================================================== the scoring math


def scalar_score_row(z, tok, scale, mlp, q):
    """A plain float64 port of perplexity.cpp's `log_softmax(..., base_log_prob, tok, kld)` for one row."""
    mx = max(z)
    lse = math.log(sum(math.exp(v - mx) for v in z))
    nll = mx + lse - z[tok]
    plb = [scale * qi + mlp for qi in q]
    nll_base = -plb[tok]
    kld = sum(math.exp(lp) * (lp - zi + mx + lse) for lp, zi in zip(plb, z) if lp > -16.0)
    top = max(range(len(z)), key=lambda i: (z[i], -i))
    top_b = max(range(len(plb)), key=lambda i: (plb[i], -i))
    return nll, nll_base, kld, top == top_b, math.exp(-nll) - math.exp(-nll_base)


class ScoringMath(unittest.TestCase):
    def test_hand_computed_row(self):
        # a saved row: log p = -2.0 + 0.25 * q = [-0.25, -0.5, -1.0, -2.0] (not normalised: llama.cpp does not need it to be)
        scale, mlp, q = np.float32(0.25), np.float32(-2.0), np.array([[7, 6, 4, 0]], dtype=np.uint16)
        z = np.array([[1.0, 0.5, 0.0, -1.0]], dtype=np.float32)
        sc = K.KldScorer()
        sc.add_logits(z, np.array([1]), np.array([scale]), np.array([mlp]), q)
        s = sc.summary()
        lse = math.log(math.e + math.exp(0.5) + 1.0 + math.exp(-1.0))
        lp_new = [v - lse for v in (1.0, 0.5, 0.0, -1.0)]
        lp_base = [-0.25, -0.5, -1.0, -2.0]
        self.assertAlmostEqual(s["ppl"], math.exp(-lp_new[1]), places=5)             # one row: PPL = 1 / p_new(target)
        self.assertAlmostEqual(s["ppl_base"], math.exp(0.5), places=5)                 # nll_base = 0.5
        self.assertAlmostEqual(s["kld_mean"], sum(math.exp(b) * (b - n) for b, n in zip(lp_base, lp_new)), places=5)
        self.assertEqual(s["same_top"], 1.0)                                          # argmax 0 on both sides
        self.assertAlmostEqual(s["dp_max"], math.exp(lp_new[1]) - math.exp(-0.5), places=6)
        self.assertAlmostEqual(s["dnll_mean"], -lp_new[1] - 0.5, places=5)
        # now a candidate whose top-1 is token 2: not the same top
        sc = K.KldScorer()
        sc.add_logits(np.array([[0.0, 0.5, 3.0, 0.0]], dtype=np.float32), np.array([0]), np.array([scale]), np.array([mlp]), q)
        self.assertEqual(sc.summary()["same_top"], 0.0)

    def test_vectorised_scorer_agrees_with_a_scalar_port_on_awkward_rows(self):
        rng = np.random.RandomState(4)
        rows = [rng.normal(size=19) * 3, rng.normal(size=19) * 12,       # a spread wider than the 16-nat window: clamped entries
                np.full(19, 2.5),                                          # constant: scale 0
                np.concatenate([[40.0], rng.normal(size=18)])]             # one dominant token
        z = np.array(rows, dtype=np.float32)
        base_rows = z + rng.normal(size=z.shape).astype(np.float32) * 0.3
        base_rows[2] = z[2]                                                # the constant row stays constant
        scale, mlp, q = K.quantise_rows(base_rows)
        self.assertEqual(float(scale[2]), 0.0)
        self.assertTrue((q[2] == 0).all())
        self.assertGreater(int((q[1] == 0).sum()), 0)                       # something was clamped
        tgt = np.array([3, 0, 5, 0])
        sc = K.KldScorer()
        sc.add_logits(z, tgt, scale, mlp, q)
        want = [scalar_score_row(z[i].astype(np.float64).tolist(), int(tgt[i]), float(scale[i]), float(mlp[i]), q[i].tolist()) for i in range(4)]
        s = sc.summary()
        self.assertAlmostEqual(s["kld_mean"], np.mean([w[2] for w in want]), delta=2e-5)
        self.assertAlmostEqual(s["kld_max"], max(w[2] for w in want), delta=2e-5)
        self.assertEqual(round(s["same_top"] * 4), sum(w[3] for w in want))
        self.assertAlmostEqual(s["dnll_mean"], np.mean([w[0] - w[1] for w in want]), delta=2e-5)
        self.assertAlmostEqual(s["dp_max_abs"], max(abs(w[4]) for w in want), delta=2e-6)

    def test_a_row_scored_against_itself(self):
        z = np.random.RandomState(9).normal(size=(5, 33)).astype(np.float32) * 4
        scale, mlp, q = K.quantise_rows(z)
        sc = K.KldScorer()
        sc.add_logits(z, np.arange(5), scale, mlp, q)
        s = sc.summary()
        self.assertLess(abs(s["kld_mean"]), 1e-4)
        self.assertEqual(s["same_top"], 1.0)

    def test_non_finite_rows_are_counted_and_left_out(self):
        z = np.random.RandomState(1).normal(size=(4, 17)).astype(np.float32)
        scale, mlp, q = K.quantise_rows(z)
        bad = z.copy()
        bad[1, 3] = np.nan
        bad[3, 0] = np.inf
        sc = K.KldScorer()
        sc.add_logits(bad, np.zeros(4, dtype=int), scale, mlp, q)
        s = sc.summary()
        self.assertEqual((s["count"], s["nonfinite"]), (2, 2))

    def test_percentile_and_median_are_llama_cpps(self):
        v = np.sort(np.array([5, 1, 3, 2, 4], dtype=np.float32))
        self.assertEqual(K.median(v), 3.0)
        self.assertEqual(K.median(np.array([1, 2, 3, 4], dtype=np.float32)), 2.5)
        self.assertAlmostEqual(K.percentile(v, 0.5), 3.0)
        self.assertAlmostEqual(K.percentile(v, 0.9), 4.6, places=5)              # p = 0.9 * 4 = 3.6: 0.4 * v[3] + 0.6 * v[4]
        self.assertEqual(K.percentile(v, 0.0), 1.0)
        self.assertEqual(K.percentile(v, 1.0), 5.0)

    def test_target_only_scoring_matches_the_full_row_where_both_exist(self):
        rng = np.random.RandomState(2)
        z = rng.normal(size=(12, 40)).astype(np.float32) * 2
        base = z + rng.normal(size=z.shape).astype(np.float32) * 0.2
        scale, mlp, q = K.quantise_rows(base)
        tgt = rng.randint(0, 40, size=12)
        full = K.KldScorer()
        full.add_logits(z, tgt, scale, mlp, q)
        lsm = z.astype(np.float64) - (z.max(axis=1, keepdims=True) + np.log(np.exp(z - z.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)))
        part = K.KldScorer()
        part.add_target(lsm[np.arange(12), tgt], z.argmax(axis=1), tgt, scale, mlp, lambda i: q[i])
        a, b = full.summary(), part.summary()
        self.assertFalse(b["has_kl"])
        self.assertNotIn("kld_mean", b)
        for k in ("ppl", "ppl_base", "same_top", "dnll_mean", "dp_rms"):
            self.assertAlmostEqual(a[k], b[k], delta=1e-5 * max(1.0, abs(a[k])), msg=k)
        self.assertIn("not available", K.format_summary(b))


class BosTests(unittest.TestCase):
    def gguf(self, kv: dict) -> Path:
        def s(x):
            e = x.encode()
            return struct.pack("<Q", len(e)) + e
        body = b""
        for k, (t, v) in kv.items():
            body += s(k) + struct.pack("<I", t) + (s(v) if t == 8 else struct.pack("<?" if t == 7 else "<I", v))
        p = self.d / f"m{len(list(self.d.iterdir()))}.gguf"
        p.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv)) + body)
        return p

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_bpe_vocabulary_adds_no_bos_unless_the_gguf_says_so(self):
        self.assertIsNone(K.bos_substitution(self.gguf({"tokenizer.ggml.model": (8, "gpt2")})))                  # Qwen / GPT-2 family
        self.assertIsNone(K.bos_substitution(self.gguf({"tokenizer.ggml.model": (8, "gpt2"), "tokenizer.ggml.add_bos_token": (7, False),
                                                       "tokenizer.ggml.bos_token_id": (4, 11)})))
        self.assertEqual(K.bos_substitution(self.gguf({"tokenizer.ggml.model": (8, "gpt2"), "tokenizer.ggml.add_bos_token": (7, True),
                                                      "tokenizer.ggml.bos_token_id": (4, 11)})), 11)

    def test_spm_vocabulary_adds_bos_by_default(self):
        self.assertEqual(K.bos_substitution(self.gguf({"tokenizer.ggml.model": (8, "llama"), "tokenizer.ggml.bos_token_id": (4, 1)})), 1)
        with self.assertRaises(K.KldError):
            K.bos_substitution(self.gguf({"tokenizer.ggml.model": (8, "llama")}))


# ===================================================================================================== --exact


class ExactMath(unittest.TestCase):
    def rows(self, ref, cand):
        r, c = np.asarray(ref, dtype=np.float32), np.asarray(cand, dtype=np.float32)
        return G.exact_compare_rows(lambda lo, hi: r[lo:hi], lambda lo, hi: c[lo:hi], list(range(100, 100 + len(r))))

    def test_ulp_key_counts_representable_floats_between(self):
        a = np.array([1.0, -1.0, 0.0, 3.5], dtype=np.float32)
        up = np.nextafter(a, np.float32(10))
        for x, y in zip(a, up):
            self.assertEqual(int(abs(G.ulp_key(np.array([x]))[0] - G.ulp_key(np.array([y]))[0])), 1)
        two = np.nextafter(up, np.float32(10))
        self.assertEqual(int(abs(G.ulp_key(a)[3] - G.ulp_key(two)[3])), 2)
        self.assertEqual(int(G.ulp_key(np.array([0.0], np.float32))[0]), int(G.ulp_key(np.array([-0.0], np.float32))[0]))
        self.assertEqual(int(abs(G.ulp_key(np.array([-1.0], np.float32))[0] - G.ulp_key(np.array([np.nextafter(np.float32(-1.0), np.float32(0))], np.float32))[0])), 1)
        # across zero: the smallest positive and the smallest negative are two ULPs apart (0.0 sits between them)
        self.assertEqual(int(abs(G.ulp_key(np.array([1e-45], np.float32))[0] - G.ulp_key(np.array([-1e-45], np.float32))[0])), 2)

    def test_identical(self):
        x = np.random.RandomState(0).normal(size=(5, 20)).astype(np.float32)
        e = self.rows(x, x.copy())
        self.assertTrue(e["identical"])
        self.assertEqual((e["rows"], e["rows_differing"], e["values_differing"], e["max_ulp"], e["first_diff_position"]), (5, 0, 0, 0, None))

    def test_one_differing_ulp(self):
        x = np.random.RandomState(0).normal(size=(5, 20)).astype(np.float32)
        y = x.copy()
        y[3, 7] = np.nextafter(y[3, 7], np.float32(100))
        e = self.rows(x, y)
        self.assertFalse(e["identical"])
        self.assertEqual((e["rows_differing"], e["values_differing"], e["max_ulp"]), (1, 1, 1))
        self.assertEqual((e["first_diff_position"], e["first_diff_index"]), (103, 7))
        self.assertEqual(e["differing_positions"], [103])
        self.assertEqual(e["top1_changed_rows"], 0)
        self.assertGreater(e["max_abs_diff"], 0.0)
        self.assertLess(e["max_abs_diff"], 1e-6)
        self.assertEqual(e["first_diff_values"], [float(x[3, 7]), float(y[3, 7])])

    def test_a_flipped_top1_and_the_largest_distance(self):
        x = np.zeros((2, 4), dtype=np.float32)
        x[:, 1] = 1.0
        y = x.copy()
        y[1, 2] = 1.0000001                # row 1: tokens 1 and 2 now tie / token 2 wins by one ULP
        y[1, 1] = np.nextafter(np.float32(1.0), np.float32(0))
        e = self.rows(x, y)
        self.assertEqual(e["top1_changed_rows"], 1)
        self.assertEqual(e["rows_differing"], 1)
        self.assertGreaterEqual(e["max_ulp"], 1)

    def test_nan_is_a_difference(self):
        x = np.random.RandomState(0).normal(size=(4, 9)).astype(np.float32)
        y = x.copy()
        y[2, 1] = np.nan
        e = self.rows(x, y)
        self.assertFalse(e["identical"])
        self.assertEqual((e["rows_differing"], e["nan_values"], e["max_ulp"]), (1, 1, 0))
        # the same NaN on both sides is identical bits
        e = self.rows(y, y.copy())
        self.assertTrue(e["identical"])

    def test_no_rows_is_an_error_and_shapes_must_match(self):
        with self.assertRaises(G.HarnessError):
            G.exact_compare_rows(lambda lo, hi: None, lambda lo, hi: None, [])
        with self.assertRaises(G.HarnessError):
            G.exact_compare_rows(lambda lo, hi: np.zeros((2, 3), np.float32), lambda lo, hi: np.zeros((2, 4), np.float32), [1, 2])

    def test_logpos_comparison(self):
        r = {10: {"target": 5, "logprob": -1.5, "top": 5, "top_logprob": -1.5}, 11: {"target": 6, "logprob": -2.0, "top": 3, "top_logprob": -0.5}}
        e = G.exact_compare_logpos(r, json.loads(json.dumps(r)) and {k: dict(v) for k, v in r.items()})
        self.assertTrue(e["identical"])
        c = {k: dict(v) for k, v in r.items()}
        c[11]["logprob"] = -2.000000001
        e = G.exact_compare_logpos(r, c)
        self.assertFalse(e["identical"])
        self.assertEqual((e["rows_differing"], e["first_diff_position"]), (1, 11))
        self.assertAlmostEqual(e["max_abs_diff"], 1e-9, places=12)
        del c[10]
        e = G.exact_compare_logpos(r, c)
        self.assertEqual(e["only_reference"], 1)


# ===================================================================================================== the flows, with the fake engine


def fake_logits(ids: list[int], noise: float = 0.0) -> np.ndarray:
    """The fake engine's model: logits as a function of the context (test_golden_compare.FAKE_ENGINE)."""
    seed = int.from_bytes(hashlib.sha256(repr(tuple(int(i) for i in ids)).encode()).digest()[:4], "little")
    z = np.random.RandomState(seed).normal(size=V) * 3.0
    return z.astype("<f4")


class FlowBase(unittest.TestCase):
    N_CTX = 64
    N_CHUNK = 3

    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        (cls.d / "fake_strata.py").write_text(T.FAKE_ENGINE)
        (cls.d / "pack" / "full").mkdir(parents=True)
        (cls.d / "pack_native" / "full").mkdir(parents=True)
        (cls.d / "pack_native" / "full" / "native_experts.txt").write_text("# native\n")
        cls.cfg = {"exe": str(cls.d / "fake_strata.py"), "cwd": str(cls.d),
                   "args": ["--pack", "pack/full", "--prefill", "auto", "--spec", "4", "--mtp", "x", "--max-context", "4096", "--serve"]}
        (cls.d / "cfg.json").write_text(json.dumps(cls.cfg))
        (cls.d / "cfg_native.json").write_text(json.dumps(dict(cls.cfg, args=["--pack", "pack_native/full", "--prefill", "auto", "--spec", "4",
                                                                              "--mtp", "x", "--expert-cache", "auto"])))
        rng = np.random.RandomState(21)
        cls.tokens = rng.randint(0, V, size=(cls.N_CHUNK, cls.N_CTX)).astype(np.int32)
        cls.first = cls.N_CTX // 2

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def make_base(self, name: str, bos: int | None = None, trained_on=None) -> Path:
        """A llama.cpp-format base file whose saved distributions are the fake model's own (so the fake engine, noise free, matches them)."""
        toks = self.tokens.copy()
        rows = []
        for c in range(self.N_CHUNK):
            ctx = [int(t) for t in toks[c]]
            if bos is not None and trained_on != "no_bos":
                ctx[0] = bos
            rows.append(np.stack([fake_logits(ctx[:p + 1]) for p in range(self.first, self.N_CTX - 1)]))
        p = self.d / name
        K.write_base(p, self.N_CTX, toks, rows)
        return p

    def main(self, *argv, env=None, name="w"):
        w = self.d / name
        shutil.rmtree(w, ignore_errors=True)
        old = dict(os.environ)
        os.environ.update(env or {})
        out, err = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = G.main([*argv, "--workdir", str(w)])
        finally:
            os.environ.clear()
            os.environ.update(old)
        rep = json.loads((w / "report.json").read_text()) if (w / "report.json").exists() else None
        return rc, rep, out.getvalue() + err.getvalue(), w


class RefKldFlow(FlowBase):
    def test_dump_source_noise_free_matches_the_base(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none")
        self.assertEqual(rc, 0, out)
        k = rep["kld"]["summary"]
        self.assertEqual(k["count"], 3 * 31)
        self.assertLess(abs(k["kld_mean"]), 2e-4)                  # only the base file's 16-bit rounding
        self.assertEqual(k["same_top"], 1.0)
        self.assertEqual(rep["kld"]["verdict"], "MEASURED")
        self.assertEqual(rep["config"]["source"], "dump")
        self.assertIn("MEASURED", out)
        self.assertIn("Mean    KLD", out)
        # one engine launch per chunk, each on exactly the chunk's tokens, the first half batched
        for c in range(3):
            head = (w / f"chunk{c}.log").read_text().splitlines()[0]
            self.assertIn(f"--tokens-file {w}/chunk{c}.ids", head)
            self.assertIn("--prefill-until 32", head)
            self.assertEqual([int(x) for x in (w / f"chunk{c}.ids").read_text().split()], [int(t) for t in self.tokens[c]])
        self.assertFalse((w / "chunk0.bin").exists())                      # the (large) dump is deleted after scoring

    def test_keep_logits_and_chunk_limit(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none", "--kld-chunks", "2",
                                    "--keep-logits")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["kld"]["summary"]["count"], 2 * 31)
        self.assertTrue((w / "chunk1.bin").exists())
        self.assertFalse((w / "chunk2.ids").exists())

    def test_noise_shows_up_as_kl_and_as_a_verdict(self):
        base = self.make_base("b.kld")
        env = {"FAKE_NOISE": "0.5"}
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none", env=env)
        k = rep["kld"]["summary"]
        self.assertGreater(k["kld_mean"], 0.005)
        self.assertLess(k["same_top"], 1.0)
        # the noise floor between two llama.cpp backends, as `llama-perplexity --kl-divergence` prints it
        floor = self.d / "floor.log"
        floor.write_text("====== Perplexity statistics ======\nMean ln(PPL(Q)/PPL(base))     :   0.000100 ±   0.000500\n\n"
                         "====== KL divergence statistics ======\nMean    KLD:   0.000500 ±   0.000010\n"
                         "Maximum KLD:   0.1\n====== Token probability statistics ======\nSame top p: 99.500 ± 0.100 %\n", encoding="utf-8")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    "--kld-floor", str(floor), env=env)
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["kld"]["verdict"], "FAIL")
        self.assertTrue(any("mean KLD" in r for r in rep["reasons"]), rep["reasons"])
        self.assertIn("KLD VERDICT (provisional", out)
        # a lenient floor (and absolute floors) passes the same run
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    "--kld-floor", str(floor), "--kld-abs-kl", "0.5", "--kld-abs-top1", "1.0", env=env)
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["kld"]["verdict"], "PASS")
        # no noise: it passes against the strict floor too (the PPL change is zero, within 2 standard errors)
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    "--kld-floor", str(floor))
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["kld"]["verdict"], "PASS")
        self.assertIn("noise floor, llama.cpp against itself", out)

    def test_explicit_thresholds_without_a_floor(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none", "--max-kl", "1e-9",
                                    env={"FAKE_NOISE": "0.5"})
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["kld"]["verdict"], "FAIL")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none", "--kld-min-top1", "0.5")
        self.assertEqual((rc, rep["kld"]["verdict"]), (0, "PASS"))

    def test_bos_is_written_over_the_first_token_of_every_chunk(self):
        base = self.make_base("b_bos.kld", bos=7)                          # llama.cpp evaluated the chunks with BOS = 7 in front
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "7")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["config"]["bos"], 7)
        self.assertLess(abs(rep["kld"]["summary"]["kld_mean"]), 2e-4)
        self.assertEqual(int((w / "chunk2.ids").read_text().split()[0]), 7)
        self.assertEqual(int((w / "chunk2.ids").read_text().split()[1]), int(self.tokens[2, 1]))
        # forgetting it is visible: the contexts differ, so the distributions do
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none")
        self.assertGreater(rep["kld"]["summary"]["kld_mean"], 0.5)

    def test_bos_auto_needs_the_gguf(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base))
        self.assertEqual(rc, 2)
        self.assertIn("--kld-bos", out)

    def test_logpos_source_with_full_logits(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg_native.json"), "--ref-kld", str(base), "--kld-bos", "none")
        self.assertEqual(rc, 0, out)
        s = rep["kld"]["summary"]
        self.assertTrue(s["has_kl"], out)
        self.assertEqual(rep["config"]["source"], "logpos")
        splits = [rep["runs"][f"chunk{c}"]["split_at"] for c in range(3)]
        self.assertTrue(all(sp >= self.first for sp in splits))
        self.assertEqual(s["count"], sum(self.N_CTX - 1 - sp for sp in splits))
        self.assertLess(abs(s["kld_mean"]), 2e-4)
        self.assertEqual(s["same_top"], 1.0)
        for c in range(3):
            head = (w / f"chunk{c}.log").read_text().splitlines()[0]
            sp = splits[c]
            tok = G.choose_turn_after([int(t) for t in self.tokens[c]], self.first)[1]
            self.assertIn(f"--turn-token {tok}", head)
            self.assertIn(f"--short-read {self.N_CTX - 1 - sp}", head)         # margin 0: the first part must stay above it
            self.assertIn("--prompt-cache-root 0", head)
            self.assertIn("--serve", head)
            self.assertFalse((w / f"chunk{c}.logits").exists())                  # deleted after scoring
        self.assertTrue(any("one `strata --serve` launch per chunk" in n for n in rep["notes"]))

    def test_logpos_source_noise_is_seen_in_kl(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg_native.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    env={"FAKE_NOISE": "0.5"})
        s = rep["kld"]["summary"]
        self.assertGreater(s["kld_mean"], 0.005)
        self.assertLess(s["same_top"], 1.0)

    def test_logpos_source_without_the_dump_hook_has_no_kl_and_says_so(self):
        base = self.make_base("b.kld")
        floor = self.d / "floor2.log"
        floor.write_text("Mean ln(PPL(Q)/PPL(base))     :   0.000100 ±   0.000500\nMean    KLD:   0.000500 ±   0.000010\n"
                         "Same top p: 99.500 ± 0.100 %\n", encoding="utf-8")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg_native.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    "--kld-floor", str(floor), env={"FAKE_NO_DUMP_HOOK": "1"})
        self.assertEqual(rc, 0, out)
        s = rep["kld"]["summary"]
        self.assertFalse(s["has_kl"])
        self.assertNotIn("kld_mean", s)
        self.assertEqual(s["same_top"], 1.0)
        self.assertLess(abs(s["ln_ppl_ratio"]), 1e-3)
        self.assertTrue(any("KL divergence is NOT available" in n for n in rep["notes"]))
        self.assertTrue(rep["kld"]["verdict_info"].get("partial"))
        self.assertIn("partial", out)
        self.assertIn("target log-probabilities only", out)

    def test_nan_fails(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    env={"FAKE_NAN_AT": "40"})
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["kld"]["summary"]["nonfinite"], 3)
        self.assertTrue(any("non-finite" in r for r in rep["reasons"]))

    def test_engine_crash_is_a_fail_with_the_log(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none",
                                    env={"FAKE_CRASH": "1"})
        self.assertEqual(rc, 1, out)
        self.assertIn("chunk0.log", " ".join(rep["reasons"]))

    def test_a_vocabulary_that_does_not_match_is_refused_unless_trimmed(self):
        # the base file was made for a vocabulary of 80; the fake engine writes 64 logits per position
        rows = [np.random.RandomState(c).normal(size=(31, 80)).astype(np.float32) for c in range(3)]
        p = self.d / "v80.kld"
        K.write_base(p, 64, self.tokens, rows)
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(p), "--kld-bos", "none")
        self.assertEqual(rc, 2)
        self.assertIn("vocabulary", out)

    def test_dry_run_prints_the_chunk_command_and_runs_nothing(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "none", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("--prefill-until 32", out)
        self.assertIn("--dump-logits", out)
        self.assertIn("3 chunk(s)", out)
        self.assertFalse((w / "chunk0.log").exists())
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg_native.json"), "--ref-kld", str(base), "--kld-bos", "none", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("STRATA_LOGITS_DUMP=", out)
        self.assertIn("--turn-token", out)
        self.assertIn("--short-read", out)

    def test_choose_turn_after(self):
        toks = [1] * 10 + [2, 3, 2, 4, 5, 5, 6, 3]
        # from position 10: token 2 occurs again (12), 3 occurs again (17); position 13 holds 4, which occurs once
        self.assertEqual(G.choose_turn_after(toks, 10), (13, 4))
        # nothing unique, but a last occurrence: position 15 (5 occurs at 14 and 15, 15 is the last)
        toks = [1] * 10 + [2, 3, 2, 3, 5, 5, 2, 3]
        p, t = G.choose_turn_after(toks, 10)
        self.assertEqual(toks.index(t, p), p)
        self.assertEqual(max(i for i, x in enumerate(toks) if x == t), p)
        with self.assertRaises(G.HarnessError):
            G.choose_turn_after([1, 2] * 40, 40, 5)

    def test_combinations_are_refused(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--exact")
        self.assertEqual(rc, 2)
        self.assertIn("--exact", out)

    def test_the_first_token_note_and_the_sparse_note(self):
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ref-kld", str(base), "--kld-bos", "7")
        self.assertTrue(any("not the BOS 7" in n for n in rep["notes"]))
        self.assertTrue(any("sparse" in n for n in rep["notes"]))


class ExactFlow(FlowBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        ids = np.random.RandomState(5).randint(0, V, size=3000)
        (cls.d / "ids.txt").write_text(" ".join(map(str, ids)))
        lp_ids = list(np.random.RandomState(11).randint(0, 60, size=3000))
        lp_ids[2650] = 63
        (cls.d / "ids_lp.txt").write_text(" ".join(map(str, lp_ids)))

    def go(self, cfg, ids, *extra, env=None, name="wx"):
        return self.main("--engine-config", str(self.d / cfg), "--ids-file", str(self.d / ids), "--tail", "300", "--exact", *extra, env=env, name=name)

    def test_dump_source_identical(self):
        rc, rep, out, w = self.go("cfg.json", "ids.txt", "--ref-env", "")        # the same configuration twice: determinism
        self.assertEqual(rc, 0, out)
        e = rep["exact"]
        self.assertTrue(e["identical"])
        self.assertEqual((e["kind"], e["rows"], e["rows_differing"], e["max_ulp"]), ("logits", 300, 0, 0))
        self.assertIn("RESULT: IDENTICAL", out)
        self.assertIn("bit for bit", out)
        self.assertIn("determinism check", out)

    def test_dump_source_differs_and_reports_where_and_how_much(self):
        rc, rep, out, w = self.go("cfg.json", "ids.txt", env={"FAKE_NOISE": "1e-4"})
        self.assertEqual(rc, 1, out)
        e = rep["exact"]
        self.assertFalse(e["identical"])
        self.assertEqual(e["rows_differing"], 300)                                # the noise touches every row
        self.assertEqual(e["first_diff_position"], 2700)
        self.assertGreater(e["max_ulp"], 1)
        self.assertIn("RESULT: DIFFERENT", out)
        self.assertIn("largest distance", out)
        self.assertEqual(rep["reasons"][0].split()[0], "300")

    def test_one_row_differing_is_found(self):
        rc, rep, out, w = self.go("cfg.json", "ids.txt", env={"FAKE_NAN_AT": "2800"})   # a NaN in ONE row of the candidate
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["exact"]["rows_differing"], 1)
        self.assertEqual(rep["exact"]["first_diff_position"], 2800)
        self.assertEqual(rep["exact"]["nan_values"], 1)

    def test_logpos_source_with_full_logits_is_a_bitwise_statement(self):
        rc, rep, out, w = self.go("cfg_native.json", "ids_lp.txt", "--ref-env", "", name="wxl")
        self.assertEqual(rc, 0, out)
        e = rep["exact"]
        self.assertEqual(e["kind"], "logits")
        self.assertEqual(e["values_per_row"], V)
        self.assertEqual(e["rows"], 3000 - 1 - 2650)
        self.assertTrue((w / "ref.logits").exists() and (w / "cand.logits").exists())
        rc, rep, out, w = self.go("cfg_native.json", "ids_lp.txt", env={"FAKE_NOISE": "1e-6"}, name="wxl2")
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["exact"]["kind"], "logits")
        self.assertGreater(rep["exact"]["rows_differing"], 0)

    def test_logpos_source_without_the_hook_compares_printed_log_probabilities_and_says_so(self):
        rc, rep, out, w = self.go("cfg_native.json", "ids_lp.txt", "--ref-env", "", env={"FAKE_NO_DUMP_HOOK": "1"}, name="wxn")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["exact"]["kind"], "logpos")
        self.assertIn("NOT a bitwise statement", out)
        self.assertTrue(any("no full logits" in n for n in rep["notes"]))
        rc, rep, out, w = self.go("cfg_native.json", "ids_lp.txt", env={"FAKE_NO_DUMP_HOOK": "1", "FAKE_NOISE": "0.01"}, name="wxn2")
        self.assertEqual(rc, 1, out)
        self.assertGreater(rep["exact"]["max_abs_diff"], 0.0)

    def test_greedy_mode_cannot_be_exact(self):
        rc, rep, out, w = self.go("cfg.json", "ids.txt", "--mode", "greedy")
        self.assertEqual(rc, 2)
        self.assertIn("teacher", out)

    def test_reuse_ref_takes_the_reference_from_an_earlier_workdir(self):
        rc, rep, out, w = self.go("cfg.json", "ids.txt", "--ref-env", "", name="wbase")            # the baseline, and the determinism row
        self.assertEqual(rc, 0, out)
        before = (w / "ref.log").read_text()
        # the candidate differs (noise only reaches a run whose environment does not say STRATA_VOLTA_ATTN=0), the reference is not run again
        rc, rep2, out2, w2 = self.main("--engine-config", str(self.d / "cfg.json"), "--ids-file", str(self.d / "ids.txt"), "--tail", "300", "--exact",
                                       "--reuse-ref", str(w), env={"FAKE_NOISE": "1e-4"}, name="wreuse")
        self.assertEqual(rc, 1, out2)
        self.assertFalse(rep2["exact"]["identical"])
        self.assertEqual((w2 / "ref.log").read_text(), before)
        self.assertEqual(rep2["runs"]["reference"]["reused_from"], str(w))
        self.assertIn("(--reuse-ref)", out2)
        # a different prompt is refused
        (self.d / "other.txt").write_text(" ".join(map(str, np.random.RandomState(8).randint(0, V, size=3000))))
        rc, rep3, out3, w3 = self.main("--engine-config", str(self.d / "cfg.json"), "--ids-file", str(self.d / "other.txt"), "--tail", "300", "--exact",
                                       "--reuse-ref", str(w), name="wreuse2")
        self.assertEqual(rc, 2)
        self.assertIn("different prompt", out3)

    def test_ref_exe_and_cand_exe(self):
        shutil.copy(self.d / "fake_strata.py", self.d / "fake_upstream.py")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg.json"), "--ids-file", str(self.d / "ids.txt"), "--tail", "300", "--exact",
                                    "--ref-exe", str(self.d / "fake_upstream.py"), "--ref-env", "")
        self.assertEqual(rc, 0, out)
        self.assertIn("fake_upstream.py", (w / "ref.log").read_text().splitlines()[0])
        self.assertNotIn("fake_upstream.py", (w / "cand.log").read_text().splitlines()[0])
        self.assertEqual(rep["runs"]["reference"]["exe"], str(self.d / "fake_upstream.py"))
        self.assertNotIn("determinism check", out)                                # two binaries: not a determinism check
        self.assertIn("fake_upstream.py with", rep["config"]["reference"])

    def test_dry_run_shows_each_side_its_own_engine(self):
        shutil.copy(self.d / "fake_strata.py", self.d / "fake_upstream.py")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg_native.json"), "--ids-file", str(self.d / "ids_lp.txt"), "--tail", "300",
                                    "--exact", "--dry-run", "--ref-exe", str(self.d / "fake_upstream.py"))
        self.assertEqual(rc, 0)
        ref_line = [ln for ln in out.splitlines() if ln.startswith("[reference]")][0]
        cand_line = [ln for ln in out.splitlines() if ln.startswith("[candidate]")][0]
        self.assertIn("fake_upstream.py", ref_line)
        self.assertNotIn("fake_upstream.py", cand_line)
        self.assertIn("STRATA_LOGITS_DUMP=", cand_line)


class PrintConfig(FlowBase):
    def test_print_config_lines(self):
        gg = self.d / "model-00001-of-00002.gguf"
        gg.write_bytes(b"x")
        cfg = dict(self.cfg, args=["--pack", "pack_native/full", "--native", str(gg), "--spec", "4", "--mtp", "m", "--expert-cache", "auto"])
        (self.d / "cfg_print.json").write_text(json.dumps(cfg))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = G.main(["--engine-config", str(self.d / "cfg_print.json"), "--print-config"])
        self.assertEqual(rc, 0)
        kv = dict(ln.split("=", 1) for ln in out.getvalue().splitlines())
        self.assertEqual(kv["NATIVE_PACK"], "1")
        self.assertEqual(kv["GGUF"], str(gg))
        self.assertEqual(kv["EXE"], str(self.d / "fake_strata.py"))
        self.assertEqual(kv["PACK"], str(self.d / "pack_native" / "full"))
        self.assertEqual(kv["SPEC"], "4")
        self.assertFalse((self.d / "golden_out").exists())          # --print-config makes no workdir

    def test_bos_auto_reads_the_engine_args_gguf(self):
        b = BosTests()
        b.setUp()
        try:
            gg = b.gguf({"tokenizer.ggml.model": (8, "gpt2")})
        finally:
            pass
        cfg = dict(self.cfg, args=["--pack", "pack/full", "--native", str(gg), "--prefill", "auto", "--max-context", "4096"])
        (self.d / "cfg_gguf.json").write_text(json.dumps(cfg))
        base = self.make_base("b.kld")
        rc, rep, out, w = self.main("--engine-config", str(self.d / "cfg_gguf.json"), "--ref-kld", str(base))
        b.tearDown()
        self.assertEqual(rc, 0, out)
        self.assertIsNone(rep["config"]["bos"])
        self.assertTrue(any("adds none" in n for n in rep["notes"]))


if __name__ == "__main__":
    unittest.main()
