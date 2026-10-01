"""Tests for tools/ds41/ds1_e2e.py: the mini end-to-end fixture and the checker the C++ ctest calls (exit 0 pass / 1 fail / 2 could not run)."""
import contextlib
import hashlib
import io
import json
import pathlib
import shutil
import unittest

import numpy as np

import ds1_compare as C
import ds1_e2e as E
import ds1_testlib as T
from ref.ds41 import trace_io as TI

N_PROMPT, MAX_NEW = 4, 12


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = E.main(argv)
    return rc, out.getvalue(), err.getvalue()


def sha(paths):
    return [hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sorted(paths)]


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = T.root() / "e2e_fixture"
        cls.fx = E.prepare(cls.dir, n_prompt=N_PROMPT, max_new=MAX_NEW)
        cls.model = T.model("int8-kv", "float64", max_seq_len=96)
        cls.model = C.load_oracle_model(cls.dir / cls.fx["gguf"], C.quant_from_name("int8-kv"), dtype="float64", max_seq_len=96)
        # a stand-in for the engine: the float64 oracle decoding greedily on its own tokens (same mathematics, different rounding, its own continuation)
        gen = TI.greedy_continue(cls.model, cls.fx["prompt"], MAX_NEW)
        cls.tokens = cls.fx["prompt"] + gen
        d = T.root() / "e2e_engine_good"
        shutil.rmtree(d, ignore_errors=True)
        TI.run_oracle_trace(cls.model, cls.tokens[:-1], d, n_prompt=N_PROMPT, geometry="MiniGeom", extra_meta={"producer": "engine"})
        cls.set_tokens(d, cls.tokens)
        cls.good = d

    @staticmethod
    def set_tokens(d, tokens):
        j = json.loads((pathlib.Path(d) / "trace.json").read_text())
        j["tokens"] = list(tokens)
        j["n_prompt"] = N_PROMPT
        (pathlib.Path(d) / "trace.json").write_text(json.dumps(j))

    def engine_variant(self, name, mutate=None, drop=None, tokens=None):
        d = T.root() / name
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(self.good, d)
        for (stage, pos, layer) in drop or ():
            (d / TI.stage_filename(stage, pos, layer)).unlink()
        for (stage, pos, layer), fn in (mutate or {}).items():
            p = d / TI.stage_filename(stage, pos, layer)
            a = np.load(p)
            np.save(p, np.ascontiguousarray(fn(a), dtype=a.dtype))
        if tokens is not None:
            self.set_tokens(d, tokens)
        return d

    # ------------------------------------------------------------------------------------------------ prepare
    def test_fixture_contents(self):
        fx = self.fx
        self.assertEqual((fx["n_prompt"], fx["max_new"], fx["geometry"]), (N_PROMPT, MAX_NEW, "MiniGeom"))
        self.assertEqual(len(fx["oracle_tokens"]), N_PROMPT + MAX_NEW)
        self.assertEqual(fx["oracle_tokens"][:N_PROMPT], fx["prompt"])
        self.assertNotIn(1, fx["oracle_tokens"][N_PROMPT:])                                  # the EOS id never appears in the continuation
        self.assertTrue((self.dir / fx["gguf"]).is_file())
        ref = TI.Trace(self.dir / "oracle")
        self.assertEqual(ref.tokens, fx["oracle_tokens"])
        self.assertEqual(len(ref.positions()), N_PROMPT + MAX_NEW)
        self.assertEqual(ref.quant(), C.quant_from_name("int8-kv"))
        self.assertEqual(ref.meta["geometry"], "MiniGeom")

    def test_prepare_is_idempotent_and_deterministic(self):
        again = E.prepare(self.dir, n_prompt=N_PROMPT, max_new=MAX_NEW)
        self.assertEqual(again, self.fx)                                                      # kept, not rebuilt
        other = T.root() / "e2e_fixture_2"
        shutil.rmtree(other, ignore_errors=True)
        fx2 = E.prepare(other, n_prompt=N_PROMPT, max_new=MAX_NEW)
        self.assertEqual(sha(sorted((self.dir / "gguf").glob("*.gguf"))), sha(sorted((other / "gguf").glob("*.gguf"))))      # byte-identical GGUF
        self.assertEqual(fx2["prompt"], self.fx["prompt"])
        self.assertEqual(fx2["oracle_tokens"], self.fx["oracle_tokens"])
        a, b = TI.Trace(self.dir / "oracle"), TI.Trace(other / "oracle")
        for p in (0, N_PROMPT + MAX_NEW - 1):
            np.testing.assert_array_equal(a.get("logits", p), b.get("logits", p))

    def test_args_command(self):
        rc, out, _ = run(["args", "--fixture", str(self.dir)])
        self.assertEqual(rc, 0)
        self.assertEqual(out.strip(), E.engine_args(self.fx, self.dir))
        self.assertIn(f"--max-new {MAX_NEW}", out)
        self.assertIn("--tokens " + ",".join(str(t) for t in self.fx["prompt"]), out)
        self.assertIn(".gguf", out)

    # ------------------------------------------------------------------------------------------------ check
    def test_a_correct_engine_passes_with_exit_0(self):
        rc, out, err = run(["check", "--fixture", str(self.dir), "--engine", str(self.good)])
        self.assertEqual(rc, 0, out + err)
        self.assertIn("RESULT: PASS", out)
        self.assertIn("PASS  layer mode", out)
        self.assertIn("PASS  greedy", out)

    def test_exactly_the_fixtures_own_oracle_trace_passes_and_uses_the_cached_oracle(self):
        d = T.root() / "e2e_engine_is_oracle"
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(self.dir / "oracle", d)
        j = json.loads((d / "trace.json").read_text())
        j["producer"] = "engine"
        (d / "trace.json").write_text(json.dumps(j))
        for p in (d / TI.stage_filename("logits", N_PROMPT + MAX_NEW - 1),):
            p.unlink()                                                                           # the last token has no forward pass in an engine
        v = E.check(self.dir, d)
        self.assertTrue(v.ok, v.text(True))
        self.assertTrue(any("identical to the oracle's" in name for name, ok, _ in v.checks))

    def test_json_report(self):
        out = T.root() / "e2e_report.json"
        rc, _, _ = run(["check", "--fixture", str(self.dir), "--engine", str(self.good), "--json", str(out)])
        self.assertEqual(rc, 0)
        j = json.loads(out.read_text())
        self.assertTrue(j["ok"])
        self.assertIn("layer mode (stage-isolated replay)", j["reports"])
        self.assertIn("whole run (the oracle's own trajectory)", j["reports"])

    def test_an_injected_error_fails_with_exit_1_and_names_the_stage(self):
        for stage, pos, layer in (("attn_out", 5, 3), ("ffn_out", 7, 6), ("kv_win", 3, 2), ("latent", 5, 2), ("logits", 9, None)):
            with self.subTest(stage=stage):
                d = self.engine_variant(f"e2e_bad_{stage}", mutate={(stage, pos, layer): T.noise(0.4)})
                rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(d)])
                self.assertEqual(rc, 1, out)
                self.assertIn("RESULT: FAIL", out)
                self.assertIn(f"FIRST FAILURE: position {pos}, " + (f"layer {layer}, " if layer is not None else "") + f"stage {stage}", out)

    def test_incomplete_traces_fail(self):
        d = self.engine_variant("e2e_incomplete", drop=[("ffn_out", 6, 4), ("index_k", 5, 2)])
        rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(d)])
        self.assertEqual(rc, 1)
        self.assertIn("MISSING in the engine trace", out)
        self.assertIn("ffn_out", out)
        d = self.engine_variant("e2e_nopos", drop=[("embed", 9, None)])
        rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(d)])
        self.assertEqual(rc, 1)

    def test_wrong_prompt_or_token_count_fails(self):
        toks = list(self.tokens)
        toks[1] = (toks[1] + 1) % 500 + 3
        d = self.engine_variant("e2e_prompt", tokens=toks)
        rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(d)])
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  trace: tokens start with the prompt", out)
        d = self.engine_variant("e2e_short", tokens=self.tokens[:-3])
        rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(d)])
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  trace: the engine generated the requested number of tokens", out)

    def test_the_engine_must_follow_its_own_logits(self):
        toks = list(self.tokens)
        d_pos = N_PROMPT + 2
        toks[d_pos] = (toks[d_pos] + 7) % 500 + 3                                  # a token the engine's logits did not choose
        d = self.engine_variant("e2e_not_argmax", tokens=toks)
        rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(d)])
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  greedy: every generated token is the argmax of the engine's own logits", out)

    def test_no_trace_is_a_failure_and_no_fixture_is_cannot_run(self):
        empty = T.root() / "e2e_empty"
        shutil.rmtree(empty, ignore_errors=True)
        empty.mkdir()
        rc, out, _ = run(["check", "--fixture", str(self.dir), "--engine", str(empty)])
        self.assertEqual(rc, 1)
        self.assertIn("the engine wrote a trace", out)
        rc, _, err = run(["check", "--fixture", str(T.root() / "no_such_fixture"), "--engine", str(self.good)])
        self.assertEqual(rc, 2)
        self.assertIn("cannot run", err)
        broken = T.root() / "e2e_broken_fixture"
        shutil.rmtree(broken, ignore_errors=True)
        broken.mkdir()
        (broken / "fixture.json").write_text(json.dumps({**self.fx, "gguf": "gone.gguf"}))
        rc, _, err = run(["check", "--fixture", str(broken), "--engine", str(self.good)])
        self.assertEqual(rc, 2)

    # ------------------------------------------------------------------------------------------------ the greedy rule in isolation
    def test_greedy_rule_unexplained_divergence_fails_noise_explained_passes(self):
        ref = TI.Trace(self.dir / "oracle")
        d_pos = N_PROMPT + 3
        o_tok = self.fx["oracle_tokens"][d_pos]
        other = (o_tok + 11) % 500 + 3
        toks = list(self.fx["oracle_tokens"])
        toks[d_pos] = other
        # the engine's logits at position d_pos-1 make `other` the argmax by a wide margin: no near-tie
        eng_dir = T.engine_copy(ref, "e2e_greedy_rule", mutate={("logits", d_pos - 1, None): lambda a: np.where(np.arange(a.size) == other, a.max() + 5.0, a).astype(a.dtype)})
        eng = TI.Trace(eng_dir.path)
        eng.meta["tokens"] = toks
        v = E.Verdict(True)
        E.greedy_check(eng, ref, self.fx, None, v)
        self.assertFalse(v.ok)
        self.assertIn("no numerical deviation before it", v.text(True))
        # the same difference with a numerical deviation reported before it is explained
        rep = C.Report("full")
        rep.onsets.append((d_pos - 2, 3, "ffn_out", "rms_rel 3e-4"))
        v = E.Verdict(True)
        E.greedy_check(eng, ref, self.fx, rep, v)
        self.assertTrue(v.ok, v.text(True))
        self.assertIn("amplify", v.text(True))
        # ... but not one that comes after it
        rep = C.Report("full")
        rep.onsets.append((d_pos + 5, 3, "ffn_out", "late"))
        v = E.Verdict(True)
        E.greedy_check(eng, ref, self.fx, rep, v)
        self.assertFalse(v.ok)
        # a near-tie of the oracle's logits explains it on its own
        eng2_dir = T.engine_copy(ref, "e2e_greedy_tie", mutate={("logits", d_pos - 1, None): lambda a: a})
        ref2_dir = T.engine_copy(ref, "e2e_greedy_tie_ref", mutate={("logits", d_pos - 1, None): lambda a: np.where(np.arange(a.size) == other, a[o_tok] - 1e-7, a).astype(a.dtype)})
        eng2 = TI.Trace(eng2_dir.path)
        eng2.meta["tokens"] = toks
        eng2_logits = np.load(eng2_dir.path / TI.stage_filename("logits", d_pos - 1))
        eng2_logits[other] = eng2_logits[o_tok] + 1e-7                                       # the engine's argmax flips on a 1e-7 difference
        np.save(eng2_dir.path / TI.stage_filename("logits", d_pos - 1), eng2_logits)
        v = E.Verdict(True)
        E.greedy_check(eng2, TI.Trace(ref2_dir.path), self.fx, None, v)
        self.assertTrue(any("near-tie" in t for _, ok, t in v.checks if ok), v.text(True))


if __name__ == "__main__":
    unittest.main()
