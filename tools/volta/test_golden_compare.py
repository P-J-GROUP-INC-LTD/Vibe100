"""tools/volta/test_golden_compare.py - tests of golden_compare.py: the metric math on small hand-computed cases, the
logits-file reader (header, short files, .npy), the engine-argument rewriting, the parser of the engine's own output
lines, and the whole flow against a FAKE ENGINE that imitates `strata`'s command line and --dump-logits semantics
(src/program/generate.cpp) well enough to exercise teacher mode, greedy mode, `--prefill-until`, native packs, a NaN, a
crash and noise.

    python3 -m unittest tools.volta.test_golden_compare        (from the repository root; needs numpy)
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "prompts"))
import golden_compare as G  # noqa: E402
import make_long_prompt as M  # noqa: E402

V = 64   # the fake model's vocabulary

# A stand-in for `strata`.  Same flags as generate.cpp; the logits at position p are a deterministic function of the
# context (the tokens up to p), the "reference" environment (STRATA_VOLTA_ATTN=0) adds no noise and the candidate adds
# FAKE_NOISE * N(0,1) (also a function of the context, so reruns are bit-identical).  The dump follows the engine's rules:
# header (n_vocab, row_count(total, stride)), then only the rows of the positions the token loop visits.
FAKE_ENGINE = textwrap.dedent('''
    import hashlib, os, sys
    import numpy as np
    V = 64
    VAL = {"--pack", "--native", "--ple-gguf", "--expert-profile", "--expert-cache", "--prefill", "--spec", "--spec-min-p",
           "--mtp", "--max-context", "--kv", "--tokens-file", "--tokens", "--max-new", "--dump-logits", "--logits-stride",
           "--prefill-until", "--pcie-frac", "--adapt-swaps", "--suffix-draft", "--seed", "--turn-token", "--short-read",
           "--prompt-cache-root", "--numa"}
    args = sys.argv[1:]
    SERVE = "--serve" in args
    if "--expert-cache" in args:
        v = args[args.index("--expert-cache") + 1]
        print("strata generate: expert cache %s slots, 12.34 GiB of VRAM; policy is" % (1234 if v == "auto" else v), file=sys.stderr, flush=True)
    opt, i = {}, 0
    while i < len(args):
        a = args[i]
        if a in VAL:
            opt[a] = args[i + 1]; i += 2
        else:
            opt[a] = True; i += 1
    if SERVE:
        toks = []
        n, max_new = 0, 1
    else:
        toks = [int(x) for x in open(opt["--tokens-file"]).read().split()]
        n, max_new = len(toks), int(opt["--max-new"])
    stride = int(opt.get("--logits-stride", 1))
    if stride > 1 and max_new != 1:
        print("strata generate: --logits-stride > 1 requires --max-new 1 and --dump-logits", file=sys.stderr); sys.exit(2)
    pack = opt.get("--pack", "pack/full")
    native = os.path.exists(os.path.join(pack, "native_experts.txt"))
    pf = opt.get("--prefill", "0")
    prefill = int(pf) if pf.isdigit() else (8192 if pf == "auto" else 0)
    until = int(opt.get("--prefill-until", 0))
    total = n - 1 + max_new
    if prefill > 0 and n > 1:
        n_batched = until if 0 < until < n - 1 else n - 1
    else:
        n_batched = 0
    spec = int(opt.get("--spec", 0))
    if os.environ.get("FAKE_CRASH") and os.environ.get("STRATA_VOLTA_ATTN") != "0":
        print("CUDA error: unspecified launch failure", file=sys.stderr); sys.exit(1)
    noise = float(os.environ.get("FAKE_NOISE", "0")) if os.environ.get("STRATA_VOLTA_ATTN") != "0" else 0.0
    if os.environ.get("STRATA_IQ512") == "1":       # the AVX-512 tier of the expert rows sums in another order than the AVX2 one
        noise = float(os.environ.get("FAKE_NOISE_IQ512", "0")) or noise
    if "--numa" in args and args[args.index("--numa") + 1] == "mirror" and os.environ.get("FAKE_MIRROR_LINE"):
        print("strata generate: NUMA: expert arena MIRRORED: primary 1.00 GiB on node 1 + 1 replica", file=sys.stderr, flush=True)
    nan_at = int(os.environ.get("FAKE_NAN_AT", "-1")) if os.environ.get("STRATA_VOLTA_ATTN") != "0" else -1
    diverge_at = int(os.environ.get("FAKE_DIVERGE_AT", "-1")) if os.environ.get("STRATA_VOLTA_ATTN") != "0" else -1

    def logits(ctx):
        seed = int.from_bytes(hashlib.sha256(repr(tuple(ctx)).encode()).digest()[:4], "little")
        z = np.random.RandomState(seed).normal(size=V) * 3.0
        if noise:
            z = z + noise * np.random.RandomState(seed ^ 0x5bd1e995).normal(size=V)
        return z.astype("<f4")

    def selected(p):
        return p % stride == 0 or p == total - 1

    def lsm(z):
        z = z.astype("f8")
        m = z.max()
        return z - (m + np.log(np.exp(z - m).sum()))

    if SERVE:
        ctx_n = int(opt.get("--max-context", 4096))
        print("INFO engine=fake kv=int8", flush=True)
        print("READY %d stop" % ctx_n, flush=True)
        S = 4
        for line in sys.stdin:
            line = line.strip()
            if line == "QUIT":
                break
            if not line.startswith("GEN "):
                continue
            _, mn, idtxt = line.split(" ", 2)
            ids = [int(x) for x in idtxt.split(",")]
            n, mn = len(ids), int(mn)
            turn, short = int(opt.get("--turn-token", 248045)), int(opt.get("--short-read", 64))
            lp_path = os.environ.get("STRATA_LOGPOS")
            lpf = open(lp_path, "ab") if lp_path else None
            dump_path = os.environ.get("STRATA_LOGITS_DUMP") if os.environ.get("FAKE_NO_DUMP_HOOK") is None else None
            dumpf = open(dump_path, "ab") if dump_path else None
            if dumpf is not None and dumpf.tell() == 0:
                np.array([V, 2**31 - 1], dtype="<i4").tofile(dumpf)
            turn_at = -1
            for i in range(n - 1, 0, -1):
                if ids[i] == turn:
                    turn_at = i
                    break
            at = 0
            for to in (turn_at, n - 1):
                if to <= at:
                    continue
                if to - at <= short:
                    for q in range(at, to, S):
                        for p in range(q, min(q + S, to)):
                            z = logits(ids[: p + 1]).astype("f8")
                            if p == nan_at:
                                z[3] = np.nan
                            tgt = ids[p + 1]
                            good = bool(np.isfinite(z).all())
                            l = lsm(z) if good else z
                            top = int(np.nanargmax(z))
                            lp = float(l[tgt]) if good else float("nan")
                            tl = float(l[top]) if good else float("nan")
                            row = "%d\\t%d\\t%.9f\\t%d\\t%.9f\\t%d\\t%.9f\\t%.9f\\n" % (p, tgt, lp, top, tl, int(top == tgt), -5.0, lp)
                            if dumpf:
                                z.astype("<f4").tofile(dumpf)
                            if lpf:
                                lpf.write(row.encode())
                        if lpf:
                            lpf.flush()
                else:
                    print("PP %d %d %.0f %.1f" % (to, n, 500.0, 1000.0 * (to - at) / 500.0), flush=True)
                at = to
            if lpf:
                lpf.close()
            if dumpf:
                dumpf.close()
            ctx = list(ids)
            for j in range(mn):
                z = logits(ctx)
                nxt = int(np.argmax(z))
                ctx.append(nxt)
                print("T %d" % nxt, flush=True)
            print("DONE %d %d %.1f %.1f length 0 0 0" % (mn, n, 800.0, 100.0), flush=True)
        sys.exit(0)

    out = open(opt["--dump-logits"], "wb")
    rows = sum(1 for p in range(total) if selected(p))
    np.array([V, rows], dtype="<i4").tofile(out)
    ctx = list(toks)
    produced = []
    loop_visits = not native          # the token loop is skipped for a native pack
    first_pos = n_batched
    last = total
    if loop_visits and spec > 0 and prefill > 0 and not opt.get("--spec-plain"):
        last = n                      # speculative decoding takes over after the first generated token
    if loop_visits:
        for p in range(first_pos, last):
            lg = logits(ctx[: p + 1])
            if p == nan_at:
                lg[3] = np.nan
            if p >= n - 1 and p - (n - 1) >= diverge_at >= 0:
                lg[(int(lg.argmax()) + 1) % V] += 100.0     # a different greedy pick from here on
            if selected(p):
                lg.tofile(out)
            if p >= n - 1:
                nxt = int(np.argmax(lg))
                produced.append(nxt)
                ctx.append(nxt)
    else:
        for j in range(max_new):      # native: verify windows produce the tokens, no rows are written
            lg = logits(ctx)
            nxt = int(np.argmax(lg))
            produced.append(nxt); ctx.append(nxt)
    if loop_visits and last < total:
        for j in range(len(produced), max_new):
            lg = logits(ctx)
            nxt = int(np.argmax(lg)); produced.append(nxt); ctx.append(nxt)
    out.close()
    print("prompt  : " + " ".join(map(str, toks[:8])))
    print("output  : " + " ".join(map(str, produced)))
    print("%-24s %d tokens in %.1f ms  ->  %.2f tok/s" % ("decode", len(produced), 100.0, 10.0))
    if n > 1:
        print("%-24s %d tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)" % ("prefill", n - 1, 500.0, 1000.0 * (n - 1) / 500.0, 500.0))
''')


def toks(n: int, seed: int = 0) -> list[int]:
    return list(np.random.RandomState(seed).randint(0, V, size=n))


def write_logits(path: Path, rows: np.ndarray, header_rows: int | None = None) -> None:
    with open(path, "wb") as f:
        np.array([rows.shape[1], rows.shape[0] if header_rows is None else header_rows], dtype="<i4").tofile(f)
        rows.astype("<f4").tofile(f)


class MathTests(unittest.TestCase):
    def test_log_softmax_matches_definition(self):
        x = np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 1000.0]])
        lsm = G.log_softmax(x)
        self.assertTrue(np.allclose(np.exp(lsm).sum(axis=1), 1.0))
        self.assertAlmostEqual(lsm[0, 2], 3 - math.log(math.exp(1) + math.exp(2) + math.exp(3)), places=12)
        self.assertTrue(np.allclose(lsm[1], -math.log(3.0)))      # no overflow at 1000

    def test_compare_rows_hand_computed(self):
        d = Path(tempfile.mkdtemp())
        try:
            ref = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 5.0]], dtype=np.float32)
            cand = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 1.5], [1.0, 0.0, 5.0]], dtype=np.float32)   # row 1 flips the top-1
            write_logits(d / "r.bin", ref)
            write_logits(d / "c.bin", cand)
            r = G.read_logits(d / "r.bin").place([0, 1, 2])
            c = G.read_logits(d / "c.bin").place([0, 1, 2])
            m = G.compare_rows(r, c, [0, 1, 2], {0: 0, 1: 1, 2: 2})
            self.assertEqual((m["rows_scored"], m["top1_agree"]), (3, 2))
            self.assertAlmostEqual(m["top1"], 2 / 3)
            self.assertAlmostEqual(m["max_abs_dlogit"], 2.0)
            lr, lc = G.log_softmax(ref.astype(np.float64)), G.log_softmax(cand.astype(np.float64))
            kl = (np.exp(lr) * (lr - lc)).sum(axis=1)
            self.assertAlmostEqual(m["kl_mean"], kl.mean(), places=12)
            self.assertAlmostEqual(m["kl_max"], kl[1], places=12)
            self.assertEqual(kl[0], 0.0)
            ppl_r = math.exp(-(lr[0, 0] + lr[1, 1] + lr[2, 2]) / 3)
            ppl_c = math.exp(-(lc[0, 0] + lc[1, 1] + lc[2, 2]) / 3)
            self.assertAlmostEqual(m["ppl_ref"], ppl_r, places=12)
            self.assertAlmostEqual(m["ppl_cand"], ppl_c, places=12)
            self.assertAlmostEqual(m["ppl_rel"], abs(ppl_c - ppl_r) / ppl_r, places=12)
            self.assertAlmostEqual(m["flip_margin_median"], 2.0)      # ref row 1: 3.0 - 1.0
            self.assertEqual(m["flipped_positions"], [1])
            self.assertEqual((m["nonfinite_cand"], m["nonfinite_ref"]), (0, 0))
        finally:
            shutil.rmtree(d)

    def test_identical_is_perfect_and_nan_is_counted(self):
        d = Path(tempfile.mkdtemp())
        try:
            rows = np.random.RandomState(1).normal(size=(5, 16)).astype(np.float32)
            bad = rows.copy()
            bad[2, 7] = np.nan
            bad[4, 0] = np.inf
            write_logits(d / "a.bin", rows)
            write_logits(d / "b.bin", bad)
            a = G.read_logits(d / "a.bin").place(range(5))
            same = G.read_logits(d / "a.bin").place(range(5))
            b = G.read_logits(d / "b.bin").place(range(5))
            m = G.compare_rows(a, same, range(5), {p: p for p in range(5)})
            self.assertEqual((m["top1"], m["kl_max"], m["max_abs_dlogit"], m["ppl_rel"]), (1.0, 0.0, 0.0, 0.0))
            m = G.compare_rows(a, b, range(5), {p: p for p in range(5)})
            self.assertEqual(m["nonfinite_cand"], 2)
            self.assertEqual(m["rows_scored"], 3)           # the two bad rows are left out of the other statistics
            ok, why = G.verdict(m, min_top1=0.99, max_ppl_rel=0.02)
            self.assertFalse(ok)
            self.assertIn("non-finite", why[0])
        finally:
            shutil.rmtree(d)

    def test_verdict_thresholds(self):
        base = {"nonfinite_cand": 0, "nonfinite_ref": 0, "top1": 0.995, "ppl_rel": 0.01, "kl_mean": 1e-3, "max_abs_dlogit": 0.5}
        self.assertTrue(G.verdict(base, min_top1=0.99, max_ppl_rel=0.02)[0])
        self.assertFalse(G.verdict({**base, "top1": 0.98}, min_top1=0.99, max_ppl_rel=0.02)[0])
        self.assertFalse(G.verdict({**base, "ppl_rel": 0.03}, min_top1=0.99, max_ppl_rel=0.02)[0])
        self.assertFalse(G.verdict(base, min_top1=0.99, max_ppl_rel=0.02, max_kl=1e-4)[0])
        self.assertFalse(G.verdict(base, min_top1=0.99, max_ppl_rel=0.02, max_abs_dlogit=0.1)[0])
        self.assertFalse(G.verdict({**base, "nonfinite_ref": 1}, min_top1=0.99, max_ppl_rel=0.02)[0])
        self.assertTrue(G.verdict(base, min_top1=0.995, max_ppl_rel=0.02)[0])      # the boundary is inclusive

    def test_identical_prefix(self):
        self.assertEqual(G.identical_prefix([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(G.identical_prefix([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(G.identical_prefix([1, 2], [1, 2, 3]), 2)
        self.assertEqual(G.identical_prefix([], [1]), 0)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d)

    def test_selected_positions_is_the_engines_rule(self):
        # logits_selection::selected / row_count in include/strata/program/logits_selection.hpp
        sel = lambda total, stride, first=0: G.selected_positions(total, stride, first).tolist()
        self.assertEqual(sel(10, 1), list(range(10)))
        self.assertEqual(sel(10, 4), [0, 4, 8, 9])
        self.assertEqual(sel(9, 4), [0, 4, 8])                  # the last position is a multiple: once
        self.assertEqual(sel(10, 4, first=5), [8, 9])           # after a batched prefill the token loop starts at 5
        self.assertEqual(sel(10, 1, first=10), [])
        for total in range(1, 40):
            for stride in (1, 2, 3, 7):
                rc = (total - 1) // stride + 1 + ((total - 1) % stride != 0)       # logits_selection::row_count
                self.assertEqual(len(sel(total, stride)), rc, (total, stride))

    def test_header_and_short_file(self):
        rows = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
        write_logits(self.d / "x.bin", rows, header_rows=10)        # the engine after --prefill-until: header counts skipped rows
        lg = G.read_logits(self.d / "x.bin")
        self.assertEqual((lg.n_rows, lg.vocab, lg.header_rows), (3, 8, 10))
        lg.place([7, 8, 9])
        self.assertEqual(lg.row_of(8), 1)
        with self.assertRaises(G.HarnessError):
            G.read_logits(self.d / "x.bin").place([7, 8])           # wrong number of positions: say so
        self.assertTrue(np.array_equal(np.asarray(lg.rows[1]), rows[1]))

    def test_header_only_file_has_zero_rows(self):
        with open(self.d / "h.bin", "wb") as f:
            np.array([V, 5], dtype="<i4").tofile(f)
        lg = G.read_logits(self.d / "h.bin")
        self.assertEqual(lg.n_rows, 0)
        lg.place([])

    def test_malformed_files(self):
        (self.d / "t.bin").write_bytes(b"\x01\x02")
        with self.assertRaises(G.HarnessError):
            G.read_logits(self.d / "t.bin")
        write_logits(self.d / "a.bin", np.zeros((2, 8), np.float32), header_rows=1)       # more rows than the header says
        with self.assertRaises(G.HarnessError):
            G.read_logits(self.d / "a.bin")
        write_logits(self.d / "b.bin", np.zeros((2, 8), np.float32))
        data = (self.d / "b.bin").read_bytes()
        (self.d / "b.bin").write_bytes(data[:-3])                                          # truncated mid-row
        with self.assertRaises(G.HarnessError):
            G.read_logits(self.d / "b.bin")
        with self.assertRaises(G.HarnessError):
            G.read_logits(self.d / "missing.bin")

    def test_npy(self):
        arr = np.random.RandomState(0).normal(size=(4, 8)).astype(np.float32)
        np.save(self.d / "r.npy", arr)
        lg = G.read_logits(self.d / "r.npy")
        self.assertEqual((lg.n_rows, lg.vocab), (4, 8))
        np.save(self.d / "bad.npy", np.zeros(5))
        with self.assertRaises(G.HarnessError):
            G.read_logits(self.d / "bad.npy")


class ArgsTests(unittest.TestCase):
    CFG = ["--pack", "pack/full", "--native", "/m/model-00001-of-00002.gguf", "--ple-gguf", "/m/model-00002-of-00002.gguf",
           "--expert-profile", "p.bin", "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
           "--mtp", "/m/rt", "--max-context", "8192", "--kv", "int8", "--serve"]

    def plan(self, **kw):
        d = dict(mode="teacher", n_tokens=33000, max_new=1, stride=1, prefill_until=32488, native=False)
        d.update(kw)
        return G.Plan(**d)

    def build(self, plan, **kw):
        k = dict(tokens_file="t.ids", dump="o.bin", fixed_experts=True, keep_spec=False)
        k.update(kw)
        return G.build_engine_args(self.CFG, plan, **k)

    def test_teacher_prefill_until(self):
        p = self.plan()
        a = self.build(p)
        self.assertNotIn("--serve", a)
        for f in ("--spec", "--mtp", "--spec-min-p"):
            self.assertNotIn(f, a)                                       # plain decode: the token loop writes the logits
        self.assertEqual(G.flag_value(a, "--prefill"), "auto")
        self.assertEqual(G.flag_value(a, "--prefill-until"), "32488")
        self.assertEqual(G.flag_value(a, "--max-new"), "1")
        self.assertEqual(G.flag_value(a, "--dump-logits"), "o.bin")
        self.assertEqual(G.flag_value(a, "--pcie-frac"), "0")
        self.assertEqual(G.flag_value(a, "--adapt-swaps"), "0")
        self.assertEqual(int(G.flag_value(a, "--max-context")), 33792)   # 33000 + 1 + 8, rounded up to 1024
        self.assertEqual((p.pos_start, p.total), (32488, 33000))
        self.assertIn("--greedy", a)
        self.assertNotIn("--logits-stride", a)

    def test_teacher_token_path(self):
        p = self.plan(prefill_until=0, n_tokens=700)
        a = self.build(p)
        self.assertNotIn("--prefill", a)
        self.assertNotIn("--prefill-until", a)
        self.assertEqual((p.pos_start, p.total), (0, 700))
        self.assertEqual(int(G.flag_value(a, "--max-context")), 8192)    # the config's context is kept when it is enough

    def test_stride_and_keep_spec_and_native(self):
        a = self.build(self.plan(stride=16))
        self.assertEqual(G.flag_value(a, "--logits-stride"), "16")
        a = self.build(self.plan(mode="greedy", prefill_until=None, max_new=256, n_tokens=5000), keep_spec=True)
        self.assertEqual(G.flag_value(a, "--spec"), "4")
        a = self.build(self.plan(mode="greedy", prefill_until=None, max_new=256, native=True))
        self.assertEqual(G.flag_value(a, "--spec"), "4")                 # a native pack cannot run without it
        self.assertEqual(G.flag_value(a, "--mtp"), "/m/rt")

    def test_greedy_prefills_the_whole_prompt(self):
        p = self.plan(mode="greedy", prefill_until=None, max_new=256, n_tokens=5000)
        a = G.build_engine_args(["--pack", "x"], p, tokens_file="t", dump="d", fixed_experts=False, keep_spec=False)
        self.assertEqual(G.flag_value(a, "--prefill"), "auto")          # added: the batched path is what is being tested
        self.assertEqual(p.pos_start, 4999)                              # rows only for the generation positions
        self.assertEqual(p.total, 4999 + 256)

    def test_leftovers_do_not_fight(self):
        p = self.plan()
        base = self.CFG + ["--tokens-file", "old", "--max-new", "99", "--dump-logits", "z", "--prefill-until", "5", "--greedy"]
        a = G.build_engine_args(base, p, tokens_file="t", dump="d", fixed_experts=True, keep_spec=False)
        self.assertEqual(a.count("--max-new"), 1)
        self.assertEqual(G.flag_value(a, "--tokens-file"), "t")
        self.assertEqual(G.flag_value(a, "--prefill-until"), "32488")
        self.assertEqual(a.count("--greedy"), 1)

    def test_env_pairs(self):
        self.assertEqual(G.parse_env_pairs("A=1 B=two  C="), {"A": "1", "B": "two", "C": ""})
        self.assertEqual(G.parse_env_pairs(""), {})
        with self.assertRaises(G.HarnessError):
            G.parse_env_pairs("A")


class LogposTests(unittest.TestCase):
    def test_choose_split_prefers_a_unique_token_near_k0(self):
        toks_ = [1, 2, 3, 4] * 100 + [7] + [1, 2, 3, 4] * 100 + [9] + [1, 2, 3, 4] * 10
        k, t = G.choose_split(toks_, len(toks_) - 30)
        self.assertEqual((k, t), (801, 9))                     # looking down from k0: 9 occurs once
        k, t = G.choose_split(toks_, 700)
        self.assertEqual((k, t), (400, 7))
        with self.assertRaises(G.HarnessError):
            G.choose_split([1, 2] * 50, 60)

    def test_choose_split_falls_back_to_a_last_occurrence(self):
        toks_ = [5, 6, 5, 6, 5, 6, 8, 5, 6, 8, 5, 6, 5, 6]
        # 8 occurs twice, so nothing is unique; the last 8 (position 9) is the last occurrence of its token
        k, t = G.choose_split(toks_, 10, search=100)
        self.assertEqual(t, 8)
        self.assertEqual(k, 9)

    def test_read_logpos_and_compare(self):
        d = Path(tempfile.mkdtemp())
        try:
            (d / "a.logpos").write_text(
                "10\t5\t-1.000000000\t5\t-1.000000000\t1\t-9.0\t-1.0\n"
                "11\t6\t-2.000000000\t3\t-0.500000000\t0\t-9.0\t-2.0\n"
                "12\t7\t-0.100000000\t7\t-0.100000000\t1\t-9.0\t-0.1\n")
            (d / "b.logpos").write_text(
                "10\t5\t-1.100000000\t5\t-1.100000000\t1\t-9.0\t-1.1\n"
                "11\t6\t-2.000000000\t6\t-2.000000000\t1\t-9.0\t-2.0\n"
                "12\t7\tnan\t7\tnan\t1\t-9.0\tnan\n")
            a, bad_a = G.read_logpos(d / "a.logpos")
            b, bad_b = G.read_logpos(d / "b.logpos")
            self.assertEqual((len(a), bad_a, len(b), bad_b), (3, 0, 3, 1))
            m = G.compare_logpos(a, b, bad_a, bad_b)
            self.assertEqual(m["nonfinite_cand"], 1)
            self.assertEqual(m["rows_scored"], 2)              # the nan row is left out of the statistics
            self.assertAlmostEqual(m["top1"], 0.5)             # row 11: top-1 3 vs 6
            self.assertAlmostEqual(m["max_abs_dlogit"], 0.1)
            self.assertAlmostEqual(m["ppl_ref"], math.exp(1.5))
            self.assertAlmostEqual(m["ppl_cand"], math.exp(1.55))
            ok, why = G.verdict(m, min_top1=0.99, max_ppl_rel=0.02)
            self.assertFalse(ok)
            self.assertEqual(G.read_logpos(d / "missing")[0], {})
        finally:
            shutil.rmtree(d)

    def test_build_serve_args(self):
        plan = G.Plan("teacher", 33000, 1, 1, 32488, True)
        plan.source, plan.turn_token, plan.split_at = "logpos", 123456, 32488
        base = ArgsTests.CFG
        a = G.build_serve_args(base, plan, fixed_experts=True, keep_spec=False)
        self.assertEqual(a[0], "--serve")
        self.assertEqual(a.count("--serve"), 1)
        self.assertEqual(G.flag_value(a, "--turn-token"), "123456")
        self.assertEqual(G.flag_value(a, "--short-read"), str(33000 - 1 - 32488 + 8))
        self.assertEqual(G.flag_value(a, "--spec"), "4")               # native: kept
        self.assertEqual(G.flag_value(a, "--prefill"), "auto")
        self.assertEqual(G.flag_value(a, "--adapt-swaps"), "0")
        self.assertEqual(int(G.flag_value(a, "--max-context")), 33792)
        # a NON-native pack: `strata --serve` refuses to start without --spec T (T >= 2), --mtp DIR and --prefill CHUNK
        # (generate.cpp:3681-3685), so the flags stay whatever the pack (they used to be dropped: exit status 2)
        plan.native = False
        a = G.build_serve_args(base, plan, fixed_experts=False, keep_spec=False)
        self.assertEqual(G.flag_value(a, "--spec"), "4")
        self.assertEqual(G.flag_value(a, "--mtp"), "/m/rt")
        self.assertEqual(G.flag_value(a, "--prefill"), "auto")

    def test_serve_args_without_spec_or_mtp_are_refused_with_the_reason(self):
        plan = G.Plan("teacher", 33000, 1, 1, 32488, False)
        plan.source, plan.turn_token, plan.split_at = "logpos", 123456, 32488
        no_spec = [x for x in ArgsTests.CFG if x not in ("--spec", "4")]
        with self.assertRaises(G.HarnessError) as cm:
            G.build_serve_args(no_spec, plan, fixed_experts=True, keep_spec=False)
        self.assertIn("--spec", str(cm.exception))
        self.assertIn("3681", str(cm.exception))
        self.assertIn("--teacher-source dump", str(cm.exception))          # a non-native pack has the other source
        i = ArgsTests.CFG.index("--mtp")
        no_mtp = ArgsTests.CFG[:i] + ArgsTests.CFG[i + 2:]
        with self.assertRaises(G.HarnessError) as cm:
            G.build_serve_args(no_mtp, plan, fixed_experts=True)
        self.assertIn("no --mtp", str(cm.exception))
        one = ArgsTests.CFG[:ArgsTests.CFG.index("--spec") + 1] + ["1"] + ArgsTests.CFG[ArgsTests.CFG.index("--spec") + 2:]
        with self.assertRaises(G.HarnessError):                              # --spec 1 is below the engine's minimum of 2
            G.build_serve_args(one, plan, fixed_experts=True)


class OutputParseTests(unittest.TestCase):
    def test_exact_engine_formats(self):
        # the strings are the printf formats at the end of generate.cpp's main(), filled in
        text = ("strata generate: prefill 32488 tokens in 4 chunks, 51234.5 ms (634.1 tok/s); experts streamed 10 (0 by DMA, host 1.0 ms), "
                "resident 5; PLE 3.0 ms\n"
                + "%-24s %s\n" % ("logits dumped", "cand.bin")
                + "prompt  : 1 2 3\noutput  : 11 22 33 44\n"
                + "%-24s %d tokens in %.1f ms  ->  %.2f tok/s\n" % ("decode", 4, 410.0, 9.76)
                + "%-24s %d tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n" % ("prefill", 32999, 51234.5, 644.07, 51300.0))
        info = G.parse_engine_log(text)
        self.assertEqual(info["output"], [11, 22, 33, 44])
        self.assertEqual(info["decode"], {"tokens": 4, "ms": 410.0, "tok_s": 9.76})
        self.assertEqual(info["prefill"]["tokens"], 32999)
        self.assertEqual(info["batched_prefill"], {"tokens": 32488, "chunks": 4, "ms": 51234.5, "tok_s": 634.1})
        self.assertIsNone(G.parse_engine_log("nothing here\n")["output"])


class LongPromptTests(unittest.TestCase):
    def test_build_ids_reaches_the_target_and_is_deterministic(self):
        files = M.collect_sources(M.ROOT)
        self.assertGreater(len(files), 20)
        enc = lambda t: list(t.encode("utf-8"))               # a stand-in tokenizer: one id per byte
        a = M.build_ids(files, enc, 33000)
        b = M.build_ids(files, enc, 33000)
        self.assertEqual(len(a), 33000)
        self.assertEqual(a, b)
        self.assertGreater(len(a), G.SPARSE_FROM)             # the long prompt really exceeds the selection width

    def test_text_mode_and_guard(self):
        out = M.build_text(M.collect_sources(M.ROOT), 5000)
        self.assertEqual(len(out), 5000)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(M.main(["--tokens", "1000", "--text-out", os.devnull]), 2)    # <= 2051: refused

    def test_shipped_prompts_exist(self):
        for n in ("chat", "code"):
            self.assertGreater(len((G.PROMPT_DIR / f"{n}.txt").read_text()), 200)


try:
    import regex  # noqa: F401
    HAVE_REGEX = True
except ImportError:
    HAVE_REGEX = False


@unittest.skipUnless(HAVE_REGEX, "tools/strata_tokenizer.py needs the `regex` package")
class TokenizeTests(unittest.TestCase):
    """The prompt side against a miniature byte-level tokenizer directory (the shape setup.py extracts into the pack)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import strata_tokenizer as ST
        cls.d = Path(tempfile.mkdtemp())
        b2u = ST.bytes_to_unicode()
        toks_ = [b2u[i] for i in range(256)] + ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "he"]
        (cls.d / "tok").mkdir()
        (cls.d / "tok" / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(toks_)}))
        (cls.d / "tok" / "merges.txt").write_text("h e")
        (cls.d / "tok" / "token_type.json").write_text(json.dumps([1] * 256 + [3, 3, 4, 4, 1]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def ns(self, **kw):
        import argparse
        d = dict(ids_file=None, prompt=None, prompt_file=None, prompt_name=None, chat=False, no_chat=False, max_prompt_tokens=0)
        d.update(kw)
        return argparse.Namespace(**d)

    def test_chat_prompt_is_wrapped_in_the_chat_template(self):
        ids = G.get_tokens(self.ns(prompt_name="chat"), str(self.d / "tok"), self.d)
        self.assertEqual(ids[0], 256)                         # <|im_start|>
        self.assertIn(257, ids)                               # <|im_end|>
        self.assertGreater(len(ids), 100)
        plain = G.get_tokens(self.ns(prompt_name="chat", no_chat=True), str(self.d / "tok"), self.d)
        self.assertNotIn(256, plain)

    def test_text_and_truncation(self):
        ids = G.get_tokens(self.ns(prompt="hello hello"), str(self.d / "tok"), self.d)
        self.assertIn(260, ids)                               # the "he" merge
        self.assertEqual(len(G.get_tokens(self.ns(prompt="x" * 100, max_prompt_tokens=7), str(self.d / "tok"), self.d)), 7)
        with self.assertRaises(G.HarnessError):
            G.get_tokens(self.ns(prompt="hello"), None, self.d)       # no tokenizer
        with self.assertRaises(G.HarnessError):
            G.get_tokens(self.ns(), str(self.d / "tok"), self.d)      # no prompt at all

    def test_long_prompt_is_built_once_and_exceeds_the_selection_width(self):
        w = self.d / "w"
        w.mkdir(exist_ok=True)
        ids = G.get_tokens(self.ns(prompt_name="long"), str(self.d / "tok"), w)
        self.assertEqual(len(ids), G.LONG_TOKENS)
        self.assertGreater(len(ids), G.SPARSE_FROM)
        self.assertGreaterEqual(len(ids), 32768)              # the plan's "32K tokens"
        again = G.get_tokens(self.ns(prompt_name="long"), str(self.d / "tok"), w)     # read back from long.ids
        self.assertEqual(ids, again)
        with self.assertRaises(G.HarnessError):
            G.get_tokens(self.ns(prompt_name="long"), None, self.d / "elsewhere")


class FlowTests(unittest.TestCase):
    """The whole harness against the fake engine."""

    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        (cls.d / "fake_strata.py").write_text(FAKE_ENGINE)
        (cls.d / "pack" / "full").mkdir(parents=True)
        (cls.d / "pack_native" / "full").mkdir(parents=True)
        (cls.d / "pack_native" / "full" / "native_experts.txt").write_text("# native\n")
        ids = toks(3000, 5)
        (cls.d / "ids.txt").write_text(" ".join(map(str, ids)))
        cls.ids = ids
        lp_ids = list(np.random.RandomState(11).randint(0, 60, size=3000))
        lp_ids[2650] = 63                                   # the one token that occurs once: the logpos split lands here
        (cls.d / "ids_lp.txt").write_text(" ".join(map(str, lp_ids)))
        cls.lp_ids = lp_ids
        cls.cfg = {"exe": str(cls.d / "fake_strata.py"), "args": ["--pack", "pack/full", "--prefill", "auto", "--spec", "4",
                                                                  "--mtp", "x", "--max-context", "4096", "--serve"],
                   "cwd": str(cls.d)}
        (cls.d / "cfg.json").write_text(json.dumps(cls.cfg))
        native = dict(cls.cfg, args=["--pack", "pack_native/full", "--prefill", "auto", "--spec", "4", "--mtp", "x"])
        (cls.d / "cfg_native.json").write_text(json.dumps(native))
        nospec = dict(cls.cfg, args=["--pack", "pack/full", "--prefill", "auto", "--max-context", "4096"])
        (cls.d / "cfg_nospec.json").write_text(json.dumps(nospec))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def go(self, *extra, cfg="cfg.json", env=None, name="w"):
        w = self.d / name
        shutil.rmtree(w, ignore_errors=True)
        argv = ["--engine-config", str(self.d / cfg), "--ids-file", str(self.d / "ids.txt"), "--workdir", str(w), *extra]
        old = dict(os.environ)
        os.environ.update(env or {})
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = G.main(argv)
        finally:
            os.environ.clear()
            os.environ.update(old)
        rep = json.loads((w / "report.json").read_text()) if (w / "report.json").exists() else None
        return rc, rep, out.getvalue()

    def test_teacher_identical_passes(self):
        # no noise: candidate == reference, positions >= prefill-until scored, header lies about the row count
        rc, rep, out = self.go("--tail", "300")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["config"]["prefill_until"], 2700)
        self.assertEqual(rep["metrics"]["rows_scored"], 300)
        self.assertEqual(rep["metrics"]["top1"], 1.0)
        self.assertEqual(rep["metrics"]["ppl_rel"], 0.0)
        self.assertEqual((rep["config"]["first_pos"], rep["config"]["last_pos"]), (2700, 2999))
        self.assertNotIn("determinism check", out)       # the default reference environment differs from the candidate's

    def test_expert_cache_auto_is_pinned_for_the_candidate(self):
        cfg = dict(self.cfg, args=self.cfg["args"] + ["--expert-cache", "auto"])
        (self.d / "cfg_cache.json").write_text(json.dumps(cfg))
        rc, rep, out = self.go("--tail", "300", cfg="cfg_cache.json")
        self.assertEqual(rc, 0, out)
        ref_cmd = (self.d / "w" / "ref.log").read_text().splitlines()[0]
        cand_cmd = (self.d / "w" / "cand.log").read_text().splitlines()[0]
        self.assertIn("--expert-cache auto", ref_cmd)
        self.assertIn("--expert-cache 1234", cand_cmd)              # the slot count the reference printed
        self.assertTrue(any("pinned" in n for n in rep["notes"]))
        rc, rep, out = self.go("--tail", "300", "--no-pin-expert-cache", cfg="cfg_cache.json")
        self.assertIn("--expert-cache auto", (self.d / "w" / "cand.log").read_text().splitlines()[0])

    def test_default_reference_env_is_set_for_the_reference_only(self):
        rc, rep, out = self.go("--tail", "300", env={"FAKE_NOISE": "0.002"})
        self.assertEqual(rep["runs"]["reference"]["env"], {"STRATA_VOLTA_ATTN": "0", "STRATA_PREFILL_F16_GEMM": "0"})
        self.assertEqual(rep["runs"]["candidate"]["env"], {})
        self.assertNotIn("determinism check", out)
        # the noise (sigma 0.002 on logits of sigma 3) flips almost nothing and moves the perplexity by far less than 2%
        self.assertEqual(rc, 0, out)
        self.assertGreater(rep["metrics"]["max_abs_dlogit"], 0.0)
        self.assertGreaterEqual(rep["metrics"]["top1"], 0.99)

    def test_heavy_noise_fails_top1(self):
        rc, rep, out = self.go("--tail", "300", env={"FAKE_NOISE": "4.0"})
        self.assertEqual(rc, 1, out)
        self.assertLess(rep["metrics"]["top1"], 0.99)
        self.assertTrue(any("top-1" in r for r in rep["reasons"]))

    def test_nan_fails(self):
        rc, rep, out = self.go("--tail", "300", env={"FAKE_NAN_AT": "2800"})
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["metrics"]["nonfinite_cand"], 1)
        self.assertTrue(any("non-finite" in r for r in rep["reasons"]))

    def test_candidate_crash_is_a_fail_reference_crash_is_an_error(self):
        rc, rep, out = self.go("--tail", "300", env={"FAKE_CRASH": "1"})
        self.assertEqual(rc, 1, out)
        self.assertIn("CANDIDATE", rep["reasons"][0])
        # reference crashes when the environment says the same as the candidate
        rc, rep, out = self.go("--tail", "300", "--ref-env", "", env={"FAKE_CRASH": "1"})
        self.assertEqual(rc, 2, out)

    def test_short_prompt_token_path_and_notes(self):
        rc, rep, out = self.go("--prefill-until", "0", "--tail", "300")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["metrics"]["rows_scored"], 3000)       # every position has a row
        self.assertTrue(any("not exercised" in n for n in rep["notes"]))
        rc, rep, out = self.go("--tail", "2900")                    # auto: 3000 - 2900 < 256 -> the token path
        self.assertTrue(any("NOT exercised" in n for n in rep["notes"]))

    def test_stride(self):
        rc, rep, out = self.go("--tail", "300", "--stride", "16")
        self.assertEqual(rc, 0, out)
        sel = G.selected_positions(3000, 16, 2700)
        self.assertEqual(rep["metrics"]["rows_scored"], len(sel))

    def test_greedy_identical_and_diverging(self):
        rc, rep, out = self.go("--mode", "greedy", "--max-new", "40")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["greedy"]["prefix"], 40)
        self.assertIsNone(rep["greedy"]["first_diff"])
        self.assertEqual(rep["metrics"]["rows_scored"], 40)
        # the candidate diverges at generated token 10: rows 0..10 are comparable (row 10 is where they differ)
        rc, rep, out = self.go("--mode", "greedy", "--max-new", "40", env={"FAKE_DIVERGE_AT": "10"})
        self.assertEqual(rep["greedy"]["prefix"], 10)
        self.assertEqual(rep["greedy"]["first_diff"], 10)
        self.assertEqual(rep["metrics"]["rows_scored"], 11)
        self.assertEqual(rep["metrics"]["top1_agree"], 10)
        self.assertEqual(rc, 1, out)                               # 10/11 = 90.9% < 99%
        rc, rep, out = self.go("--mode", "greedy", "--max-new", "40", "--min-top1", "0.9", "--max-ppl-rel", "1e9",
                               env={"FAKE_DIVERGE_AT": "10"})
        self.assertEqual(rc, 0, out)
        # the same environment on both sides is flagged as a determinism check
        rc, rep, out = self.go("--mode", "greedy", "--max-new", "40", "--ref-env", "")
        self.assertEqual(rc, 0, out)
        self.assertTrue(any("determinism" in n for n in rep["notes"]))

    def test_native_pack_dump_source_falls_back_to_greedy_on_tokens(self):
        rc, rep, out = self.go("--tail", "300", "--teacher-source", "dump", cfg="cfg_native.json")
        self.assertEqual(rep["config"]["mode"], "greedy")
        self.assertTrue(rep["config"]["native_pack"])
        self.assertTrue(rep["metrics"]["tokens_only"])
        self.assertTrue(any("NATIVE" in n for n in rep["notes"]))
        self.assertEqual(rc, 0, out)

    def go_lp(self, *extra, cfg="cfg_native.json", env=None, name="wlp"):
        w = self.d / name
        shutil.rmtree(w, ignore_errors=True)
        argv = ["--engine-config", str(self.d / cfg), "--ids-file", str(self.d / "ids_lp.txt"), "--workdir", str(w), *extra]
        old = dict(os.environ)
        os.environ.update(env or {})
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = G.main(argv)
        finally:
            os.environ.clear()
            os.environ.update(old)
        rep = json.loads((w / "report.json").read_text()) if (w / "report.json").exists() else None
        return rc, rep, out.getvalue()

    def test_logpos_is_the_default_for_a_native_pack(self):
        rc, rep, out = self.go_lp("--tail", "300")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["config"]["mode"], "teacher")
        self.assertEqual(rep["config"]["source"], "logpos")
        self.assertEqual(rep["config"]["split_at"], 2650)
        self.assertEqual(rep["config"]["turn_token"], 63)
        m = rep["metrics"]
        self.assertTrue(m["logprobs_only"])
        self.assertEqual(m["rows_scored"], 3000 - 1 - 2650)          # positions 2650 .. 2998: targets 2651 .. 2999
        self.assertEqual((m["top1"], m["ppl_rel"], m["max_abs_dlogit"]), (1.0, 0.0, 0.0))
        self.assertNotIn("kl_mean", m)
        # the engine was started as a server, with the split token, a window long enough for the tail, and the logpos file
        cmd = rep["runs"]["candidate"]["cmd"] if "cmd" in rep["runs"]["candidate"] else None
        log = (self.d / "wlp" / "cand.log").read_text()
        self.assertIn("--serve", log.splitlines()[0])
        self.assertIn("--turn-token 63", log.splitlines()[0])
        self.assertIn("--short-read 357", log.splitlines()[0])          # rows 349 + 8
        self.assertIn("--spec 4", log.splitlines()[0])                  # a native pack keeps speculative decoding

    def test_logpos_noise_and_nan(self):
        rc, rep, out = self.go_lp("--tail", "300", env={"FAKE_NOISE": "4.0"})
        self.assertEqual(rc, 1, out)
        self.assertLess(rep["metrics"]["top1"], 0.99)
        rc, rep, out = self.go_lp("--tail", "300", env={"FAKE_NAN_AT": "2800"})
        self.assertEqual(rc, 1, out)
        self.assertEqual(rep["metrics"]["nonfinite_cand"], 1)
        rc, rep, out = self.go_lp("--tail", "300", env={"FAKE_NOISE": "0.002"})
        self.assertEqual(rc, 0, out)
        self.assertGreater(rep["metrics"]["max_abs_dlogit"], 0.0)

    def test_logpos_on_a_non_native_pack_when_asked_and_crash_handling(self):
        rc, rep, out = self.go_lp("--tail", "300", "--teacher-source", "logpos", cfg="cfg.json")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["config"]["source"], "logpos")
        log = (self.d / "wlp" / "cand.log").read_text().splitlines()[0]
        self.assertIn("--spec 4", log)                                   # `--serve` cannot start without them, whatever the pack
        self.assertIn("--mtp x", log)
        rc, rep, out = self.go_lp("--tail", "300", env={"FAKE_CRASH": "1"})
        self.assertEqual(rc, 1, out)
        self.assertIn("CANDIDATE", rep["reasons"][0])

    def test_logpos_without_spec_flags_stops_with_the_reason(self):
        # no --spec / --mtp in the engine arguments: `strata --serve` could not start (exit 2 from the engine); the harness says why
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = G.main(["--engine-config", str(self.d / "cfg_nospec.json"), "--ids-file", str(self.d / "ids_lp.txt"),
                         "--workdir", str(self.d / "wnospec"), "--tail", "300", "--teacher-source", "logpos"])
        self.assertEqual(rc, 2)
        self.assertIn("--spec", err.getvalue())
        self.assertIn("--mtp", err.getvalue())

    def test_every_path_handed_to_the_engine_is_absolute(self):
        # The engine runs in the config's cwd (here self.d), the harness is started from somewhere else with RELATIVE --workdir, --exe and
        # --pack: the tokens file, the dumps, STRATA_LOGPOS, the engine and --pack must still name the files the harness reads back.
        caller = self.d / "caller"
        caller.mkdir(exist_ok=True)
        shutil.copy(self.d / "fake_strata.py", caller / "engine_here.py")     # exists in the caller's directory only
        shutil.rmtree(caller / "rel_out", ignore_errors=True)
        shutil.rmtree(caller / "rel_lp", ignore_errors=True)
        old = os.getcwd()
        os.chdir(caller)
        try:
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = G.main(["--exe", "engine_here.py", "--cwd", str(self.d), "--pack", "../pack/full", "--engine-args", "--prefill auto",
                             "--ids-file", str(self.d / "ids.txt"), "--workdir", "rel_out", "--tail", "300"])
            self.assertEqual(rc, 0, out.getvalue())
            w = caller / "rel_out"
            head = (w / "cand.log").read_text().splitlines()[0].split()
            self.assertEqual(Path(head[2]).name, "engine_here.py")
            self.assertTrue(Path(head[2]).is_absolute())
            joined = " ".join(head)
            self.assertIn(f"--tokens-file {w}/tokens.ids", joined)
            self.assertIn(f"--dump-logits {w}/cand.bin", joined)
            self.assertIn(f"--pack {self.d}/pack/full", joined)
            # the logpos source: STRATA_LOGPOS names a file under the caller's workdir, where the harness looks for it
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                rc = G.main(["--engine-config", str(self.d / "cfg_native.json"), "--ids-file", str(self.d / "ids_lp.txt"),
                             "--workdir", "rel_lp", "--tail", "300"])
            self.assertEqual(rc, 0, out.getvalue())
            self.assertTrue((caller / "rel_lp" / "cand.logpos").exists())
            self.assertFalse((self.d / "rel_lp").exists())                       # nothing was written under the engine's cwd
        finally:
            os.chdir(old)

    def test_gate_q_line(self):
        rc, rep, out = self.go("--tail", "500", name="gq")             # 500 scored rows after 2,500 batched tokens: the gate's own size
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE Q (docs/volta/PLAN.md): PASS", out)
        self.assertEqual(rep["gate_q"]["status"], "PASS")
        rc, rep, out = self.go("--tail", "300", name="gq2")
        self.assertEqual(rc, 0, out)
        self.assertIn("GATE Q (docs/volta/PLAN.md): not established", out)
        self.assertIn("300 rows scored", out)
        rc, rep, out = self.go("--tail", "500", env={"FAKE_NOISE": "4.0"}, name="gq3")
        self.assertEqual(rc, 1, out)
        self.assertIn("GATE Q (docs/volta/PLAN.md): FAIL", out)
        rc, rep, out = self.go("--mode", "greedy", "--max-new", "40", name="gq4")
        self.assertIn("greedy mode", out.split("GATE Q")[1])

    def test_dry_run_prints_the_commands_and_runs_nothing(self):
        w = self.d / "wdry"
        shutil.rmtree(w, ignore_errors=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = G.main(["--engine-config", str(self.d / "cfg_native.json"), "--ids-file", str(self.d / "ids_lp.txt"), "--workdir", str(w),
                         "--tail", "300", "--dry-run"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("source logpos", text)
        self.assertIn("STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0 STRATA_LOGPOS=", text)
        self.assertIn("--turn-token 63", text)
        self.assertFalse((w / "ref.log").exists())
        with contextlib.redirect_stdout(out := io.StringIO()):
            rc = G.main(["--engine-config", str(self.d / "cfg.json"), "--ids-file", str(self.d / "ids.txt"), "--workdir", str(w), "--tail", "300",
                         "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertIn("--prefill-until 2700", out.getvalue())
        self.assertIn("--dump-logits", out.getvalue())

    def test_logpos_tail_must_have_a_unique_token(self):
        w = self.d / "wlp2"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = G.main(["--engine-config", str(self.d / "cfg_native.json"), "--ids-file", str(self.d / "ids.txt"), "--workdir", str(w),
                         "--tail", "300"])
        self.assertEqual(rc, 2)             # the random 64-symbol prompt has no token that occurs once

    def test_logpos_candidate_against_an_external_logits_reference(self):
        # the reference is a logits file (llama.cpp / another GPU): converted to log-probabilities for the tail positions
        d = self.d / "ext"
        d.mkdir(exist_ok=True)
        n = len(self.lp_ids)
        rows = np.zeros((n, V), np.float32)
        # the fake model's own distribution at each position (what the candidate engine computes with no noise)
        sys.path.insert(0, str(self.d))
        import hashlib
        for p in range(2600, n):
            seed = int.from_bytes(hashlib.sha256(repr(tuple(int(x) for x in self.lp_ids[: p + 1])).encode()).digest()[:4], "little")
            rows[p] = (np.random.RandomState(seed).normal(size=V) * 3.0).astype(np.float32)
        np.save(d / "ref.npy", rows[2600:])           # rows for positions 2600 .. 2999: ends at the last position (n - 1 + 1 - 1)
        rc, rep, out = self.go_lp("--tail", "300", "--ref-logits", str(d / "ref.npy"), name="wlp3")
        self.assertEqual(rc, 0, out)
        self.assertEqual(rep["metrics"]["rows_scored"], n - 1 - 2650)
        self.assertEqual(rep["metrics"]["top1"], 1.0)
        self.assertLess(rep["metrics"]["ppl_rel"], 1e-6)

    def test_offline_files(self):
        d = self.d / "off"
        d.mkdir(exist_ok=True)
        rng = np.random.RandomState(3)
        ref = rng.normal(size=(40, 32)).astype(np.float32)
        write_logits(d / "ref.bin", ref)
        np.save(d / "cand.npy", ref + 0.001 * rng.normal(size=ref.shape).astype(np.float32))
        w = d / "w"
        ids = list(np.random.RandomState(9).randint(0, 32, size=100))
        (d / "ids.txt").write_text(" ".join(map(str, ids)))
        with contextlib.redirect_stdout(io.StringIO()):
            rc = G.main(["--ids-file", str(d / "ids.txt"), "--workdir", str(w), "--ref-logits", str(d / "ref.bin"),
                         "--cand-logits", str(d / "cand.npy")])
        rep = json.loads((w / "report.json").read_text())
        self.assertEqual(rc, 0)
        # both files end at the last position (99): they cover 60..99
        self.assertEqual((rep["config"]["first_pos"], rep["config"]["last_pos"]), (60, 99))
        self.assertEqual(rep["metrics"]["rows_scored"], 40)
        # the reference file's first position given explicitly
        with contextlib.redirect_stdout(io.StringIO()):
            rc = G.main(["--ids-file", str(d / "ids.txt"), "--workdir", str(w), "--ref-logits", str(d / "ref.bin"),
                         "--cand-logits", str(d / "cand.npy"), "--ref-first-position", "0", "--cand-first-position", "0"])
        self.assertEqual(json.loads((w / "report.json").read_text())["config"]["last_pos"], 39)

    def test_reuse_skips_the_engine(self):
        rc, rep, out = self.go("--tail", "300", name="reuse")
        self.assertEqual(rc, 0)
        w = self.d / "reuse"
        argv = ["--engine-config", str(self.d / "cfg.json"), "--ids-file", str(self.d / "ids.txt"), "--workdir", str(w),
                "--tail", "300", "--reuse"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = G.main(argv)
        self.assertEqual(rc, 0)
        self.assertIn("reusing", out.getvalue())


if __name__ == "__main__":
    unittest.main()
