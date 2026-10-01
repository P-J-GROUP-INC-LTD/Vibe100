"""tools/volta/test_logit_identity.py - tests of tools/volta/logit_identity.sh: the syntax, `--dry-run` (every command, nothing run, nothing
created, no GPU / model / llama.cpp needed), and the whole matrix against FAKES - a fake engine (test_golden_compare.FAKE_ENGINE, also
standing in for the upstream build), a fake `llama-perplexity` that writes and scores base files with kld_format.py, a miniature GGUF - so
the plumbing of every row (the commands, the report reading, PASS / FAIL / SKIP / INFO, the exit status) is exercised on a machine with
nothing but Python, bash and numpy.  The upstream measurement patch is checked against the upstream snapshot's verify.cpp.

    python3 -m unittest discover -s tools/volta        (needs bash, numpy)
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_golden_compare as T  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "logit_identity.sh"
HAVE_BASH = shutil.which("bash") is not None

FAKE_LLAMA = textwrap.dedent('''
    #!/usr/bin/env python3
    # a stand-in for llama-perplexity: --kl-divergence-base writes a base file of the fake model's distributions, --kl-divergence scores the
    # "CPU backend" (the same model plus FAKE_LLAMA_CPU_NOISE) against it and prints llama.cpp's statistics blocks
    import hashlib, os, sys
    import numpy as np
    sys.path.insert(0, %(tools)r)
    import kld_format as K
    V = 64
    a = sys.argv[1:]
    def opt(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    ctx, chunks, text = int(opt("-c")), int(opt("--chunks", 0) or 0), open(opt("-f")).read()
    def logits(ids):
        seed = int.from_bytes(hashlib.sha256(repr(tuple(int(i) for i in ids)).encode()).digest()[:4], "little")
        return (np.random.RandomState(seed).normal(size=V) * 3.0).astype("<f4")
    def rows_of(chunk_tokens, noise):
        out = np.stack([logits(chunk_tokens[: p + 1]) for p in range(ctx // 2, ctx - 1)])
        if noise:
            out = out + np.random.RandomState(7).normal(size=out.shape).astype("f4") * noise
        return out
    if not os.path.exists(opt("-m")):
        print("fake llama-perplexity: no model", file=sys.stderr); sys.exit(1)
    if "--kl-divergence" not in a:
        toks = np.random.RandomState(len(text)).randint(0, V, size=(chunks, ctx)).astype(np.int32)
        K.write_base(opt("--kl-divergence-base"), ctx, toks, [rows_of([int(t) for t in toks[c]], 0.0) for c in range(chunks)])
        print("fake llama-perplexity: wrote", opt("--kl-divergence-base"), "(%%s)" %% os.path.basename(sys.argv[0]))
    else:
        b = K.read_base(opt("--kl-divergence-base"))
        sc = K.KldScorer()
        for c in range(b.chunks_present):
            blk = b.block(c); s_, m_ = b.header_of(blk)
            sc.add_logits(rows_of([int(t) for t in b.tokens[c]], float(os.environ.get("FAKE_LLAMA_CPU_NOISE", "0.002"))), b.targets(c), s_, m_, blk[:, 4:4 + V])
        print(K.format_summary(sc.summary()))
''')


def mini_gguf(path: Path, kv: dict) -> None:
    def s(x):
        e = x.encode()
        return struct.pack("<Q", len(e)) + e
    body = b""
    for k, (t, v) in kv.items():
        body += s(k) + struct.pack("<I", t) + (s(v) if t == 8 else struct.pack("<?" if t == 7 else "<I", v))
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, len(kv)) + body)


def executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@unittest.skipUnless(HAVE_BASH, "needs bash")
class Syntax(unittest.TestCase):
    def test_bash_n_and_help(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(SCRIPT)]).returncode, 0)
        r = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        for opt in ("--config", "--upstream-hook", "--llama-cuda-args", "--dry-run", "--kv-fp16-row", "--only"):
            self.assertIn(opt, r.stdout)
        bad = subprocess.run(["bash", str(SCRIPT), "--nonsense"], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 2)


@unittest.skipUnless(HAVE_BASH, "needs bash")
class DryRun(unittest.TestCase):
    def dry(self, *args):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        out = d / "never_created"
        r = subprocess.run(["bash", str(SCRIPT), "--dry-run", "--out", str(out), "--work", str(d / "work_never"), *args],
                           capture_output=True, text=True, cwd=d)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertFalse(out.exists(), "a dry run creates nothing")
        self.assertFalse((d / "work_never").exists())
        return r.stdout

    def test_every_row_and_every_command_is_printed(self):
        out = self.dry("--config", str(d_cfg()))
        for row in ("1c", "1a", "1b", "1d", "2a", "2b", "2c", "2d", "2e", "2f"):
            self.assertRegex(out, rf"row {row}: ")
        self.assertNotIn("row 2g", out)                                      # only on request
        # tier 1
        self.assertIn("--exact", out)
        self.assertIn("--reuse-ref", out)                                    # 1a, 1b, 1d take the baseline of 1c
        self.assertIn("--cand-exe", out)                                     # 1a: upstream's binary
        self.assertIn("upstream-build/strata", out)
        self.assertIn("--numa\\ mirror", out)
        self.assertIn("STRATA_IQ512=1", out)
        self.assertIn("STRATA_PROMPT_ATTN_OLD=1", out)
        # the upstream build: a worktree of the first commit, sm_70, CUDA 12.8, nothing else patched but the measurement hook
        self.assertIn("worktree add --detach", out)
        self.assertIn("3906943", out)
        self.assertIn("upstream_logits_dump.patch", out)
        self.assertIn("-DSTRATA_EXPERIMENTAL_SM60=ON", out)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=70", out)
        self.assertIn("-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc", out)
        self.assertIn("-DSTRATA_BUILD_TESTS=OFF", out)
        # tier 2: llama.cpp twice, the reference, the floor, the engine twice
        self.assertIn("-DGGML_CUDA=ON", out)
        self.assertIn("-DGGML_CUDA=OFF", out)
        self.assertIn("--kl-divergence-base", out)
        self.assertRegex(out, r"llama-cpu/bin/llama-perplexity .* --kl-divergence --kl-divergence-base")
        self.assertIn("-c 4096 --chunks 4", out)
        self.assertIn("--ref-kld", out)
        self.assertIn("--kld-floor", out)
        self.assertIn("STRATA_VOLTA_ATTN=0\\ STRATA_PREFILL_F16_GEMM=0", out)
        self.assertIn("ROWS: 0 PASS, 0 FAIL, 0 SKIP, 0 INFO (dry run", out)

    def test_selection_and_options(self):
        out = self.dry("--config", str(d_cfg()), "--only", "2c,2e,2g", "--ctx", "8192", "--chunks", "5", "--upstream-hook", "none",
                       "--llama-cuda-args", "-ngl 99 --cpu-moe", "--gpu", "0", "--text", "/some/text.txt")
        self.assertRegex(out, r"row 2e: ")
        self.assertRegex(out, r"row 2g: ")
        self.assertNotRegex(out, r"row 2f: ")
        self.assertNotRegex(out, r"row 1c: ")
        self.assertIn("--kv\\ fp16", out)
        self.assertIn("-c 8192 --chunks 5", out)
        self.assertIn("-ngl 99 --cpu-moe", out)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", out)
        self.assertIn("/some/text.txt", out)
        out = self.dry("--config", str(d_cfg()), "--tier", "1", "--upstream-hook", "none")
        self.assertNotIn("upstream_logits_dump.patch", out)
        self.assertNotRegex(out, r"row 2c: ")

    def test_works_with_no_config_at_all(self):
        out = self.dry("--config", "/nonexistent/strata-x.json")
        self.assertIn("placeholders", out)

    def test_missing_config_is_an_error_without_dry_run(self):
        r = subprocess.run(["bash", str(SCRIPT), "--config", "/nonexistent/strata-x.json", "--out", tempfile.mkdtemp(), "--work", tempfile.mkdtemp()],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("setup.sh", r.stderr)


_CFG_DIR = None


def d_cfg() -> Path:
    """A throw-away engine config for the dry runs (the engine need not exist: a dry run never starts it)."""
    global _CFG_DIR
    if _CFG_DIR is None:
        _CFG_DIR = Path(tempfile.mkdtemp())
        (_CFG_DIR / "cfg.json").write_text(json.dumps({"exe": "/nonexistent/strata", "cwd": str(_CFG_DIR), "tokenizer": str(_CFG_DIR / "tok"),
                                                       "args": ["--pack", "pack/full", "--native", str(_CFG_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
                                                                "--spec", "4", "--mtp", "m"]}))
        (_CFG_DIR / "pack" / "full").mkdir(parents=True)
        (_CFG_DIR / "pack" / "full" / "native_experts.txt").write_text("x")
    return _CFG_DIR / "cfg.json"


def tearDownModule():
    if _CFG_DIR is not None:
        shutil.rmtree(_CFG_DIR, ignore_errors=True)


@unittest.skipUnless(HAVE_BASH, "needs bash")
class WholeMatrixWithFakes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        d = cls.d
        executable(d / "fake_strata", "#!/usr/bin/env python3\n" + T.FAKE_ENGINE)
        (d / "pack" / "full").mkdir(parents=True)
        (d / "pack" / "full" / "native_experts.txt").write_text("# native\n")
        mini_gguf(d / "model.gguf", {"tokenizer.ggml.model": (8, "gpt2")})
        (d / "text.txt").write_text("the quick brown fox " * 40)
        cls.cfg = d / "cfg.json"
        cls.cfg.write_text(json.dumps({"exe": str(d / "fake_strata"), "cwd": str(d), "args": [
            "--pack", "pack/full", "--native", str(d / "model.gguf"), "--prefill", "auto", "--spec", "4", "--mtp", "x", "--max-context", "4096",
            "--expert-cache", "auto", "--serve"]}))
        ids = list(np.random.RandomState(11).randint(0, 60, size=3000))
        ids[2650] = 63
        (d / "ids.txt").write_text(" ".join(map(str, ids)))
        w = d / "work"
        for sub in ("llama-cuda", "llama-cpu"):
            executable(w / sub / "bin" / "llama-perplexity", "#!/usr/bin/env python3\n" + FAKE_LLAMA % {"tools": str(HERE)})
        executable(w / "upstream-build" / "strata", "#!/usr/bin/env python3\n" + T.FAKE_ENGINE)       # the "upstream" engine
        cls.work = w

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d)

    def run_script(self, *args, env=None, name="out"):
        out = self.d / name
        shutil.rmtree(out, ignore_errors=True)
        e = dict(os.environ)
        e.update({"LOGIT_IDENTITY_NUMA_NODES": "1"})
        e.update(env or {})
        r = subprocess.run(["bash", str(SCRIPT), "--config", str(self.cfg), "--out", str(out), "--work", str(self.work), "--skip-build",
                            "--ids-file", str(self.d / "ids.txt"), "--tail", "300", "--text", str(self.d / "text.txt"), "--ctx", "64", "--chunks", "3",
                            "--llama-dir", str(self.d), *args], capture_output=True, text=True, env=e, cwd=self.d)
        rows = {}
        if (out / "results.tsv").exists():
            for ln in (out / "results.tsv").read_text().splitlines():
                f = ln.split("\t")
                rows[f[0]] = (f[1], f[3])
        return r, rows, out

    def test_clean_run_every_row_passes_or_says_why_not(self):
        r, rows, out = self.run_script(env={"FAKE_NOISE_IQ512": "1e-5", "LOGIT_IDENTITY_CPU_FLAGS": "avx512f avx512bw avx512vl avx512_vbmi",
                                            "FAKE_MIRROR_LINE": "1", "LOGIT_IDENTITY_NUMA_NODES": "2"})
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        want = {"1c": "PASS", "1a": "PASS", "1b": "PASS", "1d": "INFO", "2a": "PASS", "2b": "PASS", "2c": "PASS", "2d": "INFO", "2e": "PASS", "2f": "PASS"}
        self.assertEqual({k: v[0] for k, v in rows.items()}, want, r.stdout + r.stderr)
        self.assertIn("bitwise identical", rows["1c"][1])
        self.assertIn("rows of 64 float32 logits", rows["1a"][1])               # the fake upstream has the hook: bits, not printed log-probs
        self.assertIn("differ", rows["1d"][1])                                   # INFO: how far apart
        self.assertIn("ULP", rows["1d"][1])
        self.assertIn("n_ctx 64, n_vocab 64, 3 of 3 chunks", rows["2c"][1])          # the reference file, as kld_format.py info describes it
        self.assertIn("noise floor", rows["2d"][1])
        self.assertIn("mean KLD", rows["2e"][1])
        self.assertIn("allowed", rows["2e"][1])
        self.assertRegex(r.stdout, r"ROWS: \d+ PASS, 0 FAIL, 0 SKIP, 2 INFO")
        for f in ("results.tsv", "summary.txt", "env.txt", "tier1/1c.json", "tier1/1a.json", "tier2/2e.json", "tier2/cuda.kld", "tier2/llama_cpu_vs_cuda.log"):
            self.assertTrue((out / f).exists(), f)
        self.assertTrue((out / "tier1" / "1a" / "report.txt").exists())
        # 1a ran the OTHER binary; 1b the mirror flag; 1c's reference run was reused, not repeated
        self.assertIn("upstream-build/strata", (out / "tier1" / "1a" / "cand.log").read_text().splitlines()[0])
        self.assertIn("--numa mirror", (out / "tier1" / "1b" / "cand.log").read_text().splitlines()[0])
        self.assertEqual((out / "tier1" / "1a" / "ref.log").read_text(), (out / "tier1" / "1c" / "ref.log").read_text())
        # tier 2's engine runs: one launch per chunk, on the reference's tokens
        self.assertTrue((out / "tier2" / "2e" / "chunk2.ids").exists())
        self.assertIn("KLD VERDICT (provisional", (out / "tier2" / "2e" / "report.txt").read_text())

    def test_noise_fails_the_right_rows(self):
        # candidate runs that do not say STRATA_VOLTA_ATTN=0 get noise from the fake engine: row 1a (upstream's environment has none of the port's
        # switches) differs, and so does tier 2's engine with its fast paths ON; with them off (2f) it is clean
        r, rows, out = self.run_script(env={"FAKE_NOISE": "0.5", "LOGIT_IDENTITY_NUMA_NODES": "1"}, name="out_noise")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        st = {k: v[0] for k, v in rows.items()}
        self.assertEqual((st["1c"], st["1a"], st["2e"], st["2f"]), ("PASS", "FAIL", "FAIL", "PASS"), r.stdout + r.stderr)
        self.assertIn("rows differ", rows["1a"][1])
        self.assertIn("top-1 differs", rows["2e"][1])
        self.assertEqual(st["1b"], "SKIP")                                        # one NUMA node here
        self.assertIn("NUMA node", rows["1b"][1])
        self.assertIn("ROWS", r.stdout)

    def test_skips_say_why(self):
        # no VBMI and a canonical (non-native) pack: upstream cannot run it; no AVX-512 at all: 1d has nothing to compare
        d = self.d
        cfg = d / "cfg_q2.json"
        cfg.write_text(json.dumps({"exe": str(d / "fake_strata"), "cwd": str(d), "args": [
            "--pack", "pack_q2/full", "--native", str(d / "model.gguf"), "--prefill", "auto", "--spec", "4", "--mtp", "x", "--max-context", "4096", "--serve"]}))
        (d / "pack_q2" / "full").mkdir(parents=True, exist_ok=True)
        out = d / "out_skip"
        shutil.rmtree(out, ignore_errors=True)
        e = dict(os.environ, LOGIT_IDENTITY_NUMA_NODES="1", LOGIT_IDENTITY_CPU_FLAGS="sse2 avx2")
        r = subprocess.run(["bash", str(SCRIPT), "--config", str(cfg), "--out", str(out), "--work", str(self.work), "--skip-build", "--tier", "1",
                            "--ids-file", str(d / "ids.txt"), "--tail", "300"], capture_output=True, text=True, env=e, cwd=d)
        rows = {ln.split("\t")[0]: ln.split("\t") for ln in (out / "results.tsv").read_text().splitlines()}
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(rows["1c"][1], "PASS")
        for k in ("1a", "1b", "1d"):
            self.assertEqual(rows[k][1], "SKIP", k)
        self.assertIn("AVX512-VBMI", rows["1a"][3])
        self.assertIn("NUMA node", rows["1b"][3])
        self.assertIn("no AVX2 kernel", rows["1d"][3])

    def test_mirror_that_did_not_happen_is_a_skip_not_a_pass(self):
        r, rows, out = self.run_script("--only", "1b", env={"LOGIT_IDENTITY_NUMA_NODES": "2"}, name="out_mirror")     # no 'MIRRORED' line from the fake
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(rows["1b"][0], "SKIP")
        self.assertIn("ONE copy", rows["1b"][1])

    def test_missing_gguf_skips_tier_2_and_says_how_to_fetch_it(self):
        d = self.d
        cfg = d / "cfg_nogguf.json"
        cfg.write_text(json.dumps({"exe": str(d / "fake_strata"), "cwd": str(d), "args": [
            "--pack", "pack/full", "--native", str(d / "models" / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"), "--prefill", "auto",
            "--spec", "4", "--mtp", "x", "--serve"]}))
        out = d / "out_nogguf"
        shutil.rmtree(out, ignore_errors=True)
        r = subprocess.run(["bash", str(SCRIPT), "--config", str(cfg), "--out", str(out), "--work", str(self.work), "--skip-build", "--tier", "2",
                            "--text", str(d / "text.txt")], capture_output=True, text=True, cwd=d)
        rows = {ln.split("\t")[0]: ln.split("\t") for ln in (out / "results.tsv").read_text().splitlines()}
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for k in ("2c", "2d", "2e", "2f"):
            self.assertEqual(rows[k][1], "SKIP", k)
            self.assertIn("is not there", rows[k][3])
        self.assertIn("huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS/", r.stderr)
        self.assertIn("setup.sh", r.stderr)

    def test_a_broken_reference_backend_fails_the_floor_row(self):
        r, rows, out = self.run_script("--only", "2c,2d", env={"FAKE_LLAMA_CPU_NOISE": "3.0"}, name="out_floor")
        self.assertEqual(rows["2c"][0], "PASS")
        self.assertEqual(rows["2d"][0], "FAIL")
        self.assertIn("disagree", rows["2d"][1])
        self.assertEqual(r.returncode, 1)


class UpstreamPatch(unittest.TestCase):
    PATCH = HERE / "upstream_logits_dump.patch"

    def test_the_patch_only_adds_the_logits_hook_and_applies_to_the_upstream_snapshot(self):
        text = self.PATCH.read_text()
        files = [ln for ln in text.splitlines() if ln.startswith("+++ ")]
        self.assertEqual(files, ["+++ b/src/core/verify.cpp"])                       # one file
        self.assertFalse([ln for ln in text.splitlines() if ln.startswith("-") and not ln.startswith("---")])   # it removes nothing
        self.assertIn("STRATA_LOGITS_DUMP", text)
        added = [ln[1:] for ln in text.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
        self.assertLess(len(added), 60)
        # the same lines are in this tree's verify.cpp (the port's engine has the hook)
        mine = (ROOT / "src" / "core" / "verify.cpp").read_text()
        for ln in added:
            self.assertIn(ln, mine)
        if shutil.which("git") and subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", "3906943:src/core/verify.cpp"], capture_output=True).returncode == 0:
            d = Path(tempfile.mkdtemp())
            try:
                up = subprocess.run(["git", "-C", str(ROOT), "show", "3906943:src/core/verify.cpp"], capture_output=True, check=True).stdout
                (d / "src" / "core").mkdir(parents=True)
                (d / "src" / "core" / "verify.cpp").write_bytes(up)
                subprocess.run(["git", "init", "-q", str(d)], check=True)
                r = subprocess.run(["git", "-C", str(d), "apply", "--check", str(self.PATCH)], capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
            finally:
                shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
