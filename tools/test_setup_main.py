"""tools/test_setup_main.py - setup.main() driven through its steps on a faked PC (a Cascade Lake Xeon and an RTX 4090; no GPU,
no network, nothing installed; the pack tools are recorded, not run):

  * which pack form the Q2_0 model gets, for the ready-made upstream engine and for one compiled from this source, and what
    happens to a pack folder that holds the other form (audit: setup chose the canonical pack for the ready-made engine,
    which exits "missing AVX512-VBMI" on a CPU without VBMI);
  * the --numa setting surviving a `--setup` run again and a new Strata folder adopting the install.

    python3 tools/test_setup_main.py
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import setup  # noqa: E402

RTX40 = {"index": 0, "name": "RTX 4090", "vram_gb": 24.0, "arch": "89", "driver": "580.95"}
XEON = "Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz"
PREBUILT = {"version": "0.1.31", "archs": [75, 80, 86, 89, 120], "ptx": True, "cuda": "13.0"}      # upstream's release
PACK_TOOLS = ("strata_pack.py", "pack_index.py", "iq_pack.py")


class FakePC(unittest.TestCase):
    """A Strata folder (self.root) beside a data folder, on a PC with an RTX 4090 and `cpu` (vbmi: the CPU has AVX-512 VBMI)."""

    cpu = (XEON, True, True)
    vbmi = False

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.root, self.data = self.tmp / "strata", self.tmp / "data"
        for sub in ("engine", "serve", "data"):
            (self.root / sub).mkdir(parents=True)
        (self.root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.31 LANGUAGES CXX)")
        (self.root / "data" / "expert-profile.bin").write_text("p")
        (self.root / "data" / "draft_vocab.bin").write_text("v")
        models = self.data / "models" / "Q2_0"
        models.mkdir(parents=True)
        for i in (1, 2):
            shard = models / f"Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{i}-of-00002.gguf"
            shard.write_text("g")
            shard.with_name(shard.name + ".done").write_text("x")
        self.pack = self.data / "packs" / "q2_0"
        (self.pack / "tokenizer").mkdir(parents=True)
        (self.pack / "tokenizer" / "vocab.json").write_text("{}")
        mtp = self.data / "mtp" / "rt"
        mtp.mkdir(parents=True)
        (mtp / "experts.bin").write_text("m")
        self.ran = []

    def engine(self, kind, exe_text="--numa\0"):
        """kind: 'prebuilt' (upstream's release) or 'local' (compiled from this source)."""
        meta = PREBUILT if kind == "prebuilt" else {"source": "local", "version": "0.1.31", "archs": [89], "vision": "none",
                                                     "cuda_dirs": [], "src": None}
        with mock.patch.object(setup, "ROOT", self.root):
            if kind == "local":
                meta = {**meta, "src": setup.source_hash(setup.ENGINE_SOURCES)}
        (self.root / "engine" / "BUILD.json").write_text(json.dumps(meta))
        (self.root / "engine" / setup.EXE).write_text(exe_text)

    def pack_files(self, *names):
        for n in names:
            (self.pack / n).write_text(n)

    def left(self):
        return sorted(p.name for p in self.pack.iterdir())

    def main(self, *args, root=None):
        root = root or self.root
        buf = io.StringIO()

        def fake_run(cmd, *a, **k):
            self.ran.append(Path(str(cmd[1])).name if len(cmd) > 1 else str(cmd[0]))
            return mock.Mock(returncode=0)

        class FakeGGUF:
            def __init__(self, path):
                self.tensors = [types.SimpleNamespace(name="per_layer_token_embd.weight")]

        patches = [
            mock.patch.object(sys, "argv", ["setup.py", *args, "--data-dir", str(self.data)]),
            mock.patch.object(setup, "download", side_effect=AssertionError("download attempted")),
            mock.patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError("network used")),
            mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "WIN", False),
            mock.patch.object(setup, "gpus", return_value=[RTX40]), mock.patch.object(setup, "amd_gpus", return_value=[]),
            mock.patch.object(setup, "show_missing_gpus"),
            mock.patch.object(setup, "cpu_info", return_value=self.cpu), mock.patch.object(setup, "cpu_has_vbmi", return_value=self.vbmi),
            mock.patch.object(setup, "ram_gb", return_value=128.0), mock.patch.object(setup, "free_gb", return_value=2000.0),
            mock.patch.object(setup, "pip_install"), mock.patch.object(setup, "get_llama_cpp", return_value=self.tmp / "llama"),
            mock.patch.object(setup, "get_prebuilt", side_effect=lambda *a, **k: root / "engine"),
            mock.patch.object(setup, "build_engine", side_effect=lambda *a, **k: root / "engine"),
            mock.patch.object(setup, "run", fake_run), mock.patch.object(setup, "check_shards"),
            mock.patch.object(setup, "load_settings", return_value={}), mock.patch.object(setup, "save_settings"),
            mock.patch.object(setup, "page_file_gb", return_value=None), mock.patch.object(setup, "refresh_draft_vocab"),
            mock.patch.object(setup, "saved_calibration", return_value=None), mock.patch.object(setup, "cuda_lib_dirs", return_value=[]),
            mock.patch.object(setup, "start", return_value=0),        # (a run without --no-start would start the server)
            mock.patch.dict(sys.modules, {"gguf_reader": types.SimpleNamespace(GGUFFile=FakeGGUF)}),
        ]
        with contextlib.ExitStack() as st, contextlib.redirect_stdout(buf):
            for p in patches:
                st.enter_context(p)
            try:
                rc = setup.main()
            except SystemExit as e:
                rc = e
        self.out = buf.getvalue()
        return rc

    SETUP = ("--yes", "--model", "Q2_0", "--family", "qwen", "--context", "32768", "--vision", "no", "--no-start")

    def pack_tools(self):
        return [r for r in self.ran if r in PACK_TOOLS]


class PackFormChoice(FakePC):
    def test_cascade_lake_with_the_ready_made_engine_gets_the_native_pack(self):
        self.engine("prebuilt")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), ["iq_pack.py"])                 # never strata_pack.py: that pack needs VBMI there
        self.assertIn("AVX-512 VNNI but not VBMI", self.out)                # and it says why, and what to do
        self.assertIn("--setup --build", self.out)

    def test_cascade_lake_with_an_engine_compiled_here_gets_the_canonical_pack(self):
        self.engine("local")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), ["strata_pack.py", "pack_index.py"])
        self.assertNotIn("not VBMI", self.out)

    def test_ice_lake_keeps_the_canonical_pack_with_the_ready_made_engine(self):
        self.cpu, self.vbmi = (XEON, True, True), True
        self.engine("prebuilt")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), ["strata_pack.py", "pack_index.py"])

    def test_no_avx512_native_pack_either_way(self):
        self.cpu = ("AMD Ryzen 9 5900X", True, False)
        for kind in ("prebuilt", "local"):
            self.ran = []
            self.engine(kind)
            self.assertEqual(self.main(*self.SETUP), 0, self.out)
            self.assertEqual(self.pack_tools(), ["iq_pack.py"], kind)

    def test_other_models_are_native_whatever_the_engine(self):
        self.engine("local")
        models = self.data / "models" / "IQ2_XS"
        models.mkdir()
        for i in (1, 2):
            shard = models / f"Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-0000{i}-of-00002.gguf"
            shard.write_text("g")
            shard.with_name(shard.name + ".done").write_text("x")
        (self.data / "packs" / "iq2_xs" / "tokenizer").mkdir(parents=True)
        (self.data / "packs" / "iq2_xs" / "tokenizer" / "vocab.json").write_text("{}")
        self.assertEqual(self.main("--yes", "--model", "IQ2_XS", "--family", "qwen", "--context", "32768", "--vision", "no",
                                   "--no-start"), 0, self.out)
        self.assertEqual(self.pack_tools(), ["iq_pack.py"])

    def test_a_native_pack_is_converted_when_the_engine_now_compiled_here_reads_canonical(self):
        # an upstream install on a Cascade Lake (native pack), set up again with an engine compiled from this source
        self.engine("local")
        self.pack_files("native_experts.txt", "index.txt", "dense.bin", "conversions.json")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), ["strata_pack.py", "pack_index.py"])
        self.assertEqual(self.left(), ["tokenizer"])                       # no native_experts.txt left to mislead the engine
        self.assertIn("holds the native form", self.out)

    def test_a_native_pack_stays_for_the_ready_made_engine(self):
        self.engine("prebuilt")
        self.pack_files("native_experts.txt", "index.txt", "dense.bin")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), [])
        self.assertEqual(self.left(), ["dense.bin", "index.txt", "native_experts.txt", "tokenizer"])

    def test_a_canonical_pack_is_replaced_for_the_ready_made_engine(self):
        # the data folder shared with a Strata folder that has an engine compiled here: this folder's engine cannot read it
        self.engine("prebuilt")
        self.pack_files("manifest.json", "index.txt", "dense.bin", "embd.bin", "experts.bin")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), ["iq_pack.py"])
        self.assertEqual(self.left(), ["tokenizer"])

    def test_a_canonical_pack_stays_for_an_engine_compiled_here(self):
        self.engine("local")
        self.pack_files("manifest.json", "index.txt", "dense.bin", "embd.bin", "experts.bin")
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        self.assertEqual(self.pack_tools(), [])
        self.assertEqual(len(self.left()), 6)


class NumaSurvives(FakePC):
    def config(self, folder: Path, **extra):
        cfg = {"exe": str(folder / "engine" / setup.EXE), "cwd": str(folder), "model_name": "qwen3.8-flash-next-q2_0",
               "args": ["--pack", "p", "--max-context", "32768", "--kv", "int8"], "port": 8080, **extra}
        (folder / "strata-q2_0.json").write_text(json.dumps(cfg))

    def saved(self, folder=None):
        return json.loads(((folder or self.root) / "strata-q2_0.json").read_text())

    def test_setup_again_keeps_a_saved_off(self):
        self.engine("local")
        self.config(self.root, numa="off", args=["--max-context", "32768", "--numa", "off"])
        self.assertEqual(self.main(*self.SETUP), 0, self.out)
        cfg = self.saved()
        self.assertEqual(cfg["numa"], "off")
        self.assertEqual(cfg["args"][-2:], ["--numa", "off"])

    def test_an_explicit_numa_wins_over_the_saved_one(self):
        self.engine("local")
        self.config(self.root, numa="off")
        self.assertEqual(self.main(*self.SETUP, "--numa", "mirror"), 0, self.out)
        self.assertEqual(self.saved()["numa"], "mirror")
        self.assertEqual(self.main(*self.SETUP, "--numa", "auto"), 0, self.out)
        cfg = self.saved()
        self.assertEqual(cfg["numa"], "auto")
        self.assertNotIn("--numa", cfg["args"])

    def test_a_new_folder_adopting_the_install_takes_the_numa_setting(self):
        old = self.tmp / "Strata-old"
        old.mkdir()
        (old / "setup.py").write_text("")
        (old / "engine").mkdir()
        self.config(old, numa="mirror")
        self.engine("local")
        with mock.patch.object(setup, "load_settings", return_value={"installs": [str(old)]}):
            self.assertEqual(self.main(), 0, self.out)               # no arguments: it sets itself up like the earlier one
        self.assertIn("Found your earlier install", self.out)
        cfg = self.saved()
        self.assertEqual(cfg["numa"], "mirror")
        self.assertEqual(cfg["args"][-2:], ["--numa", "mirror"])

    def test_the_ready_made_engine_is_not_given_an_option_it_does_not_know(self):
        self.engine("prebuilt", exe_text="the upstream engine: no such option")
        self.assertEqual(self.main(*self.SETUP, "--numa", "mirror"), 0, self.out)
        cfg = self.saved()
        self.assertEqual(cfg["numa"], "mirror")                          # kept for an engine that has it
        self.assertNotIn("--numa", cfg["args"])
        self.assertIn("no such option", self.out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
