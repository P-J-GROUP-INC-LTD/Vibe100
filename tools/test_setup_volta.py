"""tools/test_setup_volta.py - setup.py's V100 (Volta, sm_70) handling - no GPU, no network, nothing installed.

    python3 tools/test_setup_volta.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import setup  # noqa: E402

V100 = {"index": 0, "name": "Tesla V100-SXM2-32GB", "vram_gb": 32.0, "arch": "70", "driver": "570.172"}
V100_16 = {"index": 0, "name": "Tesla V100-PCIE-16GB", "vram_gb": 16.0, "arch": "70", "driver": "570.172"}
RTX20 = {"index": 1, "name": "RTX 2080 Ti", "vram_gb": 11.0, "arch": "75", "driver": "580.1"}
RTX40 = {"index": 2, "name": "RTX 4090", "vram_gb": 24.0, "arch": "89", "driver": "580.1"}
RTX50 = {"index": 3, "name": "RTX 5090", "vram_gb": 32.0, "arch": "120", "driver": "580.1"}
P4000 = {"index": 4, "name": "Quadro P4000", "vram_gb": 8.0, "arch": "61", "driver": "580.1"}
XAVIER = {"index": 5, "name": "Xavier", "vram_gb": 16.0, "arch": "72", "driver": "580.1"}


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            return fn(*a, **kw), buf.getvalue()
        except SystemExit as e:
            return e, buf.getvalue()


def fake_nvcc(path: Path, release: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {release}, V{release}.1'\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class Remap:
    """setup.Path with /usr/local and /opt (and the Windows toolkit folder) redirected into a temp folder."""

    def __init__(self, tmp: Path):
        self.tmp = tmp

    def __call__(self, *a, **kw):
        if a and isinstance(a[0], (str, os.PathLike)):
            s = str(a[0])
            for src, dst in (("/usr/local", "usr_local"), ("/opt", "opt"),
                             (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", "wincuda")):
                if s == src or s.startswith(src + "/"):
                    return Path(self.tmp / dst / s[len(src):].lstrip("/"), *a[1:], **kw)
        return Path(*a, **kw)


class GpuProblem(unittest.TestCase):
    def test_volta_admitted_same_vram_rules(self):
        self.assertIsNone(setup.gpu_problem(V100))
        self.assertIsNone(setup.gpu_problem(V100_16))
        self.assertIsNone(setup.gpu_problem(V100, together=True))
        small = {**V100, "vram_gb": 6.0}
        self.assertIn("not supported together", setup.gpu_problem(small, together=True))
        self.assertIsNone(setup.gpu_problem(small))

    def test_others(self):
        self.assertIsNone(setup.gpu_problem(RTX20))
        self.assertIsNone(setup.gpu_problem(RTX50))
        p = setup.gpu_problem(P4000)
        self.assertIn("older than Volta (V100", p)
        self.assertIn("6.1", p)
        self.assertIn("not a desktop GPU", setup.gpu_problem(XAVIER))

    def test_cc(self):
        self.assertEqual(setup.cc(V100), "7.0")


class Selection(unittest.TestCase):
    def test_needs_cuda12(self):
        self.assertTrue(setup.needs_cuda12(["70"]))
        self.assertTrue(setup.needs_cuda12([70, 89]))
        self.assertFalse(setup.needs_cuda12(["75", "120"]))
        self.assertFalse(setup.needs_cuda12([]))

    def test_conflict_is_only_volta_with_rtx50(self):
        self.assertTrue(setup.arch_conflict(["70", "120"]))
        self.assertFalse(setup.arch_conflict(["70", "75", "80", "86", "89"]))
        self.assertFalse(setup.arch_conflict(["75", "120"]))
        self.assertFalse(setup.arch_conflict(["70"]))
        self.assertFalse(setup.arch_conflict(["120"]))

    def test_together_ok(self):
        self.assertEqual([g["index"] for g in setup.together_ok([V100, RTX40])], [2, 0])      # newest first
        self.assertEqual(setup.together_ok([V100, RTX50]), [])
        self.assertEqual([g["index"] for g in setup.together_ok([V100, RTX40, RTX50])], [3, 2])

    def test_check_gpus_refuses_v100_with_rtx50(self):
        found = [V100, RTX50]
        r, out = quiet(setup.check_gpus, [0, 3], found)
        self.assertIsInstance(r, SystemExit)
        self.assertIn("cannot share one engine", out)
        self.assertIn("CUDA 13", out)
        self.assertIn("--gpu", out)
        # each alone, and V100 + RTX 4090, are fine
        self.assertIsNone(quiet(setup.check_gpus, [0], found)[0])
        self.assertIsNone(quiet(setup.check_gpus, [3], found)[0])
        self.assertIsNone(quiet(setup.check_gpus, [0, 2], [V100, RTX40])[0])

    def test_check_gpus_hint_names_what_works(self):
        found = [V100, RTX40, RTX50]
        r, out = quiet(setup.check_gpus, [0, 3], found)
        self.assertIsInstance(r, SystemExit)
        self.assertIn("--gpus 3,2", out)          # the RTX 50 + RTX 40 pair

    def test_parse_all_skips_the_v100_beside_an_rtx50(self):
        found = [V100, RTX40, RTX50]
        self.assertEqual(setup.parse_gpus("all", found), [3, 2])
        r, out = quiet(setup.parse_gpus, "all", [V100, RTX50])
        self.assertIsInstance(r, SystemExit)
        self.assertIn("Note:", out)               # the table explains the conflict

    def test_choose_gpus_v100_and_rtx50_runs_one_card(self):
        class A:
            gpus = None
            gpu = None
            yes = True
            check = False
        with mock.patch.object(setup, "say", lambda *a, **k: None):
            self.assertEqual(setup.choose_gpus(A(), [V100_16, RTX50]), [3])      # the most VRAM, no pair offered
            self.assertEqual(setup.choose_gpus(A(), [V100, RTX50]), [0])         # a 32 GB tie: upstream's lower number

    def test_choose_gpus_v100_alone(self):
        class A:
            gpus = None
            gpu = None
            yes = True
            check = False
        self.assertEqual(setup.choose_gpus(A(), [V100]), [0])

    def test_choose_gpus_v100_and_rtx40_offers_pair(self):
        class A:
            gpus = None
            gpu = None
            yes = True
            check = False
        r, out = quiet(setup.choose_gpus, A(), [V100, RTX40])
        self.assertEqual(r, [2, 0])               # recommended pair, newest first

    def test_nothing_usable_hint_mentions_v100(self):
        class A:
            gpus = None
            gpu = None
            yes = True
            check = False
        r, out = quiet(setup.choose_gpus, A(), [P4000])
        self.assertIsInstance(r, SystemExit)
        self.assertIn("V100", out)


class Prebuilt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(setup, "ROOT", self.tmp)
        self.patch.start()
        (self.tmp / "engine").mkdir()
        self.addCleanup(self.patch.stop)

    def install(self, meta):
        (self.tmp / "engine" / "BUILD.json").write_text(json.dumps(meta))
        (self.tmp / "engine" / setup.EXE).write_text("x")

    def no_network(self):
        return mock.patch.object(setup.urllib.request, "urlopen", side_effect=AssertionError("network used"))

    def test_v100_never_gets_prebuilt_no_engine_installed(self):
        with self.no_network():
            r, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {**V100, "archs": ["70"]}, "gpu")
        self.assertIsNone(r)
        self.assertIn("CUDA 12.8", out)
        r, out = quiet(setup.get_prebuilt, setup.PREBUILT_URL, V100, "gpu")        # no "archs" key
        self.assertIsNone(r)

    def test_v100_with_a_prebuilt_installed(self):
        for meta in ({"version": "0.1.31", "archs": [75, 80, 86, 89, 120], "ptx": True},
                     {"version": "0.1.31", "archs": [75, 80, 86, 89, 120]},
                     {"version": "0.1.31", "archs": []}):                         # even a BUILD.json without archs
            self.install(meta)
            with self.no_network():
                self.assertIsNone(quiet(setup.get_prebuilt, setup.PREBUILT_URL, V100, "gpu")[0], meta)
                self.assertEqual(quiet(setup.get_prebuilt, setup.PREBUILT_URL, RTX40, "gpu")[0], self.tmp / "engine", meta)

    def test_v100_plus_rtx40_also_compiled(self):
        with self.no_network():
            r, _ = quiet(setup.get_prebuilt, setup.PREBUILT_URL, {**RTX40, "archs": ["70", "89"]}, "gpu")
        self.assertIsNone(r)

    def test_rtx_cards_unchanged_download_path_reached(self):
        # an RTX 4090 with nothing installed still goes for the download (the network is asked, here refused)
        with mock.patch.object(setup.urllib.request, "urlopen", side_effect=OSError("offline")):
            r, out = quiet(setup.get_prebuilt, "https://example.invalid/", {**RTX40, "archs": ["89"]}, "gpu")
        self.assertIsNone(r)
        self.assertIn("compiling instead", out)

    def test_engine_runs_on(self):
        self.install({"version": "0.1.31", "archs": [75, 80, 86, 89, 120], "ptx": True})
        self.assertFalse(setup.engine_runs_on(V100))
        self.assertTrue(setup.engine_runs_on(RTX40))
        self.install({"source": "local", "version": "0.1.31", "archs": [70]})
        self.assertTrue(setup.engine_runs_on(V100))
        self.assertFalse(setup.engine_runs_on(RTX40))

    def test_update_never_replaces_a_local_v100_engine_with_a_prebuilt(self):
        self.install({"source": "local", "version": "0.1.31", "archs": [70], "src": "old", "vision": "none"})
        built = []
        with self.no_network(), mock.patch.object(setup, "gpu_info", return_value={**V100, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=lambda g, *a: built.append(g["archs"])), \
                mock.patch.object(setup, "get_prebuilt", side_effect=AssertionError("prebuilt asked for")):
            quiet(setup.update_installed_engine, setup.PREBUILT_URL)
        self.assertEqual(built, [[70]])

    def test_update_local_v100_engine_when_an_rtx50_is_the_biggest_card(self):
        self.install({"source": "local", "version": "0.1.31", "archs": [70], "src": "old", "vision": "none"})
        built = []
        with self.no_network(), mock.patch.object(setup, "gpu_info", return_value={**RTX50, "count": 2}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=lambda g, *a: built.append(g["archs"])):
            quiet(setup.update_installed_engine, setup.PREBUILT_URL)
        self.assertEqual(built, [[70]])           # rebuilt for the cards it was made for, not 70+120

    def test_update_local_v100_engine_gets_the_rtx40_too(self):
        self.install({"source": "local", "version": "0.1.31", "archs": [70], "src": "old", "vision": "none"})
        built = []
        with self.no_network(), mock.patch.object(setup, "gpu_info", return_value={**RTX40, "count": 2}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=lambda g, *a: built.append(g["archs"])):
            quiet(setup.update_installed_engine, setup.PREBUILT_URL)
        self.assertEqual(built, [[70, 89]])

    def test_update_prebuilt_with_v100_biggest_keeps_installed_engine(self):
        meta = {"version": "0.1.20", "archs": [75, 80, 86, 89, 120], "ptx": True}
        self.install(meta)
        with self.no_network(), mock.patch.object(setup, "gpu_info", return_value={**V100, "count": 2}):
            r, out = quiet(setup.update_installed_engine, setup.PREBUILT_URL)
        self.assertIn("could not update the engine", out)
        self.assertEqual(json.loads((self.tmp / "engine" / "BUILD.json").read_text()), meta)

    def test_ensure_engine_for_a_v100_added_to_a_prebuilt_install(self):
        self.install({"version": "0.1.31", "archs": [75, 80, 86, 89, 120], "ptx": True})
        got = []
        cfg_path = self.tmp / "cfg.json"
        cfg_path.write_text("{}")

        def fake_build(g, vision, yes, llama):
            got.append(g["archs"])
            (self.tmp / "engine" / "BUILD.json").write_text(json.dumps({"source": "local", "archs": g["archs"],
                                                                         "cuda_dirs": ["/c"]}))
        with mock.patch.object(setup, "gpu_info", return_value={**V100, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=fake_build):
            r, out = quiet(setup.ensure_engine_for, [V100], cfg_path, {}, True)
        self.assertEqual(got, [[70]])
        self.assertIn("no code for", out)

    def test_ensure_engine_for_rtx_card_added_to_prebuilt_unchanged(self):
        self.install({"version": "0.1.31", "archs": [80, 86], "ptx": False})
        got = []
        cfg_path = self.tmp / "cfg.json"
        cfg_path.write_text("{}")

        def fake_build(g, vision, yes, llama):
            got.append(g["archs"])
            (self.tmp / "engine" / "BUILD.json").write_text(json.dumps({"source": "local", "archs": g["archs"]}))
        with mock.patch.object(setup, "gpu_info", return_value={**RTX40, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=fake_build):
            quiet(setup.ensure_engine_for, [{**RTX20, "arch": "75"}], cfg_path, {}, True)
        self.assertEqual(got, [[75, 80, 86]])     # upstream behaviour: the generations it had, plus the new one

    def test_build_engine_refuses_v100_with_rtx50(self):
        r, out = quiet(setup.build_engine, {**RTX50, "archs": ["70", "120"]}, "none", True, Path("/x"))
        self.assertIsInstance(r, SystemExit)
        self.assertIn("cannot be built for sm_70, sm_120", out)
        self.assertIn("--gpu", out)

    def test_build_engine_refuses_an_rtx50_added_to_a_local_v100_engine(self):
        self.install({"source": "local", "version": "0.1.31", "archs": [70], "src": "x", "vision": "none"})
        r, out = quiet(setup.build_engine, {**RTX50, "archs": ["120"]}, "none", True, Path("/x"))
        self.assertIsInstance(r, SystemExit)
        self.assertIn("cannot be built for sm_70, sm_120", out)


class Toolkits(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.paths = mock.patch.object(setup, "Path", Remap(self.tmp))
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.on_path = None
        self.which = mock.patch.object(setup.shutil, "which", side_effect=lambda n: self.on_path if n == "nvcc" else "/usr/bin/" + n)
        self.which.start()
        self.addCleanup(self.which.stop)
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("CUDA_PATH", None)
        self.addCleanup(self.env.stop)

    def tool(self, name, release):
        return fake_nvcc(self.tmp / "usr_local" / name / "bin" / "nvcc", release)

    def test_default_picks_newest(self):
        self.on_path = self.tool("cuda-13.0", "13.0")
        self.tool("cuda-12.8", "12.8")
        self.assertEqual(setup.find_nvcc()[1], (13, 0))                    # unchanged behaviour
        self.assertEqual(setup.find_nvcc([89])[1], (13, 0))
        self.assertEqual(setup.find_nvcc([75, 120])[1], (13, 0))

    def test_volta_passes_over_cuda13_for_a_12(self):
        self.on_path = self.tool("cuda-13.0", "13.0")
        p128 = self.tool("cuda-12.8", "12.8")
        self.assertEqual(setup.find_nvcc([70]), (p128, (12, 8)))
        self.assertEqual(setup.find_nvcc(["70", "89"]), (p128, (12, 8)))
        self.assertEqual(setup.find_nvcc([70])[0], p128)

    def test_volta_prefers_newest_12(self):
        self.on_path = self.tool("cuda-13.0", "13.0")
        self.tool("cuda-12.8", "12.8")
        p129 = self.tool("cuda-12.9", "12.9")
        self.assertEqual(setup.find_nvcc([70]), (p129, (12, 9)))

    def test_volta_with_only_cuda13(self):
        self.on_path = self.tool("cuda-13.0", "13.0")
        self.assertEqual(setup.find_nvcc([70]), (None, None))

    def test_volta_opt_cuda(self):
        p = fake_nvcc(self.tmp / "opt" / "cuda-12.8" / "bin" / "nvcc", "12.8")
        self.assertEqual(setup.find_nvcc([70])[0], p)

    def test_volta_windows_layout(self):
        p = fake_nvcc(self.tmp / "wincuda" / "v12.8" / "bin" / "nvcc.exe", "12.8")
        fake_nvcc(self.tmp / "wincuda" / "v13.0" / "bin" / "nvcc.exe", "13.0")
        with mock.patch.object(setup, "WIN", True):
            self.assertEqual(setup.find_nvcc([70])[0], p)
            self.assertEqual(setup.find_nvcc()[1], (13, 0))

    def test_install_build_tools_uses_the_12_next_to_a_13(self):
        self.on_path = self.tool("cuda-13.0", "13.0")
        p128 = self.tool("cuda-12.8", "12.8")
        r, out = quiet(setup.install_build_tools, {**V100, "archs": ["70"]}, True)
        self.assertEqual(r[0], p128)
        # an RTX 4090 alone keeps what it did: the newest
        r, out = quiet(setup.install_build_tools, {**RTX40, "archs": ["89"]}, True)
        self.assertEqual(r[0], self.on_path)

    def test_install_build_tools_ubuntu_installs_cuda_toolkit_12_8(self):
        self.on_path = self.tool("cuda-13.0", "13.0")
        cmds = []

        def fake_run(cmd, *a, **k):
            cmds.append([str(c) for c in cmd])
            if cmd[:3] == ["sudo", "apt-get", "install"] and cmd[-1] == "cuda-toolkit-12-8":
                self.tool("cuda-12.8", "12.8")
            return mock.Mock(returncode=0)
        real_open = open
        osr = "ID=ubuntu\nVERSION_ID=\"24.04\"\n"

        def fake_open(f, *a, **k):
            if f == "/etc/os-release":
                import io as _io
                return _io.StringIO(osr)
            return real_open(f, *a, **k)
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup, "download", lambda *a, **k: None), \
                mock.patch.object(setup, "open", fake_open, create=True), mock.patch.object(setup, "WIN", False):
            r, out = quiet(setup.install_build_tools, {**V100, "archs": ["70"]}, True)
        self.assertTrue(any(c[-1] == "cuda-toolkit-12-8" for c in cmds), cmds)
        self.assertFalse(any("cuda-toolkit-13-0" in c for c in cmds), cmds)
        self.assertIn("CUDA 13", out)             # says why
        self.assertTrue(r[0].endswith("cuda-12.8/bin/nvcc"), r)

    def test_install_build_tools_ubuntu_rtx_still_installs_13(self):
        cmds = []

        def fake_run(cmd, *a, **k):
            cmds.append([str(c) for c in cmd])
            if cmd[:3] == ["sudo", "apt-get", "install"] and cmd[-1] == "cuda-toolkit-13-0":
                self.tool("cuda-13.0", "13.0")
            return mock.Mock(returncode=0)
        real_open = open

        def fake_open(f, *a, **k):
            if f == "/etc/os-release":
                import io as _io
                return _io.StringIO("ID=ubuntu\nVERSION_ID=\"22.04\"\n")
            return real_open(f, *a, **k)
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup, "download", lambda *a, **k: None), \
                mock.patch.object(setup, "open", fake_open, create=True), mock.patch.object(setup, "WIN", False):
            r, out = quiet(setup.install_build_tools, {**RTX40, "archs": ["89"]}, True)
        self.assertTrue(any(c[-1] == "cuda-toolkit-13-0" for c in cmds), cmds)

    def test_install_build_tools_windows_winget_12_8(self):
        cmds = []
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "find_vcvars", return_value=Path("/vc")), \
                mock.patch.object(setup, "run", lambda cmd, *a, **k: cmds.append([str(c) for c in cmd])):
            r, out = quiet(setup.install_build_tools, {**V100, "archs": ["70"]}, True)
        self.assertIsInstance(r, SystemExit)       # nothing really installed: "did not install", with the manual link
        self.assertIn("cuda-12-8-1-download-archive", out)
        wg = [c for c in cmds if "Nvidia.CUDA" in c]
        self.assertEqual(len(wg), 1)
        self.assertEqual(wg[0][wg[0].index("--version") + 1], "12.8")
        self.assertEqual(wg[0][wg[0].index("--id") + 1], "Nvidia.CUDA")

    def test_install_build_tools_refuses_conflict(self):
        r, out = quiet(setup.install_build_tools, {**RTX50, "archs": ["70", "120"]}, True)
        self.assertIsInstance(r, SystemExit)


    def _apt(self, installed):
        cmds = []

        def fake_run(cmd, *a, **k):
            cmds.append([str(c) for c in cmd])
            if cmd[:3] == ["sudo", "apt-get", "install"] and cmd[-1] == "cuda-toolkit-12-8":
                self.tool("cuda-12.8", "12.8")
            return mock.Mock(returncode=0)
        real_open = open

        def fake_open(f, *a, **k):
            if f == "/etc/os-release":
                import io as _io
                return _io.StringIO("ID=ubuntu\nVERSION_ID=\"24.04\"\n")
            return real_open(f, *a, **k)
        return cmds, fake_run, fake_open

    def test_volta_with_only_the_ubuntu_cuda_12_0_installs_12_8(self):
        # Ubuntu 24.04's own nvidia-cuda-toolkit is 12.0, whose nvcc rejects g++ 13: treated like a missing toolkit
        self.on_path = self.tool("cuda-12.0", "12.0")
        cmds, fake_run, fake_open = self._apt(False)
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup, "download", lambda *a, **k: None), \
                mock.patch.object(setup, "open", fake_open, create=True), mock.patch.object(setup, "WIN", False):
            r, out = quiet(setup.install_build_tools, {**V100, "archs": ["70"]}, True)
        self.assertTrue(any(c[-1] == "cuda-toolkit-12-8" for c in cmds), cmds)
        self.assertIn("12.0 you have is too old", out)
        self.assertTrue(r[0].endswith("cuda-12.8/bin/nvcc"), r)

    def test_volta_with_cuda_12_4_is_enough(self):
        self.on_path = self.tool("cuda-12.4", "12.4")
        cmds, fake_run, fake_open = self._apt(False)
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup, "open", fake_open, create=True):
            r, out = quiet(setup.install_build_tools, {**V100, "archs": ["70"]}, True)
        self.assertEqual(cmds, [])
        self.assertEqual(r[0], self.on_path)

    def test_rtx_cards_keep_cuda_12_0_as_good_enough(self):
        self.on_path = self.tool("cuda-12.0", "12.0")
        cmds, fake_run, fake_open = self._apt(False)
        with mock.patch.object(setup, "run", fake_run), mock.patch.object(setup, "open", fake_open, create=True):
            r, out = quiet(setup.install_build_tools, {**RTX40, "archs": ["89"]}, True)
        self.assertEqual(cmds, [])
        self.assertEqual(r[0], self.on_path)


class BuildFolder(unittest.TestCase):
    def mk(self, tmp, cuda, archs):
        b = tmp / "build"
        (b / "CMakeFiles" / "3.28.3").mkdir(parents=True)
        (b / "CMakeFiles" / "3.28.3" / "CMakeCUDACompiler.cmake").write_text(f'set(CMAKE_CUDA_COMPILER_VERSION "{cuda}.0.88")\n')
        (b / "CMakeCache.txt").write_text(f"CMAKE_CUDA_ARCHITECTURES:STRING={archs}\n")
        return b

    def test_fresh_only_when_volta_and_major_changed(self):
        tmp = Path(tempfile.mkdtemp())
        n128 = fake_nvcc(tmp / "a" / "nvcc", "12.8")
        n130 = fake_nvcc(tmp / "b" / "nvcc", "13.0")
        b = self.mk(tmp, 13, "75;89")
        quiet(setup.fresh_build_folder, b, n128, [89])                 # no Volta: left as it was (upstream behaviour)
        self.assertTrue(b.exists())
        quiet(setup.fresh_build_folder, b, n128, [70])                 # a V100 engine over a CUDA 13 folder
        self.assertFalse(b.exists())
        b = self.mk(tmp, 12, "70")
        quiet(setup.fresh_build_folder, b, n128, [70])                 # same toolkit major: only what changed
        self.assertTrue(b.exists())
        quiet(setup.fresh_build_folder, b, n130, [89])                 # the folder was a V100's, now CUDA 13
        self.assertFalse(b.exists())
        quiet(setup.fresh_build_folder, tmp / "none", n128, [70])      # nothing there: nothing happens


class EngineFolder(unittest.TestCase):
    """A temporary Strata folder with an engine/ in it, and fakes for the compiler and cmake (no tests of its own)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.patch = mock.patch.object(setup, "ROOT", self.tmp)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        (self.tmp / "engine").mkdir()
        (self.tmp / "CMakeLists.txt").write_text("project(strata VERSION 0.1.31 LANGUAGES CXX)")
        self.calls = []

    def install(self, meta, vision_exe=None):
        meta = {"source": "local", "version": "0.1.31", "vision": "none", **meta}
        if meta.get("src") == "current":
            meta["src"] = setup.source_hash(setup.ENGINE_SOURCES)
        (self.tmp / "engine" / "BUILD.json").write_text(json.dumps(meta))
        (self.tmp / "engine" / setup.EXE).write_text("x")
        if vision_exe:
            (self.tmp / "engine" / setup.VEXE).write_text(vision_exe)

    def fake_tools(self, nvcc="/usr/local/cuda-12.8/bin/nvcc"):
        return mock.patch.object(setup, "install_build_tools",
                                 side_effect=lambda g, y: self.calls.append(("tools", g["archs"])) or (nvcc, None))

    def fake_cmake(self):
        def build(src, bdir, target, defs, vcvars, bat):
            self.calls.append((target, [d for d in defs if d.startswith("-DCMAKE_CUDA_ARCHITECTURES")]))
            if target == "strata":
                (self.tmp / "build").mkdir(exist_ok=True)
                (self.tmp / "build" / setup.EXE).write_text("new engine")
            else:
                (self.tmp / "build-vision" / "bin").mkdir(parents=True, exist_ok=True)
                (self.tmp / "build-vision" / "bin" / setup.VEXE).write_text("new encoder")
        return mock.patch.object(setup, "cmake_build", side_effect=build)

    def archs_in_build_json(self):
        return json.loads((self.tmp / "engine" / "BUILD.json").read_text())["archs"]


class ArchSet(EngineFolder):
    """The architectures of an engine compiled here follow the cards the PC has: a generation whose card is gone no longer
    pins it (audit finding: the set only grew, so a V100 plus an RTX 50 stopped setup for good)."""

    def test_still_present_keeps_only_cards_that_are_there(self):
        with mock.patch.object(setup, "gpus", return_value=[RTX20, RTX40]):
            self.assertEqual(setup.still_present({70, 75, 86, 89}), {75, 89})
        with mock.patch.object(setup, "gpus", return_value=[]):               # nvidia-smi silent: nothing can be told
            self.assertEqual(setup.still_present({70, 75}), {70, 75})

    def test_v100_added_to_a_docker_rtx_list_is_compiled_for_alone(self):
        # BUILD.json [75,80,86,89,120] (the old Docker default, CUDA 12.8) + a V100: 70 + 120 is impossible, and no card needs 120
        self.install({"archs": [75, 80, 86, 89, 120], "src": "current"})
        with mock.patch.object(setup, "gpus", return_value=[V100]), self.fake_tools(), self.fake_cmake():
            r, out = quiet(setup.build_engine, {**V100, "archs": ["70"]}, "none", True, Path("/x"))
        self.assertNotIsInstance(r, SystemExit, out)
        self.assertEqual(self.calls[0], ("tools", [70]))
        self.assertEqual(self.archs_in_build_json(), [70])

    def test_v100_swapped_for_an_rtx5090(self):
        self.install({"archs": [70], "src": "current"})
        cfg = self.tmp / "cfg.json"
        cfg.write_text("{}")
        with mock.patch.object(setup, "gpus", return_value=[RTX50]), \
                mock.patch.object(setup, "gpu_info", return_value={**RTX50, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                self.fake_tools("/usr/local/cuda-13.0/bin/nvcc"), self.fake_cmake():
            r, out = quiet(setup.ensure_engine_for, [RTX50], cfg, {}, True)
        self.assertIsInstance(r, dict, out)
        self.assertEqual(self.calls[0], ("tools", [120]))
        self.assertEqual(self.archs_in_build_json(), [120])

    def test_v100_and_rtx5090_both_in_the_pc_are_still_refused(self):
        self.install({"archs": [70], "src": "current"})
        with mock.patch.object(setup, "gpus", return_value=[V100, RTX50]), self.fake_tools(), self.fake_cmake():
            r, out = quiet(setup.build_engine, {**RTX50, "archs": ["120"]}, "none", True, Path("/x"))
        self.assertIsInstance(r, SystemExit)
        self.assertIn("cannot be built for sm_70, sm_120", out)
        self.assertEqual(self.calls, [])

    def test_the_other_cards_generations_are_kept(self):
        self.install({"archs": [75, 86], "src": "current"})
        with mock.patch.object(setup, "gpus", return_value=[RTX20, {**RTX40, "arch": "86"}, RTX40]), self.fake_tools(), \
                self.fake_cmake():
            quiet(setup.build_engine, {**RTX40, "archs": ["89"]}, "none", True, Path("/x"))
        self.assertEqual(self.archs_in_build_json(), [75, 86, 89])

    def test_a_card_that_is_gone_is_dropped(self):
        self.install({"archs": [75, 86], "src": "current"})
        with mock.patch.object(setup, "gpus", return_value=[RTX20, RTX40]), self.fake_tools(), self.fake_cmake():
            quiet(setup.build_engine, {**RTX40, "archs": ["89"]}, "none", True, Path("/x"))
        self.assertEqual(self.archs_in_build_json(), [75, 89])        # 86's card is not in the PC any more

    def test_update_of_a_local_engine_follows_the_cards_too(self):
        self.install({"archs": [70], "src": "old"})
        built = []
        with mock.patch.object(setup, "gpus", return_value=[RTX40]), \
                mock.patch.object(setup, "gpu_info", return_value={**RTX40, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=lambda g, *a: built.append(g["archs"])):
            quiet(setup.update_installed_engine, setup.PREBUILT_URL)
        self.assertEqual(built, [[89]])                               # the V100 it was made for is gone

    def test_update_refreshes_the_lib_dirs_of_every_config(self):
        self.install({"archs": [89], "src": "old"})
        for name, backend in (("strata-q2_0.json", None), ("strata-iq3_s.json", None), ("strata-hip.json", "hip")):
            (self.tmp / name).write_text(json.dumps({"lib_dirs": ["/old/cuda/bin"], **({"backend": backend} if backend else {})}))

        def rebuilt(g, *a):
            meta = json.loads((self.tmp / "engine" / "BUILD.json").read_text())
            meta.update({"src": "new", "cuda_dirs": ["/usr/local/cuda-12.9/bin"]})
            (self.tmp / "engine" / "BUILD.json").write_text(json.dumps(meta))
        with mock.patch.object(setup, "gpus", return_value=[RTX40]), \
                mock.patch.object(setup, "gpu_info", return_value={**RTX40, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), \
                mock.patch.object(setup, "build_engine", side_effect=rebuilt):
            quiet(setup.update_installed_engine, setup.PREBUILT_URL)
        self.assertEqual(json.loads((self.tmp / "strata-q2_0.json").read_text())["lib_dirs"],
                         ["/usr/local/cuda-12.9/bin", "/old/cuda/bin"])
        self.assertEqual(json.loads((self.tmp / "strata-iq3_s.json").read_text())["lib_dirs"][0], "/usr/local/cuda-12.9/bin")
        self.assertEqual(json.loads((self.tmp / "strata-hip.json").read_text())["lib_dirs"], ["/old/cuda/bin"])   # AMD: untouched

    def test_cuda_version_is_recorded(self):
        nvcc = fake_nvcc(self.tmp / "cuda-12.8" / "bin" / "nvcc", "12.8")
        with mock.patch.object(setup, "gpus", return_value=[V100]), self.fake_tools(nvcc), self.fake_cmake():
            quiet(setup.build_engine, {**V100, "archs": ["70"]}, "none", True, Path("/x"))
        self.assertEqual(json.loads((self.tmp / "engine" / "BUILD.json").read_text())["cuda"], "12.8")

    def test_cmake_dir_is_part_of_the_source_fingerprint(self):
        self.assertIn("cmake", setup.ENGINE_SOURCES)


class VisionEncoder(EngineFolder):
    """The image encoder the ready-made install brought (CUDA 13: no sm_70 code, needs libcudart.so.13) is not kept when the
    engine turns into one compiled here."""

    def test_prebuilt_install_plus_v100_rebuilds_the_encoder(self):
        (self.tmp / "engine" / "BUILD.json").write_text(json.dumps({"version": "0.1.31", "archs": [75, 80, 86, 89, 120],
                                                                    "ptx": True, "cuda": "13.0"}))
        (self.tmp / "engine" / setup.EXE).write_text("prebuilt")
        (self.tmp / "engine" / setup.VEXE).write_text("prebuilt-cuda13-encoder")
        cfg = self.tmp / "cfg.json"
        cfg.write_text("{}")
        with mock.patch.object(setup, "gpus", return_value=[V100]), \
                mock.patch.object(setup, "gpu_info", return_value={**V100, "count": 1}), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/x")), self.fake_tools(), self.fake_cmake():
            r, out = quiet(setup.ensure_engine_for, [V100], cfg, {"lib_dirs": ["/venv/nvidia/cu13/lib"]}, True)
        self.assertIsInstance(r, dict, out)
        self.assertEqual([c[0] for c in self.calls if c[0] != "tools"], ["strata", "strata-vision"])
        self.assertEqual((self.tmp / "engine" / setup.VEXE).read_text(), "new encoder")
        meta = json.loads((self.tmp / "engine" / "BUILD.json").read_text())
        self.assertEqual((meta["vision"], meta["archs"], meta["vision_archs"]), ("gpu", [70], [70]))
        self.assertEqual(meta["vision_src"], setup.source_hash(setup.VISION_SOURCES))

    def test_prebuilt_encoder_is_replaced_even_without_a_new_engine_compile_reason(self):
        # `--build` on a ready-made install: the engine AND the encoder are compiled (the encoder used to be kept)
        (self.tmp / "engine" / "BUILD.json").write_text(json.dumps({"version": "0.1.31", "archs": [75, 89]}))
        (self.tmp / "engine" / setup.EXE).write_text("prebuilt")
        (self.tmp / "engine" / setup.VEXE).write_text("prebuilt-encoder")
        with mock.patch.object(setup, "gpus", return_value=[RTX40]), self.fake_tools(), self.fake_cmake():
            quiet(setup.build_engine, {**RTX40, "archs": ["89"]}, "gpu", True, Path("/x"))
        self.assertEqual((self.tmp / "engine" / setup.VEXE).read_text(), "new encoder")

    def test_a_current_local_encoder_is_kept(self):
        self.install({"archs": [89], "src": "current", "vision": "gpu", "vision_src": setup.source_hash(setup.VISION_SOURCES),
                      "vision_archs": [89]}, vision_exe="local-encoder")
        with mock.patch.object(setup, "gpus", return_value=[RTX40]), self.fake_tools(), self.fake_cmake():
            r, out = quiet(setup.build_engine, {**RTX40, "archs": ["89"]}, "gpu", True, Path("/x"))
        self.assertIn("already built", out)
        self.assertEqual(self.calls, [])
        self.assertEqual((self.tmp / "engine" / setup.VEXE).read_text(), "local-encoder")

    def test_a_gpu_encoder_without_code_for_a_new_card_is_compiled_again(self):
        # a local RTX 40 engine + encoder, a V100 added: the engine grows to 70+89, and so must the encoder
        self.install({"archs": [89], "src": "current", "vision": "gpu", "vision_src": setup.source_hash(setup.VISION_SOURCES)},
                     vision_exe="encoder-for-89")                                    # (an old BUILD.json: no vision_archs)
        with mock.patch.object(setup, "gpus", return_value=[V100, RTX40]), self.fake_tools(), self.fake_cmake():
            quiet(setup.build_engine, {**RTX40, "archs": ["70", "89"]}, "gpu", True, Path("/x"))
        self.assertEqual(self.archs_in_build_json(), [70, 89])
        self.assertEqual((self.tmp / "engine" / setup.VEXE).read_text(), "new encoder")
        self.assertEqual(json.loads((self.tmp / "engine" / "BUILD.json").read_text())["vision_archs"], [70, 89])

    def test_a_cpu_encoder_has_no_arch_to_miss(self):
        self.install({"archs": [89], "src": "current", "vision": "cpu", "vision_src": setup.source_hash(setup.VISION_SOURCES)},
                     vision_exe="cpu-encoder")
        with mock.patch.object(setup, "gpus", return_value=[RTX40, V100]), self.fake_tools(), self.fake_cmake():
            quiet(setup.build_engine, {**RTX40, "archs": ["70", "89"]}, "cpu", True, Path("/x"))
        self.assertEqual([c[0] for c in self.calls if c[0] != "tools"], ["strata"])      # the engine only
        self.assertEqual((self.tmp / "engine" / setup.VEXE).read_text(), "cpu-encoder")


class MissingGpus(unittest.TestCase):
    """nvidia-smi lists fewer GPUs than the PCI bus has: the driver is the usual reason for a V100 (sysfs / proc faked)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def pci(self, addr, vendor, cls, device):
        d = self.tmp / "sys" / "bus" / "pci" / "devices" / addr
        d.mkdir(parents=True)
        for name, v in (("vendor", vendor), ("class", cls), ("device", device)):
            (d / name).write_text(v + "\n")

    def driver(self, line):
        d = self.tmp / "proc" / "driver" / "nvidia"
        d.mkdir(parents=True, exist_ok=True)
        (d / "version").write_text(line + "\nGCC version:  gcc version 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)\n")

    def notes(self, found):
        return setup.missing_gpu_notes(found, str(self.tmp / "sys"), str(self.tmp / "proc"))

    def test_only_nvidia_graphics_and_compute_controllers_count(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")      # V100 PCIe (3D controller)
        self.pci("0000:3b:00.1", "0x10de", "0x040300", "0x10f0")      # a GPU's audio function
        self.pci("0000:5e:00.0", "0x10de", "0x030000", "0x2684")      # RTX 4090 (VGA)
        self.pci("0000:00:02.0", "0x8086", "0x030000", "0x3e92")      # Intel
        self.assertEqual(setup.nvidia_pci_gpus(str(self.tmp / "sys")), [("0000:3b:00.0", "0x1db4"), ("0000:5e:00.0", "0x2684")])

    def test_nothing_missing_nothing_said(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")
        self.driver("NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.95.05  Wed Aug 20 2025")
        self.assertEqual(self.notes([V100]), [])
        self.assertEqual(setup.missing_gpu_notes([], str(self.tmp / "nowhere"), str(self.tmp / "proc")), [])   # no sysfs: no PCI GPUs

    def test_open_kernel_module_cannot_drive_a_v100(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")
        self.pci("0000:5e:00.0", "0x10de", "0x030000", "0x2684")
        self.driver("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.95.05  Release Build  (dvs-builder@U22)")
        text = " ".join(self.notes([RTX40]))
        self.assertIn("2 NVIDIA GPUs", text)
        self.assertIn("nvidia-smi lists 1", text)
        self.assertIn("open kernel module", text)
        self.assertIn("Turing", text)
        self.assertIn("nvidia-driver-580 (not nvidia-driver-580-open)", text)

    def test_a_driver_newer_than_580_has_dropped_volta(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")
        self.driver("NVRM version: NVIDIA UNIX x86_64 Kernel Module  590.48.01  Tue Sep  9 2025")
        text = " ".join(self.notes([]))
        self.assertIn("590", text)
        self.assertIn("R580 is the last driver branch that supports Volta", text)
        self.assertIn("apt-mark hold nvidia-driver-580", text)
        self.assertNotIn("open kernel module", text)

    def test_no_driver_loaded(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")
        self.assertIn("not loaded", " ".join(self.notes([])))

    def test_another_driver_a_generic_hint(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")
        self.driver("NVRM version: NVIDIA UNIX x86_64 Kernel Module  575.57.08  Thu Jul 10 2025")
        text = " ".join(self.notes([]))
        self.assertIn("570-580", text)
        self.assertNotIn("R580 is the last", text)

    def test_windows_says_nothing(self):
        self.pci("0000:3b:00.0", "0x10de", "0x030200", "0x1db4")
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "missing_gpu_notes",
                                                                     side_effect=AssertionError("looked")):
            quiet(setup.show_missing_gpus, [])


class Texts(unittest.TestCase):
    def test_no_stale_rtx20_only_claims(self):
        text = (REPO / "setup.py").read_text()
        self.assertNotIn("Strata needs 7.5 or newer", text)
        self.assertNotIn("older than the RTX 20 series", text)


if __name__ == "__main__":
    unittest.main(verbosity=1)
