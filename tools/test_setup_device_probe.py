"""tools/test_setup_device_probe.py - the run-time check of src/core/device.cu that a CUDA binary has the RIGHT code for its card
(include/strata/core/device.hpp: older_code_problem), on the host: no GPU needed.

    python3 tools/test_setup_device_probe.py

The probe (cudaFuncGetAttributes) says an engine has SOME code for a card.  An engine built for sm_70 alone runs its sm_70 SASS on
a cc 7.5 card and JIT-compiles its compute_70 PTX on an 8.x one, so the probe passes - but every kernel has the body chosen for
__CUDA_ARCH__ 700, where the Turing-and-newer paths are traps.  older_code_problem() refuses that, from the probe's ptxVersion.
A g++ compiles the pure function (the header needs no CUDA); an nvcc, when there is a CUDA 12 one, compiles device.cu itself.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DRIVER = r"""
#include "strata/core/device.hpp"
#include <iostream>
#include <string>
// stdin: "major minor ptxVersion binaryVersion built_for" per line (built_for "-" = unknown); stdout: the problem, or OK
int main() {
    int major, minor, ptx, bin;
    std::string built;
    while (std::cin >> major >> minor >> ptx >> bin >> built) {
        const std::string why = strata::core::older_code_problem("GPU 0 (card)", major, minor, ptx, bin,
                                                                 built == "-" ? nullptr : built.c_str());
        std::cout << (why.empty() ? std::string("OK") : why) << "\n";
    }
}
"""


class OlderCodeProblem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
        if cxx is None:
            raise unittest.SkipTest("no C++ compiler")
        cls.tmp = Path(tempfile.mkdtemp())
        src = cls.tmp / "probe.cpp"
        src.write_text(DRIVER)
        cls.exe = cls.tmp / "probe"
        r = subprocess.run([cxx, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(REPO / "include"), str(src), "-o",
                            str(cls.exe)], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("the probe test does not compile:\n" + r.stderr)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, True)

    def run_cases(self, *cases):
        text = "".join(f"{c[0]} {c[1]} {c[2]} {c[3]} {c[4]}\n" for c in cases)
        r = subprocess.run([str(self.exe)], input=text, capture_output=True, text=True, check=True)
        return r.stdout.splitlines()

    def one(self, major, minor, ptx, binary, built="-"):
        return self.run_cases((major, minor, ptx, binary, built))[0]

    def test_code_that_matches_the_card_is_left_alone(self):
        self.assertEqual(self.one(7, 0, 70, 70, "70"), "OK")              # a V100 with its sm_70 engine
        self.assertEqual(self.one(7, 0, 70, 70, "70,75,80,86,89"), "OK")
        self.assertEqual(self.one(7, 5, 75, 75, "70,75"), "OK")           # an RTX 2080 with an engine that has sm_75
        self.assertEqual(self.one(8, 6, 86, 86, "75,86"), "OK")
        self.assertEqual(self.one(8, 9, 89, 89, "70,75,80,86,89"), "OK")
        self.assertEqual(self.one(8, 6, 75, 86, "70,75"), "OK")           # JIT-compiled from compute_75 PTX: what upstream's
                                                                          # PTX fallback does, and the kernels have sm_75 bodies
        self.assertEqual(self.one(12, 0, 120, 120, "75,80,86,89,120"), "OK")
        self.assertEqual(self.one(12, 0, 89, 120, "75,80,86,89"), "OK")   # an RTX 50 on the sm_89 PTX of an RTX 20-40 engine

    def test_unknown_attributes_say_nothing(self):
        self.assertEqual(self.one(8, 6, 0, 0, "70"), "OK")                # (cubins from before CUDA 3 report 0)
        self.assertEqual(self.one(7, 5, 0, 70, "70"), "OK")

    def test_an_engine_for_sm70_alone_on_an_rtx20(self):
        why = self.one(7, 5, 70, 70, "70")                                # sm_70 SASS runs natively on cc 7.5
        self.assertIn("GPU 0 (card) is compute capability 7.5 (sm_75)", why)
        self.assertIn("compiled for sm_70 only", why)
        self.assertIn("binary sm_70, PTX sm_70", why)
        self.assertIn("(CMAKE_CUDA_ARCHITECTURES=70; ", why)
        self.assertIn('-DCMAKE_CUDA_ARCHITECTURES="70;75"', why)         # the card's own arch added to what it had
        self.assertIn("CUDA_VISIBLE_DEVICES", why)

    def test_an_engine_for_sm70_alone_on_an_rtx30_and_40(self):
        why = self.one(8, 6, 70, 86, "70")                                # the compute_70 PTX JIT-compiled for the card
        self.assertIn("compute capability 8.6 (sm_86)", why)
        self.assertIn("binary sm_86, PTX sm_70", why)
        self.assertIn('"70;86"', why)
        self.assertIn('"70;89"', self.one(8, 9, 70, 89, "70"))

    def test_the_list_it_was_built_for_is_merged_not_replaced(self):
        why = self.one(7, 5, 70, 70, "70,80")                             # no sm_75 in it: the sm_70 SASS runs
        self.assertIn('"70;75;80"', why)
        self.assertIn('"70;75;80"', self.one(7, 5, 70, 70, "80,70-real,70"))       # in any order, suffixes ignored, no duplicates

    def test_an_unknown_list_still_names_the_card(self):
        why = self.one(8, 6, 70, 86, "-")
        self.assertNotIn("(CMAKE_CUDA_ARCHITECTURES=", why)
        self.assertIn('"86"', why)

    def test_an_rtx50_cannot_share_the_v100_engine(self):
        why = self.one(12, 0, 70, 120, "70")                              # CUDA 13 for it, CUDA 12 for the V100: two engines
        self.assertIn("compute capability 12.0 (sm_120)", why)
        self.assertIn("CUDA 13", why)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=120", why)
        self.assertNotIn("70;120", why)
        self.assertIn("CUDA 13", self.one(12, 0, 70, 120, "-"))           # (whatever the list: an arch below 75 means CUDA 12)

    def test_every_case_in_one_run(self):
        out = self.run_cases((7, 0, 70, 70, "70"), (7, 5, 70, 70, "70"), (7, 5, 75, 75, "75"))
        self.assertEqual([o == "OK" for o in out], [True, False, True])


class DeviceCu(unittest.TestCase):
    def test_the_probe_keeps_what_the_driver_picked_and_asks_the_function(self):
        text = (REPO / "src" / "core" / "device.cu").read_text()
        self.assertIn("d.binary_version = attr.binaryVersion;", text)
        self.assertIn("d.ptx_version = attr.ptxVersion;", text)
        self.assertIn("older_code_problem(", text)
        self.assertIn("STRATA_CUDA_ARCHS", text)

    def test_it_compiles_for_sm70_when_there_is_a_cuda12_toolkit(self):
        nvcc = next((c for c in (shutil.which("nvcc"), "/usr/local/cuda-12.8/bin/nvcc", "/usr/local/cuda-12.9/bin/nvcc")
                     if c and Path(c).exists()), None)
        if nvcc is None:
            self.skipTest("no nvcc")
        version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
        if "release 12." not in version:
            self.skipTest("a CUDA 12 toolkit is needed for sm_70 (CUDA 13 dropped it)")
        with tempfile.TemporaryDirectory() as tmp:
            for archs, arch_flag in (("70", "--generate-code=arch=compute_70,code=[compute_70,sm_70]"),
                                     ("75,89", "--generate-code=arch=compute_75,code=[compute_75,sm_75]")):
                r = subprocess.run([nvcc, "-Wno-deprecated-gpu-targets", "-Xcompiler=-Wall,-Wextra,-Werror", f'-DSTRATA_CUDA_ARCHS="{archs}"',
                                    "-I", str(REPO / "include"), "-std=c++20", arch_flag, "-x", "cu", "-c",
                                    str(REPO / "src" / "core" / "device.cu"), "-o", os.path.join(tmp, "device.o")],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=1)
