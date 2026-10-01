"""tools/test_setup_cpu.py - setup.py's CPU detection: which CPUs count as "AVX-512" for the choice of the canonical
Q2_0 pack (the fastest model, which the engine's AVX-512 expert kernels run).  No network, nothing installed.

    python3 tools/test_setup_cpu.py

The rule is the engine's own (CpuFeatures::usable() in include/strata/kernels/cpu/expert.hpp): AVX-512 F, BW, VL, DQ and
VNNI.  AVX-512 VBMI is not needed - Intel Cascade Lake has VNNI but no VBMI and runs the kernels' no-VBMI build - and the
last test reads the engine's header, so the two cannot drift apart without this failing.
"""
from __future__ import annotations

import builtins
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import setup  # noqa: E402

BASE = "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx " \
       "lm constant_tsc rep_good nopl xtopology cpuid pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt " \
       "aes xsave avx f16c rdrand hypervisor lahf_lm abm bmi1 avx2 bmi2"
SKYLAKE_X = BASE + " avx512f avx512dq avx512cd avx512bw avx512vl"
CASCADE_LAKE = SKYLAKE_X + " avx512_vnni"                      # Xeon Gold 6226: VNNI, no VBMI
ICE_LAKE = CASCADE_LAKE + " avx512vbmi avx512_vbmi2 avx512_bitalg avx512_vpopcntdq"
ZEN4 = BASE + " avx512f avx512dq avx512cd avx512bw avx512vl avx512_vnni avx512vbmi avx512_bf16"
ZEN3 = BASE
NO_DQ = BASE + " avx512f avx512cd avx512bw avx512vl avx512_vnni"

CPUINFO = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: {name}\nflags\t\t: {flags}\nbogomips\t: 5400.00\n"


def fake_open(text):
    real = builtins.open

    def opener(path, *a, **kw):
        if str(path) == "/proc/cpuinfo":
            import io
            return io.StringIO(text)
        return real(path, *a, **kw)
    return opener


class AvxFlags(unittest.TestCase):
    def test_cascade_lake_counts(self):
        self.assertTrue(setup.avx512_from_flags(CASCADE_LAKE.split()))

    def test_ice_lake_and_zen4_count(self):
        self.assertTrue(setup.avx512_from_flags(ICE_LAKE.split()))
        self.assertTrue(setup.avx512_from_flags(ZEN4.split()))

    def test_without_vnni_or_dq_or_avx512_it_does_not(self):
        self.assertFalse(setup.avx512_from_flags(SKYLAKE_X.split()))     # AVX-512 but no VNNI: the engine uses AVX2
        self.assertFalse(setup.avx512_from_flags(NO_DQ.split()))
        self.assertFalse(setup.avx512_from_flags(ZEN3.split()))
        self.assertFalse(setup.avx512_from_flags([]))


class Cpuid7(unittest.TestCase):
    F, DQ, CD, BW, VL = 1 << 16, 1 << 17, 1 << 28, 1 << 30, 1 << 31
    VBMI, VNNI = 1 << 1, 1 << 11

    def test_cascade_lake(self):          # CPUID.7.0: EBX has F DQ CD BW VL; ECX has VNNI, not VBMI
        self.assertTrue(setup.avx512_from_cpuid7(self.F | self.DQ | self.CD | self.BW | self.VL | (1 << 5), self.VNNI))

    def test_ice_lake(self):
        self.assertTrue(setup.avx512_from_cpuid7(self.F | self.DQ | self.CD | self.BW | self.VL, self.VNNI | self.VBMI))

    def test_missing_pieces(self):
        self.assertFalse(setup.avx512_from_cpuid7(self.F | self.DQ | self.BW | self.VL, self.VBMI))   # VBMI alone is no VNNI
        self.assertFalse(setup.avx512_from_cpuid7(self.F | self.BW | self.VL, self.VNNI))            # no DQ
        self.assertFalse(setup.avx512_from_cpuid7(0, 0))


class CpuInfo(unittest.TestCase):
    def info(self, flags, name="Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz"):
        with mock.patch.object(setup, "WIN", False), \
                mock.patch("builtins.open", fake_open(CPUINFO.format(name=name, flags=flags))):
            return setup.cpu_info()

    def test_the_target_box(self):        # Dell Precision 7920, 2x Xeon Gold 6226 (Cascade Lake)
        name, avx2, avx512 = self.info(CASCADE_LAKE)
        self.assertEqual(name, "Intel(R) Xeon(R) Gold 6226 CPU @ 2.70GHz")
        self.assertTrue(avx2)
        self.assertTrue(avx512)

    def test_skylake_x_and_zen3_stay_on_avx2(self):
        self.assertEqual(self.info(SKYLAKE_X)[1:], (True, False))
        self.assertEqual(self.info(ZEN3, "AMD Ryzen 9 5900X")[1:], (True, False))

    def test_ice_lake_still_counts(self):
        self.assertEqual(self.info(ICE_LAKE)[1:], (True, True))


class Vbmi(unittest.TestCase):
    """The ready-made upstream engine's canonical Q2_0 kernel needs AVX-512 VBMI on top (its cpu_require_expert_support exits
    "missing AVX512-VBMI" on a Cascade Lake); an engine compiled from this source does not."""

    def vbmi(self, flags):
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", fake_open(CPUINFO.format(name="x", flags=flags))):
            return setup.cpu_has_vbmi()

    def test_flag(self):
        self.assertFalse(self.vbmi(CASCADE_LAKE))
        self.assertFalse(self.vbmi(SKYLAKE_X))
        self.assertFalse(self.vbmi(ZEN3))
        self.assertTrue(self.vbmi(ICE_LAKE))
        self.assertTrue(self.vbmi(ZEN4))

    def test_unreadable_cpuinfo(self):
        with mock.patch.object(setup, "WIN", False), mock.patch("builtins.open", side_effect=OSError):
            self.assertFalse(setup.cpu_has_vbmi())

    def test_windows_cpuid(self):
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "_cpuid7_regs", return_value=(0, 1 << 1 | 1 << 11)):
            self.assertTrue(setup.cpu_has_vbmi())
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "_cpuid7_regs", return_value=(0, 1 << 11)):
            self.assertFalse(setup.cpu_has_vbmi())
        with mock.patch.object(setup, "WIN", True), mock.patch.object(setup, "_cpuid7_regs", return_value=None):
            self.assertFalse(setup.cpu_has_vbmi())


class PackChoice(unittest.TestCase):
    """Which engine gets the canonical Q2_0 pack (setup.py picks it by this: the AVX-512 kernel)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        p = mock.patch.object(setup, "ROOT", self.tmp)
        p.start()
        self.addCleanup(p.stop)
        (self.tmp / "CMakeLists.txt").write_text("project(strata VERSION 0.1.31 LANGUAGES CXX)")
        self.src = setup.source_hash(setup.ENGINE_SOURCES)

    PREBUILT = {"version": "0.1.31", "archs": [75, 80, 86, 89, 120], "ptx": True, "cuda": "13.0"}      # upstream's release

    def test_built_from_this_source(self):
        self.assertTrue(setup.built_from_this_source({"source": "local", "src": self.src}))
        self.assertTrue(setup.built_from_this_source({"source": "local-hip", "backend": "hip", "src": self.src}))
        self.assertFalse(setup.built_from_this_source(self.PREBUILT))
        self.assertFalse(setup.built_from_this_source({"source": "local", "src": "an-older-checkout"}))   # a git pull since
        self.assertFalse(setup.built_from_this_source({"source": "local"}))

    def test_cascade_lake_with_the_ready_made_engine_gets_the_native_pack(self):
        # Xeon Gold 62xx / W-22xx / i9-10980XE with an RTX 20-50 card and the downloaded engine, Q2_0 (the audit's finding)
        self.assertTrue(setup.avx512_from_flags(CASCADE_LAKE.split()))
        self.assertFalse(setup.pack_avx512(True, setup.vbmi_from_flags(CASCADE_LAKE.split()),
                                           setup.built_from_this_source(self.PREBUILT)))

    def test_cascade_lake_with_a_local_engine_gets_the_canonical_pack(self):
        self.assertTrue(setup.pack_avx512(True, False, setup.built_from_this_source({"source": "local", "src": self.src})))

    def test_ice_lake_and_zen4_keep_the_canonical_pack_with_either_engine(self):
        self.assertTrue(setup.pack_avx512(True, True, False))
        self.assertTrue(setup.pack_avx512(True, True, True))

    def test_no_avx512_never_gets_it(self):
        for from_source in (False, True):
            self.assertFalse(setup.pack_avx512(False, False, from_source))
            self.assertFalse(setup.pack_avx512(False, True, from_source))

    def test_the_engine_and_the_hardware_in_one_install(self):
        # the installed engine is read from engine/BUILD.json
        (self.tmp / "engine").mkdir()
        self.assertFalse(setup.installed_engine_local())
        (self.tmp / "engine" / "BUILD.json").write_text(json.dumps(self.PREBUILT))
        self.assertFalse(setup.installed_engine_local())
        (self.tmp / "engine" / "BUILD.json").write_text(json.dumps({"source": "local", "src": self.src}))
        self.assertTrue(setup.installed_engine_local())


class PackForm(unittest.TestCase):
    """A Q2_0 pack folder holds one form of the pack, and the engine tells them apart by native_experts.txt alone: a switch
    of form must not leave the old form's marker (or its experts.bin) behind."""

    def setUp(self):
        self.pack = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.pack, True)
        (self.pack / "tokenizer").mkdir()
        (self.pack / "tokenizer" / "vocab.json").write_text("{}")

    def make(self, *names):
        for n in names:
            (self.pack / n).write_text(n)

    def left(self):
        return sorted(str(p.relative_to(self.pack)) for p in self.pack.rglob("*") if p.is_file())

    NATIVE = ("native_experts.txt", "index.txt", "dense.bin", "conversions.json")
    CANONICAL = ("manifest.json", "embd.bin", "dense.bin", "experts.bin", "index.txt")

    def test_native_pack_to_canonical(self):
        # an upstream install on a Cascade Lake had the native pack; a re-run with an engine from this source wants canonical
        self.make(*self.NATIVE, "experts.bin", "experts.bin.src.json")
        self.assertTrue(setup.drop_other_pack_form(self.pack, True))
        self.assertEqual(self.left(), ["tokenizer/vocab.json"])        # no native_experts.txt: the engine reads the new pack right

    def test_canonical_pack_to_native(self):
        self.make(*self.CANONICAL)
        self.assertTrue(setup.drop_other_pack_form(self.pack, False))
        self.assertEqual(self.left(), ["tokenizer/vocab.json"])        # (experts.bin of the canonical pack would be read as native)

    def test_the_wanted_form_is_left_alone(self):
        self.make(*self.CANONICAL)
        self.assertFalse(setup.drop_other_pack_form(self.pack, True))
        self.assertEqual(len(self.left()), len(self.CANONICAL) + 1)
        for p in self.pack.glob("*.*"):
            p.unlink()
        self.make(*self.NATIVE, "experts.bin", "experts.bin.src.json")
        self.assertFalse(setup.drop_other_pack_form(self.pack, False))
        self.assertEqual(len(self.left()), len(self.NATIVE) + 2 + 1)

    def test_a_half_converted_pack_of_both_forms_is_cleared_either_way(self):
        for want in (True, False):
            self.make(*self.NATIVE, "manifest.json", "embd.bin", "experts.bin")
            self.assertTrue(setup.drop_other_pack_form(self.pack, want))
            self.assertEqual(self.left(), ["tokenizer/vocab.json"])

    def test_an_empty_folder_and_a_missing_one(self):
        self.assertFalse(setup.drop_other_pack_form(self.pack, True))
        self.assertFalse(setup.drop_other_pack_form(self.pack / "nope", False))

    def test_a_hard_linked_dense_bin_is_not_written_through(self):
        # tools/iq_pack.py --base hard-links the base pack's dense.bin: removing the name must not touch the other pack
        other = Path(tempfile.mkdtemp()) / "dense.bin"
        other.write_text("base pack")
        self.make("native_experts.txt", "index.txt")
        try:
            (self.pack / "dense.bin").hardlink_to(other)
        except OSError:
            self.skipTest("no hard links here")
        setup.drop_other_pack_form(self.pack, True)
        self.assertEqual(other.read_text(), "base pack")


class SameRuleAsTheEngine(unittest.TestCase):
    def test_usable_in_expert_hpp(self):
        text = (REPO / "include/strata/kernels/cpu/expert.hpp").read_text()
        body = re.search(r"bool usable\(\) const \{ return ([^;]*);", text).group(1)
        names = set(re.findall(r"[a-z0-9_]+", body)) - {"os_avx512"}        # the OS-state check has no /proc/cpuinfo flag
        self.assertNotIn("avx512_vbmi", names, "the engine must not require VBMI (Cascade Lake runs its no-VBMI build)")
        self.assertEqual(names, set(setup.AVX512_FLAGS))


if __name__ == "__main__":
    unittest.main()
