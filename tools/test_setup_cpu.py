"""tools/test_setup_cpu.py - setup.py's CPU detection: which CPUs count as "AVX-512" for the choice of the canonical
Q2_0 pack (the fastest model, which the engine's AVX-512 expert kernels run).  No network, nothing installed.

    python3 tools/test_setup_cpu.py

The rule is the engine's own (CpuFeatures::usable() in include/strata/kernels/cpu/expert.hpp): AVX-512 F, BW, VL, DQ and
VNNI.  AVX-512 VBMI is not needed - Intel Cascade Lake has VNNI but no VBMI and runs the kernels' no-VBMI build - and the
last test reads the engine's header, so the two cannot drift apart without this failing.
"""
from __future__ import annotations

import builtins
import re
import sys
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


class SameRuleAsTheEngine(unittest.TestCase):
    def test_usable_in_expert_hpp(self):
        text = (REPO / "include/strata/kernels/cpu/expert.hpp").read_text()
        body = re.search(r"bool usable\(\) const \{ return ([^;]*);", text).group(1)
        names = set(re.findall(r"[a-z0-9_]+", body)) - {"os_avx512"}        # the OS-state check has no /proc/cpuinfo flag
        self.assertNotIn("avx512_vbmi", names, "the engine must not require VBMI (Cascade Lake runs its no-VBMI build)")
        self.assertEqual(names, set(setup.AVX512_FLAGS))


if __name__ == "__main__":
    unittest.main()
