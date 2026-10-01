"""The activation quantiser of CONTRACTS.md ("Activations"), three implementations of one rule:

  * the oracle (ref/ds41/quant.py: quantize_int8_blocks, act_quant_int8), NumPy,
  * DS-C's C++ kernels (src/ds41/cpu: scalar, AVX2, AVX-512), compiled on the fly by cpu_pack_xcheck.py,
  * (the GPU kernel and its emulator are compared with the C++ library inside ds41_cuda_emu_test --quant.)

They must produce the same BYTES (the int8 values and the fp32 scale as bits), on ordinary blocks and on the ones that used to differ: all-zero,
1e-37 everywhere (127 / amax overflows), the 2^-100 boundary, ties (round half to EVEN, not away from zero), an Inf or a NaN anywhere in the block
(d = NaN, q = 0), and random bit patterns.  Skipped when g++ or the DS-C sources are missing.
"""
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

import fixtures as T
import cpu_pack_xcheck as CP

sys.path.insert(0, str(T.REPO))
from ref.ds41 import quant as Q  # noqa: E402


def f32(bits):
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def edge_blocks():
    """(name, 32 floats, expected d bits, expected q head): the hand-written table the C++ tests (mxfp4_expert_test.cpp, ds41_parity_lib.hpp) use."""
    nan, inf = f32(0x7FC00000), np.float32(np.inf)
    a = f32(0x0D800000)

    def blk(head, fill=0.0):
        b = np.full(32, fill, np.float32)
        b[:len(head)] = head
        return b

    out = [
        ("all zero", blk([]), 0, []),
        ("all -0.0", blk([], -0.0), 0, []),
        ("1e-37 everywhere", blk([], 1e-37), 0, []),
        ("-1e-37 everywhere", blk([], -1e-37), 0, []),
        ("denormals", blk([1e-40, -3e-39, 1e-45]), 0, []),
        ("just below 2^-100", blk([f32(0x0D7FFFFF), -f32(0x0D7FFFFF)]), 0, []),
        ("exactly 2^-100", blk([a, -a, a * np.float32(0.5)]), int(np.float32(a / np.float32(127.0)).view(np.uint32)), [127, -127, 64]),
        ("ties to even", blk([254, 5, 1, -5]), int(np.float32(2.0).view(np.uint32)), [127, 2, 0, -2]),
        ("ties to even, odd side", blk([254, 3, -3, 7, -1, 9]), int(np.float32(2.0).view(np.uint32)), [127, 2, -2, 4, 0, 4]),
        ("NaN first", blk([nan, 1, 2], 0.5), 0x7FC00000, []),
        ("Inf first", blk([inf, 1, 2], 0.5), 0x7FC00000, []),
        ("negative NaN", blk([f32(0xFFC00000)], 1.0), 0x7FC00000, []),
        ("signalling NaN", blk([f32(0x7FA00000)], 1.0), 0x7FC00000, []),
        ("FLT_MAX", blk([np.finfo(np.float32).max, -np.finfo(np.float32).max, np.finfo(np.float32).max * np.float32(0.25)]),
         int((np.finfo(np.float32).max / np.float32(127.0)).view(np.uint32)), [127, -127, 32]),
    ]
    for pos in (1, 7, 8, 15, 16, 24, 31):
        base = (np.arange(32, dtype=np.float32) - 15) * np.float32(3.0)
        w_nan, w_inf = base.copy(), base.copy()
        w_nan[pos] = nan
        w_inf[pos] = inf if pos & 1 else -inf
        out += [(f"NaN at {pos}", w_nan, 0x7FC00000, []), (f"Inf at {pos}", w_inf, 0x7FC00000, [])]
    return out


@unittest.skipUnless(CP.available(), "DS-C sources / C++ compiler not available: " + (CP.why_not() or ""))
class QuantiserRule(unittest.TestCase):
    def isas(self):
        return [n for n in CP.ISAS if CP.isa_supported(n)]

    def check(self, x, label):
        """x [n] float32, n a multiple of 128: oracle == every C++ ISA, bit for bit."""
        with np.errstate(all="ignore"):
            q, d = Q.quantize_int8_blocks(x, 32)
        q = q.reshape(-1)
        d = d.reshape(-1)
        for isa in self.isas():
            cq, cd = CP.quantize_act(x, isa)
            self.assertTrue(np.array_equal(q, cq), f"{label}: int8 values differ from {isa}")
            self.assertEqual(d.view(np.uint32).tolist(), cd.view(np.uint32).tolist(), f"{label}: scales differ from {isa}")

    def test_scalar_is_always_there(self):
        self.assertIn("scalar", self.isas())

    def test_edge_blocks_expected_bytes_in_every_implementation(self):
        edges = edge_blocks()
        pad = (-len(edges)) % 4
        names = [e[0] for e in edges] + ["pad"] * pad
        x = np.concatenate([e[1] for e in edges] + [np.zeros(32, np.float32)] * pad)
        with np.errstate(all="ignore"):
            oq, od = Q.quantize_int8_blocks(x, 32)
        results = {"oracle": (oq.reshape(-1), od.reshape(-1))}
        for isa in self.isas():
            results[isa] = CP.quantize_act(x, isa)
        for who, (q, d) in results.items():
            for i, (name, _blk, d_bits, qhead) in enumerate(edges):
                want = np.zeros(32, np.int8)
                want[:len(qhead)] = qhead
                self.assertEqual(int(d.view(np.uint32)[i]), d_bits, f"{who}: block '{name}' ({names[i]}): scale")
                self.assertTrue(np.array_equal(q[i * 32:(i + 1) * 32], want), f"{who}: block '{name}': int8 values {q[i * 32:(i + 1) * 32]}")

    def test_random_blocks(self):
        rng = np.random.default_rng(5)
        for rep in range(20):
            x = (rng.standard_normal(5120) * np.exp2(rng.integers(-3, 4, size=(160, 1)).repeat(32, 1).reshape(-1))).astype(np.float32)
            self.check(x, f"normal rep {rep}")

    def test_random_bit_patterns_and_magnitudes(self):
        rng = np.random.default_rng(6)
        for rep in range(20):
            raw = rng.integers(0, 2 ** 32, size=1280, dtype=np.uint64).astype(np.uint32).view(np.float32)
            self.check(raw, f"bit patterns rep {rep}")
            mix = (rng.standard_normal(1280) * np.exp2(rng.integers(-140, 100, size=(40, 1)).repeat(32, 1).reshape(-1))).astype(np.float32)
            self.check(mix, f"magnitudes rep {rep}")
            ties = (rng.integers(-300, 300, size=1280) * 0.5).astype(np.float32)
            self.check(ties, f"ties rep {rep}")
            straddle = (rng.standard_normal(1280) * 2.0 ** -100 * (1 + 1e-3 * rng.integers(-5, 6, size=(40, 1)).repeat(32, 1).reshape(-1))).astype(np.float32)
            self.check(straddle, f"2^-100 rep {rep}")


if __name__ == "__main__":
    unittest.main()
