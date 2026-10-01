"""Tests for ggml_codecs.py: MXFP4 / Q8_0 / BF16 / F16 / F32 encoders and decoders.

The decisive tests compare with GGML's own C code (ggml-quants.c compiled by ggml_c_oracle.py) bit for bit; they skip
when no ggml source tree or C compiler is available (set STRATA_GGML_SRC to the directory holding ggml-quants.c).
"""
import unittest

import numpy as np

import fixtures  # noqa: F401  (puts tools/ds41 on sys.path)
import ggml_c_oracle as O
import ggml_codecs as C

ORACLE = O.available()
SKIP = "ggml C source or compiler not available (set STRATA_GGML_SRC): " + (O.why_not() if not ORACLE else "")


def u32(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


class E8M0(unittest.TestCase):
    def test_values(self):
        h = C.E8M0_HALF
        self.assertEqual(float(h[128]), 1.0)             # 2^(128 - 128)
        self.assertEqual(float(h[127]), 0.5)
        self.assertEqual(float(h[129]), 2.0)
        self.assertEqual(float(h[2]), 2.0 ** -126)       # the smallest normal float32
        self.assertEqual(float(h[1]), 2.0 ** -127)       # denormal
        self.assertEqual(float(h[0]), 2.0 ** -128)       # denormal
        self.assertEqual(float(h[255]), 2.0 ** 127)
        for e in range(256):
            self.assertEqual(float(h[e]), 2.0 ** (e - 128), e)

    @unittest.skipUnless(ORACLE, SKIP)
    def test_matches_ggml(self):
        self.assertTrue(np.array_equal(u32(C.E8M0_HALF), u32(O.e8m0_half_table())))


class Mxfp4(unittest.TestCase):
    def test_handmade_block(self):
        # scale e = 128 -> d = 1.0; codes 0..15 in the low nibbles, 15..0 in the high nibbles
        blk = bytearray([128]) + bytearray((j | ((15 - j) << 4)) for j in range(16))
        out = C.dequant_mxfp4(bytes(blk))
        kv = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
        self.assertEqual(out[:16].tolist(), [float(kv[j]) for j in range(16)])           # low nibble -> j
        self.assertEqual(out[16:].tolist(), [float(kv[15 - j]) for j in range(16)])      # high nibble -> j + 16

    def test_scale_range(self):
        for e, want in ((0, 2.0 ** -128), (1, 2.0 ** -127), (2, 2.0 ** -126), (127, 0.5), (254, 2.0 ** 126)):
            blk = bytes([e, 0x02] + [0] * 15)            # value 0 = code 2 (kv 2), others 0
            self.assertEqual(float(C.dequant_mxfp4(blk)[0]), 2 * want, e)

    def test_encode_error_bound(self):
        rng = np.random.default_rng(5)
        for scale in (1e-6, 1.0, 1e4):
            x = (rng.standard_normal(32 * 500) * scale).astype(np.float32)
            raw = C.quant_mxfp4(x)
            y = C.dequant_mxfp4(raw)
            d = C.E8M0_HALF[np.frombuffer(raw, np.uint8).reshape(-1, 17)[:, 0]]
            err = np.abs(x - y).reshape(-1, 32).max(axis=1)
            self.assertTrue(np.all(err <= 4 * d * (1 + 1e-6)), scale)   # worst gap between codes (8 -> 12) is 4 d

    def test_zero_block(self):
        raw = C.quant_mxfp4(np.zeros(64, dtype=np.float32))
        self.assertEqual(raw, bytes(34))
        self.assertTrue(np.all(C.dequant_mxfp4(raw) == 0))

    def test_fixed_point(self):
        """Requantising dequantised values reproduces the values (the e may move, the values must not)."""
        rng = np.random.default_rng(6)
        raw = rng.integers(0, 256, size=17 * 3000, dtype=np.uint8)
        raw.reshape(-1, 17)[:, 0] = rng.integers(20, 240, size=3000)
        v = C.dequant_mxfp4(raw)
        v2 = C.dequant_mxfp4(C.quant_mxfp4(v))
        self.assertTrue(np.array_equal(u32(v), u32(v2)))

    def test_bad_sizes(self):
        with self.assertRaises(ValueError):
            C.dequant_mxfp4(bytes(18))
        with self.assertRaises(ValueError):
            C.quant_mxfp4(np.zeros(33, dtype=np.float32))

    @unittest.skipUnless(ORACLE, SKIP)
    def test_encode_matches_ggml(self):
        rng = np.random.default_rng(7)
        for scale in (1e-30, 1e-3, 1.0, 50.0, 1e30):
            x = (rng.standard_normal(32 * 3000) * scale).astype(np.float32)
            x[:96] = 0                                              # zero blocks
            x[96:128] = np.float32(1.5) * np.arange(32)             # exact-tie-ish values
            self.assertEqual(C.quant_mxfp4(x), O.quantize_mxfp4(x), scale)

    @unittest.skipUnless(ORACLE, SKIP)
    def test_encode_matches_ggml_around_powers_of_two(self):
        """floorf(log2f(amax)) rounds up for the float just below a power of two; the exponent must follow suit."""
        vals = []
        for k in range(-126, 127):
            p = np.float32(2.0) ** k
            lo = hi = p
            vals.append(p)
            for _ in range(3):
                lo, hi = np.nextafter(lo, np.float32(0)), np.nextafter(hi, np.float32(1e38))
                vals += [lo, hi]
        x = np.zeros((len(vals), 32), dtype=np.float32)
        x[:, 0] = vals
        x[:, 1] = np.float32(0.3) * np.array(vals, dtype=np.float32)
        self.assertEqual(C.quant_mxfp4(x), O.quantize_mxfp4(x))

    @unittest.skipUnless(ORACLE, SKIP)
    def test_decode_matches_ggml_every_scale(self):
        rng = np.random.default_rng(8)
        raw = rng.integers(0, 256, size=17 * 256 * 8, dtype=np.uint8)
        raw.reshape(-1, 17)[:, 0] = np.tile(np.arange(256, dtype=np.uint8), 8)    # e = 0, 1 ... 255, every code mix
        self.assertTrue(np.array_equal(u32(C.dequant_mxfp4(raw)), u32(O.dequantize_mxfp4(raw.tobytes()))))


class OfficialFp4(unittest.TestCase):
    """The official safetensors FP4 packing differs from GGML's block layout; the converter must reproduce values."""
    TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                     dtype=np.float32)       # convert.py FP4_TABLE

    @classmethod
    def official_dequant(cls, packed, scale):
        """convert.py:32-34 -- element 2k = low nibble, 2k+1 = high nibble; scale 2^(e-127) per 32 elements."""
        low, high = packed & 0x0F, packed >> 4
        vals = np.stack([cls.TABLE[low], cls.TABLE[high]], axis=-1).reshape(packed.shape[0], -1)
        s = np.ldexp(np.float32(1.0), scale.astype(np.int32) - 127).astype(np.float32)
        return vals * np.repeat(s, 32, axis=1)

    def test_values_survive_the_repack_and_bytes_do_not(self):
        rng = np.random.default_rng(13)
        rows, nb = 6, 5
        packed = rng.integers(0, 256, size=(rows, nb * 16), dtype=np.uint8)
        scale = rng.integers(100, 140, size=(rows, nb), dtype=np.uint8)
        ggml = C.mxfp4_from_official(packed, scale)
        self.assertEqual(len(ggml), rows * nb * 17)
        self.assertTrue(np.array_equal(C.dequant_mxfp4(ggml).reshape(rows, -1), self.official_dequant(packed, scale)))
        # the nibble bytes themselves differ (so "bit-identical" is wrong), the scale bytes are the same
        blocks = np.frombuffer(ggml, np.uint8).reshape(rows, nb, 17)
        self.assertFalse(np.array_equal(blocks[..., 1:].reshape(rows, -1), packed))
        self.assertTrue(np.array_equal(blocks[..., 0], scale))
        p2, s2 = C.mxfp4_to_official(ggml, nb)
        self.assertTrue(np.array_equal(p2, packed) and np.array_equal(s2, scale))


class Q80(unittest.TestCase):
    def test_roundtrip_bound(self):
        rng = np.random.default_rng(9)
        x = (rng.standard_normal(32 * 400) * 3).astype(np.float32)
        raw = C.quant_q8_0(x)
        y = C.dequant_q8_0(raw)
        d = np.frombuffer(raw, np.uint8).reshape(-1, 34)[:, :2].copy().view("<f2").astype(np.float32)
        err = np.abs(x - y).reshape(-1, 32).max(axis=1)
        self.assertTrue(np.all(err <= 0.6 * d[:, 0]))
        self.assertEqual(len(raw), 400 * 34)

    def test_round_half_away(self):
        # amax = 127 -> d = 1 -> id = 1: the value 2.5 must become 3 and -2.5 -> -3 (roundf), not 2 (rint)
        x = np.zeros(32, dtype=np.float32)
        x[0], x[1], x[2], x[3] = 127.0, 2.5, -2.5, 0.49999997
        y = C.dequant_q8_0(C.quant_q8_0(x))
        self.assertEqual(y[:4].tolist(), [127.0, 3.0, -3.0, 0.0])

    def test_zero_block(self):
        self.assertEqual(C.quant_q8_0(np.zeros(32, dtype=np.float32)), bytes(34))

    @unittest.skipUnless(ORACLE, SKIP)
    def test_matches_ggml(self):
        rng = np.random.default_rng(10)
        for scale in (1e-3, 1.0, 50.0, 1e4):
            x = (rng.standard_normal(32 * 3000) * scale).astype(np.float32)
            x[:64] = 0
            x[64:96] = np.float32(0.5) * np.arange(32)
            raw = C.quant_q8_0(x)
            self.assertEqual(raw, O.quantize_q8_0(x), scale)
            self.assertTrue(np.array_equal(u32(C.dequant_q8_0(raw)), u32(O.dequantize_q8_0(raw))))


class Bf16(unittest.TestCase):
    def test_round_to_nearest_even(self):
        one = np.float32(1.0)
        half_ulp = np.float32(2.0 ** -8)            # bf16 has 7 mantissa bits: 1 + 2^-8 is the tie between 1 and 1+2^-7
        self.assertEqual(C.bf16_bits_to_f32(C.f32_to_bf16_bits(np.array([one + half_ulp])))[0], 1.0)           # to even
        self.assertEqual(C.bf16_bits_to_f32(C.f32_to_bf16_bits(np.array([one + 3 * half_ulp])))[0],
                         np.float32(1 + 2.0 ** -6))                                                          # to even
        self.assertTrue(np.isnan(C.bf16_bits_to_f32(C.f32_to_bf16_bits(np.array([np.nan], dtype=np.float32)))[0]))

    def test_exact_on_bf16_values(self):
        bits = np.arange(0, 65536, 7, dtype=np.uint16)
        f = C.bf16_bits_to_f32(bits)
        ok = ~np.isnan(f)
        self.assertTrue(np.array_equal(C.f32_to_bf16_bits(f)[ok], bits[ok]))

    @unittest.skipUnless(ORACLE, SKIP)
    def test_matches_ggml(self):
        rng = np.random.default_rng(11)
        x = (rng.standard_normal(100000) * 3).astype(np.float32)
        x[:6] = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-42]
        self.assertTrue(np.array_equal(C.f32_to_bf16_bits(x), O.fp32_to_bf16(x)))
        bits = rng.integers(0, 65536, size=100000).astype(np.uint16)
        self.assertTrue(np.array_equal(u32(C.bf16_bits_to_f32(bits)), u32(O.bf16_to_fp32(bits))))


class Generic(unittest.TestCase):
    def test_encode_decode_all_types(self):
        rng = np.random.default_rng(12)
        x = (rng.standard_normal(32 * 8) * 0.3).astype(np.float32)
        for t, tol in (("F32", 0.0), ("F16", 1e-3), ("BF16", 1e-2), ("Q8_0", 5e-2), ("MXFP4", 0.5)):
            raw = C.encode(t, x)
            self.assertEqual(len(raw), C.tensor_nbytes(t, [x.size]), t)
            y = C.decode(t, raw)
            self.assertLessEqual(float(np.abs(x - y).max()), tol * float(np.abs(x).max()) + 1e-12, t)

    def test_geometry(self):
        self.assertEqual(C.tensor_nbytes("MXFP4", [5120, 2304, 384]), 2_406_481_920)
        self.assertEqual(C.tensor_nbytes("MXFP4", [256, 384_006_168]), 384_006_168 * 136)
        self.assertEqual(C.tensor_nbytes("Q8_0", [5120, 1280]), 5120 // 32 * 34 * 1280)
        with self.assertRaises(ValueError):
            C.tensor_nbytes("MXFP4", [100, 3])

    def test_types_agree_with_reader(self):
        import gguf_reader as R
        for name, tid in C.TYPE_ID.items():
            self.assertEqual(R.GGML_TYPES[tid], name)
            self.assertEqual(R.BLOCK_GEOMETRY[name], C.BLOCK[name])


if __name__ == "__main__":
    unittest.main()
