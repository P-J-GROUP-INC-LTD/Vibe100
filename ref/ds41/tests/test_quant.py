"""Number formats: every decoder against an independent formulation, and the fake-quantisers against kernel_stub
(pure-torch transcription of kernel.py) bit for bit."""
import struct

import numpy as np
import pytest
import torch

from ref.ds41 import quant as Q
from ref.ds41.tests import kernel_stub as K


# ------------------------------------------------------------------------------------------------ small formats
def test_e4m3_table_matches_torch():
    allb = torch.arange(256, dtype=torch.uint8)
    ref = allb.view(torch.float8_e4m3fn).float().numpy()
    mine = Q.fp8_e4m3_to_f32(allb.numpy())
    assert np.array_equal(np.isnan(ref), np.isnan(mine))
    assert np.array_equal(ref[~np.isnan(ref)], mine[~np.isnan(mine)])


def test_e8m0_matches_torch():
    allb = torch.arange(256, dtype=torch.uint8)
    ref = allb.view(torch.float8_e8m0fnu).float().numpy()
    mine = Q.e8m0_to_f32(allb.numpy())
    assert np.array_equal(np.isnan(ref), np.isnan(mine))
    ok = ~np.isnan(ref)
    assert np.array_equal(ref[ok], mine[ok])


def test_bf16_roundtrip_and_rounding():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(5000) * 10 ** rng.uniform(-6, 6, 5000)).astype(np.float32)
    ref = torch.from_numpy(x).bfloat16().float().numpy()
    assert np.array_equal(Q.f32_to_bf16_round(x), ref)
    assert np.array_equal(Q.bf16_to_f32(Q.f32_to_bf16_bits(x)), ref)


def test_round_e4m3_matches_torch_cast():
    rng = np.random.default_rng(1)
    v = np.concatenate([(rng.standard_normal(200000) * 10 ** rng.uniform(-4, 2.6, 200000)).astype(np.float32),
                        np.array([0, 448, -448, 2 ** -9, 2 ** -10, 1.5 * 2 ** -9, 0.5 * 2 ** -9], np.float32)])
    v = np.clip(v, -448, 448)
    ref = torch.from_numpy(v).to(torch.float8_e4m3fn).float().numpy()
    assert np.array_equal(Q.round_e4m3(v), ref)


def test_round_e2m1_independent_formulation():
    """Nearest grid point, ties to the even code, by brute force over the 8 magnitudes."""
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.uniform(-7, 7, 20000), np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -5.0, 6, 7.5])])
    exp = np.empty_like(v)
    for i, x in enumerate(v):
        a = abs(x)
        d = np.abs(grid - a)
        m = d.min()
        cands = np.nonzero(d == m)[0]
        code = cands[np.argmin(cands % 2)] if len(cands) > 1 else cands[0]       # tie -> even code
        exp[i] = np.copysign(grid[min(code, 7)], x)
    assert np.array_equal(Q.round_e2m1(v), exp)


# ------------------------------------------------------------------------------------------------ MXFP4 (GGML)
def _ggml_dequant_mxfp4_scalar(block: bytes) -> list:
    """Straight transcription of ggml-quants.c:dequantize_row_mxfp4 + ggml-impl.h:ggml_e8m0_to_fp32_half."""
    kvalues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
    e = block[0]
    bits = (0x00200000 << e) if e < 2 else ((e - 1) << 23)
    d = np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])
    y = [0.0] * 32
    with np.errstate(over="ignore"):
        for j in range(16):
            y[j] = float(np.float32(kvalues[block[1 + j] & 0x0F]) * d)          # float32 product, as in C
            y[j + 16] = float(np.float32(kvalues[block[1 + j] >> 4]) * d)
    return y


def test_mxfp4_dequant_vs_ggml_scalar_all_scales():
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 256, size=(256 * 2, 17), dtype=np.uint8)
    blocks[:256, 0] = np.arange(256, dtype=np.uint8)                     # every E8M0 byte, incl. 0, 1, 255
    got = Q.dequant_mxfp4(blocks)                                         # [N, 17] raw rows -> [N, 32]
    assert got.shape == (512, 32)
    for i in range(blocks.shape[0]):
        assert got[i].tolist() == _ggml_dequant_mxfp4_scalar(bytes(blocks[i])), i
    # the three special scales explicitly
    one = np.zeros((1, 17), np.uint8)
    one[0, 1] = 0x21                                                      # codes 1 (low) and 2 (high)
    for e, d in ((0, 2.0 ** -128), (1, 2.0 ** -127), (127, 0.5), (255, 2.0 ** 127)):
        one[0, 0] = e
        y = Q.dequant_mxfp4(one)[0]
        with np.errstate(over="ignore"):
            assert y[0] == np.float32(1 * d) and y[16] == np.float32(2 * d), e


def test_mxfp4_layout_row_major_blocks():
    """[rows, nb*17] raw bytes == [rows, nb, 17]: block b of a row covers elements 32b..32b+31."""
    rng = np.random.default_rng(4)
    raw = rng.integers(0, 256, size=(3, 5 * 17), dtype=np.uint8)
    a = Q.dequant_mxfp4(raw)
    b = Q.dequant_mxfp4(raw.reshape(3, 5, 17))
    assert a.shape == (3, 160) and np.array_equal(a, b)


def test_official_fp4_repack_to_ggml_is_lossless_and_matches_torch_decode():
    """official packed e2m1 + E8M0 -> GGML blocks -> dequant == the stub's (torch) decode of the official layout."""
    rng = np.random.default_rng(5)
    n, k = 12, 128
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    scales = rng.integers(100, 140, size=(n, k // 32), dtype=np.uint8)
    off = Q.dequant_fp4_official(packed, scales)
    blocks = Q.mxfp4_blocks_from_official(packed, scales)
    assert blocks.shape == (n, k // 32, 17)
    assert np.array_equal(Q.dequant_mxfp4(blocks), off)
    t = torch.from_numpy(packed)
    ref = (K._decode_fp4_bytes(t).unflatten(-1, (-1, 32)) * torch.from_numpy(scales).view(torch.float8_e8m0fnu).float()
           .unsqueeze(-1)).flatten(-2).numpy()
    assert np.array_equal(off, ref)


def test_ggml_mxfp4_quantizer_is_lossless_on_fp4_data():
    rng = np.random.default_rng(6)
    codes = rng.integers(0, 16, size=(20, 64), dtype=np.uint8)
    scales = rng.integers(110, 135, size=(20, 2), dtype=np.uint8)
    vals = Q.dequant_fp4_official(Q.pack_fp4_official(codes), scales)
    # negative zero codes (8) dequantise to -0.0 == 0.0: harmless
    again = Q.dequant_mxfp4(Q.quantize_mxfp4_ggml(vals))
    assert np.array_equal(again, vals)


# ------------------------------------------------------------------------------------------------ Q8_0 / FP8 blocks
def test_q8_0_dequant_vs_scalar_and_roundtrip():
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((6, 96)) * 3).astype(np.float32)
    blocks = Q.quantize_q8_0(x)
    assert blocks.shape == (6, 3, 34)
    got = Q.dequant_q8_0(blocks)
    for r in range(6):
        for b in range(3):
            raw = bytes(blocks[r, b])
            d = float(np.frombuffer(raw[:2], np.float16)[0])
            q = np.frombuffer(raw[2:], np.int8)
            assert np.allclose(got[r, b * 32:(b + 1) * 32], q.astype(np.float32) * np.float32(d), rtol=0, atol=0)
    assert np.abs(got - x).max() <= np.abs(x).max() / 127 * 0.51 + 1e-6      # quantisation step bound


def test_fp8_block_dequant_matches_torch():
    rng = np.random.default_rng(8)
    n, k = 96, 160
    w = torch.from_numpy((rng.standard_normal((n, k)) * 5).astype(np.float32)).clamp(-448, 448)
    q = w.to(torch.float8_e4m3fn)
    s = torch.from_numpy(rng.integers(120, 135, size=(n // 32, k // 32), dtype=np.uint8)).view(torch.float8_e8m0fnu)
    ref = (q.float() * s.float().repeat_interleave(32, 0).repeat_interleave(32, 1)).numpy()
    got = Q.dequant_fp8_block(q.view(torch.uint8).numpy(), s.view(torch.uint8).numpy(), 32)
    assert np.array_equal(got, ref)


# ------------------------------------------------------------------------------------------------ fake-quant
def _data(rng, shape, big=1e4):
    base = rng.standard_normal(shape)
    mag = 10.0 ** rng.uniform(-3, 2.2, size=shape[:-1] + (1,))
    x = (base * mag).astype(np.float32)
    x[0, :32] = 0.0                                      # an all-zero block (the 1e-4 / 6*2^-9 floors)
    x[1, :32] = 1e-9                                     # below the floors
    x[2, 5] = big                                        # an outlier that saturates the block's scale
    return x


def test_act_quant_fp8_equals_kernel_stub():
    rng = np.random.default_rng(10)
    x = _data(rng, (300, 64))
    t = torch.from_numpy(x.copy())
    K.act_quant(t, 32, "ue8m0", torch.float8_e8m0fnu, True)
    assert np.array_equal(t.numpy(), Q.act_quant_fp8(x, 32))
    # non-inplace form (the GEMM path): dequantise q * s
    q, s = K.act_quant(torch.from_numpy(x.copy()), 32, "ue8m0", torch.float8_e8m0fnu)
    deq = (q.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2).numpy()
    assert np.array_equal(deq, Q.act_quant_fp8(x, 32))


def test_fp4_quant_e8m0_equals_kernel_stub():
    rng = np.random.default_rng(11)
    x = _data(rng, (300, 128))
    t = torch.from_numpy(x.copy())
    K.fp4_act_quant(t, 32, True)
    assert np.array_equal(t.numpy(), Q.fp4_quant_e8m0(x, 32))


def test_fp4_quant_e4m3_equals_kernel_stub():
    rng = np.random.default_rng(12)
    x = _data(rng, (300, 64), big=2000.0)                # scale = e4m3(amax/6) must stay <= 448 (amax <= 2688)
    t = torch.from_numpy(x.copy())
    K.fp4_act_quant(t, 16, True, scale_dtype=torch.float8_e4m3fn)
    assert np.array_equal(t.numpy(), Q.fp4_quant_e4m3(x, 16))


def test_scale_rounding_bit_trick_independent():
    """kernel.py:fast_log2_ceil by float32 bit manipulation == the oracle's frexp formulation."""
    rng = np.random.default_rng(13)
    amax = np.concatenate([10.0 ** rng.uniform(-4, 4, 5000), 448.0 * 2.0 ** np.arange(-10, 10), [1e-4, 448.0]]).astype(np.float32)
    t = (amax * np.float32(1 / 448)).astype(np.float32)
    bits = t.view(np.uint32)
    exp_x = ((bits >> 23) & 0xFF).astype(np.int64)
    man = bits & ((1 << 23) - 1)
    ceil_log2 = exp_x - 127 + (man != 0)                             # kernel.py:22-27
    expect = np.ldexp(np.float32(1), ceil_log2.astype(np.int32))
    assert np.array_equal(Q._round_scale_pow2(amax, np.float32(1 / 448)), expect)


def _bits(f):
    import struct
    return struct.unpack("<I", struct.pack("<f", float(np.float32(f))))[0]


def _f32(bits):
    import struct
    return np.float32(struct.unpack("<f", struct.pack("<I", bits))[0])


def _scalar_rule(block):
    """An independent, scalar, element-by-element transcription of CONTRACTS.md "Activations" (the C++ kernels' rule):
    -> (q list, d float32).  Python's round() is round-half-to-even; the arithmetic is np.float32 scalars."""
    m = max(_bits(v) & 0x7FFFFFFF for v in block)
    if m >= 0x7F800000:
        return [0] * len(block), _f32(0x7FC00000)
    if m < 0x0D800000:
        return [0] * len(block), np.float32(0.0)
    amax = _f32(m)
    d = np.float32(amax / np.float32(127.0))
    idv = np.float32(np.float32(127.0) / amax)
    q = [max(-127, min(127, round(float(np.float32(np.float32(v) * idv))))) for v in block]
    return q, d


def test_act_quant_int8_vs_scalar_formulation():
    """CONTRACTS.md: int8 per 32 with an fp32 scale d = amax / 127, q = rint(x * (127 / amax)) - round half to EVEN, ggml's x86 SIMD
    quantize_row_q8_0, NOT quantize_row_q8_0_ref (1 / d, ties away from zero)."""
    rng = np.random.default_rng(14)
    x = (rng.standard_normal((40, 64)) * 10.0 ** rng.uniform(-3, 3, (40, 1))).astype(np.float32)
    x[0, :32] = 0.0
    got = Q.act_quant_int8(x, 32)
    qb, db = Q.quantize_int8_blocks(x, 32)
    for r in range(x.shape[0]):
        for b in range(2):
            blk = x[r, b * 32:(b + 1) * 32]
            q, d = _scalar_rule(list(blk))
            assert list(qb[r, b]) == q and db[r, b].tobytes() == d.tobytes(), (r, b)
            exp = np.array([np.float32(v) * d for v in q], np.float32)
            assert np.array_equal(got[r, b * 32:(b + 1) * 32], exp), (r, b)
            if d != 0:
                assert np.abs(got[r, b * 32:(b + 1) * 32] - blk).max() <= d * 0.5001 + 1e-30       # half a step
    # the extreme value of every block is hit exactly (127 * d)
    assert np.allclose(np.abs(got[1:]).reshape(39, 2, 32).max(-1), np.abs(x[1:]).reshape(39, 2, 32).max(-1), rtol=1e-6)


def _edge(head, fill=0.0):
    b = np.full(32, fill, np.float32)
    b[:len(head)] = head
    return b


def test_act_quant_int8_edge_blocks_hand_written():
    """The blocks on which the implementations used to disagree, with hand-written expected bytes (the same table as the C++ tests:
    src/ds41/cpu/mxfp4_expert_test.cpp, src/ds41/cuda/ds41_parity_lib.hpp)."""
    nan, inf = _f32(0x7FC00000), np.float32(np.inf)
    a = _f32(0x0D800000)                                    # 2^-100
    cases = [
        ("all zero", _edge([]), 0, []),
        ("all -0.0", _edge([], -0.0), 0, []),
        ("1e-37 everywhere", _edge([], 1e-37), 0, []),                         # 127 / 1e-37 overflows float32: the old rule gave q = -127 / -128
        ("denormals", _edge([1e-40, -3e-39, 1e-45]), 0, []),
        ("just below 2^-100", _edge([_f32(0x0D7FFFFF), -_f32(0x0D7FFFFF)]), 0, []),
        ("exactly 2^-100", _edge([a, -a, a * np.float32(0.5)]), _bits(a / np.float32(127.0)), [127, -127, 64]),   # 63.5 -> 64 (ties to even)
        ("ties to even", _edge([254, 5, 1, -5]), _bits(2.0), [127, 2, 0, -2]),                         # amax 254, id 0.5: 2.5 -> 2, 0.5 -> 0, -2.5 -> -2
        ("ties to even, odd side", _edge([254, 3, -3, 7, -1, 9]), _bits(2.0), [127, 2, -2, 4, 0, 4]),   # 1.5 -> 2, 3.5 -> 4, 0.5 -> 0, 4.5 -> 4
        ("NaN first", _edge([nan, 1, 2], 0.5), 0x7FC00000, []),
        ("Inf first", _edge([inf, 1, 2], 0.5), 0x7FC00000, []),
        ("-Inf last", np.concatenate([np.ones(31, np.float32), [-inf]]).astype(np.float32), 0x7FC00000, []),
        ("negative NaN", _edge([_f32(0xFFC00000)], 1.0), 0x7FC00000, []),
        ("signalling NaN", _edge([_f32(0x7FA00000)], 1.0), 0x7FC00000, []),
        ("FLT_MAX", _edge([np.finfo(np.float32).max, -np.finfo(np.float32).max, np.finfo(np.float32).max * np.float32(0.25)]),
         _bits(np.finfo(np.float32).max / np.float32(127.0)), [127, -127, 32]),
    ]
    for pos in (1, 7, 8, 15, 16, 24, 31):
        base = (np.arange(32, dtype=np.float32) - 15) * np.float32(3.0)
        withnan, withinf = base.copy(), base.copy()
        withnan[pos] = nan
        withinf[pos] = inf if pos & 1 else -inf
        cases += [(f"NaN at {pos}", withnan, 0x7FC00000, []), (f"Inf at {pos}", withinf, 0x7FC00000, [])]
    for what, blk, d_bits, qhead in cases:
        q, d = Q.quantize_int8_blocks(blk, 32)
        want = np.zeros(32, np.int8)
        want[:len(qhead)] = qhead
        assert q.shape == (1, 32) or q.shape == (32,) or q.size == 32
        assert np.array_equal(q.reshape(32), want), (what, q.reshape(32))
        assert int(np.float32(d).reshape(()).view(np.uint32)) == d_bits, (what, hex(int(np.float32(d).reshape(()).view(np.uint32))))
        deq = Q.act_quant_int8(blk, 32)
        if d_bits == 0x7FC00000:
            assert np.isnan(deq).all(), what                  # the NaN reaches the output (every element of the block)
        elif d_bits == 0:
            assert (deq == 0).all(), what


def test_act_quant_int8_random_bit_patterns_equal_the_scalar_rule():
    """Random float32 BIT PATTERNS (NaN, Inf, denormals, tiny, huge in the same blocks) and scaled mixtures: vectorised == scalar rule, bit for bit."""
    rng = np.random.default_rng(31)
    raw = rng.integers(0, 2 ** 32, size=(60, 32), dtype=np.uint64).astype(np.uint32).view(np.float32)
    mix = (rng.standard_normal((60, 32)) * np.exp2(rng.integers(-140, 100, size=(60, 1)))).astype(np.float32)
    ties = (rng.integers(-300, 300, size=(60, 32)) * 0.5).astype(np.float32)
    straddle = (rng.standard_normal((60, 32)) * 2.0 ** -100 * (1 + 1e-3 * rng.integers(-5, 6, size=(60, 1)))).astype(np.float32)
    for name, x in (("bit patterns", raw), ("magnitudes", mix), ("ties", ties), ("2^-100", straddle)):
        with np.errstate(all="ignore"):
            qb, db = Q.quantize_int8_blocks(x, 32)
        for r in range(x.shape[0]):
            q, d = _scalar_rule(list(x[r]))
            assert list(qb[r, 0]) == q and db[r, 0].tobytes() == d.tobytes(), (name, r)
    # float64 input is rounded to float32 first, and a float64 above the float32 range is an Inf (a non-finite block)
    big = np.zeros((1, 32))
    big[0, 0] = 1e300
    with np.errstate(all="ignore"):
        q, d = Q.quantize_int8_blocks(big, 32)
    assert not q.any() and np.float32(d[0, 0]).view(np.uint32) == 0x7FC00000


def test_nan_propagates_through_the_swiglu_clamps():
    """CONTRACTS.md: NaN passes the clamps as through torch.clamp (np.clip / np.minimum do; np.fmin / np.fmax would return the constant)."""
    from ref.ds41 import moe as M
    rng = np.random.default_rng(32)
    x = rng.standard_normal((3, 64)).astype(np.float32)
    w1, w3, w2 = (rng.standard_normal(s).astype(np.float32) / 8 for s in ((96, 64), (96, 64), (64, 96)))
    wt = np.full(3, 0.7, np.float32)
    base = M.expert(x, w1, w3, w2, swiglu_limit=10.0, weights=wt, quant=Q.QuantConfig.exact())
    assert np.isfinite(base).all()
    for which in ("g", "u"):
        a, b = w1.copy(), w3.copy()
        (a if which == "g" else b)[5, 7] = np.nan               # NaN in one row of W1 (g) or W3 (u): h[:, 5] is NaN, so is every y
        for quant in (Q.QuantConfig.exact(), Q.QuantConfig.int8()):
            y = M.expert(x, a, b, w2, swiglu_limit=10.0, weights=wt, quant=quant)
            assert np.isnan(y).all(), (which, quant)
    # a NaN in x reaches the output through the quantiser too (int8 mode) and through the matmul (exact mode)
    xn = x.copy()
    xn[1, 3] = np.nan
    for quant in (Q.QuantConfig.exact(), Q.QuantConfig.int8()):
        y = M.expert(xn, w1, w3, w2, swiglu_limit=10.0, weights=wt, quant=quant)
        assert np.isnan(y[1]).all() and np.isfinite(y[[0, 2]]).all()


def test_int8_mode_in_linear_and_expert():
    from ref.ds41 import moe as M
    from ref.ds41.ops import linear
    rng = np.random.default_rng(15)
    x = rng.standard_normal((6, 64)).astype(np.float32)
    w = rng.standard_normal((48, 64)).astype(np.float32) / 8
    exact = linear(x, w, act_quant=True, quant=Q.QuantConfig.exact())
    i8 = linear(x, w, act_quant=True, quant=Q.QuantConfig.int8())
    assert np.array_equal(i8, Q.act_quant_int8(x) @ w.T) and 0 < np.abs(i8 - exact).max() < 0.02 * np.abs(exact).max()
    assert np.array_equal(linear(x, w, act_quant=False, quant=Q.QuantConfig.int8()), exact)     # bf16 weights: untouched
    w1, w3, w2 = (rng.standard_normal(s).astype(np.float32) / 8 for s in ((96, 64), (96, 64), (64, 96)))
    e = M.expert(x, w1, w3, w2, swiglu_limit=10.0, weights=np.full(6, 0.7, np.float32), quant=Q.QuantConfig.exact())
    q = M.expert(x, w1, w3, w2, swiglu_limit=10.0, weights=np.full(6, 0.7, np.float32), quant=Q.QuantConfig.int8())
    assert 0 < np.abs(q - e).max() / np.abs(e).max() < 0.03          # int8 error is smaller than e4m3's (see README)


def test_quant_config_modes():
    assert not any(vars(Q.QuantConfig.exact()).values())
    r = vars(Q.QuantConfig.reference())
    assert r.pop("int8_act") is False and all(r.values())
    i = vars(Q.QuantConfig.int8())
    assert i.pop("int8_act") is True and not any(i.values())
