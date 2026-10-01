"""ref/ds41/quant.py - number formats: decoders for every stored encoding and the reference's activation fake-quant.

Stored formats the oracle can read (docs/deepseek/RESEARCH.md s1, s10):
  * MXFP4, GGML `block_mxfp4` (17 B / 32 values)          `dequant_mxfp4`            routed experts, Engram rows (GGUF)
  * Q8_0, GGML `block_q8_0` (34 B / 32 values)            `dequant_q8_0`             FP8-origin weights in the GGUF
  * BF16 / F16 / F32                                      `bf16_to_f32`, ...
  * FP8 e4m3 + E8M0 scale per 32x32 weight block          `dequant_fp8_block`        official checkpoint
  * official packed e2m1 (2 per byte, element 2k = low nibble) + E8M0 per 32 along K   `dequant_fp4_official`,
    `mxfp4_blocks_from_official` (repack to the GGML layout - what the GGUF author did)

Activation fake-quant the *reference* applies (kernel.py): `act_quant_fp8` (act_quant, in place, e4m3 with a power of
two scale per 32 - applied before every fp8/fp4 GEMM and to the SWA KV), `fp4_quant_e8m0` (fp4_act_quant, indexer q/k,
block 32, power-of-two scale), `fp4_quant_e4m3` (fp4_act_quant with E4M3 scales, block 16: the compressed KV).
They are *optional* in the oracle (see `QuantConfig`): the C++ engine uses int8 activations (CONTRACTS.md), which are
more precise than the reference's e4m3 ones, so the difference between the two modes is something to measure.

Everything is NumPy.  Float32 is used wherever kernel.py computes scales in fp32 (the scale arithmetic is done in
float32 even when the data is float64, so that power-of-two scale choices agree with the GPU kernels).
"""
# Attribution: the MXFP4 / Q8_0 / E8M0 semantics and the int8 activation rule follow ggml (llama.cpp ggml-quants.c, ggml-impl.h, ggml-cpu/arch/x86/quants.c), MIT License, Copyright (c) 2023-2026 The ggml authors (notice: src/ds41/cuda/ds41_math.cuh, third_party/ggml/LICENSE).
from __future__ import annotations

import dataclasses

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# small float formats
# ---------------------------------------------------------------------------------------------------------------


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """bfloat16 bit patterns (uint16) -> float32 (exact)."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bfloat16 (RNE), returned as float32.  NaN/inf pass through."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)
    out = rounded.view(np.float32)
    return np.where(np.isfinite(x), out, x)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    return (f32_to_bf16_round(x).view(np.uint32) >> 16).astype(np.uint16)


def _build_e4m3_table() -> np.ndarray:
    t = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 0xF and m == 7:
            t[b] = np.nan                                  # e4m3fn: the only NaN (no inf)
        elif e == 0:
            t[b] = s * m * 2.0 ** -9                       # subnormal: m/8 * 2^-6
        else:
            t[b] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return t


E4M3_TABLE = _build_e4m3_table()
FP4_E2M1_TABLE = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0.0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
#: GGML's doubled table (ggml-common.h kvalues_fp4); value = KVALUES_FP4[code] * 2^(e - 128)
KVALUES_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.int8)


def fp8_e4m3_to_f32(u8: np.ndarray) -> np.ndarray:
    return E4M3_TABLE[np.asarray(u8, dtype=np.uint8)]


def e8m0_to_f32(u8: np.ndarray) -> np.ndarray:
    """E8M0 scale byte -> 2^(b - 127) (official / torch.float8_e8m0fnu: 255 is NaN, 0 is 2^-127)."""
    b = np.asarray(u8, dtype=np.uint8).astype(np.int32)
    with np.errstate(over="ignore"):
        out = np.ldexp(np.float32(1.0), b - 127).astype(np.float32)
    return np.where(b == 255, np.float32(np.nan), out)


def e8m0_half_ggml(u8: np.ndarray) -> np.ndarray:
    """ggml-impl.h ggml_e8m0_to_fp32_half: 2^(e - 128) for every e in 0..255 (e = 0, 1 are fp32 denormals,
    e = 255 is the finite 2^127 here, not NaN)."""
    b = np.asarray(u8, dtype=np.uint8).astype(np.int32)
    with np.errstate(over="ignore"):
        return np.ldexp(np.float32(1.0), b - 128).astype(np.float32)


def round_e4m3(v: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even onto the e4m3fn grid (no saturation: clamp first).  Matches torch's float32 ->
    float8_e4m3fn cast (checked in tests/test_quant.py)."""
    v = np.asarray(v)
    a = np.abs(v)
    _, e = np.frexp(a)                                     # a = m * 2^e, m in [0.5, 1)
    ex = np.maximum(e - 1, -6)                             # floor(log2 a), clamped at the subnormal binade
    step = np.ldexp(np.ones_like(a), ex - 3)               # 3 mantissa bits
    return np.copysign(np.rint(a / step) * step, v).astype(v.dtype, copy=False)


def round_e2m1(v: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even onto the e2m1 grid {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating at +-6 (the
    `T.clamp(.., -6, 6)` + cast of kernel.py:fp4_quant_kernel)."""
    v = np.asarray(v)
    a = np.minimum(np.abs(v), 6.0)
    _, e = np.frexp(a)
    ex = np.maximum(e - 1, 0)                              # binades [0,2), [2,4), [4,8): steps .5, 1, 2
    step = np.ldexp(np.ones_like(a), ex - 1)
    return np.copysign(np.rint(a / step) * step, v).astype(v.dtype, copy=False)


# ---------------------------------------------------------------------------------------------------------------
# the reference's activation fake-quantisation (kernel.py)
# ---------------------------------------------------------------------------------------------------------------

_F32_INV448 = np.float32(1.0 / 448.0)
_F32_INV6 = np.float32(1.0 / 6.0)


def _round_scale_pow2(amax: np.ndarray, inv_max: np.float32) -> np.ndarray:
    """kernel.py fast_round_scale: 2^ceil(log2(amax * inv_max)), computed on the float32 product."""
    t = (amax.astype(np.float32) * inv_max).astype(np.float32)
    m, e = np.frexp(t)                                     # t = m * 2^e, m in [0.5, 1)
    ceil_log2 = np.where(m == 0.5, e - 1, e)
    return np.ldexp(np.float32(1.0), ceil_log2).astype(np.float32)


def act_quant_fp8(x: np.ndarray, block: int = 32) -> np.ndarray:
    """kernel.py:act_quant(inplace=True, scale_fmt='ue8m0'): per `block` along the last axis,
    s = 2^ceil(log2(max(amax, 1e-4)/448)); x <- e4m3(clamp(x/s, +-448)) * s.  Returns the dequantised array."""
    x = np.asarray(x)
    n = x.shape[-1]
    assert n % block == 0, (n, block)
    xb = x.reshape(*x.shape[:-1], n // block, block)
    amax = np.maximum(np.abs(xb).max(axis=-1, keepdims=True), 1e-4)
    s = _round_scale_pow2(amax, _F32_INV448).astype(x.dtype)
    y = round_e4m3(np.clip(xb / s, -448.0, 448.0)) * s
    return y.reshape(x.shape).astype(x.dtype, copy=False)


def fp4_quant_e8m0(x: np.ndarray, block: int = 32) -> np.ndarray:
    """kernel.py:fp4_act_quant(inplace=True) with E8M0 scales (indexer q and index-K): per `block`,
    s = 2^ceil(log2(max(amax, 6*2^-126)/6)); x <- e2m1(clamp(x/s, +-6)) * s."""
    x = np.asarray(x)
    n = x.shape[-1]
    assert n % block == 0, (n, block)
    xb = x.reshape(*x.shape[:-1], n // block, block)
    amax = np.maximum(np.abs(xb).max(axis=-1, keepdims=True), 6.0 * 2.0 ** -126)
    s = _round_scale_pow2(amax, _F32_INV6).astype(x.dtype)
    y = round_e2m1(np.clip(xb / s, -6.0, 6.0)) * s
    return y.reshape(x.shape).astype(x.dtype, copy=False)


def fp4_quant_e4m3(x: np.ndarray, block: int = 16) -> np.ndarray:
    """kernel.py:fp4_act_quant(inplace=True, scale_dtype=e4m3) (the compressed KV, block 16): per `block`,
    s = e4m3(max(amax, 6*2^-9)/6) (an *ordinary* e4m3 value, not a power of two); x <- e2m1(clamp(x/s, +-6)) * s."""
    x = np.asarray(x)
    n = x.shape[-1]
    assert n % block == 0, (n, block)
    xb = x.reshape(*x.shape[:-1], n // block, block)
    amax = np.maximum(np.abs(xb).max(axis=-1, keepdims=True), 6.0 * 2.0 ** -9)
    s = round_e4m3((amax.astype(np.float32) / np.float32(6.0)).astype(np.float32))
    s = np.minimum(s, 448.0).astype(x.dtype)               # (e4m3 has no larger value; real KV never gets near)
    y = round_e2m1(np.clip(xb / s, -6.0, 6.0)) * s
    return y.reshape(x.shape).astype(x.dtype, copy=False)


_INT8_INF_BITS = np.uint32(0x7F800000)         # |x| bits >= this: Inf or NaN
_INT8_TINY_BITS = np.uint32(0x0D800000)        # 2^-100: exponent field 127 - 100 = 27
_INT8_NAN = np.array([0x7FC00000], np.uint32).view(np.float32)[0]      # the canonical quiet NaN (a pinned payload: the bytes are the same everywhere)


def quantize_int8_blocks(x: np.ndarray, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """The engine's activation quantiser, CONTRACTS.md "Activations", bit for bit the rule of the C++ kernels (CPU scalar / AVX2 /
    AVX-512, the GPU kernel, the GPU emulator).  x [..., n] -> (q int8 [..., n/block, block], d float32 [..., n/block]).

    This is ggml's x86 SIMD `quantize_row_q8_0` (ggml-cpu/arch/x86/quants.c) with an FP32 d, and NOT `quantize_row_q8_0_ref` (which
    divides, `x * (1 / d)` with ties away from zero, `roundf`): per block of 32, in float32 throughout,
      * the largest magnitude is the INTEGER maximum of `bits & 0x7FFFFFFF` (order-independent; a float max with a NaN operand is not);
      * non-finite block (that maximum >= 0x7F800000: an Inf or a NaN):  d = NaN (0x7FC00000), every q = 0  - the NaN reaches the output;
      * amax < 2^-100 (all-zero blocks and denormals included; 127 / amax would overflow below ~3.7e-37):  d = 0, every q = 0;
      * otherwise  d = amax / 127,  id = 127 / amax  (two float32 divisions),  q = rint(x * id)  (float32 product, round half to EVEN),
        clamped to [-127, 127].
    float64 input is rounded to float32 first (the kernels see float32 activations)."""
    x = np.asarray(x)
    n = x.shape[-1]
    assert n % block == 0, (n, block)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xb = np.ascontiguousarray(x.reshape(*x.shape[:-1], n // block, block), dtype=np.float32)
        m = (xb.view(np.uint32) & np.uint32(0x7FFFFFFF)).max(axis=-1)
        nonfinite = m >= _INT8_INF_BITS
        tiny = m < _INT8_TINY_BITS
        special = nonfinite | tiny
        amax = np.ascontiguousarray(m).view(np.float32)
        d = amax / np.float32(127.0)
        idv = np.float32(127.0) / amax
        d = np.where(nonfinite, _INT8_NAN, np.where(tiny, np.float32(0.0), d)).astype(np.float32)
        idv = np.where(special, np.float32(0.0), idv).astype(np.float32)
        v = xb * idv[..., None]
        q = np.where(special[..., None], np.float32(0.0), np.clip(np.rint(v), -127, 127)).astype(np.int8)
    return q, d


def act_quant_int8(x: np.ndarray, block: int = 32) -> np.ndarray:
    """The engine's activation format (docs/deepseek/CONTRACTS.md): int8 per `block` along the last axis with an FP32 scale, by the rule of
    `quantize_int8_blocks` (d = max|x| / 127, q = rint(x * (127 / max|x|)) with ties to EVEN, the Inf / NaN / tiny cases pinned; NOT the
    `1 / d`, ties-away-from-zero rule of ggml's `quantize_row_q8_0_ref` this function used to follow).  Returns the dequantised array q * d
    in the input's floating dtype (float32 for anything else): an all-zero or tiny block stays 0, a block holding an Inf or a NaN becomes NaN."""
    x = np.asarray(x)
    dt = x.dtype if x.dtype in (np.float32, np.float64) else np.dtype(np.float32)
    q, d = quantize_int8_blocks(x, block)
    with np.errstate(invalid="ignore", over="ignore"):
        return (q.astype(dt) * d.astype(dt)[..., None]).reshape(x.shape)


@dataclasses.dataclass(frozen=True)
class QuantConfig:
    """Which activation (fake-)quantisations the oracle applies.

    `QuantConfig.exact()`      none: weights are used as stored, activations stay in the compute dtype (the mode the
                               real-weight comparisons use);
    `QuantConfig.reference()`  all of the official reference's, as in model.py + kernel.py (e4m3 activations before every
                               fp8/fp4 GEMM, fp8 SWA KV, fp4 indexer q/k, fp4(e4m3-scale) compressed KV);
    `QuantConfig.int8()`       the C++ engine's choice (CONTRACTS.md): int8 per 32 + fp32 scale before every GEMM whose
                               weight the reference stores fp8/fp4, no KV quantisation (fp16 KV).
    """
    linear_act: bool = False      # act_quant(x, 32) before every fp8/fp4-weight GEMM       (model.py:181-207)
    window_kv: bool = False       # act_quant(kv, 32, inplace) on the SWA KV                (model.py:704-707)
    compressed_kv: bool = False   # fp4_act_quant(latent, 16, e4m3) on the compressed KV    (model.py:757)
    index: bool = False           # fp4_act_quant(q|k, 32, e8m0) in the indexer             (model.py:550, 561)
    int8_act: bool = False        # engine: int8 activations at the same GEMM sites as linear_act (mutually exclusive)

    @classmethod
    def exact(cls) -> "QuantConfig":
        return cls()

    @classmethod
    def reference(cls) -> "QuantConfig":
        return cls(True, True, True, True)

    @classmethod
    def int8(cls) -> "QuantConfig":
        return cls(int8_act=True)


# ---------------------------------------------------------------------------------------------------------------
# weight decoders
# ---------------------------------------------------------------------------------------------------------------

def dequant_fp8_block(w: np.ndarray, scale: np.ndarray, block: int = 32) -> np.ndarray:
    """Official FP8 weight [N, K] (uint8 e4m3 bytes, or already-decoded floats) with one E8M0 scale (uint8 byte, or
    already-decoded float) per `block` x `block` tile ([ceil(N/block), ceil(K/block)]) -> float32 [N, K].
    model.py:210-240 (Linear), kernel.py:fp8_gemm semantics."""
    w = np.asarray(w)
    wf = fp8_e4m3_to_f32(w) if w.dtype == np.uint8 else w.astype(np.float32)
    scale = np.asarray(scale)
    sf = e8m0_to_f32(scale) if scale.dtype == np.uint8 else scale.astype(np.float32)
    n, k = wf.shape
    if n % block == 0 and k % block == 0:                       # broadcast the tile scales, no expanded copy
        return (wf.reshape(n // block, block, k // block, block) * sf[:, None, :, None]).reshape(n, k)
    s = np.repeat(np.repeat(sf, block, axis=0), block, axis=1)[:n, :k]
    return wf * s


def unpack_fp4_official(packed: np.ndarray) -> np.ndarray:
    """Official packed e2m1 [N, K/2] (uint8; element 2k = low nibble, 2k+1 = high nibble) -> codes [N, K] (uint8)."""
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty((*p.shape[:-1], p.shape[-1] * 2), np.uint8)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = p >> 4
    return out


def pack_fp4_official(codes: np.ndarray) -> np.ndarray:
    c = np.asarray(codes, dtype=np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def dequant_fp4_official(packed: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Official routed-expert weight: packed e2m1 [N, K/2] + E8M0 bytes [N, K/32] (one per 32 along K) -> float32
    [N, K].  kernel.py:fp4_gemm semantics (value = table[code] * 2^(e-127))."""
    codes = unpack_fp4_official(packed)
    n, k = codes.shape
    vals = FP4_E2M1_TABLE[codes].reshape(n, k // 32, 32)
    return (vals * e8m0_to_f32(scale_u8)[:, :, None]).reshape(n, k).astype(np.float32)


def mxfp4_blocks_from_official(packed: np.ndarray, scale_u8: np.ndarray) -> np.ndarray:
    """Official (packed e2m1 [N, K/2], E8M0 [N, K/32]) -> GGML MXFP4 blocks uint8 [N, K/32, 17].
    Official element order inside a 32-block is 0..31 (2k = low nibble); GGML's is value j = low nibble of qs[j],
    value j+16 = high nibble.  The scale byte carries over unchanged (table * 2^(e-127) == doubled table * 2^(e-128))."""
    codes = unpack_fp4_official(packed)
    n, k = codes.shape
    c = codes.reshape(n, k // 32, 32)
    out = np.empty((n, k // 32, 17), np.uint8)
    out[..., 0] = np.asarray(scale_u8, dtype=np.uint8)
    out[..., 1:] = c[..., :16] | (c[..., 16:] << 4)
    return out


def _as_blocks(b: np.ndarray, size: int) -> np.ndarray:
    """uint8 [..., nb, size] (ndim >= 3) or raw rows [..., nb*size] -> [..., nb, size]."""
    b = np.asarray(b, dtype=np.uint8)
    if b.ndim >= 3 and b.shape[-1] == size:
        return b
    assert b.shape[-1] % size == 0, b.shape
    return b.reshape(*b.shape[:-1], b.shape[-1] // size, size)


_MXFP4_DT = np.dtype([("e", "u1"), ("qs", "u1", (16,))])             # 17 bytes, packed
_Q8_0_DT = np.dtype([("d", "<f2"), ("q", "i1", (32,))])               # 34 bytes, packed
_LO32 = KVALUES_FP4[np.arange(256) & 0x0F].astype(np.float32)         # byte -> value of its low nibble
_HI32 = KVALUES_FP4[np.arange(256) >> 4].astype(np.float32)           # byte -> value of its high nibble


def dequant_mxfp4(blocks: np.ndarray) -> np.ndarray:
    """GGML block_mxfp4 -> float32.  `blocks` is uint8 [..., nb, 17] (ndim >= 3) or raw rows [..., nb*17]; returns
    [..., nb*32].  ggml-quants.c:dequantize_row_mxfp4: y[j] = kvalues[qs[j] & 15] * d, y[j+16] = kvalues[qs[j] >> 4] * d,
    d = ggml_e8m0_to_fp32_half(e) (= 2^(e-128), exact for all e)."""
    b = np.ascontiguousarray(_as_blocks(blocks, 17))
    rec = b.view(_MXFP4_DT)[..., 0]                                    # zero-copy structured view [..., nb]
    qs = rec["qs"]
    out = np.concatenate([_LO32[qs], _HI32[qs]], axis=-1)
    with np.errstate(over="ignore"):
        out *= e8m0_half_ggml(rec["e"])[..., None]
    return out.reshape(*out.shape[:-2], out.shape[-2] * 32)


def dequant_q8_0(blocks: np.ndarray) -> np.ndarray:
    """GGML block_q8_0 (fp16 d, int8 qs[32]) -> float32.  `blocks` uint8 [..., nb, 34] (ndim >= 3) or raw rows
    [..., nb*34]."""
    b = np.ascontiguousarray(_as_blocks(blocks, 34))
    rec = b.view(_Q8_0_DT)[..., 0]
    out = rec["q"].astype(np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        out *= rec["d"].astype(np.float32)[..., None]
    return out.reshape(*out.shape[:-2], out.shape[-2] * 32)


# ---------------------------------------------------------------------------------------------------------------
# encoders (test fixtures / mini-GGUF content only; they mirror ggml's reference quantisers)
# ---------------------------------------------------------------------------------------------------------------

def quantize_mxfp4_ggml(x: np.ndarray) -> np.ndarray:
    """ggml-quants.c:quantize_row_mxfp4_ref: float32 [..., K] -> uint8 [..., K/32, 17]."""
    x = np.asarray(x, dtype=np.float32)
    k = x.shape[-1]
    assert k % 32 == 0
    xb = x.reshape(*x.shape[:-1], k // 32, 32)
    amax = np.abs(xb).max(axis=-1)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.floor(np.log2(np.maximum(amax, 1e-45))) - 2 + 127, 0)
    e = np.clip(e, 0, 255).astype(np.uint8)
    d = e8m0_half_ggml(e)[..., None]
    lo, hi = xb[..., :16], xb[..., 16:]

    def best(v):  # first index of the minimal |kvalues*d - v|  (ggml best_index_mxfp4)
        kv = KVALUES_FP4.astype(np.float32)
        vv = v.reshape(-1, 16)
        dd = d.reshape(-1, 1, 1)
        err = np.abs(kv[None, None, :] * dd - vv[:, :, None])
        return np.argmin(err, axis=-1).astype(np.uint8).reshape(v.shape)

    out = np.empty((*xb.shape[:-1], 17), np.uint8)
    out[..., 0] = e
    out[..., 1:] = best(lo) | (best(hi) << 4)
    return out


def quantize_q8_0(x: np.ndarray) -> np.ndarray:
    """ggml-quants.c:quantize_row_q8_0_ref: float32 [..., K] -> uint8 [..., K/32, 34] (fp16 d, int8 qs)."""
    x = np.asarray(x, dtype=np.float32)
    k = x.shape[-1]
    assert k % 32 == 0
    xb = x.reshape(*x.shape[:-1], k // 32, 32)
    d = (np.abs(xb).max(axis=-1) / 127.0).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, 1.0 / d, 0.0).astype(np.float32)
    v = xb * idv[..., None]
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)       # C roundf: half away from zero
    out = np.empty((*xb.shape[:-1], 34), np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(*d.shape, 2)
    out[..., 2:] = q.view(np.uint8)
    return out
