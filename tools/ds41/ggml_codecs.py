"""ggml_codecs.py - numpy encoders and decoders for the GGML types the DeepSeek-V4.1 GGUF uses.

Types: F32, F16, BF16, Q8_0 and MXFP4 (type 39).  Each one is a line-for-line port of the semantics of GGML's C
reference (ggml/src/ggml-quants.c and ggml-impl.h), checked bit for bit against that C code by
tools/ds41/test_codecs.py whenever a ggml source tree and a C compiler are available (see ggml_c_oracle.py).

    MXFP4  block = 1 byte E8M0 scale `e` + 16 bytes of e2m1 codes (17 B per 32 values).  Element j (0..15) is the LOW
           nibble of qs[j], element j + 16 the HIGH nibble.  value = KVALUES_FP4[code] * 2^(e - 128) where
           KVALUES_FP4 is the e2m1 table doubled, so the scale is "half" of the E8M0 value (ggml_e8m0_to_fp32_half).
           This holds for every e, including 0 (2^-128, a float32 denormal), 1 and 255 (2^127).
    Q8_0   block = fp16 d + 32 int8 (34 B per 32 values); value = qs[j] * d.  The encoder is
           d = amax / 127, qs = roundf(x / d) (round half away from zero) with d stored as fp16.
    BF16   the top 16 bits of an fp32; the encoder rounds to nearest even and quiets NaNs.

Everything here works on flat float32 data; callers reshape to (rows, ne0).  Row length (ne0) must be a multiple of
the block size for the block types, exactly as in GGML.
"""
from __future__ import annotations

import numpy as np

QK = 32  # values per block, MXFP4 and Q8_0

# type name -> (values per block, bytes per block).  Same numbers as tools/gguf_reader.py BLOCK_GEOMETRY.
BLOCK = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "MXFP4": (32, 17)}
# type name -> ggml_type id (tools/gguf_reader.py GGML_TYPES, the other way round)
TYPE_ID = {"F32": 0, "F16": 1, "Q8_0": 8, "BF16": 30, "MXFP4": 39}
TYPE_NAME = {v: k for k, v in TYPE_ID.items()}

KVALUES_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8)
_KV_F32 = KVALUES_FP4.astype(np.float32)


def _build_e8m0_half() -> np.ndarray:
    """ggml_e8m0_to_fp32_half for all 256 inputs, from the same bit patterns as the C code."""
    bits = np.empty(256, dtype=np.uint32)
    for x in range(256):
        bits[x] = (0x00200000 << x) if x < 2 else ((x - 1) << 23)
    return bits.view(np.float32)


E8M0_HALF = _build_e8m0_half()  # E8M0_HALF[e] == 2.0 ** (e - 128), as float32


# ---------------------------------------------------------------------------------------------- geometry
def row_bytes(type_name: str, ne0: int) -> int:
    """Bytes of one row of ne0 elements."""
    be, bb = BLOCK[type_name]
    if ne0 % be:
        raise ValueError(f"{type_name}: row of {ne0} elements is not a multiple of the block size {be}")
    return ne0 // be * bb


def tensor_nbytes(type_name: str, dims) -> int:
    """Bytes of a tensor with GGUF dims (ne0, ne1, ...)."""
    n = 1
    for d in dims[1:]:
        n *= int(d)
    return row_bytes(type_name, int(dims[0])) * n


def _as_u8(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    return np.frombuffer(raw, dtype=np.uint8)


# ---------------------------------------------------------------------------------------------- MXFP4
def dequant_mxfp4(raw) -> np.ndarray:
    """MXFP4 blocks (any shape of uint8 / bytes, size a multiple of 17) -> flat float32, 32 values per block."""
    b = _as_u8(raw)
    if b.size % 17:
        raise ValueError(f"MXFP4 data of {b.size} bytes is not a whole number of 17-byte blocks")
    blk = b.reshape(-1, 17)
    d = E8M0_HALF[blk[:, 0]][:, None]
    qs = blk[:, 1:]
    out = np.empty((blk.shape[0], 32), dtype=np.float32)
    with np.errstate(over="ignore"):  # e = 254, 255 with a big code overflows float32 to inf, exactly as in C
        out[:, :16] = _KV_F32[qs & 0x0F] * d
        out[:, 16:] = _KV_F32[qs >> 4] * d
    return out.reshape(-1)


def quant_mxfp4(x, chunk_blocks: int = 1 << 16) -> bytes:
    """float32 values (multiple of 32) -> MXFP4 blocks, as quantize_row_mxfp4_ref.

    e = floor(log2f(amax)) - 2 + 127 (0 for an all-zero block, wrapped to 8 bits like the C cast); every value goes to
    the nearest of the 16 doubled-e2m1 codes times 2^(e-128), ties to the lowest code index.
    """
    xf = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if xf.size % QK:
        raise ValueError(f"MXFP4 needs a multiple of {QK} values, got {xf.size}")
    xb = xf.reshape(-1, QK)
    out = np.empty((xb.shape[0], 17), dtype=np.uint8)
    for s in range(0, xb.shape[0], chunk_blocks):
        blk = xb[s:s + chunk_blocks]
        amax = np.abs(blk).max(axis=1)
        # float32 log2 then floor, like floorf(log2f(amax)): for the one float just below a power of two the log2 rounds
        # up to the integer, so e is one higher than the exact floor would give (frexp would differ there)
        e = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, np.float32(1)))).astype(np.int64) - 2 + 127,
                     0) & 0xFF
        d = E8M0_HALF[e]
        with np.errstate(over="ignore"):   # a wrapped e of 255 overflows 12 * 2^127 to inf, as in C (the block is garbage)
            err = np.abs(_KV_F32[None, None, :] * d[:, None, None] - blk[:, :, None])
        idx = err.argmin(axis=2).astype(np.uint8)  # first minimum, like the strict `<` loop in the C code
        out[s:s + blk.shape[0], 0] = e.astype(np.uint8)
        out[s:s + blk.shape[0], 1:] = idx[:, :16] | (idx[:, 16:] << 4)
    return out.tobytes()


# ---------------------------------------------------------------------------------------------- official FP4 <-> MXFP4
# DeepSeek's safetensors store a routed expert row as K/2 bytes (I8) plus K/32 E8M0 scale bytes (F8_E8M0).  Byte k holds
# element 2k in its LOW nibble and element 2k + 1 in its HIGH nibble (third_party/.../inference/convert.py:32-34),
# code table [0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, ...], value = table[code] * 2^(e - 127).  GGML's block_mxfp4 uses the
# same codes and the same scale byte, but byte j of a block holds element j (low) and element j + 16 (high).  So the
# two are NOT bit-identical (docs/deepseek/RESEARCH.md section 1 says they are): the nibbles are permuted within every
# 32-value block, the scale bytes are copied.  A GGUF produced from the official weights has applied this permutation.
def mxfp4_from_official(packed, scale) -> bytes:
    """packed (rows, K/2) int8/uint8, scale (rows, K/32) E8M0 bytes -> GGML MXFP4 blocks, row after row."""
    p = np.ascontiguousarray(packed).view(np.uint8)
    sc = np.ascontiguousarray(scale).view(np.uint8)
    rows = p.shape[0]
    p = p.reshape(rows, -1, 16)                       # (rows, blocks, 16 bytes)
    sc = sc.reshape(rows, -1)
    if p.shape[1] != sc.shape[1]:
        raise ValueError(f"{p.shape[1]} blocks of nibbles but {sc.shape[1]} scales per row")
    codes = np.empty((rows, p.shape[1], 32), dtype=np.uint8)
    codes[..., 0::2] = p & 0x0F                       # element 2k   = low nibble of byte k
    codes[..., 1::2] = p >> 4                         # element 2k+1 = high nibble of byte k
    out = np.empty((rows, p.shape[1], 17), dtype=np.uint8)
    out[..., 0] = sc
    out[..., 1:] = codes[..., :16] | (codes[..., 16:] << 4)
    return out.tobytes()


def mxfp4_to_official(raw, row_blocks: int):
    """Inverse of mxfp4_from_official: GGML blocks -> (packed (rows, K/2) uint8, scale (rows, K/32) uint8)."""
    b = _as_u8(raw).reshape(-1, row_blocks, 17)
    rows = b.shape[0]
    codes = np.empty((rows, row_blocks, 32), dtype=np.uint8)
    codes[..., :16] = b[..., 1:] & 0x0F
    codes[..., 16:] = b[..., 1:] >> 4
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.reshape(rows, row_blocks * 16), b[..., 0].copy()


# ---------------------------------------------------------------------------------------------- Q8_0
def _round_half_away(v: np.ndarray) -> np.ndarray:
    """C roundf for float32 (ties away from zero), exact: no `v + 0.5` rounding trap."""
    t = np.trunc(v)
    frac = v - t
    return t + np.where(np.abs(frac) >= np.float32(0.5), np.sign(frac), np.float32(0.0)).astype(np.float32)


def dequant_q8_0(raw) -> np.ndarray:
    b = _as_u8(raw)
    if b.size % 34:
        raise ValueError(f"Q8_0 data of {b.size} bytes is not a whole number of 34-byte blocks")
    blk = b.reshape(-1, 34)
    d = np.ascontiguousarray(blk[:, :2]).view("<f2").astype(np.float32)  # (n, 1)
    q = np.ascontiguousarray(blk[:, 2:]).view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)


def quant_q8_0(x) -> bytes:
    xf = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if xf.size % QK:
        raise ValueError(f"Q8_0 needs a multiple of {QK} values, got {xf.size}")
    xb = xf.reshape(-1, QK)
    amax = np.abs(xb).max(axis=1)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    q = _round_half_away(xb * idv[:, None]).astype(np.int8)
    out = np.empty((xb.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype("<f2").view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


# ---------------------------------------------------------------------------------------------- BF16 / F16 / F32
def f32_to_bf16_bits(x) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).reshape(-1).view(np.uint32)
    nan = (u & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    with np.errstate(over="ignore"):
        r = (u + (np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1)))) >> np.uint32(16)
    r = np.where(nan, (u >> np.uint32(16)) | np.uint32(64), r)
    return r.astype(np.uint16)


def bf16_bits_to_f32(bits) -> np.ndarray:
    b = np.ascontiguousarray(bits, dtype=np.uint16).reshape(-1)
    return (b.astype(np.uint32) << np.uint32(16)).view(np.float32)


# ---------------------------------------------------------------------------------------------- generic front end
def encode(type_name: str, x) -> bytes:
    """float32 data (rows of ne0 values, flattened) -> the GGUF bytes of that type."""
    if type_name == "F32":
        return np.ascontiguousarray(x, dtype="<f4").tobytes()
    if type_name == "F16":
        return np.ascontiguousarray(x, dtype=np.float32).astype("<f2").tobytes()
    if type_name == "BF16":
        return f32_to_bf16_bits(x).astype("<u2").tobytes()
    if type_name == "Q8_0":
        return quant_q8_0(x)
    if type_name == "MXFP4":
        return quant_mxfp4(x)
    raise NotImplementedError(f"no encoder for {type_name}")


def decode(type_name: str, raw) -> np.ndarray:
    """GGUF bytes of a type -> flat float32."""
    if type_name == "F32":
        return np.frombuffer(raw, dtype="<f4").astype(np.float32)
    if type_name == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if type_name == "BF16":
        return bf16_bits_to_f32(np.frombuffer(raw, dtype="<u2"))
    if type_name == "Q8_0":
        return dequant_q8_0(raw)
    if type_name == "MXFP4":
        return dequant_mxfp4(raw)
    raise NotImplementedError(f"no decoder for {type_name} (this tool reads F32, F16, BF16, Q8_0, MXFP4)")
