"""ref/ds41/rope.py - partial RoPE with YaRN (adjacent-pair complex rotation) and its inverse.

Transcribes model.py:precompute_freqs_cis (369-390) and apply_rotary_emb (392-407).

Only the last `rope_head_dim` (64) of the 512 head channels are rotated; the rotation pairs *adjacent* elements
(x[2i], x[2i+1]) as a complex number.  Which theta/YaRN a layer uses (model.py:Attention.__init__, 682-691):
  * compress_ratio == 0 (layers 0, 1; DSpark): theta = rope_theta (10000), YaRN OFF (original_seq_len = 0);
  * compress_ratio  > 0 (layers 2..39):        theta = compress_rope_theta (160000), YaRN with
    factor / beta_fast / beta_slow / original_seq_len -- used for q, the SWA KV and the compressed KV alike.
`layer_rope_params(cfg, layer)` returns that choice.  There is no attention-temperature (mscale) term in this model.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Config


def layer_rope_params(cfg: Config, layer: int) -> dict:
    if cfg.compress_ratios[layer]:
        return dict(original_seq_len=cfg.original_seq_len, base=cfg.compress_rope_theta)
    return dict(original_seq_len=0, base=cfg.rope_theta)


def rope_table(dim: int, seqlen: int, original_seq_len: int, base: float, factor: float, beta_fast: float,
               beta_slow: float, dtype=np.float64) -> tuple[np.ndarray, np.ndarray]:
    """(cos, sin), each [seqlen, dim/2].  model.py:precompute_freqs_cis.  Computed in float64 by default (the
    reference uses float32; a float32 table is available with dtype=np.float32 for bit-closer comparisons)."""
    dt = np.dtype(dtype)
    freqs = (1.0 / (dt.type(base) ** (np.arange(0, dim, 2, dtype=dt) / dt.type(dim)))).astype(dt)
    if original_seq_len > 0:
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=dt) - low) / max(high - low, 1e-3), 0, 1)
        smooth = 1 - ramp
        freqs = (freqs / factor * (1 - smooth) + freqs * smooth).astype(dt)
    ang = np.outer(np.arange(seqlen, dtype=dt), freqs)
    return np.cos(ang).astype(dt), np.sin(ang).astype(dt)


def layer_rope_table(cfg: Config, layer: int, seqlen: int | None = None, dtype=np.float64):
    p = layer_rope_params(cfg, layer)
    return rope_table(cfg.rope_head_dim, seqlen or cfg.max_seq_len, p["original_seq_len"], p["base"],
                      cfg.rope_factor, cfg.beta_fast, cfg.beta_slow, dtype)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Rotate adjacent pairs of the LAST axis of x (length rd): (a + ib) * (cos + i sin); `inverse` conjugates.
    x is [s, ..., rd]; cos/sin are [s, rd/2] and broadcast over the middle axes.  Returns a new array."""
    x = np.asarray(x)
    c = cos.reshape(cos.shape[0], *([1] * (x.ndim - 2)), cos.shape[-1]).astype(x.dtype, copy=False)
    s = sin.reshape(sin.shape[0], *([1] * (x.ndim - 2)), sin.shape[-1]).astype(x.dtype, copy=False)
    if inverse:
        s = -s
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def apply_rope_tail(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, rd: int, inverse: bool = False) -> np.ndarray:
    """Rotate only the last `rd` channels of x [s, ..., d]; the first d-rd (the 'nope' part) pass through."""
    out = np.array(x, copy=True)
    out[..., -rd:] = apply_rope(x[..., -rd:], cos, sin, inverse)
    return out
