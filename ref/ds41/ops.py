"""ref/ds41/ops.py - elementwise / normalisation primitives and the linear layer.

RMSNorm    model.py:281-293   fp32, eps = norm_eps = 1e-20 (the checkpoint's `rms_norm_eps`), weight applied after
linear     model.py:181-207   x @ W^T; the reference first fake-quantises x to e4m3 (act_quant, 32) when the weight is
                              stored fp8/fp4 -- `QuantConfig.linear_act` switches that on (`int8_act`: the engine's
                              int8 per 32 instead).
"""
from __future__ import annotations

import numpy as np

from .quant import QuantConfig, act_quant_fp8, act_quant_int8


def _wide(x: np.ndarray) -> np.ndarray:
    """At least float32 (model.py does `x.float()` before every statistic)."""
    x = np.asarray(x)
    return x if x.dtype in (np.float32, np.float64) else x.astype(np.float32)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """model.py:RMSNorm.forward: x * rsqrt(mean(x^2, -1) + eps) * weight, in fp32 (eps 1e-20 is *not* a typo)."""
    x = _wide(x)
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    return (weight * (x / np.sqrt(var + eps))).astype(x.dtype, copy=False)


def rms_scale(x: np.ndarray, eps: float) -> np.ndarray:
    """rsqrt(mean(x^2, -1) + eps), keepdims (used by mHC and Engram, which fold the scale into a dot product)."""
    x = _wide(x)
    return 1.0 / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = _wide(x)
    # split form: no overflow for large |x|
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e)).astype(x.dtype, copy=False)


def silu(x: np.ndarray) -> np.ndarray:
    x = _wide(x)
    return (x * sigmoid(x)).astype(x.dtype, copy=False)


def softplus(x: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """torch.nn.functional.softplus (beta = 1, threshold = 20): log1p(exp(x)), identity above the threshold."""
    x = _wide(x)
    return np.where(x > threshold, x, np.log1p(np.exp(np.minimum(x, threshold)))).astype(x.dtype, copy=False)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = _wide(x)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def linear(x: np.ndarray, w: np.ndarray, *, act_quant: bool = False, quant: QuantConfig | None = None) -> np.ndarray:
    """y = x @ w.T for w [out, in] (dense float).  `act_quant` marks weights the reference stores as fp8/fp4: with
    `quant.linear_act` the activation is first fake-quantised to e4m3 per 32 along K (model.py:linear -> act_quant ->
    fp8_gemm/fp4_gemm; the GEMM then equals dequantised-activation @ dequantised-weight^T)."""
    if act_quant and quant is not None:
        if quant.linear_act:
            x = act_quant_fp8(x, 32)
        elif quant.int8_act:
            x = act_quant_int8(x, 32)
    return x @ w.T
