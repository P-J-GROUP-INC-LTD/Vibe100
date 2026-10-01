"""ref/ds41/mhc.py - manifold-constrained hyper-connections (mHC): the 4-copy residual stream.

Transcribes model.py:Block.hc_mixes / hc_pre / hc_post (948-966) and kernel.py:hc_split_sinkhorn_kernel (407-462).

Per sub-layer (attention and FFN each have their own hc_fn [24, 4*dim], hc_base [24], hc_scale [3]):

    xf     = flatten(stream [4, dim])                                   (fp32)
    mixes  = (hc_fn @ xf) * rsqrt(mean(xf^2) + norm_eps)                 (norm_eps = 1e-20, NOT hc_eps)
    pre    = sigmoid(mixes[0:4]  * scale[0] + base[0:4])  + hc_eps
    post   = 2 * sigmoid(mixes[4:8] * scale[1] + base[4:8])
    comb   = mixes[8:24] * scale[2] + base[8:24]   viewed [4(j), 4(k)]
    comb   = softmax(comb, k) + eps;  comb /= (colsum + eps)                        (colsum = sum over j)
    19 x:    comb /= (rowsum + eps);  comb /= (colsum + eps)                        (20 Sinkhorn iterations in all)
    x_in   = sum_c pre[c] * stream[c]
    new[k] = post[k] * f(x_in) + sum_j comb[j, k] * stream[j]

The "single-pass one-block lag" lives in `model.Model.block`: a block's attention collapses with the `pre` that the
PREVIOUS block's FFN mixes produced (block 0 uses [1, 0, 0, 0]); its FFN collapses with the `pre` of its own
attention mixes; the final head fold uses the last block's FFN `pre`.
"""
from __future__ import annotations

import numpy as np

from .ops import _wide, sigmoid, softmax


def sinkhorn_split(mixes: np.ndarray, scale: np.ndarray, base: np.ndarray, hc: int = 4, iters: int = 20,
                   eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kernel.py:hc_split_sinkhorn.  mixes [n, (2+hc)*hc] -> pre [n, hc], post [n, hc], comb [n, hc, hc]."""
    mixes = _wide(mixes)
    n = mixes.shape[0]
    pre = sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * sigmoid(mixes[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (mixes[:, 2 * hc:] * scale[2] + base[2 * hc:]).reshape(n, hc, hc)
    comb = softmax(comb, axis=-1) + eps                              # row softmax (over k)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)           # column normalise (sum over j)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)       # row
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)       # column
    return pre, post, comb


def hc_mixes(x: np.ndarray, fn: np.ndarray, scale: np.ndarray, base: np.ndarray, *, norm_eps: float, hc: int,
             iters: int, eps: float):
    """x [n, hc, dim] -> (pre, post, comb).  model.py:Block.hc_mixes."""
    xf = _wide(x).reshape(x.shape[0], -1)
    rs = 1.0 / np.sqrt(np.mean(np.square(xf), axis=-1, keepdims=True) + norm_eps)
    mixes = (xf @ fn.T) * rs
    return sinkhorn_split(mixes, scale, base, hc, iters, eps)


def hc_pre(x: np.ndarray, pre: np.ndarray) -> np.ndarray:
    """[n, hc, dim] x [n, hc] -> [n, dim] (model.py:Block.hc_pre; also the final 'head fold')."""
    return np.sum(pre[:, :, None] * _wide(x), axis=1)


def hc_post(x: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    """x [n, dim], residual [n, hc, dim], post [n, hc], comb [n, hc(j), hc(k)] -> [n, hc, dim]
    (model.py:Block.hc_post):  new[k] = post[k] * x + sum_j comb[j, k] * residual[j]."""
    return post[:, :, None] * x[:, None, :] + np.einsum("njk,njd->nkd", comb, _wide(residual))
