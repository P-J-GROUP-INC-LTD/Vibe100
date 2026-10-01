"""ref/ds41/moe.py - router, routed/shared expert, MoE layer.

Router   model.py:Gate.forward (809-827), docs/deepseek/CONTRACTS.md "The math"
Expert   model.py:Expert.forward (841-851): the clamps, silu, the routing weight applied BEFORE W2
MoE      model.py:MoE.forward (889-904): sum of routed experts, then + the shared expert, fp32 accumulation
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from .config import Config
from .ops import linear, silu, sigmoid, softmax, softplus, _wide
from .quant import QuantConfig


@dataclasses.dataclass
class RouterOut:
    weights: np.ndarray     # [n, k] float, normalised and scaled (the multipliers of the expert outputs)
    indices: np.ndarray     # [n, k] int64, selected experts in descending (score + bias) order
    scores: np.ndarray      # [n, E] unbiased scores s = sqrt(softplus(logits))
    margin: np.ndarray      # [n] (k-th best) - (k+1-th best) of (s + bias): near 0 = a near-tie the reference may break differently


def router(x: np.ndarray, gate_w: np.ndarray, bias: np.ndarray, *, topk: int, route_scale: float,
           score_func: str = "sqrtsoftplus", gate_temp: float = 1.0, norm_topk_prob: bool = True) -> RouterOut:
    """x [n, dim], gate_w [E, dim] (bf16 in the checkpoint, used in fp32), bias [E] fp32 (`exp_probs_b.bias`).

        s   = sqrt(softplus(x @ Wg^T / gate_temp))          (fp32; `softplus` has torch's threshold-20 branch)
        idx = top-k(s + bias)                                (the bias only SELECTS)
        w   = s[idx]  (unbiased);  w /= (sum(w) + 1e-20);  w *= route_scale

    Ties in (s + bias) are broken towards the lower expert index (torch.topk's tie order is unspecified); `margin`
    reports how close the k-th/(k+1)-th candidates were so that tests can skip near-ties.
    """
    xf = _wide(x)
    logits = (xf @ _wide(gate_w).T) / gate_temp
    if score_func == "softmax":
        s = softmax(logits)
    elif score_func == "sigmoid":
        s = sigmoid(logits)
    else:
        s = np.sqrt(softplus(logits))
    sel = s + bias
    order = np.argsort(-sel, axis=-1, kind="stable")
    idx = order[:, :topk]
    n = s.shape[0]
    w = np.take_along_axis(s, idx, axis=-1)
    if norm_topk_prob and topk > 1:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)     # not norm_eps: "matches training"
    w = w * route_scale
    sorted_sel = np.take_along_axis(sel, order[:, : topk + 1], axis=-1) if sel.shape[-1] > topk else None
    margin = (sorted_sel[:, topk - 1] - sorted_sel[:, topk]) if sorted_sel is not None else np.full(n, np.inf)
    return RouterOut(w.astype(xf.dtype, copy=False), idx.astype(np.int64), s, margin)


def expert(x: np.ndarray, w1: np.ndarray, w3: np.ndarray, w2: np.ndarray, *, swiglu_limit: float,
           weights: np.ndarray | None = None, quant: QuantConfig | None = None) -> np.ndarray:
    """One SwiGLU FFN on x [n, dim] (w1/w3 [inter, dim], w2 [dim, inter]; fp8/fp4 in the reference, so the reference
    quantises the input of all three GEMMs).

        g = W1 x;  u = W3 x;   u = clamp(u, -L, L);  g = min(g, L)          (L = swiglu_limit = 10, only if > 0)
        h = silu(g) * u;       h = w * h        <- routing weight BEFORE the down projection
        y = W2 h                                  (fp32 out)
    `weights` [n] is the routing weight (None for the shared expert)."""
    g = linear(x, w1, act_quant=True, quant=quant)
    u = linear(x, w3, act_quant=True, quant=quant)
    if swiglu_limit > 0:
        u = np.clip(u, -swiglu_limit, swiglu_limit)
        g = np.minimum(g, swiglu_limit)
    h = silu(g) * u
    if weights is not None:
        h = weights[:, None] * h
    return linear(h, w2, act_quant=True, quant=quant)


def moe(x: np.ndarray, *, gate_w: np.ndarray, gate_bias: np.ndarray, get_expert: Callable[[int], tuple],
        shared: tuple, cfg: Config, quant: QuantConfig | None = None, router_out: RouterOut | None = None
        ) -> np.ndarray:
    """MoE FFN: y = sum_k expert_{idx_k}(x; w_k) + shared_expert(x).  x [n, dim] -> [n, dim].
    `get_expert(e)` returns (w1, w3, w2) dense float arrays for routed expert e (lazy: only selected experts are
    ever requested); `shared` is the same triple for the shared expert."""
    r = router_out or router(x, gate_w, gate_bias, topk=cfg.n_activated_experts, route_scale=cfg.route_scale,
                             score_func=cfg.score_func, gate_temp=cfg.gate_temp, norm_topk_prob=cfg.norm_topk_prob)
    y = np.zeros(x.shape, dtype=np.promote_types(_wide(x).dtype, np.float32))
    for e in np.unique(r.indices):
        rows, slot = np.nonzero(r.indices == e)
        w1, w3, w2 = get_expert(int(e))
        y[rows] += expert(x[rows], w1, w3, w2, swiglu_limit=cfg.swiglu_limit, weights=r.weights[rows, slot],
                          quant=quant)
    s1, s3, s2 = shared
    y += expert(x, s1, s3, s2, swiglu_limit=cfg.swiglu_limit, weights=None, quant=quant)
    return y
