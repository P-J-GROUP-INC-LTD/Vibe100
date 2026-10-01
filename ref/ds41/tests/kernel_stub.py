"""Pure-torch stand-in for third_party/deepseek-v41-flash-reference/inference/kernel.py (TileLang, GPU only).

It provides exactly the functions model.py imports -- act_quant, fp4_act_quant, fp4_gemm, fp8_gemm,
hc_split_sinkhorn, sparse_attn -- with the semantics read from the TileLang kernels (line references below are to
kernel.py).  The stubs are written in torch idioms, deliberately NOT sharing code with the oracle's NumPy versions
(ref/ds41/quant.py, attention.py, mhc.py), so that agreement between the two is evidence, not tautology.

Run `install()` before importing the official `model` module.
"""
from __future__ import annotations

import math
import sys
import types

import torch

class Flags:
    """Which fake-quantisations the stubs apply (mirrors ref.ds41.quant.QuantConfig): switching one off turns the
    corresponding kernel call into the identity, which is what lets the oracle be checked in its *exact* mode."""
    linear_act = True      # act_quant(inplace=False) feeding fp8_gemm / fp4_gemm
    window_kv = True       # act_quant(inplace=True)
    compressed_kv = True   # fp4_act_quant(scale_dtype=e4m3)
    index = True           # fp4_act_quant(scale_dtype=e8m0)

    @classmethod
    def set(cls, quant):
        cls.linear_act, cls.window_kv, cls.compressed_kv, cls.index = (
            quant.linear_act, quant.window_kv, quant.compressed_kv, quant.index)


FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_INV448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
_INV6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)


def _pow2_ceil_scale(amax: torch.Tensor, inv: torch.Tensor):
    """kernel.py:fast_round_scale: 2^ceil(log2(amax * inv)) in fp32; returns (scale, integer exponent)."""
    t = amax.float() * inv
    m, e = torch.frexp(t)                              # t = m * 2^e, m in [0.5, 1)
    e = torch.where(m == 0.5, e - 1, e)                # exact powers of two: ceil(log2) = e - 1
    return torch.ldexp(torch.ones_like(t), e), e


def _round_fp4(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value, ties to the even code (the RNE of a cast to float4_e2m1fn); saturates at +-6."""
    a = v.double().abs().clamp(max=6.0)
    d = (a.unsqueeze(-1) - FP4_GRID).abs()
    d = d + (torch.arange(8) % 2).double() * 1e-12     # exact ties -> even code index
    q = FP4_GRID[d.argmin(dim=-1)]
    return torch.copysign(q, v.double()).to(v.dtype)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    n = x.size(-1)
    assert n % block_size == 0
    if inplace and not Flags.window_kv:
        return x
    if not inplace and not Flags.linear_act:
        return x, None                                   # fp8_gemm / fp4_gemm see a float activation: no quantisation
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    if scale_fmt is not None:
        s, e = _pow2_ceil_scale(amax, _INV448)
    else:
        s, e = amax * _INV448, None
    y = (xb / s.unsqueeze(-1)).clamp(-448.0, 448.0)
    q = y.to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    if scale_dtype == torch.float8_e8m0fnu:
        sb = torch.empty(s.shape, dtype=torch.float8_e8m0fnu)
        sb.view(torch.uint8).copy_((e + 127).to(torch.uint8))
        s_out = sb
    else:
        s_out = s.to(scale_dtype)
    return q.flatten(-2).reshape(x.shape), s_out


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace, "model.py only ever calls fp4_act_quant in place"
    n = x.size(-1)
    assert n % block_size == 0
    if not (Flags.compressed_kv if scale_dtype == torch.float8_e4m3fn else Flags.index):
        return x
    xb = x.float().unflatten(-1, (-1, block_size))
    amax = xb.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s, _ = _pow2_ceil_scale(amax, _INV6)
    y = (xb / s.unsqueeze(-1)).clamp(-6.0, 6.0)
    x.copy_((_round_fp4(y) * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
    return x


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    k = a.size(-1)
    n = b.size(0)
    af = a.float() if a_s is None else (a.float().unflatten(-1, (-1, block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    bs = b_s.float().repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:n, :k]
    bf = b.float() * bs
    return (af @ bf.T).to(torch.get_default_dtype())


def _decode_fp4_bytes(b: torch.Tensor) -> torch.Tensor:
    """convert.py's nibble decode: element 2k = low nibble, 2k+1 = high nibble."""
    u = b.view(torch.uint8)
    lo, hi = (u & 0x0F).long(), (u >> 4).long()
    return torch.stack([FP4_TABLE[lo], FP4_TABLE[hi]], dim=-1).flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    af = a.float() if a_s is None else (a.float().unflatten(-1, (-1, act_block_size)) * a_s.float().unsqueeze(-1)).flatten(-2)
    w = _decode_fp4_bytes(b)                                           # [N, K]
    n, k = w.shape
    w = (w.unflatten(-1, (-1, 32)) * b_s.float().unsqueeze(-1)).flatten(-2)
    return (af @ w.T).to(torch.get_default_dtype())


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """The torch statements quoted in the kernel's own comments (kernel.py:440-458)."""
    b, s, _ = mixes.shape
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).view(b, s, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """kernel.py:sparse_attn_kernel, literally: scores_max starts at the finite -1e30, the sink only joins sum_exp."""
    b, m, h, d = q.shape
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp_min(0).long()
    g = torch.stack([kv[i][idx[i]] for i in range(b)])                         # [b, m, k, d]
    logits = torch.einsum("bmhd,bmkd->bmhk", q.float(), g.float()) * softmax_scale
    logits = logits.masked_fill(~valid.unsqueeze(2), -math.inf)
    mx = logits.amax(dim=-1).clamp_min(-1e30)
    p = torch.exp(logits - mx.unsqueeze(-1))
    sum_exp = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", p, g.float()) / sum_exp.unsqueeze(-1)
    return o.to(q.dtype)


def install():
    """Register the stubs as `kernel` (+ the two vision modules model.py imports, which are not vendored)."""
    k = types.ModuleType("kernel")
    for name in ("act_quant", "fp4_act_quant", "fp8_gemm", "fp4_gemm", "hc_split_sinkhorn", "sparse_attn"):
        setattr(k, name, globals()[name])
    sys.modules["kernel"] = k
    ip = types.ModuleType("image_processor")
    ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START, ip.TEXT = 0, 1, 2, 3, -1
    sys.modules["image_processor"] = ip
    vi = types.ModuleType("vision")
    vi.Aligner = vi.ViT = type("Unused", (), {})
    sys.modules["vision"] = vi
