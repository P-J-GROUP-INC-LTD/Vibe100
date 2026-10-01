"""Every oracle op against the corresponding OFFICIAL torch function/module on small random inputs.

The official code runs on CPU with kernel.py replaced by `kernel_stub` (see tests/official.py).  `Flags` selects
whether the stubs fake-quantise: ops are compared in 'exact' mode (tight, 1e-5 relative) and, where the op contains an
fp8/fp4 GEMM, also in 'reference' mode (identical fake-quant sites; a rare quantisation-boundary flip caused by
1e-7 float noise is possible, so those comparisons use fixed seeds and a looser bound - see README).
"""
import types

import numpy as np
import pytest
import torch

from ref.ds41 import attention as A
from ref.ds41 import engram as E
from ref.ds41 import mhc as H
from ref.ds41 import moe as M
from ref.ds41 import ops as OPS
from ref.ds41 import rope as R
from ref.ds41.config import Config
from ref.ds41.quant import QuantConfig
from ref.ds41.tests import kernel_stub as K
from ref.ds41.tests import official as O

EXACT, REFERENCE = QuantConfig.exact(), QuantConfig.reference()


ERRS: dict = {}          # test id -> max relative error seen (printed at the end of the module with -s)
_CURRENT = [""]


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    e = float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))
    ERRS[_CURRENT[0]] = max(ERRS.get(_CURRENT[0], 0.0), e)
    return e


@pytest.fixture(autouse=True)
def _flags(request):
    _CURRENT[0] = request.node.name
    K.Flags.set(EXACT)
    yield
    K.Flags.set(REFERENCE)


@pytest.fixture(autouse=True, scope="module")
def _report():
    yield
    if ERRS:
        print("\n[op errors vs official, max relative]")
        for k, v in sorted(ERRS.items()):
            print(f"  {k:60s} {v:.2e}")


# ------------------------------------------------------------------------------------------------ RMSNorm
@pytest.mark.parametrize("dim,eps", [(64, 1e-20), (512, 1e-20), (1280, 1e-6)])
def test_rmsnorm(official_mod, dim, eps):
    mm, _ = official_mod
    rng = np.random.default_rng(0)
    n = mm.RMSNorm(dim, eps)
    w = (1 + 0.3 * rng.standard_normal(dim)).astype(np.float32)
    n.weight.data = torch.from_numpy(w)
    x = (rng.standard_normal((9, dim)) * 3).astype(np.float32)
    ref = n(torch.from_numpy(x)).numpy()
    assert rel(OPS.rmsnorm(x.astype(np.float64), w, eps), ref) < 1e-6
    assert rel(OPS.rmsnorm(x, w, eps), ref) < 1e-6                         # float32 path
    # eps = 1e-20 really is negligible but must not produce NaN for an all-zero row
    z = np.zeros((1, dim), np.float32)
    assert np.isfinite(OPS.rmsnorm(z, w, 1e-20)).all() and np.array_equal(OPS.rmsnorm(z, w, 1e-20), n(torch.from_numpy(z)).numpy())


# ------------------------------------------------------------------------------------------------ router / expert / MoE
def _moe_args(**kw):
    cfg = O.tiny_config(**kw)
    return cfg, O.model_args(cfg)


@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "softmax", "sigmoid"])
def test_router_vs_gate(official_mod, score_func):
    mm, _ = official_mod
    rng = np.random.default_rng(1)
    cfg, args = _moe_args()
    args.score_func = score_func
    gate = mm.Gate(3, args)
    gw = (rng.standard_normal(gate.weight.shape) * 1.5).astype(np.float32)
    gb = (rng.standard_normal(gate.bias.shape) * 0.2).astype(np.float32)
    gate.weight.data, gate.bias.data = torch.from_numpy(gw), torch.from_numpy(gb)
    x = (rng.standard_normal((400, cfg.dim)) * 4).astype(np.float32)           # big logits: exercises softplus' x > 20 branch
    w_ref, i_ref = gate(torch.from_numpy(x))
    out = M.router(x.astype(np.float64), gw, gb, topk=cfg.n_activated_experts, route_scale=cfg.route_scale,
                   score_func=score_func)
    solid = out.margin > 1e-5                                                  # near-ties may legitimately flip
    assert solid.mean() > 0.95
    assert np.array_equal(out.indices[solid], i_ref.numpy()[solid])
    assert rel(out.weights[solid], w_ref.numpy()[solid]) < 1e-5
    if score_func == "sqrtsoftplus":
        assert (np.abs(x @ gw.T) > 20).any(), "test should reach softplus' linear branch"
        # weights are normalised (sum to route_scale) before the 1.5 scale
        assert np.allclose(out.weights.sum(-1), cfg.route_scale, atol=1e-5)


@pytest.mark.parametrize("dtype_name", ["fp4", "fp8"])
@pytest.mark.parametrize("quant", [EXACT, REFERENCE], ids=["exact", "reference"])
def test_expert_vs_official(official_mod, dtype_name, quant):
    mm, _ = official_mod
    rng_t = torch.Generator().manual_seed(5)
    rnd = lambda *s: torch.randn(*s, generator=rng_t)
    dim, inter = 64, 96
    dtype = torch.float4_e2m1fn_x2 if dtype_name == "fp4" else torch.float8_e4m3fn
    ex = mm.Expert(dim, inter, dtype=dtype, swiglu_limit=10.0)
    for lin in (ex.w1, ex.w2, ex.w3):
        O.fill_linear(lin, rnd, 1.0 / np.sqrt(lin.in_features) * 3.0)
    x = (rnd(40, dim) * 6).numpy().astype(np.float32)
    wts = (0.5 + torch.rand(40, generator=rng_t)).numpy().astype(np.float32)
    K.Flags.set(quant)
    ref = ex(torch.from_numpy(x), torch.from_numpy(wts)[:, None]).numpy()
    ref_nw = ex(torch.from_numpy(x)).numpy()
    w1, w2, w3 = (O.dequant_linear(m) for m in (ex.w1, ex.w2, ex.w3))
    got = M.expert(x.astype(np.float64), w1, w3, w2, swiglu_limit=10.0, weights=wts.astype(np.float64), quant=quant)
    got_nw = M.expert(x.astype(np.float64), w1, w3, w2, swiglu_limit=10.0, weights=None, quant=quant)
    tol = 1e-5 if quant is EXACT else 2e-2
    assert rel(got, ref) < tol and rel(got_nw, ref_nw) < tol
    # the clamps are active in this test
    g, u = x @ w1.T, x @ w3.T
    assert (g > 10).any() and (u > 10).any() and (u < -10).any()
    if quant is EXACT:
        # the weight multiplies BEFORE w2: y(w) == w2 @ (w * h) -> linear in w only through h, so scaling the weight
        # by 2 must scale the output by exactly 2
        got2 = M.expert(x.astype(np.float64), w1, w3, w2, swiglu_limit=10.0, weights=2 * wts.astype(np.float64))
        assert rel(got2, 2 * got) < 1e-12


@pytest.mark.parametrize("quant", [EXACT, REFERENCE], ids=["exact", "reference"])
def test_moe_vs_official(official_mod, quant):
    mm, _ = official_mod
    cfg, args = _moe_args()
    torch.manual_seed(11)
    layer = mm.MoE(3, args)
    rng_t = torch.Generator().manual_seed(6)
    rnd = lambda *s: torch.randn(*s, generator=rng_t)
    for e in layer.experts:
        for lin in (e.w1, e.w2, e.w3):
            O.fill_linear(lin, rnd, 2.0 / np.sqrt(lin.in_features))
    for lin in (layer.shared_experts.w1, layer.shared_experts.w2, layer.shared_experts.w3):
        O.fill_linear(lin, rnd, 2.0 / np.sqrt(lin.in_features))
    layer.gate.weight.data = (rnd(*layer.gate.weight.shape) / 8).float()
    layer.gate.bias.data = rnd(*layer.gate.bias.shape) * 0.1
    x = (rnd(50, cfg.dim) * 3).numpy().astype(np.float32)
    K.Flags.set(quant)
    ref = layer(torch.from_numpy(x).view(1, 50, -1)).numpy()[0]
    exp = [tuple(O.dequant_linear(m) for m in (e.w1, e.w3, e.w2)) for e in layer.experts]
    r = M.router(x.astype(np.float64), layer.gate.weight.detach().numpy(), layer.gate.bias.detach().numpy(),
                 topk=cfg.n_activated_experts, route_scale=cfg.route_scale)
    assert r.margin.min() > 1e-4, "scenario must be tie-free"
    got = M.moe(x.astype(np.float64), gate_w=layer.gate.weight.detach().numpy().astype(np.float64),
                gate_bias=layer.gate.bias.detach().numpy().astype(np.float64), get_expert=lambda e: exp[e],
                shared=tuple(O.dequant_linear(m) for m in (layer.shared_experts.w1, layer.shared_experts.w3,
                                                            layer.shared_experts.w2)), cfg=cfg, quant=quant)
    assert rel(got, ref) < (1e-5 if quant is EXACT else 3e-2)
    # lazy access: only the selected experts are ever requested
    asked = set()
    M.moe(x.astype(np.float64), gate_w=layer.gate.weight.detach().numpy().astype(np.float64),
          gate_bias=layer.gate.bias.detach().numpy().astype(np.float64),
          get_expert=lambda e: (asked.add(e), exp[e])[1], shared=exp[0], cfg=cfg, quant=EXACT)
    assert asked == set(np.unique(r.indices).tolist())


# ------------------------------------------------------------------------------------------------ RoPE / YaRN
@pytest.mark.parametrize("orig,base", [(0, 10000.0), (65536, 160000.0), (32, 160000.0), (4096, 10000.0)])
def test_rope_table_and_rotation(official_mod, orig, base):
    mm, _ = official_mod
    dim, seq, factor = 64, 512, 16
    ref = mm.precompute_freqs_cis(dim, seq, orig, base, factor, 32, 1)
    for dt, tol in ((np.float32, 5e-5), (np.float64, 2e-4)):                   # fp32 table follows torch (angles carry ~3e-5 rad fp32 rounding at pos 500); fp64 is "truth"
        cos, sin = R.rope_table(dim, seq, orig, base, factor, 32, 1, dt)
        assert np.abs(cos - ref.real.numpy()).max() < tol and np.abs(sin - ref.imag.numpy()).max() < tol, (dt, orig)
    cos, sin = R.rope_table(dim, seq, orig, base, factor, 32, 1, np.float32)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 33, 4, dim)).astype(np.float32)
    for inv in (False, True):
        r = mm.apply_rotary_emb(torch.from_numpy(x.copy()), ref[:33], inv).numpy()
        o = R.apply_rope(x[0], cos[:33], sin[:33], inverse=inv)
        assert np.abs(r[0] - o).max() < 1e-4
    # inverse undoes forward; the nope part is untouched by apply_rope_tail
    y = R.apply_rope(x[0], cos[:33], sin[:33])
    assert np.abs(R.apply_rope(y, cos[:33], sin[:33], inverse=True) - x[0]).max() < 1e-5
    t = R.apply_rope_tail(x[0].reshape(33, 4, dim)[..., :], cos[:33, :16], sin[:33, :16], 32)
    assert np.array_equal(t[..., :32], x[0][..., :32])


def test_yarn_all_three_regimes_present():
    """Real config (dim 64, theta 160000, factor 16, original 65536): high-frequency dims keep their frequency, low
    ones are divided by the factor, the ramp in between is strictly between."""
    cos0, sin0 = R.rope_table(64, 4, 0, 160000.0, 16, 32, 1)
    cos1, sin1 = R.rope_table(64, 4, 65536, 160000.0, 16, 32, 1)
    f0 = np.arctan2(sin0[1], cos0[1])
    f1 = np.arctan2(sin1[1], cos1[1])
    ratio = f1 / f0
    assert np.isclose(ratio[0], 1.0) and np.isclose(ratio[-1], 1 / 16) and ((ratio > 1 / 16 + 1e-9) & (ratio < 1 - 1e-9)).any()


def test_layer_rope_params():
    cfg = Config()
    assert R.layer_rope_params(cfg, 0) == dict(original_seq_len=0, base=10000.0)
    assert R.layer_rope_params(cfg, 1) == dict(original_seq_len=0, base=10000.0)
    assert R.layer_rope_params(cfg, 2) == dict(original_seq_len=65536, base=160000.0)
    assert R.layer_rope_params(cfg, 39) == dict(original_seq_len=65536, base=160000.0)
    assert R.layer_rope_params(cfg, 40) == dict(original_seq_len=0, base=10000.0)        # DSpark block 0


# ------------------------------------------------------------------------------------------------ window / candidates
@pytest.mark.parametrize("seqlen", [1, 5, 8, 9, 20, 64])
def test_window_idxs_prefill(official_mod, seqlen):
    mm, _ = official_mod
    ref = mm.get_window_topk_idxs(8, 1, seqlen, 0).numpy()[0]
    assert np.array_equal(A.window_topk_idxs(8, seqlen, 0), ref)


@pytest.mark.parametrize("start_pos", [1, 3, 7, 8, 9, 15, 16, 23, 100])
def test_window_idxs_decode(official_mod, start_pos):
    mm, _ = official_mod
    ref = mm.get_window_topk_idxs(8, 1, 1, start_pos).numpy()[0]
    assert np.array_equal(A.window_topk_idxs(8, 1, start_pos), ref)


@pytest.mark.parametrize("n,bs,tb", [(40, 8, 3), (37, 8, 100), (16, 4, 2), (33, 2, 5)])
def test_select_candidate_blocks(official_mod, n, bs, tb):
    mm, _ = official_mod
    rng = np.random.default_rng(n)
    s = n                                                                         # ratio-1 prefill: row t reaches t+1
    logits = rng.standard_normal((s, n)).astype(np.float32)
    lens = np.arange(1, s + 1)[:, None]
    logits = np.where(np.arange(n)[None, :] >= lens, -np.inf, logits).astype(np.float32)
    ref = mm.select_candidate_blocks(torch.from_numpy(logits).unsqueeze(0), torch.from_numpy(lens), tb, bs).numpy()[0]
    assert np.array_equal(A.select_candidate_blocks(logits, lens, tb, bs), ref)
    # decode form: one query, integer compress_lens
    for cl in (1, 5, n):
        row = np.where(np.arange(n) >= cl, -np.inf, rng.standard_normal(n)).astype(np.float32)[None]
        ref = mm.select_candidate_blocks(torch.from_numpy(row).unsqueeze(0), cl, tb, bs).numpy()[0]
        assert np.array_equal(A.select_candidate_blocks(row, cl, tb, bs), ref), cl
    # the newest reachable block is always kept (pinned), even when a lone old block outscores it
    row = np.full((1, 16), -np.inf, np.float32)
    row[0, :4] = 100.0
    row[0, 4] = -5.0
    keep = A.select_candidate_blocks(row, 5, 1, 4)
    assert keep[0, 4:8].all() and not keep[0, :4].any()


# ------------------------------------------------------------------------------------------------ sparse attention
def _concat_softmax_attention(q, kv, sink, idxs, scale):
    """Independent formulation: append the sink as an extra logit, softmax, drop its column (no online softmax)."""
    s, h, d = q.shape
    out = np.zeros_like(q)
    for t in range(s):
        sel = [i for i in idxs[t] if i >= 0]
        if not sel:
            continue
        k = kv[sel]                                                  # [m, d]
        logits = (q[t] @ k.T) * scale                                # [h, m]
        full = np.concatenate([logits, sink[:, None]], axis=1)       # [h, m+1]
        p = np.exp(full - full.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        out[t] = p[:, :-1] @ k
    return out


def test_sparse_attn_three_ways(official_mod):
    mm, _ = official_mod
    rng = np.random.default_rng(3)
    s, h, d, n, k = 7, 4, 64, 30, 12
    q = rng.standard_normal((s, h, d)).astype(np.float32)
    kv = rng.standard_normal((n, d)).astype(np.float32)
    sink = rng.standard_normal(h).astype(np.float32)
    idxs = np.stack([np.concatenate([rng.choice(n, size=int(rng.integers(0, k + 1)), replace=False),
                                     -np.ones(k, np.int64)])[:k] for _ in range(s)]).astype(np.int32)
    idxs[2] = -1                                                      # a query with nothing to attend to
    scale = d ** -0.5
    ref = mm.sparse_attn(torch.from_numpy(q)[None], torch.from_numpy(kv)[None], torch.from_numpy(sink),
                         torch.from_numpy(idxs)[None], scale)[0].numpy()
    got = A.sparse_attn(q.astype(np.float64), kv.astype(np.float64), sink.astype(np.float64), idxs.astype(np.int64), scale)
    alt = _concat_softmax_attention(q.astype(np.float64), kv.astype(np.float64), sink.astype(np.float64), idxs, scale)
    assert rel(got, ref) < 1e-6 and rel(got, alt) < 1e-12 and rel(alt, ref) < 1e-6
    assert np.array_equal(got[2], np.zeros_like(got[2]))              # no valid index -> zeros (kernel convention)
    # a large sink swallows the attention: output -> 0
    big = A.sparse_attn(q.astype(np.float64), kv.astype(np.float64), np.full(h, 80.0), idxs.astype(np.int64), scale)
    assert np.abs(big).max() < 1e-6
    # chunking does not change anything
    c = A.sparse_attn(q.astype(np.float64), kv.astype(np.float64), sink.astype(np.float64), idxs.astype(np.int64), scale,
                      max_chunk_elems=k * d)
    assert np.array_equal(c, got)


# ------------------------------------------------------------------------------------------------ compressor
@pytest.mark.parametrize("layer", [2, 6], ids=["ratio2", "ratio1"])
def test_compressor_prefill_and_decode(official_mod, layer):
    mm, _ = official_mod
    cfg, args = _moe_args()
    comp = mm.Compressor(args, layer)
    rng_t = torch.Generator().manual_seed(9)
    rnd = lambda *s: torch.randn(*s, generator=rng_t)
    ratio = cfg.compress_ratios[layer]
    comp.wkv.weight.data = (rnd(cfg.head_dim, cfg.dim) / 8).float()
    if ratio > 1:
        comp.wgate.weight.data = (rnd(cfg.head_dim, cfg.dim) / 4).float()
    comp.norm.weight.data = 1 + 0.2 * rnd(cfg.head_dim)
    wkv = comp.wkv.weight.detach().numpy().astype(np.float64)
    wgate = comp.wgate.weight.detach().numpy().astype(np.float64) if ratio > 1 else None
    nw = comp.norm.weight.detach().numpy().astype(np.float64)
    lc = A.LayerCache.new(cfg, layer, np.float64)
    for plen in (1, 2, 7, 12):                                        # prompts shorter than / odd / even w.r.t. the ratio
        x = (rnd(1, plen + 9, cfg.dim)).numpy().astype(np.float32)
        with torch.no_grad():
            r0 = comp(torch.from_numpy(x[:, :plen]), 0)
        g0 = A.compressor(cfg, ratio, x[0, :plen].astype(np.float64), 0, wkv=wkv, wgate=wgate, norm_w=nw, lc=lc)
        assert (r0 is None) == (g0 is None)
        if r0 is not None:
            assert rel(g0, r0.numpy()[0]) < 1e-5
        for pos in range(plen, plen + 9):                              # then one token per step
            with torch.no_grad():
                r = comp(torch.from_numpy(x[:, pos:pos + 1]), pos)
            g = A.compressor(cfg, ratio, x[0, pos:pos + 1].astype(np.float64), pos, wkv=wkv, wgate=wgate, norm_w=nw, lc=lc)
            assert (r is None) == (g is None), (plen, pos)
            if r is not None:
                assert rel(g, r.numpy()[0]) < 1e-5, (plen, pos)


# ------------------------------------------------------------------------------------------------ mHC
def _block_stub(cfg):
    return types.SimpleNamespace(norm_eps=cfg.norm_eps, hc_mult=cfg.hc_mult, hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                                 hc_eps=cfg.hc_eps)


def test_mhc_vs_block_methods(official_mod):
    mm, _ = official_mod
    cfg = Config.tiny()
    rng = np.random.default_rng(4)
    hc, d, n = cfg.hc_mult, cfg.dim, 6
    x = rng.standard_normal((n, hc, d)).astype(np.float32)
    fn = (rng.standard_normal((24, hc * d)) / np.sqrt(hc * d) * 3).astype(np.float32)
    base = (rng.standard_normal(24) * 0.5).astype(np.float32)
    scale = np.array([0.7, 0.9, 1.3], np.float32)
    stub = _block_stub(cfg)
    pre_r, post_r, comb_r = mm.Block.hc_mixes(stub, torch.from_numpy(x)[None], torch.from_numpy(fn),
                                              torch.from_numpy(scale), torch.from_numpy(base))
    pre, post, comb = H.hc_mixes(x.astype(np.float64), fn, scale, base, norm_eps=cfg.norm_eps, hc=hc,
                                 iters=cfg.hc_sinkhorn_iters, eps=cfg.hc_eps)
    assert rel(pre, pre_r[0].numpy()) < 1e-6 and rel(post, post_r[0].numpy()) < 1e-6 and rel(comb, comb_r[0].numpy()) < 1e-6
    # hc_pre / hc_post
    y_r = mm.Block.hc_pre(stub, torch.from_numpy(x)[None], pre_r)
    assert rel(H.hc_pre(x.astype(np.float64), pre), y_r[0].numpy()) < 1e-6
    sub = rng.standard_normal((n, d)).astype(np.float32)
    p_r = mm.Block.hc_post(stub, torch.from_numpy(sub)[None], torch.from_numpy(x)[None], post_r, comb_r)
    assert rel(H.hc_post(sub.astype(np.float64), x.astype(np.float64), post, comb), p_r[0].numpy()) < 1e-6
    # structure: pre in (eps, 1+eps), post in (0, 2), comb is (nearly) doubly stochastic after 20 iterations
    assert pre.min() > cfg.hc_eps and pre.max() < 1 + 2 * cfg.hc_eps and post.min() > 0 and post.max() < 2
    assert np.abs(comb.sum(-2) - 1).max() < 5 * cfg.hc_eps              # columns normalised last (sum = 1 - O(eps))
    assert np.abs(comb.sum(-1) - 1).max() < 0.1                         # rows: only approximately after 20 iterations


def test_sinkhorn_literal_loops():
    """Second formulation of kernel.py:hc_split_sinkhorn_kernel: explicit scalar loops over (j, k)."""
    rng = np.random.default_rng(5)
    hc, eps, iters = 4, 1e-6, 20
    mixes = rng.standard_normal((3, 24))
    scale, base = np.array([0.5, 1.2, 0.8]), rng.standard_normal(24) * 0.4
    pre, post, comb = H.sinkhorn_split(mixes, scale, base, hc, iters, eps)
    for i in range(3):
        c = [[mixes[i][j * hc + k + 2 * hc] * scale[2] + base[j * hc + k + 2 * hc] for k in range(hc)] for j in range(hc)]
        for j in range(hc):                                        # comb = softmax(row) + eps
            m = max(c[j])
            e = [np.exp(v - m) for v in c[j]]
            c[j] = [v / sum(e) + eps for v in e]
        for k in range(hc):                                        # column normalise
            cs = sum(c[j][k] for j in range(hc))
            for j in range(hc):
                c[j][k] = c[j][k] / (cs + eps)
        for _ in range(iters - 1):
            for j in range(hc):                                    # row
                rs = sum(c[j])
                c[j] = [v / (rs + eps) for v in c[j]]
            for k in range(hc):                                    # column
                cs = sum(c[j][k] for j in range(hc))
                for j in range(hc):
                    c[j][k] = c[j][k] / (cs + eps)
        assert np.abs(np.array(c) - comb[i]).max() < 1e-12
        for j in range(hc):
            assert abs(pre[i][j] - (1 / (1 + np.exp(-(mixes[i][j] * scale[0] + base[j]))) + eps)) < 1e-12
            assert abs(post[i][j] - 2 / (1 + np.exp(-(mixes[i][j + hc] * scale[1] + base[j + hc])))) < 1e-12


# ------------------------------------------------------------------------------------------------ Engram
def test_engram_hash_vs_official(tiny, tiny_cfg):
    model, tok, w, tm = tiny
    h_off = model.engram_hash
    rng = np.random.default_rng(7)
    ids = rng.integers(0, tiny_cfg.vocab_size, 29)
    hasher = E.NgramHasher(tiny_cfg, tm)
    assert np.array_equal(hasher.primes.reshape(2, -1), h_off.primes.numpy().reshape(2, -1))
    assert np.array_equal(hasher.offsets, h_off.offsets.numpy())
    assert np.array_equal(hasher.multipliers, h_off.multipliers.numpy())
    ref0 = h_off(torch.from_numpy(ids[:20])[None], 0)[0].numpy()
    got0 = hasher(ids[:20], 0)
    assert np.array_equal(got0, ref0)
    for pos in range(20, 29):                                           # decode: look-back through the cache
        ref = h_off(torch.from_numpy(ids[pos:pos + 1])[None], pos)[0].numpy()
        assert np.array_equal(hasher(ids[pos:pos + 1], pos), ref), pos
    # every row address stays inside its layer's table
    for j, rows in enumerate(tiny_cfg.engram_num_embeddings):
        assert 0 <= got0[:, j].min() and got0[:, j].max() < rows


@pytest.mark.parametrize("quant", [EXACT, REFERENCE], ids=["exact", "reference"])
def test_engram_layer_vs_official(tiny, tiny_cfg, quant):
    model, tok, w, tm = tiny
    cfg = tiny_cfg
    eng = model.layers[1].engram
    rng = np.random.default_rng(8)
    s = 11
    x = (rng.standard_normal((s, cfg.hc_mult, cfg.dim)) * 2).astype(np.float32)
    ids = rng.integers(0, cfg.vocab_size, s)
    hasher = E.NgramHasher(cfg, tm)
    hashes = hasher(ids, 0)[:, 0, :]                                    # layer index 0 of the Engram layers
    K.Flags.set(quant)
    with torch.no_grad():
        ref = eng(torch.from_numpy(x)[None], torch.from_numpy(hashes)[None])[0].numpy()
    table = w.get("layers.1.engram.embed.weight")
    got = E.engram_layer(x.astype(np.float64), table[hashes].astype(np.float64),
                         wkv=w.get("layers.1.engram.wkv.weight").astype(np.float64),
                         q_weight=w.get("layers.1.engram.q_weight"), k_weight=w.get("layers.1.engram.k_weight"),
                         eps=cfg.norm_eps, quant=quant)
    assert rel(got, ref) < (1e-5 if quant is EXACT else 3e-2)
    # the gate is (0, 1): the update stays within |value| of the stream
    assert np.isfinite(got).all()
