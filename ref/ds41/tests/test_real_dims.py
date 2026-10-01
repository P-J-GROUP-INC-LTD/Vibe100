"""Real dimensions (dim 5120, 64 heads x 512, rope 64 + real YaRN, 8 output groups, 32x128 indexer, 384 experts), random
weights, ONE layer at a time: the oracle's attention layer vs the official `Attention` module for each layer mode, and the
router / a routed expert / the shared expert at full width.  Complements the tiny-model test, which binds the selections
(small window / top-k) but not the real shapes and YaRN parameters."""
import numpy as np
import pytest
import torch

from ref.ds41 import attention as A
from ref.ds41 import moe as M
from ref.ds41.config import Config, Mode, layer_modes
from ref.ds41.quant import QuantConfig
from ref.ds41.rope import layer_rope_table
from ref.ds41.tests import kernel_stub as K
from ref.ds41.tests import official as O

EXACT = QuantConfig.exact()
LAYERS = (0, 2, 3, 20, 24)                       # SWA, FULL r2, REUSE (after 2), FULL r1 + candidate source, REINDEX
N_PRE, N_DEC = 9, 2


def real_cfg(max_seq_len=64):
    cfg = Config.from_official_json(O.REF_DIR / "config.json")
    cfg.max_seq_len = max_seq_len
    return cfg


def _attention_weights(att) -> dict:
    t = {}
    for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
        t[f"{name}.weight"] = O.dequant_linear(getattr(att, name))
    for name in ("q_norm", "kv_norm"):
        t[f"{name}.weight"] = getattr(att, name).weight.detach().numpy().astype(np.float64)
    t["attn_sink"] = att.attn_sink.detach().numpy().astype(np.float64)
    if att.compressor is not None:
        c = att.compressor
        t["compressor.wkv.weight"] = O.dequant_linear(c.wkv)
        if c.compress_ratio > 1:
            t["compressor.wgate.weight"] = O.dequant_linear(c.wgate)
        t["compressor.norm.weight"] = c.norm.weight.detach().numpy().astype(np.float64)
    if att.indexer is not None:
        ix = att.indexer
        t["indexer.wq_b.weight"] = O.dequant_linear(ix.wq_b)
        t["indexer.weights_proj.weight"] = O.dequant_linear(ix.weights_proj)
        if ix.owns_k:
            t["indexer.wk.weight"] = O.dequant_linear(ix.wk)
            t["indexer.k_norm.weight"] = ix.k_norm.weight.detach().numpy().astype(np.float64)
    return t


def _randomize_attention(att, seed):
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *s: torch.randn(*s, generator=g)
    mods = [att.wq_a, att.wq_b, att.wkv, att.wo_a, att.wo_b]
    if att.compressor is not None:
        mods += [att.compressor.wkv] + ([att.compressor.wgate] if att.compressor.compress_ratio > 1 else [])
    if att.indexer is not None:
        mods += [att.indexer.wq_b, att.indexer.weights_proj] + ([att.indexer.wk] if att.indexer.owns_k else [])
    for m in mods:
        O.fill_linear(m, rnd, 1.5 / np.sqrt(m.in_features))
    att.attn_sink.data = rnd(*att.attn_sink.shape) * 0.5
    for n in (att.q_norm, att.kv_norm) + ((att.compressor.norm,) if att.compressor is not None else ()) + (
            (att.indexer.k_norm,) if att.indexer is not None and att.indexer.owns_k else ()):
        n.weight.data = 1 + 0.2 * rnd(*n.weight.shape)
    for p in att.parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.float()


@pytest.mark.slow
def test_attention_layers_at_real_dimensions(official_mod):
    mm, _ = official_mod
    cfg = real_cfg()
    args = O.model_args(cfg)
    modes = layer_modes(cfg)
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((N_PRE + N_DEC, cfg.dim)).astype(np.float32)       # post-attn_norm hidden states
    caches = [A.LayerCache.new(cfg, l, np.float32) for l in range(cfg.n_layers)]
    shared = A.SharedState()
    K.Flags.set(EXACT)
    mods, ws, ropes = {}, {}, {}
    for l in LAYERS:
        mods[l] = mm.Attention(l, args)
        _randomize_attention(mods[l], seed=100 + l)
        ws[l] = {k: v.astype(np.float32) for k, v in _attention_weights(mods[l]).items()}   # fp8 values: exact in fp32
        ropes[l] = layer_rope_table(cfg, l, cfg.max_seq_len, np.float64)
    worst = {}
    # every forward pass runs the layers in order (shared state flows 2 -> 3 and 20 -> 24), as in the model
    for start, n in [(0, N_PRE)] + [(N_PRE + i, 1) for i in range(N_DEC)]:
        x = xs[start:start + n]
        for l in LAYERS:
            with torch.no_grad():
                ref = mods[l](torch.from_numpy(x)[None], start)[0].numpy()
            got = A.attention_layer(cfg, l, modes[l], x, start, ropes[l], ws[l].__getitem__, lc=caches[l],
                                    caches=caches, shared=shared, quant=EXACT)
            err = np.abs(got - ref).max() / np.abs(ref).max()
            worst[l] = max(worst.get(l, 0), err)
            assert err < 1e-4, (l, modes[l], start, err)
    print("\n[real dims] max rel err per layer (SWA 0, FULL-r2 2, REUSE 3, FULL-r1 20, REINDEX 24):",
          {l: f"{e:.1e}" for l, e in worst.items()})
    assert [modes[l] for l in LAYERS] == [Mode.SWA, Mode.FULL, Mode.REUSE, Mode.FULL, Mode.REINDEX]


def test_router_and_experts_at_real_dimensions(official_mod):
    mm, _ = official_mod
    cfg = real_cfg()
    args = O.model_args(cfg)
    g = torch.Generator().manual_seed(7)
    rnd = lambda *s: torch.randn(*s, generator=g)
    gate = mm.Gate(5, args)
    gw = (rnd(384, 5120) / np.sqrt(5120) * 3).numpy().astype(np.float32)
    gb = (rnd(384) * 0.1).numpy().astype(np.float32)
    gate.weight.data, gate.bias.data = torch.from_numpy(gw), torch.from_numpy(gb)
    x = (rnd(64, 5120) * 2).numpy().astype(np.float32)
    K.Flags.set(EXACT)
    w_ref, i_ref = gate(torch.from_numpy(x))
    out = M.router(x.astype(np.float64), gw, gb, topk=6, route_scale=1.5)
    solid = out.margin > 1e-5
    assert solid.mean() > 0.9 and np.array_equal(out.indices[solid], i_ref.numpy()[solid])
    assert np.abs(out.weights[solid] - w_ref.numpy()[solid]).max() < 1e-5
    assert np.allclose(out.weights.sum(-1), 1.5, atol=1e-5) and out.indices.shape == (64, 6)
    # a routed expert (fp4, 2304 x 5120) and the shared expert (fp8): clamps at 10
    for dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        ex = mm.Expert(5120, 2304, dtype=dtype, swiglu_limit=10.0)
        for lin in (ex.w1, ex.w2, ex.w3):
            O.fill_linear(lin, rnd, 4.0 / np.sqrt(lin.in_features))
        xe = x[:5]
        wts = np.array([0.4, 0.2, 0.1, 0.9, 0.6], np.float32)
        ref = ex(torch.from_numpy(xe), torch.from_numpy(wts)[:, None]).numpy()
        got = M.expert(xe.astype(np.float64), O.dequant_linear(ex.w1), O.dequant_linear(ex.w3), O.dequant_linear(ex.w2),
                       swiglu_limit=10.0, weights=wts.astype(np.float64), quant=EXACT)
        assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-5
