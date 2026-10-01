#!/usr/bin/env python3
"""gen_golden.py - DS1-D: golden data for the C++ tests of mHC and Engram (src/ds41/cuda/mhc_emu_test.cpp, engram_emu_test.cpp).

    python3 src/ds41/engram/golden/gen_golden.py --out <dir> [--seed 1]

Everything is produced by the NUMPY ORACLE (ref/ds41: mhc.py, engram.py, quant.py), never by a re-implementation:

  * mHC:     `hc_mixes` / `sinkhorn_split` / `hc_pre` / `hc_post`, fed random streams and random mHC parameters, at MiniGeom's hidden size (256) and
             at RealGeom's (5120); each case is run twice: in FLOAT64 (the reference value) and in FLOAT32 (what the oracle computes when fed the
             engine's dtype; its distance from the FLOAT64 run is the FP32 noise floor the C++ tolerances are derived from).  A chain of blocks in
             `model.py: Model.block` order (the one-block `pre` lag, the final head fold) with a toy elementwise sub-layer.
  * hasher:  `NgramHasher` over token streams, with the constants of the MINI GGUF (`tools/ds41/make_mini_gguf.py`, read back through
             `constants_from_gguf_metadata`) and with the constants of the REAL mxxm-t GGUF (the saved shard header + the extracted token map).
             Prefill (one call), one token at a time and chunked calls are asserted equal here; the C++ replays must match EXACTLY.
  * Engram:  `quant.dequant_mxfp4` (rows of both geometries, all E8M0 scales that matter), `engram_layer` (its own code, with `linear` replaced by the
             identity so that a given kv can be fed in), and `engram_layer` end to end on the mini GGUF's real tables / Q8_0 wkv in QuantConfig.int8().

Files are little-endian .npy, C order, under <out>/<case>/ ; a case is complete only when <out>/DONE exists (written last).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "ds41"))

from ref.ds41 import engram as eg  # noqa: E402
from ref.ds41 import mhc, quant  # noqa: E402
from ref.ds41.config import Config  # noqa: E402

GEOMS = {"mini": dict(hidden=256, e_heads=2, e_head_dim=64), "real": dict(hidden=5120, e_heads=8, e_head_dim=256)}
HC, ITERS, HC_EPS, NORM_EPS = 4, 20, 1e-6, 1e-20


def save(out: pathlib.Path, rel: str, arr) -> None:
    p = out / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(arr)
    if not arr.dtype.isnative or arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    np.save(p, arr, allow_pickle=False)


# =============================================================================================================
# mHC
# =============================================================================================================

def mhc_params(rng, K, regime):
    """hc_fn [24, K], scale [3], base [24] (float32).  Regime a: the mini GGUF's mild values (scale 0.1, base ~ 0.1) with mixes ~ N(0, 1); regime b: stress -
    large scale and base, mixes ~ N(0, 16): the sigmoids and softmaxes saturate and the Sinkhorn iteration starts far from uniform."""
    if regime == "a":
        fn = rng.standard_normal((24, K)) / np.sqrt(K)
        scale = 0.1 + 0.01 * rng.standard_normal(3)
        base = 0.1 * rng.standard_normal(24)
    else:
        fn = 4.0 * rng.standard_normal((24, K)) / np.sqrt(K)
        scale = np.array([1.5, 2.0, 3.0]) + 0.1 * rng.standard_normal(3)
        base = rng.standard_normal(24)
    return fn.astype(np.float32), scale.astype(np.float32), base.astype(np.float32)


def mhc_stream(rng, T, H):
    """T streams [T, 4, H]: plain normal, a large scale, a tiny scale, heavy-tailed, an offset one, ... (cycled)."""
    x = np.empty((T, HC, H), np.float32)
    for t in range(T):
        kind = t % 5
        z = rng.standard_normal((HC, H))
        if kind == 0:
            v = z
        elif kind == 1:
            v = 7.0 * z
        elif kind == 2:
            v = 0.02 * z
        elif kind == 3:
            v = rng.standard_t(3, size=(HC, H))
        else:
            v = z + 3.0 + np.arange(HC)[:, None]
        x[t] = v
    return x


def oracle_mixes(x, fn, scale, base, dt):
    """The `mixes` line of mhc.hc_mixes (it is not returned by the oracle): checked against hc_mixes below."""
    xd = x.astype(dt)
    xf = xd.reshape(xd.shape[0], -1)
    rs = 1.0 / np.sqrt(np.mean(np.square(xf), axis=-1, keepdims=True) + NORM_EPS)
    return (xf @ fn.astype(dt).T) * rs


def pack_coef(pre, post, comb):
    n = pre.shape[0]
    return np.concatenate([pre, post, comb.reshape(n, 16)], axis=1)


def mhc_oracle(x, fn, scale, base, dt):
    kw = dict(norm_eps=NORM_EPS, hc=HC, iters=ITERS, eps=HC_EPS)
    pre, post, comb = mhc.hc_mixes(x.astype(dt), fn.astype(dt), scale.astype(dt), base.astype(dt), **kw)
    mixes = oracle_mixes(x, fn, scale, base, dt)
    p2, q2, c2 = mhc.sinkhorn_split(mixes, scale.astype(dt), base.astype(dt), HC, ITERS, HC_EPS)
    assert np.array_equal(pre, p2) and np.array_equal(post, q2) and np.array_equal(comb, c2), "mixes line out of sync with mhc.hc_mixes"
    return mixes, pack_coef(pre, post, comb)


def gen_mhc(out, rng, geom):
    H = GEOMS[geom]["hidden"]
    K = HC * H
    for regime, T in (("a", 5), ("b", 5)):
        d = f"mhc_{geom}/{regime}"
        x = mhc_stream(rng, T, H)
        fn, scale, base = mhc_params(rng, K, regime)
        f = rng.standard_normal((T, H)).astype(np.float32) * np.float32(2.0)
        mixes64, coef64 = mhc_oracle(x, fn, scale, base, np.float64)
        mixes32, coef32o = mhc_oracle(x, fn, scale, base, np.float32)
        coef32 = coef64.astype(np.float32)                    # the record the C++ hc_pre / hc_post tests are fed
        c = coef32.astype(np.float64)
        y_pre64 = mhc.hc_pre(x.astype(np.float64), c[:, 0:4])
        post64 = mhc.hc_post(f.astype(np.float64), x.astype(np.float64), c[:, 4:8], c[:, 8:24].reshape(T, 4, 4))
        c32 = coef32
        y_pre32 = mhc.hc_pre(x, c32[:, 0:4])
        post32 = mhc.hc_post(f, x, c32[:, 4:8], c32[:, 8:24].reshape(T, 4, 4))
        for name, arr in dict(x=x, fn=fn, scale=scale, base=base, f=f, mixes64=mixes64, coef64=coef64, mixes32o=mixes32, coef32o=coef32o,
                              coef32=coef32, y_pre64=y_pre64, post64=post64, y_pre32o=y_pre32, post32o=post32).items():
            save(out, f"{d}/{name}.npy", arr)

    # the split alone, on mixes of every magnitude: saturated sigmoids / softmaxes (|m * scale + base| up to 400: exp underflow, sigmoid overflow
    # guard), zeros, mixed signs, tiny values
    d = f"split_{geom}"
    r = np.random.default_rng(1234)
    rows = [r.standard_normal(24), 30 * r.standard_normal(24), 400 * r.standard_normal(24), np.zeros(24), np.full(24, 5.0), -np.full(24, 5.0),
            1e-4 * r.standard_normal(24), np.where(np.arange(24) % 2 == 0, 80.0, -80.0)]
    rows += [r.standard_normal(24) * s for s in (0.1, 1, 3, 10, 50, 100, 200, 1000)]
    mixes = np.stack(rows).astype(np.float32)
    scale = np.array([0.9, 1.1, 1.3], np.float32)
    base = (0.5 * r.standard_normal(24)).astype(np.float32)
    kw = dict(hc=HC, iters=ITERS, eps=HC_EPS)
    p, q, cmb = mhc.sinkhorn_split(mixes.astype(np.float64), scale.astype(np.float64), base.astype(np.float64), **kw)
    p3, q3, c3 = mhc.sinkhorn_split(mixes, scale, base, **kw)
    save(out, f"{d}/mixes.npy", mixes)
    save(out, f"{d}/scale.npy", scale)
    save(out, f"{d}/base.npy", base)
    save(out, f"{d}/coef64.npy", pack_coef(p, q, cmb))
    save(out, f"{d}/coef32o.npy", pack_coef(p3, q3, c3))


def toy_attn(y):          # exact in binary floating point: the oracle's float64 chain and the engine's float32 chain apply the same function
    return 0.5 * y


def toy_ffn(y):
    return 0.25 * y + 0.5


def gen_chain(out, rng, geom, L, T):
    """model.py: Model.block / Model.forward order of the mHC calls, with elementwise toy sub-layers: pre = [1, 0, 0, 0] at block 0, attention
    collapses with the previous block's FFN `pre`, FFN collapses with the attention's own `pre`, the final fold uses the last FFN `pre`."""
    H = GEOMS[geom]["hidden"]
    K = HC * H
    d = f"chain_{geom}"
    emb = rng.standard_normal((T, H)).astype(np.float32)
    x0 = np.repeat(emb[:, None, :], HC, axis=1)                   # model.py: np.repeat(h[:, None, :], hc_mult, axis=1)
    fns, scales, bases = [], [], []
    for _ in range(2 * L):                                          # order: (attn, ffn) of block 0, (attn, ffn) of block 1, ...
        fn, scale, base = mhc_params(rng, K, "a" if len(fns) % 2 == 0 else "b")
        fns.append(fn), scales.append(scale), bases.append(base)

    def run(dt):
        kw = dict(norm_eps=NORM_EPS, hc=HC, iters=ITERS, eps=HC_EPS)
        x = x0.astype(dt)
        pre_mix = np.zeros((T, HC), dt)
        pre_mix[:, 0] = 1.0                                       # make_identity_pre_mix
        for l in range(L):
            a = 2 * l
            a_pre, a_post, a_comb = mhc.hc_mixes(x, fns[a].astype(dt), scales[a].astype(dt), bases[a].astype(dt), **kw)
            y = toy_attn(mhc.hc_pre(x, pre_mix))
            x = mhc.hc_post(y, x, a_post, a_comb).astype(dt)
            f_pre, f_post, f_comb = mhc.hc_mixes(x, fns[a + 1].astype(dt), scales[a + 1].astype(dt), bases[a + 1].astype(dt), **kw)
            y = toy_ffn(mhc.hc_pre(x, a_pre))                     # uses the ATTENTION's pre
            x = mhc.hc_post(y, x, f_post, f_comb).astype(dt)
            pre_mix = f_pre                                       # consumed by the next block's attention
        return x, mhc.hc_pre(x, pre_mix)

    x64, h64 = run(np.float64)
    x32, h32 = run(np.float32)
    save(out, f"{d}/x0.npy", x0)
    save(out, f"{d}/fn.npy", np.stack(fns))
    save(out, f"{d}/scale.npy", np.stack(scales))
    save(out, f"{d}/base.npy", np.stack(bases))
    save(out, f"{d}/stream64.npy", x64)
    save(out, f"{d}/final64.npy", h64)
    save(out, f"{d}/stream32o.npy", x32)
    save(out, f"{d}/final32o.npy", h32)
    save(out, f"{d}/params.npy", np.array([L, T], np.int64))


# =============================================================================================================
# Engram: hasher
# =============================================================================================================

def hasher_streams(vocab, pad, rng):
    s0 = rng.integers(0, vocab, 96)
    s1 = np.array([5, vocab - 1])
    s2 = np.concatenate([[0, vocab - 1, pad, pad, pad], np.full(10, 7), np.tile([3, 11], 8), [vocab - 1] * 6, rng.integers(0, vocab, 8)])
    s3 = rng.integers(0, vocab, 300)
    return {"s0": s0, "s1": s1, "s2": s2, "s3": s3}


def hash_with_oracle(cfg, consts, tokens):
    """-> rows [n, L, cols] of the whole stream; asserts that prefill, token-by-token and chunked calls agree (the cache carries the look-back)."""
    def make():
        return eg.NgramHasher(cfg, consts["token_map"], consts["layout"], multipliers=consts["multipliers"], max_seq_len=len(tokens) + 8)
    a = make()(np.asarray(tokens), 0)
    b = make()
    one = np.concatenate([b(np.asarray(tokens[i:i + 1]), i) for i in range(len(tokens))])
    c = make()
    chunks, i = [], 0
    for step in (1, 2, 3, 5, 7, 11, 13):
        while i < len(tokens):
            chunks.append(c(np.asarray(tokens[i:i + step]), i))
            i += step
            break
    while i < len(tokens):
        chunks.append(c(np.asarray(tokens[i:i + 9]), i))
        i += 9
    chunked = np.concatenate(chunks)
    assert np.array_equal(a, one) and np.array_equal(a, chunked), "oracle NgramHasher: prefill / decode / chunked disagree"
    return a


def dump_hasher(out, name, cfg, consts, meta_pad, rng):
    d = f"hasher_{name}"
    lay = consts["layout"]
    tm = np.asarray(consts["token_map"], np.int64)
    save(out, f"{d}/layer_ids.npy", np.array(lay.layer_ids, np.int32))
    save(out, f"{d}/params.npy", np.array([lay.n_heads, lay.max_ngram_size, meta_pad, cfg.engram_compressed_vocab_size, len(tm)], np.int64))
    save(out, f"{d}/primes.npy", lay.flat_primes.reshape(-1))
    save(out, f"{d}/offsets.npy", lay.offsets.reshape(-1))
    save(out, f"{d}/multipliers.npy", np.asarray(consts["multipliers"], np.int64).reshape(-1))
    save(out, f"{d}/token_map.npy", tm.astype(np.int32))
    save(out, f"{d}/num_embeddings.npy", np.array(lay.num_embeddings, np.int64))
    for k, toks in hasher_streams(len(tm), meta_pad, rng).items():
        rows = hash_with_oracle(cfg, consts, toks)
        for li, n in enumerate(lay.num_embeddings):
            assert rows[:, li, :].min() >= 0 and rows[:, li, :].max() < n, "row outside its table"
        save(out, f"{d}/tokens_{k}.npy", np.asarray(toks, np.int32))
        save(out, f"{d}/rows_{k}.npy", rows.astype(np.int64))


# =============================================================================================================
# Engram: dequantisation, combine, whole layer
# =============================================================================================================

E_SCALES = np.array([0, 1, 2, 3, 100, 126, 127, 128, 129, 200, 253, 254, 255], np.uint8)


def gen_dequant(out, rng, geom):
    hd = GEOMS[geom]["e_head_dim"]
    nb = hd // 32
    rows = 96
    raw = rng.integers(0, 256, (rows, nb, 17), dtype=np.uint8)
    # scale bytes: half from the special list (denormals at 0 / 1, the finite 2^127 at 255), half the realistic range
    e = np.where(rng.random((rows, nb)) < 0.5, rng.choice(E_SCALES, (rows, nb)), rng.integers(110, 130, (rows, nb))).astype(np.uint8)
    raw[..., 0] = e
    raw[0, :, 1:] = 0x88                                         # code 8 (= +0 in ggml's table), both nibbles
    raw[1, :, 1:] = 0xFF                                         # code 15 = -12
    raw[2, :, 1:] = 0x77                                         # code 7 = +12
    flat = raw.reshape(rows, nb * 17)
    deq = quant.dequant_mxfp4(flat)
    assert deq.shape == (rows, hd) and deq.dtype == np.float32
    save(out, f"dequant_{geom}/rows.npy", flat)
    save(out, f"dequant_{geom}/out.npy", deq)


def bf16_pair(rng, shape, centre=1.0, spread=0.3):
    v = (centre + spread * rng.standard_normal(shape)).astype(np.float32)
    return quant.f32_to_bf16_bits(v)


def oracle_combine(x, kv, qb, kb, dt):
    """engram.engram_layer with `linear` replaced by the identity: rows = kv, so the oracle's own code runs on a given kv."""
    s = x.shape[0]
    qw = quant.bf16_to_f32(qb).astype(dt)
    kw = quant.bf16_to_f32(kb).astype(dt)
    with mock.patch.object(eg, "linear", lambda a, w, **kws: a):
        return eg.engram_layer(x.astype(dt), kv.astype(dt).reshape(s, 1, -1), wkv=None, q_weight=qw, k_weight=kw, eps=NORM_EPS)


def gen_combine(out, rng, geom):
    H = GEOMS[geom]["hidden"]
    for case, T in (("a", 4), ("b", 4)):
        d = f"combine_{geom}/{case}"
        if case == "a":
            x = rng.standard_normal((T, HC, H)).astype(np.float32)
            kv = rng.standard_normal((T, (HC + 1) * H)).astype(np.float32)
            qb, kb = bf16_pair(rng, (HC, H)), bf16_pair(rng, (HC, H))
        else:                                                    # wild scales per (token, copy); a correlated key so that |dot| is large; tiny / huge values
            x = rng.standard_normal((T, HC, H)).astype(np.float32) * np.array([1e-3, 1.0, 30.0, 1e3], np.float32)[None, :, None]
            kv = rng.standard_normal((T, (HC + 1) * H)).astype(np.float32) * np.float32(5.0)
            kv[:, :H] = np.float32(0.8) * x[:, 0, :] + np.float32(0.2) * kv[:, :H]      # key copy 0 aligned with h copy 0 (a large positive dot)
            kv[:, H:2 * H] = -x[:, 1, :] * np.float32(3.0)                              # key copy 1 anti-aligned with h copy 1 (a large negative dot)
            qb, kb = bf16_pair(rng, (HC, H), 0.5, 1.0), bf16_pair(rng, (HC, H), 1.0, 2.0)
        with np.errstate(all="ignore"):
            out64 = oracle_combine(x, kv, qb, kb, np.float64)
            out32 = oracle_combine(x, kv, qb, kb, np.float32)
        for name, arr in dict(x=x, kv=kv, q_bf16=qb, k_bf16=kb, out64=out64, out32o=out32).items():
            save(out, f"{d}/{name}.npy", arr)


def gen_layer_mini(out, tmp):
    """The mini GGUF's own Engram layers end to end through the oracle: hasher -> rows (dequantised table rows) -> wkv (Q8_0, int8 activations) -> combine."""
    import make_mini_gguf as mm
    from ref.ds41.weights import load_from_gguf
    res = mm.build_mini(tmp, mm.MiniConfig(n_shards=1))
    w, cfg = load_from_gguf(res["paths"])
    consts = eg.constants_from_gguf_metadata(w.metadata)
    S = 12
    rng = np.random.default_rng(77)
    toks = rng.integers(0, cfg.vocab_size, S)
    toks[3] = cfg.engram_pad_id                                  # a pad token inside the stream
    hasher = eg.NgramHasher(cfg, consts["token_map"], consts["layout"], multipliers=consts["multipliers"], max_seq_len=64)
    hashes = hasher(toks, 0)                                      # [S, L, cols]
    # the constants exactly as the C++ hasher takes them from the metadata
    dump_hasher(out, "mini", cfg, consts, cfg.engram_pad_id, np.random.default_rng(5))
    save(out, "layer_mini/tokens.npy", toks.astype(np.int32))
    save(out, "layer_mini/rows_idx.npy", hashes.astype(np.int64))
    save(out, "layer_mini/num_layers.npy", np.array([len(cfg.engram_layer_ids)], np.int64))
    for j, L in enumerate(cfg.engram_layer_ids):
        d = f"layer_mini/{j}"
        tbl = w.tensor_info(f"layers.{L}.engram.embed.weight")
        wkv_t = w.tensor_info(f"layers.{L}.engram.wkv.weight")
        qt = w.tensor_info(f"layers.{L}.engram.q_weight")
        kt = w.tensor_info(f"layers.{L}.engram.k_weight")
        x = (rng.standard_normal((S, 4, cfg.dim)) * np.array([1.0, 2.0, 0.5, 4.0])[None, :, None]).astype(np.float32)
        rows = np.asarray(w.engram_rows(L, hashes[:, j, :]), np.float32)          # [S, cols, head_dim] (oracle: dequantised table rows)
        wkv = w.get(f"layers.{L}.engram.wkv.weight")
        qw, kw = w.get(f"layers.{L}.engram.q_weight"), w.get(f"layers.{L}.engram.k_weight")
        outs = {}
        for tag, qc in (("int8", quant.QuantConfig.int8()), ("exact", quant.QuantConfig.exact())):
            outs[tag] = eg.engram_layer(x, rows, wkv=wkv, q_weight=qw, k_weight=kw, eps=cfg.norm_eps, quant=qc)
        kv_int8 = eg.linear(rows.reshape(S, -1), wkv, act_quant=True, quant=quant.QuantConfig.int8())
        save(out, f"{d}/table.npy", tbl.raw(0, None))
        save(out, f"{d}/wkv_q8.npy", wkv_t.raw(0, None))
        save(out, f"{d}/q_bf16.npy", np.ascontiguousarray(qt.raw(0, None)).view(np.uint16))
        save(out, f"{d}/k_bf16.npy", np.ascontiguousarray(kt.raw(0, None)).view(np.uint16))
        save(out, f"{d}/x.npy", x)
        save(out, f"{d}/rows_f32.npy", rows)
        save(out, f"{d}/kv_int8.npy", np.asarray(kv_int8, np.float32))
        save(out, f"{d}/out_int8.npy", np.asarray(outs["int8"], np.float32))
        save(out, f"{d}/out_exact.npy", np.asarray(outs["exact"], np.float32))
        save(out, f"{d}/layer_id.npy", np.array([L], np.int64))


def real_constants():
    from ref.ds41 import selfcheck as sc
    kv = sc.load_gguf_metadata()
    kv = dict(kv)
    tm = sc.load_token_map_fixture()
    kv["deepseek41.engram.token_map"] = [int(v) for v in tm]
    consts = eg.constants_from_gguf_metadata(kv)
    cfg = Config.from_official_json(sc.REF_DIR / "inference" / "config.json")
    cfg.max_seq_len = 1024
    assert tuple(consts["layout"].num_embeddings) == (384006168, 384016682), "real constants are not the released model's"
    assert cfg.engram_pad_id == kv["deepseek41.engram.pad_token_id"]
    return cfg, consts


def gen_layer_real(out, rng, T=2, R=3000):
    """RealGeom shapes with random data: a synthetic table of R rows (the real hash rows are reduced modulo R), the real hasher; wkv's output is injected
    (a 167 MB Q8_0 matrix is not shipped) so the table -> dequantised rows -> combine path is checked at the real shapes."""
    cfg, consts = real_constants()
    toks = np.random.default_rng(9).integers(0, 129280, T)
    hasher = eg.NgramHasher(cfg, consts["token_map"], consts["layout"], multipliers=consts["multipliers"], max_seq_len=64)
    hashes = hasher(toks, 0)[:, 0, :]                            # [T, 24] rows of the first Engram layer
    idx = (hashes % R).astype(np.int64)
    tbl = rng.integers(0, 256, (R, 8, 17), dtype=np.uint8)
    tbl[..., 0] = rng.integers(112, 128, (R, 8))
    flat = tbl.reshape(R, 136)
    rows = quant.dequant_mxfp4(flat[idx.reshape(-1)]).reshape(T, 24, 256)
    H = 5120
    x = rng.standard_normal((T, HC, H)).astype(np.float32)
    kv = (rng.standard_normal((T, (HC + 1) * H)) * 2.0).astype(np.float32)
    qb, kb = bf16_pair(rng, (HC, H)), bf16_pair(rng, (HC, H))
    with np.errstate(all="ignore"):
        out64 = oracle_combine(x, kv, qb, kb, np.float64)
    d = "layer_real"
    for name, arr in dict(table=flat, idx=idx, rows_f32=rows, x=x, kv=kv, q_bf16=qb, k_bf16=kb, out64=out64).items():
        save(out, f"{d}/{name}.npy", arr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "DONE").unlink(missing_ok=True)
    rng = np.random.default_rng(args.seed)
    for geom in GEOMS:
        gen_mhc(out, rng, geom)
        gen_dequant(out, rng, geom)
        gen_combine(out, rng, geom)
    gen_chain(out, rng, "mini", L=4, T=3)
    gen_chain(out, rng, "real", L=2, T=2)
    with tempfile.TemporaryDirectory(prefix="ds1d-mini-") as tmp:
        gen_layer_mini(out, pathlib.Path(tmp))
    cfg, consts = real_constants()
    dump_hasher(out, "real", cfg, consts, cfg.engram_pad_id, np.random.default_rng(6))
    gen_layer_real(out, rng)
    (out / "DONE").write_text("ok\n")
    print(f"gen_golden: wrote {sum(1 for _ in out.rglob('*.npy'))} arrays to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
