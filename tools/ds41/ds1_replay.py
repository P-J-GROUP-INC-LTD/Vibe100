#!/usr/bin/env python3
"""ds1_replay.py - layer-by-layer replay: run ONLY layer L of the oracle on the engine's own inputs, then compare (DS1.md section 7, real weights).

WHY.  A 40-layer comparison of two float32 implementations accumulates error (and, with int8 / fp8 / fp4 activations in the loop, the odd rounding
flip) from layer to layer; on the 411-GB model the oracle also cannot afford to run the whole network for every token.  Here the oracle is
handed the engine's value of everything a stage reads, computes ONE stage, and that stage alone is compared:

    stage                    oracle computes it from (engine values)                                         compared with the engine's
    embed                    the token id                                                                    embed
    engram_out    (L1,L14)   block_out.{L-1} (or the embedded token), the token ids -> hashed Engram rows    engram_out.L
    attn_in                  the stream entering the block (engram_out / block_out.{L-1}) and pre_mix.{L-1}  attn_in.L
    q, kv_win, latent,       attn_in.L, the layer's cache rows of positions < p (kv_win, latent, index_k),   q.L, kv_win.L, latent.L, index_k.L,
    index_k, topk, attn_out  the compressor's group state (attn_in of the open group), and for REUSE layers   topk.L, attn_out.L
                             the engine's own top-k, for REINDEX layers the candidate pool
    ffn_in                   the stream, attn_out.L (engine) and the attention mHC mixes                      ffn_in.L
    router_idx, router_w     ffn_in.L                                                                        router_idx.L, router_w.L
    ffn_out                  ffn_in.L (the oracle's router and routed + shared experts)                      ffn_out.L
    block_out, pre_mix       ffn_out.L (engine), the stream after attention, the FFN mHC mixes               block_out.L, pre_mix.L
    final_hidden, logits     block_out.{N-1}, pre_mix.{N-1}; the head on final_hidden                        final_hidden, logits    (pseudo-layer `head`)

So an error in the engine's layer 7 q-projection shows up at stage `q` of layer 7 and nowhere else: attn_out of layer 7 is recomputed from the engine's
attn_in, not from its q.  The cache rows the engine dumped ARE the cache state: no earlier position has to be re-run by the oracle (see
DS1_VERIFY.md: what the engine must dump).  When a selection (top-k, routed experts) differs from the engine's, it is reported against the oracle's
margin, and everything downstream is recomputed a second time with the ENGINE's selection (`--no-force` turns that off), so one near-tie does not hide
the numbers of the stages after it.

The orchestration below repeats `Model.block` / `Model.forward` call for call (the oracle's own functions, nothing re-implemented); the test suite
replays the oracle's own trace through it and requires a bit-identical result.
"""
from __future__ import annotations

import contextlib
import dataclasses
import sys
import time
from collections import OrderedDict

import numpy as np

import ds1_compare as C
from ref.ds41 import attention as A
from ref.ds41 import trace_io as TI
from ref.ds41.attention import LayerCache, SharedState
from ref.ds41.config import Mode
from ref.ds41.engram import engram_layer
from ref.ds41.mhc import hc_mixes, hc_post, hc_pre
from ref.ds41.moe import RouterOut, moe, router
from ref.ds41.ops import _wide, rmsnorm
from ref.ds41.trace_io import Trace, TraceError


class ReplayError(Exception):
    """The engine trace lacks something the replay of (layer, position) needs."""


@contextlib.contextmanager
def _patched(module, name, fn):
    orig = getattr(module, name)
    setattr(module, name, fn)
    try:
        yield
    finally:
        setattr(module, name, orig)


class Replayer:
    def __init__(self, model, eng: Trace, *, expert_cache: int = 16, force: bool = True):
        self.m, self.cfg, self.eng, self.force = model, model.cfg, eng, force
        self.dt = model.dt
        self.modes = model.modes
        self.tokens = eng.tokens
        if not self.tokens:
            raise ReplayError("the engine trace has no tokens in trace.json (the engine must write them)")
        if len(self.tokens) > self.cfg.max_seq_len:
            raise ReplayError(f"the oracle was built with max_seq_len {self.cfg.max_seq_len} < {len(self.tokens)} tokens")
        self._hashes = None
        self._experts: OrderedDict = OrderedDict()
        self._expert_cache = expert_cache
        self.cap = TI._Capture(model)

    # ------------------------------------------------------------------------------------------------ helpers
    def need(self, stage, pos, layer=None):
        a = self.eng.get(stage, pos, layer)
        if a is None:
            raise ReplayError(f"the engine trace has no {TI.stage_filename(stage, pos, layer)}")
        return a

    def f(self, a):
        return np.asarray(a, dtype=self.dt)

    def owner(self, L: int) -> int:
        """The FULL layer whose compressed KV / index-K layer L reads (itself for a FULL layer)."""
        if self.modes[L] is Mode.FULL:
            return L
        o = [l for l in range(L) if self.modes[l] is Mode.FULL]
        if not o:
            raise ReplayError(f"layer {L} has no FULL layer before it")
        return o[-1]

    def stream_in(self, L: int, p: int) -> np.ndarray:
        """[1, hc, dim]: the stream entering block L (after Engram on an Engram layer), and pre_mix [1, hc]."""
        cfg = self.cfg
        if L in cfg.engram_layer_ids:
            x = self.need("engram_out", p, L)
        elif L == 0:
            x = np.repeat(self.need("embed", p)[None, :], cfg.hc_mult, axis=0)
        else:
            x = self.need("block_out", p, L - 1)
        if L == 0:
            pm = np.zeros(cfg.hc_mult)
            pm[0] = 1.0
        else:
            pm = self.need("pre_mix", p, L - 1)
        return self.f(x)[None], self.f(pm)[None]

    def stream_pre_engram(self, L: int, p: int):
        cfg = self.cfg
        x = np.repeat(self.need("embed", p)[None, :], cfg.hc_mult, axis=0) if L == 0 else self.need("block_out", p, L - 1)
        return self.f(x)[None]

    def hashes(self):
        if self._hashes is None:
            hasher = self.m.new_cache().hasher
            self._hashes = hasher(np.array(self.tokens, dtype=np.int64), 0)          # [n, n_engram_layers, cols]
        return self._hashes

    def expert(self, l: int, e: int) -> tuple:
        key = (l, e)
        if key in self._experts:
            self._experts.move_to_end(key)
            return self._experts[key]
        t = tuple(np.asarray(a, dtype=self.dt) for a in self.m.w.expert(l, e))
        self._experts[key] = t
        while len(self._experts) > self._expert_cache:
            self._experts.popitem(last=False)
        return t

    # ------------------------------------------------------------------------------------------------ state
    def build_state(self, L: int, p: int):
        """The oracle's caches as the ENGINE left them before position p: -> (caches list, shared SharedState)."""
        cfg, m = self.cfg, self.m
        W, ratio, mode = cfg.window_size, cfg.compress_ratios[L], self.modes[L]
        caches: list = [None] * cfg.n_layers
        lc = LayerCache.new(cfg, L, self.dt)
        for q in range(max(0, p - W + 1), p):
            lc.win_kv[q % W] = self.need("kv_win", q, L)
        caches[L] = lc
        shared = SharedState()
        if ratio:
            O = self.owner(L)
            lo = lc if O == L else LayerCache.new(cfg, O, self.dt)
            caches[O] = lo
            shared.compress_owner = shared.index_k_owner = O
            before = p // ratio if O == L else (p + 1) // ratio        # rows the owner has published by the time layer L runs at p
            for j in range(before):
                q = (j + 1) * ratio - 1
                lo.comp_kv[j] = self.need("latent", q, O)
                ik = self.eng.get("index_k", q, O)
                if ik is None and (mode in (Mode.FULL, Mode.REINDEX)):
                    raise ReplayError(f"the engine trace has no {TI.stage_filename('index_k', q, O)}: it is needed to replay the indexer of layer {L} "
                                      "(the engine must dump index_k.L at every position where `latent` is published, DS1_VERIFY.md section 4)")
                if ik is not None:
                    lo.index_k[j] = ik
            if O == L and ratio > 1 and p > 0:
                slot = p % ratio
                wkv, wgate = m.p(f"layers.{L}.attn.compressor.wkv.weight"), m.p(f"layers.{L}.attn.compressor.wgate.weight")
                for q in range(p - slot, p):                         # the open group's earlier tokens: their pooling inputs, from the engine's attn_in
                    xf = _wide(self.f(self.need("attn_in", q, L))[None])
                    lc.kv_state[q % ratio] = (xf @ wkv.T)[0]
                    lc.score_state[q % ratio] = (xf @ wgate.T)[0]
            if mode is Mode.REUSE:
                shared.topk_idxs = self.engine_topk(L, p)
            elif mode is Mode.REINDEX:
                shared.candidates = self.candidate_mask(p)
        return caches, shared

    def engine_topk(self, L: int, p: int) -> np.ndarray:
        """The engine's selection for layer L at p as the oracle's [1, k] int64 array (valid entries only)."""
        t = self.eng.get("topk", p, L)
        if t is None:                                                 # a REUSE layer may omit it: the latest index source before it holds the same
            for l in range(L - 1, -1, -1):
                if l in self.cfg.index_source_layers and self.eng.has("topk", p, l):
                    t = self.eng.get("topk", p, l)
                    break
        if t is None:
            raise ReplayError(f"the engine trace has no {TI.stage_filename('topk', p, L)}")
        return np.array([C.selection(t)], dtype=np.int64).reshape(1, -1)

    def candidate_mask(self, p: int) -> np.ndarray:
        """[1, n] bool candidate pool at position p: the engine's `cand_blocks` if it dumped them, else the candidate-source layer's indexer run on
        the engine's state (the pool is a function of that layer's scores only)."""
        cfg = self.cfg
        C0 = cfg.candidate_source_layer
        ratio = cfg.compress_ratios[C0]
        width = (p + 1) // ratio
        blocks = self.eng.get("cand_blocks", p, C0)
        if blocks is not None:
            keep = np.zeros(-(-width // cfg.candidate_block_size) if width else 0, dtype=bool)
            for b in C.selection(blocks):
                if b < len(keep):
                    keep[b] = True
            return np.repeat(keep, cfg.candidate_block_size)[:width][None]
        if self.modes[C0] is not Mode.FULL:
            raise ReplayError("the candidate-source layer is not a FULL layer: not supported")
        out = self.run_attention(C0, p, self.need("attn_in", p, C0))
        return out["_shared"].candidates

    # ------------------------------------------------------------------------------------------------ attention
    def run_attention(self, L: int, p: int, x_attn, *, force_topk=None) -> dict:
        cfg, m = self.cfg, self.m
        if p + 1 > cfg.max_seq_len:
            raise ReplayError(f"position {p} needs max_seq_len > {p}")
        caches, shared = self.build_state(L, p)
        pre = f"layers.{L}."
        get = lambda n: m.p(pre + "attn." + n)                          # noqa: E731
        patches = []
        if force_topk is not None:
            orig_indexer = A.indexer

            def forced(*a, **kw):
                orig_indexer(*a, **kw)                                  # keep its side effects (index_k row, candidate pool, margins)
                return force_topk

            patches.append(_patched(A, "indexer", forced))
        with contextlib.ExitStack() as st:
            st.enter_context(self.cap)
            for pt in patches:
                st.enter_context(pt)
            self.cap.per_layer.clear()
            y = self.cap.M.attention_layer(cfg, L, self.modes[L], self.f(x_attn)[None], p, m._rope_tables(L), get, lc=caches[L], caches=caches,
                                           shared=shared, quant=m.quant, stale_index_k=m.stale_index_k)
        c = self.cap.per_layer[L]
        return {"attn_out": y, "cap": c, "_shared": shared}

    # ------------------------------------------------------------------------------------------------ one (layer, position)
    def replay(self, L: int, p: int) -> dict:
        """-> {(stage): ndarray} of the oracle's value of every stage of layer L at position p (stage-isolated: see the module docstring); plus
        "notes": [str]."""
        cfg, m = self.cfg, self.m
        out: dict = {}
        notes: list = []
        hc_kw = dict(norm_eps=cfg.norm_eps, hc=cfg.hc_mult, iters=cfg.hc_sinkhorn_iters, eps=cfg.hc_eps)
        pre = f"layers.{L}."

        if L == 0:
            out["embed"] = np.asarray(m.w.rows("embed.weight", np.array([self.tokens[p]])), dtype=np.float32)[0]
        if L in cfg.engram_layer_ids:
            j = cfg.engram_layer_ids.index(L)
            rows = self.f(m.w.engram_rows(L, self.hashes()[p:p + 1, j, :]))
            h = engram_layer(self.stream_pre_engram(L, p), rows, wkv=m.p(f"layers.{L}.engram.wkv.weight"), q_weight=m.p(f"layers.{L}.engram.q_weight"),
                             k_weight=m.p(f"layers.{L}.engram.k_weight"), eps=cfg.norm_eps, quant=m.quant).astype(self.dt, copy=False)
            out["engram_out"] = h[0]

        x, pre_mix = self.stream_in(L, p)
        # ---- attention sub-layer
        a_pre, a_post, a_comb = hc_mixes(x, m.p(pre + "hc_attn_fn"), m.p(pre + "hc_attn_scale"), m.p(pre + "hc_attn_base"), **hc_kw)
        y = rmsnorm(hc_pre(x, pre_mix), m.p(pre + "attn_norm.weight"), cfg.norm_eps)
        out["attn_in"] = y[0]
        attn_in_eng = self.need("attn_in", p, L)
        res = self.run_attention(L, p, attn_in_eng)
        c = res["cap"]
        ratio = cfg.compress_ratios[L]
        out["q"] = c["q"][0]
        out["kv_win"] = c["kv_new"][0]
        if ratio:
            out["topk"] = c["topk"][0]
            if p in c.get("pub", {}):
                out["latent"], out["index_k"] = c["pub"][p]
        mg = c.get("margins", {})
        if "index" in mg:
            out["index_margin"] = np.array([mg["index"][0][0], mg["index"][1][0]], dtype=np.float32)
        if "blocks" in mg:
            out["cand_margin"] = np.array([mg["blocks"][0][0], mg["blocks"][1][0]], dtype=np.float32)
        attn_out = res["attn_out"]
        if self.force and ratio and self.modes[L] in (Mode.FULL, Mode.REINDEX):
            eng_topk = self.engine_topk(L, p)
            if C.selection(eng_topk) != C.selection(out["topk"]):
                notes.append(f"layer {L} position {p}: the engine's top-k differs from the oracle's; attn_out below is recomputed with the engine's selection")
                attn_out = self.run_attention(L, p, attn_in_eng, force_topk=eng_topk)["attn_out"]
        out["attn_out"] = attn_out[0]

        # ---- the stream after attention, from the engine's attention output
        attn_out_eng = self.f(self.need("attn_out", p, L))[None]
        xm = hc_post(attn_out_eng, x, a_post, a_comb)
        f_pre, f_post, f_comb = hc_mixes(xm, m.p(pre + "hc_ffn_fn"), m.p(pre + "hc_ffn_scale"), m.p(pre + "hc_ffn_base"), **hc_kw)
        ffn_in = rmsnorm(hc_pre(xm, a_pre), m.p(pre + "ffn_norm.weight"), cfg.norm_eps)
        out["ffn_in"] = ffn_in[0]

        # ---- FFN from the engine's ffn_in
        ffn_in_eng = self.f(self.need("ffn_in", p, L))[None]
        gate_w, gate_b = m.p(pre + "ffn.gate.weight"), m.p(pre + "ffn.gate.bias")
        r = router(ffn_in_eng, gate_w, gate_b, topk=cfg.n_activated_experts, route_scale=cfg.route_scale, score_func=cfg.score_func,
                   gate_temp=cfg.gate_temp, norm_topk_prob=cfg.norm_topk_prob)
        out["router_idx"] = r.indices[0]
        out["router_w"] = r.weights[0]
        out["router_margin"] = np.array([r.margin[0], float(np.max(np.abs(np.asarray(r.scores, dtype=np.float64) + np.asarray(gate_b, dtype=np.float64))))],
                                        dtype=np.float32)
        use = r
        eng_idx = self.eng.get("router_idx", p, L)
        if self.force and eng_idx is not None and C.selection(eng_idx) != C.selection(r.indices[0]) and len(set(C.selection(eng_idx))) == cfg.n_activated_experts:
            notes.append(f"layer {L} position {p}: the engine's routed experts differ from the oracle's; ffn_out below is recomputed with the engine's experts")
            use = forced_router(r, np.array([C.selection(eng_idx)], dtype=np.int64), cfg)
        shared_w = tuple(np.asarray(a, dtype=self.dt) for a in m.w.shared_expert(L))
        yf = moe(ffn_in_eng, gate_w=gate_w, gate_bias=gate_b, get_expert=lambda e: self.expert(L, e), shared=shared_w, cfg=cfg, quant=m.quant, router_out=use)
        out["ffn_out"] = yf[0]

        # ---- the block's output from the engine's ffn_out
        ffn_out_eng = self.f(self.need("ffn_out", p, L))[None]
        out["block_out"] = hc_post(ffn_out_eng, xm, f_post, f_comb).astype(self.dt, copy=False)[0]
        out["pre_mix"] = f_pre[0]
        out["_notes"] = notes
        return out

    # ------------------------------------------------------------------------------------------------ the head
    def replay_head(self, positions) -> dict:
        """-> {pos: {"final_hidden": ..., "logits": ...}} (one pass over the head for all positions)."""
        cfg, m = self.cfg, self.m
        last = cfg.n_layers - 1
        hs = []
        for p in positions:
            h = self.f(self.need("block_out", p, last))[None]
            pm = self.f(self.need("pre_mix", p, last))[None]
            hs.append(rmsnorm(hc_pre(h, pm), m.p("norm.weight"), cfg.norm_eps)[0])
        fin = np.stack(hs)
        res = {}
        eng_fin = np.stack([self.f(self.need("final_hidden", p)) for p in positions])
        lg = m.w.matmul_t("head.weight", eng_fin.astype(np.promote_types(self.dt, np.float32)))
        for i, p in enumerate(positions):
            res[p] = {"final_hidden": fin[i], "logits": lg[i]}
        return res


def forced_router(r: RouterOut, idx: np.ndarray, cfg) -> RouterOut:
    """The oracle's router output with the ENGINE's expert set (weights by the oracle's formula from its own scores: w = s[idx] / (sum + 1e-20) * route_scale)."""
    w = np.take_along_axis(r.scores, idx, axis=-1)
    if cfg.norm_topk_prob and idx.shape[-1] > 1:
        w = w / (w.sum(axis=-1, keepdims=True) + 1e-20)
    w = w * cfg.route_scale
    return RouterOut(w.astype(r.weights.dtype, copy=False), idx, r.scores, r.margin)


# ---------------------------------------------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------------------------------------------


def replay_all(model, eng: Trace, layers, positions, *, force: bool = True, expert_cache: int = 16, progress=None, include_head: bool = False) -> tuple:
    """Replay every (layer, position) -> (DictSource of the oracle's stage values, keys in execution order, notes)."""
    rp = Replayer(model, eng, expert_cache=expert_cache, force=force)
    ref = C.DictSource()
    keys = []
    notes: list = []
    t0 = time.time()
    for L in layers:                                  # layer-major: a layer's weights are decoded once and serve every position (the cache holds a few layers)
        for p in positions:
            try:
                out = rp.replay(L, p)
            except ReplayError as e:
                notes.append(f"layer {L} position {p}: cannot replay: {e}")
                continue
            notes += out.pop("_notes")
            for stage, arr in out.items():
                layer = None if stage == "embed" else L
                ref.put(stage, p, layer, arr)
                if not (TI.STAGE[stage].role == "oracle"):
                    keys.append((p, layer, stage))
            if progress:
                progress(f"layer {L} position {p} replayed ({time.time() - t0:.0f} s)")
    if include_head:
        done = [p for p in positions if eng.has("block_out", p, model.cfg.n_layers - 1)]
        try:
            for p, d in rp.replay_head(done).items():
                for stage, arr in d.items():
                    ref.put(stage, p, None, arr)
                    keys.append((p, None, stage))
        except ReplayError as e:
            notes.append(f"head: cannot replay: {e}")
    keys.sort(key=lambda k: C.exec_key(*k))
    return ref, keys, notes


def replay_compare(model, eng: Trace, layers, positions, *, tol_scale: float = 1.0, strict: bool = False, force: bool = True, expert_cache: int = 16,
                   progress=None, include_head: bool = False) -> C.Report:
    """Layer-by-layer comparison of `eng` against the oracle `model` for the given layers and positions (mode "layer")."""
    ref, keys, notes = replay_all(model, eng, layers, positions, force=force, expert_cache=expert_cache, progress=progress, include_head=include_head)
    tols = C.Tolerances(C.Dims.from_summary(TI.model_summary(model)), model.quant, tol_scale)
    rep = C.compare_sources(eng, ref, keys, "layer", tols=tols, strict=strict,
                            title=f"engine {eng.path} vs oracle replay (layer-by-layer) {len(layers)} layer(s) x {len(positions)} position(s)")
    rep.notes += notes
    rep.errors += [n for n in notes if "cannot replay" in n]     # a (layer, position) that could not be checked is not a pass
    return rep


def baseline_report(model_a, model_b, eng: Trace, layers, positions, *, expert_cache: int = 16, progress=None, include_head: bool = False) -> C.Report:
    """The oracle's own noise floor on THESE engine inputs: the replay by `model_a` (the oracle the engine is compared with) against the replay by `model_b`
    (the same oracle in the other floating-point type), stage by stage.  Both compute every stage from the engine's values, so the differences are exactly
    what two correct implementations disagree by here: float summation order and the odd int8 / fp8 / fp4 rounding flip.  An engine should look like this
    report; where it is much worse than this, it is not rounding.  (DS1_VERIFY.md section 3.5)"""
    a_src, keys, _ = replay_all(model_a, eng, layers, positions, force=False, expert_cache=expert_cache, progress=progress, include_head=include_head)
    b_src, _, _ = replay_all(model_b, eng, layers, positions, force=False, expert_cache=expert_cache, progress=progress, include_head=include_head)
    tols = C.Tolerances(C.Dims.from_summary(TI.model_summary(model_a)), model_a.quant)
    rep = C.compare_sources(a_src, b_src, keys, "layer", tols=tols,
                            title=f"baseline: oracle {model_a.dt} replay vs oracle {model_b.dt} replay on the same engine inputs (the noise floor)")
    return rep


def main_layers(a) -> int:
    eng = Trace(a.engine)
    quant = TI.quant_from_dict(eng.meta.get("quant")) if a.quant == "from-engine" else C.quant_from_name(a.quant)
    if a.quant == "from-engine" and "quant" not in eng.meta:
        quant = C.quant_from_name("int8-kv")
    n_tok = len(eng.tokens)
    model = C.load_oracle_model(a.gguf, quant, dtype=a.dtype, max_seq_len=max(n_tok + 8, 64), cache_bytes=int(a.cache_gib * (1 << 30)))
    cfg = model.cfg
    words = [w for w in (a.layers or "").split(",") if w.strip()]
    include_head = "head" in words or not words
    layers = C._ints(",".join(w for w in words if w != "head")) if words else None
    if layers is None:
        layers = [l for l in range(cfg.n_layers) if eng.has("block_out", eng.positions()[0], l)] if eng.positions() else []
    positions = C._ints(a.positions)
    if positions is None:
        ps = eng.positions()
        positions = sorted({ps[0], ps[-1]}) if ps else []
    positions = [p for p in positions if p in set(eng.positions())]
    if not positions or not layers and not include_head:
        print("ds1_compare layers: nothing to compare (no common positions / layers)", file=sys.stderr)
        return 2
    rep = replay_compare(model, eng, layers, positions, tol_scale=a.tol_scale, strict=a.strict, force=not a.no_force, expert_cache=a.expert_cache,
                         progress=lambda s: print(s, file=sys.stderr), include_head=include_head)
    print(rep.text(verbose=a.verbose))
    if a.baseline:
        other = "float64" if a.dtype == "float32" else "float32"
        model_b = C.load_oracle_model(a.gguf, quant, dtype=other, max_seq_len=max(n_tok + 8, 64), cache_bytes=int(a.cache_gib * (1 << 30)))
        base = baseline_report(model, model_b, eng, layers, positions, expert_cache=a.expert_cache, progress=lambda s: print("baseline: " + s, file=sys.stderr),
                               include_head=include_head)
        print()
        print(base.text(verbose=False))
        fr = lambda r: sum(1 for s in r.samples if s.level != C.Level.OK) / max(len(r.samples), 1)       # noqa: E731
        print(f"samples above the soft tolerance: engine {100 * fr(rep):.1f} %, oracle against itself {100 * fr(base):.1f} %  (the engine is held to its budget either way; "
              "a much higher rate than the baseline's is the thing to look at)")
    if a.json:
        import json
        import pathlib
        pathlib.Path(a.json).write_text(json.dumps(rep.to_json(), indent=1))
    return 0 if rep.ok else 1


if __name__ == "__main__":
    import ds1_compare
    sys.exit(ds1_compare.main(["layers"] + sys.argv[1:]))
