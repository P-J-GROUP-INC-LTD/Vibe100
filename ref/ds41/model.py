"""ref/ds41/model.py - the block and the full-model forward (prefill over a token sequence, incremental decode).

Transcribes model.py:Block (907-994) and Transformer.forward (1242-1273) for one sequence (batch 1, text only, no
DSpark head, no vision):

    h   = embed[ids]  ->  4 identical copies  [s, 4, dim]
    pre = [1, 0, 0, 0]                                    (make_identity_pre_mix)
    for layer i:
        if i in engram_layer_ids:  h = engram(h, hash rows of the n-grams ending at each position)
        h, pre = block_i(h, pre)
    h      = hc_pre(h, pre)                                (the LAST block's FFN `pre`: the single-pass lag)
    logits = head(RMSNorm(h))                              (fp32; last position unless full_logits)

    block(x, pre_mix):                                     (the one-block lag lives here)
        attn_pre, attn_post, attn_comb = hc_mixes(x, attn mHC params)
        x = x + attention:  hc_post( attention(RMSNorm(hc_pre(x, pre_mix))), x, attn_post, attn_comb )
        ffn_pre,  ffn_post,  ffn_comb  = hc_mixes(x, ffn mHC params)
        x = hc_post( moe(RMSNorm(hc_pre(x, attn_pre))), x, ffn_post, ffn_comb )       <- uses attn_pre!
        return x, ffn_pre                                  (consumed by the NEXT block's attention)

Decode is `forward(ids[1], start_pos)` with start_pos > 0 on the cache the prefill (start_pos == 0) created, exactly
the contract of the reference (`Transformer.forward`): prefill is one chunk starting at 0; afterwards one token per call.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .attention import LayerCache, SharedState, attention_layer
from .config import Config, layer_modes
from .engram import NgramHasher, engram_layer
from .mhc import hc_mixes, hc_post, hc_pre
from .moe import moe, router
from .ops import rmsnorm
from .quant import QuantConfig
from .rope import layer_rope_params, rope_table
from .weights import Weights


@dataclasses.dataclass
class ModelCache:
    layers: list
    shared: SharedState
    hasher: NgramHasher | None
    pos: int = 0                       # number of tokens already processed


class Model:
    def __init__(self, cfg: Config, weights: Weights, *, quant: QuantConfig | None = None, dtype=np.float32,
                 token_map=None, stale_index_k: bool = False, rope_dtype=np.float64, engram_constants: dict | None = None):
        """quant: which of the reference's activation fake-quantisations to apply (default: none = exact);
        dtype: compute dtype of activations/parameters (float32 for real use, float64 for validation);
        token_map: 129,280 -> 99,092 map (required when cfg.engram_layer_ids is non-empty);
        stale_index_k: reproduce the reference's stale `shared_attn.index_k` quirk (see attention.py);
        engram_constants: dict(layout, multipliers, token_map) taken from a GGUF (engram.constants_from_gguf_metadata);
        when given it overrides the derivation from cfg (and supplies token_map)."""
        self.cfg, self.w = cfg, weights
        self.quant = quant or QuantConfig.exact()
        self.dt = np.dtype(dtype)
        self.modes = layer_modes(cfg)
        self.stale_index_k = stale_index_k
        self.engram_constants = engram_constants
        if engram_constants is not None and token_map is None:
            token_map = engram_constants["token_map"]
        self.token_map = None if token_map is None else np.asarray(token_map, dtype=np.int64)
        if cfg.engram_layer_ids and self.token_map is None:
            raise ValueError("Engram layers present: pass token_map (see engram.build_compressed_token_map)")
        self._rope: dict = {}
        self._rope_dtype = rope_dtype

    # ------------------------------------------------------------------ parameter access
    def p(self, name: str) -> np.ndarray:
        return np.asarray(self.w.get(name), dtype=self.dt)

    def _rope_tables(self, layer: int):
        p = layer_rope_params(self.cfg, layer)
        key = (p["original_seq_len"], p["base"])
        if key not in self._rope:
            c = self.cfg
            self._rope[key] = rope_table(c.rope_head_dim, c.max_seq_len, p["original_seq_len"], p["base"],
                                         c.rope_factor, c.beta_fast, c.beta_slow, self._rope_dtype)
        return self._rope[key]

    def new_cache(self) -> ModelCache:
        cfg = self.cfg
        hasher = None
        if cfg.engram_layer_ids:
            ec = self.engram_constants or {}
            hasher = NgramHasher(cfg, self.token_map, ec.get("layout"), multipliers=ec.get("multipliers"))
        return ModelCache([LayerCache.new(cfg, l, self.dt) for l in range(cfg.n_layers)], SharedState(), hasher)

    # ------------------------------------------------------------------ one block
    def block(self, l: int, x: np.ndarray, start_pos: int, pre_mix: np.ndarray, cache: ModelCache,
              trace: dict | None = None):
        cfg = self.cfg
        pre = f"layers.{l}."
        hc_kw = dict(norm_eps=cfg.norm_eps, hc=cfg.hc_mult, iters=cfg.hc_sinkhorn_iters, eps=cfg.hc_eps)

        # ---- attention sub-layer
        residual = x
        a_pre, a_post, a_comb = hc_mixes(x, self.p(pre + "hc_attn_fn"), self.p(pre + "hc_attn_scale"),
                                         self.p(pre + "hc_attn_base"), **hc_kw)
        y = hc_pre(x, pre_mix)
        y = rmsnorm(y, self.p(pre + "attn_norm.weight"), cfg.norm_eps)
        y = attention_layer(cfg, l, self.modes[l], y, start_pos, self._rope_tables(l),
                            lambda n: self.p(pre + "attn." + n), lc=cache.layers[l], caches=cache.layers,
                            shared=cache.shared, quant=self.quant, stale_index_k=self.stale_index_k)
        if trace is not None:
            trace.setdefault("attn_out", {})[l] = y.copy()
        x = hc_post(y, residual, a_post, a_comb)

        # ---- FFN sub-layer (collapsed with the attention's own `pre`)
        residual = x
        f_pre, f_post, f_comb = hc_mixes(x, self.p(pre + "hc_ffn_fn"), self.p(pre + "hc_ffn_scale"),
                                         self.p(pre + "hc_ffn_base"), **hc_kw)
        y = hc_pre(x, a_pre)
        y = rmsnorm(y, self.p(pre + "ffn_norm.weight"), cfg.norm_eps)
        gate_w, gate_b = self.p(pre + "ffn.gate.weight"), self.p(pre + "ffn.gate.bias")
        r = router(y, gate_w, gate_b, topk=cfg.n_activated_experts, route_scale=cfg.route_scale,
                   score_func=cfg.score_func, gate_temp=cfg.gate_temp, norm_topk_prob=cfg.norm_topk_prob)
        y = moe(y, gate_w=gate_w, gate_bias=gate_b,
                get_expert=lambda e: tuple(np.asarray(a, dtype=self.dt) for a in self.w.expert(l, e)),
                shared=tuple(np.asarray(a, dtype=self.dt) for a in self.w.shared_expert(l)), cfg=cfg,
                quant=self.quant, router_out=r)
        if trace is not None:
            trace.setdefault("ffn_out", {})[l] = y.copy()
            trace.setdefault("router_idx", {})[l] = r.indices.copy()
            trace.setdefault("router_w", {})[l] = r.weights.copy()
            trace.setdefault("router_margin", {})[l] = r.margin.copy()
        x = hc_post(y, residual, f_post, f_comb)
        return x.astype(self.dt, copy=False), f_pre

    # ------------------------------------------------------------------ whole model
    def forward(self, ids, start_pos: int, cache: ModelCache, *, full_logits: bool = False,
                trace: dict | None = None) -> np.ndarray:
        """ids: int array [s].  start_pos == 0: prefill of the whole prompt (resets the caches);
        start_pos > 0: ONE new token (s == 1) appended after `start_pos` tokens.  Returns fp32 logits [vocab]
        (or [s, vocab] with full_logits)."""
        cfg = self.cfg
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        s = ids.shape[0]
        if start_pos == 0:
            fresh = self.new_cache()
            cache.layers, cache.shared, cache.hasher, cache.pos = fresh.layers, fresh.shared, fresh.hasher, 0
        else:
            assert s == 1 and start_pos == cache.pos, "decode: one token, positions must be consecutive"
        assert start_pos + s <= cfg.max_seq_len

        hashes = cache.hasher(ids, start_pos) if cache.hasher is not None else None
        h = np.asarray(self.w.rows("embed.weight", ids), dtype=self.dt)
        h = np.repeat(h[:, None, :], cfg.hc_mult, axis=1)
        pre_mix = np.zeros((s, cfg.hc_mult), dtype=self.dt)
        pre_mix[:, 0] = 1.0                                              # make_identity_pre_mix
        for l in range(cfg.n_layers):
            if l in cfg.engram_layer_ids:
                j = cfg.engram_layer_ids.index(l)
                rows = np.asarray(self.w.engram_rows(l, hashes[:, j, :]), dtype=self.dt)      # [s, cols, head_dim]
                h = engram_layer(h, rows, wkv=self.p(f"layers.{l}.engram.wkv.weight"),
                                 q_weight=self.p(f"layers.{l}.engram.q_weight"),
                                 k_weight=self.p(f"layers.{l}.engram.k_weight"), eps=cfg.norm_eps,
                                 quant=self.quant).astype(self.dt, copy=False)
                if trace is not None:
                    trace.setdefault("engram_out", {})[l] = h.copy()
            h, pre_mix = self.block(l, h, start_pos, pre_mix, cache, trace)
            if trace is not None:
                trace.setdefault("block_out", {})[l] = h.copy()
                trace.setdefault("pre_mix", {})[l] = pre_mix.copy()
        cache.pos = start_pos + s
        x = hc_pre(h, pre_mix)
        x = rmsnorm(x, self.p("norm.weight"), cfg.norm_eps)
        if trace is not None:
            trace["final_hidden"] = x.copy()
        if not full_logits:
            x = x[-1:]
        logits = self.w.matmul_t("head.weight", x.astype(np.promote_types(self.dt, np.float32)))       # fp32 head (model.py:ParallelHead)
        return logits if full_logits else logits[0]

    # convenience wrappers
    def prefill(self, ids, cache: ModelCache, **kw) -> np.ndarray:
        return self.forward(ids, 0, cache, **kw)

    def decode(self, token: int, cache: ModelCache, **kw) -> np.ndarray:
        return self.forward(np.array([token]), cache.pos, cache, **kw)

    def generate_greedy(self, prompt, n_new: int) -> list:
        """Greedy decode (argmax) for smoke tests; returns the n_new generated ids."""
        cache = self.new_cache()
        logits = self.prefill(prompt, cache)
        out = []
        for _ in range(n_new):
            t = int(np.argmax(logits))
            out.append(t)
            logits = self.decode(t, cache)
        return out


def model_from_gguf(paths, *, quant: QuantConfig | None = None, dtype=np.float32, max_seq_len: int = 4096,
                    stale_index_k: bool = False, cache_bytes: int = 2 << 30) -> Model:
    """Open the GGUF shard(s) lazily and build a ready-to-run `Model` (config, Engram constants and token map all from
    the file's own metadata)."""
    from .engram import constants_from_gguf_metadata
    from .weights import load_from_gguf
    w, cfg = load_from_gguf(paths, cache_bytes=cache_bytes)
    cfg.max_seq_len = max_seq_len
    ec = constants_from_gguf_metadata(w.metadata) if cfg.engram_layer_ids else None
    return Model(cfg, w, quant=quant, dtype=dtype, stale_index_k=stale_index_k, engram_constants=ec)
