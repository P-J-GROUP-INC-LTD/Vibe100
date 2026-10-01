"""ref/ds41/attention.py - MQA attention with attention sink over [SWA window ; selected compressed KV].

Transcribes model.py:
  get_window_topk_idxs   410-426      which cache slots a query's sliding window covers
  Compressor.forward     458-485      ratio-2 gated softmax pooling over pairs; ratio-1 plain projection
  Indexer.forward        527-580      32 x 128 relu-weighted scores, top-512, position-sorted
  select_candidate_blocks 583-610     the candidate pool: 8-position blocks, newest pinned, top 2048 blocks
  Attention.*            700-789      window KV, compressed KV, sparse attention, grouped output projection
and kernel.py:sparse_attn_kernel (311-389) for the attention itself (softmax with an extra `sink` logit).

State that the reference keeps in module buffers / the global `shared_attn` lives in `LayerCache` / `SharedState`:
a layer's SWA ring (128 slots), its compressed-KV cache and index-K cache (KV-source layers only), the ratio-2 pooling
state, and the cross-layer hand-off of compressed KV, index-K, top-k indices and the candidate mask.

Known reference quirk (flag `stale_index_k`): model.py:560 reads `shared_attn.index_k`, which an index-K *owner* layer
only refreshes when its own group completed (`latent is not None`, line 548-549).  On ratio-2 decode steps where the
group is still incomplete, layers 2/8/14 therefore score against the previous owner's cache (layer 20's, a different
ratio) instead of their own.  The default here is the intended behaviour (own cache); `stale_index_k=True` reproduces
the reference exactly so that the tiny-model test can match it bit for bit.  See README "Known deviations".
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .config import Config, Mode
from .ops import _wide, linear, rmsnorm, softmax
from .quant import QuantConfig, act_quant_fp8, fp4_quant_e4m3, fp4_quant_e8m0
from .rope import apply_rope_tail


# ---------------------------------------------------------------------------------------------------------------
# caches and the cross-layer hand-off
# ---------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class LayerCache:
    win_kv: np.ndarray                       # [window, head_dim]  ring: slot = position % window (post-RoPE, post-quant)
    comp_kv: np.ndarray | None = None        # [max_seq // ratio, head_dim]   KV-source layers
    index_k: np.ndarray | None = None        # [max_seq // ratio, index_head_dim]   KV-source layers
    kv_state: np.ndarray | None = None       # [ratio, head_dim]  ratio > 1: tokens of the incomplete group
    score_state: np.ndarray | None = None

    @classmethod
    def new(cls, cfg: Config, layer: int, dtype=np.float32) -> "LayerCache":
        ratio = cfg.compress_ratios[layer]
        lc = cls(win_kv=np.zeros((cfg.window_size, cfg.head_dim), dtype))
        if ratio and layer in cfg.kv_source_layers:
            n = cfg.max_seq_len // ratio
            lc.comp_kv = np.zeros((n, cfg.head_dim), dtype)
            lc.index_k = np.zeros((n, cfg.index_head_dim), dtype)
            if ratio > 1:
                sdt = np.promote_types(dtype, np.float32)             # the reference keeps these in fp32
                lc.kv_state = np.zeros((ratio, cfg.head_dim), sdt)
                lc.score_state = np.full((ratio, cfg.head_dim), -np.inf, sdt)
        return lc


@dataclasses.dataclass
class SharedState:
    """model.py:SharedAttentionRuntime: layers run in order and every source writes before its consumers read."""
    compress_owner: int | None = None        # layer whose comp_kv the following layers read
    index_k_owner: int | None = None         # layer whose index_k the indexers read
    topk_idxs: np.ndarray | None = None      # [s, k] compressed-cache positions chosen by the last index source (-1 = none);
                                             # the reference stores them already offset by the window length
    candidates: np.ndarray | None = None     # [s, n] bool candidate-pool mask from the candidate-source layer
    diag: dict = dataclasses.field(default_factory=dict)   # selection margins (near-ties), see `note_margin`


def note_margin(shared: SharedState, key: str, scores: np.ndarray, k: int) -> None:
    """Record, per selection (top-k of `scores` [rows, n]), the smallest gap between the k-th and (k+1)-th best finite
    scores.  A gap near 0 means the reference's torch.topk may legitimately break the tie the other way; tests use
    `min(shared.diag[key])` to prove that their scenario is tie-free before comparing selections."""
    if scores.shape[-1] <= k:
        return
    top = -np.sort(-scores, axis=-1)
    a, b = top[:, k - 1], top[:, k]
    ok = np.isfinite(b) & np.isfinite(a)
    if ok.any():
        shared.diag.setdefault(key, []).append(float(np.min(a[ok] - b[ok])))


# ---------------------------------------------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------------------------------------------

def window_topk_idxs(window_size: int, seqlen: int, start_pos: int) -> np.ndarray:
    """model.py:get_window_topk_idxs.  Prefill (start_pos == 0): row t lists the chunk positions
    [max(0, t-W+1), t] (padded with -1 up to min(seqlen, W) entries); the KV there is the chunk itself.  Decode: the
    single query sees the whole ring, `-1` for slots that hold nothing yet."""
    if start_pos == 0:
        end = np.arange(seqlen)[:, None]
        idxs = np.maximum(end - window_size + 1, 0) + np.arange(min(seqlen, window_size))
        idxs = np.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = np.concatenate([np.arange(oldest, window_size), np.arange(oldest)])
        idxs = np.where(idxs > start_pos, -1, idxs)[None, :]
    return idxs.astype(np.int64)


def window_kv(cfg: Config, x: np.ndarray, start_pos: int, cos: np.ndarray, sin: np.ndarray, *, wkv: np.ndarray,
              kv_norm_w: np.ndarray, lc: LayerCache, quant: QuantConfig):
    """model.py:Attention._window_kv.  kv = RMSNorm512(wkv x); RoPE on the last 64; fp8 fake-quant over all 512
    (when quant.window_kv); seed (prefill) / update (decode) the ring.  Returns (kv rows to attend over, idxs)."""
    s = x.shape[0]
    win = cfg.window_size
    kv = rmsnorm(linear(x, wkv, act_quant=True, quant=quant), kv_norm_w, cfg.norm_eps)
    kv = apply_rope_tail(kv, cos, sin, cfg.rope_head_dim)
    if quant.window_kv:
        kv = act_quant_fp8(kv, 32)
    if start_pos == 0:
        if s <= win:
            lc.win_kv[:s] = kv
        else:
            cutoff = s % win
            lc.win_kv[cutoff:win] = kv[-win:][: win - cutoff]
            lc.win_kv[:cutoff] = kv[-win:][win - cutoff:]
        rows = kv
    else:
        assert s == 1, "decode takes one token at a time"
        lc.win_kv[start_pos % win] = kv[0]
        rows = lc.win_kv
    return rows, window_topk_idxs(win, s, start_pos)


# ---------------------------------------------------------------------------------------------------------------
# compressor
# ---------------------------------------------------------------------------------------------------------------

def compressor(cfg: Config, ratio: int, x: np.ndarray, start_pos: int, *, wkv: np.ndarray, wgate: np.ndarray | None,
               norm_w: np.ndarray, lc: LayerCache) -> np.ndarray | None:
    """model.py:Compressor.forward.  Returns the PRE-RoPE latent [m, head_dim] (RMSNorm512'd), or None while the
    current group is incomplete.

    ratio 1 (layers 20-39): c = RMSNorm512(wkv x) - one latent per token, no gate.
    ratio 2 (layers 2-19):  kv = wkv x, score = wgate x (both fp32, 512 channels); per group of 2 tokens
        c = sum_r softmax_r(score_r) * kv_r  (softmax over the group, independently per channel); c = RMSNorm512(c).
    A latent for group j is available once both its tokens have been seen; it stands for position j * ratio.
    The tail of an incomplete group waits in lc.kv_state / lc.score_state."""
    s = x.shape[0]
    if ratio == 1:
        return rmsnorm(x @ wkv.T, norm_w, cfg.norm_eps)
    xf = _wide(x)
    kv, score = xf @ wkv.T, xf @ wgate.T
    if start_pos == 0:
        should = s >= ratio
        rem = s % ratio
        cutoff = s - rem
        lc.kv_state[:] = 0
        lc.score_state[:] = -np.inf
        if rem:
            lc.kv_state[:rem] = kv[cutoff:]
            lc.score_state[:rem] = score[cutoff:]
        kvg = kv[:cutoff].reshape(-1, ratio, kv.shape[-1])
        scg = score[:cutoff].reshape(-1, ratio, score.shape[-1])
        latent = np.sum(kvg * softmax(scg, axis=1), axis=1)
    else:
        assert s == 1
        should = (start_pos + 1) % ratio == 0
        slot = start_pos % ratio
        lc.kv_state[slot] = kv[0]
        lc.score_state[slot] = score[0]
        latent = None
        if should:
            latent = np.sum(lc.kv_state * softmax(lc.score_state, axis=0), axis=0, keepdims=True)
    if not should:
        return None
    return rmsnorm(latent, norm_w, cfg.norm_eps)


# ---------------------------------------------------------------------------------------------------------------
# indexer + candidate pool
# ---------------------------------------------------------------------------------------------------------------

def select_candidate_blocks(logits: np.ndarray, compress_lens, topk_blocks: int, block_size: int, *,
                            shared: SharedState | None = None, key: str = "blocks") -> np.ndarray:
    """model.py:select_candidate_blocks.  logits [s, n] (unreachable positions already -inf); compress_lens [s, 1]
    or int.  Per query: score every block of `block_size` positions by its best position, pin the block holding the
    query's newest reachable position (score +inf), keep the `topk_blocks` best blocks that are reachable; returns a
    bool mask [s, n] (block-expanded).  Ties between block scores go to the lower block index."""
    s, width = logits.shape
    pad = -width % block_size
    scores = np.concatenate([logits, np.full((s, pad), -np.inf, logits.dtype)], axis=-1) if pad else logits
    scores = scores.reshape(s, -1, block_size).max(axis=-1)
    nb = scores.shape[-1]
    last = (np.asarray(compress_lens) - 1) // block_size                     # floor division: 0 reachable -> -1 (no pin)
    scores = np.where(np.arange(nb)[None, :] == np.reshape(last, (-1, 1)), np.inf, scores)
    k = min(topk_blocks, nb)
    if shared is not None:
        note_margin(shared, key, scores, k)
    order = np.argsort(-scores, axis=-1, kind="stable")[:, :k]
    top_vals = np.take_along_axis(scores, order, axis=-1)
    keep = np.zeros((s, nb), dtype=bool)
    np.put_along_axis(keep, order, top_vals > -np.inf, axis=-1)
    return np.repeat(keep, block_size, axis=-1)[:, :width]


def indexer(cfg: Config, layer: int, x: np.ndarray, qr: np.ndarray, latent: np.ndarray | None, start_pos: int,
            cos: np.ndarray, sin: np.ndarray, *, wq_b: np.ndarray, weights_proj: np.ndarray,
            wk: np.ndarray | None, k_norm_w: np.ndarray | None, lc: LayerCache, shared: SharedState,
            caches: list, quant: QuantConfig, stale_index_k: bool = False) -> np.ndarray:
    """model.py:Indexer.forward.  Returns [s, topk] int64 compressed-position indices, ascending, -1 where the
    position is not reachable by that query (not yet complete / outside the candidate pool).

        index-K (owner layers 2/8/14/20 only):  k = RMSNorm128(wk(latent)); RoPE (last 64) at group*ratio; fp4 (block 32)
        q  = wq_b(qr) -> [32, 128]; RoPE (last 64) at the query position; fp4 (block 32)
        w  = weights_proj(x) * 128^-0.5 * 32^-0.5
        score[t] = sum_h relu(q[h] . k[t]) * w[h]
        a block of `ratio` tokens is reachable once the query has passed its last token (compress_lens = (t+1)//ratio)
        candidate-source layer (20): shared.candidates = select_candidate_blocks(score);  later indexer layers (24..36):
        score masked to the candidate pool; finally top-k, re-sorted into position order.
    """
    s = x.shape[0]
    ratio, rd, end_pos = cfg.compress_ratios[layer], cfg.rope_head_dim, start_pos + s
    owns_k = layer in cfg.kv_source_layers
    if owns_k and latent is not None:
        m = latent.shape[0]
        # a latent stands for the first token of its group: group j takes position j * ratio
        pos = np.arange(0, s - s % ratio, ratio) if start_pos == 0 else np.array([start_pos + 1 - ratio])
        k = rmsnorm(latent @ wk.T, k_norm_w, cfg.norm_eps)
        k = apply_rope_tail(k, cos[pos], sin[pos], rd)
        if quant.index:
            k = fp4_quant_e8m0(k, 32)
        lc.index_k[start_pos // ratio: start_pos // ratio + m] = k
        shared.index_k_owner = layer
    q = linear(qr, wq_b, act_quant=True, quant=quant).reshape(s, cfg.index_n_heads, cfg.index_head_dim)
    q = apply_rope_tail(q, cos[start_pos:end_pos], sin[start_pos:end_pos], rd)
    if quant.index:
        q = fp4_quant_e8m0(q, 32)

    owner = layer if (owns_k and not stale_index_k) else shared.index_k_owner
    index_k = caches[owner].index_k[: end_pos // ratio]
    weights = (x @ weights_proj.T) * (cfg.index_head_dim ** -0.5 * cfg.index_n_heads ** -0.5)        # [s, 32]
    sc = np.einsum("shd,td->sht", q, index_k)
    score = np.sum(np.maximum(sc, 0) * weights[:, :, None], axis=1)                                # [s, T]

    if start_pos == 0:
        compress_lens = (np.arange(1, s + 1) // ratio)[:, None]
        score = np.where(np.arange(s // ratio)[None, :] >= compress_lens, -np.inf, score)
    else:
        compress_lens = end_pos // ratio
    is_candidate_source = layer == cfg.candidate_source_layer
    uses_candidates = 0 <= cfg.candidate_source_layer < layer
    if is_candidate_source:
        shared.candidates = select_candidate_blocks(score, compress_lens, cfg.candidate_topk_blocks,
                                                    cfg.candidate_block_size, shared=shared, key=f"blocks.{layer}")
    elif uses_candidates:
        score = np.where(shared.candidates, score, -np.inf)                 # level two: inside the source's pool only

    topk = min(cfg.index_topk, end_pos // ratio)
    note_margin(shared, f"index.{layer}", score, topk)
    # top-k by score (ties -> lower position), then position order; -inf (unreachable) -> -1
    pick = np.argsort(-score, axis=-1, kind="stable")[:, :topk]
    vals = np.take_along_axis(score, pick, axis=-1)
    pick = np.where(vals > -np.inf, pick, np.iinfo(np.int64).max)
    pick = np.sort(pick, axis=-1)
    return np.where(pick < compress_lens, pick, -1).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------
# sparse attention with sink
# ---------------------------------------------------------------------------------------------------------------

def sparse_attn(q: np.ndarray, kv: np.ndarray, sink: np.ndarray, idxs: np.ndarray, scale: float, *,
                max_chunk_elems: int = 1 << 26) -> np.ndarray:
    """kernel.py:sparse_attn.  q [s, h, d], kv [n, d] (K == V), sink [h], idxs [s, k] into kv (-1 = skip).

        logit_i = q . kv[idx_i] * scale          over the valid idx_i of that query
        o = sum_i exp(logit_i - m) kv[idx_i] / (sum_i exp(logit_i - m) + exp(sink - m))     m = max(max_i logit_i, sink)
    A query with no valid index yields zeros (the kernel's convention).  The sink is a logit that owns no value
    vector: it only enlarges the denominator."""
    q, kv = _wide(q), _wide(kv)
    s, h, d = q.shape
    k = idxs.shape[1]
    out = np.zeros_like(q)
    if k == 0:
        return out
    step = max(1, max_chunk_elems // (k * d))
    sink = _wide(sink)
    for a in range(0, s, step):
        b = min(s, a + step)
        ix = idxs[a:b]
        valid = ix >= 0
        g = kv[np.where(valid, ix, 0)]                                           # [c, k, d]
        logits = np.einsum("chd,ckd->chk", q[a:b], g) * scale
        logits = np.where(valid[:, None, :], logits, -np.inf)
        m = np.maximum(logits.max(axis=-1), sink[None, :])                       # [c, h]
        p = np.exp(logits - m[:, :, None])                                       # exp(-inf) = 0 for invalid
        denom = p.sum(axis=-1) + np.exp(sink[None, :] - m)
        out[a:b] = np.einsum("chk,ckd->chd", p, g) / denom[:, :, None]
    return out


# ---------------------------------------------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------------------------------------------

def attention_layer(cfg: Config, layer: int, mode: Mode, x: np.ndarray, start_pos: int, rope: tuple, get, *,
                    lc: LayerCache, caches: list, shared: SharedState, quant: QuantConfig,
                    stale_index_k: bool = False) -> np.ndarray:
    """model.py:Attention.forward.  x [s, dim] (already attn_norm'd) -> [s, dim].

    `rope` = (cos, sin) tables [max_seq, rd/2] of this layer; `get(name)` returns this layer's attention parameter
    `attn.<name>` as a dense float array of the compute dtype.
    """
    s = x.shape[0]
    cos, sin = rope
    rd = cfg.rope_head_dim
    nh, hd = cfg.n_heads, cfg.head_dim
    ratio = cfg.compress_ratios[layer]
    pos = slice(start_pos, start_pos + s)

    qr = rmsnorm(linear(x, get("wq_a.weight"), act_quant=True, quant=quant), get("q_norm.weight"), cfg.norm_eps)
    q = linear(qr, get("wq_b.weight"), act_quant=True, quant=quant).reshape(s, nh, hd)
    q = apply_rope_tail(q, cos[pos], sin[pos], rd)

    kv, idxs = window_kv(cfg, x, start_pos, cos[pos], sin[pos], wkv=get("wkv.weight"),
                         kv_norm_w=get("kv_norm.weight"), lc=lc, quant=quant)
    if ratio:
        offset = kv.shape[0]
        compress_len = (start_pos + s) // ratio
        latent = None
        if mode is Mode.FULL:
            latent = compressor(cfg, ratio, x, start_pos, wkv=get("compressor.wkv.weight"),
                                wgate=get("compressor.wgate.weight") if ratio > 1 else None,
                                norm_w=get("compressor.norm.weight"), lc=lc)
            shared.compress_owner = layer
        # -- which compressed positions each query attends to (the indexer needs the PRE-RoPE latent) --
        if mode in (Mode.FULL, Mode.REINDEX):
            if compress_len == 0:
                cidx = np.empty((s, 0), np.int64)
            else:
                is_full = mode is Mode.FULL
                cidx = indexer(cfg, layer, x, qr, latent, start_pos, cos, sin, wq_b=get("indexer.wq_b.weight"),
                               weights_proj=get("indexer.weights_proj.weight"),
                               wk=get("indexer.wk.weight") if is_full else None,
                               k_norm_w=get("indexer.k_norm.weight") if is_full else None,
                               lc=lc, shared=shared, caches=caches, quant=quant, stale_index_k=stale_index_k)
            shared.topk_idxs = cidx
        else:
            cidx = shared.topk_idxs
        if latent is not None:
            m = latent.shape[0]
            p_ = np.arange(0, s - s % ratio, ratio) if start_pos == 0 else np.array([start_pos + 1 - ratio])
            latent = apply_rope_tail(latent, cos[p_], sin[p_], rd)
            if quant.compressed_kv:
                latent = fp4_quant_e4m3(latent, 16)
            lc.comp_kv[start_pos // ratio: start_pos // ratio + m] = latent
        comp = caches[shared.compress_owner].comp_kv[:compress_len]
        kv = np.concatenate([kv, comp], axis=0)
        idxs = np.concatenate([idxs, np.where(cidx >= 0, cidx + offset, -1)], axis=1)

    o = sparse_attn(q, kv, get("attn_sink"), idxs, cfg.softmax_scale)
    o = apply_rope_tail(o, cos[pos], sin[pos], rd, inverse=True)

    # wo_a is block-diagonal over the o_groups (bf16 weight, not fp8: no activation quant); wo_b is a plain fp8 Linear
    g = cfg.o_groups
    o = o.reshape(s, g, -1)
    wo_a = get("wo_a.weight").reshape(g, cfg.o_lora_rank, -1)
    o = np.einsum("sgd,grd->sgr", o, wo_a)
    return linear(o.reshape(s, -1), get("wo_b.weight"), act_quant=True, quant=quant)
