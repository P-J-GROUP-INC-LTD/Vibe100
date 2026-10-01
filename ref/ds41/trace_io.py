"""ref/ds41/trace_io.py - the DS-1 trace: write / read a trace directory, and run the oracle with tracing.

The trace is what the C++ engine (`strata-ds41 --trace DIR`) and the oracle both write, and what `tools/ds41/ds1_compare.py`
compares (docs/deepseek/DS1.md section 6, docs/deepseek/DS1_VERIFY.md for the full description).  NumPy only.

LAYOUT.  One directory per run:

    trace.json                              metadata (below)
    <stage>.p<POS>.npy                      stages that exist once per position:        embed, final_hidden, logits
    <stage>.L<LL>.p<POS>.npy                stages that exist per (layer, position):    attn_in.L03.p00017.npy ...

POS is the 0-based position of the token in the sequence (5 digits, more are accepted when reading), LL the layer (2 digits).  Every
file is a standard .npy (version 1.0 or 2.0, C order), little-endian: `<f4` for float stages, `<i4` for the integer stages `topk`,
`router_idx`, `cand_blocks`.  ONE POSITION PER FILE (not one file per stage with a position axis) because the engine produces one token at
a time (prefill in DS-1 is the decode path), a crashed run still leaves every finished position readable, and layer-by-layer replay reads
single positions.  The stage table `STAGES` (names, shapes, dtypes, meaning, which are required of the engine) is the contract.

trace.json (all keys optional on the reader's side except `format`, `tokens`, `quant`; the engine fills what it knows):

    {"format": "ds41-trace", "version": 1, "producer": "engine" | "oracle", "geometry": "MiniGeom" | "RealGeom" | ...,
     "tokens": [t_0, t_1, ...],        every token of the sequence the trace covers (prompt, then generated tokens); position p's
                                       token is tokens[p]; positions present in the files are a subset of range(len(tokens))
     "n_prompt": n,                    how many of them were the prompt (the rest were sampled greedily by the producer)
     "quant": {"int8_act": true, "window_kv": true, "compressed_kv": true, "index": true, "linear_act": false},
     "positions": [0, 1, ...],         (written by TraceWriter.close(); the reader falls back to scanning the files)
     "mode": "token_by_token" | "prefill", "stale_index_k": false, "compute_dtype": "float32",     (oracle)
     "model": {n_layers, dim, ..., compress_ratios, layer_modes, ...}}                              (oracle; cross-checked if present)

`run_oracle_trace(model, ids, out_dir)` runs the oracle (`Model.forward(..., trace=)`) and writes this trace.  The stages the oracle's own
`trace=` dict lacks (attn_in, q, kv_win, latent, index_k, topk, ffn_in, the selection margins) are captured by wrapping the functions the
oracle calls (`attention.window_kv / sparse_attn / note_margin`, `model.attention_layer / moe`) for the duration of the call: the wrappers
call the originals with the same arguments and return their results untouched, so the numerics are bit-identical to an untraced run
(tools/ds41/test_ds1_trace.py asserts it).  The oracle's own files are not edited.  The wrapping is process-global: one traced run at a time.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
from typing import Iterable

import numpy as np

from .quant import QuantConfig

FORMAT = "ds41-trace"
VERSION = 1

# ---------------------------------------------------------------------------------------------------------------
# the stage table
# ---------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    layered: bool              # one file per (layer, position) instead of per position
    dtype: str                 # "f4" | "i4"
    kind: str                  # float | kvq (fake-quantised cache row) | set (integer set) | weights | logits | margin
    required: bool             # the engine must write it (DS1.md section 6); False = optional / oracle-only
    role: str                  # state (cache row: needed at every position <= p to replay position p), input, check, oracle
    doc: str


STAGES: tuple = (
    Stage("embed", False, "f4", "float", True, "input", "[dim] embedding row of the token at the position (the stream before layer 0 is 4 copies of it)"),
    Stage("engram_out", True, "f4", "float", True, "input", "[hc, dim] the mHC stream after the Engram layer (Engram layers only); the input of that layer's block"),
    Stage("attn_in", True, "f4", "float", True, "state", "[dim] attention input: hc_pre(stream, pre_mix) then attn_norm (kept at every position of a ratio>1 FULL layer: it rebuilds the compressor's group state)"),
    Stage("q", True, "f4", "float", True, "check", "[heads, head_dim] queries after wq_a, q_norm, wq_b and RoPE: the q the sparse attention reads"),
    Stage("kv_win", True, "f4", "kvq", True, "state", "[head_dim] the sliding-window KV row written at the position (kv_norm, RoPE on the tail, fp8 fake-quant when window_kv)"),
    Stage("latent", True, "f4", "kvq", True, "state", "[head_dim] the compressed-KV cache row published at this position, i.e. when a group of `ratio` tokens completes (FULL layers only): "
                                                      "RMSNorm, RoPE at group*ratio, fp4 fake-quant when compressed_kv: exactly what the cache holds"),
    Stage("index_k", True, "f4", "kvq", True, "state", "[index_head_dim] the index-K cache row published together with `latent` (FULL layers): RMSNorm of wk(PRE-RoPE latent), RoPE, fp4 fake-quant when index "
                                                         "(not recoverable from `latent`, which is post-RoPE and fp4: the layer replay of every indexer layer needs it)"),
    Stage("topk", True, "i4", "set", True, "input", "int32 [<= index_topk] the compressed-cache positions this layer attends to (ratio>0 layers, REUSE layers included): "
                                                    "ascending, entries < 0 are padding and are ignored by the reader; order is irrelevant"),
    Stage("attn_out", True, "f4", "float", True, "check", "[dim] attention sub-layer output (after wo_b), before hc_post"),
    Stage("ffn_in", True, "f4", "float", True, "check", "[dim] FFN input: hc_pre(stream, attn pre) then ffn_norm; what the router and the experts read"),
    Stage("router_idx", True, "i4", "set", True, "check", "int32 [top_k] the selected routed experts (order irrelevant)"),
    Stage("router_w", True, "f4", "weights", True, "check", "[top_k] the routing weights (normalised, times route_scale), in the order of router_idx"),
    Stage("ffn_out", True, "f4", "float", True, "check", "[dim] routed + shared expert output, before hc_post"),
    Stage("block_out", True, "f4", "float", True, "input", "[hc, dim] the mHC stream after the block; the input of the next layer"),
    Stage("pre_mix", True, "f4", "float", True, "input", "[hc] the `pre` mix the block hands to the next block (its FFN mixes: the single-pass lag)"),
    Stage("final_hidden", False, "f4", "float", True, "check", "[dim] final hc_pre + RMSNorm, the head's input"),
    Stage("logits", False, "f4", "logits", True, "check", "[vocab] the head's output"),
    # written by the oracle only (the engine need not)
    Stage("router_margin", True, "f4", "margin", False, "oracle", "[2] (k-th minus (k+1)-th best of s + bias, max |s + bias|): how close the router's choice was to a tie"),
    Stage("index_margin", True, "f4", "margin", False, "oracle", "[2] (k-th minus (k+1)-th best indexer score, max |score|) of the top-k selection; inf = no choice was made"),
    Stage("cand_margin", True, "f4", "margin", False, "oracle", "[2] the same for the candidate-pool block selection (candidate-source layer)"),
    Stage("cand_blocks", True, "i4", "set", False, "input", "int32 candidate-pool block ids at the candidate-source layer (optional; lets REINDEX layers be replayed with the engine's pool)"),
)
STAGE = {s.name: s for s in STAGES}
STAGE_ORDER = {s.name: i for i, s in enumerate(STAGES)}
PER_POSITION = tuple(s.name for s in STAGES if not s.layered)
PER_LAYER = tuple(s.name for s in STAGES if s.layered)

_FILE_RE = re.compile(r"^(?P<stage>[a-z_]+?)(?:\.L(?P<layer>\d+))?\.p(?P<pos>\d+)\.npy$")


class TraceError(Exception):
    """A trace directory that cannot be read, or a stage that is not there."""


def stage_filename(stage: str, pos: int, layer: int | None = None) -> str:
    st = STAGE.get(stage)
    if st is not None and st.layered != (layer is not None):
        raise TraceError(f"stage {stage!r} is {'per layer' if st.layered else 'global'}: layer={layer!r}")
    return f"{stage}.L{layer:02d}.p{pos:05d}.npy" if layer is not None else f"{stage}.p{pos:05d}.npy"


def quant_to_dict(q: QuantConfig) -> dict:
    return {f.name: bool(getattr(q, f.name)) for f in dataclasses.fields(QuantConfig)}


def quant_from_dict(d: dict | None) -> QuantConfig:
    names = {f.name for f in dataclasses.fields(QuantConfig)}
    return QuantConfig(**{k: bool(v) for k, v in (d or {}).items() if k in names})


# ---------------------------------------------------------------------------------------------------------------
# writer / reader
# ---------------------------------------------------------------------------------------------------------------


class TraceWriter:
    """Writes a trace directory.  `put()` per array; `close()` rewrites trace.json with the positions that exist."""

    def __init__(self, path, meta: dict | None = None):
        self.path = pathlib.Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.meta = {"format": FORMAT, "version": VERSION, **(meta or {})}
        self.meta.setdefault("tokens", [])
        self.meta.setdefault("quant", quant_to_dict(QuantConfig.exact()))
        self._positions: set = set()
        self._stages: set = set()
        self._write_json()

    def _write_json(self):
        (self.path / "trace.json").write_text(json.dumps(self.meta, indent=1, sort_keys=True))

    def put(self, stage: str, pos: int, array, layer: int | None = None) -> None:
        st = STAGE.get(stage)
        dt = np.dtype("<i4") if (st is not None and st.dtype == "i4") else np.dtype("<f4")
        a = np.asarray(array)
        if dt.kind == "i":
            if a.size and (a.max() > np.iinfo(np.int32).max or a.min() < np.iinfo(np.int32).min):
                raise TraceError(f"{stage}: value does not fit int32")
        np.save(self.path / stage_filename(stage, int(pos), layer), np.ascontiguousarray(a, dtype=dt), allow_pickle=False)
        self._positions.add(int(pos))
        self._stages.add(stage)

    def close(self) -> "TraceWriter":
        self.meta["positions"] = sorted(self._positions)
        self.meta["stages"] = sorted(self._stages, key=lambda n: STAGE_ORDER.get(n, 999))
        self._write_json()
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Trace:
    """A trace directory opened for reading."""

    def __init__(self, path):
        self.path = pathlib.Path(path)
        jp = self.path / "trace.json"
        if not jp.is_file():
            raise TraceError(f"{self.path}: no trace.json (not a trace directory, or the run did not start)")
        try:
            self.meta = json.loads(jp.read_text())
        except ValueError as e:
            raise TraceError(f"{jp}: not valid JSON ({e})") from e
        if self.meta.get("format") != FORMAT:
            raise TraceError(f"{jp}: format {self.meta.get('format')!r}, expected {FORMAT!r}")
        if int(self.meta.get("version", 1)) > VERSION:
            raise TraceError(f"{jp}: version {self.meta.get('version')} is newer than this reader ({VERSION})")
        self._files: dict = {}
        with os.scandir(self.path) as it:
            for e in it:
                m = _FILE_RE.match(e.name)
                if m:
                    layer = None if m.group("layer") is None else int(m.group("layer"))
                    self._files[(m.group("stage"), layer, int(m.group("pos")))] = e.path
        self._by_stage: dict = {}
        for (stage, layer, pos) in self._files:
            self._by_stage.setdefault(stage, set()).add((layer, pos))

    # -- metadata
    @property
    def tokens(self) -> list:
        return [int(t) for t in self.meta.get("tokens", [])]

    @property
    def n_prompt(self) -> int:
        return int(self.meta.get("n_prompt", len(self.tokens)))

    @property
    def producer(self) -> str:
        return str(self.meta.get("producer", "?"))

    def quant(self) -> QuantConfig:
        return quant_from_dict(self.meta.get("quant"))

    def positions(self) -> list:
        return sorted({p for (_, _, p) in self._files})

    def stages(self) -> set:
        return set(self._by_stage)

    def layers_of(self, stage: str) -> list:
        return sorted({l for (l, _) in self._by_stage.get(stage, ()) if l is not None})

    def positions_of(self, stage: str, layer: int | None = None) -> list:
        return sorted(p for (l, p) in self._by_stage.get(stage, ()) if l == layer)

    # -- arrays
    def has(self, stage: str, pos: int, layer: int | None = None) -> bool:
        return (stage, layer, int(pos)) in self._files

    def path_of(self, stage: str, pos: int, layer: int | None = None) -> str | None:
        return self._files.get((stage, layer, int(pos)))

    def get(self, stage: str, pos: int, layer: int | None = None):
        p = self._files.get((stage, layer, int(pos)))
        if p is None:
            return None
        try:
            return np.load(p, allow_pickle=False)
        except Exception as e:                           # truncated write (the engine crashed mid-file)
            raise TraceError(f"{p}: unreadable .npy ({e})") from e

    def need(self, stage: str, pos: int, layer: int | None = None):
        a = self.get(stage, pos, layer)
        if a is None:
            raise TraceError(f"{self.path}: no {stage_filename(stage, pos, layer)}")
        return a


# ---------------------------------------------------------------------------------------------------------------
# the logits dump of tools/volta/golden_compare.py (int32 n_vocab, int32 n_rows, float32 rows)
# ---------------------------------------------------------------------------------------------------------------


def write_logits_dump(path, rows) -> None:
    """The engine's `--dump-logits` format, so `golden_compare.py --ref-logits` can read oracle logits."""
    rows = np.ascontiguousarray(rows, dtype="<f4")
    with open(path, "wb") as f:
        np.array([rows.shape[1], rows.shape[0]], dtype="<i4").tofile(f)
        rows.tofile(f)


def read_logits_dump(path) -> np.ndarray:
    """-> float32 [rows, n_vocab]; the row count comes from the file size (an engine run with --prefill-until leaves the header's count too high)."""
    size = os.path.getsize(path)
    if size < 8:
        raise TraceError(f"{path}: {size} bytes, too short for a logits dump")
    hdr = np.fromfile(path, dtype="<i4", count=2)
    vocab = int(hdr[0])
    if vocab <= 0 or (size - 8) % (4 * vocab):
        raise TraceError(f"{path}: {size - 8} data bytes is not a whole number of {vocab}-float rows")
    return np.fromfile(path, dtype="<f4", offset=8).reshape(-1, vocab)


# ---------------------------------------------------------------------------------------------------------------
# running the oracle with tracing
# ---------------------------------------------------------------------------------------------------------------


def model_summary(model) -> dict:
    """The oracle config fields a comparison tool needs, plus the per-layer mode map."""
    c = model.cfg
    keys = ("n_layers", "dim", "hc_mult", "n_heads", "head_dim", "rope_head_dim", "window_size", "index_n_heads", "index_head_dim",
            "index_topk", "vocab_size", "n_activated_experts", "n_routed_experts", "moe_inter_dim", "candidate_source_layer",
            "candidate_block_size", "candidate_topk_blocks", "o_groups", "o_lora_rank", "q_lora_rank")
    d = {k: int(getattr(c, k)) for k in keys}
    d["compress_ratios"] = [int(r) for r in c.compress_ratios[: c.n_layers]]
    d["kv_source_layers"] = [int(x) for x in c.kv_source_layers]
    d["index_source_layers"] = [int(x) for x in c.index_source_layers]
    d["engram_layer_ids"] = [int(x) for x in c.engram_layer_ids]
    d["layer_modes"] = [m.value for m in model.modes]
    return d


def _margin_pair(scores: np.ndarray, k: int) -> tuple:
    """Per row: (k-th best - (k+1)-th best finite score, max |finite score|); the gap is inf when there was no choice to make."""
    s = np.asarray(scores, dtype=np.float64)
    rows = s.shape[0]
    fin = np.isfinite(s)
    scale = np.max(np.where(fin, np.abs(s), 0.0), axis=-1) if s.shape[-1] else np.zeros(rows)
    if s.shape[-1] <= k or k < 1:
        return np.full(rows, np.inf), scale
    top = -np.sort(-s, axis=-1)
    a, b = top[:, k - 1], top[:, k]
    ok = np.isfinite(a) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        return np.where(ok, a - b, np.inf), scale


class _Capture:
    """Records, per layer, what the oracle's helper functions see during one `Model.forward` call (see the module docstring)."""

    def __init__(self, model):
        from . import attention as A, model as M
        self.model, self.A, self.M = model, A, M
        self.per_layer: dict = {}
        self.cur: dict | None = None
        self.orig: dict = {}

    # -- patching
    def __enter__(self):
        A, M, model = self.A, self.M, self.model
        cap = self
        self.orig = {"window_kv": A.window_kv, "sparse_attn": A.sparse_attn, "note_margin": A.note_margin,
                     "attention_layer": M.attention_layer, "moe": M.moe, "router": M.router}
        o = self.orig

        def window_kv(cfg, x, start_pos, cos, sin, **kw):
            out = o["window_kv"](cfg, x, start_pos, cos, sin, **kw)
            rows = out[0]
            cap.cur["kv_new"] = np.array(rows, copy=True) if start_pos == 0 else np.array(rows[start_pos % cfg.window_size], copy=True)[None]
            return out

        def sparse_attn(q, kv, sink, idxs, scale, **kw):
            cap.cur["q"] = np.array(q, copy=True)
            return o["sparse_attn"](q, kv, sink, idxs, scale, **kw)

        def note_margin(shared, key, scores, k):
            o["note_margin"](shared, key, scores, k)
            cap.cur.setdefault("margins", {})[key.split(".")[0]] = _margin_pair(scores, k)

        def attention_layer(cfg, layer, mode, x, start_pos, rope, get, *, lc, caches, shared, quant, stale_index_k=False):
            cap.cur = cap.per_layer[layer] = {"x": np.array(x, copy=True), "start": start_pos, "s": x.shape[0]}
            out = o["attention_layer"](cfg, layer, mode, x, start_pos, rope, get, lc=lc, caches=caches, shared=shared, quant=quant,
                                       stale_index_k=stale_index_k)
            ratio = cfg.compress_ratios[layer]
            s = x.shape[0]
            if ratio:
                cap.cur["topk"] = [r[r >= 0].astype(np.int32) for r in shared.topk_idxs]
                if mode.value == "full":
                    # the cache rows published by this call: group j completes at position (j + 1) * ratio - 1
                    done = range(start_pos // ratio, (start_pos + s) // ratio)
                    cap.cur["pub"] = {(j + 1) * ratio - 1: (np.array(lc.comp_kv[j], copy=True), np.array(lc.index_k[j], copy=True)) for j in done}
            return out

        def moe(y, **kw):
            cap.cur["ffn_in"] = np.array(y, copy=True)
            return o["moe"](y, **kw)

        def router(x, gate_w, bias, **kw):
            r = o["router"](x, gate_w, bias, **kw)
            cap.cur["router_scale"] = np.max(np.abs(np.asarray(r.scores, dtype=np.float64) + np.asarray(bias, dtype=np.float64)), axis=-1)
            return r

        A.window_kv, A.sparse_attn, A.note_margin = window_kv, sparse_attn, note_margin
        M.attention_layer, M.moe, M.router = attention_layer, moe, router
        return self

    def __exit__(self, *exc):
        A, M, o = self.A, self.M, self.orig
        A.window_kv, A.sparse_attn, A.note_margin = o["window_kv"], o["sparse_attn"], o["note_margin"]
        M.attention_layer, M.moe, M.router = o["attention_layer"], o["moe"], o["router"]


def _emit_call(w: TraceWriter, model, cap: _Capture, tr: dict, ids, start_pos: int, logits) -> None:
    """Write the rows of ONE `Model.forward` call (s = len(ids) positions starting at start_pos)."""
    cfg, s = model.cfg, len(ids)
    emb = np.asarray(model.w.rows("embed.weight", ids), dtype=np.float32)
    for t in range(s):
        w.put("embed", start_pos + t, emb[t])
    for l in range(cfg.n_layers):
        c = cap.per_layer[l]
        ratio = cfg.compress_ratios[l]
        for t in range(s):
            p = start_pos + t
            if l in tr.get("engram_out", {}):
                w.put("engram_out", p, tr["engram_out"][l][t], l)
            w.put("attn_in", p, c["x"][t], l)
            w.put("q", p, c["q"][t], l)
            w.put("kv_win", p, c["kv_new"][t], l)
            if ratio:
                w.put("topk", p, c["topk"][t], l)
                if p in c.get("pub", {}):
                    lat, ik = c["pub"][p]
                    w.put("latent", p, lat, l)
                    w.put("index_k", p, ik, l)
            w.put("attn_out", p, tr["attn_out"][l][t], l)
            w.put("ffn_in", p, c["ffn_in"][t], l)
            w.put("router_idx", p, tr["router_idx"][l][t], l)
            w.put("router_w", p, tr["router_w"][l][t], l)
            w.put("ffn_out", p, tr["ffn_out"][l][t], l)
            w.put("block_out", p, tr["block_out"][l][t], l)
            w.put("pre_mix", p, tr["pre_mix"][l][t], l)
            # selection margins (oracle only)
            w.put("router_margin", p, np.array([tr["router_margin"][l][t], c["router_scale"][t]], dtype=np.float32), l)
            mg = c.get("margins", {})
            if "index" in mg:
                w.put("index_margin", p, np.array([mg["index"][0][t], mg["index"][1][t]], dtype=np.float32), l)
            if "blocks" in mg:
                w.put("cand_margin", p, np.array([mg["blocks"][0][t], mg["blocks"][1][t]], dtype=np.float32), l)
    fh = np.asarray(tr["final_hidden"], dtype=np.float32)
    lg = np.asarray(logits, dtype=np.float32).reshape(s, -1)
    for t in range(s):
        w.put("final_hidden", start_pos + t, fh[t])
        w.put("logits", start_pos + t, lg[t])


def run_oracle_trace(model, ids, out_dir, *, mode: str = "token_by_token", n_prompt: int | None = None, geometry: str = "oracle",
                     extra_meta: dict | None = None) -> Trace:
    """Run `model` (a `ref.ds41.Model`) over the token ids with tracing and write the trace directory.

    mode "token_by_token" (default): position 0 is a one-token prefill, every later position a decode step - what the engine does in DS-1
    ("prefill = the prompt fed one token at a time through the decode path"), and it gives per-position selection margins.
    mode "prefill": one `forward` over all ids (faster for many tokens; equal to the above to float round-off, CONTRACTS.md index-K decision).
    The tokens are teacher-forced: nothing is sampled; `n_prompt` only records how many of them the producer treated as the prompt."""
    if mode not in ("token_by_token", "prefill"):
        raise ValueError(f"mode {mode!r}: token_by_token | prefill")
    ids = [int(t) for t in np.asarray(ids).reshape(-1)]
    if not ids:
        raise ValueError("no tokens")
    meta = {"producer": "oracle", "geometry": geometry, "mode": mode, "tokens": ids, "n_prompt": int(n_prompt if n_prompt is not None else len(ids)),
            "quant": quant_to_dict(model.quant), "stale_index_k": bool(model.stale_index_k), "compute_dtype": str(model.dt),
            "model": model_summary(model)}
    meta.update(extra_meta or {})
    w = TraceWriter(out_dir, meta)
    cache = model.new_cache()
    with _Capture(model) as cap:
        if mode == "prefill":
            tr: dict = {}
            logits = model.forward(np.array(ids), 0, cache, full_logits=True, trace=tr)
            _emit_call(w, model, cap, tr, ids, 0, logits)
        else:
            for p, t in enumerate(ids):
                tr = {}
                cap.per_layer.clear()
                logits = model.forward(np.array([t]), p, cache, trace=tr)
                _emit_call(w, model, cap, tr, [t], p, logits)
    w.close()
    return Trace(out_dir)


def greedy_continue(model, prompt: Iterable[int], n_new: int, *, token_by_token: bool = True) -> list:
    """The oracle's greedy continuation of `prompt`.  token_by_token=True feeds the prompt one token at a time through the decode path (what the engine does in DS-1);
    False is `Model.generate_greedy` (one prefill).  The two differ only by float round-off in the logits, which can flip an argmax near a tie."""
    prompt = [int(t) for t in prompt]
    if not token_by_token:
        return [int(t) for t in model.generate_greedy(prompt, n_new)]
    cache = model.new_cache()
    logits = None
    for p, t in enumerate(prompt):
        logits = model.forward(np.array([t]), p, cache)
    out = []
    for _ in range(n_new):
        t = int(np.argmax(logits))
        out.append(t)
        logits = model.forward(np.array([t]), cache.pos, cache)
    return out
