"""ref/ds41/weights.py - weight providers: what the oracle's model reads its parameters from.

The model asks a `Weights` object for parameters by the OFFICIAL model.py state-dict names (the "canonical" names),
e.g. `layers.3.attn.wq_a.weight`, `layers.3.ffn.gate.bias`, `layers.3.hc_attn_fn`, `embed.weight`, and always gets a
dense float32 array [out, in] (dequantised).  Three kinds of access are lazy, because the real model does not fit
in RAM as float32:

    expert(layer, e)            the (w1, w3, w2) of ONE routed expert        (MXFP4 in the GGUF: 18.8 MB packed)
    engram_rows(layer, idx)     a few rows of an Engram table                (MXFP4, 136 B/row, 52 GB/table)
    rows(name, ids) / matmul_t(name, x)    embedding rows / the head, without materialising 129,280 x 5120 floats

`DictWeights`   an in-memory {name: ndarray} (tests; the exported tiny official model)
`GGUFWeights`   the mxxm-t GGUF (docs/deepseek/RESEARCH.md s10), mapped lazily from the shard files; see `load_from_gguf`
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import OrderedDict
from typing import Iterable, Mapping

import numpy as np

from . import quant as Q

# tools/gguf_reader.py is the repo's header reader; reuse it read-only
_TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"


def _gguf_reader():
    if str(_TOOLS) not in sys.path:
        sys.path.insert(0, str(_TOOLS))
    import gguf_reader  # noqa: WPS433  (read-only reuse of the repo's reader)
    return gguf_reader


class Weights:
    """Abstract provider (canonical names = official model.py state-dict names)."""

    def has(self, name: str) -> bool:
        raise NotImplementedError

    def get(self, name: str) -> np.ndarray:
        raise NotImplementedError

    def rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        return self.get(name)[ids]

    def matmul_t(self, name: str, x: np.ndarray) -> np.ndarray:
        """x @ W.T for a big matrix W (the output head)."""
        return x @ self.get(name).T

    def expert(self, layer: int, e: int) -> tuple:
        p = f"layers.{layer}.ffn.experts.{e}."
        return self.get(p + "w1.weight"), self.get(p + "w3.weight"), self.get(p + "w2.weight")

    def shared_expert(self, layer: int) -> tuple:
        p = f"layers.{layer}.ffn.shared_experts."
        return self.get(p + "w1.weight"), self.get(p + "w3.weight"), self.get(p + "w2.weight")

    def engram_rows(self, layer: int, idx: np.ndarray) -> np.ndarray:
        return self.get(f"layers.{layer}.engram.embed.weight")[idx]


class DictWeights(Weights):
    """{canonical name: ndarray}.  Routed experts either per expert (`layers.L.ffn.experts.E.w1.weight`) or stacked
    (`layers.L.ffn.experts.w1.weight` of shape [E, out, in])."""

    def __init__(self, tensors: Mapping[str, np.ndarray]):
        self.t = dict(tensors)

    def has(self, name):
        return name in self.t

    def get(self, name):
        return self.t[name]

    def expert(self, layer, e):
        p = f"layers.{layer}.ffn.experts."
        if p + "w1.weight" in self.t:
            return tuple(self.t[p + n + ".weight"][e] for n in ("w1", "w3", "w2"))
        return super().expert(layer, e)


# ---------------------------------------------------------------------------------------------------------------
# GGUF
# ---------------------------------------------------------------------------------------------------------------

# GGUF tensor name (without "blk.N.") -> canonical name (without "layers.N.")
_BLK_MAP = {
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "indexer_compressor_kv.weight": "attn.indexer.wk.weight",
    "indexer_compressor_norm.weight": "attn.indexer.k_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "hc_attn_fn.weight": "hc_attn_fn", "hc_attn_base.weight": "hc_attn_base", "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn", "hc_ffn_base.weight": "hc_ffn_base", "hc_ffn_scale.weight": "hc_ffn_scale",
    "engram_embed.weight": "engram.embed.weight",
    "engram_wkv.weight": "engram.wkv.weight",
    "engram_q.weight": "engram.q_weight",
    "engram_k.weight": "engram.k_weight",
}
# fused expert tensors: GGUF name -> official expert matrix name (w1 = gate, w3 = up, w2 = down)
_EXPERT_MAP = {"ffn_gate_exps.weight": "w1", "ffn_up_exps.weight": "w3", "ffn_down_exps.weight": "w2"}
_GLOBAL_MAP = {"token_embd.weight": "embed.weight", "output.weight": "head.weight", "output_norm.weight": "norm.weight"}
# the vcruz305 GGUF family spells a few tensors differently (docs/deepseek/RESEARCH.md s10)
_ALIASES = {"engram_embd.weight": "engram_embed.weight", "indexer.attn_k.weight": "indexer_compressor_kv.weight",
            "indexer.k_norm.weight": "indexer_compressor_norm.weight"}


def canonical_name(gguf_name: str) -> tuple[str, str | None]:
    """GGUF tensor name -> (canonical name, expert-matrix) ; the second item is 'w1'/'w2'/'w3' for fused expert
    tensors (whose canonical name is then the stacked `layers.N.ffn.experts.<w>.weight`), else None.  KeyError if unknown."""
    if gguf_name in _GLOBAL_MAP:
        return _GLOBAL_MAP[gguf_name], None
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if not m:
        raise KeyError(gguf_name)
    layer, rest = int(m.group(1)), _ALIASES.get(m.group(2), m.group(2))
    if rest in _EXPERT_MAP:
        return f"layers.{layer}.ffn.experts.{_EXPERT_MAP[rest]}.weight", _EXPERT_MAP[rest]
    return f"layers.{layer}.{_BLK_MAP[rest]}", None


_MAPS: dict = {}


def _map_file(path) -> np.ndarray:
    """One read-only byte mapping per shard file (shared by all its tensors)."""
    key = str(path)
    if key not in _MAPS:
        _MAPS[key] = np.memmap(path, dtype=np.uint8, mode="r")
    return _MAPS[key]


class _Tensor:
    """One GGUF tensor: type, ggml dims (ne0 first), absolute byte offset in its shard."""

    def __init__(self, path, info, data_start):
        self.path, self.info = path, info
        self.type = info.type_name
        self.ne = list(info.shape)                       # ne0 (fastest) first
        self.shape = tuple(reversed(self.ne))            # numpy (C order) shape
        self.offset = data_start + info.offset

    def _map(self):
        return _map_file(self.path)

    @property
    def block(self) -> tuple[int, int]:
        elems, nbytes = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "MXFP4": (32, 17)}[self.type]
        return elems, nbytes

    def row_bytes(self) -> int:
        elems, nbytes = self.block
        return self.ne[0] // elems * nbytes

    def raw(self, first_row: int = 0, n_rows: int | None = None, outer: int = 0) -> np.ndarray:
        """Raw bytes of `n_rows` rows (rows = slices along ne1) of outer slice `outer` (ne2), as [n_rows, row_bytes]."""
        rb = self.row_bytes()
        ne1 = self.ne[1] if len(self.ne) > 1 else 1
        n_rows = ne1 - first_row if n_rows is None else n_rows
        start = self.offset + (outer * ne1 + first_row) * rb
        return self._map()[start: start + n_rows * rb].reshape(n_rows, rb)

    def decode(self, raw: np.ndarray) -> np.ndarray:
        """[rows, row_bytes] -> float32 [rows, ne0]."""
        t = self.type
        if t == "F32":
            return np.ascontiguousarray(raw).view(np.float32).reshape(raw.shape[0], -1)
        if t == "F16":
            return np.ascontiguousarray(raw).view(np.float16).astype(np.float32).reshape(raw.shape[0], -1)
        if t == "BF16":
            return Q.bf16_to_f32(np.ascontiguousarray(raw).view(np.uint16)).reshape(raw.shape[0], -1)
        if t == "Q8_0":
            return Q.dequant_q8_0(raw)
        if t == "MXFP4":
            return Q.dequant_mxfp4(raw)
        raise NotImplementedError(f"GGUF type {t} is not supported by the oracle loader")

    def dense(self) -> np.ndarray:
        """Whole tensor as float32 in numpy shape (reversed ggml dims); 1-D tensors stay 1-D."""
        assert len(self.ne) <= 2, "use rows()/expert() for 3-D tensors"
        n = self.ne[1] if len(self.ne) > 1 else 1
        out = self.decode(self.raw(0, n))
        return out.reshape(self.shape) if len(self.ne) > 1 else out.reshape(self.ne[0])


class GGUFWeights(Weights):
    """Lazy view of the mxxm-t GGUF (one or more shards).  Nothing is read until asked: dense tensors are decoded on
    `get` (LRU-cached up to `cache_bytes`), experts one at a time, Engram rows by row index."""

    def __init__(self, paths: Iterable, *, cache_bytes: int = 2 << 30):
        rd = _gguf_reader()
        self.paths = [pathlib.Path(p) for p in paths]
        self.metadata: dict = {}
        self._t: dict[str, _Tensor] = {}           # canonical name -> tensor (fused experts under ...experts.w1 etc.)
        self.unmapped: list[str] = []
        for p in self.paths:
            g = rd.GGUFFile(p)
            for k, v in g.metadata.items():
                if k.startswith("split."):
                    continue
                self.metadata.setdefault(k, v)
            for info in g.tensors:
                try:
                    name, _ = canonical_name(info.name)
                except KeyError:
                    self.unmapped.append(info.name)
                    continue
                self._t[name] = _Tensor(p, info, g.data_start)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_bytes, self._cached = cache_bytes, 0

    # -- introspection
    def names(self) -> list[str]:
        return sorted(self._t)

    def tensor_info(self, name: str) -> _Tensor:
        return self._t[name]

    def has(self, name: str) -> bool:
        return name in self._t or re.sub(r"experts\.\d+\.", "experts.", name) in self._t

    # -- dense access
    def get(self, name: str) -> np.ndarray:
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        m = re.fullmatch(r"(layers\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.weight", name)
        if m:
            return self._expert_matrix(int(re.search(r"layers\.(\d+)", name).group(1)), int(m.group(2)), m.group(3))
        t = self._t[name]
        arr = t.dense()
        if arr.nbytes <= self._cache_bytes:
            self._cache[name] = arr
            self._cached += arr.nbytes
            while self._cached > self._cache_bytes:
                _, old = self._cache.popitem(last=False)
                self._cached -= old.nbytes
        return arr

    def rows(self, name: str, ids: np.ndarray) -> np.ndarray:
        t = self._t[name]
        ids = np.asarray(ids)
        out = np.empty((ids.size, t.ne[0]), np.float32)
        for i, r in enumerate(ids.reshape(-1)):
            out[i] = t.decode(t.raw(int(r), 1))[0]
        return out.reshape(*ids.shape, t.ne[0])

    def matmul_t(self, name: str, x: np.ndarray, chunk_rows: int = 8192) -> np.ndarray:
        t = self._t[name]
        n = t.ne[1]
        out = np.empty((*x.shape[:-1], n), np.promote_types(x.dtype, np.float32))
        for a in range(0, n, chunk_rows):
            b = min(n, a + chunk_rows)
            out[..., a:b] = x @ t.decode(t.raw(a, b - a)).T
        return out

    # -- lazy per-expert access
    def _expert_matrix(self, layer: int, e: int, which: str) -> np.ndarray:
        t = self._t[f"layers.{layer}.ffn.experts.{which}.weight"]
        assert len(t.ne) == 3 and 0 <= e < t.ne[2]
        return t.decode(t.raw(0, t.ne[1], outer=e))            # [out, in] float32

    def expert(self, layer: int, e: int) -> tuple:
        return tuple(self._expert_matrix(layer, e, w) for w in ("w1", "w3", "w2"))

    def expert_raw(self, layer: int, e: int) -> dict:
        """The three packed MXFP4 slices of expert e (uint8 [rows, row_bytes] views of the mapping): the exact bytes
        the C++ kernels consume (GPU blob = [gate][up][down])."""
        return {w: self._t[f"layers.{layer}.ffn.experts.{w}.weight"].raw(0, None, outer=e) for w in ("w1", "w3", "w2")}

    def engram_rows(self, layer: int, idx: np.ndarray) -> np.ndarray:
        return self.rows(f"layers.{layer}.engram.embed.weight", idx)


def expected_shape(cfg, name: str) -> tuple | None:
    """numpy shape the canonical parameter `name` must have for `cfg` (None for names this function does not know).
    The loader refuses a file whose tensors disagree (the geometry is a contract, as in geometry.hpp)."""
    d, hd, nh = cfg.dim, cfg.head_dim, cfg.n_heads
    hc, g = cfg.hc_mult, cfg.o_groups
    if name == "embed.weight" or name == "head.weight":
        return (cfg.vocab_size, d)
    if name == "norm.weight":
        return (d,)
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
    if not m:
        return None
    layer, rest = int(m.group(1)), re.sub(r"experts\.\d+\.", "experts.", m.group(2))
    table = {
        "attn_norm.weight": (d,), "ffn_norm.weight": (d,),
        "attn.wq_a.weight": (cfg.q_lora_rank, d), "attn.q_norm.weight": (cfg.q_lora_rank,),
        "attn.wq_b.weight": (nh * hd, cfg.q_lora_rank), "attn.wkv.weight": (hd, d), "attn.kv_norm.weight": (hd,),
        "attn.wo_a.weight": (g * cfg.o_lora_rank, nh * hd // g), "attn.wo_b.weight": (d, g * cfg.o_lora_rank),
        "attn.attn_sink": (nh,),
        "attn.compressor.wkv.weight": (hd, d), "attn.compressor.wgate.weight": (hd, d),
        "attn.compressor.norm.weight": (hd,),
        "attn.indexer.wq_b.weight": (cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank),
        "attn.indexer.weights_proj.weight": (cfg.index_n_heads, d), "attn.indexer.wk.weight": (cfg.index_head_dim, hd),
        "attn.indexer.k_norm.weight": (cfg.index_head_dim,),
        "ffn.gate.weight": (cfg.n_routed_experts, d), "ffn.gate.bias": (cfg.n_routed_experts,),
        "ffn.experts.w1.weight": (cfg.n_routed_experts, cfg.moe_inter_dim, d),
        "ffn.experts.w3.weight": (cfg.n_routed_experts, cfg.moe_inter_dim, d),
        "ffn.experts.w2.weight": (cfg.n_routed_experts, d, cfg.moe_inter_dim),
        "ffn.shared_experts.w1.weight": (cfg.moe_inter_dim, d), "ffn.shared_experts.w3.weight": (cfg.moe_inter_dim, d),
        "ffn.shared_experts.w2.weight": (d, cfg.moe_inter_dim),
        "engram.wkv.weight": ((hc + 1) * d, cfg.n_hash_cols * cfg.engram_head_dim),
        "engram.q_weight": (hc, d), "engram.k_weight": (hc, d),
    }
    for a in ("attn", "ffn"):
        table.update({f"hc_{a}_fn": ((2 + hc) * hc, hc * d), f"hc_{a}_base": ((2 + hc) * hc,), f"hc_{a}_scale": (3,)})
    if rest == "engram.embed.weight" and layer in cfg.engram_layer_ids:
        return (cfg.engram_num_embeddings[cfg.engram_layer_ids.index(layer)], cfg.engram_head_dim)
    return table.get(rest)


def validate_shapes(w: "GGUFWeights", cfg) -> list[str]:
    """Problems found comparing every mapped tensor's shape with `expected_shape` (empty list = consistent)."""
    problems = []
    for name in w.names():
        t = w.tensor_info(name)
        exp = expected_shape(cfg, name)
        if exp is None:
            problems.append(f"{name}: not a parameter of this model")
        elif tuple(t.shape) != tuple(exp):
            problems.append(f"{name}: file has {tuple(t.shape)}, config implies {tuple(exp)}")
    return problems


def load_from_gguf(paths, *, cache_bytes: int = 2 << 30, strict: bool = True) -> tuple["GGUFWeights", "Config"]:
    """Open the mxxm-t GGUF shard(s) lazily -> (weights, config).  `paths` is one path or a list (all shards of the
    split file, in any order; metadata lives in shard 0).  The Config comes from the `deepseek41.*` metadata.
    With strict=True (default) a tensor whose shape disagrees with that config, or that is not part of the model, raises
    (refuse rather than mis-index); `weights.unmapped` lists GGUF tensors with no canonical name (e.g. vision)."""
    from .config import Config
    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    w = GGUFWeights(paths, cache_bytes=cache_bytes)
    vocab = w.tensor_info("embed.weight").ne[1] if w.has("embed.weight") else None
    cfg = Config.from_gguf_metadata(w.metadata, vocab_size=vocab)
    if strict:
        problems = validate_shapes(w, cfg)
        if problems:
            raise ValueError("GGUF does not match its own deepseek41 metadata:\n  " + "\n  ".join(problems[:10]))
    return w, cfg
