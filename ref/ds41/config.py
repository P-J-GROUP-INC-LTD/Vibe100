"""ref/ds41/config.py - model hyper-parameters and the per-layer attention-mode map.

`Config` carries the field names of the official `ModelArgs` (third_party/deepseek-v41-flash-reference/inference/
model.py:45-137) so that `Config(**official_args_dict)` works, plus a few oracle-only knobs (`max_seq_len`).

Sources of a Config:
  * `Config.from_official_json(path)`   the vendored inference/config.json (the released model);
  * `Config.from_gguf_metadata(meta)`   the `deepseek41.*` metadata of the target GGUF (docs/deepseek/RESEARCH.md s10);
  * `Config.tiny(...)`                  a structure-preserving miniature for tests (see tests/official.py).

`layer_modes(cfg)` is the layer-mode map of docs/deepseek/RESEARCH.md s2:

    SWA      compress_ratio == 0: sliding window only (layers 0, 1)
    FULL     a KV source (kv_source_layers): own compressor + own indexer (+ index-K cache) -> global KV
    REINDEX  an index source that is not a KV source: own indexer queries, KV and index-K from the latest FULL layer
    REUSE    neither: re-uses the compressed KV *and* the top-k indices of the latest index source

Nothing here imports torch.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Mapping


class Mode(enum.Enum):
    SWA = "swa"
    FULL = "full"
    REUSE = "reuse"
    REINDEX = "reindex"


def _tup(x) -> tuple:
    return tuple(x) if x is not None else ()


@dataclass
class Config:
    # ---- model.py:ModelArgs field names (released-model values as defaults) ----
    vocab_size: int = 129280
    dim: int = 5120
    moe_inter_dim: int = 2304
    n_layers: int = 40
    n_heads: int = 64
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    score_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    q_lora_rank: int = 1280
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: tuple = (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)
    kv_source_layers: tuple = (2, 8, 14, 20)
    index_source_layers: tuple = (2, 8, 14, 20, 24, 28, 32, 36)
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_theta: float = 10000.0
    rope_factor: float = 16
    beta_fast: int = 32
    beta_slow: int = 1
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512
    candidate_source_layer: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    engram_layer_ids: tuple = (1, 14)
    engram_num_embeddings: tuple = (384006168, 384016682)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092
    # ---- oracle-only ----
    max_seq_len: int = 4096          # sizes the caches only; not a model property

    def __post_init__(self):
        for f in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids",
                  "engram_num_embeddings"):
            setattr(self, f, _tup(getattr(self, f)))
        self.validate()

    # ------------------------------------------------------------------ derived
    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim ** -0.5

    @property
    def n_hash_cols(self) -> int:          # engram.py: (max_ngram_size - 1) * n_heads rows per token per layer
        return (self.engram_max_ngram_size - 1) * self.engram_n_heads

    def validate(self) -> None:
        assert len(self.compress_ratios) >= self.n_layers, "compress_ratios needs one entry per layer"
        assert self.head_dim > self.rope_head_dim and self.rope_head_dim % 2 == 0
        assert self.n_heads % self.o_groups == 0
        for l in self.kv_source_layers:
            assert self.compress_ratios[l] > 0, f"kv source layer {l} must compress"
            assert l in self.index_source_layers, f"kv source layer {l} must also own an indexer"
        for l in self.index_source_layers:
            assert self.compress_ratios[l] > 0, f"index source layer {l} must compress"
        if self.candidate_source_layer >= 0:
            assert self.candidate_source_layer in self.index_source_layers
            assert self.candidate_topk_blocks > 0 and self.candidate_block_size > 0

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Config":
        """Accepts the official inference/config.json (unknown keys such as vision_* / dspark_* are ignored; the
        HF-style spellings candidate_source_layer_id / engram_pad_token_id are accepted too)."""
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        if "candidate_source_layer" not in kw and "candidate_source_layer_id" in d:
            kw["candidate_source_layer"] = d["candidate_source_layer_id"]
        if "engram_pad_id" not in kw and "engram_pad_token_id" in d:
            kw["engram_pad_id"] = d["engram_pad_token_id"]
        return cls(**kw)

    @classmethod
    def from_official_json(cls, path) -> "Config":
        """The released model: third_party/deepseek-v41-flash-reference/inference/config.json."""
        d = json.loads(pathlib.Path(path).read_text())          # compress_ratios has 43 entries: 40 layers + 3 DSpark
        return cls.from_dict(d)

    @classmethod
    def from_gguf_metadata(cls, meta: Mapping[str, Any], *, vocab_size: int | None = None, max_seq_len: int = 4096
                           ) -> "Config":
        """The `deepseek41.*` keys of the mxxm-t GGUF (docs/deepseek/RESEARCH.md s10)."""
        p = "deepseek41."
        g = lambda k, default=None: meta.get(p + k, default)
        assert g("expert_gating_func") in (None, 4), "only the sqrtsoftplus gate (ggml func 4) is implemented"
        clamp = g("swiglu_clamp_exp")
        assert clamp is None or len(set(clamp)) == 1, "per-layer swiglu clamps differ: not supported"
        kw: dict[str, Any] = dict(
            vocab_size=vocab_size or len(meta.get("tokenizer.ggml.tokens", [])) or 129280,
            n_layers=g("block_count"), dim=g("embedding_length"), n_heads=g("attention.head_count"),
            head_dim=g("attention.key_length"), rope_head_dim=g("rope.dimension_count"),
            q_lora_rank=g("attention.q_lora_rank"), window_size=g("attention.sliding_window"),
            n_routed_experts=g("expert_count"), n_activated_experts=g("expert_used_count"),
            n_shared_experts=g("expert_shared_count"), moe_inter_dim=g("expert_feed_forward_length"),
            route_scale=g("expert_weights_scale"), norm_topk_prob=bool(g("expert_weights_norm", True)),
            swiglu_limit=clamp[0] if clamp else 0.0, norm_eps=g("attention.layer_norm_rms_epsilon"),
            o_groups=g("attention.output_group_count"), o_lora_rank=g("attention.output_lora_rank"),
            compress_ratios=g("attention.compress_ratios"), compress_rope_theta=g("attention.compress_rope_freq_base"),
            rope_theta=g("rope.freq_base"), rope_factor=g("rope.scaling.factor"),
            original_seq_len=g("rope.scaling.original_context_length"),
            beta_fast=g("rope.scaling.yarn_beta_fast"), beta_slow=g("rope.scaling.yarn_beta_slow"),
            hc_mult=g("hyper_connection.count"), hc_sinkhorn_iters=g("hyper_connection.sinkhorn_iterations"),
            hc_eps=g("hyper_connection.epsilon"),
            kv_source_layers=g("attention.kv_source_layer_ids"),
            index_source_layers=g("attention.index_source_layer_ids"),
            index_n_heads=g("attention.indexer.head_count"), index_head_dim=g("attention.indexer.key_length"),
            index_topk=g("attention.indexer.top_k"),
            candidate_source_layer=g("attention.candidate_source_layer_id"),
            candidate_topk_blocks=g("attention.candidate_topk_blocks"),
            candidate_block_size=g("attention.candidate_block_size"),
            engram_layer_ids=g("engram.layer_ids"), engram_num_embeddings=g("engram.num_embeddings"),
            engram_max_ngram_size=g("engram.max_ngram_size"), engram_n_heads=g("engram.n_heads"),
            engram_head_dim=g("engram.head_dim"), engram_pad_id=g("engram.pad_token_id"),
            engram_compressed_vocab_size=g("engram.compressed_vocab_size"), max_seq_len=max_seq_len,
        )
        missing = [k for k, v in kw.items() if v is None]
        assert not missing, f"GGUF metadata lacks {missing}"
        kw["rope_factor"] = float(kw["rope_factor"])
        kw["beta_fast"], kw["beta_slow"] = int(kw["beta_fast"]), int(kw["beta_slow"])
        # the GGUF has no `engram.vocab_size`; it only seeds the prime search (primes are stored explicitly)
        return cls(**kw)

    @classmethod
    def tiny(cls, **over) -> "Config":
        """A structure-preserving miniature (all four layer modes, both compressor ratios, the candidate pool forced
        on, hc_mult 4, two Engram layers).  `tests/official.py` builds the matching official ModelArgs."""
        d = dict(
            vocab_size=512, dim=64, moe_inter_dim=64, n_layers=10, n_heads=4, n_routed_experts=16,
            n_activated_experts=2, route_scale=1.5, swiglu_limit=10.0, q_lora_rank=64, head_dim=64,
            rope_head_dim=16, o_groups=2, o_lora_rank=32, window_size=8,
            compress_ratios=(0, 0, 2, 2, 2, 2, 1, 1, 1, 1),
            kv_source_layers=(2, 4, 6), index_source_layers=(2, 4, 6, 8),
            compress_rope_theta=160000.0, original_seq_len=32, rope_theta=10000.0, rope_factor=4.0,
            index_n_heads=8, index_head_dim=32, index_topk=4,
            candidate_source_layer=6, candidate_topk_blocks=4, candidate_block_size=2,
            engram_layer_ids=(1, 5), engram_num_embeddings=(0, 0), engram_max_ngram_size=4,
            engram_vocab_size=1000, engram_n_heads=2, engram_head_dim=32, engram_pad_id=2,
            engram_compressed_vocab_size=0, max_seq_len=64,
        )
        d.update(over)
        return cls(**d)


def layer_modes(cfg: Config) -> list[Mode]:
    """The layer-mode map (see module docstring); one Mode per backbone layer."""
    out = []
    for l in range(cfg.n_layers):
        if cfg.compress_ratios[l] == 0:
            out.append(Mode.SWA)
        elif l in cfg.kv_source_layers:
            out.append(Mode.FULL)
        elif l in cfg.index_source_layers:
            out.append(Mode.REINDEX)
        else:
            out.append(Mode.REUSE)
    return out


def describe_modes(cfg: Config) -> str:
    return " ".join(f"{l}:{m.value}" for l, m in enumerate(layer_modes(cfg)))
