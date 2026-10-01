"""ds41_spec.py - the `deepseek41` GGUF contract: which metadata keys, layer modes and tensors the port needs.

Sources: the headers of mxxm-t/DeepSeek-V4.1-Flash-GGUF (docs/deepseek/RESEARCH.md section 10),
include/strata/ds41/geometry.hpp and the reference config.json.  Everything below is a table; manifest.py walks it.

Layer modes (docs/deepseek/RESEARCH.md section 2), from compress_ratios / kv_source_layer_ids / index_source_layer_ids:
    SWA      compress ratio 0                           128-token window only
    Full     layer in kv_source_layer_ids               owns a compressor (kv + norm; gate only at ratio 2) -> global KV
    Reindex  in index_source_layer_ids, not kv source   own indexer q / weights, reuses the owner's KV and index-K
    Reuse    everything else with a ratio               reuses the owner's global KV
    An indexer (attn_q_b + proj) sits on every index-source layer; the indexer's own compressor (index-K) on the
    layers that are both kv and index sources.

Two geometry modes: "ds41" requires the compile-time geometry of geometry.hpp (5120 hidden, 384 experts, ...), "self"
takes every dimension from the file's own metadata (the mini test GGUF) and checks only the structure.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any

ARCH = "deepseek41"
SIDECAR_ARCH = "dflash"
MISSING = object()

# ---- config.json text_config defaults (a unit test pins these against third_party/.../config.json) ----------------
DEFAULT_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]     # 40 layers + 3 DSpark entries
DEFAULT_KV_SOURCE = [2, 8, 14, 20]
DEFAULT_INDEX_SOURCE = [2, 8, 14, 20, 24, 28, 32, 36]
DEFAULT_ENGRAM_LAYERS = [1, 14]
ENGRAM_VOCAB_SIZE = 16_000_000          # config engram_vocab_size: Engram primes are the next primes above this - 1
INT64_MAX = 2 ** 63 - 1


@dataclasses.dataclass(frozen=True)
class Key:
    name: str                  # relative to "<arch>.", or absolute when it starts with "general." / "tokenizer."
    field: str                 # Cfg attribute it fills
    expect: Any = MISSING      # value the compile-time geometry requires ("ds41" mode); MISSING = no check
    default: Any = MISSING     # config.json value used (with a warning) when the key is absent; MISSING = required
    aliases: tuple = ()        # other spellings seen in published GGUFs (relative names)
    kind: str = "num"          # num | str | bool | list


KEYS = [
    Key("block_count", "n_layer", 40, 40),
    Key("context_length", "context", 1048576, 1048576),
    Key("embedding_length", "hidden", 5120, 5120),
    Key("attention.head_count", "n_head", 64, 64),
    Key("attention.head_count_kv", "n_kv_head", 1, 1),
    Key("rope.scaling.type", "rope_type", "yarn", "yarn", kind="str"),
    Key("rope.scaling.factor", "rope_factor", 16.0, 16.0),
    Key("rope.scaling.original_context_length", "rope_orig_ctx", 65536, 65536),
    Key("rope.scaling.yarn_beta_fast", "yarn_beta_fast", 32.0, 32.0),
    Key("rope.scaling.yarn_beta_slow", "yarn_beta_slow", 1.0, 1.0),
    Key("rope.freq_base", "rope_base", 10000.0, 10000.0),
    Key("attention.layer_norm_rms_epsilon", "rms_eps", 1e-20, 1e-20),
    Key("expert_count", "n_expert", 384, 384),
    Key("expert_used_count", "n_used", 6, 6),
    Key("expert_gating_func", "gating", 4, 4),
    Key("attention.key_length", "head_dim", 512, 512),
    Key("attention.value_length", "value_dim", 512, 512),
    Key("rope.dimension_count", "rope_dim", 64, 64),
    Key("attention.q_lora_rank", "q_lora", 1280, 1280),
    Key("attention.sliding_window", "sliding_window", 128, 128),
    Key("expert_feed_forward_length", "ff", 2304, 2304),
    Key("expert_shared_count", "n_shared", 1, 1),
    Key("expert_weights_scale", "route_scale", 1.5, 1.5),
    Key("expert_weights_norm", "route_norm", True, True, kind="bool"),
    Key("swiglu_clamp_exp", "clamp_exp", MISSING, MISSING, kind="list"),     # [10.0] * n_layer, checked in code
    Key("swiglu_clamp_shexp", "clamp_shexp", MISSING, MISSING, kind="list"),
    Key("attention.indexer.head_count", "idx_heads", 32, 32),
    Key("attention.indexer.key_length", "idx_dim", 128, 128),
    Key("attention.indexer.top_k", "idx_topk", 512, 512),
    Key("attention.output_group_count", "o_groups", 8, 8),
    Key("attention.output_lora_rank", "o_lora", 1024, 1024),
    Key("attention.compress_ratios", "ratios", DEFAULT_RATIOS, DEFAULT_RATIOS, kind="list"),
    Key("attention.compress_rope_freq_base", "compress_rope_base", 160000.0, 160000.0),
    Key("hyper_connection.count", "hc", 4, 4),
    Key("hyper_connection.sinkhorn_iterations", "hc_iters", 20, 20),
    Key("hyper_connection.epsilon", "hc_eps", 1e-6, 1e-6),
    Key("hash_layer_count", "hash_layers", 0, 0),
    Key("attention.kv_source_layer_ids", "kv_source", DEFAULT_KV_SOURCE, DEFAULT_KV_SOURCE, kind="list"),
    Key("attention.index_source_layer_ids", "index_source", DEFAULT_INDEX_SOURCE, DEFAULT_INDEX_SOURCE, kind="list"),
    Key("attention.candidate_source_layer_id", "cand_source", 20, 20),
    Key("attention.candidate_block_size", "cand_block", 8, 8),
    Key("attention.candidate_topk_blocks", "cand_topk", 2048, 2048),
    Key("engram.layer_ids", "engram_layers", DEFAULT_ENGRAM_LAYERS, DEFAULT_ENGRAM_LAYERS, kind="list"),
    Key("engram.head_dim", "eh_dim", 256, 256, aliases=("engram.key_length",)),
    Key("engram.n_heads", "e_heads", 8, 8, aliases=("engram.head_count",)),
    Key("engram.max_ngram_size", "e_ngram", 4, 4),
    Key("engram.pad_token_id", "e_pad", 2, 2, aliases=("engram.pad_id",)),
    Key("engram.compressed_vocab_size", "e_cvocab", 99092, 99092),
    # engram.num_embeddings (derived from the table shapes when absent), engram.primes / offsets / multipliers /
    # token_map are handled in code
]

TOKENIZER_VOCAB = 129280


# ---------------------------------------------------------------------------------------------- findings
@dataclasses.dataclass
class Finding:
    level: str      # "error" | "warn" | "info"
    code: str
    text: str


class Findings:
    def __init__(self):
        self.items: list[Finding] = []

    def add(self, level, code, text):
        self.items.append(Finding(level, code, text))

    def error(self, code, text):
        self.add("error", code, text)

    def warn(self, code, text):
        self.add("warn", code, text)

    def info(self, code, text):
        self.add("info", code, text)

    @property
    def errors(self):
        return [f for f in self.items if f.level == "error"]

    @property
    def warnings(self):
        return [f for f in self.items if f.level == "warn"]

    def has(self, code, level=None):
        return any(f.code == code and (level is None or f.level == level) for f in self.items)


def _close(a, b, rel=1e-6):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)
    return a == b


# ---------------------------------------------------------------------------------------------- metadata -> Cfg
def resolve_config(kv: dict, arch: str, findings: Findings, geometry: str = "ds41"):
    """Metadata -> (Cfg, notes).  Missing keys fall back to the config.json default with a warning; aliased spellings
    are accepted with an info note; with geometry == "ds41" every value is compared with the compile-time geometry."""
    cfg = SimpleNamespace()
    notes = {"aliases": {}, "derived": {}}

    def full(name):
        return name if name.startswith(("general.", "tokenizer.")) else f"{arch}.{name}"

    for k in KEYS:
        value, src = MISSING, None
        for cand in (k.name,) + tuple(k.aliases):
            if full(cand) in kv:
                value, src = kv[full(cand)], cand
                break
        if src is not None and src != k.name:
            notes["aliases"][full(k.name)] = full(src)
            findings.info("META_ALIAS", f"metadata `{full(src)}` accepted as `{full(k.name)}`")
        if isinstance(value, str) and value.startswith("<array len"):
            # a headers JSON keeps only the length of arrays above 64 entries: nothing to check, use the default
            findings.warn("META_SUMMARY", f"`{full(k.name)}` is only summarised ({value}); cannot be checked")
            value = MISSING
            if k.default is MISSING:
                continue
        if value is MISSING:
            if k.default is MISSING:
                if k.name.startswith("swiglu_clamp"):
                    value = [10.0] * int(kv.get(f"{arch}.block_count", 40))   # (contract: both clamps are 10)
                    notes["derived"][full(k.name)] = "[10.0] * block_count"
                    findings.warn("META_DERIVED", f"metadata `{full(k.name)}` missing; assuming {value[:1]} x block_count")
                else:
                    findings.error("META_MISSING", f"required metadata `{full(k.name)}` is missing")
                    continue
            else:
                value = k.default
                shown = value if not isinstance(value, list) or len(value) <= 8 else f"[{len(value)} entries]"
                notes["derived"][full(k.name)] = value
                findings.warn("META_DERIVED", f"metadata `{full(k.name)}` missing; derived from the config.json "
                                              f"default {shown}")
        setattr(cfg, k.field, value)
        if geometry == "ds41" and k.expect is not MISSING and src is not None:
            if not _close(value, k.expect):
                lvl = findings.warn if k.field == "context" else findings.error
                shown = value if not isinstance(value, list) or len(value) <= 8 else f"[{len(value)} entries]"
                lvl("META_VALUE", f"`{full(k.name)}` = {shown}, the port's geometry requires {k.expect}"
                    if not isinstance(k.expect, list) else f"`{full(k.name)}` differs from the expected list")
    if hasattr(cfg, "n_layer"):
        for f in ("clamp_exp", "clamp_shexp"):
            vals = getattr(cfg, f, None)
            if vals is not None and (len(vals) != cfg.n_layer or any(not _close(v, 10.0) for v in vals)) \
                    and geometry == "ds41":
                findings.error("META_VALUE", f"{arch}.swiglu_clamp_{'exp' if f == 'clamp_exp' else 'shexp'} must be "
                                             f"10.0 for each of the {cfg.n_layer} layers")
    return cfg, notes


# ---------------------------------------------------------------------------------------------- layer modes
def layer_modes(cfg, findings: Findings):
    """-> list of dicts per layer: {layer, ratio, mode, kv_owner, indexer, index_compressor, engram}."""
    n = cfg.n_layer
    ratios = list(cfg.ratios)
    if len(ratios) not in (n, n + 3):
        findings.error("LAYERS_RATIOS", f"compress_ratios has {len(ratios)} entries; expected {n} (+3 DSpark) layers")
        ratios = (ratios + [0] * n)[:n]
    if any(r not in (0, 1, 2) for r in ratios[:n]):
        findings.error("LAYERS_RATIOS", f"compress_ratios holds a ratio other than 0/1/2: {sorted(set(ratios[:n]))}")
    kv_src, idx_src = list(cfg.kv_source), list(cfg.index_source)
    for L in kv_src + idx_src + [cfg.cand_source]:
        if not 0 <= L < n:
            findings.error("LAYERS_IDS", f"layer id {L} is outside 0..{n - 1}")
    for L in kv_src:
        if 0 <= L < n and ratios[L] == 0:
            findings.error("LAYERS_IDS", f"kv_source layer {L} has compress ratio 0")
    for L in idx_src:
        if 0 <= L < n and ratios[L] == 0:
            findings.error("LAYERS_IDS", f"index_source layer {L} has compress ratio 0")
    if cfg.cand_source not in kv_src:
        findings.error("LAYERS_IDS", f"candidate_source_layer_id {cfg.cand_source} is not a kv_source layer")
    out = []
    for L in range(n):
        r = ratios[L]
        if r == 0:
            mode, owner = "SWA", None
        elif L in kv_src:
            mode, owner = "Full", L
        else:
            owners = [s for s in kv_src if s < L and 0 <= s < n and ratios[s] == r]
            owner = max(owners) if owners else None
            if owner is None:
                findings.error("LAYERS_OWNER", f"layer {L} (ratio {r}) has no earlier kv_source layer with that ratio")
            mode = "Reindex" if L in idx_src else "Reuse"
        out.append({"layer": L, "ratio": r, "mode": mode, "kv_owner": owner,
                    "indexer": L in idx_src, "index_compressor": L in idx_src and L in kv_src,
                    "compressor_gate": L in kv_src and r == 2, "engram": L in cfg.engram_layers})
    return out


# ---------------------------------------------------------------------------------------------- tensor table
# strict type per tensor class (the mxxm-t MXFP4 GGUF).  klass drives the relaxed (--allow-non-mxfp4) check:
#   float  -> F32 only          quant -> any type we can size           expert / engram -> see manifest.py
@dataclasses.dataclass(frozen=True)
class TSpec:
    name: str            # canonical
    dims: tuple
    strict: str          # exact type in the MXFP4 GGUF
    klass: str           # float | quant | expert | engram
    group: str           # memory-plan group
    layer: int | None


# file spelling aliases, by canonical suffix after "blk.N." -> accepted spellings
TENSOR_ALIASES = {
    "engram_embed.weight": ("engram_embd.weight",),
    "indexer_compressor_kv.weight": ("indexer.attn_k.weight",),
    "indexer_compressor_norm.weight": ("indexer.k_norm.weight",),
}
# present in some GGUFs, unused by the text-only port: accepted and ignored (suffix after "blk.N.")
IGNORED_SUFFIXES = ("exp_probs_b_vl.bias",)


def derive_dims(cfg):
    d = SimpleNamespace()
    d.nq = cfg.n_head * cfg.head_dim
    d.hc_mix = (2 + cfg.hc) * cfg.hc
    d.e_orders = cfg.e_ngram - 1
    d.e_wkv_in = d.e_orders * cfg.e_heads * cfg.eh_dim
    d.e_wkv_out = (cfg.hc + 1) * cfg.hidden
    return d


def tensor_specs(cfg, modes, vocab: int, e_rows):
    """Every tensor the port expects, canonical names, in layer order.  e_rows: {engram layer: table rows}."""
    d = derive_dims(cfg)
    H, FF, E = cfg.hidden, cfg.ff, cfg.n_expert
    S: list[TSpec] = []

    def add(name, dims, strict, klass, group, layer=None):
        S.append(TSpec(name, tuple(dims), strict, klass, group, layer))

    add("token_embd.weight", (H, vocab), "BF16", "quant", "embd")
    for m in modes:
        L = m["layer"]
        p = f"blk.{L}."
        add(p + "attn_norm.weight", (H,), "F32", "float", "norm", L)
        add(p + "attn_q_a.weight", (H, cfg.q_lora), "Q8_0", "quant", "attn", L)
        add(p + "attn_q_a_norm.weight", (cfg.q_lora,), "F32", "float", "attn", L)
        add(p + "attn_q_b.weight", (cfg.q_lora, d.nq), "Q8_0", "quant", "attn", L)
        add(p + "attn_kv.weight", (H, cfg.head_dim), "Q8_0", "quant", "attn", L)
        add(p + "attn_kv_a_norm.weight", (cfg.head_dim,), "F32", "float", "attn", L)
        add(p + "attn_sinks.weight", (cfg.n_head,), "F32", "float", "attn", L)
        add(p + "attn_output_a.weight", (d.nq // cfg.o_groups, cfg.o_groups * cfg.o_lora), "Q8_0", "quant", "attn", L)
        add(p + "attn_output_b.weight", (cfg.o_groups * cfg.o_lora, H), "Q8_0", "quant", "attn", L)
        add(p + "ffn_norm.weight", (H,), "F32", "float", "norm", L)
        add(p + "ffn_gate_inp.weight", (H, E), "BF16", "quant", "router", L)
        add(p + "exp_probs_b.bias", (E,), "F32", "float", "router", L)
        add(p + "ffn_gate_exps.weight", (H, FF, E), "MXFP4", "expert", "expert", L)
        add(p + "ffn_down_exps.weight", (FF, H, E), "MXFP4", "expert", "expert", L)
        add(p + "ffn_up_exps.weight", (H, FF, E), "MXFP4", "expert", "expert", L)
        add(p + "ffn_gate_shexp.weight", (H, FF), "Q8_0", "quant", "shared", L)
        add(p + "ffn_down_shexp.weight", (FF, H), "Q8_0", "quant", "shared", L)
        add(p + "ffn_up_shexp.weight", (H, FF), "Q8_0", "quant", "shared", L)
        for ffn in ("attn", "ffn"):
            add(p + f"hc_{ffn}_fn.weight", (cfg.hc * H, d.hc_mix), "F32", "quant", "mhc", L)
            add(p + f"hc_{ffn}_base.weight", (d.hc_mix,), "F32", "float", "mhc", L)
            add(p + f"hc_{ffn}_scale.weight", (3,), "F32", "float", "mhc", L)
        if m["mode"] == "Full":
            add(p + "attn_compressor_kv.weight", (H, cfg.head_dim), "BF16", "quant", "compressor", L)
            add(p + "attn_compressor_norm.weight", (cfg.head_dim,), "F32", "float", "compressor", L)
            if m["compressor_gate"]:
                add(p + "attn_compressor_gate.weight", (H, cfg.head_dim), "BF16", "quant", "compressor", L)
            if m["index_compressor"]:
                add(p + "indexer_compressor_kv.weight", (cfg.head_dim, cfg.idx_dim), "BF16", "quant", "indexer", L)
                add(p + "indexer_compressor_norm.weight", (cfg.idx_dim,), "F32", "float", "indexer", L)
        if m["indexer"]:
            add(p + "indexer.attn_q_b.weight", (cfg.q_lora, cfg.idx_heads * cfg.idx_dim), "Q8_0", "quant", "indexer", L)
            add(p + "indexer.proj.weight", (H, cfg.idx_heads), "BF16", "quant", "indexer", L)
        if m["engram"]:
            add(p + "engram_embed.weight", (cfg.eh_dim, e_rows.get(L, 0)), "MXFP4", "engram", "engram_embed", L)
            add(p + "engram_k.weight", (H, cfg.hc), "BF16", "quant", "engram_proj", L)
            add(p + "engram_q.weight", (H, cfg.hc), "BF16", "quant", "engram_proj", L)
            add(p + "engram_wkv.weight", (d.e_wkv_in, d.e_wkv_out), "Q8_0", "quant", "engram_proj", L)
    add("output_norm.weight", (H,), "F32", "float", "head")
    add("output.weight", (H, vocab), "BF16", "quant", "head")
    return S


def sidecar_specs(cfg, vocab: int):
    """The DSpark sidecar (arch `dflash`): 3 SWA blocks with 128 routed experts + conditioning / Markov / confidence."""
    H, FF = cfg.hidden, cfg.ff
    nq = cfg.n_head * cfg.head_dim
    n_blocks, n_exp, rank = 3, 128, 256
    hc_mix = (2 + cfg.hc) * cfg.hc
    S: list[TSpec] = []

    def add(name, dims, strict, klass, group, layer=None):
        S.append(TSpec(name, tuple(dims), strict, klass, group, layer))

    for L in range(n_blocks):
        p = f"blk.{L}."
        add(p + "attn_norm.weight", (H,), "F32", "float", "norm", L)
        add(p + "attn_q_a.weight", (H, cfg.q_lora), "Q8_0", "quant", "attn", L)
        add(p + "attn_q_a_norm.weight", (cfg.q_lora,), "F32", "float", "attn", L)
        add(p + "attn_q_b.weight", (cfg.q_lora, nq), "Q8_0", "quant", "attn", L)
        add(p + "attn_kv.weight", (H, cfg.head_dim), "Q8_0", "quant", "attn", L)
        add(p + "attn_kv_a_norm.weight", (cfg.head_dim,), "F32", "float", "attn", L)
        add(p + "attn_sinks.weight", (cfg.n_head,), "F32", "float", "attn", L)
        add(p + "attn_output_a.weight", (nq // cfg.o_groups, cfg.o_groups * cfg.o_lora), "Q8_0", "quant", "attn", L)
        add(p + "attn_output_b.weight", (cfg.o_groups * cfg.o_lora, H), "Q8_0", "quant", "attn", L)
        add(p + "ffn_norm.weight", (H,), "F32", "float", "norm", L)
        add(p + "ffn_gate_inp.weight", (H, n_exp), "BF16", "quant", "router", L)
        add(p + "exp_probs_b.bias", (n_exp,), "F32", "float", "router", L)
        add(p + "ffn_gate_exps.weight", (H, FF, n_exp), "MXFP4", "expert", "expert", L)
        add(p + "ffn_down_exps.weight", (FF, H, n_exp), "MXFP4", "expert", "expert", L)
        add(p + "ffn_up_exps.weight", (H, FF, n_exp), "MXFP4", "expert", "expert", L)
        add(p + "ffn_gate_shexp.weight", (H, FF), "Q8_0", "quant", "shared", L)
        add(p + "ffn_down_shexp.weight", (FF, H), "Q8_0", "quant", "shared", L)
        add(p + "ffn_up_shexp.weight", (H, FF), "Q8_0", "quant", "shared", L)
        for ffn in ("attn", "ffn"):
            add(p + f"hc_{ffn}_fn.weight", (cfg.hc * H, hc_mix), "F32", "quant", "mhc", L)
            add(p + f"hc_{ffn}_base.weight", (hc_mix,), "F32", "float", "mhc", L)
            add(p + f"hc_{ffn}_scale.weight", (3,), "F32", "float", "mhc", L)
    add("enc.output_norm.weight", (H,), "F32", "float", "head")
    add("fc.weight", (3 * H, H), "Q8_0", "quant", "head")                 # main_proj: 3 target layers' states -> hidden
    add("conf_proj.weight", (H + rank, 1), "BF16", "quant", "head")
    add("markov_w1.weight", (rank, vocab), "BF16", "quant", "head")
    add("markov_w2.weight", (rank, vocab), "BF16", "quant", "head")
    add("output_norm.weight", (H,), "F32", "float", "head")
    return S


# ---------------------------------------------------------------------------------------------- Engram constants
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def next_prime_above(n: int) -> int:
    n += 1
    while not _is_prime(n):
        n += 1
    return n


def engram_primes(first_above: int, count: int) -> list:
    out, p = [], first_above
    for _ in range(count):
        p = next_prime_above(p)
        out.append(p)
    return out


def engram_multipliers(layer_id: int, compressed_vocab: int, ngram: int):
    """default_rng(10007 * layer_id).integers(0, (INT64_MAX // compressed_vocab) // 2, size=ngram) * 2 + 1
    (third_party/deepseek-v41-flash-reference/inference/engram.py)."""
    import numpy as np
    rng = np.random.default_rng(10007 * layer_id)
    return [int(v) for v in rng.integers(0, (INT64_MAX // compressed_vocab) // 2, size=ngram) * 2 + 1]


def check_engram_constants(cfg, kv: dict, arch: str, e_rows: dict, findings: Findings, geometry: str):
    """Primes / offsets / multipliers / num_embeddings of the metadata, against each other and against the tensors."""
    layers = list(cfg.engram_layers)
    n_per_layer = (cfg.e_ngram - 1) * cfg.e_heads
    primes, offsets = kv.get(f"{arch}.engram.primes"), kv.get(f"{arch}.engram.offsets")
    mults, n_emb = kv.get(f"{arch}.engram.multipliers"), kv.get(f"{arch}.engram.num_embeddings")
    for nm, v, want in (("primes", primes, n_per_layer * len(layers)), ("offsets", offsets, n_per_layer * len(layers)),
                        ("multipliers", mults, cfg.e_ngram * len(layers))):
        if v is None:
            findings.error("ENGRAM_META", f"{arch}.engram.{nm} is missing")
        elif not isinstance(v, list) or len(v) != want:
            findings.error("ENGRAM_META", f"{arch}.engram.{nm} has {v if not isinstance(v, list) else len(v)} entries, "
                                          f"expected {want} ({len(layers)} layers)")
    if n_emb is None:
        findings.warn("META_DERIVED", f"{arch}.engram.num_embeddings missing; using the table shapes "
                                      f"{[e_rows.get(L) for L in layers]}")
        n_emb = [e_rows.get(L) for L in layers]
    elif list(n_emb) != [e_rows.get(L) for L in layers]:
        findings.error("ENGRAM_META", f"engram.num_embeddings {list(n_emb)} != the engram_embed table rows "
                                      f"{[e_rows.get(L) for L in layers]}")
    if not (isinstance(primes, list) and len(primes) == n_per_layer * len(layers)):
        return
    if not all(_is_prime(p) for p in primes):
        findings.error("ENGRAM_META", "engram.primes holds a non-prime")
    if len(set(primes)) != len(primes) or primes != sorted(primes):
        findings.error("ENGRAM_META", "engram.primes must be distinct and increasing (never reused)")
    consecutive = engram_primes(primes[0] - 1, len(primes))
    if primes != consecutive:
        findings.error("ENGRAM_META", "engram.primes are not consecutive primes")
    if geometry == "ds41" and primes[:1] != [next_prime_above(ENGRAM_VOCAB_SIZE - 1)]:
        findings.error("ENGRAM_META", f"engram.primes[0] = {primes[0]}, the reference starts at the next prime above "
                                      f"{ENGRAM_VOCAB_SIZE - 1}")
    for li, L in enumerate(layers):
        seg = primes[li * n_per_layer:(li + 1) * n_per_layer]
        if sum(seg) != e_rows.get(L):
            findings.error("ENGRAM_META", f"layer {L}: the {n_per_layer} primes sum to {sum(seg)}, the table has "
                                          f"{e_rows.get(L)} rows")
        if isinstance(offsets, list) and len(offsets) == len(primes):
            want, acc = [], 0
            for p in seg:
                want.append(acc)
                acc += p
            if offsets[li * n_per_layer:(li + 1) * n_per_layer] != want:
                findings.error("ENGRAM_META", f"layer {L}: engram.offsets are not the running sums of the primes")
        if isinstance(mults, list) and len(mults) == cfg.e_ngram * len(layers):
            want_m = engram_multipliers(L, cfg.e_cvocab, cfg.e_ngram)
            if mults[li * cfg.e_ngram:(li + 1) * cfg.e_ngram] != want_m:
                findings.warn("ENGRAM_MULT", f"layer {L}: engram.multipliers differ from "
                                             "default_rng(10007 * layer).integers(...) * 2 + 1 of the reference")
