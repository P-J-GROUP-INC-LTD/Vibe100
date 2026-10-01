#!/usr/bin/env python3
"""make_mini_gguf.py - write a tiny `deepseek41`-shaped GGUF (split into shards) as a test fixture and oracle input.

    python3 tools/ds41/make_mini_gguf.py /tmp/mini --seed 0 --n-shards 3
    python3 tools/ds41/manifest.py /tmp/mini --geometry self --no-plan

What it is: the same tensor NAMES, ggml TYPES (MXFP4 experts and Engram tables, Q8_0 attention / shared experts /
indexer q / Engram wkv, BF16 router / embeddings / compressors, F32 norms / sinks / mHC) and metadata KEYS as
mxxm-t/DeepSeek-V4.1-Flash-GGUF, with small dimensions.  With the defaults:

    hidden 256, 8 layers, 16 experts top-2, expert FF 256, 4 heads x 64, q_lora 64, 2 output groups x 32,
    indexer 4 heads x 32, hc 4, Engram 2 tables (layers 1 and 3), 2 heads x 64 values per row, vocab 512
    layer modes  L0 L1 SWA | L2 Full (ratio 2, own compressor + gate + indexer) | L3 Reuse |
                 L4 Full (ratio 1, candidate source) | L5 Reuse | L6 Reindex | L7 Reuse

Everything is deterministic from --seed: every tensor draws from its own generator seeded with (seed, crc32(name)).
Weights are quantised with the encoders of ggml_codecs.py, which are bit-identical to GGML's C reference.

Constraints (checked): every Q8_0 / MXFP4 row length is a multiple of 32; the hidden size is a multiple of 128 and the expert FF
a multiple of 256, which is what the C++ CPU kernels need (strata::ds41::cpu check_view: a CPU half has FF / 2 = a multiple of 128 rows,
four 32-blocks per kernel group) - so with the defaults the file also runs through them at its own, runtime, dimensions - and the
CPU halves of CONTRACTS.md fall on MXFP4 block boundaries.

NOTE for kernel authors: the real model's geometry is a compile-time contract (include/strata/ds41/geometry.hpp:
kHidden = 5120, kFF = 2304, 384 experts, ...).  Code with those constants compiled in cannot run this file; use it for
runtime-dimensioned code (loaders, the numpy oracle, tools/ds41/expert_layout.py), or give the kernel tests their own
random blocks at the real shapes.  Validate it with `manifest.py --geometry self`.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ds41_spec as S  # noqa: E402
import ggml_codecs as C  # noqa: E402
import gguf_io as G  # noqa: E402


@dataclasses.dataclass
class MiniConfig:
    hidden: int = 256
    n_layer: int = 8
    n_expert: int = 16
    n_used: int = 2
    ff: int = 256
    n_head: int = 4
    head_dim: int = 64
    rope_dim: int = 16
    q_lora: int = 64
    o_groups: int = 2
    o_lora: int = 32
    idx_heads: int = 4
    idx_dim: int = 32
    idx_topk: int = 8
    hc: int = 4
    hc_iters: int = 20
    vocab: int = 512
    compressed_vocab: int = 300
    sliding_window: int = 16
    cand_block: int = 4
    cand_topk: int = 4
    kv_source: tuple = (2, 4)
    index_source: tuple = (2, 4, 6)
    cand_source: int = 4
    engram_layers: tuple = (1, 3)
    e_heads: int = 2
    e_head_dim: int = 64
    e_ngram: int = 4
    e_pad: int = 2
    engram_vocab: int = 1000          # Engram primes are the consecutive primes above engram_vocab - 1
    n_shards: int = 3
    alignment: int = 32
    seed: int = 0

    # ---- derived
    @property
    def nq(self):
        return self.n_head * self.head_dim

    @property
    def ratios(self):
        """compress ratio per layer (+3 DSpark entries): 0 before the first kv source, 2 up to the candidate source,
        then 1 -- the shape of the real model's [0,0,2 x18,1 x20] map."""
        r = []
        for L in range(self.n_layer):
            if L < min(self.kv_source):
                r.append(0)
            elif L < self.cand_source:
                r.append(2)
            else:
                r.append(1)
        return r + [0, 0, 0]

    def validate(self):
        errs = []

        def mult32(name, v):
            if v % 32:
                errs.append(f"{name} = {v} must be a multiple of 32 (Q8_0 / MXFP4 rows)")

        mult32("hidden", self.hidden)
        if self.hidden % 128:
            errs.append(f"hidden = {self.hidden} must be a multiple of 128 (four MXFP4 blocks per kernel group: the C++ CPU kernels' check_view)")
        if self.ff % 256:
            errs.append(f"ff = {self.ff} must be a multiple of 256 (a CPU half is FF / 2 rows and must be a whole number of 4-block kernel groups: "
                        "the C++ CPU kernels' check_view; the CONTRACTS.md halves alone would need only 64)")
        mult32("q_lora", self.q_lora)
        mult32("n_head * head_dim / o_groups", self.nq // max(1, self.o_groups))
        if self.nq % self.o_groups:
            errs.append("n_head * head_dim must divide by o_groups")
        mult32("o_groups * o_lora", self.o_groups * self.o_lora)
        mult32("e_head_dim", self.e_head_dim)
        mult32("(e_ngram - 1) * e_heads * e_head_dim", (self.e_ngram - 1) * self.e_heads * self.e_head_dim)
        if self.n_used > self.n_expert:
            errs.append("n_used > n_expert")
        if self.vocab < self.compressed_vocab:
            errs.append("vocab < compressed_vocab")
        if self.n_shards < 1:
            errs.append("n_shards >= 1")
        if self.cand_source not in self.kv_source or any(L >= self.n_layer for L in
                                                         self.kv_source + self.index_source + self.engram_layers):
            errs.append("layer ids must be < n_layer and the candidate source must be a kv source")
        if errs:
            raise ValueError("invalid MiniConfig: " + "; ".join(errs))


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode("utf-8"))])


def _weights(cfg: MiniConfig, spec: S.TSpec) -> np.ndarray:
    """float32 values of a tensor, flat, in ggml row order (ne0 fastest)."""
    rng = _rng(cfg.seed, spec.name)
    n = 1
    for d in spec.dims:
        n *= d
    base = spec.name.split(".", 2)[-1] if spec.name.startswith("blk.") else spec.name
    if base.endswith("_norm.weight") or base in ("output_norm.weight",):
        return (1.0 + 0.05 * rng.standard_normal(n)).astype(np.float32)
    if base == "attn_sinks.weight":
        return (0.5 * rng.standard_normal(n)).astype(np.float32)
    if base == "exp_probs_b.bias":
        return (0.01 * rng.standard_normal(n)).astype(np.float32)
    if base.endswith("_base.weight"):
        return (0.1 * rng.standard_normal(n)).astype(np.float32)
    if base.endswith("_scale.weight"):
        return (0.1 + 0.01 * rng.standard_normal(n)).astype(np.float32)
    if spec.klass == "engram":
        return (0.5 * rng.standard_normal(n)).astype(np.float32)
    return (rng.standard_normal(n) / np.sqrt(spec.dims[0])).astype(np.float32)


def engram_tables(cfg: MiniConfig):
    """-> (primes, offsets, multipliers, num_embeddings) laid out like the real model's metadata."""
    per_layer = (cfg.e_ngram - 1) * cfg.e_heads
    primes = S.engram_primes(cfg.engram_vocab - 1, per_layer * len(cfg.engram_layers))
    offsets, num_emb, mults = [], [], []
    for li, L in enumerate(cfg.engram_layers):
        seg = primes[li * per_layer:(li + 1) * per_layer]
        acc = 0
        for p in seg:
            offsets.append(acc)
            acc += p
        num_emb.append(acc)
        mults.extend(S.engram_multipliers(L, cfg.compressed_vocab, cfg.e_ngram))
    return primes, offsets, mults, num_emb


def metadata(cfg: MiniConfig):
    """[(key, type, value)] with the keys and types of the real file (RESEARCH.md section 10)."""
    a = S.ARCH
    primes, offsets, mults, num_emb = engram_tables(cfg)
    rng = _rng(cfg.seed, "token_map")
    tm = np.concatenate([np.arange(cfg.compressed_vocab), rng.integers(0, cfg.compressed_vocab,
                                                                       cfg.vocab - cfg.compressed_vocab)])
    rng.shuffle(tm)
    kv = [
        ("general.architecture", "string", a),
        ("tokenizer.chat_template", "string", "{{ messages }}"),
        ("general.type", "string", "model"),
        ("general.name", "string", "DeepSeek V4.1 Flash (mini test fixture)"),
        ("general.size_label", "string", f"{cfg.n_expert}x{cfg.hidden}"),
        ("general.license", "string", "mit"),
        ("general.tags", "array:string", ["test-fixture"]),
        (f"{a}.block_count", "u32", cfg.n_layer),
        (f"{a}.context_length", "u32", 1048576),
        (f"{a}.embedding_length", "u32", cfg.hidden),
        (f"{a}.attention.head_count", "u32", cfg.n_head),
        (f"{a}.attention.head_count_kv", "u32", 1),
        (f"{a}.rope.scaling.type", "string", "yarn"),
        (f"{a}.rope.scaling.factor", "f32", 16.0),
        (f"{a}.rope.scaling.original_context_length", "u32", 65536),
        (f"{a}.rope.scaling.yarn_beta_fast", "f32", 32.0),
        (f"{a}.rope.scaling.yarn_beta_slow", "f32", 1.0),
        (f"{a}.rope.freq_base", "f32", 10000.0),
        (f"{a}.attention.layer_norm_rms_epsilon", "f32", 1e-20),
        (f"{a}.expert_count", "u32", cfg.n_expert),
        (f"{a}.expert_used_count", "u32", cfg.n_used),
        (f"{a}.expert_gating_func", "u32", 4),
        (f"{a}.attention.key_length", "u32", cfg.head_dim),
        (f"{a}.attention.value_length", "u32", cfg.head_dim),
        ("general.file_type", "u32", 38),
        (f"{a}.rope.dimension_count", "u32", cfg.rope_dim),
        (f"{a}.attention.q_lora_rank", "u32", cfg.q_lora),
        (f"{a}.attention.sliding_window", "u32", cfg.sliding_window),
        (f"{a}.expert_feed_forward_length", "u32", cfg.ff),
        (f"{a}.expert_shared_count", "u32", 1),
        (f"{a}.expert_weights_scale", "f32", 1.5),
        (f"{a}.expert_weights_norm", "bool", True),
        (f"{a}.swiglu_clamp_exp", "array:f32", [10.0] * cfg.n_layer),
        (f"{a}.swiglu_clamp_shexp", "array:f32", [10.0] * cfg.n_layer),
        (f"{a}.attention.indexer.head_count", "u32", cfg.idx_heads),
        (f"{a}.attention.indexer.key_length", "u32", cfg.idx_dim),
        (f"{a}.attention.indexer.top_k", "u32", cfg.idx_topk),
        (f"{a}.attention.output_group_count", "u32", cfg.o_groups),
        (f"{a}.attention.output_lora_rank", "u32", cfg.o_lora),
        (f"{a}.attention.compress_ratios", "array:i32", cfg.ratios),
        (f"{a}.attention.compress_rope_freq_base", "f32", 160000.0),
        (f"{a}.hyper_connection.count", "u32", cfg.hc),
        (f"{a}.hyper_connection.sinkhorn_iterations", "u32", cfg.hc_iters),
        (f"{a}.hyper_connection.epsilon", "f32", 1e-6),
        (f"{a}.hash_layer_count", "u32", 0),
        (f"{a}.attention.kv_source_layer_ids", "array:u32", list(cfg.kv_source)),
        (f"{a}.attention.index_source_layer_ids", "array:u32", list(cfg.index_source)),
        (f"{a}.attention.candidate_source_layer_id", "u32", cfg.cand_source),
        (f"{a}.attention.candidate_block_size", "u32", cfg.cand_block),
        (f"{a}.attention.candidate_topk_blocks", "u32", cfg.cand_topk),
        (f"{a}.engram.layer_ids", "array:u32", list(cfg.engram_layers)),
        (f"{a}.engram.head_dim", "u32", cfg.e_head_dim),
        (f"{a}.engram.n_heads", "u32", cfg.e_heads),
        (f"{a}.engram.max_ngram_size", "u32", cfg.e_ngram),
        (f"{a}.engram.pad_token_id", "u32", cfg.e_pad),
        (f"{a}.engram.compressed_vocab_size", "u32", cfg.compressed_vocab),
        (f"{a}.engram.num_embeddings", "array:u32", num_emb),
        (f"{a}.engram.primes", "array:u32", primes),
        (f"{a}.engram.offsets", "array:u32", offsets),
        (f"{a}.engram.multipliers", "array:i64", mults),
        (f"{a}.engram.token_map", "array:u32", [int(v) for v in tm]),
        ("general.quantization_version", "u32", 2),
        ("tokenizer.ggml.model", "string", "gpt2"),
        ("tokenizer.ggml.pre", "string", "joyai-llm"),
        ("tokenizer.ggml.tokens", "array:string", [f"<t{i}>" for i in range(cfg.vocab)]),
        ("tokenizer.ggml.token_type", "array:i32", [1] * cfg.vocab),
        ("tokenizer.ggml.merges", "array:string", [f"<t{i}> <t{i + 1}>" for i in range(16)]),
        ("tokenizer.ggml.bos_token_id", "u32", 0),
        ("tokenizer.ggml.eos_token_id", "u32", 1),
        ("tokenizer.ggml.padding_token_id", "u32", 1),
        ("tokenizer.ggml.add_bos_token", "bool", False),
        ("tokenizer.ggml.add_eos_token", "bool", False),
    ]
    return kv


def _as_cfg(cfg: MiniConfig):
    """The spec's Cfg for a MiniConfig (the same fields manifest.py reads from the metadata)."""
    from types import SimpleNamespace
    return SimpleNamespace(n_layer=cfg.n_layer, hidden=cfg.hidden, n_expert=cfg.n_expert, ff=cfg.ff,
                           n_head=cfg.n_head, head_dim=cfg.head_dim, q_lora=cfg.q_lora, o_groups=cfg.o_groups,
                           o_lora=cfg.o_lora, idx_heads=cfg.idx_heads, idx_dim=cfg.idx_dim, hc=cfg.hc,
                           e_heads=cfg.e_heads, eh_dim=cfg.e_head_dim, e_ngram=cfg.e_ngram,
                           ratios=cfg.ratios, kv_source=list(cfg.kv_source), index_source=list(cfg.index_source),
                           cand_source=cfg.cand_source, engram_layers=list(cfg.engram_layers))


def tensor_order(specs):
    """The real file's order: expert tensors of every layer (gate, down, up), then the dense tensors, then Engram."""
    exps, dense, engram = [], [], []
    for sp in specs:
        if sp.group == "expert":
            exps.append(sp)
        elif sp.group in ("engram_embed", "engram_proj"):
            engram.append(sp)
        else:
            dense.append(sp)
    rank = {"gate": 0, "down": 1, "up": 2}
    exps.sort(key=lambda sp: (sp.layer, rank[sp.name.split("ffn_")[1].split("_")[0]]))
    return exps, dense, engram


def build_mini(out_dir, cfg: MiniConfig | None = None, basename: str = "mini-ds41-MXFP4") -> dict:
    """Write the shards into out_dir.  -> {"paths": [...], "config": cfg, "tensors": {name: (type, dims)}}"""
    cfg = cfg or MiniConfig()
    cfg.validate()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    primes, offsets, mults, num_emb = engram_tables(cfg)
    e_rows = {L: num_emb[i] for i, L in enumerate(cfg.engram_layers)}
    c = _as_cfg(cfg)
    modes = S.layer_modes(c, S.Findings())
    specs = S.tensor_specs(c, modes, cfg.vocab, e_rows)
    exps, dense, engram = tensor_order(specs)

    def nb(sp):
        return G.nbytes_for(sp.strict, sp.dims)

    body = exps + dense
    n_body_shards = max(1, cfg.n_shards - 1) if (cfg.n_shards > 1 and engram) else cfg.n_shards
    groups: list = []
    if cfg.n_shards == 1:
        groups = [body + engram]
    else:
        total = sum(nb(sp) for sp in body)
        cur, acc, k = [], 0, 1
        for sp in body:
            cur.append(sp)
            acc += nb(sp)
            if len(groups) < n_body_shards - 1 and acc >= k * total / n_body_shards:
                groups.append(cur)
                cur, k = [], k + 1
        groups.append(cur)
        if engram:
            groups.append(engram)
        while len(groups) < cfg.n_shards:      # more shards asked than the data can fill: split the largest group
            i = max(range(len(groups)), key=lambda j: len(groups[j]))
            if len(groups[i]) < 2:
                break
            half = len(groups[i]) // 2
            groups[i:i + 1] = [groups[i][:half], groups[i][half:]]
    n = len(groups)
    total_tensors = len(specs)
    kv_main = metadata(cfg)
    paths = []
    for i, grp in enumerate(groups):
        tensors = [(sp.name, sp.dims, sp.strict, C.encode(sp.strict, _weights(cfg, sp))) for sp in grp]
        kv = list(kv_main) if i == 0 else []
        if n > 1:
            kv += [("split.no", "u16", i), ("split.count", "u16", n), ("split.tensors.count", "i32", total_tensors)]
        name = f"{basename}.gguf" if n == 1 else f"{basename}-{i + 1:05d}-of-{n:05d}.gguf"
        G.write_gguf(out / name, kv, tensors, alignment=cfg.alignment)
        paths.append(out / name)
    return {"paths": paths, "config": cfg, "tensors": {sp.name: (sp.strict, sp.dims) for sp in specs},
            "modes": modes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--basename", default="mini-ds41-MXFP4")
    d = MiniConfig()
    for f in dataclasses.fields(MiniConfig):
        if f.name in ("kv_source", "index_source", "engram_layers"):
            ap.add_argument("--" + f.name.replace("_", "-"), type=lambda s: tuple(int(x) for x in s.split(",")),
                            default=getattr(d, f.name), help="comma-separated layer ids")
        else:
            ap.add_argument("--" + f.name.replace("_", "-"), type=type(getattr(d, f.name)), default=getattr(d, f.name))
    args = ap.parse_args(argv)
    cfg = MiniConfig(**{f.name: getattr(args, f.name) for f in dataclasses.fields(MiniConfig)})
    try:
        res = build_mini(args.out_dir, cfg, args.basename)
    except ValueError as exc:
        print(f"make_mini_gguf.py: {exc}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in res["paths"])
    print(f"wrote {len(res['paths'])} shard(s), {len(res['tensors'])} tensors, {total:,} bytes:")
    for p in res["paths"]:
        print(f"  {p}  ({p.stat().st_size:,} B)")
    print(f"validate: python3 tools/ds41/manifest.py {args.out_dir} --geometry self --no-plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
