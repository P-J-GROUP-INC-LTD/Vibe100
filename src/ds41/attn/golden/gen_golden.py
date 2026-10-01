#!/usr/bin/env python3
"""gen_golden.py - DS1-C: run the NumPy oracle's `attention_layer` (ref/ds41/attention.py) on a MiniGeom-shaped model for enough decode steps that every
layer role and every cache limit is exercised, and save the inputs, the weights (in the GGUF formats) and every stage's output for the C++ replay
(src/ds41/attn/attn_emu_test.cpp: `ds41_attn_emu_test --oracle DIR`).

MODEL (tools/ds41/make_mini_gguf.py's shape, attention part): 8 layers  L0 L1 SWA | L2 FULL ratio 2 (own compressor + gate + indexer) | L3 REUSE |
L4 FULL ratio 1, candidate source | L5 REUSE | L6 REINDEX | L7 REUSE;  hidden 256, 4 heads x 64 (RoPE on the last 16), q_lora 64, 2 output groups x 32,
window 16, indexer 4 heads x 32, top-k 8, candidate pool 4 blocks x 4 positions.  Every limit binds within the run: the 16-slot ring wraps, the ratio-2 caches
exceed top-k 8 (so the selection is a real choice), the ratio-1 cache exceeds the 16-position candidate pool (so the pool restricts and REINDEX is a real
restriction), prefill scenarios leave a half group pending in the ratio-2 compressor.

SCENARIOS (each its own random weights and inputs, `scn_<name>.bin`):
  decode_all_on      QuantConfig(int8_act, window_kv, compressed_kv, index)  one token at a time from position 0
  decode_all_off     QuantConfig.int8()  (the KV flags off)
  prefill_all_on     the oracle's PREFILL of 37 tokens (start_pos 0, s = 37) then decode; the engine replays token by token (prefill(N) == token by token)
  prefill_all_off    prefill of 40 tokens (even) with the flags off, then decode
  decode_win_only    window_kv only                                            (the three flags are independent)
  decode_ckv_idx     compressed_kv + index only

WHAT IS RECORDED per layer and position (the engine's stage names, docs/deepseek/DS1.md section 6):  out (attention output), q (post-RoPE), o (the attention
output after the inverse RoPE: the input of wo_a), kvwin (the SWA KV row), latent (the PRE-RoPE normalised group latent, on the position that completes a group),
comp / idxk (the cache rows published then), topk (the position-sorted compressed positions, -1 padded to kIdxTopK), idx_score (the indexer's scores of the
query, after the candidate mask; -inf = unreachable), blk_score + cand (the candidate source's block scores and pool flags).  The stages are captured by
wrapping the oracle's own functions for the duration of the call (the wrappers return the originals' results untouched); the run is cross-checked against an
unwrapped second run of `attention_layer` (bit-identical outputs).

Binary format (little endian): "DS1CGOLD", u32 version, u32 n; n x { u32 name_len, name, u32 dtype (0 f32, 1 i32, 2 u8, 3 u16), u32 ndim, u32 dims[ndim], data }.
Usage:  gen_golden.py --out DIR [--scenario NAME ...]
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import struct
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from ref.ds41 import attention as A  # noqa: E402
from ref.ds41.config import Config, layer_modes  # noqa: E402
from ref.ds41.quant import (QuantConfig, bf16_to_f32, dequant_q8_0, f32_to_bf16_bits, quantize_q8_0)  # noqa: E402
from ref.ds41.rope import layer_rope_params, rope_table  # noqa: E402

N_POS = 100
RATIOS = (0, 0, 2, 2, 1, 1, 1, 1)


def mini_config(max_seq: int) -> Config:
    return Config(
        vocab_size=512, dim=256, moe_inter_dim=256, n_layers=8, n_heads=4, n_routed_experts=16, n_activated_experts=2, q_lora_rank=64,
        head_dim=64, rope_head_dim=16, o_groups=2, o_lora_rank=32, window_size=16, compress_ratios=RATIOS, kv_source_layers=(2, 4),
        index_source_layers=(2, 4, 6), compress_rope_theta=160000.0, original_seq_len=64, rope_theta=10000.0, rope_factor=4.0,
        index_n_heads=4, index_head_dim=32, index_topk=8, candidate_source_layer=4, candidate_topk_blocks=4, candidate_block_size=4,
        engram_layer_ids=(1, 3), engram_num_embeddings=(0, 0), engram_pad_id=2, max_seq_len=max_seq)


# ---------------------------------------------------------------------------------------------------------------
# weights (the formats the GGUF stores: Q8_0 blocks, BF16, F32); the oracle gets the dequantised values, exactly what the engine will decode
# ---------------------------------------------------------------------------------------------------------------
def q8(rng, rows, cols, scale):
    w = (rng.standard_normal((rows, cols)) * scale).astype(np.float32)
    blocks = quantize_q8_0(w)                               # [rows, cols/32, 34] uint8
    return blocks, dequant_q8_0(blocks).astype(np.float32)


def bf16(rng, rows, cols, scale):
    w = (rng.standard_normal((rows, cols)) * scale).astype(np.float32)
    bits = f32_to_bf16_bits(w)
    return bits, bf16_to_f32(bits).astype(np.float32)


def norm_w(rng, n):
    return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)


def make_layer_weights(cfg: Config, layer: int, mode, rng):
    """-> (raw: {name: ndarray in the stored format}, dense: {name: float32 array the oracle reads})."""
    D, QL, H, HID = cfg.head_dim, cfg.q_lora_rank, cfg.n_heads, cfg.dim
    ID, IH = cfg.index_head_dim, cfg.index_n_heads
    raw, dense = {}, {}

    def put_q8(name, rows, cols, scale):
        raw[name], dense[name] = q8(rng, rows, cols, scale)

    def put_bf16(name, rows, cols, scale):
        raw[name], dense[name] = bf16(rng, rows, cols, scale)

    def put_f32(name, v):
        raw[name] = dense[name] = np.asarray(v, np.float32)

    put_q8("wq_a.weight", QL, HID, HID ** -0.5)
    put_f32("q_norm.weight", norm_w(rng, QL))
    put_q8("wq_b.weight", H * D, QL, QL ** -0.5)
    put_q8("wkv.weight", D, HID, HID ** -0.5)
    put_f32("kv_norm.weight", norm_w(rng, D))
    put_f32("attn_sink", 0.5 * rng.standard_normal(H))
    put_q8("wo_a.weight", cfg.o_groups * cfg.o_lora_rank, H * D // cfg.o_groups, (H * D // cfg.o_groups) ** -0.5)
    put_q8("wo_b.weight", HID, cfg.o_groups * cfg.o_lora_rank, (cfg.o_groups * cfg.o_lora_rank) ** -0.5)
    if mode.value == "full":
        ratio = cfg.compress_ratios[layer]
        put_bf16("compressor.wkv.weight", D, HID, HID ** -0.5)
        if ratio > 1:
            put_bf16("compressor.wgate.weight", D, HID, 2.0 * HID ** -0.5)
        put_f32("compressor.norm.weight", norm_w(rng, D))
        put_bf16("indexer.wk.weight", ID, D, D ** -0.5)
        put_f32("indexer.k_norm.weight", norm_w(rng, ID))
    if mode.value in ("full", "reindex"):
        put_q8("indexer.wq_b.weight", IH * ID, QL, QL ** -0.5)
        put_bf16("indexer.weights_proj.weight", IH, HID, HID ** -0.5)
    return raw, dense


# ---------------------------------------------------------------------------------------------------------------
# the oracle with its stages captured
# ---------------------------------------------------------------------------------------------------------------
class Capture:
    """Wraps the oracle's functions (module globals of ref.ds41.attention, which attention_layer looks up at call time); the wrappers return the originals'
    results untouched."""

    def __init__(self):
        self.reset()
        self._orig = {}

    def reset(self):
        self.q = self.o_rot = self.kv_rows = self.latent = None
        self.scores = {}

    def __enter__(self):
        mod = A
        for name in ("window_kv", "compressor", "sparse_attn", "apply_rope_tail", "note_margin"):
            self._orig[name] = getattr(mod, name)
        cap = self

        def window_kv(cfg, x, start_pos, cos, sin, **kw):
            rows, idxs = cap._orig["window_kv"](cfg, x, start_pos, cos, sin, **kw)
            # prefill: the chunk's own rows [s, D]; decode: the whole ring -> the row this position wrote
            cap.kv_rows = np.array(rows) if start_pos == 0 else np.array(rows[start_pos % cfg.window_size])[None, :]
            return rows, idxs

        def compressor(*a, **kw):
            lat = cap._orig["compressor"](*a, **kw)
            cap.latent = None if lat is None else np.array(lat)
            return lat

        def sparse_attn(q, kv, sink, idxs, scale, **kw):
            cap.q = np.array(q)
            return cap._orig["sparse_attn"](q, kv, sink, idxs, scale, **kw)

        def apply_rope_tail(x, cos, sin, rd, inverse=False):
            y = cap._orig["apply_rope_tail"](x, cos, sin, rd, inverse)
            if inverse:
                cap.o_rot = np.array(y)
            return y

        def note_margin(shared, key, scores, k):
            cap.scores[key] = np.array(scores)
            return cap._orig["note_margin"](shared, key, scores, k)

        for n, f in (("window_kv", window_kv), ("compressor", compressor), ("sparse_attn", sparse_attn), ("apply_rope_tail", apply_rope_tail),
                     ("note_margin", note_margin)):
            setattr(mod, n, f)
        return self

    def __exit__(self, *exc):
        for n, f in self._orig.items():
            setattr(A, n, f)


def run_scenario(name: str, quant: QuantConfig, chunks: list, seed: int) -> dict:
    n_pos = sum(s for _, s in chunks)
    cfg = mini_config(n_pos + 8)
    modes = layer_modes(cfg)
    rng = np.random.default_rng(seed)
    raw, dense, rope = {}, {}, {}
    for l in range(cfg.n_layers):
        raw[l], dense[l] = make_layer_weights(cfg, l, modes[l], rng)
        p = layer_rope_params(cfg, l)
        rope[l] = rope_table(cfg.rope_head_dim, cfg.max_seq_len, p["original_seq_len"], p["base"], cfg.rope_factor, cfg.beta_fast, cfg.beta_slow)
    xs = [rng.standard_normal((n_pos, cfg.dim)).astype(np.float32) for _ in range(cfg.n_layers)]

    def new_state():
        return [A.LayerCache.new(cfg, l) for l in range(cfg.n_layers)], A.SharedState()

    caches, shared = new_state()
    caches2, shared2 = new_state()                      # the unwrapped cross-check run
    D, IK = cfg.head_dim, cfg.index_topk
    out = {}

    def rec(key, l, shape, dtype, fill):
        k = f"{key}.{l}"
        if k not in out:
            out[k] = np.full((n_pos,) + shape, fill, dtype)
        return out[k]

    for start, s in chunks:
        for l in range(cfg.n_layers):
            x = xs[l][start:start + s]
            get = (lambda l_: (lambda n: dense[l_][n]))(l)
            rp = (rope[l][0], rope[l][1])
            cap = Capture()
            with cap:
                cap.reset()
                y = A.attention_layer(cfg, l, modes[l], x, start, rp, get, lc=caches[l], caches=caches, shared=shared, quant=quant)
            y2 = A.attention_layer(cfg, l, modes[l], x, start, rp, get, lc=caches2[l], caches=caches2, shared=shared2, quant=quant)
            assert np.array_equal(y, y2), f"{name}: layer {l} start {start}: the captured run differs from the plain run"
            ratio = cfg.compress_ratios[l]
            rec("out", l, (cfg.dim,), np.float32, np.nan)[start:start + s] = y
            rec("q", l, (cfg.n_heads * D,), np.float32, np.nan)[start:start + s] = cap.q.reshape(s, -1)
            rec("o", l, (cfg.n_heads * D,), np.float32, np.nan)[start:start + s] = cap.o_rot.reshape(s, -1)
            rec("kvwin", l, (D,), np.float32, np.nan)[start:start + s] = cap.kv_rows
            if ratio:
                tk = np.full((s, IK), -1, np.int32)
                cidx = shared.topk_idxs
                tk[:, :cidx.shape[1]] = cidx
                rec("topk", l, (IK,), np.int32, -1)[start:start + s] = tk
            if modes[l].value == "full":
                lat_rows = cap.latent                                   # [m, D] pre-RoPE, or None
                if lat_rows is not None:
                    m = lat_rows.shape[0]
                    g0 = start // ratio                                  # first group index written
                    lat_a, comp_a, idxk_a = rec("latent", l, (D,), np.float32, np.nan), rec("comp", l, (D,), np.float32, np.nan), rec("idxk", l, (cfg.index_head_dim,), np.float32, np.nan)
                    for j in range(m):
                        pos = (g0 + j + 1) * ratio - 1                   # the position that completes group g0 + j
                        lat_a[pos] = lat_rows[j]
                        comp_a[pos] = caches[l].comp_kv[g0 + j]
                        idxk_a[pos] = caches[l].index_k[g0 + j]
            if modes[l].value in ("full", "reindex"):
                key = f"index.{l}"
                if key in cap.scores:
                    sc = cap.scores[key]                                  # [s, n]
                    arr = rec("idx_score", l, (n_pos,), np.float32, np.nan)
                    nn = rec("idx_n", l, (), np.int32, 0)
                    for r in range(s):
                        arr[start + r, :sc.shape[1]] = sc[r]
                        nn[start + r] = (start + r + 1) // ratio          # the entries reachable by this query (the rest of a prefill row is -inf)
                if l == cfg.candidate_source_layer and f"blocks.{l}" in cap.scores:
                    bs = cap.scores[f"blocks.{l}"]                        # [s, nb]
                    nb_max = (n_pos + cfg.candidate_block_size - 1) // cfg.candidate_block_size
                    barr = rec("blk_score", l, (nb_max,), np.float32, np.nan)
                    carr = rec("cand", l, (nb_max,), np.uint8, 0)
                    cand = shared.candidates                              # [s, n] bool, block-expanded
                    for r in range(s):
                        barr[start + r, :bs.shape[1]] = bs[r]
                        nblk = (cand.shape[1] + cfg.candidate_block_size - 1) // cfg.candidate_block_size
                        carr[start + r, :nblk] = cand[r, ::cfg.candidate_block_size][:nblk]
    meta = dict(
        ratios=np.array(RATIOS, np.int32), kv_source=np.array(cfg.kv_source_layers, np.int32), index_source=np.array(cfg.index_source_layers, np.int32),
        cand_source=np.array([cfg.candidate_source_layer], np.int32), n_pos=np.array([n_pos], np.int32),
        chunks=np.array([v for c in chunks for v in c], np.int32),
        quant=np.array([int(quant.int8_act), int(quant.window_kv), int(quant.compressed_kv), int(quant.index)], np.int32),
        eps=np.array([cfg.norm_eps], np.float32))
    rec_out = {f"cfg.{k}": v for k, v in meta.items()}
    for l in range(cfg.n_layers):
        for n, a in raw[l].items():
            rec_out[f"w.{l}.{n}"] = a
        rec_out[f"rope.{l}.cos"] = rope[l][0].astype(np.float32)
        rec_out[f"rope.{l}.sin"] = rope[l][1].astype(np.float32)
        rec_out[f"x.{l}"] = xs[l]
    rec_out.update(out)
    # selection margins of the oracle (how close the k-th and (k+1)-th best scores were), for the report
    rec_out["cfg.margin_index"] = np.array([min(v) for k, v in sorted(shared.diag.items()) if k.startswith("index.")] or [0.0], np.float32)
    return rec_out


def write_bin(path: pathlib.Path, recs: dict) -> None:
    dt = {np.dtype(np.float32): 0, np.dtype(np.int32): 1, np.dtype(np.uint8): 2, np.dtype(np.uint16): 3}
    with open(path, "wb") as f:
        f.write(b"DS1CGOLD")
        f.write(struct.pack("<II", 1, len(recs)))
        for name, a in recs.items():
            a = np.ascontiguousarray(a)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)) + nb)
            f.write(struct.pack("<II", dt[a.dtype], a.ndim) + b"".join(struct.pack("<I", d) for d in a.shape))
            f.write(a.astype(a.dtype.newbyteorder("<")).tobytes())


DECODE = [(p, 1) for p in range(N_POS)]
SCENARIOS = {
    "decode_all_on": (QuantConfig(int8_act=True, window_kv=True, compressed_kv=True, index=True), DECODE, 11),
    "decode_all_off": (QuantConfig.int8(), DECODE, 12),
    "prefill_all_on": (QuantConfig(int8_act=True, window_kv=True, compressed_kv=True, index=True), [(0, 37)] + [(p, 1) for p in range(37, N_POS)], 13),
    "prefill_all_off": (QuantConfig.int8(), [(0, 40)] + [(p, 1) for p in range(40, N_POS)], 16),
    "decode_win_only": (QuantConfig(int8_act=True, window_kv=True), DECODE, 14),
    "decode_ckv_idx": (QuantConfig(int8_act=True, compressed_kv=True, index=True), DECODE, 15),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenario", action="append", help="only these (default: all)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (quant, chunks, seed) in SCENARIOS.items():
        if a.scenario and name not in a.scenario:
            continue
        recs = run_scenario(name, quant, chunks, seed)
        write_bin(out / f"scn_{name}.bin", recs)
        print(f"{name}: {len(recs)} records, min index margin {float(recs['cfg.margin_index'][0]):.3g}", flush=True)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
