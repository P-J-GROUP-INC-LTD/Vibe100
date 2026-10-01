#!/usr/bin/env python3
"""model_fixture.py - the fixtures of the DS1-A tests (ds41_model_test, ds41_model_real_plan_test), written by ctest before them.

    python3 src/ds41/model/model_fixture.py --out <dir> [--seed 0]

Writes under <dir>:
    mini/*.gguf          the mini deepseek41 GGUF in 3 shards (tools/ds41/make_mini_gguf.py): every tensor class, MXFP4 experts, Engram tables
    golden/              what the C++ loader is compared with, all produced by readers that share no code with it:
        shards.txt         shard index, file, size, alignment, data start, split keys     (tools/gguf_reader.py through tools/ds41/gguf_io.py)
        meta.txt           every metadata key of shard 1 (numbers, bools, short strings; arrays whole; string arrays as a count)
        config.txt         the oracle's Config (ref/ds41/config.py from_gguf_metadata) as `field value...` lines
        modes.txt          `layer mode` (ref/ds41/config.py layer_modes)
        tokens.txt         `index string` of a few tokens (string array read back on demand)
        rope_swa.f32 / rope_csa.f32   ref/ds41/rope.py rope_table, 64 positions, [cos | sin] float32
        tensors.txt        name type n_dims ne0 ne1 ne2 shard abs_offset nbytes              (tools/ds41/gguf_io.py load_shard)
        dequant.txt        name file n_floats [row,row,...]; dequant/NNN.f32 = the oracle's float32 decode (ref/ds41/weights.py)
        experts.txt        layer expert blob half0 half1; the three layouts (tools/ds41/expert_layout.py), files under experts/
    real_headers.txt     the REAL model's metadata and 1,006-tensor table from the saved headers JSON, as text (no weights needed)
    real/*.gguf          (--sparse-real) the real model as 12 sparse shards: real metadata and tensor table, data a hole (411 GB apparent, a few MB on disk)
    real_sparse.txt      their tensor offsets and sizes, written by Python
    real_expect.txt      byte totals of the real model computed here from the same table, for the C++ tally to match
Everything is deterministic.  The C++ side parses these files with nothing but the standard library.
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import pathlib
import struct
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
for p in (REPO, REPO / "tools", REPO / "tools" / "ds41"):
    sys.path.insert(0, str(p))

import expert_layout as EL  # noqa: E402
import gguf_io as G  # noqa: E402
import make_mini_gguf as MM  # noqa: E402
from ref.ds41.config import Config, layer_modes  # noqa: E402
from ref.ds41.weights import GGUFWeights, canonical_name  # noqa: E402

HEADERS = REPO / "third_party" / "deepseek-v41-flash-reference" / "gguf-headers-mxxm-t-MXFP4.json.gz"


# ------------------------------------------------------------------------------------------------ value text
def kv_line(key: str, v) -> str | None:
    """One `K` line for a metadata value; None for what the tests do not compare (long / multi-line strings)."""
    if isinstance(v, bool):
        return f"K {key} b {int(v)}"
    if isinstance(v, int):
        return f"K {key} i {v}"
    if isinstance(v, float):
        return f"K {key} f {v!r}"
    if isinstance(v, str):
        import re
        m = re.fullmatch(r"<array len (\d+)>", v)
        if m:
            return f"K {key} ac {m.group(1)}"
        if "\n" in v or len(v) > 120:
            return None
        return f"K {key} s {v}"
    if isinstance(v, list):
        if v and isinstance(v[0], str):
            return f"K {key} as {len(v)}"
        if any(isinstance(x, float) for x in v):
            return f"K {key} af {len(v)} " + " ".join(repr(float(x)) for x in v)
        if v and isinstance(v[0], bool):
            return f"K {key} ai {len(v)} " + " ".join(str(int(x)) for x in v)
        return f"K {key} ai {len(v)} " + " ".join(str(int(x)) for x in v)
    return None


def num_list(v) -> str:
    return " ".join(repr(float(x)) if isinstance(x, float) else str(int(x)) for x in v)


# ------------------------------------------------------------------------------------------------ the mini model
def write_mini(out: pathlib.Path, seed: int) -> list[pathlib.Path]:
    cfg = dataclasses.replace(MM.MiniConfig(), n_shards=3, seed=seed)
    mini = out / "mini"
    mini.mkdir(parents=True, exist_ok=True)
    for old in mini.glob("*.gguf"):
        old.unlink()
    MM.build_mini(mini, cfg)
    return sorted(mini.glob("*.gguf"))


def write_golden(out: pathlib.Path, paths: list[pathlib.Path]) -> None:
    gold = out / "golden"
    (gold / "dequant").mkdir(parents=True, exist_ok=True)
    (gold / "experts").mkdir(parents=True, exist_ok=True)

    # ---- shards and tensor table (gguf_io: the repo's reader)
    recs = [G.load_shard(p, i) for i, p in enumerate(paths)]
    with open(gold / "shards.txt", "w") as f:
        for r in recs:
            f.write(f"{r.index} {r.file} {r.size} {r.alignment} {r.data_start} {r.kv.get('split.no')} {r.kv.get('split.count')} {r.kv.get('split.tensors.count')}\n")
    with open(gold / "tensors.txt", "w") as f:
        for r in recs:
            for t in r.tensors:
                ne = list(t.dims) + [1] * (3 - len(t.dims))
                f.write(f"{t.name} {t.type} {len(t.dims)} {ne[0]} {ne[1]} {ne[2]} {t.shard} {t.abs_offset} {t.nbytes}\n")

    # ---- metadata, whole, from the first shard
    with open(gold / "meta.txt", "w") as f:
        for k, v in recs[0].kv.items():
            line = kv_line(k, v)
            if line:
                f.write(line + "\n")

    with open(gold / "tokens.txt", "w") as f:       # a few tokens of the string array, by index
        toks = recs[0].kv["tokenizer.ggml.tokens"]
        for i in (0, 1, 2, 17, len(toks) - 1):
            f.write(f"{i} {toks[i]}\n")

    # ---- the oracle's own view of the config and the layer map
    w = GGUFWeights(paths)
    vocab = w.tensor_info("embed.weight").ne[1]
    cfg = Config.from_gguf_metadata(w.metadata, vocab_size=vocab)
    with open(gold / "config.txt", "w") as f:
        for fld in dataclasses.fields(cfg):
            if fld.name == "max_seq_len":
                continue
            v = getattr(cfg, fld.name)
            f.write(f"{fld.name} {num_list(v) if isinstance(v, (tuple, list)) else (int(v) if isinstance(v, bool) else repr(v) if isinstance(v, float) else v)}\n")
    from ref.ds41.rope import layer_rope_table
    for name, layer in (("rope_swa", 0), ("rope_csa", cfg.kv_source_layers[0])):    # the two RoPE kinds: plain theta, YaRN; [cos | sin], 64 positions
        cos, sin = layer_rope_table(cfg, layer, 64)
        np.concatenate([cos.reshape(-1), sin.reshape(-1)]).astype(np.float32).tofile(gold / f"{name}.f32")
    with open(gold / "modes.txt", "w") as f:
        for l, m in enumerate(layer_modes(cfg)):
            f.write(f"{l} {m.value}\n")

    # ---- dequantised tensors (the oracle's decode): every dense tensor whole, token_embd / Engram tables by rows
    n = 0
    lines = []

    def put(name: str, arr: np.ndarray, rows=None) -> None:
        nonlocal n
        fn = f"dequant/{n:03d}.f32"
        np.ascontiguousarray(arr, dtype=np.float32).tofile(gold / fn)
        lines.append(f"{name} {fn} {arr.size}" + (f" {','.join(map(str, rows))}" if rows is not None else ""))
        n += 1

    for r in recs:
        for t in r.tensors:
            canon, expert = canonical_name(t.name)
            if expert is not None:
                continue
            if t.name == "token_embd.weight":
                rows = [0, 1, 2, 17, 255, 256, vocab - 1]
                put(t.name, w.rows(canon, np.array(rows)), rows)
            elif "engram_embed" in t.name:
                rows = [0, 1, 17, 1000, int(t.dims[1]) - 1]
                put(t.name, w.rows(canon, np.array(rows)), rows)
            else:
                put(t.name, w.get(canon))
    (gold / "dequant.txt").write_text("\n".join(lines) + "\n")

    # ---- experts in the three layouts
    geom = EL.ExpertGeom(hidden=cfg.dim, ff=cfg.moe_inter_dim, n_expert=cfg.n_routed_experts)
    lines = []
    for layer, e in [(0, 0), (0, 15), (1, 7), (3, 5), (5, 15), (7, 8)]:
        raw = w.expert_raw(layer, e)
        blob = b"".join(np.ascontiguousarray(raw[k]).tobytes() for k in ("w1", "w3", "w2"))
        h0, h1 = EL.blob_to_halves(blob, geom)
        base = f"experts/L{layer}_E{e}"
        (gold / (base + ".blob")).write_bytes(blob)
        (gold / (base + ".half0")).write_bytes(h0)
        (gold / (base + ".half1")).write_bytes(h1)
        lines.append(f"{layer} {e} {base}.blob {base}.half0 {base}.half1")
    (gold / "experts.txt").write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------------ the real model's headers
def write_real(out: pathlib.Path) -> None:
    with gzip.open(HEADERS, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    files = [k for k in raw if "-of-00012" in k]
    assert len(files) == 12, files
    first = raw[files[0]]
    lines = []
    for i, name in enumerate(files):
        lines.append(f"S {i} {name}")
    for k, v in first["kv"].items():
        line = kv_line(k, v)
        if line:
            lines.append(line)
    totals: dict[str, int] = {}
    n = 0
    for i, name in enumerate(files):
        for tname, ty, dims, off in raw[name]["tensors"]:
            nb = G.nbytes_for(ty, dims)
            lines.append(f"T {i} {tname} {ty} {len(dims)} " + " ".join(str(int(d)) for d in dims) + f" {off} {nb}")
            n += 1
            grp = "expert" if "_exps" in tname else "engram_embed" if "engram_embed" in tname else "embd" if tname == "token_embd.weight" else "dense"
            totals[grp] = totals.get(grp, 0) + nb
    (out / "real_headers.txt").write_text("\n".join(lines) + "\n")
    exp = [f"n_tensors {n}"] + [f"{k} {v}" for k, v in sorted(totals.items())]
    (out / "real_expect.txt").write_text("\n".join(exp) + "\n")


# ------------------------------------------------------------------------------------------------ the real model as SPARSE files
_GGUF_T = {"u32": 4, "i32": 5, "f32": 6, "bool": 7, "string": 8, "array": 9, "i64": 11, "f64": 12}
_TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "MXFP4": 39}


def _pstr(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pscalar(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", _GGUF_T["bool"]) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", _GGUF_T["i64"]) + struct.pack("<q", v) if abs(v) >= 2 ** 31 else struct.pack("<I", _GGUF_T["u32"]) + struct.pack("<I", v) if v >= 0 else struct.pack("<I", _GGUF_T["i32"]) + struct.pack("<i", v)
    if isinstance(v, float):
        if struct.unpack("<f", struct.pack("<f", v))[0] == v:
            return struct.pack("<I", _GGUF_T["f32"]) + struct.pack("<f", v)
        return struct.pack("<I", _GGUF_T["f64"]) + struct.pack("<d", v)
    if isinstance(v, str):
        return struct.pack("<I", _GGUF_T["string"]) + _pstr(v)
    raise TypeError(type(v))


def _parray(items: list) -> bytes:
    if not items:
        return struct.pack("<I", _GGUF_T["array"]) + struct.pack("<IQ", _GGUF_T["i32"], 0)
    if isinstance(items[0], str):
        return struct.pack("<I", _GGUF_T["array"]) + struct.pack("<IQ", _GGUF_T["string"], len(items)) + b"".join(_pstr(x) for x in items)
    if isinstance(items[0], float):
        return struct.pack("<I", _GGUF_T["array"]) + struct.pack("<IQ", _GGUF_T["f32"], len(items)) + struct.pack(f"<{len(items)}f", *items)
    if max(abs(int(x)) for x in items) >= 2 ** 31:
        return struct.pack("<I", _GGUF_T["array"]) + struct.pack("<IQ", _GGUF_T["i64"], len(items)) + struct.pack(f"<{len(items)}q", *items)
    return struct.pack("<I", _GGUF_T["array"]) + struct.pack("<IQ", _GGUF_T["i32"], len(items)) + struct.pack(f"<{len(items)}i", *items)


def write_sparse_real(out: pathlib.Path) -> None:
    """The real model's 12 shards as GGUF files whose DATA is a hole: the real metadata (arrays the headers JSON only summarises are synthesised), the real
    1,006-tensor table with the real offsets, and a data section of the real size that was never written (411 GB of apparent size, a few MB on disk)."""
    d = out / "real"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(HEADERS, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    files = [k for k in raw if "-of-00012" in k]
    vocab = 129280
    lines = []
    total = 0
    for i, name in enumerate(files):
        kv = dict(raw[name]["kv"])
        if i == 0:
            kv["tokenizer.ggml.tokens"] = [f"<tok{j}>" for j in range(vocab)]
            kv["tokenizer.ggml.token_type"] = [1] * vocab
            kv["deepseek41.engram.token_map"] = [(j * 7919) % 99092 for j in range(vocab)]
            kv.pop("tokenizer.ggml.merges", None)
            kv["split.tensors.count"] = sum(len(raw[f]["tensors"]) for f in files)
        body = b"".join(_pstr(k) + (_parray(v) if isinstance(v, list) else _pscalar(v)) for k, v in kv.items())
        tens = raw[name]["tensors"]
        tbl = b""
        data_end = 0
        for tname, ty, dims, off in tens:
            tbl += _pstr(tname) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", _TYPE_ID[ty], off)
            nb = G.nbytes_for(ty, dims)
            data_end = max(data_end, off + nb)
            lines.append(f"{tname} {i} {off} {nb} {ty}")
        header = struct.pack("<IIQQ", 0x46554747, 3, len(tens), len(kv)) + body + tbl
        data_start = (len(header) + 31) // 32 * 32
        path = d / name
        with open(path, "wb") as f:
            f.write(header)
            f.write(b"\0" * (data_start - len(header)))
            f.truncate(data_start + (data_end + 31) // 32 * 32)
        total += path.stat().st_size
        lines.append(f"#shard {i} {name} {path.stat().st_size} {data_start}")
    (out / "real_sparse.txt").write_text("\n".join(lines) + "\n")
    print(f"sparse real model: 12 shards, {total / 1e9:.1f} GB apparent")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sparse-real", action="store_true", help="also write the real model as 12 sparse shards under <out>/real (411 GB apparent, no data)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = write_mini(out, a.seed)
    write_golden(out, paths)
    write_real(out)
    if a.sparse_real:
        write_sparse_real(out)
    print(f"fixture written to {out}: {len(paths)} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
