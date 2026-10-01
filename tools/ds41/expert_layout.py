#!/usr/bin/env python3
"""expert_layout.py - routed experts out of the GGUF, as the GPU blob and as the two CPU halves, plus their checks.

    # write expert (layer 3, expert 17) as l03_e017.blob / .half0 / .half1
    python3 tools/ds41/expert_layout.py extract manifest.json --layer 3 --expert 17 --out /tmp/experts
    # prove the three layouts are the same expert (bit-exact dequantisation, partial sums add up)
    python3 tools/ds41/expert_layout.py verify /models/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf --sample 4

The first argument is a manifest.json written by manifest.py from real shards (headers-JSON manifests have no absolute
offsets and are refused), or the .gguf shards themselves.  The shard files are mmapped; nothing is read but the expert.

Layouts (docs/deepseek/CONTRACTS.md, include/strata/ds41/geometry.hpp), E = experts, H = hidden, F = expert FF:
  GGUF    ffn_gate_exps / ffn_up_exps  [ne0 = H, ne1 = F, ne2 = E]  expert e = rows [e*F, (e+1)*F), H/32 blocks of 17 B
          ffn_down_exps                [ne0 = F, ne1 = H, ne2 = E]  expert e = rows [e*H, (e+1)*H), F/32 blocks of 17 B
  blob    [gate][up][down]            the expert's three GGUF slices back to back   (real: 18,800,640 B)
  half k  [gate rows kF/2..][up rows kF/2..][down: every row, blocks k*F/64 ..]       (real: 9,400,320 B)
          k = 0, 1; F/2 must be a whole number of 32-value blocks (F % 64 == 0)

verify checks, per expert:
  1. halves_to_blob(blob_to_halves(blob)) == blob                       (the split is a bijection)
  2. dequantised halves, concatenated, equal the dequantised blob        (gate/up along rows, down along columns)
  3. y_0 + y_1 == y on random inputs, with the contract's math (silu(min(g, 10)) * clamp(u, -10, 10), * route weight,
     down projection) in float64; the two halves' partial results add up to the full expert
  4. with --c-oracle: the numpy MXFP4 dequantisation equals GGML's own dequantize_row_mxfp4 compiled from C, bit for bit
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ggml_codecs as C  # noqa: E402
import gguf_io as G  # noqa: E402

BLOCK_BYTES = 17
QK = 32


@dataclasses.dataclass(frozen=True)
class ExpertGeom:
    hidden: int
    ff: int
    n_expert: int

    def __post_init__(self):
        if self.hidden % QK or self.ff % (2 * QK):
            raise ValueError(f"hidden {self.hidden} must be a multiple of 32 and ff {self.ff} of 64 "
                             "(a half must be whole MXFP4 blocks)")

    gate_row_blocks = property(lambda s: s.hidden // QK)
    down_row_blocks = property(lambda s: s.ff // QK)
    gate_row_bytes = property(lambda s: s.gate_row_blocks * BLOCK_BYTES)
    down_row_bytes = property(lambda s: s.down_row_blocks * BLOCK_BYTES)
    gate_bytes = property(lambda s: s.ff * s.gate_row_bytes)
    down_bytes = property(lambda s: s.hidden * s.down_row_bytes)
    blob_bytes = property(lambda s: 2 * s.gate_bytes + s.down_bytes)
    half_ff = property(lambda s: s.ff // 2)
    half_down_blocks = property(lambda s: s.down_row_blocks // 2)
    half_gate_bytes = property(lambda s: s.half_ff * s.gate_row_bytes)
    half_down_row_bytes = property(lambda s: s.half_down_blocks * BLOCK_BYTES)
    half_down_bytes = property(lambda s: s.hidden * s.half_down_row_bytes)
    half_bytes = property(lambda s: 2 * s.half_gate_bytes + s.half_down_bytes)


REAL = ExpertGeom(hidden=5120, ff=2304, n_expert=384)


# ---------------------------------------------------------------------------------------------- layouts (pure)
def _u8(b) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8) if isinstance(b, (bytes, bytearray, memoryview)) else \
        np.ascontiguousarray(b).reshape(-1).view(np.uint8)


def blob_parts(blob, g: ExpertGeom):
    """blob -> (gate (ff, row_bytes), up, down (hidden, down_row_bytes)) views."""
    b = _u8(blob)
    if b.size != g.blob_bytes:
        raise ValueError(f"blob is {b.size} bytes, expected {g.blob_bytes}")
    gate = b[:g.gate_bytes].reshape(g.ff, g.gate_row_bytes)
    up = b[g.gate_bytes:2 * g.gate_bytes].reshape(g.ff, g.gate_row_bytes)
    down = b[2 * g.gate_bytes:].reshape(g.hidden, g.down_row_bytes)
    return gate, up, down


def blob_to_halves(blob, g: ExpertGeom):
    """GPU blob -> (half0, half1) as bytes, CONTRACTS.md layout."""
    gate, up, down = blob_parts(blob, g)
    hf, hb = g.half_ff, g.half_down_blocks
    down3 = down.reshape(g.hidden, g.down_row_blocks, BLOCK_BYTES)
    out = []
    for k in range(2):
        out.append(b"".join((gate[k * hf:(k + 1) * hf].tobytes(), up[k * hf:(k + 1) * hf].tobytes(),
                             np.ascontiguousarray(down3[:, k * hb:(k + 1) * hb, :]).tobytes())))
    return out[0], out[1]


def half_parts(half, g: ExpertGeom):
    h = _u8(half)
    if h.size != g.half_bytes:
        raise ValueError(f"half is {h.size} bytes, expected {g.half_bytes}")
    gate = h[:g.half_gate_bytes].reshape(g.half_ff, g.gate_row_bytes)
    up = h[g.half_gate_bytes:2 * g.half_gate_bytes].reshape(g.half_ff, g.gate_row_bytes)
    down = h[2 * g.half_gate_bytes:].reshape(g.hidden, g.half_down_row_bytes)
    return gate, up, down


def halves_to_blob(h0, h1, g: ExpertGeom) -> bytes:
    """Inverse of blob_to_halves."""
    g0, u0, d0 = half_parts(h0, g)
    g1, u1, d1 = half_parts(h1, g)
    down = np.concatenate([d0.reshape(g.hidden, g.half_down_blocks, BLOCK_BYTES),
                           d1.reshape(g.hidden, g.half_down_blocks, BLOCK_BYTES)], axis=1)
    return b"".join((np.concatenate([g0, g1]).tobytes(), np.concatenate([u0, u1]).tobytes(),
                     np.ascontiguousarray(down).tobytes()))


# ---------------------------------------------------------------------------------------------- dequantisation, math
def _deq(rows_u8: np.ndarray, n_rows: int) -> np.ndarray:
    return C.dequant_mxfp4(rows_u8).reshape(n_rows, -1)


def dequant_blob(blob, g: ExpertGeom):
    """-> (Wg (ff, hidden), Wu (ff, hidden), Wd (hidden, ff)) float32."""
    gate, up, down = blob_parts(blob, g)
    return _deq(gate, g.ff), _deq(up, g.ff), _deq(down, g.hidden)


def dequant_half(half, g: ExpertGeom):
    """-> (Wg_k (ff/2, hidden), Wu_k, Wd_k (hidden, ff/2)) float32."""
    gate, up, down = half_parts(half, g)
    return _deq(gate, g.half_ff), _deq(up, g.half_ff), _deq(down, g.hidden)


def _silu(x):
    return 0.5 * x * (1.0 + np.tanh(0.5 * x))      # = x * sigmoid(x), no overflow for very negative x


def expert_hidden(x, Wg, Wu, route_w=1.0, limit=10.0):
    """h = silu(min(Wg x, limit)) * clamp(Wu x, -limit, limit) * route_w   (CONTRACTS.md; float64)."""
    g = Wg.astype(np.float64) @ x
    u = Wu.astype(np.float64) @ x
    n_clamped = int((g > limit).sum() + (np.abs(u) > limit).sum())
    return _silu(np.minimum(g, limit)) * np.clip(u, -limit, limit) * route_w, n_clamped


def expert_forward(x, Wg, Wu, Wd, route_w=1.0, limit=10.0):
    h, n = expert_hidden(x, Wg, Wu, route_w, limit)
    return Wd.astype(np.float64) @ h, n


def verify_expert(blob, h0, h1, g: ExpertGeom, *, seed: int = 0, n_inputs: int = 4, c_oracle: bool = False) -> dict:
    """The four checks of the module docstring.  -> report dict; report["ok"] is the verdict."""
    rep: dict = {"ok": True, "checks": {}}

    def check(name, ok, detail=""):
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}
        rep["ok"] = rep["ok"] and bool(ok)

    check("halves_to_blob(blob_to_halves(blob)) == blob", halves_to_blob(h0, h1, g) == bytes(_u8(blob)))
    Wg, Wu, Wd = dequant_blob(blob, g)
    Wg0, Wu0, Wd0 = dequant_half(h0, g)
    Wg1, Wu1, Wd1 = dequant_half(h1, g)
    same = (np.array_equal(np.concatenate([Wg0, Wg1]), Wg) and np.array_equal(np.concatenate([Wu0, Wu1]), Wu)
            and np.array_equal(np.concatenate([Wd0, Wd1], axis=1), Wd))
    check("dequantised halves == dequantised blob (gate/up rows, down columns)", same)
    rng = np.random.default_rng(seed)
    worst, clamped = 0.0, 0
    for i in range(n_inputs):
        x = rng.standard_normal(g.hidden) * (1.0 if i % 2 == 0 else 6.0)   # the louder inputs reach the clamps
        w = float(rng.uniform(0.2, 1.5))
        y, nc = expert_forward(x, Wg, Wu, Wd, w)
        h_0, _ = expert_hidden(x, Wg0, Wu0, w)
        h_1, _ = expert_hidden(x, Wg1, Wu1, w)
        y2 = Wd0.astype(np.float64) @ h_0 + Wd1.astype(np.float64) @ h_1
        scale = max(float(np.abs(y).max()), 1e-300)
        worst = max(worst, float(np.abs(y - y2).max()) / scale)
        clamped += nc
    check("y_0 + y_1 == y on random inputs", worst < 1e-12, f"max relative error {worst:.2e}, clamp hits {clamped}")
    rep["max_rel_err"] = worst
    if c_oracle:
        import ggml_c_oracle as O
        if not O.available():
            check("numpy dequant == GGML C dequantize_row_mxfp4", False, "oracle unavailable: " + O.why_not())
        else:
            raw = bytes(_u8(blob))
            same_c = np.array_equal(np.concatenate([Wg.reshape(-1), Wu.reshape(-1), Wd.reshape(-1)]).view(np.uint32),
                                    np.concatenate([O.dequantize_mxfp4(raw[:g.gate_bytes]),
                                                    O.dequantize_mxfp4(raw[g.gate_bytes:2 * g.gate_bytes]),
                                                    O.dequantize_mxfp4(raw[2 * g.gate_bytes:])]).view(np.uint32))
            check("numpy dequant == GGML C dequantize_row_mxfp4", same_c)
    return rep


# ---------------------------------------------------------------------------------------------- GGUF access
class ExpertSource:
    """Routed experts of a manifest (real shards), by mmap."""

    def __init__(self, manifest: dict, base_dir=None):
        if manifest.get("ok") is False:
            raise ValueError("this manifest records a failed validation (ok = false); fix the model first")
        if manifest.get("from_headers_json") or any(s["data_start"] is None for s in manifest["shards"]):
            raise ValueError("this manifest was built from a headers JSON and has no absolute data offsets; "
                             "run manifest.py on the real shard files")
        self.m = manifest
        self.base_dir = Path(base_dir) if base_dir else None
        self._files: dict = {}
        t = manifest["tensors"]
        self.layers = sorted({v["layer"] for k, v in t.items() if v["group"] == "expert"})
        if not self.layers:
            raise ValueError("the manifest has no routed expert tensors")
        gate = t[f"blk.{self.layers[0]}.ffn_gate_exps.weight"]
        down = t[f"blk.{self.layers[0]}.ffn_down_exps.weight"]
        if gate["type"] != "MXFP4" or down["type"] != "MXFP4":
            raise ValueError(f"routed experts are {gate['type']}/{down['type']}, not MXFP4: no blob layout")
        self.geom = ExpertGeom(hidden=gate["dims"][0], ff=gate["dims"][1], n_expert=gate["dims"][2])
        if down["dims"] != [self.geom.ff, self.geom.hidden, self.geom.n_expert]:
            raise ValueError(f"ffn_down_exps dims {down['dims']} do not match ffn_gate_exps {gate['dims']}")

    def _file(self, shard: int) -> G.MappedFile:
        if shard not in self._files:
            s = self.m["shards"][shard]
            path = (self.base_dir / s["file"]) if self.base_dir else Path(s["path"])
            self._files[shard] = G.MappedFile(path)
        return self._files[shard]

    def _view(self, layer: int, which: str) -> np.ndarray:
        t = self.m["tensors"][f"blk.{layer}.ffn_{which}_exps.weight"]
        g = self.geom
        rows, row_bytes = (g.ff, g.gate_row_bytes) if which in ("gate", "up") else (g.hidden, g.down_row_bytes)
        arr = self._file(t["shard"]).view(t["abs_offset"], t["nbytes"])
        return arr.reshape(g.n_expert, rows, row_bytes)

    def slices(self, layer: int, e: int):
        if not 0 <= e < self.geom.n_expert:
            raise IndexError(f"expert {e} outside 0..{self.geom.n_expert - 1}")
        return self._view(layer, "gate")[e], self._view(layer, "up")[e], self._view(layer, "down")[e]

    def blob(self, layer: int, e: int) -> bytes:
        gate, up, down = self.slices(layer, e)
        return gate.tobytes() + up.tobytes() + down.tobytes()

    def halves(self, layer: int, e: int):
        return blob_to_halves(self.blob(layer, e), self.geom)

    def close(self):
        for f in self._files.values():
            f.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def load_manifest_arg(inputs, geometry: str = "self") -> dict:
    """A manifest.json path, or .gguf shard path(s) / a directory -> manifest dict (refusing a failed validation)."""
    import manifest as M
    if len(inputs) == 1 and str(inputs[0]).endswith(".json"):
        return json.loads(Path(inputs[0]).read_text())
    files, source = M.collect_inputs(inputs, dspark="none")
    manifest, F = M.analyze(files, geometry=geometry, source=source)
    if F.errors:
        raise SystemExit("the GGUF does not satisfy the deepseek41 contract:\n  " +
                         "\n  ".join(f"{f.code}: {f.text}" for f in F.errors[:10]))
    return manifest


# ---------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "verify"):
        p = sub.add_parser(name)
        p.add_argument("inputs", nargs="+", help="manifest.json from manifest.py, or the .gguf shard(s)")
        p.add_argument("--layer", type=int)
        p.add_argument("--expert", type=int)
        p.add_argument("--geometry", choices=("ds41", "self"), default="ds41")
        p.add_argument("--base-dir", help="directory holding the shard files if they moved since manifest.py ran")
        if name == "extract":
            p.add_argument("--out", required=True, help="output directory")
        else:
            p.add_argument("--sample", type=int, default=0, help="verify N random (layer, expert) pairs")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--c-oracle", action="store_true", help="also compare with GGML's C dequantizer")
    args = ap.parse_args(argv)
    manifest = load_manifest_arg(args.inputs, args.geometry)
    with ExpertSource(manifest, args.base_dir) as src:
        g = src.geom
        if args.cmd == "extract":
            if args.layer is None or args.expert is None:
                ap.error("extract needs --layer and --expert")
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            blob = src.blob(args.layer, args.expert)
            h0, h1 = blob_to_halves(blob, g)
            stem = out / f"l{args.layer:02d}_e{args.expert:03d}"
            for suffix, data in ((".blob", blob), (".half0", h0), (".half1", h1)):
                Path(str(stem) + suffix).write_bytes(data)
            print(f"layer {args.layer} expert {args.expert}: blob {len(blob):,} B, halves {len(h0):,} B + {len(h1):,} B "
                  f"-> {stem}.blob / .half0 / .half1")
            return 0
        pairs = []
        if args.layer is not None and args.expert is not None:
            pairs.append((args.layer, args.expert))
        rng = np.random.default_rng(args.seed)
        for _ in range(args.sample):
            pairs.append((int(rng.choice(src.layers)), int(rng.integers(0, g.n_expert))))
        if not pairs:
            ap.error("verify needs --layer and --expert, or --sample N")
        bad = 0
        for layer, e in pairs:
            blob = src.blob(layer, e)
            h0, h1 = blob_to_halves(blob, g)
            rep = verify_expert(blob, h0, h1, g, seed=args.seed + layer * 1000 + e, c_oracle=args.c_oracle)
            print(f"layer {layer:2d} expert {e:3d}: {'OK' if rep['ok'] else 'FAIL'}  "
                  f"(blob {len(blob):,} B, max rel err {rep['max_rel_err']:.1e})")
            for name, c in rep["checks"].items():
                if not c["ok"]:
                    print(f"    FAILED: {name} {c['detail']}")
            bad += not rep["ok"]
        return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
