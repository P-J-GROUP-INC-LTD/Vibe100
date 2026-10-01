"""gguf_io.py - shard-level GGUF I/O for the DS41 tools.

* load_shard(path)         header of a real .gguf (via tools/gguf_reader.py) -> ShardRec, with absolute data offsets
* load_headers_json(path)  the saved output of tools/ds41_gguf_remote_headers.py -> {file: ShardRec}; no data
                           section start is known there (the JSON has neither the header length nor arrays > 64
                           entries), so abs_offset / data_start / size stay None
* write_gguf(...)          a GGUF v3 writer for F32 / F16 / BF16 / Q8_0 / MXFP4 tensors and every metadata type,
                           with `general.alignment` support (tools/gguf_writer.py only writes F32, F16 and Q2_0)
* read_tensor_f32(...)     dequantise one tensor of a real file (mmap) to a numpy array shaped (ne_last, ..., ne0)

tools/gguf_reader.py and tools/gguf_writer.py are used, not modified.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import mmap
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # tools/  (gguf_reader.py)
sys.path.insert(0, str(HERE))          # tools/ds41/

import ggml_codecs  # noqa: E402
from gguf_reader import BLOCK_GEOMETRY, GGUF_MAGIC, GGUFFile  # noqa: E402

DEFAULT_ALIGNMENT = 32
_SPLIT_RE = re.compile(r"^(?P<stem>.*)-(?P<no>\d{5})-of-(?P<n>\d{5})\.gguf$")
_ARRAY_RE = re.compile(r"^<array len (\d+)>$")


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def nbytes_for(type_name: str, dims) -> int | None:
    """Bytes of a tensor of `dims` in `type_name`; None when the type is unknown or the row is not block-aligned."""
    geom = BLOCK_GEOMETRY.get(type_name)
    if geom is None or not dims:
        return None
    be, bb = geom
    if int(dims[0]) % be:
        return None
    n = 1
    for d in dims:
        n *= int(d)
    return n // be * bb


def array_len(v) -> int | None:
    """Length of a metadata array, whether it is the real list or the "<array len N>" summary of the headers JSON."""
    if isinstance(v, (list, tuple)):
        return len(v)
    if isinstance(v, str):
        m = _ARRAY_RE.match(v)
        if m:
            return int(m.group(1))
    return None


def is_array_summary(v) -> bool:
    return isinstance(v, str) and _ARRAY_RE.match(v) is not None


# ---------------------------------------------------------------------------------------------- records
@dataclasses.dataclass
class TensorRec:
    name: str                      # as spelled in the file
    dims: tuple                    # ggml ne0, ne1, ...
    type: str                      # "MXFP4", "Q8_0", ...
    offset: int                    # relative to the data section of its shard
    shard: int                     # index into the list of ShardRec
    nbytes: int | None             # from type + dims
    abs_offset: int | None         # data_start + offset; None when the data section start is unknown


@dataclasses.dataclass
class ShardRec:
    index: int
    file: str                      # basename
    path: str | None               # absolute path of a real file; None for the headers JSON
    size: int | None               # file size
    alignment: int
    data_start: int | None         # absolute offset of the data section
    kv: dict
    tensors: list
    from_json: bool = False

    @property
    def data_bytes(self) -> int:
        """Bytes of the data section implied by the tensor table (aligned end of the last tensor)."""
        end = 0
        for t in self.tensors:
            if t.nbytes is not None:
                end = max(end, align_up(t.offset + t.nbytes, self.alignment))
        return end


def load_shard(path, index: int = 0) -> ShardRec:
    p = Path(path).resolve()
    g = GGUFFile(p)
    tensors = [TensorRec(t.name, tuple(int(d) for d in t.shape), t.type_name, t.offset, index,
                         nbytes_for(t.type_name, t.shape), g.data_start + t.offset) for t in g.tensors]
    return ShardRec(index, p.name, str(p), p.stat().st_size, g.alignment, g.data_start, dict(g.metadata), tensors)


def load_headers_json(path) -> list:
    """The JSON that tools/ds41_gguf_remote_headers.py prints ({file: {"kv": ..., "tensors": [[name, type, dims, off]]}}),
    gzip or plain, -> [ShardRec] in file order."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for i, (fname, v) in enumerate(raw.items()):
        kv = dict(v["kv"])
        align = kv.get("general.alignment")
        align = align if isinstance(align, int) and align else DEFAULT_ALIGNMENT
        tensors = [TensorRec(name, tuple(int(d) for d in dims), ty, int(off), i, nbytes_for(ty, dims), None)
                   for name, ty, dims, off in v["tensors"]]
        out.append(ShardRec(i, fname, None, None, align, None, kv, tensors, from_json=True))
    return out


def expand_split(paths) -> list:
    """Paths as given -> paths with the missing siblings of `-00001-of-00012.gguf` style splits added (llama.cpp's
    convention), in split order; a directory expands to its *.gguf files."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.glob("*.gguf")))
        else:
            out.append(p)
    seen: dict[Path, None] = {}
    for p in out:
        m = _SPLIT_RE.match(p.name)
        if m:
            n = int(m.group("n"))
            for k in range(1, n + 1):
                sib = p.with_name(f"{m.group('stem')}-{k:05d}-of-{n:05d}.gguf")
                if sib.exists() or sib == p:
                    seen.setdefault(sib, None)
        else:
            seen.setdefault(p, None)
    return list(seen)


# ---------------------------------------------------------------------------------------------- writer
_META_IDS = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "string": 8,
             "array": 9, "u64": 10, "i64": 11, "f64": 12}
_META_FMT = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i", "f32": "<f",
             "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_scalar(type_name: str, v) -> bytes:
    if type_name == "string":
        return _pack_str(v)
    return struct.pack(_META_FMT[type_name], v)


def pack_kv(key: str, type_name: str, value) -> bytes:
    """One metadata entry.  type_name is a scalar ("u32", "f32", "bool", "string", ...) or "array:<scalar>"."""
    out = bytearray(_pack_str(key))
    if type_name.startswith("array:"):
        elem = type_name.split(":", 1)[1]
        out += struct.pack("<II", _META_IDS["array"], _META_IDS[elem])
        out += struct.pack("<Q", len(value))
        if elem == "string":
            for v in value:
                out += _pack_str(v)
        else:
            out += struct.pack(f"<{len(value)}{_META_FMT[elem][1]}", *value)
    else:
        out += struct.pack("<I", _META_IDS[type_name]) + _pack_scalar(type_name, value)
    return bytes(out)


def write_gguf(path, kv, tensors, alignment: int = DEFAULT_ALIGNMENT) -> dict:
    """Write a GGUF v3 file.

    kv:       [(key, type_name, value), ...]  in order; `general.alignment` is added when alignment != 32
    tensors:  [(name, dims, type_name, data_bytes), ...]  dims in ggml order (ne0 first); data_bytes is the encoded
              tensor and its length must match type + dims
    Returns {"data_start": int, "offsets": {name: relative offset}, "size": file size}.
    """
    kv = list(kv)
    if alignment != DEFAULT_ALIGNMENT:
        kv.append(("general.alignment", "u32", alignment))
    head = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(kv)))
    for k, t, v in kv:
        head += pack_kv(k, t, v)
    offsets: dict = {}
    off = 0
    for name, dims, type_name, data in tensors:
        want = nbytes_for(type_name, dims)
        if want is None or want != len(data):
            raise ValueError(f"{name}: {len(data)} bytes of {type_name} for dims {list(dims)} (expected {want})")
        offsets[name] = off
        head += _pack_str(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        head += struct.pack("<IQ", ggml_codecs.TYPE_ID[type_name], off)
        off = align_up(off + len(data), alignment)
    data_start = align_up(len(head), alignment)
    head += b"\0" * (data_start - len(head))
    p = Path(path)
    with p.open("wb") as fh:
        fh.write(head)
        for _name, _dims, _type, data in tensors:
            fh.write(data)
            fh.write(b"\0" * (align_up(len(data), alignment) - len(data)))
    return {"data_start": data_start, "offsets": offsets, "size": p.stat().st_size}


# ---------------------------------------------------------------------------------------------- tensor reads
class MappedFile:
    """Read-only mmap of one file; .view(offset, nbytes) is a zero-copy uint8 numpy array."""

    def __init__(self, path):
        self.path = str(path)
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = len(self._mm)
        self._arr = np.frombuffer(self._mm, dtype=np.uint8)

    def view(self, offset: int, nbytes: int) -> np.ndarray:
        if offset < 0 or offset + nbytes > self.size:
            raise ValueError(f"{self.path}: bytes [{offset}, {offset + nbytes}) are outside the file ({self.size} B): "
                             "truncated or incomplete download?")
        return self._arr[offset:offset + nbytes]

    def close(self) -> None:
        self._arr = None  # drop the buffer export before closing the map
        try:
            self._mm.close()
        except BufferError:
            pass  # a caller still holds a view of the map; it is released when that view is dropped
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_tensor_f32(path, abs_offset: int, type_name: str, dims) -> np.ndarray:
    """Dequantise one tensor to float32, shaped like numpy (ne_last, ..., ne1, ne0)."""
    nbytes = nbytes_for(type_name, dims)
    if nbytes is None:
        raise ValueError(f"cannot size a {type_name} tensor of dims {list(dims)}")
    with MappedFile(path) as mf:
        raw = bytes(mf.view(abs_offset, nbytes))
    return ggml_codecs.decode(type_name, raw).reshape(tuple(int(d) for d in dims)[::-1])
