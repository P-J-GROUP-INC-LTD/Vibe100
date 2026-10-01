"""cpu_pack_xcheck.py - optional cross-check of expert_layout.blob_to_halves against DS-C's C++ `pack_cpu_half`.

DS-C (src/ds41/cpu/mxfp4_expert.cpp) packs the two CPU halves of a GPU blob in C++ (what the engine's loader will do);
expert_layout.py does it in Python (what the tests and tools use).  Both implement docs/deepseek/CONTRACTS.md, and this
compiles the C++ side into a shared library and lets the tests compare them byte for byte at the real dimensions.

It needs g++ / clang++ and the DS-C sources; `available()` is False otherwise (the tests then skip, and a change of
DS-C's API makes it skip with the compiler's message rather than fail).  Not needed for anything else.
"""
from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "ds41" / "cpu"
INC = REPO / "include"

_WRAP = """
#include "strata/ds41/cpu/mxfp4_expert.hpp"
extern "C" void ds41_pack_cpu_half(const uint8_t* blob, int h, uint8_t* out) {
    strata::ds41::cpu::pack_cpu_half(blob, h, out);
}
"""
_FILES = [("mxfp4_expert.cpp", []), ("mxfp4_avx2.cpp", ["-mavx2", "-mfma", "-mf16c"]),
          ("mxfp4_avx512.cpp", ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mavx512vnni", "-mfma",
                                "-mf16c"])]

_lib = None
_why = ""


def _build():
    global _why
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        _why = "no C++ compiler"
        return None
    if not all((SRC / f).is_file() for f, _ in _FILES) or not (INC / "strata/ds41/cpu/mxfp4_expert.hpp").is_file():
        _why = "DS-C sources (src/ds41/cpu/mxfp4_expert.cpp ...) not present"
        return None
    key = hashlib.sha1((_WRAP + "".join(str((SRC / f).stat().st_mtime_ns) for f, _ in _FILES) +
                        str((INC / "strata/ds41/cpu/mxfp4_expert.hpp").stat().st_mtime_ns) +
                        str((INC / "strata/ds41/geometry.hpp").stat().st_mtime_ns)).encode()).hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / f"ds41_cpu_pack_{key}"
    so = out_dir / "libpack.so"
    if not so.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "wrap.cpp").write_text(_WRAP)
        objs = []
        for name, flags in [("wrap.cpp", [])] + _FILES:
            src = out_dir / name if name == "wrap.cpp" else SRC / name
            obj = out_dir / (name + ".o")
            r = subprocess.run([cxx, "-O1", "-std=c++17", "-fPIC", "-c", str(src), "-o", str(obj), f"-I{INC}",
                                f"-I{SRC}", *flags], capture_output=True, text=True)
            if r.returncode != 0:
                _why = f"compiling {name} failed: " + r.stderr[-300:]
                return None
            objs.append(str(obj))
        r = subprocess.run([cxx, "-shared", "-o", str(so), *objs, "-lpthread"], capture_output=True, text=True)
        if r.returncode != 0:
            _why = "linking failed: " + r.stderr[-300:]
            return None
    lib = ctypes.CDLL(str(so))
    lib.ds41_pack_cpu_half.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    return lib


def _get():
    global _lib
    if _lib is None:
        try:
            _lib = _build() or False
        except Exception as exc:  # noqa: BLE001
            global _why
            _why = f"{type(exc).__name__}: {exc}"
            _lib = False
    return _lib or None


def available() -> bool:
    return _get() is not None


def why_not() -> str:
    _get()
    return _why


def pack_cpu_half(blob, h: int, half_bytes: int = 9_400_320) -> bytes:
    """DS-C's pack_cpu_half(blob, h) at the real dimensions (the blob must be 18,800,640 bytes)."""
    lib = _get()
    if lib is None:
        raise RuntimeError("DS-C pack_cpu_half unavailable: " + _why)
    b = np.frombuffer(blob, dtype=np.uint8) if isinstance(blob, (bytes, bytearray)) else np.ascontiguousarray(blob)
    out = np.zeros(half_bytes, dtype=np.uint8)
    lib.ds41_pack_cpu_half(b.ctypes.data_as(ctypes.c_void_p), h, out.ctypes.data_as(ctypes.c_void_p))
    return out.tobytes()
