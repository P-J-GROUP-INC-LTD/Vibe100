"""cpu_pack_xcheck.py - optional cross-checks of the Python tools and the oracle against DS-C's C++ CPU kernels.

DS-C (src/ds41/cpu/mxfp4_expert.cpp) packs the two CPU halves of a GPU blob in C++ (what the engine's loader will do);
expert_layout.py does it in Python (what the tests and tools use).  Both implement docs/deepseek/CONTRACTS.md, and this
compiles the C++ side into a shared library and lets the tests compare them byte for byte at the real dimensions.

The same library carries two more entry points, for the tests that tie the Python side to the kernels:
  * `quantize_act(x, isa)`  the activation quantiser of CONTRACTS.md on the scalar / AVX2 / AVX-512 implementation, to compare with the
                            oracle's `ref.ds41.quant.quantize_int8_blocks` bit for bit (test_quant_xcheck.py);
  * `expert_run(blob, hidden, ff, half, x, w, isa)`  one expert (or one CPU half of it) of ANY runtime dimensions the kernels accept
                            (hidden % 128 == 0, ff % 256 == 0): proves that the mini GGUF's experts run through the C++ kernels.

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
#include <cstring>
#include <vector>
using namespace strata::ds41::cpu;
static Isa isa_of(int i) { return i == 0 ? Isa::kScalar : i == 1 ? Isa::kAvx2 : Isa::kAvx512; }
extern "C" void ds41_pack_cpu_half(const uint8_t* blob, int h, uint8_t* out) {
    strata::ds41::cpu::pack_cpu_half(blob, h, out);
}
extern "C" int ds41_isa_supported(int isa) { return isa_supported(isa_of(isa)) ? 1 : 0; }
// n values (a multiple of 128) -> natural-order int8 and the fp32 scale per 32-block
extern "C" void ds41_quantize_act(const float* x, int n, int isa, int8_t* q, float* scale) {
    static ActQ a;
    quantize_act(x, n, a, isa_of(isa));
    act_unpack(a, n, q);
    std::memcpy(scale, a.scale, (size_t) (n / 32) * sizeof(float));
}
// One expert of a blob [gate][up][down] with runtime dims, whole (half = -1) or one CPU half (0 / 1), for T tokens: y[T][hidden] += expert(x) * w.
extern "C" void ds41_expert_run(const uint8_t* blob, int hidden, int ff, int half, const float* x, int T, const float* w, float* y, int isa) {
    const size_t grow = (size_t) (hidden / 32) * 17, drow = (size_t) (ff / 32) * 17;
    ExpertView v;
    v.hidden = hidden;
    v.down_row_stride = drow;
    v.gate = blob;
    v.up = blob + (size_t) ff * grow;
    v.down = blob + 2 * (size_t) ff * grow;
    v.ff = ff;
    if (half >= 0) {
        v.ff = ff / 2;
        v.gate += (size_t) half * v.ff * grow;
        v.up += (size_t) half * v.ff * grow;
        v.down += (size_t) half * (v.ff / 32) * 17;
    }
    std::vector<ActQ> xq((size_t) T);
    std::vector<ExpertScratch> sc(1);
    quantize_acts(x, hidden, T, xq.data(), isa_of(isa));
    expert_run(isa_of(isa), v, xq.data(), T, w, sc[0], y);
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
                        str((SRC / "mxfp4_internal.hpp").stat().st_mtime_ns) +
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
    lib.ds41_isa_supported.argtypes = [ctypes.c_int]
    lib.ds41_isa_supported.restype = ctypes.c_int
    lib.ds41_quantize_act.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.ds41_expert_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
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


ISAS = {"scalar": 0, "avx2": 1, "avx512": 2}


def isa_supported(name: str) -> bool:
    lib = _get()
    return lib is not None and bool(lib.ds41_isa_supported(ISAS[name]))


def quantize_act(x, isa: str = "scalar"):
    """The CPU kernels' activation quantiser (CONTRACTS.md "Activations") on float32 x [n] (n a multiple of 128): -> (q int8 [n], d float32 [n / 32])."""
    lib = _get()
    if lib is None:
        raise RuntimeError("DS-C kernels unavailable: " + _why)
    xs = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    n = xs.size
    q = np.zeros(n, np.int8)
    d = np.zeros(n // 32, np.float32)
    lib.ds41_quantize_act(xs.ctypes.data_as(ctypes.c_void_p), n, ISAS[isa], q.ctypes.data_as(ctypes.c_void_p), d.ctypes.data_as(ctypes.c_void_p))
    return q, d


def expert_run(blob, hidden: int, ff: int, half: int, x, w, isa: str = "scalar"):
    """One routed expert of a blob [gate][up][down] with runtime dimensions through the C++ kernels (they abort on dimensions they do not support:
    hidden % 128 and ff % 256, see expert_layout.ExpertGeom): x [T, hidden] float32, w [T] route weights -> y [T, hidden] float32.
    `half` = -1 the whole expert, 0 / 1 one CPU half (its partial y_k; the two partials add up to the whole)."""
    lib = _get()
    if lib is None:
        raise RuntimeError("DS-C kernels unavailable: " + _why)
    if hidden % 128 or ff % 256:
        raise ValueError(f"the C++ kernels need hidden % 128 == 0 and ff % 256 == 0, got hidden {hidden}, ff {ff}")
    b = np.frombuffer(blob, dtype=np.uint8) if isinstance(blob, (bytes, bytearray)) else np.ascontiguousarray(blob, dtype=np.uint8).reshape(-1)
    xs = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, hidden)
    ws = np.ascontiguousarray(w, dtype=np.float32).reshape(-1)
    y = np.zeros(xs.shape, np.float32)
    lib.ds41_expert_run(b.ctypes.data_as(ctypes.c_void_p), hidden, ff, half, xs.ctypes.data_as(ctypes.c_void_p), xs.shape[0],
                        ws.ctypes.data_as(ctypes.c_void_p), y.ctypes.data_as(ctypes.c_void_p), ISAS[isa])
    return y
