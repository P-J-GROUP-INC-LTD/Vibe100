"""ggml_c_oracle.py - GGML's own C quantisation code, compiled on the fly, as a bit-exact oracle for ggml_codecs.py.

It compiles ggml/src/ggml-quants.c (plus a four-line wrapper for the inline BF16 / E8M0 helpers of ggml-impl.h) into a
shared library with the system C compiler and calls it through ctypes:

    quantize_row_mxfp4_ref / dequantize_row_mxfp4     (MXFP4, type 39)
    quantize_row_q8_0_ref  / dequantize_row_q8_0      (Q8_0)
    ggml_compute_fp32_to_bf16 / ggml_compute_bf16_to_fp32, ggml_e8m0_to_fp32_half

The ggml source tree is found, in order, at $STRATA_GGML_SRC (the directory that holds ggml-quants.c), then
<repo>/third_party/llama.cpp/ggml/src, then <repo>/build*/_deps/strata_llamacpp-src/ggml/src and, if $SP is set, the
same under $SP.  `available()` is False when no source or no C compiler is found; the tests then skip.
Nothing here is needed to read or write a GGUF.
"""
from __future__ import annotations

import ctypes
import glob
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

_WRAPPER = r"""
#include "ggml-impl.h"
void w_fp32_to_bf16(const float * x, uint16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) y[i] = GGML_FP32_TO_BF16(x[i]).bits;
}
void w_bf16_to_fp32(const uint16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; i++) { ggml_bf16_t h; h.bits = x[i]; y[i] = GGML_BF16_TO_FP32(h); }
}
void w_e8m0_half(float * y) {
    for (int i = 0; i < 256; i++) y[i] = GGML_E8M0_TO_FP32_HALF((uint8_t) i);
}
/* ggml-quants.c reaches these only from ggml_validate_row_data and its abort paths; they are never called here. */
void ggml_abort(const char * file, int line, const char * fmt, ...) { (void) file; (void) line; (void) fmt; __builtin_trap(); }
size_t ggml_row_size(enum ggml_type t, int64_t ne) { (void) t; (void) ne; return 0; }
size_t ggml_type_size(enum ggml_type t) { (void) t; return 0; }
const char * ggml_type_name(enum ggml_type t) { (void) t; return "?"; }
"""


def find_source() -> Path | None:
    cands: list[Path] = []
    env = os.environ.get("STRATA_GGML_SRC")
    if env:
        cands.append(Path(env))
    cands.append(REPO / "third_party" / "llama.cpp" / "ggml" / "src")
    roots = [REPO]
    if os.environ.get("SP"):
        roots.append(Path(os.environ["SP"]))
    for root in roots:
        for g in sorted(glob.glob(str(root / "build*" / "_deps" / "strata_llamacpp-src" / "ggml" / "src"))):
            cands.append(Path(g))
    for c in cands:
        if (c / "ggml-quants.c").is_file() and (c / "ggml-impl.h").is_file():
            return c
    return None


_lib = None
_why_not = ""


def _build() -> ctypes.CDLL | None:
    global _why_not
    src = find_source()
    if src is None:
        _why_not = "ggml source tree not found (set STRATA_GGML_SRC to the directory holding ggml-quants.c)"
        return None
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        _why_not = "no C compiler"
        return None
    key = hashlib.sha1((str(src) + str((src / "ggml-quants.c").stat().st_mtime_ns) + _WRAPPER).encode()).hexdigest()[:12]
    out_dir = Path(tempfile.gettempdir()) / f"ds41_ggml_oracle_{key}"
    so = out_dir / "libggmlq.so"
    if not so.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        wrapper = out_dir / "w.c"
        wrapper.write_text(_WRAPPER)
        cmd = [cc, "-O1", "-shared", "-fPIC", "-w", f"-I{src}", f"-I{src.parent / 'include'}", "-DGGML_COMMON_IMPL_C",
               "-o", str(so), str(wrapper), str(src / "ggml-quants.c"), "-lm"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not so.is_file():
            _why_not = "compiling ggml-quants.c failed: " + r.stderr[-400:]
            return None
    lib = ctypes.CDLL(str(so))
    fp, u8p, u16p = ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16)
    lib.quantize_row_mxfp4_ref.argtypes = [fp, u8p, ctypes.c_int64]
    lib.dequantize_row_mxfp4.argtypes = [u8p, fp, ctypes.c_int64]
    lib.quantize_row_q8_0_ref.argtypes = [fp, u8p, ctypes.c_int64]
    lib.dequantize_row_q8_0.argtypes = [u8p, fp, ctypes.c_int64]
    lib.w_fp32_to_bf16.argtypes = [fp, u16p, ctypes.c_int64]
    lib.w_bf16_to_fp32.argtypes = [u16p, fp, ctypes.c_int64]
    lib.w_e8m0_half.argtypes = [fp]
    return lib


def _get() -> ctypes.CDLL | None:
    global _lib
    if _lib is None:
        try:
            _lib = _build() or False
        except Exception as exc:  # noqa: BLE001 - any failure just means "no oracle here"
            global _why_not
            _why_not = f"{type(exc).__name__}: {exc}"
            _lib = False
    return _lib or None


def available() -> bool:
    return _get() is not None


def why_not() -> str:
    _get()
    return _why_not


def _f32p(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _need() -> ctypes.CDLL:
    lib = _get()
    if lib is None:
        raise RuntimeError("ggml C oracle unavailable: " + _why_not)
    return lib


def quantize_mxfp4(x) -> bytes:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(x.size // 32 * 17, dtype=np.uint8)
    _need().quantize_row_mxfp4_ref(_f32p(x), out.ctypes.data_as(ctypes.c_void_p), x.size)
    return out.tobytes()


def dequantize_mxfp4(raw) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).copy()
    out = np.zeros(b.size // 17 * 32, dtype=np.float32)
    _need().dequantize_row_mxfp4(b.ctypes.data_as(ctypes.c_void_p), _f32p(out), out.size)
    return out


def quantize_q8_0(x) -> bytes:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(x.size // 32 * 34, dtype=np.uint8)
    _need().quantize_row_q8_0_ref(_f32p(x), out.ctypes.data_as(ctypes.c_void_p), x.size)
    return out.tobytes()


def dequantize_q8_0(raw) -> np.ndarray:
    b = np.frombuffer(raw, dtype=np.uint8).copy()
    out = np.zeros(b.size // 34 * 32, dtype=np.float32)
    _need().dequantize_row_q8_0(b.ctypes.data_as(ctypes.c_void_p), _f32p(out), out.size)
    return out


def fp32_to_bf16(x) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.zeros(x.size, dtype=np.uint16)
    _need().w_fp32_to_bf16(_f32p(x), out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), x.size)
    return out


def bf16_to_fp32(bits) -> np.ndarray:
    b = np.ascontiguousarray(bits, dtype=np.uint16).reshape(-1)
    out = np.zeros(b.size, dtype=np.float32)
    _need().w_bf16_to_fp32(b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), _f32p(out), b.size)
    return out


def e8m0_half_table() -> np.ndarray:
    out = np.zeros(256, dtype=np.float32)
    _need().w_e8m0_half(_f32p(out))
    return out
