"""ds1_testlib.py - shared helpers of tools/ds41/test_ds1_*.py (not a test module): cached mini oracles and traces, fake "engine" traces with injected errors."""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ds1_compare as C  # noqa: E402
import fixtures as F  # noqa: E402
from ref.ds41 import trace_io as TI  # noqa: E402

VOCAB = 512
_models: dict = {}
_traces: dict = {}
_root: pathlib.Path | None = None


def root() -> pathlib.Path:
    """A scratch directory shared by the tests of one process (removed at exit by fixtures.tmp_root)."""
    global _root
    if _root is None:
        _root = F.tmp_root() / "ds1_tests"
        _root.mkdir(exist_ok=True)
    return _root


def gguf_shard1() -> pathlib.Path:
    return pathlib.Path(F.mini()["paths"][0])


def model(quant: str = "int8-kv", dtype: str = "float32", max_seq_len: int = 96):
    """The oracle on the mini GGUF (cached per process)."""
    key = (quant, dtype, max_seq_len)
    if key not in _models:
        _models[key] = C.load_oracle_model(gguf_shard1(), C.quant_from_name(quant), dtype=dtype, max_seq_len=max_seq_len)
    return _models[key]


def tokens(n: int, seed: int = 1) -> list:
    return [int(t) for t in np.random.default_rng(seed).integers(3, VOCAB, n)]


def oracle_trace(quant: str = "int8-kv", dtype: str = "float32", n: int = 20, seed: int = 1, mode: str = "token_by_token") -> TI.Trace:
    """The oracle's trace of `n` random tokens (cached per process)."""
    key = (quant, dtype, n, seed, mode)
    if key not in _traces:
        d = root() / f"oracle_{quant}_{dtype}_{n}_{seed}_{mode}"
        _traces[key] = TI.run_oracle_trace(model(quant, dtype), tokens(n, seed), d, mode=mode)
    return _traces[key]


def engine_copy(src: TI.Trace, name: str, *, mutate: dict | None = None, drop=None, meta: dict | None = None) -> TI.Trace:
    """A copy of trace `src` posing as an engine's: producer "engine", `mutate` {(stage, pos, layer): fn(array) -> array} applied to the arrays,
    `drop` an iterable of (stage, pos, layer) files to leave out, `meta` entries overriding trace.json's."""
    dst = root() / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src.path, dst)
    for key in drop or ():
        stage, pos, layer = key
        (dst / TI.stage_filename(stage, pos, layer)).unlink()
    for (stage, pos, layer), fn in (mutate or {}).items():
        p = dst / TI.stage_filename(stage, pos, layer)
        a = np.load(p)
        np.save(p, np.ascontiguousarray(fn(a), dtype=a.dtype))
    j = json.loads((dst / "trace.json").read_text())
    j["producer"] = "engine"
    j.update(meta or {})
    (dst / "trace.json").write_text(json.dumps(j))
    return TI.Trace(dst)


def noise(rel: float, seed: int = 0):
    """A mutation adding Gaussian noise of rms `rel` x the array's rms (an arbitrary, clearly wrong stage)."""
    def f(a):
        r = np.random.default_rng(seed)
        rms = float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2))) or 1.0
        return (a + rel * rms * r.standard_normal(a.shape)).astype(a.dtype)
    return f
