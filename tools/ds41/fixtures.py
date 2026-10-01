"""fixtures.py - shared fixtures for the tools/ds41 tests (not a test module itself)."""
from __future__ import annotations

import atexit
import copy
import dataclasses
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

import gguf_io as G  # noqa: E402
import make_mini_gguf as MM  # noqa: E402

REF = REPO / "third_party" / "deepseek-v41-flash-reference"
MXXM_HEADERS = REF / "gguf-headers-mxxm-t-MXFP4.json.gz"
VCRUZ_HEADERS = REF / "gguf-headers-vcruz305-Q2_K.json.gz"

_tmp_root: Path | None = None
_cache: dict = {}


def tmp_root() -> Path:
    global _tmp_root
    if _tmp_root is None:
        _tmp_root = Path(tempfile.mkdtemp(prefix="ds41_tests_"))
        atexit.register(shutil.rmtree, _tmp_root, ignore_errors=True)
    return _tmp_root


def mini(**overrides) -> dict:
    """Build (once per distinct override set) a mini GGUF; -> build_mini() result."""
    key = tuple(sorted(overrides.items()))
    if key not in _cache:
        cfg = dataclasses.replace(MM.MiniConfig(), **overrides)
        _cache[key] = MM.build_mini(tmp_root() / f"mini{len(_cache)}", cfg)
    return _cache[key]


def fresh_mini(**overrides) -> dict:
    """A mini GGUF in its own directory (for tests that damage the files)."""
    cfg = dataclasses.replace(MM.MiniConfig(), **overrides)
    return MM.build_mini(tmp_root() / f"scratch{len(_cache)}_{id(cfg)}", cfg)


def headers(path) -> list:
    """Fresh ShardRecs of a headers JSON (tests mutate them)."""
    return G.load_headers_json(path)


def clone(recs) -> list:
    return copy.deepcopy(recs)
