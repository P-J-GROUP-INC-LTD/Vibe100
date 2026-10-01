"""ref/ds41/selfcheck.py - constants self-checks against the target GGUF (mxxm-t MXFP4) and the official config.

    python -m ref.ds41.selfcheck [--tokenizer tokenizer.json]

Checks (each prints PASS/FAIL):
  * Engram primes (48), offsets (48) and multipliers (8) derived exactly as engram.py does equal the values stored in
    the GGUF metadata (`deepseek41.engram.*`, read from the saved header
    third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz);
  * each layer's 24 primes sum to `engram_num_embeddings[layer]` (= the table row count in both the config and the GGUF);
  * the compressed-token map has 129,280 entries over 99,092 ids; with a tokenizer.json (HF
    deepseek-ai/DeepSeek-V4.1-Flash) it is rebuilt from scratch and compared entry for entry with the map stored in the
    GGUF (fixture tests/data/engram_token_map_mxxm.u32.gz = gzip of 129,280 little-endian uint32, extracted from the real shard 1) and, when the `tokenizers`
    package is present, with the official `engram.build_compressed_token_map`.
"""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

from .config import Config
from .engram import build_compressed_token_map, build_layout, compute_hash_multipliers

REPO = pathlib.Path(__file__).resolve().parents[2]
REF_DIR = REPO / "third_party" / "deepseek-v41-flash-reference"
HEADER_GZ = REF_DIR / "gguf-headers-mxxm-t-MXFP4.json.gz"
TOKEN_MAP_FIXTURE = pathlib.Path(__file__).resolve().parent / "tests" / "data" / "engram_token_map_mxxm.u32.gz"
TOKENIZER_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/tokenizer.json"


def load_gguf_metadata(path=HEADER_GZ) -> dict:
    """The `kv` dict of shard 1 from the saved header (arrays longer than 64 entries are summarised as strings)."""
    hdr = json.load(gzip.open(path))
    first = sorted(k for k in hdr if "00001-of" in k)[0]
    return hdr[first]["kv"]


def load_gguf_tensor_table(path=HEADER_GZ) -> dict:
    """{tensor name: (ggml type, [ne0, ne1, ...])} over all 12 main shards (the DSpark sidecar is a different file)."""
    hdr = json.load(gzip.open(path))
    out = {}
    for fn, d in hdr.items():
        if "DSpark" in fn:
            continue
        for name, typ, dims, _off in d["tensors"]:
            out[name] = (typ, dims)
    return out


def tokenizer_json_path(download: bool = True) -> pathlib.Path | None:
    """$DS41_TOKENIZER_JSON, else a cached copy, else (download=True) fetch it from the Hugging Face repo."""
    env = os.environ.get("DS41_TOKENIZER_JSON")
    if env and pathlib.Path(env).exists():
        return pathlib.Path(env)
    cache = pathlib.Path(os.environ.get("DS41_CACHE", tempfile.gettempdir())) / "ds41_cache" / "tokenizer.json"
    if cache.exists() and cache.stat().st_size > 1_000_000:
        return cache
    if not download:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["curl", "-sSL", "--max-time", "120", "-o", str(cache), TOKENIZER_URL], check=True,
                       capture_output=True)
        if cache.stat().st_size > 1_000_000:
            return cache
    except Exception:
        pass
    return None


def load_token_map_fixture(path=TOKEN_MAP_FIXTURE) -> np.ndarray:
    """The GGUF's own `deepseek41.engram.token_map` (int64 [129280])."""
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype="<u4").astype(np.int64)


def derive_engram(cfg: Config) -> dict:
    """primes / offsets / multipliers / row counts exactly as engram.py derives them from the config."""
    lay = build_layout(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size)
    return dict(primes=lay.flat_primes, offsets=lay.offsets, num_embeddings=lay.num_embeddings,
                multipliers=compute_hash_multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size,
                                                     cfg.engram_compressed_vocab_size))


def run_checks(tokenizer_json: str | None = None, verbose: bool = True) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def rec(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

    kv = load_gguf_metadata()
    cfg = Config.from_official_json(REF_DIR / "inference" / "config.json")
    d = derive_engram(cfg)
    n_layers = len(cfg.engram_layer_ids)
    g_primes = np.array(kv["deepseek41.engram.primes"], dtype=np.int64).reshape(n_layers, -1)
    g_offsets = np.array(kv["deepseek41.engram.offsets"], dtype=np.int64).reshape(n_layers, -1)
    g_mult = np.array(kv["deepseek41.engram.multipliers"], dtype=np.int64).reshape(n_layers, -1)
    rec("engram primes == GGUF deepseek41.engram.primes", np.array_equal(d["primes"], g_primes),
        f"{d['primes'].size} values, e.g. {d['primes'][0, :3].tolist()} ... {d['primes'][-1, -1]}")
    rec("engram offsets == GGUF deepseek41.engram.offsets", np.array_equal(d["offsets"], g_offsets),
        f"{d['offsets'].size} values")
    rec("engram multipliers == GGUF deepseek41.engram.multipliers", np.array_equal(d["multipliers"], g_mult),
        f"{d['multipliers'].size} values (rng seeds 10007*{list(cfg.engram_layer_ids)})")
    sums = d["primes"].sum(axis=1).tolist()
    rec("sum of each layer's 24 primes == engram_num_embeddings (config.json)",
        sums == list(cfg.engram_num_embeddings), f"{sums}")
    rec("... == GGUF deepseek41.engram.num_embeddings", sums == list(kv["deepseek41.engram.num_embeddings"]))
    rec("24 primes per layer, all distinct across layers", d["primes"].shape == (2, 24) and len(set(d["primes"].ravel())) == 48)
    rec("config.json engram fields == GGUF metadata",
        list(cfg.engram_layer_ids) == kv["deepseek41.engram.layer_ids"]
        and cfg.engram_head_dim == kv["deepseek41.engram.head_dim"]
        and cfg.engram_n_heads == kv["deepseek41.engram.n_heads"]
        and cfg.engram_max_ngram_size == kv["deepseek41.engram.max_ngram_size"]
        and cfg.engram_pad_id == kv["deepseek41.engram.pad_token_id"]
        and cfg.engram_compressed_vocab_size == kv["deepseek41.engram.compressed_vocab_size"])

    fx = load_token_map_fixture()
    rec("GGUF token_map: 129,280 entries over 99,092 ids",
        fx.shape == (129280,) and fx.min() == 0 and fx.max() + 1 == 99092 and len(np.unique(fx)) == 99092)
    rec("GGUF token_map is in first-appearance order", bool(np.all(fx <= np.maximum.accumulate(np.r_[-1, fx[:-1]]) + 1)))
    path = tokenizer_json or tokenizer_json_path()
    if path is None:
        rec("token map rebuilt from tokenizer.json", False, "SKIPPED: tokenizer.json unavailable (set DS41_TOKENIZER_JSON)")
    else:
        lk, n = build_compressed_token_map(path)
        rec("oracle-rebuilt token map == GGUF token_map (all 129,280 entries)",
            n == 99092 and np.array_equal(np.array(lk), fx), f"rebuilt size {n}")
        try:
            from tokenizers import Tokenizer
            sys.path.insert(0, str(REF_DIR / "inference"))
            import engram as official_engram            # needs sympy + torch
            tk = type("T", (), {"backend_tokenizer": Tokenizer.from_file(str(path)),
                                "__len__": lambda self: self.backend_tokenizer.get_vocab_size(True)})()
            lo, no = official_engram.build_compressed_token_map(tk)
            rec("official engram.build_compressed_token_map == oracle's", lo == lk and no == n)
        except ImportError as e:
            rec("official build_compressed_token_map", False, f"SKIPPED: {e}")
    return results


if __name__ == "__main__":
    tj = sys.argv[sys.argv.index("--tokenizer") + 1] if "--tokenizer" in sys.argv else None
    res = run_checks(tj)
    sys.exit(0 if all(ok for _, ok, d in res if not d.startswith("SKIPPED")) else 1)
