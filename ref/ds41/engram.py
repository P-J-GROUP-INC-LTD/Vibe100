"""ref/ds41/engram.py - Engram: hashed n-gram embeddings added into the residual stream.

Transcribes third_party/deepseek-v41-flash-reference/inference/engram.py (all of it) and model.py:Engram (328-366):

  * `is_prime`, `find_next_prime`, `build_layout`      engram.py:9-14, EngramLayout.from_args (103-127)
  * `compute_hash_multipliers`                          engram.py:64-84
  * `build_compressed_token_map*`                       engram.py:17-61  (stdlib re-implementation of the HF-`tokenizers`
                                                        normaliser pipeline; validated against the real thing in tests)
  * `NgramHasher`                                       engram.py:NgramHashState (129-184), text-only (no image DEAD tokens)
  * `engram_layer`                                      model.py:Engram.forward (350-366)

All of it is deterministic and independent of the weights: the 48 row addresses a token needs are a pure function of
the last four token ids, which is what lets the engine prefetch Engram rows from the SSD.
"""
from __future__ import annotations

import dataclasses
import json
import re
import unicodedata
from typing import Sequence

import numpy as np

from .config import Config
from .ops import _wide, sigmoid
from .quant import QuantConfig
from .ops import linear


# ---------------------------------------------------------------------------------------------------------------
# primes / layout / multipliers
# ---------------------------------------------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin (exact for n < 3.3e24); the reference uses sympy.isprime."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """engram.py:find_next_prime: the smallest prime above `start` that has not been handed out yet."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclasses.dataclass(frozen=True)
class EngramLayout:
    """engram.py:EngramLayout - bucket moduli per (layer, n-gram order, head), plus derived offsets."""
    max_ngram_size: int
    layer_ids: tuple
    primes: tuple            # [layer][order (2-gram .. max_ngram_size-gram)][head]
    n_heads: int

    @property
    def flat_primes(self) -> np.ndarray:            # [layer, (max_ngram_size-1)*n_heads] int64
        return np.array([[p for per in layer for p in per] for layer in self.primes], dtype=np.int64)

    @property
    def offsets(self) -> np.ndarray:                # [layer, cols]: cumulative sums of the layer's own primes
        return np.array([np.cumsum([0, *row[:-1]]) for row in self.flat_primes], dtype=np.int64)

    @property
    def num_embeddings(self) -> tuple:              # table rows per layer = sum of that layer's primes
        return tuple(int(r.sum()) for r in self.flat_primes)


def build_layout(layer_ids: Sequence[int], max_ngram_size: int, n_heads: int, engram_vocab_size: int) -> EngramLayout:
    """engram.py:EngramLayout.from_args.  Primes are drawn in (layer, order, head) order, each order's search restarting
    at engram_vocab_size - 1 and skipping every prime already handed out, so all ranges are disjoint."""
    primes, seen = [], set()
    for _ in layer_ids:
        per_ngram = []
        for _ in range(max_ngram_size - 1):
            sizes, current = [], engram_vocab_size - 1
            for _ in range(n_heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                sizes.append(current)
            per_ngram.append(tuple(sizes))
        primes.append(tuple(per_ngram))
    return EngramLayout(max_ngram_size, tuple(layer_ids), tuple(primes), n_heads)


def compute_hash_multipliers(layer_ids: Sequence[int], max_ngram_size: int, tokenizer_vocab_size: int) -> np.ndarray:
    """engram.py:compute_hash_multipliers: [layer, max_ngram_size] odd int64, from default_rng(10007 * layer_id)."""
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        gen = np.random.default_rng(10007 * layer_id)
        v = gen.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(v * 2 + 1)
    return np.stack(rows)


# ---------------------------------------------------------------------------------------------------------------
# token -> compressed id (129,280 -> 99,092)
# ---------------------------------------------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte <-> printable-unicode table (the ByteLevel alphabet)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_CHAR_TO_BYTE = {c: b for b, c in _bytes_to_unicode().items()}

# Rust char::is_whitespace == Unicode White_Space (Python's str.isspace() also counts U+001C..U+001F)
_RUST_WS = frozenset("\t\n\x0b\x0c\r \x85\xa0           "
                     "     　")
_SENTINEL = ""
_WS_RUN = re.compile(r"[ \t\r\n]+")


def bytelevel_decode_token(token: str) -> str:
    """HF ByteLevel decoder on ONE token: map each char through the byte table and decode UTF-8 lossily; a token with
    any char outside the table (specials such as '<｜begin▁of▁sentence｜>', or one containing a plain space) passes
    through unchanged."""
    try:
        raw = bytes(_CHAR_TO_BYTE[c] for c in token)
    except KeyError:
        return token
    return raw.decode("utf-8", errors="replace")


def normalize_token_text(s: str) -> str:
    """The normaliser Sequence of engram.py:build_compressed_token_map, with `unicodedata` instead of Rust:
    NFKC, NFD, StripAccents (drop every combining mark: categories Mn, Mc, Me), Lowercase (per char), collapse [ \\t\\r\\n]+ to one space,
    protect a lone space with a sentinel, Strip (Rust whitespace), restore the sentinel."""
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", s))
    s = "".join(c for c in s if not unicodedata.category(c).startswith("M"))    # Mn, Mc and Me (checked vs tokenizers)
    s = "".join(c.lower() for c in s)
    s = _WS_RUN.sub(" ", s)
    if s == " ":
        s = _SENTINEL
    i, j = 0, len(s)
    while i < j and s[i] in _RUST_WS:
        i += 1
    while j > i and s[j - 1] in _RUST_WS:
        j -= 1
    return s[i:j].replace(_SENTINEL, " ")


def build_compressed_token_map_from_tokens(token_strings: Sequence[str]) -> tuple[list[int], int]:
    """engram.py:build_compressed_token_map given the tokenizer's raw token strings (byte-level alphabet), indexed by
    id.  Returns (lookup[id] -> compressed id in first-appearance order, size of the compressed vocabulary)."""
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(token_strings)
    for token_id, tok in enumerate(token_strings):
        text = bytelevel_decode_token(tok)
        if "�" in text:
            key = tok                                    # partial UTF-8 byte token: key by its raw form
        else:
            normalized = normalize_token_text(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def tokens_from_tokenizer_json(path_or_obj) -> list[str]:
    """id -> token string for a HF `tokenizer.json` (BPE vocab, overridden by added_tokens at their ids)."""
    t = json.load(open(path_or_obj, encoding="utf-8")) if isinstance(path_or_obj, (str, bytes)) or hasattr(
        path_or_obj, "__fspath__") else path_or_obj
    vocab = t["model"]["vocab"]
    added = {a["id"]: a["content"] for a in t.get("added_tokens", [])}
    n = max(max(vocab.values()), max(added, default=-1)) + 1
    out = [None] * n
    for tok, i in vocab.items():
        out[i] = tok
    for i, c in added.items():
        out[i] = c
    assert all(x is not None for x in out), "tokenizer.json has holes in its id space"
    return out


def build_compressed_token_map(tokenizer_json) -> tuple[list[int], int]:
    return build_compressed_token_map_from_tokens(tokens_from_tokenizer_json(tokenizer_json))


# ---------------------------------------------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------------------------------------------

class NgramHasher:
    """engram.py:NgramHashState (text only).  `__call__(input_ids, start_pos)` returns the table rows
    [s, n_engram_layers, n_hash_cols] (int64) of the n-grams ending at each position; the cache of compressed ids
    carries the look-back across the prefill/decode split.  Position t hashes tokens t, t-1, t-2, t-3 (the pad id's
    compressed id where the sequence has not started)."""

    def __init__(self, cfg: Config, token_map, layout: EngramLayout | None = None, *, multipliers=None,
                 max_seq_len: int | None = None):
        self.cfg = cfg
        self.layout = layout or build_layout(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads,
                                             cfg.engram_vocab_size)
        self.token_map = np.asarray(token_map, dtype=np.int64)
        self.pad_id = int(self.token_map[cfg.engram_pad_id])
        vocab = int(self.token_map.max()) + 1
        assert vocab == cfg.engram_compressed_vocab_size, (vocab, cfg.engram_compressed_vocab_size)
        self.primes = self.layout.flat_primes.reshape(len(cfg.engram_layer_ids), cfg.engram_max_ngram_size - 1,
                                                      cfg.engram_n_heads)
        self.offsets = self.layout.offsets
        self.multipliers = (np.asarray(multipliers, dtype=np.int64) if multipliers is not None else
                            compute_hash_multipliers(cfg.engram_layer_ids, cfg.engram_max_ngram_size, vocab))
        self.cache = np.zeros(max_seq_len or cfg.max_seq_len, dtype=np.int64)

    def __call__(self, input_ids: np.ndarray, start_pos: int) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int64)
        s = ids.shape[0]
        self.cache[start_pos:start_pos + s] = self.token_map[ids]
        positions = np.arange(start_pos, start_pos + s)
        tokens, blocked = [], np.zeros(s, dtype=bool)
        for shift in range(self.cfg.engram_max_ngram_size):
            source = self.cache[np.maximum(positions - shift, 0)]
            blocked = blocked | (positions < shift)
            tokens.append(np.where(blocked, self.pad_id, source))
        tokens = np.stack(tokens, axis=-1)                                   # [s, max_ngram_size]
        # XOR the multiplied ids one lookback at a time: the running value after step i hashes the (i+1)-gram
        products = tokens[:, None, :] * self.multipliers[None]               # [s, L, max_ngram_size]  (int64, no overflow)
        rolling, hashes = products[..., 0], []
        for i in range(1, self.cfg.engram_max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[:, i - 1][None])  # [s, L, n_heads]
        return np.concatenate(hashes, axis=-1) + self.offsets[None]          # [s, L, cols]


def constants_from_gguf_metadata(meta) -> dict:
    """The Engram constants a GGUF carries (`deepseek41.engram.*`): the authoritative values for THAT file.
    -> dict(layout, multipliers, token_map).  `offsets` are re-derived from the primes and must equal the stored ones
    (the self-check in selfcheck.py proves derived == stored for the released file)."""
    p = "deepseek41.engram."
    layers, heads, ngram = meta[p + "layer_ids"], meta[p + "n_heads"], meta[p + "max_ngram_size"]
    flat = np.asarray(meta[p + "primes"], dtype=np.int64).reshape(len(layers), ngram - 1, heads)
    layout = EngramLayout(ngram, tuple(layers), tuple(tuple(tuple(int(x) for x in h) for h in lay) for lay in flat), heads)
    stored = np.asarray(meta[p + "offsets"], dtype=np.int64).reshape(len(layers), -1)
    assert np.array_equal(layout.offsets, stored), "GGUF engram.offsets disagree with its primes"
    mult = np.asarray(meta[p + "multipliers"], dtype=np.int64).reshape(len(layers), ngram)
    tm = meta.get(p + "token_map")
    assert tm is not None and not isinstance(tm, str), "GGUF engram.token_map missing (header summaries drop it)"
    return dict(layout=layout, multipliers=mult, token_map=np.asarray(tm, dtype=np.int64))


# ---------------------------------------------------------------------------------------------------------------
# the layer
# ---------------------------------------------------------------------------------------------------------------

def engram_layer(x: np.ndarray, rows: np.ndarray, *, wkv: np.ndarray, q_weight: np.ndarray, k_weight: np.ndarray,
                 eps: float, quant: QuantConfig | None = None, clamp_value: float = 1e-6) -> np.ndarray:
    """model.py:Engram.forward.  x [s, hc, dim] (the 4-copy stream); rows [s, n_hash_cols, head_dim] (the looked-up
    table rows, dequantised); wkv [(hc+1)*dim, n_hash_cols*head_dim]; q_weight/k_weight [hc, dim].

        kv    = wkv(concat rows)  ->  key [hc, dim], value [dim]
        rstd  = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)          per (token, copy), over dim
        dot   = sum_d h * (q_w * k_w) * key  *  rstd * dim^-0.5
        gate  = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
        out   = h + gate * value                                            per copy
    """
    s, hc, d = x.shape
    kv = linear(rows.reshape(s, -1), wkv, act_quant=True, quant=quant)
    key = kv[:, : hc * d].reshape(s, hc, d)
    value = kv[:, hc * d:]
    weight = _wide(q_weight) * _wide(k_weight)
    h = _wide(x)
    rstd = (1.0 / np.sqrt(np.mean(np.square(h), axis=-1) + eps)) * (1.0 / np.sqrt(np.mean(np.square(key), axis=-1) + eps))
    dot = np.sum(h * weight * key, axis=-1) * rstd * d ** -0.5
    gate = sigmoid(np.copysign(np.sqrt(np.maximum(np.abs(dot), clamp_value)), dot))
    return h + gate[..., None] * _wide(value)[:, None, :]
