#!/usr/bin/env python3
"""ds1_tokenizer.py - the DeepSeek-V4.1-Flash tokenizer from the GGUF's own metadata (`tokenizer.ggml.*`): text <-> ids, and the chat template.

    ds1_tokenizer.py encode   --gguf SHARD1 [--no-special] [--bos]  "text" | --file F | (stdin)      -> ids (JSON list)
    ds1_tokenizer.py decode   --gguf SHARD1 --ids 1,2,3 [--skip-special]                           -> text
    ds1_tokenizer.py chat     --gguf SHARD1 --messages '[{"role":"user","content":"hi"}]' [--thinking] [--no-generation-prompt] [--ids]
    ds1_tokenizer.py info     --gguf SHARD1                                                           -> sizes, special ids, pre-tokenizer
    ds1_tokenizer.py selfcheck [--gguf SHARD1] [--tokenizer-json tokenizer.json]                    -> exit 0 pass, 1 mismatch, 2 could not run

(`--tokenizer-json F` builds the same tokenizer from a Hugging Face tokenizer.json instead of a GGUF.)

WHAT THE GGUF SAYS (the real mxxm-t file, third_party/deepseek-v41-flash-reference/gguf-headers-*.json.gz; the arrays are not in the vendored headers):
    tokenizer.ggml.model = "gpt2"            byte-level BPE (GPT-2 alphabet)
    tokenizer.ggml.pre   = "joyai-llm"       llama.cpp's name for this model family's pre-tokenizer: the three regexes below (= DeepSeek-V3's)
    tokenizer.ggml.tokens / .token_type / .merges   129,280 tokens (type 3 = control, 4 = user-defined: both matched literally in the text), 127,741 merges "a b"
    tokenizer.ggml.bos_token_id = 0, eos_token_id = 1, padding_token_id = 1 (the HF pad token <｜▁pad▁｜> is id 2: the Engram pad id), add_bos/add_eos = False
    tokenizer.chat_template                  the DeepSeek-V4 template (thinking mode, DSML tool calls)

ENCODING (identical to the Hugging Face tokenizer.json of deepseek-ai/DeepSeek-V4.1-Flash, which tests/test_ds1_tokenizer.py checks token for token):
    1. text is cut at every literal occurrence of a control / user-defined token (leftmost-longest; `parse_special=False` turns that off);
    2. the pieces in between go through three splitting regexes in sequence (HF `Split(..., Isolated)`: the matches AND the gaps between them become pieces):
           \\p{N}{1,3}
           [一-龥぀-ゟ゠-ヿ]+
           [!"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+
       (`\\s` is Unicode White_Space exactly, not Python's wider set; \\p{..} come from `unicodedata` (Unicode 14) plus UNICODE_PATCH, the code points the official tokenizer
       classifies that Unicode 14 does not);
    3. each piece is mapped byte by byte to the GPT-2 alphabet and merged by BPE (lowest merge rank first, leftmost on ties), no prefix space, no normaliser.
DECODING: the byte-level characters of every ordinary token are mapped back to bytes, special tokens contribute their literal text, the whole byte string is
decoded as UTF-8 once with replacement characters (so a multi-byte character split across tokens decodes right).

Pure Python + the standard library (numpy is not needed here).  `jinja2` is used for chat templates only to cross-check the native renderer.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import heapq
import json
import pathlib
import re
import sys
import unicodedata

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# token types of GGUF (llama.cpp llama_token_attr / gguf TokenType)
TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6

# ---------------------------------------------------------------------------------------------------------------
# the pre-tokenizer
# ---------------------------------------------------------------------------------------------------------------

# Unicode White_Space (Rust / Oniguruma `\s`); Python's own `\s` and str.isspace() also match U+001C..U+001F
WS_CLASS = "\\t\\n\\x0b\\x0c\\r \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000"

# the three regexes of tokenizer.json's pre_tokenizer, as written there (CR / LF spelled as escapes)
OFFICIAL_V3 = (
    r"\p{N}{1,3}",
    "[一-龥぀-ゟ゠-ヿ]+",
    r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
)

# tokenizer.ggml.pre -> regexes.  llama.cpp's JOYAI_LLM and DEEPSEEK3_LLM entries use this same list; an unknown name is an error (never guess).
PRE_TOKENIZERS = {"joyai-llm": OFFICIAL_V3, "deepseek-v3": OFFICIAL_V3, "deepseek-v4": OFFICIAL_V3}


# Python 3.11's `unicodedata` is Unicode 14; the official tokenizer (Hugging Face `tokenizers`, Oniguruma) knows later versions.  Code points that are unassigned in
# 14 but classified by the official tokenizer, as (first, last, group): "A" letter / mark (L or M), "N" number, "C" punctuation / symbol (P or S) - the three groups the
# pre-tokenizer regexes can tell apart.  Found by probing every code point with the official tokenizer.json's own pre-tokenizer (`probe_unicode_patch` below);
# tests/test_ds1_tokenizer.py `test_unicode_patch_is_complete` re-derives it when the `tokenizers` package and the tokenizer.json are available.
UNICODE_PATCH = (
    (0x897, 0x897, "A"), (0xcf3, 0xcf3, "A"), (0xece, 0xece, "A"), (0x1b4e, 0x1b4f, "C"), (0x1b7f, 0x1b7f, "C"), (0x1c89, 0x1c8a, "A"),
    (0x2427, 0x2429, "C"), (0x2ffc, 0x2fff, "C"), (0x31e4, 0x31e5, "C"), (0x31ef, 0x31ef, "C"), (0xa7cb, 0xa7cd, "A"), (0xa7da, 0xa7dc, "A"),
    (0x105c0, 0x105f3, "A"), (0x10d40, 0x10d49, "N"), (0x10d4a, 0x10d65, "A"), (0x10d69, 0x10d6d, "A"), (0x10d6e, 0x10d6e, "C"),
    (0x10d6f, 0x10d85, "A"), (0x10d8e, 0x10d8f, "C"), (0x10ec2, 0x10ec4, "A"), (0x10efc, 0x10eff, "A"), (0x1123f, 0x11241, "A"),
    (0x11380, 0x11389, "A"), (0x1138b, 0x1138b, "A"), (0x1138e, 0x1138e, "A"), (0x11390, 0x113b5, "A"), (0x113b7, 0x113c0, "A"),
    (0x113c2, 0x113c2, "A"), (0x113c5, 0x113c5, "A"), (0x113c7, 0x113ca, "A"), (0x113cc, 0x113d3, "A"), (0x113d4, 0x113d5, "C"),
    (0x113d7, 0x113d8, "C"), (0x113e1, 0x113e2, "A"), (0x116d0, 0x116e3, "N"), (0x11b00, 0x11b09, "C"), (0x11bc0, 0x11be0, "A"),
    (0x11be1, 0x11be1, "C"), (0x11bf0, 0x11bf9, "N"), (0x11f00, 0x11f10, "A"), (0x11f12, 0x11f3a, "A"), (0x11f3e, 0x11f42, "A"),
    (0x11f43, 0x11f4f, "C"), (0x11f50, 0x11f59, "N"), (0x11f5a, 0x11f5a, "A"), (0x1342f, 0x1342f, "A"), (0x13440, 0x13455, "A"),
    (0x13460, 0x143fa, "A"), (0x16100, 0x1612f, "A"), (0x16130, 0x16139, "N"), (0x16d40, 0x16d6c, "A"), (0x16d6d, 0x16d6f, "C"),
    (0x16d70, 0x16d79, "N"), (0x18cff, 0x18cff, "A"), (0x1b132, 0x1b132, "A"), (0x1b155, 0x1b155, "A"), (0x1cc00, 0x1ccef, "C"),
    (0x1ccf0, 0x1ccf9, "N"), (0x1cd00, 0x1ceb3, "C"), (0x1d2c0, 0x1d2d3, "N"), (0x1df25, 0x1df2a, "A"), (0x1e030, 0x1e06d, "A"),
    (0x1e08f, 0x1e08f, "A"), (0x1e4d0, 0x1e4ef, "A"), (0x1e4f0, 0x1e4f9, "N"), (0x1e5d0, 0x1e5f0, "A"), (0x1e5f1, 0x1e5fa, "N"),
    (0x1e5ff, 0x1e5ff, "C"), (0x1f6dc, 0x1f6dc, "C"), (0x1f774, 0x1f776, "C"), (0x1f77b, 0x1f77f, "C"), (0x1f7d9, 0x1f7d9, "C"),
    (0x1f8b2, 0x1f8bb, "C"), (0x1f8c0, 0x1f8c1, "C"), (0x1fa75, 0x1fa77, "C"), (0x1fa87, 0x1fa89, "C"), (0x1fa8f, 0x1fa8f, "C"),
    (0x1faad, 0x1faaf, "C"), (0x1fabb, 0x1fabf, "C"), (0x1fac6, 0x1fac6, "C"), (0x1face, 0x1facf, "C"), (0x1fada, 0x1fadc, "C"),
    (0x1fadf, 0x1fadf, "C"), (0x1fae8, 0x1fae9, "C"), (0x1faf7, 0x1faf8, "C"), (0x1fbcb, 0x1fbef, "C"), (0x2b739, 0x2b739, "A"),
    (0x2ebf0, 0x2ee5d, "A"), (0x31350, 0x323af, "A"),
)
_PATCH_PREFIX = {"A": "L", "N": "N", "C": "P"}          # the class an "A" / "N" / "C" code point is added to


@functools.lru_cache(maxsize=None)
def _class_content(prefix: str) -> str:
    """Content of a [...] class for the Unicode general categories starting with `prefix` (`unicodedata` + UNICODE_PATCH), as ranges of \\UXXXXXXXX escapes."""
    extra = {cp for lo, hi, g in UNICODE_PATCH if _PATCH_PREFIX[g] == prefix for cp in range(lo, hi + 1)}
    out, start, prev = [], None, None
    cat = unicodedata.category
    for cp in range(0x110000):
        if cat(chr(cp)).startswith(prefix) or cp in extra:
            if start is None:
                start = cp
            prev = cp
        elif start is not None:
            out.append((start, prev))
            start = None
    if start is not None:
        out.append((start, prev))

    def esc(c):
        return f"\\U{c:08x}"
    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in out)


def translate_pattern(pat: str, prop_content=None) -> str:
    """Rewrite an Oniguruma / Rust pattern for Python: `\\s` and `\\S` become the exact Unicode White_Space class (Python's own `\\s` also matches U+001C..U+001F),
    and `\\p{X}` becomes the class generated from `unicodedata` by `prop_content(X)` (None keeps `\\p{X}`, for the `regex` module)."""
    out, i, in_class = [], 0, False
    while i < len(pat):
        c = pat[i]
        if c == "\\":
            n = pat[i + 1]
            if n == "p" and pat[i + 2] == "{":
                j = pat.index("}", i)
                if prop_content is None:
                    out.append(pat[i:j + 1])
                else:
                    content = prop_content(pat[i + 3:j])
                    out.append(content if in_class else f"[{content}]")
                i = j + 1
                continue
            if n == "s":
                out.append(WS_CLASS if in_class else f"[{WS_CLASS}]")
            elif n == "S":
                out.append(f"[^{WS_CLASS}]")
            else:
                out.append(pat[i:i + 2])
            i += 2
            continue
        if c == "[" and not in_class:
            in_class = True
        elif c == "]" and in_class:
            in_class = False
        out.append(c)
        i += 1
    return "".join(out)


def compile_pretokenizer(regexes, engine: str = "auto"):
    """-> list of compiled patterns.  engine "regex" uses the `regex` module's \\p{..} classes, "re" the classes generated from `unicodedata`."""
    use_regex = False
    if engine == "regex":
        import regex  # noqa: F401  (its Unicode tables are newer than the official tokenizer's: differs on code points assigned after Unicode 15 / 16)
        use_regex = True
    out = []
    for r in regexes:
        if use_regex:
            import regex
            out.append(regex.compile(translate_pattern(r, None)))
        else:
            out.append(re.compile(translate_pattern(r, lambda name: _class_content(name))))
    return out


def split_isolated(pattern, text: str) -> list:
    """Hugging Face `Split(pattern, behavior="Isolated")`: the matches and the text between them, in order."""
    out, last = [], 0
    for m in pattern.finditer(text):
        s, e = m.span()
        if e == s:
            continue
        if s > last:
            out.append(text[last:s])
        out.append(text[s:e])
        last = e
    if last < len(text):
        out.append(text[last:])
    return out


def probe_unicode_patch(tokenizer_json) -> list:
    """Re-derive UNICODE_PATCH with the official tokenizer: for every code point >= 0x80 outside the CJK / kana ranges the second regex isolates, how does the
    official pre-tokenizer group it?  Needs the `tokenizers` package.  -> [(first, last, group)] where the official grouping differs from unicodedata's."""
    from tokenizers import Tokenizer
    pt = Tokenizer.from_file(str(tokenizer_json)).pre_tokenizer

    def n(s):
        return len(pt.pre_tokenize_str(s))

    def group_py(cp):
        c = unicodedata.category(chr(cp))
        return "A" if c[0] in "LM" else "N" if c[0] == "N" else "C" if c[0] in "PS" else "D"

    def group_official(cp):
        x = chr(cp)
        a, b = n("a" + x), n(x + "a")
        if a == 1 and b == 1:
            return "A"
        if a == 2 and b == 1:
            return "D"
        return ("N" if n(x * 4) == 2 else "C") if (a == 2 and b == 2) else "?"
    out: list = []
    for cp in range(0x80, 0x110000):
        if 0xD800 <= cp <= 0xDFFF or 0x4E00 <= cp <= 0x9FA5 or 0x3040 <= cp <= 0x30FF or cp in (0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000) or 0x2000 <= cp <= 0x200A:
            continue
        g = group_official(cp)
        if g != group_py(cp):
            if out and out[-1][2] == g and out[-1][1] + 1 == cp:
                out[-1] = (out[-1][0], cp, g)
            else:
                out.append((cp, cp, g))
    return out


# ---------------------------------------------------------------------------------------------------------------
# byte-level alphabet
# ---------------------------------------------------------------------------------------------------------------


def bytes_to_unicode() -> dict:
    """GPT-2's byte -> printable character table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}


class TokenizerError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# the tokenizer
# ---------------------------------------------------------------------------------------------------------------


class Ds41Tokenizer:
    def __init__(self, tokens, merges, token_types=None, *, pre: str = "joyai-llm", model: str = "gpt2", bos_id=None, eos_id=None, pad_id=None,
                 add_bos: bool = False, add_eos: bool = False, chat_template: str | None = None, engine: str = "auto", strict: bool = False):
        if model != "gpt2":
            raise TokenizerError(f"tokenizer.ggml.model {model!r}: only the byte-level BPE ('gpt2') is implemented")
        if pre not in PRE_TOKENIZERS:
            raise TokenizerError(f"tokenizer.ggml.pre {pre!r}: unknown pre-tokenizer (known: {sorted(PRE_TOKENIZERS)}); refusing to guess a regex set")
        self.tokens = list(tokens)
        self.types = list(token_types) if token_types is not None else [TT_NORMAL] * len(self.tokens)
        if len(self.types) != len(self.tokens):
            raise TokenizerError(f"token_type has {len(self.types)} entries for {len(self.tokens)} tokens")
        self.pre, self.model = pre, model
        self.bos_id, self.eos_id, self.pad_id = bos_id, eos_id, pad_id
        self.add_bos, self.add_eos = bool(add_bos), bool(add_eos)
        self.chat_template = chat_template
        self.vocab: dict = {}
        for i, t in enumerate(self.tokens):
            self.vocab.setdefault(t, i)
        self.ranks: dict = {}
        for r, m in enumerate(merges):
            a, _, b = m.partition(" ")
            if not b or " " in b:
                raise TokenizerError(f"merge #{r} {m!r} is not 'left right'")
            self.ranks[(a, b)] = r
        self.n_merges = len(self.ranks)
        # a merge whose result is not a token can never fire (Hugging Face refuses such a file; llama.cpp ignores it at run time): `strict` raises, otherwise it is
        # dropped and counted (`bad_merges`; selfcheck reports it - the mini test GGUF has placeholder merges)
        self.bad_merges = [(a, b) for (a, b) in self.ranks if (a + b) not in self.vocab]
        if self.bad_merges:
            if strict:
                a, b = self.bad_merges[0]
                raise TokenizerError(f"merge {a!r} + {b!r} = {a + b!r} is not in the vocabulary ({len(self.bad_merges)} such merges)")
            for k in self.bad_merges:
                del self.ranks[k]
        self.patterns = compile_pretokenizer(PRE_TOKENIZERS[pre], engine)
        # tokens matched literally in the text: control and user-defined, plus bos / eos / pad
        self.special_ids = {i for i, ty in enumerate(self.types) if ty in (TT_CONTROL, TT_USER_DEFINED)}
        for x in (bos_id, eos_id, pad_id):
            if x is not None and 0 <= x < len(self.tokens):
                self.special_ids.add(x)
        lit = sorted({self.tokens[i] for i in self.special_ids if self.tokens[i]}, key=lambda s: (-len(s), s))
        self._special_re = re.compile("|".join(re.escape(s) for s in lit)) if lit else None
        self._cache: dict = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_metadata(cls, md: dict, engine: str = "auto", strict: bool = False) -> "Ds41Tokenizer":
        """`md`: GGUF metadata (tools/gguf_reader.GGUFFile.metadata): tokenizer.ggml.* arrays and scalars."""
        for k in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            v = md.get(k)
            if not isinstance(v, (list, tuple)):
                raise TokenizerError(f"{k} is missing or not an array in the metadata (got {type(v).__name__}: a headers-only JSON has summaries, not arrays)")
        return cls(md["tokenizer.ggml.tokens"], md["tokenizer.ggml.merges"], md.get("tokenizer.ggml.token_type"), pre=md.get("tokenizer.ggml.pre", "joyai-llm"),
                   model=md.get("tokenizer.ggml.model", "gpt2"), bos_id=md.get("tokenizer.ggml.bos_token_id"), eos_id=md.get("tokenizer.ggml.eos_token_id"),
                   pad_id=md.get("tokenizer.ggml.padding_token_id"), add_bos=bool(md.get("tokenizer.ggml.add_bos_token", False)),
                   add_eos=bool(md.get("tokenizer.ggml.add_eos_token", False)), chat_template=md.get("tokenizer.chat_template"), engine=engine, strict=strict)

    @classmethod
    def from_gguf(cls, path, engine: str = "auto", strict: bool = False) -> "Ds41Tokenizer":
        """From shard 1 of a (split) GGUF; only the metadata is read."""
        import gguf_reader
        paths = [path] if isinstance(path, (str, pathlib.Path)) else list(path)
        return cls.from_metadata(gguf_reader.GGUFFile(pathlib.Path(paths[0])).metadata, engine, strict)

    @classmethod
    def from_tokenizer_json(cls, path, chat_template: str | None = None, engine: str = "auto", strict: bool = False) -> "Ds41Tokenizer":
        return cls.from_metadata(metadata_from_tokenizer_json(path, chat_template), engine, strict)

    # ------------------------------------------------------------------ queries
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def token_to_id(self, tok: str):
        return self.vocab.get(tok)

    def id_to_token(self, i: int) -> str:
        return self.tokens[i]

    @property
    def bos_token(self):
        return None if self.bos_id is None else self.tokens[self.bos_id]

    @property
    def eos_token(self):
        return None if self.eos_id is None else self.tokens[self.eos_id]

    # ------------------------------------------------------------------ encode
    def _bpe(self, piece: str) -> tuple:
        """ids of one pre-tokenised piece (already in the GPT-2 alphabet)."""
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        chars = list(piece)
        n = len(chars)
        if n > 1:
            ranks = self.ranks
            sym = chars[:]
            alive = [True] * n
            prev = list(range(-1, n - 1))
            nxt = list(range(1, n + 1))
            nxt[-1] = -1
            heap: list = []
            for i in range(n - 1):
                r = ranks.get((sym[i], sym[i + 1]))
                if r is not None:
                    heap.append((r, i, sym[i], sym[i + 1]))
            heapq.heapify(heap)
            while heap:
                r, i, a, b = heapq.heappop(heap)
                j = nxt[i]
                if not alive[i] or sym[i] != a or j == -1 or not alive[j] or sym[j] != b:
                    continue
                sym[i] = a + b
                alive[j] = False
                nxt[i] = nxt[j]
                if nxt[j] != -1:
                    prev[nxt[j]] = i
                if prev[i] != -1:
                    p = prev[i]
                    r2 = ranks.get((sym[p], sym[i]))
                    if r2 is not None:
                        heapq.heappush(heap, (r2, p, sym[p], sym[i]))
                if nxt[i] != -1:
                    r2 = ranks.get((sym[i], sym[nxt[i]]))
                    if r2 is not None:
                        heapq.heappush(heap, (r2, i, sym[i], sym[nxt[i]]))
            chars, i = [], 0
            while i != -1:
                chars.append(sym[i])
                i = nxt[i]
        try:
            ids = tuple(self.vocab[c] for c in chars)
        except KeyError as e:
            raise TokenizerError(f"BPE produced {e.args[0]!r}, which is not in the vocabulary (a damaged vocabulary / merges pair)") from None
        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def _encode_plain(self, text: str, out: list) -> None:
        pieces = [text]
        for pat in self.patterns:
            nxt: list = []
            for p in pieces:
                nxt.extend(split_isolated(pat, p))
            pieces = nxt
        b2c = BYTE_TO_CHAR
        for p in pieces:
            out.extend(self._bpe("".join(b2c[b] for b in p.encode("utf-8", "replace"))))

    def encode(self, text: str, *, parse_special: bool = True, add_bos: bool | None = None, add_eos: bool | None = None) -> list:
        """text -> ids.  `parse_special`: literal control / user-defined token strings in the text become their ids (the Hugging Face default);
        `add_bos` / `add_eos` default to the GGUF's `add_bos_token` / `add_eos_token` (both False for this model: the chat template writes the BOS itself)."""
        out: list = []
        if (self.add_bos if add_bos is None else add_bos) and self.bos_id is not None:
            out.append(self.bos_id)
        if parse_special and self._special_re is not None:
            last = 0
            for m in self._special_re.finditer(text):
                if m.start() > last:
                    self._encode_plain(text[last:m.start()], out)
                out.append(self.vocab[m.group(0)])
                last = m.end()
            if last < len(text):
                self._encode_plain(text[last:], out)
        elif text:
            self._encode_plain(text, out)
        if (self.add_eos if add_eos is None else add_eos) and self.eos_id is not None:
            out.append(self.eos_id)
        return out

    # ------------------------------------------------------------------ decode
    def decode_bytes(self, ids, *, skip_special: bool = False) -> bytes:
        buf = bytearray()
        for i in ids:
            if not 0 <= i < len(self.tokens):
                raise TokenizerError(f"token id {i} outside the vocabulary (0..{len(self.tokens) - 1})")
            tok = self.tokens[i]
            if i in self.special_ids and self.types[i] in (TT_CONTROL, TT_USER_DEFINED) or not all(c in CHAR_TO_BYTE for c in tok):
                if skip_special and self.types[i] in (TT_CONTROL, TT_UNKNOWN):
                    continue
                buf += tok.encode("utf-8")
            else:
                buf += bytes(CHAR_TO_BYTE[c] for c in tok)
        return bytes(buf)

    def decode(self, ids, *, skip_special: bool = False) -> str:
        return self.decode_bytes(ids, skip_special=skip_special).decode("utf-8", "replace")

    # ------------------------------------------------------------------ chat
    def apply_chat_template(self, messages, *, add_generation_prompt: bool = False, thinking: bool = False, tools=None, response_format=None,
                            reasoning_effort=None, drop_thinking: bool = True, engine: str = "auto") -> str:
        """The GGUF's chat template rendered to text.  engine "native": the Python transcription below (needs the template to be the known DeepSeek-V4 one,
        checked by hash); "jinja": jinja2 on the template string in the GGUF (any template; needs jinja2); "auto": native when the hash matches, else jinja."""
        if self.chat_template is None and engine != "native":
            raise TokenizerError("the GGUF has no tokenizer.chat_template")
        known = self.chat_template is None or template_hash(self.chat_template) in KNOWN_TEMPLATE_HASHES
        if engine == "auto":
            engine = "native" if known else "jinja"
        if engine == "native":
            if not known:
                raise TokenizerError("the GGUF's chat template is not the DeepSeek-V4 one the native renderer transcribes; use engine='jinja'")
            return render_deepseek_v4(messages, bos_token=self.bos_token or "", add_generation_prompt=add_generation_prompt, thinking=thinking, tools=tools,
                                      response_format=response_format, reasoning_effort=reasoning_effort, drop_thinking=drop_thinking)
        return render_jinja(self.chat_template, messages, bos_token=self.bos_token or "", add_generation_prompt=add_generation_prompt, thinking=thinking,
                            tools=tools, response_format=response_format, reasoning_effort=reasoning_effort, drop_thinking=drop_thinking)

    def encode_chat(self, messages, **kw) -> list:
        return self.encode(self.apply_chat_template(messages, **kw), parse_special=True, add_bos=False)


# ---------------------------------------------------------------------------------------------------------------
# a tokenizer.json (Hugging Face) as GGUF-style metadata
# ---------------------------------------------------------------------------------------------------------------


def metadata_from_tokenizer_json(path, chat_template: str | None = None) -> dict:
    """What a converter writes to the GGUF for this tokenizer.json (the ids, strings, merges and token types: control for `special` added tokens, user-defined
    for the others, normal for the rest).  Used to check a GGUF against the official file and to run the tests without a 400 GB GGUF."""
    t = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    vocab = t["model"]["vocab"]
    added = {a["id"]: a for a in t.get("added_tokens", [])}
    n = max(max(vocab.values()), max(added, default=-1)) + 1
    tokens: list = [None] * n
    types = [TT_NORMAL] * n
    for tok, i in vocab.items():
        tokens[i] = tok
    for i, a in added.items():
        tokens[i] = a["content"]
        types[i] = TT_CONTROL if a.get("special") else TT_USER_DEFINED
    if any(x is None for x in tokens):
        raise TokenizerError("tokenizer.json has holes in its id space")
    merges = [m if isinstance(m, str) else " ".join(m) for m in t["model"]["merges"]]
    md = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "joyai-llm", "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": types,
          "tokenizer.ggml.merges": merges, "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.add_eos_token": False}
    for name, key in (("<｜begin▁of▁sentence｜>", "bos_token_id"), ("<｜end▁of▁sentence｜>", "eos_token_id")):
        if name in vocab:
            md["tokenizer.ggml." + key] = vocab[name]
    if chat_template is not None:
        md["tokenizer.chat_template"] = chat_template
    return md


def check_against_tokenizer_json(tok: Ds41Tokenizer, path) -> list:
    """Differences between a tokenizer built from a GGUF and the official tokenizer.json (token strings, types, merges, special ids, pre-tokenizer
    regexes): [] when the GGUF carries exactly the official tokenizer."""
    md = metadata_from_tokenizer_json(path)
    t = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    probs = []
    if tok.vocab_size != len(md["tokenizer.ggml.tokens"]):
        probs.append(f"vocabulary size {tok.vocab_size} vs official {len(md['tokenizer.ggml.tokens'])}")
    bad = [i for i, (a, b) in enumerate(zip(tok.tokens, md["tokenizer.ggml.tokens"])) if a != b]
    if bad:
        probs.append(f"{len(bad)} token strings differ, first at id {bad[0]}: {tok.tokens[bad[0]]!r} vs official {md['tokenizer.ggml.tokens'][bad[0]]!r}")
    ob = [i for i, (a, b) in enumerate(zip(tok.types, md["tokenizer.ggml.token_type"])) if a != b]
    if ob:
        probs.append(f"{len(ob)} token types differ, first at id {ob[0]}: {tok.types[ob[0]]} vs official {md['tokenizer.ggml.token_type'][ob[0]]} ({tok.tokens[ob[0]]!r})")
    off = md["tokenizer.ggml.merges"]
    mine = sorted(tok.ranks, key=tok.ranks.get)
    if len(mine) != len(off) or any(f"{a} {b}" != m for (a, b), m in zip(mine, off)):
        probs.append(f"merges differ ({len(mine)} vs official {len(off)})")
    off_re = [p["pattern"]["Regex"].replace("\r", "\\r").replace("\n", "\\n") for p in t["pre_tokenizer"]["pretokenizers"] if p.get("type") == "Split"]
    if off_re != list(PRE_TOKENIZERS[tok.pre]):
        probs.append(f"pre-tokenizer regexes differ from the official file's: {off_re} vs {list(PRE_TOKENIZERS[tok.pre])}")
    for name, key in (("<｜begin▁of▁sentence｜>", "bos_id"), ("<｜end▁of▁sentence｜>", "eos_id")):
        if getattr(tok, key) != md.get("tokenizer.ggml." + key.replace("_id", "_token_id")):
            probs.append(f"{key} {getattr(tok, key)} vs official {md.get('tokenizer.ggml.' + key.replace('_id', '_token_id'))}")
    return probs


# ---------------------------------------------------------------------------------------------------------------
# the chat template
# ---------------------------------------------------------------------------------------------------------------

DSML = "｜DSML｜"
THINK_START, THINK_END = "<think>", "</think>"
USER_TOKEN, ASSISTANT_TOKEN, EOS_TEXT = "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"
REASONING_EFFORT_MAX = ("Reasoning Effort: Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your thinking and comprehensively decompose "
                        "the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
                        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to "
                        "ensure absolutely no assumption is left unchecked.\n\n")
RESPONSE_FORMAT_TEMPLATE = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
TOOLS_HEADER = ("## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + DSML + "tool_calls>\" block like the following:\n\n"
                "<" + DSML + "tool_calls>\n<" + DSML + "invoke name=\"$TOOL_NAME\">\n<" + DSML + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + DSML
                + "parameter>\n...\n</" + DSML + "invoke>\n<" + DSML + "invoke name=\"$TOOL_NAME2\">\n...\n</" + DSML + "invoke>\n</" + DSML + "tool_calls>\n\n"
                "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format "
                "and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by " + THINK_START + "), you MUST output your complete reasoning inside " + THINK_START
                + "..." + THINK_END + " BEFORE any tool calls or final response.\n\nOtherwise, output directly after " + THINK_END + " with tool calls or final response.\n\n"
                "### Available Tool Schemas\n\n")
TOOLS_FOOTER = "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"


def tojson(x) -> str:
    """The `tojson` filter of Hugging Face chat templates: json.dumps without ASCII escaping or key sorting."""
    return json.dumps(x, ensure_ascii=False)


def render_deepseek_v4(messages, *, bos_token: str = "<｜begin▁of▁sentence｜>", add_generation_prompt: bool = False, thinking: bool = False, tools=None,
                       response_format=None, reasoning_effort=None, drop_thinking: bool = True) -> str:
    """The DeepSeek-V4 chat template (tokenizer.chat_template of the GGUF), transcribed.  `messages`: [{"role": system|user|developer|tool|assistant,
    "content": str, "reasoning_content": str, "tool_calls": [{"function": {"name", "arguments": dict | JSON string}}]}]."""
    messages = list(messages or [])
    system_prompt, first_sp, has_tool_calls = "", True, False
    for m in messages:
        if m["role"] == "system":
            if first_sp:
                system_prompt += (m.get("content") or "")
                first_sp = False
            else:
                system_prompt += "\n\n" + (m.get("content") or "")
    has_tools = False
    if tools:
        has_tools = True
        schemas = ""
        for tool in tools:
            if tool.get("type") == "function":
                schemas += tojson(tool["function"]) + "\n"
        system_prompt = (system_prompt + "\n\n" + TOOLS_HEADER + schemas + TOOLS_FOOTER) if system_prompt else (TOOLS_HEADER + schemas + TOOLS_FOOTER)
    if response_format is not None:
        if system_prompt:
            system_prompt += "\n\n"
        system_prompt += RESPONSE_FORMAT_TEMPLATE + tojson(response_format)
    out = [bos_token]
    if messages and thinking and reasoning_effort is not None and reasoning_effort == "max":
        out.append(REASONING_EFFORT_MAX)
    out.append(system_prompt)
    last_user_idx = -1
    for i, m in enumerate(messages):
        if m["role"] in ("user", "developer", "tool"):
            last_user_idx = i
    in_user = False
    for m in messages:
        if m["role"] == "tool":
            has_tool_calls = True
    for i, m in enumerate(messages):
        role = m["role"]
        if role in ("user", "developer"):
            if in_user:
                out.append("\n\n")
            else:
                out.append(USER_TOKEN)
                in_user = True
            out.append(m.get("content") or "")
        elif role == "tool":
            if in_user:
                out.append("\n\n")
            else:
                out.append(USER_TOKEN)
                in_user = True
            out.append("<tool_result>" + (m.get("content") or "") + "</tool_result>")
        elif role == "assistant":
            in_user = False
            out.append(ASSISTANT_TOKEN)
            is_after_last_user = i > last_user_idx
            keep_reasoning = thinking and ((not drop_thinking) or has_tools or is_after_last_user or has_tool_calls)
            if keep_reasoning:
                out.append(THINK_START)
                if m.get("reasoning_content"):
                    out.append(m["reasoning_content"])
                out.append(THINK_END)
            else:
                out.append(THINK_END)
            if m.get("content"):
                out.append(m["content"])
            if m.get("tool_calls"):
                out.append("\n\n<" + DSML + "tool_calls>\n")
                for tool in m["tool_calls"]:
                    func = tool["function"]
                    out.append("<" + DSML + "invoke name=\"" + func["name"] + "\">\n")
                    args = func["arguments"]
                    if isinstance(args, str):
                        args = json.loads(args)
                    for key, val in args.items():
                        if isinstance(val, str):
                            out.append("<" + DSML + "parameter name=\"" + key + "\" string=\"true\">" + val + "</" + DSML + "parameter>\n")
                        else:
                            out.append("<" + DSML + "parameter name=\"" + key + "\" string=\"false\">" + tojson(val) + "</" + DSML + "parameter>\n")
                    if not args:
                        out.append("\n")
                    out.append("</" + DSML + "invoke>\n")
                out.append("</" + DSML + "tool_calls>")
            out.append(EOS_TEXT)
    if add_generation_prompt:
        out.append(ASSISTANT_TOKEN)
        out.append(THINK_START if thinking else THINK_END)
    return "".join(out)


def template_hash(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


# sha256 of the chat template of the mxxm-t GGUF (third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz); filled in below
KNOWN_TEMPLATE_HASHES: set = set()


def render_jinja(template: str, messages, *, bos_token: str, add_generation_prompt: bool = False, thinking: bool = False, tools=None, response_format=None,
                 reasoning_effort=None, drop_thinking: bool = True, **extra) -> str:
    """The template string rendered with jinja2 the way Hugging Face does (sandbox, `tojson` without escaping, plus `from_json`, `raise_exception`)."""
    try:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError as e:
        raise TokenizerError("jinja2 is not installed: use engine='native' for the DeepSeek-V4 template") from e

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env.filters["from_json"] = lambda s: json.loads(s)
    env.globals["raise_exception"] = raise_exception
    ctx = {"messages": list(messages), "bos_token": bos_token, "add_generation_prompt": add_generation_prompt, "thinking": thinking, "drop_thinking": drop_thinking}
    if tools:
        ctx["tools"] = tools
    if response_format is not None:
        ctx["response_format"] = response_format
    if reasoning_effort is not None:
        ctx["reasoning_effort"] = reasoning_effort
    ctx.update(extra)
    return env.from_string(template).render(**ctx)


def vendored_chat_template() -> str | None:
    """The chat template string of the real GGUF, from the vendored header dump (None if the dump is not there)."""
    import gzip
    p = REPO / "third_party" / "deepseek-v41-flash-reference" / "gguf-headers-mxxm-t-MXFP4.json.gz"
    if not p.is_file():
        return None
    hdr = json.load(gzip.open(p, "rt", encoding="utf-8"))
    first = sorted(k for k in hdr if "00001-of" in k)[0]
    return hdr[first]["kv"].get("tokenizer.chat_template")


_vt = vendored_chat_template()
if _vt is not None:
    KNOWN_TEMPLATE_HASHES.add(template_hash(_vt))
del _vt


# ---------------------------------------------------------------------------------------------------------------
# checks and command line
# ---------------------------------------------------------------------------------------------------------------

# known ids of the real tokenizer (tokenizer.json of deepseek-ai/DeepSeek-V4.1-Flash; bos / eos / pad of the GGUF header dump)
KNOWN_IDS = {"<｜begin▁of▁sentence｜>": 0, "<｜end▁of▁sentence｜>": 1, "<｜▁pad▁｜>": 2, "<｜System｜>": 128799, "<｜User｜>": 128803, "<｜Assistant｜>": 128804,
             "<|EOT|>": 128805, "<think>": 128821, "</think>": 128822, "｜DSML｜": 128825, "<｜fim▁begin｜>": 128801, "<｜tool▁calls▁begin｜>": 128806}

SELFCHECK_TEXTS = [
    "", " ", "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading and trailing  ", "tabs\tand\nnewlines\r\n\r\n\nand   spaces",
    "1234567890 12 345 6789 0.5e-10 3.14159", "def f(x):\n    return x**2  # square\n\n\nclass A:\n\tpass\n", "if (a<b && c>=d) { x[i] = y->z; }",
    "你好，世界！今天天气很好。日本語のテキスト、ひらがなカタカナ。한국어 문장입니다.", "Привет, мир! Ελληνικά, עברית, العربية, हिन्दी", "emoji 😀👍🏽 family 👨‍👩‍👧‍👦 flags 🇯🇵",
    "naïve café Zürich — “quotes” ‘single’ … ©®™ €£¥", "https://example.com/a/b?c=d&e=f#g  user@host.org", "snake_case camelCase PascalCase kebab-case SCREAMING_CASE",
    "<｜begin▁of▁sentence｜>hi<｜User｜>question<｜Assistant｜><think>", "</think>answer<｜end▁of▁sentence｜>", "a<｜User｜><｜User｜>b", "<think><think></think>",
    "\x00\x01\x02 control \x7f chars \x85 \xa0 nbsp \u2003 em-space \u3000 ideographic", "e\u0301 combining é, \u0928\u093f", "\u200b zero width \ufeff bom",
    "a" * 300, " " * 100 + "x", "\n" * 50, "ab" * 500, "0123456789" * 40, "  \n  \n  ", "x\n\n\ny\n \nz", "$$ ## !! ?? ,, .. -- __ ~~ ``", "'s 're 'll 'd don't can't",
]


def _rand_texts(n: int = 300, seed: int = 0) -> list:
    import random
    rng = random.Random(seed)
    alpha = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n\r.,;:!?'\"()[]{}<>/\\|@#$%^&*-_=+`~"
             "éüñßøåçÅÉ你好世界日本語ひらがなカタカナ한국어Привет😀👍\u0301\u200b\u3000\u00a0\x00")
    out = []
    for _ in range(n):
        out.append("".join(rng.choice(alpha) for _ in range(rng.randint(0, 60))))
    return out


def selfcheck(tok: Ds41Tokenizer, tokenizer_json=None, verbose: bool = True) -> tuple:
    """-> (n_failed, n_skipped).  Round trips, the known ids, and (given a tokenizer.json) the GGUF-vs-official metadata comparison and token-for-token
    equality with the Hugging Face `tokenizers` library when it is installed."""
    failed = skipped = 0

    def rec(name, ok, detail=""):
        nonlocal failed
        if not ok:
            failed += 1
        if verbose:
            print(("PASS  " if ok else "FAIL  ") + name + (("  " + detail) if detail else ""))

    def skip(name, why):
        nonlocal skipped
        skipped += 1
        if verbose:
            print(f"SKIP  {name}  ({why})")

    texts = SELFCHECK_TEXTS + _rand_texts()
    bad = [t for t in texts if tok.decode(tok.encode(t, parse_special=False)) != t.encode("utf-8", "replace").decode("utf-8", "replace")]
    rec("every merge result is a token", not tok.bad_merges, f"{tok.n_merges} merges" + (f"; {len(tok.bad_merges)} dropped, first {tok.bad_merges[0]}" if tok.bad_merges else ""))
    rec("decode(encode(text)) == text (no special parsing)", not bad, f"{len(texts)} texts" + (f"; first failure {bad[0]!r}" if bad else ""))
    known = {k: v for k, v in KNOWN_IDS.items() if k in tok.vocab}
    if known and tok.vocab_size > 128000:
        wrong = {k: (tok.vocab.get(k), v) for k, v in known.items() if tok.vocab.get(k) != v}
        rec("known token ids", not wrong, f"{len(known)} checked" + (f"; wrong {wrong}" if wrong else ""))
    else:
        skip("known token ids", "not the DeepSeek-V4.1-Flash vocabulary")
    if tok.bos_id is not None:
        rec("bos / eos ids are tokens", tok.bos_token is not None and tok.eos_token is not None, f"bos {tok.bos_token!r} eos {tok.eos_token!r}")
    sp = [t for t in tok.special_ids if tok.tokens[t]]
    ok = all(tok.encode(tok.tokens[i]) == [i] for i in sp)
    rec("every special token string encodes to its id", ok, f"{len(sp)} tokens")
    rec("special tokens are not parsed with parse_special=False", all(tok.encode(tok.tokens[i], parse_special=False) != [i] for i in sorted(sp)[:20]))
    if tokenizer_json is None:
        skip("official tokenizer.json comparison", "no tokenizer.json (set DS41_TOKENIZER_JSON or pass --tokenizer-json)")
        return failed, skipped
    probs = check_against_tokenizer_json(tok, tokenizer_json)
    rec("GGUF tokenizer == official tokenizer.json (tokens, types, merges, regexes, ids)", not probs, "; ".join(probs))
    try:
        from tokenizers import Tokenizer
    except ImportError:
        skip("token-for-token vs the `tokenizers` library", "package not installed")
        return failed, skipped
    ref = Tokenizer.from_file(str(tokenizer_json))
    diff = [t for t in texts if ref.encode(t, add_special_tokens=False).ids != tok.encode(t, parse_special=True, add_bos=False, add_eos=False)]
    rec("encode == `tokenizers` encode", not diff, f"{len(texts)} texts" + (f"; first difference {diff[0]!r}: {tok.encode(diff[0])[:20]} vs {ref.encode(diff[0], add_special_tokens=False).ids[:20]}" if diff else ""))
    ids_list = [tok.encode(t) for t in texts]
    dd = [i for i, ids in enumerate(ids_list) if ref.decode(ids, skip_special_tokens=False) != tok.decode(ids)]
    rec("decode == `tokenizers` decode", not dd, f"{len(ids_list)} sequences" + (f"; first difference {texts[dd[0]]!r}" if dd else ""))
    return failed, skipped


def _load(a, engine="auto") -> Ds41Tokenizer:
    if getattr(a, "tokenizer_json", None) and not getattr(a, "gguf", None):
        return Ds41Tokenizer.from_tokenizer_json(a.tokenizer_json, vendored_chat_template(), engine)
    if not getattr(a, "gguf", None):
        raise SystemExit("give --gguf SHARD1 (or --tokenizer-json FILE)")
    return Ds41Tokenizer.from_gguf(a.gguf, engine)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def src(p):
        p.add_argument("--gguf", help="shard 1 of the GGUF (the tokenizer arrays live in its metadata)")
        p.add_argument("--tokenizer-json", help="a Hugging Face tokenizer.json instead of a GGUF")
    p = sub.add_parser("encode"); src(p)
    p.add_argument("text", nargs="?"); p.add_argument("--file"); p.add_argument("--no-special", action="store_true"); p.add_argument("--bos", action="store_true")
    p = sub.add_parser("decode"); src(p)
    p.add_argument("--ids", required=True); p.add_argument("--skip-special", action="store_true")
    p = sub.add_parser("chat"); src(p)
    p.add_argument("--messages", required=True, help="JSON list (or @file)"); p.add_argument("--thinking", action="store_true")
    p.add_argument("--no-generation-prompt", action="store_true"); p.add_argument("--ids", action="store_true"); p.add_argument("--engine", default="auto")
    p = sub.add_parser("info"); src(p)
    p = sub.add_parser("selfcheck"); src(p)
    a = ap.parse_args(argv)
    try:
        tok = _load(a)
        if a.cmd == "encode":
            text = a.text if a.text is not None else (pathlib.Path(a.file).read_text(encoding="utf-8") if a.file else sys.stdin.read())
            print(json.dumps(tok.encode(text, parse_special=not a.no_special, add_bos=True if a.bos else None)))
        elif a.cmd == "decode":
            sys.stdout.write(tok.decode([int(x) for x in a.ids.split(",") if x.strip()], skip_special=a.skip_special))
            print()
        elif a.cmd == "chat":
            raw = a.messages
            msgs = json.loads(pathlib.Path(raw[1:]).read_text() if raw.startswith("@") else raw)
            text = tok.apply_chat_template(msgs, add_generation_prompt=not a.no_generation_prompt, thinking=a.thinking, engine=a.engine)
            print(json.dumps(tok.encode(text, add_bos=False)) if a.ids else text)
        elif a.cmd == "info":
            print(json.dumps({"model": tok.model, "pre": tok.pre, "vocab_size": tok.vocab_size, "merges": tok.n_merges, "bos": tok.bos_id, "eos": tok.eos_id, "pad": tok.pad_id,
                              "add_bos": tok.add_bos, "add_eos": tok.add_eos, "bad_merges": len(tok.bad_merges), "special_tokens": len(tok.special_ids), "has_chat_template": tok.chat_template is not None}, indent=1))
        elif a.cmd == "selfcheck":
            tj = a.tokenizer_json
            if tj is None:
                try:
                    from ref.ds41.selfcheck import tokenizer_json_path
                    tj = tokenizer_json_path(download=False)
                except Exception:
                    tj = None
            failed, skipped = selfcheck(tok, tj)
            print(f"{'FAILED' if failed else 'ok'}: {failed} failed, {skipped} skipped")
            return 1 if failed else 0
    except (TokenizerError, OSError, ValueError, KeyError) as e:
        print(f"ds1_tokenizer: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
