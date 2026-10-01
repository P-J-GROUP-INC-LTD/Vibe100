#!/usr/bin/env python3
"""tools/volta/prompts/make_long_prompt.py - the LONG prompt of the Phase-2 harness (tools/volta/golden_compare.py): this
repository's own sources, concatenated, cut to a fixed number of tokens.

WHY A LONG PROMPT AT ALL.  Strata's QSA attention layers select at most `idx_top_k + idx_block - 1 = 2048 + 4 - 1 = 2,051`
cells per query (`qsa_selection_width`, include/strata/kernels/qsa.hpp).  At or below 2,051 tokens of context the selection
is the identity - every query attends to every earlier token - and the indexer's block scores and top-k never decide
anything.  Above it they do, and the prompt-attention kernel (the one the port replaces on Volta) runs on 2,051 SCATTERED
cells per query instead of a contiguous prefix.  The dense prefill GEMMs are chunked too (`--prefill auto`: up to 8,192
tokens per chunk), and the FP16 route of the port scales each call.  A correctness check on a 500-token prompt therefore
checks neither; the default here is 33,000 tokens (16x the selection width, four-plus chunks of 8,192).

WHY SOURCE CODE: it is available on every machine that has this repository, it is long, it is diverse (C++, CUDA, Python,
Markdown), and it has long-range structure (names defined early and used much later), which is what makes attention over a
long context matter for the next-token prediction - a lorem-ipsum prompt would pass any check.

    python3 tools/volta/prompts/make_long_prompt.py --tokenizer pack/full/tokenizer --tokens 33000 --out long.ids
    python3 tools/volta/prompts/make_long_prompt.py --text-out long.txt --chars-per-token 3.3 --tokens 33000   # text only, no tokenizer

The ids file is what `strata ... --tokens-file` and `golden_compare.py --ids-file` read (whitespace-separated).  The
tokenizer is tools/strata_tokenizer.py with the directory setup.py extracts (`<pack>/tokenizer/`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[3]
SUFFIXES = {".cpp", ".cu", ".cuh", ".hpp", ".h", ".py", ".md", ".cmake", ".txt", ".sh"}
SKIP_PARTS = {"third_party", "build", ".git", ".venv", "__pycache__", "volta", "pack", "golden_out", "engine", "data", "web", "hip", "vision"}
MAX_FILE = 400_000     # bytes: skip generated tables and fixtures


def collect_sources(root: Path = ROOT) -> list[Path]:
    """Source files in a fixed order (sorted paths, src/ and include/ first), so the prompt is the same on every machine
    that has the same checkout."""
    files = []
    for top in ("src", "include", "tools", "serve", "docs", "CMakeLists.txt", "setup.py"):
        p = root / top
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                rel = f.relative_to(root)
                if (f.is_file() and f.suffix in SUFFIXES and not (set(rel.parts) & SKIP_PARTS)
                        and f.stat().st_size <= MAX_FILE and f.stat().st_size > 0):
                    files.append(f)
    return files


def source_text(f: Path, root: Path = ROOT) -> str:
    try:
        body = f.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    return f"// ===== {f.relative_to(root)} =====\n{body.rstrip()}\n\n"


def build_ids(files: Iterable[Path], encode: Callable[[str], list[int]], target_tokens: int, root: Path = ROOT) -> list[int]:
    """Tokenize the files in order until there are `target_tokens`; returns exactly that many ids (the last file is cut)."""
    ids: list[int] = []
    for f in files:
        text = source_text(f, root)
        if text:
            ids.extend(encode(text))
        if len(ids) >= target_tokens:
            return ids[:target_tokens]
    raise SystemExit(f"only {len(ids)} tokens of source were found, wanted {target_tokens}")


def build_text(files: Iterable[Path], chars: int, root: Path = ROOT) -> str:
    out, n = [], 0
    for f in files:
        t = source_text(f, root)
        out.append(t)
        n += len(t)
        if n >= chars:
            return "".join(out)[:chars]
    raise SystemExit(f"only {n} characters of source were found, wanted {chars}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=int, default=33000, help="length of the prompt in tokens (default 33000; must exceed 2051)")
    ap.add_argument("--tokenizer", help="the pack's tokenizer/ directory (needed for --out)")
    ap.add_argument("--out", help="write the token ids here (whitespace-separated)")
    ap.add_argument("--text-out", help="write the text here (with --tokenizer: the text of the cut; without: an estimate)")
    ap.add_argument("--chars-per-token", type=float, default=3.3, help="source code is ~3.3 characters per token (--text-out without a tokenizer)")
    ap.add_argument("--root", default=str(ROOT), help="the repository to take sources from")
    a = ap.parse_args(argv)
    if a.tokens <= 2051:
        print(f"make_long_prompt: --tokens {a.tokens} does not exceed the QSA selection width (2,051): the sparse selection "
              "would never engage", file=sys.stderr)
        return 2
    root = Path(a.root)
    files = collect_sources(root)
    if a.out:
        if not a.tokenizer:
            print("make_long_prompt: --out needs --tokenizer", file=sys.stderr)
            return 2
        sys.path.insert(0, str(ROOT / "tools"))
        import json
        import strata_tokenizer as ST
        d = Path(a.tokenizer)
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        toks = [None] * len(vocab)
        for t, i in vocab.items():
            toks[i] = t
        tok = ST.Tokenizer(toks, (d / "merges.txt").read_text(encoding="utf-8").split("\n"),
                           json.loads((d / "token_type.json").read_text()))
        ids = build_ids(files, tok.encode, a.tokens, root)
        Path(a.out).write_text(" ".join(map(str, ids)), encoding="utf-8")
        print(f"make_long_prompt: {len(ids)} tokens from {len(files)} candidate files -> {a.out}")
        if a.text_out:
            Path(a.text_out).write_text(tok.decode(ids), encoding="utf-8")
            print(f"make_long_prompt: text -> {a.text_out}")
    elif a.text_out:
        Path(a.text_out).write_text(build_text(files, int(a.tokens * a.chars_per_token), root), encoding="utf-8")
        print(f"make_long_prompt: ~{a.tokens} tokens of text -> {a.text_out}")
    else:
        print("make_long_prompt: nothing to do (give --out and --tokenizer, or --text-out)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
