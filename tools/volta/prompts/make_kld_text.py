#!/usr/bin/env python3
"""tools/volta/prompts/make_kld_text.py - the text of tools/volta/prompts/kld_text.txt, the default text of the llama.cpp comparison.

`tools/volta/logit_identity.sh` runs `llama-perplexity -f kld_text.txt -c 4096` and then scores the Strata engine on the same chunks
(docs/volta/LOGIT_IDENTITY.md).  The text has to be varied - one kind of text exercises one kind of distribution - and it has to be the SAME
bytes every time, because the llama.cpp reference file is tied to it.  So the result is committed (kld_text.txt) and this script only
documents where it came from and can rebuild it:

    segment                                            kind                   about
    Alice's Adventures in Wonderland (Carroll, 1865)   public-domain prose    17 KB   Project Gutenberg #11
    src/kernels/cuda/qsa_select.cu                      CUDA / C++ code        13 KB   upstream Strata 0.1.31 (3906943)
    A Tale of Two Cities (Dickens, 1859)                public-domain prose    17 KB   Project Gutenberg #98
    docs/DETAILS.md                                     technical prose        14 KB   upstream Strata 0.1.31 (3906943)
    The Adventures of Sherlock Holmes (Doyle, 1892)     public-domain prose    17 KB   Project Gutenberg #1661
    tools/iq_pack.py                                    Python code            13 KB   upstream Strata 0.1.31 (3906943)
    Moby Dick (Melville, 1851)                          public-domain prose    17 KB   Project Gutenberg #2701
    src/kernels/cpu/expert_layout.cpp                   C++ code               10 KB   upstream Strata 0.1.31 (3906943)

About 118 KB: roughly 29,000 tokens of a Qwen-style tokenizer, i.e. seven 4,096-token chunks; the first four already hold every kind.  The prose is
the Project Gutenberg plain-text edition with its licence header and footer removed, hard line wraps joined into paragraphs and typographic
quotes / dashes made ASCII; the books are in the public domain (their authors died more than 70 years ago).  Code and the technical page come from
this repository's upstream snapshot, `git show 3906943:<path>` (the first commit of the Vibe100 history), so rebuilding does not depend on
the files as they are today.  Well-known novels are memorised by a language model, which makes its distribution sharper than on new text: that is
why they are mixed with code and a page the model cannot have seen.

    python3 tools/volta/prompts/make_kld_text.py --out /tmp/kld_text.txt && cmp /tmp/kld_text.txt tools/volta/prompts/kld_text.txt
    (needs network for the books, or --cache DIR holding pg11.txt pg98.txt pg1661.txt pg2701.txt; and the repository's git history)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM = "3906943"

# (kind, source, start marker (prose: the first text after it), bytes)
SEGMENTS = [
    ("prose", "pg11", "CHAPTER I.\nDown the Rabbit-Hole", 17000),
    ("code", "git:src/kernels/cuda/qsa_select.cu", None, 13000),
    ("prose", "pg98", "CHAPTER I.\nThe Period", 17000),
    ("docs", "git:docs/DETAILS.md", None, 14000),
    ("prose", "pg1661", "I. A SCANDAL IN BOHEMIA", 17000),
    ("code", "git:tools/iq_pack.py", None, 13000),
    ("prose", "pg2701", "CHAPTER 1. Loomings.", 17000),
    ("code", "git:src/kernels/cpu/expert_layout.cpp", None, 10000),
]

ASCII = {"‘": "'", "’": "'", "“": '"', "”": '"', "—": "--", "–": "-", "…": "...", " ": " ", "﻿": ""}


def fetch(book: str, cache: Path | None) -> str:
    num = book[2:]
    if cache and (cache / f"{book}.txt").exists():
        return (cache / f"{book}.txt").read_text(encoding="utf-8")
    with urllib.request.urlopen(f"https://www.gutenberg.org/cache/epub/{num}/{book}.txt", timeout=60) as r:
        text = r.read().decode("utf-8")
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{book}.txt").write_text(text, encoding="utf-8")
    return text


def prose(text: str, start: str, nbytes: int) -> str:
    text = text.replace("\r\n", "\n")
    body = text[text.index("*** START OF"):]
    body = body[body.index("\n") + 1:]
    # the heading also appears in the table of contents: the chapter itself is the LAST occurrence in the first part of the book
    at = [m.start() for m in re.finditer(re.escape(start), body[:80000])][-1]
    out = body[at:at + nbytes * 2]
    for k, v in ASCII.items():
        out = out.replace(k, v)
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", out)]
    out = "\n\n".join(p for p in paras if p)
    cut = out[:nbytes]
    return cut[:cut.rfind("\n\n")] if "\n\n" in cut else cut      # end on a paragraph


def code(text: str, nbytes: int) -> str:
    cut = text.replace("\r\n", "\n")[:nbytes]
    blank = cut.rfind("\n\n")
    return cut[:blank + 1] if blank > nbytes * 0.7 else cut[:cut.rfind("\n") + 1]     # end between two functions, else on a line


def build(cache: Path | None) -> str:
    parts = []
    for kind, src, start, nbytes in SEGMENTS:
        if src.startswith("pg"):
            parts.append(prose(fetch(src, cache), start, nbytes))
        else:
            raw = subprocess.run(["git", "-C", str(ROOT), "show", f"{UPSTREAM}:{src[4:]}"], check=True, capture_output=True).stdout.decode("utf-8")
            for k, v in ASCII.items():
                raw = raw.replace(k, v)
            parts.append(code(raw, nbytes))
    return "\n\n".join(p.strip("\n") for p in parts) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="a directory with the Project Gutenberg files (pg11.txt, ...), or where to put them")
    a = ap.parse_args(argv)
    text = build(Path(a.cache) if a.cache else None)
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out}: {len(text)} bytes, {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
