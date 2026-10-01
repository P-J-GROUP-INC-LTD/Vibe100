#!/usr/bin/env python3
"""tools/volta/tokenize_prompt.py - a prompt (text or one of tools/volta/prompts/) to the token-id file `strata --tokens-file` reads.

    python3 tools/volta/tokenize_prompt.py --tokenizer pack/full/tokenizer --prompt-name chat --out chat.ids
    python3 tools/volta/tokenize_prompt.py --tokenizer pack/full/tokenizer --prompt-file notes.txt --chat --out notes.ids
    python3 tools/volta/tokenize_prompt.py --ids-file big.ids --first 64 --out short.ids       # the first 64 ids of a file

The tokenizer is tools/strata_tokenizer.py over the directory setup.py extracts into the pack (`<pack>/tokenizer/`: vocab.json,
merges.txt, token_type.json).  `chat` and `code` prompts are wrapped in the model's chat template (as tools/calibrate.py does).
Used by profile_decode.sh; golden_compare.py does the same internally.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import golden_compare as G  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", help="the pack's tokenizer/ directory")
    ap.add_argument("--prompt-name", choices=["chat", "code"])
    ap.add_argument("--prompt-file")
    ap.add_argument("--prompt")
    ap.add_argument("--ids-file", help="start from an existing id file (with --first, to shorten it)")
    ap.add_argument("--chat", action="store_true", help="wrap text in the chat template")
    ap.add_argument("--no-chat", action="store_true")
    ap.add_argument("--first", type=int, default=0, help="keep only the first N ids")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.max_prompt_tokens = 0
    try:
        ids = G.get_tokens(a, a.tokenizer, Path("."))
    except G.HarnessError as e:
        print(f"tokenize_prompt: {e}", file=sys.stderr)
        return 2
    if a.first:
        ids = ids[: a.first]
    Path(a.out).write_text(" ".join(map(str, ids)) + "\n", encoding="utf-8")
    print(f"tokenize_prompt: {len(ids)} tokens -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
