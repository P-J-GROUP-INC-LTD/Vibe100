#!/usr/bin/env python3
"""strata_ds41.py - DS1-E: the text front end of `strata-ds41` (the engine takes token ids; this wrapper tokenises with DS1-F's DeepSeek tokenizer, tools/ds41/ds1_tokenizer.py, runs the
engine and decodes what it generated).

    python3 src/ds41/program/strata_ds41.py --gguf SHARD1 --text "The capital of France is" --max-new 16 [--engine build-sm70/strata-ds41] [engine options...]
    python3 src/ds41/program/strata_ds41.py --gguf SHARD1 --chat '[{"role":"user","content":"Hi"}]' [--thinking] --max-new 64 --stats
    python3 src/ds41/program/strata_ds41.py --gguf SHARD1 --text-file prompt.txt ...

Everything after the options below is passed to the engine unchanged (--n-slots, --kv-quant, --trace, --stats, ...).  The engine is found with --engine, the environment variable
STRATA_DS41, or build-sm70/strata-ds41 / build/strata-ds41 under the repository.  Exit status: the engine's; 2 for a usage or tokeniser problem.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "tools" / "ds41")):
    if p not in sys.path:
        sys.path.insert(0, p)


def find_engine(given: str | None) -> str:
    cands = [given] if given else [os.environ.get("STRATA_DS41"), str(REPO / "build-sm70" / "strata-ds41"), str(REPO / "build" / "strata-ds41")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("strata_ds41: the engine (strata-ds41) was not found; pass --engine PATH or set STRATA_DS41")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="shard 1 of the GGUF (the engine opens the whole set; the tokenizer arrays are in its metadata)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="the prompt text")
    g.add_argument("--text-file", help="a file holding the prompt text")
    g.add_argument("--chat", help="a JSON list of {role, content} messages (or @file): rendered with the model's chat template")
    ap.add_argument("--thinking", action="store_true", help="chat template: thinking mode")
    ap.add_argument("--bos", action="store_true", help="prepend the BOS token to --text / --text-file")
    ap.add_argument("--engine", help="the strata-ds41 executable")
    ap.add_argument("--print-ids", action="store_true", help="print the prompt's token ids and exit (do not run the engine)")
    a, rest = ap.parse_known_args()
    try:
        import ds1_tokenizer as T
        tok = T.Ds41Tokenizer.from_gguf(a.gguf)
        if a.chat is not None:
            raw = a.chat
            msgs = json.loads(pathlib.Path(raw[1:]).read_text() if raw.startswith("@") else raw)
            ids = tok.encode(tok.apply_chat_template(msgs, add_generation_prompt=True, thinking=a.thinking), add_bos=False)
        else:
            text = a.text if a.text is not None else pathlib.Path(a.text_file).read_text(encoding="utf-8")
            ids = tok.encode(text, add_bos=True if a.bos else None)
    except (T.TokenizerError, OSError, ValueError, KeyError, ImportError) as e:       # type: ignore[name-defined]
        print(f"strata_ds41: tokeniser: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if not ids:
        print("strata_ds41: the prompt is empty", file=sys.stderr)
        return 2
    print(f"prompt: {len(ids)} tokens", file=sys.stderr)
    if a.print_ids:
        print(",".join(str(i) for i in ids))
        return 0
    cmd = [find_engine(a.engine), "--gguf", a.gguf, "--tokens", ",".join(str(i) for i in ids), *rest]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    sys.stdout.write(p.stdout)
    for line in p.stdout.splitlines():
        if line.startswith("generated ("):
            out = [int(t) for t in line.split(":", 1)[1].split()]
            print("\n---- text ----\n" + tok.decode(out))
    return p.returncode


if __name__ == "__main__":
    sys.exit(main())
