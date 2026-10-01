#!/usr/bin/env python3
"""tools/volta/golden_compare.py - the Phase-2 correctness harness: do the Volta fast paths give the same answers?

THE QUESTION.  The port replaces three things on a V100: the prompt attention (a WMMA tensor-core kernel instead of
upstream's Turing/Ampere mma.sync one), the dense prefill GEMMs (FP16 tensor cores instead of cuBLAS bf16, which is
CUDA-core speed on Volta) and nothing else that changes the arithmetic (the decode kernels and ggml's MMQ are the
upstream ones).  None of the replacements is bit-identical to what it replaces - they sum in another order, round
through FP16 - so "same answer" has to mean "same up to the noise the model itself tolerates".  This script measures
that noise against a reference run on THE SAME CARD:

  top-1 agreement        the fraction of positions where candidate and reference pick the same next token
  max |dlogit|           the largest difference of any logit anywhere (and the mean per-position maximum)
  KL(ref || cand)        mean / p99 / max over positions, in nats: how far apart the two next-token distributions are
  perplexity             of each on the sequence's own next tokens, and |dPPL| / PPL: does either model find the
                         text more or less surprising?  (also the paired delta of the per-token NLL with its standard
                         error, which says whether a 0.5 % PPL difference is signal)
  NaN / inf scan         over every logit of both runs

PASS needs top-1 >= 0.99, |dPPL| / PPL <= 0.02 and no NaN/inf (all overridable).  Gate Q of docs/volta/PLAN.md is "top-1 >= 99 %
over 500 tokens, PPL within 1-2 %, no NaN/inf": this is that, with the 500 tokens being `--tail`.  The report's last line says
whether the run establishes Gate Q ("GATE Q: PASS") or only passed a smaller check ("GATE Q: not established - ...": greedy mode,
fewer than 500 scored rows, a prompt too short to exercise the sparse selection).

THE REFERENCE is, by default, a second run of the same engine binary with
    STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0
i.e. the FP32 CUDA-core prompt attention (the decode kernel, what upstream falls back to) and the cuBLAS bf16 GEMMs
(upstream's own, on the CUDA cores of a V100).  Other references: `--ref-env "STRATA_PROMPT_ATTN_OLD=1 ..."` (upstream's
own switch for the same FP32 attention), `--ref-args "..."` (extra engine flags for the reference run only), or
`--ref-logits FILE` - logits produced elsewhere (another GPU running upstream, llama.cpp) in the format below.
NOT references for the Volta prefill kernels: `--no-fast-attn` / `--no-fast-select` (generate.cpp:1831-1834, layer.cpp:931-941) only
switch the token loop's decode attention / selection to the old kernels; the batched prefill and the verify windows never read
them.  (`STRATA_SELECT_OLD=1`, the FP32 warp block scorer, is already the only scorer on a V100: cc < 8.)

TWO MODES

  teacher   One fixed token sequence, no sampling.  The engine runs the first K tokens through the BATCHED prefill path
            (where the Volta kernels live) and the last `--tail` tokens one at a time through the token path, writing
            the logits of every one of those tail positions; the logit at position p predicts token p+1, which IS
            the next token of the text, so perplexity and top-1 are measured on real text.  This is the engine's own
            validation mode (`--prefill-until K`, "so the logits of positions >= K - which depend on the batched
            state - can be scored", src/program/generate.cpp), and it is what bench/results/2026-09-27-esp/esp_kl.py
            does by hand.  K = prompt length - tail unless `--prefill-until` says otherwise; `--prefill-until 0`
            runs the WHOLE prompt through the token path (every position has logits, the batched path is not
            exercised at all - only for short prompts and for checking the token path itself).  (`dump` source; the `logpos`
            source splits the same way with `--turn-token`, see below.)
  greedy    Generate `--max-new` tokens greedily from each run (the prompt goes through the batched path: that is what
            makes the run depend on the Volta prefill kernels).  Reported: the length of the identical prefix, the
            positional token agreement, and the logit metrics on the rows whose contexts are still identical (up to
            and including the first position where the two runs differ).  The top-1 gate is applied to those rows, so
            it asks that the first divergence comes after >= 99 tokens (or never).  Brittle by nature - one near-tie
            flip ends the comparison - hence `teacher` is the real gate; `greedy` shows what a user would see.

  NATIVE (IQ) PACKS HAVE NO LOGITS DUMP, and a Cascade Lake box can only run those.  `--dump-logits` is written only by the
  engine's token loop, and a pack with native_experts.txt (IQ2_XS / IQ3_XXS / IQ3_S / IQ1_M) runs verify windows only: the
  token loop is skipped (generate.cpp: `if (native_pack) { spec_pos = pos; break; }`), so the dump has a header and no rows.
  (The canonical Q2_0 pack does write rows; its CPU kernels need AVX-512 F/BW/VL/DQ + VNNI, which Cascade Lake has - since the
  Vibe100 no-VBMI build the engine no longer refuses it there; on a CPU without VNNI generate.cpp `cpu_require_expert_support`
  still does.)  So teacher mode has two SOURCES, `--teacher-source`:

    dump     the standalone engine and `--dump-logits` / `--prefill-until K`: FULL logits per position -> top-1, max |dlogit|,
             KL, perplexity.  Non-native packs only.
    logpos   the resident engine (`strata --serve`, the protocol serve/server.py speaks: `GEN <max_new> <ids>` ->
             `T <id>` ... `DONE`) with STRATA_LOGPOS=<file>: while it reads a prompt part through the verify windows it
             writes, per position, the log-probability of the prompt's own next token and the top-1 token with its log-probability
             (verify.cpp Verifier::window_logprobs).  The prompt is split where the engine itself splits a conversation: at the
             last occurrence of `--turn-token` (generate.cpp, "the prompt is read in two parts"): everything before it goes
             through the BATCHED path (the Volta kernels), the tail (<= `--short-read` tokens) through the windows with the
             log-probs written.  The harness picks a token that occurs once in the prompt, at the position `--tail` from the end,
             and passes it as --turn-token.  Any pack - but `strata --serve` refuses to start without `--spec T` (T >= 2), `--mtp DIR`
             and `--prefill CHUNK` (generate.cpp:3681-3685: it always reads a prompt through the verify windows, which need the
             draft layer), so this source KEEPS the config's speculative-decoding flags for every pack (setup.py's config has
             them) and stops with a message when the engine arguments have none.  Metrics: top-1 agreement, perplexity of the
             target tokens, the paired NLL delta, max |dlogp|, NaN / inf in the log-probs; NOT KL or max |dlogit| (the logits
             themselves are not written) - unless the engine has STRATA_LOGITS_DUMP (verify.cpp: the whole row of every logpos line, same
             layout as --dump-logits), which `--exact` and `--ref-kld` switch on (`--logits-dump on` forces it): then the logits exist and
             those two modes compare / score them in full.
    auto     dump for a non-native pack, logpos for a native one.

  Either way the scored rows depend on the batched prefill of the tokens before them - which is the point.

KEEPING THE TWO RUNS COMPARABLE.  Candidate and reference run one after the other on the same card with the same arguments, apart
from `--ref-env` / `--ref-args`.  Unless `--no-fixed-experts` / `--no-pin-expert-cache` say otherwise the harness adds
`--pcie-frac 0 --adapt-swaps 0` (no PCIe share of the misses, no adaptive swaps: the GPU's expert set stays what the profile
chose, bench/results/2026-09-27-esp/esp_kl.py does the same) and pins the candidate's `--expert-cache auto` to the slot
count the reference reported - the GPU and the CPU round an expert differently, upstream measured 2-5 % of top-1 flips just
from the expert cache being on or off, which would drown the Volta kernels' own noise.  Speculative decoding is dropped for the
dump source on a non-native pack (plain decode is what writes logits) and kept for a native pack and for the logpos source (neither
can run without it).

WHICH PROMPT.  Everything that matters on a long prompt is invisible on a short one:
  * the QSA layers (12 of the 48) select at most idx_top_k + idx_block - 1 = 2,051 cells per query
    (qsa_selection_width in include/strata/kernels/qsa.hpp).  At or below 2,051 tokens of context the selection is the
    identity and every query attends to everything; ABOVE it the indexer's block scores and the top-k selection decide
    which cells each query sees, and the prompt-attention kernel runs on 2,051 scattered cells per query.  A prompt
    shorter than that never exercises the selection;
  * the dense GEMMs are chunked (`--prefill auto`: up to 8,192 tokens per chunk) and the FP16 route scales per call,
    so several chunks matter.
`--prompt-name long` therefore builds a prompt of at least 33,000 tokens (tools/volta/prompts/make_long_prompt.py:
this repository's own sources, concatenated) - well past 2,051 and past 4 chunks.  Use it for the gate; `chat` and `code`
are for quick checks of the plumbing.

BIT-EXACT: `--exact`.  The metrics above say "close"; `--exact` says "identical", for the cases where identical is the claim: the same
engine twice (determinism), the port with its fast paths off against upstream built for sm_70, the NUMA mirror on against off.  It compares
the two runs' logits BIT FOR BIT (the `dump` source, and the `logpos` source when both engines write STRATA_LOGITS_DUMP: float32 rows) or
their log-probabilities (`logpos` without the dump: the file has them to 9 decimals, so that is "identical to 1e-9", not a statement about
bits - the report says which it was).  Reported: identical or not, the first differing position (and vocabulary index), the number of rows
and values that differ, the largest distance in float32 ULPs (the number of representable floats between the two values; 1 = adjacent),
the largest absolute difference, and how many rows changed their top-1.  Exit status 0 only when everything is identical; 1 when not.
Two references need more than the same binary twice: `--ref-exe` / `--cand-exe` give a side its own engine (upstream's), and
`--reuse-ref DIR` takes the reference run from an earlier workdir instead of running it again (several candidates, one baseline).

AGAINST LLAMA.CPP: `--ref-kld BASE`.  Bit-exact against llama.cpp is impossible - its CPU and CUDA builds do not match each other bit for
bit, every kernel sums in its own order - so the claim is "no more different from llama.cpp than llama.cpp's own backends are from each
other".  BASE is the file `llama-perplexity -m MODEL.gguf -f text -c 4096 --kl-divergence-base BASE` writes (format and scoring rule:
tools/volta/kld_format.py, from perplexity.cpp).  It holds the tokens of the text split into independent n_ctx-token chunks and, for the
second half of each chunk, llama.cpp's next-token distribution.  The harness runs the Strata engine on EXACTLY those chunk tokens - one engine
launch per chunk, each an independent sequence from position 0 like llama.cpp's - with the first n_ctx/2 tokens through the batched prefill
(where the Volta kernels live) and the rest through the decode path, scores the same positions, and prints llama.cpp's own statistics: mean
KL divergence, same-top-1 fraction, PPL of both, the paired change of ln PPL with its standard error, change of the right token's probability.
  * `dump` source (non-native packs: Q2_0): full logits -> everything above.
  * `logpos` source (native IQ packs): through `strata --serve`; with an engine that has STRATA_LOGITS_DUMP (verify.cpp, this port) the
    full logits rows are written too -> everything above.  Without it only the target token's log-probability and the top-1 id are known:
    PPL, the paired NLL change, same-top-1 and the probability change - NOT the KL, which needs the whole row; the report says so.
  * `--kld-floor LOG` is the stdout of `llama-perplexity --kl-divergence --kl-divergence-base BASE` run with a DIFFERENT llama.cpp backend
    (CPU build against a CUDA-built BASE): the noise floor between two trusted implementations.  Then the verdict (PROVISIONAL, to be
    confirmed by the first real run) is PASS when the mean KL and 1 - same-top-1 are no worse than max(2 x the floor's, a small absolute
    floor: --kld-abs-kl, --kld-abs-top1) and the paired ln-PPL change is within 2 standard errors of zero (or within 2 x the floor's own).
    Without `--kld-floor` the run is only MEASURED, unless --max-kl or --kld-min-top1 is given.
The first n_ctx/2 tokens go through the batched path, so the QSA selection (2,051 cells per query) is only exercised past n_ctx = 4,102;
a BASE made with -c 4096 does not reach it (Gate Q's 33,000-token prompt does), one made with -c 8192 does.
Exit status 0 PASS / MEASURED, 1 FAIL, 2 the harness could not run.

LOGITS FILE FORMAT (what `--dump-logits` writes, and what `--ref-logits` reads): little-endian
    int32 n_vocab, int32 n_rows, then rows of n_vocab float32.
A `.npy` file of shape (rows, n_vocab) is accepted too (numpy.save of a float32 array).  The engine writes the rows of
the positions it processed itself; with `--prefill-until K` the file holds only positions >= K although the header
still counts all of them, so the reader goes by the file size, not the header.  An external file covers the LAST
`rows` positions of the sequence unless `--ref-first-position` says otherwise (a llama.cpp run over the whole text:
first position 0).  Producing one from numpy: `np.array([V, R], np.int32).tofile(f); logits.astype('<f4').tofile(f)`.

EXAMPLES  (run from the repository root, or give absolute paths; every path the harness hands to the engine - the tokens file,
the logits dumps, STRATA_LOGPOS, the engine itself - is made absolute against where YOU run it, because the engine runs in the
config's `cwd`.  Paths INSIDE --engine-args or the config are the engine's own and are relative to its cwd.)
    # Gate Q on a V100.  The config is setup.py's strata-<model>.json in the repository root: strata-q2_0.json (the canonical pack:
    # the logits dump works), strata-iq3_xxs.json / strata-iq2_xs.json / strata-iq3_s.json (native packs: log-probabilities)
    .venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --tail 512
    # the same, standalone (--ple-gguf and the rest are ENGINE arguments, so they go in --engine-args; --pack is forwarded to it):
    .venv/bin/python tools/volta/golden_compare.py --exe build-sm70/strata --pack pack/full \\
        --engine-args "--native /data/model-00001-of-00002.gguf --ple-gguf /data/model-00002-of-00002.gguf --expert-profile data/expert-profile.bin --expert-cache auto" \\
        --prompt-name long --tail 512                    # (the logpos source also needs --spec 4 --mtp <dir> in --engine-args)
    # a native (IQ) pack: log-probabilities of the last 512 tokens of a 33,000-token prompt, via the resident engine
    .venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --tail 512
    # the same pack, greedy: do 256 generated tokens agree?
    .venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --mode greedy --max-new 256
    # compare two logits files made elsewhere:
    .venv/bin/python tools/volta/golden_compare.py --prompt-file text.txt --cand-logits v100.bin --ref-logits llamacpp.npy
    # bit-exact: the same command twice (determinism), then the port against upstream's sm_70 build (--ref-exe) with fast paths off
    .venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --tail 512 --exact --ref-env ""
    # against llama.cpp on the same GGUF (BASE from llama-perplexity --kl-divergence-base), with llama.cpp's own CPU-vs-CUDA numbers as the floor
    .venv/bin/python tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --ref-kld cuda.kld --kld-floor cpu_vs_cuda.log

PYTHON.  Needs numpy, and `regex` to tokenize text (tools/strata_tokenizer.py): `./setup.sh` installs both into the repository's
.venv, so run this with `.venv/bin/python`; with another interpreter `pip install numpy regex`.

Exit status: 0 PASS, 1 FAIL (a threshold, a NaN, or the candidate engine failed), 2 the harness could not run (the
reference failed, a file is missing or malformed).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("golden_compare: this tool needs numpy (and `regex` to tokenize text), and the Python running it (%s) has none.\n"
             "  Use the repository's environment: .venv/bin/python %s ... (./setup.sh creates it),\n"
             "  or install them: %s -m pip install numpy regex" % (sys.executable, sys.argv[0] if sys.argv and sys.argv[0] else "tools/volta/golden_compare.py",
                                                                    sys.executable))

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROMPT_DIR = HERE / "prompts"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import kld_format as K  # noqa: E402  (llama.cpp's --kl-divergence-base file and the scoring rule)

DEFAULT_REF_ENV = "STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0"
CHAT_WRAP = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # tools/calibrate.py:chat_ids
SPARSE_FROM = 2051     # qsa_selection_width: idx_top_k 2048 + idx_block 4 - 1.  Up to it the selection is the identity.
LONG_TOKENS = 33000    # the `long` prompt: past 2,051 and past four 8,192-token prefill chunks
GATE_Q_ROWS = 500      # Gate Q of docs/volta/PLAN.md: top-1 >= 99 % over 500 tokens


class HarnessError(Exception):
    """Something the harness could not do (exit status 2)."""


# ================================================================================================ logits files


class Logits:
    """Rows of logits (float32, memory-mapped when possible) and the sequence position each row belongs to."""

    def __init__(self, rows, vocab: int, path: str, header_rows: int | None):
        self.rows = rows
        self.vocab = vocab
        self.path = path
        self.header_rows = header_rows
        self.positions = np.zeros(0, dtype=np.int64)
        self._index: dict[int, int] | None = None

    @property
    def n_rows(self) -> int:
        return int(self.rows.shape[0])

    def place(self, positions) -> "Logits":
        positions = np.asarray(positions, dtype=np.int64)
        if len(positions) != self.n_rows:
            raise HarnessError(f"{self.path}: the file has {self.n_rows} rows but {len(positions)} positions were expected "
                               f"(header says {self.header_rows}) - a different --prefill-until / --stride / --max-new from "
                               "the one that wrote it, or an interrupted run?")
        self.positions = positions
        self._index = None
        return self

    def row_of(self, position: int) -> int:
        if self._index is None:
            self._index = {int(p): i for i, p in enumerate(self.positions)}
        return self._index[int(position)]


def selected_positions(total: int, stride: int, first: int = 0) -> np.ndarray:
    """The positions the engine's dump selects (`logits_selection::selected`: multiples of `stride`, and the last), from
    `first` on: the token loop starts at `first` (after a batched prefill) and never visits the positions before it."""
    if total <= 0 or first >= total:
        return np.zeros(0, dtype=np.int64)
    p = np.arange(first, total, dtype=np.int64)
    return p[(p % stride == 0) | (p == total - 1)]


def read_logits(path: str | Path) -> Logits:
    """Open a logits file: the engine's (int32 n_vocab, int32 n_rows, float32 rows) or a .npy (rows, n_vocab) array.
    The row count comes from the FILE SIZE: after `--prefill-until K` the engine's header still counts the positions it
    skipped.  Call `.place()` to say which position each row is."""
    path = Path(path)
    if not path.exists():
        raise HarnessError(f"{path}: no such logits file")
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            raise HarnessError(f"{path}: expected a 2-D (rows, n_vocab) array, got shape {arr.shape}")
        return Logits(arr, int(arr.shape[1]), str(path), None)
    size = path.stat().st_size
    if size < 8:
        raise HarnessError(f"{path}: {size} bytes - too short for the 8-byte header")
    hdr = np.fromfile(path, dtype="<i4", count=2)
    vocab, header_rows = int(hdr[0]), int(hdr[1])
    if vocab <= 0 or header_rows < 0:
        raise HarnessError(f"{path}: implausible header (n_vocab {vocab}, n_rows {header_rows}) - not a logits file?")
    body = size - 8
    if body % (4 * vocab) != 0:
        raise HarnessError(f"{path}: {body} data bytes is not a whole number of {vocab}-float rows "
                           "(truncated write: did the engine crash mid-run?)")
    rows = body // (4 * vocab)
    if rows > header_rows:
        raise HarnessError(f"{path}: {rows} rows in the file but the header says {header_rows}")
    arr = np.memmap(path, dtype="<f4", mode="r", offset=8, shape=(rows, vocab)) if rows else np.zeros((0, vocab), np.float32)
    return Logits(arr, vocab, str(path), header_rows)


# ============================================================================================ the metrics math


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def compare_rows(ref: Logits, cand: Logits, positions, targets: dict[int, int], chunk: int = 16) -> dict:
    """Metrics of `cand` against `ref` over `positions` (rows present in both).  `targets[p]` is the token the row at
    position p is scored against for perplexity (absent: the row is not scored).  float64 throughout; rows with a
    non-finite logit in either run are counted and left out of the other statistics."""
    positions = np.asarray(positions, dtype=np.int64)
    for name, lg in (("reference", ref), ("candidate", cand)):
        try:
            for p in positions[:1]:
                lg.row_of(int(p))
        except KeyError:
            raise HarnessError(f"position {int(p)} is not in the {name}'s logits") from None
    top1_agree = rows_used = nonfinite_c = nonfinite_r = 0
    max_abs, row_max_abs, kl_rows, nll_r, nll_c, margins, flipped = 0.0, [], [], [], [], [], []
    for s in range(0, len(positions), chunk):
        pos = positions[s:s + chunk]
        r = np.asarray(ref.rows[[ref.row_of(int(p)) for p in pos]], dtype=np.float64)
        c = np.asarray(cand.rows[[cand.row_of(int(p)) for p in pos]], dtype=np.float64)
        nonfinite_r += int((~np.isfinite(r)).sum())
        nonfinite_c += int((~np.isfinite(c)).sum())
        ok = np.isfinite(r).all(axis=1) & np.isfinite(c).all(axis=1)
        if not ok.any():
            continue
        r, c, pos_ok = r[ok], c[ok], pos[ok]
        rows_used += len(pos_ok)
        lr, lc = log_softmax(r), log_softmax(c)
        agree = r.argmax(axis=1) == c.argmax(axis=1)
        top1_agree += int(agree.sum())
        d = np.abs(r - c).max(axis=1)
        row_max_abs.extend(d.tolist())
        max_abs = max(max_abs, float(d.max()))
        kl_rows.extend((np.exp(lr) * (lr - lc)).sum(axis=1).tolist())
        if (~agree).any():
            top2 = np.partition(r[~agree], -2, axis=1)
            margins.extend((top2[:, -1] - top2[:, -2]).tolist())
            flipped.extend(int(p) for p in pos_ok[~agree])
        for k, p in enumerate(pos_ok):
            t = targets.get(int(p))
            if t is not None:
                nll_r.append(float(-lr[k, t]))
                nll_c.append(float(-lc[k, t]))
    out: dict = {"rows_compared": int(len(positions)), "rows_scored": rows_used, "nonfinite_cand": nonfinite_c,
                 "nonfinite_ref": nonfinite_r}
    if rows_used:
        kl = np.array(kl_rows)
        out.update({"top1_agree": top1_agree, "top1": top1_agree / rows_used,
                    "max_abs_dlogit": max_abs, "mean_rowmax_abs_dlogit": float(np.mean(row_max_abs)),
                    "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()),
                    "flipped_positions": flipped[:50],
                    "flip_margin_median": float(np.median(margins)) if margins else None,
                    "flip_margin_max": float(max(margins)) if margins else None})
    if nll_r:
        a, b = np.array(nll_r), np.array(nll_c)
        d = b - a
        ppl_r, ppl_c = float(np.exp(a.mean())), float(np.exp(b.mean()))
        out.update({"ppl_rows": len(a), "ppl_ref": ppl_r, "ppl_cand": ppl_c, "ppl_rel": abs(ppl_c - ppl_r) / ppl_r,
                    "dnll_mean": float(d.mean()),
                    "dnll_se": float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None})
    return out


def verdict(m: dict, *, min_top1: float, max_ppl_rel: float, max_kl: float | None = None,
            max_abs_dlogit: float | None = None, need_ppl: bool = True) -> tuple[bool, list[str]]:
    """PASS / FAIL and the reasons, from compare_rows' output."""
    why = []
    if m.get("nonfinite_cand", 0):
        why.append(f"{m['nonfinite_cand']} non-finite logits in the candidate run")
    if m.get("nonfinite_ref", 0):
        why.append(f"{m['nonfinite_ref']} non-finite logits in the REFERENCE run (the comparison is meaningless)")
    if "top1" not in m:
        why.append("no comparable rows")
        return False, why
    if m["top1"] < min_top1:
        why.append(f"top-1 agreement {100 * m['top1']:.2f}% < {100 * min_top1:.2f}%")
    if need_ppl and "ppl_rel" in m and m["ppl_rel"] > max_ppl_rel:
        why.append(f"|dPPL|/PPL {100 * m['ppl_rel']:.2f}% > {100 * max_ppl_rel:.2f}%")
    if need_ppl and "ppl_rel" not in m:
        why.append("no perplexity could be computed (no scored targets)")
    if max_kl is not None and m.get("kl_mean", 0) > max_kl:
        why.append(f"mean KL {m['kl_mean']:.3g} > {max_kl:g}")
    if max_abs_dlogit is not None and m.get("max_abs_dlogit", 0) > max_abs_dlogit:
        why.append(f"max |dlogit| {m['max_abs_dlogit']:.3g} > {max_abs_dlogit:g}")
    return not why, why


def identical_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ================================================================================================ engine arguments


def flag_value(args: list[str], flag: str) -> str | None:
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return args[i + 1]
    return None


def drop_flag(args: list[str], flag: str, takes_value: bool = True) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 2 if takes_value else 1
            continue
        out.append(args[i])
        i += 1
    return out


def set_flag(args: list[str], flag: str, value: str) -> list[str]:
    return drop_flag(args, flag) + [flag, value]


# speculative decoding: the token loop (the only writer of --dump-logits) is skipped once it takes over
SPEC_FLAGS = ["--spec", "--mtp", "--spec-min-p", "--suffix-draft", "--mtp-max-t", "--mtp-window", "--spec-oracle",
              "--spec-corrupt"]
# what the harness sets itself: a leftover from a server config (or --engine-args) must not fight it
OWN_FLAGS = ["--tokens", "--tokens-file", "--max-new", "--dump-logits", "--logits-stride", "--prefill-until", "--seed",
             "--max-context"]


@dataclass
class Plan:
    mode: str
    n_tokens: int
    max_new: int
    stride: int
    prefill_until: int | None   # teacher: K (0 = token path for everything); greedy: None
    native: bool
    pos_start: int = 0          # the first position the token loop visits (= the first row of the dump)
    total: int = 0              # the engine's dump_positions: n_tokens - 1 + max_new
    max_context: int = 0
    source: str = "dump"        # teacher mode: "dump" (--dump-logits) or "logpos" (--serve + STRATA_LOGPOS)
    turn_token: int = -1        # logpos: the token whose last occurrence splits the prompt
    split_at: int = 0           # logpos: its position = the number of tokens read through the batched path

    def __post_init__(self):
        self.total = self.n_tokens - 1 + self.max_new


def build_engine_args(base: list[str], plan: Plan, *, tokens_file: str, dump: str, fixed_experts: bool, keep_spec: bool,
                      stats: bool = False, extra: list[str] | None = None) -> list[str]:
    """The engine's command line for one run, from the configured arguments `base`; fills plan.pos_start / total /
    max_context."""
    args = list(base)
    for f in ("--serve", "--greedy", "--stats", "--spec-split"):
        args = drop_flag(args, f, takes_value=False)
    for f in OWN_FLAGS:
        args = drop_flag(args, f)
    cfg_ctx = flag_value(base, "--max-context")
    need = plan.n_tokens + plan.max_new + 8
    plan.max_context = (max(int(cfg_ctx) if cfg_ctx and cfg_ctx.isdigit() else 0, need) + 1023) // 1024 * 1024
    if not plan.native and not keep_spec:
        for f in SPEC_FLAGS:
            args = drop_flag(args, f)
    # the batched prefill path is on unless teacher mode asked for the token path (prefill_until == 0)
    if plan.mode == "teacher" and plan.prefill_until == 0:
        args = drop_flag(args, "--prefill")
    elif flag_value(args, "--prefill") in (None, "0"):
        args = set_flag(args, "--prefill", "auto")
    prefilled = flag_value(args, "--prefill") not in (None, "0") and plan.n_tokens > 1
    if prefilled and plan.prefill_until and 0 < plan.prefill_until < plan.n_tokens - 1:
        args += ["--prefill-until", str(plan.prefill_until)]
        plan.pos_start = plan.prefill_until
    else:
        plan.pos_start = plan.n_tokens - 1 if prefilled else 0
    if fixed_experts:   # esp_kl.py's recipe: the same experts on the GPU in both runs (no adaptive swaps, no PCIe share)
        args = set_flag(args, "--pcie-frac", "0")
        args = set_flag(args, "--adapt-swaps", "0")
    args += ["--tokens-file", tokens_file, "--max-new", str(plan.max_new), "--max-context", str(plan.max_context), "--greedy",
             "--dump-logits", dump]
    if plan.stride > 1:
        args += ["--logits-stride", str(plan.stride)]
    if stats:
        args.append("--stats")
    return args + (extra or [])


def parse_env_pairs(text: str) -> dict[str, str]:
    out = {}
    for tok in shlex.split(text or ""):
        if "=" not in tok:
            raise HarnessError(f"bad environment entry {tok!r}: expected NAME=VALUE")
        k, v = tok.split("=", 1)
        out[k] = v
    return out


# ============================================================================================ the engine's output

_OUT = re.compile(r"^output\s*:(.*)$", re.M)
_PERF = {
    # std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s ...", "decode" / "prefill", ...)  (generate.cpp, end of main)
    "prefill": re.compile(r"^prefill\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M),
    "decode": re.compile(r"^decode\s+(\d+) tokens in ([\d.]+) ms\s+->\s+([\d.]+) tok/s", re.M),
    # std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts ...")
    "batched": re.compile(r"strata generate: prefill (\d+) tokens in (\d+) chunks, ([\d.]+) ms \(([\d.]+) tok/s\)"),
}


def parse_engine_log(text: str) -> dict:
    """The generated ids and the speeds from an engine run's merged stdout / stderr."""
    out: dict = {"output": None}
    m = _OUT.search(text)
    if m:
        out["output"] = [int(x) for x in m.group(1).split() if x.lstrip("-").isdigit()]
    for k in ("prefill", "decode"):
        m = _PERF[k].search(text)
        if m:
            out[k] = {"tokens": int(m.group(1)), "ms": float(m.group(2)), "tok_s": float(m.group(3))}
    m = _PERF["batched"].search(text)
    if m:
        out["batched_prefill"] = {"tokens": int(m.group(1)), "chunks": int(m.group(2)), "ms": float(m.group(3)),
                                  "tok_s": float(m.group(4))}
    return out


# ================================================================================================ running the engine


_CACHE_SLOTS = re.compile(r"strata generate: expert cache (\d+) slots, ")


def expert_cache_slots(log: str | Path) -> int | None:
    """The slot count of the GPU expert tier an engine run settled on (its last `expert cache N slots` line)."""
    try:
        found = _CACHE_SLOTS.findall(Path(log).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None
    return int(found[-1]) if found else None


def load_engine_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if "exe" not in cfg or "args" not in cfg:
        raise HarnessError(f"{path}: not an engine config (needs 'exe' and 'args', as setup.py writes them)")
    return cfg


def base_environment(cfg: dict, gpu: str | None) -> dict[str, str]:
    """serve/server.py:child_env for the one engine process: the config's env, the CUDA libraries, the GPU choice."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k, v in (cfg.get("env") or {}).items():
        env[str(k)] = str(v)
    dirs = [d for d in cfg.get("lib_dirs") or [] if Path(d).is_dir()]
    if dirs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    return env


def run_engine(label: str, exe: list[str], args: list[str], base_env: dict[str, str], env_extra: dict[str, str],
               cwd: str | None, log: Path, timeout: float | None) -> dict:
    cmd = [*exe, *args]
    env = dict(base_env)
    env.update(env_extra)
    print(f"[{label}] {shlex.quote(cmd[0])} {' '.join(shlex.quote(a) for a in args[:6])} ...", flush=True)
    if env_extra:
        print(f"[{label}] env: {' '.join(f'{k}={v}' for k, v in env_extra.items())}", flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as lf:
        lf.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n# env: " + " ".join(f"{k}={v}" for k, v in env_extra.items()) + "\n")
        lf.flush()
        try:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=cwd, timeout=timeout or None).returncode
        except subprocess.TimeoutExpired:
            rc = -999
        except FileNotFoundError as e:
            raise HarnessError(f"cannot run the engine: {e}") from e
    secs = time.time() - t0
    text = log.read_text(encoding="utf-8", errors="replace")
    info = parse_engine_log(text)
    info.update({"kind": "engine", "returncode": rc, "seconds": secs, "log": str(log), "cmd": cmd, "env": env_extra})
    speed = f"; batched prefill {info['batched_prefill']['tok_s']:.0f} tok/s" if "batched_prefill" in info else ""
    print(f"[{label}] exit {rc} in {secs:.0f} s{speed}", flush=True)
    if rc != 0:
        print(f"[{label}] FAILED (see {log}); last lines:\n" + "\n".join(text.strip().splitlines()[-12:]), flush=True)
    return info


# ============================================================================================ the prompt


def load_tokenizer(tok_dir: str):
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import strata_tokenizer as ST   # needs `regex`
    except ImportError as e:
        raise HarnessError(f"cannot import tools/strata_tokenizer.py ({e}); it needs the `regex` package: run this with the repository's "
                           f".venv/bin/python (./setup.sh installs it), or `{sys.executable} -m pip install regex`, or pass --ids-file") from e
    d = Path(tok_dir)
    if not (d / "vocab.json").exists():
        raise HarnessError(f"{d}: no vocab.json - pass --tokenizer PACK/tokenizer (the directory setup.py extracts)")
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (d / "merges.txt").read_text(encoding="utf-8").split("\n"),
                        json.loads((d / "token_type.json").read_text()))


def read_ids(path: str | Path) -> list[int]:
    try:
        ids = [int(x) for x in Path(path).read_text(encoding="utf-8").replace(",", " ").split()]
    except ValueError as e:
        raise HarnessError(f"{path}: not a list of token ids ({e})") from e
    if not ids:
        raise HarnessError(f"{path}: no token ids")
    return ids


def get_tokens(a, tokenizer_dir: str | None, workdir: Path) -> list[int]:
    """The prompt as token ids: pretokenized, or tokenized here (tools/strata_tokenizer.py with the pack's tokenizer/)."""
    if a.ids_file:
        return read_ids(a.ids_file)
    chat = a.chat
    if a.prompt:
        text = a.prompt
    elif a.prompt_file:
        text = Path(a.prompt_file).read_text(encoding="utf-8")
    elif a.prompt_name in ("chat", "code"):
        text = (PROMPT_DIR / f"{a.prompt_name}.txt").read_text(encoding="utf-8")
        chat = not a.no_chat
    elif a.prompt_name == "long":
        ids_path = workdir / "long.ids"
        if not ids_path.exists():
            if not tokenizer_dir:
                raise HarnessError("--prompt-name long needs a tokenizer to count tokens (--tokenizer PACK/tokenizer, or an "
                                   "engine config that names one); or build the ids with tools/volta/prompts/make_long_prompt.py "
                                   "and pass --ids-file")
            sys.path.insert(0, str(PROMPT_DIR))
            import make_long_prompt as M
            ids = M.build_ids(M.collect_sources(ROOT), load_tokenizer(tokenizer_dir).encode, LONG_TOKENS)
            ids_path.write_text(" ".join(map(str, ids)), encoding="utf-8")
            print(f"[prompt] built the long prompt: {len(ids)} tokens -> {ids_path}", flush=True)
        return read_ids(ids_path)
    else:
        raise HarnessError("give a prompt: --prompt-name chat|code|long, --prompt-file, --prompt or --ids-file")
    if not tokenizer_dir:
        raise HarnessError("tokenizing text needs --tokenizer PACK/tokenizer (or give --ids-file)")
    if chat:
        text = CHAT_WRAP.format(text=text.strip())
    ids = load_tokenizer(tokenizer_dir).encode(text, parse_special=True)
    return ids[: a.max_prompt_tokens] if a.max_prompt_tokens else ids


# ============================================================================================ logpos: serve + STRATA_LOGPOS


def choose_split(tokens: list[int], k0: int, search: int = 6000) -> tuple[int, int]:
    """(position, token) for the prompt split of the logpos source: the engine reads [0, p) batched and [p, n-1) through the
    verify windows when `tokens[p]` is the LAST occurrence of --turn-token.  Looks from k0 down for a token that occurs
    exactly once in the whole prompt (so no earlier occurrence adds a "root" split either), then for one that merely
    does not occur again after p.  Position 0 never counts (the engine's scans start at 1)."""
    n = len(tokens)
    count: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, t in enumerate(tokens):
        count[t] = count.get(t, 0) + 1
        last[t] = i
    lo = max(1, k0 - search)
    for p in range(min(k0, n - 2), lo - 1, -1):
        if count[tokens[p]] == 1:
            return p, tokens[p]
    for p in range(min(k0, n - 2), lo - 1, -1):
        if last[tokens[p]] == p:
            return p, tokens[p]
    raise HarnessError(f"no token that occurs once (or last) was found between positions {lo} and {k0}: pass --turn-token ID "
                       "and --prefill-until, or use a prompt with more variety")


def build_serve_args(base: list[str], plan: Plan, *, fixed_experts: bool, keep_spec: bool = True, extra: list[str] | None = None,
                     short_margin: int = 8, no_root: bool = False) -> list[str]:
    """`strata --serve ...` for the logpos source: the configured arguments, the split token and the window length.

    `strata --serve` refuses to start without `--spec T` (T >= 2), `--mtp DIR` and `--prefill CHUNK` > 0 (generate.cpp:3681-3685: a
    prompt is read through the verify windows, which need the draft layer), so the speculative-decoding flags are kept for EVERY
    pack - `keep_spec` is accepted for symmetry with `build_engine_args` and ignored here - and HarnessError says what is missing
    when the arguments have none."""
    args = list(base)
    for f in ("--serve", "--greedy", "--stats", "--spec-split"):
        args = drop_flag(args, f, takes_value=False)
    for f in OWN_FLAGS + ["--turn-token", "--short-read"]:
        args = drop_flag(args, f)
    cfg_ctx = flag_value(base, "--max-context")
    plan.max_context = (max(int(cfg_ctx) if cfg_ctx and cfg_ctx.isdigit() else 0, plan.n_tokens + plan.max_new + 8) + 1023) // 1024 * 1024
    spec, mtp = flag_value(args, "--spec"), flag_value(args, "--mtp")
    if not (spec and spec.isdigit() and int(spec) >= 2) or not mtp:
        raise HarnessError("the logpos source starts `strata --serve`, which refuses to run without --spec T (T >= 2) and --mtp DIR "
                           "(and --prefill CHUNK; generate.cpp:3681-3685), and the engine arguments have "
                           + ("no --spec" if not spec else f"--spec {spec}") + (" and no --mtp" if not mtp else "")
                           + ": use setup.py's config (it passes --spec 4 --mtp <dir>), or add them in --engine-args"
                           + (", or use --teacher-source dump (the logits dump; non-native packs only)" if not plan.native else ""))
    if flag_value(args, "--prefill") in (None, "0"):
        args = set_flag(args, "--prefill", "auto")
    if fixed_experts:
        args = set_flag(args, "--pcie-frac", "0")
        args = set_flag(args, "--adapt-swaps", "0")
    rows = plan.n_tokens - 1 - plan.split_at
    if no_root:     # --ref-kld: no extra "system prompt" split at an earlier occurrence of the turn token (the chunks are plain text)
        args = set_flag(args, "--prompt-cache-root", "0")
    # `short_margin`: the tail may be this many tokens longer than the rows scored.  The first part (split_at tokens) must be LONGER than
    # --short-read to go through the batched path, so --ref-kld, whose split is at n_ctx/2, needs 0 (split_at > rows)
    args += ["--max-context", str(plan.max_context), "--turn-token", str(plan.turn_token), "--short-read", str(rows + short_margin)]
    return ["--serve"] + args + (extra or [])


class ServeSession:
    """One `strata --serve` process spoken to as serve/server.py speaks to it."""

    def __init__(self, cmd: list[str], env: dict[str, str], cwd: str | None, log: Path):
        self.log = open(log, "w", encoding="utf-8")
        self.log.write("# " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        self.log.flush()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                     env=env, cwd=cwd)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def read_until(self, prefix: str, timeout: float | None, collect: list[str] | None = None) -> str:
        """Lines until one starts with `prefix` (returned).  Raises HarnessError when the engine ends or says ERR first."""
        deadline = time.time() + timeout if timeout else None
        while True:
            left = None if deadline is None else max(0.1, deadline - time.time())
            try:
                line = self.lines.get(timeout=left if left is not None else 60)
            except queue.Empty:
                if deadline is not None and time.time() >= deadline:
                    raise HarnessError(f"the engine did not answer within {timeout:.0f} s (waiting for {prefix!r})") from None
                continue
            if line is None:
                raise HarnessError(f"the engine exited (code {self.proc.poll()}) before {prefix!r}")
            if collect is not None:
                collect.append(line.rstrip("\n"))
            if line.startswith(prefix):
                return line.rstrip("\n")
            if line.startswith("ERR"):
                raise HarnessError("the engine reported " + line.strip())

    def send(self, text: str) -> None:
        try:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()
        except OSError as e:
            raise HarnessError(f"the engine's pipe is gone (exit code {self.proc.poll()}): {e}") from e

    def close(self) -> int | None:
        try:
            self.send("QUIT")
            self.proc.wait(timeout=60)
        except (HarnessError, subprocess.TimeoutExpired):
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        return self.proc.returncode


LOGPOS_FIELDS = ("pos", "target", "logprob", "top", "top_logprob", "hit", "extra", "without")


def read_logpos(path: Path) -> tuple[dict[int, dict], int]:
    """STRATA_LOGPOS lines -> {position: {...}}, and the number of non-finite numbers.  The format is verify.cpp's
    window_logprobs: pos, target, log p(target), top-1 id, log p(top-1), hit, log p(extra id), log p(target | not extra)."""
    rows: dict[int, dict] = {}
    bad = 0
    if not path.exists():
        return rows, 0
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        f = ln.split("\t")
        if len(f) < 6:
            continue
        try:
            pos, tgt, top = int(f[0]), int(f[1]), int(f[3])
            lp, tlp = float(f[2]), float(f[4])
        except ValueError:
            bad += 1
            continue
        if not (math.isfinite(lp) and math.isfinite(tlp)):
            bad += 1
        rows[pos] = {"target": tgt, "logprob": lp, "top": top, "top_logprob": tlp}
    return rows, bad


def compare_logpos(ref: dict[int, dict], cand: dict[int, dict], bad_ref: int, bad_cand: int) -> dict:
    """The metrics a log-probability file allows: top-1 agreement, perplexity of the targets, paired NLL delta, max |dlogp|."""
    common = sorted(set(ref) & set(cand))
    out: dict = {"rows_compared": len(common), "rows_scored": 0, "nonfinite_cand": bad_cand, "nonfinite_ref": bad_ref,
                 "logprobs_only": True}
    good = [p for p in common if all(math.isfinite(x) for x in (ref[p]["logprob"], cand[p]["logprob"], ref[p]["top_logprob"], cand[p]["top_logprob"]))]
    if not good:
        return out
    r = np.array([ref[p]["logprob"] for p in good])
    c = np.array([cand[p]["logprob"] for p in good])
    agree = np.array([ref[p]["top"] == cand[p]["top"] for p in good])
    d = c - r
    out.update({"rows_scored": len(good), "top1_agree": int(agree.sum()), "top1": float(agree.mean()),
                "flipped_positions": [p for p, a in zip(good, agree) if not a][:50],
                "max_abs_dlogit": float(np.abs(d).max()), "mean_rowmax_abs_dlogit": float(np.abs(d).mean()),
                "max_abs_dtop_logprob": float(max(abs(cand[p]["top_logprob"] - ref[p]["top_logprob"]) for p in good)),
                "ppl_rows": len(good), "ppl_ref": float(np.exp(-r.mean())), "ppl_cand": float(np.exp(-c.mean()))})
    out["ppl_rel"] = abs(out["ppl_cand"] - out["ppl_ref"]) / out["ppl_ref"]
    out["dnll_mean"] = float(-d.mean())
    out["dnll_se"] = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else None
    return out


def run_serve_teacher(label: str, exe: list[str], args: list[str], base_env: dict[str, str], env_extra: dict[str, str], cwd: str | None,
                      workdir: Path, tokens: list[int], timeout: float | None, stem: str | None = None,
                      logits: bool = False) -> dict:
    """Start `strata --serve`, send one `GEN 1 <ids>`, wait for DONE, QUIT.  STRATA_LOGPOS goes to <stem>.logpos (stem: ref / cand by label).
    `logits`: also set STRATA_LOGITS_DUMP=<stem>.logits - the full logits row of every logpos line, from an engine that has the hook
    (verify.cpp); an engine without it simply writes no such file and the info has no "logits" key."""
    short = stem or ("ref" if label == "reference" else "cand")
    logpos = workdir / f"{short}.logpos"
    logpos.unlink(missing_ok=True)
    env = dict(base_env)
    env.update(env_extra)
    env["STRATA_LOGPOS"] = str(logpos)
    logits_path = workdir / f"{short}.logits"
    logits_path.unlink(missing_ok=True)
    if logits:
        env["STRATA_LOGITS_DUMP"] = str(logits_path)
    else:
        env.pop("STRATA_LOGITS_DUMP", None)
    cmd = [*exe, *args]
    log = workdir / f"{short}.log"
    print(f"[{label}] {shlex.quote(cmd[0])} {' '.join(shlex.quote(a) for a in args[:5])} ... (serve; STRATA_LOGPOS={logpos.name})", flush=True)
    if env_extra:
        print(f"[{label}] env: {' '.join(f'{k}={v}' for k, v in env_extra.items())}", flush=True)
    t0 = time.time()
    info: dict = {"kind": "engine", "log": str(log), "cmd": cmd, "env": env_extra, "logpos": str(logpos), "returncode": 0}
    sess = ServeSession(cmd, env, cwd, log)
    try:
        sess.read_until("READY", timeout or 7200)
        print(f"[{label}] engine ready after {time.time() - t0:.0f} s; sending the {len(tokens)}-token prompt", flush=True)
        t1 = time.time()
        sess.send("GEN 1 " + ",".join(map(str, tokens)))
        seen: list[str] = []
        done = sess.read_until("DONE", timeout or 7200, collect=seen)
        f = done.split()
        info["serve"] = {"generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]), "decode_ms": float(f[4]),
                         "finish": f[5] if len(f) > 5 else ""}
        info["output"] = [int(x[2:]) for x in seen if x.startswith("T ")]
        pps = [x.split() for x in seen if x.startswith("PP ") and len(x.split()) >= 5]
        if pps:
            info["pp"] = [{"done": int(p[1]), "total": int(p[2]), "ms": float(p[3]), "tok_s": float(p[4])} for p in pps]
        info["seconds"] = time.time() - t1
    except HarnessError as e:
        info["returncode"] = sess.proc.poll() if sess.proc.poll() not in (None, 0) else 1
        info["error"] = str(e)
        info["seconds"] = time.time() - t0
        print(f"[{label}] FAILED: {e} (see {log})", flush=True)
    finally:
        rc = sess.close()
        if info["returncode"] == 0 and rc not in (0, None):
            info["returncode"] = rc
    if logits and logits_path.exists() and logits_path.stat().st_size > 8:
        info["logits"] = str(logits_path)
    print(f"[{label}] exit {info['returncode']} in {info.get('seconds', 0):.0f} s", flush=True)
    return info


# ====================================================================================================== --exact: bit for bit


def ulp_key(x: np.ndarray) -> np.ndarray:
    """float32 -> int64 that grows with the value, so |key(a) - key(b)| is the number of representable float32 values between a and b
    (1 = neighbours).  -0.0 and +0.0 share key 0.  Meaningless for NaN (the callers screen it)."""
    s = np.ascontiguousarray(x, dtype=np.float32).view("<i4").astype(np.int64)
    return np.where(s >= 0, s, -(s & 0x7FFFFFFF))


def exact_compare_rows(ref_rows, cand_rows, positions: list[int], chunk: int = 16) -> dict:
    """Compare two sets of logits rows BIT FOR BIT.  `ref_rows(lo, hi)` / `cand_rows(lo, hi)` return rows lo..hi-1 of each side (float32,
    same order as `positions`).  A value is different when its 32 bits differ (so a NaN payload, or +0.0 against -0.0, counts);
    the ULP distance and the absolute difference are taken over the values that differ and are not NaN."""
    n = len(positions)
    if n == 0:
        raise HarnessError("--exact: the two runs have no position in common")
    out: dict = {"kind": "logits", "rows": n, "values_per_row": None, "rows_differing": 0, "values_differing": 0, "nan_values": 0,
                 "first_diff_position": None, "first_diff_index": None, "first_diff_values": None, "max_ulp": 0, "max_abs_diff": 0.0,
                 "top1_changed_rows": 0, "differing_positions": []}
    for s0 in range(0, n, chunk):
        r = np.ascontiguousarray(ref_rows(s0, s0 + chunk), dtype=np.float32)
        c = np.ascontiguousarray(cand_rows(s0, s0 + chunk), dtype=np.float32)
        if r.shape != c.shape:
            raise HarnessError(f"--exact: the two runs' rows have different shapes ({r.shape} against {c.shape}): not the same vocabulary?")
        out["values_per_row"] = int(r.shape[1])
        diff = r.view("<u4") != c.view("<u4")
        row_diff = diff.any(axis=1)
        if not row_diff.any():
            continue
        out["rows_differing"] += int(row_diff.sum())
        out["values_differing"] += int(diff.sum())
        nan = np.isnan(r) | np.isnan(c)
        out["nan_values"] += int((diff & nan).sum())
        real = diff & ~nan
        if real.any():
            out["max_ulp"] = max(out["max_ulp"], int(np.abs(ulp_key(r) - ulp_key(c))[real].max()))
            with np.errstate(invalid="ignore"):
                d = np.abs(r.astype(np.float64) - c.astype(np.float64))
            d = np.where(real & np.isfinite(d), d, 0.0)
            out["max_abs_diff"] = max(out["max_abs_diff"], float(d.max()))
        rows = np.nonzero(row_diff)[0]
        out["top1_changed_rows"] += int((r[rows].argmax(axis=1) != c[rows].argmax(axis=1)).sum())
        for i in rows:
            if len(out["differing_positions"]) < 20:
                out["differing_positions"].append(int(positions[s0 + int(i)]))
        if out["first_diff_position"] is None:
            i = int(rows[0])
            j = int(np.argmax(diff[i]))
            out["first_diff_position"], out["first_diff_index"] = int(positions[s0 + i]), j
            out["first_diff_values"] = [float(r[i, j]), float(c[i, j])]
    out["identical"] = out["rows_differing"] == 0
    return out


def exact_compare_logpos(ref: dict[int, dict], cand: dict[int, dict]) -> dict:
    """The `logpos` source without full logits: the log-probability of the target, the top-1 id and its log-probability per position, as the
    engine printed them (9 decimals).  Equal text = equal here; this is NOT a statement about the bits of the logits."""
    common = sorted(set(ref) & set(cand))
    same = lambda a, b: a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))
    out: dict = {"kind": "logpos", "rows": len(common), "only_reference": len(set(ref) - set(cand)), "only_candidate": len(set(cand) - set(ref)),
                 "rows_differing": 0, "first_diff_position": None, "first_diff_values": None, "max_abs_diff": 0.0, "top1_changed_rows": 0,
                 "differing_positions": []}
    if not common:
        raise HarnessError("--exact: the two runs have no position in common")
    for p in common:
        a, b = ref[p], cand[p]
        if all(same(a[k], b[k]) for k in ("target", "logprob", "top", "top_logprob")):
            continue
        out["rows_differing"] += 1
        if len(out["differing_positions"]) < 20:
            out["differing_positions"].append(p)
        if out["first_diff_position"] is None:
            out["first_diff_position"], out["first_diff_values"] = p, [a["logprob"], b["logprob"]]
        out["top1_changed_rows"] += int(a["top"] != b["top"])
        for k in ("logprob", "top_logprob"):
            if math.isfinite(a[k]) and math.isfinite(b[k]):
                out["max_abs_diff"] = max(out["max_abs_diff"], abs(a[k] - b[k]))
    out["identical"] = out["rows_differing"] == 0 and not out["only_reference"] and not out["only_candidate"]
    return out


def format_exact_report(rep: dict) -> str:
    c, e = rep["config"], rep["exact"]
    L = ["=" * 100, f"golden_compare --exact: mode {c['mode']}, {c['n_tokens']} prompt tokens; reference: {c.get('reference', '?')}"]
    L += [f"  note: {x}" for x in rep["notes"]]
    L += run_lines(rep)
    L.append("-" * 100)
    span = f" (positions {c['first_pos']}..{c['last_pos']})" if c.get("first_pos") is not None else ""
    if e["kind"] == "logits":
        L.append(f"  compared                 {e['rows']} rows of {e['values_per_row']} float32 logits{span}, bit for bit")
        L.append(f"  rows differing           {e['rows_differing']} of {e['rows']}   values differing {e['values_differing']}"
                 + (f"   (of which NaN involved: {e['nan_values']})" if e["nan_values"] else ""))
        if not e["identical"]:
            fv = e["first_diff_values"]
            L.append(f"  first difference         position {e['first_diff_position']}, vocabulary index {e['first_diff_index']}: "
                     f"reference {fv[0]!r} vs candidate {fv[1]!r}")
            L.append(f"  largest distance         {e['max_ulp']} ULP (float32 neighbours = 1), |difference| {e['max_abs_diff']:.3e}")
            L.append(f"  top-1 changed            {e['top1_changed_rows']} of {e['rows']} rows")
            L.append(f"  first differing rows     positions {e['differing_positions']}")
    else:
        L.append(f"  compared                 {e['rows']} log-probability rows{span} as printed (9 decimals): equal text, "
                 "NOT a bitwise statement about the logits (no STRATA_LOGITS_DUMP on one side)")
        L.append(f"  rows differing           {e['rows_differing']} of {e['rows']}   only in the reference {e['only_reference']}, "
                 f"only in the candidate {e['only_candidate']}")
        if not e["identical"] and e["first_diff_position"] is not None:
            fv = e["first_diff_values"]
            L.append(f"  first difference         position {e['first_diff_position']}: log p(target) reference {fv[0]!r} vs candidate {fv[1]!r}")
            L.append(f"  largest |d log p|        {e['max_abs_diff']:.3e} nats;  top-1 changed {e['top1_changed_rows']} rows")
    L += ["=" * 100, "RESULT: " + ("IDENTICAL" if e["identical"] else "DIFFERENT")]
    L += [f"  - {r}" for r in rep["reasons"]]
    return "\n".join(L)


# ====================================================================================================== the report


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def gate_q_status(rep: dict) -> tuple[str, list[str]]:
    """Does this run establish Gate Q of docs/volta/PLAN.md (top-1 >= 99 % over 500 tokens, PPL within 1-2 %, no NaN / inf, on a prompt
    long enough to exercise the batched path and the sparse selection)?  -> ("PASS" | "FAIL" | "not established", what is missing)."""
    if not rep["pass"]:
        return "FAIL", []
    c, m = rep["config"], rep.get("metrics") or {}
    gaps: list[str] = []
    if c["mode"] != "teacher":
        gaps.append("greedy mode shows what a user would see, it is not the gate (the default teacher mode is)")
    if m.get("tokens_only"):
        gaps.append("token ids only: no perplexity, no NaN scan")
    rows = m.get("rows_scored", 0)
    if rows < GATE_Q_ROWS:
        gaps.append(f"{rows} rows scored, the gate asks for {GATE_Q_ROWS} (--tail {GATE_Q_ROWS})")
    runs = rep.get("runs", {})
    if any(r.get("kind") == "file" for r in runs.values()):
        gaps.append("a side is a logits file made elsewhere: this harness cannot tell that it exercised the batched Volta paths")
    elif c["mode"] == "teacher":
        batched = c.get("split_at") if c.get("source") == "logpos" else c.get("prefill_until")
        if not batched or batched <= SPARSE_FROM:
            gaps.append(f"only {batched or 0} prompt tokens went through the batched path (the Volta kernels): the gate needs more than "
                        f"{SPARSE_FROM} (the QSA selection is the identity up to there); use --prompt-name long")
    return ("PASS" if not gaps else "not established"), gaps


def run_lines(rep: dict) -> list[str]:
    """One line per side of a two-run report: how long it took, how fast, and the environment it ran in."""
    L = []
    for k in ("candidate", "reference"):
        r = rep["runs"].get(k)
        if not r:
            continue
        if r["kind"] == "engine":
            sp = (f", batched prefill {r['batched_prefill']['tok_s']:.0f} tok/s" if "batched_prefill" in r else "") + \
                 (f", decode {r['decode']['tok_s']:.1f} tok/s" if "decode" in r else "") + \
                 (f", prompt read in {r['serve']['prompt_ms'] / 1000:.0f} s" if "serve" in r else "")
            pp = [p for p in r.get("pp", []) if p["done"] == r.get("batched_to")] if r.get("batched_to") else []
            if pp:
                sp += f" (batched part {pp[0]['tok_s']:.0f} tok/s)"
            env = " ".join(f"{a}={b}" for a, b in r["env"].items()) or "(the base environment)"
            exe = f"; engine {r['exe']}" if r.get("exe") else ""
            L.append(f"  {k:<9} {r['seconds']:.0f} s, exit {r['returncode']}{sp}; env: {env}{exe}")
        else:
            L.append(f"  {k:<9} logits file {r['path']}")
    return L


def format_report(rep: dict) -> str:
    c, m, g, th = rep["config"], rep.get("metrics"), rep.get("greedy"), rep["thresholds"]
    L = ["=" * 100, f"golden_compare: mode {c['mode']}, {c['n_tokens']} prompt tokens; reference: {c.get('reference', '?')}"]
    L += [f"  note: {x}" for x in rep["notes"]]
    L += run_lines(rep)
    if m:
        L.append("-" * 100)
        span = f"   (positions {c['first_pos']}..{c['last_pos']})" if c.get("first_pos") is not None else ""
        L.append(f"  rows compared            {m['rows_scored']} of {m['rows_compared']}{span}")
        if m.get("logprobs_only") and "top1" in m:
            L.append(f"  top-1 agreement          {pct(m['top1'])}   ({m['top1_agree']}/{m['rows_scored']}; gate >= {pct(th['min_top1'])})")
            L.append(f"  |d log p(target)|        max {m['max_abs_dlogit']:.4g}   mean {m['mean_rowmax_abs_dlogit']:.4g}   (nats)")
            L.append(f"  |d log p(top-1)|         max {m['max_abs_dtop_logprob']:.4g}")
            L.append("  KL / max |dlogit|        not available: this source writes log-probabilities, not logits")
        elif "kl_mean" in m:
            L.append(f"  top-1 agreement          {pct(m['top1'])}   ({m['top1_agree']}/{m['rows_scored']}; gate >= {pct(th['min_top1'])})")
            L.append(f"  max |dlogit|             {m['max_abs_dlogit']:.4g}   (mean of the per-row maxima {m['mean_rowmax_abs_dlogit']:.4g})")
            L.append(f"  KL(ref || cand)          mean {m['kl_mean']:.3e}   p99 {m['kl_p99']:.3e}   max {m['kl_max']:.3e}  nats")
            if m.get("flip_margin_median") is not None:
                L.append(f"  at the {m['rows_scored'] - m['top1_agree']} flipped rows    the reference's top-2 margin: median "
                         f"{m['flip_margin_median']:.3f}, max {m['flip_margin_max']:.3f} logits  (small = a near tie)")
        elif "top1" in m:
            L.append(f"  top-1 agreement          {pct(m['top1'])}   ({m['top1_agree']}/{m['rows_scored']}; gate >= {pct(th['min_top1'])}; token ids only)")
        if "ppl_ref" in m:
            se = f" +- {m['dnll_se']:.4f}" if m.get("dnll_se") is not None else ""
            L.append(f"  perplexity               ref {m['ppl_ref']:.4f}   cand {m['ppl_cand']:.4f}   |d|/PPL {pct(m['ppl_rel'])}"
                     f"   (gate <= {pct(th['max_ppl_rel'])}; {m['ppl_rows']} scored tokens)")
            L.append(f"  paired NLL delta         cand - ref = {m['dnll_mean']:+.5f}{se} nats/token")
        L.append(f"  non-finite logits        candidate {m['nonfinite_cand']}   reference {m['nonfinite_ref']}")
    if g:
        L.append("-" * 100)
        L.append(f"  generated tokens         candidate {g['n_cand']}, reference {g['n_ref']}")
        L.append(f"  identical prefix         {g['prefix']} of {g['n_max']} tokens; positional agreement {pct(g['token_agreement'])}")
        if g["first_diff"] is not None:
            L.append(f"  first difference         token #{g['first_diff']}: candidate {g['cand_at_diff']} vs reference {g['ref_at_diff']}")
    L += ["=" * 100, "RESULT: " + ("PASS" if rep["pass"] else "FAIL")]
    L += [f"  - {r}" for r in rep["reasons"]]
    status, gaps = gate_q_status(rep)
    rep["gate_q"] = {"status": status, "missing": gaps}
    L.append(f"GATE Q (docs/volta/PLAN.md): {status}" + (" - " + "; ".join(gaps) if gaps else ""))
    return "\n".join(L)


def finish(a, workdir: Path, rep: dict) -> int:
    if rep.get("exact") is not None:
        text = format_exact_report(rep)
    elif rep.get("kld") is not None:
        text = format_kld_report(rep)
    else:
        text = format_report(rep) if rep.get("metrics") is not None else \
            "golden_compare: " + ("; ".join(rep["reasons"]) or "no result") + "".join(f"\n  note: {x}" for x in rep["notes"])
    print(text)
    (workdir / "report.txt").write_text(text + "\n", encoding="utf-8")
    jp = a.json or str(workdir / "report.json")
    slim = json.loads(json.dumps(rep, default=str))
    for r in slim["runs"].values():
        r.pop("cmd", None)
    Path(jp).write_text(json.dumps(slim, indent=1), encoding="utf-8")
    print(f"golden_compare: report {jp}")
    if a.delete_logits:
        for f in ("ref.bin", "cand.bin"):
            (workdir / f).unlink(missing_ok=True)
    return rep["exit"]


# ====================================================================================================== main flow


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("engine")
    g.add_argument("--engine-config", help="the strata-*.json that setup.py writes ({exe, args, cwd, lib_dirs, env, tokenizer})")
    g.add_argument("--exe", help="the engine binary (instead of / overriding the config's); relative paths are taken from where you run this")
    g.add_argument("--engine-args", default="", help="engine arguments, one shell-quoted string (added to the config's)")
    g.add_argument("--pack", help="the pack directory (relative to where you run this): forwarded to the engine as --pack (replacing the config's), "
                                         "and used to detect a native pack")
    g.add_argument("--cwd", help="working directory of the engine (default: the config's - the repository root -, else the current one)")
    g.add_argument("--gpu", help="CUDA_VISIBLE_DEVICES for both runs (numbered as nvidia-smi does)")
    g.add_argument("--native-pack", choices=["auto", "yes", "no"], default="auto",
                   help="is the pack native (IQ)?  auto: <pack>/native_experts.txt exists.  Native packs have no logits dump")
    g.add_argument("--keep-spec", action="store_true",
                   help="keep the config's --spec/--mtp (speculative decoding); default: plain decode, which is what writes "
                        "logits (a native pack always keeps it - it cannot run without)")
    g.add_argument("--no-fixed-experts", action="store_true",
                   help="do not add --pcie-frac 0 --adapt-swaps 0 (which keep the GPU's expert set identical in both runs)")
    g.add_argument("--no-pin-expert-cache", action="store_true",
                   help="do not pin the candidate's `--expert-cache auto` to the slot count the reference run reported (the two "
                        "runs then size the GPU expert tier from their own free VRAM, which can differ by a slot and moves a few "
                        "near-ties: the GPU and the CPU round an expert differently)")
    g.add_argument("--stats", action="store_true", help="pass --stats to the engine (more in the logs)")
    g.add_argument("--timeout", type=float, default=0, help="seconds per engine run (default: none)")
    p = ap.add_argument_group("prompt")
    p.add_argument("--prompt-name", choices=["chat", "code", "long"],
                   help="tools/volta/prompts/: chat, code, or long (>= 33,000 tokens of this repository's sources)")
    p.add_argument("--prompt-file", help="a text file to tokenize")
    p.add_argument("--prompt", help="the prompt text itself")
    p.add_argument("--ids-file", help="pretokenized prompt: token ids separated by commas / whitespace")
    p.add_argument("--chat", action="store_true", help="wrap --prompt/--prompt-file in the chat template (chat and code prompts always are)")
    p.add_argument("--no-chat", action="store_true", help="do not wrap the chat / code prompts in the chat template")
    p.add_argument("--tokenizer", help="the pack's tokenizer/ directory (default: the config's, else <pack>/tokenizer)")
    p.add_argument("--max-prompt-tokens", type=int, default=0, help="truncate the tokenized prompt to this many tokens")
    r = ap.add_argument_group("the run")
    r.add_argument("--mode", choices=["teacher", "greedy"], default="teacher")
    r.add_argument("--tail", type=int, default=512,
                   help="teacher: positions scored through the token path after the batched prefill (default 512)")
    r.add_argument("--prefill-until", default="auto",
                   help="teacher: positions batched before the scored tail: auto (= prompt length - tail), 0 (the whole prompt "
                        "through the token path; the batched path is NOT exercised) or a number")
    r.add_argument("--stride", type=int, default=1, help="teacher (dump source): --logits-stride (store every Nth tail row; default 1)")
    r.add_argument("--teacher-source", choices=["auto", "dump", "logpos"], default="auto",
                   help="teacher mode: dump = standalone engine + --dump-logits (full logits, non-native packs); logpos = "
                        "`strata --serve` + STRATA_LOGPOS (log-probabilities, any pack); auto: dump, or logpos for a native pack")
    r.add_argument("--turn-token", type=int, default=None,
                   help="logpos source: the token id whose LAST occurrence splits the prompt into the batched part and the scored "
                        "tail (default: a token that occurs once, near --tail from the end)")
    r.add_argument("--max-new", type=int, default=0, help="greedy: tokens to generate (default 256); teacher mode generates one")
    r.add_argument("--workdir", help="where the logits and logs go (default ./golden_out/<time> under the directory you run this from; always "
                                             "made absolute, because the engine runs in its own cwd)")
    r.add_argument("--reuse", action="store_true", help="reuse the runs already in --workdir instead of running again")
    r.add_argument("--delete-logits", action="store_true", help="delete the (large) logits files afterwards")
    c = ap.add_argument_group("candidate and reference")
    c.add_argument("--cand-env", default="", help="NAME=VALUE ... for the candidate run (default none: the port's own paths)")
    c.add_argument("--cand-args", default="", help="extra engine arguments for the candidate run only")
    c.add_argument("--ref-env", default=DEFAULT_REF_ENV,
                   help=f"environment of the reference run (default {DEFAULT_REF_ENV!r}; '' = the same as the candidate: a determinism check)")
    c.add_argument("--ref-args", default="", help="extra engine arguments for the reference run only")
    c.add_argument("--ref-logits", help="reference logits from elsewhere (engine format or .npy) instead of a second run")
    c.add_argument("--ref-first-position", type=int, help="sequence position of the reference file's first row (default: it ends at the last position)")
    c.add_argument("--cand-logits", help="candidate logits from elsewhere (skips the candidate run)")
    c.add_argument("--cand-first-position", type=int, help="sequence position of the candidate file's first row")
    t = ap.add_argument_group("thresholds")
    t.add_argument("--min-top1", type=float, default=0.99)
    t.add_argument("--max-ppl-rel", type=float, default=0.02, help="max |PPL_cand - PPL_ref| / PPL_ref")
    t.add_argument("--max-kl", type=float, default=None, help="optional: max mean KL (nats)")
    t.add_argument("--max-abs-dlogit", type=float, default=None, help="optional: max |dlogit|")
    e = ap.add_argument_group("bit-exact comparison (--exact)")
    e.add_argument("--exact", action="store_true",
                   help="compare the two runs BIT FOR BIT (logits; or printed log-probabilities when a side has no STRATA_LOGITS_DUMP) and say "
                        "identical / the first differing position / how many rows / the largest ULP distance; exit 0 only if identical")
    e.add_argument("--ref-exe", help="the engine binary of the REFERENCE run only (e.g. upstream's sm_70 build); default: the candidate's")
    e.add_argument("--cand-exe", help="the engine binary of the CANDIDATE run only; default: the reference's")
    e.add_argument("--reuse-ref", metavar="DIR", help="take the reference run from the --workdir of an earlier run on the SAME prompt and plan "
                   "(its ref.* files) instead of running it again: one baseline, several candidates")
    e.add_argument("--logits-dump", choices=["auto", "on", "off"], default="auto",
                   help="serve-mode (logpos) runs: ask the engine for STRATA_LOGITS_DUMP, the full logits rows.  auto: on for --exact and "
                        "--ref-kld, off otherwise.  An engine without the hook just writes nothing and the comparison falls back")
    k = ap.add_argument_group("against llama.cpp (--ref-kld)")
    k.add_argument("--ref-kld", metavar="BASE", help="a llama.cpp `--kl-divergence-base` file: score the engine on its text, chunk by chunk, "
                   "with llama.cpp's own statistics (KL, same top-1, PPL, paired ln PPL change, dp)")
    k.add_argument("--kld-chunks", type=int, default=0, help="use only the first N chunks of the file (default: all)")
    k.add_argument("--kld-bos", default="auto", metavar="auto|none|ID",
                   help="what llama.cpp writes over the first token of every chunk: none, a BOS token id, or auto = read the GGUF "
                        "(--kld-gguf, else the engine's --native) for add_bos_token / bos_token_id")
    k.add_argument("--kld-gguf", help="the GGUF llama.cpp ran on (for --kld-bos auto; default: the engine arguments' --native)")
    k.add_argument("--kld-trim-vocab", action="store_true",
                   help="the engine's logits rows are longer than the file's n_vocab (padding rows): compare the first n_vocab columns")
    k.add_argument("--kld-floor", metavar="LOG", help="stdout of `llama-perplexity --kl-divergence` on a DIFFERENT llama.cpp backend against the "
                   "same BASE (the noise floor): turns the run into the provisional PASS / FAIL rule (see the header)")
    k.add_argument("--kld-factor", type=float, default=2.0, help="the rule allows this many times the floor's KL and top-1 mismatch (default 2)")
    k.add_argument("--kld-abs-kl", type=float, default=5e-4,
                   help="... but never less than this mean KL in nats (default 5e-4: a different engine is not a different backend of the "
                        "same one; below it a mean KL is rounding noise, a Q8_0 quantisation of a 7B model scores about that)")
    k.add_argument("--kld-abs-top1", type=float, default=0.01,
                   help="... and never less than this share of positions whose top-1 differs (default 0.01)")
    k.add_argument("--kld-min-top1", type=float, default=None, help="no floor given: PASS needs same-top-1 >= this (default: no verdict)")
    k.add_argument("--kld-search", type=int, default=256,
                   help="logpos source: how far past n_ctx/2 to look for a token to split the chunk at (its last occurrence; default 256)")
    k.add_argument("--keep-logits", action="store_true", help="keep the per-chunk logits files (about 2 GB per 4096-token chunk) after scoring")
    ap.add_argument("--summarize-report", metavar="REPORT_JSON",
                    help="print `STATUS<TAB>one line` for a report.json this tool wrote, and exit (for scripts)")
    ap.add_argument("--print-config", action="store_true",
                    help="print KEY=VALUE lines about the engine config (EXE, GGUF, PACK, NATIVE_PACK, ...) for scripts, and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the two engine commands, run nothing (the runs take minutes each)")
    ap.add_argument("--json", help="write the full report as JSON (default <workdir>/report.json)")
    return ap


# ================================================================================================ --ref-kld: against llama.cpp


def pinned_gguf_urls(gguf: str) -> list[str]:
    """Where setup.py got a Qwen3.8-Flash-Next GGUF shard from (HF_REVISIONS pins the commit): the URLs of its two shards, so a script can say how to
    fetch the SAME file llama.cpp must load.  Only the original model's repository (the Coder and Swift releases have their own); [] when the
    file name or setup.py is not of that shape."""
    m = re.match(r"^Qwen3\.8-Flash-Next-GSQ-RCO-(?P<q>[A-Z0-9_]+)-0000\d-of-00002\.gguf$", Path(gguf).name)
    try:
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
    except OSError:
        return []
    rev = re.search(r'"ISTA-DASLab/Qwen3\.8-Flash-Next-GSQ-RCO-GGUF":\s*"([0-9a-f]{40})"', text)
    if not m or not rev:
        return []
    q = m.group("q")
    return [f"https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/{rev.group(1)}/{q}/Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf"
            for i in (1, 2)]


def print_config(a) -> int:
    """--print-config: what a script needs to know about an engine config, as KEY=VALUE lines (no quoting: none of the values has a newline)."""
    ctx = engine_context(a, offline=True, make_workdir=False)
    native_gguf = flag_value(ctx.base_args, "--native")
    rows = {"EXE": ctx.cand_exe[-1] if ctx.cand_exe else "", "CWD": ctx.cwd or "", "PACK": str(ctx.pack_dir),
            "NATIVE_PACK": "1" if ctx.native else "0", "GGUF": absolute(native_gguf, ctx.cwd) or "" if native_gguf else "",
            "TOKENIZER": ctx.tok_dir or "", "SPEC": flag_value(ctx.base_args, "--spec") or "", "MTP": flag_value(ctx.base_args, "--mtp") or "",
            "EXPERT_CACHE": flag_value(ctx.base_args, "--expert-cache") or "", "MAX_CONTEXT": flag_value(ctx.base_args, "--max-context") or "",
            "KV": flag_value(ctx.base_args, "--kv") or ""}
    rows["GGUF_URLS"] = " ".join(pinned_gguf_urls(rows["GGUF"])) if rows["GGUF"] else ""
    for k, v in rows.items():
        print(f"{k}={v}")
    return 0


def choose_turn_after(tokens: list[int], first: int, search: int = 256) -> tuple[int, int]:
    """(position, token) to split a chunk at for the logpos source: the first position p >= first whose token has no later occurrence (the
    engine splits at the LAST occurrence of --turn-token), preferring - within `search` positions - a token that occurs once in the
    whole chunk.  The positions before p go through the batched path, p .. n-2 through the verify windows, whose log-probabilities are
    written; llama.cpp scores first .. n-2, so p == first loses nothing and p > first loses the p - first positions before it."""
    n = len(tokens)
    count: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, t in enumerate(tokens):
        count[t] = count.get(t, 0) + 1
        last[t] = i
    hi = min(n - 2, first + search)
    for p in range(first, hi + 1):
        if count[tokens[p]] == 1:
            return p, tokens[p]
    for p in range(first, hi + 1):
        if last[tokens[p]] == p:
            return p, tokens[p]
    raise HarnessError(f"no token between positions {first} and {hi} of this chunk has its last occurrence there, so the verify-window split "
                       f"cannot be placed (--kld-search {search}); raise --kld-search")


def resolve_bos(a, ctx: Ctx, kb, notes: list[str]) -> int | None:
    v = str(a.kld_bos)
    if v == "none":
        bos = None
    elif v != "auto":
        if not v.lstrip("-").isdigit():
            raise HarnessError(f"--kld-bos {v!r}: expected auto, none or a token id")
        bos = int(v)
    else:
        gguf = a.kld_gguf or flag_value(ctx.base_args, "--native")
        if not gguf:
            raise HarnessError("--kld-bos auto reads the GGUF llama.cpp ran on: give --kld-gguf PATH (or an engine config with --native), or say "
                               "--kld-bos none (a Qwen / GPT-2 vocabulary: llama.cpp adds no BOS) or --kld-bos ID")
        gguf = absolute(gguf, ctx.cwd)
        if not Path(gguf).exists():
            raise HarnessError(f"--kld-bos auto: {gguf} does not exist; give the GGUF (--kld-gguf) or --kld-bos none|ID")
        bos = K.bos_substitution(gguf)
        notes.append(f"BOS: {gguf} says llama.cpp " + ("adds none" if bos is None else f"writes token {bos} over the first token of every chunk"))
    if bos is not None and int(kb.tokens[0, 0]) != bos:
        notes.append(f"the file's first token is {int(kb.tokens[0, 0])}, not the BOS {bos}: llama.cpp tokenised the text without it, but it still "
                     "overwrites the first token of every chunk with BOS when it evaluates, and so does this run")
    return bos


def kld_verdict(s: dict, floor: dict | None, a) -> tuple[str, list[str], dict]:
    """PASS / FAIL / MEASURED, the reasons, and the numbers the rule used.  PROVISIONAL: the first real run on the box confirms the rule."""
    why: list[str] = []
    info: dict = {"provisional": True}
    if s.get("nonfinite"):
        why.append(f"{s['nonfinite']} rows had a non-finite logit (NaN / inf)")
    if not s.get("count"):
        return "FAIL", why + ["no scored rows"], info
    if floor is not None:
        f_kld, f_top = floor["kld"][0], 1.0 - floor["same_top"][0] / 100.0
        f_dn = floor["ln_ppl_ratio"][0]
        kl_bar, top_bar = max(a.kld_factor * f_kld, a.kld_abs_kl), max(a.kld_factor * f_top, a.kld_abs_top1)
        se = s["ln_ppl_ratio_unc"]
        dn_bar = max(2.0 * se, a.kld_factor * abs(f_dn))
        info.update({"floor_kld": f_kld, "floor_top1_mismatch": f_top, "floor_ln_ppl_ratio": f_dn, "kl_bar": kl_bar, "top1_bar": top_bar,
                     "dnll_bar": dn_bar})
        if s.get("has_kl"):
            if s["kld_mean"] > kl_bar:
                why.append(f"mean KLD {s['kld_mean']:.3e} > {kl_bar:.3e} (= max({a.kld_factor:g} x the floor's {f_kld:.3e}, {a.kld_abs_kl:g}))")
        else:
            info["partial"] = True
        if s["top1_mismatch"] > top_bar:
            why.append(f"top-1 differs at {100 * s['top1_mismatch']:.3f}% of positions > {100 * top_bar:.3f}% "
                       f"(= max({a.kld_factor:g} x the floor's {100 * f_top:.3f}%, {100 * a.kld_abs_top1:.2f}%))")
        if abs(s["dnll_mean"]) > dn_bar:
            why.append(f"paired change of ln PPL {s['dnll_mean']:+.5f} is outside {dn_bar:.5f} (= max(2 standard errors {2 * se:.5f}, "
                       f"{a.kld_factor:g} x the floor's |{f_dn:+.5f}|))")
        return ("FAIL" if why else "PASS"), why, info
    checks = []
    if a.max_kl is not None and s.get("has_kl"):
        checks.append(("mean KLD", s["kld_mean"], a.max_kl, False))
    if a.kld_min_top1 is not None:
        checks.append(("same top-1", s["same_top"], a.kld_min_top1, True))
    if not checks:
        return ("FAIL" if why else "MEASURED"), why, info
    for name, got, bar, at_least in checks:
        if (got < bar) if at_least else (got > bar):
            why.append(f"{name} {got:.4g} {'<' if at_least else '>'} {bar:g}")
    return ("FAIL" if why else "PASS"), why, info


def format_kld_report(rep: dict) -> str:
    c, k = rep["config"], rep["kld"]
    L = ["=" * 100, f"golden_compare --ref-kld {c['ref_kld']}",
         f"  llama.cpp base: n_ctx {c['n_ctx']}, vocabulary {c['n_vocab']}, {c['chunks']} chunks used; positions {c['first']}..{c['n_ctx'] - 2} of each "
         f"are scored (the first {c['first']} are context); source {c['source']}"]
    L += [f"  note: {x}" for x in rep["notes"]]
    for name, r in rep["runs"].items():
        if r.get("returncode", 0) != 0:
            L.append(f"  {name}: FAILED, exit {r['returncode']} (see {r.get('log')})")
            continue
        sp = f", batched prefill {r['batched_prefill']['tok_s']:.0f} tok/s" if "batched_prefill" in r else ""
        sp += f", prompt read in {r['serve']['prompt_ms'] / 1000:.0f} s" if "serve" in r else ""
        full = "full logits" if r.get("rows_kind") == "logits" else "target log-probabilities only"
        L.append(f"  {name}: {r.get('rows_scored', 0)} rows ({full}){', split at ' + str(r['split_at']) if 'split_at' in r else ''}; "
                 f"{r.get('seconds', 0):.0f} s{sp}")
    L.append("-" * 100)
    L.append(K.format_summary(k["summary"]))
    if k.get("floor"):
        f, i = k["floor"], k["verdict_info"]
        L += ["-" * 100, f"  noise floor, llama.cpp against itself ({c.get('kld_floor')}): mean KLD {f['kld'][0]:.3e}, same top p "
                         f"{f['same_top'][0]:.3f}%, ln(PPL(Q)/PPL(base)) {f['ln_ppl_ratio'][0]:+.5f} +- {f['ln_ppl_ratio'][1]:.5f}"]
        s = k["summary"]
        if s.get("has_kl"):
            ratio = f" = {s['kld_mean'] / f['kld'][0]:.2f} x the floor" if f["kld"][0] > 0 else ""
            L.append(f"  this engine against llama.cpp: mean KLD {s['kld_mean']:.3e}{ratio} "
                     f"(allowed {i['kl_bar']:.3e}); top-1 differs at {100 * s['top1_mismatch']:.3f}% (allowed {100 * i['top1_bar']:.3f}%); "
                     f"ln PPL change {s['dnll_mean']:+.5f} +- {s['dnll_se'] or 0:.5f} (allowed +-{i['dnll_bar']:.5f})")
        else:
            L.append(f"  this engine against llama.cpp: top-1 differs at {100 * s['top1_mismatch']:.3f}% (allowed {100 * i['top1_bar']:.3f}%); "
                     f"ln PPL change {s['dnll_mean']:+.5f} +- {s['dnll_se'] or 0:.5f} (allowed +-{i['dnll_bar']:.5f}); KL not checked: no full logits")
    v = k["verdict"]
    L += ["=" * 100, f"KLD VERDICT (provisional, to be confirmed by the first real run): {v}"
          + (" (partial: KL not available, only the target log-probabilities were written)" if k.get("verdict_info", {}).get("partial") else "")]
    if v == "MEASURED":
        L.append("  no pass rule was given: --kld-floor LOG (llama.cpp's own CPU-vs-CUDA numbers), or --max-kl / --kld-min-top1")
    L += [f"  - {r}" for r in rep["reasons"]]
    L.append("RESULT: " + ("FAIL" if rep["exit"] else "PASS" if v == "PASS" else "MEASURED"))
    return "\n".join(L)


def run_kld(a) -> int:
    for flag, name in ((a.ref_logits, "--ref-logits"), (a.cand_logits, "--cand-logits"), (a.exact, "--exact"), (a.reuse_ref, "--reuse-ref"),
                       (a.mode == "greedy", "--mode greedy")):
        if flag:
            raise HarnessError(f"--ref-kld cannot be combined with {name}: it runs the engine once per chunk of the llama.cpp file and scores it")
    ctx = engine_context(a)
    kb = K.read_base(a.ref_kld)
    n_chunks = min(kb.chunks_present, a.kld_chunks) if a.kld_chunks > 0 else kb.chunks_present
    notes: list[str] = list(kb.notes)
    bos = resolve_bos(a, ctx, kb, notes)
    source = a.teacher_source if a.teacher_source != "auto" else ("logpos" if ctx.native else "dump")
    if source == "dump" and ctx.native:
        raise HarnessError("the pack is NATIVE (IQ): the engine's logits dump is empty for it; --teacher-source logpos (the default) scores it "
                           "through `strata --serve`")
    n_ctx, first = kb.n_ctx, kb.first
    if first <= SPARSE_FROM:
        notes.append(f"only the first {first} tokens of a chunk go through the batched path: up to {SPARSE_FROM} the QSA selection is the identity, "
                     f"so the sparse top-k prompt attention is not exercised (llama-perplexity -c {2 * (SPARSE_FROM + 1) + 2} or more, i.e. 8192, "
                     "does; Gate Q's long prompt does too)")
    if source == "logpos":
        notes.append("logpos source: one `strata --serve` launch per chunk (the turn token that splits the prompt is an engine argument and "
                     "differs per chunk); the first n_ctx/2 tokens (up to the split) are read batched, the rest through the verify windows")
    base_env = ctx.base_env
    cand_env = parse_env_pairs(a.cand_env)
    th = {"max_kl": a.max_kl, "kld_min_top1": a.kld_min_top1, "factor": a.kld_factor, "abs_kl": a.kld_abs_kl, "abs_top1": a.kld_abs_top1}

    def chunk_tokens(c: int) -> list[int]:
        toks = kb.chunk_tokens(c)
        if bos is not None:
            toks[0] = bos
        return toks

    def plan_of(c: int, toks: list[int]) -> Plan:
        plan = Plan("teacher", n_ctx, 1, 1, first, ctx.native)
        plan.source = source
        if source == "logpos":
            plan.split_at, plan.turn_token = choose_turn_after(toks, first, a.kld_search)
        return plan

    if a.dry_run:
        toks = chunk_tokens(0)
        plan = plan_of(0, toks)
        print(f"golden_compare --ref-kld (dry run): {a.ref_kld}: n_ctx {n_ctx}, vocabulary {kb.n_vocab}, {n_chunks} chunk(s) of {kb.chunks_present}; "
              f"BOS {'none' if bos is None else bos}; source {source}; workdir {ctx.workdir}")
        print(f"  each chunk: the first {plan.split_at if source == 'logpos' else first} tokens batched, positions "
              f"{plan.split_at if source == 'logpos' else first}..{n_ctx - 2} scored against tokens {first + 1}..{n_ctx - 1}")
        for n in notes:
            print(f"  note: {n}")
        ids = ctx.workdir / "chunk0.ids"
        env = dict(cand_env)
        if source == "logpos":
            args = build_serve_args(ctx.base_args, plan, fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec,
                                    extra=shlex.split(a.cand_args), short_margin=0, no_root=True)
            env["STRATA_LOGPOS"] = str(ctx.workdir / "chunk0.logpos")
            if want_logits(a):
                env["STRATA_LOGITS_DUMP"] = str(ctx.workdir / "chunk0.logits")
            print(f"  (chunk 0 split at position {plan.split_at}, turn token {plan.turn_token}; the other chunks have their own)")
        else:
            args = build_engine_args(ctx.base_args, plan, tokens_file=str(ids), dump=str(ctx.workdir / "chunk0.bin"),
                                     fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec, stats=a.stats, extra=shlex.split(a.cand_args))
        print("\n[chunk 0] " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + (" " if env else "")
              + " ".join(shlex.quote(c) for c in [*ctx.cand_exe, *args]))
        return 0

    scorer = K.KldScorer()
    runs: dict = {}
    splits: list[int] = []
    cand_base = ctx.base_args
    t_all = time.time()
    for c in range(n_chunks):
        toks = chunk_tokens(c)
        plan = plan_of(c, toks)
        stem = f"chunk{c}"
        ids_file = ctx.workdir / f"{stem}.ids"
        ids_file.write_text(" ".join(map(str, toks)), encoding="utf-8")
        block = kb.block(c)
        scale, mlp = kb.header_of(block)
        targets = kb.targets(c)
        V = kb.n_vocab
        label = f"chunk {c + 1}/{n_chunks}"
        run_info: dict
        if source == "dump":
            dump, log = ctx.workdir / f"{stem}.bin", ctx.workdir / f"{stem}.log"
            args = build_engine_args(cand_base, plan, tokens_file=str(ids_file), dump=str(dump), fixed_experts=not a.no_fixed_experts,
                                     keep_spec=a.keep_spec, stats=a.stats, extra=shlex.split(a.cand_args))
            run_info = run_engine(label, ctx.cand_exe, args, base_env, cand_env, ctx.cwd, log, a.timeout or None)
            runs[stem] = run_info
            if run_info["returncode"] != 0:
                return finish(a, ctx.workdir, kld_rep(a, ctx, kb, source, n_chunks, bos, notes, runs, th, None, None, 1,
                                                      [f"the engine failed on chunk {c} (exit {run_info['returncode']}, see {log})"]))
            lg = place_engine_logits(dump, plan)
            if lg.n_rows < kb.n_scored:
                raise HarnessError(f"{dump}: {lg.n_rows} rows, expected at least {kb.n_scored} (positions {first}..{n_ctx - 2})")
            if lg.vocab != V:
                if lg.vocab > V and a.kld_trim_vocab:
                    notes.append(f"the engine's rows have {lg.vocab} columns, the file's vocabulary is {V}: the first {V} are compared")
                else:
                    raise HarnessError(f"the engine writes {lg.vocab} logits per position, the llama.cpp file has n_vocab {V}: not the same "
                                       "vocabulary (--kld-trim-vocab compares the first n_vocab columns when the engine's are padding)")
            scorer.add_logits(lg.rows[:kb.n_scored, :V], targets, scale, mlp, block[:, 4:4 + V])
            run_info.update({"rows_scored": kb.n_scored, "rows_kind": "logits"})
            if not a.keep_logits:
                dump.unlink(missing_ok=True)
        else:
            args = build_serve_args(cand_base, plan, fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec,
                                    extra=shlex.split(a.cand_args), short_margin=0, no_root=True)
            run_info = run_serve_teacher(label, ctx.cand_exe, args, base_env, cand_env, ctx.cwd, ctx.workdir, toks, a.timeout or None,
                                         stem=stem, logits=want_logits(a))
            run_info["batched_to"] = plan.split_at
            run_info["split_at"] = plan.split_at
            runs[stem] = run_info
            splits.append(plan.split_at)
            if run_info["returncode"] != 0:
                return finish(a, ctx.workdir, kld_rep(a, ctx, kb, source, n_chunks, bos, notes, runs, th, None, None, 1,
                                                      [f"the engine failed on chunk {c} (exit {run_info['returncode']}, see {ctx.workdir / (stem + '.log')})"]))
            rows_d, bad = read_logpos(Path(run_info["logpos"]))
            span = list(range(plan.split_at, n_ctx - 1))
            if set(rows_d) != set(span):
                got = sorted(rows_d)
                raise HarnessError(f"chunk {c}: the engine wrote log-probabilities for {len(got)} positions"
                                   + (f" ({got[0]}..{got[-1]})" if got else "") + f", expected {plan.split_at}..{n_ctx - 2}: did the tail go through the "
                                   f"verify windows?  See {ctx.workdir / (stem + '.log')} (--short-read / --turn-token {plan.turn_token})")
            wrong = [p for p in span if rows_d[p]["target"] != int(targets[p - first])]
            if wrong:
                raise HarnessError(f"chunk {c}: the engine's target token at position {wrong[0]} is {rows_d[wrong[0]]['target']}, the file's is "
                                   f"{int(targets[wrong[0] - first])}: the two did not read the same tokens")
            i0 = plan.split_at - first
            sub_t, sub_s, sub_m = targets[i0:], scale[i0:], mlp[i0:]
            order = list(rows_d)
            if run_info.get("logits"):
                lg = read_logits(run_info["logits"])
                if lg.n_rows != len(order):
                    raise HarnessError(f"{run_info['logits']}: {lg.n_rows} rows for {len(order)} log-probability lines")
                if lg.vocab != V:
                    if lg.vocab > V and a.kld_trim_vocab:
                        notes.append(f"the engine's rows have {lg.vocab} columns, the file's vocabulary is {V}: the first {V} are compared")
                    else:
                        raise HarnessError(f"the engine writes {lg.vocab} logits per position, the llama.cpp file has n_vocab {V} (--kld-trim-vocab)")
                rows = lg.rows if order == span else lg.rows[[order.index(p) for p in span]]
                scorer.add_logits(rows[:, :V], sub_t, sub_s, sub_m, block[i0:, 4:4 + V])
                run_info["rows_kind"] = "logits"
                if not a.keep_logits:
                    Path(run_info["logits"]).unlink(missing_ok=True)
            else:
                lp = np.array([rows_d[p]["logprob"] for p in span])
                top = np.array([rows_d[p]["top"] for p in span])
                scorer.add_target(lp, top, sub_t, sub_s, sub_m, lambda i, b=block, o=i0: np.asarray(b[o + i, 4:4 + V]))
                run_info["rows_kind"] = "logprobs"
                scorer.nonfinite += bad
            run_info["rows_scored"] = len(span)
        done = scorer.summary()
        if done.get("count"):
            kl = f"mean KLD {done['kld_mean']:.3e}, " if done.get("has_kl") else ""
            print(f"[{label}] scored; so far {done['count']} rows: {kl}same top-1 {100 * done['same_top']:.2f}%, ln PPL change "
                  f"{done['ln_ppl_ratio']:+.5f}", flush=True)
        if c == 0 and not a.no_pin_expert_cache and flag_value(ctx.base_args, "--expert-cache") in ("auto", "-1"):
            slots = expert_cache_slots(ctx.workdir / f"{stem}.log")
            if slots:
                cand_base = set_flag(ctx.base_args, "--expert-cache", str(slots))
                notes.append(f"--expert-cache auto is pinned to chunk 0's {slots} slots for the other chunks, so every chunk runs with the same GPU "
                             "expert set (--no-pin-expert-cache to disable)")
    if source == "logpos" and any(sp > first for sp in splits):
        lost = sum(sp - first for sp in splits)
        notes.append(f"the split positions are {splits}: {lost} of {n_chunks * kb.n_scored} positions (those between n_ctx/2 and each split) went "
                     "through the batched path and are not scored")
    summary = scorer.summary()
    if not summary.get("count"):
        raise HarnessError("no row could be scored (every row had a non-finite logit?)")
    if source == "logpos" and not summary["has_kl"]:
        notes.append("KL divergence is NOT available: the engine wrote only the target token's log-probability (an engine without "
                     "STRATA_LOGITS_DUMP, or --logits-dump off); PPL, the paired change, same-top-1 and dp are")
    floor = K.parse_llama_log(Path(a.kld_floor).read_text(encoding="utf-8", errors="replace")) if a.kld_floor else None
    verdict_s, why, vinfo = kld_verdict(summary, floor, a)
    print(f"[done] {n_chunks} chunks in {time.time() - t_all:.0f} s", flush=True)
    return finish(a, ctx.workdir, kld_rep(a, ctx, kb, source, n_chunks, bos, notes, runs, th, summary, (floor, verdict_s, vinfo),
                                          1 if verdict_s == "FAIL" else 0, why))


def kld_rep(a, ctx: Ctx, kb, source: str, n_chunks: int, bos, notes: list[str], runs: dict, th: dict, summary, floor_verdict, exit_code: int,
            reasons: list[str]) -> dict:
    floor, verdict_s, vinfo = floor_verdict if floor_verdict else (None, "FAIL", {})
    return {"config": {"mode": "kld", "source": source, "ref_kld": str(a.ref_kld), "n_ctx": kb.n_ctx, "n_vocab": kb.n_vocab, "first": kb.first,
                       "chunks": n_chunks, "bos": bos, "kld_floor": a.kld_floor, "n_tokens": kb.n_ctx, "native_pack": ctx.native},
            "thresholds": th, "runs": runs, "metrics": None, "exact": None, "notes": notes, "reasons": reasons, "exit": exit_code,
            "pass": exit_code == 0,
            "kld": {"summary": summary or {"count": 0}, "floor": floor, "verdict": verdict_s, "verdict_info": vinfo}}


def summarize_report(path: str) -> int:
    """--summarize-report FILE: `STATUS<TAB>one line` for a report.json this tool wrote (scripts/logit_identity.sh print it per row).  STATUS is
    IDENTICAL / DIFFERENT for --exact, PASS / FAIL / MEASURED for --ref-kld, PASS / FAIL for the two-run metrics."""
    try:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"UNKNOWN\tno readable report: {e}")
        return 0
    e, k, m = rep.get("exact"), rep.get("kld"), rep.get("metrics")
    if e is not None:
        if e["identical"]:
            what = (f"bitwise identical: {e['rows']} rows of {e['values_per_row']} float32 logits" if e["kind"] == "logits" else
                    f"printed log-probabilities identical on {e['rows']} rows (9 decimals: NOT a bitwise statement, no STRATA_LOGITS_DUMP on a side)")
            print(f"IDENTICAL\t{what}")
        else:
            bits = [f"{e['rows_differing']} of {e['rows']} rows differ"]
            if e.get("first_diff_position") is not None:
                bits.append(f"first at position {e['first_diff_position']}")
            if e["kind"] == "logits":
                bits.append(f"{e['values_differing']} values, largest {e['max_ulp']} ULP (|d| {e['max_abs_diff']:.2e})")
                bits.append(f"top-1 changed in {e['top1_changed_rows']} rows")
            else:
                bits.append(f"largest |d log p| {e['max_abs_diff']:.2e}")
            print("DIFFERENT\t" + "; ".join(bits))
    elif k is not None:
        s, i = k["summary"], k.get("verdict_info", {})
        if not s.get("count"):
            print("FAIL\t" + "; ".join(rep.get("reasons") or ["no scored rows"]))
            return 0
        kl = f"mean KLD {s['kld_mean']:.3e}" + (f" (floor {i['floor_kld']:.3e}, allowed {i['kl_bar']:.3e})" if "floor_kld" in i else "") if s.get("has_kl") else "KL n/a"
        top = f"top-1 differs {100 * s['top1_mismatch']:.3f}%" + (f" (floor {100 * i['floor_top1_mismatch']:.3f}%, allowed {100 * i['top1_bar']:.3f}%)" if "top1_bar" in i else "")
        dn = f"ln PPL {s['dnll_mean']:+.5f} +- {s.get('dnll_se') or 0:.5f}" + (f" (allowed +-{i['dnll_bar']:.5f})" if "dnll_bar" in i else "")
        why = "; ".join(rep.get("reasons") or [])
        print(f"{k['verdict']}\t{s['count']} rows: {kl}; {top}; {dn}" + (" [partial: no KL]" if i.get("partial") else "") + (f" -- {why}" if why else ""))
    elif m is not None:
        bits = [f"top-1 {100 * m['top1']:.2f}%"] if "top1" in m else []
        if "ppl_rel" in m:
            bits.append(f"|dPPL|/PPL {100 * m['ppl_rel']:.2f}%")
        print(("PASS" if rep.get("pass") else "FAIL") + "\t" + ", ".join(bits + rep.get("reasons", [])))
    else:
        print("FAIL\t" + ("; ".join(rep.get("reasons") or []) or "no result in the report"))
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.summarize_report:
        return summarize_report(a.summarize_report)
    try:
        return print_config(a) if a.print_config else run(a)
    except (HarnessError, K.KldError) as e:
        print(f"golden_compare: {e}", file=sys.stderr)
        return 2


def make_plan(a, tokens: list[int], native: bool, notes: list[str], offline: bool) -> Plan:
    n = len(tokens)
    mode = a.mode
    if mode == "teacher" and a.max_new not in (0, 1):
        raise HarnessError("--max-new does not apply to teacher mode (it generates one token); use --mode greedy")
    source = a.teacher_source if a.teacher_source != "auto" else ("logpos" if native else "dump")
    if offline:
        source = "dump"                       # two files: nothing to run
    if native and mode == "teacher" and source == "dump":
        notes.append("the pack is NATIVE (IQ): the engine writes no logits for it (its token loop is skipped), so --teacher-source dump "
                     "is impossible; running greedy mode on token ids (use --teacher-source logpos for log-probabilities)")
        mode = "greedy"
    if mode == "greedy":
        if n <= SPARSE_FROM:
            notes.append(f"the prompt is {n} tokens (<= {SPARSE_FROM}): the QSA sparse selection is not exercised "
                         "(use --prompt-name long for the gate)")
        return Plan("greedy", n, a.max_new or 256, 1, None, native)
    if source == "logpos":
        if a.stride != 1:
            notes.append("--stride does not apply to the logpos source (every tail position is scored)")
        k0 = int(a.prefill_until) if a.prefill_until != "auto" else n - 1 - a.tail
        if k0 < 1:
            raise HarnessError("the logpos source needs a batched part: the prompt must be longer than --tail "
                               "(or give --prefill-until K >= 1)")
        plan = Plan("teacher", n, 1, 1, k0, native)
        plan.source = "logpos"
        if a.turn_token is not None:
            where = [i for i, t in enumerate(tokens) if t == a.turn_token and i >= 1]
            if not where or where[-1] >= n - 1:
                raise HarnessError("--turn-token must occur in the prompt, after position 0 and before its last token")
            plan.split_at, plan.turn_token = where[-1], a.turn_token
        else:
            plan.split_at, plan.turn_token = choose_split(tokens, k0)
        rows = n - 1 - plan.split_at
        if rows > 4096:
            notes.append(f"the tail is {rows} tokens: the engine's windows read it one window at a time, which is slow")
        if plan.split_at <= SPARSE_FROM:
            notes.append(f"only {plan.split_at} tokens are batched: up to {SPARSE_FROM} the QSA selection is the identity, so the sparse "
                         "top-k path is not exercised")
        return plan
    if a.prefill_until == "auto":
        k = n - a.tail if n - a.tail >= 256 else 0
        if k == 0:
            notes.append(f"the prompt ({n} tokens) is not much longer than --tail ({a.tail}): the WHOLE prompt runs through the "
                         "token path and the batched prefill (the Volta kernels) is NOT exercised.  Use --prompt-name long.")
    else:
        k = int(a.prefill_until)
        if k == 0:
            notes.append("--prefill-until 0: the whole prompt runs through the token path; the batched path is not exercised")
    if 0 < k <= SPARSE_FROM:
        notes.append(f"only {k} tokens are batched: up to {SPARSE_FROM} the QSA selection is the identity (every cell is "
                     "attended), so the sparse top-k path is not exercised")
    return Plan("teacher", n, 1, max(1, a.stride), k, native)


def want_logits(a) -> bool:
    """Ask a serve-mode engine for STRATA_LOGITS_DUMP (the whole rows, ~1 MB per scored position): for --exact and --ref-kld, where the
    bits / the KL need them; not for the Gate Q comparison, which has always been log-probabilities only."""
    return a.logits_dump == "on" or (a.logits_dump == "auto" and bool(a.exact or a.ref_kld))


def adopt_reference(a, plan: Plan, workdir: Path, tokens_file: Path, notes: list[str]) -> dict:
    """--reuse-ref DIR: the reference run of an earlier workdir (same prompt, same plan) instead of running it again."""
    src = Path(absolute(a.reuse_ref))
    if not src.is_dir():
        raise HarnessError(f"--reuse-ref {src}: not a directory (the --workdir of an earlier golden_compare run)")
    old_tokens = src / "tokens.ids"
    if not old_tokens.exists() or old_tokens.read_text(encoding="utf-8") != tokens_file.read_text(encoding="utf-8"):
        raise HarnessError(f"--reuse-ref {src}: that run used a different prompt (its tokens.ids differs from this one's)")
    meta = {}
    if (src / "report.json").exists():
        meta = json.loads((src / "report.json").read_text(encoding="utf-8"))
        c = meta.get("config", {})
        for k, v in (("mode", plan.mode), ("prefill_until", plan.prefill_until), ("source", plan.source if plan.mode == "teacher" else None),
                     ("split_at", plan.split_at if plan.source == "logpos" else None)):
            if v is not None and k in c and c[k] != v:
                raise HarnessError(f"--reuse-ref {src}: its plan has {k} {c[k]!r}, this run's is {v!r}: not the same comparison")
    logpos_src = plan.mode == "teacher" and plan.source == "logpos"
    need = ["ref.log"] + (["ref.logpos"] if logpos_src else ["ref.bin"])
    for nm in need + (["ref.logits"] if logpos_src else []):
        f = src / nm
        if not f.exists():
            if nm == "ref.logits":
                continue
            raise HarnessError(f"--reuse-ref {src}: it has no {nm}")
        dst = workdir / nm
        if dst.resolve() == f.resolve():
            continue
        dst.unlink(missing_ok=True)
        try:
            os.link(f, dst)
        except OSError:
            shutil.copy2(f, dst)
    log = workdir / "ref.log"
    info = {"kind": "engine", **parse_engine_log(log.read_text(errors="replace")), "returncode": 0, "seconds": 0, "log": str(log),
            "env": (meta.get("runs", {}).get("reference", {}) or {}).get("env", {}), "reused_from": str(src)}
    if logpos_src:
        info["logpos"] = str(workdir / "ref.logpos")
        info["batched_to"] = plan.split_at
        if (workdir / "ref.logits").exists():
            info["logits"] = str(workdir / "ref.logits")
    notes.append(f"the reference run is the one of {src} (--reuse-ref): not run again")
    print(f"[reference] reusing the run of {src}", flush=True)
    return info


def exe_has_option(exe: list[str], option: str) -> bool:
    """Whether the engine binary (the last path-like element of `exe`) contains the text of `option` - how setup.py tells an
    engine compiled from this tree (it has --numa) from upstream's (it does not).  True when the binary cannot be read (a
    wrapper script, a fake engine in the tests): then nothing is removed and the engine itself reports what it rejects."""
    for part in reversed(exe):
        path = Path(part)
        if path.is_file() and path.stat().st_size > 1 << 20:
            try:
                return option.encode() in path.read_bytes()
            except OSError:
                return True
    return True


def drop_unsupported_numa(args: list[str], exe: list[str]) -> list[str]:
    """`--numa X` taken out of the arguments for an engine that has no such option (upstream Strata, row 1a of
    logit_identity.sh): it would refuse to start on the config's saved `--numa`, and it has one copy of the arena by first
    touch anyway - the same as the port's `--numa off`."""
    if not any(x == "--numa" or x.startswith("--numa=") for x in args) or exe_has_option(exe, "--numa"):
        return args
    out, skip = [], False
    for x in args:
        if skip:
            skip = False
            continue
        if x == "--numa":
            skip = True
            continue
        if not x.startswith("--numa="):
            out.append(x)
    print("[engine] this engine has no --numa option (upstream): the config's --numa is left out", flush=True)
    return out


def obtain_run(label: str, a, plan: Plan, tokens: list[int], base_args: list[str], exe: list[str], base_env: dict, cwd, workdir: Path,
               env_extra: dict, extra_args: str, tokens_file: Path) -> dict:
    base_args = drop_unsupported_numa(list(base_args), exe)
    extra_args = shlex.join(drop_unsupported_numa(shlex.split(extra_args), exe))
    if plan.mode == "teacher" and plan.source == "logpos":
        short = "ref" if label == "reference" else "cand"
        log = workdir / f"{short}.log"
        logpos = workdir / f"{short}.logpos"
        if a.reuse and logpos.exists() and log.exists():
            print(f"[{label}] reusing {logpos}", flush=True)
            info = {"kind": "engine", "returncode": 0, "seconds": 0, "env": env_extra, "log": str(log), "logpos": str(logpos)}
            if (workdir / f"{short}.logits").exists():
                info["logits"] = str(workdir / f"{short}.logits")
            return info
        args = build_serve_args(base_args, plan, fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec,
                                extra=shlex.split(extra_args))
        info = run_serve_teacher(label, exe, args, base_env, env_extra, cwd, workdir, tokens, a.timeout or None, logits=want_logits(a))
        info["batched_to"] = plan.split_at
        return info
    dump = workdir / ("ref.bin" if label == "reference" else "cand.bin")
    log = workdir / ("ref.log" if label == "reference" else "cand.log")
    args = build_engine_args(base_args, plan, tokens_file=str(tokens_file), dump=str(dump), fixed_experts=not a.no_fixed_experts,
                             keep_spec=a.keep_spec, stats=a.stats, extra=shlex.split(extra_args))
    if a.reuse and dump.exists() and log.exists():
        print(f"[{label}] reusing {dump}", flush=True)
        return {"kind": "engine", **parse_engine_log(log.read_text(errors="replace")), "returncode": 0, "seconds": 0, "env": env_extra,
                "log": str(log)}
    return run_engine(label, exe, args, base_env, env_extra, cwd, log, a.timeout or None)


def rows_from_logits(lg: Logits, tokens: list[int], positions: range) -> tuple[dict[int, dict], int]:
    """logpos-style rows for the positions of `positions` that a logits file covers (so a reference made elsewhere can be
    compared with a candidate that only has log-probabilities)."""
    rows: dict[int, dict] = {}
    bad = 0
    for p in positions:
        try:
            r = lg.row_of(p)
        except KeyError:
            continue
        x = np.asarray(lg.rows[r], dtype=np.float64)
        if not np.isfinite(x).all():
            bad += int((~np.isfinite(x)).sum())
            continue
        lsm = log_softmax(x)
        top = int(x.argmax())
        rows[p] = {"target": tokens[p + 1], "logprob": float(lsm[tokens[p + 1]]), "top": top, "top_logprob": float(lsm[top])}
    return rows, bad


def place_engine_logits(path: Path, plan: Plan, partial_ok: bool = False) -> Logits:
    lg = read_logits(path)
    if lg.n_rows == 0:
        return lg.place([])      # a header and nothing else: the engine ran a path that writes no rows (native pack)
    if partial_ok:               # speculative decoding kept: the token loop stops after the first generated token
        return lg.place(plan.pos_start + np.arange(lg.n_rows))
    return lg.place(selected_positions(plan.total, plan.stride, plan.pos_start))


def place_external_logits(path: str, plan: Plan, first: int | None) -> Logits:
    lg = read_logits(path)
    if first is None:
        first = max(0, plan.total - lg.n_rows)      # an external file ends at the last position
    return lg.place(np.arange(first, first + lg.n_rows) if plan.stride == 1 else selected_positions(plan.total, plan.stride, first))


def absolute(path: str | Path | None, base: str | Path | None = None) -> str | None:
    """`path` as an absolute path: relative paths are taken from `base` (default: the directory this harness was started in).  No
    symlink resolution, and the file need not exist."""
    if path is None or str(path) == "":
        return None
    p = Path(path).expanduser()
    return str(p) if p.is_absolute() else os.path.normpath(str(Path(base or Path.cwd()) / p))


@dataclass
class Ctx:
    """What both flows (two runs / --ref-kld) need before they differ: the engine's arguments, where it runs, the workdir."""
    cfg: dict
    base_args: list[str]
    cwd: str | None
    exe: list[str]
    ref_exe: list[str]
    cand_exe: list[str]
    workdir: Path
    pack_dir: Path
    tok_dir: str | None
    native: bool
    base_env: dict


def engine_command(path: str | None) -> list[str]:
    """An engine binary (or a .py stand-in, run with this interpreter) as a command prefix; absolute against where we were started."""
    p = absolute(path)
    return ([sys.executable, p] if p.endswith(".py") else [p]) if p else []


def engine_context(a, offline: bool = False, make_workdir: bool = True) -> Ctx:
    cfg = load_engine_config(a.engine_config) if a.engine_config else {}
    base_args = list(cfg.get("args", [])) + shlex.split(a.engine_args)
    # The engine runs in `cwd` (the config's, normally the repository root), but every path WE hand to it - the tokens file, the logits
    # dumps, STRATA_LOGPOS, the engine binary, --pack - must name the same file wherever the harness was started: made absolute here.
    cwd = absolute(a.cwd) or cfg.get("cwd") or None
    exe_s = absolute(a.exe) if a.exe else (absolute(cfg["exe"], cwd) if cfg.get("exe") else None)
    if not exe_s and not offline and not (a.ref_exe and a.cand_exe):
        raise HarnessError("which engine?  --engine-config strata-*.json, or --exe PATH [--engine-args ...]")
    exe = engine_command(exe_s)
    workdir = Path(absolute(a.workdir) if a.workdir else Path.cwd() / "golden_out" / time.strftime("%Y%m%d-%H%M%S"))
    if make_workdir:
        workdir.mkdir(parents=True, exist_ok=True)

    if a.pack:      # an explicit --pack is the engine's too (it was only used to detect a native pack before)
        base_args = set_flag(base_args, "--pack", absolute(a.pack))
    pack = flag_value(base_args, "--pack") or "pack/full"
    pack_dir = Path(absolute(pack, cwd))
    tok_dir = absolute(a.tokenizer) or cfg.get("tokenizer") or (str(pack_dir / "tokenizer") if (pack_dir / "tokenizer" / "vocab.json").exists() else None)
    if tok_dir:
        tok_dir = absolute(tok_dir, cwd)
    native = a.native_pack == "yes" or (a.native_pack == "auto" and (pack_dir / "native_experts.txt").exists())
    return Ctx(cfg, base_args, cwd, exe, engine_command(a.ref_exe) or exe, engine_command(a.cand_exe) or exe, workdir, pack_dir, tok_dir,
               native, base_environment(cfg, a.gpu))


def run(a) -> int:
    if a.ref_kld:
        return run_kld(a)
    offline = bool(a.cand_logits and a.ref_logits)
    ctx = engine_context(a, offline)
    cfg, base_args, cwd, exe, workdir, tok_dir = ctx.cfg, ctx.base_args, ctx.cwd, ctx.exe, ctx.workdir, ctx.tok_dir
    tokens = get_tokens(a, tok_dir, workdir)
    n = len(tokens)
    if n < 2:
        raise HarnessError("the prompt has fewer than 2 tokens")
    tokens_file = workdir / "tokens.ids"
    tokens_file.write_text(" ".join(map(str, tokens)), encoding="utf-8")

    notes: list[str] = []
    native, base_env = ctx.native, ctx.base_env
    plan = make_plan(a, tokens, native, notes, offline)
    cand_env, ref_env = parse_env_pairs(a.cand_env), parse_env_pairs(a.ref_env)
    if a.exact and plan.mode != "teacher":
        raise HarnessError("--exact compares logits, which needs teacher mode (the default).  A native (IQ) pack runs teacher mode through "
                           "--teacher-source logpos (the default for it).")
    if a.dry_run:
        return dry_run(a, plan, tokens, base_args, exe, cand_env, ref_env, notes, tokens_file, workdir, ref_exe=ctx.ref_exe,
                       cand_exe=ctx.cand_exe)
    runs: dict = {}
    th = {"min_top1": a.min_top1, "max_ppl_rel": a.max_ppl_rel, "max_kl": a.max_kl, "max_abs_dlogit": a.max_abs_dlogit}
    ref_desc = (f"logits file {a.ref_logits}" if a.ref_logits else
                (f"the run of {a.reuse_ref}" if a.reuse_ref else
                 f"the same engine with {a.ref_env or '(the candidate environment)'}" if ctx.ref_exe == ctx.cand_exe else
                 f"{ctx.ref_exe[-1]} with {a.ref_env or '(the base environment)'}"))

    def rep(exit_code: int, reasons: list[str], metrics=None, greedy=None, cfg_extra=None, exact=None) -> dict:
        return {"config": {"mode": plan.mode, "n_tokens": n, "reference": ref_desc, "prefill_until": plan.prefill_until,
                           "tail": a.tail, "max_new": plan.max_new, "stride": plan.stride, "native_pack": native,
                           "pos_start": plan.pos_start, "tokens_file": str(tokens_file), **(cfg_extra or {})},
                "thresholds": th, "runs": runs, "metrics": metrics, "greedy": greedy, "exact": exact, "notes": notes,
                "reasons": reasons, "pass": exit_code == 0, "exit": exit_code}

    # ---- the two sides
    if a.ref_logits:
        runs["reference"] = {"kind": "file", "path": a.ref_logits}
    elif a.reuse_ref:
        runs["reference"] = adopt_reference(a, plan, workdir, tokens_file, notes)
    else:
        runs["reference"] = obtain_run("reference", a, plan, tokens, base_args, ctx.ref_exe, base_env, cwd, workdir, ref_env, a.ref_args,
                                       tokens_file)
        if runs["reference"]["returncode"] != 0:
            return finish(a, workdir, rep(2, ["the REFERENCE engine run failed: there is nothing to compare against"]))
    cand_base = base_args
    if (not a.no_pin_expert_cache and runs["reference"]["kind"] == "engine" and runs["reference"].get("log")
            and flag_value(base_args, "--expert-cache") in ("auto", "-1")):
        slots = expert_cache_slots(runs["reference"]["log"])
        if slots:
            cand_base = set_flag(base_args, "--expert-cache", str(slots))
            notes.append(f"the candidate's --expert-cache auto is pinned to the reference's {slots} slots, so both runs put the same "
                         "experts on the GPU (--no-pin-expert-cache to disable)")
    if a.cand_logits:
        runs["candidate"] = {"kind": "file", "path": a.cand_logits}
    else:
        runs["candidate"] = obtain_run("candidate", a, plan, tokens, cand_base, ctx.cand_exe, base_env, cwd, workdir, cand_env, a.cand_args,
                                       tokens_file)
        if runs["candidate"]["returncode"] != 0:
            return finish(a, workdir, rep(1, [f"the CANDIDATE engine run failed (exit {runs['candidate']['returncode']}, see "
                                              f"{workdir / 'cand.log'}); a BPT.TRAP / launch failure on the V100 looks like this"]))
    if ctx.ref_exe != ctx.cand_exe:       # two different binaries: say which (the report shows it; the determinism note below keys on it)
        if runs["reference"]["kind"] == "engine":
            runs["reference"]["exe"] = ctx.ref_exe[-1]
        if runs["candidate"]["kind"] == "engine":
            runs["candidate"]["exe"] = ctx.cand_exe[-1]
    reasons: list[str] = []
    greedy = None
    cfg_extra: dict = {}
    if plan.mode == "teacher" and plan.source == "logpos":
        if a.exact:
            exact, cfg_extra = exact_logpos_runs(a, plan, tokens, runs, notes)
            return finish(a, workdir, rep(0 if exact["identical"] else 1, exact_reasons(exact), None, None, cfg_extra, exact))
        metrics, cfg_extra = compare_logpos_runs(a, plan, tokens, runs, notes, workdir)
        ok, why = verdict(metrics, min_top1=a.min_top1, max_ppl_rel=a.max_ppl_rel, max_kl=a.max_kl,
                          max_abs_dlogit=a.max_abs_dlogit)
        return finish(a, workdir, rep(0 if not why else 1, why, metrics, None, cfg_extra))
    partial = plan.mode == "greedy" and (a.keep_spec or plan.native)
    ref = (place_external_logits(a.ref_logits, plan, a.ref_first_position) if a.ref_logits
           else place_engine_logits(workdir / "ref.bin", plan, partial))
    cand = (place_external_logits(a.cand_logits, plan, a.cand_first_position) if a.cand_logits
            else place_engine_logits(workdir / "cand.bin", plan, partial))

    if plan.mode == "greedy":
        metrics, greedy = compare_greedy(runs, tokens, ref, cand, plan, notes, reasons)
    else:
        if ref.n_rows == 0 or cand.n_rows == 0:
            raise HarnessError(f"the {'candidate' if cand.n_rows == 0 else 'reference'} logits file has a header but no rows: the engine "
                               "ran a path that writes none (a native pack? speculative decoding kept with --keep-spec?)")
        common = np.intersect1d(ref.positions, cand.positions)
        if len(common) == 0:
            raise HarnessError("the two logits files share no positions: wrong --ref-first-position / --cand-first-position?")
        cfg_extra = {"first_pos": int(common.min()), "last_pos": int(common.max())}
        if a.exact:
            ri, ci = [ref.row_of(int(q)) for q in common], [cand.row_of(int(q)) for q in common]
            exact = exact_compare_rows(lambda lo, hi: ref.rows[ri[lo:hi]], lambda lo, hi: cand.rows[ci[lo:hi]], [int(q) for q in common])
            if runs["reference"]["kind"] == runs["candidate"]["kind"] == "engine":
                same_config(a, runs, notes)
            return finish(a, workdir, rep(0 if exact["identical"] else 1, exact_reasons(exact), None, None, cfg_extra, exact))
        metrics = compare_rows(ref, cand, common, {int(p): tokens[int(p) + 1] for p in common if int(p) + 1 < n})
    ok, why = verdict(metrics, min_top1=a.min_top1, max_ppl_rel=a.max_ppl_rel, max_kl=a.max_kl,
                      max_abs_dlogit=a.max_abs_dlogit, need_ppl=not metrics.get("tokens_only"))
    reasons += why
    if plan.mode == "teacher" and metrics.get("rows_scored", 0) < 100:
        notes.append(f"only {metrics.get('rows_scored', 0)} rows scored: the 99% top-1 gate says little below ~100 rows")
    if runs["reference"]["kind"] == runs["candidate"]["kind"] == "engine":
        same_config(a, runs, notes)
    return finish(a, workdir, rep(0 if not reasons else 1, reasons, metrics, greedy, cfg_extra))


def same_config(a, runs: dict, notes: list[str]) -> None:
    """The note a determinism check deserves: both sides are the same binary in the same environment with the same extra arguments."""
    if (runs["reference"]["env"] == runs["candidate"]["env"] and a.ref_args == a.cand_args
            and runs["reference"].get("exe") == runs["candidate"].get("exe")):
        notes.append("candidate and reference run the SAME configuration: this is a determinism check, not a Volta check")


def exact_reasons(e: dict) -> list[str]:
    if e["identical"]:
        return []
    why = [f"{e['rows_differing']} of {e['rows']} rows differ"
           + (f"; first at position {e['first_diff_position']}" if e.get("first_diff_position") is not None else "")]
    if e.get("max_ulp"):
        why.append(f"the largest difference is {e['max_ulp']} ULP")
    if e.get("only_reference") or e.get("only_candidate"):
        why.append(f"positions only in the reference: {e['only_reference']}, only in the candidate: {e['only_candidate']}")
    return why


def dry_run(a, plan: Plan, tokens: list[int], base_args: list[str], exe: list[str], cand_env: dict, ref_env: dict, notes: list[str],
            tokens_file: Path, workdir: Path, ref_exe: list[str] | None = None, cand_exe: list[str] | None = None) -> int:
    print(f"golden_compare (dry run): mode {plan.mode}, {plan.n_tokens} prompt tokens, workdir {workdir}")
    if plan.mode == "teacher":
        what = (f"source logpos: the first {plan.split_at} tokens batched, split at token id {plan.turn_token}, "
                f"{plan.n_tokens - 1 - plan.split_at} tail positions scored through the verify windows"
                if plan.source == "logpos" else
                f"source dump: --prefill-until {plan.prefill_until}, the last {plan.n_tokens - (plan.prefill_until or 0)} positions "
                "written by the token loop")
        print("  " + what)
    for n in notes:
        print(f"  note: {n}")
    for label, env, extra in (("reference", ref_env, a.ref_args), ("candidate", cand_env, a.cand_args)):
        exe_side = (ref_exe if label == "reference" else cand_exe) or exe
        if label == "reference" and a.reuse_ref:
            print(f"\n[reference] (not run: --reuse-ref {a.reuse_ref})")
            continue
        if plan.mode == "teacher" and plan.source == "logpos":
            args = build_serve_args(base_args, plan, fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec, extra=shlex.split(extra))
            stem = "ref" if label == "reference" else "cand"
            env = dict(env, STRATA_LOGPOS=str(workdir / f"{stem}.logpos"))
            if want_logits(a):
                env["STRATA_LOGITS_DUMP"] = str(workdir / f"{stem}.logits")
        else:
            args = build_engine_args(base_args, plan, tokens_file=str(tokens_file),
                                     dump=str(workdir / ("ref.bin" if label == "reference" else "cand.bin")),
                                     fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec, stats=a.stats, extra=shlex.split(extra))
        print(f"\n[{label}] " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + (" " if env else "")
              + " ".join(shlex.quote(c) for c in [*exe_side, *args]))
    return 0


def compare_logpos_runs(a, plan: Plan, tokens: list[int], runs: dict, notes: list[str], workdir: Path) -> tuple[dict, dict]:
    n = len(tokens)
    span = range(plan.split_at, n - 1)          # the positions whose log-probs the engine writes (targets split_at+1 .. n-1)

    def side(label: str, ext: str | None, first: int | None) -> tuple[dict, int]:
        if ext:
            lg = place_external_logits(ext, Plan("teacher", n, 1, 1, None, False), first)
            return rows_from_logits(lg, tokens, span)
        return read_logpos(Path(runs[label]["logpos"]))

    ref, bad_r = side("reference", a.ref_logits, a.ref_first_position)
    cand, bad_c = side("candidate", a.cand_logits, a.cand_first_position)
    rows_expected = len(span)
    for name, rows in (("reference", ref), ("candidate", cand)):
        if len(rows) < rows_expected and not (name == "reference" and a.ref_logits) and not (name == "candidate" and a.cand_logits):
            raise HarnessError(f"the {name} wrote {len(rows)} log-probability rows, expected {rows_expected} (positions "
                               f"{span.start}..{span.stop - 1}): did the tail go through the verify windows?  Check --short-read / "
                               f"--turn-token in {runs[name].get('log')} (the engine splits the prompt at the LAST occurrence of the turn token)")
    m = compare_logpos(ref, cand, bad_r, bad_c)
    cfg_extra = {"first_pos": plan.split_at, "last_pos": n - 2, "split_at": plan.split_at, "turn_token": plan.turn_token,
                 "source": "logpos"}
    if (not (a.ref_logits or a.cand_logits) and runs["reference"].get("env") == runs["candidate"].get("env") and a.ref_args == a.cand_args
            and runs["reference"].get("exe") == runs["candidate"].get("exe")):
        notes.append("candidate and reference run the SAME configuration: this is a determinism check, not a Volta check")
    if m.get("rows_scored", 0) < 100:
        notes.append(f"only {m.get('rows_scored', 0)} rows scored: the 99% top-1 gate says little below ~100 rows")
    return m, cfg_extra


def exact_logpos_runs(a, plan: Plan, tokens: list[int], runs: dict, notes: list[str]) -> tuple[dict, dict]:
    """--exact on the logpos source: the logits rows bit for bit when BOTH engines wrote STRATA_LOGITS_DUMP, and always the printed
    log-probabilities.  Rows of the dump belong to the logpos lines, one for one, in file order."""
    if a.ref_logits or a.cand_logits:
        raise HarnessError("--exact on the logpos source compares two engine runs: a logits file made elsewhere has no log-probability "
                           "lines to match (use two --ref-logits / --cand-logits dump files, or two engine runs)")
    n = len(tokens)
    ref, bad_r = read_logpos(Path(runs["reference"]["logpos"]))
    cand, bad_c = read_logpos(Path(runs["candidate"]["logpos"]))
    for name, rows in (("reference", ref), ("candidate", cand)):
        if not rows:
            raise HarnessError(f"the {name} wrote no log-probability rows ({runs[name].get('logpos')}): see {runs[name].get('log')}")
    lp = exact_compare_logpos(ref, cand)
    out = dict(lp)
    r_lg, c_lg = runs["reference"].get("logits"), runs["candidate"].get("logits")
    if r_lg and c_lg:
        rl, cl = read_logits(r_lg), read_logits(c_lg)
        for name, lg, rows in (("reference", rl, ref), ("candidate", cl, cand)):
            if lg.n_rows != len(rows):
                raise HarnessError(f"the {name}'s STRATA_LOGITS_DUMP has {lg.n_rows} rows but its log-probability file has {len(rows)} lines")
        pr, pc = list(ref), list(cand)
        ir, ic = {p: i for i, p in enumerate(pr)}, {p: i for i, p in enumerate(pc)}
        common = [p for p in pr if p in ic]
        ri, ci = [ir[p] for p in common], [ic[p] for p in common]
        if rl.vocab != cl.vocab:
            raise HarnessError(f"the two runs' vocabularies differ ({rl.vocab} against {cl.vocab})")
        out = exact_compare_rows(lambda lo, hi: rl.rows[ri[lo:hi]], lambda lo, hi: cl.rows[ci[lo:hi]], common)
        out["logpos_rows_differing"] = lp["rows_differing"]
        out["only_reference"], out["only_candidate"] = lp["only_reference"], lp["only_candidate"]
        out["identical"] = out["identical"] and lp["identical"]
    else:
        missing = [s for s, v in (("reference", r_lg), ("candidate", c_lg)) if not v]
        notes.append("no full logits from the " + " or ".join(missing) + " (an engine without STRATA_LOGITS_DUMP: verify.cpp of this port "
                     "has it, upstream's has not - tools/volta/upstream_logits_dump.patch adds it): the comparison is of the printed "
                     "log-probabilities (9 decimals), not of the bits of the logits")
    span = sorted(set(ref) & set(cand))
    cfg_extra = {"first_pos": span[0] if span else None, "last_pos": span[-1] if span else None, "split_at": plan.split_at,
                 "turn_token": plan.turn_token, "source": "logpos"}
    same_config(a, runs, notes)
    return out, cfg_extra


def compare_greedy(runs: dict, tokens: list[int], ref: Logits, cand: Logits, plan: Plan, notes: list[str], reasons: list[str]):
    """Greedy mode: prefix / agreement from the generated ids, logit metrics on the rows whose contexts are identical."""
    n = len(tokens)
    r_out, c_out = runs["reference"].get("output"), runs["candidate"].get("output")
    if r_out is None or c_out is None:
        raise HarnessError("greedy mode needs both runs' 'output :' lines (engine runs, not --ref-logits / --cand-logits)")
    L = identical_prefix(c_out, r_out)
    nmax = max(len(c_out), len(r_out))
    same_len = len(c_out) == len(r_out)
    first_diff = None if (L == nmax) else L
    g = {"n_cand": len(c_out), "n_ref": len(r_out), "n_max": nmax, "prefix": L,
         "token_agreement": sum(1 for x, y in zip(c_out, r_out) if x == y) / nmax if nmax else 0.0,
         "first_diff": first_diff, "cand_at_diff": c_out[L] if L < len(c_out) else None,
         "ref_at_diff": r_out[L] if L < len(r_out) else None, "cand_tokens": c_out, "ref_tokens": r_out, "same_length": same_len}
    if not c_out:
        reasons.append("the candidate generated no tokens")
    if ref.n_rows and cand.n_rows:
        # every prompt-position row, and the generation rows j <= L (the same context in both runs; row L is where they differ)
        last = n - 1 + min(L, max(len(r_out), 1) - 1)
        common = np.intersect1d(ref.positions, cand.positions)
        common = common[common <= last]
        targets = {}
        for p in common:
            p = int(p)
            if p < n - 1:
                targets[p] = tokens[p + 1]
            elif p - (n - 1) < len(r_out):
                targets[p] = r_out[p - (n - 1)]       # the reference's own pick
        return compare_rows(ref, cand, common, targets), g
    rows = min(L + 1, nmax) if nmax else 0
    agree = min(L, rows)
    notes.append("no logit rows were written (native pack / speculative loop): the metrics are on token ids only - NaN scan, "
                 "perplexity and KL are not available")
    return {"rows_compared": rows, "rows_scored": rows, "top1_agree": agree, "top1": agree / rows if rows else 0.0,
            "nonfinite_cand": 0, "nonfinite_ref": 0, "tokens_only": True}, g


if __name__ == "__main__":
    sys.exit(main())
