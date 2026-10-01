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

PASS needs top-1 >= 0.99, |dPPL| / PPL <= 0.02 and no NaN/inf (all overridable).  The plan's Gate 2 is "top-1 >= 99 %
over 500 tokens, PPL within 1-2 %, no NaN/inf": this is that, with the 500 tokens being `--tail`.

THE REFERENCE is, by default, a second run of the same engine binary with
    STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0
i.e. the FP32 CUDA-core prompt attention (the decode kernel, what upstream falls back to) and the cuBLAS bf16 GEMMs
(upstream's own, on the CUDA cores of a V100).  Other references: `--ref-env "STRATA_PROMPT_ATTN_OLD=1 ..."` (upstream's
own switch for the same FP32 attention), `--ref-args "..."` (extra engine flags for the reference run only), or
`--ref-logits FILE` - logits produced elsewhere (another GPU running upstream, llama.cpp) in the format below.
NOT references for the Volta prefill kernels: `--no-fast-attn` / `--no-fast-select` (generate.cpp:1803-1806, layer.cpp:931-941) only
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
             and passes it as --turn-token.  Any pack.  Metrics: top-1 agreement, perplexity of the target tokens, the paired
             NLL delta, max |dlogp|, NaN / inf in the log-probs; NOT KL or max |dlogit| (the logits themselves are not written).
    auto     dump for a non-native pack, logpos for a native one.

  Either way the scored rows depend on the batched prefill of the tokens before them - which is the point.

KEEPING THE TWO RUNS COMPARABLE.  Candidate and reference run one after the other on the same card with the same arguments, apart
from `--ref-env` / `--ref-args`.  Unless `--no-fixed-experts` / `--no-pin-expert-cache` say otherwise the harness adds
`--pcie-frac 0 --adapt-swaps 0` (no PCIe share of the misses, no adaptive swaps: the GPU's expert set stays what the profile
chose, bench/results/2026-09-27-esp/esp_kl.py does the same) and pins the candidate's `--expert-cache auto` to the slot
count the reference reported - the GPU and the CPU round an expert differently, upstream measured 2-5 % of top-1 flips just
from the expert cache being on or off, which would drown the Volta kernels' own noise.  Speculative decoding is dropped for a
non-native pack (plain decode is what writes logits) and kept for a native one (it cannot run without).

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

LOGITS FILE FORMAT (what `--dump-logits` writes, and what `--ref-logits` reads): little-endian
    int32 n_vocab, int32 n_rows, then rows of n_vocab float32.
A `.npy` file of shape (rows, n_vocab) is accepted too (numpy.save of a float32 array).  The engine writes the rows of
the positions it processed itself; with `--prefill-until K` the file holds only positions >= K although the header
still counts all of them, so the reader goes by the file size, not the header.  An external file covers the LAST
`rows` positions of the sequence unless `--ref-first-position` says otherwise (a llama.cpp run over the whole text:
first position 0).  Producing one from numpy: `np.array([V, R], np.int32).tofile(f); logits.astype('<f4').tofile(f)`.

EXAMPLES
    # Gate 2 on a V100, the engine configured by setup.py:
    python3 tools/volta/golden_compare.py --engine-config strata-q2_0.json --prompt-name long --tail 512
    # the same, standalone:
    python3 tools/volta/golden_compare.py --exe build-sm70/strata --pack pack/full --ple-gguf ... \\
        --engine-args "--native /data/model-00001-of-00002.gguf --expert-cache 2000" --prompt-file big.txt
    # a native (IQ3) pack: log-probabilities of the last 512 tokens of a 33,000-token prompt, via the resident engine
    python3 tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --tail 512
    # the same pack, greedy: do 256 generated tokens agree?
    python3 tools/volta/golden_compare.py --engine-config strata-iq3_xxs.json --prompt-name long --mode greedy --max-new 256
    # compare two logits files made elsewhere:
    python3 tools/volta/golden_compare.py --prompt-file text.txt --cand-logits v100.bin --ref-logits llamacpp.npy

Exit status: 0 PASS, 1 FAIL (a threshold, a NaN, or the candidate engine failed), 2 the harness could not run (the
reference failed, a file is missing or malformed).  Needs only numpy (and tools/strata_tokenizer.py + `regex` to tokenize
text).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROMPT_DIR = HERE / "prompts"

DEFAULT_REF_ENV = "STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0"
CHAT_WRAP = "<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # tools/calibrate.py:chat_ids
SPARSE_FROM = 2051     # qsa_selection_width: idx_top_k 2048 + idx_block 4 - 1.  Up to it the selection is the identity.
LONG_TOKENS = 33000    # the `long` prompt: past 2,051 and past four 8,192-token prefill chunks


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
        raise HarnessError(f"cannot import tools/strata_tokenizer.py ({e}); pip install regex, or pass --ids-file") from e
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


def build_serve_args(base: list[str], plan: Plan, *, fixed_experts: bool, keep_spec: bool, extra: list[str] | None = None) -> list[str]:
    """`strata --serve ...` for the logpos source: the configured arguments, the split token and the window length."""
    args = list(base)
    for f in ("--serve", "--greedy", "--stats", "--spec-split"):
        args = drop_flag(args, f, takes_value=False)
    for f in OWN_FLAGS + ["--turn-token", "--short-read"]:
        args = drop_flag(args, f)
    cfg_ctx = flag_value(base, "--max-context")
    plan.max_context = (max(int(cfg_ctx) if cfg_ctx and cfg_ctx.isdigit() else 0, plan.n_tokens + plan.max_new + 8) + 1023) // 1024 * 1024
    if not plan.native and not keep_spec:
        for f in SPEC_FLAGS:
            args = drop_flag(args, f)
    if flag_value(args, "--prefill") in (None, "0"):
        args = set_flag(args, "--prefill", "auto")
    if fixed_experts:
        args = set_flag(args, "--pcie-frac", "0")
        args = set_flag(args, "--adapt-swaps", "0")
    rows = plan.n_tokens - 1 - plan.split_at
    args += ["--max-context", str(plan.max_context), "--turn-token", str(plan.turn_token), "--short-read", str(rows + 8)]
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
                      workdir: Path, tokens: list[int], timeout: float | None) -> dict:
    """Start `strata --serve`, send one `GEN 1 <ids>`, wait for DONE, QUIT.  STRATA_LOGPOS goes to <label>.logpos."""
    short = "ref" if label == "reference" else "cand"
    logpos = workdir / f"{short}.logpos"
    logpos.unlink(missing_ok=True)
    env = dict(base_env)
    env.update(env_extra)
    env["STRATA_LOGPOS"] = str(logpos)
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
    print(f"[{label}] exit {info['returncode']} in {info.get('seconds', 0):.0f} s", flush=True)
    return info


# ====================================================================================================== the report


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def format_report(rep: dict) -> str:
    c, m, g, th = rep["config"], rep.get("metrics"), rep.get("greedy"), rep["thresholds"]
    L = ["=" * 100, f"golden_compare: mode {c['mode']}, {c['n_tokens']} prompt tokens; reference: {c.get('reference', '?')}"]
    L += [f"  note: {x}" for x in rep["notes"]]
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
            L.append(f"  {k:<9} {r['seconds']:.0f} s, exit {r['returncode']}{sp}; env: {env}")
        else:
            L.append(f"  {k:<9} logits file {r['path']}")
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
    return "\n".join(L)


def finish(a, workdir: Path, rep: dict) -> int:
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
    g.add_argument("--exe", help="the engine binary (instead of / overriding the config's)")
    g.add_argument("--engine-args", default="", help="engine arguments, one shell-quoted string (added to the config's)")
    g.add_argument("--pack", help="the pack directory, if the engine arguments do not say (used to detect a native pack)")
    g.add_argument("--cwd", help="working directory of the engine (default: the config's, else the current one)")
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
    r.add_argument("--workdir", help="where the logits and logs go (default ./golden_out/<time>)")
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
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the two engine commands, run nothing (the runs take minutes each)")
    ap.add_argument("--json", help="write the full report as JSON (default <workdir>/report.json)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return run(a)
    except HarnessError as e:
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


def obtain_run(label: str, a, plan: Plan, tokens: list[int], base_args: list[str], exe: list[str], base_env: dict, cwd, workdir: Path,
               env_extra: dict, extra_args: str, tokens_file: Path) -> dict:
    if plan.mode == "teacher" and plan.source == "logpos":
        short = "ref" if label == "reference" else "cand"
        log = workdir / f"{short}.log"
        logpos = workdir / f"{short}.logpos"
        if a.reuse and logpos.exists() and log.exists():
            print(f"[{label}] reusing {logpos}", flush=True)
            return {"kind": "engine", "returncode": 0, "seconds": 0, "env": env_extra, "log": str(log), "logpos": str(logpos)}
        args = build_serve_args(base_args, plan, fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec,
                                extra=shlex.split(extra_args))
        info = run_serve_teacher(label, exe, args, base_env, env_extra, cwd, workdir, tokens, a.timeout or None)
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


def run(a) -> int:
    cfg = load_engine_config(a.engine_config) if a.engine_config else {}
    base_args = list(cfg.get("args", [])) + shlex.split(a.engine_args)
    exe_s = a.exe or cfg.get("exe")
    offline = bool(a.cand_logits and a.ref_logits)
    if not exe_s and not offline:
        raise HarnessError("which engine?  --engine-config strata-*.json, or --exe PATH [--engine-args ...]")
    exe = ([sys.executable, exe_s] if exe_s.endswith(".py") else [exe_s]) if exe_s else []
    cwd = a.cwd or cfg.get("cwd") or None
    workdir = Path(a.workdir) if a.workdir else Path("golden_out") / time.strftime("%Y%m%d-%H%M%S")
    workdir.mkdir(parents=True, exist_ok=True)

    pack = a.pack or flag_value(base_args, "--pack") or "pack/full"
    pack_dir = Path(cwd or ".") / pack
    tok_dir = a.tokenizer or cfg.get("tokenizer") or (str(pack_dir / "tokenizer") if (pack_dir / "tokenizer" / "vocab.json").exists() else None)
    tokens = get_tokens(a, tok_dir, workdir)
    n = len(tokens)
    if n < 2:
        raise HarnessError("the prompt has fewer than 2 tokens")
    tokens_file = workdir / "tokens.ids"
    tokens_file.write_text(" ".join(map(str, tokens)), encoding="utf-8")

    notes: list[str] = []
    native = a.native_pack == "yes" or (a.native_pack == "auto" and (pack_dir / "native_experts.txt").exists())
    plan = make_plan(a, tokens, native, notes, offline)
    cand_env, ref_env = parse_env_pairs(a.cand_env), parse_env_pairs(a.ref_env)
    base_env = base_environment(cfg, a.gpu)
    if a.dry_run:
        return dry_run(a, plan, tokens, base_args, exe, cand_env, ref_env, notes, tokens_file, workdir)
    runs: dict = {}
    th = {"min_top1": a.min_top1, "max_ppl_rel": a.max_ppl_rel, "max_kl": a.max_kl, "max_abs_dlogit": a.max_abs_dlogit}
    ref_desc = f"logits file {a.ref_logits}" if a.ref_logits else f"the same engine with {a.ref_env or '(the candidate environment)'}"

    def rep(exit_code: int, reasons: list[str], metrics=None, greedy=None, cfg_extra=None) -> dict:
        return {"config": {"mode": plan.mode, "n_tokens": n, "reference": ref_desc, "prefill_until": plan.prefill_until,
                           "tail": a.tail, "max_new": plan.max_new, "stride": plan.stride, "native_pack": native,
                           "pos_start": plan.pos_start, "tokens_file": str(tokens_file), **(cfg_extra or {})},
                "thresholds": th, "runs": runs, "metrics": metrics, "greedy": greedy, "notes": notes, "reasons": reasons,
                "pass": exit_code == 0, "exit": exit_code}

    # ---- the two sides
    if a.ref_logits:
        runs["reference"] = {"kind": "file", "path": a.ref_logits}
    else:
        runs["reference"] = obtain_run("reference", a, plan, tokens, base_args, exe, base_env, cwd, workdir, ref_env, a.ref_args, tokens_file)
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
        runs["candidate"] = obtain_run("candidate", a, plan, tokens, cand_base, exe, base_env, cwd, workdir, cand_env, a.cand_args, tokens_file)
        if runs["candidate"]["returncode"] != 0:
            return finish(a, workdir, rep(1, [f"the CANDIDATE engine run failed (exit {runs['candidate']['returncode']}, see "
                                              f"{workdir / 'cand.log'}); a BPT.TRAP / launch failure on the V100 looks like this"]))
    reasons: list[str] = []
    greedy = None
    cfg_extra: dict = {}
    if plan.mode == "teacher" and plan.source == "logpos":
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
        metrics = compare_rows(ref, cand, common, {int(p): tokens[int(p) + 1] for p in common if int(p) + 1 < n})
        cfg_extra = {"first_pos": int(common.min()), "last_pos": int(common.max())}
    ok, why = verdict(metrics, min_top1=a.min_top1, max_ppl_rel=a.max_ppl_rel, max_kl=a.max_kl,
                      max_abs_dlogit=a.max_abs_dlogit, need_ppl=not metrics.get("tokens_only"))
    reasons += why
    if plan.mode == "teacher" and metrics.get("rows_scored", 0) < 100:
        notes.append(f"only {metrics.get('rows_scored', 0)} rows scored: the 99% top-1 gate says little below ~100 rows")
    if (runs["reference"]["kind"] == runs["candidate"]["kind"] == "engine" and runs["reference"]["env"] == runs["candidate"]["env"]
            and a.ref_args == a.cand_args):
        notes.append("candidate and reference run the SAME configuration: this is a determinism check, not a Volta check")
    return finish(a, workdir, rep(0 if not reasons else 1, reasons, metrics, greedy, cfg_extra))


def dry_run(a, plan: Plan, tokens: list[int], base_args: list[str], exe: list[str], cand_env: dict, ref_env: dict, notes: list[str],
            tokens_file: Path, workdir: Path) -> int:
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
        if plan.mode == "teacher" and plan.source == "logpos":
            args = build_serve_args(base_args, plan, fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec, extra=shlex.split(extra))
            env = dict(env, STRATA_LOGPOS=str(workdir / ("ref.logpos" if label == "reference" else "cand.logpos")))
        else:
            args = build_engine_args(base_args, plan, tokens_file=str(tokens_file),
                                     dump=str(workdir / ("ref.bin" if label == "reference" else "cand.bin")),
                                     fixed_experts=not a.no_fixed_experts, keep_spec=a.keep_spec, stats=a.stats, extra=shlex.split(extra))
        print(f"\n[{label}] " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + (" " if env else "")
              + " ".join(shlex.quote(c) for c in [*exe, *args]))
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
    if not (a.ref_logits or a.cand_logits) and runs["reference"].get("env") == runs["candidate"].get("env") and a.ref_args == a.cand_args:
        notes.append("candidate and reference run the SAME configuration: this is a determinism check, not a Volta check")
    if m.get("rows_scored", 0) < 100:
        notes.append(f"only {m.get('rows_scored', 0)} rows scored: the 99% top-1 gate says little below ~100 rows")
    return m, cfg_extra


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
