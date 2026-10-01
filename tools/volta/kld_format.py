#!/usr/bin/env python3
"""tools/volta/kld_format.py - llama.cpp's `--kl-divergence-base` file: a reader, a writer (for tests) and the scoring rule, in NumPy.

WHY.  "Put input into the model and get the same output as llama.cpp" cannot be a bit-for-bit statement: llama.cpp's CPU and CUDA builds
(and its own ggml backends against each other) already differ in the last bits, because every kernel sums in its own order.  What
llama.cpp itself uses to say "this backend / this quantisation did not degrade the model" is
    llama-perplexity -m M.gguf -f text -c 4096 --kl-divergence-base BASE         (once, on the reference)
    llama-perplexity -m M.gguf -f text -c 4096 --kl-divergence --kl-divergence-base BASE     (the thing under test)
which prints, over the SAME text, the mean KL divergence of the new next-token distributions from the saved ones, the fraction of
positions where both pick the same top-1 token ("Same top p"), the perplexity of both, the paired change of the per-token NLL and
the change of the probability of the right token (Delta p).  This module reads the BASE file and computes exactly those numbers for
logits that did not come from llama.cpp at all - the Strata engine's - so `golden_compare.py --ref-kld BASE` produces numbers that
mean what llama.cpp's own printout means, and can be set against llama.cpp's CPU-vs-CUDA numbers (its noise floor).

THE FILE (tools/perplexity/perplexity.cpp at the commit setup.py pins, 3cf03257f219afbe7334045ff7c6a06ac68c627d: `perplexity()` writes it,
`kl_divergence()` reads it), little-endian:

    8 bytes   "_logits_"
    uint32    n_ctx          the context of one chunk
    int32     n_vocab
    int32     n_chunk
    int32     tokens[n_chunk * n_ctx]       the text's tokens as llama.cpp tokenised them (BOS added by the tokenizer if the vocab says so);
                                            chunk c is tokens[c*n_ctx : (c+1)*n_ctx].  The file keeps the ORIGINAL token of each chunk's
                                            first position; llama.cpp replaces it with BOS when it evaluates the chunk (see below)
    then for each chunk, n_ctx - 1 - n_ctx/2 rows, each  nv = 2 * ((n_vocab + 1) / 2) + 4  uint16:
        float32 scale, float32 min_log_prob          (the first four uint16)
        uint16  q[n_vocab]  (+ one pad uint16 when n_vocab is odd)
      where, with m = the row's max logit, lse = log(sum(exp(logit - m))), lo = max(min logit, m - 16):
        scale = (m - lo) / 65535,   min_log_prob = lo - m - lse,   q[i] = round((logit[i] - lo) / scale)  (0 where logit[i] <= lo)
      so the saved log-probability of token i is   scale * q[i] + min_log_prob   (16-bit resolution over a window of at most 16 nats:
      <= 1.2e-4 nats of rounding, and everything more than 16 nats below the top is clamped to one value).

WHICH POSITIONS.  Chunks are consecutive, independent n_ctx-token sequences: every chunk is evaluated from position 0 with an empty KV
cache.  With first = n_ctx / 2 (integer division) the logits of positions first .. n_ctx-1 are produced and the rows saved / scored are
those of positions first .. n_ctx-2 (n_ctx - 1 - first of them: 2047 for n_ctx = 4096), position p being scored against the token at p + 1.
The last position has no target and is not scored.  The first n_ctx/2 positions are context only.  If the vocabulary adds a BOS token
(`tokenizer.ggml.add_bos_token`), llama.cpp overwrites the first token of EVERY chunk with BOS before evaluating it, so to reproduce
its contexts an engine must do the same (`bos_substitution`).

THE SCORING RULE (`kl_divergence_result` / `log_softmax(..., base_log_prob, ...)` in perplexity.cpp), per scored row, with the new
logits z (float32), the saved row p_base(i) = exp(scale * q[i] + min_log_prob) and the target t:
    nll        = max(z) + log(sum exp(z - max z)) - z[t]                      the new model's -log p(t)
    nll_base   = -(scale * q[t] + min_log_prob)                               the saved model's -log p(t)
    KLD        = sum over i with (scale*q[i] + min_log_prob) > -16 of  p_base(i) * (log p_base(i) - log p_new(i))      KL(base || new), nats
    same top   = argmax(z) == argmax of the saved row (first maximum of each)
    dp         = exp(-nll) - exp(-nll_base)                                   the change of the right token's probability
and over all rows: PPL = exp(mean nll), the ratio ln(PPL_new / PPL_base) with the uncertainty of the PAIRED difference (covariance term),
KLD mean +- its standard error, percentiles (linear interpolation, as `percentile` in perplexity.cpp), dp mean / RMS / percentiles /
maximum.  NumPy reproduces the float32 intermediate steps; the sums are accumulated in a different order than llama.cpp's worker
threads do, so the two agree to ~1e-9 relative.  Cross-checked against the real binary (tools/volta/test_kld.py, fixtures made by llama.cpp itself
with testdata/make_kld_fixture.py): the reader reads its file, the scorer reproduces the numbers `--kl-divergence` printed, the writer writes the same bytes.

THE WRITER (`write_base`, `quantise_rows`) is the C++ one in NumPy: it exists so the tests can make a file without llama.cpp and so a
logits dump from ANY engine can be turned into a base file (the other direction: "is llama.cpp close to the engine that I trust").

COMMAND LINE
    kld_format.py info BASE                     header, chunks, scored positions, size checks
    kld_format.py bos GGUF                      what llama.cpp's vocab does at the start of each chunk: `none`, or the BOS id it substitutes
    kld_format.py parse-log FILE                the numbers of a `llama-perplexity --kl-divergence` printout, as JSON (the noise floor)
    kld_format.py floor-summary FILE            the same as one line (`OK<TAB>...` / `BAD<TAB>...`), what logit_identity.sh prints for row 2d
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("kld_format: needs numpy (run it with the repository's .venv/bin/python)")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

MAGIC = b"_logits_"
HEADER_BYTES = 8 + 4 + 4 + 4
LOG_FLOOR = -16.0          # the saved window is at most 16 nats wide, and the KL sum skips what is below it
ROW_BATCH = 16             # rows scored per NumPy step (16 x 248,320 float32 = 16 MB per temporary)


class KldError(Exception):
    """A file that is not a (complete) llama.cpp KL-divergence base file, or inputs that do not match it."""


# ================================================================================================ the file


def row_width(n_vocab: int) -> int:
    """uint16 per saved row: 4 for (scale, min_log_prob) + n_vocab rounded up to even (`nv` in perplexity.cpp)."""
    return 2 * ((n_vocab + 1) // 2) + 4


def first_scored(n_ctx: int) -> int:
    return n_ctx // 2


def n_scored(n_ctx: int) -> int:
    """Rows per chunk: positions n_ctx/2 .. n_ctx-2."""
    return n_ctx - 1 - n_ctx // 2


@dataclass
class KldBase:
    """An opened base file.  `tokens` is (n_chunk, n_ctx) int32; the rows are memory-mapped and read per chunk."""
    path: str
    n_ctx: int
    n_vocab: int
    n_chunk: int
    tokens: np.ndarray
    chunks_present: int
    data_offset: int
    notes: list[str] = field(default_factory=list)

    @property
    def first(self) -> int:
        return first_scored(self.n_ctx)

    @property
    def n_scored(self) -> int:
        return n_scored(self.n_ctx)

    @property
    def nv(self) -> int:
        return row_width(self.n_vocab)

    def positions(self) -> np.ndarray:
        """The sequence positions whose rows are saved (and scored): first .. n_ctx-2."""
        return np.arange(self.first, self.n_ctx - 1, dtype=np.int64)

    def targets(self, chunk: int) -> np.ndarray:
        """The token each saved row is scored against: tokens[first+1 .. n_ctx-1] of the chunk."""
        return self.tokens[chunk, self.first + 1:].astype(np.int64)

    def chunk_tokens(self, chunk: int) -> list[int]:
        return [int(t) for t in self.tokens[chunk]]

    def block(self, chunk: int) -> np.ndarray:
        """The chunk's saved rows, (n_scored, nv) uint16, memory-mapped."""
        if not 0 <= chunk < self.chunks_present:
            raise KldError(f"{self.path}: chunk {chunk} is not in the file ({self.chunks_present} complete chunks)")
        off = self.data_offset + chunk * self.n_scored * self.nv * 2
        return np.memmap(self.path, dtype="<u2", mode="r", offset=off, shape=(self.n_scored, self.nv))

    def header_of(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(scale, min_log_prob) per row, float32."""
        hd = np.ascontiguousarray(block[:, :4]).view("<f4")
        return hd[:, 0].astype(np.float32), hd[:, 1].astype(np.float32)


def read_base(path: str | Path) -> KldBase:
    """Open a base file; KldError says what is wrong with one that is not (the checks `kl_divergence()` makes, and the size)."""
    p = Path(path)
    if not p.exists():
        raise KldError(f"{p}: no such file (make it with `llama-perplexity -m M.gguf -f text -c N --kl-divergence-base {p}`)")
    size = p.stat().st_size
    with open(p, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES or head[:8] != MAGIC:
            raise KldError(f"{p}: does not look like a file containing log-probabilities (llama.cpp's `_logits_` header is missing)")
        n_ctx, n_vocab, n_chunk = struct.unpack("<Iii", head[8:])
        if n_ctx < 4 or n_vocab <= 0 or n_chunk <= 0 or n_ctx > (1 << 24) or n_chunk > (1 << 24):
            raise KldError(f"{p}: implausible header (n_ctx {n_ctx}, n_vocab {n_vocab}, n_chunk {n_chunk})")
        tok_bytes = n_chunk * n_ctx * 4
        if size < HEADER_BYTES + tok_bytes:
            raise KldError(f"{p}: {size} bytes cannot hold the {n_chunk} x {n_ctx} tokens its header announces "
                           "(a run that was interrupted before the text was written?)")
        tokens = np.frombuffer(f.read(tok_bytes), dtype="<i4").reshape(n_chunk, n_ctx).astype(np.int32)
    offset = HEADER_BYTES + tok_bytes
    per_chunk = n_scored(n_ctx) * row_width(n_vocab) * 2
    present = (size - offset) // per_chunk
    notes: list[str] = []
    if present < n_chunk:
        notes.append(f"the file holds {present} of the {n_chunk} chunks its header announces (llama-perplexity was interrupted): "
                     f"only those are used")
    if present == 0:
        raise KldError(f"{p}: no complete chunk of log-probabilities after the tokens")
    if (size - offset) % per_chunk and present == n_chunk:
        notes.append(f"{(size - offset) - present * per_chunk} bytes after the last chunk are ignored")
    bad = tokens[:present]
    if (bad < 0).any() or (bad >= n_vocab).any():
        raise KldError(f"{p}: token ids outside 0..{n_vocab - 1} in the stored text: not a base file of this format")
    return KldBase(str(p), int(n_ctx), int(n_vocab), int(n_chunk), tokens, int(present), offset, notes)


# ================================================================================================ the writer (C++ in NumPy)


def _nearest_int(x: np.ndarray) -> np.ndarray:
    """perplexity.cpp `nearest_int`: round-to-nearest-even through the 1.5 * 2^23 trick, float32 in, int32 out."""
    v = (x.astype(np.float32) + np.float32(12582912.0)).view(np.int32)
    return (v & 0x007fffff) - 0x00400000


def quantise_rows(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """perplexity.cpp `log_softmax(n_vocab, logits, log_prob, tok)`'s saved part, for (rows, n_vocab) float32 logits ->
    (scale float32 (rows,), min_log_prob float32 (rows,), q uint16 (rows, n_vocab))."""
    z = np.ascontiguousarray(logits, dtype=np.float32)
    mx = z.max(axis=1)
    mn = np.maximum(z.min(axis=1), mx - np.float32(16))
    sum_exp = np.exp(z - mx[:, None]).astype(np.float64).sum(axis=1)
    lse = np.log(sum_exp).astype(np.float32)
    min_log_prob = (mn - mx - lse).astype(np.float32)
    scale = ((mx - mn) / np.float32(65535.0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inv = np.where(scale != 0, np.float32(1) / scale, np.float32(0)).astype(np.float32)
    qi = _nearest_int(inv[:, None] * (z - mn[:, None]))
    q = np.where((z > mn[:, None]) & (scale[:, None] != 0), qi, 0).astype(np.uint16)
    return scale, min_log_prob, q


def write_base(path: str | Path, n_ctx: int, tokens: np.ndarray, rows_per_chunk) -> None:
    """Write a base file.  `tokens` is (n_chunk, n_ctx); `rows_per_chunk` yields, per chunk, the (n_scored, n_vocab) float32 logits of
    positions n_ctx/2 .. n_ctx-2.  Byte-for-byte what `perplexity()` writes (the pad uint16 of an odd vocabulary is zero)."""
    tokens = np.asarray(tokens, dtype="<i4")
    n_chunk, tn = tokens.shape
    if tn != n_ctx:
        raise KldError(f"tokens are {tn} per chunk, n_ctx is {n_ctx}")
    chunks = iter(rows_per_chunk)
    with open(path, "wb") as f:
        n_vocab = None
        header_pos = None
        for c in range(n_chunk):
            rows = np.asarray(next(chunks), dtype=np.float32)
            if rows.shape[0] != n_scored(n_ctx):
                raise KldError(f"chunk {c}: {rows.shape[0]} rows, expected {n_scored(n_ctx)} (positions {n_ctx // 2}..{n_ctx - 2})")
            if n_vocab is None:
                n_vocab = rows.shape[1]
                f.write(MAGIC + struct.pack("<Iii", n_ctx, n_vocab, n_chunk))
                f.write(tokens.tobytes())
            nv = row_width(n_vocab)
            scale, mlp, q = quantise_rows(rows)
            out = np.zeros((rows.shape[0], nv), dtype="<u2")
            out[:, :4] = np.stack([scale, mlp], axis=1).astype("<f4").view("<u2")
            out[:, 4:4 + n_vocab] = q
            f.write(out.tobytes())


# ================================================================================================ the scoring rule


def _f32(x):
    return np.float32(x)


def mean_and_uncertainty(total: float, total2: float, count: int) -> tuple[float, float]:
    """perplexity.cpp `mean_and_uncertainty`: the mean and the standard error of the mean (zero below 11 values)."""
    if count < 1:
        return 0.0, 0.0
    f = total / count
    df = total2 / count - f * f
    return f, (math.sqrt(df / (count - 1)) if df > 0 and count > 10 else 0.0)


def covariance(suma: float, sumb: float, sumab: float, count: int) -> float:
    """perplexity.cpp `covariance` of two sample means."""
    if count < 10:
        return 0.0
    var = sumab / count - (suma / count) * (sumb / count)
    return var / (count - 1)


def percentile(sorted_values: np.ndarray, fraction: float) -> float:
    """perplexity.cpp `percentile`: linear interpolation between neighbours of a sorted float32 array, in float32."""
    v = sorted_values
    if fraction <= 0:
        return float(v[0])
    if fraction >= 1:
        return float(v[-1])
    p = _f32(fraction) * _f32(len(v) - 1)
    ip = int(p)
    p = _f32(p - _f32(ip))
    return float((_f32(1) - p) * v[ip] + p * v[min(ip + 1, len(v) - 1)])


def median(sorted_values: np.ndarray) -> float:
    n = len(sorted_values)
    if n % 2 == 0:
        return float(_f32(0.5) * (sorted_values[n // 2] + sorted_values[n // 2 - 1]))
    return float(sorted_values[n // 2])


KLD_PERCENTILES = (0.999, 0.99, 0.95, 0.90, 0.10, 0.05, 0.01, 0.001)
DP_PERCENTILES = (0.999, 0.99, 0.95, 0.90, 0.75, 0.25, 0.10, 0.05, 0.01, 0.001)


class KldScorer:
    """Accumulates `kl_divergence_result` over rows.  `add_logits` takes the new model's full logits (KL available); `add_target`
    takes only its log-probability of the target token and its top-1 id (what STRATA_LOGPOS gives: no KL)."""

    def __init__(self):
        self.sum_nll = self.sum_nll2 = self.sum_nll_base = self.sum_nll_base2 = self.sum_nll_nll_base = 0.0
        self.sum_kld = self.sum_kld2 = self.sum_dp = self.sum_dp2 = self.sum_dp4 = 0.0
        self.max_dp = 0.0
        self.n_same_top = self.count = 0
        self.nll: list[np.ndarray] = []
        self.nll_base: list[np.ndarray] = []
        self.kld: list[np.ndarray] = []
        self.dp: list[np.ndarray] = []
        self.has_kl = True
        self.nonfinite = 0

    # -- shared bookkeeping
    def _accumulate(self, nll: np.ndarray, nll_base: np.ndarray, kld: np.ndarray | None, dp: np.ndarray, same_top: int) -> None:
        nll = nll.astype(np.float32)
        nll_base = nll_base.astype(np.float32)
        self.sum_nll += float(nll.astype(np.float64).sum())
        self.sum_nll2 += float((nll * nll).astype(np.float64).sum())
        self.sum_nll_base += float(nll_base.astype(np.float64).sum())
        self.sum_nll_base2 += float((nll_base * nll_base).astype(np.float64).sum())
        self.sum_nll_nll_base += float((nll * nll_base).astype(np.float64).sum())
        if kld is not None:
            self.sum_kld += float(kld.sum())
            self.sum_kld2 += float((kld * kld).sum())
            self.kld.append(kld.astype(np.float32))
        else:
            self.has_kl = False
        dp = dp.astype(np.float32)
        dp2 = dp.astype(np.float64) ** 2
        self.sum_dp += float(dp.astype(np.float64).sum())
        self.sum_dp2 += float(dp2.sum())
        self.sum_dp4 += float((dp2 * dp2).sum())
        self.max_dp = max(self.max_dp, float(np.abs(dp).max()) if len(dp) else 0.0)
        self.n_same_top += same_top
        self.count += len(nll)
        self.nll.append(nll)
        self.nll_base.append(nll_base)
        self.dp.append(dp)

    def add_logits(self, logits: np.ndarray, targets: np.ndarray, scale: np.ndarray, min_log_prob: np.ndarray, q: np.ndarray) -> None:
        """Score rows.  logits (r, V) float32; targets (r,) token ids; scale / min_log_prob (r,) float32 and q (r, V) uint16 are the
        saved rows.  Rows with a non-finite logit are counted in `nonfinite` and skipped."""
        for s in range(0, logits.shape[0], ROW_BATCH):
            sl = slice(s, s + ROW_BATCH)      # (`logits` and `q` may be memory-mapped: only this slice is read)
            z = np.asarray(logits[sl], dtype=np.float32)
            t, sc, ml, qq = np.asarray(targets[sl]), scale[sl], min_log_prob[sl], np.asarray(q[sl])
            ok = np.isfinite(z).all(axis=1)
            if not ok.all():
                self.nonfinite += int((~ok).sum())
                z, t, sc, ml, qq = z[ok], t[ok], sc[ok], ml[ok], qq[ok]
            if len(z) == 0:
                continue
            r = np.arange(len(z))
            mx = z.max(axis=1)
            imax = z.argmax(axis=1)
            sum_exp = np.exp(z - mx[:, None]).astype(np.float64).sum(axis=1)
            lse = np.log(sum_exp).astype(np.float32)
            nll = (mx + lse - z[r, t]).astype(np.float32)
            plb = sc[:, None] * qq.astype(np.float32) + ml[:, None]                   # saved log-probabilities, float32
            nll_base = -(sc * qq[r, t].astype(np.float32) + ml)
            top_base = plb.argmax(axis=1)
            tot = (mx + lse).astype(np.float32)
            with np.errstate(over="ignore", invalid="ignore"):
                term = np.exp(plb) * (plb - z + tot[:, None])
            kld = np.where(plb > LOG_FLOOR, term, 0.0).astype(np.float64).sum(axis=1)
            dp = (np.exp(-nll) - np.exp(-nll_base)).astype(np.float32)
            self._accumulate(nll, nll_base, kld, dp, int((imax == top_base).sum()))

    def add_target(self, logprob_target: np.ndarray, top_id: np.ndarray, targets: np.ndarray, scale: np.ndarray,
                   min_log_prob: np.ndarray, q_rows) -> None:
        """Score rows from log-probabilities only: `logprob_target` = log p_new(target) (float64), `top_id` = the new model's top-1
        token; `q_rows(i)` returns saved row i's uint16 values (for the saved top-1 and the saved target's value)."""
        lp = np.asarray(logprob_target, dtype=np.float64)
        ok = np.isfinite(lp)
        self.nonfinite += int((~ok).sum())
        if not ok.any():
            return
        idx = np.nonzero(ok)[0]
        nll = (-lp[idx]).astype(np.float32)
        nll_base = np.empty(len(idx), dtype=np.float32)
        same = 0
        for k, i in enumerate(idx):
            row = q_rows(int(i))
            nll_base[k] = -(scale[i] * np.float32(row[int(targets[i])]) + min_log_prob[i])
            plb = scale[i] * row.astype(np.float32) + min_log_prob[i]
            same += int(int(top_id[i]) == int(plb.argmax()))
        dp = (np.exp(-nll) - np.exp(-nll_base)).astype(np.float32)
        self._accumulate(nll, nll_base, None, dp, same)

    # -- the report
    def summary(self) -> dict:
        n = self.count
        out: dict = {"count": n, "nonfinite": self.nonfinite, "has_kl": self.has_kl and n > 0}
        if n == 0:
            return out
        lp, lp_unc = mean_and_uncertainty(self.sum_nll, self.sum_nll2, n)
        lb, lb_unc = mean_and_uncertainty(self.sum_nll_base, self.sum_nll_base2, n)
        cov = covariance(self.sum_nll, self.sum_nll_base, self.sum_nll_nll_base, n)
        ppl, ppl_base = math.exp(lp), math.exp(lb)
        ratio_unc2 = lp_unc ** 2 + lb_unc ** 2 - 2.0 * cov
        ratio_unc = math.sqrt(ratio_unc2) if ratio_unc2 > 0 else 0.0
        ln_ratio = lp - lb
        ppl_unc, ppl_base_unc = ppl * lp_unc, ppl_base * lb_unc
        ppl_cov = ppl * ppl_base * cov
        diff_unc2 = ppl_unc ** 2 + ppl_base_unc ** 2 - 2.0 * ppl_cov
        out.update({
            "ppl": ppl, "ppl_unc": ppl_unc, "ppl_base": ppl_base, "ppl_base_unc": ppl_base_unc,
            "ppl_cor": cov / (lp_unc * lb_unc) if lp_unc > 0 and lb_unc > 0 else None,
            "ln_ppl_ratio": ln_ratio, "ln_ppl_ratio_unc": ratio_unc,
            "ppl_ratio": math.exp(ln_ratio), "ppl_ratio_unc": math.exp(ln_ratio) * ratio_unc,
            "ppl_diff": ppl - ppl_base, "ppl_diff_unc": math.sqrt(diff_unc2) if diff_unc2 > 0 else 0.0,
        })
        nll, nllb = np.concatenate(self.nll).astype(np.float64), np.concatenate(self.nll_base).astype(np.float64)
        d = nll - nllb
        out["dnll_mean"] = float(d.mean())
        out["dnll_se"] = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None
        same_top = self.n_same_top / n
        out["same_top"] = same_top
        out["same_top_unc"] = math.sqrt(same_top * (1 - same_top) / (n - 1)) if n > 1 else 0.0
        out["top1_mismatch"] = 1.0 - same_top
        dp_mean, dp_unc = mean_and_uncertainty(self.sum_dp, self.sum_dp2, n)
        mse, mse_unc = mean_and_uncertainty(self.sum_dp2, self.sum_dp4, n)
        rms = math.sqrt(mse)
        dp_sorted = np.sort(np.concatenate(self.dp).astype(np.float32))
        out.update({"dp_mean": dp_mean, "dp_unc": dp_unc, "dp_rms": rms, "dp_rms_unc": 0.5 / rms * mse_unc if rms > 0 else 0.0,
                    "dp_max": float(dp_sorted[-1]), "dp_min": float(dp_sorted[0]), "dp_max_abs": self.max_dp,
                    "dp_median": median(dp_sorted),
                    "dp_percentiles": {str(p): percentile(dp_sorted, p) for p in DP_PERCENTILES}})
        if self.has_kl:
            k, k_unc = mean_and_uncertainty(self.sum_kld, self.sum_kld2, n)
            ks = np.sort(np.concatenate(self.kld).astype(np.float32))
            out.update({"kld_mean": k, "kld_unc": k_unc, "kld_max": float(ks[-1]), "kld_min": float(ks[0]),
                        "kld_median": median(ks), "kld_percentiles": {str(p): percentile(ks, p) for p in KLD_PERCENTILES}})
        return out


def format_summary(s: dict, title: str = "") -> str:
    """The numbers in the layout of llama.cpp's own printout (`====== Perplexity statistics ======` ...), plus what it does not show."""
    if not s.get("count"):
        return "no scored rows"
    L = []
    if title:
        L.append(title)
    pm = lambda a, b, w=10, p=6: f"{a:{w}.{p}f} +- {b:{w}.{p}f}"
    L.append("====== Perplexity statistics ======")
    L.append(f"Mean PPL(Q)                   : {pm(s['ppl'], s['ppl_unc'])}")
    L.append(f"Mean PPL(base)                : {pm(s['ppl_base'], s['ppl_base_unc'])}")
    if s.get("ppl_cor") is not None:
        L.append(f"Cor(ln(PPL(Q)), ln(PPL(base))): {100 * s['ppl_cor']:6.2f}%")
    L.append(f"Mean ln(PPL(Q)/PPL(base))     : {pm(s['ln_ppl_ratio'], s['ln_ppl_ratio_unc'])}")
    L.append(f"Mean PPL(Q)/PPL(base)         : {pm(s['ppl_ratio'], s['ppl_ratio_unc'])}")
    L.append(f"Mean PPL(Q)-PPL(base)         : {pm(s['ppl_diff'], s['ppl_diff_unc'])}")
    if s.get("has_kl"):
        L += ["", "====== KL divergence statistics ======", f"Mean    KLD: {pm(s['kld_mean'], s['kld_unc'])}",
              f"Maximum KLD: {s['kld_max']:10.6f}"]
        for p in KLD_PERCENTILES[:4]:
            L.append(f"{100 * p:4.1f}%   KLD: {s['kld_percentiles'][str(p)]:10.6f}")
        L.append(f"Median  KLD: {s['kld_median']:10.6f}")
        for p in KLD_PERCENTILES[4:]:
            L.append(f"{100 * p:4.1f}%   KLD: {s['kld_percentiles'][str(p)]:10.6f}")
        L.append(f"Minimum KLD: {s['kld_min']:10.6f}")
    else:
        L += ["", "KL divergence: not available - this run has the log-probability of the TARGET token only, not the whole logits row"]
    L += ["", "====== Token probability statistics ======",
          f"Mean    dp: {100 * s['dp_mean']:6.3f} +- {100 * s['dp_unc']:5.3f} %",
          f"Maximum dp: {100 * s['dp_max']:6.3f}%   (largest |dp|: {100 * s['dp_max_abs']:.3f}%)"]
    for p in DP_PERCENTILES[:5]:
        L.append(f"{100 * p:4.1f}%   dp: {100 * s['dp_percentiles'][str(p)]:6.3f}%")
    L.append(f"Median  dp: {100 * s['dp_median']:6.3f}%")
    for p in DP_PERCENTILES[5:]:
        L.append(f"{100 * p:4.1f}%   dp: {100 * s['dp_percentiles'][str(p)]:6.3f}%")
    L.append(f"Minimum dp: {100 * s['dp_min']:6.3f}%")
    L.append(f"RMS dp    : {100 * s['dp_rms']:6.3f} +- {100 * s['dp_rms_unc']:5.3f} %")
    L.append(f"Same top p: {100 * s['same_top']:6.3f} +- {100 * s['same_top_unc']:5.3f} %")
    if s.get("nonfinite"):
        L.append(f"rows with a non-finite logit (left out): {s['nonfinite']}")
    return "\n".join(L)


# ================================================================================================ BOS (what llama.cpp does at the start of a chunk)


def bos_substitution(gguf: str | Path) -> int | None:
    """None when llama.cpp's vocab for this GGUF adds no BOS (`tokenizer.ggml.add_bos_token` false or absent for a BPE vocabulary: the
    Qwen / GPT-2 family), else the BOS token id it writes over the first token of every chunk (`tokenizer.ggml.bos_token_id`).  Reads only
    the GGUF header through tools/gguf_reader.py (which also knows the Q2_0 type gguf-py does not)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile   # noqa: E402
    md = GGUFFile(Path(gguf)).metadata
    model = md.get("tokenizer.ggml.model")
    add = md.get("tokenizer.ggml.add_bos_token")
    if add is None:
        add = model in ("llama", "spm", "bert", "wpm")     # llama-vocab.cpp: SPM and WPM add BOS by default, BPE does not
    if not add:
        return None
    bos = md.get("tokenizer.ggml.bos_token_id")
    if bos is None:
        raise KldError(f"{gguf}: the vocabulary adds BOS (add_bos_token) but names no bos_token_id; llama.cpp's own default for the "
                       "tokenizer model would apply: pass --kld-bos ID")
    return int(bos)


# ================================================================================================ llama.cpp's own printout -> numbers

_NUM = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_PM = _NUM + r"\s*(?:±|\+-|\+/-)\s*" + _NUM
_LINES = {
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*" + _PM,
    "ppl_base": r"Mean PPL\(base\)\s*:\s*" + _PM,
    "ln_ppl_ratio": r"Mean ln\(PPL\(Q\)/PPL\(base\)\)\s*:\s*" + _PM,
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*" + _PM,
    "ppl_diff": r"Mean PPL\(Q\)-PPL\(base\)\s*:\s*" + _PM,
    "kld": r"Mean\s+KLD:\s*" + _PM,
    "dp_mean": r"Mean\s+(?:Δp|dp):\s*" + _PM + r"\s*%",
    "dp_rms": r"RMS\s+(?:Δp|dp)\s*:\s*" + _PM + r"\s*%",
    "same_top": r"Same top p:\s*" + _PM + r"\s*%",
}
_SINGLE = {
    "kld_max": r"Maximum KLD:\s*" + _NUM,
    "kld_p999": r"99\.9%\s+KLD:\s*" + _NUM,
    "kld_p99": r"99\.0%\s+KLD:\s*" + _NUM,
    "kld_median": r"Median\s+KLD:\s*" + _NUM,
    "dp_max": r"Maximum (?:Δp|dp):\s*" + _NUM + r"\s*%",
    "ppl_cor": r"Cor\(ln\(PPL\(Q\)\), ln\(PPL\(base\)\)\):\s*" + _NUM + r"\s*%",
}


_TABLE_ROW = re.compile(r"^\s*(\d+)\s+" + r"\s+".join([_PM] * 3 + [_PM + r"\s*%", _PM + r"\s*%"]), re.M)


def parse_llama_log(text: str) -> dict:
    """The statistics block of `llama-perplexity --kl-divergence` -> {name: [value, uncertainty] | value}.  Percent quantities (dp, Same top p)
    are kept in percent as llama.cpp prints them.  Raises KldError when the block is missing (the run failed, or saw < 100 positions)."""
    out: dict = {}
    for k, pat in _LINES.items():
        m = re.search(pat, text)
        if m:
            out[k] = [float(m.group(1)), float(m.group(2))]
    for k, pat in _SINGLE.items():
        m = re.search(pat, text)
        if m:
            out[k] = float(m.group(1))
    if "kld" not in out or "same_top" not in out:
        # llama.cpp's log thread can lose the last lines when the process exits (seen with a pipe); the per-chunk table printed before them
        # carries the same running totals in its last row, with fewer digits
        rows = _TABLE_ROW.findall(text)
        if not rows:
            raise KldError("no `====== KL divergence statistics ======` block (and no per-chunk table) in the log: the run did not finish, or "
                           "scored fewer than 100 positions (llama.cpp prints no statistics below that)")
        r = [float(x) for x in rows[-1][1:]]
        out.update({"ppl_q": r[0:2], "ln_ppl_ratio": r[2:4], "kld": r[4:6], "dp_rms": r[6:8], "same_top": r[8:10], "from_table": True})
    return out


# ================================================================================================ command line


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="header and layout of a base file")
    p.add_argument("base")
    p = sub.add_parser("bos", help="the BOS id llama.cpp substitutes at each chunk's start, or `none`")
    p.add_argument("gguf")
    p = sub.add_parser("parse-log", help="the numbers of a llama-perplexity --kl-divergence printout, as JSON")
    p.add_argument("log")
    p = sub.add_parser("floor-summary", help="one line about a noise-floor printout, for scripts: `OK<TAB>text`, or `BAD<TAB>reason` when the "
                       "two backends disagree beyond --max-kl (then they are no reference for each other) or the log is unreadable")
    p.add_argument("log")
    p.add_argument("--max-kl", type=float, default=0.05)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "info":
            b = read_base(a.base)
            print(f"{b.path}: n_ctx {b.n_ctx}, n_vocab {b.n_vocab}, {b.chunks_present} of {b.n_chunk} chunks present")
            print(f"  scored positions per chunk: {b.first}..{b.n_ctx - 2} ({b.n_scored} rows of {b.nv} uint16), targets are tokens "
                  f"{b.first + 1}..{b.n_ctx - 1}")
            print(f"  first tokens of the chunks: {[int(t) for t in b.tokens[:b.chunks_present, 0]]}")
            for n in b.notes:
                print(f"  note: {n}")
        elif a.cmd == "bos":
            bos = bos_substitution(a.gguf)
            print("none" if bos is None else bos)
        elif a.cmd == "parse-log":
            print(json.dumps(parse_llama_log(Path(a.log).read_text(encoding="utf-8", errors="replace")), indent=1))
        elif a.cmd == "floor-summary":
            try:
                p = parse_llama_log(Path(a.log).read_text(encoding="utf-8", errors="replace"))
            except KldError as e:
                print(f"BAD\t{e}")
                return 0
            text = (f"mean KLD {p['kld'][0]:.3e} +- {p['kld'][1]:.1e}, same top-1 {p['same_top'][0]:.3f}%, ln(PPL cpu / PPL cuda) "
                    f"{p['ln_ppl_ratio'][0]:+.5f} +- {p['ln_ppl_ratio'][1]:.5f}" + (" (from the per-chunk table: the log lost its last lines)" if p.get("from_table") else ""))
            print(("OK" if p["kld"][0] <= a.max_kl else "BAD") + "\t" + text + ("" if p["kld"][0] <= a.max_kl else f" - above {a.max_kl:g}"))
    except KldError as e:
        print(f"kld_format: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"kld_format: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
