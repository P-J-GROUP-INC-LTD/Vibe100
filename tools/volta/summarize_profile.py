#!/usr/bin/env python3
"""tools/volta/summarize_profile.py - the Phase-0 table of docs/volta/PLAN.md from what profile_decode.sh collected.

Phase 0 asks one question before any kernel is touched: where does a decode step on THIS V100 box actually go?  The plan
expects the answer to be the expert cache hit rate, the CPU's expert throughput and PCIe - not the GPU kernels - and the
Gate-1 target (IQ3 decode >= 40 tok/s) depends on which of those is the limit.  This script turns the engine's `--stats`
printout and Nsight Systems' CSV summaries into the table that decides that:

    GPU kernel time by kernel      top 15, % of all kernel time               (nsys cuda_gpu_kern_sum)
    CPU expert compute time        the pool's ms per round / per token         (--stats: "pool multi", "the CPU expert pool")
    GPU idle share                 1 - (pure GPU floor) / (wall time per token)   (--gpu-only-full vs --stats "decode")
                                   and kernel-busy share of the profiled run   (nsys)
    expert cache hit rate          GPU tier hits / lookups                     (--stats "R4 expert-cache hits")
    MTP acceptance                 tokens per verify round, drafts accepted    (--stats "speculation")
    PCIe traffic per token         H2D + D2H MB / generated tokens             (nsys cuda_gpu_mem_size_sum)
    tok/s                          decode and prefill                          (--stats "decode" / "prefill")

Every number is parsed from the exact strings src/program/generate.cpp prints at the end of main() (the printf formats are
reproduced in test_summarize_profile.py, so a change of wording there breaks a test here instead of silently producing an
empty table).  Missing inputs leave their rows out and say so; nothing is guessed.

    python3 tools/volta/summarize_profile.py --stats stats.txt [--gpu-floor floor.txt] \\
        [--nsys-kern cuda_gpu_kern_sum.csv] [--nsys-memsize cuda_gpu_mem_size_sum.csv] [--nsys-memtime cuda_gpu_mem_time_sum.csv] \\
        [--nsys-run run.txt] [--serve-log strata-q2_0.log] [--tokens N] [--json out.json] [--gpu "Tesla V100-PCIE-32GB, 7.0"]

`--serve-log` reads a deployed engine's log (`strata serve:` lines on stderr, `DONE` lines, the server's own `[strata] done:`
lines) and tabulates the last requests: prompt speed, decode speed, draft acceptance, expert cache hit rate.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------------- the engine's --stats

_F = r"(-?[\d.]+(?:e[-+]?\d+)?)"     # a printf %f / %g number
_I = r"(\d+)"

PATTERNS = {
    # std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", ...)
    "decode": rf"^decode\s+{_I} tokens in {_F} ms\s+->\s+{_F} tok/s",
    # std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill", ...)
    "prefill": rf"^prefill\s+{_I} tokens in {_F} ms\s+->\s+{_F} tok/s(?:\s+\(time to first token {_F} ms\))?",
    # std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms)
    "per_token": rf"^\s*per token\s+{_F} ms/token",
    # std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n", "  the CPU expert pool", ...)
    "cpu_pool_token": rf"^\s*the CPU expert pool\s+{_F} ms/token over {_I} layers \({_F} positions, {_I} dispatches\)",
    # "%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n", "  pool phases"
    "pool_phases": rf"^\s*pool phases\s+wait-park {_F}\s+drain {_F}\s+re-park {_F}\s+ms/token",
    "expert_blobs": rf"^\s*expert blobs\s+{_I} blobs read",
    # "%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n", "  R4 expert-cache hits"
    "cache_hits": rf"^\s*R4 expert-cache hits\s+{_I} of {_I} = {_F}\s+\({_I} admitted, {_I} refused, cache {_F}% full\)",
    # "%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n", "speculation"
    "speculation": rf"^speculation\s+{_I} rounds of {_I}, drafts accepted {_I} of {_I} \({_F}\), {_F} tokens per round",
    "window_sizes": r"^window sizes\s+(.*?)\s+\(min draft probability ([\d.]+)\)",
    "accepted_hist": r"^accepted per round\s+(.*)$",
    "suffix": rf"^suffix drafts\s+{_I} windows, drafts accepted {_I} of {_I}",
    # "%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f distinct / %.2f routed per layer\n"
    "verify_window": rf"^verify window\s+wait for rings {_F}\s+pool {_F}\s+host {_F}\s+commit {_F} ms/round; CPU experts {_F} distinct / {_F} routed per layer",
    # "%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; CPU pool call %.3f ms/round\n"
    "pool_multi": rf"^pool multi\s+gate/up {_F}\s+quantize {_F}\s+down {_F} ms/round; {_F} GB/s over the rows phases; CPU pool call {_F} ms/round",
    "dispatch": rf"^dispatch\s+plan {_F}\s+activation quantize {_F}\s+jobs {_F}\s+run {_F} ms/round",
    "pcie": rf"^pcie experts\s+{_F} distinct experts per layer read over PCIe \(share {_I}/256 of the misses\)",
    # "%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp"
    "mtp": rf"^mtp\s+{_F} ms/round drafting \({_I} rounds\), MTP prompt {_F} ms, {_F} MiB of VRAM",
    "tiers": rf"^expert tiers\s+decode: RAM {_I} blobs, files {_I} blobs, {_F} MB read from the files",
    # std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts streamed ...")
    "batched_prefill": rf"strata generate: prefill {_I} tokens in {_I} chunks, {_F} ms \({_F} tok/s\)",
    # std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ...) and the three lines under it
    # std::fprintf(stderr, "strata generate: this CPU has no AVX-512: the expert kernels run on %s (multi-token for the i-quant gate/up rows)\n", ...)
    "no_avx512": r"strata generate: this CPU has no AVX-512: the expert kernels run on (AVX-2|ggml-cpu vec_dot[^\n(]*(?:\([^)]*\))?)",
    "gpu_floor": rf"^GPU floor pre\+post\+head\s+{_F} ms per token\s+->\s+{_F} tok/s of GPU work",
    "gpu_floor_layers": rf"^\s+{_F} ms layers \({_I} x pre\+post\)",
    "gpu_floor_head": rf"^\s+{_F} ms LM head",
}
_RX = {k: re.compile(v, re.M) for k, v in PATTERNS.items()}


def _num(x: str):
    return float(x) if any(c in x for c in ".e") else int(x)


def parse_stats(text: str) -> dict:
    """Everything `strata --stats` (and `--gpu-only-full`) printed that Phase 0 uses.  Keys absent from the text are absent."""
    out: dict = {}
    g = lambda k: _RX[k].search(text)
    m = g("decode")
    if m:
        out["decode"] = {"tokens": int(m.group(1)), "ms": float(m.group(2)), "tok_s": float(m.group(3))}
    m = g("prefill")
    if m:
        out["prefill"] = {"tokens": int(m.group(1)), "ms": float(m.group(2)), "tok_s": float(m.group(3)),
                          "ttft_ms": float(m.group(4)) if m.group(4) else None}
    m = g("per_token")
    if m:
        out["ms_per_token"] = float(m.group(1))
    m = g("cpu_pool_token")
    if m:
        out["cpu_pool_token"] = {"ms_per_token": float(m.group(1)), "layers": int(m.group(2)), "positions": float(m.group(3)),
                                 "dispatches": int(m.group(4))}
    m = g("pool_phases")
    if m:
        out["pool_phases"] = {"wait_park": float(m.group(1)), "drain": float(m.group(2)), "re_park": float(m.group(3))}
    m = g("expert_blobs")
    if m:
        out["expert_blobs"] = int(m.group(1))
    m = g("cache_hits")
    if m:
        out["cache"] = {"hits": int(m.group(1)), "lookups": int(m.group(2)), "rate": float(m.group(3)),
                        "admitted": int(m.group(4)), "refused": int(m.group(5)), "full_pct": float(m.group(6))}
    m = g("speculation")
    if m:
        out["speculation"] = {"rounds": int(m.group(1)), "window": int(m.group(2)), "accepted": int(m.group(3)),
                              "offered": int(m.group(4)), "accept_rate": float(m.group(5)), "tokens_per_round": float(m.group(6))}
    m = g("window_sizes")
    if m:
        out["window_sizes"] = {int(a): int(b) for a, b in re.findall(r"T(\d+):(\d+)", m.group(1))}
    m = g("accepted_hist")
    if m:
        out["accepted_hist"] = {int(a): int(b) for a, b in re.findall(r"(\d+):(\d+)", m.group(1))}
    m = g("suffix")
    if m:
        out["suffix_drafts"] = {"windows": int(m.group(1)), "accepted": int(m.group(2)), "offered": int(m.group(3))}
    m = g("verify_window")
    if m:
        out["verify_window"] = {"wait_rings": float(m.group(1)), "pool": float(m.group(2)), "host": float(m.group(3)),
                                "commit": float(m.group(4)), "cpu_distinct": float(m.group(5)), "cpu_routed": float(m.group(6))}
    m = g("pool_multi")
    if m:
        out["pool_multi"] = {"gate_up": float(m.group(1)), "quantize": float(m.group(2)), "down": float(m.group(3)),
                             "gb_s": float(m.group(4)), "pool_call": float(m.group(5))}
    m = g("dispatch")
    if m:
        out["dispatch"] = {"plan": float(m.group(1)), "act_quant": float(m.group(2)), "jobs": float(m.group(3)), "run": float(m.group(4))}
    m = g("pcie")
    if m:
        out["pcie_experts"] = {"distinct_per_layer": float(m.group(1)), "share_256": int(m.group(2))}
    m = g("mtp")
    if m:
        out["mtp"] = {"ms_draft": float(m.group(1)), "rounds": int(m.group(2)), "prompt_ms": float(m.group(3)), "vram_mib": float(m.group(4))}
    m = g("tiers")
    if m:
        out["tiers"] = {"ram_blobs": int(m.group(1)), "file_blobs": int(m.group(2)), "file_mb": float(m.group(3))}
    m = g("batched_prefill")
    if m:
        out["batched_prefill"] = {"tokens": int(m.group(1)), "chunks": int(m.group(2)), "ms": float(m.group(3)), "tok_s": float(m.group(4))}
    m = g("no_avx512")
    if m:
        out["cpu_kernels"] = m.group(1).strip()
    m = g("gpu_floor")
    if m:
        out["gpu_floor"] = {"ms_per_token": float(m.group(1)), "tok_s": float(m.group(2))}
        ml, mh = g("gpu_floor_layers"), g("gpu_floor_head")
        if ml:
            out["gpu_floor"]["ms_layers"] = float(ml.group(1))
        if mh:
            out["gpu_floor"]["ms_head"] = float(mh.group(1))
    return out


# ------------------------------------------------------------------------------------------------ the server's logs

_SERVE = re.compile(
    rf"strata serve: prompt {_I} tokens = {_I} reused \+ {_I} read in {_F} ms \({_F} tok/s\), {_I} generated in {_F} ms \({_F} tok/s\), "
    rf"drafts accepted {_I} of {_I}")
_SERVE_HIT = re.compile(rf"strata serve: decode expert cache hit rate: {_F}% \({_I} hits / {_I} lookups\)")
_DONE = re.compile(rf"^DONE {_I} {_I} {_F} {_F} (\S+) {_I} {_I} {_I}(?: {_I} {_I})?", re.M)
_SRV_DONE = re.compile(rf"\[strata\] done: {_I} tokens in {_F} s \({_F} tok/s\)(?: \(([^)]*)\))?(?:, expert cache {_F}% hit)?")


def parse_serve_log(text: str, last: int = 10) -> list[dict]:
    """The last `last` requests of a deployed engine's log (or the server's console), oldest first."""
    reqs: list[dict] = []
    for line in text.splitlines():
        m = _SERVE.search(line)
        if m:
            reqs.append({"prompt_tokens": int(m.group(1)), "reused": int(m.group(2)), "fresh": int(m.group(3)),
                         "prompt_ms": float(m.group(4)), "prompt_tok_s": float(m.group(5)), "generated": int(m.group(6)),
                         "decode_ms": float(m.group(7)), "decode_tok_s": float(m.group(8)), "drafts_accepted": int(m.group(9)),
                         "drafts_offered": int(m.group(10))})
            continue
        m = _SERVE_HIT.search(line)
        if m and reqs:
            reqs[-1].update(hit_rate=float(m.group(1)) / 100.0, hits=int(m.group(2)), lookups=int(m.group(3)))
            continue
        m = _SRV_DONE.search(line)
        if m:
            reqs.append({"server_done": True, "generated": int(m.group(1)), "seconds": float(m.group(2)), "decode_tok_s": float(m.group(3)),
                         "finish": m.group(4), **({"hit_rate": float(m.group(5)) / 100.0} if m.group(5) else {})})
    return reqs[-last:]


# ------------------------------------------------------------------------------------------------- nsys CSV reports


def _norm(h: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", h.lower())


def parse_nsys_csv(text: str, needle: str = "name") -> list[dict]:
    """Rows of one `nsys stats --format csv` report.  nsys prints progress lines and a `** Report name **` banner before the
    CSV header and a blank line after the last row; the header is the first line that has `needle` as one of its columns
    (`name` for kernels, `operation` for the memory reports).  Column names are normalised (`Total Time (ns)` ->
    `totaltimens`), so the nsys versions that spell them slightly differently all parse."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if "," in ln and needle in [_norm(c) for c in next(csv.reader([ln]))]:
            start = i
            break
    if start is None:
        return []
    rows = []
    reader = csv.reader(lines[start:])
    header = [_norm(c) for c in next(reader)]
    for r in reader:
        if not r or all(not c.strip() for c in r):
            if rows:
                break          # the blank line that ends this report (a second report may follow)
            continue
        if len(r) < len(header):
            continue
        rows.append(dict(zip(header, r)))
    return rows


def _pick(row: dict, *names: str) -> float | None:
    for n in names:
        if n in row and row[n].strip():
            try:
                return float(row[n].replace(",", ""))
            except ValueError:
                pass
    return None


# kernel names that run on the tensor cores: cuBLAS' Volta HMMA kernels (volta_h884gemm_*, volta_fp16_s884gemm_*), Turing / Ampere ones, CUTLASS /
# xmma tensor-op kernels, WMMA kernels, and the port's own Volta prompt attention.  A NAME heuristic: it can miss a renamed kernel,
# which is why the table lists what matched.
TENSOR_CORE_NAMES = re.compile(r"884|1688|16816|tensorop|hmma|wmma|xmma|prompt_attn_volta", re.I)


def kernel_table(rows: list[dict], top: int = 15) -> dict:
    items = []
    for r in rows:
        t = _pick(r, "totaltimens", "totaltime", "time")
        if t is None:
            continue
        items.append({"name": r.get("name", "?"), "total_ns": t, "instances": int(_pick(r, "instances", "count") or 0),
                      "avg_ns": _pick(r, "avgns", "avg")})
    total = sum(i["total_ns"] for i in items)
    items.sort(key=lambda i: -i["total_ns"])
    for i in items:
        i["pct"] = 100.0 * i["total_ns"] / total if total else 0.0
    tc = [i for i in items if TENSOR_CORE_NAMES.search(i["name"])]
    return {"total_ns": total, "n_kernels": len(items), "top": items[:top], "top_pct": sum(i["pct"] for i in items[:top]),
            "tensor_core_ns": sum(i["total_ns"] for i in tc), "tensor_core_names": [i["name"] for i in tc[:5]]}


def mem_tables(size_rows: list[dict], time_rows: list[dict]) -> dict:
    out: dict = {"size_mb": {}, "count": {}, "time_ns": {}}
    for r in size_rows:
        op = r.get("operation", "")
        mb = _pick(r, "totalmb", "total")
        if mb is not None:
            out["size_mb"][op] = mb
            out["count"][op] = int(_pick(r, "count") or 0)
    for r in time_rows:
        op = r.get("operation", "")
        t = _pick(r, "totaltimens", "totaltime")
        if t is not None:
            out["time_ns"][op] = t
    return out


def _sum_ops(d: dict, *needles: str) -> float:
    return sum(v for k, v in d.items() if any(n in k.lower() for n in needles))


# -------------------------------------------------------------------------------------------------------- the table


def build_summary(stats: dict, floor: dict, kern: dict | None, mem: dict | None, nsys_run: dict, tokens: int | None) -> dict:
    s: dict = {"rows": []}
    dec = stats.get("decode")
    n_dec = tokens or (dec["tokens"] if dec else None)
    row = lambda name, value, note="": s["rows"].append({"name": name, "value": value, "note": note})
    if dec:
        row("decode tok/s", f"{dec['tok_s']:.2f}", f"{dec['tokens']} tokens in {dec['ms']:.0f} ms (wall clock, includes drafting and the host)")
    pf = stats.get("prefill")
    if pf:
        row("prefill tok/s", f"{pf['tok_s']:.1f}", f"{pf['tokens']} prompt tokens in {pf['ms']:.0f} ms" +
            (f"; time to first token {pf['ttft_ms']:.0f} ms" if pf.get("ttft_ms") else ""))
    sp = stats.get("speculation")
    if sp:
        row("MTP acceptance (tokens per verify round)", f"{sp['tokens_per_round']:.2f}",
            f"drafts accepted {sp['accepted']} of {sp['offered']} ({100 * sp['accept_rate']:.1f}%), {sp['rounds']} rounds of window {sp['window']}")
    c = stats.get("cache")
    if c:
        row("expert cache hit rate (GPU tier)", f"{100 * c['rate']:.1f}%", f"{c['hits']} of {c['lookups']} lookups; cache {c['full_pct']:.1f}% full, {c['refused']} refused")
    elif stats.get("serve_hit") is not None:
        row("expert cache hit rate (GPU tier)", f"{100 * stats['serve_hit']:.1f}%", "from the server log")
    # the CPU's share
    row("CPU expert kernels", stats.get("cpu_kernels", "AVX-512 (no 'no AVX-512' notice in the log)"),
        "the engine's own startup line: its AVX-512 kernels need F/BW/VL/DQ + VNNI (Cascade Lake, Ice Lake, Zen 4 and newer); without them it runs AVX-2 / ggml-cpu"
        if stats.get("cpu_kernels") else "")
    pm = stats.get("pool_multi")
    ct = stats.get("cpu_pool_token")
    if pm:
        per_tok = pm["pool_call"] / sp["tokens_per_round"] if sp and sp["tokens_per_round"] else None
        row("CPU expert compute (verify rounds)", f"{pm['pool_call']:.2f} ms/round",
            (f"= {per_tok:.2f} ms per generated token; " if per_tok else "") + f"gate/up {pm['gate_up']:.2f} + quantize {pm['quantize']:.2f} + down {pm['down']:.2f} ms; {pm['gb_s']:.1f} GB/s over the rows phases")
    elif ct:
        row("CPU expert compute (token path)", f"{ct['ms_per_token']:.2f} ms/token", f"over {ct['layers']} layers, {ct['positions']:.0f} positions")
    vw = stats.get("verify_window")
    if vw:
        row("verify round, per-stage host time", f"wait {vw['wait_rings']:.2f} / pool {vw['pool']:.2f} / host {vw['host']:.2f} / commit {vw['commit']:.2f} ms",
            f"CPU experts {vw['cpu_distinct']:.1f} distinct of {vw['cpu_routed']:.1f} routed per layer")
    # the GPU's share
    wall = None
    if dec and dec["tokens"]:
        wall = dec["ms"] / dec["tokens"]
    gf = floor.get("gpu_floor") or stats.get("gpu_floor")
    if gf:
        row("GPU floor (pure GPU time per token)", f"{gf['ms_per_token']:.2f} ms", f"{gf['tok_s']:.1f} tok/s of GPU work, T=1 graphs, no CPU pool"
            + (f"; layers {gf['ms_layers']:.1f} ms + head {gf['ms_head']:.1f} ms" if "ms_layers" in gf else ""))
        if wall:
            idle = max(0.0, 1.0 - gf["ms_per_token"] / wall)
            row("GPU idle share", f"{100 * idle:.0f}%", f"1 - floor {gf['ms_per_token']:.1f} / wall {wall:.1f} ms per token: the time the GPU waits for the CPU pool, PCIe, drafting and the host")
    if kern:
        run_ms = None
        if nsys_run.get("decode") or nsys_run.get("prefill"):
            run_ms = (nsys_run.get("decode", {}).get("ms", 0) or 0) + (nsys_run.get("prefill", {}).get("ms", 0) or 0)
        busy = kern["total_ns"] / 1e6
        row("GPU kernel time (nsys, whole profiled run)", f"{busy:.0f} ms", f"{kern['n_kernels']} kernels; top 15 = {kern['top_pct']:.0f}%"
            + (f"; {100 * busy / run_ms:.0f}% of the run's {run_ms:.0f} ms wall clock (an upper bound: concurrent streams overlap)" if run_ms else ""))
    if kern:
        tcs = kern.get("tensor_core_ns", 0)
        row("tensor-core kernel time (by name)", f"{100 * tcs / kern['total_ns']:.1f}%" if kern["total_ns"] else "-",
            "; ".join(kern.get("tensor_core_names", [])[:2])[:110] or "no kernel name looks like a tensor-core kernel (a decode-only profile has none; profile a long "
            "prompt with --nsys-prompt-tokens 0 to see the prefill GEMMs and the Volta attention)")
    if mem:
        h2d = _sum_ops(mem["size_mb"], "host-to-device", "htod")
        d2h = _sum_ops(mem["size_mb"], "device-to-host", "dtoh")
        n_nsys = nsys_run.get("decode", {}).get("tokens") or n_dec
        if n_nsys:
            row("PCIe traffic per token", f"{(h2d + d2h) / n_nsys:.1f} MB", f"H2D {h2d:.0f} MB + D2H {d2h:.0f} MB over {n_nsys} generated tokens"
                " (includes the prompt's prefill traffic: profile a short prompt for a pure decode figure)")
        else:
            row("PCIe traffic", f"H2D {h2d:.0f} MB, D2H {d2h:.0f} MB", "no token count: pass --tokens")
        t_cpy = _sum_ops(mem["time_ns"], "memcpy") / 1e6
        if t_cpy:
            row("copy engine busy time (nsys)", f"{t_cpy:.0f} ms", "sum of memcpy durations")
    t = stats.get("tiers")
    if t:
        row("expert tiers during decode", f"RAM {t['ram_blobs']} / file {t['file_blobs']} blobs", f"{t['file_mb']:.1f} MB read from the files")
    pe = stats.get("pcie_experts")
    if pe:
        row("experts read over PCIe by the GPU", f"{pe['distinct_per_layer']:.2f} per layer", f"share {pe['share_256']}/256 of the misses (--pcie-frac)")
    return s


def format_table(summary: dict, kern: dict | None, gpu: str | None, serve: list[dict] | None) -> str:
    L = ["Phase 0 - where a decode step goes" + (f"   [{gpu}]" if gpu else "")]
    if not summary["rows"]:
        L.append("  (no recognisable --stats output: did the run print its summary? is the file the engine's stdout?)")
    else:
        w = max(len(r["name"]) for r in summary["rows"])
        v = max(len(r["value"]) for r in summary["rows"])
        L.append(f"  {'metric'.ljust(w)}  {'value'.ljust(v)}  note")
        L.append("  " + "-" * w + "  " + "-" * v + "  " + "-" * 40)
        for r in summary["rows"]:
            L.append(f"  {r['name'].ljust(w)}  {r['value'].ljust(v)}  {r['note']}")
    if kern and kern["top"]:
        L += ["", f"GPU kernel time by kernel (nsys cuda_gpu_kern_sum; total {kern['total_ns'] / 1e6:.1f} ms, {kern['n_kernels']} kernels)",
              f"  {'%':>6}  {'total ms':>10}  {'calls':>8}  {'avg us':>9}  kernel"]
        for k in kern["top"]:
            name = k["name"]
            name = name if len(name) <= 96 else name[:93] + "..."
            avg = f"{k['avg_ns'] / 1000:.1f}" if k.get("avg_ns") else "-"
            L.append(f"  {k['pct']:>6.1f}  {k['total_ns'] / 1e6:>10.2f}  {k['instances']:>8}  {avg:>9}  {name}")
        L.append(f"  top {len(kern['top'])} = {kern['top_pct']:.1f}% of all kernel time")
    if serve:
        L += ["", "Deployed engine, last requests (from the log)"]
        for r in serve:
            if r.get("server_done"):
                L.append(f"  server: {r['generated']} tokens in {r['seconds']:.0f} s, {r['decode_tok_s']:.1f} tok/s ({r['finish']})"
                         + (f", expert cache {100 * r['hit_rate']:.1f}% hit" if "hit_rate" in r else ""))
            else:
                acc = f", drafts {r['drafts_accepted']}/{r['drafts_offered']}" if r["drafts_offered"] else ""
                hit = f", cache {100 * r['hit_rate']:.1f}% hit" if "hit_rate" in r else ""
                L.append(f"  prompt {r['prompt_tokens']} tokens ({r['fresh']} read) {r['prompt_tok_s']:.0f} tok/s; "
                         f"{r['generated']} generated at {r['decode_tok_s']:.1f} tok/s{acc}{hit}")
    return "\n".join(L)


def read_text(p: str | None) -> str:
    if not p:
        return ""
    try:
        return Path(p).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"summarize_profile: cannot read {p}: {e}", file=sys.stderr)
        return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", help="the engine's stdout/stderr of a `--stats` run")
    ap.add_argument("--gpu-floor", help="the engine's output of a `--gpu-only-full` run")
    ap.add_argument("--nsys-kern", help="`nsys stats --report cuda_gpu_kern_sum --format csv` output")
    ap.add_argument("--nsys-memsize", help="... cuda_gpu_mem_size_sum")
    ap.add_argument("--nsys-memtime", help="... cuda_gpu_mem_time_sum")
    ap.add_argument("--nsys-run", help="the engine's own output of the run nsys profiled (its decode / prefill wall clock)")
    ap.add_argument("--serve-log", help="a deployed engine's log (strata-*.log) or the server's console")
    ap.add_argument("--tokens", type=int, help="generated tokens of the profiled run (default: from the engine output)")
    ap.add_argument("--top", type=int, default=15, help="kernels in the table (default 15)")
    ap.add_argument("--gpu", help="the card, for the heading (nvidia-smi name, compute capability)")
    ap.add_argument("--json", help="write everything parsed, as JSON")
    a = ap.parse_args(argv)

    stats = parse_stats(read_text(a.stats))
    floor = parse_stats(read_text(a.gpu_floor))
    nsys_run = parse_stats(read_text(a.nsys_run))
    kern = kernel_table(parse_nsys_csv(read_text(a.nsys_kern), "name"), a.top) if a.nsys_kern else None
    mem = None
    if a.nsys_memsize or a.nsys_memtime:
        mem = mem_tables(parse_nsys_csv(read_text(a.nsys_memsize), "operation") if a.nsys_memsize else [],
                         parse_nsys_csv(read_text(a.nsys_memtime), "operation") if a.nsys_memtime else [])
    serve = parse_serve_log(read_text(a.serve_log)) if a.serve_log else None
    if serve:
        hits = [r["hit_rate"] for r in serve if "hit_rate" in r]
        if hits and "cache" not in stats:
            stats["serve_hit"] = hits[-1]
    if not stats and not floor and not kern and not serve:
        print("summarize_profile: nothing to summarize (give --stats, --nsys-kern, --serve-log ...)", file=sys.stderr)
        return 2
    summary = build_summary(stats, floor, kern, mem, nsys_run, a.tokens)
    print(format_table(summary, kern, a.gpu, serve))
    if a.json:
        Path(a.json).write_text(json.dumps({"gpu": a.gpu, "stats": stats, "gpu_floor": floor.get("gpu_floor"), "summary": summary,
                                            "kernels": kern, "memory": mem, "serve": serve}, indent=1), encoding="utf-8")
        print(f"summarize_profile: wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
