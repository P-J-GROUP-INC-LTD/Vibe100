#!/usr/bin/env python3
"""manifest.py - validate a DeepSeek-V4.1-Flash GGUF against the `deepseek41` contract and write its tensor manifest.

    python3 tools/ds41/manifest.py /models/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf --out manifest.json
    python3 tools/ds41/manifest.py third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz

INPUT is one or more of: a .gguf (any shard of a `-0000N-of-0000M` split pulls in its siblings, and a `*DSpark*.gguf`
next to it is picked up as the optional sidecar), a directory of .gguf files, or a headers JSON written by
tools/ds41_gguf_remote_headers.py.  With real files the manifest carries absolute data offsets (header end rounded up to
general.alignment + the tensor's own offset) and checks that every tensor lies inside its file; from a headers JSON the
data section start is unknown, so offsets stay relative to it.

Checks (each finding has a code; errors make the exit status 2):
  * general.architecture == deepseek41; split.no / split.count / split.tensors.count consistent with the files given
  * every metadata key the port needs (ds41_spec.KEYS), against the compile-time geometry of
    include/strata/ds41/geometry.hpp (`--geometry ds41`, default) or only against itself (`--geometry self`, the mini
    test GGUF); published spellings of the other GGUF family (vcruz305) are accepted as aliases, its missing CSA2 /
    Engram keys are derived from the config.json defaults with a warning
  * the layer-mode map (compress_ratios, kv_source_layer_ids, index_source_layer_ids, engram_layer_ids) and, per layer,
    that exactly the expected tensors exist with the expected ggml type and dims: 1,006 for the mxxm-t GGUF
  * tensor offsets: aligned, ascending, not overlapping, and (real files) inside the file
  * the routed experts are MXFP4 (the port's decided format): anything else is REFUSED unless --allow-non-mxfp4
  * Engram constants: primes are consecutive primes, per-layer sums == table rows, offsets, multipliers
  * the optional DSpark sidecar (arch `dflash`, 78 tensors)
Then it prints the memory plan for the box given by --vram-gib / --ram-gib / --numa-nodes (memplan.py).

Exit status: 0 ok (warnings allowed), 2 the model does not satisfy the contract, 1 bad usage or unreadable input.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ds41_spec as S  # noqa: E402
import gguf_io as G  # noqa: E402
import memplan  # noqa: E402

FORMAT = "ds41-manifest/1"


# ---------------------------------------------------------------------------------------------- name handling
def canonical(name: str):
    """File tensor name -> (canonical name, spelled-as-alias, ignorable)."""
    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not m:
        return name, False, False
    layer, suf = m.groups()
    if suf in S.IGNORED_SUFFIXES:
        return name, False, True
    for canon, aliases in S.TENSOR_ALIASES.items():
        if suf in aliases:
            return f"blk.{layer}.{canon}", True, False
    return name, False, False


def _jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
        return str(v)
    return v


# ---------------------------------------------------------------------------------------------- the analysis
def _check_shards(main, F, check_files: bool):
    first = main[0]
    count = first.kv.get("split.count")
    nos = [s.kv.get("split.no") for s in main]
    if len(main) == 1 and count in (None, 1):
        pass
    else:
        want = int(count) if count is not None else len(main)
        if len(main) != want:
            missing = sorted(set(range(want)) - {n for n in nos if isinstance(n, int)})
            F.error("SHARD_COUNT", f"split.count = {want} but {len(main)} shard file(s) were given "
                                   f"(missing split.no {missing})")
        elif sorted(n for n in nos if isinstance(n, int)) != list(range(want)):
            F.error("SHARD_COUNT", f"split.no values {nos} are not 0..{want - 1}")
        for s in main:
            c = s.kv.get("split.count")
            if c is not None and c != want:
                F.error("SHARD_COUNT", f"{s.file}: split.count = {c}, expected {want}")
    total = sum(len(s.tensors) for s in main)
    tcount = first.kv.get("split.tensors.count")
    if tcount is not None and tcount != total:
        F.error("SHARD_TENSORS", f"split.tensors.count = {tcount} but the shards hold {total} tensors")
    _check_tensor_tables(main, check_files, F.error, F.warn)


def _check_tensor_tables(shards, check_files: bool, error, warn):
    """Offsets aligned, ascending, not overlapping; sizes known; (real files) every tensor inside its file."""
    for s in shards:
        prev_end = 0
        for t in sorted(s.tensors, key=lambda t: t.offset):
            if t.nbytes is None:
                error("TENSOR_SIZE", f"{s.file}: `{t.name}` type {t.type} dims {list(t.dims)} has no known byte size "
                                     "(unknown type or row not a whole number of blocks)")
                continue
            if t.offset % s.alignment:
                error("OFFSET_ALIGN", f"{s.file}: `{t.name}` offset {t.offset} is not a multiple of the "
                                      f"alignment {s.alignment}")
            if t.offset < prev_end:
                error("OFFSET_OVERLAP", f"{s.file}: `{t.name}` starts at {t.offset}, inside the previous tensor "
                                        f"(ends {prev_end})")
            elif t.offset > prev_end:
                warn("OFFSET_GAP", f"{s.file}: {t.offset - prev_end} unused bytes before `{t.name}`")
            prev_end = max(prev_end, G.align_up(t.offset + t.nbytes, s.alignment))
            if check_files and t.abs_offset + t.nbytes > s.size:
                error("SHARD_TRUNCATED", f"{s.file} is {s.size} bytes but `{t.name}` ends at "
                                         f"{t.abs_offset + t.nbytes}: truncated or still downloading?")


def _check_sidecar(sidecar, cfg, vocab, relaxed, F, level, check_files):
    """DSpark sidecar: reported as warnings unless the caller requires it.  -> (tensor map, bytes)."""
    emit = F.error if level == "error" else F.warn
    _check_tensor_tables(sidecar, check_files, emit, F.warn)
    kv = sidecar[0].kv
    for key, want in (("block_count", 3), ("expert_count", 128), ("expert_used_count", 3), ("block_size", 5),
                      ("target_layers", [37, 38, 39])):
        got = kv.get(f"{S.SIDECAR_ARCH}.{key}")
        if got != want and cfg is not None and cfg.n_layer == 40:
            emit("SIDECAR_META", f"{S.SIDECAR_ARCH}.{key} = {got}, expected {want}")
    tens = {}
    for s in sidecar:
        for t in s.tensors:
            tens[t.name] = t
    if cfg is None:
        return tens, sum(t.nbytes or 0 for t in tens.values())
    specs = S.sidecar_specs(cfg, vocab)
    want_names = {sp.name for sp in specs}
    for sp in specs:
        t = tens.get(sp.name)
        if t is None:
            emit("SIDECAR_TENSOR", f"sidecar tensor `{sp.name}` is missing")
            continue
        if tuple(t.dims) != sp.dims:
            emit("SIDECAR_TENSOR", f"sidecar `{sp.name}` has dims {list(t.dims)}, expected {list(sp.dims)}")
        ok = (t.type == sp.strict) if not relaxed else (t.type == "F32" if sp.klass == "float" else t.nbytes is not None)
        if not ok:
            emit("SIDECAR_TENSOR", f"sidecar `{sp.name}` is {t.type}, expected {sp.strict}")
    for n in sorted(set(tens) - want_names):
        emit("SIDECAR_TENSOR", f"unexpected sidecar tensor `{n}`")
    return tens, sum(t.nbytes or 0 for t in tens.values())


def analyze(files, *, allow_non_mxfp4: bool = False, geometry: str = "ds41", require_dspark: bool = False,
            strict_metadata: bool = False, source: str = ""):
    """files: [ShardRec] (real or from a headers JSON, DSpark sidecar included) -> (manifest dict, Findings)."""
    F = S.Findings()
    from_json = all(s.from_json for s in files)
    sidecar = [s for s in files if s.kv.get("general.architecture") == S.SIDECAR_ARCH]
    main = [s for s in files if s not in sidecar]
    manifest: dict = {"format": FORMAT, "source": source, "geometry": geometry, "from_headers_json": from_json}
    if not main:
        F.error("SHARD_NONE", "no deepseek41 shard among the inputs")
        return manifest, F
    main.sort(key=lambda s: (s.kv.get("split.no") if isinstance(s.kv.get("split.no"), int) else 0))
    for i, s in enumerate(main):
        s.index = i
        for t in s.tensors:
            t.shard = i
    first = main[0]
    arch = first.kv.get("general.architecture")
    manifest["architecture"] = arch
    if arch != S.ARCH:
        F.error("ARCH", f"general.architecture = {arch!r} in {first.file}; this port reads `{S.ARCH}` "
                        "(is shard 1 -- the one holding the metadata -- among the inputs?)")
        return manifest, F
    _check_shards(main, F, check_files=not from_json)

    # ---- metadata
    cfg, notes = S.resolve_config(first.kv, S.ARCH, F, geometry)
    missing_fields = [k.field for k in S.KEYS if not hasattr(cfg, k.field)]
    if missing_fields:
        F.error("META_INCOMPLETE", f"cannot check the tensors: metadata incomplete ({', '.join(missing_fields)})")
        return manifest, F

    # ---- tensors by canonical name
    by_canon: dict = {}
    aliases_used: dict = {}
    ignored: list = []
    for s in main:
        for t in s.tensors:
            canon, aliased, ign = canonical(t.name)
            if ign:
                ignored.append(t.name)
                continue
            if canon in by_canon:
                F.error("TENSOR_DUP", f"tensor `{canon}` appears twice ({by_canon[canon].name} and {t.name})")
                continue
            by_canon[canon] = t
            if aliased:
                aliases_used[canon] = t.name
    if aliases_used:
        rev = {a: c for c, als in S.TENSOR_ALIASES.items() for a in als}
        kinds = collections.Counter(re.sub(r"^blk\.\d+\.", "", v) for v in aliases_used.values())
        F.info("TENSOR_ALIAS", "tensor spellings accepted as aliases: " + ", ".join(
            f"blk.N.{n} x{c} (read as blk.N.{rev[n]})" for n, c in sorted(kinds.items())))
    if ignored:
        F.info("TENSOR_IGNORED", f"{len(ignored)} tensor(s) ignored (not used by the text-only port): "
                                 f"{re.sub(r'^blk[.][0-9]+[.]', '', ignored[0])} x{len(ignored)}")

    e_rows = {}
    for canon, t in by_canon.items():
        m = re.match(r"^blk\.(\d+)\.engram_embed\.weight$", canon)
        if m and len(t.dims) == 2:
            e_rows[int(m.group(1))] = t.dims[1]
    tok = by_canon.get("token_embd.weight")
    vocab = tok.dims[1] if tok is not None and len(tok.dims) == 2 else None
    if vocab is None:
        F.error("TENSOR_MISSING", "`token_embd.weight` is missing; cannot size the vocabulary")
        vocab = S.TOKENIZER_VOCAB
    if geometry == "ds41" and vocab != S.TOKENIZER_VOCAB:
        F.error("META_VALUE", f"vocabulary {vocab} (token_embd), the port's geometry requires {S.TOKENIZER_VOCAB}")
    for key in ("tokenizer.ggml.tokens", "tokenizer.ggml.token_type"):
        n = G.array_len(first.kv.get(key))
        if n is None:
            F.error("TOKENIZER", f"`{key}` is missing")
        elif n != vocab:
            F.error("TOKENIZER", f"`{key}` has {n} entries, the vocabulary is {vocab}")
    if "tokenizer.ggml.merges" not in first.kv:
        F.warn("TOKENIZER", "`tokenizer.ggml.merges` is missing")
    token_map = first.kv.get(f"{S.ARCH}.engram.token_map")
    tm_len = G.array_len(token_map)
    if tm_len is None:
        F.error("ENGRAM_META", f"`{S.ARCH}.engram.token_map` is missing")
    elif tm_len != vocab:
        F.error("ENGRAM_META", f"`{S.ARCH}.engram.token_map` has {tm_len} entries, the vocabulary is {vocab}")
    elif isinstance(token_map, list):
        lo, hi = min(token_map), max(token_map)
        derived = f"{S.ARCH}.engram.compressed_vocab_size" in notes["derived"]
        if lo < 0 or hi + 1 > cfg.e_cvocab and not derived:
            F.error("ENGRAM_META", f"token_map ids span [{lo}, {hi}], compressed_vocab_size is {cfg.e_cvocab}")
        elif derived:
            cfg.e_cvocab = hi + 1
            F.info("META_DERIVED", f"engram.compressed_vocab_size = {hi + 1} taken from the token_map")
        elif hi + 1 != cfg.e_cvocab:
            F.warn("ENGRAM_META", f"token_map max id + 1 = {hi + 1}, compressed_vocab_size is {cfg.e_cvocab}")

    # ---- layer modes and expected tensors
    modes = S.layer_modes(cfg, F)
    specs = S.tensor_specs(cfg, modes, vocab, e_rows)
    spec_by = {sp.name: sp for sp in specs}
    found = {n: t for n, t in by_canon.items() if n in spec_by}
    exp_types = sorted({t.type for n, t in found.items() if spec_by[n].klass == "expert" and t.type != "MXFP4"})
    if exp_types:
        msg = (f"routed experts are {'/'.join(exp_types)}, not MXFP4: the port's decided expert format is MXFP4 "
               "(docs/deepseek/PLAN.md section 1; the expert blob layout and every expert kernel assume 17-byte "
               "MXFP4 blocks)")
        if allow_non_mxfp4:
            F.warn("EXPERT_TYPE", msg + " -- validating anyway (--allow-non-mxfp4)")
        else:
            F.error("EXPERT_TYPE", msg + ". Use an MXFP4 GGUF (mxxm-t/DeepSeek-V4.1-Flash-GGUF) or pass "
                                         "--allow-non-mxfp4 to validate the file anyway")
    relaxed = bool(exp_types)
    for sp in specs:
        t = by_canon.get(sp.name)
        if t is None:
            F.error("TENSOR_MISSING", f"tensor `{sp.name}` is missing")
            continue
        if tuple(t.dims) != sp.dims:
            F.error("TENSOR_SHAPE", f"`{sp.name}` has dims {list(t.dims)}, expected {list(sp.dims)}")
        if relaxed:
            ok = (t.type == "F32") if sp.klass == "float" else (t.nbytes is not None)
        else:
            ok = t.type == sp.strict
        if not ok:
            if relaxed:
                F.error("TENSOR_TYPE", f"`{sp.name}` is {t.type}, expected F32" if sp.klass == "float"
                        else f"`{sp.name}` is {t.type}, a type this tool cannot size")
            else:
                F.error("TENSOR_TYPE", f"`{sp.name}` is {t.type}, expected {sp.strict}")
    for n in sorted(set(by_canon) - set(spec_by)):
        F.error("TENSOR_UNEXPECTED", f"tensor `{n}` ({by_canon[n].type} {list(by_canon[n].dims)}) is not part of the "
                                     "deepseek41 contract")

    S.check_engram_constants(cfg, first.kv, S.ARCH, e_rows, F, geometry)

    if strict_metadata:      # a key that had to be derived from the config.json defaults is an error
        for f in F.items:
            if f.code == "META_DERIVED" and f.level == "warn":
                f.level, f.code = "error", "META_MISSING"

    # ---- sidecar
    sc_tensors, sc_bytes = {}, 0
    if sidecar:
        sc_tensors, sc_bytes = _check_sidecar(sidecar, cfg, vocab, relaxed, F, "error" if require_dspark else "warn",
                                         not from_json)
    else:
        (F.error if require_dspark else F.info)("SIDECAR_ABSENT", "no DSpark sidecar among the inputs (optional; "
                                                                  "drafting is the last milestone)")

    # ---- groups, totals
    groups: dict = collections.Counter()
    for n, t in by_canon.items():
        groups[spec_by[n].group if n in spec_by else "other"] += t.nbytes or 0
    exps0 = [by_canon.get(f"blk.{m['layer']}.ffn_{k}_exps.weight") for m in modes[:1] for k in ("gate", "up", "down")]
    expert_bytes = (sum(t.nbytes for t in exps0 if t is not None and t.nbytes) // cfg.n_expert) if cfg.n_expert else 0
    n_experts_total = sum(1 for m in modes if by_canon.get(f"blk.{m['layer']}.ffn_gate_exps.weight")) * cfg.n_expert
    e0 = by_canon.get(f"blk.{cfg.engram_layers[0]}.engram_embed.weight") if cfg.engram_layers else None
    e_row_bytes = (e0.nbytes // e0.dims[1]) if e0 is not None and e0.nbytes and e0.dims[1] else None
    types_hist = collections.Counter(t.type for t in by_canon.values())

    manifest.update({
        "profile": "non-mxfp4 (allowed)" if (relaxed and allow_non_mxfp4) else "mxfp4",
        "config": _jsonable(vars(cfg)),
        "aliases": {"metadata": notes["aliases"], "tensors": aliases_used},
        "derived_metadata": _jsonable(notes["derived"]),
        "ignored_tensors": ignored,
        "layers": modes,
        "vocab": vocab,
        "shards": [{"index": s.index, "file": s.file, "path": s.path, "size": s.size, "alignment": s.alignment,
                    "data_start": s.data_start, "data_bytes": s.data_bytes, "n_tensors": len(s.tensors)}
                   for s in main],
        "tensors": {n: {"shard": t.shard, "file": main[t.shard].file, "file_name": t.name, "offset": t.offset,
                        "abs_offset": t.abs_offset,
                        "type": t.type, "dims": list(t.dims), "nbytes": t.nbytes,
                        "group": spec_by[n].group if n in spec_by else "other",
                        "layer": spec_by[n].layer if n in spec_by else None}
                    for n, t in by_canon.items()},
        "totals": {
            "tensors": len(by_canon), "file_tensors": len(by_canon) + len(ignored), "expected_tensors": len(specs), "bytes": sum(t.nbytes or 0 for t in by_canon.values()),
            "by_group": dict(groups), "by_type": dict(types_hist),
            "expert_bytes_each": expert_bytes, "experts_total": n_experts_total,
            "expert_blob_formula": (f"2 x {cfg.ff} x {cfg.hidden // 32 * 17} + {cfg.hidden} x {cfg.ff // 32 * 17}"
                                    if not relaxed else None),
            "engram_row_bytes": e_row_bytes, "engram_rows": {str(k): v for k, v in e_rows.items()},
            "sidecar_tensors": len(sc_tensors), "sidecar_bytes": sc_bytes,
        },
        "sidecar": ({"shards": [{"file": s.file, "path": s.path, "size": s.size, "data_start": s.data_start,
                                 "alignment": s.alignment} for s in sidecar],
                     "tensors": {n: {"shard": 0, "file": sidecar[0].file, "offset": t.offset,
                                     "abs_offset": t.abs_offset, "type": t.type,
                                     "dims": list(t.dims), "nbytes": t.nbytes} for n, t in sc_tensors.items()}}
                    if sidecar else None),
        "ok": not F.errors,
    })
    return manifest, F


def make_plan(manifest: dict, **kw) -> dict:
    cfg = manifest["config"]
    t = manifest["totals"]
    return memplan.plan(
        t["by_group"], expert_bytes=t["expert_bytes_each"], n_experts=t["experts_total"],
        engram_row_bytes=t["engram_row_bytes"],
        engram_rows_per_token=(cfg["e_ngram"] - 1) * cfg["e_heads"] * len(cfg["engram_layers"]),
        dspark_bytes=t["sidecar_bytes"], **kw)


# ---------------------------------------------------------------------------------------------- CLI
def collect_inputs(paths, dspark: str = "auto"):
    """-> (list of ShardRec, source description)."""
    paths = [Path(p) for p in paths]
    if any(p.suffix in (".json", ".gz") for p in paths):
        if len(paths) != 1:
            raise SystemExit("give either one headers JSON or .gguf files, not both")
        return G.load_headers_json(paths[0]), f"headers JSON {paths[0]}"
    files = G.expand_split(paths)
    if dspark == "auto":
        dirs = {p.parent for p in files}
        for d in dirs:
            for extra in sorted(d.glob("*DSpark*.gguf")):
                if extra not in files:
                    files.append(extra)
    if not files:
        raise SystemExit("no .gguf files found")
    recs = []
    for i, p in enumerate(files):
        try:
            recs.append(G.load_shard(p, i))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"cannot read {p}: {exc}")
    return recs, f"{len(files)} GGUF file(s) from {files[0].parent}"


def format_report(manifest: dict, F: S.Findings, plan_text: str | None, max_per_code: int = 12) -> str:
    L: list = []
    w = L.append
    t = manifest.get("totals")
    w(f"source: {manifest.get('source')}")
    if "shards" in manifest:
        sh = manifest["shards"]
        known = all(s["data_start"] is not None for s in sh)
        w(f"architecture {manifest['architecture']}, {len(sh)} shard(s)"
          + (f" + {len(manifest['sidecar']['shards'])} DSpark sidecar" if manifest.get("sidecar") else "")
          + f", {t['tensors']} tensors (contract expects {t['expected_tensors']}"
          + (f"; {t['file_tensors']} in the files, {t['file_tensors'] - t['tensors']} ignored" if t["file_tensors"] != t["tensors"] else "")
          + f"), alignment {sh[0]['alignment']}, "
            + ("absolute data offsets computed" if known else "data section starts unknown (headers JSON): "
                                                              "offsets are relative to each shard's data section"))
        lm = collections.defaultdict(list)
        for m in manifest["layers"]:
            lm[m["mode"]].append(m["layer"])
        w("layer modes: " + " | ".join(f"{k} {','.join(map(str, v))}" for k, v in
                                       sorted(lm.items(), key=lambda kv: kv[1][0])))
        w("tensor types: " + ", ".join(f"{k} x{v}" for k, v in sorted(t["by_type"].items(), key=lambda kv: -kv[1])))
        g = t["by_group"]
        n_exp = t["experts_total"]
        w(f"routed experts: {g.get('expert', 0):,} B = {n_exp} x {t['expert_bytes_each']:,} B"
          + (f"  [{t['expert_blob_formula']}]" if t["expert_blob_formula"] else ""))
        if g.get("engram_embed"):
            w(f"Engram tables:  {g['engram_embed']:,} B, rows {', '.join(f'{k}: {v:,}' for k, v in t['engram_rows'].items())}"
              f" of {t['engram_row_bytes']} B")
        dense = sum(v for k, v in g.items() if k not in memplan.NON_DENSE)
        w(f"dense (GPU) {dense:,} B, token_embd {g.get('embd', 0):,} B, all tensors {t['bytes']:,} B"
          + (f", DSpark sidecar {t['sidecar_bytes']:,} B ({t['sidecar_tensors']} tensors)" if t["sidecar_tensors"] else ""))
    by = collections.defaultdict(list)
    for f in F.items:
        by[(f.level, f.code)].append(f)
    for level in ("error", "warn", "info"):
        for (lv, code), items in by.items():
            if lv != level:
                continue
            tag = {"error": "ERROR", "warn": "warn ", "info": "info "}[lv]
            for f in items[:max_per_code]:
                w(f"  [{tag}] {code}: {f.text}")
            if len(items) > max_per_code:
                w(f"  [{tag}] {code}: ... and {len(items) - max_per_code} more")
    if plan_text:
        w("")
        w(plan_text)
    w("")
    w(("RESULT: OK" if not F.errors else f"RESULT: REFUSED ({len(F.errors)} error(s))")
      + f", {len(F.warnings)} warning(s)")
    return "\n".join(L)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".gguf file(s) / directory / headers .json[.gz]")
    ap.add_argument("--out", help="write the manifest JSON here")
    ap.add_argument("--geometry", choices=("ds41", "self"), default="ds41",
                    help="ds41: require the compile-time geometry (default); self: dims from the file's own metadata")
    ap.add_argument("--allow-non-mxfp4", action="store_true", help="validate a file whose routed experts are not MXFP4")
    ap.add_argument("--dspark", choices=("auto", "none"), default="auto", help="pick up a *DSpark*.gguf next to the shards")
    ap.add_argument("--require-dspark", action="store_true", help="treat a missing or bad DSpark sidecar as an error")
    ap.add_argument("--strict-metadata", action="store_true",
                    help="treat a metadata key derived from the config.json defaults as an error (mxxm-t has them all)")
    ap.add_argument("--no-plan", action="store_true", help="skip the memory plan")
    g = ap.add_argument_group("memory plan (defaults: the target box)")
    g.add_argument("--vram-gib", type=float, default=32.0)
    g.add_argument("--ram-gib", type=float, default=384.0)
    g.add_argument("--numa-nodes", type=int, default=2)
    g.add_argument("--ctx-tokens", type=int, default=131072, help="KV cache length to reserve")
    g.add_argument("--kv-bytes-per-token", type=int, default=3200, help="fp16 KV, PLAN.md section 2")
    g.add_argument("--gpu-reserve-gib", type=float, default=3.0, help="activations, prompt buffers, CUDA context")
    g.add_argument("--os-gib", type=float, default=8.0)
    g.add_argument("--host-buffers-gib", type=float, default=3.0)
    g.add_argument("--kernel-overhead-pct", type=float, default=1.8, help="RAM the kernel keeps (384 -> ~377 GiB)")
    g.add_argument("--embd-on-gpu", action="store_true", help="put token_embd in VRAM instead of RAM")
    g.add_argument("--dspark-on-cpu", action="store_true", help="count the DSpark sidecar against RAM")
    g.add_argument("--cache-gib", type=float, help="also report how many experts this much VRAM holds")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files, source = collect_inputs(args.inputs, args.dspark)
    except SystemExit as exc:
        print(f"manifest.py: {exc}", file=sys.stderr)
        return 1
    manifest, F = analyze(files, allow_non_mxfp4=args.allow_non_mxfp4, geometry=args.geometry,
                          require_dspark=args.require_dspark, strict_metadata=args.strict_metadata, source=source)
    plan_text = None
    if "totals" in manifest and not args.no_plan:
        plan = make_plan(manifest, vram_gib=args.vram_gib, ram_gib=args.ram_gib, numa_nodes=args.numa_nodes,
                         ctx_tokens=args.ctx_tokens, kv_bytes_per_token=args.kv_bytes_per_token,
                         gpu_reserve_gib=args.gpu_reserve_gib, os_gib=args.os_gib,
                         host_buffers_gib=args.host_buffers_gib, kernel_overhead_pct=args.kernel_overhead_pct,
                         embd_on_gpu=args.embd_on_gpu, dspark_on_cpu=args.dspark_on_cpu, cache_gib=args.cache_gib)
        manifest["memory_plan"] = plan
        plan_text = memplan.format_plan(plan, n_layers=manifest["config"]["n_layer"])
    manifest["findings"] = [{"level": f.level, "code": f.code, "text": f.text} for f in F.items]
    print(format_report(manifest, F, plan_text))
    if args.out:
        Path(args.out).write_text(json.dumps(_jsonable(manifest), indent=1) + "\n")
        print(f"manifest written to {args.out}")
    return 0 if not F.errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
