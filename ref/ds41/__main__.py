"""python -m ref.ds41 <command>

  selfcheck [--tokenizer tokenizer.json]       Engram constants / token map vs the GGUF metadata (selfcheck.py)
  info GGUF...                                 config, layer-mode map and tensor-mapping report of a GGUF (all shards)
  run GGUF... --ids 1,2,3 [--decode N] [--quant exact|reference] [--dtype float32|float64] [--stale-index-k]
                                               [--max-seq-len 4096] [--npz out.npz]
        prefill the token ids, then greedy-decode N tokens; prints the top-5 logits of every step; --npz stores every
        step's logits and the per-layer block outputs (the oracle values a C++ engine's layer dumps compare against).
        Slow by design (NumPy, lazy dequantisation): meant for single-layer / few-token ground truth on real weights.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _cmd_selfcheck(a):
    from . import selfcheck
    res = selfcheck.run_checks(a.tokenizer)
    sys.exit(0 if all(ok for _, ok, d in res if not d.startswith("SKIPPED")) else 1)


def _cmd_info(a):
    from .config import describe_modes
    from .weights import load_from_gguf, validate_shapes
    w, cfg = load_from_gguf(a.gguf, strict=False)
    print(f"tensors mapped: {len(w.names())}; unmapped (ignored): {w.unmapped}")
    problems = validate_shapes(w, cfg)
    print("shape contract:", "OK" if not problems else problems[:10])
    print("layer modes:", describe_modes(cfg))
    print({k: v for k, v in vars(cfg).items() if k not in ("compress_ratios",)})


def _cmd_run(a):
    from .model import model_from_gguf
    from .quant import QuantConfig
    m = model_from_gguf(a.gguf, quant=QuantConfig.reference() if a.quant == "reference" else QuantConfig.exact(),
                        dtype=np.dtype(a.dtype), max_seq_len=a.max_seq_len, stale_index_k=a.stale_index_k)
    ids = np.array([int(x) for x in a.ids.split(",")], dtype=np.int64)
    cache = m.new_cache()
    trace: dict = {}
    logits = m.forward(ids, 0, cache, trace=trace)
    steps = [logits]
    traces = [trace]
    out = []
    for _ in range(a.decode):
        t = int(np.argmax(logits))
        out.append(t)
        trace = {}
        logits = m.decode(t, cache, trace=trace)
        steps.append(logits)
        traces.append(trace)
    for i, lg in enumerate(steps):
        top = np.argsort(-lg)[:5]
        print(f"step {i}: top5 {[(int(t), round(float(lg[t]), 4)) for t in top]}")
    print("generated:", out)
    if a.npz:
        arrs = {"ids": ids, "generated": np.array(out), "logits": np.stack(steps)}
        for s, tr in enumerate(traces):
            for l, v in tr.get("block_out", {}).items():
                arrs[f"step{s}.block_out.{l}"] = v
            for l, v in tr.get("router_idx", {}).items():
                arrs[f"step{s}.router_idx.{l}"] = v
        np.savez_compressed(a.npz, **arrs)
        print("wrote", a.npz)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ref.ds41", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selfcheck")
    s.add_argument("--tokenizer")
    s.set_defaults(fn=_cmd_selfcheck)
    s = sub.add_parser("info")
    s.add_argument("gguf", nargs="+")
    s.set_defaults(fn=_cmd_info)
    s = sub.add_parser("run")
    s.add_argument("gguf", nargs="+")
    s.add_argument("--ids", required=True)
    s.add_argument("--decode", type=int, default=0)
    s.add_argument("--quant", choices=("exact", "reference"), default="exact")
    s.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    s.add_argument("--max-seq-len", type=int, default=4096)
    s.add_argument("--stale-index-k", action="store_true")
    s.add_argument("--npz")
    s.set_defaults(fn=_cmd_run)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
