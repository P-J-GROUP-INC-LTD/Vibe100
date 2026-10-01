"""Harness: run the OFFICIAL PyTorch reference (third_party/deepseek-v41-flash-reference/inference/model.py) on CPU.

 * `kernel` (TileLang) is replaced by `kernel_stub` (pure torch); the unvendored `vision` / `image_processor` modules
   model.py imports are stubbed; world_size/rank stay 1 (torch.distributed is never initialised).
 * `build_tiny(cfg, seed)` instantiates the official `Transformer` with a reduced `ModelArgs` of the real structure and
   fills it with random weights *in the released storage formats*: FP8-e4m3 + E8M0 32x32-block scales for attention /
   shared experts / indexer wq_b / Engram wkv, packed FP4 e2m1 + E8M0 per-32 for the routed experts, FP8 rows for the
   Engram tables, bf16-representable values for the bf16 tensors (then widened to fp32: the CPU model runs in fp32).
 * `export(model)` dequantises every parameter with the ORACLE's own decoders (ref/ds41/quant.py) into a
   `DictWeights`, so the oracle is fed the very same numbers.
The vendored files are not touched.
"""
from __future__ import annotations

import dataclasses
import pathlib
import sys

import numpy as np
import torch

from ref.ds41 import quant as Q
from ref.ds41.config import Config
from ref.ds41.engram import build_layout, _bytes_to_unicode
from ref.ds41.weights import DictWeights

from . import kernel_stub

REF_DIR = pathlib.Path(__file__).resolve().parents[3] / "third_party" / "deepseek-v41-flash-reference" / "inference"
_official = None


def load_official():
    """Import the vendored model.py / engram.py with the stubs installed (idempotent)."""
    global _official
    if _official is None:
        kernel_stub.install()
        sys.path.insert(0, str(REF_DIR))
        import engram as engram_mod  # noqa: WPS433
        import model as model_mod     # noqa: WPS433
        _official = (model_mod, engram_mod)
    return _official


# ---------------------------------------------------------------------------------------------------------------
# a synthetic byte-level tokenizer (exercises the compressed-token-map code path without the real vocabulary)
# ---------------------------------------------------------------------------------------------------------------

class _Backend:
    def __init__(self, tokens):
        self.tokens = tokens

    def decode(self, ids, skip_special_tokens=False):
        from ref.ds41.engram import bytelevel_decode_token
        return "".join(bytelevel_decode_token(self.tokens[i]) for i in ids)

    def id_to_token(self, i):
        return self.tokens[i]


class FakeTokenizer:
    """`len()`, `.backend_tokenizer.decode/id_to_token` - all that engram.build_compressed_token_map touches."""

    def __init__(self, tokens):
        self.backend_tokenizer = _Backend(tokens)
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)


def synthetic_tokens(n: int, seed: int = 0) -> list[str]:
    """n byte-level token strings: words in several casings/accents/spacings (so that many collapse together after
    normalisation), single bytes (incl. partial UTF-8 ones), and a few specials."""
    rng = np.random.default_rng(seed)
    b2u = _bytes_to_unicode()
    enc = lambda s: "".join(b2u[b] for b in s.encode("utf-8"))
    words = ["the", "The", "THE", "cafe", "café", "CAFÉ", "naïve", "naive", "x", "X", "ß", "İ", "ǅ", "fi", "ﬁ", "42",
             "Straße", "strasse", "δ", "Δ", "Ω", "ω", "a", "A", "ab", "AB", "é", "é"]
    toks = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>"]        # ids 0..2 (pad id 2)
    toks += [enc(chr(i)) for i in range(33, 127)]
    toks += [b2u[b] for b in (0x80, 0xC3, 0xA9, 0xE2, 0x82)]                      # lone / partial UTF-8 bytes
    toks += [enc(" "), enc("  "), enc("\n"), enc("\t"), enc(" \n")]
    while len(toks) < n:
        w = words[rng.integers(len(words))]
        sp = " " * int(rng.integers(0, 2))
        toks.append(enc(sp + w + ("" if rng.random() < 0.7 else str(rng.integers(0, 4)))))
    return toks[:n]


def tiny_config(token_strings=None, **over) -> Config:
    """Config.tiny() with the Engram sizes that follow from the (synthetic) tokenizer and the prime search."""
    from ref.ds41.engram import build_compressed_token_map_from_tokens
    cfg = Config.tiny(**over)
    toks = token_strings or synthetic_tokens(cfg.vocab_size)
    _, n_comp = build_compressed_token_map_from_tokens(toks)
    layout = build_layout(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, cfg.engram_vocab_size)
    cfg.engram_compressed_vocab_size = n_comp
    cfg.engram_num_embeddings = layout.num_embeddings
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# build + randomise the official model
# ---------------------------------------------------------------------------------------------------------------

def model_args(cfg: Config):
    mm, _ = load_official()
    names = {f.name for f in dataclasses.fields(mm.ModelArgs)}
    kw = {k: v for k, v in dataclasses.asdict(cfg).items() if k in names}
    kw.update(max_batch_size=1, max_seq_len=cfg.max_seq_len, dtype="fp8", expert_dtype="fp4", temperature=0.0,
              n_mtp_layers=0, dspark_block_size=0, compress_ratios=tuple(cfg.compress_ratios))
    return mm.ModelArgs(**kw)


def _quant_fp8_block(w: torch.Tensor):
    """float [N, K] -> (fp8 e4m3 tensor, E8M0 scale [ceil(N/32), ceil(K/32)]) with power-of-two block scales."""
    n, k = w.shape
    npad, kpad = -n % 32, -k % 32
    wp = torch.nn.functional.pad(w, (0, kpad, 0, npad))
    blocks = wp.view(wp.shape[0] // 32, 32, wp.shape[1] // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12)
    e = torch.ceil(torch.log2(amax / 448.0))
    s = torch.exp2(e)
    q = (blocks / s[:, None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    q = q.reshape(wp.shape)[:n, :k].contiguous()
    sc = torch.empty(s.shape, dtype=torch.float8_e8m0fnu)
    sc.view(torch.uint8).copy_((e + 127).to(torch.uint8))
    return q, sc


def _quant_fp4_rows(w: torch.Tensor):
    """float [N, K] -> (packed e2m1 [N, K/2] as float4_e2m1fn_x2, E8M0 [N, K/32]) one power-of-two scale per 32 along K."""
    n, k = w.shape
    blocks = w.view(n, k // 32, 32)
    amax = blocks.abs().amax(dim=-1).clamp_min(1e-12)
    e = torch.ceil(torch.log2(amax / 6.0))
    s = torch.exp2(e)
    v = (blocks / s.unsqueeze(-1)).numpy()
    q = Q.round_e2m1(v.astype(np.float32))                                       # values on the e2m1 grid
    grid = Q.FP4_E2M1_TABLE[:8]
    codes = np.abs(q)[..., None] == grid
    code = codes.argmax(-1).astype(np.uint8) | ((q < 0).astype(np.uint8) << 3)
    packed = Q.pack_fp4_official(code.reshape(n, k))
    pw = torch.empty((n, k // 2), dtype=torch.float4_e2m1fn_x2)
    pw.view(torch.uint8).copy_(torch.from_numpy(packed))
    sc = torch.empty((n, k // 32), dtype=torch.float8_e8m0fnu)
    sc.view(torch.uint8).copy_((e + 127).to(torch.uint8))
    return pw, sc


def _bf16r(t: torch.Tensor) -> torch.Tensor:
    return t.bfloat16().float()


@torch.no_grad()
def fill_linear(mod, rnd, std: float, widen: bool = True):
    """Fill an official Linear in its storage format with N(0, std^2) values (fp8: 32x32 blocks, fp4: 1x32 along K)."""
    w = mod.weight
    out, inp = mod.out_features, mod.in_features
    if w.dtype == torch.float8_e4m3fn:
        q, sc = _quant_fp8_block(rnd(out, inp) * std)
        w.view(torch.uint8).copy_(q.view(torch.uint8))
        mod.scale.view(torch.uint8).copy_(sc.view(torch.uint8))
    elif w.dtype == torch.float4_e2m1fn_x2:
        pw, sc = _quant_fp4_rows(rnd(out, inp) * std)
        w.view(torch.uint8).copy_(pw.view(torch.uint8))
        mod.scale.view(torch.uint8).copy_(sc.view(torch.uint8))
    else:
        w.copy_(_bf16r(rnd(out, inp) * std).to(w.dtype))
        if widen and w.dtype == torch.bfloat16:
            mod.weight.data = mod.weight.data.float()


def dequant_linear(mod) -> np.ndarray:
    """Dense float32 [out, in] of an official Linear, decoded with the oracle's own decoders."""
    w = mod.weight.detach()
    if w.dtype == torch.float8_e4m3fn:
        return Q.dequant_fp8_block(w.view(torch.uint8).numpy(), mod.scale.detach().view(torch.uint8).numpy(), 32)
    if w.dtype == torch.float4_e2m1fn_x2:
        return Q.dequant_fp4_official(w.view(torch.uint8).numpy(), mod.scale.detach().view(torch.uint8).numpy())
    return w.float().numpy().copy()


@torch.no_grad()
def randomize(model, cfg: Config, seed: int = 0, gain: float = 1.0, widen: bool = True):
    """Fill every parameter with random values in its storage format (see module docstring)."""
    mm, _ = load_official()
    g = torch.Generator().manual_seed(seed)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float32)

    for name, mod in model.named_modules():
        if isinstance(mod, mm.Linear):
            fill_linear(mod, rnd, gain / np.sqrt(mod.in_features), widen)
    for name, p in model.named_parameters():
        if p.dtype in (torch.float8_e4m3fn, torch.float4_e2m1fn_x2, torch.float8_e8m0fnu):
            continue
        if name.endswith("embed.weight") and ".engram." not in name:
            p.copy_(_bf16r(rnd(*p.shape) * 0.7))
        elif name == "head.weight":
            p.copy_(_bf16r(rnd(*p.shape) / np.sqrt(p.shape[1]) * 2.0))
        elif name.endswith("norm.weight"):
            p.copy_(1.0 + 0.2 * rnd(*p.shape))
        elif name.endswith("attn_sink"):
            p.copy_(rnd(*p.shape) * 0.5)
        elif name.endswith("gate.weight"):
            p.copy_(_bf16r(rnd(*p.shape) / np.sqrt(p.shape[1]) * 2.0))
        elif name.endswith("gate.bias"):
            p.copy_(rnd(*p.shape) * 0.1)
        elif "hc_" in name and name.endswith("_fn"):
            p.copy_(rnd(*p.shape) / np.sqrt(p.shape[1]) * 2.0)
        elif "hc_" in name and name.endswith("_base"):
            p.copy_(rnd(*p.shape) * 0.5)
        elif "hc_" in name and name.endswith("_scale"):
            p.copy_(0.5 + 0.5 * torch.rand(*p.shape, generator=g))
        elif name.endswith("q_weight") or name.endswith("k_weight"):
            p.copy_(_bf16r(1.0 + 0.3 * rnd(*p.shape)))
    # Engram tables: FP8 rows (+ per-32 E8M0 scales), as in the checkpoint
    for name, mod in model.named_modules():
        if isinstance(mod, mm.ParallelEngramEmbedding):
            rows, dim = mod.weight.shape
            q, sc = [], []
            for blk in range(0, rows, 4096):
                w = rnd(min(4096, rows - blk), dim) * 0.5
                wb = w.view(w.shape[0], dim // 32, 32)
                amax = wb.abs().amax(-1).clamp_min(1e-12)
                e = torch.ceil(torch.log2(amax / 448.0))
                qq = (wb / torch.exp2(e).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).view(w.shape)
                q.append(qq)
                ss = torch.empty(e.shape, dtype=torch.float8_e8m0fnu)
                ss.view(torch.uint8).copy_((e + 127).to(torch.uint8))
                sc.append(ss)
            mod.weight.view(torch.uint8).copy_(torch.cat(q).view(torch.uint8))
            mod.scale.view(torch.uint8).copy_(torch.cat(sc).view(torch.uint8))
    # every remaining bf16 parameter (wo_a, weights_proj, wk, compressor.wkv at ratio 1) -> fp32 for the CPU run
    if widen:
        for p in model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.float()


def build_tiny(cfg: Config, token_strings=None, seed: int = 0, gain: float = 1.0, bf16: bool = False):
    """-> (official Transformer in eval mode, FakeTokenizer).  Weights are random but deterministic in `seed`.
    bf16=True builds it as deployed (torch default dtype bfloat16: bf16 activations and bf16-stored tensors); the
    caller must then run it under `torch.set_default_dtype(torch.bfloat16)` (see `bf16_default`)."""
    mm, _ = load_official()
    torch.manual_seed(seed)
    toks = token_strings or synthetic_tokens(cfg.vocab_size)
    tok = FakeTokenizer(toks)
    args = model_args(cfg)
    prev = torch.get_default_dtype()
    if bf16:
        torch.set_default_dtype(torch.bfloat16)
    try:
        model = mm.Transformer(args, tok)
        randomize(model, cfg, seed, gain, widen=not bf16)
    finally:
        torch.set_default_dtype(prev)
    model.eval()
    return model, tok


class bf16_default:
    """Context manager: torch default dtype = bfloat16 (what generate.py sets before running the model)."""

    def __enter__(self):
        self.prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)

    def __exit__(self, *a):
        torch.set_default_dtype(self.prev)


# ---------------------------------------------------------------------------------------------------------------
# export to the oracle
# ---------------------------------------------------------------------------------------------------------------

def export(model) -> DictWeights:
    """Dequantise every official parameter with the oracle's decoders; names are the official state-dict names."""
    mm, _ = load_official()
    t: dict[str, np.ndarray] = {}
    mods = dict(model.named_modules())
    for name, p in model.named_parameters():
        if name.endswith(".scale") and p.dtype == torch.float8_e8m0fnu:
            continue
        if name.endswith("engram.embed.scale"):
            continue
        if p.dtype == torch.float8_e4m3fn:
            mod_name = name.rsplit(".", 1)[0]
            mod = mods[mod_name]
            wb = p.detach().view(torch.uint8).numpy()
            sc = mod.scale.detach().view(torch.uint8).numpy()
            if isinstance(mod, mm.ParallelEngramEmbedding):
                vals = Q.fp8_e4m3_to_f32(wb).reshape(wb.shape[0], -1, 32) * Q.e8m0_to_f32(sc)[:, :, None]
                t[name] = vals.reshape(wb.shape).astype(np.float32)
            else:
                t[name] = Q.dequant_fp8_block(wb, sc, 32)
        elif p.dtype == torch.float4_e2m1fn_x2:
            mod = mods[name.rsplit(".", 1)[0]]
            t[name] = Q.dequant_fp4_official(p.detach().view(torch.uint8).numpy(),
                                             mod.scale.detach().view(torch.uint8).numpy())
        else:
            t[name] = p.detach().float().numpy().copy()
    return DictWeights(t)


# ---------------------------------------------------------------------------------------------------------------
# running both sides
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def run_official(model, ids: np.ndarray, n_prefill: int, quant=None):
    """Prefill ids[:n_prefill] then decode the rest one token at a time (teacher forcing).
    Returns (logits [len(ids), V] float32 for every position, trace {layer -> [per-call block outputs]})."""
    if quant is not None:
        kernel_stub.Flags.set(quant)
    head = model.head
    orig = type(head).forward
    head.forward = lambda x, full_logits=False: orig(head, x, full_logits=True)
    trace = {"block_out": {}, "engram_out": {}, "attn_out": {}, "ffn_out": {}}
    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(lambda m, a, o, i=i: trace["block_out"].setdefault(i, []).append(
            o[0].detach().clone().float().numpy()[0])))
        hooks.append(layer.attn.register_forward_hook(lambda m, a, o, i=i: trace["attn_out"].setdefault(i, []).append(
            o.detach().clone().float().numpy()[0])))
        hooks.append(layer.ffn.register_forward_hook(lambda m, a, o, i=i: trace["ffn_out"].setdefault(i, []).append(
            o.detach().clone().float().numpy()[0])))
        if layer.engram is not None:
            hooks.append(layer.engram.register_forward_hook(lambda m, a, o, i=i: trace["engram_out"].setdefault(
                i, []).append(o.detach().clone().float().numpy()[0])))
    try:
        t = torch.tensor(np.asarray(ids), dtype=torch.long).unsqueeze(0)
        outs = [model(t[:, :n_prefill], 0)[1][0].float().numpy()]
        for pos in range(n_prefill, t.shape[1]):
            outs.append(model(t[:, pos:pos + 1], pos)[1].reshape(1, -1).float().numpy())
        return np.concatenate(outs, axis=0), trace
    finally:
        for h in hooks:
            h.remove()
        del head.forward
        kernel_stub.Flags.set(type("Q", (), dict(linear_act=True, window_kv=True, compressed_kv=True, index=True)))


def run_oracle(model, ids: np.ndarray, n_prefill: int):
    """Same schedule on the oracle.  Returns (logits [len(ids), V] float32, trace {name: {layer: [per-call arrays]}})."""
    cache = model.new_cache()
    trace: dict = {}
    logits = []
    calls = []
    tr = {}
    out = model.forward(ids[:n_prefill], 0, cache, full_logits=True, trace=tr)
    calls.append(tr)
    logits.append(out)
    for pos in range(n_prefill, len(ids)):
        tr = {}
        logits.append(model.forward(ids[pos:pos + 1], pos, cache, full_logits=True, trace=tr))
        calls.append(tr)
    for key in ("block_out", "engram_out", "attn_out", "ffn_out"):
        trace[key] = {}
        for tr in calls:
            for l, v in tr.get(key, {}).items():
                trace[key].setdefault(l, []).append(v)
    return np.concatenate(logits, axis=0), trace


# ---------------------------------------------------------------------------------------------------------------
# reference quirk: stale `shared_attn.index_k`
# ---------------------------------------------------------------------------------------------------------------

def patch_index_k_fix(model):
    """Wrap every index-K owner's Indexer.forward so that it always scores against ITS OWN k_cache (model.py:560 reads
    `shared_attn.index_k`, which an owner only refreshes when its compressor emitted a latent this step).  This is the
    one-line behaviour the oracle implements by default; returns an `undo()` callable.  Not a vendored-file edit."""
    mm, _ = load_official()
    patched = []
    for layer in model.layers:
        idx = layer.attn.indexer
        if idx is not None and idx.owns_k:
            orig = idx.forward

            def fwd(*a, _orig=orig, _idx=idx, **k):
                mm.shared_attn.index_k = _idx.k_cache
                return _orig(*a, **k)

            idx.forward = fwd
            patched.append(idx)

    def undo():
        for idx in patched:
            del idx.forward

    return undo


# ---------------------------------------------------------------------------------------------------------------
# recording every fake-quant call (for the call-site / bit-exactness tests)
# ---------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def record_official_quant_calls(model, ids: np.ndarray, n_prefill: int):
    """Run prefill + decode on the official model with the stubs' fake-quant ON and record every quantiser call as
    (kind, input float32 array, dequantised output float32 array), in call order."""
    mm, _ = load_official()
    kernel_stub.Flags.set(type("Q", (), dict(linear_act=True, window_kv=True, compressed_kv=True, index=True)))
    rec = []
    oa, of4 = mm.act_quant, mm.fp4_act_quant

    def act_spy(x, bs=128, fmt=None, sd=torch.float32, inplace=False):
        xin = x.clone().float().numpy()
        out = oa(x, bs, fmt, sd, inplace)
        if inplace:
            rec.append(("fp8_window_kv", xin, x.clone().float().numpy()))
        else:
            q, s = out
            deq = (q.float().unflatten(-1, (-1, bs)) * s.float().unsqueeze(-1)).flatten(-2).numpy()
            rec.append(("fp8_linear_act", xin, deq))
        return out

    def fp4_spy(x, bs=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        xin = x.clone().float().numpy()
        out = of4(x, bs, inplace, scale_dtype)
        kind = "fp4_e4m3_compressed_kv" if scale_dtype == torch.float8_e4m3fn else "fp4_e8m0_index"
        rec.append((kind, xin, x.clone().float().numpy()))
        return out

    mm.act_quant, mm.fp4_act_quant = act_spy, fp4_spy
    try:
        t = torch.tensor(np.asarray(ids), dtype=torch.long).unsqueeze(0)
        model(t[:, :n_prefill], 0)
        for pos in range(n_prefill, t.shape[1]):
            model(t[:, pos:pos + 1], pos)
    finally:
        mm.act_quant, mm.fp4_act_quant = oa, of4
    return rec


class OracleQuantRecorder:
    """Context manager: record every call the oracle makes to its fake-quantisers as (kind, input, output)."""

    def __enter__(self):
        from ref.ds41 import attention as AT, ops as OPS
        self.rec, self._saved = [], []
        wrap = lambda fn, kind: (lambda x, *a, **k: self._call(fn, kind, x, *a, **k))
        for mod, name, kind in ((OPS, "act_quant_fp8", "fp8_linear_act"), (AT, "act_quant_fp8", "fp8_window_kv"),
                                (AT, "fp4_quant_e8m0", "fp4_e8m0_index"), (AT, "fp4_quant_e4m3", "fp4_e4m3_compressed_kv")):
            self._saved.append((mod, name, getattr(mod, name)))
            setattr(mod, name, wrap(getattr(mod, name), kind))
        return self

    def _call(self, fn, kind, x, *a, **k):
        out = fn(x, *a, **k)
        self.rec.append((kind, np.array(x, dtype=np.float64, copy=True), np.array(out, dtype=np.float64, copy=True)))
        return out

    def __exit__(self, *exc):
        for mod, name, fn in self._saved:
            setattr(mod, name, fn)
