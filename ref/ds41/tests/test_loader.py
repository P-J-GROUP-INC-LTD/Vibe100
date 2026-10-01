"""The loader interface: `load_from_gguf(paths)` on a mini GGUF (tools/ds41/make_mini_gguf.py: real tensor names, real
GGML types, small dims), lazy per-expert access, dequantisation cross-checked against the independent codecs of
tools/ds41/ggml_codecs.py, the Engram constants carried by the file, and a full prefill/decode on the loaded model.
Also: the dict provider."""
import glob
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from ref.ds41 import DictWeights, Model, QuantConfig, load_from_gguf, model_from_gguf
from ref.ds41 import engram as E
from ref.ds41 import quant as Q
from ref.ds41.config import layer_modes
import re
from ref.ds41.weights import GGUFWeights, expected_shape, validate_shapes

REPO = pathlib.Path(__file__).resolve().parents[3]
MAKER = REPO / "tools" / "ds41" / "make_mini_gguf.py"

pytestmark = pytest.mark.skipif(not MAKER.exists(), reason="tools/ds41/make_mini_gguf.py not present")


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    d = tmp_path_factory.mktemp("mini_gguf")
    subprocess.run([sys.executable, str(MAKER), str(d), "--seed", "3", "--n-shards", "3"], check=True,
                   capture_output=True)
    paths = sorted(glob.glob(str(d / "*.gguf")))
    assert len(paths) == 3
    return paths


@pytest.fixture(scope="module")
def codecs():
    sys.path.insert(0, str(REPO / "tools" / "ds41"))
    try:
        import ggml_codecs
    except Exception as e:                                   # pragma: no cover
        pytest.skip(f"tools/ds41/ggml_codecs.py unavailable: {e}")
    return ggml_codecs


def test_load_config_and_names(mini):
    w, cfg = load_from_gguf(mini)
    assert isinstance(w, GGUFWeights)
    assert w.unmapped == []                                  # every tensor of the file has a canonical name
    assert cfg.n_layers == 8 and cfg.dim == 256 and cfg.n_routed_experts == 16 and cfg.hc_mult == 4
    assert [m.value for m in layer_modes(cfg)] == ["swa", "swa", "full", "reuse", "full", "reuse", "reindex", "reuse"]
    assert validate_shapes(w, cfg) == []
    # shard order does not matter; metadata comes from shard 0
    w2, cfg2 = load_from_gguf(list(reversed(mini)))
    assert cfg2 == cfg and w2.names() == w.names()
    # everything the model reads exists
    for l in range(cfg.n_layers):
        for n in ("attn_norm.weight", "ffn_norm.weight", "attn.wq_a.weight", "attn.wo_b.weight", "ffn.gate.weight",
                  "ffn.gate.bias", "ffn.shared_experts.w1.weight", "hc_attn_fn", "hc_ffn_scale", "ffn.experts.0.w1.weight"):
            assert w.has(f"layers.{l}." + n), (l, n)
    assert w.has("embed.weight") and w.has("head.weight") and w.has("norm.weight")


def test_shape_mismatch_is_detected(mini):
    """The geometry is a contract: a config that disagrees with the tensors yields problems (load_from_gguf raises on
    them in strict mode, 'refuse rather than mis-index')."""
    w, cfg = load_from_gguf(mini)
    assert validate_shapes(w, cfg) == []
    cfg.dim += 32
    problems = validate_shapes(w, cfg)
    assert problems and any("attn.wq_a.weight" in p for p in problems)
    cfg.dim -= 32
    cfg.n_routed_experts += 1
    assert any("ffn.experts.w1.weight" in p or "ffn.gate.weight" in p for p in validate_shapes(w, cfg))


def test_dequantisation_matches_independent_codecs(mini, codecs):
    w, cfg = load_from_gguf(mini)
    seen_types = set()
    for name in w.names():
        t = w.tensor_info(name)
        if len(t.ne) == 3 or t.type == "MXFP4" and "engram" in name:
            continue                                          # fused experts / Engram: checked below
        raw = np.asarray(t._map()[t.offset: t.offset + t.ne[0] // t.block[0] * t.block[1] * (t.ne[1] if len(t.ne) > 1 else 1)])
        ref = codecs.decode(t.type, raw).reshape(t.shape)
        got = w.get(name)
        assert got.shape == t.shape and np.array_equal(got, ref.astype(np.float32)), name
        seen_types.add(t.type)
    assert seen_types == {"F32", "BF16", "Q8_0"}


def test_lazy_expert_access(mini, codecs):
    w, cfg = load_from_gguf(mini)
    E_, ff, d = cfg.n_routed_experts, cfg.moe_inter_dim, cfg.dim
    for layer, e in ((0, 0), (3, 7), (7, E_ - 1)):
        w1, w3, w2 = w.expert(layer, e)
        assert w1.shape == (ff, d) and w3.shape == (ff, d) and w2.shape == (d, ff)
        raw = w.expert_raw(layer, e)
        assert raw["w1"].shape == (ff, d // 32 * 17) and raw["w2"].shape == (d, ff // 32 * 17)
        # the raw views are the exact GGUF bytes, and decoding them equals the loader's matrices
        assert np.array_equal(Q.dequant_mxfp4(raw["w1"]), w1) and np.array_equal(Q.dequant_mxfp4(raw["w2"]), w2)
        assert np.array_equal(codecs.decode("MXFP4", np.ascontiguousarray(raw["w3"]).reshape(-1)).reshape(ff, d), w3)
        # per expert: 3 slices of ff*row bytes (GPU blob = [gate][up][down])
        assert sum(r.size for r in raw.values()) == ff * (d // 32 * 17) * 2 + d * (ff // 32 * 17)
    # name-based access (what the Model uses) agrees
    assert np.array_equal(w.get("layers.3.ffn.experts.7.w1.weight"), w.expert(3, 7)[0])
    assert not np.array_equal(w.expert(3, 7)[0], w.expert(3, 8)[0])


def test_engram_rows_embedding_rows_and_head(mini, codecs):
    w, cfg = load_from_gguf(mini)
    t = w.tensor_info("layers.1.engram.embed.weight")
    assert t.type == "MXFP4" and t.shape == (cfg.engram_num_embeddings[0], cfg.engram_head_dim)
    idx = np.array([[0, 5, t.shape[0] - 1], [17, 17, 3]])
    rows = w.engram_rows(1, idx)
    assert rows.shape == (2, 3, cfg.engram_head_dim)
    full = Q.dequant_mxfp4(np.asarray(t._map()[t.offset: t.offset + t.shape[0] * t.row_bytes()]).reshape(t.shape[0], -1))
    assert np.array_equal(rows, full[idx])
    ids = np.array([0, 9, 9, 100])
    assert np.array_equal(w.rows("embed.weight", ids), w.get("embed.weight")[ids])
    x = np.random.default_rng(0).standard_normal((3, cfg.dim)).astype(np.float32)
    assert np.allclose(w.matmul_t("head.weight", x, chunk_rows=100), x @ w.get("head.weight").T, rtol=1e-5, atol=1e-5)


def test_engram_constants_in_the_file(mini):
    w, cfg = load_from_gguf(mini)
    ec = E.constants_from_gguf_metadata(w.metadata)
    assert ec["layout"].num_embeddings == tuple(cfg.engram_num_embeddings)        # sum of each layer's primes
    assert ec["token_map"].shape == (cfg.vocab_size,) and ec["token_map"].max() + 1 == cfg.engram_compressed_vocab_size
    # the file's own constants reproduce what derivation from its engram_vocab_size (1000) gives
    lay = E.build_layout(cfg.engram_layer_ids, cfg.engram_max_ngram_size, cfg.engram_n_heads, 1000)
    assert lay.primes == ec["layout"].primes
    assert np.array_equal(E.compute_hash_multipliers(cfg.engram_layer_ids, 4, cfg.engram_compressed_vocab_size),
                          ec["multipliers"])


def test_model_runs_on_the_gguf_and_prefill_equals_decode(mini):
    m = model_from_gguf(mini, dtype=np.float64, max_seq_len=48)
    cfg = m.cfg
    ids = np.random.default_rng(1).integers(3, cfg.vocab_size, 24)
    full = m.forward(ids, 0, m.new_cache(), full_logits=True)
    assert np.isfinite(full).all() and full.std() > 0.1 and len(set(full.argmax(1))) > 5
    cache = m.new_cache()
    inc = [m.forward(ids[:15], 0, cache, full_logits=True)]
    for p in range(15, 24):
        inc.append(m.forward(ids[p:p + 1], p, cache, full_logits=True))
    inc = np.concatenate(inc)
    assert np.abs(full - inc).max() / np.abs(full).max() < 1e-12
    # last-position logits are what forward() returns by default
    assert np.allclose(m.forward(ids, 0, m.new_cache()), full[-1], rtol=1e-11, atol=1e-11)
    # exact vs the reference's fake-quantisation: a measurable, bounded difference (int8-vs-e4m3 question of CONTRACTS)
    mq = model_from_gguf(mini, quant=QuantConfig.reference(), dtype=np.float64, max_seq_len=48)
    fq = mq.forward(ids, 0, mq.new_cache(), full_logits=True)
    d = np.abs(fq - full).max() / np.abs(full).max()
    assert 1e-4 < d < 1.0
    print(f"\n[mini GGUF] exact vs reference-fake-quant logits: max rel diff {d:.3e}")


def test_gguf_and_dict_providers_give_identical_models(mini):
    """Dequantise every tensor through the loader into a DictWeights: the two providers are interchangeable."""
    w, cfg = load_from_gguf(mini)
    t = {}
    for name in w.names():
        if ".ffn.experts." in name:
            layer = int(name.split(".")[1])
            which = name.split(".")[-2]
            for e in range(cfg.n_routed_experts):
                t[f"layers.{layer}.ffn.experts.{e}.{which}.weight"] = w.expert(layer, e)[("w1", "w3", "w2").index(which)]
        elif "engram.embed" in name:
            t[name] = w.engram_rows(int(name.split(".")[1]), np.arange(w.tensor_info(name).shape[0]))
        else:
            t[name] = w.get(name)
    dw = DictWeights(t)
    ec = E.constants_from_gguf_metadata(w.metadata)
    ids = np.random.default_rng(2).integers(3, cfg.vocab_size, 14)
    cfg.max_seq_len = 32
    a = Model(cfg, w, dtype=np.float64, engram_constants=ec).forward(ids, 0, Model(cfg, w, dtype=np.float64, engram_constants=ec).new_cache(), full_logits=True)
    m2 = Model(cfg, dw, dtype=np.float64, engram_constants=ec)
    b = m2.forward(ids, 0, m2.new_cache(), full_logits=True)
    assert np.array_equal(a, b)


def test_dict_weights_stacked_experts():
    rng = np.random.default_rng(0)
    stacked = {"layers.0.ffn.experts.w1.weight": rng.standard_normal((4, 3, 2)),
               "layers.0.ffn.experts.w3.weight": rng.standard_normal((4, 3, 2)),
               "layers.0.ffn.experts.w2.weight": rng.standard_normal((4, 2, 3))}
    dw = DictWeights(stacked)
    w1, w3, w2 = dw.expert(0, 2)
    assert np.array_equal(w1, stacked["layers.0.ffn.experts.w1.weight"][2]) and w2.shape == (2, 3)
