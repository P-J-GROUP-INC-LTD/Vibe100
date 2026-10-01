"""Configuration sources agree; the layer-mode map; the GGUF tensor-name mapping and shape contract against the REAL
saved header of the target GGUF (1006 tensors)."""
import collections

import numpy as np
import pytest

from ref.ds41 import selfcheck as S
from ref.ds41.config import Config, Mode, describe_modes, layer_modes
from ref.ds41.weights import canonical_name, expected_shape

REAL_MODES = (["swa"] * 2 + ["full"] + ["reuse"] * 5 + ["full"] + ["reuse"] * 5 + ["full"] + ["reuse"] * 5 + ["full"]
              + ["reuse"] * 3 + ["reindex", "reuse", "reuse", "reuse"] * 4)


@pytest.fixture(scope="module")
def real_cfg():
    return Config.from_official_json(S.REF_DIR / "inference" / "config.json")


def test_layer_mode_map_of_the_released_model(real_cfg):
    modes = [m.value for m in layer_modes(real_cfg)]
    assert len(modes) == 40
    # RESEARCH s2: L0-1 SWA; L2 Full, 3-7 Reuse; L8 Full, 9-13; L14 Full, 15-19; L20 Full (ratio 1), 21-23 Reuse;
    # L24/28/32/36 Reindex each followed by 3 Reuse
    assert modes == REAL_MODES, describe_modes(real_cfg)
    c = collections.Counter(modes)
    assert c == {"swa": 2, "full": 4, "reindex": 4, "reuse": 30}
    assert [l for l, m in enumerate(layer_modes(real_cfg)) if m is Mode.FULL] == [2, 8, 14, 20]
    assert [l for l, m in enumerate(layer_modes(real_cfg)) if m is Mode.REINDEX] == [24, 28, 32, 36]
    assert real_cfg.compress_ratios[:40] == (0, 0) + (2,) * 18 + (1,) * 20          # DSpark entries 40-42 are 0


def test_configs_from_official_json_and_gguf_metadata_agree(real_cfg):
    kv = S.load_gguf_metadata()
    c = Config.from_gguf_metadata(kv, vocab_size=129280)
    for f in ("vocab_size", "dim", "moe_inter_dim", "n_layers", "n_heads", "n_routed_experts", "n_activated_experts",
              "n_shared_experts", "route_scale", "swiglu_limit", "q_lora_rank", "head_dim", "rope_head_dim", "o_groups",
              "o_lora_rank", "window_size", "kv_source_layers", "index_source_layers", "compress_rope_theta",
              "original_seq_len", "rope_theta", "rope_factor", "beta_fast", "beta_slow", "index_n_heads", "index_head_dim",
              "index_topk", "candidate_source_layer", "candidate_topk_blocks", "candidate_block_size", "hc_mult",
              "hc_sinkhorn_iters", "engram_layer_ids", "engram_num_embeddings", "engram_max_ngram_size",
              "engram_n_heads", "engram_head_dim", "engram_pad_id", "engram_compressed_vocab_size", "norm_topk_prob",
              "score_func"):
        assert getattr(c, f) == getattr(real_cfg, f), f
    assert c.compress_ratios[:40] == real_cfg.compress_ratios[:40]
    assert layer_modes(c) == layer_modes(real_cfg)
    assert np.isclose(c.norm_eps, 1e-20, rtol=1e-6) and np.isclose(c.hc_eps, 1e-6, rtol=1e-6)   # float32-rounded in the GGUF
    assert real_cfg.dim == 5120 and real_cfg.n_heads * real_cfg.head_dim == 32768 and real_cfg.softmax_scale == 512 ** -0.5


def test_all_1006_gguf_tensors_map_and_have_the_shapes_the_config_implies(real_cfg):
    table = S.load_gguf_tensor_table()
    assert len(table) == 1006
    seen = {}
    for gname, (typ, dims) in table.items():
        canon, which = canonical_name(gname)
        assert canon not in seen, (gname, canon)
        seen[canon] = gname
        exp = expected_shape(real_cfg, canon)
        assert exp is not None, canon
        assert tuple(reversed(dims)) == tuple(exp), (gname, dims, exp)
        assert typ in {"F32", "BF16", "Q8_0", "MXFP4"}, (gname, typ)
    # expert tensors are the only 3-D ones and stay MXFP4; Engram tables are MXFP4 rows of 8 blocks
    assert {t for n, (t, d) in table.items() if "_exps" in n} == {"MXFP4"}
    assert {t for n, (t, d) in table.items() if "engram_embed" in n} == {"MXFP4"}
    assert table["blk.1.engram_embed.weight"][1] == [256, 384006168]
    # one expert = 3 x 6,266,880 B
    gate = table["blk.0.ffn_gate_exps.weight"][1]
    assert 3 * gate[1] * (gate[0] // 32 * 17) == 18800640
    # every layer has the parts its mode needs (and only those)
    for l, mode in enumerate(layer_modes(real_cfg)):
        has = lambda suffix: f"blk.{l}.{suffix}" in table
        assert has("attn_q_a.weight") and has("ffn_gate_exps.weight") and has("hc_ffn_fn.weight")
        assert has("attn_compressor_kv.weight") == (mode is Mode.FULL)
        assert has("attn_compressor_gate.weight") == (mode is Mode.FULL and real_cfg.compress_ratios[l] == 2)
        assert has("indexer.attn_q_b.weight") == (mode in (Mode.FULL, Mode.REINDEX))
        assert has("indexer_compressor_kv.weight") == (mode is Mode.FULL)
        assert has("engram_embed.weight") == (l in real_cfg.engram_layer_ids)


def test_expected_shape_rejects_unknown_and_vcruz_aliases():
    assert expected_shape(Config(), "layers.3.bogus.weight") is None
    # the other GGUF family's spellings map onto the same canonical names
    assert canonical_name("blk.14.engram_embd.weight")[0] == "layers.14.engram.embed.weight"
    assert canonical_name("blk.20.indexer.attn_k.weight")[0] == "layers.20.attn.indexer.wk.weight"
    assert canonical_name("blk.20.indexer.k_norm.weight")[0] == "layers.20.attn.indexer.k_norm.weight"
    with pytest.raises(KeyError):
        canonical_name("blk.0.exp_probs_b_vl.bias")             # the VL bias: not used by the text-only port
