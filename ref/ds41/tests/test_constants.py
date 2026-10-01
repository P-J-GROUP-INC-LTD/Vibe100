"""Engram constants: derived exactly as engram.py does, then checked against the target GGUF's own metadata; the
129,280 -> 99,092 token map rebuilt from the tokenizer and compared with the map stored in the GGUF."""
import numpy as np
import pytest

from ref.ds41 import engram as E
from ref.ds41 import selfcheck as S
from ref.ds41.config import Config
from ref.ds41.tests import official as O


@pytest.fixture(scope="module")
def real_cfg():
    return Config.from_official_json(S.REF_DIR / "inference" / "config.json")


@pytest.fixture(scope="module")
def gguf_map():
    return S.load_token_map_fixture()


@pytest.fixture(scope="module")
def tokenizer_json():
    p = S.tokenizer_json_path()
    if p is None:
        pytest.skip("tokenizer.json unavailable (no network); set DS41_TOKENIZER_JSON")
    return p


def test_selfcheck_all_constants_pass():
    res = S.run_checks(verbose=False)
    failed = [(n, d) for n, ok, d in res if not ok and not d.startswith("SKIPPED")]
    assert not failed, failed
    names = " | ".join(n for n, _, _ in res)
    for needle in ("primes", "offsets", "multipliers", "sum of each layer's 24 primes"):
        assert needle in names


def test_primes_offsets_multipliers_equal_gguf_metadata(real_cfg):
    kv = S.load_gguf_metadata()
    d = S.derive_engram(real_cfg)
    assert d["primes"].tolist() == [kv["deepseek41.engram.primes"][:24], kv["deepseek41.engram.primes"][24:]]
    assert d["offsets"].ravel().tolist() == kv["deepseek41.engram.offsets"]
    assert d["multipliers"].ravel().tolist() == kv["deepseek41.engram.multipliers"]
    # each layer's 24 primes sum to its table's row count; the offsets are the running sums
    for j, rows in enumerate(kv["deepseek41.engram.num_embeddings"]):
        assert int(d["primes"][j].sum()) == rows == real_cfg.engram_num_embeddings[j]
        assert d["offsets"][j][0] == 0 and d["offsets"][j][-1] + d["primes"][j][-1] == rows
    assert d["multipliers"].shape == (2, 4) and (d["multipliers"] % 2 == 1).all()
    # smallest prime above 15,999,999 is 16,000,057 and nothing is reused
    assert d["primes"][0, 0] == 16000057 and len(set(d["primes"].ravel().tolist())) == 48
    # multipliers x token ids cannot overflow int64 (engram.py's bound)
    assert int(d["multipliers"].max()) * 99091 < 2 ** 63


def test_is_prime_against_sympy():
    sympy = pytest.importorskip("sympy")
    for n in list(range(0, 3000)) + list(range(15999900, 16000300)) + [2 ** 31 - 1, 2 ** 61 - 1, 3215031751]:
        assert E.is_prime(n) == bool(sympy.isprime(n)), n


def test_layout_and_multipliers_equal_the_official_functions(official_mod, real_cfg):
    mm, em = official_mod
    args = O.model_args(real_cfg)
    args.engram_num_embeddings = real_cfg.engram_num_embeddings
    lay = em.EngramLayout.from_args(args)
    mine = E.build_layout(real_cfg.engram_layer_ids, real_cfg.engram_max_ngram_size, real_cfg.engram_n_heads,
                          real_cfg.engram_vocab_size)
    assert lay.primes == mine.primes
    assert np.array_equal(em.compute_hash_multipliers(lay.layer_ids, 4, 99092).numpy(),
                          E.compute_hash_multipliers(real_cfg.engram_layer_ids, 4, 99092))


def test_gguf_token_map_size_and_structure(gguf_map):
    assert gguf_map.shape == (129280,) and gguf_map.min() == 0 and gguf_map.max() == 99091
    assert len(np.unique(gguf_map)) == 99092
    # ids are assigned in order of first appearance
    first_seen = np.r_[True, gguf_map[1:] > np.maximum.accumulate(gguf_map)[:-1]]
    assert np.array_equal(gguf_map[first_seen], np.arange(99092))
    # the first special/ASCII tokens are all distinct, so the pad token (id 2) keeps compressed id 2
    assert gguf_map[:3].tolist() == [0, 1, 2]


def test_oracle_rebuilds_the_gguf_token_map_from_the_tokenizer(tokenizer_json, gguf_map):
    lk, n = E.build_compressed_token_map(tokenizer_json)
    assert len(lk) == 129280 and n == 99092
    assert np.array_equal(np.array(lk), gguf_map)


def test_oracle_map_equals_official_builder_with_real_tokenizers(official_mod, tokenizer_json):
    tk = pytest.importorskip("tokenizers")
    mm, em = official_mod

    class Tok:
        backend_tokenizer = tk.Tokenizer.from_file(str(tokenizer_json))

        def __len__(self):
            return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)

    lo, no = em.build_compressed_token_map(Tok())
    lk, n = E.build_compressed_token_map(tokenizer_json)
    assert no == n == 99092 and lo == lk


def test_spot_values_derived_independently(tokenizer_json, gguf_map):
    """Facts about the map that follow from the normaliser alone: case/accents/whitespace variants of a word share one
    compressed id, distinct words do not."""
    toks = E.tokens_from_tokenizer_json(tokenizer_json)
    ident = {t: i for i, t in enumerate(toks)}
    b2u = E._bytes_to_unicode()
    enc = lambda s: "".join(b2u[b] for b in s.encode("utf-8"))
    m = lambda s: int(gguf_map[ident[enc(s)]])
    assert m("a") == m("A") and m("the") == m(" the") == m("The") == m(" The") == m(" THE")
    assert m("e") != m("a") and m("the") != m("this")
    for s in ("é", "É"):
        if enc(s) in ident:
            assert m(s) == m("e")
    assert gguf_map[ident["<｜end▁of▁sentence｜>"]] == 1 and gguf_map[ident["<｜▁pad▁｜>"]] == 2
    assert len({int(gguf_map[ident[enc(c)]]) for c in "0123456789"}) == 10        # digits stay distinct


def test_stdlib_normaliser_matches_tokenizers_on_synthetic_vocab(official_mod):
    """Always runs (no tokenizer.json needed): the synthetic byte-level vocabulary of the tiny model, official
    normaliser pipeline (HF `tokenizers`) vs the oracle's unicodedata re-implementation."""
    tk = pytest.importorskip("tokenizers")
    mm, em = official_mod
    toks = O.synthetic_tokens(512)
    lo, no = em.build_compressed_token_map(O.FakeTokenizer(toks))
    lk, n = E.build_compressed_token_map_from_tokens(toks)
    assert lo == lk and no == n and n < 512 and n > 50
