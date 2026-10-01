"""Tests for tools/ds41/ds1_tokenizer.py: byte-level BPE from GGUF-style metadata, pre-tokenizer, special tokens, decode, chat template.

Three layers of evidence, each skipped when its inputs are missing (the first needs nothing):
  1. a small synthetic vocabulary: hand-derived expectations, and token-for-token equality with the Hugging Face `tokenizers` library when it is installed;
  2. the real GGUF header dump vendored in third_party/ (ids, counts, the chat template);
  3. the official tokenizer.json (env DS41_TOKENIZER_JSON, or the cache ref/ds41/selfcheck.py fills): metadata equality and equality with `tokenizers` on text.
"""
import gzip
import itertools
import json
import os
import pathlib
import random
import tempfile
import unittest

import ds1_tokenizer as K

REPO = pathlib.Path(__file__).resolve().parents[2]
HEADERS = REPO / "third_party" / "deepseek-v41-flash-reference" / "gguf-headers-mxxm-t-MXFP4.json.gz"

try:
    import tokenizers as HF                   # noqa: N812
except ImportError:                           # pragma: no cover
    HF = None
try:
    import jinja2                             # noqa: F401
    HAVE_JINJA = True
except ImportError:                           # pragma: no cover
    HAVE_JINJA = False


def official_json():
    env = os.environ.get("DS41_TOKENIZER_JSON")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    try:
        from ref.ds41.selfcheck import tokenizer_json_path
        return tokenizer_json_path(download=False)
    except Exception:                         # noqa: BLE001
        return None


# ---------------------------------------------------------------------------------------------------------------
# 1. a synthetic vocabulary
# ---------------------------------------------------------------------------------------------------------------

MERGES = ["Ġ t", "Ġ a", "i n", "h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o", "Ġwo r", "Ġwor l", "Ġworl d", "t h", "Ġ th", "Ġth e", "1 2", "12 3", "Ġ Ġ", "ĠĠ ĠĠ", "a a", "Ċ Ċ"]
SPECIALS = [("<s>", K.TT_CONTROL), ("</s>", K.TT_CONTROL), ("<|user|>", K.TT_USER_DEFINED), ("<|user|><|user|>", K.TT_USER_DEFINED), ("<think>", K.TT_USER_DEFINED)]


def synthetic_metadata(merges=MERGES, specials=SPECIALS, pre="joyai-llm") -> dict:
    tokens = [K.BYTE_TO_CHAR[b] for b in range(256)]
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b) if (a + b) not in tokens else None
    types = [K.TT_NORMAL] * len(tokens)
    for s, ty in specials:
        tokens.append(s)
        types.append(ty)
    return {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre, "tokenizer.ggml.tokens": tokens, "tokenizer.ggml.token_type": types,
            "tokenizer.ggml.merges": list(merges), "tokenizer.ggml.bos_token_id": tokens.index("<s>"), "tokenizer.ggml.eos_token_id": tokens.index("</s>"),
            "tokenizer.ggml.padding_token_id": tokens.index("</s>"), "tokenizer.ggml.add_bos_token": False, "tokenizer.ggml.add_eos_token": False}


def synthetic_tokenizer_json(md: dict) -> str:
    """The same spec as a Hugging Face tokenizer.json (the official file's structure)."""
    tokens, types = md["tokenizer.ggml.tokens"], md["tokenizer.ggml.token_type"]
    pre = [{"type": "Split", "pattern": {"Regex": r}, "behavior": "Isolated", "invert": False} for r in K.PRE_TOKENIZERS[md["tokenizer.ggml.pre"]]]
    pre.append({"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False})
    return json.dumps({"version": "1.0", "truncation": None, "padding": None,
                       "added_tokens": [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": ty == K.TT_CONTROL}
                                        for i, (t, ty) in enumerate(zip(tokens, types)) if ty in (K.TT_CONTROL, K.TT_USER_DEFINED)],
                       "normalizer": {"type": "Sequence", "normalizers": []}, "pre_tokenizer": {"type": "Sequence", "pretokenizers": pre},
                       "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
                       "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
                       "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None, "end_of_word_suffix": None, "fuse_unk": False,
                                 "byte_fallback": False, "vocab": {t: i for i, t in enumerate(tokens) if types[i] == K.TT_NORMAL}, "merges": md["tokenizer.ggml.merges"]}},
                      ensure_ascii=False)


def _pieces(tok, s):
    ps = [s]
    for pat in tok.patterns:
        nxt = []
        for p in ps:
            nxt.extend(K.split_isolated(pat, p))
        ps = nxt
    return ps


class PreTokenizer(unittest.TestCase):
    """Expected pieces are the official tokenizer.json's pre-tokenizer output (checked against `tokenizers` when it was written)."""
    @classmethod
    def setUpClass(cls):
        cls.tok = K.Ds41Tokenizer.from_metadata(synthetic_metadata())

    CASES = [
        ("Hello, world!", ["Hello", ",", " world", "!"]),
        ("1234567", ["123", "456", "7"]),
        ("a  b", ["a", " ", " b"]),
        ("  x", [" ", " x"]),
        ("x  ", ["x", "  "]),
        ("你好，世界！ok", ["你好", "，", "世界", "！", "ok"]),
        ("ひらがなカタカナabc", ["ひらがなカタカナ", "abc"]),
        ("foo.bar(baz)", ["foo", ".bar", "(baz", ")"]),
        ("end.\n\nNext", ["end", ".\n\n", "Next"]),
        ("x = 3.14159;", ["x", " =", " ", "3", ".", "141", "59", ";"]),
        ("snake_case", ["snake", "_case"]),
        ("it's", ["it", "'s"]),
        ("a\n \nb", ["a", "\n \n", "b"]),
        ("$$a", ["$$", "a"]),
        ("\t\tindent", ["\t", "\tindent"]),
        ("http://x.org/a?b=1", ["http", "://", "x", ".org", "/a", "?b", "=", "1"]),
        ("été", ["été"]),
        ("", []),
    ]

    def test_pieces(self):
        for s, want in self.CASES:
            with self.subTest(text=s):
                self.assertEqual(_pieces(self.tok, s), want)

    def test_white_space_is_the_unicode_white_space_property(self):
        """Python's `\\s` also matches U+001C..U+001F, which Rust / Oniguruma do not; U+0085, U+00A0, U+2028 and the Zs / Zl / Zp characters are white space."""
        cases = [("a\x1fb", ["a", "\x1fb"]), (" \x1f ", [" ", "\x1f", " "]), ("\x1f\x1f", ["\x1f\x1f"]), ("a \x1f", ["a", " ", "\x1f"]),
                 ("a\x1c\x1d b", ["a", "\x1c\x1d", " b"]), ("x\xa0\xa0y", ["x", "\xa0", "\xa0y"]), ("x\u3000 \u2003y", ["x", "\u3000 ", "\u2003y"]),
                 ("\x85\x85", ["\x85\x85"]), ("a\u2028b", ["a", "\u2028b"])]
        for s, want in cases:
            with self.subTest(text=s):
                self.assertEqual(_pieces(self.tok, s), want)
        for ch in "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u200a\u2028\u2029\u202f\u205f\u3000":
            self.assertEqual(len(_pieces(self.tok, ch * 3)), 1, hex(ord(ch)) + ": a run of white space is one piece")

    def test_digits_in_groups_of_three_from_the_left(self):
        for n in range(1, 12):
            self.assertEqual("".join(_pieces(self.tok, "1" * n)), "1" * n)
            self.assertEqual([len(p) for p in _pieces(self.tok, "1" * n)], [3] * (n // 3) + ([n % 3] if n % 3 else []))

    def test_unicode_classes_follow_unicodedata_plus_the_patch(self):
        import unicodedata
        for cp in (0x61, 0xe9, 0x3b1, 0x5d0, 0x627, 0x928, 0xac00):                      # letters
            self.assertEqual(_pieces(self.tok, " " + chr(cp) * 3), [" " + chr(cp) * 3])
        for cp in (0x300, 0x93f):                                                          # marks join a word
            self.assertEqual(_pieces(self.tok, "a" + chr(cp)), ["a" + chr(cp)])
        for cp in (0xa9, 0x2022, 0x20ac, 0x1f600, 0x3001):                                 # symbols and punctuation do not
            self.assertEqual(_pieces(self.tok, "a" + chr(cp)), ["a", chr(cp)], hex(cp))
            self.assertEqual(_pieces(self.tok, chr(cp) * 3), [chr(cp) * 3], hex(cp))
        for lo, hi, g in K.UNICODE_PATCH[:5] + K.UNICODE_PATCH[-5:]:                       # code points the patch adds are what its group says
            self.assertTrue(unicodedata.category(chr(lo)) == "Cn", hex(lo))
            if g == "A":
                self.assertEqual(_pieces(self.tok, "a" + chr(lo)), ["a" + chr(lo)], hex(lo))
            elif g == "N":
                self.assertEqual(_pieces(self.tok, chr(lo) * 4), [chr(lo) * 3, chr(lo)], hex(lo))
            else:
                self.assertEqual(_pieces(self.tok, "a" + chr(lo)), ["a", chr(lo)], hex(lo))

    def test_translate_pattern(self):
        self.assertEqual(K.translate_pattern(r"\p{N}{1,3}", lambda n: "0-9"), "[0-9]{1,3}")
        self.assertEqual(K.translate_pattern(r"[^\r\n\p{L}]", lambda n: "a-z"), r"[^\r\na-z]")
        self.assertIn(K.WS_CLASS, K.translate_pattern(r"\s+(?!\S)"))
        self.assertEqual(K.translate_pattern(r"\p{L}", None), r"\p{L}")
        self.assertEqual(K.translate_pattern(r"[\[\\\]]\p{N}", lambda n: "0-9"), r"[\[\\\]][0-9]")


class Bpe(unittest.TestCase):
    def tok(self, merges, specials=()):
        return K.Ds41Tokenizer.from_metadata(synthetic_metadata(merges, list(specials) + [("<s>", K.TT_CONTROL), ("</s>", K.TT_CONTROL)]))

    def toks(self, tok, s):
        return [tok.tokens[i] for i in tok.encode(s)]

    def test_lowest_rank_first_leftmost_on_ties(self):
        t = self.tok(["a a"])
        self.assertEqual(self.toks(t, "aaa"), ["aa", "a"])
        self.assertEqual(self.toks(t, "aaaa"), ["aa", "aa"])
        self.assertEqual(self.toks(t, "aaaaa"), ["aa", "aa", "a"])
        t = self.tok(["b c", "a b"])
        self.assertEqual(self.toks(t, "abc"), ["a", "bc"])                  # b c has the lower rank
        t = self.tok(["a b", "b c"])
        self.assertEqual(self.toks(t, "abc"), ["ab", "c"])
        t = self.tok(["h e", "he l", "hel l", "hell o"])
        self.assertEqual(self.toks(t, "hello"), ["hello"])
        self.assertEqual(self.toks(t, "help"), ["hel", "p"])
        self.assertEqual(self.toks(t, "hex"), ["he", "x"])

    def test_long_runs_are_fast_and_lossless(self):
        t = K.Ds41Tokenizer.from_metadata(synthetic_metadata())
        s = "a" * 20000
        ids = t.encode(s)
        self.assertEqual(t.decode(ids), s)
        self.assertEqual(len(ids), 10000)

    def test_bytes_that_are_not_valid_text_round_trip_per_token(self):
        t = K.Ds41Tokenizer.from_metadata(synthetic_metadata())
        ids = t.encode("é")
        self.assertEqual(len(ids), 2)                                       # two byte tokens, no merge for them
        self.assertEqual(t.decode(ids), "é")
        self.assertEqual(t.decode(ids[:1]), "�")                       # half a character
        self.assertEqual(t.decode(ids + ids), "éé")

    def test_merges_and_vocabulary_are_validated(self):
        md = synthetic_metadata()
        md["tokenizer.ggml.merges"] = md["tokenizer.ggml.merges"] + ["x y z"]
        with self.assertRaises(K.TokenizerError):
            K.Ds41Tokenizer.from_metadata(md)
        md = synthetic_metadata()
        md["tokenizer.ggml.merges"] = md["tokenizer.ggml.merges"] + ["x y"]       # xy is not a token: such a merge can never fire
        with self.assertRaises(K.TokenizerError):
            K.Ds41Tokenizer.from_metadata(md, strict=True)
        lenient = K.Ds41Tokenizer.from_metadata(md)
        self.assertEqual(lenient.bad_merges, [("x", "y")])
        self.assertEqual(lenient.encode("xy"), K.Ds41Tokenizer.from_metadata(synthetic_metadata()).encode("xy"))
        with self.assertRaises(K.TokenizerError):
            K.Ds41Tokenizer.from_metadata(synthetic_metadata(pre="something-new"))
        md = synthetic_metadata()
        md["tokenizer.ggml.model"] = "llama"
        with self.assertRaises(K.TokenizerError):
            K.Ds41Tokenizer.from_metadata(md)
        md = synthetic_metadata()
        md["tokenizer.ggml.tokens"] = "<array len 129280>"                          # a headers-only JSON has summaries, not arrays
        with self.assertRaises(K.TokenizerError):
            K.Ds41Tokenizer.from_metadata(md)
        md = synthetic_metadata()
        md["tokenizer.ggml.token_type"] = md["tokenizer.ggml.token_type"][:-1]
        with self.assertRaises(K.TokenizerError):
            K.Ds41Tokenizer.from_metadata(md)


class Specials(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = K.Ds41Tokenizer.from_metadata(synthetic_metadata())
        cls.v = cls.tok.vocab

    def test_literal_special_tokens(self):
        t, v = self.tok, self.v
        self.assertEqual(t.encode("<s>"), [v["<s>"]])
        self.assertEqual(t.encode("a<s>b</s>"), t.encode("a") + [v["<s>"]] + t.encode("b") + [v["</s>"]])
        self.assertEqual(t.encode("<think>"), [v["<think>"]])                       # user-defined: matched too
        self.assertNotEqual(t.encode("<s>", parse_special=False), [v["<s>"]])      # as plain text
        self.assertEqual(t.decode(t.encode("<s>", parse_special=False)), "<s>")
        self.assertNotEqual(t.encode("<s"), [v["<s>"]])

    def test_leftmost_longest(self):
        t, v = self.tok, self.v
        self.assertEqual(t.encode("<|user|><|user|>"), [v["<|user|><|user|>"]])
        self.assertEqual(t.encode("<|user|><|user|><|user|>"), [v["<|user|><|user|>"], v["<|user|>"]])
        self.assertEqual(t.encode("<|user|>x<|user|>"), [v["<|user|>"]] + t.encode("x") + [v["<|user|>"]])

    def test_add_bos_eos_flags(self):
        t, v = self.tok, self.v
        self.assertEqual(t.encode("a", add_bos=True, add_eos=True), [v["<s>"]] + t.encode("a") + [v["</s>"]])
        md = synthetic_metadata()
        md["tokenizer.ggml.add_bos_token"] = True
        t2 = K.Ds41Tokenizer.from_metadata(md)
        self.assertEqual(t2.encode("a")[0], v["<s>"])
        self.assertEqual(t2.encode("a", add_bos=False), t.encode("a"))
        self.assertEqual(t.bos_token, "<s>")
        self.assertEqual(t.eos_token, "</s>")

    def test_decode(self):
        t, v = self.tok, self.v
        ids = t.encode("hello <s>world</s>")
        self.assertEqual(t.decode(ids), "hello <s>world</s>")
        self.assertEqual(t.decode(ids, skip_special=True), "hello world")
        ids = t.encode("<think>x")
        self.assertEqual(t.decode(ids, skip_special=True), "<think>x")              # user-defined tokens are text, not control
        with self.assertRaises(K.TokenizerError):
            t.decode([10 ** 6])
        self.assertEqual(t.decode([]), "")


class Fuzz(unittest.TestCase):
    """Round trips on the synthetic vocabulary, and equality with the Hugging Face library built from the same spec."""
    @classmethod
    def setUpClass(cls):
        cls.md = synthetic_metadata()
        cls.tok = K.Ds41Tokenizer.from_metadata(cls.md)
        rng = random.Random(5)
        alpha = "abcdefghijklmnopqrstuvwxyz ABC0123456789\t\n\r.,;:!?'\"()[]{}<>/\\@#$%^&*-_=+`~éüß你好世界日本語ひらがなカタカナ한국😀👍́​　 \x00\x1f"
        cls.texts = ["".join(rng.choice(alpha) for _ in range(rng.randint(0, 60))) for _ in range(600)] + K.SELFCHECK_TEXTS
        cls.texts += ["<s>x</s> <|user|> hello world the 123 thhe  \n\n\n", "hello world the 12 123 1234"]

    def test_round_trip(self):
        for s in self.texts:
            self.assertEqual(self.tok.decode(self.tok.encode(s, parse_special=False)), s.encode("utf-8", "replace").decode("utf-8", "replace"))

    @unittest.skipIf(HF is None, "the `tokenizers` package is not installed")
    def test_equal_to_the_hugging_face_library(self):
        ref = HF.Tokenizer.from_str(synthetic_tokenizer_json(self.md))
        for s in self.texts:
            self.assertEqual(self.tok.encode(s), ref.encode(s, add_special_tokens=False).ids, repr(s))
            ids = self.tok.encode(s)
            self.assertEqual(self.tok.decode(ids), ref.decode(ids, skip_special_tokens=False), repr(s))

    def test_regex_module_agrees_with_the_generated_classes_on_text_unicode_14_knows(self):
        try:
            import regex  # noqa: F401
        except ImportError:
            self.skipTest("the `regex` module is not installed")
        t2 = K.Ds41Tokenizer.from_metadata(self.md, engine="regex")
        for s in self.texts[:300]:
            if all(ord(c) < 0x10000 for c in s):
                self.assertEqual(t2.encode(s), self.tok.encode(s), repr(s))


# ---------------------------------------------------------------------------------------------------------------
# 2. the GGUF header dump
# ---------------------------------------------------------------------------------------------------------------


def header_kv() -> dict:
    hdr = json.load(gzip.open(HEADERS, "rt", encoding="utf-8"))
    return hdr[sorted(k for k in hdr if "00001-of" in k)[0]]["kv"]


@unittest.skipUnless(HEADERS.is_file(), "the vendored GGUF header dump is missing")
class RealHeaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kv = header_kv()

    def test_what_the_gguf_says(self):
        kv = self.kv
        self.assertEqual(kv["tokenizer.ggml.model"], "gpt2")
        self.assertIn(kv["tokenizer.ggml.pre"], K.PRE_TOKENIZERS)
        self.assertEqual((kv["tokenizer.ggml.bos_token_id"], kv["tokenizer.ggml.eos_token_id"], kv["tokenizer.ggml.padding_token_id"]), (0, 1, 1))
        self.assertEqual((kv["tokenizer.ggml.add_bos_token"], kv["tokenizer.ggml.add_eos_token"]), (False, False))
        self.assertEqual(kv["tokenizer.ggml.tokens"], "<array len 129280>")
        self.assertEqual(kv["tokenizer.ggml.merges"], "<array len 127741>")
        self.assertEqual(kv["deepseek41.engram.pad_token_id"], 2)                              # the HF pad token <｜▁pad▁｜>, not the GGUF's padding id

    def test_the_chat_template_is_the_one_the_native_renderer_transcribes(self):
        t = self.kv["tokenizer.chat_template"]
        self.assertEqual(K.template_hash(t), K.template_hash(K.vendored_chat_template()))
        self.assertIn(K.template_hash(t), K.KNOWN_TEMPLATE_HASHES)

    def test_ids_the_engine_relies_on_exist_in_the_official_vocabulary(self):
        for tok, i in K.KNOWN_IDS.items():
            self.assertLess(i, 129280, tok)


# ---------------------------------------------------------------------------------------------------------------
# chat template
# ---------------------------------------------------------------------------------------------------------------

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather of a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                                          "required": ["city"]}}},
         {"type": "function", "function": {"name": "ünïcode", "description": "日本語", "parameters": {}}}]
CONVERSATIONS = {
    "single": [{"role": "user", "content": "Hello"}],
    "system": [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi"}],
    "two system": [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "Hi"}],
    "multi turn": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1", "reasoning_content": "think1"}, {"role": "user", "content": "q2"},
                   {"role": "assistant", "content": "a2", "reasoning_content": "think2"}, {"role": "user", "content": "q3"}],
    "consecutive users": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}, {"role": "developer", "content": "c"}],
    "tool calls": [{"role": "user", "content": "weather?"},
                   {"role": "assistant", "content": "", "reasoning_content": "need a tool",
                    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris", "n": 3, "flag": True, "obj": {"a": [1, 2], "é": "ü"}}}},
                                   {"function": {"name": "noargs", "arguments": {}}}]},
                   {"role": "tool", "content": "sunny"}, {"role": "tool", "content": "more"}, {"role": "assistant", "content": "It is sunny."}],
    "json string arguments": [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{\"a\": \"s\", \"b\": 2.5}"}}]}],
    "assistant last": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "partial"}],
    "empty": [],
    "none content": [{"role": "user", "content": None}, {"role": "assistant", "content": None, "reasoning_content": None}],
}


class ChatTemplate(unittest.TestCase):
    BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"

    def render(self, msgs, **kw):
        return K.render_deepseek_v4(msgs, bos_token=self.BOS, **kw)

    def test_golden_renderings(self):
        self.assertEqual(self.render(CONVERSATIONS["single"], add_generation_prompt=True), self.BOS + "<｜User｜>Hello<｜Assistant｜></think>")
        self.assertEqual(self.render(CONVERSATIONS["single"], add_generation_prompt=True, thinking=True), self.BOS + "<｜User｜>Hello<｜Assistant｜><think>")
        self.assertEqual(self.render(CONVERSATIONS["single"]), self.BOS + "<｜User｜>Hello")
        self.assertEqual(self.render(CONVERSATIONS["system"], add_generation_prompt=True), self.BOS + "You are helpful.<｜User｜>Hi<｜Assistant｜></think>")
        self.assertEqual(self.render(CONVERSATIONS["two system"]), self.BOS + "A\n\nB<｜User｜>Hi")
        self.assertEqual(self.render(CONVERSATIONS["consecutive users"]), self.BOS + "<｜User｜>a\n\nb\n\nc")
        self.assertEqual(self.render(CONVERSATIONS["multi turn"], add_generation_prompt=True),
                         self.BOS + "<｜User｜>q1<｜Assistant｜></think>a1" + self.EOS + "<｜User｜>q2<｜Assistant｜></think>a2" + self.EOS + "<｜User｜>q3<｜Assistant｜></think>")
        # thinking mode keeps the reasoning of assistant turns after the last user turn only
        msgs = CONVERSATIONS["multi turn"][:3] + [{"role": "assistant", "content": "a2", "reasoning_content": "think2"}]
        self.assertEqual(self.render(msgs, thinking=True), self.BOS + "<｜User｜>q1<｜Assistant｜></think>a1" + self.EOS + "<｜User｜>q2<｜Assistant｜><think>think2</think>a2" + self.EOS)
        self.assertEqual(self.render(msgs, thinking=True, drop_thinking=False),
                         self.BOS + "<｜User｜>q1<｜Assistant｜><think>think1</think>a1" + self.EOS + "<｜User｜>q2<｜Assistant｜><think>think2</think>a2" + self.EOS)
        out = self.render(CONVERSATIONS["tool calls"])
        self.assertIn("<｜Assistant｜></think>\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n", out)
        self.assertIn("<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n", out)
        self.assertIn("<｜DSML｜parameter name=\"obj\" string=\"false\">{\"a\": [1, 2], \"é\": \"ü\"}</｜DSML｜parameter>\n", out)
        self.assertIn("<｜User｜><tool_result>sunny</tool_result>\n\n<tool_result>more</tool_result><｜Assistant｜>", out)
        self.assertIn("<｜DSML｜invoke name=\"noargs\">\n\n</｜DSML｜invoke>\n", out)

    def test_tools_and_reasoning_effort(self):
        out = self.render(CONVERSATIONS["system"], tools=TOOLS)
        self.assertIn("## Tools\n\nYou have access to a set of tools", out)
        self.assertIn(json.dumps(TOOLS[0]["function"], ensure_ascii=False) + "\n" + json.dumps(TOOLS[1]["function"], ensure_ascii=False) + "\n", out)
        self.assertTrue(out.startswith(self.BOS + "You are helpful.\n\n## Tools"))
        self.assertTrue(out.endswith("<｜User｜>Hi"))
        self.assertIn("Reasoning Effort: Absolute maximum", self.render(CONVERSATIONS["single"], thinking=True, reasoning_effort="max"))
        self.assertNotIn("Reasoning Effort", self.render(CONVERSATIONS["single"], thinking=False, reasoning_effort="max"))
        self.assertNotIn("Reasoning Effort", self.render(CONVERSATIONS["single"], thinking=True, reasoning_effort="high"))
        self.assertIn("## Response Format:", self.render(CONVERSATIONS["single"], response_format={"type": "json_object"}))

    @unittest.skipUnless(HAVE_JINJA and HEADERS.is_file(), "needs jinja2 and the vendored template")
    def test_native_renderer_equals_the_jinja_template_of_the_gguf(self):
        tpl = K.vendored_chat_template()
        n = 0
        for (name, msgs), agp, th, tl, dt, eff, rf in itertools.product(CONVERSATIONS.items(), (False, True), (False, True), (None, TOOLS), (True, False),
                                                                      (None, "max", "high"), (None, {"type": "json_object"})):
            kw = dict(add_generation_prompt=agp, thinking=th, tools=tl, drop_thinking=dt, reasoning_effort=eff, response_format=rf)
            self.assertEqual(self.render(msgs, **kw), K.render_jinja(tpl, msgs, bos_token=self.BOS, **kw), (name, kw))
            n += 1
        self.assertGreater(n, 900)

    @unittest.skipUnless(HEADERS.is_file(), "needs the vendored template")
    def test_tokenizer_object_uses_the_template_and_encodes_the_markers_as_single_tokens(self):
        md = synthetic_metadata(specials=SPECIALS + [("<｜begin▁of▁sentence｜>", K.TT_CONTROL), ("<｜User｜>", K.TT_USER_DEFINED), ("<｜Assistant｜>", K.TT_USER_DEFINED),
                                                     ("</think>", K.TT_USER_DEFINED), (K.EOS_TEXT, K.TT_CONTROL)])
        md["tokenizer.ggml.bos_token_id"] = md["tokenizer.ggml.tokens"].index("<｜begin▁of▁sentence｜>")
        md["tokenizer.chat_template"] = K.vendored_chat_template()
        tok = K.Ds41Tokenizer.from_metadata(md)
        text = tok.apply_chat_template(CONVERSATIONS["system"], add_generation_prompt=True)
        self.assertEqual(text, self.BOS + "You are helpful.<｜User｜>Hi<｜Assistant｜></think>")
        ids = tok.encode_chat(CONVERSATIONS["system"], add_generation_prompt=True)
        v = tok.vocab
        self.assertEqual(ids[0], v[self.BOS])
        self.assertEqual(ids[-2:], [v["<｜Assistant｜>"], v["</think>"]])
        self.assertEqual(ids.count(v["<｜User｜>"]), 1)
        self.assertEqual(tok.decode(ids), text)
        self.assertEqual(tok.apply_chat_template(CONVERSATIONS["system"], engine="auto"), tok.apply_chat_template(CONVERSATIONS["system"], engine="native"))

    def test_an_unknown_template_needs_jinja(self):
        md = synthetic_metadata()
        md["tokenizer.chat_template"] = "{{ bos_token }}{% for m in messages %}[{{ m['role'] }}:{{ m['content'] }}]{% endfor %}"
        tok = K.Ds41Tokenizer.from_metadata(md)
        with self.assertRaises(K.TokenizerError):
            tok.apply_chat_template(CONVERSATIONS["single"], engine="native")
        if HAVE_JINJA:
            self.assertEqual(tok.apply_chat_template(CONVERSATIONS["single"]), "<s>[user:Hello]")


# ---------------------------------------------------------------------------------------------------------------
# GGUF file
# ---------------------------------------------------------------------------------------------------------------


class FromGguf(unittest.TestCase):
    def test_mini_gguf(self):
        """The mini GGUF has the real keys with a placeholder vocabulary (<t0> ... <t511>, no byte tokens): it loads, decodes literally, and refuses to encode."""
        import fixtures as F
        tok = K.Ds41Tokenizer.from_gguf(F.mini()["paths"][0])
        self.assertEqual((tok.vocab_size, tok.pre, tok.model, tok.bos_id, tok.eos_id, tok.add_bos), (512, "joyai-llm", "gpt2", 0, 1, False))
        self.assertEqual(tok.decode([5, 6]), "<t5><t6>")
        with self.assertRaises(K.TokenizerError):
            tok.encode("x")
        self.assertEqual(tok.chat_template, "{{ messages }}")

    def test_command_line(self):
        import contextlib
        import io
        import fixtures as F
        path = str(F.mini()["paths"][0])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(K.main(["info", "--gguf", path]), 0)
        self.assertEqual(json.loads(out.getvalue())["vocab_size"], 512)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(K.main(["decode", "--gguf", path, "--ids", "3,4"]), 0)
        self.assertEqual(out.getvalue().strip(), "<t3><t4>")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(K.main(["encode", "--gguf", path, "x"]), 2)                  # the vocabulary cannot encode: could-not-run, not a crash
            self.assertEqual(K.main(["info", "--gguf", "/nonexistent.gguf"]), 2)


# ---------------------------------------------------------------------------------------------------------------
# 3. the official tokenizer.json
# ---------------------------------------------------------------------------------------------------------------


@unittest.skipUnless(official_json() is not None, "no tokenizer.json (set DS41_TOKENIZER_JSON)")
class Official(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = official_json()
        cls.md = K.metadata_from_tokenizer_json(cls.path, K.vendored_chat_template())
        cls.tok = K.Ds41Tokenizer.from_metadata(cls.md)

    def test_sizes_agree_with_the_gguf_header(self):
        if not HEADERS.is_file():
            self.skipTest("no header dump")
        kv = header_kv()
        self.assertEqual(self.tok.vocab_size, 129280)
        self.assertEqual(self.tok.n_merges, 127741)
        self.assertEqual((self.tok.bos_id, self.tok.eos_id), (kv["tokenizer.ggml.bos_token_id"], kv["tokenizer.ggml.eos_token_id"]))
        self.assertEqual(self.tok.pre, kv["tokenizer.ggml.pre"])

    def test_known_ids(self):
        for tok, i in K.KNOWN_IDS.items():
            self.assertEqual(self.tok.vocab[tok], i, tok)
        self.assertEqual(self.tok.encode("<｜User｜>hi<｜Assistant｜>")[0], 128803)
        self.assertEqual(self.tok.encode("<｜User｜>hi<｜Assistant｜>")[-1], 128804)

    def test_metadata_check_reports_a_damaged_gguf(self):
        self.assertEqual(K.check_against_tokenizer_json(self.tok, self.path), [])
        types = list(self.md["tokenizer.ggml.token_type"])
        types[128803] = K.TT_CONTROL                                         # <｜User｜> is user-defined in the official file
        damaged = K.Ds41Tokenizer.from_metadata({**self.md, "tokenizer.ggml.token_type": types})
        self.assertTrue(any("token types differ" in p for p in K.check_against_tokenizer_json(damaged, self.path)))
        tokens = list(self.md["tokenizer.ggml.tokens"])
        tokens[5000], tokens[5001] = tokens[5001], tokens[5000]
        merges = list(self.md["tokenizer.ggml.merges"])
        swapped = K.Ds41Tokenizer.from_metadata({**self.md, "tokenizer.ggml.tokens": tokens})
        self.assertTrue(any("token strings differ" in p for p in K.check_against_tokenizer_json(swapped, self.path)))
        merges[10], merges[11] = merges[11], merges[10]
        self.assertTrue(any("merges differ" in p for p in K.check_against_tokenizer_json(K.Ds41Tokenizer.from_metadata({**self.md, "tokenizer.ggml.merges": merges}), self.path)))

    def test_selfcheck_passes(self):
        failed, _skipped = K.selfcheck(self.tok, self.path, verbose=False)
        self.assertEqual(failed, 0)

    @unittest.skipIf(HF is None, "the `tokenizers` package is not installed")
    def test_token_for_token_equal_to_the_library_on_text(self):
        ref = HF.Tokenizer.from_file(str(self.path))
        rng = random.Random(3)
        files = sorted(REPO.glob("docs/**/*.md"))[:6] + sorted(REPO.glob("tools/ds41/*.py"))[:6] + sorted(REPO.glob("src/ds41/**/*.cpp"))[:6]
        texts = [f.read_text(encoding="utf-8", errors="replace") for f in files if f.is_file()]
        texts += K.SELFCHECK_TEXTS
        def cp():
            r = rng.random()
            if r < 0.4:
                return rng.randint(0x20, 0x7e)
            if r < 0.6:
                return rng.randint(0x80, 0x2fff)
            if r < 0.75:
                return rng.randint(0x3000, 0x9fff)
            if r < 0.85:
                return rng.randint(0x1f300, 0x1faff)
            c = rng.randint(0, 0x10ffff)                                       # anything, including code points Unicode 14 does not know
            return c if not 0xD800 <= c <= 0xDFFF else 0x41
        for _ in range(1500):
            texts.append("".join(chr(cp()) for _ in range(rng.randint(0, 40))))
        for s in texts:
            self.assertEqual(self.tok.encode(s, add_bos=False), ref.encode(s, add_special_tokens=False).ids, repr(s[:80]))
            ids = self.tok.encode(s, add_bos=False)
            self.assertEqual(self.tok.decode(ids), ref.decode(ids, skip_special_tokens=False), repr(s[:80]))

    @unittest.skipIf(HF is None, "the `tokenizers` package is not installed")
    def test_unicode_patch_is_complete(self):
        """UNICODE_PATCH is exactly where the official pre-tokenizer groups a code point differently from Python's Unicode 14 tables (about 8 s)."""
        self.assertEqual(K.probe_unicode_patch(self.path), [tuple(x) for x in K.UNICODE_PATCH])

    @unittest.skipIf(HF is None, "the `tokenizers` package is not installed")
    def test_special_tokens_in_text(self):
        ref = HF.Tokenizer.from_file(str(self.path))
        for s in ["<｜begin▁of▁sentence｜>hi", "a<｜User｜>b<｜Assistant｜><think>c</think>d<｜end▁of▁sentence｜>", "<｜User｜><｜User｜>", "｜DSML｜", "<dsml:x>",
                  "<｜place▁holder▁no▁7｜>", "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>", "<|EOT|>", "<｜fim▁hole｜>"]:
            self.assertEqual(self.tok.encode(s, add_bos=False), ref.encode(s, add_special_tokens=False).ids, s)

    def test_chat_encoding_is_stable(self):
        ids = self.tok.encode_chat([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[-2:], [128804, 128822])
        self.assertEqual(self.tok.decode(ids), "<｜begin▁of▁sentence｜>Be brief.<｜User｜>What is 2+2?<｜Assistant｜></think>")


if __name__ == "__main__":
    unittest.main()
