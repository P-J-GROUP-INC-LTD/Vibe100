"""Tests for manifest.py / ds41_spec.py / memplan.py.

The real-model tests run on the saved headers of mxxm-t/DeepSeek-V4.1-Flash-GGUF (1,006 tensors) and of
vcruz305/DeepSeek-V4.1-Flash-GGUF (Q2_K family); the file-level tests (absolute offsets, truncation, missing shards)
run on the mini GGUF, whose files really exist.
"""
import contextlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

import fixtures as T
import ds41_spec as S
import gguf_io as G
import manifest as M
import memplan
from gguf_reader import GGUFFile

GIB = 2 ** 30
EXPERT_BLOB = 18_800_640


def analyze_json(path=T.MXXM_HEADERS, mutate=None, **kw):
    recs = T.headers(path)
    if mutate:
        mutate(recs)
    return M.analyze(recs, source="test", **kw)


def find(recs, name):
    for s in recs:
        for t in s.tensors:
            if t.name == name:
                return s, t
    raise KeyError(name)


def codes(F, level="error"):
    return {f.code for f in F.items if f.level == level}


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        rc = M.main(list(argv))
    return rc, out.getvalue()


# ---------------------------------------------------------------------------------------------- mxxm-t
class MxxmHeaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.F = analyze_json()

    def test_passes_with_1006_tensors(self):
        self.assertEqual([f.text for f in self.F.errors], [])
        self.assertEqual(self.m["totals"]["tensors"], 1006)
        self.assertEqual(self.m["totals"]["expected_tensors"], 1006)
        self.assertEqual(self.m["totals"]["file_tensors"], 1006)
        self.assertEqual(len(self.m["tensors"]), 1006)
        self.assertEqual(len(self.m["shards"]), 12)
        self.assertEqual(self.m["sidecar"] and len(self.m["sidecar"]["tensors"]), 78)
        self.assertTrue(self.m["ok"])
        self.assertEqual(self.F.warnings, [])

    def test_byte_totals(self):
        t = self.m["totals"]
        # 40 layers x 384 experts x 18,800,640 B.  (The brief quoted 288,779,059,200; the product is this.)
        self.assertEqual(40 * 384 * EXPERT_BLOB, 288_777_830_400)
        self.assertEqual(t["by_group"]["expert"], 288_777_830_400)
        self.assertEqual(t["expert_bytes_each"], EXPERT_BLOB)
        self.assertEqual(t["experts_total"], 15360)
        # Engram: two tables of 136 B rows (256 values = 8 MXFP4 blocks)
        self.assertEqual(t["engram_row_bytes"], 136)
        self.assertEqual(t["engram_rows"], {"1": 384_006_168, "14": 384_016_682})
        self.assertEqual(t["by_group"]["engram_embed"], (384_006_168 + 384_016_682) * 136)
        self.assertEqual(t["by_group"]["engram_embed"], 104_451_107_600)
        self.assertEqual(t["by_group"]["embd"], 5120 * 129280 * 2)
        dense = sum(v for k, v in t["by_group"].items() if k not in memplan.NON_DENSE)
        self.assertEqual(dense, 8_944_456_128)
        self.assertEqual(t["bytes"], 288_777_830_400 + 104_451_107_600 + 1_323_827_200 + 8_944_456_128)
        self.assertEqual(t["sidecar_bytes"], 7_967_705_480)
        self.assertEqual(t["by_type"], {"F32": 489, "Q8_0": 330, "MXFP4": 122, "BF16": 65})

    def test_per_tensor_geometry(self):
        tn = self.m["tensors"]
        g = tn["blk.7.ffn_gate_exps.weight"]
        self.assertEqual((g["type"], g["dims"], g["nbytes"]), ("MXFP4", [5120, 2304, 384], 384 * 6_266_880))
        d = tn["blk.7.ffn_down_exps.weight"]
        self.assertEqual((d["type"], d["dims"], d["nbytes"]), ("MXFP4", [2304, 5120, 384], 384 * 6_266_880))
        e = tn["blk.14.engram_embed.weight"]
        self.assertEqual((e["type"], e["dims"], e["nbytes"]), ("MXFP4", [256, 384_016_682], 384_016_682 * 136))
        self.assertEqual(g["file"], "DeepSeek-V4.1-Flash-MXFP4-00002-of-00012.gguf")      # layer 7: shard 2
        self.assertEqual(self.m["sidecar"]["tensors"]["markov_w1.weight"]["file"],
                         "DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf")
        for v in tn.values():
            self.assertIsNone(v["abs_offset"])           # headers JSON: the data section start is unknown
            self.assertEqual(v["offset"] % 32, 0)
        self.assertTrue(all(s["data_start"] is None for s in self.m["shards"]))

    def test_layer_modes(self):
        by = {}
        for l in self.m["layers"]:
            by.setdefault(l["mode"], []).append(l["layer"])
        self.assertEqual(by["SWA"], [0, 1])
        self.assertEqual(by["Full"], [2, 8, 14, 20])
        self.assertEqual(by["Reindex"], [24, 28, 32, 36])
        self.assertEqual(len(by["Reuse"]), 30)
        owners = {l["layer"]: l["kv_owner"] for l in self.m["layers"]}
        self.assertEqual((owners[3], owners[7], owners[9], owners[19], owners[21], owners[24], owners[39]),
                         (2, 2, 8, 14, 20, 20, 20))
        self.assertIsNone(owners[0])

    def test_tensors_per_layer_mode(self):
        names = set(self.m["tensors"])

        def layers(suffix):
            return sorted(int(re.match(r"blk\.(\d+)\.", n).group(1)) for n in names if n.endswith(suffix))

        self.assertEqual(layers(".attn_compressor_gate.weight"), [2, 8, 14])             # ratio-2 owners only
        self.assertEqual(layers(".attn_compressor_kv.weight"), [2, 8, 14, 20])
        self.assertEqual(layers(".attn_compressor_norm.weight"), [2, 8, 14, 20])
        self.assertEqual(layers(".indexer_compressor_kv.weight"), [2, 8, 14, 20])
        self.assertEqual(layers(".indexer_compressor_norm.weight"), [2, 8, 14, 20])
        self.assertEqual(layers(".indexer.attn_q_b.weight"), [2, 8, 14, 20, 24, 28, 32, 36])  # the 8 index sources
        self.assertEqual(layers(".indexer.proj.weight"), [2, 8, 14, 20, 24, 28, 32, 36])
        self.assertEqual(layers(".engram_embed.weight"), [1, 14])
        self.assertEqual(layers(".engram_wkv.weight"), [1, 14])
        self.assertEqual(len(layers(".ffn_gate_exps.weight")), 40)

    def test_config_extracted(self):
        c = self.m["config"]
        self.assertEqual((c["n_layer"], c["hidden"], c["n_expert"], c["n_used"], c["ff"]), (40, 5120, 384, 6, 2304))
        self.assertEqual(c["kv_source"], [2, 8, 14, 20])
        self.assertEqual(c["engram_layers"], [1, 14])
        self.assertEqual(self.m["vocab"], 129280)

    def test_cli_ok_and_manifest_file(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.json"
            rc, text = run_cli(str(T.MXXM_HEADERS), "--out", str(out))
            self.assertEqual(rc, 0, text)
            self.assertIn("RESULT: OK", text)
            self.assertIn("Memory plan", text)
            m = json.loads(out.read_text())
            self.assertEqual(m["format"], M.FORMAT)
            self.assertEqual(len(m["tensors"]), 1006)
            self.assertIn("memory_plan", m)

    def test_cli_bad_input(self):
        rc, _ = run_cli("/nonexistent/dir/nothing.gguf")
        self.assertEqual(rc, 1)


class MxxmMutations(unittest.TestCase):
    """Damage the real headers one way at a time; the contract check must name the problem."""

    def err(self, mutate, **kw):
        _, F = analyze_json(mutate=mutate, **kw)
        return F

    def test_missing_gate_on_ratio2_owner(self):
        def mut(recs):
            s, t = find(recs, "blk.8.attn_compressor_gate.weight")
            s.tensors.remove(t)
        F = self.err(mut)
        self.assertIn("TENSOR_MISSING", codes(F))
        self.assertTrue(any("blk.8.attn_compressor_gate.weight" in f.text for f in F.errors))

    def test_gate_on_ratio1_layer_is_unexpected(self):
        def mut(recs):
            recs[7].tensors.append(G.TensorRec("blk.20.attn_compressor_gate.weight", (5120, 512), "BF16", 10 ** 9, 7,
                                               5120 * 512 * 2, None))
        F = self.err(mut)
        self.assertTrue(any(f.code == "TENSOR_UNEXPECTED" and "blk.20.attn_compressor_gate" in f.text for f in F.errors))

    def test_indexer_on_reuse_layer_is_unexpected(self):
        def mut(recs):
            recs[7].tensors.append(G.TensorRec("blk.3.indexer.proj.weight", (5120, 32), "BF16", 10 ** 9, 7, 5120 * 64, None))
        F = self.err(mut)
        self.assertTrue(any(f.code == "TENSOR_UNEXPECTED" and "blk.3.indexer.proj" in f.text for f in F.errors))

    def test_missing_indexer_on_reindex_layer(self):
        def mut(recs):
            s, t = find(recs, "blk.28.indexer.attn_q_b.weight")
            s.tensors.remove(t)
        self.assertTrue(any("blk.28.indexer.attn_q_b" in f.text for f in self.err(mut).errors))

    def test_wrong_dims(self):
        def mut(recs):
            s, t = find(recs, "blk.5.attn_q_a.weight")
            t.dims = (5120, 1281)
        F = self.err(mut)
        self.assertTrue(any(f.code == "TENSOR_SHAPE" and "blk.5.attn_q_a" in f.text for f in F.errors))

    def test_wrong_type(self):
        def mut(recs):
            s, t = find(recs, "blk.9.attn_kv.weight")
            t.type = "BF16"
        F = self.err(mut)
        self.assertTrue(any(f.code == "TENSOR_TYPE" and "blk.9.attn_kv" in f.text for f in F.errors))

    def test_engram_table_must_be_mxfp4(self):
        def mut(recs):
            s, t = find(recs, "blk.1.engram_embed.weight")
            t.type = "Q8_0"
        F = self.err(mut)
        self.assertTrue(any(f.code == "TENSOR_TYPE" and "engram_embed" in f.text for f in F.errors))

    def test_non_mxfp4_experts_refused_unless_allowed(self):
        def mut(recs):
            s, t = find(recs, "blk.0.ffn_gate_exps.weight")
            t.type = "Q4_K"
        F = self.err(mut)
        self.assertIn("EXPERT_TYPE", codes(F))
        F2 = self.err(mut, allow_non_mxfp4=True)
        self.assertNotIn("EXPERT_TYPE", codes(F2))
        self.assertIn("EXPERT_TYPE", codes(F2, "warn"))

    def test_metadata_value_mismatch(self):
        def mut(recs):
            recs[0].kv["deepseek41.expert_used_count"] = 8
        F = self.err(mut)
        self.assertTrue(any(f.code == "META_VALUE" and "expert_used_count" in f.text for f in F.errors))
        # ... but with --geometry self it is simply the file's own value
        _, F2 = analyze_json(mutate=mut, geometry="self")
        self.assertNotIn("META_VALUE", codes(F2))

    def test_wrong_architecture(self):
        def mut(recs):
            recs[0].kv["general.architecture"] = "llama"
        self.assertIn("ARCH", codes(self.err(mut)))

    def test_missing_shard(self):
        def mut(recs):
            del recs[4]
        F = self.err(mut)
        self.assertIn("SHARD_COUNT", codes(F))
        self.assertIn("SHARD_TENSORS", codes(F))

    def test_misaligned_offset(self):
        def mut(recs):
            recs[8].tensors[0].offset = 8
        self.assertIn("OFFSET_ALIGN", codes(self.err(mut)))

    def test_overlapping_tensors(self):
        def mut(recs):
            s, t = find(recs, "blk.0.ffn_down_exps.weight")
            t.offset = 32
        self.assertIn("OFFSET_OVERLAP", codes(self.err(mut)))

    def test_layer_map_inconsistent(self):
        def mut(recs):
            recs[0].kv["deepseek41.attention.kv_source_layer_ids"] = [2, 8, 14]      # L20 no longer a kv source
        F = self.err(mut)
        self.assertTrue(codes(F) & {"LAYERS_IDS", "LAYERS_OWNER"})

    def test_ratio_map_length(self):
        def mut(recs):
            recs[0].kv["deepseek41.attention.compress_ratios"] = [0, 0, 2]
        self.assertIn("LAYERS_RATIOS", codes(self.err(mut)))

    def test_engram_constants(self):
        def mut(recs):
            recs[0].kv["deepseek41.engram.primes"][5] += 1
        self.assertIn("ENGRAM_META", codes(self.err(mut)))

        def mut2(recs):
            recs[0].kv["deepseek41.engram.num_embeddings"][1] += 1
        self.assertIn("ENGRAM_META", codes(self.err(mut2)))

        def mut3(recs):
            recs[0].kv["deepseek41.engram.token_map"] = "<array len 100>"
        self.assertIn("ENGRAM_META", codes(self.err(mut3)))

        def mut4(recs):
            recs[0].kv["deepseek41.engram.multipliers"][2] += 2
        self.assertIn("ENGRAM_MULT", codes(self.err(mut4), "warn"))

    def test_missing_metadata_is_derived_with_a_warning_or_an_error_when_strict(self):
        def mut(recs):
            del recs[0].kv["deepseek41.attention.kv_source_layer_ids"]
        _, F = analyze_json(mutate=mut)
        self.assertEqual(codes(F), set())
        self.assertIn("META_DERIVED", codes(F, "warn"))
        _, F = analyze_json(mutate=mut, strict_metadata=True)
        self.assertEqual(codes(F), {"META_MISSING"})
        # the unmodified mxxm-t headers satisfy even the strict mode; vcruz305 does not
        self.assertEqual(analyze_json(strict_metadata=True)[1].errors, [])
        self.assertIn("META_MISSING", codes(analyze_json(T.VCRUZ_HEADERS, allow_non_mxfp4=True, strict_metadata=True)[1]))

    def test_tokenizer_length(self):
        def mut(recs):
            recs[0].kv["tokenizer.ggml.tokens"] = "<array len 1000>"
        self.assertIn("TOKENIZER", codes(self.err(mut)))

    def test_sidecar_damage_is_a_warning_unless_required(self):
        def mut(recs):
            s, t = find(recs, "markov_w1.weight")
            s.tensors.remove(t)
        _, F = analyze_json(mutate=mut)
        self.assertEqual(F.errors, [])
        self.assertIn("SIDECAR_TENSOR", codes(F, "warn"))
        _, F2 = analyze_json(mutate=mut, require_dspark=True)
        self.assertIn("SIDECAR_TENSOR", codes(F2))

    def test_no_sidecar(self):
        def mut(recs):
            del recs[12]
        m, F = analyze_json(mutate=mut)
        self.assertEqual(F.errors, [])
        self.assertTrue(F.has("SIDECAR_ABSENT", "info"))
        self.assertIsNone(m["sidecar"])
        _, F2 = analyze_json(mutate=mut, require_dspark=True)
        self.assertIn("SIDECAR_ABSENT", codes(F2))


# ---------------------------------------------------------------------------------------------- vcruz305
class VcruzHeaders(unittest.TestCase):
    def test_refused_on_expert_type_with_alias_report(self):
        m, F = analyze_json(T.VCRUZ_HEADERS)
        self.assertEqual({f.code for f in F.errors}, {"EXPERT_TYPE"})
        self.assertIn("Q2_K/Q3_K", F.errors[0].text)
        self.assertIn("--allow-non-mxfp4", F.errors[0].text)
        self.assertFalse(m["ok"])
        # tensor spellings
        ta = m["aliases"]["tensors"]
        self.assertEqual(sorted(re.sub(r"^blk\.\d+\.", "", v) for v in ta.values()),
                         sorted(["engram_embd.weight"] * 2 + ["indexer.attn_k.weight"] * 4 + ["indexer.k_norm.weight"] * 4))
        self.assertEqual(ta["blk.14.engram_embed.weight"], "blk.14.engram_embd.weight")
        self.assertEqual(ta["blk.20.indexer_compressor_kv.weight"], "blk.20.indexer.attn_k.weight")
        self.assertEqual(ta["blk.8.indexer_compressor_norm.weight"], "blk.8.indexer.k_norm.weight")
        self.assertEqual(len(m["ignored_tensors"]), 40)                       # exp_probs_b_vl.bias
        self.assertTrue(F.has("TENSOR_IGNORED", "info"))
        self.assertTrue(all(n.endswith("exp_probs_b_vl.bias") for n in m["ignored_tensors"]))
        self.assertEqual(m["totals"]["tensors"], 1006)
        self.assertEqual(m["totals"]["file_tensors"], 1046)
        # metadata spellings and derived keys
        self.assertEqual(m["aliases"]["metadata"], {
            "deepseek41.engram.head_dim": "deepseek41.engram.key_length",
            "deepseek41.engram.n_heads": "deepseek41.engram.head_count",
            "deepseek41.engram.pad_token_id": "deepseek41.engram.pad_id"})
        d = m["derived_metadata"]
        for k in ("attention.kv_source_layer_ids", "attention.index_source_layer_ids",
                  "attention.candidate_source_layer_id", "attention.candidate_block_size",
                  "attention.candidate_topk_blocks", "engram.compressed_vocab_size"):
            self.assertIn("deepseek41." + k, d)
        self.assertEqual(d["deepseek41.attention.kv_source_layer_ids"], [2, 8, 14, 20])
        self.assertEqual(len([f for f in F.warnings if f.code == "META_DERIVED"]), 7)   # + num_embeddings from shapes
        self.assertTrue(F.has("TENSOR_ALIAS", "info"))

    def test_allowed_with_flag(self):
        m, F = analyze_json(T.VCRUZ_HEADERS, allow_non_mxfp4=True)
        self.assertEqual(F.errors, [])
        self.assertEqual(m["profile"], "non-mxfp4 (allowed)")
        self.assertIn("EXPERT_TYPE", codes(F, "warn"))
        self.assertEqual(m["totals"]["expert_bytes_each"], 12_810_240)            # Q2_K gate/up + Q3_K down

    def test_cli_exit_codes(self):
        rc, text = run_cli(str(T.VCRUZ_HEADERS), "--no-plan")
        self.assertEqual(rc, 2)
        self.assertIn("REFUSED", text)
        self.assertIn("engram_embd", text)
        rc, text = run_cli(str(T.VCRUZ_HEADERS), "--no-plan", "--allow-non-mxfp4")
        self.assertEqual(rc, 0, text)


# ---------------------------------------------------------------------------------------------- real files (mini)
class MiniFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = T.mini()

    def analyze(self, paths=None, **kw):
        recs = [G.load_shard(p, i) for i, p in enumerate(paths or self.res["paths"])]
        return M.analyze(recs, geometry="self", **kw)

    def test_valid(self):
        m, F = self.analyze()
        self.assertEqual([f.text for f in F.errors], [])
        self.assertEqual(F.warnings, [])
        self.assertEqual(m["totals"]["tensors"], len(self.res["tensors"]))
        self.assertEqual(m["vocab"], 512)

    def test_absolute_offsets(self):
        m, F = self.analyze()
        readers = {s["file"]: GGUFFile(s["path"]) for s in m["shards"]}
        for name, t in m["tensors"].items():
            sh = m["shards"][t["shard"]]
            g = readers[sh["file"]]
            self.assertEqual(sh["data_start"], g.data_start)
            self.assertEqual(t["abs_offset"], g.data_start + t["offset"], name)
            self.assertEqual(t["abs_offset"] % sh["alignment"], 0)
            self.assertLessEqual(t["abs_offset"] + t["nbytes"], sh["size"])
        # a shard's data section ends exactly at the file end
        for s in m["shards"]:
            self.assertEqual(s["data_start"] + s["data_bytes"], s["size"])

    def test_alignment_64(self):
        res = T.fresh_mini(alignment=64)
        recs = [G.load_shard(p, i) for i, p in enumerate(res["paths"])]
        m, F = M.analyze(recs, geometry="self")
        self.assertEqual(F.errors, [])
        for s in m["shards"]:
            self.assertEqual((s["alignment"], s["data_start"] % 64), (64, 0))
        for t in m["tensors"].values():
            self.assertEqual(t["abs_offset"] % 64, 0)
        # data really sits where the manifest says
        t = m["tensors"]["blk.2.ffn_gate_exps.weight"]
        path = m["shards"][t["shard"]]["path"]
        import numpy as np
        got = G.read_tensor_f32(path, t["abs_offset"], t["type"], t["dims"])
        self.assertEqual(got.shape, (16, 64, 256))
        self.assertTrue(np.isfinite(got).all())
        self.assertGreater(float(np.abs(got).max()), 0)

    def test_ds41_geometry_rejects_the_mini(self):
        recs = [G.load_shard(p, i) for i, p in enumerate(self.res["paths"])]
        m, F = M.analyze(recs, geometry="ds41")
        self.assertIn("META_VALUE", codes(F))

    def test_truncated_shard(self):
        res = T.fresh_mini()
        p = res["paths"][1]
        data = p.read_bytes()
        p.write_bytes(data[:-100])
        recs = [G.load_shard(q, i) for i, q in enumerate(res["paths"])]
        m, F = M.analyze(recs, geometry="self")
        self.assertIn("SHARD_TRUNCATED", codes(F))

    def test_missing_shard_file(self):
        recs = [G.load_shard(p, i) for i, p in enumerate(self.res["paths"])]
        m, F = M.analyze([recs[0], recs[2]], geometry="self")
        self.assertIn("SHARD_COUNT", codes(F))

    def test_first_shard_pulls_in_the_rest(self):
        files, source = M.collect_inputs([self.res["paths"][0]], "auto")
        self.assertEqual(len(files), 3)
        files, _ = M.collect_inputs([self.res["paths"][0].parent], "auto")
        self.assertEqual(len(files), 3)

    def test_shard_without_metadata_is_refused(self):
        recs = [G.load_shard(p, i) for i, p in enumerate(self.res["paths"])]
        m, F = M.analyze(recs[1:], geometry="self")
        self.assertIn("ARCH", codes(F))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.json"
            rc, text = run_cli(str(self.res["paths"][0]), "--geometry", "self", "--out", str(out), "--vram-gib", "1",
                               "--ram-gib", "1")
            self.assertEqual(rc, 0, text)
            m = json.loads(out.read_text())
            self.assertEqual(m["tensors"]["blk.0.ffn_gate_exps.weight"]["type"], "MXFP4")
            self.assertIsNotNone(m["tensors"]["blk.0.ffn_gate_exps.weight"]["abs_offset"])
            self.assertRegex(m["tensors"]["blk.0.ffn_gate_exps.weight"]["file"], r"-0000\d-of-00003\.gguf$")
            rc, text = run_cli(str(self.res["paths"][0]), "--no-plan")      # real-model geometry: must refuse
            self.assertEqual(rc, 2)

    def test_layer_modes_of_mini(self):
        m, _ = self.analyze()
        modes = {l["layer"]: l["mode"] for l in m["layers"]}
        self.assertEqual(modes, {0: "SWA", 1: "SWA", 2: "Full", 3: "Reuse", 4: "Full", 5: "Reuse", 6: "Reindex",
                                 7: "Reuse"})


# ---------------------------------------------------------------------------------------------- memory plan
class MemoryPlan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, _ = analyze_json()
        cls.plan = M.make_plan(cls.m, vram_gib=32, ram_gib=384, numa_nodes=2)

    def test_user_box(self):
        p = self.plan
        g, r = p["gpu"], p["ram"]
        self.assertEqual(g["dense_bytes"], 8_944_456_128)
        self.assertAlmostEqual(g["dense_bytes"] / GIB, 8.33, places=2)                   # PLAN.md table
        self.assertEqual(g["kv_bytes"], 131072 * 3200)
        budget = 32 * GIB - g["dense_bytes"] - g["kv_bytes"] - 3 * GIB
        self.assertEqual(g["cache_budget_bytes"], budget)
        self.assertEqual(g["experts_fit"], budget // EXPERT_BLOB)
        self.assertEqual(g["experts_fit"], 1158)
        self.assertAlmostEqual(r["items"]["routed_experts"] / GIB, 268.95, places=2)
        self.assertAlmostEqual(r["engram_bytes"] / GIB, 97.28, places=2)
        self.assertAlmostEqual(r["items"]["token_embd_host"] / GIB, 1.23, places=2)
        self.assertAlmostEqual(r["usable_bytes"] / GIB, 377.09, places=1)                 # "~377 GiB usable"
        self.assertGreater(r["page_cache_bytes"] / GIB, 85)                               # PLAN: ~85-95
        self.assertLess(r["page_cache_bytes"] / GIB, 97.28)
        self.assertEqual(len(r["per_node"]), 2)
        self.assertEqual(r["per_node"][0]["experts_bytes"], 288_777_830_400 // 2)         # 134.47 GiB each
        self.assertAlmostEqual(r["per_node"][0]["experts_bytes"] / GIB, 134.47, places=2)
        self.assertEqual(p["engram"]["bytes_per_token"], 48 * 136)

    def test_cache_table_and_explicit_size(self):
        p = M.make_plan(self.m, cache_gib=21.5)
        sizes = {t["cache_gib"]: t["experts"] for t in p["cache_table"]}
        self.assertEqual(sizes[20], int(20 * GIB // EXPERT_BLOB))
        self.assertEqual(sizes[21.5], int(21.5 * GIB // EXPERT_BLOB))
        self.assertEqual(p["explicit_cache"]["experts"], sizes[21.5])
        text = memplan.format_plan(self.plan)
        self.assertIn("1158 experts", text)
        self.assertIn("left for Engram's page cache", text)

    def test_small_gpu_and_ram_warn(self):
        p = M.make_plan(self.m, vram_gib=8, ram_gib=200)
        self.assertEqual(p["gpu"]["experts_fit"], 0)
        self.assertTrue(any("nothing" in w for w in p["warnings"]))
        self.assertTrue(any("do not fit in RAM" in w for w in p["warnings"]))

    def test_numa_and_options(self):
        p1 = M.make_plan(self.m, numa_nodes=1)
        self.assertEqual(p1["ram"]["per_node"][0]["experts_bytes"], 288_777_830_400)
        p4 = M.make_plan(self.m, numa_nodes=4)
        self.assertTrue(any("1 or 2 nodes" in w for w in p4["warnings"]))
        base = self.plan["ram"]["page_cache_bytes"]
        withd = M.make_plan(self.m, dspark_on_cpu=True)["ram"]["page_cache_bytes"]
        self.assertEqual(base - withd, 7_967_705_480)
        on_gpu = M.make_plan(self.m, embd_on_gpu=True)
        self.assertEqual(on_gpu["gpu"]["dense_bytes"], self.plan["gpu"]["dense_bytes"] + 1_323_827_200)
        self.assertEqual(on_gpu["ram"]["items"]["token_embd_host"], 0)


# ---------------------------------------------------------------------------------------------- contract pins
class Pins(unittest.TestCase):
    def test_defaults_match_config_json(self):
        cfg = json.loads((T.REF / "config.json").read_text())["text_config"]
        d = {k.field: k.default for k in S.KEYS}
        self.assertEqual(d["n_layer"], cfg["num_hidden_layers"])
        self.assertEqual(d["hidden"], cfg["hidden_size"])
        self.assertEqual(d["n_expert"], cfg["n_routed_experts"])
        self.assertEqual(d["n_used"], cfg["num_experts_per_tok"])
        self.assertEqual(d["ff"], cfg["moe_intermediate_size"])
        self.assertEqual(d["n_head"], cfg["num_attention_heads"])
        self.assertEqual(d["head_dim"], cfg["head_dim"])
        self.assertEqual(d["q_lora"], cfg["q_lora_rank"])
        self.assertEqual(d["o_lora"], cfg["o_lora_rank"])
        self.assertEqual(d["o_groups"], cfg["o_groups"])
        self.assertEqual(d["sliding_window"], cfg["sliding_window"])
        self.assertEqual(d["rope_dim"], cfg["qk_rope_head_dim"])
        self.assertEqual(d["ratios"], cfg["compress_ratios"])
        self.assertEqual(d["kv_source"], cfg["kv_source_layer_ids"])
        self.assertEqual(d["index_source"], cfg["index_source_layer_ids"])
        self.assertEqual(d["cand_source"], cfg["candidate_source_layer_id"])
        self.assertEqual(d["cand_block"], cfg["candidate_block_size"])
        self.assertEqual(d["cand_topk"], cfg["candidate_topk_blocks"])
        self.assertEqual(d["idx_heads"], cfg["index_n_heads"])
        self.assertEqual(d["idx_dim"], cfg["index_head_dim"])
        self.assertEqual(d["idx_topk"], cfg["index_topk"])
        self.assertEqual(d["hc"], cfg["hc_mult"])
        self.assertEqual(d["hc_iters"], cfg["hc_sinkhorn_iters"])
        self.assertEqual(d["engram_layers"], cfg["engram_layer_ids"])
        self.assertEqual(d["eh_dim"], cfg["engram_head_dim"])
        self.assertEqual(d["e_heads"], cfg["engram_n_heads"])
        self.assertEqual(d["e_ngram"], cfg["engram_max_ngram_size"])
        self.assertEqual(d["e_pad"], cfg["engram_pad_token_id"])
        self.assertEqual(d["e_cvocab"], cfg["engram_compressed_vocab_size"])
        self.assertEqual(S.ENGRAM_VOCAB_SIZE, cfg["engram_vocab_size"])
        self.assertEqual(d["route_scale"], cfg["routed_scaling_factor"])
        self.assertEqual(d["rope_factor"], cfg["rope_scaling"]["factor"])
        self.assertEqual(d["rope_orig_ctx"], cfg["rope_scaling"]["original_max_position_embeddings"])
        self.assertEqual(d["compress_rope_base"], cfg["compress_rope_theta"])

    def test_geometry_hpp_agrees(self):
        hpp = (T.REPO / "include" / "strata" / "ds41" / "geometry.hpp")
        if not hpp.is_file():
            self.skipTest("geometry.hpp not present")
        text = hpp.read_text()

        def const(name):
            m = re.search(r"constexpr\s+\w+\s+" + name + r"\s*=\s*([0-9.]+)f?;", text)
            self.assertIsNotNone(m, name)
            return float(m.group(1)) if "." in m.group(1) else int(m.group(1))

        cfg = {k.field: k.expect for k in S.KEYS}
        self.assertEqual(const("kLayers"), cfg["n_layer"])
        self.assertEqual(const("kHidden"), cfg["hidden"])
        self.assertEqual(const("kExperts"), cfg["n_expert"])
        self.assertEqual(const("kTopK"), cfg["n_used"])
        self.assertEqual(const("kFF"), cfg["ff"])
        self.assertEqual(const("kRouteScale"), cfg["route_scale"])
        self.assertEqual(const("kSwigluLimit"), 10.0)
        m = re.search(r"static_assert\(kBlobBytes == (\d+)", text)
        self.assertEqual(int(m.group(1)), EXPERT_BLOB)


if __name__ == "__main__":
    unittest.main()
