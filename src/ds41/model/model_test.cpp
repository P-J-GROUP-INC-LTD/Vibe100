// src/ds41/model/model_test.cpp - DS1-A: the loader on the mini deepseek41 GGUF (3 shards), against readers that share no code with it.
//
//   ds41_model_test <fixture dir>        (written by src/ds41/model/model_fixture.py, a ctest fixture; see cmake/ds41_model.cmake)
//
// What is checked, and against what:
//   shards / metadata / tokens     every key of shard 1 and the split keys of every shard           tools/gguf_reader.py (via gguf_io.py)
//   tensor table                   name, type, dims, shard, absolute offset, bytes of all 218       tools/ds41/gguf_io.py load_shard
//   Ds41Config, layer roles        every field the oracle reads, the layer map, the Engram arrays   ref/ds41/config.py (+ rope.py tables)
//   host dequantisation            every dense tensor whole, token_embd / Engram table rows         ref/ds41/weights.py, bit for bit
//   expert layouts                 CPU halves and blob of six experts                                tools/ds41/expert_layout.py, bit for bit
//   the load                       device tensors == the mapped bytes, role-dependent presence, the arena (all 128 experts), a filled cache,
//                                  the residency table, the memory plan, every device allocation given back        (HostDev = the emulator's Dev)
//   refusals                       dims != G, a changed dim, a wrong type, a missing / extra tensor, bad metadata, a truncated or missing shard
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/model/model.hpp"

using namespace strata::ds41;
using namespace strata::ds41::model;
namespace fs = std::filesystem;
namespace platform = strata::platform;

namespace {

int g_checks = 0, g_fail = 0;

#define CHECK(...)                                                                        \
    do {                                                                                  \
        ++g_checks;                                                                       \
        if (!(__VA_ARGS__)) {                                                             \
            ++g_fail;                                                                     \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__);   \
        }                                                                                 \
    } while (0)

#define CHECK_EQ_STR(a, b)                                                                                        \
    do {                                                                                                          \
        ++g_checks;                                                                                               \
        const std::string _a = (a), _b = (b);                                                                     \
        if (_a != _b) {                                                                                           \
            ++g_fail;                                                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s == %s\n  got      `%s`\n  expected `%s`\n", __FILE__, __LINE__, #a, #b, _a.c_str(), _b.c_str()); \
        }                                                                                                         \
    } while (0)

/// `fn` must throw a ModelError whose message contains every needle.
void expect_refusal(const char* what, const std::function<void()>& fn, std::initializer_list<const char*> needles) {
    ++g_checks;
    try {
        fn();
        ++g_fail;
        std::fprintf(stderr, "FAIL %s: nothing was refused\n", what);
    } catch (const ModelError& e) {
        const std::string m = e.what();
        for (const char* n : needles)
            if (m.find(n) == std::string::npos) {
                ++g_fail;
                std::fprintf(stderr, "FAIL %s: the refusal does not say `%s`:\n%s\n", what, n, m.c_str());
                return;
            }
    } catch (const std::exception& e) {
        ++g_fail;
        std::fprintf(stderr, "FAIL %s: a %s that is not a ModelError: %s\n", what, typeid(e).name(), e.what());
    }
}

// ---------------------------------------------------------------------------------------------- small file helpers
std::vector<std::string> read_lines(const fs::path& p) {
    std::ifstream f(p);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", p.string().c_str());
        std::exit(2);
    }
    std::vector<std::string> out;
    std::string l;
    while (std::getline(f, l))
        if (!l.empty()) out.push_back(l);
    return out;
}

std::vector<std::string> words(const std::string& l) {
    std::istringstream is(l);
    std::vector<std::string> w;
    std::string x;
    while (is >> x) w.push_back(x);
    return w;
}

std::vector<uint8_t> read_bin(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", p.string().c_str());
        std::exit(2);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void write_bin(const fs::path& p, const std::vector<uint8_t>& v) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), (std::streamsize) v.size());
}

std::vector<int64_t> to_ints(const std::vector<std::string>& w, size_t from) {
    std::vector<int64_t> v;
    for (size_t i = from; i < w.size(); ++i) v.push_back(std::stoll(w[i]));
    return v;
}
std::vector<double> to_doubles(const std::vector<std::string>& w, size_t from) {
    std::vector<double> v;
    for (size_t i = from; i < w.size(); ++i) v.push_back(std::strtod(w[i].c_str(), nullptr));
    return v;
}

struct Fx {
    fs::path dir, gold;
    std::string shard1;
};

// ---------------------------------------------------------------------------------------------- 1. shards, metadata, tokens
void test_shards_and_meta(const Fx& fx, const GgufSet& g) {
    const auto shards = read_lines(fx.gold / "shards.txt");
    CHECK(g.n_shards() == shards.size() && g.n_shards() == 3);
    for (size_t i = 0; i < shards.size() && i < g.n_shards(); ++i) {
        const auto w = words(shards[i]);       // index file size alignment data_start split.no split.count split.tensors.count
        const ShardInfo& s = g.shard(i);
        CHECK(fs::path(s.path).filename().string() == w[1]);
        CHECK(s.size == std::stoull(w[2]));
        CHECK(s.alignment == std::stoull(w[3]));
        CHECK(s.data_start == std::stoull(w[4]));
        CHECK(s.version == 3);
        const MetaStore& m = g.shard_meta(i);
        CHECK(m.get_int("split.no") == std::stoll(w[5]));
        CHECK(m.get_int("split.count") == std::stoll(w[6]));
        CHECK(m.get_int("split.tensors.count") == std::stoll(w[7]));
    }
    // every metadata key of shard 1
    const MetaStore& m = g.meta();
    size_t n = 0;
    for (const std::string& line : read_lines(fx.gold / "meta.txt")) {
        const size_t a = line.find(' ', 2), b = line.find(' ', a + 1);      // "K key kind rest"
        const std::string key = line.substr(2, a - 2), kind = line.substr(a + 1, b - a - 1), rest = b == std::string::npos ? "" : line.substr(b + 1);
        const auto w = words(rest);
        ++n;
        if (!m.has(key)) {
            CHECK(!"a key the independent reader found is missing");
            std::fprintf(stderr, "  key %s\n", key.c_str());
            continue;
        }
        if (kind == "s") CHECK(m.get_string(key) == rest);
        else if (kind == "i") CHECK(m.get_int(key) == std::stoll(rest));
        else if (kind == "b") CHECK(m.get_bool(key) == (rest == "1"));
        else if (kind == "f") CHECK(m.get_float(key) == std::strtod(rest.c_str(), nullptr));
        else if (kind == "ai") CHECK(m.get_ints(key) == to_ints(w, 1));
        else if (kind == "af") CHECK(m.get_floats(key) == to_doubles(w, 1));
        else if (kind == "as") CHECK(m.array_count(key) == std::stoull(w[0]));
        else CHECK(!"unknown golden kind");
    }
    CHECK(n > 50);
    CHECK(m.all().size() >= n);
    // string arrays are read back on demand
    const std::vector<std::string> toks = g.read_string_array("tokenizer.ggml.tokens");
    CHECK(toks.size() == 512);
    for (const std::string& line : read_lines(fx.gold / "tokens.txt")) {
        const size_t sp = line.find(' ');
        const size_t i = std::stoull(line.substr(0, sp));
        CHECK(i < toks.size() && toks[i] == line.substr(sp + 1));
    }
    // typed access refuses what is not that type, naming the key
    expect_refusal("get_int of a string", [&] { (void) m.get_int("general.name"); }, {"general.name", "integer"});
    expect_refusal("missing key", [&] { (void) m.get_int("deepseek41.no_such_key"); }, {"deepseek41.no_such_key", "missing"});
    expect_refusal("get_string of an int", [&] { (void) m.get_string("deepseek41.block_count"); }, {"deepseek41.block_count", "string"});
    expect_refusal("read_string_array of a numeric array", [&] { (void) g.read_string_array("deepseek41.engram.primes"); }, {"deepseek41.engram.primes"});
}

// ---------------------------------------------------------------------------------------------- 2. the tensor table
void test_tensor_table(const Fx& fx, const GgufSet& g) {
    const TensorDir dir = g.directory();
    const auto lines = read_lines(fx.gold / "tensors.txt");
    CHECK(dir.size() == lines.size() && dir.size() == 218);
    CHECK(g.tensors().size() == dir.size());
    std::set<std::string> seen;
    for (const std::string& line : lines) {
        const auto w = words(line);     // name type n_dims ne0 ne1 ne2 shard abs_offset nbytes
        const TensorLoc* t = dir.find(w[0]);
        if (!t) {
            CHECK(!"a tensor of the independent read is missing from the directory");
            std::fprintf(stderr, "  %s\n", w[0].c_str());
            continue;
        }
        seen.insert(w[0]);
        CHECK(type_name(t->type) == w[1]);
        CHECK(t->n_dims == std::stoi(w[2]));
        CHECK(t->ne[0] == std::stoull(w[3]) && t->ne[1] == std::stoull(w[4]) && t->ne[2] == std::stoull(w[5]));
        CHECK(t->shard == std::stoi(w[6]));
        CHECK(t->abs_offset == std::stoull(w[7]));
        CHECK(t->nbytes == std::stoull(w[8]));
        CHECK(t->abs_offset + t->nbytes <= g.shard((size_t) t->shard).size);
        CHECK(g.data(*t) == g.shard_base((size_t) t->shard) + t->abs_offset);
        CHECK(t->name == w[0]);
    }
    CHECK(seen.size() == dir.size());
    // dims text
    CHECK_EQ_STR(dir.at("blk.0.ffn_gate_exps.weight").dims_text(), "[256, 256, 16]");
    expect_refusal("TensorDir::at on a missing name", [&] { (void) dir.at("blk.99.nothing"); }, {"blk.99.nothing", "missing"});
    TensorDir twice;
    twice.add(dir.at("token_embd.weight"));
    expect_refusal("a duplicate tensor name", [&] { twice.add(dir.at("token_embd.weight")); }, {"token_embd.weight", "twice"});
    // the type table
    int be = 0, bb = 0;
    CHECK(type_block(GgmlType::Q8_0, be, bb) && be == 32 && bb == 34);
    CHECK(type_block(GgmlType::MXFP4, be, bb) && be == 32 && bb == 17);
    CHECK(type_block(GgmlType::BF16, be, bb) && be == 1 && bb == 2);
    CHECK(type_block(GgmlType::F32, be, bb) && be == 1 && bb == 4);
    const uint64_t ne[4] = {5120, 2304, 384, 1};
    CHECK(tensor_nbytes(GgmlType::MXFP4, ne, 3) == 5120ull / 32 * 17 * 2304 * 384);
    const uint64_t odd[4] = {100, 4, 1, 1};
    CHECK(tensor_nbytes(GgmlType::Q8_0, odd, 2) == 0);                       // 100 is not a whole number of blocks
    CHECK(tensor_nbytes((GgmlType) 9999, ne, 3) == 0);
    CHECK_EQ_STR(type_name((GgmlType) 9999), "type9999");
}

// ---------------------------------------------------------------------------------------------- 3. the configuration
std::string lower(std::string s) {
    for (char& c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

void test_config(const Fx& fx, const GgufSet& g) {
    const Ds41Config c = read_config(g.meta());
    std::map<std::string, std::vector<std::string>> oracle;
    for (const std::string& line : read_lines(fx.gold / "config.txt")) {
        const auto w = words(line);
        oracle[w[0]] = std::vector<std::string>(w.begin() + 1, w.end());
    }
    const auto I = [&](const char* k) { return std::stoll(oracle.at(k).at(0)); };
    const auto F = [&](const char* k) { return std::strtod(oracle.at(k).at(0).c_str(), nullptr); };
    const auto L = [&](const char* k) {
        std::vector<int> v;
        for (int64_t x : to_ints(oracle.at(k), 0)) v.push_back((int) x);
        return v;
    };
    CHECK(c.architecture == "deepseek41");
    CHECK(c.vocab == I("vocab_size") && c.hidden == I("dim") && c.ff == I("moe_inter_dim") && c.n_layer == I("n_layers"));
    CHECK(c.n_head == I("n_heads") && c.head_dim == I("head_dim") && c.rope_dim == I("rope_head_dim") && c.q_lora == I("q_lora_rank"));
    CHECK(c.n_expert == I("n_routed_experts") && c.n_shared == I("n_shared_experts") && c.n_used == I("n_activated_experts"));
    CHECK(c.route_scale == (float) F("route_scale") && c.swiglu_limit == (float) F("swiglu_limit") && c.rms_eps == (float) F("norm_eps"));
    CHECK(c.route_norm == (I("norm_topk_prob") != 0) && c.window == I("window_size") && c.o_groups == I("o_groups") && c.o_lora == I("o_lora_rank"));
    CHECK(c.rope_swa.base == F("rope_theta") && c.rope_csa.base == F("compress_rope_theta") && c.rope_csa.factor == F("rope_factor"));
    CHECK(c.rope_csa.orig_ctx == I("original_seq_len") && c.rope_csa.beta_fast == F("beta_fast") && c.rope_csa.beta_slow == F("beta_slow"));
    CHECK(!c.rope_swa.yarn && c.rope_csa.yarn && c.rope_swa.orig_ctx == 0);
    CHECK(c.idx_heads == I("index_n_heads") && c.idx_dim == I("index_head_dim") && c.idx_topk == I("index_topk"));
    CHECK(c.cand_source == I("candidate_source_layer") && c.cand_block == I("candidate_block_size") && c.cand_topk_blocks == I("candidate_topk_blocks"));
    CHECK(c.hc == I("hc_mult") && c.hc_iters == I("hc_sinkhorn_iters") && c.hc_eps == (float) F("hc_eps"));
    const std::vector<int> all_ratios = L("compress_ratios");              // the file's 11 entries: 8 layers + the 3 DSpark ones
    CHECK(all_ratios.size() == 11 && std::equal(c.compress_ratios.begin(), c.compress_ratios.end(), all_ratios.begin()));
    CHECK(c.kv_source == L("kv_source_layers") && c.index_source == L("index_source_layers"));
    CHECK(c.n_ratio_entries == 11 && (int) c.compress_ratios.size() == c.n_layer);     // 8 layers + the 3 DSpark entries of the file
    CHECK(c.n_kv_head == 1 && c.value_dim == c.head_dim && c.gating_func == 4 && c.n_hc_mixes() == 24 && c.n_q() == c.n_head * c.head_dim);
    // Engram
    const EngramConfig& e = c.engram;
    CHECK(e.layers == L("engram_layer_ids") && e.heads == I("engram_n_heads") && e.head_dim == I("engram_head_dim") && e.ngram == I("engram_max_ngram_size"));
    CHECK(e.pad_id == I("engram_pad_id") && e.cvocab == I("engram_compressed_vocab_size"));
    CHECK(e.num_embeddings == to_ints(oracle.at("engram_num_embeddings"), 0));
    CHECK(e.cols() == (e.ngram - 1) * e.heads && e.cols() == 6);
    CHECK(e.primes == g.meta().get_ints("deepseek41.engram.primes") && e.offsets == g.meta().get_ints("deepseek41.engram.offsets"));
    CHECK(e.multipliers == g.meta().get_ints("deepseek41.engram.multipliers"));
    const std::vector<int64_t> tm = g.meta().get_ints("deepseek41.engram.token_map");
    CHECK(e.token_map.size() == tm.size() && (int) e.token_map.size() == c.vocab);
    bool same = e.token_map.size() == tm.size();
    for (size_t i = 0; same && i < tm.size(); ++i) same = e.token_map[i] == tm[i];
    CHECK(same);
    CHECK(e.pad_compressed() == e.token_map[(size_t) e.pad_id]);
    CHECK(e.prime(1, 2) == e.primes[(size_t) (1 * e.cols() + 2)] && e.offset(1, 0) == 0 && e.multiplier(1, 3) == e.multipliers[(size_t) (1 * e.ngram + 3)]);
    CHECK(e.offset(0, 1) == e.prime(0, 0));
    // layer roles: the oracle's layer_modes, then the derived ownership (hand-derived for the mini file: ratios 0 0 2 2 1 1 1 1)
    const auto modes = read_lines(fx.gold / "modes.txt");
    CHECK((int) modes.size() == c.n_layer && (int) c.layers.size() == c.n_layer);
    for (const std::string& line : modes) {
        const auto w = words(line);
        CHECK(lower(c.layer(std::stoi(w[0])).mode()) == w[1]);
    }
    struct Want {
        Role role;
        int ratio, kv_owner, idx_owner, topk, engram;
        bool comp, gate, indexer, idx_comp, cand_src, cand_pool;
    };
    const Want want[8] = {
        {Role::SWA, 0, -1, -1, -1, -1, false, false, false, false, false, false},
        {Role::SWA, 0, -1, -1, -1, 0, false, false, false, false, false, false},
        {Role::FULL, 2, 2, 2, 2, -1, true, true, true, true, false, false},
        {Role::REUSE, 2, 2, -1, 2, 1, false, false, false, false, false, false},
        {Role::FULL, 1, 4, 4, 4, -1, true, false, true, true, true, false},
        {Role::REUSE, 1, 4, -1, 4, -1, false, false, false, false, false, false},
        {Role::REINDEX, 1, 4, 4, 6, -1, false, false, true, false, false, true},
        {Role::REUSE, 1, 4, -1, 6, -1, false, false, false, false, false, false},
    };
    for (int l = 0; l < 8; ++l) {
        const LayerInfo& li = c.layer(l);
        const Want& w = want[l];
        CHECK(li.layer == l && li.role == w.role && li.ratio == w.ratio);
        CHECK(li.kv_owner == w.kv_owner && li.index_k_owner == w.idx_owner && li.topk_source == w.topk && li.engram_slot == w.engram);
        CHECK(li.has_compressor == w.comp && li.has_compressor_gate == w.gate && li.has_indexer == w.indexer && li.has_index_compressor == w.idx_comp);
        CHECK(li.is_candidate_source == w.cand_src && li.uses_candidate_pool == w.cand_pool && li.is_engram() == (w.engram >= 0));
        CHECK(&c.rope_of(l) == (w.ratio ? &c.rope_csa : &c.rope_swa));
    }
    CHECK(c.n_engram() == 2);
    CHECK_EQ_STR(role_name(Role::REINDEX), "REINDEX");
    const std::string d = c.describe();
    CHECK(d.find("REINDEX 6") != std::string::npos && d.find("Engram: layers 1,3") != std::string::npos && d.find("layer roles") != std::string::npos);

    // RoPE tables against the oracle's rope_table (float64 in the oracle, float32 here: one float32 rounding apart at most)
    for (int kind = 0; kind < 2; ++kind) {
        const RopeTable t = build_rope_table(kind ? c.rope_csa : c.rope_swa, c.rope_dim, 64);
        const std::vector<uint8_t> raw = read_bin(fx.gold / (kind ? "rope_csa.f32" : "rope_swa.f32"));
        const size_t half = (size_t) c.rope_dim / 2 * 64;
        CHECK(raw.size() == 2 * half * 4 && t.cos.size() == half && t.sin.size() == half && t.dim == c.rope_dim && t.n_pos == 64);
        double worst = 0;
        for (size_t i = 0; i < half; ++i) {
            float gc, gs;
            std::memcpy(&gc, raw.data() + 4 * i, 4);
            std::memcpy(&gs, raw.data() + 4 * (half + i), 4);
            worst = std::max({worst, (double) std::fabs(gc - t.cos[i]), (double) std::fabs(gs - t.sin[i])});
        }
        CHECK(worst <= 1.3e-7);
    }

    // the geometry check: the mini file is MiniGeom, not RealGeom; the refusal says which key and what both sides have
    require_geometry<MiniGeom>(c);
    Findings f;
    check_geometry<MiniGeom>(c, f);
    CHECK(f.ok());
    expect_refusal("RealGeom against the mini file", [&] { require_geometry<RealGeom>(c); },
                   {"does not match the compiled geometry 'real'", "`deepseek41.embedding_length` = 256", "expects 5120 (G::kHidden)", "`deepseek41.block_count` = 8", "expects 40",
                    "deepseek41.expert_count"});
    Findings fr;
    check_geometry<RealGeom>(c, fr);
    CHECK(fr.errors.size() >= 15);
    // a config that differs from MiniGeom in ONE dimension names exactly that one
    for (int which = 0; which < 3; ++which) {
        Ds41Config bad = c;
        const char* key = which == 0 ? "deepseek41.attention.head_count" : which == 1 ? "deepseek41.expert_feed_forward_length" : "deepseek41.engram.n_heads";
        if (which == 0) bad.n_head = 8;
        if (which == 1) bad.ff = 512;
        if (which == 2) bad.engram.heads = 4;
        Findings ff;
        check_geometry<MiniGeom>(bad, ff);
        CHECK(ff.errors.size() == 1 && ff.errors[0].find(key) != std::string::npos);
    }
}

// metadata refusals: each mutation of a copy of shard 1's metadata is refused with the key named
void test_config_refusals(const GgufSet& g) {
    const auto mutated = [&](const std::function<void(MetaStore&)>& fn) {
        MetaStore m = g.meta();
        fn(m);
        return m;
    };
    const auto setint = [](MetaStore& m, const char* key, int64_t v) {
        MetaValue x;
        x.type = MetaType::I64;
        x.i = v;
        m.set(key, x);
    };
    const auto setf = [](MetaStore& m, const char* key, double v) {
        MetaValue x;
        x.type = MetaType::F64;
        x.f = v;
        m.set(key, x);
    };
    const auto setints = [](MetaStore& m, const char* key, std::vector<int64_t> v) {
        MetaValue x;
        x.type = MetaType::Array;
        x.elem = MetaType::I64;
        x.count = v.size();
        x.ia = std::move(v);
        m.set(key, x);
    };
    // missing keys: all of them at once, not just the first
    expect_refusal("two missing keys", [&] {
        MetaStore m = g.meta();
        MetaStore n("tampered");
        for (const auto& kv : m.all())
            if (kv.first != "deepseek41.expert_count" && kv.first != "deepseek41.attention.q_lora_rank") n.set(kv.first, kv.second);
        (void) read_config(n);
    }, {"deepseek41.expert_count", "deepseek41.attention.q_lora_rank", "missing"});
    expect_refusal("another architecture", [&] { (void) read_config(mutated([](MetaStore& m) { MetaValue v; v.type = MetaType::String; v.s = "qwen3next"; m.set("general.architecture", v); })); },
                   {"qwen3next", "deepseek41"});
    expect_refusal("another gate", [&] { (void) read_config(mutated([&](MetaStore& m) { setint(m, "deepseek41.expert_gating_func", 1); })); }, {"expert_gating_func", "sqrtsoftplus"});
    expect_refusal("another route scale", [&] { (void) read_config(mutated([&](MetaStore& m) { setf(m, "deepseek41.expert_weights_scale", 2.5); })); }, {"expert_weights_scale"});
    expect_refusal("hash layers", [&] { (void) read_config(mutated([&](MetaStore& m) { setint(m, "deepseek41.hash_layer_count", 3); })); }, {"hash_layer_count", "not implemented"});
    expect_refusal("two shared experts", [&] { (void) read_config(mutated([&](MetaStore& m) { setint(m, "deepseek41.expert_shared_count", 2); })); }, {"expert_shared_count"});
    expect_refusal("GQA", [&] { (void) read_config(mutated([&](MetaStore& m) { setint(m, "deepseek41.attention.head_count_kv", 4); })); }, {"head_count_kv", "MQA"});
    expect_refusal("a swiglu clamp that differs", [&] {
        (void) read_config(mutated([&](MetaStore& m) {
            MetaValue v = *m.find("deepseek41.swiglu_clamp_exp");
            v.fa[3] = 7.0;
            m.set("deepseek41.swiglu_clamp_exp", v);
        }));
    }, {"swiglu_clamp", "layer 3"});
    expect_refusal("compress_ratios of the wrong length", [&] { (void) read_config(mutated([&](MetaStore& m) { setints(m, "deepseek41.attention.compress_ratios", {0, 0, 2}); })); },
                   {"compress_ratios", "3 entries", "expected 8"});
    expect_refusal("a kv source that does not compress", [&] { (void) read_config(mutated([&](MetaStore& m) { setints(m, "deepseek41.attention.kv_source_layer_ids", {0, 2}); })); },
                   {"kv source layer 0", "compress ratio 0"});
    expect_refusal("a candidate source that is not a kv source", [&] { (void) read_config(mutated([&](MetaStore& m) { setint(m, "deepseek41.attention.candidate_source_layer_id", 6); })); },
                   {"candidate_source_layer_id", "kv source"});
    expect_refusal("a layer id out of range", [&] { (void) read_config(mutated([&](MetaStore& m) { setints(m, "deepseek41.engram.layer_ids", {1, 9}); })); }, {"engram.layer_ids", "layer 9"});
    expect_refusal("primes that are not consecutive", [&] {
        (void) read_config(mutated([&](MetaStore& m) {
            std::vector<int64_t> p = m.get_ints("deepseek41.engram.primes");
            p[2] += 2;
            setints(m, "deepseek41.engram.primes", p);
        }));
    }, {"engram.primes"});
    expect_refusal("offsets that are not running sums", [&] {
        (void) read_config(mutated([&](MetaStore& m) {
            std::vector<int64_t> p = m.get_ints("deepseek41.engram.offsets");
            p[1] += 1;
            setints(m, "deepseek41.engram.offsets", p);
        }));
    }, {"engram.offsets", "running sums"});
    expect_refusal("a token map outside the compressed vocabulary", [&] {
        (void) read_config(mutated([&](MetaStore& m) {
            std::vector<int64_t> t = m.get_ints("deepseek41.engram.token_map");
            t[5] = 100000;
            setints(m, "deepseek41.engram.token_map", t);
        }));
    }, {"token_map", "compressed_vocab_size"});
    expect_refusal("a token map of the wrong length", [&] {
        (void) read_config(mutated([&](MetaStore& m) {
            std::vector<int64_t> t = m.get_ints("deepseek41.engram.token_map");
            t.pop_back();
            setints(m, "deepseek41.engram.token_map", t);
        }));
    }, {"token_map", "511 entries"});
    // without the big arrays: a headers-only metadata is accepted and the map is simply empty
    const Ds41Config light = read_config(g.meta(), false);
    CHECK(light.engram.token_map.empty() && light.engram.pad_compressed() == -1);
}

// ---------------------------------------------------------------------------------------------- 4. tensor validation
void test_validation(const GgufSet& g) {
    const Ds41Config c = read_config(g.meta());
    const TensorDir dir = g.directory();
    const std::vector<TensorSpec> specs = expected_tensors(c);
    CHECK(specs.size() == dir.size());
    Findings f;
    size_t ignored = 7;
    validate_tensors(c, dir, f, {}, &ignored);
    CHECK(f.ok() && f.warnings.empty() && ignored == 0);
    if (!f.ok())
        for (const std::string& e : f.errors) std::fprintf(stderr, "  unexpected: %s\n", e.c_str());

    const auto errors_of = [&](const TensorDir& d, bool allow = false) {
        Findings ff;
        ValidateOptions vo;
        vo.allow_unexpected = allow;
        validate_tensors(c, d, ff, vo);
        return ff;
    };
    const auto without = [&](const std::string& name) {
        TensorDir d;
        for (const auto& kv : dir.all())
            if (kv.first != name) d.add(kv.second);
        return d;
    };
    const auto changed = [&](const std::string& name, const std::function<void(TensorLoc&)>& fn) {
        TensorDir d;
        for (const auto& kv : dir.all()) {
            TensorLoc t = kv.second;
            if (t.name == name) fn(t);
            d.add(t);
        }
        return d;
    };
    {   // dims != the metadata's: named, both shapes shown
        const Findings ff = errors_of(changed("blk.2.attn_q_a.weight", [](TensorLoc& t) { t.ne[0] = 128; t.ne[1] = 128; }));
        CHECK(ff.errors.size() == 1 && ff.errors[0].find("blk.2.attn_q_a.weight") != std::string::npos && ff.errors[0].find("[128, 128]") != std::string::npos &&
              ff.errors[0].find("[256, 64]") != std::string::npos);
    }
    {   // the wrong type: a dense tensor, an expert tensor (the MXFP4 explanation), an Engram table
        Findings ff = errors_of(changed("blk.0.attn_norm.weight", [](TensorLoc& t) { t.type = GgmlType::BF16; }));
        CHECK(ff.errors.size() == 1 && ff.errors[0].find("blk.0.attn_norm.weight") != std::string::npos && ff.errors[0].find("BF16") != std::string::npos && ff.errors[0].find("F32") != std::string::npos);
        ff = errors_of(changed("blk.3.ffn_up_exps.weight", [](TensorLoc& t) { t.type = (GgmlType) 12; }));
        bool said = false;
        for (const std::string& e : ff.errors) said = said || (e.find("Q4_K") != std::string::npos && e.find("not MXFP4") != std::string::npos);
        CHECK(said);
    }
    {   // a missing tensor, per role: a FULL layer's compressor gate, the Engram table, token_embd
        for (const char* name : {"blk.2.attn_compressor_gate.weight", "blk.1.engram_embed.weight", "token_embd.weight", "blk.6.indexer.proj.weight", "output.weight"}) {
            const Findings ff = errors_of(without(name));
            CHECK(ff.errors.size() == 1 && ff.errors[0].find(name) != std::string::npos && ff.errors[0].find("missing") != std::string::npos);
        }
    }
    {   // a tensor that the layer's ROLE does not have is unexpected (L3 is a REUSE layer: no compressor), and so is an unknown name
        TensorDir d = dir;
        TensorLoc extra = dir.at("blk.2.attn_compressor_kv.weight");
        extra.name = "blk.3.attn_compressor_kv.weight";
        d.add(extra);
        Findings ff = errors_of(d);
        CHECK(ff.errors.size() == 1 && ff.errors[0].find("blk.3.attn_compressor_kv.weight") != std::string::npos && ff.errors[0].find("not part of the deepseek41 contract") != std::string::npos);
        ff = errors_of(d, true);
        CHECK(ff.errors.empty() && ff.warnings.size() == 1);
        TensorDir v = dir;
        TensorLoc ignored_t = dir.at("blk.0.attn_norm.weight");
        ignored_t.name = "blk.0.exp_probs_b_vl.bias";
        v.add(ignored_t);
        size_t n_ign = 0;
        Findings fv;
        validate_tensors(c, v, fv, {}, &n_ign);
        CHECK(fv.ok() && n_ign == 1);
    }
    // the contract in bytes: per group
    const ByteTally t = tally(specs, c);
    CHECK(t.n_tensors == 218 && t.expert_bytes_each == Derived<MiniGeom>::kExpertBlobBytes);
    CHECK(t.group[(int) Group::Expert] == 8ull * 16 * Derived<MiniGeom>::kExpertBlobBytes);
    CHECK(t.group[(int) Group::Embd] == 512ull * 256 * 2);
    CHECK(t.group[(int) Group::Head] == 512ull * 256 * 2 + 256 * 4);
    CHECK(t.dense_device == t.total - t.group[(int) Group::Embd] - t.group[(int) Group::Expert] - t.group[(int) Group::EngramEmbed]);
    uint64_t file_bytes = 0;
    for (const TensorLoc& l : g.tensors()) file_bytes += l.nbytes;
    CHECK(t.total == file_bytes);
    CHECK_EQ_STR(group_name(Group::EngramEmbed), "engram_embed");
}

// ---------------------------------------------------------------------------------------------- 5. host dequantisation
void test_dequant(const Fx& fx, const GgufSet& g) {
    const TensorDir dir = g.directory();
    size_t n = 0, n_rows = 0;
    std::set<std::string> types;
    for (const std::string& line : read_lines(fx.gold / "dequant.txt")) {
        const auto w = words(line);      // name file n_floats [row,row,...]
        const TensorLoc& t = dir.at(w[0]);
        const std::vector<uint8_t> raw = read_bin(fx.gold / w[1]);
        const size_t nf = std::stoull(w[2]);
        CHECK(raw.size() == nf * 4);
        std::vector<float> got;
        if (w.size() > 3) {      // rows of a big table: each row decoded alone
            std::vector<int64_t> rows;
            std::string s = w[3];
            for (size_t p = 0; p < s.size();) {
                const size_t c = s.find(',', p);
                rows.push_back(std::stoll(s.substr(p, c == std::string::npos ? std::string::npos : c - p)));
                p = c == std::string::npos ? s.size() : c + 1;
            }
            const uint64_t rb = t.row_bytes();
            for (int64_t r : rows) {
                std::vector<float> row((size_t) t.ne[0]);
                dequantize(t.type, g.data(t) + (uint64_t) r * rb, t.ne[0], row.data());
                got.insert(got.end(), row.begin(), row.end());
            }
            ++n_rows;
        } else {
            got.resize((size_t) (t.ne[0] * t.nrows()));
            dequantize(t.type, g.data(t), (uint64_t) got.size(), got.data());
        }
        CHECK(got.size() == nf);
        if (got.size() == nf && std::memcmp(got.data(), raw.data(), raw.size()) != 0) {
            CHECK(!"host dequantisation differs from the oracle's");
            std::fprintf(stderr, "  tensor %s (%s)\n", w[0].c_str(), type_name(t.type).c_str());
        }
        types.insert(type_name(t.type));
        ++n;
    }
    CHECK(n > 100 && n_rows == 3);       // token_embd + the two Engram tables
    CHECK(types.count("F32") && types.count("BF16") && types.count("Q8_0") && types.count("MXFP4"));
    // the decoders' special values, independent of any file
    CHECK(f16_to_f32(0x3C00) == 1.0f && f16_to_f32(0xC000) == -2.0f && f16_to_f32(0x0001) == std::ldexp(1.0f, -24) && std::isinf(f16_to_f32(0x7C00)) && std::isnan(f16_to_f32(0x7E00)));
    CHECK(bf16_to_f32(0x3F80) == 1.0f && bf16_to_f32(0xC000) == -2.0f);
    CHECK(e8m0_half(128) == 1.0f && e8m0_half(127) == 0.5f && e8m0_half(129) == 2.0f && e8m0_half(0) == std::ldexp(1.0f, -128) && e8m0_half(1) == std::ldexp(1.0f, -127) &&
          e8m0_half(255) == std::ldexp(1.0f, 127));
    uint8_t blk[17] = {128};      // scale 1.0; code 0x8F: low nibble 15 (-12), high nibble 8 (-0)
    blk[1] = 0x8F;
    blk[2] = 0x21;                // low 1 (1), high 2 (2)
    float out[32];
    dequantize(GgmlType::MXFP4, blk, 32, out);
    CHECK(out[0] == -12.0f && out[16] == 0.0f && out[1] == 1.0f && out[17] == 2.0f);
    expect_refusal("dequantise a partial block", [&] { dequantize(GgmlType::Q8_0, blk, 20, out); }, {"whole blocks"});
    expect_refusal("dequantise an unknown type", [&] { dequantize((GgmlType) 12, blk, 32, out); }, {"no host decoder"});
}

// ---------------------------------------------------------------------------------------------- 6. expert layouts
void test_expert_layouts(const Fx& fx, const GgufSet& g) {
    const TensorDir dir = g.directory();
    const ExpertDims d = expert_dims<MiniGeom>();
    CHECK(d.valid() && d.blob_bytes() == 104448 && d.half_bytes() == 52224 && 2 * d.half_bytes() == d.blob_bytes() && d.half_ff() == 128 && d.half_down_blocks() == 4);
    const ExpertDims real = expert_dims<RealGeom>();
    CHECK(real.blob_bytes() == kBlobBytes && real.half_bytes() == kHalfBytes && real.n_halves() == 40ull * 384);
    CHECK(!ExpertDims{1, 1, 100, 256}.valid() && !ExpertDims{1, 1, 128, 96}.valid() && !ExpertDims{0, 1, 128, 256}.valid());
    for (const std::string& line : read_lines(fx.gold / "experts.txt")) {
        const auto w = words(line);      // layer expert blob half0 half1
        const int l = std::stoi(w[0]), e = std::stoi(w[1]);
        const std::vector<uint8_t> blob = read_bin(fx.gold / w[2]), h0 = read_bin(fx.gold / w[3]), h1 = read_bin(fx.gold / w[4]);
        CHECK(blob.size() == d.blob_bytes() && h0.size() == d.half_bytes() && h1.size() == d.half_bytes());
        const std::string p = "blk." + std::to_string(l) + ".";
        const TensorLoc &tg = dir.at(p + "ffn_gate_exps.weight"), &tu = dir.at(p + "ffn_up_exps.weight"), &td = dir.at(p + "ffn_down_exps.weight");
        const uint8_t *gate = g.data(tg) + (uint64_t) e * d.gate_bytes(), *up = g.data(tu) + (uint64_t) e * d.gate_bytes(), *down = g.data(td) + (uint64_t) e * d.down_bytes();
        // the blob is the three slices back to back
        CHECK(std::memcmp(blob.data(), gate, d.gate_bytes()) == 0 && std::memcmp(blob.data() + d.gate_bytes(), up, d.gate_bytes()) == 0 &&
              std::memcmp(blob.data() + 2 * d.gate_bytes(), down, d.down_bytes()) == 0);
        std::vector<uint8_t> a(d.half_bytes()), b(d.half_bytes());
        pack_half(d, gate, up, down, 0, a.data());
        pack_half(d, gate, up, down, 1, b.data());
        CHECK(a == h0 && b == h1);
        std::vector<uint8_t> back(d.blob_bytes(), 0xEE);
        unpack_halves(d, h0.data(), h1.data(), back.data());
        CHECK(back == blob);
    }
    expect_refusal("pack_half of half 2", [&] { std::vector<uint8_t> o(d.half_bytes()); pack_half(d, nullptr, nullptr, nullptr, 2, o.data()); }, {"half 2"});
    // the packer is a bijection at a shape that is not the mini's either (hidden 128, ff 256, 3 experts): random bytes
    const ExpertDims odd{1, 3, 128, 256};
    std::vector<uint8_t> blob(odd.blob_bytes());
    for (size_t i = 0; i < blob.size(); ++i) blob[i] = (uint8_t) ((i * 2654435761u) >> 13);
    std::vector<uint8_t> h0(odd.half_bytes()), h1(odd.half_bytes()), back(odd.blob_bytes());
    pack_half(odd, blob.data(), blob.data() + odd.gate_bytes(), blob.data() + 2 * odd.gate_bytes(), 0, h0.data());
    pack_half(odd, blob.data(), blob.data() + odd.gate_bytes(), blob.data() + 2 * odd.gate_bytes(), 1, h1.data());
    unpack_halves(odd, h0.data(), h1.data(), back.data());
    CHECK(back == blob);
}

// ---------------------------------------------------------------------------------------------- 7. expert lists
void test_expert_lists() {
    // static fill: an equal share per layer, the first n % layers get one more, by index
    for (int n_slots : {0, 1, 7, 8, 20, 64, 128, 500}) {
        const auto v = static_fill_by_index(8, 16, n_slots);
        const int total = std::min(n_slots, 128);
        CHECK((int) v.size() == total);
        std::map<int, int> per;
        std::set<std::pair<int, int>> uniq;
        bool by_index = true;
        for (const ExpertId& e : v) {
            by_index = by_index && e.expert == per[e.layer];
            ++per[e.layer];
            uniq.insert({e.layer, e.expert});
        }
        CHECK(by_index && uniq.size() == v.size());
        for (int l = 0; l < 8; ++l) CHECK(per[l] == total / 8 + (l < total % 8 ? 1 : 0));
    }
    const auto real = static_fill_by_index(40, 384, 1150);      // the real plan's ~1,150 slots: 28 per layer + 30 layers with one more
    CHECK(real.size() == 1150 && real[0].layer == 0 && real[0].expert == 0);
    CHECK(static_fill_by_index(0, 16, 5).empty() && static_fill_by_index(8, 0, 5).empty());

    // "L:E,L:E0-E1,L:*": duplicates dropped, order kept
    const auto v = parse_expert_list("3:5, 3:7-9,4:*,3:8", 8, 16);
    CHECK(v.size() == 1 + 3 + 16);
    CHECK(v[0] == ExpertId{3, 5} && v[1] == ExpertId{3, 7} && v[3] == ExpertId{3, 9} && v[4] == ExpertId{4, 0} && v.back() == ExpertId{4, 15});
    CHECK(parse_expert_list("", 8, 16).empty() && parse_expert_list("  ", 8, 16).empty());
    CHECK(parse_expert_list("0-1:0-1", 8, 16).size() == 4);
    CHECK(parse_expert_list("*:0", 8, 16).size() == 8);
    expect_refusal("an item without a colon", [] { (void) parse_expert_list("3", 8, 16); }, {"`3`", "LAYER:EXPERT"});
    expect_refusal("a non-number", [] { (void) parse_expert_list("3:x", 8, 16); }, {"`3:x`", "expert", "not a number"});
    expect_refusal("a layer out of range", [] { (void) parse_expert_list("3:1,8:0", 8, 16); }, {"`8:0`", "layer", "outside 0..7"});
    expect_refusal("an expert out of range", [] { (void) parse_expert_list("2:16", 8, 16); }, {"`2:16`", "expert", "outside 0..15"});
    expect_refusal("a backwards range", [] { (void) parse_expert_list("2:9-3", 8, 16); }, {"backwards"});
    expect_refusal("a stray comma", [] { (void) parse_expert_list("2:1,,3:1", 8, 16); }, {"empty item"});
    expect_refusal("a negative number", [] { (void) parse_expert_list("2:-1", 8, 16); }, {"`2:-1`"});
}

// ---------------------------------------------------------------------------------------------- 8. the GPU cache, on its own
void test_cache() {
    cuda::HostDev dev;
    const ExpertDims d{3, 4, 128, 256};          // 3 layers x 4 experts; a blob is 2 * 256 * 68 + 128 * 136 B
    const uint64_t blob = d.blob_bytes();
    {
        GpuExpertCache c(dev, d, 5);
        CHECK(c.n_slots() == 5 && c.n_resident() == 0 && c.dims().blob_bytes() == blob);
        CHECK(c.device_bytes() == 5 * blob + 3 * 4 * 4 && c.residency_bytes() == 48);
        // the table starts at -1 everywhere (0xFF bytes), never 0
        std::vector<int32_t> t(12, 0);
        dev.d2h(t.data(), c.residency(), 48);
        for (int32_t v : t) CHECK(v == -1);
        CHECK(c.host_residency().size() == 12 && c.slot_of(2, 3) == -1 && c.owner(0).layer == -1);
        // slot pointers are n * blob apart
        CHECK(c.slot_ptr(3) == c.slots() + 3 * blob);

        // a synthetic expert whose three slices are recognisable
        std::vector<uint8_t> g(4 * d.gate_bytes()), u(4 * d.gate_bytes()), dn(4 * d.down_bytes());
        for (size_t i = 0; i < g.size(); ++i) g[i] = (uint8_t) (i * 7 + 1);
        for (size_t i = 0; i < u.size(); ++i) u[i] = (uint8_t) (i * 11 + 2);
        for (size_t i = 0; i < dn.size(); ++i) dn[i] = (uint8_t) (i * 13 + 3);
        ExpertSlices s;
        s.gate = g.data();
        s.up = u.data();
        s.down = dn.data();
        s.d = d;
        const auto slot_bytes = [&](int slot) {
            std::vector<uint8_t> b(blob);
            dev.d2h(b.data(), c.slot_ptr(slot), blob);
            return b;
        };
        const auto want_blob = [&](int e) {
            std::vector<uint8_t> b;
            b.insert(b.end(), s.gate_of(e), s.gate_of(e) + d.gate_bytes());
            b.insert(b.end(), s.up_of(e), s.up_of(e) + d.gate_bytes());
            b.insert(b.end(), s.down_of(e), s.down_of(e) + d.down_bytes());
            return b;
        };
        const auto table = [&]() {
            std::vector<int32_t> v(12);
            dev.d2h(v.data(), c.residency(), 48);
            return v;
        };
        c.fill_from_gguf(dev, 2, 1, 3, s);
        CHECK(slot_bytes(2) == want_blob(3) && c.n_resident() == 1 && c.slot_of(1, 3) == 2 && c.owner(2) == ExpertId{1, 3});
        CHECK(table() == c.host_residency() && table()[1 * 4 + 3] == 2);
        // the same expert into another slot: the first slot is free, the table points at the new one
        c.fill_from_gguf(dev, 4, 1, 3, s);
        CHECK(c.slot_of(1, 3) == 4 && c.owner(2).layer == -1 && c.owner(4) == ExpertId{1, 3} && c.n_resident() == 1 && table() == c.host_residency());
        // a different expert into an occupied slot: the tenant becomes non-resident
        c.fill_from_gguf(dev, 4, 0, 1, s);
        CHECK(c.slot_of(1, 3) == -1 && c.slot_of(0, 1) == 4 && c.n_resident() == 1 && slot_bytes(4) == want_blob(1) && table() == c.host_residency());
        // deferred uploads: the host mirror moves, the device does not, until upload_residency
        c.fill_from_gguf(dev, 0, 2, 0, s, true);
        c.fill_from_gguf(dev, 1, 2, 2, s, true);
        CHECK(c.n_resident() == 3 && c.slot_of(2, 0) == 0 && c.slot_of(2, 2) == 1);
        CHECK(table()[2 * 4 + 0] == -1 && table()[2 * 4 + 2] == -1 && table() != c.host_residency());
        c.upload_residency(dev);
        CHECK(table() == c.host_residency() && table()[2 * 4 + 0] == 0 && table()[2 * 4 + 2] == 1);
        c.evict(dev, 0);
        CHECK(c.slot_of(2, 0) == -1 && c.n_resident() == 2 && table() == c.host_residency());
        c.evict(dev, 0);        // idempotent
        CHECK(c.n_resident() == 2);
        // refills of the same expert in the same slot change nothing in the accounting
        c.fill_from_gguf(dev, 1, 2, 2, s);
        CHECK(c.n_resident() == 2 && c.slot_of(2, 2) == 1);
        // bounds
        expect_refusal("slot out of range", [&] { c.fill_from_gguf(dev, 5, 0, 0, s); }, {"slot 5 of 5"});
        expect_refusal("expert out of range", [&] { c.fill_from_gguf(dev, 0, 0, 4, s); }, {"expert 4 of 4"});
        expect_refusal("evict a slot out of range", [&] { c.evict(dev, -1); }, {"slot -1"});
        expect_refusal("fill from an arena that is not built", [&] { ExpertArena a; c.fill_from_arena(dev, 0, 0, 0, a); }, {"not built"});
        expect_refusal("fill without slices", [&] { c.fill_from_gguf(dev, 0, 0, 0, ExpertSlices{}); }, {"no expert slices"});
        // a cache with no slots is legal (a CPU-only run): the table still exists and is -1
        GpuExpertCache none(dev, d, 0);
        CHECK(none.n_slots() == 0 && none.slots() == nullptr && none.residency() != nullptr && none.n_resident() == 0);
        // moves keep the allocations
        GpuExpertCache moved = std::move(none);
        CHECK(moved.residency() != nullptr && none.residency() == nullptr);
    }
    CHECK(dev.live.empty() && dev.mapped.empty());           // everything given back (HostDev also checked its guard zones on each release)
    expect_refusal("negative slots", [&] { GpuExpertCache c(dev, d, -1); }, {"-1 slots"});
    expect_refusal("more slots than experts", [&] { GpuExpertCache c(dev, d, 13); }, {"13 slots", "12 experts"});
    expect_refusal("an invalid shape", [&] { GpuExpertCache c(dev, ExpertDims{1, 1, 100, 256}, 1); }, {"not valid"});
    CHECK(dev.live.empty());
}

// ---------------------------------------------------------------------------------------------- 9. NUMA placement
platform::NumaNode node(int id, std::vector<int> cpus, int package) {
    platform::NumaNode n;
    n.id = id;
    n.cpus = std::move(cpus);
    n.package = package;
    return n;
}

void test_numa_choice(const GgufSet& g) {
    platform::NumaTopology t;
    t.available = false;
    t.why = "not Linux";
    ArenaNodes a = choose_arena_nodes(t);
    CHECK(!a.bind && a.note.find("not Linux") != std::string::npos);
    t.available = true;
    t.nodes = {node(0, {0, 1, 2, 3}, 0)};
    CHECK(!choose_arena_nodes(t).bind);
    t.nodes = {node(0, {0, 1}, 0), node(1, {2, 3}, 1)};
    a = choose_arena_nodes(t);
    CHECK(a.bind && a.node[0] == 0 && a.node[1] == 1 && a.cpus[1] == std::vector<int>({2, 3}));
    t.nodes = {node(0, {0, 1}, 0), node(2, {2, 3}, 1)};              // ids need not be contiguous
    a = choose_arena_nodes(t);
    CHECK(a.bind && a.node[0] == 0 && a.node[1] == 2);
    t.nodes = {node(0, {0, 1}, 0), node(1, {}, 0)};                  // a memory-only node
    CHECK(!choose_arena_nodes(t).bind && choose_arena_nodes(t).note.find("no CPUs") != std::string::npos);
    t.nodes = {node(0, {0}, 0), node(1, {1}, 0), node(2, {2}, 1), node(3, {3}, 1)};      // sub-NUMA clustering: 2 nodes per socket
    a = choose_arena_nodes(t);
    CHECK(a.bind && a.node[0] == 0 && a.node[1] == 2 && a.note.find("ONE sub-node") != std::string::npos);
    t.nodes = {node(0, {0}, -1), node(1, {1}, -1), node(2, {2}, -1), node(3, {3}, -1)};  // package unknown
    CHECK(!choose_arena_nodes(t).bind);
    t.nodes = {node(0, {0}, 0), node(1, {1}, 1), node(2, {2}, 2)};                       // three sockets: not a two-way split
    CHECK(!choose_arena_nodes(t).bind);

    // a real arena on whatever this machine is: unbound here (one node), bound where there are two; the contents do not depend on it.  Built from the
    // mini file's expert slices, then checked against pack_half.
    const Ds41Config c = read_config(g.meta());
    const TensorDir dir = g.directory();
    const ExpertDims d = expert_dims<MiniGeom>();
    std::vector<ExpertSlices> slices;
    for (int l = 0; l < c.n_layer; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        ExpertSlices s;
        s.gate = g.data(dir.at(p + "ffn_gate_exps.weight"));
        s.up = g.data(dir.at(p + "ffn_up_exps.weight"));
        s.down = g.data(dir.at(p + "ffn_down_exps.weight"));
        s.d = d;
        slices.push_back(s);
    }
    const auto check_arena = [&](const ExpertArena& arena) {
        CHECK(arena.built() && arena.half_bytes() == d.half_bytes() && arena.bytes_per_socket() == 8ull * 16 * d.half_bytes());
        std::vector<uint8_t> want(d.half_bytes()), blob(d.blob_bytes());
        bool all = true;
        for (int l = 0; l < d.n_layer; ++l)
            for (int e = 0; e < d.n_expert; ++e) {
                for (int h = 0; h < 2; ++h) {
                    pack_half(d, slices[(size_t) l].gate_of(e), slices[(size_t) l].up_of(e), slices[(size_t) l].down_of(e), h, want.data());
                    all = all && std::memcmp(arena.half(h, l, e), want.data(), want.size()) == 0;
                }
                arena.assemble_blob(l, e, blob.data());
                all = all && std::memcmp(blob.data(), slices[(size_t) l].gate_of(e), d.gate_bytes()) == 0 &&
                      std::memcmp(blob.data() + 2 * d.gate_bytes(), slices[(size_t) l].down_of(e), d.down_bytes()) == 0;
            }
        CHECK(all);
        CHECK(arena.half(1, 2, 3) == arena.base(1) + (2ull * 16 + 3) * d.half_bytes());
    };
    std::vector<int> done;
    std::mutex mu;
    for (int threads : {1, 3}) {
        ArenaOptions ao;
        ao.threads = threads;
        ao.layer_done = [&](int l) {
            std::lock_guard<std::mutex> lk(mu);
            done.push_back(l);
        };
        done.clear();
        const ExpertArena arena = ExpertArena::build(d, slices, ao);
        check_arena(arena);
        std::sort(done.begin(), done.end());
        CHECK(done == std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}));        // each layer reported exactly once
        std::fprintf(stderr, "  arena (%d thread%s): node %d/%d bound %d/%d | %s\n", threads, threads > 1 ? "s" : "", arena.node(0), arena.node(1), (int) arena.bound(0), (int) arena.bound(1),
                     arena.note(0).c_str());
    }
    {   // a topology that names two nodes this machine may not have: the bind is reported (not bound) and the copy still works
        platform::NumaTopology fake;
        fake.available = true;
        fake.faked = true;
        fake.nodes = {node(0, {0}, 0), node(7, {1}, 1)};
        ArenaOptions ao;
        ao.threads = 2;
        ao.topology = &fake;
        std::vector<std::string> log;
        ao.log = [&](const std::string& m) { log.push_back(m); };
        const ExpertArena arena = ExpertArena::build(d, slices, ao);
        check_arena(arena);
        CHECK(arena.node(0) == 0 && arena.node(1) == 7);
        CHECK(!arena.bound(1));                                           // there is no node 7
        bool said = false;
        for (const std::string& l : log) said = said || l.find("half 0 on node 0, half 1 on node 7") != std::string::npos;
        CHECK(said);
    }
    {   // NUMA switched off, and 4 KiB pages
        ArenaOptions ao;
        ao.numa = false;
        ao.hugepages = false;
        const ExpertArena arena = ExpertArena::build(d, slices, ao);
        check_arena(arena);
        CHECK(arena.node(0) == -1 && arena.node(1) == -1 && !arena.bound(0));
    }
    expect_refusal("arena with the wrong number of layers", [&] { std::vector<ExpertSlices> s(slices.begin(), slices.begin() + 3); (void) ExpertArena::build(d, s, ArenaOptions{}); },
                   {"3 layers of expert slices", "expected 8"});
    expect_refusal("arena without slices", [&] { std::vector<ExpertSlices> s(slices); s[4].up = nullptr; (void) ExpertArena::build(d, s, ArenaOptions{}); }, {"layer 4", "no expert slices"});
}

// ---------------------------------------------------------------------------------------------- 10. the load itself, under HostDev
struct Captured {
    std::vector<std::string> lines;
    std::string all() const {
        std::string s;
        for (const std::string& l : lines) s += l + "\n";
        return s;
    }
};

std::vector<uint8_t> device_bytes(cuda::Dev& dev, const uint8_t* p, uint64_t n) {
    std::vector<uint8_t> v((size_t) n);
    dev.d2h(v.data(), p, (size_t) n);
    return v;
}

void test_load(const Fx& fx, const GgufSet& g) {
    const Ds41Config cfg = read_config(g.meta());
    const TensorDir dir = g.directory();
    const std::vector<TensorSpec> specs = expected_tensors(cfg);
    const ExpertDims d = expert_dims<MiniGeom>();
    cuda::HostDev dev;
    {
        Captured cap;
        LoadOptions opt;
        opt.n_slots = 20;
        opt.arena.threads = 2;
        opt.log = [&](const std::string& m) { cap.lines.push_back(m); };
        std::unique_ptr<Ds41Model<MiniGeom>> m = Ds41Model<MiniGeom>::load(dev, fx.shard1, opt);
        CHECK(m->config().n_layer == 8 && m->gguf().n_shards() == 3 && m->directory().size() == 218 && m->warnings().empty());
        for (const std::string& w : m->warnings()) std::fprintf(stderr, "  warning: %s\n", w.c_str());

        // ---- the weights: every dense tensor on the device equals the mapped bytes; the role decides what exists
        const Ds41Weights<MiniGeom>& w = m->weights;
        CHECK((int) w.layer.size() == 8 && w.cfg == &m->config() && w.device_bytes() > 0);
        size_t n_dev = 0, n_host = 0;
        for (const TensorSpec& s : specs) {
            const TensorLoc& loc = dir.at(s.name);
            const DevTensor* dt = nullptr;
            if (s.lt == LT::OutputNorm) dt = &w.output_norm;
            else if (s.lt == LT::Output) dt = &w.head;
            else if (s.layer >= 0) dt = w.at(s.layer).slot(s.lt);
            if (dt) {                                                   // a device tensor
                ++n_dev;
                CHECK((bool) *dt && dt->nbytes == loc.nbytes && dt->type == loc.type && dt->ne0 == (int64_t) loc.ne[0] && dt->rows() == (int64_t) loc.nrows());
                CHECK(reinterpret_cast<uintptr_t>(dt->p) % 256 == 0);
                CHECK(device_bytes(dev, dt->p, dt->nbytes) == std::vector<uint8_t>(g.data(loc), g.data(loc) + loc.nbytes));
                CHECK(dt->row_bytes() == loc.row_bytes() && dt->row((int64_t) dt->rows() - 1) == dt->p + (dt->rows() - 1) * dt->row_bytes());
            } else {
                ++n_host;
            }
        }
        CHECK(n_dev + n_host == specs.size());
        // presence by role: a field is null exactly when the layer's role has no such tensor
        for (int l = 0; l < 8; ++l) {
            const LayerInfo& li = cfg.layer(l);
            const LayerWeights& lw = w.at(l);
            CHECK(lw.layer == l);
            CHECK((bool) lw.wq_a && (bool) lw.wo_b && (bool) lw.gate && (bool) lw.gate_bias && (bool) lw.sh_gate && (bool) lw.hc_ffn_fn && (bool) lw.attn_sinks);
            CHECK((bool) lw.comp_kv == li.has_compressor && (bool) lw.comp_norm == li.has_compressor && (bool) lw.comp_gate == li.has_compressor_gate);
            CHECK((bool) lw.idx_q_b == li.has_indexer && (bool) lw.idx_proj == li.has_indexer);
            CHECK((bool) lw.idx_comp_kv == li.has_index_compressor && (bool) lw.idx_comp_norm == li.has_index_compressor);
            CHECK((bool) lw.eng_q == li.is_engram() && (bool) lw.eng_k == li.is_engram() && (bool) lw.eng_wkv == li.is_engram() && (bool) lw.eng_table == li.is_engram());
            CHECK(lw.slot(LT::ExpGate) == nullptr && lw.slot(LT::TokenEmbd) == nullptr && lw.slot(LT::OutputNorm) == nullptr);
        }
        // formats and shapes the kernels rely on (ggml dims: ne0 = the input width)
        const LayerWeights& l2 = w.at(2);
        CHECK(l2.wq_a.type == GgmlType::Q8_0 && l2.wq_a.ne0 == 256 && l2.wq_a.ne1 == 64);
        CHECK(l2.wq_b.ne0 == 64 && l2.wq_b.ne1 == 256 && l2.wkv.ne0 == 256 && l2.wkv.ne1 == 64);
        CHECK(l2.wo_a.type == GgmlType::Q8_0 && l2.wo_a.ne0 == (int64_t) Derived<MiniGeom>::kOGroupIn && l2.wo_a.ne1 == (int64_t) Derived<MiniGeom>::kOMid);
        CHECK(l2.wo_b.ne0 == (int64_t) Derived<MiniGeom>::kOMid && l2.wo_b.ne1 == 256);
        CHECK(l2.gate.type == GgmlType::BF16 && l2.gate.ne0 == 256 && l2.gate.ne1 == 16 && l2.gate_bias.type == GgmlType::F32 && l2.gate_bias.ne0 == 16);
        CHECK(l2.hc_attn_fn.type == GgmlType::F32 && l2.hc_attn_fn.ne0 == (int64_t) Derived<MiniGeom>::kHcFlat && l2.hc_attn_fn.ne1 == 24 && l2.hc_attn_scale.ne0 == 3);
        CHECK(l2.comp_gate.type == GgmlType::BF16 && l2.idx_comp_kv.ne0 == 64 && l2.idx_comp_kv.ne1 == 32 && l2.idx_q_b.ne1 == (int64_t) Derived<MiniGeom>::kIdxQ);
        CHECK(w.at(1).eng_wkv.ne0 == (int64_t) Derived<MiniGeom>::kEngramIn && w.at(1).eng_wkv.ne1 == (int64_t) Derived<MiniGeom>::kEngramOut && w.at(1).eng_q.ne1 == 4);
        CHECK(w.head.type == GgmlType::BF16 && w.head.ne0 == 256 && w.head.ne1 == 512 && w.output_norm.type == GgmlType::F32 && w.output_norm.ne0 == 256);
        CHECK(l2.wq_a.as<uint8_t>() == l2.wq_a.p && l2.gate.as<uint16_t>() == reinterpret_cast<const uint16_t*>(l2.gate.p));

        // ---- the host tensors: token_embd (row lookup), the Engram tables, the expert slices
        CHECK(w.token_embd && w.token_embd.type == GgmlType::BF16 && w.token_embd.rows() == 512 && w.token_embd.ne0 == 256 && w.token_embd.row_bytes() == 512);
        const GgufSet& mg = m->gguf();            // the model's own mapping (the test's `g` is a second one)
        const TensorDir& md = m->directory();
        CHECK(w.token_embd.p == mg.data(md.at("token_embd.weight")));       // left in the mapping, not copied
        const EngramConfig& e = cfg.engram;
        for (int s = 0; s < 2; ++s) {
            const HostTensor& et = w.at(e.layers[(size_t) s]).eng_table;
            CHECK(et.type == GgmlType::MXFP4 && et.rows() == e.num_embeddings[(size_t) s] && et.row_bytes() == Derived<MiniGeom>::kEngramRowBytes && et.ne0 == 64);
            CHECK(et.p == mg.data(md.at("blk." + std::to_string(e.layers[(size_t) s]) + ".engram_embed.weight")));
            expect_refusal("a row past the end of a table", [&] { (void) et.row(et.rows()); }, {"outside a table"});
            expect_refusal("a negative row", [&] { (void) et.row(-1); }, {"outside a table"});
            (void) et.row(et.rows() - 1);
        }
        {   // the row lookup against the oracle's decode
            float row[256];
            const auto gl = read_lines(fx.gold / "dequant.txt");
            for (const std::string& line : gl) {
                const auto wl = words(line);
                if (wl[0] != "token_embd.weight") continue;
                const std::vector<uint8_t> raw = read_bin(fx.gold / wl[1]);
                const int rows[7] = {0, 1, 2, 17, 255, 256, 511};
                for (int i = 0; i < 7; ++i) {
                    w.token_row_f32(rows[i], row);
                    CHECK(std::memcmp(row, raw.data() + (size_t) i * 256 * 4, 256 * 4) == 0);
                    CHECK(w.token_row(rows[i]) == w.token_embd.p + (size_t) rows[i] * 512);
                }
            }
            expect_refusal("token id past the vocabulary", [&] { (void) w.token_row(512); }, {"512", "512 rows"});
        }
        for (int l = 0; l < 8; ++l) {
            const ExpertSlices& s = w.at(l).experts;
            CHECK(s && s.gate == mg.data(md.at("blk." + std::to_string(l) + ".ffn_gate_exps.weight")) && s.down == mg.data(md.at("blk." + std::to_string(l) + ".ffn_down_exps.weight")));
            CHECK(s.loc_gate && s.loc_gate->name == "blk." + std::to_string(l) + ".ffn_gate_exps.weight" && s.loc_up->name == "blk." + std::to_string(l) + ".ffn_up_exps.weight" &&
                  s.loc_down->name == "blk." + std::to_string(l) + ".ffn_down_exps.weight");
            CHECK(s.d.blob_bytes() == d.blob_bytes() && s.up_of(5) == s.up + 5 * d.gate_bytes() && s.down_of(5) == s.down + 5 * d.down_bytes());
        }

        // ---- the arena: all 128 experts, both halves, against the golden halves and pack_half
        CHECK(m->arena.built() && m->arena.bytes_per_socket() == 128 * d.half_bytes());
        for (const std::string& line : read_lines(fx.gold / "experts.txt")) {
            const auto wl = words(line);
            const int l = std::stoi(wl[0]), ex = std::stoi(wl[1]);
            const std::vector<uint8_t> h0 = read_bin(fx.gold / wl[3]), h1 = read_bin(fx.gold / wl[4]), blob = read_bin(fx.gold / wl[2]);
            CHECK(std::memcmp(m->arena.half(0, l, ex), h0.data(), h0.size()) == 0 && std::memcmp(m->arena.half(1, l, ex), h1.data(), h1.size()) == 0);
            std::vector<uint8_t> back(d.blob_bytes());
            m->arena.assemble_blob(l, ex, back.data());
            CHECK(back == blob);
        }

        // ---- the cache: 20 slots filled by index (2 per layer, the first 4 layers 3), residency table, slot bytes
        const GpuExpertCache& c = m->cache;
        CHECK(c.n_slots() == 20 && c.n_resident() == 20 && c.device_bytes() == 20 * d.blob_bytes() + 8 * 16 * 4);
        std::vector<int32_t> table(128);
        dev.d2h(table.data(), c.residency(), 128 * 4);
        CHECK(table == c.host_residency());
        const auto fill = static_fill_by_index(8, 16, 20);
        std::vector<int32_t> want(128, -1);
        for (size_t s = 0; s < fill.size(); ++s) want[(size_t) fill[s].layer * 16 + (size_t) fill[s].expert] = (int32_t) s;
        CHECK(table == want);
        int n_hit = 0;
        for (int32_t r : table) n_hit += (r >= 0 && r < c.n_slots());
        CHECK(n_hit == 20);
        bool slots_ok = true;
        for (size_t s = 0; s < fill.size(); ++s) {
            std::vector<uint8_t> blob(d.blob_bytes());
            m->arena.assemble_blob(fill[s].layer, fill[s].expert, blob.data());
            slots_ok = slots_ok && device_bytes(dev, c.slot_ptr((int) s), d.blob_bytes()) == blob && c.owner((int) s) == fill[s];
        }
        CHECK(slots_ok);
        // one golden slot byte for byte: expert (0,0) is slot 0, (1,7)? only when filled: (0,0) is
        for (const std::string& line : read_lines(fx.gold / "experts.txt")) {
            const auto wl = words(line);
            const int l = std::stoi(wl[0]), ex = std::stoi(wl[1]);
            const int slot = want[(size_t) l * 16 + (size_t) ex];
            if (slot >= 0) CHECK(device_bytes(dev, c.slot_ptr(slot), d.blob_bytes()) == read_bin(fx.gold / wl[2]));
        }
        // ---- the plan was printed, with what this model holds
        const std::string text = cap.all();
        CHECK(text.find("Memory plan:") != std::string::npos && text.find("hot-expert cache budget") != std::string::npos && text.find("20 cache slots") != std::string::npos);
        CHECK(text.find("expert arena:") != std::string::npos && text.find("loaded: device") != std::string::npos && text.find("GPU expert cache: 20 experts") != std::string::npos);
        CHECK(m->plan().in.n_slots == 20 && m->plan().cache_slots_fit == 128 && m->plan().gpu_fits && m->plan().tally.n_tensors == 218);
        CHECK(m->plan().ram_experts == 8ull * 16 * d.blob_bytes());
    }
    CHECK(dev.live.empty() && dev.mapped.empty());    // the destructor gave back the weights block, the cache and its pinned staging buffer; HostDev checked every guard zone

    // ---- variants of the load
    {   // the cache allocated but not filled; no arena (a GPU-only or test run): residency all -1
        LoadOptions opt;
        opt.n_slots = 12;
        opt.fill_cache = false;
        opt.build_arena = false;
        opt.log = [](const std::string&) {};
        auto m = Ds41Model<MiniGeom>::load(dev, fx.shard1, opt);
        CHECK(!m->arena.built() && m->cache.n_slots() == 12 && m->cache.n_resident() == 0);
        std::vector<int32_t> t(128, 0);
        dev.d2h(t.data(), m->cache.residency(), 512);
        for (int32_t v : t) CHECK(v == -1);
    }
    {   // an explicit list, filled straight from the GGUF slices (no arena); n_slots = -1 asks the plan
        LoadOptions opt;
        opt.n_slots = -1;
        opt.build_arena = false;
        opt.initial_fill = parse_expert_list("3:5,3:7-9,4:*,0:0", 8, 16);
        opt.log = [](const std::string&) {};
        auto m = Ds41Model<MiniGeom>::load(dev, fx.shard1, opt);
        CHECK(m->cache.n_slots() == 128 && m->cache.n_resident() == (int) opt.initial_fill.size() && m->cache.n_resident() == 21);
        const auto& list = opt.initial_fill;
        bool ok = true;
        for (size_t s = 0; s < list.size(); ++s) {
            const ExpertSlices& sl = m->weights.at(list[s].layer).experts;
            std::vector<uint8_t> blob;
            blob.insert(blob.end(), sl.gate_of(list[s].expert), sl.gate_of(list[s].expert) + d.gate_bytes());
            blob.insert(blob.end(), sl.up_of(list[s].expert), sl.up_of(list[s].expert) + d.gate_bytes());
            blob.insert(blob.end(), sl.down_of(list[s].expert), sl.down_of(list[s].expert) + d.down_bytes());
            ok = ok && device_bytes(dev, m->cache.slot_ptr((int) s), d.blob_bytes()) == blob && m->cache.slot_of(list[s].layer, list[s].expert) == (int) s;
        }
        CHECK(ok && m->cache.slot_of(3, 6) == -1 && m->cache.slot_of(4, 15) == 3 + 1 + 15);
    }
    {   // bounded staging: with an odd 1,000-byte piece (every tensor above that goes up in many pieces) the device holds the same bytes
        LoadOptions opt;
        opt.n_slots = 0;
        opt.build_arena = false;
        opt.upload_piece_bytes = 1000;
        opt.log = [](const std::string&) {};
        auto m = Ds41Model<MiniGeom>::load(dev, fx.shard1, opt);
        bool same = true;
        size_t n = 0;
        for (const TensorSpec& s : specs) {
            const TensorLoc& loc = dir.at(s.name);
            const DevTensor* dt = s.lt == LT::OutputNorm ? &m->weights.output_norm : s.lt == LT::Output ? &m->weights.head : s.layer >= 0 ? m->weights.at(s.layer).slot(s.lt) : nullptr;
            if (!dt) continue;
            ++n;
            same = same && device_bytes(dev, dt->p, dt->nbytes) == std::vector<uint8_t>(g.data(loc), g.data(loc) + loc.nbytes);
        }
        CHECK(same && n > 150 && m->cache.n_slots() == 0 && m->cache.n_resident() == 0);
        expect_refusal("an upload piece of 0 bytes", [&] { LoadOptions o = opt; o.upload_piece_bytes = 0; (void) Ds41Model<MiniGeom>::load(dev, fx.shard1, o); }, {"piece of 0 bytes"});
    }
    {   // refusals of the options
        LoadOptions opt;
        opt.n_slots = 2;
        opt.log = [](const std::string&) {};
        opt.initial_fill = parse_expert_list("0:0-4", 8, 16);
        expect_refusal("more initial experts than slots", [&] { (void) Ds41Model<MiniGeom>::load(dev, fx.shard1, opt); }, {"5 experts", "2 slots"});
        opt.initial_fill = {ExpertId{9, 0}};
        expect_refusal("an initial expert outside the model", [&] { (void) Ds41Model<MiniGeom>::load(dev, fx.shard1, opt); }, {"9:0", "outside the model"});
    }
    CHECK(dev.live.empty());
}

// ---------------------------------------------------------------------------------------------- 11. files that are not this model
void patch_after_name(std::vector<uint8_t>& file, const std::string& name, const std::function<void(uint8_t* after_name)>& fn) {
    // tensor table entry: u64 len, name, u32 n_dims, u64 ne[n_dims], u32 type, u64 offset
    const uint64_t len = name.size();
    const std::string key = std::string(reinterpret_cast<const char*>(&len), 8) + name;
    const auto it = std::search(file.begin(), file.end(), key.begin(), key.end());
    if (it == file.end()) {
        std::fprintf(stderr, "test bug: `%s` is not in the header\n", name.c_str());
        std::exit(2);
    }
    fn(&*it + key.size());
}

void test_bad_files(const Fx& fx, const GgufSet& g) {
    cuda::HostDev dev;
    LoadOptions opt;
    opt.log = [](const std::string&) {};
    opt.n_slots = 4;
    opt.arena.threads = 1;
    const auto shard = [&](int i) { return g.shard((size_t) i).path; };
    const fs::path neg = fx.dir / "neg";
    fs::remove_all(neg);
    fs::create_directories(neg);
    const auto copy_all = [&](const std::string& sub) {
        const fs::path d = neg / sub;
        fs::create_directories(d);
        for (size_t i = 0; i < g.n_shards(); ++i) fs::copy_file(shard((int) i), d / fs::path(shard((int) i)).filename());
        return d;
    };
    const auto first_of = [&](const fs::path& d) { return (d / fs::path(shard(0)).filename()).string(); };

    // an intact copy loads (so the refusals below are the patches' doing)
    {
        const fs::path d = copy_all("intact");
        auto m = Ds41Model<MiniGeom>::load(dev, first_of(d), opt);
        CHECK(m->cache.n_slots() == 4);
    }
    // the file is the mini model, the build is RealGeom
    expect_refusal("RealGeom build, mini file", [&] { (void) Ds41Model<RealGeom>::load(dev, fx.shard1, opt); },
                   {"does not match the compiled geometry 'real'", "`deepseek41.embedding_length` = 256", "expects 5120"});
    {   // dims that disagree with the metadata, same byte size so the file is otherwise intact: [256, 64] -> [128, 128]
        const fs::path d = copy_all("dims");
        for (size_t i = 0; i < g.n_shards(); ++i) {
            const fs::path p = d / fs::path(shard((int) i)).filename();
            std::vector<uint8_t> f = read_bin(p);
            const std::string nm = "blk.0.attn_q_a.weight";
            if (std::search(f.begin(), f.end(), nm.begin(), nm.end()) == f.end()) continue;
            patch_after_name(f, nm, [](uint8_t* a) {
                uint32_t nd;
                std::memcpy(&nd, a, 4);
                const uint64_t ne0 = 128, ne1 = 128;
                std::memcpy(a + 4, &ne0, 8);
                std::memcpy(a + 12, &ne1, 8);
                (void) nd;
            });
            write_bin(p, f);
        }
        expect_refusal("changed dims", [&] { (void) Ds41Model<MiniGeom>::load(dev, first_of(d), opt); },
                       {"blk.0.attn_q_a.weight", "dims [128, 128]", "implies [256, 64]", "Q8_0"});
    }
    {   // a norm stored as F16 instead of F32: the table is still consistent (the tensor just got smaller), so the refusal is the contract's, by name
        const fs::path d = copy_all("type");
        for (size_t i = 0; i < g.n_shards(); ++i) {
            const fs::path p = d / fs::path(shard((int) i)).filename();
            std::vector<uint8_t> f = read_bin(p);
            const std::string nm = "blk.1.ffn_norm.weight";
            if (std::search(f.begin(), f.end(), nm.begin(), nm.end()) == f.end()) continue;
            patch_after_name(f, nm, [](uint8_t* a) {
                const uint32_t ty = 1;      // F16
                std::memcpy(a + 4 + 8, &ty, 4);
            });
            write_bin(p, f);
        }
        expect_refusal("changed type", [&] { (void) Ds41Model<MiniGeom>::load(dev, first_of(d), opt); }, {"blk.1.ffn_norm.weight", "F16", "F32"});
    }
    {   // a shard that is not there
        const fs::path d = copy_all("missing");
        fs::remove(d / fs::path(shard(1)).filename());
        expect_refusal("a missing shard", [&] { (void) GgufSet::open(first_of(d)); }, {"missing model shard", "00002-of-00003"});
    }
    {   // one that stopped in the middle of the data (a download that is still running)
        const fs::path d = copy_all("truncated");
        const fs::path p = d / fs::path(shard(1)).filename();
        fs::resize_file(p, fs::file_size(p) / 2);
        expect_refusal("a truncated shard", [&] { (void) GgufSet::open(first_of(d)); }, {"00002-of-00003", "truncated"});
        const fs::path q = d / fs::path(shard(2)).filename();
        fs::resize_file(q, 100);        // inside the header
        expect_refusal("a header cut short", [&] { (void) GgufSet::open(first_of(d)); }, {"00003-of-00003", "end of file"});
    }
    {   // shard 1 opened alone says it is shard 1 of 3
        expect_refusal("a split model opened as one file", [&] { GgufSet one({shard(0)}); }, {"shard 1 of 3", "first shard"});
        // shards of different splits: the second does not declare itself shard 2 of 3
        expect_refusal("shards in the wrong order", [&] { GgufSet two({shard(0), shard(2), shard(1)}); }, {"does not declare itself shard 2 of 3"});
    }
    {   // not a GGUF
        const fs::path d = copy_all("magic");
        const fs::path p = d / fs::path(shard(0)).filename();
        std::vector<uint8_t> f = read_bin(p);
        f[0] = 'X';
        write_bin(p, f);
        expect_refusal("bad magic", [&] { (void) GgufSet::open(first_of(d)); }, {"not a GGUF"});
        expect_refusal("a missing file", [&] { (void) GgufSet::open((d / "nothing-00001-of-00001.gguf").string()); }, {"nothing"});
    }
    {   // any shard names the model (the others are found from its name) ...
        const std::string second = shard(1);
        auto m = Ds41Model<MiniGeom>::load(dev, second, opt);
        CHECK(m->gguf().n_shards() == 3 && m->config().n_layer == 8);
        // ... but the metadata is shard 1's: shard 2's own key/value table is not a model configuration
        expect_refusal("a non-metadata shard's keys as the model's", [&] { (void) read_config(g.shard_meta(1)); }, {"general.architecture", "missing"});
    }
    fs::remove_all(neg);
    CHECK(dev.live.empty());
}

// ---------------------------------------------------------------------------------------------- 12. helpers for the other packages
struct FakeEngramConstants {                 // the member names of DS1-D's cuda::EngramConstants
    std::vector<int32_t> layer_ids;
    int32_t n_heads = 0, max_ngram_size = 0, pad_token_id = 0, compressed_vocab_size = 0;
    std::vector<int64_t> primes, offsets, multipliers;
    std::vector<int32_t> token_map;
    std::vector<int64_t> num_embeddings;
};
struct FakeTableView {                       // ... and of cuda::EngramTableView
    const uint8_t* base = nullptr;
    int64_t rows = 0;
};

void test_helpers(const Fx& fx, const GgufSet& g) {
    const Ds41Config c = read_config(g.meta());
    const FakeEngramConstants ec = engram_constants_as<FakeEngramConstants>(c);
    CHECK(ec.layer_ids == std::vector<int32_t>({1, 3}) && ec.n_heads == 2 && ec.max_ngram_size == 4 && ec.pad_token_id == 2 && ec.compressed_vocab_size == 300);
    CHECK(ec.primes == c.engram.primes && ec.offsets == c.engram.offsets && ec.multipliers == c.engram.multipliers && ec.token_map == c.engram.token_map && ec.num_embeddings == c.engram.num_embeddings);
    CHECK(ec.token_map.size() == 512 && ec.primes.size() == 12);

    // RoPE tables on the device: both kinds, per layer, equal to the host tables and (within a float32 rounding) to the oracle's
    cuda::HostDev dev;
    {
        const DeviceRope rope(dev, c, 64);
        CHECK(rope.n_pos() == 64 && rope.device_bytes() == 4ull * 64 * 8 * 4);
        const size_t n = 64 * 8;
        for (int kind = 0; kind < 2; ++kind) {
            const RopeTable t = build_rope_table(kind ? c.rope_csa : c.rope_swa, c.rope_dim, 64);
            const int layer = kind ? 2 : 0;
            std::vector<float> cs(n), sn(n);
            dev.d2h(cs.data(), rope.cos(layer), n * 4);
            dev.d2h(sn.data(), rope.sin(layer), n * 4);
            CHECK(cs == t.cos && sn == t.sin);
            const std::vector<uint8_t> raw = read_bin(fx.gold / (kind ? "rope_csa.f32" : "rope_swa.f32"));
            double worst = 0;
            for (size_t i = 0; i < n; ++i) {
                float gc, gs;
                std::memcpy(&gc, raw.data() + 4 * i, 4);
                std::memcpy(&gs, raw.data() + 4 * (n + i), 4);
                worst = std::max({worst, (double) std::fabs(gc - cs[i]), (double) std::fabs(gs - sn[i])});
            }
            CHECK(worst <= 1.3e-7);
        }
        CHECK(rope.cos(0) == rope.cos(1) && rope.cos(2) == rope.cos(7) && rope.cos(0) != rope.cos(2) && rope.sin(0) != rope.sin(4));
        DeviceRope moved = std::move(const_cast<DeviceRope&>(rope));
        CHECK(moved.cos(2) != nullptr && rope.n_pos() == 0);
    }
    CHECK(dev.live.empty());
    expect_refusal("RoPE tables of 0 positions", [&] { DeviceRope r(dev, c, 0); }, {"0 positions"});

    // page-cache advice on a mapping must not change what is read
    const TensorDir dir = g.directory();
    const TensorLoc& t = dir.at("blk.0.ffn_gate_exps.weight");
    const std::vector<uint8_t> before(g.data(t), g.data(t) + t.nbytes);
    (void) g.advise_random(t);
    g.prefetch(t);
    g.drop_cache(t);
    g.drop_cache(t, 100, 5000);
    g.drop_cache(t, t.nbytes + 10, 5);        // past the end: nothing
    CHECK(std::vector<uint8_t>(g.data(t), g.data(t) + t.nbytes) == before);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ds41_model_test <fixture dir>\n");
        return 2;
    }
    Fx fx;
    fx.dir = argv[1];
    fx.gold = fx.dir / "golden";
    for (const auto& e : fs::directory_iterator(fx.dir / "mini"))
        if (e.path().filename().string().find("-00001-of-") != std::string::npos) fx.shard1 = e.path().string();
    if (fx.shard1.empty()) {
        std::fprintf(stderr, "no shard 1 under %s/mini\n", argv[1]);
        return 2;
    }
    std::unique_ptr<GgufSet> g;
    try {
        g = GgufSet::open(fx.shard1);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: opening the mini model: %s\n", e.what());
        return 1;
    }
    const struct {
        const char* name;
        std::function<void()> fn;
    } tests[] = {
        {"shards and metadata", [&] { test_shards_and_meta(fx, *g); }},
        {"tensor table", [&] { test_tensor_table(fx, *g); }},
        {"config and roles", [&] { test_config(fx, *g); }},
        {"config refusals", [&] { test_config_refusals(*g); }},
        {"tensor validation", [&] { test_validation(*g); }},
        {"host dequantisation", [&] { test_dequant(fx, *g); }},
        {"expert layouts", [&] { test_expert_layouts(fx, *g); }},
        {"expert lists", [&] { test_expert_lists(); }},
        {"GPU cache", [&] { test_cache(); }},
        {"NUMA and arena", [&] { test_numa_choice(*g); }},
        {"load under HostDev", [&] { test_load(fx, *g); }},
        {"files that are not this model", [&] { test_bad_files(fx, *g); }},
        {"helpers (Engram constants, RoPE tables, page-cache advice)", [&] { test_helpers(fx, *g); }},
    };
    for (const auto& t : tests) {
        const int before = g_fail;
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_fail;
            std::fprintf(stderr, "FAIL %s: unexpected exception: %s\n", t.name, e.what());
        }
        std::printf("%s %s\n", g_fail == before ? "PASS" : "FAIL", t.name);
        std::fflush(stdout);
    }
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
