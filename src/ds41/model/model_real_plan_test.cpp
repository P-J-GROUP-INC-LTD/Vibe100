// src/ds41/model/model_real_plan_test.cpp - DS1-A: the REAL model's shapes, from the saved headers of mxxm-t/DeepSeek-V4.1-Flash-GGUF (no weights needed).
//
//   ds41_model_real_plan_test <fixture dir>     reads <dir>/real_headers.txt and real_expect.txt (src/ds41/model/model_fixture.py)
//
// The metadata of shard 1 and the 1,006-tensor table of the 12 shards go through the same code as a real load: read_config (every key, the layer map),
// require_geometry<RealGeom>, validate_tensors (every name, ggml type and dims), the byte tally (against totals Python computed from the same table) and the
// memory plan (RESEARCH.md section 10, PLAN.md section 2).  Prints the plan.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "strata/ds41/model/model.hpp"

using namespace strata::ds41;
using namespace strata::ds41::model;

namespace {
int g_checks = 0, g_fail = 0;
#define CHECK(...)                                                                      \
    do {                                                                                \
        ++g_checks;                                                                     \
        if (!(__VA_ARGS__)) {                                                           \
            ++g_fail;                                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #__VA_ARGS__); \
        }                                                                               \
    } while (0)

std::vector<std::string> words(const std::string& l) {
    std::istringstream is(l);
    std::vector<std::string> w;
    std::string x;
    while (is >> x) w.push_back(x);
    return w;
}

GgmlType type_of(const std::string& n) {
    for (GgmlType t : {GgmlType::F32, GgmlType::F16, GgmlType::Q8_0, GgmlType::BF16, GgmlType::MXFP4})
        if (type_name(t) == n) return t;
    std::fprintf(stderr, "unknown type %s in the headers\n", n.c_str());
    std::exit(2);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ds41_model_real_plan_test <fixture dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::ifstream hdr(dir + "/real_headers.txt");
    if (!hdr) {
        std::fprintf(stderr, "cannot open %s/real_headers.txt\n", dir.c_str());
        return 2;
    }
    MetaStore meta("real_headers.txt");
    TensorDir tdir;
    std::vector<TensorLoc> locs;
    std::vector<uint64_t> py_nbytes;
    int n_shards = 0;
    std::string line;
    while (std::getline(hdr, line)) {
        if (line.size() < 3) continue;
        if (line[0] == 'S') {
            ++n_shards;
        } else if (line[0] == 'K') {
            const size_t a = line.find(' ', 2), b = line.find(' ', a + 1);
            const std::string key = line.substr(2, a - 2), kind = line.substr(a + 1, b - a - 1), rest = b == std::string::npos ? "" : line.substr(b + 1);
            const auto w = words(rest);
            MetaValue v;
            if (kind == "i") { v.type = MetaType::I64; v.i = std::stoll(rest); }
            else if (kind == "f") { v.type = MetaType::F64; v.f = std::strtod(rest.c_str(), nullptr); }
            else if (kind == "b") { v.type = MetaType::Bool; v.i = rest == "1"; }
            else if (kind == "s") { v.type = MetaType::String; v.s = rest; }
            else if (kind == "ai" || kind == "af") {
                v.type = MetaType::Array;
                v.count = std::stoull(w[0]);
                if (kind == "ai") { v.elem = MetaType::I64; for (size_t i = 1; i < w.size(); ++i) v.ia.push_back(std::stoll(w[i])); }
                else { v.elem = MetaType::F64; for (size_t i = 1; i < w.size(); ++i) v.fa.push_back(std::strtod(w[i].c_str(), nullptr)); }
            } else if (kind == "as" || kind == "ac") {                     // a string array / a summarised array: its length only
                v.type = MetaType::Array;
                v.elem = kind == "as" ? MetaType::String : MetaType::I32;
                v.count = std::stoull(w[0]);
            } else {
                std::fprintf(stderr, "unknown kind %s\n", kind.c_str());
                return 2;
            }
            meta.set(key, v);
        } else if (line[0] == 'T') {
            const auto w = words(line);      // T shard name type nd dims... offset nbytes
            TensorLoc t;
            t.shard = std::stoi(w[1]);
            t.name = w[2];
            t.type = type_of(w[3]);
            t.n_dims = std::stoi(w[4]);
            for (int d = 0; d < t.n_dims; ++d) t.ne[d] = std::stoull(w[5 + (size_t) d]);
            t.offset = std::stoull(w[5 + (size_t) t.n_dims]);
            t.nbytes = tensor_nbytes(t.type, t.ne, t.n_dims);
            py_nbytes.push_back(std::stoull(w[6 + (size_t) t.n_dims]));
            locs.push_back(t);
            tdir.add(t);
        }
    }
    CHECK(n_shards == 12 && locs.size() == 1006 && tdir.size() == 1006);
    for (size_t i = 0; i < locs.size(); ++i) CHECK(locs[i].nbytes == py_nbytes[i] && locs[i].nbytes > 0);

    // ---- the configuration
    const Ds41Config c = read_config(meta, /*with_big_arrays=*/false);
    require_geometry<RealGeom>(c);
    CHECK(c.n_layer == 40 && c.hidden == 5120 && c.vocab == 129280 && c.n_expert == 384 && c.n_used == 6 && c.ff == 2304 && c.n_head == 64 && c.head_dim == 512 && c.rope_dim == 64);
    CHECK(c.q_lora == 1280 && c.o_groups == 8 && c.o_lora == 1024 && c.window == 128 && c.idx_heads == 32 && c.idx_dim == 128 && c.idx_topk == 512 && c.hc == 4 && c.hc_iters == 20);
    CHECK(c.cand_source == 20 && c.cand_block == 8 && c.cand_topk_blocks == 2048 && c.context_length == 1048576);
    CHECK(c.rope_swa.base == 10000.0 && c.rope_csa.base == 160000.0 && c.rope_csa.factor == 16.0 && c.rope_csa.orig_ctx == 65536 && c.rope_csa.beta_fast == 32.0 && c.rope_csa.beta_slow == 1.0);
    CHECK(c.n_ratio_entries == 43 && c.compress_ratios.size() == 40);
    CHECK(c.engram.layers == std::vector<int>({1, 14}) && c.engram.heads == 8 && c.engram.head_dim == 256 && c.engram.ngram == 4 && c.engram.cols() == 24 && c.engram.pad_id == 2 &&
          c.engram.cvocab == 99092 && c.engram.num_embeddings == std::vector<int64_t>({384006168, 384016682}));
    CHECK(c.engram.primes.size() == 48 && c.engram.offsets.size() == 48 && c.engram.multipliers.size() == 8 && c.engram.token_map.empty());
    CHECK(c.engram.offset(0, 1) == c.engram.prime(0, 0) && c.engram.offset(1, 0) == 0);
    int64_t sum = 0;
    for (int col = 0; col < 24; ++col) sum += c.engram.prime(1, col);
    CHECK(sum == 384016682);
    std::printf("%s", c.describe().c_str());

    // ---- the layer roles of RESEARCH.md section 2
    for (int l = 0; l < 40; ++l) {
        const LayerInfo& li = c.layer(l);
        const Role want = l < 2 ? Role::SWA : (l == 2 || l == 8 || l == 14 || l == 20) ? Role::FULL : (l == 24 || l == 28 || l == 32 || l == 36) ? Role::REINDEX : Role::REUSE;
        CHECK(li.role == want && li.ratio == (l < 2 ? 0 : l < 20 ? 2 : 1));
        CHECK(li.is_engram() == (l == 1 || l == 14));
        CHECK(li.is_candidate_source == (l == 20) && li.uses_candidate_pool == (l == 24 || l == 28 || l == 32 || l == 36));
    }
    CHECK(c.layer(5).kv_owner == 2 && c.layer(5).topk_source == 2 && c.layer(9).kv_owner == 8 && c.layer(19).kv_owner == 14 && c.layer(21).kv_owner == 20 && c.layer(23).topk_source == 20);
    CHECK(c.layer(24).kv_owner == 20 && c.layer(24).index_k_owner == 20 && c.layer(24).topk_source == 24 && c.layer(25).topk_source == 24 && c.layer(39).topk_source == 36 && c.layer(39).kv_owner == 20);
    CHECK(c.layer(2).has_compressor_gate && !c.layer(20).has_compressor_gate && c.layer(20).has_compressor && c.layer(20).has_index_compressor && !c.layer(24).has_index_compressor);
    int n_role[4] = {0, 0, 0, 0};
    for (const LayerInfo& li : c.layers) ++n_role[(int) li.role];
    CHECK(n_role[(int) Role::SWA] == 2 && n_role[(int) Role::FULL] == 4 && n_role[(int) Role::REINDEX] == 4 && n_role[(int) Role::REUSE] == 30);

    // ---- every tensor of the 12 shards against the contract
    Findings f;
    size_t ignored = 0;
    validate_tensors(c, tdir, f, {}, &ignored);
    for (const std::string& e : f.errors) std::fprintf(stderr, "  %s\n", e.c_str());
    CHECK(f.ok() && f.warnings.empty());
    const std::vector<TensorSpec> specs = expected_tensors(c);
    CHECK(specs.size() == 1006 && ignored == 0);
    CHECK(specs.front().name == "token_embd.weight" && specs.back().name == "output.weight");
    // the table of the file in layer order: the expert tensors of layer 0 and the Engram table of layer 1, by name, with their sizes
    CHECK(tdir.at("blk.0.ffn_gate_exps.weight").nbytes == 384ull * kGateBytes && tdir.at("blk.0.ffn_down_exps.weight").nbytes == 384ull * kDownBytes);
    CHECK(tdir.at("blk.1.engram_embed.weight").nbytes == 384006168ull * 136 && tdir.at("blk.14.engram_embed.weight").nbytes == 384016682ull * 136);
    CHECK(tdir.at("blk.0.attn_output_a.weight").ne[0] == 4096 && tdir.at("blk.0.attn_output_a.weight").ne[1] == 8192);
    CHECK(tdir.at("blk.1.engram_wkv.weight").ne[0] == 6144 && tdir.at("blk.1.engram_wkv.weight").ne[1] == 25600);
    CHECK(tdir.at("blk.14.hc_ffn_fn.weight").ne[0] == 20480 && tdir.at("blk.14.hc_ffn_fn.weight").ne[1] == 24);
    // every shard's layout: aligned (32), no overlap, in file order
    {
        std::map<int, std::vector<const TensorLoc*>> by_shard;
        for (const TensorLoc& t : locs) by_shard[t.shard].push_back(&t);
        bool ok = true;
        for (auto& kv : by_shard) {
            uint64_t end = 0;
            for (const TensorLoc* t : kv.second) {
                ok = ok && t->offset % 32 == 0 && t->offset >= end;
                end = t->offset + t->nbytes;
            }
        }
        CHECK(ok);
    }

    // ---- the byte tally against Python's totals from the same table
    std::map<std::string, uint64_t> expect;
    {
        std::ifstream ex(dir + "/real_expect.txt");
        std::string k;
        uint64_t v;
        while (ex >> k >> v) expect[k] = v;
    }
    const ByteTally t = tally(specs, c);
    CHECK(t.n_tensors == 1006 && (int64_t) t.n_tensors == (int64_t) expect["n_tensors"]);
    CHECK(t.group[(int) Group::Expert] == expect["expert"] && t.group[(int) Group::EngramEmbed] == expect["engram_embed"] && t.group[(int) Group::Embd] == expect["embd"]);
    CHECK(t.dense_device == expect["dense"] && t.total == expect["dense"] + expect["expert"] + expect["engram_embed"] + expect["embd"]);
    CHECK(t.expert_bytes_each == kBlobBytes && t.group[(int) Group::Expert] == (uint64_t) 40 * 384 * kBlobBytes);
    CHECK(t.group[(int) Group::EngramEmbed] == 104451107600ull && t.group[(int) Group::Embd] == 129280ull * 5120 * 2);

    // ---- the memory plan on the target box (32 GiB V100, 384 GiB RAM, two sockets)
    PlanInputs in;
    const MemoryPlan first = make_memory_plan(c, t, in);
    in.n_slots = first.cache_slots_fit;
    const MemoryPlan p = make_memory_plan(c, t, in);
    std::printf("%s\n", p.text().c_str());
    const double GiB = 1073741824.0;
    CHECK(p.gpu_dense == expect["dense"] && (double) p.gpu_dense / GiB > 8.3 && (double) p.gpu_dense / GiB < 8.4);
    CHECK(p.cache_slots_fit > 1100 && p.cache_slots_fit < 1200 && p.gpu_fits && p.in.n_slots == p.cache_slots_fit);
    CHECK(p.gpu_cache == (uint64_t) p.cache_slots_fit * kBlobBytes && p.gpu_residency == 40ull * 384 * 4);
    CHECK((double) p.ram_experts / GiB > 268.9 && (double) p.ram_experts / GiB < 269.0 && p.ram_experts_per_node == p.ram_experts / 2);
    CHECK((double) p.ram_engram / GiB > 97.2 && (double) p.ram_engram / GiB < 97.3 && (double) p.ram_token_embd / GiB > 1.2 && (double) p.ram_token_embd / GiB < 1.25);
    CHECK(p.ram_page_cache > 0 && p.engram_cached_fraction > 0.9 && p.warnings.empty());
    // per socket the arena holds half of every expert
    const ExpertDims d = expert_dims<RealGeom>();
    CHECK(d.n_halves() * d.half_bytes() == p.ram_experts / 2 && d.n_halves() * d.half_bytes() == 40ull * 384 * kHalfBytes);
    std::printf("arena per socket: %.2f GiB (%llu halves of %llu B); cache: %d slots = %.2f GiB; dense on the device: %.2f GiB\n", (double) d.n_halves() * d.half_bytes() / GiB,
                (unsigned long long) d.n_halves(), (unsigned long long) d.half_bytes(), p.in.n_slots, (double) p.gpu_cache / GiB, (double) p.gpu_dense / GiB);
    // a box that cannot hold the experts, and a GPU that cannot hold the dense weights, are said so
    PlanInputs small = in;
    small.ram_total = 256ull << 30;
    const MemoryPlan ps = make_memory_plan(c, t, small);
    bool said = false;
    for (const std::string& w : ps.warnings) said = said || w.find("do not fit in RAM") != std::string::npos;
    CHECK(said && ps.ram_page_cache < 0);
    PlanInputs tiny = in;
    tiny.vram_total = 8ull << 30;
    tiny.n_slots = 100;
    const MemoryPlan pt = make_memory_plan(c, t, tiny);
    CHECK(!pt.gpu_fits && pt.cache_slots_fit == 0 && !pt.warnings.empty());
    // the static fill of the plan's cache
    const auto fill = static_fill_by_index(40, 384, p.cache_slots_fit);
    CHECK((int) fill.size() == p.cache_slots_fit && fill.front().layer == 0 && fill.back().layer == 39);

    // ---- a geometry that is not the file's: the refusal names the key, what it found and what the build expects
    Ds41Config other = c;
    other.n_expert = 256;
    Findings fo;
    check_geometry<RealGeom>(other, fo);
    CHECK(fo.errors.size() == 1 && fo.errors[0].find("deepseek41.expert_count") != std::string::npos && fo.errors[0].find("256") != std::string::npos && fo.errors[0].find("384") != std::string::npos);
    try {
        require_geometry<MiniGeom>(c);
        CHECK(!"the real model must not pass as MiniGeom");
    } catch (const ModelError& e) {
        CHECK(std::string(e.what()).find("compiled geometry 'mini'") != std::string::npos);
    }
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
