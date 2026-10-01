// src/ds41/model/config.cpp - DS1-A: metadata -> Ds41Config, the layer roles, the RoPE tables (include/strata/ds41/model/config.hpp).
#include "strata/ds41/model/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

#include "strata/ds41/geometry.hpp"

namespace strata::ds41::model {

const char* role_name(Role r) {
    switch (r) {
    case Role::SWA: return "SWA";
    case Role::FULL: return "FULL";
    case Role::REUSE: return "REUSE";
    case Role::REINDEX: return "REINDEX";
    }
    return "?";
}

namespace {

bool is_prime(int64_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (int64_t i = 3; i * i <= n; i += 2)
        if (n % i == 0) return false;
    return true;
}

int64_t next_prime_above(int64_t n) {
    ++n;
    while (!is_prime(n)) ++n;
    return n;
}

bool close_rel(double a, double b, double rel) { return std::fabs(a - b) <= rel * std::fabs(b); }

std::string join_ints(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + std::to_string(v[i]);
    return s;
}

/// Reader that records a missing / ill-typed key as a finding and returns a zero, so that every problem of a file is reported at once.
struct Reader {
    const MetaStore& m;
    Findings& f;
    const std::string P = "deepseek41.";
    template <class T, class Fn> T guard(Fn&& fn) {
        try {
            return fn();
        } catch (const ModelError& e) {
            f.error(e.what());
            return T{};
        }
    }
    int64_t i64(const std::string& k) { return guard<int64_t>([&] { return m.get_int(k); }); }
    int i32(const std::string& k) {
        const int64_t v = i64(k);
        if (v < INT32_MIN || v > INT32_MAX) {
            f.error("metadata `" + k + "` = " + std::to_string(v) + " does not fit an int");
            return 0;
        }
        return (int) v;
    }
    double f64(const std::string& k) { return guard<double>([&] { return m.get_float(k); }); }
    bool boolean(const std::string& k) { return guard<bool>([&] { return m.get_bool(k); }); }
    std::string str(const std::string& k) { return guard<std::string>([&] { return m.get_string(k); }); }
    std::vector<int64_t> ints(const std::string& k) { return guard<std::vector<int64_t>>([&] { return m.get_ints(k); }); }
    std::vector<int> ints32(const std::string& k) {
        std::vector<int> out;
        for (int64_t v : ints(k)) {
            if (v < INT32_MIN || v > INT32_MAX) {
                f.error("metadata `" + k + "` holds " + std::to_string(v) + ", which does not fit an int");
                v = 0;
            }
            out.push_back((int) v);
        }
        return out;
    }
    std::vector<double> floats(const std::string& k) { return guard<std::vector<double>>([&] { return m.get_floats(k); }); }
    // the model's own keys
    int64_t I64(const char* k) { return i64(P + k); }
    int I(const char* k) { return i32(P + k); }
    double F(const char* k) { return f64(P + k); }
};

}  // namespace

Ds41Config read_config(const MetaStore& meta, bool with_big_arrays) {
    Findings f;
    Reader r{meta, f};
    Ds41Config c;

    // ---- identity
    c.architecture = r.str("general.architecture");
    if (f.ok() && c.architecture != "deepseek41")
        throw ModelError(meta.source() + ": general.architecture = `" + c.architecture + "`, this loader reads `deepseek41` (is shard 1 -- the one that carries the metadata -- the file given?)");
    if (const MetaValue* n = meta.find("general.name"); n && n->is_string()) c.name = n->s;

    // ---- trunk and attention
    c.n_layer = r.I("block_count");
    c.context_length = r.I64("context_length");
    c.hidden = r.I("embedding_length");
    c.n_head = r.I("attention.head_count");
    c.n_kv_head = r.I("attention.head_count_kv");
    c.head_dim = r.I("attention.key_length");
    c.value_dim = r.I("attention.value_length");
    c.rope_dim = r.I("rope.dimension_count");
    c.q_lora = r.I("attention.q_lora_rank");
    c.o_groups = r.I("attention.output_group_count");
    c.o_lora = r.I("attention.output_lora_rank");
    c.window = r.I("attention.sliding_window");
    c.rms_eps = (float) r.F("attention.layer_norm_rms_epsilon");
    // vocabulary: the tokenizer's token count (the oracle: len(tokenizer.ggml.tokens)); token_embd / output are checked against it
    c.vocab = 0;
    if (const MetaValue* t = meta.find("tokenizer.ggml.tokens"); t && t->is_array()) c.vocab = (int) t->count;
    else f.error(meta.source() + ": required metadata key `tokenizer.ggml.tokens` is missing (it gives the vocabulary size)");

    // ---- RoPE
    const std::string rope_type = r.str(r.P + "rope.scaling.type");
    if (f.ok() && rope_type != "yarn") f.error("`deepseek41.rope.scaling.type` = `" + rope_type + "`, this port implements `yarn` (ref/ds41/rope.py)");
    c.rope_swa.base = r.F("rope.freq_base");
    c.rope_swa.yarn = false;
    c.rope_swa.orig_ctx = 0;
    c.rope_csa.base = r.F("attention.compress_rope_freq_base");
    c.rope_csa.yarn = true;
    c.rope_csa.factor = r.F("rope.scaling.factor");
    c.rope_csa.orig_ctx = r.I64("rope.scaling.original_context_length");
    c.rope_csa.beta_fast = r.F("rope.scaling.yarn_beta_fast");
    c.rope_csa.beta_slow = r.F("rope.scaling.yarn_beta_slow");

    // ---- MoE
    c.n_expert = r.I("expert_count");
    c.n_used = r.I("expert_used_count");
    c.n_shared = r.I("expert_shared_count");
    c.ff = r.I("expert_feed_forward_length");
    c.gating_func = r.I("expert_gating_func");
    c.route_scale = (float) r.F("expert_weights_scale");
    c.route_norm = r.boolean(r.P + "expert_weights_norm");
    const std::vector<double> clamp_exp = r.floats(r.P + "swiglu_clamp_exp"), clamp_sh = r.floats(r.P + "swiglu_clamp_shexp");
    const int64_t hash_layers = r.I64("hash_layer_count");

    // ---- indexer and the candidate pool
    c.idx_heads = r.I("attention.indexer.head_count");
    c.idx_dim = r.I("attention.indexer.key_length");
    c.idx_topk = r.I("attention.indexer.top_k");
    c.cand_source = r.I("attention.candidate_source_layer_id");
    c.cand_block = r.I("attention.candidate_block_size");
    c.cand_topk_blocks = r.I("attention.candidate_topk_blocks");
    const std::vector<int> ratios = r.ints32(r.P + "attention.compress_ratios");
    c.kv_source = r.ints32(r.P + "attention.kv_source_layer_ids");
    c.index_source = r.ints32(r.P + "attention.index_source_layer_ids");

    // ---- mHC
    c.hc = r.I("hyper_connection.count");
    c.hc_iters = r.I("hyper_connection.sinkhorn_iterations");
    c.hc_eps = (float) r.F("hyper_connection.epsilon");

    // ---- Engram
    EngramConfig& e = c.engram;
    e.layers = r.ints32(r.P + "engram.layer_ids");
    e.heads = r.I("engram.n_heads");
    e.head_dim = r.I("engram.head_dim");
    e.ngram = r.I("engram.max_ngram_size");
    e.pad_id = r.I("engram.pad_token_id");
    e.cvocab = r.I("engram.compressed_vocab_size");
    e.num_embeddings = r.ints(r.P + "engram.num_embeddings");
    e.primes = r.ints(r.P + "engram.primes");
    e.offsets = r.ints(r.P + "engram.offsets");
    e.multipliers = r.ints(r.P + "engram.multipliers");
    f.throw_if_errors(meta.source() + ": the metadata is incomplete");

    // ---- contract constants the kernels hard-code (geometry.hpp / DS1.md section 2): a different model is refused, not mis-computed
    if (c.n_kv_head != 1) f.error("`deepseek41.attention.head_count_kv` = " + std::to_string(c.n_kv_head) + ", the model is MQA (one K == V head)");
    if (c.value_dim != c.head_dim)
        f.error("`deepseek41.attention.value_length` = " + std::to_string(c.value_dim) + " differs from key_length " + std::to_string(c.head_dim) + " (K == V)");
    if (c.n_shared != 1) f.error("`deepseek41.expert_shared_count` = " + std::to_string(c.n_shared) + ", this port has exactly one shared expert");
    if (c.gating_func != 4) f.error("`deepseek41.expert_gating_func` = " + std::to_string(c.gating_func) + ", only the sqrtsoftplus gate (ggml func 4) is implemented");
    if (!c.route_norm) f.error("`deepseek41.expert_weights_norm` is false; the router always renormalises the top-k weights");
    if (!close_rel(c.route_scale, kRouteScale, 1e-6))
        f.error("`deepseek41.expert_weights_scale` = " + std::to_string(c.route_scale) + ", the kernels compile in " + std::to_string(kRouteScale));
    if (hash_layers != 0) f.error("`deepseek41.hash_layer_count` = " + std::to_string(hash_layers) + ": hash-routed layers are not implemented");
    c.swiglu_limit = (float) kSwigluLimit;
    for (const auto* v : {&clamp_exp, &clamp_sh}) {
        if (v->size() != (size_t) c.n_layer) {
            f.error("`deepseek41.swiglu_clamp_*` has " + std::to_string(v->size()) + " entries, expected one per layer (" + std::to_string(c.n_layer) + ")");
            continue;
        }
        for (size_t l = 0; l < v->size(); ++l)
            if (!close_rel((*v)[l], kSwigluLimit, 1e-6)) {
                f.error("`deepseek41.swiglu_clamp_exp` / `swiglu_clamp_shexp` of layer " + std::to_string(l) + " is " + std::to_string((*v)[l]) +
                        ", the kernels compile in " + std::to_string(kSwigluLimit) + " for every layer");
                break;
            }
    }
    if (!close_rel(c.rms_eps, 1e-20, 1e-4))
        c.warnings.push_back("`deepseek41.attention.layer_norm_rms_epsilon` = " + std::to_string(c.rms_eps) + " (the reference uses 1e-20)");
    if (!close_rel(c.hc_eps, 1e-6, 1e-4))
        c.warnings.push_back("`deepseek41.hyper_connection.epsilon` = " + std::to_string(c.hc_eps) + " (the reference uses 1e-6)");
    if (c.rope_dim <= 0 || c.rope_dim % 2 || c.rope_dim >= c.head_dim)
        f.error("`deepseek41.rope.dimension_count` = " + std::to_string(c.rope_dim) + " must be even and below key_length " + std::to_string(c.head_dim));
    if (c.o_groups <= 0 || c.n_head % std::max(c.o_groups, 1))
        f.error("attention.head_count " + std::to_string(c.n_head) + " must divide by attention.output_group_count " + std::to_string(c.o_groups));
    if (c.rope_csa.factor <= 0 || c.rope_csa.orig_ctx <= 0 || c.rope_csa.base <= 1 || c.rope_swa.base <= 1)
        f.error("the RoPE parameters are not usable (factor " + std::to_string(c.rope_csa.factor) + ", original context " +
                std::to_string(c.rope_csa.orig_ctx) + ", bases " + std::to_string(c.rope_swa.base) + " / " + std::to_string(c.rope_csa.base) + ")");
    if (c.n_layer <= 0) f.error("`deepseek41.block_count` = " + std::to_string(c.n_layer));

    // ---- the layer map (ref/ds41/config.py validate + layer_modes; docs/deepseek/DS1.md section 3)
    if (c.n_layer > 0) {
        if ((int) ratios.size() != c.n_layer && (int) ratios.size() != c.n_layer + 3)
            f.error("`deepseek41.attention.compress_ratios` has " + std::to_string(ratios.size()) + " entries, expected " + std::to_string(c.n_layer) +
                    " (or " + std::to_string(c.n_layer + 3) + " with the DSpark entries)");
        else {
            c.n_ratio_entries = (int) ratios.size();
            c.compress_ratios.assign(ratios.begin(), ratios.begin() + c.n_layer);
        }
    }
    f.throw_if_errors(meta.source());   // the derivation below indexes the ratios
    const auto in_range = [&](const std::vector<int>& v, const char* what) {
        for (int l : v)
            if (l < 0 || l >= c.n_layer) f.error(std::string(what) + " holds layer " + std::to_string(l) + ", outside 0.." + std::to_string(c.n_layer - 1));
    };
    in_range(c.kv_source, "`attention.kv_source_layer_ids`");
    in_range(c.index_source, "`attention.index_source_layer_ids`");
    in_range(c.engram.layers, "`engram.layer_ids`");
    if (c.cand_source >= c.n_layer) f.error("`attention.candidate_source_layer_id` = " + std::to_string(c.cand_source) + " is outside the layers");
    for (int l = 0; l < c.n_layer; ++l)
        if (c.compress_ratios[(size_t) l] < 0 || c.compress_ratios[(size_t) l] > 2)
            f.error("`attention.compress_ratios`[" + std::to_string(l) + "] = " + std::to_string(c.compress_ratios[(size_t) l]) + ", only 0, 1 and 2 exist");
    f.throw_if_errors(meta.source());

    const std::set<int> kv_set(c.kv_source.begin(), c.kv_source.end()), idx_set(c.index_source.begin(), c.index_source.end());
    for (int l : kv_set) {
        if (c.compress_ratios[(size_t) l] == 0) f.error("kv source layer " + std::to_string(l) + " has compress ratio 0");
        if (!idx_set.count(l)) f.error("kv source layer " + std::to_string(l) + " is not an index source (a compressor layer owns an indexer)");
    }
    for (int l : idx_set)
        if (c.compress_ratios[(size_t) l] == 0) f.error("index source layer " + std::to_string(l) + " has compress ratio 0");
    if (c.cand_source >= 0 && !kv_set.count(c.cand_source))
        f.error("`attention.candidate_source_layer_id` = " + std::to_string(c.cand_source) + " is not a kv source (FULL) layer");
    if (c.cand_source >= 0 && (c.cand_block <= 0 || c.cand_topk_blocks <= 0)) f.error("the candidate pool sizes must be positive");
    if (std::set<int>(c.engram.layers.begin(), c.engram.layers.end()).size() != c.engram.layers.size()) f.error("`engram.layer_ids` repeats a layer");
    f.throw_if_errors(meta.source());

    c.layers.resize((size_t) c.n_layer);
    int latest_full = -1, latest_idx = -1;
    for (int l = 0; l < c.n_layer; ++l) {
        LayerInfo& li = c.layers[(size_t) l];
        li.layer = l;
        li.ratio = c.compress_ratios[(size_t) l];
        if (li.ratio == 0) li.role = Role::SWA;
        else if (kv_set.count(l)) li.role = Role::FULL;
        else if (idx_set.count(l)) li.role = Role::REINDEX;
        else li.role = Role::REUSE;
        li.has_compressor = li.role == Role::FULL;
        li.has_compressor_gate = li.role == Role::FULL && li.ratio > 1;
        li.has_indexer = li.role == Role::FULL || li.role == Role::REINDEX;
        li.has_index_compressor = li.role == Role::FULL;
        if (li.role == Role::FULL) {
            latest_full = l;
            latest_idx = l;
            li.kv_owner = li.index_k_owner = li.topk_source = l;
        } else if (li.role != Role::SWA) {
            if (latest_full < 0 || c.compress_ratios[(size_t) latest_full] != li.ratio) {
                f.error("layer " + std::to_string(l) + " (" + role_name(li.role) + ", ratio " + std::to_string(li.ratio) +
                        ") has no earlier FULL layer with the same compress ratio to read its compressed KV from");
                continue;
            }
            li.kv_owner = latest_full;
            if (li.role == Role::REINDEX) {
                li.index_k_owner = latest_full;
                li.topk_source = l;
                latest_idx = l;
            } else {
                if (latest_idx < 0) {
                    f.error("layer " + std::to_string(l) + " (REUSE) has no earlier indexer layer to take its top-k indices from");
                    continue;
                }
                li.topk_source = latest_idx;
            }
        }
        li.is_candidate_source = c.cand_source >= 0 && l == c.cand_source;
        li.uses_candidate_pool = li.has_indexer && c.cand_source >= 0 && c.cand_source < l;
    }
    for (size_t s = 0; s < c.engram.layers.size(); ++s) c.layers[(size_t) c.engram.layers[s]].engram_slot = (int) s;

    // ---- Engram constants (tools/ds41/ds41_spec.py check_engram_constants; the oracle's constants_from_gguf_metadata)
    if (!c.engram.layers.empty()) {
        const int n_slots = (int) c.engram.layers.size(), cols = c.engram.cols();
        if (e.heads <= 0 || e.ngram < 2 || e.head_dim <= 0 || e.cvocab <= 0) f.error("the Engram dimensions are not usable");
        else {
            const auto size_is = [&](const char* k, size_t got, size_t want) {
                if (got != want)
                    f.error(std::string("`deepseek41.engram.") + k + "` has " + std::to_string(got) + " entries, expected " + std::to_string(want) + " (" +
                            std::to_string(n_slots) + " layers)");
            };
            size_is("num_embeddings", e.num_embeddings.size(), (size_t) n_slots);
            size_is("primes", e.primes.size(), (size_t) n_slots * cols);
            size_is("offsets", e.offsets.size(), (size_t) n_slots * cols);
            size_is("multipliers", e.multipliers.size(), (size_t) n_slots * e.ngram);
            if (f.ok() && e.primes.size() == (size_t) n_slots * cols && e.offsets.size() == e.primes.size() && e.num_embeddings.size() == (size_t) n_slots) {
                for (int64_t p : e.primes)
                    if (!is_prime(p)) {
                        f.error("`deepseek41.engram.primes` holds " + std::to_string(p) + ", which is not prime");
                        break;
                    }
                for (size_t i = 1; i < e.primes.size(); ++i)
                    if (e.primes[i] != next_prime_above(e.primes[i - 1])) {
                        f.error("`deepseek41.engram.primes` are not consecutive primes (entry " + std::to_string(i) + ": " + std::to_string(e.primes[i]) +
                                " follows " + std::to_string(e.primes[i - 1]) + ")");
                        break;
                    }
                for (int s = 0; s < n_slots; ++s) {
                    int64_t sum = 0;
                    for (int col = 0; col < cols; ++col) {
                        if (e.offset(s, col) != sum) {
                            f.error("engram layer " + std::to_string(c.engram.layers[(size_t) s]) + ": `engram.offsets` are not the running sums of the primes (column " +
                                    std::to_string(col) + ": " + std::to_string(e.offset(s, col)) + ", expected " + std::to_string(sum) + ")");
                            break;
                        }
                        sum += e.prime(s, col);
                    }
                    if (sum != e.num_embeddings[(size_t) s])
                        f.error("engram layer " + std::to_string(c.engram.layers[(size_t) s]) + ": the primes sum to " + std::to_string(sum) +
                                " but `engram.num_embeddings` says " + std::to_string(e.num_embeddings[(size_t) s]) + " rows");
                }
            }
        }
        for (int64_t m : e.multipliers)
            if (m <= 0 || m % 2 == 0) {
                f.error("`deepseek41.engram.multipliers` must be positive odd numbers; found " + std::to_string(m));
                break;
            }
        if (e.pad_id < 0 || e.pad_id >= c.vocab) f.error("`deepseek41.engram.pad_token_id` = " + std::to_string(e.pad_id) + " is outside the vocabulary");
        const MetaValue* tm = meta.find("deepseek41.engram.token_map");
        if (!tm || !tm->is_array()) f.error(meta.source() + ": required metadata key `deepseek41.engram.token_map` is missing");
        else if ((int64_t) tm->count != c.vocab)
            f.error("`deepseek41.engram.token_map` has " + std::to_string(tm->count) + " entries, the vocabulary is " + std::to_string(c.vocab));
        else if (with_big_arrays) {
            if (tm->ia.size() != tm->count) f.error("`deepseek41.engram.token_map` was not read whole");
            else {
                int64_t lo = INT64_MAX, hi = INT64_MIN;
                e.token_map.reserve(tm->ia.size());
                for (int64_t v : tm->ia) {
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                    e.token_map.push_back((int32_t) v);
                }
                if (lo < 0 || hi >= e.cvocab)
                    f.error("`deepseek41.engram.token_map` ids span [" + std::to_string(lo) + ", " + std::to_string(hi) + "], compressed_vocab_size is " + std::to_string(e.cvocab));
                else if (hi + 1 != e.cvocab)
                    c.warnings.push_back("`engram.token_map` max id + 1 = " + std::to_string(hi + 1) + ", compressed_vocab_size is " + std::to_string(e.cvocab));
            }
        }
    }
    f.throw_if_errors(meta.source() + ": the model configuration is not consistent");
    return c;
}

std::string Ds41Config::describe() const {
    std::string s;
    char b[512];
    std::snprintf(b, sizeof b, "deepseek41 `%s`: %d layers, hidden %d, vocab %d, %d heads x %d (rope %d), q_lora %d, o %d x %d, window %d\n", name.c_str(),
                  n_layer, hidden, vocab, n_head, head_dim, rope_dim, q_lora, o_groups, o_lora, window);
    s += b;
    std::snprintf(b, sizeof b, "  MoE: %d experts, top-%d + %d shared, ff %d, route scale %g, swiglu limit %g | indexer %d x %d top-%d, candidate source L%d (blocks of %d, %d kept)\n",
                  n_expert, n_used, n_shared, ff, (double) route_scale, (double) swiglu_limit, idx_heads, idx_dim, idx_topk, cand_source, cand_block, cand_topk_blocks);
    s += b;
    std::snprintf(b, sizeof b, "  mHC %d copies, %d Sinkhorn iterations | rope: plain base %g; compressed base %g, YaRN factor %g, original ctx %lld, beta %g / %g | rms eps %g\n", hc,
                  hc_iters, rope_swa.base, rope_csa.base, rope_csa.factor, (long long) rope_csa.orig_ctx, rope_csa.beta_fast, rope_csa.beta_slow, (double) rms_eps);
    s += b;
    s += "  layer roles:";
    for (Role want : {Role::SWA, Role::FULL, Role::REINDEX, Role::REUSE}) {
        std::vector<int> ls;
        for (const LayerInfo& li : layers)
            if (li.role == want) ls.push_back(li.layer);
        if (!ls.empty()) s += std::string(" ") + role_name(want) + " " + join_ints(ls) + " |";
    }
    s.back() = '\n';
    if (!engram.layers.empty()) {
        std::snprintf(b, sizeof b, "  Engram: layers %s, %d heads x %d values, %d-grams (%d rows per token per layer), pad token %d, compressed vocab %d, tables:", join_ints(engram.layers).c_str(),
                      engram.heads, engram.head_dim, engram.ngram, engram.cols(), engram.pad_id, engram.cvocab);
        s += b;
        for (int64_t n : engram.num_embeddings) s += " " + std::to_string(n);
        s += " rows\n";
    }
    for (const std::string& w : warnings) s += "  warning: " + w + "\n";
    return s;
}

RopeTable build_rope_table(const RopeParams& p, int dim, int n_pos) {
    RopeTable t;
    t.dim = dim;
    t.n_pos = n_pos;
    const int half = dim / 2;
    std::vector<double> freqs((size_t) half);
    for (int i = 0; i < half; ++i) freqs[(size_t) i] = 1.0 / std::pow(p.base, (double) (2 * i) / (double) dim);
    if (p.yarn && p.orig_ctx > 0) {
        const auto corrected = [&](double rotations) {
            return dim * std::log((double) p.orig_ctx / (rotations * 2.0 * 3.14159265358979323846)) / (2.0 * std::log(p.base));
        };
        const double low = std::max(std::floor(corrected(p.beta_fast)), 0.0);
        const double high = std::min(std::ceil(corrected(p.beta_slow)), (double) (dim - 1));
        const double span = std::max(high - low, 1e-3);
        for (int i = 0; i < half; ++i) {
            const double ramp = std::min(std::max(((double) i - low) / span, 0.0), 1.0);
            const double smooth = 1.0 - ramp;
            const double fq = freqs[(size_t) i];
            freqs[(size_t) i] = fq / p.factor * (1.0 - smooth) + fq * smooth;
        }
    }
    t.cos.resize((size_t) n_pos * half);
    t.sin.resize((size_t) n_pos * half);
    for (int pos = 0; pos < n_pos; ++pos)
        for (int i = 0; i < half; ++i) {
            const double ang = (double) pos * freqs[(size_t) i];
            t.cos[(size_t) pos * half + i] = (float) std::cos(ang);
            t.sin[(size_t) pos * half + i] = (float) std::sin(ang);
        }
    return t;
}

}  // namespace strata::ds41::model
