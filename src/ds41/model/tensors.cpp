// src/ds41/model/tensors.cpp - DS1-A: the expected tensor table and its validation (include/strata/ds41/model/tensors.hpp).
#include "strata/ds41/model/tensors.hpp"

#include <algorithm>
#include <cstring>
#include <set>

namespace strata::ds41::model {

const char* group_name(Group g) {
    static const char* const n[] = {"embd", "norm", "attn", "compressor", "indexer", "router", "expert", "shared", "mhc", "engram_embed", "engram_proj", "head"};
    return (int) g >= 0 && (int) g < (int) Group::kCount ? n[(int) g] : "?";
}

std::string TensorSpec::dims_text() const {
    std::string s = "[";
    for (int d = 0; d < n_dims; ++d) s += (d ? ", " : "") + std::to_string(ne[d]);
    return s + "]";
}

namespace {

/// When a layer has the tensor.
enum class Need { Always, Compressor, CompressorGate, IndexCompressor, Indexer, Engram };

/// The ggml dims of a tensor, from the configuration (tools/ds41/ds41_spec.py tensor_specs; the oracle's weights.expected_shape reversed).
enum class Shape {
    Hidden, HcFn, HcMixes, Three,
    WqA, QLora, WqB, Wkv, HeadDim, Heads, WoA, WoB,
    CompProj, IdxCompKv, IdxDim,
    IdxQB, IdxProj,
    Router, Experts,
    ExpGateUp, ExpDown, ShGateUp, ShDown,
    EngProj, EngWkv, EngEmbed,
    Vocab,
};

struct Def {
    LT lt;
    const char* suffix;
    GgmlType type;
    Group group;
    Need need;
    Shape shape;
};

constexpr GgmlType F32 = GgmlType::F32, BF16 = GgmlType::BF16, Q8 = GgmlType::Q8_0, MX = GgmlType::MXFP4;

// the per-layer tensors, in the order of ds41_spec.tensor_specs
constexpr Def kLayerDefs[] = {
    {LT::AttnNorm, "attn_norm.weight", F32, Group::Norm, Need::Always, Shape::Hidden},
    {LT::WqA, "attn_q_a.weight", Q8, Group::Attn, Need::Always, Shape::WqA},
    {LT::QNorm, "attn_q_a_norm.weight", F32, Group::Attn, Need::Always, Shape::QLora},
    {LT::WqB, "attn_q_b.weight", Q8, Group::Attn, Need::Always, Shape::WqB},
    {LT::Wkv, "attn_kv.weight", Q8, Group::Attn, Need::Always, Shape::Wkv},
    {LT::KvNorm, "attn_kv_a_norm.weight", F32, Group::Attn, Need::Always, Shape::HeadDim},
    {LT::Sinks, "attn_sinks.weight", F32, Group::Attn, Need::Always, Shape::Heads},
    {LT::WoA, "attn_output_a.weight", Q8, Group::Attn, Need::Always, Shape::WoA},
    {LT::WoB, "attn_output_b.weight", Q8, Group::Attn, Need::Always, Shape::WoB},
    {LT::FfnNorm, "ffn_norm.weight", F32, Group::Norm, Need::Always, Shape::Hidden},
    {LT::Gate, "ffn_gate_inp.weight", BF16, Group::Router, Need::Always, Shape::Router},
    {LT::GateBias, "exp_probs_b.bias", F32, Group::Router, Need::Always, Shape::Experts},
    {LT::ExpGate, "ffn_gate_exps.weight", MX, Group::Expert, Need::Always, Shape::ExpGateUp},
    {LT::ExpDown, "ffn_down_exps.weight", MX, Group::Expert, Need::Always, Shape::ExpDown},
    {LT::ExpUp, "ffn_up_exps.weight", MX, Group::Expert, Need::Always, Shape::ExpGateUp},
    {LT::ShGate, "ffn_gate_shexp.weight", Q8, Group::Shared, Need::Always, Shape::ShGateUp},
    {LT::ShDown, "ffn_down_shexp.weight", Q8, Group::Shared, Need::Always, Shape::ShDown},
    {LT::ShUp, "ffn_up_shexp.weight", Q8, Group::Shared, Need::Always, Shape::ShGateUp},
    {LT::HcAttnFn, "hc_attn_fn.weight", F32, Group::Mhc, Need::Always, Shape::HcFn},
    {LT::HcAttnBase, "hc_attn_base.weight", F32, Group::Mhc, Need::Always, Shape::HcMixes},
    {LT::HcAttnScale, "hc_attn_scale.weight", F32, Group::Mhc, Need::Always, Shape::Three},
    {LT::HcFfnFn, "hc_ffn_fn.weight", F32, Group::Mhc, Need::Always, Shape::HcFn},
    {LT::HcFfnBase, "hc_ffn_base.weight", F32, Group::Mhc, Need::Always, Shape::HcMixes},
    {LT::HcFfnScale, "hc_ffn_scale.weight", F32, Group::Mhc, Need::Always, Shape::Three},
    {LT::CompKv, "attn_compressor_kv.weight", BF16, Group::Compressor, Need::Compressor, Shape::CompProj},
    {LT::CompNorm, "attn_compressor_norm.weight", F32, Group::Compressor, Need::Compressor, Shape::HeadDim},
    {LT::CompGate, "attn_compressor_gate.weight", BF16, Group::Compressor, Need::CompressorGate, Shape::CompProj},
    {LT::IdxCompKv, "indexer_compressor_kv.weight", BF16, Group::Indexer, Need::IndexCompressor, Shape::IdxCompKv},
    {LT::IdxCompNorm, "indexer_compressor_norm.weight", F32, Group::Indexer, Need::IndexCompressor, Shape::IdxDim},
    {LT::IdxQB, "indexer.attn_q_b.weight", Q8, Group::Indexer, Need::Indexer, Shape::IdxQB},
    {LT::IdxProj, "indexer.proj.weight", BF16, Group::Indexer, Need::Indexer, Shape::IdxProj},
    {LT::EngEmbed, "engram_embed.weight", MX, Group::EngramEmbed, Need::Engram, Shape::EngEmbed},
    {LT::EngK, "engram_k.weight", BF16, Group::EngramProj, Need::Engram, Shape::EngProj},
    {LT::EngQ, "engram_q.weight", BF16, Group::EngramProj, Need::Engram, Shape::EngProj},
    {LT::EngWkv, "engram_wkv.weight", Q8, Group::EngramProj, Need::Engram, Shape::EngWkv},
};

bool needed(Need n, const LayerInfo& li) {
    switch (n) {
    case Need::Always: return true;
    case Need::Compressor: return li.has_compressor;
    case Need::CompressorGate: return li.has_compressor_gate;
    case Need::IndexCompressor: return li.has_index_compressor;
    case Need::Indexer: return li.has_indexer;
    case Need::Engram: return li.is_engram();
    }
    return false;
}

void dims_of(Shape s, const Ds41Config& c, int engram_slot, uint64_t* ne, int& nd) {
    const uint64_t H = (uint64_t) c.hidden, E = (uint64_t) c.n_expert, FF = (uint64_t) c.ff, hd = (uint64_t) c.head_dim, nq = (uint64_t) c.n_q();
    const uint64_t hc = (uint64_t) c.hc, mix = (uint64_t) c.n_hc_mixes(), ql = (uint64_t) c.q_lora, og = (uint64_t) c.o_groups, ol = (uint64_t) c.o_lora;
    const uint64_t id = (uint64_t) c.idx_dim, ih = (uint64_t) c.idx_heads;
    nd = 2;
    switch (s) {
    case Shape::Hidden: nd = 1; ne[0] = H; break;
    case Shape::HcFn: ne[0] = hc * H; ne[1] = mix; break;
    case Shape::HcMixes: nd = 1; ne[0] = mix; break;
    case Shape::Three: nd = 1; ne[0] = 3; break;
    case Shape::WqA: ne[0] = H; ne[1] = ql; break;
    case Shape::QLora: nd = 1; ne[0] = ql; break;
    case Shape::WqB: ne[0] = ql; ne[1] = nq; break;
    case Shape::Wkv: ne[0] = H; ne[1] = hd; break;
    case Shape::HeadDim: nd = 1; ne[0] = hd; break;
    case Shape::Heads: nd = 1; ne[0] = (uint64_t) c.n_head; break;
    case Shape::WoA: ne[0] = nq / og; ne[1] = og * ol; break;
    case Shape::WoB: ne[0] = og * ol; ne[1] = H; break;
    case Shape::CompProj: ne[0] = H; ne[1] = hd; break;
    case Shape::IdxCompKv: ne[0] = hd; ne[1] = id; break;
    case Shape::IdxDim: nd = 1; ne[0] = id; break;
    case Shape::IdxQB: ne[0] = ql; ne[1] = ih * id; break;
    case Shape::IdxProj: ne[0] = H; ne[1] = ih; break;
    case Shape::Router: ne[0] = H; ne[1] = E; break;
    case Shape::Experts: nd = 1; ne[0] = E; break;
    case Shape::ExpGateUp: nd = 3; ne[0] = H; ne[1] = FF; ne[2] = E; break;
    case Shape::ExpDown: nd = 3; ne[0] = FF; ne[1] = H; ne[2] = E; break;
    case Shape::ShGateUp: ne[0] = H; ne[1] = FF; break;
    case Shape::ShDown: ne[0] = FF; ne[1] = H; break;
    case Shape::EngProj: ne[0] = H; ne[1] = hc; break;
    case Shape::EngWkv: ne[0] = (uint64_t) c.engram.cols() * (uint64_t) c.engram.head_dim; ne[1] = (hc + 1) * H; break;
    case Shape::EngEmbed:
        ne[0] = (uint64_t) c.engram.head_dim;
        ne[1] = engram_slot >= 0 && (size_t) engram_slot < c.engram.num_embeddings.size() ? (uint64_t) c.engram.num_embeddings[(size_t) engram_slot] : 0;
        break;
    case Shape::Vocab: ne[0] = H; ne[1] = (uint64_t) c.vocab; break;
    }
}

TensorSpec make(const Ds41Config& c, const Def& d, int layer, int engram_slot) {
    TensorSpec t;
    t.name = "blk." + std::to_string(layer) + "." + d.suffix;
    t.type = d.type;
    t.group = d.group;
    t.layer = layer;
    t.lt = d.lt;
    dims_of(d.shape, c, engram_slot, t.ne, t.n_dims);
    return t;
}

TensorSpec make_global(const Ds41Config& c, LT lt, const char* name, GgmlType type, Group g, Shape s) {
    TensorSpec t;
    t.name = name;
    t.type = type;
    t.group = g;
    t.layer = -1;
    t.lt = lt;
    dims_of(s, c, -1, t.ne, t.n_dims);
    return t;
}

/// Dims as the file stores them, padded with 1 up to three (a 2-D tensor is stored with n_dims == 2).
bool same_dims(const TensorLoc& t, const TensorSpec& s) {
    if (t.ne[3] != 1) return false;
    for (int d = 0; d < 3; ++d) {
        const uint64_t a = d < t.n_dims ? t.ne[d] : 1, b = d < s.n_dims ? s.ne[d] : 1;
        if (a != b) return false;
    }
    return true;
}

}  // namespace

std::vector<TensorSpec> expected_tensors(const Ds41Config& c) {
    std::vector<TensorSpec> out;
    out.push_back(make_global(c, LT::TokenEmbd, "token_embd.weight", BF16, Group::Embd, Shape::Vocab));
    for (const LayerInfo& li : c.layers)
        for (const Def& d : kLayerDefs)
            if (needed(d.need, li)) out.push_back(make(c, d, li.layer, li.engram_slot));
    out.push_back(make_global(c, LT::OutputNorm, "output_norm.weight", F32, Group::Head, Shape::Hidden));
    out.push_back(make_global(c, LT::Output, "output.weight", BF16, Group::Head, Shape::Vocab));
    return out;
}

bool is_ignored_tensor(const std::string& name) {
    const size_t dot = name.find('.', 4);
    if (name.compare(0, 4, "blk.") != 0 || dot == std::string::npos) return false;
    return name.compare(dot + 1, std::string::npos, "exp_probs_b_vl.bias") == 0;
}

void validate_tensors(const Ds41Config& c, const TensorDir& dir, Findings& f, const ValidateOptions& opt, size_t* ignored) {
    const std::vector<TensorSpec> specs = expected_tensors(c);
    std::set<std::string> names;
    std::set<std::string> bad_expert_types;
    for (const TensorSpec& s : specs) {
        names.insert(s.name);
        const TensorLoc* t = dir.find(s.name);
        if (!t) {
            f.error("tensor `" + s.name + "` is missing (expected " + type_name(s.type) + " " + s.dims_text() + ")");
            continue;
        }
        if (!same_dims(*t, s))
            f.error("tensor `" + s.name + "` has dims " + t->dims_text() + ", but the model's metadata implies " + s.dims_text() + " (" + type_name(s.type) + ")");
        if (t->type != s.type) {
            if (s.group == Group::Expert) bad_expert_types.insert(type_name(t->type));
            else f.error("tensor `" + s.name + "` is " + type_name(t->type) + ", expected " + type_name(s.type));
        } else if (t->nbytes == 0) {
            f.error("tensor `" + s.name + "` (" + type_name(t->type) + " " + t->dims_text() + ") has no known byte size: a row is not a whole number of blocks");
        }
    }
    if (!bad_expert_types.empty()) {
        std::string types;
        for (const std::string& ty : bad_expert_types) types += (types.empty() ? "" : "/") + ty;
        f.error("the routed experts are " + types + ", not MXFP4: the port's expert format is MXFP4 (docs/deepseek/PLAN.md section 1); the expert blob layout and "
                "every expert kernel assume 17-byte MXFP4 blocks. Use an MXFP4 GGUF (mxxm-t/DeepSeek-V4.1-Flash-GGUF)");
    }
    size_t n_ignored = 0;
    std::vector<std::string> extra;
    for (const auto& kv : dir.all()) {
        if (names.count(kv.first)) continue;
        if (is_ignored_tensor(kv.first)) {
            ++n_ignored;
            continue;
        }
        extra.push_back(kv.first + " (" + type_name(kv.second.type) + " " + kv.second.dims_text() + ")");
    }
    if (!extra.empty()) {
        std::string list;
        for (size_t i = 0; i < extra.size() && i < 4; ++i) list += (i ? ", " : "") + extra[i];
        const std::string msg = std::to_string(extra.size()) + " tensor(s) are not part of the deepseek41 contract, e.g. " + list;
        if (opt.allow_unexpected) f.warn(msg);
        else f.error(msg);
    }
    if (ignored) *ignored = n_ignored;
}

ByteTally tally(const std::vector<TensorSpec>& specs, const Ds41Config& c) {
    ByteTally t;
    for (const TensorSpec& s : specs) {
        const uint64_t b = s.nbytes();
        t.group[(int) s.group] += b;
        t.total += b;
        ++t.n_tensors;
        if (s.group != Group::Embd && s.group != Group::Expert && s.group != Group::EngramEmbed) t.dense_device += b;
        if (s.layer == c.layers.front().layer && s.group == Group::Expert) t.expert_bytes_each += b / (uint64_t) c.n_expert;
    }
    return t;
}

}  // namespace strata::ds41::model
