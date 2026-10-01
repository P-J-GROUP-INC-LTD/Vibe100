// src/ds41/model/model.cpp - DS1-A: the weights (dense tensors to the device, big tables left in the mapping) and the load sequence
// (include/strata/ds41/model/model.hpp).
#include "strata/ds41/model/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>

namespace strata::ds41::model {

namespace {

constexpr uint64_t kDevAlign = 256;                // every tensor starts on a 256-byte boundary of the device block

uint64_t align_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

/// Which LT slots live on the device: the first block of the enum, plus the output norm and the head.
bool on_device(LT lt) { return (int) lt < (int) LT::kDeviceLayerCount || lt == LT::OutputNorm || lt == LT::Output; }

DevTensor make_dev(const TensorLoc& loc, const uint8_t* dev_ptr) {
    DevTensor t;
    t.p = dev_ptr;
    t.nbytes = loc.nbytes;
    t.type = loc.type;
    t.ne0 = (int64_t) loc.ne[0];
    t.ne1 = loc.n_dims > 1 ? (int64_t) loc.ne[1] : 1;
    t.ne2 = loc.n_dims > 2 ? (int64_t) loc.ne[2] : 1;
    return t;
}

HostTensor make_host(const TensorLoc& loc, const uint8_t* host_ptr) {
    HostTensor t;
    t.p = host_ptr;
    t.nbytes = loc.nbytes;
    t.type = loc.type;
    t.ne0 = (int64_t) loc.ne[0];
    t.ne1 = loc.n_dims > 1 ? (int64_t) loc.ne[1] : 1;
    t.ne2 = loc.n_dims > 2 ? (int64_t) loc.ne[2] : 1;
    return t;
}

std::string gib_text(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.2f GiB", (double) b / 1073741824.0);
    return buf;
}

}  // namespace

// ================================================================================================ weights
DevTensor* LayerWeights::slot(LT lt) {
    switch (lt) {
    case LT::AttnNorm: return &attn_norm;
    case LT::FfnNorm: return &ffn_norm;
    case LT::HcAttnFn: return &hc_attn_fn;
    case LT::HcAttnBase: return &hc_attn_base;
    case LT::HcAttnScale: return &hc_attn_scale;
    case LT::HcFfnFn: return &hc_ffn_fn;
    case LT::HcFfnBase: return &hc_ffn_base;
    case LT::HcFfnScale: return &hc_ffn_scale;
    case LT::WqA: return &wq_a;
    case LT::QNorm: return &q_norm;
    case LT::WqB: return &wq_b;
    case LT::Wkv: return &wkv;
    case LT::KvNorm: return &kv_norm;
    case LT::Sinks: return &attn_sinks;
    case LT::WoA: return &wo_a;
    case LT::WoB: return &wo_b;
    case LT::CompKv: return &comp_kv;
    case LT::CompNorm: return &comp_norm;
    case LT::CompGate: return &comp_gate;
    case LT::IdxQB: return &idx_q_b;
    case LT::IdxProj: return &idx_proj;
    case LT::IdxCompKv: return &idx_comp_kv;
    case LT::IdxCompNorm: return &idx_comp_norm;
    case LT::Gate: return &gate;
    case LT::GateBias: return &gate_bias;
    case LT::ShGate: return &sh_gate;
    case LT::ShUp: return &sh_up;
    case LT::ShDown: return &sh_down;
    case LT::EngQ: return &eng_q;
    case LT::EngK: return &eng_k;
    case LT::EngWkv: return &eng_wkv;
    default: return nullptr;
    }
}

WeightsCore& WeightsCore::operator=(WeightsCore&& o) noexcept {
    if (this != &o) {
        release();
        layer = std::move(o.layer);
        head = o.head;
        output_norm = o.output_norm;
        token_embd = o.token_embd;
        cfg = o.cfg;
        dev_ = o.dev_;
        dev_base_ = o.dev_base_;
        dev_bytes_ = o.dev_bytes_;
        o.layer.clear();
        o.head = o.output_norm = DevTensor{};
        o.token_embd = HostTensor{};
        o.cfg = nullptr;
        o.dev_ = nullptr;
        o.dev_base_ = nullptr;
        o.dev_bytes_ = 0;
    }
    return *this;
}

void WeightsCore::release() {
    if (dev_ && dev_base_) dev_->release(dev_base_);
    dev_base_ = nullptr;
    dev_bytes_ = 0;
    layer.clear();
    head = output_norm = DevTensor{};
    token_embd = HostTensor{};
}

void WeightsCore::load(ModelDev& dev, const GgufSet& gguf, const Ds41Config& c, const std::vector<TensorSpec>& specs, const TensorDir& dir, const LogFn& log,
                       uint64_t piece_bytes) {
    release();
    if (piece_bytes == 0) throw ModelError("WeightsCore::load: an upload piece of 0 bytes");
    dev_ = &dev;
    cfg = &c;
    layer.assign((size_t) c.n_layer, LayerWeights{});
    for (int l = 0; l < c.n_layer; ++l) layer[(size_t) l].layer = l;

    // ---- the device block: one allocation, every dense tensor at a 256-byte boundary, in file order of the spec list
    struct Item {
        const TensorSpec* spec;
        const TensorLoc* loc;
        uint64_t off;
    };
    std::vector<Item> items;
    uint64_t total = 0;
    for (const TensorSpec& s : specs) {
        if (!on_device(s.lt)) continue;
        const TensorLoc& loc = dir.at(s.name);
        if (loc.nbytes == 0) throw ModelError("tensor `" + s.name + "` has no known size");
        const uint64_t off = align_up(total, kDevAlign);
        items.push_back({&s, &loc, off});
        total = off + loc.nbytes;
    }
    if (log) log("weights: " + std::to_string(items.size()) + " dense tensors, " + gib_text(total) + " to the device");
    dev_base_ = dev.alloc((size_t) total);
    dev_bytes_ = total;
    uint8_t* const base = static_cast<uint8_t*>(dev_base_);

    // ---- upload in bounded pieces straight from the mapping
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t done = 0;
    for (const Item& it : items) {
        const uint8_t* src = gguf.data(*it.loc);
        for (uint64_t pos = 0; pos < it.loc->nbytes; pos += piece_bytes)
            dev.h2d(base + it.off + pos, src + pos, (size_t) std::min<uint64_t>(piece_bytes, it.loc->nbytes - pos));
        done += it.loc->nbytes;
        const DevTensor dt = make_dev(*it.loc, base + it.off);
        if (it.spec->lt == LT::OutputNorm) output_norm = dt;
        else if (it.spec->lt == LT::Output) head = dt;
        else {
            LayerWeights& lw = layer[(size_t) it.spec->layer];
            DevTensor* slot = lw.slot(it.spec->lt);
            if (!slot) throw ModelError("internal: tensor `" + it.spec->name + "` has no weight slot");
            *slot = dt;
        }
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (log) {
        char b[160];
        std::snprintf(b, sizeof b, "weights: uploaded %s in %.1f s (%.2f GB/s)", gib_text(done).c_str(), sec, sec > 0 ? (double) done / 1e9 / sec : 0.0);
        log(b);
    }

    // ---- the tensors that stay in the file
    for (const TensorSpec& s : specs) {
        if (on_device(s.lt)) continue;
        const TensorLoc& loc = dir.at(s.name);
        const HostTensor ht = make_host(loc, gguf.data(loc));
        switch (s.lt) {
        case LT::TokenEmbd:
            token_embd = ht;
            gguf.advise_random(loc);
            break;
        case LT::EngEmbed:
            layer[(size_t) s.layer].eng_table = ht;
            gguf.advise_random(loc);
            break;
        case LT::ExpGate:
        case LT::ExpUp:
        case LT::ExpDown: {
            ExpertSlices& ex = layer[(size_t) s.layer].experts;
            const uint8_t** p = s.lt == LT::ExpGate ? &ex.gate : s.lt == LT::ExpUp ? &ex.up : &ex.down;
            const TensorLoc** pl = s.lt == LT::ExpGate ? &ex.loc_gate : s.lt == LT::ExpUp ? &ex.loc_up : &ex.loc_down;
            *p = ht.p;
            *pl = &loc;
            break;
        }
        default: throw ModelError("internal: tensor `" + s.name + "` has no home");
        }
    }
}

// ================================================================================================ RoPE tables on the device
DeviceRope::DeviceRope(ModelDev& dev, const Ds41Config& c, int n_pos) : dev_(&dev), n_pos_(n_pos) {
    if (n_pos <= 0) throw ModelError("DeviceRope: " + std::to_string(n_pos) + " positions");
    if (c.rope_dim <= 0 || c.rope_dim % 2) throw ModelError("DeviceRope: rope dimension " + std::to_string(c.rope_dim));
    try {
        for (int kind = 0; kind < 2; ++kind) {
            const RopeTable t = build_rope_table(kind ? c.rope_csa : c.rope_swa, c.rope_dim, n_pos);
            const size_t n = t.cos.size() * sizeof(float);
            cos_[kind] = static_cast<float*>(dev.alloc(n));
            sin_[kind] = static_cast<float*>(dev.alloc(n));
            dev.h2d(cos_[kind], t.cos.data(), n);
            dev.h2d(sin_[kind], t.sin.data(), n);
            bytes_ += 2 * n;
        }
    } catch (...) {
        release();
        throw;
    }
    kind_.resize((size_t) c.n_layer);
    for (int l = 0; l < c.n_layer; ++l) kind_[(size_t) l] = c.layer(l).ratio ? 1 : 0;
}

DeviceRope& DeviceRope::operator=(DeviceRope&& o) noexcept {
    if (this != &o) {
        release();
        dev_ = o.dev_;
        for (int k = 0; k < 2; ++k) {
            cos_[k] = o.cos_[k];
            sin_[k] = o.sin_[k];
            o.cos_[k] = o.sin_[k] = nullptr;
        }
        kind_ = std::move(o.kind_);
        n_pos_ = o.n_pos_;
        bytes_ = o.bytes_;
        o.dev_ = nullptr;
        o.n_pos_ = 0;
        o.bytes_ = 0;
    }
    return *this;
}

void DeviceRope::release() {
    for (int k = 0; k < 2; ++k) {
        if (dev_ && cos_[k]) dev_->release(cos_[k]);
        if (dev_ && sin_[k]) dev_->release(sin_[k]);
        cos_[k] = sin_[k] = nullptr;
    }
    kind_.clear();
    n_pos_ = 0;
    bytes_ = 0;
}

// ================================================================================================ the load sequence
void load_parts(ModelDev& dev, const GgufSet& gguf, const Ds41Config& cfg, const TensorDir& dir, const ExpertDims& dims, const LoadOptions& opt, WeightsCore& weights,
                ExpertArena& arena, GpuExpertCache& cache, MemoryPlan& plan, std::vector<std::string>& warnings) {
    std::mutex log_mu;
    const LogFn base_log = opt.log ? opt.log : LogFn([](const std::string& m) { std::fprintf(stderr, "ds41: %s\n", m.c_str()); });
    const LogFn log = [&](const std::string& m) {
        std::lock_guard<std::mutex> lk(log_mu);
        base_log(m);
    };

    // ---- the expert shape is G's; the file must agree with it and with itself
    if (dims.n_layer != cfg.n_layer || dims.n_expert != cfg.n_expert || dims.hidden != cfg.hidden || dims.ff != cfg.ff)
        throw ModelError("internal: the expert shape (" + std::to_string(dims.n_layer) + " layers x " + std::to_string(dims.n_expert) + " experts, hidden " + std::to_string(dims.hidden) +
                         ", ff " + std::to_string(dims.ff) + ") is not the file's (" + std::to_string(cfg.n_layer) + " x " + std::to_string(cfg.n_expert) + ", " + std::to_string(cfg.hidden) +
                         ", " + std::to_string(cfg.ff) + ")");
    if (!dims.valid()) throw ModelError("the expert shape is not supported (hidden a multiple of 32, ff a multiple of 64 required)");

    // ---- every tensor: present, typed, shaped, accounted for
    Findings f;
    size_t ignored = 0;
    ValidateOptions vo;
    vo.allow_unexpected = opt.allow_unexpected_tensors;
    validate_tensors(cfg, dir, f, vo, &ignored);
    const std::vector<TensorSpec> specs = expected_tensors(cfg);
    for (const TensorSpec& s : specs)       // the expert tensors must be exactly n_expert slices of the ExpertDims layout
        if (s.group == Group::Expert) {
            const TensorLoc* t = dir.find(s.name);
            const uint64_t want = (s.lt == LT::ExpDown ? dims.down_bytes() : dims.gate_bytes()) * (uint64_t) dims.n_expert;
            if (t && t->type == GgmlType::MXFP4 && t->nbytes != want)
                f.error("tensor `" + s.name + "` is " + std::to_string(t->nbytes) + " bytes, " + std::to_string(dims.n_expert) + " experts of this shape take " + std::to_string(want));
        }
    f.throw_if_errors("the GGUF's tensors do not match the deepseek41 contract");
    for (const std::string& w : f.warnings) warnings.push_back(w);
    for (const std::string& w : cfg.warnings) warnings.push_back(w);
    if (ignored) warnings.push_back(std::to_string(ignored) + " tensor(s) of the file are not part of the model and were ignored (vision / unused bias)");

    // ---- the memory plan, printed once
    const ByteTally tl = tally(specs, cfg);
    PlanInputs pin = opt.plan;
    const MemoryPlan first = make_memory_plan(cfg, tl, pin);
    int n_slots = opt.n_slots;
    if (n_slots < 0) n_slots = first.cache_slots_fit;
    n_slots = std::min<int64_t>(n_slots, (int64_t) cfg.n_layer * cfg.n_expert);
    pin.n_slots = n_slots;
    plan = make_memory_plan(cfg, tl, pin);
    log(plan.text());
    for (const std::string& w : plan.warnings) warnings.push_back(w);
    if (opt.initial_fill.size() > (size_t) n_slots)
        throw ModelError("initial fill lists " + std::to_string(opt.initial_fill.size()) + " experts but the cache has " + std::to_string(n_slots) + " slots");
    for (const ExpertId& e : opt.initial_fill)
        if (e.layer < 0 || e.layer >= cfg.n_layer || e.expert < 0 || e.expert >= cfg.n_expert)
            throw ModelError("initial fill names expert " + std::to_string(e.layer) + ":" + std::to_string(e.expert) + ", outside the model");

    // ---- the dense weights go to the device; the big tables are mapped
    weights.load(dev, gguf, cfg, specs, dir, log, opt.upload_piece_bytes);
    if (opt.drop_page_cache) {
        for (const TensorSpec& s : specs)
            if (on_device(s.lt)) gguf.drop_cache(dir.at(s.name));      // the device has its copy; the page cache need not keep the file's
    }

    // ---- the CPU arena
    std::vector<ExpertSlices> slices;
    slices.reserve((size_t) cfg.n_layer);
    for (int l = 0; l < cfg.n_layer; ++l) {
        ExpertSlices s = weights.layer[(size_t) l].experts;
        s.d = dims;
        slices.push_back(s);
    }
    for (int l = 0; l < cfg.n_layer; ++l) weights.layer[(size_t) l].experts.d = dims;
    if (opt.build_arena) {
        ArenaOptions ao = opt.arena;
        ao.log = log;
        const auto prefetch_layer = [&](int l) {
            if (l < 0 || l >= cfg.n_layer) return;
            for (const TensorLoc* t : {slices[(size_t) l].loc_gate, slices[(size_t) l].loc_up, slices[(size_t) l].loc_down})
                if (t) gguf.prefetch(*t);
        };
        for (int l = 0; l < std::min(opt.prefetch_layers, cfg.n_layer); ++l) prefetch_layer(l);
        const std::function<void(int)> user_done = opt.arena.layer_done;
        int n_done = 0;
        ao.layer_done = [&, user_done](int l) {
            if (opt.prefetch_layers > 0) prefetch_layer(l + opt.prefetch_layers);
            if (opt.drop_page_cache) {
                if (slices[(size_t) l].loc_gate) gguf.drop_cache(*slices[(size_t) l].loc_gate);
                if (slices[(size_t) l].loc_up) gguf.drop_cache(*slices[(size_t) l].loc_up);
                if (slices[(size_t) l].loc_down) gguf.drop_cache(*slices[(size_t) l].loc_down);
            }
            {
                std::lock_guard<std::mutex> lk(log_mu);
                ++n_done;
                if (n_done % 10 == 0 || n_done == cfg.n_layer) base_log("expert arena: " + std::to_string(n_done) + " of " + std::to_string(cfg.n_layer) + " layers copied");
            }
            if (user_done) user_done(l);
        };
        arena = ExpertArena::build(dims, slices, ao);
    }

    // ---- the GPU cache: allocated empty (residency all -1), then filled statically
    cache = GpuExpertCache(dev, dims, n_slots);
    if (opt.fill_cache && n_slots > 0) {
        std::vector<ExpertId> list = opt.initial_fill.empty() ? static_fill_by_index(cfg.n_layer, cfg.n_expert, n_slots) : opt.initial_fill;
        const auto t0 = std::chrono::steady_clock::now();
        int slot = 0;
        for (const ExpertId& e : list) {
            if (arena.built()) cache.fill_from_arena(dev, slot, e.layer, e.expert, arena, true);
            else cache.fill_from_gguf(dev, slot, e.layer, e.expert, slices[(size_t) e.layer], true);
            ++slot;
            if (slot % 200 == 0) log("GPU expert cache: " + std::to_string(slot) + " of " + std::to_string(list.size()) + " experts uploaded");
        }
        cache.upload_residency(dev);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        char b[200];
        std::snprintf(b, sizeof b, "GPU expert cache: %d experts (%s) uploaded in %.1f s from the %s", (int) list.size(), gib_text((uint64_t) list.size() * dims.blob_bytes()).c_str(), sec,
                      arena.built() ? "CPU arena" : "GGUF");
        log(b);
    }

    // ---- what is actually held
    char b[512];
    std::snprintf(b, sizeof b, "loaded: device %s dense + %s cache (%d slots, %d resident) | host: arena %s per socket%s, token_embd %s and the Engram tables mapped", gib_text(weights.device_bytes()).c_str(),
                  gib_text(cache.device_bytes()).c_str(), cache.n_slots(), cache.n_resident(), arena.built() ? gib_text(arena.bytes_per_socket()).c_str() : "(not built)",
                  arena.built() && arena.bound(0) && arena.bound(1) ? " (NUMA-bound)" : "", gib_text(weights.token_embd.nbytes).c_str());
    log(b);
}

}  // namespace strata::ds41::model
