// src/ds41/model/experts.cpp - DS1-A: the routed experts' memory: the half packer, the per-socket CPU arena, the GPU expert cache and the
// lists that fill it (include/strata/ds41/model/model.hpp).
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include <new>
#include <set>
#include <thread>

#include "strata/ds41/model/model.hpp"

namespace strata::ds41::model {

namespace {
uint64_t round_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
}  // namespace

// ================================================================================================ the half packer
void pack_half(const ExpertDims& d, const uint8_t* gate, const uint8_t* up, const uint8_t* down, int h, uint8_t* out) {
    if (!d.valid()) throw ModelError("pack_half: the expert shape is not valid (hidden " + std::to_string(d.hidden) + ", ff " + std::to_string(d.ff) + ")");
    if (h < 0 || h > 1) throw ModelError("pack_half: half " + std::to_string(h) + " (there are two)");
    const uint64_t rows = (uint64_t) d.half_ff(), grb = d.gate_row_bytes();
    std::memcpy(out, gate + (uint64_t) h * rows * grb, rows * grb);
    std::memcpy(out + d.half_gate_bytes(), up + (uint64_t) h * rows * grb, rows * grb);
    uint8_t* o = out + 2 * d.half_gate_bytes();
    const uint64_t drb = d.down_row_bytes(), hrb = d.half_down_row_bytes();
    for (int r = 0; r < d.hidden; ++r) std::memcpy(o + (uint64_t) r * hrb, down + (uint64_t) r * drb + (uint64_t) h * hrb, hrb);
}

void unpack_halves(const ExpertDims& d, const uint8_t* half0, const uint8_t* half1, uint8_t* out) {
    if (!d.valid()) throw ModelError("unpack_halves: the expert shape is not valid");
    const uint64_t rows = (uint64_t) d.half_ff(), grb = d.gate_row_bytes(), drb = d.down_row_bytes(), hrb = d.half_down_row_bytes();
    const uint8_t* half[2] = {half0, half1};
    for (int k = 0; k < 2; ++k) {
        std::memcpy(out + (uint64_t) k * rows * grb, half[k], rows * grb);
        std::memcpy(out + d.gate_bytes() + (uint64_t) k * rows * grb, half[k] + d.half_gate_bytes(), rows * grb);
        const uint8_t* src = half[k] + 2 * d.half_gate_bytes();
        uint8_t* dst = out + 2 * d.gate_bytes() + (uint64_t) k * hrb;
        for (int r = 0; r < d.hidden; ++r) std::memcpy(dst + (uint64_t) r * drb, src + (uint64_t) r * hrb, hrb);
    }
}

// ================================================================================================ NUMA placement
ArenaNodes choose_arena_nodes(const platform::NumaTopology& topo) {
    ArenaNodes r;
    if (!topo.available) {
        r.note = "no NUMA topology (" + (topo.why.empty() ? std::string("unavailable") : topo.why) + "): the halves are not bound";
        return r;
    }
    std::vector<const platform::NumaNode*> with_cpus;
    for (const platform::NumaNode& n : topo.nodes)
        if (!n.cpus.empty()) with_cpus.push_back(&n);
    if (topo.nodes.size() < 2) {
        r.note = "one NUMA node: the halves are not bound";
        return r;
    }
    if (topo.nodes.size() == 2) {
        if (with_cpus.size() != 2) {
            r.note = "two NUMA nodes, but one has no CPUs (a memory-only node): the halves are not bound";
            return r;
        }
        for (int h = 0; h < 2; ++h) {
            r.node[h] = with_cpus[(size_t) h]->id;
            r.cpus[h] = with_cpus[(size_t) h]->cpus;
        }
        r.bind = true;
        r.note = "two NUMA nodes: half 0 on node " + std::to_string(r.node[0]) + ", half 1 on node " + std::to_string(r.node[1]);
        return r;
    }
    // more than two nodes: sub-NUMA clustering (nodes per socket) is the case that makes sense; group by package
    std::map<int, std::vector<const platform::NumaNode*>> by_pkg;
    bool unknown = false;
    for (const platform::NumaNode* n : with_cpus) {
        if (n->package < 0) unknown = true;
        by_pkg[n->package].push_back(n);
    }
    if (unknown || by_pkg.size() != 2) {
        r.note = std::to_string(topo.nodes.size()) + " NUMA nodes on " + (unknown ? std::string("an unknown number of") : std::to_string(by_pkg.size())) +
                 " package(s): the port splits an expert over exactly two sockets, so the halves are not bound";
        return r;
    }
    int h = 0;
    for (const auto& kv : by_pkg) {
        const platform::NumaNode* lowest = kv.second.front();
        r.node[h] = lowest->id;
        r.cpus[h] = lowest->cpus;
        ++h;
    }
    r.bind = true;
    r.note = std::to_string(topo.nodes.size()) + " NUMA nodes on 2 sockets (sub-NUMA clustering): half 0 on node " + std::to_string(r.node[0]) + ", half 1 on node " +
             std::to_string(r.node[1]) + " (each half lives in ONE sub-node of its socket)";
    return r;
}

// ================================================================================================ the CPU arena
ExpertArena ExpertArena::build(const ExpertDims& d, const std::vector<ExpertSlices>& slices, const ArenaOptions& opt) {
    if (!d.valid()) throw ModelError("ExpertArena: the expert shape is not valid (hidden " + std::to_string(d.hidden) + ", ff " + std::to_string(d.ff) + ")");
    if ((int) slices.size() != d.n_layer)
        throw ModelError("ExpertArena: " + std::to_string(slices.size()) + " layers of expert slices, expected " + std::to_string(d.n_layer));
    for (int l = 0; l < d.n_layer; ++l)
        if (!slices[(size_t) l].gate || !slices[(size_t) l].up || !slices[(size_t) l].down)
            throw ModelError("ExpertArena: layer " + std::to_string(l) + " has no expert slices");
    const LogFn log = opt.log ? opt.log : LogFn([](const std::string&) {});

    ExpertArena a;
    a.d_ = d;
    platform::NumaTopology local;
    const platform::NumaTopology* topo = nullptr;
    if (opt.numa) {
        if (opt.topology) topo = opt.topology;
        else {
            local = platform::numa_discover();
            topo = &local;
        }
    }
    ArenaNodes an;
    if (topo) an = choose_arena_nodes(*topo);
    else an.note = "NUMA binding switched off";
    log("expert arena: " + an.note);

    const uint64_t bytes = a.bytes_per_socket();
    for (int h = 0; h < 2; ++h) {
        Block& b = a.blk_[h];
        b.bytes = bytes;
        const int want_node = an.bind ? an.node[h] : -1;
        const std::vector<int> no_cpus;
        if (b.nb.allocate(bytes, want_node, opt.hugepages, an.bind ? an.cpus[h] : no_cpus)) {
            b.data = b.nb.data();
            b.node = want_node;
            b.bound = b.nb.bound();
            b.note = std::string(platform::page_kind_name(b.nb.page_kind())) + "; " + b.nb.note();
        } else {
            const std::string why = b.nb.note();
            b.plain.reset(new (std::nothrow) uint8_t[(size_t) bytes + 64]);
            if (!b.plain)
                throw ModelError("cannot allocate " + std::to_string(bytes >> 20) + " MiB for CPU expert half " + std::to_string(h) + " (out of memory; mapping said: " + why + ")");
            b.data = reinterpret_cast<uint8_t*>(round_up(reinterpret_cast<uintptr_t>(b.plain.get()), 64));
            b.node = -1;
            b.bound = false;
            b.note = "plain memory (" + why + ")";
        }
        char line[512];
        std::snprintf(line, sizeof line, "expert arena half %d: %.2f GiB, node %d%s, %s", h, (double) bytes / (1ull << 30), b.node, b.bound ? " (bound)" : " (not bound)", b.note.c_str());
        log(line);
    }

    // the copy: tasks of a few experts, layer-major, so that the file is read in order and a finished layer's pages can be dropped at once
    constexpr int kPerTask = 8;
    struct Task {
        int layer, e0, e1;
    };
    std::vector<Task> tasks;
    std::vector<std::atomic<int>> remaining((size_t) d.n_layer);
    for (int l = 0; l < d.n_layer; ++l) {
        int n = 0;
        for (int e0 = 0; e0 < d.n_expert; e0 += kPerTask, ++n) tasks.push_back({l, e0, std::min(d.n_expert, e0 + kPerTask)});
        remaining[(size_t) l].store(n);
    }
    int threads = opt.threads > 0 ? opt.threads : (int) std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    threads = std::max(1, std::min<int>(threads, (int) tasks.size()));
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr first_error;
    std::mutex err_mu;
    const auto work = [&]() {
        try {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= tasks.size() || failed.load()) return;
                const Task& t = tasks[i];
                const ExpertSlices& s = slices[(size_t) t.layer];
                for (int e = t.e0; e < t.e1; ++e)
                    for (int h = 0; h < 2; ++h) pack_half(d, s.gate_of(e), s.up_of(e), s.down_of(e), h, a.base(h) + ((uint64_t) t.layer * (uint64_t) d.n_expert + (uint64_t) e) * d.half_bytes());
                if (remaining[(size_t) t.layer].fetch_sub(1) == 1 && opt.layer_done) opt.layer_done(t.layer);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lk(err_mu);
            if (!first_error) first_error = std::current_exception();
            failed.store(true);
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(work);
    work();
    for (std::thread& th : pool) th.join();
    if (first_error) std::rethrow_exception(first_error);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    char line[256];
    std::snprintf(line, sizeof line, "expert arena: %d layers x %d experts packed into halves with %d thread(s) in %.1f s (%.2f GiB per socket)", d.n_layer, d.n_expert, threads, sec,
                  (double) bytes / (1ull << 30));
    log(line);
    return a;
}

// ================================================================================================ expert lists
std::vector<ExpertId> static_fill_by_index(int n_layer, int n_expert, int n_slots) {
    std::vector<ExpertId> out;
    if (n_layer <= 0 || n_expert <= 0 || n_slots <= 0) return out;
    const int64_t total = std::min<int64_t>(n_slots, (int64_t) n_layer * n_expert);
    const int share = (int) (total / n_layer), extra = (int) (total % n_layer);
    out.reserve((size_t) total);
    for (int l = 0; l < n_layer; ++l) {
        const int n = share + (l < extra ? 1 : 0);
        for (int e = 0; e < n; ++e) out.push_back({l, e});
    }
    return out;
}

namespace {
/// "7", "3-9" or "*" -> [lo, hi] within [0, limit); false (with `err`) for anything else.
bool parse_range(const std::string& s, int limit, int& lo, int& hi, std::string& err) {
    if (s == "*") {
        lo = 0;
        hi = limit - 1;
        return true;
    }
    const auto num = [&](const std::string& t, int& v) {
        if (t.empty() || t.size() > 9 || t.find_first_not_of("0123456789") != std::string::npos) return false;
        v = std::stoi(t);
        return true;
    };
    const size_t dash = s.find('-');
    if (dash == std::string::npos) {
        if (!num(s, lo)) { err = "`" + s + "` is not a number, a range or *"; return false; }
        hi = lo;
    } else if (!num(s.substr(0, dash), lo) || !num(s.substr(dash + 1), hi)) {
        err = "`" + s + "` is not a number, a range or *";
        return false;
    }
    if (lo > hi) { err = "the range `" + s + "` runs backwards"; return false; }
    if (hi >= limit) { err = "`" + s + "` is outside 0.." + std::to_string(limit - 1); return false; }
    return true;
}
}  // namespace

std::vector<ExpertId> parse_expert_list(const std::string& spec, int n_layer, int n_expert) {
    std::vector<ExpertId> out;
    std::set<std::pair<int, int>> seen;
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t comma = spec.find(',', pos);
        std::string item = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        pos = comma == std::string::npos ? spec.size() + 1 : comma + 1;
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
        if (item.empty()) {
            if (spec.find_first_not_of(" \t") == std::string::npos) break;     // an empty spec is an empty list
            throw ModelError("expert list `" + spec + "`: an empty item (a stray comma?)");
        }
        const size_t colon = item.find(':');
        if (colon == std::string::npos) throw ModelError("expert list item `" + item + "`: expected LAYER:EXPERT (`3:17`, `3:0-15`, `3:*`, `0-39:0-7`)");
        int l0, l1, e0, e1;
        std::string err;
        if (!parse_range(item.substr(0, colon), n_layer, l0, l1, err)) throw ModelError("expert list item `" + item + "`, layer: " + err);
        if (!parse_range(item.substr(colon + 1), n_expert, e0, e1, err)) throw ModelError("expert list item `" + item + "`, expert: " + err);
        for (int l = l0; l <= l1; ++l)
            for (int e = e0; e <= e1; ++e)
                if (seen.insert({l, e}).second) out.push_back({l, e});
    }
    return out;
}

// ================================================================================================ the GPU cache
GpuExpertCache::GpuExpertCache(ModelDev& dev, const ExpertDims& d, int n_slots) : dev_(&dev), d_(d), n_slots_(n_slots) {
    if (!d.valid()) throw ModelError("GpuExpertCache: the expert shape is not valid");
    if (n_slots < 0) throw ModelError("GpuExpertCache: " + std::to_string(n_slots) + " slots");
    if ((int64_t) n_slots > (int64_t) d.n_layer * d.n_expert)
        throw ModelError("GpuExpertCache: " + std::to_string(n_slots) + " slots for " + std::to_string((int64_t) d.n_layer * d.n_expert) + " experts: a slot would never be used");
    if (n_slots > 0) slots_ = static_cast<uint8_t*>(dev.alloc((size_t) n_slots * d.blob_bytes()));
    try {
        residency_ = static_cast<int32_t*>(dev.alloc((size_t) residency_bytes()));
        dev.fill(residency_, 0xFF, (size_t) residency_bytes());           // every int32 is -1: not resident (never 0, which would mean slot 0)
    } catch (...) {
        release();
        throw;
    }
    res_.assign((size_t) d.n_layer * (size_t) d.n_expert, -1);
    owner_.assign((size_t) n_slots, ExpertId{});
}

GpuExpertCache& GpuExpertCache::operator=(GpuExpertCache&& o) noexcept {
    if (this != &o) {
        release();
        dev_ = o.dev_;
        d_ = o.d_;
        n_slots_ = o.n_slots_;
        slots_ = o.slots_;
        residency_ = o.residency_;
        res_ = std::move(o.res_);
        owner_ = std::move(o.owner_);
        staging_ = o.staging_;
        o.staging_ = nullptr;
        n_resident_ = o.n_resident_;
        o.dev_ = nullptr;
        o.slots_ = nullptr;
        o.residency_ = nullptr;
        o.n_slots_ = 0;
        o.n_resident_ = 0;
    }
    return *this;
}

void GpuExpertCache::release() {
    if (dev_) {
        if (slots_) dev_->release(slots_);
        if (residency_) dev_->release(residency_);
        if (staging_) dev_->release_mapped(staging_);
    }
    slots_ = nullptr;
    residency_ = nullptr;
    res_.clear();
    owner_.clear();
    staging_ = nullptr;
    n_slots_ = 0;
    n_resident_ = 0;
}

void GpuExpertCache::set_owner(ModelDev& dev, int slot, int layer, int expert, bool defer) {
    if (slot < 0 || slot >= n_slots_) throw ModelError("GpuExpertCache: slot " + std::to_string(slot) + " of " + std::to_string(n_slots_));
    if (layer < 0 || layer >= d_.n_layer || expert < 0 || expert >= d_.n_expert)
        throw ModelError("GpuExpertCache: expert " + std::to_string(layer) + ":" + std::to_string(expert) + " is outside the model");
    const auto write = [&](size_t idx, int32_t v) {
        if (!defer) dev.h2d(residency_ + idx, &v, sizeof v);
    };
    const size_t idx = (size_t) layer * (size_t) d_.n_expert + (size_t) expert;
    const int prev = res_[idx];
    const ExpertId now{layer, expert};
    if (owner_[(size_t) slot].layer >= 0)                // the callers evict the slot's previous tenant before its bytes change
        throw ModelError("internal: GpuExpertCache::set_owner on a slot that still has a tenant");
    if (prev >= 0) {                                     // already resident in another slot: that slot is free from now on (one table entry, one slot)
        owner_[(size_t) prev] = ExpertId{};
        --n_resident_;
    }
    ++n_resident_;
    owner_[(size_t) slot] = now;
    res_[idx] = slot;
    write(idx, slot);
}

void GpuExpertCache::fill_from_gguf(ModelDev& dev, int slot, int layer, int expert, const ExpertSlices& s, bool defer_upload) {
    if (!s) throw ModelError("GpuExpertCache: no expert slices for layer " + std::to_string(layer));
    if (slot < 0 || slot >= n_slots_) throw ModelError("GpuExpertCache: slot " + std::to_string(slot) + " of " + std::to_string(n_slots_));
    if (expert < 0 || expert >= d_.n_expert) throw ModelError("GpuExpertCache: expert " + std::to_string(expert) + " of " + std::to_string(d_.n_expert));
    uint8_t* dst = slot_ptr(slot);
    // the slot's tenant is gone as soon as its bytes change: keep the table honest even if an upload throws half way
    if (owner_[(size_t) slot].layer >= 0) evict(dev, slot, defer_upload);
    dev.h2d(dst, s.gate_of(expert), (size_t) d_.gate_bytes());
    dev.h2d(dst + d_.gate_bytes(), s.up_of(expert), (size_t) d_.gate_bytes());
    dev.h2d(dst + 2 * d_.gate_bytes(), s.down_of(expert), (size_t) d_.down_bytes());
    set_owner(dev, slot, layer, expert, defer_upload);
}

void GpuExpertCache::fill_from_arena(ModelDev& dev, int slot, int layer, int expert, const ExpertArena& a, bool defer_upload) {
    if (!a.built()) throw ModelError("GpuExpertCache: the CPU arena is not built");
    if (slot < 0 || slot >= n_slots_) throw ModelError("GpuExpertCache: slot " + std::to_string(slot) + " of " + std::to_string(n_slots_));
    if (layer < 0 || layer >= d_.n_layer || expert < 0 || expert >= d_.n_expert)
        throw ModelError("GpuExpertCache: expert " + std::to_string(layer) + ":" + std::to_string(expert) + " is outside the model");
    if (!staging_) staging_ = static_cast<uint8_t*>(dev.alloc_mapped((size_t) d_.blob_bytes()));      // pinned on a GPU: the copy below is a plain DMA
    a.assemble_blob(layer, expert, staging_);
    if (owner_[(size_t) slot].layer >= 0) evict(dev, slot, defer_upload);
    dev.h2d(slot_ptr(slot), staging_, (size_t) d_.blob_bytes());
    set_owner(dev, slot, layer, expert, defer_upload);
}

void GpuExpertCache::evict(ModelDev& dev, int slot, bool defer_upload) {
    if (slot < 0 || slot >= n_slots_) throw ModelError("GpuExpertCache: slot " + std::to_string(slot) + " of " + std::to_string(n_slots_));
    const ExpertId old = owner_[(size_t) slot];
    if (old.layer < 0) return;
    const size_t oi = (size_t) old.layer * (size_t) d_.n_expert + (size_t) old.expert;
    res_[oi] = -1;
    if (!defer_upload) {
        const int32_t v = -1;
        dev.h2d(residency_ + oi, &v, sizeof v);
    }
    owner_[(size_t) slot] = ExpertId{};
    --n_resident_;
}

void GpuExpertCache::upload_residency(ModelDev& dev) {
    if (residency_ && !res_.empty()) dev.h2d(residency_, res_.data(), res_.size() * sizeof(int32_t));
}

}  // namespace strata::ds41::model
