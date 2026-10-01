// src/ds41/model/plan.cpp - DS1-A: the memory plan (a port of tools/ds41/memplan.py to the loader's own byte tally), printed at load.
#include <algorithm>
#include <cstdarg>
#include <cstdio>

#include "strata/ds41/model/model.hpp"

namespace strata::ds41::model {

namespace {
constexpr double kGiB = 1073741824.0;
double gib(uint64_t b) { return (double) b / kGiB; }
double gib(int64_t b) { return (double) b / kGiB; }

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}
}  // namespace

MemoryPlan make_memory_plan(const Ds41Config& c, const ByteTally& t, const PlanInputs& in) {
    MemoryPlan p;
    p.in = in;
    p.tally = t;
    const uint64_t blob = t.expert_bytes_each;
    const int64_t n_experts_total = (int64_t) c.n_layer * c.n_expert;

    // ---- GPU
    p.gpu_dense = t.dense_device;
    p.gpu_kv = in.ctx_tokens * in.kv_bytes_per_token;
    p.gpu_reserve = in.gpu_reserve;
    const int64_t budget = (int64_t) in.vram_total - (int64_t) p.gpu_dense - (int64_t) p.gpu_kv - (int64_t) p.gpu_reserve;
    p.gpu_cache_budget = budget > 0 ? (uint64_t) budget : 0;
    p.cache_slots_fit = (blob > 0 && budget > 0) ? (int) std::min<int64_t>(n_experts_total, (int64_t) ((uint64_t) budget / blob)) : 0;
    const int slots = (int) std::max<int64_t>(0, std::min<int64_t>(in.n_slots, n_experts_total));
    p.gpu_cache = (uint64_t) slots * blob;
    p.gpu_residency = (uint64_t) n_experts_total * 4;
    p.gpu_fits = p.gpu_dense + p.gpu_kv + p.gpu_reserve + p.gpu_cache + p.gpu_residency <= in.vram_total;
    if (budget <= 0)
        p.warnings.push_back(fmt("GPU: dense weights + KV + reserve (%.2f GiB) leave nothing of %.2f GiB for an expert cache", gib(p.gpu_dense + p.gpu_kv + p.gpu_reserve), gib(in.vram_total)));
    else if (slots > p.cache_slots_fit)
        p.warnings.push_back(fmt("GPU: %d cache slots asked for, the budget holds %d (%.2f GiB of %.2f GiB free): the allocation is likely to fail", slots, p.cache_slots_fit,
                                 gib(p.gpu_cache_budget), gib(in.vram_total)));
    if (in.n_slots > n_experts_total)
        p.warnings.push_back(fmt("GPU: %d cache slots asked for, the model has only %lld experts", in.n_slots, (long long) n_experts_total));

    // ---- RAM
    p.ram_usable = (uint64_t) ((double) in.ram_total * (1.0 - in.kernel_overhead_pct / 100.0));
    p.ram_experts = t.group[(int) Group::Expert];
    const uint64_t nodes = (uint64_t) std::max(1, in.numa_nodes);
    p.ram_experts_per_node = p.ram_experts / nodes;
    p.ram_token_embd = t.group[(int) Group::Embd];
    p.ram_engram = t.group[(int) Group::EngramEmbed];
    const int64_t items = (int64_t) p.ram_experts + (int64_t) in.os_reserve + (int64_t) in.host_buffers + (int64_t) p.ram_token_embd;
    p.ram_page_cache = (int64_t) p.ram_usable - items;
    const int64_t cached = std::max<int64_t>(0, std::min<int64_t>(p.ram_page_cache, (int64_t) p.ram_engram));
    p.engram_cached_fraction = p.ram_engram ? (double) cached / (double) p.ram_engram : 1.0;
    if (p.ram_experts_per_node > in.ram_total / nodes)
        p.warnings.push_back(fmt("RAM: a node's share of the experts (%.1f GiB) exceeds its RAM (%.1f GiB)", gib(p.ram_experts_per_node), gib(in.ram_total / nodes)));
    if (p.ram_page_cache < 0)
        p.warnings.push_back(fmt("RAM: experts + reserves (%.1f GiB) exceed the usable %.1f GiB; the experts do not fit in RAM", gib(items), gib(p.ram_usable)));
    else if (p.ram_engram && p.engram_cached_fraction < 0.9)
        p.warnings.push_back(fmt("RAM: only %.0f %% of the Engram tables fit in the page cache; the rest is read from the SSD on demand", 100.0 * p.engram_cached_fraction));
    return p;
}

std::string MemoryPlan::text() const {
    std::string s;
    const auto line = [&](const std::string& l) { s += l + "\n"; };
    line(fmt("Memory plan: GPU %.2f GiB, RAM %.2f GiB, %d CPU expert half(ves) per socket", gib(in.vram_total), gib(in.ram_total), 2));
    line(fmt("  GPU (%.2f GiB)", gib(in.vram_total)));
    std::string br;
    for (int g = 0; g < (int) Group::kCount; ++g) {
        if (g == (int) Group::Embd || g == (int) Group::Expert || g == (int) Group::EngramEmbed || tally.group[g] == 0) continue;
        br += (br.empty() ? "" : ", ") + std::string(group_name((Group) g)) + fmt(" %.2f", gib(tally.group[g]));
    }
    line(fmt("    dense weights, GPU-resident        %8.2f GiB   (%s)", gib(gpu_dense), br.c_str()));
    line(fmt("    KV cache, %llu tokens x %llu B       %8.2f GiB", (unsigned long long) in.ctx_tokens, (unsigned long long) in.kv_bytes_per_token, gib(gpu_kv)));
    line(fmt("    activations / prompt buffers / CUDA %7.2f GiB   (reserve)", gib(gpu_reserve)));
    line(fmt("    hot-expert cache budget            %8.2f GiB   -> %d experts of %llu B", gib(gpu_cache_budget), cache_slots_fit, (unsigned long long) tally.expert_bytes_each));
    line(fmt("    this load: %d cache slots = %.2f GiB + residency table %.2f MiB   (%s)", in.n_slots, gib(gpu_cache), (double) gpu_residency / 1048576.0,
             gpu_fits ? "fits" : "DOES NOT FIT"));
    line(fmt("  RAM (%.2f GiB, %.1f GiB usable after %g %% kernel overhead)", gib(in.ram_total), gib(ram_usable), in.kernel_overhead_pct));
    line(fmt("    routed experts, all resident        %8.2f GiB   (%.2f per socket: two halves per expert)", gib(ram_experts), gib(ram_experts / (uint64_t) std::max(1, in.numa_nodes))));
    line(fmt("    OS, services                        %8.2f GiB", gib(in.os_reserve)));
    line(fmt("    engine host buffers                 %8.2f GiB", gib(in.host_buffers)));
    line(fmt("    token_embd (host lookup)            %8.2f GiB", gib(ram_token_embd)));
    line(fmt("    left for Engram's page cache        %8.2f GiB   of %.2f GiB of Engram tables -> %.1f %% cached", gib(ram_page_cache), gib(ram_engram), 100.0 * engram_cached_fraction));
    for (const std::string& w : warnings) line("  WARNING: " + w);
    if (!s.empty() && s.back() == '\n') s.pop_back();
    return s;
}

}  // namespace strata::ds41::model
