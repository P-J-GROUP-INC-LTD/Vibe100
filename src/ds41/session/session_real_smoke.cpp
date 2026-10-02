// src/ds41/session/session_real_smoke.cpp - DS1-E: the session built at the REAL model's shapes, without the weights and without a GPU.
//
//   ds41_session_real_smoke <fixture dir>     reads <dir>/real/*.gguf (src/ds41/model/model_fixture.py --sparse-real: the real 12-shard file with the real metadata and the real
//                                             1,006-tensor table, every data byte a hole that reads as zero)
//
// What it proves (what a mini-shaped test cannot): that `Ds41Session<RealGeom>` constructs on the real file - every attention weight the real layer-role map needs is present
// (set_weights refuses a FULL layer without its compressor, an indexer layer without its q), the Engram layers are wired to their 52 GB tables, the scratch is sized for T = 1 and
// T = 8, the RoPE tables and the attention caches fit the numbers the memory plan assumes, and everything is given back.  It does NOT run a token (an emulated real-shaped token
// takes hours): the numbers are the mini end-to-end test's.  The "device" is a ledger over untouched virtual memory (as DS1-A's model_sparse_test), so the machine needs no 9 GB.
#if !defined(__linux__)
#include <cstdio>
int main() {
    std::printf("SKIP: needs Linux (mmap)\n");
    return 0;
}
#else
#include <sys/mman.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>

#include "session_impl.hpp"
#include "strata/ds41/geom.hpp"

using namespace strata::ds41;
namespace fs = std::filesystem;

DS41_INSTANTIATE_SESSION(::strata::ds41::RealGeom)

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

/// Allocations are untouched virtual memory; copies are not made (the source pages are read, as a real copy would fault them in).
class LedgerDev final : public cuda::Dev {
public:
    std::map<void*, size_t> blocks;
    uint64_t allocated = 0, sink = 0;
    void* alloc(size_t bytes) override {
        void* p = mmap(nullptr, bytes ? bytes : 16, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("LedgerDev: cannot reserve " + std::to_string(bytes) + " bytes of address space");
        blocks[p] = bytes ? bytes : 16;
        allocated += bytes;
        return p;
    }
    void release(void* p) override {
        if (!p) return;
        munmap(p, blocks.at(p));
        blocks.erase(p);
    }
    void* alloc_mapped(size_t bytes) override { return alloc(bytes); }
    void release_mapped(void* p) override { release(p); }
    void h2d(void* dst, const void* src, size_t n) override {
        if (n <= (1u << 20)) std::memcpy(dst, src, n);                    // small uploads (tables, states) land; the weights' 64 MiB pieces are only read
        else for (size_t i = 0; i < n; i += 4096) sink += static_cast<const uint8_t*>(src)[i];
    }
    void d2h(void* dst, const void* src, size_t n) override { std::memcpy(dst, src, std::min<size_t>(n, 1u << 20)); }
    void fill(void* p, int byte, size_t n) override {
        if (n <= (1u << 20)) std::memset(p, byte, n);
    }
    cuda::Stream stream() override { return nullptr; }
    void sync() override {}
    double time_us(const std::function<void()>& fn, int) override {
        fn();
        return 0;
    }
    bool is_emulation() const override { return true; }
};
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ds41_session_real_smoke <fixture dir>\n");
        return 2;
    }
    const fs::path dir = argv[1];
    std::string first;
    if (fs::exists(dir / "real"))
        for (const auto& e : fs::directory_iterator(dir / "real"))
            if (e.path().filename().string().find("-00001-of-") != std::string::npos) first = e.path().string();
    if (first.empty()) {
        std::printf("SKIPPED: no %s/real/*-00001-of-*.gguf (model_fixture.py --sparse-real)\n", argv[1]);
        return 2;
    }
    try {
        LedgerDev dev;
        model::LoadOptions lo;
        lo.n_slots = 0;
        lo.fill_cache = false;
        lo.build_arena = false;
        lo.log = [](const std::string&) {};
        std::unique_ptr<model::Ds41Model<RealGeom>> m = model::Ds41Model<RealGeom>::load(dev, first, lo);
        const model::Ds41Config& cfg = m->config();
        int n_swa = 0, n_full = 0, n_reuse = 0, n_reindex = 0;
        for (const model::LayerInfo& li : cfg.layers) {
            n_swa += li.role == model::Role::SWA;
            n_full += li.role == model::Role::FULL;
            n_reuse += li.role == model::Role::REUSE;
            n_reindex += li.role == model::Role::REINDEX;
        }
        std::printf("real file: %d layers: %d SWA, %d FULL, %d REUSE, %d REINDEX; %d Engram layers; dense weights %.2f GiB\n", cfg.n_layer, n_swa, n_full, n_reuse, n_reindex, cfg.n_engram(),
                    (double) m->weights.device_bytes() / (1ull << 30));
        CHECK(cfg.n_layer == 40 && n_full > 0 && n_swa > 0 && cfg.n_engram() == 2);
        for (int window : {1, 8}) {
            for (int ctx : {256, 131072}) {
                session::SessionOptions so;
                so.max_context = ctx;
                so.max_window = window;
                so.use_cpu_pool = false;
                const size_t blocks_before = dev.blocks.size();
                {
                    session::Ds41Session<RealGeom> s(dev, *m, so);
                    const double gib = (double) s.device_bytes() / (1ull << 30);
                    std::printf("session: window %d, max_context %6d: %.3f GiB of device memory\n", window, ctx, gib);
                    CHECK(s.position() == 0 && s.device_bytes() > 0 && s.options().max_context == ctx);
                    CHECK(!s.describe().empty());
                    if (ctx == 131072) CHECK(gib < 2.5);                      // the plan reserves 3 GiB for everything but the dense weights and the cache, and 6,400 B / token of KV
                }
                CHECK(dev.blocks.size() == blocks_before);                    // everything the session allocated was given back (the weights' blocks stay)
            }
        }
        // a second construction after the first ones were destroyed: nothing leaked into the ledger
        const size_t blocks_now = dev.blocks.size();
        { session::SessionOptions so; so.max_context = 128; session::Ds41Session<RealGeom> s(dev, *m, so); }
        CHECK(dev.blocks.size() == blocks_now);
        // refusals
        bool threw = false;
        try { session::SessionOptions so; so.max_window = 9; session::Ds41Session<RealGeom> s(dev, *m, so); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        CHECK(dev.blocks.size() == blocks_now);
        m.reset();
        CHECK(dev.blocks.empty());
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
#endif
