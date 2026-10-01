// src/ds41/model/model_sparse_test.cpp - DS1-A: the loader on the REAL model's files, with the data section a hole.
//
//   ds41_model_sparse_test <fixture dir>     reads <dir>/real/*.gguf and <dir>/real_sparse.txt (src/ds41/model/model_fixture.py --sparse-real)
//
// The fixture writes the 12 shards of mxxm-t/DeepSeek-V4.1-Flash-GGUF as GGUF files with the real metadata (the arrays the saved headers only summarise are
// synthesised), the real 1,006-tensor table with the real offsets, and a data section of the real size that was never written: 403.5 GB of apparent size,
// 3.4 MB on disk, every byte reads as zero.  So everything that depends on SIZES is exercised at the real scale without the weights:
//   * a header parse with 129,280 token strings, offsets and sizes far beyond 32 bits, a 12-shard split, a 403 GB mapping;
//   * the loader's own geometry / tensor validation at RealGeom (Ds41Model<RealGeom>), the memory plan it prints;
//   * the dense upload: every dense tensor goes up in <= 64 MiB pieces that tile the tensor exactly, from the right place in the right shard, and the
//     resident memory stays one piece (the file pages are dropped as they go) instead of the 8.9 GB of dense weights;
//   * token_embd / the Engram tables / the expert slices are left in the mapping: their last rows and last experts are addressable and readable.
// The "device" records instead of storing (a ledger over untouched virtual memory), so the machine needs no 9 GB, no 21 GB cache and no 269 GB arena.
#if !defined(__linux__)
#include <cstdio>
int main() {
    std::printf("SKIP: the sparse-file test needs Linux (mmap, /proc)\n");
    return 0;
}
#else
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/model/model.hpp"

using namespace strata::ds41;
using namespace strata::ds41::model;
namespace fs = std::filesystem;

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

uint64_t rss_bytes() {
    std::ifstream f("/proc/self/statm");
    uint64_t size = 0, res = 0;
    f >> size >> res;
    return res * (uint64_t) sysconf(_SC_PAGESIZE);
}

/// A Dev that stores nothing: allocations are untouched virtual memory (MAP_NORESERVE), copies are recorded and READ (one byte per page of the source, as a
/// real copy would fault it in), the peak resident size is sampled at every copy.
class LedgerDev final : public cuda::Dev {
public:
    struct Piece {
        const uint8_t* dst;
        const uint8_t* src;
        size_t n;
    };
    std::vector<Piece> pieces;
    std::map<void*, size_t> blocks;
    uint64_t max_rss = 0;
    uint64_t sink = 0;

    void* alloc(size_t bytes) override {
        void* p = mmap(nullptr, bytes ? bytes : 16, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) throw ModelError("LedgerDev: cannot reserve " + std::to_string(bytes) + " bytes of address space");
        blocks[p] = bytes ? bytes : 16;
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
        max_rss = std::max(max_rss, rss_bytes());                    // sampled BEFORE this copy touches its pages: what the previous pieces left behind
        pieces.push_back({static_cast<const uint8_t*>(dst), static_cast<const uint8_t*>(src), n});
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < n; i += 4096) sink += s[i];
        if (n) sink += s[n - 1];
    }
    void d2h(void* dst, const void* src, size_t n) override {
        if (n > (1u << 20)) throw ModelError("LedgerDev: d2h of a big region");
        std::memcpy(dst, src, n);
    }
    void fill(void* p, int byte, size_t n) override {
        if (n <= (1u << 20)) std::memset(p, byte, n);                // the residency table; nothing else is filled
    }
    cuda::Stream stream() override { return nullptr; }
    void sync() override {}
    double time_us(const std::function<void()>& fn, int) override {
        fn();
        return 0;
    }
    bool is_emulation() const override { return true; }
};

std::vector<std::string> words(const std::string& l) {
    std::istringstream is(l);
    std::vector<std::string> w;
    std::string x;
    while (is >> x) w.push_back(x);
    return w;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: ds41_model_sparse_test <fixture dir>\n");
        return 2;
    }
    const fs::path dir = argv[1];
    std::string first;
    if (!fs::exists(dir / "real")) {
        std::fprintf(stderr, "no %s/real: run model_fixture.py with --sparse-real\n", argv[1]);
        return 2;
    }
    for (const auto& e : fs::directory_iterator(dir / "real"))
        if (e.path().filename().string().find("-00001-of-") != std::string::npos) first = e.path().string();
    if (first.empty()) {
        std::fprintf(stderr, "no shard 1 under %s/real\n", argv[1]);
        return 2;
    }
    struct PyTensor {
        int shard;
        uint64_t off, nbytes;
        std::string type;
    };
    std::map<std::string, PyTensor> py;
    struct PyShard {
        std::string name;
        uint64_t size, data_start;
    };
    std::vector<PyShard> pyshards;
    {
        std::ifstream f(dir / "real_sparse.txt");
        std::string line;
        while (std::getline(f, line)) {
            const auto w = words(line);
            if (w.empty()) continue;
            if (w[0] == "#shard") pyshards.push_back({w[2], std::stoull(w[3]), std::stoull(w[4])});
            else py[w[0]] = {std::stoi(w[1]), std::stoull(w[2]), std::stoull(w[3]), w[4]};
        }
    }
    CHECK(py.size() == 1006 && pyshards.size() == 12);

    // ---- open the 12 shards: nothing but headers is read
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_ptr<GgufSet> g = GgufSet::open(first);
    const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CHECK(g->n_shards() == 12 && g->tensors().size() == 1006);
    uint64_t total = 0;
    for (size_t i = 0; i < g->n_shards(); ++i) {
        total += g->shard(i).size;
        CHECK(fs::path(g->shard(i).path).filename().string() == pyshards[i].name && g->shard(i).size == pyshards[i].size && g->shard(i).data_start == pyshards[i].data_start);
        CHECK(g->shard_meta(i).get_int("split.no") == (int64_t) i && g->shard_meta(i).get_int("split.count") == 12);
    }
    CHECK(total > 400000000000ull);
    for (const TensorLoc& t : g->tensors()) {
        const PyTensor& p = py.at(t.name);
        CHECK(t.shard == p.shard && t.offset == p.off && t.nbytes == p.nbytes && type_name(t.type) == p.type && t.abs_offset == pyshards[(size_t) p.shard].data_start + p.off);
    }
    std::printf("opened %zu shards (%.1f GB apparent, %zu tensors) in %.3f s\n", g->n_shards(), (double) total / 1e9, g->tensors().size(), open_s);
    CHECK(open_s < 10.0);

    // ---- the configuration, with the 129,280-entry arrays
    const Ds41Config cfg = read_config(g->meta());
    require_geometry<RealGeom>(cfg);
    CHECK(cfg.vocab == 129280 && cfg.engram.token_map.size() == 129280 && cfg.engram.pad_compressed() == cfg.engram.token_map[2] && cfg.n_ratio_entries == 43);
    CHECK(g->read_string_array("tokenizer.ggml.tokens").size() == 129280 && g->read_string_array("tokenizer.ggml.tokens")[129279] == "<tok129279>");
    CHECK(cfg.warnings.empty());

    // ---- the load: RealGeom, the plan's cache (allocated, not filled), no arena
    std::vector<std::string> log;
    LoadOptions opt;
    opt.n_slots = -1;
    opt.fill_cache = false;
    opt.build_arena = false;
    opt.log = [&](const std::string& m) { log.push_back(m); };
    LedgerDev dev;
    const uint64_t rss0 = rss_bytes();
    const auto t1 = std::chrono::steady_clock::now();
    std::unique_ptr<Ds41Model<RealGeom>> m = Ds41Model<RealGeom>::load(dev, first, opt);
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    const MemoryPlan& plan = m->plan();
    std::printf("loaded in %.2f s; resident memory %.0f MB before, peak during the upload %.0f MB, %.0f MB after\n", load_s, (double) rss0 / 1e6, (double) dev.max_rss / 1e6, (double) rss_bytes() / 1e6);
    CHECK(m->warnings().empty());
    for (const std::string& w : m->warnings()) std::fprintf(stderr, "  warning: %s\n", w.c_str());
    CHECK(plan.cache_slots_fit > 1100 && plan.cache_slots_fit < 1200 && m->cache.n_slots() == plan.cache_slots_fit && m->cache.n_resident() == 0);
    CHECK(m->cache.device_bytes() == (uint64_t) m->cache.n_slots() * kBlobBytes + 40ull * 384 * 4);
    CHECK(m->weights.device_bytes() >= plan.gpu_dense && m->weights.device_bytes() < plan.gpu_dense + 1000 * 256);      // padding to 256-byte boundaries only
    CHECK(plan.gpu_dense == 8944456128ull && plan.ram_experts == 288777830400ull && plan.ram_engram == 104451107600ull);
    bool said = false;
    for (const std::string& l : log) said = said || (l.find("Memory plan:") != std::string::npos && l.find("1121 cache slots") != std::string::npos);
    CHECK(said);

    // ---- the dense upload: every dense tensor is tiled by pieces of <= 64 MiB, in order, from its own place in the mapping
    const std::vector<TensorSpec> specs = expected_tensors(cfg);
    std::map<const uint8_t*, std::vector<const LedgerDev::Piece*>> by_dst;       // piece start in the device block -> piece
    for (const LedgerDev::Piece& p : dev.pieces) by_dst[p.dst].push_back(&p);
    uint64_t covered = 0, n_dense = 0, biggest = 0;
    size_t n_pieces_used = 0;
    for (const TensorSpec& s : specs) {
        const TensorLoc& loc = m->directory().at(s.name);
        const DevTensor* dt = s.lt == LT::OutputNorm ? &m->weights.output_norm : s.lt == LT::Output ? &m->weights.head : s.layer >= 0 ? m->weights.at(s.layer).slot(s.lt) : nullptr;
        if (!dt) continue;
        ++n_dense;
        CHECK((bool) *dt && dt->nbytes == loc.nbytes && reinterpret_cast<uintptr_t>(dt->p) % 256 == 0);
        uint64_t pos = 0;
        while (pos < loc.nbytes) {
            const auto it = by_dst.find(dt->p + pos);
            if (it == by_dst.end() || it->second.size() != 1) {
                CHECK(!"a hole or a double upload in a dense tensor");
                std::fprintf(stderr, "  tensor %s at +%llu\n", s.name.c_str(), (unsigned long long) pos);
                break;
            }
            const LedgerDev::Piece& pc = *it->second[0];
            CHECK(pc.src == m->gguf().data(loc) + pos && pc.n <= (64u << 20) && pc.n > 0 && pos + pc.n <= loc.nbytes);
            pos += pc.n;
            ++n_pieces_used;
            biggest = std::max<uint64_t>(biggest, pc.n);
        }
        covered += loc.nbytes;
    }
    CHECK(n_dense > 700);                                                        // every layer's dense tensors, the head and the output norm
    CHECK(covered == plan.gpu_dense && n_pieces_used == dev.pieces.size());      // nothing else was uploaded: not an expert, not an Engram table, not token_embd
    CHECK(biggest == (64u << 20));                                               // the head (1.3 GB) really went up in 64 MiB pieces
    CHECK(dev.max_rss < rss0 + (400ull << 20));                                  // one piece resident at a time, not 8.9 GB

    // ---- what stayed in the mapping, at the far end of each
    const Ds41Weights<RealGeom>& w = m->weights;
    CHECK(w.token_embd.rows() == 129280 && w.token_embd.row_bytes() == 10240 && w.token_embd.type == GgmlType::BF16);
    std::vector<float> row(5120, 1.0f);
    w.token_row_f32(129279, row.data());
    CHECK(row[0] == 0.0f && row[5119] == 0.0f);
    bool refused = false;
    try {
        (void) w.token_row(129280);
    } catch (const ModelError&) {
        refused = true;
    }
    CHECK(refused);
    for (int s = 0; s < 2; ++s) {
        const LayerWeights& lw = w.at(cfg.engram.layers[(size_t) s]);
        const int64_t rows = cfg.engram.num_embeddings[(size_t) s];
        CHECK(lw.eng_table.rows() == rows && lw.eng_table.row_bytes() == 136 && lw.eng_table.nbytes == (uint64_t) rows * 136);
        const uint8_t* last = lw.eng_table.row(rows - 1);
        float vals[256];
        dequantize(GgmlType::MXFP4, last, 256, vals);
        CHECK(last == lw.eng_table.p + (uint64_t) (rows - 1) * 136 && vals[0] == 0.0f && vals[255] == 0.0f);
        const TensorLoc& loc = m->directory().at("blk." + std::to_string(cfg.engram.layers[(size_t) s]) + ".engram_embed.weight");
        CHECK(lw.eng_table.p == m->gguf().data(loc) && last + 136 == m->gguf().data(loc) + loc.nbytes);
    }
    uint64_t sum = 0;
    for (int l = 0; l < 40; ++l) {
        const ExpertSlices& ex = w.at(l).experts;
        const uint8_t* g_last = ex.gate_of(383) + kGateBytes - 1;
        const uint8_t* u_last = ex.up_of(383) + kGateBytes - 1;
        const uint8_t* d_last = ex.down_of(383) + kDownBytes - 1;
        sum += *g_last + *u_last + *d_last;
        CHECK(ex.d.blob_bytes() == kBlobBytes && d_last + 1 == m->gguf().data(*ex.loc_down) + ex.loc_down->nbytes);
        CHECK(ex.gate_of(1) == ex.gate + kGateBytes && ex.down_of(383) == ex.down + 383 * kDownBytes);
    }
    CHECK(sum == 0);                                                             // holes read as zeros

    // ---- everything given back
    m.reset();
    CHECK(dev.blocks.empty());
    std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
#endif
