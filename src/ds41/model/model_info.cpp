// src/ds41/model/model_info.cpp - DS1-A: `ds41_model_info <any shard of the GGUF>`: what the loader would make of a file, from the headers alone.
//
// Opens every shard (mmap, headers only: nothing of the 411 GB is read), prints the configuration with the layer-role map, checks the file against both
// geometries this build knows (real / mini) and every tensor against the deepseek41 contract, and prints the memory plan for the box
// (a V100 32 GB and 384 GB of RAM unless told otherwise).  Exit code 0: the file is the model of the named geometry; 1: it is refused (and why).
//   --geometry real|mini|auto   (default auto: real, else mini)      --vram-gib N   --ram-gib N   --slots N   --ctx N
#include <cstdio>
#include <cstring>
#include <string>

#include "strata/ds41/model/model.hpp"

using namespace strata::ds41;
using namespace strata::ds41::model;

int main(int argc, char** argv) {
    std::string path, geometry = "auto";
    PlanInputs in;
    int slots = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--geometry") geometry = next();
        else if (a == "--vram-gib") in.vram_total = (uint64_t) (std::stod(next()) * (double) (1ull << 30));
        else if (a == "--ram-gib") in.ram_total = (uint64_t) (std::stod(next()) * (double) (1ull << 30));
        else if (a == "--slots") slots = std::stoi(next());
        else if (a == "--ctx") in.ctx_tokens = std::stoull(next());
        else if (a == "-h" || a == "--help") {
            std::printf("usage: ds41_model_info <GGUF shard> [--geometry real|mini|auto] [--vram-gib N] [--ram-gib N] [--slots N] [--ctx TOKENS]\n");
            return 0;
        } else if (!a.empty() && a[0] != '-') path = a;
        else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (path.empty()) {
        std::fprintf(stderr, "usage: ds41_model_info <GGUF shard> [--geometry real|mini|auto] [--vram-gib N] [--ram-gib N] [--slots N] [--ctx TOKENS]\n");
        return 2;
    }
    try {
        const std::unique_ptr<GgufSet> g = GgufSet::open(path);
        std::printf("%zu shard(s):", g->n_shards());
        uint64_t total = 0;
        for (size_t i = 0; i < g->n_shards(); ++i) total += g->shard(i).size;
        std::printf(" %.2f GB, %zu tensors\n", (double) total / 1e9, g->tensors().size());
        const Ds41Config cfg = read_config(g->meta());
        std::printf("%s", cfg.describe().c_str());

        Findings fr, fm;
        check_geometry<RealGeom>(cfg, fr);
        check_geometry<MiniGeom>(cfg, fm);
        const char* which = geometry == "real" ? "real" : geometry == "mini" ? "mini" : fr.ok() ? "real" : fm.ok() ? "mini" : "real";
        const Findings& fg = std::strcmp(which, "real") == 0 ? fr : fm;
        std::printf("geometry '%s': %s\n", which, fg.ok() ? "the file matches" : "the file does NOT match");
        for (const std::string& e : fg.errors) std::printf("  - %s\n", e.c_str());

        const TensorDir dir = g->directory();
        Findings ft;
        size_t ignored = 0;
        validate_tensors(cfg, dir, ft, {}, &ignored);
        std::printf("tensors: %zu in the file, %s%s\n", dir.size(), ft.ok() ? "all as the contract says" : "PROBLEMS", ignored ? (" (" + std::to_string(ignored) + " ignored)").c_str() : "");
        for (size_t i = 0; i < ft.errors.size() && i < 20; ++i) std::printf("  - %s\n", ft.errors[i].c_str());
        if (ft.errors.size() > 20) std::printf("  - ... and %zu more\n", ft.errors.size() - 20);

        if (ft.ok()) {
            const ByteTally t = tally(expected_tensors(cfg), cfg);
            const MemoryPlan first = make_memory_plan(cfg, t, in);
            in.n_slots = slots >= 0 ? slots : first.cache_slots_fit;
            std::printf("%s\n", make_memory_plan(cfg, t, in).text().c_str());
        }
        return fg.ok() && ft.ok() ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "refused: %s\n", e.what());
        return 1;
    }
}
