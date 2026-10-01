// include/strata/ds41/geom.hpp - DS-1: the model geometry as a compile-time POLICY, so one source runs two models.
//
// Every DS-1 kernel and host module is a template over a geometry G.  Two exist:
//   RealGeom  DeepSeek-V4.1-Flash as released (mxxm-t/DeepSeek-V4.1-Flash-GGUF; docs/deepseek/RESEARCH.md sections 1 and 10).
//             The engine binary instantiates only this one.
//   MiniGeom  the defaults of tools/ds41/make_mini_gguf.py (a tiny file with the real tensor names, types and metadata keys),
//             instantiated by the tests: with -DDS41_EMU the whole forward pass runs on the CPU through the emulator and is
//             compared with the NumPy oracle (ref/ds41) on the same GGUF.
// The loader reads the GGUF metadata into a runtime Ds41Config and REFUSES a file whose shapes differ from its G (as the Qwen
// path refuses rather than mis-indexes).  Layer ROLES (SWA / Full / Reuse / Reindex, Engram layers, sources) are not here: they are
// control flow, read from the metadata at run time (docs/deepseek/DS1.md, "Layer roles").
// geometry.hpp keeps the byte layouts of one routed expert (DS-C / DS-D); RealGeom's values must agree with it (static_asserts below).
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/ds41/geometry.hpp"

namespace strata::ds41 {

struct RealGeom {
    static constexpr const char* kName = "real";
    // trunk
    static constexpr int kLayers = 40;
    static constexpr int kHidden = 5120;
    static constexpr int kVocab = 129280;
    static constexpr int kHc = 4;                    // mHC residual copies
    static constexpr int kHcMixes = (2 + kHc) * kHc; // 24 = pre[4] + post[4] + comb[4x4]
    static constexpr int kHcIters = 20;              // Sinkhorn iterations
    // MoE
    static constexpr int kExperts = 384;
    static constexpr int kTopK = 6;
    static constexpr int kFF = 2304;                 // routed and shared expert intermediate
    // attention (MQA: kHeads query heads, one K == V head)
    static constexpr int kHeads = 64;
    static constexpr int kHeadDim = 512;
    static constexpr int kRopeDim = 64;              // the LAST kRopeDim channels of a head are rotated (adjacent pairs)
    static constexpr int kQLora = 1280;              // wq_a output / q_norm width
    static constexpr int kOGroups = 8;               // wo_a: block-diagonal, kOGroups x (kHeads*kHeadDim/kOGroups -> kOLora)
    static constexpr int kOLora = 1024;
    static constexpr int kWindow = 128;              // sliding-window KV ring
    // indexer / candidate pool
    static constexpr int kIdxHeads = 32;
    static constexpr int kIdxDim = 128;
    static constexpr int kIdxTopK = 512;
    static constexpr int kCandBlock = 8;             // candidate-pool block size (compressed positions)
    static constexpr int kCandTopBlocks = 2048;
    // Engram
    static constexpr int kEngramHeads = 8;
    static constexpr int kEngramHeadDim = 256;       // values per table row (MXFP4: kEngramHeadDim / 32 blocks of 17 B)
    static constexpr int kEngramNgram = 4;           // max n-gram size; orders 2..4 -> kEngramNgram - 1 orders
    static constexpr int kEngramRows = (kEngramNgram - 1) * kEngramHeads;   // 24 rows per Engram layer per token
};

struct MiniGeom {
    static constexpr const char* kName = "mini";
    static constexpr int kLayers = 8;
    static constexpr int kHidden = 256;
    static constexpr int kVocab = 512;
    static constexpr int kHc = 4;
    static constexpr int kHcMixes = (2 + kHc) * kHc;
    static constexpr int kHcIters = 20;
    static constexpr int kExperts = 16;
    static constexpr int kTopK = 2;
    static constexpr int kFF = 256;
    static constexpr int kHeads = 4;
    static constexpr int kHeadDim = 64;
    static constexpr int kRopeDim = 16;
    static constexpr int kQLora = 64;
    static constexpr int kOGroups = 2;
    static constexpr int kOLora = 32;
    static constexpr int kWindow = 16;
    static constexpr int kIdxHeads = 4;
    static constexpr int kIdxDim = 32;
    static constexpr int kIdxTopK = 8;
    static constexpr int kCandBlock = 4;
    static constexpr int kCandTopBlocks = 4;
    static constexpr int kEngramHeads = 2;
    static constexpr int kEngramHeadDim = 64;
    static constexpr int kEngramNgram = 4;
    static constexpr int kEngramRows = (kEngramNgram - 1) * kEngramHeads;
};

// ---- derived quantities (the same formulas for both) ----
template <class G> struct Derived {
    static constexpr int kQ = G::kHeads * G::kHeadDim;                       // 32,768: wq_b output, attention output
    static constexpr int kOGroupIn = kQ / G::kOGroups;                       // 4,096: wo_a input per group
    static constexpr int kOMid = G::kOGroups * G::kOLora;                    // 8,192: wo_a output = wo_b input
    static constexpr int kHcFlat = G::kHc * G::kHidden;                      // 20,480: hc_fn input
    static constexpr int kEngramIn = G::kEngramRows * G::kEngramHeadDim;     // 6,144: engram wkv input
    static constexpr int kEngramOut = (G::kHc + 1) * G::kHidden;             // 25,600: key per copy + value
    static constexpr int kEngramRowBytes = G::kEngramHeadDim / kQK * kBlockBytes;   // 136 B (MXFP4)
    static constexpr int kIdxQ = G::kIdxHeads * G::kIdxDim;                  // 4,096: indexer wq_b output
    // one routed expert in GGML MXFP4, [gate][up][down] (geometry.hpp's blob, at this G's shape)
    static constexpr size_t kExpertGateBytes = (size_t) G::kFF * (G::kHidden / kQK) * kBlockBytes;
    static constexpr size_t kExpertDownBytes = (size_t) G::kHidden * (G::kFF / kQK) * kBlockBytes;
    static constexpr size_t kExpertBlobBytes = 2 * kExpertGateBytes + kExpertDownBytes;
};

static_assert(RealGeom::kLayers == kLayers && RealGeom::kHidden == kHidden && RealGeom::kExperts == kExperts &&
                  RealGeom::kTopK == kTopK && RealGeom::kFF == kFF,
              "RealGeom agrees with geometry.hpp");
static_assert(Derived<RealGeom>::kExpertBlobBytes == kBlobBytes, "RealGeom's expert blob is geometry.hpp's");
static_assert(Derived<RealGeom>::kEngramIn == 6144 && Derived<RealGeom>::kEngramOut == 25600 &&
                  Derived<RealGeom>::kEngramRowBytes == 136 && Derived<RealGeom>::kOMid == 8192,
              "RealGeom matches the GGUF's tensor shapes (RESEARCH.md section 10)");
// the constraints the kernels rely on, checked for both
template <class G> constexpr bool geom_ok() {
    return G::kHidden % 128 == 0 && G::kFF % 256 == 0 && G::kHeadDim % 32 == 0 && G::kRopeDim % 2 == 0 &&
           G::kRopeDim <= G::kHeadDim && G::kQLora % 32 == 0 && Derived<G>::kOGroupIn % 32 == 0 &&
           Derived<G>::kOMid % 32 == 0 && G::kEngramHeadDim % 32 == 0 && G::kIdxDim % 32 == 0 &&
           G::kTopK <= G::kExperts && G::kHcMixes == (2 + G::kHc) * G::kHc;
}
static_assert(geom_ok<RealGeom>() && geom_ok<MiniGeom>(), "kernel constraints");

}  // namespace strata::ds41
