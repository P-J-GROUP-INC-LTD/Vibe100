// include/strata/ds41/model/tensors.hpp - DS1-A: the tensors of the deepseek41 GGUF, as a table, and the validation of a file against it.
//
// `expected_tensors(cfg)` is the contract of mxxm-t/DeepSeek-V4.1-Flash-GGUF (docs/deepseek/RESEARCH.md section 10) at the file's own
// dimensions: every tensor's name, ggml type and ggml dims, per layer role (a FULL layer owns a compressor and an indexer compressor,
// FULL and REINDEX layers an indexer, Engram layers the Engram projections and table).  It mirrors tools/ds41/ds41_spec.py
// `tensor_specs` and ref/ds41/weights.py `expected_shape`, and the loader uploads exactly the tensors this table lists, into the slot
// named by `LT`, so the validation and the upload cannot disagree about a name.  `validate_tensors` REFUSES a file with a missing,
// mistyped, mis-shaped or unexpected tensor, naming it, what was found and what was expected.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/ds41/model/config.hpp"
#include "strata/ds41/model/gguf.hpp"

namespace strata::ds41::model {

/// Memory-plan group of a tensor (tools/ds41/ds41_spec.py TSpec.group).
enum class Group { Embd, Norm, Attn, Compressor, Indexer, Router, Expert, Shared, Mhc, EngramEmbed, EngramProj, Head, kCount };
const char* group_name(Group g);

/// Where a tensor goes.  The first block are per-layer DEVICE tensors (a LayerWeights field each), then the host-mapped per-layer ones,
/// then the global ones.
enum class LT : int {
    AttnNorm, FfnNorm,
    HcAttnFn, HcAttnBase, HcAttnScale, HcFfnFn, HcFfnBase, HcFfnScale,
    WqA, QNorm, WqB, Wkv, KvNorm, Sinks, WoA, WoB,
    CompKv, CompNorm, CompGate,
    IdxQB, IdxProj, IdxCompKv, IdxCompNorm,
    Gate, GateBias,
    ShGate, ShUp, ShDown,
    EngQ, EngK, EngWkv,
    kDeviceLayerCount,                                   // number of per-layer device slots
    EngEmbed = kDeviceLayerCount, ExpGate, ExpUp, ExpDown,   // per layer, left in the mmapped file
    TokenEmbd, OutputNorm, Output,                       // global (layer == -1): token_embd stays in the file, the other two go to the device
};

struct TensorSpec {
    std::string name;                  ///< as the GGUF spells it: "blk.2.attn_q_a.weight", "token_embd.weight"
    GgmlType type = GgmlType::F32;
    int n_dims = 1;
    uint64_t ne[3] = {1, 1, 1};        ///< ggml order
    Group group = Group::Norm;
    int layer = -1;                    ///< -1 for the global tensors
    LT lt = LT::AttnNorm;
    uint64_t nbytes() const { return tensor_nbytes(type, ne, n_dims); }
    std::string dims_text() const;
};

/// Every tensor the file must have (1,006 for the real model), in layer order: token_embd, the layers, output_norm, output.
std::vector<TensorSpec> expected_tensors(const Ds41Config& c);

/// The file spells nothing but the canonical names of the mxxm-t GGUF; this one is accepted and ignored (vision bias).
bool is_ignored_tensor(const std::string& name);

struct ValidateOptions {
    bool allow_unexpected = false;     ///< an unexpected tensor is a warning, not an error
};

/// Checks `dir` against `expected_tensors(c)`: presence, ggml type, dims, and (for files) that no tensor of the model is of an unknown
/// size.  Adds every problem to `f`; fills `unexpected_ignored` with the count of accepted-and-ignored tensors.
void validate_tensors(const Ds41Config& c, const TensorDir& dir, Findings& f, const ValidateOptions& opt = {}, size_t* ignored = nullptr);

/// Bytes per group of a spec list (the memory plan's input) and the total.
struct ByteTally {
    uint64_t group[(int) Group::kCount] = {};
    uint64_t total = 0;
    uint64_t expert_bytes_each = 0;    ///< one routed expert (gate + up + down slices)
    uint64_t dense_device = 0;         ///< everything the loader puts on the device: all but Embd / Expert / EngramEmbed
    int n_tensors = 0;
};
ByteTally tally(const std::vector<TensorSpec>& specs, const Ds41Config& c);

}  // namespace strata::ds41::model
