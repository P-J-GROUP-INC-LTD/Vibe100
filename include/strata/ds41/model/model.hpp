// include/strata/ds41/model/model.hpp - DS1-A: the loaded DeepSeek-V4.1-Flash model: weights on the device, the CPU expert arena, the GPU expert
// cache and the Engram tables.  The public contract of the loader (docs/deepseek/DS1.md section 4, WP DS1-A).
//
//   auto m = Ds41Model<RealGeom>::load(dev, "/models/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf", opts);
//   m->config()                        Ds41Config: every metadata key, the layer roles, the Engram constants       (config.hpp)
//   m->weights.layer[l].wq_a           DevTensor: a device pointer to the tensor's GGUF bytes (Q8_0 blocks verbatim, BF16, F32)
//   m->weights.layer[l].eng_table      HostTensor: the Engram table, MMAPPED from the GGUF (MXFP4 rows of 136 B), 52 GB, never read whole
//   m->weights.token_row(t)            token_embd is left in the mapped file (BF16 rows); the head and the output norm are on the device
//   m->arena.half(h, l, e)             the CPU expert arena: socket h's half of (layer l, expert e), DS-C's layout (CONTRACTS.md), NUMA-bound
//   m->cache.slot_ptr(s) / residency() the GPU expert cache: n_slots blobs [gate][up][down], and the int32 [layer][expert] residency table
//
// Everything is geometry-checked: `load` refuses a file whose metadata or tensors differ from `G` (config.hpp, tensors.hpp), with a message
// that names what it found and what it expects.  Device memory is reached only through `ModelDev` = DS1-G's shared `Dev`
// (include/strata/ds41/cuda/ds41_dev.hpp: `CudaDev` on the V100, `HostDev` under the emulator and in the tests).
//
// Reading is streaming and bounded: shards are mmapped, a dense tensor is uploaded in 64 MiB pieces straight from the mapping, an expert slice
// is copied once into the arena halves (and the pages of the expert tensors are then dropped from this process and the page cache so that
// they do not push the Engram tables out), the Engram tables and token_embd are never read except row by row.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"
#include "strata/ds41/geometry.hpp"
#include "strata/ds41/model/config.hpp"
#include "strata/ds41/model/gguf.hpp"
#include "strata/ds41/model/host_dequant.hpp"
#include "strata/ds41/model/tensors.hpp"
#include "strata/platform/numa.hpp"

namespace strata::ds41::model {

using LogFn = std::function<void(const std::string&)>;

// ================================================================================================ device access
/// The device the loader allocates on and uploads to: DS1-G's shared interface (include/strata/ds41/cuda/ds41_dev.hpp: `CudaDev` on a V100,
/// `HostDev` in the emulator build and the tests).  Only alloc / release / h2d / d2h / fill / is_emulation are used.  Pointers are device pointers
/// (host memory under the emulation).  The Dev must outlive every object that holds a device allocation of it (weights, cache).
using ModelDev = ::strata::ds41::cuda::Dev;

// ================================================================================================ tensors
/// A tensor on the device: the GGUF bytes of the tensor, verbatim.  Q8_0 / MXFP4 / BF16 / F32 as the file has them; rows are `ne0` elements
/// (ggml order: a weight W[out][in] has ne0 = in, ne1 = out), `ne1 * ne2` rows of `row_bytes()` bytes.
struct DevTensor {
    const uint8_t* p = nullptr;
    uint64_t nbytes = 0;
    GgmlType type = GgmlType::F32;
    int64_t ne0 = 0, ne1 = 1, ne2 = 1;
    explicit operator bool() const { return p != nullptr; }
    int64_t rows() const { return ne1 * ne2; }
    uint64_t row_bytes() const { return rows() > 0 ? nbytes / (uint64_t) rows() : 0; }
    const uint8_t* row(int64_t r) const { return p + (uint64_t) r * row_bytes(); }
    /// The device pointer as the type a kernel wrapper takes (`const void*` for Q8_0, `const uint16_t*` for BF16, `const float*` for F32).
    template <class T> const T* as() const { return reinterpret_cast<const T*>(p); }
};

/// A tensor left in the mmapped GGUF (token_embd, the Engram tables, the expert tensors): a HOST pointer, valid while the model lives.
struct HostTensor {
    const uint8_t* p = nullptr;
    uint64_t nbytes = 0;
    GgmlType type = GgmlType::F32;
    int64_t ne0 = 0, ne1 = 1, ne2 = 1;
    explicit operator bool() const { return p != nullptr; }
    int64_t rows() const { return ne1 * ne2; }
    uint64_t row_bytes() const { return rows() > 0 ? nbytes / (uint64_t) rows() : 0; }
    /// Row `r` (bounds-checked: a hash that lands outside its table is a bug to find, not a wild read of a 52 GB mapping).
    const uint8_t* row(int64_t r) const {
        if (r < 0 || r >= rows()) throw ModelError("row " + std::to_string(r) + " is outside a table of " + std::to_string(rows()) + " rows");
        return p + (uint64_t) r * row_bytes();
    }
};

// ================================================================================================ expert layout
/// The byte layout of one routed expert at a given shape (geometry.hpp at RealGeom's; `expert_dims<G>()` for any G).
struct ExpertDims {
    int n_layer = 0, n_expert = 0, hidden = 0, ff = 0;
    static constexpr int kQK = 32, kBlockBytes = 17;
    constexpr bool valid() const { return n_layer > 0 && n_expert > 0 && hidden > 0 && hidden % kQK == 0 && ff > 0 && ff % (2 * kQK) == 0; }
    constexpr uint64_t gate_row_bytes() const { return (uint64_t) (hidden / kQK) * kBlockBytes; }
    constexpr uint64_t down_row_bytes() const { return (uint64_t) (ff / kQK) * kBlockBytes; }
    constexpr uint64_t gate_bytes() const { return (uint64_t) ff * gate_row_bytes(); }        ///< gate (and up) of one expert
    constexpr uint64_t down_bytes() const { return (uint64_t) hidden * down_row_bytes(); }
    constexpr uint64_t blob_bytes() const { return 2 * gate_bytes() + down_bytes(); }           ///< a GPU cache slot: [gate][up][down]
    constexpr int half_ff() const { return ff / 2; }
    constexpr int half_down_blocks() const { return ff / kQK / 2; }
    constexpr uint64_t half_gate_bytes() const { return (uint64_t) half_ff() * gate_row_bytes(); }
    constexpr uint64_t half_down_row_bytes() const { return (uint64_t) half_down_blocks() * kBlockBytes; }
    constexpr uint64_t half_down_bytes() const { return (uint64_t) hidden * half_down_row_bytes(); }
    constexpr uint64_t half_bytes() const { return 2 * half_gate_bytes() + half_down_bytes(); }  ///< a CPU half: [gate rows][up rows][down rows]
    constexpr uint64_t n_halves() const { return (uint64_t) n_layer * (uint64_t) n_expert; }    ///< per socket
};

template <class G> constexpr ExpertDims expert_dims() {
    return ExpertDims{G::kLayers, G::kExperts, G::kHidden, G::kFF};
}
static_assert(expert_dims<RealGeom>().blob_bytes() == kBlobBytes && expert_dims<RealGeom>().half_bytes() == kHalfBytes &&
                  expert_dims<RealGeom>().half_gate_bytes() == kHalfGateBytes && expert_dims<RealGeom>().half_down_bytes() == kHalfDownBytes,
              "ExpertDims at RealGeom is geometry.hpp's layout");
static_assert(expert_dims<RealGeom>().blob_bytes() == Derived<RealGeom>::kExpertBlobBytes && expert_dims<MiniGeom>().blob_bytes() == Derived<MiniGeom>::kExpertBlobBytes,
              "ExpertDims agrees with Derived<G>::kExpertBlobBytes");

/// One layer's routed experts as the GGUF stores them (mapped, host): expert e of gate / up is rows [e * ff, (e + 1) * ff) of
/// `ffn_{gate,up}_exps`, of down rows [e * hidden, (e + 1) * hidden) of `ffn_down_exps` - each slice contiguous.
struct ExpertSlices {
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    const TensorLoc* loc_gate = nullptr;   ///< the tensors' directory entries (to drop their pages once consumed)
    const TensorLoc* loc_up = nullptr;
    const TensorLoc* loc_down = nullptr;
    ExpertDims d;
    const uint8_t* gate_of(int e) const { return gate + (uint64_t) e * d.gate_bytes(); }
    const uint8_t* up_of(int e) const { return up + (uint64_t) e * d.gate_bytes(); }
    const uint8_t* down_of(int e) const { return down + (uint64_t) e * d.down_bytes(); }
    explicit operator bool() const { return gate != nullptr; }
};

/// Half `h` (0 / 1) of one expert from its three GGUF slices, in the CPU layout of CONTRACTS.md: gate rows [h * ff/2 ..), up rows [h * ff/2 ..),
/// then every down row's blocks [h * ff/64 ..).  `out` has d.half_bytes() bytes.  Bit-identical to DS-C's `pack_cpu_half` (a blob's half) at
/// RealGeom and to tools/ds41/expert_layout.py `blob_to_halves` at any shape (both are tests).
void pack_half(const ExpertDims& d, const uint8_t* gate, const uint8_t* up, const uint8_t* down, int h, uint8_t* out);
/// The inverse: the two halves of an expert -> the GPU blob [gate][up][down] (`out`: d.blob_bytes()).
void unpack_halves(const ExpertDims& d, const uint8_t* half0, const uint8_t* half1, uint8_t* out);

// ================================================================================================ per-layer weights
struct LayerWeights {
    int layer = -1;
    // ---- device tensors; a field is null (operator bool false) when the layer's role has no such tensor
    DevTensor attn_norm, ffn_norm;                                   ///< F32 [hidden]
    DevTensor hc_attn_fn, hc_attn_base, hc_attn_scale;               ///< F32: fn (ne0 = hc*hidden, ne1 = hc_mixes), base [hc_mixes], scale [3]
    DevTensor hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    DevTensor wq_a, q_norm, wq_b, wkv, kv_norm, attn_sinks;          ///< Q8_0 / F32: wq_a (hidden -> q_lora), q_norm [q_lora], wq_b (q_lora -> heads*head_dim), wkv (hidden -> head_dim), kv_norm [head_dim], sinks [heads]
    DevTensor wo_a, wo_b;                                            ///< Q8_0: wo_a is grouped (o_groups x (heads*head_dim/o_groups -> o_lora)), stored as ne0 = 4096, ne1 = o_groups*o_lora
    DevTensor comp_kv, comp_norm, comp_gate;                         ///< FULL: compressor kv / gate (BF16 hidden -> head_dim; the gate only at ratio 2) and norm (F32 [head_dim])
    DevTensor idx_q_b, idx_proj;                                     ///< FULL, REINDEX: indexer q (Q8_0 q_lora -> idx_heads*idx_dim) and weights_proj (BF16 hidden -> idx_heads)
    DevTensor idx_comp_kv, idx_comp_norm;                            ///< FULL: index-K from the compressed latent (BF16 head_dim -> idx_dim) and its norm (F32 [idx_dim])
    DevTensor gate, gate_bias;                                       ///< router: BF16 [experts][hidden] and F32 [experts] (the selection bias)
    DevTensor sh_gate, sh_up, sh_down;                               ///< shared expert (Q8_0)
    DevTensor eng_q, eng_k, eng_wkv;                                 ///< Engram layers: BF16 [hc][hidden] x2, Q8_0 (cols * head_dim -> (hc + 1) * hidden)
    // ---- host (mmapped) tensors
    HostTensor eng_table;                                            ///< Engram layers: MXFP4 table, rows of head_dim / 32 * 17 bytes (136 for the real model)
    ExpertSlices experts;                                            ///< the routed experts as the GGUF stores them
    /// The device slot of tensor class `lt` (null for the host-mapped ones).
    DevTensor* slot(LT lt);
    const DevTensor* slot(LT lt) const { return const_cast<LayerWeights*>(this)->slot(lt); }
};

/// The geometry-independent part of the weights (the loader's work is the same at every shape; Ds41Weights<G> adds the G-checked entry).
class WeightsCore {
public:
    WeightsCore() = default;
    ~WeightsCore() { release(); }
    WeightsCore(const WeightsCore&) = delete;
    WeightsCore& operator=(const WeightsCore&) = delete;
    WeightsCore(WeightsCore&& o) noexcept { *this = std::move(o); }
    WeightsCore& operator=(WeightsCore&& o) noexcept;

    std::vector<LayerWeights> layer;
    DevTensor head;                    ///< output.weight, BF16 [vocab][hidden] on the device
    DevTensor output_norm;             ///< F32 [hidden] on the device
    HostTensor token_embd;             ///< BF16 [vocab][hidden], left in the mapped file
    const Ds41Config* cfg = nullptr;

    uint64_t device_bytes() const { return dev_bytes_; }
    /// BF16 bytes of the embedding row of `token` (hidden * 2 bytes, in the mapping).
    const uint8_t* token_row(int token) const { return token_embd.row(token); }
    /// The row dequantised to FP32 (exact).
    void token_row_f32(int token, float* out) const { dequantize(token_embd.type, token_embd.row(token), (uint64_t) token_embd.ne0, out); }

    /// Uploads every dense tensor of `specs` (their LT slots) into ONE device allocation and maps the host tensors.  The files must have been
    /// validated against `specs` (validate_tensors).  `dir` / `gguf` outlive the weights.
    void load(ModelDev& dev, const GgufSet& gguf, const Ds41Config& cfg, const std::vector<TensorSpec>& specs, const TensorDir& dir, const LogFn& log);
    void release();

private:
    ModelDev* dev_ = nullptr;
    void* dev_base_ = nullptr;
    uint64_t dev_bytes_ = 0;
};

/// The weights of a model at geometry G.
template <class G> class Ds41Weights : public WeightsCore {
public:
    static constexpr int kLayers = G::kLayers;
    static constexpr ExpertDims kExpert = expert_dims<G>();
    static_assert(geom_ok<G>(), "the geometry's kernel constraints");
    const LayerWeights& at(int l) const { return layer[(size_t) l]; }
};

// ================================================================================================ the CPU expert arena
struct ArenaOptions {
    int threads = 0;                                   ///< copy threads (0: min(8, hardware threads))
    bool numa = true;                                  ///< bind the two halves to the two NUMA nodes when exactly two exist
    const platform::NumaTopology* topology = nullptr;  ///< use this topology instead of discovering one (tests)
    bool hugepages = true;                             ///< hugetlb, else THP (numa.hpp NumaBuffer); false: 4 KiB pages
    LogFn log;
    std::function<void(int layer)> layer_done;         ///< called (from a worker thread) when every expert of a layer has been copied
};

/// Which NUMA node holds which half.  Exactly two nodes with CPUs -> half 0 on the lower node id, half 1 on the other.  More nodes (sub-NUMA
/// clustering): when the nodes sit on exactly two physical packages, half h goes to the lowest-numbered node of package h (a half then lives in
/// ONE sub-node; the note says so).  Anything else (one node, no topology, a memory-only node, three sockets): not bound, and the note says why.
struct ArenaNodes {
    bool bind = false;
    int node[2] = {-1, -1};            ///< node ids of half 0 / 1 (valid when `bind`)
    std::vector<int> cpus[2];          ///< their CPUs (the prefault threads of a hugetlb mapping run there)
    std::string note;                  ///< one line for the startup log
};
ArenaNodes choose_arena_nodes(const platform::NumaTopology& topo);

/// Every routed expert of every layer as two CPU halves (one buffer per socket): half h of (layer l, expert e) is at
/// `half(h, l, e)`, d.half_bytes() bytes, layout [gate rows][up rows][down rows] (CONTRACTS.md).  Built once from the GGUF slices.
class ExpertArena {
public:
    ExpertArena() = default;
    ExpertArena(ExpertArena&&) noexcept = default;
    ExpertArena& operator=(ExpertArena&&) noexcept = default;
    /// Allocates the two buffers (bound to NUMA nodes 0 / 1 of the topology when it has exactly two, plain memory otherwise) and fills them
    /// from `slices` (one entry per layer) with `threads` workers.  Streaming: nothing is read except the expert slices, once.
    static ExpertArena build(const ExpertDims& d, const std::vector<ExpertSlices>& slices, const ArenaOptions& opt);

    bool built() const { return blk_[0].data != nullptr; }
    const ExpertDims& dims() const { return d_; }
    uint64_t half_bytes() const { return d_.half_bytes(); }
    uint64_t bytes_per_socket() const { return d_.n_halves() * d_.half_bytes(); }
    const uint8_t* half(int h, int layer, int expert) const {
        return blk_[h].data + ((uint64_t) layer * (uint64_t) d_.n_expert + (uint64_t) expert) * d_.half_bytes();
    }
    uint8_t* base(int h) const { return blk_[h].data; }
    /// NUMA node id of half h's buffer (-1: plain memory, not bound), whether the bind took effect, how the buffer is backed.
    int node(int h) const { return blk_[h].node; }
    bool bound(int h) const { return blk_[h].bound; }
    const std::string& note(int h) const { return blk_[h].note; }
    /// The GPU blob of an expert reassembled from its two halves (what a cache promotion uploads): `out` has d.blob_bytes() bytes.
    void assemble_blob(int layer, int expert, uint8_t* out) const { unpack_halves(d_, half(0, layer, expert), half(1, layer, expert), out); }

private:
    struct Block {
        platform::NumaBuffer nb;                       // a mapping bound to a node, or unbound
        std::unique_ptr<uint8_t, void (*)(void*)> plain{nullptr, [](void* p) { std::free(p); }};   // the fallback where mapping failed
        uint8_t* data = nullptr;
        uint64_t bytes = 0;
        int node = -1;
        bool bound = false;
        std::string note;
    };
    ExpertDims d_;
    Block blk_[2];
};

// ================================================================================================ the GPU expert cache
struct ExpertId {
    int layer = -1, expert = -1;
    bool operator==(const ExpertId& o) const { return layer == o.layer && expert == o.expert; }
};

/// DS-1's static initial fill: the first experts by index, an equal share per layer (the first `n_slots % n_layer` layers get one more).
std::vector<ExpertId> static_fill_by_index(int n_layer, int n_expert, int n_slots);
/// "L:E,L:E0-E1,L:*,..." (as for a command line `--cache-experts`) -> experts, duplicates removed, in order.  Throws a ModelError for a
/// malformed or out-of-range item, naming it.
std::vector<ExpertId> parse_expert_list(const std::string& spec, int n_layer, int n_expert);
/// How many whole blobs fit `bytes`.
inline int slots_for_budget(uint64_t bytes, uint64_t blob_bytes) { return blob_bytes ? (int) (bytes / blob_bytes) : 0; }

/// `n_slots` blobs of d.blob_bytes() on the device and the residency table: int32 [layer][expert] = the slot holding the expert, or -1 (all
/// bytes 0xFF at creation, never 0).  A residency value r is a hit only if 0 <= r < n_slots (CONTRACTS.md); the host mirror is kept in step.
class GpuExpertCache {
public:
    GpuExpertCache() = default;
    GpuExpertCache(ModelDev& dev, const ExpertDims& d, int n_slots);
    ~GpuExpertCache() { release(); }
    GpuExpertCache(const GpuExpertCache&) = delete;
    GpuExpertCache& operator=(const GpuExpertCache&) = delete;
    GpuExpertCache(GpuExpertCache&& o) noexcept { *this = std::move(o); }
    GpuExpertCache& operator=(GpuExpertCache&& o) noexcept;
    void release();

    int n_slots() const { return n_slots_; }
    const ExpertDims& dims() const { return d_; }
    uint8_t* slots() const { return slots_; }                          ///< device: n_slots * blob_bytes (null when n_slots == 0)
    uint8_t* slot_ptr(int s) const { return slots_ + (uint64_t) s * d_.blob_bytes(); }
    int32_t* residency() const { return residency_; }                 ///< device: int32 [n_layer][n_expert]
    uint64_t device_bytes() const { return (uint64_t) n_slots_ * d_.blob_bytes() + residency_bytes(); }
    uint64_t residency_bytes() const { return (uint64_t) d_.n_layer * (uint64_t) d_.n_expert * 4; }
    /// Host mirror of the table (what the device holds after the last fill / upload_residency).
    const std::vector<int32_t>& host_residency() const { return res_; }
    int slot_of(int layer, int expert) const { return res_[(size_t) layer * d_.n_expert + expert]; }
    ExpertId owner(int slot) const { return owner_[(size_t) slot]; }   ///< {-1, -1}: the slot is free
    int n_resident() const { return n_resident_; }

    /// Puts expert (layer, expert) into `slot` from its three GGUF slices (three uploads straight from the mapping); the slot's previous
    /// tenant, if any, becomes non-resident.  Updates the device table entry (a 4-byte upload) unless `defer_upload`.
    void fill_from_gguf(ModelDev& dev, int slot, int layer, int expert, const ExpertSlices& s, bool defer_upload = false);
    /// The same from the CPU arena (the two halves reassembled into a host staging blob, one upload): what a promotion does at run time.
    void fill_from_arena(ModelDev& dev, int slot, int layer, int expert, const ExpertArena& a, bool defer_upload = false);
    /// Marks a slot free and its expert non-resident (-1).
    void evict(ModelDev& dev, int slot, bool defer_upload = false);
    /// Uploads the whole host mirror of the table (after a batch of deferred updates).
    void upload_residency(ModelDev& dev);

private:
    void set_owner(ModelDev& dev, int slot, int layer, int expert, bool defer);
    ModelDev* dev_ = nullptr;
    ExpertDims d_;
    int n_slots_ = 0;
    uint8_t* slots_ = nullptr;
    int32_t* residency_ = nullptr;
    std::vector<int32_t> res_;
    std::vector<ExpertId> owner_;
    std::vector<uint8_t> staging_;
    int n_resident_ = 0;
};

// ================================================================================================ the memory plan
struct PlanInputs {
    int n_slots = 0;                                   ///< GPU cache slots asked for (the plan also says how many the budget holds)
    int numa_nodes = 2;                                ///< expert halves per socket: the arena is split over this many
    uint64_t vram_total = 32ull << 30;
    uint64_t ram_total = 384ull << 30;
    uint64_t ctx_tokens = 131072;                      ///< KV cache length to reserve
    uint64_t kv_bytes_per_token = 3200;                ///< PLAN.md section 2
    uint64_t gpu_reserve = 3ull << 30;                 ///< activations, prompt buffers, CUDA context
    uint64_t os_reserve = 8ull << 30;
    uint64_t host_buffers = 3ull << 30;
    double kernel_overhead_pct = 1.8;                  ///< RAM the kernel keeps (384 -> ~377 GiB)
};

struct MemoryPlan {
    PlanInputs in;
    ByteTally tally;
    // GPU
    uint64_t gpu_dense = 0, gpu_cache = 0, gpu_residency = 0, gpu_kv = 0, gpu_reserve = 0, gpu_cache_budget = 0;
    int cache_slots_fit = 0;                           ///< whole experts the budget (VRAM - dense - KV - reserve) holds
    bool gpu_fits = true;
    // RAM
    uint64_t ram_usable = 0, ram_experts = 0, ram_experts_per_node = 0, ram_token_embd = 0, ram_engram = 0;
    int64_t ram_page_cache = 0;                        ///< what is left for the Engram tables' page cache (negative: the experts do not fit)
    double engram_cached_fraction = 0;
    std::vector<std::string> warnings;
    std::string text() const;                          ///< the printed table (like tools/ds41/memplan.py)
};
/// From the byte tally of a file (or of `expected_tensors` at any shape: the real model's plan needs no file).
MemoryPlan make_memory_plan(const Ds41Config& c, const ByteTally& t, const PlanInputs& in);

// ================================================================================================ the model
struct LoadOptions {
    int n_slots = 0;                                   ///< GPU expert-cache slots; -1: as many as the memory plan says fit
    std::vector<ExpertId> initial_fill;                ///< experts to put in the cache at load (empty: static_fill_by_index); more than n_slots is an error
    bool fill_cache = true;                            ///< false: allocate the cache (residency all -1) and leave it empty
    bool build_arena = true;                           ///< false: skip the CPU arena (tests, or a GPU-only run)
    ArenaOptions arena;                                ///< threads / NUMA / log are taken from here, `log` below overrides its log
    bool drop_page_cache = true;                       ///< drop the expert tensors' pages after they were copied (keeps the page cache for Engram)
    bool allow_unexpected_tensors = false;
    PlanInputs plan;                                   ///< the box described to the memory plan (n_slots is filled from the field above)
    LogFn log;                                         ///< progress and the memory plan; null: stderr
};

/// A loaded model at geometry G.  Owns the mapped files, the device allocations (given back to `dev` on destruction: `dev` must outlive it),
/// the arena and the cache.  Not movable (the weights point at the config and the mappings): it is handed out in a unique_ptr.
template <class G> class Ds41Model {
    // declared first so that they are destroyed LAST: the weights, the arena and the cache point into the mappings and the config
    std::unique_ptr<GgufSet> gguf_;
    Ds41Config cfg_;
    TensorDir dir_;
    MemoryPlan plan_;
    std::vector<std::string> warnings_;

public:
    /// `gguf_path` is any shard of the split model (or the single file).  Throws a ModelError for a file that is not this model at geometry G.
    static std::unique_ptr<Ds41Model> load(ModelDev& dev, const std::string& gguf_path, const LoadOptions& opt = {});

    Ds41Model(const Ds41Model&) = delete;
    Ds41Model& operator=(const Ds41Model&) = delete;

    const Ds41Config& config() const { return cfg_; }
    const GgufSet& gguf() const { return *gguf_; }
    const TensorDir& directory() const { return dir_; }
    const MemoryPlan& plan() const { return plan_; }
    const std::vector<std::string>& warnings() const { return warnings_; }

    Ds41Weights<G> weights;
    ExpertArena arena;
    GpuExpertCache cache;

private:
    Ds41Model() = default;
};

/// The non-template body of Ds41Model::load, after the geometry has been checked (the same work at every shape).
void load_parts(ModelDev& dev, const GgufSet& gguf, const Ds41Config& cfg, const TensorDir& dir, const ExpertDims& dims, const LoadOptions& opt,
                WeightsCore& weights, ExpertArena& arena, GpuExpertCache& cache, MemoryPlan& plan, std::vector<std::string>& warnings);

template <class G> std::unique_ptr<Ds41Model<G>> Ds41Model<G>::load(ModelDev& dev, const std::string& gguf_path, const LoadOptions& opt) {
    std::unique_ptr<Ds41Model> m(new Ds41Model());
    m->gguf_ = GgufSet::open(gguf_path);
    m->cfg_ = read_config(m->gguf_->meta());
    require_geometry<G>(m->cfg_);
    m->dir_ = m->gguf_->directory();
    load_parts(dev, *m->gguf_, m->cfg_, m->dir_, expert_dims<G>(), opt, m->weights, m->arena, m->cache, m->plan_, m->warnings_);
    return m;
}

}  // namespace strata::ds41::model
