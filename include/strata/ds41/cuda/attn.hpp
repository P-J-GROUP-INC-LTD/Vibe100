// include/strata/ds41/cuda/attn.hpp - DS1-C: the CSA2 attention of DeepSeek-V4.1-Flash for the V100 (sm_70), one layer at a time.
//
// WHAT.  `Ds41Attention<G>` is everything `ref/ds41/attention.py: attention_layer` does, for one decode token (and, by looping, for a window
// of T <= 8 consecutive positions: see "T" below), on the device:
//     q path        wq_a (int8 act) -> q_norm -> wq_b (int8 act) -> [kHeads][kHeadDim] -> RoPE on the last kRopeDim channels
//     SWA KV        wkv (int8 act) -> kv_norm -> RoPE -> fp8 e4m3 fake-quant per 32 (flag window_kv) -> the layer's ring, slot = pos % kWindow
//     compressor    FULL layers: ratio >= 2: wkv + wgate (BF16, FP32 act), group state across steps, softmax over the group, RMSNorm,
//                   RoPE at group * ratio, fp4 e4m3-scale fake-quant per 16 (flag compressed_kv); ratio 1: RMSNorm(wkv(x)), one entry per token
//     indexer       FULL / REINDEX layers: index_k from the PRE-RoPE latent (BF16 wk + k_norm + RoPE + fp4 e8m0 per 32, flag index), iq =
//                   wq_b(qr) (int8 act, RoPE, fp4), w = weights_proj(x) * (kIdxDim * kIdxHeads)^-1/2, score[t] = sum_h relu(iq[h].index_k[t]) * w[h],
//                   candidate pool on the candidate-source layer (block max, newest block pinned, top blocks), REINDEX layers masked by it,
//                   top-kIdxTopK by score (ties -> the LOWER position), position-sorted, -1 padding
//     attention     keys = <= kWindow ring entries + the selected compressed entries of the compress owner's cache; K == V; scores * kHeadDim^-1/2;
//                   softmax with the sink (exp(sink - max) in the denominator only); inverse RoPE on o
//     output        grouped wo_a (FP32 activations, block-diagonal over kOGroups) -> wo_b (int8 act)
// plus the caches (FP32, device memory, sized by a RUNTIME max context) and the cross-layer shared state of one decode step (compress owner,
// index-K owner, top-k indices, candidate mask), exactly `attention.SharedState` with the port's index-K rule (CONTRACTS.md: each owner scores
// its own cache).  Layer roles come from the GGUF metadata at run time (`Ds41LayerRoles`), never from constants.
//
// HOW IT IS BUILT.  The kernels live in src/ds41/cuda/attn_impl.cuh, written once and compiled twice (nvcc for sm_70: attn_real.cu instantiates
// AttnKernels<RealGeom>; the host compiler with -DDS41_EMU against the emulator: the tests).  The per-layer driver is host code
// (src/ds41/attn/attn_host_impl.hpp, instantiated for RealGeom in attn_real.cpp).  The dense work (Q8_0 / BF16 GEMVs, the activation quantiser)
// is NOT here: it goes through `AttnDenseOps` below, implemented by `Ds41AttnDense<G>` (attn_dense.hpp: DS1-B's dense.hpp GEMVs + DS1-G's
// ds41_quantize_acts<G>, ActOrder::kNatural); the tests also have a plain-C++ FP64 implementation.  Everything else - RMSNorm, RoPE, the three KV fake-quantisers, the compressor, the
// indexer, top-k, sparse attention - is in this package, so the numerics of the attention do not depend on the other packages' details.
//
// T.  forward() takes T = 1..kMaxT consecutive positions pos0 .. pos0 + T - 1.  The dense projections run batched over the T rows (DS1-B's rule:
// a row's result does not depend on T), the cache-dependent part runs token by token in position order, so T rows give BIT-IDENTICAL rows to T
// calls with T = 1.  DS-1 uses T = 1 (prefill = the prompt one token at a time, which equals the oracle's prefill: CONTRACTS.md).  Layers must be
// called in order within a step (layer 0 .. n-1) for the whole window, as the oracle's SharedState assumes; a window of T tokens is run
// layer-by-layer (all T tokens through layer l before layer l + 1), the shared top-k / candidate state therefore has one slot per window row.
//
// DEVICE POINTERS.  Every pointer in the structs below is device memory (HostDev's malloc in the emulation).  Alignment: 16 bytes for every
// FP32 / BF16 array, 2 bytes for Q8_0 weights (34-byte blocks), 4 for int32.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"

namespace strata::ds41::cuda {

// ---------------------------------------------------------------------------------------------------------------------------------
// layer roles (docs/deepseek/DS1.md section 3; ref/ds41/config.py: layer_modes)
// ---------------------------------------------------------------------------------------------------------------------------------
enum class AttnMode : int {
    kSwa = 0,      // compress ratio 0: the sliding window only
    kFull = 1,     // a KV source: own compressor + indexer; writes comp_kv and index_k; the candidate source also builds the pool
    kReuse = 2,    // reads the compress owner's comp_kv and the last top-k indices
    kReindex = 3,  // own indexer q (wq_b, weights_proj), the owner's index_k and comp_kv, scores restricted to the candidate pool
};

struct Ds41LayerRoles {
    std::vector<int> compress_ratio;       // one entry per backbone layer (the GGUF's array has 3 more for the DSpark blocks: pass the first n_layers)
    std::vector<int> kv_source_layers;     // attention.kv_source_layer_ids
    std::vector<int> index_source_layers;  // attention.index_source_layer_ids
    int candidate_source_layer = -1;       // attention.candidate_source_layer_id (-1: no candidate pool)

    int n_layers() const { return (int) compress_ratio.size(); }
    AttnMode mode(int layer) const;
    /// config.py `uses_candidates`: a layer after the candidate source masks its index scores with the pool.
    bool uses_candidates(int layer) const { return candidate_source_layer >= 0 && candidate_source_layer < layer; }
    bool is_candidate_source(int layer) const { return layer == candidate_source_layer; }
    /// "" when consistent (config.py `Config.validate` plus what this package needs: the ratio of a layer equals its owner's, <= kMaxCompressRatio).
    std::string validate() const;
};

/// The roles from DS1-A's Ds41Config (model/config.hpp) or anything with the same member names (compress_ratios holding exactly n_layers entries, kv_source,
/// index_source, cand_source): the four vectors the loader read from the GGUF metadata.
template <class Cfg>
Ds41LayerRoles ds41_attn_roles(const Cfg& c) {
    Ds41LayerRoles r;
    r.compress_ratio.assign(c.compress_ratios.begin(), c.compress_ratios.end());
    r.kv_source_layers.assign(c.kv_source.begin(), c.kv_source.end());
    r.index_source_layers.assign(c.index_source.begin(), c.index_source.end());
    r.candidate_source_layer = c.cand_source;
    return r;
}

/// The three KV fake-quantisations (QuantConfig.window_kv / compressed_kv / index); default all on (DS1.md section 2).
struct AttnQuantFlags {
    bool window_kv = true;       // fp8 e4m3, power-of-two scale per 32, over the whole post-RoPE SWA KV vector
    bool compressed_kv = true;   // fp4 e2m1, e4m3 scale per 16, on the post-RoPE compressed KV
    bool index = true;           // fp4 e2m1, power-of-two scale per 32, on index_k and iq
};

// ---------------------------------------------------------------------------------------------------------------------------------
// the dense work this package needs from DS1-B / DS1-G (implemented by Ds41AttnDense<G>, attn_dense.hpp; semantics as DS1.md section 2 / 5)
// ---------------------------------------------------------------------------------------------------------------------------------
struct AttnDenseOps {
    virtual ~AttnDenseOps() = default;
    /// x fp32 [T][k] (k a multiple of 32) -> xq int8 [T][k] (natural order inside each 32-block), xs fp32 [T][k / 32]: CONTRACTS.md's quantiser
    /// (d = amax / 127, q = rint(x * 127 / amax), a non-finite block -> d = NaN, an all-zero / tiny block -> 0).
    virtual void quantize_acts(const float* x, int T, int k, int8_t* xq, float* xs, Stream s) = 0;
    /// y[t][r] = sum over the k / 32 blocks b of  d_w[r][b] * xs[t][b] * (sum_j wq[r][b][j] * xq[t][b][j])   (the integer sum exact, FP32 elsewhere),
    /// w = GGML Q8_0 blocks (fp16 d + int8 qs[32], 34 bytes, no padding between rows: row r starts at w + r * (k / 32) * 34).  y fp32 [T][n].
    virtual void gemv_q8(const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream s) = 0;
    /// The grouped (block-diagonal) wo_a: w = Q8_0 rows [groups * rows_per_group][k_per_group]; y[t][g * rows_per_group + r] =
    /// sum_d x[t][g * k_per_group + d] * dequant(w[g * rows_per_group + r][d]); FP32 activations and accumulation (the reference keeps wo_a bf16).
    virtual void gemv_q8_grouped_f32(const void* w, int groups, int rows_per_group, int k_per_group, const float* x, int T, float* y, Stream s) = 0;
    /// y[t][r] = sum_d x[t][d] * bf16_to_f32(w[r][d]); FP32 activations and accumulation; w raw BF16 bits [n][k].
    virtual void gemv_bf16(const uint16_t* w, int n, int k, const float* x, int T, float* y, Stream s) = 0;
};

// ---------------------------------------------------------------------------------------------------------------------------------
// weights of one layer (all device pointers; formats as in the GGUF, RESEARCH.md section 10)
// ---------------------------------------------------------------------------------------------------------------------------------
struct AttnRope {
    const float* cos = nullptr;   // fp32 [max_context][kRopeDim / 2], from ref/ds41/rope.py: rope_table (double on the host) cast to fp32
    const float* sin = nullptr;   // layers 0-1: theta 10000 plain; layers >= 2: theta 160000 + YaRN (rope.py: layer_rope_params)
};

struct AttnLayerWeights {
    // every layer
    const void* wq_a = nullptr;             // Q8_0 [kQLora][kHidden]            attn_q_a
    const float* q_norm = nullptr;          // F32  [kQLora]                     attn_q_a_norm
    const void* wq_b = nullptr;             // Q8_0 [kHeads * kHeadDim][kQLora]  attn_q_b
    const void* wkv = nullptr;              // Q8_0 [kHeadDim][kHidden]          attn_kv
    const float* kv_norm = nullptr;         // F32  [kHeadDim]                   attn_kv_a_norm
    const float* sink = nullptr;            // F32  [kHeads]                     attn_sinks
    const void* wo_a = nullptr;             // Q8_0 [kOGroups * kOLora][kHeads * kHeadDim / kOGroups]   attn_output_a
    const void* wo_b = nullptr;             // Q8_0 [kHidden][kOGroups * kOLora]                        attn_output_b
    AttnRope rope;
    // FULL layers
    const uint16_t* comp_wkv = nullptr;     // BF16 [kHeadDim][kHidden]          attn_compressor_kv
    const uint16_t* comp_wgate = nullptr;   // BF16 [kHeadDim][kHidden]          attn_compressor_gate (ratio >= 2 only)
    const float* comp_norm = nullptr;       // F32  [kHeadDim]                   attn_compressor_norm
    const uint16_t* idx_wk = nullptr;       // BF16 [kIdxDim][kHeadDim]          indexer_compressor_kv
    const float* idx_k_norm = nullptr;      // F32  [kIdxDim]                    indexer_compressor_norm
    // FULL and REINDEX layers
    const void* idx_wq_b = nullptr;         // Q8_0 [kIdxHeads * kIdxDim][kQLora]  indexer.attn_q_b
    const uint16_t* idx_weights_proj = nullptr;   // BF16 [kIdxHeads][kHidden]     indexer.proj
};

// ---------------------------------------------------------------------------------------------------------------------------------
// the kernels (src/ds41/cuda/attn_impl.cuh).  Host wrappers: each launches one kernel on `s`; pointers are device pointers.
// ---------------------------------------------------------------------------------------------------------------------------------
inline constexpr int kAttnMaxCompressRatio = 4;      // the compressor's per-group registers; the real model uses 1 and 2
inline constexpr int kAttnTopkThreads = 256;

struct AttnCompressArgs {
    const float* kv_c = nullptr;        // fp32 [kHeadDim]  wkv(x)
    const float* score_c = nullptr;     // fp32 [kHeadDim]  wgate(x)   (ratio >= 2)
    float* kv_state = nullptr;          // fp32 [ratio][kHeadDim]      (ratio >= 2)
    float* score_state = nullptr;       // fp32 [ratio][kHeadDim]      (ratio >= 2)
    int ratio = 1;
    int slot = 0;                       // pos % ratio
    int complete = 1;                   // (pos + 1) % ratio == 0: a group is done, emit the latent
    const float* norm_w = nullptr;      // fp32 [kHeadDim]
    float eps = 1e-20f;
    const float* cos_row = nullptr;     // rope row of the position group * ratio  (pos + 1 - ratio)
    const float* sin_row = nullptr;
    int fp4 = 1;                        // compressed_kv fake-quant
    float* latent_out = nullptr;        // fp32 [kHeadDim]: the PRE-RoPE normalised latent (the indexer's input); written when complete
    float* comp_row = nullptr;          // fp32 [kHeadDim]: the cache entry (RoPE'd, fake-quantised); written when complete
};

template <class G>
struct AttnKernels {
    static constexpr int kHeadsPerBlock = G::kHeads >= 8 ? 8 : G::kHeads;   // sparse attention: query heads per block (one warp each in the softmax)

    /// y = rmsnorm(x [n]) over one row: x * (1 / sqrt(mean(x^2) + eps)) * w, n a multiple of 32.
    static void rmsnorm_row(const float* x, const float* w, int n, float eps, float* y, Stream s);
    /// in place: RoPE on the last kRopeDim channels of each of the kHeads heads of q [kHeads][kHeadDim].  cos_row / sin_row: kRopeDim / 2 floats.
    static void q_rope(float* q, const float* cos_row, const float* sin_row, Stream s);
    /// kv_norm + RoPE + (fp8) fake-quant of one SWA KV row; writes the ring row.
    static void swa_kv(const float* kv_raw, const float* norm_w, float eps, const float* cos_row, const float* sin_row, bool fp8, float* ring_row, Stream s);
    /// compressor: state update, (when a group completes) softmax-pool, RMSNorm, RoPE, fp4.  One block.
    static void compress_step(const AttnCompressArgs& a, Stream s);
    /// index_k row: RMSNorm(ik_raw [kIdxDim]) * w, RoPE on the last kRopeDim, fp4 e8m0 per 32 (fp4).
    static void index_k_finish(const float* ik_raw, const float* norm_w, float eps, const float* cos_row, const float* sin_row, bool fp4, float* out_row, Stream s);
    /// in place on iq [kIdxHeads][kIdxDim]: RoPE on the last kRopeDim of every head, fp4 e8m0 per 32 (fp4).
    static void index_q_finish(float* iq, const float* cos_row, const float* sin_row, bool fp4, Stream s);
    /// score[t] = sum_h relu(iq[h] . index_k[t]) * (wproj[h] * wscale) for t < n; with cand_flags != null a position whose block (t / cand_block)
    /// is not flagged gets -inf (the REINDEX mask).
    static void index_scores(const float* iq, const float* wproj, float wscale, const float* index_k, int n, const uint8_t* cand_flags, int cand_block,
                             float* score, Stream s);
    /// bs[b] = max of score over block b of `block` positions (n positions), the block of position n - 1 pinned to +inf; bs has ceil(n / block) entries.
    static void block_scores(const float* score, int n, int block, float* bs, Stream s);
    /// Select the k best of scores[0..n) (finite or +inf; -inf and NaN are never selected; ties -> the lower index); k <= n.  A selection smaller
    /// than k (fewer eligible entries) is fine.  out_idx (optional): the selected indices ascending, then -1 up to out_len;  out_flags (optional):
    /// out_flags[i] = 1 for the selected, 0 for the rest (n entries).  One block of kAttnTopkThreads threads, a radix select: deterministic.
    static void topk_select(const float* scores, int n, int k, int32_t* out_idx, int out_len, uint8_t* out_flags, Stream s);
    /// One query: ring window (slot pos % kWindow newest) + the compressed rows named by topk_row [kIdxTopK] (ascending positions, -1 padding;
    /// null = SWA layer) -> o_rot [kHeads][kHeadDim] (softmax with sink, inverse RoPE on the tail).  win_kv [kWindow][kHeadDim],
    /// comp_kv [..][kHeadDim], sink [kHeads].
    static void sparse_attn(const float* q, const float* win_kv, int pos, const float* comp_kv, const int32_t* topk_row, const float* sink,
                            const float* cos_row, const float* sin_row, float* o_rot, Stream s);
    /// Test hook: fake-quantise x [n] (n a multiple of 32): kind 0 = fp8 e4m3 / 32, 1 = fp4 e8m0 / 32, 2 = fp4 e4m3 / 16.
    static void fakequant(int kind, const float* x, int n, float* y, Stream s);
};

// ---------------------------------------------------------------------------------------------------------------------------------
// the per-layer driver
// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
class Ds41Attention {
   public:
    static constexpr int kMaxT = 8;

    /// `dev` allocates the caches and the scratch (init); `dense` does the GEMVs.  Both must outlive this object.
    Ds41Attention(Dev& dev, AttnDenseOps& dense);
    ~Ds41Attention();
    Ds41Attention(const Ds41Attention&) = delete;
    Ds41Attention& operator=(const Ds41Attention&) = delete;

    /// Allocates the per-layer caches for positions 0 .. max_context - 1 and the shared state.  norm_eps = the GGUF's attention.layer_norm_rms_epsilon
    /// (1e-20 for the real model).  Throws std::invalid_argument on inconsistent roles.
    void init(const Ds41LayerRoles& roles, int max_context, const AttnQuantFlags& quant, float norm_eps = 1e-20f);
    void set_weights(int layer, const AttnLayerWeights& w);
    void set_quant(const AttnQuantFlags& q);
    /// A new sequence: forget the shared state and clear the group state of the compressors (the caches need no clearing: nothing reads past
    /// the positions written).
    void reset();

    /// x_normed fp32 [T][kHidden] (the attn_norm'd block input) at positions pos0 .. pos0 + T - 1 -> out fp32 [T][kHidden].  Stream-ordered on `s`.
    /// Requires pos0 + T <= max_context and, per layer, pos0 equal to the number of tokens that layer has already seen (checked).
    void forward(int layer, const float* x_normed, int T, int pos0, float* out, Stream s);

    // ---- what the last forward() of a layer left in device memory (trace stages; valid until the next forward() of any layer) ----
    const float* trace_q(int t) const;                          // fp32 [kHeads][kHeadDim], post-RoPE
    const float* trace_o(int t) const;                          // fp32 [kHeads][kHeadDim], after the inverse RoPE (the input of wo_a)
    const float* trace_latent(int t) const;                     // fp32 [kHeadDim], the pre-RoPE normalised group latent; null if token t completed no group
    const int32_t* trace_topk(int t) const;                     // int32 [kIdxTopK], ascending, -1 padding (REUSE layers: the owner's)
    /// The indexer's scores of the LAST token the last forward() scored (an index source: FULL / REINDEX layer): fp32 [*n] (after the candidate mask:
    /// -inf = unreachable); null / *n = 0 if no layer scored.  Valid until the next forward().  With T > 1 only the last window row's.
    const float* trace_scores(int* n) const;
    const float* trace_block_scores(int* n) const;              // the candidate source's block scores of the last token (the newest block +inf), same rules
    const uint8_t* trace_cand(int t, int* nb) const;            // the candidate pool flags (1 per kCandBlock positions) the candidate source left for window row t
    const float* kv_win_row(int layer, int pos) const;          // the ring row of position pos (valid while pos is within the last kWindow)
    const float* comp_kv_row(int layer, int index) const;       // null unless `layer` is a FULL layer
    const float* index_k_row(int layer, int index) const;       // null unless `layer` is a FULL layer
    AttnMode mode(int layer) const;
    int compress_ratio(int layer) const;
    int max_context() const;
    size_t device_bytes() const;                                // everything init() allocated

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strata::ds41::cuda
