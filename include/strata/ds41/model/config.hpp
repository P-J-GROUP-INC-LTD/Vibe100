// include/strata/ds41/model/config.hpp - DS1-A: the model configuration read from the GGUF metadata, and checked against the
// compile-time geometry.
//
// `Ds41Config` carries every key that ref/ds41/config.py `Config.from_gguf_metadata` reads (the oracle's field names are in the
// comments), plus the ones the port needs beyond it (tokenizer vocabulary, Engram primes / offsets / multipliers / token_map,
// per-layer RoPE) and the DERIVED layer roles of docs/deepseek/DS1.md section 3 (`layers`, the same map as config.py `layer_modes`).
//
//   read_config(meta)             metadata -> Ds41Config; refuses a missing / ill-typed key and a self-inconsistent layer map
//   check_geometry<G>(cfg, f)     every dimension the kernels compile in (G::k*) against the file's; one finding per mismatch
//   require_geometry<G>(cfg)      the same, thrown as one ModelError ("`deepseek41.embedding_length` = 384, this build expects 256")
//
// Values the kernels hard-code that G does not carry (route scale 1.5, SwiGLU clamp 10, RMSNorm eps 1e-20, one shared expert, the
// sqrtsoftplus gate, no hash layers) are checked by `read_config` itself against geometry.hpp, so a different model is refused here
// and not computed wrongly.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/ds41/geom.hpp"
#include "strata/ds41/model/gguf.hpp"

namespace strata::ds41::model {

/// What a layer's attention does (docs/deepseek/DS1.md section 3; ref/ds41/config.py `Mode`).
enum class Role { SWA, FULL, REUSE, REINDEX };
const char* role_name(Role r);     ///< "SWA" / "FULL" / "REUSE" / "REINDEX"

/// RoPE of one layer kind (ref/ds41/rope.py `layer_rope_params`): layers without compression use `rope.freq_base` and no YaRN,
/// compressed layers `attention.compress_rope_freq_base` with YaRN (factor, beta_fast, beta_slow, original context).
struct RopeParams {
    double base = 10000.0;
    bool yarn = false;
    int64_t orig_ctx = 0;              ///< original_seq_len; 0 with yarn == false
    double factor = 1.0, beta_fast = 32.0, beta_slow = 1.0;
};

struct LayerInfo {
    int layer = 0;
    int ratio = 0;                     ///< attention.compress_ratios[layer]: 0 (SWA only), 1 or 2
    Role role = Role::SWA;
    bool has_compressor = false;       ///< FULL: attn_compressor_kv + attn_compressor_norm (global KV is built here)
    bool has_compressor_gate = false;  ///< FULL and ratio 2: attn_compressor_gate (the softmax pooling gate)
    bool has_indexer = false;          ///< FULL or REINDEX: indexer.attn_q_b + indexer.proj
    bool has_index_compressor = false; ///< FULL: indexer_compressor_kv + indexer_compressor_norm (index-K is built here)
    int kv_owner = -1;                 ///< layer whose compressed KV this layer reads: itself for FULL, the latest FULL layer before it, -1 for SWA
    int index_k_owner = -1;            ///< layer whose index-K the indexer scores: itself for FULL, the owner for REINDEX, -1 otherwise
    int topk_source = -1;              ///< layer whose top-k indices this layer attends with: itself for FULL / REINDEX, the latest indexer layer before a REUSE one
    bool is_candidate_source = false;  ///< FULL layer whose scores build the candidate pool (attention.candidate_source_layer_id)
    bool uses_candidate_pool = false;  ///< an indexer layer after the candidate source: its scores are masked to the pool
    int engram_slot = -1;              ///< index into EngramConfig::layers, or -1
    bool is_engram() const { return engram_slot >= 0; }
    const char* mode() const { return role_name(role); }
};

/// The Engram constants of the file (`deepseek41.engram.*`): authoritative for THAT file (ref/ds41/engram.py
/// `constants_from_gguf_metadata`).  Rows of table `slot` are addressed as offsets[slot][col] + hash % primes[slot][col], with
/// col = (order - 2) * heads + head, so a layer's table has sum(primes[slot]) rows (= num_embeddings[slot]).
struct EngramConfig {
    std::vector<int> layers;           ///< engram.layer_ids (the order is the slot order)
    int heads = 0;                     ///< engram.n_heads                  (oracle: engram_n_heads)
    int head_dim = 0;                  ///< engram.head_dim: values per row (engram_head_dim)
    int ngram = 0;                     ///< engram.max_ngram_size           (engram_max_ngram_size)
    int pad_id = 0;                    ///< engram.pad_token_id: a TOKEN id (engram_pad_id)
    int cvocab = 0;                    ///< engram.compressed_vocab_size    (engram_compressed_vocab_size)
    std::vector<int64_t> num_embeddings;   ///< rows per table
    std::vector<int64_t> primes;       ///< [slot][col]
    std::vector<int64_t> offsets;      ///< [slot][col]
    std::vector<int64_t> multipliers;  ///< [slot][ngram]
    std::vector<int32_t> token_map;    ///< token id -> compressed id, size == vocab (empty only when read_config was told to skip it)
    int cols() const { return (ngram - 1) * heads; }               ///< table rows read per token per layer (24 for the real model)
    int64_t prime(int slot, int col) const { return primes[(size_t) slot * cols() + col]; }
    int64_t offset(int slot, int col) const { return offsets[(size_t) slot * cols() + col]; }
    int64_t multiplier(int slot, int k) const { return multipliers[(size_t) slot * ngram + k]; }
    int pad_compressed() const { return token_map.empty() ? -1 : token_map[(size_t) pad_id]; }   ///< the pad token's compressed id
};

struct Ds41Config {
    std::string architecture, name;
    // ---- trunk
    int n_layer = 0;                   ///< block_count             (n_layers)
    int hidden = 0;                    ///< embedding_length        (dim)
    int vocab = 0;                     ///< the tokenizer's size (tokenizer.ggml.tokens); token_embd / output have this many rows
    int64_t context_length = 0;        ///< context_length: the most the model was trained for (the engine's --max-context is a runtime cap)
    float rms_eps = 0;                 ///< attention.layer_norm_rms_epsilon (norm_eps)
    // ---- attention
    int n_head = 0, n_kv_head = 0;     ///< attention.head_count / head_count_kv (n_heads; one K == V head)
    int head_dim = 0, value_dim = 0;   ///< attention.key_length / value_length (head_dim)
    int rope_dim = 0;                  ///< rope.dimension_count (rope_head_dim): the LAST rope_dim channels are rotated
    int q_lora = 0;                    ///< attention.q_lora_rank
    int o_groups = 0, o_lora = 0;      ///< attention.output_group_count / output_lora_rank
    int window = 0;                    ///< attention.sliding_window (window_size)
    RopeParams rope_swa, rope_csa;     ///< rope.freq_base plain; compress_rope_freq_base + YaRN (rope.scaling.*)
    // ---- MoE
    int n_expert = 0, n_used = 0, n_shared = 0, ff = 0;   ///< expert_count / expert_used_count / expert_shared_count / expert_feed_forward_length
    int gating_func = 0;               ///< expert_gating_func (4 = sqrtsoftplus)
    float route_scale = 0;             ///< expert_weights_scale (route_scale)
    bool route_norm = false;           ///< expert_weights_norm (norm_topk_prob)
    float swiglu_limit = 0;            ///< swiglu_clamp_exp / swiglu_clamp_shexp (equal for every layer)
    // ---- indexer and candidate pool
    int idx_heads = 0, idx_dim = 0, idx_topk = 0;   ///< attention.indexer.head_count / key_length / top_k
    int cand_source = -1;              ///< attention.candidate_source_layer_id (-1: no pool)
    int cand_block = 0, cand_topk_blocks = 0;       ///< attention.candidate_block_size / candidate_topk_blocks
    std::vector<int> compress_ratios;  ///< attention.compress_ratios, the first n_layer entries
    int n_ratio_entries = 0;           ///< entries the file has (n_layer, or n_layer + 3 with the DSpark ones)
    std::vector<int> kv_source, index_source;       ///< attention.kv_source_layer_ids / index_source_layer_ids
    // ---- mHC
    int hc = 0, hc_iters = 0;          ///< hyper_connection.count / sinkhorn_iterations (hc_mult, hc_sinkhorn_iters)
    float hc_eps = 0;                  ///< hyper_connection.epsilon
    // ---- Engram
    EngramConfig engram;
    // ---- derived
    std::vector<LayerInfo> layers;     ///< n_layer entries
    std::vector<std::string> warnings;

    const LayerInfo& layer(int l) const { return layers[(size_t) l]; }
    const RopeParams& rope_of(int l) const { return layers[(size_t) l].ratio ? rope_csa : rope_swa; }
    int n_engram() const { return (int) engram.layers.size(); }
    int n_hc_mixes() const { return (2 + hc) * hc; }
    int n_q() const { return n_head * head_dim; }                   ///< wq_b output = the attention output width
    /// A multi-line description: dimensions, the layer-role map, Engram layers, RoPE.
    std::string describe() const;
};

/// Reads every key and derives the layer roles.  `with_big_arrays` false skips the 129,280-entry engram.token_map (a headers-only
/// test's metadata has none; its length is still checked when present).  Throws a ModelError listing every problem found.
Ds41Config read_config(const MetaStore& meta, bool with_big_arrays = true);

/// ref/ds41/rope.py `rope_table`, in double, stored as FP32: cos / sin [n_pos][dim / 2] (dim = rope_dim: the pairs of the rotated tail).
struct RopeTable {
    int dim = 0, n_pos = 0;
    std::vector<float> cos, sin;
};
RopeTable build_rope_table(const RopeParams& p, int dim, int n_pos);

// ---- the geometry check (templates over G: geom.hpp) ---------------------------------------------------------------------------
namespace detail {
inline void mismatch(Findings& f, const char* key, int64_t got, int64_t want, const char* gname, const char* geom) {
    f.error(std::string("`") + key + "` = " + std::to_string(got) + ", but this build (geometry '" + geom + "') expects " +
            std::to_string(want) + " (G::" + gname + ")");
}
}  // namespace detail

/// Every dimension a kernel compiles in, against the file's.  The vocabulary is the tokenizer's size; the tensors' shapes are
/// checked separately (tensors.hpp) against the config, so a file whose metadata agrees with G but whose tensors do not is refused too.
template <class G> void check_geometry(const Ds41Config& c, Findings& f) {
#define DS41_CHECK_GEOM(KEY, FIELD, GNAME) \
    if ((int64_t) (FIELD) != (int64_t) G::GNAME) detail::mismatch(f, KEY, (FIELD), G::GNAME, #GNAME, G::kName)
    DS41_CHECK_GEOM("deepseek41.block_count", c.n_layer, kLayers);
    DS41_CHECK_GEOM("deepseek41.embedding_length", c.hidden, kHidden);
    DS41_CHECK_GEOM("tokenizer.ggml.tokens (vocabulary)", c.vocab, kVocab);
    DS41_CHECK_GEOM("deepseek41.hyper_connection.count", c.hc, kHc);
    DS41_CHECK_GEOM("deepseek41.hyper_connection.sinkhorn_iterations", c.hc_iters, kHcIters);
    DS41_CHECK_GEOM("deepseek41.expert_count", c.n_expert, kExperts);
    DS41_CHECK_GEOM("deepseek41.expert_used_count", c.n_used, kTopK);
    DS41_CHECK_GEOM("deepseek41.expert_feed_forward_length", c.ff, kFF);
    DS41_CHECK_GEOM("deepseek41.attention.head_count", c.n_head, kHeads);
    DS41_CHECK_GEOM("deepseek41.attention.key_length", c.head_dim, kHeadDim);
    DS41_CHECK_GEOM("deepseek41.rope.dimension_count", c.rope_dim, kRopeDim);
    DS41_CHECK_GEOM("deepseek41.attention.q_lora_rank", c.q_lora, kQLora);
    DS41_CHECK_GEOM("deepseek41.attention.output_group_count", c.o_groups, kOGroups);
    DS41_CHECK_GEOM("deepseek41.attention.output_lora_rank", c.o_lora, kOLora);
    DS41_CHECK_GEOM("deepseek41.attention.sliding_window", c.window, kWindow);
    DS41_CHECK_GEOM("deepseek41.attention.indexer.head_count", c.idx_heads, kIdxHeads);
    DS41_CHECK_GEOM("deepseek41.attention.indexer.key_length", c.idx_dim, kIdxDim);
    DS41_CHECK_GEOM("deepseek41.attention.indexer.top_k", c.idx_topk, kIdxTopK);
    DS41_CHECK_GEOM("deepseek41.attention.candidate_block_size", c.cand_block, kCandBlock);
    DS41_CHECK_GEOM("deepseek41.attention.candidate_topk_blocks", c.cand_topk_blocks, kCandTopBlocks);
    DS41_CHECK_GEOM("deepseek41.engram.n_heads", c.engram.heads, kEngramHeads);
    DS41_CHECK_GEOM("deepseek41.engram.head_dim", c.engram.head_dim, kEngramHeadDim);
    DS41_CHECK_GEOM("deepseek41.engram.max_ngram_size", c.engram.ngram, kEngramNgram);
#undef DS41_CHECK_GEOM
}

template <class G> void require_geometry(const Ds41Config& c) {
    Findings f;
    check_geometry<G>(c, f);
    f.throw_if_errors(std::string("this GGUF does not match the compiled geometry '") + G::kName + "'");
}

}  // namespace strata::ds41::model
