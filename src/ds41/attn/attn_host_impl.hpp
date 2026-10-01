// src/ds41/attn/attn_host_impl.hpp - DS1-C: the per-layer driver of the CSA2 attention (host code), `Ds41Attention<G>`.
//
// Included by src/ds41/attn/attn_real.cpp (RealGeom, linked with the nvcc-built kernels of attn_real.cu) and by the emulator build
// (attn_emu_impl.cpp: any G, kernels compiled for the host).  It transcribes `ref/ds41/attention.py: attention_layer` (and the layer-mode map of
// `config.py: layer_modes`), calling AttnKernels<G> for everything that is not a dense GEMV and AttnDenseOps for the GEMVs.
//
// ORACLE LINE MAP (ref/ds41/attention.py)               here
//   attention_layer 317-319 (qr, q, RoPE on q)          forward(): quantize_acts, gemv_q8(wq_a), rmsnorm_row(q_norm), quantize_acts, gemv_q8(wq_b); token(): q_rope
//   window_kv 105-127 (kv, RoPE, fp8, the ring)         forward(): gemv_q8(wkv);  token(): swa_kv
//   attention_layer 323-331 (ratio, compress_len,       token(): `ratio > 0` branch; shared.compress_owner = layer on FULL layers
//     FULL: compressor, shared.compress_owner)
//   compressor 134-172                                  forward(): gemv_bf16(comp_wkv / comp_wgate);  token(): compress_step (state, softmax-pool, norm)
//   indexer 202-261                                     forward(): gemv_q8(idx_wq_b), gemv_bf16(idx_weights_proj);  token(): index_k_finish (owner layers, when
//                                                       a group completes), index_q_finish, index_scores, block_scores + topk_select (candidate source),
//                                                       topk_select (the indices)
//   attention_layer 333-345 (cidx; REUSE: shared.topk)  token(): the per-window-row `topk` slot, written by the index source, read by REUSE layers
//   attention_layer 346-352 (RoPE + fp4 on latent,      compress_step (it needs the latent before the indexer reads it: it hands the PRE-RoPE latent
//     comp_kv write)                                    to the indexer through latent_out and writes the cache row in the same kernel)
//   attention_layer 353-357 (comp rows, sparse_attn)    token(): sparse_attn over the ring + L[compress_owner].comp
//   attention_layer 358-365 (inverse RoPE, wo_a, wo_b)  sparse_attn does the inverse RoPE; forward(): gemv_q8_grouped_f32(wo_a), quantize_acts, gemv_q8(wo_b)
// The index-K rule is the port's (CONTRACTS.md): a FULL layer scores against its own index_k (`owner = layer if owns_k and not stale_index_k`).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "strata/ds41/cuda/attn.hpp"

namespace strata::ds41::cuda {

// ---------------------------------------------------------------------------------------------------------------------------------
// roles
// ---------------------------------------------------------------------------------------------------------------------------------
namespace attn_detail {
inline bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
}  // namespace attn_detail

inline AttnMode Ds41LayerRoles::mode(int layer) const {
    if (compress_ratio.at((size_t) layer) == 0) return AttnMode::kSwa;
    if (attn_detail::contains(kv_source_layers, layer)) return AttnMode::kFull;
    if (attn_detail::contains(index_source_layers, layer)) return AttnMode::kReindex;
    return AttnMode::kReuse;
}

inline std::string Ds41LayerRoles::validate() const {
    const int n = n_layers();
    if (n <= 0) return "no layers";
    for (int l = 0; l < n; ++l)
        if (compress_ratio[(size_t) l] < 0 || compress_ratio[(size_t) l] > kAttnMaxCompressRatio)
            return "layer " + std::to_string(l) + ": compress ratio " + std::to_string(compress_ratio[(size_t) l]) + " is not 0.." + std::to_string(kAttnMaxCompressRatio);
    for (int l : kv_source_layers) {
        if (l < 0 || l >= n) return "kv source layer " + std::to_string(l) + " out of range";
        if (compress_ratio[(size_t) l] <= 0) return "kv source layer " + std::to_string(l) + " must compress";
        if (!attn_detail::contains(index_source_layers, l)) return "kv source layer " + std::to_string(l) + " must also own an indexer";
    }
    for (int l : index_source_layers) {
        if (l < 0 || l >= n) return "index source layer " + std::to_string(l) + " out of range";
        if (compress_ratio[(size_t) l] <= 0) return "index source layer " + std::to_string(l) + " must compress";
    }
    if (candidate_source_layer >= 0) {
        if (!attn_detail::contains(index_source_layers, candidate_source_layer)) return "the candidate source must be an index source";
        if (!attn_detail::contains(kv_source_layers, candidate_source_layer)) return "the candidate source must be a kv source (it builds the pool from its own scores)";
    }
    // every compressing layer needs a kv source before it, with the same ratio (the compressed entries are counted by the layer's own ratio)
    int owner = -1;
    for (int l = 0; l < n; ++l) {
        const AttnMode m = mode(l);
        if (m == AttnMode::kFull) owner = l;
        if (m == AttnMode::kReuse || m == AttnMode::kReindex) {
            if (owner < 0) return "layer " + std::to_string(l) + " reads a compressed cache but no kv source precedes it";
            if (compress_ratio[(size_t) owner] != compress_ratio[(size_t) l])
                return "layer " + std::to_string(l) + " has compress ratio " + std::to_string(compress_ratio[(size_t) l]) + " but its kv source " + std::to_string(owner) +
                       " has " + std::to_string(compress_ratio[(size_t) owner]);
        }
        if ((m == AttnMode::kFull || m == AttnMode::kReindex) && uses_candidates(l)) {
            if (compress_ratio[(size_t) candidate_source_layer] != compress_ratio[(size_t) l])
                return "layer " + std::to_string(l) + " masks its scores with the candidate pool of layer " + std::to_string(candidate_source_layer) + " but their ratios differ";
        }
    }
    return "";
}

// ---------------------------------------------------------------------------------------------------------------------------------
// the driver
// ---------------------------------------------------------------------------------------------------------------------------------
template <class G>
struct Ds41Attention<G>::Impl {
    using K = AttnKernels<G>;
    static constexpr int D = G::kHeadDim, ID = G::kIdxDim, QL = G::kQLora, HID = G::kHidden, KT = Ds41Attention<G>::kMaxT;
    static constexpr int Q = G::kHeads * G::kHeadDim;                    // Derived<G>::kQ
    static constexpr int IQ = G::kIdxHeads * G::kIdxDim;                 // Derived<G>::kIdxQ
    static constexpr int OMID = G::kOGroups * G::kOLora;                 // Derived<G>::kOMid
    static constexpr int OGIN = Q / G::kOGroups;                         // Derived<G>::kOGroupIn

    struct Layer {
        AttnMode mode = AttnMode::kSwa;
        int ratio = 0;
        AttnLayerWeights w;
        bool has_w = false;
        float* win = nullptr;            // [kWindow][kHeadDim]
        float* comp = nullptr;           // [max_comp][kHeadDim]                 FULL
        float* idxk = nullptr;           // [max_comp][kIdxDim]                  FULL
        float* kv_state = nullptr;       // [ratio][kHeadDim]                    FULL, ratio >= 2
        float* score_state = nullptr;    // [ratio][kHeadDim]
        int max_comp = 0;
        int seen = 0;                    // tokens this layer has processed = the next position
    };

    Dev& dev;
    AttnDenseOps& dense;
    Ds41LayerRoles roles;
    AttnQuantFlags quant;
    float eps = 1e-20f;
    int max_context = 0;
    bool inited = false;
    std::vector<Layer> L;
    std::vector<void*> allocs;
    size_t bytes = 0;

    // shared state of one decode step (attention.SharedState)
    int compress_owner = -1;
    int index_k_owner = -1;
    int32_t* topk = nullptr;             // [KT][kIdxTopK]
    uint8_t* cand = nullptr;             // [KT][cand_stride]: candidate flags per block of kCandBlock positions
    int cand_stride = 0;

    // scratch (one set: layers run one after the other on one stream)
    int8_t *xq0 = nullptr, *xq1 = nullptr, *xq2 = nullptr;
    float *xs0 = nullptr, *xs1 = nullptr, *xs2 = nullptr;
    float *qa = nullptr, *qr = nullptr, *q = nullptr, *kvraw = nullptr, *ckv = nullptr, *csc = nullptr, *latent = nullptr, *ikraw = nullptr;
    float *iq = nullptr, *iw = nullptr, *o_rot = nullptr, *omid = nullptr, *score = nullptr, *bs = nullptr;
    bool latent_valid[KT] = {};
    float wscale = 1.0f;

    Impl(Dev& d, AttnDenseOps& n) : dev(d), dense(n) {}
    ~Impl() {
        for (void* p : allocs) dev.release(p);
    }

    template <class T>
    T* alloc(size_t n) {
        T* p = static_cast<T*>(dev.alloc(n * sizeof(T) + 16));
        allocs.push_back(p);
        bytes += n * sizeof(T) + 16;
        return p;
    }
    template <class T>
    T* alloc_zero(size_t n) {
        T* p = alloc<T>(n);
        dev.fill(p, 0, n * sizeof(T));
        return p;
    }

    void init(const Ds41LayerRoles& r, int max_ctx, const AttnQuantFlags& qf, float norm_eps) {
        if (inited) throw std::logic_error("Ds41Attention::init called twice");
        const std::string why = r.validate();
        if (!why.empty()) throw std::invalid_argument("Ds41Attention: " + why);
        if (max_ctx < 1) throw std::invalid_argument("Ds41Attention: max_context must be >= 1");
        roles = r;
        quant = qf;
        eps = norm_eps;
        max_context = max_ctx;
        const int n_layers = r.n_layers();
        L.assign((size_t) n_layers, Layer{});
        int max_n = 1;                                   // the longest score row: max_context / ratio over the layers that score
        for (int l = 0; l < n_layers; ++l) {
            Layer& La = L[(size_t) l];
            La.mode = r.mode(l);
            La.ratio = r.compress_ratio[(size_t) l];
            La.win = alloc_zero<float>((size_t) G::kWindow * D);
            if (La.mode == AttnMode::kFull) {
                La.max_comp = max_ctx / La.ratio;
                if (La.max_comp < 1) La.max_comp = 1;
                La.comp = alloc_zero<float>((size_t) La.max_comp * D);
                La.idxk = alloc_zero<float>((size_t) La.max_comp * ID);
                if (La.ratio > 1) {
                    La.kv_state = alloc_zero<float>((size_t) La.ratio * D);
                    La.score_state = alloc<float>((size_t) La.ratio * D);
                }
            }
            if (La.mode == AttnMode::kFull || La.mode == AttnMode::kReindex) max_n = std::max(max_n, std::max(1, max_ctx / La.ratio));
        }
        int cand_blocks = 1;
        if (r.candidate_source_layer >= 0) cand_blocks = std::max(1, (max_ctx / r.compress_ratio[(size_t) r.candidate_source_layer] + G::kCandBlock - 1) / G::kCandBlock);
        cand_stride = cand_blocks;
        topk = alloc<int32_t>((size_t) KT * G::kIdxTopK);
        cand = alloc_zero<uint8_t>((size_t) KT * cand_stride);
        xq0 = alloc<int8_t>((size_t) KT * HID);
        xs0 = alloc<float>((size_t) KT * HID / 32);
        xq1 = alloc<int8_t>((size_t) KT * QL);
        xs1 = alloc<float>((size_t) KT * QL / 32);
        xq2 = alloc<int8_t>((size_t) KT * OMID);
        xs2 = alloc<float>((size_t) KT * OMID / 32);
        qa = alloc<float>((size_t) KT * QL);
        qr = alloc<float>((size_t) KT * QL);
        q = alloc<float>((size_t) KT * Q);
        kvraw = alloc<float>((size_t) KT * D);
        ckv = alloc<float>((size_t) KT * D);
        csc = alloc<float>((size_t) KT * D);
        latent = alloc<float>((size_t) KT * D);
        ikraw = alloc<float>((size_t) KT * ID);
        iq = alloc<float>((size_t) KT * IQ);
        iw = alloc<float>((size_t) KT * G::kIdxHeads);
        o_rot = alloc<float>((size_t) KT * Q);
        omid = alloc<float>((size_t) KT * OMID);
        score = alloc<float>((size_t) max_n);
        bs = alloc<float>((size_t) cand_blocks + 1);
        wscale = (float) (std::pow((double) G::kIdxDim, -0.5) * std::pow((double) G::kIdxHeads, -0.5));
        inited = true;
        reset();
    }

    void reset() {
        compress_owner = index_k_owner = -1;
        for (Layer& La : L) {
            La.seen = 0;
            if (La.kv_state) dev.fill(La.kv_state, 0, (size_t) La.ratio * D * sizeof(float));
            if (La.score_state) {
                std::vector<float> ninf((size_t) La.ratio * D, -std::numeric_limits<float>::infinity());
                dev.h2d(La.score_state, ninf.data(), ninf.size() * sizeof(float));
            }
        }
        for (bool& v : latent_valid) v = false;
    }

    void set_weights(int layer, const AttnLayerWeights& w) {
        Layer& La = L.at((size_t) layer);
        const char* what = nullptr;
        if (!w.wq_a || !w.q_norm || !w.wq_b || !w.wkv || !w.kv_norm || !w.sink || !w.wo_a || !w.wo_b || !w.rope.cos || !w.rope.sin) what = "a base attention tensor or the rope table";
        if (La.mode == AttnMode::kFull && (!w.comp_wkv || !w.comp_norm || (La.ratio > 1 && !w.comp_wgate) || !w.idx_wk || !w.idx_k_norm)) what = "a compressor / index-K tensor of a FULL layer";
        if ((La.mode == AttnMode::kFull || La.mode == AttnMode::kReindex) && (!w.idx_wq_b || !w.idx_weights_proj)) what = "an indexer tensor";
        if (what) throw std::invalid_argument("Ds41Attention::set_weights(layer " + std::to_string(layer) + "): missing " + what);
        La.w = w;
        La.has_w = true;
    }

    // ---- one token of one layer: everything that depends on the caches, in the oracle's order ----
    void token(int l, int t, int p, Stream s) {
        Layer& La = L[(size_t) l];
        const AttnLayerWeights& w = La.w;
        const int ratio = La.ratio;
        const size_t half = (size_t) G::kRopeDim / 2;
        const float* cosr = w.rope.cos + (size_t) p * half;
        const float* sinr = w.rope.sin + (size_t) p * half;
        float* qt = q + (size_t) t * Q;
        latent_valid[t] = false;

        K::q_rope(qt, cosr, sinr, s);
        K::swa_kv(kvraw + (size_t) t * D, w.kv_norm, eps, cosr, sinr, quant.window_kv, La.win + (size_t) (p % G::kWindow) * D, s);

        const int32_t* topk_row = nullptr;
        const float* comp_base = nullptr;
        if (ratio > 0) {
            const int compress_len = (p + 1) / ratio;
            int32_t* tk = topk + (size_t) t * G::kIdxTopK;
            if (La.mode == AttnMode::kFull) {
                compress_owner = l;
                const bool complete = (p + 1) % ratio == 0;
                // the latent stands for position group * ratio = p + 1 - ratio (only meaningful when complete)
                const size_t gp = complete ? (size_t) (p + 1 - ratio) * half : 0;
                AttnCompressArgs a;
                a.kv_c = ckv + (size_t) t * D;
                a.score_c = ratio > 1 ? csc + (size_t) t * D : nullptr;
                a.kv_state = La.kv_state;
                a.score_state = La.score_state;
                a.ratio = ratio;
                a.slot = p % ratio;
                a.complete = complete ? 1 : 0;
                a.norm_w = w.comp_norm;
                a.eps = eps;
                a.cos_row = w.rope.cos + gp;
                a.sin_row = w.rope.sin + gp;
                a.fp4 = quant.compressed_kv ? 1 : 0;
                a.latent_out = latent + (size_t) t * D;
                a.comp_row = La.comp + (size_t) (p / ratio) * D;
                K::compress_step(a, s);
                if (complete) {
                    latent_valid[t] = true;
                    // index_k = RMSNorm(wk(latent)) with RoPE and fp4, from the PRE-RoPE latent (the oracle: indexer, `owns_k and latent is not None`)
                    dense.gemv_bf16(w.idx_wk, ID, D, latent + (size_t) t * D, 1, ikraw + (size_t) t * ID, s);
                    K::index_k_finish(ikraw + (size_t) t * ID, w.idx_k_norm, eps, w.rope.cos + gp, w.rope.sin + gp, quant.index, La.idxk + (size_t) (p / ratio) * ID, s);
                }
            }
            if (compress_owner < 0) throw std::logic_error("Ds41Attention: a compressing layer ran before any kv source of this step");
            if (La.mode == AttnMode::kFull || La.mode == AttnMode::kReindex) {
                if (compress_len == 0) {
                    K::topk_select(score, 0, 0, tk, G::kIdxTopK, nullptr, s);       // `cidx = np.empty((s, 0))`: an all -1 row
                } else {
                    if (La.mode == AttnMode::kFull) index_k_owner = l;
                    if (index_k_owner < 0) throw std::logic_error("Ds41Attention: a REINDEX layer ran before any index-K owner of this step");
                    const Layer& O = L[(size_t) (La.mode == AttnMode::kFull ? l : index_k_owner)];
                    const uint8_t* mask = roles.uses_candidates(l) ? cand + (size_t) t * cand_stride : nullptr;
                    K::index_q_finish(iq + (size_t) t * IQ, cosr, sinr, quant.index, s);
                    K::index_scores(iq + (size_t) t * IQ, iw + (size_t) t * G::kIdxHeads, wscale, O.idxk, compress_len, mask, G::kCandBlock, score, s);
                    if (roles.is_candidate_source(l)) {
                        const int nb = (compress_len + G::kCandBlock - 1) / G::kCandBlock;
                        K::block_scores(score, compress_len, G::kCandBlock, bs, s);
                        K::topk_select(bs, nb, std::min(G::kCandTopBlocks, nb), nullptr, 0, cand + (size_t) t * cand_stride, s);
                    }
                    K::topk_select(score, compress_len, std::min(G::kIdxTopK, compress_len), tk, G::kIdxTopK, nullptr, s);
                }
            }
            topk_row = tk;                                      // REUSE layers: the slot the index source wrote for this window row
            comp_base = L[(size_t) compress_owner].comp;
        }
        K::sparse_attn(qt, La.win, p, comp_base, topk_row, w.sink, cosr, sinr, o_rot + (size_t) t * Q, s);
    }

    void forward(int l, const float* x, int T, int pos0, float* out, Stream s) {
        if (!inited) throw std::logic_error("Ds41Attention::forward before init");
        if (l < 0 || l >= (int) L.size()) throw std::out_of_range("Ds41Attention::forward: layer");
        if (T < 1 || T > KT) throw std::invalid_argument("Ds41Attention::forward: T must be 1.." + std::to_string(KT));
        Layer& La = L[(size_t) l];
        if (!La.has_w) throw std::logic_error("Ds41Attention::forward: no weights for layer " + std::to_string(l));
        if (pos0 != La.seen) throw std::invalid_argument("Ds41Attention::forward: layer " + std::to_string(l) + " has seen " + std::to_string(La.seen) + " tokens, not " + std::to_string(pos0));
        if (pos0 + T > max_context) throw std::invalid_argument("Ds41Attention::forward: position beyond max_context");
        s = stream_or_default(dev, s);
        const AttnLayerWeights& w = La.w;
        const bool has_idx = La.mode == AttnMode::kFull || La.mode == AttnMode::kReindex;
        // ---- dense projections of the T rows (a row's result does not depend on T: DS1.md section 2) ----
        dense.quantize_acts(x, T, HID, xq0, xs0, s);
        dense.gemv_q8(w.wq_a, QL, HID, xq0, xs0, T, qa, s);
        for (int t = 0; t < T; ++t) K::rmsnorm_row(qa + (size_t) t * QL, w.q_norm, QL, eps, qr + (size_t) t * QL, s);
        dense.quantize_acts(qr, T, QL, xq1, xs1, s);
        dense.gemv_q8(w.wq_b, Q, QL, xq1, xs1, T, q, s);
        dense.gemv_q8(w.wkv, D, HID, xq0, xs0, T, kvraw, s);
        if (La.mode == AttnMode::kFull) {
            dense.gemv_bf16(w.comp_wkv, D, HID, x, T, ckv, s);
            if (La.ratio > 1) dense.gemv_bf16(w.comp_wgate, D, HID, x, T, csc, s);
        }
        if (has_idx) {
            dense.gemv_q8(w.idx_wq_b, IQ, QL, xq1, xs1, T, iq, s);
            dense.gemv_bf16(w.idx_weights_proj, G::kIdxHeads, HID, x, T, iw, s);
        }
        // ---- the cache-dependent part, token by token in position order ----
        for (int t = 0; t < T; ++t) token(l, t, pos0 + t, s);
        // ---- output: grouped wo_a (FP32 activations), then wo_b (int8 activations) ----
        dense.gemv_q8_grouped_f32(w.wo_a, G::kOGroups, G::kOLora, OGIN, o_rot, T, omid, s);
        dense.quantize_acts(omid, T, OMID, xq2, xs2, s);
        dense.gemv_q8(w.wo_b, HID, OMID, xq2, xs2, T, out, s);
        La.seen += T;
    }
};

template <class G>
Ds41Attention<G>::Ds41Attention(Dev& dev, AttnDenseOps& dense) : impl_(new Impl(dev, dense)) {}
template <class G>
Ds41Attention<G>::~Ds41Attention() = default;
template <class G>
void Ds41Attention<G>::init(const Ds41LayerRoles& roles, int max_context, const AttnQuantFlags& quant, float norm_eps) {
    impl_->init(roles, max_context, quant, norm_eps);
}
template <class G>
void Ds41Attention<G>::set_weights(int layer, const AttnLayerWeights& w) {
    impl_->set_weights(layer, w);
}
template <class G>
void Ds41Attention<G>::set_quant(const AttnQuantFlags& q) {
    impl_->quant = q;
}
template <class G>
void Ds41Attention<G>::reset() {
    impl_->reset();
}
template <class G>
void Ds41Attention<G>::forward(int layer, const float* x_normed, int T, int pos0, float* out, Stream s) {
    impl_->forward(layer, x_normed, T, pos0, out, s);
}
template <class G>
const float* Ds41Attention<G>::trace_q(int t) const {
    return impl_->q + (size_t) t * Impl::Q;
}
template <class G>
const float* Ds41Attention<G>::trace_o(int t) const {
    return impl_->o_rot + (size_t) t * Impl::Q;
}
template <class G>
const float* Ds41Attention<G>::trace_latent(int t) const {
    return impl_->latent_valid[t] ? impl_->latent + (size_t) t * Impl::D : nullptr;
}
template <class G>
const int32_t* Ds41Attention<G>::trace_topk(int t) const {
    return impl_->topk + (size_t) t * G::kIdxTopK;
}
template <class G>
const float* Ds41Attention<G>::kv_win_row(int layer, int pos) const {
    return impl_->L.at((size_t) layer).win + (size_t) (pos % G::kWindow) * Impl::D;
}
template <class G>
const float* Ds41Attention<G>::comp_kv_row(int layer, int index) const {
    const auto& La = impl_->L.at((size_t) layer);
    return La.comp && index >= 0 && index < La.max_comp ? La.comp + (size_t) index * Impl::D : nullptr;
}
template <class G>
const float* Ds41Attention<G>::index_k_row(int layer, int index) const {
    const auto& La = impl_->L.at((size_t) layer);
    return La.idxk && index >= 0 && index < La.max_comp ? La.idxk + (size_t) index * Impl::ID : nullptr;
}
template <class G>
AttnMode Ds41Attention<G>::mode(int layer) const {
    return impl_->L.at((size_t) layer).mode;
}
template <class G>
int Ds41Attention<G>::compress_ratio(int layer) const {
    return impl_->L.at((size_t) layer).ratio;
}
template <class G>
int Ds41Attention<G>::max_context() const {
    return impl_->max_context;
}
template <class G>
size_t Ds41Attention<G>::device_bytes() const {
    return impl_->bytes;
}

}  // namespace strata::ds41::cuda
