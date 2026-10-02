// src/ds41/session/session_impl.hpp - DS1-E: Ds41Session<G> (include/strata/ds41/session/session.hpp).  Included by the translation units that instantiate it
// (session_real.cpp: RealGeom, linked with the nvcc kernels; the mini emulator program: MiniGeom, linked with the emulated kernels).
//
// ORACLE LINE MAP (ref/ds41/model.py; the step numbers are the comments in run_window below)
//    forward 150            hasher(ids, start_pos)                         step 1   NgramHasher::push_many
//    forward 151            embedding row                                  step 2   token_row_f32 -> device
//    forward 152-154        np.repeat(h, hc_mult); pre_mix = [1, 0, 0, 0]  step 3   ds41_hc_expand, ds41_hc_begin
//    forward 156-164        Engram (on engram_layer_ids)                   step 4   EngramRunner::run
//    block   102-104        hc_mixes(attn), hc_pre(x, pre_mix)             step 5   ds41_hc_attn_in           (the one-block lag lives in HcRecords)
//    block   105            attn_norm                                      step 6   ds41_rmsnorm
//    block   106-110        attention_layer                                step 7   Ds41Attention::forward
//    block   111            hc_post(attention)                             step 8   ds41_hc_attn_out
//    block   115-117        hc_mixes(ffn), hc_pre(x, attn pre)             step 9   ds41_hc_ffn_in
//    block   118            ffn_norm                                       step 10  ds41_rmsnorm
//    block   119-121        router                                         step 11  router_forward
//    block   122, moe 87-96 routed experts + shared expert, FP32 sum       step 12  split -> [GPU hits || CPU pool] + shared expert -> ds41_moe_combine
//    block   131            hc_post(ffn)                                   step 13  ds41_hc_ffn_out
//    forward 170-177        hc_pre(h, pre_mix), norm, head                 step 14  ds41_hc_head_fold, ds41_rmsnorm, ds41_head, ds41_argmax
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/ds41/cpu/mxfp4_expert.hpp"
#include "strata/ds41/cuda/attn.hpp"
#include "strata/ds41/cuda/attn_dense.hpp"
#include "strata/ds41/cuda/dense.hpp"
#include "strata/ds41/cuda/ds41_cuda.hpp"
#include "strata/ds41/cuda/engram.hpp"
#include "strata/ds41/cuda/mhc.hpp"
#include "strata/ds41/session/moe_combine.hpp"
#include "strata/ds41/session/session.hpp"

namespace strata::ds41::session {

template <class G>
struct Ds41Session<G>::Impl {
    using Clock = std::chrono::steady_clock;
    static constexpr int H = G::kHidden, HC = G::kHc, V = G::kVocab, K = G::kTopK, L = G::kLayers;

    cuda::Dev& dev;
    model::Ds41Model<G>& m;
    const model::Ds41Config& cfg;
    SessionOptions opt;
    int tmax = 1;
    int pos = 0;
    bool broken = false;
    SessionStats st;
    TraceWriter* trace = nullptr;
    Clock::time_point t_last;

    // ---- device allocations (released in the destructor, in reverse order)
    std::vector<void*> dev_ptrs, mapped_ptrs;
    uint64_t own_bytes = 0;
    template <class T> T* dalloc(size_t n) {
        T* p = static_cast<T*>(dev.alloc(std::max<size_t>(n, 1) * sizeof(T)));
        dev_ptrs.push_back(p);
        own_bytes += std::max<size_t>(n, 1) * sizeof(T);
        return p;
    }
    template <class T> T* malloc_mapped(size_t n) {
        T* p = static_cast<T*>(dev.alloc_mapped(std::max<size_t>(n, 1) * sizeof(T)));
        mapped_ptrs.push_back(p);
        return p;
    }

    // ---- the stream and the per-step scratch
    float *emb = nullptr, *x = nullptr, *y = nullptr, *xn = nullptr, *attn_out = nullptr, *ffn_out = nullptr, *hfold = nullptr, *hn = nullptr, *logits_dev = nullptr;
    float *shared_y = nullptr, *parts = nullptr, *router_logits = nullptr, *wts = nullptr, *cpu_rows_dev = nullptr, *argmax_val_dev = nullptr;
    int32_t *ids = nullptr, *cpu_row_idx_dev = nullptr, *argmax_dev = nullptr;
    cuda::HitEntry* hits = nullptr;
    cuda::HitGroup* groups = nullptr;
    cuda::SplitCounts* counts = nullptr;
    void* hc_rec_mem = nullptr;
    void* hc_ws = nullptr;
    size_t hc_ws_bytes = 0;
    cuda::ExpertScratch exp_scratch{};
    cuda::SharedExpertScratch sh_scratch{};
    uint8_t* dummy_cache = nullptr;
    // mapped (pinned) host memory the device writes or reads while the host works
    float* xn_host = nullptr;
    float* cpu_rows_host = nullptr;
    int32_t* cpu_idx_host = nullptr;
    cuda::SplitHostRecord* split_rec = nullptr;       // [L]: one record per layer in flight
    cuda::MissEntry* miss_list = nullptr;             // [L][K * tmax]
    void* xn_event = nullptr;
    std::vector<uint32_t> seq;                        // per layer: the last doorbell sequence number used

    // ---- the model's parts
    model::DeviceRope rope;
    cuda::Ds41AttnDense<G> dense;
    cuda::Ds41Attention<G> attn;
    cuda::HcParams hcp;
    std::unique_ptr<cuda::NgramHasher> hasher;
    std::unique_ptr<cuda::EngramRunner<G>> runner;
    struct EngramLayer {
        cuda::EngramLayerWeights w;
        cuda::EngramWkvFn wkv;
    };
    std::vector<EngramLayer> eng;                     // per Engram slot
    std::vector<int64_t> hash_rows;                   // [tmax][n_engram][rows_per_layer]
    std::unique_ptr<CpuExpertPool> pool;
    std::vector<cpu::ActQ> actq;
    std::vector<CpuMiss> cpu_miss;

    // ---- host staging
    std::vector<float> host_stage, trace_tmp, logits_host;
    std::vector<int32_t> argmax_host, itmp;
    Want last_want = Want::kNone;

    Impl(cuda::Dev& d, model::Ds41Model<G>& model, const SessionOptions& o)
        : dev(d), m(model), cfg(model.config()), opt(o), rope(d, model.config(), std::max(1, o.max_context)), dense(d), attn(d, dense) {
        tmax = opt.max_window;
        if (tmax < 1 || tmax > cuda::Ds41Attention<G>::kMaxT) throw std::invalid_argument("Ds41Session: max_window must be 1.." + std::to_string(cuda::Ds41Attention<G>::kMaxT));
        if (opt.max_context < 1) throw std::invalid_argument("Ds41Session: max_context must be >= 1");
        if (cfg.n_layer != L || cfg.hidden != H) throw std::invalid_argument("Ds41Session: the model is not this geometry");
        if (!m.weights.token_embd || !m.weights.head || !m.weights.output_norm) throw std::invalid_argument("Ds41Session: the model has no embedding / head / output norm");
        hcp.hc_eps = cfg.hc_eps;
        hcp.norm_eps = cfg.rms_eps;
        try {
            allocate();
            wire_attention();
            wire_engram();
            wire_pool();
        } catch (...) {
            release_all();
            throw;
        }
        seq.assign((size_t) L, 0u);
        t_last = Clock::now();
    }
    ~Impl() { release_all(); }

    void release_all() {
        if (xn_event) {
            dev.event_destroy(xn_event);
            xn_event = nullptr;
        }
        pool.reset();
        runner.reset();
        for (auto it = dev_ptrs.rbegin(); it != dev_ptrs.rend(); ++it) dev.release(*it);
        dev_ptrs.clear();
        for (auto it = mapped_ptrs.rbegin(); it != mapped_ptrs.rend(); ++it) dev.release_mapped(*it);
        mapped_ptrs.clear();
    }

    // ================================================================================================ construction
    void allocate() {
        const size_t T = (size_t) tmax;
        emb = dalloc<float>(T * H);
        x = dalloc<float>(T * HC * H);
        y = dalloc<float>(T * H);
        xn = dalloc<float>(T * H);
        attn_out = dalloc<float>(T * H);
        ffn_out = dalloc<float>(T * H);
        hfold = dalloc<float>(T * H);
        hn = dalloc<float>(T * H);
        logits_dev = dalloc<float>(T * V);
        shared_y = dalloc<float>(T * H);
        parts = dalloc<float>(T * K * H);
        router_logits = dalloc<float>(T * G::kExperts);
        ids = dalloc<int32_t>(T * K);
        wts = dalloc<float>(T * K);
        hits = dalloc<cuda::HitEntry>(T * K);
        groups = dalloc<cuda::HitGroup>(T * K);
        counts = dalloc<cuda::SplitCounts>(1);
        hc_rec_mem = dalloc<uint8_t>(cuda::hc_records_bytes<G>(tmax));
        hc_ws_bytes = cuda::hc_scratch_bytes<G>(tmax);
        hc_ws = dalloc<uint8_t>(hc_ws_bytes);
        exp_scratch = cuda::expert_scratch_carve<G>(dalloc<uint8_t>(cuda::expert_scratch_bytes<G>(tmax)), tmax);
        sh_scratch = cuda::shared_expert_scratch_carve<G>(dalloc<uint8_t>(cuda::shared_expert_scratch_bytes<G>(tmax)), tmax);
        cpu_rows_dev = dalloc<float>(T * K * H);
        cpu_row_idx_dev = dalloc<int32_t>(T * K);
        argmax_dev = dalloc<int32_t>(T);
        argmax_val_dev = dalloc<float>(T);
        dummy_cache = dalloc<uint8_t>(256);                       // the cache base when the cache has no slots (no hit ever names one)
        xn_host = malloc_mapped<float>(T * H);
        cpu_rows_host = malloc_mapped<float>(T * K * H);
        cpu_idx_host = malloc_mapped<int32_t>(T * K);
        split_rec = malloc_mapped<cuda::SplitHostRecord>((size_t) L);
        miss_list = malloc_mapped<cuda::MissEntry>((size_t) L * T * K);
        xn_event = dev.event_create();
        actq.resize((size_t) tmax);
        cpu_miss.resize((size_t) tmax * K);
        argmax_host.assign((size_t) tmax, 0);
        host_stage.resize((size_t) tmax * H);
        dev.fill(counts, 0, sizeof(cuda::SplitCounts));
    }

    void wire_attention() {
        cuda::AttnQuantFlags flags;
        flags.window_kv = opt.window_kv;
        flags.compressed_kv = opt.compressed_kv;
        flags.index = opt.index;
        attn.init(cuda::ds41_attn_roles(cfg), opt.max_context, flags, cfg.rms_eps);
        for (int l = 0; l < L; ++l) {
            const model::LayerWeights& lw = m.weights.at(l);
            cuda::AttnLayerWeights aw;
            aw.wq_a = lw.wq_a.p;
            aw.q_norm = lw.q_norm.template as<float>();
            aw.wq_b = lw.wq_b.p;
            aw.wkv = lw.wkv.p;
            aw.kv_norm = lw.kv_norm.template as<float>();
            aw.sink = lw.attn_sinks.template as<float>();
            aw.wo_a = lw.wo_a.p;
            aw.wo_b = lw.wo_b.p;
            aw.rope.cos = rope.cos(l);
            aw.rope.sin = rope.sin(l);
            if (lw.comp_kv) aw.comp_wkv = lw.comp_kv.template as<uint16_t>();
            if (lw.comp_gate) aw.comp_wgate = lw.comp_gate.template as<uint16_t>();
            if (lw.comp_norm) aw.comp_norm = lw.comp_norm.template as<float>();
            if (lw.idx_comp_kv) aw.idx_wk = lw.idx_comp_kv.template as<uint16_t>();
            if (lw.idx_comp_norm) aw.idx_k_norm = lw.idx_comp_norm.template as<float>();
            if (lw.idx_q_b) aw.idx_wq_b = lw.idx_q_b.p;
            if (lw.idx_proj) aw.idx_weights_proj = lw.idx_proj.template as<uint16_t>();
            attn.set_weights(l, aw);
        }
    }

    void wire_engram() {
        if (cfg.n_engram() == 0) return;
        hasher.reset(new cuda::NgramHasher(cuda::engram_constants_from(cfg.engram)));
        runner.reset(new cuda::EngramRunner<G>(dev, tmax));
        cuda::Dev& d = dev;
        const cuda::EngramGemvQ8Fn gemv = [&d](const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* yy, cuda::Stream s) {
            cuda::ds41_gemv_q8_int8<G>(d, w, n, k, xq, xs, T, yy, s);
        };
        eng.resize((size_t) cfg.n_engram());
        for (int l = 0; l < L; ++l) {
            const model::LayerInfo& li = cfg.layer(l);
            if (!li.is_engram()) continue;
            const model::LayerWeights& lw = m.weights.at(l);
            EngramLayer& el = eng[(size_t) li.engram_slot];
            el.w.table = model::engram_table_view<cuda::EngramTableView>(lw);
            el.w.q_bf16 = lw.eng_q.template as<uint16_t>();
            el.w.k_bf16 = lw.eng_k.template as<uint16_t>();
            el.wkv = cuda::engram_wkv_q8<G>(dev, lw.eng_wkv.p, runner->xq(), runner->xs(), gemv);
        }
        hash_rows.assign((size_t) tmax * hasher->n_layers() * hasher->rows_per_layer(), 0);
    }

    void wire_pool() {
        if (!opt.use_cpu_pool || !m.arena.built()) return;
        CpuPoolOptions po = opt.pool;
        po.max_tokens = tmax;
        po.top_k = K;
        pool.reset(new CpuExpertPool(m.arena, po));
    }

    // ================================================================================================ small helpers
    static double secs(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }
    void lap(int stage) {
        if (opt.stage_timing) dev.sync();
        const Clock::time_point now = Clock::now();
        st.seconds[stage] += secs(t_last, now);
        t_last = now;
    }
    /// Time spent in the trace code is not charged to any stage.
    struct TraceScope {
        Impl& s;
        Clock::time_point t0;
        explicit TraceScope(Impl& im) : s(im), t0(Clock::now()) {}
        ~TraceScope() {
            const Clock::time_point t1 = Clock::now();
            s.st.trace_s += secs(t0, t1);
            s.t_last += (t1 - t0);
        }
    };
    void tr_f32(const char* stage, int p, int layer, const float* dptr, const std::vector<int64_t>& shape) {
        size_t n = 1;
        for (int64_t d : shape) n *= (size_t) d;
        trace_tmp.resize(n);
        dev.d2h(trace_tmp.data(), dptr, n * sizeof(float));
        trace->put_f32(stage, p, layer, trace_tmp.data(), shape);
    }
    void tr_i32(const char* stage, int p, int layer, const int32_t* dptr, int64_t n) {
        itmp.resize((size_t) n);
        dev.d2h(itmp.data(), dptr, (size_t) n * sizeof(int32_t));
        trace->put_i32(stage, p, layer, itmp.data(), n);
    }

    // ================================================================================================ the step
    void run_window(const int32_t* tokens, int T, Want want) {
        const cuda::Stream s = dev.stream();
        const int p0 = pos;
        const Clock::time_point t_step = Clock::now();
        t_last = t_step;
        for (int t = 0; t < T; ++t)
            if (tokens[t] < 0 || tokens[t] >= V) throw std::out_of_range("Ds41Session::step: token id " + std::to_string(tokens[t]) + " is outside the vocabulary of " + std::to_string(V));

        // step 1: the n-gram hasher takes the tokens (model.py:150); step 2: the embedding rows (151)
        if (hasher) hasher->push_many(tokens, T, hash_rows.data());
        for (int t = 0; t < T; ++t) m.weights.token_row_f32(tokens[t], host_stage.data() + (size_t) t * H);
        dev.h2d(emb, host_stage.data(), (size_t) T * H * sizeof(float));
        // step 3: four copies of the embedding, pre_mix = [1, 0, 0, 0] (152-154)
        cuda::ds41_hc_expand<G>(dev, emb, T, x, s);
        cuda::HcRecords<G> rec(hc_rec_mem, T);
        cuda::ds41_hc_begin<G>(dev, rec, T, s);
        lap(SessionStats::kEmbed);
        if (trace) {
            TraceScope ts(*this);
            for (int t = 0; t < T; ++t) tr_f32("embed", p0 + t, -1, emb + (size_t) t * H, {H});
        }

        for (int l = 0; l < L; ++l) {
            const model::LayerWeights& lw = m.weights.at(l);
            const model::LayerInfo& li = cfg.layer(l);

            // step 4: Engram, before the block, rewriting the stream in place (156-164)
            if (li.is_engram()) {
                EngramLayer& el = eng[(size_t) li.engram_slot];
                const int per = hasher->rows_per_layer();
                runner->run(el.w, hash_rows.data() + (size_t) li.engram_slot * per, T, (int64_t) hasher->n_layers() * per, x, el.wkv, cfg.rms_eps, s);
                lap(SessionStats::kEngram);
                if (trace) {
                    TraceScope ts(*this);
                    for (int t = 0; t < T; ++t) tr_f32("engram_out", p0 + t, l, x + (size_t) t * HC * H, {HC, H});
                }
            }

            // ---- attention sub-layer.  step 5: mixes of x, hc_pre with the LAGGED pre (102-104); step 6: attn_norm (105)
            cuda::ds41_hc_attn_in<G>(dev, x, cuda::hc_weights_of(lw.hc_attn_fn, lw.hc_attn_scale, lw.hc_attn_base), T, hcp, rec, y, hc_ws, hc_ws_bytes, s);
            cuda::ds41_rmsnorm<G>(dev, y, lw.attn_norm.template as<float>(), T, H, xn, cfg.rms_eps, s);
            lap(SessionStats::kAttnIn);
            if (trace) {
                TraceScope ts(*this);
                for (int t = 0; t < T; ++t) tr_f32("attn_in", p0 + t, l, xn + (size_t) t * H, {H});
            }
            // step 7: attention (106-110)
            attn.forward(l, xn, T, p0, attn_out, s);
            lap(SessionStats::kAttention);
            if (trace) {
                TraceScope ts(*this);
                trace_attention(l, T, p0);
            }
            // step 8: x <- hc_post(attention, x) (111)
            cuda::ds41_hc_attn_out<G>(dev, attn_out, x, T, rec, s);
            lap(SessionStats::kAttnOut);

            // ---- FFN sub-layer.  step 9: mixes of the NEW x, hc_pre with the attention's own pre (115-117); step 10: ffn_norm (118)
            cuda::ds41_hc_ffn_in<G>(dev, x, cuda::hc_weights_of(lw.hc_ffn_fn, lw.hc_ffn_scale, lw.hc_ffn_base), T, hcp, rec, y, hc_ws, hc_ws_bytes, s);
            cuda::ds41_rmsnorm<G>(dev, y, lw.ffn_norm.template as<float>(), T, H, xn, cfg.rms_eps, s);
            if (pool) {                                                 // the CPU pool quantises x itself: a copy of it to pinned host memory, ordered before the router
                dev.d2h_async(xn_host, xn, (size_t) T * H * sizeof(float), s);
                dev.event_record(xn_event, s);
            }
            lap(SessionStats::kFfnIn);
            if (trace) {
                TraceScope ts(*this);
                for (int t = 0; t < T; ++t) tr_f32("ffn_in", p0 + t, l, xn + (size_t) t * H, {H});
            }

            // step 11: router (119-121); step 12: the split (hits -> the GPU, misses -> the CPU) with its doorbell
            cuda::router_forward<G>(xn, lw.gate.template as<uint16_t>(), lw.gate_bias.template as<float>(), T, router_logits, ids, wts, s);
            cuda::SplitHostRecord* rec_h = split_rec + l;
            cuda::MissEntry* miss_h = miss_list + (size_t) l * tmax * K;
            const uint32_t sq = seq[(size_t) l] = cuda::split_next_seq(seq[(size_t) l]);
            cuda::split_hits_misses<G>(ids, wts, T, m.cache.residency(), m.cache.n_slots(), l, hits, miss_h, groups, counts, rec_h, sq, s);
            lap(SessionStats::kRouterSplit);

            // the doorbell: the host learns the misses the moment the split is done, while the GPU goes on
            const Clock::time_point tw0 = Clock::now();
            if (!cuda::split_host_wait(*rec_h, sq, opt.doorbell_timeout_s)) {
                dev.sync();                                            // surfaces a failed kernel (CudaDev: exits with the CUDA error)
                throw std::runtime_error("Ds41Session: layer " + std::to_string(l) + ": the split's doorbell never rang (waited " + std::to_string(opt.doorbell_timeout_s) + " s)");
            }
            st.doorbell_wait_s += secs(tw0, Clock::now());
            const int n_hits = rec_h->n_hits, n_miss = rec_h->n_misses;
            if (rec_h->n_bad_slots != 0)
                throw std::runtime_error("Ds41Session: layer " + std::to_string(l) + ": the residency table holds " + std::to_string(rec_h->n_bad_slots) + " corrupt slot entries");
            if (n_hits < 0 || n_miss < 0 || n_hits + n_miss != T * K)
                throw std::runtime_error("Ds41Session: layer " + std::to_string(l) + ": the split reports " + std::to_string(n_hits) + " hits + " + std::to_string(n_miss) + " misses for " + std::to_string(T * K) + " activations");
            st.hits += (uint64_t) n_hits;
            st.misses += (uint64_t) n_miss;

            // the CPU's share starts now: quantise x with DS-C's quantiser (bit-identical to the GPU's), hand the misses to the pool
            if (n_miss > 0) {
                if (!pool)
                    throw std::runtime_error("Ds41Session: layer " + std::to_string(l) + ": " + std::to_string(n_miss) +
                                             " routed experts are not resident in the GPU cache and there is no CPU expert pool (the expert arena was not built)");
                dev.event_sync(xn_event);
                cpu::quantize_acts(xn_host, H, T, actq.data());
                for (int i = 0; i < n_miss; ++i) cpu_miss[(size_t) i] = CpuMiss{miss_h[i].token, miss_h[i].k, miss_h[i].expert, miss_h[i].weight};
                pool->start(l, actq.data(), T, cpu_miss.data(), n_miss);
                ++st.layers_with_misses;
            }
            // meanwhile the GPU: the hits (DS-D's experts) and the shared expert (DS1-B), enqueued behind the split
            if (n_hits > 0)
                cuda::experts_hits<G>(m.cache.n_slots() > 0 ? m.cache.slots() : dummy_cache, xn, hits, groups, counts, T, exp_scratch, parts, s);
            cuda::SharedExpertWeights sw{lw.sh_gate.p, lw.sh_up.p, lw.sh_down.p};
            cuda::ds41_shared_expert<G>(dev, sw, xn, T, shared_y, sh_scratch, true, cfg.swiglu_limit, s);
            // the CPU's result: both sockets' partials added, uploaded behind the hits
            std::fill(cpu_idx_host, cpu_idx_host + (size_t) T * K, -1);
            if (n_miss > 0) {
                const Clock::time_point tp0 = Clock::now();
                pool->wait(cpu_rows_host);
                st.cpu_pool_s += secs(tp0, Clock::now());
                for (int i = 0; i < n_miss; ++i) cpu_idx_host[(size_t) miss_h[i].token * K + miss_h[i].k] = i;
                dev.h2d_async(cpu_rows_dev, cpu_rows_host, (size_t) n_miss * H * sizeof(float), s);
            }
            dev.h2d_async(cpu_row_idx_dev, cpu_idx_host, (size_t) T * K * sizeof(int32_t), s);
            lap(SessionStats::kExperts);

            // the FP32 sum in the oracle's order (moe.py 87-96), then step 13: x <- hc_post(ffn, x), and the lag moves on (131)
            cuda::ds41_moe_combine<G>(dev, ids, parts, cpu_row_idx_dev, cpu_rows_dev, shared_y, T, ffn_out, s);
            cuda::ds41_hc_ffn_out<G>(dev, ffn_out, x, T, rec, s);
            lap(SessionStats::kCombine);
            st.layer_steps += (uint64_t) T;
            if (trace) {
                TraceScope ts(*this);
                for (int t = 0; t < T; ++t) {
                    tr_i32("router_idx", p0 + t, l, ids + (size_t) t * K, K);
                    tr_f32("router_w", p0 + t, l, wts + (size_t) t * K, {K});
                    tr_f32("ffn_out", p0 + t, l, ffn_out + (size_t) t * H, {H});
                    tr_f32("block_out", p0 + t, l, x + (size_t) t * HC * H, {HC, H});
                    tr_f32("pre_mix", p0 + t, l, rec.lag + (size_t) t * G::kHcMixes, {HC});
                }
            }
        }

        // step 14: the final fold with the last FFN's pre, output_norm, the head (170-177)
        const bool head = want != Want::kNone || trace != nullptr;
        if (head) {
            cuda::ds41_hc_head_fold<G>(dev, x, rec, T, hfold, s);
            cuda::ds41_rmsnorm<G>(dev, hfold, m.weights.output_norm.template as<float>(), T, H, hn, cfg.rms_eps, s);
            cuda::ds41_head<G>(dev, m.weights.head.template as<uint16_t>(), hn, T, logits_dev, s);
            cuda::ds41_argmax<G>(dev, logits_dev, T, argmax_dev, argmax_val_dev, s);
            dev.d2h(argmax_host.data(), argmax_dev, (size_t) T * sizeof(int32_t));        // blocking: the stream has finished the step
            const bool need_logits = want == Want::kLogits || trace != nullptr;
            if (need_logits) {
                logits_host.resize((size_t) tmax * V);
                dev.d2h(logits_host.data(), logits_dev, (size_t) T * V * sizeof(float));
            }
            lap(SessionStats::kHead);
            if (trace) {
                TraceScope ts(*this);
                for (int t = 0; t < T; ++t) {
                    tr_f32("final_hidden", p0 + t, -1, hn + (size_t) t * H, {H});
                    trace->put_f32("logits", p0 + t, -1, logits_host.data() + (size_t) t * V, {V});
                }
            }
        } else {
            lap(SessionStats::kHead);
        }
        last_want = head ? want : Want::kNone;
        pos += T;
        st.tokens += (uint64_t) T;
        st.total_s += secs(t_step, Clock::now());
    }

    /// The attention's own stages for layer l (DS1.md section 6 + DS1-F's optional ones), per window row.
    void trace_attention(int l, int T, int p0) {
        const model::LayerInfo& li = cfg.layer(l);
        const int ratio = li.ratio;
        for (int t = 0; t < T; ++t) {
            const int p = p0 + t;
            tr_f32("q", p, l, attn.trace_q(t), {G::kHeads, G::kHeadDim});
            tr_f32("attn_o", p, l, attn.trace_o(t), {G::kHeads, G::kHeadDim});
            tr_f32("kv_win", p, l, attn.kv_win_row(l, p), {G::kHeadDim});
            if (ratio > 0) tr_i32("topk", p, l, attn.trace_topk(t), G::kIdxTopK);
            if (li.has_compressor && ratio > 0 && (p + 1) % ratio == 0) {
                if (const float* c = attn.comp_kv_row(l, p / ratio)) tr_f32("latent", p, l, c, {G::kHeadDim});
                if (const float* k = attn.index_k_row(l, p / ratio)) tr_f32("index_k", p, l, k, {G::kIdxDim});
                if (const float* lp = attn.trace_latent(t)) tr_f32("latent_pre", p, l, lp, {G::kHeadDim});
            }
        }
        if (T == 1) {                                                   // these accessors describe the LAST window row only
            int n = 0;
            if (li.has_indexer)
                if (const float* sc = attn.trace_scores(&n))
                    if (n > 0) tr_f32("index_scores", p0, l, sc, {n});
            if (li.is_candidate_source) {
                int nbs = 0;
                if (const float* bs = attn.trace_block_scores(&nbs))
                    if (nbs > 0) tr_f32("block_scores", p0, l, bs, {nbs});
                int nb = 0;
                if (const uint8_t* flags = attn.trace_cand(0, &nb)) {
                    std::vector<uint8_t> f((size_t) nb);
                    dev.d2h(f.data(), flags, (size_t) nb);
                    std::vector<int32_t> sel;
                    for (int b = 0; b < nb; ++b)
                        if (f[(size_t) b]) sel.push_back(b);
                    trace->put_i32("cand_blocks", p0, l, sel.data(), (int64_t) sel.size());
                }
            }
        }
    }
};

// ================================================================================================ the public class
template <class G>
Ds41Session<G>::Ds41Session(cuda::Dev& dev, model::Ds41Model<G>& model, const SessionOptions& opt) : impl_(new Impl(dev, model, opt)) {}
template <class G>
Ds41Session<G>::~Ds41Session() = default;
template <class G>
const SessionOptions& Ds41Session<G>::options() const {
    return impl_->opt;
}
template <class G>
int Ds41Session<G>::position() const {
    return impl_->pos;
}
template <class G>
void Ds41Session<G>::reset() {
    impl_->attn.reset();
    if (impl_->hasher) impl_->hasher->reset();
    impl_->pos = 0;
    impl_->broken = false;
    impl_->last_want = Want::kNone;
}
template <class G>
void Ds41Session<G>::step(const int32_t* tokens, int T, Want want) {
    Impl& s = *impl_;
    if (s.broken) throw std::logic_error("Ds41Session::step: an earlier step failed; call reset()");
    if (T < 1 || T > s.tmax) throw std::invalid_argument("Ds41Session::step: T must be 1.." + std::to_string(s.tmax));
    if (s.pos + T > s.opt.max_context)
        throw std::out_of_range("Ds41Session::step: position " + std::to_string(s.pos + T - 1) + " is beyond max_context " + std::to_string(s.opt.max_context) + " (raise --max-context)");
    try {
        s.run_window(tokens, T, want);
    } catch (...) {
        s.broken = true;
        throw;
    }
}
template <class G>
int32_t Ds41Session<G>::argmax(int t) const {
    if (impl_->last_want == Want::kNone) throw std::logic_error("Ds41Session::argmax: the last step did not run the head");
    return impl_->argmax_host.at((size_t) t);
}
template <class G>
const float* Ds41Session<G>::logits(int t) const {
    if (impl_->last_want != Want::kLogits) throw std::logic_error("Ds41Session::logits: the last step was not asked for logits");
    return impl_->logits_host.data() + (size_t) t * G::kVocab;
}
template <class G>
void Ds41Session<G>::set_trace(TraceWriter* trace) {
    impl_->trace = trace;
}
template <class G>
const SessionStats& Ds41Session<G>::stats() const {
    return impl_->st;
}
template <class G>
void Ds41Session<G>::reset_stats() {
    impl_->st = SessionStats();
}
template <class G>
uint64_t Ds41Session<G>::device_bytes() const {
    return impl_->own_bytes + impl_->attn.device_bytes() + impl_->rope.device_bytes();
}
template <class G>
const CpuExpertPool* Ds41Session<G>::pool() const {
    return impl_->pool.get();
}
template <class G>
std::string Ds41Session<G>::describe() const {
    const Impl& s = *impl_;
    char b[640];
    std::snprintf(b, sizeof b, "session: %s geometry, max_context %d, window %d, KV fake-quant window/compressed/index = %s/%s/%s; own device memory %.2f GiB (scratch %.1f MiB, attention caches %.2f GiB, RoPE %.1f MiB)\n",
                  G::kName, s.opt.max_context, s.tmax, s.opt.window_kv ? "on" : "off", s.opt.compressed_kv ? "on" : "off", s.opt.index ? "on" : "off",
                  (double) device_bytes() / (1ull << 30), (double) s.own_bytes / (1 << 20), (double) s.attn.device_bytes() / (1ull << 30), (double) s.rope.device_bytes() / (1 << 20));
    std::string r = b;
    r += s.pool ? "  " + s.pool->describe() : std::string("  no CPU expert pool (every routed expert must be resident in the GPU cache)");
    return r;
}

}  // namespace strata::ds41::session
