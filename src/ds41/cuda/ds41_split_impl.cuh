// src/ds41/cuda/ds41_split_impl.cuh - DS-D: the GPU-cache hit / miss split of one layer's routing (V100, sm_70).  Included by
// ds41_split.cu (nvcc) and by the CPU emulation test (-DDS41_EMU).
//
// ONE BLOCK, any T.  The router's ids [T][6] are walked in (token, k) order in chunks of blockDim.x entries; for every entry the
// layer's residency row (int32 [384], slot or -1) says hit or miss; the hits and the misses are compacted with a warp-ballot
// prefix count plus the per-warp totals, so both lists come out in (token, k) order, run to run (no atomics, no races).
//   hits   [n_hits]   {token, k, slot, weight}
//   misses [n_misses] {token, k, expert, weight}     <- what the host's CPU pool reads
//   counts {n_hits, n_misses, n_groups, 0}           written last
// For T <= 8 the hits are then GROUPED by slot (groups != nullptr): hit i joins the group of the first hit with the same slot
// (= the same expert asked by several tokens) and a group lists its members in hit order.  Groups are numbered by DECREASING size
// (ties: order of first appearance), so the expert kernels can run the rare big groups (5..8 tokens) with a wide-token specialisation
// and everything else with a lean one, each launch looking at a prefix / the rest of the same list.
// A group holds at most 8 hits (a token never asks for the same expert twice, so 8 tokens give at most 8).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "ds41_math.cuh"
#include "strata/ds41/cuda/ds41_cuda.hpp"

namespace strata::ds41::cuda::dev {

inline constexpr int kSpMaxGroupedHits = kMaxExpertTokens * kTopK;     // 48

DS41_KERNEL DS41_LAUNCH_BOUNDS(1024) void split_kernel(const int32_t* DS41_RESTRICT ids, const float* DS41_RESTRICT weights, int T,
                                                       const int32_t* DS41_RESTRICT residency_row, HitEntry* DS41_RESTRICT hits,
                                                       MissEntry* DS41_RESTRICT misses, HitGroup* DS41_RESTRICT groups,
                                                       SplitCounts* DS41_RESTRICT counts, int host_visible) {
    DS41_SHARED int s_wh[32];
    DS41_SHARED int s_wm[32];
    DS41_SHARED int s_first[kSpMaxGroupedHits];
    DS41_SHARED int s_lead[kSpMaxGroupedHits];
    DS41_SHARED int s_gid[kSpMaxGroupedHits];
    DS41_SHARED int s_size[kSpMaxGroupedHits];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nthreads = blockDim.x;
    const int nwarps = nthreads >> 5;
    const int total = T * kTopK;

    int hit_base = 0, miss_base = 0;
    for (int c0 = 0; c0 < total; c0 += nthreads) {
        const int e = c0 + tid;
        const bool valid = e < total;
        int id = -1, slot = -1;
        if (valid) {
            id = ids[e];
            slot = (id >= 0 && id < kExperts) ? residency_row[id] : -1;      // an invalid id is a miss, passed on unchanged
        }
        const bool is_hit = valid && slot >= 0, is_miss = valid && slot < 0;
        const uint32_t bh = ballot(is_hit), bm = ballot(is_miss);
        const uint32_t below = (1u << lane) - 1u;
        const int ph = popc(bh & below), pm = popc(bm & below);
        if (lane == 0) {
            s_wh[warp] = popc(bh);
            s_wm[warp] = popc(bm);
        }
        sync_block();
        int oh = 0, om = 0, th = 0, tm = 0;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                oh += s_wh[w];
                om += s_wm[w];
            }
            th += s_wh[w];
            tm += s_wm[w];
        }
        if (is_hit) {
            HitEntry h;
            h.token = e / kTopK;
            h.k = e - h.token * kTopK;
            h.slot = slot;
            h.weight = weights[e];
            hits[hit_base + oh + ph] = h;
        } else if (is_miss) {
            MissEntry m;
            m.token = e / kTopK;
            m.k = e - m.token * kTopK;
            m.expert = id;
            m.weight = weights[e];
            misses[miss_base + om + pm] = m;
        }
        hit_base += th;
        miss_base += tm;
        sync_block();                                                          // s_wh / s_wm are reused by the next chunk
    }

    int n_groups = 0;
    if (groups != nullptr) {
        const int nh = hit_base;                                               // <= 48 when T <= 8 (the caller guarantees it)
        int slot_i = -1, pos = 0, first = tid;
        if (tid < nh) {
            slot_i = hits[tid].slot;
            for (int j = 0; j < tid; ++j)
                if (hits[j].slot == slot_i) {
                    if (pos == 0) first = j;
                    ++pos;
                }
            s_first[tid] = first;
            s_lead[tid] = pos == 0;
        }
        sync_block();
        if (tid < nh && pos == 0) {                                            // a group's leader: its size
            int n = 0;
            for (int j = tid; j < nh; ++j) n += s_first[j] == tid;
            s_size[tid] = n < kMaxExpertTokens ? n : kMaxExpertTokens;
        }
        sync_block();
        if (tid < nh && pos == 0) {                                            // its rank: larger groups first, ties by first appearance
            int gid = 0;
            for (int j = 0; j < nh; ++j)
                if (s_lead[j] && (s_size[j] > s_size[tid] || (s_size[j] == s_size[tid] && j < tid))) ++gid;
            s_gid[tid] = gid;
        }
        sync_block();
        if (tid < nh) {
            const int g = s_gid[first];
            if (pos < kMaxExpertTokens) groups[g].hit[pos] = tid;
            if (pos == 0) {
                groups[g].slot = slot_i;
                groups[g].n = s_size[tid];
            }
        }
        if (tid == 0)
            for (int j = 0; j < nh; ++j) n_groups += s_lead[j];
    }
    // The lists first, then (after a system-scope fence when the miss list / counts are mapped host memory) the counts.
    if (host_visible) threadfence_system();
    sync_block();
    if (tid == 0) {
        counts->n_hits = hit_base;
        counts->n_misses = miss_base;
        counts->n_groups = n_groups;
        if (host_visible) threadfence_system();            // `reserved` is the doorbell: written last, after the three counts
        counts->reserved = 0;
    }
}

}  // namespace strata::ds41::cuda::dev

namespace strata::ds41::cuda {

void split_hits_misses(const int32_t* ids, const float* weights, int T, const int32_t* residency_table, int layer, HitEntry* hits,
                       MissEntry* misses, HitGroup* groups, SplitCounts* counts, bool host_visible, void* stream) {
    if (T < 1) throw std::invalid_argument("split_hits_misses: T must be >= 1");
    if (layer < 0 || layer >= kLayers) throw std::invalid_argument("split_hits_misses: layer out of range");
    if (groups != nullptr && T > kMaxExpertTokens) throw std::invalid_argument("split_hits_misses: groups need T <= 8");
    const int total = T * kTopK;
    int threads = (total + 31) & ~31;
    threads = threads < 64 ? 64 : (threads > 1024 ? 1024 : threads);
    const int32_t* row = residency_table + (size_t) layer * kExperts;
#if defined(DS41_EMU)
    (void) stream;
    ds41_emu::launch(dim3(1), dim3((unsigned) threads), 0,
                     [&] { dev::split_kernel(ids, weights, T, row, hits, misses, groups, counts, host_visible ? 1 : 0); });
#else
    dev::split_kernel<<<1, threads, 0, (cudaStream_t) stream>>>(ids, weights, T, row, hits, misses, groups, counts, host_visible ? 1 : 0);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("ds41 split_hits_misses: ") + cudaGetErrorString(e));
#endif
}

}  // namespace strata::ds41::cuda
