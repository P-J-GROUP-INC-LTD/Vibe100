// src/ds41/cuda/ds41_split_impl.cuh - DS-D / DS1-G: the GPU-cache hit / miss split of one layer's routing (V100, sm_70), a template over the geometry G
// (geom.hpp: kExperts, kTopK, kLayers).  Included by ds41_split.cu (nvcc: RealGeom) and by ds41_emu_impl.cpp (-DDS41_EMU: RealGeom and MiniGeom).
//
// ONE BLOCK, any T.  The router's ids [T][6] are walked in (token, k) order in chunks of blockDim.x entries; for every entry the
// layer's residency row (int32 [384], slot or -1) says hit or miss - a value is a HIT only if 0 <= r < n_slots, anything else (-1 =
// not resident, or a corrupt / stale entry) is a MISS, and a corrupt one (neither -1 nor in range) is also COUNTED in
// counts->n_bad_slots so that the caller can assert; the hits and the misses are compacted with a warp-ballot
// prefix count plus the per-warp totals, so both lists come out in (token, k) order, run to run (no atomics, no races).
//   hits   [n_hits]   {token, k, slot, weight}                                  DEVICE
//   misses [n_misses] {token, k, expert, weight}     <- what the host's CPU pool reads   (device, or mapped host memory: the caller's choice)
//   counts {n_hits, n_misses, n_groups, n_bad_slots}                  DEVICE: what the expert kernels read (n_groups), once per block, from L2
//   host   {seq, n_hits, n_misses, n_groups, n_bad_slots}  optional, MAPPED HOST memory, one record per layer in flight: the same four
//          counts, then (after one fence by one thread) the sequence number `seq` stored with RELEASE semantics = the doorbell.
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

template <class G>
DS41_KERNEL DS41_LAUNCH_BOUNDS(1024) void split_kernel(const int32_t* DS41_RESTRICT ids, const float* DS41_RESTRICT weights, int T,
                                                       const int32_t* DS41_RESTRICT residency_row, int n_slots,
                                                       HitEntry* DS41_RESTRICT hits, MissEntry* DS41_RESTRICT misses,
                                                       HitGroup* DS41_RESTRICT groups, SplitCounts* DS41_RESTRICT counts,
                                                       SplitHostRecord* DS41_RESTRICT host, uint32_t seq) {
    // Enumerators, NOT local `constexpr int` variables: a local constexpr variable perturbs nvcc's front end enough that ptxas allocates this kernel's
    // registers differently (same instruction count, different order and register numbers: found by the SASS diff against the pre-template code).
    enum : int { kExperts = G::kExperts, kTopK = G::kTopK, kSpMaxGroupedHits = kMaxExpertTokens * G::kTopK };   // 48: the hits of 8 tokens (a group's members, the leader table)
    static_assert(kSpMaxGroupedHits <= 64, "split: the grouping runs one thread per hit of a block of >= 64 threads (8 tokens x kTopK <= 64)");
    DS41_SHARED int s_wh[32];
    DS41_SHARED int s_wm[32];
    DS41_SHARED int s_wb[32];
    DS41_SHARED int s_first[kSpMaxGroupedHits];
    DS41_SHARED int s_lead[kSpMaxGroupedHits];
    DS41_SHARED int s_gid[kSpMaxGroupedHits];
    DS41_SHARED int s_size[kSpMaxGroupedHits];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nthreads = blockDim.x;
    const int nwarps = nthreads >> 5;
    const int total = T * kTopK;

    int hit_base = 0, miss_base = 0, bad_base = 0;
    for (int c0 = 0; c0 < total; c0 += nthreads) {
        const int e = c0 + tid;
        const bool valid = e < total;
        int id = -1, slot = -1;
        bool bad = false;
        if (valid) {
            id = ids[e];
            const int r = (id >= 0 && id < kExperts) ? residency_row[id] : -1;      // an invalid id is a miss, passed on unchanged
            const bool in_range = r >= 0 && r < n_slots;
            slot = in_range ? r : -1;                                                // a hit ONLY inside [0, n_slots)
            bad = r != -1 && !in_range;                                              // -1 is the "not resident" marker, anything else out of range is corrupt
        }
        const bool is_hit = valid && slot >= 0, is_miss = valid && slot < 0;
        const uint32_t bh = ballot(is_hit), bm = ballot(is_miss), bb = ballot(bad);
        const uint32_t below = (1u << lane) - 1u;
        const int ph = popc(bh & below), pm = popc(bm & below);
        if (lane == 0) {
            s_wh[warp] = popc(bh);
            s_wm[warp] = popc(bm);
            s_wb[warp] = popc(bb);
        }
        sync_block();
        int oh = 0, om = 0, th = 0, tm = 0, tb = 0;
        for (int w = 0; w < nwarps; ++w) {
            if (w < warp) {
                oh += s_wh[w];
                om += s_wm[w];
            }
            th += s_wh[w];
            tm += s_wm[w];
            tb += s_wb[w];
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
        bad_base += tb;
        sync_block();                                                          // s_wh / s_wm / s_wb are reused by the next chunk
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
    // The lists first (every thread's stores, ordered before thread 0's by the barrier), then the counts.  The expert kernels read the counts
    // from DEVICE memory.  When the caller wants the result on the host, thread 0 also writes the host record, issues ONE fence.acq_rel.sys
    // and stores the sequence number with release semantics: a host that reads `seq` with an acquire load and finds the value it expects sees
    // the counts and every entry of the miss list (written, if `misses` is mapped host memory, before the barrier).  No other thread fences.
    sync_block();
    if (tid == 0) {
        counts->n_hits = hit_base;
        counts->n_misses = miss_base;
        counts->n_groups = n_groups;
        counts->n_bad_slots = bad_base;
        if (host != nullptr) {
            host->n_hits = hit_base;
            host->n_misses = miss_base;
            host->n_groups = n_groups;
            host->n_bad_slots = bad_base;
            fence_acq_rel_sys();
            store_release_sys(&host->seq, seq);
        }
    }
}

}  // namespace strata::ds41::cuda::dev

namespace strata::ds41::cuda {

template <class G>
void split_hits_misses(const int32_t* ids, const float* weights, int T, const int32_t* residency_table, int n_slots, int layer, HitEntry* hits,
                       MissEntry* misses, HitGroup* groups, SplitCounts* counts, SplitHostRecord* host, uint32_t seq, void* stream) {
    static_assert(G::kLayers >= 1 && G::kExperts >= 1 && G::kTopK >= 1, "split: a model has layers, experts and a top-k");
    if (T < 1) throw std::invalid_argument("split_hits_misses: T must be >= 1");
    if (layer < 0 || layer >= G::kLayers) throw std::invalid_argument("split_hits_misses: layer out of range");
    if (n_slots < 0) throw std::invalid_argument("split_hits_misses: n_slots must be >= 0 (the number of cache slots the residency table may name)");
    if (groups != nullptr && T > kMaxExpertTokens) throw std::invalid_argument("split_hits_misses: groups need T <= 8");
    if (host != nullptr && seq == 0) throw std::invalid_argument("split_hits_misses: seq 0 is the cleared state of a host record; start at 1");
    if (host != nullptr && reinterpret_cast<uintptr_t>(host) % 16 != 0) throw std::invalid_argument("split_hits_misses: the host record must be 16-byte aligned");
    const int total = T * G::kTopK;
    int threads = (total + 31) & ~31;
    threads = threads < 64 ? 64 : (threads > 1024 ? 1024 : threads);
    const int32_t* row = residency_table + (size_t) layer * G::kExperts;
    dev::launch(dev::split_kernel<G>, dim3(1), dim3((unsigned) threads), 0, stream, ids, weights, T, row, n_slots, hits, misses, groups, counts, host, seq);
    dev::check_launch("split_hits_misses");
}

}  // namespace strata::ds41::cuda

/// Explicit instantiation of the split's host entry point for geometry G (the .cu: RealGeom; the emulator build: RealGeom and MiniGeom).
#define DS41_INSTANTIATE_SPLIT(G)                                                                                                                        \
    template void ::strata::ds41::cuda::split_hits_misses<G>(const int32_t*, const float*, int, const int32_t*, int, int, ::strata::ds41::cuda::HitEntry*, \
                                                             ::strata::ds41::cuda::MissEntry*, ::strata::ds41::cuda::HitGroup*, ::strata::ds41::cuda::SplitCounts*, \
                                                             ::strata::ds41::cuda::SplitHostRecord*, uint32_t, void*);
