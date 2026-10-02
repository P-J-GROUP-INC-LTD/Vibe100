// include/strata/ds41/session/cpu_pool.hpp - DS1-E: the CPU expert pool of the DeepSeek-V4.1-Flash engine: the routed experts the GPU cache missed, computed on the two
// sockets, one worker group per socket, each socket computing ITS HALF of every missed expert.
//
// WHAT IT COMPUTES (docs/deepseek/CONTRACTS.md "The math", include/strata/ds41/cpu/mxfp4_expert.hpp): for a miss (token t, expert e, routing weight w) half h of the expert
// (the arena's `half(h, layer, e)`: gate rows [h * ff/2 ..), up rows [h * ff/2 ..), every down row's blocks [h * ff/64 ..)) gives
//     h_h = silu(min(W1_h . x, 10)) * clamp(W3_h . x, -10, 10) * w          (phase 1: DS-C's expert_gate_up, rows cut into 32-row CHUNKS, int8 activations)
//     y_h = W2[:, half h's columns] . h_h                                    (phase 2: DS-C's expert_down, the 5120 output rows cut into ranges)
// and the expert's output is y_0 + y_1.  Everything is FP32 with DS-C's quantiser, so a half is bit-identical whatever the thread count and the row cut.
//
// THE DESIGN (the answers a reviewer asks for):
//   * TWO GROUPS, ONE PER HALF.  Group h has `threads_per_group` workers, PINNED to the CPUs of the node that holds half h's arena (platform::numa topology through the
//     loader's `choose_arena_nodes`: exactly two nodes, or two sockets with sub-NUMA clustering).  On a machine that cannot bind (one node, no topology, `--numa off`)
//     the two groups still exist and are unpinned, each with half the CPUs: the arithmetic is the same, so a run's numbers do not depend on the topology.
//   * A LAYER'S MISSES ARE ONE JOB.  The pool groups the misses by expert (a verify window can send several tokens to one expert: the group then reads the half's weights
//     once for all of them, DS-C's T-token path), gives each group its own ExpertScratch and its own y rows, and runs  PHASE 1 of every group, ONE BARRIER inside the socket,
//     PHASE 2 of every group.  Phase 1 and phase 2 of different experts never touch the same memory, so no per-expert barrier is needed.
//   * ONE y PER SOCKET, STATIC ROW OWNERSHIP, NO ATOMICS.  Socket h writes its own partial `y_h` (an FP32 [rows][hidden] buffer written, hence first-touched, by the socket's own
//     workers, i.e. in its node's memory).  Both phases cut their units - 32-row chunks of the intermediate in phase 1, 16-row units of the 5120 output rows in phase 2 - into
//     one contiguous part of the FLATTENED (expert, unit) list per worker, so every worker has the same number of units to within one whatever the shapes divide into (36 chunks
//     over 23 workers would not).  A phase-2 unit is zeroed and then added to by the SAME worker, and nobody else, in this socket or the other, touches those elements: no thread
//     ever writes an element another thread writes, and the reduction order of every element is fixed.  The two sockets never write one buffer.
//   * THE TWO PARTIALS ARE ADDED ON THE HOST, once per layer, after BOTH sockets are finished: rows[i] = y_0[row i] + y_1[row i] (FP32, h0 + h1 in that order) into the
//     caller's buffer (a pinned host buffer that is uploaded to the GPU next: one h2d per layer, not two).
//   * JOBS ARE ASYNCHRONOUS: `start()` hands the layer to the workers and returns, `wait()` joins and adds the partials, so the caller can enqueue the GPU's hits and the
//     shared expert while the CPU works (the session starts the pool the moment the split's doorbell rings).  Idle workers spin briefly and then block (atomic wait), so a
//     pool on an oversubscribed machine does not burn the cores the GPU's driver thread needs.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "strata/ds41/cpu/mxfp4_expert.hpp"
#include "strata/ds41/model/model.hpp"
#include "strata/platform/numa.hpp"

namespace strata::ds41::session {

/// One missed (token, expert) pair as the split's miss list reports it (cuda::MissEntry has the same four fields).
struct CpuMiss {
    int32_t token = 0;
    int32_t k = 0;
    int32_t expert = 0;
    float weight = 0;
};

struct CpuPoolOptions {
    int threads_per_group = 0;               ///< workers per half-group (plan_cpu_pool's default: the node's CPUs minus one - that CPU stays free for the thread that drives the GPU; unpinned two-group fallback: half the CPUs, at least one)
    bool pin = true;                         ///< pin each worker to one CPU of its group's list (when the list is non-empty)
    std::vector<int> cpus[2];                ///< the CPUs of half 0's and half 1's node (empty: unpinned)
    cpu::Isa isa = cpu::Isa::kAuto;
    int max_tokens = 8;                      ///< tokens per layer call (T of a window): sizes the buffers (misses <= max_tokens * top_k)
    int top_k = 6;                           ///< routed experts per token
    int spin_iterations = 4000;              ///< idle workers spin this many polls before they block
    std::string note;                        ///< filled by plan_cpu_pool: the topology line for the startup log
};

/// Decides the pool's layout from the machine: `numa` false (--numa off) or a topology that cannot bind -> two unpinned groups, otherwise one group per node.
/// `threads_per_socket` (0 = auto) is the worker count of each group; `allowed` = the CPUs the process may run on (sched_getaffinity; empty = all).
CpuPoolOptions plan_cpu_pool(const platform::NumaTopology& topo, bool numa, int threads_per_socket, const std::vector<int>& allowed);

struct CpuPoolStats {
    uint64_t jobs = 0;                       ///< layers computed
    uint64_t experts = 0;                    ///< distinct (layer, expert) groups computed (both halves each)
    uint64_t entries = 0;                    ///< (token, expert) misses served
    double seconds = 0;                      ///< wall time of start() .. wait()
};

class CpuExpertPool {
public:
    /// `arena` must be built (both halves) and outlive the pool.
    CpuExpertPool(const model::ExpertArena& arena, const CpuPoolOptions& opt);
    ~CpuExpertPool();
    CpuExpertPool(const CpuExpertPool&) = delete;
    CpuExpertPool& operator=(const CpuExpertPool&) = delete;

    int groups() const { return 2; }
    int threads_per_group() const { return tpg_; }
    int pinned_workers() const { return pinned_; }
    cpu::Isa isa() const { return isa_; }
    std::string describe() const;
    const CpuPoolStats& stats() const { return stats_; }

    /// Hands layer `layer`'s misses to the workers and returns at once.  `x` = the quantised activations of the T tokens (DS-C's quantiser: cpu::quantize_acts), which must
    /// stay valid until wait(); `miss[i]` (i < n, in (token, k) order) names a token < T.  At most one job at a time.
    void start(int layer, const cpu::ActQ* x, int T, const CpuMiss* miss, int n);
    /// Joins the job and writes `out[i * hidden .. (i + 1) * hidden)` = y_0 + y_1 of miss i (FP32).  Returns the number of distinct experts computed.
    int wait(float* out);
    /// start() + wait().
    int run(int layer, const cpu::ActQ* x, int T, const CpuMiss* miss, int n, float* out) {
        start(layer, x, T, miss, n);
        return wait(out);
    }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int tpg_ = 0, pinned_ = 0;
    cpu::Isa isa_ = cpu::Isa::kAuto;
    CpuPoolStats stats_;
};

}  // namespace strata::ds41::session
