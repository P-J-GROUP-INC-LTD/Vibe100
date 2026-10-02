// include/strata/ds41/session/moe_combine.hpp - DS1-E: the FP32 sum of one MoE layer, on the device.
//
//     y[t] = ((((0 + e_a) + e_b) + ...) + shared[t])        e_a, e_b, ... = the token's routed-expert outputs in ASCENDING expert id
//
// exactly the oracle's order (ref/ds41/moe.py `moe`: `for e in np.unique(r.indices): y[rows] += expert(...)`, then `y += shared`; model.py Model.block).  Where an
// expert's output y_(t,k) comes from depends on whether the layer's split sent it to the GPU (a hit: row (t * kTopK + k) of `parts`, written by experts_down) or
// to the CPU pool (a miss: the pool's row `cpu_row[t * kTopK + k]` of `cpu_rows`, the two sockets' partials already added).  FP32, no FMA, one fixed order
// per output element (no atomics), independent of how the threads are scheduled.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/ds41/cuda/ds41_dev.hpp"
#include "strata/ds41/geom.hpp"

namespace strata::ds41::cuda {

/// ids int32 [T][kTopK] (the router's, device); parts fp32 [T * kTopK][kHidden] (experts_down's, device: only the hit rows are meaningful); cpu_row int32
/// [T * kTopK] (device: the row of `cpu_rows` holding that (token, k)'s output, or -1 for a hit); cpu_rows fp32 [n][kHidden] (device; may be any valid pointer
/// when no entry refers to it); shared fp32 [T][kHidden]; out fp32 [T][kHidden] (may not overlap an input).  All 16-byte aligned.
template <class G>
void ds41_moe_combine(Dev& dev, const int32_t* ids, const float* parts, const int32_t* cpu_row, const float* cpu_rows, const float* shared, int T, float* out,
                      Stream stream = nullptr);

}  // namespace strata::ds41::cuda
