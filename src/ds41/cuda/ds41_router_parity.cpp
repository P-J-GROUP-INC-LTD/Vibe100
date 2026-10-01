// src/ds41/cuda/ds41_router_parity.cpp - DS-D: router parity on the GPU (V100).  Synthetic data, no model.
//
//   ds41_router_parity                  checks + timing
//   ds41_router_parity --selftest       checks only (exit status != 0 on any FAIL)
//   ds41_router_parity --bench          timing only
//   options: --seed N  --T 1,2,3,8,33,70  --reps N
//
// Checks: logits vs an FP64 host reference (error bound 3e-6 x sum|x w|), top-6 ids (identical sets and order; a difference explained by
// a near-tie at the 6th place is REPORTED, not failed), weights vs FP64 on the GPU's own ids, determinism (bit-identical second run), the
// lowest-index tie-break on exactly tied scores.  PASS looks like "ALL PASS (n checks)".
#include "ds41_cuda_dev.hpp"

int main(int argc, char** argv) {
    using namespace strata::ds41::cuda::parity;
    Args a{argc, argv};
    CudaDev dev;
    print_device();
    RouterOpts o;
    o.seed = (uint64_t) a.num("--seed", 1);
    o.Ts = a.list("--T", o.Ts);
    o.timing_reps = a.has("--selftest") ? 0 : (int) a.num("--reps", 100);
    Report rep;
    if (a.has("--bench")) {
        o.Ts = {1};
        o.tie_test = false;
    }
    run_router_parity(dev, o, rep);
    return rep.summary();
}
