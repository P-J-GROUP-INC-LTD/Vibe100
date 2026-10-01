// src/ds41/cuda/ds41_router_parity.cpp - DS-D: router parity on the GPU (V100).  Synthetic data, no model.
//
//   ds41_router_parity                  checks + timing
//   ds41_router_parity --selftest       checks only (exit status != 0 on any FAIL)
//   ds41_router_parity --bench          timing only
//   options: --seed N  --T 1,2,3,8,33,70  --reps N  --no-indep
//
// Checks: logits vs an FP64 host reference (error bound 3e-6 x sum|x w|), top-6 ids (identical sets and order; a difference explained by
// a near-tie at the 6th place is REPORTED, not failed), weights vs FP64 on the GPU's own ids, determinism (bit-identical second run), the
// lowest-index tie-break on exactly tied scores, and ROUTING DOES NOT DEPEND ON T (CONTRACTS.md): the first T tokens of one pool routed as a
// batch of T = 2 .. 40, windows that start mid-batch and single tokens routed alone (the GEMV kernel, T <= 32) against the same tokens in a
// T = 100 batch (the prefill kernel): logits compared as BITS, ids and weights exactly (--no-indep skips it).  PASS looks like
// "ALL PASS (n checks)".
//
// On a V100:  ds41_router_parity --selftest        (checks only; exit status != 0 on any FAIL)
//             ds41_router_parity --bench           (timing: T = 1 only; the full table without --selftest / --bench)
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
    if (a.has("--no-indep")) o.indep_Ts.clear();
    Report rep;
    if (a.has("--bench")) {
        o.Ts = {1};
        o.tie_test = false;
        o.indep_Ts.clear();
    }
    run_router_parity<strata::ds41::RealGeom>(dev, o, rep);
    return rep.summary();
}
