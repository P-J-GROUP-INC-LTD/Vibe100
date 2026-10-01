// src/ds41/cuda/ds41_split_parity.cpp - DS-D: hit / miss split parity on the GPU.  Synthetic data, no model.
//
//   ds41_split_parity [--selftest] [--seed N] [--T 1,2,3,8,40,1000]
//
// Random router outputs against five residency tables (about 40 % resident, nothing, everything, a handful, a hot expert) for
// T = 1..1000: the hit list, the miss list (the host CPU pool's input) and the counts must equal the host reference EXACTLY
// (order = (token, k)), the slot groups (T <= 8) likewise, with out-of-range expert ids counted as misses.  PASS looks like
// "ALL PASS (n checks)".
#include "ds41_cuda_dev.hpp"

int main(int argc, char** argv) {
    using namespace strata::ds41::cuda::parity;
    Args a{argc, argv};
    CudaDev dev;
    print_device();
    SplitOpts o;
    o.seed = (uint64_t) a.num("--seed", 2);
    o.Ts = a.list("--T", o.Ts);
    Report rep;
    run_split_parity(dev, o, rep);
    return rep.summary();
}
