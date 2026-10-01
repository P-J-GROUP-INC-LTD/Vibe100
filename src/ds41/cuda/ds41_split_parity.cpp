// src/ds41/cuda/ds41_split_parity.cpp - DS-D: hit / miss split parity on the GPU.  Synthetic data, no model.
//
//   ds41_split_parity [--selftest] [--seed N] [--T 1,2,3,8,40,1000] [--queued N]
//
// Random router outputs against eight residency tables (about 40 % resident, nothing, everything, a handful, a hot expert, slots beyond
// n_slots, junk values, a zero-initialised table with n_slots = 0) for T = 1..1000: the hit list, the miss list (the host CPU pool's input)
// and the counts must equal the host reference EXACTLY (order = (token, k)), the slot groups (T <= 8) likewise, with out-of-range expert ids
// counted as misses and a residency value outside [0, n_slots) a miss that is COUNTED in n_bad_slots.  The odd tables hand their result to the
// host through a mapped record + miss list: the host waits for the doorbell (the sequence number, stored last with release semantics) BEFORE
// synchronising the stream.  The last test queues N launches (default 6) back to back, each with its own record and miss list, waits for them in
// REVERSE order, and does it twice with the same records (seq 1, then 2): the host must never see the previous round's result.  PASS looks like
// "ALL PASS (n checks)".
//
// On a V100:  ds41_split_parity --selftest
#include "ds41_cuda_dev.hpp"

int main(int argc, char** argv) {
    using namespace strata::ds41::cuda::parity;
    Args a{argc, argv};
    CudaDev dev;
    print_device();
    SplitOpts o;
    o.seed = (uint64_t) a.num("--seed", 2);
    o.Ts = a.list("--T", o.Ts);
    o.queued_layers = (int) a.num("--queued", o.queued_layers);
    Report rep;
    run_split_parity(dev, o, rep);
    return rep.summary();
}
