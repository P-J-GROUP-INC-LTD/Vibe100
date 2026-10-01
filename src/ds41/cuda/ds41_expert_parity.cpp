// src/ds41/cuda/ds41_expert_parity.cpp - DS-D: hit-expert parity and timing on the GPU (V100).  Synthetic random MXFP4 blobs, no model.
//
//   ds41_expert_parity                  checks + timing
//   ds41_expert_parity --selftest       checks only (exit status != 0 on any FAIL)
//   ds41_expert_parity --bench          timing only
//   options: --slots N (default 24; 48 adds the T = 8 timing)  --seed N  --reps N  --hits-checked N  --no-edge  --T 1,2,3,4,5,6,7,8
//
// What is checked (CONTRACTS.md; every number is printed):
//   * quantize x: the int8 values and fp32 scales are bit-identical to the CPU quantiser (strata::ds41::cpu::quantize_act's rule);
//   * quantize, the contract's special blocks: all-zero, 1e-37 everywhere, the 2^-100 boundary, ties to even, an Inf / a NaN at every position
//     (d = NaN, q = 0), and random bit patterns - hand-written expected bytes and the CPU rule (ds41_ref.hpp: quantize_block);
//   * NaN through the clamps: NaN in g only, in u only, in x: every y element of that token is NaN (the SASS has FSETP / FSEL, not FMNMX), the
//     finite control token stays finite;
//   * split: counts and groups of a routed case (the same expert for several tokens included), delivered to the host through a mapped record:
//     the host waits for the doorbell (sequence number, release store) while the expert kernels are still running;
//   * phase 1: dequantised h = silu(min(g,10)) clamp(u,+-10) w against an FP64 reference computed from the same int8 x (the bound is the
//     quantiser's own half step);
//   * phase 2: W2 . h against FP64 on the GPU's own int8 h: exact integer arithmetic on both sides, bound 5e-6 x sum|block terms|;
//   * the whole pipeline against FP64 on the unquantised activations (relative L2 error, bound 3 %);
//   * miss rows of `parts` untouched (sentinel NaN), hit rows written;
//   * E8M0 scale bytes 0, 1, 127, 254, 255 (all 256 decoded bit-exactly as ggml does) through both phases;
//   * timing: GB/s and microseconds per layer for 6 hits at T = 1 and 24 hits at T = 4 (and 48 at T = 8 with --slots 48), per phase.
// PASS looks like "ALL PASS (n checks)".
//
// On a V100:  ds41_expert_parity --selftest        (checks only)      ds41_expert_parity --bench [--slots 48]   (timing only)
#include "ds41_cuda_dev.hpp"

int main(int argc, char** argv) {
    using namespace strata::ds41::cuda::parity;
    Args a{argc, argv};
    CudaDev dev;
    print_device();
    ExpertOpts o;
    o.seed = (uint64_t) a.num("--seed", 3);
    o.slots = (int) a.num("--slots", 24);
    o.Ts = a.list("--T", o.Ts);
    o.max_hits_checked = (int) a.num("--hits-checked", 1000000);
    o.timing_reps = a.has("--selftest") ? 0 : (int) a.num("--reps", 100);
    o.edge = !a.has("--no-edge");
    if (a.has("--bench")) {
        o.full_cases = false;
        o.edge = false;
    }
    Report rep;
    run_expert_parity<strata::ds41::RealGeom>(dev, o, rep);
    return rep.summary();
}
