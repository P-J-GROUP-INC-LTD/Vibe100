// src/ds41/program/strata_ds41_main.cpp - DS1-E: `strata-ds41`, the DeepSeek-V4.1-Flash engine for the V100 (RealGeom, sm_70).  The program is src/ds41/program/cli.hpp; this file only
// brings up the device.  Run `ds41_model_info <shard>` first (what the loader makes of the file and the memory plan), then
//   strata-ds41 --gguf <shard 1> --tokens 0,128803,... --max-new 16 --trace /tmp/tr --stats
#include <cstdio>

#include "cli.hpp"
#include "strata/ds41/cuda/ds41_cuda_runtime.hpp"
#include "strata/ds41/geom.hpp"

int main(int argc, char** argv) {
    using namespace strata::ds41;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) {
        std::fprintf(stderr, "strata-ds41: no CUDA device found\n");
        return 1;
    }
    cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
        std::fprintf(stderr, "ds41: GPU: %s, compute capability %d.%d, %.0f MiB\n", p.name, p.major, p.minor, (double) p.totalGlobalMem / 1048576.0);
        if (p.major != 7 || p.minor != 0) std::fprintf(stderr, "ds41: warning: this is not a V100 (sm_70): the numbers are valid, the timings are not what the box gives\n");
    }
    try {
        cuda::CudaDev dev;
        return program::run_cli<RealGeom>(dev, argc, argv, "strata-ds41");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-ds41: error: %s\n", e.what());
        return 1;
    }
}
