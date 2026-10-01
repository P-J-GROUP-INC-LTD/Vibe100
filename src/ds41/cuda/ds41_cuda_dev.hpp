// src/ds41/cuda/ds41_cuda_dev.hpp - DS-D / DS1-G: what the three V100 parity programs share: the shared CudaDev (ds41_cuda_runtime.hpp: cudaMalloc, mapped
// pinned memory, streams, events), the device banner and the little command-line helpers.  Host code only (compiled as C++, linked with cudart).
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ds41_parity_lib.hpp"
#include "strata/ds41/cuda/ds41_cuda_runtime.hpp"

namespace strata::ds41::cuda::parity {

// The CUDA-runtime Dev (CudaDev, cuda_check) is the shared one: include/strata/ds41/cuda/ds41_cuda_runtime.hpp.
using ::strata::ds41::cuda::CudaDev;
using ::strata::ds41::cuda::cuda_check;

/// "NVIDIA Tesla V100 (cc 7.0, 80 SMs, 31.7 GB)" and a warning when the card is not a Volta.
inline void print_device() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        std::printf("INFO no CUDA device\n");
        return;
    }
    int mem_clock = 0, bus = 0;
    cudaDeviceGetAttribute(&mem_clock, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    std::printf("INFO device: %s (cc %d.%d, %d SMs, %.1f GB, peak HBM %.0f GB/s)%s\n", p.name, p.major, p.minor, p.multiProcessorCount,
                (double) p.totalGlobalMem / 1e9, 2.0 * mem_clock * 1e3 * (bus / 8) / 1e9,
                (p.major == 7 && p.minor == 0) ? "" : "  <-- NOT a Volta (sm_70) card: the numbers say nothing about the V100");
}

struct Args {
    int argc;
    char** argv;
    bool has(const char* flag) const {
        for (int i = 1; i < argc; ++i)
            if (!std::strcmp(argv[i], flag)) return true;
        return false;
    }
    long num(const char* flag, long def) const {
        for (int i = 1; i + 1 < argc; ++i)
            if (!std::strcmp(argv[i], flag)) return std::strtol(argv[i + 1], nullptr, 10);
        return def;
    }
    std::vector<int> list(const char* flag, const std::vector<int>& def) const {
        for (int i = 1; i + 1 < argc; ++i)
            if (!std::strcmp(argv[i], flag)) {
                std::vector<int> v;
                const char* p = argv[i + 1];
                while (*p) {
                    char* end;
                    v.push_back((int) std::strtol(p, &end, 10));
                    p = *end == ',' ? end + 1 : end;
                    if (end == p && *p) break;
                }
                return v;
            }
        return def;
    }
};

}  // namespace strata::ds41::cuda::parity
