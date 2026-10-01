// src/ds41/cuda/ds41_cuda_dev.hpp - DS-D: the CUDA-runtime implementation of the parity `Dev` interface (cudaMalloc, events), plus
// the little command-line helpers the three V100 parity programs share.  Host code only (compiled as C++, linked with cudart).
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ds41_parity_lib.hpp"

namespace strata::ds41::cuda::parity {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL cuda: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct CudaDev : Dev {
    cudaStream_t s = nullptr;
    CudaDev() {
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaStreamCreate(&s), "cudaStreamCreate");
    }
    ~CudaDev() override { cudaStreamDestroy(s); }
    void* alloc(size_t bytes) override {
        void* p = nullptr;
        cuda_check(cudaMalloc(&p, bytes ? bytes : 16), "cudaMalloc");
        return p;
    }
    void release(void* p) override { cudaFree(p); }
    void h2d(void* dst, const void* src, size_t n) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s), "h2d"); cuda_check(cudaStreamSynchronize(s), "h2d sync"); }
    void d2h(void* dst, const void* src, size_t n) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, s), "d2h"); cuda_check(cudaStreamSynchronize(s), "d2h sync"); }
    void fill(void* p, int byte, size_t n) override { cuda_check(cudaMemsetAsync(p, byte, n, s), "memset"); }
    void sync() override {
        cuda_check(cudaStreamSynchronize(s), "stream sync (a kernel failed)");
        cuda_check(cudaGetLastError(), "cudaGetLastError");
    }
    void* stream() override { return s; }
    double time_us(const std::function<void()>& fn, int reps) override {
        fn();
        sync();
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a, s);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        sync();
        return 1000.0 * (double) ms / reps;
    }
    bool is_emulation() const override { return false; }
};

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
