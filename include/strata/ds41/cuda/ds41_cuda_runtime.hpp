// include/strata/ds41/cuda/ds41_cuda_runtime.hpp - DS-1: CudaDev, the CUDA-runtime implementation of the Dev interface (ds41_dev.hpp).
// Host code only (compiled as C++, linked with cudart): cudaMalloc, mapped pinned host memory, one own stream, events.  The engine and the
// V100 parity programs use it; the emulation uses HostDev.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "strata/ds41/cuda/ds41_dev.hpp"

namespace strata::ds41::cuda {

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL cuda: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

/// An allocation that failed: thrown (not exit(2)) so the caller can say what it was allocating and stop cleanly.  The message carries the request and the
/// device's free / total memory.
inline std::string cuda_alloc_failure_text(const char* what, size_t bytes, cudaError_t e) {
    size_t fr = 0, tot = 0;
    const bool known = cudaMemGetInfo(&fr, &tot) == cudaSuccess;
    (void) cudaGetLastError();                                 // the failed allocation's sticky-looking error must not poison the next call
    char b[512];
    if (known)
        std::snprintf(b, sizeof b, "%s of %.1f MiB failed: %s (device: %.1f MiB free of %.1f MiB)", what, (double) bytes / 1048576.0, cudaGetErrorString(e), (double) fr / 1048576.0,
                      (double) tot / 1048576.0);
    else
        std::snprintf(b, sizeof b, "%s of %.1f MiB failed: %s", what, (double) bytes / 1048576.0, cudaGetErrorString(e));
    return b;
}

struct CudaDev : Dev {
    cudaStream_t s = nullptr;
    CudaDev() {
        cudaSetDeviceFlags(cudaDeviceMapHost);       // mapped pinned memory for the host records (a no-op under UVA; its error, if the device is already active, is not one)
        (void) cudaGetLastError();
        cuda_check(cudaSetDevice(0), "cudaSetDevice");
        cuda_check(cudaStreamCreate(&s), "cudaStreamCreate");
    }
    ~CudaDev() override { cudaStreamDestroy(s); }
    void* alloc(size_t bytes) override {
        void* p = nullptr;
        const cudaError_t e = cudaMalloc(&p, bytes ? bytes : 16);
        if (e != cudaSuccess) throw std::runtime_error("out of device memory: cudaMalloc " + cuda_alloc_failure_text("allocation", bytes, e));
        return p;
    }
    bool mem_info(size_t& free_bytes, size_t& total_bytes) override {
        free_bytes = total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            (void) cudaGetLastError();
            free_bytes = total_bytes = 0;
            return false;
        }
        return true;
    }
    void release(void* p) override { cudaFree(p); }
    // mapped pinned host memory: the host reads it while the kernels run.  Under UVA (64-bit Linux / Windows with WDDM2 or TCC) the host pointer
    // IS the device pointer; anything else cannot be used as a kernel argument here, so it is refused.
    void* alloc_mapped(size_t bytes) override {
        void* p = nullptr;
        const cudaError_t e = cudaHostAlloc(&p, bytes ? bytes : 16, cudaHostAllocMapped);
        if (e != cudaSuccess) throw std::runtime_error("cannot allocate mapped (pinned) host memory: " + cuda_alloc_failure_text("cudaHostAlloc", bytes, e));
        std::memset(p, 0, bytes ? bytes : 16);
        void* dp = nullptr;
        cuda_check(cudaHostGetDevicePointer(&dp, p, 0), "cudaHostGetDevicePointer");
        if (dp != p) {
            std::fprintf(stderr, "FAIL cuda: the device pointer of mapped host memory differs from the host pointer (no unified addressing): not supported by these programs\n");
            std::exit(2);
        }
        return p;
    }
    void release_mapped(void* p) override { cudaFreeHost(p); }
    void h2d(void* dst, const void* src, size_t n) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s), "h2d"); cuda_check(cudaStreamSynchronize(s), "h2d sync"); }
    void d2h(void* dst, const void* src, size_t n) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, s), "d2h"); cuda_check(cudaStreamSynchronize(s), "d2h sync"); }
    void h2d_async(void* dst, const void* src, size_t n, Stream st) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, as(st)), "h2d_async"); }
    void d2h_async(void* dst, const void* src, size_t n, Stream st) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, as(st)), "d2h_async"); }
    void d2d_async(void* dst, const void* src, size_t n, Stream st) override { cuda_check(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, as(st)), "d2d_async"); }
    void fill(void* p, int byte, size_t n) override { cuda_check(cudaMemsetAsync(p, byte, n, s), "memset"); }
    void fill_async(void* p, int byte, size_t n, Stream st) override { cuda_check(cudaMemsetAsync(p, byte, n, as(st)), "memset_async"); }
    void sync() override {
        cuda_check(cudaStreamSynchronize(s), "stream sync (a kernel failed)");
        cuda_check(cudaGetLastError(), "cudaGetLastError");
    }
    Stream stream() override { return s; }
    Stream stream_create() override {
        cudaStream_t st = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
        return st;
    }
    void stream_destroy(Stream st) override { cudaStreamDestroy(as(st)); }
    void stream_sync(Stream st) override {
        cuda_check(cudaStreamSynchronize(as(st)), "stream sync (a kernel failed)");
        cuda_check(cudaGetLastError(), "cudaGetLastError");
    }
    void* event_create() override {
        cudaEvent_t e = nullptr;
        cuda_check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
        return e;
    }
    void event_destroy(void* ev) override { cudaEventDestroy(static_cast<cudaEvent_t>(ev)); }
    void event_record(void* ev, Stream st) override { cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(ev), as(st)), "cudaEventRecord"); }
    void stream_wait_event(Stream st, void* ev) override { cuda_check(cudaStreamWaitEvent(as(st), static_cast<cudaEvent_t>(ev), 0), "cudaStreamWaitEvent"); }
    void event_sync(void* ev) override { cuda_check(cudaEventSynchronize(static_cast<cudaEvent_t>(ev)), "cudaEventSynchronize"); }
    bool event_done(void* ev) override {
        const cudaError_t e = cudaEventQuery(static_cast<cudaEvent_t>(ev));
        if (e == cudaSuccess) return true;
        if (e != cudaErrorNotReady) cuda_check(e, "cudaEventQuery");
        return false;
    }
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

   private:
    cudaStream_t as(Stream st) const { return st ? static_cast<cudaStream_t>(st) : s; }       // nullptr = the Dev's own stream
};

}  // namespace strata::ds41::cuda
