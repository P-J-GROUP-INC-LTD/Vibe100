// include/strata/ds41/cuda/ds41_dev.hpp - DS-1: the device-access interface every DS-1 host module codes against, so the SAME host code runs
// on a V100 (CudaDev, ds41_cuda_runtime.hpp) and in the CPU emulation of the kernels (HostDev, below; src/ds41/cuda/ds41_emu.hpp).
//
// Plain C++: no CUDA header, no kernel.  What is here:
//   Dev          the abstract device: allocation (device memory, MAPPED pinned host memory), blocking and stream-ordered copies, streams,
//                events, a stopwatch.  The kernels never see it; the host wrappers take a `Stream` (= dev.stream() or one made by
//                dev.stream_create()) and the buffers come from alloc().
//   DevBuf<T>    RAII device array with up()/down()/zero() (tests, scratch).         MappedBuf<T>  RAII mapped host array.
//   HostDev      the emulation's Dev: "device memory" is malloc'd host memory between two 256-byte guard zones that are CHECKED at release (a
//                kernel that stores outside its buffer fails there; the V100 would silently corrupt a neighbour), filled with 0xCD (fresh
//                device memory is not zero).  Every launch of the emulated kernels runs to completion on the calling thread, so streams and
//                events are no-ops with distinct tokens.
//
// THE LAUNCH PATTERN for a host wrapper (src/ds41/cuda/ds41_dev.cuh, `dev::launch`): a kernel is a `DS41_KERNEL` function template over G;
// the wrapper is a `template <class G> void ds41_<op>(...)` that computes the grid, calls
//     dev::launch(kernel<G, ...>, grid, block, smem_bytes, stream, kernel args...);   dev::check_launch("what");
// and the .cu file (nvcc) instantiates it for RealGeom, the emulator build (ds41_emu_impl.cpp) for RealGeom and MiniGeom.  One source, two
// compilers.  Streams: Stream is `void*` (a cudaStream_t on a GPU, ignored by the emulation).  Pass `dev.stream()` to the kernel wrappers; in the
// Dev's own copy / event methods nullptr means dev.stream() as well (on a kernel launch a null stream would be the CUDA default stream).
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::ds41::cuda {

using Stream = void*;

struct Dev {
    virtual ~Dev() = default;
    // ---- memory ----
    /// Device memory, 256-byte aligned.  A failed allocation THROWS (DS1-E: std::runtime_error "... out of device memory ..."): a caller (the loader,
    /// the engine) reports it and exits cleanly instead of the process dying inside the runtime.
    virtual void* alloc(size_t bytes) = 0;
    virtual void release(void* p) = 0;
    /// Device memory now: bytes free / total (a GPU: cudaMemGetInfo; HostDev: the host's MemAvailable / MemTotal, so a plan made against it is
    /// meaningful in the emulator build too).  false (and zeros) when the device cannot say.
    virtual bool mem_info(size_t& free_bytes, size_t& total_bytes) {
        free_bytes = total_bytes = 0;
        return false;
    }
    /// Memory the HOST can read while the device runs (a GPU: mapped pinned memory, cudaHostAlloc(cudaHostAllocMapped); the same pointer is
    /// valid in kernels under UVA; the emulation: plain memory).  Zero-filled.  A failed allocation throws, like alloc().
    virtual void* alloc_mapped(size_t bytes) = 0;
    virtual void release_mapped(void* p) = 0;
    // ---- copies ----
    /// Blocking: ordered on stream() and complete when the call returns.
    virtual void h2d(void* dst, const void* src, size_t n) = 0;
    virtual void d2h(void* dst, const void* src, size_t n) = 0;
    /// Stream-ordered, asynchronous (a GPU: the host side must be pinned and stay valid until the stream has passed it; the emulation: done
    /// at once).
    virtual void h2d_async(void* dst, const void* src, size_t n, Stream s) { (void) s; h2d(dst, src, n); }
    virtual void d2h_async(void* dst, const void* src, size_t n, Stream s) { (void) s; d2h(dst, src, n); }
    virtual void d2d_async(void* dst, const void* src, size_t n, Stream s) { (void) s; std::memmove(dst, src, n); }
    /// memset on stream() (asynchronous); `fill_async` on a given stream.
    virtual void fill(void* p, int byte, size_t n) = 0;
    virtual void fill_async(void* p, int byte, size_t n, Stream s) { (void) s; fill(p, byte, n); }
    // ---- streams and events ----
    /// stream(): the Dev's own stream, the one the parity programs and the default wrappers use.  sync(): wait for it and throw / abort if a kernel
    /// failed.
    virtual Stream stream() = 0;
    virtual void sync() = 0;
    virtual Stream stream_create() { return reinterpret_cast<Stream>(++token_); }
    virtual void stream_destroy(Stream s) { (void) s; }
    virtual void stream_sync(Stream s) { (void) s; sync(); }
    virtual void* event_create() { return reinterpret_cast<void*>(++token_); }
    virtual void event_destroy(void* ev) { (void) ev; }
    virtual void event_record(void* ev, Stream s) { (void) ev; (void) s; }
    /// Work submitted to `s` after this call starts after everything recorded in `ev` has completed.
    virtual void stream_wait_event(Stream s, void* ev) { (void) s; (void) ev; }
    virtual void event_sync(void* ev) { (void) ev; }
    /// true once the work recorded in `ev` has completed.
    virtual bool event_done(void* ev) { (void) ev; return true; }
    // ---- timing / identity ----
    /// average microseconds of fn() over `reps` calls (after one warm-up), the device synchronised around the whole loop
    virtual double time_us(const std::function<void()>& fn, int reps) = 0;
    /// true for the CPU emulation of the kernels (no GPU: timings and register budgets mean nothing there).
    virtual bool is_emulation() const = 0;

   private:
    uintptr_t token_ = 0x1000;
};

/// The stream a wrapper uses when the caller passes nullptr: the Dev's own.
inline Stream stream_or_default(Dev& dev, Stream s) { return s ? s : dev.stream(); }

template <class T>
struct DevBuf {
    Dev& dev;
    T* p = nullptr;
    size_t n = 0;
    DevBuf(Dev& d, size_t count) : dev(d), n(count) { p = static_cast<T*>(dev.alloc(count * sizeof(T))); }
    ~DevBuf() { dev.release(p); }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    void up(const std::vector<T>& v) { dev.h2d(p, v.data(), v.size() * sizeof(T)); }
    std::vector<T> down(size_t count) const {
        std::vector<T> v(count);
        dev.d2h(v.data(), p, count * sizeof(T));
        return v;
    }
    std::vector<T> down() const { return down(n); }
    void zero() { dev.fill(p, 0, n * sizeof(T)); }
};

/// A zero-filled host-visible array (see Dev::alloc_mapped); read it through the pointer, no copy.
template <class T>
struct MappedBuf {
    Dev& dev;
    T* p = nullptr;
    size_t n = 0;
    MappedBuf(Dev& d, size_t count) : dev(d), n(count) { p = static_cast<T*>(dev.alloc_mapped(count * sizeof(T))); }
    ~MappedBuf() { dev.release_mapped(p); }
    MappedBuf(const MappedBuf&) = delete;
    MappedBuf& operator=(const MappedBuf&) = delete;
};

/// The emulation's device: see the header comment.  Header-only so that every emulated test of every package can use it.
struct HostDev : Dev {
    static constexpr size_t kGuard = 256;
    std::map<void*, size_t> live;
    std::map<void*, size_t> mapped;

    void* alloc(size_t bytes) override {
        if (bytes == 0) bytes = 16;
        const size_t body = (bytes + 255) & ~(size_t) 255;
        unsigned char* base = static_cast<unsigned char*>(std::aligned_alloc(256, body + 2 * kGuard));
        if (!base) throw std::runtime_error("out of device memory (emulation): cannot allocate " + std::to_string(bytes) + " bytes of host memory");
        std::memset(base, 0xAB, body + 2 * kGuard);
        std::memset(base + kGuard, 0xCD, body);                        // fresh "device memory" is not zero
        live[base + kGuard] = bytes;
        return base + kGuard;
    }
    // "mapped host memory": plain memory here (zero-filled); the doorbell protocol is exercised, not the PCIe
    void* alloc_mapped(size_t bytes) override {
        if (bytes == 0) bytes = 16;
        const size_t body = (bytes + 255) & ~(size_t) 255;
        void* p = std::aligned_alloc(256, body);
        if (!p) throw std::runtime_error("cannot allocate " + std::to_string(bytes) + " bytes of mapped host memory (emulation)");
        std::memset(p, 0, body);
        mapped[p] = body;
        return p;
    }
    void release_mapped(void* p) override {
        if (!p) return;
        mapped.erase(p);
        std::free(p);
    }
    void release(void* p) override {
        if (!p) return;
        const size_t bytes = live.at(p);
        const size_t body = (bytes + 255) & ~(size_t) 255;
        unsigned char* base = static_cast<unsigned char*>(p) - kGuard;
        bool ok = true;
        for (size_t i = 0; i < kGuard; ++i) ok = ok && base[i] == 0xAB;
        for (size_t i = bytes; i < body; ++i) ok = ok && base[kGuard + i] == 0xCD;       // the padding of the last 256 bytes
        for (size_t i = 0; i < kGuard; ++i) ok = ok && base[kGuard + body + i] == 0xAB;
        if (!ok) {
            std::printf("FAIL guard: a kernel or the test wrote outside a device buffer of %zu bytes\n", bytes);
            std::exit(3);
        }
        live.erase(p);
        std::free(base);
    }
    void h2d(void* dst, const void* src, size_t n) override { std::memcpy(dst, src, n); }
    void d2h(void* dst, const void* src, size_t n) override { std::memcpy(dst, src, n); }
    void fill(void* p, int byte, size_t n) override { std::memset(p, byte, n); }
    void sync() override {}
    Stream stream() override { return nullptr; }
    double time_us(const std::function<void()>& fn, int reps) override {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; ++i) fn();
        return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
    }
    bool is_emulation() const override { return true; }
    /// MemAvailable / MemTotal of /proc/meminfo (Linux); false elsewhere.
    bool mem_info(size_t& free_bytes, size_t& total_bytes) override {
        free_bytes = total_bytes = 0;
        std::FILE* f = std::fopen("/proc/meminfo", "r");
        if (!f) return false;
        char line[256];
        unsigned long long kb = 0;
        bool have_total = false, have_avail = false;
        while (std::fgets(line, sizeof line, f)) {
            if (std::sscanf(line, "MemTotal: %llu kB", &kb) == 1) { total_bytes = (size_t) kb * 1024; have_total = true; }
            else if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { free_bytes = (size_t) kb * 1024; have_avail = true; }
        }
        std::fclose(f);
        return have_total && have_avail;
    }
};

}  // namespace strata::ds41::cuda
