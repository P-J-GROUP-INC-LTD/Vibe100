// include/strata/core/device.hpp - P2.S1: the device arena and the runtime's device facts.
//
// `DeviceArena` is ONE cudaMalloc per planner region with bump sub-allocation below it and no frees.  That is
// not a simplification for the first version: the memory plan from P1.S9 is fixed at startup, so the set of
// regions and their sizes is known before anything is allocated, and an allocator that can free would be
// solving a problem the engine does not have while adding fragmentation and failure modes it does.
//
// The reason to get this in early is that the VRAM budget is the binding constraint of the whole design
// (5.95 GB pooled between KV and the expert cache, 33.97 GB of experts in DRAM).  A runtime that discovers at
// token 4000 that it has overcommitted has already lost; the plan is printed against `cudaMemGetInfo` at
// startup so the discrepancy is visible immediately.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::core {

struct DeviceInfo {
    int ordinal = -1;
    std::string name;
    int cc_major = 0, cc_minor = 0;
    uint64_t total_bytes = 0;      // as reported by cudaMemGetInfo at query time
    uint64_t free_bytes = 0;
    int driver_version = 0, runtime_version = 0;
    int multi_processor_count = 0;
    std::string arch;              // HIP: gcnArchName without its feature suffix ("gfx1201"); empty on CUDA
    // CUDA: the code the driver picked for a kernel of this binary on this card (cudaFuncGetAttributes), as sm numbers
    // ("86"): the SASS it runs (binaryVersion; a JIT-compiled one is the card's own) and the architecture the PTX it
    // came from was compiled for (ptxVersion, i.e. the __CUDA_ARCH__ / 10 the kernels' bodies were chosen with).  0: not known.
    int binary_version = 0, ptx_version = 0;
};

// CUDA builds, a pure function so a plain host test can run it (tests: tools/test_setup_device_probe.py): "" when the
// code the driver will run on a card was compiled for an architecture that card can use, else the problem in a sentence.
// The probe in device.cu (cudaFuncGetAttributes) says a binary has SOME code for a card; it does not say WHICH.  An engine
// built for sm_70 alone (-DCMAKE_CUDA_ARCHITECTURES=70 emits compute_70 PTX and sm_70 SASS) runs its sm_70 SASS natively on
// a compute capability 7.5 card and JIT-compiles its compute_70 PTX on an 8.x one - the probe passes on an RTX 20 / 30 /
// 40 - but every kernel then has the body chosen for __CUDA_ARCH__ 700, and the Turing-and-newer ones (the prompt attention's
// tensor-core path, ggml's turing-mma MMQ paths, selected by the device's own cc) are trap stubs there: the first prompt
// dies with "unspecified launch failure".  So a card of cc 7.5 or newer whose code was compiled below sm_75 is refused here.
// `built_for` is STRATA_CUDA_ARCHS ("70,75"; nullptr or "" when unknown), `card` names the GPU for the message.
inline std::string older_code_problem(const std::string& card, int cc_major, int cc_minor, int ptx_version,
                                      int binary_version, const char* built_for) {
    const int sm = cc_major * 10 + cc_minor;
    if (sm < 75 || ptx_version <= 0 || ptx_version >= 75) return "";
    std::vector<int> have;       // the architectures the engine was built for, then this card's
    const std::string list = built_for ? built_for : "";
    for (size_t i = 0; i < list.size();) {
        if (std::isdigit((unsigned char) list[i])) {
            int v = 0;
            while (i < list.size() && std::isdigit((unsigned char) list[i])) v = v * 10 + (list[i++] - '0');
            have.push_back(v);                     // "75-real" and "75-virtual" are still 75
        } else {
            ++i;
        }
    }
    std::string why = card + " is compute capability " + std::to_string(cc_major) + "." + std::to_string(cc_minor) +
                      " (sm_" + std::to_string(sm) + "), but this Strata engine was compiled for sm_" +
                      std::to_string(ptx_version) + " only (" +
                      (list.empty() ? std::string() : "CMAKE_CUDA_ARCHITECTURES=" + list + "; ") +
                      "the driver runs its sm_" + std::to_string(ptx_version) + " code on this card: " +
                      (binary_version > 0 ? "binary sm_" + std::to_string(binary_version) + ", " : std::string()) +
                      "PTX sm_" + std::to_string(ptx_version) + ") and that code lacks the kernels a card of this "
                      "generation runs - its first prompt fails with \"unspecified launch failure\".  ";
    if (sm >= 120) {          // (the engine has an arch below 75, so it is a CUDA 12 build)
        why += "An RTX 50 card needs an engine built with CUDA 13 and a Volta one needs CUDA 12, so they cannot share one "
               "engine: build a second one for this card with -DCMAKE_CUDA_ARCHITECTURES=" + std::to_string(sm) +
               " (CUDA 13), or run ./setup.sh in another Strata folder";
    } else {
        have.push_back(sm);
        std::sort(have.begin(), have.end());
        have.erase(std::unique(have.begin(), have.end()), have.end());
        std::string archs;
        for (const int v : have) archs += (archs.empty() ? "" : ";") + std::to_string(v);
        why += "Rebuild the engine with its own arch for this card, -DCMAKE_CUDA_ARCHITECTURES=\"" + archs +
               "\" (./setup.sh does this for the cards it finds)";
    }
    return why + ", or choose another GPU with CUDA_VISIBLE_DEVICES";
}

// HIP builds: whether GPU `ordinal` can run this binary - its architecture must be one the binary was COMPILED
// for (STRATA_HIP_ARCHS, set by cmake/hip_backend.cmake) and it must run wave32.  "" when it can (or when there
// is no such device: the caller's own device errors apply), else the reason in a sentence.  A binary carried to
// another card would otherwise fail later with "invalid device function".  CUDA builds: whether the binary has code
// for GPU `ordinal` at all, and code compiled for sm_75 or newer when the card is 7.5 or newer (older_code_problem).
std::string gpu_arch_problem(int ordinal);

// The GPU architectures this binary was compiled for ("gfx1100,gfx1201"); "" on CUDA builds.
const char* compiled_gpu_archs();

// Throws when there is no CUDA device, when the device is older than the engine supports (compute capability
// 7.0, Volta; 6.0 in the experimental Pascal build), or when this binary carries no code for it.  `CMakeLists.txt`
// already refuses to COMPILE for an unsupported architecture, and this is the matching check at run time (a binary
// can be carried to a different machine).
DeviceInfo device_info(int ordinal = 0);

class CudaError : public std::runtime_error {
public:
    CudaError(const std::string& what, int code) : std::runtime_error(what), code_(code) {}
    int code() const { return code_; }

private:
    int code_;
};

// One cudaMalloc, bump-allocated below.  `poison` fills new allocations with a NaN-ish pattern in a debug
// build so that reading uninitialised VRAM gives a NaN rather than a plausible number - the same reasoning as
// the harness work in Phase 1: a wrong value that looks right is the expensive kind.
class DeviceArena {
public:
    explicit DeviceArena(uint64_t bytes, int ordinal = 0, bool poison = false);
    ~DeviceArena();
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;

    // `align` must be a power of two; 256 keeps every sub-allocation at a sector boundary.
    void* alloc(uint64_t bytes, uint64_t align = 256);

    uint64_t capacity() const { return capacity_; }
    uint64_t used() const { return used_; }
    uint64_t peak() const { return used_; }        // no frees, so used IS the peak
    int ordinal() const { return ordinal_; }
    void* base() const { return base_; }

private:
    void* base_ = nullptr;
    uint64_t capacity_ = 0, used_ = 0;
    int ordinal_ = 0;
    bool poison_ = false;
};

}  // namespace strata::core
