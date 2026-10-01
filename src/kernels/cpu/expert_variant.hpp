// src/kernels/cpu/expert_variant.hpp - the two builds of the AVX-512 Q2_0 kernels (internal to strata_kernels_cpu).
//
// `expert.cpp` holds the kernels and is compiled twice:
//
//   expert.cpp         -mavx512vbmi, STRATA_EXPERT_NO_VBMI undefined  -> namespace `vbmi`,   expert_kernels_vbmi()
//   expert_novbmi.cpp  no VBMI flag, STRATA_EXPERT_NO_VBMI defined    -> namespace `novbmi`, expert_kernels_novbmi()
//                      (it only defines the macro and includes expert.cpp)
//
// The only difference is the 2-bit unpack (`unpack64_q2_0` / `unpack_q2_0`).  Neither build exports a kernel under
// its public name: the public entry points of expert.hpp live in expert_dispatch.cpp, which is compiled WITHOUT
// AVX-512 flags and forwards through the table below, chosen once at the first call.  That keeps VBMI instructions
// out of every function that can run on a CPU without VBMI, and it means a CPU without AVX-512 never executes a
// byte of either build.
#pragma once

#include "strata/kernels/cpu/expert.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// The Q8_0-oracle switch (`expert_set_oracle_q8_0`): one flag, defined in expert_dispatch.cpp, read by both builds.
extern std::atomic<bool> g_oracle_q8_0;

struct ExpertKernels {
    const char* name;
    void (*act_quant)(const float* x, int n, ActQ& a);                                   // act_quant_q8_1
    void (*expert_q)(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws);   // s2_expert_vnni_q
    void (*gu_rows)(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1);      // s2_expert_gu_rows
    void (*down_rows)(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1);   // s2_expert_down_rows
    void (*gu_rows_multi)(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0, int r1);
    void (*down_rows_multi)(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0, int r1);
    void (*expert_multi)(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                         ExpertScratchMulti& ws);                                           // s2_expert_vnni_multi
    void (*q2g_rows_multi)(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                           float* const* out, int r0, int r1);                              // q2_0_gguf_rows_multi
    /// Test hooks: the 2-bit unpack on its own.  64 codes from 16 bytes (the 512-bit kernels), 32 codes from 8 bytes
    /// (the 256-bit ones).  Output byte j is `(codes[j / 4] >> (2 * (j % 4))) & 3`.
    void (*unpack64)(const uint8_t* codes16, uint8_t* out64);
    void (*unpack32)(const uint8_t* codes8, uint8_t* out32);
};

/// Compiled with -mavx512vbmi: call only where `cpu_features().vbmi()`.
const ExpertKernels& expert_kernels_vbmi();
/// Compiled without it: call wherever `cpu_features().usable()`.
const ExpertKernels& expert_kernels_novbmi();

}  // namespace strata::kernels::cpu
