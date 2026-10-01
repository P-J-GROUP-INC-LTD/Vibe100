// src/kernels/cpu/expert_dispatch.cpp - the public entry points of expert.hpp.
//
// Compiled WITHOUT any AVX-512 flag, on purpose.  The AVX-512 kernels are in expert.cpp (compiled twice: with and
// without VBMI, see expert_variant.hpp); the functions here only pick one of the two builds and forward to it, so
// nothing in this file - including the scalar oracle and the feature probes' callers - can fault on a CPU that lacks
// AVX-512, and a CPU with AVX-512 but no VBMI never executes a VBMI instruction.
//
// The choice is made once, at the first call, from `cpu_q2_expert_isa()` (CPUID, the OS's register-state mask and the
// STRATA_FORCE_AVX512_NOVBMI override), and kept for the life of the process: the pool's workers all call through
// the same table, so a layer's experts are never split across two builds.  Both builds give bit-identical results
// (expert.cpp, "TWO BUILDS"), so the choice is only about speed and about which instructions the CPU has.
#include "strata/kernels/cpu/expert.hpp"
#include "expert_variant.hpp"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {

std::atomic<bool> g_oracle_q8_0{false};

namespace {

/// fp16 -> fp32, written out rather than using `_cvtsh_ss`, because the F16C intrinsic's behaviour on
/// subnormals is the one place the two can differ and the scales in this artifact are small.
inline float h2f(const uint8_t* p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) {
            f = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; --exp; }
            man &= 0x3FFu;
            f = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
}

const ExpertKernels& kernels() {
    static const ExpertKernels* const k = []() -> const ExpertKernels* {
        switch (cpu_q2_expert_isa()) {
            case ExpertIsa::Avx512Vbmi: return &expert_kernels_vbmi();
            case ExpertIsa::Avx512Vnni: return &expert_kernels_novbmi();
            default: break;
        }
        // Reached only by a caller that skipped cpu_require_expert_support() (the engine calls it before any pack is
        // read; the test programs check `cpu_features().usable()`).  Said by name instead of an illegal instruction.
        std::fprintf(stderr, "strata: internal error: an AVX-512 expert kernel was called on a CPU that cannot run it: %s\n",
                     cpu_features().reason());
        std::abort();
    }();
    return *k;
}

}  // namespace

void expert_set_oracle_q8_0(bool enabled) { g_oracle_q8_0.store(enabled, std::memory_order_relaxed); }

bool expert_oracle_q8_0_enabled() { return g_oracle_q8_0.load(std::memory_order_relaxed); }

void act_quant_q8_1(const float* x, int n, ActQ& a) { kernels().act_quant(x, n, a); }

void s2_expert_vnni(const uint8_t* blob, const float* x, float* out, ExpertScratch& ws) {
    kernels().act_quant(x, H, ws.a1);
    kernels().expert_q(blob, ws.a1, out, ws);
}

void s2_expert_vnni_q(const uint8_t* blob, const ActQ& a1, float* out, ExpertScratch& ws) {
    kernels().expert_q(blob, a1, out, ws);
}

void s2_expert_gu_rows(const uint8_t* blob, const ActQ& a1, float* ff, int r0, int r1) {
    kernels().gu_rows(blob, a1, ff, r0, r1);
}

void s2_expert_down_rows(const uint8_t* blob, const ActQ& a2, float* out, int r0, int r1) {
    kernels().down_rows(blob, a2, out, r0, r1);
}

void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* ff, int r0,
                             int r1) {
    kernels().gu_rows_multi(blob, a1, n_tokens, ff, r0, r1);
}

void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const* a2, int n_tokens, float* const* out, int r0,
                               int r1) {
    kernels().down_rows_multi(blob, a2, n_tokens, out, r0, r1);
}

void s2_expert_vnni_multi(const uint8_t* blob, const ActQ* const* a1, int n_tokens, float* const* out,
                          ExpertScratchMulti& ws) {
    kernels().expert_multi(blob, a1, n_tokens, out, ws);
}

void q2_0_gguf_rows_multi(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt,
                          float* const* out, int r0, int r1) {
    kernels().q2g_rows_multi(w, row_bytes, nblocks, a, nt, out, r0, r1);
}

void s2_expert_scalar(const uint8_t* blob, const float* x_in, float* out, bool quant_acts) {
    // BOTH STAGES, which is what `quant_acts` promises and what `bench/micro/cpu_s2.cpp` does NOT do.
    //
    // The original's comment says the oracle "consume[s] the SAME INT8 activation values the VNNI path uses,
    // at both stages", but its code quantizes only the INTERMEDIATE - the gate/up projections still see raw
    // f32 `x`.  Feeding it an already-quantized input (which is what the engine does) makes the two coincide
    // and hides the discrepancy entirely; feeding it a raw f32 input makes the comparison one between two
    // DIFFERENT computations, and the gap measured here was 1.19e-02 - three times P2.S3's tolerance, and
    // read as a kernel bug when it was a fixture bug.
    float xq[H];
    const float* x = x_in;
    if (quant_acts) {
        ActQ a1;
        act_quant_q8_1(x_in, H, a1);
        for (int i = 0; i < H; ++i) xq[i] = a1.scale[i / QKA] * (float) a1.q[i];
        x = xq;
    }
    float ff[FF];
    for (int r = 0; r < FF; ++r) {
        const uint8_t* gc = blob + O_GU_CODES + (size_t) (2 * r) * ROW_GU;
        const uint8_t* gs = blob + O_GU_SCALES + (size_t) (2 * r) * SC_GU * 2;
        const uint8_t* uc = blob + O_GU_CODES + (size_t) (2 * r + 1) * ROW_GU;
        const uint8_t* us = blob + O_GU_SCALES + (size_t) (2 * r + 1) * SC_GU * 2;
        float sg = 0.f, su = 0.f;
        for (int b = 0; b < SC_GU; ++b) {
            const float dg = h2f(gs + 2 * b), du = h2f(us + 2 * b);
            for (int j = 0; j < QK; ++j) {
                const int o = b * QK + j;
                sg += (float) (((gc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * dg * x[o];
                su += (float) (((uc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * du * x[o];
            }
        }
        ff[r] = (sg / (1.f + std::exp(-sg))) * su;
    }
    if (quant_acts) {
        // Replace the exact intermediate with the INT8 values the VNNI path actually sees, so the two differ
        // only by FP32 evaluation order.
        ActQ a2;
        act_quant_q8_1(ff, FF, a2);
        for (int i = 0; i < FF; ++i) ff[i] = a2.scale[i / QKA] * (float) a2.q[i];
    }
    for (int r = 0; r < H; ++r) {
        const uint8_t* dc = blob + O_D_CODES + (size_t) r * ROW_D;
        const uint8_t* ds = blob + O_D_SCALES + (size_t) r * SC_D * 2;
        float acc = 0.f;
        for (int b = 0; b < SC_D; ++b) {
            const float d = h2f(ds + 2 * b);
            for (int j = 0; j < QK; ++j)
                acc += (float) (((dc[b * 16 + (j >> 2)] >> (2 * (j & 3))) & 3) - 1) * d * ff[b * QK + j];
        }
        out[r] = acc;
    }
}

}  // namespace strata::kernels::cpu
