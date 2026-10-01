// src/kernels/cpu/expert_parity.cpp - P2.S3's test for the CPU expert path.
//
// WHAT IS BEING CHECKED, and the two tolerances are different questions:
//
//   1. VNNI vs the scalar oracle with `quant_acts = true`.  Both sides consume the SAME INT8 activation
//      values, so the only difference is FP32 evaluation ORDER.  This is the check that proves the KERNEL -
//      the VNNI unpack, the 32-element chunk split, the scalar correction term - and P2.S3 asks for 3e-3.
//   2. VNNI vs the oracle with `quant_acts = false`.  Now the oracle uses exact FP32 activations, so the gap
//      IS the activation contract (`docs/activation-contract.md`).  It is REPORTED, not asserted tightly,
//      because it is a property of the model's quantisation and not of this code.
//
// The activation contract is asserted OBSERVABLE between the two: if rounding the activation to Q8_1 made no
// difference, then one of the two paths is not doing what it claims and check 1 would be vacuous.
//
// `--synthetic` runs the same checks on a random expert instead of one read from the pack (no 66 GB file needed): the
// kernel check (1) is about the arithmetic, not the data, and this is what runs on a machine without the model.  The
// structural checks about the file (a next expert that differs) are skipped there, as they have nothing to look at.
//
// WHICH BUILD IS CHECKED: the one the CPU uses (`cpu_q2_expert_isa()`: the VBMI build on Ice Lake / Zen 4, the no-VBMI
// build on Cascade Lake).  STRATA_FORCE_AVX512_NOVBMI=1 checks the no-VBMI build on a CPU that has VBMI.
//
// A SELF-CONSISTENT TEST CANNOT CATCH A WRONG FILE OFFSET.  Both sides read the same blob, so pointing at the
// wrong expert would still agree perfectly.  What is checked instead is structural: the blob must decode to
// finite, non-zero scales, and the blob at a DIFFERENT layer must be different data - which is what a wrong
// stride would break.
#include "strata/kernels/cpu/expert.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

/// Normalised L1 difference, with the reference magnitude returned for reporting.
double rel_l1(const float* a, const float* b, int n, double* mag_out = nullptr) {
    double d = 0, mag = 0;
    for (int i = 0; i < n; ++i) {
        d += std::fabs((double) a[i] - (double) b[i]);
        mag += std::fabs((double) a[i]);
    }
    if (mag_out) *mag_out = mag / n;
    return d / (mag > 1e-30 ? mag : 1e-30);
}

bool read_blob(const char* path, long long index, std::vector<uint8_t>& out) {
    out.assign(cpu::BLOB, 0);
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
#if defined(_MSC_VER)
    if (_fseeki64(f, (long long) index * (long long) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#else
    if (fseeko(f, (off_t) index * (off_t) cpu::BLOB, SEEK_SET) != 0) { std::fclose(f); return false; }
#endif
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    return got == out.size();
}

/// A random expert: random codes, fp16 scales of about 0.004 - 0.016 (the artifact's range).
void make_synthetic_blob(std::vector<uint8_t>& b) {
    std::mt19937 rng(31337);
    b.assign(cpu::BLOB, 0);
    for (size_t i = 0; i < cpu::O_GU_SCALES; ++i) b[i] = (uint8_t) rng();
    for (size_t i = cpu::O_GU_SCALES; i < cpu::BLOB; i += 2) {
        const uint16_t h = (uint16_t) (0x1C00 + rng() % 0x0800);
        std::memcpy(b.data() + i, &h, 2);
    }
}

/// The fp16 scale at a byte offset, decoded by the SAME rule the kernel uses.  Used only for the structural
/// sanity check, so it is deliberately a second, independent transcription of the three lines in question.
float f16_at(const std::vector<uint8_t>& b, size_t off) {
    uint16_t h;
    std::memcpy(&h, b.data() + off, 2);
    const uint32_t sign = (uint32_t) (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; --exp; } man &= 0x3FFu;
               f = (sign << 31) | (exp << 23) | (man << 13); }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float o;
    std::memcpy(&o, &f, 4);
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false, synthetic = false;
    const char* path = "pack/full/experts.bin";
    long long layer = 0, expert = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") selftest = true;
        else if (a == "--synthetic") synthetic = true;
        else if (a == "--file" && i + 1 < argc) path = argv[++i];
        else if (a == "--layer" && i + 1 < argc) layer = std::atoll(argv[++i]);
        else if (a == "--expert" && i + 1 < argc) expert = std::atoll(argv[++i]);
        else { std::fprintf(stderr, "usage: expert_parity [--selftest] [--synthetic] [--file P] [--layer N] [--expert N]\n");
               return 2; }
    }

    int bad = 0;

    // ---- the CPU must be able to run the kernel, and the report must say WHICH feature is missing
    const cpu::CpuFeatures feat = cpu::cpu_features();
    std::printf("  %-44s %s\n", "AVX512 F/BW/VL/DQ/VNNI (VBMI optional)",
                feat.usable() ? "all present" : feat.reason());
    if (feat.usable())
        std::printf("  %-44s %s\n", "Q2_0 kernel build under test", cpu::expert_isa_name(cpu::cpu_q2_expert_isa()));
    if (!feat.usable()) {
        std::printf("      the VNNI path cannot run here; only the scalar oracle can be exercised.\n");
        std::printf("      P2.S3's kernel check is SKIPPED, not passed.\n");
        std::printf("\nexpert_parity: 0 failures, 1 SKIPPED\n");
        return 0;
    }

    // ---- a real blob out of the pack: layer stride is 512 experts
    const long long idx = layer * 512 + expert;
    std::vector<uint8_t> blob;
    if (synthetic) {
        make_synthetic_blob(blob);
        std::printf("  %-44s %s\n", "expert", "synthetic (random codes, fp16 scales 0.004-0.016)");
    } else if (!read_blob(path, idx, blob)) {
        std::fprintf(stderr, "cannot read expert %lld (layer %lld) from %s\n", expert, layer, path);
        return 2;
    }

    // ---- structural sanity, because the agreement checks below are self-consistent and would pass on the
    //      wrong blob.  A mis-strided read lands on a different expert's data, which is still plausible Q2_0,
    //      so the check that has teeth is the DIFFERENCE between layers.
    {
        int nonzero = 0;
        float smin = 1e30f, smax = -1e30f;
        for (int b = 0; b < cpu::SC_GU; ++b) {
            const float s = f16_at(blob, cpu::O_GU_SCALES + (size_t) b * 2);
            if (s != 0.f) ++nonzero;
            smin = std::fmin(smin, std::fabs(s));
            smax = std::fmax(smax, std::fabs(s));
        }
        const bool ok = nonzero > cpu::SC_GU / 2 && std::isfinite(smin) && std::isfinite(smax) && smax > 0.f;
        std::printf("  %-44s %s (%d/%d non-zero, |s| in [%.4g, %.4g])\n", "blob decodes to live scales",
                    ok ? "yes" : "*** NO ***", nonzero, cpu::SC_GU, (double) smin, (double) smax);
        if (!ok) ++bad;
    }
    {
        std::vector<uint8_t> other;
        if (!synthetic && read_blob(path, (layer * 512 + expert) + 1, other)) {
            long long diff = 0;
            for (size_t i = 0; i < blob.size(); ++i) if (blob[i] != other[i]) ++diff;
            const double frac = (double) diff / (double) blob.size();
            const bool ok = frac > 0.1;
            std::printf("  %-44s %s (%.1f%% of bytes differ)\n", "the next expert is different data",
                        ok ? "yes" : "*** NO ***", frac * 100);
            if (!ok) ++bad;
        }
    }

    // ---- a realistic activation
    std::mt19937 rng(4242);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> x(cpu::H);
    for (auto& v : x) v = gauss(rng);

    cpu::ExpertScratch ws;
    std::vector<float> vnni(cpu::H), scalar_q(cpu::H), scalar_fp(cpu::H);
    cpu::s2_expert_vnni(blob.data(), x.data(), vnni.data(), ws);
    cpu::s2_expert_scalar(blob.data(), x.data(), scalar_q.data(), true);
    cpu::s2_expert_scalar(blob.data(), x.data(), scalar_fp.data(), false);

    double mag = 0;
    const double rel_kernel = rel_l1(scalar_q.data(), vnni.data(), cpu::H, &mag);
    const double rel_contract = rel_l1(scalar_fp.data(), vnni.data(), cpu::H);
    const double trap_contract = rel_l1(scalar_fp.data(), scalar_q.data(), cpu::H);

    // ---- the activation contract must be OBSERVABLE, or check 1 is vacuous
    {
        const bool visible = trap_contract > 1e-4;
        std::printf("  %-44s %s (%.4f%% apart)\n", "the Q8_1 activation contract is observable",
                    visible ? "yes" : "*** NO ***", trap_contract * 100);
        if (!visible) ++bad;
    }

    // ---- 1. the kernel, against an oracle fed the SAME activations
    std::printf("  %-44s rel %.3e   (P2.S3 asks <= 3e-3, mean |ref| %.4f)\n", "VNNI vs scalar (INT8 acts)",
                rel_kernel, mag);
    if (!(rel_kernel <= 3e-3)) { std::printf("    *** over P2.S3's 3e-3 ***\n"); ++bad; }

    // ---- 2. the activation contract's own cost, reported rather than asserted
    std::printf("  %-44s rel %.3e   (the cost of the contract, REPORTED)\n",
                "VNNI vs scalar (FP32 acts)", rel_contract);

    if (synthetic) std::printf("\nexpert_parity: %d failures (synthetic expert)\n", bad);
    else std::printf("\nexpert_parity: %d failures (layer %lld expert %lld of %s)\n", bad, layer, expert, path);
    if (bad) return 1;
    if (selftest) std::printf("expert_parity OK\n");
    return 0;
}
