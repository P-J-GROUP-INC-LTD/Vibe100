// include/strata/ds41/model/host_dequant.hpp - DS1-A: host dequantisation of the GGUF encodings this model stores, bit-exact against GGML's
// C reference (ggml-quants.c; see tools/ds41/ggml_codecs.py): F32, F16, BF16, Q8_0, MXFP4.
//
// Used by the loader's helpers (token_embd row lookup), by the tests, and by anyone who needs a weight on the host.  The kernels have their
// own dequantisation; this is the reference they are compared with.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "strata/ds41/model/gguf.hpp"

namespace strata::ds41::model {

inline float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// IEEE half -> float, exact for every input (denormals, infinities, NaN payloads).
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) bits = sign;
        else {
            const float v = std::ldexp((float) man, -24);          // man * 2^-24: exact
            std::memcpy(&bits, &v, 4);
            bits |= sign;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// ggml_e8m0_to_fp32_half(e) = 2^(e - 128), exactly (e = 0 and 1 are float32 denormals, e = 255 is 2^127).
inline float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t) (e - 1) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// kvalues_mxfp4: the e2m1 table doubled (the 0.5 lives in e8m0_half).
inline constexpr int8_t kMxfp4Values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/// `n` elements (a whole number of blocks for Q8_0 / MXFP4) of encoding `t` at `raw` -> `out`.  Throws for another type.
inline void dequantize(GgmlType t, const uint8_t* raw, uint64_t n, float* out) {
    switch (t) {
    case GgmlType::F32:
        std::memcpy(out, raw, (size_t) n * 4);
        return;
    case GgmlType::F16:
        for (uint64_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, raw + 2 * i, 2);
            out[i] = f16_to_f32(h);
        }
        return;
    case GgmlType::BF16:
        for (uint64_t i = 0; i < n; ++i) {
            uint16_t h;
            std::memcpy(&h, raw + 2 * i, 2);
            out[i] = bf16_to_f32(h);
        }
        return;
    case GgmlType::Q8_0:
        if (n % 32) throw ModelError("dequantize: Q8_0 needs whole blocks of 32 values");
        for (uint64_t b = 0; b < n / 32; ++b) {
            const uint8_t* blk = raw + b * 34;
            uint16_t h;
            std::memcpy(&h, blk, 2);
            const float d = f16_to_f32(h);
            for (int j = 0; j < 32; ++j) out[b * 32 + j] = (float) (int8_t) blk[2 + j] * d;
        }
        return;
    case GgmlType::MXFP4:
        if (n % 32) throw ModelError("dequantize: MXFP4 needs whole blocks of 32 values");
        for (uint64_t b = 0; b < n / 32; ++b) {
            const uint8_t* blk = raw + b * 17;
            const float d = e8m0_half(blk[0]);
            for (int j = 0; j < 16; ++j) {
                out[b * 32 + j] = (float) kMxfp4Values[blk[1 + j] & 0x0F] * d;
                out[b * 32 + j + 16] = (float) kMxfp4Values[blk[1 + j] >> 4] * d;
            }
        }
        return;
    default:
        throw ModelError("dequantize: no host decoder for " + type_name(t));
    }
}

}  // namespace strata::ds41::model
