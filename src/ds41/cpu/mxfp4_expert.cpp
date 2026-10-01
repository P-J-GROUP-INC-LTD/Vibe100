// src/ds41/cpu/mxfp4_expert.cpp - DS-C: the portable half of the MXFP4 CPU experts.
//
// Read include/strata/ds41/cpu/mxfp4_expert.hpp first; it says what is computed and why it is shaped this way.
// This file has no ISA flags: the phases, the row ranges, the SwiGLU, the activation layout, CPUID, the scalar
// reference and the dispatch to the AVX2 / AVX-512 inner loops (mxfp4_avx2.cpp, mxfp4_avx512.cpp).
#include "strata/ds41/cpu/mxfp4_expert.hpp"

#include "mxfp4_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace strata::ds41::cpu {

const int8_t kMxfp4Values[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

float e8m0_half(uint8_t e) {
    // ggml-impl.h ggml_e8m0_to_fp32_half, verbatim: 2^(e - 128); the two denormals by hand, the rest by the exponent
    // field.  (ggml's own comment: NaN (e = 255) is not handled - it decodes to 2^127.)
    uint32_t bits;
    if (e < 2) bits = 0x00200000u << e;
    else bits = (uint32_t) (e - 1) << 23;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// ---- which instruction set ---------------------------------------------------------------------------------------
namespace {

struct CpuBits {
    bool avx2 = false, fma = false, f16c = false;
    bool avx512f = false, avx512dq = false, avx512bw = false, avx512vl = false, avx512vnni = false;
};

CpuBits probe_cpu() {
    CpuBits b;
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    auto cpuid = [&](unsigned leaf, unsigned sub) {
#if defined(_MSC_VER)
        int r[4];
        __cpuidex(r, (int) leaf, (int) sub);
        eax = (unsigned) r[0]; ebx = (unsigned) r[1]; ecx = (unsigned) r[2]; edx = (unsigned) r[3];
#else
        __cpuid_count(leaf, sub, eax, ebx, ecx, edx);
#endif
    };
    cpuid(0, 0);
    const unsigned max_leaf = eax;
    if (max_leaf < 7) return b;
    cpuid(1, 0);
    const bool osxsave = (ecx >> 27) & 1u;
    b.fma = (ecx >> 12) & 1u;
    b.f16c = (ecx >> 29) & 1u;
    const bool avx = (ecx >> 28) & 1u;
    // The OS must have enabled the register state, or the instructions fault: XCR0 bits 1,2 (SSE, AVX) for ymm and
    // 5,6,7 (opmask, ZMM_Hi256, Hi16_ZMM) for zmm.
    uint64_t xcr0 = 0;
    if (osxsave) {
#if defined(_MSC_VER)
        xcr0 = _xgetbv(0);
#else
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        xcr0 = ((uint64_t) hi << 32) | lo;
#endif
    }
    const bool ymm_ok = avx && osxsave && (xcr0 & 0x6u) == 0x6u;
    const bool zmm_ok = ymm_ok && (xcr0 & 0xE0u) == 0xE0u;
    cpuid(7, 0);
    b.avx2 = ymm_ok && ((ebx >> 5) & 1u);
    if (zmm_ok) {
        b.avx512f = (ebx >> 16) & 1u;
        b.avx512dq = (ebx >> 17) & 1u;
        b.avx512bw = (ebx >> 30) & 1u;
        b.avx512vl = (ebx >> 31) & 1u;
        b.avx512vnni = (ecx >> 11) & 1u;
    }
#endif
    return b;
}

const CpuBits& cpu_bits() {
    static const CpuBits b = probe_cpu();
    return b;
}

std::atomic<int> g_prefetch_bytes{8192};

[[noreturn]] void die(const char* what) {
    std::fprintf(stderr, "ds41 cpu mxfp4: %s\n", what);
    std::abort();
}

}  // namespace

bool isa_supported(Isa isa) {
    const CpuBits& c = cpu_bits();
    switch (isa) {
        case Isa::kAuto:
        case Isa::kScalar: return true;
        case Isa::kAvx2: return detail::kernels_avx2() != nullptr && c.avx2 && c.fma && c.f16c;
        case Isa::kAvx512:
            return detail::kernels_avx512() != nullptr && c.avx512f && c.avx512dq && c.avx512bw && c.avx512vl &&
                   c.avx512vnni;
    }
    return false;
}

Isa resolve_isa(Isa isa) {
    if (isa != Isa::kAuto) return isa;
    if (isa_supported(Isa::kAvx512)) return Isa::kAvx512;
    if (isa_supported(Isa::kAvx2)) return Isa::kAvx2;
    return Isa::kScalar;
}

const char* isa_name(Isa isa) {
    switch (isa) {
        case Isa::kAuto: return "auto";
        case Isa::kScalar: return "scalar";
        case Isa::kAvx2: return "avx2";
        case Isa::kAvx512: return "avx512";
    }
    return "?";
}

void set_prefetch_bytes(int bytes) { g_prefetch_bytes.store(bytes < 0 ? 0 : bytes, std::memory_order_relaxed); }
int prefetch_bytes() { return g_prefetch_bytes.load(std::memory_order_relaxed); }

namespace {
const detail::Kernels& kernels_for(Isa isa) {
    isa = resolve_isa(isa);
    if (!isa_supported(isa)) die("the requested ISA is not supported by this CPU / build");
    switch (isa) {
        case Isa::kAvx512: return *detail::kernels_avx512();
        case Isa::kAvx2: return *detail::kernels_avx2();
        default: return detail::kernels_scalar();
    }
}
}  // namespace

// ---- activations -------------------------------------------------------------------------------------------------
void act_unpack(const ActQ& a, int n, int8_t* out) {
    for (int i = 0; i < n; ++i) out[i] = act_q(a, i);
}

void act_dequant(const ActQ& a, int n, float* out) {
    for (int i = 0; i < n; ++i) out[i] = (float) act_q(a, i) * a.scale[i >> 5];
}

namespace detail {
namespace {

// The scalar quantiser: the definition the SIMD ones are bit-identical to.  d = amax / 127 (FP32 division), the
// codes are round-to-nearest-even of x * (127 / amax) (FP32 multiply, then the FP rounding of cvtps2dq / lrintf), the
// all-zero block has d = 0 and q = 0.
void quantize_block_scalar(const float* x, ActQ& a, int blk) {
    float amax = 0.0f;
    for (int j = 0; j < kQK; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float d = amax / 127.0f;
    const float id = amax != 0.0f ? 127.0f / amax : 0.0f;
    int8_t qv[kQK];
    for (int j = 0; j < kQK; ++j) {
        long v = std::lrintf(x[j] * id);
        qv[j] = (int8_t) std::min(127L, std::max(-127L, v));
    }
    const int g = blk >> 2, k = blk & 3;
    for (int j = 0; j < 16; ++j) {
        a.q[g * kGroupValues + k * 16 + j] = qv[j];
        a.q[g * kGroupValues + 64 + k * 16 + j] = qv[16 + j];
    }
    // The correction of lane l (it covers low-half bytes 4l..4l+3 and high-half bytes 4l..4l+3 of this block).
    for (int l = 0; l < 4; ++l) {
        int s = 0;
        for (int i = 0; i < 4; ++i) s += qv[4 * l + i] + qv[16 + 4 * l + i];
        a.corr[g * 16 + k * 4 + l] = -12 * s;
        a.sc4[g * 16 + k * 4 + l] = d;
    }
    a.scale[blk] = d;
}

// The scalar dot: exact integer arithmetic on the natural-order values, ONE fp32 product for the block scale, the
// sum in double.  It never looks at corr / sc4, so it also checks that the quantiser put the right thing in q.
void dot_rows_scalar(const uint8_t* w, size_t row_stride, int ng, int nrows, const ActQ* x, int T, float* out,
                     size_t out_stride, bool accumulate, int /*pf*/) {
    const int nb = ng * kGroupBlocks;
    std::vector<int8_t> qn((size_t) T * nb * kQK);
    for (int t = 0; t < T; ++t) act_unpack(x[t], nb * kQK, qn.data() + (size_t) t * nb * kQK);
    for (int r = 0; r < nrows; ++r) {
        const uint8_t* row = w + (size_t) r * row_stride;
        for (int t = 0; t < T; ++t) {
            const int8_t* q = qn.data() + (size_t) t * nb * kQK;
            double sum = 0.0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* blk = row + (size_t) b * kBlockBytes;
                const float s = e8m0_half(blk[0]) * x[t].scale[b];
                int isum = 0;
                for (int j = 0; j < 16; ++j) {
                    isum += kMxfp4Values[blk[1 + j] & 15] * q[b * kQK + j];
                    isum += kMxfp4Values[blk[1 + j] >> 4] * q[b * kQK + 16 + j];
                }
                sum += (double) s * (double) isum;
            }
            float& o = out[(size_t) t * out_stride + r];
            o = accumulate ? o + (float) sum : (float) sum;
        }
    }
}

const Kernels kScalar = {dot_rows_scalar, quantize_block_scalar, "scalar"};

}  // namespace

const Kernels& kernels_scalar() { return kScalar; }

}  // namespace detail

void act_from_q8(const int8_t* q, const float* scale, int n, ActQ& out) {
    if (n <= 0 || n % kGroupValues != 0 || n > kActMaxBlocks * kQK) die("act_from_q8: n must be a multiple of 128, at most 5120");
    for (int i = 0; i < n; ++i) out.q[act_index(i)] = q[i];
    for (int b = 0; b < n / kQK; ++b) {
        const int g = b >> 2, k = b & 3;
        for (int l = 0; l < 4; ++l) {
            int s = 0;
            for (int i = 0; i < 4; ++i) s += q[b * kQK + 4 * l + i] + q[b * kQK + 16 + 4 * l + i];
            out.corr[g * 16 + k * 4 + l] = -12 * s;
            out.sc4[g * 16 + k * 4 + l] = scale[b];
        }
        out.scale[b] = scale[b];
    }
}

void quantize_act(const float* x, int n, ActQ& out, Isa isa) {
    if (n <= 0 || n % kGroupValues != 0 || n > kActMaxBlocks * kQK) die("quantize_act: n must be a multiple of 128, at most 5120");
    const detail::Kernels& K = kernels_for(isa);
    for (int b = 0; b < n / kQK; ++b) K.quantize_block(x + (size_t) b * kQK, out, b);
}

void quantize_acts(const float* x, int n, int T, ActQ* out, Isa isa) {
    for (int t = 0; t < T; ++t) quantize_act(x + (size_t) t * n, n, out[t], isa);
}

// ---- weights -----------------------------------------------------------------------------------------------------
ExpertView view_cpu_half(const uint8_t* half) {
    ExpertView v;
    v.gate = half + kHalfGate;
    v.up = half + kHalfUp;
    v.down = half + kHalfDown;
    v.down_row_stride = (size_t) kHalfDownRowBlocks * kBlockBytes;
    v.hidden = kHidden;
    v.ff = kHalfFF;
    return v;
}

ExpertView view_blob_half(const uint8_t* blob, int h) {
    ExpertView v;
    v.gate = blob + kBlobGate + (size_t) h * kHalfGateBytes;
    v.up = blob + kBlobUp + (size_t) h * kHalfGateBytes;
    v.down = blob + kBlobDown + (size_t) h * kHalfDownRowBlocks * kBlockBytes;
    v.down_row_stride = kDownRowBytes;
    v.hidden = kHidden;
    v.ff = kHalfFF;
    return v;
}

ExpertView view_blob(const uint8_t* blob) {
    ExpertView v;
    v.gate = blob + kBlobGate;
    v.up = blob + kBlobUp;
    v.down = blob + kBlobDown;
    v.down_row_stride = kDownRowBytes;
    v.hidden = kHidden;
    v.ff = kFF;
    return v;
}

void pack_cpu_half(const uint8_t* blob, int h, uint8_t* out) {
    const ExpertView v = view_blob_half(blob, h);
    std::memcpy(out + kHalfGate, v.gate, kHalfGateBytes);
    std::memcpy(out + kHalfUp, v.up, kHalfGateBytes);
    const size_t piece = (size_t) kHalfDownRowBlocks * kBlockBytes;
    for (int r = 0; r < kHidden; ++r)
        std::memcpy(out + kHalfDown + (size_t) r * piece, v.down + (size_t) r * v.down_row_stride, piece);
}

// ---- the primitive -----------------------------------------------------------------------------------------------
void mxfp4_dot_rows(Isa isa, const uint8_t* w, size_t row_stride, int nblocks, int nrows, const ActQ* x, int T,
                    float* out, size_t out_stride, bool accumulate) {
    if (nblocks <= 0 || nblocks % kGroupBlocks != 0 || nblocks > kActMaxBlocks) die("dot_rows: nblocks must be a multiple of 4");
    if (T < 1 || T > kMaxTokens) die("dot_rows: T must be 1..8");
    if (nrows <= 0) return;
    kernels_for(isa).dot_rows(w, row_stride, nblocks / kGroupBlocks, nrows, x, T, out, out_stride, accumulate,
                              prefetch_bytes());
}

// ---- the expert, in row ranges -----------------------------------------------------------------------------------
void split_range(int n, int parts, int part, int align, int& lo, int& hi) {
    if (align < 1) align = 1;
    const int units = (n + align - 1) / align;                 // `n` rounded up to whole alignment units
    const int u0 = (int) ((long long) units * part / parts);
    const int u1 = (int) ((long long) units * (part + 1) / parts);
    lo = std::min(n, u0 * align);
    hi = std::min(n, u1 * align);
}

namespace {

// h = silu(min(g, 10)) * clamp(u, -10, 10) * w.  FP32, expf: ONE definition for all ISAs, so the intermediate differs
// between them only through g and u (whose FP32 sums differ in the last bits).
inline float swiglu_weighted(float g, float u, float w) {
    g = std::min(g, kSwigluLimit);
    u = std::min(std::max(u, -kSwigluLimit), kSwigluLimit);
    const float s = g / (1.0f + std::exp(-g));
    return (s * u) * w;
}

void check_view(const ExpertView& v) {
    if (v.hidden % kGroupValues != 0 || v.ff % kChunkRows != 0 || (v.ff / kQK) % kGroupBlocks != 0 ||
        v.hidden > kActMaxBlocks * kQK || v.ff > kActMaxBlocks * kQK)
        die("ExpertView: hidden and ff must be multiples of 128 and at most 5120");
}

}  // namespace

void expert_gate_up(Isa isa, const ExpertView& v, const ActQ* x, int T, const float* route_w, ExpertScratch& s,
                    int chunk0, int chunk1) {
    check_view(v);
    if (T < 1 || T > kMaxTokens) die("expert_gate_up: T must be 1..8");
    if (chunk0 < 0 || chunk1 > v.chunks() || chunk0 > chunk1) die("expert_gate_up: chunk range out of bounds");
    if (chunk0 == chunk1) return;
    const detail::Kernels& K = kernels_for(isa);
    const int ng = v.gate_row_blocks() / kGroupBlocks;
    const size_t rb = v.gate_row_bytes();
    const int pf = prefetch_bytes();
    float g[kMaxTokens * kChunkRows], u[kMaxTokens * kChunkRows], hf[kChunkRows];
    for (int c = chunk0; c < chunk1; ++c) {
        const size_t off = (size_t) c * kChunkRows * rb;
        K.dot_rows(v.gate + off, rb, ng, kChunkRows, x, T, g, kChunkRows, false, pf);
        K.dot_rows(v.up + off, rb, ng, kChunkRows, x, T, u, kChunkRows, false, pf);
        for (int t = 0; t < T; ++t) {
            for (int r = 0; r < kChunkRows; ++r)
                hf[r] = swiglu_weighted(g[t * kChunkRows + r], u[t * kChunkRows + r], route_w[t]);
            K.quantize_block(hf, s.h[t], c);
        }
    }
}

void expert_down(Isa isa, const ExpertView& v, const ExpertScratch& s, int T, float* y, int row0, int row1) {
    check_view(v);
    if (T < 1 || T > kMaxTokens) die("expert_down: T must be 1..8");
    if (row0 < 0 || row1 > v.hidden || row0 > row1) die("expert_down: row range out of bounds");
    if (row0 == row1) return;
    const detail::Kernels& K = kernels_for(isa);
    K.dot_rows(v.down + (size_t) row0 * v.down_row_stride, v.down_row_stride, v.down_row_blocks() / kGroupBlocks,
               row1 - row0, s.h, T, y + row0, (size_t) v.hidden, true, prefetch_bytes());
}

void expert_run(Isa isa, const ExpertView& v, const ActQ* x, int T, const float* route_w, ExpertScratch& s, float* y) {
    expert_gate_up(isa, v, x, T, route_w, s, 0, v.chunks());
    expert_down(isa, v, s, T, y, 0, v.hidden);
}

void expert_run_blob(Isa isa, const uint8_t* blob, const ActQ* x, int T, const float* route_w, ExpertScratch& s,
                     float* y) {
    expert_run(isa, view_blob(blob), x, T, route_w, s, y);
}

}  // namespace strata::ds41::cpu
