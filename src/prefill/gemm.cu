// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// A setup call whose failure the engine survives (the handle keeps its defaults), as before #240 - but said.
void note(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d (continuing)\n", what, (int) s);
}

#if !defined(__HIPCC__)
// ================================================================================================================
// bf16 PROJECTIONS ON A DEVICE WITHOUT BF16 TENSOR CORES: THE FP16 ROUTE (Volta sm_70, Turing sm_75).
//
// WHY.  `cublasGemmEx` with CUDA_R_16BF inputs runs on the tensor cores from Ampere on.  Volta and Turing have FP16
// tensor cores only, so the same call falls to the FP32 FMA pipe: on a V100 ~15 TFLOPS against 125 for HMMA.  Every
// resident-BF16 projection of the prompt path (the hyper-connection read/write, the router, the indexer, the PLE
// key/value, the GDN gates) goes through `Gemm::bf16`, so on those cards it is the prefill's largest avoidable cost.
//
// WHAT.  bf16 -> fp16 on the device, then the FP16 tensor-core GEMM with FP32 accumulation and FP32 output (the
// call `Gemm::f16` already makes for the experts).  The conversion is EXACT whenever the value is in fp16's normal
// range, because a bf16 has 8 significant bits and an fp16 11: for every chunk without a value above 2^15 the route
// multiplies the very numbers the bf16 route multiplies - each product is exact in FP32 - and the two differ only in
// the order the FP32 accumulation adds them (the same ~1e-6..1e-5 of the output scale any two cuBLAS algorithms
// differ by).  The activation contract (BF16 activations against BF16 weights) is untouched.
//
// THE RANGE.  bf16 has 8 exponent bits, fp16 has 5.  Above 65504 an fp16 is infinite; below 2^-14 it is subnormal
// (a 2^-24 grid), below 2^-25 zero.  Activations of large models do exceed 65504 (the "massive" channels of the
// residual stream), and one infinity in an operand would poison every output it touches.  So each operand chunk is
// scaled by a POWER OF TWO, chosen ON THE DEVICE: a reduction finds the largest finite |bf16| of the chunk, its
// biased exponent E gives k = max(0, E - 141) - the smallest k >= 0 with max * 2^-k < 2^15 - and the chunk is
// converted as x * 2^-k.  A power of two changes no mantissa bit.  The product of an X chunk (k_x) and a W chunk
// (k_w) is then 2^-(k_x + k_w) times the true one, which the GEMM's own alpha puts back, in FP32, exactly:
// alpha = 2^(k_x + k_w) (clamped at 2^127, which is only reachable when |x| |w| > 2^155 and the upstream route
// overflows FP32 as well).  k = 0 - alpha = 1, inputs bit-identical to the bf16 route - for every chunk whose
// largest finite value is below 32768, i.e. for every ordinary activation and weight.  Because alpha and beta are
// then DEVICE values the call is made in CUBLAS_POINTER_MODE_DEVICE and the handle's pointer mode is restored; the
// host never waits for the reduction, so nothing here synchronizes and the route is as asynchronous as the bf16 call
// (it also allocates nothing, so it adds nothing that stream capture forbids).
//   * Inf and NaN are left out of the maximum, converted to Inf / NaN, and reach the same outputs as in the bf16
//     route; a chunk with one Inf and otherwise ordinary values keeps k = 0 and its finite rows stay exact.
//   * THE UNDERFLOW CONSEQUENCE.  Scaling down gives up the bottom of the range for the top.  After x * 2^-k an
//     element is rounded to fp16: it keeps all its bits down to 2^(-14+k), is rounded to a grid of 2^(-24+k) below
//     that (absolute error <= 2^(-25+k)) and is lost below 2^(-25+k).  With k = 0 that is already the case for the
//     tiny elements of an ordinary chunk (< 6.1e-5), whose absolute error is <= 3e-8 each.  A chunk with a huge outlier
//     (its maximum is ~2^(14+k) after scaling) loses precision on what lies more than ~2^28 below the maximum and
//     all of what lies more than ~2^39 below it - in the row or column that holds the outlier far under the FP32
//     accumulation noise, but in the OTHER rows of the chunk (the scale is the chunk's, not the row's) it is a real
//     error: an outlier of 1e8 among unit-variance activations (k = 12) puts the fifth of that chunk's elements on a
//     2.4e-4 grid, ~3e-5 of the output scale (the parity case "one outlier among ordinary").  Real activations
//     exceed 65504 by a small factor, k stays 1..3 and nothing is lost that matters.  A chunk dominated by tiny values
//     (rms below ~1e-4; the relative error then approaches 2^-25 / rms) belongs to the upstream route
//     (STRATA_PREFILL_F16_GEMM=0).  Scaling UP by a power of two would remove that case at no cost in mantissa bits;
//     it is not done, so that k = 0 keeps meaning "the same inputs as the bf16 route".
//
// WHERE THE FP16 COPIES LIVE: IN THE GEMM SCRATCH, so the route costs no VRAM and cannot fail to allocate.  The
// scratch is this object's own temporary - `native` dequantizes into it and consumes it before returning, and in
// the prompt path it is borrowed expert-cache slots lent for the run - so between calls it is dead.  The copies go
// in the stretch of it that this call's X, W and Y do not occupy (a caller may well have dequantized the W of this
// very call into the scratch, and a W used by two calls must survive the first: nothing is converted IN PLACE, a
// caller's operand is never written).  The engine's scratch is 64 MiB: the X of a 8192-token chunk (K = 2560) is
// 40 MiB and the largest resident W 12.5 MiB, so those run as one GEMM; K = 10240 (the hyper-connection) fits 3276
// rows, and the X / W rows are cut into T / N chunks - never K, a K split would need beta = 1 passes and would write
// every Y element twice.  Each chunk pair is one GEMM into its own sub-block of Y (Y + t0 * ldy + n0, row stride
// ldy), so every Y element is written by exactly one call and beta means what it meant: beta = 0 never reads Y,
// beta = 1 adds.  The 64-row rounding of a chunk keeps the sub-blocks' addresses 256-byte aligned.  If the stretch
// cannot hold 64 rows of X and of W (a tiny scratch, one with no scratch at all, or both operands in it) the route
// is not taken, one line says so, and the upstream bf16 call runs - no allocation, so no OOM path.  The 256-byte
// device state (the two maxima, alpha, beta) sits at the head of the same stretch and is rewritten on every call.
// ORDER: everything is on stream_, behind whatever wrote the operands and in front of whatever reads Y or reuses the
// scratch (`native`'s next dequantization), which is all the scratch needs.
//
// WHICH CALLS.  STRATA_PREFILL_F16_GEMM (once): unset / `auto` - this route when the CURRENT device has FP16 tensor
// cores and no BF16 ones, 7.0 <= cc < 8.0 (a layer split can mix GPUs, so it is decided per call; cc and SM count
// are cached per ordinal; before Volta there are no tensor cores to gain, and a GeForce Pascal runs FP16 at 1/64 of
// its FP32 rate), `0` - the upstream call always, `1` - this route on any architecture (to test it on a newer card).
// Not worth the conversion, so left to the bf16 call: T < 64, N < 64 or T * N * K < 2^29.  The conversion moves ~6
// bytes per element of X and of W (the maximum pass reads 2, the conversion reads 2 and writes 2) and costs ~7 launches
// (~30 us); what it buys is 2 T N K (1/11e12 - 1/90e12) s = 1.6e-13 T N K s at V100 rates.  The two are equal near
// min(T, N) ~ 50 and the fixed cost is repaid from T N K ~ 2e8 (2^29 leaves a margin), so below those the bf16 call
// is as fast.  That keeps the N = 1 router gate, the 4-wide injection and the 48-wide GDN gates (all bandwidth-bound
// reads of X) and every draft-layer call (T <= 32) exactly as upstream.
//
// TENSOR CORES.  `CUBLAS_GEMM_DEFAULT` is the call: the handle is in CUBLAS_DEFAULT_MATH (init), and with FP16 inputs
// and CUBLAS_COMPUTE_32F cuBLAS >= 11 selects tensor-core kernels by itself - which is what `Gemm::f16`, the expert
// path, has always relied on - provided the pointers and leading dimensions are aligned: hence the 256-byte aligned
// copies and K padded with zeros (in both operands, so the pad adds nothing) to a multiple of 8, lda = ldb = Kp a
// multiple of 16 bytes for any K.  The *_TENSOR_OP algorithm enumerators (CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99) are
// still declared in the 12.8 header and are not needed; the header marks only CUBLAS_TENSOR_OP_MATH deprecated (the
// cuBLAS manual is not in this tree, so the proof is the measurement: `gemm_volta_parity` prints the achieved
// TFLOPS of this route and of a bare FP16 cublasGemmEx, and more than ~35 can only be HMMA).
// ================================================================================================================
constexpr int kRouteOff = 0, kRouteOn = 1, kRouteAll = 2, kRouteAuto = 3;   // f16_mode_ 0 / 1 / 2, and the variable's `auto`
constexpr int kF16Threads = 256;
constexpr int64_t kF16StateBytes = 256;           // the device scalars at the head of the stretch
enum : int { kStMaxX = 0, kStMaxW = 1, kStAlpha = 2, kStBeta = 3 };   // their uint32 words
constexpr uint32_t kBf16InfBits = 0x7F80u;        // |bf16| from this value up is Inf or NaN
constexpr uint32_t kBf16E2p14 = 141u;             // the biased exponent of 2^14: a scaled chunk's maximum stays below 2^15

// the magnitude bits of a bf16, 0 for an Inf or a NaN (which do not take part in choosing the scale)
__host__ __device__ __forceinline__ uint32_t finite_mag(uint32_t b) {
    b &= 0x7FFFu;
    return b < kBf16InfBits ? b : 0u;
}
__host__ __device__ __forceinline__ uint32_t finite_mag2(uint32_t w) {
    const uint32_t lo = finite_mag(w & 0xFFFFu), hi = finite_mag(w >> 16);
    return lo > hi ? lo : hi;
}
// k of a chunk whose largest finite magnitude has these bits: the smallest k >= 0 with max * 2^-k < 2^15
__host__ __device__ __forceinline__ uint32_t scale_k(uint32_t max_bits) {
    const uint32_t e = max_bits >> 7;
    return e > kBf16E2p14 ? e - kBf16E2p14 : 0u;
}

// max over the finite |values| of count bf16 starting at p (any 2-byte alignment): 16-byte loads over the aligned
// body, scalars for the head and tail; one atomicMax per block (non-negative, so the integer order is the float's)
__global__ void __launch_bounds__(kF16Threads) bf16_absmax_kernel(const uint16_t* __restrict__ p, int64_t count,
                                                                  uint32_t* __restrict__ slot) {
    uint32_t m = 0;
    const int64_t tid = (int64_t) blockIdx.x * kF16Threads + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * kF16Threads;
    int64_t head = (int64_t) (((16u - ((uint32_t) (uintptr_t) p & 15u)) & 15u) >> 1);   // elements to the 16-byte boundary
    if (head > count) head = count;
    const int64_t nvec = (count - head) >> 3;
    const int64_t tail0 = head + nvec * 8;
    const uint4* v = reinterpret_cast<const uint4*>(p + head);
#pragma unroll 4
    for (int64_t i = tid; i < nvec; i += stride) {
        const uint4 q = __ldg(v + i);
        m = max(m, max(max(finite_mag2(q.x), finite_mag2(q.y)), max(finite_mag2(q.z), finite_mag2(q.w))));
    }
    const int64_t nscalar = head + (count - tail0);
    for (int64_t r = tid; r < nscalar; r += stride) m = max(m, finite_mag(p[r < head ? r : tail0 + (r - head)]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
    __shared__ uint32_t sh[kF16Threads / 32];
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        m = threadIdx.x < kF16Threads / 32 ? sh[threadIdx.x] : 0u;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (threadIdx.x == 0 && m != 0u) atomicMax(slot, m);
    }
}

// bf16 bits -> the fp16 bits of x * 2^-k, round-to-nearest-even, in integers: the exponent field moves by 112 + k
// (bf16 bias 127, fp16 bias 15) and the 7 mantissa bits go to the top of the 10, so a value that stays in fp16's
// normal range is converted exactly and with no arithmetic at all; below it the implicit 1 is restored and shifted into
// the 2^-24 grid (subnormal, rounded to nearest even, the carry into the smallest normal falls out of the add), and
// below 2^-25 the result is a signed zero.  It deliberately does not go through `__float2half`: this is the
// conversion the route's correctness rests on, and f16_bits.hpp records that `__float2half` + `__half_as_ushort` once
// produced wrong bits in this tree (0x2600 for 0x26DB).  Being plain integer code it also runs on the host, which is
// how it was checked against numpy's float16 over every bf16 pattern and every k.  Inf and NaN stay Inf and NaN.
__host__ __device__ __forceinline__ uint32_t bf16_to_f16_bits(uint32_t b, uint32_t k) {
    const uint32_t E = (b >> 7) & 0xFFu;
    const uint32_t e = E - k - 112u;                           // the fp16 biased exponent of x * 2^-k (wraps when < 0)
    if (e - 1u < 30u && E != 0xFFu)                            // e in 1..30, and not an Inf / NaN: the common case, exact
        return (b & 0x8000u) | (e << 10) | ((b & 0x7Fu) << 3);
    const uint32_t sign = b & 0x8000u, m = b & 0x7Fu;
    if (E == 0xFFu) return sign | (m ? 0x7E00u : 0x7C00u);
    if (E == 0u) return sign;                                  // bf16 zero or subnormal (< 2^-126): zero in fp16
    if ((int) e >= 31) return sign | 0x7C00u;                  // unreachable for a value <= its chunk's maximum
    const int sh = -((int) e + 2);                             // subnormal: q = (128 + m) * 2^(e + 2), rounded
    const uint32_t mant = 128u | m;
    if (sh <= 0) return sign | (mant << (uint32_t) (-sh));     // e = 0, -1, -2: exact
    if (sh > 8) return sign;                                   // below half the smallest subnormal
    uint32_t q = mant >> sh;
    const uint32_t rem = mant & ((1u << sh) - 1u), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (q & 1u))) ++q;
    return sign | q;
}
// two bf16 (the halves of w) -> two fp16 in one word
__device__ __forceinline__ uint32_t bf16x2_to_f16x2(uint32_t w, uint32_t k) {
    return bf16_to_f16_bits(w & 0xFFFFu, k) | (bf16_to_f16_bits(w >> 16, k) << 16);
}

// the common case, K a multiple of 8 and the source 16-byte aligned: the chunk is one contiguous run of 8-element
// groups, converted 16 bytes to 16 bytes, coalesced, no index arithmetic beyond the grid-stride
__global__ void __launch_bounds__(kF16Threads) bf16_to_f16_flat_kernel(const uint4* __restrict__ in, uint4* __restrict__ out,
                                                                       int64_t n8, const uint32_t* __restrict__ slot) {
    const uint32_t k = scale_k(__ldg(slot));
    const int64_t stride = (int64_t) gridDim.x * kF16Threads;
    for (int64_t i = (int64_t) blockIdx.x * kF16Threads + threadIdx.x; i < n8; i += stride) {
        const uint4 q = in[i];
        uint4 r;
        r.x = bf16x2_to_f16x2(q.x, k); r.y = bf16x2_to_f16x2(q.y, k);
        r.z = bf16x2_to_f16x2(q.z, k); r.w = bf16x2_to_f16x2(q.w, k);
        out[i] = r;
    }
}
// any other K or alignment: row r of the source (stride K) to row r of the destination (stride Kp, the pad zero)
__global__ void __launch_bounds__(kF16Threads) bf16_to_f16_rows_kernel(const uint16_t* __restrict__ in,
                                                                       uint16_t* __restrict__ out, int64_t rows, int64_t K,
                                                                       int64_t Kp, const uint32_t* __restrict__ slot) {
    const uint32_t k = scale_k(__ldg(slot));
    for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        const uint16_t* src = in + r * K;
        uint16_t* dst = out + r * Kp;
        for (int64_t c = threadIdx.x; c < Kp; c += kF16Threads) dst[c] = (uint16_t) (c < K ? bf16_to_f16_bits(src[c], k) : 0u);
    }
}
// alpha = 2^(k_x + k_w), beta as given: the GEMM's scalars, from the two maxima (a launch argument is captured with
// the launch, so this stays right in a graph)
__global__ void f16_scale_kernel(uint32_t* st, float beta) {
    const uint32_t k = min(scale_k(st[kStMaxX]) + scale_k(st[kStMaxW]), 127u);
    st[kStAlpha] = (127u + k) << 23;
    st[kStBeta] = __float_as_uint(beta);
}

void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "prefill gemm: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
unsigned grid_for(int64_t work, int sms) {   // ~one 256-thread block per 256 work items, at most 8 per SM
    return (unsigned) std::min<int64_t>(std::max<int64_t>((work + kF16Threads - 1) / kF16Threads, 1), (int64_t) sms * 8);
}
// the largest finite |value| of count bf16 -> *slot (bits of the bf16 magnitude; 0 when there is none)
void absmax_chunk(const uint16_t* src, int64_t count, uint32_t* slot, int sms, cudaStream_t st) {
    cudaMemsetAsync(slot, 0, sizeof(uint32_t), st);
    bf16_absmax_kernel<<<grid_for(count / 8 + 1, sms), kF16Threads, 0, st>>>(src, count, slot);
    launch_check("bf16 max");
}
// rows x K bf16 (stride K) -> rows x Kp fp16 (stride Kp), scaled by 2^-k of *slot
void convert_chunk(const uint16_t* src, uint16_t* dst, int64_t rows, int64_t K, int64_t Kp, const uint32_t* slot,
                   int sms, cudaStream_t st) {
    if (Kp == K && (reinterpret_cast<uintptr_t>(src) & 15u) == 0) {
        const int64_t n8 = rows * K / 8;
        bf16_to_f16_flat_kernel<<<grid_for(n8, sms), kF16Threads, 0, st>>>(reinterpret_cast<const uint4*>(src),
                                                                           reinterpret_cast<uint4*>(dst), n8, slot);
    } else {
        bf16_to_f16_rows_kernel<<<(unsigned) std::min<int64_t>(rows, (int64_t) sms * 8), kF16Threads, 0, st>>>(
            src, dst, rows, K, Kp, slot);
    }
    launch_check("bf16 -> f16");
}

// STRATA_PREFILL_F16_GEMM, read once
int f16_env_mode() {
    static const int mode = [] {
        const char* e = std::getenv("STRATA_PREFILL_F16_GEMM");
        if (!e || !*e || std::strcmp(e, "auto") == 0) return kRouteAuto;
        if (std::strcmp(e, "0") == 0) return kRouteOff;
        if (std::strcmp(e, "1") == 0) return kRouteOn;
        std::fprintf(stderr, "prefill gemm: STRATA_PREFILL_F16_GEMM=%s is not auto, 0 or 1; using auto\n", e);
        return kRouteAuto;
    }();
    return mode;
}
// the CURRENT device's compute capability (major * 10 + minor) and SM count, per ordinal, cached
bool f16_device(int& cc, int& sms) {
    static std::atomic<int> cc_of[64], sms_of[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return false; }
    int c = cc_of[dev].load(std::memory_order_relaxed), n = sms_of[dev].load(std::memory_order_relaxed);
    if (c == 0 || n == 0) {
        int major = 0, minor = 0, count = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || major <= 0 || count <= 0) {
            cudaGetLastError();
            return false;
        }
        c = major * 10 + minor;
        n = count;
        cc_of[dev].store(c, std::memory_order_relaxed);
        sms_of[dev].store(n, std::memory_order_relaxed);
    }
    cc = c;
    sms = n;
    return true;
}
// The T / N chunk sizes for room for R rows (of Kp halves) in all: both operands whole when they fit; else whichever
// is at most half of R whole and the other takes the rest, else half each.  A chunk that is cut is a multiple of 64
// rows, which keeps the Y sub-blocks' addresses aligned.  false: not even 64 rows of each (or all of a smaller one) fit.
bool plan_f16_chunks(int64_t R, int64_t T, int64_t N, int64_t& Tc, int64_t& Nc) {
    Nc = N;
    Tc = T;
    if (R <= 0) return false;
    if (Nc + Tc > R) {
        if (N <= R / 2) { Nc = N; Tc = R - N; }
        else if (T <= R / 2) { Tc = T; Nc = R - T; }
        else { Nc = R / 2; Tc = R - Nc; }
        if (Nc < N) Nc = Nc / 64 * 64;
        if (Tc < T) Tc = Tc / 64 * 64;
    }
    return Nc >= std::min<int64_t>(N, 64) && Tc >= std::min<int64_t>(T, 64);
}
// the largest stretch [a0, a1) of [s0, s1) that none of the operand spans touches
struct Span { uintptr_t b, e; };
void free_stretch(uintptr_t s0, uintptr_t s1, const Span* ops, int n, uintptr_t& a0, uintptr_t& a1) {
    uintptr_t lo = s1, hi = s0;
    bool any = false;
    for (int i = 0; i < n; ++i) {
        if (ops[i].b >= s1 || ops[i].e <= s0) continue;
        any = true;
        lo = std::min(lo, std::max(ops[i].b, s0));
        hi = std::max(hi, std::min(ops[i].e, s1));
    }
    if (!any) { a0 = s0; a1 = s1; return; }
    if (lo - s0 >= s1 - hi) { a0 = s0; a1 = lo; } else { a0 = hi; a1 = s1; }
}
#endif

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

Gemm::~Gemm() {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    workspace_ = workspace;
    note(cublasSetWorkspace(h, workspace_, ws_bytes), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    // #240: every failure names the call and the real status, so "no VRAM" can be told from a broken install
    cublasHandle_t h = nullptr;
    if (const cublasStatus_t s = cublasCreate(&h); s != CUBLAS_STATUS_SUCCESS) {
        err = "prefill gemm: cublasCreate: cuBLAS status " + std::to_string((int) s);
        return false;
    }
    handle_ = h;
    stream_ = stream;
    note(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (const cudaError_t e = cudaMalloc(&workspace_, ws); e != cudaSuccess) {
        err = std::string("prefill gemm: workspace of 32 MiB: ") + cudaGetErrorString(e);
        return false;
    }
    note(cublasSetWorkspace(h, workspace_, ws), "cublasSetWorkspace");
    note(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0) {
        if (const cudaError_t e = cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2); e != cudaSuccess) {
            err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB: " +
                  cudaGetErrorString(e);
            return false;
        }
    }
    scratch_elems_ = scratch_elems;
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        return;
    }
#endif
#if !defined(__HIPCC__)
    // No BF16 tensor cores before Ampere: FP16 HMMA on exactly-converted, power-of-two-scaled copies (the long comment
    // above).  false: the route was not taken - by choice, or for want of room - and nothing was launched.
    if (bf16_via_f16(X, W, Y, T, N, K, ldy, beta)) return;
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

#if !defined(__HIPCC__)
bool Gemm::bf16_via_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                        float beta) {
    // ---- is this call for the FP16 route?  (per call, from the CURRENT device)
    const int mode = f16_mode_ >= 0 ? f16_mode_ : f16_env_mode();
    if (mode == kRouteOff) return false;
    int cc = 0, sms = 0;
    if (!f16_device(cc, sms)) return false;
    // auto: FP16 tensor cores without BF16 ones - Volta and Turing.  Before Volta there are no tensor cores to gain
    // (and a GeForce Pascal runs FP16 at 1/64 of its FP32 rate), so there the bf16 call stays; `1` forces the route.
    if (mode == kRouteAuto && (cc < 70 || cc >= 80)) return false;
    // cublasGemmEx takes int sizes; ldy < N is the caller's error, which the bf16 call reports as before
    if (K < 1 || K > INT_MAX - 8 || N > INT_MAX || T > INT_MAX || ldy > INT_MAX || ldy < N) return false;
    // too small to repay the conversion (see "WHICH CALLS" above); the N = 1 router gate is one of these
    if (mode != kRouteAll && (T < 64 || N < 64 || (double) T * (double) N * (double) K < 536870912.0)) return false;

    // ---- the room: the part of the scratch that X, W and Y do not occupy
    const int64_t Kp = (K + 7) & ~(int64_t) 7;     // zero-padded to a multiple of 8: 16-byte rows for HMMA
    const int64_t row_bytes = Kp * 2;
    uintptr_t a0 = 0, a1 = 0;
    if (scratch_ != nullptr && scratch_elems_ > 0) {
        const uintptr_t s0 = reinterpret_cast<uintptr_t>(scratch_), s1 = s0 + (uintptr_t) scratch_elems_ * 2;
        const Span ops[3] = {
            {reinterpret_cast<uintptr_t>(X), reinterpret_cast<uintptr_t>(X) + (uintptr_t) (T * K) * 2},
            {reinterpret_cast<uintptr_t>(W), reinterpret_cast<uintptr_t>(W) + (uintptr_t) (N * K) * 2},
            {reinterpret_cast<uintptr_t>(Y), reinterpret_cast<uintptr_t>(Y) + (uintptr_t) ((T - 1) * ldy + N) * 4}};
        free_stretch(s0, s1, ops, 3, a0, a1);
        a0 = (a0 + 255) & ~(uintptr_t) 255;
    }
    // rows of Kp halves that fit after the state and the alignment of the second block (256 bytes each)
    const int64_t avail = a1 > a0 ? (int64_t) (a1 - a0) - kF16StateBytes - 256 : 0;
    const int64_t R = avail > 0 ? avail / row_bytes : 0;
    int64_t Nc = 0, Tc = 0;
    if (!plan_f16_chunks(R, T, N, Tc, Nc)) {
        ++f16_fallbacks_;
        if (!f16_noted_) {
            f16_noted_ = true;
            std::fprintf(stderr,
                         "prefill gemm: no room for the FP16 route in the %lld MiB scratch (T=%lld N=%lld K=%lld); "
                         "using the bf16 cuBLAS call (CUDA cores: no BF16 tensor cores before Ampere)\n",
                         (long long) (scratch_elems_ * 2 >> 20), (long long) T, (long long) N, (long long) K);
        }
        return false;
    }
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true)) {
        std::fprintf(stderr,
                     "prefill gemm: bf16 projections run as FP16 tensor-core GEMMs on this device (cc %d.%d: exact "
                     "bf16->fp16, power-of-two scaled, FP32 accumulate; staged in the GEMM scratch, no extra VRAM); "
                     "STRATA_PREFILL_F16_GEMM=0 restores the cuBLAS bf16 call\n",
                     cc / 10, cc % 10);
    }

    // ---- the work: (W chunk, X chunk) tiles, each one GEMM into its own block of Y, all on stream_
    uint8_t* const base = reinterpret_cast<uint8_t*>(a0);
    uint32_t* const st = reinterpret_cast<uint32_t*>(base);
    uint16_t* const w16 = reinterpret_cast<uint16_t*>(base + kF16StateBytes);
    uint16_t* const x16 = reinterpret_cast<uint16_t*>(base + kF16StateBytes + (((uint64_t) Nc * (uint64_t) row_bytes + 255) & ~255ull));
    const cudaStream_t cs = (cudaStream_t) stream_;
    cublasHandle_t h = (cublasHandle_t) handle_;
    // alpha and beta are device values (they depend on the data): the handle's pointer mode is ours until we are done
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    ck(cublasGetPointerMode(h, &pointer_mode), "cublasGetPointerMode");
    ck(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE), "cublasSetPointerMode");
    int64_t x_chunk = -1;                         // the t0 of the X chunk now in x16 (it is kept across W chunks when X is whole)
    for (int64_t n0 = 0; n0 < N; n0 += Nc) {
        const int64_t nn = std::min(Nc, N - n0);
        absmax_chunk(W + n0 * K, nn * K, st + kStMaxW, sms, cs);
        convert_chunk(W + n0 * K, w16, nn, K, Kp, st + kStMaxW, sms, cs);
        for (int64_t t0 = 0; t0 < T; t0 += Tc) {
            const int64_t tt = std::min(Tc, T - t0);
            if (x_chunk != t0) {
                absmax_chunk(X + t0 * K, tt * K, st + kStMaxX, sms, cs);
                convert_chunk(X + t0 * K, x16, tt, K, Kp, st + kStMaxX, sms, cs);
                x_chunk = t0;
            }
            f16_scale_kernel<<<1, 1, 0, cs>>>(st, beta);
            launch_check("f16 scale");
            // as the bf16 call: Y^T[nn, tt] = W16[nn, Kp] (col-major K x nn, transposed) . X16^T[Kp, tt], in Y's block
            ck(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) tt, (int) Kp, st + kStAlpha, w16, CUDA_R_16F,
                            (int) Kp, x16, CUDA_R_16F, (int) Kp, st + kStBeta, Y + t0 * ldy + n0, CUDA_R_32F, (int) ldy,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
               "cublasGemmEx bf16 via f16");
            ++f16_tiles_;
        }
    }
    ck(cublasSetPointerMode(h, pointer_mode), "cublasSetPointerMode");
    ++f16_calls_;
    return true;
}
#endif

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        return;
    }
#endif
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
