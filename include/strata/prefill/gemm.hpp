// include/strata/prefill/gemm.hpp - plan v0.3 P5: the batched projections of prompt processing.
//
// Every projection of a chunk of T tokens is Y[T, N] = X[T, K] . W[N, K]^T with W row-major (the GGUF / pack layout)
// and FP32 outputs.  Weights are BF16 on the device - either already (the pack's BF16 tensors) or dequantized from
// their native GGUF blocks into a reusable scratch (`dequant_bf16`) right before the product - and activations are
// rounded to BF16, which is also what llama.cpp's batched CUDA path does.  Tensor-core GEMM through cuBLAS.
//
// VOLTA (the Vibe100 port): it has FP16 tensor cores and no BF16 ones, so a `CUDA_R_16BF` cuBLAS call runs on the
// CUDA cores (~8x below the HMMA peak on a V100).  On a Volta (7.0 <= cc < 7.5) `bf16` therefore converts its
// operands to FP16 on the device - exactly, with a per-chunk power-of-two scale chosen on the device so that no value
// can overflow FP16 and small values are not lost to FP16's subnormal range - and runs the FP16 tensor-core GEMM with
// FP32 accumulation.  The FP16 copies live in the part of the dequantization scratch the call's own operands do not
// occupy, so the route costs no VRAM, allocates nothing (it cannot run out of memory) and never waits on the host.
// See the long comment in gemm.cu.  Turing (which would gain the same way: opt in with STRATA_PREFILL_F16_GEMM=1),
// Ampere and newer, and the HIP build keep the upstream bf16 call, bit for bit.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::prefill {

class Gemm {
public:
    Gemm() = default;
    ~Gemm();
    Gemm(const Gemm&) = delete;
    Gemm& operator=(const Gemm&) = delete;

    /// `scratch_elems`: BF16 elements of the dequantization scratch (the largest weight dequantized at once).
    bool init(void* stream, int64_t scratch_elems, std::string& err);
    /// The same with caller-owned device buffers (the prompt path borrowing expert-cache slots).
    bool init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                       std::string& err);

    /// Y[T, N] (fp32, row stride ldy) = X[T, K] (bf16, row-major) . W[N, K]^T (bf16, row-major).  `beta` = 1 adds.
    /// X and W are only read.  The scratch is this object's own temporary: it may be used while `bf16` runs, except
    /// for the bytes X, W or Y occupy in it (a W dequantized into the scratch is fine; anything else live there is not).
    void bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
              float beta = 0.0f);

    /// Which route `bf16` takes on this object.  -1 (the default): the process-wide STRATA_PREFILL_F16_GEMM, read once -
    /// unset or `auto`: the FP16 tensor-core route when the CURRENT device is a Volta (7.0 <= cc < 7.5), `0`: the
    /// upstream cuBLAS bf16 call everywhere, `1`: the FP16 route on every architecture.  0 / 1 here: the same, for
    /// this object only (they win over the variable).  2: the FP16 route for every shape, including the small ones
    /// (T or N below 64, or too little work to repay the conversion) that 1 and `auto` leave to cuBLAS bf16 - for
    /// tests.  CUDA only: the HIP build always takes its upstream route.
    void set_f16_route(int mode) { f16_mode_ = mode; }
    /// Counters (tests, benchmarks): `bf16` calls that ran the FP16 route, calls that were due to but found no room
    /// for it in the scratch and ran the bf16 call, and the FP16 GEMMs launched (a call is one per T/N chunk).
    int64_t f16_calls() const { return f16_calls_; }
    int64_t f16_fallbacks() const { return f16_fallbacks_; }
    int64_t f16_tiles() const { return f16_tiles_; }

    /// Y = X . W^T with both in FP16 (bits).
    void f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy = 0,
             float beta = 0.0f);

    /// W given as native GGUF blocks of `ggml_type`, dequantized to FP16 in the scratch, X in FP16.
    void native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                int64_t ldy = 0, float beta = 0.0f);

    /// Caller-owned buffers only: the scratch and workspace moved (the prompt path laid its buffers out again).
    void rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes);

    uint16_t* scratch() const { return scratch_; }
    int64_t scratch_elems() const { return scratch_elems_; }
    void* stream() const { return stream_; }

private:
    // The FP16 route of `bf16` (CUDA, gemm.cu): false = not taken (nothing was launched), the caller runs the bf16 call.
    bool bf16_via_f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                      float beta);

    void* handle_ = nullptr;
    void* stream_ = nullptr;
    uint16_t* scratch_ = nullptr;
    int64_t scratch_elems_ = 0;
    void* workspace_ = nullptr;
    bool external_ = false;
    void* hipblaslt_state_ = nullptr;
    int f16_mode_ = -1;
    int64_t f16_calls_ = 0, f16_fallbacks_ = 0, f16_tiles_ = 0;
    bool f16_noted_ = false;               // the "scratch too small" note, once per object
};



}  // namespace strata::prefill
