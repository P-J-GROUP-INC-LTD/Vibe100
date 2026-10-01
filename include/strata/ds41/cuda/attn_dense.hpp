// include/strata/ds41/cuda/attn_dense.hpp - DS1-C: the AttnDenseOps of the attention (attn.hpp) implemented with the dense kernels of DS1-B (dense.hpp) and the
// activation quantiser of DS1-G / DS-D (ds41_cuda.hpp).  This is the adapter the engine uses:
//
//     Ds41AttnDense<RealGeom> dense(dev);                 // link strata_ds41_dense (nvcc) or, for a CPU-emulated program, strata_ds41_dense_emu
//     Ds41Attention<RealGeom> attn(dev, dense);           // + strata_ds41_cuda (the quantiser), or compile src/ds41/cuda/ds41_emu_impl.cpp into the program
//
// Header-only and a template, so the attention package itself (strata_ds41_attn) does not depend on those libraries: only a program that instantiates this
// class links them.  Semantics are exactly the four AttnDenseOps contracts (attn.hpp): the activation quantiser is ds41_quantize_acts<G>(..., ActOrder::kNatural)
// (CONTRACTS.md's rule bit for bit; xq int8 [T][k/32][32] natural order, xs fp32 [T][k/32]), the Q8_0 GEMV with int8 activations is ds41_gemv_q8_int8 (exact
// integer sums, fma(d_w * d_x, isum, acc) per block), wo_a is ds41_gemv_q8_grouped_f32 (FP32 activations), the BF16 projections are ds41_gemv_bf16.
// Alignment (DS1-B's rules, which the emulator asserts): FP32 activations and BF16 weights 16 bytes (the driver's scratch and the GGUF tensors are), k % 8 == 0.
#pragma once

#include "strata/ds41/cuda/attn.hpp"
#include "strata/ds41/cuda/dense.hpp"
#include "strata/ds41/cuda/ds41_cuda.hpp"

namespace strata::ds41::cuda {

template <class G>
class Ds41AttnDense final : public AttnDenseOps {
   public:
    explicit Ds41AttnDense(Dev& dev) : dev_(dev) {}
    void quantize_acts(const float* x, int T, int k, int8_t* xq, float* xs, Stream s) override { ds41_quantize_acts<G>(dev_, x, T, k, xq, xs, s, ActOrder::kNatural); }
    void gemv_q8(const void* w, int n, int k, const int8_t* xq, const float* xs, int T, float* y, Stream s) override {
        ds41_gemv_q8_int8<G>(dev_, w, n, k, xq, xs, T, y, s);
    }
    void gemv_q8_grouped_f32(const void* w, int groups, int rows_per_group, int k_per_group, const float* x, int T, float* y, Stream s) override {
        ds41_gemv_q8_grouped_f32<G>(dev_, w, groups, rows_per_group, k_per_group, x, T, y, s);
    }
    void gemv_bf16(const uint16_t* w, int n, int k, const float* x, int T, float* y, Stream s) override { ds41_gemv_bf16<G>(dev_, w, n, k, x, T, y, s); }

   private:
    Dev& dev_;
};

}  // namespace strata::ds41::cuda
