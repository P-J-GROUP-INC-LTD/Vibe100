// src/ds41/cuda/attn_real.cu - DS1-C: the attention kernels (attn_impl.cuh) built by nvcc for sm_70, for the real model's geometry.
// MiniGeom is instantiated only by the emulator tests (src/ds41/attn/attn_emu_test.cpp), which compile attn_impl.cuh for the host.
#include "attn_impl.cuh"

namespace strata::ds41::cuda {
template struct AttnKernels<RealGeom>;
}  // namespace strata::ds41::cuda
