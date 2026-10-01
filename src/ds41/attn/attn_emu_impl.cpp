// src/ds41/attn/attn_emu_impl.cpp - DS1-C: the attention kernels (src/ds41/cuda/attn_impl.cuh) and the per-layer driver (attn_host_impl.hpp) compiled for the
// HOST: -DDS41_EMU runs every GPU thread on the CPU through the thread-model emulation (src/ds41/cuda/ds41_emu.hpp).  Same source as the nvcc build
// (attn_real.cu / attn_real.cpp), so the logic that runs here is the logic that runs on the V100.  Instantiated for both geometries: MiniGeom (the
// end-to-end replay against the oracle) and RealGeom (kernels at the real shapes, random data).
#define DS41_EMU 1
#include "../cuda/attn_impl.cuh"
#include "attn_host_impl.hpp"

namespace strata::ds41::cuda {
template struct AttnKernels<RealGeom>;
template struct AttnKernels<MiniGeom>;
template class Ds41Attention<RealGeom>;
template class Ds41Attention<MiniGeom>;
}  // namespace strata::ds41::cuda
