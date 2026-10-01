// src/ds41/cuda/engram_emu_impl.cpp - DS1-D: the Engram kernels and entry points compiled for the HOST (CPU emulation of the CUDA thread model,
// ds41_emu.hpp).  The same source (engram_impl.cuh) the nvcc build compiles for sm_70; the nvcc build instantiates RealGeom only, the emulation both.
#define DS41_EMU 1
#include "engram_impl.cuh"

#define DS1D_INSTANTIATE_ENGRAM(G)                                                                                                                   \
    namespace strata::ds41::cuda {                                                                                                                    \
    template void ds41_engram_dequant_rows<G>(Dev&, const uint8_t*, int, float*, Stream);                                                             \
    template void ds41_engram_combine<G>(Dev&, float*, const float*, const uint16_t*, const uint16_t*, int, float, Stream);                           \
    template EngramKernelInfo ds41_engram_kernel_info<G>(int);                                                                                        \
    }

DS1D_INSTANTIATE_ENGRAM(::strata::ds41::RealGeom)
DS1D_INSTANTIATE_ENGRAM(::strata::ds41::MiniGeom)
