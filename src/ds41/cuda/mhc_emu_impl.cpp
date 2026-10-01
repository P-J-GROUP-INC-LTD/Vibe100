// src/ds41/cuda/mhc_emu_impl.cpp - DS1-D: the mHC kernels and entry points compiled for the HOST (CPU emulation of the CUDA thread model, ds41_emu.hpp).
// The same source (mhc_impl.cuh) the nvcc build compiles for sm_70; the nvcc build instantiates RealGeom only, the emulation both geometries.
#define DS41_EMU 1
#include "mhc_impl.cuh"

#define DS1D_INSTANTIATE_MHC(G)                                                                                                                      \
    namespace strata::ds41::cuda {                                                                                                                    \
    template void ds41_hc_mixes<G>(Dev&, const float*, const HcWeights&, int, const HcParams&, float*, void*, size_t, float*, Stream);                 \
    template void ds41_hc_split<G>(Dev&, const float*, const float*, const float*, int, const HcParams&, float*, Stream);                            \
    template void ds41_hc_pre<G>(Dev&, const float*, const float*, int, float*, Stream);                                                              \
    template void ds41_hc_post<G>(Dev&, const float*, const float*, const float*, int, float*, Stream);                                               \
    template void ds41_hc_set_identity_pre<G>(Dev&, float*, int, Stream);                                                                             \
    template void ds41_hc_expand<G>(Dev&, const float*, int, float*, Stream);                                                                         \
    template HcKernelInfo ds41_hc_kernel_info<G>(int);                                                                                                \
    }

DS1D_INSTANTIATE_MHC(::strata::ds41::RealGeom)
DS1D_INSTANTIATE_MHC(::strata::ds41::MiniGeom)
