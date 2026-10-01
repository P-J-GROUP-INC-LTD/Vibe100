// src/ds41/cuda/mhc.cu - DS1-D: mHC kernels for the V100 (nvcc, sm_70), instantiated for the real geometry.  Kernels and host wrappers are in
// mhc_impl.cuh (shared with the CPU emulation test); the API is include/strata/ds41/cuda/mhc.hpp.
#include "mhc_impl.cuh"

namespace strata::ds41::cuda {

template void ds41_hc_mixes<RealGeom>(Dev&, const float*, const HcWeights&, int, const HcParams&, float*, void*, size_t, float*, Stream);
template void ds41_hc_split<RealGeom>(Dev&, const float*, const float*, const float*, int, const HcParams&, float*, Stream);
template void ds41_hc_pre<RealGeom>(Dev&, const float*, const float*, int, float*, Stream);
template void ds41_hc_post<RealGeom>(Dev&, const float*, const float*, const float*, int, float*, Stream);
template void ds41_hc_set_identity_pre<RealGeom>(Dev&, float*, int, Stream);
template void ds41_hc_expand<RealGeom>(Dev&, const float*, int, float*, Stream);
template HcKernelInfo ds41_hc_kernel_info<RealGeom>(int);

}  // namespace strata::ds41::cuda
