// src/ds41/cuda/engram.cu - DS1-D: Engram dequantisation and combine kernels for the V100 (nvcc, sm_70), instantiated for the real geometry.  Kernels and
// host wrappers are in engram_impl.cuh (shared with the CPU emulation test); the API is include/strata/ds41/cuda/engram.hpp.
#include "engram_impl.cuh"

namespace strata::ds41::cuda {

template void ds41_engram_dequant_rows<RealGeom>(Dev&, const uint8_t*, int, float*, Stream);
template void ds41_engram_combine<RealGeom>(Dev&, float*, const float*, const uint16_t*, const uint16_t*, int, float, Stream);
template EngramKernelInfo ds41_engram_kernel_info<RealGeom>(int);

}  // namespace strata::ds41::cuda
