// src/ds41/attn/attn_real.cpp - DS1-C: the host driver `Ds41Attention<RealGeom>` (plain C++, linked with the nvcc-built kernels of src/ds41/cuda/attn_real.cu).
#include "attn_host_impl.hpp"

namespace strata::ds41::cuda {
template class Ds41Attention<RealGeom>;
}  // namespace strata::ds41::cuda
