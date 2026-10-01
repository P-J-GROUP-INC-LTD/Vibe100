// src/ds41/cuda/dense.cu - DS1-B: the dense kernels (Q8_0 / BF16 / F32 GEMV, RMSNorm, RoPE, shared expert, argmax / top-k) built by nvcc for sm_70, for the real
// model's geometry.  Kernels and host wrappers are in dense_impl.cuh (shared with the CPU emulation test, which also instantiates MiniGeom); the API is
// include/strata/ds41/cuda/dense.hpp.
#include "dense_impl.cuh"

DS41_INSTANTIATE_DENSE(::strata::ds41::RealGeom)
