// src/ds41/cuda/ds41_experts.cu - DS-D / DS1-G: the MXFP4 hit-expert kernels, the activation quantiser and their host entry points.  All of it is
// templates over the geometry G in ds41_experts_impl.cuh (shared with the CPU emulation, which also instantiates MiniGeom); this library instantiates
// RealGeom, the engine's geometry.  `test_e8m0_table` (geometry independent) is defined there too.
#include "ds41_experts_impl.cuh"

DS41_INSTANTIATE_EXPERTS(::strata::ds41::RealGeom)
