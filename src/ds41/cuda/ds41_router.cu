// src/ds41/cuda/ds41_router.cu - DS-D / DS1-G: the sqrt-softplus router.  The kernels and the host entry points are templates over the geometry G in
// ds41_router_impl.cuh (shared with the CPU emulation, which also instantiates MiniGeom); this library instantiates RealGeom, the engine's geometry.
#include "ds41_router_impl.cuh"

DS41_INSTANTIATE_ROUTER(::strata::ds41::RealGeom)
