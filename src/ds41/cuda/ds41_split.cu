// src/ds41/cuda/ds41_split.cu - DS-D / DS1-G: the hit / miss split.  The kernel and the host entry point are templates over the geometry G in
// ds41_split_impl.cuh (shared with the CPU emulation, which also instantiates MiniGeom); this library instantiates RealGeom, the engine's geometry.
#include "ds41_split_impl.cuh"

DS41_INSTANTIATE_SPLIT(::strata::ds41::RealGeom)
