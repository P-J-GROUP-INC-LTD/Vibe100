// src/ds41/cuda/ds41_emu_impl.cpp - DS-D / DS1-G: the DS-D kernels and entry points compiled for the HOST (CPU emulation of the CUDA thread
// model, see ds41_emu.hpp).  Same source as the nvcc build, so the logic that runs here is the logic that runs on the V100.  The nvcc build
// instantiates RealGeom only; the emulation instantiates both geometries (MiniGeom: the shapes of the mini model, tests at small shapes).
#define DS41_EMU 1
#include "ds41_experts_impl.cuh"
#include "ds41_router_impl.cuh"
#include "ds41_split_impl.cuh"

DS41_DEFINE_EXPERTS_COMMON()
DS41_INSTANTIATE_ROUTER(::strata::ds41::RealGeom)
DS41_INSTANTIATE_SPLIT(::strata::ds41::RealGeom)
DS41_INSTANTIATE_EXPERTS(::strata::ds41::RealGeom)
DS41_INSTANTIATE_ROUTER(::strata::ds41::MiniGeom)
DS41_INSTANTIATE_SPLIT(::strata::ds41::MiniGeom)
DS41_INSTANTIATE_EXPERTS(::strata::ds41::MiniGeom)
