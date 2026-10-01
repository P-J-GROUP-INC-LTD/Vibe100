// src/ds41/cuda/dense_emu_impl.cpp - DS1-B: the dense kernels and entry points compiled for the HOST (CPU emulation of the CUDA thread model, ds41_emu.hpp).  The
// same source as the nvcc build (dense_impl.cuh), instantiated for RealGeom and MiniGeom.  Built into strata_ds41_dense_emu, which a program that tests code
// calling dense.hpp links INSTEAD of strata_ds41_dense.
#define DS41_EMU 1
#include "dense_impl.cuh"

DS41_INSTANTIATE_DENSE(::strata::ds41::RealGeom)
DS41_INSTANTIATE_DENSE(::strata::ds41::MiniGeom)
