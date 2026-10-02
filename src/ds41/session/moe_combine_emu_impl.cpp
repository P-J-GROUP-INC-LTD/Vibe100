// src/ds41/session/moe_combine_emu_impl.cpp - DS1-E: the MoE sum compiled for the HOST (CPU emulation of the CUDA thread model, ds41_emu.hpp), RealGeom and MiniGeom.
#define DS41_EMU 1
#include "moe_combine_impl.cuh"

DS41_INSTANTIATE_MOE_COMBINE(::strata::ds41::RealGeom)
DS41_INSTANTIATE_MOE_COMBINE(::strata::ds41::MiniGeom)
