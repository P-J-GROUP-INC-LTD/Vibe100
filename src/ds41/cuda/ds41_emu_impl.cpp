// src/ds41/cuda/ds41_emu_impl.cpp - DS-D: the DS-D kernels and entry points compiled for the HOST (CPU emulation of the CUDA thread
// model, see ds41_emu.hpp).  Same source as the nvcc build, so the logic that runs here is the logic that runs on the V100.
#define DS41_EMU 1
#include "ds41_router_impl.cuh"
#include "ds41_split_impl.cuh"
#include "ds41_experts_impl.cuh"
