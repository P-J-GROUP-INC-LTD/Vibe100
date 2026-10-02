// src/ds41/session/session_real.cpp - DS1-E: Ds41Session<RealGeom>, plain C++ (linked with the nvcc-built kernels of strata_ds41_cuda / _dense / _attn / _mhc and moe_combine.cu).
#include "session_impl.hpp"

DS41_INSTANTIATE_SESSION(::strata::ds41::RealGeom)
