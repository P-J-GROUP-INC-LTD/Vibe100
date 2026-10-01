# DS1-D (mHC and Engram): the hyper-connection kernels (hc_mixes / Sinkhorn / hc_pre / hc_post), the Engram n-gram hasher + row gather (host), the MXFP4 row
# dequantisation and the combine kernel (device), and their emulator tests.  Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).
#
#   strata_ds41_engram_host   static library, plain C++ (no CUDA): NgramHasher, engram_gather_rows.  Part of `all`.
#   strata_ds41_mhc           static library, nvcc (STRATA_ENABLE_CUDA): mhc.cu + engram.cu instantiated for RealGeom (sm_70 for the V100).  Part of `all`.
#   ds41_mhc_emu_test         the SAME kernel sources (src/ds41/cuda/mhc_impl.cuh, engram_impl.cuh) compiled for the host with -DDS41_EMU and run on the CPU through
#   ds41_engram_emu_test      the thread-model emulation (ds41_emu.hpp), at MiniGeom and RealGeom shapes, in forward / reverse / shuffled order, against FP64
#                             references and the NumPy oracle's golden data (src/ds41/engram/golden/gen_golden.py).  POSIX only (ucontext).
set(_ds41m_cuda ${PROJECT_SOURCE_DIR}/src/ds41/cuda)
set(_ds41m_host ${PROJECT_SOURCE_DIR}/src/ds41/engram)

add_library(strata_ds41_engram_host STATIC ${_ds41m_host}/engram_hasher.cpp)
target_include_directories(strata_ds41_engram_host PUBLIC ${PROJECT_SOURCE_DIR}/include)

if(STRATA_ENABLE_CUDA)
  add_library(strata_ds41_mhc STATIC ${_ds41m_cuda}/mhc.cu ${_ds41m_cuda}/engram.cu)
  target_include_directories(strata_ds41_mhc PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(strata_ds41_mhc PRIVATE ${_ds41m_cuda})
  target_link_libraries(strata_ds41_mhc PUBLIC CUDA::cudart strata_ds41_engram_host)
  set_target_properties(strata_ds41_mhc PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
  # NOT --use_fast_math: the sigmoid / softmax / Sinkhorn divisions and the rsqrt of hc_mixes and Engram are the IEEE ones; the kernels pin every
  # product and sum that must not be contracted (fmul_rn / fadd_rn, ds41_dev.cuh).
endif()
