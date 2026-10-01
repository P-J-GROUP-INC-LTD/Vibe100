# DS-D (GPU hot experts + router): the sm_70 (V100) kernels for DeepSeek-V4.1-Flash and their parity programs.
# Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).
#
#   strata_ds41_cuda            static library: router (logits GEMV / tiled GEMM / top-6), hit-miss split, MXFP4 hit experts
#                               (activation quantiser, gate/up, down).  Needs STRATA_ENABLE_CUDA; built for CMAKE_CUDA_ARCHITECTURES
#                               (70 for the V100; CUDA 13 cannot build sm_70, see the top-level message).
#   ds41_router_parity          GPU programs, run on the V100 (synthetic data, no model); `--selftest` exits non-zero on a failed
#   ds41_split_parity           check, prints PASS/FAIL lines.  Not registered with ctest (a GPU-less machine would fail them).
#   ds41_expert_parity          See docs in the sources' headers for the exact commands and what PASS looks like.
#   ds41_cuda_emu_test          the SAME kernel sources compiled for the host and run on the CPU through a thread-model emulation
#                               (src/ds41/cuda/ds41_emu.hpp): checks the kernels' logic against the FP64 references with no GPU.
#                               Registered with ctest (CPU only; the expert part takes about a minute).  POSIX only (ucontext).
set(_ds41d_src ${PROJECT_SOURCE_DIR}/src/ds41/cuda)

# ---- the CPU emulation of the kernels (no CUDA toolchain needed) -------------------------------------------------------------------
if(NOT WIN32 AND (STRATA_BUILD_TESTS OR STRATA_ENABLE_CUDA))
  add_executable(ds41_cuda_emu_test ${_ds41d_src}/ds41_emu_test.cpp ${_ds41d_src}/ds41_emu_impl.cpp)
  target_include_directories(ds41_cuda_emu_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41d_src})
  if(NOT MSVC)
    # the references and the quantiser rely on separately rounded FP32 operations; the fibers re-enter functions through getcontext
    target_compile_options(ds41_cuda_emu_test PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
  endif()
  if(STRATA_BUILD_TESTS)
    add_test(NAME ds41_cuda_emu_router COMMAND ds41_cuda_emu_test --router)
    add_test(NAME ds41_cuda_emu_split COMMAND ds41_cuda_emu_test --split)
    add_test(NAME ds41_cuda_emu_experts COMMAND ds41_cuda_emu_test --experts)
    set_tests_properties(ds41_cuda_emu_router ds41_cuda_emu_split ds41_cuda_emu_experts PROPERTIES PASS_REGULAR_EXPRESSION "ALL PASS"
                         FAIL_REGULAR_EXPRESSION "FAIL ")
    set_tests_properties(ds41_cuda_emu_experts PROPERTIES TIMEOUT 1800)
  endif()
endif()

# ---- the sm_70 kernels and the V100 parity programs ---------------------------------------------------------------------------------
if(STRATA_ENABLE_CUDA)
  add_library(strata_ds41_cuda STATIC ${_ds41d_src}/ds41_router.cu ${_ds41d_src}/ds41_split.cu ${_ds41d_src}/ds41_experts.cu)
  target_include_directories(strata_ds41_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(strata_ds41_cuda PRIVATE ${_ds41d_src})
  target_link_libraries(strata_ds41_cuda PUBLIC CUDA::cudart)
  set_target_properties(strata_ds41_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
  # NOT --use_fast_math: the router's softplus / sqrt and the activation quantiser's divisions must be the IEEE ones (the quantiser is
  # bit-identical to the CPU's); the kernels pin the few operations that need it (ds41_dev.cuh) and never rely on contraction.

  option(STRATA_DS41_CUDA_PARITY "Build the DS-D GPU parity programs (ds41_router_parity, ds41_split_parity, ds41_expert_parity)" ON)
  if(STRATA_DS41_CUDA_PARITY)
    foreach(_p IN ITEMS router split expert)
      add_executable(ds41_${_p}_parity ${_ds41d_src}/ds41_${_p}_parity.cpp)
      target_include_directories(ds41_${_p}_parity PRIVATE ${_ds41d_src} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      target_link_libraries(ds41_${_p}_parity PRIVATE strata_ds41_cuda CUDA::cudart)
      if(NOT MSVC)
        target_compile_options(ds41_${_p}_parity PRIVATE -ffp-contract=off)
      endif()
    endforeach()
  endif()
endif()
