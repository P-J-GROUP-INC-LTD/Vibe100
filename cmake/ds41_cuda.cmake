# DS-D (GPU hot experts + router): the sm_70 (V100) kernels for DeepSeek-V4.1-Flash and their parity programs.
# Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).
#
#   strata_ds41_cuda            static library: router (logits GEMV / token-looping prefill kernel / top-6), hit-miss split, MXFP4 hit experts
#                               (activation quantiser, gate/up, down).  Needs STRATA_ENABLE_CUDA; built for CMAKE_CUDA_ARCHITECTURES
#                               (70 for the V100; CUDA 13 cannot build sm_70, see the top-level message).  Part of `all`.
#   ds41_router_parity          GPU programs, run on the V100 (synthetic data, no model); `--selftest` exits non-zero on a failed
#   ds41_split_parity           check, prints PASS/FAIL lines.  Not registered with ctest (a GPU-less machine would fail them).
#   ds41_expert_parity          See docs in the sources' headers for the exact commands and what PASS looks like.
#   ds41_cuda_emu_test          the SAME kernel sources compiled for the host and run on the CPU through a thread-model emulation
#                               (src/ds41/cuda/ds41_emu.hpp): checks the kernels' logic against the FP64 references with no GPU, in any
#                               fiber / block scheduling order (--order forward|reverse|shuffle).  Registered with ctest (CPU only; the
#                               expert part takes about a minute).  POSIX only (ucontext).  Links the CPU library for the quantiser cross-check.
#
# BUILD HYGIENE (audit A3-8).  The parity programs and the emulator test are test programs: like the rest of the project's tests they are part
# of `all` only when the build asks for tests (STRATA_BUILD_TESTS=ON, or STRATA_DS41_CUDA_PARITY=ON for the three GPU programs alone) and are
# EXCLUDE_FROM_ALL otherwise - but they are always DEFINED, so that
#     cmake --build <dir> --target ds41_router_parity ds41_split_parity ds41_expert_parity      (on the box with the V100)
#     cmake --build <dir> --target ds41_cuda_emu_test                                           (anywhere)
# work on any configuration.  The libraries stay in `all`.
set(_ds41d_src ${PROJECT_SOURCE_DIR}/src/ds41/cuda)

# ---- the CPU emulation of the kernels (no CUDA toolchain needed; defined on every POSIX configuration, built by `all` only with the tests) --------
if(NOT WIN32)
  set(_ds41_emu_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS)
    set(_ds41_emu_all "")
  endif()
  add_executable(ds41_cuda_emu_test ${_ds41_emu_all} ${_ds41d_src}/ds41_emu_test.cpp ${_ds41d_src}/ds41_emu_impl.cpp)
  target_include_directories(ds41_cuda_emu_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41d_src})
  target_link_libraries(ds41_cuda_emu_test PRIVATE strata_ds41_cpu)      # the --quant cross-check runs the CPU quantiser (cmake/ds41_cpu.cmake, included first)
  if(NOT MSVC)
    # the references and the quantiser rely on separately rounded FP32 operations; the fibers re-enter functions through getcontext
    target_compile_options(ds41_cuda_emu_test PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
  endif()
  if(STRATA_BUILD_TESTS)
    # every suite in the forward scheduling order, the router / split / quantiser also reversed and shuffled, the experts reversed (the
    # experts are the slow ones: ~1 minute each); see ds41_emu.hpp for what the order exposes
    add_test(NAME ds41_cuda_emu_router COMMAND ds41_cuda_emu_test --router)
    add_test(NAME ds41_cuda_emu_split COMMAND ds41_cuda_emu_test --split)
    add_test(NAME ds41_cuda_emu_experts COMMAND ds41_cuda_emu_test --experts)
    add_test(NAME ds41_cuda_emu_quant COMMAND ds41_cuda_emu_test --quant)
    add_test(NAME ds41_cuda_emu_router_reverse COMMAND ds41_cuda_emu_test --router --order reverse)
    add_test(NAME ds41_cuda_emu_split_reverse COMMAND ds41_cuda_emu_test --split --order reverse)
    add_test(NAME ds41_cuda_emu_experts_reverse COMMAND ds41_cuda_emu_test --experts --order reverse)
    add_test(NAME ds41_cuda_emu_quant_reverse COMMAND ds41_cuda_emu_test --quant --order reverse)
    add_test(NAME ds41_cuda_emu_router_shuffle COMMAND ds41_cuda_emu_test --router --order shuffle:1)
    add_test(NAME ds41_cuda_emu_split_shuffle COMMAND ds41_cuda_emu_test --split --order shuffle:2)
    add_test(NAME ds41_cuda_emu_quant_shuffle COMMAND ds41_cuda_emu_test --quant --order shuffle:3)
    # the emulator's own promises (exited lanes in votes, shared-memory poison, the order itself), in each order
    add_test(NAME ds41_cuda_emu_selftest COMMAND ds41_cuda_emu_test --emu)
    add_test(NAME ds41_cuda_emu_selftest_reverse COMMAND ds41_cuda_emu_test --emu --order reverse)
    add_test(NAME ds41_cuda_emu_selftest_shuffle COMMAND ds41_cuda_emu_test --emu --order shuffle:4)
    set(_ds41_emu_tests ds41_cuda_emu_router ds41_cuda_emu_split ds41_cuda_emu_experts ds41_cuda_emu_quant ds41_cuda_emu_router_reverse
        ds41_cuda_emu_split_reverse ds41_cuda_emu_experts_reverse ds41_cuda_emu_quant_reverse ds41_cuda_emu_router_shuffle
        ds41_cuda_emu_split_shuffle ds41_cuda_emu_quant_shuffle ds41_cuda_emu_selftest ds41_cuda_emu_selftest_reverse
        ds41_cuda_emu_selftest_shuffle)
    set_tests_properties(${_ds41_emu_tests} PROPERTIES PASS_REGULAR_EXPRESSION "ALL PASS" FAIL_REGULAR_EXPRESSION "FAIL ")
    set_tests_properties(ds41_cuda_emu_experts ds41_cuda_emu_experts_reverse PROPERTIES TIMEOUT 1800)
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

  # The three GPU programs: always defined, in `all` only when asked (tests, or this option); `--target` builds them on demand.
  option(STRATA_DS41_CUDA_PARITY "Put the DS-D GPU parity programs (ds41_router_parity, ds41_split_parity, ds41_expert_parity) in 'all' (they can always be built by name)" OFF)
  set(_ds41_parity_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS OR STRATA_DS41_CUDA_PARITY)
    set(_ds41_parity_all "")
  endif()
  foreach(_p IN ITEMS router split expert)
    add_executable(ds41_${_p}_parity ${_ds41_parity_all} ${_ds41d_src}/ds41_${_p}_parity.cpp)
    target_include_directories(ds41_${_p}_parity PRIVATE ${_ds41d_src} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(ds41_${_p}_parity PRIVATE strata_ds41_cuda CUDA::cudart)
    if(NOT MSVC)
      target_compile_options(ds41_${_p}_parity PRIVATE -ffp-contract=off)
    endif()
  endforeach()
endif()
