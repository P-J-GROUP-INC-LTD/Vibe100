# DS1-B (dense GPU kernels): Q8_0 / BF16 / F32 GEMV, RMSNorm, RoPE, the shared expert, argmax / top-k for DeepSeek-V4.1-Flash on the V100 (sm_70), and the CPU
# emulation test of the same kernel source.  Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).  API: include/strata/ds41/cuda/dense.hpp.
#
#   strata_ds41_dense        static library, nvcc (STRATA_ENABLE_CUDA): src/ds41/cuda/dense.cu instantiated for RealGeom, built for CMAKE_CUDA_ARCHITECTURES (70);
#                            links strata_ds41_cuda (DS-D / DS1-G: the activation quantiser the shared expert calls).  Part of `all`.
#   strata_ds41_dense_emu    static library, plain C++ (POSIX): the SAME kernel source compiled for the host against the thread-model emulation
#                            (src/ds41/cuda/ds41_emu.hpp), RealGeom and MiniGeom.  What a CPU-only test of code that calls dense.hpp links INSTEAD of
#                            strata_ds41_dense; such a program also compiles src/ds41/cuda/ds41_emu_impl.cpp (DS-D's emulated kernels, which provide
#                            ds41_quantize_acts<G>) into itself, as ds41_cuda_emu_test does.  EXCLUDE_FROM_ALL unless STRATA_BUILD_TESTS.
#   ds41_dense_emu_test      the emulated kernels against C++ FP64 references at both geometries, T-invariance, forward / reverse / shuffled scheduling; registered
#                            with ctest (CPU only).  See the header of src/ds41/cuda/dense_test.cpp for the options.
#   ds41_dense_parity        the same test source against the sm_70 kernels on the V100 (RealGeom) and --bench (GB/s per shape); run by hand, not in ctest.
set(_ds41b_src ${PROJECT_SOURCE_DIR}/src/ds41/cuda)

# every target below is added only when its sources exist, so a configure of the whole tree never fails on a half-written package
if(NOT WIN32 AND EXISTS ${_ds41b_src}/dense_emu_impl.cpp)
  set(_ds41b_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS)
    set(_ds41b_all "")
  endif()
  add_library(strata_ds41_dense_emu STATIC ${_ds41b_all} ${_ds41b_src}/dense_emu_impl.cpp)
  target_include_directories(strata_ds41_dense_emu PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(strata_ds41_dense_emu PRIVATE ${_ds41b_src})
  if(NOT MSVC)
    # the arithmetic is pinned operation by operation (fmul_rn / fadd_rn / fma_rn): no contraction; the emulator's fibres re-enter functions through getcontext
    target_compile_options(strata_ds41_dense_emu PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
  endif()

  if(EXISTS ${_ds41b_src}/dense_test.cpp)
    # DS-D's emulated kernels (ds41_emu_impl.cpp: ds41_quantize_acts<G>, which the shared expert calls) are compiled into the test program itself, as
    # ds41_cuda_emu_test does; strata_ds41_cpu is the CPU quantiser the --quant suite cross-checks (cmake/ds41_cpu.cmake, included first)
    add_executable(ds41_dense_emu_test ${_ds41b_all} ${_ds41b_src}/dense_test.cpp ${_ds41b_src}/ds41_emu_impl.cpp)
    target_include_directories(ds41_dense_emu_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41b_src})
    target_link_libraries(ds41_dense_emu_test PRIVATE strata_ds41_dense_emu strata_ds41_cpu)
    if(NOT MSVC)
      target_compile_options(ds41_dense_emu_test PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
    endif()
  endif()
  if(STRATA_BUILD_TESTS AND EXISTS ${_ds41b_src}/dense_test.cpp)
    # every suite in the forward scheduling order, the cheap ones also reversed and shuffled (the emulator exposes a kernel that is only right in one order)
    foreach(_s IN ITEMS quant gemv wide norm rope shared vocab args)
      add_test(NAME ds41_dense_emu_${_s} COMMAND ds41_dense_emu_test --${_s})
      add_test(NAME ds41_dense_emu_${_s}_reverse COMMAND ds41_dense_emu_test --${_s} --order reverse)
      add_test(NAME ds41_dense_emu_${_s}_shuffle COMMAND ds41_dense_emu_test --${_s} --order shuffle:7)
      set_tests_properties(ds41_dense_emu_${_s} ds41_dense_emu_${_s}_reverse ds41_dense_emu_${_s}_shuffle PROPERTIES PASS_REGULAR_EXPRESSION "ALL PASS"
                           FAIL_REGULAR_EXPRESSION "FAIL ")
    endforeach()
  endif()
endif()

if(STRATA_ENABLE_CUDA AND EXISTS ${_ds41b_src}/dense.cu)
  add_library(strata_ds41_dense STATIC ${_ds41b_src}/dense.cu)
  target_include_directories(strata_ds41_dense PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(strata_ds41_dense PRIVATE ${_ds41b_src})
  # the shared expert calls DS-D's ds41_quantize_acts<G> (strata_ds41_cuda); everything else is in this library
  target_link_libraries(strata_ds41_dense PUBLIC CUDA::cudart strata_ds41_cuda)
  set_target_properties(strata_ds41_dense PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)

  # The V100 program: the same test source against the sm_70 kernels through CudaDev (RealGeom), plus --bench.  Always defined; in `all` only with the tests or
  # STRATA_DS41_CUDA_PARITY (ds41_cuda.cmake's option), `--target ds41_dense_parity` builds it on demand.  Not registered with ctest (a GPU-less machine would fail it).
  #     ds41_dense_parity [--quant --gemv --wide --norm --rope --shared --vocab] [--big] [--bench]       (no flag: the seven suites)
  if(EXISTS ${_ds41b_src}/dense_test.cpp)
    set(_ds41b_parity_all EXCLUDE_FROM_ALL)
    if(STRATA_BUILD_TESTS OR STRATA_DS41_CUDA_PARITY)
      set(_ds41b_parity_all "")
    endif()
    add_executable(ds41_dense_parity ${_ds41b_parity_all} ${_ds41b_src}/dense_test.cpp)
    target_compile_definitions(ds41_dense_parity PRIVATE DS41_DENSE_GPU)
    target_include_directories(ds41_dense_parity PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41b_src} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(ds41_dense_parity PRIVATE strata_ds41_dense strata_ds41_cuda strata_ds41_cpu CUDA::cudart)
    if(NOT MSVC)
      target_compile_options(ds41_dense_parity PRIVATE -ffp-contract=off -fno-strict-aliasing)
    endif()
  endif()
  # NOT --use_fast_math: the norm's divisions / sqrt and the activation quantiser's divisions are the IEEE ones; the kernels pin every product and sum that must not
  # be contracted (fmul_rn / fadd_rn / fma_rn, ds41_dev.cuh / dense_dev.cuh).
endif()
