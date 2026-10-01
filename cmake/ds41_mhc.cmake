# DS1-D (mHC and Engram): the hyper-connection kernels (hc_mixes / Sinkhorn / hc_pre / hc_post), the Engram n-gram hasher + row gather (host), the MXFP4 row
# dequantisation and the combine kernel (device), and their emulator tests.  Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).
#
#   strata_ds41_engram_host   static library, plain C++ (no CUDA): NgramHasher, engram_gather_rows.  Part of `all`.
#   strata_ds41_mhc           static library, nvcc (STRATA_ENABLE_CUDA): mhc.cu + engram.cu instantiated for RealGeom (sm_70 for the V100).  Part of `all`.
#   ds41_mhc_emu_test         the SAME kernel sources (src/ds41/cuda/mhc_impl.cuh, engram_impl.cuh) compiled for the host with -DDS41_EMU (mhc_emu_impl.cpp,
#   ds41_engram_emu_test      engram_emu_impl.cpp) and run on the CPU through the thread-model emulation (ds41_emu.hpp), at MiniGeom and RealGeom shapes, in
#                             forward / reverse / shuffled scheduling order, against the NumPy oracle's golden data.  POSIX only (ucontext).
#   ds41_mhc_gpu_test         the SAME test sources (mhc_test.cpp, engram_test.cpp, -DDS1D_ON_GPU) against the kernels nvcc built for sm_70, on the V100, RealGeom only, plus
#   ds41_engram_gpu_test      a `perf` suite (time and effective bandwidth of every op: information).  Needs the card, so NOT registered with ctest; defined when CUDA is
#                             enabled, in `all` with STRATA_BUILD_TESTS or STRATA_DS41_CUDA_PARITY.  Run: <prog> --golden <build>/ds1d_golden (after the ds1d_golden fixture ran:
#                             ctest -R ds1d_golden, or python3 src/ds41/engram/golden/gen_golden.py --out DIR).
#   ds1d_golden               a CTest FIXTURE: src/ds41/engram/golden/gen_golden.py runs the oracle (ref/ds41) and the mini-GGUF generator and writes the golden .npy
#                             files into <build>/ds1d_golden (needs python3 with numpy; a failure fails every test that requires it, none is skipped).
#
# BUILD HYGIENE (as cmake/ds41_cuda.cmake): the two test programs are always DEFINED (cmake --build <dir> --target ds41_mhc_emu_test ds41_engram_emu_test) and in
# `all` only with STRATA_BUILD_TESTS=ON; the libraries stay in `all`.
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

# ---- the V100 programs: the same test sources against the sm_70 kernels (RealGeom) ----------------------------------------------------
if(STRATA_ENABLE_CUDA)
  set(_ds1d_gpu_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS OR STRATA_DS41_CUDA_PARITY)
    set(_ds1d_gpu_all "")
  endif()
  add_executable(ds41_mhc_gpu_test ${_ds1d_gpu_all} ${_ds41m_cuda}/mhc_test.cpp)
  add_executable(ds41_engram_gpu_test ${_ds1d_gpu_all} ${_ds41m_cuda}/engram_test.cpp)
  foreach(_t ds41_mhc_gpu_test ds41_engram_gpu_test)
    target_compile_definitions(${_t} PRIVATE DS1D_ON_GPU=1)
    target_include_directories(${_t} PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41m_cuda} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(${_t} PRIVATE strata_ds41_mhc CUDA::cudart)
    if(NOT MSVC)
      target_compile_options(${_t} PRIVATE -ffp-contract=off)
    endif()
  endforeach()
endif()

# ---- the emulator tests (no CUDA toolchain needed) ----------------------------------------------------------------------------------
if(NOT WIN32)
  set(_ds1d_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS)
    set(_ds1d_all "")
  endif()
  add_executable(ds41_mhc_emu_test ${_ds1d_all} ${_ds41m_cuda}/mhc_test.cpp ${_ds41m_cuda}/mhc_emu_impl.cpp)
  # + DS1-G's emulator build of the shared helpers (ds41_emu_impl.cpp: ds41_quantize_acts<MiniGeom> in front of the wkv hook)
  add_executable(ds41_engram_emu_test ${_ds1d_all} ${_ds41m_cuda}/engram_test.cpp ${_ds41m_cuda}/engram_emu_impl.cpp ${_ds41m_cuda}/ds41_emu_impl.cpp)
  foreach(_t ds41_mhc_emu_test ds41_engram_emu_test)
    target_include_directories(${_t} PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41m_cuda})
    if(NOT MSVC)
      # the references pin every product and sum separately (no contraction); the fibers re-enter functions through getcontext
      target_compile_options(${_t} PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
    endif()
  endforeach()
  target_link_libraries(ds41_engram_emu_test PRIVATE strata_ds41_engram_host)

  if(STRATA_BUILD_TESTS)
    # the golden data: python3 with numpy (the oracle is NumPy); same probing as the i-quant fixtures in the top-level file
    set(_ds1d_python "")
    set(_ds1d_candidates "")
    if(IQ_FIXTURE_PYTHON)
      list(APPEND _ds1d_candidates ${IQ_FIXTURE_PYTHON})
    else()
      find_package(Python3 COMPONENTS Interpreter QUIET)
      if(Python3_Interpreter_FOUND)
        list(APPEND _ds1d_candidates ${Python3_EXECUTABLE})
      endif()
      list(APPEND _ds1d_candidates "${PROJECT_SOURCE_DIR}/.venv/bin/python" "${PROJECT_SOURCE_DIR}/.venv/Scripts/python.exe" /usr/bin/python3 python3)
    endif()
    foreach(_cand IN LISTS _ds1d_candidates)
      if(_cand STREQUAL "")
        continue()
      endif()
      execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _ds1d_probe)
      if(_ds1d_probe EQUAL 0)
        set(_ds1d_python "${_cand}")
        break()
      endif()
    endforeach()
    if(_ds1d_python STREQUAL "")
      message(STATUS "DS1-D tests: no python3 with numpy was found; the golden fixture is registered with plain python3 and will FAIL (so will the tests that need it)")
      set(_ds1d_python python3)
    endif()
    set(_ds1d_golden ${CMAKE_BINARY_DIR}/ds1d_golden)
    add_test(NAME ds1d_golden COMMAND ${_ds1d_python} ${_ds41m_host}/golden/gen_golden.py --out ${_ds1d_golden})
    set_tests_properties(ds1d_golden PROPERTIES FIXTURES_SETUP ds1d_golden TIMEOUT 900)

    set(_ds1d_orders forward reverse shuffle:11)
    set(_ds1d_names "")
    foreach(_o IN LISTS _ds1d_orders)
      string(REPLACE ":" "_" _on "${_o}")
      add_test(NAME ds41_mhc_emu_${_on} COMMAND ds41_mhc_emu_test --golden ${_ds1d_golden} --order ${_o})
      add_test(NAME ds41_engram_emu_${_on} COMMAND ds41_engram_emu_test --golden ${_ds1d_golden} --order ${_o})
      list(APPEND _ds1d_names ds41_mhc_emu_${_on} ds41_engram_emu_${_on})
    endforeach()
    set_tests_properties(${_ds1d_names} PROPERTIES FIXTURES_REQUIRED ds1d_golden PASS_REGULAR_EXPRESSION "ALL PASS" FAIL_REGULAR_EXPRESSION "FAIL " TIMEOUT 1800)
  endif()
endif()
