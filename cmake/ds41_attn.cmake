# DS1-C (attention, CSA2): the MQA attention of DeepSeek-V4.1-Flash with the compressed sparse attention: q path, SWA ring, compressor, indexer, candidate pool,
# sparse attention with the sink, grouped output projection.  Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).
#
#   strata_ds41_attn       static library, nvcc (STRATA_ENABLE_CUDA): attn_real.cu (the kernels, src/ds41/cuda/attn_impl.cuh, for RealGeom, sm_70 for the V100)
#                          + attn_real.cpp (the per-layer driver Ds41Attention<RealGeom>, src/ds41/attn/attn_host_impl.hpp).  Part of `all`.
#   ds41_attn_emu_test     the SAME kernel and driver sources compiled for the host with -DDS41_EMU and run on the CPU through the thread-model emulation
#                          (src/ds41/cuda/ds41_emu.hpp): kernel tests at MiniGeom and RealGeom shapes against FP64 / exact references, and the end-to-end replay
#                          of the NumPy oracle's attention_layer (ref/ds41/attention.py) at MiniGeom over every layer role, KV quantisation on and off.  The
#                          oracle's inputs and outputs come from src/ds41/attn/golden/gen_golden.py (system python3 + numpy), generated at test time into the build
#                          directory (nothing is checked in).  POSIX only (ucontext).
set(_ds41a_cuda ${PROJECT_SOURCE_DIR}/src/ds41/cuda)
set(_ds41a_host ${PROJECT_SOURCE_DIR}/src/ds41/attn)

if(NOT WIN32)
  set(_ds41a_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS)
    set(_ds41a_all "")
  endif()
  set(_ds41a_test_srcs ${_ds41a_host}/attn_emu_test.cpp ${_ds41a_host}/attn_emu_impl.cpp)
  # the integration check (--dense real) runs the replay on DS1-B's emulated dense kernels and DS1-G's quantiser: as DS1-B's own test does, the program links
  # strata_ds41_dense_emu and compiles DS-D's emulated kernels (ds41_emu_impl.cpp: ds41_quantize_acts<G>) into itself; absent them, only --dense ref exists
  set(_ds41a_real_dense OFF)
  if(EXISTS ${_ds41a_cuda}/dense_emu_impl.cpp AND EXISTS ${_ds41a_cuda}/ds41_emu_impl.cpp)
    set(_ds41a_real_dense ON)
    list(APPEND _ds41a_test_srcs ${_ds41a_cuda}/ds41_emu_impl.cpp)
  endif()
  add_executable(ds41_attn_emu_test ${_ds41a_all} ${_ds41a_test_srcs})
  target_include_directories(ds41_attn_emu_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41a_cuda} ${_ds41a_host})
  if(_ds41a_real_dense)
    target_compile_definitions(ds41_attn_emu_test PRIVATE ATTN_TEST_REAL_DENSE=1)
    target_link_libraries(ds41_attn_emu_test PRIVATE strata_ds41_dense_emu)
  endif()
  if(NOT MSVC)
    # the references pin separately rounded FP32 operations; the fibers re-enter functions through getcontext
    target_compile_options(ds41_attn_emu_test PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
  endif()

  # the oracle's golden data: python3 + numpy are needed; without them the golden-driven tests are not registered (the kernel tests still are)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  set(_ds41a_have_numpy OFF)
  if(Python3_Interpreter_FOUND)
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy" RESULT_VARIABLE _ds41a_np_rc OUTPUT_QUIET ERROR_QUIET)
    if(_ds41a_np_rc EQUAL 0)
      set(_ds41a_have_numpy ON)
    endif()
  endif()
  if(STRATA_BUILD_TESTS)
    add_test(NAME ds41_attn_emu_kernels COMMAND ds41_attn_emu_test --kernels)
    add_test(NAME ds41_attn_emu_kernels_reverse COMMAND ds41_attn_emu_test --kernels --order reverse)
    add_test(NAME ds41_attn_emu_kernels_shuffle COMMAND ds41_attn_emu_test --kernels --order shuffle:5)
    set(_ds41a_tests ds41_attn_emu_kernels ds41_attn_emu_kernels_reverse ds41_attn_emu_kernels_shuffle)
    # the driver at the real shapes with random weights (plumbing; no oracle, no python)
    add_test(NAME ds41_attn_emu_real_smoke COMMAND ds41_attn_emu_test --real-smoke 24)
    set_tests_properties(ds41_attn_emu_real_smoke PROPERTIES TIMEOUT 1800)
    list(APPEND _ds41a_tests ds41_attn_emu_real_smoke)
    if(_ds41a_have_numpy)
      set(_ds41a_golden ${CMAKE_BINARY_DIR}/ds41_attn_golden)
      add_test(NAME ds41_attn_golden_gen COMMAND ${Python3_EXECUTABLE} ${_ds41a_host}/golden/gen_golden.py --out ${_ds41a_golden})
      set_tests_properties(ds41_attn_golden_gen PROPERTIES FIXTURES_SETUP ds41_attn_golden TIMEOUT 1800)
      # all six scenarios + the bit-identity extras (windows T = 1..8, reset(), two other scheduling orders) in the forward order; two scenarios in the other orders
      add_test(NAME ds41_attn_emu_oracle_forward COMMAND ds41_attn_emu_test --oracle ${_ds41a_golden})
      add_test(NAME ds41_attn_emu_oracle_reverse COMMAND ds41_attn_emu_test --oracle ${_ds41a_golden} --scenario decode_all_on --scenario prefill_all_off --no-extras --order reverse)
      add_test(NAME ds41_attn_emu_oracle_shuffle COMMAND ds41_attn_emu_test --oracle ${_ds41a_golden} --scenario decode_ckv_idx --scenario prefill_all_on --no-extras --order shuffle:6)
      foreach(_t IN ITEMS forward reverse shuffle)
        set_tests_properties(ds41_attn_emu_oracle_${_t} PROPERTIES FIXTURES_REQUIRED ds41_attn_golden TIMEOUT 1800)
        list(APPEND _ds41a_tests ds41_attn_emu_oracle_${_t})
      endforeach()
      if(_ds41a_real_dense)
        add_test(NAME ds41_attn_emu_oracle_dense_real COMMAND ds41_attn_emu_test --oracle ${_ds41a_golden} --scenario decode_all_on --scenario prefill_all_on --dense real)
        set_tests_properties(ds41_attn_emu_oracle_dense_real PROPERTIES FIXTURES_REQUIRED ds41_attn_golden TIMEOUT 1800)
        list(APPEND _ds41a_tests ds41_attn_emu_oracle_dense_real)
      endif()
    endif()
    set_tests_properties(${_ds41a_tests} PROPERTIES PASS_REGULAR_EXPRESSION "ALL PASS" FAIL_REGULAR_EXPRESSION "FAIL ")
  endif()
endif()

if(STRATA_ENABLE_CUDA)
  add_library(strata_ds41_attn STATIC ${_ds41a_cuda}/attn_real.cu ${_ds41a_host}/attn_real.cpp)
  target_include_directories(strata_ds41_attn PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(strata_ds41_attn PRIVATE ${_ds41a_cuda} ${_ds41a_host})
  target_link_libraries(strata_ds41_attn PUBLIC CUDA::cudart)
  set_target_properties(strata_ds41_attn PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
  # NOT --use_fast_math: the softmax / RMSNorm divisions and exp are the IEEE ones; the kernels pin every product and sum that must not be contracted
  # (fmul_rn / fadd_rn, ds41_dev.cuh) and use fmaf only where the dot products are meant to fuse.
endif()
