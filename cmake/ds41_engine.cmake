# DS1-E (the engine): the decode session, the two-socket CPU expert pool, the trace writer, the `strata-ds41` command line and its emulated MiniGeom twin, and the
# end-to-end test against the NumPy oracle.  Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake, alphabetically: after ds41_dense, before ds41_mhc / ds41_model;
# every target named below is resolved at generate time, so the order does not matter).
#
#   strata_ds41_session       static library, plain C++: CpuExpertPool (cpu_pool.cpp: one worker group per socket, each socket's half of every missed expert, a partial y per
#                             socket), TraceWriter / LogitsDump (trace.cpp: DS1-F's trace format, golden_compare's logits dump).  Part of `all`.
#   strata_ds41_session_gpu   static library (STRATA_ENABLE_CUDA): the session for RealGeom (session_real.cpp: Ds41Session<RealGeom>) + the MoE sum kernel (moe_combine.cu, sm_70).
#   strata-ds41               the engine for the V100 (RealGeom): src/ds41/program/strata_ds41_main.cpp + cli.hpp.  Links every nvcc library of DS-1; the SASS audit
#                             (tools/volta/sass_audit.py) covers the libraries.  Part of `all` with STRATA_ENABLE_CUDA.
#   strata-ds41-mini-emu      the SAME session and command line instantiated for MiniGeom on the emulated device, with every kernel compiled for the host (-DDS41_EMU: the
#                             thread-model emulation, src/ds41/cuda/ds41_emu.hpp): the mini GGUF of tools/ds41/make_mini_gguf.py runs end to end on any machine.  POSIX only
#                             (ucontext); EXCLUDE_FROM_ALL unless STRATA_BUILD_TESTS, always defined (`cmake --build <dir> --target strata-ds41-mini-emu`).
#   ds41_session_test         unit tests: the CPU pool (bit-identical to DS-C's expert_run on each half + h0 + h1, 1..7 workers, windows, real and mini shapes, the layout for a
#                             faked two-node topology), the MoE sum kernel (the oracle's order, emulated, three scheduling orders), the trace writer.  ctest: ds41_session_*.
#   ds41_e2e_mini_kv_on/off   ctest: tools/ds41/ds1_e2e.py prepares the mini model + the oracle's trace (QuantConfig int8_act [+ the three KV flags]), strata-ds41-mini-emu runs the
#                             same 8 + 56 tokens with --trace, ds1_e2e.py check compares stage by stage (needs python3 + numpy: skipped, with a message, when absent).
#   ds41_e2e_mini_variants    ctest: the engine's own invariants on the mini model - determinism (two runs, identical logits), CPU-only vs GPU-only vs mixed expert placement,
#                             windows of 4 tokens vs single tokens, the CPU pool at several thread counts: src/ds41/program/e2e_mini.py --variants.
set(_ds41e_src ${PROJECT_SOURCE_DIR}/src/ds41/session)
set(_ds41e_prog ${PROJECT_SOURCE_DIR}/src/ds41/program)
set(_ds41e_cuda ${PROJECT_SOURCE_DIR}/src/ds41/cuda)
set(_ds41e_attn ${PROJECT_SOURCE_DIR}/src/ds41/attn)

find_package(Threads REQUIRED)
add_library(strata_ds41_session STATIC ${_ds41e_src}/cpu_pool.cpp ${_ds41e_src}/trace.cpp)
target_include_directories(strata_ds41_session PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(strata_ds41_session PUBLIC strata_ds41_cpu strata_ds41_model Threads::Threads)

# ---- the engine for the V100 ------------------------------------------------------------------------------------------------------------
if(STRATA_ENABLE_CUDA)
  add_library(strata_ds41_session_gpu STATIC ${_ds41e_src}/moe_combine.cu ${_ds41e_src}/session_real.cpp)
  target_include_directories(strata_ds41_session_gpu PUBLIC ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(strata_ds41_session_gpu PRIVATE ${_ds41e_src} ${_ds41e_cuda})
  target_link_libraries(strata_ds41_session_gpu PUBLIC strata_ds41_session strata_ds41_attn strata_ds41_dense strata_ds41_cuda strata_ds41_mhc strata_ds41_engram_host CUDA::cudart)
  set_target_properties(strata_ds41_session_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
  # NOT --use_fast_math (the MoE sum is plain FP32 adds; the kernel pins them with fadd_rn like the rest of DS-1)

  add_executable(strata-ds41 ${_ds41e_prog}/strata_ds41_main.cpp)
  target_include_directories(strata-ds41 PRIVATE ${_ds41e_prog} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_link_libraries(strata-ds41 PRIVATE strata_ds41_session_gpu CUDA::cudart)
endif()

# ---- the emulated MiniGeom engine ---------------------------------------------------------------------------------------------------------
if(NOT WIN32)
  set(_ds41e_all EXCLUDE_FROM_ALL)
  if(STRATA_BUILD_TESTS)
    set(_ds41e_all "")
  endif()
  add_executable(strata-ds41-mini-emu ${_ds41e_all}
                 ${_ds41e_prog}/strata_ds41_mini_emu_main.cpp ${_ds41e_src}/session_mini.cpp ${_ds41e_src}/moe_combine_emu_impl.cpp
                 ${_ds41e_cuda}/ds41_emu_impl.cpp ${_ds41e_cuda}/mhc_emu_impl.cpp ${_ds41e_cuda}/engram_emu_impl.cpp ${_ds41e_attn}/attn_emu_impl.cpp)
  target_include_directories(strata-ds41-mini-emu PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41e_cuda} ${_ds41e_attn} ${_ds41e_src} ${_ds41e_prog})
  target_link_libraries(strata-ds41-mini-emu PRIVATE strata_ds41_session strata_ds41_dense_emu strata_ds41_engram_host)
  if(NOT MSVC)
    # the kernels pin every product and sum (no contraction); the emulator's fibres re-enter functions through getcontext
    target_compile_options(strata-ds41-mini-emu PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
  endif()

  # unit tests of the engine's own parts (the CPU pool against DS-C's reference arrangement, the MoE sum kernel against the oracle's order, the trace writer): no GGUF, no GPU
  add_executable(ds41_session_test ${_ds41e_all} ${_ds41e_src}/session_test.cpp ${_ds41e_src}/moe_combine_emu_impl.cpp)
  target_include_directories(ds41_session_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41e_cuda} ${_ds41e_src})
  target_link_libraries(ds41_session_test PRIVATE strata_ds41_session)
  if(NOT MSVC)
    target_compile_options(ds41_session_test PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
  endif()
  if(STRATA_BUILD_TESTS)
    add_test(NAME ds41_session_pool COMMAND ds41_session_test --pool)
    add_test(NAME ds41_session_combine COMMAND ds41_session_test --combine)
    add_test(NAME ds41_session_combine_reverse COMMAND ds41_session_test --combine --order reverse)
    add_test(NAME ds41_session_trace COMMAND ds41_session_test --trace)
    set_tests_properties(ds41_session_pool ds41_session_combine ds41_session_combine_reverse ds41_session_trace PROPERTIES PASS_REGULAR_EXPRESSION "ALL PASS" FAIL_REGULAR_EXPRESSION "FAIL "
                         TIMEOUT 600)
  endif()

  # Ds41Session<RealGeom> constructed on the REAL file's shapes (DS1-A's sparse fixture: real metadata and tensor table, data = holes) against a ledger device and the emulated kernels: the
  # wiring of every real layer role, Engram table, scratch size and cache size, without weights or a GPU (no token is run).  Linux only.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(ds41_session_real_smoke ${_ds41e_all}
                   ${_ds41e_src}/session_real_smoke.cpp ${_ds41e_src}/moe_combine_emu_impl.cpp
                   ${_ds41e_cuda}/ds41_emu_impl.cpp ${_ds41e_cuda}/mhc_emu_impl.cpp ${_ds41e_cuda}/engram_emu_impl.cpp ${_ds41e_attn}/attn_emu_impl.cpp)
    target_include_directories(ds41_session_real_smoke PRIVATE ${PROJECT_SOURCE_DIR}/include ${_ds41e_cuda} ${_ds41e_attn} ${_ds41e_src})
    target_link_libraries(ds41_session_real_smoke PRIVATE strata_ds41_session strata_ds41_dense_emu strata_ds41_engram_host)
    if(NOT MSVC)
      target_compile_options(ds41_session_real_smoke PRIVATE -fno-strict-aliasing -ffp-contract=off -Wno-clobbered)
    endif()
    if(STRATA_BUILD_TESTS AND TARGET ds41_model_sparse_test)
      # the fixture DS1-A's tests use (cmake/ds41_model.cmake): the sparse real shards are written by its setup test and removed by its cleanup test
      add_test(NAME ds41_session_real_smoke COMMAND ds41_session_real_smoke ${CMAKE_BINARY_DIR}/ds41_model_fixture)
      set_tests_properties(ds41_session_real_smoke PROPERTIES FIXTURES_REQUIRED ds41_model_fixture SKIP_RETURN_CODE 2 TIMEOUT 600 PASS_REGULAR_EXPRESSION "PASS: " FAIL_REGULAR_EXPRESSION "FAIL")
    endif()
  endif()

  if(STRATA_BUILD_TESTS AND EXISTS ${_ds41e_prog}/e2e_mini.py)
    # python3 with numpy (the oracle is NumPy); the driver itself reports "SKIPPED" and exits 2 (ctest: skipped) when numpy is missing
    set(_ds41e_python "")
    set(_ds41e_candidates "")
    if(IQ_FIXTURE_PYTHON)
      list(APPEND _ds41e_candidates ${IQ_FIXTURE_PYTHON})
    else()
      find_package(Python3 COMPONENTS Interpreter QUIET)
      if(Python3_Interpreter_FOUND)
        list(APPEND _ds41e_candidates ${Python3_EXECUTABLE})
      endif()
      list(APPEND _ds41e_candidates "${PROJECT_SOURCE_DIR}/.venv/bin/python" "${PROJECT_SOURCE_DIR}/.venv/Scripts/python.exe" /usr/bin/python3 python3)
    endif()
    foreach(_cand IN LISTS _ds41e_candidates)
      if(_cand STREQUAL "")
        continue()
      endif()
      execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _ds41e_probe)
      if(_ds41e_probe EQUAL 0)
        set(_ds41e_python "${_cand}")
        break()
      endif()
    endforeach()
    if(_ds41e_python STREQUAL "")
      message(STATUS "DS1-E end-to-end tests: no python3 with numpy was found; ds41_e2e_mini_* are registered and will be SKIPPED")
      set(_ds41e_python python3)
    endif()
    set(_ds41e_work ${CMAKE_BINARY_DIR}/ds41_e2e)
    foreach(_kv IN ITEMS on off)
      add_test(NAME ds41_e2e_mini_kv_${_kv} COMMAND ${_ds41e_python} ${_ds41e_prog}/e2e_mini.py --engine $<TARGET_FILE:strata-ds41-mini-emu> --work ${_ds41e_work}/kv_${_kv} --kv ${_kv})
      set_tests_properties(ds41_e2e_mini_kv_${_kv} PROPERTIES SKIP_RETURN_CODE 2 TIMEOUT 3600 FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
    endforeach()
    add_test(NAME ds41_e2e_mini_variants COMMAND ${_ds41e_python} ${_ds41e_prog}/e2e_mini.py --engine $<TARGET_FILE:strata-ds41-mini-emu> --work ${_ds41e_work}/variants --variants)
    set_tests_properties(ds41_e2e_mini_variants PROPERTIES SKIP_RETURN_CODE 2 TIMEOUT 3600 FAIL_REGULAR_EXPRESSION "RESULT: FAIL")
  endif()
endif()
