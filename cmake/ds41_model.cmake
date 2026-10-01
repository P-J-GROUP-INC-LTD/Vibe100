# DS1-A (loader and model state): the multi-shard GGUF reader, Ds41Config (every `deepseek41.*` key + the layer roles), the tensor table and its
# validation, the weights (dense tensors to the device), the CPU expert arena (per-socket halves), the GPU expert cache and the memory plan.
# Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).  Plain C++: no CUDA, no GPU; the device is reached through DS1-G's `Dev`
# interface (include/strata/ds41/cuda/ds41_dev.hpp), whose `HostDev` runs everything here in the emulator build and the tests.
#
#   strata_ds41_model          static library (gguf / config / tensors / experts / plan / model .cpp).  Part of `all`.
#   ds41_model_test            the loader on the mini GGUF (tools/ds41/make_mini_gguf.py, 3 shards): config values and layer roles, every tensor's
#                              location against an independent read (tools/ds41/gguf_io.py), host dequantisation against the oracle's
#                              (ref/ds41/weights.py), CPU halves against tools/ds41/expert_layout.py, a filled cache slot's bytes, the residency table,
#                              the refusals (dims != G, a wrong type, a missing tensor, a truncated / missing shard), the device side under HostDev.
#   ds41_model_real_plan_test  the REAL model's shapes from the saved headers (third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz):
#                              config, the 1,006 tensors, the byte tallies and the memory plan; no weights needed.
#   ds41_model_halves_test     the half packer against DS-C's `pack_cpu_half` at RealGeom (one random expert).
#
# The tests need Python 3 with NumPy for the fixtures (a fixture-setup test writes the mini GGUF and the golden data of the independent readers);
# without it they are not registered.  Like the other DS-1 test programs they are part of `all` only with STRATA_BUILD_TESTS and always defined.
set(_ds41a_src ${PROJECT_SOURCE_DIR}/src/ds41/model)

find_package(Threads REQUIRED)
add_library(strata_ds41_model STATIC ${_ds41a_src}/gguf.cpp ${_ds41a_src}/config.cpp ${_ds41a_src}/tensors.cpp ${_ds41a_src}/experts.cpp
            ${_ds41a_src}/plan.cpp ${_ds41a_src}/model.cpp)
target_include_directories(strata_ds41_model PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(strata_ds41_model PUBLIC strata_numa Threads::Threads)

set(_ds41a_all EXCLUDE_FROM_ALL)
if(STRATA_BUILD_TESTS)
  set(_ds41a_all "")
endif()

# the test programs run the loader on HostDev (DS1-G's emulator Dev: std::aligned_alloc, POSIX only like the emulator itself)
# a test program is defined once its source exists (the package grows test by test; a missing file must never break the configure of the others)
if(NOT MSVC AND EXISTS ${_ds41a_src}/model_test.cpp)
  add_executable(ds41_model_test ${_ds41a_all} ${_ds41a_src}/model_test.cpp)
  target_link_libraries(ds41_model_test PRIVATE strata_ds41_model)
endif()
if(NOT MSVC AND EXISTS ${_ds41a_src}/model_real_plan_test.cpp)
  add_executable(ds41_model_real_plan_test ${_ds41a_all} ${_ds41a_src}/model_real_plan_test.cpp)
  target_link_libraries(ds41_model_real_plan_test PRIVATE strata_ds41_model)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND EXISTS ${_ds41a_src}/model_sparse_test.cpp)
  add_executable(ds41_model_sparse_test ${_ds41a_all} ${_ds41a_src}/model_sparse_test.cpp)
  target_link_libraries(ds41_model_sparse_test PRIVATE strata_ds41_model)
endif()
if(NOT MSVC AND TARGET strata_ds41_cpu AND EXISTS ${_ds41a_src}/model_halves_test.cpp)
  add_executable(ds41_model_halves_test ${_ds41a_all} ${_ds41a_src}/model_halves_test.cpp)
  target_link_libraries(ds41_model_halves_test PRIVATE strata_ds41_model strata_ds41_cpu)
endif()

if(STRATA_BUILD_TESTS)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND AND EXISTS ${_ds41a_src}/model_fixture.py)
    set(_ds41a_fx ${CMAKE_CURRENT_BINARY_DIR}/ds41_model_fixture)
    # writes the mini GGUF (3 shards), the independent reads (golden/*) and the real model's headers as text
    set(_ds41a_fx_args "")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(_ds41a_fx_args --sparse-real)     # the real model's 12 shards as sparse files (403 GB apparent, a few MB on disk) for ds41_model_sparse_test
    endif()
    add_test(NAME ds41_model_fixture COMMAND ${Python3_EXECUTABLE} ${_ds41a_src}/model_fixture.py --out ${_ds41a_fx} ${_ds41a_fx_args})
    set_tests_properties(ds41_model_fixture PROPERTIES FIXTURES_SETUP ds41_model_fixture TIMEOUT 300)
    # the sparse shards have a huge apparent size: remove them when the tests that use them are done
    add_test(NAME ds41_model_fixture_cleanup COMMAND ${CMAKE_COMMAND} -E remove_directory ${_ds41a_fx}/real)
    set_tests_properties(ds41_model_fixture_cleanup PROPERTIES FIXTURES_CLEANUP ds41_model_fixture)
    if(TARGET ds41_model_sparse_test)
      add_test(NAME ds41_model_sparse_test COMMAND ds41_model_sparse_test ${_ds41a_fx})
      set_tests_properties(ds41_model_sparse_test PROPERTIES FIXTURES_REQUIRED ds41_model_fixture TIMEOUT 600)
    endif()
    if(TARGET ds41_model_test)
      add_test(NAME ds41_model_test COMMAND ds41_model_test ${_ds41a_fx})
      set_tests_properties(ds41_model_test PROPERTIES FIXTURES_REQUIRED ds41_model_fixture TIMEOUT 300)
    endif()
    if(TARGET ds41_model_real_plan_test)
      add_test(NAME ds41_model_real_plan_test COMMAND ds41_model_real_plan_test ${_ds41a_fx})
      set_tests_properties(ds41_model_real_plan_test PROPERTIES FIXTURES_REQUIRED ds41_model_fixture TIMEOUT 300)
    endif()
  else()
    message(STATUS "ds41_model: Python 3 (or the fixture script) not found - the loader tests (they need the mini GGUF fixture) are not registered")
  endif()
  if(TARGET ds41_model_halves_test)
    add_test(NAME ds41_model_halves_test COMMAND ds41_model_halves_test)
    set_tests_properties(ds41_model_halves_test PROPERTIES TIMEOUT 120)
  endif()
endif()
