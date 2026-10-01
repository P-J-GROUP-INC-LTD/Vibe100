# DS-C (CPU experts): the MXFP4 routed-expert kernels, their tests and the benchmark.
# Included by the top-level CMakeLists.txt (cmake/ds41_*.cmake).  Nothing here needs CUDA; the library needs nothing
# but a compiler, the optional ggml cross-check needs the ggml targets the top level already builds.
set(_ds41c_src ${PROJECT_SOURCE_DIR}/src/ds41/cpu)

add_library(strata_ds41_cpu STATIC ${_ds41c_src}/mxfp4_expert.cpp ${_ds41c_src}/mxfp4_avx2.cpp
            ${_ds41c_src}/mxfp4_avx512.cpp)
target_include_directories(strata_ds41_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include)
find_package(Threads REQUIRED)
target_link_libraries(strata_ds41_cpu PUBLIC Threads::Threads)
# Per-file ISA flags, as upstream does: only the kernel files get them, the portable file runs on any x86-64 and the
# kernels are chosen at run time by CPUID (resolve_isa).  AVX-512 here means F/BW/VL/DQ + VNNI and deliberately NOT
# VBMI: Cascade Lake (Xeon Gold 6226) has no VBMI.
if(MSVC)
  set_source_files_properties(${_ds41c_src}/mxfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(${_ds41c_src}/mxfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties(${_ds41c_src}/mxfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  set_source_files_properties(${_ds41c_src}/mxfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                              "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mfma;-mf16c")
endif()

if(STRATA_BUILD_TESTS)
  # the benchmark: GB/s of expert weights consumed vs the plain-read ceiling (a program, not a test; run it by hand)
  add_executable(ds41_cpu_mxfp4_bench ${_ds41c_src}/mxfp4_expert_bench.cpp)
  target_link_libraries(ds41_cpu_mxfp4_bench PRIVATE strata_ds41_cpu)
  if(MSVC)
    set_source_files_properties(${_ds41c_src}/mxfp4_expert_bench.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    # the bench's read-bandwidth probe uses zmm loads (it checks CPUID itself before running)
    set_source_files_properties(${_ds41c_src}/mxfp4_expert_bench.cpp PROPERTIES COMPILE_OPTIONS
                                "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq")
  endif()

  add_executable(ds41_cpu_mxfp4_test ${_ds41c_src}/mxfp4_expert_test.cpp)
  target_link_libraries(ds41_cpu_mxfp4_test PRIVATE strata_ds41_cpu)
  # skips an ISA the CPU lacks (and says so); the second registration is the gate for the target Xeon
  add_test(NAME ds41_cpu_mxfp4_test COMMAND ds41_cpu_mxfp4_test)
  add_test(NAME ds41_cpu_mxfp4_test_require_avx512 COMMAND ds41_cpu_mxfp4_test --quick --require-avx512 --require-avx2)
  set_tests_properties(ds41_cpu_mxfp4_test ds41_cpu_mxfp4_test_require_avx512 PROPERTIES TIMEOUT 600)

  # MXFP4 semantics against ggml itself (to_float and the CPU vec_dot for MXFP4 x Q8_0), when ggml is part of the build
  if(TARGET ggml-cpu AND TARGET ggml-base)
    add_executable(ds41_cpu_mxfp4_ggml_xcheck ${_ds41c_src}/mxfp4_ggml_xcheck.cpp)
    target_link_libraries(ds41_cpu_mxfp4_ggml_xcheck PRIVATE strata_ds41_cpu ggml-cpu ggml-base)
    add_test(NAME ds41_cpu_mxfp4_ggml_xcheck COMMAND ds41_cpu_mxfp4_ggml_xcheck)
    set_tests_properties(ds41_cpu_mxfp4_ggml_xcheck PROPERTIES TIMEOUT 300)
  endif()
endif()
