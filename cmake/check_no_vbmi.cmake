# cmake/check_no_vbmi.cmake - the guard behind the `expert_novbmi_no_vbmi` test.
#
# The AVX-512 kernels run on Cascade Lake (AVX-512 F/BW/VL/DQ + VNNI, no VBMI) only if the objects that can run there hold
# no AVX512-VBMI / VBMI2 instruction.  Compiling them without -mavx512vbmi keeps the compiler from emitting one, and this
# proves it on the finished objects, so a later edit that adds a `_mm512_permutexvar_epi8` (vpermb) or an `-mavx512vbmi` to
# the wrong file fails a test instead of faulting on the target machine.
#
#   cmake -DOBJDUMP=<objdump> -DAR=<ar> -DARCHIVE=<libstrata_kernels_cpu.a> -DWORKDIR=<scratch dir>
#         "-DNO_VBMI=expert_novbmi.cpp.o;iq_avx512.cpp.o;..."   archive members that must hold none
#         "-DHAS_VBMI=expert.cpp.o"                             members that must hold some (the check's own control: if
#                                                               this found none, the grep or the build is not what it claims)
#         -P check_no_vbmi.cmake
set(_vbmi "vpermb|vpermi2b|vpermt2b|vpmultishiftqb|vpcompressb|vpcompressw|vpexpandb|vpexpandw|vpshldw|vpshldd|vpshldq|vpshldvw|vpshldvd|vpshldvq|vpshrdw|vpshrdd|vpshrdq|vpshrdvw|vpshrdvd|vpshrdvq")

function(_vbmi_count member out)
  execute_process(COMMAND ${AR} x "${ARCHIVE}" "${member}" WORKING_DIRECTORY "${WORKDIR}" RESULT_VARIABLE _rc)
  if(NOT _rc EQUAL 0 OR NOT EXISTS "${WORKDIR}/${member}")
    message(FATAL_ERROR "check_no_vbmi: ${member} is not in ${ARCHIVE}")
  endif()
  execute_process(COMMAND ${OBJDUMP} -d --no-show-raw-insn "${WORKDIR}/${member}"
                  COMMAND grep -cE "[[:space:]](${_vbmi})[[:space:]]"
                  OUTPUT_VARIABLE _n OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(${out} "${_n}" PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY "${WORKDIR}")
set(_bad 0)
foreach(_m ${NO_VBMI})
  _vbmi_count(${_m} _n)
  message(STATUS "${_m}: ${_n} VBMI-family instructions (must be 0)")
  if(NOT _n STREQUAL "0")
    set(_bad 1)
  endif()
endforeach()
foreach(_m ${HAS_VBMI})
  _vbmi_count(${_m} _n)
  message(STATUS "${_m}: ${_n} VBMI-family instructions (the VBMI build; must be more than 0)")
  if(_n STREQUAL "0" OR _n STREQUAL "")
    set(_bad 1)
  endif()
endforeach()
if(_bad)
  message(FATAL_ERROR "check_no_vbmi: failed")
endif()
