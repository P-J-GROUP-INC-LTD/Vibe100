// src/kernels/cpu/expert_novbmi.cpp - the AVX-512 Q2_0 kernels for CPUs with VNNI but without VBMI (Cascade Lake).
//
// This is expert.cpp compiled a second time, WITHOUT -mavx512vbmi (see CMakeLists.txt), with the macro below
// selecting the shuffle / variable-shift / mask unpack instead of `vpmultishiftqb`.  Because the compiler is not
// given VBMI it cannot emit a VBMI instruction anywhere in this object, however it schedules the loops - which is the
// property that lets a Cascade Lake CPU run it.  `objdump -d` of the object has no vpermb, vpermi2b, vpermt2b,
// vpmultishiftqb, vpcompressb/w, vpexpandb/w or vpshld/vpshrd (the `expert_novbmi_no_vbmi` test, cmake/check_no_vbmi.cmake, greps for them).
#define STRATA_EXPERT_NO_VBMI 1
#include "expert.cpp"
