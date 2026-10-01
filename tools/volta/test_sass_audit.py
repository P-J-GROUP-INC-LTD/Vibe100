"""tools/volta/test_sass_audit.py - unit tests of sass_audit.py's parsers and gate on synthetic cuobjdump text, plus one
end-to-end test on a real object when nvcc is available.   python3 -m unittest tools.volta.test_sass_audit  (or run it)"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sass_audit as A  # noqa: E402

SASS = """
member /x/libfoo.a:foo.cu.o:

Fatbin elf code:
================
arch = sm_70

\t\tFunction : _ZN3foo11mma_kernelEPf
\t.headerflags\t@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM70 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM70)"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;                      /* 0x00000a00ff017624 */
                                                                                                /* 0x000fe400078e00ff */
        /*0010*/              @!PT SHFL.IDX PT, RZ, RZ, RZ, RZ ;                                /* 0x000000fffffff389 */
                                                                                                /* 0x000fe200000e00ff */
        /*0020*/                   HMMA.884.F32.F32.STEP0 R8, R2.reuse.ROW, R6.reuse.COL, R8 ;   /* 0x000000060208723c */
                                                                                                /* 0x000fe20000000000 */
        /*0030*/                   HMMA.884.F32.F32.STEP1 R10, R2.reuse.ROW, R6.reuse.COL, R10 ; /* 0x000000060210723c */
                                                                                                /* 0x000fe20000000000 */
        /*0040*/                   STL [R1+0x4], R9 ;                                           /* 0x0000040109007387 */
                                                                                                /* 0x000fe20000100800 */
        /*0050*/                   LDL.64 R4, [R1+0x8] ;                                        /* 0x0000080001047983 */
                                                                                                /* 0x000fe20000100a00 */
        /*0060*/                   EXIT ;                                                       /* 0x000000000000794d */
                                                                                                /* 0x000fea0003800000 */
        /*0070*/                   BRA 0x70 ;                                                   /* 0xfffffff000007947 */
                                                                                                /* 0x000fc0000383ffff */
        /*0080*/                   NOP;                                                         /* 0x0000000000007918 */
                                                                                                /* 0x000fc00000000000 */
\t\t.....................


\t\tFunction : _ZN3foo11trap_kernelEPi
\t.headerflags\t@"EF_CUDA_SM70"
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;                      /* 0x00000a00ff017624 */
                                                                                                /* 0x000fe400078e00ff */
        /*0010*/              @P0  BPT.TRAP 0x1 ;                                               /* 0x000000040000795c */
                                                                                                /* 0x000fea0003800000 */
        /*0020*/                   BPT.INT 0x1 ;                                                /* 0x000000040000795c */
                                                                                                /* 0x000fea0003800000 */
        /*0030*/                   EXIT ;                                                       /* 0x000000000000794d */
                                                                                                /* 0x000fea0003800000 */
"""

RES = """
Fatbin elf code:
================
arch = sm_70

Resource usage:
 Common:
  GLOBAL:0
 Function _ZN3foo11mma_kernelEPf:
  REG:96 STACK:24 SHARED:2560 LOCAL:8 CONSTANT[2]:8 CONSTANT[0]:464 TEXTURE:0 SURFACE:0 SAMPLER:0
 Function _ZN3foo11trap_kernelEPi:
  REG:12 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:360 TEXTURE:0 SURFACE:0 SAMPLER:0
"""


def rules(text: str):
    d = tempfile.mkdtemp()
    p = Path(d) / "r.txt"
    p.write_text(text, encoding="utf-8")
    try:
        return A.read_rules(p)
    finally:
        shutil.rmtree(d)


class ParserTests(unittest.TestCase):
    def test_sass_counts(self):
        got = A.parse_sass(io.StringIO(SASS))
        self.assertEqual(set(got), {"_ZN3foo11mma_kernelEPf", "_ZN3foo11trap_kernelEPi"})
        m = got["_ZN3foo11mma_kernelEPf"]
        self.assertEqual((m["hmma"], m["trap"], m["local_ops"]), (2, 0, 2))
        # IMAD, SHFL, 2 HMMA, STL, LDL, EXIT, BRA = 8 instructions; the NOP is not counted
        self.assertEqual(m["instr"], 8)
        self.assertEqual(m["member"], "foo.cu.o")
        t = got["_ZN3foo11trap_kernelEPi"]
        # a predicated BPT.TRAP counts, BPT.INT (a breakpoint, not a trap) does not
        self.assertEqual((t["trap"], t["hmma"]), (1, 0))

    def test_duplicate_kernels_are_not_added_up(self):
        got = A.parse_sass(io.StringIO(SASS + SASS))
        self.assertEqual(got["_ZN3foo11mma_kernelEPf"]["hmma"], 2)
        self.assertEqual(got["_ZN3foo11trap_kernelEPi"]["trap"], 1)

    def test_res_usage(self):
        got = A.parse_res_usage(io.StringIO(RES))
        m = got["_ZN3foo11mma_kernelEPf"]
        self.assertEqual((m["reg"], m["stack"], m["shared"], m["local"], m["const0"]), (96, 24, 2560, 8, 464))
        self.assertEqual(got["_ZN3foo11trap_kernelEPi"]["reg"], 12)

    def test_short_name(self):
        self.assertEqual(A.short_name("void strata::kernels::(anonymous namespace)::prompt_attn_kernel<1>(float const*, "
                                      "strata::kernels::QsaAttnPools, int)"), "prompt_attn_kernel<1>")
        self.assertEqual(A.short_name("void mul_mat_q<(ggml_type)21, 72, false>(char const*, int const*, uint3)"),
                         "mul_mat_q<t21, 72, false>")
        self.assertEqual(A.short_name("plain_name"), "plain_name")


def kernels_from(sass: str = SASS, res: str = RES):
    scan = {"path": "/x/foo.o", "sass": A.parse_sass(io.StringIO(sass)), "res": A.parse_res_usage(io.StringIO(res)), "note": ""}
    return A.merge([scan])


class GateTests(unittest.TestCase):
    def test_unexplained_trap_fails(self):
        ks = kernels_from()
        g = A.evaluate(ks, rules("nothing_matches\twhy\n"), [])
        self.assertFalse(g["pass"])
        self.assertEqual(g["unexplained_traps"], ["_ZN3foo11trap_kernelEPi"])

    def test_allowlisted_trap_passes_and_stale_entry_is_reported(self):
        ks = kernels_from()
        allow = rules("# comment\n\ntrap_kernel\tguarded in foo.cpp:launch\nnever_matches\tx\n")
        g = A.evaluate(ks, allow, [])
        self.assertTrue(g["pass"], g["reasons"])
        self.assertEqual(g["allowlisted_traps"]["trap_kernel"], ["_ZN3foo11trap_kernelEPi"])
        self.assertEqual(g["allowlisted_traps"]["never_matches"], [])

    def test_required_hmma(self):
        ks = kernels_from()
        allow = rules("trap_kernel\tj\n")
        self.assertTrue(A.evaluate(ks, allow, rules("mma_kernel\tmust use tensor cores\n"))["pass"])
        # more HMMA than the kernel has
        g = A.evaluate(ks, allow, rules("mma_kernel\t5\tneeds five\n"))
        self.assertFalse(g["pass"])
        self.assertIn("2 HMMA (< 5)", g["reasons"][0])
        # a kernel that matches but has none
        g = A.evaluate(ks, allow, rules("trap_kernel\tj\n"))
        self.assertFalse(g["pass"])

    def test_required_pattern_matching_nothing_fails(self):
        ks = kernels_from()
        g = A.evaluate(ks, rules("trap_kernel\tj\n"), rules("prompt_attn_volta\tthe port's kernel\n"))
        self.assertFalse(g["pass"])
        self.assertIn("matched NO kernel", g["reasons"][0])

    def test_rule_file_errors(self):
        with self.assertRaises(A.AuditError):
            rules("only_one_column\n")
        with self.assertRaises(A.AuditError):
            rules("(unclosed\tj\n")

    def test_shipped_rule_files_parse(self):
        allow = A.read_rules(A.DEFAULT_ALLOWLIST)
        self.assertGreaterEqual(len(allow), 6)
        for rx, text, fields in allow:
            self.assertTrue(fields[0], f"allowlist entry {text!r} has no justification")
        self.assertTrue(A.read_rules(A.DEFAULT_REQUIRED))

    def test_shipped_allowlist_regexes_match_the_real_names(self):
        allow = A.read_rules(A.DEFAULT_ALLOWLIST)
        def hit(name):
            return any(rx.search(name) for rx, _, _ in allow)
        args = "(char const*, int const*, int const*, int const*, float*, float*, float const*, uint3, int)"
        # trapped on the upstream sm_70 build ...
        self.assertTrue(hit("void mul_mat_q<(ggml_type)21, 72, false>" + args))
        self.assertTrue(hit("void mul_mat_q<(ggml_type)42, 24, true>" + args))
        self.assertTrue(hit("void quantize_mmq_nvfp4<true, false>(float const*)"))
        self.assertTrue(hit("strata::kernels::(anonymous namespace)::block_scores_tc_kernel(float const*)"))
        self.assertTrue(hit("void strata::kernels::(anonymous namespace)::prompt_attn_kernel<3>(float const*)"))
        self.assertTrue(hit("strata::kernels::(anonymous namespace)::prompt_attn_i8_kernel(float const*)"))
        # ... and the widths the host DOES launch must not be waved through
        for j in (8, 16, 32, 64, 128):
            self.assertFalse(hit(f"void mul_mat_q<(ggml_type)21, {j}, true>" + args), j)
        for j in (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128):
            self.assertFalse(hit(f"void mul_mat_q<(ggml_type)21, {j}, false>" + args), j)
        # the port's own kernels are not covered by the upstream stubs' entries
        self.assertFalse(hit("void strata::kernels::(anonymous namespace)::prompt_attn_volta_kernel<1>(float const*)"))
        self.assertFalse(hit("strata::kernels::(anonymous namespace)::prompt_attn_volta_i8_kernel(float const*)"))


@unittest.skipUnless(A.find_tool("cuobjdump") and shutil.which("nvcc") or Path("/usr/local/cuda-12.8/bin/nvcc").exists(),
                     "needs a CUDA 12.x toolkit")
class EndToEnd(unittest.TestCase):
    def test_object_with_wmma_and_trap(self):
        nvcc = shutil.which("nvcc") or "/usr/local/cuda-12.8/bin/nvcc"
        d = tempfile.mkdtemp()
        try:
            src = Path(d) / "t.cu"
            src.write_text("""
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
__global__ void prompt_attn_volta_kernel(const half* a, const half* b, float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
    wmma::fill_fragment(fc, 0.f);
    wmma::load_matrix_sync(fa, a, 16); wmma::load_matrix_sync(fb, b, 16);
    wmma::mma_sync(fc, fa, fb, fc);
    wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}
__global__ void trapper_kernel(int* p) { if (p[0] == 7) __trap(); p[1] = 3; }
""")
            obj = Path(d) / "t.o"
            r = subprocess.run([nvcc, "-arch=sm_70", "-Wno-deprecated-gpu-targets", "-c", str(src), "-o", str(obj)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                self.skipTest("this nvcc cannot target sm_70: " + r.stderr[:200])
            js = Path(d) / "a.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = A.main([str(obj), "--json", str(js)])
            self.assertEqual(rc, 1)   # trapper_kernel is not in the allowlist
            doc = json.loads(js.read_text())
            ks = {k["demangled"].split("(")[0]: k for k in doc["kernels"]}
            self.assertGreater(ks["prompt_attn_volta_kernel"]["hmma"], 0)
            self.assertGreater(ks["trapper_kernel"]["trap"], 0)
            self.assertEqual(doc["gate"]["unexplained_traps"], [ks["trapper_kernel"]["name"]])
            self.assertEqual(doc["gate"]["required"][0]["failing"], [])
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
