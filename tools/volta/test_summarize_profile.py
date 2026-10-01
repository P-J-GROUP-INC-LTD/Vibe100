"""tools/volta/test_summarize_profile.py - tests of summarize_profile.py on synthetic text built from the EXACT printf formats of
src/program/generate.cpp (the test also checks that each format is still present in that file, so a reworded engine line
fails here rather than silently emptying the Phase-0 table).

    python3 -m unittest tools.volta.test_summarize_profile        (from the repository root)
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_profile as S  # noqa: E402

GEN = Path(__file__).resolve().parents[2] / "src" / "program" / "generate.cpp"


def c(fmt: str, *args) -> str:
    """printf with C's length modifiers: %lld / %zu -> %d."""
    return fmt.replace("%lld", "%d").replace("%zu", "%d").replace("%llu", "%d") % args


# name -> (the C format string literal exactly as it appears in generate.cpp, the arguments)
FORMATS = {
    "decode": ('"%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\\n"', ("decode", 200, 9100.0, 21.98)),
    "prefill": ('"%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\\n"',
                ("prefill", 32999, 51234.5, 644.07, 51300.0)),
    "per_token": ('"%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\\n"', ("  per token", 45.5)),
    "cpu_pool": ('"%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\\n"',
                 ("  the CPU expert pool", 19.1, 48, 233.0, 11184)),
    "pool_phases": ('"%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\\n"', ("  pool phases", 0.1, 17.2, 1.8)),
    "blobs": ('"%-24s %lld blobs read\\n"', ("  expert blobs", 4321)),
    "cache": ('"%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\\n"',
              ("  R4 expert-cache hits", 5430, 10000, 0.543, 120, 0, 97.5)),
    "spec": ('"%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\\n"',
             ("speculation", 70, 6, 130, 350, 0.371, 2.86)),
    "minp": ('"  (min draft probability %.2f)\\n"', (0.5,)),
    "suffix": ('"%-24s %lld windows, drafts accepted %lld of %lld\\n"', ("suffix drafts", 5, 9, 14)),
    "verify": ('"%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "'
               '"distinct / %.2f routed per layer\\n"', ("verify window", 4.2, 12.9, 1.1, 0.4, 3.1, 6.8)),
    "pool_multi": ('"%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "'
                   '"CPU pool call %.3f ms/round\\n"', ("pool multi", 5.5, 0.6, 2.9, 61.5, 12.4)),
    "pcie": ('"%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\\n"', ("pcie experts", 1.25, 51)),
    "mtp": ('"%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\\n"', ("mtp", 2.3, 70, 880.0, 1500.0)),
    "tiers": ('"%-24s decode: RAM %lld blobs, files %lld blobs, %.1f MB read from the files (%.2f MB/round, "'
              '"%.1f ms/round of reading, %.2f GB/s per reading thread)%s; prompt copies %.1f MB\\n"',
              ("expert tiers", 900, 0, 0.0, 0.0, 0.0, 0.0, "", 4100.0)),
    "batched": ('"strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "'
                '"streamed %lld (%lld by DMA, host %.1f ms), resident %lld; PLE %.1f ms\\n"',
                (32488, 4, 51234.5, 634.1, 10, 0, 1.0, 5, 3.0)),
    "floor": ('"GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\\n"', (11.87, 84.25)),
    "floor_layers": ('"                  %8.3f ms layers (%lld x pre+post)\\n"', (10.2, 48)),
    "floor_head": ('"                  %8.3f ms LM head\\n"', (1.67,)),
}


def fmt(name: str) -> str:
    """The C format literal (adjacent literals joined) -> text, with the arguments filled in."""
    lit, args = FORMATS[name]
    f = lit.replace('""', "")[1:-1].replace("\\n", "\n")
    return c(f, *args)


def stats_text() -> str:
    window = "window sizes            T1:3 T2:12 T3:20 T4:15 T5:10 T6:10  (min draft probability 0.50)\n"
    hist = "accepted per round     0:20 1:18 2:12 3:9 4:6 5:5\n"
    return "".join([
        "strata generate: prompt chunk auto: 8192 tokens\n", fmt("batched"),
        fmt("spec"), window, fmt("suffix"), hist, fmt("verify"), fmt("pool_multi"), fmt("pcie"), fmt("mtp"), fmt("tiers"),
        "prompt  : 1 2 3\noutput  : 4 5 6\n",
        fmt("decode"), fmt("prefill"), fmt("per_token"), fmt("cpu_pool"), fmt("pool_phases"), fmt("blobs"), fmt("cache"),
    ])


def floor_text() -> str:
    return fmt("floor") + fmt("floor_layers") + "                      0.254 ms per layer\n" + fmt("floor_head")


def normalized_source() -> str:
    """generate.cpp with adjacent string literals joined and argument lists unwrapped, so a format split over several
    source lines can be found as one literal."""
    src = GEN.read_text(encoding="utf-8")
    src = re.sub(r'"\s*\n\s*"', "", src)
    return src


class FormatDriftTests(unittest.TestCase):
    @unittest.skipUnless(GEN.exists(), "src/program/generate.cpp not found")
    def test_every_format_is_still_in_the_engine_source(self):
        src = normalized_source()
        for name, (lit, args) in FORMATS.items():
            self.assertIn(lit.replace('""', ""), src, f"generate.cpp no longer contains the {name} format {lit!r}: "
                                                      "update summarize_profile.py's patterns")

    @unittest.skipUnless(GEN.exists(), "src/program/generate.cpp not found")
    @unittest.skipUnless(GEN.exists(), "src/program/generate.cpp not found")
    def test_cpu_kernel_notice_is_still_in_the_source(self):
        self.assertIn("strata generate: this CPU has no AVX-512: the expert kernels run on %s (multi-token for the i-quant gate/up rows)",
                      normalized_source())

    def test_server_lines_are_still_in_the_source(self):
        src = normalized_source()
        for piece in ('"strata serve: prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), '
                      '%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\\n"',
                      '"strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)\\n"',
                      '"DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f\\n"'):
            self.assertIn(piece, src)
        srv = (Path(__file__).resolve().parents[2] / "serve" / "server.py").read_text(encoding="utf-8")
        self.assertIn('print(f"[strata] done: {n} tokens in {el:.0f} s ({rate:.1f} tok/s) "', srv)


class StatsParseTests(unittest.TestCase):
    def setUp(self):
        self.st = S.parse_stats(stats_text())

    def test_throughput(self):
        self.assertEqual(self.st["decode"], {"tokens": 200, "ms": 9100.0, "tok_s": 21.98})
        self.assertEqual(self.st["prefill"]["tokens"], 32999)
        self.assertEqual(self.st["prefill"]["ttft_ms"], 51300.0)
        self.assertEqual(self.st["ms_per_token"], 45.5)
        self.assertEqual(self.st["batched_prefill"], {"tokens": 32488, "chunks": 4, "ms": 51234.5, "tok_s": 634.1})

    def test_speculation(self):
        sp = self.st["speculation"]
        self.assertEqual((sp["rounds"], sp["window"], sp["accepted"], sp["offered"]), (70, 6, 130, 350))
        self.assertAlmostEqual(sp["tokens_per_round"], 2.86)
        self.assertEqual(self.st["window_sizes"], {1: 3, 2: 12, 3: 20, 4: 15, 5: 10, 6: 10})
        self.assertEqual(self.st["accepted_hist"], {0: 20, 1: 18, 2: 12, 3: 9, 4: 6, 5: 5})
        self.assertEqual(self.st["suffix_drafts"], {"windows": 5, "accepted": 9, "offered": 14})

    def test_cpu_and_cache(self):
        self.assertEqual(self.st["cache"], {"hits": 5430, "lookups": 10000, "rate": 0.543, "admitted": 120, "refused": 0, "full_pct": 97.5})
        self.assertEqual(self.st["pool_multi"], {"gate_up": 5.5, "quantize": 0.6, "down": 2.9, "gb_s": 61.5, "pool_call": 12.4})
        self.assertEqual(self.st["verify_window"]["pool"], 12.9)
        self.assertEqual(self.st["verify_window"]["cpu_routed"], 6.8)
        self.assertEqual(self.st["cpu_pool_token"], {"ms_per_token": 19.1, "layers": 48, "positions": 233.0, "dispatches": 11184})
        self.assertEqual(self.st["pool_phases"], {"wait_park": 0.1, "drain": 17.2, "re_park": 1.8})
        self.assertEqual(self.st["expert_blobs"], 4321)
        self.assertEqual(self.st["pcie_experts"], {"distinct_per_layer": 1.25, "share_256": 51})
        self.assertEqual(self.st["mtp"]["rounds"], 70)

    def test_gpu_floor(self):
        f = S.parse_stats(floor_text())
        self.assertEqual(f["gpu_floor"], {"ms_per_token": 11.87, "tok_s": 84.25, "ms_layers": 10.2, "ms_head": 1.67})

    def test_cpu_kernel_notice(self):
        st = S.parse_stats("strata generate: this CPU has no AVX-512: the expert kernels run on AVX-2 (multi-token for the i-quant gate/up rows)\n")
        self.assertEqual(st["cpu_kernels"], "AVX-2")
        st = S.parse_stats("strata generate: this CPU has no AVX-512: the expert kernels run on ggml-cpu vec_dot (STRATA_NO_IQ256 set) "
                           "(multi-token for the i-quant gate/up rows)\n")
        self.assertTrue(st["cpu_kernels"].startswith("ggml-cpu vec_dot"))
        rows = S.build_summary({"cpu_kernels": "AVX-2"}, {}, None, None, {}, None)["rows"]
        self.assertEqual(rows[0]["name"], "CPU expert kernels")
        self.assertEqual(rows[0]["value"], "AVX-2")

    def test_absent_lines_are_absent(self):
        st = S.parse_stats("strata generate: nothing useful\noutput  : 1 2\n")
        self.assertEqual(st, {})


class NsysTests(unittest.TestCase):
    KERN = (
        "Generating SQLite file decode.sqlite from decode.nsys-rep\n"
        "Processing [decode.sqlite] with [/opt/nsight/reports/cuda_gpu_kern_sum.py]...\n"
        "\n ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):\n\n"
        "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
        '60.0,6000000,100,60000.0,60000,50000,70000,1000.0,"void foo<(ggml_type)21, 72, false>(char const*, int)"\n'
        '30.0,3000000,300,10000.0,10000,9000,11000,100.0,"strata::kernels::(anonymous namespace)::gr_down_multi_kernel<2560>(strata::kernels::GrMulti)"\n'
        '10.0,1000000,50,20000.0,20000,19000,21000,100.0,"copy_kernel"\n\n'
        "Processing [decode.sqlite] with [/opt/nsight/reports/cuda_gpu_mem_size_sum.py]...\n")
    MEMSIZE = ("\n ** GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):\n\n"
               "Total (MB),Count,Avg (MB),Med (MB),Min (MB),Max (MB),StdDev (MB),Operation\n"
               '4000.5,2000,2.0,1.0,0.001,64.0,5.0,"[CUDA memcpy Host-to-Device]"\n'
               '12.0,400,0.03,0.01,0.001,1.0,0.1,"[CUDA memcpy Device-to-Host]"\n'
               '0.5,10,0.05,0.05,0.05,0.05,0.0,"[CUDA memset]"\n')
    MEMTIME = ("Time (%),Total Time (ns),Count,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Operation\n"
               '95.0,380000000,2000,190000.0,1,1,1,1,"[CUDA memcpy Host-to-Device]"\n'
               '5.0,20000000,400,50000.0,1,1,1,1,"[CUDA memcpy Device-to-Host]"\n')

    def test_kernel_summary(self):
        rows = S.parse_nsys_csv(self.KERN, "name")
        self.assertEqual(len(rows), 3)                      # the trailing "Processing ..." line is not a row
        k = S.kernel_table(rows, top=2)
        self.assertEqual(k["n_kernels"], 3)
        self.assertEqual(len(k["top"]), 2)
        self.assertAlmostEqual(k["top"][0]["pct"], 60.0)
        self.assertEqual(k["top"][0]["name"], "void foo<(ggml_type)21, 72, false>(char const*, int)")
        self.assertAlmostEqual(k["top_pct"], 90.0)
        self.assertEqual(k["total_ns"], 10_000_000)

    def test_tensor_core_share_by_name(self):
        csv_ = ("Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
                '50.0,5000,10,500.0,1,1,1,1,"volta_fp16_s884gemm_fp16_256x128_ldg8_f2f_nn"\n'
                '25.0,2500,10,250.0,1,1,1,1,"volta_sgemm_128x64_nn"\n'
                '25.0,2500,10,250.0,1,1,1,1,"void strata::kernels::(anonymous namespace)::prompt_attn_volta_kernel<1>(float const*)"\n')
        k = S.kernel_table(S.parse_nsys_csv(csv_, "name"))
        self.assertEqual(k["tensor_core_ns"], 7500)
        self.assertEqual(len(k["tensor_core_names"]), 2)
        rows = S.build_summary({}, {}, k, None, {}, None)["rows"]
        tc = [r for r in rows if r["name"].startswith("tensor-core kernel time")][0]
        self.assertEqual(tc["value"], "75.0%")

    def test_older_header_spelling(self):
        old = "Time(%),Total Time(ns),Instances,Average(ns),Minimum(ns),Maximum(ns),Name\n7.0,700,3,233.0,1,2,k1\n93.0,9300,5,1860.0,1,2,k2\n"
        k = S.kernel_table(S.parse_nsys_csv(old, "name"))
        self.assertEqual(k["top"][0]["name"], "k2")
        self.assertEqual(k["top"][0]["instances"], 5)

    def test_memory_reports(self):
        mem = S.mem_tables(S.parse_nsys_csv(self.MEMSIZE, "operation"), S.parse_nsys_csv(self.MEMTIME, "operation"))
        self.assertEqual(mem["size_mb"]["[CUDA memcpy Host-to-Device]"], 4000.5)
        self.assertEqual(S._sum_ops(mem["size_mb"], "host-to-device"), 4000.5)
        self.assertEqual(S._sum_ops(mem["time_ns"], "memcpy"), 400000000)

    def test_garbage_gives_no_rows(self):
        self.assertEqual(S.parse_nsys_csv("nothing\nat all\n", "name"), [])


class ServeLogTests(unittest.TestCase):
    def test_deployed_log(self):
        log = (c("strata serve: prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), "
                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                 5000, 0, 5000, 8000.0, 625.0, 300, 7000.0, 42.9, 410, 900, 2, "")
               + c("strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)\n", 61.3, 613, 1000)
               + c("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld %lld %lld %.1f\n", 300, 5000, 8000.0, 7000.0, "length", 410, 900, 0, 613, 1000, 5, 0, 0.0)
               + "[strata] done: 300 tokens in 16 s (43.0 tok/s) (length, cancel=False), expert cache 61.3% hit\n")
        reqs = S.parse_serve_log(log)
        self.assertEqual(len(reqs), 2)
        self.assertEqual(reqs[0]["prompt_tokens"], 5000)
        self.assertEqual(reqs[0]["decode_tok_s"], 42.9)
        self.assertEqual(reqs[0]["drafts_accepted"], 410)
        self.assertAlmostEqual(reqs[0]["hit_rate"], 0.613)
        self.assertTrue(reqs[1]["server_done"])
        self.assertAlmostEqual(reqs[1]["hit_rate"], 0.613)


class EndToEndTests(unittest.TestCase):
    def test_cli_builds_the_table(self):
        d = Path(tempfile.mkdtemp())
        (d / "stats.txt").write_text(stats_text())
        (d / "floor.txt").write_text(floor_text())
        (d / "kern.csv").write_text(NsysTests.KERN)
        (d / "ms.csv").write_text(NsysTests.MEMSIZE)
        (d / "mt.csv").write_text(NsysTests.MEMTIME)
        (d / "run.txt").write_text(fmt("decode") + fmt("prefill"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = S.main(["--stats", str(d / "stats.txt"), "--gpu-floor", str(d / "floor.txt"), "--nsys-kern", str(d / "kern.csv"),
                         "--nsys-memsize", str(d / "ms.csv"), "--nsys-memtime", str(d / "mt.csv"), "--nsys-run", str(d / "run.txt"),
                         "--gpu", "Tesla V100-PCIE-32GB, 7.0", "--json", str(d / "o.json")])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        for needle in ("decode tok/s", "21.98", "MTP acceptance", "2.86", "expert cache hit rate", "54.3%", "CPU expert compute",
                       "12.40 ms/round", "GPU idle share", "PCIe traffic per token", "Tesla V100", "void foo<(ggml_type)21, 72, false>"):
            self.assertIn(needle, text)
        # idle share: wall = 9100 ms / 200 tokens = 45.5 ms per token; floor 11.87 ms -> 1 - 11.87/45.5 = 74%
        self.assertIn("74%", text)
        # PCIe per token: (4000.5 + 12) MB over the profiled run's 200 generated tokens = 20.1 MB
        self.assertIn("20.1 MB", text)
        js = json.loads((d / "o.json").read_text())
        self.assertEqual(js["kernels"]["n_kernels"], 3)
        self.assertEqual(js["gpu_floor"]["ms_per_token"], 11.87)

    def test_nothing_to_do(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(S.main([]), 2)


if __name__ == "__main__":
    unittest.main()
