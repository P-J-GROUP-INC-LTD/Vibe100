# What the ninfer V100 forks offer this port (code study, 2026-10-01)

Read-only study of upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) and its V100 forks (geoffwatts,
dollarwong, JimmyMax, andreasknopke, justxiami tp4, mylordmonkeyman's Flash-Next backport), plus two V100 sources
found on the way: [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) and
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM). Numbers are the authors' own unless marked.

## The headline numbers, decomposed

| Claim | What it is |
|---|---|
| ~30 tok/s | **Base decode**, Qwen3.8-27B NVFP4 (23.7 GiB), one V100, no speculation (30.66 dollarwong; 29.28 → 15.4 at 2K → 150K context, geoffwatts). Bandwidth-bound: ~32 ms per pass. |
| 45-123 tok/s | **MTP speculative decoding on real text** (K=3: 67.7 at 2K / 60 % acceptance, 92.7 at 8K / 79 %, 55 at 32K; JimmyMax HTTP sweep average 84, peak 123). |
| 219-236 tok/s | A **tiled benchmark corpus** where context-lookup drafting verifies up to 15 tokens per pass (12.9 tokens/round measured on repeated text). Not representative of new text. |

So on a V100 the format (NVFP4) is not the lever — **speculation is**, and it is already in Strata (MTP drafting,
2.4-3.2 tokens/pass, plus prompt lookup since 0.1.7).

## Licensing

All ninfer forks are Apache-2.0 (no NOTICE files except tp4's). Apache code cannot be relicensed to MIT: copying
it would mean shipping the Apache text, a NOTICE crediting "Neroued/ninfer contributors" and change notices. Prefer
the MIT sources (llama.cpp's fragment maps and FA kernels, v100-skinny's kernels) or reimplement — the useful
tricks are a few lines each.

## Worth taking (ranked)

**For the Qwen3.8-Flash-Next Volta port**

1. Cross-check our WMMA prompt attention against the **m8n8k4 fragment maps** (`volta_mma.cuh`, transcribed from
   llama.cpp, MIT) and ninfer's small-T attention tile (Br 32, Bc 16, 8-half smem padding: +2.56x on the INT8
   variant). Port llama.cpp's Volta FA kernel only if the WMMA kernel underperforms.
2. **ReplaySSM** (record GDN inputs, replay only the accepted prefix on verify) — a smaller verify cost for the
   GDN layers under speculation. Compare with how Strata's verify handles GDN state before porting.
3. **QPN** M=2..8 GEMV (v100-skinny, MIT): only the spec-verify widths gain; at M=1 plain SIMT is already near the
   bandwidth ceiling (752 of ~794 GB/s measured for W8).

**For the DeepSeek MXFP4 GPU expert kernels (DS-D)**

1. **1Cat-vLLM `csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu`** (Apache-2.0; layout from MIT v100-skinny): an M=1
   MXFP4 tensor-core expert GEMV for DeepSeek V4 on V100 — all routed experts in one launch (`grid.y` = route),
   device-side expert ids, split-K 8/16. Hardcoded for hidden 4096 / 256 experts (ours: 5120 / 384). The design
   alternative to the planned dp4a + LUT kernel; benchmark both on the V100.
2. Its **compact grouped launch** of only the active experts (1Cat design notes claim 54.6 → 1.96 ms per token on
   8x V100) — maps onto launching only the GPU cache hits.
3. Two decode tricks: E8M0 → fp16 by rebasing the exponent (`<<10`, one ×16384 multiply), and repack-on-fill so a
   cache slot holds the kernel's preferred nibble order (prepacked ran 1.5-2x faster than in-kernel permutes).
   Caveat: E8M0's range exceeds fp16's — clamp or prescale per row (the dp4a design applies scales in fp32 and has
   no such limit).

## Not worth taking

The ninfer runtime/serving/artifact format, NVFP4's E4M3 scale + per-tensor divisor machinery (pure overhead on
Volta), CUTLASS Sm70 wrappers (cuBLAS FP16 does the same), tensor parallel/NCCL (one GPU here), and the
**Flash-Next backport**: a compile-only port with no tests or measurements that streams every selected expert over
PCIe each layer (~1.3 GB/token; ~9 tok/s at best by estimate) — the opposite of Strata's design (the CPU computes
missed experts where they live; the GPU keeps the hot ones).
