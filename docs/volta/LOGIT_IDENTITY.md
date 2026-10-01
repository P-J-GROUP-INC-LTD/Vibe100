# Logit identity: does the V100 port give the model's answers?

The requirement: *put input into the model and get the same output on this engine as on llama.cpp; the model's quality must not be
degraded; and a logit-identical test between this engine and running it normally.* This page says what can be proven, by which test, and how
to read the result. One script runs everything: `tools/volta/logit_identity.sh` (step 6b of [RUNBOOK.md](RUNBOOK.md)).

**Nothing here has run on a V100 yet.** The tools were tested against fakes and, for the llama.cpp side, against the real llama.cpp at the
pinned commit on a tiny model ([What was checked against the real llama.cpp](#what-was-checked-against-the-real-llamacpp)). The pass rule of
tier 2 is **provisional**: the first real run on the box confirms it or moves it.

## The short version

| Tier | Row | Compares | PASS means | FAIL means |
|---|---|---|---|---|
| **1: bit-exact** (same card, same pack, same prompt, GPU expert set pinned) | **1c** | the same command twice | the engine is deterministic: every logit of every scored position has the same 32 bits | nothing below can be read as "exact": the difference is run-to-run noise |
| | **1a** | the port with its fast paths **off** against **upstream Strata 3906943 built for sm_70** | the port changed none of the base arithmetic: same logits, bit for bit, as upstream on this card | something besides the two Volta kernels moved a number (dispatch, build flags, device code) |
| | **1b** | NUMA mirror on against off | the mirror copies bytes and nothing else: identical logits | a replica differs from the primary, or a kernel read the wrong copy |
| | **1d** | CPU expert tier AVX-512 against AVX2 | (INFO, no pass mark) identical, or how far apart: the answer is "tier 2's noise budget" | - |
| **2: quality against llama.cpp on the same GGUF** | **2c** | llama.cpp (CUDA, sm_70) writes the reference | the reference file is usable | llama.cpp could not run the model |
| | **2d** | llama.cpp CPU against that reference: **the noise floor** | (INFO) how far two trusted llama.cpp backends are from each other | the two disagree beyond 0.05 nats: broken reference |
| | **2e** | **Strata, port defaults (every fast path on)** against the reference | mean KL and top-1 mismatch no worse than max(2 x the floor, an absolute floor), and the paired ln-PPL change within 2 standard errors of zero | the engine is further from llama.cpp than llama.cpp's backends are from each other |
| | **2f** | Strata with the fast paths off (`STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0`) | the same rule, without the Volta kernels | as 2e: a FAIL here and a PASS in 2e (or the reverse) says which side owns the difference |
| | **2g** | (`--kv-fp16-row`) Strata with a 16-bit KV cache | the same rule with the engine's int8 KV quantisation taken out | - |

Rows are PASS / FAIL / SKIP (with the reason) / INFO (a measurement with no pass mark). Exit status 0 means no row FAILED.

## Why "the same output as llama.cpp" cannot mean the same bits

* **Every kernel sums in its own order.** A logit is a sum of millions of products; floating-point addition is not associative, so another
  summation order (another tile size, another thread mapping, FP16 or FP32 accumulators, tensor cores or CUDA cores) changes the last
  bits, and those last bits propagate through 48 layers and a router that picks experts by comparing numbers.
* **llama.cpp's own CPU and CUDA builds do not match each other.** The same GGUF run by `llama-perplexity` with and without a GPU gives
  slightly different distributions - the weights are the same, the arithmetic is not (different matrix kernels, different orders of accumulation,
  different attention implementations; how large the difference is for this model is exactly what row 2d measures). Bit-identity with *a* llama.cpp
  would be identity with one of its backends only.
* **Strata and llama.cpp are two independent implementations of the architecture** (`qwen4exp` in llama.cpp, Strata's own kernels, router
  and expert placement). Where an expert runs - the GPU cache or the CPU - already changes rounding inside Strata; upstream measured 2-5 %
  top-1 flips from the expert cache being on or off.

So the claim that can be made, and measured, is: **the engine is no further from llama.cpp than llama.cpp's own backends are from each
other** (tier 2) - plus, where bit-identity *is* meaningful because it is the same code on the same card, exact identity (tier 1).

## Why KL against llama.cpp's own noise floor is the right "no quality loss" test

This is how llama.cpp itself measures whether a quantisation (or a backend) degraded a model:

```
llama-perplexity -m M.gguf -f text -c 4096 --kl-divergence-base BASE                          # the reference, once
llama-perplexity -m M.gguf -f text -c 4096 --kl-divergence --kl-divergence-base BASE          # the thing under test
```

It prints, over the same text, the **mean KL divergence** of the new next-token distributions from the saved ones (nats; 0 = identical
distributions), the fraction of positions where both pick the **same top-1** token ("Same top p"), the perplexity of both and the **paired
change of ln PPL** with its standard error, and the change of the right token's probability (**Delta p**). Those numbers see what a
user sees - a different distribution, a different likely token, a surprised or unsurprised model - and, unlike a maximum |logit
difference|, they are weighted by how much probability mass moved. Comparing two backends of llama.cpp gives the **floor**: how much two
correct implementations of the same model differ just from arithmetic. The Strata engine is scored on the *same* chunks against the *same*
reference; if it lands inside the floor it is, statistically, one more correct implementation.

`tools/volta/golden_compare.py --ref-kld BASE` does that for Strata: it reads BASE (the format, and the scoring rule reproduced in NumPy:
`tools/volta/kld_format.py`), runs the engine on **exactly the chunks in the file** - one engine launch per chunk, each an independent sequence
from position 0 like llama.cpp's - with the first n_ctx/2 tokens through the batched prefill (where the Volta kernels live) and the rest
through the decode path, scores the same 2,047 positions per chunk (n_ctx = 4096) and prints llama.cpp's own statistics.

**The rule (PROVISIONAL).** With `floor` = the CPU-vs-CUDA numbers of row 2d: PASS when

* mean KL <= max(2 x floor KL, **5e-4** nats), and
* the share of positions whose top-1 differs <= max(2 x the floor's, **1 %**), and
* the paired ln-PPL change is within 2 standard errors of zero (or within 2 x the floor's own change, when the floor itself has a
  significant bias).

The absolute floors exist because an independent engine is not a second backend of the same code: if llama.cpp's two backends happened to agree
to 3e-5, "twice the floor" would be an unreasonably strict bar for a different implementation. 5e-4 nats is far above what the base file's own
16-bit rounding contributes (the fixture's base scored against itself: -2.7e-7) and, as a prior from llama.cpp's published quantisation tables (not measured here),
well below the 1e-3 to 1e-2 a 4- to 5-bit quantisation of a model costs; 1 % of top-1
positions is about nine standard errors of a top-1 rate near 99 % over ~8,000 positions, so a real shift and not sampling noise. Both are
guesses to be checked against the first real floor, and are options (`--abs-kl`, `--abs-top1` on the script; `--kld-abs-kl`, `--kld-abs-top1`,
`--kld-factor` on `golden_compare.py`).

## What each tier does

### Tier 1: bit-exact (`golden_compare.py --exact`)

The same engine, card, pack and prompt (the 33,000-token `long` prompt, last 512 positions scored: the batched path and the QSA sparse selection both
run), compared **bit for bit** - every float32 logit of every scored position (`--exact` reports identical or not, the first
differing position and vocabulary index, the number of rows and values that differ, the largest distance in ULPs, and how many rows changed their
top-1). Conditions that make "identical" a fair demand:

| Pinned | How | Why |
|---|---|---|
| the GPU's expert set | `--pcie-frac 0 --adapt-swaps 0`, `--expert-cache` fixed to the first run's slot count | a GPU cache hit and a CPU miss round an expert differently |
| rows of the CPU experts | `STRATA_IQ_MT_MIN=1` | below 2 tokens ggml's dot product runs, from 2 the multi-token kernel: another rounding; pinned so a window's contents cannot matter |
| the CPU tier of the expert rows | `STRATA_FORCE_AVX2=1` (upstream's tier on a Cascade Lake) | the AVX-512 rows sum in another order (row 1d measures it) |
| the Volta kernels | `STRATA_VOLTA_ATTN=0 STRATA_PREFILL_F16_GEMM=0` (+ `STRATA_PROMPT_ATTN_OLD=1`, upstream's own FP32 attention switch) | the WMMA attention and the FP16 GEMM are *deliberately* not bit-identical to what they replace; tier 2 and Gate Q bound them |
| NUMA | `--numa off` for the baseline | row 1b changes only this |

The baseline "B0" is run twice (row 1c, the control) and then reused as the reference of rows 1a, 1b and 1d (`--reuse-ref`), so each further row costs one engine run.

* **1a** needs upstream: `git worktree add --detach build-logit-identity/upstream-src 3906943` (this repository's first commit, which *is* upstream
  Strata 0.1.31) built with CUDA 12.8 for sm_70 using upstream's own flags plus its experimental sm_70 switch
  (`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_GGML_DIR=third_party/llama.cpp`),
  run with `STRATA_PROMPT_ATTN_OLD=1` so it never launches its Turing / Ampere prompt-attention kernel (a `__trap()` stub on sm_70; `tools/volta/dispatch_audit.md`, R7).
  **No patch is applied to make upstream run on compute capability 7.0**: its experimental sm_70 switch and that environment variable are its own, and a V100 owner ran upstream
  this way (PLAN.md section 1). If it still does not run (`unspecified launch failure` in `tier1/1a/cand.log`), row 1a FAILS with that log, and the patch that makes it run belongs in a file next to
  the one described next. The only patch applied, `tools/volta/upstream_logits_dump.patch` (by default,
  `--upstream-hook none` skips it), adds the measurement hook `STRATA_LOGITS_DUMP` to upstream's `verify.cpp` - 19 lines that write the full logits row of each scored
  position when the variable is set and do nothing otherwise - because without it upstream can only print log-probabilities to 9 decimals and row 1a could not say
  "bit for bit". The same hook is in this port's `verify.cpp`.
  The engine config's arguments go to upstream as they are, so a config with port-only options (`--numa` chosen at setup) makes upstream refuse to start: use a config without them for this row.
  On a Cascade Lake box upstream's own AVX-512 tier is off (it needs VBMI), so upstream runs the AVX2 expert rows - what `STRATA_FORCE_AVX2=1` gives the port.
  **SKIPPED for the canonical Q2_0 pack** on a CPU without VBMI: upstream's only kernel for it is AVX-512 + VBMI. Use an IQ pack (the default).
* **1b** needs two NUMA nodes (BIOS: Node Interleaving disabled) and is SKIPPED, with the reason, when the engine's log has no
  `expert arena MIRRORED` line (it kept one copy: the run did not exercise the mirror).
* **1d**: `STRATA_IQ512=1` makes the i-quant rows use the AVX-512 kernels on this CPU (the default keeps AVX2 on a Cascade Lake: measured faster). A difference is expected
  to be summation order only; it is reported, not failed, and belongs to tier 2.

### Tier 2: quality against llama.cpp (`--ref-kld`)

1. Builds of llama.cpp from `third_party/llama.cpp` (the pinned commit `3cf03257`, which knows `qwen4exp`): CUDA for sm_70 with CUDA 12.8
   (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70`), and CPU-only; both target `llama-perplexity`, cached under `build-logit-identity/`.
2. **The GGUF.** `./setup.sh` keeps the original model files next to the pack: `Strata-data/models/Qwen3.8-Flash-Next-GSQ-RCO-<SIZE>-0000{1,2}-of-00002.gguf`
   (the packs in `Strata-data/packs/` are made from them and the engine is started with `--native <shard 1>`); the script takes the path from the engine
   config's `--native`, so llama.cpp loads **the same file** (shard 1 pulls in shard 2). If it is not there the rows are SKIPPED and the script prints how to fetch the
   same two files at the revision `setup.py` pins (`HF_REVISIONS`), e.g. for IQ3_XXS
   `https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf`
   (and `-00002-of-00002.gguf`), into the folder the config names.
3. **The text**: `tools/volta/prompts/kld_text.txt`, 116 KB (about 29,000 tokens, seven 4,096-token chunks): public-domain prose (Alice in Wonderland, A Tale of Two Cities, Sherlock
   Holmes, Moby Dick - Project Gutenberg plain text, licence header removed), code (CUDA, Python, C++ from the upstream snapshot) and a technical page (upstream's `docs/DETAILS.md`). Varied on purpose,
   and some of it new to the model: well-known novels are memorised and make distributions sharper. `make_kld_text.py` records the sources and rebuilds it byte for byte.
   `--text` takes another (wikitext-2 raw is the usual choice in llama.cpp's own comparisons; it needs >= 2 x ctx tokens).
4. **2c**: `llama-perplexity (CUDA) -c 4096 --chunks 4 --kl-divergence-base cuda.kld` - the reference. By default llama.cpp's own `--fit` decides what goes on the 32 GB card; `--llama-cuda-args "-ngl 99 --cpu-moe"` keeps every expert on the CPU if it runs out of memory. The file is large: 2,047 rows of 248,324 16-bit values per chunk = 1.0 GB per chunk for a 248,320-token vocabulary (4 GB for 4 chunks).
5. **2d**: `llama-perplexity (CPU) --kl-divergence --kl-divergence-base cuda.kld` - the floor. (If llama.cpp's log thread loses the last lines at exit, the floor is read from the per-chunk table, which holds the same totals.)
6. **2e / 2f / 2g**: `golden_compare.py --ref-kld cuda.kld --kld-floor llama_cpu_vs_cuda.log`: one engine launch per chunk with the chunk's tokens,
   `--prefill-until 2048` (non-native pack, full logits through `--dump-logits`) or `strata --serve` with the split described below (native IQ packs,
   full logits through `STRATA_LOGITS_DUMP`). The dump of a chunk is 2 GB; it is deleted after scoring (`--keep-logits` keeps it).

**Which positions, and what "the same" means.** llama.cpp evaluates each chunk from position 0 with an empty KV cache, and scores positions n_ctx/2 .. n_ctx-2 against the next token (the first half
is context only; the last position has no target). If the vocabulary adds a BOS token (`tokenizer.ggml.add_bos_token`), llama.cpp overwrites the first token of **every** chunk with BOS before
evaluating it; the script reads that setting from the GGUF (`kld_format.py bos`; none for a Qwen / GPT-2 BPE vocabulary) and the harness does the same. A native (IQ) pack scores through
`strata --serve`, whose prompt reading is split at the *last occurrence of a turn token* - an engine argument, so each chunk gets its own launch with a token whose last occurrence is the first
position at or after n_ctx/2 (`--prompt-cache-root 0`, `--short-read` = the number of rows scored, so the first part stays on the batched path and the rest goes through the verify windows). A few positions
between n_ctx/2 and that split may go unscored; the report counts them.

## Running it

```
tools/volta/logit_identity.sh --dry-run                       # every command, nothing run (works anywhere)
tools/volta/logit_identity.sh --config strata-iq3_xxs.json --out ~/logit_identity/iq3_xxs
tools/volta/logit_identity.sh --tier 1                        # only the bit-exact rows
tools/volta/logit_identity.sh --only 2e,2f --out ... --skip-build   # again, without rebuilding
tools/volta/logit_identity.sh --ctx 8192 --chunks 3           # also exercises the QSA sparse selection in the batched path (see below)
```

Disk: the builds (llama.cpp twice, upstream once) a few GB, the reference file 4 GB, one 2 GB dump at a time. Time (estimates, not measurements): the three builds 30-60 minutes once;
tier 1 five engine runs, each a start-up plus a 33,000-token prefill; tier 2 the llama.cpp CUDA and CPU runs over 16,384 tokens (the CPU one is the slow one), then four engine launches per Strata row.
All of tier 2 needs the GGUF and about 80 GB of page cache, as the engine does; do not run it beside the server.

## Reading the output

The script ends with one line per row (`results.tsv`, `summary.txt`):

```
ROW   STATUS  SECONDS  REASON
1c    PASS        412  bitwise identical: 512 rows of 248320 float32 logits
1a    PASS        395  bitwise identical: 512 rows of 248320 float32 logits
1b    PASS        401  bitwise identical: ...
1d    INFO        398  93 of 512 rows differ; first at position 33011; 94,310 values, largest 4 ULP (|d| 1.9e-05); top-1 changed in 0 rows
2d    INFO        ...  llama.cpp CPU against CUDA, the noise floor: mean KLD 3.1e-04 +- 1.0e-05, same top-1 99.4%, ln(PPL cpu / PPL cuda) +0.00031 +- 0.00040
2e    PASS        ...  8188 rows: mean KLD 4.2e-04 (floor 3.1e-04, allowed 6.2e-04); top-1 differs 0.71% (floor 0.60%, allowed 1.200%); ln PPL +0.0004 +- 0.0006 (allowed +-0.0012)
```

(Illustrative layout, not measured numbers.) Each row's folder has the tool's own `report.txt` and `report.json`:

* **`--exact`** (tier 1): `rows differing`, `first difference` (position and vocabulary index with both values), `largest distance` in float32 ULPs (1 = neighbouring floats), `top-1 changed`. A
  report that says "printed log-probabilities (9 decimals) ... NOT a bitwise statement" means one side had no `STRATA_LOGITS_DUMP`.
* **`--ref-kld`** (tier 2): llama.cpp's own blocks (`Mean PPL(Q)`, `Mean ln(PPL(Q)/PPL(base))`, `Mean KLD`, percentiles, `Same top p`, `RMS dp`), then the floor and the allowed values, then
  `KLD VERDICT (provisional ...)`. `(partial: KL not available ...)` means the engine had no `STRATA_LOGITS_DUMP` and only the target token's log-probability was written: PPL, the paired change, top-1 and dp are checked; KL is not.
  The notes list what was approximated (positions lost to the split, the BOS handling, the pinned expert-cache size).
* **Statistics.** The "+-" are standard errors; with 8,000 positions a top-1 rate has +-0.1 % and a paired ln-PPL change a few 1e-4. A difference smaller than that is not a finding.

## If a row fails

| Symptom | Likely meaning | Next step |
|---|---|---|
| 1c FAIL | the engine is not deterministic under the pinned conditions (a race, a thread-count-dependent sum, an expert placement that moved) | read the first differing position; `--ref-env ""` runs; do not trust 1a/1b/1d |
| 1a FAIL | the port changed a number outside the two Volta kernels | the first differing position / ULP size; `tools/volta/dispatch_audit.md` lists every compute-capability decision |
| 1b FAIL | the mirror's replica or a worker's copy differs | the engine's `NUMA:` log lines; `STRATA_NUMA_MIRROR=0` |
| 2e FAIL, 2f PASS | the Volta fast paths cost quality | Gate Q's report; try `STRATA_VOLTA_ATTN=0` and `STRATA_PREFILL_F16_GEMM=0` separately (`--cand-env`) |
| 2e FAIL, 2f FAIL | the engine is further from llama.cpp than llama.cpp's backends are from each other, not because of Volta | 2g (the int8 KV cache); the same `--ref-kld` run on upstream (`--cand-exe build-logit-identity/upstream-build/strata --cand-env STRATA_PROMPT_ATTN_OLD=1`) tells whether upstream is the same distance |
| 2d FAIL | llama.cpp CPU and CUDA disagree beyond 0.05 nats: the reference is broken (bad CUDA build, out-of-memory fallback) | `tier2/llama_cuda.log` |
| 2e only "MEASURED" | no floor (row 2d missing) | run 2d |

## What this does not cover

* **The sparse selection in the batched path.** With `-c 4096` only 2,048 tokens of a chunk go through the batched prefill, and the QSA selection is the identity up to 2,051 cells. The
  `-c 8192` run (`--ctx 8192`, 2 GB per chunk) reaches it. Gate Q's 33,000-token prompt (Volta fast paths against the FP32 paths of the same engine) covers it too.
* **Only the decode path is scored.** The positions llama.cpp scores are computed by the engine's token path / verify windows; the batched prefill matters through the KV it builds for them (which is the point: that is where the Volta kernels run).
  Speculative decoding, sampling, the chat template and long-context behaviour are not part of this.
* **The int8 KV cache is part of what is measured** (the engine's default above 8K context; llama.cpp's reference is a 16-bit KV cache). Row 2g (`--kv-fp16-row`) takes it out.
* **Where experts run.** The llama.cpp CUDA run puts as much on the card as `--fit` allows; with `--cpu-moe` the experts are the CPU's in both llama.cpp runs, so the floor then reflects attention and dense math only.
* **~8,000 scored positions.** Enough to see a 1 % top-1 shift, not a 0.1 % one.

## What was checked against the real llama.cpp

`tools/volta/testdata/tiny_kld/` was produced by the **real** llama.cpp at the pinned commit (CPU build; `make_kld_fixture.py` regenerates it): a base file written by `llama-perplexity --kl-divergence-base` for a tiny random
`llama`-architecture GGUF with an odd vocabulary (121), the logits of a second, perturbed model at the scored positions of every chunk (from `llama_decode`), and the statistics `llama-perplexity --kl-divergence` printed for it.
`test_kld.py` checks that `kld_format.py` **reads that file**, that the NumPy scorer **reproduces the printed numbers** (mean KLD and its uncertainty, maximum, median and 99th percentile of KLD, same top p,
PPL of both models, ln PPL ratio and its uncertainty, RMS dp, maximum dp) from those logits, and that the NumPy writer **writes the same bytes** as the C++ writer (including the scale / min-log-prob header floats and the pad of an odd vocabulary).
The same check was made on a 601-token vocabulary with BOS substitution (SPM tokenizer) at larger scale while developing. What has *not* been run against the real llama.cpp: the Qwen model (`qwen4exp`), CUDA, split GGUFs.

## The same path for DeepSeek

`golden_compare.py --ref-kld` needs only (a) a base file and (b) an engine that can be run on a token list and write logits (`--dump-logits`) or per-position log-probabilities. It does not know the model. When the
DeepSeek-V4.1-Flash engine exists, its comparison with the llama.cpp build the user runs DeepSeek on today is the same two commands: `llama-perplexity --kl-divergence-base` from that build (the `mxxm-t` MXFP4 GGUF targets the
`mx-llama.cpp` fork; its `perplexity.cpp` must still write this format, which `kld_format.py info` and `read_base` check - magic, header, sizes, token range), a CPU-vs-GPU floor from the same fork, and `--ref-kld` on the engine. The DeepSeek vocabulary and BOS
rule differ; `--kld-bos`, `--kld-gguf` and `--kld-trim-vocab` exist for that.

## The pieces

| File | What |
|---|---|
| `tools/volta/logit_identity.sh` | the matrix: builds, runs, rows, results folder |
| `tools/volta/golden_compare.py` | `--exact`, `--ref-kld` (+ `--kld-*`), `--reuse-ref`, `--ref-exe` / `--cand-exe`, `--summarize-report`, `--print-config` |
| `tools/volta/kld_format.py` | llama.cpp's `--kl-divergence-base` reader / writer, the scoring rule, `bos` / `info` / `parse-log` / `floor-summary` |
| `src/core/verify.cpp` | `STRATA_LOGITS_DUMP=<file>`: full logits per scored position next to `STRATA_LOGPOS` (default off) |
| `tools/volta/upstream_logits_dump.patch` | the same hook for upstream's `verify.cpp` (row 1a) |
| `tools/volta/prompts/kld_text.txt`, `make_kld_text.py` | the default text and where it came from |
| `tools/volta/test_kld.py`, `test_logit_identity.py`, `testdata/` | the tests, and the fixtures made by the real llama.cpp |
