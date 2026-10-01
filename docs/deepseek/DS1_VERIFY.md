# DS-1 verification: the trace, the tolerances, the layer-by-layer replay, the tokenizer, the mini end-to-end fixture

Package DS1-F. Contract: `docs/deepseek/DS1.md` (§2 numerics, §6 trace, §7 gate). Oracle: `ref/ds41/`. Everything here is plain Python + NumPy
(`python3`, no torch); `jinja2` and the Hugging Face `tokenizers` package are used only to cross-check, and their tests skip when absent.

| File | What it is |
|---|---|
| `ref/ds41/trace_io.py` | the trace format: `TraceWriter` / `Trace` (the C++ engine writes the same files), `run_oracle_trace` (the oracle with tracing) |
| `tools/ds41/ds1_compare.py` | the comparison: `layers` (stage-isolated replay), `trace` (whole run), `oracle`, `logits`, `cost`; the tolerance model (`Tolerances`) |
| `tools/ds41/ds1_replay.py` | the layer-by-layer replay (the oracle's layer L on the engine's own inputs), the `--baseline` noise floor |
| `tools/ds41/ds1_noise.py` | measures the numerical noise the tolerances come from (`sums`, `flips`, `mini`, `tolerances`) |
| `tools/ds41/ds1_tokenizer.py` | the DeepSeek tokenizer + chat template from the GGUF metadata (or a `tokenizer.json`) |
| `tools/ds41/ds1_e2e.py` | the mini end-to-end fixture (`prepare`, `args`) and the checker the C++ ctest calls (`check`) |
| `tools/ds41/test_ds1_*.py`, `ds1_testlib.py` | 100+ tests (§8) |

Run the tests: `python3 -m unittest discover -s tools/ds41 -p 'test_ds1_*.py'` (or `python3 -m pytest tools/ds41/test_ds1_*.py`). About 5 minutes; the official-tokenizer
tests need `DS41_TOKENIZER_JSON=/path/to/tokenizer.json` (or the cache `ref/ds41/selfcheck.py` fills) and skip without it.

## 1. The trace

One directory per run. `trace.json` plus one little-endian `.npy` (version 1.0, C order, `<f4` or `<i4`) per (stage, layer, position):

    <stage>.p<POS>.npy            stages that exist once per position:   embed, final_hidden, logits
    <stage>.L<LL>.p<POS>.npy      stages that exist per layer:           attn_in.L03.p00017.npy

`POS` is the 0-based position (5 digits; more are accepted), `LL` the layer (2 digits). **One position per file**, not one file per stage: the engine produces one
token at a time (prefill in DS-1 is the decode path), a crashed run leaves every finished position readable, and the replay reads single positions. A `.npy` is
what `numpy.save` writes; the C++ writer needs only the 128-byte-aligned header `{'descr': '<f4', 'fortran_order': False, 'shape': (n,), }` + raw data
(`tests`: `test_files_are_plain_little_endian_npy`). Integer stages are `int32`.

`trace.json` (required: `format`, `tokens`, `quant`; the rest is documentation and cross-checks):

    {"format": "ds41-trace", "version": 1, "producer": "engine", "geometry": "MiniGeom",
     "tokens": [t_0, ... t_{n-1}],     every token of the sequence: the prompt, then the generated ones; position p's token is tokens[p]
     "n_prompt": 8,
     "quant": {"int8_act": true, "window_kv": true, "compressed_kv": true, "index": true, "linear_act": false},     the engine's flags
     "positions": [0, 1, ...]}          (written by `close()`; the reader scans the files anyway)

The engine rewrites `trace.json` after every token (the generated tokens are needed by the replay and the checker; the file is small). The oracle's trace adds
`mode`, `stale_index_k`, `compute_dtype` and a `model` block (dimensions, `layer_modes`, `compress_ratios`, ...) the tolerance model reads.

### The stages (`trace_io.STAGES` is the table the code uses)

| stage | per | shape / dtype | meaning | role |
|---|---|---|---|---|
| `embed` | pos | `[dim]` f4 | embedding row of the token (the stream before layer 0 is 4 copies of it) | input |
| `engram_out` | layer (Engram layers) | `[hc, dim]` | mHC stream after the Engram layer; the input of that layer's block | input |
| `attn_in` | layer | `[dim]` | `hc_pre(stream, pre_mix)` then `attn_norm`: what the attention reads. Also rebuilds a ratio>1 compressor's open group | state |
| `q` | layer | `[heads, head_dim]` | queries after `wq_a`, `q_norm`, `wq_b`, RoPE: what the sparse attention read | check |
| `kv_win` | layer | `[head_dim]` | the sliding-window KV row written at this position (`kv_norm`, RoPE on the tail, fp8 fake-quant when `window_kv`) | state |
| `latent_pre` | layer (FULL), optional | `[head_dim]` | the normalised group latent BEFORE RoPE and the fp4 fake-quant (published together with `latent`): separates a compressor error from a RoPE / quantiser error | check |
| `latent` | layer (FULL) | `[head_dim]` | the compressed-KV cache row published when a group of `ratio` tokens completes (position `(j+1)*ratio-1`): RMSNorm, RoPE at `j*ratio`, fp4 fake-quant when `compressed_kv` — exactly what the cache holds | state |
| `index_k` | layer (FULL) | `[index_head_dim]` | the index-K row published together with `latent`: RMSNorm of `wk(PRE-RoPE latent)`, RoPE, fp4 fake-quant when `index` | state |
| `index_scores` | layer (FULL, REINDEX), optional | `[(p+1)//ratio]` | the indexer's scores of this position over the compressed positions after the candidate mask (`-inf` = unreachable; a position is reachable once its group completed); `topk` is made from these | check |
| `block_scores` | layer (candidate source), optional | `[ceil(((p+1)//ratio)/block)]` | the candidate source's score per block (its best position; the block of the newest reachable position is `+inf`) | check |
| `topk` | layer (ratio>0, REUSE included) | `int32 [<= index_topk]` | the compressed positions this layer attends to; ascending; entries < 0 are padding the reader ignores; order irrelevant | input |
| `cand_blocks` | layer (candidate source), optional | `int32 [..]` | candidate-pool block ids (lets REINDEX layers replay with the engine's pool) | input |
| `attn_o` | layer, optional | `[heads, head_dim]` | the sparse attention's output after the inverse RoPE (the input of `wo_a`): with `attn_out` it separates an attention-core error from an output-projection error | check |
| `attn_out` | layer | `[dim]` | attention sub-layer output (after `wo_b`), before `hc_post` | check |
| `ffn_in` | layer | `[dim]` | `hc_pre(stream, attn pre)` then `ffn_norm`: what the router and the experts read | check |
| `router_idx` | layer | `int32 [top_k]` | selected routed experts (order irrelevant) | check |
| `router_w` | layer | `[top_k]` | routing weights in the order of `router_idx` | check |
| `ffn_out` | layer | `[dim]` | routed + shared expert output, before `hc_post` | check |
| `block_out` | layer | `[hc, dim]` | the mHC stream after the block: the next layer's input | input |
| `pre_mix` | layer | `[hc]` | the `pre` mix the block hands to the next block (the single-pass lag) | input |
| `final_hidden` | pos | `[dim]` | final `hc_pre` + RMSNorm, the head's input | check |
| `logits` | pos | `[vocab]` | the head's output | check |
| `router_margin`, `index_margin`, `cand_margin` | layer | `[2]` f4 | **oracle only**: (k-th minus (k+1)-th best candidate, max abs score) of a selection | oracle |

The engine writes every stage marked required in `STAGES` (all except the five optional ones above and the three oracle-only margins); an optional stage is compared when
both sides have it. **Names aligned with DS1-C's trace accessors** (`Ds41Attention<G>`): `trace_q(t)` = `q`; `trace_o(t)` = `attn_o`; `trace_latent(t)` (the PRE-RoPE normalised
group latent) = `latent_pre` — **not** `latent`, which is the cache row = `comp_kv_row(layer, index)`; `index_k_row` = `index_k`; `kv_win_row` = `kv_win`; `trace_topk(t)` = `topk`
(ascending, `-1` padding: the reader ignores it); `trace_scores(&n)` = `index_scores` (`n = (p+1)//ratio`; DS-1 decodes one token at a time, so "the last row" is the position's);
`trace_block_scores(&n)` = `block_scores`; `trace_cand(t, &nb)` flags = `cand_blocks` as the **indices of the flagged blocks** (the writer converts). **Two additions to DS1.md §6:** `index_k.L` (required;
the layer replay of every indexer layer needs it — it cannot be recovered from `latent`, which is post-RoPE and fp4) and the optional `cand_blocks.L<C0>`.

**What `Model.forward(..., trace=)` lacks and how it is captured** (`run_oracle_trace`): `attn_in`, `q`, `kv_win`, `latent`, `index_k`, `topk`, `ffn_in`,
`cand_blocks`, the optional `latent_pre` / `index_scores` / `block_scores` / `attn_o`, and the margins. `_Capture` replaces `attention.window_kv / sparse_attn / note_margin / apply_rope_tail / compressor` and `model.attention_layer / moe / router` for the duration of
one call by wrappers that call the original with the same arguments, return its result untouched, and record what they saw; the originals are restored on exit,
also after an exception. The numerics are bit-identical to an untraced run (`test_traced_run_is_bit_identical_to_the_untraced_one`). Nothing of the contract
stage list is impossible to capture. Not captured because the contract does not ask for it: the oracle's internal intermediates (`q_lora` after `q_norm`, the
indexer's `q` and weights, the compressor's group state, per-expert hidden vectors). The capture is process-global: one traced run at a time.

`run_oracle_trace(model, ids, dir, mode=)`: `token_by_token` (default; position 0 is a one-token prefill, then decode steps, what the engine does) or `prefill` (one
`forward` over all ids; equal to float round-off *without* quantisers, tested).

**Size** (`ds1_compare.py cost --real`): real model, required stages 12.69 MB per token position (`block_out` 3.3 MB, `q` 5.2 MB, `attn_in` / `attn_out` / `ffn_in` / `ffn_out`
0.82 MB each, `logits` 0.52 MB, `engram_out` 0.16 MB, `kv_win` 0.08 MB, `topk` 0.08 MB, the rest small): 406 MB for 32 tokens; with every optional stage 17.96 MB (`attn_o` is another 5.2 MB;
`index_scores` 27 KB at 1 024 positions, growing with the position). `q` is never read by the replay
(it is compared only), so an engine may skip it (7.44 MB per token) at the price of not checking the q path of layer L by itself (a wrong q still fails `attn_out`
in whole-run mode). The mini model: 0.09 MB per token.

## 2. The comparison

For every (position, layer, stage) in execution order (embed, then layer by layer in `STAGES` order, then `final_hidden`, `logits`):

* **float stages** (`float`, `kvq`, `weights`, `logits`): `rms_rel = rms(E-O)/rms(O)`, `max_rel = max|E-O|/max|O|` (each relative to the tensor's own scale), cosine.
  NaN/Inf in the engine where the oracle is finite is always a FAIL.
* **integer stages** (`topk`, `router_idx`, `cand_blocks`): exact set equality (order and `-1` padding ignored; duplicates fail). A difference is excused only as a
  **near-tie**: the oracle's recorded margin (gap between its k-th and (k+1)-th candidate) is within the stage's window times the score scale, and at most
  `max_diff_frac` of the selection differs. Why that is sound: a swap across the selection boundary needs `|s_a - s_b| <= 2 x noise`, and the gap is at most
  `|s_a - s_b|`, so a gap above the noise window proves the difference is not rounding. A near-tie is reported (`near-ties:` list) and counted as a FLIP; never a pass.
* `router_w` is compared per expert id when the two selections agree; `logits` additionally give argmax agreement and KL(oracle || engine).
* **Levels**: OK (at or below `soft`), FLIP (above soft, within `hard`), FAIL (above hard, or NaN/Inf, or a wrong selection). A stage with more FLIP samples than its
  **budget** fails ("a systematic difference, not isolated flips"). The verdict names the **first failure in execution order**: position, layer, stage.

**Two modes** (they answer different questions):

* `layers` (stage-isolated): the oracle is handed the *engine's* value of everything a stage reads and computes one stage. No error accumulates, so every stage of
  every layer and position is checked strictly. This is the check that works on the real model and the one that names the broken stage.
* `trace` (whole run): the oracle runs its own trajectory. Two correct implementations of a model with int8 activation quantisers **diverge** (§3.4): a 1e-7
  difference flips an int8 rounding decision, the next layer's quantisers turn the resulting 1e-4 into 1e-3 … 1e-2. So the first sample above float noise is reported as
  the **onset**; samples downstream of it (`downstream_of`: the rest of that token's layers, the final head, later positions of the layers it reaches — not the layers
  below it) are judged against the chaos ceiling only (finite, `rms_rel <= 4`; an expert swap gives ~1.0–1.4); everything before the onset, and the layers it cannot
  reach, stay strict. The report says how many samples were checked strictly (e.g. 64 % on the mini model, ~0 % on the real model, where the onset is in layer 0–2 of
  position 0: **use `layers` there**).

Exit status of every command: 0 pass, 1 outside tolerance (or, with `--strict`, a required stage missing), 2 the tool could not run (missing file, unreadable trace,
oracle could not load).

## 3. Tolerances

### 3.1 Where a difference between two correct implementations comes from

With the engine and the oracle fed the same inputs (stage-isolated), a stage's two outputs differ by (a) **float32 rounding in sums**, because the reduction order
differs (DS1.md §2: the engine's order is fixed but is not numpy's), and (b) the odd **rounding decision of an int8 / fp8 / fp4 quantiser inside the stage** that
fell the other way, because the quantiser's input (an intermediate of the stage) carries (a). A quantiser whose input is bit-identical on both sides (the stage's
own input, dumped by the engine) decides identically: the engine's quantiser is `quantize_int8_blocks` bit for bit (CONTRACTS.md), the oracle's `QuantConfig.int8_act`
uses the same function. So in stage-isolated mode flips can only happen at *internal* sites:

| stage | internal quantiser sites (K = elements per site) |
|---|---|
| `q` | int8 of `q_lora` (after `q_norm`) before `wq_b`: K = 1280 |
| `kv_win` | fp8 (e4m3 per 32) of the KV row when `window_kv`: K = 512 |
| `latent`, `index_k` | fp4 of the cache row when `compressed_kv` (e4m3 scales per 16) / `index` (e8m0 per 32): K = 512 / 128 |
| `attn_o`, `attn_out` | the oracle recomputes `q` from the engine's `attn_in`, so the int8 `q_lora` site (K = 1280) is inside these stages too; `attn_out` also has the int8 of the `wo_a` output before `wo_b`: K = o_groups x o_lora = 8192 |
| `index_scores`, `block_scores`, `topk`, `cand_blocks` | the indexer's `q` passes the same int8 `q_lora` site, then the fp4 fake-quant of `q` (K = 32 x 128) when `index` |
| `ffn_out` | int8 of `silu(g)*u` (times the routing weight) before `w2`, in each of the 6 routed + 1 shared expert: 7 x K = 2304 |
| all others | none (hc_pre/post, norms, router, head, embedding, Engram: float arithmetic only) |

### 3.2 Float32 summation noise (`ds1_noise.py sums`)

Relative error (rms over outputs / rms of the result) of a float32 dot product of K terms against float64, standard-normal data:

| K | naive left-to-right | lane-strided (32 partial sums + tree) | numpy BLAS |
|---:|---:|---:|---:|
| 256 | 2.3e-7 | 8.3e-8 | 1.1e-7 |
| 1280 | 6.2e-7 | 1.4e-7 | 2.3e-7 |
| 2304 | 8.4e-7 | 1.8e-7 | 3.3e-7 |
| 5120 | 1.4e-6 | 2.7e-7 | 3.9e-7 |
| 8192 | 1.6e-6 | 2.7e-7 | 4.3e-7 |
| 20480 | 2.7e-6 | 5.0e-7 | 4.5e-7 |

The naive sum grows as `1.8e-8 sqrt(K)`; the engine's lane-strided sums and numpy's BLAS are 3–6x better. The soft tolerance is set from the **naive** sum (the worst
reasonable engine) with a factor 2.2: `soft_rms = max(2e-6, 4e-8 sqrt(K))`, K the longest float32 sum feeding the stage (`dim` for the norm/hc stages and the
BF16 router/compressor GEMVs, `hc*dim` = 20480 for `pre_mix`, `heads*head_dim/o_groups` = 4096 for `wo_a`, ...); `soft_max = 2.5 x soft_rms` (measured
max/rms of a sum error: 1.2–1.6); the indexer's scores (`index_scores`, `block_scores`: sums of terms of both signs, they cancel) get 6x that (measured 8.5e-7 / 1.4e-6 rms on the mini model). The **float32 oracle against the float64 oracle**, stage-isolated, on three mini models x 64 tokens x three quantiser presets
(`ds1_noise.py mini`, 28 000 stage samples): the worst stage error without a flip is **5.2e-7 rms** (`attn_out`), 1e-7 .. 4e-7 for most stages, i.e. the soft tolerance
(2e-6 on the mini model) sits 5x (`attn_out`) to 20x above the measured noise and 1.4x–4x above even the naive-sum worst case.

### 3.3 Quantiser flips (`ds1_noise.py flips`)

Monte Carlo with the engine's quantisers on Gaussian (and 1 %-outlier "heavy") activations perturbed by relative noise eps, the float noise of the stage that made them:

* **int8 per 32** (`quantize_int8_blocks`): flips per element `1.2e-5` at eps = 3e-7, `4.0e-5` at 1e-6 (linear in eps); the model uses `FLIP_RATE = 2.2e-5`
  per element (eps ~ 5.5e-7). A flip changes the following GEMV output by `rms_rel = 1.9e-2 / sqrt(K)` (measured 0.0172–0.0185 x 1/sqrt(K), K = 256 … 8192; heavy
  tails 0.013); the p99.9 over vectors is at most 4x the single flip (1.4e-3 at K = 256, 1.1e-3 at K = 8192).
* **fp8 e4m3 / fp4 e2m1 cache rows**: per element `2e-6 … 3e-5`; one flip moves the row by `0.25 / sqrt(K)` (fp8), `0.65 / sqrt(K)` (fp4 e4m3 scales), `0.45 / sqrt(K)` (fp4
  e8m0) of its rms, and one element by up to 8 % / 19 % / 12 % of the row's max.

**Cross-check from DS1-C** (the C++ attention against the oracle, MiniGeom, int8 activations, KV fake-quant on and off, 4 800 layer-steps): the only deviations were quantiser rounding flips,
8 events (1 fp8 flip in a SWA KV row, 7 int8-code flips), each 2.5e-3 … 6.2e-3 relative to the row max in `q`, `o` or the layer output, everything else to ~2e-6 with top-k identical and block scores
exact: 1 per ~800 layer-steps. The stage-isolated oracle-against-oracle measurement of this package agrees: the attention stages (`q`, `attn_o`, `attn_out`) flip in 0.1 – 0.2 % of samples with
sizes 2.6e-3 … 6.7e-3. Both are classified FLIP (reported with position, layer and stage under "quantiser flips", counted against a budget with a floor of 3 per stage), not FAIL. DS1-C re-syncs
the engine's cache row to the oracle's after a flip so trajectories do not diverge; **the replay needs no re-sync**: the state the oracle replays from IS the engine's dumped rows.

The consequences for the tolerance model (class `Tolerances`, constants at the top of its section in `ds1_compare.py`):

* `hard_rms` of a stage with int8 sites = `max(5e-3, 8 x 1.9e-2 / sqrt(K_min))` (eight single flips); with a fp8/fp4 row = `2 x KV_FLIP_RMS / sqrt(K)`; a stage with no quantiser
  inside has no excuse above `hard = 50 x soft`. `hard_max = 2 x hard_rms` (rows: the per-element figures above x 2).
* **budget** (fraction of a stage's samples allowed in the FLIP band, and never fewer than 3 samples per stage) = `0.02 + 2 x (1 - exp(-lambda))`, `lambda = 2.2e-5 x (sum of the stage's
  int8 sites' K) + 1e-5 x K_row` flips per sample, capped at 0.9. The mini model's `ffn_out` has `lambda = 0.017` (budget 5 %); the real one `lambda = 0.36` (budget 62 %): at K = 2304 x 7, **a third
  of all `ffn_out` samples legitimately contain a flip**.
* A near-tie window of `2e-5` of the score scale for the router (float noise only); for the indexer's top-k / candidate blocks the largest of `2e-5`, `4 x 1.9e-2 / sqrt(q_lora)` when
  int8 activations are on (one flip of the shared int8 `q_lora` site: 9.5e-3 on the mini model, 2.1e-3 real) and `5e-2` when `index` is on (a flip in the fp4 fake-quant of the indexer's
  `q` moves its scores by 1–4 % of their scale), with at most 10 % (top-k) / 20 % (router) / 25 % (candidate blocks) of the selection differing.

Measured on the mini model (3 seeds x 64 tokens, stage-isolated, float64 oracle replaying the float32 oracle's inputs, `ds1_noise.py mini`; 6 016 samples per run):
every stage within the tolerances in all 9 runs (0 FAIL); flips: `int8-kv` 3 per run (q 1, attn_out 2, ffn_out 6 over the three seeds), `int8` 2–5, `exact` 0. Worst flip
`attn_out` 6.7e-3 (hard 1.9e-2), `ffn_out` 3.0e-3 (hard 9.5e-3), `q` 2.7e-3 (hard 1.9e-2): 2.9x–7x below hard. The measured flip frequency of `ffn_out` (0.4 % of samples) is the
model's prediction at the oracle's own noise (eps ~ 1.5e-7).

### 3.4 Why the whole run diverges (and what the "64 tokens identical" gate can mean)

On the mini model, float32 and float64 oracle, int8-kv, whole run: the first deviation above float noise appears after 16–28 positions (1 flip per ~3 000 stage
samples); once there, the next layer's quantisers amplify it (measured at position 2 of seed 0: `ffn_out` 3.2e-4 → `attn_in` of the next layer 1.1e-4 → `q` 3.4e-3 →
`attn_out` 3.7e-3 → `ffn_out` 7.5e-3), saturating at the quantiser noise floor (~1 %). 48 tokens later `final_hidden` differs by 27 % rms (an expert swapped) and top-1
agreement is 43/48. The same run with `exact` (no quantisers) agrees to 1e-7 throughout. This is a property of int8-activation inference, not of this implementation,
and it is why the strict check is stage-isolated. For the gate it means: **"greedy continuations identical" can only be required up to the first numerical deviation**;
`ds1_e2e.py check` requires the engine's tokens to equal the oracle's unless the first difference (a) is a near-tie of the oracle's logits (gap within twice the logits
tolerance), or (b) comes at or after a reported onset; a difference with no earlier deviation is a bug (the logits agreed to 1e-6 and the argmax still differed).
The shipped fixture: a float64 oracle decoding greedily on its own tokens diverges from the float32 oracle at token 41 of 64, 13 positions after its first onset (position 28).

### 3.5 The tables

Mini model (the C++ MiniGeom end-to-end test; `ds1_noise.py tolerances`), flags `int8_act + window_kv + compressed_kv + index`:

| stage | soft rms | soft max | hard rms | hard max | flips/sample | budget |
|---|---:|---:|---:|---:|---:|---:|
| embed | 1e-12 | 1e-12 | 1e-9 | 1e-9 | 0 | 0 |
| engram_out, attn_in, latent_pre, ffn_in, router_w, block_out, pre_mix, final_hidden, logits | 2.0e-6 | 5.0e-6 | 1.0e-4 | 2.5e-4 | 0 | 0.02 |
| q, attn_o | 2.0e-6 | 5.0e-6 | 1.9e-2 | 3.8e-2 | 0.001 | 0.02 |
| kv_win | 2.0e-6 | 5.0e-6 | 6.2e-2 | 2.0e-1 | 0.001 | 0.02 |
| latent | 2.0e-6 | 5.0e-6 | 1.6e-1 | 5.0e-1 | 0.001 | 0.02 |
| index_k | 2.0e-6 | 5.0e-6 | 1.6e-1 | 3.0e-1 | 0.000 | 0.02 |
| index_scores, block_scores | 1.2e-5 | 3.0e-5 | 8.0e-2 | 3.0e-1 | 0.003 | 0.03 |
| attn_out | 2.0e-6 | 5.0e-6 | 1.9e-2 | 3.8e-2 | 0.003 | 0.03 |
| ffn_out | 2.0e-6 | 5.0e-6 | 9.5e-3 | 1.9e-2 | 0.017 | 0.05 |
| router_idx | near-tie window 2e-5 of the score scale; <= 20 % of the selection may differ; budget 0.02 | | | | | |
| topk, cand_blocks | near-tie window 5e-2 of the score scale (fp4 index flips); <= 10 % / 25 %; budget 0.05 | | | | | |

Real model (`ds1_noise.py tolerances --real`; the tolerance object is built from the oracle trace's `model` block, so a real-model comparison uses these automatically):

| stage | soft rms | soft max | hard rms | hard max | flips/sample | budget |
|---|---:|---:|---:|---:|---:|---:|
| embed | 1e-12 | 1e-12 | 1e-9 | 1e-9 | 0 | 0 |
| engram_out, attn_in, latent_pre, ffn_in, router_w, final_hidden, logits | 2.9e-6 | 7.2e-6 | 1.4e-4 | 3.6e-4 | 0 | 0.02 |
| block_out | 2.0e-6 | 5.0e-6 | 1.0e-4 | 2.5e-4 | 0 | 0.02 |
| pre_mix | 5.7e-6 | 1.4e-5 | 2.9e-4 | 7.2e-4 | 0 | 0.02 |
| q, attn_o | 2.0e-6 | 5.0e-6 | 5.0e-3 | 1.0e-2 | 0.028 | 0.08 |
| kv_win | 2.0e-6 | 5.0e-6 | 2.2e-2 | 2.0e-1 | 0.005 | 0.03 |
| latent | 2.9e-6 | 7.2e-6 | 5.7e-2 | 5.0e-1 | 0.005 | 0.03 |
| index_k | 2.9e-6 | 7.2e-6 | 8.0e-2 | 3.0e-1 | 0.001 | 0.02 |
| index_scores, block_scores | 1.7e-5 | 4.3e-5 | 1.4e-2 | 3.0e-1 | 0.069 | 0.15 |
| attn_out | 2.6e-6 | 6.4e-6 | 5.0e-3 | 1.0e-2 | 0.208 | 0.40 |
| ffn_out | 2.0e-6 | 5.0e-6 | 5.0e-3 | 1.0e-2 | 0.355 | 0.62 |
| router_idx / topk / cand_blocks | as above (the top-k window on the real model: 5e-2, set by the fp4 index flips) | | | | | |

With the KV flags off (`QuantConfig.int8`) the `kv_win`, `latent`, `index_k` rows have no flip site (hard = 50 x soft) and the indexer's near-tie window is 2e-5.
`--tol-scale F` multiplies the float values (a loosening you must justify in the log; it does not touch selections or budgets).

### 3.6 The real model: a prior, and the oracle's own noise floor

The real-dimension rows are extrapolated from the measured rates (the real model is only on the owner's box), so they are a prior. `ds1_compare.py layers ... --baseline`
replaces the prior by a measurement on the real data: it replays the same engine inputs with the oracle in float32 **and** in float64 and prints the second report (the
oracle against itself, per stage, flips included) next to the engine's, plus the two fractions of samples above soft. An engine should look like that report.
If the real-model budget of a stage is violated while the baseline's flip fraction is just as high, the budget prior was too tight: say so in `BOX_LOG.md` with the
numbers; do not use `--tol-scale`.

### 3.7 What the tolerances do not catch

A systematic defect of the size of the flip band (a 1e-3 relative error in every sample of a stage) is caught only through the budget (tested: `q` at every position
fails, a single sample does not). A rounding-rule defect of the quantiser (round-half-away instead of ties-to-even) is invisible here (it differs only on exact
ties) and is the business of the kernel parity programs (DS-D, DS1-G). `index_k` / `latent` flips of the fp4 rows are rare (1e-3 per sample) and large (up to 19 %
of one element); one such sample passes. The indexer's near-tie window (5 % of the score scale) is wide by necessity; an indexer bug shows as many differing
selections, i.e. over the 10 % bound or over the budget.

## 4. Layer-by-layer replay for the real model

`ds1_compare.py layers --engine DIR --gguf SHARD1 [--layers 0-39,head] [--positions 0,7,15,31] [--baseline]`. The oracle runs **only layer L**, on the engine's own
input to layer L, for each requested (layer, position); one layer's tensors plus the routed experts that token routes to are in RAM at a time.

### 4.1 What the oracle computes from what (all engine values; `ds1_replay.py`)

| stage | oracle computes it from | compared with |
|---|---|---|
| `embed` | the token id | `embed` |
| `engram_out` | `block_out.L-1` (or the embedded token), the token ids → hashed Engram rows | `engram_out.L` |
| `attn_in` | the stream entering the block (`engram_out.L` / `block_out.L-1`) and `pre_mix.L-1` | `attn_in.L` |
| `q`, `kv_win`, `latent`, `index_k`, `topk`, `attn_out` | `attn_in.L` (the engine's), the layer's cache rows of earlier positions (`kv_win`, `latent`, `index_k` of the engine), the compressor's open group (`attn_in` of the earlier tokens), the engine's `topk` for REUSE layers, the candidate pool for REINDEX layers | the engine's own |
| `ffn_in` | the stream, the engine's `attn_out.L`, the attention mHC mixes | `ffn_in.L` |
| `router_idx`, `router_w`, `ffn_out` | the engine's `ffn_in.L` (the oracle's router, routed + shared experts) | the engine's own |
| `block_out`, `pre_mix` | the engine's `ffn_out.L`, the stream after attention, the FFN mHC mixes | the engine's own |
| `final_hidden`, `logits` (pseudo-layer `head`) | `block_out.N-1`, `pre_mix.N-1`; the head on the engine's `final_hidden` | the engine's own |

An error in the engine's layer 7 q-projection shows up at `q` of layer 7 and nowhere else (`test_layer_mode_isolates_the_error`). When a selection (top-k, routed experts)
differs from the engine's, it is judged against the oracle's margin (§2) and everything downstream is recomputed a second time with the **engine's** selection
(`--no-force` turns that off), so one near-tie does not hide the numbers of the stages after it. The orchestration repeats `Model.block` / `forward` call for call using the
oracle's own functions: fed the oracle's own trace as the engine, every stage of every layer and position of the replay equals the full run **bit for bit**
(`test_every_layer_and_position_bit_exact`, 1 840 stage arrays).

### 4.2 What the engine must dump: the per-position trace of §1, no cache snapshots

The replay protocol is the trace itself: the engine writes each cache row once, when it produces it, and the oracle rebuilds the state of layer L at position p from the
rows of earlier positions. The files layer L at position p reads (`needed_files` in `test_ds1_replay.py`; the test replays from **exactly** these files and shows each is
necessary):

* the inputs: `embed.p` (L = 0) or `block_out.p.L-1` and `pre_mix.p.L-1`; `engram_out.p.L` on an Engram layer;
* the layer's own state: `attn_in.p.L`; `kv_win.q.L` for the `window-1` positions before p (the sliding window ring); on a FULL layer the owner's `latent.q` and
  `index_k.q` for every published group before p (`(j+1)*ratio-1`), and for a ratio > 1 compressor `attn_in.q.L` of the earlier tokens of the open group; on a REUSE layer
  `topk.p.L`; on a REINDEX layer the owner's `latent` / `index_k` rows up to and including p and `cand_blocks.p.<candidate source>` (without it the oracle re-runs the candidate source's indexer from that layer's own
  state: more files, same answer, tested);
* the values it is compared with: `attn_out`, `ffn_in`, `ffn_out`, `router_idx`, `router_w`, `block_out`, `pre_mix`, `kv_win.p`, `q.p`, and `latent.p`/`index_k.p`/`topk.p` where they exist.

So a replay of any subset of (layer, position) needs the trace of **all earlier positions' state stages** and of the checked positions' other stages. A missing file is reported as
`cannot replay ... the engine trace has no <file>` and the run fails (a layer that could not be checked is not a pass).

### 4.3 Cost

* **The engine's dump**: 12.7 MB per token (7.4 MB without `q`); 406 MB for the 32 tokens of the gate; GPU → host copies of `block_out` / `attn_in` / `ffn_out`-sized
  arrays per layer (a few 100 µs each) — a debugging mode, not a production one. The stages of §1 are the whole interface (`--trace DIR`, DS1-E).
* **The oracle** (single process, numpy; measured on a 4-core cloud container, the box will differ): decoding is the cost. MXFP4 → float32 45 Mvalues/s (one routed expert,
  35.4 M values = 0.79 s), Q8_0 → float32 63 Mvalues/s (`wq_b` 42 M values = 0.67 s), a float32 GEMV of an expert matrix 11 ms. Per layer the dense weights (attention, shared
  expert, ~170 M values, ~680 MB as float32) are decoded **once** (the replay runs layer-major and the weight cache holds a few layers: `--cache-gib 4`) ≈ 3 s; the routed experts
  are decoded when a token routes to them (`--expert-cache 48` keeps 48 decoded experts = 6.8 GB): 32 positions x 6 draws from 384 experts ≈ 150 distinct experts ≈ 2 min per layer.
  **About 80 minutes for 32 positions x 40 layers (about 2 minutes per layer, 20 minutes for 8 positions)**; the head (one batched matmul over all positions, 129 280 x 5120 BF16) a
  few seconds. Peak RAM about 12 GB. `--positions` and `--layers` select a subset (`--layers 0,1,2,14,20,24,39,head`: one layer of each role).
* **A whole-run oracle comparison on the real model** (`ds1_compare.py trace --gguf`) costs the same per token as the replay of all 40 layers (~2–3 min per token): 500 tokens
  is about a day. That is the limit of the "top-1 over 500 tokens against the oracle" half of Gate DS-1 on this oracle; a process pool over the six experts of a token would give
  about 6x and is not done here. The 32-token layer-by-layer half is the one that localises a defect; the 500-token half only counts argmax agreement
  (`ds1_compare.py logits --engine ENGINE.bin --oracle ORACLE.bin [--near-tie-gap G] [--first N]`, NaN/Inf rows fail, `--min-top1 0.99`).

### 4.4 Procedure on the box

1. Tokens: `ds1_tokenizer.py chat --gguf SHARD1 --messages @msgs.json --ids` (or `encode`).
2. `strata-ds41 --gguf SHARD1 --tokens <ids> --max-new 8 --trace DIR` (the real prompt, 32 positions).
3. `ds1_compare.py layers --engine DIR --gguf SHARD1 --positions 0,3,15,31 --layers 0,1,2,14,20,24,39,head --baseline` (about 10 minutes), then all layers at the
   positions that matter; the first failure is printed as `FIRST FAILURE: position P, layer L, stage S`.
4. Log the numbers in `docs/volta/BOX_LOG.md`.

## 5. The tokenizer (`tools/ds41/ds1_tokenizer.py`)

Built from `tokenizer.ggml.*`: `model = gpt2` (byte-level BPE), `pre = joyai-llm`, `tokens` (129 280), `token_type` (3 control, 4 user-defined: both matched literally in the
text), `merges` (127 741, `"a b"`), `bos 0`, `eos 1`, `padding 1`, `add_bos/eos = false`, `chat_template`. The pre-tokenizer is the three regexes of the official tokenizer.json
(`\p{N}{1,3}`; `[一-龥぀-ゟ゠-ヿ]+`; the letters / punctuation / newline / whitespace pattern), applied as Hugging Face `Split(Isolated)` in sequence, then the GPT-2 byte alphabet and
BPE (lowest merge rank first, leftmost on ties, heap-based so long runs are not quadratic). `\s` is Unicode White_Space exactly (Python's own `\s` also matches U+001C–1F).
Decoding maps byte-level characters back to bytes, special tokens contribute their literal text, and the whole byte string is decoded as UTF-8 once with replacement
characters. `parse_special=False` turns literal special-token parsing off; `add_bos` / `add_eos` default to the GGUF's flags (the chat template writes the BOS itself).
`ds1_tokenizer.py {encode,decode,chat,info,selfcheck}`.

**Status: verified token for token against the official `tokenizer.json` of deepseek-ai/DeepSeek-V4.1-Flash** (obtained through the network; `DS41_TOKENIZER_JSON` or the
`ref/ds41/selfcheck.py` cache): GGUF-style metadata derived from it (tokens, types, merges, ids, regexes) equals what the code builds; `encode` equals the `tokenizers` library on
86 000 random and special-case texts (including all 1.1 M code points, 40 000 of them swept in context) and on 1.4 M characters of this repository's sources and docs; `decode` equals
`tokenizers.decode`; the known ids (`<｜User｜>` 128803, `<｜Assistant｜>` 128804, `<think>` 128821, `</think>` 128822, `<｜end▁of▁sentence｜>` 1, ...) match; the sizes match the vendored GGUF
header dump (129 280 tokens, 127 741 merges, `pre = joyai-llm`). **The real GGUF's own token arrays are not in the vendored headers**: `ds1_tokenizer.py selfcheck --gguf SHARD1
--tokenizer-json tokenizer.json` (run it on the box once) compares them with the official file entry by entry (token strings, types, merges, ids) and reports every difference.
Two facts worth knowing: the GGUF's `padding_token_id` is 1 while the HF pad token `<｜▁pad▁｜>` is id 2 (the Engram pad id, `deepseek41.engram.pad_token_id`); and the GGUF
metadata does not say which added tokens are control and which user-defined (the converter's rule — `special` → control — is assumed; both are matched literally, so it does not
change encoding).

**One known limit, handled**: Python 3.11's `unicodedata` is Unicode 14, the official tokenizer's tables are newer. `UNICODE_PATCH` (89 ranges, 10 294 code points, all unassigned
in Unicode 14: new letters, symbols such as the Unicode 15–16 emoji, digits) records how the official pre-tokenizer groups them; it was found by probing every code point with the
official pre-tokenizer and `test_unicode_patch_is_complete` re-derives it (8 s). With it the two agree on every code point probed. A later Unicode version would need the probe re-run
(`probe_unicode_patch`). The optional `regex` engine (`Ds41Tokenizer(..., engine="regex")`) has newer tables than the official one and differs on code points assigned after Unicode 15/16.

**Chat template**: `apply_chat_template(messages, add_generation_prompt, thinking, tools, response_format, reasoning_effort, drop_thinking)` renders the DeepSeek-V4 template
(thinking mode, DSML tool calls, `<tool_result>`, the reasoning-effort preamble). The native Python renderer is checked against jinja2 rendering the exact template string from the
GGUF header on 960 combinations (10 conversations x generation prompt x thinking x tools x drop_thinking x effort x response format): identical. It is used only when the
GGUF's template hash is the known one; any other template needs jinja2 (`engine="jinja"`, HF-style `tojson` without escaping, plus `from_json`). `encode_chat` = template + `encode`.

## 6. The mini end-to-end fixture and the checker (`tools/ds41/ds1_e2e.py`)

    python3 tools/ds41/ds1_e2e.py prepare --out $BUILD/ds1_e2e                      # exit 0; ~10 s the first time, instant after (idempotent)
    ARGS=$(python3 tools/ds41/ds1_e2e.py args --fixture $BUILD/ds1_e2e)            # "--gguf FILE --tokens a,b,... --max-new 56"
    <engine> $ARGS --trace $BUILD/ds1_e2e/engine                                    # MiniGeom, emulated; the trace of §1
    python3 tools/ds41/ds1_e2e.py check --fixture $BUILD/ds1_e2e --engine $BUILD/ds1_e2e/engine [--json R.json] [-v]    # exit 0 pass / 1 fail / 2 could not run

`prepare` writes `gguf/` (the mini model of `make_mini_gguf.py`, seed 0, three shards — **byte-identical on every run**, tested), `oracle/` (the oracle's trace over the prompt plus its
greedy continuation: 8 + 56 = 64 positions, flags `int8_act + window_kv + compressed_kv + index`, the engine default) and `fixture.json` (seed, GGUF path, prompt ids, the oracle's tokens,
flags). The prompt is chosen deterministically so that the oracle's continuation never contains EOS (id 1).

`check` decides, and prints PASS/FAIL per criterion plus the first failure as `FIRST FAILURE: position P, layer L, stage S`:

1. **the trace is complete**: the engine's tokens start with the prompt; it generated the requested number of tokens (fewer is accepted only if it ended on EOS); a file for every
   position 0 … n-2 (the last generated token has no forward pass); every required stage of the oracle's trace present (`strict`);
2. **layer mode** (`ds1_replay.replay_compare`, all positions, all 8 layers, the head): every sample within its tolerance and budget, no sample that could not be replayed;
3. **whole-run mode** (`ds1_compare.compare_traces`, the oracle run on the *engine's* tokens when they differ from the fixture's): strict before the first deviation, chaos ceiling after it,
   no NaN/Inf; info line with top-1 agreement and KL;
4. **greedy**: every generated token is the argmax of the engine's own logits at the previous position (exact: a sampler / argmax defect fails here), and the continuation equals the oracle's
   greedy continuation unless the first difference is a near-tie of the oracle's logits or comes at/after a reported onset (§3.4).

Exit 2 only for a problem of the checker itself: no/unreadable fixture, its GGUF missing, the oracle cannot load it, numpy missing. An engine that wrote no trace is a **failure** (1).
In CMake: `set_tests_properties(ds41_e2e_mini PROPERTIES SKIP_RETURN_CODE 2)` turns exit 2 into a skipped test. Needs `python3` with `numpy` only; ~10 s for `check`.

Tested with an independent implementation standing in for the engine (a float64 oracle decoding greedily): passes; with an error of 40 % rms injected in one of
`attn_out`, `ffn_out`, `kv_win`, `latent`, `logits`: exit 1 and the named stage/layer/position; incomplete traces, a wrong prompt, a wrong token count, a token that is not the argmax of the
engine's logits, no trace, no fixture: exit 1 / 1 / 1 / 1 / 1 / 2.

## 7. Gate DS-1 against this package

| DS1.md §7 | here |
|---|---|
| mini model: every stage within its documented tolerance for 64 tokens | `ds1_e2e.py check`: layer mode strict over all 63 positions x 8 layers x every stage (§3.5 mini table), whole-run strict up to the first onset |
| mini model: greedy continuations identical | `check` criterion 4, with the near-tie / onset rule of §3.4 (the literal reading is unattainable for two independent implementations of an int8-quantised model: a float64 oracle diverges from the float32 one at token 41 of 64) |
| real model: layer by layer within tolerance for 32 tokens of a real prompt | `ds1_compare.py layers` (§4), the real-dimension table of §3.5 (+ `--baseline`) |
| real model: top-1 >= 99 % over 500 tokens against the oracle; no NaN / inf | `ds1_compare.py logits` over the engine's `--dump-logits` and the oracle's (`ds1_compare.py oracle --logits-dump`; cost §4.3: about a day on one process) |
| llama.cpp comparison as information | `tools/volta/golden_compare.py` reads the same dump format (`trace_io.write_logits_dump` writes it) |

Requests to the integrator (files DS1-F does not own): DS1.md §6 should list `index_k.L` (required) and the optional `cand_blocks.L`; §7's "identical" should read as in §3.4.

## 8. Tests

`tools/ds41/test_ds1_trace.py` (18: format, naming, round trip, truncated / corrupt files, oracle trace bit-identical to the untraced run, stage contents = the oracle's cache rows, prefill vs
token-by-token), `test_ds1_compare.py` (27: tolerance formulas, metrics, levels, near-ties, the taint rule, oracle against itself, float32 against float64 with margin >= 2 below soft,
an error injected into each of 14 stages named by stage / layer / position in both modes, layer-mode isolation, flip band vs budget, NaN/Inf, wrong selection, missing stages, the
CLI exit codes), `test_ds1_replay.py` (10: replay equals the full run bit for bit, the minimal file set and the necessity of each file, missing input is an error, REUSE layers use the
engine's selection, head, KV flags), `test_ds1_tokenizer.py` (34: pre-tokenizer pieces, BPE rank / tie rules, specials, decode, the 960-way chat-template equivalence, the GGUF header
facts, the mini GGUF, and against the official tokenizer.json: metadata, known ids, 1 500 random texts + sources, the Unicode patch), `test_ds1_e2e.py` (12: fixture, determinism, exit codes,
injections, the greedy rule). Results: see the DS1-F entry of the final report / `git log` (all pass with `pytest` in the reference venv and with `unittest` in the system Python).
