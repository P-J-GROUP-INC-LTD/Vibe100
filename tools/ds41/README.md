# tools/ds41 - GGUF tooling for the DeepSeek-V4.1-Flash port (work package DS-B)

Validates the MXFP4 GGUF (`mxxm-t/DeepSeek-V4.1-Flash-GGUF`, 12 shards + a DSpark sidecar, 411 GB) against the
`deepseek41` contract, writes a manifest of every tensor (shard, absolute offset, type, dims, bytes), plans the memory
of the target box, extracts routed experts as the GPU blob and the two CPU halves, and writes a tiny `deepseek41`
GGUF for loader tests. Python 3.10+ and numpy only; `tools/gguf_reader.py` is reused, not modified. The real files were
**not available while this was written**: everything was run on the saved headers
(`third_party/deepseek-v41-flash-reference/gguf-headers-*.json.gz`, which hold every tensor's name, type, dims and
offset) and on the mini GGUF. Nothing has read real expert bytes yet - see "Not verified" at the end.

| File | Role |
|---|---|
| `manifest.py` | validate + manifest JSON + memory plan (CLI, and `analyze()` for tests) |
| `ds41_spec.py` | the contract as tables: metadata keys, layer modes, expected tensors per layer, Engram constants |
| `memplan.py` | where the bytes go on a GPU box |
| `expert_layout.py` | expert `(layer, e)` -> GPU blob / CPU halves, dequantisation, the consistency checks |
| `make_mini_gguf.py` | tiny deterministic `deepseek41` GGUF (3 shards) with correct MXFP4 / Q8_0 / BF16 / F32 |
| `ggml_codecs.py` | numpy MXFP4 / Q8_0 / BF16 / F16 / F32 encoders and decoders, bit-identical to GGML's C code |
| `gguf_io.py` | shard loading (real files or headers JSON), GGUF writer with every metadata type, mmap reads |
| `ggml_c_oracle.py`, `cpu_pack_xcheck.py` | optional: GGML's own `ggml-quants.c` and DS-C's `pack_cpu_half`, compiled on the fly, as oracles |
| `test_*.py`, `fixtures.py` | the tests |

## Usage on the user's box

The 13 files are `DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf` ... `-00012-of-00012.gguf` and
`DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf`; put them in one directory (below `$GGUF`).

```bash
cd Vibe100
GGUF=/data/DeepSeek-V4.1-Flash-GGUF            # wherever the files are

# 1. validate everything, write the manifest, print the memory plan (default box: 32 GiB GPU, 384 GiB RAM, 2 nodes).
#    Naming shard 1 is enough: its 11 siblings and a *DSpark*.gguf in the same directory are picked up.
python3 tools/ds41/manifest.py $GGUF/DeepSeek-V4.1-Flash-MXFP4-00001-of-00012.gguf --out ds41.manifest.json
#    equivalently, spelled out:
python3 tools/ds41/manifest.py $GGUF/DeepSeek-V4.1-Flash-MXFP4-000{01..12}-of-00012.gguf \
        $GGUF/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --out ds41.manifest.json
#    your box, if different:  --vram-gib 31.75 --ram-gib 376 --numa-nodes 2 --cache-gib 18 --dspark-on-cpu

# 2. is expert (layer, e) laid out right?  (reads only that expert, via mmap)
python3 tools/ds41/expert_layout.py verify ds41.manifest.json --sample 8 --c-oracle     # --c-oracle needs a ggml SOURCE tree: see below
python3 tools/ds41/expert_layout.py extract ds41.manifest.json --layer 20 --expert 100 --out /tmp/experts
#    -> l20_e100.blob (18,800,640 B), .half0, .half1 (9,400,320 B each)

#    `verify --c-oracle` compiles ggml's own ggml-quants.c on the fly to compare the dequantisation bit for bit, so it needs a ggml SOURCE
#    tree and a C compiler: after ./setup.sh it is found automatically (third_party/llama.cpp/ggml/src, or the llama.cpp the engine build
#    fetched, build*/_deps/strata_llamacpp-src/ggml/src); otherwise `export STRATA_GGML_SRC=<llama.cpp>/ggml/src` (the directory holding
#    ggml-quants.c). Without one the check fails with "oracle unavailable"; leave the flag out to skip it.

# without the files (headers JSON only; offsets stay relative to each shard's data section):
python3 tools/ds41/manifest.py third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz
```

Exit status of `manifest.py`: 0 = contract satisfied (warnings allowed), 2 = REFUSED, 1 = unreadable input. Loaders must
refuse a manifest whose `"ok"` is `false` (`expert_layout.ExpertSource` does). Useful flags: `--allow-non-mxfp4`,
`--strict-metadata` (a key derived from the config.json defaults becomes an error), `--require-dspark`,
`--geometry self` (take the dims from the file itself instead of geometry.hpp: for the mini GGUF), `--no-plan`.

The other published family is refused with a clear reason, and its spellings are reported (7 shards, Q2_K/Q3_K experts):

```
$ python3 tools/ds41/manifest.py third_party/deepseek-v41-flash-reference/gguf-headers-vcruz305-Q2_K.json.gz --no-plan
  [ERROR] EXPERT_TYPE: routed experts are Q2_K/Q3_K, not MXFP4: the port's decided expert format is MXFP4 (...). Use an
          MXFP4 GGUF (mxxm-t/DeepSeek-V4.1-Flash-GGUF) or pass --allow-non-mxfp4 to validate the file anyway
  [warn ] META_DERIVED: metadata `deepseek41.attention.kv_source_layer_ids` missing; derived from the config.json default [2, 8, 14, 20]
  ... (index_source_layer_ids, candidate_*, engram.compressed_vocab_size, engram.num_embeddings from the table shapes)
  [info ] META_ALIAS: metadata `deepseek41.engram.key_length` accepted as `deepseek41.engram.head_dim`  (+ head_count, pad_id)
  [info ] TENSOR_ALIAS: blk.N.engram_embd.weight x2, blk.N.indexer.attn_k.weight x4, blk.N.indexer.k_norm.weight x4 (read as ...)
  [info ] TENSOR_IGNORED: 40 tensor(s) ignored (not used by the text-only port): exp_probs_b_vl.bias x40
RESULT: REFUSED (1 error(s)), 7 warning(s)          # exit status 2; with --allow-non-mxfp4: RESULT: OK, 8 warning(s)
```

### Manifest JSON (`--out`)

`format` (`ds41-manifest/1`), `ok`, `source`, `geometry`, `from_headers_json`, `config` (every metadata value the port
uses, by field name), `layers` (per layer: `ratio`, `mode`, `kv_owner`, `indexer`, `index_compressor`,
`compressor_gate`, `engram`), `shards` (`file`, `path`, `size`, `alignment`, `data_start`, `data_bytes`, `n_tensors`),
`tensors` (canonical name -> `shard`, `file`, `file_name` as spelled in the file, `offset` relative to the shard's data
section, `abs_offset` = `data_start + offset` or `null` from a headers JSON, `type`, `dims` in ggml order, `nbytes`,
`group`, `layer`), `aliases`, `derived_metadata`, `totals`, `sidecar` (the DSpark tensors, same fields), `memory_plan`,
`findings`. Tensors are keyed by the canonical (mxxm-t) name whatever the file spelled.

## Memory plan for the target box

From the real headers (`manifest.py <headers.json.gz>`, defaults = V100 32 GB, 384 GiB RAM, 2 NUMA nodes). Sizes are
exact from the tensor table; the reserves are the plan's numbers (flags), not measurements.

```
Memory plan: GPU 32 GiB, RAM 384 GiB, 2 NUMA node(s)
  GPU (32 GiB)
    dense weights, GPU-resident            8.33 GiB   (attn 5.01, shared 1.40, head 1.23, engram_proj 0.31, router 0.15, mhc 0.15, indexer 0.04, compressor 0.03, norm 0.00)
    KV cache, fp16, 131072 tokens x 3200 B       0.39 GiB
    activations / prompt buffers / CUDA      3.00 GiB   (--gpu-reserve-gib)
    hot-expert cache budget               20.28 GiB   -> 1158 experts of 18,800,640 B  (28.9 per layer, 7.5 % of all)
    cache size -> experts (per layer, share of all):  16 GiB -> 913 (22.8, 5.9 %)   18 GiB -> 1028 (25.7, 6.7 %)   20 GiB -> 1142 (28.6, 7.4 %)   22 GiB -> 1256 (31.4, 8.2 %)   24 GiB -> 1370 (34.2, 8.9 %)
  RAM (384 GiB, 377.1 GiB usable after 1.8 % kernel/struct-page overhead)
    routed experts, all resident          268.95 GiB   (134.47 per node: row-split halves)
    OS, services                            8.00 GiB   (--os-gib)
    engine host buffers                     3.00 GiB   (--host-buffers-gib)
    token_embd (host lookup)                1.23 GiB
    (DSpark sidecar 7.42 GiB not counted; --dspark-on-cpu adds it)
    left for Engram's page cache           95.91 GiB   of 97.28 GiB of Engram tables -> 98.6 % cached
    node 0: 192 GiB RAM, experts 134.47 GiB, 57.53 GiB free
    node 1: 192 GiB RAM, experts 134.47 GiB, 57.53 GiB free
  Engram: 48 rows/token x 136 B = 6,528 B/token (+ 4 KiB page granularity on a cache miss)
```

Reading it: routed experts 288,777,830,400 B = 15,360 x 18,800,640 B (268.95 GiB); Engram 2 tables, 384,006,168 and
384,016,682 rows of 136 B (97.28 GiB); dense weights 8,944,456,128 B (8.33 GiB, the plan's table); `token_embd`
1.23 GiB; DSpark sidecar 7,967,705,480 B (7.42 GiB). With `--dspark-on-cpu` the page cache drops to 88.5 GiB (91 % of
Engram). `--vram-gib` is nominal: a V100 32 GB reports 32,510 MiB = 31.75 GiB (1143 experts instead of 1158). Take
`--ram-gib` from `free -g` if you want the kernel overhead counted from the real figure (then pass `--kernel-overhead-pct 0`).
Dense weights count as GPU-resident (the host copy is dropped after upload); `token_embd` stays on the host
(`--embd-on-gpu` moves it). Experts are assumed copied into per-node arenas (halves), i.e. anonymous memory, not page
cache; Engram stays mmapped. The split is defined for 1 or 2 NUMA nodes.

## What each tool checks

**manifest.py** (finding codes; `error` -> exit 2)

| Code | Checks |
|---|---|
| `ARCH`, `SHARD_NONE` | `general.architecture == deepseek41` in shard 1 (the shard that holds the metadata) |
| `SHARD_COUNT`, `SHARD_TENSORS` | `split.no` / `split.count` / `split.tensors.count` against the files given |
| `SHARD_TRUNCATED` | every tensor ends inside its file (real files) - catches a partial download |
| `OFFSET_ALIGN`, `OFFSET_OVERLAP`, `OFFSET_GAP` | offsets multiples of `general.alignment` (default 32), ascending, tightly packed; absolute offset = data-section start (header end rounded up to the alignment) + tensor offset |
| `META_VALUE`, `META_MISSING`, `META_DERIVED`, `META_ALIAS`, `META_SUMMARY` | every key of `ds41_spec.KEYS` (block count, hidden, experts 384/top-6, FF 2304, YaRN, MQA 64 x 512, q/o LoRA, indexer 32 x 128 top-512, mHC 4 / 20 / 1e-6, clamps 10.0 x 40, `compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`, `candidate_*`, Engram ids / heads / head dim / n-gram / pad / compressed vocab) against `geometry.hpp` (`--geometry ds41`); an absent key falls back to the config.json default with a warning; other spellings (`engram.key_length`, `engram.head_count`, `engram.pad_id`) are accepted as aliases |
| `LAYERS_*` | the layer-mode map: ratios 0/1/2 for 40 (+3 DSpark) layers, SWA / Full / Reuse / Reindex, each Reuse/Reindex layer has a Full owner of the same ratio, candidate source is a kv source |
| `TENSOR_MISSING`, `TENSOR_UNEXPECTED`, `TENSOR_SHAPE`, `TENSOR_TYPE`, `TENSOR_SIZE`, `TENSOR_DUP` | per layer exactly the expected tensors, with the expected ggml type and dims (1,006 in total): compressor kv + norm on the 4 kv sources, **gate only on the 3 ratio-2 owners**, indexer q + proj on the 8 index sources, index-K compressor on the 4 layers that are both, Engram tables and projections on layers 1 and 14 |
| `EXPERT_TYPE` | routed experts must be MXFP4; otherwise the model is **refused** with a message (a warning with `--allow-non-mxfp4`, which also relaxes the other type checks to "any sizeable type, F32 for norms/biases") |
| `TENSOR_ALIAS`, `TENSOR_IGNORED` | accepted spellings of the vcruz305 family: `engram_embd`, `indexer.attn_k` / `indexer.k_norm` (= `indexer_compressor_kv` / `_norm`); `exp_probs_b_vl` is ignored (no vision) |
| `TOKENIZER`, `ENGRAM_META`, `ENGRAM_MULT` | tokens / token types / `engram.token_map` have vocab entries; token-map ids fit the compressed vocab (real files); Engram primes are 48 consecutive primes (first = next prime above 15,999,999), the 24 per layer sum to the table's row count, offsets are their running sums, multipliers equal `default_rng(10007 * layer).integers(...) * 2 + 1` of the reference (warning only) |
| `SIDECAR_*` | the DSpark sidecar (`dflash`, 78 tensors, 128 experts per block) - warnings unless `--require-dspark` |

**expert_layout.py `verify`**: `halves_to_blob(blob_to_halves(blob)) == blob`; the dequantised halves concatenated equal
the dequantised blob (gate/up along rows, down along columns); `y_0 + y_1 == y` on random inputs with the contract's math
in float64 (clamps are hit); with `--c-oracle` the numpy dequantisation equals GGML's `dequantize_row_mxfp4` bit for bit.
`ExpertSource` refuses a manifest without absolute offsets (headers JSON), with `ok: false`, or with non-MXFP4 experts.

## Layout facts read from the headers (for loader / engine authors)

* The shards: 1-7 hold only expert tensors (16 each, 38,503,710,720 B of data), 8 holds the rest of the experts and all
  dense tensors (886 tensors), 9 and 11 are the two Engram tables, 10 and 12 their projections. Layers 5, 10, 21, 26 and
  37 have their three expert tensors in two different shards: always go through the manifest, per tensor.
* Within a layer the file order is `gate, down, up`, **not** the blob order `[gate][up][down]`: building a blob is a
  gather of three 6,266,880 B slices (each contiguous inside its tensor; expert `e` of a tensor starts at
  `abs_offset + e * 6,266,880`), not one `memcpy`.
* The official safetensors FP4 packing is **not** bit-identical to GGML's block: byte `k` of a safetensors row holds
  elements `2k` (low) and `2k+1` (high) (`inference/convert.py:32-34`), a GGML block's byte `j` holds `j` (low) and
  `j+16` (high); the scale byte is the same. `ggml_codecs.mxfp4_from_official` / `mxfp4_to_official` convert, and a test
  proves the values survive. The GGUF's author must have applied it.

## The mini GGUF

```bash
python3 tools/ds41/make_mini_gguf.py /tmp/mini --seed 0 --n-shards 3        # or any --hidden/--ff/--n-expert/... (see --help)
python3 tools/ds41/manifest.py /tmp/mini --geometry self --no-plan
python3 tools/ds41/expert_layout.py verify /tmp/mini --geometry self --sample 5
```

Same tensor names, ggml types and the 72 metadata keys of shard 1 as mxxm-t, 218 tensors, 13.4 MB of experts: hidden 256,
8 layers (L0-1 SWA, L2 and L4 Full, L3/L5/L7 Reuse, L6 Reindex), 16 experts top-2, FF 256, Engram tables on layers 1 and 3
(6,126 and 6,332 rows of 34 B), vocab 512. Every tensor is generated from `(seed, crc32(name))` and quantised with
`ggml_codecs` (bit-identical to GGML's `quantize_row_mxfp4_ref` / `quantize_row_q8_0_ref`), so the same seed gives the
same bytes. `--alignment 64` and `--n-shards 1..n` work. **Kernels whose geometry is compiled in (`kHidden = 5120`, `kFF =
2304`, 384 experts, ...) cannot run this file**: use it for runtime-dimensioned code (loaders, the numpy oracle, these
tools), and test the fixed-geometry kernels with random blocks at the real shapes (`expert_layout.REAL`, see
`test_random_blob_at_real_shape`). The CPU kernels are runtime-dimensioned (`ExpertView`), and the default dims satisfy their `check_view`: hidden a
multiple of 128 and the expert FF a multiple of 256 (a CPU half is FF / 2 rows, in whole 4-block kernel groups; the CONTRACTS.md halves alone would need only
FF % 64), so a mini expert's blob and halves can be fed to `src/ds41/cpu`'s kernels. `MiniConfig.validate` and `expert_layout.ExpertGeom` enforce the same rule.

## DS-1 verification tools (engine vs the NumPy oracle; `docs/deepseek/DS1_VERIFY.md`)

| Tool | What it does |
|---|---|
| `ds1_tokenizer.py` | DeepSeek byte-level BPE from the GGUF's `tokenizer.ggml.*` metadata (or a `tokenizer.json`), encode / decode, chat template; `selfcheck --gguf SHARD1 --tokenizer-json tokenizer.json` compares the GGUF's own token arrays with the official tokenizer (run once on the box) |
| `ds1_compare.py` | compares an engine trace with the oracle: `trace` (whole run, onset detection) and `layers` (each stage fed the engine's own inputs: the strict check); quantiser rounding flips are counted against budgets, not failed |
| `ds1_replay.py` | the layer-by-layer replay for the real model: oracle layer L on the engine's dumped inputs, one layer's weights in RAM (`--baseline` prints the oracle's own float32-vs-float64 noise next to the engine's) |
| `ds1_noise.py` | measures the oracle's float32-vs-float64 noise per stage (the basis of the tolerance tables) |
| `ds1_e2e.py` | the mini end-to-end fixture: `prepare --out DIR` (mini GGUF + oracle trace), `args --fixture DIR` (the engine command line), `check --fixture DIR --engine TRACE_DIR` (exit 0 pass / 1 fail / 2 could not run) |

The trace format (`ref/ds41/trace_io.py`): one directory per run, `trace.json` (format, tokens, quant flags) and one little-endian
`.npy` per stage, layer and position (`<stage>.L<LL>.p<POS>.npy`). On the real model the required stages are ~12.7 MB per token.

## Tests

```bash
python3 -m unittest discover tools/ds41           # 106 tests, ~40 s (the mini GGUF's experts are 256 x 256 now)
export STRATA_GGML_SRC=<llama.cpp>/ggml/src       # directory holding ggml-quants.c: enables the bit-exact C cross-checks
```

* `test_codecs.py` - MXFP4 / Q8_0 / BF16 against hand-made blocks, error bounds, `e` = 0, 1, 255, and (with the ggml
  source and a C compiler) bit-for-bit against GGML's C encoders and decoders; official-FP4 repack.
* `test_manifest.py` - the mxxm-t headers pass with 1,006 tensors and the exact byte totals; vcruz305 is refused on the
  expert type with the alias report and passes with `--allow-non-mxfp4`; about 20 one-fault mutations of the real headers
  each name their problem; absolute offsets, alignment 64, truncated and missing shards on real (mini) files; the
  memory plan; the defaults pinned to `config.json` and to `geometry.hpp`.
* `test_mini_gguf.py` - the mini file read back with `tools/gguf_reader.py` (types, dims, metadata, offsets, bytes ==
  an independent re-encoding), determinism, same key set and tensor-name/type set as mxxm-t, writer round trips for every
  metadata type at alignments 32/64/128.
* `test_quant_xcheck.py` - the activation quantiser of CONTRACTS.md: the oracle's `quantize_int8_blocks` against DS-C's C++ kernels (scalar / AVX2 /
  AVX-512, compiled on the fly by `cpu_pack_xcheck.py`) on hand-written special blocks (all-zero, 1e-37, the 2^-100 boundary, ties to even, Inf / NaN at every
  position) and random bit patterns: identical bytes (int8 values and scale bits). The GPU kernel is compared with the same C++ library in `ds41_cuda_emu_test --quant`.
* `test_expert_layout.py` - blob = the three GGUF slices (cut out with the reference reader), halves re-derived with
  plain loops, every mini expert verified, wrong halves detected, the file against the codec-only quantised weights,
  real-shape random blob, DS-C's C++ `pack_cpu_half` byte-identical to the Python halves, and the mini GGUF's experts (whole
  and as two CPU halves, every ISA) run through the C++ CPU kernels at their runtime dimensions and match the float64 contract (all
  skipped when g++ or the DS-C sources are absent); `ExpertGeom` rejects dimensions the kernels' `check_view` would abort on.

## Not verified / limits

* No real shard was read: the real-file code paths (absolute offsets, truncation check, mmap, halves of real experts) are
  exercised on the mini GGUF only. First thing to run on the box: `manifest.py` on the shards, then `expert_layout.py
  verify --sample 8 --c-oracle`.
* That the GGUF's expert nibbles follow GGML's order (rather than the official pairing) cannot be seen in the headers; the
  dequantised experts should be compared once against the official safetensors with `mxfp4_from_official`.
* From a headers JSON the data-section start of a shard is unknown (the JSON has neither the header length nor the
  tokenizer array contents), so only relative offsets and sizes are available there.
* Dense types other than the mxxm-t set (F16 attention, ...) are reported as `TENSOR_TYPE`; the loader decides whether it
  can read them. `expert_gating_func = 4` is checked for equality only (its meaning lives in the mx-llama.cpp fork).
