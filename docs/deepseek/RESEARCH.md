# DeepSeek-V4.1-Flash — engineering spec for the port

Compiled 2026-10-01. Tags: **[PRIMARY]** read from the official HF repo files (`config.json`, `inference/*.py`,
safetensors headers, tech report) — vendored in `third_party/deepseek-v41-flash-reference/`; **[DERIVED]**
arithmetic on primary data; **[3RD]** community repos/PRs as reported (re-check before relying on them);
**UNVERIFIED** could not be confirmed.

Sources: HF = https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash · PAPER = arXiv 2609.19969 (tech report,
2026-09-17) · NEWS = https://api-docs.deepseek.com/news/news260910/

## 0. What exists

- Only `deepseek-ai/DeepSeek-V4.1-Flash` (released 2026-09-10). No V4.1-Pro/Base yet. V4-Flash/V4-Pro are
  different architectures.
- `config.json` and the full reference code are public (`inference/model.py`, `engram.py`, `kernel.py`,
  `convert.py`). Readable reference; needs torch ≥ 2.10 + TileLang + Hopper/Blackwell FP8/FP4 GEMMs — it does not
  run on a V100 as is.
- Checkpoint 510,286,023,000 B (475.25 GiB), 48 shards; `DeepseekV41ForCausalLM`, `model_type deepseek_v41`; LLM
  + ViT + DSpark head (`mtp.*`) in one repo. **License: MIT** (weights and code).

## 1. config.json (text part) [PRIMARY]

```
vocab 129280, hidden 5120, moe_intermediate 2304, layers 40, heads 64, kv_heads 1, head_dim 512,
qk_rope_head_dim 64, q_lora_rank 1280, o_lora_rank 1024, o_groups 8, silu, swiglu_limit 10.0, rms_norm_eps 1e-20,
untied embeddings, max_position 1048576, rope_theta 10000, yarn {factor 16, beta_fast 32, beta_slow 1, orig 65536},
n_routed_experts 384, n_shared_experts 1, experts_per_tok 6, scoring sqrtsoftplus, noaux_tc, norm_topk_prob,
routed_scaling 1.5, sliding_window 128,
compress_ratios [0,0, 2 x18 (L2-19), 1 x20 (L20-39), 0,0,0]   (43 = 40 layers + 3 DSpark blocks),
compress_rope_theta 160000, kv_source_layer_ids [2,8,14,20], index_source_layer_ids [2,8,14,20,24,28,32,36],
index_n_heads 32, index_head_dim 128, index_topk 512, candidate_source_layer_id 20, candidate_topk_blocks 2048,
candidate_block_size 8, hc_mult 4, hc_sinkhorn_iters 20, hc_eps 1e-6,
engram_layer_ids [1,14], engram_num_embeddings [384006168, 384016682], engram_max_ngram_size 4,
engram_vocab_size 16000000, engram_n_heads 8, engram_head_dim 256, engram_pad_token_id 2,
engram_compressed_vocab_size 99092, num_nextn_predict_layers 3, dspark_block_size 5, dspark_noise_token_id 128799,
dspark_target_layer_ids [37,38,39], dspark_markov_rank 256, dspark_n_routed_experts 128, dspark_num_experts_per_tok 3
quantization: fp8 e4m3, weight blocks 32x32, scale ue8m0, experts fp4
```

- **All 40 layers are MoE** (no dense first layers, no hash-routed layers). Layers 0-19 = causal encoder, 20-39 =
  decoder (CED, PAPER 2.2).
- **No MLA.** Attention is MQA: 64 q-heads × 512, one KV head of 512, K == V (same vector); the last 64 channels
  get RoPE (adjacent-pair rotation), the first 448 do not.
- Routing: `s = sqrt(softplus(x·Wgᵀ))` in fp32; top-6 of `s + bias` (bias selects only; weights from unbiased
  `s`); `w /= (Σw + 1e-20)`; `w *= 1.5`. Gate bf16 [384,5120]; `bias`, `bias_vl` fp32 [384] (image tokens use
  `bias_vl`). Expert = `silu(clamp(w1·x, max=10)) * clamp(w3·x, -10, 10)` then `w2`; same for the shared expert.
  Output = fp32 sum of routed + shared.
- RoPE: layers 0, 1 and DSpark use theta 10000 without YaRN; all 38 CSA2 layers use theta 160000 + YaRN for q,
  window KV and compressed KV. Softmax scale 512^-0.5. RMSNorm eps really is 1e-20 (fp32).

**Stored formats [PRIMARY]**

- Routed experts: **MXFP4** — `w1/w3.weight` I8 [2304,2560] (packed e2m1, K=5120), `w2.weight` I8 [5120,1152]
  (K=2304); scales F8_E8M0 [2304,160] / [5120,72] = one power-of-two scale per 32 K-elements. Nibble table
  `[0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6]`, element 2k = low nibble, 2k+1 = high nibble (`convert.py`), scales in a
  separate tensor. **Same codes, scales and size as GGML MXFP4, different byte layout**: GGML interleaves the scale
  with its block (17 B) and packs element j / j+16 in the low / high nibble of byte j. A GGUF is already in GGML's
  layout; converting from the safetensors permutes nibbles (`tools/ds41/ggml_codecs.py: mxfp4_from_official`).
  **One expert = 3 × (5,898,240 + 368,640) = 18,800,640 B** (checked against the shard headers).
- Attention, shared experts, indexer `wq_b`, Engram `wkv`, DSpark `main_proj`: FP8 e4m3 + E8M0 scale per 32×32
  block; activations quantised to e4m3 per 32 along K (dynamic).
- BF16: embed, head, gate, compressor `wkv/wgate`, indexer `wk/weights_proj`, norms, Engram q/k weights. F32:
  `attn_sink`, gate biases, all mHC tensors.
- Engram tables: FP8 rows of 256 + 8 E8M0 scales = **264 B/row**.
- KV is QAT'd (PAPER 2.4.4): SWA KV fp8 (block 32); global KV FP4 e2m1 with an e4m3 scale per 16 channels,
  quantised after RoPE; indexer q/k FP4. The reference fake-quantises in place, so an fp16-KV port loses the
  QAT-matched rounding (effect size UNVERIFIED — measure).

## 2. Attention [PRIMARY: model.py, kernel.py, PAPER 2.2-2.3]

```
qr = RMSNorm1280(wq_a(x)); q = wq_b(qr).view(64,512); rope(q[...,448:])
kv_win = RMSNorm512(wkv(x)); rope(kv_win[...,448:]); fp8 fake-quant         # SWA ring of 128
keys = [window kv (<=128) ; selected compressed kv (<=512)]                   # <= 640 per query at ANY context
o[h] = softmax([scores..., attn_sink[h]]) @ kv     # the sink only adds exp(sink - max) to the denominator
rope_inverse(o[...,448:]); o -> 8 groups x 4096 -> wo_a (4096->1024 per group, block-diagonal) -> 8192 -> wo_b -> 5120
```

Layer map: L0-1 SWA only · L2 Full (ratio 2), L3-7 Reuse · L8 Full, L9-13 Reuse · L14 Full, L15-19 Reuse ·
L20 Full (ratio 1, candidate-pool source), L21-23 Reuse · L24/28/32/36 Reindex (own indexer q, reuse L20's KV and
index-K, restricted to the candidate pool), each followed by 3 Reuse. Only L2/8/14/20 produce global KV; every
layer keeps its own Q and its own 128-token SWA KV.

- Compressor ratio 2: `kv = wkv(x)`, `score = wgate(x)` (512-d, fp32), groups of 2, `c = Σ softmax_group(score) ·
  kv`, RMSNorm512, RoPE at `group*ratio`, fp4 fake-quant. Ratio 1 (L20): `c = RMSNorm512(wkv(x))` with x = the
  last encoder hidden state — decoder global KV comes from the encoder (CED).
- Indexer: `index_k = RMSNorm128(wk(c))` (rope last 64, fp4); `iq = wq_b(qr).view(32,128)`;
  `w = weights_proj(x) · 128^-0.5 · 32^-0.5`; `score[t] = Σ_h relu(iq[h]·index_k[t]) · w[h]`; top-512,
  position-sorted. Candidate pool (L20): 8-position block max, newest block pinned, top 2048 blocks = 16,384
  positions; reindex layers mask outside it (no effect below 16,384 compressed positions). An entry is visible
  once its whole group is complete.
- Global KV = 890 B/token at the trained precisions [DERIVED, matches the paper]; fp16 ≈ 3.2 KB/token.
- **Reference bug (found by the port's oracle; decided):** on decode steps where a ratio-2 owner's group is incomplete (every other
  token), the official code scores against the last-published index-K cache (layer 20's, from the previous token) instead of the
  owner's own. The stale scoring happens in the three ratio-2 owners (L2, L8, L14), but their top-k indices are what the Reuse layers
  3-7, 9-13 and 15-19 consume (`model.py:722-736`), so **all 18 ratio-2 layers** are affected, not three - once a layer has more than
  512 compressed positions (context above ~1,024 tokens). The port follows the intent (each owner scores against its own cache); the
  oracle implements both. See "Decided: the index-K cache..." in `docs/deepseek/CONTRACTS.md`.

## 3. Engram [PRIMARY: engram.py, model.py; PAPER 2.4.2]

Hashed n-gram embeddings, static, indices depend only on token ids; applied on the 4-copy mHC stream just before
blocks 1 and 14.

- Token → compressed id (129,280 → 99,092) via NFKC, NFD, strip accents, lowercase, collapse whitespace, strip —
  must be replicated exactly (or shipped as a table).
- Position t uses tokens t..t-3 (pad id 2 where missing). Multipliers per Engram layer:
  `default_rng(10007*layer_id).integers(0, (INT64_MAX//99092)//2, size=4)*2+1`. Rolling XOR
  `r = tok0*m0; r ^= tok_i*m_i (i=1..3); idx = r % prime[layer][order][head] + offset`. 3 orders × 8 heads = **24
  rows per layer per token**. Primes: next primes above 15,999,999, never reused, in (layer, order, head) order;
  offsets = cumulative sums; the 24 primes of a layer sum to `engram_num_embeddings[layer]` (a self-check).
- Tables: ~384M rows × 264 B per layer; **196.61B params, 202.76 GB total**.
- Combine: `kv = wkv(concat 24 rows = 6144)` (fp8 [25600,6144]) → key[4×5120] + value[5120]; per mHC copy c:
  `rstd = rsqrt(mean(h²)+eps)·rsqrt(mean(key_c²)+eps)`, `dot = Σ h·(q_w·k_w)·key_c · rstd · 5120^-0.5`,
  `gate = sigmoid(sign(dot)·sqrt(max(|dot|,1e-6)))`, `stream[c] = h + gate·value`.
- **SSD/RAM by design**: 48 rows/token × 264 B = 12,672 B/token; addresses are known from the tokens, so rows can
  be prefetched. Same access pattern as Strata's PLE n-gram table (`src/ngram/ple_reader.cpp`).

## 4. mHC (manifold-constrained hyper-connections) [PRIMARY: model.py Block, PAPER 2.4.1]

4 residual copies [4,5120] (identical after the embedding). Per sub-layer (attn, ffn each with own params):

```
mixes = (hc_fn[24,20480] @ flatten(stream)) * rsqrt(mean(flatten²) + 1e-20)        # fp32
pre[4]  = sigmoid(mixes[0:4]*scale[0] + base[0:4]) + 1e-6
post[4] = 2*sigmoid(mixes[4:8]*scale[1] + base[4:8])
comb[4x4] = mixes[8:24]*scale[2] + base[8:24] -> row softmax + eps -> col normalise -> 19 x (row, col normalise)
x_in = Σ_c pre[c]*stream[c];  new_stream[k] = post[k]*f(x_in) + Σ_j comb[j,k]*stream[j]
```

Single-pass = a one-block lag in `pre`: block l's attention uses the `pre` from block l-1's FFN mixes; block 0 uses
pre = [1,0,0,0]. After block 39: `hc_pre(h, last ffn_pre)` → norm → head. 157 MB fp32 in total; keep it fp32.

## 5. DSpark (speculative drafting) [PRIMARY: model.py DSpark*]

`mtp.0..2`: 3 SWA-only blocks + MoE (128 routed, 3 active, 1 shared) + mHC; block size 5; conditioned on the mean
hidden state at the input of layers 37-39 (`main_proj` fp8 [5120,15360]); Markov head + confidence head.
14.2B params, 7.93 GB. Forward only — no speculative loop in the reference. [3RD] gains on offload rigs are small
(≈0 % median on new content, +12-15 % on verbatim content); a CPU-resident DSpark composes with an expert cache.

## 6. Parameters and bytes [DERIVED; tensor sum = HF total 763,205,315,794 params]

| Component | Params | Bytes |
|---|---|---|
| Routed experts 40 × 384 × 35,389,440 | 543.58B | 288.78 GB |
| Engram tables | 196.61B | 202.76 GB |
| Attention × 40 | 5.06B | 5.07 GB |
| Shared experts × 40 | 1.42B | 1.42 GB |
| embed + head (bf16) | 1.32B | 2.64 GB |
| gate / mHC / indexers + compressors / Engram proj | 0.49B | 0.71 GB |
| DSpark | 14.2B | 7.93 GB |
| Vision | 0.49B | 0.97 GB |

Decode active ≈ 16B params: routed 8.49B (6 × 40 experts) + shared 1.42B + attention 5.06B + head 0.66B + misc.
Everything per token except the routed experts ≈ 8.3B params (≈ 8.5 GB at fp8).

| Expert format | Bytes/expert | Experts in 20 GB | per layer |
|---|---|---|---|
| **MXFP4 (as released)** | 18,800,640 | 1,063 | 26.6 |
| Q4_K | 19,906,560 | 1,004 | 25.1 |
| Q8_0 | 37,601,280 | 531 | 13.3 |
| Q2_K | ~11.6 M | 1,722 | 43 |

All-miss worst case: 240 expert reads/token × 18.8 MB = 4.51 GB/token.

## 7. Expert-usage skew [3RD — DeepSeek publishes none]

- 0xBakeer (10,760 teacher-forced tokens): every layer uses ~all 384 experts; top-25 % covers 59-83 % of routed
  slots (deeper layers more skewed). Whole-model residents vs hit: 3,000 → static 0.67 / LRU 0.81; 6,000 → 0.86 /
  0.92. Coding vs general hot sets overlap only Jaccard 0.18-0.31 → adaptive caching matters. No data below 3,000.
- ik_llama.cpp #2449 (RTX 3090, 32 slots/layer): hit 0.41-0.54.
- JigSawPT (RTX 5090, 18 GiB cache ≈ 25 experts/layer): **62.7 % VRAM hit rate**.
- Estimate for ~26 experts/layer on a V100 32 GB: 0.5-0.65 with recency-aware (LRU-like) caching. **Measure.**

## 8. Ecosystem (as of 2026-10-01) [3RD]

- **llama.cpp upstream cannot run V4.1 correctly**: PR #28696 (`deepseek41` arch) is open; sparse attention and the
  candidate mask are not implemented. V4 (previous gen) is merged (#24162).
- Volta datapoints exist only for V4-Flash (8× V100, layer split, f16 KV, ~12.5 tok/s). Quantised KV gives garbage
  for this family (issue #25382; rotation path) → use f16 KV. Sparse FlashAttention is gated behind Turing MMA;
  a V100 enablement PR (#28887: 4.8-14.6× faster attention) was closed unmerged.
- V4.1 offload rigs, as REPORTED by others (not re-measured; different engines, prompts and settings): RTX 5090 + 126 GiB RAM,
  5.1 tok/s on new content; RTX 3090 + 128 GB DDR4, 6-8 tok/s. They are the two rigs of §7 (JigSawPT's RTX 5090, ik_llama.cpp #2449's
  RTX 3090), matched by hardware: check the report before quoting a figure.
- **The target box's own baseline: 4-5 tok/s**, which its owner reports measuring today on this machine (another engine; their figure,
  not measured by this port). That, not the figures above, is the number the port has to beat; `docs/deepseek/PLAN.md` §2 uses it.

## 9. Confidence

High for §1-6 (primary files; the totals reproduce HF's parameter and byte counts). Medium for everything
tagged [3RD]. UNVERIFIED: any V4.1-on-V100 measurement, fp16 range safety of V4.1 activations, hit rates at ~1,000
residents, quality of re-quantised Engram tables.

## 10. The GGUF this port targets: `mxxm-t/DeepSeek-V4.1-Flash-GGUF` (MXFP4) [PRIMARY: its shard headers]

Read 2026-10-01 from the 12 shard headers and the DSpark sidecar by HTTP range requests
(`tools/ds41_gguf_remote_headers.py`); the parsed headers are kept in
`third_party/deepseek-v41-flash-reference/gguf-headers-mxxm-t-MXFP4.json.gz` (and the vcruz305 Q2_K ones beside it).

- 375.8 GiB in 12 shards + `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf` (7.43 GiB, `general.architecture = dflash`).
  Built for the `mx-llama.cpp` fork; text only (no vision). MIT.
- `general.architecture = deepseek41`; 1,006 tensors. **Every constant the port needs is in the metadata**: the
  CSA2 layer maps (`attention.compress_ratios`, `kv_source_layer_ids`, `index_source_layer_ids`,
  `candidate_*`), mHC (`hyper_connection.count/sinkhorn_iterations/epsilon`), and Engram's `primes` (48),
  `offsets` (48), `multipliers` (8) and `token_map` (129,280 → compressed ids) — so the tokenizer normaliser does
  not have to be re-implemented, only checked against `engram.py`.
- Types (per block `N` unless stated):

| Tensor | Type | ggml dims (ne0, ne1[, ne2]) | Notes |
|---|---|---|---|
| `ffn_gate_exps` / `ffn_up_exps` | **MXFP4** | 5120, 2304, 384 | expert e = slice e: 2304 rows × 160 blocks × 17 B = 6,266,880 B |
| `ffn_down_exps` | **MXFP4** | 2304, 5120, 384 | 5120 rows × 72 blocks × 17 B = 6,266,880 B → **18,800,640 B per expert** |
| `ffn_{gate,up,down}_shexp` | Q8_0 | 5120×2304 / 2304×5120 | shared expert |
| `ffn_gate_inp` | BF16 | 5120, 384 | router; `exp_probs_b.bias` F32 [384] (no `_vl` bias: no vision) |
| `attn_q_a` / `attn_q_b` | Q8_0 | 5120×1280 / 1280×32768 | `attn_q_a_norm` F32 [1280] |
| `attn_kv` | Q8_0 | 5120×512 | `attn_kv_a_norm` F32 [512]; the SWA KV projection |
| `attn_output_a` / `attn_output_b` | Q8_0 | 4096×8192 / 8192×5120 | grouped wo_a: 8 groups × (4096 → 1024) |
| `attn_sinks` | F32 | 64 | |
| `attn_compressor_{kv,gate,norm}` | BF16/F32 | 5120×512 | kv+norm on L2/8/14/20, gate on L2/8/14 (ratio 2) |
| `indexer.attn_q_b` / `indexer.proj` | Q8_0 / BF16 | 1280×4096 / 5120×32 | on the 8 indexer layers |
| `indexer_compressor_{kv,norm}` | BF16/F32 | 512×128 | index-K from the compressed latent (L2/8/14/20) |
| `hc_{attn,ffn}_{fn,base,scale}` | F32 | 20480×24, 24, 3 | mHC |
| `engram_embed` (L1, L14) | **MXFP4** | 256 × 384,006,168 / 384,016,682 | 8 blocks × 17 B = **136 B per row**; 52.2 GB per table |
| `engram_wkv` / `engram_{q,k}` | Q8_0 / BF16 | 6144×25600 / 5120×4 | |
| `token_embd` / `output` | BF16 | 5120 × 129,280 | 1.32 GB each |

- **Format notes for the port.** The FP8 tensors were dequantised and stored as **Q8_0** (lossless to ~8 bits;
  good for Volta: dp4a kernels exist). The routed experts are a lossless re-pack of DeepSeek's FP4. **The Engram
  tables are MXFP4, not DeepSeek's FP8** — a lossy re-quantisation by the GGUF's author (effect UNVERIFIED; the
  reference oracle can measure it per row). Per token Engram reads 48 rows × 136 B.
- **vcruz305/DeepSeek-V4.1-Flash-GGUF** (Q2_K…Q8_0, no MXFP4) uses the same architecture and shapes but differs in
  names (`engram_embd`, `indexer.attn_k`/`indexer.k_norm` for `indexer_compressor_{kv,norm}`, extra
  `exp_probs_b_vl`) and metadata keys (no CSA2 layer maps; `engram.head_count/key_length/pad_id`). The pack tool
  accepts both spellings; only mxxm-t's has MXFP4 experts.

**Memory on the target box (24 × 16 GiB = 384 GiB, ~377 GiB usable):** experts 268.95 GiB + Engram 97.28 GiB =
366.2 GiB. With the experts resident and ~12 GiB for the OS, engine buffers and `token_embd`, ~96 GiB remain for
Engram's page cache — ~98 % of it (`tools/ds41/manifest.py` memory plan); with DSpark on the CPU ~91 %. The rest
is read from the SSD (the GGUF author measured NVMe ≈ 2× SATA for prompt processing).
