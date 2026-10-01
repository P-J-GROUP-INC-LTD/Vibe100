# Usage ledger: expert placement that learns from weeks of use

Design note (2026-10-01). Applies to both ports: Qwen3.8-Flash-Next on the V100 (`docs/volta/PLAN.md`, Phase 3)
and DeepSeek-V4.1 (`docs/deepseek/PLAN.md`, DS-2).

## What exists upstream

- **A starting profile** (`--expert-profile`, `STRP` format, `tools/make_profile.py`): every (layer, expert) pair
  ranked by routing frequency over some traces; at start the VRAM tier is filled in that order. Shipped profiles
  come from the author's prompts, not yours; a personal one needs `--dump-routing` runs and a manual rebuild.
- **A short-term adaptive tier** (`--adapt-every 4 --adapt-swaps 96`): the engine keeps decayed routing counts and,
  every 4 rounds, swaps up to 96 of the most-routed missing experts into VRAM in place of the least-routed
  resident ones. It follows the current conversation and forgets everything at exit.

## What the ledger adds

1. **Two clocks per expert.** Beside the existing short-term count, a long-term count with a half-life measured
   in routed tokens (default ≈ 2 weeks of the user's own usage, e.g. a few million tokens). It costs one float per
   (layer, expert): 24,576 for Qwen, 15,360 for DeepSeek.
2. **Persistence.** The ledger (long-term counts, token clock, measured hit rates, model fingerprint) is written
   atomically (temp file + rename) every few minutes and at exit, next to the model's config, and read at start.
   It holds expert ids and counts only — no text.
3. **Placement from it.**
   - **GPU, "core" share (default ~75 % of the slots):** the experts the long-term counts rank highest, allocated
     per layer by marginal benefit (uniform per-layer allocation beat a global ranking in published V4.1 traces).
     Set at start; refreshed slowly in the background (a few experts per minute, never a stall).
   - **GPU, "flex" share (~25 %):** the existing short-term adaptive swaps, for what the current task needs.
     The split is a setting, chosen by the measured hit rate the ledger records.
   - **CPU (everything else): split evenly across the NUMA nodes by construction.** Every non-GPU expert's rows
     are divided between the two sockets and each socket computes its half from local memory, so the commonly
     used experts *are* divided equally across the nodes — at every token, not just on average. Placing whole
     experts by frequency instead would balance only over time: on a given token the 3-5 missed experts of a
     layer often land on one socket, and that socket sets the latency. So nothing moves between nodes at run time
     (migrating pages of a 289 GB arena would cost more than it saves).
   - **RAM vs SSD** (only when the experts do not all fit in RAM — the low-RAM modes): the ledger's ranking
     decides which experts stay resident (`--resident-budget-gib` already ranks by a profile).
4. **Per-workload ledgers.** Coding and general chat use different hot sets (published V4.1 traces: overlap
   Jaccard 0.18-0.31), so the ledger file is chosen per model config (one per server / use); a ledger can also
   be exported as an `STRP` profile and shared.

### Co-activation (which experts fire together)

The ledger also keeps, per layer, how often pairs of experts are routed to the same token (a 384×384 or
512×512 count per layer, a few MB in all). With every CPU expert split in half across the sockets, co-activation
cannot unbalance them — two experts that always fire together are each computed by both sockets — so the counts
are not needed for the NUMA layout. They are kept because they predict: when expert A is routed, the experts that
usually accompany A (in this layer, and in the next) are the ones to prefetch from the SSD or promote into VRAM
first. If measurement ever shows whole experts per socket beating the split (e.g. the halves make the CPU
kernels less efficient), the same counts give the placement for that variant: a balanced max-cut per layer that
puts experts which fire together on opposite sockets.

## How fast placement can change

Promotions into VRAM travel over PCIe Gen3 (~12 GB/s): a DeepSeek MXFP4 expert (18.8 MB) takes ~1.6 ms, a Qwen
IQ expert well under 1 ms. The short-term tier already swaps in the background; the long-term core changes by a
few experts per minute, so neither ever blocks a token.

## Measuring that it works

The ledger records, per session, the GPU hit rate of the core and flex shares and the CPU's bytes per token.
`profile_decode.sh` / `--stats` print them; the acceptance test is a higher hit rate after a week of use than with
the shipped profile, on the user's own prompts.
