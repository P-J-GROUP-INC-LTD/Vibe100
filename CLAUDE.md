# Vibe100 - notes for an AI agent working in this repository

Upstream Strata 0.1.31 (MIT) ported to the NVIDIA V100 (sm_70) for a Dell Precision 7920 (2x Xeon Gold 6226, 384 GiB, one V100 32 GB),
plus the start of a DeepSeek-V4.1-Flash port. `git diff 3906943` is the whole port.

**On the owner's box, read `docs/HANDOFF.md` first**: what the port changed and where, the rules for editing, every switch back to
upstream behaviour, and a failure playbook for each runbook step. The day-one steps are `docs/volta/RUNBOOK.md`.

Rules that matter most:
- Work on your own branch (`box/qwen-v100`), not on `claude/volta-v100-conversion-8f17vp` (the cloud session develops DeepSeek there).
- Do not edit DeepSeek code (`src/ds41`, `include/strata/ds41`, `ref/ds41`, `tools/ds41`, `cmake/ds41_*`, `docs/deepseek`).
- Never skip, disable or loosen a test to get green; a numerics change needs its parity program, `golden_compare` (Gate Q) and the
  SASS audit; a change meant to be bit-identical needs `golden_compare.py --exact`.
- Prefer the environment switch back to upstream behaviour (HANDOFF section 4) over a patch while diagnosing.
- Log every result and change in `docs/volta/BOX_LOG.md`.
- Build/check: `tools/volta/run_parity.sh --ggml-dir third_party/llama.cpp --require-v100`; one file:
  `.venv/bin/python tools/volta/compile_one.py --build build-sm70 <file>`; tool tests:
  `.venv/bin/python -m unittest discover -s tools/volta`.
