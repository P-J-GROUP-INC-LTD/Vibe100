# DeepSeek-V4.1-Flash reference (vendored, MIT)

Copied unmodified from https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash on 2026-10-01 (MIT, see
`LICENSE`): `config.json` and `inference/` (the official PyTorch + TileLang reference implementation). It is the
ground truth for the DeepSeek port (`docs/deepseek/PLAN.md`); nothing in the engine imports it.

`tensors.json.gz` is not a DeepSeek file: it is the name -> [dtype, shape, bytes] inventory of all 96,085
tensors, parsed from the 48 safetensors shard headers, so the pack tools can be tested without the 475 GiB
checkpoint.
