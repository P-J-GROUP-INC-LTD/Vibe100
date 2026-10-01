#!/usr/bin/env python3
"""tools/volta/testdata/make_kld_fixture.py - regenerate the fixtures of test_kld.py with the REAL llama.cpp (the commit setup.py pins).

What the tests need is a base file written by llama.cpp itself, the real `--kl-divergence` printout for a second model against it, and that
second model's logits - so the NumPy reader / scorer in kld_format.py can be checked against the C++ ones, bit-format and numbers.  Everything
is tiny (a 2-layer, 64-wide `llama` model with random F32 weights and an ODD vocabulary of 121 pieces, so the pad uint16 of a row is exercised;
n_ctx 64, 4 chunks: 124 scored positions, just past the 100 below which llama.cpp prints no statistics), about 160 KB in all.

    llama.cpp CPU build (no GPU needed), from third_party/llama.cpp or any checkout of the pinned commit:
        cmake -S third_party/llama.cpp -B /tmp/llcpu -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \\
              -DLLAMA_BUILD_EXAMPLES=OFF && cmake --build /tmp/llcpu -j --target llama-perplexity llama
    python3 tools/volta/testdata/make_kld_fixture.py --llama-src third_party/llama.cpp --llama-build /tmp/llcpu --out tools/volta/testdata/tiny_kld

Writes  base.kld  (llama-perplexity --kl-divergence-base, model A),  base.logits.bin / q.logits.bin  (model A's / B's logits at the scored
positions of every chunk, the engine's --dump-logits layout, from llama.cpp's own llama_decode),  q_kl.txt  (the statistics blocks of
llama-perplexity --kl-divergence for model B against base.kld) and  text.txt.  B is A with its weights perturbed by 5 % noise.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

DUMPER = r'''
#include "llama.h"
#include <cstdio>
#include <cstdint>
#include <vector>
int main(int argc, char ** argv) {
    FILE * f = fopen(argv[2], "rb"); char magic[9] = {0};
    if (fread(magic, 1, 8, f) != 8) return 2;
    uint32_t n_ctx; int n_vocab, n_chunk;
    if (fread(&n_ctx, 4, 1, f) != 1 || fread(&n_vocab, 4, 1, f) != 1 || fread(&n_chunk, 4, 1, f) != 1) return 2;
    std::vector<int32_t> tokens((size_t) n_ctx * n_chunk);
    if (fread(tokens.data(), 4, tokens.size(), f) != tokens.size()) return 2;
    fclose(f);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx; cp.n_threads = 2; cp.n_threads_batch = 2;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    const int first = n_ctx / 2;
    FILE * out = fopen(argv[3], "wb");
    int32_t hdr[2] = {n_vocab, (int32_t) (n_chunk * (n_ctx - 1 - first))}; fwrite(hdr, 4, 2, out);
    for (int c = 0; c < n_chunk; ++c) {
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(n_ctx, 0, 1);
        for (uint32_t k = 0; k < n_ctx; ++k) {
            batch.token[k] = (k == 0 && add_bos) ? llama_vocab_bos(vocab) : tokens[(size_t) c * n_ctx + k];   // what perplexity.cpp evaluates
            batch.pos[k] = k; batch.n_seq_id[k] = 1; batch.seq_id[k][0] = 0; batch.logits[k] = 1;
        }
        batch.n_tokens = n_ctx;
        if (llama_decode(ctx, batch)) return 1;
        for (uint32_t p = first; p < n_ctx - 1; ++p) fwrite(llama_get_logits_ith(ctx, p), 4, n_vocab, out);
        llama_batch_free(batch);
    }
    fclose(out);
    return 0;
}
'''


def write_model(path: Path, gguf_py: Path, noise: float, seed: int) -> None:
    sys.path.insert(0, str(gguf_py))
    from gguf import GGMLQuantizationType, GGUFWriter, TokenType
    rng, nrng = np.random.RandomState(1234), np.random.RandomState(seed + 99)
    n_embd, n_head, n_kv, n_layer, n_ff, hd = 64, 4, 2, 2, 128, 16
    toks, scores, types = ["<unk>", "<s>", "</s>"], [0.0] * 3, [TokenType.UNKNOWN, TokenType.CONTROL, TokenType.CONTROL]
    pieces = ["▁"] + list("abcdefghijklmnopqrstuvwxyz.,") + ["▁" + c for c in "abcdefghijklmnopqrstuvwxyz"] + \
        ["▁the", "▁a", "▁of", "▁and", "▁to", "▁in", "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or"]
    for i, w in enumerate(dict.fromkeys(pieces)):      # llama.cpp asserts that every token string is unique
        toks.append(w); scores.append(-1.0 - i * 0.01); types.append(TokenType.NORMAL)
    while len(toks) < 121:
        toks.append(f"<extra{len(toks)}>"); scores.append(-100.0); types.append(TokenType.UNUSED)
    toks, scores, types = toks[:121], scores[:121], types[:121]
    V = len(toks)
    w = GGUFWriter(str(path), "llama")
    w.add_name("tiny-kld-fixture")
    w.add_context_length(256); w.add_embedding_length(n_embd); w.add_block_count(n_layer); w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head); w.add_head_count_kv(n_kv); w.add_layer_norm_rms_eps(1e-5); w.add_rope_dimension_count(hd)
    w.add_tokenizer_model("llama"); w.add_token_list(toks); w.add_token_scores(scores); w.add_token_types(types)
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_unk_token_id(0); w.add_add_bos_token(True); w.add_add_eos_token(False)

    def mat(r, c, std):
        return (rng.normal(size=(r, c)) * std + nrng.normal(size=(r, c)) * std * noise).astype(np.float32)

    def add(name, arr):
        w.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F32)
    add("token_embd.weight", mat(V, n_embd, 0.3))
    add("output_norm.weight", np.ones(n_embd, np.float32))
    add("output.weight", mat(V, n_embd, 0.08))
    for l in range(n_layer):
        add(f"blk.{l}.attn_norm.weight", np.ones(n_embd, np.float32))
        add(f"blk.{l}.attn_q.weight", mat(n_head * hd, n_embd, 0.15))
        add(f"blk.{l}.attn_k.weight", mat(n_kv * hd, n_embd, 0.15))
        add(f"blk.{l}.attn_v.weight", mat(n_kv * hd, n_embd, 0.15))
        add(f"blk.{l}.attn_output.weight", mat(n_embd, n_head * hd, 0.15))
        add(f"blk.{l}.ffn_norm.weight", np.ones(n_embd, np.float32))
        add(f"blk.{l}.ffn_gate.weight", mat(n_ff, n_embd, 0.15))
        add(f"blk.{l}.ffn_up.weight", mat(n_ff, n_embd, 0.15))
        add(f"blk.{l}.ffn_down.weight", mat(n_embd, n_ff, 0.15))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-src", required=True)
    ap.add_argument("--llama-build", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src, build, out = Path(a.llama_src).resolve(), Path(a.llama_build).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(5)
    words = "the quick brown fox jumps over a lazy dog while it is raining in the old town and they said that we can use one of them".split()
    # lowercase letters, spaces and full stops only: the vocabulary has no byte-fallback tokens, so a newline would not tokenize
    text = " ".join(" ".join(rng.choice(words) for _ in range(rng.randint(6, 14))) + "." for _ in range(60))
    (out / "text.txt").write_text(text, encoding="utf-8")
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    write_model(tmp / "a.gguf", src / "gguf-py", 0.0, 0)
    write_model(tmp / "b.gguf", src / "gguf-py", 0.05, 1)
    lp = str(build / "bin" / "llama-perplexity")
    common = ["-f", str(out / "text.txt"), "-c", "64", "--chunks", "4", "-ngl", "0", "-t", "2"]
    subprocess.run([lp, "-m", str(tmp / "a.gguf"), *common, "--kl-divergence-base", str(out / "base.kld")], check=True, capture_output=True)
    for attempt in range(8):
        # to a file, not a pipe; and checked: llama.cpp's log thread can lose the last lines when the process exits (seen once in a pipe)
        with open(tmp / "kl.out", "w", encoding="utf-8") as fo, open(tmp / "kl.err", "w") as fe:
            subprocess.run([lp, "-m", str(tmp / "b.gguf"), *common, "--kl-divergence", "--kl-divergence-base", str(out / "base.kld")],
                           check=True, stdout=fo, stderr=fe)
        blocks = (tmp / "kl.out").read_text(encoding="utf-8")
        if "Same top p" in blocks.split("====== Token probability statistics ======")[-1]:
            break
    else:
        raise SystemExit("llama-perplexity's statistics block was incomplete 8 times")
    keep = [ln for ln in blocks.splitlines() if re.match(r"^(chunk |\s+\d+\s+[\d.]+ ±|=+ |Cor|Mean|Maximum|\s*\d+\.\d%|Median|Minimum|RMS|Same top|$)", ln)]
    (out / "q_kl.txt").write_text("\n".join(keep) + "\n", encoding="utf-8")
    (tmp / "dump.cpp").write_text(DUMPER)
    subprocess.run(["g++", "-O1", "-std=c++17", f"-I{src}/include", f"-I{src}/ggml/include", str(tmp / "dump.cpp"), "-o", str(tmp / "dump"),
                    f"-L{build}/bin", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{build}/bin"], check=True)
    subprocess.run([str(tmp / "dump"), str(tmp / "b.gguf"), str(out / "base.kld"), str(out / "q.logits.bin")], check=True, capture_output=True)
    subprocess.run([str(tmp / "dump"), str(tmp / "a.gguf"), str(out / "base.kld"), str(out / "base.logits.bin")], check=True, capture_output=True)
    subprocess.run(["rm", "-rf", str(tmp)], check=True)
    print("wrote", sorted(p.name for p in out.iterdir()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
