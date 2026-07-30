# autoresearch (Erlang port)

A **pure-Erlang** port of `autoresearch`, focused on **loading GGUF-format LLMs
and generating text** — no NIFs, no C, no external dependencies beyond OTP.

Erlang has no ML/tensor ecosystem, so everything here is implemented from
scratch: the GGUF binary parser (using Erlang's bit-syntax, which is a great
fit for the format), tensor dequantization, the tensor math, a Llama-family
forward pass, a GPT-2 byte-level BPE tokenizer, and the sampling loop.

> **Scope / performance.** This is a from-scratch, list-based numeric engine.
> It is correct and dependency-free, but **not fast** — it recomputes the whole
> sequence each step and uses plain Erlang floats/lists. It is meant for small
> models and for demonstrating the algorithms end to end, not for running
> multi-billion-parameter models. For production-scale inference use the Rust
> port (`../rust`) or `llama.cpp`.

## What it does

- **`info`** — parse any GGUF file and print a summary: architecture,
  hyperparameters, quantization mix (tensors per dtype), and (with `--verbose`)
  all metadata.
- **`generate`** — load a `llama`-architecture GGUF and generate text with
  temperature / top-k sampling.

### Supported

- **Architecture:** `llama` for generation (Llama 1/2/3, Mistral, TinyLlama, …).
  Any GGUF can be inspected with `info`.
- **Dtypes (dequantization):** `F32`, `F16`, `Q8_0`, `Q4_0`. Other quantizations
  (k-quants like `Q4_K`) inspect fine but cannot yet be run.
- **Tokenizer:** the `gpt2` byte-level BPE family, reconstructed from the
  embedded `tokenizer.ggml.*` metadata (SentencePiece/`llama` unigram is not
  implemented).

## Requirements

Erlang/OTP (tested on OTP 25). On Debian/Ubuntu:

```bash
sudo apt-get install erlang-base erlang-crypto erlang-eunit erlang-tools
```

## Build & run

```bash
cd erlang
make                       # compiles src/ into ebin/
./autoresearch info model.gguf
./autoresearch info model.gguf --verbose
./autoresearch generate model.gguf --prompt "Once upon a time" -n 64 --temperature 0.8
./autoresearch generate model.gguf --prompt "2 + 2 =" -n 8 --temperature 0    # greedy
```

`generate` options: `-p/--prompt`, `-n/--max-tokens`, `--temperature`
(`0` = greedy), `--top-k`, `--seed`, `--no-bos`.

## Tests

```bash
make test
```

Fixtures are synthesized in-memory (`test/ar_synth.erl`) — no downloads. Coverage:
- GGUF header/metadata/tensor-table parsing,
- dequantization of `F16`, `Q8_0`, `Q4_0` against hand-crafted blocks,
- the GPT-2 BPE tokenizer (merge + encode/decode roundtrip, special-token skip),
- tensor math (dot, matvec, softmax, argmax),
- an **end-to-end** load → forward → sample → decode run on a complete tiny
  `llama` GGUF with real weights.

## Layout

```
erlang/
  Makefile
  autoresearch          — escript CLI entry
  src/
    ar_gguf.erl         — GGUF parser + dequantization
    ar_tensor.erl       — vector/matrix math (dot, matvec, rmsnorm, softmax, silu)
    ar_llama.erl        — Llama forward pass (RoPE, GQA, SwiGLU)
    ar_tokenizer.erl    — GPT-2 byte-level BPE from GGUF metadata
    ar_generate.erl     — sampling loop
    ar_cli.erl          — `info` / `generate` commands
  test/
    ar_synth.erl        — in-memory GGUF synthesizer (test helper)
    ar_tests.erl        — EUnit tests
```

## Notes on correctness

The forward pass follows the standard Llama recipe (RMSNorm, rotary embeddings
in the NeoX half-split convention, grouped-query attention with a causal mask,
SwiGLU MLP, tied-or-separate output projection). The RoPE/normalization
conventions are intended to match llama.cpp/GGUF; because real models cannot be
downloaded in every environment, numerical parity against a reference is not
asserted in CI — the tests validate shapes, determinism, and the full pipeline
on synthesized weights.

## License

MIT (same as the parent project).
