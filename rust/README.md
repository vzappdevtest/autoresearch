# autoresearch (Rust port)

A Rust port of `autoresearch`, refocused on **running large language models
distributed in the GGUF format** — the container used by `llama.cpp` / GGML for
quantized LLM weights.

Where the original Python project *trains* a small GPT from scratch on a single
NVIDIA GPU (PyTorch + FlashAttention-3 + `torch.compile`), this port lets you
**load an existing GGUF LLM and generate text from it**. It is built on
[`candle`](https://github.com/huggingface/candle), HuggingFace's pure-Rust ML
framework, so it runs on **CPU out of the box** and on CUDA / Metal when built
with the matching feature flag.

The original Python files (`train.py`, `prepare.py`, `program.md`) are left
untouched in the repository root; this Rust project lives entirely under
`rust/`.

## What it does

- **`info`** — parse any GGUF file and print a summary: architecture,
  hyperparameters (context length, embedding dim, heads, layers, …),
  quantization mix (tensors per dtype), and optionally the full metadata.
- **`generate`** — load a GGUF LLM and stream generated text from a prompt,
  with temperature / top-k / top-p sampling and a repeat penalty.

### Supported architectures

Generation dispatches on the GGUF `general.architecture` field. Currently
supported: **`llama`** (covers Llama 1/2/3, Mistral, TinyLlama, CodeLlama and
other models that use the `llama` arch in GGUF), **`qwen2`**, **`qwen3`**,
**`phi3`**, and **`gemma3`**. Any GGUF can still be inspected with `info`.

### Tokenizer

Two sources, in priority order:

1. `--tokenizer path/to/tokenizer.json` — a HuggingFace tokenizer file
   (most accurate; recommended).
2. **Embedded** — reconstructed from the `tokenizer.ggml.*` arrays inside the
   GGUF, so a model file is self-contained. Supports the `gpt2` (byte-level
   BPE: Llama 3, Qwen2, Phi-3, …) and `llama`/`spm` (SentencePiece unigram:
   Llama 2, Mistral, …) families. The embedded path is best-effort; prefer a
   real `tokenizer.json` for byte-exact behavior.

## Build

```bash
cd rust
cargo build --release        # CPU
cargo build --release --features cuda    # NVIDIA GPU
cargo build --release --features metal   # Apple Silicon
```

## Usage

```bash
# Inspect a model
cargo run --release -- info model.gguf
cargo run --release -- info model.gguf --verbose      # full metadata dump

# Generate (embedded tokenizer)
cargo run --release -- generate model.gguf \
    --prompt "The capital of France is" \
    --max-tokens 128 --temperature 0.8 --top-p 0.95

# Generate with an explicit tokenizer.json
cargo run --release -- generate model.gguf \
    --tokenizer tokenizer.json --prompt "Once upon a time"
```

Key `generate` flags: `-p/--prompt`, `-n/--max-tokens`, `--temperature`
(`0` = greedy), `--top-p`, `--top-k`, `--repeat-penalty`, `--repeat-last-n`,
`--seed`, `--no-bos`, `--cpu`, `-t/--tokenizer`.

## Getting a model

Download any GGUF LLM, e.g. from HuggingFace:

```bash
# examples (any llama/qwen2/qwen3/phi3/gemma3 GGUF works)
#   TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF   -> tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
#   Qwen/Qwen2-0.5B-Instruct-GGUF            -> qwen2-0_5b-instruct-q4_0.gguf
```

> Note: in some sandboxed/CI environments outbound access to `huggingface.co`
> is blocked by egress policy, so you may need to fetch the file elsewhere and
> copy it in.

## Tests

```bash
cargo test
```

The test suite synthesizes GGUF files in-memory (no downloads) and covers:
- GGUF header parsing and the metadata value formatter,
- building the embedded byte-level BPE tokenizer and an encode/decode roundtrip,
- an **end-to-end** load + forward + sample + streaming-decode run against a
  complete tiny `llama`-architecture GGUF with real (random) weights.

## Project layout

```
rust/
  Cargo.toml
  src/
    main.rs       — CLI (clap): `info`, `generate`
    lib.rs        — library entrypoint
    gguf.rs       — GGUF parsing + `info` summary
    tokenizer.rs  — tokenizer.json / GGUF-embedded tokenizer
    model.rs      — arch detection + quantized model loading + forward
    generate.rs   — autoregressive sampling loop
  tests/          — in-memory GGUF integration tests
```

## License

MIT (same as the parent project).
