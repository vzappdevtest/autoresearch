//! autoresearch (Rust) — a library for loading GGUF-format LLMs and generating
//! text, built on `candle`.
//!
//! Modules:
//! - [`gguf`]: parse and summarize GGUF files, read metadata values.
//! - [`tokenizer`]: obtain a tokenizer from `tokenizer.json` or GGUF metadata.
//! - [`model`]: load a quantized model and run its forward pass.
//! - [`generate`]: the autoregressive sampling loop.

pub mod generate;
pub mod gguf;
pub mod model;
pub mod tokenizer;
