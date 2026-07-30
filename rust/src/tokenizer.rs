//! Tokenizer handling for GGUF LLMs.
//!
//! Two ways to obtain a tokenizer:
//!   1. A HuggingFace `tokenizer.json` file (most accurate — preferred).
//!   2. Reconstructed from the tokenizer arrays embedded in the GGUF metadata
//!      (`tokenizer.ggml.*`). This makes a GGUF file self-contained: no extra
//!      download needed. We support the two common families:
//!        - "gpt2"  : byte-level BPE (Llama 3, Qwen2, GPT-2, Phi-3, …)
//!        - "llama" : SentencePiece unigram (Llama 2, Mistral, …)
//!
//! The GGUF-embedded path is best-effort; for byte-exact behavior prefer a
//! real `tokenizer.json`.

use anyhow::{bail, Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use std::path::Path;
use tokenizers::models::bpe::{Vocab, BPE};
use tokenizers::models::unigram::Unigram;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::{AddedToken, Tokenizer};

/// Thin wrapper around a `tokenizers::Tokenizer` plus the special token ids we
/// need for generation (BOS to prepend, EOS to stop on).
pub struct LlmTokenizer {
    inner: Tokenizer,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
}

impl LlmTokenizer {
    /// Load from a HuggingFace `tokenizer.json`.
    pub fn from_json(path: &Path) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer.json {}: {e}", path.display()))?;
        Ok(Self {
            inner,
            bos_id: None,
            eos_id: None,
        })
    }

    /// Reconstruct a tokenizer from `tokenizer.ggml.*` metadata inside a GGUF file.
    pub fn from_gguf(content: &Content) -> Result<Self> {
        let model = content
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .context(
                "GGUF has no embedded tokenizer (tokenizer.ggml.model missing); \
                 pass --tokenizer <tokenizer.json>",
            )?;

        let tokens = get_string_array(content, "tokenizer.ggml.tokens")
            .context("GGUF missing tokenizer.ggml.tokens")?;

        let bos_id = content
            .metadata
            .get("tokenizer.ggml.bos_token_id")
            .and_then(|v| v.to_u32().ok());
        let eos_id = content
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());
        let unk_id = content
            .metadata
            .get("tokenizer.ggml.unknown_token_id")
            .and_then(|v| v.to_u32().ok());

        let mut inner = match model.as_str() {
            "gpt2" => build_bpe(content, &tokens)?,
            "llama" | "spm" => build_unigram(content, &tokens, unk_id)?,
            other => bail!(
                "unsupported embedded tokenizer model '{other}'; pass --tokenizer <tokenizer.json>"
            ),
        };

        // Register BOS/EOS as special tokens so they are handled/skipped on decode.
        let mut specials = Vec::new();
        if let Some(id) = bos_id {
            if let Some(t) = tokens.get(id as usize) {
                specials.push(AddedToken::from(t.clone(), true));
            }
        }
        if let Some(id) = eos_id {
            if let Some(t) = tokens.get(id as usize) {
                specials.push(AddedToken::from(t.clone(), true));
            }
        }
        if !specials.is_empty() {
            inner.add_special_tokens(&specials);
        }

        Ok(Self {
            inner,
            bos_id,
            eos_id,
        })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special)
            .map_err(|e| anyhow::anyhow!("decode failed: {e}"))
    }

    /// A streaming decoder that yields text incrementally, one token at a time.
    /// Correctly handles multi-byte pieces that only complete across tokens.
    pub fn decode_stream(&self, skip_special: bool) -> TokenStream<'_> {
        TokenStream {
            inner: self.inner.decode_stream(skip_special),
        }
    }
}

/// Incremental decoder wrapper around `tokenizers::DecodeStream`.
pub struct TokenStream<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    >,
}

impl TokenStream<'_> {
    /// Feed one token id; returns any newly-decodable text.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner
            .step(id)
            .map_err(|e| anyhow::anyhow!("decode_stream step failed: {e}"))
    }
}

/// Build a byte-level BPE tokenizer (gpt2 family) from GGUF arrays.
fn build_bpe(content: &Content, tokens: &[String]) -> Result<Tokenizer> {
    // vocab: token string -> id (id is just the array index in GGUF).
    let mut vocab: Vocab = Vocab::default();
    for (id, tok) in tokens.iter().enumerate() {
        vocab.insert(tok.clone(), id as u32);
    }

    // merges: "A B" -> (A, B), in priority order.
    let merges_raw = get_string_array(content, "tokenizer.ggml.merges").unwrap_or_default();
    let mut merges: Vec<(String, String)> = Vec::with_capacity(merges_raw.len());
    for m in merges_raw.iter() {
        if let Some((a, b)) = m.split_once(' ') {
            merges.push((a.to_string(), b.to_string()));
        }
    }

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .ignore_merges(true)
        .build()
        .map_err(|e| anyhow::anyhow!("building BPE model: {e}"))?;

    let mut tok = Tokenizer::new(bpe);
    // add_prefix_space=false matches modern GGUF/HF byte-level configs.
    tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tok.with_decoder(Some(ByteLevel::new(false, true, true)));
    Ok(tok)
}

/// Build a SentencePiece unigram tokenizer (llama family) from GGUF arrays.
fn build_unigram(content: &Content, tokens: &[String], unk_id: Option<u32>) -> Result<Tokenizer> {
    let scores = get_f32_array(content, "tokenizer.ggml.scores").unwrap_or_default();
    if scores.len() != tokens.len() {
        // Unigram requires a score per token; without them we cannot proceed.
        bail!(
            "GGUF unigram tokenizer missing per-token scores ({} tokens, {} scores); \
             pass --tokenizer <tokenizer.json>",
            tokens.len(),
            scores.len()
        );
    }
    let vocab: Vec<(String, f64)> = tokens
        .iter()
        .zip(scores.iter())
        .map(|(t, s)| (t.clone(), *s as f64))
        .collect();

    let unigram = Unigram::from(vocab, unk_id.map(|x| x as usize), true)
        .map_err(|e| anyhow::anyhow!("building Unigram model: {e}"))?;

    let mut tok = Tokenizer::new(unigram);
    // SentencePiece: '▁' replaces spaces, with a dummy prefix space.
    let metaspace = Metaspace::new('▁', PrependScheme::Always, false);
    tok.with_pre_tokenizer(Some(metaspace.clone()));
    tok.with_decoder(Some(metaspace));
    Ok(tok)
}

fn get_string_array(content: &Content, key: &str) -> Option<Vec<String>> {
    match content.metadata.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(it.to_string().ok()?.clone());
            }
            Some(out)
        }
        _ => None,
    }
}

fn get_f32_array(content: &Content, key: &str) -> Option<Vec<f32>> {
    match content.metadata.get(key) {
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                out.push(it.to_f32().ok()?);
            }
            Some(out)
        }
        _ => None,
    }
}
