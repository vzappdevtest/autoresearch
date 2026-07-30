//! Integration tests that synthesize a tiny GGUF file in-memory and exercise
//! the header parsing and the GGUF-embedded tokenizer reconstruction. This
//! covers the non-model-weight code paths without needing to download a model.

use candle_core::quantized::gguf_file::{self, Value};
use std::io::Cursor;

use autoresearch::gguf;
use autoresearch::tokenizer::LlmTokenizer;

/// Build a minimal gpt2-style byte-level BPE GGUF with a handful of tokens and
/// merges, enough to tokenize "ab" -> "ab" via the merge `a b`.
fn synth_gpt2_gguf() -> Vec<u8> {
    // Byte-level vocab: a special BOS/EOS plus single-byte tokens and a merged
    // token "ab". BOS/EOS are distinct control tokens (id 0 and 5) so they do
    // not collide with ordinary content, mirroring real models.
    let tokens: Vec<Value> = ["<bos>", "a", "b", "c", "ab", "<eos>"]
        .iter()
        .map(|s| Value::String(s.to_string()))
        .collect();
    let merges: Vec<Value> = ["a b"]
        .iter()
        .map(|s| Value::String(s.to_string()))
        .collect();

    let arch = Value::String("llama".to_string());
    let name = Value::String("synthetic-test".to_string());
    let tok_model = Value::String("gpt2".to_string());
    let tokens_v = Value::Array(tokens);
    let merges_v = Value::Array(merges);
    let eos = Value::U32(5);
    let bos = Value::U32(0);

    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch),
        ("general.name", &name),
        ("tokenizer.ggml.model", &tok_model),
        ("tokenizer.ggml.tokens", &tokens_v),
        ("tokenizer.ggml.merges", &merges_v),
        ("tokenizer.ggml.bos_token_id", &bos),
        ("tokenizer.ggml.eos_token_id", &eos),
    ];

    let mut buf = Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &metadata, &[]).expect("writing synthetic gguf");
    buf.into_inner()
}

#[test]
fn parses_header_and_architecture() {
    let bytes = synth_gpt2_gguf();
    let mut cur = Cursor::new(bytes);
    let content = gguf_file::Content::read(&mut cur).expect("reading gguf header");
    assert_eq!(gguf::architecture(&content).as_deref(), Some("llama"));
    assert_eq!(content.metadata.len(), 7);
}

#[test]
fn builds_embedded_bpe_tokenizer_and_roundtrips() {
    let bytes = synth_gpt2_gguf();
    let mut cur = Cursor::new(bytes);
    let content = gguf_file::Content::read(&mut cur).expect("reading gguf header");

    let tok = LlmTokenizer::from_gguf(&content).expect("building embedded tokenizer");
    assert_eq!(tok.bos_id, Some(0));
    assert_eq!(tok.eos_id, Some(5));

    // "ab" should merge into the single token id 4 ("ab").
    let ids = tok.encode("ab", false).expect("encode");
    assert_eq!(ids, vec![4], "expected the `a b`->`ab` merge to apply");

    // Roundtrip decode.
    let text = tok.decode(&ids, false).expect("decode");
    assert_eq!(text, "ab");
}

#[test]
fn value_to_string_truncates_arrays() {
    let arr = Value::Array((0..100).map(Value::U32).collect());
    let s = gguf::value_to_string(&arr);
    assert!(s.contains("100 items"), "got: {s}");
}
