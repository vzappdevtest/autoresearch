//! End-to-end test: synthesize a *complete* tiny llama-architecture GGUF with
//! real (random) F32 weights, then load it and run a few generation steps on
//! CPU. This exercises the whole pipeline — GGUF parse, model load, forward
//! pass, sampling, streaming decode — without downloading a multi-GB model.

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::io::Cursor;

use autoresearch::generate::{self, GenConfig};
use autoresearch::model;
use autoresearch::tokenizer::LlmTokenizer;

// Tiny model dimensions (kept small so the forward pass is instant on CPU).
const VOCAB: usize = 6; // matches the 6-token embedded tokenizer below
const EMBD: usize = 32;
const N_HEAD: usize = 4;
const HEAD_DIM: usize = EMBD / N_HEAD; // 8, even -> valid for RoPE
const FFN: usize = 64;
const N_LAYER: usize = 2;

fn qf32(shape: &[usize], device: &Device) -> QTensor {
    // Small random weights so logits stay finite.
    let t = Tensor::randn(0f32, 0.02f32, shape, device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// Write a fully-formed tiny llama GGUF into a byte buffer.
fn synth_llama_gguf() -> Vec<u8> {
    let device = Device::Cpu;

    // ---- metadata ----
    let arch = Value::String("llama".to_string());
    let name = Value::String("tiny-llama-test".to_string());
    let head_count = Value::U32(N_HEAD as u32);
    let head_count_kv = Value::U32(N_HEAD as u32);
    let block_count = Value::U32(N_LAYER as u32);
    let embd = Value::U32(EMBD as u32);
    let rope_dim = Value::U32(HEAD_DIM as u32);
    let ffn_len = Value::U32(FFN as u32);
    let rms_eps = Value::F32(1e-5);
    let ctx_len = Value::U32(64);

    // embedded gpt2 tokenizer with VOCAB tokens
    let tok_model = Value::String("gpt2".to_string());
    let tokens = Value::Array(
        ["<bos>", "the", " cat", " sat", " on", "<eos>"]
            .iter()
            .map(|s| Value::String(s.to_string()))
            .collect(),
    );
    let merges = Value::Array(vec![]);
    let bos = Value::U32(0);
    let eos = Value::U32(5);

    let metadata: Vec<(&str, &Value)> = vec![
        ("general.architecture", &arch),
        ("general.name", &name),
        ("llama.attention.head_count", &head_count),
        ("llama.attention.head_count_kv", &head_count_kv),
        ("llama.block_count", &block_count),
        ("llama.embedding_length", &embd),
        ("llama.feed_forward_length", &ffn_len),
        ("llama.rope.dimension_count", &rope_dim),
        ("llama.attention.layer_norm_rms_epsilon", &rms_eps),
        ("llama.context_length", &ctx_len),
        ("tokenizer.ggml.model", &tok_model),
        ("tokenizer.ggml.tokens", &tokens),
        ("tokenizer.ggml.merges", &merges),
        ("tokenizer.ggml.bos_token_id", &bos),
        ("tokenizer.ggml.eos_token_id", &eos),
    ];

    // ---- tensors ----
    let tok_embd = qf32(&[VOCAB, EMBD], &device);
    let output_norm = qf32(&[EMBD], &device);
    let output = qf32(&[VOCAB, EMBD], &device);

    // Per-layer tensors, kept alive in vectors so we can borrow them below.
    let mut layer_tensors: Vec<(String, QTensor)> = Vec::new();
    for i in 0..N_LAYER {
        let p = format!("blk.{i}");
        layer_tensors.push((format!("{p}.attn_q.weight"), qf32(&[EMBD, EMBD], &device)));
        layer_tensors.push((format!("{p}.attn_k.weight"), qf32(&[EMBD, EMBD], &device)));
        layer_tensors.push((format!("{p}.attn_v.weight"), qf32(&[EMBD, EMBD], &device)));
        layer_tensors.push((
            format!("{p}.attn_output.weight"),
            qf32(&[EMBD, EMBD], &device),
        ));
        layer_tensors.push((format!("{p}.ffn_gate.weight"), qf32(&[FFN, EMBD], &device)));
        layer_tensors.push((format!("{p}.ffn_down.weight"), qf32(&[EMBD, FFN], &device)));
        layer_tensors.push((format!("{p}.ffn_up.weight"), qf32(&[FFN, EMBD], &device)));
        layer_tensors.push((format!("{p}.attn_norm.weight"), qf32(&[EMBD], &device)));
        layer_tensors.push((format!("{p}.ffn_norm.weight"), qf32(&[EMBD], &device)));
    }

    let mut tensors: Vec<(&str, &QTensor)> = vec![
        ("token_embd.weight", &tok_embd),
        ("output_norm.weight", &output_norm),
        ("output.weight", &output),
    ];
    for (name, t) in layer_tensors.iter() {
        tensors.push((name.as_str(), t));
    }

    let mut buf = Cursor::new(Vec::new());
    gguf_file::write(&mut buf, &metadata, &tensors).expect("writing tiny llama gguf");
    buf.into_inner()
}

#[test]
fn loads_tiny_llama_and_generates() {
    let bytes = synth_llama_gguf();

    // Parse header.
    let mut cur = Cursor::new(bytes.clone());
    let content = gguf_file::Content::read(&mut cur).expect("read header");
    assert_eq!(
        autoresearch::gguf::architecture(&content).as_deref(),
        Some("llama")
    );

    // Build tokenizer + model.
    let tokenizer = LlmTokenizer::from_gguf(&content).expect("embedded tokenizer");
    let device = Device::Cpu;
    let mut file = Cursor::new(bytes);
    // re-read content because `load` consumes it
    let content = gguf_file::Content::read(&mut file).expect("read header 2");
    let mut model = model::load(content, &mut file, "llama", &device).expect("load model");

    // Generate a few tokens greedily; just needs to run and produce finite output.
    let cfg = GenConfig {
        prompt: "the".to_string(),
        max_tokens: 5,
        temperature: 0.0, // argmax -> deterministic
        top_p: None,
        top_k: None,
        seed: 42,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        add_bos: true,
    };
    let stats = generate::run(&mut model, &tokenizer, &device, &cfg).expect("generation");
    assert!(stats.prompt_tokens >= 1);
    assert!(stats.generated_tokens >= 1);
    assert!(stats.generated_tokens <= 5);
}
