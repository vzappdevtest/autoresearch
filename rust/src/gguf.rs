//! GGUF file helpers: reading metadata values and printing a human-readable
//! summary of a GGUF model file (architecture, hyperparameters, tensors).

use anyhow::{Context, Result};
use candle_core::quantized::gguf_file::{Content, Value};
use std::fs::File;
use std::path::Path;

/// Open a GGUF file and parse its header (metadata + tensor table). The tensor
/// *data* is not read here — only offsets — so this is cheap even for huge files.
pub fn read_content(path: &Path) -> Result<(Content, File)> {
    let mut file =
        File::open(path).with_context(|| format!("opening GGUF file {}", path.display()))?;
    let content = Content::read(&mut file)
        .map_err(|e| e.with_path(path))
        .with_context(|| format!("parsing GGUF header of {}", path.display()))?;
    Ok((content, file))
}

/// Render a metadata [`Value`] as a compact one-line string. Long arrays are
/// truncated so that dumping metadata never floods the terminal.
pub fn value_to_string(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => format!("{x}"),
        Value::F64(x) => format!("{x}"),
        Value::Bool(x) => x.to_string(),
        Value::String(x) => truncate(x, 80),
        Value::Array(items) => {
            let n = items.len();
            let preview: Vec<String> = items.iter().take(6).map(value_to_string).collect();
            if n > preview.len() {
                format!("[{}, … {} items]", preview.join(", "), n)
            } else {
                format!("[{}]", preview.join(", "))
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() > max {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    } else {
        s.to_string()
    }
}

/// The `general.architecture` field, e.g. "llama", "qwen2", "gemma".
pub fn architecture(content: &Content) -> Option<String> {
    content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .cloned()
}

/// Print a readable summary of a GGUF file to stdout.
pub fn print_info(path: &Path, verbose: bool) -> Result<()> {
    let (content, _file) = read_content(path)?;

    let arch = architecture(&content).unwrap_or_else(|| "<unknown>".to_string());
    let name = content
        .metadata
        .get("general.name")
        .and_then(|v| v.to_string().ok())
        .cloned()
        .unwrap_or_else(|| "<unnamed>".to_string());

    println!("File:          {}", path.display());
    println!("Name:          {name}");
    println!("Architecture:  {arch}");
    println!("Metadata keys: {}", content.metadata.len());
    println!("Tensors:       {}", content.tensor_infos.len());

    // Highlight the parameters most useful for understanding the model.
    let highlights = [
        ("Context length", format!("{arch}.context_length")),
        ("Embedding dim", format!("{arch}.embedding_length")),
        ("Block/layer count", format!("{arch}.block_count")),
        ("FFN dim", format!("{arch}.feed_forward_length")),
        ("Attention heads", format!("{arch}.attention.head_count")),
        ("KV heads", format!("{arch}.attention.head_count_kv")),
        ("RoPE freq base", format!("{arch}.rope.freq_base")),
        ("Tokenizer model", "tokenizer.ggml.model".to_string()),
    ];
    println!("\nKey parameters:");
    for (label, key) in highlights.iter() {
        if let Some(v) = content.metadata.get(key.as_str()) {
            println!("  {label:<18}: {}", value_to_string(v));
        }
    }
    // Vocab size is best read off the token list length when present.
    if let Some(Value::Array(tokens)) = content.metadata.get("tokenizer.ggml.tokens") {
        println!("  {:<18}: {}", "Vocab size", tokens.len());
    }

    // Quantization mix: count tensors per ggml dtype.
    let mut dtype_counts: std::collections::BTreeMap<String, usize> = Default::default();
    for info in content.tensor_infos.values() {
        *dtype_counts
            .entry(format!("{:?}", info.ggml_dtype))
            .or_default() += 1;
    }
    println!("\nQuantization (tensors per dtype):");
    for (dtype, count) in dtype_counts.iter() {
        println!("  {dtype:<10}: {count}");
    }

    if verbose {
        println!("\nAll metadata:");
        let mut keys: Vec<&String> = content.metadata.keys().collect();
        keys.sort();
        for key in keys {
            // Skip the giant tokenizer arrays in the full dump unless truly wanted.
            let v = &content.metadata[key];
            println!("  {key} = {}", value_to_string(v));
        }
    }

    Ok(())
}
