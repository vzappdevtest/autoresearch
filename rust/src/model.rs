//! Loading a quantized LLM from a GGUF file and running its forward pass.
//!
//! Different model families store slightly different metadata and need
//! different loaders, so we detect `general.architecture` and dispatch to the
//! matching candle-transformers quantized model. All of them expose the same
//! `forward(tokens, index_pos) -> logits` shape, which we unify behind `Model`.

use anyhow::{bail, Result};
use candle_core::quantized::gguf_file::Content;
use candle_core::{Device, Tensor};
use candle_transformers::models::{
    quantized_gemma3, quantized_llama, quantized_phi3, quantized_qwen2, quantized_qwen3,
};
use std::io::{Read, Seek};

/// A loaded quantized model, one variant per supported architecture.
pub enum Model {
    Llama(quantized_llama::ModelWeights),
    Qwen2(quantized_qwen2::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Phi3(quantized_phi3::ModelWeights),
    Gemma3(quantized_gemma3::ModelWeights),
}

impl Model {
    /// Run one forward pass. `tokens` is shape `[1, seq_len]`; `index_pos` is
    /// the number of tokens already in the KV cache. Returns last-token logits
    /// of shape `[1, vocab_size]`.
    pub fn forward(&mut self, tokens: &Tensor, index_pos: usize) -> Result<Tensor> {
        let logits = match self {
            Model::Llama(m) => m.forward(tokens, index_pos)?,
            Model::Qwen2(m) => m.forward(tokens, index_pos)?,
            Model::Qwen3(m) => m.forward(tokens, index_pos)?,
            Model::Phi3(m) => m.forward(tokens, index_pos)?,
            Model::Gemma3(m) => m.forward(tokens, index_pos)?,
        };
        Ok(logits)
    }
}

/// Architectures we know how to load. Anything else gets a clear error.
pub const SUPPORTED_ARCHS: &[&str] = &["llama", "qwen2", "qwen3", "phi3", "gemma3"];

/// Load a GGUF file into a [`Model`], dispatching on its architecture.
pub fn load<R: Read + Seek>(
    content: Content,
    reader: &mut R,
    arch: &str,
    device: &Device,
) -> Result<Model> {
    let model = match arch {
        // "llama" arch in GGUF covers Llama 1/2/3, Mistral, TinyLlama, CodeLlama, …
        "llama" => Model::Llama(quantized_llama::ModelWeights::from_gguf(
            content, reader, device,
        )?),
        "qwen2" => Model::Qwen2(quantized_qwen2::ModelWeights::from_gguf(
            content, reader, device,
        )?),
        "qwen3" => Model::Qwen3(quantized_qwen3::ModelWeights::from_gguf(
            content, reader, device,
        )?),
        "phi3" => Model::Phi3(quantized_phi3::ModelWeights::from_gguf(
            false, content, reader, device,
        )?),
        "gemma3" => Model::Gemma3(quantized_gemma3::ModelWeights::from_gguf(
            content, reader, device,
        )?),
        other => bail!(
            "unsupported architecture '{other}'. Supported: {}. \
             (The GGUF loads fine for `info`; only generation is arch-specific.)",
            SUPPORTED_ARCHS.join(", ")
        ),
    };
    Ok(model)
}

/// Pick the compute device. CPU unless built with the `cuda`/`metal` feature.
pub fn select_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
    }
    #[cfg(feature = "metal")]
    {
        if let Ok(d) = Device::new_metal(0) {
            return Ok(d);
        }
    }
    Ok(Device::Cpu)
}
