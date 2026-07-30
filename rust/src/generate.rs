//! Autoregressive text generation loop.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::utils::apply_repeat_penalty;
use std::io::Write;
use std::time::Instant;

use crate::model::Model;
use crate::tokenizer::LlmTokenizer;

/// Knobs controlling a generation run.
pub struct GenConfig {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// Prepend the tokenizer's BOS token (if known) to the prompt.
    pub add_bos: bool,
}

/// Statistics returned after generation for a compact summary line.
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_secs: f64,
    pub gen_secs: f64,
}

/// Generate text, streaming decoded pieces to stdout as they are produced.
pub fn run(
    model: &mut Model,
    tokenizer: &LlmTokenizer,
    device: &Device,
    cfg: &GenConfig,
) -> Result<GenStats> {
    // Encode the prompt. GGUF-embedded tokenizers usually don't inject BOS, so
    // we optionally prepend it explicitly.
    let mut tokens = tokenizer.encode(&cfg.prompt, false)?;
    if cfg.add_bos {
        if let Some(bos) = tokenizer.bos_id {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
    }
    if tokens.is_empty() {
        anyhow::bail!("prompt encoded to zero tokens");
    }
    let prompt_tokens = tokens.len();

    // Sampling strategy from temperature / top-k / top-p.
    let sampling = if cfg.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        match (cfg.top_k, cfg.top_p) {
            (None, None) => Sampling::All {
                temperature: cfg.temperature,
            },
            (Some(k), None) => Sampling::TopK {
                k,
                temperature: cfg.temperature,
            },
            (None, Some(p)) => Sampling::TopP {
                p,
                temperature: cfg.temperature,
            },
            (Some(k), Some(p)) => Sampling::TopKThenTopP {
                k,
                p,
                temperature: cfg.temperature,
            },
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(cfg.seed, sampling);

    let mut all_tokens: Vec<u32> = tokens.clone();
    let mut stream = tokenizer.decode_stream(true);
    let stdout = std::io::stdout();

    // ---- Prompt pass: feed the whole prompt, sample the first new token. ----
    let t_prompt = Instant::now();
    let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next = logits_processor.sample(&logits)?;
    let prompt_secs = t_prompt.elapsed().as_secs_f64();

    all_tokens.push(next);
    emit(&mut stream, next, &stdout)?;

    // ---- Decode loop: one token at a time. ----
    let eos = tokenizer.eos_id;
    let t_gen = Instant::now();
    let mut generated = 1usize;
    for _ in 1..cfg.max_tokens {
        if Some(next) == eos {
            break;
        }
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens + generated - 1)?;
        let logits = logits.squeeze(0)?;

        // Apply repeat penalty over the recent context window.
        let logits = if cfg.repeat_penalty == 1.0 || cfg.repeat_last_n == 0 {
            logits
        } else {
            let start = all_tokens.len().saturating_sub(cfg.repeat_last_n);
            apply_repeat_penalty(&logits, cfg.repeat_penalty, &all_tokens[start..])?
        };

        next = logits_processor.sample(&logits)?;
        all_tokens.push(next);
        generated += 1;
        if Some(next) == eos {
            break;
        }
        emit(&mut stream, next, &stdout)?;
    }
    let gen_secs = t_gen.elapsed().as_secs_f64();
    println!();

    Ok(GenStats {
        prompt_tokens,
        generated_tokens: generated,
        prompt_secs,
        gen_secs,
    })
}

/// Push one token into the streaming decoder and print any resulting text.
fn emit(
    stream: &mut crate::tokenizer::TokenStream<'_>,
    id: u32,
    stdout: &std::io::Stdout,
) -> Result<()> {
    if let Some(piece) = stream.step(id)? {
        let mut lock = stdout.lock();
        print!("{piece}");
        lock.flush().ok();
    }
    Ok(())
}
