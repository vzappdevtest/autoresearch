//! autoresearch (Rust) — load GGUF-format LLMs and generate text.
//!
//! A Rust port focused on running large language models distributed in the
//! GGUF format (the llama.cpp / GGML container). Built on `candle`, so it runs
//! on CPU out of the box and on CUDA/Metal when built with those features.

use anyhow::{Context, Result};
use autoresearch::generate::GenConfig;
use autoresearch::tokenizer::LlmTokenizer;
use autoresearch::{generate, gguf, model};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "autoresearch",
    version,
    about = "Load GGUF-format LLMs and generate text (Rust/candle port of autoresearch)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Inspect a GGUF file: architecture, hyperparameters, quantization, tensors.
    Info {
        /// Path to the .gguf model file.
        model: PathBuf,
        /// Also dump every metadata key.
        #[arg(short, long)]
        verbose: bool,
    },
    /// Generate text from a prompt using a GGUF LLM.
    Generate {
        /// Path to the .gguf model file.
        model: PathBuf,
        /// The prompt to complete.
        #[arg(short, long, default_value = "Hello, my name is")]
        prompt: String,
        /// Optional tokenizer.json. If omitted, the tokenizer embedded in the
        /// GGUF metadata is used (recommended: pass one for exact behavior).
        #[arg(short, long)]
        tokenizer: Option<PathBuf>,
        /// Maximum number of tokens to generate.
        #[arg(short = 'n', long, default_value_t = 128)]
        max_tokens: usize,
        /// Sampling temperature. 0 = greedy/argmax.
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        /// Nucleus (top-p) sampling threshold.
        #[arg(long)]
        top_p: Option<f64>,
        /// Top-k sampling cutoff.
        #[arg(long)]
        top_k: Option<usize>,
        /// RNG seed for reproducible sampling.
        #[arg(long, default_value_t = 299792458)]
        seed: u64,
        /// Repeat penalty (1.0 disables).
        #[arg(long, default_value_t = 1.1)]
        repeat_penalty: f32,
        /// How many recent tokens the repeat penalty considers.
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
        /// Don't prepend the model's BOS token.
        #[arg(long)]
        no_bos: bool,
        /// Force CPU even if built with a GPU backend.
        #[arg(long)]
        cpu: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Info { model, verbose } => gguf::print_info(&model, verbose),
        Command::Generate {
            model,
            prompt,
            tokenizer,
            max_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            repeat_penalty,
            repeat_last_n,
            no_bos,
            cpu,
        } => {
            let cfg = GenConfig {
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                seed,
                repeat_penalty,
                repeat_last_n,
                add_bos: !no_bos,
            };
            run_generate(&model, tokenizer.as_deref(), cpu, &cfg)
        }
    }
}

fn run_generate(
    model_path: &std::path::Path,
    tokenizer_path: Option<&std::path::Path>,
    cpu: bool,
    cfg: &GenConfig,
) -> Result<()> {
    let device = model::select_device(cpu)?;

    // Parse the GGUF header once; reuse the open file for tensor data.
    let (content, mut file) = gguf::read_content(model_path)?;
    let arch =
        gguf::architecture(&content).context("GGUF file has no general.architecture metadata")?;
    eprintln!(
        "Loading {} (arch: {arch}, device: {:?}) …",
        model_path.display(),
        device
    );

    // Resolve the tokenizer before consuming `content` in the model loader.
    let tokenizer = match tokenizer_path {
        Some(p) => LlmTokenizer::from_json(p)?,
        None => LlmTokenizer::from_gguf(&content)
            .context("no --tokenizer given and GGUF-embedded tokenizer could not be built")?,
    };

    let mut model = model::load(content, &mut file, &arch, &device)?;

    eprintln!("Generating…\n");
    let stats = generate::run(&mut model, &tokenizer, &device, cfg)?;

    eprintln!(
        "\n--- {} prompt tokens in {:.2}s ({:.1} tok/s) | {} generated in {:.2}s ({:.1} tok/s)",
        stats.prompt_tokens,
        stats.prompt_secs,
        stats.prompt_tokens as f64 / stats.prompt_secs.max(1e-9),
        stats.generated_tokens,
        stats.gen_secs,
        stats.generated_tokens as f64 / stats.gen_secs.max(1e-9),
    );
    Ok(())
}
