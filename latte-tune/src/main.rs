use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use colored::*;
use latte_ai::models::{ApiType, Message, Model, TokenUsage};
use latte_ai::params::GenerateParams;
use latte_ai::AiClient;
use crate::report::UsageDisplay;

mod prompts;
mod report;
mod sweeper;

#[derive(Parser)]
#[command(name = "latte-tune", version, about = "AI model parameter tuner")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a parameter sweep on a model
    Sweep {
        /// Model ID (e.g. "deepseek-chat", "claude-sonnet-4-20250514")
        model: String,

        /// API type: openai or anthropic
        #[arg(long, default_value = "openai")]
        api: String,

        /// Base URL for the API
        #[arg(long)]
        base_url: Option<String>,

        /// API key (or set env var: LATTE_API_KEY)
        #[arg(long)]
        api_key: Option<String>,

        /// Provider name
        #[arg(long, default_value = "custom")]
        provider: String,

        /// Max tokens
        #[arg(long, default_value_t = 4096)]
        max_tokens: u32,

        /// Context window
        #[arg(long, default_value_t = 65536)]
        context_window: u32,

        /// Use quick sweep (fewer combinations)
        #[arg(long)]
        quick: bool,

        /// Only run prompts matching this name substring
        #[arg(long)]
        prompt_filter: Option<String>,

        /// Only run sweeps matching this label substring
        #[arg(long)]
        sweep_filter: Option<String>,

        /// Use streaming (shows output as it arrives)
        #[arg(long)]
        stream: bool,

        /// Config file with model definitions (TOML)
        #[arg(long)]
        config: Option<PathBuf>,
    },

    /// List available test prompts
    ListPrompts,

    /// Compare parameters for a specific prompt
    Compare {
        /// Model ID
        model: String,

        /// API type
        #[arg(long, default_value = "openai")]
        api: String,

        /// Base URL
        #[arg(long)]
        base_url: Option<String>,

        /// API key
        #[arg(long)]
        api_key: Option<String>,

        /// Provider name
        #[arg(long, default_value = "custom")]
        provider: String,

        /// Max tokens
        #[arg(long, default_value_t = 4096)]
        max_tokens: u32,

        /// Prompt name to test
        #[arg(long)]
        prompt: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::ListPrompts => list_prompts(),
        Commands::Sweep {
            model, api, base_url, api_key, provider,
            max_tokens, context_window, quick,
            prompt_filter, sweep_filter, stream: _stream, config,
        } => run_sweep(
            model, api, base_url, api_key, provider,
            max_tokens, context_window, quick,
            prompt_filter, sweep_filter, config,
        ).await,
        Commands::Compare {
            model, api, base_url, api_key, provider, max_tokens, prompt,
        } => run_compare(
            model, api, base_url, api_key, provider, max_tokens, prompt,
        ).await,
    }
}

// ── Commands ───────────────────────────────────────────────────────────

fn list_prompts() -> anyhow::Result<()> {
    let test_prompts = prompts::all_prompts();
    println!("{}\n", "Available test prompts:".bold().underline());
    for tp in &test_prompts {
        println!("  {}  [{}]", tp.name.bold().cyan(), tp.category);
        println!("    {}\n", tp.description.dimmed());
    }
    println!("Total: {} prompts", test_prompts.len());
    Ok(())
}

async fn run_sweep(
    model_id: String, api: String, base_url: Option<String>, api_key: Option<String>,
    provider: String, max_tokens: u32, context_window: u32, quick: bool,
    prompt_filter: Option<String>, sweep_filter: Option<String>,
    config: Option<PathBuf>,
) -> anyhow::Result<()> {
    let models = if let Some(config_path) = config {
        load_models_from_config(&config_path)?
    } else {
        vec![build_model(&model_id, &api, &base_url, &api_key, &provider, max_tokens, context_window)]
    };

    for model_cfg in &models {
        println!("\n{}\n",
            format!("Testing: {} ({})", model_cfg.name.bold().white(), model_cfg.id.dimmed()));

        let client = AiClient::new(model_cfg.clone())?;
        let test_prompts = filter_prompts(prompt_filter.as_deref());
        let all_sweeps = if quick { sweeper::quick_sweep() } else { sweeper::programming_sweep() };
        let sweeps = filter_sweeps(&all_sweeps, sweep_filter.as_deref());

        for sweep in &sweeps {
            println!("  {} {}\n", "━━━".dimmed(), sweep.label.bold().yellow());
            let mut results = Vec::new();
            let mut total_usage = TokenUsage::default();

            for tp in &test_prompts {
                let messages = tp.to_messages();
                print!("    {} {} ... ", "→".dimmed(), tp.name.dimmed());

                let result = run_test(&client, &messages, &sweep.params, tp.name).await;
                match &result {
                    Ok(r) => {
                        total_usage.input_tokens += r.input_tokens;
                        total_usage.output_tokens += r.output_tokens;
                        total_usage.thinking_tokens += r.thinking_tokens;
                        results.push(r.clone());
                        println!("{} ({}) [{}ms]", "✓".green(), r.usage_string(), r.duration_ms);
                    }
                    Err(e) => {
                        eprintln!("{} {}: {}", "✗".red(), tp.name, e);
                    }
                }
            }

            println!("\n    {} Sweep total: {} prompts, {} in {}ms\n",
                "📊".bold(), results.len(), total_usage.usage_string(),
                results.iter().map(|r| r.duration_ms).sum::<u64>());
        }
    }

    Ok(())
}

async fn run_compare(
    model_id: String, api: String, base_url: Option<String>, api_key: Option<String>,
    provider: String, max_tokens: u32, prompt_name: String,
) -> anyhow::Result<()> {
    let model = build_model(&model_id, &api, &base_url, &api_key, &provider, max_tokens, 65536);
    let client = AiClient::new(model.clone())?;

    let test_prompts = prompts::all_prompts();
    let tp = test_prompts.iter()
        .find(|p| p.name == prompt_name)
        .ok_or_else(|| anyhow::anyhow!(
            "Prompt '{}' not found. Use `list-prompts` to see available prompts.", prompt_name))?;

    let sweeps = sweeper::programming_sweep();
    let mut all_results = Vec::new();

    for sweep in &sweeps {
        print!("  {} {} ... ", "→".dimmed(), sweep.label.dimmed());
        let messages = tp.to_messages();
        let result = run_test(&client, &messages, &sweep.params, tp.name).await;
        match &result {
            Ok(r) => {
                println!("{} ({}ms)", "✓".green(), r.duration_ms);
                all_results.push(r.clone());
            }
            Err(e) => {
                eprintln!("{}: {}", "✗".red(), e);
            }
        }
    }

    let comparison = report::format_comparison(&model.name, tp.name, &all_results);
    println!("{comparison}");

    Ok(())
}

// ── Helpers ────────────────────────────────────────────────────────────

async fn run_test(
    client: &AiClient,
    messages: &[Message],
    params: &GenerateParams,
    prompt_name: &str,
) -> anyhow::Result<sweeper::SweepResult> {
    let start = Instant::now();

    match client.chat(messages, params).await {
        Ok(completion) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            Ok(sweeper::SweepResult {
                sweep_label: params.label(),
                prompt_name: prompt_name.to_string(),
                category: "test".to_string(),
                output: completion.content,
                input_tokens: completion.usage.input_tokens,
                output_tokens: completion.usage.output_tokens,
                thinking_tokens: completion.usage.thinking_tokens,
                duration_ms,
                error: None,
            })
        }
        Err(e) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            Ok(sweeper::SweepResult {
                sweep_label: params.label(),
                prompt_name: prompt_name.to_string(),
                category: "test".to_string(),
                output: String::new(),
                input_tokens: 0,
                output_tokens: 0,
                thinking_tokens: 0,
                duration_ms,
                error: Some(e.to_string()),
            })
        }
    }
}

fn build_model(
    model_id: &str, api: &str, base_url: &Option<String>,
    api_key: &Option<String>, provider: &str,
    max_tokens: u32, context_window: u32,
) -> Model {
    let api_type = match api.to_lowercase().as_str() {
        "anthropic" => ApiType::AnthropicMessages,
        _ => ApiType::OpenAiCompletions,
    };
    let default_base_url = match api_type {
        ApiType::OpenAiCompletions => "https://api.openai.com",
        ApiType::AnthropicMessages => "https://api.anthropic.com",
    };
    let resolved_key = api_key.clone()
        .or_else(|| std::env::var("LATTE_API_KEY").ok())
        .unwrap_or_default();

    Model {
        id: model_id.to_string(),
        name: model_id.to_string(),
        api: api_type,
        provider: provider.to_string(),
        base_url: base_url.clone().unwrap_or_else(|| default_base_url.into()),
        api_key: resolved_key,
        context_window,
        max_tokens,
        supports_thinking: api_type == ApiType::AnthropicMessages,
        cost_per_million_input: 0.0,
        cost_per_million_output: 0.0,
    }
}

fn filter_prompts(filter: Option<&str>) -> Vec<prompts::TestPrompt> {
    let all = prompts::all_prompts();
    match filter {
        Some(f) if !f.is_empty() => all.into_iter()
            .filter(|p| p.name.contains(f))
            .collect(),
        _ => all,
    }
}

fn filter_sweeps<'a>(
    sweeps: &'a [sweeper::ParamSweep],
    filter: Option<&str>,
) -> Vec<&'a sweeper::ParamSweep> {
    match filter {
        Some(f) if !f.is_empty() => sweeps.iter()
            .filter(|s| s.label.contains(f))
            .collect(),
        _ => sweeps.iter().collect(),
    }
}

fn load_models_from_config(path: &PathBuf) -> anyhow::Result<Vec<Model>> {
    let content = std::fs::read_to_string(path)?;

    #[derive(serde::Deserialize)]
    struct ModelConfig { models: Vec<ModelEntry> }

    #[derive(serde::Deserialize)]
    struct ModelEntry {
        id: String,
        name: Option<String>,
        api: String,
        provider: Option<String>,
        base_url: String,
        api_key: Option<String>,
        context_window: Option<u32>,
        max_tokens: Option<u32>,
    }

    let cfg: ModelConfig = toml::from_str(&content)?;
    let mut models = Vec::new();

    for entry in &cfg.models {
        let api_type = match entry.api.to_lowercase().as_str() {
            "anthropic" | "anthropic-messages" => ApiType::AnthropicMessages,
            _ => ApiType::OpenAiCompletions,
        };
        models.push(Model {
            id: entry.id.clone(),
            name: entry.name.clone().unwrap_or_else(|| entry.id.clone()),
            api: api_type,
            provider: entry.provider.clone().unwrap_or_else(|| "custom".into()),
            base_url: resolve_env(&entry.base_url),
            api_key: resolve_env(&entry.api_key.clone().unwrap_or_default()),
            context_window: entry.context_window.unwrap_or(65536),
            max_tokens: entry.max_tokens.unwrap_or(4096),
            supports_thinking: api_type == ApiType::AnthropicMessages,
            cost_per_million_input: 0.0,
            cost_per_million_output: 0.0,
        });
    }

    Ok(models)
}

fn resolve_env(value: &str) -> String {
    if value.starts_with("${") && value.ends_with('}') {
        let var_name = &value[2..value.len() - 1];
        std::env::var(var_name).unwrap_or_default()
    } else {
        value.to_string()
    }
}
