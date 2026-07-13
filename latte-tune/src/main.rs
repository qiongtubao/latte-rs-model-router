//! `latte-tune` — AI model parameter tuner / single-model chat client.
//!
//! Loads models from `~/.latte/models.d/` and `./.latte/models.d/` via
//! `latte-router::ModelCatalog`, or from `--config <dir>` if given.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use colored::*;
use latte_ai::models::{ApiType, Message, Model, StreamEvent, TokenUsage};
use latte_ai::params::GenerateParams;
use latte_ai::AiClient;
use latte_router::{ModelCatalog, ModelEntry};

mod prompts;
mod report;
mod sweeper;

use crate::prompts::TestPrompt;
use crate::report::UsageDisplay;
use crate::sweeper::SweepResult;

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
        /// Model ID (looked up in models.d/)
        model: Option<String>,

        /// API type: openai or anthropic (only used when --config is not given)
        #[arg(long, default_value = "openai")]
        api: String,

        /// Base URL (only used when --config is not given)
        #[arg(long)]
        base_url: Option<String>,

        /// API key (only used when --config is not given)
        #[arg(long)]
        api_key: Option<String>,

        /// Provider name (only used when --config is not given)
        #[arg(long, default_value = "custom")]
        provider: String,

        /// Max tokens (only used when --config is not given)
        #[arg(long, default_value_t = 4096)]
        max_tokens: u32,

        /// Context window (only used when --config is not given)
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

        /// Config dir (overrides default models.d/ lookup)
        #[arg(long)]
        config: Option<PathBuf>,
    },

    /// List configured models
    ListModels {
        /// Config dir (overrides default models.d/ lookup)
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

    /// Chat with a model (defaults to interactive REPL if no prompt given)
    Chat {
        /// Prompt text. Reads from stdin if piped. Enters interactive REPL if omitted.
        prompt: Option<String>,
        /// Model ID to use (uses first configured model if not specified)
        #[arg(short, long)]
        model: Option<String>,
        /// Config dir (overrides default models.d/ lookup)
        #[arg(long)]
        config: Option<PathBuf>,
        /// Disable streaming output
        #[arg(long)]
        no_stream: bool,
        /// Force interactive REPL mode
        #[arg(short = 'r', long)]
        repl: bool,
    },
}

fn resolve_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if s == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(s)
}

/// Load models from `~/.latte/models.d/` (global) + `./.latte/models.d/` (project override),
/// or from `--config <dir>` if given.
fn load_models_resolved(config_dir: Option<&Path>) -> Result<Vec<Model>> {
    let mut catalog = ModelCatalog::new();

    let (global_dir, project_dir) = match config_dir {
        Some(p) => (p.to_path_buf(), None),
        None => {
            let global = resolve_tilde("~/.latte/models.d");
            let project = PathBuf::from(".latte/models.d");
            (global, Some(project))
        }
    };

    let _ = catalog.load_dir(&global_dir);
    if let Some(p) = project_dir {
        let _ = catalog.load_dir(&p);
    }

    if catalog.is_empty() {
        anyhow::bail!(
            "no models found; create {} (and/or ./.latte/models.d) or pass --config <dir>",
            global_dir.display()
        );
    }

    let mut models: Vec<Model> = catalog
        .ids()
        .map(|id| entry_to_model(catalog.get(id).expect("just enumerated")))
        .collect();
    models.sort_by(|a, b| a.provider.cmp(&b.provider).then(a.name.cmp(&b.name)));
    Ok(models)
}

/// Convert a `latte_router::ModelEntry` to a `latte_ai::models::Model`.
fn entry_to_model(entry: &ModelEntry) -> Model {
    Model {
        id: entry.id.clone(),
        name: entry.display_name().to_string(),
        api: entry.api,
        provider: entry.provider.clone(),
        base_url: entry.base_url.clone(),
        api_key: entry.api_key.clone(),
        context_window: entry.context_window,
        max_tokens: entry.max_tokens,
        supports_thinking: entry.api == ApiType::AnthropicMessages,
        supports_vision: entry.supports_vision,
        cost_per_million_input: 0.0,
        cost_per_million_output: 0.0,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::ListModels { config } => list_models(config.as_deref()),
        Commands::ListPrompts => list_prompts(),
        Commands::Sweep {
            model,
            api,
            base_url,
            api_key,
            provider,
            max_tokens,
            context_window,
            quick,
            prompt_filter,
            sweep_filter,
            stream: _stream,
            config,
        } => {
            run_sweep(
                model,
                api,
                base_url,
                api_key,
                provider,
                max_tokens,
                context_window,
                quick,
                prompt_filter,
                sweep_filter,
                config.as_deref(),
            )
            .await
        }
        Commands::Compare {
            model,
            api,
            base_url,
            api_key,
            provider,
            max_tokens,
            prompt,
        } => {
            run_compare(
                model, api, base_url, api_key, provider, max_tokens, prompt, None,
            )
            .await
        }
        Commands::Chat { prompt, model, config, no_stream, repl } => {
            run_chat(prompt, model, config.as_deref(), no_stream, repl).await
        }
    }
}

// ── ListModels ────────────────────────────────────────────────────────

fn list_models(config_dir: Option<&Path>) -> Result<()> {
    let models = load_models_resolved(config_dir)?;
    if models.is_empty() {
        println!("No models configured.");
        return Ok(());
    }
    print_model_list(&models, None);
    Ok(())
}

fn list_prompts() -> Result<()> {
    let test_prompts = prompts::all_prompts();
    println!("{}\n", "Available test prompts:".bold().underline());
    for tp in &test_prompts {
        println!("  {}  [{}]", tp.name.bold().cyan(), tp.category);
        println!("    {}\n", tp.description.dimmed());
    }
    println!("Total: {} prompts", test_prompts.len());
    Ok(())
}

// ── Model resolution helpers ─────────────────────────────────────────

fn find_model_index(models: &[Model], query: &str) -> Option<usize> {
    if let Ok(n) = query.parse::<usize>() {
        if n >= 1 && n <= models.len() {
            return Some(n - 1);
        }
    }
    let q = query.to_lowercase();
    models
        .iter()
        .position(|m| m.id.to_lowercase().contains(&q) || m.name.to_lowercase().contains(&q))
}

fn resolve_model(filter: Option<&str>, config_dir: Option<&Path>) -> Result<Model> {
    let models = load_models_resolved(config_dir)?;
    if models.is_empty() {
        anyhow::bail!("No models configured.");
    }
    if let Some(f) = filter {
        find_model_index(&models, f)
            .map(|i| models[i].clone())
            .with_context(|| format!("Model '{f}' not found in models.d/"))
    } else {
        Ok(models.into_iter().next().unwrap())
    }
}

fn print_model_list(models: &[Model], current: Option<usize>) {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<&str, Vec<(usize, &Model)>> = BTreeMap::new();
    for (i, m) in models.iter().enumerate() {
        groups.entry(&m.provider).or_default().push((i, m));
    }
    eprintln!();
    for (provider, entries) in &groups {
        eprintln!("  {}", provider.bold().underline());
        for (idx, m) in entries {
            let mark = if Some(*idx) == current { "◀".green() } else { " ".normal() };
            eprintln!("    {} {:2}. {} {}",
                mark, idx + 1, m.name.cyan(), m.id.dimmed());
        }
    }
    eprintln!();
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max.saturating_sub(1)])
    }
}

// ── Chat ─────────────────────────────────────────────────────────────

async fn run_chat(
    prompt: Option<String>,
    model_filter: Option<String>,
    config_dir: Option<&Path>,
    no_stream: bool,
    force_repl: bool,
) -> Result<()> {
    use std::io::{self, BufRead, IsTerminal, Read, Write};

    let is_piped = !std::io::stdin().is_terminal();
    let enter_repl = force_repl || (prompt.is_none() && !is_piped);

    if enter_repl {
        let all_models = load_models_resolved(config_dir)?;
        if all_models.is_empty() {
            anyhow::bail!("No models found in models.d/.");
        }
        let model_idx = if let Some(ref filter) = model_filter {
            find_model_index(&all_models, filter).with_context(|| {
                format!("Model '{filter}' not found. Use /models to list.")
            })?
        } else {
            0
        };
        let model = &all_models[model_idx];
        let mut current_idx = model_idx;
        let mut client = AiClient::new(model.clone())?;
        let params = GenerateParams::code_defaults();
        let mut history: Vec<Message> = Vec::new();

        eprintln!("{}", "╔══════════════════════════════════════════╗".dimmed());
        eprintln!("{}", "║  latte chat — interactive REPL           ║".dimmed());
        eprintln!("{}", format!("║  Model: {:33} ║", truncate(&model.name, 33)).dimmed());
        eprintln!(
            "{}",
            format!(
                "║  {} models  /exit  /clear  /model  /models║",
                all_models.len()
            )
            .dimmed()
        );
        eprintln!("{}", "╚══════════════════════════════════════════╝".dimmed());

        let mut stdin = io::BufReader::new(io::stdin());
        loop {
            eprint!("\n{} ", "▶".cyan().bold());
            io::stderr().flush()?;

            let mut line = String::new();
            if stdin.read_line(&mut line)? == 0 {
                break;
            }
            let input = line.trim().to_string();
            if input.is_empty() {
                continue;
            }
            if input.starts_with('/') {
                match input.as_str() {
                    "/exit" | "/quit" | "/q" => break,
                    "/clear" | "/c" => {
                        history.clear();
                        eprintln!("  {}", "History cleared.".dimmed());
                        continue;
                    }
                    "/models" | "/ls" => {
                        print_model_list(&all_models, Some(current_idx));
                        continue;
                    }
                    cmd if cmd == "/model" || cmd.starts_with("/model ") => {
                        let arg = cmd.strip_prefix("/model").unwrap().trim();
                        if arg.is_empty() {
                            print_model_list(&all_models, Some(current_idx));
                            continue;
                        }
                        if let Some(idx) = find_model_index(&all_models, arg) {
                            let m = &all_models[idx];
                            eprintln!("  {} {} ({})",
                                "Switched to:".green(), m.name.cyan(), m.id.dimmed());
                            client = AiClient::new(m.clone())?;
                            current_idx = idx;
                            history.clear();
                        } else {
                            eprintln!("  {}: '{}' not found. Use /models to list.",
                                "Error".red(), arg);
                        }
                        continue;
                    }
                    "/help" | "/h" | "/?" => {
                        eprintln!("  /exit, /quit, /q     — 退出");
                        eprintln!("  /clear, /c           — 清空对话历史");
                        eprintln!("  /model [name|idx]    — 切换模型（无参列出）");
                        eprintln!("  /models, /ls         — 列出所有可用模型");
                        eprintln!("  /help, /h            — 显示帮助");
                        continue;
                    }
                    _ => {
                        eprintln!("  {}: `{}` (use /help)", "Unknown command".yellow(), input);
                        continue;
                    }
                }
            }
            history.push(Message::user(input));

            eprint!("\n{} ", "◀".green().bold());
            io::stderr().flush()?;

            if no_stream {
                let completion = client.chat(&history, &params).await?;
                println!("{}", completion.content);
                history.push(Message::assistant(completion.content.clone()));
                eprintln!("\n  {}", completion.usage);
            } else {
                let mut stream = client.chat_stream(&history, &params).await?;
                let mut full = String::new();
                let mut usage = TokenUsage::default();
                while let Some(event) = stream.recv().await {
                    match event {
                        StreamEvent::Delta { content, usage: u } => {
                            print!("{}", content);
                            full.push_str(&content);
                            if let Some(u) = u {
                                usage = u;
                            }
                        }
                        StreamEvent::Done { usage: u, .. } => usage = u,
                        StreamEvent::Error(e) => eprintln!("\n  {}", e.to_string().red()),
                    }
                }
                history.push(Message::assistant(full));
                eprintln!("\n  {}", usage);
            }
        }
        eprintln!("\n  {}", "Goodbye!".dimmed());
    } else {
        let prompt_text = match prompt {
            Some(p) if !p.is_empty() => p,
            _ => {
                let mut buf = String::new();
                io::stdin().read_to_string(&mut buf)?;
                if buf.trim().is_empty() {
                    anyhow::bail!("No prompt provided.");
                }
                buf
            }
        };

        let model = resolve_model(model_filter.as_deref(), config_dir)?;
        eprintln!("  Model: {} ({})", model.name.bold().cyan(), model.id.dimmed());

        let client = AiClient::new(model)?;
        let params = GenerateParams::code_defaults();
        let messages = vec![Message::user(prompt_text)];

        if no_stream {
            let completion = client.chat(&messages, &params).await?;
            println!("{}", completion.content);
            eprintln!("\n  {}", completion.usage);
        } else {
            let mut stream = client.chat_stream(&messages, &params).await?;
            let mut usage = TokenUsage::default();
            while let Some(event) = stream.recv().await {
                match event {
                    StreamEvent::Delta { content, usage: u } => {
                        print!("{}", content);
                        if let Some(u) = u {
                            usage = u;
                        }
                    }
                    StreamEvent::Done { usage: u, .. } => usage = u,
                    StreamEvent::Error(e) => eprintln!("\n  {}", e.to_string().red()),
                }
            }
            eprintln!("\n\n  {}", usage);
        }
    }
    Ok(())
}

// ── Sweep ────────────────────────────────────────────────────────────

async fn run_sweep(
    model_id: Option<String>,
    api: String,
    base_url: Option<String>,
    api_key: Option<String>,
    provider: String,
    max_tokens: u32,
    context_window: u32,
    quick: bool,
    prompt_filter: Option<String>,
    sweep_filter: Option<String>,
    config_dir: Option<&Path>,
) -> Result<()> {
    let models = if let Some(dir) = config_dir {
        load_models_resolved(Some(dir))?
    } else {
        match load_models_resolved(None) {
            Ok(m) if !m.is_empty() => m,
            _ => {
                let id = model_id.with_context(|| {
                    "no model specified. Provide a model ID or populate models.d/"
                })?;
                vec![build_model(&id, &api, &base_url, &api_key, &provider, max_tokens, context_window)]
            }
        }
    };

    for model_cfg in &models {
        println!(
            "\n{}\n",
            format!("Testing: {} ({})", model_cfg.name.bold().white(), model_cfg.id.dimmed())
        );

        let client = AiClient::new(model_cfg.clone())?;
        let test_prompts = filter_prompts(prompt_filter.as_deref());
        let all_sweeps = if quick {
            sweeper::quick_sweep()
        } else {
            sweeper::programming_sweep()
        };
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

            println!(
                "\n    {} Sweep total: {} prompts, {} in {}ms\n",
                "📊".bold(),
                results.len(),
                total_usage.usage_string(),
                results.iter().map(|r| r.duration_ms).sum::<u64>()
            );
        }
    }
    Ok(())
}

// ── Compare ──────────────────────────────────────────────────────────

async fn run_compare(
    model_id: String,
    api: String,
    base_url: Option<String>,
    api_key: Option<String>,
    provider: String,
    max_tokens: u32,
    prompt_name: String,
    config_dir: Option<&Path>,
) -> Result<()> {
    let model = if let Some(dir) = config_dir {
        resolve_model(Some(&model_id), Some(dir))?
    } else {
        build_model(&model_id, &api, &base_url, &api_key, &provider, max_tokens, 65536)
    };
    let client = AiClient::new(model.clone())?;

    let test_prompts = prompts::all_prompts();
    let tp = test_prompts
        .iter()
        .find(|p| p.name == prompt_name)
        .ok_or_else(|| {
            anyhow::anyhow!("Prompt '{prompt_name}' not found. Use `list-prompts` to see available.")
        })?;

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

// ── Sweep helpers ────────────────────────────────────────────────────

async fn run_test(
    client: &AiClient,
    messages: &[Message],
    params: &GenerateParams,
    prompt_name: &str,
) -> Result<SweepResult> {
    let start = Instant::now();

    match client.chat(messages, params).await {
        Ok(completion) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            Ok(SweepResult {
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
            Ok(SweepResult {
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
    model_id: &str,
    api: &str,
    base_url: &Option<String>,
    api_key: &Option<String>,
    provider: &str,
    max_tokens: u32,
    context_window: u32,
) -> Model {
    let api_type = match api.to_lowercase().as_str() {
        "anthropic" | "anthropic-messages" => ApiType::AnthropicMessages,
        _ => ApiType::OpenAiCompletions,
    };
    let default_base_url = match api_type {
        ApiType::OpenAiCompletions => "https://api.openai.com",
        ApiType::AnthropicMessages => "https://api.anthropic.com",
    };
    let resolved_key = api_key
        .clone()
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
        supports_vision: false,
        cost_per_million_input: 0.0,
        cost_per_million_output: 0.0,
    }
}

fn filter_prompts(filter: Option<&str>) -> Vec<TestPrompt> {
    let all = prompts::all_prompts();
    match filter {
        Some(f) if !f.is_empty() => all.into_iter().filter(|p| p.name.contains(f)).collect(),
        _ => all,
    }
}

fn filter_sweeps<'a>(
    sweeps: &'a [sweeper::ParamSweep],
    filter: Option<&str>,
) -> Vec<&'a sweeper::ParamSweep> {
    match filter {
        Some(f) if !f.is_empty() => sweeps.iter().filter(|s| s.label.contains(f)).collect(),
        _ => sweeps.iter().collect(),
    }
}
