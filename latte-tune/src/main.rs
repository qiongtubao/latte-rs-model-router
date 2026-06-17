use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use colored::*;
use latte_ai::models::{ApiType, Message, Model, Role, StreamEvent, TokenUsage};
use latte_ai::params::GenerateParams;
use latte_ai::AiClient;
use latte_ai::vendor::{VendorId, VendorRegistry};
use latte_ai::vendor_toml::registry_from_toml_str;
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
        /// Model ID (e.g. "deepseek-chat", "claude-sonnet-4-20250514"). Optional when --config is used.
        model: Option<String>,

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

        /// Config file with model definitions (YAML or TOML)
        #[arg(long)]
        config: Option<PathBuf>,
    },

    /// List configured models
    ListModels {
        /// Config file (auto-discovered if not specified)
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

        /// Config file (auto-discovered if not specified)
        #[arg(long)]
        config: Option<PathBuf>,

        /// Disable streaming output
        #[arg(long)]
        no_stream: bool,

        /// Force interactive REPL mode (default if no prompt and stdin is a terminal)
        #[arg(short = 'r', long)]
        repl: bool,
    },
    /// Manage vendor configs (list / status / refresh / discover)
    Vendors {
        #[command(subcommand)]
        action: VendorsAction,

        /// Config file (auto-discovered if not specified)
        #[arg(long, global = true)]
        config: Option<PathBuf>,
    },
}

/// `latte-tune vendors` 子命令的二级动作
#[derive(Subcommand)]
enum VendorsAction {
    /// 列出所有 vendor
    List,
    /// 查看单个 vendor 的健康 + token 状态
    Status { id: String },
    /// 强制刷新 vendor token
    Refresh { id: String },
    /// 拉 vendor 的 model 列表
    Discover { id: String },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::ListModels { config } => list_models(config),
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
        Commands::Chat { prompt, model, config, no_stream, repl } => {
            run_chat(prompt, model, config, no_stream, repl).await
        }
        Commands::Vendors { action, config } => run_vendors(action, config).await,
    }
}
// ── Commands ───────────────────────────────────────────────────────────

fn list_models(config: Option<PathBuf>) -> anyhow::Result<()> {
    let models = load_models_resolved(config.as_ref())?;
    if models.is_empty() {
        println!("No models configured. Create ~/.latte/models.yaml");
        return Ok(());
    }
    print_model_list(&models, None);
    Ok(())
}

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

async fn run_chat(
    prompt: Option<String>, model_filter: Option<String>,
    config: Option<PathBuf>, no_stream: bool, force_repl: bool,
) -> anyhow::Result<()> {
    use std::io::{self, BufRead, IsTerminal, Read, Write};

    // Decide mode: one-shot vs REPL
    let is_piped = !std::io::stdin().is_terminal();
    let enter_repl = force_repl || (prompt.is_none() && !is_piped);

    if enter_repl {
        // ── Interactive REPL ──────────────────────────────────────────
        // Load all models from config — used for /model switching and /models listing
        let all_models = load_models_resolved(config.as_ref())?;
        if all_models.is_empty() {
            anyhow::bail!("No models found in config.");
        }
        // Resolve which model to start with
        let model_idx = if let Some(ref filter) = model_filter {
            find_model_index(&all_models, filter)
                .ok_or_else(|| anyhow::anyhow!("Model '{}' not found. Use /models to list.", filter))?
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
        eprintln!("{}", format!("║  {} models  /exit  /clear  /model  /models║", all_models.len()).dimmed());
        eprintln!("{}", "╚══════════════════════════════════════════╝".dimmed());

        let mut stdin = io::BufReader::new(io::stdin());
        loop {
            // Prompt
            eprint!("\n{} ", "▶".cyan().bold());
            io::stderr().flush()?;

            let mut line = String::new();
            if stdin.read_line(&mut line)? == 0 {
                break; // Ctrl+D
            }
            let input = line.trim().to_string();
            if input.is_empty() { continue; }

            // Commands
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
                            // No arg: show list
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
            // Add user message to history, then call API
            history.push(Message { role: Role::User, content: input });

            eprint!("\n{} ", "◀".green().bold());
            io::stderr().flush()?;

            if no_stream {
                let completion = client.chat(&history, &params).await?;
                println!("{}", completion.content);
                history.push(Message { role: Role::Assistant, content: completion.content.clone() });
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
                            if let Some(u) = u { usage = u; }
                        }
                        StreamEvent::Done { usage: u, .. } => { usage = u; }
                        StreamEvent::Error(e) => eprintln!("\n  {}", e.to_string().red()),
                    }
                }
                history.push(Message { role: Role::Assistant, content: full });
                eprintln!("\n  {}", usage);
            }
        }
        eprintln!("\n  {}", "Goodbye!".dimmed());
    } else {
        // ── One-shot mode ──────────────────────────────────────────────
        let prompt_text = match prompt {
            Some(p) if !p.is_empty() => p,
            _ => {
                let mut buf = String::new();
                io::stdin().read_to_string(&mut buf)?;
                if buf.trim().is_empty() {
                    anyhow::bail!("No prompt provided. Pass it as an argument, pipe via stdin, or run without args for REPL.");
                }
                buf
            }
        };

        let model = resolve_model(model_filter.as_deref(), config.as_ref())?;
        eprintln!("  Model: {} ({})", model.name.bold().cyan(), model.id.dimmed());

        let client = AiClient::new(model)?;
        let params = GenerateParams::code_defaults();
        let messages = vec![Message { role: Role::User, content: prompt_text }];

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
                        if let Some(u) = u { usage = u; }
                    }
                    StreamEvent::Done { usage: u, .. } => { usage = u; }
                    StreamEvent::Error(e) => eprintln!("\n  {}", e.to_string().red()),
                }
            }
            eprintln!("\n\n  {}", usage);
        }
    }

    Ok(())
}

/// Resolve which model to use: config or CLI args.
fn resolve_model(filter: Option<&str>, config: Option<&PathBuf>) -> anyhow::Result<Model> {
    let models = load_models_resolved(config)?;
    if models.is_empty() {
        anyhow::bail!("Config file has no models defined.");
    }
    if let Some(filter) = filter {
        find_model_index(&models, filter)
            .map(|i| models[i].clone())
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in config. Use --help to list.", filter))
    } else {
        Ok(models.into_iter().next().unwrap())
    }
}

/// Load all models from config.
///
/// If --config is specified, load only that file.
/// Otherwise, merge global config (~/.latte/models.yaml) as base
/// with project-local config overrides (by id).
fn load_models_resolved(config: Option<&PathBuf>) -> anyhow::Result<Vec<Model>> {
    if let Some(path) = config {
        return load_models_from_config(path);
    }

    // Build merged model map: global base + project overrides
    let mut map: std::collections::HashMap<String, Model> = std::collections::HashMap::new();

    // 1. Load global config as base
    for ext in &["yaml", "yml", "json", "toml"] {
        let p = dot_config_path(ext);
        if p.exists() {
            eprintln!("  Global config: {}", p.display());
            for m in load_models_from_config(&p)? {
                map.insert(m.id.clone(), m);
            }
            break;
        }
    }
    if map.is_empty() {
        for ext in &["yaml", "yml", "json", "toml"] {
            let p = xdg_config_path(ext);
            if p.exists() {
                eprintln!("  Global config: {}", p.display());
                for m in load_models_from_config(&p)? {
                    map.insert(m.id.clone(), m);
                }
                break;
            }
        }
    }

    // 2. Load project-local config as override
    for candidate in &[
        "latte.yaml", "latte.yml", "latte.json",
        "models.yaml", "models.yml", "models.json",
        "latte.toml", "models.toml",
    ] {
        let p = PathBuf::from(candidate);
        if p.exists() {
            eprintln!("  Project config: {}", p.display());
            for m in load_models_from_config(&p)? {
                map.insert(m.id.clone(), m); // override by id
            }
            break;
        }
    }

    if map.is_empty() {
        anyhow::bail!("No config found. Create ~/.latte/models.yaml or use --config.");
    }

    // Sort by provider then name for stable order
    let mut models: Vec<Model> = map.into_values().collect();
    models.sort_by(|a, b| a.provider.cmp(&b.provider).then(a.name.cmp(&b.name)));
    Ok(models)
}


/// Find model by id/name substring or numeric index (1-based).
fn find_model_index(models: &[Model], query: &str) -> Option<usize> {
    // Try numeric index first (1-based)
    if let Ok(n) = query.parse::<usize>() {
        if n >= 1 && n <= models.len() {
            return Some(n - 1);
        }
    }
    // Substring match on id or name
    let q = query.to_lowercase();
    models.iter().position(|m| m.id.to_lowercase().contains(&q) || m.name.to_lowercase().contains(&q))
}

/// Print model list grouped by provider.
fn print_model_list(models: &[Model], current: Option<usize>) {
    use std::collections::BTreeMap;
    // Group by provider
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
async fn run_sweep(
    model_id: Option<String>, api: String, base_url: Option<String>, api_key: Option<String>,
    provider: String, max_tokens: u32, context_window: u32, quick: bool,
    prompt_filter: Option<String>, sweep_filter: Option<String>,
    config: Option<PathBuf>,
) -> anyhow::Result<()> {
    let models = if let Some(config_path) = config {
        load_models_from_config(&config_path)?
    } else {
        match load_models_resolved(None) {
            Ok(models) if !models.is_empty() => models,
            _ => {
                let id = model_id.ok_or_else(||
                    anyhow::anyhow!("No model specified. Provide a model ID or create ~/.latte/models.yaml.")
                )?;
                vec![build_model(&id, &api, &base_url, &api_key, &provider, max_tokens, context_window)]
            }
        }
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
    struct ModelConfig {
        models: Vec<ModelEntry>,
    }

    #[derive(serde::Deserialize)]
    struct ModelEntry {
        // ── 必填字段 ──────────────────────────────
        /// 模型标识符 (e.g. "deepseek-chat")
        id: String,
        /// API 类型: "openai" | "anthropic"
        api: String,
        /// API 端点地址
        base_url: String,

        // ── 可选: 基本信息 ─────────────────────────
        /// 人类可读的名称 (默认同 id)
        name: Option<String>,
        #[allow(dead_code)]
        description: Option<String>,
        /// 提供商名称 (e.g. "deepseek", "anthropic")
        provider: Option<String>,

        // ── 可选: 认证 ────────────────────────────
        /// API 密钥，支持 ${ENV_VAR} 展开
        api_key: Option<String>,

        // ── 可选: 容量 ────────────────────────────
        /// 上下文窗口大小 (token)，默认 65536
        context_window: Option<u32>,
        /// 最大输出 token 数，默认 4096
        max_tokens: Option<u32>,

        // ── 可选: 推理能力 ─────────────────────────
        /// 是否支持思考/推理 (默认: anthropic API 自动为 true)
        reasoning: Option<bool>,

        // ── 可选: 成本 ────────────────────────────
        /// 每百万输入 token 成本 (USD)
        cost_per_million_input: Option<f64>,
        /// 每百万输出 token 成本 (USD)
        cost_per_million_output: Option<f64>,
    }

    // Auto-detect format by file extension
    let cfg: ModelConfig = match path.extension().and_then(|e| e.to_str()) {
        Some("yaml" | "yml") => serde_yaml::from_str(&content)?,
        Some("json") => serde_json::from_str(&content)?,
        _ => toml::from_str(&content)?,
    };

    let mut models = Vec::new();
    for entry in &cfg.models {
        let api_type = match entry.api.to_lowercase().as_str() {
            "anthropic" | "anthropic-messages" => ApiType::AnthropicMessages,
            _ => ApiType::OpenAiCompletions,
        };
        let resolved_key = resolve_env(&entry.api_key.clone().unwrap_or_default());
        let raw_key = entry.api_key.as_deref().unwrap_or("");

        // Warn if API key resolved to empty (env var not set)
        if resolved_key.is_empty() && raw_key.starts_with("${") && raw_key.ends_with('}')
            && raw_key != "ollama"
        {
            let var_name = &raw_key[2..raw_key.len() - 1];
            eprintln!(
                "  {} {}: {} — set with: export {}=\"...\"",
                "⚠".yellow(),
                entry.id,
                format!("API key env var ${} not set", var_name).red(),
                var_name,
            );
        }

        models.push(Model {
            id: entry.id.clone(),
            name: entry.name.clone().unwrap_or_else(|| entry.id.clone()),
            api: api_type,
            provider: entry.provider.clone().unwrap_or_else(|| "custom".into()),
            base_url: resolve_env(&entry.base_url),
            api_key: resolved_key,
            context_window: entry.context_window.unwrap_or(65536),
            max_tokens: entry.max_tokens.unwrap_or(4096),
            supports_thinking: entry.reasoning.unwrap_or(false)
                || api_type == ApiType::AnthropicMessages,
            cost_per_million_input: entry.cost_per_million_input.unwrap_or(0.0),
            cost_per_million_output: entry.cost_per_million_output.unwrap_or(0.0),
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


/// `~/.latte/models.{ext}`
fn dot_config_path(ext: &str) -> PathBuf {
    let filename = format!("models.{}", ext);
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".latte").join(&filename)
    } else {
        PathBuf::from(".latte").join(&filename)
    }
}

/// `$XDG_CONFIG_HOME/latte/models.{ext}` or `~/.config/latte/models.{ext}`
fn xdg_config_path(ext: &str) -> PathBuf {
    let filename = format!("models.{}", ext);
    if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(dir).join("latte").join(&filename)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".config").join("latte").join(&filename)
    } else {
        PathBuf::from(".config/latte").join(&filename)
    }
}
// ── Vendors 子命令 ─────────────────────────────────────────────────

/// 加载 vendor registry：显式 --config → cwd `vendors.toml` → `~/.latte/vendors.toml`
fn load_vendors_registry(config: Option<&std::path::Path>) -> anyhow::Result<VendorRegistry> {
    use anyhow::Context;
    let candidates: Vec<std::path::PathBuf> = match config {
        Some(p) => vec![p.to_path_buf()],
        None => {
            let cwd = std::path::PathBuf::from("vendors.toml");
            let home = std::env::var_os("HOME")
                .map(|h| std::path::PathBuf::from(h).join(".latte").join("vendors.toml"));
            let mut v = vec![cwd];
            if let Some(h) = home {
                v.push(h);
            }
            v
        }
    };
    for path in &candidates {
        if path.exists() {
            let s = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            return registry_from_toml_str(&s)
                .with_context(|| format!("parse {}", path.display()));
        }
    }
    anyhow::bail!(
        "no vendors.toml found (tried: {})",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// `latte-tune vendors` 入口
async fn run_vendors(action: VendorsAction, config: Option<PathBuf>) -> anyhow::Result<()> {
    let reg = load_vendors_registry(config.as_deref())?;
    match action {
        VendorsAction::List => vendors_print_list(&reg),
        VendorsAction::Status { id } => vendors_print_status(&reg, &id).await,
        VendorsAction::Refresh { id } => vendors_print_refresh(&reg, &id).await,
        VendorsAction::Discover { id } => vendors_print_discover(&reg, &id).await,
    }
}

fn vendors_print_list(reg: &VendorRegistry) -> anyhow::Result<()> {
    let vendors = reg.list();
    if vendors.is_empty() {
        println!("{}", "No vendors configured.".yellow());
        return Ok(());
    }
    println!("{}", format!("Configured vendors ({}):", vendors.len()).bold());
    for v in vendors {
        let disabled = if v.disabled_features.is_empty() {
            "-".dimmed().to_string()
        } else {
            v.disabled_features
                .iter()
                .map(|f| format!("{:?}", f).to_lowercase())
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "  {} {} {} disabled=[{}]",
            v.id.as_str().green().bold(),
            v.base_url.dimmed(),
            v.auth.kind().cyan(),
            disabled
        );
    }
    Ok(())
}

async fn vendors_print_status(reg: &VendorRegistry, id: &str) -> anyhow::Result<()> {
    let vid = VendorId::new(id);
    let s = reg
        .status(&vid)
        .await
        .map_err(|e| anyhow::anyhow!("status({id}) failed: {e}"))?;
    let token_remaining = match s.token_remaining_secs {
        Some(secs) => format!("{secs}s"),
        None => "never expires".to_string(),
    };
    let health_latency = match s.health_latency_ms {
        Some(ms) => format!("{ms}ms"),
        None => "n/a".to_string(),
    };
    println!("{}", format!("Vendor: {}", s.vendor.as_str()).bold());
    println!("  auth_kind:    {}", s.auth_kind.cyan());
    println!("  auth_valid:   {}", if s.auth_valid { "yes".green() } else { "no".red() });
    println!("  token_left:   {token_remaining}");
    println!("  health:       {} ({})", if s.health_ok { "ok".green() } else { "fail".red() }, health_latency);
    if let Some(n) = s.discovered_models {
        println!("  models:       {n}");
    }
    Ok(())
}

async fn vendors_print_refresh(reg: &VendorRegistry, id: &str) -> anyhow::Result<()> {
    let vid = VendorId::new(id);
    let token = reg
        .refresh_now(&vid)
        .await
        .map_err(|e| anyhow::anyhow!("refresh({id}) failed: {e}"))?;
    let masked = if token.len() > 8 {
        format!("{}…{}", &token[..4], &token[token.len() - 4..])
    } else {
        token.clone()
    };
    println!("{} refreshed token: {}", "✓".green(), masked.cyan());
    Ok(())
}

async fn vendors_print_discover(reg: &VendorRegistry, id: &str) -> anyhow::Result<()> {
    let vid = VendorId::new(id);
    let models = reg
        .discover_models(&vid)
        .await
        .map_err(|e| anyhow::anyhow!("discover({id}) failed: {e}"))?;
    if models.is_empty() {
        println!("{}", "No models discovered.".yellow());
        return Ok(());
    }
    println!(
        "{}",
        format!("Discovered {} model(s) from {}:", models.len(), id).bold()
    );
    for m in &models {
        let ctx = match m.context_window {
            Some(c) => format!("ctx={c}"),
            None => "-".to_string(),
        };
        println!("  {} {} {}", m.id.green(), m.display_name.dimmed(), ctx.dimmed());
    }
    Ok(())
}
