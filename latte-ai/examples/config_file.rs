//! 从配置文件加载模型 (YAML 或 TOML)
//!
//! 支持 YAML (推荐) 和 TOML (兼容) 两种格式，自动按扩展名检测。
//! 支持 ${ENV_VAR} 环境变量展开。
//!
//! 运行:
//!   DEEPSEEK_API_KEY="sk-..." cargo run --example config_file -- models.yaml
//!   DEEPSEEK_API_KEY="sk-..." cargo run --example config_file -- models.toml

use std::path::Path;

use latte_ai::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models.yaml".to_string());

    let models = load_models(&config_path)?;
    if models.is_empty() {
        anyhow::bail!("config file {} has no models", config_path);
    }

    println!("Loaded {} models from {}\n", models.len(), config_path);
    for m in &models {
        println!("  {} ({})", m.name, m.id);
        println!("    provider: {}  base_url: {}  reasoning: {}", m.provider, m.base_url, m.supports_thinking);
    }

    // Use the first model
    let model = models.into_iter().next().unwrap();
    let client = AiClient::new(model)?;
    let params = GenerateParams::code_defaults();

    let completion = client.chat(
        &[Message { role: Role::User, content: "说一句鼓励程序员的话".into() }],
        &params,
    ).await?;

    println!("\n{}", completion.content);
    println!("\n用量: {}", completion.usage);

    Ok(())
}

// ── 配置加载 ────────────────────────────────────────────

fn load_models(path: &str) -> anyhow::Result<Vec<Model>> {
    let content = std::fs::read_to_string(path)?;
    let ext = Path::new(path).extension().and_then(|e| e.to_str());

    #[derive(serde::Deserialize)]
    struct Config {
        models: Vec<ModelEntry>,
    }

    #[derive(serde::Deserialize)]
    struct ModelEntry {
        id: String,
        api: String,
        base_url: String,
        name: Option<String>,
        #[allow(dead_code)]
        description: Option<String>,
        provider: Option<String>,
        api_key: Option<String>,
        context_window: Option<u32>,
        max_tokens: Option<u32>,
        reasoning: Option<bool>,
        cost_per_million_input: Option<f64>,
        cost_per_million_output: Option<f64>,
    }

    let cfg: Config = match ext {
        Some("yaml" | "yml") => serde_yaml::from_str(&content)?,
        _ => toml::from_str(&content)?,
    };

    Ok(cfg.models.iter().map(|e| {
        let api = match e.api.as_str() {
            "anthropic" | "anthropic-messages" => ApiType::AnthropicMessages,
            _ => ApiType::OpenAiCompletions,
        };
        Model {
            id: e.id.clone(),
            name: e.name.clone().unwrap_or_else(|| e.id.clone()),
            api,
            provider: e.provider.clone().unwrap_or_else(|| "custom".into()),
            base_url: e.base_url.clone(),
            api_key: resolve_env(&e.api_key.clone().unwrap_or_default()),
            context_window: e.context_window.unwrap_or(65536),
            max_tokens: e.max_tokens.unwrap_or(4096),
            supports_thinking: e.reasoning.unwrap_or(false)
                || matches!(api, ApiType::AnthropicMessages),
            cost_per_million_input: e.cost_per_million_input.unwrap_or(0.0),
            cost_per_million_output: e.cost_per_million_output.unwrap_or(0.0),
        }
    }).collect())
}

fn resolve_env(value: &str) -> String {
    if value.starts_with("${") && value.ends_with('}') {
        let var = &value[2..value.len() - 1];
        std::env::var(var).unwrap_or_default()
    } else {
        value.to_string()
    }
}
