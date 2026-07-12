//! 基础非流式对话示例
//!
//! 运行:
//!   DEEPSEEK_API_KEY="sk-..." cargo run --example basic

use latte_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // ── 方式一：代码中构建 Model ──────────────────────────────
    let model = Model {
        id: "deepseek-chat".into(),
        name: "DeepSeek Chat".into(),
        api: ApiType::OpenAiCompletions,
        provider: "deepseek".into(),
        base_url: "https://api.deepseek.com".into(),
        api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
        context_window: 65536,
        max_tokens: 8192,
        supports_thinking: false,
        supports_vision: false,
        cost_per_million_input: 0.0,
        cost_per_million_output: 0.0,
    };

    let client = AiClient::new(model)?;

    // ── 方式一：用便捷预设参数 ─────────────────────────────────
    let params = GenerateParams::code_defaults();
    let completion = client
        .chat(&[Message::user("用 Rust 写一个求和函数")], &params)
        .await?;
    println!("=== code_defaults ===\n{}\n", completion.content);
    println!(
        "用量: {} input, {} output tokens\n",
        completion.usage.input_tokens, completion.usage.output_tokens
    );

    // ── 方式二：完全手动指定参数 ──────────────────────────────
    let custom_params = GenerateParams {
        temperature: Some(0.3),
        top_p: Some(0.95),
        min_p: Some(0.02),
        max_tokens: Some(2048),
        ..Default::default()
    };
    let completion = client
        .chat(
            &[Message::user("用三句话解释 Rust 的所有权")],
            &custom_params,
        )
        .await?;
    println!("=== custom_params ===\n{}\n", completion.content);
    println!("用量: {}", completion.usage);

    // ── 方式三：使用系统提示 (System Prompt) ──────────────────
    let params = GenerateParams::code_defaults();
    let completion = client
        .chat(
            &[
                Message::system("你是一个 Rust 专家，回答简洁，使用英文变量名。"),
                Message::user("写一个二分查找函数"),
            ],
            &params,
        )
        .await?;
    println!("=== with_system_prompt ===\n{}\n", completion.content);
    println!("用量: {}", completion.usage);

    Ok(())
}
