//! 流式对话示例
//!
//! 运行:
//!   DEEPSEEK_API_KEY="sk-..." cargo run --example streaming

use latte_ai::prelude::*;
#[tokio::main]
async fn main() -> Result<()> {
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
        cost_per_million_input: 0.0,
        cost_per_million_output: 0.0,
    };

    let client = AiClient::new(model)?;
    let params = GenerateParams {
        temperature: Some(0.1),
        max_tokens: Some(2048),
        ..Default::default()
    };

    println!(">>> 流式输出开始...\n");

    let mut stream = client.chat_stream(
        &[Message { role: Role::User, content: "用 Rust 实现斐波那契数列，逐行讲解".into() }],
        &params,
    ).await?;

    let mut full_content = String::new();
    while let Some(event) = stream.recv().await {
        match event {
            StreamEvent::Delta { content, .. } => {
                print!("{}", content);
                full_content.push_str(&content);
            }
            StreamEvent::Done { usage, .. } => {
                println!("\n\n<<< 完成");
                println!("用量: {}", usage);
            }
            StreamEvent::Error(e) => {
                eprintln!("\n<<< 错误: {}", e);
            }
        }
    }

    Ok(())
}
