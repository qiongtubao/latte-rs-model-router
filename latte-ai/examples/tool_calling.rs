//! 多轮工具调用示例
//!
//! 演示完整链路：定义工具 → 发送请求 → 拿到 tool_calls → 假装执行工具
//! → 把 assistant 自己发起的 tool_calls 和 tool_result 消息都喂回
//! history → 让模型基于工具结果生成最终回复。
//!
//! 运行:
//!   DEEPSEEK_API_KEY="sk-..." cargo run --example tool_calling

use latte_ai::prelude::*;

/// 演示用的工具：查天气。实际工程里这里会接真 API / 数据库。
fn fake_get_weather(city: &str) -> String {
    match city {
        "北京" | "Beijing" => "晴，25°C，湿度 40%".to_string(),
        "上海" | "Shanghai" => "多云，22°C，湿度 65%".to_string(),
        "深圳" | "Shenzhen" => "阵雨，28°C，湿度 80%".to_string(),
        other => format!("{other}: 数据暂缺"),
    }
}

/// 演示用的工具：摄氏 → 华氏。
fn fake_c_to_f(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

#[tokio::main]
async fn main() -> Result<()> {
    // ── 1. 准备 client + tools ───────────────────────────────────
    let model = Model {
        id: "deepseek-chat".into(),
        name: "DeepSeek Chat".into(),
        api: ApiType::OpenAiCompletions,
        provider: "deepseek".into(),
        base_url: "https://api.deepseek.com".into(),
        api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
        context_window: 65536,
        max_tokens: 4096,
        supports_thinking: false,
        supports_vision: false,
        cost_per_million_input: 0.0,
        cost_per_million_output: 0.0,
        timeout_secs: None,
    };
    let client = AiClient::new(model)?;

    // 工具列表：每个 Tool 的 parameters 是 JSON Schema。
    let tools = vec![
        Tool {
            name: "get_weather".into(),
            description: Some("查询指定城市的天气".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如 北京/上海/深圳"}
                },
                "required": ["city"]
            }),
            strict: None,
        },
        Tool {
            name: "celsius_to_fahrenheit".into(),
            description: Some("把摄氏温度转成华氏".into()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "celsius": {"type": "number", "description": "摄氏温度值"}
                },
                "required": ["celsius"]
            }),
            strict: None,
        },
    ];

    let mut params = GenerateParams::code_defaults();
    params.tools = tools;

    // ── 2. 第一轮：用户提问，模型决定要不要调工具 ──────────────
    let mut history = vec![Message::user("北京今天多少度？顺便帮我转成华氏。")];

    println!(">>> 用户: 北京今天多少度？顺便帮我转成华氏。\n");
    let completion = client.chat(&history, &params).await?;

    if completion.tool_calls.is_empty() {
        // 模型直接答了，没调工具 —— 也行，演示就退出。
        println!("<<< 模型直接回答:\n{}\n", completion.content);
        return Ok(());
    }

    // ── 3. 模型要调工具：把它的回复（带 tool_calls）存档 ───────
    //
    // 这一步很关键：assistant 自己发起的 tool_calls 必须随 message
    // 一起回传给 model，否则下一轮的 tool_result 找不到对应 id。
    // 用 `Message::assistant_with_tool_calls` 一行搞定。
    println!("<<< 模型请求调用 {} 个工具:", completion.tool_calls.len());
    for tc in &completion.tool_calls {
        println!("    - {}({})  id={}", tc.name, tc.arguments, tc.id);
    }
    history.push(Message::assistant_with_tool_calls("", completion.tool_calls.clone()));

    // ── 4. 假装执行工具，把结果作为 Role::Tool 消息塞回 ──────
    for tc in &completion.tool_calls {
        let result = match tc.name.as_str() {
            "get_weather" => {
                let city = tc.arguments["city"].as_str().unwrap_or("");
                fake_get_weather(city)
            }
            "celsius_to_fahrenheit" => {
                let c = tc.arguments["celsius"].as_f64().unwrap_or(0.0);
                fake_c_to_f(c).to_string()
            }
            other => format!("[unknown tool: {other}]"),
        };
        println!("    → 执行 {} 返回: {}", tc.name, result);
        history.push(Message::tool_result(&tc.id, result));
    }
    println!();

    // ── 5. 第二轮：模型基于工具结果生成最终回复 ──────────────
    let final_completion = client.chat(&history, &params).await?;
    println!("<<< 模型最终回答:\n{}", final_completion.content);
    println!("\n用量: {}", final_completion.usage);

    Ok(())
}
