use std::time::Duration;

use futures_util::StreamExt;
use reqwest::Client as HttpClient;
use reqwest_eventsource::{Event, RequestBuilderExt};
use tracing::{debug, warn};

use crate::error::{AiError, Result};
use crate::models::*;
use crate::params::GenerateParams;

/// `LATTE_AI_DEBUG_HTTP=1` 启用：把 `chat_openai` / `chat_anthropic` 实际
/// 发的 request body 和失败时的 response 原始字节打到 stderr。
///
/// 用途：定位 vendor 集成 bug —— 当 `test` 命令报 "error decoding
/// response body" 这种**无法**从错误字符串反推的错时，开这个开关能
/// 直接看到 "AiClient 发了什么" 和 "vendor 回了什么字节序列"。
///
/// 设计：默认关（零成本），用 env var 触发（不需要改 config 也不需要
/// 重启 binary 之外的依赖）。打印走 eprintln，**不**走 tracing：
/// 普通 `RUST_LOG=info` 用户不会被噪音打到；想用的人显式开。
fn debug_http_enabled() -> bool {
    std::env::var("LATTE_AI_DEBUG_HTTP")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}


/// A client for interacting with AI models via OpenAI-compatible or Anthropic APIs.
#[derive(Clone)]
pub struct AiClient {
    http: HttpClient,
    model: Model,
}

impl std::fmt::Debug for AiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AiClient").field("model", &self.model).finish()
    }
}

impl AiClient {
    /// Create a new client for the given model.
    pub fn new(model: Model) -> Result<Self> {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self { http, model })
    }

    // ── public API ─────────────────────────────────────────────

    /// Send a non-streaming chat completion request.
    pub async fn chat(&self, messages: &[Message], params: &GenerateParams) -> Result<Completion> {
        self.check_api_key()?;
        match self.model.api {
            ApiType::OpenAiCompletions => self.chat_openai(messages, params).await,
            ApiType::AnthropicMessages => self.chat_anthropic(messages, params).await,
        }
    }

    /// Send a streaming chat completion request.
    ///
    /// Returns a receiver that yields `StreamEvent` values as they arrive.
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        params: &GenerateParams,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        self.check_api_key()?;
        match self.model.api {
            ApiType::OpenAiCompletions => self.stream_openai(messages, params).await,
            ApiType::AnthropicMessages => self.stream_anthropic(messages, params).await,
        }
    }

    /// Reject calls when the resolved `api_key` is blank. Without this guard the
    /// HTTP client would send an empty `Authorization: Bearer ` / `x-api-key: `
    /// header and the vendor would reject the request with a 401 — which is a
    /// configuration mistake on the caller side, not a transient failure.
    fn check_api_key(&self) -> Result<()> {
        if self.model.api_key.trim().is_empty() {
            return Err(AiError::Config(format!(
                "model '{}' has no api_key configured; set it in your global \
                 config (~/.latte/models.yaml), project config \
                 (config/models.toml), --api-key flag, or the corresponding \
                 ${{ENV_VAR}} in api_key",
                self.model.id
            )));
        }
        Ok(())
    }

    // ── OpenAI chat completions (non-streaming) ─────────────────────────

    async fn chat_openai(
        &self,
        messages: &[Message],
        params: &GenerateParams,
    ) -> Result<Completion> {
        let url = format!(
            "{}/chat/completions",
            self.model.base_url.trim_end_matches('/')
        );

        let req = self.build_openai_request(messages, params, false);
        debug!(url, model = %req.model, "OpenAI request");

        // Debug 钩子：发包前打印 body。`LATTE_AI_DEBUG_HTTP=1` 启用。
        // 排查 vendor 集成 bug 时用 —— `--raw` 看不到 AiClient 实际发的字段。
        if debug_http_enabled() {
            if let Ok(body_str) = serde_json::to_string(&req) {
                eprintln!("[latte_ai debug] POST {} (model={})", url, req.model);
                eprintln!("[latte_ai debug] body: {}", body_str);
            }
        }

        let resp = self.http
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.model.api_key))
            .json(&req)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    AiError::Other("Request timed out".into())
                } else if e.is_connect() {
                    AiError::Other(format!("Connection failed: {}", e))
                } else {
                    AiError::Http(e)
                }
            })?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(if status.as_u16() == 429 {
                AiError::RateLimited { retry_after: 10.0, message: body }
            } else if status.as_u16() == 401 {
                AiError::Auth(body)
            } else {
                AiError::Api { status: status.as_u16(), message: body }
            });
        }

        // `resp.bytes().await` 拿走了 resp，所以**先 clone headers**才能在错时打。
        let headers = resp.headers().clone();
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                if debug_http_enabled() {
                    eprintln!("[latte_ai debug] body read failed: {e}");
                    eprintln!("[latte_ai debug] status: {}", status);
                    eprintln!("[latte_ai debug] content-type: {:?}", headers.get("content-type"));
                    eprintln!("[latte_ai debug] content-encoding: {:?}", headers.get("content-encoding"));
                    eprintln!("[latte_ai debug] transfer-encoding: {:?}", headers.get("transfer-encoding"));
                }
                return Err(AiError::Http(e));
            }
        };
        if debug_http_enabled() {
            eprintln!("[latte_ai debug] response status: {}", status);
            eprintln!("[latte_ai debug] body bytes: {} (first 400 below)", bytes.len());
            // utf-8 lossy —— vendor body 大概率是合法 JSON，但保底不 panic。
            let preview = String::from_utf8_lossy(&bytes[..bytes.len().min(400)]);
            eprintln!("[latte_ai debug] body preview: {}", preview);
        }
        let data: OpenAiChatResponse = match serde_json::from_slice(&bytes) {
            Ok(d) => d,
            Err(e) => {
                if debug_http_enabled() {
                    eprintln!("[latte_ai debug] JSON parse FAILED: {e}");
                    eprintln!("[latte_ai debug] full body dump:");
                    eprintln!("{}", String::from_utf8_lossy(&bytes));
                }
                return Err(AiError::Serde(e));
            }
        };
        let choice = data.choices.into_iter().next()
            .ok_or_else(|| AiError::Other("No choices in response".into()))?;
        let (parts, tool_calls) = extract_openai_response(choice.message.content, choice.message.tool_calls);
        let content = parts.iter()
            .filter_map(|p| match p { ContentPart::Text { text } => Some(text.clone()), _ => None })
            .collect::<Vec<_>>()
            .join("");
        Ok(Completion {
            content,
            content_parts: parts,
            tool_calls,
            stop_reason: choice.finish_reason.unwrap_or_default(),
            usage: TokenUsage {
                input_tokens: data.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                output_tokens: data.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
                thinking_tokens: 0,
            },
        })
    }

    // ── OpenAI chat completions (streaming) ─────────────────────────────

    async fn stream_openai(
        &self,
        messages: &[Message],
        params: &GenerateParams,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        let url = format!(
            "{}/chat/completions",
            self.model.base_url.trim_end_matches('/')
        );

        let req = self.build_openai_request(messages, params, true);
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let client = self.http.clone();
        let api_key = self.model.api_key.clone();

        tokio::spawn(async move {
            // Send streaming request
            let resp = match client
                .post(&url)
                .header("Authorization", format!("Bearer {api_key}"))
                .json(&req)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    tx.send(StreamEvent::Error(format!("Request failed: {e}"))).await.ok();
                    return;
                }
            };

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                // Fall back to non-streaming
                let mut ns = req.clone();
                ns.stream = false;
                match client
                    .post(&url)
                    .header("Authorization", format!("Bearer {api_key}"))
                    .json(&ns)
                    .send()
                    .await
                {
                    Ok(r) if r.status().is_success() => {
                        match r.json::<OpenAiChatResponse>().await {
                            Ok(data) => {
                                let content = merge_openai_text(
                                    data.choices.into_iter().next()
                                        .and_then(|c| c.message.content)
                                );
                                let usage = data.usage.map(|u| TokenUsage {
                                    input_tokens: u.prompt_tokens,
                                    output_tokens: u.completion_tokens,
                                    thinking_tokens: 0,
                                }).unwrap_or_default();
                                tx.send(StreamEvent::Delta { content: content.clone(), usage: Some(usage.clone()) }).await.ok();
                                tx.send(StreamEvent::Done { content, tool_calls: vec![], usage }).await.ok();
                            }
                            Err(e) => {
                                tx.send(StreamEvent::Error(format!("Parse error: {e}"))).await.ok();
                            }
                        }
                    }
                    Ok(r) => {
                        let b = r.text().await.unwrap_or_default();
                        tx.send(StreamEvent::Error(format!("Stream {status} — {body}; non-stream also failed: {b}"))).await.ok();
                    }
                    Err(e) => {
                        tx.send(StreamEvent::Error(format!("Stream {status} — {body}; fallback failed: {e}"))).await.ok();
                    }
                }
                return;
            }

            // Parse SSE manually from byte stream
            use futures_util::StreamExt;
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut full_text = String::new();
            let mut usage = TokenUsage::default();
            let mut tool_call_acc: Vec<ToolCallAccum> = Vec::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        // Stream broken mid-way — fall back to non-streaming
                        let mut ns = req.clone();
                        ns.stream = false;
                        if let Ok(r) = client
                            .post(&url)
                            .header("Authorization", format!("Bearer {api_key}"))
                            .json(&ns)
                            .send()
                            .await
                        {
                            if r.status().is_success() {
                                if let Ok(data) = r.json::<OpenAiChatResponse>().await {
                                    let content = merge_openai_text(
                                        data.choices.into_iter().next().and_then(|c| c.message.content)
                                    );
                                    let u = data.usage.map(|u| TokenUsage {
                                        input_tokens: u.prompt_tokens,
                                        output_tokens: u.completion_tokens,
                                        thinking_tokens: 0,
                                    }).unwrap_or_default();
                                    tx.send(StreamEvent::Delta { content: content.clone(), usage: Some(u.clone()) }).await.ok();
                                    tx.send(StreamEvent::Done { content, tool_calls: vec![], usage: u }).await.ok();
                                    return;
                                }
                            }
                        }
                        tx.send(StreamEvent::Error(format!("Stream broken: {e}"))).await.ok();
                        return;
                    }
                };
                buf.push_str(&String::from_utf8_lossy(&bytes));

                // Process complete SSE events (delimited by \n\n)
                while let Some(pos) = buf.find("\n\n") {
                    let event = buf[..pos].to_string();
                    buf = buf[pos + 2..].to_string();

                    for line in event.lines() {
                        if let Some(data) = line.strip_prefix("data: ") {
                            if data == "[DONE]" {
                                let final_calls = build_stream_tool_calls(&tool_call_acc);
                                tx.send(StreamEvent::Done {
                                    content: vec![ContentPart::Text { text: full_text.clone() }],
                                    tool_calls: final_calls,
                                    usage: usage.clone(),
                                }).await.ok();
                                return;
                            }
                            match serde_json::from_str::<OpenAiStreamChunk>(data) {
                                Ok(chunk) => {
                                    if let Some(choice) = chunk.choices.into_iter().next() {
                                        if let Some(delta) = choice.delta.content {
                                            full_text.push_str(&delta);
                                            tx.send(StreamEvent::Delta {
                                                content: vec![ContentPart::Text { text: delta }],
                                                usage: None,
                                            }).await.ok();
                                        }
                                        for td in choice.delta.tool_calls {
                                            let idx = td.index as usize;
                                            while tool_call_acc.len() <= idx {
                                                tool_call_acc.push(ToolCallAccum::default());
                                            }
                                            let entry = &mut tool_call_acc[idx];
                                            if let Some(id) = td.id {
                                                if !id.is_empty() { entry.id = id; }
                                            }
                                            if let Some(name) = td.function.as_ref().and_then(|f| f.name.as_ref()) {
                                                if !name.is_empty() { entry.name = name.clone(); }
                                            }
                                            if let Some(args) = td.function.as_ref().and_then(|f| f.arguments.as_ref()) {
                                                entry.arguments.push_str(args);
                                            }
                                        }
                                        if choice.finish_reason.is_some() {
                                            if let Some(u) = &chunk.usage {
                                                usage = TokenUsage {
                                                    input_tokens: u.prompt_tokens,
                                                    output_tokens: u.completion_tokens,
                                                    thinking_tokens: 0,
                                                };
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    warn!("SSE parse error: {e}");
                                }
                            }
                        }
                    }
                }
            }

            // Stream ended without [DONE] — send what we have
            let final_calls = build_stream_tool_calls(&tool_call_acc);
            tx.send(StreamEvent::Done {
                content: vec![ContentPart::Text { text: full_text }],
                tool_calls: final_calls,
                usage,
            }).await.ok();
        });

        Ok(rx)
    }
    // ── Anthropic messages API (non-streaming) ──────────────────────────

    async fn chat_anthropic(
        &self,
        messages: &[Message],
        params: &GenerateParams,
    ) -> Result<Completion> {
        let url = format!(
            "{}/v1/messages",
            self.model.base_url.trim_end_matches('/')
        );

        let req = self.build_anthropic_request(messages, params, false);
        debug!(url, model = %req.model, "Anthropic request");

        let resp = self.http
            .post(&url)
            .header("x-api-key", &self.model.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&req)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(if status.as_u16() == 429 {
                AiError::RateLimited { retry_after: 30.0, message: body }
            } else {
                AiError::Api { status: status.as_u16(), message: body }
            });
        }

        let data: AnthropicResponse = resp.json().await?;

        let (parts, tool_calls) = extract_anthropic_response(data.content);
        let content = parts.iter()
            .filter_map(|p| match p { ContentPart::Text { text } => Some(text.clone()), _ => None })
            .collect::<Vec<_>>()
            .join("");

        Ok(Completion {
            content,
            content_parts: parts,
            tool_calls,
            stop_reason: data.stop_reason.unwrap_or_default(),
            usage: TokenUsage {
                input_tokens: data.usage.input_tokens,
                output_tokens: data.usage.output_tokens,
                thinking_tokens: 0,
            },
        })
    }

    // ── Anthropic messages API (streaming) ──────────────────────────────

    async fn stream_anthropic(
        &self,
        messages: &[Message],
        params: &GenerateParams,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        let url = format!(
            "{}/v1/messages",
            self.model.base_url.trim_end_matches('/')
        );

        let req = self.build_anthropic_request(messages, params, true);
        let (tx, rx) = tokio::sync::mpsc::channel(128);
        let client = self.http.clone();
        let api_key = self.model.api_key.clone();

        tokio::spawn(async move {
            let es = match client
                .post(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .json(&req)
                .eventsource()
            {
                Ok(es) => es,
                Err(e) => {
                    tx.send(StreamEvent::Error(format!("Failed to open stream: {e}"))).await.ok();
                    return;
                }
            };

            let mut full_text = String::new();
            let mut usage = TokenUsage::default();
            let mut tool_use_accum: std::collections::HashMap<u32, (String, String, String)> =
                std::collections::HashMap::new();

            let mut es = es;
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => continue,
                    Ok(Event::Message(msg)) => {
                        match serde_json::from_str::<AnthropicStreamEvent>(&msg.data) {
                            Ok(evt) => {
                                match evt.type_.as_str() {
                                    "content_block_start" => {
                                        if let (Some(idx), Some(cb)) = (evt.index, evt.content_block.as_ref()) {
                                            if let AnthropicContentBlock::ToolUse { id, name, input: _ } = cb {
                                                let entry = tool_use_accum.entry(idx).or_insert_with(|| {
                                                    (id.clone(), name.clone(), String::new())
                                                });
                                                if entry.0.is_empty() { entry.0 = id.clone(); }
                                                if entry.1.is_empty() { entry.1 = name.clone(); }
                                            }
                                        }
                                    }
                                    "content_block_delta" => {
                                        if let Some(delta) = &evt.delta {
                                            if let Some(text) = &delta.text {
                                                full_text.push_str(text);
                                                tx.send(StreamEvent::Delta {
                                                    content: vec![ContentPart::Text { text: text.clone() }],
                                                    usage: None,
                                                }).await.ok();
                                            }
                                            if let (Some(idx), Some(chunk)) = (evt.index, delta.partial_json.as_ref()) {
                                                if let Some(entry) = tool_use_accum.get_mut(&idx) {
                                                    entry.2.push_str(chunk);
                                                }
                                            }
                                        }
                                    }
                                    "message_delta" => {
                                        if let Some(u) = &evt.usage {
                                            usage = TokenUsage {
                                                input_tokens: u.input_tokens,
                                                output_tokens: u.output_tokens,
                                                thinking_tokens: 0,
                                            };
                                        }
                                    }
                                    "message_stop" => {
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse Anthropic SSE: {e} body={}", msg.data);
                            }
                        }
                    }
                    Err(e) => {
                        tx.send(StreamEvent::Error(format!("Stream error: {e}"))).await.ok();
                        return;
                    }
                }
            }

            let mut final_calls: Vec<ToolCall> = Vec::new();
            let mut indices: Vec<u32> = tool_use_accum.keys().copied().collect();
            indices.sort();
            for idx in indices {
                if let Some((id, name, raw)) = tool_use_accum.remove(&idx) {
                    if id.is_empty() { continue; }
                    // 跟 OpenAI 流式一样：解析失败时把错误信息
                    // 写进 arguments_parse_error。
                    let (arguments, arguments_parse_error) =
                        match serde_json::from_str(&raw) {
                            Ok(v) => (v, None),
                            Err(e) => (serde_json::Value::Null, Some(e.to_string())),
                        };
                    final_calls.push(ToolCall {
                        id, name, arguments, arguments_raw: if raw.is_empty() { None } else { Some(raw) },
                        arguments_parse_error,
                    });
                }
            }
            tx.send(StreamEvent::Done {
                content: if full_text.is_empty() {
                    vec![]
                } else {
                    vec![ContentPart::Text { text: full_text }]
                },
                tool_calls: final_calls,
                usage,
            }).await.ok();
        });

        Ok(rx)
    }

    // ── Request builders ───────────────────────────────────────────────

    fn build_openai_request(
        &self,
        messages: &[Message],
        params: &GenerateParams,
        stream: bool,
    ) -> OpenAiChatRequest {
        OpenAiChatRequest {
            model: self.model.id.clone(),
            messages: messages.iter().map(|m| {
                // `Role::Tool` 的 content 必须是 string（OpenAI 协议要求）。
                let content = match m.role {
                    Role::Tool => OpenAiMessageContent::ToolString(
                        m.content.iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                    _ => OpenAiMessageContent::Parts(
                        m.content.iter().map(to_openai_content_part).collect(),
                    ),
                };
                OpenAiMessage {
                    role: match m.role {
                        Role::System => "system",
                        Role::User => "user",
                        Role::Assistant => "assistant",
                        Role::Tool => "tool",
                    }.into(),
                    content,
                    tool_call_id: m.tool_call_id.clone(),
                    tool_calls: m.tool_calls.as_ref().map(|calls| {
                        calls.iter().map(|c| OpenAiToolCall {
                            id: c.id.clone(),
                            type_: Some("function".into()),
                            function: OpenAiFunctionCall {
                                name: c.name.clone(),
                                arguments: c.arguments_raw.clone().unwrap_or_else(|| {
                                    serde_json::to_string(&c.arguments).unwrap_or_default()
                                }),
                            },
                        }).collect()
                    }),
                }
            }).collect(),
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            min_p: params.min_p,
            presence_penalty: params.presence_penalty,
            frequency_penalty: params.frequency_penalty,
            repetition_penalty: params.repetition_penalty,
            max_tokens: params.max_tokens,
            stop: params.stop_sequences.clone(),
            seed: params.seed,
            tools: params.tools.iter().map(|t| OpenAiTool {
                type_: "function".into(),
                function: OpenAiFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            }).collect(),
            tool_choice: if params.tools.is_empty() && params.tool_choice == ToolChoice::Auto {
                None
            } else {
                Some(OpenAiToolChoice::from(&params.tool_choice))
            },
            stream,
        }
    }

    fn build_anthropic_request(
        &self,
        messages: &[Message],
        params: &GenerateParams,
        _stream: bool,
    ) -> AnthropicRequest {
        let max_tokens = params.max_tokens.unwrap_or(self.model.max_tokens);
        let thinking = params.thinking_budget.map(|tb| AnthropicThinking {
            type_: "enabled".into(),
            budget_tokens: tb.token_budget(),
        });

        AnthropicRequest {
            model: self.model.id.clone(),
            // Anthropic 没有 tool role —— `Role::Tool` 转成 user role
            // 消息 + 一个 tool_result block。
            messages: messages.iter().map(|m| {
                let role = match m.role {
                    Role::System => "user",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "user",
                }.to_string();
                let content: Vec<AnthropicContentBlock> = match m.role {
                    Role::Tool => {
                        let tool_use_id = m.tool_call_id.clone().unwrap_or_default();
                        let body = m.content.iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        vec![AnthropicContentBlock::ToolResult {
                            tool_use_id,
                            content: body,
                            is_error: None,
                        }]
                    }
                    _ => m.content.iter().map(to_anthropic_content_block).collect(),
                };
                AnthropicMessage { role, content }
            }).collect(),
            max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            stop_sequences: params.stop_sequences.clone(),
            tools: params.tools.iter().map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            }).collect(),
            tool_choice: if params.tools.is_empty() && params.tool_choice == ToolChoice::Auto {
                None
            } else {
                AnthropicToolChoice::from(&params.tool_choice)
            },
            thinking,
        }
    }
}


/// Concatenate OpenAI response content into a single text string.
/// 同时支持 string 形态（minimax 等）和 array 形态（OpenAI 官方）。
/// Image / tool / refusal parts 贡献空串但不报错。
fn merge_openai_text(content: Option<OpenAiResponseContent>) -> Vec<ContentPart> {
    let mut out: Vec<ContentPart> = Vec::new();
    if let Some(c) = content {
        match c {
            OpenAiResponseContent::Plain(s) => {
                if !s.is_empty() { out.push(ContentPart::Text { text: s }); }
            }
            OpenAiResponseContent::Parts(parts) => {
                for p in parts {
                    if let OpenAiResponseContentPart::Text { text } = p {
                        out.push(ContentPart::Text { text });
                    }
                }
            }
        }
    }
    out
}

/// Concatenate Anthropic response content blocks into a single text string.
/// Image / tool_use blocks contribute no text but don't error.
fn merge_anthropic_text(blocks: &[AnthropicContentBlock]) -> Vec<ContentPart> {
    let mut out: Vec<ContentPart> = Vec::new();
    for b in blocks {
        if let AnthropicContentBlock::Text { text } = b {
            out.push(ContentPart::Text { text: text.clone() });
        }
    }
    out
}

/// OpenAI tool_call 流式累积单元。
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

impl Default for ToolCallAccum {
    fn default() -> Self {
        Self { id: String::new(), name: String::new(), arguments: String::new() }
    }
}

fn build_stream_tool_calls(acc: &[ToolCallAccum]) -> Vec<ToolCall> {
    let mut out: Vec<ToolCall> = Vec::with_capacity(acc.len());
    for a in acc {
        if a.id.is_empty() && a.name.is_empty() { continue; }
        // 跟 extract_openai_response 一样：JSON 解析失败时把错误信息
        // 写进 `arguments_parse_error`，调用方根据这个判断要不要
        // fallback / panic。
        let (arguments, arguments_parse_error) =
            match serde_json::from_str(&a.arguments) {
                Ok(v) => (v, None),
                Err(e) => (serde_json::Value::Null, Some(e.to_string())),
            };
        out.push(ToolCall {
            id: a.id.clone(),
            name: a.name.clone(),
            arguments,
            arguments_raw: if a.arguments.is_empty() { None } else { Some(a.arguments.clone()) },
            arguments_parse_error,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ApiType, Message};

    fn model_with_key(api: ApiType, key: &str) -> Model {
        Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api,
            provider: "test".into(),
            base_url: "http://127.0.0.1:1".into(),
            api_key: key.into(),
            context_window: 1024,
            max_tokens: 256,
            supports_thinking: false,
            supports_vision: false,
            cost_per_million_input: 0.0,
            cost_per_million_output: 0.0,
        }
    }

    #[tokio::test]
    async fn chat_openai_blank_api_key_fails_fast() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "")).unwrap();
        let msg = vec![Message::user("hi")];
        let params = GenerateParams::default();
        let err = client.chat(&msg, &params).await.unwrap_err();
        match err {
            AiError::Config(m) => assert!(m.contains("test-model"), "got: {m}"),
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_anthropic_blank_api_key_fails_fast() {
        // Regression: previously the request fired with an empty `x-api-key`
        // header and the vendor returned "x-api-key header is required".
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "")).unwrap();
        let msg = vec![Message::user("hi")];
        let params = GenerateParams::default();
        let err = client.chat(&msg, &params).await.unwrap_err();
        match err {
            AiError::Config(m) => assert!(m.contains("test-model"), "got: {m}"),
            other => panic!("expected Config, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_whitespace_only_api_key_fails_fast() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "   ")).unwrap();
        let msg = vec![Message::user("hi")];
        let params = GenerateParams::default();
        let err = client.chat(&msg, &params).await.unwrap_err();
        assert!(matches!(err, AiError::Config(_)));
    }

    #[tokio::test]
    async fn chat_stream_blank_api_key_fails_fast() {
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "")).unwrap();
        let msg = vec![Message::user("hi")];
        let params = GenerateParams::default();
        let err = client.chat_stream(&msg, &params).await.unwrap_err();
        assert!(matches!(err, AiError::Config(_)));
    }

    // ─── Tool use wire 序列化单测 ──────────────────────────────────────
    //
    // 直接调 build_openai_request / build_anthropic_request，把结果
    // 序列化成 JSON 字符串验证关键字段。
    // 这是单测覆盖 P0 的核心：Role::Tool → ToolString / tool_result 块、
    // assistant tool_calls 进 wire、tools 数组进 wire 这几条路径。

    #[test]
    fn openai_request_with_tools_serializes_function_definitions() {
        // 验证 tools 数组以 `{type:"function", function:{...}}` 形态
        // 出现在 wire 请求中。
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "get_weather".into(),
            description: Some("查天气".into()),
            parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}),
        });
        let req = client.build_openai_request(&[Message::user("北京天气?")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        // tools 进 wire
        assert_eq!(j["tools"][0]["type"], "function");
        assert_eq!(j["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(j["tools"][0]["function"]["parameters"]["type"], "object");
        // 有 tools 时默认 tool_choice = "auto"
        assert_eq!(j["tool_choice"], "auto");
    }

    #[test]
    fn openai_request_without_tools_omits_tools_field() {
        // 没传 tools 时，整个 tools 字段不出现（Vec::is_empty skip）。
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert!(j.get("tools").is_none());
        assert!(j.get("tool_choice").is_none());
    }

    #[test]
    fn openai_request_role_tool_serializes_as_string_content() {
        // Role::Tool 的 content 必须是 string（OpenAI 协议要求），不是
        // 数组。`OpenAiMessageContent` 的 untagged 序列化自动选 ToolString 变体。
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![
            Message::user("北京天气?"),
            Message::tool_result("call_abc", "晴，25°C"),
        ];
        let req = client.build_openai_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        // tool 消息的 content 是 string，不是 array
        let tool_msg = &j["messages"][1];
        assert_eq!(tool_msg["role"], "tool");
        assert_eq!(tool_msg["content"], "晴，25°C");
        assert_eq!(tool_msg["tool_call_id"], "call_abc");
    }

    #[test]
    fn openai_request_assistant_carries_its_own_tool_calls() {
        // 多轮里 assistant 消息需要带自己上一轮发起的 tool_calls，否则
        // tool_result 消息找不到对应的 id。这条测验证 Message::tool_calls
        // 字段 → wire 端 `OpenAiMessage.tool_calls` 数组的转换。
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let assistant = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "get_weather".into(),
                arguments: serde_json::json!({"city": "上海"}),
                arguments_raw: Some(r#"{"city":"上海"}"#.into()),
                arguments_parse_error: None,
            }],
        );
        let req = client.build_openai_request(&[assistant], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["messages"][0]["role"], "assistant");
        assert_eq!(j["messages"][0]["tool_calls"][0]["id"], "call_1");
        assert_eq!(j["messages"][0]["tool_calls"][0]["type"], "function");
        assert_eq!(j["messages"][0]["tool_calls"][0]["function"]["name"], "get_weather");
        // arguments 是 JSON 字符串（OpenAI 协议规定）
        assert_eq!(j["messages"][0]["tool_calls"][0]["function"]["arguments"], r#"{"city":"上海"}"#);
    }

    #[test]
    fn anthropic_request_with_tools_serializes_input_schema() {
        // Anthropic 用 input_schema 而不是 parameters。
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "get_weather".into(),
            description: Some("查天气".into()),
            parameters: serde_json::json!({"type":"object"}),
        });
        let req = client.build_anthropic_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tools"][0]["name"], "get_weather");
        assert_eq!(j["tools"][0]["description"], "查天气");
        // 关键：Anthropic 用 input_schema，不是 parameters
        assert!(j["tools"][0].get("parameters").is_none());
        assert_eq!(j["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn anthropic_request_role_tool_becomes_tool_result_block() {
        // Anthropic 没有 tool role —— Role::Tool 转成 user role 消息
        // + 一个 tool_result content block。这是关键路径。
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![
            Message::user("北京天气?"),
            Message::tool_result("toolu_abc", "晴，25°C"),
        ];
        let req = client.build_anthropic_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        // 第二条消息的 role 必须是 user（Anthropic 协议没有 tool role）
        let second = &j["messages"][1];
        assert_eq!(second["role"], "user");
        // content 是一个 tool_result block，不是 text
        assert_eq!(second["content"][0]["type"], "tool_result");
        assert_eq!(second["content"][0]["tool_use_id"], "toolu_abc");
        assert_eq!(second["content"][0]["content"], "晴，25°C");
        // is_error 没显式设 → 跳过
        assert!(second["content"][0].get("is_error").is_none());
    }

    #[test]
    fn anthropic_request_system_message_uses_user_role() {
        // 历史约定：Anthropic 没有 system role，build_anthropic_request
        // 把 Role::System 转成 user 角色。这条行为可能改但目前是这样。
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![Message::system("you are helpful")];
        let req = client.build_anthropic_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["messages"][0]["role"], "user");
    }

    // ─── ToolChoice wire 序列化 ────────────────────────────────────────
    //
    // 验证 `ToolChoice` enum → wire 形态的映射：
    // - OpenAI: Auto/None/Required → string, Specific → {type, function}
    // - Anthropic: Auto → {type:auto}, Required → {type:any},
    //              Specific → {type:tool, name}, None → 跳过
    // - 没传 tools 且 tool_choice == Auto 时整段省略

    #[test]
    fn openai_tool_choice_auto_with_tools_emits_string_auto() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "x".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"], "auto");
    }

    #[test]
    fn openai_tool_choice_required_emits_string_required() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "x".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        params.tool_choice = ToolChoice::Required;
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"], "required");
    }

    #[test]
    fn openai_tool_choice_specific_emits_function_object() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        params.tool_choice = ToolChoice::Specific("get_weather".into());
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"]["type"], "function");
        assert_eq!(j["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn openai_tool_choice_none_emits_string_none() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "x".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        params.tool_choice = ToolChoice::None;
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"], "none");
    }

    #[test]
    fn openai_tool_choice_default_with_no_tools_omits_field() {
        // 没传 tools 且 tool_choice == Auto → 整个 tool_choice 字段跳过。
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert!(j.get("tool_choice").is_none());
    }

    #[test]
    fn anthropic_tool_choice_required_emits_type_any() {
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "x".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        params.tool_choice = ToolChoice::Required;
        let req = client.build_anthropic_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"]["type"], "any");
    }

    #[test]
    fn anthropic_tool_choice_specific_emits_type_tool_with_name() {
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "get_weather".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        params.tool_choice = ToolChoice::Specific("get_weather".into());
        let req = client.build_anthropic_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"]["type"], "tool");
        assert_eq!(j["tool_choice"]["name"], "get_weather");
    }

    #[test]
    fn anthropic_tool_choice_none_is_dropped() {
        // Anthropic 没有 None 语义 —— 不传 tool_choice 字段。
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool {
            name: "x".into(),
            description: None,
            parameters: serde_json::json!({}),
        });
        params.tool_choice = ToolChoice::None;
        let req = client.build_anthropic_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert!(j.get("tool_choice").is_none());
    }
}

