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
        Ok(Completion {
            content: merge_openai_text(choice.message.content),
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
                                tx.send(StreamEvent::Done { content, usage }).await.ok();
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
                                    tx.send(StreamEvent::Done { content, usage: u }).await.ok();
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
                                tx.send(StreamEvent::Done { content: full_text.clone(), usage: usage.clone() }).await.ok();
                                return;
                            }
                            match serde_json::from_str::<OpenAiStreamChunk>(data) {
                                Ok(chunk) => {
                                    if let Some(choice) = chunk.choices.into_iter().next() {
                                        if let Some(delta) = choice.delta.content {
                                            full_text.push_str(&delta);
                                            tx.send(StreamEvent::Delta {
                                                content: delta,
                                                usage: None,
                                            }).await.ok();
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
            tx.send(StreamEvent::Done { content: full_text, usage }).await.ok();
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

        let content = merge_anthropic_text(&data.content);

        Ok(Completion {
            content,
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

            let mut es = es;
            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => continue,
                    Ok(Event::Message(msg)) => {
                        match serde_json::from_str::<AnthropicStreamEvent>(&msg.data) {
                            Ok(evt) => {
                                match evt.type_.as_str() {
                                    "content_block_delta" => {
                                        if let Some(delta) = &evt.delta {
                                            if let Some(text) = &delta.text {
                                                full_text.push_str(text);
                                                tx.send(StreamEvent::Delta {
                                                    content: text.clone(),
                                                    usage: None,
                                                }).await.ok();
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

            tx.send(StreamEvent::Done { content: full_text, usage }).await.ok();
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
            messages: messages.iter().map(|m| OpenAiMessage {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                }.into(),
                content: m.content.iter().map(to_openai_content_part).collect(),
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
            messages: messages.iter().map(|m| AnthropicMessage {
                role: match m.role {
                    Role::System => "user",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                }.into(),
                content: m.content.iter().map(to_anthropic_content_block).collect(),
            }).collect(),
            max_tokens,
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            stop_sequences: params.stop_sequences.clone(),
            thinking,
        }
    }
}


/// Concatenate OpenAI response content into a single text string.
/// 同时支持 string 形态（minimax 等）和 array 形态（OpenAI 官方）。
/// Image / tool / refusal parts 贡献空串但不报错。
fn merge_openai_text(content: Option<OpenAiResponseContent>) -> String {
    let mut out = String::new();
    if let Some(c) = content {
        match c {
            // minimax 路径：直接拿到 string，**原样返回** —— 包括 vendor 的
            // `<think>...</think>` 块。这块如果调用方想要剥离，可以在上层
            // 用 `extract_thinking` 之类的工具函数处理；merge 不应该擅自
            // 改 vendor 的内容。
            OpenAiResponseContent::Plain(s) => out.push_str(&s),
            // OpenAI 官方路径：拼所有 text parts。
            OpenAiResponseContent::Parts(parts) => {
                for p in parts {
                    if let OpenAiResponseContentPart::Text { text } = p {
                        out.push_str(&text);
                    }
                }
            }
        }
    }
    out
}

/// Concatenate Anthropic response content blocks into a single text string.
/// Image / tool_use blocks contribute no text but don't error.
fn merge_anthropic_text(blocks: &[AnthropicContentBlock]) -> String {
    let mut out = String::new();
    for b in blocks {
        if let AnthropicContentBlock::Text { text } = b {
            out.push_str(text);
        }
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
}

