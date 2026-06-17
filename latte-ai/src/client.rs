use std::time::Duration;

use futures_util::StreamExt;
use reqwest::Client as HttpClient;
use reqwest_eventsource::{Event, RequestBuilderExt};
use tracing::{debug, warn};

use crate::error::{AiError, Result};
use crate::models::*;
use std::collections::HashSet;
use std::sync::Arc;

use crate::vendor::{Dispatcher, VendorFeature, VendorId};
use crate::params::GenerateParams;
/// A client for interacting with AI models via OpenAI-compatible or Anthropic APIs.
#[derive(Clone)]
pub struct AiClient {
    http: HttpClient,
    model: Model,
    dispatcher: Option<Arc<Dispatcher>>,
}

impl std::fmt::Debug for AiClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AiClient")
            .field("model", &self.model)
            .field("dispatcher", &self.dispatcher.as_ref().map(|_| "<dispatcher>"))
            .finish()
    }
}

impl AiClient {
    /// Create a new client for the given model.
    pub fn new(model: Model) -> Result<Self> {
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(300))
            .build()?;

        Ok(Self { http, model, dispatcher: None })
    }

    /// 挂载 vendor dispatcher（feature gate / token 注入）。
    ///
    /// 挂载后 `chat_with_features` 会先调 `dispatcher.check()`，
    /// `chat()` 仍走原路径（向后兼容）。
    pub fn with_dispatcher(mut self, dispatcher: Arc<Dispatcher>) -> Self {
        self.dispatcher = Some(dispatcher);
        self
    }

    /// 拿当前 dispatcher（如有）
    pub fn dispatcher(&self) -> Option<&Arc<Dispatcher>> {
        self.dispatcher.as_ref()
    }

    // ── public API ─────────────────────────────────────────────────────

    /// Send a non-streaming chat completion request.
    pub async fn chat(&self, messages: &[Message], params: &GenerateParams) -> Result<Completion> {
        match self.model.api {
            ApiType::OpenAiCompletions => self.chat_openai(messages, params).await,
            ApiType::AnthropicMessages => self.chat_anthropic(messages, params).await,
        }
    }

    /// Send a non-streaming chat with explicit feature list (enables dispatcher check).
    ///
    /// 若挂载了 dispatcher 且 vendor 禁用了 `requested_features` 中的任一 feature，
    /// 返 `Err(AiError::Other("vendor: ..."))`。
    /// 未挂载 dispatcher 时等价于 `chat()`。
    pub async fn chat_with_features(
        &self,
        messages: &[Message],
        params: &GenerateParams,
        requested_features: &HashSet<VendorFeature>,
    ) -> Result<Completion> {
        if let Some(d) = &self.dispatcher {
            let vendor_id = VendorId::new(self.model.provider.clone());
            d.check(&vendor_id, requested_features)?;
        }
        self.chat(messages, params).await
    }

    /// Send a streaming chat completion request.
    ///
    /// Returns a receiver that yields `StreamEvent` values as they arrive.
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        params: &GenerateParams,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>> {
        match self.model.api {
            ApiType::OpenAiCompletions => self.stream_openai(messages, params).await,
            ApiType::AnthropicMessages => self.stream_anthropic(messages, params).await,
        }
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

        let data: OpenAiChatResponse = resp.json().await?;
        let choice = data.choices.into_iter().next()
            .ok_or_else(|| AiError::Other("No choices in response".into()))?;

        Ok(Completion {
            content: choice.message.content.unwrap_or_default(),
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
                                let content = data.choices.into_iter().next()
                                    .map(|c| c.message.content.unwrap_or_default())
                                    .unwrap_or_default();
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
                                    let content = data.choices.into_iter().next()
                                        .map(|c| c.message.content.unwrap_or_default())
                                        .unwrap_or_default();
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

        let mut content = String::new();
        for block in &data.content {
            if let Some(text) = &block.text {
                content.push_str(text);
            }
        }

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
                content: m.content.clone(),
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
                content: m.content.clone(),
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
