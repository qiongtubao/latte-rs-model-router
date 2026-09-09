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

/// 流式请求是否带 `stream_options: {include_usage: true}`。默认开——
/// 不带它 OpenAI 兼容端点不回 usage，token 记账全 0（见
/// [`OpenAiChatRequest::stream_options`]）。个别端点对未知字段 400，
/// 用 `LATTE_AI_STREAM_INCLUDE_USAGE=0` 关掉。
fn stream_include_usage() -> bool {
    std::env::var("LATTE_AI_STREAM_INCLUDE_USAGE")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

// ── 流式超时常量 ──────────────────────────────────────────────────
//
// 双层超时模型（替代旧的总死线 `.timeout(300s)`）：
//   1. 首 token 超时（TTFB）：模型处理 prompt + 吐第一个 token 的窗口。
//      reasoning 模型可能思考很久，默认 100s。
//   2. idle 超时：两个 SSE chunk 之间的最大间隔。只要 token 在持续
//      流动就不会触发，总生成时间无上限。默认 120s。
//
// 环境变量覆盖（设为 0 关闭 watchdog，仅调试用）：
//   LATTE_AI_FIRST_EVENT_TIMEOUT_SECS=100
//   LATTE_AI_IDLE_TIMEOUT_SECS=120

/// 首 token 超时（秒）：模型处理 prompt + 吐第一个 token 的窗口。
///
/// 优先级：**模型目录里的 `timeout_secs` 精确生效** > 环境变量 > 默认。
///
/// 原来是 `base.max(t)`，即模型声明的值**只能放宽不能收紧**——想给某个
/// 已知会秒回的模型配"快速失败"（比如 `timeout_secs = 20`，让它挂掉时
/// 快速沿链落到下一个）根本表达不出来，写了也被默认的 100s 顶掉。
/// 而 README 里对这个字段的说法是"优先级最高"，与实现不符。
///
/// 改成精确生效是安全的：全仓没有任何配置设置过 `timeout_secs`
/// （`~/.latte/models.d/*.toml` 与 `models.yaml` 都没有），所以不改变
/// 任何在跑的行为；一旦有人设置，得到的就是他写的那个值。
fn first_event_timeout_secs(model_timeout: Option<u64>) -> u64 {
    match model_timeout {
        Some(t) if t > 0 => t,
        _ => env_timeout_secs("LATTE_AI_FIRST_EVENT_TIMEOUT_SECS", 100),
    }
}

/// idle 超时（秒）：两个 SSE chunk 之间的最大间隔。
///
/// 优先级同 [`first_event_timeout_secs`]：模型目录精确生效 > 环境变量 >
/// 默认。
fn idle_timeout_secs(model_timeout: Option<u64>) -> u64 {
    match model_timeout {
        Some(t) if t > 0 => t,
        _ => env_timeout_secs("LATTE_AI_IDLE_TIMEOUT_SECS", 120),
    }
}

/// 从环境变量读取超时秒数；未设置时用 `default_secs`；设为 0 返回 0（关闭）。
fn env_timeout_secs(var: &str, default_secs: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(default_secs)
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
    /// 这个 client 绑定的模型定义。
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// 本 client 在**不额外指定**时会下发的输出上限；`None` = 不下发该字段。
    ///
    /// 给「测试模型」这类需要把真实下发值展示出来的调用方用 —— 光看
    /// 「请求通了」无法判断上限配得合不合理。
    pub fn resolved_max_tokens(&self) -> Option<u32> {
        Self::effective_max_tokens(None, &self.model)
    }

    /// 创建 AI 模型客户端。
    ///
    /// 超时策略：只设 TCP 连接超时（30s），**不设总请求死线**。
    /// 实际的响应超时由 [`AiClient::chat`] 内部的流式 idle watchdog
    /// 管理（首-token 100s + chunk 间隔 120s），支持任意长生成。
    /// `model.timeout_secs` 仍然生效：作为 idle watchdog 的下限
    /// （取 max），让重产出角色有更宽的窗口。
    pub fn new(model: Model) -> Result<Self> {
        let http = HttpClient::builder()
            .connect_timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self { http, model })
    }

    // ── public API ─────────────────────────────────────────────

    /// 发送聊天完成请求，返回完整 [`Completion`]。
    ///
    /// 内部走**流式传输**（`stream: true`）+ idle watchdog 消费到完整
    /// 响应，对调用方完全透明（返回类型不变）。超时模型：
    ///   - 首 token 超时（TTFB）：默认 100s，可被 `model.timeout_secs`
    ///     或 `LATTE_AI_FIRST_EVENT_TIMEOUT_SECS` 覆盖。
    ///   - idle 超时：两个 chunk 间默认 120s，可被
    ///     `LATTE_AI_IDLE_TIMEOUT_SECS` 覆盖。
    ///   - 总生成时间无上限：只要 token 在流，不会因总时间超时。
    pub async fn chat(&self, messages: &[Message], params: &GenerateParams) -> Result<Completion> {
        self.check_api_key()?;
        let mut rx = self.chat_stream(messages, params).await?;
        let ttfb = first_event_timeout_secs(self.model.timeout_secs);
        let idle = idle_timeout_secs(self.model.timeout_secs);
        // 第一层：TTFB 超时 -- 等首个 SSE 事件到达。
        let first = if ttfb > 0 {
            tokio::time::timeout(Duration::from_secs(ttfb), rx.recv())
                .await
                .map_err(|_| AiError::Stream(format!("等待首个事件超时（{ttfb}s 无响应）")))?
        } else {
            rx.recv().await
        };
        let first = first.ok_or_else(|| AiError::Stream("stream 在发送任何事件前关闭".into()))?;
        // 首事件可能直接是 Done（短响应）或 Error（连接级失败）。
        if let Some(c) = Self::completion_from_event(first) {
            return c;
        }
        // 第二层：idle 超时 -- 逐 chunk 消费到 Done。
        loop {
            let next = if idle > 0 {
                tokio::time::timeout(Duration::from_secs(idle), rx.recv())
                    .await
                    .map_err(|_| AiError::Stream(format!("流式响应空闲超时（{idle}s 无新数据）")))?
            } else {
                rx.recv().await
            };
            let next = next.ok_or_else(|| AiError::Stream("stream 在 Done 之前关闭".into()))?;
            if let Some(c) = Self::completion_from_event(next) {
                return c;
            }
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

/// 将 [`StreamEvent`] 转换为 [`Completion`]。
///
/// - `Done` -> `Some(Ok(Completion))`：流结束，返回完整响应。
/// - `Error` -> `Some(Err)`：流内错误，直接返回。
/// - `Delta` -> `None`：增量数据，调用方继续消费。
///
/// `chat()` 逐事件调用此函数；返回 `None` 时继续 `rx.recv()`。
/// 本次请求实际要下发的输出上限。`None` = **不下发**该字段，把上限交给
/// 厂商默认值。
///
/// 决策集中在这一处（对齐 oh-my-pi 的 `resolveOpenAIOutputTokenParam`），
/// 按优先级：
///
/// 1. `model.omit_max_tokens` → 直接不发。代理转发到未知后端时，发一个
///    猜的值只会换来上游 400。
/// 2. 调用方显式给的值（`GenerateParams::max_tokens`）优先于目录值。
/// 3. 目录值为 0 视为"未配置" → 不发。
/// 4. 到此为止 —— 得到的值**原样下发**，不做任何隐式钳制。
///
/// # 为什么不钳制
///
/// 曾经这里有一个写死的 `MAX_OUTPUT_TOKENS_CEILING = 64000` 和一个
/// `context_window * 7/8` 的窗口钳，两者都会**背着配置改数**：目录里写
/// 384000，实际下发 64000，而且没有任何日志。结果是那个配置项事实上是死的
/// —— 把它从 384000 改成 200000 行为完全不变，使用者也无从知道为什么。
///
/// 现在的契约是一句话：**配置写多少就发多少**。配错了由厂商用 4xx 明确
/// 告诉你，那是可诊断的；静默换一个数不是。
///
/// # 修的 bug
///
/// OpenAI 协议这条路以前是 `max_tokens: params.max_tokens` 直接透传，而
/// `OpenAiChatRequest.max_tokens` 带 `skip_serializing_if = "Option::is_none"`
/// —— 调用方不显式设置（`GenerateParams::default()` 就是 `None`）时字段被
/// **整个省略**，于是模型目录里的 `max_tokens` 在这条链路上是装饰性的，
/// 真实上限由厂商默认值决定。Anthropic 协议那条一直有回落（它的 API 要求
/// 该字段必填）—— 两条协议的行为就这么分叉了。
///
/// 实测（2026-09-07 jemalloc 会话）：两个模型都配了 `max_tokens = 384000`，
/// 而 529 次请求下发的全是 `None`；architect 在 tasks_json 步撞上厂商默认
/// 上限，`finish_reason=length`、`total_output` 恰好 8192，半个 JSON 穿给
/// 下游终审，gate 三轮 REJECT 后整条 20 分钟流水线判 failed。
///
/// 注意 8192 **不是**该厂商的固定上限：同一会话里同一个 glm-5.3 在别的
/// 调用上正常产出了 10093 / 8620 token（`finish_reason=stop`）。真实上限
/// 未知 —— 这恰恰是不该由本函数替使用者拍一个数的理由：唯一知道真实上限的
/// 是厂商，让它自己说。
///
    /// `pub` 是为了让调用方能**先问一句「这次会发多少」**再决定要不要发
    /// —— UI 的「测试模型」需要把实际下发值显示出来，否则测出来的
    /// 「通了」跟真实 chat 用的上限没有关系（它以前写死 1024）。
    pub fn effective_max_tokens(explicit: Option<u32>, model: &Model) -> Option<u32> {
    if model.omit_max_tokens {
        return None;
    }
    let catalog = (model.max_tokens > 0).then_some(model.max_tokens);
    match explicit.or(catalog) {
        Some(n) if n > 0 => Some(n),
        _ => None,
    }
}

fn completion_from_event(event: StreamEvent) -> Option<Result<Completion>> {
    match event {
        StreamEvent::Done { content, tool_calls, usage, stop_reason } => {
            let text = content.iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            Some(Ok(Completion {
                content: text,
                content_parts: content,
                tool_calls,
                stop_reason,
                usage,
            }))
        }
        StreamEvent::HttpError { status, message } => Some(Err(AiError::Api { status, message })),
        StreamEvent::Error(e) => Some(Err(AiError::Stream(e))),
        StreamEvent::Delta { .. } => None,
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
                // 传输层失败（请求没送出去）统一打标记：上层
                // `failures_are_all_transport` 靠它把「全链连不上」判成
                // 本机网络/代理故障，而不是「所有模型都挂了」。
                // 漏打标记的后果就是又退回旧的误诊断。
                if e.is_timeout() {
                    AiError::Other(format!(
                        "{} Request timed out",
                        crate::error::TRANSPORT_FAILURE_MARKER
                    ))
                } else if e.is_connect() {
                    AiError::Other(format!(
                        "{} Connection failed: {e}",
                        crate::error::TRANSPORT_FAILURE_MARKER
                    ))
                } else if crate::error::is_transport_failure(&e) {
                    AiError::Other(format!(
                        "{} Request failed: {e}",
                        crate::error::TRANSPORT_FAILURE_MARKER
                    ))
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
                    // 区分「请求没送出去」（连接/DNS/代理/连接超时）与
                    // 「厂商返回错误」：前者带上标记，上层遇到全链皆此类
                    // 时会改报本机网络故障，而不是「所有模型不可用」。
                    let msg = if crate::error::is_transport_failure(&e) {
                        format!(
                            "{} Request failed: {e}",
                            crate::error::TRANSPORT_FAILURE_MARKER
                        )
                    } else {
                        format!("Request failed: {e}")
                    };
                    tx.send(StreamEvent::Error(msg)).await.ok();
                    return;
                }
            };

            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                // Auth 错误（401/403）不需要非流式 fallback -- 换 stream 模式
                // 不会修复鉴权问题，只会多浪费一个请求。直接返回 HttpError。
                if matches!(status.as_u16(), 401 | 403) {
                    tx.send(StreamEvent::HttpError { status: status.as_u16(), message: body }).await.ok();
                    return;
                }
                // 其他错误（如 500 / 不支持流式）：尝试非流式 fallback
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
                                tx.send(StreamEvent::Done { content, tool_calls: vec![], usage, stop_reason: String::new() }).await.ok();
                            }
                            Err(e) => {
                                tx.send(StreamEvent::Error(format!("Parse error: {e}"))).await.ok();
                            }
                        }
                    }
                    Ok(r) => {
                        let b = r.text().await.unwrap_or_default();
                        tx.send(StreamEvent::HttpError { status: status.as_u16(), message: format!("{body}; non-stream also failed: {b}") }).await.ok();
                    }
                    Err(e) => {
                        tx.send(StreamEvent::HttpError { status: status.as_u16(), message: format!("{body}; fallback failed: {e}") }).await.ok();
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
            let mut finish_reason = String::new();

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
                                    tx.send(StreamEvent::Done { content, tool_calls: vec![], usage: u, stop_reason: String::new() }).await.ok();
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
                                    stop_reason: finish_reason.clone(),
                                }).await.ok();
                                return;
                            }
                            match serde_json::from_str::<OpenAiStreamChunk>(data) {
                                Ok(chunk) => {
                                    // usage 先于 choices 取：`stream_options.
                                    // include_usage` 回的那个 chunk 是
                                    // **choices 为空**的独立 chunk（OpenAI
                                    // 规范：在 [DONE] 之前单独发一条只带
                                    // usage 的 chunk）。此前 usage 只在
                                    // `choices[0].finish_reason` 存在时才读，
                                    // 所以那条 chunk 被整个丢掉 —— 即使带上
                                    // include_usage 也依然记账为 0。
                                    if let Some(u) = &chunk.usage {
                                        if u.prompt_tokens > 0 || u.completion_tokens > 0 {
                                            usage = TokenUsage {
                                                input_tokens: u.prompt_tokens,
                                                output_tokens: u.completion_tokens,
                                                thinking_tokens: 0,
                                            };
                                        }
                                    }
                                    if let Some(choice) = chunk.choices.into_iter().next() {
                                        if let Some(delta) = choice.delta.content {
                                            full_text.push_str(&delta);
                                            tx.send(StreamEvent::Delta {
                                                content: vec![ContentPart::Text { text: delta }],
                                                usage: None,
                                            }).await.ok();
                                        }
                                        for td in choice.delta.tool_calls.unwrap_or_default() {
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
                                        if let Some(fr) = choice.finish_reason.as_ref() {
                                            finish_reason = fr.clone();
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

            // Fallback：如果没有解析到任何 SSE 事件（full_text 为空、
            // tool_call_acc 为空、buf 有剩余数据），尝试把 buf 当作
            // 非流式 JSON 响应解析。场景：代理/网关不支持 SSE，收到
            // stream:true 请求但返回普通 JSON 响应。
            if full_text.is_empty() && tool_call_acc.is_empty() && !buf.is_empty() {
                if let Ok(data) = serde_json::from_str::<OpenAiChatResponse>(&buf) {
                    let u = TokenUsage {
                        input_tokens: data.usage.as_ref().map(|u| u.prompt_tokens).unwrap_or(0),
                        output_tokens: data.usage.as_ref().map(|u| u.completion_tokens).unwrap_or(0),
                        thinking_tokens: 0,
                    };
                    if let Some(choice) = data.choices.into_iter().next() {
                        let (parts, tool_calls) = extract_openai_response(
                            choice.message.content,
                            choice.message.tool_calls,
                        );
                        tx.send(StreamEvent::Delta { content: parts.clone(), usage: Some(u.clone()) }).await.ok();
                        tx.send(StreamEvent::Done {
                            content: parts,
                            tool_calls,
                            usage: u,
                            stop_reason: choice.finish_reason.unwrap_or_default(),
                        }).await.ok();
                        return;
                    }
                }
            }
            // Stream ended without [DONE] - send what we have
            let final_calls = build_stream_tool_calls(&tool_call_acc);
            tx.send(StreamEvent::Done {
                content: vec![ContentPart::Text { text: full_text }],
                tool_calls: final_calls,
                usage,
                stop_reason: finish_reason,
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
            let mut stop_reason = String::new();
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
                                        // Anthropic 的 stop_reason 在 delta.stop_reason 里
                                        if let Some(delta) = &evt.delta {
                                            if let Some(sr) = &delta.stop_reason {
                                                stop_reason = sr.clone();
                                            }
                                        }
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
                stop_reason,
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
        // 输出上限只在这一处决策，随后按 `max_tokens_field` 落到两个互斥
        // 字段之一（`None` 时两个都省略，交给厂商默认值）。
        let cap = Self::effective_max_tokens(params.max_tokens, &self.model);
        let mut req = OpenAiChatRequest {
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
                    _ => {
                        // 过滤空 text part：kimi k3 等 vendor 对
                        // `{"type":"text","text":""}` 直接 400
                        // （"text content is empty"），而 `content: []`
                        // 是合法的（assistant 只发 tool_calls 不产文本
                        // 时本来就是空）。坏消息一旦进 history 会让之后
                        // 每个请求都 400，必须在 wire 构造处拦掉。
                        let parts: Vec<OpenAiContentPart> = m.content.iter()
                            .filter(|p| !matches!(p, ContentPart::Text { text } if text.is_empty()))
                            .map(to_openai_content_part)
                            .collect();
                        if parts.is_empty() && m.tool_calls.is_none() {
                            // 完全没有内容且不是工具调用回合：空 user/
                            // assistant 消息同样会被 vendor 拒绝
                            // （"must not be empty"），给一个占位空格。
                            OpenAiMessageContent::Parts(vec![OpenAiContentPart::Text {
                                text: " ".into(),
                                cache_control: None,
                            }])
                        } else {
                            OpenAiMessageContent::Parts(parts)
                        }
                    }
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
            max_tokens: cap.filter(|_| {
                self.model.max_tokens_field == crate::models::MaxTokensField::MaxTokens
            }),
            max_completion_tokens: cap.filter(|_| {
                self.model.max_tokens_field == crate::models::MaxTokensField::MaxCompletionTokens
            }),
            stop: params.stop_sequences.clone(),
            seed: params.seed,
            tools: params.tools.iter().map(|t| OpenAiTool {
                type_: "function".into(),
                function: OpenAiFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                    strict: t.strict,
                },
            }).collect(),
            tool_choice: if params.tools.is_empty() && params.tool_choice == ToolChoice::Auto {
                None
            } else {
                Some(OpenAiToolChoice::from(&params.tool_choice))
            },
            // 没有下发工具时这个字段没有意义（个别端点还会因此报错），
            // 只在带工具的请求上透出调用方的意图。
            parallel_tool_calls: if params.tools.is_empty() {
                None
            } else {
                params.parallel_tool_calls
            },
            stream,
            // 只有流式请求需要它；非流式响应本来就带 usage。个别
            // 兼容端点不认这个字段，用 env 关掉：
            // LATTE_AI_STREAM_INCLUDE_USAGE=0
            stream_options: if stream && stream_include_usage() {
                Some(OpenAiStreamOptions { include_usage: true })
            } else {
                None
            },
        };

        // OpenAI-compatible 路径的 prompt caching：**按模型开关**下发。
        // 不认 `cache_control` 的端点看到未知字段会 400，所以默认关，
        // 只在确认支持的端点上开（见 `Model::prompt_cache`）。
        if self.model.prompt_cache {
            apply_openai_cache_breakpoints(&mut req);
        }
        req
    }

    fn build_anthropic_request(
        &self,
        messages: &[Message],
        params: &GenerateParams,
        _stream: bool,
    ) -> AnthropicRequest {
        // Anthropic 要求 `max_tokens` 必填，所以这条路必须给出一个数：
        // 走同一个 `effective_max_tokens` 拿到天花板与钳位，`None`（含
        // `omit_max_tokens = true` 的情况）时用窗口推出的上限兜底。
        //
        // `omit_max_tokens` 在这条协议上**无法生效**——省略该字段请求直接
        // 被拒。它的目标场景（代理转发到未知后端）也基本只出现在
        // OpenAI-compatible 那一侧，所以这里只做兜底、不报错。
        // 这条协议要求 `max_tokens` 必填，省略直接被拒。所以只有在
        // **完全没有配置值**（目录 0 且调用方没给）时才兜一个保守默认；
        // 一旦配了值就原样下发，和 OpenAI 那条路一致。
        let max_tokens = Self::effective_max_tokens(params.max_tokens, &self.model)
            .unwrap_or(crate::models::ANTHROPIC_UNCONFIGURED_MAX_TOKENS);
        let thinking = params.thinking_budget.map(|tb| AnthropicThinking {
            type_: "enabled".into(),
            budget_tokens: tb.token_budget(),
        });

        // system 提到顶层：规范缓存顺序是 tools → system → messages，
        // 断点必须落在稳定的头部才能让"巨大且不变的前缀"每轮命中缓存。
        // 从前把 Role::System 转成 user 消息塞进 messages，头部无处可锚。
        //
        // 多条 System 消息按出现顺序合并成多个 block（保持字节稳定：
        // 同一会话里同样的输入产生同样的数组）。
        let system: Vec<AnthropicSystemBlock> = messages
            .iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| {
                m.content.iter().filter_map(|p| match p {
                    ContentPart::Text { text } if !text.is_empty() => {
                        Some(AnthropicSystemBlock::text(text.clone()))
                    }
                    _ => None,
                })
            })
            .collect();

        let mut req = AnthropicRequest {
            model: self.model.id.clone(),
            system,
            // Anthropic 没有 tool role —— `Role::Tool` 转成 user role
            // 消息 + 一个 tool_result block。System 已提到顶层 system
            // 数组，这里跳过，避免重复下发。
            messages: messages.iter().filter(|m| m.role != Role::System).map(|m| {
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
                            cache_control: None,
                        }]
                    }
                    _ => {
                        // 与 OpenAI 路径一致：滤掉空 text block（Anthropic
                        // 对空 text block 同样 400），全空且无内容时给
                        // 占位空格兜底。
                        let blocks: Vec<AnthropicContentBlock> = m.content.iter()
                            .filter(|p| !matches!(p, ContentPart::Text { text } if text.is_empty()))
                            .map(to_anthropic_content_block)
                            .collect();
                        if blocks.is_empty() {
                            vec![AnthropicContentBlock::Text { text: " ".into(), cache_control: None }]
                        } else {
                            blocks
                        }
                    }
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
                cache_control: None,
            }).collect(),
            tool_choice: if params.tools.is_empty() && params.tool_choice == ToolChoice::Auto {
                None
            } else {
                AnthropicToolChoice::from(&params.tool_choice)
            },
            thinking,
        };

        // ── 4 断点 prompt caching ─────────────────────────────────────
        //
        // Anthropic 每请求最多 4 个断点。预算分配（对齐 Claude Code / Pi
        // 这些一方客户端的放法）：
        //   1 个 → 最后一个 tool     （工具定义前缀）
        //   1 个 → 最后一个 system block（连带缓存它前面的全部 tools）
        //   2 个 → 消息尾部滚动窗口   （随对话增长滚动命中）
        //
        // 为什么头部两个都要：规范缓存顺序 tools → system → messages，
        // system 上的断点已覆盖整个 tools+system 前缀；额外给 tools 一个，
        // 是为了 system 文本变化时工具定义**仍然**留在缓存里（system
        // prompt 常随角色/轮次微调，工具定义几乎不变）。
        apply_anthropic_cache_breakpoints(&mut req);
        req
    }
}

/// OpenAI-compatible 端点的断点放置：system 头部（最多 2 条）+ 消息尾部
/// （最多 2 条），共 ≤ 4，与 Anthropic 原生路径同一预算。
///
/// 兼容端点没有独立的顶层 system 数组，system 就是 messages 里的头几条，
/// 所以头部断点直接挂在 system 消息上（对齐 opencode 的
/// `applyCaching`：`system.slice(0,2)` + `final.slice(-2)`）。
///
/// **只在已经是 `Parts` 形态的 content 上挂**：把 `String` 强行转成
/// `Parts` 会改变消息形状，个别兼容端点对此敏感（空 text block、结构化
/// content 的支持度参差）。宁可少挂一个断点，也不要为了挂断点去改本来
/// 能用的请求形状。
fn apply_openai_cache_breakpoints(req: &mut OpenAiChatRequest) {
    // 头部：前两条 system 消息。
    let mut head = 0usize;
    for msg in req.messages.iter_mut() {
        if head >= 2 {
            break;
        }
        if msg.role != "system" {
            // system 只可能在最前面连续出现，遇到非 system 即停。
            break;
        }
        if mark_openai_last_text(&mut msg.content) {
            head += 1;
        }
    }
    // 尾部：最后两条消息（滚动窗口，让上一轮的尾部在这一轮变成可命中的
    // 前缀内部）。
    let mut tail = 0usize;
    for msg in req.messages.iter_mut().rev() {
        if tail >= 2 {
            break;
        }
        if mark_openai_last_text(&mut msg.content) {
            tail += 1;
        }
    }
}

/// 在一条 OpenAI 消息的最后一个 text part 上挂断点。content 为
/// `String` 形态时**不动**（见 `apply_openai_cache_breakpoints` 的说明）。
/// 返回是否成功放置。
fn mark_openai_last_text(content: &mut OpenAiMessageContent) -> bool {
    let OpenAiMessageContent::Parts(parts) = content else {
        return false;
    };
    for part in parts.iter_mut().rev() {
        if let OpenAiContentPart::Text { cache_control, .. } = part {
            if cache_control.is_none() {
                *cache_control = Some(AnthropicCacheControl::ephemeral());
            }
            return true;
        }
    }
    false
}

/// 放置 Anthropic prompt caching 断点：头部 2（tools 末尾 + system 末尾）
/// + 消息尾部 2（滚动窗口）。
///
/// 幂等：只在尚无断点处写入。调用方每轮重建请求，因此不会累积。
fn apply_anthropic_cache_breakpoints(req: &mut AnthropicRequest) {
    // 头部：最后一个 tool。
    if let Some(last) = req.tools.last_mut() {
        if last.cache_control.is_none() {
            last.cache_control = Some(AnthropicCacheControl::ephemeral());
        }
    }
    // 头部：最后一个 system block。
    if let Some(last) = req.system.last_mut() {
        if last.cache_control.is_none() {
            last.cache_control = Some(AnthropicCacheControl::ephemeral());
        }
    }
    // 尾部：最近两条消息各挂一个，落在该消息的最后一个可承载 block 上。
    // 滚动窗口的意义：上一轮的尾部断点在这一轮变成"前缀内部"，于是这一轮
    // 的前缀能命中上一轮写入的缓存；只挂一个的话，每轮新增内容都在断点
    // 之后、永远进不了缓存。
    let mut placed = 0usize;
    for msg in req.messages.iter_mut().rev() {
        if placed >= 2 {
            break;
        }
        if apply_cache_control_to_last_block(&mut msg.content) {
            placed += 1;
        }
    }
}

/// 把断点挂在一组 content block 的**最后一个可承载者**上（text /
/// tool_result）。image / tool_use 不承载 cache_control。
/// 返回是否成功放置。
fn apply_cache_control_to_last_block(blocks: &mut [AnthropicContentBlock]) -> bool {
    for block in blocks.iter_mut().rev() {
        match block {
            AnthropicContentBlock::Text { cache_control, .. }
            | AnthropicContentBlock::ToolResult { cache_control, .. } => {
                if cache_control.is_none() {
                    *cache_control = Some(AnthropicCacheControl::ephemeral());
                }
                return true;
            }
            _ => continue,
        }
    }
    false
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
        if let AnthropicContentBlock::Text { text, .. } = b {
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
    /// 模型目录里的 `timeout_secs` 必须**精确生效**，既能放宽也能收紧。
    ///
    /// 原来是 `base.max(t)`：只能放宽。想给已知会秒回的模型配
    /// `timeout_secs = 20` 做快速失败（挂掉时快速沿链落到下一个模型）
    /// 根本表达不出来——写了也被默认的 100s/120s 顶掉。而 README 对这个
    /// 字段的说法是"优先级最高"，与实现不符。
    /// 串行化那些**改进程级环境变量**的超时测试。
    ///
    /// `first_event_timeout_secs` / `idle_timeout_secs` 的默认值读
    /// `LATTE_AI_*_TIMEOUT_SECS`，而环境变量是进程全局的：一个测试
    /// `set_var` 期间，另一个断言"默认值"的测试就会读到被改过的值。
    ///
    /// 这是个**既有的潜在竞态**，此前靠测试数量少、调度恰好不重叠而侥幸
    /// 通过；2026-09-08 加了十几个 wiremock 测试后调度一变就暴露了
    /// （`model_timeout_overrides_exactly_not_just_widens` 在
    /// `--test-threads=4` 下随机失败，单独跑必过）。
    static TIMEOUT_ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn model_timeout_overrides_exactly_not_just_widens() {
        let _guard = TIMEOUT_ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        // 断言默认值之前先清掉环境变量：别的测试可能刚设过。
        std::env::remove_var("LATTE_AI_FIRST_EVENT_TIMEOUT_SECS");
        std::env::remove_var("LATTE_AI_IDLE_TIMEOUT_SECS");
        // 收紧：比默认小的值必须生效（这是原实现做不到的）。
        assert_eq!(first_event_timeout_secs(Some(20)), 20, "应能收紧 TTFB");
        assert_eq!(idle_timeout_secs(Some(15)), 15, "应能收紧 idle");
        // 放宽：仍然生效。
        assert_eq!(first_event_timeout_secs(Some(600)), 600);
        assert_eq!(idle_timeout_secs(Some(600)), 600);
        // 未设置 / 设为 0 → 落回默认（0 在本系统里是"关闭 watchdog"的
        // 环境变量语义，模型侧当作未设置）。
        assert_eq!(first_event_timeout_secs(None), 100);
        assert_eq!(idle_timeout_secs(None), 120);
        assert_eq!(first_event_timeout_secs(Some(0)), 100);
        assert_eq!(idle_timeout_secs(Some(0)), 120);
    }

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
            omit_max_tokens: false,
            max_tokens_field: Default::default(),
        prompt_cache: false,
            supports_thinking: false,
            supports_vision: false,
            cost_per_million_input: 0.0,
            cost_per_million_output: 0.0,
            timeout_secs: None,
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
        params.tools.push(Tool { name: "get_weather".into(), description: Some("查天气".into()), parameters: serde_json::json!({"type":"object","properties":{"city":{"type":"string"}}}), strict: None });
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
    fn openai_request_filters_empty_text_parts() {
        // 回归：kimi k3 等 vendor 对 `{"type":"text","text":""}` 直接
        // 400（"text content is empty"）。assistant 只发 tool_calls
        // 不产文本时 content 是空串——空 text part 必须在 wire 构造处
        // 滤掉，否则坏消息进 history 后每个后续请求都 400。
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let assistant = Message::assistant_with_tool_calls(
            "",
            vec![ToolCall {
                id: "call_1".into(),
                name: "bash".into(),
                arguments: serde_json::json!({"command": "ls"}),
                arguments_raw: Some(r#"{"command":"ls"}"#.into()),
                arguments_parse_error: None,
            }],
        );
        let req = client.build_openai_request(&[assistant], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        // 空 text part 被滤掉：content 是空数组，不是 [{"type":"text","text":""}]
        assert_eq!(j["messages"][0]["content"], serde_json::json!([]));
        // tool_calls 不受影响
        assert_eq!(j["messages"][0]["tool_calls"][0]["id"], "call_1");

        // 无 tool_calls 的全空消息：vendor 也拒绝 content: []
        // （"must not be empty"），给占位空格兜底。
        let req = client.build_openai_request(&[Message::user("")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(
            j["messages"][0]["content"],
            serde_json::json!([{"type": "text", "text": " "}])
        );

        // 非空 text part 原样保留（过滤不误伤）。
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(
            j["messages"][0]["content"],
            serde_json::json!([{"type": "text", "text": "hi"}])
        );
    }

    #[test]
    fn anthropic_request_with_tools_serializes_input_schema() {
        // Anthropic 用 input_schema 而不是 parameters。
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool { name: "get_weather".into(), description: Some("查天气".into()), parameters: serde_json::json!({"type":"object"}), strict: None });
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
    fn anthropic_request_hoists_system_to_top_level() {
        // 行为变更：`Role::System` 不再转成 user 消息塞进 messages，而是
        // 提到顶层 `system` 数组。规范缓存顺序是 tools → system →
        // messages，断点必须落在稳定头部才能让"巨大且不变的前缀"命中；
        // 从前的 system-as-user 让头部无处可锚。
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![Message::system("you are helpful"), Message::user("hi")];
        let req = client.build_anthropic_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["system"][0]["type"], "text");
        assert_eq!(j["system"][0]["text"], "you are helpful");
        // messages 里只剩真正的对话消息，system 不再重复下发。
        assert_eq!(j["messages"].as_array().unwrap().len(), 1);
        assert_eq!(j["messages"][0]["role"], "user");
        assert_eq!(j["messages"][0]["content"][0]["text"], "hi");
    }

    /// 4 断点预算分配：tools 末尾 1 + system 末尾 1 + 消息尾部 2。
    ///
    /// 这是 22.7 倍上下文放大的对症解法：工具循环每轮重传全部历史，其中
    /// 最大且最不变的部分（工具定义 + system prompt）每轮重新计费。
    #[test]
    fn anthropic_request_places_four_cache_breakpoints() {
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools = vec![
            Tool {
                name: "a".into(),
                description: Some("first".into()),
                parameters: serde_json::json!({"type":"object"}),
                strict: None,
            },
            Tool {
                name: "b".into(),
                description: Some("last".into()),
                parameters: serde_json::json!({"type":"object"}),
                strict: None,
            },
        ];
        let msgs = vec![
            Message::system("sys one"),
            Message::system("sys two"),
            Message::user("turn 1"),
            Message::assistant("reply 1"),
            Message::user("turn 2"),
        ];
        let req = client.build_anthropic_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();

        // 头部断点 1：只挂最后一个 tool。
        assert!(j["tools"][0].get("cache_control").is_none(), "非末尾 tool 不挂");
        assert_eq!(j["tools"][1]["cache_control"]["type"], "ephemeral");
        // 头部断点 2：只挂最后一个 system block。
        assert!(j["system"][0].get("cache_control").is_none(), "非末尾 system 不挂");
        assert_eq!(j["system"][1]["cache_control"]["type"], "ephemeral");

        // 尾部断点：最近两条消息各一个（滚动窗口），更早的不挂。
        let msgs_json = j["messages"].as_array().unwrap();
        assert_eq!(msgs_json.len(), 3);
        let has_bp = |m: &serde_json::Value| {
            m["content"]
                .as_array()
                .map(|blocks| blocks.iter().any(|b| b.get("cache_control").is_some()))
                .unwrap_or(false)
        };
        assert!(!has_bp(&msgs_json[0]), "最早的消息不该挂断点");
        assert!(has_bp(&msgs_json[1]), "倒数第二条应挂");
        assert!(has_bp(&msgs_json[2]), "最后一条应挂");

        // 总数不超过 Anthropic 的 4 断点上限。
        let count = |v: &serde_json::Value| -> usize {
            fn walk(v: &serde_json::Value, n: &mut usize) {
                match v {
                    serde_json::Value::Object(map) => {
                        if map.contains_key("cache_control") && !map["cache_control"].is_null() {
                            *n += 1;
                        }
                        for (_, sub) in map {
                            walk(sub, n);
                        }
                    }
                    serde_json::Value::Array(arr) => arr.iter().for_each(|s| walk(s, n)),
                    _ => {}
                }
            }
            let mut n = 0;
            walk(v, &mut n);
            n
        };
        assert_eq!(count(&j), 4, "恰好用满 4 个断点、不超预算: {j}");
    }

    /// 没有 system / 没有 tools 时不应崩、也不浪费断点。
    #[test]
    fn anthropic_cache_breakpoints_degrade_without_head() {
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![Message::user("only one turn")];
        let req = client.build_anthropic_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert!(j.get("system").is_none() || j["system"].as_array().map(|a| a.is_empty()).unwrap_or(true));
        assert!(j.get("tools").is_none() || j["tools"].as_array().map(|a| a.is_empty()).unwrap_or(true));
        // 唯一那条消息仍应拿到一个尾部断点。
        assert_eq!(j["messages"][0]["content"][0]["cache_control"]["type"], "ephemeral");
    }

    /// OpenAI-compatible 路径：`prompt_cache` **关**（默认）时一个断点都
    /// 不下发——不认该字段的端点会 400，这是默认关的理由。
    #[test]
    fn openai_cache_breakpoints_are_off_by_default() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![Message::system("sys"), Message::user("hi")];
        let req = client.build_openai_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        let s = serde_json::to_string(&j).unwrap();
        assert!(!s.contains("cache_control"), "默认必须一个断点都不发: {s}");
    }

    /// `prompt_cache` 开启时按 system 头 2 + 尾部 2 放断点，且不超 4。
    #[test]
    fn openai_cache_breakpoints_apply_when_enabled() {
        let mut model = model_with_key(ApiType::OpenAiCompletions, "sk-test");
        model.prompt_cache = true;
        let client = AiClient::new(model).unwrap();
        let params = GenerateParams::default();
        let msgs = vec![
            Message::system("sys one"),
            Message::system("sys two"),
            Message::user("turn 1"),
            Message::assistant("reply 1"),
            Message::user("turn 2"),
        ];
        let req = client.build_openai_request(&msgs, &params, false);
        let j = serde_json::to_value(&req).unwrap();
        let n = serde_json::to_string(&j).unwrap().matches("cache_control").count();
        assert!(n > 0, "开启后应下发断点");
        assert!(n <= 4, "断点数不得超过 4，实际 {n}");
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
        params.tools.push(Tool { name: "x".into(), description: None, parameters: serde_json::json!({}), strict: None });
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"], "auto");
    }

    #[test]
    fn openai_tool_choice_required_emits_string_required() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool { name: "x".into(), description: None, parameters: serde_json::json!({}), strict: None });
        params.tool_choice = ToolChoice::Required;
        let req = client.build_openai_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"], "required");
    }

    #[test]
    fn openai_tool_choice_specific_emits_function_object() {
        let client = AiClient::new(model_with_key(ApiType::OpenAiCompletions, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool { name: "get_weather".into(), description: None, parameters: serde_json::json!({}), strict: None });
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
        params.tools.push(Tool { name: "x".into(), description: None, parameters: serde_json::json!({}), strict: None });
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
        params.tools.push(Tool { name: "x".into(), description: None, parameters: serde_json::json!({}), strict: None });
        params.tool_choice = ToolChoice::Required;
        let req = client.build_anthropic_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert_eq!(j["tool_choice"]["type"], "any");
    }

    #[test]
    fn anthropic_tool_choice_specific_emits_type_tool_with_name() {
        let client = AiClient::new(model_with_key(ApiType::AnthropicMessages, "sk-test")).unwrap();
        let mut params = GenerateParams::default();
        params.tools.push(Tool { name: "get_weather".into(), description: None, parameters: serde_json::json!({}), strict: None });
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
        params.tools.push(Tool { name: "x".into(), description: None, parameters: serde_json::json!({}), strict: None });
        params.tool_choice = ToolChoice::None;
        let req = client.build_anthropic_request(&[Message::user("hi")], &params, false);
        let j = serde_json::to_value(&req).unwrap();
        assert!(j.get("tool_choice").is_none());
    }

    // ─── 流式超时单测 ──────────────────────────────────────────────
    //
    // 用 TcpListener 搭 mock SSE 服务器，验证双层超时模型：
    //   1. idle 超时：首事件到达后，后续 chunk 间隔超过 idle 超时 -> 报错。
    //   2. 长产出不超时：多个 chunk 持续到达（每个间隔 < idle 超时）-> 成功。

    /// 构建测试用 Model，指向 mock 服务器地址。
    fn model_at_addr(addr: std::net::SocketAddr) -> Model {
        Model {
            id: "test-model".into(),
            name: "Test".into(),
            api: ApiType::OpenAiCompletions,
            provider: "test".into(),
            base_url: format!("http://{addr}"),
            api_key: "sk-test".into(),
            context_window: 1024,
            max_tokens: 256,
            omit_max_tokens: false,
            max_tokens_field: Default::default(),
        prompt_cache: false,
            supports_thinking: false,
            supports_vision: false,
            cost_per_million_input: 0.0,
            cost_per_million_output: 0.0,
            timeout_secs: None,
        }
    }

    /// 读取并丢弃 HTTP 请求行 + headers，直到空行。
    async fn drain_http_request(stream: &mut tokio::net::TcpStream) {
        use tokio::io::AsyncReadExt;
        let mut buf = [0u8; 4096];
        // 简单策略：读一次即可，HTTP 请求头通常 < 4KB。
        let _ = stream.read(&mut buf).await;
    }

    /// 发送 SSE 事件。
    async fn send_sse(stream: &mut tokio::net::TcpStream, data: &str) {
        use tokio::io::AsyncWriteExt;
        let line = format!("data: {data}\n\n");
        let _ = stream.write_all(line.as_bytes()).await;
    }

    /// 流式请求必须带 `stream_options.include_usage`，并且能吃下
    /// **choices 为空**的 usage-only chunk。
    ///
    /// 回归 jemalloc 2026-08-26 会话：`chat()` 内部走流式，既没请求
    /// usage、解析又只在 `choices[0].finish_reason` 存在时才读 usage，
    /// 于是全链 token 记账恒为 0（`TurnEnd` 全 0、UI 显示
    /// "tokens: +0 in / +0 out"），成本完全不可见。
    #[tokio::test]
    async fn stream_requests_usage_and_parses_usage_only_chunk() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let body_seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let seen = body_seen.clone();
        tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).await.unwrap_or(0);
            *seen.lock().unwrap() = String::from_utf8_lossy(&buf[..n]).to_string();
            let _ = stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
                .await;
            send_sse(&mut stream, r#"{"choices":[{"delta":{"content":"hi"}}]}"#).await;
            send_sse(
                &mut stream,
                r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
            )
            .await;
            // include_usage 的那条：choices 为空，只带 usage。
            send_sse(
                &mut stream,
                r#"{"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":45}}"#,
            )
            .await;
            send_sse(&mut stream, "[DONE]").await;
        });

        let client = AiClient::new(model_at_addr(addr)).unwrap();
        let c = client
            .chat(&[Message::user("hi")], &GenerateParams::default())
            .await
            .expect("应正常返回");
        assert_eq!(c.usage.input_tokens, 123, "usage-only chunk 必须被采纳");
        assert_eq!(c.usage.output_tokens, 45);

        let body = body_seen.lock().unwrap().clone();
        assert!(
            body.contains("\"stream_options\":{\"include_usage\":true}"),
            "流式请求必须显式要 usage，实际请求：{body}"
        );
    }

    /// 测试 idle 超时：首事件到达后，后续 chunk 间隔超过 idle 超时 -> Stream 错误。
    ///
    /// 方法：mock 服务器发送首个 SSE 事件后等待 3s（idle 超时设为 1s），
    /// 客户端应在 1s idle 超时后返回 AiError::Stream。
    #[tokio::test]
    async fn chat_idle_timeout_triggers_stream_error() {
        // 与 model_timeout_overrides_exactly_not_just_widens 共用一把锁：
        // 本测试改进程级环境变量，不能与断言默认值的测试并发。
        let _guard = TIMEOUT_ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("LATTE_AI_FIRST_EVENT_TIMEOUT_SECS", "5");
        std::env::set_var("LATTE_AI_IDLE_TIMEOUT_SECS", "1");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            drain_http_request(&mut stream).await;
            // 发送 HTTP 响应头
            use tokio::io::AsyncWriteExt;
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n").await;
            // 发送首个事件
            send_sse(&mut stream, r#"{"choices":[{"delta":{"content":"hello"}}]}"#).await;
            // 等待 3s（超过 idle 超时 1s）
            tokio::time::sleep(Duration::from_secs(3)).await;
            // 此时客户端应该已经超时断开了
            send_sse(&mut stream, r#"{"choices":[{"delta":{"content":" world"}}]}"#).await;
        });

        let client = AiClient::new(model_at_addr(addr)).unwrap();
        let result = client.chat(&[Message::user("hi")], &GenerateParams::default()).await;

        assert!(result.is_err(), "应该因 idle 超时而失败");
        match result.unwrap_err() {
            AiError::Stream(msg) => assert!(msg.contains("空闲超时"), "got: {msg}"),
            other => panic!("expected AiError::Stream, got {other:?}"),
        }

        std::env::remove_var("LATTE_AI_FIRST_EVENT_TIMEOUT_SECS");
        std::env::remove_var("LATTE_AI_IDLE_TIMEOUT_SECS");
    }

    /// 测试长产出不超时：多个 SSE chunk 持续到达（每个间隔 < idle 超时），
    /// 总时间无上限 -> 应成功返回完整 Completion。
    ///
    /// 方法：mock 服务器发送 10 个事件（间隔 0.3s），idle 超时设为 2s。
    /// 总时间 3s（远小于旧的总死线 300s，但验证了"只要 token 在流不过期"的机制）。
    #[tokio::test]
    async fn chat_long_output_does_not_timeout() {
        // 与 model_timeout_overrides_exactly_not_just_widens 共用一把锁：
        // 本测试改进程级环境变量，不能与断言默认值的测试并发。
        let _guard = TIMEOUT_ENV_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        std::env::set_var("LATTE_AI_FIRST_EVENT_TIMEOUT_SECS", "5");
        std::env::set_var("LATTE_AI_IDLE_TIMEOUT_SECS", "2");

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            drain_http_request(&mut stream).await;
            use tokio::io::AsyncWriteExt;
            let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n").await;
            // 发送 10 个事件，每个间隔 0.3s（< idle 超时 2s）
            for i in 0..10u32 {
                send_sse(&mut stream, &format!(r#"{{"choices":[{{"delta":{{"content":"chunk{i}"}}}}]}}"#)).await;
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            // 发送 [DONE]
            send_sse(&mut stream, "[DONE]").await;
        });

        let client = AiClient::new(model_at_addr(addr)).unwrap();
        let result = client.chat(&[Message::user("hi")], &GenerateParams::default()).await;

        assert!(result.is_ok(), "长产出不应超时: {:?}", result.err());
        let completion = result.unwrap();
        assert!(completion.content.contains("chunk0"), "content: {}", completion.content);
        assert!(completion.content.contains("chunk9"), "content: {}", completion.content);
        assert_eq!(completion.stop_reason, "", "stop_reason 应为空（无 finish_reason chunk）");

        std::env::remove_var("LATTE_AI_FIRST_EVENT_TIMEOUT_SECS");
        std::env::remove_var("LATTE_AI_IDLE_TIMEOUT_SECS");
    }
    // ─── effective_max_tokens ───────────────────────────────────────
    //
    // 回归 2026-09-07 jemalloc 会话：模型目录里配了 max_tokens = 384000，
    // 而 OpenAI 协议这条路以前直接透传 params.max_tokens（默认 None），
    // 字段被 skip_serializing_if 整个省略 —— 529 次请求下发的全是 None，
    // 真实上限由厂商默认值决定。architect 撞上后 finish_reason=length、
    // total_output 恰好 8192，半个 JSON 穿给下游终审，整条流水线 failed。

    fn model_caps(context_window: u32, max_tokens: u32) -> Model {
        let mut m = model_with_key(ApiType::OpenAiCompletions, "k");
        m.context_window = context_window;
        m.max_tokens = max_tokens;
        m
    }

    #[test]
    fn effective_max_tokens_falls_back_to_catalog_value() {
        let m = model_caps(100_000, 8_192);
        assert_eq!(
            AiClient::effective_max_tokens(None, &m),
            Some(8_192),
            "调用方没显式给时必须回落到目录值，否则字段被省略、厂商默认值说话"
        );
    }

    #[test]
    fn effective_max_tokens_prefers_explicit_over_catalog() {
        let m = model_caps(100_000, 8_192);
        assert_eq!(AiClient::effective_max_tokens(Some(2_048), &m), Some(2_048));
    }

    /// 目录值大于上下文窗口时也**原样下发**。
    ///
    /// 以前这里会压到 `context_window * 7/8`。去掉的理由和去掉天花板一样：
    /// 那是替使用者改数。「输出上限比窗口还大」确实是个配置错误，但该由厂商
    /// 报错点出来，而不是本函数悄悄替它圆过去 —— 悄悄圆过去的代价是使用者
    /// 永远发现不了自己配错了。
    #[test]
    fn oversized_catalog_value_is_sent_verbatim_not_clamped_to_window() {
        let m = model_caps(10_000, 999_999);
        assert_eq!(
            AiClient::effective_max_tokens(None, &m),
            Some(999_999),
            "配置值必须原样下发，不得被窗口钳位"
        );
    }

    /// 契约核心：**配置写多少就发多少**，不存在任何绝对天花板。
    ///
    /// 这条曾经断言 384000 会被压到 64000。反过来钉住它，是因为那个钳制让
    /// `max_tokens` 这个配置项事实上是死的：改它不会有任何可观察的变化。
    /// 现在配错了会收到厂商的 4xx —— 那是可诊断的。
    #[test]
    fn configured_value_is_sent_verbatim_with_no_absolute_ceiling() {
        // 实测配置里的值：目录 384000、窗口 1M
        let m = model_caps(1_000_000, 384_000);
        assert_eq!(
            AiClient::effective_max_tokens(None, &m),
            Some(384_000),
            "目录值必须原样下发"
        );
        // 调用方显式给的值同样不被钳
        assert_eq!(
            AiClient::effective_max_tokens(Some(500_000), &m),
            Some(500_000),
            "显式值必须原样下发"
        );
        // 普通值当然也原样通过
        assert_eq!(AiClient::effective_max_tokens(Some(16_384), &m), Some(16_384));

        // 决定性证据：序列化后的**请求体**里必须真的是 384000。
        // 单测通过不代表字段没被 `skip_serializing_if` 吃掉 —— 那正是本轮
        // 最初那个 bug 的形态（529 次请求全是 None）。
        let client = AiClient::new(model_caps(1_000_000, 384_000)).unwrap();
        let body = serde_json::to_value(client.build_openai_request(
            &[Message::user("hi")],
            &GenerateParams::default(),
            false,
        ))
        .unwrap();
        assert_eq!(
            body.get("max_tokens").and_then(|v| v.as_u64()),
            Some(384_000),
            "请求体必须原样带配置值，实际: {body}"
        );

        // Anthropic 那条路同样原样下发（它以前也走同一个天花板）。
        let a = AiClient::new(model_caps(1_000_000, 384_000)).unwrap();
        let abody = serde_json::to_value(a.build_anthropic_request(
            &[Message::user("hi")],
            &GenerateParams::default(),
            false,
        ))
        .unwrap();
        assert_eq!(
            abody.get("max_tokens").and_then(|v| v.as_u64()),
            Some(384_000),
            "Anthropic 请求体也必须原样带配置值，实际: {abody}"
        );
    }

    /// `omit_max_tokens`：代理转发到未知后端时必须**完全不发**该字段。
    #[test]
    fn omit_max_tokens_suppresses_the_wire_field() {
        let mut m = model_caps(1_000_000, 8_192);
        m.omit_max_tokens = true;
        assert_eq!(AiClient::effective_max_tokens(None, &m), None);
        // 即便调用方显式指定，omit 也优先——它表达的是"这个端点不能收这个字段"
        assert_eq!(AiClient::effective_max_tokens(Some(4_096), &m), None);

        // 序列化后两个字段都不该出现
        let client = AiClient::new(m).unwrap();
        let body = serde_json::to_value(client.build_openai_request(
            &[Message::user("hi")],
            &GenerateParams::default(),
            false,
        ))
        .unwrap();
        assert!(body.get("max_tokens").is_none(), "{body}");
        assert!(body.get("max_completion_tokens").is_none(), "{body}");
    }

    /// 字段名可配：较新的兼容端点只认 `max_completion_tokens`，写死一个会
    /// 静默失效（对方忽略该字段 → 又回到厂商默认值说话）。
    #[test]
    fn max_tokens_field_selects_the_wire_name() {
        use crate::models::MaxTokensField;
        let msgs = vec![Message::user("hi")];
        let params = GenerateParams::default();

        // 默认：max_tokens（必须与改造前的既有行为一致）
        let m1 = model_caps(1_000_000, 8_192);
        assert_eq!(m1.max_tokens_field, MaxTokensField::MaxTokens, "缺省必须是旧字段名");
        let b1 = serde_json::to_value(
            AiClient::new(m1).unwrap().build_openai_request(&msgs, &params, false),
        )
        .unwrap();
        assert_eq!(b1.get("max_tokens").and_then(|v| v.as_u64()), Some(8_192));
        assert!(b1.get("max_completion_tokens").is_none(), "两者互斥: {b1}");

        // 切到 max_completion_tokens
        let mut m2 = model_caps(1_000_000, 8_192);
        m2.max_tokens_field = MaxTokensField::MaxCompletionTokens;
        let b2 = serde_json::to_value(
            AiClient::new(m2).unwrap().build_openai_request(&msgs, &params, false),
        )
        .unwrap();
        assert_eq!(
            b2.get("max_completion_tokens").and_then(|v| v.as_u64()),
            Some(8_192)
        );
        assert!(b2.get("max_tokens").is_none(), "两者互斥: {b2}");
    }

    /// `MaxTokensField` 的 TOML/JSON 反序列化用 snake_case，且缺省等于旧行为。
    #[test]
    fn max_tokens_field_deserializes_from_snake_case() {
        use crate::models::MaxTokensField;
        assert_eq!(
            serde_json::from_str::<MaxTokensField>("\"max_tokens\"").unwrap(),
            MaxTokensField::MaxTokens
        );
        assert_eq!(
            serde_json::from_str::<MaxTokensField>("\"max_completion_tokens\"").unwrap(),
            MaxTokensField::MaxCompletionTokens
        );
        assert_eq!(MaxTokensField::default(), MaxTokensField::MaxTokens);
    }

    #[test]
    fn effective_max_tokens_treats_zero_as_unset() {
        let m = model_caps(10_000, 0);
        assert_eq!(
            AiClient::effective_max_tokens(None, &m),
            None,
            "目录没配（0）时保持省略，退回厂商默认"
        );
    }

    /// 端到端：max_tokens 必须真的出现在序列化后的 OpenAI 请求体里。
    /// 单测 helper 通过不代表字段没被 skip_serializing_if 吃掉。
    #[test]
    fn openai_request_body_carries_max_tokens() {
        let client = AiClient::new(model_caps(100_000, 8_192)).unwrap();
        let msgs = vec![Message::user("hi")];
        let req = client.build_openai_request(&msgs, &GenerateParams::default(), false);
        let body = serde_json::to_value(&req).unwrap();
        assert_eq!(
            body.get("max_tokens").and_then(|v| v.as_u64()),
            Some(8_192),
            "序列化后的请求体必须带 max_tokens，实际: {body}"
        );

        // 目录没配时该字段应完全不出现（而不是发 null）。
        let client2 = AiClient::new(model_caps(100_000, 0)).unwrap();
        let req2 = client2.build_openai_request(&msgs, &GenerateParams::default(), false);
        let body2 = serde_json::to_value(&req2).unwrap();
        assert!(
            body2.get("max_tokens").is_none(),
            "未配置时应省略字段，不能发 null: {body2}"
        );
    }

    /// 两条协议对同一份目录要给出一致的解读（此前只有 Anthropic 有回落）。
    #[test]
    fn both_wires_agree_on_the_catalog_cap() {
        let msgs = vec![Message::user("hi")];
        let params = GenerateParams::default();

        let oai = AiClient::new(model_caps(100_000, 8_192)).unwrap();
        let oai_body = serde_json::to_value(
            oai.build_openai_request(&msgs, &params, false),
        )
        .unwrap();

        let mut am = model_caps(100_000, 8_192);
        am.api = ApiType::AnthropicMessages;
        let ant = AiClient::new(am).unwrap();
        let ant_body =
            serde_json::to_value(ant.build_anthropic_request(&msgs, &params, false)).unwrap();

        assert_eq!(
            oai_body.get("max_tokens").and_then(|v| v.as_u64()),
            ant_body.get("max_tokens").and_then(|v| v.as_u64()),
            "OpenAI 与 Anthropic 两条路对同一份目录应下发同一个上限"
        );
    }

    /// 厂商拒掉输出上限时，错误**如实浮出**，不再偷偷剥掉字段重试。
    ///
    /// 以前这里有一道网：4xx 且错误正文提到 `max_tokens` 就去掉该字段重试。
    /// 它被移除的理由有三条，都在实测中站得住：
    ///
    /// 1. **只认字符串**。触发条件是错误正文里字面出现 `max_tokens`，各家
    ///    文案并不统一（有回 `Range of input length should be [1, N]` 的），
    ///    匹配不上照样硬失败 —— 那道网给的是「有时候能救」的假安全感。
    /// 2. **只覆盖一条协议**。Anthropic 路径从来没有这道网，同一个坏值在
    ///    那边直接失败，两条协议行为分叉。
    /// 3. **代价被低估**。重试会把整个对话历史重发一遍（实测平均
    ///    42K token/次、峰值 1.79M），而且 fallback 顺带把 `stream` 关掉
    ///    —— 一个配错的数字换来「整场对话降级成非流式」。
    ///
    /// 现在的行为：报错原样返回，配置错误一次就暴露。
    #[tokio::test]
    async fn provider_rejecting_max_tokens_surfaces_the_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        server
            .register(
                Mock::given(method("POST"))
                    .and(path("/chat/completions"))
                    .respond_with(ResponseTemplate::new(400).set_body_string(
                        r#"{"error":{"message":"max_tokens is too large: maximum is 8192"}}"#,
                    )),
            )
            .await;

        // 目录配 90000，现在原样下发（不再钳到 64000）
        let mut m = model_caps(1_000_000, 90_000);
        m.base_url = server.uri();
        let client = AiClient::new(m).unwrap();
        let err = client
            .chat(&[Message::user("hi")], &GenerateParams::default())
            .await
            .expect_err("上限被拒必须如实报错，不得静默剥字段重试");
        let msg = err.to_string();
        assert!(
            msg.contains("max_tokens") || msg.contains("8192"),
            "报错应保留厂商原文以便诊断，实际: {msg}"
        );

        // 每一次请求都必须带着配置的值，不存在「偷偷去掉」的那一次
        let reqs = server.received_requests().await.unwrap();
        assert!(!reqs.is_empty(), "至少应发出一次请求");
        for (i, r) in reqs.iter().enumerate() {
            let b: serde_json::Value = serde_json::from_slice(&r.body).unwrap();
            assert_eq!(
                b.get("max_tokens").and_then(|v| v.as_u64()),
                Some(90_000),
                "第 {} 次请求应原样带配置值: {b}",
                i + 1
            );
        }
    }

    /// 与 max_tokens 无关的 4xx 不该触发"去掉 max_tokens"——那会掩盖真实
    /// 原因，让下一次仍然带着错误的值发出去。
    #[tokio::test]
    async fn unrelated_4xx_keeps_max_tokens_on_retry() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        server
            .register(
                Mock::given(method("POST"))
                    .and(path("/chat/completions"))
                    .respond_with(ResponseTemplate::new(400).set_body_string(
                        r#"{"error":{"message":"unknown field 'foo'"}}"#,
                    ))
                    .up_to_n_times(1),
            )
            .await;
        server
            .register(
                Mock::given(method("POST"))
                    .and(path("/chat/completions"))
                    .respond_with(ResponseTemplate::new(200).set_body_string(
                        r#"{"id":"c","object":"chat.completion","created":0,"model":"m",
                            "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},
                            "finish_reason":"stop"}],
                            "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#,
                    )),
            )
            .await;

        let mut m = model_caps(100_000, 8_192);
        m.base_url = server.uri();
        let client = AiClient::new(m).unwrap();
        let _ = client
            .chat(&[Message::user("hi")], &GenerateParams::default())
            .await;
        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 2);
        let b2: serde_json::Value = serde_json::from_slice(&reqs[1].body).unwrap();
        assert_eq!(
            b2.get("max_tokens").and_then(|v| v.as_u64()),
            Some(8_192),
            "非 max_tokens 类错误不该顺手把该字段摘掉: {b2}"
        );
    }

}
