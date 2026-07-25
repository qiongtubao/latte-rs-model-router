use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};

/// Supported API types for model providers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ApiType {
    #[serde(rename = "openai-completions", alias = "openai")]
    OpenAiCompletions,

    #[serde(rename = "anthropic-messages", alias = "anthropic")]
    AnthropicMessages,
}

/// A model definition with its capabilities and connection details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Model identifier (e.g. "claude-sonnet-4-20250514", "deepseek-chat").
    pub id: String,

    /// Human-readable name.
    pub name: String,

    /// API type.
    pub api: ApiType,

    /// Provider name (e.g. "anthropic", "deepseek", "openai").
    pub provider: String,

    /// Base URL for API requests.
    pub base_url: String,

    /// API key for authentication.
    #[serde(skip_serializing)]
    pub api_key: String,

    /// Context window size in tokens.
    pub context_window: u32,

    /// Maximum output tokens.
    pub max_tokens: u32,

    /// Whether the model supports thinking/reasoning.
    pub supports_thinking: bool,

    /// Whether the model can accept image inputs (multimodal / vision).
    /// Defaults to `false` so existing TOML configs that pre-date this
    /// field keep loading unchanged.
    #[serde(default)]
    pub supports_vision: bool,

    /// Cost per million input tokens (USD).
    pub cost_per_million_input: f64,

    /// Cost per million output tokens (USD).
    pub cost_per_million_output: f64,
}

/// One piece of a message's content — either plain text or an inline image.
///
/// Image data is kept as raw bytes at the library level. The provider-specific
/// client (OpenAI / Anthropic) is responsible for base64-encoding it into the
/// wire format the vendor expects. Callers don't need to know the encoding
/// detail; they hand over `Vec<u8>` plus a MIME type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text segment.
    Text { text: String },
    /// An image attachment. `data` is the raw byte payload (e.g. PNG bytes);
    /// the provider client base64-encodes it on the way out.
    Image {
        /// MIME type, e.g. `"image/png"`, `"image/jpeg"`, `"image/webp"`.
        media_type: String,
        /// Raw image bytes. Empty is allowed (treat as malformed upstream).
        data: Vec<u8>,
    },
}

impl ContentPart {
    /// Convenience constructor for a text part.
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    /// Convenience constructor for an image part.
    pub fn image(media_type: impl Into<String>, data: Vec<u8>) -> Self {
        Self::Image {
            media_type: media_type.into(),
            data,
        }
    }
}

/// A message in a chat conversation.
///
/// `content` is a vector of `ContentPart` to support multimodal input
/// (text + image). The common "just text" case stays readable via the
/// `Message::text` / `Message::user` / `Message::system` helpers, and
/// `Message::with_image` adds an image onto an existing message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    /// `Role::Tool` 消息对应的 assistant `ToolCall.id`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Assistant 消息携带的它自己请求调用的工具列表。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Build a single-text-part message with the given role.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::text(text)],
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Build a `Role::Tool` 消息：把工具执行结果回传给模型。
    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentPart::text(content)],
            tool_call_id: Some(id.into()),
            tool_calls: None,
        }
    }

    /// Assistant 消息携带它请求调用的工具。多轮里把 Completion.tool_calls
    /// 喂回 history 时用。
    pub fn assistant_with_tool_calls(text: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentPart::text(text)],
            tool_call_id: None,
            tool_calls: Some(calls),
        }
    }

    /// Build a user-role text message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::text(Role::User, text)
    }

    /// Build a system-role text message.
    pub fn system(text: impl Into<String>) -> Self {
        Self::text(Role::System, text)
    }

    /// Build an assistant-role text message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::text(Role::Assistant, text)
    }

    /// Append an image to this message's content.
    pub fn with_image(mut self, media_type: impl Into<String>, data: Vec<u8>) -> Self {
        self.content.push(ContentPart::image(media_type, data));
        self
    }

    /// Owned convenience: joined text or empty string. Most callers that
    /// historically treated `Message::content` as a `String` should reach
    /// for this. Skips image parts entirely.
    pub fn as_text(&self) -> String {
        self.joined_text().unwrap_or_default()
    }

    /// Whether this message contains at least one image content part.
    /// The `Agent::request` method uses this to skip non-vision models
    /// in the model_chain when `has_image()` is true. Text-only
    /// messages can still use any model regardless of `supports_vision`.
    pub fn has_image(&self) -> bool {
        self.content.iter().any(|p| matches!(p, ContentPart::Image { .. }))
    }

    /// Returns all text parts joined, or `None` when
    /// Concatenate all `Text` parts into a single string. Returns `None` if
    /// the message has no text at all (image-only).
    pub fn joined_text(&self) -> Option<String> {
        let mut out = String::new();
        let mut any = false;
        for p in &self.content {
            if let ContentPart::Text { text } = p {
                out.push_str(text);
                any = true;
            }
        }
        if any { Some(out) } else { None }
    }
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Role {
    #[serde(rename = "system")]
    System,

    #[serde(rename = "user")]
    User,

    #[serde(rename = "assistant")]
    Assistant,

    /// `tool` 角色：前一轮 assistant 调用工具的结果回传。
    #[serde(rename = "tool")]
    Tool,
}

/// A completion response from a model.
#[derive(Debug, Clone)]
pub struct Completion {
    /// 文本便捷视图：所有 `ContentPart::Text` 的拼接。
    pub content: String,

    /// 结构化 content（text + image）。
    pub content_parts: Vec<ContentPart>,

    /// 模型请求调用方执行的工具列表。
    pub tool_calls: Vec<ToolCall>,

    /// Reason why generation stopped.
    pub stop_reason: String,

    /// Token usage statistics.
    pub usage: TokenUsage,
}

impl Completion {
    /// Joined text of all `ContentPart::Text` parts. Equivalent to
    /// `self.content` —— 给历史代码用。
    pub fn as_text(&self) -> String {
        self.content.clone()
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub thinking_tokens: u32,
}

/// 工具描述（请求侧）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// 模型请求调用某个工具（响应侧）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_raw: Option<String>,
    /// 模型输出的 `arguments` JSON 解析失败时的错误信息。
    /// `None` 表示 `arguments` 解析成功（或者还没尝试解析）。
    /// 解析失败时 `arguments` 会 fallback 成 `Value::Null`，
    /// 但 `arguments_raw` 和这个字段让调用方可以：
    /// - 知道发生了错误（避免 panic 在 `arguments["key"]`）
    /// - 拿到原始串做手工解析 / fallback / debug
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_parse_error: Option<String>,
}


/// 工具调用策略：控制模型是否调工具 / 调哪个 / 必须调。
///
/// OpenAI 协议下：`Auto` → `"auto"`, `None` → `"none"`, `Required` →
/// `"required"`, `Specific(name)` → `{type:"function", function:{name}}`。
///
/// Anthropic 协议下：`Auto` → `{type:"auto"}`, `Required` →
/// `{type:"any"}`, `Specific(name)` → `{type:"tool", name}`。`None` 在
/// Anthropic 协议下不支持（不传 tools 就够了），会被跳过。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// 让模型自己决定要不要调（默认行为）。
    Auto,
    /// 显式禁止调工具。Anthropic 不支持，会跳过。
    None,
    /// 必须调至少一个工具。Anthropic 用 `any`。
    Required,
    /// 必须调指定名称的工具。
    Specific(String),
}

impl Default for ToolChoice {
    fn default() -> Self {
        ToolChoice::Auto
    }
}

/// OpenAI 协议 tool_choice wire 类型 —— 不同的变体序列化到不同的形状。
#[derive(Serialize, Clone)]
#[serde(untagged)]
pub(crate) enum OpenAiToolChoice {
    /// `"auto" | "none" | "required"`
    Keyword(&'static str),
    /// `{type:"function", function:{name:"..."}}`
    Specific { #[serde(rename = "type")] type_: &'static str, function: OpenAiToolChoiceName },
}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiToolChoiceName {
    pub name: String,
}

impl OpenAiToolChoice {
    pub(crate) fn from(c: &ToolChoice) -> Self {
        match c {
            ToolChoice::Auto => OpenAiToolChoice::Keyword("auto"),
            ToolChoice::None => OpenAiToolChoice::Keyword("none"),
            ToolChoice::Required => OpenAiToolChoice::Keyword("required"),
            ToolChoice::Specific(name) => OpenAiToolChoice::Specific {
                type_: "function",
                function: OpenAiToolChoiceName { name: name.clone() },
            },
        }
    }
}

/// Anthropic 协议 tool_choice wire 类型。
#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicToolChoice {
    Auto,
    /// `{type:"any"}` —— 必须调一个工具
    Any,
    /// `{type:"tool", name:"..."}` —— 必须调指定工具
    Tool { name: String },
}

impl AnthropicToolChoice {
    /// 从统一的 `ToolChoice` 转换。`ToolChoice::None` 在 Anthropic 协议下
    /// 没有对应形态 —— 返回 `None` 让调用方跳过序列化。
    pub(crate) fn from(c: &ToolChoice) -> Option<Self> {
        match c {
            ToolChoice::Auto => Some(AnthropicToolChoice::Auto),
            ToolChoice::Required => Some(AnthropicToolChoice::Any),
            ToolChoice::Specific(name) => Some(AnthropicToolChoice::Tool { name: name.clone() }),
            ToolChoice::None => None,
        }
    }
}

impl std::fmt::Display for TokenUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "↑{} ↓{}", self.input_tokens, self.output_tokens)?;
        if self.thinking_tokens > 0 {
            write!(f, " ({} thinking)", self.thinking_tokens)?;
        }
        Ok(())
    }
}

/// A stream event during generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    Delta {
        content: Vec<ContentPart>,
        usage: Option<TokenUsage>,
    },
    Done {
        content: Vec<ContentPart>,
        tool_calls: Vec<ToolCall>,
        usage: TokenUsage,
    },
    Error(String),
}

// ─── OpenAI wire format ──────────────────────────────────────────
//
// OpenAI Chat Completions uses an array of content parts under `content`:
//   {"type": "text",      "text": "..."}
//   {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<OpenAiToolChoice>,
    pub stream: bool,
}

#[derive(Serialize, Clone)]
#[serde(untagged)]
pub(crate) enum OpenAiMessageContent {
    Parts(Vec<OpenAiContentPart>),
    ToolString(String),
}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiMessage {
    pub role: String,
    pub content: OpenAiMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Serialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OpenAiContentPart {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiImageUrl {
    /// Always a `data:` URL — the library does not pass external URLs through.
    pub url: String,
}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiTool {
    #[serde(rename = "type")]
    pub type_: String,
    pub function: OpenAiFunction,
}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Deserialize, Clone, Serialize)]
pub(crate) struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub type_: Option<String>,
    pub function: OpenAiFunctionCall,
}

#[derive(Deserialize, Clone, Serialize)]
pub(crate) struct OpenAiFunctionCall {
    pub name: String,
    /// 工具参数原始 JSON 字符串。
    pub arguments: String,
}

#[derive(Deserialize, Clone)]
pub(crate) struct OpenAiToolCallDelta {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub type_: Option<String>,
    #[serde(default)]
    pub function: Option<OpenAiFunctionCallDelta>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct OpenAiFunctionCallDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Convert a `ContentPart` to the OpenAI wire format. Images get base64'd
/// into a `data:` URL.
pub(crate) fn to_openai_content_part(p: &ContentPart) -> OpenAiContentPart {
    match p {
        ContentPart::Text { text } => OpenAiContentPart::Text { text: text.clone() },
        ContentPart::Image { media_type, data } => {
            let url = format!("data:{};base64,{}", media_type, BASE64.encode(data));
            OpenAiContentPart::ImageUrl {
                image_url: OpenAiImageUrl { url },
            }
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct OpenAiChatResponse {
    pub choices: Vec<OpenAiChoice>,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiChoice {
    pub message: OpenAiResponseMessage,
    pub finish_reason: Option<String>,
}

/// **OpenAI 协议里 `message.content` 同时有两种合法形态**：
/// - `String` — 简单文本响应（minimax / DeepSeek / 大多数第三方 vendor 默认走这个）
/// - `Vec<OpenAiResponseContentPart>` — 标准 OpenAI 官方 / 多模态响应
///
/// 之前写死 `Option<Vec<...>>` 是 bug：minimax 等 vendor 总是返回 string，
/// 反序列化直接 `Serde` 错。`#[serde(untagged)]` 让两种都能 match 上。
/// `merge_openai_text` 同时支持两种形态并把 text parts 拼起来。
#[derive(Deserialize, Clone)]
#[serde(untagged)]
pub(crate) enum OpenAiResponseContent {
    /// minimax / DeepSeek / 多数国产 vendor 默认形态。
    Plain(String),
    /// OpenAI 官方 + vision 场景。
    Parts(Vec<OpenAiResponseContentPart>),
}

#[derive(Deserialize)]
pub(crate) struct OpenAiResponseMessage {
    #[serde(default)]
    pub content: Option<OpenAiResponseContent>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
}

#[derive(Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum OpenAiResponseContentPart {
    Text { text: String },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamChunk {
    pub choices: Vec<OpenAiStreamChoice>,
    pub usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiStreamChoice {
    pub delta: OpenAiDelta,
    pub finish_reason: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct OpenAiDelta {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OpenAiToolCallDelta>,
}

// ─── Anthropic wire format ────────────────────────────────────────
//
// Anthropic Messages API also uses content blocks:
//   {"type": "text",  "text": "..."}
//   {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}

#[derive(Serialize)]
pub(crate) struct AnthropicRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
}

#[derive(Serialize, Clone)]
pub(crate) struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Serialize, Clone)]
pub(crate) struct AnthropicMessage {
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum AnthropicContentBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct AnthropicImageSource {
    /// Always `"base64"` — the only inline image mode Anthropic supports.
    #[serde(rename = "type")]
    pub type_: String,
    pub media_type: String,
    pub data: String,
}

/// Convert a `ContentPart` to the Anthropic wire format. Images are
/// base64-encoded into the `source.data` field.
pub(crate) fn to_anthropic_content_block(p: &ContentPart) -> AnthropicContentBlock {
    match p {
        ContentPart::Text { text } => AnthropicContentBlock::Text { text: text.clone() },
        ContentPart::Image { media_type, data } => AnthropicContentBlock::Image {
            source: AnthropicImageSource {
                type_: "base64".to_string(),
                media_type: media_type.clone(),
                data: BASE64.encode(data),
            },
        },
    }
}

#[derive(Serialize)]
pub(crate) struct AnthropicThinking {
    #[serde(rename = "type")]
    pub type_: String,
    pub budget_tokens: u32,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicResponse {
    pub content: Vec<AnthropicContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default)]
    pub index: Option<u32>,
    pub delta: Option<AnthropicStreamDelta>,
    pub content_block: Option<AnthropicContentBlock>,
    pub message: Option<AnthropicResponse>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Deserialize)]
pub(crate) struct AnthropicStreamDelta {
    pub text: Option<String>,
    #[serde(rename = "type")]
    pub type_: Option<String>,
    pub thinking: Option<String>,
    #[serde(default)]
    pub partial_json: Option<String>,
}

// ─── Response extractors ──────────────────────────────────────────

pub(crate) fn extract_openai_response(
    content: Option<OpenAiResponseContent>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
) -> (Vec<ContentPart>, Vec<ToolCall>) {
    let parts: Vec<ContentPart> = match content {
        None => Vec::new(),
        Some(OpenAiResponseContent::Plain(text)) => {
            if text.is_empty() {
                Vec::new()
            } else {
                vec![ContentPart::Text { text }]
            }
        }
        Some(OpenAiResponseContent::Parts(ps)) => ps
            .into_iter()
            .map(|p| match p {
                OpenAiResponseContentPart::Text { text } => ContentPart::Text { text },
                OpenAiResponseContentPart::Other => ContentPart::Text { text: String::new() },
            })
            .collect(),
    };
    let calls: Vec<ToolCall> = tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| {
            // 解析失败时把错误信息写进 `arguments_parse_error`，
            // 让调用方知道这里出错了 —— 避免 `arguments["city"]` 这种
            // 访问直接 panic。`arguments` fallback 成 `Value::Null`，
            // `arguments_raw` 留原串给调用方手工 fallback。
            match serde_json::from_str(&tc.function.arguments) {
                Ok(parsed) => ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: parsed,
                    arguments_raw: Some(tc.function.arguments),
                    arguments_parse_error: None,
                },
                Err(e) => ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: serde_json::Value::Null,
                    arguments_raw: Some(tc.function.arguments),
                    arguments_parse_error: Some(e.to_string()),
                },
            }
        })
        .collect();
    (parts, calls)
}

pub(crate) fn extract_anthropic_response(
    blocks: Vec<AnthropicContentBlock>,
) -> (Vec<ContentPart>, Vec<ToolCall>) {
    let mut parts: Vec<ContentPart> = Vec::new();
    let mut calls: Vec<ToolCall> = Vec::new();
    for b in blocks {
        match b {
            AnthropicContentBlock::Text { text } => parts.push(ContentPart::Text { text }),
            AnthropicContentBlock::Image { source } => match BASE64.decode(&source.data) {
                Ok(data) => parts.push(ContentPart::Image {
                    media_type: source.media_type,
                    data,
                }),
                Err(_) => parts.push(ContentPart::Text {
                    text: format!("[image decode failed: {}]", source.media_type),
                }),
            },
            AnthropicContentBlock::ToolUse { id, name, input } => calls.push(ToolCall {
                id, name,
                arguments: input,
                // Anthropic 协议里 tool_use.input 已经是 Value，
                // 不需要 JSON parse，所以一定没有错误。
                arguments_raw: None,
                arguments_parse_error: None,
            }),
            AnthropicContentBlock::ToolResult { content, .. } => {
                parts.push(ContentPart::Text { text: content });
            }
        }
    }
    (parts, calls)
}

// ─── tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_part_text_serializes_as_snake_case_tag() {
        let p = ContentPart::text("hello");
        let j = serde_json::to_value(&p).unwrap();
        assert_eq!(j, serde_json::json!({"type": "text", "text": "hello"}));
    }

    #[test]
    fn message_user_helper_is_single_text_part() {
        let m = Message::user("hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content.len(), 1);
        assert_eq!(
            m.content[0],
            ContentPart::Text { text: "hi".to_string() }
        );
    }

    #[test]
    fn message_with_image_appends_after_text() {
        let m = Message::user("describe")
            .with_image("image/png", vec![0x89, 0x50, 0x4e, 0x47]);
        assert_eq!(m.content.len(), 2);
        assert!(matches!(m.content[0], ContentPart::Text { .. }));
        match &m.content[1] {
            ContentPart::Image { media_type, data } => {
                assert_eq!(media_type, "image/png");
                assert_eq!(data, &vec![0x89, 0x50, 0x4e, 0x47]);
            }
            _ => panic!("expected image part"),
        }
    }

    #[test]
    fn joined_text_concatenates_text_parts() {
        let m = Message::user("").with_image("image/png", vec![1, 2, 3]);
        // Empty text + image → joined_text returns Some("") (text exists, even if empty)
        assert_eq!(m.joined_text().as_deref(), Some(""));
    }

    #[test]
    fn to_openai_content_part_wraps_image_as_data_url() {
        let p = ContentPart::image("image/png", b"abc".to_vec());
        let wire = to_openai_content_part(&p);
        let j = serde_json::to_value(&wire).unwrap();
        let expected_b64 = BASE64.encode(b"abc");
        assert_eq!(
            j,
            serde_json::json!({
                "type": "image_url",
                "image_url": {"url": format!("data:image/png;base64,{}", expected_b64)}
            })
        );
    }

    #[test]
    fn to_openai_content_part_passes_text_through() {
        let p = ContentPart::text("hi");
        let wire = to_openai_content_part(&p);
        let j = serde_json::to_value(&wire).unwrap();
        assert_eq!(j, serde_json::json!({"type": "text", "text": "hi"}));
    }

    #[test]
    fn to_anthropic_content_block_image_uses_base64_source() {
        let p = ContentPart::image("image/jpeg", b"\xff\xd8".to_vec());
        let wire = to_anthropic_content_block(&p);
        let j = serde_json::to_value(&wire).unwrap();
        let expected_b64 = BASE64.encode(b"\xff\xd8");
        assert_eq!(
            j,
            serde_json::json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": expected_b64,
                }
            })
        );
    }

    #[test]
    fn to_anthropic_content_block_passes_text_through() {
        let p = ContentPart::text("hi");
        let wire = to_anthropic_content_block(&p);
        let j = serde_json::to_value(&wire).unwrap();
        assert_eq!(j, serde_json::json!({"type": "text", "text": "hi"}));
    }

    #[test]
    fn model_supports_vision_defaults_to_false_for_backcompat() {
        // TOML without the field should still load — serde(default) on bool
        // is `false`. Use a hand-rolled JSON to simulate missing field.
        let j = serde_json::json!({
            "id": "x", "name": "X", "api": "openai", "provider": "p",
            "base_url": "https://x", "api_key": "k", "context_window": 1000,
            "max_tokens": 100, "supports_thinking": false,
            "cost_per_million_input": 0.0, "cost_per_million_output": 0.0
        });
        let m: Model = serde_json::from_value(j).unwrap();
        assert!(!m.supports_vision);
    }

    #[test]
    fn openai_response_message_with_null_content_deserializes() {
        // Common case: assistant tool-call turn returns content: null.
        let j = serde_json::json!({"content": null});
        let m: OpenAiResponseMessage = serde_json::from_value(j).unwrap();
        assert!(m.content.is_none());
    }

    #[test]
    fn openai_response_message_with_missing_content_field_deserializes() {
        // Some OpenAI-compatible vendors omit `content` entirely.
        let j = serde_json::json!({});
        let m: OpenAiResponseMessage = serde_json::from_value(j).unwrap();
        assert!(m.content.is_none());
    }

    #[test]
    fn openai_response_message_with_text_part_deserializes() {
        // OpenAI 官方形态：`content` 是 parts 数组。
        let j = serde_json::json!({"content": [{"type": "text", "text": "hi"}]});
        let m: OpenAiResponseMessage = serde_json::from_value(j).unwrap();
        let content = m.content.unwrap();
        match content {
            OpenAiResponseContent::Parts(parts) => {
                assert_eq!(parts.len(), 1);
                match &parts[0] {
                    OpenAiResponseContentPart::Text { text } => assert_eq!(text, "hi"),
                    OpenAiResponseContentPart::Other => panic!("expected text part"),
                }
            }
            OpenAiResponseContent::Plain(_) => panic!("expected Parts variant"),
        }
    }

    /// **直接复现 minimax 形态**：`content` 是 string 而非 array。
    /// 这是 `error decoding response body` panic 的根因 —— 旧版 `OpenAiResponseMessage`
    /// 写死 `Vec<...>`，minimax 返 string 就 Serde 错。
    /// 这个测试是回归保险：以后谁改回 `Vec<...>` 写法就会立刻挂掉。
    #[test]
    fn openai_response_message_with_plain_string_content_deserializes() {
        // 模拟 minimax 在 `content: [{"type":"text","text":"..."}]` 请求下
        // 实际返回的 body —— content 是 string，带 `<think>` 块。
        let body = r#"{
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "<think>The user asks...</think>\n\n我是 MiniMax-M3, 由 MiniMax 开发。",
                    "role": "assistant"
                }
            }],
            "usage": {"prompt_tokens": 180, "completion_tokens": 75}
        }"#;
        let resp: OpenAiChatResponse = serde_json::from_str(body).unwrap();
        let choice = resp.choices.into_iter().next().unwrap();
        match choice.message.content.unwrap() {
            OpenAiResponseContent::Plain(s) => {
                assert!(s.contains("<think>"));
                assert!(s.contains("MiniMax-M3"));
            }
            OpenAiResponseContent::Parts(_) => panic!("minimax should return Plain variant"),
        }
    }
    /// `extract_openai_response` 处理非法 JSON 的 `arguments`：把
    /// `arguments` fallback 成 `Value::Null`，但同时把错误信息写到
    /// `arguments_parse_error`，让调用方知道这里出错了 —— 避免直接
    /// `arguments["city"]` panic。
    #[test]
    fn extract_openai_response_marks_malformed_arguments() {
        let wire_tcs = vec![OpenAiToolCall {
            id: "call_bad".into(),
            type_: Some("function".into()),
            function: OpenAiFunctionCall {
                name: "get_weather".into(),
                // 不是合法 JSON —— 缺右括号
                arguments: r#"{"city":"北京"#.into(),
            },
        }];
        let (_parts, calls) = extract_openai_response(None, Some(wire_tcs));
        assert_eq!(calls.len(), 1);
        // arguments fallback 成 Null
        assert_eq!(calls[0].arguments, serde_json::Value::Null);
        // raw 串保留
        assert_eq!(calls[0].arguments_raw.as_deref(), Some("{\"city\":\"北京"));
        // parse error 写进字段
        let err = calls[0].arguments_parse_error.as_ref()
            .expect("malformed JSON should set arguments_parse_error");
        assert!(err.contains("EOF") || err.contains("expected"),
            "expected serde error msg, got: {err}");
    }

    /// 合法 JSON 的 `arguments` 应该 `arguments_parse_error: None`。
    #[test]
    fn extract_openai_response_clears_parse_error_on_success() {
        let wire_tcs = vec![OpenAiToolCall {
            id: "call_ok".into(),
            type_: Some("function".into()),
            function: OpenAiFunctionCall {
                name: "x".into(),
                arguments: r#"{"k":"v"}"#.into(),
            },
        }];
        let (_parts, calls) = extract_openai_response(None, Some(wire_tcs));
        assert!(calls[0].arguments_parse_error.is_none());
        assert_eq!(calls[0].arguments["k"], "v");
    }

}
