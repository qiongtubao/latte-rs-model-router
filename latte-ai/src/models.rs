use serde::{Deserialize, Serialize};

/// Supported API types for model providers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ApiType {
    #[serde(rename = "openai-completions")]
    OpenAiCompletions,

    #[serde(rename = "anthropic-messages")]
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

    /// Cost per million input tokens (USD).
    pub cost_per_million_input: f64,

    /// Cost per million output tokens (USD).
    pub cost_per_million_output: f64,
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
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
}

/// A completion response from a model.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The generated text content.
    pub content: String,

    /// Reason why generation stopped.
    pub stop_reason: String,

    /// Token usage statistics.
    pub usage: TokenUsage,
}

/// Token usage statistics.
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub thinking_tokens: u32,
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
    /// A text delta has been received.
    Delta {
        content: String,
        usage: Option<TokenUsage>,
    },
    /// The stream has completed.
    Done {
        content: String,
        usage: TokenUsage,
    },
    /// An error occurred during streaming.
    Error(String),
}

// ─── serialization helpers for OpenAI API ──────────────────────────────

#[derive(Serialize)]
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
    pub stream: bool,
}

#[derive(Serialize)]
pub(crate) struct OpenAiMessage {
    pub role: String,
    pub content: String,
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

#[derive(Deserialize)]
pub(crate) struct OpenAiResponseMessage {
    pub content: Option<String>,
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
}

// ─── serialization helpers for Anthropic API ─────────────────────────

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
}

#[derive(Serialize)]
pub(crate) struct AnthropicMessage {
    pub role: String,
    pub content: String,
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
pub(crate) struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: Option<String>,
    pub thinking: Option<String>,
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
}
