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
}

impl Message {
    /// Build a single-text-part message with the given role.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::text(text)],
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
    pub stream: bool,
}

#[derive(Serialize, Clone)]
pub(crate) struct OpenAiMessage {
    pub role: String,
    pub content: Vec<OpenAiContentPart>,
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

#[derive(Deserialize)]
pub(crate) struct OpenAiResponseMessage {
    /// OpenAI returns `null` for assistant tool-call turns; `Some([])` is
    /// technically possible but not in practice. Treated as "no text".
    #[serde(default)]
    pub content: Option<Vec<OpenAiResponseContentPart>>,
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
    /// Streamed text delta. OpenAI streams only text deltas, not image
    /// responses, so this stays as `Option<String>`.
    pub content: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
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
        let j = serde_json::json!({"content": [{"type": "text", "text": "hi"}]});
        let m: OpenAiResponseMessage = serde_json::from_value(j).unwrap();
        let parts = m.content.unwrap();
        assert_eq!(parts.len(), 1);
        match &parts[0] {
            OpenAiResponseContentPart::Text { text } => assert_eq!(text, "hi"),
            OpenAiResponseContentPart::Other => panic!("expected text part"),
        }
    }
}
