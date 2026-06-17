//! latte-ai — a Rust AI model client library.
//!
//! Supports:
//! - OpenAI chat completions API (universal, used by most providers)
//! - Anthropic messages API
//! - Streaming and non-streaming requests
//! - Thinking/reasoning budget
//!
//! # Quick start
//!
//! ```rust,no_run
//! use latte_ai::prelude::*;
//!
//! # async fn example() -> Result<()> {
//! let model = Model {
//!     id: "deepseek-chat".into(),
//!     name: "DeepSeek Chat".into(),
//!     api: ApiType::OpenAiCompletions,
//!     provider: "deepseek".into(),
//!     base_url: "https://api.deepseek.com".into(),
//!     api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
//!     context_window: 65536,
//!     max_tokens: 8192,
//!     supports_thinking: false,
//!     cost_per_million_input: 0.27,
//!     cost_per_million_output: 1.10,
//! };
//!
//! let client = AiClient::new(model)?;
//! let params = GenerateParams::code_defaults();
//!
//! let completion = client.chat(&[
//!     Message { role: Role::User, content: "Write a Rust function to sum a Vec".into() },
//! ], &params).await?;
//!
//! println!("{}", completion.content);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod error;
pub mod models;
pub mod params;
pub mod vendor;
pub use client::AiClient;
pub use error::{AiError, Result};
pub use models::*;
pub use params::*;

/// Convenience re-exports for library users.
pub mod prelude {
    pub use crate::client::AiClient;
    pub use crate::error::Result;
    pub use crate::models::{
        ApiType, Completion, Message, Model, Role, StreamEvent, TokenUsage,
    };
    pub use crate::params::{GenerateParams, ThinkingBudget};
}
