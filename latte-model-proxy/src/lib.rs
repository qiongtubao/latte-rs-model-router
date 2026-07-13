//! `latte-model-proxy` — HTTP proxy server that fronts multiple downstream AI vendors.
//!
//! Exposes endpoints compatible with Ollama (`/api/tags`, `/api/show`,
//! `/api/chat`), OpenAI (`/v1/models`, `/v1/chat/completions`), and
//! Anthropic (`/v1/messages`). Routing and resilience are handled by
//! [`latte_router`]; this crate provides the axum HTTP shell.

pub mod cli;
pub mod image_detect;
pub mod server;

pub use cli::Args;
pub use latte_router::ModelEntry;
pub use server::{Server, ServerHandle, ServerRuntime, serve};
