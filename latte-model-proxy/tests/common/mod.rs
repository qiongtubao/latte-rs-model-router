//! Shared helpers for proxy integration tests.
//! Some functions may go "unused" in individual test files; that's expected.
#![allow(dead_code)]

use chrono::{TimeZone, Utc};
use latte_ai::models::ApiType;
use latte_model_proxy::{ModelEntry, Server, ServerRuntime};
use latte_router::Router;
use std::sync::Arc;

pub fn make_entry(id: &str, api: ApiType, base_url: String) -> ModelEntry {
    ModelEntry {
        id: id.to_string(),
        name: None,
        api,
        provider: "test".to_string(),
        base_url,
        api_key: "sk-test".to_string(),
        context_window: 65536,
        max_tokens: 4096,
        rate_limit_refresh_anchor: Utc.timestamp_opt(0, 0).unwrap(),
        rate_limit_refresh_interval_secs: 60,
        retry_count_5xx: 5,
        cooldown_5xx_secs: 600,
        retry_on: vec![403],
        retry_on_count: 10,
        retry_on_cooldown_secs: 600,
    }
}

pub fn make_anthropic_entry(id: &str, base_url: String) -> ModelEntry {
    make_entry(id, ApiType::AnthropicMessages, base_url)
}

pub fn make_openai_entry(id: &str, base_url: String) -> ModelEntry {
    make_entry(id, ApiType::OpenAiCompletions, base_url)
}

pub fn router_with(entries: Vec<ModelEntry>) -> Arc<Router> {
    Arc::new(Router::with_system_clock(entries))
}

/// Build a test server from just model entries.
pub fn make_server(entries: Vec<ModelEntry>) -> Server {
    Server::with_entries(entries, "test".to_string())
}

/// Build a `ServerRuntime` with the given pool (silent-selection candidates)
/// and the full catalog as the router pool. Used in tests that exercise the
/// `proxy-default` silent-selection path.
pub fn make_runtime_with_pool(entries: Vec<ModelEntry>, pool: Vec<String>) -> ServerRuntime {
    ServerRuntime {
        router: Arc::new(Router::with_system_clock(entries)),
        version: "test".to_string(),
        api_key: None,
        proxy_default_model: "proxy-default".to_string(),
        pool,
    }
}
