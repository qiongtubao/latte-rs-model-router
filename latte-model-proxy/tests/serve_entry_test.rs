//! Tests for the `serve()` entry point.

mod common;

use std::time::Duration;

use latte_model_proxy::{Server, ServerRuntime, serve};
use latte_router::RouterError;

#[tokio::test]
async fn serve_binds_to_configured_address_and_serves_root() {
    let runtime = ServerRuntime {
        router: std::sync::Arc::new(latte_router::Router::with_system_clock(vec![])),
        version: "test".to_string(),
        api_key: None,
        proxy_default_model: "proxy-default".to_string(),
        pool: Vec::new(),
    };
    let handle = serve(runtime, "127.0.0.1:0".to_string())
        .await
        .expect("bind");
    let url = format!("http://{}/", handle.addr);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .unwrap();
    let resp = client.get(&url).send().await.expect("http get");
    assert!(resp.status().is_success());
    let body = resp.text().await.expect("body");
    assert!(body.contains("latte-model-proxy"), "got body {body:?}");
}

#[tokio::test]
async fn server_with_one_entry_selects_that_model() {
    use latte_ai::models::ApiType;
    use latte_model_proxy::ModelEntry;
    use chrono::{TimeZone, Utc};

    let entry = ModelEntry {
        id: "stub".to_string(),
        name: None,
        api: ApiType::OpenAiCompletions,
        provider: "test".to_string(),
        base_url: "http://stub.invalid".to_string(),
        api_key: "k".to_string(),
        context_window: 65536,
        max_tokens: 4096,
        rate_limit_refresh_anchor: Utc.timestamp_opt(0, 0).unwrap(),
        rate_limit_refresh_interval_secs: 60,
        retry_count_5xx: 5,
        cooldown_5xx_secs: 600,
        retry_on: vec![403],
        retry_on_count: 10,
        retry_on_cooldown_secs: 600,
        supports_vision: false,
    };
    let server = Server::with_entries(vec![entry], "test".to_string());

    match server.router().select("stub") {
        Ok(route) => {
            assert_eq!(route.model_id, "stub");
        }
        Err(RouterError::AllUnavailable { .. }) | Err(RouterError::UnknownModel(_)) => {
            panic!("expected stub to be selectable")
        }
        Err(e) => panic!("unexpected error: {e}"),
    }

    match server.router().select("missing") {
        Err(RouterError::UnknownModel(name)) => assert_eq!(name, "missing"),
        other => panic!("expected UnknownModel, got {other:?}"),
    }
}
