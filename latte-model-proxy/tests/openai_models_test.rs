//! Tests for OpenAI-compat `/v1/models` (list endpoint).

mod common;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use latte_ai::models::ApiType;
use latte_model_proxy::Server;

#[tokio::test]
async fn get_v1_models_returns_pool_in_openai_shape() {
    let entries = vec![
        common::make_openai_entry("deepseek-chat", "http://x".into()),
        common::make_openai_entry("deepseek-reasoner", "http://x".into()),
        common::make_anthropic_entry("claude-sonnet-4-20250514", "http://x".into()),
    ];
    let server = Server::with_entries(entries, "test".to_string());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let body = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");

    assert_eq!(json.get("object").and_then(|v| v.as_str()), Some("list"));

    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("data array");
    assert_eq!(data.len(), 3);

    let id_set: std::collections::HashSet<&str> = data
        .iter()
        .filter_map(|m| m.get("id").and_then(|v| v.as_str()))
        .collect();
    assert!(id_set.contains("deepseek-chat"));
    assert!(id_set.contains("deepseek-reasoner"));
    assert!(id_set.contains("claude-sonnet-4-20250514"));

    for entry in data {
        assert_eq!(entry.get("object").and_then(|v| v.as_str()), Some("model"));
    }
}

#[tokio::test]
async fn get_v1_models_with_empty_pool_returns_empty_list() {
    let server = Server::with_entries(Vec::new(), "test".to_string());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let body = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    let data = json
        .get("data")
        .and_then(|v| v.as_array())
        .expect("data array");
    assert!(data.is_empty());
}
