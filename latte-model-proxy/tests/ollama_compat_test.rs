//! Tests for Ollama-compat endpoints (`/api/tags`, `/api/show`, `/api/chat`).

mod common;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use latte_model_proxy::Server;

fn server_with(entries: Vec<latte_model_proxy::ModelEntry>) -> Server {
    Server::with_entries(entries, "test".to_string())
}

#[tokio::test]
async fn get_api_tags_lists_every_pool_model() {
    let entries = vec![
        common::make_openai_entry("deepseek-chat", "http://x".into()),
        common::make_openai_entry("deepseek-reasoner", "http://x".into()),
        common::make_anthropic_entry("claude-sonnet-4-20250514", "http://x".into()),
    ];
    let server = server_with(entries);

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .uri("/api/tags")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let body = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let models = json
        .get("models")
        .and_then(|v| v.as_array())
        .expect("models array");
    assert_eq!(models.len(), 3);

    let ids: std::collections::HashSet<String> = models
        .iter()
        .filter_map(|m| m.get("name").and_then(|v| v.as_str()).map(String::from))
        .collect();
    assert!(ids.contains("deepseek-chat"));
    assert!(ids.contains("deepseek-reasoner"));
    assert!(ids.contains("claude-sonnet-4-20250514"));
}

#[tokio::test]
async fn post_api_show_returns_detail_for_known_model() {
    let entries = vec![common::make_openai_entry(
        "deepseek-chat",
        "http://x".into(),
    )];
    let server = server_with(entries);

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/show")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"deepseek-chat"}"#))
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let body = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(json.get("name").and_then(|v| v.as_str()), Some("deepseek-chat"));
}

#[tokio::test]
async fn post_api_show_returns_404_for_unknown_model() {
    let server = server_with(Vec::new());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/show")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"name":"missing"}"#))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), 404);
}

#[tokio::test]
async fn post_api_chat_translates_ollama_shape_to_openai_vendor() {
    let mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "deepseek-chat",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "hello" },
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&mock)
        .await;

    let entry = common::make_openai_entry("deepseek-chat", mock.uri());
    let server = server_with(vec![entry]);

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "deepseek-chat",
                        "messages": [{ "role": "user", "content": "hi" }],
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), 200);
    let body = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(json.get("model").and_then(|v| v.as_str()), Some("deepseek-chat"));
    let content = json
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str());
    assert_eq!(content, Some("hello"));
}
