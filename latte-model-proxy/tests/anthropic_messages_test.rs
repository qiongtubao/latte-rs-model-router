//! Tests for Anthropic-compat `/v1/messages`.

mod common;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use latte_model_proxy::Server;

#[tokio::test]
async fn post_v1_messages_forwards_to_anthropic_vendor() {
    let mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "sk-test"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{ "type": "text", "text": "ok" }],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        })))
        .expect(1)
        .mount(&mock)
        .await;

    let entry = common::make_anthropic_entry("claude-sonnet-4-20250514", mock.uri());
    let server = Server::with_entries(vec![entry], "test".to_string());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1024,
                        "messages": [{ "role": "user", "content": "hi" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(json.get("id").and_then(|v| v.as_str()), Some("msg_01"));
    let text = json
        .get("content")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("text"))
        .and_then(|v| v.as_str());
    assert_eq!(text, Some("ok"));
}

#[tokio::test]
async fn post_v1_messages_returns_400_when_openai_model_called() {
    let entry = common::make_openai_entry("deepseek-chat", "http://x".into());
    let server = Server::with_entries(vec![entry], "test".to_string());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "deepseek-chat",
                        "max_tokens": 1024,
                        "messages": [{ "role": "user", "content": "hi" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}
