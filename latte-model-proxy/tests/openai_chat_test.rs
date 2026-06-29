//! Tests for `/v1/chat/completions` non-stream forwarding.

mod common;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use latte_model_proxy::Server;

#[tokio::test]
async fn post_v1_chat_completions_forwards_to_vendor_and_returns_translated_json() {
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
                "message": { "role": "assistant", "content": "hi back" },
                "finish_reason": "stop"
            }]
        })))
        .expect(1)
        .mount(&mock)
        .await;

    let entry = common::make_openai_entry("deepseek-chat", mock.uri());
    let server = Server::with_entries(vec![entry], "test".to_string());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "deepseek-chat",
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
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    assert_eq!(json.get("id").and_then(|v| v.as_str()), Some("chatcmpl-1"));
    let content = json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str());
    assert_eq!(content, Some("hi back"));
}

#[tokio::test]
async fn post_v1_chat_completions_returns_404_for_unknown_model() {
    let server = Server::with_entries(Vec::new(), "test".to_string());

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "missing",
                        "messages": [{ "role": "user", "content": "hi" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
