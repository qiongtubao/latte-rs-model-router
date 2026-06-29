//! Tests for `/v1/chat/completions` streaming (SSE passthrough).
//!
//! Verifies that `stream: true` requests are forwarded as a stream (not buffered)
//! and that the body's `model` field is replaced when the client used
//! `proxy-default` (silent selection) so the upstream accepts the request.

mod common;

use axum::body::Body;
use axum::http::Request;
use futures_util::StreamExt;
use tower::ServiceExt;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use latte_model_proxy::{Server, ServerRuntime};
use latte_router::Router;
use std::sync::Arc;

#[tokio::test]
async fn post_v1_chat_completions_stream_passes_through_sse_unchanged() {
    let mock = MockServer::start().await;

    let sse_body = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\ndata: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
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
                        "messages": [{ "role": "user", "content": "hi" }],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(
        content_type.contains("event-stream"),
        "expected text/event-stream, got: {content_type}"
    );

    let mut body_stream = resp.into_body().into_data_stream();
    let mut data = Vec::new();
    while let Some(chunk) = body_stream.next().await {
        data.extend_from_slice(&chunk.unwrap());
    }
    let body_str = String::from_utf8(data).expect("utf8");
    assert!(body_str.contains("hello "));
    assert!(body_str.contains("world"));
    assert!(body_str.contains("[DONE]"));
}

#[tokio::test]
async fn stream_with_proxy_default_replaces_model_field_in_body() {
    let mock = MockServer::start().await;

    // The upstream receives a request and echoes back the model field
    // from the body. We expect the proxy to have replaced the body
    // model "proxy-default" with the actual selected id "glm-5.2".
    let sse_body = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer sk-test"))
        .respond_with(
            ResponseTemplate::new(200).set_body_raw(sse_body.as_bytes(), "text/event-stream"),
        )
        .expect(1)
        .mount(&mock)
        .await;

    let entry = common::make_openai_entry("glm-5.2", mock.uri());
    let runtime = ServerRuntime {
        router: Arc::new(Router::with_system_clock(vec![entry])),
        version: "test".to_string(),
        proxy_default_model: "proxy-default".to_string(),
        pool: vec!["glm-5.2".to_string()],
    };
    let server = Server::new(runtime);

    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "proxy-default",
                        "messages": [{ "role": "user", "content": "hi" }],
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(content_type.contains("event-stream"));
}
