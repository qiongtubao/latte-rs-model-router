//! Tests for the optional proxy-level API key auth.
//!
//! When `proxy.toml` has `server.api_key = "..."` (or CLI `--api-key=...`),
//! the proxy requires `Authorization: Bearer <key>` on protected routes.
//! Probes + discovery are always public.

mod common;

use axum::body::Body;
use axum::http::Request;
use latte_model_proxy::{Server, ServerRuntime};
use latte_router::Router;
use std::sync::Arc;
use tower::ServiceExt;

fn runtime_with_api_key(api_key: Option<String>) -> ServerRuntime {
    let entry = common::make_openai_entry("stub", "http://stub.invalid".to_string());
    ServerRuntime {
        router: Arc::new(Router::with_system_clock(vec![entry])),
        version: "test".to_string(),
        api_key,
        proxy_default_model: "proxy-default".to_string(),
        pool: vec!["stub".to_string()],
    }
}

fn server(api_key: Option<String>) -> Server {
    Server::new(runtime_with_api_key(api_key))
}

/// Build a fresh router per request to avoid the moved-value issue
/// (axum::Router::oneshot consumes the router).
async fn run(app: &axum::Router, req: Request<Body>) -> axum::response::Response {
    app.clone().oneshot(req).await.expect("response")
}

#[tokio::test]
async fn no_api_key_configured_means_no_auth_required() {
    let server = server(None);

    let resp = run(
        &server.axum_router(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap(),
    )
    .await;
    assert_ne!(
        resp.status(),
        401,
        "no auth configured → should not be 401"
    );
}

#[tokio::test]
async fn api_key_configured_and_no_header_returns_401() {
    let server = server(Some("secret-key".to_string()));

    let resp = run(
        &server.axum_router(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn api_key_configured_and_wrong_key_returns_401() {
    let server = server(Some("secret-key".to_string()));

    let resp = run(
        &server.axum_router(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer wrong-key")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn api_key_configured_and_correct_key_passes_auth() {
    let server = server(Some("secret-key".to_string()));

    let resp = run(
        &server.axum_router(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer secret-key")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap(),
    )
    .await;
    assert_ne!(
        resp.status(),
        401,
        "correct key should pass auth check (may be 502/404 for stub upstream, not 401)"
    );
}

#[tokio::test]
async fn api_key_does_not_apply_to_health_or_ready_or_models() {
    let server = server(Some("secret-key".to_string()));

    // /health — no auth header
    let resp = run(
        &server.axum_router(),
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(resp.status(), 200);

    // /ready — no auth header
    let resp = run(
        &server.axum_router(),
        Request::builder()
            .uri("/ready")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(resp.status(), 200);

    // /v1/models — no auth header (discovery is public)
    let resp = run(
        &server.axum_router(),
        Request::builder()
            .uri("/v1/models")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn auth_header_without_bearer_prefix_returns_401() {
    let server = server(Some("secret-key".to_string()));

    let resp = run(
        &server.axum_router(),
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "secret-key") // no "Bearer " prefix
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#,
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(resp.status(), 401);
}
