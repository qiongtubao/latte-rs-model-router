//! Tests for health, readiness, meta routes, and body size limit.

mod common;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use latte_model_proxy::Server;

fn server_with(entries: Vec<latte_model_proxy::ModelEntry>) -> Server {
    Server::with_entries(entries, env!("CARGO_PKG_VERSION").to_string())
}

fn empty_server() -> Server {
    server_with(Vec::new())
}

#[tokio::test]
async fn get_root_returns_health_banner() {
    let server = empty_server();
    let app = server.axum_router();

    let resp = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let body = to_bytes(resp.into_body(), 4096).await.unwrap();
    let text = std::str::from_utf8(&body).expect("utf8 body");
    assert!(
        text.contains("latte-model-proxy"),
        "expected banner to mention server name, got: {text:?}"
    );
}

#[tokio::test]
async fn get_health_returns_200() {
    let server = empty_server();
    let app = server.axum_router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), 200);
    let body = to_bytes(resp.into_body(), 1024).await.unwrap();
    let text = std::str::from_utf8(&body).expect("utf8");
    assert!(text.contains("ok"));
}

#[tokio::test]
async fn get_ready_returns_200_when_catalog_has_models() {
    let entries = vec![common::make_openai_entry(
        "stub",
        "http://stub.invalid".to_string(),
    )];
    let server = server_with(entries);
    let app = server.axum_router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn get_ready_returns_503_when_pool_is_empty() {
    let server = empty_server();
    let app = server.axum_router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn get_api_version_returns_version_json() {
    let server = empty_server();
    let app = server.axum_router();

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/api/version")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");

    assert!(resp.status().is_success());
    let body = to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).expect("json body");
    let version = json.get("version").and_then(|v| v.as_str()).expect("version");
    assert_eq!(version, env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn get_root_supports_head_too() {
    let server = empty_server();
    let app = server.axum_router();

    let resp = app
        .oneshot(
            Request::builder()
                .method("HEAD")
                .uri("/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("response");
    assert!(resp.status().is_success());
}

#[tokio::test]
async fn request_body_over_10mb_is_rejected() {
    let server = empty_server();
    let app = server.axum_router();

    // 11 MB body — over the 10 MB cap.
    let big_body = "x".repeat(11 * 1024 * 1024);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(big_body))
                .unwrap(),
        )
        .await
        .expect("response");

    // axum's body limit layer returns 413 Payload Too Large.
    assert_eq!(resp.status(), 413, "expected 413 for over-limit body");
}
