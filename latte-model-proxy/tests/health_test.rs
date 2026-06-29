//! Tests for health + meta routes (no upstream calls required).

mod common;

use axum::body::{to_bytes, Body};
use axum::http::Request;
use tower::ServiceExt;

use latte_model_proxy::Server;

fn empty_server() -> Server {
    Server::with_entries(Vec::new(), env!("CARGO_PKG_VERSION").to_string())
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
