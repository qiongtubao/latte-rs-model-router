//! `latte_ai::vendor` 集成测试（用 wiremock 模拟 vendor API）
//!
//! 覆盖：
//! - [`AnthropicModelsApi::discover`] 真实 HTTP 流程
//! - [`OpenAiModelsApi::discover`] 真实 HTTP 流程
//! - 401 / 404 错误路径
//! - 响应数据解析
//! - Bearer token 正确发送
//! - api_key 正确发送

use std::sync::Arc;
use std::time::Duration;

use latte_ai::vendor::bearer::FixedIntervalRefresher;
use latte_ai::vendor::discover::{AnthropicModelsApi, OpenAiModelsApi};
use wiremock::matchers::{bearer_token, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use latte_ai::vendor::{
    ApiKeyProvider, BearerProvider, HealthCheck, ModelDiscovery, VendorConfig, VendorId,
    VendorRegistry,
};

// ─── AnthropicModelsApi ──────────────────────────────────────

#[tokio::test]
async fn anthropic_discover_parses_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"id": "claude-sonnet-4-20250514", "display_name": "Claude Sonnet 4", "type": "model"},
                {"id": "claude-opus-4-20250514",  "display_name": "Claude Opus 4",   "type": "model"}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let api = AnthropicModelsApi;
    let models = api
        .discover(&server.uri(), "test-key")
        .await
        .expect("discover should succeed");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "claude-sonnet-4-20250514");
    assert_eq!(models[0].display_name, "Claude Sonnet 4");
    assert_eq!(models[0].vendor_specific.get("type").map(|s| s.as_str()), Some("model"));
    assert_eq!(models[1].id, "claude-opus-4-20250514");
}

#[tokio::test]
async fn anthropic_discover_uses_display_name_fallback() {
    // 如果 model 没有 display_name 字段，应该 fallback 到 id
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "claude-fallback"}]
        })))
        .mount(&server)
        .await;

    let api = AnthropicModelsApi;
    let models = api.discover(&server.uri(), "k").await.unwrap();
    assert_eq!(models[0].id, "claude-fallback");
    assert_eq!(models[0].display_name, "claude-fallback"); // fallback 到 id
}

#[tokio::test]
async fn anthropic_discover_handles_401() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;

    let api = AnthropicModelsApi;
    let result = api.discover(&server.uri(), "bad-key").await;
    assert!(matches!(result, Err(latte_ai::vendor::VendorError::Network(_))));
}

#[tokio::test]
async fn anthropic_discover_handles_invalid_json() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
        .mount(&server)
        .await;

    let api = AnthropicModelsApi;
    let result = api.discover(&server.uri(), "k").await;
    assert!(matches!(result, Err(latte_ai::vendor::VendorError::RefreshFailed(_))));
}

// ─── OpenAiModelsApi ─────────────────────────────────────────

#[tokio::test]
async fn openai_discover_parses_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(bearer_token("test-token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"id": "gpt-4o", "owned_by": "openai"},
                {"id": "gpt-4o-mini", "owned_by": "openai"}
            ]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let api = OpenAiModelsApi;
    let models = api
        .discover(&server.uri(), "test-token")
        .await
        .expect("discover should succeed");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "gpt-4o");
    // OpenAI 不给 display_name → fallback 到 id
    assert_eq!(models[0].display_name, "gpt-4o");
    assert_eq!(
        models[0].vendor_specific.get("owned_by").map(|s| s.as_str()),
        Some("openai")
    );
    assert_eq!(models[1].id, "gpt-4o-mini");
}

#[tokio::test]
async fn openai_discover_handles_404() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&server)
        .await;

    let api = OpenAiModelsApi;
    let result = api.discover(&server.uri(), "k").await;
    assert!(matches!(result, Err(latte_ai::vendor::VendorError::Network(_))));
}

// ─── VendorRegistry end-to-end ──────────────────────────────

#[tokio::test]
async fn registry_discover_models_uses_authorization_header() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .and(bearer_token("secret-bearer-xyz"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [{"id": "model-a"}, {"id": "model-b"}]
        })))
        .expect(1)
        .mount(&server)
        .await;

    let auth: Arc<dyn latte_ai::vendor::TokenProvider> = Arc::new(BearerProvider::new(
        "test",
        "secret-bearer-xyz",
        Duration::from_secs(3600),
        FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("secret-bearer-xyz".into()) }),
        ),
    ));
    let discovery: Arc<dyn latte_ai::vendor::ModelDiscovery> = Arc::new(OpenAiModelsApi);
    let v = VendorConfig::new("test", &server.uri(), auth, discovery);
    let reg = VendorRegistry::new(vec![v]);

    let models = reg.discover_models(&VendorId::new("test")).await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "model-a");
    assert_eq!(models[1].id, "model-b");
}

#[tokio::test]
async fn registry_status_pings_health_check_endpoint() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .and(header("x-api-key", "k"))
        .respond_with(ResponseTemplate::new(200).set_body_string("ok"))
        .expect(1)
        .mount(&server)
        .await;

    let auth: Arc<dyn latte_ai::vendor::TokenProvider> =
        Arc::new(ApiKeyProvider::new("anthropic", "k"));
    let discovery: Arc<dyn latte_ai::vendor::ModelDiscovery> = Arc::new(AnthropicModelsApi);
    let v = VendorConfig::new("anthropic", &server.uri(), auth, discovery)
        .with_health_check(HealthCheck::Http { path: "/health".into() });
    let reg = VendorRegistry::new(vec![v]);

    let s = reg.status(&VendorId::new("anthropic")).await.unwrap();
    assert!(s.auth_valid);
    assert!(s.health_ok);
    assert!(s.health_latency_ms.is_some());
    // 静态 api_key → remaining = None
    assert_eq!(s.token_remaining_secs, None);
}

#[tokio::test]
async fn registry_status_reports_failed_health_check() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(503))
        .mount(&server)
        .await;

    let auth: Arc<dyn latte_ai::vendor::TokenProvider> =
        Arc::new(ApiKeyProvider::new("anthropic", "k"));
    let discovery: Arc<dyn latte_ai::vendor::ModelDiscovery> = Arc::new(AnthropicModelsApi);
    let v = VendorConfig::new("anthropic", &server.uri(), auth, discovery)
        .with_health_check(HealthCheck::Http { path: "/health".into() });
    let reg = VendorRegistry::new(vec![v]);

    let s = reg.status(&VendorId::new("anthropic")).await.unwrap();
    assert!(!s.health_ok);
    assert!(s.health_latency_ms.is_some());
}

#[tokio::test]
async fn registry_get_token_uses_api_key_provider() {
    let auth: Arc<dyn latte_ai::vendor::TokenProvider> = Arc::new(ApiKeyProvider::new(
        "test",
        "test-static-key",
    ));
    let discovery: Arc<dyn latte_ai::vendor::ModelDiscovery> = Arc::new(OpenAiModelsApi);
    let v = VendorConfig::new("test", "https://api.x", auth, discovery);
    let reg = VendorRegistry::new(vec![v]);

    let token = reg.get_token(&VendorId::new("test")).await.unwrap();
    assert_eq!(token, "test-static-key");
}

#[tokio::test]
async fn registry_refresh_now_returns_new_token() {
    let auth: Arc<dyn latte_ai::vendor::TokenProvider> = Arc::new(BearerProvider::new(
        "test",
        "initial",
        Duration::from_secs(3600),
        FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("refreshed-token".into()) }),
        ),
    ));
    let discovery: Arc<dyn latte_ai::vendor::ModelDiscovery> = Arc::new(OpenAiModelsApi);
    let v = VendorConfig::new("test", "https://api.x", auth, discovery);
    let reg = VendorRegistry::new(vec![v]);

    let token = reg.refresh_now(&VendorId::new("test")).await.unwrap();
    assert_eq!(token, "refreshed-token");
}
