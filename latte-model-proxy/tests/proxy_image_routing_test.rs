//! proxy-default 路径上的图片请求路由测试。
//!
//! 覆盖：
//! 1. 图片请求 → 按 supports_vision 过滤后从高优先级挑首个可用模型
//! 2. 纯文本请求 → 不走 vision 过滤，按 pool 原顺序选
//! 3. 池子里没有任何 vision 模型 → 503（明确错误）
//! 4. 显式 model + 图片 body → 原样转发，不重选
//! 5. 所有 vision 模型均被熔断拉出 → 503
//!
//! 验证方式：构造多个 mock upstream，用 wiremock `.expect(N)` 锁定每台
//! mock 被命中的次数，再在响应体里 echo model id 用于断言。

mod common;

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use chrono::TimeZone;
use tower::ServiceExt;

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use latte_model_proxy::{Server, ServerRuntime};
use latte_router::{ModelEntry, Router};

/// 构造一个声明 supports_vision=true 的 OpenAI 模型 entry。
fn openai_vision(id: &str, base_url: String) -> ModelEntry {
    let mut entry = common::make_openai_entry(id, base_url);
    entry.supports_vision = true;
    // 把 429 冷却拉长，方便在测试里制造"被拉出"的状态。
    entry.rate_limit_refresh_anchor = chrono::Utc.timestamp_opt(0, 0).unwrap();
    entry.rate_limit_refresh_interval_secs = 3600;
    entry
}

/// 用给定 pool + catalog 构造一个默认代理运行时。
fn runtime(pool: Vec<String>, entries: Vec<ModelEntry>) -> ServerRuntime {
    ServerRuntime {
        router: Arc::new(Router::with_system_clock(entries)),
        version: "test".to_string(),
        api_key: None,
        proxy_default_model: "proxy-default".to_string(),
        pool,
    }
}

/// 构造一个 chat completions 形状的 200 响应，model 字段直接 echo 传入的 id。
fn chat_200(model_echo: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(serde_json::json!({
        "id": "cmpl-test",
        "object": "chat.completion",
        "model": model_echo,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "ok" },
            "finish_reason": "stop"
        }]
    }))
}

#[tokio::test]
async fn image_request_via_proxy_default_picks_highest_priority_vision_model() {
    // pool 顺序：text-a(text-only) → vision-b(vision) → vision-c(vision)
    // 图片请求必须跳过 text-a，选 vision-b。
    let text_mock = MockServer::start().await;
    let vision_b_mock = MockServer::start().await;
    let vision_c_mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(chat_200("vision-b"))
        .expect(1)
        .mount(&vision_b_mock)
        .await;
    // text-a、vision-c 不应被命中，挂 expect(0) 兜底（wiremock 默认 404，
    // 在断言未命中的同时也让意外打到该 mock 的请求落在一个清晰的失败上）。
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&text_mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&vision_c_mock)
        .await;

    let runtime = runtime(
        vec!["text-a".into(), "vision-b".into(), "vision-c".into()],
        vec![
            common::make_entry("text-a", latte_ai::models::ApiType::OpenAiCompletions, text_mock.uri()),
            openai_vision("vision-b", vision_b_mock.uri()),
            openai_vision("vision-c", vision_c_mock.uri()),
        ],
    );
    let server = Server::new(runtime);
    let app = server.axum_router();

    let body = serde_json::json!({
        "model": "proxy-default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]
        }]
    });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json.get("model").and_then(|v| v.as_str()), Some("vision-b"));
}

#[tokio::test]
async fn text_only_request_via_proxy_default_is_unaffected_by_vision_filter() {
    // 纯文本请求不应触发 supports_vision 过滤，按 pool 原顺序选首个 text-a。
    let text_mock = MockServer::start().await;
    let vision_mock = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(chat_200("text-a"))
        .expect(1)
        .mount(&text_mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&vision_mock)
        .await;

    let runtime = runtime(
        vec!["text-a".into(), "vision-b".into()],
        vec![
            common::make_entry("text-a", latte_ai::models::ApiType::OpenAiCompletions, text_mock.uri()),
            openai_vision("vision-b", vision_mock.uri()),
        ],
    );
    let server = Server::new(runtime);

    let body = serde_json::json!({
        "model": "proxy-default",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json.get("model").and_then(|v| v.as_str()), Some("text-a"));
}

#[tokio::test]
async fn image_request_returns_503_when_no_vision_model_in_pool() {
    // 池子里所有模型都未声明 supports_vision=true → 应立刻 503。
    let text_mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&text_mock)
        .await;

    let runtime = runtime(
        vec!["text-a".into()],
        vec![common::make_entry(
            "text-a",
            latte_ai::models::ApiType::OpenAiCompletions,
            text_mock.uri(),
        )],
    );
    let server = Server::new(runtime);

    let body = serde_json::json!({
        "model": "proxy-default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]
        }]
    });
    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json.get("error").and_then(|v| v.as_str()),
        Some("no vision-capable model available in priority pool")
    );
}

#[tokio::test]
async fn explicit_model_with_image_is_forwarded_as_is() {
    // 显式 `model` 路径不受 supports_vision 过滤影响；含图片的 body
    // 仍原样转发到客户端选定的模型——上游自行决定接受/拒绝。
    let text_mock = MockServer::start().await;
    let vision_mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(chat_200("text-a"))
        .expect(1)
        .mount(&text_mock)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500))
        .expect(0)
        .mount(&vision_mock)
        .await;

    let runtime = runtime(
        vec!["text-a".into(), "vision-b".into()],
        vec![
            common::make_entry("text-a", latte_ai::models::ApiType::OpenAiCompletions, text_mock.uri()),
            openai_vision("vision-b", vision_mock.uri()),
        ],
    );
    let server = Server::new(runtime);

    let body = serde_json::json!({
        "model": "text-a",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]
        }]
    });
    let resp = server
        .axum_router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("response");

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 16384).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // 即便 body 含图片，proxy 也不重选 → 上游收到 text-a 的请求。
    assert_eq!(json.get("model").and_then(|v| v.as_str()), Some("text-a"));
}

#[tokio::test]
async fn image_request_returns_503_when_only_vision_model_is_pulled_out() {
    // 唯一支持 vision 的模型因为 429 被熔断拉出 → 再次发图片请求，
    // filter 后 candidates=[vision-b] 但 breaker 命中 Unavailable，
    // 应返回 503（all upstream models in cooldown）。
    let vision_mock = MockServer::start().await;

    // 第一次命中：上游 429 + Retry-After → 代理 record() 拉出该模型。
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("Retry-After", "3600")
                .set_body_json(serde_json::json!({"error": "rate-limited"})),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(&vision_mock)
        .await;

    let runtime = runtime(
        vec!["vision-b".into()],
        vec![openai_vision("vision-b", vision_mock.uri())],
    );
    let server = Server::new(runtime);
    let app = server.axum_router();

    // 1st request: 触发 429 + 拉出 vision-b。
    let body = serde_json::json!({
        "model": "proxy-default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]
        }]
    });
    let resp1 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("first response");
    assert_eq!(resp1.status(), StatusCode::TOO_MANY_REQUESTS);

    // 2nd request: vision-b 已拉出 → filter 出 [vision-b] 但 select_candidates
    // 全部 Unavailable → 503 all upstream models in cooldown。
    let resp2 = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .expect("second response");
    assert_eq!(resp2.status(), StatusCode::SERVICE_UNAVAILABLE);
    let bytes = to_bytes(resp2.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        json.get("error").and_then(|v| v.as_str()),
        Some("all upstream models in cooldown")
    );
}
