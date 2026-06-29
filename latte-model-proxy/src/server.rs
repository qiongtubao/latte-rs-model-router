//! axum HTTP server entry point.
//!
//! Thin HTTP shell over [`latte_router::Router`]. Two request paths:
//!
//! 1. Client sends `model = "<proxy_default_model>"` (typically `"proxy-default"`):
//!    the proxy walks the configured `pool` in priority order and silently picks
//!    the first available physical model. The caller never learns which was used.
//! 2. Client sends `model = "<real model id>"` (must be in `models.d/`):
//!    the proxy uses that specific model directly.
//!
//! Streaming (`stream: true` in the body) is supported: the proxy pipes the
//! upstream's chunked body to the client without buffering.
//!
//! Operational:
//! - `/health` (always 200) and `/ready` (200 / 503) for k8s probes.
//! - Graceful shutdown on SIGINT / SIGTERM (drain in-flight requests).
//! - `DefaultBodyLimit::max(10 MB)` on all routes to prevent OOM.
//! - Optional proxy-level API key (set in `proxy.toml` `server.api_key`).
//!   When set, `Authorization: Bearer <key>` is required for chat endpoints.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use reqwest::Client as HttpClient;
use serde_json::json;
use tokio::net::TcpListener;
use tracing::{debug, info, warn};

use latte_ai::models::ApiType;
use latte_router::{ModelEntry, Route, Router, RouterError};

/// Hard cap on incoming request body. Chat-completion bodies are typically
/// under 100 KB; 10 MB leaves headroom for large message arrays while
/// preventing OOM from a malicious client.
const MAX_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Routes that require auth when `api_key` is configured. Probes + discovery
/// stay public.
const PROTECTED_PATHS: &[&str] = &[
    "/v1/chat/completions",
    "/v1/messages",
    "/api/chat",
];

/// Bundle of everything the proxy needs to serve requests.
#[derive(Debug, Clone)]
pub struct ServerRuntime {
    pub router: Arc<Router>,
    pub version: String,
    /// Magic name that triggers silent priority selection.
    pub proxy_default_model: String,
    /// Priority pool used when the client sends `proxy_default_model`.
    pub pool: Vec<String>,
    /// Optional API key. When set, `Authorization: Bearer <key>` is required
    /// for `PROTECTED_PATHS`.
    pub api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ServerHandle {
    pub addr: SocketAddr,
}

#[derive(Clone)]
pub struct AppState {
    http: HttpClient,
    router: Arc<Router>,
    version: String,
    proxy_default_model: String,
    pool: Vec<String>,
    api_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Server {
    pub runtime: ServerRuntime,
}

impl Server {
    pub fn new(runtime: ServerRuntime) -> Self {
        Self { runtime }
    }

    /// Test/embed convenience: build a server from just entries + version.
    /// Priority pool is empty; clients using `proxy-default` will get 503.
    /// No API key configured.
    pub fn with_entries(entries: Vec<ModelEntry>, version: String) -> Self {
        Self::new(ServerRuntime {
            router: Arc::new(Router::with_system_clock(entries)),
            version,
            proxy_default_model: "proxy-default".to_string(),
            pool: Vec::new(),
            api_key: None,
        })
    }

    pub fn router(&self) -> &Arc<Router> {
        &self.runtime.router
    }

    pub fn axum_router(&self) -> AxumRouter {
        let state = AppState {
            http: HttpClient::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("reqwest build"),
            router: self.runtime.router.clone(),
            version: self.runtime.version.clone(),
            proxy_default_model: self.runtime.proxy_default_model.clone(),
            pool: self.runtime.pool.clone(),
            api_key: self.runtime.api_key.clone(),
        };
        AxumRouter::new()
            .route("/", get(root_handler).head(root_handler))
            .route("/health", get(health_handler))
            .route("/ready", get(ready_handler))
            .route("/api/version", get(version_handler))
            .route("/api/tags", get(ollama_tags_handler))
            .route("/api/show", post(ollama_show_handler))
            .route("/api/chat", post(ollama_chat_handler))
            .route("/v1/models", get(openai_list_models_handler))
            .route("/v1/chat/completions", post(openai_chat_handler))
            .route("/v1/messages", post(anthropic_messages_handler))
            .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
            .layer(from_fn_with_state(state.clone(), auth_middleware))
            .with_state(state)
    }

    pub async fn serve(&self, listener: TcpListener) -> std::io::Result<()> {
        let app = self.axum_router();
        info!(target: "latte_model_proxy", "server ready (graceful shutdown on SIGINT/SIGTERM)");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
    }
}

/// Liveness probe — always 200 if the process is up.
async fn health_handler() -> Response {
    (StatusCode::OK, "ok\n").into_response()
}

/// Readiness probe — 200 if the catalog has models; 503 otherwise.
async fn ready_handler(State(state): State<AppState>) -> Response {
    if state.router.pool().is_empty() {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "not ready: no models in pool\n",
        )
            .into_response()
    } else {
        (StatusCode::OK, "ready\n").into_response()
    }
}

/// Auth middleware: when `api_key` is configured, requests to protected
/// routes must include `Authorization: Bearer <key>`. Other routes are
/// always public.
async fn auth_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();
    let needs_auth = PROTECTED_PATHS.contains(&path);

    if needs_auth && state.api_key.is_some() {
        let expected = state.api_key.as_deref().unwrap_or("");
        let auth = req
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        let valid = auth
            .and_then(|h| h.strip_prefix("Bearer ").map(str::trim))
            .map(|t| t == expected)
            .unwrap_or(false);
        if !valid {
            warn!(
                target: "latte_model_proxy",
                path = %path,
                has_auth_header = auth.is_some(),
                "auth failed (401)"
            );
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "invalid or missing api key" })),
            )
                .into_response();
        }
    }
    next.run(req).await
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM (k8s shutdown). Used by
/// `axum::serve(...).with_graceful_shutdown(...)` to drain in-flight requests.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install SIGTERM handler")
            .recv()
            .await;
    };
    tokio::select! {
        _ = ctrl_c => info!(target: "latte_model_proxy", "received SIGINT, shutting down"),
        _ = terminate => info!(target: "latte_model_proxy", "received SIGTERM, shutting down"),
    }
}

async fn root_handler(State(state): State<AppState>) -> Response {
    let body = format!(
        "latte-model-proxy {} is running\n  proxy_default_model: {}\n  pool: {}\n  api_key: {}\n",
        state.version,
        state.proxy_default_model,
        state.pool.join(", "),
        if state.api_key.is_some() { "configured" } else { "none" },
    );
    let mut resp = (StatusCode::OK, body).into_response();
    resp.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    resp
}

async fn version_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "version": state.version,
        "proxy_default_model": state.proxy_default_model,
        "pool": state.pool,
    }))
}

async fn openai_list_models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let data: Vec<serde_json::Value> = state
        .router
        .pool()
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "owned_by": m.provider,
                "vendor": m.provider,
                "protocol": api_to_protocol_str(m.api),
                "base_url": m.base_url,
            })
        })
        .collect();
    Json(json!({ "object": "list", "data": data }))
}

async fn openai_chat_handler(State(state): State<AppState>, body: String) -> Response {
    forward_passthrough(
        state,
        &body,
        ApiType::OpenAiCompletions,
        "/chat/completions",
        |req, token| req.header("authorization", format!("Bearer {token}")),
    )
    .await
}

async fn anthropic_messages_handler(State(state): State<AppState>, body: String) -> Response {
    forward_passthrough(
        state,
        &body,
        ApiType::AnthropicMessages,
        "/v1/messages",
        |req, token| {
            req.header("x-api-key", token)
                .header("anthropic-version", "2023-06-01")
        },
    )
    .await
}

async fn ollama_tags_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models: Vec<serde_json::Value> = state
        .router
        .pool()
        .iter()
        .map(|m| {
            json!({
                "name": m.id,
                "model": m.id,
                "modified_at": "1970-01-01T00:00:00Z",
                "size": 0,
                "details": {
                    "family": m.provider,
                    "parameter_size": "",
                },
                "vendor": m.provider,
                "protocol": api_to_protocol_str(m.api),
            })
        })
        .collect();
    Json(json!({ "models": models }))
}

async fn ollama_show_handler(State(state): State<AppState>, body: String) -> Response {
    let parsed: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return json_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}")),
    };
    let name = match parsed.get("name").and_then(|m| m.as_str()) {
        Some(m) => m.to_string(),
        None => return json_error(StatusCode::BAD_REQUEST, "missing 'name' field".into()),
    };
    let entry = match state.router.pool().iter().find(|m| m.id == name) {
        Some(e) => e,
        None => {
            return json_error(
                StatusCode::NOT_FOUND,
                format!("model '{name}' not served"),
            );
        }
    };
    Json(json!({
        "name": entry.id,
        "model": entry.id,
        "modified_at": "1970-01-01T00:00:00Z",
        "size": 0,
        "details": {
            "family": entry.provider,
            "parameter_size": "",
        },
        "vendor": entry.provider,
        "protocol": api_to_protocol_str(entry.api),
    }))
    .into_response()
}

async fn ollama_chat_handler(State(state): State<AppState>, body: String) -> Response {
    let raw = forward_passthrough(
        state,
        &body,
        ApiType::OpenAiCompletions,
        "/chat/completions",
        |req, token| req.header("authorization", format!("Bearer {token}")),
    )
    .await;

    if raw.status() != StatusCode::OK {
        return raw;
    }

    let body_bytes = match axum::body::to_bytes(raw.into_body(), 65536).await {
        Ok(b) => b,
        Err(e) => {
            return json_error(StatusCode::BAD_GATEWAY, format!("body read: {e}"));
        }
    };
    let upstream: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream JSON parse: {e}"),
            );
        }
    };

    let msg_content = upstream
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();
    let model = upstream
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let done_reason = upstream
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .unwrap_or("stop")
        .to_string();

    Json(json!({
        "model": model,
        "created_at": "1970-01-01T00:00:00Z",
        "message": { "role": "assistant", "content": msg_content },
        "done": true,
        "done_reason": done_reason,
    }))
    .into_response()
}

fn json_error(status: StatusCode, msg: String) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}

/// Resolve a model request and forward the body to the upstream.
/// - `model == proxy_default_model` → silent priority selection (Router::select_candidates)
/// - any other `model` → direct lookup (Router::select)
/// - `body.stream == true` → stream the upstream response chunk-by-chunk
async fn forward_passthrough(
    state: AppState,
    body: &str,
    expected_api: ApiType,
    url_suffix: &str,
    auth_header: impl Fn(reqwest::RequestBuilder, &str) -> reqwest::RequestBuilder,
) -> Response {
    static REQ_ID: AtomicU64 = AtomicU64::new(0);
    let request_id = format!("r-{}", REQ_ID.fetch_add(1, Ordering::Relaxed));
    let start = Instant::now();

    debug!(
        target: "latte_model_proxy",
        request_id = %request_id,
        expected_api = ?expected_api,
        body_bytes = body.len(),
        "request received"
    );

    let parsed: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(e) => {
            warn!(
                target: "latte_model_proxy",
                request_id = %request_id,
                error = %e,
                "body parse failed"
            );
            return json_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}"));
        }
    };

    let model = match parsed.get("model").and_then(|m| m.as_str()) {
        Some(m) => m.to_string(),
        None => {
            warn!(
                target: "latte_model_proxy",
                request_id = %request_id,
                "missing model field in body"
            );
            return json_error(StatusCode::BAD_REQUEST, "missing 'model' field".into());
        }
    };

    let route: Route = if model == state.proxy_default_model {
        debug!(
            target: "latte_model_proxy",
            request_id = %request_id,
            pool_size = state.pool.len(),
            "silent selection (proxy-default)"
        );
        match state.router.select_candidates(&state.pool) {
            Ok(r) => {
                info!(
                    target: "latte_model_proxy",
                    request_id = %request_id,
                    selected_model = %r.model_id,
                    "silent selection picked"
                );
                r
            }
            Err(RouterError::AllUnavailable { retry_after_secs }) => {
                warn!(
                    target: "latte_model_proxy",
                    request_id = %request_id,
                    retry_after_secs = retry_after_secs,
                    "silent selection: all candidates in cooldown (503)"
                );
                let mut resp = json_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "all upstream models in cooldown".into(),
                );
                if let Ok(v) = HeaderValue::from_str(&retry_after_secs.to_string()) {
                    resp.headers_mut().insert(header::RETRY_AFTER, v);
                }
                return resp;
            }
            Err(e) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("router: {e}"),
                );
            }
        }
    } else {
        match state.router.select(&model) {
            Ok(r) => r,
            Err(RouterError::UnknownModel(_)) => {
                warn!(
                    target: "latte_model_proxy",
                    request_id = %request_id,
                    requested = %model,
                    "model not in pool (404)"
                );
                return json_error(
                    StatusCode::NOT_FOUND,
                    format!("model '{model}' not served"),
                );
            }
            Err(RouterError::AllUnavailable { retry_after_secs }) => {
                warn!(
                    target: "latte_model_proxy",
                    request_id = %request_id,
                    requested = %model,
                    retry_after_secs = retry_after_secs,
                    "all upstream in cooldown (503)"
                );
                let mut resp = json_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "all upstream models in cooldown".into(),
                );
                if let Ok(v) = HeaderValue::from_str(&retry_after_secs.to_string()) {
                    resp.headers_mut().insert(header::RETRY_AFTER, v);
                }
                return resp;
            }
            Err(e) => {
                return json_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("router: {e}"),
                );
            }
        }
    };

    if route.api != expected_api {
        warn!(
            target: "latte_model_proxy",
            request_id = %request_id,
            requested = %model,
            selected_model = %route.model_id,
            expected = ?expected_api,
            actual = ?route.api,
            "api shape mismatch (400)"
        );
        return json_error(
            StatusCode::BAD_REQUEST,
            format!(
                "model '{model}' is served via {:?}, not the requested API shape",
                route.api
            ),
        );
    }

    let mut body_to_send = parsed;
    if model == state.proxy_default_model {
        if let Some(obj) = body_to_send.as_object_mut() {
            obj.insert("model".to_string(), json!(route.model_id.clone()));
        }
    }

    let stream_flag = body_to_send
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let upstream_url = format!(
        "{}/{}",
        route.base_url.trim_end_matches('/'),
        url_suffix.trim_start_matches('/')
    );

    info!(
        target: "latte_model_proxy",
        request_id = %request_id,
        requested = %model,
        selected_model = %route.model_id,
        upstream_url = %upstream_url,
        stream = stream_flag,
        body_model_replaced = (model == state.proxy_default_model),
        "forwarding to upstream"
    );

    let req = state
        .http
        .post(&upstream_url)
        .header("content-type", "application/json");
    let req = auth_header(req, &route.api_key);

    let resp = match req.json(&body_to_send).send().await {
        Ok(r) => r,
        Err(e) => {
            warn!(
                target: "latte_model_proxy",
                request_id = %request_id,
                selected_model = %route.model_id,
                upstream_url = %upstream_url,
                error = %e,
                "upstream network error"
            );
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream network error: {e}"),
            );
        }
    };

    let status = resp.status();
    let retry_after_secs = parse_retry_after_header(resp.headers());
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    if stream_flag {
        let response_status = status.as_u16();
        state
            .router
            .record(&route.model_id, response_status, retry_after_secs);
        info!(
            target: "latte_model_proxy",
            request_id = %request_id,
            selected_model = %route.model_id,
            status = response_status,
            content_type = %ctype,
            "streaming response started"
        );
        let mut out = Response::new(Body::from_stream(resp.bytes_stream()));
        *out.status_mut() =
            StatusCode::from_u16(response_status).unwrap_or(StatusCode::BAD_GATEWAY);
        if let Ok(ct) = HeaderValue::from_str(&ctype) {
            out.headers_mut().insert(header::CONTENT_TYPE, ct);
        }
        if let Some(ra) = retry_after_secs {
            if let Ok(v) = HeaderValue::from_str(&ra.to_string()) {
                out.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        return out;
    }

    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            warn!(
                target: "latte_model_proxy",
                request_id = %request_id,
                selected_model = %route.model_id,
                error = %e,
                "upstream body read failed"
            );
            return json_error(StatusCode::BAD_GATEWAY, format!("upstream body read: {e}"));
        }
    };

    let duration_ms = start.elapsed().as_millis() as u64;
    let response_status = status.as_u16();
    state
        .router
        .record(&route.model_id, response_status, retry_after_secs);

    if status.is_success() {
        info!(
            target: "latte_model_proxy",
            request_id = %request_id,
            selected_model = %route.model_id,
            status = response_status,
            duration_ms = duration_ms,
            body_bytes = bytes.len(),
            "upstream response OK"
        );
    } else {
        warn!(
            target: "latte_model_proxy",
            request_id = %request_id,
            selected_model = %route.model_id,
            status = response_status,
            duration_ms = duration_ms,
            body_bytes = bytes.len(),
            retry_after_header = retry_after_secs.unwrap_or(0),
            "upstream non-2xx"
        );
    }

    let mut out = Response::new(Body::from(bytes));
    *out.status_mut() = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if let Ok(ct) = HeaderValue::from_str(&ctype) {
        out.headers_mut().insert(header::CONTENT_TYPE, ct);
    }
    out
}

fn parse_retry_after_header(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let h = headers.get(header::RETRY_AFTER)?.to_str().ok()?;
    h.parse::<u64>().ok()
}

fn api_to_protocol_str(api: ApiType) -> &'static str {
    match api {
        ApiType::OpenAiCompletions => "openai",
        ApiType::AnthropicMessages => "anthropic",
    }
}

pub async fn serve(
    runtime: ServerRuntime,
    bind_addr: String,
) -> std::io::Result<ServerHandle> {
    let listener = TcpListener::bind(&bind_addr).await?;
    let addr = listener.local_addr()?;
    let server = Server::new(runtime);
    tokio::spawn(async move {
        let _ = server.serve(listener).await;
    });
    Ok(ServerHandle { addr })
}
