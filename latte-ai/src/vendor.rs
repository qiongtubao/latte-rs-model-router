//! 厂商抽象：VendorId + TokenProvider + ModelDiscovery
//!
//! # Session 2 范围
//!
//! - `VendorId` newtype
//! - `VendorError` 错误类型
//! - [`TokenProvider`] trait + [`ApiKeyProvider`] / [`BearerProvider`] impl
//! - [`ModelDescriptor`] 轻量结构
//! - [`ModelDiscovery`] trait + [`discover::AnthropicModelsApi`] / [`discover::OpenAiModelsApi`] impl
//!
//! # 后续 sessions（不在本 session 范围）
//!
//! - Session 3: VendorRegistry + config 解析（V1/V2 兼容）
//! - Session 4: 单测 + examples
//!
//! # 用法
//!
//! ```no_run
//! use latte_ai::vendor::{ApiKeyProvider, TokenProvider};
//!
//! # async fn example() -> Result<(), latte_ai::vendor::VendorError> {
//! // 静态 API key
//! let api = ApiKeyProvider::new("anthropic", "${ANTHROPIC_API_KEY}");
//! let token = api.token().await?;
//!
//! // Bearer + 固定间隔刷新
//! use latte_ai::vendor::bearer::FixedIntervalRefresher;
//! let refresher = FixedIntervalRefresher::new(
//!     std::time::Duration::from_secs(3600),
//!     || Box::pin(async { Ok("refreshed-token".into()) }),
//! );
//! let bearer = BearerProvider::new("anthropic", "initial-token", refresher);
//! let token = bearer.token().await?;
//!
//! // Discover models
//! use latte_ai::vendor::{ModelDiscovery, discover::AnthropicModelsApi};
//! let discover = AnthropicModelsApi;
//! let models = discover.discover("https://api.anthropic.com", &token).await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Vendor 错误
#[derive(Debug, thiserror::Error)]
pub enum VendorError {
    #[error("token refresh failed: {0}")]
    RefreshFailed(String),
    #[error("network error: {0}")]
    Network(String),
    #[error("env var not set: {0}")]
    EnvMissing(String),
    #[error("not implemented: {0}")]
    NotImplemented(String),
}

pub type VendorResult<T> = Result<T, VendorError>;

/// Vendor ID newtype（防止字符串拼写错误）
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VendorId(pub String);

impl VendorId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for VendorId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl std::fmt::Display for VendorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// 解析 `${ENV_VAR}` 形式的字符串；非变量引用原样返回
fn resolve_env(s: &str) -> String {
    if s.len() >= 4 && s.starts_with("${") && s.ends_with('}') {
        let var = &s[2..s.len() - 1];
        std::env::var(var).unwrap_or_else(|_| s.to_string())
    } else {
        s.to_string()
    }
}

// ─── TokenProvider trait ─────────────────────────────────────

/// Token 抽象接口
///
/// 实现要求：
/// - `token()` 永远不 panic：内部错误包成 `VendorError`
/// - `token()` 内部可能做异步 I/O（OAuth refresh、secret manager 读等）
/// - `is_expired()` 是无锁 / cheap 的（用于 health check UI）
#[async_trait]
pub trait TokenProvider: Send + Sync {
    /// Vendor ID（如 `"anthropic"` / `"deepseek"`）
    fn vendor_id(&self) -> &VendorId;

    /// 拿当前有效 token；过期时自动 refresh
    async fn token(&self) -> VendorResult<String>;

    /// 强制立即 refresh
    async fn refresh_now(&self) -> VendorResult<String>;

    /// Token 是否已过期（无锁读）
    fn is_expired(&self) -> bool;

    /// Token 剩余有效时间（无锁读）
    fn remaining(&self) -> Duration;

    /// Token 类型描述（`"api_key"` / `"bearer"`）
    fn kind(&self) -> &'static str;
}

// ─── ApiKeyProvider（无刷新） ───────────────────────────────

/// 静态 API key provider
///
/// - key 写在配置里，env 替换（`${ENV_VAR}`）
/// - 不过期，不 refresh
#[derive(Debug, Clone)]
pub struct ApiKeyProvider {
    vendor_id: VendorId,
    key: String,
}

impl ApiKeyProvider {
    pub fn new(vendor: impl Into<VendorId>, key: impl Into<String>) -> Self {
        Self {
            vendor_id: vendor.into(),
            key: key.into(),
        }
    }
}

#[async_trait]
impl TokenProvider for ApiKeyProvider {
    fn vendor_id(&self) -> &VendorId {
        &self.vendor_id
    }
    async fn token(&self) -> VendorResult<String> {
        Ok(resolve_env(&self.key))
    }
    async fn refresh_now(&self) -> VendorResult<String> {
        Ok(resolve_env(&self.key))
    }
    fn is_expired(&self) -> bool {
        // 静态 key 不过期
        false
    }
    fn remaining(&self) -> Duration {
        // 静态 key 无限期有效
        Duration::from_secs(u64::MAX / 2)
    }
    fn kind(&self) -> &'static str {
        "api_key"
    }
}

// ─── BearerProvider（固定间隔自动刷新） ───────────────────

/// 内部 state（`RwLock` 允许 refresh 时更新）
#[derive(Debug)]
struct BearerState {
    token: String,
    started_at: Instant,
}

/// Bearer token provider + 固定间隔自动刷新
///
/// 每次 `token()` 检查 `now - started_at > interval`：
/// - 未过期：返回当前 token
/// - 过期：调 `refresher.refresh()`，更新 token + 重置 `started_at`
pub struct BearerProvider {
    vendor_id: VendorId,
    state: Arc<RwLock<BearerState>>,
    interval: Duration,
    refresher: Arc<dyn BearerRefresher>,
}

impl std::fmt::Debug for BearerProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.read();
        let preview: String = state.token.chars().take(8).collect();
        f.debug_struct("BearerProvider")
            .field("vendor", &self.vendor_id)
            .field("token", &format!("{preview}..."))
            .field("age", &state.started_at.elapsed())
            .field("interval", &self.interval)
            .finish()
    }
}

/// Bearer 刷新回调：用户实现这个 trait 来定义如何拿新 token
/// （OAuth client_credentials、secret manager 读、文件读等）
#[async_trait]
pub trait BearerRefresher: Send + Sync {
    async fn refresh(&self) -> VendorResult<String>;
}

impl BearerProvider {
    /// 创建：起始时间 = now
    pub fn new<R: BearerRefresher + 'static>(
        vendor: impl Into<VendorId>,
        initial_token: impl Into<String>,
        interval: Duration,
        refresher: R,
    ) -> Self {
        Self {
            vendor_id: vendor.into(),
            state: Arc::new(RwLock::new(BearerState {
                token: initial_token.into(),
                started_at: Instant::now(),
            })),
            interval,
            refresher: Arc::new(refresher),
        }
    }

    /// 自定义起始时间（用于从外部注入"token 实际签发时间"）
    pub fn with_started_at<R: BearerRefresher + 'static>(
        vendor: impl Into<VendorId>,
        initial_token: impl Into<String>,
        started_at: Instant,
        interval: Duration,
        refresher: R,
    ) -> Self {
        Self {
            vendor_id: vendor.into(),
            state: Arc::new(RwLock::new(BearerState {
                token: initial_token.into(),
                started_at,
            })),
            interval,
            refresher: Arc::new(refresher),
        }
    }

    /// 强制立即 refresh
    pub async fn refresh_now(&self) -> VendorResult<String> {
        let new_token = self.refresher.refresh().await?;
        let mut state = self.state.write();
        state.token = new_token.clone();
        state.started_at = Instant::now();
        Ok(new_token)
    }
}

#[async_trait]
impl TokenProvider for BearerProvider {
    fn vendor_id(&self) -> &VendorId {
        &self.vendor_id
    }

    async fn token(&self) -> VendorResult<String> {
        let expired = {
            let state = self.state.read();
            state.started_at.elapsed() > self.interval
        };
        if expired {
            self.refresh_now().await
        } else {
            Ok(self.state.read().token.clone())
        }
    }

    async fn refresh_now(&self) -> VendorResult<String> {
        // 直接调自己的方法（避免 async 递归）
        BearerProvider::refresh_now(self).await
    }

    fn is_expired(&self) -> bool {
        let state = self.state.read();
        state.started_at.elapsed() > self.interval
    }

    fn remaining(&self) -> Duration {
        let state = self.state.read();
        let age = state.started_at.elapsed();
        self.interval.saturating_sub(age)
    }

    fn kind(&self) -> &'static str {
        "bearer"
    }
}

// ─── Bearer 辅助实现 ──────────────────────────────────────────

/// Bearer 刷新相关的辅助工具
pub mod bearer {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;

    /// 固定间隔刷新：起 + 间隔；过期时调 `refresh_fn`
    pub struct FixedIntervalRefresher {
        started_at: Instant,
        interval: Duration,
        refresh_fn:
            Arc<dyn Fn() -> Pin<Box<dyn Future<Output = VendorResult<String>> + Send>> + Send + Sync>,
    }

    impl std::fmt::Debug for FixedIntervalRefresher {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("FixedIntervalRefresher")
                .field("started_at_age", &self.started_at.elapsed())
                .field("interval", &self.interval)
                .finish()
        }
    }

    impl FixedIntervalRefresher {
        /// 创建：起始 = now
        pub fn new<F, Fut>(interval: Duration, refresh_fn: F) -> Self
        where
            F: Fn() -> Fut + Send + Sync + 'static,
            Fut: Future<Output = VendorResult<String>> + Send + 'static,
        {
            Self {
                started_at: Instant::now(),
                interval,
                refresh_fn: Arc::new(move || Box::pin(refresh_fn())),
            }
        }

        /// 自定义起始时间（用于测试）
        pub fn with_started_at<F, Fut>(
            started_at: Instant,
            interval: Duration,
            refresh_fn: F,
        ) -> Self
        where
            F: Fn() -> Fut + Send + Sync + 'static,
            Fut: Future<Output = VendorResult<String>> + Send + 'static,
        {
            Self {
                started_at,
                interval,
                refresh_fn: Arc::new(move || Box::pin(refresh_fn())),
            }
        }
    }

    #[async_trait]
    impl BearerRefresher for FixedIntervalRefresher {
        async fn refresh(&self) -> VendorResult<String> {
            (self.refresh_fn)().await
        }
    }
}

// ─── ModelDescriptor（discover 结果） ────────────────────────

/// 轻量模型描述符（discover API 返回的）
///
/// 与 `Model` 区别：不含 pricing / context_window 默认值等（vendor API 通常不返回这些）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelDescriptor {
    /// Model ID（如 `"claude-sonnet-4-20250514"` / `"gpt-4o"`）
    pub id: String,
    /// 人类可读名（如 `"Claude Sonnet 4"` / `"GPT-4o"`）
    pub display_name: String,
    /// Context window（vendor API 给的话就有）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Vendor 特定元数据（Anthropic `display_name`、OpenAI `owned_by` 等）
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub vendor_specific: HashMap<String, String>,
}

impl ModelDescriptor {
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            display_name: display_name.into(),
            context_window: None,
            vendor_specific: HashMap::new(),
        }
    }
}

/// 模型发现策略（用户可实现 trait 接入自建 vendor）
#[async_trait]
pub trait ModelDiscovery: Send + Sync {
    /// Vendor 协议名（用于 metrics / 调试）
    fn protocol(&self) -> &'static str;

    /// 从 vendor API 拉所有可用 model
    ///
    /// `base_url`: vendor 的 API base URL（无尾斜杠）
    /// `auth_token`: 通过 `TokenProvider::token()` 拿到的当前有效 token
    async fn discover(
        &self,
        base_url: &str,
        auth_token: &str,
    ) -> VendorResult<Vec<ModelDescriptor>>;
}

// ─── 内置 discover impls ────────────────────────────────────

/// Model 发现策略（Anthropic + OpenAI 等内置）
pub mod discover {
    use super::*;

    /// Anthropic models API: `GET /v1/models`，返 `{data: [{id, display_name, ...}]}`
    pub struct AnthropicModelsApi;

    #[async_trait]
    impl ModelDiscovery for AnthropicModelsApi {
        fn protocol(&self) -> &'static str {
            "anthropic"
        }

        async fn discover(
            &self,
            base_url: &str,
            auth_token: &str,
        ) -> VendorResult<Vec<ModelDescriptor>> {
            let url = format!("{}/v1/models?limit=100", base_url.trim_end_matches('/'));
            let resp = reqwest::Client::new()
                .get(&url)
                .header("x-api-key", auth_token)
                .header("anthropic-version", "2023-06-01")
                .send()
                .await
                .map_err(|e| VendorError::Network(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                return Err(VendorError::Network(format!(
                    "anthropic /v1/models returned {status}"
                )));
            }
            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| VendorError::RefreshFailed(format!("json parse: {e}")))?;
            let data = body
                .get("data")
                .and_then(|v| v.as_array())
                .ok_or_else(|| VendorError::RefreshFailed("missing 'data' field".into()))?;
            data.iter()
                .map(|m| {
                    let id = m
                        .get("id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| VendorError::RefreshFailed("missing 'id'".into()))?
                        .to_string();
                    let display_name = m
                        .get("display_name")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&id)
                        .to_string();
                    let mut specific = HashMap::new();
                    if let Some(typ) = m.get("type").and_then(|v| v.as_str()) {
                        specific.insert("type".to_string(), typ.to_string());
                    }
                    Ok(ModelDescriptor {
                        id,
                        display_name,
                        context_window: None, // Anthropic API 不返回
                        vendor_specific: specific,
                    })
                })
                .collect()
        }
    }

    /// OpenAI models API: `GET /v1/models`，返 `{data: [{id, owned_by, ...}]}`
    pub struct OpenAiModelsApi;

    #[async_trait]
    impl ModelDiscovery for OpenAiModelsApi {
        fn protocol(&self) -> &'static str {
            "openai"
        }

        async fn discover(
            &self,
            base_url: &str,
            auth_token: &str,
        ) -> VendorResult<Vec<ModelDescriptor>> {
            let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
            let resp = reqwest::Client::new()
                .get(&url)
                .bearer_auth(auth_token)
                .send()
                .await
                .map_err(|e| VendorError::Network(e.to_string()))?;
            let status = resp.status();
            if !status.is_success() {
                return Err(VendorError::Network(format!(
                    "openai /v1/models returned {status}"
                )));
            }
            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| VendorError::RefreshFailed(format!("json parse: {e}")))?;
            let data = body
                .get("data")
                .and_then(|v| v.as_array())
                .ok_or_else(|| VendorError::RefreshFailed("missing 'data' field".into()))?;
            data.iter()
                .map(|m| {
                    let id = m
                        .get("id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| VendorError::RefreshFailed("missing 'id'".into()))?
                        .to_string();
                    let owned_by = m
                        .get("owned_by")
                        .and_then(|v| v.as_str())
                        .unwrap_or("openai")
                        .to_string();
                    let mut specific = HashMap::new();
                    specific.insert("owned_by".to_string(), owned_by);
                    Ok(ModelDescriptor {
                        id: id.clone(),
                        display_name: id, // OpenAI 不给 display_name，用 id
                        context_window: None, // OpenAI API 不返回
                        vendor_specific: specific,
                    })
                })
                .collect()
        }
    }

    /// 静态列表（用户手填 / 兜底）
    pub struct Manual(pub Vec<ModelDescriptor>);

    #[async_trait]
    impl ModelDiscovery for Manual {
        fn protocol(&self) -> &'static str {
            "manual"
        }

        async fn discover(
            &self,
            _base_url: &str,
            _auth_token: &str,
        ) -> VendorResult<Vec<ModelDescriptor>> {
            Ok(self.0.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::bearer::FixedIntervalRefresher;
    use super::discover::{AnthropicModelsApi, Manual as ManualDiscover, OpenAiModelsApi};
    use std::time::Duration;

    // ── VendorId ─────────────────────────────────────────────

    #[test]
    fn vendor_id_serde_uses_string_value() {
        let v: VendorId = serde_json::from_str("\"anthropic\"").unwrap();
        assert_eq!(v.0, "anthropic");
        let s = serde_json::to_string(&v).unwrap();
        assert_eq!(s, "\"anthropic\"");
    }

    #[test]
    fn vendor_id_display() {
        let v = VendorId::new("openai");
        assert_eq!(format!("{v}"), "openai");
        assert_eq!(v.as_str(), "openai");
    }

    // ── ApiKeyProvider ───────────────────────────────────────

    #[tokio::test]
    async fn api_key_provider_returns_resolved_key() {
        std::env::set_var("LATTE_TEST_KEY", "sk-test-123");
        let p = ApiKeyProvider::new("anthropic", "${LATTE_TEST_KEY}");
        assert_eq!(p.token().await.unwrap(), "sk-test-123");
        std::env::remove_var("LATTE_TEST_KEY");
    }

    #[tokio::test]
    async fn api_key_provider_passes_through_plain_key() {
        let p = ApiKeyProvider::new("anthropic", "sk-plain");
        assert_eq!(p.token().await.unwrap(), "sk-plain");
    }

    #[test]
    fn api_key_provider_never_expired() {
        let p = ApiKeyProvider::new("anthropic", "k");
        assert!(!p.is_expired());
        assert_eq!(p.kind(), "api_key");
        assert_eq!(p.vendor_id().0, "anthropic");
    }

    // ── BearerProvider ──────────────────────────────────────

    #[tokio::test]
    async fn bearer_provider_returns_initial_token_when_not_expired() {
        let refresher = FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("new-token".into()) }),
        );
        let p = BearerProvider::new("anthropic", "initial", Duration::from_secs(3600), refresher);
        assert_eq!(p.token().await.unwrap(), "initial");
        assert!(!p.is_expired());
        assert_eq!(p.kind(), "bearer");
    }

    #[tokio::test]
    async fn bearer_provider_refreshes_when_expired() {
        let started_at = Instant::now() - Duration::from_secs(120);
        let refresher = FixedIntervalRefresher::with_started_at(
            started_at,
            Duration::from_secs(60),
            || Box::pin(async { Ok("new-token".into()) }),
        );
        let p = BearerProvider::with_started_at(
            "anthropic",
            "old",
            started_at,
            Duration::from_secs(60),
            refresher,
        );
        assert!(p.is_expired());
        let token = p.token().await.unwrap();
        assert_eq!(token, "new-token");
        assert!(!p.is_expired());
    }

    #[tokio::test]
    async fn bearer_provider_refresh_now_forces_refresh() {
        let refresher = FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("forced-new".into()) }),
        );
        let p = BearerProvider::new("anthropic", "initial", Duration::from_secs(3600), refresher);
        assert!(!p.is_expired());
        // 强制 refresh — 调 refresh_now（不走 token() 的过期判断）
        let token = p.refresh_now().await.unwrap();
        assert_eq!(token, "forced-new");
    }

    #[tokio::test]
    async fn bearer_provider_refresh_failure_propagates() {
        let refresher = FixedIntervalRefresher::new(
            Duration::from_secs(0),
            || Box::pin(async {
                Err(VendorError::RefreshFailed("token endpoint 503".into()))
            }),
        );
        let p = BearerProvider::new("anthropic", "stale", Duration::from_secs(0), refresher);
        let result = p.token().await;
        assert!(matches!(result, Err(VendorError::RefreshFailed(_))));
    }

    #[test]
    fn bearer_provider_remaining_seconds() {
        let refresher = FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("t".into()) }),
        );
        let p = BearerProvider::new("anthropic", "t", Duration::from_secs(3600), refresher);
        let remaining = p.remaining();
        assert!(remaining > Duration::from_secs(3599));
        assert!(remaining <= Duration::from_secs(3600));
    }

    // ── ModelDescriptor ────────────────────────────────────

    #[test]
    fn model_descriptor_new_works() {
        let d = ModelDescriptor::new("gpt-4o", "GPT-4o");
        assert_eq!(d.id, "gpt-4o");
        assert_eq!(d.display_name, "GPT-4o");
        assert!(d.context_window.is_none());
        assert!(d.vendor_specific.is_empty());
    }

    #[test]
    fn model_descriptor_serde_roundtrip() {
        let mut d = ModelDescriptor::new("claude-sonnet-4", "Claude Sonnet 4");
        d.context_window = Some(200_000);
        d.vendor_specific.insert("type".into(), "model".into());
        let json = serde_json::to_string(&d).unwrap();
        let back: ModelDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }

    // ── ModelDiscovery impls（Manual 不打网络）────────────

    #[tokio::test]
    async fn manual_discover_returns_preset_models() {
        let models = vec![
            ModelDescriptor::new("gpt-4o", "GPT-4o"),
            ModelDescriptor::new("gpt-4o-mini", "GPT-4o Mini"),
        ];
        let discover = ManualDiscover(models.clone());
        let result = discover.discover("ignored", "ignored").await.unwrap();
        assert_eq!(result, models);
        assert_eq!(discover.protocol(), "manual");
    }

    #[test]
    fn discovery_protocols() {
        assert_eq!(AnthropicModelsApi.protocol(), "anthropic");
        assert_eq!(OpenAiModelsApi.protocol(), "openai");
        assert_eq!(ManualDiscover(vec![]).protocol(), "manual");
    }
}
