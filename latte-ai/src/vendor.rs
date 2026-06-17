//! 厂商抽象：VendorId + TokenProvider trait + ApiKey/Bearer 两个实现
//!
//! # Session 1 范围
//!
//! 本 session 只做「TokenProvider 抽象 + 两个 impl」：
//! - `VendorId` newtype（防字符串拼写错误）
//! - `VendorError` 错误类型
//! - [`TokenProvider`] trait：`token()` / `refresh_now()` / `is_expired()`
//! - [`ApiKeyProvider`]：静态 key，env 替换 (`${ENV_VAR}`)
//! - [`BearerProvider`]：固定间隔自动刷新（started_at + interval + refresh 回调）
//!
//! # 后续 sessions（不在本 session 范围）
//!
//! - Session 2: ModelDiscovery trait + Anthropic / OpenAI discover impl
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
//! # Ok(())
//! # }
//! ```

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

// ─── ApiKeyProvider（无刷新） ─────────────────────────────────

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

// ─── BearerProvider（固定间隔自动刷新） ─────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;
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
        // 起始时间 = now，interval = 1 小时 → 未过期
        let refresher = bearer::FixedIntervalRefresher::new(
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
        // 起始 = 2 分钟前，interval = 1 分钟 → 已过期
        let started_at = Instant::now() - Duration::from_secs(120);
        let refresher = bearer::FixedIntervalRefresher::with_started_at(
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
        // refresh 后未过期
        assert!(!p.is_expired());
    }

    #[tokio::test]
    async fn bearer_provider_refresh_now_forces_refresh() {
        // 起始 = now，interval = 1 小时 → 未过期
        let refresher = bearer::FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("forced-new".into()) }),
        );
        let p = BearerProvider::new("anthropic", "initial", Duration::from_secs(3600), refresher);
        assert!(!p.is_expired());
        // 强制 refresh
        assert!(!p.is_expired());
        // 强制 refresh — 调 refresh_now（不走 token() 的过期判断）
        let token = p.refresh_now().await.unwrap();
        assert_eq!(token, "forced-new");
    }

    #[tokio::test]
    async fn bearer_provider_refresh_failure_propagates() {
        // interval = 0 → 立即过期
        let refresher = bearer::FixedIntervalRefresher::new(
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
        // interval = 1 小时，age = 0 → remaining 接近 1 小时
        let refresher = bearer::FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("t".into()) }),
        );
        let p = BearerProvider::new("anthropic", "t", Duration::from_secs(3600), refresher);
        let remaining = p.remaining();
        assert!(remaining > Duration::from_secs(3599));
        assert!(remaining <= Duration::from_secs(3600));
    }
}
