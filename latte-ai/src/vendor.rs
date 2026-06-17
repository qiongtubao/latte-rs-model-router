//! 厂商抽象：VendorId + TokenProvider + ModelDiscovery + VendorRegistry
//!
//! # Session 3 范围
//!
//! - `VendorId` newtype
//! - `VendorError` 错误类型
//! - [`TokenProvider`] trait + [`ApiKeyProvider`] / [`BearerProvider`] impl
//! - [`ModelDescriptor`] 轻量结构
//! - [`ModelDiscovery`] trait + [`discover::AnthropicModelsApi`] / [`discover::OpenAiModelsApi`] impl
//! - [`VendorConfig`] 完整 vendor 配置
//! - [`VendorRegistry`] 集中管理多个 vendor
//! - [`VendorStatus`] 状态查询结果
//! - [`HealthCheck`] 健康检查策略
//! - [`VendorFeature`] 厂商功能细粒度开关
//!
//! # 后续 sessions（不在本 session 范围）
//!
//! - Session 4: 单测 + examples
//!
//! # 用法
//!
//! ```rust
//! use std::sync::Arc;
//! use latte_ai::vendor::{
//!     ApiKeyProvider, ModelDescriptor, VendorConfig, VendorId, VendorRegistry,
//!     discover::Manual,
//! };
//!
//! let discovery: Arc<dyn latte_ai::vendor::ModelDiscovery> = Arc::new(Manual(vec![
//!     ModelDescriptor::new("gpt-4o", "GPT-4o"),
//! ]));
//! let auth: Arc<dyn latte_ai::vendor::TokenProvider> =
//!     Arc::new(ApiKeyProvider::new("test", "k"));
//! let v = VendorConfig::new("test", "https://api.x", auth, discovery);
//! let _reg = VendorRegistry::new(vec![v]);
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::models::TokenUsage;

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
    #[error("feature {feature:?} disabled for vendor {vendor}")]
    FeatureDisabled {
        vendor: VendorId,
        feature: VendorFeature,
    },
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
                    "openai /v1 models returned {status}"
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

// ─── VendorConfig + Registry + Status + Features ────────────

/// Vendor 可禁用的功能（细粒度开关，per-vendor 配置；可被 model-level 覆盖）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VendorFeature {
    /// Anthropic prompt caching（ephemeral / long TTL）
    PromptCaching,
    /// 1 小时 cache TTL（`extended-cache-ttl-2025-04-11` beta header）
    ExtendedCacheTtl,
    /// Tool use / function calling
    ToolUse,
    /// Thinking / reasoning budget
    Thinking,
    /// 视觉（图输入）
    Vision,
    /// Streaming response
    Stream,
    /// o1 / o3 reasoning_effort
    Reasoning,
}

impl VendorFeature {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PromptCaching => "prompt_caching",
            Self::ExtendedCacheTtl => "extended_cache_ttl",
            Self::ToolUse => "tool_use",
            Self::Thinking => "thinking",
            Self::Vision => "vision",
            Self::Stream => "stream",
            Self::Reasoning => "reasoning",
        }
    }
}

/// Vendor 健康检查策略
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HealthCheck {
    /// HTTP GET 某个路径，要求 2xx
    Http { path: String },
    /// 仅 TCP 握手
    Tcp,
    /// 不检查
    None,
}

impl Default for HealthCheck {
    fn default() -> Self {
        HealthCheck::None
    }
}

/// Vendor 完整配置（构造 `AiClient` 时用）
///
/// 不 derive `Serialize/Deserialize`：`auth: Arc<dyn TokenProvider>` 和
/// `discovery: Arc<dyn ModelDiscovery>` 是 trait object，没法直接序列化。
/// 也不 derive `Debug/Clone`：trait object 没法 derive；我们手写 `Debug` impl（不 clone）。
/// TOML config 走单独的 `VendorConfigToml`（Session 4 写）反序列化后
/// 构造 trait object，再构造成 `VendorConfig`。
pub struct VendorConfig {
    /// Vendor 标识
    pub id: VendorId,
    /// Vendor API base URL（无尾斜杠）
    pub base_url: String,
    /// Token provider（ApiKey / Bearer / 自定义）
    pub auth: Arc<dyn TokenProvider>,
    /// Model discovery strategy（Anthropic / OpenAI / Manual / 自定义）
    pub discovery: Arc<dyn ModelDiscovery>,
    /// 可选健康检查
    pub health_check: Option<HealthCheck>,
    /// 禁用的厂商功能
    pub disabled_features: HashSet<VendorFeature>,
}

impl std::fmt::Debug for VendorConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VendorConfig")
            .field("id", &self.id)
            .field("base_url", &self.base_url)
            .field("auth_kind", &self.auth.kind())
            .field("discovery", &self.discovery.protocol())
            .field("health_check", &self.health_check)
            .field("disabled_features", &self.disabled_features)
            .finish()
    }
}

impl VendorConfig {
    /// 构造 vendor config
    pub fn new(
        id: impl Into<VendorId>,
        base_url: impl Into<String>,
        auth: Arc<dyn TokenProvider>,
        discovery: Arc<dyn ModelDiscovery>,
    ) -> Self {
        Self {
            id: id.into(),
            base_url: base_url.into(),
            auth,
            discovery,
            health_check: None,
            disabled_features: HashSet::new(),
        }
    }

    pub fn with_health_check(mut self, hc: HealthCheck) -> Self {
        self.health_check = Some(hc);
        self
    }

    pub fn disable_feature(mut self, f: VendorFeature) -> Self {
        self.disabled_features.insert(f);
        self
    }
}

/// Vendor 状态查询结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorStatus {
    pub vendor: VendorId,
    /// `"api_key"` / `"bearer"` / custom
    pub auth_kind: String,
    /// Token 剩余有效秒数（`None` = 永不过期）
    pub token_remaining_secs: Option<u64>,
    /// 当前 token 是否可用
    pub auth_valid: bool,
    /// 健康检查延迟（ms）
    pub health_latency_ms: Option<u64>,
    /// 健康检查是否通过
    pub health_ok: bool,
    /// 上次 discover 出的 model 数
    pub discovered_models: Option<usize>,
}

/// VendorRegistry：所有 vendor 的中央索引
///
/// 典型用法：
/// ```no_run
/// # async fn example(reg: latte_ai::vendor::VendorRegistry) -> Result<(), latte_ai::vendor::VendorError> {
/// let token = reg.get_token(&"anthropic".into()).await?;
/// let status = reg.status(&"anthropic".into()).await?;
/// let models = reg.discover_models(&"anthropic".into()).await?;
/// # Ok(())
/// # }
/// ```
pub struct VendorRegistry {
    vendors: HashMap<VendorId, VendorConfig>,
    /// Per-vendor cumulative usage (token / cost) — see `record_usage` / `usage`
    usages: Arc<parking_lot::RwLock<HashMap<VendorId, VendorUsage>>>,
}

impl VendorRegistry {
    pub fn new(vendors: Vec<VendorConfig>) -> Self {
        Self {
            vendors: vendors.into_iter().map(|v| (v.id.clone(), v)).collect(),
            usages: Arc::new(parking_lot::RwLock::new(HashMap::new())),
        }
    }

    /// 拿 vendor 配置
    pub fn get(&self, id: &VendorId) -> Option<&VendorConfig> {
        self.vendors.get(id)
    }

    /// 列出所有 vendor
    pub fn list(&self) -> Vec<&VendorConfig> {
        self.vendors.values().collect()
    }

    /// 拿当前有效 token
    pub async fn get_token(&self, id: &VendorId) -> VendorResult<String> {
        let v = self
            .vendors
            .get(id)
            .ok_or_else(|| VendorError::NotImplemented(format!("vendor not found: {id}")))?;
        v.auth.token().await
    }

    /// 强制 refresh
    pub async fn refresh_now(&self, id: &VendorId) -> VendorResult<String> {
        let v = self
            .vendors
            .get(id)
            .ok_or_else(|| VendorError::NotImplemented(format!("vendor not found: {id}")))?;
        v.auth.refresh_now().await
    }

    /// 拉 vendor 的可用 model 列表
    pub async fn discover_models(
        &self,
        id: &VendorId,
    ) -> VendorResult<Vec<ModelDescriptor>> {
        let v = self
            .vendors
            .get(id)
            .ok_or_else(|| VendorError::NotImplemented(format!("vendor not found: {id}")))?;
        let token = v.auth.token().await?;
        v.discovery.discover(&v.base_url, &token).await
    }
    /// Record a chat's token usage + cost for the given vendor.
    ///
    /// 调用方在 `chat()` 返 `Completion` 后手动调用（或通过 `Dispatcher::record_usage` 间接调）。
    /// 未知 vendor 静默忽略（不报错），方便部分 registry 的使用。
    pub fn record_usage(&self, id: &VendorId, usage: &TokenUsage, cost_usd: f64) {
        if !self.vendors.contains_key(id) {
            return; // 未知 vendor = 静默 no-op
        }
        let mut usages = self.usages.write();
        let entry = usages.entry(id.clone()).or_default();
        entry.request_count += 1;
        entry.input_tokens += usage.input_tokens as u64;
        entry.output_tokens += usage.output_tokens as u64;
        entry.thinking_tokens += usage.thinking_tokens as u64;
        entry.total_cost_usd += cost_usd;
        entry.last_request_at = Some(Instant::now());
    }

    /// Get a snapshot of cumulative usage for one vendor
    pub fn usage(&self, id: &VendorId) -> Option<VendorUsage> {
        self.usages.read().get(id).cloned()
    }

    /// Get snapshots of cumulative usage for all vendors that have been recorded
    pub fn usages(&self) -> HashMap<VendorId, VendorUsage> {
        self.usages.read().clone()
    }

    /// Reset all usage stats (e.g., for billing period rollover)
    pub fn reset_usage(&self) {
        self.usages.write().clear();
    }

    /// 检查 vendor 健康 + token 状态（不触发 refresh）
    pub async fn status(&self, id: &VendorId) -> VendorResult<VendorStatus> {
        let v = self
            .vendors
            .get(id)
            .ok_or_else(|| VendorError::NotImplemented(format!("vendor not found: {id}")))?;
        let auth_kind = v.auth.kind().to_string();
        let auth_valid = !v.auth.is_expired();
        let token_remaining_secs = {
            let r = v.auth.remaining();
            if r >= Duration::from_secs(u64::MAX / 2) {
                None
            } else {
                Some(r.as_secs())
            }
        };
        let (health_ok, health_latency_ms) = self.ping_health(v).await;
        Ok(VendorStatus {
            vendor: v.id.clone(),
            auth_kind,
            token_remaining_secs,
            auth_valid,
            health_latency_ms,
            health_ok,
            discovered_models: None,
        })
    }

    async fn ping_health(&self, v: &VendorConfig) -> (bool, Option<u64>) {
        match &v.health_check {
            Some(HealthCheck::Http { path }) => {
                let url = format!("{}{}", v.base_url.trim_end_matches('/'), path);
                let start = std::time::Instant::now();
                let result = match v.auth.token().await {
                    Ok(t) => {
                        let client = reqwest::Client::new();
                        let req = match v.auth.kind() {
                            "api_key" => client.get(&url).header("x-api-key", t),
                            "bearer" => client.get(&url).bearer_auth(t),
                            _ => client.get(&url),
                        };
                        req.send().await
                    }
                    Err(_) => reqwest::Client::new().get(&url).send().await,
                };
                let latency = start.elapsed().as_millis() as u64;
                match result {
                    Ok(resp) if resp.status().is_success() => (true, Some(latency)),
                    _ => (false, Some(latency)),
                }
            }
            Some(HealthCheck::Tcp) | Some(HealthCheck::None) | None => (true, None),
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

    // ── VendorConfig ───────────────────────────────────────

    #[test]
    fn vendor_config_new_works() {
        let auth: Arc<dyn TokenProvider> = Arc::new(ApiKeyProvider::new("anthropic", "k"));
        let discovery: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![]));
        let v = VendorConfig::new(
            VendorId::new("anthropic"),
            "https://api.anthropic.com",
            auth,
            discovery,
        );
        assert_eq!(v.id.0, "anthropic");
        assert_eq!(v.base_url, "https://api.anthropic.com");
        assert!(v.disabled_features.is_empty());
        assert!(v.health_check.is_none());
    }

    #[test]
    fn vendor_config_builder_disable_feature() {
        let auth: Arc<dyn TokenProvider> = Arc::new(ApiKeyProvider::new("anthropic", "k"));
        let discovery: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![]));
        let v = VendorConfig::new("anthropic", "https://api.x", auth, discovery)
            .disable_feature(VendorFeature::PromptCaching)
            .disable_feature(VendorFeature::ToolUse);
        assert!(v.disabled_features.contains(&VendorFeature::PromptCaching));
        assert!(v.disabled_features.contains(&VendorFeature::ToolUse));
        assert!(!v.disabled_features.contains(&VendorFeature::Stream));
    }

    #[test]
    fn vendor_feature_strings_are_stable() {
        assert_eq!(VendorFeature::PromptCaching.as_str(), "prompt_caching");
        assert_eq!(VendorFeature::ExtendedCacheTtl.as_str(), "extended_cache_ttl");
        assert_eq!(VendorFeature::ToolUse.as_str(), "tool_use");
        assert_eq!(VendorFeature::Thinking.as_str(), "thinking");
        assert_eq!(VendorFeature::Vision.as_str(), "vision");
        assert_eq!(VendorFeature::Stream.as_str(), "stream");
        assert_eq!(VendorFeature::Reasoning.as_str(), "reasoning");
    }

    #[test]
    fn vendor_config_manual_debug() {
        let auth: Arc<dyn TokenProvider> = Arc::new(ApiKeyProvider::new("anthropic", "k"));
        let discovery: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![]));
        let v = VendorConfig::new("anthropic", "https://api.x", auth, discovery);
        let dbg = format!("{v:?}");
        assert!(dbg.contains("VendorConfig"));
        assert!(dbg.contains("anthropic"));
        assert!(dbg.contains("api_key"));
    }

    // ── VendorRegistry ─────────────────────────────────────

    fn make_registry() -> VendorRegistry {
        let auth: Arc<dyn TokenProvider> = Arc::new(ApiKeyProvider::new("anthropic", "k1"));
        let discovery: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![
            ModelDescriptor::new("claude-sonnet-4", "Claude Sonnet 4"),
        ]));
        let v1 = VendorConfig::new(
            "anthropic",
            "https://api.anthropic.com",
            auth,
            discovery,
        );

        let auth2: Arc<dyn TokenProvider> = Arc::new(ApiKeyProvider::new("deepseek", "k2"));
        let discovery2: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![
            ModelDescriptor::new("deepseek-chat", "DeepSeek Chat"),
        ]));
        let v2 = VendorConfig::new("deepseek", "https://api.deepseek.com", auth2, discovery2);

        VendorRegistry::new(vec![v1, v2])
    }

    #[test]
    fn registry_get_returns_correct_vendor() {
        let reg = make_registry();
        let v = reg.get(&VendorId::new("anthropic")).unwrap();
        assert_eq!(v.base_url, "https://api.anthropic.com");
        assert!(reg.get(&VendorId::new("missing")).is_none());
    }

    #[test]
    fn registry_list_returns_all() {
        let reg = make_registry();
        let list = reg.list();
        assert_eq!(list.len(), 2);
        let ids: Vec<_> = list.iter().map(|v| v.id.0.clone()).collect();
        assert!(ids.contains(&"anthropic".to_string()));
        assert!(ids.contains(&"deepseek".to_string()));
    }

    #[tokio::test]
    async fn registry_get_token_returns_vendor_token() {
        let reg = make_registry();
        assert_eq!(reg.get_token(&VendorId::new("anthropic")).await.unwrap(), "k1");
        assert_eq!(reg.get_token(&VendorId::new("deepseek")).await.unwrap(), "k2");
    }

    #[tokio::test]
    async fn registry_discover_models_uses_preset_for_manual() {
        let reg = make_registry();
        let models = reg.discover_models(&VendorId::new("anthropic")).await.unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "claude-sonnet-4");
    }

    #[tokio::test]
    async fn registry_status_for_api_key() {
        let reg = make_registry();
        let s = reg.status(&VendorId::new("anthropic")).await.unwrap();
        assert_eq!(s.vendor.0, "anthropic");
        assert_eq!(s.auth_kind, "api_key");
        assert!(s.auth_valid);
        // 静态 api key → token_remaining = None
        assert_eq!(s.token_remaining_secs, None);
        // health_check = None → health_ok = true, latency = None
        assert!(s.health_ok);
        assert_eq!(s.health_latency_ms, None);
    }

    #[tokio::test]
    async fn registry_status_for_missing_vendor_errors() {
        let reg = make_registry();
        let result = reg.status(&VendorId::new("nope")).await;
        assert!(matches!(result, Err(VendorError::NotImplemented(_))));
    }

    #[tokio::test]
    async fn registry_status_for_bearer() {
        let started_at = Instant::now() - Duration::from_secs(120);
        let refresher = FixedIntervalRefresher::with_started_at(
            started_at,
            Duration::from_secs(60),
            || Box::pin(async { Ok("refreshed".into()) }),
        );
        let auth: Arc<dyn TokenProvider> = Arc::new(BearerProvider::with_started_at(
            "anthropic",
            "old",
            started_at,
            Duration::from_secs(60),
            refresher,
        ));
        let discovery: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![]));
        let v = VendorConfig::new("anthropic", "https://x", auth, discovery);
        let reg = VendorRegistry::new(vec![v]);

        let s = reg.status(&VendorId::new("anthropic")).await.unwrap();
        assert_eq!(s.auth_kind, "bearer");
        // token 已过期（age 120s > interval 60s）→ auth_valid = false
        assert!(!s.auth_valid);
        // remaining saturates to 0
        assert_eq!(s.token_remaining_secs, Some(0));
    }

    #[tokio::test]
    async fn registry_refresh_now_delegates_to_provider() {
        let refresher = FixedIntervalRefresher::new(
            Duration::from_secs(3600),
            || Box::pin(async { Ok("refreshed-token".into()) }),
        );
        let auth: Arc<dyn TokenProvider> = Arc::new(BearerProvider::new(
            "anthropic",
            "initial",
            Duration::from_secs(3600),
            refresher,
        ));
        let discovery: Arc<dyn ModelDiscovery> = Arc::new(ManualDiscover(vec![]));
        let v = VendorConfig::new("anthropic", "https://x", auth, discovery);
        let reg = VendorRegistry::new(vec![v]);

        let token = reg.refresh_now(&VendorId::new("anthropic")).await.unwrap();
        assert_eq!(token, "refreshed-token");
    }
}
// ── Dispatcher ─────────────────────────────────────────────────────

/// 集中执行 vendor 策略（feature gate / token 注入 / 限流）的 hook。
///
/// 持有 `Arc<VendorRegistry>` 引用，`check()` 在 `AiClient::chat` 入口处
/// 拦截禁用特性，避免 vendor 禁用功能被偷偷使用。
///
/// # 用法
///
/// ```no_run
/// use std::sync::Arc;
/// use latte_ai::vendor::{Dispatcher, VendorRegistry, VendorId, VendorFeature};
/// use std::collections::HashSet;
///
/// # async fn example(reg: VendorRegistry) -> Result<(), Box<dyn std::error::Error>> {
/// let dispatcher = Arc::new(Dispatcher::new(Arc::new(reg)));
/// let requested: HashSet<VendorFeature> = [VendorFeature::PromptCaching].into_iter().collect();
/// dispatcher.check(&VendorId::new("anthropic"), &requested)?;
/// # Ok(())
/// # }
/// ```
pub struct Dispatcher {
    registry: Arc<VendorRegistry>,
}

impl Dispatcher {
    /// 构造 dispatcher
    pub fn new(registry: Arc<VendorRegistry>) -> Self {
        Self { registry }
    }

    /// 拿底层 registry（用于测试 / 高级查询）
    pub fn registry(&self) -> &Arc<VendorRegistry> {
        &self.registry
    }

    /// 检查 vendor 是否允许使用给定 features
    ///
    /// 返回值：被禁用的 features（空集 = 全部允许）。
    /// 行为：发现被禁用的 feature **立即短路**返 `Err(VendorError::FeatureDisabled)`。
    /// 不做 silent strip —— 调用方要明确知道被拒。
    ///
    /// vendor 不存在时返空集（不报错，让调用方继续走）。
    pub fn check(
        &self,
        vendor_id: &VendorId,
        requested: &HashSet<VendorFeature>,
    ) -> VendorResult<()> {
        let Some(v) = self.registry.get(vendor_id) else {
            return Ok(()); // vendor 未注册 = 无策略 = 放行
        };
        for f in requested {
            if v.disabled_features.contains(f) {
                return Err(VendorError::FeatureDisabled {
                    vendor: vendor_id.clone(),
                    feature: *f,
                });
            }
        }
        Ok(())
    }

    /// 拿到 vendor 配置的 `disabled_features` 引用（用于展示 / 调试）
    pub fn disabled_features(&self, vendor_id: &VendorId) -> Option<&HashSet<VendorFeature>> {
        self.registry.get(vendor_id).map(|v| &v.disabled_features)
    }
}

impl std::fmt::Debug for Dispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dispatcher")
            .field("vendors", &self.registry.list().len())
            .finish()
    }
}

#[cfg(test)]
mod dispatcher_tests {
    use super::*;
    use std::collections::HashSet;
    use crate::vendor_toml::registry_from_toml_str;

    fn make_registry() -> VendorRegistry {
        // 直接用 from_toml 构造最简 registry
        let toml = r#"
            [[vendors]]
            id = "a"
            base_url = "https://a"
            auth = { type = "api_key", key = "k" }
            disabled_features = ["prompt_caching", "tool_use"]

            [[vendors]]
            id = "b"
            base_url = "https://b"
            auth = { type = "api_key", key = "k" }
        "#;
        registry_from_toml_str(toml).expect("parse")
    }

    #[test]
    fn check_passes_when_no_disabled() {
        let reg = Arc::new(make_registry());
        let d = Dispatcher::new(reg);
        let req: HashSet<VendorFeature> = [VendorFeature::Thinking].into_iter().collect();
        d.check(&VendorId::new("b"), &req).expect("ok");
    }

    #[test]
    fn check_errors_on_disabled_feature() {
        let reg = Arc::new(make_registry());
        let d = Dispatcher::new(reg);
        let req: HashSet<VendorFeature> = [VendorFeature::PromptCaching].into_iter().collect();
        let err = d
            .check(&VendorId::new("a"), &req)
            .expect_err("should fail");
        assert!(matches!(err, VendorError::FeatureDisabled { .. }));
    }

    #[test]
    fn check_passes_for_unknown_vendor() {
        let reg = Arc::new(make_registry());
        let d = Dispatcher::new(reg);
        let req: HashSet<VendorFeature> = [VendorFeature::ToolUse].into_iter().collect();
        d.check(&VendorId::new("ghost"), &req).expect("unknown vendor ok");
    }

    #[test]
    fn check_empty_request_set_always_passes() {
        let reg = Arc::new(make_registry());
        let d = Dispatcher::new(reg);
        d.check(&VendorId::new("a"), &HashSet::new()).expect("empty ok");
    }

    #[test]
    fn disabled_features_lookup() {
        let reg = Arc::new(make_registry());
        let d = Dispatcher::new(reg);
        let feats = d
            .disabled_features(&VendorId::new("a"))
            .expect("vendor a");
        assert!(feats.contains(&VendorFeature::PromptCaching));
        assert!(feats.contains(&VendorFeature::ToolUse));
        let feats_b = d
            .disabled_features(&VendorId::new("b"))
            .expect("vendor b");
        assert!(feats_b.is_empty());
    }

    #[test]
    fn check_returns_first_disabled_only() {
        let reg = Arc::new(make_registry());
        let d = Dispatcher::new(reg);
        // 同时请求 2 个：1 个允许 + 1 个禁用 → 报错
        let req: HashSet<VendorFeature> = [
            VendorFeature::Thinking,
            VendorFeature::ToolUse,
        ]
        .into_iter()
        .collect();
        let err = d.check(&VendorId::new("a"), &req).expect_err("fail");
        match err {
            VendorError::FeatureDisabled { feature, vendor } => {
                assert_eq!(feature, VendorFeature::ToolUse);
                assert_eq!(vendor, VendorId::new("a"));
            }
            _ => panic!("expected FeatureDisabled"),
        }
    }
}
// ── Token / cost usage tracking ────────────────────────────────────────

/// Per-vendor cumulative usage stats
///
/// Updated by [`VendorRegistry::record_usage`] 在每次 `chat()` 返 `Completion` 之后由
/// 调用方手动 record（也可由 dispatcher 自动 record，见 `Dispatcher::record_usage`）。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VendorUsage {
    /// 累计请求数
    pub request_count: u64,
    /// 累计 input tokens
    pub input_tokens: u64,
    /// 累计 output tokens
    pub output_tokens: u64,
    /// 累计 thinking tokens
    pub thinking_tokens: u64,
    /// 累计花费（USD）
    pub total_cost_usd: f64,
    /// 上次 record 的 wall-clock 时间
    pub last_request_at: Option<std::time::Instant>,
}

impl VendorUsage {
    /// 平均 input tokens / request
    pub fn avg_input_tokens(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.input_tokens as f64 / self.request_count as f64
        }
    }

    /// 平均 output tokens / request
    pub fn avg_output_tokens(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.output_tokens as f64 / self.request_count as f64
        }
    }

    /// 平均花费 / request
    pub fn avg_cost_usd(&self) -> f64 {
        if self.request_count == 0 {
            0.0
        } else {
            self.total_cost_usd / self.request_count as f64
        }
    }
}

#[cfg(test)]
mod usage_tests {
    use super::*;
    use crate::models::TokenUsage;
    use crate::vendor_toml::registry_from_toml_str;

    fn make_registry() -> VendorRegistry {
        let toml = r#"
            [[vendors]]
            id = "anthropic"
            base_url = "https://api.anthropic.com"
            auth = { type = "api_key", key = "k" }
        "#;
        registry_from_toml_str(toml).expect("parse")
    }

    #[test]
    fn fresh_vendor_has_no_recorded_usage() {
        let reg = make_registry();
        // 未 record_usage → usage() 返 None；unwrap_or_default() 拿默认值
        let u = reg.usage(&"anthropic".into());
        assert!(u.is_none());
        let u = u.unwrap_or_default();
        assert_eq!(u.request_count, 0);
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
        assert_eq!(u.total_cost_usd, 0.0);
    }

    #[test]
    fn unknown_vendor_returns_none() {
        let reg = make_registry();
        assert!(reg.usage(&"ghost".into()).is_none());
    }

    #[test]
    fn record_usage_increments_counts() {
        let reg = make_registry();
        let id = VendorId::new("anthropic");
        reg.record_usage(
            &id,
            &TokenUsage {
                input_tokens: 100,
                output_tokens: 50,
                thinking_tokens: 10,
                ..Default::default()
            },
            0.001,
        );
        reg.record_usage(
            &id,
            &TokenUsage {
                input_tokens: 200,
                output_tokens: 80,
                thinking_tokens: 20,
                ..Default::default()
            },
            0.002,
        );
        let u = reg.usage(&id).expect("usage");
        assert_eq!(u.request_count, 2);
        assert_eq!(u.input_tokens, 300);
        assert_eq!(u.output_tokens, 130);
        assert_eq!(u.thinking_tokens, 30);
        assert!((u.total_cost_usd - 0.003).abs() < 1e-9);
        assert!(u.last_request_at.is_some());
        // averages
        assert!((u.avg_input_tokens() - 150.0).abs() < 1e-9);
        assert!((u.avg_output_tokens() - 65.0).abs() < 1e-9);
        assert!((u.avg_cost_usd() - 0.0015).abs() < 1e-9);
    }

    #[test]
    fn usages_returns_all_vendors() {
        let reg = make_registry();
        let id = VendorId::new("anthropic");
        reg.record_usage(
            &id,
            &TokenUsage {
                input_tokens: 10,
                ..Default::default()
            },
            0.0,
        );
        let all = reg.usages();
        assert_eq!(all.len(), 1);
        assert!(all.contains_key(&id));
    }

    #[test]
    fn record_usage_concurrent_safe() {
        use std::sync::Arc;
        use std::thread;
        let reg = Arc::new(make_registry());
        let id = VendorId::new("anthropic");
        let mut handles = vec![];
        for _ in 0..8 {
            let reg = reg.clone();
            let id = id.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    reg.record_usage(
                        &id,
                        &TokenUsage {
                            input_tokens: 1,
                            output_tokens: 2,
                            ..Default::default()
                        },
                        0.0,
                    );
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let u = reg.usage(&id).expect("usage");
        assert_eq!(u.request_count, 800);
        assert_eq!(u.input_tokens, 800);
        assert_eq!(u.output_tokens, 1600);
    }
}
