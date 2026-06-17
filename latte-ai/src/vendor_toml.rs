//! TOML config 解析（Session 5）
//!
//! 提供 [`VendorConfigToml`]（deserialize 自 TOML）和 [`VendorConfig::from_toml`]
//! 转换器。配合 [`registry_from_toml_str`] 一行构造 `VendorRegistry`。
//!
//! # TOML 格式示例
//!
//! ```toml
//! [[vendors]]
//! id = "anthropic"
//! protocol = "anthropic"
//! base_url = "https://api.anthropic.com"
//! auth = { type = "api_key", key = "${ANTHROPIC_API_KEY}" }
//! disabled_features = ["extended_cache_ttl"]
//! health_check = { type = "http", path = "/v1/messages" }
//! discovery = "anthropic"
//!
//! [[vendors]]
//! id = "internal"
//! protocol = "openai"
//! base_url = "http://localhost"
//! auth = { type = "api_key", key = "k" }
//! discovery = { type = "manual", models = [
//!     { id = "custom-1", display_name = "Custom 1", context_window = 8192 }
//! ] }
//! ```
//!
//! # 用法
//!
//! use latte_ai::vendor_toml::registry_from_toml_str;
//! let toml = r#"
//! [[vendors]]
//! id = "anthropic"
//! base_url = "https://api.anthropic.com"
//! auth = { type = "api_key", key = "k" }
//! "#;
//! let reg = registry_from_toml_str(toml)?;
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::vendor::bearer::FixedIntervalRefresher;
use crate::vendor::discover::{AnthropicModelsApi, Manual, OpenAiModelsApi};
use crate::vendor::{
    ApiKeyProvider, BearerProvider, HealthCheck, ModelDescriptor, ModelDiscovery, TokenProvider,
    VendorConfig, VendorError, VendorFeature, VendorId, VendorRegistry, VendorResult,
};

/// TOML 顶层包装（绕开 `toml::from_str::<Vec<T>>` 不支持 array-of-tables 的限制）
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VendorConfigFile {
    pub vendors: Vec<VendorConfigToml>,
}

fn default_protocol() -> String {
    "openai".to_string()
}

/// TOML-friendly vendor config（可 deserialize）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VendorConfigToml {
    pub id: String,
    pub base_url: String,
    /// 协议标识（用于 metrics / 调试；discovery impl 决定实际 wire format）
    /// 默认 `"openai"`
    #[serde(default = "default_protocol")]
    pub protocol: String,
    pub auth: VendorAuthToml,
    #[serde(default)]
    pub discovery: Option<DiscoveryField>,
    #[serde(default)]
    pub health_check: Option<HealthCheckToml>,
    #[serde(default)]
    pub disabled_features: HashSet<VendorFeature>,
}

/// 接受字符串简写（如 `discovery = "openai"`）或完整结构
///（`discovery = { type = "manual", models = [...] }`）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DiscoveryField {
    /// `discovery = "openai"` / `"anthropic"` / 未知字符串 → 包装为 single-model manual
    Shortcut(String),
    /// 完整结构
    Full(VendorDiscoveryToml),
}

impl DiscoveryField {
    fn into_inner(self) -> VendorDiscoveryToml {
        match self {
            DiscoveryField::Shortcut(s) => match s.as_str() {
                "anthropic" | "anthropic_models_api" => VendorDiscoveryToml::AnthropicModelsApi,
                "openai" | "openai_models_api" => VendorDiscoveryToml::OpenAiModelsApi,
                other => VendorDiscoveryToml::Manual {
                    models: vec![ModelDescriptorToml {
                        id: other.to_string(),
                        display_name: other.to_string(),
                        context_window: None,
                    }],
                },
            },
            DiscoveryField::Full(f) => f,
        }
    }
}

/// TOML-friendly auth config
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VendorAuthToml {
    /// 静态 API key：`{ type = "api_key", key = "..." }`
    ApiKey { key: String },
    /// Bearer + 固定间隔自动 refresh：
    /// `{ type = "bearer", token = "...", interval_secs = 3600, refresh_env = "ENV_VAR" }`
    ///
    /// - `token` 初始 token
    /// - `interval_secs` token 寿命
    /// - `refresh_env` 可选：env var 名，过期时从 env 重读
    Bearer {
        token: String,
        interval_secs: u64,
        #[serde(default)]
        refresh_env: Option<String>,
    },
}

/// TOML-friendly discovery config
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VendorDiscoveryToml {
    /// `discovery = "anthropic"` → AnthropicModelsApi
    AnthropicModelsApi,
    /// `discovery = "openai"` → OpenAiModelsApi
    OpenAiModelsApi,
    /// 静态列表：
    /// `discovery = { type = "manual", models = [{ id = "...", display_name = "..." }] }`
    Manual { models: Vec<ModelDescriptorToml> },
}

/// TOML-friendly model descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescriptorToml {
    pub id: String,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
}

impl From<ModelDescriptorToml> for ModelDescriptor {
    fn from(t: ModelDescriptorToml) -> Self {
        ModelDescriptor {
            id: t.id,
            display_name: t.display_name,
            context_window: t.context_window,
            vendor_specific: HashMap::new(),
        }
    }
}

/// TOML-friendly health check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HealthCheckToml {
    Http { path: String },
    Tcp,
    None,
}

impl From<HealthCheckToml> for HealthCheck {
    fn from(t: HealthCheckToml) -> Self {
        match t {
            HealthCheckToml::Http { path } => HealthCheck::Http { path },
            HealthCheckToml::Tcp => HealthCheck::Tcp,
            HealthCheckToml::None => HealthCheck::None,
        }
    }
}

impl VendorConfig {
    /// 从 TOML config 构造 `VendorConfig`（构造 trait object + 注入 refresh callback）
    pub fn from_toml(t: VendorConfigToml) -> VendorResult<Self> {
        let id = VendorId::new(t.id);
        let auth: Arc<dyn TokenProvider> = match t.auth {
            VendorAuthToml::ApiKey { key } => Arc::new(ApiKeyProvider::new(id.clone(), key)),
            VendorAuthToml::Bearer { token, interval_secs, refresh_env } => {
                let interval = Duration::from_secs(interval_secs);
                // BearerProvider::new 要 R: BearerRefresher + 'static，传 concrete 类型
                // （函数内部会包成 Arc<dyn>）。refresh_env 为 Some 时重读 env，None 时返空串（no-op）。
                let refresher: FixedIntervalRefresher = match refresh_env {
                    Some(env_var) => {
                        let env_var_for_closure = env_var.clone();
                        FixedIntervalRefresher::new(interval, move || {
                            let env_var = env_var_for_closure.clone();
                            Box::pin(async move {
                                Ok(std::env::var(&env_var).unwrap_or_default())
                            })
                        })
                    }
                    None => FixedIntervalRefresher::new(interval, || {
                        Box::pin(async { Ok(String::new()) })
                    }),
                };
                Arc::new(BearerProvider::new(id.clone(), token, interval, refresher))
            }
        };
        let discovery: Arc<dyn ModelDiscovery> =
            match t.discovery.map(DiscoveryField::into_inner) {
                Some(VendorDiscoveryToml::AnthropicModelsApi) => Arc::new(AnthropicModelsApi),
                Some(VendorDiscoveryToml::OpenAiModelsApi) => Arc::new(OpenAiModelsApi),
                Some(VendorDiscoveryToml::Manual { models }) => {
                    let descs: Vec<ModelDescriptor> =
                        models.into_iter().map(Into::into).collect();
                    Arc::new(Manual(descs))
                }
                None => Arc::new(Manual(vec![])),
            };
        let health_check = t.health_check.map(Into::into);
        Ok(VendorConfig {
            id,
            base_url: t.base_url,
            auth,
            discovery,
            health_check,
            disabled_features: t.disabled_features,
        })
    }
}

/// 从 TOML 字符串直接构造 `VendorRegistry`
///
/// 用 `toml::de::Deserializer` + `VendorConfigFile` 顶层包装，避开
/// `toml::from_str::<Vec<T>>` 不支持 array-of-tables 的限制。
pub fn registry_from_toml_str(s: &str) -> VendorResult<VendorRegistry> {
    use serde::de::Deserialize;
    let file: VendorConfigFile = VendorConfigFile::deserialize(toml::de::Deserializer::new(s))
        .map_err(|e| VendorError::RefreshFailed(format!("TOML parse error: {e}")))?;
    let mut configs = Vec::with_capacity(file.vendors.len());
    for c in file.vendors {
        configs.push(VendorConfig::from_toml(c)?);
    }
    Ok(VendorRegistry::new(configs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn parse_minimal_api_key_vendor() {
        let toml = r#"
            [[vendors]]
            id = "anthropic"
            base_url = "https://api.anthropic.com"
            auth = { type = "api_key", key = "sk-test" }
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        assert_eq!(reg.list().len(), 1);
        let v = reg.get(&"anthropic".into()).unwrap();
        assert_eq!(v.base_url, "https://api.anthropic.com");
        assert!(v.disabled_features.is_empty());
        assert!(v.health_check.is_none());
    }

    #[tokio::test]
    async fn parse_default_protocol_is_openai() {
        let toml = r#"
            [[vendors]]
            id = "x"
            base_url = "https://x"
            auth = { type = "api_key", key = "k" }
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        assert_eq!(reg.list().len(), 1);
    }

    #[tokio::test]
    async fn parse_bearer_with_refresh_env() {
        std::env::set_var("LATTE_TEST_REFRESH", "refreshed-from-env");
        let toml = r#"
            [[vendors]]
            id = "x"
            base_url = "https://x"
            auth = { type = "bearer", token = "initial", interval_secs = 1, refresh_env = "LATTE_TEST_REFRESH" }
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        // Bearer + 1s interval → 1s 后过期
        sleep(Duration::from_millis(1100)).await;
        let token = reg.get_token(&"x".into()).await.expect("token");
        assert_eq!(token, "refreshed-from-env");
        std::env::remove_var("LATTE_TEST_REFRESH");
    }

    #[tokio::test]
    async fn parse_manual_discovery() {
        let toml = r#"
            [[vendors]]
            id = "internal"
            base_url = "http://localhost"
            auth = { type = "api_key", key = "k" }
            discovery = { type = "manual", models = [
                { id = "custom-1", display_name = "Custom 1", context_window = 8192 },
                { id = "custom-2", display_name = "Custom 2" }
            ] }
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        let models = reg
            .discover_models(&"internal".into())
            .await
            .expect("discover");
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id, "custom-1");
        assert_eq!(models[0].display_name, "Custom 1");
        assert_eq!(models[0].context_window, Some(8192));
        assert_eq!(models[1].id, "custom-2");
        assert_eq!(models[1].context_window, None);
    }

    #[tokio::test]
    async fn parse_discover_string_shortcuts() {
        // discovery = "openai" / "anthropic" 作为字符串简写
        let toml = r#"
            [[vendors]]
            id = "x"
            base_url = "https://x"
            auth = { type = "api_key", key = "k" }
            discovery = "openai"
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        // 仅验证 parse 成功（不展开 discover 网络调用）
        assert_eq!(reg.list().len(), 1);
    }

    #[tokio::test]
    async fn parse_disabled_features_string_array() {
        let toml = r#"
            [[vendors]]
            id = "anthropic"
            base_url = "https://x"
            auth = { type = "api_key", key = "k" }
            disabled_features = ["prompt_caching", "extended_cache_ttl"]
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        let v = reg.get(&"anthropic".into()).unwrap();
        assert!(v.disabled_features.contains(&VendorFeature::PromptCaching));
        assert!(v.disabled_features.contains(&VendorFeature::ExtendedCacheTtl));
        assert!(!v.disabled_features.contains(&VendorFeature::ToolUse));
    }

    #[tokio::test]
    async fn parse_health_check_http() {
        let toml = r#"
            [[vendors]]
            id = "x"
            base_url = "https://x"
            auth = { type = "api_key", key = "k" }
            health_check = { type = "http", path = "/health" }
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        let v = reg.get(&"x".into()).unwrap();
        match &v.health_check {
            Some(HealthCheck::Http { path }) => assert_eq!(path, "/health"),
            other => panic!("expected Http health check, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn parse_multiple_vendors() {
        let toml = r#"
            [[vendors]]
            id = "anthropic"
            base_url = "https://api.anthropic.com"
            auth = { type = "api_key", key = "sk-1" }

            [[vendors]]
            id = "deepseek"
            base_url = "https://api.deepseek.com"
            auth = { type = "api_key", key = "sk-2" }
        "#;
        let reg = registry_from_toml_str(toml).expect("parse should succeed");
        assert_eq!(reg.list().len(), 2);
        assert!(reg.get(&"anthropic".into()).is_some());
        assert!(reg.get(&"deepseek".into()).is_some());
    }

    #[tokio::test]
    async fn parse_invalid_toml_errors() {
        let toml = "this is not valid TOML ====";
        let result = registry_from_toml_str(toml);
        assert!(matches!(result, Err(VendorError::RefreshFailed(_))));
    }
}
