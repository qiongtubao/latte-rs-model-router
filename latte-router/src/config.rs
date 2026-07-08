//! Core data types: `ModelEntry`, `Route`, `RouterError`.

use std::path::PathBuf;
use std::time::Duration;

use chrono::{DateTime, Utc};
use latte_ai::models::ApiType;
use serde::{Deserialize, Serialize};

/// One model entry from a `models.d/*.toml` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,

    #[serde(default)]
    pub name: Option<String>,

    pub api: ApiType,
    pub provider: String,
    pub base_url: String,
    pub api_key: String,

    #[serde(default = "default_context_window")]
    pub context_window: u32,

    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,

    // 429 冷却调度：429 后 model 离开可选池到下个 refresh 时间点
    #[serde(default = "default_rl_anchor")]
    pub rate_limit_refresh_anchor: DateTime<Utc>,

    #[serde(default = "default_rl_interval")]
    pub rate_limit_refresh_interval_secs: u64,

    // 5xx 冷却：连续 N 次 5xx → model 离开可选池
    #[serde(default = "default_srv_threshold")]
    pub retry_count_5xx: u32,

    #[serde(default = "default_srv_cooldown")]
    pub cooldown_5xx_secs: u64,

    // 可重试的状态码：收到这些 code 后透明 fallback 到 pool 下一个 model
    // 连续 retry_on_count 次后拉出 retry_on_cooldown_secs 秒
    #[serde(default = "default_retry_on")]
    pub retry_on: Vec<u16>,

    #[serde(default = "default_retry_on_count")]
    pub retry_on_count: u32,

    #[serde(default = "default_retry_on_cooldown")]
    pub retry_on_cooldown_secs: u64,
}

fn default_context_window() -> u32 {
    65536
}
fn default_max_tokens() -> u32 {
    4096
}
fn default_rl_anchor() -> DateTime<Utc> {
    DateTime::from_timestamp(0, 0).expect("epoch is valid")
}
fn default_rl_interval() -> u64 {
    60
}
fn default_retry_on() -> Vec<u16> {
    vec![403]
}
fn default_retry_on_count() -> u32 {
    10
}
fn default_retry_on_cooldown() -> u64 {
    60 // 1 min
}
fn default_srv_threshold() -> u32 {
    5
}
fn default_srv_cooldown() -> u64 {
    600
}

impl ModelEntry {
    pub fn display_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }

    /// Compute the next refresh time strictly after `now` based on
    /// `rate_limit_refresh_anchor` + `rate_limit_refresh_interval_secs`.
    ///
    /// - If `now < anchor`: returns `anchor`.
    /// - Otherwise: returns the smallest `anchor + k*interval` such that
    ///   it is strictly greater than `now`.
    /// - If `interval == 0`: returns `now`.
    pub fn next_refresh(&self, now: DateTime<Utc>) -> DateTime<Utc> {
        if self.rate_limit_refresh_interval_secs == 0 {
            return now;
        }
        let anchor = self.rate_limit_refresh_anchor.timestamp();
        let interval = self.rate_limit_refresh_interval_secs as i64;
        let now_secs = now.timestamp();
        if now_secs < anchor {
            return self.rate_limit_refresh_anchor;
        }
        let elapsed = now_secs - anchor;
        let cycles = elapsed / interval + 1;
        let next_secs = anchor + cycles * interval;
        DateTime::from_timestamp(next_secs, 0).unwrap_or(now)
    }
    pub fn cooldown_5xx(&self) -> Duration {
        Duration::from_secs(self.cooldown_5xx_secs)
    }
    pub fn cooldown_retry_on(&self) -> Duration {
        Duration::from_secs(self.retry_on_cooldown_secs)
    }
}

/// A resolved route ready to be forwarded to an upstream vendor.
#[derive(Debug, Clone)]
pub struct Route {
    pub model_id: String,
    pub api: ApiType,
    pub base_url: String,
    pub api_key: String,
}

#[derive(Debug, thiserror::Error)]
pub enum RouterError {
    #[error("model '{0}' not in pool")]
    UnknownModel(String),

    #[error("all upstream models in cooldown (next available in {retry_after_secs}s)")]
    AllUnavailable { retry_after_secs: u64 },

    #[error("failed to read catalog file {0}: {1}")]
    CatalogIo(PathBuf, String),

    #[error("failed to parse catalog file {0}: {1}")]
    CatalogParse(PathBuf, String),
}
