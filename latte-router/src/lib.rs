//! `latte-router` — model catalog + priority-ordered route selection + circuit breaker.
//!
//! Reads `*.toml` from a models directory (typically `~/.latte/models.d/` plus
//! `./.latte/models.d/`), builds a static catalog of [`ModelEntry`]s, and
//! provides a [`Router`] that picks a [`Route`] based on pool order (priority
//! by array position) while tracking per-model rate-limit and 5xx cooldowns.
//!
//! 断路器支持状态机：
//! - 冷却到期 → HalfOpen 等待一个真实用户请求作为探针
//! - 探针成功 → Warmup（降权），连续成功 N 次 → Closed（正常）
//! - 探针失败 → Open（指数退避冷却）
//!
//! ```no_run
//! use std::sync::Arc;
//! use latte_router::{Router, SystemClock};
//!
//! let pool = vec![ /* ModelEntry ... */ ];
//! let router = Arc::new(Router::with_system_clock(pool));
//! match router.select("claude-sonnet-4-20250514") {
//!     Ok(route) => { /* forward to route.base_url with route.api_key */ }
//!     Err(e) => { /* 503 / 404 / etc. */ }
//! }
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};
use tracing::{debug, info, warn};

pub mod breaker;
pub mod catalog;
pub mod clock;
pub mod config;
pub mod proxy_config;
pub mod selector;

pub use breaker::{Availability, CircuitBreaker};
pub use catalog::ModelCatalog;
pub use clock::{Clock, SystemClock};
pub use config::{ModelEntry, Route, RouterError};
pub use proxy_config::{CatalogConfig, ProxyConfig, ServerConfig};
pub use selector::RouteSelector;

#[derive(Debug, Clone)]
pub struct Router {
    pool: Vec<ModelEntry>,
    catalog: ModelCatalog,
    breaker: CircuitBreaker,
    clock: Arc<dyn Clock>,
}

impl Router {
    /// Build a router from a pool of models in priority order (index 0 = highest weight).
    pub fn new(pool: Vec<ModelEntry>, clock: Arc<dyn Clock>) -> Self {
        let catalog = ModelCatalog::from_entries(pool.clone());
        let breaker = CircuitBreaker::new();
        Self {
            pool,
            catalog,
            breaker,
            clock,
        }
    }

    /// Build a router using the system clock.
    pub fn with_system_clock(pool: Vec<ModelEntry>) -> Self {
        Self::new(pool, Arc::new(SystemClock))
    }

    /// Resolve a request for `model` to a [`Route`]. Walks the pool from
    /// `model`'s position onward, skipping entries whose breaker is unavailable.
    ///
    /// Half-Open 语义：冷却到期后模型进入 HalfOpen，下一个 select 会放行一个
    /// 真实请求作为探针。探针成功 → Closed，失败 → Open。
    pub fn select(&self, model: &str) -> Result<Route, RouterError> {
        let now = self.clock.now();
        debug!(
            target: "latte_router",
            requested = %model,
            pool_size = self.pool.len(),
            "router select start"
        );
        let result = RouteSelector::new(&self.pool, &self.breaker, now).select(model);
        match &result {
            Ok(route) => debug!(
                target: "latte_router",
                requested = %model,
                selected_model = %route.model_id,
                "router select OK"
            ),
            Err(RouterError::AllUnavailable { retry_after_secs }) => warn!(
                target: "latte_router",
                requested = %model,
                retry_after_secs = retry_after_secs,
                "router select: all pool members in cooldown"
            ),
            Err(RouterError::UnknownModel(_)) => warn!(
                target: "latte_router",
                requested = %model,
                pool_size = self.pool.len(),
                known = ?self.pool.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
                "router select: unknown model"
            ),
            Err(_) => {}
        }
        result
    }

    /// Walk a list of candidate model ids in order. Return the first one whose
    /// breaker is available (Closed or HalfOpen probe allowed). The model must be
    /// in the catalog (loaded from `models.d/`); ids not in the catalog are silently
    /// skipped.
    ///
    /// Half-Open 语义：冷却到期后模型进入 HalfOpen，`select_candidates` 会放行一个
    /// 真实请求作为探针。如果放行的探针还未返回（probe_in_flight=true），则跳过该模型。
    ///
    /// Use this for the "silent priority selection" path: the proxy holds a
    /// priority list (typically a configured `pool`) and asks the router to
    /// pick the first available physical model without exposing the choice to
    /// the caller.
    pub fn select_candidates(&self, candidates: &[String]) -> Result<Route, RouterError> {
        if candidates.is_empty() {
            return Err(RouterError::AllUnavailable { retry_after_secs: 60 });
        }
        let now = self.clock.now();
        debug!(
            target: "latte_router",
            candidates = ?candidates,
            "router select_candidates start"
        );

        // 收集所有可选 candidate 及其权重
        // (weight, candidate_index, model_id, is_probe)
        // weight 越高越优先，ProbeAllowed 最低
        let mut available: Vec<(u32, usize, &str, bool)> = Vec::new();

        for (idx, candidate_id) in candidates.iter().enumerate() {
            let Some(entry) = self.catalog.get(candidate_id) else {
                continue;
            };
            let avail = self.breaker.check_availability_mut(candidate_id, now);
            match avail {
                Availability::Available => {
                    // Closed: 最高权重
                    available.push((100, idx, &entry.id, false));
                }
                Availability::AvailableWithWeight(weight) => {
                    // Warmup: 按当前权重
                    available.push((weight, idx, &entry.id, false));
                }
                Availability::ProbeAllowed { .. } => {
                    // HalfOpen 探针: 权重设为 99（低于 Closed 100，高于 Warmup）
                    // 这样在没有 Closed 模型时优先放行探针
                    available.push((99, idx, &entry.id, true));
                }
                Availability::Unavailable => {
                    debug!(
                        target: "latte_router",
                        skipped = %entry.id,
                        "router select_candidates skip (unavailable)"
                    );
                }
            }
        }

        // 按权重降序、原始顺序升序排序
        available.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));

        if let Some((_weight, _idx, selected_id, is_probe)) = available.into_iter().next() {
            if is_probe {
                self.breaker.record_probe_sent(selected_id, now);
            }
            let entry = self.catalog.get(selected_id).expect("just checked");
            debug!(
                target: "latte_router",
                selected = %entry.id,
                weight = _weight,
                "router select_candidates OK"
            );
            return Ok(Route {
                model_id: entry.id.clone(),
                api: entry.api,
                base_url: entry.base_url.clone(),
                api_key: entry.api_key.clone(),
            });
        }

        // All unavailable
        let mut min_until: Option<DateTime<Utc>> = None;
        for id in candidates {
            if let Some(until) = self.breaker.unavailable_until(id) {
                if until > now {
                    min_until = Some(match min_until {
                        Some(m) => m.min(until),
                        None => until,
                    });
                }
            }
        }
        let retry_after_secs = match min_until {
            Some(until) => (until - now).num_seconds().max(0) as u64,
            None => 60,
        };
        warn!(
            target: "latte_router",
            candidates = ?candidates,
            retry_after_secs = retry_after_secs,
            "router select_candidates: all in cooldown"
        );
        Err(RouterError::AllUnavailable { retry_after_secs })
    }

    /// Record an upstream response for `model_id`. Updates the breaker:
    ///
    /// - `429`: pull model out until `max(breaker.next_refresh(now), now + retry_after)`
    /// - `entry.retry_on` (e.g. `[403]`): count consecutive hits; if it reaches
    ///   `entry.retry_on_count`, pull out for `entry.retry_on_cooldown_secs`
    /// - `5xx`: increment counter; if it reaches `entry.retry_count_5xx`, pull
    ///   out for `entry.cooldown_5xx_secs`
    /// - other: reset the 5xx and retry_on counters (also transitions from HalfOpen to Closed)
    pub fn record(&self, model_id: &str, status: u16, retry_after_secs: Option<u64>) {
        let Some(entry) = self.pool.iter().find(|m| m.id == model_id) else {
            warn!(
                target: "latte_router",
                model_id = %model_id,
                status = status,
                "router record on unknown model"
            );
            return;
        };
        let now = self.clock.now();
        if status == 429 {
            let is_probe = self.breaker.is_probe_in_flight(model_id);
            let computed = entry.next_refresh(now);
            let computed_secs = computed.timestamp();
            let now_secs = now.timestamp();
            let retry_secs = retry_after_secs.unwrap_or(0) as i64;
            let until_secs = std::cmp::max(computed_secs, now_secs + retry_secs);
            let until = DateTime::from_timestamp(until_secs, 0).unwrap_or(now);
            if is_probe {
                // 探针失败，翻倍退避
                let base = (until_secs - now_secs).max(60) as u64;
                self.breaker.pull_out_with_backoff(model_id, base, now, "429 probe failure");
            } else {
                info!(
                    target: "latte_router",
                    model_id = %model_id,
                    status = status,
                    retry_after_header = retry_after_secs.unwrap_or(0),
                    scheduled_refresh = computed.to_rfc3339(),
                    effective_until = %until.to_rfc3339(),
                    "upstream 429, applying pull-out"
                );
                self.breaker.pull_out(model_id, until, "429 rate limit");
            }
        } else if entry.retry_on.contains(&status) {
            let is_probe = self.breaker.is_probe_in_flight(model_id);
            if is_probe {
                // 探针失败，翻倍退避
                let base = entry.retry_on_cooldown_secs;
                self.breaker.pull_out_with_backoff(model_id, base, now, "retry-on probe failure");
            } else {
                self.breaker.record_retry_on(
                    model_id,
                    status,
                    entry.retry_on_count,
                    now,
                    entry.cooldown_retry_on(),
                );
            }
        } else if (500..600).contains(&status) {
            self.breaker
                .record_5xx(model_id, entry.retry_count_5xx, now, entry.cooldown_5xx());
        } else {
            self.breaker.reset(model_id);
            debug!(
                target: "latte_router",
                model_id = %model_id,
                status = status,
                "upstream success/4xx (reset counters, probe succeeded)"
            );
        }
    }

    /// Models in the pool, in priority order.
    pub fn pool(&self) -> &[ModelEntry] {
        &self.pool
    }

    /// Model ids in the pool, in priority order.
    pub fn model_ids(&self) -> Vec<String> {
        self.pool.iter().map(|m| m.id.clone()).collect()
    }

    /// Catalog of all known models (a superset of the pool).
    pub fn catalog(&self) -> &ModelCatalog {
        &self.catalog
    }

    /// 查询模型当前的熔断状态名称（"Closed" / "Open" / "HalfOpen" / "Warmup"）。
    pub fn breaker_state(&self, model_id: &str) -> &'static str {
        self.breaker.state_name(model_id)
    }

    /// 检查模型当前的可用性（会触发 Open→HalfOpen 的自动迁移）。
    pub fn check_availability(&self, model_id: &str) -> Availability {
        self.breaker.check_availability_mut(model_id, self.clock.now())
    }
}