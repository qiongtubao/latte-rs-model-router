//! Per-model circuit breaker with Half-Open and Warmup support.
//!
//! State machine:
//!
//! ```text
//!                 429/5xx/403
//!     Closed ──────────────────→ Open
//!       ↑                          │
//!       │                    cooldown 到期
//!       │                          │
//!       │                     HalfOpen
//!       │                     (被动探测)
//!       │                          │
//!       │                 ┌────────┴────────┐
//!       │          probe 成功         probe 失败
//!       │                 │                │
//!       │        Warmup(降权)               │
//!       │       连续成功 N 次   重新 Open (退避翻倍)
//!       │                 │
//!       └─────────────────┘
//!             恢复到 Closed
//! ```
//!
//! **探活策略：不发探活请求，被动探测。**
//! Half-Open 时模型通过 `ProbeAllowed` 放行一个真实用户请求作为探针。
//!
//! **Warmup 阶段**：探针成功后不直接回到 Closed，而是进入降权状态。
//! `select_candidates` 中 Warmup 模型权重低，优先选其他稳定模型。
//! 连续成功 N 次后逐步恢复到 full weight，最终回到 Closed。
//!
//! **指数退避**：`pull_out` 时连续失败退避翻倍（60s → 120s → …）。

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use tracing::{debug, info, warn};

/// 检查模型可用性的结果
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Availability {
    /// 模型可用（Closed 状态）
    Available,
    /// 模型可用但处于 Warmup 阶段（降权），附带当前权重分
    /// 权重在 1~100 之间，`select_candidates` 按此值排序
    AvailableWithWeight(u32),
    /// 模型不可用（Open 冷却中，或 HalfOpen 已有探针在飞）
    Unavailable,
    /// 冷却到期，允许放行一个真实请求作为探针（HalfOpen）
    ProbeAllowed {
        since: DateTime<Utc>,
    },
}

/// 单个模型的断路器状态
#[derive(Debug, Clone, Copy)]
enum BreakerEntry {
    /// 正常状态，模型完全可用
    Closed,
    /// 熔断状态，模型被拉出
    Open {
        unavailable_until: DateTime<Utc>,
        #[allow(dead_code)]
        reason: &'static str,
    },
    /// 半开状态：冷却已到期，等待一个真实请求作为探针
    HalfOpen {
        available_since: DateTime<Utc>,
        probe_in_flight: bool,
    },
    /// 预热状态：探针成功，模型可用但权重低。
    /// 连续成功到 `required` 次后回到 Closed。
    Warmup {
        /// 预热期内连续成功次数
        success_count: u32,
        /// 达到此次数后回到 Closed
        required: u32,
        /// 当前权重分 (5~100)
        weight: u32,
    },
}

impl BreakerEntry {
    fn is_half_open(&self) -> bool {
        matches!(self, BreakerEntry::HalfOpen { .. })
    }
    fn is_warmup(&self) -> bool {
        matches!(self, BreakerEntry::Warmup { .. })
    }
    fn weight(&self) -> u32 {
        match self {
            BreakerEntry::Closed => 100,
            BreakerEntry::Warmup { weight, .. } => *weight,
            _ => 0,
        }
    }
}

impl Default for BreakerEntry {
    fn default() -> Self {
        Self::Closed
    }
}

/// 预热阶段最低权重
const WARMUP_MIN_WEIGHT: u32 = 5;
/// 预热阶段每次成功增加的权重
const WARMUP_WEIGHT_STEP: u32 = 19;
/// 预热阶段所需连续成功次数
const WARMUP_REQUIRED_SUCCESSES: u32 = 5;

const MIN_BACKOFF_SECS: u64 = 60;
const MAX_BACKOFF_SECS: u64 = 3600;

#[derive(Debug, Default, Clone)]
struct VendorBreaker {
    consecutive_5xx: u32,
    consecutive_403: u32,
    backoff_secs: u64,
    state: BreakerEntry,
}

impl VendorBreaker {
    fn double_backoff(&mut self, base_secs: u64) -> u64 {
        let new_secs = if self.backoff_secs == 0 {
            base_secs.max(MIN_BACKOFF_SECS)
        } else {
            self.backoff_secs.saturating_mul(2).min(MAX_BACKOFF_SECS)
        };
        self.backoff_secs = new_secs;
        new_secs
    }

    fn reset_backoff(&mut self) {
        self.backoff_secs = 0;
    }
}

#[derive(Debug, Clone, Default)]
pub struct CircuitBreaker {
    state: Arc<Mutex<HashMap<String, VendorBreaker>>>,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// 检查模型当前的可选性（只读，不修改状态）。
    /// 如需 HalfOpen 的自动迁移，调用方应使用 `check_availability_mut`。
    pub fn check_availability(&self, model_id: &str, now: DateTime<Utc>) -> Availability {
        let state = self.state.lock();
        let Some(vb) = state.get(model_id) else {
            return Availability::Available;
        };
        match vb.state {
            BreakerEntry::Closed => Availability::Available,
            BreakerEntry::Warmup { weight, .. } => Availability::AvailableWithWeight(weight),
            BreakerEntry::Open { unavailable_until, .. } => {
                if now >= unavailable_until {
                    // 只读版本不迁移，返回 Unavailable 让调用方决定
                    Availability::Unavailable
                } else {
                    Availability::Unavailable
                }
            }
            BreakerEntry::HalfOpen { probe_in_flight, .. } => {
                if probe_in_flight {
                    Availability::Unavailable
                } else {
                    Availability::ProbeAllowed { since: now }
                }
            }
        }
    }

    /// 检查并可能自动迁移状态（Open 到期 → HalfOpen）。
    /// 这是 `select` / `select_candidates` 使用的主方法。
    pub fn check_availability_mut(&self, model_id: &str, now: DateTime<Utc>) -> Availability {
        let mut state = self.state.lock();
        let vb = state.entry(model_id.to_string()).or_default();
        match vb.state {
            BreakerEntry::Closed => Availability::Available,
            BreakerEntry::Warmup { weight, .. } => Availability::AvailableWithWeight(weight),
            BreakerEntry::Open { unavailable_until, .. } => {
                if now >= unavailable_until {
                    vb.state = BreakerEntry::HalfOpen {
                        available_since: now,
                        probe_in_flight: false,
                    };
                    debug!(
                        target: "latte_router::breaker",
                        model_id = %model_id,
                        "cooldown expired, entering HalfOpen"
                    );
                    Availability::ProbeAllowed { since: now }
                } else {
                    Availability::Unavailable
                }
            }
            BreakerEntry::HalfOpen { probe_in_flight, .. } => {
                if probe_in_flight {
                    Availability::Unavailable
                } else {
                    Availability::ProbeAllowed { since: now }
                }
            }
        }
    }

    /// 获取模型当前权重（Closed=100, Warmup=5~100, 其他=0）
    pub fn weight(&self, model_id: &str) -> u32 {
        let state = self.state.lock();
        match state.get(model_id) {
            Some(vb) => vb.state.weight(),
            None => 100,
        }
    }

    /// 记录探针已被放行。
    pub fn record_probe_sent(&self, model_id: &str, now: DateTime<Utc>) {
        let mut state = self.state.lock();
        let vb = state.entry(model_id.to_string()).or_default();
        match vb.state {
            BreakerEntry::HalfOpen { .. } => {
                vb.state = BreakerEntry::HalfOpen {
                    available_since: now,
                    probe_in_flight: true,
                };
                debug!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    "HalfOpen probe dispatched"
                );
            }
            _ => {}
        }
    }

    pub fn unavailable_until(&self, model_id: &str) -> Option<DateTime<Utc>> {
        let state = self.state.lock();
        match state.get(model_id).and_then(|b| match b.state {
            BreakerEntry::Open { unavailable_until, .. } => Some(unavailable_until),
            _ => None,
        }) {
            Some(until) => Some(until),
            None => None,
        }
    }

    pub fn state_name(&self, model_id: &str) -> &'static str {
        let state = self.state.lock();
        match state.get(model_id).map(|b| b.state) {
            Some(BreakerEntry::Closed) => "Closed",
            Some(BreakerEntry::Open { .. }) => "Open",
            Some(BreakerEntry::HalfOpen { .. }) => "HalfOpen",
            Some(BreakerEntry::Warmup { .. }) => "Warmup",
            None => "Closed",
        }
    }

    pub fn is_cooling(&self, model_id: &str, now: DateTime<Utc>) -> bool {
        let state = self.state.lock();
        match state.get(model_id).and_then(|b| match b.state {
            BreakerEntry::Open { unavailable_until, .. } => Some(unavailable_until),
            _ => None,
        }) {
            Some(until) => now < until,
            None => false,
        }
    }

    /// 当前是否处于探针等待响应中
    pub fn is_probe_in_flight(&self, model_id: &str) -> bool {
        let state = self.state.lock();
        match state.get(model_id).map(|b| b.state) {
            Some(BreakerEntry::HalfOpen { probe_in_flight: true, .. }) => true,
            _ => false,
        }
    }

    /// 当前是否处于 Warmup 阶段
    pub fn is_warmup(&self, model_id: &str) -> bool {
        let state = self.state.lock();
        match state.get(model_id).map(|b| b.state) {
            Some(BreakerEntry::Warmup { .. }) => true,
            _ => false,
        }
    }

    pub fn pull_out(&self, model_id: &str, until: DateTime<Utc>, reason: &'static str) {
        let mut state = self.state.lock();
        let entry = state.entry(model_id.to_string()).or_default();
        entry.state = BreakerEntry::Open {
            unavailable_until: until,
            reason,
        };
        entry.consecutive_5xx = 0;
        entry.consecutive_403 = 0;
        entry.double_backoff(60);
        drop(state);
        warn!(
            target: "latte_router::breaker",
            model_id = %model_id,
            until = %until.to_rfc3339(),
            reason = reason,
            "model pulled out (Open)"
        );
    }

    pub fn pull_out_with_backoff(
        &self,
        model_id: &str,
        base_secs: u64,
        now: DateTime<Utc>,
        reason: &'static str,
    ) {
        let mut state = self.state.lock();
        let entry = state.entry(model_id.to_string()).or_default();
        let backoff = entry.double_backoff(base_secs);
        let cd = chrono::Duration::seconds(backoff as i64);
        let until = now + cd;
        entry.state = BreakerEntry::Open {
            unavailable_until: until,
            reason,
        };
        entry.consecutive_5xx = 0;
        entry.consecutive_403 = 0;
        drop(state);
        warn!(
            target: "latte_router::breaker",
            model_id = %model_id,
            backoff_secs = backoff,
            until = %until.to_rfc3339(),
            reason = reason,
            "model pulled out with exponential backoff (Open)"
        );
    }

    pub fn record_5xx(
        &self,
        model_id: &str,
        threshold: u32,
        now: DateTime<Utc>,
        cooldown: Duration,
    ) {
        let until_opt = {
            let mut state = self.state.lock();
            let entry = state.entry(model_id.to_string()).or_default();
            entry.consecutive_5xx += 1;
            if entry.consecutive_5xx >= threshold {
                let cd_chrono = chrono::Duration::from_std(cooldown).unwrap_or_default();
                let until = now + cd_chrono;
                entry.state = BreakerEntry::Open {
                    unavailable_until: until,
                    reason: "5xx threshold",
                };
                Some(until)
            } else {
                None
            }
        };
        match until_opt {
            Some(until) => info!(
                target: "latte_router::breaker",
                model_id = %model_id,
                consecutive_5xx = threshold,
                threshold = threshold,
                cooldown_secs = cooldown.as_secs(),
                until = %until.to_rfc3339(),
                "5xx breaker opened"
            ),
            None => debug!(
                target: "latte_router::breaker",
                model_id = %model_id,
                consecutive_5xx = threshold,
                threshold = threshold,
                "5xx counted (below threshold)"
            ),
        }
    }

    pub fn record_retry_on(
        &self,
        model_id: &str,
        status: u16,
        threshold: u32,
        now: DateTime<Utc>,
        cooldown: Duration,
    ) {
        enum RecordResult {
            Opened { until: DateTime<Utc> },
            Counted { count: u32 },
        }
        let result = {
            let mut state = self.state.lock();
            let entry = state.entry(model_id.to_string()).or_default();
            entry.consecutive_403 += 1;
            if entry.consecutive_403 >= threshold {
                let cd = chrono::Duration::from_std(cooldown).unwrap_or_default();
                let until = now + cd;
                entry.state = BreakerEntry::Open {
                    unavailable_until: until,
                    reason: "retry-on threshold",
                };
                let count = entry.consecutive_403;
                entry.consecutive_403 = 0;
                RecordResult::Opened { until }
            } else {
                RecordResult::Counted { count: entry.consecutive_403 }
            }
        };
        match result {
            RecordResult::Opened { until } => info!(
                target: "latte_router::breaker",
                model_id = %model_id,
                status = status,
                consecutive = threshold,
                threshold = threshold,
                cooldown_secs = cooldown.as_secs(),
                until = %until.to_rfc3339(),
                "retry-on breaker opened"
            ),
            RecordResult::Counted { count } => debug!(
                target: "latte_router::breaker",
                model_id = %model_id,
                status = status,
                consecutive = count,
                threshold = threshold,
                "retry-on counted (below threshold)"
            ),
        }
    }

    /// Record a success. 根据当前状态决定：
    /// - HalfOpen 探针成功 → 进入 Warmup（不是直接 Closed）
    /// - Warmup 阶段成功 → 递增计数，达到要求后回到 Closed
    /// - 其他 → reset 计数和 backoff
    pub fn reset(&self, model_id: &str) {
        let (prev_5xx, prev_403, prev_backoff, was_half_open, was_warmup) = {
            let mut state = self.state.lock();
            let entry = state.entry(model_id.to_string()).or_default();
            let p5 = entry.consecutive_5xx;
            let p4 = entry.consecutive_403;
            let pb = entry.backoff_secs;
            let was_ho = entry.state.is_half_open();
            let was_wu = entry.state.is_warmup();

            if entry.state.is_half_open() {
                // 探针成功 → Warmup
                entry.state = BreakerEntry::Warmup {
                    success_count: 1,
                    required: WARMUP_REQUIRED_SUCCESSES,
                    weight: WARMUP_MIN_WEIGHT,
                };
                info!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    weight = WARMUP_MIN_WEIGHT,
                    required = WARMUP_REQUIRED_SUCCESSES,
                    "probe succeeded, entering Warmup"
                );
            } else if let BreakerEntry::Warmup { success_count, required, weight } = entry.state {
                let new_count = success_count + 1;
                if new_count >= required {
                    // 达到要求 → Closed
                    entry.state = BreakerEntry::Closed;
                    info!(
                        target: "latte_router::breaker",
                        model_id = %model_id,
                        successes = new_count,
                        "warmup complete, returning to Closed"
                    );
                } else {
                    // 继续 warmup，提升权重
                    let new_weight = (weight + WARMUP_WEIGHT_STEP).min(100);
                    entry.state = BreakerEntry::Warmup {
                        success_count: new_count,
                        required,
                        weight: new_weight,
                    };
                    debug!(
                        target: "latte_router::breaker",
                        model_id = %model_id,
                        successes = new_count,
                        required = required,
                        weight = new_weight,
                        "warmup progress"
                    );
                }
            } else {
                // Closed 或其他 → reset 即可
                entry.consecutive_5xx = 0;
                entry.consecutive_403 = 0;
                entry.reset_backoff();
            }
            (p5, p4, pb, was_ho, was_wu)
        };

        if !was_half_open && !was_warmup {
            if prev_backoff > 0 {
                info!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    previous_backoff_secs = prev_backoff,
                    "breaker backoff reset on success"
                );
            }
            if prev_5xx > 0 || prev_403 > 0 {
                debug!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    previous_5xx = prev_5xx,
                    previous_403 = prev_403,
                    "breaker counters reset on success"
                );
            }
        }
    }
}