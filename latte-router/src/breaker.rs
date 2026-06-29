//! Per-model circuit breaker state (rate-limit cooldown + 5xx threshold).
//!
//! Time is provided by the caller (via the [`Clock`](crate::clock::Clock) on
//! the owning `Router`); the breaker itself is time-agnostic.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use tracing::{debug, info};

#[derive(Debug, Default, Clone, Copy)]
struct VendorBreaker {
    consecutive_5xx: u32,
    consecutive_403: u32,
    unavailable_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default)]
pub struct CircuitBreaker {
    state: Arc<Mutex<HashMap<String, VendorBreaker>>>,
}

impl CircuitBreaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` when the model can be selected right now.
    pub fn is_available(&self, model_id: &str, now: DateTime<Utc>) -> bool {
        let state = self.state.lock();
        match state.get(model_id).and_then(|b| b.unavailable_until) {
            Some(until) => now >= until,
            None => true,
        }
    }

    /// When the model is pulled out. `None` means no cooldown active.
    pub fn unavailable_until(&self, model_id: &str) -> Option<DateTime<Utc>> {
        self.state.lock().get(model_id).and_then(|b| b.unavailable_until)
    }

    /// Force-pull the model out until `until`. Resets 5xx counter.
    pub fn pull_out(&self, model_id: &str, until: DateTime<Utc>) {
        let mut state = self.state.lock();
        let entry = state.entry(model_id.to_string()).or_default();
        entry.unavailable_until = Some(until);
        entry.consecutive_5xx = 0;
        entry.consecutive_403 = 0;
        drop(state);
        info!(
            target: "latte_router::breaker",
            model_id = %model_id,
            until = %until.to_rfc3339(),
            "model pulled out"
        );
    }

    /// Record a 5xx response. Opens the breaker when consecutive count hits `threshold`.
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
                let cd = chrono::Duration::from_std(cooldown).unwrap_or_default();
                let until = now + cd;
                entry.unavailable_until = Some(until);
                Some((entry.consecutive_5xx, until))
            } else {
                None
            }
        };
        match until_opt {
            Some((count, until)) => {
                info!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    consecutive_5xx = count,
                    threshold = threshold,
                    cooldown_secs = cooldown.as_secs(),
                    until = %until.to_rfc3339(),
                    "5xx breaker opened"
                );
            }
            None => {
                debug!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    consecutive_5xx = threshold,
                    threshold = threshold,
                    "5xx counted (below threshold)"
                );
            }
        }
    }

    /// Record a retry-eligible response (e.g. 403). Opens the breaker
    /// when consecutive count hits `threshold`.
    pub fn record_retry_on(
        &self,
        model_id: &str,
        status: u16,
        threshold: u32,
        now: DateTime<Utc>,
        cooldown: Duration,
    ) {
        enum RecordResult {
            Opened { count: u32, until: DateTime<Utc> },
            Counted { count: u32 },
        }
        let result = {
            let mut state = self.state.lock();
            let entry = state.entry(model_id.to_string()).or_default();
            entry.consecutive_403 += 1;
            if entry.consecutive_403 >= threshold {
                let cd = chrono::Duration::from_std(cooldown).unwrap_or_default();
                let until = now + cd;
                entry.unavailable_until = Some(until);
                let count = entry.consecutive_403;
                entry.consecutive_403 = 0;
                RecordResult::Opened { count, until }
            } else {
                RecordResult::Counted { count: entry.consecutive_403 }
            }
        };
        match result {
            RecordResult::Opened { count, until } => {
                info!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    status = status,
                    consecutive = count,
                    threshold = threshold,
                    cooldown_secs = cooldown.as_secs(),
                    until = %until.to_rfc3339(),
                    "retry-on breaker opened"
                );
            }
            RecordResult::Counted { count } => {
                debug!(
                    target: "latte_router::breaker",
                    model_id = %model_id,
                    status = status,
                    consecutive = count,
                    threshold = threshold,
                    "retry-on counted (below threshold)"
                );
            }
        }
    }

    /// Record a success (or 4xx non-429). Resets the 5xx and 403 counters.
    pub fn reset(&self, model_id: &str) {
        let (prev_5xx, prev_403) = {
            let mut state = self.state.lock();
            let entry = state.entry(model_id.to_string()).or_default();
            let p5 = entry.consecutive_5xx;
            let p4 = entry.consecutive_403;
            entry.consecutive_5xx = 0;
            entry.consecutive_403 = 0;
            (p5, p4)
        };
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
