//! Route selection: walk the pool from the requested model, skipping cooldown.

use chrono::{DateTime, Utc};

use crate::breaker::CircuitBreaker;
use crate::config::{ModelEntry, Route, RouterError};

pub struct RouteSelector<'a> {
    pool: &'a [ModelEntry],
    breaker: &'a CircuitBreaker,
    now: DateTime<Utc>,
}

impl<'a> RouteSelector<'a> {
    pub fn new(pool: &'a [ModelEntry], breaker: &'a CircuitBreaker, now: DateTime<Utc>) -> Self {
        Self { pool, breaker, now }
    }

    /// Resolve a request for `model` to a `Route`.
    ///
    /// Finds `model` in the pool and walks forward, returning the first entry
    /// whose breaker is closed. If every entry from `model`'s position onward
    /// is cooling, returns [`RouterError::AllUnavailable`].
    pub fn select(&self, model: &str) -> Result<Route, RouterError> {
        let start = match self.pool.iter().position(|m| m.id == model) {
            Some(idx) => idx,
            None => return Err(RouterError::UnknownModel(model.to_string())),
        };

        for entry in &self.pool[start..] {
            if self.breaker.is_available(&entry.id, self.now) {
                return Ok(self.build_route(entry));
            }
        }

        let retry_after_secs = self.min_retry_after_secs();
        Err(RouterError::AllUnavailable { retry_after_secs })
    }

    fn build_route(&self, entry: &ModelEntry) -> Route {
        Route {
            model_id: entry.id.clone(),
            api: entry.api,
            base_url: entry.base_url.clone(),
            api_key: entry.api_key.clone(),
        }
    }

    fn min_retry_after_secs(&self) -> u64 {
        let mut min_until: Option<DateTime<Utc>> = None;
        for entry in self.pool {
            if let Some(until) = self.breaker.unavailable_until(&entry.id) {
                if until > self.now {
                    min_until = Some(match min_until {
                        Some(m) => m.min(until),
                        None => until,
                    });
                }
            }
        }
        match min_until {
            Some(until) => (until - self.now).num_seconds().max(0) as u64,
            None => 60,
        }
    }
}
