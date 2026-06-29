//! Time abstraction for testing.

use chrono::{DateTime, Utc};

pub trait Clock: Send + Sync + std::fmt::Debug {
    fn now(&self) -> DateTime<Utc>;
}

#[derive(Debug, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}
