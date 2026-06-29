//! Integration tests for `latte-router` — catalog loading, selection, breaker.

use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use latte_ai::models::ApiType;

use latte_router::{Clock, ModelCatalog, ModelEntry, Router, RouterError, SystemClock};

#[derive(Debug)]
struct MockClock {
    current: AtomicI64,
}

impl MockClock {
    fn new(initial: chrono::DateTime<Utc>) -> Self {
        Self {
            current: AtomicI64::new(initial.timestamp()),
        }
    }

    fn advance(&self, secs: i64) {
        self.current.fetch_add(secs, Ordering::SeqCst);
    }
}

impl Clock for MockClock {
    fn now(&self) -> chrono::DateTime<Utc> {
        Utc.timestamp_opt(self.current.load(Ordering::SeqCst), 0).unwrap()
    }
}

fn make_entry(
    id: &str,
    anchor: chrono::DateTime<Utc>,
    interval: u64,
    threshold: u32,
    cooldown: u64,
) -> ModelEntry {
    ModelEntry {
        id: id.to_string(),
        name: None,
        api: ApiType::OpenAiCompletions,
        provider: "test".to_string(),
        base_url: "http://test".to_string(),
        api_key: "k".to_string(),
        context_window: 65536,
        max_tokens: 4096,
        rate_limit_refresh_anchor: anchor,
        rate_limit_refresh_interval_secs: interval,
        retry_count_5xx: threshold,
        cooldown_5xx_secs: cooldown,
    }
}

#[test]
fn select_returns_requested_model_when_available() {
    let entry = make_entry("sonnet", Utc.timestamp_opt(0, 0).unwrap(), 60, 5, 600);
    let router = Router::with_system_clock(vec![entry]);
    let route = router.select("sonnet").unwrap();
    assert_eq!(route.model_id, "sonnet");
}

#[test]
fn select_unknown_model_returns_error() {
    let entry = make_entry("sonnet", Utc.timestamp_opt(0, 0).unwrap(), 60, 5, 600);
    let router = Router::with_system_clock(vec![entry]);
    let err = router.select("unknown").unwrap_err();
    assert!(matches!(err, RouterError::UnknownModel(s) if s == "unknown"));
}

#[test]
fn select_falls_back_to_next_pool_member_when_requested_is_cooling() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 5, 600);
    let b = make_entry("b", t0, 60, 5, 600);
    let router = Router::new(vec![a, b], clock);

    router.record("a", 429, None);

    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "b");
}

#[test]
fn record_429_with_retry_after_overrides_next_refresh() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    // interval = 60s; next_refresh(now) = t0 (since now == anchor)
    let a = make_entry("a", t0, 60, 5, 600);
    let router = Router::new(vec![a], clock.clone());

    // 429 with retry_after=120s; max(t0, t0+120) = t0+120
    router.record("a", 429, Some(120));

    clock.advance(60);
    let err = router.select("a").unwrap_err();
    assert!(matches!(err, RouterError::AllUnavailable { .. }));

    clock.advance(61);
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");
}

#[test]
fn record_429_without_retry_after_uses_next_refresh() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    // interval = 100s; next_refresh(t0) = t0, next_refresh(t0+30) = t0+100
    let a = make_entry("a", t0, 100, 5, 600);
    let router = Router::new(vec![a], clock.clone());

    // At t0+30, no Retry-After → next_refresh = t0+100 (70s away)
    clock.advance(30);
    router.record("a", 429, None);

    let err = router.select("a").unwrap_err();
    if let RouterError::AllUnavailable { retry_after_secs } = err {
        assert_eq!(retry_after_secs, 70);
    } else {
        panic!("expected AllUnavailable");
    }
}

#[test]
fn record_5xx_opens_breaker_after_threshold() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 3, 600);
    let b = make_entry("b", t0, 60, 3, 600);
    let router = Router::new(vec![a, b], clock);

    router.record("a", 500, None);
    router.record("a", 503, None);
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");

    router.record("a", 502, None);
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "b");
}

#[test]
fn record_success_resets_5xx_counter() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 3, 600);
    let router = Router::new(vec![a], clock);

    router.record("a", 500, None);
    router.record("a", 500, None);
    router.record("a", 200, None);
    router.record("a", 500, None);
    router.record("a", 500, None);

    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");
}

#[test]
fn all_pool_members_cooling_returns_all_unavailable_with_min_retry() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 60);
    let b = make_entry("b", t0, 60, 1, 120);
    let router = Router::new(vec![a, b], clock);

    router.record("a", 500, None);
    router.record("b", 500, None);

    let err = router.select("a").unwrap_err();
    if let RouterError::AllUnavailable { retry_after_secs } = err {
        assert_eq!(retry_after_secs, 60);
    } else {
        panic!("expected AllUnavailable");
    }
}

#[test]
fn after_cooldown_model_returns_to_pool() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let b = make_entry("b", t0, 60, 1, 30);
    let router = Router::new(vec![a, b], clock.clone());

    router.record("a", 500, None);
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "b");

    clock.advance(31);
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");
}

#[test]
fn catalog_loads_toml_files_from_dir() {
    let dir = std::env::temp_dir().join(format!("latte-router-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let path = dir.join("anthropic.toml");
    std::fs::write(
        &path,
        r#"
[[models]]
id = "claude-sonnet-4-20250514"
name = "Claude Sonnet 4"
api = "anthropic"
provider = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "k-test"
context_window = 200000
max_tokens = 8192
"#,
    )
    .unwrap();

    let mut catalog = ModelCatalog::new();
    let count = catalog.load_dir(&dir).unwrap();
    assert_eq!(count, 1);
    let entry = catalog.get("claude-sonnet-4-20250514").unwrap();
    assert_eq!(entry.display_name(), "Claude Sonnet 4");
    assert_eq!(entry.api_key, "k-test");
    assert_eq!(entry.context_window, 200000);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn catalog_expands_env_var_in_api_key() {
    std::env::set_var("LATTE_TEST_KEY", "secret-123");
    let dir = std::env::temp_dir().join(format!("latte-router-env-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let path = dir.join("a.toml");
    std::fs::write(
        &path,
        r#"
[[models]]
id = "a"
api = "openai"
provider = "p"
base_url = "http://x"
api_key = "${LATTE_TEST_KEY}"
"#,
    )
    .unwrap();

    let mut catalog = ModelCatalog::new();
    catalog.load_dir(&dir).unwrap();
    let entry = catalog.get("a").unwrap();
    assert_eq!(entry.api_key, "secret-123");

    std::env::remove_var("LATTE_TEST_KEY");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn catalog_load_dir_silent_skip_for_missing_dir() {
    let mut catalog = ModelCatalog::new();
    let count = catalog
        .load_dir(&PathBuf::from("/nonexistent/path/xyz"))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn catalog_merge_later_overrides_earlier() {
    let dir1 = std::env::temp_dir().join(format!("latte-router-m1-{}", std::process::id()));
    let dir2 = std::env::temp_dir().join(format!("latte-router-m2-{}", std::process::id()));
    std::fs::create_dir_all(&dir1).unwrap();
    std::fs::create_dir_all(&dir2).unwrap();

    std::fs::write(
        dir1.join("a.toml"),
        r#"
[[models]]
id = "a"
api = "openai"
provider = "p"
base_url = "http://base1"
api_key = "k1"
"#,
    )
    .unwrap();
    std::fs::write(
        dir2.join("a.toml"),
        r#"
[[models]]
id = "a"
api = "openai"
provider = "p"
base_url = "http://base2"
api_key = "k2"
"#,
    )
    .unwrap();

    let mut catalog = ModelCatalog::new();
    catalog.load_dir(&dir1).unwrap();
    catalog.load_dir(&dir2).unwrap();
    assert_eq!(catalog.get("a").unwrap().base_url, "http://base2");
    assert_eq!(catalog.get("a").unwrap().api_key, "k2");

    std::fs::remove_dir_all(&dir1).ok();
    std::fs::remove_dir_all(&dir2).ok();
}

#[test]
fn next_refresh_returns_strictly_future_time() {
    let anchor = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let entry = make_entry("a", anchor, 300, 5, 600);

    // now = anchor → first cycle strictly after now, returns anchor + 300
    assert_eq!(entry.next_refresh(anchor), anchor + chrono::Duration::seconds(300));

    // now = anchor + 1s → still first cycle, returns anchor + 300
    let t1 = anchor + chrono::Duration::seconds(1);
    assert_eq!(entry.next_refresh(t1), anchor + chrono::Duration::seconds(300));

    // now = anchor + 300s (exactly on a cycle) → next cycle, returns anchor + 600
    let t2 = anchor + chrono::Duration::seconds(300);
    assert_eq!(entry.next_refresh(t2), anchor + chrono::Duration::seconds(600));

    // now = anchor + 450s → mid second cycle, returns anchor + 600
    let t3 = anchor + chrono::Duration::seconds(450);
    assert_eq!(entry.next_refresh(t3), anchor + chrono::Duration::seconds(600));
}
