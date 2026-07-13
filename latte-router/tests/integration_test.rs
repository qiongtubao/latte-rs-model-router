//! Integration tests for `latte-router` — catalog loading, selection, breaker.

use std::sync::atomic::AtomicI64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use chrono::{DateTime, TimeZone, Utc};
use latte_ai::models::ApiType;
use latte_router::{Availability, Clock, ModelCatalog, ModelEntry, Router, RouterError, SystemClock};

#[derive(Debug)]
struct MockClock {
    current: AtomicI64,
}

impl MockClock {
    fn new(t0: DateTime<Utc>) -> Self {
        Self { current: AtomicI64::new(t0.timestamp()) }
    }
    fn advance(&self, secs: i64) {
        self.current.fetch_add(secs, Ordering::SeqCst);
    }
}

impl Clock for MockClock {
    fn now(&self) -> DateTime<Utc> {
        Utc.timestamp_opt(self.current.load(Ordering::SeqCst), 0).unwrap()
    }
}

fn make_entry(id: &str, anchor: DateTime<Utc>, interval: u64, threshold: u32, cooldown: u64) -> ModelEntry {
    ModelEntry {
        id: id.to_string(), name: None, api: ApiType::OpenAiCompletions,
        provider: "test".to_string(), base_url: "http://test".to_string(), api_key: "k".to_string(),
        context_window: 65536, max_tokens: 4096,
        rate_limit_refresh_anchor: anchor, rate_limit_refresh_interval_secs: interval,
        retry_count_5xx: threshold, cooldown_5xx_secs: cooldown,
        retry_on: vec![403], retry_on_count: 10, retry_on_cooldown_secs: 600,
        supports_vision: false,
    }
}

// ==================== basic selection ====================

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

// ==================== breaker / pull-out / fallback ====================

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
    let a = make_entry("a", t0, 60, 5, 600);
    let router = Router::new(vec![a], clock.clone());
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
    let a = make_entry("a", t0, 100, 5, 600);
    let router = Router::new(vec![a], clock.clone());
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
fn record_403_below_threshold_does_not_pull_out() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 5, 600);
    let router = Router::new(vec![a], clock.clone());
    for _ in 0..9 {
        router.record("a", 403, None);
    }
    assert!(router.select("a").is_ok());
}

#[test]
fn record_403_after_threshold_pulls_out_model() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 5, 600);
    let b = make_entry("b", t0, 60, 5, 600);
    let router = Router::new(vec![a, b], clock.clone());
    for _ in 0..10 {
        router.record("a", 403, None);
    }
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "b");
    let route = router.select_candidates(&["a".to_string(), "b".to_string()]).unwrap();
    assert_eq!(route.model_id, "b");
}

#[test]
fn record_403_counter_resets_on_success() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 5, 600);
    let router = Router::new(vec![a], clock.clone());
    for _ in 0..9 {
        router.record("a", 403, None);
    }
    router.record("a", 200, None);
    router.record("a", 403, None);
    assert!(router.select("a").is_ok());
}

// ==================== half-open + warmup lifecycle ====================

/// 探针成功后进入 Warmup（非直接 Closed），连续 5 次成功后恢复
#[test]
fn probe_success_enters_warmup_then_recovers() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let b = make_entry("b", t0, 60, 1, 30);
    let router = Router::new(vec![a, b], clock.clone());

    // 拉出 a
    router.record("a", 500, None);
    assert_eq!(router.breaker_state("a"), "Open");

    // 冷却到期 → HalfOpen
    clock.advance(31);
    let _avail = router.check_availability("a");
    assert_eq!(router.breaker_state("a"), "HalfOpen");

    // 放行探针，探针成功 → Warmup
    let _route = router.select("a").unwrap();
    router.record("a", 200, None);
    assert_eq!(router.breaker_state("a"), "Warmup");

    // Warmup 阶段权重低，select_candidates 应选 Closed 的 b
    let route = router.select_candidates(&["a".to_string(), "b".to_string()]).unwrap();
    assert_eq!(route.model_id, "b");

    // 连续成功 4 次（第 2-5 次）→ Closed
    for _ in 0..4 {
        router.record("a", 200, None);
    }
    assert_eq!(router.breaker_state("a"), "Closed");

    // 恢复到 Closed，select_candidates 应选 a（优先级高）
    let route = router.select_candidates(&["a".to_string(), "b".to_string()]).unwrap();
    assert_eq!(route.model_id, "a");
}

/// Warmup 阶段失败 → 回到 Open
#[test]
fn warmup_failure_reopens_breaker() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let b = make_entry("b", t0, 60, 1, 30);
    let router = Router::new(vec![a, b], clock.clone());

    // 拉出 → HalfOpen → 探针成功 → Warmup
    router.record("a", 500, None);
    clock.advance(31);
    let _avail = router.check_availability("a");
    let _route = router.select("a").unwrap();
    router.record("a", 200, None);
    assert_eq!(router.breaker_state("a"), "Warmup");

    // Warmup 阶段失败 → 回到 Open
    router.record("a", 429, None);
    assert_eq!(router.breaker_state("a"), "Open");
}

/// `select` 支持 Warmup 状态（精确选模型时可选中 warmup 模型）
#[test]
fn select_works_in_warmup_state() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let router = Router::new(vec![a], clock.clone());

    router.record("a", 500, None);
    clock.advance(31);
    let _avail = router.check_availability("a");
    let _route = router.select("a").unwrap();
    router.record("a", 200, None);
    assert_eq!(router.breaker_state("a"), "Warmup");

    // select("a") 在 warmup 下仍然可用
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");
}

/// 无 Closed 模型时，优先放行探针（weight=99 > warmup=5）
#[test]
fn probe_has_higher_priority_than_warmup() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30); // 被拉出 → HalfOpen
    let b = make_entry("b", t0, 60, 1, 30); // Warmup(降权)
    let router = Router::new(vec![a, b], clock.clone());

    // b 进入 Warmup
    router.record("b", 500, None);
    clock.advance(31);
    let _avail = router.check_availability("b");
    let _route = router.select("b").unwrap();
    router.record("b", 200, None);
    assert_eq!(router.breaker_state("b"), "Warmup");

    // a 被拉出进入 HalfOpen
    router.record("a", 500, None);
    clock.advance(31);
    let _avail = router.check_availability("a");
    assert_eq!(router.breaker_state("a"), "HalfOpen");

    // 没有 Closed 模型时，ProbeAllowed(99) > Warmup(5)
    // 应选 a（放行探针）而非 b（warmup 中）
    let route = router.select_candidates(&["a".to_string(), "b".to_string()]).unwrap();
    assert_eq!(route.model_id, "a");
}

// ==================== half-open probe (basics) ====================

#[test]
fn halfopen_only_one_probe_allowed_at_a_time() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let b = make_entry("b", t0, 60, 1, 30);
    let router = Router::new(vec![a, b], clock.clone());
    router.record("a", 500, None);
    clock.advance(31);
    let _avail = router.check_availability("a");
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");
    assert_eq!(router.breaker_state("a"), "HalfOpen");
    // 第二个请求 fallback 到 b（探针在飞）
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "b");
}

#[test]
fn halfopen_probe_failure_reopens_breaker() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let b = make_entry("b", t0, 60, 1, 30);
    let router = Router::new(vec![a, b], clock.clone());
    router.record("a", 500, None);
    clock.advance(31);
    let _avail = router.check_availability("a");
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "a");
    // 探针失败
    router.record("a", 500, None);
    assert_eq!(router.breaker_state("a"), "Open");
    let route = router.select("a").unwrap();
    assert_eq!(route.model_id, "b");
}

// ==================== exponential backoff ====================

#[test]
fn probe_failure_triggers_exponential_backoff() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let router = Router::new(vec![a], clock.clone());
    // 第一次 429：pull_out → backoff_secs=60
    router.record("a", 429, None);
    clock.advance(61);
    let _avail = router.check_availability("a");
    assert_eq!(router.breaker_state("a"), "HalfOpen");
    // 放行探针，探针 429 失败 → pull_out_with_backoff → backoff=120
    let _route = router.select("a").unwrap();
    router.record("a", 429, None);
    assert_eq!(router.breaker_state("a"), "Open");
    clock.advance(120);
    let _avail = router.check_availability("a");
    assert_eq!(router.breaker_state("a"), "HalfOpen");
    // 探针成功 → Warmup
    let _route = router.select("a").unwrap();
    router.record("a", 200, None);
    assert_eq!(router.breaker_state("a"), "Warmup");
    // 连续成功 4 次 → Closed
    for _ in 0..4 {
        router.record("a", 200, None);
    }
    assert_eq!(router.breaker_state("a"), "Closed");
    // 再次 429，backoff 已重置
    router.record("a", 429, None);
    clock.advance(61);
    let _avail = router.check_availability("a");
    assert_eq!(_avail, Availability::ProbeAllowed { since: clock.now() });
}

#[test]
fn non_probe_5xx_uses_config_cooldown() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let a = make_entry("a", t0, 60, 1, 30);
    let router = Router::new(vec![a], clock.clone());
    router.record("a", 500, None);
    assert_eq!(router.breaker_state("a"), "Open");
    clock.advance(29);
    assert_eq!(router.check_availability("a"), Availability::Unavailable);
    clock.advance(2);
    let avail = router.check_availability("a");
    assert_eq!(avail, Availability::ProbeAllowed { since: clock.now() });
}

#[test]
fn non_probe_403_uses_retry_on_cooldown() {
    let t0 = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let clock = Arc::new(MockClock::new(t0));
    let mut a = make_entry("a", t0, 60, 1, 30);
    a.retry_on_count = 1;
    a.retry_on_cooldown_secs = 30;
    let router = Router::new(vec![a], clock.clone());
    router.record("a", 403, None);
    assert_eq!(router.breaker_state("a"), "Open");
    clock.advance(29);
    assert_eq!(router.check_availability("a"), Availability::Unavailable);
    clock.advance(2);
    let avail = router.check_availability("a");
    assert_eq!(avail, Availability::ProbeAllowed { since: clock.now() });
}

// ==================== catalog ====================

#[test]
fn catalog_loads_toml_files_from_dir() {
    let mut dir = std::env::temp_dir();
    dir.push(format!("latte_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let f1 = dir.join("01-openai.toml");
    std::fs::write(&f1, r#"
[[models]]
id = "gpt-4"
api = "openai-completions"
provider = "openai"
base_url = "https://api.openai.com"
api_key = "sk-xxx"
"#).unwrap();
    let f2 = dir.join("02-anthropic.toml");
    std::fs::write(&f2, r#"
[[models]]
id = "claude-opus"
api = "anthropic-messages"
provider = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "sk-xxx"
"#).unwrap();
    let mut cat = ModelCatalog::new();
    let n = cat.load_dir(&dir).unwrap();
    assert_eq!(n, 2);
    assert!(cat.get("gpt-4").is_some());
    assert!(cat.get("claude-opus").is_some());
    assert_eq!(cat.get("gpt-4").unwrap().api, ApiType::OpenAiCompletions);
    assert_eq!(cat.get("claude-opus").unwrap().api, ApiType::AnthropicMessages);
    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn catalog_toml_supports_vision_field() {
    // supports_vision 缺省 = false；显式 true 时反序列化并保留。
    // 这是 proxy-default 路径上按能力过滤候选的依赖。
    let mut dir = std::env::temp_dir();
    dir.push(format!("latte_test_vision_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let legacy = dir.join("legacy.toml");
    std::fs::write(&legacy, r#"
[[models]]
id = "no-vision"
api = "openai-completions"
provider = "p"
base_url = "https://example"
api_key = "sk-xxx"
"#).unwrap();

    let vision = dir.join("vision.toml");
    std::fs::write(&vision, r#"
[[models]]
id = "yes-vision"
api = "openai-completions"
provider = "p"
base_url = "https://example"
api_key = "sk-xxx"
supports_vision = true
"#).unwrap();

    let mut cat = ModelCatalog::new();
    cat.load_dir(&dir).unwrap();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(!cat.get("no-vision").unwrap().supports_vision);
    assert!(cat.get("yes-vision").unwrap().supports_vision);
}

#[test]
fn catalog_expands_env_var_in_api_key() {
    std::env::set_var("LATTE_TEST_KEY", "env-val");
    let toml_str = r#"[[models]]
id = "m"
api = "openai-completions"
provider = "p"
base_url = "http://u"
api_key = "${LATTE_TEST_KEY}""#;
    let mut cat = ModelCatalog::new();
    let dir = std::env::temp_dir();
    let f = dir.join("test_expand.toml");
    std::fs::write(&f, toml_str).unwrap();
    cat.load_file(&f).unwrap();
    let _ = std::fs::remove_file(&f);
    assert_eq!(cat.get("m").unwrap().api_key, "env-val");
}

#[test]
fn catalog_load_dir_silent_skip_for_missing_dir() {
    let mut dir = std::env::temp_dir();
    dir.push("nonexistent_latte_test_dir_12345");
    let _ = std::fs::remove_dir_all(&dir);
    let mut cat = ModelCatalog::new();
    let n = cat.load_dir(&dir).unwrap();
    assert_eq!(n, 0);
}

#[test]
fn catalog_merge_later_overrides_earlier() {
    let a = ModelEntry {
        id: "m".to_string(), name: None, api: ApiType::OpenAiCompletions,
        provider: "p1".to_string(), base_url: "http://a".to_string(), api_key: "k1".to_string(),
        context_window: 65536, max_tokens: 4096,
        rate_limit_refresh_anchor: Utc.timestamp_opt(0, 0).unwrap(), rate_limit_refresh_interval_secs: 60,
        retry_count_5xx: 5, cooldown_5xx_secs: 600,
        retry_on: vec![403], retry_on_count: 10, retry_on_cooldown_secs: 600,
        supports_vision: false,
    };
    let b = ModelEntry { provider: "p2".to_string(), base_url: "http://b".to_string(), api_key: "k2".to_string(), ..a.clone() };
    let cat1 = ModelCatalog::from_entries(vec![a]);
    let cat2 = ModelCatalog::from_entries(vec![b]);
    let mut merged = cat1.clone();
    merged.merge(cat2);
    let m = merged.get("m").unwrap();
    assert_eq!(m.provider, "p2");
    assert_eq!(m.base_url, "http://b");
    assert_eq!(m.api_key, "k2");
}

// ==================== next_refresh ====================

#[test]
fn next_refresh_returns_strictly_future_time() {
    let anchor = Utc.timestamp_opt(1_700_000_000, 0).unwrap();
    let entry = make_entry("m", anchor, 100, 5, 600);
    let next = entry.next_refresh(anchor);
    assert_eq!(next, Utc.timestamp_opt(1_700_000_100, 0).unwrap());
    let now2 = Utc.timestamp_opt(1_700_000_050, 0).unwrap();
    let next2 = entry.next_refresh(now2);
    assert_eq!(next2, Utc.timestamp_opt(1_700_000_100, 0).unwrap());
    let now3 = Utc.timestamp_opt(1_700_000_250, 0).unwrap();
    let next3 = entry.next_refresh(now3);
    assert_eq!(next3, Utc.timestamp_opt(1_700_000_300, 0).unwrap());
    let entry_z = make_entry("m", anchor, 0, 5, 600);
    assert_eq!(entry_z.next_refresh(now3), now3);
}