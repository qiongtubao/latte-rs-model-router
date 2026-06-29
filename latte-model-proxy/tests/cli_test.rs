use clap::Parser;

use latte_model_proxy::Args;

#[test]
fn args_default_pool_and_proxy_default_are_empty_or_default() {
    let args = Args::parse_from(std::iter::empty::<std::ffi::OsString>());
    assert!(args.host.is_none());
    assert!(args.port.is_none());
    assert!(args.models_dir.is_none());
    assert!(args.proxy_default_model.is_none());
    assert!(args.pool.is_empty());
}

#[test]
fn args_parse_host_and_port() {
    let args = Args::parse_from([
        "latte-model-proxy",
        "--host",
        "0.0.0.0",
        "--port",
        "7777",
    ]);
    assert_eq!(args.host.as_deref(), Some("0.0.0.0"));
    assert_eq!(args.port, Some(7777));
}

#[test]
fn args_parse_pool_csv() {
    let args = Args::parse_from([
        "latte-model-proxy",
        "--pool",
        "claude-sonnet-4-20250514,deepseek-v4-flash",
    ]);
    assert_eq!(
        args.pool,
        vec![
            "claude-sonnet-4-20250514".to_string(),
            "deepseek-v4-flash".to_string(),
        ]
    );
}

#[test]
fn args_parse_models_dir() {
    let args = Args::parse_from([
        "latte-model-proxy",
        "--models-dir",
        "/tmp/custom",
    ]);
    assert_eq!(args.models_dir.as_deref(), Some("/tmp/custom"));
}

#[test]
fn args_parse_proxy_default_model() {
    let args = Args::parse_from([
        "latte-model-proxy",
        "--proxy-default-model",
        "default-route",
    ]);
    assert_eq!(args.proxy_default_model.as_deref(), Some("default-route"));
}
