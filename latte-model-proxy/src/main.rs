//! `latte-model-proxy` — HTTP proxy in front of multiple downstream AI vendors.
//!
//! Loads `proxy.toml` (CLI override → `./proxy.toml` → `~/.latte/proxy.toml`)
//! and `models.d/*.toml` from the configured directory plus `./.latte/models.d`,
//! builds a [`latte_router::Router`], and serves axum on the configured host:port.
//!
//! Two request paths:
//! 1. Client sends `model = "proxy-default"` (or whatever `proxy.toml` names it):
//!    proxy walks `pool` in order and silently picks the first available physical
//!    model. The client never learns which model is used.
//! 2. Client sends `model = "<real model id>"` (any id present in `models.d/`):
//!    proxy uses that specific model directly.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;

use latte_model_proxy::cli::Args;
use latte_model_proxy::server::{Server, ServerRuntime};
use latte_router::{ModelCatalog, ModelEntry, Router};

fn resolve_tilde(s: &str) -> PathBuf {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if s == "~" {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(s)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args = Args::parse();
    let proxy_cfg = args
        .load_proxy_config()
        .map_err(anyhow::Error::msg)
        .context("loading proxy.toml")?;

    let host = args.host.clone().unwrap_or(proxy_cfg.server.host);
    let port = args.port.unwrap_or(proxy_cfg.server.port);
    let proxy_default_model = args
        .proxy_default_model
        .clone()
        .unwrap_or_else(|| proxy_cfg.catalog.proxy_default_model.clone());
    let pool = if !args.pool.is_empty() {
        args.pool.clone()
    } else {
        proxy_cfg.catalog.pool.clone()
    };
    let models_dir_str = args
        .models_dir
        .clone()
        .unwrap_or(proxy_cfg.catalog.models_dir);
    let models_dir = resolve_tilde(&models_dir_str);

    // Load catalog: project layer (./.latte/models.d) then user layer.
    let mut catalog = ModelCatalog::new();
    let project_dir = PathBuf::from(".latte/models.d");
    let _ = catalog.load_dir(&project_dir);
    let _ = catalog.load_dir(&models_dir);
    info!(
        target: "latte_model_proxy",
        catalog_size = catalog.len(),
        "catalog loaded"
    );

    if catalog.is_empty() {
        return Err(anyhow::anyhow!(
            "no models in catalog; create ~/.latte/models.d or ./.latte/models.d"
        ));
    }

    // Validate pool references against the catalog. Start-up error if any
    // pool id is unknown — fail fast rather than silently dropping at runtime.
    let mut pool_missing: Vec<&str> = Vec::new();
    for id in &pool {
        if catalog.get(id).is_none() {
            pool_missing.push(id);
        }
    }
    if !pool_missing.is_empty() {
        return Err(anyhow::anyhow!(
            "pool references unknown model(s): {}; check ids in models.d/*.toml",
            pool_missing.join(", ")
        ));
    }

    // Router pool = all catalog models. The client may send any of them
    // directly via `model = "<id>"`; the proxy uses Router::select for that.
    let router_pool: Vec<ModelEntry> = catalog
        .ids()
        .map(|id| catalog.get(id).expect("just enumerated").clone())
        .collect();

    info!(
        target: "latte_model_proxy",
        proxy_default_model = %proxy_default_model,
        pool = ?pool,
        "priority pool (silent selection)"
    );

    let bind_addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("bind {bind_addr}"))?;
    let local_addr = listener.local_addr()?;
    info!(
        target: "latte_model_proxy",
        bind_addr = %local_addr,
        "listening"
    );

    let runtime = ServerRuntime {
        router: std::sync::Arc::new(Router::with_system_clock(router_pool)),
        version: env!("CARGO_PKG_VERSION").to_string(),
        proxy_default_model,
        pool,
    };
    let server = Server::new(runtime);
    server.serve(listener).await?;
    Ok(())
}
