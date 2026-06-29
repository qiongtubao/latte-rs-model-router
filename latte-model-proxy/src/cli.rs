//! CLI argument parsing for `latte-model-proxy`.
//!
//! Default config lookup: `--config <path>` → `./proxy.toml` → `~/.latte/proxy.toml`.

use std::path::PathBuf;

use clap::Parser;

use latte_router::ProxyConfig;

#[derive(Debug, Clone, Parser)]
#[command(name = "latte-model-proxy", version)]
pub struct Args {
    /// Path to `proxy.toml`. Search order:
    ///   1. `--config <path>` (if provided)
    ///   2. `./proxy.toml`
    ///   3. `~/.latte/proxy.toml`
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Override `server.host` from proxy.toml.
    #[arg(long)]
    pub host: Option<String>,

    /// Override `server.port` from proxy.toml.
    #[arg(long)]
    pub port: Option<u16>,

    /// Override `catalog.models_dir` from proxy.toml.
    #[arg(long)]
    pub models_dir: Option<String>,

    /// Override `catalog.proxy_default_model` from proxy.toml.
    /// The client sends this name in `model` to trigger silent priority selection.
    #[arg(long)]
    pub proxy_default_model: Option<String>,

    /// Override `catalog.pool` from proxy.toml. Comma-separated candidate ids
    /// in priority order (first = highest).
    #[arg(long, value_delimiter = ',')]
    pub pool: Vec<String>,
}

impl Args {
    /// Resolve the `proxy.toml` path: CLI override → cwd default → home default.
    pub fn resolve_config_path(&self) -> Option<PathBuf> {
        if let Some(p) = self.config.clone() {
            return Some(p);
        }
        let cwd = std::env::current_dir().ok()?.join("proxy.toml");
        if cwd.exists() {
            return Some(cwd);
        }
        let home = std::env::var_os("HOME")?;
        let home_path = PathBuf::from(home).join(".latte/proxy.toml");
        if home_path.exists() {
            return Some(home_path);
        }
        None
    }

    /// Load `proxy.toml` (if found) and merge with CLI overrides.
    pub fn load_proxy_config(&self) -> Result<ProxyConfig, String> {
        let cfg = match self.resolve_config_path() {
            Some(p) => ProxyConfig::from_toml_path(&p)
                .map_err(|e| format!("invalid config at {}: {e}", p.display()))?,
            None => ProxyConfig::default(),
        };
        Ok(cfg)
    }
}
