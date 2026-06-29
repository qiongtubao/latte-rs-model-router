//! `proxy.toml` schema + loader.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::RouterError;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProxyConfig {
    #[serde(default)]
    pub server: ServerConfig,

    #[serde(default)]
    pub catalog: CatalogConfig,
}

impl ProxyConfig {
    pub fn from_toml_path(path: &Path) -> Result<Self, RouterError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RouterError::CatalogIo(path.to_path_buf(), e.to_string()))?;
        toml::from_str(&text)
            .map_err(|e| RouterError::CatalogParse(path.to_path_buf(), e.to_string()))
    }

    pub fn from_toml_str(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,

    #[serde(default = "default_port")]
    pub port: u16,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogConfig {
    #[serde(default = "default_models_dir")]
    pub models_dir: String,

    /// Magic model name that triggers silent priority selection.
    /// When the client sends `model = proxy_default_model`, the proxy walks
    /// `pool` in order and picks the first available physical model. The caller
    /// never learns which one was chosen.
    #[serde(default = "default_proxy_default_model")]
    pub proxy_default_model: String,

    /// Priority pool used when the client sends `proxy_default_model`.
    /// Order = priority (first = highest). Used together with the breaker.
    #[serde(default)]
    pub pool: Vec<String>,
}

impl Default for CatalogConfig {
    fn default() -> Self {
        Self {
            models_dir: default_models_dir(),
            proxy_default_model: default_proxy_default_model(),
            pool: Vec::new(),
        }
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    11434
}
fn default_models_dir() -> String {
    "~/.latte/models.d".to_string()
}
fn default_proxy_default_model() -> String {
    "proxy-default".to_string()
}
