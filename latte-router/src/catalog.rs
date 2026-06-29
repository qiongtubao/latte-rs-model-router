//! Model catalog loaded from `models.d/*.toml`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::config::{ModelEntry, RouterError};

#[derive(Debug, Default, Clone)]
pub struct ModelCatalog {
    by_id: HashMap<String, ModelEntry>,
}

impl ModelCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_entries(entries: Vec<ModelEntry>) -> Self {
        let mut catalog = Self::new();
        for entry in entries {
            catalog.insert(entry);
        }
        catalog
    }

    pub fn insert(&mut self, entry: ModelEntry) {
        self.by_id.insert(entry.id.clone(), entry);
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.by_id.get(id)
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.by_id.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Load every `*.toml` from `dir`, in lexicographic order. Missing dir is a
    /// silent no-op. Returns the count of files successfully loaded.
    pub fn load_dir(&mut self, dir: &Path) -> Result<usize, RouterError> {
        if !dir.is_dir() {
            return Ok(0);
        }
        let entries = std::fs::read_dir(dir)
            .map_err(|e| RouterError::CatalogIo(dir.to_path_buf(), e.to_string()))?;
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().and_then(|s| s.to_str()) == Some("toml"))
            .collect();
        paths.sort();
        let mut count = 0;
        for path in &paths {
            self.load_file(path)?;
            count += 1;
        }
        info!(
            target: "latte_router::catalog",
            dir = %dir.display(),
            file_count = count,
            "catalog dir loaded"
        );
        Ok(count)
    }

    pub fn load_file(&mut self, path: &Path) -> Result<(), RouterError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RouterError::CatalogIo(path.to_path_buf(), e.to_string()))?;
        let mut doc: CatalogFile = toml::from_str(&text)
            .map_err(|e| RouterError::CatalogParse(path.to_path_buf(), e.to_string()))?;
        let mut ids = Vec::new();
        for entry in &mut doc.models {
            ids.push(entry.id.clone());
            let raw = entry.api_key.clone();
            entry.api_key = resolve_env(&entry.api_key);
            if entry.api_key.is_empty()
                && raw.starts_with("${")
                && raw.ends_with('}')
            {
                let var = &raw[2..raw.len() - 1];
                warn!(
                    target: "latte_router::catalog",
                    path = %path.display(),
                    model_id = %entry.id,
                    env_var = %var,
                    "api_key env var not set"
                );
            }
        }
        let before = self.by_id.len();
        for entry in doc.models {
            self.insert(entry);
        }
        let added = self.by_id.len() - before;
        info!(
            target: "latte_router::catalog",
            path = %path.display(),
            entries = added,
            ids = ?ids,
            "catalog file loaded"
        );
        Ok(())
    }

    /// Merge another catalog in. Later entries override earlier ones for matching ids.
    pub fn merge(&mut self, other: ModelCatalog) {
        for (id, entry) in other.by_id {
            self.by_id.insert(id, entry);
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
struct CatalogFile {
    #[serde(default)]
    models: Vec<ModelEntry>,
}

/// Resolve `${ENV_VAR}` in `s`; non-variable references pass through unchanged.
fn resolve_env(s: &str) -> String {
    if s.len() >= 4 && s.starts_with("${") && s.ends_with('}') {
        let var = &s[2..s.len() - 1];
        std::env::var(var).unwrap_or_else(|_| s.to_string())
    } else {
        s.to_string()
    }
}
