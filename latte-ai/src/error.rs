use thiserror::Error;

#[derive(Error, Debug)]
pub enum AiError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error: {status} - {message}")]
    Api { status: u16, message: String },

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Provider not supported: {0}")]
    UnsupportedProvider(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Rate limited, retry after {retry_after}s: {message}")]
    RateLimited {
        retry_after: f64,
        message: String,
    },

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AiError>;
