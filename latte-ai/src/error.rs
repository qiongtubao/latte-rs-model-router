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

/// 「请求没能送出去」类失败的稳定标记。
///
/// 连接建立阶段就失败（DNS 解析不了、TCP 连不上、代理拒绝、连接超时）
/// 与「厂商返回了错误」是两件事，但两者过去在日志里长得一样，都变成
/// `Stream error: Request failed: …`，再被上层归并成「所有模型不可用」。
///
/// jemalloc 2026-08-26 会话：minimaxi.com / vectide.cn / api.deepseek.com
/// 三个互不相关的域名同一秒全部 `error sending request for url`，实际是
/// 本机网络断了，日志却报「all models unavailable」，把排查引向配额和
/// 鉴权方向；model_chain 还挨个把 4 个模型都试了一遍，纯白等。
///
/// 带上这个标记，上层就能在「全链都是连接失败」时改口径报本机网络故障。
/// 用标记字符串而不是错误类型，是因为流式失败要穿过
/// `StreamEvent::Error(String)` 这道 channel，类型信息在那里已经丢了。
pub const TRANSPORT_FAILURE_MARKER: &str = "[transport]";

/// reqwest 错误是否属于「请求没送出去」——连接、DNS、代理、连接超时。
///
/// 注意 `is_timeout()` 也算：连接阶段超时同样意味着没送达。响应已经
/// 开始返回之后的读超时走 `AiError::Stream` 的空闲超时路径，不经过这里。
pub fn is_transport_failure(e: &reqwest::Error) -> bool {
    e.is_connect() || e.is_timeout() || e.is_request()
}

pub type Result<T> = std::result::Result<T, AiError>;
