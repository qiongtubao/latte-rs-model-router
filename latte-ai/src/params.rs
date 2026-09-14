use serde::{Deserialize, Serialize};

/// Generation parameters for AI model calls.
///
/// All optional fields default to `None`, meaning the provider's default
/// (or the model's baked-in default) will be used.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateParams {
    /// Temperature for sampling. 0.0 = deterministic, 2.0 = very random.
    /// Typical range: 0.0–1.0. Code: 0.0–0.3.
    pub temperature: Option<f64>,

    /// Nucleus sampling: cumulative probability cutoff. 0.9 means "consider
    /// tokens that make up the top 90% of probability mass".
    pub top_p: Option<f64>,

    /// Top-K sampling: only sample from the K most likely tokens.
    pub top_k: Option<u32>,

    /// Minimum probability for a token to be considered (relative to top token).
    /// 0.01–0.1 is typical for quality filtering.
    pub min_p: Option<f64>,

    /// Penalize tokens that already appear in the text. Range -2.0 to 2.0.
    pub presence_penalty: Option<f64>,

    /// Penalize tokens based on their frequency in the text. Range -2.0 to 2.0.
    pub frequency_penalty: Option<f64>,

    /// Generic repetition penalty. >1.0 penalizes repetition.
    pub repetition_penalty: Option<f64>,

    /// Maximum number of tokens to generate.
    pub max_tokens: Option<u32>,

    /// Sequences where the model will stop generating.
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Thinking/reasoning budget. Maps to Anthropic's `thinking` or
    /// OpenAI's `reasoning_effort`.
    pub thinking_budget: Option<ThinkingBudget>,

    /// Seed for reproducible generation (if supported by provider).
    pub seed: Option<u64>,

    /// 工具列表。空表示不传 `tools`，模型就当普通聊天处理。
    #[serde(default)]
    pub tools: Vec<crate::models::Tool>,

    /// 工具调用策略。`ToolChoice::Auto`（默认）让模型自己决定。
    /// OpenAI 协议下 `None` 表示显式禁调；Anthropic 协议下不支持 None。
    #[serde(default)]
    pub tool_choice: crate::models::ToolChoice,

    /// 是否允许模型在**一条响应里返回多个 tool_call**（OpenAI 协议的
    /// `parallel_tool_calls`）。
    ///
    /// - `None`（默认）→ 字段不下发，走供应商默认（OpenAI 兼容端点默认 `true`）；
    /// - `Some(true)` → 显式要求可并行；
    /// - `Some(false)` → 强制每轮最多一个工具调用。
    ///
    /// 为什么要能显式 `None`：并非所有 OpenAI 兼容端点都认这个字段，
    /// 见 litellm #22637（Bedrock Converse 在 Claude 4.5+ 上收到它直接失败）。
    /// 遇到这种端点把它设回 `None` 即可，不必改协议实现。
    ///
    /// Anthropic 协议没有对应的**开启**开关（默认就允许并行），只有
    /// `tool_choice.disable_parallel_tool_use` 能关，因此这个字段在
    /// Anthropic 路径上不下发。
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,

    /// **缓存亲和键**：OpenAI 协议的顶层 `prompt_cache_key`。
    ///
    /// 与 [`crate::models::Model::prompt_cache`]（Anthropic 式**显式断点**）
    /// 是两条独立的机制：
    ///
    /// - 显式断点告诉供应商「从这里开始缓存」，需要端点认 `cache_control`，
    ///   不认的会 400，所以默认关；
    /// - `prompt_cache_key` 走的是供应商的**自动前缀缓存**，它不声明缓存
    ///   边界，只声明「这些请求属于同一个会话」。
    ///
    /// 为什么单靠字节稳定的前缀不够：命中自动缓存要求请求落到**持有那份
    /// KV cache 的那台机器**上，而网关的负载均衡不保证这一点。这个键就是
    /// 给供应商用来做路由亲和的。
    ///
    /// 对齐 oh-my-pi 的 `supportsPromptCacheKey` + `prompt_cache_key`
    /// （`packages/ai/src/providers/openai-completions.ts`）——它为此专门有
    /// 一组 cache-affinity 测试。
    ///
    /// `None`（默认）→ 字段不下发，行为与从前逐字节一致。调用方应传一个
    /// **整个会话内稳定**的值（通常是 session id）。
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
}

/// Thinking/reasoning budget levels.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ThinkingBudget {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Custom(u32),
}

impl ThinkingBudget {
    pub fn token_budget(&self) -> u32 {
        match self {
            ThinkingBudget::Minimal => 1024,
            ThinkingBudget::Low => 2048,
            ThinkingBudget::Medium => 8192,
            ThinkingBudget::High => 16384,
            ThinkingBudget::XHigh => 32768,
            ThinkingBudget::Custom(n) => *n,
        }
    }
}

impl GenerateParams {
    /// Create a default set of parameters suitable for code generation.
    pub fn code_defaults() -> Self {
        Self {
            temperature: Some(0.1),
            top_p: Some(0.9),
            top_k: Some(40),
            min_p: Some(0.05),
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            max_tokens: Some(4096),
            stop_sequences: vec![],
            thinking_budget: None,
            seed: None,
            tools: vec![],
            tool_choice: crate::models::ToolChoice::Auto,
            parallel_tool_calls: None,
            prompt_cache_key: None,
        }
    }

    /// Create a default set of parameters suitable for creative tasks.
    pub fn creative_defaults() -> Self {
        Self {
            temperature: Some(0.8),
            top_p: Some(0.95),
            top_k: None,
            min_p: None,
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.1),
            repetition_penalty: None,
            max_tokens: Some(4096),
            stop_sequences: vec![],
            thinking_budget: None,
            seed: None,
            tools: vec![],
            tool_choice: crate::models::ToolChoice::Auto,
            parallel_tool_calls: None,
            prompt_cache_key: None,
        }
    }

    /// Create a default set of parameters suitable for analysis/debugging.
    pub fn analysis_defaults() -> Self {
        Self {
            temperature: Some(0.2),
            top_p: Some(0.9),
            top_k: None,
            min_p: Some(0.02),
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            max_tokens: Some(8192),
            stop_sequences: vec![],
            thinking_budget: Some(ThinkingBudget::Medium),
            seed: None,
            tools: vec![],
            tool_choice: crate::models::ToolChoice::Auto,
            parallel_tool_calls: None,
            prompt_cache_key: None,
        }
    }

    /// Return a human-readable label summarizing this parameter set.
    pub fn label(&self) -> String {
        let mut parts: Vec<String> = vec![];
        if let Some(t) = self.temperature {
            parts.push(format!("t={t}"));
        }
        if let Some(p) = self.top_p {
            parts.push(format!("p={p}"));
        }
        if let Some(k) = self.top_k {
            parts.push(format!("k={k}"));
        }
        if let Some(m) = self.min_p {
            parts.push(format!("mp={m}"));
        }
        if let Some(pp) = self.presence_penalty {
            parts.push(format!("pp={pp}"));
        }
        if let Some(fp) = self.frequency_penalty {
            parts.push(format!("fp={fp}"));
        }
        if let Some(rp) = self.repetition_penalty {
            parts.push(format!("rp={rp}"));
        }
        if let Some(tb) = self.thinking_budget {
            parts.push(format!("tb={}", tb.token_budget()));
        }
        if let Some(s) = self.max_tokens {
            parts.push(format!("mt={s}"));
        }
        if parts.is_empty() {
            "defaults".into()
        } else {
            parts.join(",")
        }
    }
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            max_tokens: None,
            stop_sequences: vec![],
            thinking_budget: None,
            seed: None,
            tools: vec![],
            tool_choice: crate::models::ToolChoice::Auto,
            parallel_tool_calls: None,
            prompt_cache_key: None,
        }
    }
}
