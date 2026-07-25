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
        }
    }
}
