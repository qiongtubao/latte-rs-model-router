use latte_ai::params::{GenerateParams, ThinkingBudget};

/// A single parameter combination to test.
#[derive(Debug, Clone)]
pub struct ParamSweep {
    pub label: String,
    pub params: GenerateParams,
}

/// A sweep result for one parameter combination on one test prompt.
#[derive(Debug, Clone)]
pub struct SweepResult {
    pub sweep_label: String,
    pub prompt_name: String,
    pub category: String,
    pub output: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub thinking_tokens: u32,
    pub duration_ms: u64,
    pub error: Option<String>,
}

/// Generate a focused sweep of common programming parameter combinations.
pub fn programming_sweep() -> Vec<ParamSweep> {
    vec![
        // Conservative (deterministic code)
        ParamSweep {
            label: "conservative".into(),
            params: GenerateParams {
                temperature: Some(0.0),
                top_p: Some(0.9),
                top_k: Some(40),
                min_p: Some(0.05),
                ..Default::default()
            },
        },
        // Low temperature (typical code)
        ParamSweep {
            label: "low-temp".into(),
            params: GenerateParams {
                temperature: Some(0.1),
                top_p: Some(0.9),
                top_k: Some(40),
                min_p: Some(0.05),
                ..Default::default()
            },
        },
        // Balanced
        ParamSweep {
            label: "balanced".into(),
            params: GenerateParams {
                temperature: Some(0.3),
                top_p: Some(0.92),
                top_k: Some(50),
                min_p: Some(0.02),
                ..Default::default()
            },
        },
        // Creative
        ParamSweep {
            label: "creative".into(),
            params: GenerateParams {
                temperature: Some(0.7),
                top_p: Some(0.95),
                ..Default::default()
            },
        },
        // No minP, medium temperature
        ParamSweep {
            label: "no-minp".into(),
            params: GenerateParams {
                temperature: Some(0.2),
                top_p: Some(0.9),
                min_p: None,
                ..Default::default()
            },
        },
        // With repetition penalty
        ParamSweep {
            label: "anti-repetition".into(),
            params: GenerateParams {
                temperature: Some(0.3),
                top_p: Some(0.9),
                frequency_penalty: Some(0.2),
                presence_penalty: Some(0.1),
                ..Default::default()
            },
        },
        // Thinking mode (for models that support it)
        ParamSweep {
            label: "thinking-medium".into(),
            params: GenerateParams {
                temperature: Some(0.1),
                top_p: Some(0.9),
                thinking_budget: Some(ThinkingBudget::Medium),
                ..Default::default()
            },
        },
        // Thinking high
        ParamSweep {
            label: "thinking-high".into(),
            params: GenerateParams {
                temperature: Some(0.2),
                top_p: Some(0.9),
                thinking_budget: Some(ThinkingBudget::High),
                ..Default::default()
            },
        },
        // Max tokens limited
        ParamSweep {
            label: "short-output".into(),
            params: GenerateParams {
                temperature: Some(0.1),
                max_tokens: Some(1024),
                ..Default::default()
            },
        },
        // High temp + high topP (most creative)
        ParamSweep {
            label: "very-creative".into(),
            params: GenerateParams {
                temperature: Some(1.0),
                top_p: Some(0.98),
                ..Default::default()
            },
        },
    ]
}

/// Generate a minimal quick sweep (fewer combinations for faster testing).
pub fn quick_sweep() -> Vec<ParamSweep> {
    vec![
        ParamSweep {
            label: "deterministic".into(),
            params: GenerateParams {
                temperature: Some(0.0),
                top_p: Some(0.9),
                ..Default::default()
            },
        },
        ParamSweep {
            label: "default-code".into(),
            params: GenerateParams {
                temperature: Some(0.1),
                top_p: Some(0.92),
                min_p: Some(0.03),
                ..Default::default()
            },
        },
        ParamSweep {
            label: "balanced".into(),
            params: GenerateParams {
                temperature: Some(0.3),
                top_p: Some(0.9),
                ..Default::default()
            },
        },
        ParamSweep {
            label: "thinking".into(),
            params: GenerateParams {
                temperature: Some(0.1),
                thinking_budget: Some(ThinkingBudget::Medium),
                ..Default::default()
            },
        },
    ]
}
