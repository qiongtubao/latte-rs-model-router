use std::collections::BTreeMap;

use colored::*;
use latte_ai::models::TokenUsage;

use crate::sweeper::SweepResult;

/// Trait for token-usage display across result types.
pub trait UsageDisplay {
    fn usage_string(&self) -> String;
}

impl UsageDisplay for SweepResult {
    fn usage_string(&self) -> String {
        format!("↑{} ↓{}", self.input_tokens, self.output_tokens)
    }
}

impl UsageDisplay for TokenUsage {
    fn usage_string(&self) -> String {
        let mut s = format!("↑{} ↓{}", self.input_tokens, self.output_tokens);
        if self.thinking_tokens > 0 {
            s.push_str(&format!(" t:{}", self.thinking_tokens));
        }
        s
    }
}

/// Format a single sweep test result for display.
pub fn format_result(result: &SweepResult, max_width: usize) -> String {
    let mut out = String::new();

    let prompt_header = format!(
        "[{}] {}",
        result.category.bold().cyan(),
        result.prompt_name.bold(),
    );

    let status = if result.error.is_some() {
        "ERROR".red().bold()
    } else {
        "OK".green().bold()
    };

    out.push_str(&format!(
        "  {prompt_header} — {status} — {} — {}ms\n",
        result.usage_string(),
        result.duration_ms,
    ));

    if let Some(err) = &result.error {
        out.push_str(&format!("    Error: {}\n", err.red()));
    } else {
        let preview = truncate_mid(&result.output, max_width);
        for line in preview.lines().take(10) {
            out.push_str(&format!("    │ {}\n", line.dimmed()));
        }
    }

    out
}

/// Format the full sweep report for one parameter set.
pub fn format_report(
    model_name: &str,
    param_label: &str,
    results: &[SweepResult],
    token_usage: &TokenUsage,
) -> String {
    let mut out = String::new();

    let header = format!(
        "═══ {} ─ {} ═══",
        model_name.bold().white(),
        param_label.bold().yellow(),
    );
    out.push_str(&format!("\n{}\n\n", header));

    let mut by_category: BTreeMap<String, Vec<&SweepResult>> = BTreeMap::new();
    for r in results {
        by_category.entry(r.category.clone()).or_default().push(r);
    }

    for (cat, cat_results) in &by_category {
        out.push_str(&format!("  {}:\n", cat.bold().cyan().underline()));
        for result in cat_results {
            out.push_str(&format_result(result, 600));
            out.push('\n');
        }
    }

    out.push_str(&format!(
        "{}: {} prompts, {} total\n\n",
        "Summary".bold(),
        results.len(),
        token_usage.usage_string(),
    ));

    out
}

/// Compare outputs across parameter sweeps for one prompt.
pub fn format_comparison(
    model_name: &str,
    prompt_name: &str,
    all_results: &[SweepResult],
) -> String {
    let mut out = String::new();
    let header = format!(
        "═══ {model_name} → {prompt_name} — parameter comparison ═══\n\n",
        model_name = model_name.bold().white(),
        prompt_name = prompt_name.bold().cyan(),
    );
    out.push_str(&header);

    let mut by_sweep: BTreeMap<String, Vec<&SweepResult>> = BTreeMap::new();
    for r in all_results {
        by_sweep.entry(r.sweep_label.clone()).or_default().push(r);
    }

    for (label, results) in &by_sweep {
        out.push_str(&format!("  ─ {} ─\n", label.bold().yellow()));
        for r in results {
            if let Some(err) = &r.error {
                out.push_str(&format!("    ⚠ Error: {}\n", err.red()));
            } else {
                let preview = truncate_mid(&r.output, 300);
                out.push_str(&format!("    {}ms · {}\n", r.duration_ms, r.usage_string()));
                for line in preview.lines().take(8) {
                    out.push_str(&format!("    │ {}\n", line));
                }
            }
            out.push('\n');
        }
    }

    out
}

fn truncate_mid(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let half = max / 2;
    let first_end = half.min(text.len());
    let last_start = text.len().saturating_sub(half);
    let truncated_len = text.len() - max;
    format!(
        "{}…[{} chars]…{}",
        &text[..first_end],
        truncated_len,
        &text[last_start..]
    )
}
