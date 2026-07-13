//! Detect whether a chat-completion body carries image content.
//!
//! Used by the proxy to filter the priority pool to vision-capable models when
//! the client used `proxy-default`. Three on-wire shapes must be recognised:
//!
//! * OpenAI `/v1/chat/completions` — content parts with `type = "image_url"`.
//! * Anthropic `/v1/messages` — content blocks with `type = "image"`.
//! * Ollama `/api/chat` — legacy `images: [base64]` at message root, or
//!   newer `content` array containing `{type:"image"}` parts.
//!
//! The detector is intentionally a permissive JSON walker rather than a strict
//! per-schema parser: we only need to decide "image present / not" before
//! route selection. A false negative is recoverable (request still succeeds on
//! a text-only pool member that just refuses the image silently upstream); a
//! false positive only triggers the capability filter, which is harmless when
//! every pool member supports vision.
//!
//! 检测规则（递归遍历任意深度）：
//! * 当前对象含 `type` ∈ `{"image_url", "image", "input_image"}` → 命中
//! * 当前对象含非空 `images` 数组 → 命中（Ollama 旧格式）

/// Returns `true` when `body` contains image-bearing content under any of the
/// recognised on-wire shapes. Returns `false` for missing / malformed
/// `messages` arrays so the caller doesn't get stuck on bad payloads.
pub fn body_contains_image(body: &serde_json::Value) -> bool {
    contains_image(body)
}

/// Recursive walker. Stops at the first hit.
fn contains_image(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Object(map) => {
            // Ollama legacy: {"images": ["base64", ...]} at any message level.
            // We treat a non-empty `images` array as a hit. A bare empty array
            // would otherwise always be present on ollama messages ("images": []),
            // so we require at least one element.
            if let Some(serde_json::Value::Array(items)) = map.get("images") {
                if !items.is_empty() {
                    return true;
                }
            }

            // OpenAI / Anthropic / Ollama-new content part markers.
            if let Some(t) = map.get("type").and_then(|v| v.as_str()) {
                if matches!(t, "image" | "image_url" | "input_image") {
                    return true;
                }
            }

            // Recurse into every remaining field.
            for (_k, child) in map {
                if contains_image(child) {
                    return true;
                }
            }
            false
        }
        serde_json::Value::Array(items) => {
            items.iter().any(contains_image)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    //! 单测：覆盖三种 on-wire body 形态与阴性用例。
    //! 测试方法：构造 JSON body，调 body_contains_image，断言布尔结果。
    use super::body_contains_image;
    use serde_json::json;

    #[test]
    fn detects_openai_image_url_part() {
        let body = json!({
            "model": "x",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]
            }]
        });
        assert!(body_contains_image(&body));
    }

    #[test]
    fn detects_anthropic_image_block() {
        let body = json!({
            "model": "x",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    {"type": "text", "text": "describe"}
                ]
            }]
        });
        assert!(body_contains_image(&body));
    }

    #[test]
    fn detects_ollama_legacy_images_array() {
        let body = json!({
            "model": "x",
            "messages": [{
                "role": "user",
                "content": "what is this",
                "images": ["base64-png"]
            }]
        });
        assert!(body_contains_image(&body));
    }

    #[test]
    fn ollama_empty_images_array_does_not_trigger() {
        // Ollama messages 默认带 `images: []`；空数组不应判为图片，
        // 否则所有 ollama 请求都会被强制走 vision-capable 路径。
        let body = json!({
            "model": "x",
            "messages": [{"role": "user", "content": "hi", "images": []}]
        });
        assert!(!body_contains_image(&body));
    }

    #[test]
    fn text_only_body_returns_false() {
        let body_openai = json!({
            "model": "x",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let body_anthropic = json!({
            "model": "x",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        });
        assert!(!body_contains_image(&body_openai));
        assert!(!body_contains_image(&body_anthropic));
    }

    #[test]
    fn malformed_messages_does_not_panic() {
        let body = json!({ "model": "x" });
        assert!(!body_contains_image(&body));
    }

    #[test]
    fn input_image_marker_recognised() {
        // Responses-API-style 部分 `type: input_image`，未来兼容。
        let body = json!({
            "model": "x",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
                ]
            }]
        });
        assert!(body_contains_image(&body));
    }
}
