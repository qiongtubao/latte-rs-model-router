//! `latte_ai::vendor` 端到端示例
//!
//! 演示：
//! - 构造 `VendorRegistry`（多个 vendor）
//! - 用 [`ApiKeyProvider`] 装静态 key
//! - 用 [`discover::Manual`] 装静态 model 列表（避免打网络；想打网络用 `AnthropicModelsApi`）
//! - `get_token()` / `refresh_now()` / `status()` / `discover_models()`
//! - 用 `disable_feature()` 禁掉某些 vendor 功能
//!
//! 运行：
//! ```sh
//! ANTHROPIC_API_KEY=sk-test DEEPSEEK_API_KEY=sk-test2 cargo run --example vendor_demo
//! ```

use std::sync::Arc;
use std::time::Duration;

use latte_ai::vendor::bearer::FixedIntervalRefresher;
use latte_ai::vendor::discover::{AnthropicModelsApi, Manual};
use latte_ai::vendor::{
    ApiKeyProvider, BearerProvider, HealthCheck, ModelDescriptor, VendorConfig, VendorFeature,
    VendorId, VendorRegistry,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 初始化 tracing-subscriber（让 VendorRegistry 的 log 能输出）
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // 2. 构造 2 个 vendor：anthropic（ApiKey + 真实 Anthropic discover）+ deepseek（ApiKey + Manual discover）
    let anthropic = VendorConfig::new(
        VendorId::new("anthropic"),
        "https://api.anthropic.com",
        Arc::new(ApiKeyProvider::new(
            "anthropic",
            "${ANTHROPIC_API_KEY:-sk-demo-anthropic}",
        )),
        Arc::new(AnthropicModelsApi),
    )
    .with_health_check(HealthCheck::Http {
        path: "/v1/messages".into(),
    })
    .disable_feature(VendorFeature::ExtendedCacheTtl); // 1h cache 暂时不要

    // 演示 Bearer + 固定间隔刷新
    // refresh_fn 闭包每 1h 被调一次，返回新 token
    let bearer = BearerProvider::new(
        "deepseek",
        "initial-deepseek-token",
        Duration::from_secs(3600),
        FixedIntervalRefresher::new(Duration::from_secs(3600), || {
            Box::pin(async {
                // 真实场景调 OAuth client_credentials
                // 这里用 std::env 模拟 secret manager
                Ok(std::env::var("DEEPSEEK_API_KEY").unwrap_or_default())
            })
        }),
    );

    let deepseek = VendorConfig::new(
        VendorId::new("deepseek"),
        "https://api.deepseek.com",
        Arc::new(bearer),
        Arc::new(Manual(vec![
            ModelDescriptor::new("deepseek-chat", "DeepSeek Chat"),
            ModelDescriptor::new("deepseek-coder", "DeepSeek Coder"),
            ModelDescriptor::new("deepseek-reasoner", "DeepSeek Reasoner"),
        ])),
    );

    let registry = VendorRegistry::new(vec![anthropic, deepseek]);

    // 3. 列出所有 vendor
    println!("=== vendors ===");
    for v in registry.list() {
        println!(
            "  {:<12}  base_url={:<35}  auth={:<10}  discovery={}",
            v.id,
            v.base_url,
            v.auth.kind(),
            v.discovery.protocol()
        );
    }
    println!();

    // 4. 拿 token
    println!("=== get_token ===");
    for id in [VendorId::new("anthropic"), VendorId::new("deepseek")] {
        let token = registry.get_token(&id).await?;
        // 仅显示前 6 个字符（安全）
        let preview: String = token.chars().take(6).collect();
        println!("  {} token = {preview}...", id,);
    }
    println!();

    // 5. 状态查询
    println!("=== status ===");
    for id in [VendorId::new("anthropic"), VendorId::new("deepseek")] {
        let s = registry.status(&id).await?;
        println!(
            "  {:<12}  auth_kind={:<10}  auth_valid={}  remaining={:?}s  health_ok={}  latency={:?}ms",
            s.vendor, s.auth_kind, s.auth_valid, s.token_remaining_secs, s.health_ok, s.health_latency_ms,
        );
    }
    println!();

    // 6. Discover models（anthropic 会真打网络；deepseek 用 Manual）
    println!("=== discover_models ===");
    for id in [VendorId::new("anthropic"), VendorId::new("deepseek")] {
        match registry.discover_models(&id).await {
            Ok(models) => {
                println!("  {}: {} models", id, models.len());
                for m in models.iter().take(3) {
                    println!("    - {} ({})", m.id, m.display_name);
                }
                if models.len() > 3 {
                    println!("    ... and {} more", models.len() - 3);
                }
            }
            Err(e) => println!("  {}: error: {e}", id),
        }
    }
    println!();

    // 7. 强制 refresh
    println!("=== refresh_now ===");
    let new_token = registry.refresh_now(&VendorId::new("deepseek")).await?;
    let preview: String = new_token.chars().take(6).collect();
    println!("  deepseek refreshed token = {preview}...");
    println!();

    println!("Done. Inspect the registry via the methods shown above.");

    Ok(())
}
