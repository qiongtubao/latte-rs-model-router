# latte-rs-model-router

Rust AI model 客户端库 + 参数调优工具。

## 项目结构

```
latte-rs-model-router/
├── latte-ai/          # AI client 库
│   ├── params.rs      # 生成参数类型: GenerateParams, ThinkingBudget
│   ├── models.rs      # 模型定义: Model, Message, Role, Completion
│   ├── client.rs      # 客户端: AiClient (OpenAI + Anthropic)
│   └── error.rs       # 错误类型
├── latte-tune/        # 参数调优 CLI 工具
│   ├── sweeper.rs     # 参数扫描组合
│   ├── prompts.rs     # 编程测试 prompt 集
│   └── report.rs      # 结果格式化输出
├── models.toml        # 示例模型配置文件
└── README.md
```

---

## 1. 模型配置 (Model Configuration)

在 `models.toml` 或代码中配置模型，每个模型需要以下字段：

### 必填字段

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `id` | string | 模型标识符，对应 API 的 model 参数 | `"claude-sonnet-4-20250514"` |
| `api` | string | API 类型：`"openai"` 或 `"anthropic"` | `"openai"` |
| `base_url` | string | API 端点地址 | `"https://api.deepseek.com"` |
| `api_key` | string | API 密钥，支持 `${ENV_VAR}` 环境变量 | `"${OPENAI_API_KEY}"` |

### 可选字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `name` | 同 `id` | 人类可读的名称 |
| `provider` | `"custom"` | 提供商名称（deepseek, anthropic, openai 等） |
| `context_window` | 65536 | 模型上下文窗口大小（token） |
| `max_tokens` | 4096 | 最大输出 token 数 |

### 示例

```toml
[[models]]
id = "deepseek-chat"
name = "DeepSeek Chat V3"
api = "openai"
provider = "deepseek"
base_url = "https://api.deepseek.com"
api_key = "${DEEPSEEK_API_KEY}"
context_window = 65536
max_tokens = 8192

[[models]]
id = "claude-sonnet-4-20250514"
name = "Claude Sonnet 4"
api = "anthropic"
provider = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "${ANTHROPIC_API_KEY}"
context_window = 200000
max_tokens = 8192
```

### 已知提供商配置参考

| 提供商 | API 类型 | base_url | 默认模型 |
|--------|----------|----------|----------|
| OpenAI | `openai` | `https://api.openai.com` | gpt-4o |
| Anthropic | `anthropic` | `https://api.anthropic.com` | claude-sonnet-4-20250514 |
| DeepSeek | `openai` | `https://api.deepseek.com` | deepseek-chat |
| Ollama (本地) | `openai` | `http://localhost:11434` | qwen2.5-coder:32b |
| Google (Gemini) | `openai` | `https://generativelanguage.googleapis.com/v1beta/openai/` | gemini-2.5-pro |
| Groq | `openai` | `https://api.groq.com/openai` | llama-3.3-70b |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` | 任意 |

---

## 2. 生成参数 (Generation Parameters)

所有参数都是可选的（`Option<T>`）。设为 `None` 表示使用模型提供商的默认值。

### 采样参数

#### temperature — 温度 (0.0 ~ 2.0)

**作用**: 控制输出随机性。越低越确定，越高越随机。

| 值域 | 行为 | 适用场景 |
|------|------|----------|
| 0.0 | 完全确定：总是选概率最高的 token | 代码生成、数学计算 |
| 0.1 ~ 0.3 | 轻微随机：高质量变异 | **推荐代码场景** |
| 0.5 ~ 0.7 | 中等随机 | 代码解释、设计讨论 |
| 0.8 ~ 1.0 | 高随机 | 创意写作、头脑风暴 |
| > 1.0 | 极高随机 | 实验性探索 |

**原则**: 编程任务从 `0.0` 开始，如果输出太机械/重复，逐步提高到 `0.2`。`0.1` 是代码生成的最佳起点。

#### topP — 核采样 (0.0 ~ 1.0)

**作用**: 只考虑累积概率达到 topP 的 token 集合。与 temperature 配合使用。

| 值 | 行为 |
|----|------|
| 1.0 | 考虑所有 token（等同于关闭） |
| 0.9 | 只考虑概率前 90% 的 token |
| 0.5 | 只考虑概率前 50% 的 token |

**原则**: 通常保持 `0.9 ~ 1.0`，让 temperature 做主要控制。降低 topP 可以压缩输出质量，但过度降低会减少多样性。

#### topK — 候选数量 (1 ~ 100)

**作用**: 只从概率最高的 K 个 token 中采样。更简单粗暴的过滤。

| 值 | 行为 |
|----|------|
| 1 | 完全确定（等价 temperature=0） |
| 20~40 | **编程推荐** |
| 50~100 | 更开放 |
| 不设置 | 不限制候选数 |

**原则**: 代码生成设置 `topK=40` 可以在质量和多样性之间取得平衡。较小的模型可能需要更大的 topK。

#### minP — 最小概率阈值 (0.0 ~ 1.0)

**作用**: 动态过滤低概率 token。以当前最高概率 token 为基准，排除概率低于 `minP × 最高概率` 的 token。

| 值 | 行为 |
|----|------|
| 0.0 | 不过滤 |
| 0.01~0.02 | 保守过滤，只去除极低质量 token |
| 0.03~0.05 | **推荐**：有效减少退化输出 |
| 0.1 | 激进过滤，可能影响创造力 |

**原则**: `minP` 是最有用的质量控制参数之一。设置 `0.05` 可以在不影响正常输出的情况下，大幅减少模型"胡言乱语"的概率。

### 惩罚参数

#### presencePenalty — 存在惩罚 (-2.0 ~ 2.0)

**作用**: 惩罚**已经出现过**的 token，鼓励模型谈论新话题。

| 值 | 行为 |
|----|------|
| 0.0 | 不惩罚 |
| > 0 | 惩罚已出现的 token |
| < 0 | 偏好已出现的 token |

#### frequencyPenalty — 频率惩罚 (-2.0 ~ 2.0)

**作用**: 根据 token 出现的**频率**进行惩罚，频率越高惩罚越大。

| 值 | 行为 |
|----|------|
| 0.0 | 不惩罚 |
| 0.1~0.3 | 轻微减少重复 |
| 0.5+ | 强去重 |

**原则**: 编程任务通常保持 `0.0`。如果模型输出有重复模式（如重复解释、重复代码注释），可以设 `frequencyPenalty=0.1~0.2`。

#### repetitionPenalty — 重复惩罚

**作用**: 通用的重复惩罚系数。`1.0` = 无惩罚，`> 1.0` = 惩罚重复。

**原则**: 较少使用。`frequencyPenalty` 更精确。只有在对重复特别敏感的场景才用。

### 输出控制

#### maxTokens — 最大输出长度

**作用**: 限制模型生成的 token 数量。

| 场景 | 推荐值 |
|------|--------|
| 短回答 / 代码片段 | 512~1024 |
| 函数/方法级代码 | 2048~4096 |
| 完整文件 / 复杂分析 | 4096~8192 |
| 长篇文档 / 架构设计 | 8192+ |

**原则**: 设得太大浪费 token（以及费用），太小输出会被截断。`stop_reason` 中的 `"length"` 表示被 max_tokens 截断。

#### stopSequences — 停止序列

**作用**: 模型遇到这些字符串时停止生成。

**原则**: 一般不需要设置。可以在需要精确控制输出边界时使用（如只生成一个函数定义）。

### 推理参数

#### thinkingBudget — 思考预算

**作用**: 控制模型在回答前进行"思考"的 token 预算。对复杂推理任务显著提升质量。

| 级别 | Token | 适用场景 |
|------|-------|----------|
| `Minimal` | 1024 | 简单格式化、翻译 |
| `Low` | 2048 | 基础问答、简单代码 |
| `Medium` | 8192 | **推荐**：复杂代码生成、Debug |
| `High` | 16384 | 架构设计、复杂推理 |
| `XHigh` | 32768 | 长篇分析、数学证明 |
| `Custom(n)` | 自定义 | 精确控制 |

**原则**: 
- 只需要 Anthropic 模型（Claude Sonnet/Opus）在 `api = "anthropic"` 时支持
- OpenAI 兼容的模型通过 `reasoning_effort` 支持类似功能，但不是所有模型都支持
- 思考不是免费的——它会消耗输出 token 预算。`thinkingBudget` 是**额外**的思考 token，不计入最终输出

#### seed — 随机种子

**作用**: 设置后，在相同参数和输入下，模型应产生相同的输出（**部分**模型支持）。

**原则**: 用于测试和回归验证。不是所有模型/提供商都保证确定性输出。

---

## 3. 最佳参数配置

### 按场景推荐

#### 代码生成 (Code Generation)

```rust
GenerateParams {
    temperature: Some(0.1),        // 低温度确保确定性
    top_p: Some(0.9),              // 标准核采样
    top_k: Some(40),               // 限制候选
    min_p: Some(0.05),             // 过滤低质量 token
    presence_penalty: None,        // 不惩罚
    frequency_penalty: None,       // 不惩罚
    max_tokens: Some(4096),        // 足够输出完整函数
    thinking_budget: None,         // 简单代码不需要思考
}
```

#### 复杂 Debug / 架构设计

```rust
GenerateParams {
    temperature: Some(0.2),        // 稍微提高允许一些探索
    top_p: Some(0.9),
    min_p: Some(0.02),             // 降低过滤保留更多可能性
    max_tokens: Some(8192),
    thinking_budget: Some(ThinkingBudget::Medium),  // 需要推理
    // ...其余默认
}
```

#### 创造性 / 头脑风暴

```rust
GenerateParams {
    temperature: Some(0.7),        // 高温度鼓励多样性
    top_p: Some(0.95),             // 更大的采样池
    presence_penalty: Some(0.1),   // 轻微鼓励新话题
    max_tokens: Some(2048),
    // ...其余默认
}
```

#### 确定性输出 (测试/生产)

```rust
GenerateParams {
    temperature: Some(0.0),        // 完全确定
    top_p: Some(0.9),
    min_p: Some(0.05),
    seed: Some(42),                // 固定种子
    // ...其余默认
}
```

### 参数调整优先级

调优时应按以下顺序调整参数：

1. **temperature** — 影响最大，最先调
2. **minP** — 质量过滤器，几乎总是有益
3. **thinkingBudget** — 复杂任务显著提升质量
4. **topK** — 给 temperature 锦上添花
5. **topP** — 除非有特殊需求，否则保持 0.9
6. **惩罚参数** — 只在看到重复问题时才调

### 常见陷阱

| 陷阱 | 说明 |
|------|------|
| 同时大幅调整多个参数 | 难以判断哪个参数的效果。一次只改一个。 |
| temperature=0 还设 topP<1 | temperature=0 时 topP/topK 不生效。 |
| 过度惩罚重复 | 会让输出不连贯、漏掉必要的重复（如代码结构）。 |
| minP 设太高 | 会过滤掉正常 token，导致输出质量下降。 |
| thinkingBudget 设太大 | 模型可能过度思考简单问题，浪费 token 和时间。 |

---

## 4. 使用 latte-tune 调优

`latte-tune` 可以自动遍历参数组合，让你直观对比不同参数下的输出质量。

### 基本用法

```bash
# 先看有哪些测试 prompt
./target/release/latte-tune list-prompts

# 快速扫一遍（4种参数组合 x 8个prompt）
LATTE_API_KEY="sk-..." ./target/release/latte-tune sweep deepseek-chat \
  --api openai \
  --base-url https://api.deepseek.com \
  --quick

# 完整扫描（10种参数组合）
ANTHROPIC_API_KEY="sk-..." ./target/release/latte-tune sweep claude-sonnet-4-20250514 \
  --api anthropic \
  --base-url https://api.anthropic.com

# 用配置文件扫描多个模型
ANTHROPIC_API_KEY="sk-..." DEEPSEEK_API_KEY="sk-..." \
  ./target/release/latte-tune sweep --config models.toml --quick

# 只测特定 prompt
./target/release/latte-tune sweep deepseek-chat --prompt-filter "rust-parse"

# 只测特定 sweep
./target/release/latte-tune sweep deepseek-chat --sweep-filter "conservative"

# 对比单个 prompt 在所有参数下的输出
./target/release/latte-tune compare deepseek-chat --prompt rust-parse-json
```

### 扫描参数组合

**快速扫描** (4 种):

| 标签 | temperature | topP | minP | thinking |
|------|------------|------|------|----------|
| deterministic | 0.0 | 0.9 | - | - |
| default-code | 0.1 | 0.92 | 0.03 | - |
| balanced | 0.3 | 0.9 | - | - |
| thinking | 0.1 | - | - | Medium |

**完整扫描** (10 种): 包含 conservative, low-temp, balanced, creative, no-minp, anti-repetition, thinking-medium, thinking-high, short-output, very-creative。

### 如何判断 "最优"

1. **看输出质量**: 代码是否完整？语法正确？风格一致？
2. **看 token 用量**: 相同质量下 token 越少越好
3. **看响应速度**: 延迟是否可接受？
4. **看一致性**: 重复跑几次，输出是否稳定？

没有通用的"最优参数"——同一个模型在不同任务上的最佳参数可能不同。`latte-tune` 帮你直观对比，找到最适合**你的使用场景**的参数。

---

## 5. 代码中使用

```rust
use latte_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let model = Model {
        id: "deepseek-chat".into(),
        name: "DeepSeek Chat".into(),
        api: ApiType::OpenAiCompletions,
        provider: "deepseek".into(),
        base_url: "https://api.deepseek.com".into(),
        api_key: std::env::var("DEEPSEEK_API_KEY").unwrap_or_default(),
        context_window: 65536,
        max_tokens: 8192,
        supports_thinking: false,
        cost_per_million_input: 0.27,
        cost_per_million_output: 1.10,
    };

    let client = AiClient::new(model)?;

    // 使用推荐参数
    let params = GenerateParams {
        temperature: Some(0.1),
        top_p: Some(0.9),
        min_p: Some(0.05),
        ..Default::default()
    };

    let completion = client.chat(&[
        Message {
            role: Role::User,
            content: "Write a Rust function to sum a Vec".into(),
        },
    ], &params).await?;

    println!("{}", completion.content);
    println!("Usage: {} input, {} output tokens",
        completion.usage.input_tokens,
        completion.usage.output_tokens);

    Ok(())
}
```
