# latte-rs-model-router

Rust AI model 客户端库 + 参数调优工具。

## 项目结构

```
latte-rs-model-router/
├── latte-ai/                  # AI client 库（OpenAI + Anthropic 协议 + 流式 + thinking）
│   ├── params.rs              # 生成参数类型: GenerateParams, ThinkingBudget
│   ├── models.rs              # 类型: Model / Message / Role / Completion / ApiType / TokenUsage
│   ├── client.rs              # 客户端: AiClient（chat + chat_stream）
│   ├── error.rs               # AiError + Result
│   └── examples/              # 示例: basic, streaming
├── latte-router/              # 模型 catalog + 优先级选择 + 断路器
│   ├── config.rs              # ModelEntry（per-model 冷却配置）+ Route + RouterError
│   ├── clock.rs               # Clock trait + SystemClock
│   ├── catalog.rs             # ModelCatalog 加载 ~/.latte/models.d/*.toml + ./.latte/models.d
│   ├── breaker.rs             # CircuitBreaker：429 调度冷却 + 5xx 阈值熔断
│   ├── selector.rs            # 选路：按 pool 顺序，跳过冷却中
│   ├── proxy_config.rs        # proxy.toml schema (server + catalog)
│   ├── lib.rs                 # Router 串起来
│   └── tests/                 # 15 个验收用例
├── latte-tune/                # 参数调优 + 单 model 聊天 CLI
│   ├── sweeper.rs             # 参数扫描组合
│   ├── prompts.rs             # 编程测试 prompt 集
│   ├── report.rs              # 结果格式化输出
│   └── main.rs                # CLI 入口：sweep / list-models / list-prompts / compare / chat
├── latte-model-proxy/         # HTTP 代理（Ollama / OpenAI / Anthropic 兼容）
│   ├── cli.rs                 # clap 参数定义（--config / --host / --port / --models / --models-dir）
│   ├── server.rs              # axum 路由：/api/{tags,show,chat}、/v1/{models,chat/completions}、/v1/messages
│   ├── main.rs                # 入口：latte-model-proxy --models=id1,id2 --port=6666
│   └── tests/                 # 20 个集成测试（wiremock 模拟下游）
└── README.md
```

---

## 编译与安装 (Build & Install)

### 前置条件

- Rust 工具链 (1.80+): [rustup.rs](https://rustup.rs)
- Cargo (随 Rust 安装)

### 编译

```bash
# Debug 构建（开发调试用）
cargo build

# Release 构建（推荐日常使用，性能好很多）
cargo build --release
```

编译产物：
- `target/debug/latte-tune` — debug 版 CLI
- `target/release/latte-tune` — release 版 CLI
- `target/debug/liblatte_ai.rlib` / `target/release/liblatte_ai.rlib` — 库文件

### 安装到系统

```bash
# 安装 latte-tune CLI 到 ~/.cargo/bin/
cargo install --path latte-tune

# 之后可以直接运行
latte-tune --help
```

### 快速验证

```bash
# 查看帮助
cargo run -- --help
cargo run -- sweep --help
cargo run -- compare --help

# 列出所有测试 prompt（不需要 API key）
cargo run -- list-prompts
```

---

## 1. 模型配置 (Model Configuration)

模型配置文件使用 **YAML** 格式（推荐），同时向后兼容 **TOML** 格式。
自动检测文件扩展名：`.yaml` / `.yml` → YAML，`.toml` → TOML。

### 1.1 配置文件结构

```yaml
# models.yaml
models:
  - id: "deepseek-chat"
    api: "openai"
    base_url: "https://api.deepseek.com"
    api_key: "${DEEPSEEK_API_KEY}"

  - id: "claude-sonnet-4-20250514"
    api: "anthropic"
    base_url: "https://api.anthropic.com"
    api_key: "${ANTHROPIC_API_KEY}"
```

顶层为 `models` 数组，每个元素定义一个模型。

### 1.2 字段参考

#### 必填字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | `string` | 模型标识符，对应 API 的 `model` 参数。例: `"deepseek-chat"`, `"claude-sonnet-4-20250514"` |
| `api` | `string` | API 协议类型。`"openai"` — OpenAI Chat Completions 兼容 API（支持 DeepSeek、Ollama、Groq 等）；`"anthropic"` — Anthropic Messages API |
| `base_url` | `string` | API 端点地址。支持 `${ENV_VAR}` 环境变量展开。例: `"https://api.deepseek.com"`, `"${CUSTOM_BASE_URL}"` |
| `api_key` | `string` | API 认证密钥。支持 `${ENV_VAR}` 展开。例: `"${DEEPSEEK_API_KEY}"`。设为 `"ollama"` 等占位值可跳过认证（本地部署） |

#### 基本信息

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `string` | 同 `id` | 人类可读的模型名称，用于报告和日志显示。例: `"DeepSeek Chat V3"` |
| `description` | `string` | — | 模型描述说明，仅作文档用。例: `"通用对话模型，性价比极高"` |
| `provider` | `string` | `"custom"` | 提供商标识。用于分组和筛选。例: `"deepseek"`, `"anthropic"`, `"openai"`, `"ollama"` |

#### 容量参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `context_window` | `u32` | `65536` (64K) | 模型上下文窗口大小（token 数）。实际使用由模型决定。例: Claude `200000`, GPT-4o `128000`, DeepSeek `65536` |
| `max_tokens` | `u32` | `4096` | 单次请求最大输出 token 数。请求中 `max_tokens` 参数的上限。例: `8192`, `16384` |

#### 推理能力

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `reasoning` | `bool` | `true` (anthropic API) / `false` (openai API) | 模型是否支持思考/推理（thinking/reasoning）。设为 `true` 后，`ThinkingBudget` 参数生效。Anthropic 原生支持；OpenAI 兼容 API 中，`reasoning_effort` 参数仅部分模型支持（如 DeepSeek R1、o1 系列） |

#### 成本（USD / 百万 token）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cost_per_million_input` | `f64` | `0.0` | 每百万输入 token 成本（美元）。用于费用估算和对比。例: DeepSeek `0.27`, GPT-4o `2.50`, Claude Opus `15.0` |
| `cost_per_million_output` | `f64` | `0.0` | 每百万输出 token 成本（美元）。注意输出通常比输入贵 3~5 倍。例: DeepSeek `1.10`, GPT-4o `10.0`, Claude Opus `75.0` |

### 1.3 完整示例

```yaml
models:
  - id: "deepseek-chat"
    name: "DeepSeek Chat V3"
    description: "通用对话模型，性价比极高"
    api: "openai"
    provider: "deepseek"
    base_url: "https://api.deepseek.com"
    api_key: "${DEEPSEEK_API_KEY}"
    context_window: 65536
    max_tokens: 8192
    reasoning: false
    cost_per_million_input: 0.27
    cost_per_million_output: 1.10

  - id: "claude-sonnet-4-20250514"
    name: "Claude Sonnet 4"
    description: "Anthropic 中端模型，适合代码生成与日常对话"
    api: "anthropic"
    provider: "anthropic"
    base_url: "https://api.anthropic.com"
    api_key: "${ANTHROPIC_API_KEY}"
    context_window: 200000
    max_tokens: 8192
    reasoning: true
    cost_per_million_input: 3.0
    cost_per_million_output: 15.0

  # 本地 Ollama 模型
  - id: "qwen2.5-coder-32b-instruct"
    name: "Qwen 2.5 Coder 32B"
    api: "openai"
    provider: "ollama"
    base_url: "http://localhost:11434"
    api_key: "ollama"
    context_window: 32768
    max_tokens: 4096
```

完整配置示例见仓库根目录的 `models.yaml`。

### 1.4 TOML 格式 (向后兼容)

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
```

字段与 YAML 完全相同，`[[models]]` 对应 YAML 的 `models:` 数组。

### 1.5 已知提供商配置参考

| 提供商 | api | base_url | 默认模型 | reasoning |
|--------|-----|----------|----------|-----------|
| OpenAI | `openai` | `https://api.openai.com` | gpt-4o | 部分模型 |
| Anthropic | `anthropic` | `https://api.anthropic.com` | claude-sonnet-4-20250514 | ✅ |
| DeepSeek | `openai` | `https://api.deepseek.com` | deepseek-chat | R1 系列 |
| Ollama (本地) | `openai` | `http://localhost:11434` | 取决于部署 | ❌ |
| Google (Gemini) | `openai` | `https://generativelanguage.googleapis.com/v1beta/openai/` | gemini-2.5-pro | ❌ |
| Groq | `openai` | `https://api.groq.com/openai` | llama-3.3-70b | ❌ |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` | 任意 | 取决于路由模型 |
### 1.6 配置文件自动发现

`latte-tune` 在不指定 `--config` 时，按以下优先级自动查找配置文件：

| 优先级 | 路径 | 格式 | 说明 |
|--------|------|------|------|
| 1 | `./latte.yaml` / `./latte.yml` / `./latte.json` | YAML/JSON | 项目本地配置（推荐） |
| 2 | `./models.yaml` / `./models.yml` / `./models.json` | YAML/JSON | 项目本地配置 |
| 3 | `./latte.toml` / `./models.toml` | TOML | 项目本地配置（兼容旧版） |
| 4 | `~/.latte/models.yaml` / `~/.latte/models.json` | YAML/JSON | 用户全局配置（dot-dir，推荐） |
| 5 | `~/.config/latte/models.yaml` / `~/.config/latte/models.json` | YAML/JSON | 用户全局配置（XDG 标准） |
| 6 | `~/.latte/models.toml` / `~/.config/latte/models.toml` | TOML | 用户全局配置（兼容旧版） |

找到第一个存在的文件即停止。都找不到时才使用命令行参数指定的单个模型。

**全局配置** — 推荐放到 `~/.latte/models.yaml`（简洁）或 `~/.config/latte/models.yaml`（XDG 标准）：

```bash
mkdir -p ~/.latte
cp models.yaml ~/.latte/models.yaml
# 编辑填入 API key 后，在任何目录直接运行
latte-tune sweep --quick
```

也支持 JSON 格式：`~/.latte/models.json`。

**项目配置** — 项目根目录创建 `latte.yaml`，团队成员共用（密钥用环境变量保护）：

```yaml
# latte.yaml
models:
  - id: "deepseek-chat"
    api: "openai"
    provider: "deepseek"
    base_url: "https://api.deepseek.com"
    api_key: "${DEEPSEEK_API_KEY}"   # 每人设自己的环境变量
```

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

## 4. CLI 使用指南

`latte-tune` 是参数调优 CLI 工具，支持三个子命令。

### 4.1 命令概览

```
latte-tune <COMMAND>

Commands:
  chat          与模型对话（一键直连）
  sweep         对模型运行参数扫描
  compare       对单个 prompt 对比所有参数组合
  list-prompts  列出所有测试 prompt
  help          查看帮助
```

### 4.2 chat — 一键对话 & 交互 REPL

默认进入交互式 REPL（循环对话），也可以单次发送。

**交互模式**（直接运行，维护对话历史）：

```bash
latte-tune chat
# 进入 REPL，可多轮对话：
#   ▶ 用 Rust 写一个二分查找
#   ◀ (AI 回复...)
#   ▶ 给这个函数加上单元测试
#   ◀ (AI 回复...)
#   ▶ /clear   清空历史
#   ▶ /model deepseek  切换模型
#   ▶ /exit    退出
```

**单次模式**：

```bash
# 命令行传 prompt
latte-tune chat "用 Rust 写一个二分查找"

# 选模型
latte-tune chat -m claude "解释 Rust 所有权"

# 管道输入
cat src/main.rs | latte-tune chat "Review this code:"

# 显式进入 REPL（即使有 stdin）
latte-tune chat -r
```

> 运行 `latte-tune chat --help` 查看完整参数。

REPL 内建命令：
| 命令 | 作用 |
|------|------|
| `/exit` `/quit` `/q` | 退出 |
| `/clear` `/c` | 清空对话历史 |
| `/model <id>` | 切换模型 |
| `/help` `/h` | 帮助 |

### 4.3 环境变量


| 变量 | 用途 | 必填 |
|------|------|------|
| `LATTE_API_KEY` | CLI 直接指定模型时的默认 API key | 否（也可用 `--api-key`） |
| `ANTHROPIC_API_KEY` | Anthropic 模型 API key（配置文件 `${ANTHROPIC_API_KEY}` 展开） | 按需 |
| `DEEPSEEK_API_KEY` | DeepSeek 模型 API key（配置文件 `${DEEPSEEK_API_KEY}` 展开） | 按需 |
| `OPENAI_API_KEY` | OpenAI 模型 API key（配置文件 `${OPENAI_API_KEY}` 展开） | 按需 |
| `RUST_LOG` | 日志级别: `error`, `warn`, `info`, `debug`, `trace`（默认 `error`） | 否 |

### 4.4 sweep — 参数扫描

对模型运行多组参数组合，每组参数在所有测试 prompt 上执行，输出对比结果。

```bash
# 完整命令签名
latte-tune sweep [OPTIONS] [MODEL]

# 参数说明
#   [MODEL]               模型 ID（可选，使用 --config 或全局配置时不需要）
#   --api <API>           API 类型: openai 或 anthropic [default: openai]
#   --base-url <URL>      自定义 API 端点
#   --api-key <KEY>       API key（也可设 LATTE_API_KEY 环境变量）
#   --provider <NAME>     提供商名称 [default: custom]
#   --max-tokens <N>      最大输出 token 数 [default: 4096]
#   --context-window <N>  上下文窗口大小 [default: 65536]
#   --quick               快速扫描（4 种组合，代替默认的 10 种）
#   --prompt-filter <S>   只运行名称包含 S 的 prompt
#   --sweep-filter <S>    只运行标签包含 S 的参数组合
#   --config <PATH>       从配置文件加载模型 (YAML / JSON / TOML，此时 MODEL 参数被忽略)
```

#### 使用模式

**模式一：命令行直接指定模型**

```bash
# 快速扫描 DeepSeek
LATTE_API_KEY="sk-..." latte-tune sweep deepseek-chat \
  --api openai \
  --base-url https://api.deepseek.com \
  --quick

# 完整扫描 Claude
ANTHROPIC_API_KEY="sk-..." latte-tune sweep claude-sonnet-4-20250514 \
  --api anthropic \
  --base-url https://api.anthropic.com
```

**模式二：从配置文件加载模型**
```bash
# 一次性扫描 models.yaml 中所有模型
ANTHROPIC_API_KEY="sk-..." DEEPSEEK_API_KEY="sk-..." \
  latte-tune sweep --config models.yaml --quick
```

当使用 `--config` 时，配置文件中每个 `[[models]]` 条目都会被依次测试。

**过滤**

```bash
# 只测特定 prompt
latte-tune sweep deepseek-chat --prompt-filter "rust-parse"

# 只测特定参数组合
latte-tune sweep deepseek-chat --sweep-filter "thinking"

# 组合使用
latte-tune sweep deepseek-chat --prompt-filter "debug" --sweep-filter "low-temp" --quick
```

### 4.5 list-prompts — 列出测试 prompt

不需要 API key，纯本地操作。

```bash
latte-tune list-prompts
```

输出示例：
```
Available test prompts:
  rust-parse-json  [code-gen]
    Checks code structure, error handling, and idiomatic Rust
  debug-memory-leak  [debug]
    Checks ability to identify memory issues and propose fixes
  ...
Total: 8 prompts
```

Prompt 分类：`code-gen`、`debug`、`refactor`、`explain`、`architecture`。

### 4.6 compare — 单 prompt 对比

对**一个** prompt 运行所有参数组合，并列显示各组合的输出，方便横向对比。

```bash
latte-tune compare [OPTIONS] --prompt <PROMPT> <MODEL>

# 示例
latte-tune compare deepseek-chat --prompt rust-parse-json
```

与 `sweep` 的区别：`sweep` 跑所有 prompt × 所有参数组合；`compare` 跑一个 prompt × 所有参数组合，输出更适合 A/B 对比。

### 4.7 扫描参数组合详情

**快速扫描** (4 种，`--quick`):

| 标签 | temperature | topP | minP | thinking |
|------|------------|------|------|----------|
| deterministic | 0.0 | 0.9 | - | - |
| default-code | 0.1 | 0.92 | 0.03 | - |
| balanced | 0.3 | 0.9 | - | - |
| thinking | 0.1 | - | - | Medium |

**完整扫描** (10 种，默认):

| 标签 | temperature | topP | minP | topK | thinking | 其他 |
|------|------------|------|------|------|----------|------|
| conservative | 0.0 | 0.9 | - | - | - | - |
| low-temp | 0.1 | 0.92 | 0.03 | - | - | - |
| balanced | 0.3 | 0.9 | - | - | - | - |
| creative | 0.7 | 0.95 | - | - | - | presence_penalty=0.1 |
| no-minp | 0.3 | 0.9 | - | - | - | - |
| anti-repetition | 0.3 | 0.9 | - | - | - | frequency_penalty=0.3 |
| thinking-medium | 0.1 | - | - | - | Medium | - |
| thinking-high | 0.2 | - | - | - | High | - |
| short-output | 0.3 | 0.9 | - | - | - | max_tokens=512 |
| very-creative | 0.9 | 0.99 | - | 80 | - | - |

### 4.8 结果解读

1. **看输出质量**: 代码是否完整？语法正确？风格一致？
2. **看 token 用量**: 相同质量下 token 越少越好（`↑N ↓M`：N 是输入 token，M 是输出 token）
3. **看响应速度**: 延迟是否可接受？
4. **看一致性**: 重复跑几次，输出是否稳定？

没有通用的"最优参数"——同一个模型在不同任务上的最佳参数可能不同。`latte-tune` 帮你直观对比，找到最适合**你的使用场景**的参数。

---

## 5. 代码中使用

`latte-ai` 提供了完整的 Rust API。所有示例位于 `latte-ai/examples/`，可直接运行：

```bash
# 基础对话
DEEPSEEK_API_KEY="sk-..." cargo run --example basic

# 流式对话
DEEPSEEK_API_KEY="sk-..." cargo run --example streaming
# 从配置文件加载模型
DEEPSEEK_API_KEY="sk-..." cargo run --example config_file -- models.yaml
```

### 5.1 基本用法

```rust
use latte_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // ── 构建模型 ──────────────────────────────────────
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

    // ── 三种传参方式 ──────────────────────────────────

    // 1. 便捷预设
    let params = GenerateParams::code_defaults();

    // 2. 手动指定
    let params = GenerateParams {
        temperature: Some(0.1),
        top_p: Some(0.9),
        min_p: Some(0.05),
        max_tokens: Some(4096),
        ..Default::default()
    };

    // 3. 使用默认值（全部 None，由模型提供商决定）
    let params = GenerateParams::default();

    // ── 发送请求 ──────────────────────────────────────

    // 简单对话
    let completion = client.chat(&[
        Message { role: Role::User, content: "用 Rust 写一个求和函数".into() },
    ], &params).await?;
    println!("{}", completion.content);
    println!("用量: {}", completion.usage);

    // 带系统提示的多轮对话
    let completion = client.chat(&[
        Message { role: Role::System, content: "你是 Rust 专家，代码简洁，用英文命名。".into() },
        Message { role: Role::User, content: "写一个二分查找".into() },
    ], &params).await?;

    Ok(())
}
```

### 5.2 流式输出

```rust
use latte_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = AiClient::new(model)?;

    let mut stream = client.chat_stream(
        &[Message { role: Role::User, content: "讲解 Rust 所有权".into() }],
        &GenerateParams::default(),
    ).await?;

    while let Some(event) = stream.recv().await {
        match event {
            StreamEvent::Delta { content, .. } => print!("{}", content),
            StreamEvent::Done { usage, .. } => println!("\n用量: {}", usage),
            StreamEvent::Error(e) => eprintln!("错误: {}", e),
        }
    }

    Ok(())
}
```

### 5.3 从配置文件加载模型

YAML 或 TOML 配置文件可以在代码中加载，自动按扩展名检测格式：

```rust
use std::path::Path;
use latte_ai::prelude::*;

fn load_models(path: &str) -> anyhow::Result<Vec<Model>> {
    let content = std::fs::read_to_string(path)?;
    let ext = Path::new(path).extension().and_then(|e| e.to_str());

    #[derive(serde::Deserialize)]
    struct Config { models: Vec<ModelEntry> }

    #[derive(serde::Deserialize)]
    struct ModelEntry {
        id: String,
        api: String,
        base_url: String,
        name: Option<String>,
        provider: Option<String>,
        api_key: Option<String>,
        context_window: Option<u32>,
        max_tokens: Option<u32>,
        reasoning: Option<bool>,
        cost_per_million_input: Option<f64>,
        cost_per_million_output: Option<f64>,
    }

    // 按扩展名自动检测格式
    let cfg: Config = match ext {
        Some("yaml" | "yml") => serde_yaml::from_str(&content)?,
        _ => toml::from_str(&content)?,
    };

    Ok(cfg.models.iter().map(|e| {
        let api = match e.api.as_str() {
            "anthropic" | "anthropic-messages" => ApiType::AnthropicMessages,
            _ => ApiType::OpenAiCompletions,
        };
        Model {
            id: e.id.clone(),
            name: e.name.clone().unwrap_or_else(|| e.id.clone()),
            api,
            provider: e.provider.clone().unwrap_or_else(|| "custom".into()),
            base_url: e.base_url.clone(),
            api_key: resolve_env(&e.api_key.clone().unwrap_or_default()),
            context_window: e.context_window.unwrap_or(65536),
            max_tokens: e.max_tokens.unwrap_or(4096),
            supports_thinking: e.reasoning.unwrap_or(false)
                || matches!(api, ApiType::AnthropicMessages),
            cost_per_million_input: e.cost_per_million_input.unwrap_or(0.0),
            cost_per_million_output: e.cost_per_million_output.unwrap_or(0.0),
        }
    }).collect())
}

fn resolve_env(value: &str) -> String {
    if value.starts_with("${") && value.ends_with('}') {
        let var = &value[2..value.len() - 1];
        std::env::var(var).unwrap_or_default()
    } else {
        value.to_string()
    }
}
```

完整示例见 `latte-ai/examples/config_file.rs`。

### 5.4 可用预设参数

```rust
GenerateParams::code_defaults()      // 代码生成: t=0.1, p=0.9, k=40, mp=0.05, mt=4096
GenerateParams::analysis_defaults()  // 分析/Debug: t=0.2, p=0.9, mp=0.02, mt=8192, thinking=Medium
GenerateParams::creative_defaults()  // 创意写作: t=0.8, p=0.95, pp=0.1, fp=0.1, mt=4096
GenerateParams::default()            // 全部 None，使用模型提供商默认值
```

---

## 6. 路由 + 弹性（`latte-router`）

`latte-router` 提供静态模型 catalog、按优先级选路、断路器（429 调度 + 5xx 阈值）—— 与 `latte-ai` 的 wire 协议解耦。`latte-model-proxy` 的 HTTP 层只做翻译。

### 6.1 proxy 配置文件

proxy 的所有行为在 `proxy.toml` 里（canonical / 最全参数），CLI flags 只覆盖其中字段。

```toml
# proxy.toml
[server]
host = "127.0.0.1"
port = 11434

[catalog]
models_dir = "~/.latte/models.d"        # 支持 ~ 展开
models = ["claude-sonnet-4-20250514", "deepseek-v4-flash"]  # 数组顺序 = 权重从大到小
```

字段说明：

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `server.host` | string | `127.0.0.1` | 绑定地址 |
| `server.port` | u16 | `11434` | 监听端口 |
| `catalog.models_dir` | string | `~/.latte/models.d` | 模型定义目录；启动时同时找 `./.latte/models.d` 做项目级覆盖 |
| `catalog.models` | array | `[]` | 暴露的 model id 列表；**数组顺序 = 优先级**（rank 0 首选，降级到 rank 1） |

CLI flags（覆盖 proxy.toml）：

| flag | 覆盖 |
|---|---|
| `--config <path>` | proxy.toml 路径 |
| `--host <X>` | `server.host` |
| `--port <N>` | `server.port` |
| `--models id1,id2` | `catalog.models` |
| `--models-dir <path>` | `catalog.models_dir` |

### 6.2 模型定义（`models.d/*.toml`）

每个文件可放一个或多个 `[[models]]`。冷却参数 per-model 写在每个 model 下。

```toml
# ~/.latte/models.d/anthropic.toml
[[models]]
id = "claude-sonnet-4-20250514"
name = "Claude Sonnet 4"
api = "anthropic"                # 也接受 "anthropic-messages"
provider = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "${ANTHROPIC_API_KEY}"  # ${ENV} 加载时展开
context_window = 200000
max_tokens = 8192

# 429 调度冷却（per-model）
rate_limit_refresh_anchor        = "2026-06-29T00:00:00Z"  # 调度起点
rate_limit_refresh_interval_secs = 18000                    # 调度间隔（5h）

# 5xx 熔断（per-model）
retry_count_5xx   = 5
cooldown_5xx_secs = 600   # 默认 10 min
```

```toml
# ~/.latte/models.d/deepseek.toml
[[models]]
id = "deepseek-v4-flash"
api = "openai"                    # 也接受 "openai-completions"
provider = "deepseek"
base_url = "https://api.deepseek.com"
api_key = "${DEEPSEEK_API_KEY}"

# 1 分钟限流窗口
rate_limit_refresh_anchor        = "2026-06-29T00:00:00Z"
rate_limit_refresh_interval_secs = 60

retry_count_5xx   = 5
cooldown_5xx_secs = 600
```

`[[models]]` 字段表：

| 字段 | 类型 | 必填 | 默认 | 含义 |
|---|---|---|---|---|
| `id` | string | 是 | — | 模型唯一标识 |
| `name` | string | 否 | = `id` | 人类可读名（展示用） |
| `api` | string | 是 | — | 协议：`"anthropic"` / `"openai"`（也接受长写 `"anthropic-messages"` / `"openai-completions"`） |
| `provider` | string | 是 | — | 厂商标识（分组/筛选用） |
| `base_url` | string | 是 | — | 上游 API 根 URL |
| `api_key` | string | 是 | — | 认证密钥；`${ENV}` 形式加载时展开 |
| `context_window` | u32 | 否 | `65536` | 输入 token 上限（信息性） |
| `max_tokens` | u32 | 否 | `4096` | 输出 token 上限（信息性） |
| `rate_limit_refresh_anchor` | string (ISO 8601 UTC) | 否 | `1970-01-01T00:00:00Z` | 429 调度起点 |
| `rate_limit_refresh_interval_secs` | u64 | 否 | `60` | 429 调度间隔 |
| `retry_count_5xx` | u32 | 否 | `5` | 5xx 连续失败 N 次 → 熔断 |
| `cooldown_5xx_secs` | u64 | 否 | `600` | 5xx 熔断后冷却秒数（默认 10 min） |

### 6.3 限流与熔断

两个机制并存：

1. **HTTP 429** → 拉出该 model 直到下个 refresh 时间点 `max(配置算的 refresh, now + Retry-After)`
2. **HTTP 5xx** → 连续 N 次 → 拉出该 model N 秒（默认 600s = 10 min）

成功响应（2xx / 4xx 非 429）清零 5xx 计数器。冷却中的 model 不参与选路，自动降级到 pool 下一个。

全部冷却中 → 返 `RouterError::AllUnavailable { retry_after_secs }`，proxy 映射为 `503 + Retry-After`。

### 6.4 启动示例

```bash
# 使用默认 proxy.toml + 默认 models.d/
latte-model-proxy

# 显式指定
latte-model-proxy \
  --config ~/.latte/proxy.toml \
  --models=claude-sonnet-4-20250514,deepseek-v4-flash \
  --port=6666
```

省略 `--config`：自动找 `./proxy.toml` → `~/.latte/proxy.toml`。省略 `--models`：用 `proxy.toml` 的 `catalog.models`。都没给 → 启动失败。

### 6.5 程序化使用

```rust
use latte_router::{ModelEntry, ModelCatalog, Router, SystemClock};
use std::sync::Arc;

// 1. 直接构造（不读文件）
let pool = vec![/* ModelEntry ... */];
let router = Arc::new(Router::with_system_clock(pool));

// 2. 选路
match router.select("claude-sonnet-4-20250514") {
    Ok(route) => { /* 转发到 route.base_url，用 route.api_key */ }
    Err(e) => eprintln!("{e}"),
}

// 3. 喂响应信号
router.record("claude-sonnet-4-20250514", 200, None);  // 成功
router.record("claude-sonnet-4-20250514", 429, Some(60));  // 429 + Retry-After
router.record("claude-sonnet-4-20250514", 500, None);  // 5xx

// 4. 加载目录
let mut catalog = ModelCatalog::new();
catalog.load_dir("~/.latte/models.d")?;
```
## 7. 开发与测试

```bash
# 全量测试（40 个）
cargo test --workspace

# 单 crate
cargo test -p latte-ai
cargo test -p latte-router
cargo test -p latte-model-proxy
cargo test -p latte-tune

# 编译验证
cargo build --workspace --release
```

## 8. HTTP 代理服务（`latte-model-proxy`）

`latte-model-proxy` 是一个本地 HTTP 代理服务，把多个下游 AI 厂商聚合到一个端点上，
客户端用 OpenAI / Anthropic / Ollama 任意一种 API shape 调用，本服务按
请求里的 `model` 路由到对应的下游厂商。可作为 OpenAI SDK、Claude Code、
Ollama 生态的统一入口。选路 + 限流冷却 + 5xx 熔断都委托给 `latte-router`。

### 8.1 启动

```bash
# 默认：自动找 ./proxy.toml → ~/.latte/proxy.toml，models 来自 proxy.toml
latte-model-proxy

# 显式指定 model 列表（CLI 覆盖 proxy.toml）
latte-model-proxy --models=claude-sonnet-4-20250514,deepseek-v4-flash --port=6666
```

### 8.2 完整示例

```bash
# 1. 准备配置
mkdir -p ~/.latte/models.d

cat > ~/.latte/proxy.toml <<'EOF'
[server]
host = "127.0.0.1"
port = 11434

[catalog]
models_dir = "~/.latte/models.d"
models = ["claude-sonnet-4-20250514", "deepseek-v4-flash"]
EOF

cat > ~/.latte/models.d/anthropic.toml <<'EOF'
[[models]]
id = "claude-sonnet-4-20250514"
api = "anthropic"
provider = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "${ANTHROPIC_API_KEY}"
EOF

cat > ~/.latte/models.d/deepseek.toml <<'EOF'
[[models]]
id = "deepseek-v4-flash"
api = "openai"
provider = "deepseek"
base_url = "https://api.deepseek.com"
api_key = "${DEEPSEEK_API_KEY}"
EOF

# 2. 启动
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=sk-...
latte-model-proxy

# 3. 客户端调用（OpenAI SDK 即可）
# base_url 指向 proxy；model 字段直接写下游 model id 或 CLI 暴露的 alias
```

### 8.3 暴露的路由

| 路径 | 方法 | 说明 |
| --- | --- | --- |
| `/`                            | GET, HEAD  | 健康检查横幅 |
| `/api/version`                 | GET         | 版本号 JSON |
| `/v1/models`                   | GET         | OpenAI 形状，列出 pool 中所有 model |
| `/v1/chat/completions`         | POST        | OpenAI Chat Completions，支持 `stream: true`（SSE 透传） |
| `/v1/messages`                  | POST        | Anthropic Messages，支持 `stream: true` |
| `/api/tags`                    | GET         | Ollama 形状 |
| `/api/show`                    | POST        | Ollama 形状（单模型详情 + 厂商信息） |
| `/api/chat`                    | POST        | Ollama 形状，转发到 OpenAI `/chat/completions` 并翻译响应 |

三个形态是同一组后端：客户端可以用任何一种 SDK 直接对接。

### 8.4 限流 + 熔断行为

- 请求里 `model` 字段是 pool 中的 id → `Router::select` 命中
- 命中 model 冷却中 → 自动降级到 pool 下一个 model
- 全部冷却 → 503 + `Retry-After: <秒数>`
- 上游 429 → 该 model 拉出到下个 refresh 时间点（`max(配置算的, Retry-After)`）
- 上游 5xx 连续 N 次 → 该 model 拉出 N 秒

### 8.5 使用示例（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="not-used",
)

resp = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "用 Rust 写一个二分查找"}],
)
print(resp.choices[0].message.content)
```

### 8.6 程序化使用（库 API）

```rust,no_run
use latte_model_proxy::{ModelEntry, Server, serve};

# async fn run() -> anyhow::Result<()> {
let pool: Vec<ModelEntry> = vec![/* 从 latte-router::ModelCatalog 加载 */];
let handle = serve(pool, "127.0.0.1:11434".to_string()).await?;
// ...
# Ok(())
# }
```

### 8.7 测试

20 个集成测试覆盖（`cargo test -p latte-model-proxy`）：

- `cli_test`：Args 解析（host/port/models/models-dir/config）
- `health_test`、`openai_models_test`、`openai_chat_test`、`openai_chat_stream_test`：OpenAI 兼容（含 SSE 透传）
- `anthropic_messages_test`：Anthropic 兼容 + 协议不匹配返 400
- `ollama_compat_test`：`/api/tags`、`/api/show`、`/api/chat`（Ollama ↔ OpenAI 翻译）
- `serve_entry_test`：`serve()` 绑端口 + Router::select / UnknownModel 路径

```bash
cargo test -p latte-model-proxy
```