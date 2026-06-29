# latte_project_debug

跑 `latte-model-proxy` 用的最小化本地配置。

## 文件

```
latte_project_debug/
├── proxy.toml                      # proxy 启动配置
├── .latte/
│   └── models.d/                   # 项目级 model 定义（覆盖全局 ~/.latte/models.d/）
│       ├── anthropic.toml          # Claude Sonnet 4
│       ├── deepseek.toml           # DeepSeek V4 Flash
│       └── ollama-local.toml       # 本地 Ollama 兜底
└── README.md
```

## 启动

需要设环境变量（key 来自上游 API）：

```bash
cd latte_project_debug
export ANTHROPIC_API_KEY=sk-ant-...
export DEEPSEEK_API_KEY=sk-...

# release 二进制（推荐）
latte-model-proxy --config=./proxy.toml

# 或 cargo run
cargo run -p latte-model-proxy --release -- --config=./proxy.toml
```

启动成功输出：

```
[INFO latte_model_proxy] listening on http://127.0.0.1:11434; models: claude-sonnet-4-20250514, deepseek-v4-flash, qwen2.5-coder-32b-instruct
```

## 测试

```bash
# 列表
curl http://127.0.0.1:11434/v1/models

# 聊天
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "hi"}]
  }'
```

请求里 `model` 字段写 pool 里的 id。命中的 model 冷却中 → 自动降级到下一个；全部冷却 → 503 + `Retry-After`。

## 调权重 / 冷却

- 改 `proxy.toml` 的 `catalog.models` 数组顺序 = 调权重
- 改 `models.d/*.toml` 的 4 个冷却字段 = 调单个 model 的限流 / 熔断行为
- 不改 proxy.toml，只改 `models.d/*.toml` → 自动重新加载
