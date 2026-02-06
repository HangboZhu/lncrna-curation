# 2026-02-json-curation

## 运行

```bash
uv run main.py
```

## .env 配置说明（速度 vs 保守）

**API/模型**
- `OPENAI_BASE_URL`: 反代或网关地址
- `OPENAI_API_KEY`: 密钥
- `MODEL_NAME`: 模型名称（通常 `flash` 更快，`pro` 更稳但慢）

**JSON 模式**
- `VERIFIER_JSON_MODE=true`: 仅当模型支持 `response_format` 时启用（更稳但可能更慢/不兼容）
- `VERIFIER_JSON_MODE=false`: 兼容性更好（默认）

**提示词模式（速度 vs 保守）**
- `CURATOR_PROMPT_MODE=compact`: 快（提示词短）
- `CURATOR_PROMPT_MODE=full`: 慢但更保守（提示词长）
- `VERIFIER_PROMPT_MODE=compact`: 快
- `VERIFIER_PROMPT_MODE=full`: 慢但更保守

**Verifier 触发策略**
- `VERIFIER_CONDITIONAL=true`: 仅在疑似嵌套/复合词时才跑 Verifier（更快）
- `VERIFIER_CONDITIONAL=false`: 每次都跑 Verifier（更慢但更保守）

## 推荐组合

- **速度优先**：
  - `CURATOR_PROMPT_MODE=compact`
  - `VERIFIER_PROMPT_MODE=compact`
  - `VERIFIER_CONDITIONAL=true`
  - `MODEL_NAME=...flash`

- **保守优先**：
  - `CURATOR_PROMPT_MODE=full`
  - `VERIFIER_PROMPT_MODE=full`
  - `VERIFIER_CONDITIONAL=false`
  - `MODEL_NAME=...pro`
