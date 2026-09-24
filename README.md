# Rift Companion Agent

「峡谷陪练」—— 虚构英雄联盟陪玩预约平台的对话式 AI Agent。
设计文档见 [DEV_SPEC.md](DEV_SPEC.md)。

## 快速开始

```bash
uv sync
uv run pytest -q
```

## 仓库结构

| 目录 | 内容 |
|---|---|
| `packages/common` | `rift_common`：配置、LLM/Embedding Provider、Trace |
| `packages/domain` | `rift_domain`：纯代码领域核心（槽位、规则、匹配、计价、退款） |
| `services/booking_mcp` | 预约 MCP Server |
| `services/knowledge_mcp` | 知识检索 MCP Server（迁入 RAG 项目） |
| `apps/agent` | LangGraph 编排 |
| `apps/web` | FastAPI Web 前端 |
| `apps/dashboard` | Streamlit 看板 |
| `training` | 本地槽位抽取小模型后训练流水线 |
| `eval` | L2/L3/L4 评测数据集、剧本与报告 |
| `config` | `settings.yaml`、`domain.yaml`、prompts |

## 配置与密钥

复制 `.env.example` 为 `.env` 并填写。密钥只从环境变量读取，`config/*.yaml` 中以 `${VAR}` 引用。
