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

## booking-mcp（预约工具 MCP Server）

```bash
uv run python scripts/seed_all.py --reset          # 30 名陪玩师 + 14 天档期 + 演示用户 demo(id=1)
uv run booking-mcp --transport http --port 8101    # Agent 使用：http://127.0.0.1:8101/mcp
uv run booking-mcp --transport stdio               # 桌面客户端使用
uv run python scripts/mcp_smoke.py                 # stdio 冒烟：找人 → 报价 → 下单
```

在 Claude Desktop 中挂载的配置片段与错误码说明见 [docs/mcp_desktop.md](docs/mcp_desktop.md)。
