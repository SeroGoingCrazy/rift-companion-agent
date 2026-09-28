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

## 知识库（knowledge-mcp）

知识库由 `config/domain.yaml` 与陪玩师种子生成，Embedding 使用本地 `BAAI/bge-small-zh-v1.5`（首次运行自动下载），不需要 API key：

```bash
uv sync --extra local-embedding                     # sentence-transformers（含 torch）
uv run python scripts/gen_kb_docs.py                # 生成 kb/（39 篇 Markdown）
uv run python scripts/ingest_kb.py                  # 导入 3 个集合；未变更文件跳过，修改/删除的文件先清旧 chunk
uv run python services/knowledge_mcp/scripts/query.py --query "开局前三小时取消退多少"
```

修改 `domain.yaml` 后重新执行后两步即可同步；`--force` 全量重导入，`--collections platform_rules` 只导入指定集合。

## Agent（命令行对话）

需要 `DEEPSEEK_API_KEY`。booking-mcp 可走 HTTP（`config/settings.yaml` 中的地址），也可在进程内运行：

```bash
uv run python scripts/chat_cli.py --booking-db data/booking.db
```

知识检索（咨询类问题）依赖 knowledge-mcp（阶段 E）；未启动时 Agent 会如实提示"规则查询服务暂时连不上"，预约与订单管理不受影响。

## Web 界面

```bash
export RIFT_WEB_SECRET=...                                   # cookie 签名密钥（不设则每次启动随机）
uv run python -m rift_web --booking-db data/booking.db       # http://127.0.0.1:8000
```

页面：昵称登录 → 聊天预约（SSE 流式、候选卡片、确认卡片）/ 我的订单（模拟支付、退款预估后取消）/ 陪玩师（按模式、段位、位置、性别筛选，展示近 3 天空闲时段）。
