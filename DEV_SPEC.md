# Developer Specification (DEV_SPEC)

> 项目：**Rift Companion Agent**（虚构平台「峡谷陪练」的英雄联盟陪玩预约 AI Agent）
> 版本：v0.1（设计定稿，待开发）
> 来源：整合 `smart-appointment-ai-agent`（业务编排）+ `MODULAR-RAG-MCP-SERVER`（知识检索与工程底座）+ `post-training-slot-extractor`（本地小模型后训练）

## 目录

1. 项目概述
2. 核心特点
3. 技术选型
4. 测试方案
5. 系统架构与模块设计
6. 项目排期
7. 可扩展性与未来展望
8. 附录：设计决策记录（ADR）

---

## 1. 项目概述

「峡谷陪练」是一个虚构的英雄联盟（LoL）陪玩预约平台。本项目为其构建一个对话式 AI 前台：用户用自然语言咨询平台规则、预约陪玩师、查看和取消订单；Agent 负责理解意图、多轮收集预约信息、匹配陪玩师、报价确认并落单。

本项目不是从零造轮子，而是把三个已验证的项目的设计**整合为一个端到端闭环**：

| 来源项目 | 在本项目中的角色 |
|---|---|
| smart-appointment-ai-agent | 业务骨架：意图分类 → 多轮槽位收集 → 匹配 → 预约的状态机思路 |
| MODULAR-RAG-MCP-SERVER | 知识服务（Hybrid Search + Rerank，以 MCP 暴露）、可插拔 Provider 工厂、Trace 与 Dashboard、评估体系 |
| post-training-slot-extractor | 本地 0.6B 槽位抽取模型的完整后训练流水线：Teacher 造数 → SFT/DPO → 确定性评测 → GGUF 量化部署 |

### 设计理念 (Design Philosophy)

1. **LLM 负责理解，代码负责决策 (LLM Understands, Code Decides)**
   模型只输出"用户这一轮说了什么"（槽位增量 + 轮次意图 + 确认态）；必填校验、缺失字段、动作选择、匹配、计价、退款全部由确定性代码完成。好处：可测试、可解释、规则可热更新无需重训，小模型输出更短更准。
2. **大小模型协同与优雅降级 (Hybrid Model Routing with Graceful Degradation)**
   高频、规则明确的槽位抽取交给本地小模型；分类、润色、RAG 回答交给远程大模型（DeepSeek）。小模型输出校验失败 → 重试 → 降级到大模型 → 兜底澄清，任何一环失败都不中断对话。
3. **工具协议化 (Everything is an MCP Tool)**
   知识检索与预约操作分别封装为两个独立的 MCP Server（Streamable HTTP）。Agent 只是它们的一个客户端；同样的工具也可以直接挂进 Claude Desktop / Cursor 使用。
4. **唯一数据源 (Single Source of Truth)**
   模式、段位、价格系数、退款规则写在 `config/domain.yaml` 一处；代码逻辑、数据库种子、知识库文档（Markdown）都从它生成，杜绝"知识库说一套、代码做一套"。
5. **可观测驱动的数据飞轮 (Observability-Driven Data Flywheel)**
   每一轮对话产生层级 Trace（本地 SQLite + Langfuse）。影子模式对比大小模型输出，看板挖掘不一致样本，回流为下一轮训练数据。
6. **先可测，再集成 (Test First, Integrate Later)**
   排期上先完成零 LLM 依赖的领域核心并用 L1 测试锁死，再接入模型与服务。

### 非目标 (Non-Goals, v1)

- 不做真实支付（仅模拟状态流转）、不做改期（= 取消 + 新建，放入第 7 章）
- 只支持英雄联盟一款游戏
- 不做密码账户体系（昵称登录）
- 不做 LLM 用户模拟器评测（放入第 7 章）
- 不自托管 Langfuse（使用 Langfuse Cloud）

---

## 2. 核心特点

### 2.1 业务能力

- **意图分类**：首轮由大模型分类为 `booking / consult / manage / other`。
- **多轮预约**：6 种模式（单双排、灵活组排、匹配、大乱斗、海克斯大乱斗、斗魂竞技场），按小时计费，支持段位要求、位置偏好、性别、开麦、预算、风格偏好、指定陪玩师。
- **插话式咨询**：预约进行中问"取消要扣钱吗"，Agent 调 RAG 回答后提示继续预约，已收集槽位不丢失。
- **智能匹配与放宽**：硬过滤（SQL）+ 软排序（位置命中 + 风格 embedding 相似度 + 评分）；零结果按 `时间±1h → 性别 → 位置` 逐级放宽并在回复中说明，预算与段位永不放宽。
- **确认与防重**：落单前展示结构化摘要与代码计算的总价，LangGraph `interrupt` 等待确认；写入时事务内二次冲突检测，两个会话抢同一时段只有一个成功。
- **订单管理**：查看我的订单、按退款规则计算退款并确认取消、模拟支付、开局提醒文案。

### 2.2 小模型后训练（亮点）

- Qwen3-0.6B（1.7B 对照）LoRA SFT，协议为**纯抽取器 + delta 三态语义**，输出约 80 token。
- 更强模型作 Teacher 造数，场景规格驱动 + 校验 + 覆盖审计 + 数据卡版本化。
- 错误驱动的对比集（Contrast Sets）多轮迭代。
- **DPO 消融**：规则扰动偏好对 vs **On-Policy 偏好对**（SFT 模型真实错误作 rejected），β ∈ {0.1, 0.3}。
- GGUF Q4_K_M + 前缀缓存预热，CPU 部署。

### 2.3 工程能力

- uv workspace monorepo；LangGraph 编排 + SqliteSaver checkpointer；自有可插拔 Provider 工厂（不依赖 LangChain 本体）。
- 两个 MCP Server（Streamable HTTP，保留 stdio 供桌面客户端）。
- 层级 Span Trace，跨服务 `trace_id` 关联；本地 SQLite（统计）+ Langfuse Cloud（排查）双写，Langfuse 不可达静默降级。
- 四层评测（L1 组件 / L2 槽位 / L3 RAG / L4 端到端剧本），三种模型配置对比表。
- 影子模式 + 样本挖掘看板 + 至少一轮完整数据飞轮。
- docker compose 一键启动；GitHub Actions 离线 CI（零 API Key）。

### 2.4 简历核心指标（纯大模型列由 H5 产出，其余待阶段 J/K 产出）

| 指标 | 纯大模型 | 纯小模型 | 小模型 + 降级 |
|---|---|---|---|
| L4 任务完成率 | 100%（22/22） | – | – |
| 平均轮数 | 3.32 | – | – |
| 单轮 P50 / P95 延迟 | 804 / 1937 ms | – | – |
| 远程 LLM 调用次数 / 会话 | 4.14 | – | – |
| L2 主集 / holdout 通过率 | 98.3% / 88.9% | – | – |

「纯大模型」列由阶段 H5 产出（DeepSeek，见 `eval/reports/baseline_llm_summary.md`）。

---

## 3. 技术选型

### 3.1 总体技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| 语言与包管理 | Python 3.12 + uv workspace | 与 slot-extractor 一致 |
| 编排 | LangGraph（`StateGraph` + `SqliteSaver` + `interrupt`） | 控制流是显式状态机；**不使用** `langchain-openai` / `AgentExecutor` |
| LLM / Embedding | 自有 Provider 工厂（迁自 RAG 项目 `libs/llm`） | 远程默认 DeepSeek（OpenAI 兼容）；本地 `llama_server`；Embedding 可配 |
| MCP | 官方 `mcp` Python SDK | Streamable HTTP（主）+ stdio（桌面客户端） |
| 知识服务 | 迁入 MODULAR-RAG-MCP-SERVER | Chroma + BM25(jieba) + RRF + Rerank；新增 `MarkdownLoader`；关闭多模态 |
| 业务存储 | SQLite + SQLAlchemy 2.x | 陪玩师、档期、订单、用户 |
| Web | FastAPI + Jinja2 + 原生 JS + SSE | 聊天 / 我的订单 / 陪玩师列表 |
| 看板 | Streamlit | 复用 RAG 6 页 + 新增 3 页 |
| 可观测 | 自有 Span Trace → SQLite + Langfuse Cloud | Prompt 留在仓库，不用 Langfuse Prompt 管理 |
| 训练 | LLaMA-Factory（LoRA SFT / DPO）、llama.cpp（GGUF 量化与服务） | 租用 24GB GPU |
| 部署 | docker compose（web+agent、booking-mcp、knowledge-mcp、llama-server） | 另附无 Docker 的本地脚本 |
| 质量 | pytest、ruff、mypy、GitHub Actions | CI 只跑离线测试 |

### 3.2 领域模型与槽位协议 (Domain & Slot Contract)

#### 3.2.1 领域枚举（`config/domain.yaml`，唯一数据源）

| 概念 | 取值 |
|---|---|
| `game_mode` | `ranked_solo_duo`（单双排）/ `ranked_flex`（灵活组排）/ `normal_draft`（匹配）/ `aram`（大乱斗）/ `aram_mayhem`（海克斯大乱斗）/ `arena`（斗魂竞技场） |
| `rank` | `iron` < `bronze` < `silver` < `gold` < `platinum` < `emerald` < `diamond` < `master` < `grandmaster` < `challenger` |
| `role` | `top` / `jungle` / `mid` / `adc` / `support` |
| `service_type` | `climb`（上分）/ `casual`（娱乐）/ `coaching`（教学） |
| `gender` | `female` / `male` |

领域规则（均在 `domain.yaml` 中配置，代码读取）：

- **必填规则**：`game_mode`、`start_time`、`duration_hours`；`game_mode ∈ {ranked_solo_duo, ranked_flex}` 时额外要求 `rank_requirement`。
- **不适用字段**：`aram / aram_mayhem / arena` 下 `rank_requirement` 与 `role_preference` 强制置空（代码处理）。
- **服务类型推断**：用户未明确说明时，排位 → `climb`，其余 → `casual`；用户明确说明以用户为准。
- **单双排段位规则（平台虚构规则）**：陪玩师段位须满足 `rank_requirement ≤ companion.rank ≤ rank_requirement + max_tier_gap`（默认 `max_tier_gap = 2`）。
- **计价**：`总价 = companion.hourly_price × service_multiplier[service_type] × duration_hours`；默认系数 `casual 1.0 / climb 1.2 / coaching 1.5`。
- **退款**：距开局 ≥ 15 分钟全额；< 15 分钟（含已开局）不退（可配；2026-10 由原 24h / 2h 三档改为两档）。
- **时长**：0.5h 为步长，1–8h。
- **放宽顺序**：`time_window(±1h) → companion_gender → role_preference`；`budget`、`rank_requirement` 永不放宽。

#### 3.2.2 槽位抽取器输出协议（Slot Extractor Output）

模型每轮输入：系统提示词 + `current_state`（已合并的槽位）+ `candidates`（若已展示候选陪玩师）+ 最近 N 轮对话 + 本轮用户输入。模型输出**严格 JSON**：

```json
{
  "turn_intent": "booking | consult | unrelated",
  "delta": {
    "game_mode": "ranked_solo_duo",
    "start_time_expr": "明晚八点",
    "duration_hours": 2,
    "rank_requirement": "diamond",
    "role_preference": ["jungle"],
    "service_type": "climb",
    "companion_gender": "female",
    "voice_required": true,
    "budget_per_hour": 80,
    "style_preference": "温柔会聊天",
    "companion_name": "阿狸酱"
  },
  "confirmation": "yes | no | none"
}
```

**Delta 三态语义**（核心约定）：

| 情形 | 表示 | 合并行为 |
|---|---|---|
| 本轮未提及 | 键**不出现** | 保持原值 |
| 明确不限（"段位无所谓"） | `"any"` | 置为 ANY，满足必填但不参与过滤 |
| 撤回（"时间先不定"） | `null` | 清空，重新变为缺失 |
| 替换（"打野换成辅助"） | 新值；list 整体替换 | 覆盖 |

- 时间只输出**原始表达** `start_time_expr`，由代码中文时间解析器结合当前时间与 `current_state.start_time` 解析（支持"明晚八点""周六下午""改成晚一小时"）。
- `companion_name` 可引用候选列表（"就第二个""选阿狸酱"）→ 模型输出候选中的确切姓名。
- `turn_intent = consult` 表示预约中插话咨询；`unrelated` 表示与平台无关。
- 严格校验：Pydantic `extra="forbid"`，枚举越界、类型错误、多余字段一律判失败。

#### 3.2.3 代码侧会话状态 (`BookingState`)

合并后的槽位 + 派生字段：`start_time`（绝对时间）、`service_type_effective`、`missing_fields`、`candidates`、`selected_companion_id`、`quote`、`relaxations`、`pending_action`（`await_confirm_booking` / `await_confirm_cancel` / none）。

### 3.3 Agent 编排设计 (LangGraph Orchestration)

#### 3.3.1 会话阶段 (Phase)

`IDLE → BOOKING / MANAGE`；`CONSULT` 为单轮处理后回到原阶段。

#### 3.3.2 状态图

```
                 ┌─────────────┐
 user input ───▶ │ route_phase │
                 └──────┬──────┘
          IDLE          │           BOOKING                     MANAGE
     ┌──────────────────┼──────────────────────┐                   │
     ▼                  │                      ▼                   ▼
 ┌────────┐             │              ┌───────────────┐   ┌────────────────┐
 │classify│ (LLM)       │              │ extract_slots │   │ manage_bookings│
 └───┬────┘             │              └───────┬───────┘   │ list / cancel  │
     │ booking ─────────┘                      │           └───────┬────────┘
     │ consult ──▶ consult ──▶ render   turn_intent?               │
     │ manage  ──▶ manage_bookings      ├─ consult ──▶ consult_interject ──▶ render
     │ other   ──▶ render(fallback)     ├─ unrelated ──▶ reclassify / 礼貌拉回
                                        └─ booking
                                              ▼
                                   merge_state → resolve_time → compute_rules
                                              ▼
                                           decide ──┬─ missing → ask_missing → render
                                                    ├─ need candidates → find_companions(MCP) → present → render
                                                    ├─ companion selected → quote_price(MCP) → confirm(interrupt)
                                                    └─ confirmation=yes → create_booking(MCP) → reminder → render
```

- **显式切换**：用户说"不约了"→ `confirmation=no` 且无其他 delta 时归档状态、回到 IDLE。
- **确认断点**：`confirm` 节点使用 `interrupt()`，下一轮用户输入经 `extract_slots` 得到 `confirmation` 后 resume；若同时带有 delta（"改成三小时再下单"），回到 `merge_state` 重新报价。
- **持久化**：`SqliteSaver`，`thread_id = session_id`；每轮结束自动 checkpoint，服务重启不丢会话。
- **异常兜底**：任一节点异常 → 记录 Span 错误 → 回复澄清话术，状态保持（不像原项目那样强制重置）。

#### 3.3.3 回复生成

- 默认 **模板渲染**（`config/prompts/templates/replies.yaml`，按 `reply_type` + 槽位插值），零延迟、完全确定。
- 配置开关 `reply.polish: true` 时交给 DeepSeek 润色（输入为模板结果 + 结构化事实，要求不得改动事实），演示时可对比延迟与效果。

### 3.4 模型层与路由设计 (Model Layer & Routing)

#### 3.4.1 Provider 工厂

迁自 RAG 项目：`BaseLLM` / `BaseEmbedding` + `register_provider()` 注册式工厂，注册时校验子类，未知 provider fail-fast。提供：

- `openai_compatible`（DeepSeek 默认；兼容 Qwen、OpenAI、Azure）
- `llama_server`（本地 llama.cpp，OpenAI 兼容 `/v1`，temperature 0）
- `mock`（测试与 CI 用，按 fixture 返回）

密钥**只从环境变量读取**（`${DEEPSEEK_API_KEY}` 形式在 YAML 中引用），`.env` 在 `.gitignore` 中。

#### 3.4.2 SlotExtractor 路由

```
SlotExtractor (interface): extract(ctx) -> SlotExtraction
  ├─ LLMSlotExtractor     # DeepSeek，同一 prompt + JSON schema
  ├─ LocalSlotExtractor   # llama-server + GGUF
  └─ RoutedSlotExtractor  # 组合器
       1. primary.extract → Pydantic 严格校验
       2. 失败 → 带错误信息重试 1 次
       3. 仍失败 / 超时 (>5s) / 不可达 → fallback.extract
       4. 全失败 → SlotExtraction.failed()（上层回复澄清，状态不变）
       5. shadow 开启 → 异步调用 shadow 实现，仅记录 Trace
```

配置里程碑：

| 阶段 | primary | fallback | shadow |
|---|---|---|---|
| M1（F 阶段完成） | llm | – | – |
| 影子评估（J 阶段） | llm | – | local |
| M3（上线态） | local | llm | – |

### 3.5 MCP 服务设计 (MCP Services)

#### 3.5.1 booking-mcp

| Tool | 入参 | 出参 | 说明 |
|---|---|---|---|
| `find_companions` | 合并后的过滤条件 + `style_preference` + `top_k` | 候选列表（含得分、命中原因）+ `relaxations` | 硬过滤 → 软排序 → 零结果逐级放宽 |
| `quote_price` | `companion_id, service_type, duration_hours` | 单价、系数、总价 | 纯计算 |
| `create_booking` | `user_id, companion_id, start_time, duration_hours, game_mode, service_type, ...` | 订单 / 冲突错误 | 事务内二次冲突检测；状态 `pending_payment` |
| `list_my_bookings` | `user_id, status?` | 订单列表 | |
| `cancel_booking` | `user_id, booking_id, dry_run` | 退款金额 / 执行结果 | `dry_run=true` 用于确认前报价 |

- 错误映射为 JSON-RPC 错误码（沿用 RAG `protocol_handler` 思路）：`SLOT_CONFLICT`、`NOT_FOUND`、`FORBIDDEN`、`INVALID_ARGUMENT`。
- 每个工具接受可选 `_meta.trace_id`，用于跨服务 Trace 关联。

#### 3.5.2 knowledge-mcp

迁入 RAG 项目，保留三个工具：`query_knowledge_hub`、`list_collections`、`get_document_summary`。改动：

- 新增 `MarkdownLoader`（按标题层级保留 section metadata），通过 loader 工厂注册。
- 新增 Streamable HTTP 启动模式（`--transport http --port 8102`），stdio 保留。
- `vision_llm` 默认关闭；图片相关 transform 在配置中禁用，代码保留。
- 读取 `_meta.trace_id` 作为 parent trace id。

#### 3.5.3 知识库内容（全部由种子数据生成）

| 集合 | 来源 | 示例问题 |
|---|---|---|
| `platform_rules` | `domain.yaml` → `scripts/gen_kb_docs.py` | "开局前 3 小时取消退多少？""教学怎么收费？" |
| `modes_and_ranks` | `domain.yaml` + 手写说明模板 | "钻石能和铂金双排吗？""斗魂竞技场是几个人？" |
| `companion_profiles` | `companions` 种子 → 每人一页 | "谁的阿狸最好？""有没有会玩辅助的小姐姐？" |

### 3.6 可观测性设计 (Observability)

#### 3.6.1 Span 结构

```
turn (session_id, user_id, turn_idx, input, reply, phase_before/after, latency)
 ├─ classify           (model, label, latency)
 ├─ extract_slots      (model, raw_output, valid, retried, fell_back, latency, shadow_output?)
 ├─ merge_state        (state_diff, time_expr → resolved)
 ├─ decide             (missing_fields, action)
 ├─ tool:<name>        (args, result_summary, relaxations, error)
 │    └─ (knowledge-mcp 侧 query trace，parent = trace_id)
 └─ render_reply       (mode: template|polish, latency)
```

- LangGraph 节点通过 `@traced_node` 装饰器自动打点。
- **双写**：`SqliteSink`（结构化列，供统计与看板）+ `LangfuseSink`（span、LLM 输入输出、token 成本、session 视图）；Langfuse 异常只记 warning，不影响对话。
- knowledge-mcp 保留原 JSONL trace 与 Langfuse exporter。

#### 3.6.2 看板分工

| 能力 | 位置 |
|---|---|
| Span 瀑布、LLM I/O、token 成本、会话回放 | Langfuse Cloud |
| RAG Ingestion / Query 追踪、数据浏览、评估面板 | Streamlit（复用 RAG 6 页） |
| **槽位质量**：大小模型一致率、降级率、逐字段不一致分布 | Streamlit 新增 |
| **业务指标**：转化率、平均轮数、零结果与放宽频率、取消率 | Streamlit 新增 |
| **样本挖掘**：筛选不一致 / 降级 / 用户重复澄清轮次 → 导出待标注 JSONL | Streamlit 新增 |

### 3.7 训练与数据设计 (Post-Training)

沿用 slot-extractor 流水线，替换"字段契约、工具、场景规格、评分器"四件套。

| 项 | 方案 |
|---|---|
| 基座 | Qwen3-0.6B（主）、Qwen3-1.7B（对照），`enable_thinking: false` |
| Teacher | 可插拔后端（Claude / OpenAI 旗舰模型），JSON Schema 约束输出 |
| 场景类别 | 首轮抽取、多轮增量、三态语义、相对时间表达、候选引用、插话咨询、unrelated、确认/拒绝、6 模式说法与黑话（双排、大乱斗、斗魂、上分、带飞） |
| 数据流 | 场景规格 → Teacher 生成 → 校验器 → 覆盖/标签审计 → 渲染 SFT → 数据卡版本化 `data/processed/sft/v0.x` |
| 规模 | 第 1 轮 ≈ 800 → 对比集 2–3 轮 → ≈ 1500 |
| SFT | LoRA r16 / α32 / all modules / lr 1e-4 / 3 epochs / cutoff 2048 / mask_history / bf16 |
| DPO 消融 | ① 规则扰动偏好对；② On-Policy 偏好对（SFT 模型在训练集上采样，错误输出为 rejected，Teacher 输出为 chosen）；β ∈ {0.1, 0.3}，lr 5e-6，1 epoch |
| 量化 | merge → GGUF → Q4_K_M（主）/ Q8_0（质量档），imatrix 可选；启动时预热共享前缀缓存 |
| 数据飞轮 | 影子模式不一致样本 → Teacher 重标 + 人工抽检 → 并入下一轮，至少完整跑通 1 轮 |

**评测集隔离**：L2 主集与 holdout 由人工编写与审核，不使用任何 Teacher 生成，并在训练前冻结（记录 checksum）。

---

## 4. 测试方案

### 4.1 设计理念

测试驱动：每个任务都有明确的验收命令；领域核心零 LLM 依赖先行；外部依赖（LLM、MCP、llama-server）在单测中全部通过 `mock` provider 或 fixture 替代。

### 4.2 四层评测

| 层 | 对象 | 数据 | 评分 | 运行时机 |
|---|---|---|---|---|
| **L1 组件** | 时间解析、状态合并、必填规则、服务类型推断、匹配过滤排序与放宽、计价、退款、冲突检测 | 表驱动手写用例 | 精确断言 | 每次提交（CI） |
| **L2 槽位抽取** | 单轮 delta、turn_intent、confirmation | 主集 ≈ 60 + holdout ≈ 25（人工） | 协议层（JSON 合法、schema、枚举）+ 任务层（逐字段；`style_preference` 用本地 embedding 模糊匹配）；通过 = 协议通过且任务分 ≥ 0.95 | 每轮训练；LLM/本地各跑作对照 |
| **L3 RAG** | 规则、模式、陪玩师问答 | ≈ 30 条 golden QA（标准答案由唯一数据源生成） | Hit Rate、MRR、ragas faithfulness / answer relevancy | 知识库变动时 |
| **L4 端到端** | 完整多轮流程 | ≈ 20 个 YAML 剧本 | 断言最终 DB 状态与关键 Span（订单是否创建、总价、放宽项、退款额），不评文案 | 发布前；三种模型配置 |

### 4.3 L4 剧本示例（`eval/scenarios/*.yaml`）

```yaml
id: interject_refund_then_continue
now: "2026-10-01 14:00"
user: { nickname: tester_01 }
turns:
  - say: "帮我约个明晚八点的单双排，钻石以上"
  - say: "等下，取消要扣钱吗？"
    expect_span: { tool: query_knowledge_hub }
  - say: "那就两小时吧"
  - say: "选第一个"
  - say: "确认"
expect:
  booking:
    status: pending_payment
    game_mode: ranked_solo_duo
    start_time: "2026-10-02 20:00"
    duration_hours: 2
  state_phase: IDLE
```

其余必备剧本：模式黑话识别、三态撤回、相对时间修改、段位不足触发放宽、预算永不放宽、双会话抢同一时段、取消退款三档、unrelated 拉回、显式放弃预约、服务重启后续聊。

### 4.4 测试分层目录

```
tests/
  unit/          # L1，离线，CI
  integration/   # SQLite、MCP 进程内、LangGraph 全图（mock LLM）
  e2e/           # 真实服务 + 真实 LLM（手动/发布前）
  fixtures/
eval/
  datasets/      # slot_main.jsonl, slot_holdout.jsonl, rag_golden.jsonl（带 checksum）
  scenarios/     # L4 剧本
  reports/       # 评测输出
```

pytest markers：`unit`、`integration`、`e2e`、`llm`（需要真实 API）、`slow`。

### 4.5 CI

GitHub Actions：`ruff` + `mypy`（核心包）+ `pytest -m "unit or integration"`，全部使用 `mock` provider，不需要任何 API Key；额外跑 1 个 L4 冒烟剧本（mock 抽取器按 fixture 回放）。

---

## 5. 系统架构与模块设计

### 5.1 整体架构

```
┌──────────────────────────── docker compose ─────────────────────────────┐
│                                                                         │
│  ┌──────────── web (FastAPI :8000) ────────────┐                        │
│  │ Jinja2 + JS + SSE  │ 昵称登录 / 聊天 / 订单 / 陪玩师               │
│  │        │                                    │                        │
│  │  rift_agent (LangGraph, SqliteSaver)        │                        │
│  │   ├─ classify / polish / consult ─────────────────▶ DeepSeek API     │
│  │   ├─ RoutedSlotExtractor ─── local ──────────────▶ llama-server :8080│
│  │   │                      └── fallback ────────────▶ DeepSeek API     │
│  │   ├─ MCP client ── Streamable HTTP ─▶ booking-mcp   :8101 ─▶ SQLite  │
│  │   └─ MCP client ── Streamable HTTP ─▶ knowledge-mcp :8102 ─▶ Chroma  │
│  │                                                            + BM25    │
│  │  Trace: SqliteSink ─▶ traces.db   LangfuseSink ─▶ Langfuse Cloud     │
│  └─────────────────────────────────────────────┘                        │
│                                                                         │
│  dashboard (Streamlit :8501) ─ 读 traces.db / booking.db / RAG traces   │
└─────────────────────────────────────────────────────────────────────────┘
  training/ (离线，GPU 机器) ──▶ GGUF ──▶ llama-server
```

### 5.2 目录结构

```
rift-companion-agent/
├── pyproject.toml                 # uv workspace 根
├── uv.lock
├── .env.example                   # DEEPSEEK_API_KEY, LANGFUSE_*, TEACHER_* ...
├── .gitignore
├── README.md
├── DEV_SPEC.md
├── config/
│   ├── settings.yaml              # 服务地址、模型路由、trace、reply.polish ...
│   ├── domain.yaml                # 唯一数据源：模式/段位/系数/退款/放宽/必填
│   └── prompts/
│       ├── classify.txt
│       ├── slot_extract.txt       # 线上与训练共用（训练渲染时读取同一文件）
│       ├── reply_polish.txt
│       ├── consult_answer.txt
│       └── templates/replies.yaml
├── packages/
│   ├── common/src/rift_common/
│   │   ├── settings.py            # 加载 + 环境变量展开 + 校验
│   │   ├── llm/                   # base, factory, openai_compatible, llama_server, mock
│   │   ├── embedding/             # base, factory, openai_compatible, local(sentence-transformers), mock
│   │   ├── trace/                 # span.py, context.py, decorators.py, sinks/{sqlite,langfuse}.py
│   │   └── logging.py
│   └── domain/src/rift_domain/
│       ├── config.py              # DomainConfig（读取 domain.yaml）
│       ├── enums.py
│       ├── slots.py               # SlotDelta / SlotExtraction / BookingState（Pydantic）
│       ├── timeparse.py           # 中文时间表达解析
│       ├── merge.py               # delta 三态合并
│       ├── rules.py               # 必填、不适用字段、服务类型推断
│       ├── matching.py            # 硬过滤条件构造、软排序、放宽策略
│       ├── pricing.py
│       └── refund.py
├── services/
│   ├── booking_mcp/src/booking_mcp/
│   │   ├── server.py              # stdio / http
│   │   ├── db/{models.py, repo.py, session.py}
│   │   ├── tools/{find_companions,quote_price,create_booking,list_my_bookings,cancel_booking}.py
│   │   └── seed.py                # companions + schedules 种子
│   └── knowledge_mcp/             # 迁入 MODULAR-RAG-MCP-SERVER（src/ 结构保留）
│       └── src/libs/loader/markdown_loader.py   # 新增
├── apps/
│   ├── agent/src/rift_agent/
│   │   ├── graph/{state.py, builder.py, nodes/*.py}
│   │   ├── extractors/{base.py, llm.py, local.py, routed.py, prompt.py}
│   │   ├── mcp_clients.py
│   │   └── replies/{templates.py, polisher.py}
│   ├── web/src/rift_web/{app.py, routes/, templates/, static/}
│   └── dashboard/                 # Streamlit：复用 RAG 页面 + 新增 3 页
├── training/
│   ├── src/rift_training/
│   │   ├── contract/              # 输出协议（直接 import rift_domain.slots）
│   │   ├── data/{specs, generate.py, validate.py, audit.py, render_sft.py, dpo_rule.py, dpo_onpolicy.py, card.py}
│   │   ├── inference/{base.py, llama_server.py, teacher_anthropic.py, teacher_openai.py, mock.py}
│   │   └── evaluation/{protocol.py, task.py, preferences.py, runner.py}
│   ├── configs/{data, training/llamafactory, quantization}/
│   ├── data/{raw, processed, eval}/
│   ├── experiments/               # 每轮 problem / strategy / conclusion
│   └── scripts/
├── eval/{datasets, scenarios, runners, reports}/
├── scripts/{seed_all.py, gen_kb_docs.py, ingest_kb.py, dev_up.ps1, dev_up.sh}
├── docker/{compose.yaml, web.Dockerfile, booking.Dockerfile, knowledge.Dockerfile, llama.Dockerfile}
├── tests/{unit, integration, e2e, fixtures}/
└── .github/workflows/ci.yml
```

### 5.3 模块依赖规则

```
apps/web ──▶ apps/agent ──▶ packages/domain ──▶ (无外部依赖，纯 Python + Pydantic)
                 │                ▲
                 ├──▶ packages/common
services/booking_mcp ──▶ packages/domain, packages/common
services/knowledge_mcp ──▶ (自带 libs；可选依赖 packages/common.trace)
training ──▶ packages/domain（协议与规则同源，保证线上/训练一致）
```

- `packages/domain` 不得依赖任何网络、LLM、数据库库。
- `apps/agent` 只通过 MCP 访问业务数据，不直接连 `booking.db`。
- 线上 prompt 与训练 prompt 读同一个 `config/prompts/slot_extract.txt`；训练数据卡记录该文件的 hash，线上启动时比对并告警不一致。

### 5.4 数据流

**预约一轮**：
`web /chat (SSE)` → `graph.invoke(thread_id=session_id)` → `route_phase(BOOKING)` → `extract_slots`（Routed）→ `merge_state` → `resolve_time` → `compute_rules` → `decide` → `find_companions`（MCP）→ `render_reply` → SSE 推送 → checkpoint 落盘 → trace 双写。

**插话咨询**：`extract_slots.turn_intent=consult` → `consult_interject`（`query_knowledge_hub` + DeepSeek 带引用作答）→ 附加续约提示（基于 `missing_fields`）→ 阶段保持 BOOKING。

**取消订单**：`classify=manage` → `list_my_bookings` → 用户选择 → `cancel_booking(dry_run)` 报退款 → `interrupt` 确认 → `cancel_booking` → 回复。

**知识库构建**：`domain.yaml + companions seed` → `gen_kb_docs.py` → `kb/*.md` → `ingest_kb.py`（knowledge-mcp ingestion pipeline，MarkdownLoader）→ Chroma + BM25。

**数据飞轮**：`traces.db (shadow 不一致)` → 看板导出 `mined.jsonl` → Teacher 重标 + 人工抽检 → `data/raw/v0.x` → SFT → L2/L4 回归 → 替换 GGUF。

### 5.5 配置驱动

```yaml
# config/settings.yaml（节选）
llm:
  default: { provider: openai_compatible, base_url: https://api.deepseek.com/v1, model: deepseek-chat, api_key: ${DEEPSEEK_API_KEY} }
  local_slot: { provider: llama_server, base_url: http://llama-server:8080/v1, model: rift-slot-0.6b-q4km, timeout_s: 5 }
embedding:
  provider: local            # sentence-transformers 中文小模型，风格匹配与 L2 模糊评分共用
slot_extractor:
  primary: llm               # llm | local
  fallback: null             # llm | local | null
  shadow: null
  max_retries: 1
reply:
  polish: false
mcp:
  booking:   { url: http://booking-mcp:8101/mcp }
  knowledge: { url: http://knowledge-mcp:8102/mcp, collection_default: platform_rules }
trace:
  sqlite_path: data/traces.db
  langfuse: { enabled: true, host: https://cloud.langfuse.com, public_key: ${LANGFUSE_PUBLIC_KEY}, secret_key: ${LANGFUSE_SECRET_KEY} }
session:
  checkpoint_path: data/checkpoints.db
  history_turns: 6
```

### 5.6 扩展性要点

- **新领域**：替换 `domain.yaml` + `slots.py` 字段 + 场景规格 + 评分器 → 重跑训练流水线（即 slot-extractor "可迁移 pattern" 的实证）。
- **新模型后端**：`register_provider()`。
- **新工具**：booking-mcp 新增 tool 模块并注册；Agent 侧新增节点与边。
- **新看板页**：`st.Page` 注册。

---

## 6. 项目排期

> 原则：每个任务 ≈ 1h 可验收；完成后在标题与进度表标 ✅ 并原子提交。任务格式：目标 / 修改文件 / 实现类与函数 / 验收标准 / 测试方法。

### 阶段总览（大阶段 → 目的）

| 阶段 | 目的 | 里程碑 |
|---|---|---|
| A 工程骨架 | 可导入、可测试、可配置、CI 跑通 | CI 绿 |
| B 公共库 | Provider 工厂、Embedding、Trace 双写 | |
| C 领域核心 | 纯代码业务规则，零 LLM | **L1 全绿** |
| D booking-mcp | 预约工具以 MCP 对外 | Claude Desktop 可预约 |
| E knowledge-mcp | RAG 迁入 + Markdown + 知识库生成 | **L3 达标** |
| F Agent 编排 | LangGraph 全流程（LLM 抽取器） | **M1：大模型版端到端可演示** |
| G Web 前端 | 登录、聊天、订单、陪玩师、模拟支付 | |
| H 评测基座 | L2 数据集、L4 剧本、Runner | 大模型基线数据 |
| I 数据与训练 | Teacher 造数 → SFT → 对比集 → DPO 消融 → 量化 | **M2：小模型过 L2 门槛** |
| J 小模型接入 | Local 抽取器、降级、影子模式、compose | **M3：三配置对比表** |
| K 看板与飞轮 | 新增 3 页、样本挖掘、飞轮 1 轮 | **M4：闭环** |
| L 收尾 | 一键启动、README、演示脚本、简历素材 | 可交付 |

### 📊 进度跟踪表 (Progress Tracking)

| 阶段 | 任务 | 状态 |
|---|---|---|
| A | A1 A2 A3 A4 | ✅✅✅✅ |
| B | B1 B2 B3 B4 B5 B6 | ✅✅✅✅✅✅ |
| C | C1 C2 C3 C4 C5 C6 C7 C8 C9 | ✅✅✅✅✅✅✅✅✅ |
| D | D1 D2 D3 D4 D5 D6 D7 | ✅✅✅✅✅✅✅ |
| E | E1 E2 E3 E4 E5 E6 | ✅✅✅✅✅✅ |
| F | F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 | ✅✅✅✅✅✅✅✅✅✅ |
| G | G1 G2 G3 G4 G5 | ✅✅✅✅✅ |
| H | H1 H2 H3 H4 H5 | ✅✅✅✅✅ |
| I | I1 I2 I3 I4 I5 I6 I7 I8 I9 I10 I11 | ✅✅✅✅✅✅⬜⬜⬜⬜⬜ |
| J | J1 J2 J3 J4 J5 | ⬜⬜⬜⬜⬜ |
| K | K1 K2 K3 K4 K5 | ⬜⬜⬜⬜⬜ |
| L | L1 L2 L3 L4 | ⬜⬜⬜⬜ |

### 📈 总体进度

`58 / 77`

---

## 阶段 A：工程骨架与测试基座（目标：先可导入，再可测试） ✅

### A1：uv workspace 与目录骨架 ✅
- **目标**：按 5.2 创建目录与各 workspace 成员（`packages/common`、`packages/domain`、`services/booking_mcp`、`apps/agent`、`apps/web`、`training`），`services/knowledge_mcp` 暂留空目录。
- **修改文件**：`pyproject.toml`（workspace members）、各成员 `pyproject.toml`、`src/**/__init__.py`、`.gitignore`（含 `.env`、`data/*.db`、`training/data/raw`、`*.gguf`）、`.env.example`、`README.md`。
- **实现类/函数**：无。
- **验收标准**：`uv sync` 成功；`uv run python -c "import rift_common, rift_domain, booking_mcp, rift_agent, rift_web, rift_training"` 通过。
- **测试方法**：`uv run python -m compileall packages services apps training`。

### A2：pytest / ruff / mypy 基座 ✅
- **目标**：建立 `tests/unit|integration|e2e|fixtures` 与 markers（unit、integration、e2e、llm、slow）。
- **修改文件**：根 `pyproject.toml`（pytest、ruff、mypy 配置）、`tests/unit/test_smoke_imports.py`。
- **验收标准**：`uv run pytest -q` 通过；`uv run ruff check .` 无错误。
- **测试方法**：`uv run pytest -q tests/unit/test_smoke_imports.py`。

### A3：Settings 加载与环境变量展开 ✅
- **目标**：读取 `config/settings.yaml`，支持 `${ENV}` 展开，frozen dataclass/Pydantic 结构，缺字段 fail-fast 并报出字段路径。**密钥只允许来自环境变量**（YAML 中出现以 `sk-` 开头的字面量直接报错）。
- **修改文件**：`packages/common/src/rift_common/settings.py`、`config/settings.yaml`、`tests/unit/test_settings.py`。
- **实现类/函数**：`Settings`、`load_settings(path) -> Settings`、`expand_env(obj)`、`validate_settings(s)`。
- **验收标准**：缺 `llm.default.provider` 时报 `llm.default.provider is required`；YAML 写入明文 key 时报错；环境变量缺失时报出变量名。
- **测试方法**：`uv run pytest -q tests/unit/test_settings.py`。

### A4：GitHub Actions CI ✅
- **目标**：CI 运行 ruff、mypy（`packages/`）、`pytest -m "unit or integration"`，不注入任何密钥。
- **修改文件**：`.github/workflows/ci.yml`。
- **验收标准**：推送后 CI 绿（本地可用 `act` 或直接推送验证）。
- **测试方法**：本地执行与 CI 相同的命令序列全部通过。

---

## 阶段 B：公共库（目标：Provider 工厂可用，Trace 双写可用） ✅

### B1：LLM 抽象接口与注册式工厂 ✅
- **目标**：迁移 RAG 项目 `libs/llm` 的 `BaseLLM` + `LLMFactory` 思路：`register_provider()` 注册时 `issubclass` 校验，未知 provider fail-fast。接口：`chat(messages, *, response_format=None, temperature, max_tokens, timeout) -> LLMResult`，`astream(...)`。
- **修改文件**：`packages/common/src/rift_common/llm/{base.py, factory.py, types.py}`、`tests/unit/test_llm_factory.py`。
- **实现类/函数**：`BaseLLM`、`LLMResult(text, usage, latency_ms, model)`、`LLMFactory.create(cfg)`、`register_provider(name, cls)`。
- **验收标准**：注册非子类抛 `TypeError`；未知 provider 抛可读错误。
- **测试方法**：`uv run pytest -q tests/unit/test_llm_factory.py`。

### B2：OpenAI 兼容 Provider（DeepSeek 默认）+ Mock Provider ✅
- **目标**：实现 `openai_compatible`（支持 `response_format={"type":"json_object"}`、流式、超时）；实现 `mock`（按 fixture 文件 / 回调返回，用于单测与 CI）。
- **修改文件**：`llm/openai_compatible.py`、`llm/mock.py`、`tests/unit/test_llm_mock.py`、`tests/integration/test_deepseek_live.py`（marker `llm`）。
- **验收标准**：mock 可按输入匹配返回；真实 DeepSeek 调用（手动）能返回合法 JSON。
- **测试方法**：`uv run pytest -q tests/unit/test_llm_mock.py`；`uv run pytest -q -m llm tests/integration/test_deepseek_live.py`（需 key）。

### B3：llama_server Provider ✅
- **目标**：对接 llama.cpp OpenAI 兼容接口，temperature 0、`max_tokens` 默认 160、超时可配；健康检查 `ping()`。
- **修改文件**：`llm/llama_server.py`、`tests/unit/test_llama_server_provider.py`（用 `httpx.MockTransport`）。
- **验收标准**：超时抛 `LLMTimeout`；不可达抛 `LLMUnavailable`（供降级逻辑区分）。
- **测试方法**：`uv run pytest -q tests/unit/test_llama_server_provider.py`。

### B4：Embedding 抽象与本地实现 ✅
- **目标**：`BaseEmbedding.embed(texts) -> list[vector]`；实现 `local`（sentence-transformers 中文小模型，懒加载）、`openai_compatible`、`mock`（确定性哈希向量）。
- **修改文件**：`packages/common/src/rift_common/embedding/*`、`tests/unit/test_embedding.py`。
- **验收标准**：mock 相同文本向量相同、不同文本余弦 < 1；本地模型首次加载后缓存。
- **测试方法**：`uv run pytest -q tests/unit/test_embedding.py`。

### B5：Span Trace 核心与 SQLite Sink ✅
- **目标**：`TraceContext`（`trace_id`、`session_id`、层级 span 栈，基于 `contextvars`）、`Span(name, attrs, start, end, status, error)`、`@traced(name)` 装饰器（同步/异步皆可）；`SqliteSink` 在 turn 结束时批量写入 `turns`、`spans` 两表。
- **修改文件**：`trace/{context.py, span.py, decorators.py, sinks/base.py, sinks/sqlite.py}`、`tests/unit/test_trace.py`。
- **验收标准**：嵌套 span 父子关系正确；异常时 span 记录 error 并继续抛出；写入后可按 `session_id` 查询。
- **测试方法**：`uv run pytest -q tests/unit/test_trace.py`。

### B6：Langfuse Sink（可选、静默降级） ✅
- **目标**：将 turn 映射为 Langfuse trace（`session_id`、`user_id`），span 映射为 span/generation（LLM 调用带 input/output/usage）。`enabled=false` 或网络失败时仅 warning。
- **修改文件**：`trace/sinks/langfuse.py`、`tests/unit/test_langfuse_sink.py`（mock client）。
- **验收标准**：Langfuse 客户端抛异常时对话主流程不受影响；手动在 Langfuse Cloud 上可见一条测试 trace。
- **测试方法**：`uv run pytest -q tests/unit/test_langfuse_sink.py`；手动运行 `scripts/smoke_langfuse.py`。
- **实现备注**：Langfuse SDK v4 基于 OpenTelemetry，不支持回填 span 起止时间；旧的 `/api/public/ingestion` 批量接口将于 2026-11-16 在 Cloud 下线。因此 Sink 采用 span 处理器模式（`on_span_start / on_span_end / on_turn_end`）**实时镜像**，SQLite Sink 仍在 turn 结束时批量写入。`trace_id` 统一为 32 位 hex（OTel 兼容），跨服务可直接复用。

---

## 阶段 C：领域核心（目标：纯代码业务规则，L1 全绿，零 LLM） ✅

### C1：DomainConfig 与 domain.yaml ✅
- **目标**：编写 `config/domain.yaml`（模式、段位序、角色、服务类型、系数、退款档、时长范围、放宽顺序、必填规则、不适用字段、`max_tier_gap`、模式中文别名表）；加载为不可变对象并做一致性校验（如别名不重复、放宽顺序中不含 budget/rank）。
- **修改文件**：`config/domain.yaml`、`packages/domain/src/rift_domain/{config.py, enums.py}`、`tests/unit/test_domain_config.py`。
- **验收标准**：非法配置（系数为负、放宽含 budget）加载时报错。
- **测试方法**：`uv run pytest -q tests/unit/test_domain_config.py`。

### C2：槽位协议模型（SlotDelta / SlotExtraction / BookingState） ✅
- **目标**：Pydantic v2 严格模型，`extra="forbid"`；实现三态语义的类型表达（未出现 / `"any"` / `null` / 值），导出 JSON Schema 供 LLM 约束与训练使用。
- **修改文件**：`rift_domain/slots.py`、`tests/unit/test_slots_schema.py`。
- **实现类/函数**：`SlotDelta`、`SlotExtraction(turn_intent, delta, confirmation)`、`BookingState`、`ANY` 哨兵、`extraction_json_schema()`。
- **验收标准**：多余字段、非法枚举、`duration_hours=0.3` 均校验失败；"键缺失"与"值为 null"可区分（`model_fields_set`）。
- **测试方法**：`uv run pytest -q tests/unit/test_slots_schema.py`。

### C3：中文时间表达解析器 ✅
- **目标**：`parse_time_expr(expr, now, current=None) -> TimeParseResult(value|None, ambiguous: bool, reason)`。覆盖：今天/明天/后天/周X/下周X、上午/下午/晚上/凌晨、点/点半/X点Y分、"晚一小时/提前半小时"（相对 `current`）、"今晚""明晚"。**模糊表达（"晚上""找时间"）返回 ambiguous，不猜小时。**
- **修改文件**：`rift_domain/timeparse.py`、`tests/unit/test_timeparse.py`（≥ 40 条表驱动用例）。
- **验收标准**：用例全过；过去时间返回 `reason=past`。
- **测试方法**：`uv run pytest -q tests/unit/test_timeparse.py`。
- **实现备注**：无时段词的 1–11 点仅在 `h` 与 `h+12` 恰有一个在未来时才确定，否则返回 `ambiguous_ampm`；修改已有时间（传入 `current`）且未给日期时沿用原日期并取最接近 `current` 的上/下午；不带前缀的“周X”恰为今天且已过时顺延一周；同一表达式出现两个不同日期返回 `ambiguous_date`。

### C4：Delta 合并器 ✅
- **目标**：`merge(state, delta) -> (new_state, diff)`，实现三态语义与 list 整体替换；`start_time_expr` 触发时间解析；`companion_name` 变化时清空旧报价。
- **修改文件**：`rift_domain/merge.py`、`tests/unit/test_merge.py`。
- **验收标准**：未出现键不变、`any` 置 ANY、`null` 清空、list 替换；diff 准确列出变更字段。
- **测试方法**：`uv run pytest -q tests/unit/test_merge.py`。
- **实现备注**：签名为 `merge(state, delta, now)`（时间解析需要 `now`）。上一轮未解析的时间表达式会与本轮拼接再试（“明晚”+“八点”）；未解析/过去时间使 `start_time` 置空并在 `diff.time_result` 给出原因。除 `companion_name` 外，影响价格的槽位变化也会清空报价，过滤条件变化会清空候选、放宽记录与待确认动作。

### C5：业务规则（必填 / 不适用 / 服务类型推断） ✅
- **目标**：`compute_rules(state, domain) -> RulesResult(missing_fields, cleared_fields, service_type_effective)`；缺失字段按固定顺序输出。
- **修改文件**：`rift_domain/rules.py`、`tests/unit/test_rules.py`。
- **验收标准**：排位缺段位 → missing 含 `rank_requirement`；大乱斗带段位 → 被清空并记录；用户明说"教学"覆盖推断。
- **测试方法**：`uv run pytest -q tests/unit/test_rules.py`。
- **实现备注**：`RulesResult` 额外包含 `invalid_fields`（如时长超出 1–8h），这类字段被清空并计入 `missing_fields`；`apply_rules` 将结果写回状态。

### C6：匹配——硬过滤条件与软排序 ✅
- **目标**：`build_filters(state, domain) -> CompanionFilter`（纯数据，供 D 阶段转 SQL）；`rank_candidates(companions, state, style_scores) -> list[ScoredCandidate]`（位置命中、风格相似度、评分加权，权重在 `domain.yaml`）。
- **修改文件**：`rift_domain/matching.py`、`tests/unit/test_matching.py`。
- **验收标准**：单双排段位区间规则正确；`ANY` 字段不参与过滤；排序稳定且可解释（返回命中原因）。
- **测试方法**：`uv run pytest -q tests/unit/test_matching.py`。
- **实现备注**：预算按用户实付单价（`hourly_price × service_multiplier`）比较，SQL 侧使用 `CompanionFilter.max_base_price`；`rank_candidates` 额外接收 `domain` 与 `top_k`，只对适用的得分项归一化权重。

### C7：放宽策略 ✅
- **目标**：`relaxation_plan(filter, domain) -> list[(step_name, relaxed_filter)]`，按 `time±1h → gender → role` 逐级生成；budget 与 rank 永不出现在计划中。
- **修改文件**：`rift_domain/matching.py`（追加）、`tests/unit/test_relaxation.py`。
- **验收标准**：用户未指定的字段跳过对应步骤；计划累积放宽（第 2 步同时放宽时间与性别）。
- **测试方法**：`uv run pytest -q tests/unit/test_relaxation.py`。

### C8：计价与退款 ✅
- **目标**：`quote(hourly_price, service_type, hours, domain) -> Quote`；`refund(booking, now, domain) -> Refund(ratio, amount, tier)`；金额使用 `Decimal`，保留两位。
- **修改文件**：`rift_domain/{pricing.py, refund.py}`、`tests/unit/test_pricing_refund.py`。
- **验收标准**：三档退款边界（恰好 24h、恰好 2h）行为与 `domain.yaml` 描述一致。
- **测试方法**：`uv run pytest -q tests/unit/test_pricing_refund.py`。

### C9：领域核心覆盖率收口 ✅
- **目标**：`packages/domain` 行覆盖率 ≥ 90%，补齐边界用例；mypy strict 通过。
- **修改文件**：`tests/unit/*`、`pyproject.toml`（coverage 配置）。
- **验收标准**：`pytest --cov=rift_domain` ≥ 90%；`mypy --strict packages/domain` 无错误。
- **测试方法**：`uv run pytest -q --cov=rift_domain tests/unit`。

---

## 阶段 D：booking-mcp（目标：预约工具以 MCP 对外，Claude Desktop 可用） ✅

### D1：数据库模型与会话 ✅
- **目标**：SQLAlchemy 2.x 模型：`users(id, nickname)`、`companions(...)`、`schedules(companion_id, start, end, status, booking_id)`、`bookings(id, user_id, companion_id, start, end, game_mode, service_type, hours, unit_price, multiplier, total, status, created_at)`；状态机 `pending_payment → confirmed → completed | cancelled`。
- **修改文件**：`services/booking_mcp/src/booking_mcp/db/{models.py, session.py}`、`tests/integration/test_booking_db.py`。
- **验收标准**：建表成功；非法状态流转抛错。
- **测试方法**：`uv run pytest -q tests/integration/test_booking_db.py`。
- **实现备注**：金额/时长/系数以 `ScaledDecimal`（缩放整数）存储，SQLite 中可精确比较；陪玩师的位置/模式/服务类型拆为三张关联表便于 SQL 过滤，另存 `rank_tier` 用于段位区间比较。关闭 pysqlite 自带事务，由 `begin` 事件发 `BEGIN`；`write_session` 发 `BEGIN IMMEDIATE`。档期状态 `open / blocked / booked / released`：可用 = 被某个 `open` 窗口覆盖且不与 `blocked/booked` 重叠（首尾相接不算重叠）；取消时 hold 置为 `released` 以保留历史。

### D2：种子数据 ✅
- **目标**：生成 ≈ 30 名陪玩师（性别、段位、角色、可接模式、服务类型、单价、开麦、tags、bio、rating 分布合理）与未来 14 天档期（含部分已占用）；种子可重复生成（固定随机种子）。
- **修改文件**：`booking_mcp/seed.py`、`scripts/seed_all.py`、`tests/integration/test_seed.py`。
- **验收标准**：每种模式至少 5 人可接；每个段位区间都有覆盖；重复运行结果一致。
- **测试方法**：`uv run python scripts/seed_all.py --reset && uv run pytest -q tests/integration/test_seed.py`。
- **实现备注**：昵称列表顺序即 id 顺序；段位按固定配额分布（每个段位都有人），单双排由白银及以上承接，模式不足 `MIN_PER_MODE` 时补足；陪玩师自动具备其各模式默认服务类型，教学仅限钻石以上。另建演示用户 `demo`（id=1）。`seed_all.py` 不带 `--reset` 时不覆盖已有数据。

### D3：Repository 与冲突检测 ✅
- **目标**：`CompanionRepo.search(filter)`（CompanionFilter → SQL）、`ScheduleRepo.is_free(companion_id, start, end)`、`BookingRepo.create_atomic(...)`（`BEGIN IMMEDIATE` 事务内二次检测，冲突抛 `SlotConflict`）。
- **修改文件**：`db/repo.py`、`tests/integration/test_repo_conflict.py`。
- **验收标准**：两个线程并发创建同一时段，只有一个成功。
- **测试方法**：`uv run pytest -q tests/integration/test_repo_conflict.py`。
- **实现备注**：`CompanionRepo.static_query` 与 `rift_domain.matching.matches` 等价（测试逐例比对 SQL 路径与内存谓词）；预算换算为基础单价后向下取整到分再比较。时间窗放宽时以 30 分钟为步长尝试 `原时间 → +0.5h → -0.5h → …`，并排除早于当前时间的开始时间，返回实际可约的开始时间。`create_atomic` 若不在 `BEGIN IMMEDIATE` 事务内调用直接报错。

### D4：Tool：find_companions / quote_price ✅
- **目标**：`find_companions` = 硬过滤 → 风格相似度（embedding）→ 软排序 → 零结果按放宽计划重试，返回 `candidates + relaxations`；`quote_price` 调用 `rift_domain.pricing`。
- **修改文件**：`tools/{find_companions.py, quote_price.py}`、`tests/integration/test_tool_find.py`。
- **验收标准**：构造"女 + 钻石 + 周六 20 点无人"场景，返回放宽后的结果并列出 `relaxations=["time_window"]`；预算过低时返回空列表而非放宽预算。
- **测试方法**：`uv run pytest -q tests/integration/test_tool_find.py`。
- **实现备注**：工具为纯函数 `(BookingService, 输入模型) -> 输出模型`，`BookingService` 持有 DB、领域配置、可注入时钟与可选 embedder（风格向量按陪玩师缓存，embedding 失败则不计风格分）。入参先经 `apply_rules` 去掉不适用字段（如大乱斗的段位）。放宽耗尽仍无人时返回空列表并在 `relaxations_tried` 中列出尝试过的步骤。金额以两位小数字符串输出。

### D5：Tool：create_booking / list_my_bookings / cancel_booking ✅
- **目标**：`create_booking` 写订单与档期；`cancel_booking(dry_run)` 计算退款，非 dry_run 时释放档期并置 `cancelled`；越权（他人订单）返回 `FORBIDDEN`。
- **修改文件**：`tools/{create_booking.py, list_my_bookings.py, cancel_booking.py}`、`tests/integration/test_tool_booking.py`。
- **验收标准**：创建 → 列表可见 → dry_run 报退款 → 取消 → 档期释放。
- **测试方法**：`uv run pytest -q tests/integration/test_tool_booking.py`。
- **实现备注**：`cancel_booking` 默认 `dry_run=true`；未支付订单取消不退钱（`refund_amount=0`），但仍返回按 C8 计算的 `policy_refund_amount` 与档位，便于 Agent 如实说明。已取消/已完成订单再取消返回 `INVALID_ARGUMENT`。

### D6：MCP Server（stdio + Streamable HTTP）与错误映射 ✅
- **目标**：官方 SDK 注册 5 个工具（JSON Schema 入参）；`--transport stdio|http --port 8101`；日志只写 stderr；业务异常映射为错误码；读取 `_meta.trace_id` 创建子 span。
- **修改文件**：`booking_mcp/server.py`、`tests/integration/test_booking_mcp_protocol.py`（进程内 client）。
- **验收标准**：`tools/list` 返回 5 个工具；冲突返回 `SLOT_CONFLICT`。
- **测试方法**：`uv run pytest -q tests/integration/test_booking_mcp_protocol.py`。
- **实现备注**：MCP SDK 固定为 `mcp>=1.9,<2`（RAG 项目基于 1.x，同一 venv 需统一；2.x 已重命名/重构 API）。SDK 1.x 会把工具内的任何异常转成 `isError` 结果，因此业务错误以工具错误返回，结构化内容为 `{"error": {code, jsonrpc_code, message, details}}`（`SLOT_CONFLICT -32001 / NOT_FOUND -32002 / FORBIDDEN -32003 / INVALID_ARGUMENT -32602 / INTERNAL -32603`），LLM 客户端可读、Agent 可映射回异常。HTTP 模式为无状态 + JSON 响应，端点 `/mcp`。`_meta.trace_id`（及可选 `parent_span_id`、`session_id`）存在时在服务端延续该 trace，生成 `tool:<name>` span。

### D7：Claude Desktop 联调 ✅
- **目标**：提供 `claude_desktop_config` 示例片段，在 Claude Desktop 中通过 stdio 完成"找陪玩 → 报价 → 下单"。
- **修改文件**：`README.md`（MCP 配置小节）、`docs/mcp_desktop.md`。
- **验收标准**：手动演示成功并截图存入 `docs/img/`。
- **测试方法**：手动。
- **实现备注**：配置片段与错误码见 `docs/mcp_desktop.md`；`scripts/mcp_smoke.py` 以 stdio 子进程方式自动跑通 找人 → 报价 → 下单 → 查订单 → 退款试算（CI 覆盖）。已在 Claude Desktop 实机完成 找陪玩（`style_preference=温柔` 排序）→ 报价 → 下单（待支付），截图见 `docs/img/mcp-desktop-0{1..4}-*.png`。联调中发现输出 Schema 把本地时间标为 `date-time`（要求时区偏移），严格校验的客户端会拒收，已改为本地时间字符串模式。

---

## 阶段 E：knowledge-mcp（目标：RAG 迁入、Markdown 知识库、L3 达标）

### E1：迁入 RAG 源码 ✅
- **目标**：将 `MODULAR-RAG-MCP-SERVER` 的 `src/`、`config/`（去除任何明文 key，改为 `${ENV}`）、相关测试复制到 `services/knowledge_mcp/`，作为 workspace 成员；原有 unit 测试在新位置通过。
- **修改文件**：`services/knowledge_mcp/**`、根 `pyproject.toml`。
- **验收标准**：`uv run pytest -q services/knowledge_mcp/tests/unit` 通过；仓库内 `grep -r "sk-" services/knowledge_mcp/config` 无结果。
- **测试方法**：同上。
- **实现备注**：`src/`、`scripts/`、`tests/`、`config/prompts` 原样复制（包名保持 `src`，hatchling 打包），`config/settings.yaml` 由原文件生成：三处明文 key 替换为 `${OPENAI_API_KEY:-}`，`vision_llm` 关闭；`load_settings` 新增 `${VAR}` / `${VAR:-default}` 展开（未设置时为空串，离线可加载）。补齐原项目未声明的依赖（`pymupdf`、`pillow`），`langchain-community` 固定 `<0.4`（ragas 0.4.3 需要）。原项目在源仓库中本就有 13 个单测失败，均为测试与代码漂移：Windows 下 `time.time()`/`monotonic()` 精度导致耗时为 0（改用 `perf_counter`）、`list_collections` 位置参数调用、embedding 测试用裸 `Mock` 当配置、分词测试仍按旧正则分词器断言、trace 测试夹具格式过旧——已逐一修正，1212 个单测全部通过并加入 CI。另为 `*.pdf`/`*.png` 等加 `binary` 属性，防止 git 改写 PDF 夹具换行。

### E2：MarkdownLoader ✅
- **目标**：按标题层级切分 section，metadata 保留 `source`、`title_path`、`collection`；通过 loader 工厂注册，ingestion pipeline 按扩展名选择 loader；SHA256 增量检查沿用。
- **修改文件**：`services/knowledge_mcp/src/libs/loader/markdown_loader.py`、loader 工厂、`tests/unit/test_markdown_loader.py`。
- **验收标准**：多级标题文档切分正确；重复导入未变更文件被跳过。
- **测试方法**：`uv run pytest -q services/knowledge_mcp/tests/unit/test_markdown_loader.py`。
- **实现备注**：`MarkdownLoader` 仍按 `BaseLoader` 契约返回单个 `Document`，section 列表放在 `metadata["sections"]`（`title_path` 用 ` > ` 连接成字符串，便于 Chroma 存储与过滤；忽略代码块内的 `#`；无正文的容器标题不单独成 section）。`DocumentChunker` 检测到 `sections` 时逐 section 切分，chunk 不跨 section 且带 `title_path`、`section_index`，长 section 的后续片段补上标题行。新增 `LoaderFactory`（`.pdf` / `.md` / `.markdown` 注册）与 `ExtensionRoutingLoader`，pipeline 用它按扩展名分派，`scripts/ingest.py` 默认扫描所有已注册扩展名。增量检查仍由 pipeline 的 SHA256 完成，测试以真实 SQLite 记录验证未变更文件被跳过、修改后重新导入。

### E3：知识库文档生成（唯一数据源） ✅
- **目标**：`gen_kb_docs.py` 从 `domain.yaml` 生成 `kb/platform_rules/*.md`（计费、退款、迟到补偿、违规、陪玩师等级）、`kb/modes_and_ranks/*.md`，从 `companions` 种子生成 `kb/companion_profiles/*.md`；使用 Jinja2 模板，数值全部插值。
- **修改文件**：`scripts/gen_kb_docs.py`、`scripts/kb_templates/*.md.j2`、`tests/unit/test_gen_kb_docs.py`。
- **验收标准**：修改 `domain.yaml` 中系数后重新生成，文档数值同步变化（测试断言）。
- **测试方法**：`uv run pytest -q tests/unit/test_gen_kb_docs.py`。
- **实现备注**：迟到补偿、违规处理、陪玩师等级原本不在 `domain.yaml`，为保持唯一数据源新增 `policies` 节（`DomainConfig.policies`，含 `level_for(rating)`）。计价与退款示例直接调用 `rift_domain.pricing.quote` / `refund.refund` 计算；共生成 39 篇（平台规则 6、模式与段位 3、陪玩师 30）。`kb/` 为生成物，已加入 `.gitignore`。

### E4：知识库导入与配置调整 ✅
- **目标**：`ingest_kb.py` 按三个 collection 导入；`vision_llm` 与图片 transform 关闭；Embedding 与 LLM 使用 DeepSeek / 本地 embedding 配置。
- **修改文件**：`scripts/ingest_kb.py`、`services/knowledge_mcp/config/settings.yaml`。
- **验收标准**：`list_collections` 返回 3 个集合；`query_knowledge_hub("开局前三小时取消退多少")` Top-3 命中退款文档。
- **测试方法**：`uv run python scripts/ingest_kb.py && uv run python services/knowledge_mcp/scripts/query.py --query "..."`。
- **实现备注**：DeepSeek 没有 embedding 接口，knowledge-mcp 新增 `local` embedding provider（sentence-transformers，`BAAI/bge-small-zh-v1.5`，512 维，懒加载、按模型共享；维度与配置不符时直接报错），依赖放在 `local-embedding` extra（`uv sync --extra local-embedding`，CI 不装）；LLM 改为 DeepSeek。`chunk_refiner` / `metadata_enricher` 的 LLM 模式关闭：知识库由 `domain.yaml` 生成，LLM 改写可能改动数字；`vision_llm` 关闭时 ImageCaptioner 直接跳过。`ingest_kb.py` 按目录导入三个集合，未变更文件靠 SHA256 跳过；文件修改（或 `--force`）时先删除旧 chunk 再导入，文件删除时从集合中移除。默认集合为 `platform_rules`（`query_knowledge_hub` 与 `query.py` 不传集合时使用）。实测：39 篇 → 123 个 chunk（平台规则 22、模式与段位 11、陪玩师 90）；`list_collections` 返回 3 个集合；"开局前三小时取消退多少" Top-3 全部来自 `refund.md`（退款档位、退款示例、陪玩师原因）。导入时发现并修复了几个原有问题：BM25 posting 用的是向量 ID（`{sha256(路径)[:8]}_序号_内容哈希`），而重新导入和 `delete_document` 分别按 doc id、文件哈希删旧 posting，一直删不掉，文档改过之后旧内容仍能被 BM25 检出，现在统一按向量 ID 前缀删除；`list_collections`、`get_document_summary`、`query.py` 把 Chroma / BM25 的相对路径按当前工作目录解析，从仓库根目录运行时找不到数据，现在统一按服务根目录解析。

### E5：Streamable HTTP 模式与 trace_id 关联 ✅
- **目标**：新增 `--transport http --port 8102`；工具调用读取 `_meta.trace_id` 写入 RAG trace 的 `parent_trace_id`。
- **修改文件**：`services/knowledge_mcp/src/mcp_server/server.py`、trace 相关、`tests/integration/test_knowledge_http.py`。
- **验收标准**：HTTP 客户端可调用三个工具；trace 文件中出现 `parent_trace_id`。
- **测试方法**：`uv run pytest -q tests/integration/test_knowledge_http.py`。
- **实现备注**：新增 `knowledge-mcp` 命令（`--transport stdio|http --host --port`，默认 stdio / 8102），HTTP 与 booking-mcp 一致：无状态 Streamable HTTP、JSON 响应、路径 `/mcp`。`tools/call` 处理函数读取 `_meta.trace_id` / `_meta.parent_span_id`，放入请求级 `ContextVar`（`caller_trace`）；调用期间新建的 `TraceContext` 自动带上 `parent_trace_id` / `parent_span_id`，写入 `logs/traces.jsonl`（没有 `_meta` 时不写这两个字段）。`TraceCollector` 的默认路径改为使用时再解析，便于测试和换根目录。集成测试现场生成 3 篇 Markdown，用确定性的字符 bigram 哈希 embedding 走真实导入流程（Chroma + BM25），再用 uvicorn 起 HTTP 服务调用三个工具，全程离线。另用真实知识库和 bge 手动验证：三个工具都能调用，trace 中的 `parent_trace_id` 与调用方一致。

### E6：L3 RAG 评测集与达标 ✅
- **目标**：从 `domain.yaml` 与种子自动生成 ≈ 30 条 golden QA（问题模板 + 标准答案 + 期望来源文档）；运行 Hit Rate / MRR / ragas。
- **修改文件**：`eval/datasets/rag_golden.jsonl`、`eval/runners/run_rag_eval.py`。
- **验收标准**：Hit@3 ≥ 0.9，MRR ≥ 0.8（首次跑出后可按实际调整并记录）。
- **测试方法**：`uv run python eval/runners/run_rag_eval.py`（加 `--ragas` 需 `DEEPSEEK_API_KEY`：`uv run --env-file .env python eval/runners/run_rag_eval.py --ragas`）。
- **实现备注**：`scripts/gen_rag_golden.py` 从 `domain.yaml` + 陪玩师种子生成 34 条 QA（平台规则 20、模式与段位 7、陪玩师 7），标准答案里的金额与比例用 `rift_domain.pricing` / `refund` 计算，并写出 `.sha256`；runner 会检查 checksum，还会在内存中重新生成一遍比对，过期时给出警告，单测在 CI 中守住这一点。评测方式与 Agent 的 consult 一致：每题只在自己的集合内检索（`QueryKnowledgeHubTool.retrieve`）；Hit@3 指期望文档的某个 chunk 排进前 3，MRR 取前 10 名内第一个命中 chunk 的倒数名次。ragas 的做法：用 Agent 的 `consult_answer.txt` 提示词、基于前 3 个 chunk 让 DeepSeek 作答，再由 DeepSeek 当评审，embedding 用本地 bge。为此 RagasEvaluator 新增了 deepseek 分支，并能调用项目自带的 embedding 工厂；评审用的异步客户端改为在事件循环内关闭（原来会报 "Event loop is closed"）。**首次结果**（`eval/reports/rag_eval_baseline.json`）：Hit@3 = 1.000、MRR = 0.971（平台规则中计费公式、默认服务类型两题的正确文档排第 2），faithfulness = 0.892，answer relevancy = 0.858。平台规则 faithfulness 最低（0.833）：评审对"答案里出现资料中没有的计算结果"（如 288.00 元、退 100.00 元）和同义改写（"信用分扣10分" 对 "-10"）判得较严，这几题的答案本身是正确的。题目问法接近文档原文，检索满分偏乐观，之后可加入口语化或错别字问法再测。带时间戳的报告不进 git。退款规则改为「≥ 15 分钟全额、不足 15 分钟不退」后，重新生成题库并重跑（基线已覆盖）：Hit@3 = 1.000、MRR = 0.971、faithfulness = 0.900、answer relevancy = 0.862。

---

## 阶段 F：Agent 编排（目标：M1 —— 大模型版端到端可演示） ✅

### F1：Graph State 与节点装饰器 ✅
- **目标**：定义 LangGraph `AgentState`（phase、booking_state、pending_action、messages 窗口、last_reply、turn meta）；`@traced_node` 装饰器把节点接入 B5 Trace。
- **修改文件**：`apps/agent/src/rift_agent/graph/{state.py, tracing.py}`、`tests/unit/test_agent_state.py`。
- **验收标准**：State 可序列化进 SqliteSaver；节点异常被 span 记录。
- **测试方法**：`uv run pytest -q tests/unit/test_agent_state.py`。
- **实现备注**：State 全部为 JSON 原生类型（槽位以 `BookingState.model_dump(mode="json")` 存于 `booking`），测试逐值断言无枚举/Decimal/datetime 对象，避免 checkpointer 自定义序列化。运行时依赖（LLM、抽取器、MCP 客户端、模板）经 `config["configurable"]["deps"]` 注入，不进 checkpoint。`traced_node` 把 `interrupt()` 记为正常 span（`interrupted=true`），其余异常转为 `error` 回复且状态不变。

### F2：MCP 客户端封装 ✅
- **目标**：`McpClients`（booking、knowledge），Streamable HTTP，自动注入 `_meta.trace_id`，错误码转为 Python 异常；测试模式支持进程内 server。
- **修改文件**：`rift_agent/mcp_clients.py`、`tests/integration/test_mcp_clients.py`。
- **验收标准**：对真实 booking-mcp 调用 `find_companions` 成功；`SLOT_CONFLICT` 转为 `SlotConflictError`。
- **测试方法**：`uv run pytest -q tests/integration/test_mcp_clients.py`。
- **实现备注**：每次调用新建短连接（服务端无状态），天然适配 LangGraph 任务；`_meta` 注入 `trace_id` / `parent_span_id` / `session_id`。进程内连接器用于测试与单进程演示；为此 booking-mcp 处理请求时先 `detached()` 脱离调用方 trace 上下文（修复同进程嵌套 turn 报错）。知识库客户端 `query_all` 依次检索 `settings.mcp.knowledge.collections` 中的全部集合。

### F3：意图分类节点 ✅
- **目标**：`classify` 使用 `config/prompts/classify.txt`，DeepSeek JSON 输出 `booking|consult|manage|other`；解析失败回退 `other`。
- **修改文件**：`graph/nodes/classify.py`、`config/prompts/classify.txt`、`tests/unit/test_classify_node.py`（mock LLM）。
- **验收标准**：mock 下四类路由正确；解析失败不抛异常。
- **测试方法**：`uv run pytest -q tests/unit/test_classify_node.py`。

### F4：SlotExtractor 接口与 LLMSlotExtractor ✅
- **目标**：`SlotExtractor.extract(ctx) -> ExtractionResult(extraction|None, raw, valid, errors, model, latency)`；`build_prompt(ctx)` 读取 `slot_extract.txt`（线上/训练共用）并注入 `current_state`、`candidates`、历史、当前时间；`LLMSlotExtractor` 用 DeepSeek JSON 模式 + C2 校验。
- **修改文件**：`rift_agent/extractors/{base.py, prompt.py, llm.py}`、`config/prompts/slot_extract.txt`、`tests/unit/test_llm_extractor.py`。
- **验收标准**：非法 JSON / schema 错误时 `valid=False` 且 errors 可读；prompt 渲染确定（同输入同输出，便于前缀缓存）。
- **测试方法**：`uv run pytest -q tests/unit/test_llm_extractor.py`。
- **实现备注**：system 为静态 `slot_extract.txt`（前缀可缓存），user 为 `sort_keys` 的上下文 JSON；抽取器从不抛异常，返回 `error_kind ∈ {invalid, timeout, unavailable}` 供路由区分重试与降级。

### F5：RoutedSlotExtractor（重试 / 降级 / 影子） ✅
- **目标**：实现 3.4.2 路由；shadow 调用在后台任务中执行并写入 span 属性 `shadow_output`、`shadow_agree`（逐字段比对结果）。
- **修改文件**：`extractors/routed.py`、`tests/unit/test_routed_extractor.py`。
- **验收标准**：primary 校验失败 → 重试 1 次 → fallback；primary 超时直接 fallback；shadow 失败不影响主结果。
- **测试方法**：`uv run pytest -q tests/unit/test_routed_extractor.py`。
- **实现备注**：shadow 与 primary 并发，主结果返回后最多再等 `shadow_wait_s`（默认 2s），超时即取消并记 `shadow_error=timeout`；`shadow_agree` 为逐字段比对（`turn_intent`、`confirmation`、`delta.<key>`）。`local` 抽取器在 J1 接入，当前配置为 local 时 fail-fast。

### F6：预约主链路节点 ✅
- **目标**：`extract_slots → merge_state → resolve_time → compute_rules → decide → ask_missing | find_companions → present_candidates`；`decide` 输出 `reply_type` 与动作。
- **修改文件**：`graph/nodes/{extract.py, merge.py, decide.py, find.py}`、`tests/integration/test_graph_booking.py`（mock LLM + 进程内 booking-mcp）。
- **验收标准**：3 轮对话从零收集到展示候选；模糊时间触发追问具体几点。
- **测试方法**：`uv run pytest -q tests/integration/test_graph_booking.py`。
- **实现备注**：`resolve_time` 由 C4 `merge` 完成，`merge_state` 同时执行 `apply_rules`。未解析的时间表达在后续轮次仍会被追问（"「明晚」具体是几点"）。回复模板用宽松 undefined（可选 facts 缺省为空），由集成测试覆盖输出。builder 中目标节点未注册的路由统一落到 `render`，便于逐任务扩展。

### F7：选择、报价与确认断点 ✅
- **目标**：候选选择 → `quote_price` → `confirm` 节点 `interrupt()`；resume 时根据 `confirmation` 与 delta：yes → `create_booking`；no → 回到候选；带 delta → 重新合并与报价；冲突 → 提示并重新 `find_companions`。
- **修改文件**：`graph/nodes/{select.py, confirm.py, book.py}`、`tests/integration/test_graph_confirm.py`。
- **验收标准**：确认前数据库无订单；"改成三小时再下单"重新报价；冲突场景给出替代候选。
- **测试方法**：`uv run pytest -q tests/integration/test_graph_confirm.py`。
- **实现备注**：先 `render` 确认卡再进入 `confirm` 节点 `interrupt()`，因此确认问题属于本轮回复；下一句经 `Command(resume=...)` 进入抽取。修改槽位会使 C4 清空候选与选择，`merge_state` 记录 `keep_companion_id`，重新检索时若原陪玩师仍可约则自动重选并重新报价（"改成三小时再下单"）。

### F8：咨询与插话 ✅
- **目标**：`consult`（IDLE 下独立咨询）与 `consult_interject`（BOOKING 下插话）：`query_knowledge_hub` → DeepSeek 基于检索结果带引用作答（`consult_answer.txt`）；插话后按 `missing_fields` 追加续约提示，phase 保持。`unrelated` → 礼貌拉回；"不约了" → 归档并回 IDLE。
- **修改文件**：`graph/nodes/{consult.py, unrelated.py, abandon.py}`、`config/prompts/consult_answer.txt`、`tests/integration/test_graph_interject.py`。
- **验收标准**：插话后下一句"那就两小时吧"被正确并入原预约。
- **测试方法**：`uv run pytest -q tests/integration/test_graph_interject.py`。
- **实现备注**：检索不可用 / 无结果 / LLM 失败分别降级为提示语、"暂未查到"、直接给出最相关片段；插话回复末尾按缺失字段、待选候选或待确认订单生成续约提示。

### F9：订单管理流程 ✅
- **目标**：`manage_bookings`：列出订单 → 用户选择 → `cancel_booking(dry_run)` 报退款 → `interrupt` 确认 → 执行；支持"我有哪些订单"。
- **修改文件**：`graph/nodes/manage.py`、`tests/integration/test_graph_manage.py`。
- **验收标准**：退款金额与 C8 计算一致；取消后档期释放。
- **测试方法**：`uv run pytest -q tests/integration/test_graph_manage.py`。
- **实现备注**：订单指代与是/否判断为确定性规则（序号、订单号、陪玩师名、今天/明天）；取消属破坏性操作，"否"类词优先，带问号的回答只有含"确认/确定"才视为同意，其余一律再次确认。未支付订单取消时如实说明"不产生费用"。

### F10：回复渲染、Graph 组装与持久化 ✅
- **目标**：模板渲染（`replies.yaml`，按 `reply_type`）+ 可选润色（`reply_polish.txt`，失败回退模板）；开局提醒模板；`builder.py` 组装全图，接入 `SqliteSaver`；提供 `run_turn(session_id, user_id, text) -> AsyncIterator[str]` 流式接口；CLI `scripts/chat_cli.py`。
- **修改文件**：`rift_agent/replies/*`、`graph/builder.py`、`rift_agent/api.py`、`scripts/chat_cli.py`、`tests/integration/test_graph_persistence.py`。
- **验收标准**：重建 graph 实例后同一 `session_id` 续聊状态不丢；`polish=true` 时润色失败自动回退模板。**M1：CLI 中用 DeepSeek 完成一次完整预约 + 一次插话咨询 + 一次取消。**
- **测试方法**：`uv run pytest -q tests/integration/test_graph_persistence.py`；手动 `uv run python scripts/chat_cli.py`。
- **实现备注**：`Agent`（`rift_agent.api`）提供 `run` / `run_turn_events`（token、candidates、confirm、booked、done、error 事件，供 G2 SSE）/ `run_turn`（纯文本流）；同一会话串行执行。润色结果必须保留模板中的全部数字，否则回退模板。`scripts/chat_cli.py` 支持 `--booking-db` 进程内运行 booking-mcp。**M1 验证**：已用 DeepSeek 实测 预约（抽取、候选引用、报价确认、下单）→ 查订单 → 取消 全流程；插话咨询的真实检索需 knowledge-mcp（阶段 E1 待迁入），目前验证的是检索不可用时的降级回复。

---

## 阶段 G：Web 前端（目标：可演示的用户界面） ✅

### G1：FastAPI 应用骨架与昵称登录 ✅
- **目标**：`create_app()` 工厂 + `lifespan`（替代已废弃的 `on_event`）；昵称登录 → 创建/查找 `users` → 签名 cookie 存 `user_id`；每个浏览器标签页生成 `session_id`；CORS 仅允许本地来源。
- **修改文件**：`apps/web/src/rift_web/{app.py, auth.py, routes/auth.py}`、`templates/login.html`、`tests/integration/test_web_auth.py`。
- **验收标准**：未登录访问聊天页重定向到登录页；cookie 篡改被拒绝。
- **测试方法**：`uv run pytest -q tests/integration/test_web_auth.py`。
- **实现备注**：登录经 booking-mcp 新增的 `ensure_user` 工具创建/查找用户（Web 不直连数据库）。cookie 由 `itsdangerous` 签名（`RIFT_WEB_SECRET`），HttpOnly + SameSite=Lax；`next` 只接受站内相对路径，防开放重定向。浏览器标签页的 `session_id` 在服务端以 `u<uid>-<session_id>` 作为 Agent 线程 ID，他人无法凭 session_id 读取会话。

### G2：聊天页与 SSE 流式 ✅
- **目标**：`POST /api/chat` 返回 SSE（事件类型：`token`、`candidates`、`confirm`、`done`、`error`）；前端渲染候选卡片（点击即发送"选 xxx"）与确认卡片（确认 / 取消按钮）。
- **修改文件**：`routes/chat.py`、`templates/chat.html`、`static/chat.js`、`tests/integration/test_web_chat.py`（mock agent）。
- **验收标准**：错误以 `error` 事件下发（不再用 `[ERROR]` 字符串混入正文）；候选与确认卡片正确渲染。
- **测试方法**：`uv run pytest -q tests/integration/test_web_chat.py`；浏览器手动验证。
- **实现备注**：`POST /api/chat` 基于 F10 的 `Agent.run_turn_events` 输出 SSE（`token`/`candidates`/`confirm`/`booked`/`done`/`error`），异常一律以 `error` 事件下发；前端用 fetch 读取流（EventSource 不支持 POST），候选卡片"选这位"、确认卡片"确认下单/换一位"直接发送对应话术；刷新后通过 `/api/chat/history` 恢复文本记录。已用 DeepSeek + 进程内 booking-mcp 在浏览器中走通 预约 → 确认 → 下单。

### G3：我的订单页与模拟支付 ✅
- **目标**：订单列表（状态徽标）、"模拟支付"按钮（`pending_payment → confirmed`）、取消按钮（展示退款金额后确认）。页面数据通过 booking-mcp 获取（与 Agent 一致，不直连 DB）。
- **修改文件**：`routes/bookings.py`、`templates/bookings.html`、`static/bookings.js`、`tests/integration/test_web_bookings.py`。
- **验收标准**：支付与取消后状态实时刷新。
- **测试方法**：`uv run pytest -q tests/integration/test_web_bookings.py`。
- **实现备注**：booking-mcp 新增 `pay_booking` 工具（`pending_payment → confirmed`，越权 FORBIDDEN）；取消先 `dry_run` 展示退款再执行，操作后列表实时刷新。浏览器已验证 支付 → 退款预估 → 取消。

### G4：陪玩师列表页 ✅
- **目标**：筛选（模式、段位、位置、性别）+ 卡片展示（价格、标签、评分、近 3 天空闲时段）。
- **修改文件**：`routes/companions.py`、`templates/companions.html`。
- **验收标准**：筛选结果与 `find_companions` 硬过滤一致。
- **测试方法**：手动 + `tests/integration/test_web_companions.py`。
- **实现备注**：booking-mcp 新增 `browse_companions` 工具：选定模式时与 `find_companions` 使用同一套 `apply_rules` + `build_filters` + SQL 硬过滤（测试在多组筛选上逐一比对），未选模式时只按位置/性别过滤；卡片含各服务类型实际单价、等级与近 3 天可约时段（不短于最短时长）。"约 TA"跳转聊天页并预填输入。至此 booking-mcp 共 8 个工具：Agent 使用的 5 个 + Web 支撑的 3 个（`ensure_user`、`pay_booking`、`browse_companions`）。

### G5：前端打磨与移动端适配 ✅
- **目标**：统一样式（深色峡谷主题）、移动端宽度可用、加载与错误态、"新会话"按钮。
- **修改文件**：`static/*.css`、`templates/base.html`。
- **验收标准**：375px 宽度无横向滚动；截图存入 `docs/img/`。
- **测试方法**：手动（浏览器 DevTools）。
- **实现备注**：深色峡谷主题（金色强调）、`100dvh` 聊天布局、网格 `minmax(min(100%, …))` 在窄屏自动单列、长文本 `overflow-wrap`、键盘焦点环与 reduced-motion。已在浏览器 375×812 视口逐页验证 `scrollWidth == 375`（登录后四个页面）。截图见 `docs/img/web-01-login.png`。

---

## 阶段 H：评测基座（目标：大模型基线数据） ✅

### H1：L2 评测集编写（人工） ✅
- **目标**：编写 `slot_main.jsonl`（≈ 60）与 `slot_holdout.jsonl`（≈ 25），覆盖 3.7 所列全部场景类别；每条包含输入上下文（now、current_state、candidates、history、user_input）与期望 `SlotExtraction`。由 Claude 起草、**人工逐条审核**，记录 checksum 后冻结。
- **修改文件**：`eval/datasets/{slot_main.jsonl, slot_holdout.jsonl, CHECKSUMS}`、`eval/datasets/README.md`（类别分布表）。
- **验收标准**：所有样本通过 C2 schema；类别分布表中每类 ≥ 3 条；holdout 与 main 无重复用户输入。
- **测试方法**：`uv run python eval/runners/validate_datasets.py`。
- **实现备注**：主集 60 条、holdout 27 条（每类 3 条，比 ≈ 25 多两条，保证每类 ≥ 3），9 个类别的分布见 `eval/datasets/README.md`，标注约定也写在那里：时间照抄原话；服务类型只在用户明说时标；"不限" 标 any、"先不定" 标 null；带修改的确认标 `delta + yes`。样本模型 `rift_training.evaluation.dataset.SlotSample` 用 C2 严格 schema 校验 `expected`，用 `BookingState` 校验 `current_state`。校验器还检查：候选引用的名字必须在 `candidates` 里；consult / unrelated 轮次的 delta 为空；`yes` 只出现在待确认时；`history` 以助手消息结尾；main 与 holdout 没有相同的用户输入。31 个时间表达都已确认能被 C3 解析到预期时刻。`CHECKSUMS` 为 sha256sum 格式，用 LF 写出。**审核状态**：样本由 Claude 起草，仍需人工逐条审核；审核时如有改动，重跑 `--write-checksums` 后再冻结。

### H2：L2 评分器 ✅
- **目标**：迁移 slot-extractor 评分思路：协议层（JSON、schema、枚举）+ 任务层（`turn_intent`、`confirmation`、delta 逐字段：键集合一致性 + 值一致；`style_preference` 用本地 embedding 相似度阈值；`start_time_expr` 通过 C3 解析后比较绝对时间）。
- **修改文件**：`training/src/rift_training/evaluation/{protocol.py, task.py, preferences.py}`、`tests/unit/test_scorers.py`。
- **验收标准**：多写一个键（本应"未出现"）判错；`"明晚8点"` 与 `"明天晚上八点"` 解析一致判对。
- **测试方法**：`uv run pytest -q tests/unit/test_scorers.py`。
- **实现备注**：协议层把错误分为 `empty / json / not_object / extra_key / missing_key / enum / type` 几类。与线上解析一致，不剥离 Markdown 代码块；`1` 对应 `bool | "any"` 时算类型错误，未知字符串算枚举错误。任务层逐字段比较 `turn_intent`、`confirmation` 以及任一方输出过的所有 delta 键，只在一方出现的键记为 `missing_key` 或 `extra_key`。各字段的比较方式：时间按与 `merge` 完全相同的方式解析（考虑 `now`、当前 `start_time` 和之前未能解析的表达式），解析不出来时比较规范化后的文本；`role_preference` 按集合比较；`style_preference` 用 `EmbeddingStyleMatcher`（本地 bge，规范化后相同直接判对）。风格阈值 0.7 是在种子风格标签上标定的：同义说法的相似度 ≥ 0.74，不同风格 ≤ 0.59。任务分 = 正确字段数 / 比较字段数；协议通过且任务分 ≥ 0.95 才算通过，由于每条样本只有 2–7 个字段，实际上要求全部字段正确。`rift-training` 新增依赖 `rift-common`，`training/src` 纳入 CI 的 mypy 检查。

### H3：L2 Runner 与报告 ✅
- **目标**：`run_slot_eval.py --extractor llm|local|routed --dataset main|holdout`，输出 JSON + Markdown 报告（通过率、逐字段错误分布、错误样例、延迟分布）。
- **修改文件**：`eval/runners/run_slot_eval.py`、`eval/reports/`。
- **验收标准**：DeepSeek 基线报告生成并提交。
- **测试方法**：`uv run python eval/runners/run_slot_eval.py --extractor llm --dataset main`。
- **实现备注**：每条样本都构造与 `extract_slots` 节点相同的 `ExtractionContext`，因此 prompt 与线上逐字节一致。`llm` / `local` 只调用一次、不重试，衡量的是模型本身；`routed` 走 M3 路由（local 为主，失败重试，再降级到 DeepSeek）。`rift_agent.factory` 新增 `EXTRACTOR_CONFIGS`、`with_extractor_config` 和 `build_single_extractor`；其中 `local` 暂时是连接 llama-server OpenAI 兼容接口的 `LLMSlotExtractor`（name=`local`），J1 会换成专用的 `LocalSlotExtractor`。报告分 JSON（逐条样本）和 Markdown（总览、分类别、协议错误、逐字段错误分布、错误样例、延迟分位数）两份；`--out` 可指定文件名，带时间戳的报告不进 git。数据集 checksum 不符时打印警告并写入报告。基线报告：`eval/reports/baseline_llm_slot_{main,holdout}.{json,md}`。

### H4：L4 剧本与 Runner ✅
- **目标**：编写 ≈ 20 个 YAML 剧本（见 4.3），Runner 用固定 `now` 与临时数据库（每个剧本重新 seed）驱动 `run_turn`，断言最终 DB 状态与关键 span；统计任务完成率、平均轮数、单轮 P50/P95、远程 LLM 调用数。
- **修改文件**：`eval/scenarios/*.yaml`、`eval/runners/run_e2e.py`、`tests/integration/test_e2e_smoke.py`（mock 回放 1 个剧本，进 CI）。
- **验收标准**：Runner 可按 `--config llm|local|routed` 切换；CI 中冒烟剧本通过。
- **测试方法**：`uv run python eval/runners/run_e2e.py --config llm`。
- **实现备注**：共 22 个剧本，覆盖 4.3 列出的全部必备场景，另外加了一句话下单、拒绝后改选、确认时改时长和查看订单列表。剧本格式写在 `run_e2e.py` 的模块文档里。在 4.3 的 `expect_span` 和最终 `booking` 之外，每轮还可以断言 `expect_reply_type` / `expect_slots` / `expect_relaxations`；剧本支持多会话（`session` / `user`）、`restart`（用同一个 checkpoint 库重建 Agent）和 `setup.bookings`（开聊前通过 booking-mcp 下单或支付）；最终断言还有 `no_booking`、按用户统计的 `bookings`、`setup_bookings`（含 `refund_ratio`）和 `state_phase`。每个剧本使用独立的临时目录：按 `now` 当天重新 seed 的 SQLite（booking-mcp 在进程内运行）、固定时钟和新的 checkpoint 库。知识库有三种模式：`rag`（knowledge-mcp 在进程内运行，使用已导入的知识库）、`http`、`stub`（对 `kb/` 做字符二元组检索；没有 `kb/` 时改用内置段落，CI 中就是这种情况）。远程 LLM 调用数按 `llm:*` generation span 统计，不含 `llm:llama_server`。`--config mock` 只运行每轮都带 `mock`（意图 + 抽取结果）的剧本，CI 冒烟用这种方式回放 `interject_refund_then_continue`。测试里还故意写错期望值，确认这些断言确实会报失败。

### H5：大模型基线 ✅
- **目标**：用 DeepSeek 跑 L2（main + holdout）与 L4，产出基线报告，填入 2.4 表"纯大模型"列。
- **修改文件**：`eval/reports/baseline_llm_*.md`、`DEV_SPEC.md` 2.4 表。
- **验收标准**：报告提交；分析 DeepSeek 主要错误类型（作为造数重点）。
- **测试方法**：同 H3 / H4。
- **实现备注**：L2 主集通过率 98.3%（59/60），holdout 88.9%（24/27），协议层均为 100%。L4 完成率 22/22，多次运行结果一致；平均 3.32 轮，单轮 P50 / P95 为 804 / 1937 ms（退款规则改为 15 分钟后重跑），每个会话 4.14 次远程调用。DeepSeek 的 4 个错误都出在"该不该输出某个键"上：非必填字段把 "不限" 输出成 `null`（2 个）；拒绝确认时多输出了 `companion_name: null`；把 "声音好听" 当成了 `voice_required`。据此定下的 I2 造数重点见 `eval/reports/baseline_llm_summary.md`。目前的 L4 剧本对大模型区分度不高，后续可以加入口语化、多意图的剧本。

---

## 阶段 I：数据与训练（目标：M2 —— 小模型通过 L2 门槛）

### I1：训练侧协议与 Prompt 同源 ✅
- **目标**：`training` 直接 import `rift_domain.slots` 与 `rift_agent.extractors.prompt.build_prompt`，保证训练样本的 prompt 与线上逐字节一致；数据卡记录 `slot_extract.txt` 的 sha256。
- **修改文件**：`training/src/rift_training/contract/__init__.py`、`training/src/rift_training/data/card.py`、`tests/unit/test_training_contract.py`。
- **验收标准**：同一上下文由线上与训练侧渲染的 prompt 完全相同（测试断言）。
- **测试方法**：`uv run pytest -q tests/unit/test_training_contract.py`。
- **实现备注**：线上渲染函数实际叫 `build_messages`（不是 `build_prompt`）。`rift_training.contract` 不复制任何协议代码：`context_from_sample` 把样本转成 `ExtractionContext`，`render_prompt` 直接调用线上的 `build_messages`，L2 Runner 的 `build_context` 也改为复用它；唯一新增的是 `render_target`（训练目标 JSON：按 schema 字段顺序，保留"未出现 / null / any"三态，整数值的浮点数写成 `2` 而不是 `2.0`）。测试把 87 条 L2 样本逐条组装成图状态，经真实的 `extract_slots` 节点拿到上下文，断言与训练侧渲染的消息逐字节相同。数据卡（`data_card.json` + `DATA_CARD.md`）记录 `slot_extract.txt` 的 sha256、类别 / intent / confirmation 分布和逐字段三态计数，`prompt_mismatch` 用于检测 prompt 改动后数据过期。`rift-training` 新增依赖 `rift-agent`。

### I2：场景规格（Scenario Specs） ✅
- **目标**：YAML 描述场景类别、槽位组合、说法风格（口语、黑话、错别字、中英混杂如"打 jg""辅助位"）、上下文轮数、期望 turn_intent 分布；采样器按配额生成"生成任务"。
- **修改文件**：`training/configs/data/specs_v0.1.yaml`、`rift_training/data/specs.py`、`tests/unit/test_specs_sampler.py`。
- **验收标准**：采样 800 个任务，各类别配额误差 < 5%。
- **测试方法**：`uv run pytest -q tests/unit/test_specs_sampler.py`。
- **实现备注**：每个任务固定答案的"形状"：类别 / 子场景、模式、`now`、说法风格、历史轮数、候选人数、是否待确认、`current_state` 里已有哪些槽位，以及 delta 中每个键是给值、`any` 还是 `null`；Teacher 只负责写对话和具体取值，I3 的校验器按任务核对。类别和子场景数量都用最大余数法按配额分配，800 条时误差为 0；其余随机项由固定种子决定。各模式可用的字段从 `domain.yaml` 推出（段位只在排位模式，大乱斗类不出现段位和位置），哪些字段能取 `any` 从 `SlotDelta` 的类型注解推出。v0.1 配额：首轮 16%、多轮增量 14%、三态 16%、相对时间 10%、候选引用 10%、插话咨询 8%、无关 6%、确认 / 拒绝 10%、黑话 10%；按 H5 的错误分析加入了非必填字段 `any` 与 `null` 的对比、拒绝换人时 delta 为空、"声音好听"属于风格而不是 `voice_required` 三类子场景。说法风格分标准 / 口语 / 黑话 / 错别字 / 中英混杂五种。`training/scripts/data/sample_tasks.py --spec v0.1` 打印分布并可导出任务 JSONL。

### I3：Teacher 生成与校验 ✅
- **目标**：可插拔 Teacher 后端（Anthropic / OpenAI，JSON Schema 约束）；每个任务生成"上下文 + 用户话术 + 期望输出"；校验器（C2 schema + 业务一致性：如大乱斗不应输出段位、时间表达可被 C3 解析）过滤；失败样本进入 `rejected/` 附原因；并发与断点续跑。
- **修改文件**：`rift_training/inference/{teacher_anthropic.py, teacher_openai.py}`、`data/{generate.py, validate.py}`、`training/configs/inference/*.yaml`。
- **验收标准**：生成 v0.1 ≈ 800 条，校验通过率 ≥ 90%；API Key 只从环境变量读取。
- **测试方法**：`uv run python training/scripts/data/generate.py --spec v0.1 --limit 20`（小样试跑）后全量。
- **实现备注**：Teacher 用 OpenAI `gpt-5`（Chat Completions + Structured Outputs，strict JSON Schema，`reasoning_effort: low`）；没有实现 Anthropic 后端，接口保留可插拔。strict 模式下没法表示"键不出现"，所以 Teacher 不直接写 delta：每个任务按 I2 的规格动态生成 schema，Teacher 只写上下文（候选、state、history）、用户这一句和"给值"字段的取值，`any` / `null` / 不出现由代码按任务组装，三态标签由构造保证。校验器依次检查答案结构、C2 协议与 L2 一致性规则、业务一致性（时长、时间说法原样照抄且 C3 可解析、state 时间在未来、模式、候选名、修改后的值确实变了）；不通过时把问题列表发回 Teacher 重写一次。逐行写盘、断点续跑；429/5xx 退避重试，额度耗尽或鉴权失败立即停止整轮，只因 API 失败的任务下次自动重跑。v0.1（800 条，通过率 100%）人工复核发现取值严重扎堆（段位几乎全是钻石、时长一半是 2.5 小时、时间说法 208 次里只有 76 种），于是新增 specs v0.2：每个任务按分布抽取目标值（段位、时长、位置、预算、性别、语音、服务类型、候选序号、风格主题，时间给目标时刻 + 说法类型），schema 锁定可枚举的值，校验器用线上时间解析器核对时间必须解析到目标时刻。v0.2 共 795 / 800 条通过，校验通过率 99.5%，约 330 万 token。复核中还修正了 Teacher 模板：`role_preference` 是对陪玩师位置的要求（"我打下路"不算）；修复了采样依赖进程 hash seed 的问题。发现线上时间解析器不支持"往后推三十分钟"这类分钟级调整，另行跟进。

### I4：覆盖审计、去重与数据卡 ✅
- **目标**：类别 / 字段 / 三态语义 / 模式别名覆盖统计；与 L2 评测集做近似去重（embedding 相似度阈值），防止泄漏；渲染 LLaMA-Factory SFT 格式；生成数据卡（版本、数量、分布、prompt hash、Teacher 型号）。
- **修改文件**：`data/{audit.py, dedup.py, render_sft.py, card.py}`、`training/data/processed/sft/v0.1/`。
- **验收标准**：与评测集相似度 > 阈值的样本被剔除并记录；数据卡生成。
- **测试方法**：`uv run python training/scripts/data/build_sft.py --version v0.2`。
- **实现备注**：去重分两步：训练集内部（同一句话 + 同一上下文）和与 L2 评测集（归一化后完全相同，或 bge-small-zh 余弦 ≥ 0.92）；阈值依据 v0.1 的相似度分布选定，命中的都是"确认下单""就第二个吧""改成晚一小时"这类与评测集几乎相同的短句。生成阶段不读取评测集，这里是训练数据与评测集唯一接触的地方，剔除记录写入 `removed.jsonl` 和数据卡。按类别切分 train / val（5%），渲染为 LLaMA-Factory sharegpt 格式（OpenAI 风格 `messages`：线上 system + user prompt 加标准答案），注册为 `rift_sft_v0_2_train` / `_val`。数据卡记录 prompt 与 Teacher 模板的 sha256、Teacher 型号、生成通过率、去重明细和覆盖审计（类别、子场景、风格、模式、三态字段、别名出现次数）。v0.2：795 → 去重 2 条、剔除 11 条与评测集过近 → 782 条（train 743 / val 39），提交在 `training/data/processed/sft/v0.2/`。仍未出现的别名（如"单排""极地大乱斗""硬辅""教练"）留给 I7 对比集。

### I5：SFT 训练 r001（0.6B + 1.7B） ✅
- **目标**：LLaMA-Factory LoRA SFT（参数见 3.7），租用 GPU 训练两个尺寸；记录训练曲线与耗时成本。
- **修改文件**：`training/configs/training/llamafactory/{_base_sft.yaml, sft_0.6b_r001.yaml, sft_1.7b_r001.yaml}`、`training/experiments/r001/{problem.md, strategy.md, conclusion.md}`。
- **验收标准**：adapter 产出；loss 正常收敛。
- **测试方法**：GPU 机器执行 `llamafactory-cli train ...`。
- **实现备注**：没有租云 GPU，在本机 WSL2（Ubuntu 24.04）+ RTX 5080 16GB 上训练：uv 虚拟环境，torch 2.11+cu128（5080 为 sm_120），LLaMA-Factory 0.9.5。数据用 `sft/v0.2`（782 条）。模板选 `qwen3` + `enable_thinking: false`，`training/scripts/train/check_template.py` 验证与官方 chat template（`enable_thinking=False`，assistant 前带空 think 块）在 743 条上逐 token 一致，因此线上 llama-server 必须关闭 thinking（J1 落实，评测时用服务端 `--chat-template-kwargs`）。0.6B 首次用 batch 8 × 2 时显存溢出到系统内存，每步从 5 s 变成 32 s，改为 4 × 4 后 6 分钟训完；1.7B 用 2 × 8，10 分钟。eval loss 0.6B 0.028、1.7B 0.018，曲线平稳。`_base_sft.yaml` 没有单独建：LLaMA-Factory 配置不支持继承，两份配置各自完整。

### I6：本地推理评测 r001 ✅
- **目标**：merge adapter → 用 llama-server（F16）加载 → 跑 L2 main + holdout，对比 base 模型与 DeepSeek 基线。
- **修改文件**：`training/scripts/eval/run_local_eval.py`、`training/experiments/r001/conclusion.md`。
- **验收标准**：产出三方对比表与错误分布。
- **测试方法**：`uv run python eval/runners/run_slot_eval.py --extractor local --dataset main`。
- **实现备注**：`llamafactory-cli export` 合并 adapter（合并后的 chat template 与官方一致）→ llama.cpp `convert_hf_to_gguf.py` 转 F16 → CPU 编译的 llama-server（WSL 没有 nvcc，CPU 推理也正是 I10 的部署场景）。新增环境变量 `LOCAL_SLOT_TIMEOUT_S` 覆盖本地抽取器超时（默认仍为 5 s）。L2 结果：DeepSeek 98.3% / 88.9%，0.6B 基座 0% / 0%（不懂协议），**0.6B SFT 73.3% / 74.1%**，**1.7B SFT 81.7% / 92.6%**（holdout 超过 DeepSeek），两者协议层均 100%；CPU F16 单次延迟 P50 0.6B 1.1 s、1.7B 2.3 s。主要错误：时间吞掉时长（"明晚八点两小时"）、一句多槽位漏抽、候选序号（"1号吧"）、非必填字段的 `any`，以及规格遗漏导致的 `game_mode: null`（训练集一条都没有）。0.6B 距 M2 门槛差约 20pp，详见 `training/experiments/r001/conclusion.md`。

### I7：对比集迭代 r002–r004
- **目标**：按错误分布设计对比集（复制场景并改一个条件使正确输出翻转，例如"大乱斗 + 钻石"→ 段位应被清空、"不限女"vs"要女"），每轮追加 150–250 条，重训重评；每轮写 problem/strategy/conclusion。
- **修改文件**：`training/configs/data/contrast_v0.x.yaml`、`training/experiments/r00x/*`、`training/data/processed/sft/v0.x/`。
- **验收标准**：0.6B 在 L2 main 通过率 ≥ DeepSeek 基线 − 5pp，holdout 通过率 ≥ 85%（首次基线出来后可修订并记录原因）。**M2**。
- **测试方法**：同 I6。

### I8：DPO 偏好对构造（规则扰动 + On-Policy）
- **目标**：① 规则扰动：对 chosen 做字段替换、三态混淆（`null`↔缺失↔`any`）、intent 翻转；② On-Policy：用最佳 SFT 模型在训练集上 temperature 0.7 采样 k=4，未通过评分器的输出为 rejected，Teacher 答案为 chosen；两套数据各自出数据卡。
- **修改文件**：`data/{dpo_rule.py, dpo_onpolicy.py}`、`training/data/processed/dpo/{rule_v1, onpolicy_v1}/`。
- **验收标准**：On-Policy 偏好对 ≥ 300 对；rejected 错误类型分布写入数据卡。
- **测试方法**：`uv run python training/scripts/data/build_dpo.py --mode rule|onpolicy`。

### I9：DPO 消融实验
- **目标**：在最佳 SFT 上训练 4 组：{rule, onpolicy} × β{0.1, 0.3}（lr 5e-6、1 epoch、sigmoid），0.6B 必做、1.7B 可选；评测 L2 main / holdout，挑最优再跑 L4。
- **修改文件**：`training/configs/training/llamafactory/dpo_*.yaml`、`training/experiments/dpo/{ablation.md}`。
- **验收标准**：产出消融表（SFT vs 4 组 DPO）与分析：是否提升、提升/退化集中在哪些字段、与上个项目规则扰动 DPO 无效的对照解释。无论正负结论均记录。
- **测试方法**：同 I6。

### I10：量化与性能基准
- **目标**：最终模型 merge → GGUF → F16 / Q8_0 / Q4_K_M（可选 imatrix）；llama-server 启动时预热共享前缀；在同一台 CPU 机器上测 TTFT、E2E、decode tok/s、质量（L2）——**同一次运行中对比**，避免上个项目"不同批次对比"的问题。
- **修改文件**：`training/configs/quantization/*.yaml`、`training/scripts/quantize/*`、`training/experiments/quant/report.md`。
- **验收标准**：Q4_K_M 相对 F16 质量下降 ≤ 2 条；报告提交。
- **测试方法**：`uv run python training/scripts/quantize/bench.py`。

### I11：模型发布
- **目标**：GGUF 与模型卡（训练数据版本、评测结果、许可证）放入 `models/`（git 忽略），提供下载/放置说明；可选上传 HuggingFace（需用户确认后手动操作）。
- **修改文件**：`models/README.md`、`training/MODEL_CARD.md`。
- **验收标准**：按说明可在新机器上放置模型并被 llama-server 加载。
- **测试方法**：手动。

---

## 阶段 J：小模型接入（目标：M3 —— 三种配置对比表）

### J1：LocalSlotExtractor
- **目标**：基于 B3 provider 实现本地抽取器，`max_tokens` 160、JSON 模式（llama.cpp grammar 由 C2 JSON Schema 生成，可配置开关）。
- **修改文件**：`extractors/local.py`、`tests/unit/test_local_extractor.py`。
- **验收标准**：grammar 开启时协议合法率 100%（L2 main）。
- **测试方法**：`uv run python eval/runners/run_slot_eval.py --extractor local`。

### J2：llama-server 容器与前缀预热
- **目标**：`llama.Dockerfile`（llama.cpp server，挂载 GGUF），启动脚本发送一次仅含系统提示词的请求预热前缀缓存；健康检查。
- **修改文件**：`docker/llama.Dockerfile`、`docker/llama_warmup.py`。
- **验收标准**：容器启动后首个真实请求 TTFT 接近稳态值。
- **测试方法**：`docker compose up llama-server` 后运行 `scripts/bench_slot_latency.py`。

### J3：影子模式评估
- **目标**：配置 `primary=llm, shadow=local`，跑 L4 全部剧本 + 手动对话 ≥ 50 轮，统计一致率与逐字段不一致分布。
- **修改文件**：`config/settings.shadow.yaml`、`eval/reports/shadow_*.md`。
- **验收标准**：报告含一致率、主要分歧类型及示例。
- **测试方法**：`uv run python eval/runners/run_e2e.py --config shadow`。

### J4：降级链路验证
- **目标**：`primary=local, fallback=llm`；故障注入：停掉 llama-server、构造超时、强制输出非法 JSON，验证降级与 span 记录。
- **修改文件**：`tests/integration/test_fallback_injection.py`。
- **验收标准**：三种故障下对话均正常完成，span 中 `fell_back=true` 且原因正确。
- **测试方法**：`uv run pytest -q tests/integration/test_fallback_injection.py`。

### J5：三配置对比
- **目标**：`llm` / `local` / `routed(local→llm)` 三配置跑 L2 + L4，填写 2.4 表；计算远程调用节省比例与延迟变化。
- **修改文件**：`eval/reports/comparison.md`、`DEV_SPEC.md` 2.4 表。
- **验收标准**：**M3**：对比表完整，结论可复现（记录命令与 commit）。
- **测试方法**：`uv run python eval/runners/run_all.py`。

---

## 阶段 K：看板与数据飞轮（目标：M4 —— 闭环）

### K1：看板迁入与多数据源
- **目标**：迁入 RAG Streamlit（6 页），新增数据源服务读取 `traces.db`、`booking.db`；统一导航。
- **修改文件**：`apps/dashboard/{app.py, services/*.py, pages/*}`。
- **验收标准**：原 6 页正常；可选择时间范围。
- **测试方法**：`uv run streamlit run apps/dashboard/app.py`；`tests/e2e/test_dashboard_smoke.py`（AppTest）。

### K2：槽位质量页
- **目标**：大小模型一致率趋势、降级率、逐字段不一致热力图、延迟分布，按抽取器 / 模型版本筛选。
- **修改文件**：`apps/dashboard/pages/slot_quality.py`。
- **验收标准**：J3 数据可正确展示。
- **测试方法**：AppTest 冒烟。

### K3：业务指标页
- **目标**：预约漏斗（进入 BOOKING → 展示候选 → 报价 → 下单 → 支付）、平均轮数、零结果与各放宽步骤频率、取消率与退款分布、热门模式与时段。
- **修改文件**：`apps/dashboard/pages/business.py`。
- **验收标准**：数据与 SQL 手工核对一致。
- **测试方法**：AppTest 冒烟 + 手工核对。

### K4：样本挖掘与导出
- **目标**：筛选条件：shadow 不一致、发生降级、同一槽位连续被追问 ≥ 2 次、用户说"不是/我说的是"；列表展示上下文；勾选导出为训练侧可读的 `mined.jsonl`（含完整上下文，不含 cookie 等敏感信息）。
- **修改文件**：`apps/dashboard/pages/mining.py`、`training/src/rift_training/data/import_mined.py`。
- **验收标准**：导出文件可被 `import_mined.py` 读取并送 Teacher 重标。
- **测试方法**：`uv run pytest -q tests/unit/test_import_mined.py`。

### K5：飞轮一轮
- **目标**：挖掘样本 → Teacher 重标 → 人工抽检 ≥ 20% → 并入 v0.x+1 → 重训 → L2/L4 回归 → 替换 GGUF；写实验记录（挖掘量、修正率、指标变化）。
- **修改文件**：`training/experiments/flywheel_r1/*`。
- **验收标准**：**M4**：完整一轮记录；新模型在挖掘样本上的通过率提升，且 L2 holdout 不退化。
- **测试方法**：同 I6 / J5。

---

## 阶段 L：收尾（目标：开箱即用、可复现、可讲述）

### L1：docker compose 一键启动
- **目标**：`compose.yaml` 编排 web、booking-mcp、knowledge-mcp、llama-server（可选 profile `local-model`），初始化容器执行 seed + 生成知识库 + 导入；`dev_up.ps1 / dev_up.sh` 提供无 Docker 方案。
- **修改文件**：`docker/*`、`scripts/dev_up.*`。
- **验收标准**：干净机器上 `docker compose up` 后 5 分钟内可在浏览器完成一次预约。
- **测试方法**：手动（全新克隆验证）。

### L2：README
- **目标**：项目简介、架构图、快速开始、配置说明、MCP 接入 Claude Desktop、评测与训练复现命令、核心指标表、截图/GIF。
- **修改文件**：`README.md`、`docs/img/*`。
- **验收标准**：按 README 从零复现成功。
- **测试方法**：手动。

### L3：演示脚本
- **目标**：5 分钟演示路线：黑话预约 → 插话问退款 → 放宽匹配 → 双窗口抢时段 → 取消退款 → 切换 local/llm 对比延迟 → Langfuse 看 trace → 看板看一致率与挖掘。
- **修改文件**：`docs/demo_script.md`。
- **验收标准**：按脚本完整走通一遍。
- **测试方法**：手动。

### L4：全链路验收与简历素材
- **目标**：全部测试与评测重跑一遍并记录；整理设计决策（附录 ADR）、指标、难点与取舍为简历/面试素材。
- **修改文件**：`docs/resume_notes.md`、`DEV_SPEC.md`（进度收口）。
- **验收标准**：CI 绿；2.4 表全部填写；进度表全部 ✅。
- **测试方法**：`uv run pytest -q && uv run python eval/runners/run_all.py`。

### 交付里程碑

| 里程碑 | 完成阶段 | 可演示内容 |
|---|---|---|
| M1 | A–F | CLI 中用 DeepSeek 完成预约、插话咨询、取消 |
| — | G–H | Web 界面 + 大模型基线评测报告 |
| M2 | I | 0.6B 小模型通过 L2 门槛，DPO 消融表 |
| M3 | J | 三配置对比表（完成率 / 延迟 / 远程调用） |
| M4 | K | 看板 + 一轮数据飞轮 |
| 交付 | L | 一键启动 + README + 演示脚本 |

---

## 7. 可扩展性与未来展望

- **改期**：`reschedule_booking` = 事务内"取消 + 新建"，退款规则按改期政策单独配置。
- **多游戏**：`domain.yaml` 按游戏分节（王者荣耀段位与模式体系），槽位加 `game`，重跑数据与训练——验证"换一套配置即可迁移"。
- **LLM 用户模拟器**：画像 + 目标驱动的多轮自由对话，LLM 裁判判定成败，扩大 L4 覆盖。
- **意图分类下沉小模型**：在输出协议中加入首轮分类，或单独训练轻量分类器，进一步减少远程调用。
- **Langfuse 自托管**：Postgres + ClickHouse + Redis + MinIO 编入 compose profile。
- **真实支付与通知**：支付回调、开局前短信/站内提醒（定时任务）。
- **个性化推荐**：基于历史订单与反馈的陪玩师推荐（原预约项目的行为分析思路）。
- **多实例部署**：checkpointer 与业务库迁移到 Postgres，web 无状态水平扩展。

---

## 8. 附录：设计决策记录（ADR）

| # | 决策 | 备选 | 理由 |
|---|---|---|---|
| 1 | 小模型换领域后重训，训练流水线纳入 monorepo，前期先用大模型实现同一接口 | 不训练 / 训练留在原仓库 | 叙事完整（基线 → 蒸馏替换 → 飞轮）；协议与规则同源 |
| 2 | 批量训练数据由 Teacher API 生成；评测集人工编写审核 | 全部对话内手写 / 全部 Teacher 生成 | 可复现、规模化；避免"自己考自己" |
| 3 | 只做英雄联盟；6 种模式（含斗魂竞技场）；按小时计费 | 多游戏 / 按局计费 | 降低枚举与数据量；时长确定便于档期冲突检测 |
| 4 | `missing_info` 等规则由代码计算 | 模型输出 | 规则可热更新无需重训；模型输出更短 |
| 5 | 模型为纯抽取器；动作决策与回复由代码负责；回复默认模板，可选润色 | 抽取+回复 / 保持原协议 | 消除 action 与回复一致性错误；输出 ≈ 80 token 降低延迟；"LLM 理解，代码决策" |
| 6 | Delta 输出 + 三态语义（缺失 / any / null）；时间输出原始表达由代码解析 | 完整状态输出；模型直接输出绝对时间 | 状态继承由代码保证；小模型不做日期算术 |
| 7 | 陪玩师只存单双排段位；价格 = 基础价 × 服务系数；放宽顺序 时间 → 性别 → 位置，预算与段位不放宽 | — | 简化模型；可解释的降级 |
| 8 | 知识库三类内容由唯一数据源生成；RAG 源码复制进 monorepo；新增 MarkdownLoader；关闭多模态 | submodule / PDF 导入 | 数据一致；不污染通用 RAG 仓库；展示可插拔扩展 |
| 9 | 预约中插话咨询不切换阶段；显式放弃才清空；`turn_intent` 三值替代 `unrelated` | 原项目状态切换 | 修复插话后预约进度丢失 |
| 10 | 确认 + 防重 + 取消 + 模拟支付；不做改期；booking 工具独立为 MCP Server；昵称登录 | — | 控制范围；工具可被外部 Agent 复用 |
| 11 | LangGraph 编排 + 自有 Provider 工厂；不用 LangChain 本体 / AgentExecutor | 全用 LangChain / 完全自研 | 显式状态机与 checkpointer / interrupt 契合；避免 LangChain 抽象层与版本变动 |
| 12 | MCP 走 Streamable HTTP（保留 stdio）；docker compose；uv workspace；前端 Jinja2 + SSE，看板 Streamlit | stdio / React | 服务独立启停；重点在后端与 AI |
| 13 | 校验 → 重试 → 降级 → 兜底；影子模式；远程大模型 DeepSeek | 单模型 | 可用性与渐进上线；影子数据驱动飞轮 |
| 14 | 层级 Span、跨服务 trace_id；SQLite（统计）+ Langfuse Cloud（排查）双写；Prompt 留仓库 | 仅 JSONL / 自托管 Langfuse | 各取所长；零运维；Prompt 与训练严格一致 |
| 15 | 四层评测；L4 剧本断言 DB 状态；暂不做用户模拟器；离线 CI | LLM 裁判 | 确定性、可回归 |
| 16 | Qwen3-0.6B 主 + 1.7B 对照；SFT + 对比集迭代；DPO 做规则扰动 vs On-Policy 消融；Teacher 用更强模型 | 仅 SFT / DeepSeek 作 Teacher | 用数据回答"DPO 为何上次无效"；小模型有机会在窄任务上超过兜底模型 |
| 17 | 先领域核心（C）后编排（F） | 先做最小可聊版本 | 地基先用 L1 锁死，降低返工 |
| 18 | 项目名 `rift-companion-agent`，虚构平台「峡谷陪练」 | — | 无版权与冒充风险 |
