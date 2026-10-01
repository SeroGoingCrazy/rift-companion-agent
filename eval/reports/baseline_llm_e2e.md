# L4 端到端评测：llm

| 项 | 值 |
|---|---|
| 配置 | `llm`（deepseek-chat） |
| 知识库 | `rag` |
| 剧本 | 22 个（`eval/scenarios/`） |
| 时间 | 2026-10-01 11:08:00 |

## 总览

| 指标 | 值 |
|---|---:|
| **任务完成率** | **100.0%**（22/22） |
| 平均轮数 | 3.32 |
| 单轮延迟 P50 / P95 / max（ms） | 804 / 1937 / 9247 |
| 远程 LLM 调用 / 会话 | 4.14 |
| 远程 LLM 调用 / 轮 | 1.25 |

## 剧本结果

| 剧本 | 结果 | 轮数 | LLM 调用 | 最慢一轮（ms） |
|---|---|---:|---:|---:|
| `budget_never_relaxed` | ✅ | 1 | 2 | 1224 |
| `cancel_refund_boundary` | ✅ | 2 | 1 | 705 |
| `cancel_refund_full` | ✅ | 2 | 1 | 618 |
| `cancel_refund_none` | ✅ | 2 | 1 | 767 |
| `confirm_with_change` | ✅ | 4 | 5 | 1122 |
| `consult_only` | ✅ | 2 | 4 | 9247 |
| `decline_then_pick_another` | ✅ | 5 | 6 | 1963 |
| `double_session_same_slot` | ✅ | 6 | 8 | 1697 |
| `explicit_abandon` | ✅ | 2 | 3 | 1341 |
| `interject_refund_then_continue` | ✅ | 5 | 7 | 1454 |
| `list_orders` | ✅ | 1 | 1 | 587 |
| `multi_turn_collect` | ✅ | 6 | 7 | 1283 |
| `one_shot_booking` | ✅ | 3 | 4 | 1351 |
| `relative_time_edit` | ✅ | 4 | 5 | 1663 |
| `relax_rank_gap` | ✅ | 3 | 4 | 1827 |
| `restart_resume` | ✅ | 3 | 4 | 1404 |
| `slang_aram_mayhem` | ✅ | 3 | 4 | 1688 |
| `slang_arena` | ✅ | 3 | 4 | 1595 |
| `slang_doubles_jungle` | ✅ | 3 | 4 | 1937 |
| `tri_state_any_rank` | ✅ | 4 | 5 | 1610 |
| `tri_state_withdraw_time` | ✅ | 5 | 6 | 1310 |
| `unrelated_pull_back` | ✅ | 4 | 5 | 1673 |

## 失败详情

无。
