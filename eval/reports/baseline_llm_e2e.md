# L4 端到端评测：llm

| 项 | 值 |
|---|---|
| 配置 | `llm`（deepseek-chat） |
| 知识库 | `rag` |
| 剧本 | 22 个（`eval/scenarios/`） |
| 时间 | 2026-10-01 10:35:04 |

## 总览

| 指标 | 值 |
|---|---:|
| **任务完成率** | **100.0%**（22/22） |
| 平均轮数 | 3.32 |
| 单轮延迟 P50 / P95 / max（ms） | 834 / 1686 / 9549 |
| 远程 LLM 调用 / 会话 | 4.14 |
| 远程 LLM 调用 / 轮 | 1.25 |

## 剧本结果

| 剧本 | 结果 | 轮数 | LLM 调用 | 最慢一轮（ms） |
|---|---|---:|---:|---:|
| `budget_never_relaxed` | ✅ | 1 | 2 | 2109 |
| `cancel_refund_full` | ✅ | 2 | 1 | 881 |
| `cancel_refund_half` | ✅ | 2 | 1 | 756 |
| `cancel_refund_none` | ✅ | 2 | 1 | 753 |
| `confirm_with_change` | ✅ | 4 | 5 | 1831 |
| `consult_only` | ✅ | 2 | 4 | 9549 |
| `decline_then_pick_another` | ✅ | 5 | 6 | 1561 |
| `double_session_same_slot` | ✅ | 6 | 8 | 1572 |
| `explicit_abandon` | ✅ | 2 | 3 | 1493 |
| `interject_refund_then_continue` | ✅ | 5 | 7 | 1483 |
| `list_orders` | ✅ | 1 | 1 | 616 |
| `multi_turn_collect` | ✅ | 6 | 7 | 1420 |
| `one_shot_booking` | ✅ | 3 | 4 | 1568 |
| `relative_time_edit` | ✅ | 4 | 5 | 1516 |
| `relax_rank_gap` | ✅ | 3 | 4 | 1366 |
| `restart_resume` | ✅ | 3 | 4 | 1186 |
| `slang_aram_mayhem` | ✅ | 3 | 4 | 1650 |
| `slang_arena` | ✅ | 3 | 4 | 1306 |
| `slang_doubles_jungle` | ✅ | 3 | 4 | 1394 |
| `tri_state_any_rank` | ✅ | 4 | 5 | 1630 |
| `tri_state_withdraw_time` | ✅ | 5 | 6 | 1348 |
| `unrelated_pull_back` | ✅ | 4 | 5 | 1171 |

## 失败详情

无。
