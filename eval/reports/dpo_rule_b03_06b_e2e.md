# L4 端到端评测：local

| 项 | 值 |
|---|---|
| 配置 | `local`（deepseek-chat） |
| 知识库 | `rag` |
| 剧本 | 22 个（`eval/scenarios/`） |
| 时间 | 2026-10-02 12:05:54 |

## 总览

| 指标 | 值 |
|---|---:|
| **任务完成率** | **90.9%**（20/22） |
| 平均轮数 | 3.32 |
| 单轮延迟 P50 / P95 / max（ms） | 1346 / 5144 / 9293 |
| 远程 LLM 调用 / 会话 | 1.41 |
| 远程 LLM 调用 / 轮 | 0.42 |

## 剧本结果

| 剧本 | 结果 | 轮数 | LLM 调用 | 最慢一轮（ms） |
|---|---|---:|---:|---:|
| `budget_never_relaxed` | ✅ | 1 | 1 | 5508 |
| `cancel_refund_boundary` | ✅ | 2 | 1 | 603 |
| `cancel_refund_full` | ✅ | 2 | 1 | 608 |
| `cancel_refund_none` | ✅ | 2 | 1 | 634 |
| `confirm_with_change` | ✅ | 4 | 1 | 3812 |
| `consult_only` | ✅ | 2 | 4 | 9293 |
| `decline_then_pick_another` | ✅ | 5 | 1 | 4032 |
| `double_session_same_slot` | ❌ | 6 | 6 | 6472 |
| `explicit_abandon` | ✅ | 2 | 1 | 3984 |
| `interject_refund_then_continue` | ✅ | 5 | 2 | 4268 |
| `list_orders` | ✅ | 1 | 1 | 604 |
| `multi_turn_collect` | ✅ | 6 | 1 | 3384 |
| `one_shot_booking` | ✅ | 3 | 1 | 4500 |
| `relative_time_edit` | ✅ | 4 | 1 | 4690 |
| `relax_rank_gap` | ❌ | 3 | 1 | 4743 |
| `restart_resume` | ✅ | 3 | 1 | 4279 |
| `slang_aram_mayhem` | ✅ | 3 | 1 | 4219 |
| `slang_arena` | ✅ | 3 | 1 | 4123 |
| `slang_doubles_jungle` | ✅ | 3 | 1 | 5144 |
| `tri_state_any_rank` | ✅ | 4 | 1 | 4069 |
| `tri_state_withdraw_time` | ✅ | 5 | 1 | 4368 |
| `unrelated_pull_back` | ✅ | 4 | 1 | 4385 |

## 失败详情

### double_session_same_slot

- turn 1 (帮我约个明晚八点的单双排，挑战者，两小时): reply_type: expected ['candidates'], got 'extraction_failed'
- turn 2 (帮我约个明晚八点的单双排，挑战者，两小时): reply_type: expected ['candidates'], got 'extraction_failed'
- turn 3 (第一个): reply_type: expected ['confirm_booking'], got 'help'
- turn 4 (第一个): reply_type: expected ['confirm_booking'], got 'help'
- turn 5 (确认): reply_type: expected ['booked'], got 'help'
- turn 6 (确认): reply_type: expected ['candidates', 'no_candidates'], got 'help'
- booking: expected a booking, none was created
- bookings[tester_01]: expected 1, got 0

| # | 会话 | 用户输入 | reply_type | 工具 |
|---:|---|---|---|---|
| 1 | main | 帮我约个明晚八点的单双排，挑战者，两小时 | `extraction_failed` | - |
| 2 | other | 帮我约个明晚八点的单双排，挑战者，两小时 | `extraction_failed` | - |
| 3 | main | 第一个 | `help` | - |
| 4 | other | 第一个 | `help` | - |
| 5 | main | 确认 | `help` | - |
| 6 | other | 确认 | `help` | - |

### relax_rank_gap

- turn 1 (约个单双排，宗师以上，要会玩上单的小姐姐，明晚八点两小时): slot rank_requirement: expected 'grandmaster', got 'challenger'

| # | 会话 | 用户输入 | reply_type | 工具 |
|---:|---|---|---|---|
| 1 | main | 约个单双排，宗师以上，要会玩上单的小姐姐，明晚八点两小时 | `candidates` | find_companions |
| 2 | main | 第一个 | `confirm_booking` | quote_price |
| 3 | main | 确认 | `booked` | create_booking |
