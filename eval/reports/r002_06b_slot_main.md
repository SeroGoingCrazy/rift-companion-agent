# L2 槽位抽取评测：local / main

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_main.jsonl`（60 条，sha256 `ba0a7e27dca3`） |
| 时间 | 2026-10-01 23:31:59 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **85.0%**（51/60） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.958 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 912 / 1719 / 1806 / 4485 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 7/8 | 87.5% |
| 多轮增量 `multi_turn` | 6/7 | 85.7% |
| 三态语义 `tri_state` | 7/8 | 87.5% |
| 相对时间表达 `relative_time` | 7/7 | 100.0% |
| 候选引用 `candidate_ref` | 4/6 | 66.7% |
| 插话咨询 `consult` | 4/6 | 66.7% |
| 无关话题 `unrelated` | 5/5 | 100.0% |
| 确认 / 拒绝 `confirmation` | 7/7 | 100.0% |
| 模式说法与黑话 `mode_slang` | 4/6 | 66.7% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.role_preference` | 1 | 0 | 1 | 2 |
| `delta.companion_name` | 1 | 0 | 1 | 2 |
| `turn_intent` | 0 | 0 | 2 | 2 |
| `delta.service_type` | 2 | 0 | 0 | 2 |
| `delta.style_preference` | 0 | 0 | 1 | 1 |

## 错误样例（9/9）

### main-005（首轮抽取）

- 输入：约个打野，单双排上分，铂金段位，每小时预算60以内
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["jungle"], "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "platinum", "budget_per_hour": 60}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "rank_requirement": "platinum", "service_type": "climb", "budget_per_hour": 60}, "confirmation": "none"}`
- `delta.role_preference` missing_key：期望 `["jungle"]`，实际 `"<absent>"`

### main-012（多轮增量）

- 输入：打野换成辅助吧
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["support"]}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"role_preference": ["jungle", "support"]}, "confirmation": "none"}`
- `delta.role_preference` wrong_value：期望 `["support"]`，实际 `["jungle", "support"]`

### main-023（三态语义）

- 输入：风格没要求，段位改成大师
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "any", "rank_requirement": "master"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"rank_requirement": "master", "style_preference": null}, "confirmation": "none"}`
- `delta.style_preference` wrong_value：期望 `"any"`，实际 `null`

### main-035（候选引用）

- 输入：第三个吧，再把时长改成三小时
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "小鹿乱撞", "duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "none"}`
- `delta.companion_name` missing_key：期望 `"小鹿乱撞"`，实际 `"<absent>"`

### main-036（候选引用）

- 输入：1号吧
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "星河滚烫"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"companion_name": "1号吧"}, "confirmation": "none"}`
- `delta.companion_name` wrong_value：期望 `"星河滚烫"`，实际 `"1号吧"`

### main-040（插话咨询）

- 输入：要是陪玩师迟到了怎么办
- 期望：`{"turn_intent": "consult", "delta": {}, "confirmation": "none"}`
- 输出：`{"turn_intent": "unrelated", "delta": {}, "confirmation": "none"}`
- `turn_intent` wrong_value：期望 `"consult"`，实际 `"unrelated"`

### main-042（插话咨询）

- 输入：下单之后多久之内要付款？
- 期望：`{"turn_intent": "consult", "delta": {}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {}, "confirmation": "none"}`
- `turn_intent` wrong_value：期望 `"consult"`，实际 `"booking"`

### main-057（模式说法与黑话）

- 输入：带我上分，双排，jg位，翡翠，明天下午两点三小时
- 期望：`{"turn_intent": "booking", "delta": {"service_type": "climb", "game_mode": "ranked_solo_duo", "role_preference": ["jungle"], "rank_requirement": "emerald", "start_time_expr": "明天下午两点", "duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明天下午两点", "duration_hours": 3, "rank_requirement": "emerald", "role_preference": ["jungle"]}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`

### main-060（模式说法与黑话）

- 输入：找个会玩辅助位的小哥哥陪我单排上分，黄金，周六晚上七点，两小时
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["support"], "companion_gender": "male", "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "gold", "start_time_expr": "周六晚上七点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "周六晚上七点", "duration_hours": 2, "rank_requirement": "gold", "role_preference": ["support"], "companion_gender": "male"}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`
