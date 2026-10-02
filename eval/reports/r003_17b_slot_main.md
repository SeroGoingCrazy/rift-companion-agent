# L2 槽位抽取评测：local / main

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_main.jsonl`（60 条，sha256 `ba0a7e27dca3`） |
| 时间 | 2026-10-02 10:32:59 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **91.7%**（55/60） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.984 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 2097 / 3996 / 4379 / 7918 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 7/8 | 87.5% |
| 多轮增量 `multi_turn` | 7/7 | 100.0% |
| 三态语义 `tri_state` | 7/8 | 87.5% |
| 相对时间表达 `relative_time` | 7/7 | 100.0% |
| 候选引用 `candidate_ref` | 6/6 | 100.0% |
| 插话咨询 `consult` | 6/6 | 100.0% |
| 无关话题 `unrelated` | 5/5 | 100.0% |
| 确认 / 拒绝 `confirmation` | 6/7 | 85.7% |
| 模式说法与黑话 `mode_slang` | 4/6 | 66.7% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.service_type` | 3 | 0 | 0 | 3 |
| `delta.style_preference` | 0 | 0 | 1 | 1 |
| `confirmation` | 0 | 0 | 1 | 1 |

## 错误样例（5/5）

### main-005（首轮抽取）

- 输入：约个打野，单双排上分，铂金段位，每小时预算60以内
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["jungle"], "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "platinum", "budget_per_hour": 60}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "rank_requirement": "platinum", "budget_per_hour": 60, "role_preference": ["jungle"]}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`

### main-023（三态语义）

- 输入：风格没要求，段位改成大师
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "any", "rank_requirement": "master"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"rank_requirement": "master", "style_preference": null}, "confirmation": "none"}`
- `delta.style_preference` wrong_value：期望 `"any"`，实际 `null`

### main-053（确认 / 拒绝）

- 输入：改成三小时再下单
- 期望：`{"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "yes"}`
- 输出：`{"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "none"}`
- `confirmation` wrong_value：期望 `"yes"`，实际 `"none"`

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
