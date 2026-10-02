# L2 槽位抽取评测：local / main

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_main.jsonl`（60 条，sha256 `ba0a7e27dca3`） |
| 时间 | 2026-10-02 13:41:53 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **88.3%**（53/60） |
| 协议通过率 | 95.0% |
| 平均任务分（协议通过样本） | 0.980 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 475 / 734 / 805 / 826 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 7/8 | 87.5% |
| 多轮增量 `multi_turn` | 7/7 | 100.0% |
| 三态语义 `tri_state` | 8/8 | 100.0% |
| 相对时间表达 `relative_time` | 7/7 | 100.0% |
| 候选引用 `candidate_ref` | 3/6 | 50.0% |
| 插话咨询 `consult` | 6/6 | 100.0% |
| 无关话题 `unrelated` | 5/5 | 100.0% |
| 确认 / 拒绝 `confirmation` | 6/7 | 85.7% |
| 模式说法与黑话 `mode_slang` | 4/6 | 66.7% |

## 协议错误

| 类型 | 次数 |
|---|---:|
| `enum` | 3 |

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.companion_name` | 0 | 0 | 3 | 3 |
| `delta.companion_gender` | 1 | 0 | 0 | 1 |
| `delta.service_type` | 1 | 0 | 0 | 1 |

## 错误样例（7/7）

### main-005（首轮抽取）

- 输入：约个打野，单双排上分，铂金段位，每小时预算60以内
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["jungle"], "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "platinum", "budget_per_hour": 60}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "rank_requirement": "platinum", "service_type": "climb", "budget_per_hour": 60, "role_preference": "top"}, "confirmation": "none"}`
- 协议错误 `enum`：delta.role_preference.function-after[_unique_roles(), list[str-enum[Role]]]: Input should be a valid array；delta.role_preference.literal['any']: Input should be 'any'

### main-033（候选引用）

- 输入：最后那个看着不错
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "小鹿乱撞"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"companion_name": "最后那个"}, "confirmation": "none"}`
- `delta.companion_name` wrong_value：期望 `"小鹿乱撞"`，实际 `"最后那个"`

### main-035（候选引用）

- 输入：第三个吧，再把时长改成三小时
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "小鹿乱撞", "duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"duration_hours": 3, "companion_name": "第三个"}, "confirmation": "none"}`
- `delta.companion_name` wrong_value：期望 `"小鹿乱撞"`，实际 `"第三个"`

### main-036（候选引用）

- 输入：1号吧
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "星河滚烫"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"companion_name": "1号吧"}, "confirmation": "none"}`
- `delta.companion_name` wrong_value：期望 `"星河滚烫"`，实际 `"1号吧"`

### main-048（确认 / 拒绝）

- 输入：确认
- 期望：`{"turn_intent": "booking", "delta": {}, "confirmation": "yes"}`
- 输出：`{"turn_intent": "confirmation", "delta": {}, "confirmation": "yes"}`
- 协议错误 `enum`：turn_intent: Input should be 'booking', 'consult' or 'unrelated'

### main-057（模式说法与黑话）

- 输入：带我上分，双排，jg位，翡翠，明天下午两点三小时
- 期望：`{"turn_intent": "booking", "delta": {"service_type": "climb", "game_mode": "ranked_solo_duo", "role_preference": ["jungle"], "rank_requirement": "emerald", "start_time_expr": "明天下午两点", "duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明天下午两点", "duration_hours": 3, "rank_requirement": "emerald", "role_preference": "jungle"}, "confirmation": "none"}`
- 协议错误 `enum`：delta.role_preference.function-after[_unique_roles(), list[str-enum[Role]]]: Input should be a valid array；delta.role_preference.literal['any']: Input should be 'any'

### main-060（模式说法与黑话）

- 输入：找个会玩辅助位的小哥哥陪我单排上分，黄金，周六晚上七点，两小时
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["support"], "companion_gender": "male", "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "gold", "start_time_expr": "周六晚上七点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "周六晚上七点", "duration_hours": 2, "rank_requirement": "gold", "role_preference": ["support"]}, "confirmation": "none"}`
- `delta.companion_gender` missing_key：期望 `"male"`，实际 `"<absent>"`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`
