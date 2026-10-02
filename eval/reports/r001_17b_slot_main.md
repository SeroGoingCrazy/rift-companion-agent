# L2 槽位抽取评测：local / main

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_main.jsonl`（60 条，sha256 `ba0a7e27dca3`） |
| 时间 | 2026-10-01 21:58:40 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **81.7%**（49/60） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.946 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 2302 / 4353 / 4809 / 7507 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 5/8 | 62.5% |
| 多轮增量 `multi_turn` | 7/7 | 100.0% |
| 三态语义 `tri_state` | 6/8 | 75.0% |
| 相对时间表达 `relative_time` | 7/7 | 100.0% |
| 候选引用 `candidate_ref` | 5/6 | 83.3% |
| 插话咨询 `consult` | 6/6 | 100.0% |
| 无关话题 `unrelated` | 5/5 | 100.0% |
| 确认 / 拒绝 `confirmation` | 6/7 | 85.7% |
| 模式说法与黑话 `mode_slang` | 2/6 | 33.3% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.service_type` | 4 | 1 | 0 | 5 |
| `delta.start_time_expr` | 0 | 0 | 3 | 3 |
| `delta.game_mode` | 2 | 0 | 0 | 2 |
| `delta.duration_hours` | 2 | 0 | 0 | 2 |
| `delta.role_preference` | 1 | 0 | 0 | 1 |
| `delta.voice_required` | 0 | 0 | 1 | 1 |
| `delta.style_preference` | 1 | 0 | 0 | 1 |
| `delta.companion_name` | 0 | 0 | 1 | 1 |
| `confirmation` | 0 | 0 | 1 | 1 |

## 错误样例（11/11）

### main-001（首轮抽取）

- 输入：帮我约个明晚八点的单双排，钻石以上，打两个小时
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明晚八点", "rank_requirement": "diamond", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"start_time_expr": "明晚八点", "duration_hours": 2, "rank_requirement": "diamond"}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `"ranked_solo_duo"`，实际 `"<absent>"`

### main-004（首轮抽取）

- 输入：有没有温柔一点会聊天的陪玩，明天晚上9点打匹配
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "温柔会聊天", "start_time_expr": "明天晚上9点", "game_mode": "normal_draft"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"start_time_expr": "明天晚上9点", "service_type": "casual", "style_preference": "温柔会聊天"}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `"normal_draft"`，实际 `"<absent>"`
- `delta.service_type` extra_key：期望 `"<absent>"`，实际 `"casual"`

### main-005（首轮抽取）

- 输入：约个打野，单双排上分，铂金段位，每小时预算60以内
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["jungle"], "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "platinum", "budget_per_hour": 60}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "rank_requirement": "platinum", "budget_per_hour": 60}, "confirmation": "none"}`
- `delta.role_preference` missing_key：期望 `["jungle"]`，实际 `"<absent>"`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`

### main-022（三态语义）

- 输入：开不开麦都无所谓
- 期望：`{"turn_intent": "booking", "delta": {"voice_required": "any"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"voice_required": null}, "confirmation": "none"}`
- `delta.voice_required` wrong_value：期望 `"any"`，实际 `null`

### main-023（三态语义）

- 输入：风格没要求，段位改成大师
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "any", "rank_requirement": "master"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"rank_requirement": "master"}, "confirmation": "none"}`
- `delta.style_preference` missing_key：期望 `"any"`，实际 `"<absent>"`

### main-036（候选引用）

- 输入：1号吧
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "星河滚烫"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"companion_name": "1号"}, "confirmation": "none"}`
- `delta.companion_name` wrong_value：期望 `"星河滚烫"`，实际 `"1号"`

### main-053（确认 / 拒绝）

- 输入：改成三小时再下单
- 期望：`{"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "yes"}`
- 输出：`{"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "none"}`
- `confirmation` wrong_value：期望 `"yes"`，实际 `"none"`

### main-057（模式说法与黑话）

- 输入：带我上分，双排，jg位，翡翠，明天下午两点三小时
- 期望：`{"turn_intent": "booking", "delta": {"service_type": "climb", "game_mode": "ranked_solo_duo", "role_preference": ["jungle"], "rank_requirement": "emerald", "start_time_expr": "明天下午两点", "duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明天下午两点三小时", "rank_requirement": "emerald", "role_preference": ["jungle"]}, "confirmation": "none"}`
- `delta.duration_hours` missing_key：期望 `3.0`，实际 `"<absent>"`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`
- `delta.start_time_expr` wrong_value：期望 `"明天下午两点"`，实际 `"明天下午两点三小时"`（2026-10-02 14:00:00 vs 2026-10-02 14:03:00）

### main-058（模式说法与黑话）

- 输入：约个斗魂，今晚十点一个小时
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": "arena", "start_time_expr": "今晚十点", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "arena", "start_time_expr": "今晚十点一个小时"}, "confirmation": "none"}`
- `delta.duration_hours` missing_key：期望 `1.0`，实际 `"<absent>"`
- `delta.start_time_expr` wrong_value：期望 `"今晚十点"`，实际 `"今晚十点一个小时"`（2026-10-01 22:00:00 vs 2026-10-01 22:01:00）

### main-059（模式说法与黑话）

- 输入：随便打打匹配，娱乐局，明晚八点两小时
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": "normal_draft", "service_type": "casual", "start_time_expr": "明晚八点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "normal_draft", "start_time_expr": "明晚八点两小时", "duration_hours": 2}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"casual"`，实际 `"<absent>"`
- `delta.start_time_expr` wrong_value：期望 `"明晚八点"`，实际 `"明晚八点两小时"`（2026-10-02 20:00:00 vs 2026-10-02 20:02:00）

### main-060（模式说法与黑话）

- 输入：找个会玩辅助位的小哥哥陪我单排上分，黄金，周六晚上七点，两小时
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["support"], "companion_gender": "male", "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "gold", "start_time_expr": "周六晚上七点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "周六晚上七点", "duration_hours": 2, "rank_requirement": "gold", "role_preference": ["support"], "companion_gender": "male"}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`
