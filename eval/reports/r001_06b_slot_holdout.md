# L2 槽位抽取评测：local / holdout

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_holdout.jsonl`（27 条，sha256 `c0d49233b0b4`） |
| 时间 | 2026-10-01 21:54:42 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **74.1%**（20/27） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.927 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 916 / 1650 / 1782 / 4263 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 1/3 | 33.3% |
| 多轮增量 `multi_turn` | 3/3 | 100.0% |
| 三态语义 `tri_state` | 2/3 | 66.7% |
| 相对时间表达 `relative_time` | 3/3 | 100.0% |
| 候选引用 `candidate_ref` | 2/3 | 66.7% |
| 插话咨询 `consult` | 3/3 | 100.0% |
| 无关话题 `unrelated` | 2/3 | 66.7% |
| 确认 / 拒绝 `confirmation` | 3/3 | 100.0% |
| 模式说法与黑话 `mode_slang` | 1/3 | 33.3% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.game_mode` | 2 | 0 | 0 | 2 |
| `delta.service_type` | 2 | 0 | 0 | 2 |
| `delta.style_preference` | 1 | 0 | 0 | 1 |
| `delta.rank_requirement` | 1 | 0 | 0 | 1 |
| `delta.companion_name` | 1 | 0 | 0 | 1 |
| `turn_intent` | 0 | 0 | 1 | 1 |
| `delta.role_preference` | 1 | 0 | 0 | 1 |

## 错误样例（7/7）

### holdout-002（首轮抽取）

- 输入：我想找个声音好听的妹子陪我玩匹配，今晚9点半，一小时
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "声音好听", "companion_gender": "female", "game_mode": "normal_draft", "start_time_expr": "今晚9点半", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"start_time_expr": "今晚9点半", "duration_hours": 1, "companion_gender": "female"}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `"normal_draft"`，实际 `"<absent>"`
- `delta.style_preference` missing_key：期望 `"声音好听"`，实际 `"<absent>"`

### holdout-003（首轮抽取）

- 输入：约个单双排的教学复盘，钻石，明天中午12点，一个半小时
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "service_type": "coaching", "rank_requirement": "diamond", "start_time_expr": "明天中午12点", "duration_hours": 1.5}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明天中午12点", "duration_hours": 1.5}, "confirmation": "none"}`
- `delta.rank_requirement` missing_key：期望 `"diamond"`，实际 `"<absent>"`
- `delta.service_type` missing_key：期望 `"coaching"`，实际 `"<absent>"`

### holdout-009（三态语义）

- 输入：模式我再想想，先别定
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": null}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `null`，实际 `"<absent>"`

### holdout-015（候选引用）

- 输入：就选第一个，时间改成明晚九点
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "咕噜咕噜", "start_time_expr": "明晚九点"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"start_time_expr": "明晚九点"}, "confirmation": "none"}`
- `delta.companion_name` missing_key：期望 `"咕噜咕噜"`，实际 `"<absent>"`

### holdout-020（无关话题）

- 输入：帮我查一下明天去上海的航班
- 期望：`{"turn_intent": "unrelated", "delta": {}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {}, "confirmation": "none"}`
- `turn_intent` wrong_value：期望 `"unrelated"`，实际 `"consult"`

### holdout-026（模式说法与黑话）

- 输入：带飞，单双排，翡翠，周五晚上8点，两小时
- 期望：`{"turn_intent": "booking", "delta": {"service_type": "climb", "game_mode": "ranked_solo_duo", "rank_requirement": "emerald", "start_time_expr": "周五晚上8点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "周五晚上8点", "duration_hours": 2, "rank_requirement": "emerald"}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`

### holdout-027（模式说法与黑话）

- 输入：想约个打中路的陪我五排，明天下午四点，一小时
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["mid"], "game_mode": "ranked_flex", "start_time_expr": "明天下午四点", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_flex", "start_time_expr": "明天下午四点", "duration_hours": 1}, "confirmation": "none"}`
- `delta.role_preference` missing_key：期望 `["mid"]`，实际 `"<absent>"`
