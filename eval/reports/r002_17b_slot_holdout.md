# L2 槽位抽取评测：local / holdout

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_holdout.jsonl`（27 条，sha256 `c0d49233b0b4`） |
| 时间 | 2026-10-01 23:36:33 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **88.9%**（24/27） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.973 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 2291 / 4439 / 5251 / 6290 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 2/3 | 66.7% |
| 多轮增量 `multi_turn` | 3/3 | 100.0% |
| 三态语义 `tri_state` | 3/3 | 100.0% |
| 相对时间表达 `relative_time` | 3/3 | 100.0% |
| 候选引用 `candidate_ref` | 3/3 | 100.0% |
| 插话咨询 `consult` | 3/3 | 100.0% |
| 无关话题 `unrelated` | 3/3 | 100.0% |
| 确认 / 拒绝 `confirmation` | 3/3 | 100.0% |
| 模式说法与黑话 `mode_slang` | 1/3 | 33.3% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.game_mode` | 1 | 0 | 0 | 1 |
| `delta.style_preference` | 1 | 0 | 0 | 1 |
| `delta.voice_required` | 0 | 1 | 0 | 1 |
| `delta.duration_hours` | 0 | 0 | 1 | 1 |
| `delta.service_type` | 1 | 0 | 0 | 1 |

## 错误样例（3/3）

### holdout-002（首轮抽取）

- 输入：我想找个声音好听的妹子陪我玩匹配，今晚9点半，一小时
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "声音好听", "companion_gender": "female", "game_mode": "normal_draft", "start_time_expr": "今晚9点半", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"start_time_expr": "今晚9点半", "duration_hours": 1, "companion_gender": "female", "voice_required": true}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `"normal_draft"`，实际 `"<absent>"`
- `delta.style_preference` missing_key：期望 `"声音好听"`，实际 `"<absent>"`
- `delta.voice_required` extra_key：期望 `"<absent>"`，实际 `true`

### holdout-025（模式说法与黑话）

- 输入：今晚十点来两把大乱斗，一个小时
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "今晚十点", "game_mode": "aram", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "aram", "start_time_expr": "今晚十点", "duration_hours": 2}, "confirmation": "none"}`
- `delta.duration_hours` wrong_value：期望 `1.0`，实际 `2.0`

### holdout-026（模式说法与黑话）

- 输入：带飞，单双排，翡翠，周五晚上8点，两小时
- 期望：`{"turn_intent": "booking", "delta": {"service_type": "climb", "game_mode": "ranked_solo_duo", "rank_requirement": "emerald", "start_time_expr": "周五晚上8点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "周五晚上8点", "duration_hours": 2, "rank_requirement": "emerald"}, "confirmation": "none"}`
- `delta.service_type` missing_key：期望 `"climb"`，实际 `"<absent>"`
