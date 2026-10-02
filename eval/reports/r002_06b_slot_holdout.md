# L2 槽位抽取评测：local / holdout

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_holdout.jsonl`（27 条，sha256 `c0d49233b0b4`） |
| 时间 | 2026-10-01 23:33:05 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **88.9%**（24/27） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.972 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 936 / 2013 / 2067 / 3715 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 2/3 | 66.7% |
| 多轮增量 `multi_turn` | 3/3 | 100.0% |
| 三态语义 `tri_state` | 2/3 | 66.7% |
| 相对时间表达 `relative_time` | 3/3 | 100.0% |
| 候选引用 `candidate_ref` | 3/3 | 100.0% |
| 插话咨询 `consult` | 3/3 | 100.0% |
| 无关话题 `unrelated` | 3/3 | 100.0% |
| 确认 / 拒绝 `confirmation` | 3/3 | 100.0% |
| 模式说法与黑话 `mode_slang` | 2/3 | 66.7% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.game_mode` | 2 | 0 | 0 | 2 |
| `delta.style_preference` | 1 | 0 | 0 | 1 |
| `delta.companion_name` | 0 | 1 | 0 | 1 |

## 错误样例（3/3）

### holdout-002（首轮抽取）

- 输入：我想找个声音好听的妹子陪我玩匹配，今晚9点半，一小时
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "声音好听", "companion_gender": "female", "game_mode": "normal_draft", "start_time_expr": "今晚9点半", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"start_time_expr": "今晚9点半", "duration_hours": 1, "companion_gender": "female"}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `"normal_draft"`，实际 `"<absent>"`
- `delta.style_preference` missing_key：期望 `"声音好听"`，实际 `"<absent>"`

### holdout-009（三态语义）

- 输入：模式我再想想，先别定
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": null}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {}, "confirmation": "none"}`
- `delta.game_mode` missing_key：期望 `null`，实际 `"<absent>"`

### holdout-027（模式说法与黑话）

- 输入：想约个打中路的陪我五排，明天下午四点，一小时
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["mid"], "game_mode": "ranked_flex", "start_time_expr": "明天下午四点", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_flex", "start_time_expr": "明天下午四点", "duration_hours": 1, "role_preference": ["mid"], "companion_name": "any"}, "confirmation": "none"}`
- `delta.companion_name` extra_key：期望 `"<absent>"`，实际 `"any"`
