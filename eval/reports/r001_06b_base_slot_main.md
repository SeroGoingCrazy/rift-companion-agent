# L2 槽位抽取评测：local / main

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_main.jsonl`（60 条，sha256 `ba0a7e27dca3`） |
| 时间 | 2026-10-01 21:55:38 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **0.0%**（0/60） |
| 协议通过率 | 1.7% |
| 平均任务分（协议通过样本） | 0.077 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 1141 / 4195 / 5729 / 8287 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 0/8 | 0.0% |
| 多轮增量 `multi_turn` | 0/7 | 0.0% |
| 三态语义 `tri_state` | 0/8 | 0.0% |
| 相对时间表达 `relative_time` | 0/7 | 0.0% |
| 候选引用 `candidate_ref` | 0/6 | 0.0% |
| 插话咨询 `consult` | 0/6 | 0.0% |
| 无关话题 `unrelated` | 0/5 | 0.0% |
| 确认 / 拒绝 `confirmation` | 0/7 | 0.0% |
| 模式说法与黑话 `mode_slang` | 0/6 | 0.0% |

## 协议错误

| 类型 | 次数 |
|---|---:|
| `extra_key` | 58 |
| `json` | 1 |

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `turn_intent` | 0 | 0 | 1 | 1 |
| `confirmation` | 0 | 0 | 1 | 1 |
| `delta.budget_per_hour` | 0 | 1 | 0 | 1 |
| `delta.companion_gender` | 0 | 1 | 0 | 1 |
| `delta.duration_hours` | 0 | 1 | 0 | 1 |
| `delta.game_mode` | 0 | 1 | 0 | 1 |
| `delta.rank_requirement` | 0 | 1 | 0 | 1 |
| `delta.role_preference` | 0 | 1 | 0 | 1 |
| `delta.service_type` | 0 | 1 | 0 | 1 |
| `delta.start_time_expr` | 0 | 1 | 0 | 1 |
| `delta.style_preference` | 0 | 1 | 0 | 1 |
| `delta.voice_required` | 0 | 1 | 0 | 1 |

## 错误样例（15/60）

### main-001（首轮抽取）

- 输入：帮我约个明晚八点的单双排，钻石以上，打两个小时
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "明晚八点", "rank_requirement": "diamond", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"user_input": "帮我约个明晚八点的单双排，钻石以上，打两个小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-002（首轮抽取）

- 输入：想找个小姐姐陪我玩大乱斗，今晚十点开始，一个小时
- 期望：`{"turn_intent": "booking", "delta": {"companion_gender": "female", "game_mode": "aram", "start_time_expr": "今晚十点", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "想找个小姐姐陪我玩大乱斗，今晚十点开始，一个小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-003（首轮抽取）

- 输入：周六下午三点想约个教学，灵活组排，三个小时，要能开麦的
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "周六下午三点", "service_type": "coaching", "game_mode": "ranked_flex", "duration_hours": 3, "voice_required": true}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "周六下午三点想约个教学，灵活组排，三个小时，要能开麦的"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-004（首轮抽取）

- 输入：有没有温柔一点会聊天的陪玩，明天晚上9点打匹配
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "温柔会聊天", "start_time_expr": "明天晚上9点", "game_mode": "normal_draft"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "有没有温柔一点会聊天的陪玩，明天晚上9点打匹配"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-005（首轮抽取）

- 输入：约个打野，单双排上分，铂金段位，每小时预算60以内
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["jungle"], "game_mode": "ranked_solo_duo", "service_type": "climb", "rank_requirement": "platinum", "budget_per_hour": 60}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "约个打野，单双排上分，铂金段位，每小时预算60以内"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-006（首轮抽取）

- 输入：我想约阿狸酱明晚八点双排两个半小时
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "阿狸酱", "start_time_expr": "明晚八点", "game_mode": "ranked_solo_duo", "duration_hours": 2.5}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"user_input": "我想约阿狸酱明晚八点双排两个半小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-007（首轮抽取）

- 输入：你好，我想约个陪玩
- 期望：`{"turn_intent": "booking", "delta": {}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "你好，我想约个陪玩"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-008（首轮抽取）

- 输入：后天下午四点半斗魂竞技场，男生女生都行，打一个半小时
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "后天下午四点半", "game_mode": "arena", "companion_gender": "any", "duration_hours": 1.5}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "后天下午四点半斗魂竞技场，男生女生都行，打一个半小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-009（多轮增量）

- 输入：明晚八点吧，打两个小时
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "明晚八点", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "明晚八点吧，打两个小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-010（多轮增量）

- 输入：至少要翡翠
- 期望：`{"turn_intent": "booking", "delta": {"rank_requirement": "emerald"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "至少要翡翠", "rank_requirement": "any", "role_preference": ["gold", "emerald"], "companion_name": "none", "companion_gender": "female", "service_type": "none", "voice_required": false, "budget_per_hour": null, "style_preference": "温柔会聊天", "companion_gender": "female"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted；delta.role_preference.function-after[_unique_roles(), list[str-enum[Role]]].0: Input should be 'top', 'jungle', 'mid', 'adc' or 'support'；delta.role_preference.function-after[_unique_roles(), list[str-enum[Role]]].1: Input should be 'top', 'jungle', 'mid', 'adc' or 'support'；delta.role_preference.literal['any']: Input should be 'any'；delta.service_type: Input should be 'climb', 'casual' or 'coaching'

### main-011（多轮增量）

- 输入：再加个条件，要女生，能开麦
- 期望：`{"turn_intent": "booking", "delta": {"companion_gender": "female", "voice_required": true}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "再加个条件，要女生，能开麦"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-012（多轮增量）

- 输入：打野换成辅助吧
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["support"]}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "打野换成辅助吧"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-013（多轮增量）

- 输入：时间加到三个小时
- 期望：`{"turn_intent": "booking", "delta": {"duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "时间加到三个小时"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-014（多轮增量）

- 输入：预算一小时50块左右
- 期望：`{"turn_intent": "booking", "delta": {"budget_per_hour": 50}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "预算一小时50块左右"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### main-015（多轮增量）

- 输入：中单和上单都可以
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["mid", "top"]}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "中单和上单都可以"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted
