# L2 槽位抽取评测：local / holdout

| 项 | 值 |
|---|---|
| 抽取器 | `local`（rift-slot-0.6b-q4km） |
| 数据集 | `eval\datasets\slot_holdout.jsonl`（27 条，sha256 `c0d49233b0b4`） |
| 时间 | 2026-10-01 21:57:25 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **0.0%**（0/27） |
| 协议通过率 | 3.7% |
| 平均任务分（协议通过样本） | 0.077 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 1147 / 4475 / 4602 / 5191 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 0/3 | 0.0% |
| 多轮增量 `multi_turn` | 0/3 | 0.0% |
| 三态语义 `tri_state` | 0/3 | 0.0% |
| 相对时间表达 `relative_time` | 0/3 | 0.0% |
| 候选引用 `candidate_ref` | 0/3 | 0.0% |
| 插话咨询 `consult` | 0/3 | 0.0% |
| 无关话题 `unrelated` | 0/3 | 0.0% |
| 确认 / 拒绝 `confirmation` | 0/3 | 0.0% |
| 模式说法与黑话 `mode_slang` | 0/3 | 0.0% |

## 协议错误

| 类型 | 次数 |
|---|---:|
| `extra_key` | 25 |
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

## 错误样例（15/27）

### holdout-001（首轮抽取）

- 输入：下周一晚上七点约个灵活组排，大师以上，两小时
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "下周一晚上七点", "game_mode": "ranked_flex", "rank_requirement": "master", "duration_hours": 2}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "下周一晚上七点约个灵活组排，大师以上，两小时"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-002（首轮抽取）

- 输入：我想找个声音好听的妹子陪我玩匹配，今晚9点半，一小时
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "声音好听", "companion_gender": "female", "game_mode": "normal_draft", "start_time_expr": "今晚9点半", "duration_hours": 1}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "我想找个声音好听的妹子陪我玩匹配，今晚9点半，一小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-003（首轮抽取）

- 输入：约个单双排的教学复盘，钻石，明天中午12点，一个半小时
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": "ranked_solo_duo", "service_type": "coaching", "rank_requirement": "diamond", "start_time_expr": "明天中午12点", "duration_hours": 1.5}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"user_input": "约个单双排的教学复盘，钻石，明天中午12点，一个半小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-004（多轮增量）

- 输入：10月5号晚上8点，打三个小时
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "10月5号晚上8点", "duration_hours": 3}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "10月5号晚上8点，打三个小时"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-005（多轮增量）

- 输入：要会玩ADC的
- 期望：`{"turn_intent": "booking", "delta": {"role_preference": ["adc"]}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "要会玩ADC的"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-006（多轮增量）

- 输入：预算可以加到80
- 期望：`{"turn_intent": "booking", "delta": {"budget_per_hour": 80}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "预算可以加到80"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-007（三态语义）

- 输入：性别不限
- 期望：`{"turn_intent": "booking", "delta": {"companion_gender": "any"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "性别不限"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-008（三态语义）

- 输入：不指定阿狸酱了，换谁都行
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "any"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "不指定阿狸酱了，换谁都行"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-009（三态语义）

- 输入：模式我再想想，先别定
- 期望：`{"turn_intent": "booking", "delta": {"game_mode": null}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "模式我再想想，先别定"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-010（相对时间表达）

- 输入：早一个小时开始
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "早一个小时"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "早一个小时开始"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-011（相对时间表达）

- 输入：推迟半小时
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "推迟半小时"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "推迟半小时"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-012（相对时间表达）

- 输入：明晚8:30
- 期望：`{"turn_intent": "booking", "delta": {"start_time_expr": "明晚8:30"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "明晚8:30"}, "confirmation": "none"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-013（候选引用）

- 输入：第二个
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "雪落无声"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"game_mode": "ranked_solo_duo", "start_time_expr": "2026-10-02 20:00 星期五", "duration_hours": 2.0, "rank_requirement": "any", "role_preference": ["adc", "support"], "service_type": "climb", "companion_gender": "female", "voice_required": false, "budget_per_hour": 100, "style_preference": "温柔会聊天", "companion_name": "雪落无声"}, "confirmation": "yes"}`
- `turn_intent` wrong_value：期望 `"booking"`，实际 `"consult"`
- `confirmation` wrong_value：期望 `"none"`，实际 `"yes"`
- `delta.budget_per_hour` extra_key：期望 `"<absent>"`，实际 `100.0`
- `delta.companion_gender` extra_key：期望 `"<absent>"`，实际 `"female"`
- `delta.duration_hours` extra_key：期望 `"<absent>"`，实际 `2.0`
- `delta.game_mode` extra_key：期望 `"<absent>"`，实际 `"ranked_solo_duo"`
- `delta.rank_requirement` extra_key：期望 `"<absent>"`，实际 `"any"`
- `delta.role_preference` extra_key：期望 `"<absent>"`，实际 `["adc", "support"]`
- `delta.service_type` extra_key：期望 `"<absent>"`，实际 `"climb"`
- `delta.start_time_expr` extra_key：期望 `"<absent>"`，实际 `"2026-10-02 20:00 星期五"`
- `delta.style_preference` extra_key：期望 `"<absent>"`，实际 `"温柔会聊天"`
- `delta.voice_required` extra_key：期望 `"<absent>"`，实际 `false`

### holdout-014（候选引用）

- 输入：南风知我意吧，看着挺靠谱
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "南风知我意"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "南风知我意吧，看着挺靠谱"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted

### holdout-015（候选引用）

- 输入：就选第一个，时间改成明晚九点
- 期望：`{"turn_intent": "booking", "delta": {"companion_name": "咕噜咕噜", "start_time_expr": "明晚九点"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "consult", "delta": {"user_input": "就选第一个，时间改成明晚九点"}, "confirmation": "yes"}`
- 协议错误 `extra_key`：delta.user_input: Extra inputs are not permitted
