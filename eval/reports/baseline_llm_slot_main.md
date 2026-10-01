# L2 槽位抽取评测：llm / main

| 项 | 值 |
|---|---|
| 抽取器 | `llm`（deepseek-chat） |
| 数据集 | `eval\datasets\slot_main.jsonl`（60 条，sha256 `ba0a7e27dca3`） |
| 时间 | 2026-10-01 10:24:10 |
| 通过标准 | 协议通过且任务分 ≥ 0.95 |
| 风格匹配 | EmbeddingStyleMatcher（阈值 0.7） |

## 总览

| 指标 | 值 |
|---|---:|
| **通过率** | **98.3%**（59/60） |
| 协议通过率 | 100.0% |
| 平均任务分（协议通过样本） | 0.996 |
| 重试 / 降级次数 | 0 / 0 |
| 延迟 P50 / P90 / P95 / max（ms） | 775 / 1014 / 1099 / 1332 |

## 分类别通过率

| 类别 | 通过 | 通过率 |
|---|---:|---:|
| 首轮抽取 `first_turn` | 8/8 | 100.0% |
| 多轮增量 `multi_turn` | 7/7 | 100.0% |
| 三态语义 `tri_state` | 7/8 | 87.5% |
| 相对时间表达 `relative_time` | 7/7 | 100.0% |
| 候选引用 `candidate_ref` | 6/6 | 100.0% |
| 插话咨询 `consult` | 6/6 | 100.0% |
| 无关话题 `unrelated` | 5/5 | 100.0% |
| 确认 / 拒绝 `confirmation` | 7/7 | 100.0% |
| 模式说法与黑话 `mode_slang` | 6/6 | 100.0% |

## 协议错误

无。

## 逐字段错误分布

| 字段 | 漏抽 missing_key | 多抽 extra_key | 值错 wrong_value | 合计 |
|---|---:|---:|---:|---:|
| `delta.style_preference` | 0 | 0 | 1 | 1 |

## 错误样例（1/1）

### main-023（三态语义）

- 输入：风格没要求，段位改成大师
- 期望：`{"turn_intent": "booking", "delta": {"style_preference": "any", "rank_requirement": "master"}, "confirmation": "none"}`
- 输出：`{"turn_intent": "booking", "delta": {"style_preference": null, "rank_requirement": "master"}, "confirmation": "none"}`
- `delta.style_preference` wrong_value：期望 `"any"`，实际 `null`
