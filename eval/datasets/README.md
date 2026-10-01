# 评测数据集

| 文件 | 层 | 来源 | 说明 |
|---|---|---|---|
| `slot_main.jsonl` | L2 槽位抽取主集 | 人工编写 | 60 条，每轮训练后跑 |
| `slot_holdout.jsonl` | L2 槽位抽取 holdout | 人工编写 | 27 条，只在里程碑时看，不参与调参 |
| `CHECKSUMS` | L2 | `validate_datasets.py --write-checksums` | 两个 L2 集合的 sha256（`sha256sum` 格式） |
| `rag_golden.jsonl` | L3 RAG | `scripts/gen_rag_golden.py` 生成 | 见 DEV_SPEC E6 |

L2 集合**不使用任何 Teacher 生成**，训练前冻结；造数（阶段 I）不得读取这两个文件。

**审核状态**：v1 由 Claude 起草（2026-10-01），待人工逐条审核。审核中如有修改，重跑
`validate_datasets.py --write-checksums`，并重跑 DeepSeek 基线（H5）。

## L2 样本格式

每行一个 JSON 对象：

```json
{
  "id": "main-031",
  "category": "candidate_ref",
  "now": "2026-10-01 14:00",
  "current_state": {"game_mode": "ranked_solo_duo", "start_time": "2026-10-02 20:00", "duration_hours": 2, "rank_requirement": "diamond"},
  "candidates": ["阿狸酱", "夜雨声烦", "小鹿乱撞"],
  "pending_confirmation": false,
  "history": [{"role": "user", "content": "钻石以上"}, {"role": "assistant", "content": "…想约哪位？"}],
  "user_input": "就第二个吧",
  "expected": {"turn_intent": "booking", "delta": {"companion_name": "夜雨声烦"}, "confirmation": "none"},
  "note": "可选：标注理由"
}
```

- 输入字段与线上 `ExtractionContext` 一一对应（`current_state` 只允许槽位字段与 `start_time_expr`），
  Runner 用 `rift_agent.extractors.prompt` 渲染成与线上逐字节相同的 prompt。
- `expected` 必须通过 C2 严格 schema（`parse_extraction`），`delta` 严格遵守三态：
  没提到的键**不出现**，"不限" 为 `"any"`，撤回为 `null`。
- `candidates`、`pending_confirmation`、`history` 为空时可省略；`history` 以助手消息结尾。

## 类别分布

| 类别 | key | main | holdout |
|---|---|---:|---:|
| 首轮抽取 | `first_turn` | 8 | 3 |
| 多轮增量 | `multi_turn` | 7 | 3 |
| 三态语义 | `tri_state` | 8 | 3 |
| 相对时间表达 | `relative_time` | 7 | 3 |
| 候选引用 | `candidate_ref` | 6 | 3 |
| 插话咨询 | `consult` | 6 | 3 |
| 无关话题 | `unrelated` | 5 | 3 |
| 确认 / 拒绝 | `confirmation` | 7 | 3 |
| 模式说法与黑话 | `mode_slang` | 6 | 3 |
| **合计** | | **60** | **27** |

## 标注约定（易错点）

- **时间**：`start_time_expr` 照抄原话中的时间部分（"改成晚一小时" 标为 "晚一小时"）；评分时两边都经
  C3 解析器（结合 `now` 与 `current_state.start_time`）换算成绝对时间再比较，所以 "明晚8点" 与
  "明天晚上八点" 等价。所有标注的时间表达都能被解析器解析到预期时刻。
- **服务类型**：只有用户明说（上分 / 带飞 / 娱乐 / 教学 / 复盘）才标 `service_type`，排位默认 climb
  由代码推断，不进 delta。
- **黑话**：双排 / 单排 → `ranked_solo_duo`，五排 → `ranked_flex`，斗魂 → `arena`，jg → `jungle`，
  小姐姐 / 妹子 → `female`，小哥哥 → `male`。
- **"不限" vs 撤回**："无所谓 / 都行 / 不限 / 没要求 / 换谁都行" 标 `"any"`，对非必填字段（风格、
  指定陪玩师、性别等）同样适用；"先不定 / 再想想 / 先别定" 标 `null`。
- **声音**："声音好听" 一类说法是风格偏好（`style_preference`），只有 "开麦 / 能语音" 才标 `voice_required`。
- **候选引用**：输出 `candidates` 中的确切名字（"1号""最后那个" 也一样）。
- **确认**：`yes` 只用于回答待确认订单（`pending_confirmation: true`）；"改成三小时再下单" 标为
  `delta + yes`（代码先合并、重新报价）；没有待确认订单时的 "不约了" 标 `no`（显式放弃）。
  拒绝确认（"不了，换一个人吧"）只标 `confirmation: no`，delta 为空，不撤回 `companion_name`，换人由代码处理。
- **consult / unrelated** 轮次 `delta` 为空。

## 校验与冻结

```bash
uv run python eval/runners/validate_datasets.py                    # schema、一致性、分布、checksum
uv run python eval/runners/validate_datasets.py --write-checksums  # 人工审核通过后冻结
```

修改任一 L2 样本都必须重新审核并重写 `CHECKSUMS`；Runner 在 checksum 不符时会给出警告并写入报告。
