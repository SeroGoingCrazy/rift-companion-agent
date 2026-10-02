# DPO 消融（DEV_SPEC I9）

2026-10-02。起点：SFT r003 0.6B（`sft/v0.4`，merged）。偏好数据：`dpo/rule_v1`（1084 对）、`dpo/onpolicy_v1`（316 对），
均只取自 SFT train 部分。参数：LoRA r16 / α32，lr 5e-6，1 epoch，sigmoid，有效 batch 16（单卡 1 × 16；batch 2 时
参考模型的全词表 log_softmax 触发 `CUDA driver error: device not ready`）。评测：CPU F16 llama-server，与 SFT 相同的 L2 流程。
报告：`eval/reports/dpo_{rule,onpolicy}_{b01,b03}_06b_slot_{main,holdout}.*`，L4：`dpo_rule_b03_06b_e2e.*`。

## 消融表

| 配置 | 步数 | 最终 DPO loss | L2 主集 | L2 holdout | 主集协议 | 输出与 SFT 相同 |
|---|---:|---:|---:|---:|---:|---:|
| SFT r003（基准） | – | – | 90.0%（54/60） | 92.6%（25/27） | 100% | – |
| rule, β = 0.1 | 68 | 0.445 | **93.3%**（56/60） | 88.9%（24/27） | 98.3% | 55/60、25/27 |
| rule, β = 0.3 | 68 | 0.204 | **93.3%**（56/60） | 88.9%（24/27） | 98.3% | 55/60、25/27 |
| onpolicy, β = 0.1 | 20 | 0.689 | 90.0%（54/60） | 92.6%（25/27） | 100% | 60/60、27/27 |
| onpolicy, β = 0.3 | 20 | 0.683 | 90.0%（54/60） | 92.6%（25/27） | 100% | 60/60、27/27 |

DeepSeek 基线 98.3% / 88.9%；M2 门槛 0.6B 主集 ≥ 93.3%、holdout ≥ 85%。

## 提升与退化在哪些字段（rule，两个 β 完全一致）

| 样本 | 变化 | 字段 | 对应的扰动 |
|---|---|---|---|
| main-023 "风格没要求，段位改成大师" | 修好：`null` → `"any"` | style_preference | any_to_null |
| main-036 "1号吧" | 修好：照抄 → 候选名"星河滚烫" | companion_name | name_literal |
| main-005 "约个打野，单双排上分…" | 修好：补上 role_preference | role_preference | drop_key |
| main-008 "男生女生都行" | **新错**：`"male or female"`（枚举越界，协议错误） | companion_gender | any_to_null / any_to_absent 的反面 |
| holdout-027 "想约个打中路的…" | **新错**：mid → adc | role_preference | value_swap 的副作用 |
| main-035 "第三个吧，再把时长改成三小时" | 仍错，但从漏选人变成照抄 `"第三个"` | companion_name | drop_key 教会了"别省略"，没教会"要解析" |

修好的 3 条正是扰动设计瞄准的边界（any/null、照抄、漏键）；退化的 2 条是 DPO 的典型副作用：把 rejected 的概率压下去，
概率质量流向了训练分布之外的输出（`"male or female"` 这种 SFT 从未产生过的值）。main-012（列表替换）在训练集中只有 12 对
role_append，没有变化。

## 结论

1. **规则扰动 DPO 有效但有代价**：主集 +3 / −1，达到 M2 主集门槛（93.3%），holdout −1（仍 ≥ 85%）；引入 1 个协议错误。
   J1 计划的 grammar 约束解码（由 C2 JSON Schema 生成）会把枚举越界直接挡掉，路由器的重试 / 降级也能兜底，线上影响有限。
2. **β 不敏感**：0.1 与 0.3 在 87 条评测上的输出完全相同（loss 0.445 vs 0.204，训练集上的偏好幅度不同，但没改变 argmax）。
   选 β = 0.3 作为最终模型：结果相同时，KL 约束更强、离 SFT 更近更稳妥。
3. **On-Policy DPO 本轮无效，原因是信号太少而不是方法不行**：r003 在自己的训练集上 t=0.7 采样通过率 96.6%，
   提温到 t=1.0、k=8 也只挖出 316 对；1 epoch 仅 20 步、lr 5e-6，loss 停在 0.69（≈ ln 2，几乎没动），评测输出与 SFT 逐字相同。
   公平比较需要相近的更新量（例如 3 epoch 或 lr 5e-5），或在训练集之外的提示上采样（例如 L4 / 影子模式的真实对话，K2 的样本挖掘）。
4. **与上个项目"规则扰动 DPO 无效"的对照**：上个项目的扰动是通用的字段替换，rejected 与 chosen 差别大、模型本来就不会犯，
   偏好信号几乎为零；这里的扰动逐条对准 SFT 三轮里修不掉的具体错误（"同一句话、两种标签"的边界），rejected 都是
   协议合法、模型真的会产生的错误，因此能改动 argmax。代价也来自同一点：只告诉模型"不要什么"，没有约束"要什么"，
   会把概率推到分布外（main-008）。

## L4（最优组 rule β = 0.3，`--config local`）

完成率 **20/22（90.9%）**，平均 3.32 轮，单轮 P50 / P95 1346 / 5144 ms（CPU F16），远程 LLM 调用 **1.41 次 / 会话**
（DeepSeek 基线 4.14）。两个失败剧本都是段位说法：

- `double_session_same_slot`："挑战者"（`domain.yaml` 只有"王者 / 最强王者"别名）→ 抽取失败；
- `relax_rank_gap`："宗师以上" → `challenger`（应为 grandmaster）。

用 SFT r003 单独重跑这两个剧本，失败方式完全相同，属于 SFT 已有的段位别名问题，不是 DPO 引入的退化。
后续：在 `domain.yaml` 与抽取 prompt 中补充"挑战者"等别名，对比集加入宗师 / 王者 / 大师的区分。
