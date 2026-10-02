# r002 策略

- 对比集 `training/configs/data/contrast_v0.3.yaml`（kind: contrast，200 个任务，gpt-5 Teacher）：
  compact_dense 60、after_vs_duration 20、withdraw_mode 25、any_x_not_x 35、ordinal_forms 30、
  ordinal_and_change 15、near_platform 10、errand 5。提示里的示例刻意避开 L2 原句。
- 生成 196 / 200 通过（2 条被拒是解析器不认识的分钟说法"90分钟后""一小时半后"，2 条截断）；
  人工删 1 条把提示模板"X不X"照抄进话术的样本。
- `sft/v0.3` = v0.2（782）+ 对比集（195）→ 去重 2、剔除与评测集过近 14 条（含一句与 main-022 完全相同的
  "开不开麦都无所谓"）→ 974 条（train 924 / val 50）。
- 训练参数与 r001 完全相同（`sft_{0.6b,1.7b}_r002.yaml`），`run_round.sh` 一次跑完训练、合并、转 GGUF。
