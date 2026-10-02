# r003 策略

- 对比集 `training/configs/data/contrast_v0.4.yaml`（200 个任务）：dense_service_role 20、rounds_vs_duration 10、
  role_replace 20（目标列表换掉一个位置而不是追加）、any_vs_null 35、plain_any 15、mode_soft_withdraw 20、
  hao_number 25、ordinal_then_change 25、platform_policy 30。生成 198 / 200（2 条截断），无校验失败。
- `sft/v0.4` = v0.2 + r002 对比集 + r003 对比集 → 1170 条（train 1110 / val 60），参数不变。
- 只训练、再单独合并评测（训练中途会话结束过一次，重新启动前确认无残留进程）。
