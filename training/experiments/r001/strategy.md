# r001 策略

## 数据

- `training/data/processed/sft/v0.2`：782 条（train 743 / val 39），specs v0.2 + gpt-5 Teacher，
  见同目录 `DATA_CARD.md`。已剔除与 L2 主集 / holdout 相似度 ≥ 0.92 的 11 条。
- 每条样本是线上抽取器的完整 prompt（system + 上下文 JSON）加标准答案，只在答案上计算 loss。
  样本长度中位数 949 token、最长 1345；答案中位数 29 token、最长 74。

## 训练（LLaMA-Factory 0.9.5，WSL2 + RTX 5080 16GB，torch 2.11+cu128）

| 项 | 0.6B | 1.7B |
|---|---|---|
| 配置 | `sft_0.6b_r001.yaml` | `sft_1.7b_r001.yaml` |
| LoRA | r16 / α32 / dropout 0.05 / all linear | 同左 |
| lr / 调度 | 1e-4，cosine，warmup 10% | 同左 |
| epoch | 3 | 3 |
| 有效 batch | 16（8 × 2） | 16 |
| 精度 | bf16 | bf16 |

## 模板（与线上推理一致）

`template: qwen3` + `enable_thinking: false`，训练渲染与官方 chat template
（`enable_thinking=False`，assistant 前带空的 `<think>\n\n</think>\n\n`）逐 token 一致：
`training/scripts/train/check_template.py` 在 743 条训练样本上 0 处不一致。
**约束**：线上 llama-server 调用必须传 `chat_template_kwargs: {"enable_thinking": false}`（J1）。

## 评测（I6）

merge adapter → GGUF F16 → llama-server → `run_slot_eval.py --extractor local`，
main + holdout，与基座模型和 DeepSeek 基线对比。
