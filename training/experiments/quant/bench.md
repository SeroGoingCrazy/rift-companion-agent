# 量化基准：rift-slot-0.6b

- 时间：2026-10-02 13:33:02；机器：AMD Ryzen 5 9600X 6-Core Processor，llama.cpp `5fc4f3c`
- 源模型：`training/outputs/gguf/rift-slot-0.6b-dpo_rule_b03-f16.gguf`
- llama-server：`-t 6 -c 4096 -np 1`，temperature 0，max_tokens 160，`response_format: json_object`，thinking 关闭
- 所有变体在同一次运行中依次加载、预热共享前缀后测量；速度用 L2 全部样本的线上 prompt，质量走线上 `local` 抽取器路径（与 I6–I9 相同）。

## 质量（L2）

| 变体 | 大小 MB | main | holdout | 合计 | 协议 | 相对基准 |
|---|---:|---:|---:|---:|---:|---|
| `f16` | 1143 | 56/60（93.3%） | 24/27（88.9%） | 80/87 | 98.3% | 基准 |
| `q8_0` | 610 | 56/60（93.3%） | 24/27（88.9%） | 80/87 | 98.3% | −0 / +0 |
| `q4_k_m` | 378 | 52/60（86.7%） | 24/27（88.9%） | 76/87 | 96.3% | −5 / +1 |
| `q4_k_m_imat` | 378 | 53/60（88.3%） | 24/27（88.9%） | 77/87 | 95.0% | −6 / +3 |

门槛：`q4_k_m_imat` 比 `f16` 多错 3 条（10 vs 7，上限 2）→ **未通过**。
变差：holdout/holdout-020, main/main-005, main/main-033, main/main-036, main/main-048, main/main-060；变好：holdout/holdout-027, main/main-008, main/main-012。

## 速度（CPU）

| 变体 | 加载 s | 峰值 RSS MB | 冷启动首请求 TTFT ms | TTFT P50 / P95 ms | 无缓存 TTFT P50 ms | E2E P50 / P95 ms | decode tok/s | 平均 prompt 计算 / 复用 tok | 平均输出 tok |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `f16` | 6.8 | 1686 | 934 | 160 / 373 | 934 | 901 / 1872 | 34.1 | 79 / 731 | 31 |
| `q8_0` | 4.1 | 1152 | 1430 | 190 / 538 | 1362 | 687 / 1167 | 57.2 | 79 / 731 | 30 |
| `q4_k_m` | 3.1 | 1128 | 730 | 134 / 296 | 745 | 465 / 798 | 78.8 | 79 / 731 | 30 |
| `q4_k_m_imat` | 3.1 | 1128 | 789 | 146 / 353 | 812 | 486 / 824 | 79.4 | 79 / 731 | 30 |

冷启动首请求即预热请求（共享 system prompt 尚未进入 KV 缓存）；无缓存 TTFT 为 `cache_prompt: false` 的 10 条请求。
速度阶段的流式输出与质量阶段抽取器输出逐字比较：全部一致（temperature 0 下可复现）。

L2 逐条报告：`eval/reports/quant_f16_06b_slot_main.md`、`eval/reports/quant_f16_06b_slot_holdout.md`、`eval/reports/quant_q8_0_06b_slot_main.md`、`eval/reports/quant_q8_0_06b_slot_holdout.md`、`eval/reports/quant_q4_k_m_06b_slot_main.md`、`eval/reports/quant_q4_k_m_06b_slot_holdout.md`、`eval/reports/quant_q4_k_m_imat_06b_slot_main.md`、`eval/reports/quant_q4_k_m_imat_06b_slot_holdout.md`。
