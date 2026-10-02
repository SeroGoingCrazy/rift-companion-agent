# 模型卡：rift-slot-0.6b

「峡谷陪练」陪玩预约 Agent 的本地槽位抽取器。输入是线上 `extract_slots` 节点构造的 prompt（system 提示词 +
当前状态 / 候选 / 历史 / 用户这一句的 JSON），输出一个 `SlotExtraction` JSON：`turn_intent`、`delta`（字段级
"未出现 / null / any"三态）、`confirmation`。它只做抽取，不做决策、不回复用户。

| 项 | 值 |
|---|---|
| 发布文件 | `rift-slot-0.6b-q8_0.gguf`（Q8_0，639 446 816 字节） |
| sha256 | `8b197e753a62fb9a4853c70f2928db6e12d9a69f490c537daf480eae229573e4` |
| 基座 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)（Apache-2.0） |
| 训练 | LoRA SFT（r003）→ 规则扰动 DPO（β = 0.3）→ merge → GGUF F16 → Q8_0 |
| 推理 | llama.cpp `llama-server`，**必须关闭 thinking**，temperature 0，max_tokens 160，`response_format: json_object` |
| 版本日期 | 2026-10-02 |

## 训练

**数据**（全部只来自 Teacher 生成 + 校验，L2 评测集在训练前冻结，生成阶段不读取评测集，并按 bge 余弦 ≥ 0.92 去重）：

| 阶段 | 数据 | 规模 | 说明 |
|---|---|---:|---|
| SFT | `training/data/processed/sft/v0.4` | 1170 条（train 1110 / val 60） | v0.2 基础集 + r002 / r003 对比集；Teacher：OpenAI `gpt-5`（`reasoning_effort: low`，strict JSON Schema） |
| DPO | `training/data/processed/dpo/rule_v1` | 1084 对 | 只取自 SFT train；rejected 为单处、协议合法、评分器判错的扰动（any/null/缺失、漏键、照抄候选原文、列表替换写成追加、时间吞时长等） |

prompt（`config/prompts/slot_extract.txt`）sha256 `cf9ad121a31205283b60d92f29716f9fdc26a0f73f30deb3c0f3a49ea6e39c2d`；
prompt 改动后本模型需要重新评估（数据卡中的 `prompt_mismatch` 用于检测）。

**超参数**：

- SFT：LoRA r16 / α32 / dropout 0.05 / 全部线性层，lr 1e-4，3 epoch，cosine，warmup 0.1，有效 batch 16，cutoff 2048，
  `mask_history`，bf16，模板 `qwen3` + `enable_thinking: false`（`training/configs/training/llamafactory/sft_0.6b_r003.yaml`）。
- DPO：在 merged SFT 上做 LoRA r16 / α32，sigmoid，β 0.3，lr 5e-6，1 epoch（68 步），有效 batch 16
  （`dpo_0.6b_rule_b03.yaml`）。
- 硬件：单张 RTX 5080 16GB（WSL2），LLaMA-Factory 0.9.5，SFT 9.5 分钟，DPO 8.5 分钟。

## 评测

L2 是人工编写、人工审核的槽位抽取评测集：主集 60 条、holdout 27 条（checksum 记录于 `eval/datasets/CHECKSUMS`）。
通过标准为协议通过且任务分 ≥ 0.95。所有 L2 数字都在同一台 CPU（Ryzen 5 9600X，`-t 6`）上测得。

| 模型 | L2 主集 | L2 holdout | 协议通过率（主集） | E2E P50 / P95（CPU） |
|---|---:|---:|---:|---:|
| DeepSeek（远程大模型基线） | 98.3% | 88.9% | 100% | – |
| Qwen3-0.6B 基座 | 0% | 0% | – | – |
| SFT r003（F16） | 90.0% | 92.6% | 100% | – |
| SFT + DPO（F16） | 93.3% | 88.9% | 98.3% | 901 / 1872 ms |
| **SFT + DPO（Q8_0，本发布）** | **93.3%** | **88.9%** | **98.3%** | **687 / 1167 ms** |
| SFT + DPO（Q4_K_M，未发布） | 86.7% | 88.9% | 98.3% | 465 / 798 ms |

Q8_0 的 87 条输出与 F16 逐字相同。Q4_K_M（含 imatrix 版）比 F16 多错 3–4 条，没有达到"≤ 2 条"的门槛，所以不发布
（分析见 `training/experiments/quant/report.md`）。预热共享前缀后 Q8_0 的 TTFT P50 为 190 ms，decode 57 tok/s。

L4 端到端剧本（22 个，`--config local`，F16）：完成率 20/22，每个会话 1.41 次远程 LLM 调用（DeepSeek 基线 4.14）。
Q8_0 在 L2 上与 F16 逐字相同，但 L4 没有单独用 Q8_0 重跑。

## 已知局限

- **段位别名**："挑战者"无法识别，"宗师以上"会抽成 `challenger`（L4 两个失败剧本，SFT 阶段就有）。
- **长句漏键**：一句话里槽位很多时偶尔漏掉 `service_type` / `role_preference`；量化到 4 bit 后更明显。
- **拒绝换人**：偶尔在拒绝时多输出一个陪玩师名字（delta 应为空）。
- **枚举越界**："男生女生都行"会输出 `"male or female"`（DPO 带来的退化，协议层判错）。线上应配合 J1 的
  grammar 约束解码和路由器的重试 / 降级。
- 只覆盖英雄联盟陪玩预约这个窄领域的中文口语；领域之外的输入应判为 `unrelated`，但没有对抗性测试。
- 评测集只有 87 条，1 条约等于 1.1–3.7 个百分点，小差异不应过度解读。

## 使用方式

见 `models/README.md`。要点：llama-server 用 `--jinja --chat-template-kwargs '{"enable_thinking":false}'` 启动
（训练时 assistant 前带空 think 块，开着 thinking 会让输出偏离训练分布），启动后先发一次请求预热 system prompt。

## 许可证与使用限制

- 权重派生自 Qwen3-0.6B，基座许可证为 Apache-2.0，再分发时须附带其许可证与声明。
- 训练数据由 OpenAI `gpt-5` 生成，使用与再分发模型须遵守 OpenAI 的使用条款（包括对使用其输出开发竞争性模型的限制）。
  本仓库目前没有单独的许可证文件；公开发布（例如上传 HuggingFace）前需要所有者确认许可证与条款。
- 仅用于本项目的槽位抽取。它不做价格、退款等业务决策，这些由领域规则代码负责。
